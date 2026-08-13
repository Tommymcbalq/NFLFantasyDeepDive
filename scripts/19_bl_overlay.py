"""§J — Black-Litterman views overlay (EDA_PLAN4.md).

Strictly downstream of the frozen statistical board. Nothing here feeds back into any
fit, LOSO score, or edge test; the overlay always reports the statistical column and the
posterior column side by side.

Construction
------------
pi  : market-implied value. The §6.1 isotonic ADP->PPG curve evaluated at each player's
      slot. This is the reverse-engineering step -- price in, implied expected value out.
Sigma : covariance of TRUE value around that curve.
      diagonal   = tier residual variance around the isotonic curve, MINUS the per-game
                   sampling component sigma_W^2 / G, because the residual spread of a
                   realised season contains both uncertainty about theta and the noise of
                   estimating it from ~17 games. BL needs the former only.
      off-diag   = teammate block. Estimated from the historical panel (same-team board
                   pairs, within-year demeaned residuals): r = +0.016, cluster-bootstrap
                   CI [-0.321, +0.320] over 71 pairs, vs a random-within-year null of
                   -0.041 (p = .69). Not distinguishable from zero, so set to zero per the
                   §J1 rule. NOTE this is a power limitation, not a refutation: the
                   share-constraint prediction of rho ~ -0.3 sits inside that CI. The
                   TEAMMATE_RHO knob below exists to run it as a declared sensitivity, and
                   is zero by default.
views : rows of P. Absolute (single 1) or relative (entries sum to 0). q is the stated
      magnitude, Omega the diagonal confidence. Omega is DECLARED, never fitted.
posterior : theta_bar = [(tau*Sigma)^-1 + P' Omega^-1 P]^-1 [(tau*Sigma)^-1 pi + P' Omega^-1 q]
      with an exact per-view decomposition of the shift.

Usage
-----
    python3 scripts/19_bl_overlay.py --selftest        # J4 machinery assertions only
    python3 scripts/19_bl_overlay.py --views results/views_2026.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Declared sensitivity knob, not an estimate. See module docstring / sectionJ_notes.md.
TEAMMATE_RHO = 0.0

# Confidence scale for Omega: view uncertainty as a multiple of the prior SD of the view
# portfolio, sqrt(p' tau Sigma p). Declared before any view is written.
CONFIDENCE = {"low": 2.0, "medium": 1.0, "high": 0.5, "certain": 1e-6}


# ------------------------------------------------------------------ prior
def market_curve():
    """Return a callable ADP -> implied PPG from the fitted isotonic knots (§6.1)."""
    k = pd.read_csv(RESULTS / "market_prior_iso_knots.csv").sort_values("log_adp")
    x, y = k["log_adp"].to_numpy(), k["m"].to_numpy()

    def f(adp):
        return np.interp(np.log(np.asarray(adp, dtype=float)), x, y)

    return f


def build_prior(board, tier_var, sigma_w, tau_scalar):
    """pi and Sigma for the players on `board`.

    board     : DataFrame with columns name, adp, tier
    tier_var  : DataFrame tier -> tau2_iso  (residual variance around the curve)
    sigma_w   : DataFrame name -> sigma_W, games  (per-game noise, §1); may be partial
    """
    f = market_curve()
    pi = f(board["adp"].to_numpy())

    tv = dict(zip(tier_var["tier"], tier_var["tau2_iso"]))
    resid_var = board["tier"].map(tv).to_numpy(dtype=float)

    sw = sigma_w.set_index("name") if sigma_w is not None else None
    sampling = np.zeros(len(board))
    for i, nm in enumerate(board["name"]):
        if sw is not None and nm in sw.index:
            g = float(sw.loc[nm, "games_per_season"])
            sampling[i] = float(sw.loc[nm, "sigma_W"]) ** 2 / max(g, 1.0)
        else:
            sampling[i] = np.nanmedian(
                (sw["sigma_W"] ** 2 / sw["games_per_season"]).to_numpy()) if sw is not None else 0.0

    # Uncertainty about TRUE value = residual spread minus the noise of measuring it.
    diag = np.maximum(resid_var - sampling, 0.25 * resid_var)  # floor: never below 25%
    Sigma = np.diag(diag)

    if TEAMMATE_RHO != 0.0 and "team" in board.columns:
        for a in range(len(board)):
            for b in range(a + 1, len(board)):
                if board["team"].iloc[a] == board["team"].iloc[b]:
                    c = TEAMMATE_RHO * np.sqrt(diag[a] * diag[b])
                    Sigma[a, b] = Sigma[b, a] = c

    return pi, Sigma, diag, sampling, resid_var


# ------------------------------------------------------------------ views
def build_views(views, names, pi, Sigma, tau):
    """P, q, Omega from a tidy views table.

    Columns: view_id, player, weight, q, confidence [, rationale, date]
    Absolute view = one row (weight 1). Relative view = rows summing to zero.
    """
    idx = {n: i for i, n in enumerate(names)}
    ids = list(dict.fromkeys(views["view_id"]))
    P = np.zeros((len(ids), len(names)))
    q = np.zeros(len(ids))
    omega = np.zeros(len(ids))

    for r, vid in enumerate(ids):
        g = views[views["view_id"] == vid]
        for _, row in g.iterrows():
            if row["player"] not in idx:
                raise KeyError(f"view {vid}: '{row['player']}' is not on the board")
            P[r, idx[row["player"]]] = float(row["weight"])
        qs = g["q"].unique()
        if len(qs) != 1:
            raise ValueError(f"view {vid}: q must be a single value per view, got {qs}")
        q[r] = float(qs[0])
        conf = g["confidence"].iloc[0]
        if conf not in CONFIDENCE:
            raise ValueError(f"view {vid}: confidence '{conf}' not in {list(CONFIDENCE)}")
        prior_sd = np.sqrt(P[r] @ (tau * Sigma) @ P[r])
        omega[r] = max((CONFIDENCE[conf] * prior_sd) ** 2, 1e-12)

    return P, q, np.diag(omega), ids


def posterior(pi, Sigma, P, q, Omega, tau):
    """Return theta_bar, its covariance, and the exact per-view shift decomposition."""
    if P is None or len(P) == 0:
        return pi.copy(), tau * Sigma, np.zeros((0, len(pi)))
    A = np.linalg.inv(tau * Sigma)
    Oi = np.linalg.inv(Omega)
    M = np.linalg.inv(A + P.T @ Oi @ P)
    theta = M @ (A @ pi + P.T @ Oi @ q)
    # theta - pi = M P' Omega^-1 (q - P pi): separable by view.
    contrib = (M @ P.T @ Oi) * (q - P @ pi)[None, :]   # (n_players, n_views)
    return theta, M, contrib.T


# ------------------------------------------------------------------ J4 self-tests
def selftest():
    rng = np.random.default_rng(0)
    n = 6
    names = [f"P{i}" for i in range(n)]
    pi = np.linspace(20, 10, n)
    d = np.linspace(9.0, 4.0, n)
    Sigma = np.diag(d)
    tau = 0.5

    def views_df(rows):
        return pd.DataFrame(rows, columns=["view_id", "player", "weight", "q", "confidence"])

    # 1. no views is a no-op
    th, _, _ = posterior(pi, Sigma, np.zeros((0, n)), np.zeros(0), np.zeros((0, 0)), tau)
    assert np.allclose(th, pi), "empty view set must leave the prior untouched"

    # 2. a view stating exactly the prior is a no-op at any confidence
    for conf in CONFIDENCE:
        v = views_df([["v", "P2", 1.0, pi[2], conf]])
        P, q, Om, _ = build_views(v, names, pi, Sigma, tau)
        th, _, c = posterior(pi, Sigma, P, q, Om, tau)
        assert np.allclose(th, pi, atol=1e-8), f"zero-information view moved the board ({conf})"
        assert np.allclose(c, 0, atol=1e-8)

    # 3. certainty pins the posterior to q
    v = views_df([["v", "P1", 1.0, 25.0, "certain"]])
    P, q, Om, _ = build_views(v, names, pi, Sigma, tau)
    th, _, _ = posterior(pi, Sigma, P, q, Om, tau)
    assert abs(th[1] - 25.0) < 1e-3, f"certain view did not pin: {th[1]}"

    # 4. monotone in confidence
    shifts = []
    for conf in ["low", "medium", "high"]:
        v = views_df([["v", "P1", 1.0, 25.0, conf]])
        P, q, Om, _ = build_views(v, names, pi, Sigma, tau)
        th, _, _ = posterior(pi, Sigma, P, q, Om, tau)
        shifts.append(th[1] - pi[1])
    assert shifts[0] < shifts[1] < shifts[2], f"shift not monotone in confidence: {shifts}"

    # 5. relative view moves the pair in opposite directions and is ~zero-sum under
    #    equal prior variance
    d2 = np.full(n, 6.0)
    S2 = np.diag(d2)
    v = views_df([["v", "P0", 1.0, 3.0, "high"], ["v", "P1", -1.0, 3.0, "high"]])
    P, q, Om, _ = build_views(v, names, pi, S2, tau)
    th, _, _ = posterior(pi, S2, P, q, Om, tau)
    s = th - pi
    assert s[0] > 0 > s[1], f"relative view did not oppose: {s[:2]}"
    assert abs(s[0] + s[1]) < 1e-8, f"equal-variance relative view not zero-sum: {s[0] + s[1]}"
    assert np.allclose(s[2:], 0, atol=1e-10), "diagonal Sigma leaked a view to non-viewed players"

    # 6. with a non-zero teammate block, a view DOES leak to the teammate, with the sign
    #    the covariance implies
    S3 = np.diag(d2).astype(float)
    S3[0, 1] = S3[1, 0] = -0.3 * 6.0
    v = views_df([["v", "P0", 1.0, 25.0, "high"]])
    P, q, Om, _ = build_views(v, names, pi, S3, tau)
    th, _, _ = posterior(pi, S3, P, q, Om, tau)
    assert th[0] > pi[0] and th[1] < pi[1], "negative off-diagonal did not transmit downward"

    # 7. decomposition is exact and additive
    v = views_df([["v1", "P0", 1.0, 24.0, "medium"],
                  ["v2", "P3", 1.0, 9.0, "low"],
                  ["v3", "P4", 1.0, 14.0, "high"], ["v3", "P5", -1.0, 14.0, "high"]])
    P, q, Om, ids = build_views(v, names, pi, Sigma, tau)
    th, _, contrib = posterior(pi, Sigma, P, q, Om, tau)
    assert np.allclose(contrib.sum(axis=0), th - pi), "per-view decomposition does not sum to the shift"

    # 8. symmetry / PSD sanity
    _, M, _ = posterior(pi, Sigma, P, q, Om, tau)
    assert np.allclose(M, M.T) and np.all(np.linalg.eigvalsh(M) > 0), "posterior cov not SPD"

    print("J4 self-tests: 8/8 passed")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--views", default=None)
    ap.add_argument("--board", default=str(RESULTS / "valuation_2026_v3.csv"))
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--out", default=str(RESULTS / "board_2026_with_views.csv"))
    a = ap.parse_args()

    if a.selftest:
        selftest()
        if not a.views:
            return

    board = pd.read_csv(a.board).rename(columns={"player": "name"})
    tier_var = pd.read_csv(RESULTS / "tier_variances.csv")
    cons = pd.read_csv(RESULTS / "consistency_table.csv")
    sw = pd.DataFrame({
        "name": cons["player"],
        "sigma_W": cons["sigma_W"],
        "games_per_season": cons["n_games"] / cons["n_seasons"],
    })

    pi, Sigma, diag, sampling, resid_var = build_prior(board, tier_var, sw, a.tau)
    names = board["name"].tolist()

    out = board.copy()
    out["pi_market_implied"] = pi
    out["sigma_true_sd"] = np.sqrt(diag)

    if a.views:
        views = pd.read_csv(a.views)
        P, q, Om, ids = build_views(views, names, pi, Sigma, a.tau)
        theta, M, contrib = posterior(pi, Sigma, P, q, Om, a.tau)
        out["theta_bar_posterior"] = theta
        out["view_shift"] = theta - pi
        out["post_SD_bl"] = np.sqrt(np.diag(M))
        for r, vid in enumerate(ids):
            out[f"from_{vid}"] = contrib[r]
        # tau sensitivity
        for t in (0.25, 1.0):
            P2, q2, Om2, _ = build_views(views, names, pi, Sigma, t)
            th2, _, _ = posterior(pi, Sigma, P2, q2, Om2, t)
            out[f"theta_bar_tau{t}"] = th2

    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({len(out)} players, tau={a.tau}, TEAMMATE_RHO={TEAMMATE_RHO})")
    show = ["name", "adp", "pi_market_implied", "sigma_true_sd"]
    show += [c for c in ("theta_bar_posterior", "view_shift") if c in out.columns]
    print(out[show].to_string(index=False))


if __name__ == "__main__":
    main()

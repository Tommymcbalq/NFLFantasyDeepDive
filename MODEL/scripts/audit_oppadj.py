r"""
Audit 7 (TOP PRIORITY): is the opponent adjustment cleaning the metrics or
polluting them?

The worry, stated precisely
---------------------------
The ridge estimates O_t and D_o jointly. Team t's offensive rating is corrected
by the estimated defensive ratings of the opponents it happened to face. Those
D's are themselves estimates from thin data -- in week 3 they are almost entirely
the carried-over prior, and even later they rest on a handful of games. If
Var(D_hat - D_true) is large relative to the true spread of D, the "correction"
subtracts mostly noise, and does so with a NEGATIVE covariance structure that can
amplify rather than cancel. Since A's rating uses B's, which used A's, the noise
also circulates.

Estimators compared (identical inputs, identical memory profile, identical
downstream model -- only the adjustment differs)
------------------------------------------------
  roll    plain precision x recency weighted mean of the team's OWN realised
          off_/def_ values, centred on the league mean. No ridge, no shrinkage,
          no opponent decomposition at all. This is the obvious baseline.
  v2raw   two-stage prior/update ridge with the opponent block DELETED, i.e.
          shrinkage but no opponent adjustment. Isolates shrinkage from
          adjustment.
  v2adj   the current pipeline.
  v1adj   the previous single-pool pipeline.
  blend   Q(alpha) = Q_raw + alpha * (Q_adj - Q_raw). alpha = 0 is no
          adjustment, 1 is the full adjustment, and intermediate values shrink
          the ADJUSTMENT ITSELF toward zero. If the adjustment is mostly noise
          the CV optimum sits well below 1.

Protocol (per the methodology audit, superseding earlier instructions)
----------------------------------------------------------------------
Rolling-origin CV: for s in 2017..2023, fit on all seasons 2014..s-1, predict s.
Pooled n ~ 1832. Blocked bootstrap BY SEASON-WEEK for every interval. The 2023
season is part of the CV pool, never a separate selection set. 2024-2025 is not
touched anywhere in this file.
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import log_loss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from audit_quality import (METRICS, KAPPA, RHO, H_WITHIN, WEEKS_PER_SEASON,
                           V1_GAMMA, V1_HALF_LIFE, V1_LOOKBACK, load_panel,
                           build, _design, _ridge, _split)
from audit_model_table import build as build_table

RES = os.path.join(HERE, "..", "results")
CV_SEASONS = list(range(2017, 2024))
RNG = np.random.default_rng(20260810)
NBOOT = 2000

out_lines = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out_lines.append(s)


# ---------------------------------------------------------------------------
# The unadjusted rolling-mean baseline
# ---------------------------------------------------------------------------
def build_roll(panel, first_season=2014):
    """Plain weighted rolling mean of a team's own realised values.

    O_t = weighted mean of the team's own off_metric minus the same weighted
    league mean; D_t = weighted mean of what the team ALLOWED minus the league
    mean. Weights are plays x exp(-ln2 * weeks_ago / H) x gamma^(seasons
    crossed), i.e. the v1 memory profile. No ridge, no opponent term, so this is
    exactly "what did this team do, recently, ignoring who it played".
    """
    teams = np.sort(panel.team.dropna().unique())
    rows = []
    for season in sorted(s for s in panel.season.unique() if s >= first_season):
        weeks = sorted(panel[(panel.season == season)
                             & (panel.season_type == "REG")].week.unique())
        for week in weeks:
            t_target = season * WEEKS_PER_SEASON + week
            hist = panel[(panel.t < t_target)
                         & (panel.season >= season - V1_LOOKBACK)]
            rec = {}
            for m, (wcol, _) in METRICS.items():
                for side, key in (("off", "O"), ("def", "D")):
                    ycol, wc = f"{side}_{m}", f"{side}_{wcol}"
                    h = hist[hist[ycol].notna() & hist[wc].notna() & (hist[wc] > 0)]
                    if h.empty:
                        rec[(key, m)] = pd.Series(0.0, index=teams)
                        continue
                    rw = np.exp(-np.log(2) * (t_target - h.t.to_numpy(float))
                                / V1_HALF_LIFE)
                    cross = np.maximum(season - h.season.to_numpy(), 0)
                    w = h[wc].to_numpy(float) * rw * (V1_GAMMA ** cross)
                    y = h[ycol].to_numpy(float)
                    lg = np.average(y, weights=w)
                    num = pd.Series(w * y).groupby(h.team.to_numpy()).sum()
                    den = pd.Series(w).groupby(h.team.to_numpy()).sum()
                    rec[(key, m)] = ((num / den) - lg).reindex(teams).fillna(0.0)
            for team in teams:
                r = {"season": season, "week": week, "team": team}
                for m in METRICS:
                    r[f"O_{m}"] = rec[("O", m)][team]
                    r[f"D_{m}"] = rec[("D", m)][team]
                rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CV machinery with season-week blocked bootstrap
# ---------------------------------------------------------------------------
def load_table(est):
    d = pd.read_csv(os.path.join(HERE, "..", "data", f"audit_model_table_{est}.csv"))
    return d[d.home_win != 0.5].reset_index(drop=True)


def cv_predict(d, cols, seasons=CV_SEASONS):
    keep = d.dropna(subset=cols).reset_index(drop=True) if cols else d
    ys, ps, blocks, weeks = [], [], [], []
    for s in seasons:
        tr, te = keep[keep.season < s], keep[keep.season == s]
        if tr.empty or te.empty:
            continue
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1.0)
        X = sm.add_constant((tr[cols] - mu) / sd, has_constant="add")
        f = sm.Logit(tr.home_win.astype(int), X).fit(disp=0, maxiter=200)
        Xt = sm.add_constant((te[cols] - mu) / sd, has_constant="add")
        ps.append(np.clip(f.predict(Xt[f.params.index]).to_numpy(), 1e-6, 1 - 1e-6))
        ys.append(te.home_win.astype(int).to_numpy())
        blocks.append((te.season * 100 + te.week).to_numpy())
        weeks.append(te.week.to_numpy())
    return (np.concatenate(ys), np.concatenate(ps),
            np.concatenate(blocks), np.concatenate(weeks))


def ll(y, p):
    return float(log_loss(y, p, labels=[0, 1]))


def nll(y, p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def blocked_boot(y, pa, pb, blocks, nboot=NBOOT):
    """Paired bootstrap on log-loss difference, resampling SEASON-WEEK blocks."""
    dif = nll(y, pa) - nll(y, pb)
    uniq = np.unique(blocks)
    groups = [np.nonzero(blocks == b)[0] for b in uniq]
    sums = np.array([dif[g].sum() for g in groups])
    ns = np.array([len(g) for g in groups])
    nb = len(uniq)
    idx = RNG.integers(0, nb, size=(nboot, nb))
    bs = sums[idx].sum(axis=1) / ns[idx].sum(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pv = 2 * min((bs >= 0).mean(), (bs <= 0).mean())
    return float(dif.mean()), float(lo), float(hi), float(min(pv, 1.0))


def fold_signs(y, pa, pb, seasons_arr):
    d = nll(y, pa) - nll(y, pb)
    return pd.Series(d).groupby(seasons_arr).mean()


# ---------------------------------------------------------------------------
def main():
    panel = load_panel()

    roll_path = os.path.join(HERE, "..", "data", "team_quality_roll.csv")
    if not os.path.exists(roll_path):
        P("building unadjusted rolling-mean baseline ...")
        build_roll(panel).to_csv(roll_path, index=False)
    tbl_path = os.path.join(HERE, "..", "data", "audit_model_table_roll.csv")
    if not os.path.exists(tbl_path):
        build_table("roll").to_csv(tbl_path, index=False)

    CORE = ["off_epa_noto_diff", "def_epa_noto_diff", "net_to_rate_diff"]
    P("=" * 78)
    P("OPPONENT ADJUSTMENT: DOES IT EARN ITS PLACE?")
    P(f"rolling-origin CV, fit seasons <s, predict s, s = {CV_SEASONS[0]}..{CV_SEASONS[-1]}")
    P(f"features (incumbent spec B, rebuilt): {CORE}")
    P("=" * 78)

    got = {}
    for est in ("roll", "v2raw", "v2adj", "v1adj"):
        d = load_table(est)
        y, p, b, w = cv_predict(d, CORE)
        got[est] = (y, p, b, w, d)
        P(f"  {est:6s} n={len(y)}  CV log loss {ll(y, p):.4f}   "
          f"acc {(( p>0.5)==(y==1)).mean():.4f}")

    P("\npairwise, blocked bootstrap by season-week (negative = first is better):")
    pairs = [("v2adj", "roll", "full adjustment vs NO adjustment at all"),
             ("v2adj", "v2raw", "opponent block vs shrinkage-only"),
             ("v2raw", "roll", "shrinkage alone vs plain rolling mean"),
             ("v2adj", "v1adj", "two-stage prior vs single pooled fit")]
    seasons_arr = None
    for a, b_, label in pairs:
        ya, pa, bl, _, da = got[a]
        yb, pb, _, _, _ = got[b_]
        assert len(ya) == len(yb) and (ya == yb).all()
        m, lo, hi, pv = blocked_boot(ya, pa, pb, bl)
        sa = (bl // 100)
        fs = fold_signs(ya, pa, pb, sa)
        nneg = int((fs < 0).sum())
        verdict = "REAL" if (hi < 0 and m <= -0.005 and nneg >= 6) else \
                  ("HARMFUL" if lo > 0 else "NOISE")
        P(f"  {label:42s} {m:+.4f} [{lo:+.4f},{hi:+.4f}] p={pv:.3f} "
          f"folds_neg={nneg}/7 -> {verdict}")

    # ---------------- (b) week buckets ------------------------------------
    P("\n" + "=" * 78)
    P("(b) DOES THE ADJUSTMENT'S VALUE DEPEND ON HOW MUCH DATA IT HAS?")
    P("hypothesis: opponent ratings are prior-driven and noisy early, so the")
    P("adjustment should hurt in weeks 1-4 and help later")
    P("=" * 78)
    ya, pa, bl, wk, _ = got["v2adj"]
    _, praw, _, _, _ = got["roll"]
    _, pshr, _, _, _ = got["v2raw"]
    P(f"{'bucket':10s} {'n':>5s} {'roll':>8s} {'v2raw':>8s} {'v2adj':>8s} "
      f"{'adj-roll':>10s} {'95% CI':>20s}")
    for name, lo_w, hi_w in [("wk 1-4", 1, 4), ("wk 5-9", 5, 9), ("wk 10-18", 10, 18)]:
        s = (wk >= lo_w) & (wk <= hi_w)
        m, l, h, pv = blocked_boot(ya[s], pa[s], praw[s], bl[s])
        P(f"{name:10s} {s.sum():5d} {ll(ya[s], praw[s]):8.4f} "
          f"{ll(ya[s], pshr[s]):8.4f} {ll(ya[s], pa[s]):8.4f} "
          f"{m:+10.4f}   [{l:+.4f},{h:+.4f}]")

    # ---------------- (c)/(d) blend + shrinkage grid -----------------------
    P("\n" + "=" * 78)
    P("(c/d) SHRINK THE ADJUSTMENT ITSELF: Q(alpha) = Q_raw + alpha*(Q_adj-Q_raw)")
    P("alpha=0 no adjustment, alpha=1 the current pipeline. If the adjustment")
    P("is mostly propagated noise the CV optimum lies well below 1.")
    P("=" * 78)
    d_adj, d_raw = load_table("v2adj"), load_table("v2raw")
    assert (d_adj.game_id.to_numpy() == d_raw.game_id.to_numpy()).all()
    rows = []
    base = None
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        d = d_adj.copy()
        for c in CORE:
            d[c] = d_raw[c] + alpha * (d_adj[c] - d_raw[c])
        y, p, b, w = cv_predict(d, CORE)
        if base is None:
            base = p
            row = {"alpha": alpha, "cv_logloss": ll(y, p)}
        else:
            m, lo, hi, pv = blocked_boot(y, p, base, b)
            row = {"alpha": alpha, "cv_logloss": ll(y, p), "d_vs_alpha0": m,
                   "lo": lo, "hi": hi, "boot_p": pv}
        rows.append(row)
    r = pd.DataFrame(rows)
    P(r.round(4).to_string(index=False))
    r.to_csv(os.path.join(RES, "audit_oppadj_alpha.csv"), index=False)

    with open(os.path.join(RES, "audit_oppadj.txt"), "w") as f:
        f.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()

"""§X — mu_star: the calibrated, age-aware data arm.

    mu_star_i = a_f + b_f * mu_hat_i + c_f * z_i,    z_i = log[ f_f(age_i) / f_f(age_i - 1) ]

(a_f, b_f, c_f) are OLS on the TRAINING rows of LOSO fold f, per position; f_f is §H's era-3
relative age curve REFITTED inside the same fold (held-out season deleted from §H's panel).

Pre-registration: EDA_PLAN10.md §X1-§X4; operational definitions results/sectionX_notes.md
PART 1, written before this script was run.

Everything about the panel, the folds, the loss, B and m_hat is §W1 tier A, reused not rebuilt:
this module imports scripts/64_sectionW1_projection.py for load(), design-free helpers dm()
and the harness reproduction targets.

Outputs: results/sectionX_loso.csv, sectionX_holdout.csv, sectionX_diagnostics.csv,
         results/sectionX_coefs.csv, results/mu_star_coefs_2026.json
"""
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
from scipy import stats

warnings.simplefilter("ignore")
ROOT = Path("/Users/thomasmcnamee/NFL")
spec = importlib.util.spec_from_file_location(
    "w1", ROOT / "scripts/64_sectionW1_projection.py")
w1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w1)

YEARS = list(range(2015, 2025))
HOLDOUT_FIT = list(range(2015, 2022))          # 2015-2021
HOLDOUT_EVAL = [2022, 2023, 2024]
ERAS = ["1999-2007", "2008-2016", "2017-2025"]
ERA3 = "2017-2025"
GRID = np.arange(21.0, 36.01, 0.05)


# ====================================================================== §H age curve
# Replicated from scripts/20_sectionH_aging.py (make_basis / demean / fe_ols /
# fit_era_curves).  Nothing in the ESTIMATOR is changed; only the SAMPLE is reduced to the
# training fold.  The reproduction check below asserts that on the full sample this code
# returns results/age_curve_era.csv to 1e-9.
def h_panel():
    p = pd.read_csv(ROOT / "data/derived/age_panel_long_repaired.csv")
    p["era"] = pd.cut(p.season, [1998, 2007, 2016, 2025], labels=ERAS)
    q = p[(p.games >= 8) & (p.touches >= 40)].copy()
    q["r"] = q.ppg / q.groupby(["position", "season"]).ppg.transform("mean")
    return q.sort_values(["gsis_id", "season"]).reset_index(drop=True)


def make_basis(ages_pool):
    kn = np.quantile(ages_pool, [.2, .4, .6, .8])
    lo, hi = float(ages_pool.min()), float(ages_pool.max())

    def f(a):
        a = np.clip(np.asarray(a, float), lo, hi)
        return np.asarray(patsy.dmatrix(
            "cr(a, knots=k, lower_bound=lo, upper_bound=hi) - 1",
            {"a": a, "k": kn, "lo": lo, "hi": hi}))
    return dict(knots=kn, lo=lo, hi=hi, fn=f)


def _demean(X, g):
    df = pd.DataFrame(X)
    return (df - df.groupby(g).transform("mean")).to_numpy()


def fit_era_curves(dat, basis, outcome="r"):
    """player-FE fit of separate per-era spline blocks; returns anchored curves on GRID."""
    Bm = basis["fn"](dat.age.values)
    X = np.hstack([Bm * (dat.era.values == e)[:, None] for e in ERAS])
    y = dat[outcome].values
    yd = _demean(y.reshape(-1, 1), dat.gsis_id.values).ravel()
    Xd = _demean(X, dat.gsis_id.values)
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    k = Bm.shape[1]
    Bg = basis["fn"](GRID)
    out = {}
    for j, e in enumerate(ERAS):
        sub = (dat.era.values == e)
        if sub.sum() == 0:
            continue
        b_j = beta[j * k:(j + 1) * k]
        # §H anchoring: shift so the curve's mean over that era's observations equals the
        # sample mean of r over the same observations.  f is identified only up to a constant
        # under player FE, and the ratio f(a)/f(a-1) is NOT invariant to that constant, so the
        # convention is part of the specification.  It is §H's, applied identically every fold.
        out[e] = Bg @ b_j + (y[sub].mean() - (Bm[sub] @ b_j).mean())
    return out


class AgeCurve:
    """era-3 curve fitted on a stated subset of §H's panel; callable as a log-ratio."""

    def __init__(self, q, bases, drop_seasons=()):
        self.f = {}
        for pos in ["WR", "RB"]:
            d = q[(q.position == pos) & (~q.season.isin(drop_seasons))]
            self.f[pos] = fit_era_curves(d, bases[pos])[ERA3]

    def value(self, pos, a):
        return np.interp(np.asarray(a, float), GRID, self.f[pos])

    def logratio(self, pos, a):
        a = np.asarray(a, float)
        return np.log(self.value(pos, a) / self.value(pos, a - 1.0))

    def ratio(self, pos, a):
        return np.exp(self.logratio(pos, a))


def published_curve():
    AC = pd.read_csv(ROOT / "results/age_curve_era.csv")
    AC = AC[(AC.outcome == "relative") & (AC.era == ERA3)]
    return {p: (g.age.values, g.fit.values) for p, g in AC.groupby("position")}


PUB = published_curve()


def pub_ratio(pos, age):
    x, y = PUB[pos]
    age = np.asarray(age, float)
    return np.interp(age, x, y) / np.interp(age - 1.0, x, y)


# ====================================================================== panel
def panel():
    F = w1.load("A")
    d = F[F.in_fit & (F.n_eff > 0) & F.year.isin(YEARS)].copy()
    d = d[["gsis_id", "year", "pos", "age", "ppg", "mu_hat", "B", "m_hat",
           "n_eff", "G_last"]].dropna(subset=["age", "mu_hat", "ppg"])
    return d.reset_index(drop=True)


# ====================================================================== the arms
def fit_arms(tr, ev, pos, curve_fold, curve_pub_ratio):
    """Return a dict of out-of-sample predictions for the held-out rows `ev`.

    Every arm's parameters come from `tr` only.  `curve_fold` is fitted on `tr`'s seasons.
    """
    y = tr.ppg.values
    mu_tr, mu_ev = tr.mu_hat.values, ev.mu_hat.values
    z_tr = curve_fold.logratio(pos, tr.age.values)
    z_ev = curve_fold.logratio(pos, ev.age.values)
    r_ev = np.exp(z_ev)
    r_ev_pub = curve_pub_ratio(pos, ev.age.values)

    # --- calibration, a + b*mu_hat  (§W1's mu_cal, same estimator: OLS on training rows)
    ab = np.polyfit(mu_tr, y, 1)                     # [b, a]
    cal_ev = ab[1] + ab[0] * mu_ev

    # --- the specification: a + b*mu_hat + c*z
    X = np.column_stack([np.ones(len(tr)), mu_tr, z_tr])
    abc = np.linalg.lstsq(X, y, rcond=None)[0]
    star_ev = abc[0] + abc[1] * mu_ev + abc[2] * z_ev

    # --- same with the PUBLISHED curve (sensitivity on fold-fitting the curve)
    z_tr_p = np.log(curve_pub_ratio(pos, tr.age.values))
    Xp = np.column_stack([np.ones(len(tr)), mu_tr, z_tr_p])
    abcp = np.linalg.lstsq(Xp, y, rcond=None)[0]
    star_pub_ev = abcp[0] + abcp[1] * mu_ev + abcp[2] * np.log(r_ev_pub)

    return dict(
        mu_hat=mu_ev,
        mu_cal=cal_ev,
        mu_age_pub=mu_ev * r_ev_pub,
        mu_age=mu_ev * r_ev,
        mu_cal_x_age_pub=cal_ev * r_ev_pub,
        mu_cal_x_age=cal_ev * r_ev,
        mu_star=star_ev,
        mu_star_pub=star_pub_ev,
    ), dict(a=abc[0], b=abc[1], c=abc[2], cal_a=ab[1], cal_b=ab[0])


ARMS = ["mu_cal", "mu_age_pub", "mu_age", "mu_cal_x_age_pub", "mu_cal_x_age",
        "mu_star", "mu_star_pub"]


def loso(P, q, bases):
    rows, coefs = [], []
    for pos in ["WR", "RB"]:
        d = P[P.pos == pos]
        for Y in YEARS:
            tr, ev = d[d.year != Y], d[d.year == Y]
            if not len(ev):
                continue
            cf = AgeCurve(q, bases, drop_seasons=(Y,))
            pr, co = fit_arms(tr, ev, pos, cf, pub_ratio)
            out = ev[["gsis_id", "year", "pos", "age", "ppg", "mu_hat", "B", "m_hat",
                      "n_eff"]].copy()
            for k, v in pr.items():
                out[k] = v
            out["z"] = cf.logratio(pos, ev.age.values)
            rows.append(out)
            coefs.append(dict(pos=pos, fold=Y, **co,
                              f_peak=float(GRID[np.argmax(cf.f[pos])])))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(coefs)


def dm_abs(y, pb, pc, year):
    """DM on ABSOLUTE-error loss (declared robustness, PPG is right-skewed)."""
    d = np.abs(y - pb) - np.abs(y - pc)
    dy = pd.Series(d).groupby(pd.Series(year).values).mean()
    t = float(dy.mean() / (dy.std(ddof=1) / np.sqrt(len(dy))))
    return float(dy.mean()), float(2 * stats.t.sf(abs(t), df=len(dy) - 1))


def score(P):
    rows = []
    for pos in ["WR", "RB"]:
        d = P[P.pos == pos]
        y, base = d.ppg.values, d.mu_hat.values
        rmse = lambda v: float(np.sqrt(((y - v) ** 2).mean()))                # noqa: E731
        th = lambda v: (1 - d.B.values) * v + d.B.values * d.m_hat.values     # noqa: E731
        for a in ARMS:
            v = d[a].values
            r = dict(pos=pos, arm=a, n=len(d), rmse_mu=rmse(base), rmse_arm=rmse(v),
                     d_rmse=rmse(v) - rmse(base))
            r.update(w1.dm(y, base, v, d.year.values))
            for ref in ["mu_cal", "mu_age"]:
                g = w1.dm(y, d[ref].values, v, d.year.values)
                r[f"gain_vs_{ref}"], r[f"p_vs_{ref}"] = g["mean_gain"], g["dm_p"]
            e = w1.dm(y, th(base), th(v), d.year.values)
            r["rmse_theta_mu"] = rmse(th(base))
            r["rmse_theta_arm"] = rmse(th(v))
            r["eq7_gain"], r["eq7_p"], r["eq7_mde"] = e["mean_gain"], e["dm_p"], e["mde80"]
            r["eq7_folds"] = e["folds_improved"]
            g_abs, p_abs = dm_abs(y, base, v, d.year.values)
            r["mae_gain"], r["mae_p"] = g_abs, p_abs
            r["spearman"] = float(d.groupby("year").apply(
                lambda g: stats.spearmanr(g[a], g.ppg).statistic).mean())
            rows.append(r)
    return pd.DataFrame(rows)


# ====================================================================== temporal holdout
def holdout(P0, q, bases):
    """GENUINE holdout: coefficients AND the age curve fitted on 2015-21 only."""
    cf = AgeCurve(q, bases, drop_seasons=tuple(range(2022, 2026)))
    rows = []
    for pos in ["WR", "RB"]:
        d = P0[P0.pos == pos]
        tr = d[d.year.isin(HOLDOUT_FIT)]
        ev = d[d.year.isin(HOLDOUT_EVAL)].copy()
        pr, co = fit_arms(tr, ev, pos, cf, pub_ratio)
        y = ev.ppg.values
        rmse = lambda v: float(np.sqrt(((y - v) ** 2).mean()))                 # noqa: E731
        th = lambda v: (1 - ev.B.values) * v + ev.B.values * ev.m_hat.values   # noqa: E731
        b = pr["mu_hat"]
        for a in ARMS:
            v = pr[a]
            r = dict(pos=pos, arm=a, n_fit=len(tr), n=len(ev), rmse_mu=rmse(b),
                     rmse_arm=rmse(v), d_rmse=rmse(v) - rmse(b), beats_mu=bool(rmse(v) < rmse(b)),
                     rmse_theta_mu=rmse(th(b)), rmse_theta_arm=rmse(th(v)),
                     beats_mu_eq7=bool(rmse(th(v)) < rmse(th(b))))
            dd = w1.dm(y, b, v, ev.year.values)          # 3 years only: descriptive
            r["mean_gain_3yr"], r["dm_p_3yr"] = dd["mean_gain"], dd["dm_p"]
            r["folds_improved"] = dd["folds_improved"]
            e = w1.dm(y, th(b), th(v), ev.year.values)
            r["eq7_gain_3yr"], r["eq7_p_3yr"] = e["mean_gain"], e["dm_p"]
            for Y in HOLDOUT_EVAL:
                m = (ev.year == Y).values
                r[f"rmse_{Y}_mu"] = float(np.sqrt(((y[m] - b[m]) ** 2).mean()))
                r[f"rmse_{Y}_arm"] = float(np.sqrt(((y[m] - v[m]) ** 2).mean()))
            rows.append(r)
        rows[-1]["coefs"] = json.dumps({k: round(float(v), 4) for k, v in co.items()})
    return pd.DataFrame(rows)


# ====================================================================== diagnostics
def diagnostics(P):
    rows = []
    for pos in ["WR", "RB"]:
        d = P[P.pos == pos].copy()
        y = d.ppg.values
        d["dev"] = (d.mu_hat - d.mu_hat.mean()).abs()
        d["l_mu"] = (y - d.mu_hat) ** 2
        for cut, lab in [(pd.qcut(d.dev, 4, labels=False), "|mu_hat - mean| quartile"),
                         (pd.qcut(d.age, 3, labels=False), "age tercile")]:
            for g, sub in d.groupby(cut):
                r = dict(pos=pos, split=lab, bin=int(g), n=len(sub),
                         mean_age=float(sub.age.mean()), mean_mu=float(sub.mu_hat.mean()),
                         mean_ppg=float(sub.ppg.mean()))
                for a in ["mu_cal", "mu_age", "mu_star"]:
                    r[f"gain_{a}"] = float((((sub.ppg - sub.mu_hat) ** 2)
                                            - ((sub.ppg - sub[a]) ** 2)).mean())
                    r[f"mean_delta_{a}"] = float((sub[a] - sub.mu_hat).mean())
                rows.append(r)
        # residual structure of the fitted affine form
        res = d.ppg - d.mu_star
        for nm, x in [("mu_hat", d.mu_hat), ("age", d.age), ("z", d.z)]:
            b = np.polyfit(x, res, 1)
            b2 = np.polyfit(x, res, 2)
            rows.append(dict(pos=pos, split=f"residual(mu_star) ~ {nm}", bin=-1, n=len(d),
                             mean_age=float(np.corrcoef(x, res)[0, 1]),
                             mean_mu=float(b[0]), mean_ppg=float(b2[0])))
    return pd.DataFrame(rows)


# ====================================================================== 2026 coefficients
def coefs_2026(P0, q, bases):
    """No held-out year exists for 2026, so the same estimator is run with the fold
    restriction removed: (a,b,c) on all of 2015-2024, curve on all of §H's panel."""
    cf = AgeCurve(q, bases)
    out = {"curve_era": ERA3, "fitted_on": "2015-2024 (all folds)", "grid": list(GRID)}
    for pos in ["WR", "RB"]:
        d = P0[P0.pos == pos]
        z = cf.logratio(pos, d.age.values)
        X = np.column_stack([np.ones(len(d)), d.mu_hat.values, z])
        abc = np.linalg.lstsq(X, d.ppg.values, rcond=None)[0]
        out[pos] = dict(a=float(abc[0]), b=float(abc[1]), c=float(abc[2]),
                        n=int(len(d)), curve=list(map(float, cf.f[pos])))
    return out


# ====================================================================== main
if __name__ == "__main__":
    pd.set_option("display.width", 300)
    q = h_panel()
    bases = {p: make_basis(q[q.position == p].age.values) for p in ["WR", "RB"]}

    # ---- reproduce §H's published era-3 curve from this code, full sample
    full = AgeCurve(q, bases)
    err = {}
    for pos in ["WR", "RB"]:
        x, yv = PUB[pos]
        err[pos] = float(np.abs(np.interp(GRID, x, yv) - full.f[pos]).max())
    print("=== §H curve reproduction (this code, full sample, vs age_curve_era.csv) ===")
    print({k: f"{v:.2e}" for k, v in err.items()})
    assert max(err.values()) < 1e-8, "the §H curve re-implementation does not reproduce §H"

    P0 = panel()
    print(f"panel: WR n={(P0.pos=='WR').sum()}  RB n={(P0.pos=='RB').sum()}")

    P, CO = loso(P0, q, bases)
    S = score(P)
    S.to_csv(ROOT / "results/sectionX_loso.csv", index=False)
    P.to_csv(ROOT / "results/sectionX_predictions.csv", index=False)
    CO.to_csv(ROOT / "results/sectionX_coefs.csv", index=False)

    # ---- harness validation against §W1's published component numbers
    print("\n=== HARNESS VALIDATION vs §W1 (results/sectionW1_age.csv) ===")
    tgt = {("WR", "mu_hat_rmse"): 3.7760, ("RB", "mu_hat_rmse"): 4.4909,
           ("WR", "mu_cal"): 3.6059, ("RB", "mu_cal"): 4.1708,
           ("WR", "mu_age_pub"): 3.6186, ("RB", "mu_age_pub"): 4.3415,
           ("WR", "mu_cal_x_age_pub"): 3.5466, ("RB", "mu_cal_x_age_pub"): 4.0774}
    ok = True
    for pos in ["WR", "RB"]:
        s = S[S.pos == pos]
        got = float(s.rmse_mu.iat[0])
        print(f"  {pos} mu_hat            {got:.4f}  target {tgt[(pos,'mu_hat_rmse')]:.4f}")
        ok &= abs(got - tgt[(pos, "mu_hat_rmse")]) < 5e-4
        for a in ["mu_cal", "mu_age_pub", "mu_cal_x_age_pub"]:
            got = float(s[s.arm == a].rmse_arm.iat[0])
            print(f"  {pos} {a:17s} {got:.4f}  target {tgt[(pos,a)]:.4f}")
            ok &= abs(got - tgt[(pos, a)]) < 5e-4
    print("  REPRODUCED" if ok else "  *** DOES NOT REPRODUCE §W1 ***")

    cols = ["pos", "arm", "n", "rmse_mu", "rmse_arm", "mean_gain", "dm_p", "mde80",
            "obs_over_mde", "folds_improved", "mae_gain", "mae_p", "spearman",
            "gain_vs_mu_cal", "p_vs_mu_cal", "gain_vs_mu_age", "p_vs_mu_age",
            "rmse_theta_mu", "rmse_theta_arm", "eq7_gain", "eq7_p", "eq7_mde", "eq7_folds"]
    print("\n=== §X LOSO 2015-2024 ===")
    print(S[cols].round(4).to_string(index=False))

    # ---- additivity check (§X4)
    print("\n=== §X4 overlap: is the combination short of the sum of its parts? ===")
    for pos in ["WR", "RB"]:
        s = S[S.pos == pos].set_index("arm")
        for combo, parts in [("mu_star", ("mu_cal", "mu_age")),
                             ("mu_cal_x_age", ("mu_cal", "mu_age"))]:
            tot = s.mean_gain[parts[0]] + s.mean_gain[parts[1]]
            print(f"  {pos} {combo:14s} gain {s.mean_gain[combo]:+.4f} vs "
                  f"{parts[0]} {s.mean_gain[parts[0]]:+.4f} + {parts[1]} "
                  f"{s.mean_gain[parts[1]]:+.4f} = {tot:+.4f}   "
                  f"overlap {tot - s.mean_gain[combo]:+.4f} "
                  f"({100*(1 - s.mean_gain[combo]/tot):.0f}% of the sum lost)")

    H = holdout(P0, q, bases)
    H.to_csv(ROOT / "results/sectionX_holdout.csv", index=False)
    print("\n=== §X TEMPORAL HOLDOUT: fit 2015-21 (coefs AND curve), evaluate 2022-24 ===")
    print(H[["pos", "arm", "n_fit", "n", "rmse_mu", "rmse_arm", "d_rmse", "beats_mu",
             "mean_gain_3yr", "dm_p_3yr", "folds_improved", "rmse_theta_mu",
             "rmse_theta_arm", "beats_mu_eq7", "eq7_gain_3yr",
             "rmse_2022_arm", "rmse_2023_arm", "rmse_2024_arm"]].round(4).to_string(index=False))

    D = diagnostics(P)
    D.to_csv(ROOT / "results/sectionX_diagnostics.csv", index=False)
    print("\n=== §X diagnostics: where the gain comes from ===")
    print(D.round(4).to_string(index=False))

    print("\n=== fold coefficients (a, b, c) ===")
    print(CO.round(4).to_string(index=False))
    print(CO.groupby("pos")[["a", "b", "c", "cal_a", "cal_b"]].agg(["mean", "std"]).round(4))

    C26 = coefs_2026(P0, q, bases)
    with open(ROOT / "results/mu_star_coefs_2026.json", "w") as fh:
        json.dump(C26, fh)
    print("\n=== 2026 coefficients (fitted on all 2015-2024) ===")
    for pos in ["WR", "RB"]:
        c = C26[pos]
        print(f"  {pos}: a={c['a']:.4f}  b={c['b']:.4f}  c={c['c']:.4f}  n={c['n']}")

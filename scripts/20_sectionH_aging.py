"""§H — the aging curve for WR/RB and whether it has shifted later in calendar time.

Pre-registered in EDA_PLAN4.md §H (2026-08-09). Executed as written; nothing tuned.

Panel      : data/derived/age_panel_long.csv, WR/RB player-seasons 1999-2025, age on a
             fixed Sept-1 reference.
Qualify    : >= 8 games AND >= 40 touches (carries + targets). Same for the adjacent
             season whenever a transition is used (H4).
Outcome    : r_is = PPG_is / mean(PPG among qualified players at that position in season s).
             Removes league-wide scoring/period effects by construction; required because
             with player FE age and period are exactly collinear within player (APC).
             Absolute-PPG versions are run as a labelled, confounded sensitivity.
Eras       : 1999-2007 / 2008-2016 / 2017-2025 (calendar thirds, fixed before fitting).
Spline     : natural cubic (patsy cr) basis, interior knots at the age quintiles
             (20/40/60/80 pct) of the pooled qualified panel FOR THAT POSITION, boundary
             knots at that panel's min/max age. Basis built ONCE per position and reused
             for every era / bootstrap replicate so curves are comparable.
Estimator  : player fixed effects absorbed by within-player demeaning (FWL); OLS on the
             demeaned system; all CIs and tests by cluster bootstrap on player (B reps,
             players resampled with replacement, each draw treated as a distinct unit).
Level ident: with player FE, f is identified only up to an additive constant. Convention,
             fixed before fitting and applied identically to every era/replicate: each
             era's curve is shifted so that its sample mean over that era's observations
             equals the sample mean of r over the same observations (= 1 by construction
             of r, up to weighting). "Cliff" = first age above the peak at which the
             anchored curve falls 10% below its anchored peak value.

Outputs: results/sectionH_notes.md (written by hand from this script's stdout),
         results/age_curve_era.csv, results/exit_hazard.csv, results/figures/*.png,
         results/sectionH_h5.csv
"""
import json
import re
import sys
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.isotonic import IsotonicRegression

warnings.simplefilter("ignore")
ROOT = "/Users/thomasmcnamee/NFL"
RNG = np.random.default_rng(20260809)
B_BOOT = 600
ERAS = [("1999-2007", 1999, 2007), ("2008-2016", 2008, 2016), ("2017-2025", 2017, 2025)]
OUT = {}


def log(*a):
    print(*a)
    sys.stdout.flush()


# ---------------------------------------------------------------- panel + qualification
import os
PANEL = os.environ.get("H_PANEL", "repaired")
SUF = "" if PANEL == "repaired" else "_rawpanel"
_pf = "age_panel_long_repaired.csv" if PANEL == "repaired" else "age_panel_long.csv"
log(f"PANEL = {PANEL}  ({_pf})")
panel = pd.read_csv(f"{ROOT}/data/derived/{_pf}")
panel["era"] = pd.cut(panel.season, [1998, 2007, 2016, 2025],
                      labels=[e[0] for e in ERAS])
panel["qual"] = (panel.games >= 8) & (panel.touches >= 40)

log("=" * 78)
log("§H0  qualified-N by era (reported BEFORE any modelling)")
tab = (panel.groupby(["position", "era"], observed=True)
       .agg(player_seasons=("qual", "size"), qualified=("qual", "sum"),
            ).reset_index())
tab["qual_rate"] = tab.qualified / tab.player_seasons
q = panel[panel.qual].copy()
extra = (q.groupby(["position", "era"], observed=True)
         .agg(players=("gsis_id", "nunique"), mean_ppg=("ppg", "mean"),
              mean_age=("age", "mean"),
              age_p1=("age", lambda s: s.quantile(.01)),
              age_p99=("age", lambda s: s.quantile(.99)),
              mean_games=("games", "mean")).reset_index())
tab = tab.merge(extra, on=["position", "era"])
log(tab.to_string(index=False))

# season-level normaliser
season_mean = q.groupby(["position", "season"]).ppg.transform("mean")
q["r"] = q.ppg / season_mean
q = q.sort_values(["gsis_id", "season"]).reset_index(drop=True)
log("\nqualified players per season (position x season count) summary:")
cnt = q.groupby(["position", "season"]).size().unstack(0)
log(cnt.describe().T.to_string())


# ---------------------------------------------------------------- machinery
def make_basis(ages_pool):
    """natural cubic spline basis, interior knots at pooled age quintiles."""
    kn = np.quantile(ages_pool, [.2, .4, .6, .8])
    lo, hi = float(ages_pool.min()), float(ages_pool.max())
    di = {"knots": kn, "lo": lo, "hi": hi}

    def f(a):
        a = np.clip(np.asarray(a, float), lo, hi)
        return np.asarray(patsy.dmatrix(
            "cr(a, knots=k, lower_bound=lo, upper_bound=hi) - 1",
            {"a": a, "k": kn, "lo": lo, "hi": hi}))
    di["fn"] = f
    return di


def demean(X, g):
    """within-group demeaning (FWL absorption of group FE)."""
    df = pd.DataFrame(X)
    return (df - df.groupby(g).transform("mean")).to_numpy()


def fe_ols(y, X, g):
    yd = demean(y.reshape(-1, 1), g).ravel()
    Xd = demean(X, g)
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    resid = yd - Xd @ beta
    return beta, Xd, resid


def cluster_boot(fn, ids, B=B_BOOT, rng=RNG):
    """fn(idx_array, unit_labels) -> 1-d vector of statistics (may contain nan)."""
    uni = np.unique(ids)
    pos = {u: np.flatnonzero(ids == u) for u in uni}
    out = []
    for b in range(B):
        draw = rng.choice(uni, size=len(uni), replace=True)
        idx = np.concatenate([pos[u] for u in draw])
        lab = np.concatenate([np.full(len(pos[u]), j) for j, u in enumerate(draw)])
        try:
            out.append(fn(idx, lab))
        except Exception:
            out.append(None)
    out = [o for o in out if o is not None]
    return np.vstack(out)


def ci(v, lo=2.5, hi=97.5):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return (np.nan, np.nan)
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi)))


def curve_features(grid, vals, anchor_mean, lo_supp, hi_supp):
    """peak age, peak value, cliff age (first age > peak with val < 0.9*peak),
    slope 28->32 -- all on the anchored curve, within [lo_supp, hi_supp]."""
    m = (grid >= lo_supp) & (grid <= hi_supp)
    g, v = grid[m], vals[m]
    if len(g) < 5:
        return dict(peak=np.nan, peak_val=np.nan, cliff=np.nan, slope=np.nan)
    i = int(np.argmax(v))
    peak, pv = float(g[i]), float(v[i])
    thr = pv * 0.90          # "cliff" = 10% below the anchored peak
    after = np.flatnonzero((g > peak) & (v < thr))
    cliff = float(g[after[0]]) if len(after) else np.nan
    def at(x):
        return float(np.interp(x, g, v, left=np.nan, right=np.nan))
    s = (at(32) - at(28)) / 4.0
    return dict(peak=peak, peak_val=pv, cliff=cliff, slope=s)


GRID = np.arange(21.0, 36.01, 0.05)


def fit_era_curves(dat, basis, outcome="r", eras=None, return_beta=False):
    """player-FE fit of separate per-era spline blocks; returns anchored curves."""
    eras = eras or [e[0] for e in ERAS]
    Bm = basis["fn"](dat.age.values)
    blocks = [Bm * (dat.era.values == e)[:, None] for e in eras]
    X = np.hstack(blocks)
    y = dat[outcome].values
    beta, Xd, resid = fe_ols(y, X, dat["_unit"].values)
    k = Bm.shape[1]
    Bg = basis["fn"](GRID)
    curves = {}
    for j, e in enumerate(eras):
        raw_grid = Bg @ beta[j * k:(j + 1) * k]
        sub = dat.era.values == e
        if sub.sum() == 0:
            continue
        raw_obs = Bm[sub] @ beta[j * k:(j + 1) * k]
        c = dat[outcome].values[sub].mean() - raw_obs.mean()
        curves[e] = raw_grid + c
    if return_beta:
        return curves, beta, resid
    return curves


def orth_basis(Bm):
    """centred, rank-reduced basis (constant direction removed) for interaction tests."""
    Bc = Bm - Bm.mean(0)
    u, s, vt = np.linalg.svd(Bc, full_matrices=False)
    r = int((s > s[0] * 1e-10).sum())
    return u[:, :r] * s[:r], vt[:r]      # scores; rotation to reapply on a grid


# ---------------------------------------------------------------- H1 / H2
rows_curve, feat_rows, h2_tests = [], [], []
for pos in ["WR", "RB"]:
    dat = q[q.position == pos].copy()
    dat["_unit"] = dat.gsis_id.values
    basis = make_basis(dat.age.values)
    log("\n" + "=" * 78)
    log(f"§H1/H2  {pos}   n={len(dat)}  players={dat.gsis_id.nunique()}  "
        f"knots={np.round(basis['knots'],2)}  bounds=({basis['lo']:.1f},{basis['hi']:.1f})")

    supp = {e: (dat.loc[dat.era == e, "age"].quantile(.01),
                dat.loc[dat.era == e, "age"].quantile(.99)) for e in [x[0] for x in ERAS]}
    supp_all = (dat.age.quantile(.01), dat.age.quantile(.99))

    for outcome, label in [("r", "relative"), ("ppg", "absolute_confounded")]:
        # ---- H1 pooled ----
        one = dat.copy()
        one["era"] = "ALL"
        cur = fit_era_curves(one, basis, outcome, eras=["ALL"])
        f1 = curve_features(GRID, cur["ALL"], dat[outcome].mean(), *supp_all)

        def stat_h1(idx, lab):
            d = dat.iloc[idx].copy()
            d["_unit"] = lab
            d["era"] = "ALL"
            c = fit_era_curves(d, basis, outcome, eras=["ALL"])["ALL"]
            ff = curve_features(GRID, c, d[outcome].mean(), *supp_all)
            return np.array([ff["peak"], ff["cliff"], ff["slope"]] + list(c))

        bs = cluster_boot(stat_h1, dat.gsis_id.values)
        log(f"\n[{pos}/{label}] H1 pooled: peak {f1['peak']:.2f} "
            f"CI{np.round(ci(bs[:,0]),2)}  cliff {f1['cliff']:.2f} "
            f"CI{np.round(ci(bs[:,1]),2)}  slope28-32 {f1['slope']:+.4f} "
            f"CI{np.round(ci(bs[:,2]),4)}")
        feat_rows.append(dict(position=pos, outcome=label, era="POOLED",
                              n=len(dat), players=dat.gsis_id.nunique(),
                              peak=f1["peak"], peak_lo=ci(bs[:, 0])[0], peak_hi=ci(bs[:, 0])[1],
                              cliff=f1["cliff"], cliff_lo=ci(bs[:, 1])[0], cliff_hi=ci(bs[:, 1])[1],
                              slope28_32=f1["slope"], slope_lo=ci(bs[:, 2])[0],
                              slope_hi=ci(bs[:, 2])[1]))
        cb = bs[:, 3:]
        for gi, a in enumerate(GRID):
            rows_curve.append(dict(position=pos, outcome=label, era="POOLED", age=a,
                                   fit=cur["ALL"][gi],
                                   lo=np.nanpercentile(cb[:, gi], 2.5),
                                   hi=np.nanpercentile(cb[:, gi], 97.5)))

        # ---- H2 per-era ----
        cur_e = fit_era_curves(dat, basis, outcome)
        fe_ = {e: curve_features(GRID, cur_e[e], dat.loc[dat.era == e, outcome].mean(),
                                 *supp[e]) for e in cur_e}

        def stat_h2(idx, lab):
            d = dat.iloc[idx].copy()
            d["_unit"] = lab
            c = fit_era_curves(d, basis, outcome)
            v = []
            for e in [x[0] for x in ERAS]:
                if e not in c:
                    v += [np.nan] * 3 + [np.nan] * len(GRID)
                    continue
                ff = curve_features(GRID, c[e], d.loc[d.era == e, outcome].mean(), *supp[e])
                v += [ff["peak"], ff["cliff"], ff["slope"]] + list(c[e])
            return np.array(v)

        bs2 = cluster_boot(stat_h2, dat.gsis_id.values)
        blk = 3 + len(GRID)
        for j, (e, _, _) in enumerate(ERAS):
            o = j * blk
            ff = fe_[e]
            pk, cl, sl = ci(bs2[:, o]), ci(bs2[:, o + 1]), ci(bs2[:, o + 2])
            log(f"[{pos}/{label}] {e}: peak {ff['peak']:.2f} CI{np.round(pk,2)}  "
                f"cliff {ff['cliff']:.2f} CI{np.round(cl,2)}  "
                f"slope28-32 {ff['slope']:+.4f} CI{np.round(sl,4)}  "
                f"(support {supp[e][0]:.1f}-{supp[e][1]:.1f}, "
                f"cliff-undefined reps {np.isnan(bs2[:,o+1]).mean():.0%})")
            feat_rows.append(dict(position=pos, outcome=label, era=e,
                                  n=int((dat.era == e).sum()),
                                  players=dat.loc[dat.era == e, "gsis_id"].nunique(),
                                  peak=ff["peak"], peak_lo=pk[0], peak_hi=pk[1],
                                  cliff=ff["cliff"], cliff_lo=cl[0], cliff_hi=cl[1],
                                  slope28_32=ff["slope"], slope_lo=sl[0], slope_hi=sl[1]))
            for gi, a in enumerate(GRID):
                col = bs2[:, o + 3 + gi]
                rows_curve.append(dict(position=pos, outcome=label, era=e, age=a,
                                       fit=cur_e[e][gi],
                                       lo=np.nanpercentile(col, 2.5),
                                       hi=np.nanpercentile(col, 97.5)))
        # pairwise era differences in peak / cliff / slope (bootstrap CIs)
        for (j1, j2) in [(0, 1), (1, 2), (0, 2)]:
            o1, o2 = j1 * blk, j2 * blk
            for m, nm in [(0, "peak"), (1, "cliff"), (2, "slope28_32")]:
                d = bs2[:, o2 + m] - bs2[:, o1 + m]
                pt = fe_[ERAS[j2][0]][{"peak": "peak", "cliff": "cliff",
                                       "slope28_32": "slope"}[nm]] - \
                    fe_[ERAS[j1][0]][{"peak": "peak", "cliff": "cliff",
                                      "slope28_32": "slope"}[nm]]
                lo, hi = ci(d)
                pv = 2 * min((d < 0).mean(), (d > 0).mean()) if np.isfinite(d).sum() > 20 else np.nan
                log(f"    Δ{nm} {ERAS[j2][0]} − {ERAS[j1][0]}: {pt:+.3f} "
                    f"CI({lo:+.3f},{hi:+.3f}) boot-p {pv:.3f}")
                h2_tests.append(dict(position=pos, outcome=label, comparison=nm,
                                     eras=f"{ERAS[j2][0]}-{ERAS[j1][0]}", delta=pt,
                                     lo=lo, hi=hi, boot_p=pv))

    # ---- H2 formal interaction test (relative outcome), bootstrap-covariance Wald ----
    Bm = basis["fn"](dat.age.values)
    S, rot = orth_basis(Bm)
    D = np.column_stack([(dat.era.values == e).astype(float) for e in [x[0] for x in ERAS]])

    def build(dsub, Ssub, Dsub):
        inter = np.hstack([Ssub * Dsub[:, [j]] for j in (1, 2)])
        return np.hstack([Ssub, Dsub[:, 1:3], inter]), Ssub.shape[1]

    X0, k = build(dat, S, D)
    beta0, _, _ = fe_ols(dat.r.values, X0, dat.gsis_id.values)
    nint = 2 * k

    def stat_w(idx, lab):
        d = dat.iloc[idx].copy()
        Xb, _ = build(d, S[idx], D[idx])
        b, _, _ = fe_ols(d.r.values, Xb, lab)
        return b[-nint:]

    bw = cluster_boot(stat_w, dat.gsis_id.values)
    V = np.cov(bw.T)
    dvec = beta0[-nint:]
    W = float(dvec @ np.linalg.pinv(V) @ dvec)
    from scipy import stats as st
    pW = 1 - st.chi2.cdf(W, nint)
    log(f"\n[{pos}] H2 era×age interaction (bootstrap-cov Wald, df={nint}): "
        f"W={W:.2f}, p={pW:.4f}")
    h2_tests.append(dict(position=pos, outcome="relative", comparison="era_x_age_Wald",
                         eras="all", delta=W, lo=nint, hi=np.nan, boot_p=pW))

    # ---- smooth robustness: age spline × centered season ----
    sc = (dat.season.values - 2012) / 10.0
    Xs = np.hstack([S, sc[:, None], S * sc[:, None]])
    bs_, _, _ = fe_ols(dat.r.values, Xs, dat.gsis_id.values)

    def stat_s(idx, lab):
        d = dat.iloc[idx]
        scb = (d.season.values - 2012) / 10.0
        Xb = np.hstack([S[idx], scb[:, None], S[idx] * scb[:, None]])
        b, _, _ = fe_ols(d.r.values, Xb, lab)
        # peak/cliff at season 2003, 2012, 2021
        out = list(b[-k:])
        for yr in (2003, 2012, 2021):
            z = (yr - 2012) / 10.0
            Sg = (basis["fn"](GRID) - Bm.mean(0)) @ rot.T
            v = Sg @ b[:k] + Sg @ b[-k:] * z
            v = v - v.mean() + 1.0
            ff = curve_features(GRID, v, 1.0, *supp_all)
            out += [ff["peak"], ff["cliff"]]
        return np.array(out)

    bsm = cluster_boot(stat_s, dat.gsis_id.values)
    Vs = np.cov(bsm[:, :k].T)
    Ws = float(bs_[-k:] @ np.linalg.pinv(Vs) @ bs_[-k:])
    pWs = 1 - st.chi2.cdf(Ws, k)
    log(f"[{pos}] smooth age×centred-season interaction: W={Ws:.2f} df={k} p={pWs:.4f}")
    Sg = (basis["fn"](GRID) - Bm.mean(0)) @ rot.T
    for i_, yr in enumerate((2003, 2012, 2021)):
        z = (yr - 2012) / 10.0
        v = Sg @ bs_[:k] + Sg @ bs_[-k:] * z
        v = v - v.mean() + 1.0
        ff = curve_features(GRID, v, 1.0, *supp_all)
        c1, c2 = k + 2 * i_, k + 2 * i_ + 1
        log(f"    season {yr}: peak {ff['peak']:.2f} CI{np.round(ci(bsm[:,c1]),2)}  "
            f"cliff {ff['cliff']:.2f} CI{np.round(ci(bsm[:,c2]),2)}")
        h2_tests.append(dict(position=pos, outcome="relative_smooth",
                             comparison=f"peak_cliff_{yr}", eras=str(yr),
                             delta=ff["peak"], lo=ff["cliff"], hi=np.nan, boot_p=pWs))

    OUT[f"basis_{pos}"] = basis
    OUT[f"supp_{pos}"] = supp
    OUT[f"supp_all_{pos}"] = supp_all

pd.DataFrame(rows_curve).to_csv(f"{ROOT}/results/age_curve_era{SUF}.csv", index=False)
pd.DataFrame(feat_rows).to_csv(f"{ROOT}/results/age_curve_features{SUF}.csv", index=False)
pd.DataFrame(h2_tests).to_csv(f"{ROOT}/results/age_era_tests{SUF}.csv", index=False)
log("\nwrote results/age_curve_era.csv, age_curve_features.csv, age_era_tests.csv")

# ---------------------------------------------------------------- H3a balanced cohort
log("\n" + "=" * 78)
log("§H3a  balanced cohort: players with >= 6 qualified seasons")
bal_rows = []
for pos in ["WR", "RB"]:
    dat = q[q.position == pos].copy()
    keep = dat.groupby("gsis_id").size()
    keep = keep[keep >= 6].index
    d = dat[dat.gsis_id.isin(keep)].copy()
    d["_unit"] = d.gsis_id.values
    basis = OUT[f"basis_{pos}"]
    supp = {e: (d.loc[d.era == e, "age"].quantile(.01),
                d.loc[d.era == e, "age"].quantile(.99)) for e in [x[0] for x in ERAS]}
    log(f"{pos}: players {len(keep)} of {dat.gsis_id.nunique()}, rows {len(d)} of {len(dat)}; "
        f"per-era rows {[int((d.era==e).sum()) for e,_,_ in ERAS]}")
    cur = fit_era_curves(d, basis, "r")

    def stat(idx, lab, d=d, basis=basis, supp=supp):
        dd = d.iloc[idx].copy()
        dd["_unit"] = lab
        c = fit_era_curves(dd, basis, "r")
        v = []
        for e, _, _ in ERAS:
            if e not in c or (dd.era == e).sum() < 20:
                v += [np.nan] * 3
                continue
            ff = curve_features(GRID, c[e], 1.0, *supp[e])
            v += [ff["peak"], ff["cliff"], ff["slope"]]
        return np.array(v)

    bs = cluster_boot(stat, d.gsis_id.values, B=400)
    for j, (e, _, _) in enumerate(ERAS):
        ff = curve_features(GRID, cur[e], 1.0, *supp[e])
        log(f"  {e}: peak {ff['peak']:.2f} CI{np.round(ci(bs[:,3*j]),2)}  "
            f"cliff {ff['cliff']:.2f} CI{np.round(ci(bs[:,3*j+1]),2)}  "
            f"slope {ff['slope']:+.4f} CI{np.round(ci(bs[:,3*j+2]),4)}")
        bal_rows.append(dict(position=pos, era=e, n=int((d.era == e).sum()),
                             peak=ff["peak"], peak_lo=ci(bs[:, 3*j])[0],
                             peak_hi=ci(bs[:, 3*j])[1], cliff=ff["cliff"],
                             cliff_lo=ci(bs[:, 3*j+1])[0], cliff_hi=ci(bs[:, 3*j+1])[1],
                             slope28_32=ff["slope"], slope_lo=ci(bs[:, 3*j+2])[0],
                             slope_hi=ci(bs[:, 3*j+2])[1]))
pd.DataFrame(bal_rows).to_csv(f"{ROOT}/results/age_curve_balanced{SUF}.csv", index=False)

# ---------------------------------------------------------------- H3b exit hazard
log("\n" + "=" * 78)
log("§H3b  discrete-time career-exit hazard: P(season s is the last qualified season | "
    "qualified at s), logit on age spline × era, cluster-robust by player")
haz_rows = []
from scipy import stats as st
for pos in ["WR", "RB"]:
    dat = q[q.position == pos].copy()
    last = dat.groupby("gsis_id").season.transform("max")
    dat["exit"] = (dat.season == last).astype(int)
    d = dat[dat.season <= 2024].copy()      # 2025 right-censored
    basis = OUT[f"basis_{pos}"]
    Bm = basis["fn"](d.age.values)
    S, rot = orth_basis(Bm)
    D = np.column_stack([(d.era.values == e).astype(float) for e, _, _ in ERAS])
    X = np.hstack([np.ones((len(d), 1)), S, D[:, 1:3],
                   S * D[:, [1]], S * D[:, [2]]])
    m = sm.Logit(d.exit.values, X).fit(disp=0, cov_type="cluster",
                                       cov_kwds={"groups": d.gsis_id.values})
    k = S.shape[1]
    nint = 2 * k
    R = np.zeros((nint, X.shape[1]))
    R[np.arange(nint), np.arange(X.shape[1] - nint, X.shape[1])] = 1
    wt = m.wald_test(R, scalar=True)
    log(f"\n{pos}: n={len(d)} player-seasons, exits={int(d.exit.sum())} "
        f"({d.exit.mean():.1%}); era×age Wald χ²({nint})={float(wt.statistic):.2f}, "
        f"p={float(wt.pvalue):.4f}")
    # per-era hazard curve + age at which hazard crosses .25/.40
    Sg = (basis["fn"](GRID) - Bm.mean(0)) @ rot.T
    def haz_at(beta, j):
        Xg = np.hstack([np.ones((len(GRID), 1)), Sg,
                        np.tile([[1.0 if j == 1 else 0, 1.0 if j == 2 else 0]], (len(GRID), 1)),
                        Sg * (1.0 if j == 1 else 0.0), Sg * (1.0 if j == 2 else 0.0)])
        return 1 / (1 + np.exp(-(Xg @ beta)))

    def cross(gr, v, t):
        i = np.flatnonzero(v >= t)
        return float(gr[i[0]]) if len(i) else np.nan

    def stat(idx, lab, d=d, X=X):
        dd = d.iloc[idx]
        try:
            mm = sm.Logit(dd.exit.values, X[idx]).fit(disp=0, method="bfgs", maxiter=200)
        except Exception:
            return None
        v = []
        for j in range(3):
            h = haz_at(mm.params, j)
            supp = OUT[f"supp_{pos}"][ERAS[j][0]]
            msk = (GRID >= supp[0]) & (GRID <= supp[1])
            v += [cross(GRID[msk], h[msk], .25), cross(GRID[msk], h[msk], .40),
                  float(np.interp(30, GRID, h)), float(np.interp(32, GRID, h))]
        return np.array(v)

    bs = cluster_boot(stat, d.gsis_id.values, B=300)
    for j, (e, _, _) in enumerate(ERAS):
        h = haz_at(m.params, j)
        supp = OUT[f"supp_{pos}"][e]
        msk = (GRID >= supp[0]) & (GRID <= supp[1])
        a25, a40 = cross(GRID[msk], h[msk], .25), cross(GRID[msk], h[msk], .40)
        h30, h32 = np.interp(30, GRID, h), np.interp(32, GRID, h)
        log(f"  {e}: age at hazard .25 = {a25:.2f} CI{np.round(ci(bs[:,4*j]),2)}   "
            f".40 = {a40:.2f} CI{np.round(ci(bs[:,4*j+1]),2)}   "
            f"h(30)={h30:.3f} CI{np.round(ci(bs[:,4*j+2]),3)}  "
            f"h(32)={h32:.3f} CI{np.round(ci(bs[:,4*j+3]),3)}")
        for gi, a in enumerate(GRID):
            haz_rows.append(dict(position=pos, era=e, age=a, hazard=h[gi],
                                 in_support=bool(msk[gi])))
    for (j1, j2) in [(0, 2), (0, 1), (1, 2)]:
        for m_, nm in [(0, "age_at_h25"), (1, "age_at_h40"), (2, "h30"), (3, "h32")]:
            dd = bs[:, 4*j2+m_] - bs[:, 4*j1+m_]
            lo, hi = ci(dd)
            pv = 2 * min((dd < 0).mean(), (dd > 0).mean())
            log(f"    Δ{nm} {ERAS[j2][0]}−{ERAS[j1][0]}: CI({lo:+.3f},{hi:+.3f}) "
                f"boot-p {pv:.3f}")
pd.DataFrame(haz_rows).to_csv(f"{ROOT}/results/exit_hazard{SUF}.csv", index=False)
log("wrote results/exit_hazard.csv")

# ---------------------------------------------------------------- H4 RB workload
log("\n" + "=" * 78)
log("§H4  RB workload carryover: Δr on prior-season touches | age spline + player FE")
rb = q[q.position == "RB"].copy().sort_values(["gsis_id", "season"])
rb["prev_season"] = rb.groupby("gsis_id").season.shift(1)
rb["prev_touch"] = rb.groupby("gsis_id").touches.shift(1)
rb["prev_r"] = rb.groupby("gsis_id").r.shift(1)
tr = rb[(rb.season - rb.prev_season == 1) & rb.prev_r.notna()].copy()
tr["dr"] = tr.r - tr.prev_r
tr["t1"] = tr.prev_touch / 100.0
tr["t2"] = tr.t1 ** 2
tr["heavy"] = (tr.prev_touch >= 350).astype(float)
basis = OUT["basis_RB"]
Bm = basis["fn"](tr.age.values)
S, _ = orth_basis(Bm)
log(f"transitions n={len(tr)}, players={tr.gsis_id.nunique()}, "
    f"heavy(>=350) n={int(tr.heavy.sum())}, mean prev touches {tr.prev_touch.mean():.0f}")
X = np.hstack([S, tr[["t1", "t2", "heavy"]].to_numpy()])
beta, _, _ = fe_ols(tr.dr.values, X, tr.gsis_id.values)


def stat(idx, lab):
    d = tr.iloc[idx]
    Xb = np.hstack([S[idx], d[["t1", "t2", "heavy"]].to_numpy()])
    b, _, _ = fe_ols(d.dr.values, Xb, lab)
    return b[-3:]


bs = cluster_boot(stat, tr.gsis_id.values)
h4 = []
for i, nm in enumerate(["touches/100 (linear)", "touches/100 squared", ">=350 indicator"]):
    lo, hi = ci(bs[:, i])
    pv = 2 * min((bs[:, i] < 0).mean(), (bs[:, i] > 0).mean())
    log(f"  {nm:24s} β={beta[-3+i]:+.4f}  CI({lo:+.4f},{hi:+.4f})  boot-p {pv:.4f}")
    h4.append(dict(term=nm, beta=beta[-3+i], lo=lo, hi=hi, boot_p=pv))
# marginal effect of going 200 -> 350 touches
mar = (beta[-3] * (3.5 - 2.0) + beta[-2] * (3.5**2 - 2.0**2) + beta[-1])
marb = bs[:, 0] * 1.5 + bs[:, 1] * (3.5**2 - 4.0) + bs[:, 2]
log(f"  implied Δr from a 200->350 prior-touch season: {mar:+.4f} "
    f"CI{np.round(ci(marb),4)} (relative-PPG units)")
h4.append(dict(term="marginal 200->350", beta=mar, lo=ci(marb)[0], hi=ci(marb)[1],
               boot_p=2 * min((marb < 0).mean(), (marb > 0).mean())))
pd.DataFrame(h4).to_csv(f"{ROOT}/results/h4_workload{SUF}.csv", index=False)

json.dump({"done": "H1-H4"}, open(f"{ROOT}/results/.h_stage1.json", "w"))
log("\nstage 1 complete")

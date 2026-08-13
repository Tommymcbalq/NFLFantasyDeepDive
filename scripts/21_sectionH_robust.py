"""§H robustness / anomaly chasing (runs after scripts/20_sectionH_aging.py).

Three things the stage-1 output forces:

(1) The pre-registered "smooth age-spline x centred-season" check is WEAKLY IDENTIFIED
    under player FE. Within a player, cohort = season - age is constant, so adding any
    phi(season - age) to the surface is absorbed by alpha_i. Inside the bilinear family
    f(a) + g(a)*z the only excluded direction is the one that would need a z^2 term, so
    f and g are separated by functional form alone -> near-collinear design, |beta| ~ 30,
    and the reconstructed per-season curve is garbage. Diagnosed here (condition number),
    and re-run under an alternative identification (pooled / no player FE, cluster-robust
    by player) where age and season are not collinear. Labelled as such.

(2) WR qualification rate is 22% in 1999-2007 vs 43% in the two later eras (40-touch bar
    in a low-volume passing era). The era-1 WR curve is therefore estimated on a much more
    selected sample, and the exit hazard's "failed to requalify" event is a harder bar in
    era 1. Post-hoc (prompted by the pre-modelling §H0 table, which the plan requires to be
    read first) constant-selection variant: qualify the top-K players per position-season
    by touches, K = the minimum per-season qualified count in the panel, so the selection
    RATE is constant by construction. Reported whichever way it comes out; the pre-
    registered >=8g/>=40-touch fit remains the headline.

(3) H4 as pre-registered regresses Delta r on prior-season touches. prior touches are
    strongly correlated with prior r, and prior r enters the outcome with coefficient -1,
    so the specification is mechanically contaminated by mean reversion. Reported as
    specified, then decomposed: (a) add prior r as a control, (b) level specification
    (r_s on prior touches | age spline + FE), (c) placebo -- same regression with NEXT
    season's touches, which cannot cause this season's decline.
"""
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats as st

ROOT = "/Users/thomasmcnamee/NFL"
RNG = np.random.default_rng(20260810)
ERAS = [("1999-2007", 1999, 2007), ("2008-2016", 2008, 2016), ("2017-2025", 2017, 2025)]
GRID = np.arange(21.0, 36.01, 0.05)


def log(*a):
    print(*a, flush=True)


def demean(X, g):
    df = pd.DataFrame(X)
    return (df - df.groupby(g).transform("mean")).to_numpy()


def fe_ols(y, X, g):
    yd = demean(y.reshape(-1, 1), g).ravel()
    Xd = demean(X, g)
    b, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    return b, Xd


def make_basis(pool):
    kn = np.quantile(pool, [.2, .4, .6, .8])
    lo, hi = float(pool.min()), float(pool.max())

    def f(a):
        a = np.clip(np.asarray(a, float), lo, hi)
        return np.asarray(patsy.dmatrix(
            "cr(a, knots=k, lower_bound=lo, upper_bound=hi) - 1",
            {"a": a, "k": kn, "lo": lo, "hi": hi}))
    return dict(fn=f, knots=kn, lo=lo, hi=hi)


def orth(Bm):
    Bc = Bm - Bm.mean(0)
    u, s, vt = np.linalg.svd(Bc, full_matrices=False)
    r = int((s > s[0] * 1e-10).sum())
    return u[:, :r] * s[:r], vt[:r]


def feats(grid, v, lo, hi):
    m = (grid >= lo) & (grid <= hi)
    g, v = grid[m], v[m]
    i = int(np.argmax(v))
    pk, pv = float(g[i]), float(v[i])
    a = np.flatnonzero((g > pk) & (v < pv * .90))
    return pk, (float(g[a[0]]) if len(a) else np.nan), \
        (np.interp(32, g, v) - np.interp(28, g, v)) / 4


def cb(fn, ids, B=400):
    uni = np.unique(ids)
    pos = {u: np.flatnonzero(ids == u) for u in uni}
    out = []
    for _ in range(B):
        dr = RNG.choice(uni, size=len(uni), replace=True)
        idx = np.concatenate([pos[u] for u in dr])
        lab = np.concatenate([np.full(len(pos[u]), j) for j, u in enumerate(dr)])
        try:
            out.append(fn(idx, lab))
        except Exception:
            pass
    return np.vstack(out)


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (np.nan, np.nan) if len(v) < 20 else (float(np.percentile(v, 2.5)),
                                                 float(np.percentile(v, 97.5)))


import os
PANEL = os.environ.get("H_PANEL", "repaired")
SUF = "" if PANEL == "repaired" else "_rawpanel"
_pf = "age_panel_long_repaired.csv" if PANEL == "repaired" else "age_panel_long.csv"
log(f"PANEL = {PANEL}  ({_pf})")
panel = pd.read_csv(f"{ROOT}/data/derived/{_pf}")
panel["era"] = pd.cut(panel.season, [1998, 2007, 2016, 2025],
                      labels=[e[0] for e in ERAS])
panel["qual"] = (panel.games >= 8) & (panel.touches >= 40)

# ================================================================= (1) identification
log("=" * 78)
log("(1) APC identification diagnostic for the smooth age x centred-season check")
q = panel[panel.qual].copy()
q["r"] = q.ppg / q.groupby(["position", "season"]).ppg.transform("mean")
smooth_rows = []
for pos in ["WR", "RB"]:
    d = q[q.position == pos].copy()
    bas = make_basis(d.age.values)
    Bm = bas["fn"](d.age.values)
    S, rot = orth(Bm)
    k = S.shape[1]
    z = (d.season.values - 2012) / 10.0
    X = np.hstack([S, z[:, None], S * z[:, None]])
    Xd = demean(X, d.gsis_id.values)
    sv = np.linalg.svd(Xd, compute_uv=False)
    b, _ = fe_ols(d.r.values, X, d.gsis_id.values)
    log(f"  {pos} player-FE smooth design: cond(Xd) = {sv[0]/sv[-1]:.1f}, "
        f"max|beta| = {np.abs(b).max():.1f}   -> f/g split not credibly identified")
    # alternative identification: pooled OLS, no player FE, cluster-robust by player
    Xp = sm.add_constant(np.hstack([S, z[:, None], S * z[:, None]]))
    m = sm.OLS(d.r.values, Xp).fit(cov_type="cluster",
                                   cov_kwds={"groups": d.gsis_id.values})
    R = np.zeros((k, Xp.shape[1]))
    R[np.arange(k), np.arange(Xp.shape[1] - k, Xp.shape[1])] = 1
    w = m.wald_test(R, scalar=True)
    log(f"  {pos} POOLED (no FE, cluster by player) age x season interaction: "
        f"F={float(w.statistic):.2f}, p={float(w.pvalue):.4f}")
    Sg = (bas["fn"](GRID) - Bm.mean(0)) @ rot.T
    lo_s, hi_s = d.age.quantile(.01), d.age.quantile(.99)

    def stat(idx, lab, d=d, S=S, Sg=Sg, k=k, lo_s=lo_s, hi_s=hi_s):
        dd = d.iloc[idx]
        zz = (dd.season.values - 2012) / 10.0
        Xp2 = sm.add_constant(np.hstack([S[idx], zz[:, None], S[idx] * zz[:, None]]))
        bb = np.linalg.lstsq(Xp2, dd.r.values, rcond=None)[0]
        out = []
        for yr in (2003, 2012, 2021):
            zv = (yr - 2012) / 10.0
            v = bb[0] + Sg @ bb[1:1+k] + (Sg @ bb[-k:]) * zv + bb[1+k] * zv
            out += list(feats(GRID, v, lo_s, hi_s))
        return np.array(out)

    bs = cb(stat, d.gsis_id.values)
    for i, yr in enumerate((2003, 2012, 2021)):
        zv = (yr - 2012) / 10.0
        v = m.params[0] + Sg @ m.params[1:1+k] + (Sg @ m.params[-k:]) * zv + m.params[1+k] * zv
        pk, cl, sl = feats(GRID, v, lo_s, hi_s)
        log(f"    {pos} pooled curve at season {yr}: peak {pk:.2f} "
            f"CI{np.round(ci(bs[:,3*i]),2)}  cliff {cl:.2f} CI{np.round(ci(bs[:,3*i+1]),2)}"
            f"  slope28-32 {sl:+.4f} CI{np.round(ci(bs[:,3*i+2]),4)}")
        smooth_rows.append(dict(position=pos, season=yr, peak=pk,
                                peak_lo=ci(bs[:, 3*i])[0], peak_hi=ci(bs[:, 3*i])[1],
                                cliff=cl, cliff_lo=ci(bs[:, 3*i+1])[0],
                                cliff_hi=ci(bs[:, 3*i+1])[1], slope=sl,
                                slope_lo=ci(bs[:, 3*i+2])[0], slope_hi=ci(bs[:, 3*i+2])[1],
                                interaction_F=float(w.statistic), interaction_p=float(w.pvalue)))
    # drift per decade in peak/cliff
    for nm, o in [("peak", 0), ("cliff", 1)]:
        dd = (bs[:, 6 + o] - bs[:, o]) / 1.8
        log(f"    {pos} pooled {nm} drift per decade: CI{np.round(ci(dd),3)} "
            f"boot-p {2*min((dd<0).mean(),(dd>0).mean()):.3f}")
pd.DataFrame(smooth_rows).to_csv(f"{ROOT}/results/age_curve_smooth_pooled{SUF}.csv", index=False)

# ================================================ (2) constant-selection qualification
log("\n" + "=" * 78)
log("(2) constant-selection-rate qualification (post-hoc robustness)")
rows = []
for pos in ["WR", "RB"]:
    d0 = panel[panel.position == pos].copy()
    K = int(panel[panel.qual & (panel.position == pos)].groupby("season").size().min())
    log(f"  {pos}: K = {K} per season (min per-season qualified count under the "
        f"pre-registered rule)")
    d = (d0[d0.games >= 8].sort_values(["season", "touches"], ascending=[True, False])
         .groupby("season").head(K).copy())
    d["r"] = d.ppg / d.groupby("season").ppg.transform("mean")
    log(f"     rows {len(d)}; per-era {[int((d.era==e).sum()) for e,_,_ in ERAS]}; "
        f"touch floor by era "
        f"{[round(d.loc[d.era==e,'touches'].min()) for e,_,_ in ERAS]}; "
        f"median touches by era {[round(d.loc[d.era==e,'touches'].median()) for e,_,_ in ERAS]}")
    bas = make_basis(d.age.values)
    Bm = bas["fn"](d.age.values)
    supp = {e: (d.loc[d.era == e, "age"].quantile(.01),
                d.loc[d.era == e, "age"].quantile(.99)) for e, _, _ in ERAS}
    Bg = bas["fn"](GRID)

    def curves(dd, lab):
        Bl = bas["fn"](dd.age.values)
        X = np.hstack([Bl * (dd.era.values == e)[:, None] for e, _, _ in ERAS])
        b, _ = fe_ols(dd.r.values, X, lab)
        kk = Bl.shape[1]
        out = {}
        for j, (e, _, _) in enumerate(ERAS):
            sub = dd.era.values == e
            if sub.sum() < 20:
                continue
            raw = Bg @ b[j*kk:(j+1)*kk]
            out[e] = raw + (dd.r.values[sub].mean() - (Bl[sub] @ b[j*kk:(j+1)*kk]).mean())
        return out

    C = curves(d, d.gsis_id.values)

    def stat(idx, lab, d=d, supp=supp):
        dd = d.iloc[idx]
        c = curves(dd, lab)
        v = []
        for e, _, _ in ERAS:
            v += list(feats(GRID, c[e], *supp[e])) if e in c else [np.nan]*3
        return np.array(v)

    bs = cb(stat, d.gsis_id.values)
    for j, (e, _, _) in enumerate(ERAS):
        pk, cl, sl = feats(GRID, C[e], *supp[e])
        log(f"     {e}: peak {pk:.2f} CI{np.round(ci(bs[:,3*j]),2)}  cliff {cl:.2f} "
            f"CI{np.round(ci(bs[:,3*j+1]),2)}  slope {sl:+.4f} CI{np.round(ci(bs[:,3*j+2]),4)}")
        rows.append(dict(position=pos, era=e, n=int((d.era == e).sum()), peak=pk,
                         peak_lo=ci(bs[:, 3*j])[0], peak_hi=ci(bs[:, 3*j])[1], cliff=cl,
                         cliff_lo=ci(bs[:, 3*j+1])[0], cliff_hi=ci(bs[:, 3*j+1])[1],
                         slope28_32=sl, slope_lo=ci(bs[:, 3*j+2])[0],
                         slope_hi=ci(bs[:, 3*j+2])[1]))
    for nm, o in [("peak", 0), ("cliff", 1), ("slope", 2)]:
        dd = bs[:, 6+o] - bs[:, o]
        log(f"     Δ{nm} era3−era1: CI{np.round(ci(dd),3)} "
            f"boot-p {2*min((dd<0).mean(),(dd>0).mean()):.3f}")

    # exit hazard under the same constant-selection rule
    dd = d.copy()
    last = dd.groupby("gsis_id").season.transform("max")
    dd["exit"] = (dd.season == last).astype(int)
    dd = dd[dd.season <= 2024]
    Bl = bas["fn"](dd.age.values)
    S, rt = orth(Bl)
    D = np.column_stack([(dd.era.values == e).astype(float) for e, _, _ in ERAS])
    X = np.hstack([np.ones((len(dd), 1)), S, D[:, 1:3], S*D[:, [1]], S*D[:, [2]]])
    m = sm.Logit(dd.exit.values, X).fit(disp=0, cov_type="cluster",
                                        cov_kwds={"groups": dd.gsis_id.values})
    kk = S.shape[1]
    R = np.zeros((2*kk, X.shape[1]))
    R[np.arange(2*kk), np.arange(X.shape[1]-2*kk, X.shape[1])] = 1
    w = m.wald_test(R, scalar=True)
    Sg = (bas["fn"](GRID) - Bl.mean(0)) @ rt.T
    log(f"     exit hazard (constant-selection): exits {dd.exit.mean():.1%}, "
        f"era×age Wald χ²({2*kk})={float(w.statistic):.2f} p={float(w.pvalue):.4f}")
    for j, (e, _, _) in enumerate(ERAS):
        Xg = np.hstack([np.ones((len(GRID), 1)), Sg,
                        np.tile([[float(j == 1), float(j == 2)]], (len(GRID), 1)),
                        Sg*float(j == 1), Sg*float(j == 2)])
        h = 1/(1+np.exp(-(Xg @ m.params)))
        log(f"       {e}: h(28)={np.interp(28,GRID,h):.3f} h(30)={np.interp(30,GRID,h):.3f} "
            f"h(32)={np.interp(32,GRID,h):.3f}")
pd.DataFrame(rows).to_csv(f"{ROOT}/results/age_curve_constsel{SUF}.csv", index=False)

# ================================================================= (3) H4 decomposition
log("\n" + "=" * 78)
log("(3) H4 mean-reversion decomposition (RB)")
rb = q[q.position == "RB"].sort_values(["gsis_id", "season"]).copy()
for c in ["season", "touches", "r"]:
    rb["prev_" + c] = rb.groupby("gsis_id")[c].shift(1)
    rb["next_" + c] = rb.groupby("gsis_id")[c].shift(-1)
tr = rb[(rb.season - rb.prev_season == 1) & rb.prev_r.notna()].copy()
tr["dr"] = tr.r - tr.prev_r
tr["t1"] = tr.prev_touch = tr.prev_touches / 100
tr["t2"] = tr.t1**2
tr["heavy"] = (tr.prev_touches >= 350).astype(float)
tr["nt1"] = tr.next_touches / 100
bas = make_basis(rb.age.values)
S, _ = orth(bas["fn"](tr.age.values))
log(f"  corr(prev touches, prev r) = {np.corrcoef(tr.prev_touches, tr.prev_r)[0,1]:.3f}")

specs = {
    "as pre-registered: Δr ~ t + t² + heavy": (["t1", "t2", "heavy"], "dr", None),
    "+ control for prior r": (["t1", "t2", "heavy", "prev_r"], "dr", None),
    "level: r_s ~ t + t² + heavy": (["t1", "t2", "heavy"], "r", None),
    "level + prior r": (["t1", "t2", "heavy", "prev_r"], "r", None),
}
h4 = []
for nm, (cols, yv, _) in specs.items():
    X = np.hstack([S, tr[cols].to_numpy()])
    b, _ = fe_ols(tr[yv].values, X, tr.gsis_id.values)

    def stat(idx, lab, cols=cols, yv=yv):
        d = tr.iloc[idx]
        Xb = np.hstack([S[idx], d[cols].to_numpy()])
        bb, _ = fe_ols(d[yv].values, Xb, lab)
        nb = len(cols)
        mar = bb[-nb] * 1.5 + bb[-nb+1] * (3.5**2 - 2.0**2) + bb[-nb+2]
        return np.concatenate([bb[-nb:], [mar]])

    bs = cb(stat, tr.gsis_id.values)
    nb = len(cols)
    mar = b[-nb]*1.5 + b[-nb+1]*(3.5**2-4.0) + b[-nb+2]
    txt = "  ".join(f"{c}={b[-nb+i]:+.3f}{'*' if 2*min((bs[:,i]<0).mean(),(bs[:,i]>0).mean())<.05 else ''}"
                    for i, c in enumerate(cols))
    lo, hi = ci(bs[:, -1])
    log(f"  {nm:42s} {txt}   marginal 200→350: {mar:+.3f} CI({lo:+.3f},{hi:+.3f})")
    h4.append(dict(spec=nm, **{f"b_{c}": b[-nb+i] for i, c in enumerate(cols)},
                   marginal_200_350=mar, marginal_lo=lo, marginal_hi=hi,
                   marginal_p=2*min((bs[:, -1] < 0).mean(), (bs[:, -1] > 0).mean())))

# placebo: NEXT season touches predicting THIS season's Δr
pl = tr[tr.next_touches.notna() & (tr.next_season - tr.season == 1)].copy()
Sp, _ = orth(bas["fn"](pl.age.values))
X = np.hstack([Sp, pl[["nt1"]].to_numpy()])
b, _ = fe_ols(pl.dr.values, X, pl.gsis_id.values)


def stat(idx, lab):
    d = pl.iloc[idx]
    bb, _ = fe_ols(d.dr.values, np.hstack([Sp[idx], d[["nt1"]].to_numpy()]), lab)
    return bb[-1:]


bs = cb(stat, pl.gsis_id.values)
log(f"  PLACEBO (next-season touches/100 on this season's Δr, n={len(pl)}): "
    f"β={b[-1]:+.3f} CI{np.round(ci(bs[:,0]),3)}")
h4.append(dict(spec="placebo: next-season touches", b_t1=b[-1], marginal_200_350=np.nan,
               marginal_lo=ci(bs[:, 0])[0], marginal_hi=ci(bs[:, 0])[1]))
pd.DataFrame(h4).to_csv(f"{ROOT}/results/h4_workload_decomp{SUF}.csv", index=False)
log("\nrobustness stage complete")

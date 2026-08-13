"""§G1 inclusion rule + §G2 variance components for the RB universe (EDA_PLAN4.md).

NOTHING is reused from the WR fits. Every quantity here is re-estimated on RB data.

PRE-REGISTERED CHOICES, all fixed from AGGREGATE distributions before any
player-level or model output was inspected (the numbers behind them are printed in
the §G1 block below so the reader can audit that they are aggregate-only):

  §G1 inclusion   regular season; drop player-games with touches = carries+targets
                  <= 1.  (Fixed by the plan itself.)  Excluded fraction and its
                  mean PPR reported, exactly as §0 did for WR.  All rates are
                  *given participation*.

  boom/bust       RB thresholds = the pooled p75 / p25 of PPR points over ALL
  thresholds      qualified (touches>=2) RB REG player-games 2014-2025 — the
                  POSITIONAL distribution, not the 30-man board, so board selection
                  cannot enter.  Computed and FROZEN as BOOM = 13.8, BUST = 3.2
                  before any rate was fitted.  (WR used 20 / 8.)

  relevance gate  §3's WR heteroskedasticity sample gated player-seasons at mean
                  targets/game >= 3, which retains 82.5% of WR player-seasons with
                  >=1 included game.  The RB analogue is set by MATCHING THAT
                  RETENTION RATE on the RB touches/game distribution: the matching
                  value is 4.35 touches/game, rounded to the nearest integer as WR's
                  was, giving **touches/game >= 4** (retains 84.9%).  A >=8
                  touches/game "feature back" cut is reported as a SENSITIVITY, not
                  as an alternative headline.

  windows         headline crossed decomposition on 2021-2025 (the 17-game era), with
                  2014-2025+season-FE, log1p and no-exclusion sensitivities — the
                  identical spec ladder script 02 ran for WR, so the RB/WR comparison
                  is like-for-like.

Estimators
  §G2a  per-player: pooled within-season sigma^2_W (df-weighted), naive between-season
        v, bias-inverted tau^2_B per eq. (3), recency-weighted mu_hat (h=1) with
        n_eff = (sum w)^2 / sum w^2, EB-stabilised boom/bust via the eq.-1.4 Beta
        moment fit.  Script 01's build_table is IMPORTED and re-used with the RB
        thresholds patched in, so the estimator is literally the round-1 estimator.
  §G2b  crossed decomposition Y_isg = mu + a_i + b_is + c_{team x season} + eps by
        REML (MixedLM, single constant group, vc_formula per factor) and by the
        method-of-moments covariance matching of script 02.  rho_max per eq. (5).
  §G2c  location-scale by experience tier: Gamma GLM (log link, dispersion 2)
        headline + Harvey log e^2 check, SEs clustered by player-season.

Outputs: results/consistency_table_rb.csv, results/variance_components_rb.csv,
         results/heteroskedasticity_rb.csv, results/sigma2_by_tier_rb.csv
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path("/Users/thomasmcnamee/NFL")
YEARS = range(2014, 2026)
G_SEASON = 17
BOOM, BUST = 13.8, 3.2            # frozen below from the positional distribution
GATE = 4.0                        # touches/game, retention-matched to WR's 3 tgt/g

spec = importlib.util.spec_from_file_location(
    "s01", ROOT / "scripts" / "01_section1_consistency.py")
s01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s01)

# ============================== load ==============================
COLS = ["player_id", "player_display_name", "position", "season", "week",
        "season_type", "team", "targets", "carries", "fantasy_points_ppr"]
allrb = pd.concat([pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{y}.csv",
                               usecols=COLS, low_memory=False) for y in YEARS],
                  ignore_index=True)
allrb = allrb[(allrb.season_type == "REG") & (allrb.position == "RB")].copy()
allrb["touches"] = allrb.carries.fillna(0) + allrb.targets.fillna(0)

print("=" * 72)
print("§G1  inclusion rule — RB, all RBs 2014-2025 REG")
print("=" * 72)
ex = allrb[allrb.touches <= 1]
print(f"player-games: {len(allrb)};  excluded (touches<=1): {len(ex)} "
      f"({len(ex)/len(allrb):.3%}),  mean PPR of excluded {ex.fantasy_points_ppr.mean():.3f}")
q = allrb[allrb.touches >= 2]
print(f"qualified: {len(q)};  mean PPR {q.fantasy_points_ppr.mean():.3f}; "
      f"PPR p25 {q.fantasy_points_ppr.quantile(.25):.2f}, "
      f"p50 {q.fantasy_points_ppr.quantile(.50):.2f}, "
      f"p75 {q.fantasy_points_ppr.quantile(.75):.2f}")
assert np.isclose(q.fantasy_points_ppr.quantile(.75), BOOM, atol=0.05)
assert np.isclose(q.fantasy_points_ppr.quantile(.25), BUST, atol=0.05)
print(f"FROZEN RB thresholds: boom > {BOOM}, bust < {BUST}   (WR: 20 / 8)")
print("For contrast, the WR rule excluded 1.9% of WR games at mean PPR 1.9 — the RB "
      "exclusion is an order of magnitude larger, which is the committee/backup "
      "structure of the position and is a §G1 finding in its own right.")

# board RBs
brd = allrb[allrb.player_id.isin(
    pd.read_csv(ROOT / "data/meta/rb_top30_meta.csv").gsis_id.dropna())].copy()
brd = brd.rename(columns={"player_id": "gsis_id"})
bex = brd[brd.touches <= 1]
print(f"\nboard-RB REG rows {len(brd)}; excluded {len(bex)} ({len(bex)/len(brd):.3%}), "
      f"mean PPR {bex.fantasy_points_ppr.mean():.3f}")

# ============================== §G2a consistency ==============================
print("\n" + "=" * 72)
print("§G2a  per-player consistency — RB board")
print("=" * 72)
s01_boom, s01_bust = BOOM, BUST


def build_rb(df):
    """script 01 build_table with the RB boom/bust thresholds."""
    src = (ROOT / "scripts" / "01_section1_consistency.py").read_text()
    src = src.replace("(y > 20).sum()", f"(y > {s01_boom}).sum()")
    src = src.replace("(y < 8).sum()", f"(y < {s01_bust}).sum()")
    ns = {"__file__": str(ROOT / "scripts" / "01_section1_consistency.py")}
    exec(compile(src.split('def main()')[0], "s01_rb", "exec"), ns)
    return ns["build_table"](df)


ct = build_rb(brd[brd.touches >= 2].copy())
ct.round(4).to_csv(ROOT / "results/consistency_table_rb.csv", index=False)
print(f"EB boom Beta(a,b) = {tuple(round(x,3) for x in ct.attrs['boom_ab'][:2])}, "
      f"bust = {tuple(round(x,3) for x in ct.attrs['bust_ab'][:2])}")
cols = ["player", "n_seasons", "n_games", "mu_hat", "n_eff", "sigma_W", "naive_v",
        "tau2_B_untrunc", "tau_B", "cv", "q25", "q90", "boom_eb", "bust_eb"]
print(ct[cols].round(3).to_string(index=False))
vets = ct[ct.n_seasons >= 4]
print(f"\nn>=4-season RBs: {len(vets)};  untruncated tau^2_B < 0 in "
      f"{(vets.tau2_B_untrunc < 0).sum()} of them "
      f"({(vets.tau2_B_untrunc < 0).mean():.0%}); "
      f"median tau^2_B_untrunc {vets.tau2_B_untrunc.median():.2f}")
print(f"pooled sigma_W: median {ct.sigma_W.median():.2f}  "
      f"(WR board median was 7.86)")

# ============================== §G2b crossed decomposition ==============================
print("\n" + "=" * 72)
print("§G2b  crossed variance components — RB board")
print("=" * 72)


def prep(years, exclusions=True, log=False):
    d = brd[brd.season.between(*years)].copy()
    if exclusions:
        d = d[d.touches >= 2]
    d["pid"] = d.gsis_id
    d["ps"] = d.gsis_id + "_" + d.season.astype(str)
    d["ts"] = d.team + "_" + d.season.astype(str)
    d["pts"] = d.pid + "_" + d.ts
    d["yv"] = np.log1p(d.fantasy_points_ppr.clip(lower=-0.99)) if log \
        else d.fantasy_points_ppr
    return d.reset_index(drop=True)


def pair_sum(y, groups):
    s = pd.Series(y).groupby(np.asarray(groups))
    g_sum, g_sq, g_n = s.sum(), s.apply(lambda v: (v ** 2).sum()), s.size()
    return float(((g_sum ** 2 - g_sq) / 2).sum()), float((g_n * (g_n - 1) / 2).sum())


def mom(d, season_fe=False):
    y = d.yv.values.astype(float)
    y = y - (d.groupby("season").yv.transform("mean").values if season_fe else y.mean())
    var_y = float(np.mean(y ** 2))
    sp_ps, np_ps = pair_sum(y, d.ps)
    sp_p, np_p = pair_sum(y, d.pid)
    sp_ts, np_ts = pair_sum(y, d.ts)
    sp_pts, np_pts = pair_sum(y, d.pts)
    c_ps = sp_ps / np_ps
    s2P = (sp_p - sp_ps) / (np_p - np_ps)
    n_mate = np_ts - np_pts
    s2T = (sp_ts - sp_pts) / n_mate if n_mate > 0 else np.nan
    s2S = c_ps - s2P - (s2T if n_mate > 0 else 0.0)
    return dict(s2P=s2P, s2S=s2S, s2T=s2T, s2G=var_y - c_ps, n_teammate_pairs=n_mate)


def reml(d, season_fe=False):
    d = d.copy(); d["const_grp"] = 1
    vc = {"player": "0 + C(pid)", "player_season": "0 + C(ps)", "team_season": "0 + C(ts)"}
    r = smf.mixedlm("yv ~ C(season)" if season_fe else "yv ~ 1", d,
                    groups="const_grp", vc_formula=vc).fit(
        reml=True, method=["lbfgs", "powell"], maxiter=2000)
    k = sorted(vc)
    o = dict(zip(k, r.vcomp))
    res = np.asarray(r.resid)
    return dict(s2P=o["player"], s2S=o["player_season"], s2T=o["team_season"],
                s2G=r.scale, converged=bool(r.converged),
                resid_skew=float(stats.skew(res)))


def summ(c):
    tot = c["s2P"] + c["s2S"] + c["s2T"] + c["s2G"]
    o = dict(c)
    for k in ["s2P", "s2S", "s2T", "s2G"]:
        o[f"icc_{k[2:]}"] = c[k] / tot
    o["rho_max"] = c["s2P"] / (c["s2P"] + c["s2S"] + c["s2T"] + c["s2G"] / G_SEASON)
    return o


rows = []
SPECS = [("headline_2021_2025_excl", (2021, 2025), True, False, False),
         ("sens_a_2014_2025_seasonFE", (2014, 2025), True, False, True),
         ("sens_b_log1p_2021_2025", (2021, 2025), True, True, False),
         ("sens_c_no_exclusions", (2021, 2025), False, False, False)]
for name, yrs, excl, log, sfe in SPECS:
    d = prep(yrs, excl, log)
    meta = dict(spec=name, n_obs=len(d), n_players=d.pid.nunique(),
                n_player_seasons=d.ps.nunique(), n_team_seasons=d.ts.nunique())
    m = summ(mom(d, sfe)); rows.append({**meta, "estimator": "MoM_covmatch", **m})
    print(f"[{name}] MoM : " + ", ".join(f"{k}={v:.4g}" for k, v in m.items()))
    try:
        r = summ(reml(d, sfe)); rows.append({**meta, "estimator": "REML_MixedLM", **r})
        print(f"[{name}] REML: " + ", ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    except Exception as e:
        print(f"[{name}] REML FAILED: {e}")
        rows.append({**meta, "estimator": "REML_MixedLM", "converged": False, "error": str(e)})
vc_rb = pd.DataFrame(rows)
vc_rb.round(5).to_csv(ROOT / "results/variance_components_rb.csv", index=False)

# lag-1 same-player covariance (the WR anomaly check), headline spec
d = prep((2021, 2025), True, False)
y = d.fantasy_points_ppr.values - d.fantasy_points_ppr.mean()
sm_ = pd.DataFrame({"pid": d.pid, "season": d.season, "y": y})
seas = sm_.groupby(["pid", "season"]).y.mean().reset_index()
mrg = seas.merge(seas.assign(season=seas.season - 1), on=["pid", "season"],
                 suffixes=("", "_next"))
print(f"\nlag-1 same-player season-mean covariance: {np.cov(mrg.y, mrg.y_next)[0,1]:.3f} "
      f"(n={len(mrg)} adjacent pairs); Var(season mean) "
      f"{seas.y.var(ddof=1):.3f}")

# ============================== §G2c heteroskedasticity ==============================
print("\n" + "=" * 72)
print("§G2c  location-scale by experience tier — all RBs 2014-2025")
print("=" * 72)
meta = pd.read_csv(ROOT / "data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "rookie_season"])
h = q.copy()                                    # qualified RB games, all RBs
ps = (h.groupby(["player_id", "season"])
      .agg(G=("fantasy_points_ppr", "size"), mtch=("touches", "mean"),
           mu_ps=("fantasy_points_ppr", "mean")).reset_index())


def het(gate, label):
    keep = ps[ps.mtch >= gate]
    d = h.merge(keep[["player_id", "season", "mu_ps"]], on=["player_id", "season"])
    d = d.merge(meta.rename(columns={"gsis_id": "player_id"}), on="player_id", how="left")
    d = d.dropna(subset=["rookie_season"])
    d["exp"] = (d.season - d.rookie_season).astype(int)
    d = d[d.exp >= 0].copy()
    d["rookie"] = (d.exp == 0).astype(int); d["soph"] = (d.exp == 1).astype(int)
    d["ps_id"] = d.player_id + "_" + d.season.astype(str)
    d["e2"] = (d.fantasy_points_ppr - d.mu_ps) ** 2
    out = []
    dA = d[d.e2 >= 1e-6]
    mA = smf.ols("np.log(e2) ~ rookie + soph", data=dA).fit(
        cov_type="cluster", cov_kwds={"groups": dA.ps_id})
    X = sm.add_constant(d[["rookie", "soph"]])
    mB = sm.GLM(d.e2, X, family=sm.families.Gamma(link=sm.families.links.Log())).fit(
        scale=2.0, cov_type="cluster", cov_kwds={"groups": d.ps_id})
    for route, m, n in [("A_harvey", mA, len(dA)), ("B_gammaGLM", mB, len(d))]:
        ci = m.conf_int()
        for t in ["rookie", "soph"]:
            out.append(dict(sample=label, route=route, term=t, est=m.params[t],
                            se=m.bse[t], ci_lo=ci.loc[t, 0], ci_hi=ci.loc[t, 1],
                            mult=np.exp(m.params[t]), mult_lo=np.exp(ci.loc[t, 0]),
                            mult_hi=np.exp(ci.loc[t, 1]), n=n,
                            n_player_seasons=d.ps_id.nunique()))
    print(f"\n[{label}] games {len(d)}, player-seasons {d.ps_id.nunique()} "
          f"(rookie {d[d.rookie==1].ps_id.nunique()}, soph {d[d.soph==1].ps_id.nunique()}, "
          f"vet {d[(d.rookie==0)&(d.soph==0)].ps_id.nunique()})")
    print(f"  gamma-GLM multipliers: rookie {np.exp(mB.params['rookie']):.3f} "
          f"[{np.exp(mB.conf_int().loc['rookie',0]):.3f},"
          f"{np.exp(mB.conf_int().loc['rookie',1]):.3f}], "
          f"soph {np.exp(mB.params['soph']):.3f}")
    print("  Wald g1=g2=0:", mB.wald_test("rookie = 0, soph = 0", scalar=True))
    print("  raw mean e2 by tier:", d.groupby(d.exp.clip(upper=2)).e2.mean().round(2).to_dict())
    # level check: does variance scale with level? (the WR mechanism)
    dl = d[(d.mu_ps > 0.5) & (d.e2 >= 1e-6)]
    ml = smf.ols("np.log(e2) ~ np.log(mu_ps) + rookie + soph", data=dl).fit(
        cov_type="cluster", cov_kwds={"groups": dl.ps_id})
    print(f"  log e2 ~ log mu: slope {ml.params['np.log(mu_ps)']:.3f} "
          f"(se {ml.bse['np.log(mu_ps)']:.3f}); tier mults controlling level: "
          f"rookie {np.exp(ml.params['rookie']):.3f} (p {ml.pvalues['rookie']:.3f}), "
          f"soph {np.exp(ml.params['soph']):.3f} (p {ml.pvalues['soph']:.3f})")
    return out, mB


rows, mB_main = het(GATE, f"primary_touch_gate_{GATE:g}")
rows2, _ = het(8.0, "sens_touch_gate_8")
pd.DataFrame(rows + rows2).to_csv(ROOT / "results/heteroskedasticity_rb.csv", index=False)

g0 = mB_main.params["const"]
sig2 = pd.DataFrame(dict(tier=["rookie", "soph", "vet"],
                         sigma2=[np.exp(g0 + mB_main.params["rookie"]),
                                 np.exp(g0 + mB_main.params["soph"]), np.exp(g0)]))
sig2.to_csv(ROOT / "results/sigma2_by_tier_rb.csv", index=False)
print("\nsigma^2(tier) RB:", sig2.set_index("tier").sigma2.round(3).to_dict(),
      " (WR was rookie 36.4 / soph 39.7 / vet 43.1)")
print("\nwrote consistency_table_rb.csv, variance_components_rb.csv, "
      "heteroskedasticity_rb.csv, sigma2_by_tier_rb.csv")

"""§3 location-scale heteroskedasticity by experience tier (EDA_PLAN.md §3).

Pre-specified (before any player-level results):
- Primary sample: ALL WRs, weekly_raw 2014-2025, inclusion rule from §1
  (season_type == REG, targets > 1), restricted to player-seasons with
  season-average targets/game >= 3. Top-30-only as sensitivity.
- Experience e = season - rookie_season (players_meta, join gsis_id).
  Tiers: rookie (e=0), soph (e=1), vet (e>=2).
- Stage 1: e_isg = Y_isg - player-season mean.
- Route A (Harvey): OLS log e^2 on tier dummies (drop e^2 < 1e-6, count them),
  cluster SEs by player-season. gamma_0 recovered by adding +1.2704 to intercept.
- Route B (headline): Gamma GLM of e^2 on tier dummies, log link, dispersion
  fixed at 2 (chi^2_1), cluster SEs by player-season; estimated Pearson scale
  also reported.
- Sensitivities: 2021-2025 window; log sigma^2 linear in min(e,5); top-30 sample.

Outputs: results/heteroskedasticity.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = "/Users/thomasmcnamee/NFL"
YEARS = range(2014, 2026)

# ---------- build sample ----------
frames = []
for y in YEARS:
    df = pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
                     usecols=["player_id", "player_display_name", "position", "season",
                              "week", "season_type", "targets", "fantasy_points_ppr"],
                     low_memory=False)
    frames.append(df[df.position == "WR"])
wr = pd.concat(frames, ignore_index=True)

n_all = len(wr)
wr = wr[wr.season_type == "REG"]
n_reg = len(wr)
wr = wr[wr.targets > 1]                       # §1 inclusion rule: targets <= 1 excluded
n_incl = len(wr)

# player-season aggregates
ps = (wr.groupby(["player_id", "season"])
        .agg(G=("fantasy_points_ppr", "size"),
             mean_tgt=("targets", "mean"),
             mu_ps=("fantasy_points_ppr", "mean"))
        .reset_index())
ps_keep = ps[ps.mean_tgt >= 3.0]
n_ps_all, n_ps_keep = len(ps), len(ps_keep)

wr = wr.merge(ps_keep[["player_id", "season", "mu_ps", "G"]],
              on=["player_id", "season"], how="inner")

# experience
meta = pd.read_csv(f"{ROOT}/data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "rookie_season"])
wr = wr.merge(meta.rename(columns={"gsis_id": "player_id"}), on="player_id", how="left")
n_no_rk = wr.rookie_season.isna().sum()
wr = wr.dropna(subset=["rookie_season"])
wr["exp"] = (wr.season - wr.rookie_season).astype(int)
neg = (wr.exp < 0).sum()
wr = wr[wr.exp >= 0]
wr["rookie"] = (wr.exp == 0).astype(int)
wr["soph"] = (wr.exp == 1).astype(int)
wr["exp_lin"] = wr.exp.clip(upper=5)
wr["ps_id"] = wr.player_id + "_" + wr.season.astype(str)

# stage-1 residuals
wr["resid"] = wr.fantasy_points_ppr - wr.mu_ps
wr["e2"] = wr.resid ** 2

print(f"rows: all WR {n_all} -> REG {n_reg} -> targets>1 {n_incl} -> "
      f"mean_tgt>=3 {len(wr)} game rows")
print(f"player-seasons: {n_ps_all} -> {n_ps_keep} after mean_tgt>=3")
print(f"dropped: no rookie_season {n_no_rk} rows, negative exp {neg} rows")
print(wr.groupby(wr.exp.clip(upper=2)).agg(games=("e2", "size"),
                                           ps=("ps_id", "nunique")))

rows = []

def run_routes(d, label):
    out = []
    # Route A: Harvey log e^2
    dA = d[d.e2 >= 1e-6]
    n_drop = len(d) - len(dA)
    mA = smf.ols("np.log(e2) ~ rookie + soph", data=dA).fit(
        cov_type="cluster", cov_kwds={"groups": dA.ps_id})
    ciA = mA.conf_int()
    for term in ["rookie", "soph"]:
        out.append(dict(sample=label, route="A_harvey", term=term,
                        est=mA.params[term], se=mA.bse[term],
                        ci_lo=ciA.loc[term, 0], ci_hi=ciA.loc[term, 1],
                        mult=np.exp(mA.params[term]),
                        mult_lo=np.exp(ciA.loc[term, 0]),
                        mult_hi=np.exp(ciA.loc[term, 1]),
                        n=len(dA), n_dropped_small_e2=n_drop, scale=np.nan,
                        gamma0=mA.params["Intercept"] + 1.2704))
    # Route B: gamma GLM, log link, dispersion 2
    X = sm.add_constant(d[["rookie", "soph"]])
    mB = sm.GLM(d.e2, X, family=sm.families.Gamma(link=sm.families.links.Log())
                ).fit(scale=2.0, cov_type="cluster", cov_kwds={"groups": d.ps_id})
    # estimated Pearson scale for reporting (refit letting scale float)
    mB_free = sm.GLM(d.e2, X, family=sm.families.Gamma(link=sm.families.links.Log())
                     ).fit(scale="X2")
    ciB = mB.conf_int()
    for term in ["rookie", "soph"]:
        out.append(dict(sample=label, route="B_gammaGLM", term=term,
                        est=mB.params[term], se=mB.bse[term],
                        ci_lo=ciB.loc[term, 0], ci_hi=ciB.loc[term, 1],
                        mult=np.exp(mB.params[term]),
                        mult_lo=np.exp(ciB.loc[term, 0]),
                        mult_hi=np.exp(ciB.loc[term, 1]),
                        n=len(d), n_dropped_small_e2=0,
                        scale=mB_free.scale, gamma0=mB.params["const"]))
    return out, mB

rows_main, mB_main = run_routes(wr, "primary_2014_2025")
rows += rows_main
rows_recent, _ = run_routes(wr[wr.season >= 2021], "window_2021_2025")
rows += rows_recent

# sensitivity: top-30 board players only
ct = pd.read_csv(f"{ROOT}/results/consistency_table.csv")
wr30 = wr[wr.player_id.isin(ct.gsis_id)]
rows_t30, _ = run_routes(wr30, "top30_sensitivity")
rows += rows_t30

# sensitivity: linear in experience 0..5+
dA = wr[wr.e2 >= 1e-6]
mlinA = smf.ols("np.log(e2) ~ exp_lin", data=dA).fit(
    cov_type="cluster", cov_kwds={"groups": dA.ps_id})
Xl = sm.add_constant(wr[["exp_lin"]])
mlinB = sm.GLM(wr.e2, Xl, family=sm.families.Gamma(link=sm.families.links.Log())
               ).fit(scale=2.0, cov_type="cluster", cov_kwds={"groups": wr.ps_id})
for route, m, term, n in [("A_harvey", mlinA, "exp_lin", len(dA)),
                          ("B_gammaGLM", mlinB, "exp_lin", len(wr))]:
    ci = m.conf_int()
    rows.append(dict(sample="linear_exp_0to5", route=route, term=term,
                     est=m.params[term], se=m.bse[term],
                     ci_lo=ci.loc[term, 0], ci_hi=ci.loc[term, 1],
                     mult=np.exp(m.params[term]),
                     mult_lo=np.exp(ci.loc[term, 0]), mult_hi=np.exp(ci.loc[term, 1]),
                     n=n, n_dropped_small_e2=len(wr) - n if route == "A_harvey" else 0,
                     scale=np.nan,
                     gamma0=(m.params.iloc[0] + 1.2704) if route == "A_harvey"
                            else m.params.iloc[0]))

res = pd.DataFrame(rows)
res.to_csv(f"{ROOT}/results/heteroskedasticity.csv", index=False)
print(res.to_string(index=False))

# headline sigma^2 by tier (Route B, primary sample) for §3.4
g0 = mB_main.params["const"]
g1, g2 = mB_main.params["rookie"], mB_main.params["soph"]
sig2 = pd.DataFrame(dict(tier=["rookie", "soph", "vet"],
                         sigma2=[np.exp(g0 + g1), np.exp(g0 + g2), np.exp(g0)]))
sig2.to_csv(f"{ROOT}/results/sigma2_by_tier.csv", index=False)
print(sig2)

# empirical check: raw mean e^2 by tier
print("raw mean e2 by tier:")
print(wr.groupby(wr.exp.clip(upper=2)).e2.mean())
# Wald test H0: g1=g2=0
print("Wald g1=g2=0:", mB_main.wald_test("rookie = 0, soph = 0", scalar=True))

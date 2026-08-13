"""§6.2/§6.3 efficiency-of-the-market test (EDA_PLAN.md §6.2-6.3).

R_is = PPG - m_iso(ADP) (from results/market_prior.csv, §6.1) regressed on the
pre-specified preseason-knowable covariate set Z. ALL choices below were fixed
BEFORE the regression was run (this docstring is the pre-registration):

Z (10 hypothesis terms, FDR family):
  1. age            — (Sept 1 of board year - birth_date)/365.25, players_meta
  2. team_change    — 1{first REG team in board year != last REG team in y-1};
                      team identification uses unfiltered REG appearances;
                      rookies: 0 (joining the league is not a team change)
  3. ts_prior       — prior-season (y-1) target share, sum/sum with team pass
                      attempts joined from stats_team_week (as §4)
  4. adot_prior     — prior-season aDOT = sum rec_air_yards / sum targets
  5. wopr_prior     — 1.5*ts_prior + 0.7*ays_prior (ays sum/sum vs team air yds)
  6. games_prior    — prior-season INCLUDED games (REG, targets>1; identical
                      definition to the panel's `games`). PRE-REGISTERED
                      ADDITION to the plan's §6.2 list, motivated by the §3.4
                      blind-run availability diagnostic (blind posterior
                      over-ranks per-game-good availability-risky players)
                      BEFORE this regression was run.
  7. rookie         — 1{exp == 0}
  8. vet_x_change   — 1{exp >= 2} * team_change
  9. agec_x_adotc   — (age - fit mean) * (adot_prior - fit mean), centered mains
 10. rook_x_epa     — rookie * z(new-team pass EPA in y-1); team pass EPA =
                      season mean of weekly passing_epa (REG), z-scored within
                      season across teams (cross-era comparability). Expected
                      no power (n=4 rookies); reported anyway per plan.

Missing-data handling (decided before fitting):
  - Rookies have no prior-season stats: ratio stats (ts, adot, wopr) imputed at
    the non-rookie fit-sample mean; the rookie dummy is the missingness
    indicator, so the imputation constant is absorbed and the ratio-stat betas
    are identified off non-rookies only. games_prior for rookies likewise
    imputed at the non-rookie mean (0 would encode "missed a full season").
  - Non-rookies with zero y-1 included games: games_prior = 0 is the TRUE
    value (availability information, kept); ratio stats imputed at the
    non-rookie mean with a nuisance indicator `no_prior_vet` added as a
    control, NOT a member of the FDR family.

Inference: OLS on in_fit rows (>=4 realized games, n=291); SEs clustered by
season (use_t, G-1 df); HC3 as robustness. Benjamini-Hochberg FDR q=0.10
across the 10 clustered p-values. Binding temporal test (§6.3, amended window
documented in section6a notes: no 2025 ADP exists at source, so fit 2015-2022
/ evaluate 2023-2024, not the plan's 2023-2025): refit FDR survivors on
2015-2022 in-fit rows, compare holdout squared-error prediction of R on
2023-2024 vs the zero prediction (market efficiency). A term is used in §6.4
only if it survives BOTH.

Outputs: results/edge_panel.csv (panel + covariates, reused by 10/11),
         results/edge_regression.csv (betas, SEs, t, p, FDR, holdout verdicts)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = "/Users/thomasmcnamee/NFL"
RNG = np.random.default_rng(60223)

# ---------------- load panel & meta ----------------
panel = pd.read_csv(f"{ROOT}/results/market_prior.csv")
meta = pd.read_csv(f"{ROOT}/data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "birth_date", "rookie_season"])
panel = panel.merge(meta, on="gsis_id", how="left", suffixes=("", "_meta"))
assert panel.birth_date.notna().all(), \
    f"missing birth dates: {panel[panel.birth_date.isna()].name.tolist()}"
panel["age"] = (pd.to_datetime(panel.year.astype(str) + "-09-01")
                - pd.to_datetime(panel.birth_date)).dt.days / 365.25

# ---------------- weekly data: prior-season stats + team ids ----------------
YEARS_WK = range(2014, 2025)   # prior seasons 2014..2023 + board years 2015..2024
wk_frames, tw_frames = [], []
for y in YEARS_WK:
    df = pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
                     usecols=["player_id", "season", "week", "season_type", "team",
                              "targets", "receiving_air_yards", "fantasy_points_ppr"],
                     low_memory=False)
    wk_frames.append(df[df.season_type == "REG"])
    tw = pd.read_csv(f"{ROOT}/data/teams/stats_team_week_{y}.csv",
                     usecols=["season", "week", "team", "season_type", "attempts",
                              "passing_air_yards", "passing_epa"])
    tw_frames.append(tw[tw.season_type == "REG"])
wk = pd.concat(wk_frames, ignore_index=True)          # unfiltered REG appearances
tw = pd.concat(tw_frames, ignore_index=True)

# team ids: first REG team in board year, last REG team in prior season
first_team = (wk.sort_values("week").groupby(["player_id", "season"])
                .team.first().rename("team_first"))
last_team = (wk.sort_values("week").groupby(["player_id", "season"])
               .team.last().rename("team_last"))

# included games (inclusion rule) with team denominators joined
inc = wk[wk.targets > 1].merge(
    tw.rename(columns={"attempts": "team_att", "passing_air_yards": "team_air",
                       "passing_epa": "team_epa"}),
    on=["season", "week", "team"], how="left")
assert inc.team_att.notna().all(), "team join failure on included games"

ps = (inc.groupby(["player_id", "season"])
         .agg(g_inc=("fantasy_points_ppr", "size"),
              tgt=("targets", "sum"), ray=("receiving_air_yards", "sum"),
              t_att=("team_att", "sum"), t_air=("team_air", "sum"))
         .reset_index())
ps["ts"] = ps.tgt / ps.t_att
ps["ays"] = ps.ray / ps.t_air
ps["wopr"] = 1.5 * ps.ts + 0.7 * ps.ays
ps["adot"] = ps.ray / ps.tgt

# team season pass EPA, z-scored within season
tse = (tw.groupby(["season", "team"]).passing_epa.mean().rename("epa_pg")
         .reset_index())
tse["epa_z"] = tse.groupby("season").epa_pg.transform(
    lambda x: (x - x.mean()) / x.std(ddof=0))

# ---------------- assemble covariates ----------------
panel["prior_season"] = panel.year - 1
p = panel.merge(ps.rename(columns={"player_id": "gsis_id", "season": "prior_season",
                                   "g_inc": "games_prior", "ts": "ts_prior",
                                   "adot": "adot_prior", "wopr": "wopr_prior"})
                  [["gsis_id", "prior_season", "games_prior", "ts_prior",
                    "adot_prior", "wopr_prior"]],
                on=["gsis_id", "prior_season"], how="left")
p = p.merge(first_team.reset_index().rename(
        columns={"player_id": "gsis_id", "season": "year", "team_first": "team_now"}),
        on=["gsis_id", "year"], how="left")
p = p.merge(last_team.reset_index().rename(
        columns={"player_id": "gsis_id", "season": "prior_season",
                 "team_last": "team_prev"}),
        on=["gsis_id", "prior_season"], how="left")
p = p.merge(tse.rename(columns={"season": "prior_season", "team": "team_now"})
              [["prior_season", "team_now", "epa_z"]],
            on=["prior_season", "team_now"], how="left")

p["rookie"] = (p.exp == 0).astype(int)
p["vet"] = (p.exp >= 2).astype(int)
p["no_prior"] = p.games_prior.isna().astype(int)          # incl. rookies
p["no_prior_vet"] = ((p.no_prior == 1) & (p.rookie == 0)).astype(int)
# team change: needs both teams observed; rookies -> 0
p["team_change"] = np.where(p.rookie == 1, 0.0,
                    np.where(p.team_now.notna() & p.team_prev.notna(),
                             (p.team_now != p.team_prev).astype(float), np.nan))

print("=== panel covariate diagnostics ===")
print(f"rows {len(p)}; in_fit {p.in_fit.sum()}")
print(f"no prior-season included games: total {p.no_prior.sum()} "
      f"(rookies {p.rookie.sum()}, non-rookie {p.no_prior_vet.sum()})")
print(p[p.no_prior_vet == 1][["year", "name", "exp", "games", "in_fit"]]
      .to_string(index=False))
print(f"team_now missing: {p.team_now.isna().sum()} "
      f"(in_fit: {(p.team_now.isna() & p.in_fit).sum()})")
print(f"team_change=1 rate among non-rookie in_fit: "
      f"{p[(p.rookie==0)&p.in_fit].team_change.mean():.3f}")
print(f"rookies on boards: {p[p.rookie==1][['year','name','epa_z']].to_string(index=False)}")

# imputation at NON-ROOKIE fit-sample means (decided before fitting; see header)
fitmask = p.in_fit
nr_fit = p[fitmask & (p.no_prior == 0)]
for c in ["ts_prior", "adot_prior", "wopr_prior"]:
    p[c] = p[c].fillna(nr_fit[c].mean())
p.loc[p.rookie == 1, "games_prior"] = nr_fit.games_prior.mean()   # rookie: impute
p.loc[(p.no_prior_vet == 1), "games_prior"] = 0.0                 # vet miss: true 0
p["games_prior"] = p.games_prior.fillna(0.0)                      # (none left)
p["rook_x_epa"] = p.rookie * p.epa_z.fillna(0.0)
p["vet_x_change"] = p.vet * p.team_change
age0 = p.loc[fitmask, "age"].mean(); adot0 = p.loc[fitmask, "adot_prior"].mean()
p["agec_x_adotc"] = (p.age - age0) * (p.adot_prior - adot0)

p.to_csv(f"{ROOT}/results/edge_panel.csv", index=False)

# ---------------- §6.2 regression ----------------
ZCOLS = ["age", "team_change", "ts_prior", "adot_prior", "wopr_prior",
         "games_prior", "rookie", "vet_x_change", "agec_x_adotc", "rook_x_epa"]
NUIS = ["no_prior_vet"] if p.loc[fitmask, "no_prior_vet"].sum() > 0 else []
d = p[fitmask].dropna(subset=["resid_iso"] + ZCOLS).copy()
print(f"\nregression n = {len(d)} (in_fit {fitmask.sum()}; "
      f"dropped for missing covariates: {int(fitmask.sum()) - len(d)})")
X = sm.add_constant(d[ZCOLS + NUIS])
yv = d.resid_iso

m_cl = sm.OLS(yv, X).fit(cov_type="cluster",
                         cov_kwds={"groups": d.year}, use_t=True)
m_h3 = sm.OLS(yv, X).fit(cov_type="HC3")

# VIF (collinearity readout, ts/wopr expected high)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.Series([variance_inflation_factor(X.values, i)
                 for i in range(X.shape[1])], index=X.columns)

tab = pd.DataFrame({
    "beta": m_cl.params, "se_cluster": m_cl.bse, "t_cluster": m_cl.tvalues,
    "p_cluster": m_cl.pvalues, "se_hc3": m_h3.bse, "p_hc3": m_h3.pvalues,
    "vif": vif})

# Benjamini-Hochberg q=0.10 across the 10 hypothesis terms (clustered p's)
fam = tab.loc[ZCOLS].copy()
o = fam.p_cluster.sort_values()
k = len(o)
bh_thresh = 0.10 * np.arange(1, k + 1) / k
passed = o.values <= bh_thresh
kmax = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
survivors = list(o.index[:kmax])
tab["fdr_survivor"] = [ix in survivors for ix in tab.index]

print("\n=== §6.2 R ~ Z, in_fit 2015-2024, clustered by season (10 clusters) ===")
print(tab.round(4).to_string())
# Degeneracy diagnostic (found, not assumed): no soph in the 300-row panel ever
# changed teams (rookie-contract selection into top-30 boards), so team_change
# == vet_x_change on the fit sample -> only their sum is identified; pinv
# splits the combined beta between them. Joint Wald therefore runs on the
# identified 9-term set (vet_x_change dropped from the constraint matrix).
tc_eq = (d.team_change == d.vet_x_change).all()
print(f"\nteam_change == vet_x_change on fit sample: {tc_eq} "
      f"(identified combined beta = {m_cl.params['team_change'] + m_cl.params['vet_x_change']:+.4f})")
wald_cols = [c for c in ZCOLS if not (tc_eq and c == "vet_x_change")]
wald = m_cl.wald_test(np.eye(len(m_cl.params))[[X.columns.get_loc(c) for c in wald_cols]],
                      scalar=True)
print(f"R2 = {m_cl.rsquared:.4f}; joint Wald ({len(wald_cols)} identified Z terms = 0, "
      f"clustered): F = {wald.statistic:.3f}, p = {wald.pvalue:.4f}")
print("  [caveat: 9 constraints vs cluster-cov rank <= G-1 = 9 -> at the rank "
      "boundary, and the degenerate rookie t inflates it; diagnostic below]")
# diagnostic: veteran-side efficiency (exclude the two rookie terms, which are
# 4 rows / 2 clusters); this is the interpretable joint test
nonrk = [c for c in wald_cols if c not in ("rookie", "rook_x_epa")]
w2 = m_cl.wald_test(np.eye(len(m_cl.params))[[X.columns.get_loc(c) for c in nonrk]],
                    scalar=True)
w2h = m_h3.wald_test(np.eye(len(m_h3.params))[[X.columns.get_loc(c) for c in nonrk]],
                     scalar=True)
print(f"diagnostic joint Wald, {len(nonrk)} non-rookie terms: clustered "
      f"F = {w2.statistic:.3f}, p = {w2.pvalue:.4f}; HC3 F = {w2h.statistic:.3f}, "
      f"p = {w2h.pvalue:.4f}")
# few-treated-clusters caveat: rookie/rook_x_epa live in n=4 rows in 2 of 10
# season clusters; cluster-robust (and HC3, n=4) inference is unreliable there.
print(f"rookie rows by season cluster: "
      f"{d[d.rookie==1].year.value_counts().to_dict()} "
      f"-> clustered t on rookie terms is a few-treated-clusters artifact; "
      f"the binding temporal gate below is the real test")
print(f"\nBH FDR q=0.10 survivors: {survivors if survivors else 'NONE'}")
print("sorted p vs BH thresholds:")
print(pd.DataFrame({"p": o.round(4), "bh": bh_thresh.round(4)}).to_string())

# ---------------- §6.3 binding temporal holdout ----------------
# (amended window: fit 2015-2022, evaluate 2023-2024; no 2025 ADP at source)
tr = d[d.year <= 2022]; te = d[d.year >= 2023]
print(f"\n=== temporal holdout: fit n={len(tr)} (2015-22), "
      f"eval n={len(te)} (2023-24) ===")
mse0 = float((te.resid_iso ** 2).mean())
print(f"zero-prediction (market) holdout MSE: {mse0:.4f}")
mi = sm.OLS(tr.resid_iso, np.ones(len(tr))).fit()
mse_int = float(((te.resid_iso - mi.params.iloc[0]) ** 2).mean())
print(f"intercept-only holdout MSE: {mse_int:.4f} (intercept {mi.params.iloc[0]:+.3f})")

hold_rows = []
def hold_eval(terms, label):
    Xt = sm.add_constant(tr[terms + NUIS]); Xe = sm.add_constant(te[terms + NUIS])
    Xe = Xe[Xt.columns]
    f = sm.OLS(tr.resid_iso, Xt).fit()
    pred = f.predict(Xe)
    mse = float(((te.resid_iso - pred) ** 2).mean())
    # per-year mean squared-error difference vs zero prediction (2 years only;
    # descriptive) + row-level paired t as reported evidence, MSE is binding
    dsq = (te.resid_iso ** 2) - (te.resid_iso - pred) ** 2   # >0 = model better
    byyr = dsq.groupby(te.year).mean()
    tt = dsq.mean() / (dsq.std(ddof=1) / np.sqrt(len(dsq)))
    hold_rows.append(dict(model=label, mse=mse, mse_zero=mse0,
                          improves=mse < mse0, dsq_mean=dsq.mean(),
                          t_rowlevel=tt,
                          dsq_2023=byyr.get(2023, np.nan),
                          dsq_2024=byyr.get(2024, np.nan)))
    print(f"  {label:28s} MSE {mse:8.4f}  improves vs zero: {mse < mse0}  "
          f"mean d_sq {dsq.mean():+.3f} (row t {tt:+.2f}; "
          f"by yr {byyr.round(3).to_dict()})")
    return f

if survivors:
    hold_eval(survivors, "joint: " + "+".join(survivors))
    for s0 in survivors:
        hold_eval([s0], f"single: {s0}")
else:
    print("no FDR survivors -> no temporal test to run; efficiency not rejected")

# survivors of BOTH gates (single-term holdout improvement is the per-term gate)
final_terms = [s0 for s0 in survivors
               if next(r for r in hold_rows if r["model"] == f"single: {s0}")
               ["improves"]] if survivors else []
print(f"\nterms surviving BOTH FDR and temporal holdout: "
      f"{final_terms if final_terms else 'NONE'}")

tab["holdout_improves"] = [
    (next((r["improves"] for r in hold_rows if r["model"] == f"single: {ix}"), None))
    for ix in tab.index]
tab["final_survivor"] = [ix in final_terms for ix in tab.index]
tab.index.name = "term"
tab.to_csv(f"{ROOT}/results/edge_regression.csv")
pd.DataFrame(hold_rows).to_csv(f"{ROOT}/results/edge_holdout.csv", index=False)
print("\nwrote results/edge_panel.csv, edge_regression.csv, edge_holdout.csv")

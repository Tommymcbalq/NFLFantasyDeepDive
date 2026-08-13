"""EDA round 2 — §D0 derived tables (situation part) + §B situation change.

All definitions below were fixed BEFORE any regression was run (this docstring
is the pre-registration). Source data = existing caches only
(data/players/weekly_raw/, data/teams/); REG rows only throughout.
Team codes in the nflverse cache are already franchise-normalized
(LA/LAC/LV in every season), verified before writing — no relocation mapping.

D0 derived tables (data/derived/):
  qb_by_team_season.csv  — per (season, team): primary QB = passer (any
      position) with most summed attempts; qb_att_share = his attempts /
      team's total player-attempts.
  situation_change.csv   — WR player-seasons 2015-2025 passing the gate
      targets/appearance >= 3 (appearance = any REG weekly row).
      team_first_s = team of first REG week in s; team_last_prev = team of
      last REG week in s-1 (any WR row in s-1, no gate on s-1 — the change is
      defined vs wherever he actually was). has_prior = played >=1 REG game
      in s-1. team_change = 1{team_first_s != team_last_prev};
      qb_change_same_team = 1{same team AND primary QB of that team differs
      s-1 -> s}; any_change = OR. All NaN when has_prior = 0.
  vacated_targets.csv    — per (team, season s in 2015-2025): share of the
      team's season-(s-1) REG targets thrown to players with zero REG
      appearances for that team in season s.

§B panel (pairs s -> s+1, s in 2014-2024 so both derived seasons exist for
change flags; B-samples use s >= 2015 per plan): WR, gate (>=3 targets/
appearance) in season s; season s+1 requires >= 4 REG appearances (an
availability screen, pre-specified — NOT a performance gate in s+1, to avoid
conditioning slopes on the outcome level; sensitivity with no s+1 screen
reported in notes). Season ratio stats are sum/sum with team weekly REG
totals joined on (season, week, team):
  TS   = sum targets / sum team attempts (over the player's weeks)
  AYS  = sum rec air yards / sum team passing_air_yards
  WOPR = 1.5*TS + 0.7*AYS ;  aDOT = sum rec air yards / sum targets
  PPG  = sum fantasy_points_ppr / n REG appearances.

B1: X_{s+1} ~ C(regime) + C(regime):X_s, regime in {no_change, team_change,
    qb_change_same_team} (pairs with has_prior... i.e. change flags for
    season s+1 vs s; by construction prior exists). OLS, SEs clustered by
    player (gsis_id). Report per-regime slope + 95% CI and the 2-df Wald
    test of slope equality.
B2: dPPG = PPG_{s+1}-PPG_s ~ team_change + qb_change_same_team +
    z_att_new + z_epa_new + vacated_new, where the "new" team = team_first
    in s+1, z's are within-season (32-team) z-scores of that team's SEASON-s
    pass attempts and mean weekly passing_epa, vacated_new = vacated share
    of that team entering s+1 (defined for movers and stayers alike).
    OLS, SEs clustered by season s+1 (use_t).
B3 (edge protocol, binding): market_prior.csv in_fit rows 2015-2024,
    R = resid_iso ~ const + {tc_x_vacated, tc_x_epa_new, qb_change_same_team}
    (3-member FDR family, exactly the plan's list; team_change main effect
    is NOT in the family — the plan pre-specifies the interactions only).
    Rookies / no-prior players: all three terms = 0 (no situation change is
    measurable; matches script 09's rookie team_change=0 convention).
    SEs clustered by season (10 clusters, use_t -> t(9)); BH-FDR q=0.10 on
    the 3 clustered p's; temporal holdout refit 2015-2022, evaluate MSE of R
    on 2023-2024 vs the zero prediction. Survival requires BOTH. A null is
    the result.

Outputs: data/derived/{qb_by_team_season,situation_change,vacated_targets}.csv
         results/situation_change.csv (B1+B2 estimates)
         results/edge_situation.csv   (B3)
         results/sectionB_notes.md    (written after anomaly chase)
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = "/Users/thomasmcnamee/NFL"
os.makedirs(f"{ROOT}/data/derived", exist_ok=True)

# ---------------- load ----------------
PCOLS = ["player_id", "player_display_name", "position_group", "season", "week",
         "season_type", "team", "attempts", "targets", "receiving_air_yards",
         "fantasy_points_ppr"]
wk = pd.concat([pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
                            usecols=PCOLS) for y in range(2014, 2026)])
wk = wk[wk.season_type == "REG"].copy()

TCOLS = ["season", "week", "team", "season_type", "attempts", "passing_air_yards",
         "passing_epa"]
tm = pd.concat([pd.read_csv(f"{ROOT}/data/teams/stats_team_week_{y}.csv",
                            usecols=TCOLS) for y in range(2014, 2026)])
tm = tm[tm.season_type == "REG"].copy()

# ---------------- D0.1 primary QB ----------------
pa = (wk[wk.attempts > 0]
      .groupby(["season", "team", "player_id", "player_display_name"], as_index=False)
      .attempts.sum())
tot = pa.groupby(["season", "team"]).attempts.sum().rename("team_player_attempts")
qb = (pa.sort_values("attempts", ascending=False)
        .groupby(["season", "team"], as_index=False).first()
        .rename(columns={"player_id": "qb_gsis_id", "player_display_name": "qb_name",
                         "attempts": "qb_attempts"})
        .merge(tot, on=["season", "team"]))
qb["qb_att_share"] = qb.qb_attempts / qb.team_player_attempts
qb = qb.sort_values(["season", "team"])
qb.to_csv(f"{ROOT}/data/derived/qb_by_team_season.csv", index=False)

# ---------------- D0.3 vacated targets ----------------
ptt = (wk.groupby(["season", "team", "player_id"], as_index=False)
         .targets.sum())          # player-team-season targets (all positions)
roster = wk.groupby(["season", "team"]).player_id.apply(set)
rows = []
for (s_prev, team), grp in ptt.groupby(["season", "team"]):
    s = s_prev + 1
    if s < 2015 or s > 2025 or (s, team) not in roster.index:
        continue
    r = roster.loc[(s, team)]
    tot_t = grp.targets.sum()
    vac = grp.loc[~grp.player_id.isin(r), "targets"].sum()
    rows.append((team, s, tot_t, vac, vac / tot_t if tot_t > 0 else np.nan))
vacated = pd.DataFrame(rows, columns=["team", "season", "prior_targets",
                                      "vacated_targets", "vacated_share"])
vacated = vacated.sort_values(["season", "team"])
vacated.to_csv(f"{ROOT}/data/derived/vacated_targets.csv", index=False)

# ---------------- WR season aggregates (sum/sum with team joins) ----------------
wr = wk[wk.position_group == "WR"].copy()
wr = wr.merge(tm[["season", "week", "team", "attempts", "passing_air_yards"]]
              .rename(columns={"attempts": "tm_att", "passing_air_yards": "tm_ay"}),
              on=["season", "week", "team"], how="left")
agg = (wr.groupby(["player_id", "player_display_name", "season"], as_index=False)
         .agg(games=("week", "size"), targets=("targets", "sum"),
              air=("receiving_air_yards", "sum"), ppr=("fantasy_points_ppr", "sum"),
              tm_att=("tm_att", "sum"), tm_ay=("tm_ay", "sum")))
agg["tpg"] = agg.targets / agg.games
agg["TS"] = agg.targets / agg.tm_att
agg["AYS"] = agg.air / agg.tm_ay
agg["WOPR"] = 1.5 * agg.TS + 0.7 * agg.AYS
agg["aDOT"] = np.where(agg.targets > 0, agg.air / agg.targets, np.nan)
agg["PPG"] = agg.ppr / agg.games

first_last = (wr.sort_values("week").groupby(["player_id", "season"])
                .team.agg(team_first="first", team_last="last").reset_index())
agg = agg.merge(first_last, on=["player_id", "season"])

# ---------------- D0.2 situation change (gated WR seasons 2015-2025) ----------------
prev = agg[["player_id", "season", "team_last"]].copy()
prev["season"] += 1
prev = prev.rename(columns={"team_last": "team_last_prev"})
sc = agg.merge(prev, on=["player_id", "season"], how="left")
qbk = qb.set_index(["season", "team"]).qb_gsis_id
sc["qb_now"] = [qbk.get((s, t), np.nan) for s, t in zip(sc.season, sc.team_first)]
sc["qb_prev"] = [qbk.get((s - 1, t), np.nan)
                 for s, t in zip(sc.season, sc.team_last_prev)]
sc["has_prior"] = sc.team_last_prev.notna()
sc["team_change"] = np.where(sc.has_prior,
                             (sc.team_first != sc.team_last_prev).astype(float), np.nan)
sc["qb_change_same_team"] = np.where(
    sc.has_prior, ((sc.team_change == 0) & (sc.qb_now != sc.qb_prev)).astype(float),
    np.nan)
sc["any_change"] = np.where(sc.has_prior,
                            ((sc.team_change == 1) |
                             (sc.qb_change_same_team == 1)).astype(float), np.nan)
sit = sc[(sc.season >= 2015) & (sc.tpg >= 3)][
    ["player_id", "player_display_name", "season", "games", "targets", "tpg",
     "team_first", "team_last_prev", "has_prior", "team_change",
     "qb_change_same_team", "any_change"]].sort_values(["season", "player_id"])
sit.to_csv(f"{ROOT}/data/derived/situation_change.csv", index=False)

# ---------------- §B pair panel ----------------
nxt = sc[["player_id", "season", "games", "TS", "WOPR", "aDOT", "PPG",
          "team_first", "team_change", "qb_change_same_team"]].copy()
nxt.columns = ["player_id", "season_next", "games1", "TS1", "WOPR1", "aDOT1",
               "PPG1", "team_new", "team_change", "qb_change_same_team"]
nxt["season"] = nxt.season_next - 1
pairs = (sc[(sc.season >= 2015) & (sc.season <= 2024) & (sc.tpg >= 3)]
         [["player_id", "player_display_name", "season", "TS", "WOPR", "aDOT", "PPG"]]
         .merge(nxt.drop(columns="season_next"), on=["player_id", "season"]))
pairs = pairs[pairs.games1 >= 4].copy()
pairs["regime"] = np.select([pairs.team_change == 1, pairs.qb_change_same_team == 1],
                            ["team_change", "qb_change"], "no_change")

# new-situation quality (season-s team stats of the s+1 team, z within season)
tse = (tm.groupby(["season", "team"], as_index=False)
         .agg(att=("attempts", "sum"), epa=("passing_epa", "mean")))
for c in ["att", "epa"]:
    tse[f"z_{c}"] = tse.groupby("season")[c].transform(lambda x: (x - x.mean()) / x.std())
q = tse[["season", "team", "z_att", "z_epa"]].rename(columns={"team": "team_new"})
pairs = pairs.merge(q, on=["season", "team_new"], how="left")   # season-s stats
pairs = pairs.merge(vacated[["team", "season", "vacated_share"]]
                    .rename(columns={"team": "team_new"})
                    .assign(season=lambda d: d.season - 1),      # entering s+1
                    on=["season", "team_new"], how="left")
pairs["dPPG"] = pairs.PPG1 - pairs.PPG

# ---------------- B1 ----------------
b1_rows = []
for X in ["TS", "WOPR", "aDOT"]:
    d = pairs.dropna(subset=[X, f"{X}1"]).copy()
    d["x"], d["y"] = d[X], d[f"{X}1"]
    m = smf.ols("y ~ C(regime, Treatment('no_change')) "
                "+ C(regime, Treatment('no_change')):x + x", data=d
                ).fit(cov_type="cluster",
                      cov_kwds={"groups": d.player_id}, use_t=True)
    base = m.params["x"]
    tc_term = "C(regime, Treatment('no_change'))[T.team_change]:x"
    qc_term = "C(regime, Treatment('no_change'))[T.qb_change]:x"
    eq = m.wald_test(f"{tc_term} = 0, {qc_term} = 0", scalar=True)
    for reg, extra in [("no_change", None), ("team_change", tc_term),
                       ("qb_change", qc_term)]:
        if extra is None:
            slope, se = base, m.bse["x"]
            lo, hi = m.conf_int().loc["x"]
        else:
            lc = m.t_test(f"x + {extra}")
            slope, se = float(np.ravel(lc.effect)[0]), float(np.ravel(lc.sd)[0])
            lo, hi = float(lc.conf_int()[0, 0]), float(lc.conf_int()[0, 1])
        b1_rows.append(dict(block="B1", stat=X, term=f"slope_{reg}",
                            n=len(d), n_regime=(d.regime == reg).sum(),
                            est=slope, se=se, ci_lo=lo, ci_hi=hi,
                            p=np.nan, note=""))
    b1_rows.append(dict(block="B1", stat=X, term="slope_equality_wald",
                        n=len(d), n_regime=np.nan, est=eq.statistic, se=np.nan,
                        ci_lo=np.nan, ci_hi=np.nan, p=eq.pvalue,
                        note="F, 2 df, player-clustered"))
    # pre-specified sensitivity: no s+1 availability screen
    d2 = (sc[(sc.season >= 2015) & (sc.season <= 2024) & (sc.tpg >= 3)]
          [["player_id", "season", X]]
          .merge(nxt.drop(columns="season_next"), on=["player_id", "season"])
          .dropna(subset=[X, f"{X}1"]))
    d2["x"], d2["y"], d2["regime"] = d2[X], d2[f"{X}1"], np.select(
        [d2.team_change == 1, d2.qb_change_same_team == 1],
        ["team_change", "qb_change"], "no_change")
    m2 = smf.ols("y ~ C(regime, Treatment('no_change'))*x", data=d2
                 ).fit(cov_type="cluster", cov_kwds={"groups": d2.player_id},
                       use_t=True)
    eq2 = m2.wald_test(
        "C(regime, Treatment('no_change'))[T.team_change]:x = 0,"
        "C(regime, Treatment('no_change'))[T.qb_change]:x = 0", scalar=True)
    b1_rows.append(dict(block="B1_sens_noscreen", stat=X, term="slope_equality_wald",
                        n=len(d2), n_regime=np.nan, est=eq2.statistic, se=np.nan,
                        ci_lo=np.nan, ci_hi=np.nan, p=eq2.pvalue, note=""))

# ---------------- B2 ----------------
d = pairs.dropna(subset=["dPPG", "team_change", "qb_change_same_team",
                         "z_att", "z_epa", "vacated_share"]).copy()
d["season_next"] = d.season + 1
m2 = smf.ols("dPPG ~ team_change + qb_change_same_team + z_att + z_epa"
             " + vacated_share", data=d
             ).fit(cov_type="cluster", cov_kwds={"groups": d.season_next},
                   use_t=True)
b2_rows = [dict(block="B2", stat="dPPG", term=t, n=len(d), n_regime=np.nan,
                est=m2.params[t], se=m2.bse[t],
                ci_lo=m2.conf_int().loc[t, 0], ci_hi=m2.conf_int().loc[t, 1],
                p=m2.pvalues[t], note="season(s+1)-clustered, use_t")
           for t in m2.params.index]
pd.DataFrame(b1_rows + b2_rows).to_csv(f"{ROOT}/results/situation_change.csv",
                                       index=False)

# ---------------- B3 ----------------
mp = pd.read_csv(f"{ROOT}/results/market_prior.csv")
mp = mp[mp.in_fit].copy()
scj = sc[["player_id", "season", "team_first", "team_change",
          "qb_change_same_team"]].rename(
    columns={"player_id": "gsis_id", "season": "year"})
mp = mp.merge(scj, on=["gsis_id", "year"], how="left")
# no-prior (rookies etc.): all family terms 0
for c in ["team_change", "qb_change_same_team"]:
    mp[c] = mp[c].fillna(0)
mp = mp.merge(vacated[["team", "season", "vacated_share"]]
              .rename(columns={"team": "team_first", "season": "year"}),
              on=["team_first", "year"], how="left")
mp = mp.merge(tse[["season", "team", "z_epa"]]
              .rename(columns={"team": "team_first"})
              .assign(year=lambda x: x.season + 1).drop(columns="season"),
              on=["team_first", "year"], how="left")
mp["tc_x_vacated"] = mp.team_change * mp.vacated_share.fillna(0)
mp["tc_x_epa_new"] = mp.team_change * mp.z_epa.fillna(0)
FAM = ["tc_x_vacated", "tc_x_epa_new", "qb_change_same_team"]
Xd = sm.add_constant(mp[FAM])
m3 = sm.OLS(mp.resid_iso, Xd).fit(cov_type="cluster",
                                  cov_kwds={"groups": mp.year}, use_t=True)
pv = m3.pvalues[FAM].sort_values()
k = len(FAM)
bh = pv.values <= 0.10 * np.arange(1, k + 1) / k
cut = np.max(np.where(bh)[0]) if bh.any() else -1
fdr = {t: (i <= cut) for i, t in enumerate(pv.index)}
# temporal holdout on FDR survivors (and, for the record, the full family)
fit_m = mp.year <= 2022
hold_m = mp.year >= 2023
mse0 = float((mp.loc[hold_m, "resid_iso"] ** 2).mean())
hold_res = {}
for label, terms in [("survivors", [t for t in FAM if fdr[t]]), ("full_family", FAM)]:
    if not terms:
        hold_res[label] = np.nan
        continue
    Xf = sm.add_constant(mp.loc[fit_m, terms])
    mm = sm.OLS(mp.loc[fit_m, "resid_iso"], Xf).fit()
    pred = sm.add_constant(mp.loc[hold_m, terms]) @ mm.params
    hold_res[label] = float(((mp.loc[hold_m, "resid_iso"] - pred) ** 2).mean())
edge = pd.DataFrame([dict(term=t, beta=m3.params[t], se_cluster=m3.bse[t],
                          t=m3.tvalues[t], p_cluster=m3.pvalues[t],
                          fdr_survivor=fdr[t],
                          holdout_mse_survivors=hold_res["survivors"],
                          holdout_mse_full=hold_res["full_family"],
                          holdout_mse_zero=mse0,
                          holdout_improves=(fdr[t] and
                                            hold_res["survivors"] < mse0),
                          final_survivor=(fdr[t] and
                                          hold_res["survivors"] < mse0))
                     for t in FAM])
edge.to_csv(f"{ROOT}/results/edge_situation.csv", index=False)

# ---------------- console report ----------------
print("== D0 sanity ==")
print("qb_by_team_season rows:", len(qb), " seasons", qb.season.min(), "-",
      qb.season.max(), " mean qb_att_share %.3f" % qb.qb_att_share.mean())
print("situation_change rows:", len(sit), " gated WR-seasons/yr ~",
      round(len(sit) / sit.season.nunique(), 1))
pr = sit[sit.has_prior]
print("team_change rate %.3f  qb_change_same_team rate %.3f  any %.3f  "
      "no_prior share %.3f" % (pr.team_change.mean(),
                               pr.qb_change_same_team.mean(),
                               pr.any_change.mean(),
                               1 - sit.has_prior.mean()))
print(sit[sit.has_prior].groupby("season")[
    ["team_change", "qb_change_same_team"]].mean().round(3))
print("vacated: league mean %.3f  by-season:" % vacated.vacated_share.mean())
print(vacated.groupby("season").vacated_share.mean().round(3))
print("\n== B1 ==")
print(pd.DataFrame(b1_rows).round(4).to_string())
print("\n== B2 ==  n=%d" % len(d))
print(m2.summary().tables[1])
print("\n== B3 ==  n=%d  (fit %d / holdout %d)" % (len(mp), fit_m.sum(),
                                                   hold_m.sum()))
print(edge.round(4).to_string())
print("regime counts in pair panel:")
print(pairs.regime.value_counts())

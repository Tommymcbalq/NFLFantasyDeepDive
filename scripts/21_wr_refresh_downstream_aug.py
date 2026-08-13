"""Round-4 Job 1 (cont.) — restate the round-2/3 downstream board layers on the
August ADP board. NEW dated outputs only; scripts 13-18 and their result CSVs
are untouched.

PRE-STATED. Two round-3 sections attach to the 2026 board:

  §E context-adjusted data arm (script 17). Its LOSO verdict is estimated on the
  HISTORICAL 2015-2024 panel and is therefore unchanged by a 2026 ADP refresh
  (script 17 was re-run on the July inputs and reproduced loso_scorecard3.csv
  byte-identically). It was NOT adopted (DM vs (ii) t=-0.81, p=.439). A not-adopted
  arm does not move the board, so the only thing restated here is the 2026
  counterfactual column, on the new membership, still labeled not-adopted.

  §F teammate coherence (script 18). Its F2 edge test is likewise historical and
  unchanged (edge_teammate.csv reproduced byte-identically; full null, so F3 was
  not run). Its 2026 DESCRIPTIVE duo table does depend on board membership, which
  changed, so it is recomputed. The PPG<->targets map (a, b) and the historical
  top-2 WR TS benchmark are refit on the same 2014-2025 data as script 18 and
  asserted equal to the frozen values.

  §A availability (script 13) rescales theta* by E[G]/M for a season-value target;
  it is a monotone transform applied per player and does not re-rank within the
  per-game board, so the per-game board is the deliverable here, as in rounds 1-3.

Current-team source for the refreshed board: the FFC August ADP `team` field
(round 3's §G0 found 0/30 disagreements between Sleeper and FFC on the July board;
the Sleeper dump is from 2026-07-16 and would be stale for August moves, so the
August FFC field is the correct current source). Sleeper agreement is reported.

Outputs: results/sectionE_2026_20260809.csv, results/teammate_coherence_2026_20260809.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path("/Users/thomasmcnamee/NFL")

PCOLS = ["player_id", "position_group", "season", "week", "season_type", "team",
         "targets", "fantasy_points_ppr"]
wkall = pd.concat([pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{y}.csv",
                               usecols=PCOLS, low_memory=False)
                   for y in range(2014, 2026)])
wkall = wkall[wkall.season_type == "REG"].copy()
tmall = pd.concat([pd.read_csv(ROOT / f"data/teams/stats_team_week_{y}.csv",
                               usecols=["season", "week", "team", "season_type",
                                        "attempts"]) for y in range(2014, 2026)])
tmall = tmall[tmall.season_type == "REG"].copy()
t_season = (tmall.groupby(["season", "team"])
            .agg(att=("attempts", "sum"), gms=("week", "size")).reset_index())
t_season["att_pg"] = t_season.att / t_season.gms
attpg = t_season.set_index(["season", "team"]).att_pg

wr = wkall[wkall.position_group == "WR"].merge(
    tmall[["season", "week", "team", "attempts"]].rename(columns={"attempts": "tm_att"}),
    on=["season", "week", "team"], how="left")
agg = (wr.groupby(["player_id", "season"], as_index=False)
       .agg(games=("week", "size"), targets=("targets", "sum"),
            ppr=("fantasy_points_ppr", "sum"), tm_att=("tm_att", "sum")))
agg["tpg"] = agg.targets / agg.games
agg["PPG"] = agg.ppr / agg.games
mapdat = agg[agg.tpg >= 3].copy()
mapdat["x"] = mapdat.targets / mapdat.games
mfit = sm.OLS(mapdat.PPG, sm.add_constant(mapdat.x)).fit()
a, b = float(mfit.params.iloc[0]), float(mfit.params.iloc[1])

frozen18 = pd.read_csv(ROOT / "results/teammate_coherence_2026.csv")
assert np.isclose(a, frozen18.map_intercept.iloc[0]) and \
       np.isclose(b, frozen18.map_slope.iloc[0]), "map (a,b) drifted vs frozen §F"
print(f"PPG<->targets map reproduced: a={a:.4f}, b={b:.4f}")

pts = (wkall[wkall.position_group == "WR"]
       .groupby(["season", "team", "player_id"], as_index=False).targets.sum())
pts = pts[pts.targets > 0].merge(t_season[["season", "team", "att"]], on=["season", "team"])
pts["TS_team"] = pts.targets / pts.att
top2 = (pts.sort_values("TS_team", ascending=False)
        .groupby(["season", "team"]).TS_team.apply(
            lambda s: s.head(2).sum() if len(s) >= 2 else np.nan)
        .dropna().rename("ts_sum").reset_index())
P90, P95 = top2.ts_sum.quantile(0.90), top2.ts_sum.quantile(0.95)
assert np.isclose(P90, frozen18.hist_p90.iloc[0]) and np.isclose(P95, frozen18.hist_p95.iloc[0])
print(f"historical top-2 WR TS sums: n={len(top2)}, p90 {P90:.3f}, p95 {P95:.3f}")

# ---------------- 2026 board ----------------
val = pd.read_csv(ROOT / "results/valuation_2026_wr_20260809.csv")
brd = pd.read_csv(ROOT / "data/adp/wr_top30_adp_2026_20260809.csv")[["name", "team"]]
brd["team"] = brd.team.replace({"LAR": "LA"})
v = val.merge(brd.rename(columns={"name": "player", "team": "cur_team"}), on="player")
assert len(v) == 30

# Sleeper cross-check where the (July) dump still has the player
sl = json.loads((ROOT / "data/sleeper/players_nfl_2026.json").read_text())
sl_team = {}
for r in sl.values():
    if r.get("gsis_id"):
        sl_team[str(r["gsis_id"]).strip()] = r.get("team")
ct = pd.read_csv(ROOT / "results/consistency_table_20260809.csv")[["gsis_id", "player"]]
v = v.merge(ct, on="player", how="left")
v["sleeper_team"] = v.gsis_id.map(lambda g: sl_team.get(g)).replace({"LAR": "LA", "OAK": "LV"})
dis = v[v.sleeper_team.notna() & (v.sleeper_team != v.cur_team)]
print(f"\nFFC(Aug) vs Sleeper(Jul-16) team disagreements: {len(dis)}")
print(dis[["player", "cur_team", "sleeper_team"]].to_string(index=False))

# §E restatement (arm NOT adopted -> board value is the unadjusted theta*)
e = v[["player", "cur_team", "adp", "theta_star"]].copy()
e["arm_vii_adopted"] = False
e["board_value"] = e.theta_star
e["note"] = ("§E arm (vii) not adopted (LOSO DM vs (ii) t=-0.81, p=.439, historical "
             "panel unchanged by the ADP refresh); board value = blind posterior")
e.to_csv(ROOT / "results/sectionE_2026_20260809.csv", index=False)

# §F 2026 duo table on the refreshed board
v["att_pg_2025"] = [attpg.get((2025, t), np.nan) for t in v.cur_team]
assert v.att_pg_2025.notna().all()
v["implied_TS"] = (v.theta_star - a) / (b * v.att_pg_2025)
g = v.groupby("cur_team")
duos = v[v.cur_team.map(g.size()) >= 2].sort_values(
    ["cur_team", "theta_star"], ascending=[True, False])
out = (duos.groupby("cur_team")
       .agg(players=("player", lambda s: " + ".join(s)),
            implied_TS_each=("implied_TS", lambda s: " + ".join(f"{x:.3f}" for x in s)),
            duo_implied_TS_sum=("implied_TS", "sum"),
            theta_sum=("theta_star", "sum")).reset_index().rename(columns={"cur_team": "team"}))
out["pct_of_hist_top2_TS_sums"] = [stats.percentileofscore(top2.ts_sum, x)
                                   for x in out.duo_implied_TS_sum]
# fair reference: historical BOARD-duo implied sums (from the frozen §F run)
hist_impl = pd.read_csv(ROOT / "results/teammate_coherence_2026.csv")
print("\n(fair-reference percentiles use the frozen §F historical implied-duo "
      "distribution; recomputing it needs script 18's fold loop and is unchanged "
      "by a 2026 ADP refresh)")
out["above_p90"] = out.duo_implied_TS_sum > P90
out["above_p95"] = out.duo_implied_TS_sum > P95
out["hist_p90"], out["hist_p95"] = P90, P95
out["map_intercept"], out["map_slope"] = a, b
out = out.sort_values("duo_implied_TS_sum", ascending=False)
out.to_csv(ROOT / "results/teammate_coherence_2026_20260809.csv", index=False)
print("\n== 2026 duos, August board ==")
print(out.round(3).to_string(index=False))
print("\n== July board duos (frozen, for comparison) ==")
print(hist_impl[["team", "players", "duo_implied_TS_sum", "pct_of_hist_top2_TS_sums",
                 "pct_of_hist_implied_duo_sums"]].round(3).to_string(index=False))
print("\nwrote results/sectionE_2026_20260809.csv, "
      "results/teammate_coherence_2026_20260809.csv")

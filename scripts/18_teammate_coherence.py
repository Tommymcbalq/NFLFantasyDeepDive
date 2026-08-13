"""EDA round 3 — §F teammate coherence (EDA_PLAN3.md).

Operational details fixed in this docstring BEFORE running (plan leaves measurement
mechanics open; choices follow round-1/2 precedent):

F1 measurement
  - PPG <-> (TS, team attempts) map: within fold Y, simple OLS
        PPG ~ a + b * (TS * team_att_pg)
    fit on gated WR-seasons (tpg >= 3, 2014-2025, the §B aggregation: sum/sum TS
    with team weekly joins) of seasons != Y; for the 2026 board, all seasons.
    x = expected targets/game; the inversion maps a valuation theta* to the target
    share it implies given the entering team's pass volume.
  - implied_TS(row, fold Y) = (theta*_row - a_{-Y}) / (b_{-Y} * att_pg(team_now, Y-1)),
    att_pg = entering team's PRIOR-season attempts per game (preseason-knowable).
    For 2026: Sleeper current team, 2025 attempts per game.
  - theta* for historical board rows: recomputed with script 10's fold machinery for
    ALL 30 rows per year (loso_predictions.csv covers in_fit rows only; teammate
    structure must not be conditioned on the teammate's realized games). In_fit rows
    are asserted equal to the frozen loso_predictions.csv values.
  - Historical benchmark: realized top-2 WR TS sums, all (team, season) 2014-2025
    with >= 2 targeted WRs: player-team-season WR targets / team season attempts,
    sum of the two largest. Duos flagged above its p90 (descriptive) / p95 (F3 cap).

F2 edge test (binding protocol, family EXACTLY as pre-registered):
  - Rows: market_prior.csv in_fit, 2015-2024; R = resid_iso (full-fit residual,
    B3 precedent).
  - teammate_on_board = 1{another top-30 board player, same year, same team_now}
    (team_now from edge_panel; board membership regardless of in_fit).
  - duo_sum = sum of fold-honest implied TS over board players on (year, team_now)
    — own implied TS when solo (the natural extension that keeps the three-term
    family non-degenerate); centered at the in_fit sample mean.
  - interaction = teammate_on_board * duo_sum_centered.
  - OLS with season-clustered SEs (10 clusters, use_t -> t(9)); BH-FDR q = 0.10
    over the 3 clustered p's; temporal holdout refit 2015-2022 -> MSE on 2023-2024
    vs zero prediction. Survival requires BOTH (a survivor = unpriced infeasibility).
F3 (arm viii) runs ONLY if F2 produces a final survivor — decision rule fixed in the
  plan. If F2 is null, F3 is NOT run and that is stated.

Outputs: results/teammate_coherence_2026.csv (2026 duo sums + historical percentile),
         results/edge_teammate.csv (F2 table), console diagnostics for notes.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.isotonic import IsotonicRegression

ROOT = "/Users/thomasmcnamee/NFL"
YEARS = list(range(2015, 2025))

# ---------------- weekly / team data ----------------
PCOLS = ["player_id", "position_group", "season", "week", "season_type", "team",
         "targets", "fantasy_points_ppr"]
wkall = pd.concat([pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
                               usecols=PCOLS, low_memory=False)
                   for y in range(2014, 2026)])
wkall = wkall[wkall.season_type == "REG"].copy()
tmall = pd.concat([pd.read_csv(f"{ROOT}/data/teams/stats_team_week_{y}.csv",
                               usecols=["season", "week", "team", "season_type",
                                        "attempts"]) for y in range(2014, 2026)])
tmall = tmall[tmall.season_type == "REG"].copy()
t_season = (tmall.groupby(["season", "team"])
            .agg(att=("attempts", "sum"), gms=("week", "size")).reset_index())
t_season["att_pg"] = t_season.att / t_season.gms
attpg = t_season.set_index(["season", "team"]).att_pg

# gated WR-season panel for the PPG ~ TS*att_pg map
wr = wkall[wkall.position_group == "WR"].merge(
    tmall[["season", "week", "team", "attempts"]].rename(columns={"attempts": "tm_att"}),
    on=["season", "week", "team"], how="left")
agg = (wr.groupby(["player_id", "season"], as_index=False)
       .agg(games=("week", "size"), targets=("targets", "sum"),
            ppr=("fantasy_points_ppr", "sum"), tm_att=("tm_att", "sum")))
agg["tpg"] = agg.targets / agg.games
agg["TS"] = agg.targets / agg.tm_att
agg["PPG"] = agg.ppr / agg.games
agg["att_pg_own"] = agg.tm_att / agg.games       # player's weeks team attempts/gm
mapdat = agg[agg.tpg >= 3].copy()
mapdat["x"] = mapdat.TS * mapdat.att_pg_own      # = targets per game

def fit_map(excl=None):
    d = mapdat if excl is None else mapdat[mapdat.season != excl]
    X = sm.add_constant(d.x)
    m = sm.OLS(d.PPG, X).fit()
    return float(m.params.iloc[0]), float(m.params.iloc[1])

# ---------------- historical top-2 WR TS sums (2014-2025) ----------------
pts = (wkall[wkall.position_group == "WR"]
       .groupby(["season", "team", "player_id"], as_index=False).targets.sum())
pts = pts[pts.targets > 0].merge(t_season[["season", "team", "att"]],
                                 on=["season", "team"])
pts["TS_team"] = pts.targets / pts.att
top2 = (pts.sort_values("TS_team", ascending=False)
        .groupby(["season", "team"]).TS_team.apply(lambda s: s.head(2).sum()
                                                   if len(s) >= 2 else np.nan)
        .dropna().rename("ts_sum").reset_index())
P90, P95 = top2.ts_sum.quantile(0.90), top2.ts_sum.quantile(0.95)
print(f"historical top-2 WR TS sums: n={len(top2)}, mean {top2.ts_sum.mean():.3f}, "
      f"p50 {top2.ts_sum.median():.3f}, p90 {P90:.3f}, p95 {P95:.3f}, "
      f"max {top2.ts_sum.max():.3f}")

# ---------------- fold theta* for ALL board rows (script 10 machinery) ----------------
panel = pd.read_csv(f"{ROOT}/results/edge_panel.csv")
wk_frames = []
for y in range(2014, 2026):
    df = pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
                     usecols=["player_id", "position", "season", "season_type",
                              "targets", "fantasy_points_ppr"], low_memory=False)
    wk_frames.append(df[(df.season_type == "REG") & (df.targets > 1)])
wk = pd.concat(wk_frames, ignore_index=True)
sm_all = (wk[wk.player_id.isin(panel.gsis_id.unique())]
          .groupby(["player_id", "season"])
          .agg(ybar=("fantasy_points_ppr", "mean")).reset_index())
wrv = wk[wk.position == "WR"].copy()
ps = (wrv.groupby(["player_id", "season"])
      .agg(mean_tgt=("targets", "mean"), mu_ps=("fantasy_points_ppr", "mean"))
      .reset_index())
wrv = wrv.merge(ps[ps.mean_tgt >= 3.0], on=["player_id", "season"], how="inner")
meta = pd.read_csv(f"{ROOT}/data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "rookie_season"]).dropna()
wrv = wrv.merge(meta.rename(columns={"gsis_id": "player_id"}), on="player_id",
                how="left").dropna(subset=["rookie_season"])
wrv["e2"] = (wrv.fantasy_points_ppr - wrv.mu_ps) ** 2
wrv["exp"] = wrv.season - wrv.rookie_season
wrv["tier"] = np.select([wrv.exp == 0, wrv.exp == 1], ["rookie", "soph"], "vet")

def mu_neff_before(gsis, Y):
    h = sm_all[(sm_all.player_id == gsis) & (sm_all.season < Y)]
    if len(h) == 0:
        return np.nan, 0.0
    S = h.season.max()
    w = 2.0 ** (-(S - h.season.values) / 1.0)
    return float((w * h.ybar.values).sum() / w.sum()), float(w.sum() ** 2 / (w ** 2).sum())

rows = []
for Y in YEARS:
    tr = panel[(panel.year != Y) & panel.in_fit].copy()
    ev = panel[panel.year == Y].copy()                      # ALL board rows
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(np.log(tr.adp.values), tr.ppg.values)
    tr["r"] = tr.ppg - iso.predict(np.log(tr.adp.values))
    ev["m_hat"] = iso.predict(np.log(ev.adp.values))
    tau2 = tr.groupby("tier").r.var(ddof=1)
    ev["tau2"] = ev.tier.map(tau2).fillna(tr.r.var(ddof=1))
    ev["sig2"] = ev.tier.map(wrv[wrv.season != Y].groupby("tier").e2.mean())
    mn = ev.gsis_id.map(lambda g: mu_neff_before(g, Y))
    ev["mu_hat"] = [t[0] for t in mn]
    ev["n_eff"] = [t[1] for t in mn]
    no_prior = ev.n_eff == 0
    with np.errstate(divide="ignore"):
        V = ev.sig2 / ev.n_eff
    B = np.where(no_prior, 1.0, V / (V + ev.tau2))
    ev["theta_star"] = np.where(no_prior, ev.m_hat,
                                (1 - B) * ev.mu_hat.fillna(0) + B * ev.m_hat)
    a, b = fit_map(excl=Y)
    ev["att_pg_prior"] = [attpg.get((Y - 1, t), np.nan) for t in ev.team_now]
    ev["implied_TS"] = (ev.theta_star - a) / (b * ev.att_pg_prior)
    ev["map_a"], ev["map_b"] = a, b
    rows.append(ev[["year", "name", "gsis_id", "adp", "tier", "in_fit", "team_now",
                    "games", "ppg", "theta_star", "att_pg_prior", "implied_TS"]])
brd = pd.concat(rows, ignore_index=True)

# consistency check vs frozen round-1 predictions (in_fit rows)
frozen = pd.read_csv(f"{ROOT}/results/loso_predictions.csv")[
    ["year", "gsis_id", "theta_star"]].rename(columns={"theta_star": "theta_frozen"})
chk = brd[brd.in_fit].merge(frozen, on=["year", "gsis_id"])
assert np.allclose(chk.theta_star, chk.theta_frozen), "theta* mismatch vs frozen file"
print(f"theta* reproduction on in_fit rows: max |diff| = "
      f"{(chk.theta_star - chk.theta_frozen).abs().max():.2e} (n={len(chk)})")
print(f"rows with missing team_now/att_pg: {brd.implied_TS.isna().sum()} of {len(brd)}")

# duo structure per (year, team_now)
grp = brd.dropna(subset=["team_now"]).groupby(["year", "team_now"])
duo_sum = grp.implied_TS.sum().rename("duo_sum")
duo_n = grp.size().rename("n_board")
brd = brd.merge(duo_sum, on=["year", "team_now"], how="left")
brd = brd.merge(duo_n, on=["year", "team_now"], how="left")
brd["teammate"] = (brd.n_board >= 2).astype(float)
hist_duos = (brd[brd.n_board >= 2]
             .groupby(["year", "team_now"])
             .agg(players=("name", lambda s: " + ".join(s)),
                  duo_sum=("duo_sum", "first")).reset_index())
hist_duos["pct_hist"] = [stats.percentileofscore(top2.ts_sum, v) for v in hist_duos.duo_sum]
print(f"\nhistorical board duos (n={len(hist_duos)}), implied-TS sums:")
print(hist_duos.sort_values("duo_sum", ascending=False).round(3).to_string(index=False))
print(f"share of historical board duos above realized p90: "
      f"{(hist_duos.duo_sum > P90).mean():.3f}")

# ---------------- F2 ----------------
mp = pd.read_csv(f"{ROOT}/results/market_prior.csv")
mp = mp[mp.in_fit].copy()
mp = mp.merge(brd[["year", "gsis_id", "teammate", "duo_sum", "implied_TS"]],
              on=["year", "gsis_id"], how="left")
print(f"\nF2 rows: {len(mp)}; missing duo_sum (no team_now): "
      f"{mp.duo_sum.isna().sum()} -> teammate=0, duo_sum=own implied TS unavailable,"
      f" centered term 0")
mp["teammate"] = mp.teammate.fillna(0)
dbar = mp.duo_sum.mean()
mp["duo_c"] = (mp.duo_sum - dbar).fillna(0)
mp["tm_x_duo"] = mp.teammate * mp.duo_c
FAM = ["teammate", "duo_c", "tm_x_duo"]
Xd = sm.add_constant(mp[FAM])
m3 = sm.OLS(mp.resid_iso, Xd).fit(cov_type="cluster",
                                  cov_kwds={"groups": mp.year}, use_t=True)
pv = m3.pvalues[FAM].sort_values()
bh = pv.values <= 0.10 * np.arange(1, 4) / 3
cut = np.max(np.where(bh)[0]) if bh.any() else -1
fdr = {t: (i <= cut) for i, t in enumerate(pv.index)}
fit_m, hold_m = mp.year <= 2022, mp.year >= 2023
mse0 = float((mp.loc[hold_m, "resid_iso"] ** 2).mean())
hold_res = {}
for label, terms in [("survivors", [t for t in FAM if fdr[t]]), ("full_family", FAM)]:
    if not terms:
        hold_res[label] = np.nan
        continue
    mm = sm.OLS(mp.loc[fit_m, "resid_iso"], sm.add_constant(mp.loc[fit_m, terms])).fit()
    pred = sm.add_constant(mp.loc[hold_m, terms]) @ mm.params
    hold_res[label] = float(((mp.loc[hold_m, "resid_iso"] - pred) ** 2).mean())
edge = pd.DataFrame([dict(term=t, beta=m3.params[t], se_cluster=m3.bse[t],
                          t=m3.tvalues[t], p_cluster=m3.pvalues[t], fdr_survivor=fdr[t],
                          holdout_mse_survivors=hold_res["survivors"],
                          holdout_mse_full=hold_res["full_family"],
                          holdout_mse_zero=mse0,
                          final_survivor=bool(fdr[t] and
                                              (hold_res["survivors"] < mse0)))
                     for t in FAM])
edge.to_csv(f"{ROOT}/results/edge_teammate.csv", index=False)
print("\n== F2 edge test ==")
print(edge.round(4).to_string(index=False))
F3_RUN = bool(edge.final_survivor.any())
print(f"\nF3 constraint arm: {'RUNS (unpriced infeasibility found)' if F3_RUN else 'NOT RUN (F2 null: no final survivor -> the market already prices duo structure; pre-specified decision rule)'}")

# ---------------- 2026 descriptive output ----------------
val = pd.read_csv(f"{ROOT}/results/valuation_2026_blind.csv")
e26 = pd.read_csv(f"{ROOT}/results/sectionE_2026.csv")[["player", "sleeper_team"]]
b26 = val.merge(e26, on="player")
ffc = pd.read_csv(f"{ROOT}/data/adp/wr_top30_adp_2026.csv")[["name", "team"]]
ffc["team"] = ffc.team.replace({"LAR": "LA"})
b26 = b26.merge(ffc.rename(columns={"name": "player", "team": "ffc_team"}), on="player")
assert (b26.sleeper_team == b26.ffc_team).all()   # verified in §G0: 0 disagreements
a, b = fit_map(excl=None)
b26["att_pg_2025"] = [attpg.get((2025, t), np.nan) for t in b26.sleeper_team]
b26["implied_TS"] = (b26.theta_star - a) / (b * b26.att_pg_2025)
g = b26.groupby("sleeper_team")
duos26 = (b26[b26.sleeper_team.map(g.size()) >= 2]
          .sort_values(["sleeper_team", "theta_star"], ascending=[True, False]))
out = (duos26.groupby("sleeper_team")
       .agg(players=("player", lambda s: " + ".join(s)),
            implied_TS_each=("implied_TS", lambda s: " + ".join(f"{v:.3f}" for v in s)),
            duo_implied_TS_sum=("implied_TS", "sum"),
            theta_sum=("theta_star", "sum")).reset_index()
       .rename(columns={"sleeper_team": "team"}))
out["pct_of_hist_top2_TS_sums"] = [stats.percentileofscore(top2.ts_sum, v)
                                   for v in out.duo_implied_TS_sum]
# fair reference (added after the F1 anomaly chase): the implied measurement runs
# ~0.06-0.10 hot vs realized (theta* embeds efficiency; the inversion books it all
# as volume), so also report the percentile within historical board-duo IMPLIED
# sums (pairs only; team-years with 3 board WRs excluded from this reference)
hist_pairs = hist_duos[~hist_duos.players.str.contains(r"\+.*\+")]
out["pct_of_hist_implied_duo_sums"] = [
    stats.percentileofscore(hist_pairs.duo_sum, v) for v in out.duo_implied_TS_sum]
out["above_p90"] = out.duo_implied_TS_sum > P90
out["above_p95"] = out.duo_implied_TS_sum > P95
out["hist_p90"], out["hist_p95"] = P90, P95
out["map_intercept"], out["map_slope"] = a, b
out = out.sort_values("duo_implied_TS_sum", ascending=False)
out.to_csv(f"{ROOT}/results/teammate_coherence_2026.csv", index=False)
print("\n== 2026 duos ==")
print(out.round(3).to_string(index=False))
print("\nindividual implied TS, full board (context: sums to eyeball):")
print(b26.sort_values("implied_TS", ascending=False)[
    ["player", "sleeper_team", "theta_star", "att_pg_2025", "implied_TS"]]
    .round(3).to_string(index=False))

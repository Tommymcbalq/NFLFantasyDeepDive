"""EDA round 3 — §G0 (Sleeper current-team data) + §E (context-adjusted data arm).

Pre-registration: EDA_PLAN3.md (2026-07-16). Operational details fixed here BEFORE
running (this docstring is written first, per project protocol):

§G0
  - Raw Sleeper dump cached at data/sleeper/players_nfl_2026.json (pulled 2026-07-16,
    one GET, never overwritten by this script).
  - gsis mapping: Sleeper records carry `gsis_id` for a subset (values sometimes have
    a leading space — stripped). Primary match on stripped gsis_id; fallback:
    normalized name + position (lowercase, alphanumerics only, Jr/Sr/II/III/IV/V
    suffixes dropped; position_group WR matches Sleeper position WR, etc.).
    Ambiguous fallback matches (>=2 Sleeper records same name+pos) resolved by
    preferring a record with a current team, else active status; still-ambiguous ->
    unmatched, reported. Tertiary rule (added after the first match report showed
    'Josh Palmer'/'Joshua Palmer' and 'Mitch Tinsley'/'Mitchell Tinsley' — active
    stayers counted as departed, one on a board-relevant team; a data-quality fix,
    not a result-driven refit): a still-unmatched player matches a Sleeper record
    iff among Sleeper records with the same (normalized last name, position)
    exactly ONE agrees with the first-name token on its first 3 characters.
    Players caught by this rule are printed. Unmatched reported among (a) 2025 fantasy-relevant WRs
    (position_group WR, REG targets/appearance >= 3 in 2025) and (b) the 30 board WRs.
  - Team codes: Sleeper LAR -> nflverse LA; Sleeper OAK -> LV (stale records);
    FFC LAR -> LA for the cross-check.
  - Mover flag per board player: 2025 primary team (team with most REG weekly rows in
    2025; ties -> team of the latest week) != current Sleeper team. Cross-checked
    against the FFC ADP `team` column; disagreements reported (Sleeper is the
    modeling source, FFC the cross-check, per plan).
  - data/derived/vacated_2026.csv: per nflverse team, share of that team's 2025 REG
    targets (all positions) to players whose current Sleeper team differs (unmatched
    or team-less players count as departed, same convention as the historical table
    where absence from the season-s roster = vacated; unmatched target mass reported
    separately as a bias check). Sanity: league mean vs historical ~0.29.

§E
  - beta_tc, beta_vac refit per LOSO fold on <= Y-1 data with the EXACT round-2 B2
    spec and panel (script 14: pairs s -> s+1, s in 2015-2024, gate tpg >= 3 in s,
    >= 4 REG appearances in s+1; dPPG ~ team_change + qb_change_same_team + z_att_new
    + z_epa_new + vacated_new). "<= Y-1 data" = pairs with season s+1 <= Y-1 (the
    outcome season must precede the held-out board year). Folds with zero training
    pairs (Y = 2015, 2016: the B2 panel starts at s+1 = 2016) get beta = 0, i.e.
    arm (vii) degenerates to arm (ii) there — no data, no adjustment. Point estimates
    only are needed per fold (no inference), so plain OLS.
  - Centering constant = training-sample mean of vacated_new (fold-honest), so a
    non-mover on an average-turnover team is untouched.
  - Eval-row context (preseason-knowable historically): mover = edge_panel
    team_change (team_first in Y != team_last in Y-1; rookies/no-prior = 0, the
    round-1/2 convention); entering-team vacated = vacated_targets.csv joined on
    (team_now, year); missing vacated -> centered term 0.
  - mu_c = mu_hat + beta_tc*mover + beta_vac*(vacated - mean_vac); V, B, m_hat
    unchanged; theta*_c = (1-B)*mu_c + B*m_hat; no-prior rows stay theta = m_hat.
  - Arms (i)/(ii) reproduced with script 10's machinery verbatim and asserted equal
    to the frozen round-1 scorecard RMSEs before (vii) is scored.
  - Adoption rule (pre-specified): DM vs (ii) p < 0.10 AND pooled RMSE improves.
  - 2026 board: beta refit on the FULL B2 panel (all seasons <= 2025), mover from
    §G0 Sleeper flag, vacated from vacated_2026.csv on the Sleeper (current) team;
    results/sectionE_2026.csv written regardless, labeled adopted / not adopted.

Outputs: data/derived/vacated_2026.csv, results/loso_scorecard3.csv,
         results/loso_predictions3.csv, results/sectionE_2026.csv,
         console diagnostics for the notes.
"""
import json
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.isotonic import IsotonicRegression

ROOT = "/Users/thomasmcnamee/NFL"
YEARS = list(range(2015, 2025))
SLEEPER_TEAM_FIX = {"LAR": "LA", "OAK": "LV"}
FFC_TEAM_FIX = {"LAR": "LA"}

# ================================================================ §G0
sl = json.load(open(f"{ROOT}/data/sleeper/players_nfl_2026.json"))
srows = []
for sid, p in sl.items():
    team = p.get("team")
    srows.append(dict(sleeper_id=sid, name=p.get("full_name"),
                      pos=p.get("position"),
                      team=SLEEPER_TEAM_FIX.get(team, team),
                      gsis=(p.get("gsis_id") or "").strip() or None,
                      active=bool(p.get("active")), status=p.get("status")))
sdf = pd.DataFrame(srows)

SUFF = re.compile(r"\b(jr|sr|ii|iii|iv|v)$")
def norm(n):
    if not isinstance(n, str):
        return ""
    s = re.sub(r"[^a-z0-9 ]", "", n.lower()).strip()
    return SUFF.sub("", s).replace(" ", "")

sdf["nname"] = sdf.name.map(norm)

# 2025 weekly (REG)
wk25 = pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_2025.csv",
                   low_memory=False,
                   usecols=["player_id", "player_display_name", "position_group",
                            "season_type", "team", "week", "targets"])
wk25 = wk25[wk25.season_type == "REG"].copy()

# gsis -> sleeper map for every 2025 participant
players25 = (wk25.groupby("player_id")
             .agg(name=("player_display_name", "first"),
                  pos=("position_group", "first"),
                  tot_targets=("targets", "sum"), n_wk=("week", "size"))
             .reset_index())
players25["nname"] = players25.name.map(norm)

by_gsis = sdf.dropna(subset=["gsis"]).drop_duplicates("gsis").set_index("gsis")
m_gsis = players25.player_id.map(by_gsis.team)
matched_via_gsis = players25.player_id.isin(by_gsis.index)

# fallback: name+pos
cand = sdf.groupby(["nname", "pos"])
def name_match(row):
    key = (row.nname, row.pos)
    try:
        g = cand.get_group(key)
    except KeyError:
        return None, "unmatched"
    if len(g) == 1:
        return g.iloc[0].team, "name"
    g2 = g[g.team.notna()]
    if len(g2) == 1:
        return g2.iloc[0].team, "name_team_pref"
    g3 = g[g.active]
    if len(g3) == 1:
        return g3.iloc[0].team, "name_active_pref"
    return None, "ambiguous"

def name_tokens(n):
    s = re.sub(r"[^a-z0-9 ]", "", (n or "").lower()).split()
    while s and s[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        s.pop()
    return s

sdf["last"] = sdf.name.map(lambda n: name_tokens(n)[-1] if name_tokens(n) else "")
sdf["first"] = sdf.name.map(lambda n: name_tokens(n)[0] if name_tokens(n) else "")
by_last_pos = {k: g for k, g in sdf.groupby(["last", "pos"])}

def tertiary_match(row):
    toks = name_tokens(row["name"])
    if not toks or len(toks[0]) < 3:
        return None
    g = by_last_pos.get((toks[-1], row.pos))
    if g is None:
        return None
    hit = g[g["first"].str[:3] == toks[0][:3]]
    return hit.iloc[0] if len(hit) == 1 else None

teams, how = [], []
for _, r in players25.iterrows():
    if matched_via_gsis[_]:
        teams.append(by_gsis.team.get(r.player_id)); how.append("gsis")
    else:
        t, h = name_match(r)
        if h in ("unmatched", "ambiguous"):
            c = tertiary_match(r)
            if c is not None:
                t, h = c.team, "lastname_prefix"
                print(f"  tertiary match: {r['name']} ({r.pos}) -> "
                      f"{c['name']} [{c.team}]")
        teams.append(t); how.append(h)
players25["sleeper_team"] = teams
players25["match_how"] = how
players25["matched"] = ~players25.match_how.isin(["unmatched", "ambiguous"])

# unmatched report: fantasy-relevant WRs 2025
frel = players25[(players25.pos == "WR")
                 & (players25.tot_targets / players25.n_wk >= 3)]
um = frel[~frel.matched]
print("== G0 match report ==")
print(f"2025 participants: {len(players25)}; matched {players25.matched.mean():.3f} "
      f"(gsis {(players25.match_how == 'gsis').mean():.3f})")
print(f"fantasy-relevant 2025 WRs (tpg>=3): {len(frel)}; unmatched: {len(um)}")
if len(um):
    print(um[["name", "tot_targets", "match_how"]].to_string(index=False))

# board players
board = pd.read_csv(f"{ROOT}/data/adp/wr_top30_adp_2026.csv")[
    ["wr_adp_rank", "name", "team", "adp"]].rename(
    columns={"wr_adp_rank": "adp_rank", "team": "ffc_team"})
board["ffc_team"] = board.ffc_team.replace(FFC_TEAM_FIX)
ct = pd.read_csv(f"{ROOT}/results/consistency_table.csv")[["gsis_id", "player"]]
board = board.merge(ct, left_on="name", right_on="player", how="left")
assert board.gsis_id.notna().all()

# 2025 primary team: most REG weekly rows; tie -> latest week's team
def primary_team(g):
    cnt = g.groupby("team").agg(n=("week", "size"), last=("week", "max"))
    return cnt.sort_values(["n", "last"], ascending=False).index[0]
prim = (wk25[wk25.player_id.isin(board.gsis_id)]
        .groupby("player_id").apply(primary_team, include_groups=False)
        .rename("team_2025"))
board = board.merge(prim, left_on="gsis_id", right_index=True, how="left")

bmatch = players25.set_index("player_id")[["sleeper_team", "match_how", "matched"]]
board = board.merge(bmatch, left_on="gsis_id", right_index=True, how="left")
n_bum = int((~board.matched.fillna(False)).sum())
print(f"board players unmatched to Sleeper: {n_bum}")
if n_bum:
    print(board[~board.matched.fillna(False)][["name", "match_how"]].to_string(index=False))
board["mover_2026"] = (board.team_2025 != board.sleeper_team).astype(float)
dis = board[board.sleeper_team != board.ffc_team]
print(f"\nSleeper-vs-FFC team disagreements: {len(dis)}")
if len(dis):
    print(dis[["name", "team_2025", "sleeper_team", "ffc_team"]].to_string(index=False))
print("\nboard movers (Sleeper 2026 team != 2025 primary team):")
print(board[board.mover_2026 == 1][["name", "team_2025", "sleeper_team", "ffc_team"]]
      .to_string(index=False))

# vacated 2026 per team
ptt = (wk25.groupby(["team", "player_id"], as_index=False).targets.sum()
       .merge(players25[["player_id", "sleeper_team", "matched"]], on="player_id"))
rows = []
for team, g in ptt.groupby("team"):
    tot = g.targets.sum()
    vac = g.loc[g.sleeper_team != team, "targets"].sum()          # None != team -> vacated
    unm = g.loc[~g.matched, "targets"].sum()
    rows.append(dict(team=team, prior_targets=tot, vacated_targets=vac,
                     vacated_share=vac / tot, unmatched_targets=unm,
                     unmatched_share=unm / tot))
vac26 = pd.DataFrame(rows).sort_values("team")
vac26.to_csv(f"{ROOT}/data/derived/vacated_2026.csv", index=False)
hist_vac = pd.read_csv(f"{ROOT}/data/derived/vacated_targets.csv")
print(f"\n2026 league-mean vacated share: {vac26.vacated_share.mean():.3f} "
      f"(historical 2015-2025 mean {hist_vac.vacated_share.mean():.3f}); "
      f"unmatched target mass mean {vac26.unmatched_share.mean():.3f}")
print(vac26.round(3).to_string(index=False))

# ================================================================ §E — B2 panel
# (verbatim rebuild of script 14's pair panel)
PCOLS = ["player_id", "player_display_name", "position_group", "season", "week",
         "season_type", "team", "attempts", "targets", "receiving_air_yards",
         "fantasy_points_ppr"]
wkall = pd.concat([pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
                               usecols=PCOLS) for y in range(2014, 2026)])
wkall = wkall[wkall.season_type == "REG"].copy()
TCOLS = ["season", "week", "team", "season_type", "attempts", "passing_air_yards",
         "passing_epa"]
tmall = pd.concat([pd.read_csv(f"{ROOT}/data/teams/stats_team_week_{y}.csv",
                               usecols=TCOLS) for y in range(2014, 2026)])
tmall = tmall[tmall.season_type == "REG"].copy()

wr = wkall[wkall.position_group == "WR"].copy()
wr = wr.merge(tmall[["season", "week", "team", "attempts"]]
              .rename(columns={"attempts": "tm_att"}),
              on=["season", "week", "team"], how="left")
agg = (wr.groupby(["player_id", "season"], as_index=False)
         .agg(games=("week", "size"), targets=("targets", "sum"),
              ppr=("fantasy_points_ppr", "sum")))
agg["tpg"] = agg.targets / agg.games
agg["PPG"] = agg.ppr / agg.games
fl = (wr.sort_values("week").groupby(["player_id", "season"])
        .team.agg(team_first="first", team_last="last").reset_index())
agg = agg.merge(fl, on=["player_id", "season"])

vacated = pd.read_csv(f"{ROOT}/data/derived/vacated_targets.csv")
qb = pd.read_csv(f"{ROOT}/data/derived/qb_by_team_season.csv")

# change flags for season s+1 WITHOUT any s+1 gate (script 14 exactly: sit's gate
# applies to season s only; joining the gated situation_change.csv here would
# condition the panel on the outcome season — bug caught in replication check)
prev = agg[["player_id", "season", "team_last"]].copy()
prev["season"] += 1
prev = prev.rename(columns={"team_last": "team_last_prev"})
scf = agg.merge(prev, on=["player_id", "season"], how="left")
qbk = qb.set_index(["season", "team"]).qb_gsis_id
scf["qb_now"] = [qbk.get((s, t), np.nan) for s, t in zip(scf.season, scf.team_first)]
scf["qb_prev"] = [qbk.get((s - 1, t), np.nan)
                  for s, t in zip(scf.season, scf.team_last_prev)]
has_prior = scf.team_last_prev.notna()
scf["team_change"] = np.where(has_prior,
                              (scf.team_first != scf.team_last_prev).astype(float),
                              np.nan)
scf["qb_change_same_team"] = np.where(
    has_prior, ((scf.team_change == 0) & (scf.qb_now != scf.qb_prev)).astype(float),
    np.nan)

nxt = scf[["player_id", "season", "games", "PPG", "team_first",
           "team_change", "qb_change_same_team"]].copy()
nxt.columns = ["player_id", "season_next", "games1", "PPG1", "team_new",
               "team_change", "qb_change_same_team"]
nxt["season"] = nxt.season_next - 1
pairs = (agg[(agg.season >= 2015) & (agg.season <= 2024) & (agg.tpg >= 3)]
         [["player_id", "season", "PPG"]]
         .merge(nxt.drop(columns="season_next"), on=["player_id", "season"]))
pairs = pairs[pairs.games1 >= 4].copy()
tse = (tmall.groupby(["season", "team"], as_index=False)
       .agg(att=("attempts", "sum"), epa=("passing_epa", "mean")))
for c in ["att", "epa"]:
    tse[f"z_{c}"] = tse.groupby("season")[c].transform(
        lambda x: (x - x.mean()) / x.std())
pairs = pairs.merge(tse[["season", "team", "z_att", "z_epa"]]
                    .rename(columns={"team": "team_new"}),
                    on=["season", "team_new"], how="left")
pairs = pairs.merge(vacated[["team", "season", "vacated_share"]]
                    .rename(columns={"team": "team_new"})
                    .assign(season=lambda d: d.season - 1),
                    on=["season", "team_new"], how="left")
pairs["dPPG"] = pairs.PPG1 - pairs.PPG
b2p = pairs.dropna(subset=["dPPG", "team_change", "qb_change_same_team",
                           "z_att", "z_epa", "vacated_share"]).copy()
b2p["season_next"] = b2p.season + 1

# replication check vs round-2 B2 (full sample)
m_full = smf.ols("dPPG ~ team_change + qb_change_same_team + z_att + z_epa"
                 " + vacated_share", data=b2p).fit()
print(f"\n== B2 panel rebuilt: n={len(b2p)} (round-2 n=958); "
      f"beta_tc={m_full.params['team_change']:+.4f} (round-2 -1.00), "
      f"beta_vac={m_full.params['vacated_share']:+.4f} (round-2 +1.92)")

def fit_b2(sub):
    if len(sub) < 20:
        return 0.0, 0.0, np.nan, len(sub)
    m = smf.ols("dPPG ~ team_change + qb_change_same_team + z_att + z_epa"
                " + vacated_share", data=sub).fit()
    return (float(m.params["team_change"]), float(m.params["vacated_share"]),
            float(sub.vacated_share.mean()), len(sub))

# ================================================================ §E — LOSO
FDR_TERMS = ["rookie", "rook_x_epa"]
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
          .agg(G=("fantasy_points_ppr", "size"),
               ybar=("fantasy_points_ppr", "mean")).reset_index())
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

preds, foldctx = [], []
for Y in YEARS:
    tr = panel[(panel.year != Y) & panel.in_fit].copy()
    ev = panel[(panel.year == Y) & panel.in_fit].copy()
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(np.log(tr.adp.values), tr.ppg.values)
    tr["r"] = tr.ppg - iso.predict(np.log(tr.adp.values))
    ev["m_hat"] = iso.predict(np.log(ev.adp.values))
    tau2 = tr.groupby("tier").r.var(ddof=1)
    ev["tau2"] = ev.tier.map(tau2).fillna(tr.r.var(ddof=1))
    sig2 = wrv[wrv.season != Y].groupby("tier").e2.mean()
    ev["sig2"] = ev.tier.map(sig2)
    mn = ev.gsis_id.map(lambda g: mu_neff_before(g, Y))
    ev["mu_hat"] = [t[0] for t in mn]
    ev["n_eff"] = [t[1] for t in mn]
    no_prior = ev.n_eff == 0
    with np.errstate(divide="ignore"):
        V = ev.sig2 / ev.n_eff
    B = np.where(no_prior, 1.0, V / (V + ev.tau2))
    ev["B"] = B
    ev["theta_star"] = np.where(no_prior, ev.m_hat,
                                (1 - B) * ev.mu_hat.fillna(0) + B * ev.m_hat)

    # ---- arm (vii): context-adjusted mu ----
    b_tc, b_vac, vbar, n_tr = fit_b2(b2p[b2p.season_next <= Y - 1])
    ev = ev.merge(vacated[["team", "season", "vacated_share"]]
                  .rename(columns={"team": "team_now", "season": "year",
                                   "vacated_share": "vac_now"}),
                  on=["team_now", "year"], how="left")
    ev["mover"] = ev.team_change.fillna(0)
    vac_c = np.where(ev.vac_now.notna() & np.isfinite(vbar),
                     ev.vac_now - vbar, 0.0)
    ev["mu_c"] = ev.mu_hat + b_tc * ev.mover + b_vac * vac_c
    ev["theta_ctx"] = np.where(no_prior, ev.m_hat,
                               (1 - B) * ev.mu_c.fillna(0) + B * ev.m_hat)
    foldctx.append(dict(Y=Y, n_train_b2=n_tr, beta_tc=b_tc, beta_vac=b_vac,
                        mean_vac_train=vbar,
                        n_movers=int(ev.mover.sum()),
                        mean_abs_adj=float((ev.theta_ctx - ev.theta_star).abs().mean())))
    preds.append(ev[["year", "name", "gsis_id", "adp", "tier", "games", "ppg",
                     "m_hat", "mu_hat", "n_eff", "B", "mover", "vac_now",
                     "theta_star", "theta_ctx"]])

pred = pd.concat(preds, ignore_index=True)
fc = pd.DataFrame(foldctx)
print("\n== fold context diagnostics ==")
print(fc.round(4).to_string(index=False))

# frozen-arm reproduction check
rmse_i = float(np.sqrt(((pred.ppg - pred.m_hat) ** 2).mean()))
rmse_ii = float(np.sqrt(((pred.ppg - pred.theta_star) ** 2).mean()))
assert abs(rmse_i - 3.5636073592015305) < 1e-9, rmse_i
assert abs(rmse_ii - 3.4631159036313717) < 1e-9, rmse_ii
print(f"\nfrozen arms reproduced: (i) RMSE {rmse_i:.10f}, (ii) {rmse_ii:.10f}")

def dm(pred, col_a, col_b):
    """paired DM: >0 means col_b better than col_a"""
    d = (pred.ppg - pred[col_a]) ** 2 - (pred.ppg - pred[col_b]) ** 2
    dyr = d.groupby(pred.year).mean()
    t = float(dyr.mean() / (dyr.std(ddof=1) / np.sqrt(len(dyr))))
    return t, float(2 * stats.t.sf(abs(t), df=len(dyr) - 1))

rows = []
for label, col in [("(i) ADP-only m_hat", "m_hat"),
                   ("(ii) blind theta*", "theta_star"),
                   ("(vii) context-adjusted theta*_c", "theta_ctx")]:
    err = pred.ppg - pred[col]
    rho = pred.groupby("year").apply(
        lambda g: stats.spearmanr(g[col], g.ppg).statistic, include_groups=False)
    ti, pi = dm(pred, "m_hat", col) if col != "m_hat" else (np.nan, np.nan)
    tb, pb = dm(pred, "theta_star", col) if col != "theta_star" else (np.nan, np.nan)
    rows.append(dict(predictor=label, rmse=float(np.sqrt((err ** 2).mean())),
                     mean_spearman=float(rho.mean()),
                     dm_t_vs_adp=ti, dm_p_vs_adp=pi,
                     dm_t_vs_blind=tb, dm_p_vs_blind=pb))
sc = pd.DataFrame(rows)
sc.to_csv(f"{ROOT}/results/loso_scorecard3.csv", index=False)
pred.to_csv(f"{ROOT}/results/loso_predictions3.csv", index=False)
print("\n== loso scorecard 3 ==")
print(sc.round(4).to_string(index=False))

r7 = sc.loc[2]
ADOPT = bool((r7.dm_p_vs_blind < 0.10) and (r7.rmse < sc.loc[1].rmse)
             and (r7.dm_t_vs_blind > 0))
print(f"\nADOPTION (vii): {'ADOPTED' if ADOPT else 'NOT adopted'} "
      f"(rule: DM vs (ii) p<0.10 AND RMSE improves)")

# per-fold loss diff (vii) vs (ii) for the notes
d = (pred.ppg - pred.theta_star) ** 2 - (pred.ppg - pred.theta_ctx) ** 2
print("yearly mean loss diff (vii better >0):",
      d.groupby(pred.year).mean().round(4).to_dict())
mv = pred[pred.mover == 1]
print(f"movers in eval rows: {len(mv)}; their loss diff mean {((mv.ppg-mv.theta_star)**2-(mv.ppg-mv.theta_ctx)**2).mean():+.4f}")

# ================================================================ §E — 2026 board
b_tc, b_vac, vbar, n_tr = fit_b2(b2p)
print(f"\nfull-sample B2 for 2026: beta_tc={b_tc:+.4f}, beta_vac={b_vac:+.4f}, "
      f"mean_vac={vbar:.4f}, n={n_tr}")
blind = pd.read_csv(f"{ROOT}/results/valuation_2026_blind.csv")
b26 = blind.merge(board[["player", "gsis_id", "team_2025", "sleeper_team",
                         "ffc_team", "mover_2026"]], on="player", how="left")
b26 = b26.merge(vac26[["team", "vacated_share"]]
                .rename(columns={"team": "sleeper_team", "vacated_share": "vac_2026"}),
                on="sleeper_team", how="left")
b26["mu_c"] = b26.mu_hat + b_tc * b26.mover_2026 + b_vac * (b26.vac_2026 - vbar)
b26["theta_star_ctx"] = (1 - b26.B) * b26.mu_c + b26.B * b26.m_adp
b26["delta_vs_blind"] = b26.theta_star_ctx - b26.theta_star
b26 = b26.sort_values("theta_star_ctx", ascending=False).reset_index(drop=True)
b26["rank_ctx"] = b26.index + 1
b26["adopted"] = ADOPT
b26["verdict"] = ("ADOPTED per pre-specified rule" if ADOPT else
                  "NOT adopted (pre-specified LOSO rule failed); for the record only")
cols = ["rank_ctx", "player", "adp", "adp_rank", "tier", "team_2025", "sleeper_team",
        "mover_2026", "vac_2026", "mu_hat", "mu_c", "B", "m_adp", "theta_star",
        "theta_star_ctx", "delta_vs_blind", "rank_theta", "adopted", "verdict"]
b26[cols].to_csv(f"{ROOT}/results/sectionE_2026.csv", index=False)
print("\n== sectionE 2026 board (theta*_c) ==")
print(b26[cols].drop(columns=["adopted", "verdict"]).round(3).to_string(index=False))

"""§W1 — feature construction for the projection engine.

Builds, for every row (player, panel year Y) of the §P wide panels, a preseason-knowable
feature vector from seasons strictly before Y plus the preseason-known year-Y team.

Tier A: weekly_raw 1999-2025 + players_meta            -> folds 2015-2024
Tier B: Tier A + data/derived/adv_*.csv, team_context  -> folds 2019-2024

Recency weighting w_s = 2^-(Smax-s)/h, h=1, identical to mu_hat (eq. 43.1), so the
projection and the incumbent see the same temporal window.

Shares use FULL-team-season denominators (player targets / team targets summed over all
team games) -- never the active-games denominator that makes target_share sum to 1.36.

Output: data/derived/w1_features_{WR,RB}.csv
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
ROOT = Path("/Users/thomasmcnamee/NFL")
HL = 1.0
TEAMFIX = {"LAR": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA"}

WCOLS = ["player_id", "position", "season", "week", "season_type", "team",
         "attempts", "completions", "passing_yards", "passing_tds",
         "targets", "receptions", "carries", "receiving_yards", "receiving_tds",
         "receiving_air_yards", "receiving_yards_after_catch", "receiving_epa",
         "rushing_yards", "rushing_tds", "fantasy_points_ppr"]

_f = []
for y in range(1999, 2026):
    d = pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{y}.csv",
                    usecols=lambda c: c in WCOLS, low_memory=False)
    _f.append(d[d.season_type == "REG"])
WK = pd.concat(_f, ignore_index=True)
NUM = ["attempts", "completions", "passing_yards", "passing_tds", "targets",
       "receptions", "carries", "receiving_yards", "receiving_tds",
       "receiving_air_yards", "receiving_yards_after_catch", "receiving_epa",
       "rushing_yards", "rushing_tds", "fantasy_points_ppr"]
for c in NUM:
    WK[c] = WK[c].fillna(0.0)
WK["touches"] = WK.carries + WK.targets

# ------------------------------------------------------------------ team-season env
TM = (WK.groupby(["team", "season"])
      .agg(tm_pass_att=("attempts", "sum"), tm_pass_yds=("passing_yards", "sum"),
           tm_pass_td=("passing_tds", "sum"), tm_car=("carries", "sum"),
           tm_rush_yds=("rushing_yards", "sum"), tm_rush_td=("rushing_tds", "sum"),
           tm_tgt=("targets", "sum"), tm_ay=("receiving_air_yards", "sum")).reset_index())
_g = WK.groupby(["team", "season"]).week.nunique().rename("tm_games").reset_index()
TM = TM.merge(_g, on=["team", "season"])
for c in ["tm_pass_att", "tm_pass_yds", "tm_car", "tm_rush_yds", "tm_tgt", "tm_ay"]:
    TM[c + "_pg"] = TM[c] / TM.tm_games
TM["tm_plays_pg"] = TM.tm_pass_att_pg + TM.tm_car_pg
TM["tm_pass_rate"] = TM.tm_pass_att / (TM.tm_pass_att + TM.tm_car)
TM["tm_td_pg"] = (TM.tm_pass_td + TM.tm_rush_td) / TM.tm_games
ENV = ["tm_pass_att_pg", "tm_tgt_pg", "tm_ay_pg", "tm_car_pg", "tm_plays_pg",
       "tm_pass_rate", "tm_td_pg", "tm_pass_yds_pg"]

# ------------------------------------------------------------------ player-season stats
PS = (WK.groupby(["player_id", "season"])
      .agg(G=("fantasy_points_ppr", "size"), ppr=("fantasy_points_ppr", "sum"),
           tgt=("targets", "sum"), rec=("receptions", "sum"), car=("carries", "sum"),
           tch=("touches", "sum"), ry=("receiving_yards", "sum"),
           rtd=("receiving_tds", "sum"), ay=("receiving_air_yards", "sum"),
           yac=("receiving_yards_after_catch", "sum"),
           repa=("receiving_epa", "sum"), rushy=("rushing_yards", "sum"),
           rushtd=("rushing_tds", "sum")).reset_index())
# team denominators: a player's team-seasons, volume-weighted
PT = (WK.groupby(["player_id", "season", "team"])
      .agg(p_tgt=("targets", "sum"), p_car=("carries", "sum")).reset_index()
      .merge(TM[["team", "season", "tm_tgt", "tm_ay", "tm_car"]], on=["team", "season"],
             how="left"))
PT = PT.groupby(["player_id", "season"], as_index=False)[["tm_tgt", "tm_ay", "tm_car"]].sum()
PS = PS.merge(PT, on=["player_id", "season"], how="left")
E = 1e-9
PS["ppg"] = PS.ppr / PS.G
PS["targets_pg"] = PS.tgt / PS.G
PS["rec_pg"] = PS.rec / PS.G
PS["carries_pg"] = PS.car / PS.G
PS["touches_pg"] = PS.tch / PS.G
PS["rec_yards_pg"] = PS.ry / PS.G
PS["rush_yards_pg"] = PS.rushy / PS.G
PS["air_yards_pg"] = PS.ay / PS.G
PS["target_share_full"] = PS.tgt / (PS.tm_tgt + E)
PS["air_yards_share_full"] = PS.ay / (PS.tm_ay + E)
PS["carry_share_full"] = PS.car / (PS.tm_car + E)
PS["wopr_full"] = 1.5 * PS.target_share_full + 0.7 * PS.air_yards_share_full
PS["adot"] = PS.ay / (PS.tgt + E)
PS["ypr"] = PS.ry / (PS.rec + E)
PS["catch_rate"] = PS.rec / (PS.tgt + E)
PS["racr_w"] = np.clip(PS.ry / np.where(PS.ay.abs() < 1, np.nan, PS.ay), 0, 3)
PS["ypt"] = PS.ry / (PS.tgt + E)
PS["ypc"] = PS.rushy / (PS.car + E)
PS["yac_per_rec"] = PS.yac / (PS.rec + E)
PS["td_per_tgt"] = PS.rtd / (PS.tgt + E)
PS["rec_epa_per_tgt"] = PS.repa / (PS.tgt + E)
# usage index (share x team volume/g), the §S arm-8 covariate
PS["usage_rec"] = PS.target_share_full * (PS.tm_tgt / PS.G.clip(lower=1)) * 0  # filled below
_tv = TM[["team", "season", "tm_tgt_pg", "tm_car_pg"]]
_pv = (WK.groupby(["player_id", "season", "team"])
       .agg(p_tgt=("targets", "sum"), p_car=("carries", "sum")).reset_index()
       .merge(_tv, on=["team", "season"], how="left")
       .merge(TM[["team", "season", "tm_tgt", "tm_car"]], on=["team", "season"], how="left"))
_pv["u_rec"] = (_pv.p_tgt / (_pv.tm_tgt + E)) * _pv.tm_tgt_pg
_pv["u_rush"] = (_pv.p_car / (_pv.tm_car + E)) * _pv.tm_car_pg
_pv = _pv.groupby(["player_id", "season"], as_index=False)[["u_rec", "u_rush"]].sum()
PS = PS.drop(columns=["usage_rec"]).merge(_pv, on=["player_id", "season"], how="left")
PS["u_tot"] = PS.u_rec + PS.u_rush

# last team of a player in a season (max volume)
_lt = (WK.groupby(["player_id", "season", "team"]).fantasy_points_ppr.sum()
       .reset_index().sort_values("fantasy_points_ppr")
       .groupby(["player_id", "season"]).tail(1)[["player_id", "season", "team"]]
       .rename(columns={"team": "team_prior"}))
PS = PS.merge(_lt, on=["player_id", "season"], how="left")

# ------------------------------------------------------------------ meta
META = pd.read_csv(ROOT / "data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "birth_date", "rookie_season", "draft_pick",
                            "draft_round"])
META["birth_date"] = pd.to_datetime(META.birth_date, errors="coerce")

TIERA_HIST = ["ppg", "targets_pg", "rec_pg", "carries_pg", "touches_pg",
              "rec_yards_pg", "rush_yards_pg", "air_yards_pg", "target_share_full",
              "air_yards_share_full", "carry_share_full", "wopr_full",
              "adot", "ypr", "catch_rate", "racr_w", "u_rec", "u_rush", "u_tot"]
# gate-rejected, carried for the declared "ungated" sensitivity only
TIERA_REJECTED = ["ypt", "yac_per_rec", "td_per_tgt", "rec_epa_per_tgt", "ypc"]


def sched(y):
    return 17.0 if y >= 2021 else 16.0


def build(panel_file, pos):
    panel = pd.read_csv(ROOT / panel_file).rename(columns={"pid": "gsis_id"})
    panel["team"] = panel.team.replace(TEAMFIX)
    ps = PS[PS.player_id.isin(panel.gsis_id.unique())]
    byp = {p: g.sort_values("season") for p, g in ps.groupby("player_id")}
    meta = META.set_index("gsis_id")

    rows = []
    for _, r in panel.iterrows():
        g, Y = r.gsis_id, int(r.year)
        rec = {"gsis_id": g, "year": Y}
        h = byp.get(g)
        h = h[h.season < Y] if h is not None else None
        if h is None or len(h) == 0:
            rec["n_prior"] = 0
            rows.append(rec)
            continue
        s = h.season.values.astype(float)
        w = 2.0 ** (-(s.max() - s) / HL)
        rec["n_prior"] = len(h)
        rec["n_eff"] = float(w.sum() ** 2 / (w ** 2).sum())
        rec["mu_hat"] = float((w * h.ppg.values).sum() / w.sum())
        rec["G_last"] = float(h.G.values[-1])
        rec["G_wtd"] = float((w * h.G.values).sum() / w.sum())
        # availability on a scheduled basis
        av = h.G.values / np.array([sched(int(x)) for x in s])
        rec["avail_wtd"] = float((w * av).sum() / w.sum())
        rec["avail_last"] = float(av[-1])
        rec["avail_career"] = float(av.mean())
        rec["ppsw_hat"] = float((w * (h.ppg.values * av)).sum() / w.sum())
        rec["gap_since_last"] = float(Y - s.max())
        rec["d_ppg"] = float(h.ppg.values[-1] - h.ppg.values[-2]) if len(h) >= 2 else 0.0
        for c in TIERA_HIST + TIERA_REJECTED:
            v = h[c].values.astype(float)
            m = np.isfinite(v)
            rec["h_" + c] = float((w[m] * v[m]).sum() / w[m].sum()) if m.any() else np.nan
            rec["h_" + c + "_last"] = float(v[-1]) if np.isfinite(v[-1]) else np.nan
        # environment: the year-Y team's PRIOR-season offence
        tprior = h.team_prior.values[-1]
        rec["team_change"] = float(r.team != tprior) if isinstance(r.team, str) else np.nan
        e = TM[(TM.team == r.team) & (TM.season == Y - 1)]
        eo = TM[(TM.team == tprior) & (TM.season == Y - 1)]
        for c in ENV:
            rec["env_" + c] = float(e[c].iat[0]) if len(e) else np.nan
            rec["oldenv_" + c] = float(eo[c].iat[0]) if len(eo) else np.nan
        rows.append(rec)

    F = pd.DataFrame(rows)
    out = panel.merge(F, on=["gsis_id", "year"], how="left")
    # structure
    bd = out.gsis_id.map(meta.birth_date)
    ref = pd.to_datetime(out.year.astype(str) + "-09-01")
    out["age"] = (ref - bd).dt.days / 365.25
    out["draft_pick"] = out.gsis_id.map(meta.draft_pick).fillna(260.0).clip(upper=260)
    out["log_draft_pick"] = np.log(out.draft_pick)
    out["expr"] = out.year - out.gsis_id.map(meta.rookie_season)
    out["sched"] = out.year.map(sched)
    out["ppsw"] = out.games * out.ppg / out.sched
    out["avail_y"] = out.games / out.sched
    out["pos"] = pos
    return out


if __name__ == "__main__":
    for pos, f in [("WR", "results/market_prior_wr_deep.csv"),
                   ("RB", "results/market_prior_rb_deep.csv")]:
        d = build(f, pos)
        d.to_csv(ROOT / f"data/derived/w1_features_{pos}.csv", index=False)
        print(pos, d.shape, "| rows with history:", int((d.n_prior > 0).sum()),
              "| in_fit & history:", int(((d.n_prior > 0) & d.in_fit).sum()))
        print("  mu_hat NaN:", int(d.mu_hat.isna().sum()),
              "| age NaN:", int(d.age.isna().sum()),
              "| env NaN:", int(d.env_tm_pass_att_pg.isna().sum()))

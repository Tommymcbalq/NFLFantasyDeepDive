#!/usr/bin/env python3
"""
build_advanced.py — derive the advanced-stats tables from the raw caches in
data/advanced/ (populated by scripts/fetch_advanced.py) plus the existing
data/players/weekly_raw/, data/teams/ pulls.

Outputs (data/derived/):
    adv_wr_te.csv       player-season, WR/TE, 2018-2025
    adv_rb.csv          player-season, RB/FB, 2018-2025
    adv_qb.csv          player-season, QB, 2018-2025
    team_context.csv    team-season offense + defense, 2018-2025
    adv_wr_te_recent3.csv / adv_rb_recent3.csv / adv_qb_recent3.csv
                        3-season recency window (2023-2025) collapsed to one row
                        per player, games-weighted, with thin_data flags
    xwalk_pfr_gsis.csv  pfr_id <-> gsis_id crosswalk actually used
    adv_join_report.csv join rates for every merge

Everything is REG season only and per-game where a rate makes sense.
No modelling, no filtering to a universe: all players with >=1 relevant
opportunity are kept; the caller decides the universe.
"""
import os
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", category=FutureWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADV = os.path.join(ROOT, "data", "advanced")
DER = os.path.join(ROOT, "data", "derived")
RES = os.path.join(ROOT, "results")
os.makedirs(DER, exist_ok=True)
os.makedirs(RES, exist_ok=True)

SEASONS = list(range(2018, 2026))
RECENT = [2023, 2024, 2025]

JOIN_LOG = []


def log_join(name, left_rows, matched, note=""):
    rate = matched / left_rows if left_rows else np.nan
    JOIN_LOG.append(dict(join=name, left_rows=left_rows, matched=matched,
                         match_rate=round(rate, 5), note=note))
    print(f"  join {name:38s} {matched}/{left_rows} = {rate:.4f}  {note}")


def norm_name(s):
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    s = re.sub(r"[^a-z]", "", s)
    return s


# ---------------------------------------------------------------- crosswalk
def build_crosswalk():
    p = pd.read_csv(os.path.join(ADV, "players", "players.csv"), low_memory=False)
    p["name_key"] = p["display_name"].map(norm_name)
    xw = (p.dropna(subset=["gsis_id", "pfr_id"])
            .drop_duplicates("pfr_id")[["pfr_id", "gsis_id", "display_name",
                                        "position", "name_key"]])
    # name-key table for fallback (unique keys only, to avoid wrong matches)
    nk = p.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")
    counts = nk["name_key"].value_counts()
    name_map = (nk[nk["name_key"].isin(counts[counts == 1].index)]
                .set_index("name_key")["gsis_id"].to_dict())
    return xw, name_map, p


def pfr_to_gsis(df, xw, name_map, pfr_col="pfr_id", name_col="player", label=""):
    """Map a PFR-keyed frame onto gsis_id, id first then unique-name fallback."""
    out = df.merge(xw[["pfr_id", "gsis_id"]], on=pfr_col, how="left")
    by_id = out["gsis_id"].notna().sum()
    miss = out["gsis_id"].isna()
    if name_col in out.columns:
        fb = out.loc[miss, name_col].map(norm_name).map(name_map)
        out.loc[miss, "gsis_id"] = fb
    total = out["gsis_id"].notna().sum()
    log_join(label, len(out), total, f"{by_id} by pfr_id, {total - by_id} by name")
    return out


# ---------------------------------------------------------------- loaders
def load_weekly():
    fr = []
    for y in SEASONS:
        f = os.path.join(ROOT, "data", "players", "weekly_raw",
                         f"stats_player_week_{y}.csv")
        fr.append(pd.read_csv(f, low_memory=False))
    w = pd.concat(fr, ignore_index=True)
    return w[w["season_type"] == "REG"].copy()


def load_team_week():
    fr = []
    for y in SEASONS:
        f = os.path.join(ROOT, "data", "teams", f"stats_team_week_{y}.csv")
        fr.append(pd.read_csv(f, low_memory=False))
    t = pd.concat(fr, ignore_index=True)
    return t[t["season_type"] == "REG"].copy()


PBP_COLS = [
    "game_id", "play_id", "season", "week", "season_type", "posteam", "defteam",
    "home_team", "away_team", "play_type", "down", "ydstogo", "yardline_100",
    "goal_to_go", "yards_gained", "air_yards", "yards_after_catch", "epa",
    "qb_epa", "cpoe", "pass", "rush", "qb_dropback", "qb_scramble", "sack",
    "qb_hit", "complete_pass", "touchdown", "pass_touchdown", "rush_touchdown",
    "interception", "passer_player_id", "receiver_player_id", "rusher_player_id",
    "fixed_drive", "game_seconds_remaining", "wp", "qtr", "score_differential",
    "xpass", "pass_oe", "special", "aborted_play", "penalty", "first_down",
    "fixed_drive_result", "drive", "two_point_attempt",
]


def load_pbp(season):
    f = os.path.join(ADV, "pbp", f"play_by_play_{season}.parquet")
    d = pd.read_parquet(f, columns=PBP_COLS)
    return d[d["season_type"] == "REG"].copy()


# ---------------------------------------------------------------- receiving
def season_shares(w, tw):
    """Exact season target / air-yards share: player sum over team sum in the
    games the player actually appeared in (not a mean of weekly shares)."""
    t = tw[["season", "week", "team", "targets", "receiving_air_yards",
            "attempts", "carries", "receptions", "receiving_yards"]].rename(
        columns={"targets": "tm_targets", "receiving_air_yards": "tm_air_yards",
                 "attempts": "tm_pass_att", "carries": "tm_carries",
                 "receptions": "tm_receptions", "receiving_yards": "tm_rec_yards"})
    m = w.merge(t, on=["season", "week", "team"], how="left")
    log_join("player_week -> team_week", len(w), m["tm_targets"].notna().sum())
    return m


def agg_receiving(w, tw):
    g = w.groupby(["player_id", "season"], as_index=False).agg(
        player_name=("player_display_name", "first"),
        position=("position", lambda s: s.mode().iat[0] if len(s.mode()) else np.nan),
        team=("team", lambda s: s.mode().iat[0] if len(s.mode()) else np.nan),
        n_teams=("team", "nunique"),
        games=("week", "nunique"),
        targets=("targets", "sum"),
        receptions=("receptions", "sum"),
        rec_yards=("receiving_yards", "sum"),
        rec_tds=("receiving_tds", "sum"),
        rec_air_yards=("receiving_air_yards", "sum"),
        rec_yac=("receiving_yards_after_catch", "sum"),
        rec_first_downs=("receiving_first_downs", "sum"),
        rec_epa=("receiving_epa", "sum"),
        carries=("carries", "sum"),
        rush_yards=("rushing_yards", "sum"),
        rush_tds=("rushing_tds", "sum"),
        ppr=("fantasy_points_ppr", "sum"),
        tm_targets=("tm_targets", "sum"),
        tm_air_yards=("tm_air_yards", "sum"),
        tm_pass_att=("tm_pass_att", "sum"),
        tm_carries=("tm_carries", "sum"),
    )
    # Two denominators, both wanted and NOT interchangeable:
    #  *_share      -> team total over the games the player was active for. This is
    #                  the projection-relevant "share while on the field" quantity.
    #                  It does NOT sum to 1 across a team-season (players miss
    #                  different games), so never use it as a team budget.
    #  *_share_full -> team total over the whole season. Sums to 1 across all
    #                  players of a team-season; use this for budget/vacated-share
    #                  arithmetic. Understates part-season players.
    full = tw.groupby(["season", "team"], as_index=False).agg(
        tm_targets_full=("targets", "sum"),
        tm_air_yards_full=("receiving_air_yards", "sum"),
        tm_carries_full=("carries", "sum"),
        tm_games=("week", "nunique"))
    return g.merge(full, on=["season", "team"], how="left")


def safe_div(a, b):
    return a / b.replace(0, np.nan)


def derive_common(g):
    """All ratios rebuilt from summed counts. Applied identically to the
    season table and to the multi-season window table, so a 3-season rate is
    sum(numerator)/sum(denominator), never a mean of season rates."""
    # Two denominators, both wanted and NOT interchangeable:
    #  *_share      -> team total over the games the player was active for. This
    #                  is the projection-relevant "share while on the field".
    #                  It does NOT sum to 1 across a team-season (players miss
    #                  different games), so never use it as a team budget.
    #  *_share_full -> team total over the whole season. Sums to ~1 across all
    #                  players of a team-season; use for budget / vacated-share
    #                  arithmetic. Understates part-season players.
    g["target_share"] = safe_div(g["targets"], g["tm_targets"])
    g["air_yards_share"] = safe_div(g["rec_air_yards"], g["tm_air_yards"])
    g["target_share_full"] = safe_div(g["targets"], g["tm_targets_full"])
    g["air_yards_share_full"] = safe_div(g["rec_air_yards"], g["tm_air_yards_full"])
    g["carry_share"] = safe_div(g["carries"], g["tm_carries"])
    g["carry_share_full"] = safe_div(g["carries"], g["tm_carries_full"])
    g["games_played_rate"] = safe_div(g["games"], g["tm_games"])
    g["wopr"] = 1.5 * g["target_share"] + 0.7 * g["air_yards_share"]
    g["racr"] = safe_div(g["rec_yards"], g["rec_air_yards"])
    g["adot_nflverse"] = safe_div(g["rec_air_yards"], g["targets"])
    g["catch_rate"] = safe_div(g["receptions"], g["targets"])
    g["rec_epa_per_target"] = safe_div(g["rec_epa"], g["targets"])
    g["ypc"] = safe_div(g["rush_yards"], g["carries"])
    g["ypr"] = safe_div(g["rec_yards"], g["receptions"])
    if "off_snaps" in g and "tm_off_snaps" in g:
        g["snap_share"] = safe_div(g["off_snaps"], g["tm_off_snaps"])
        g["off_snaps_pg"] = safe_div(g["off_snaps"], g["snap_games"])
    if "dropbacks_on_field" in g:
        g["pass_snap_share"] = safe_div(g["dropbacks_on_field"], g["tm_dropbacks"])
        g["run_snap_share"] = safe_div(g["rushes_on_field"], g["tm_rushes"])
        g["play_share_part"] = safe_div(g["off_plays_on_field"], g["tm_off_plays"])
    for c, per in [("targets", "targets_pg"), ("receptions", "rec_pg"),
                   ("rec_yards", "rec_yards_pg"), ("rec_tds", "rec_tds_pg"),
                   ("rec_air_yards", "air_yards_pg"), ("ppr", "ppr_pg"),
                   ("carries", "carries_pg"), ("rush_yards", "rush_yards_pg"),
                   ("rush_tds", "rush_tds_pg")]:
        if c in g:
            g[per] = safe_div(g[c], g["games"])
    return g


def derive_wrte(g):
    g = derive_common(g)
    # "routes" here = team dropbacks the player was on the field for. It counts
    # blocking TEs/backs as having run a route, so TPRR/YPRR are UPPER-bounded
    # proxies, not charted routes. Named *_proxy throughout for that reason.
    g["routes_proxy"] = g["dropbacks_on_field"]
    g["routes_proxy_pg"] = safe_div(g["routes_proxy"], g["games"])
    g["tprr_proxy"] = safe_div(g["targets"], g["routes_proxy"])
    g["yprr_proxy"] = safe_div(g["rec_yards"], g["routes_proxy"])
    g["pfr_drop_pct"] = safe_div(g["pfr_drops"], g["pfr_tgt"])
    g["target_epa"] = safe_div(g["target_epa_total"], g["pbp_targets"])
    for c in ["rz_targets", "i10_targets", "ez_targets", "deep_targets",
              "third_down_targets"]:
        g[c + "_pg"] = safe_div(g[c], g["games"])
    g["deep_target_rate"] = safe_div(g["deep_targets"], g["pbp_targets"])
    g["rz_target_share_of_own"] = safe_div(g["rz_targets"], g["pbp_targets"])
    g["third_down_conv_rate"] = safe_div(g["third_down_conv"], g["third_down_targets"])
    return g


def derive_rb(g):
    g = derive_common(g)
    g["touches"] = g["carries"].fillna(0) + g["receptions"].fillna(0)
    g["touches_pg"] = safe_div(g["touches"], g["games"])
    g["opportunity_pg"] = safe_div(g["carries"].fillna(0) + g["targets"].fillna(0),
                                   g["games"])
    g["stuffed_rate"] = safe_div(g["stuffed"], g["pbp_carries"])
    g["short_yd_conv_rate"] = safe_div(g["short_yd_conv"], g["short_yd_carries"])
    g["explosive_run_rate"] = safe_div(g["explosive_runs"], g["pbp_carries"])
    g["rush_epa_per_att"] = safe_div(g["rush_epa_total"], g["pbp_carries"])
    g["target_epa"] = safe_div(g["target_epa_total"], g["pbp_targets"])
    g["gl5_carry_share_of_own"] = safe_div(g["gl5_carries"], g["pbp_carries"])
    g["third_down_carry_rate"] = safe_div(g["third_down_carries"], g["pbp_carries"])
    g["pfr_drop_pct"] = safe_div(g["pfr_drops"], g["pfr_tgt"])
    g["ngs_ryoe_per_att"] = safe_div(g["ngs_rush_yards_over_expected"], g["carries"]) \
        if "ngs_rush_yards_over_expected" in g else np.nan
    g["routes_proxy"] = g["dropbacks_on_field"]
    g["tprr_proxy"] = safe_div(g["targets"], g["routes_proxy"])
    for c in ["gl5_carries", "gl10_carries", "goal_to_go_carries",
              "third_down_carries", "explosive_runs", "rz_targets"]:
        g[c + "_pg"] = safe_div(g[c], g["games"])
    return g


def derive_qb(g):
    g["comp_pct"] = safe_div(g["completions"], g["attempts"])
    g["ypa"] = safe_div(g["pass_yards"], g["attempts"])
    g["adot"] = safe_div(g["pass_air_yards"], g["attempts"])
    g["td_rate"] = safe_div(g["pass_tds"], g["attempts"])
    g["int_rate"] = safe_div(g["interceptions"], g["attempts"])
    g["td_int"] = safe_div(g["pass_tds"], g["interceptions"])
    g["epa_per_dropback"] = safe_div(g["qb_epa_total"], g["dropbacks"])
    g["cpoe"] = safe_div(g["cpoe_sum"], g["cpoe_n"])
    g["sack_rate"] = safe_div(g["sacks"], g["dropbacks"])
    g["scramble_rate"] = safe_div(g["scrambles"], g["dropbacks"])
    g["pressure_rate_pbp"] = safe_div(g["qb_hits"] + g["sacks"], g["dropbacks"])
    g["pfr_pressure_pct"] = safe_div(g["times_pressured"], g["pfr_pass_attempts"]) * 100
    g["pfr_bad_throw_pct"] = safe_div(g["bad_throws"], g["pfr_pass_attempts"]) * 100
    g["pfr_blitz_rate"] = safe_div(g["times_blitzed"], g["pfr_pass_attempts"])
    g["pfr_pa_rate"] = safe_div(g["pa_pass_att"], g["pfr_pass_attempts"])
    g["pfr_rpo_rate"] = safe_div(g["rpo_plays"], g["pfr_pass_attempts"])
    g["designed_rushes"] = g["carries"].fillna(0) - g["scrambles"].fillna(0)
    g["rush_epa_per_att"] = safe_div(g["rush_epa_total"], g["pbp_carries"])
    # rushing contribution to PPR: the §O finding is that rush VOLUME, not
    # experience, drives QB fantasy variance, so keep both level and share.
    g["rush_ppr"] = 0.1 * g["rush_yards"].fillna(0) + 6 * g["rush_tds"].fillna(0)
    g["rush_share_of_ppr"] = safe_div(g["rush_ppr"], g["ppr"])
    for c in ["attempts", "pass_yards", "pass_tds", "interceptions", "carries",
              "rush_yards", "rush_tds", "ppr", "dropbacks", "designed_rushes",
              "rush_ppr", "sacks", "scrambles", "gl5_carries"]:
        if c in g:
            g[c + "_pg"] = safe_div(g[c], g["games"])
    return g


# ---------------------------------------------------------------- snaps
def snap_shares():
    fr = []
    for y in SEASONS:
        f = os.path.join(ADV, "snap_counts", f"snap_counts_{y}.csv")
        if not os.path.exists(f):
            f = os.path.join(ROOT, "data", "snap_counts", f"snap_counts_{y}.csv")
        fr.append(pd.read_csv(f, low_memory=False))
    s = pd.concat(fr, ignore_index=True)
    s = s[s["game_type"] == "REG"].copy()
    tm = (s.groupby(["season", "week", "team"], as_index=False)["offense_snaps"]
            .max().rename(columns={"offense_snaps": "tm_off_snaps"}))
    s = s.merge(tm, on=["season", "week", "team"], how="left")
    out = s.groupby(["pfr_player_id", "season"], as_index=False).agg(
        snap_games=("offense_snaps", lambda x: int((x > 0).sum())),
        off_snaps=("offense_snaps", "sum"),
        tm_off_snaps=("tm_off_snaps", "sum"),
        snap_pct_mean=("offense_pct", "mean"),
        st_snaps=("st_snaps", "sum"),
    )
    return out.rename(columns={"pfr_player_id": "pfr_id"})


# ---------------------------------------------------------------- pbp usage
def pbp_usage(season):
    d = load_pbp(season)
    d = d[(d["aborted_play"] != 1)]
    # two-point conversion plays are excluded: nflverse weekly stats book them
    # as *_2pt_conversions, not as targets/carries, and they sit at the 2-yard
    # line so they would also inflate goal-line usage counts.
    plays = d[(d["special"] != 1) & (d["two_point_attempt"] != 1)
              & (d["play_type"] != "no_play")]

    # ---- receiving usage
    tg = plays[(plays["pass"] == 1) & plays["receiver_player_id"].notna()].copy()
    tg["rz"] = (tg["yardline_100"] <= 20).astype(int)
    tg["i10"] = (tg["yardline_100"] <= 10).astype(int)
    tg["ez"] = (tg["air_yards"] >= tg["yardline_100"]).astype(int)
    tg["deep"] = (tg["air_yards"] >= 20).astype(int)
    tg["third"] = (tg["down"] == 3).astype(int)
    tg["third_conv"] = ((tg["down"] == 3) & (tg["first_down"] == 1)).astype(int)
    rec = tg.groupby("receiver_player_id", as_index=False).agg(
        pbp_targets=("play_id", "count"),
        rz_targets=("rz", "sum"), i10_targets=("i10", "sum"),
        ez_targets=("ez", "sum"), deep_targets=("deep", "sum"),
        third_down_targets=("third", "sum"),
        third_down_conv=("third_conv", "sum"),
        target_epa_total=("epa", "sum"),
    ).rename(columns={"receiver_player_id": "player_id"})

    # ---- rushing usage
    ru = plays[plays["rush"] == 1].copy()
    ru = ru[ru["rusher_player_id"].notna()]
    ru["gl5"] = (ru["yardline_100"] <= 5).astype(int)
    ru["gl10"] = (ru["yardline_100"] <= 10).astype(int)
    ru["gtg"] = (ru["goal_to_go"] == 1).astype(int)
    ru["stuff"] = (ru["yards_gained"] <= 0).astype(int)
    ru["third"] = (ru["down"] == 3).astype(int)
    ru["short_yd"] = ((ru["down"].isin([3, 4])) & (ru["ydstogo"] <= 2)).astype(int)
    ru["short_yd_conv"] = (ru["short_yd"] * (ru["first_down"] == 1)).astype(int)
    ru["explosive"] = (ru["yards_gained"] >= 10).astype(int)
    rush = ru.groupby("rusher_player_id", as_index=False).agg(
        pbp_carries=("play_id", "count"),
        gl5_carries=("gl5", "sum"), gl10_carries=("gl10", "sum"),
        goal_to_go_carries=("gtg", "sum"),
        stuffed=("stuff", "sum"),
        third_down_carries=("third", "sum"),
        short_yd_carries=("short_yd", "sum"),
        short_yd_conv=("short_yd_conv", "sum"),
        explosive_runs=("explosive", "sum"),
        rush_epa_total=("epa", "sum"),
    ).rename(columns={"rusher_player_id": "player_id"})

    # ---- QB usage
    db = plays[plays["qb_dropback"] == 1].copy()
    # On a scramble the play is booked as a rush: passer_player_id is null and
    # the QB sits in rusher_player_id. Keying on passer_player_id alone silently
    # drops every scramble, which undercounts dropbacks, biases EPA/dropback
    # (scrambles skew positive) and zeroes out the designed-rush split.
    db["qb_id"] = db["passer_player_id"].fillna(db["rusher_player_id"])
    db = db[db["qb_id"].notna()]
    qb = db.groupby("qb_id", as_index=False).agg(
        dropbacks=("play_id", "count"),
        qb_epa_total=("qb_epa", "sum"),
        cpoe_sum=("cpoe", "sum"), cpoe_n=("cpoe", "count"),
        sacks=("sack", "sum"),
        scrambles=("qb_scramble", "sum"),
        pbp_air_yards=("air_yards", "sum"), pbp_air_yards_n=("air_yards", "count"),
        qb_hits=("qb_hit", "sum"),
    ).rename(columns={"qb_id": "player_id"})

    for f in (rec, rush, qb):
        f["season"] = season
    return rec, rush, qb, d


# ---------------------------------------------------------------- participation
def participation_shares(season):
    """Pass-snap participation: share of the team's dropbacks the player was on
    the field for. This is a ROUTE PARTICIPATION PROXY -- it counts blocking
    backs/TEs as 'in', so it is an upper bound on routes run."""
    pf = os.path.join(ADV, "participation", f"pbp_participation_{season}.parquet")
    if not os.path.exists(pf):
        return None
    par = pd.read_parquet(pf, columns=["nflverse_game_id", "play_id",
                                       "possession_team", "offense_players"])
    par = par.rename(columns={"nflverse_game_id": "game_id"})
    d = load_pbp(season)[["game_id", "play_id", "posteam", "qb_dropback",
                          "rush", "special", "aborted_play"]]
    par["play_id"] = par["play_id"].astype("float64")
    m = par.merge(d, on=["game_id", "play_id"], how="inner")
    m = m[(m["special"] != 1) & (m["aborted_play"] != 1)]
    m = m[m["offense_players"].astype(str).str.len() > 0]

    tm = m.groupby("posteam").agg(tm_off_plays=("play_id", "count"),
                                  tm_dropbacks=("qb_dropback", "sum"),
                                  tm_rushes=("rush", "sum")).reset_index()

    m["pl"] = m["offense_players"].astype(str).str.split(";")
    e = m[["posteam", "qb_dropback", "rush", "pl"]].explode("pl")
    e = e[e["pl"].str.startswith("00-")]
    out = e.groupby(["pl", "posteam"], as_index=False).agg(
        off_plays_on_field=("qb_dropback", "count"),
        dropbacks_on_field=("qb_dropback", "sum"),
        rushes_on_field=("rush", "sum"))
    out = out.rename(columns={"pl": "player_id", "posteam": "team"})
    out = out.merge(tm, left_on="team", right_on="posteam", how="left").drop(columns="posteam")
    # a player can appear for >1 team in a season: collapse
    out = out.groupby("player_id", as_index=False).agg(
        off_plays_on_field=("off_plays_on_field", "sum"),
        dropbacks_on_field=("dropbacks_on_field", "sum"),
        rushes_on_field=("rushes_on_field", "sum"),
        tm_off_plays=("tm_off_plays", "sum"),
        tm_dropbacks=("tm_dropbacks", "sum"),
        tm_rushes=("tm_rushes", "sum"))
    out["season"] = season
    return out


# ---------------------------------------------------------------- FTN charting
def ftn_team(all_pbp):
    """Team-season scheme rates from FTN charting (2022+ only; NaN before)."""
    fr = []
    for y in SEASONS:
        f = os.path.join(ADV, "ftn", f"ftn_charting_{y}.csv")
        if os.path.exists(f):
            fr.append(pd.read_csv(f, low_memory=False))
    if not fr:
        return None
    d = pd.concat(fr, ignore_index=True).rename(
        columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    key = all_pbp[["game_id", "play_id", "season", "posteam", "defteam",
                   "qb_dropback", "special", "aborted_play"]]
    m = d.merge(key, on=["game_id", "play_id"], how="inner",
                suffixes=("_ftn", ""))
    m = m[(m["special"] != 1) & (m["aborted_play"] != 1)]
    off = m.groupby(["season", "posteam"]).agg(
        ftn_plays=("play_id", "count"),
        ftn_motion_rate=("is_motion", "mean"),
        ftn_no_huddle_rate=("is_no_huddle", "mean"),
        ftn_trick_rate=("is_trick_play", "mean"),
    ).reset_index().rename(columns={"posteam": "team"})
    db = m[m["qb_dropback"] == 1]
    offp = db.groupby(["season", "posteam"]).agg(
        ftn_pa_rate=("is_play_action", "mean"),
        ftn_screen_rate=("is_screen_pass", "mean"),
        ftn_rpo_rate=("is_rpo", "mean"),
        ftn_oop_rate=("is_qb_out_of_pocket", "mean"),
        ftn_blitz_faced_rate=("n_blitzers", lambda s: (s > 0).mean()),
        ftn_pass_rushers_faced=("n_pass_rushers", "mean"),
    ).reset_index().rename(columns={"posteam": "team"})
    dfn = db.groupby(["season", "defteam"]).agg(
        ftn_def_blitz_rate=("n_blitzers", lambda s: (s > 0).mean()),
        ftn_def_pass_rushers=("n_pass_rushers", "mean"),
        ftn_def_box=("n_defense_box", "mean"),
    ).reset_index().rename(columns={"defteam": "team"})
    return off.merge(offp, on=["season", "team"], how="left") \
              .merge(dfn, on=["season", "team"], how="left")


# ---------------------------------------------------------------- team context
def team_context(all_pbp, tw, w):
    rows = []
    d = all_pbp
    off = d[(d["special"] != 1) & (d["aborted_play"] != 1) & d["posteam"].notna()].copy()

    # pace: seconds elapsed between consecutive plays of the same drive
    off = off.sort_values(["game_id", "fixed_drive", "play_id"])
    off["sec_gap"] = -off.groupby(["game_id", "fixed_drive"])["game_seconds_remaining"].diff(-1)
    off.loc[(off["sec_gap"] <= 0) | (off["sec_gap"] > 60), "sec_gap"] = np.nan
    neutral = (off["wp"].between(0.2, 0.8)) & (off["qtr"] <= 3)

    grp = off.groupby(["season", "posteam"])
    o = grp.agg(
        off_plays=("play_id", "count"),
        off_epa=("epa", "mean"),
        pass_plays=("pass", "sum"),
        rush_plays=("rush", "sum"),
        proe=("pass_oe", "mean"),
        xpass=("xpass", "mean"),
    ).reset_index().rename(columns={"posteam": "team"})

    npl = off[neutral].groupby(["season", "posteam"]).agg(
        neutral_plays=("play_id", "count"),
        neutral_pass_rate=("pass", "mean"),
        neutral_proe=("pass_oe", "mean"),
        neutral_sec_per_play=("sec_gap", "mean"),
    ).reset_index().rename(columns={"posteam": "team"})
    o = o.merge(npl, on=["season", "team"], how="left")

    dbk = off[off["qb_dropback"] == 1].groupby(["season", "posteam"])["epa"].mean()
    rsh = off[off["rush"] == 1].groupby(["season", "posteam"])["epa"].mean()
    o = o.merge(dbk.rename("off_pass_epa_play").reset_index()
                .rename(columns={"posteam": "team"}), on=["season", "team"], how="left")
    o = o.merge(rsh.rename("off_rush_epa_play").reset_index()
                .rename(columns={"posteam": "team"}), on=["season", "team"], how="left")

    # defensive EPA allowed
    dfn = off.groupby(["season", "defteam"]).agg(
        def_plays=("play_id", "count"), def_epa=("epa", "mean")).reset_index() \
        .rename(columns={"defteam": "team"})
    dpass = off[off["qb_dropback"] == 1].groupby(["season", "defteam"])["epa"].mean() \
        .rename("def_pass_epa_play").reset_index().rename(columns={"defteam": "team"})
    drush = off[off["rush"] == 1].groupby(["season", "defteam"])["epa"].mean() \
        .rename("def_rush_epa_play").reset_index().rename(columns={"defteam": "team"})
    dfn = dfn.merge(dpass, on=["season", "team"], how="left") \
             .merge(drush, on=["season", "team"], how="left")

    # red zone TD rate (offense) from drives
    rz = off[off["yardline_100"] <= 20]
    rzd = rz.groupby(["season", "posteam", "game_id", "fixed_drive"])["fixed_drive_result"] \
            .first().reset_index()
    rzd["td"] = (rzd["fixed_drive_result"] == "Touchdown").astype(int)
    rzagg = rzd.groupby(["season", "posteam"]).agg(
        rz_drives=("td", "count"), rz_td_rate=("td", "mean")).reset_index() \
        .rename(columns={"posteam": "team"})
    o = o.merge(rzagg, on=["season", "team"], how="left")

    # points scored / allowed and yards from team-week
    tws = tw.groupby(["season", "team"], as_index=False).agg(
        games=("week", "nunique"),
        pass_att=("attempts", "sum"), pass_yards=("passing_yards", "sum"),
        pass_tds=("passing_tds", "sum"), carries=("carries", "sum"),
        rush_yards=("rushing_yards", "sum"), rush_tds=("rushing_tds", "sum"),
        team_targets=("targets", "sum"), team_air_yards=("receiving_air_yards", "sum"),
        cpoe=("passing_cpoe", "mean"))
    tws["total_yards"] = tws["pass_yards"] + tws["rush_yards"]

    # points from games file
    gf = pd.read_csv(os.path.join(ROOT, "data", "teams", "games_nflverse_20260809.csv"),
                     low_memory=False)
    gf = gf[(gf["game_type"] == "REG") & gf["home_score"].notna()]
    a = gf[["season", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "pf", "home_score": "pa"})
    h = gf[["season", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "pf", "away_score": "pa"})
    pts = pd.concat([a, h]).groupby(["season", "team"], as_index=False).agg(
        g=("pf", "count"), points_for=("pf", "sum"), points_against=("pa", "sum"))

    # yards allowed: opponent offense
    ya = tw.merge(tws[["season", "team"]], on=["season", "team"], how="left")
    yal = tw.groupby(["season", "opponent_team"], as_index=False).agg(
        pass_yards_allowed=("passing_yards", "sum"),
        rush_yards_allowed=("rushing_yards", "sum"),
        pass_att_faced=("attempts", "sum"),
        carries_faced=("carries", "sum"),
        targets_faced=("targets", "sum")).rename(columns={"opponent_team": "team"})
    yal["total_yards_allowed"] = yal["pass_yards_allowed"] + yal["rush_yards_allowed"]

    # fantasy points allowed by position faced
    wp = w[w["position"].isin(["QB", "RB", "WR", "TE"])]
    fpa = wp.groupby(["season", "opponent_team", "position"], as_index=False)[
        "fantasy_points_ppr"].sum()
    fpa = fpa.pivot(index=["season", "opponent_team"], columns="position",
                    values="fantasy_points_ppr").reset_index() \
        .rename(columns={"opponent_team": "team", "QB": "fpa_qb", "RB": "fpa_rb",
                         "WR": "fpa_wr", "TE": "fpa_te"})

    t = (tws.merge(o, on=["season", "team"], how="left")
            .merge(dfn, on=["season", "team"], how="left")
            .merge(pts, on=["season", "team"], how="left")
            .merge(yal, on=["season", "team"], how="left")
            .merge(fpa, on=["season", "team"], how="left"))

    for c in ["points_for", "points_against", "total_yards", "pass_yards",
              "rush_yards", "total_yards_allowed", "pass_yards_allowed",
              "rush_yards_allowed", "pass_att", "carries", "off_plays",
              "fpa_qb", "fpa_rb", "fpa_wr", "fpa_te", "team_targets"]:
        t[c + "_pg"] = t[c] / t["games"]
    t["pass_rate"] = t["pass_plays"] / (t["pass_plays"] + t["rush_plays"])

    # ranks, 1 = best for the team in question
    higher_better = ["points_for_pg", "total_yards_pg", "pass_yards_pg",
                     "rush_yards_pg", "off_epa", "off_pass_epa_play",
                     "off_rush_epa_play", "rz_td_rate", "off_plays_pg"]
    lower_better = ["points_against_pg", "total_yards_allowed_pg",
                    "pass_yards_allowed_pg", "rush_yards_allowed_pg", "def_epa",
                    "def_pass_epa_play", "def_rush_epa_play",
                    "fpa_qb_pg", "fpa_rb_pg", "fpa_wr_pg", "fpa_te_pg"]
    ftn = ftn_team(all_pbp)
    if ftn is not None:
        t = t.merge(ftn, on=["season", "team"], how="left")

    for c in higher_better:
        t["rank_" + c] = t.groupby("season")[c].rank(ascending=False, method="min")
    for c in lower_better:
        t["rank_" + c] = t.groupby("season")[c].rank(ascending=True, method="min")
    return t.sort_values(["season", "team"]).reset_index(drop=True)


# ---------------------------------------------------------------- recency window
# Source-provided rates that CANNOT be rebuilt from counts (no denominator is
# published). These get a volume-weighted mean across the window, weight named.
RATE_WEIGHTS = {
    # NGS receiving -- per-target tracking averages
    "ngs_avg_cushion": "targets", "ngs_avg_separation": "targets",
    "ngs_avg_intended_air_yards": "targets",
    "ngs_percent_share_of_intended_air_yards": "targets",
    "ngs_avg_yac": "receptions", "ngs_avg_expected_yac": "receptions",
    "ngs_avg_yac_above_expectation": "receptions",
    "ngs_catch_percentage": "targets",
    # NGS rushing
    "ngs_efficiency": "carries",
    "ngs_percent_attempts_gte_eight_defenders": "carries",
    "ngs_avg_time_to_los": "carries",
    "ngs_rush_yards_over_expected_per_att": "carries",
    "ngs_rush_pct_over_expected": "carries",
    # NGS passing
    "ngs_avg_time_to_throw": "attempts", "ngs_avg_completed_air_yards": "attempts",
    "ngs_avg_intended_air_yards_pass": "attempts",
    "ngs_avg_air_yards_differential": "attempts", "ngs_aggressiveness": "attempts",
    "ngs_avg_air_yards_to_sticks": "attempts",
    "ngs_expected_completion_percentage": "attempts",
    "ngs_completion_percentage_above_expectation": "attempts",
    "ngs_avg_air_distance": "attempts", "ngs_passer_rating": "attempts",
    # PFR per-unit rates
    "pfr_adot": "targets", "pfr_ybc_per_rec": "receptions",
    "pfr_yac_per_rec": "receptions", "pfr_rec_per_broken_tackle": "receptions",
    "pfr_ybc_per_att": "carries", "pfr_yac_per_att": "carries",
    "pfr_att_per_broken_tackle": "carries",
    "pfr_pocket_time": "attempts", "pfr_scramble_ypa": "attempts",
    "pfr_iay_per_att": "attempts", "pfr_cay_per_comp": "completions",
    # weekly / external season rates
    "snap_pct_mean": "games", "cpoe_w": "attempts",
    "espn_qbr": "attempts", "espn_qbr_raw": "attempts",
}
# NGS max is a max, not a mean
RATE_MAX = ["ngs_max_air_distance"]
NEVER_SUM = set(RATE_WEIGHTS) | set(RATE_MAX) | {
    "season", "week", "age", "rank", "n_teams"}


def recency_window(df, seasons, derive, min_games=8):
    """Collapse to one row per player over `seasons`.

    Counts are summed; every ratio is then REBUILT from those sums by the same
    `derive` function used on the season table (so a window rate is
    sum(num)/sum(den), not a mean of season rates). Only rates with no available
    denominator fall back to a volume-weighted mean -- see RATE_WEIGHTS.
    """
    d = df[df["season"].isin(seasons)].copy()
    num = d.select_dtypes("number").columns
    count_cols = [c for c in num if c not in NEVER_SUM and not c.endswith(
        ("_pg", "_share", "_share_full", "_rate", "_pct", "_per_target",
         "_per_att", "_per_dropback", "_proxy", "wopr", "racr", "ypc", "ypr",
         "_nflverse", "epa_per_att"))]
    out = d.groupby("player_id", as_index=False)[count_cols].sum(min_count=1)

    wm = {}
    for c, wcol in RATE_WEIGHTS.items():
        if c not in d.columns or wcol not in d.columns:
            continue
        sub = d[["player_id", c, wcol]].dropna(subset=[c])
        sub = sub[sub[wcol].fillna(0) > 0]
        if sub.empty:
            continue
        sub["_num"] = sub[c] * sub[wcol]
        agg = sub.groupby("player_id").agg(n=("_num", "sum"), w=(wcol, "sum"))
        wm[c] = agg["n"] / agg["w"].replace(0, np.nan)
    for c in RATE_MAX:
        if c in d.columns:
            wm[c] = d.groupby("player_id")[c].max()
    if wm:
        out = out.merge(pd.DataFrame(wm).reset_index(), on="player_id", how="left")

    meta = d.sort_values("season").groupby("player_id", as_index=False).agg(
        player_name=("player_name", "last"), position=("position", "last"),
        team_last=("team", "last"), seasons_played=("season", "nunique"),
        last_season=("season", "max"), first_season=("season", "min"),
        n_teams=("team", "nunique"))
    out = meta.merge(out, on="player_id", how="left")
    out = derive(out)
    out["window_seasons"] = f"{min(seasons)}-{max(seasons)}"
    out["thin_data"] = ((out["seasons_played"] < 2) | (out["games"] < min_games))
    return out


# ---------------------------------------------------------------- main
def main():
    print("crosswalk")
    xw, name_map, players = build_crosswalk()
    xw.to_csv(os.path.join(DER, "xwalk_pfr_gsis.csv"), index=False)
    log_join("players.csv pfr_id<->gsis_id", len(players),
             int(players["pfr_id"].notna().sum()), "coverage of nflverse players.csv")

    print("weekly + team-week")
    w = load_weekly()
    tw = load_team_week()
    w = season_shares(w, tw)

    print("season aggregates")
    base = agg_receiving(w, tw)

    print("snap counts")
    sn = snap_shares()
    sn = pfr_to_gsis(sn, xw, name_map, label="snap_counts -> gsis_id")
    sn = sn.dropna(subset=["gsis_id"]).rename(columns={"gsis_id": "player_id"})
    sn = sn.drop_duplicates(["player_id", "season"])

    print("pbp usage + participation (per season)")
    recs, rushes, qbs, parts, pbps = [], [], [], [], []
    for y in SEASONS:
        r, ru, q, d = pbp_usage(y)
        recs.append(r); rushes.append(ru); qbs.append(q); pbps.append(d)
        p = participation_shares(y)
        if p is not None:
            parts.append(p)
        print(f"  {y}: rec {len(r)}, rush {len(ru)}, qb {len(q)}, part {0 if p is None else len(p)}")
    pbp_rec = pd.concat(recs, ignore_index=True)
    pbp_rush = pd.concat(rushes, ignore_index=True)
    pbp_qb = pd.concat(qbs, ignore_index=True)
    part = pd.concat(parts, ignore_index=True)
    all_pbp = pd.concat(pbps, ignore_index=True)

    print("PFR advanced")
    pfr_rec = pd.read_csv(os.path.join(ADV, "pfr", "advstats_season_rec.csv"))
    pfr_rec = pfr_to_gsis(pfr_rec, xw, name_map, label="pfr_rec -> gsis_id")
    pfr_rec = pfr_rec.dropna(subset=["gsis_id"]).rename(columns={"gsis_id": "player_id"})
    pfr_rec = pfr_rec.drop_duplicates(["player_id", "season"])[
        ["player_id", "season", "tgt", "rec", "adot", "ybc", "ybc_r", "yac",
         "yac_r", "brk_tkl", "rec_br", "drop", "x1d"]].rename(columns={
            "tgt": "pfr_tgt", "rec": "pfr_rec",
            "adot": "pfr_adot", "ybc": "pfr_ybc", "ybc_r": "pfr_ybc_per_rec",
            "yac": "pfr_yac", "yac_r": "pfr_yac_per_rec",
            "brk_tkl": "pfr_broken_tackles", "rec_br": "pfr_rec_per_broken_tackle",
            "drop": "pfr_drops",
            "x1d": "pfr_first_downs"})

    pfr_rush = pd.read_csv(os.path.join(ADV, "pfr", "advstats_season_rush.csv"))
    pfr_rush = pfr_to_gsis(pfr_rush, xw, name_map, label="pfr_rush -> gsis_id")
    pfr_rush = pfr_rush.dropna(subset=["gsis_id"]).rename(columns={"gsis_id": "player_id"})
    pfr_rush = pfr_rush.drop_duplicates(["player_id", "season"])[
        ["player_id", "season", "att", "ybc", "ybc_att", "yac", "yac_att",
         "brk_tkl", "att_br", "x1d"]].rename(columns={
            "att": "pfr_rush_att",
            "ybc": "pfr_rush_ybc", "ybc_att": "pfr_ybc_per_att",
            "yac": "pfr_rush_yac", "yac_att": "pfr_yac_per_att",
            "brk_tkl": "pfr_rush_broken_tackles", "att_br": "pfr_att_per_broken_tackle",
            "x1d": "pfr_rush_first_downs"})

    pfr_pass = pd.read_csv(os.path.join(ADV, "pfr", "advstats_season_pass.csv"))
    pfr_pass = pfr_to_gsis(pfr_pass, xw, name_map, label="pfr_pass -> gsis_id")
    pfr_pass = pfr_pass.dropna(subset=["gsis_id"]).rename(columns={"gsis_id": "player_id"})
    pfr_pass = pfr_pass.drop_duplicates(["player_id", "season"])[
        ["player_id", "season", "pocket_time", "times_blitzed", "times_hurried",
         "times_hit", "times_pressured", "pressure_pct", "bad_throws",
         "bad_throw_pct", "on_tgt_pct", "drop_pct", "throwaways", "batted_balls",
         "rpo_plays", "pa_pass_att", "intended_air_yards_per_pass_attempt",
         "completed_air_yards_per_completion", "scrambles",
         "scramble_yards_per_attempt", "pass_attempts"]].rename(columns={
            "pocket_time": "pfr_pocket_time", "pressure_pct": "pfr_pressure_pct",
            "bad_throw_pct": "pfr_bad_throw_pct", "on_tgt_pct": "pfr_on_target_pct",
            "drop_pct": "pfr_team_drop_pct", "pass_attempts": "pfr_pass_attempts",
            "scrambles": "pfr_scrambles",
            "scramble_yards_per_attempt": "pfr_scramble_ypa",
            "intended_air_yards_per_pass_attempt": "pfr_iay_per_att",
            "completed_air_yards_per_completion": "pfr_cay_per_comp"})
    pfr_pass["pfr_blitz_rate"] = pfr_pass["times_blitzed"] / pfr_pass["pfr_pass_attempts"]
    pfr_pass["pfr_pa_rate"] = pfr_pass["pa_pass_att"] / pfr_pass["pfr_pass_attempts"]
    pfr_pass["pfr_rpo_rate"] = pfr_pass["rpo_plays"] / pfr_pass["pfr_pass_attempts"]

    print("NGS")
    ngs_rec = pd.read_csv(os.path.join(ADV, "ngs", "ngs_receiving.csv.gz"))
    ngs_rec = ngs_rec[(ngs_rec["season_type"] == "REG") & (ngs_rec["week"] == 0)]
    ngs_rec = ngs_rec.rename(columns={"player_gsis_id": "player_id"})[
        ["player_id", "season", "avg_cushion", "avg_separation",
         "avg_intended_air_yards", "percent_share_of_intended_air_yards",
         "avg_yac", "avg_expected_yac", "avg_yac_above_expectation",
         "catch_percentage"]].add_prefix("ngs_").rename(
        columns={"ngs_player_id": "player_id", "ngs_season": "season"})

    ngs_rush = pd.read_csv(os.path.join(ADV, "ngs", "ngs_rushing.csv.gz"))
    ngs_rush = ngs_rush[(ngs_rush["season_type"] == "REG") & (ngs_rush["week"] == 0)]
    ngs_rush = ngs_rush.rename(columns={"player_gsis_id": "player_id"})[
        ["player_id", "season", "efficiency", "percent_attempts_gte_eight_defenders",
         "avg_time_to_los", "expected_rush_yards", "rush_yards_over_expected",
         "rush_yards_over_expected_per_att", "rush_pct_over_expected"]].add_prefix("ngs_") \
        .rename(columns={"ngs_player_id": "player_id", "ngs_season": "season"})

    ngs_pass = pd.read_csv(os.path.join(ADV, "ngs", "ngs_passing.csv.gz"))
    ngs_pass = ngs_pass[(ngs_pass["season_type"] == "REG") & (ngs_pass["week"] == 0)]
    ngs_pass = ngs_pass.rename(columns={"player_gsis_id": "player_id"})[
        ["player_id", "season", "avg_time_to_throw", "avg_completed_air_yards",
         "avg_intended_air_yards", "avg_air_yards_differential", "aggressiveness",
         "avg_air_yards_to_sticks", "expected_completion_percentage",
         "completion_percentage_above_expectation", "avg_air_distance",
         "max_air_distance", "passer_rating"]].add_prefix("ngs_").rename(
        columns={"ngs_player_id": "player_id", "ngs_season": "season",
                 "ngs_avg_intended_air_yards": "ngs_avg_intended_air_yards_pass"})

    print("ESPN QBR")
    qbr = pd.read_csv(os.path.join(ADV, "espn", "qbr_season_level.csv"))
    qbr = qbr[qbr["season_type"] == "Regular"] if "Regular" in set(qbr["season_type"]) \
        else qbr[qbr["season_type"].astype(str).str.upper().str.startswith("REG")]
    # ESPN athlete id -> gsis_id via players.csv espn_id (exact); name fallback after
    ex = players.dropna(subset=["espn_id"]).drop_duplicates("espn_id")[["espn_id", "gsis_id"]]
    ex["espn_id"] = ex["espn_id"].astype("int64")
    qbr = qbr.merge(ex, left_on="player_id", right_on="espn_id", how="left")
    by_id = int(qbr["gsis_id"].notna().sum())
    miss = qbr["gsis_id"].isna()
    qbr.loc[miss, "gsis_id"] = qbr.loc[miss, "name_display"].map(norm_name).map(name_map)
    log_join("espn_qbr -> gsis_id", len(qbr), int(qbr["gsis_id"].notna().sum()),
             f"{by_id} by espn_id, rest by unique-name")
    qbr = qbr.drop(columns=["player_id", "espn_id"]).rename(columns={"gsis_id": "player_id"})
    qbr = qbr.dropna(subset=["player_id"]).drop_duplicates(["player_id", "season"])[
        ["player_id", "season", "qbr_total", "qbr_raw", "pts_added", "qb_plays",
         "epa_total"]].rename(columns={"qbr_total": "espn_qbr", "qbr_raw": "espn_qbr_raw",
                                       "pts_added": "espn_pts_added"})

    # ------------------------------------------------ position tables
    def attach(df, others, label):
        for name, o in others:
            before = len(df)
            key = ["player_id", "season"]
            probe = o.columns.difference(key)[0]
            clash = set(df.columns) & set(o.columns) - set(key)
            assert not clash, f"{label} <- {name} column collision: {sorted(clash)}"
            df = df.merge(o, on=key, how="left")
            assert len(df) == before, f"{name} exploded rows"
            log_join(f"{label} <- {name}", before, int(df[probe].notna().sum()))
        return df

    part_small = part[["player_id", "season", "off_plays_on_field",
                       "dropbacks_on_field", "rushes_on_field", "tm_off_plays",
                       "tm_dropbacks", "tm_rushes"]]
    sn_small = sn[["player_id", "season", "snap_games", "off_snaps",
                   "tm_off_snaps", "snap_pct_mean", "st_snaps"]]

    # --- WR/TE
    wrte = base[base["position"].isin(["WR", "TE"])].copy()
    wrte = attach(wrte, [("snap_counts", sn_small), ("participation", part_small),
                         ("pfr_rec", pfr_rec), ("ngs_receiving", ngs_rec),
                         ("pbp_receiving", pbp_rec.drop_duplicates(["player_id", "season"]))],
                  "wr_te")
    wrte = derive_wrte(wrte)
    wrte = wrte[wrte["targets"] >= 1].sort_values(["season", "targets"],
                                                  ascending=[True, False])

    # --- RB
    rb = base[base["position"].isin(["RB", "FB"])].copy()
    rb = attach(rb, [("snap_counts", sn_small), ("participation", part_small),
                     ("pfr_rush", pfr_rush), ("pfr_rec", pfr_rec),
                     ("ngs_rushing", ngs_rush),
                     ("pbp_rushing", pbp_rush.drop_duplicates(["player_id", "season"])),
                     ("pbp_receiving", pbp_rec.drop_duplicates(["player_id", "season"]))],
                "rb")
    rb = derive_rb(rb)
    rb = rb[(rb["carries"] >= 1) | (rb["targets"] >= 1)]
    rb = rb.sort_values(["season", "touches"], ascending=[True, False])

    # --- QB
    qbase = w.groupby(["player_id", "season"], as_index=False).agg(
        player_name=("player_display_name", "first"),
        position=("position", lambda s: s.mode().iat[0] if len(s.mode()) else np.nan),
        team=("team", lambda s: s.mode().iat[0] if len(s.mode()) else np.nan),
        games=("week", "nunique"),
        attempts=("attempts", "sum"), completions=("completions", "sum"),
        pass_yards=("passing_yards", "sum"), pass_tds=("passing_tds", "sum"),
        interceptions=("passing_interceptions", "sum"),
        sacks_suffered=("sacks_suffered", "sum"),
        pass_air_yards=("passing_air_yards", "sum"),
        pass_yac=("passing_yards_after_catch", "sum"),
        pass_epa=("passing_epa", "sum"), cpoe_w=("passing_cpoe", "mean"),
        carries=("carries", "sum"), rush_yards=("rushing_yards", "sum"),
        rush_tds=("rushing_tds", "sum"), rush_epa=("rushing_epa", "sum"),
        ppr=("fantasy_points_ppr", "sum"))
    qb = qbase[(qbase["position"] == "QB") & (qbase["attempts"] >= 1)].copy()
    qb = attach(qb, [("pfr_pass", pfr_pass), ("ngs_passing", ngs_pass),
                     ("pbp_qb", pbp_qb.drop_duplicates(["player_id", "season"])),
                     ("espn_qbr", qbr),
                     ("pbp_rushing", pbp_rush.drop_duplicates(["player_id", "season"])
                      [["player_id", "season", "pbp_carries", "gl5_carries",
                        "third_down_carries", "rush_epa_total"]])],
                "qb")
    qb = derive_qb(qb)
    qb = qb.sort_values(["season", "attempts"], ascending=[True, False])

    # ------------------------------------------------ team context
    print("team context")
    tc = team_context(all_pbp, tw, w)

    # ------------------------------------------------ write
    wrte.to_csv(os.path.join(DER, "adv_wr_te.csv"), index=False)
    rb.to_csv(os.path.join(DER, "adv_rb.csv"), index=False)
    qb.to_csv(os.path.join(DER, "adv_qb.csv"), index=False)
    tc.to_csv(os.path.join(DER, "team_context.csv"), index=False)

    # ------------------------------------------------ 3-season recency window
    for name, df, der in [("adv_wr_te", wrte, derive_wrte),
                          ("adv_rb", rb, derive_rb),
                          ("adv_qb", qb, derive_qb)]:
        r = recency_window(df, RECENT, der)
        r.to_csv(os.path.join(DER, f"{name}_recent3.csv"), index=False)
        print(f"  {name}_recent3: {len(r)} players, "
              f"thin_data={int(r['thin_data'].sum())}")

    # ------------------------------------------------ validation
    print("\nvalidation")
    allp = pd.concat([wrte, rb])
    ts = allp.groupby(["season", "team"])["target_share_full"].sum()
    ts2 = allp.groupby(["season", "team"])["target_share"].sum()
    print(f"  sum of target_share_full over WR/TE/RB: mean={ts.mean():.3f} "
          f"p05={ts.quantile(.05):.3f} p95={ts.quantile(.95):.3f} "
          "(want ~0.97-1.00; residual = QB/OL/ST receivers)")
    print(f"  sum of target_share (games-played denom): mean={ts2.mean():.3f} "
          "(expected > 1 by construction, see column notes)")
    agree = wrte.dropna(subset=["pbp_targets"])
    print(f"  weekly targets vs pbp targets: max |diff| = "
          f"{(agree['targets'] - agree['pbp_targets']).abs().max():.0f}, "
          f"corr = {agree['targets'].corr(agree['pbp_targets']):.5f}")
    for nm, df, key in [("wr_te", wrte, ["player_id", "season"]),
                        ("rb", rb, ["player_id", "season"]),
                        ("qb", qb, ["player_id", "season"]),
                        ("team", tc, ["season", "team"])]:
        assert not df.duplicated(key).any(), f"{nm} duplicate keys"
    print("  primary keys unique: ok")
    print(f"  team-seasons: {len(tc)} (want 32*{len(SEASONS)}={32*len(SEASONS)})")

    pd.DataFrame(JOIN_LOG).to_csv(os.path.join(DER, "adv_join_report.csv"), index=False)

    print("\nrows: wr_te=%d rb=%d qb=%d team=%d" % (len(wrte), len(rb), len(qb), len(tc)))


if __name__ == "__main__":
    main()

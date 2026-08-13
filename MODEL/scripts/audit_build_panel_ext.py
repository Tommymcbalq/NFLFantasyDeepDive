"""
Audit 3: a corrected and EXTENDED team-game panel.

Writes data/team_game_panel_ext.csv (seasons 2012-2025). The original
data/team_game_panel.csv is left untouched.

CORRECTIONS to the original builder
-----------------------------------
1. off_fumbles / off_ints were computed as `g.fumble.fillna(0).sum()`. On a
   SeriesGroupBy, .fillna() returns an ungrouped Series aligned to the original
   index, so .sum() collapses to a SCALAR -- the league-season total -- which
   pandas then broadcasts to every team-game. Both columns were constant within
   season. Fixed with g.fumble.sum() (min_count=0 on a pre-filled column).
2. Turnovers counted `fumble_lost` under posteam. On ~1% of scrimmage fumbles
   the fumbling team is NOT posteam (a defender fumbles after an interception),
   which is a takeaway back, not a giveaway. Now conditioned on
   fumbled_1_team == posteam, and interception/fumble double-counts removed.
3. Turnovers ignored special-teams giveaways (muffed punts, kick fumbles), ~11%
   of all lost fumbles. Carried separately as *_to_all so the choice is testable.
4. Pace was measured between consecutive plays of the FILTERED universe, so a
   drive containing a penalty (play_type == 'no_play', a real snap that consumes
   real clock) had that snap's time folded into the neighbouring interval.
   sec/play was therefore inflated and contaminated by penalty rate. Now measured
   between consecutive SNAP EVENTS (pass, run, no_play, kneel, spike).

NEW FEATURES, chosen to be plausibly orthogonal to mean EPA per play
--------------------------------------------------------------------
- explosiveness / disaster rates      (tails of the EPA distribution)
- within-game EPA dispersion          (consistency, a second moment)
- sack / QB-hit / TFL rates           (pressure proxies)
- early-down pass and rush EPA        (situational split)
- third down, red zone                (leverage splits; expected to be noise)
- drive-level: points, TD rate, three-and-out, field position, plays per drive
- penalties per play and penalty yards per play
- special-teams EPA per special-teams play
- air / YAC EPA split                 (does the pass game decompose usefully)
- series conversion rate              (nflfastR's own drive-efficiency measure)
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PBP_DIR = os.path.join(HERE, "..", "data", "pbp")
OUT = os.path.join(HERE, "..", "data", "team_game_panel_ext.csv")

FIRST = 2012

COLS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam",
    "home_team", "away_team", "play_id", "drive", "fixed_drive",
    "fixed_drive_result", "drive_play_count", "drive_inside20",
    "qtr", "down", "ydstogo", "yardline_100", "yards_gained",
    "epa", "success", "pass", "rush", "qb_kneel", "qb_spike", "aborted_play",
    "cpoe", "pass_oe", "game_seconds_remaining", "half_seconds_remaining",
    "score_differential", "interception", "fumble_lost", "fumble",
    "fumbled_1_team", "play_type", "qb_dropback", "sack", "qb_hit",
    "tackled_for_loss", "first_down", "third_down_converted",
    "third_down_failed", "penalty", "penalty_team", "penalty_yards",
    "air_epa", "yac_epa", "series", "series_success", "special",
]

NEUTRAL_SCORE = 7
NEUTRAL_MAX_QTR = 3
MIN_HALF_SECS = 120
PACE_LO, PACE_HI = 5, 60

EXPL_EPA = 1.0        # explosive play threshold on EPA
DISASTER_EPA = -1.0   # disaster play threshold on EPA


# ----------------------------------------------------------------------------
def snap_pace(raw: pd.DataFrame) -> pd.DataFrame:
    """Seconds between consecutive SNAP EVENTS within a drive.

    A penalty-negated play (play_type == 'no_play') is a snap: it was run and it
    consumed clock. Excluding it, as the original builder did, folds its duration
    into the adjacent interval and inflates sec/play on penalty-heavy drives.
    """
    snap = raw.play_type.isin(["pass", "run", "no_play", "qb_kneel", "qb_spike"])
    s = raw[snap].sort_values(["game_id", "drive", "play_id"]).copy()
    same = (s.game_id == s.game_id.shift(-1)) & (s.drive == s.drive.shift(-1))
    delta = s.game_seconds_remaining - s.game_seconds_remaining.shift(-1)
    s["gap"] = np.where(same, delta, np.nan)
    s.loc[(s.gap < PACE_LO) | (s.gap > PACE_HI), "gap"] = np.nan
    neutral = (s.score_differential.abs().le(NEUTRAL_SCORE)
               & s.qtr.le(NEUTRAL_MAX_QTR)
               & s.half_seconds_remaining.ge(MIN_HALF_SECS))
    s["gap_neutral"] = s.gap.where(neutral)
    return s[["game_id", "play_id", "gap", "gap_neutral"]]


def drive_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, offence, drive) with result and starting field pos."""
    d = raw[raw.fixed_drive.notna() & raw.posteam.notna()].copy()
    first = (d.sort_values(["game_id", "fixed_drive", "play_id"])
               .groupby(["game_id", "fixed_drive"], observed=True)
               .agg(posteam=("posteam", "first"), defteam=("defteam", "first"),
                    start_yl100=("yardline_100", "first"),
                    result=("fixed_drive_result", "first"),
                    plays=("drive_play_count", "first"),
                    inside20=("drive_inside20", "max"))
               .reset_index())
    r = first.result
    first["pts"] = np.select(
        [r == "Touchdown", r == "Field goal", r == "Opp touchdown", r == "Safety"],
        [7.0, 3.0, -7.0, -2.0], default=0.0)
    first["is_td"] = (r == "Touchdown").astype(float)
    first["is_score"] = r.isin(["Touchdown", "Field goal"]).astype(float)
    first["is_to"] = r.isin(["Turnover", "Opp touchdown"]).astype(float)
    first["three_out"] = ((first.plays <= 3) & (r == "Punt")).astype(float)
    # drives that simply ran out of clock say nothing about efficiency
    first["live"] = (~r.isin(["End of half"])).astype(float)
    return first


def drive_agg(dr: pd.DataFrame, team_col: str, prefix: str) -> pd.DataFrame:
    d = dr[dr.live == 1]
    g = d.groupby(["game_id", team_col], observed=True)
    out = pd.DataFrame({
        f"{prefix}_drives": g.size(),
        f"{prefix}_pts_per_drive": g.pts.mean(),
        f"{prefix}_td_per_drive": g.is_td.mean(),
        f"{prefix}_score_rate": g.is_score.mean(),
        f"{prefix}_to_per_drive": g.is_to.mean(),
        f"{prefix}_three_out_rate": g.three_out.mean(),
        f"{prefix}_start_yl100": g.start_yl100.mean(),
        f"{prefix}_plays_per_drive": g.plays.mean(),
        f"{prefix}_rz_rate": g.inside20.mean(),
    })
    rz = d[d.inside20 == 1].groupby(["game_id", team_col], observed=True)
    out[f"{prefix}_rz_td_rate"] = rz.is_td.mean()
    out[f"{prefix}_rz_drives"] = rz.size()
    return out.reset_index().rename(columns={team_col: "team"})


def scrimmage_agg(p: pd.DataFrame, team_col: str, prefix: str) -> pd.DataFrame:
    g = p.groupby(["game_id", team_col], observed=True)
    out = pd.DataFrame({
        f"{prefix}_plays": g.epa.size(),
        f"{prefix}_epa": g.epa.mean(),
        f"{prefix}_sr": g.success.mean(),
        f"{prefix}_epa_sd": g.epa.std(),
        f"{prefix}_expl_rate": g.expl.mean(),
        f"{prefix}_disaster_rate": g.disaster.mean(),
        f"{prefix}_first_down_rate": g.first_down.mean(),
        f"{prefix}_tfl_rate": g.tackled_for_loss.mean(),
        f"{prefix}_giveaways": g.giveaway.sum(),
        f"{prefix}_ints": g.int_ev.sum(),
        f"{prefix}_fumbles": g.fum_ev.sum(),
        f"{prefix}_fumbles_lost": g.fum_lost_ev.sum(),
    })

    noto = p[p.giveaway == 0]
    gnt = noto.groupby(["game_id", team_col], observed=True)
    out[f"{prefix}_epa_noto"] = gnt.epa.mean()
    out[f"{prefix}_sr_noto"] = gnt.success.mean()

    early = p[p.down.isin([1, 2])]
    ge = early.groupby(["game_id", team_col], observed=True)
    out[f"{prefix}_early_epa"] = ge.epa.mean()
    out[f"{prefix}_early_sr"] = ge.success.mean()
    out[f"{prefix}_early_plays"] = ge.epa.size()
    gep = early[early["pass"] == 1].groupby(["game_id", team_col], observed=True)
    ger = early[early["pass"] != 1].groupby(["game_id", team_col], observed=True)
    out[f"{prefix}_early_pass_epa"] = gep.epa.mean()
    out[f"{prefix}_early_pass_plays"] = gep.epa.size()
    out[f"{prefix}_early_rush_epa"] = ger.epa.mean()
    out[f"{prefix}_early_rush_plays"] = ger.epa.size()

    third = p[p.down == 3]
    g3 = third.groupby(["game_id", team_col], observed=True)
    out[f"{prefix}_third_epa"] = g3.epa.mean()
    out[f"{prefix}_third_plays"] = g3.epa.size()
    out[f"{prefix}_third_conv"] = g3.third_down_converted.mean()

    is_pass = p["pass"] == 1
    gp = p[is_pass].groupby(["game_id", team_col], observed=True)
    gr = p[~is_pass].groupby(["game_id", team_col], observed=True)
    out[f"{prefix}_pass_plays"] = gp.epa.size()
    out[f"{prefix}_pass_epa"] = gp.epa.mean()
    out[f"{prefix}_pass_sr"] = gp.success.mean()
    out[f"{prefix}_pass_expl_rate"] = gp.expl.mean()
    out[f"{prefix}_air_epa"] = gp.air_epa.mean()
    out[f"{prefix}_yac_epa"] = gp.yac_epa.mean()
    out[f"{prefix}_rush_plays"] = gr.epa.size()
    out[f"{prefix}_rush_epa"] = gr.epa.mean()
    out[f"{prefix}_rush_sr"] = gr.success.mean()
    out[f"{prefix}_rush_expl_rate"] = gr.expl.mean()
    out[f"{prefix}_cpoe"] = gp.cpoe.mean()
    out[f"{prefix}_pass_oe"] = g.pass_oe.mean()

    db = p[p.qb_dropback == 1]
    gd = db.groupby(["game_id", team_col], observed=True)
    out[f"{prefix}_dropbacks"] = gd.epa.size()
    out[f"{prefix}_sack_rate"] = gd.sack.mean()
    out[f"{prefix}_qb_hit_rate"] = gd.qb_hit.mean()
    out[f"{prefix}_dropback_epa"] = gd.epa.mean()

    n = p[p.gap_neutral.notna()]
    gn = n.groupby(["game_id", team_col], observed=True)
    out[f"{prefix}_sec_per_play"] = gn.gap_neutral.mean()
    out[f"{prefix}_neutral_plays"] = gn.epa.size()

    return out.reset_index().rename(columns={team_col: "team"})


def series_agg(raw: pd.DataFrame, team_col: str, prefix: str) -> pd.DataFrame:
    d = raw[raw.series.notna() & raw.posteam.notna()
            & raw.series_success.notna()].copy()
    f = (d.sort_values(["game_id", "series", "play_id"])
           .groupby(["game_id", "series"], observed=True)
           .agg(posteam=("posteam", "first"), defteam=("defteam", "first"),
                succ=("series_success", "max")).reset_index())
    g = f.groupby(["game_id", team_col], observed=True)
    out = pd.DataFrame({f"{prefix}_series_conv": g.succ.mean(),
                        f"{prefix}_series": g.size()})
    return out.reset_index().rename(columns={team_col: "team"})


def penalty_agg(raw: pd.DataFrame) -> pd.DataFrame:
    """Penalties charged to a team, per its own offensive+defensive snap count."""
    d = raw[(raw.penalty == 1) & raw.penalty_team.notna()]
    g = d.groupby(["game_id", "penalty_team"], observed=True)
    out = pd.DataFrame({"pen_n": g.size(), "pen_yds": g.penalty_yards.sum()})
    return out.reset_index().rename(columns={"penalty_team": "team"})


def st_agg(raw: pd.DataFrame, team_col: str, prefix: str) -> pd.DataFrame:
    d = raw[(raw.special == 1) & raw.epa.notna() & raw.posteam.notna()]
    g = d.groupby(["game_id", team_col], observed=True)
    out = pd.DataFrame({f"{prefix}_st_epa": g.epa.mean(), f"{prefix}_st_plays": g.size()})
    return out.reset_index().rename(columns={team_col: "team"})


def turnover_all(raw: pd.DataFrame, team_col: str, prefix: str) -> pd.DataFrame:
    """All giveaways including special teams, attributed to the losing team."""
    d = raw.copy()
    d["int_ev"] = (d.interception.fillna(0) == 1).astype(float)
    fl = (d.fumble_lost.fillna(0) == 1) & (d.fumbled_1_team == d.posteam)
    d["give"] = np.maximum(d.int_ev, fl.astype(float))
    d = d[d.posteam.notna()]
    g = d.groupby(["game_id", team_col], observed=True)
    out = pd.DataFrame({f"{prefix}_to_all": g.give.sum()})
    return out.reset_index().rename(columns={team_col: "team"})


def build_season(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False, compression="gzip",
                      usecols=lambda c: c in set(COLS))
    for c in ("interception", "fumble_lost", "fumble", "sack", "qb_hit",
              "tackled_for_loss", "first_down", "third_down_converted",
              "penalty", "penalty_yards", "qb_dropback", "special"):
        raw[c] = raw[c].fillna(0)

    pace = snap_pace(raw)
    raw = raw.merge(pace, on=["game_id", "play_id"], how="left")

    keep = (raw.epa.notna() & raw.posteam.notna() & raw.defteam.notna()
            & ((raw["pass"] == 1) | (raw["rush"] == 1))
            & (raw.qb_kneel.fillna(0) != 1) & (raw.qb_spike.fillna(0) != 1)
            & (raw.aborted_play.fillna(0) != 1))
    p = raw.loc[keep].copy()

    # ---- corrected turnover / fumble events on scrimmage plays -------------
    p["int_ev"] = (p.interception == 1).astype(float)
    p["fum_ev"] = ((p.fumble == 1) & (p.fumbled_1_team == p.posteam)).astype(float)
    p["fum_lost_ev"] = ((p.fumble_lost == 1) & (p.fumbled_1_team == p.posteam)
                        & (p.interception != 1)).astype(float)
    p["giveaway"] = np.maximum(p.int_ev, p.fum_lost_ev)
    p["expl"] = (p.epa >= EXPL_EPA).astype(float)
    p["disaster"] = (p.epa <= DISASTER_EPA).astype(float)

    frames = []
    for team_col, prefix in [("posteam", "off"), ("defteam", "def")]:
        f = scrimmage_agg(p, team_col, prefix)
        f = f.merge(series_agg(raw, team_col, prefix), on=["game_id", "team"], how="left")
        f = f.merge(st_agg(raw, team_col, prefix), on=["game_id", "team"], how="left")
        f = f.merge(turnover_all(raw, team_col, prefix), on=["game_id", "team"], how="left")
        frames.append(f)
    dr = drive_table(raw)
    for i, (team_col, prefix) in enumerate([("posteam", "off"), ("defteam", "def")]):
        frames[i] = frames[i].merge(drive_agg(dr, team_col, prefix),
                                    on=["game_id", "team"], how="left")

    panel = frames[0].merge(frames[1], on=["game_id", "team"], how="outer")
    pen = penalty_agg(raw)
    panel = panel.merge(pen, on=["game_id", "team"], how="left")
    panel[["pen_n", "pen_yds"]] = panel[["pen_n", "pen_yds"]].fillna(0.0)

    meta = (p.groupby("game_id", observed=True)
              .agg(season=("season", "first"), week=("week", "first"),
                   season_type=("season_type", "first"),
                   home_team=("home_team", "first"), away_team=("away_team", "first"))
              .reset_index())
    panel = panel.merge(meta, on="game_id", how="left")
    panel["is_home"] = (panel.team == panel.home_team).astype(int)
    panel["opponent"] = np.where(panel.is_home == 1, panel.away_team, panel.home_team)
    return panel


def main():
    paths = sorted(glob.glob(os.path.join(PBP_DIR, "play_by_play_*.csv.gz")))
    paths = [p for p in paths if int(os.path.basename(p)[13:17]) >= FIRST]
    frames = []
    for path in paths:
        df = build_season(path)
        frames.append(df)
        print(f"{os.path.basename(path)}: {df.game_id.nunique()} games, "
              f"{len(df)} team-games", flush=True)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["season", "week", "game_id", "is_home"])
    panel.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}: {panel.shape}")


if __name__ == "__main__":
    main()

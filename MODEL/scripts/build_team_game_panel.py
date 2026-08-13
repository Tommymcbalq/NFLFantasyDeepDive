"""
Aggregate nflverse play-by-play to a team-game panel.

Two rows per game (one per team). Each row holds that team's OFFENSIVE production
in that game and the production it ALLOWED on defense. Everything here is
realized in-game data -- it is NOT yet a feature. The leak-free rolling
transformation happens downstream in build_features.py.

Play universe (standard nflfastR convention):
    epa non-null, posteam non-null, (pass or rush), not a kneel/spike,
    not an aborted play.
Kneels and spikes are clock-management plays carrying large negative EPA that
say nothing about team quality; including them would penalise teams for
being ahead.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

PBP_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pbp")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "team_game_panel.csv")

COLS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam",
    "home_team", "away_team", "play_id", "drive", "qtr", "down",
    "epa", "success", "pass", "rush", "qb_kneel", "qb_spike", "aborted_play",
    "cpoe", "pass_oe", "game_seconds_remaining", "half_seconds_remaining",
    "score_differential", "interception", "fumble_lost", "fumble", "play_type",
]

# Neutral game script: pace is otherwise mostly a measure of who was winning.
NEUTRAL_SCORE = 7      # |score differential| <= 7
NEUTRAL_MAX_QTR = 3    # quarters 1-3 only
MIN_HALF_SECS = 120    # drop two-minute drill
PACE_LO, PACE_HI = 5, 60   # plausible seconds between snaps


def _seconds_per_play(plays: pd.DataFrame) -> pd.DataFrame:
    """Seconds elapsed between consecutive snaps of the same drive.

    Time between snap i and snap i+1 = play duration + huddle + snap. Measured
    off game_seconds_remaining at the snap, within a drive so that changes of
    possession, timeouts between drives, and quarter breaks do not contaminate.
    """
    p = plays.sort_values(["game_id", "drive", "play_id"]).copy()
    same_drive = (p.game_id == p.game_id.shift(-1)) & (p.drive == p.drive.shift(-1))
    delta = p.game_seconds_remaining - p.game_seconds_remaining.shift(-1)
    p["sec_per_play"] = np.where(same_drive, delta, np.nan)
    p.loc[(p.sec_per_play < PACE_LO) | (p.sec_per_play > PACE_HI), "sec_per_play"] = np.nan

    neutral = (
        p.score_differential.abs().le(NEUTRAL_SCORE)
        & p.qtr.le(NEUTRAL_MAX_QTR)
        & p.half_seconds_remaining.ge(MIN_HALF_SECS)
    )
    p["sec_per_play_neutral"] = p.sec_per_play.where(neutral)
    p["neutral_play"] = neutral
    return p


def _side_aggregate(p: pd.DataFrame, team_col: str, prefix: str) -> pd.DataFrame:
    """Aggregate plays by (game, team) where team_col identifies the team.

    Called twice: once with posteam (the team's offence) and once with defteam
    (what the team's defence allowed). The statistics computed are identical --
    only the grouping side differs -- so offence and defence land on one scale.
    """
    g = p.groupby(["game_id", team_col], observed=True)

    out = pd.DataFrame({
        f"{prefix}_plays":     g.epa.size(),
        f"{prefix}_epa":       g.epa.mean(),
        f"{prefix}_sr":        g.success.mean(),
        f"{prefix}_turnovers": g.apply(
            lambda x: (x.interception.fillna(0) + x.fumble_lost.fillna(0)).sum(),
            include_groups=False),
    })

    # --- turnover-neutral EPA -------------------------------------------
    # A turnover is a huge negative EPA play, so total EPA already contains the
    # turnover result. That makes turnover margin ~55% collinear with EPA and
    # unable to contribute anything of its own. Stripping turnover plays out
    # leaves "how well did they move the ball", and lets turnover margin enter
    # as a genuinely separate signal rather than a second helping of the same one.
    noto = p[(p.interception.fillna(0) != 1) & (p.fumble_lost.fillna(0) != 1)]
    gnt = noto.groupby(["game_id", team_col], observed=True)
    out[f"{prefix}_epa_noto"] = gnt.epa.mean()
    out[f"{prefix}_sr_noto"] = gnt.success.mean()

    # --- early downs -----------------------------------------------------
    # 1st and 2nd down. Third down is high-leverage but low-volume and heavily
    # situation-dependent (distance, score, clock), so all-down EPA mixes a
    # stable signal with a noisy one. Early-down EPA is the cleaner read on
    # offensive quality.
    early = p[p.down.isin([1, 2])]
    ge = early.groupby(["game_id", team_col], observed=True)
    out[f"{prefix}_early_epa"] = ge.epa.mean()
    out[f"{prefix}_early_sr"] = ge.success.mean()
    out[f"{prefix}_early_plays"] = ge.epa.size()

    # --- turnover components, split by how lucky each one is -------------
    # Fumble RECOVERY is close to a coin flip, so fumbles LOST is a noisy
    # measure of a team's real ball-security problem. Fumbles COMMITTED
    # (recovered or not) is the skill part and predicts future fumbles lost
    # better than past fumbles lost does. Carried separately so the model can
    # weight the skill and luck components differently.
    out[f"{prefix}_fumbles"] = g.fumble.fillna(0).sum()
    out[f"{prefix}_ints"] = g.interception.fillna(0).sum()

    is_pass = p["pass"] == 1
    gp = p[is_pass].groupby(["game_id", team_col], observed=True)
    gr = p[~is_pass].groupby(["game_id", team_col], observed=True)
    out[f"{prefix}_pass_plays"] = gp.epa.size()
    out[f"{prefix}_pass_epa"]   = gp.epa.mean()
    out[f"{prefix}_pass_sr"]    = gp.success.mean()
    out[f"{prefix}_rush_plays"] = gr.epa.size()
    out[f"{prefix}_rush_epa"]   = gr.epa.mean()
    out[f"{prefix}_rush_sr"]    = gr.success.mean()
    out[f"{prefix}_cpoe"]       = gp.cpoe.mean()
    out[f"{prefix}_pass_oe"]    = g.pass_oe.mean()

    # Pace: only meaningful for the offence (the unit snapping the ball), but we
    # compute it on both sides -- a team's defensive pace-faced is a real
    # exposure (more opponent snaps = more variance) and costs nothing to carry.
    n = p[p.neutral_play]
    gn = n.groupby(["game_id", team_col], observed=True)
    out[f"{prefix}_sec_per_play"]   = gn.sec_per_play_neutral.mean()
    out[f"{prefix}_neutral_plays"]  = gn.epa.size()

    return out.reset_index().rename(columns={team_col: "team"})


def build_season(path: str) -> pd.DataFrame:
    p = pd.read_csv(path, low_memory=False, usecols=lambda c: c in set(COLS),
                    compression="gzip")

    keep = (
        p.epa.notna() & p.posteam.notna() & p.defteam.notna()
        & ((p["pass"] == 1) | (p["rush"] == 1))
        & (p.qb_kneel.fillna(0) != 1) & (p.qb_spike.fillna(0) != 1)
        & (p.get("aborted_play", pd.Series(0, index=p.index)).fillna(0) != 1)
    )
    p = p.loc[keep].copy()
    p = _seconds_per_play(p)

    off = _side_aggregate(p, "posteam", "off")
    dfn = _side_aggregate(p, "defteam", "def")
    panel = off.merge(dfn, on=["game_id", "team"], how="outer")

    meta = (p.groupby("game_id", observed=True)
              .agg(season=("season", "first"), week=("week", "first"),
                   season_type=("season_type", "first"),
                   home_team=("home_team", "first"), away_team=("away_team", "first"))
              .reset_index())
    panel = panel.merge(meta, on="game_id", how="left")
    panel["is_home"] = (panel.team == panel.home_team).astype(int)
    panel["opponent"] = np.where(panel.is_home == 1, panel.away_team, panel.home_team)
    return panel


def main() -> None:
    paths = sorted(glob.glob(os.path.join(PBP_DIR, "play_by_play_*.csv.gz")))
    if not paths:
        sys.exit(f"no pbp files in {PBP_DIR}")
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

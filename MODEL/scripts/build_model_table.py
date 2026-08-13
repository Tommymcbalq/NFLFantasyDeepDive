r"""
Assemble the game-level modelling table.

Team quality estimates (leak-free, opponent-adjusted, shrunk) are merged onto
each game for both sides and converted to differentials.

Why differentials, and the exact sign convention
------------------------------------------------
Write O_t for a team's offensive quality and D_t for its defensive quality
(negative D = good defence, since D is how much a defence adds to the offence it
faces). Expected home offensive efficiency in this matchup is O_h + D_a, and
expected away offensive efficiency is O_a + D_h. The net home edge is

    (O_h + D_a) - (O_a + D_h)  =  (O_h - O_a)  +  (D_a - D_h)
                                   \________/     \________/
                                    off_diff       def_diff

so the matchup decomposes exactly into an offensive differential and a defensive
differential. Both are oriented so that POSITIVE FAVOURS THE HOME TEAM:
off_diff is high when the home offence is better, and def_diff is high when the
away defence is worse. Under the additive model the two should carry equal
coefficients; whether they actually do is tested in diag_antisymmetry.py.

Rest
----
Raw rest days are not used. The distribution is 66% at exactly 7 days, and the
cells at 5, 9, 11, 12 days hold under 1% of team-games each -- fitting a
coefficient to those is fitting noise. Four levels only:

    short   <= 5 days   (Sunday -> Thursday)
    normal  6-8 days    (the baseline, ~81% of team-games)
    mini_bye 9-11 days  (Thursday -> Sunday)
    bye     >= 12 days

Monday -> Thursday does not exist: zero team-games at 3 days rest in the whole
2014-2024 window. The league does not schedule it, so it is not a level.
Each is entered as a home-minus-away contrast, giving three parameters.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
GAMES = os.path.join(HERE, "..", "data", "games", "games.csv")
QUALITY = os.environ.get("QUALITY_FILE",
                         os.path.join(HERE, "..", "data", "team_quality.csv"))
OUT = os.environ.get("MODEL_TABLE_OUT",
                     os.path.join(HERE, "..", "data", "model_table.csv"))

METRICS = ["epa", "pass_epa", "rush_epa", "sr", "pass_sr", "rush_sr",
           "cpoe", "pass_oe", "sec_per_play", "to_rate",
           "epa_noto", "sr_noto", "early_epa", "early_sr",
           "fumble_rate", "int_rate"]

# Metrics where a HIGHER defensive coefficient means a BETTER defence, not a
# worse one. For EPA, D is what a defence lets the opposing offence gain, so
# high D = bad. For turnover rates, D is the turnovers that defence FORCES, so
# high D = good. The generic "positive favours home" differential is therefore
# backwards for these, and the composites below are built explicitly instead.
DEFENCE_HIGHER_IS_BETTER = {"to_rate", "fumble_rate", "int_rate"}

FIRST_SEASON = 2014

# The schedules table records the franchise code in use AT THE TIME, while the
# play-by-play applies current codes retroactively. Left unmapped this silently
# drops every Rams game through 2015, Chargers through 2016 and Raiders through
# 2019 -- 168 games, all from the seasons those franchises relocated.
TEAM_FIXES = {"STL": "LA", "SD": "LAC", "OAK": "LV", "LAR": "LA", "JAC": "JAX"}


def rest_level(days: pd.Series) -> pd.Series:
    return pd.cut(days, bins=[-np.inf, 5, 8, 11, np.inf],
                  labels=["short", "normal", "mini_bye", "bye"])


def main() -> None:
    g = pd.read_csv(GAMES)
    g = g[(g.game_type == "REG") & g.result.notna() & (g.season >= FIRST_SEASON)].copy()
    for col in ("home_team", "away_team"):
        g[col] = g[col].replace(TEAM_FIXES)

    q = pd.read_csv(QUALITY)
    home = q.add_prefix("h_").rename(columns={"h_season": "season", "h_week": "week",
                                              "h_team": "home_team"})
    away = q.add_prefix("a_").rename(columns={"a_season": "season", "a_week": "week",
                                              "a_team": "away_team"})
    n0 = len(g)
    g = g.merge(home, on=["season", "week", "home_team"], how="inner")
    g = g.merge(away, on=["season", "week", "away_team"], how="inner")
    print(f"games {n0} -> {len(g)} after quality merge "
          f"({n0 - len(g)} dropped for missing features)")

    # ---- quality differentials, all oriented positive = favours home ----
    for m in METRICS:
        g[f"off_{m}_diff"] = g[f"h_O_{m}"] - g[f"a_O_{m}"]
        g[f"def_{m}_diff"] = g[f"a_D_{m}"] - g[f"h_D_{m}"]

    # Overall quality gap, used for the pace interaction below.
    g["quality_diff"] = g.off_epa_diff + g.def_epa_diff

    # ---- pace ----
    # pace_diff: the raw main effect, kept and tested on its own merits.
    # pace_sum:  total expected tempo of the game. Lower sec/play = faster, so
    #            negate it so that higher = faster and the interaction reads
    #            naturally: a fast game means more plays, less noise in the
    #            margin, and therefore the better team winning more often.
    g["pace_diff"] = g.off_sec_per_play_diff
    g["pace_sum"] = -((g.h_O_sec_per_play + g.a_O_sec_per_play) / 2.0)
    g["pace_x_quality"] = g.pace_sum * g.quality_diff

    # ---- mismatch interaction: does an elite pass offence compound against a
    # bad pass defence, beyond the additive sum of the two edges? ----
    g["home_pass_mismatch"] = g.h_O_pass_epa * g.a_D_pass_epa
    g["away_pass_mismatch"] = g.a_O_pass_epa * g.h_D_pass_epa
    g["pass_mismatch_diff"] = g.home_pass_mismatch - g.away_pass_mismatch

    # ---- turnover margins: takeaways minus giveaways, home minus away ----
    # Entered as a MARGIN rather than as separate giveaway/takeaway terms
    # because the margin is what actually swings a game, and because the
    # composite tested better than the two components entered separately.
    # Positive favours home in all three.
    for src, name in [("to_rate", "to_margin"), ("fumble_rate", "fumble_margin"),
                      ("int_rate", "int_margin")]:
        g[f"{name}_diff"] = -(g[f"off_{src}_diff"] + g[f"def_{src}_diff"])

    # ---- rest ----
    g["home_rest_lvl"] = rest_level(g.home_rest)
    g["away_rest_lvl"] = rest_level(g.away_rest)
    for lvl in ["short", "mini_bye", "bye"]:      # 'normal' is the baseline
        g[f"rest_{lvl}_diff"] = ((g.home_rest_lvl == lvl).astype(int)
                                 - (g.away_rest_lvl == lvl).astype(int))

    # ---- context ----
    g["is_neutral"] = (g.location != "Home").astype(int)
    g["div_game"] = g.div_game.fillna(0).astype(int)
    g["div_x_quality"] = g.div_game * g.quality_diff
    g["season_c"] = g.season - 2019                # era drift in home-field

    # ---- label and market benchmark ----
    g["home_win"] = np.where(g.result > 0, 1.0, np.where(g.result < 0, 0.0, 0.5))
    g["margin"] = g.result
    # FFC/nflverse convention: spread_line is points the HOME team is favoured by.
    g["spread_home"] = g.spread_line

    keep = (["game_id", "season", "week", "home_team", "away_team",
             "home_win", "margin", "spread_home", "total_line",
             "is_neutral", "div_game", "div_x_quality", "season_c",
             "quality_diff", "pace_diff", "pace_sum",
             "pace_x_quality", "pass_mismatch_diff",
             "to_margin_diff", "fumble_margin_diff", "int_margin_diff",
             "rest_short_diff", "rest_mini_bye_diff", "rest_bye_diff"]
            + [f"off_{m}_diff" for m in METRICS]
            + [f"def_{m}_diff" for m in METRICS]
            # raw per-side levels, kept so diag_antisymmetry.py can fit the
            # unconstrained parameterisation and test whether collapsing to
            # differentials is justified rather than assumed
            + [f"{s}_{k}_{m}" for m in ["epa", "pass_epa"]
               for s in ["h", "a"] for k in ["O", "D"]])
    out = g[keep].copy()
    out.to_csv(OUT, index=False)

    print(f"wrote {OUT}: {out.shape}")
    print(f"seasons {out.season.min()}-{out.season.max()}, "
          f"home win rate {out.home_win.mean():.4f}")
    print(f"missing spread_line: {out.spread_home.isna().sum()}")
    print("\nrest level counts (home side):")
    print(rest_level(g.home_rest).value_counts().to_string())


if __name__ == "__main__":
    main()

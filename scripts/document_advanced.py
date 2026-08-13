#!/usr/bin/env python3
"""
document_advanced.py — emit results/advanced_stats_notes.md.

Every column of every derived table gets a source and a definition. The script
asserts completeness: if build_advanced.py adds a column and no definition is
supplied here, this fails rather than shipping an undocumented column.
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DER = os.path.join(ROOT, "data", "derived")
RES = os.path.join(ROOT, "results")

TABLES = ["adv_wr_te", "adv_rb", "adv_qb", "team_context",
          "adv_wr_te_recent3", "adv_rb_recent3", "adv_qb_recent3"]

# source tags
W = "nflverse stats_player (weekly)"
TW = "nflverse stats_team (weekly)"
P = "nflverse pbp"
PT = "nflverse pbp_participation"
SN = "nflverse snap_counts (PFR)"
PR = "PFR advstats_season_rec"
PU = "PFR advstats_season_rush"
PP = "PFR advstats_season_pass"
NR = "NGS ngs_receiving"
NU = "NGS ngs_rushing"
NP = "NGS ngs_passing"
QBR = "ESPN qbr_season_level"
G = "nflverse schedules/games"
FT = "nflverse ftn_charting"
D = "derived"

DEFS = {
    # ---- keys / identity
    "player_id": (W, "gsis_id. Primary join key across every table."),
    "player_name": (W, "Display name, first appearance in the window."),
    "position": (W, "Modal weekly position over the player-season (last season in a window)."),
    "team": (W, "Modal weekly team over the player-season. Traded players get one team; n_teams flags them."),
    "team_last": (W, "Team of the most recent season in the window."),
    "n_teams": (W, "Distinct teams the player recorded a game for in the period."),
    "season": (W, "NFL season (REG only throughout)."),
    "games": (W, "Distinct REG weeks with a stat line."),
    "seasons_played": (W, "Distinct seasons with a stat line inside the window."),
    "first_season": (W, "Earliest season in the window."),
    "last_season": (W, "Latest season in the window."),
    "window_seasons": (D, "Window label, e.g. '2023-2025'."),
    "thin_data": (D, "TRUE if seasons_played < 2 or games < 8 in the window. Rookies and "
                     "players absent from the window are simply NOT rows here -- nothing is "
                     "imputed; the caller supplies them from ADP/draft capital."),

    # ---- receiving volume (weekly)
    "targets": (W, "Targets. Excludes two-point conversions."),
    "receptions": (W, "Receptions."),
    "rec_yards": (W, "Receiving yards."),
    "rec_tds": (W, "Receiving touchdowns."),
    "rec_air_yards": (W, "Sum of air yards on targets."),
    "rec_yac": (W, "Receiving yards after catch."),
    "rec_first_downs": (W, "Receiving first downs."),
    "rec_epa": (W, "Sum of receiving EPA."),
    "ppr": (W, "Total PPR fantasy points."),
    "carries": (W, "Rush attempts."),
    "rush_yards": (W, "Rushing yards."),
    "rush_tds": (W, "Rushing touchdowns."),
    "touches": (D, "carries + receptions."),
    "air_yards_pg": (D, "rec_air_yards / games."),
    "rec_pg": (D, "receptions / games."),
    "routes_proxy_pg": (D, "routes_proxy / games. Proxy routes run per game."),
    "designed_rushes_pg": (D, "designed_rushes / games. QB designed-run volume per game."),
    "rush_ppr_pg": (D, "rush_ppr / games."),
    "touches_pg": (D, "touches / games."),

    # ---- team denominators
    "tm_targets": (TW, "Team targets summed over the games the player was active for."),
    "tm_air_yards": (TW, "Team air yards over the player's active games."),
    "tm_pass_att": (TW, "Team pass attempts over the player's active games."),
    "tm_carries": (TW, "Team rush attempts over the player's active games."),
    "tm_targets_full": (TW, "Team targets over the FULL season."),
    "tm_air_yards_full": (TW, "Team air yards over the full season."),
    "tm_carries_full": (TW, "Team rush attempts over the full season."),
    "tm_games": (TW, "Team games in the season (16 pre-2021, 17 from 2021)."),

    # ---- shares
    "target_share": (D, "targets / tm_targets. Share WHILE ACTIVE. Does not sum to 1 within a "
                        "team-season; use for projection, not for budget arithmetic."),
    "air_yards_share": (D, "rec_air_yards / tm_air_yards, active-games denominator."),
    "carry_share": (D, "carries / tm_carries, active-games denominator."),
    "target_share_full": (D, "targets / tm_targets_full. Sums to ~1 across a team-season; use "
                             "for vacated-share / budget arithmetic. Understates part-season players."),
    "air_yards_share_full": (D, "rec_air_yards / tm_air_yards_full."),
    "carry_share_full": (D, "carries / tm_carries_full."),
    "games_played_rate": (D, "games / tm_games. Availability."),
    "wopr": (D, "1.5*target_share + 0.7*air_yards_share (active-games denominators)."),
    "racr": (D, "rec_yards / rec_air_yards."),
    "adot_nflverse": (D, "rec_air_yards / targets. nflverse-derived aDOT; compare pfr_adot."),
    "catch_rate": (D, "receptions / targets."),
    "rec_epa_per_target": (D, "rec_epa / targets."),
    "ypc": (D, "rush_yards / carries."),
    "ypr": (D, "rec_yards / receptions."),

    # ---- snaps
    "snap_games": (SN, "Games with >0 offensive snaps."),
    "off_snaps": (SN, "Offensive snaps."),
    "tm_off_snaps": (SN, "Team offensive snaps in the games the player was dressed "
                         "(max offense_snaps among the team's players that game)."),
    "snap_share": (D, "off_snaps / tm_off_snaps."),
    "snap_pct_mean": (SN, "Unweighted mean of PFR's per-game offense_pct. Kept as a cross-check "
                          "on snap_share; games-weighted mean in a window."),
    "st_snaps": (SN, "Special-teams snaps. Low ST usage is a weak signal of offensive role."),

    # ---- participation (route proxy)
    "off_plays_on_field": (PT, "Offensive plays with the player in offense_players."),
    "dropbacks_on_field": (PT, "Team dropbacks with the player on the field. ROUTE PROXY."),
    "rushes_on_field": (PT, "Team rush plays with the player on the field."),
    "tm_off_plays": (PT, "Team offensive plays with participation data."),
    "tm_dropbacks": (PT, "Team dropbacks with participation data."),
    "tm_rushes": (PT, "Team rush plays with participation data."),
    "play_share_part": (D, "off_plays_on_field / tm_off_plays. Participation-based snap share."),
    "pass_snap_share": (D, "dropbacks_on_field / tm_dropbacks."),
    "run_snap_share": (D, "rushes_on_field / tm_rushes."),
    "routes_proxy": (D, "= dropbacks_on_field. NOT charted routes: a TE or back who stays in to "
                        "block is counted as on the field, so this OVERSTATES routes run and "
                        "TPRR/YPRR built on it are conservative (biased low). Bias is small for "
                        "boundary WRs, material for blocking TEs and pass-protecting backs."),
    "tprr_proxy": (D, "targets / routes_proxy. Targets per route run, proxy denominator."),
    "yprr_proxy": (D, "rec_yards / routes_proxy. Yards per route run, proxy denominator."),

    # ---- PFR receiving
    "pfr_tgt": (PR, "PFR's own target count (denominator for its rates)."),
    "pfr_rec": (PR, "PFR's own reception count."),
    "pfr_adot": (PR, "Average depth of target, PFR charting."),
    "pfr_ybc": (PR, "Receiving yards before catch."),
    "pfr_ybc_per_rec": (PR, "Yards before catch per reception."),
    "pfr_yac": (PR, "Receiving yards after catch, PFR charting."),
    "pfr_yac_per_rec": (PR, "Yards after catch per reception."),
    "pfr_broken_tackles": (PR, "Broken tackles on receptions."),
    "pfr_rec_per_broken_tackle": (PR, "Receptions per broken tackle (lower = more elusive)."),
    "pfr_drops": (PR, "Charted drops."),
    "pfr_drop_pct": (D, "pfr_drops / pfr_tgt. Rebuilt from counts so it aggregates correctly."),
    "pfr_first_downs": (PR, "Receiving first downs, PFR."),

    # ---- PFR rushing
    "pfr_rush_att": (PU, "PFR rush attempts (denominator for its rates)."),
    "pfr_rush_ybc": (PU, "Rushing yards before contact."),
    "pfr_ybc_per_att": (PU, "Yards before contact per attempt. Blocking/scheme-heavy."),
    "pfr_rush_yac": (PU, "Rushing yards after contact."),
    "pfr_yac_per_att": (PU, "Yards after contact per attempt. Runner-heavy."),
    "pfr_rush_broken_tackles": (PU, "Broken tackles on runs."),
    "pfr_att_per_broken_tackle": (PU, "Attempts per broken tackle (lower = more elusive)."),
    "pfr_rush_first_downs": (PU, "Rushing first downs, PFR."),

    # ---- PFR passing
    "pfr_pass_attempts": (PP, "PFR pass attempts (denominator for its rates)."),
    "pfr_pocket_time": (PP, "Average time in pocket, seconds."),
    "times_blitzed": (PP, "Dropbacks facing a blitz."),
    "times_hurried": (PP, "Hurries allowed."),
    "times_hit": (PP, "QB hits taken."),
    "times_pressured": (PP, "Pressures faced (hurry + hit + sack)."),
    "pfr_pressure_pct": (D, "100 * times_pressured / pfr_pass_attempts."),
    "bad_throws": (PP, "Charted bad throws."),
    "pfr_bad_throw_pct": (D, "100 * bad_throws / pfr_pass_attempts."),
    "pfr_on_target_pct": (PP, "On-target throw %. Season-level as published (not rebuilt)."),
    "pfr_team_drop_pct": (PP, "Drop % charged to the QB's receivers."),
    "throwaways": (PP, "Deliberate throwaways."),
    "batted_balls": (PP, "Passes batted at the line."),
    "rpo_plays": (PP, "RPO plays."),
    "pfr_rpo_rate": (D, "rpo_plays / pfr_pass_attempts."),
    "pa_pass_att": (PP, "Play-action pass attempts."),
    "pfr_pa_rate": (D, "pa_pass_att / pfr_pass_attempts."),
    "pfr_blitz_rate": (D, "times_blitzed / pfr_pass_attempts."),
    "pfr_iay_per_att": (PP, "Intended air yards per attempt."),
    "pfr_cay_per_comp": (PP, "Completed air yards per completion."),
    "pfr_scrambles": (PP, "Scrambles, PFR charting (cross-check on `scrambles`)."),
    "pfr_scramble_ypa": (PP, "Yards per scramble."),

    # ---- NGS receiving
    "ngs_avg_cushion": (NR, "Average cushion (yds) from the nearest defender at snap."),
    "ngs_avg_separation": (NR, "Average separation (yds) at the moment of catch/incompletion."),
    "ngs_avg_intended_air_yards": (NR, "Average intended air yards on targets."),
    "ngs_percent_share_of_intended_air_yards": (NR, "Share of team intended air yards (%)."),
    "ngs_avg_yac": (NR, "Average YAC."),
    "ngs_avg_expected_yac": (NR, "Model-expected YAC given tracking context."),
    "ngs_avg_yac_above_expectation": (NR, "YAC over expected, per reception."),
    "ngs_catch_percentage": (NR, "NGS catch %."),

    # ---- NGS rushing
    "ngs_efficiency": (NU, "Distance travelled per yard gained (lower = more direct)."),
    "ngs_percent_attempts_gte_eight_defenders": (NU, "% of carries against 8+ in the box."),
    "ngs_avg_time_to_los": (NU, "Average seconds to line of scrimmage."),
    "ngs_expected_rush_yards": (NU, "Model-expected rush yards, season total."),
    "ngs_rush_yards_over_expected": (NU, "RYOE, season total."),
    "ngs_rush_yards_over_expected_per_att": (NU, "RYOE per attempt, as published."),
    "ngs_ryoe_per_att": (D, "ngs_rush_yards_over_expected / carries. Rebuilt from totals so it "
                            "aggregates correctly over a window; compare to the published per-att."),
    "ngs_rush_pct_over_expected": (NU, "% rush yards over expected."),

    # ---- NGS passing
    "ngs_avg_time_to_throw": (NP, "Average time to throw, seconds."),
    "ngs_avg_completed_air_yards": (NP, "Average completed air yards."),
    "ngs_avg_intended_air_yards_pass": (NP, "Average intended air yards (QB)."),
    "ngs_avg_air_yards_differential": (NP, "Completed minus intended air yards."),
    "ngs_aggressiveness": (NP, "% of throws into tight coverage (<1 yd separation)."),
    "ngs_avg_air_yards_to_sticks": (NP, "Air yards relative to the first-down marker."),
    "ngs_expected_completion_percentage": (NP, "Model-expected completion %."),
    "ngs_completion_percentage_above_expectation": (NP, "NGS CPOE."),
    "ngs_avg_air_distance": (NP, "Average air distance travelled by the ball."),
    "ngs_max_air_distance": (NP, "Longest air distance (window value = max over seasons)."),
    "ngs_passer_rating": (NP, "Passer rating as published by NGS."),

    # ---- PBP receiving usage
    "pbp_targets": (P, "Targets counted from pbp (excludes 2-pt and nullified plays). "
                       "Reconciles with weekly `targets` to |diff| <= 1."),
    "rz_targets": (P, "Targets with yardline_100 <= 20."),
    "i10_targets": (P, "Targets with yardline_100 <= 10."),
    "ez_targets": (P, "End-zone targets: air_yards >= yardline_100."),
    "deep_targets": (P, "Targets with air_yards >= 20."),
    "deep_target_rate": (D, "deep_targets / pbp_targets."),
    "rz_target_share_of_own": (D, "rz_targets / pbp_targets. Share of the player's OWN targets "
                                  "that came in the red zone (not share of team RZ targets)."),
    "third_down_targets": (P, "Targets on 3rd down."),
    "third_down_conv": (P, "3rd-down targets that produced a first down."),
    "third_down_conv_rate": (D, "third_down_conv / third_down_targets."),
    "target_epa_total": (P, "Sum of EPA on the player's targets."),
    "target_epa": (D, "target_epa_total / pbp_targets."),

    # ---- PBP rushing usage
    "pbp_carries": (P, "Carries counted from pbp (excludes 2-pt and nullified plays)."),
    "gl5_carries": (P, "Carries with yardline_100 <= 5."),
    "gl10_carries": (P, "Carries with yardline_100 <= 10."),
    "goal_to_go_carries": (P, "Carries in goal-to-go situations."),
    "gl5_carry_share_of_own": (D, "gl5_carries / pbp_carries."),
    "stuffed": (P, "Carries gaining <= 0 yards."),
    "stuffed_rate": (D, "stuffed / pbp_carries."),
    "third_down_carries": (P, "Carries on 3rd down."),
    "third_down_carry_rate": (D, "third_down_carries / pbp_carries."),
    "short_yd_carries": (P, "Carries on 3rd/4th down with <= 2 to go."),
    "short_yd_conv": (P, "Short-yardage carries converting a first down."),
    "short_yd_conv_rate": (D, "short_yd_conv / short_yd_carries."),
    "explosive_runs": (P, "Carries gaining >= 10 yards."),
    "explosive_run_rate": (D, "explosive_runs / pbp_carries."),
    "rush_epa_total": (P, "Sum of EPA on the player's carries."),
    "rush_epa_per_att": (D, "rush_epa_total / pbp_carries."),
    "opportunity_pg": (D, "(carries + targets) / games. Opportunity volume."),

    # ---- QB
    "attempts": (W, "Pass attempts."),
    "completions": (W, "Completions."),
    "pass_yards": (W, "Passing yards."),
    "pass_tds": (W, "Passing touchdowns."),
    "interceptions": (W, "Interceptions thrown."),
    "sacks_suffered": (W, "Sacks taken (weekly source)."),
    "pass_air_yards": (W, "Passing air yards."),
    "pass_yac": (W, "Passing yards after catch."),
    "pass_epa": (W, "Sum of passing EPA (weekly source)."),
    "rush_epa": (W, "Sum of rushing EPA (weekly source)."),
    "cpoe_w": (W, "Weekly passing_cpoe averaged (attempt-weighted in a window). Cross-check on `cpoe`."),
    "comp_pct": (D, "completions / attempts."),
    "ypa": (D, "pass_yards / attempts."),
    "adot": (D, "pass_air_yards / attempts."),
    "td_rate": (D, "pass_tds / attempts."),
    "int_rate": (D, "interceptions / attempts."),
    "td_int": (D, "pass_tds / interceptions."),
    "dropbacks": (P, "Plays with qb_dropback == 1 attributed to the QB. Scrambles are keyed on "
                     "rusher_player_id in pbp and ARE included (they would otherwise vanish)."),
    "qb_epa_total": (P, "Sum of qb_epa over dropbacks."),
    "epa_per_dropback": (D, "qb_epa_total / dropbacks."),
    "cpoe_sum": (P, "Sum of play-level cpoe."),
    "cpoe_n": (P, "Plays with non-null cpoe."),
    "cpoe": (D, "cpoe_sum / cpoe_n. Play-weighted CPOE."),
    "sacks": (P, "Sacks taken (pbp)."),
    "sack_rate": (D, "sacks / dropbacks."),
    "scrambles": (P, "Scrambles (pbp qb_scramble). Matches PFR at r = 0.999."),
    "scramble_rate": (D, "scrambles / dropbacks."),
    "designed_rushes": (D, "carries - scrambles. Designed QB run volume."),
    "pbp_air_yards": (P, "Sum of air yards on dropbacks."),
    "pbp_air_yards_n": (P, "Dropbacks with non-null air yards."),
    "qb_hits": (P, "QB hits taken (pbp)."),
    "pressure_rate_pbp": (D, "(qb_hits + sacks) / dropbacks. pbp-only pressure proxy; "
                             "pfr_pressure_pct is the charted version and should be preferred."),
    "rush_ppr": (D, "0.1*rush_yards + 6*rush_tds. QB rushing fantasy points (no PPR component)."),
    "rush_share_of_ppr": (D, "rush_ppr / ppr. The rushing share of QB fantasy output -- the "
                             "quantity the section-O work found drives QB variance."),
    "espn_qbr": (QBR, "ESPN Total QBR (0-100), regular season."),
    "espn_qbr_raw": (QBR, "Raw QBR before opponent adjustment."),
    "espn_pts_added": (QBR, "ESPN points added."),
    "qb_plays": (QBR, "ESPN qualifying plays."),
    "epa_total": (QBR, "ESPN EPA total (their model, not nflverse EPA)."),

    # ---- team context: offense
    "pass_att": (TW, "Team pass attempts."),
    "team_targets": (TW, "Team targets."),
    "team_air_yards": (TW, "Team air yards."),
    "cpoe": (TW, "Team mean weekly passing CPOE."),  # overridden per-table below
    "total_yards": (TW, "pass_yards + rush_yards."),
    "off_plays": (P, "Non-special, non-aborted offensive plays."),
    "off_epa": (P, "Mean EPA per offensive play."),
    "pass_plays": (P, "Plays flagged `pass`."),
    "rush_plays": (P, "Plays flagged `rush`."),
    "pass_rate": (D, "pass_plays / (pass_plays + rush_plays)."),
    "proe": (P, "Mean pass_oe: actual minus model-expected pass rate, percentage points."),
    "xpass": (P, "Mean model-expected pass probability."),
    "neutral_plays": (P, "Plays with win prob in [0.2, 0.8] and quarter <= 3."),
    "neutral_pass_rate": (P, "Pass rate in neutral situations."),
    "neutral_proe": (P, "Mean pass_oe in neutral situations."),
    "neutral_sec_per_play": (D, "Mean seconds between consecutive plays of the same drive in "
                                "neutral situations, gaps outside (0, 60] dropped. PACE: lower = faster."),
    "off_pass_epa_play": (P, "Mean EPA per dropback."),
    "off_rush_epa_play": (P, "Mean EPA per rush."),
    "rz_drives": (P, "Drives reaching inside the 20."),
    "rz_td_rate": (P, "Share of red-zone drives ending in a touchdown."),
    "points_for": (G, "Points scored, REG season."),
    "g": (G, "Games in the schedule file (sanity duplicate of `games`)."),

    # ---- team context: defense
    "def_plays": (P, "Offensive plays faced."),
    "def_epa": (P, "Mean EPA per play allowed."),
    "def_pass_epa_play": (P, "Mean EPA per dropback allowed."),
    "def_rush_epa_play": (P, "Mean EPA per rush allowed."),
    "points_against": (G, "Points allowed."),
    "pass_yards_allowed": (TW, "Passing yards allowed (opponents' offensive totals)."),
    "rush_yards_allowed": (TW, "Rushing yards allowed."),
    "total_yards_allowed": (TW, "pass + rush yards allowed."),
    "pass_att_faced": (TW, "Opponent pass attempts."),
    "carries_faced": (TW, "Opponent rush attempts."),
    "targets_faced": (TW, "Opponent targets."),
    "fpa_qb": (W, "PPR fantasy points allowed to opposing QBs (sum over all QBs facing this team)."),
    "fpa_rb": (W, "PPR fantasy points allowed to opposing RBs."),
    "fpa_wr": (W, "PPR fantasy points allowed to opposing WRs."),
    "fpa_te": (W, "PPR fantasy points allowed to opposing TEs."),

    # ---- team context: FTN charting (2022+ only, NaN for 2018-2021)
    "ftn_plays": (FT, "Offensive plays with FTN charting."),
    "ftn_motion_rate": (FT, "Share of offensive plays with pre-snap motion."),
    "ftn_no_huddle_rate": (FT, "Share of offensive plays run no-huddle."),
    "ftn_trick_rate": (FT, "Share of offensive plays charted as trick plays."),
    "ftn_pa_rate": (FT, "Share of dropbacks using play action."),
    "ftn_screen_rate": (FT, "Share of dropbacks that were screens."),
    "ftn_rpo_rate": (FT, "Share of dropbacks that were RPOs."),
    "ftn_oop_rate": (FT, "Share of dropbacks with the QB out of the pocket."),
    "ftn_blitz_faced_rate": (FT, "Share of the offense's dropbacks facing >=1 blitzer."),
    "ftn_pass_rushers_faced": (FT, "Mean pass rushers faced per dropback (offense)."),
    "ftn_def_blitz_rate": (FT, "Share of dropbacks on which this DEFENSE sent >=1 blitzer."),
    "ftn_def_pass_rushers": (FT, "Mean pass rushers this defense sent per dropback."),
    "ftn_def_box": (FT, "Mean defenders in the box this defense showed on dropbacks."),
}

PG_BASE_NOTE = "divided by games"


def resolve(col):
    if col in DEFS:
        return DEFS[col]
    if col.startswith("rank_"):
        base = col[5:]
        return (D, f"Within-season rank of {base} across the 32 teams, 1 = best "
                   f"(descending for offensive/production metrics, ascending for "
                   f"points/yards/EPA/fantasy points ALLOWED).")
    if col.endswith("_pg"):
        base = col[:-3]
        for cand in (base, base + "s", base.replace("rec_", "receptions")):
            if cand in DEFS:
                return (DEFS[cand][0], f"{cand} {PG_BASE_NOTE}.")
        return (D, f"{base} {PG_BASE_NOTE}.")
    return None


def main():
    frames = {t: pd.read_csv(os.path.join(DER, f"{t}.csv"), nrows=5, low_memory=False)
              for t in TABLES}
    all_cols = sorted({c for f in frames.values() for c in f.columns})
    missing = [c for c in all_cols if resolve(c) is None]
    assert not missing, f"undocumented columns: {missing}"

    rows = pd.read_csv(os.path.join(DER, "adv_join_report.csv"))
    lines = []
    A = lines.append

    A("# Advanced stats layer — sources, definitions, caveats\n")
    A(f"Built by `scripts/fetch_advanced.py` (download) + `scripts/build_advanced.py` "
      f"(join/derive) + this script (docs). Regular season only, 2018–2025.\n")

    A("## Sources (all direct HTTP from nflverse-data GitHub releases, no auth)\n")
    A("| release tag | asset | coverage | what it gives |")
    A("|---|---|---|---|")
    A("| `pfr_advstats` | `advstats_season_rec.csv` | 2018–2025 | aDOT, YBC, YAC, broken tackles, drops |")
    A("| `pfr_advstats` | `advstats_season_rush.csv` | 2018–2025 | yards before/after contact per att, broken tackles |")
    A("| `pfr_advstats` | `advstats_season_pass.csv` | 2018–2025 | pressure, blitz, bad throws, pocket time, play-action, RPO |")
    A("| `nextgen_stats` | `ngs_receiving.csv.gz` | 2016–2025 | separation, cushion, YAC over expected, share of intended air yards |")
    A("| `nextgen_stats` | `ngs_rushing.csv.gz` | 2016–2025 | 8+ box rate, RYOE, efficiency, time to LOS |")
    A("| `nextgen_stats` | `ngs_passing.csv.gz` | 2016–2025 | time to throw, aggressiveness, CPOE, air yards to sticks |")
    A("| `snap_counts` | `snap_counts_{year}.csv` | 2018–2025 | offensive/ST snaps (PFR-keyed) |")
    A("| `pbp` | `play_by_play_{year}.parquet` | 2018–2025 | situational usage, EPA, PROE, pace |")
    A("| `pbp_participation` | `pbp_participation_{year}.parquet` | 2018–2025 | players on field per play — route proxy |")
    A("| `ftn_charting` | `ftn_charting_{year}.csv` | 2022–2025 | play-action / motion / screen / RPO / blitz rates (team context only; NaN 2018–2021) |")
    A("| `espn_data` | `qbr_season_level.csv` | 2006–2025 | ESPN Total QBR |")
    A("| `players` | `players.csv` | all | the gsis_id ↔ pfr_id ↔ espn_id crosswalk |")
    A("")
    A("**NGS asset naming.** The per-year files (`ngs_2024_receiving.csv.gz`) are stub files "
      "(~600 bytes, header only) from 2024 on. The live assets are the un-suffixed "
      "all-season files: `ngs_receiving.csv.gz`, `ngs_rushing.csv.gz`, `ngs_passing.csv.gz`. "
      "`week == 0` rows are the season aggregates; weekly rows are `week >= 1`.\n")

    A("## Identity resolution\n")
    A("`gsis_id` is the key everywhere. PFR tables are keyed on `pfr_id` and ESPN on its own "
      "athlete id; both are mapped through `players.csv`, with a unique-normalized-name "
      "fallback (accents stripped, suffixes removed, non-alpha dropped). The name fallback is "
      "restricted to names that are unique in `players.csv`, so QB/LB namesakes are never "
      "silently fused — that restriction is why ESPN QBR is joined on `espn_id` first "
      "(name-only matching lost every Josh Allen and Lamar Jackson season).\n")
    A("Measured join rates (`data/derived/adv_join_report.csv`):\n")
    A("| join | matched / rows | rate | note |")
    A("|---|---|---|---|")
    for _, r in rows.iterrows():
        A(f"| `{r['join']}` | {r['matched']} / {r['left_rows']} | {r['match_rate']:.4f} | "
          f"{r['note'] if isinstance(r['note'], str) else ''} |")
    A("")
    A("Sub-1.0 rates on the *position tables* are coverage, not failure: NGS publishes only "
      "players clearing a volume threshold. Conditional coverage:\n")
    A("- WR/TE NGS: 0% below 30 targets, 51% at 30–60, **100% above 60 targets**.")
    A("- RB NGS: 0% below 80 carries, 87% at 80–150, **100% above 150 carries**.")
    A("- QB NGS and ESPN QBR: **100% above 300 attempts**.")
    A("- PFR is ~100% wherever the player has any relevant volume.\n")

    A("## Known constructions and their caveats\n")
    A("**Two share denominators, deliberately.** `*_share` divides by the team total over the "
      "games the player was active for; `*_share_full` divides by the full-season team total. "
      "The first is the projection-relevant 'share while on the field' and does NOT sum to 1 "
      "across a team-season (measured mean 1.36 over WR/TE/RB) — that is arithmetic, not a bug. "
      "The second sums to 0.997 on average (residual: QB/OL/ST receivers and traded players "
      "assigned to their modal team) and is the one to use for vacated-share budgets.\n")
    A("**Routes are a proxy.** nflverse publishes no charted route counts. `routes_proxy` is the "
      "count of team dropbacks with the player on the field, from `pbp_participation`. Blockers "
      "count as route-runners, so TPRR/YPRR are biased low, mildly for boundary WRs and "
      "materially for blocking TEs and pass-protecting backs. Treat cross-archetype comparisons "
      "of `*_proxy` with suspicion.\n")
    A("**Two-point conversions are excluded** from all pbp usage counts, along with `no_play` "
      "(penalty-nullified) plays. Without this, pbp target counts run ~75/season above the "
      "weekly stats and 2-pt attempts (snapped from the 2) inflate goal-line carry counts. After "
      "the exclusion, pbp targets reconcile with weekly targets at r = 1.00000, max |diff| = 1.\n")
    A("**QB scrambles.** In pbp a scramble is booked as a rush: `passer_player_id` is null and the "
      "QB appears in `rusher_player_id`. Keying dropbacks on `passer_player_id` alone drops every "
      "scramble, which undercounts dropbacks, biases EPA/dropback (scrambles skew positive) and "
      "zeroes the designed-run split. Dropbacks are therefore keyed on "
      "`passer_player_id.fillna(rusher_player_id)`; the resulting scramble counts match PFR's "
      "independent charting at r = 0.9993 (mean difference 0.03 per player-season).\n")
    A("**Window aggregation.** In the `_recent3` tables every ratio is REBUILT from summed counts "
      "— a 3-season TPRR is Σtargets / Σroutes, never a mean of season TPRRs. Only rates whose "
      "denominator is not published (NGS tracking averages, PFR per-unit rates, QBR) fall back to "
      "a volume-weighted mean, weighted by targets / receptions / carries / attempts as "
      "appropriate. `snap_pct_mean` and `cpoe_w` are kept as independent cross-checks on the "
      "rebuilt `snap_share` and `cpoe`.\n")
    A("**Pace** is the mean gap in `game_seconds_remaining` between consecutive plays of the same "
      "drive, gaps outside (0, 60] dropped, restricted to neutral situations (win prob 0.2–0.8, "
      "quarters 1–3). Lower is faster.\n")
    A("**Era breaks.** 16-game seasons through 2020, 17 from 2021; COVID 2020 had no preseason "
      "and empty stadiums. Everything here is per-game or a rate, but `tm_games` is carried so "
      "season-total comparisons can be normalised.\n")
    A("**Rookies and 2026 draftees have no rows.** Nothing is imputed. Players present but "
      "sparse in the window carry `thin_data = TRUE` (< 2 seasons or < 8 games).\n")

    A("## Tables\n")
    for t in TABLES:
        f = frames[t]
        n = len(pd.read_csv(os.path.join(DER, f"{t}.csv"), usecols=[0], low_memory=False))
        A(f"### `data/derived/{t}.csv` — {n} rows, {len(f.columns)} columns\n")
        A("| column | source | definition |")
        A("|---|---|---|")
        for c in f.columns:
            src, defn = resolve(c)
            if c == "cpoe" and t == "team_context":
                src, defn = TW, "Team mean weekly passing CPOE."
            elif c == "cpoe":
                src, defn = D, "cpoe_sum / cpoe_n. Play-weighted CPOE."
            A(f"| `{c}` | {src} | {defn} |")
        A("")

    A("## Not obtainable this session (needs web search / paywalled sources)\n")
    A("These were requested and are genuinely absent — no weaker substitute has been "
      "silently swapped in:\n")
    A("1. **2026 offensive-line projections.** No forward-looking OL data exists in nflverse.")
    A("2. **2026 team offensive/defensive projections.** Everything here is realised 2018–2025.")
    A("3. **Third-party O-line rankings** (PFF grades, ESPN run-block / pass-block win rate). "
      "Paywalled or search-gated.")
    A("4. **College production data** for 2026 draftees. Separate sourcing problem, needs web.")
    A("5. **Charted routes run.** Not in any free nflverse release; `routes_proxy` stands in.\n")
    A("Nearest available in-sample stand-ins, to be used knowingly and not as substitutes: "
      "`pfr_ybc_per_att` and `pfr_pressure_pct` carry a large offensive-line component, and "
      "`ngs_percent_attempts_gte_eight_defenders` carries box-count context. All are backward-"
      "looking.\n")

    out = os.path.join(RES, "advanced_stats_notes.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {out} ({len(all_cols)} distinct columns documented)")


if __name__ == "__main__":
    main()

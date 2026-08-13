# Advanced stats layer — sources, definitions, caveats

Built by `scripts/fetch_advanced.py` (download) + `scripts/build_advanced.py` (join/derive) + this script (docs). Regular season only, 2018–2025.

## Sources (all direct HTTP from nflverse-data GitHub releases, no auth)

| release tag | asset | coverage | what it gives |
|---|---|---|---|
| `pfr_advstats` | `advstats_season_rec.csv` | 2018–2025 | aDOT, YBC, YAC, broken tackles, drops |
| `pfr_advstats` | `advstats_season_rush.csv` | 2018–2025 | yards before/after contact per att, broken tackles |
| `pfr_advstats` | `advstats_season_pass.csv` | 2018–2025 | pressure, blitz, bad throws, pocket time, play-action, RPO |
| `nextgen_stats` | `ngs_receiving.csv.gz` | 2016–2025 | separation, cushion, YAC over expected, share of intended air yards |
| `nextgen_stats` | `ngs_rushing.csv.gz` | 2016–2025 | 8+ box rate, RYOE, efficiency, time to LOS |
| `nextgen_stats` | `ngs_passing.csv.gz` | 2016–2025 | time to throw, aggressiveness, CPOE, air yards to sticks |
| `snap_counts` | `snap_counts_{year}.csv` | 2018–2025 | offensive/ST snaps (PFR-keyed) |
| `pbp` | `play_by_play_{year}.parquet` | 2018–2025 | situational usage, EPA, PROE, pace |
| `pbp_participation` | `pbp_participation_{year}.parquet` | 2018–2025 | players on field per play — route proxy |
| `ftn_charting` | `ftn_charting_{year}.csv` | 2022–2025 | play-action / motion / screen / RPO / blitz rates (team context only; NaN 2018–2021) |
| `espn_data` | `qbr_season_level.csv` | 2006–2025 | ESPN Total QBR |
| `players` | `players.csv` | all | the gsis_id ↔ pfr_id ↔ espn_id crosswalk |

**NGS asset naming.** The per-year files (`ngs_2024_receiving.csv.gz`) are stub files (~600 bytes, header only) from 2024 on. The live assets are the un-suffixed all-season files: `ngs_receiving.csv.gz`, `ngs_rushing.csv.gz`, `ngs_passing.csv.gz`. `week == 0` rows are the season aggregates; weekly rows are `week >= 1`.

## Identity resolution

`gsis_id` is the key everywhere. PFR tables are keyed on `pfr_id` and ESPN on its own athlete id; both are mapped through `players.csv`, with a unique-normalized-name fallback (accents stripped, suffixes removed, non-alpha dropped). The name fallback is restricted to names that are unique in `players.csv`, so QB/LB namesakes are never silently fused — that restriction is why ESPN QBR is joined on `espn_id` first (name-only matching lost every Josh Allen and Lamar Jackson season).

Measured join rates (`data/derived/adv_join_report.csv`):

| join | matched / rows | rate | note |
|---|---|---|---|
| `players.csv pfr_id<->gsis_id` | 22553 / 25044 | 0.9005 | coverage of nflverse players.csv |
| `player_week -> team_week` | 140747 / 140747 | 1.0000 |  |
| `snap_counts -> gsis_id` | 17211 / 17248 | 0.9979 | 17211 by pfr_id, 0 by name |
| `pfr_rec -> gsis_id` | 4124 / 4130 | 0.9986 | 4122 by pfr_id, 2 by name |
| `pfr_rush -> gsis_id` | 2817 / 2820 | 0.9989 | 2817 by pfr_id, 0 by name |
| `pfr_pass -> gsis_id` | 848 / 848 | 1.0000 | 848 by pfr_id, 0 by name |
| `espn_qbr -> gsis_id` | 1183 / 1183 | 1.0000 | 1176 by espn_id, rest by unique-name |
| `wr_te <- snap_counts` | 2904 / 2913 | 0.9969 |  |
| `wr_te <- participation` | 2884 / 2913 | 0.9900 |  |
| `wr_te <- pfr_rec` | 2730 / 2913 | 0.9372 |  |
| `wr_te <- ngs_receiving` | 994 / 2913 | 0.3412 |  |
| `wr_te <- pbp_receiving` | 2737 / 2913 | 0.9396 |  |
| `rb <- snap_counts` | 1317 / 1319 | 0.9985 |  |
| `rb <- participation` | 1301 / 1319 | 0.9863 |  |
| `rb <- pfr_rush` | 809 / 1319 | 0.6133 |  |
| `rb <- pfr_rec` | 1145 / 1319 | 0.8681 |  |
| `rb <- ngs_rushing` | 405 / 1319 | 0.3070 |  |
| `rb <- pbp_rushing` | 1194 / 1319 | 0.9052 |  |
| `rb <- pbp_receiving` | 1145 / 1319 | 0.8681 |  |
| `qb <- pfr_pass` | 599 / 599 | 1.0000 |  |
| `qb <- ngs_passing` | 327 / 599 | 0.5459 |  |
| `qb <- pbp_qb` | 599 / 599 | 1.0000 |  |
| `qb <- espn_qbr` | 329 / 599 | 0.5493 |  |
| `qb <- pbp_rushing` | 473 / 599 | 0.7896 |  |

Sub-1.0 rates on the *position tables* are coverage, not failure: NGS publishes only players clearing a volume threshold. Conditional coverage:

- WR/TE NGS: 0% below 30 targets, 51% at 30–60, **100% above 60 targets**.
- RB NGS: 0% below 80 carries, 87% at 80–150, **100% above 150 carries**.
- QB NGS and ESPN QBR: **100% above 300 attempts**.
- PFR is ~100% wherever the player has any relevant volume.

## Known constructions and their caveats

**Two share denominators, deliberately.** `*_share` divides by the team total over the games the player was active for; `*_share_full` divides by the full-season team total. The first is the projection-relevant 'share while on the field' and does NOT sum to 1 across a team-season (measured mean 1.36 over WR/TE/RB) — that is arithmetic, not a bug. The second sums to 0.997 on average (residual: QB/OL/ST receivers and traded players assigned to their modal team) and is the one to use for vacated-share budgets.

**Routes are a proxy.** nflverse publishes no charted route counts. `routes_proxy` is the count of team dropbacks with the player on the field, from `pbp_participation`. Blockers count as route-runners, so TPRR/YPRR are biased low, mildly for boundary WRs and materially for blocking TEs and pass-protecting backs. Treat cross-archetype comparisons of `*_proxy` with suspicion.

**Two-point conversions are excluded** from all pbp usage counts, along with `no_play` (penalty-nullified) plays. Without this, pbp target counts run ~75/season above the weekly stats and 2-pt attempts (snapped from the 2) inflate goal-line carry counts. After the exclusion, pbp targets reconcile with weekly targets at r = 1.00000, max |diff| = 1.

**QB scrambles.** In pbp a scramble is booked as a rush: `passer_player_id` is null and the QB appears in `rusher_player_id`. Keying dropbacks on `passer_player_id` alone drops every scramble, which undercounts dropbacks, biases EPA/dropback (scrambles skew positive) and zeroes the designed-run split. Dropbacks are therefore keyed on `passer_player_id.fillna(rusher_player_id)`; the resulting scramble counts match PFR's independent charting at r = 0.9993 (mean difference 0.03 per player-season).

**Window aggregation.** In the `_recent3` tables every ratio is REBUILT from summed counts — a 3-season TPRR is Σtargets / Σroutes, never a mean of season TPRRs. Only rates whose denominator is not published (NGS tracking averages, PFR per-unit rates, QBR) fall back to a volume-weighted mean, weighted by targets / receptions / carries / attempts as appropriate. `snap_pct_mean` and `cpoe_w` are kept as independent cross-checks on the rebuilt `snap_share` and `cpoe`.

**Pace** is the mean gap in `game_seconds_remaining` between consecutive plays of the same drive, gaps outside (0, 60] dropped, restricted to neutral situations (win prob 0.2–0.8, quarters 1–3). Lower is faster.

**Era breaks.** 16-game seasons through 2020, 17 from 2021; COVID 2020 had no preseason and empty stadiums. Everything here is per-game or a rate, but `tm_games` is carried so season-total comparisons can be normalised.

**Rookies and 2026 draftees have no rows.** Nothing is imputed. Players present but sparse in the window carry `thin_data = TRUE` (< 2 seasons or < 8 games).

## Tables

### `data/derived/adv_wr_te.csv` — 2737 rows, 107 columns

| column | source | definition |
|---|---|---|
| `player_id` | nflverse stats_player (weekly) | gsis_id. Primary join key across every table. |
| `season` | nflverse stats_player (weekly) | NFL season (REG only throughout). |
| `player_name` | nflverse stats_player (weekly) | Display name, first appearance in the window. |
| `position` | nflverse stats_player (weekly) | Modal weekly position over the player-season (last season in a window). |
| `team` | nflverse stats_player (weekly) | Modal weekly team over the player-season. Traded players get one team; n_teams flags them. |
| `n_teams` | nflverse stats_player (weekly) | Distinct teams the player recorded a game for in the period. |
| `games` | nflverse stats_player (weekly) | Distinct REG weeks with a stat line. |
| `targets` | nflverse stats_player (weekly) | Targets. Excludes two-point conversions. |
| `receptions` | nflverse stats_player (weekly) | Receptions. |
| `rec_yards` | nflverse stats_player (weekly) | Receiving yards. |
| `rec_tds` | nflverse stats_player (weekly) | Receiving touchdowns. |
| `rec_air_yards` | nflverse stats_player (weekly) | Sum of air yards on targets. |
| `rec_yac` | nflverse stats_player (weekly) | Receiving yards after catch. |
| `rec_first_downs` | nflverse stats_player (weekly) | Receiving first downs. |
| `rec_epa` | nflverse stats_player (weekly) | Sum of receiving EPA. |
| `carries` | nflverse stats_player (weekly) | Rush attempts. |
| `rush_yards` | nflverse stats_player (weekly) | Rushing yards. |
| `rush_tds` | nflverse stats_player (weekly) | Rushing touchdowns. |
| `ppr` | nflverse stats_player (weekly) | Total PPR fantasy points. |
| `tm_targets` | nflverse stats_team (weekly) | Team targets summed over the games the player was active for. |
| `tm_air_yards` | nflverse stats_team (weekly) | Team air yards over the player's active games. |
| `tm_pass_att` | nflverse stats_team (weekly) | Team pass attempts over the player's active games. |
| `tm_carries` | nflverse stats_team (weekly) | Team rush attempts over the player's active games. |
| `tm_targets_full` | nflverse stats_team (weekly) | Team targets over the FULL season. |
| `tm_air_yards_full` | nflverse stats_team (weekly) | Team air yards over the full season. |
| `tm_carries_full` | nflverse stats_team (weekly) | Team rush attempts over the full season. |
| `tm_games` | nflverse stats_team (weekly) | Team games in the season (16 pre-2021, 17 from 2021). |
| `snap_games` | nflverse snap_counts (PFR) | Games with >0 offensive snaps. |
| `off_snaps` | nflverse snap_counts (PFR) | Offensive snaps. |
| `tm_off_snaps` | nflverse snap_counts (PFR) | Team offensive snaps in the games the player was dressed (max offense_snaps among the team's players that game). |
| `snap_pct_mean` | nflverse snap_counts (PFR) | Unweighted mean of PFR's per-game offense_pct. Kept as a cross-check on snap_share; games-weighted mean in a window. |
| `st_snaps` | nflverse snap_counts (PFR) | Special-teams snaps. Low ST usage is a weak signal of offensive role. |
| `off_plays_on_field` | nflverse pbp_participation | Offensive plays with the player in offense_players. |
| `dropbacks_on_field` | nflverse pbp_participation | Team dropbacks with the player on the field. ROUTE PROXY. |
| `rushes_on_field` | nflverse pbp_participation | Team rush plays with the player on the field. |
| `tm_off_plays` | nflverse pbp_participation | Team offensive plays with participation data. |
| `tm_dropbacks` | nflverse pbp_participation | Team dropbacks with participation data. |
| `tm_rushes` | nflverse pbp_participation | Team rush plays with participation data. |
| `pfr_tgt` | PFR advstats_season_rec | PFR's own target count (denominator for its rates). |
| `pfr_rec` | PFR advstats_season_rec | PFR's own reception count. |
| `pfr_adot` | PFR advstats_season_rec | Average depth of target, PFR charting. |
| `pfr_ybc` | PFR advstats_season_rec | Receiving yards before catch. |
| `pfr_ybc_per_rec` | PFR advstats_season_rec | Yards before catch per reception. |
| `pfr_yac` | PFR advstats_season_rec | Receiving yards after catch, PFR charting. |
| `pfr_yac_per_rec` | PFR advstats_season_rec | Yards after catch per reception. |
| `pfr_broken_tackles` | PFR advstats_season_rec | Broken tackles on receptions. |
| `pfr_rec_per_broken_tackle` | PFR advstats_season_rec | Receptions per broken tackle (lower = more elusive). |
| `pfr_drops` | PFR advstats_season_rec | Charted drops. |
| `pfr_first_downs` | PFR advstats_season_rec | Receiving first downs, PFR. |
| `ngs_avg_cushion` | NGS ngs_receiving | Average cushion (yds) from the nearest defender at snap. |
| `ngs_avg_separation` | NGS ngs_receiving | Average separation (yds) at the moment of catch/incompletion. |
| `ngs_avg_intended_air_yards` | NGS ngs_receiving | Average intended air yards on targets. |
| `ngs_percent_share_of_intended_air_yards` | NGS ngs_receiving | Share of team intended air yards (%). |
| `ngs_avg_yac` | NGS ngs_receiving | Average YAC. |
| `ngs_avg_expected_yac` | NGS ngs_receiving | Model-expected YAC given tracking context. |
| `ngs_avg_yac_above_expectation` | NGS ngs_receiving | YAC over expected, per reception. |
| `ngs_catch_percentage` | NGS ngs_receiving | NGS catch %. |
| `pbp_targets` | nflverse pbp | Targets counted from pbp (excludes 2-pt and nullified plays). Reconciles with weekly `targets` to |diff| <= 1. |
| `rz_targets` | nflverse pbp | Targets with yardline_100 <= 20. |
| `i10_targets` | nflverse pbp | Targets with yardline_100 <= 10. |
| `ez_targets` | nflverse pbp | End-zone targets: air_yards >= yardline_100. |
| `deep_targets` | nflverse pbp | Targets with air_yards >= 20. |
| `third_down_targets` | nflverse pbp | Targets on 3rd down. |
| `third_down_conv` | nflverse pbp | 3rd-down targets that produced a first down. |
| `target_epa_total` | nflverse pbp | Sum of EPA on the player's targets. |
| `target_share` | derived | targets / tm_targets. Share WHILE ACTIVE. Does not sum to 1 within a team-season; use for projection, not for budget arithmetic. |
| `air_yards_share` | derived | rec_air_yards / tm_air_yards, active-games denominator. |
| `target_share_full` | derived | targets / tm_targets_full. Sums to ~1 across a team-season; use for vacated-share / budget arithmetic. Understates part-season players. |
| `air_yards_share_full` | derived | rec_air_yards / tm_air_yards_full. |
| `carry_share` | derived | carries / tm_carries, active-games denominator. |
| `carry_share_full` | derived | carries / tm_carries_full. |
| `games_played_rate` | derived | games / tm_games. Availability. |
| `wopr` | derived | 1.5*target_share + 0.7*air_yards_share (active-games denominators). |
| `racr` | derived | rec_yards / rec_air_yards. |
| `adot_nflverse` | derived | rec_air_yards / targets. nflverse-derived aDOT; compare pfr_adot. |
| `catch_rate` | derived | receptions / targets. |
| `rec_epa_per_target` | derived | rec_epa / targets. |
| `ypc` | derived | rush_yards / carries. |
| `ypr` | derived | rec_yards / receptions. |
| `snap_share` | derived | off_snaps / tm_off_snaps. |
| `off_snaps_pg` | nflverse snap_counts (PFR) | off_snaps divided by games. |
| `pass_snap_share` | derived | dropbacks_on_field / tm_dropbacks. |
| `run_snap_share` | derived | rushes_on_field / tm_rushes. |
| `play_share_part` | derived | off_plays_on_field / tm_off_plays. Participation-based snap share. |
| `targets_pg` | nflverse stats_player (weekly) | targets divided by games. |
| `rec_pg` | derived | receptions / games. |
| `rec_yards_pg` | nflverse stats_player (weekly) | rec_yards divided by games. |
| `rec_tds_pg` | nflverse stats_player (weekly) | rec_tds divided by games. |
| `air_yards_pg` | derived | rec_air_yards / games. |
| `ppr_pg` | nflverse stats_player (weekly) | ppr divided by games. |
| `carries_pg` | nflverse stats_player (weekly) | carries divided by games. |
| `rush_yards_pg` | nflverse stats_player (weekly) | rush_yards divided by games. |
| `rush_tds_pg` | nflverse stats_player (weekly) | rush_tds divided by games. |
| `routes_proxy` | derived | = dropbacks_on_field. NOT charted routes: a TE or back who stays in to block is counted as on the field, so this OVERSTATES routes run and TPRR/YPRR built on it are conservative (biased low). Bias is small for boundary WRs, material for blocking TEs and pass-protecting backs. |
| `routes_proxy_pg` | derived | routes_proxy / games. Proxy routes run per game. |
| `tprr_proxy` | derived | targets / routes_proxy. Targets per route run, proxy denominator. |
| `yprr_proxy` | derived | rec_yards / routes_proxy. Yards per route run, proxy denominator. |
| `pfr_drop_pct` | derived | pfr_drops / pfr_tgt. Rebuilt from counts so it aggregates correctly. |
| `target_epa` | derived | target_epa_total / pbp_targets. |
| `rz_targets_pg` | nflverse pbp | rz_targets divided by games. |
| `i10_targets_pg` | nflverse pbp | i10_targets divided by games. |
| `ez_targets_pg` | nflverse pbp | ez_targets divided by games. |
| `deep_targets_pg` | nflverse pbp | deep_targets divided by games. |
| `third_down_targets_pg` | nflverse pbp | third_down_targets divided by games. |
| `deep_target_rate` | derived | deep_targets / pbp_targets. |
| `rz_target_share_of_own` | derived | rz_targets / pbp_targets. Share of the player's OWN targets that came in the red zone (not share of team RZ targets). |
| `third_down_conv_rate` | derived | third_down_conv / third_down_targets. |

### `data/derived/adv_rb.csv` — 1246 rows, 130 columns

| column | source | definition |
|---|---|---|
| `player_id` | nflverse stats_player (weekly) | gsis_id. Primary join key across every table. |
| `season` | nflverse stats_player (weekly) | NFL season (REG only throughout). |
| `player_name` | nflverse stats_player (weekly) | Display name, first appearance in the window. |
| `position` | nflverse stats_player (weekly) | Modal weekly position over the player-season (last season in a window). |
| `team` | nflverse stats_player (weekly) | Modal weekly team over the player-season. Traded players get one team; n_teams flags them. |
| `n_teams` | nflverse stats_player (weekly) | Distinct teams the player recorded a game for in the period. |
| `games` | nflverse stats_player (weekly) | Distinct REG weeks with a stat line. |
| `targets` | nflverse stats_player (weekly) | Targets. Excludes two-point conversions. |
| `receptions` | nflverse stats_player (weekly) | Receptions. |
| `rec_yards` | nflverse stats_player (weekly) | Receiving yards. |
| `rec_tds` | nflverse stats_player (weekly) | Receiving touchdowns. |
| `rec_air_yards` | nflverse stats_player (weekly) | Sum of air yards on targets. |
| `rec_yac` | nflverse stats_player (weekly) | Receiving yards after catch. |
| `rec_first_downs` | nflverse stats_player (weekly) | Receiving first downs. |
| `rec_epa` | nflverse stats_player (weekly) | Sum of receiving EPA. |
| `carries` | nflverse stats_player (weekly) | Rush attempts. |
| `rush_yards` | nflverse stats_player (weekly) | Rushing yards. |
| `rush_tds` | nflverse stats_player (weekly) | Rushing touchdowns. |
| `ppr` | nflverse stats_player (weekly) | Total PPR fantasy points. |
| `tm_targets` | nflverse stats_team (weekly) | Team targets summed over the games the player was active for. |
| `tm_air_yards` | nflverse stats_team (weekly) | Team air yards over the player's active games. |
| `tm_pass_att` | nflverse stats_team (weekly) | Team pass attempts over the player's active games. |
| `tm_carries` | nflverse stats_team (weekly) | Team rush attempts over the player's active games. |
| `tm_targets_full` | nflverse stats_team (weekly) | Team targets over the FULL season. |
| `tm_air_yards_full` | nflverse stats_team (weekly) | Team air yards over the full season. |
| `tm_carries_full` | nflverse stats_team (weekly) | Team rush attempts over the full season. |
| `tm_games` | nflverse stats_team (weekly) | Team games in the season (16 pre-2021, 17 from 2021). |
| `snap_games` | nflverse snap_counts (PFR) | Games with >0 offensive snaps. |
| `off_snaps` | nflverse snap_counts (PFR) | Offensive snaps. |
| `tm_off_snaps` | nflverse snap_counts (PFR) | Team offensive snaps in the games the player was dressed (max offense_snaps among the team's players that game). |
| `snap_pct_mean` | nflverse snap_counts (PFR) | Unweighted mean of PFR's per-game offense_pct. Kept as a cross-check on snap_share; games-weighted mean in a window. |
| `st_snaps` | nflverse snap_counts (PFR) | Special-teams snaps. Low ST usage is a weak signal of offensive role. |
| `off_plays_on_field` | nflverse pbp_participation | Offensive plays with the player in offense_players. |
| `dropbacks_on_field` | nflverse pbp_participation | Team dropbacks with the player on the field. ROUTE PROXY. |
| `rushes_on_field` | nflverse pbp_participation | Team rush plays with the player on the field. |
| `tm_off_plays` | nflverse pbp_participation | Team offensive plays with participation data. |
| `tm_dropbacks` | nflverse pbp_participation | Team dropbacks with participation data. |
| `tm_rushes` | nflverse pbp_participation | Team rush plays with participation data. |
| `pfr_rush_att` | PFR advstats_season_rush | PFR rush attempts (denominator for its rates). |
| `pfr_rush_ybc` | PFR advstats_season_rush | Rushing yards before contact. |
| `pfr_ybc_per_att` | PFR advstats_season_rush | Yards before contact per attempt. Blocking/scheme-heavy. |
| `pfr_rush_yac` | PFR advstats_season_rush | Rushing yards after contact. |
| `pfr_yac_per_att` | PFR advstats_season_rush | Yards after contact per attempt. Runner-heavy. |
| `pfr_rush_broken_tackles` | PFR advstats_season_rush | Broken tackles on runs. |
| `pfr_att_per_broken_tackle` | PFR advstats_season_rush | Attempts per broken tackle (lower = more elusive). |
| `pfr_rush_first_downs` | PFR advstats_season_rush | Rushing first downs, PFR. |
| `pfr_tgt` | PFR advstats_season_rec | PFR's own target count (denominator for its rates). |
| `pfr_rec` | PFR advstats_season_rec | PFR's own reception count. |
| `pfr_adot` | PFR advstats_season_rec | Average depth of target, PFR charting. |
| `pfr_ybc` | PFR advstats_season_rec | Receiving yards before catch. |
| `pfr_ybc_per_rec` | PFR advstats_season_rec | Yards before catch per reception. |
| `pfr_yac` | PFR advstats_season_rec | Receiving yards after catch, PFR charting. |
| `pfr_yac_per_rec` | PFR advstats_season_rec | Yards after catch per reception. |
| `pfr_broken_tackles` | PFR advstats_season_rec | Broken tackles on receptions. |
| `pfr_rec_per_broken_tackle` | PFR advstats_season_rec | Receptions per broken tackle (lower = more elusive). |
| `pfr_drops` | PFR advstats_season_rec | Charted drops. |
| `pfr_first_downs` | PFR advstats_season_rec | Receiving first downs, PFR. |
| `ngs_efficiency` | NGS ngs_rushing | Distance travelled per yard gained (lower = more direct). |
| `ngs_percent_attempts_gte_eight_defenders` | NGS ngs_rushing | % of carries against 8+ in the box. |
| `ngs_avg_time_to_los` | NGS ngs_rushing | Average seconds to line of scrimmage. |
| `ngs_expected_rush_yards` | NGS ngs_rushing | Model-expected rush yards, season total. |
| `ngs_rush_yards_over_expected` | NGS ngs_rushing | RYOE, season total. |
| `ngs_rush_yards_over_expected_per_att` | NGS ngs_rushing | RYOE per attempt, as published. |
| `ngs_rush_pct_over_expected` | NGS ngs_rushing | % rush yards over expected. |
| `pbp_carries` | nflverse pbp | Carries counted from pbp (excludes 2-pt and nullified plays). |
| `gl5_carries` | nflverse pbp | Carries with yardline_100 <= 5. |
| `gl10_carries` | nflverse pbp | Carries with yardline_100 <= 10. |
| `goal_to_go_carries` | nflverse pbp | Carries in goal-to-go situations. |
| `stuffed` | nflverse pbp | Carries gaining <= 0 yards. |
| `third_down_carries` | nflverse pbp | Carries on 3rd down. |
| `short_yd_carries` | nflverse pbp | Carries on 3rd/4th down with <= 2 to go. |
| `short_yd_conv` | nflverse pbp | Short-yardage carries converting a first down. |
| `explosive_runs` | nflverse pbp | Carries gaining >= 10 yards. |
| `rush_epa_total` | nflverse pbp | Sum of EPA on the player's carries. |
| `pbp_targets` | nflverse pbp | Targets counted from pbp (excludes 2-pt and nullified plays). Reconciles with weekly `targets` to |diff| <= 1. |
| `rz_targets` | nflverse pbp | Targets with yardline_100 <= 20. |
| `i10_targets` | nflverse pbp | Targets with yardline_100 <= 10. |
| `ez_targets` | nflverse pbp | End-zone targets: air_yards >= yardline_100. |
| `deep_targets` | nflverse pbp | Targets with air_yards >= 20. |
| `third_down_targets` | nflverse pbp | Targets on 3rd down. |
| `third_down_conv` | nflverse pbp | 3rd-down targets that produced a first down. |
| `target_epa_total` | nflverse pbp | Sum of EPA on the player's targets. |
| `target_share` | derived | targets / tm_targets. Share WHILE ACTIVE. Does not sum to 1 within a team-season; use for projection, not for budget arithmetic. |
| `air_yards_share` | derived | rec_air_yards / tm_air_yards, active-games denominator. |
| `target_share_full` | derived | targets / tm_targets_full. Sums to ~1 across a team-season; use for vacated-share / budget arithmetic. Understates part-season players. |
| `air_yards_share_full` | derived | rec_air_yards / tm_air_yards_full. |
| `carry_share` | derived | carries / tm_carries, active-games denominator. |
| `carry_share_full` | derived | carries / tm_carries_full. |
| `games_played_rate` | derived | games / tm_games. Availability. |
| `wopr` | derived | 1.5*target_share + 0.7*air_yards_share (active-games denominators). |
| `racr` | derived | rec_yards / rec_air_yards. |
| `adot_nflverse` | derived | rec_air_yards / targets. nflverse-derived aDOT; compare pfr_adot. |
| `catch_rate` | derived | receptions / targets. |
| `rec_epa_per_target` | derived | rec_epa / targets. |
| `ypc` | derived | rush_yards / carries. |
| `ypr` | derived | rec_yards / receptions. |
| `snap_share` | derived | off_snaps / tm_off_snaps. |
| `off_snaps_pg` | nflverse snap_counts (PFR) | off_snaps divided by games. |
| `pass_snap_share` | derived | dropbacks_on_field / tm_dropbacks. |
| `run_snap_share` | derived | rushes_on_field / tm_rushes. |
| `play_share_part` | derived | off_plays_on_field / tm_off_plays. Participation-based snap share. |
| `targets_pg` | nflverse stats_player (weekly) | targets divided by games. |
| `rec_pg` | derived | receptions / games. |
| `rec_yards_pg` | nflverse stats_player (weekly) | rec_yards divided by games. |
| `rec_tds_pg` | nflverse stats_player (weekly) | rec_tds divided by games. |
| `air_yards_pg` | derived | rec_air_yards / games. |
| `ppr_pg` | nflverse stats_player (weekly) | ppr divided by games. |
| `carries_pg` | nflverse stats_player (weekly) | carries divided by games. |
| `rush_yards_pg` | nflverse stats_player (weekly) | rush_yards divided by games. |
| `rush_tds_pg` | nflverse stats_player (weekly) | rush_tds divided by games. |
| `touches` | derived | carries + receptions. |
| `touches_pg` | derived | touches / games. |
| `opportunity_pg` | derived | (carries + targets) / games. Opportunity volume. |
| `stuffed_rate` | derived | stuffed / pbp_carries. |
| `short_yd_conv_rate` | derived | short_yd_conv / short_yd_carries. |
| `explosive_run_rate` | derived | explosive_runs / pbp_carries. |
| `rush_epa_per_att` | derived | rush_epa_total / pbp_carries. |
| `target_epa` | derived | target_epa_total / pbp_targets. |
| `gl5_carry_share_of_own` | derived | gl5_carries / pbp_carries. |
| `third_down_carry_rate` | derived | third_down_carries / pbp_carries. |
| `pfr_drop_pct` | derived | pfr_drops / pfr_tgt. Rebuilt from counts so it aggregates correctly. |
| `ngs_ryoe_per_att` | derived | ngs_rush_yards_over_expected / carries. Rebuilt from totals so it aggregates correctly over a window; compare to the published per-att. |
| `routes_proxy` | derived | = dropbacks_on_field. NOT charted routes: a TE or back who stays in to block is counted as on the field, so this OVERSTATES routes run and TPRR/YPRR built on it are conservative (biased low). Bias is small for boundary WRs, material for blocking TEs and pass-protecting backs. |
| `tprr_proxy` | derived | targets / routes_proxy. Targets per route run, proxy denominator. |
| `gl5_carries_pg` | nflverse pbp | gl5_carries divided by games. |
| `gl10_carries_pg` | nflverse pbp | gl10_carries divided by games. |
| `goal_to_go_carries_pg` | nflverse pbp | goal_to_go_carries divided by games. |
| `third_down_carries_pg` | nflverse pbp | third_down_carries divided by games. |
| `explosive_runs_pg` | nflverse pbp | explosive_runs divided by games. |
| `rz_targets_pg` | nflverse pbp | rz_targets divided by games. |

### `data/derived/adv_qb.csv` — 599 rows, 101 columns

| column | source | definition |
|---|---|---|
| `player_id` | nflverse stats_player (weekly) | gsis_id. Primary join key across every table. |
| `season` | nflverse stats_player (weekly) | NFL season (REG only throughout). |
| `player_name` | nflverse stats_player (weekly) | Display name, first appearance in the window. |
| `position` | nflverse stats_player (weekly) | Modal weekly position over the player-season (last season in a window). |
| `team` | nflverse stats_player (weekly) | Modal weekly team over the player-season. Traded players get one team; n_teams flags them. |
| `games` | nflverse stats_player (weekly) | Distinct REG weeks with a stat line. |
| `attempts` | nflverse stats_player (weekly) | Pass attempts. |
| `completions` | nflverse stats_player (weekly) | Completions. |
| `pass_yards` | nflverse stats_player (weekly) | Passing yards. |
| `pass_tds` | nflverse stats_player (weekly) | Passing touchdowns. |
| `interceptions` | nflverse stats_player (weekly) | Interceptions thrown. |
| `sacks_suffered` | nflverse stats_player (weekly) | Sacks taken (weekly source). |
| `pass_air_yards` | nflverse stats_player (weekly) | Passing air yards. |
| `pass_yac` | nflverse stats_player (weekly) | Passing yards after catch. |
| `pass_epa` | nflverse stats_player (weekly) | Sum of passing EPA (weekly source). |
| `cpoe_w` | nflverse stats_player (weekly) | Weekly passing_cpoe averaged (attempt-weighted in a window). Cross-check on `cpoe`. |
| `carries` | nflverse stats_player (weekly) | Rush attempts. |
| `rush_yards` | nflverse stats_player (weekly) | Rushing yards. |
| `rush_tds` | nflverse stats_player (weekly) | Rushing touchdowns. |
| `rush_epa` | nflverse stats_player (weekly) | Sum of rushing EPA (weekly source). |
| `ppr` | nflverse stats_player (weekly) | Total PPR fantasy points. |
| `pfr_pocket_time` | PFR advstats_season_pass | Average time in pocket, seconds. |
| `times_blitzed` | PFR advstats_season_pass | Dropbacks facing a blitz. |
| `times_hurried` | PFR advstats_season_pass | Hurries allowed. |
| `times_hit` | PFR advstats_season_pass | QB hits taken. |
| `times_pressured` | PFR advstats_season_pass | Pressures faced (hurry + hit + sack). |
| `pfr_pressure_pct` | derived | 100 * times_pressured / pfr_pass_attempts. |
| `bad_throws` | PFR advstats_season_pass | Charted bad throws. |
| `pfr_bad_throw_pct` | derived | 100 * bad_throws / pfr_pass_attempts. |
| `pfr_on_target_pct` | PFR advstats_season_pass | On-target throw %. Season-level as published (not rebuilt). |
| `pfr_team_drop_pct` | PFR advstats_season_pass | Drop % charged to the QB's receivers. |
| `throwaways` | PFR advstats_season_pass | Deliberate throwaways. |
| `batted_balls` | PFR advstats_season_pass | Passes batted at the line. |
| `rpo_plays` | PFR advstats_season_pass | RPO plays. |
| `pa_pass_att` | PFR advstats_season_pass | Play-action pass attempts. |
| `pfr_iay_per_att` | PFR advstats_season_pass | Intended air yards per attempt. |
| `pfr_cay_per_comp` | PFR advstats_season_pass | Completed air yards per completion. |
| `pfr_scrambles` | PFR advstats_season_pass | Scrambles, PFR charting (cross-check on `scrambles`). |
| `pfr_scramble_ypa` | PFR advstats_season_pass | Yards per scramble. |
| `pfr_pass_attempts` | PFR advstats_season_pass | PFR pass attempts (denominator for its rates). |
| `pfr_blitz_rate` | derived | times_blitzed / pfr_pass_attempts. |
| `pfr_pa_rate` | derived | pa_pass_att / pfr_pass_attempts. |
| `pfr_rpo_rate` | derived | rpo_plays / pfr_pass_attempts. |
| `ngs_avg_time_to_throw` | NGS ngs_passing | Average time to throw, seconds. |
| `ngs_avg_completed_air_yards` | NGS ngs_passing | Average completed air yards. |
| `ngs_avg_intended_air_yards_pass` | NGS ngs_passing | Average intended air yards (QB). |
| `ngs_avg_air_yards_differential` | NGS ngs_passing | Completed minus intended air yards. |
| `ngs_aggressiveness` | NGS ngs_passing | % of throws into tight coverage (<1 yd separation). |
| `ngs_avg_air_yards_to_sticks` | NGS ngs_passing | Air yards relative to the first-down marker. |
| `ngs_expected_completion_percentage` | NGS ngs_passing | Model-expected completion %. |
| `ngs_completion_percentage_above_expectation` | NGS ngs_passing | NGS CPOE. |
| `ngs_avg_air_distance` | NGS ngs_passing | Average air distance travelled by the ball. |
| `ngs_max_air_distance` | NGS ngs_passing | Longest air distance (window value = max over seasons). |
| `ngs_passer_rating` | NGS ngs_passing | Passer rating as published by NGS. |
| `dropbacks` | nflverse pbp | Plays with qb_dropback == 1 attributed to the QB. Scrambles are keyed on rusher_player_id in pbp and ARE included (they would otherwise vanish). |
| `qb_epa_total` | nflverse pbp | Sum of qb_epa over dropbacks. |
| `cpoe_sum` | nflverse pbp | Sum of play-level cpoe. |
| `cpoe_n` | nflverse pbp | Plays with non-null cpoe. |
| `sacks` | nflverse pbp | Sacks taken (pbp). |
| `scrambles` | nflverse pbp | Scrambles (pbp qb_scramble). Matches PFR at r = 0.999. |
| `pbp_air_yards` | nflverse pbp | Sum of air yards on dropbacks. |
| `pbp_air_yards_n` | nflverse pbp | Dropbacks with non-null air yards. |
| `qb_hits` | nflverse pbp | QB hits taken (pbp). |
| `espn_qbr` | ESPN qbr_season_level | ESPN Total QBR (0-100), regular season. |
| `espn_qbr_raw` | ESPN qbr_season_level | Raw QBR before opponent adjustment. |
| `espn_pts_added` | ESPN qbr_season_level | ESPN points added. |
| `qb_plays` | ESPN qbr_season_level | ESPN qualifying plays. |
| `epa_total` | ESPN qbr_season_level | ESPN EPA total (their model, not nflverse EPA). |
| `pbp_carries` | nflverse pbp | Carries counted from pbp (excludes 2-pt and nullified plays). |
| `gl5_carries` | nflverse pbp | Carries with yardline_100 <= 5. |
| `third_down_carries` | nflverse pbp | Carries on 3rd down. |
| `rush_epa_total` | nflverse pbp | Sum of EPA on the player's carries. |
| `comp_pct` | derived | completions / attempts. |
| `ypa` | derived | pass_yards / attempts. |
| `adot` | derived | pass_air_yards / attempts. |
| `td_rate` | derived | pass_tds / attempts. |
| `int_rate` | derived | interceptions / attempts. |
| `td_int` | derived | pass_tds / interceptions. |
| `epa_per_dropback` | derived | qb_epa_total / dropbacks. |
| `cpoe` | derived | cpoe_sum / cpoe_n. Play-weighted CPOE. |
| `sack_rate` | derived | sacks / dropbacks. |
| `scramble_rate` | derived | scrambles / dropbacks. |
| `pressure_rate_pbp` | derived | (qb_hits + sacks) / dropbacks. pbp-only pressure proxy; pfr_pressure_pct is the charted version and should be preferred. |
| `designed_rushes` | derived | carries - scrambles. Designed QB run volume. |
| `rush_epa_per_att` | derived | rush_epa_total / pbp_carries. |
| `rush_ppr` | derived | 0.1*rush_yards + 6*rush_tds. QB rushing fantasy points (no PPR component). |
| `rush_share_of_ppr` | derived | rush_ppr / ppr. The rushing share of QB fantasy output -- the quantity the section-O work found drives QB variance. |
| `attempts_pg` | nflverse stats_player (weekly) | attempts divided by games. |
| `pass_yards_pg` | nflverse stats_player (weekly) | pass_yards divided by games. |
| `pass_tds_pg` | nflverse stats_player (weekly) | pass_tds divided by games. |
| `interceptions_pg` | nflverse stats_player (weekly) | interceptions divided by games. |
| `carries_pg` | nflverse stats_player (weekly) | carries divided by games. |
| `rush_yards_pg` | nflverse stats_player (weekly) | rush_yards divided by games. |
| `rush_tds_pg` | nflverse stats_player (weekly) | rush_tds divided by games. |
| `ppr_pg` | nflverse stats_player (weekly) | ppr divided by games. |
| `dropbacks_pg` | nflverse pbp | dropbacks divided by games. |
| `designed_rushes_pg` | derived | designed_rushes / games. QB designed-run volume per game. |
| `rush_ppr_pg` | derived | rush_ppr / games. |
| `sacks_pg` | nflverse pbp | sacks divided by games. |
| `scrambles_pg` | nflverse pbp | scrambles divided by games. |
| `gl5_carries_pg` | nflverse pbp | gl5_carries divided by games. |

### `data/derived/team_context.csv` — 256 rows, 94 columns

| column | source | definition |
|---|---|---|
| `season` | nflverse stats_player (weekly) | NFL season (REG only throughout). |
| `team` | nflverse stats_player (weekly) | Modal weekly team over the player-season. Traded players get one team; n_teams flags them. |
| `games` | nflverse stats_player (weekly) | Distinct REG weeks with a stat line. |
| `pass_att` | nflverse stats_team (weekly) | Team pass attempts. |
| `pass_yards` | nflverse stats_player (weekly) | Passing yards. |
| `pass_tds` | nflverse stats_player (weekly) | Passing touchdowns. |
| `carries` | nflverse stats_player (weekly) | Rush attempts. |
| `rush_yards` | nflverse stats_player (weekly) | Rushing yards. |
| `rush_tds` | nflverse stats_player (weekly) | Rushing touchdowns. |
| `team_targets` | nflverse stats_team (weekly) | Team targets. |
| `team_air_yards` | nflverse stats_team (weekly) | Team air yards. |
| `cpoe` | nflverse stats_team (weekly) | Team mean weekly passing CPOE. |
| `total_yards` | nflverse stats_team (weekly) | pass_yards + rush_yards. |
| `off_plays` | nflverse pbp | Non-special, non-aborted offensive plays. |
| `off_epa` | nflverse pbp | Mean EPA per offensive play. |
| `pass_plays` | nflverse pbp | Plays flagged `pass`. |
| `rush_plays` | nflverse pbp | Plays flagged `rush`. |
| `proe` | nflverse pbp | Mean pass_oe: actual minus model-expected pass rate, percentage points. |
| `xpass` | nflverse pbp | Mean model-expected pass probability. |
| `neutral_plays` | nflverse pbp | Plays with win prob in [0.2, 0.8] and quarter <= 3. |
| `neutral_pass_rate` | nflverse pbp | Pass rate in neutral situations. |
| `neutral_proe` | nflverse pbp | Mean pass_oe in neutral situations. |
| `neutral_sec_per_play` | derived | Mean seconds between consecutive plays of the same drive in neutral situations, gaps outside (0, 60] dropped. PACE: lower = faster. |
| `off_pass_epa_play` | nflverse pbp | Mean EPA per dropback. |
| `off_rush_epa_play` | nflverse pbp | Mean EPA per rush. |
| `rz_drives` | nflverse pbp | Drives reaching inside the 20. |
| `rz_td_rate` | nflverse pbp | Share of red-zone drives ending in a touchdown. |
| `def_plays` | nflverse pbp | Offensive plays faced. |
| `def_epa` | nflverse pbp | Mean EPA per play allowed. |
| `def_pass_epa_play` | nflverse pbp | Mean EPA per dropback allowed. |
| `def_rush_epa_play` | nflverse pbp | Mean EPA per rush allowed. |
| `g` | nflverse schedules/games | Games in the schedule file (sanity duplicate of `games`). |
| `points_for` | nflverse schedules/games | Points scored, REG season. |
| `points_against` | nflverse schedules/games | Points allowed. |
| `pass_yards_allowed` | nflverse stats_team (weekly) | Passing yards allowed (opponents' offensive totals). |
| `rush_yards_allowed` | nflverse stats_team (weekly) | Rushing yards allowed. |
| `pass_att_faced` | nflverse stats_team (weekly) | Opponent pass attempts. |
| `carries_faced` | nflverse stats_team (weekly) | Opponent rush attempts. |
| `targets_faced` | nflverse stats_team (weekly) | Opponent targets. |
| `total_yards_allowed` | nflverse stats_team (weekly) | pass + rush yards allowed. |
| `fpa_qb` | nflverse stats_player (weekly) | PPR fantasy points allowed to opposing QBs (sum over all QBs facing this team). |
| `fpa_rb` | nflverse stats_player (weekly) | PPR fantasy points allowed to opposing RBs. |
| `fpa_te` | nflverse stats_player (weekly) | PPR fantasy points allowed to opposing TEs. |
| `fpa_wr` | nflverse stats_player (weekly) | PPR fantasy points allowed to opposing WRs. |
| `points_for_pg` | nflverse schedules/games | points_for divided by games. |
| `points_against_pg` | nflverse schedules/games | points_against divided by games. |
| `total_yards_pg` | nflverse stats_team (weekly) | total_yards divided by games. |
| `pass_yards_pg` | nflverse stats_player (weekly) | pass_yards divided by games. |
| `rush_yards_pg` | nflverse stats_player (weekly) | rush_yards divided by games. |
| `total_yards_allowed_pg` | nflverse stats_team (weekly) | total_yards_allowed divided by games. |
| `pass_yards_allowed_pg` | nflverse stats_team (weekly) | pass_yards_allowed divided by games. |
| `rush_yards_allowed_pg` | nflverse stats_team (weekly) | rush_yards_allowed divided by games. |
| `pass_att_pg` | nflverse stats_team (weekly) | pass_att divided by games. |
| `carries_pg` | nflverse stats_player (weekly) | carries divided by games. |
| `off_plays_pg` | nflverse pbp | off_plays divided by games. |
| `fpa_qb_pg` | nflverse stats_player (weekly) | fpa_qb divided by games. |
| `fpa_rb_pg` | nflverse stats_player (weekly) | fpa_rb divided by games. |
| `fpa_wr_pg` | nflverse stats_player (weekly) | fpa_wr divided by games. |
| `fpa_te_pg` | nflverse stats_player (weekly) | fpa_te divided by games. |
| `team_targets_pg` | nflverse stats_team (weekly) | team_targets divided by games. |
| `pass_rate` | derived | pass_plays / (pass_plays + rush_plays). |
| `ftn_plays` | nflverse ftn_charting | Offensive plays with FTN charting. |
| `ftn_motion_rate` | nflverse ftn_charting | Share of offensive plays with pre-snap motion. |
| `ftn_no_huddle_rate` | nflverse ftn_charting | Share of offensive plays run no-huddle. |
| `ftn_trick_rate` | nflverse ftn_charting | Share of offensive plays charted as trick plays. |
| `ftn_pa_rate` | nflverse ftn_charting | Share of dropbacks using play action. |
| `ftn_screen_rate` | nflverse ftn_charting | Share of dropbacks that were screens. |
| `ftn_rpo_rate` | nflverse ftn_charting | Share of dropbacks that were RPOs. |
| `ftn_oop_rate` | nflverse ftn_charting | Share of dropbacks with the QB out of the pocket. |
| `ftn_blitz_faced_rate` | nflverse ftn_charting | Share of the offense's dropbacks facing >=1 blitzer. |
| `ftn_pass_rushers_faced` | nflverse ftn_charting | Mean pass rushers faced per dropback (offense). |
| `ftn_def_blitz_rate` | nflverse ftn_charting | Share of dropbacks on which this DEFENSE sent >=1 blitzer. |
| `ftn_def_pass_rushers` | nflverse ftn_charting | Mean pass rushers this defense sent per dropback. |
| `ftn_def_box` | nflverse ftn_charting | Mean defenders in the box this defense showed on dropbacks. |
| `rank_points_for_pg` | derived | Within-season rank of points_for_pg across the 32 teams, 1 = best (descending for offensive/production metrics, ascending for points/yards/EPA/fantasy points ALLOWED). |
| `rank_total_yards_pg` | derived | Within-season rank of total_yards_pg across the 32 teams, 1 = best (descending for offensive/production metrics, ascending for points/yards/EPA/fantasy points ALLOWED). |
| `rank_pass_yards_pg` | derived | Within-season rank of pass_yards_pg across the 32 teams, 1 = best (descending for offensive/production metrics, ascending for points/yards/EPA/fantasy points ALLOWED). |
| `rank_rush_yards_pg` | derived | Within-season rank of rush_yards_pg across the 32 teams, 1 = best (descending for offensive/production metrics, ascending for points/yards/EPA/fantasy points ALLOWED). |
| `rank_off_epa` | derived | Within-season rank of off_epa across the 32 teams, 1 = best (descending for offensive/production metrics, ascending for points/yards/EPA/fantasy points ALLOWED). |
| `rank_off_pass_epa_play` | derived | Within-season rank of off_pass_epa_play across the 32 teams, 1 = best (descending for offensive/production metrics, ascending for points/yards/EPA/fantasy points ALLOWED). |
| `rank_off_rush_epa_play` | derived | Within-season rank of off_rush_epa_play across the 32 teams, 1 = best (descending for offensive/production metrics, ascending for points/yards/EPA/fantasy points ALLOWED). |
| `rank_rz_td_rate` | derived | Within-season rank of rz_td_rate across the 32 teams, 1 = best (descending for offensive/production metrics, ascending for points/yards/EPA/fantasy points ALLOWED). |
| `rank_off_plays_pg` | derived | Within-season rank of off_plays_pg across the 32 teams, 1 = best (descending for offensive/production metrics, ascending for points/yards/EPA/fantasy points ALLOWED). |
| `rank_points_against_pg` | derived | Within-season rank of points_against_pg across the 32 teams, 1 = best (descending for offensive/production metrics, ascending for points/yards/EPA/fantasy points ALLOWED). |
| `rank_total_yards_allowed_pg` | derived | Within-season rank of total_yards_allowed_pg across the 32 teams, 1 = best (descending for offensive/production metrics, ascending for points/yards/EPA/fantasy points ALLOWED). |
| `rank_pass_yards_allowed_pg` | derived | Within-season rank of pass_yards_allowed_pg across the 32 teams, 1 = best (descending for offensive/production metrics, ascending for points/yards/EPA/fantasy points ALLOWED). |
| `rank_rush_yards_allowed_pg` | derived | Within-season rank of rush_yards_allowed_pg across the 32 teams, 1 = best (descending for offensive/production metrics, ascending for points/yards/EPA/fantasy points ALLOWED). |
| `rank_def_epa` | derived | Within-season rank of def_epa across the 32 teams, 1 = best (descending for offensive/production metrics, ascending for points/yards/EPA/fantasy points ALLOWED). |
| `rank_def_pass_epa_play` | derived | Within-season rank of def_pass_epa_play across the 32 teams, 1 = best (descending for offensive/production metrics, ascending for points/yards/EPA/fantasy points ALLOWED). |
| `rank_def_rush_epa_play` | derived | Within-season rank of def_rush_epa_play across the 32 teams, 1 = best (descending for offensive/production metrics, ascending for points/yards/EPA/fantasy points ALLOWED). |
| `rank_fpa_qb_pg` | derived | Within-season rank of fpa_qb_pg across the 32 teams, 1 = best (descending for offensive/production metrics, ascending for points/yards/EPA/fantasy points ALLOWED). |
| `rank_fpa_rb_pg` | derived | Within-season rank of fpa_rb_pg across the 32 teams, 1 = best (descending for offensive/production metrics, ascending for points/yards/EPA/fantasy points ALLOWED). |
| `rank_fpa_wr_pg` | derived | Within-season rank of fpa_wr_pg across the 32 teams, 1 = best (descending for offensive/production metrics, ascending for points/yards/EPA/fantasy points ALLOWED). |
| `rank_fpa_te_pg` | derived | Within-season rank of fpa_te_pg across the 32 teams, 1 = best (descending for offensive/production metrics, ascending for points/yards/EPA/fantasy points ALLOWED). |

### `data/derived/adv_wr_te_recent3.csv` — 485 rows, 111 columns

| column | source | definition |
|---|---|---|
| `player_id` | nflverse stats_player (weekly) | gsis_id. Primary join key across every table. |
| `player_name` | nflverse stats_player (weekly) | Display name, first appearance in the window. |
| `position` | nflverse stats_player (weekly) | Modal weekly position over the player-season (last season in a window). |
| `team_last` | nflverse stats_player (weekly) | Team of the most recent season in the window. |
| `seasons_played` | nflverse stats_player (weekly) | Distinct seasons with a stat line inside the window. |
| `last_season` | nflverse stats_player (weekly) | Latest season in the window. |
| `first_season` | nflverse stats_player (weekly) | Earliest season in the window. |
| `n_teams` | nflverse stats_player (weekly) | Distinct teams the player recorded a game for in the period. |
| `games` | nflverse stats_player (weekly) | Distinct REG weeks with a stat line. |
| `targets` | nflverse stats_player (weekly) | Targets. Excludes two-point conversions. |
| `receptions` | nflverse stats_player (weekly) | Receptions. |
| `rec_yards` | nflverse stats_player (weekly) | Receiving yards. |
| `rec_tds` | nflverse stats_player (weekly) | Receiving touchdowns. |
| `rec_air_yards` | nflverse stats_player (weekly) | Sum of air yards on targets. |
| `rec_yac` | nflverse stats_player (weekly) | Receiving yards after catch. |
| `rec_first_downs` | nflverse stats_player (weekly) | Receiving first downs. |
| `rec_epa` | nflverse stats_player (weekly) | Sum of receiving EPA. |
| `carries` | nflverse stats_player (weekly) | Rush attempts. |
| `rush_yards` | nflverse stats_player (weekly) | Rushing yards. |
| `rush_tds` | nflverse stats_player (weekly) | Rushing touchdowns. |
| `ppr` | nflverse stats_player (weekly) | Total PPR fantasy points. |
| `tm_targets` | nflverse stats_team (weekly) | Team targets summed over the games the player was active for. |
| `tm_air_yards` | nflverse stats_team (weekly) | Team air yards over the player's active games. |
| `tm_pass_att` | nflverse stats_team (weekly) | Team pass attempts over the player's active games. |
| `tm_carries` | nflverse stats_team (weekly) | Team rush attempts over the player's active games. |
| `tm_targets_full` | nflverse stats_team (weekly) | Team targets over the FULL season. |
| `tm_air_yards_full` | nflverse stats_team (weekly) | Team air yards over the full season. |
| `tm_carries_full` | nflverse stats_team (weekly) | Team rush attempts over the full season. |
| `tm_games` | nflverse stats_team (weekly) | Team games in the season (16 pre-2021, 17 from 2021). |
| `snap_games` | nflverse snap_counts (PFR) | Games with >0 offensive snaps. |
| `off_snaps` | nflverse snap_counts (PFR) | Offensive snaps. |
| `tm_off_snaps` | nflverse snap_counts (PFR) | Team offensive snaps in the games the player was dressed (max offense_snaps among the team's players that game). |
| `st_snaps` | nflverse snap_counts (PFR) | Special-teams snaps. Low ST usage is a weak signal of offensive role. |
| `off_plays_on_field` | nflverse pbp_participation | Offensive plays with the player in offense_players. |
| `dropbacks_on_field` | nflverse pbp_participation | Team dropbacks with the player on the field. ROUTE PROXY. |
| `rushes_on_field` | nflverse pbp_participation | Team rush plays with the player on the field. |
| `tm_off_plays` | nflverse pbp_participation | Team offensive plays with participation data. |
| `tm_dropbacks` | nflverse pbp_participation | Team dropbacks with participation data. |
| `tm_rushes` | nflverse pbp_participation | Team rush plays with participation data. |
| `pfr_tgt` | PFR advstats_season_rec | PFR's own target count (denominator for its rates). |
| `pfr_rec` | PFR advstats_season_rec | PFR's own reception count. |
| `pfr_ybc` | PFR advstats_season_rec | Receiving yards before catch. |
| `pfr_yac` | PFR advstats_season_rec | Receiving yards after catch, PFR charting. |
| `pfr_broken_tackles` | PFR advstats_season_rec | Broken tackles on receptions. |
| `pfr_drops` | PFR advstats_season_rec | Charted drops. |
| `pfr_first_downs` | PFR advstats_season_rec | Receiving first downs, PFR. |
| `pbp_targets` | nflverse pbp | Targets counted from pbp (excludes 2-pt and nullified plays). Reconciles with weekly `targets` to |diff| <= 1. |
| `rz_targets` | nflverse pbp | Targets with yardline_100 <= 20. |
| `i10_targets` | nflverse pbp | Targets with yardline_100 <= 10. |
| `ez_targets` | nflverse pbp | End-zone targets: air_yards >= yardline_100. |
| `deep_targets` | nflverse pbp | Targets with air_yards >= 20. |
| `third_down_targets` | nflverse pbp | Targets on 3rd down. |
| `third_down_conv` | nflverse pbp | 3rd-down targets that produced a first down. |
| `target_epa_total` | nflverse pbp | Sum of EPA on the player's targets. |
| `play_share_part` | derived | off_plays_on_field / tm_off_plays. Participation-based snap share. |
| `target_epa` | derived | target_epa_total / pbp_targets. |
| `rz_target_share_of_own` | derived | rz_targets / pbp_targets. Share of the player's OWN targets that came in the red zone (not share of team RZ targets). |
| `ngs_avg_cushion` | NGS ngs_receiving | Average cushion (yds) from the nearest defender at snap. |
| `ngs_avg_separation` | NGS ngs_receiving | Average separation (yds) at the moment of catch/incompletion. |
| `ngs_avg_intended_air_yards` | NGS ngs_receiving | Average intended air yards on targets. |
| `ngs_percent_share_of_intended_air_yards` | NGS ngs_receiving | Share of team intended air yards (%). |
| `ngs_avg_yac` | NGS ngs_receiving | Average YAC. |
| `ngs_avg_expected_yac` | NGS ngs_receiving | Model-expected YAC given tracking context. |
| `ngs_avg_yac_above_expectation` | NGS ngs_receiving | YAC over expected, per reception. |
| `ngs_catch_percentage` | NGS ngs_receiving | NGS catch %. |
| `pfr_adot` | PFR advstats_season_rec | Average depth of target, PFR charting. |
| `pfr_ybc_per_rec` | PFR advstats_season_rec | Yards before catch per reception. |
| `pfr_yac_per_rec` | PFR advstats_season_rec | Yards after catch per reception. |
| `pfr_rec_per_broken_tackle` | PFR advstats_season_rec | Receptions per broken tackle (lower = more elusive). |
| `snap_pct_mean` | nflverse snap_counts (PFR) | Unweighted mean of PFR's per-game offense_pct. Kept as a cross-check on snap_share; games-weighted mean in a window. |
| `target_share` | derived | targets / tm_targets. Share WHILE ACTIVE. Does not sum to 1 within a team-season; use for projection, not for budget arithmetic. |
| `air_yards_share` | derived | rec_air_yards / tm_air_yards, active-games denominator. |
| `target_share_full` | derived | targets / tm_targets_full. Sums to ~1 across a team-season; use for vacated-share / budget arithmetic. Understates part-season players. |
| `air_yards_share_full` | derived | rec_air_yards / tm_air_yards_full. |
| `carry_share` | derived | carries / tm_carries, active-games denominator. |
| `carry_share_full` | derived | carries / tm_carries_full. |
| `games_played_rate` | derived | games / tm_games. Availability. |
| `wopr` | derived | 1.5*target_share + 0.7*air_yards_share (active-games denominators). |
| `racr` | derived | rec_yards / rec_air_yards. |
| `adot_nflverse` | derived | rec_air_yards / targets. nflverse-derived aDOT; compare pfr_adot. |
| `catch_rate` | derived | receptions / targets. |
| `rec_epa_per_target` | derived | rec_epa / targets. |
| `ypc` | derived | rush_yards / carries. |
| `ypr` | derived | rec_yards / receptions. |
| `snap_share` | derived | off_snaps / tm_off_snaps. |
| `off_snaps_pg` | nflverse snap_counts (PFR) | off_snaps divided by games. |
| `pass_snap_share` | derived | dropbacks_on_field / tm_dropbacks. |
| `run_snap_share` | derived | rushes_on_field / tm_rushes. |
| `targets_pg` | nflverse stats_player (weekly) | targets divided by games. |
| `rec_pg` | derived | receptions / games. |
| `rec_yards_pg` | nflverse stats_player (weekly) | rec_yards divided by games. |
| `rec_tds_pg` | nflverse stats_player (weekly) | rec_tds divided by games. |
| `air_yards_pg` | derived | rec_air_yards / games. |
| `ppr_pg` | nflverse stats_player (weekly) | ppr divided by games. |
| `carries_pg` | nflverse stats_player (weekly) | carries divided by games. |
| `rush_yards_pg` | nflverse stats_player (weekly) | rush_yards divided by games. |
| `rush_tds_pg` | nflverse stats_player (weekly) | rush_tds divided by games. |
| `routes_proxy` | derived | = dropbacks_on_field. NOT charted routes: a TE or back who stays in to block is counted as on the field, so this OVERSTATES routes run and TPRR/YPRR built on it are conservative (biased low). Bias is small for boundary WRs, material for blocking TEs and pass-protecting backs. |
| `routes_proxy_pg` | derived | routes_proxy / games. Proxy routes run per game. |
| `tprr_proxy` | derived | targets / routes_proxy. Targets per route run, proxy denominator. |
| `yprr_proxy` | derived | rec_yards / routes_proxy. Yards per route run, proxy denominator. |
| `pfr_drop_pct` | derived | pfr_drops / pfr_tgt. Rebuilt from counts so it aggregates correctly. |
| `rz_targets_pg` | nflverse pbp | rz_targets divided by games. |
| `i10_targets_pg` | nflverse pbp | i10_targets divided by games. |
| `ez_targets_pg` | nflverse pbp | ez_targets divided by games. |
| `deep_targets_pg` | nflverse pbp | deep_targets divided by games. |
| `third_down_targets_pg` | nflverse pbp | third_down_targets divided by games. |
| `deep_target_rate` | derived | deep_targets / pbp_targets. |
| `third_down_conv_rate` | derived | third_down_conv / third_down_targets. |
| `window_seasons` | derived | Window label, e.g. '2023-2025'. |
| `thin_data` | derived | TRUE if seasons_played < 2 or games < 8 in the window. Rookies and players absent from the window are simply NOT rows here -- nothing is imputed; the caller supplies them from ADP/draft capital. |

### `data/derived/adv_rb_recent3.csv` — 227 rows, 134 columns

| column | source | definition |
|---|---|---|
| `player_id` | nflverse stats_player (weekly) | gsis_id. Primary join key across every table. |
| `player_name` | nflverse stats_player (weekly) | Display name, first appearance in the window. |
| `position` | nflverse stats_player (weekly) | Modal weekly position over the player-season (last season in a window). |
| `team_last` | nflverse stats_player (weekly) | Team of the most recent season in the window. |
| `seasons_played` | nflverse stats_player (weekly) | Distinct seasons with a stat line inside the window. |
| `last_season` | nflverse stats_player (weekly) | Latest season in the window. |
| `first_season` | nflverse stats_player (weekly) | Earliest season in the window. |
| `n_teams` | nflverse stats_player (weekly) | Distinct teams the player recorded a game for in the period. |
| `games` | nflverse stats_player (weekly) | Distinct REG weeks with a stat line. |
| `targets` | nflverse stats_player (weekly) | Targets. Excludes two-point conversions. |
| `receptions` | nflverse stats_player (weekly) | Receptions. |
| `rec_yards` | nflverse stats_player (weekly) | Receiving yards. |
| `rec_tds` | nflverse stats_player (weekly) | Receiving touchdowns. |
| `rec_air_yards` | nflverse stats_player (weekly) | Sum of air yards on targets. |
| `rec_yac` | nflverse stats_player (weekly) | Receiving yards after catch. |
| `rec_first_downs` | nflverse stats_player (weekly) | Receiving first downs. |
| `rec_epa` | nflverse stats_player (weekly) | Sum of receiving EPA. |
| `carries` | nflverse stats_player (weekly) | Rush attempts. |
| `rush_yards` | nflverse stats_player (weekly) | Rushing yards. |
| `rush_tds` | nflverse stats_player (weekly) | Rushing touchdowns. |
| `ppr` | nflverse stats_player (weekly) | Total PPR fantasy points. |
| `tm_targets` | nflverse stats_team (weekly) | Team targets summed over the games the player was active for. |
| `tm_air_yards` | nflverse stats_team (weekly) | Team air yards over the player's active games. |
| `tm_pass_att` | nflverse stats_team (weekly) | Team pass attempts over the player's active games. |
| `tm_carries` | nflverse stats_team (weekly) | Team rush attempts over the player's active games. |
| `tm_targets_full` | nflverse stats_team (weekly) | Team targets over the FULL season. |
| `tm_air_yards_full` | nflverse stats_team (weekly) | Team air yards over the full season. |
| `tm_carries_full` | nflverse stats_team (weekly) | Team rush attempts over the full season. |
| `tm_games` | nflverse stats_team (weekly) | Team games in the season (16 pre-2021, 17 from 2021). |
| `snap_games` | nflverse snap_counts (PFR) | Games with >0 offensive snaps. |
| `off_snaps` | nflverse snap_counts (PFR) | Offensive snaps. |
| `tm_off_snaps` | nflverse snap_counts (PFR) | Team offensive snaps in the games the player was dressed (max offense_snaps among the team's players that game). |
| `st_snaps` | nflverse snap_counts (PFR) | Special-teams snaps. Low ST usage is a weak signal of offensive role. |
| `off_plays_on_field` | nflverse pbp_participation | Offensive plays with the player in offense_players. |
| `dropbacks_on_field` | nflverse pbp_participation | Team dropbacks with the player on the field. ROUTE PROXY. |
| `rushes_on_field` | nflverse pbp_participation | Team rush plays with the player on the field. |
| `tm_off_plays` | nflverse pbp_participation | Team offensive plays with participation data. |
| `tm_dropbacks` | nflverse pbp_participation | Team dropbacks with participation data. |
| `tm_rushes` | nflverse pbp_participation | Team rush plays with participation data. |
| `pfr_rush_att` | PFR advstats_season_rush | PFR rush attempts (denominator for its rates). |
| `pfr_rush_ybc` | PFR advstats_season_rush | Rushing yards before contact. |
| `pfr_rush_yac` | PFR advstats_season_rush | Rushing yards after contact. |
| `pfr_rush_broken_tackles` | PFR advstats_season_rush | Broken tackles on runs. |
| `pfr_rush_first_downs` | PFR advstats_season_rush | Rushing first downs, PFR. |
| `pfr_tgt` | PFR advstats_season_rec | PFR's own target count (denominator for its rates). |
| `pfr_rec` | PFR advstats_season_rec | PFR's own reception count. |
| `pfr_ybc` | PFR advstats_season_rec | Receiving yards before catch. |
| `pfr_yac` | PFR advstats_season_rec | Receiving yards after catch, PFR charting. |
| `pfr_broken_tackles` | PFR advstats_season_rec | Broken tackles on receptions. |
| `pfr_drops` | PFR advstats_season_rec | Charted drops. |
| `pfr_first_downs` | PFR advstats_season_rec | Receiving first downs, PFR. |
| `ngs_expected_rush_yards` | NGS ngs_rushing | Model-expected rush yards, season total. |
| `ngs_rush_yards_over_expected` | NGS ngs_rushing | RYOE, season total. |
| `pbp_carries` | nflverse pbp | Carries counted from pbp (excludes 2-pt and nullified plays). |
| `gl5_carries` | nflverse pbp | Carries with yardline_100 <= 5. |
| `gl10_carries` | nflverse pbp | Carries with yardline_100 <= 10. |
| `goal_to_go_carries` | nflverse pbp | Carries in goal-to-go situations. |
| `stuffed` | nflverse pbp | Carries gaining <= 0 yards. |
| `third_down_carries` | nflverse pbp | Carries on 3rd down. |
| `short_yd_carries` | nflverse pbp | Carries on 3rd/4th down with <= 2 to go. |
| `short_yd_conv` | nflverse pbp | Short-yardage carries converting a first down. |
| `explosive_runs` | nflverse pbp | Carries gaining >= 10 yards. |
| `rush_epa_total` | nflverse pbp | Sum of EPA on the player's carries. |
| `pbp_targets` | nflverse pbp | Targets counted from pbp (excludes 2-pt and nullified plays). Reconciles with weekly `targets` to |diff| <= 1. |
| `rz_targets` | nflverse pbp | Targets with yardline_100 <= 20. |
| `i10_targets` | nflverse pbp | Targets with yardline_100 <= 10. |
| `ez_targets` | nflverse pbp | End-zone targets: air_yards >= yardline_100. |
| `deep_targets` | nflverse pbp | Targets with air_yards >= 20. |
| `third_down_targets` | nflverse pbp | Targets on 3rd down. |
| `third_down_conv` | nflverse pbp | 3rd-down targets that produced a first down. |
| `target_epa_total` | nflverse pbp | Sum of EPA on the player's targets. |
| `play_share_part` | derived | off_plays_on_field / tm_off_plays. Participation-based snap share. |
| `touches` | derived | carries + receptions. |
| `target_epa` | derived | target_epa_total / pbp_targets. |
| `gl5_carry_share_of_own` | derived | gl5_carries / pbp_carries. |
| `ngs_efficiency` | NGS ngs_rushing | Distance travelled per yard gained (lower = more direct). |
| `ngs_percent_attempts_gte_eight_defenders` | NGS ngs_rushing | % of carries against 8+ in the box. |
| `ngs_avg_time_to_los` | NGS ngs_rushing | Average seconds to line of scrimmage. |
| `ngs_rush_yards_over_expected_per_att` | NGS ngs_rushing | RYOE per attempt, as published. |
| `ngs_rush_pct_over_expected` | NGS ngs_rushing | % rush yards over expected. |
| `pfr_adot` | PFR advstats_season_rec | Average depth of target, PFR charting. |
| `pfr_ybc_per_rec` | PFR advstats_season_rec | Yards before catch per reception. |
| `pfr_yac_per_rec` | PFR advstats_season_rec | Yards after catch per reception. |
| `pfr_rec_per_broken_tackle` | PFR advstats_season_rec | Receptions per broken tackle (lower = more elusive). |
| `pfr_ybc_per_att` | PFR advstats_season_rush | Yards before contact per attempt. Blocking/scheme-heavy. |
| `pfr_yac_per_att` | PFR advstats_season_rush | Yards after contact per attempt. Runner-heavy. |
| `pfr_att_per_broken_tackle` | PFR advstats_season_rush | Attempts per broken tackle (lower = more elusive). |
| `snap_pct_mean` | nflverse snap_counts (PFR) | Unweighted mean of PFR's per-game offense_pct. Kept as a cross-check on snap_share; games-weighted mean in a window. |
| `target_share` | derived | targets / tm_targets. Share WHILE ACTIVE. Does not sum to 1 within a team-season; use for projection, not for budget arithmetic. |
| `air_yards_share` | derived | rec_air_yards / tm_air_yards, active-games denominator. |
| `target_share_full` | derived | targets / tm_targets_full. Sums to ~1 across a team-season; use for vacated-share / budget arithmetic. Understates part-season players. |
| `air_yards_share_full` | derived | rec_air_yards / tm_air_yards_full. |
| `carry_share` | derived | carries / tm_carries, active-games denominator. |
| `carry_share_full` | derived | carries / tm_carries_full. |
| `games_played_rate` | derived | games / tm_games. Availability. |
| `wopr` | derived | 1.5*target_share + 0.7*air_yards_share (active-games denominators). |
| `racr` | derived | rec_yards / rec_air_yards. |
| `adot_nflverse` | derived | rec_air_yards / targets. nflverse-derived aDOT; compare pfr_adot. |
| `catch_rate` | derived | receptions / targets. |
| `rec_epa_per_target` | derived | rec_epa / targets. |
| `ypc` | derived | rush_yards / carries. |
| `ypr` | derived | rec_yards / receptions. |
| `snap_share` | derived | off_snaps / tm_off_snaps. |
| `off_snaps_pg` | nflverse snap_counts (PFR) | off_snaps divided by games. |
| `pass_snap_share` | derived | dropbacks_on_field / tm_dropbacks. |
| `run_snap_share` | derived | rushes_on_field / tm_rushes. |
| `targets_pg` | nflverse stats_player (weekly) | targets divided by games. |
| `rec_pg` | derived | receptions / games. |
| `rec_yards_pg` | nflverse stats_player (weekly) | rec_yards divided by games. |
| `rec_tds_pg` | nflverse stats_player (weekly) | rec_tds divided by games. |
| `air_yards_pg` | derived | rec_air_yards / games. |
| `ppr_pg` | nflverse stats_player (weekly) | ppr divided by games. |
| `carries_pg` | nflverse stats_player (weekly) | carries divided by games. |
| `rush_yards_pg` | nflverse stats_player (weekly) | rush_yards divided by games. |
| `rush_tds_pg` | nflverse stats_player (weekly) | rush_tds divided by games. |
| `touches_pg` | derived | touches / games. |
| `opportunity_pg` | derived | (carries + targets) / games. Opportunity volume. |
| `stuffed_rate` | derived | stuffed / pbp_carries. |
| `short_yd_conv_rate` | derived | short_yd_conv / short_yd_carries. |
| `explosive_run_rate` | derived | explosive_runs / pbp_carries. |
| `rush_epa_per_att` | derived | rush_epa_total / pbp_carries. |
| `third_down_carry_rate` | derived | third_down_carries / pbp_carries. |
| `pfr_drop_pct` | derived | pfr_drops / pfr_tgt. Rebuilt from counts so it aggregates correctly. |
| `ngs_ryoe_per_att` | derived | ngs_rush_yards_over_expected / carries. Rebuilt from totals so it aggregates correctly over a window; compare to the published per-att. |
| `routes_proxy` | derived | = dropbacks_on_field. NOT charted routes: a TE or back who stays in to block is counted as on the field, so this OVERSTATES routes run and TPRR/YPRR built on it are conservative (biased low). Bias is small for boundary WRs, material for blocking TEs and pass-protecting backs. |
| `tprr_proxy` | derived | targets / routes_proxy. Targets per route run, proxy denominator. |
| `gl5_carries_pg` | nflverse pbp | gl5_carries divided by games. |
| `gl10_carries_pg` | nflverse pbp | gl10_carries divided by games. |
| `goal_to_go_carries_pg` | nflverse pbp | goal_to_go_carries divided by games. |
| `third_down_carries_pg` | nflverse pbp | third_down_carries divided by games. |
| `explosive_runs_pg` | nflverse pbp | explosive_runs divided by games. |
| `rz_targets_pg` | nflverse pbp | rz_targets divided by games. |
| `window_seasons` | derived | Window label, e.g. '2023-2025'. |
| `thin_data` | derived | TRUE if seasons_played < 2 or games < 8 in the window. Rookies and players absent from the window are simply NOT rows here -- nothing is imputed; the caller supplies them from ADP/draft capital. |

### `data/derived/adv_qb_recent3.csv` — 107 rows, 104 columns

| column | source | definition |
|---|---|---|
| `player_id` | nflverse stats_player (weekly) | gsis_id. Primary join key across every table. |
| `player_name` | nflverse stats_player (weekly) | Display name, first appearance in the window. |
| `position` | nflverse stats_player (weekly) | Modal weekly position over the player-season (last season in a window). |
| `team_last` | nflverse stats_player (weekly) | Team of the most recent season in the window. |
| `seasons_played` | nflverse stats_player (weekly) | Distinct seasons with a stat line inside the window. |
| `last_season` | nflverse stats_player (weekly) | Latest season in the window. |
| `first_season` | nflverse stats_player (weekly) | Earliest season in the window. |
| `n_teams` | nflverse stats_player (weekly) | Distinct teams the player recorded a game for in the period. |
| `games` | nflverse stats_player (weekly) | Distinct REG weeks with a stat line. |
| `attempts` | nflverse stats_player (weekly) | Pass attempts. |
| `completions` | nflverse stats_player (weekly) | Completions. |
| `pass_yards` | nflverse stats_player (weekly) | Passing yards. |
| `pass_tds` | nflverse stats_player (weekly) | Passing touchdowns. |
| `interceptions` | nflverse stats_player (weekly) | Interceptions thrown. |
| `sacks_suffered` | nflverse stats_player (weekly) | Sacks taken (weekly source). |
| `pass_air_yards` | nflverse stats_player (weekly) | Passing air yards. |
| `pass_yac` | nflverse stats_player (weekly) | Passing yards after catch. |
| `pass_epa` | nflverse stats_player (weekly) | Sum of passing EPA (weekly source). |
| `carries` | nflverse stats_player (weekly) | Rush attempts. |
| `rush_yards` | nflverse stats_player (weekly) | Rushing yards. |
| `rush_tds` | nflverse stats_player (weekly) | Rushing touchdowns. |
| `rush_epa` | nflverse stats_player (weekly) | Sum of rushing EPA (weekly source). |
| `ppr` | nflverse stats_player (weekly) | Total PPR fantasy points. |
| `times_blitzed` | PFR advstats_season_pass | Dropbacks facing a blitz. |
| `times_hurried` | PFR advstats_season_pass | Hurries allowed. |
| `times_hit` | PFR advstats_season_pass | QB hits taken. |
| `times_pressured` | PFR advstats_season_pass | Pressures faced (hurry + hit + sack). |
| `bad_throws` | PFR advstats_season_pass | Charted bad throws. |
| `throwaways` | PFR advstats_season_pass | Deliberate throwaways. |
| `batted_balls` | PFR advstats_season_pass | Passes batted at the line. |
| `rpo_plays` | PFR advstats_season_pass | RPO plays. |
| `pa_pass_att` | PFR advstats_season_pass | Play-action pass attempts. |
| `pfr_scrambles` | PFR advstats_season_pass | Scrambles, PFR charting (cross-check on `scrambles`). |
| `pfr_pass_attempts` | PFR advstats_season_pass | PFR pass attempts (denominator for its rates). |
| `dropbacks` | nflverse pbp | Plays with qb_dropback == 1 attributed to the QB. Scrambles are keyed on rusher_player_id in pbp and ARE included (they would otherwise vanish). |
| `qb_epa_total` | nflverse pbp | Sum of qb_epa over dropbacks. |
| `cpoe_sum` | nflverse pbp | Sum of play-level cpoe. |
| `cpoe_n` | nflverse pbp | Plays with non-null cpoe. |
| `sacks` | nflverse pbp | Sacks taken (pbp). |
| `scrambles` | nflverse pbp | Scrambles (pbp qb_scramble). Matches PFR at r = 0.999. |
| `pbp_air_yards` | nflverse pbp | Sum of air yards on dropbacks. |
| `pbp_air_yards_n` | nflverse pbp | Dropbacks with non-null air yards. |
| `qb_hits` | nflverse pbp | QB hits taken (pbp). |
| `espn_pts_added` | ESPN qbr_season_level | ESPN points added. |
| `qb_plays` | ESPN qbr_season_level | ESPN qualifying plays. |
| `epa_total` | ESPN qbr_season_level | ESPN EPA total (their model, not nflverse EPA). |
| `pbp_carries` | nflverse pbp | Carries counted from pbp (excludes 2-pt and nullified plays). |
| `gl5_carries` | nflverse pbp | Carries with yardline_100 <= 5. |
| `third_down_carries` | nflverse pbp | Carries on 3rd down. |
| `rush_epa_total` | nflverse pbp | Sum of EPA on the player's carries. |
| `ypa` | derived | pass_yards / attempts. |
| `adot` | derived | pass_air_yards / attempts. |
| `td_int` | derived | pass_tds / interceptions. |
| `cpoe` | derived | cpoe_sum / cpoe_n. Play-weighted CPOE. |
| `pressure_rate_pbp` | derived | (qb_hits + sacks) / dropbacks. pbp-only pressure proxy; pfr_pressure_pct is the charted version and should be preferred. |
| `designed_rushes` | derived | carries - scrambles. Designed QB run volume. |
| `rush_ppr` | derived | 0.1*rush_yards + 6*rush_tds. QB rushing fantasy points (no PPR component). |
| `rush_share_of_ppr` | derived | rush_ppr / ppr. The rushing share of QB fantasy output -- the quantity the section-O work found drives QB variance. |
| `ngs_avg_time_to_throw` | NGS ngs_passing | Average time to throw, seconds. |
| `ngs_avg_completed_air_yards` | NGS ngs_passing | Average completed air yards. |
| `ngs_avg_intended_air_yards_pass` | NGS ngs_passing | Average intended air yards (QB). |
| `ngs_avg_air_yards_differential` | NGS ngs_passing | Completed minus intended air yards. |
| `ngs_aggressiveness` | NGS ngs_passing | % of throws into tight coverage (<1 yd separation). |
| `ngs_avg_air_yards_to_sticks` | NGS ngs_passing | Air yards relative to the first-down marker. |
| `ngs_expected_completion_percentage` | NGS ngs_passing | Model-expected completion %. |
| `ngs_completion_percentage_above_expectation` | NGS ngs_passing | NGS CPOE. |
| `ngs_avg_air_distance` | NGS ngs_passing | Average air distance travelled by the ball. |
| `ngs_passer_rating` | NGS ngs_passing | Passer rating as published by NGS. |
| `pfr_pocket_time` | PFR advstats_season_pass | Average time in pocket, seconds. |
| `pfr_scramble_ypa` | PFR advstats_season_pass | Yards per scramble. |
| `pfr_iay_per_att` | PFR advstats_season_pass | Intended air yards per attempt. |
| `pfr_cay_per_comp` | PFR advstats_season_pass | Completed air yards per completion. |
| `cpoe_w` | nflverse stats_player (weekly) | Weekly passing_cpoe averaged (attempt-weighted in a window). Cross-check on `cpoe`. |
| `espn_qbr` | ESPN qbr_season_level | ESPN Total QBR (0-100), regular season. |
| `espn_qbr_raw` | ESPN qbr_season_level | Raw QBR before opponent adjustment. |
| `ngs_max_air_distance` | NGS ngs_passing | Longest air distance (window value = max over seasons). |
| `comp_pct` | derived | completions / attempts. |
| `td_rate` | derived | pass_tds / attempts. |
| `int_rate` | derived | interceptions / attempts. |
| `epa_per_dropback` | derived | qb_epa_total / dropbacks. |
| `sack_rate` | derived | sacks / dropbacks. |
| `scramble_rate` | derived | scrambles / dropbacks. |
| `pfr_pressure_pct` | derived | 100 * times_pressured / pfr_pass_attempts. |
| `pfr_bad_throw_pct` | derived | 100 * bad_throws / pfr_pass_attempts. |
| `pfr_blitz_rate` | derived | times_blitzed / pfr_pass_attempts. |
| `pfr_pa_rate` | derived | pa_pass_att / pfr_pass_attempts. |
| `pfr_rpo_rate` | derived | rpo_plays / pfr_pass_attempts. |
| `rush_epa_per_att` | derived | rush_epa_total / pbp_carries. |
| `attempts_pg` | nflverse stats_player (weekly) | attempts divided by games. |
| `pass_yards_pg` | nflverse stats_player (weekly) | pass_yards divided by games. |
| `pass_tds_pg` | nflverse stats_player (weekly) | pass_tds divided by games. |
| `interceptions_pg` | nflverse stats_player (weekly) | interceptions divided by games. |
| `carries_pg` | nflverse stats_player (weekly) | carries divided by games. |
| `rush_yards_pg` | nflverse stats_player (weekly) | rush_yards divided by games. |
| `rush_tds_pg` | nflverse stats_player (weekly) | rush_tds divided by games. |
| `ppr_pg` | nflverse stats_player (weekly) | ppr divided by games. |
| `dropbacks_pg` | nflverse pbp | dropbacks divided by games. |
| `designed_rushes_pg` | derived | designed_rushes / games. QB designed-run volume per game. |
| `rush_ppr_pg` | derived | rush_ppr / games. |
| `sacks_pg` | nflverse pbp | sacks divided by games. |
| `scrambles_pg` | nflverse pbp | scrambles divided by games. |
| `gl5_carries_pg` | nflverse pbp | gl5_carries divided by games. |
| `window_seasons` | derived | Window label, e.g. '2023-2025'. |
| `thin_data` | derived | TRUE if seasons_played < 2 or games < 8 in the window. Rookies and players absent from the window are simply NOT rows here -- nothing is imputed; the caller supplies them from ADP/draft capital. |

## Not obtainable this session (needs web search / paywalled sources)

These were requested and are genuinely absent — no weaker substitute has been silently swapped in:

1. **2026 offensive-line projections.** No forward-looking OL data exists in nflverse.
2. **2026 team offensive/defensive projections.** Everything here is realised 2018–2025.
3. **Third-party O-line rankings** (PFF grades, ESPN run-block / pass-block win rate). Paywalled or search-gated.
4. **College production data** for 2026 draftees. Separate sourcing problem, needs web.
5. **Charted routes run.** Not in any free nflverse release; `routes_proxy` stands in.

Nearest available in-sample stand-ins, to be used knowingly and not as substitutes: `pfr_ybc_per_att` and `pfr_pressure_pct` carry a large offensive-line component, and `ngs_percent_attempts_gte_eight_defenders` carries box-count context. All are backward-looking.

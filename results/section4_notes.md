# §4 notes — stat reliability gate (2026-07-14)

## Sample (pre-specified before any results)

Reliability is a property of the stat, not of our 30 players, so the primary sample is
**all position == WR players** in `data/players/weekly_raw/stats_player_week_{2014..2025}.csv`,
under the §0 game-inclusion rule reused verbatim from §1 (REG only, drop player-games with
targets ≤ 1): 20,431 included player-games, 691 players. Top-30-only and 2021–2025-only are
reported sensitivities, not gates.

- Split-half: player-seasons with ≥ 10 included games (1,099 player-seasons, 350 players),
  odd vs even calendar weeks. Ratio stats computed per half as (Σ numerator)/(Σ denominator) —
  never means of per-game ratios. Share denominators (team pass attempts, team passing air
  yards) joined from `data/teams/stats_team_week_{year}.csv` so the sum/sum construction is
  exact. ρ_full = 2r_half/(1+r_half) (plan eq. 8).
- Year-over-year: season-level stat (same sum/sum construction) on consecutive-season pairs,
  both seasons ≥ 8 included games (816 pairs, 266 players).
- Bootstrap 95% CIs: resample players (cluster bootstrap), 2,000 reps, percentile.
- **Admission rule (pre-registered): ρ_full ≥ 0.5 AND r_YoY bootstrap CI excluding 0.**

## Verdicts (primary sample)

| stat | r_half | ρ_full [95% CI] | r_YoY [95% CI] | verdict |
|---|---|---|---|---|
| target_share | .811 | **.896** [.881, .909] | .703 [.652, .742] | **ADMIT** |
| air_yards_share | .777 | **.874** [.857, .890] | .709 [.662, .750] | **ADMIT** |
| WOPR | .807 | **.893** [.878, .906] | .708 [.660, .747] | **ADMIT** |
| aDOT | .679 | **.809** [.776, .835] | .666 [.605, .711] | **ADMIT** |
| RACR | .347 | **.515** [.481, .785] | .313 [.221, .593] | **ADMIT** (marginal) |
| yards/target | .200 | .334 [.254, .410] | .215 [.138, .281] | REJECT |
| TD/target | .133 | .235 [.125, .336] | .122 [.048, .195] | REJECT |
| receiving EPA/gm | .293 | .453 [.370, .518] | .321 [.229, .397] | REJECT |
| PPR PPG | .669 | **.802** [.770, .829] | .644 [.590, .690] | **ADMIT** |

Pattern matches the plan's stated expectation (checked, not enforced): usage stats are sticky,
per-target efficiency is mostly noise at season resolution. For every admitted stat
r_YoY < ρ_full, i.e. φ = r_YoY/ρ_full ≈ 0.78–0.82 for usage stats — measurement is good and
the residual YoY decay is true role movement, exactly the decomposition of plan §4.2.
Note receiving EPA/gm (ρ_full .453) and yards/target fail primarily on ρ_full; both also have
much lower r_YoY, so they are noisy *and* unstable — no near-miss controversy except RACR.

## Anomalies chased

1. **RACR's skewed bootstrap CIs (point .515 hugging the lower bound of [.48, .79]).**
   Traced to a handful of gadget/screen player-seasons with near-zero aDOT (season aDOT
   0.46–1.7): RACR = rec yards / air yards explodes as the denominator → 0 (season values
   3.5–10.5; one half-season value 21.6). Rank (Spearman) and 1/99-winsorized Pearson variants:
   split-half .44/.45 vs raw .35; YoY .50/.55 vs raw .31. So RACR's raw Pearson reliability is
   attenuated by ratio outliers, not by broadband noise, and the bootstrap skew is exactly the
   reps that do or don't resample those few players. Verdict is ADMIT either way; the headline
   number stays raw-Pearson as specified, with the caveat that RACR should enter any later model
   in robustified form (or with an aDOT floor) if used at all.

2. **Top-30 sensitivity: reliability drops for shares/PPG but rises for aDOT.** Decomposed
   signal vs noise via σ̂²_h = Var(A−B)/2: the top-30 sample compresses the between-player-season
   signal SD (air-yards share 0.067 vs 0.091; PPG 3.76 vs 4.25) while half-season noise is as
   large or larger — classic range restriction, the drop is mechanical, not evidence the stats
   are less measurable for stars. aDOT is the exception because its signal SD barely compresses
   (style, not quality, so stars still span 6–15 yds) and its noise shrinks with their high
   target counts: ρ_full .845 top-30 vs .809 primary. Receiving EPA/gm in top-30 falls to .275
   for the same reason (signal SD unchanged 1.52 vs 1.54, noise up 1.84 vs 1.61).

3. **2021–2025 window: no material differences** (usage ρ_full .81–.91, efficiency still fails).
   Verdicts are window-stable.

## Files
- `results/reliability_table.csv` — long format, samples × stats, CIs, verdicts.
- Script: `scripts/04_section4_reliability.py` (rerunnable; seed fixed).

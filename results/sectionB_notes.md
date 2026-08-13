# §B notes — situation change: carryover, level shift, market test (2026-07-15)

Pre-registration = docstring of `scripts/14_situation_change.py` (definitions fixed before
fitting). Derived inputs written first (D0, `data/derived/`): `qb_by_team_season.csv`
(384 team-seasons 2014-2025, mean primary-QB attempt share 0.829), `situation_change.csv`
(1,305 gated WR-seasons, ≥3 targets/appearance, 2015-2025), `vacated_targets.csv`
(352 team-seasons 2015-2025). Sanity: league-mean vacated share 0.294 (range of season
means 0.251-0.334); team-level primary-QB turnover 28-47%/yr; player-level rates among
gated WRs with a prior season: team_change 0.213, qb_change_same_team 0.290, any 0.503.
Face validity of vacated extremes checked: SF 2017 (0.83), LV 2019 (0.74), LAC 2024
(0.72, Allen/Williams), HOU 2023 (0.69), TEN 2022 (0.68, A.J. Brown) — all real
mass-departure situations; near-zero rows (ARI 2016, NYG/ATL 2025) are full-retention
rosters. Team codes are franchise-normalized in the nflverse cache (verified), so
relocations do not create spurious team changes.

## B1 — carryover degradation (n = 958 pairs; regimes 466 / 226 team-change / 266 QB-change)

X_{s+1} ~ regime + regime:X_s, player-clustered. Slopes (95% CI):

| stat | no change | team change | QB change (same team) | equality F(2), p |
|---|---|---|---|---|
| target share | 0.868 (0.798, 0.937) | 0.857 (0.741, 0.974) | 0.812 (0.719, 0.904) | 0.52, p = .59 |
| WOPR | 0.870 (0.801, 0.939) | 0.859 (0.735, 0.983) | 0.803 (0.708, 0.898) | 0.71, p = .49 |
| aDOT | 0.721 (0.618, 0.823) | 0.591 (0.441, 0.740) | 0.701 (0.584, 0.817) | 1.04, p = .36 |

**Verdict: no detectable carryover degradation.** Point estimates order as expected
(aDOT drops from 0.72 to 0.59 under a team change — role is team-assigned), but no
equality test approaches significance, and the pre-specified sensitivity without the
s+1 ≥4-appearance screen agrees (p = .31 / .18 / .79). The stickiness that admitted
these stats in §4 survives context turnover. Reported plainly as a negative.

## B2 — level shift (n = 958, season(s+1)-clustered)

ΔPPG_{s→s+1} = −1.38 (const, p=.001) − **1.00·team_change (p=.009)** + 0.20·qb_change
(p=.29) − 0.19·z_att_new (p=.15) − 0.00·z_epa_new (p=.99) + **1.92·vacated_new (p=.004)**.

Two real effects; both chased:
1. **team_change −1.0 PPG.** Not one season's doing: drop-one-season betas span
   −0.83 to −1.16. Not pure mean-reversion composition either: movers start lower
   (8.95 vs 11.28 PPG), but adding PPG_s as control makes the coefficient *more*
   negative (−1.58, t = −5.2). Changing teams is a genuine level-down event on average.
2. **vacated_new +1.9 PPG is an incumbent effect, not a mover effect.** Interaction
   team_change × vacated ≈ +0.99 (p = .63); stayers-only beta 1.80 (p = .013). Mechanism:
   controlling ΔTS collapses it to 0.82 (p = .053) with ΔTS·β ≈ 58.5 PPG per unit share —
   i.e. mostly mechanical target redistribution to incumbents when teammates leave, plus
   a residual (efficiency/role) remainder. Top rows are exactly that pattern (Collins→
   HOU-2023, McLaurin 2023, Johnston 2024 post-Allen), no single row decisive.
   z_epa_new is dead (−0.00) — landing-spot pass quality does not move ΔPPG once
   attempts/vacated are in.

## B3 — market test, full edge protocol (n = 291 in_fit, 2015-2024; fit 231 / holdout 60)

resid_iso ~ const + {tc_x_vacated, tc_x_epa_new, qb_change_same_team}; season-clustered
t(9); BH-FDR q = 0.10 over the 3 p's; holdout 2015-2022 → 2023-2024 vs zero prediction.

| term | β | t(9) | p_cluster | FDR | holdout | final |
|---|---|---|---|---|---|---|
| tc_x_vacated | −2.28 | −1.07 | .314 | fail | — | **no** |
| tc_x_epa_new | −1.65 | −1.70 | .122 | fail | — | **no** |
| qb_change_same_team | −0.41 | −1.37 | .205 | fail | — | **no** |

**Null. Nothing passes FDR; for the record the full 3-term family also fails the holdout
(MSE 9.534 vs zero 9.336).** Coherent with B2 + round-1 §6: situation change has real
PPG consequences (B2), but the top-30 ADP market already prices them — the residual
carries no exploitable situation-change signal. All three point estimates are negative
(movers/QB-change players slightly *over*priced if anything), none distinguishable from
zero with 10 season clusters. Consistent with script 09's team_change/vet_x_change null.
This null is the result; nothing from §B enters the posterior prior mean.

## Files
- `data/derived/qb_by_team_season.csv`, `data/derived/situation_change.csv`,
  `data/derived/vacated_targets.csv` (D0; raw caches untouched).
- `results/situation_change.csv` — B1 slopes/CIs/Wald + sensitivity rows + B2 betas.
- `results/edge_situation.csv` — B3 betas, clustered p's, FDR, holdout MSEs, verdicts.
- Script: `scripts/14_situation_change.py` (deterministic, no RNG).

## Deviations from plan text
- §B panel availability screen (≥4 REG appearances in s+1) pre-specified here, with the
  no-screen sensitivity reported (same conclusions).
- B3 family is exactly the plan's three interactions/indicators; main effects not added
  (would change the pre-specified family).
- Rookies/no-prior players in B3 carry 0 on all family terms (change undefined),
  matching the round-1 script-09 convention.

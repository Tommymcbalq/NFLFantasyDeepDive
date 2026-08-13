# EDA Round 3 — 2026 Context in the Data Arm, and Teammate Coherence
### Pre-registered 2026-07-16, before any round-3 fitting. Rules as always: no tuning, no
named-player anchors, anomalies chased, adoption only on pre-specified LOSO evidence
(DM vs frozen arm (ii), clustered by year, t(9df)), market-edge claims need FDR + holdout.

Motivation (stated before fitting): θ*'s data arm μ̂ ignores context known at draft time —
offseason team changes and vacated targets — while the market arm prices them (round-2 B2/B3).
And valuations are computed independently per player although within-team target share sums
to 1; ~⅓ of the 2026 board shares a team with another board WR.

## §G0 New data
- Sleeper API `https://api.sleeper.app/v1/players/nfl` → current (2026) team per player.
  Cache raw to data/sleeper/players_nfl_2026.json (dated); map to gsis_id via name+position
  normalization with an explicit unmatched report. Cross-check board players' teams against
  the FFC ADP team field; report disagreements.
- Derived: mover flag per 2026 board player (2025 primary team ≠ current team);
  vacated_2026.csv per team = share of that team's 2025 targets to players no longer on the
  team per Sleeper. Historical analogues already exist (data/derived/, round 2).

## §E Context-adjusted data arm — LOSO first, then the board
Adjustment uses ROUND-2 B2 estimates refit per LOSO fold on ≤Y−1 data (no peeking):
  μ̂ᶜ_i = μ̂_i + β̂_tc·1{mover} + β̂_vac·(vacated share of i's entering team) − β̂_vac·(mean vacated)
(centering so non-movers on average-turnover teams are untouched). V unchanged; θ*ᶜ per eq. (7).
- Arm (vii) in the LOSO harness (movers/vacated are preseason-knowable historically).
- Adoption rule (pre-specified): adopt for the 2026 board iff DM vs (ii) has p < 0.10 AND
  RMSE improves. Otherwise report and stop; the not-adopted 2026 board is still produced for
  the record (results/sectionE_2026.csv), clearly labeled.

## §F Teammate coherence
- F1 (measurement): implied target share for each board player-season: invert the fold-fit
  PPG↔(TS, team attempts) relation (simple OLS of PPG on TS×attempts within fold) to map θ*
  to implied TS. For teams with ≥2 board WRs, compute the implied duo sum and compare with
  the historical distribution of realized top-2 WR TS sums (all teams 2014–2025); flag duos
  above its p90.
- F2 (edge test, full protocol): on the 2015–2024 panel, regress market residual R on
  {teammate-on-board indicator, duo implied-TS sum (centered), interaction}. FDR q=0.10 +
  temporal holdout 2015–22→2023–24. Null is a result.
- F3 (constraint arm, only if F2 shows the market does NOT already price duo infeasibility):
  arm (viii) = θ* with within-team proportional scaling of data-arm means so implied duo TS
  sum ≤ historical p95; LOSO; same adoption rule as §E. If F2 is null, F3 is not run
  (the market already handles it; a constraint could only add noise) — decision rule fixed now.
- 2026 descriptive output regardless: implied TS sums for LAR/DAL/CIN/DET/CHI duos vs the
  historical distribution (results/teammate_coherence_2026.csv).

## Outputs
results/sectionE_notes.md, sectionF_notes.md, loso_scorecard3.csv, valuation_2026_v3.csv
(adopted arms only; restates prior values with verdicts if nothing adopted),
scripts/17_context_arm.py, 18_teammate_coherence.py. Figures fig13+ optional.

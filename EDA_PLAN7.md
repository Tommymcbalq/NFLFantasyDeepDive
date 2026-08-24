# EDA Round 7 — Is the mean the right summary of a player's history?
### Pre-registered 2026-08-18, before any fitting. Rules unchanged: no tuning toward expected
results, anomalies chased, adoption only on pre-specified LOSO evidence (DM vs the frozen arm,
clustered by year, t(9df)), and any claim of improvement needs the temporal holdout too.

## Motivation

The board's data arm is θ* = (1−B)·μ̂ + B·m(ADP), where **μ̂ is a recency-weighted mean of season
means** (h = 1). Six rounds have tested what to *add* to that blend — age (§C), usage projection
(§D), situation change (§E), teammate structure (§F), win totals (§I3), schedule (§K), environment
(§N) — and every one failed. **Not once has the estimator of central tendency itself been tested.**

The owner's objection is that a player mean is among the weakest available predictors. The project's
own numbers give that partial support: §2 put the ceiling on predictability at ρ_max ≈ .41, and §P
found the mean's deviation from price is worth +1.101 of face value only when the player played ≥12
games the prior season, and **+0.042 when he did not**. So μ̂ is known to be fragile in exactly the
regime where roles change — but no alternative has been tried.

## §S1 The bake-off — candidate summaries of a player's history

All candidates replace **only** μ̂ inside eq. (7). B, V, τ²(tier), m(·) and every other component are
held fixed, so the comparison isolates the summary statistic. Fixed now, no additions later:

1. **μ̂ recency-weighted mean, h = 1** — the incumbent, frozen benchmark.
2. **Median of game-level PPG** — robust to the boom weeks that drag a mean.
3. **Trimmed mean (20% each tail)** at game level.
4. **Huber M-estimator** at game level, tuning constant fixed at 1.345σ.
5. **p60 of the game-level distribution** — deliberately above median: if what persists is a
   player's *typical good day* rather than his average, this should win.
6. **Weighted by role stability**: μ̂ computed over only those prior seasons with ≥ 12 games,
   falling back to all seasons when none qualify. Directly operationalises the §P finding.
7. **Two-season slope-adjusted level**: level plus a shrunken trend term, to test whether direction
   of travel adds anything beyond level.
8. **Usage-implied mean**: prior-season target share × team volume mapped to PPG via a within-fold
   OLS. §D tested a ridge on many usage covariates and failed; this is the single-covariate version.

## §S2 Protocol

Leave-one-season-out over the 2015–2024 board panel, WR and RB separately, using the **existing**
fold structure so results are comparable to §7 and §P. Loss is squared error on realised PPG.
Comparison is each candidate vs candidate 1 by Diebold–Mariano clustered by year, t(9 df).

**Adoption rule, fixed now:** replace μ̂ only if a candidate beats it at **p < 0.05** *and* improves
RMSE *and* survives the temporal holdout (2015–21 → 2022–24). A single declared family across the
seven challengers, BH q = 0.10. Ties go to the incumbent — this is a replacement decision, not a
search for the best in-sample fit.

**Pre-registered expectations, recorded so they can be falsified:** candidates 2–5 differ from the
mean mainly in how they treat boom weeks, and §37 found dispersion does not persist year to year
(r(IQR) = .19), so they are expected to be *close to indistinguishable* from the mean. Candidate 6
is the one with a mechanism behind it. If nothing wins, that is the finding, and it means the mean
survives not because it is good but because nothing simple is better at this sample size.

## §S3 Power, computed before the fact

§7's WR gain over market-only was +0.695 PPG with an across-fold SD of 0.819, giving a 10-cluster
MDE of roughly 0.82. Differences *between* summary statistics will be far smaller than differences
between the data arm and no data arm. **This test is therefore expected to be underpowered for small
effects, and a null must be reported as uninformative rather than as evidence of equivalence**
(§28.1). Report the realised MDE next to every p-value.

## §S4 The rebuild

Independently of the outcome, rebuild the board in **one script, one pass, from raw inputs**, with
every layer as a named column and no post-hoc adjustment applied anywhere else:

    adp → m(adp) = pi            market prior, isotonic, refit on the deep panel
    mu_hat                       chosen summary of player history (§S1 winner or incumbent)
    B = V/(V+tau2)               estimated shrinkage weight
    theta_star                   = (1-B)*mu_hat + B*pi
    + views (P,q,Omega)          BL posterior, applied exactly once
    - replacement(position)      2026 ADP-composition based
    + lambda * floor_gap         floor vs positional reference, lambda = 0.10
    = final

Outputs: `scripts/50_build_board.py` (the only board builder), `results/board_2026.csv`,
`results/sectionS_notes.md`, and REPORT.md §39 with the full derivation.

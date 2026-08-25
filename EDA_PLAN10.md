# EDA Round 10 — the base forecast, rebuilt
### Pre-registered 2026-08-25, before fitting. One change to the value layer, specified in advance.

## What round 9 found and why this is not a post-hoc fit

§W1 rejected projection-from-inputs (ninth null) but found two defects in the incumbent data arm.
Each has **independent** LOSO support, obtained under a pre-registration that was testing something
else entirely:

| correction | WR | RB |
|---|---|---|
| calibrate μ̂ (slope 0.667 / 0.605) | 3.776 → 3.606, p = .046, 8/10 folds | 4.491 → 4.171, p = .001, 9/10 |
| age curve on μ̂ | 3.776 → 3.619, p = .008, 9/10 | 4.491 → 4.341, p = .004, 9/10 |
| **both** | **3.547** | **4.077** |

The *combination* was identified by decomposition, so under the project's own rule it could not be
adopted in round 9. This document pre-registers it as a specification before any combined fit is
scored, which is the correct way to promote it.

**Honest caveat, recorded now:** the components were selected on the same folds they are now
combined on, so the combined estimate is optimistically biased. The temporal holdout
(2015–21 → 2022–24) is therefore the binding screen here, not the LOSO p-value.

## §X1 The specification

Replace the data arm's μ̂ with

    mu_star_i  =  a_f  +  b_f * mu_hat_i  +  g_f( age_i )                     (X.1)

- **μ̂** — recency-weighted mean of season means, h = 1. Unchanged; it beat seven alternatives (§S).
- **a_f, b_f** — calibration intercept and slope, fitted **within each LOSO training fold** on
  (realised PPG ~ μ̂) and applied out of sample. Never fitted on the year being predicted.
- **g_f(age)** — §H's era-3 age curve, likewise fold-fitted, entering as the log-ratio
  f(age)/f(age−1) so it corrects the *transition* rather than re-levelling the player.

Everything downstream is unchanged: μ\* enters eq. (7) in place of μ̂, B = V/(V+τ²) is still
estimated, and the market prior m(ADP) is untouched.

## §X2 Adoption rule, fixed now

Adopt μ\* iff **both**:
1. LOSO RMSE improves against μ̂ at p < 0.05, DM clustered by year, t(9 df); **and**
2. it survives the temporal holdout 2015–21 → 2022–24.

Report the realised MDE beside every p-value. Report the effect **in eq. (7)** separately from the
effect on the raw arm — §W1 found age survives into θ\* at RB (+0.355, p = .010) but only weakly at
WR (+0.194, p = .083), and the posterior is what the board actually uses.

**Pre-specified expectation, recorded so it can be falsified:** the two corrections overlap — age on
an *already calibrated* μ̂ was worth only +0.442 (WR, p = .134) and +0.752 (RB, p = .121). So the
combination should land materially short of the sum of its parts. If it does not, something is wrong.

## §X3 What is NOT in the base layer, and why

- **Situation adjustments** (team change −1.0 PPG, vacated targets +1.9, both real within-player per
  §B) are **not** applied to μ\*. §E tested exactly this and it failed LOSO (p = .44) because the
  market arm already carries the move — adjusting μ̂ double-counts. The research exists; the
  adjustment does not, and that is a finding, not an omission.
- **Availability** — rejected (§W1 L2.2); the naive multiplier is significantly worse than nothing.
- **Usage/efficiency projection** — rejected (§W1, ninth null).

## §X4 The stack above it, unchanged

    mu_star  ->  eq.(7) with m(ADP), B estimated
             ->  Black-Litterman views (player + structural delta_RB)
             ->  minus replacement (PPG rank, g >= 8)
             ->  plus lambda * floor gap
             =   board

δ_RB is retained at the owner's direction as a statement about his league's meta (11 RB vs 5 WR in
the first 16 picks of his stated order, against 8/8 at public ADP), logged dated and scoreable, and
kept out of the January-scored column.

Outputs: `scripts/75_mu_star.py`, `results/sectionX_notes.md`, `results/sectionX_loso.csv`, and the
board rebuilt through `70_build_board.py --mu-star`.

# §C notes — age-detrended data arm, LOSO arm (v) (2026-07-15)

## Spec (EDA_PLAN2.md §C, executed as pre-registered)

μ̂ᵃ_i = h=1 recency-weighted mean of age-adjusted season means
Ȳᵃ_is = Ȳ_is − [f̂(age_is) − f̂(age_i at the prediction year)]; θ*ᵃ per eq. (7) with the
IDENTICAL B weights as arm (ii) (same V, τ², n_eff, m̂ — only the likelihood mean changes).
Inside each LOSO fold Y, f̂ is refit on data ≤ Y−1 only (leak-free by construction; the
fold table below shows the per-fold fits).

**Estimator choice, stated up front (the plan's explicit allowance):** f̂ per fold is the
computationally lighter FIXED-EFFECTS version of the round-1 §5 spec — OLS of game PPR on
natural cubic spline cr(age, df=4) + season FE with player fixed effects absorbed by
within-player demeaning (FWL) — not MixedLM (which needed optimizer fallbacks even once in
round 1; 10 refits would be fragile). Only the *shape* of f is identified either way (APC),
and the detrend uses only differences f̂(a₁) − f̂(a₂), which are shape-only. Fit sample per
fold = round-1 §5(a) primary sample (all WR player-seasons, ≥8 included games, ≥3
targets/game, S0 rule) restricted to seasons ≤ Y−1; full sample is 16,566 games / 383
players / 1,262 player-seasons. Ages clamped to the fold's [1st, 99th] pct training range.

**FE vs MixedLM check (full sample):** FE peak 25.9 vs MixedLM 25.8; FE decline/yr
−0.51 / −0.87 / −1.15 at 28/30/32 vs MixedLM −0.42 / −0.71 / −0.92 — FE is somewhat
steeper (max centered-shape gap 1.34 PPG on common support, concentrated at the old edge).
Expected direction: RE shrinks player intercepts, letting part of the decline load on them.

**Fold-2015 caveat:** with data ≤ 2014 there is one training season, so within-player
demeaning leaves no age variation; that fold falls back to pooled OLS (cross-sectional
identification, curve nearly flat: −0.08/yr at 30). Flagged, not patched.

## Reproduction gate

Arms (i)/(ii) recomputed inside scripts/15 and 16: RMSE 3.5636 / 3.4631, row-level θ*
identical to `results/loso_predictions.csv` (asserted). Proceeded only after this passed.

## Headline result

| arm | RMSE | mean within-yr Spearman | DM vs (i) | DM vs (ii) |
|---|---|---|---|---|
| (ii) blind θ* | 3.4631 | .4667 | t=+2.68, p=.025 | — |
| (v) θ*ᵃ | 3.4618 | .4706 | t=+1.89, p=.091 | **t=−0.02, p=.983** |

**Arm (v) is a statistical tie with (ii).** It does what it was built to do on the target
bias, but pays for it elsewhere (below), netting to zero. It does NOT enter the 2026 board.

## The target anomaly: bias by career-start cohort, before/after (rows with prior data)

| cohort | n | μ̂ bias | μ̂ᵃ bias | θ* bias | θ*ᵃ bias |
|---|---|---|---|---|---|
| career start < 2014 | 96 | +1.270 | +1.037 | +0.615 | +0.519 |
| career start ≥ 2014 | 191 | −0.024 | +0.700 | −0.036 | +0.192 |

The verified round-1 cell (pre-2014 careers, 2017–24 folds, n=60): μ̂ bias **+1.12 → +0.76**;
θ* bias +0.61 → +0.46. By experience entering the fold year:

| exp | n | μ̂ bias | μ̂ᵃ bias | θ* bias | θ*ᵃ bias | mean adj. | per-row loss diff (v)−(ii), >0 = (v) better |
|---|---|---|---|---|---|---|---|
| 0–2 | 81 | −0.958 | +0.079 | −0.350 | −0.063 | +1.04 | −0.32 |
| 3–5 | 108 | +0.334 | +1.030 | +0.048 | +0.314 | +0.70 | −0.03 |
| 6–8 | 74 | +1.259 | +0.992 | +0.574 | +0.467 | −0.27 | +0.21 |
| 9+ | 24 | +2.742 | +1.758 | +1.365 | +0.967 | −0.98 | +0.69 |

Reading, chased to its causes:
1. **The old-career fix works where it was aimed**: exp 9+ μ̂ bias 2.74 → 1.76 and the loss
   diff is positive (+0.69/row); exp 6–8 also improves. About a third of the old bias
   remains — the FE curve's decline is still shallower than board veterans' realized fades
   (survivor-conditioning in the fit sample: a vet must still qualify at ≥3 tgt/g to
   contribute, so collapse-to-exit years are censored).
2. **Young players (exp 0–2): the detrend is right on average but noisy.** Realized change
   above μ̂ is +0.96; the spline's mean up-slope adjustment is +1.04 — bias essentially
   eliminated (−0.96 → +0.08). But per-player error variance rises (10.7 → 12.5 PPG²), so
   the loss diff is still −0.32/row: the *average* growth curve is correct, which player
   grows is not knowable from age.
3. **Mid-career (exp 3–5) is where the arm loses**: mean adjustment +0.70 at ages where
   realized level is flat — the FE up-slope near the 25.9 peak overshoots for
   already-established board players (the within-player up-slope is partly role expansion,
   which a top-30-priced player has already banked). Bias goes 0.33 → 1.03.
4. **Fold sensitivity**: yearly loss diffs (v)−(ii) are +ve in 2018/20/22/24, −1.31 in 2016
   (the 2-season-FE fold applies +2.8 PPG adjustments to the entire 2014-rookie cohort —
   Evans/Robinson/Cooks/Watkins/Landry — the single largest moves in the file). Excluding
   the two data-edge folds (2015–16): t=+1.26, p=.247 — directionally positive, not
   significant. Reported, not selected on.

## Largest individual moves (θ*ᵃ − θ*)

All top moves are the 2016/2018 folds' young cohort adjusted UP (+0.7 to +1.0 at θ* level):
Robinson, Evans, Moncrief, Cooks, Watkins, Landry, OBJ (2016). On the 2026 board
(full-sample f̂, `results/sectionC_2026.csv`): Adams −1.01, Evans −0.94, McLaurin −0.67,
A.J. Brown −0.37 vs London +0.39, Nabers +0.38, Pickens +0.31 — the direction round-1 §13.1
predicted, but LOSO says these corrections don't buy out-of-sample accuracy, so they are
NOT applied to the final board (see valuation_2026_v2.csv verdict column).

## Verdict

The +1.1 PPG old-career bias is real and the age detrend removes ~1/3 of it at the μ̂ level
(+1.12 → +0.76 in the pre-registered cell), but the same machinery injects an overshoot for
mid-career players, and B ≈ 0.65 shrinkage mutes everything toward the market — net DM vs
(ii) is exactly zero. An honest null: the age curve is the right diagnosis but a shape-only,
survivor-censored f̂ is too blunt an instrument at board level.

## Files
- `results/sectionC_partial.csv` — per-fold rows: μ̂, μ̂ᵃ, θ*, θ*ᵃ + fold inputs.
- `results/age_curve_folds.csv` — per-fold f̂ on the age grid (consumed by scripts/16).
- `results/sectionC_2026.csv` — 2026 board under full-sample f̂ (not adopted).
- Script: `scripts/15_age_detrend.py`. Scorecard integration in `scripts/16`.

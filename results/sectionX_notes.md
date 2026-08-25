# §X — μ* : calibration + age inside the data arm

Pre-registration: `EDA_PLAN10.md` (2026-08-25). This file's PART 1 was written **before any
fit of the combined specification** and fixes every operational choice the plan left open.
PART 2 is the result, appended after.

---

## PART 1 — operational definitions, declared before fitting

### X.0 Panel, folds, loss

Identical to §W1 tier A, and nothing about it is re-opened:

- rows: `data/derived/w1_features_{WR,RB}.csv` joined to `results/sectionS_predictions.csv`
  on (`gsis_id`, `year`, `pos`), filtered `in_fit & n_eff > 0 & year ∈ [2015, 2024]`.
  n = 568 WR / 489 RB.
- target: `ppg` = realised PPR per game *played* in year *Y*.
- folds: leave-one-season-out, ten folds.
- loss: squared error. Diebold–Mariano on the ten yearly mean loss differentials, t(9),
  two-sided. MDE at 80% power, α = .05, printed beside every p.
- `B`, `m_hat` are taken from `sectionS_predictions.csv` unchanged — the blend weight and the
  market prior are **not** re-estimated in this section.

### X.1 The specification (from the plan, made operational)

    mu_star_i = a_f + b_f * mu_hat_i + c_f * z_i ,   z_i = log[ f_f(age_i) / f_f(age_i − 1) ]

- (a_f, b_f, c_f) are the OLS coefficients of `ppg ~ 1 + mu_hat + z` fitted on the **training
  rows of fold f only**, per position, and applied to the held-out year. Three parameters.
- `f_f` is §H's era-3 (2017–2025) *relative* age curve, **refitted inside each training fold**
  by re-running §H's own estimator on `data/derived/age_panel_long_repaired.csv` with the
  held-out season deleted: qualification `games ≥ 8 & touches ≥ 40`, outcome
  `r = ppg / mean(ppg | position, season)`, natural cubic spline (patsy `cr`) with interior
  knots at that position's pooled age quintiles and boundary knots at its pooled min/max,
  player fixed effects absorbed by within-player demeaning, and §H's anchoring convention
  (each era's curve shifted so its mean over that era's observations equals the mean of `r`
  over the same observations). Nothing about §H's estimator is changed; only the sample is
  reduced to the fold.
- age enters as the **log-ratio of adjacent curve values**, so the term corrects the
  *transition* from age−1 to age. It is not the curve level, which would re-level the player.
- everything downstream is unchanged: μ* replaces μ̂ in eq. (7), `B = V/(V+τ²)` still estimated,
  `m(ADP)` untouched.

**Declared now, because it is a real fork:** (X.1) is *additive* in a fitted coefficient on the
log-ratio, which is what the plan writes. §W1's age arm was *multiplicative with unit
coefficient*, `μ̂ × r`. The additive form is the **headline**; the multiplicative composition
`(a_f + b_f μ̂) × r` is reported as a declared sensitivity under the label `mu_cal_x_age`, and
it is also the arm whose §W1 numbers (WR 3.547 / RB 4.077) the harness must reproduce.

### X.2 Arms scored (fixed list, no additions after seeing results)

| arm | definition | role |
|---|---|---|
| `mu_hat` | incumbent (43.1) | baseline for every test |
| `mu_cal` | a_f + b_f μ̂ | §W1 component; reproduction target 3.606 / 4.171 |
| `mu_age_pub` | μ̂ × r, published §H curve | §W1 component; reproduction target 3.619 / 4.341 |
| `mu_cal_x_age_pub` | (a_f + b_f μ̂) × r, published curve | §W1 "both"; target 3.547 / 4.077 |
| **`mu_star`** | a_f + b_f μ̂ + c_f z, fold-fitted curve | **the specification** |
| `mu_star_pub` | same with the published curve | sensitivity: does fold-fitting the curve matter |
| `mu_cal_x_age` | (a_f + b_f μ̂) × r, fold-fitted curve | sensitivity: the multiplicative ordering |
| `mu_age` | μ̂ × r, fold-fitted curve | component, fold-clean |

### X.3 Adoption rule (§X2 of the plan, not negotiable)

Adopt μ* iff **both**:
1. LOSO RMSE improves on μ̂ at p < .05 (DM, t(9)); **and**
2. it survives the temporal holdout **2015–21 → 2022–24**.

The holdout is a genuine one and differs from §W1's: coefficients (a, b, c) **and** the age
curve are fitted on 2015–2021 rows only and applied to 2022–2024. §W1's `sectionW1_holdout.csv`
merely subset LOSO predictions to 2022–24, which still lets each held-out year borrow
coefficients from later years. That is recorded here as a difference in construction, not as a
criticism of the number it produced.

**The holdout governs.** The two components were selected on the same ten folds they are now
combined on, so the combined LOSO number is optimistically biased and its p-value is not a
clean α. This is stated before the fit, as the plan requires.

The effect **inside eq. (7)** — θ* = (1−B)μ* + B·m(ADP) against θ̂ = (1−B)μ̂ + B·m(ADP) — is
reported separately from the effect on the raw arm, with its own DM test and MDE. A gain on the
raw arm that does not survive the blend is not an improvement to the product.

### X.4 Pre-specified expectation (falsifiable)

The two corrections **overlap**: §W1 found age on an already-calibrated μ̂ worth only +0.442
(WR, p = .134) and +0.752 (RB, p = .121), against +1.176 / +1.311 on raw μ̂. So

    gain(mu_star)  <  gain(mu_cal) + gain(mu_age)

by a material margin. The additivity gap is reported explicitly. If the combination were to
*exceed* the sum of its parts, the specification would be treated as suspect and chased before
being believed.

### X.5 Declared diagnostics, run whether or not they flatter the result

- residuals of the fold OLS against μ̂ and against age (curvature the affine form misses);
- absolute-error (robust) loss as a second DM, since PPG is right-skewed;
- the fold-to-fold path of (a, b, c);
- decomposition of the LOSO gain by |μ̂ − mean| quartile (calibration should pay in the tails)
  and by age tercile (age should pay in the old tercile);
- the anchor caveat: f is identified only up to an additive constant under player FE, so
  f(a)/f(a−1) is *not* anchor-invariant. §H's anchoring convention is inherited unchanged and
  applied identically in every fold; it is part of the specification, not a free parameter.

### X.6 Board

`scripts/70_build_board.py --mu-star` adds μ* as a named, ablatable layer. Coefficients for the
2026 board are fitted **once, on the whole 2015–2024 panel** (there is no held-out 2026), and
the age curve is §H's era-3 curve fitted on the whole panel — i.e. the same estimator every
fold used, with the fold restriction removed because there is no fold. Every existing assertion
is retained: views applied exactly once, the Ω→0 structural check, and the incumbent
reproduction with all new layers off (which must still hold, since `--mu-star` defaults off).

---

## PART 2 — results (appended 2026-08-25, after the fit)

### Harness validation, before reading anything new

`scripts/75_mu_star.py` re-implements §H's era-3 estimator and reproduces
`results/age_curve_era.csv` to **3.3e-16 (WR) / 4.4e-16 (RB)** on the full sample, and
reproduces every §W1 component number on the shared panel:

| | WR | RB |
|---|---|---|
| μ̂ | 3.7760 (target 3.7760) | 4.4909 (4.4909) |
| μ̂_cal | 3.6059 (3.6059) | 4.1708 (4.1708) |
| μ̂ × r | 3.6186 (3.6186) | 4.3415 (4.3415) |
| (μ̂_cal) × r | 3.5466 (3.5466) | 4.0774 (4.0774) |

Any difference below is μ* and nothing else.

### LOSO 2015–2024

| | n | μ̂ | μ* | gain | p | MDE₈₀ | obs/MDE | folds |
|---|---|---|---|---|---|---|---|---|
| WR raw arm | 568 | 3.7760 | **3.5483** | +1.711 | **.0118** | 1.710 | 1.00 | 9/10 |
| RB raw arm | 489 | 4.4909 | **4.0917** | +3.389 | **.0004** | 1.929 | 1.76 | 9/10 |
| WR eq. (7) | 568 | 3.4036 | 3.3896 | +0.101 | .624 | 0.626 | 0.16 | 5/10 |
| RB eq. (7) | 489 | 3.7842 | **3.7199** | +0.475 | **.0066** | 0.426 | 1.12 | 9/10 |

Absolute-error loss agrees: WR +0.150 (p = .029), RB +0.267 (p = .006). Mean within-year
Spearman rises .5420 → .5696 (WR) and .4947 → .5035 (RB).

### Temporal holdout — coefficients AND age curve fitted on 2015–21, applied to 2022–24

| | n_fit | n | μ̂ | μ* | Δ | years improved |
|---|---|---|---|---|---|---|
| WR raw arm | 393 | 175 | 3.6912 | **3.5325** | −0.159 | 2/3 |
| RB raw arm | 347 | 142 | 4.1611 | **3.7866** | −0.375 | 3/3 |
| WR eq. (7) | 393 | 175 | 3.3644 | 3.3599 | −0.005 | — |
| RB eq. (7) | 347 | 142 | 3.6441 | **3.5956** | −0.049 | — |

Improves at both positions, on the raw arm and inside eq. (7). Three years is three
observations; the holdout is a **direction** test, not a powered one, and it is reported as such.

### §X4 — the pre-specified overlap held

| | μ̂_cal | μ̂×r | naive sum | μ* | overlap |
|---|---|---|---|---|---|
| WR | +1.287 | +1.138 | +2.425 | +1.711 | **+0.713 (29% of the sum lost)** |
| RB | +2.761 | +1.356 | +4.117 | +3.389 | **+0.728 (18%)** |

Predicted from §W1's "age on an already-calibrated μ̂" numbers: 1.287 + 0.442 = 1.729 against
1.711 observed (WR); 2.761 + 0.752 = 3.513 against 3.389 (RB). The falsifiable expectation was
met to within 0.02 / 0.12 PPG². The mechanism is visible in the slope: adding the age term
raises b from 0.667 → 0.728 (WR) and 0.605 → 0.630 (RB), because older players carry higher
μ̂, so part of what the calibration was doing was age in disguise.

### Verdict: ADOPT

Both §X2 conditions are met at both positions. **Recorded honestly:** the LOSO p-values are not
clean α — the components were chosen on these folds — which is why the holdout was declared the
binding screen in advance, and it passes on its own.

**But the eq. (7) effect is where the product lives, and it splits by position.** At RB the
posterior improves +0.475 (p = .0066, obs/MDE 1.12). At WR it does not: +0.101 against its own
MDE of 0.626, 5/10 folds — indistinguishable from zero. Section 49.5 of REPORT.md decomposes
why, and §49.6 records the consequence for the board: §P4's arm rule sends μ* to WR ADP-rank ≤ 30
only, which is precisely the stratum where the posterior gain is undetectable, while RB — where
it is detectable — takes `pi_market` and never sees μ* at all.

### Sensitivities, all declared in PART 1

| variant | WR | RB |
|---|---|---|
| μ* (headline, fold-fitted curve) | 3.5483 | 4.0917 |
| μ* with §H's published curve | 3.5416 | 4.0782 |
| (a+bμ̂)×r, fold-fitted (multiplicative ordering) | 3.5532 | 4.0818 |
| (a+bμ̂)×r, published curve (§W1's "both") | 3.5466 | 4.0774 |

Fold-fitting the age curve costs 0.007 (WR) / 0.014 (RB) of RMSE. That is the price of removing
a leak §W1 carried, and it is paid, not argued away. The additive and multiplicative
compositions are within 0.005 / 0.010 of each other — the fork declared in PART 1 does not
matter numerically, which is the best outcome for it.

### Files

`results/sectionX_loso.csv`, `sectionX_holdout.csv`, `sectionX_diagnostics.csv`,
`sectionX_coefs.csv`, `sectionX_predictions.csv`, `mu_star_coefs_2026.json`,
`sectionX_board_movers.csv`; board `results/board_2026_v5_mustar.csv`
(`scripts/70_build_board.py --mu-star`).

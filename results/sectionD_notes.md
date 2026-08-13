# §D notes — usage-based projection arm, LOSO arms (vi) and (v+vi) (2026-07-15)

## Spec (EDA_PLAN2.md §D, executed as pre-registered)

Within each LOSO training fold (data ≤ Y−1): ridge regression of PPG_{s+1} on
{target_share_s, WOPR_s, aDOT_s, team pass attempts_ts (z), age} over all WR
player-seasons with ≥3 targets/game (features from season s, outcome season s+1, BOTH
≤ Y−1; S0 inclusion rule; definitions identical to round-1 §5 conventions; WOPR =
1.5·TS + 0.7·AYshare; team attempts/game z-scored across teams within season; age at
Sept 1 of the *predicted* season). λ by 5-fold CV inside the fold, grouped by player
(same player never straddles a CV split — implementation choice, documented). Data arm
ŷ_i = ridge prediction from the board player's season-(Y−1) stats; V = training-fold
residual variance; posterior per eq. (7) with the fold's m̂/τ²(tier). Board players with
no qualifying season-(Y−1) stats fall back to arm (ii)'s posterior (9 rows across
2016–24, mostly rookies), and the entire 2015 fold falls back — the weekly window starts
2014, so no s→s+1 pair fits inside ≤2014 (same data edge round 1 documented).

**Arm (v+vi)** — age-detrend applied to the usage arm, implementation fixed before
fitting: the linear age feature is replaced by §C's fold spline — ridge fit on the
age-detrended outcome T = PPG_{s+1} − f̂(age_{s+1}) with {TS, WOPR, aDOT, att_z}, and the
prediction re-adds f̂ at the player's fold-year age; fallbacks go to arm (v).

Training sample: 1,356 pairs / 423 players (full window); growing from 112 (fold 2016)
to 1,105 (fold 2024). Attrition note: 77.1% of qualifying seasons have an s+1 with
included games — the YoY fit conditions on survival, like any adjacent-season regression.

## Ridge coefficients and fold stability (standardized features, arm (vi))

| fold | n | λ | TS | WOPR | aDOT | att_z | age | V_train | V_cv |
|---|---|---|---|---|---|---|---|---|---|
| 2016 | 112 | 0.01 | +7.87 | −4.75 | +1.12 | +1.08 | −1.01 | 14.8 | 16.1 |
| 2017 | 229 | 25.1 | +1.68 | +1.20 | −0.13 | +0.57 | −0.52 | 13.9 | 14.8 |
| 2018 | 350 | 39.8 | +1.64 | +1.17 | −0.26 | +0.48 | −0.45 | 13.4 | 14.1 |
| 2019 | 469 | 0.63 | +3.44 | −0.39 | −0.01 | +0.60 | −0.75 | 13.7 | 14.3 |
| 2020 | 595 | 0.63 | +2.94 | +0.08 | +0.11 | +0.67 | −0.57 | 15.1 | 15.6 |
| 2021 | 719 | 25.1 | +1.90 | +1.16 | −0.16 | +0.48 | −0.57 | 15.0 | 15.4 |
| 2022 | 846 | 2.51 | +3.24 | −0.22 | +0.04 | +0.57 | −0.55 | 14.9 | 15.2 |
| 2023 | 982 | 2.51 | +3.35 | −0.30 | +0.07 | +0.59 | −0.60 | 14.4 | 14.7 |
| 2024 | 1105 | 0.63 | +3.67 | −0.57 | +0.17 | +0.65 | −0.57 | 14.5 | 14.8 |

Stability, chased: individual TS/WOPR coefficients wobble (and λ jumps 0.01–39.8) because
WOPR is 1.5·TS + 0.7·AYshare — near-collinear, the CV objective is flat in the split. The
*composite* TS+WOPR loading is 2.82–3.11 in every fold including 2016, and att_z
(+0.48–0.67), age (−0.45–0.75, past-peak sample) and V are stable from n≈230 up.
Predictions, not coefficients, are the stable object. V_cv exceeds V_train by only
0.3–1.3 PPG² (mild optimism; using V_train as the plan specifies slightly overweights the
usage arm, B_r ≈ 0.56 vs arm (ii)'s mean per-player B = 0.64). Arm (v+vi) coefficients
behave identically with the age column absorbed by the spline.

## Headline result (full scorecard in results/loso_scorecard2.csv)

| arm | RMSE | mean Spearman | DM vs (i) | DM vs (ii) |
|---|---|---|---|---|
| (i) ADP-only | 3.5636 | .4610 | — | t=−2.68, p=.025 |
| (ii) blind θ* | 3.4631 | .4667 | t=+2.68, p=.025 | — |
| (v) θ*ᵃ | 3.4618 | .4706 | t=+1.89, p=.091 | t=−0.02, p=.983 |
| (vi) usage posterior | 3.5953 | .4034 | t=−0.42, p=.685 | t=−1.50, p=.168 |
| (v+vi) usage + detrend | 3.6082 | .3938 | t=−0.66, p=.525 | t=−1.75, p=.115 |

**The usage arm LOSES to the blind posterior** (and even to raw ADP on RMSE), n.s. but
consistently (negative loss diff in 6 of 9 live folds; excl. 2015–16: t=−1.71, p=.13).
(v+vi) is slightly worse still. The §2 ceiling-gap hypothesis (0.41 → 0.58, "usage should
beat raw PPG history") FAILS at board level. Reported plainly.

## Why it fails, chased (non-fallback rows, n=252)

| predictor | RMSE | bias | corr w/ realized |
|---|---|---|---|
| raw ŷ (usage ridge) | 3.944 | **−1.476** | .308 |
| μ̂ (raw PPG history) | 3.484 | +0.408 | .467 |
| m̂(ADP) | 3.401 | +0.071 | .419 |

1. **Selection-on-the-outcome bias**: the ridge is a population WR fit; board players are
   top-30 *by price*, systematically above the usage-implied conditional mean (they
   convert usage to points efficiently — that efficiency is exactly what §4 gates out as
   unreliable, so the model cannot carry it, and shrinks them to the population line).
   Raw ŷ under-predicts by −1.55/−1.13/−1.75 PPG in low/mid/top market terciles. The
   posterior inherits ~44% of it (θ_usage bias −0.60).
2. **Less signal, not more**: on the board sample corr(ŷ, realized) = .31 < corr(μ̂,
   realized) = .47 — one season of usage carries *less* rank information about a top-30
   player's next season than his h=1 multi-season PPG history. Restriction of range does
   part of this (the board compresses the usage spread the ridge relies on).
3. **Where it does help**: the loss diff vs (ii) is positive in the TOP market tercile
   (+0.28/row) — the usage arm correctly called several aging-star collapses (Julio 2021
   pred 12.3 vs realized 9.8; Jordy Nelson 2017 16.2 vs 9.8; Kupp 2023 17.4 vs 14.8) —
   but it also butchered efficiency outliers (Antonio Brown 2017: θ 17.2 vs realized
   22.2) and loses −2.1/−1.4 per row in the low/mid terciles. Net negative.
4. **V is flat across players**: the arm has no n_eff — a one-year wonder and a 10-year
   stalwart get the same likelihood precision, discarding exactly the information that
   makes arm (ii) work.

## 2026 board v2 (results/valuation_2026_v2.csv)

Pre-specified rule: adopt the best arm with DM-vs-(ii) support (t>0, p<.05). **No arm
qualifies** — (v) ties, (vi)/(v+vi) are worse — so the file restates the round-1 final
values verbatim with `round2_verdict = "unchanged: neither (v) nor (vi) beat the blind
posterior in LOSO (DM vs (ii) n.s.)"`. That is the round-2 result: the blind
empirical-Bayes posterior remains the best validated valuation.

## Deviations from plan (all stated above, none outcome-driven)
- FE spline instead of MixedLM for per-fold f̂ (plan-sanctioned; §C notes).
- GroupKFold(5) by player for λ (plan said "CV within fold" without grouping detail).
- Fold 2015 usage arm = full fallback (no trainable pairs at the 2014 data edge);
  fold-2015 f̂ = pooled OLS (§C notes).
- Board usage rows require a qualifying (≥3 tgt/g) season Y−1; 9 fallback rows 2016–24.
- Intermediate files beyond the plan's list: sectionC_partial.csv, age_curve_folds.csv,
  sectionC_2026.csv, usage_ridge_coefs.csv (inputs/audit trails, not results).

## Files
- `results/loso_scorecard2.csv`, `results/loso_predictions2.csv` (round-1 files untouched)
- `results/valuation_2026_v2.csv`, `results/usage_ridge_coefs.csv`
- Scripts: `scripts/15_age_detrend.py`, `scripts/16_usage_projection.py`

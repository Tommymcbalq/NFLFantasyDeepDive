# §I3 — Vegas team-environment edge test

Run 2026-08-09 11:45 | script `scripts/24_sectionI3_vegas_edge.py` (pre-registration in its docstring)

## 1. De-vig calibration
- s (SD of realized minus posted, win-RATE scale) = 0.16200 = 2.754 wins per 17 games, n = 320
- integer (push-possible) lines: 111 / 320 -> continuity-corrected two-way solve; half-win lines closed form
- mean two-way overround, Covers 2015-24: 1.0408
- de-vig shift |wt17 - line17|: mean 0.305, max 1.070 wins-per-17
- worked example (2025, top 3 de-vigged): Buffalo Bills line 12.50 -> 12.14; Baltimore Ravens line 11.50 -> 11.62; Philadelphia Eagles line 11.50 -> 11.28
- realized-wins integrity check (games.csv vs Covers `actual_wins`): 320 pairs, exact 302, MAD 0.0281 wins (differences are ties, scored 0.5 here vs 0 there)

## 2. Panel construction
- board rows 300; in_fit 291
- modal-team join: 297/300 (99.0%); in_fit 291/291
- unjoined rows (no REG appearance that season):
    2016 Josh Gordon games=0 in_fit=False
    2019 A.J. Green games=0 in_fit=False
    2021 Michael Thomas games=0 in_fit=False
- modal vs first-team disagreement (in-season movers): 9 rows
- regression n = 291 (in_fit 291, dropped for missing team/feature: 0)

Feature distributions on the fit sample (wins per 17):
```
            wt  surprise  d_posted
count  291.000   291.000   291.000
mean     9.296    -0.076     0.282
std      1.569     2.108     1.619
min      4.650    -5.211    -4.330
25%      8.189    -1.743    -0.709
50%      9.433    -0.070     0.206
75%     10.350     1.439     1.076
max     13.131     6.245     5.085
```
- pairwise correlations: wt~surprise -0.246, wt~d_posted 0.375, surprise~d_posted 0.088

## 3. Pre-specified regression

`R = b0 + b1*wt + b2*surprise + b3*d_posted + u`, OLS, SEs clustered by season (10 clusters, t with 9 df).
```
            beta  se_cluster  t_cluster  p_raw_cluster  se_hc3  p_raw_hc3      vif
const    -0.8067      1.6754    -0.4815         0.6417  1.2895     0.5316  44.9424
wt        0.0884      0.1769     0.4998         0.6292  0.1409     0.5303   1.2801
surprise  0.0458      0.0679     0.6746         0.5169  0.0939     0.6256   1.1088
d_posted -0.0415      0.0809    -0.5137         0.6198  0.1209     0.7312   1.2119
```
R2 = 0.00166; adj R2 = -0.00877
Joint Wald (all 3 = 0), clustered: F = 0.327, p_raw = 0.8063
Local BH q=0.10 over these 3 terms only (the binding correction is the joint round-4 family {H5, I3} at consolidation — raw p above are what to carry):
```
           p_raw  bh_thresh
surprise  0.5169     0.0333
d_posted  0.6198     0.0667
wt        0.6292     0.1000
```
Local FDR survivors: NONE

### Residual diagnostics
- residual skew -0.008, kurtosis 3.051, Jarque-Bera p = 9.83e-01 (R inherits the right skew of PPG; hence the Huber sensitivity below)
- Breusch-Pagan on the 3 terms: LM = 4.162, p = 0.2445

## 4. Pre-specified sensitivities
- (a) first-REG-team attach          n= 291  wt: +0.1110 (se 0.1820, p_raw 0.557)  surprise: +0.0497 (se 0.0677, p_raw 0.482)  d_posted: -0.0515 (se 0.0839, p_raw 0.554)
- (b) drop 2015 (cross-source prev)  n= 261  wt: +0.2180 (se 0.1340, p_raw 0.142)  surprise: +0.0271 (se 0.0737, p_raw 0.723)  d_posted: -0.0000 (se 0.0752, p_raw 1.000)
- (c) Huber robust (HuberT M-est)    n= 291  wt: +0.1322 (se 0.1452, p_raw 0.362)  surprise: +0.0450 (se 0.1005, p_raw 0.655)  d_posted: -0.0495 (se 0.1369, p_raw 0.717)
- (d) single term: wt                n= 291  wt: +0.0572 (se 0.1911, p_raw 0.771)
- (d) single term: surprise          n= 291  surprise: +0.0269 (se 0.0769, p_raw 0.735)
- (d) single term: d_posted          n= 291  d_posted: -0.0042 (se 0.1142, p_raw 0.971)

## 5. Temporal holdout (binding): fit 2015-2022, evaluate 2023-2024
- train n = 231, eval n = 60
- zero-prediction (market-efficiency) holdout MSE = 9.3356
- joint: all 3               MSE   9.4662  improves: False  mean d_sq -0.1306 (row t -0.94; 2023 -0.157, 2024 -0.104)
- single: wt                 MSE   9.4486  improves: False  mean d_sq -0.1130 (row t -1.13; 2023 -0.017, 2024 -0.209)
- single: surprise           MSE   9.4612  improves: False  mean d_sq -0.1256 (row t -1.22; 2023 -0.112, 2024 -0.139)
- single: d_posted           MSE   9.5095  improves: False  mean d_sq -0.1739 (row t -1.51; 2023 -0.171, 2024 -0.177)
- coefficient stability train (2015-22) vs eval (2023-24), reported whether or not anything survives:
    wt         train +0.0574  eval +0.2303
    surprise   train +0.0378  eval +0.0608
    d_posted   train -0.0762  eval +0.0557

**Terms surviving BOTH screens (local FDR + holdout): NONE**

## 5b. Why the null — power bound and the pricing channel

*Post-hoc and descriptive. Nothing here is a hypothesis test, nothing enters the FDR family, and no specification above was altered on the strength of it.*

### (i) Minimum detectable effect (the bound the null actually buys)
- wt        SE 0.1769 PPG per win-per-17 -> MDE at 80% power / 5% two-sided = 0.556 PPG per win; per 1 SD of the feature (1.57 wins) = 0.873 PPG/game. 95% CI on beta: [-0.312, +0.489]
- surprise  SE 0.0679 PPG per win-per-17 -> MDE at 80% power / 5% two-sided = 0.214 PPG per win; per 1 SD of the feature (2.11 wins) = 0.451 PPG/game. 95% CI on beta: [-0.108, +0.200]
- d_posted  SE 0.0809 PPG per win-per-17 -> MDE at 80% power / 5% two-sided = 0.254 PPG per win; per 1 SD of the feature (1.62 wins) = 0.412 PPG/game. 95% CI on beta: [-0.224, +0.141]
- for scale: SD(R) on the fit sample = 3.323 PPG. So the test rules out any win-total channel worth more than roughly 26% of a residual SD per SD of team quality. It does NOT rule out a small one; 10 season-clusters is the resolution.

### (ii) Leave-one-season-out coefficient trace
Motivated by sensitivity (b): dropping 2015 moved `wt` from +0.088 to +0.218 and cut its SE. Is 2015 special, or is any single cluster that influential?
```
 dropped_year  beta_wt   p_wt  beta_surprise  beta_d_posted  mean_R_dropped  within_yr_corr_wt_R
         2015   0.2180 0.1424         0.0271        -0.0000          0.4345              -0.4409
         2016   0.0409 0.8326         0.0307        -0.0080         -1.1814               0.2226
         2017   0.1185 0.5589         0.0291        -0.0466         -1.4106              -0.0776
         2018   0.0447 0.8215         0.0358        -0.0263          1.0351               0.1791
         2019   0.1613 0.3975         0.0597        -0.0796          0.1250              -0.2510
         2020   0.0930 0.6609         0.0964        -0.0236          1.2869               0.1735
         2021   0.0216 0.9096         0.0470        -0.0661          0.4488               0.3183
         2022   0.0392 0.8479         0.0425        -0.0525          0.1094               0.3253
         2023   0.0559 0.7746         0.0695        -0.0816         -0.1896               0.2768
         2024   0.0910 0.6501         0.0162        -0.0323         -0.7534              -0.0466
```
- `wt` slope ranges +0.022 to +0.218 across the ten drops (range 0.196, vs a full-sample SE of 0.177). 2015 is the extreme, not a category apart: the estimate is one-cluster-fragile in BOTH directions, which is the honest reading of a 10-cluster design, not evidence of a 2015 data problem.
- within-season corr(wt, R) by year: min -0.441, max +0.325, mean +0.068 — sign flips across seasons.

### (iii) The pricing channel: is the win total absent from production, or already in the price?
R = PPG - m_iso(ADP) by construction, so cov(wt,R) = cov(wt,PPG) - cov(wt,m_iso). A null on R has two very different explanations, and they are separable.
- realized PPG                 on wt: beta +0.2511 (cluster se 0.2069, p_raw 0.2559)
- m_iso (ADP-implied value)    on wt: beta +0.1939 (cluster se 0.0578, p_raw 0.0085)
- R (= PPG - m_iso)            on wt: beta +0.0572 (cluster se 0.1911, p_raw 0.7715)
- decomposition: a one-win-per-17 better team environment is worth +0.251 PPG in realized production, and the ADP market already charges +0.194 PPG for it — i.e. the market prices 77% of the realized effect. The residual +0.057 is the (insignificant) leftover. **This is a priced channel, not an absent one** — the mechanism the pre-registration predicted.
- caveat on that reading: `wt` is also correlated with ADP by construction of the board (good offences supply more top-30 WRs), so the m_iso regression is partly a compositional statement about who makes the board, not purely a per-player pricing elasticity. Stated, not resolved.

## 6. 2026 inputs — recorded, never fitted on
- 2026 win totals: VegasInsider/DraftKings board (2026-08-08). Only an OVER price is stored, so a paired de-vig is impossible; a 1.045 two-way overround is ASSUMED. This is a different source from the Covers historical series (Covers backfills only retrospectively) — the two are NOT one continuous series and nothing was fitted on 2026.
- prior-season inputs for 2026 use realized 2025 wins and the Covers 2025 posted line (same historical source as the fitted series).
- NO term survived, so no 2026 board adjustment is produced. The 2026 team features are emitted for the §J views layer only, unfitted.

## 7. Verdict
**Null, as pre-registered.** No pre-specified sportsbook team-environment term predicts the ADP market's errors on this panel. No context arm is built; nothing enters the LOSO harness. The de-vigged win total, the surprise relative to last season's realized wins, and the year-over-year change in the posted line are all already reflected in where drafters place receivers. Raw p-values above are carried to the joint round-4 FDR family {H5, I3}.

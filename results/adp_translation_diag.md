# L3.2 — source-translation map, ESPN -> FFC-equivalent*Constructed per §Q; §W2 of EDA_PLAN9.  Fitted 2026-08-24.*
## 1. Provenance of the overlap seasons
A rank->rank map presumes both series are preseason prices.  Test: regress the
season's realised PPR total on log(source ADP) and log(FFC ADP) jointly.  Two
genuine preseason prices of the same market are near-collinear and neither
dominates.  A series carrying hindsight drives the FFC coefficient to zero.

 season   n  rho_espn_ffc  rho_espn_outcome  rho_ffc_outcome   b_espn  p_espn   b_ffc  p_ffc
   2023 134        0.6786            0.6264           0.4023 -53.9467     0.0  1.4188 0.8811
   2024 161        0.7801            0.5782           0.4262 -80.0600     0.0 16.6541 0.1289

**Verdict: seasons [2023, 2024] are REJECTED as contaminated.** ESPN's stored history for those seasons predicts the realised season better than FFC's genuine preseason market does, and conditional on it the genuine preseason price adds nothing (p above).  That is not what one preseason board looks like against another; it is what a series refreshed with in/post-season draft activity looks like.  Fitting a translation map on them, or blending them into a 2026 price, would put future information into a preseason feature.

## 2. What is left to fit on
Overlap seasons [2023, 2024, 2026]; usable after (1): **[2026]**.  §Q budgeted ~380 matched player-seasons across three seasons; the honest count is one season.

Matched on the board's own FFC pull (`data/adp/adp_ppr_2026_all_20260809.csv`): **177** players (193 ESPN, 256 FFC).

## 3. Fit quality — and the benchmark that matters

                        map  rmse_log  mad_slots  spearman
identity (use ESPN ADP raw)    0.2708    17.5770       1.0
 isotonic, global (§Q spec)    0.2084    12.7521       1.0
           isotonic, LOO-CV    0.2544        NaN       1.0

Spearman(ESPN, FFC) on the matched pool = **0.925**, Pearson on logs = 0.967.

## 4. Declared sensitivity: position-stratified map
§35 predicted *in advance* that the ESPN-FFC disagreement is positional (ESPN's 10-team/1QB/1TE defaults price TE and QB 30-50 slots earlier).  A single monotone rank->rank map cannot represent that: monotone maps preserve order, so any player ESPN ranks ahead of another stays ahead after translation.  The residual by position is therefore the part of the disagreement the specified construction structurally cannot carry.

position  n  mean_res_log  mean_res_slots  med_res_slots
      PK 12         0.203          25.429         23.123
      QB 27         0.119           9.629          9.894
      RB 50        -0.077          -5.105         -2.290
      TE 23         0.227          19.921         18.794
      WR 65        -0.108          -7.537         -7.499
  PK: stratified in-sample RMSE(log) 0.0446 vs global 0.2638 on n=12
  QB: stratified in-sample RMSE(log) 0.0944 vs global 0.2035 on n=27
  RB: stratified in-sample RMSE(log) 0.1117 vs global 0.1717 on n=50
  TE: stratified in-sample RMSE(log) 0.0343 vs global 0.2760 on n=23
  WR: stratified in-sample RMSE(log) 0.1132 vs global 0.1964 on n=65

## 5. Sensitivity: which FFC window the map is fitted against
Refitting against the 08-24 FFC pull instead of the board's 08-09 pull moves the translated price by mean |Δ| = 1.68 slots on 179 players.  The ESPN pull is dated 08-13 and the board's FFC window is 08-01→08-08; the five-day gap is real market movement and is part of the residual above, not fit error.

## 6. Coverage — the one thing translation buys unconditionally
16 players carry an ESPN price and no FFC price; translation assigns them an FFC-equivalent slot without touching the fitted curve.

## 7. Verdict
The map is **built and emitted, not adopted into the headline price.** Three reasons, all decided by the construction rather than by the answer it gives:

1. **It cannot be validated.** §Q's whole case for the construction was that it is testable out of sample. The only two overlap seasons with realised outcomes fail the provenance test in §1, so there is no season on which translated-ESPN can be scored against FFC-alone. A 2026 fit evaluated on 2026 is in-sample by construction.
2. **On the pool where it is identified it is nearly the identity.** The two boards agree at Spearman .92 and the isotonic map's residual is barely below the raw-ESPN residual; the translation is a rank-scale calibration, not a re-pricing.
3. **The informative part of ESPN is exactly the part a monotone map deletes.** §35's finding is positional composition (TE/QB earlier under 1TE/1QB 10-team defaults). Order preservation means the translated price still carries ESPN's positional ordering into an FFC-calibrated curve — importing the scarcity assumption of a different roster format while claiming to have removed the source difference. That is worse than not translating.

The board therefore carries `adp_espn`, `adp_ffc_equiv_espn` and `pi_market_espn_equiv` as named columns and `--price consensus` as a switch, and the layer-ablation table reports what adopting it would do. It is off by default.

# §K — Strength-of-schedule edge test

Run 2026-08-09 12:50 | script `scripts/25_sectionK_schedule_edge.py` (pre-registration in its docstring)

Family: NEW and separate from the closed {§H5, §I3} family. 16 tests, declared in the script docstring before the first fit. BH q=0.10 within §K only.

## 1. Feature panel and the franchise-abbreviation trap, re-verified in the join
- `data/schedule/sos_history_2015_2026.csv`: 384 rows x 19 cols, missing cells 0, seasons 2015-2026, teams/season [32]
- the panel keys teams by ERA-CORRECT abbreviation: 2015 contains ['OAK', 'SD', 'STL']; 2020 contains ['LA', 'LAC', 'LV']. nflverse weekly files normalise all seasons to the CURRENT abbreviation. Both sides are mapped to a franchise code before joining.
- assertion passed: 32 distinct franchise codes in every season after mapping (a collision, e.g. STL and LA both present in one season, would fail here).

Mean within-season SD of each measure, 2015-2024 (this is the dispersion the §K1 power argument is built on; §K0 reported 0.245 / 0.988 for the two vegas windows and 0.977 / 2.211 for WR-FPA — reproduced here):
```
sos_vegas                0.2446
sos_vegas_w1_14          0.3023
sos_vegas_w15_17         0.9878
sos_prior_wpct           0.0337
sos_prior_wpct_w1_14     0.0388
sos_prior_wpct_w15_17    0.1075
sos_wr_fpa               0.9765
sos_wr_fpa_w15_17        2.2110
sos_rb_fpa               0.6569
sos_rb_fpa_w15_17        1.7874
```
- note: the positional measures exist in full-season and weeks-15-17 windows only in the §K0 panel; no weeks-1-14 positional column was built, so the family is 8 per panel rather than 9. No new feature is constructed for this test.

## 2. Panels and team attach
- **WR** (`results/market_prior.csv`): 300 board rows, in_fit 291, SD(R) = 3.323
    modal-team join 297/300 (99.0%); on in_fit rows 291/291 (100.0%)
    unjoined in_fit rows: none
    regression n = 291; feature-side missing after join: 0 cells (the 128-opponent-game abbreviation bug would show up here as missing rows)
- **RB** (`results/market_prior_rb.csv`): 300 board rows, in_fit 286, SD(R) = 3.624
    modal-team join 297/300 (99.0%); on in_fit rows 286/286 (100.0%)
    unjoined in_fit rows: none
    regression n = 286; feature-side missing after join: 0 cells (the 128-opponent-game abbreviation bug would show up here as missing rows)

- direct check on the franchise trap, 2015-2019 relocation franchises (LA/LAC/LV) in the WR panel:
    13 player-season rows on those franchises; sos_vegas missing 0, sos_wr_fpa missing 0

## 3. The 16 pre-registered tests

`R = b0 + b1 * (x - season mean of x) + u`, OLS, SEs clustered by season (10 clusters, t with 9 df), HC3 alongside. One measure per regression, never blended. Raw-level (uncentered) coefficient reported for all 16 as the pre-specified sensitivity.

### WR panel
```
              measure window   n    beta  se_cluster  p_raw_cluster  se_hc3  p_raw_hc3  beta_per_sd  mde_per_sd  ceiling_per_sd  power_ratio
            sos_vegas   full 291 -0.4498      0.4194         0.3114  0.7725     0.5603      -0.1100      0.3227          0.0614       5.2565
      sos_vegas_w1_14  w1_14 291 -0.6006      0.5676         0.3176  0.6685     0.3690      -0.1816      0.5398          0.0759       7.1134
     sos_vegas_w15_17 w15_17 291  0.0864      0.1870         0.6551  0.1856     0.6416       0.0853      0.5812          0.2479       2.3441
       sos_prior_wpct   full 291  3.2801      3.3586         0.3543  5.4122     0.5445       0.1106      0.3562          0.1439       2.4759
 sos_prior_wpct_w1_14  w1_14 291  0.5944      3.3707         0.8639  4.8937     0.9033       0.0230      0.4111          0.1655       2.4848
sos_prior_wpct_w15_17 w15_17 291  1.3263      1.6193         0.4339  1.7357     0.4448       0.1426      0.5478          0.4589       1.1937
           sos_wr_fpa   full 291  0.1560      0.1541         0.3379  0.2105     0.4586       0.1523      0.4734          0.1000       4.7336
    sos_wr_fpa_w15_17 w15_17 291 -0.0949      0.0517         0.0994  0.0842     0.2597      -0.2098      0.3594          0.1000       3.5940
```

### RB panel
```
              measure window   n    beta  se_cluster  p_raw_cluster  se_hc3  p_raw_hc3  beta_per_sd  mde_per_sd  ceiling_per_sd  power_ratio
            sos_vegas   full 286 -0.1616      0.5862         0.7890  0.8428     0.8479      -0.0395      0.4511          0.0614       7.3469
      sos_vegas_w1_14  w1_14 286  0.1211      0.5666         0.8355  0.6715     0.8569       0.0366      0.5388          0.0759       7.1001
     sos_vegas_w15_17 w15_17 286 -0.0985      0.2071         0.6458  0.2069     0.6340      -0.0973      0.6435          0.2479       2.5956
       sos_prior_wpct   full 286  0.5600      7.3823         0.9412  5.8100     0.9232       0.0189      0.7830          0.1439       5.4421
 sos_prior_wpct_w1_14  w1_14 286 -1.2982      4.7245         0.7897  5.3388     0.8079      -0.0503      0.5763          0.1655       3.4828
sos_prior_wpct_w15_17 w15_17 286  0.8244      2.5788         0.7565  1.9365     0.6703       0.0887      0.8724          0.4589       1.9010
           sos_rb_fpa   full 286  0.2961      0.5006         0.5688  0.3060     0.3332       0.1945      1.0345          0.1000      10.3451
    sos_rb_fpa_w15_17 w15_17 286  0.0508      0.1624         0.7616  0.1177     0.6663       0.0908      0.9130          0.1000       9.1297
```

## 4. §K1's pre-test power prediction — did it hold?

§K1, recorded before fitting: *the full-season test is predicted to be underpowered by more than an order of magnitude*. The test of that prediction is MDE(per SD) / ceiling(per SD) > 10.
```
panel               measure window  mde_per_sd  ceiling_per_sd  power_ratio
   WR             sos_vegas   full       0.323           0.061        5.256
   WR       sos_vegas_w1_14  w1_14       0.540           0.076        7.113
   WR      sos_vegas_w15_17 w15_17       0.581           0.248        2.344
   WR        sos_prior_wpct   full       0.356           0.144        2.476
   WR  sos_prior_wpct_w1_14  w1_14       0.411           0.165        2.485
   WR sos_prior_wpct_w15_17 w15_17       0.548           0.459        1.194
   WR            sos_wr_fpa   full       0.473           0.100        4.734
   WR     sos_wr_fpa_w15_17 w15_17       0.359           0.100        3.594
   RB             sos_vegas   full       0.451           0.061        7.347
   RB       sos_vegas_w1_14  w1_14       0.539           0.076        7.100
   RB      sos_vegas_w15_17 w15_17       0.644           0.248        2.596
   RB        sos_prior_wpct   full       0.783           0.144        5.442
   RB  sos_prior_wpct_w1_14  w1_14       0.576           0.165        3.483
   RB sos_prior_wpct_w15_17 w15_17       0.872           0.459        1.901
   RB            sos_rb_fpa   full       1.035           0.100       10.345
   RB     sos_rb_fpa_w15_17 w15_17       0.913           0.100        9.130
```
- full-season tests: power ratio min 2.5, max 10.3, median 5.3
- weeks-15-17 tests (§K2 primary): power ratio min 1.2, max 9.1, median 2.5
- §K1 prediction (>10x underpowered, full season): DID NOT HOLD for all full-season tests — 1/6 full-season tests exceed 10x.

## 5. BH q = 0.10 within the §K family (16 tests, primary cluster p-values)
```
panel               measure window  p_raw_cluster  bh_rank  bh_thresh  fdr_survivor
   WR     sos_wr_fpa_w15_17 w15_17        0.09944        1    0.00625         False
   WR             sos_vegas   full        0.31143        2    0.01250         False
   WR       sos_vegas_w1_14  w1_14        0.31760        3    0.01875         False
   WR            sos_wr_fpa   full        0.33786        4    0.02500         False
   WR        sos_prior_wpct   full        0.35427        5    0.03125         False
   WR sos_prior_wpct_w15_17 w15_17        0.43390        6    0.03750         False
   RB            sos_rb_fpa   full        0.56877        7    0.04375         False
   RB      sos_vegas_w15_17 w15_17        0.64579        8    0.05000         False
   WR      sos_vegas_w15_17 w15_17        0.65513        9    0.05625         False
   RB sos_prior_wpct_w15_17 w15_17        0.75649       10    0.06250         False
   RB     sos_rb_fpa_w15_17 w15_17        0.76162       11    0.06875         False
   RB             sos_vegas   full        0.78898       12    0.07500         False
   RB  sos_prior_wpct_w1_14  w1_14        0.78970       13    0.08125         False
   RB       sos_vegas_w1_14  w1_14        0.83554       14    0.08750         False
   WR  sos_prior_wpct_w1_14  w1_14        0.86393       15    0.09375         False
   RB        sos_prior_wpct   full        0.94119       16    0.10000         False
```
- BH survivors within §K: NONE
- the closed {§H5, §I3} family is NOT re-corrected; §K stands alone, as declared.

## 6. Temporal holdout (binding): fit 2015-2022, evaluate 2023-2024
```
panel               measure window    beta  beta_train  beta_eval  sign_stable  holdout_mse  mse_zero  holdout_improves
   WR     sos_wr_fpa_w15_17 w15_17 -0.0949     -0.0677    -0.1781         True       9.3559    9.3356             False
   WR             sos_vegas   full -0.4498     -0.1514    -1.0054         True       9.4387    9.3356             False
   WR       sos_vegas_w1_14  w1_14 -0.6006     -0.6013    -0.5975         True       9.4254    9.3356             False
   WR            sos_wr_fpa   full  0.1560      0.0722     0.6233         True       9.4131    9.3356             False
   WR        sos_prior_wpct   full  3.2801      1.7202    12.6515         True       9.4316    9.3356             False
   WR sos_prior_wpct_w15_17 w15_17  1.3263      2.4625    -2.6692        False       9.7074    9.3356             False
   RB            sos_rb_fpa   full  0.2961      0.3417    -0.0340        False      13.8154   13.6749             False
   RB      sos_vegas_w15_17 w15_17 -0.0985      0.0288    -0.4972        False      13.8166   13.6749             False
   WR      sos_vegas_w15_17 w15_17  0.0864      0.2033    -0.1865        False       9.6381    9.3356             False
   RB sos_prior_wpct_w15_17 w15_17  0.8244      1.5449    -3.5841        False      13.8823   13.6749             False
   RB     sos_rb_fpa_w15_17 w15_17  0.0508      0.0023     0.2531         True      13.7789   13.6749             False
   RB             sos_vegas   full -0.1616     -0.0450    -0.4110         True      13.7792   13.6749             False
   RB  sos_prior_wpct_w1_14  w1_14 -1.2982     -2.4420     6.1409        False      13.8132   13.6749             False
   RB       sos_vegas_w1_14  w1_14  0.1211     -0.3251     1.3603        False      13.8963   13.6749             False
   WR  sos_prior_wpct_w1_14  w1_14  0.5944     -0.8627     9.2447        False       9.4845    9.3356             False
   RB        sos_prior_wpct   full  0.5600      0.8902    -0.6899        False      13.7853   13.6749             False
```

**Surviving BOTH screens (BH within §K + temporal holdout): NONE**
- holdout alone would pass 0/16 (a coin-flip screen on its own; both screens are required, as pre-registered)
- train/eval sign stability: 7/16 measures keep their sign between 2015-22 and 2023-24

## 7. Post-hoc, descriptive — why the null, and which §K5 branch it supports

*Nothing in this section is a hypothesis test, nothing enters the §K FDR family, and no specification above was altered on the strength of any of it.*

### (i) The §25.1 decomposition: b_realized = b_priced + b_residual
R = PPG - m_iso(ADP) is an identity, so cov(x,R) = cov(x,PPG) - cov(x,m_iso). A flat b_residual with a flat b_realized means the MEASURE carries no signal; a flat b_residual with a live b_realized and a matching b_priced means the channel is PRICED. These are different findings and §K5 requires saying which one this is.
(all betas in PPG per within-season SD of the measure)
```
panel               measure  sd_within  b_realized  p_realized  b_priced  p_priced  b_residual  p_residual
   WR             sos_vegas     0.2446      0.0268      0.9040    0.1368    0.3966     -0.1100      0.3114
   WR       sos_vegas_w1_14     0.3023      0.0181      0.9145    0.1996    0.2906     -0.1816      0.3176
   WR      sos_vegas_w15_17     0.9878      0.0944      0.5304    0.0090    0.9533      0.0853      0.6551
   WR        sos_prior_wpct     0.0337      0.1253      0.5791    0.0147    0.9185      0.1106      0.3543
   WR  sos_prior_wpct_w1_14     0.0388      0.0418      0.8318    0.0187    0.8954      0.0230      0.8639
   WR sos_prior_wpct_w15_17     0.1075      0.1625      0.3515    0.0199    0.9083      0.1426      0.4339
   WR            sos_wr_fpa     0.9765      0.2872      0.1184    0.1349    0.4618      0.1523      0.3379
   WR     sos_wr_fpa_w15_17     2.2110      0.0524      0.7733    0.2623    0.0723     -0.2098      0.0994
   RB             sos_vegas     0.2446     -0.2629      0.2926   -0.2233    0.3107     -0.0395      0.7890
   RB       sos_vegas_w1_14     0.3023     -0.1220      0.5356   -0.1586    0.1254      0.0366      0.8355
   RB      sos_vegas_w15_17     0.9878     -0.0540      0.8867    0.0432    0.8355     -0.0973      0.6458
   RB        sos_prior_wpct     0.0337      0.2167      0.2786    0.1978    0.3161      0.0189      0.9412
   RB  sos_prior_wpct_w1_14     0.0388      0.1388      0.5871    0.1892    0.4204     -0.0503      0.7897
   RB sos_prior_wpct_w15_17     0.1075      0.2893      0.4313    0.2007    0.1903      0.0887      0.7565
   RB            sos_rb_fpa     0.6569     -0.0647      0.8341   -0.2592    0.2149      0.1945      0.5688
   RB     sos_rb_fpa_w15_17     1.7874     -0.0877      0.8010   -0.1785    0.4048      0.0908      0.7616
```

### (ii) Year-over-year persistence of the defensive ingredient
The premise of positional SOS: that a defence's prior-year allowance predicts this year's. Measured on our own weekly data, cross-team correlation of PPR allowed per game between consecutive seasons.
```
pos            RB     WR
transition              
2015->2016  0.367  0.149
2016->2017  0.388  0.458
2017->2018  0.307  0.406
2018->2019  0.000  0.209
2019->2020  0.471  0.397
2020->2021  0.341  0.241
2021->2022  0.239  0.299
2022->2023  0.447  0.203
2023->2024  0.282 -0.052
```
- WR: mean r = 0.257, last two transitions 0.203, -0.052
- RB: mean r = 0.316, last two transitions 0.447, 0.282

### (iii) The clairvoyant bound: contemporaneous positional SOS
Built from the SAME season's realized defensive allowances mapped onto the team's actual opponents. This is LOOK-AHEAD and is BARRED as a feature — no August drafter could see it. It is computed only as an upper bound: it is what a positional-SOS measure would be worth if the lag were perfect. If even this is flat, the null is about matchups; if it is live, the null is about the lag.
```
panel window   n  sd_within  beta_per_sd  se_per_sd  p_raw
   WR   full 291     0.9648       0.7595     0.2461 0.0130
   WR w15_17 291     2.1663       0.1669     0.1817 0.3822
   RB   full 286     0.7735       0.7251     0.2235 0.0101
   RB w15_17 286     1.8255       0.4186     0.2606 0.1427
```

### (iv) Leave-own-team-out: is the clairvoyant bound real or mechanical?
A defence's realized FPA is computed from the points scored AGAINST it — which includes the points scored by the very player whose residual is the outcome. Player i inflates each of his 17 opponents' season FPA by roughly (his own points in that game) / (that defence's games). His x is the mean over those opponents, so the inflation does NOT average away: it is an i-specific shift proportional to i's own production, and R is i's own production net of price. That is a mechanical positive correlation with nothing to do with matchups. Rebuilt excluding every game the defence played against the player's own team, so neither he nor his teammates can contribute to his own regressor.
```
panel window   n  sd_within  beta_per_sd_naive  beta_per_sd_LOTO  se_per_sd  p_raw
   WR   full 291     1.0805             0.7595           -0.0829     0.2771 0.7715
   WR w15_17 291     2.2682             0.1669           -0.2317     0.1394 0.1309
   RB   full 286     0.8312             0.7251            0.0267     0.2780 0.9257
   RB w15_17 286     1.8599             0.4186            0.0250     0.2902 0.9331
```

The same exclusion applied to the LAGGED (pre-registered) positional measures, as a descriptive sensitivity — the pre-registered tests above stand as run:
```
panel           measure  beta_per_sd_prereg  beta_per_sd_prereg_no2015  beta_per_sd_LOTO  p_LOTO  n_prereg  n_LOTO
   WR        sos_wr_fpa              0.1523                     0.1454            0.0352  0.8622       291     261
   WR sos_wr_fpa_w15_17             -0.2098                    -0.1923            0.0230  0.9232       291     248
   RB        sos_rb_fpa              0.1945                     0.0113           -0.1052  0.6592       286     256
   RB sos_rb_fpa_w15_17              0.0908                     0.0128            0.0112  0.9583       286     246
```
- n falls because the LOTO build needs season y-1 weekly data and the cached window starts 2015, so board year 2015 drops out. The `_no2015` column is the pre-registered spec on the same years, so the LOTO column is compared like for like rather than against a different sample.

### (v) What (iv) does to the §K5 question — the chained bound
The naive clairvoyant result in (iii) was an artifact and is withdrawn as evidence. Under the leave-own-team-out build, a positional schedule index with PERFECT FORESIGHT is flat. That is the honest input to §K5, and it has to be read with its own precision, not as a zero:
- WR: clairvoyant full-season beta -0.083 PPG per SD, 95% CI [-0.710, +0.544], MDE 0.872 PPG per SD.
    chaining: a LAGGED measure can carry at most the year-over-year persistence of the ingredient (0.26 for WR) times the clairvoyant effect, so the largest lagged effect consistent with the upper end of that CI is +0.140 PPG per SD — against a pre-registered MDE of 0.473.
- RB: clairvoyant full-season beta +0.027 PPG per SD, 95% CI [-0.602, +0.656], MDE 0.875 PPG per SD.
    chaining: a LAGGED measure can carry at most the year-over-year persistence of the ingredient (0.32 for RB) times the clairvoyant effect, so the largest lagged effect consistent with the upper end of that CI is +0.207 PPG per SD — against a pre-registered MDE of 1.035.

## 9. Reconciling §K1's power prediction with the realised design

§K1 predicted >10x underpowering and got 5.3x for the headline full-season WR vegas test. The prediction's ceiling (0.061 PPG per SD) is reproduced exactly; the discrepancy is entirely in the MDE, where §K1 imported §I3's 0.87 PPG-per-SD and the realised value is 0.323. That import is the error, and it is worth stating precisely because it is a reusable lesson about the estimand.

MDE per SD is, to first order, (t_.975 + t_.80) * SD(R) / sqrt(n_eff) — it does NOT depend on SD(x), because the per-unit MDE scales as 1/SD(x) and multiplying back by SD(x) cancels it. So a per-SD MDE cannot be transplanted between features of different dispersion *unless the error structure is the same*. It is not here:
```
panel                        spec     se  mde_per_sd
   WR  centered + cluster(season) 0.4194      0.3227
   WR raw level + cluster(season) 0.4609      0.3546
   WR              centered + HC3 0.7725      0.5294
   WR     iid benchmark (t9 crit)    NaN      0.6128
   WR     SD of season means of R    NaN      0.8903
   RB  centered + cluster(season) 0.5862      0.4511
   RB raw level + cluster(season) 0.4063      0.3126
   RB              centered + HC3 0.8428      0.5776
   RB     iid benchmark (t9 crit)    NaN      0.6740
   RB     SD of season means of R    NaN      0.7325
```
- The mechanism: season-centering x makes the regressor exactly orthogonal to season dummies, so the season-common component of R (SD of season means of R = 0.89 PPG, a large share of SD(R) = 3.32) drops out of the cluster score sum_s (sum_i x_is u_is) entirely. The cluster SE therefore lands BELOW the iid benchmark rather than above it. §I3 regressed on raw levels, where the between-season component is in the regressor and that variance is fully charged to a 10-cluster SE. Same outcome, same n, same clustering — a 2.7x difference in per-SD precision, purely from the within-vs-pooled estimand.
- So the §K1 prediction was directionally right and quantitatively wrong, in the conservative direction: the within-season design is about 2.7x more precise per SD than the number §K1 borrowed. The full-season tests remain badly underpowered (2.5x to 10.3x short of their own ceilings) — they just are not short by an order of magnitude except for the RB positional pair.

## 10. Verdict

**NULL. No schedule arm enters LOSO.** Zero of 16 tests survive BH at q = 0.10 within the §K family (smallest raw p = 0.0994 against a BH threshold of 0.00625), and zero of 16 improve on the zero prediction in the 2023-24 holdout. Both screens are binding and both are failed; the adoption decision does not depend on which screen you weight.

**On §K2's primary designation.** The weeks-15-17 window was designated the primary test in advance because it has 4x the dispersion. It delivered the family's smallest p-value (WR positional, raw p = 0.099) — and that coefficient is NEGATIVE, i.e. a softer fantasy-playoff WR schedule going with a WORSE outcome against price. That is the wrong sign for a matchup story and the right sign for a mild market overreaction; the decomposition puts b_priced at +0.26 PPG per SD (p = 0.072) against b_realized of +0.05, which is what an overpriced-but-undelivered channel looks like. It is nowhere near the BH threshold, it fails the holdout, and its leave-own-team-out rebuild is +0.02. We report it because it is the family minimum and readers will find it, not because we believe it.

Reported because it cuts against us: on that same term the pre-specified raw-level sensitivity is MORE significant than the primary (p = 0.013 vs 0.099), so the primary specification is not the one flattering the null. It still fails BH by a factor of two and still fails the holdout. The Huber M-estimator on the same term gives p = 0.213, which locates the nominal significance in the tail of a right-skewed residual rather than in the body of the panel.

**Which §K5 branch the evidence supports.** §K5 required us to say whether a positional null means the MEASURE carries little signal or that MATCHUPS do not matter. Our evidence supports NEITHER conclusion strongly, and the reason is power, not ambivalence. The decisive diagnostic — a positional index built with perfect foresight — is flat once the mechanical own-production feedback is removed, but its own MDE is ~0.87 PPG per SD. Chaining that CI through the measured ~0.26 (WR) / ~0.32 (RB) year-over-year persistence bounds the largest LAGGED effect consistent with the data at roughly 0.15 PPG per SD — which is below every pre-registered MDE in the family (0.32 to 1.03). The pre-registered tests were therefore incapable of detecting the largest effect the data allows. The correct statement is that the positional-SOS nulls are UNINFORMATIVE about matchups, and that the attenuation chain (weak persistence, then season-level aggregation) is sufficient on its own to explain why no lagged positional measure could have worked. We do not claim matchups are irrelevant; nothing here tests that.

**On the full-season tests, per §K1.** Reported as UNINFORMATIVE, not as evidence of absence. Every full-season MDE exceeds its own ceiling by 2.5x to 10.3x.

## 11. Artefacts
- `results/edge_schedule.csv` — the 16 tests: beta, cluster SE, HC3, raw p, pre-test MDE and ceiling, BH rank/threshold/survivor, holdout, sensitivities.
- `results/edge_schedule_decomposition.csv` — the b_realized / b_priced / b_residual split per measure.
- `results/edge_schedule_clairvoyant.csv` — the look-ahead positional index, naive vs leave-own-team-out. Diagnostic only; barred as a feature.
- `scripts/25_sectionK_schedule_edge.py` — rerunnable, pre-registration in the docstring.

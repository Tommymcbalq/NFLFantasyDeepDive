# §S — Is the mean the right summary of a player's history?

Pre-registered in `EDA_PLAN7.md` (2026-08-18). This file records (a) the operational
pre-specification of every candidate, written **before any fitting**, (b) the bake-off
result, and (c) the §S4 rebuild.

---

## §S1 pre-specification — operational definitions fixed before running

`EDA_PLAN7.md` names the eight candidates but does not fully operationalise them. The
following choices are made here, before any candidate has been evaluated, and are not
revisited afterwards.

### The weighting scheme, and why it is held fixed rather than dropped

The incumbent is a **recency-weighted mean of season means**:

    mu_1 = sum_s w_s * ybar_s / sum_s w_s ,    w_s = 2^{-(S_max - s)/h},  h = 1

where s ranges over the player's prior seasons, ybar_s is his mean PPR over *included*
games in season s (WR: targets >= 2; RB: carries + targets >= 2 — the §1/§G2 inclusion
rule, unchanged), and S_max is his most recent prior season.

Candidates 2–5 are **game-level** functionals. A game-level statistic computed on the raw
pooled game list would differ from the incumbent in *two* ways at once: the location
functional (mean → median/trimmed/Huber/p60) **and** the temporal weighting (recency-
weighted season means → unweighted pooled games). That would not isolate the summary
statistic, which is the whole point of the exercise.

So each game i in season s is given weight

    u_i = w_s / G_s                                                            (S1.1)

where G_s is the number of included games in season s. Under (S1.1) the weighted *mean*
of game-level PPR is **algebraically identical to mu_1**:

    sum_i u_i y_i / sum_i u_i = sum_s (w_s/G_s) * sum_{i in s} y_i / sum_s w_s
                              = sum_s w_s * ybar_s / sum_s w_s  =  mu_1

Candidates 2–5 therefore change *only* the functional applied to the weighted game
distribution. This is a nesting property, not a convenience: it guarantees that a
difference in LOSO loss is attributable to robustness/location choice and nothing else.
The unweighted-pooled variant is run as a **declared sensitivity**, reported whichever way
it comes out.

### Everything downstream of mu_hat is frozen

Per §S1, `B`, `V`, `tau2(tier)`, `sigma2(tier)`, `m(.)`, the inclusion rule and the fold
structure are identical across all eight arms. In particular **n_eff is the incumbent's
n_eff in every arm**, including candidate 6 — B is a fixed component by pre-registration.
For candidate 6 this is the conservative choice (recomputing n_eff on the qualifying
subset would shrink it harder toward the market and confound the test with a shrinkage
change); the recomputed-n_eff version is run as a declared sensitivity.

Rows with no prior NFL data (n_eff = 0) take `theta = m_hat` in every arm, as in §7/§P.

### The eight arms

| # | name | definition |
|---|---|---|
| 1 | **incumbent** | `mu_1` above. Frozen benchmark. |
| 2 | median | weighted median of the game-level PPR distribution under (S1.1) |
| 3 | trimmed mean | weighted 20%-each-tail trimmed mean: drop the lowest 20% and highest 20% of *weight*, mean of the rest |
| 4 | Huber | weighted Huber M-estimator, tuning constant **fixed at 1.345**, scale = weighted MAD/0.6745 re-estimated at each iteration, 100 iterations or tol 1e-10. If MAD = 0, falls back to the weighted mean. |
| 5 | p60 | weighted 60th percentile of the game-level distribution |
| 6 | role-stable | `mu_1` recomputed over **only** prior seasons with G_s >= 12, with w_s renormalised inside that subset (S_max taken over qualifying seasons). Falls back to all seasons when none qualify. |
| 7 | slope-adjusted | within-fold OLS `ppg ~ 1 + mu_1 + d` fitted on training rows, where `d = ybar(latest prior season) - ybar(second-latest)`, and `d = 0` for players with fewer than two prior seasons. The fitted value is the arm's mu_hat. |
| 8 | usage-implied | within-fold OLS `ppg ~ 1 + x` where `x` = (player's share of team targets in his most recent prior season) x (that team's targets per game that season) for WR; the same with touches (carries + targets) for RB. Missing `x` takes the training-fold mean of `x`. |

Weighted quantiles use the standard right-continuous inverse-CDF definition on the
cumulative weight, with weights normalised to sum to 1.

Arms 7 and 8 are fitted **inside the fold** on `in_fit` training rows only; nothing from
the held-out year enters the fit.

### Panel, loss, test

Primary panel: the §P **wide** board panels, `results/market_prior_{wr,rb}_deep.csv`
(666 WR / 603 RB rows, 2015–2024), evaluated on `in_fit` rows — the *existing* fold
structure, so numbers are directly comparable to §P's Table in REPORT §33 and, on the
`adp_rank <= 30` stratum, to §7.

Loss: squared error on realised PPG. Diebold–Mariano on the ten yearly mean loss
differentials, `t(9)`, two-sided. Positive `mean_gain` = candidate better.

MDE, reported beside every p-value, at the **adoption** alpha of 0.05:

    MDE(80% power, two-sided alpha = 0.05) = (t_{0.975,9} + t_{0.80,9}) * SD_folds / sqrt(10)   (S1.2)

### Adoption rule (from §S2, restated, fixed)

Replace mu_hat only if a candidate (i) beats the incumbent at **p < 0.05**, (ii) improves
pooled RMSE, and (iii) survives the **temporal holdout** (fit 2015–21, evaluate 2022–24).
Family = the seven challengers, **BH q = 0.10**, declared per position as written in §S2;
the pooled 14-test BH is reported as a stated robustness. **Ties go to the incumbent.**

### §S3 power statement, binding

Differences between summary statistics of the same data are far smaller than the
difference between having a data arm and not having one. §7's WR gain over market-only was
+0.695 with across-fold SD 0.819 → MDE ≈ 0.82 at 10 clusters. **This test is expected to
be underpowered for small effects. A null here is UNINFORMATIVE — it is not evidence of
equivalence.** (§28.1.) Every null below is to be read that way.

### Pre-registered expectations, recorded so they can be falsified

Candidates 2–5 differ from the mean only in how they treat boom weeks, and §37 found
dispersion does not persist year to year (r(IQR) = +.19). They are expected to be close to
indistinguishable. Candidate 6 is the one with a mechanism (§P: the deviation is worth
+1.101 of face value with a full prior season, +0.042 without).

*(Results appended below after running — nothing above this line was edited afterwards.)*

---

# §S1/§S2 RESULT — run 2026-08-18, `scripts/59_sectionS_bakeoff.py`

**Harness validation, before any candidate is read.** The incumbent arm inside this harness
reproduces §P4 exactly: WR wide panel RMSE 3.4354 (market) → 3.4011 (θ\*), mean fold gain
+0.2458, p = .2327; RB 3.7506 → 3.7909, −0.3132, p = .3282. Those are §P's published
numbers to four decimals, so any difference below is attributable to the μ̂ swap alone.

## The bake-off, primary panel (wide, `in_fit` rows)

WR n = 638, incumbent RMSE **3.4011**, mean Spearman .6124.
RB n = 573, incumbent RMSE **3.7909**, mean Spearman .6079.

| pos | arm | RMSE | ΔRMSE | mean gain | DM t(9) | **p** | **MDE** | obs/MDE | folds ↑ | BH thr | BH |
|---|---|---|---|---|---|---|---|---|---|---|---|
| WR | 2 median | 3.4979 | +0.0968 | −0.670 | −1.87 | .0937 | 1.125 | −0.60 | 2/10 | .0286 | — |
| WR | 3 trimmed 20% | 3.4342 | +0.0331 | −0.229 | −1.27 | .2355 | 0.566 | −0.40 | 3/10 | .0571 | — |
| WR | 4 Huber | 3.4137 | +0.0127 | −0.087 | −0.98 | .3542 | 0.279 | −0.31 | 4/10 | .0714 | — |
| WR | 5 p60 | 3.5356 | +0.1345 | −0.917 | −4.75 | **.0010** | 0.608 | −1.51 | 0/10 | .0143 | **reject (worse)** |
| WR | **6 seasons G ≥ 12** | 3.4326 | +0.0316 | −0.218 | −1.62 | .1387 | 0.423 | −0.52 | 2/10 | .0429 | — |
| WR | 7 slope-adjusted | 3.3924 | **−0.0086** | **+0.059** | +0.33 | .7526 | 0.573 | +0.10 | 6/10 | .0857 | — |
| WR | 8 usage-implied | 3.3998 | −0.0013 | −0.006 | −0.02 | .9830 | 0.924 | −0.01 | 5/10 | .1000 | — |
| RB | 2 median | 3.8150 | +0.0240 | −0.205 | −0.70 | .5040 | 0.927 | −0.22 | 4/10 | .0714 | — |
| RB | 3 trimmed 20% | 3.7926 | +0.0017 | −0.024 | −0.14 | .8953 | 0.560 | −0.04 | 5/10 | .1000 | — |
| RB | 4 Huber | 3.7858 | **−0.0051** | **+0.033** | +0.31 | .7655 | 0.334 | +0.10 | 6/10 | .0857 | — |
| RB | 5 p60 | 3.8487 | +0.0578 | −0.452 | −2.69 | .0246 | 0.527 | −0.86 | 2/10 | .0143 | — |
| RB | **6 seasons G ≥ 12** | 3.8592 | +0.0683 | −0.513 | −2.29 | .0474 | 0.703 | −0.73 | 1/10 | .0286 | — |
| RB | 7 slope-adjusted | 3.7510 | **−0.0399** | **+0.309** | +2.12 | .0629 | 0.458 | +0.67 | 8/10 | .0429 | — |
| RB | 8 usage-implied | 3.8123 | +0.0214 | −0.148 | −0.72 | .4898 | 0.646 | −0.23 | 6/10 | .0571 | — |

BH q = 0.10 over the declared family of seven, per position. Pooled 14-test BH changes
nothing (same single rejection). Temporal holdout in `sectionS_holdout.csv`.

## VERDICT — **no replacement. μ̂ stays the recency-weighted mean of season means, h = 1.**

Not one challenger clears the three-pronged bar. Only two arms have a positive point
estimate *and* an RMSE improvement in either position — arm 7 at RB (+0.309, p = .063) and
arm 4 at RB (+0.033, p = .766) — and neither reaches p < 0.05, so both lose the tie to the
incumbent by rule. The single BH rejection is arm 5 (p60) at WR, **in the wrong direction**.
The board is therefore rebuilt with `mu_arm = a1_mean` and is numerically unchanged.

**Read the nulls as §S3 requires.** Realised MDEs run 0.28–1.12 PPG² against observed
effects of 0.01–0.9. Arms 3, 4 and 8 are inside their own MDE at both positions: the test
**cannot** distinguish them from the incumbent, and that is not evidence they are equal.
Arm 7 at RB is the one genuinely suggestive result — 8/10 folds, obs/MDE = 0.67, and it
survives the temporal holdout (RMSE 3.7081 → 3.6642) — and it is *not adopted*, because the
rule was fixed before the number was seen. It is the strongest candidate for a
pre-registered re-test on a larger panel.

---

# §S1 follow-up — the three diagnostics (`scripts/59b_sectionS_diagnostics.py`)

Post-hoc, non-adoptable, run only after the table above was final.

## D-A. Why candidate 6 — the one with a mechanism — went the wrong way

The §P interaction is **not** in doubt; it re-estimates on this harness almost exactly as
published. Regressing realised PPG on m̂ and the deviation (θ\* − m̂):

| coefficient on the deviation | WR | RB |
|---|---|---|
| last prior season ≥ 12 games | **+1.098** (n = 458) | +0.514 (n = 359) |
| last prior season < 12 games | **−0.026** (n = 110) | −0.064 (n = 130) |

(§P published +1.101 / +0.042 pooled.) So the finding candidate 6 was built on is real.
**Candidate 6 is simply the wrong operationalisation of it,** and the decomposition says so:

| rows | n WR | gain WR | p | n RB | gain RB | p |
|---|---|---|---|---|---|---|
| candidate 6 leaves alone | 388 | 0.000 | — | 335 | 0.000 | — |
| candidate 6 changes | 250 | −0.554 | .135 | 238 | −1.137 | .042 |
|  … last season partial (< 12 g) | 68 | −0.866 | .424 | 72 | **−2.767** | .037 |
|  … last season full (≥ 12 g) | 182 | **−0.269** | .019 | 166 | −0.361 | .145 |

Two separate failures, and both are informative.

1. **On the rows the mechanism targets, dropping partial seasons makes μ̂ optimistic.**
   Mean Δμ̂ on partial-last-season rows is **+0.45 (WR) / +1.33 (RB)** — strictly upward.
   Seasons are partial mostly because of injury, and injury-shortened seasons are also
   low-PPG seasons; deleting them removes the bad draws and keeps the good ones. That is
   selection, not noise reduction. It is **the same error §T rejected**: conditioning away
   the games where things went wrong, when the forecast target includes whatever goes
   wrong next year.
2. **It also damages rows the mechanism says nothing about.** Candidate 6 drops *any*
   short season, including old ones, from players whose most recent season was complete —
   182 WR rows, −0.269, p = .019. Those players' μ̂ is being made staler for no reason.

**What §P's finding actually implies is a change to B, not to μ̂.** "The deviation is worth
+1.10 when the prior season is full and 0.00 when it is not" is a statement about how much
weight the data arm deserves. See D-C.

## D-B. Why the robust arms lose: it is *not* a level shift

Every robust functional is biased against the mean on a right-skewed distribution —
median −1.21 (WR) / −1.43 (RB), trimmed −0.87 / −0.98, Huber −0.53 / −0.61, p60 +0.90 /
+0.50. The obvious hypothesis is that the whole failure is that offset. It is not. Removing
it — recentring each arm by the **training-fold** mean offset, so the arm is unbiased by
construction — leaves every arm still worse, and at WR makes the losses *sharper*:

| recentred arm | WR gain | p | RB gain | p |
|---|---|---|---|---|
| median | −0.697 | .013 | −0.227 | .222 |
| trimmed 20% | −0.275 | **.0017** | −0.079 | .378 |
| Huber | −0.137 | **.0006** | −0.030 | .540 |
| p60 | −0.688 | .0049 | −0.323 | .053 |

The WR ordering is **monotone in how much tail is discarded**: Huber (−0.14) < trimmed
(−0.27) < median ≈ p60 (−0.69). That is a dose-response, not noise.

> **The boom weeks are signal, not contamination.** The forecast target is next season's
> *mean* PPG, and the sample mean is the efficient estimator of a mean. Robust estimators
> buy resistance to outliers by throwing away information, and there are no outliers here
> to resist — a 38-point game is a real observation of the quantity being estimated.

This **falsifies the pre-registered expectation**, which was that arms 2–5 would be
indistinguishable from the mean because §37 found dispersion does not persist. The
expectation conflated two things: §37 says last season's *shape* does not predict next
season's *shape*, which says nothing about which location estimator best predicts the
*level*. Recorded as a wrong prediction.

## D-C. The correction §P's finding does imply — reported, and NOT adopted

Two shrinkage variants, both leaving μ̂ untouched:

* **D1** — market-anchor any row whose most recent prior season had < 12 games (B := 1).
* **D2** — shrink in proportion to games played: B′ = 1 − (1 − B)·min(G_last/12, 1).

| | n treated | RMSE | ΔRMSE | mean gain | DM t(9) | p | MDE |
|---|---|---|---|---|---|---|---|
| WR D1 | 110 | 3.3621 | **−0.0390** | +0.260 | +1.74 | .117 | 0.472 |
| WR D2 | 110 | 3.3718 | −0.0292 | +0.197 | +1.84 | .099 | 0.337 |
| RB D1 | 130 | 3.7437 | **−0.0472** | +0.370 | +1.89 | .092 | 0.616 |
| RB D2 | 130 | 3.7606 | −0.0303 | +0.235 | +2.45 | **.037** | 0.302 |

Same sign, both positions, both parameterisations, and RB D2 reaches p = .037. **This is
not adopted and must not be.** It was constructed after seeing candidate 6 fail, it is one
of two variants I chose after the fact, and it carries no multiple-testing control. It is
recorded here as a **round-8 pre-registration candidate** — the first genuinely promising
one in seven rounds — to be tested under a family declared in advance.

**It is also not in conflict with §U.** §U found that on rows where the QB pairing broke,
μ̂ beats m(ADP), so shrinking those rows toward the market hurts. Those rows have *precise*
histories about a changed situation. D-C's rows have *imprecise* histories — the player
barely played. §P's interaction variable is games, not situation, and the two treatments
overlap only incidentally.

---

# §S4 — the rebuild (`scripts/50_build_board.py`)

One script, one pass, from `data/players/weekly_raw/*` + the 2026 ADP pull + the frozen
fitted objects (deep isotonic knots, tier variances). Every layer is a named column, in the
order applied: `adp → pi_market → mu_hat → B → theta_star → value_prior → view_shift →
value_post_views → replacement → vorp → floor_gap → final`. Output `results/board_2026.csv`,
204 players (88 WR / 68 RB / 24 TE / 24 QB).

**Reproduction check.** With the stored board's *rounded* replacement constants the rebuild
reproduces `results/board_2026_overall_vorp.csv` to machine precision on every layer:
`value_post_views` 5.3e-15, `vorp` 5.3e-15, `floor_gap` 1.8e-15, `final` 5.3e-15. The
rebuild is therefore a re-expression of the current board, not a new one.

**Replacement, recomputed rather than hardcoded.** The 12-team 2026 board's top 140 skill
players are 63 WR / 44 RB / 19 QB / 14 TE. Replacement is the (n+1)-th best realised season
total at each position, recency-weighted over 2021–25 with half-life 2, ÷ 17:

| pos | n in top 140 | rank used | wtd season total | **replacement PPG** | stored |
|---|---|---|---|---|---|
| RB | 44 | 45th | 105.557 | **6.209** | 6.21 |
| WR | 63 | 64th | 108.494 | **6.382** | 6.38 |
| TE | 14 | 15th | 133.725 | **7.866** | 7.87 |
| QB | 19 | 20th | 205.752 | **12.103** | 12.10 |

De-rounding is the only intended difference from the stored board: max |Δfinal| = 0.0038
PPG, Spearman 0.999994, 5/204 rank changes, all adjacent swaps.

**Floor.** p25 of PPR over scheduled weeks 2023–25 (18 weeks per season, missed games as
zeros, Rice's 2025 six-week suspension removed from the denominator, no floor below 34
eligible weeks). Recomputed here from raw and asserted equal to `results/floor_scheduled.csv`
on all 180 shared rows. Reference floor = median p25 of the top 70 by `vorp` with usable
history: **WR 4.6625, RB 6.1500, TE 7.0250, QB 9.8200**. λ = 0.10. Players without usable
history take gap 0.

**Views applied exactly once**, with five assertions in the script: (a) the BL prior is
`value_prior` bit-identical; (b) `pi_market` is re-derived from ADP alone and compared
exactly, so nothing view-shaped can have leaked into the prior; (c) Σ is diagonal and every
unviewed player moves by < 1e-9; (d) the per-view decomposition sums to the total shift
(max error 3.6e-15); (e) applying the same 31 views a *second* time moves 31 players — if
it were a no-op the prior would already contain them, which is the historical double-count
bug this guards against.

# §H notes — the WR/RB aging curve and whether it has moved (2026-08-09)

Executed as pre-registered in `EDA_PLAN4.md` §H. Nothing was tuned toward an expected
result; no named player appears anywhere in the pipeline. The one deviation from the plan
is a **data repair forced by an upstream defect** (§H0.1 below), which was applied before
any era comparison was believed and is reported in both directions (defective and repaired
panels are both on disk).

**Headline: the drop-off has not moved later. Both positions, every specification, the
point estimates move the drop-off *earlier*, and the career-exit hazard agrees.**

---

## §H0.1 Data defect found while qualifying the panel (not anticipated by the plan)

In the nflverse weekly release, `targets` is degenerate — league-wide sum ≈ 0 while
receptions and receiving yards are intact — for **seasons 2003–2008**, at every position:

| targets / receptions | 1999–2002 | 2003–2008 | 2009–2025 |
|---|---|---|---|
| WR | 1.84 → 1.77 | 0.000 – 0.005 | 1.74 → 1.60 |
| RB | 1.40 → 1.36 | 0.000 – 0.010 | 1.37 → 1.28 |

The pre-registered qualification rule is ≥ 8 games **and ≥ 40 touches (carries + targets)**,
so **zero WRs qualified in 2003–2008**. WR "era 1 (1999–2007)" was in fact 1999–2002 only,
and era 2 lost 2008. The central §H hypothesis — an era comparison — would have rested on
four seasons of WR data at one end, and the §H0 qualified-N table would have been read as a
substantive finding about a low-volume passing era (22% qualification rate in era 1 vs 43%
later) when it was an artifact.

**Repair** (`scripts/22_repair_panel.py`, deterministic, computed without reference to the
outcome, applied identically to both positions): for the six defective seasons,
`targets_hat = receptions × ρ_p`, ρ_p = the position's targets/receptions ratio pooled over
the eight nearest clean seasons (1999–2002, 2009–2012); ρ_WR = 1.7781, ρ_RB = 1.3768. PPR
points are computed from receptions, not targets, so **the outcome is untouched — only the
qualification screen changes.** Raw data is never overwritten:
`data/derived/age_panel_long_repaired.csv`.

Everything below is the repaired panel. The defective-panel run is retained as
`results/*_rawpanel.csv`; **its conclusions are the same** (WR cliff 30.65/28.80/28.10,
era×age Wald p = .90), which is the reassurance that the repair did not manufacture the
result — it only made era 1 estimable.

## §H0 Qualified-N by era (reported before any modelling)

| pos | era | player-seasons | qualified | rate | players | mean PPG | mean age | age p1–p99 |
|---|---|---|---|---|---|---|---|---|
| RB | 1999–2007 | 1273 | 638 | .501 | 220 | 9.99 | 26.49 | 21.5–33.5 |
| RB | 2008–2016 | 1265 | 688 | .544 | 243 | 9.53 | 25.93 | 21.2–33.2 |
| RB | 2017–2025 | 1365 | 718 | .526 | 230 | 9.74 | 25.60 | 21.2–33.5 |
| WR | 1999–2007 | 1697 | 821 | .484 | 239 | 10.40 | 27.39 | 21.7–37.1 |
| WR | 2008–2016 | 1847 | 888 | .481 | 292 | 10.48 | 26.55 | 21.3–35.8 |
| WR | 2017–2025 | 2099 | 903 | .430 | 301 | 10.41 | 26.09 | 21.4–33.8 |

Qualification rates are flat across eras after repair (WR .48/.48/.43, RB .50/.54/.53), so
the era contrasts are not a moving selection bar. Two facts to carry: the **mean age of a
qualified player falls monotonically** (WR 27.4 → 26.1, RB 26.5 → 25.6) and the **upper age
support shrinks** (WR p99 37.1 → 33.8). Both are consequences of the finding, not nuisances.

## Estimator and identification (as pre-specified)

- Outcome r_is = PPG_is / mean PPG among qualified players at that position in season s.
  Necessary: with player FE, age and period are exactly collinear within player (APC), so on
  the raw scale a league-wide scoring trend loads onto the linear component of f and the era
  comparison is uninterpretable. Absolute-PPG runs are reported below, labelled confounded.
- f = natural cubic spline (patsy `cr`), interior knots at the pooled qualified panel's age
  quintiles for that position (WR 23.69/25.28/27.00/29.34; RB 23.48/24.82/26.43/28.39),
  boundary knots at that panel's min/max. Basis built **once per position** and reused for
  every era and every bootstrap replicate, so the curves are comparable by construction.
- Player FE absorbed by within-player demeaning (FWL); OLS on the demeaned system. All CIs
  and tests by **cluster bootstrap on player**, B = 600 (400 for the balanced cohort, 300
  for the hazard), players resampled with replacement, each draw a distinct unit.
- **Level identification.** With player FE, f is identified only up to an additive constant.
  Convention fixed before fitting and applied identically to every era and replicate: each
  era's curve is shifted so its sample mean over that era's observations equals the sample
  mean of r over the same observations (= 1 by construction). "Cliff" = the first age above
  the peak at which the anchored curve falls 10% below its anchored peak. Features are read
  only within each era's [p1, p99] age support.

---

## §H1 Pooled age profile (relative PPG)

| pos | n | players | peak | cliff | slope 28→32 (r/yr) |
|---|---|---|---|---|---|
| WR | 2612 | 664 | **25.60** (25.20, 26.70) | **29.50** (28.25, 30.45) | −0.0509 (−0.0631, −0.0386) |
| RB | 2044 | 567 | **24.75** (23.60, 26.10) | **27.55** (26.35, 28.40) | −0.0849 (−0.1091, −0.0624) |

RB peaks ~0.9 yr earlier and declines ~1.7× faster than WR. Absolute-PPG sensitivity
(confounded by league scoring trend) gives the same peaks: WR 25.60, RB 24.60.

## §H2 Era interaction — the actual hypothesis

**Relative PPG, per era (95% cluster-bootstrap CIs):**

| pos | era | n | peak | cliff | slope 28→32 |
|---|---|---|---|---|---|
| WR | 1999–2007 | 821 | 25.75 (25.20, 29.60) | **31.05** (27.05, 32.75) | −0.0321 (−0.0558, −0.0081) |
| WR | 2008–2016 | 888 | 25.75 (23.35, 27.40) | **29.35** (27.55, 30.65) | −0.0605 (−0.0820, −0.0385) |
| WR | 2017–2025 | 903 | 25.25 (24.30, 26.10) | **28.05** (26.75, 29.60) | −0.0704 (−0.0978, −0.0440) |
| RB | 1999–2007 | 638 | 26.15 (21.50, 27.25) | **28.65** (22.05, 30.25) | −0.0981 (−0.1553, −0.0481) |
| RB | 2008–2016 | 688 | 24.80 (23.65, 27.05) | **28.30** (25.75, 31.35) | −0.0490 (−0.0861, −0.0124) |
| RB | 2017–2025 | 718 | 24.35 (21.25, 26.05) | **26.95** (25.35, 28.00) | −0.0748 (−0.1031, −0.0305) |

**Era differences (modern − oldest), bootstrap CIs and two-sided bootstrap p:**

| pos | Δpeak | Δcliff | Δslope 28→32 |
|---|---|---|---|
| WR | −0.50 (−4.40, +0.60), p = .20 | **−3.00** (−5.15, +1.45), p = .21 | **−0.038** (−0.077, −0.003), p = .033 |
| RB | −1.80 (−4.65, +3.40), p = .39 | −1.70 (−4.05, +4.75), p = .25 | +0.023 (−0.029, +0.094), p = .42 |

**Formal interaction test (bootstrap-covariance Wald on the era × spline block, df = 10):**
- WR: W = 13.75, **p = .185**
- RB: W = 11.38, **p = .329**

**Verdict on H2.** The omnibus era × age interaction is **not rejected** at either position.
Every point estimate of peak and cliff moves *earlier*, not later — WR cliff 31.05 → 29.35 →
28.05 (monotone across three eras, −3.0 yr end-to-end) — but the cliff CIs comfortably
include zero shift, so on the production curve alone the honest statement is: **no evidence
the drop-off has moved later; a directionally consistent but statistically inconclusive
signal that it has moved ~1–3 years earlier for WR.** The one era contrast that does clear
its own CI is the **WR 28→32 slope, which is 2.2× steeper in 2017–2025 than in 1999–2007**
(−0.070 vs −0.032 relative-PPG/yr, Δ CI excluding 0, p = .033) — the modern WR curve is not
shifted, it is *steeper past the peak*. The absolute-PPG (confounded) run agrees and is
somewhat stronger on the cliff (Δcliff −3.85, p = .073; Δslope −0.471, p = .010).

### Robustness: the smooth age × centred-season check is weakly identified — reported, not hidden
The pre-registered smooth alternative (age spline × continuous centred season, player FE) is
**not credibly identified**, and this is a property of the design, not of the code. Within a
player, cohort = season − age is constant, so any φ(season − age) added to the surface is
absorbed by α_i; inside the bilinear family f(a) + g(a)·z the only direction that φ cannot
reach is the one requiring a z² term, i.e. f and g are separated by functional form alone.
Diagnostics: cond(X̃) = 18,573 (WR) / 14,514 (RB), max|β| = 25.7 / 58.8, and the
reconstructed per-season curves are degenerate (peak pinned to a grid edge). Reported as a
failed check with its cause.
Re-run under an alternative identification (**pooled OLS, no player FE, cluster-robust by
player**, where age and season are not collinear): age × season interaction F = 9.50,
p = .091 (WR) and F = 8.86, p = .115 (RB); peak drift per decade CI (−5.89, +1.81) WR and
(−4.22, +3.87) RB — **no later shift under this identification either.**

### Robustness: constant-selection-rate qualification (post-hoc, prompted by §H0)
Qualify the top-K per position-season by touches (K = 85 WR / 65 RB = the minimum per-season
qualified count), so the selection *rate* is constant by construction. Touch floors become
41/45/47 (WR) and 40/57/66 (RB) by era. Result is unchanged: WR cliff 30.80 / 29.55 / 28.35,
Δcliff p = .375, **Δslope −0.077 to −0.012, p = .020**; RB cliff 28.20 / 27.90 / 27.25, all
deltas n.s.

---

## §H3 Selection guards

### H3a Balanced cohort (players with ≥ 6 qualified seasons)
WR 176/664 players, 1452 rows (462/549/441 by era); RB 131/567, 992 rows (313/365/314).

| pos | 1999–2007 | 2008–2016 | 2017–2025 |
|---|---|---|---|
| WR peak / cliff | 25.70 / 31.60 | 26.75 / 29.70 | 25.05 / 28.45 |
| WR slope 28→32 | −0.0214 (−.049, +.010) | −0.0654 (−.091, −.043) | −0.0737 (−.104, −.047) |
| RB peak / cliff | 26.50 / 28.35 | 25.15 / 29.40 | 23.95 / 27.25 |
| RB slope 28→32 | −0.1143 | −0.0549 | −0.0821 |

Same picture on long-career players only: WR cliff falls 31.6 → 28.5 and the modern slope is
3.4× the era-1 slope. Restricting to survivors does not produce a later modern drop-off.

### H3b Career-exit hazard (the guard the plan says must agree)
Discrete-time logit of P(season s is the player's last qualified season | qualified at s) on
age spline × era, cluster-robust by player; 2025 dropped as right-censored.
WR n = 2520, 572 exits (22.7%); RB n = 1970, 493 exits (25.0%).

| pos | era | h(30) | h(32) | age at hazard .40 |
|---|---|---|---|---|
| WR | 1999–2007 | **.231** (.174, .288) | **.250** (.186, .322) | > support |
| WR | 2008–2016 | .259 (.205, .326) | .327 (.248, .427) | 33.20 |
| WR | 2017–2025 | **.377** (.287, .469) | **.433** (.346, .544) | 30.95 |
| RB | 1999–2007 | **.313** (.224, .409) | .479 (.382, .592) | 31.20 |
| RB | 2008–2016 | .410 (.327, .533) | .534 (.434, .701) | 29.85 |
| RB | 2017–2025 | **.474** (.362, .604) | .519 (.429, .771) | 28.40 |

Contrasts, modern − oldest (bootstrap p): WR Δh(30) = +.039 to +.253, **p = .007**;
Δh(32) CI (+.072, +.307), **p < .001**; Δ(age at hazard .40) CI (−7.63, −1.24), **p < .001**.
RB Δh(30) CI (+.022, +.316), **p = .033**; Δ(age at hazard .40) CI (−4.25, +0.30), p = .067.
Omnibus era × age Wald: WR χ²(10) = 14.89, p = .136; RB χ²(10) = 17.07, p = .073 (under the
constant-selection rule, RB χ²(10) = 24.01, **p = .008**).

**H3b does not corroborate a later drop-off — it significantly contradicts one.** A WR aged
30 is ~63% more likely to be in his final qualified season in 2017–2025 than in 1999–2007
(.377 vs .231), and the age at which the exit hazard reaches .40 has moved *earlier* by
1.2–7.6 years. The plan's rule was "H2 agreeing with H3b is the evidence." They agree — on
the opposite of the hypothesis under test. The hazard is the sharper of the two because it
is far less exposed to the survivorship artifact, and it is the estimate that turns a
directional pattern in H2 into a defensible claim.

**Why the two guards matter here.** Survivorship attenuates H1/H2 toward *flatness* at old
ages (only the productive old survive to be measured), so the observed modern steepening is
if anything an understatement; and the modern era's shorter upper age support (p99 33.8 vs
37.1) means the era-3 curve is *less* extrapolated at the ages where it declines fastest.

---

## §H4 RB workload carryover — the pre-registered coefficient is mean reversion

Transitions with both seasons qualified: n = 1317, 388 players, 132 with ≥ 350 prior touches,
mean prior touches 201.

**As pre-registered** (Δr on prior-season touches, linear + quadratic + ≥350 indicator,
controlling f(age) + player FE):
touches/100 β = **−0.458** (−0.610, −0.321); quadratic +0.032 (−0.003, +0.070);
≥350 indicator −0.091 (−0.231, +0.056); implied Δr from a 200 → 350 prior-touch season
= **−0.513** (−0.615, −0.408).

Taken at face value that is a half-of-league-average collapse, which is not credible, so it
was decomposed. **corr(prior touches, prior r) = 0.854**, and prior r enters Δr with
coefficient −1 by construction, so the specification is mechanically loaded with regression
to the mean:

| spec | implied Δr, 200 → 350 prior touches |
|---|---|
| as pre-registered: Δr ~ t + t² + heavy | **−0.513** (−0.618, −0.416) |
| + control for prior r | **+0.002** (−0.105, +0.117) |
| level: r_s ~ t + t² + heavy | +0.115 (+0.019, +0.219) |
| level + prior r | +0.002 (−0.096, +0.124) |
| **placebo: NEXT season's touches on this season's Δr** | β = **+0.073** (+0.020, +0.125) |

The placebo is the clincher: next-season touches "predict" this season's change with a
significant coefficient, which no causal story permits. **Conclusion: there is no detectable
RB workload carryover separable from mean reversion and from age.** The decline attributed to
age does *not* decompose into heavy-prior-load damage; conditional on where a back already
is, a 350-touch season carries no additional penalty (CI ±0.11 relative PPG — a genuinely
informative null, not merely an underpowered one).

---

## §H5 Does the market price it? — **null on the binding screens**

Panel: FFC PPR ADP 2015–2024 (2025 unavailable at source), top-30 per position per year.
WR residual R = `resid_iso` from the §6.1 isotonic ADP→PPG curve (n = 291 in-fit). RB: the
§G3 analogue built identically here (isotonic on log ADP, monotone decreasing, ≥ 4 games;
296 board rows matched, 282 in-fit; `results/rb_market_prior.csv`). Spec
R ~ age + age² + era + era×age + era×age² (+ prior-season touches/100 and a no-prior
indicator for RB), age centred at the position mean. `era` on a 2015–2024 panel can only
contrast 2015–16 with 2017–24 (`late` = 1{year ≥ 2017}, n = 57/234 WR, 54/228 RB) — a real
limitation, stated rather than worked around. OLS, SEs clustered by season (10 clusters,
t with 9 df), HC3 as robustness.

**MULTIPLE TESTING: p-values below are RAW and uncorrected.** Round 4's FDR family is
{H5, I3} and §I3 is being estimated separately, so the binding correction is applied jointly
at consolidation, not here. A within-§H5 BH column is written to `sectionH_h5.csv` marked
`bh_q_provisional_H5_only` and is **not** the decision quantity.

| pos | term | β | p_raw (cluster) | p_HC3 |
|---|---|---|---|---|
| WR | age | −0.026 | .807 | .905 |
| WR | age² | −0.001 | .951 | .990 |
| WR | age × late | −0.057 | .676 | .810 |
| WR | age² × late | −0.005 | .847 | .949 |
| RB | age | −0.105 | .123 | .710 |
| RB | age² | +0.051 | **.0023** | .312 |
| RB | age × late | −0.151 | .179 | .607 |
| RB | age² × late | −0.044 | .195 | .452 |
| RB | prior touches/100 | +0.182 | .481 | .479 |

Joint (raw): WR age level F = 0.06, p = .810; WR age × era F = 0.10, p = .908.
RB age level F = 9.46, p = .0061; RB age × era F = 2.43, p = .143. R² = .007 (WR), .025 (RB).

**Temporal holdout 2015–22 → 2023–24 (the second, independent screen):**

| pos | model MSE | zero prediction | verdict |
|---|---|---|---|
| WR | 9.911 | 9.336 | **FAILS** |
| RB | 14.611 | 13.664 | **FAILS** |
| WR age-only variant | 9.602 | 9.336 | FAILS |
| RB age-only variant | 14.106 | 13.664 | FAILS |

**The RB age² term, chased.** Clustered p = .0023 against HC3 p = .312 is a large enough
discrepancy to require an explanation before it is reported as anything. Leave-one-year-out
is stable (β ∈ [0.036, 0.061]), but **year-by-year separate fits range −0.183 to +0.146 and
flip sign in 2019 and 2020**, so the effect is not year-stable; Huber RLM gives p = .329; the
joint Wald hits the cluster-covariance rank boundary (7 constraints, 10 clusters — statsmodels
reports rank 4 of 5 and rank deficiency on the 2-constraint tests too), exactly the
few-clusters pathology already documented in §6.2. The small clustered SE is an artifact of
9 residual df, not evidence. Descriptively the term says the market residual is U-shaped with
a minimum near age 26.6 — but there are only 16 board RBs aged ≥ 30 in ten years.

**Verdict — and it does not depend on the pending joint FDR.** Adoption required survival of
*both* screens. **The holdout fails for both positions, on raw p-values, before any
correction** — so no age arm enters the LOSO harness under any outcome of the joint {H5, I3}
FDR. The market prices age at least as well as this curve can, on either the old or the new
shape. Per the pre-registered decision rule, the §H1–H3 curve **informs the §J views layer
only, labelled unvalidated**, and **no age adjustment enters the statistical board on the
strength of H2.** This is the second independent time the project has reached this
conclusion (§C round 2: DM t = −0.02, p = .983).

---

## What §H establishes

1. **The hypothesis is not supported.** The modern age–production profile is not shifted
   later. Every point estimate at both positions moves the peak and the cliff *earlier*, the
   omnibus era × age interaction is not rejected (WR p = .185, RB p = .329), and the exit
   hazard — the guard designed to be robust to survivorship — moves significantly earlier
   (WR h(30) .231 → .377, p = .007; RB .313 → .474, p = .033).
2. **The one era effect that clears its CI is WR steepening, not shifting**: the 28→32 slope
   is 2.2× steeper in 2017–2025 than in 1999–2007 (p = .033; p = .020 under constant-selection
   qualification; p = .010 on the confounded absolute scale). Modern WRs peak at the same age
   and fall off the far side faster.
3. **RB is where the level of the aging problem lives** (peak 24.75, decline 1.7× WR's), but
   its *era* structure is flat — RB has been a young man's position for all 27 seasons.
4. **The "heavy workload ages a back" mechanism is not in this data** once mean reversion is
   removed (+0.002, CI ±0.11), and the pre-registered specification's −0.51 is an artifact
   its own placebo detects.
5. **None of it is tradeable through ADP.** H5 fails the temporal holdout at both positions.

## Files
- `scripts/20_sectionH_aging.py` (H0–H4), `21_sectionH_robust.py` (identification diagnostic,
  constant-selection, H4 decomposition), `22_repair_panel.py` (data repair),
  `23_sectionH_h5.py` (H5 + RB market prior), `24_sectionH_figures.py`.
  Rerun in that order; `H_PANEL=raw` reproduces the defective-panel run into `*_rawpanel.csv`.
- `results/age_curve_era.csv` (per-era anchored curves + bootstrap bands on the age grid),
  `age_curve_features.csv` (peak/cliff/slope + CIs), `age_era_tests.csv`,
  `age_curve_balanced.csv`, `age_curve_constsel.csv`, `age_curve_smooth_pooled.csv`,
  `exit_hazard.csv`, `h4_workload.csv`, `h4_workload_decomp.csv`, `sectionH_h5.csv`,
  `rb_market_prior.csv`; `*_rawpanel.csv` counterparts.
- Figures: `results/figures/sectionH_age_curve_WR.png`, `sectionH_age_curve_RB.png`,
  `sectionH_exit_hazard.png`, `sectionH_data_defect.png`, `sectionH_h4_workload.png`,
  `sectionH_h5_residual_age.png`.

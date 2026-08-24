# §W1 — The projection engine (L1 + L2)

*Operational definitions fixed here on 2026-08-24 BEFORE any model was fitted, per EDA_PLAN9.md
WS1. Nothing below the rule line was known when the rules above it were written.*

## W1.0 The question

Can expected PPG be projected from **inputs** — usage, volume, efficiency, environment, age —
better than it can be summarised from **past output**? Every arm tested in eight rounds has been a
summary of past output (§S tested seven of them) or an adjustment bolted onto one. §D tested a
ridge on usage covariates inside a different framing and failed. This is the projection model.

## W1.1 Panels, target, folds

- **Panels**: the §P wide market-prior panels, `results/market_prior_wr_deep.csv` (666 rows, 638
  `in_fit`) and `results/market_prior_rb_deep.csv` (603 / 573). Same rows, same `in_fit` screen,
  same folds as §7/§P4/§S. This makes every number here directly comparable to the published
  incumbent numbers (WR RMSE 3.4011, RB 3.7909 for θ*; μ̂-alone figures are computed fresh).
- **Target (primary)**: `ppg`, realised PPR points per game *played* in the panel year Y.
- **Target (secondary, L2.2)**: points per **scheduled** week,
  `ppsw = games·ppg / S(Y)`, S = 16 for Y ≤ 2020 (S = 16 in 2020 too — 16-game schedule),
  17 for Y ≥ 2021.
- **Folds**: leave-one-season-out, Y ∈ {2015,…,2024}. Everything — feature standardisation,
  penalty selection, gate membership where it is refit, isotonic m(·), τ², σ² — is fit on
  `year != Y` only.
- **Feature construction**: from seasons **strictly before Y** only, plus the preseason-known
  team for year Y (the ADP file's `team`, which is a preseason field).
  *[Falsified in execution — see W1.R0. The ADP file's `team` is an end-of-season label. The
  environment block was rebuilt on the prior-season team and the leaked version priced as a
  labelled sensitivity. This line is left as written, because a pre-registration that is
  silently edited is not a pre-registration.]*

## W1.2 The coverage split, and why there are two panels

The advanced-stats layer (`data/derived/adv_*.csv`, `team_context.csv`) covers **2018–2025** only.
A feature from season Y−1 therefore exists only for Y ≥ 2019. Rather than throw away four folds or
smuggle in a shorter test, two feature tiers are declared:

- **Tier A (10 folds, 2015–2024)** — built from `data/players/weekly_raw/` (1999–2025) and
  `data/meta/players_meta.csv`. This is the **primary** test, on the same ten clusters as §7.
- **Tier B (6 folds, 2019–2024)** — Tier A plus the advanced layer (snaps, routes, situational
  usage, NGS, PFR, team PROE/pace). **Secondary**, and it is understood in advance that six
  clusters is a badly underpowered DM test; the realised MDE is reported and any null there is
  labelled uninformative per §28.1.

## W1.3 Candidate features, fixed now

Tier A, all computed from prior seasons with recency weight $w_s = 2^{-(S_{\max}-s)/h}$, h = 1
(the same weighting as μ̂, so the projection and the incumbent see the same temporal window):

- **Volume**: targets/g, receptions/g, carries/g, touches/g, `target_share_full`,
  `air_yards_share_full` (both computed with a **full-team-season denominator** — player targets ÷
  team targets summed over all team games — never the active-games denominator), team volume per
  game × share (the §S arm-8 usage index), games played in Y−1, weighted games/season.
- **Efficiency**: yards/target, yards/reception, aDOT, YAC/reception, TD/target, catch rate,
  RACR (winsorized), receiving EPA/target. **Admitted only if they pass the §4 gate re-run**
  (below). Volume features are admitted on §4's published result (target share .896/.703, air-yards
  share .874/.709) but are re-gated for the record.
- **Environment**: prior-season pass attempts/g, plays/g, points/g and pass rate of the **year-Y
  team** (preseason-known), plus an indicator for team change. *[Superseded by W1.R0 — read as
  the prior-season team; `team_change` dropped.]*
- **Structure**: age at 1 Sep of year Y (natural cubic spline, df 3, knots at panel quintiles fixed
  before fitting), experience, log draft pick (undrafted = 260).

Tier B adds: snap share, `routes_proxy_pg`, `tprr_proxy`, `yprr_proxy`, red-zone / inside-10 /
end-zone / third-down targets per game, deep-target rate, `pass_snap_share`, NGS separation and
cushion and YAC-over-expected, PFR aDOT / YBC-per-rec / broken tackles / drop%, and team
`neutral_proe`, `neutral_sec_per_play`, `off_epa`, `rz_td_rate`. RB adds goal-line carries/g,
third-down carries/g, stuffed rate, explosive-run rate, RYOE/att, short-yardage conversion.

**`routes_proxy` caveat, recorded before use**: it counts blocking TEs and pass-protecting backs,
so TPRR/YPRR are biased low and the bias differs by archetype. They enter as candidates only, and
only if they pass the gate.

## W1.4 The reliability gate (§4 machinery, re-run)

Split-half odd/even weeks within a season on the **full positional population**, not the board:
$r_{\text{half}}$ → Spearman–Brown $\rho_{\text{full}} = 2r/(1+r)$; year-over-year $r_{YoY}$ with a
player-clustered bootstrap (2,000 draws). **Admission rule, unchanged from §4 and binding:**
$\rho_{\text{full}} \ge 0.5$ **and** the $r_{YoY}$ 95% CI excludes 0. A feature that fails is
excluded from every specification, regardless of in-sample correlation with the target. Season-level
features that cannot be split (snap share is weekly and can; NGS season aggregates cannot) are gated
on $r_{YoY}$ alone and that exception is declared here rather than discovered later.

## W1.5 Estimators

All three fit on training folds only, all features standardised on training-fold moments:

- **(a) Ridge / elastic net.** Penalty α on a fixed log grid $10^{-2}\ldots10^{4}$, chosen by
  5-fold CV **within the training folds, grouped by year** (so no held-out season ever touches
  penalty selection). Elastic net (l1_ratio ∈ {0.1, 0.5, 0.9}) run as a declared sensitivity.
- **(b) Gradient-boosted trees.** `HistGradientBoostingRegressor`, hyperparameters **declared now
  and not tuned**: `max_depth=3`, `learning_rate=0.05`, `max_iter=300`, `min_samples_leaf=20`,
  `l2_regularization=1.0`, early stopping off. One fixed configuration, no search.
- **(c) Hierarchical / partial pooling.** WR and RB pooled with a shared coefficient block plus
  position-specific deviation block; the deviation block carries its own penalty $\lambda/\kappa$
  with $\kappa$ chosen on the same inner grouped CV. $\kappa \to 0$ is complete pooling,
  $\kappa \to \infty$ is separate fits.

**Pre-specified expectation (falsifiable, recorded): the regularised linear model wins or ties.**
n ≈ 600 player-seasons per position against $\rho_{\max} \approx 0.41$ (§2) leaves a tree ensemble
nothing to find that a linear fit cannot. If a tree wins, the win is chased before it is believed.

## W1.6 Model scopes

- **P0 — inputs only.** No function of the player's own past PPR points enters. This is the literal
  answer to W1.0.
- **P1 — inputs + μ̂.** μ̂ enters as one more standardised feature. This is the practically relevant
  question: do inputs add anything *to* the output summary?

## W1.7 The binding comparison

Head to head against **μ̂**, the recency-weighted mean of season means, h = 1 (eq. 43.1) — the §S
incumbent — as a direct predictor of `ppg`. Primary rows: `in_fit` **and** $n_{\text{eff}} > 0$
(μ̂ exists, so the comparison is defined). Diebold–Mariano on squared-error differentials averaged
within year, t(K−1), K = 10 (Tier A) or 6 (Tier B). Realised MDE at 80% power, two-sided α = .05,
printed beside every p (§28.1); a null with |observed| < MDE is labelled **uninformative**, not
evidence of equivalence.

**A second, harder benchmark, declared before fitting (added 2026-08-24, still pre-fit).** Raw μ̂ is
an uncalibrated predictor: its regression slope on realised `ppg` is not 1. A ridge is a fitted
regression and is shrunk by construction, so a projection could beat raw μ̂ purely by recalibrating
it, with no contribution from inputs at all. So **μ̂-cal** — OLS `ppg ~ 1 + μ̂` fitted on training
folds only — is carried as a second benchmark in every table. Raw μ̂ remains the binding
pre-registered comparison (it is the object actually in the pipeline); μ̂-cal is the diagnostic that
says whether any win is about *inputs* or about *calibration*.

Reported alongside (secondary, for downstream relevance): the projection substituted for μ̂ inside
eq. (7), $\theta^* = (1-B)\hat y + B\,m(\text{ADP})$ with B, τ², σ², m(·) held exactly as in §S, so
any difference is the arm swap and nothing else.

**Multiplicity.** Declared family: {P0, P1} × {ridge, GBT, hierarchical} × {WR, RB} = 12 tests.
BH at q = 0.10 within position (6) and pooled (12), both reported.

**Temporal holdout.** Fit 2015–2021, evaluate 2022–2024, as in §S. Required for adoption.

**Adoption rule, fixed before any result is seen.** A projection replaces μ̂ as the data arm at a
position only if all four hold: (i) RMSE lower than μ̂; (ii) DM p < .05; (iii) survives BH at
q = .10 in the 12-test family; (iv) beats μ̂ on the temporal holdout. **Ties go to the incumbent.**
Anything short of all four is reported as a null with its MDE and the incumbent stands.

## W1.8 L2.1 — the age curve applied

§H estimated per-era relative-PPG age curves $f_e(\cdot)$ (era 3 = 2017–2025;
`results/age_curve_era.csv`). §H5 found **no market edge** in age, and nothing here contradicts
that: this is a *within-model structural correction*, not an edge claim. Two forms, both declared:

- **A1, multiplicative carry-forward**: $\hat y^{\text{age}} = \hat y \cdot
  f_3(\text{age}_Y)/f_3(\text{age}_{Y-1})$ — the expected relative-production change from one more
  year of age, applied to whatever the projection says.
- **A2, age in the design**: the age spline is already a Tier-A feature, so P0/P1 estimate their own
  age effect. Reported as the comparison that says whether the *external* §H curve adds anything
  over an age term fit in-sample.

A1 is adopted only under the same four-part rule as W1.7, tested against the projection it modifies.

## W1.9 L2.2 — availability as a modelled input

Two stages, both must clear their own bar:

1. **Does prior availability predict next-season availability?** Model games played in Y (out of
   S(Y)) on prior availability — recency-weighted games-played rate, games in Y−1, career mean rate,
   number of prior seasons — plus age spline and position. Beta-binomial / logistic-link fractional
   regression, LOSO, compared to the constant baseline (fold-mean rate) by DM on squared error with
   MDE. **Bar: out-of-sample MSE below the constant baseline with DM p < .05.**
2. **Only if stage 1 clears**, `ppsw` is projected as a second target and reported alongside PPG.
   Both a direct projection of `ppsw` and the decomposition $\hat y_{\text{PPG}} \times
   \widehat{\text{avail}}$ are evaluated against the incumbent $\hat\mu^{\text{ppsw}}$ (recency-
   weighted mean of prior per-scheduled-week rates). The owner has ruled out a post-hoc expected-
   games multiplier; a validated second target is a different object and is reported as such. If
   stage 1 fails, no availability term enters anything and that is the finding.

## W1.10 The §P interaction, and where it is allowed to act

§P found, re-estimated in §43.5 at **+1.098 (WR) / +0.514 (RB)** for $G_{\text{last}} \ge 12$ and
**−0.026 / −0.064** for $G_{\text{last}} < 12$: a player's history is worth face value when he
played a full prior season and nothing when he did not. §S established the **wrong** fix is deleting
partial seasons from μ̂ — that removes bad draws and keeps good ones, a selection artifact worth
+0.45 PPG (WR) / +1.33 (RB) upward on the treated rows.

The pre-registered hypothesis here, stated before testing: **the interaction belongs in the
precision of the estimate, not in the estimate.** Operationally, the data arm's variance is inflated
for short prior seasons, $B' = 1 - (1-B)\cdot\min(G_{\text{last}}/12, 1)$ (the §43.5 D2 variant,
now pre-registered rather than post hoc), and the same treatment is applied to the projection arm if
one is adopted. Tested inside the four-part rule, with the D1 hard-anchoring variant as the declared
alternative. This is the round-8 pre-registration candidate §43.8 item 3 recorded, now executed.

---
*Rule line. Everything above was fixed before fitting. Results follow.*

# Results

Scripts: `62_sectionW1_gate.py` (gate), `63_sectionW1_features.py` (features),
`64_sectionW1_projection.py` (L1 + ablation + holdout + encompassing),
`65_sectionW1_structural.py` (L2.1, L2.2), `66_sectionW1_precision.py` (W1.10 + residuals),
`67_sectionW1_project2026.py` (2026 application).

## W1.R0 A leak in the panel, found before adoption, and fixed

The FFC historical ADP `team` field is an **end-of-season label, not a draft-day label**.
Among the 42 panel rows whose player played for two teams in year Y, panel `team` matches his
**final** team 88.1% of the time and his **week-1** team 7.1%. The environment block as first
specified (year-Y team's prior-season offence) therefore leaked in-season information for traded
players, and so did `team_change`.

Fixed by defining the environment on the **prior-season team**, which is unambiguously
preseason-knowable, and dropping `team_change`. The leaked version was retained as a labelled
sensitivity so the leak could be priced: it is worth **0.023 RMSE at WR and 0.045 at RB**
(ridge_P1: 3.5072 clean vs 3.4870 leaky; 4.0712 vs 4.0261). Small, and every headline number
below is from the clean specification. **This matters beyond WS1: any layer that treats the
historical panel's `team` as preseason information is leaking.**

## W1.R1 The reliability gate (`results/sectionW1_gate.csv`)

Harness validated against §4's published WR table first (2014–2025 window): target share
.897/.686 vs published .896/.703; WOPR .895/.687 vs .893/.708; air-yards share .877/.691 vs
.874/.709; aDOT .803/.689 vs .809/.666; PPR PPG .811/.633 vs .802/.644. Reproduced.

Wide window 2006–2025, n = 1,111 WR / 934 RB / 553 TE year-pairs. **Membership (WR):**

| stat | ρ_full | r_YoY [95% CI] | verdict |
|---|---|---|---|
| target share (full denom.) | .892 | .672 [.630,.709] | **ADMIT** |
| WOPR (full denom.) | .891 | .675 [.629,.714] | **ADMIT** |
| air-yards share (full denom.) | .876 | .685 [.639,.725] | **ADMIT** |
| targets/g | .874 | .667 | **ADMIT** |
| receptions/g | .860 | .663 | **ADMIT** |
| receiving yards/g | .806 | .624 | **ADMIT** |
| PPR PPG | .810 | .633 | **ADMIT** |
| aDOT | .800 | .688 | **ADMIT** |
| RACR (winsorized) | .645 | .453 [.243,.605] | **ADMIT** |
| yards/reception | .546 | .424 | **ADMIT** |
| catch rate | .541 | .476 | **ADMIT** |
| YAC/reception | .498 | .403 | REJECT (ρ_full < .5) |
| yards/target | .357 | .264 | REJECT |
| TD/target | .245 | .169 | REJECT |
| receiving EPA/target | .308 | .231 | REJECT |

RB: carry share .956/.661, carries/g .937/.675, touches/g .937/.644, targets/g .918/.670,
catch rate .939/.642, yards/target .906/.622, rec EPA/target .730/.542 all **ADMIT**;
**ypc .384/.285 and rush-TD/carry .406/.138 REJECT.** The §4 verdict survives verbatim on a
panel 45% larger: *usage is signal, efficiency is mostly luck, TD rate is noise.*

**The Tier-B exception, and how binding it actually is.** Season-aggregated advanced stats
cannot be split odd/even, so W1.4 declared them gated on r_YoY alone. That exception turns out
to be nearly non-binding: at n ≈ 478 (WR) almost any positive correlation excludes 0, and 20/22
WR and 18/21 RB advanced stats are admitted, including `pfr_drop_pct` at r = .135 and RB
`rush_epa_per_att` at r = .101. Three fail outright (TE `rz_target_share_of_own`, RB
`stuffed_rate` .070, RB `short_yd_conv_rate` .094, RB `target_epa` .004).

A rigorous tightening is available *inside* the §4 framework rather than post hoc:
$r_{YoY} = \rho_X\varphi_X$ with $\varphi_X \le 1$, so $\rho_X \ge r_{YoY}$ — **any stat with
r_YoY ≥ 0.5 has ρ_full ≥ 0.5 certified.** That certified subset is {snap share, TPRR, YPRR,
deep targets/g, deep-target rate, third-down targets/g, NGS separation, NGS share of intended
air yards, PFR aDOT, PFR YBC/rec} at WR and {snap share, run-snap share, routes, opportunity/g,
pass-snap share} at RB. Both versions were run; the Tier-B result is a null either way (§W1.R3),
so the exception never decided anything.

## W1.R2 The LOSO scorecard vs μ̂ (`results/sectionW1_loso.csv`)

Tier A, gated, clean environment. Rows: `in_fit` and n_eff > 0. n = 568 WR / 489 RB, 10 folds.
**μ̂ RMSE 3.7760 (WR) / 4.4909 (RB).**

| pos | arm | RMSE | ΔRMSE | mean gain | DM t(9) | p | MDE₈₀ | obs/MDE | folds | vs μ̂-cal | p | θ* (eq.7) | eq7 gain | eq7 p | eq7 MDE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WR | μ̂-cal *(control)* | 3.6059 | −0.170 | 1.287 | 2.32 | .046 | 1.746 | 0.74 | 8/10 | — | — | 3.3967 | — | — | — |
| WR | ridge P0 | 3.5325 | −0.244 | 1.805 | 2.44 | **.038** | 2.329 | 0.78 | 7/10 | +0.518 | .180 | 3.3838 | +0.132 | .58 | 0.727 |
| WR | ridge P1 | 3.5072 | −0.269 | 1.982 | 2.89 | **.018** | 2.161 | 0.92 | 9/10 | +0.695 | .083 | 3.3800 | +0.158 | .48 | 0.678 |
| WR | GBT P0 | 3.6780 | −0.098 | 0.744 | 1.21 | .257 | 1.933 | 0.38 | 5/10 | −0.543 | .390 | 3.4132 | −0.071 | .76 | 0.702 |
| WR | GBT P1 | 3.6908 | −0.085 | 0.646 | 1.08 | .309 | 1.883 | 0.34 | 6/10 | −0.641 | .322 | 3.4177 | −0.102 | .63 | 0.646 |
| WR | hier P0 | 3.5343 | −0.242 | 1.801 | 2.35 | **.043** | 2.413 | 0.75 | 8/10 | +0.514 | .200 | 3.3838 | +0.134 | .59 | 0.755 |
| WR | **hier P1** | **3.4893** | −0.287 | 2.112 | 3.34 | **.0086** | 1.989 | 1.06 | 9/10 | +0.826 | .056 | 3.3759 | +0.188 | .37 | 0.632 |
| RB | μ̂-cal *(control)* | 4.1708 | −0.320 | 2.761 | 4.51 | .0015 | 1.925 | 1.43 | 9/10 | — | — | 3.7420 | — | — | — |
| RB | ridge P0 | 4.1542 | −0.337 | 2.878 | 4.02 | **.0030** | 2.251 | 1.28 | 9/10 | +0.116 | .87 | 3.7198 | +0.476 | **.022** | 0.544 |
| RB | ridge P1 | 4.0712 | −0.420 | 3.550 | 5.68 | **.0003** | 1.967 | 1.81 | 10/10 | +0.788 | .27 | 3.7018 | +0.607 | **.0070** | 0.550 |
| RB | GBT P0 | 4.3665 | −0.124 | 1.038 | 1.14 | .285 | 2.870 | 0.36 | 6/10 | −1.723 | .11 | 3.7651 | +0.124 | .65 | 0.841 |
| RB | GBT P1 | 4.2910 | −0.200 | 1.677 | 1.53 | .161 | 3.451 | 0.49 | 6/10 | −1.085 | .37 | 3.7538 | +0.204 | .55 | 1.025 |
| RB | hier P0 | 4.1486 | −0.342 | 2.928 | 4.22 | **.0022** | 2.182 | 1.34 | 10/10 | +0.166 | .81 | 3.7215 | +0.465 | **.023** | 0.532 |
| RB | **hier P1** | **4.0558** | −0.435 | 3.678 | 6.05 | **.0002** | 1.913 | 1.92 | 10/10 | +0.917 | .18 | 3.7069 | +0.572 | **.0040** | 0.470 |

BH at q = .10 over the declared 12-test family: **all four ridge/hier arms reject at both
positions**; neither GBT arm does. Temporal holdout 2022–24 (`results/sectionW1_holdout.csv`):
every ridge/hier arm beats μ̂ at both positions (WR 3.6912 → 3.4213–3.5395; RB 4.1611 →
3.8069–3.9480).

**All four parts of the W1.7 adoption rule are satisfied by ridge_P1 and hier_P1 at both
positions.** What that means is the next three sections.

**Estimator ordering — the pre-registered expectation held.** Linear/hierarchical ≫ trees at
both positions, and the mechanism is visible: GBT's mean training R² is **0.807 (WR) / 0.813
(RB)** against out-of-sample **0.245 / 0.231**, while ridge is 0.315 / 0.307 out of sample and
μ̂ is 0.210 / 0.158. With ~570 rows and ρ_max ≈ .41 the ensemble memorises. Recorded as a
confirmed prediction, not a discovery. The hierarchical arm's inner CV selected κ in
{0.05, 0.2} in most folds — i.e. it pooled WR and RB heavily, which is why it edges the
separate ridges.

## W1.R3 Tier B: the advanced layer adds nothing, and the test is honest about it

Six folds, n = 346 WR / 295 RB. WR ridge_P1 3.4569 vs Tier A's 3.5072 on the same rows; RB
3.9270 vs 4.0712. Against μ̂: WR +2.03, p = .098, **MDE₈₀ = 3.50, obs/MDE = 0.58**; RB +2.92,
p = .0016. Against μ̂-cal: WR +0.99 (p = .088), RB +0.07 (p = .90). On the temporal holdout no
Tier-B arm beats its Tier-A counterpart at either position.

**Verdict: uninformative null at WR (§28.1), and a null with adequate power at RB.** Six
clusters cannot resolve differences of the size at issue; snaps, routes, situational usage and
NGS tracking do not measurably improve a preseason PPG projection over what targets, shares and
team volume already say. `routes_proxy`-derived TPRR/YPRR are among the admitted features and
carry near-zero standardised weight, which is the outcome one should expect from a proxy that
counts blocking TEs and pass-protecting backs.

## W1.R4 The core question, answered — and the answer is no

The headline win over μ̂ is real but it is **not** a win for *inputs*. Three nested comparisons,
all LOSO on the same folds and rows, decompose it (`results/sectionW1_age_design.csv`,
`sectionW1_nested.csv`, `sectionW1_ablation.csv`):

| specification | WR RMSE | RB RMSE |
|---|---|---|
| μ̂ (raw, the incumbent) | 3.7760 | 4.4909 |
| μ̂ recalibrated (OLS on training folds) | 3.6052 | 4.1703 |
| μ̂ + age spline | 3.5471 | 4.0765 |
| μ̂ + experience | 3.5365 | 4.0903 |
| μ̂ + draft pick | 3.6091 | 4.1117 |
| μ̂ + all structure (age, exp, draft) | 3.5478 | **4.0341** |
| **full projection (ridge P1)** | **3.5097** | 4.0691 |
| full projection minus the opportunity block | 3.5848 | 4.0603 |
| opportunity block only (+ μ̂) | 3.4973 | 4.0930 |
| full projection with no age term at all | 3.5196 | 4.0592 |

**The decisive nested test — everything measurable added on top of a calibrated, age-aware μ̂:**

| pos | n | RMSE (μ̂+structure) | RMSE (full) | mean gain | DM t(9) | p | MDE₈₀ | obs/MDE | folds |
|---|---|---|---|---|---|---|---|---|---|
| WR | 568 | 3.5478 | 3.5097 | **+0.249** | 0.88 | .404 | 0.895 | 0.28 | 5/10 |
| RB | 489 | 4.0341 | 4.0691 | **−0.295** | −1.30 | .225 | 0.711 | −0.41 | 3/10 |

At WR the entire usage/volume/efficiency/environment apparatus is worth 0.038 RMSE and cannot
be distinguished from zero (obs/MDE 0.28 — **uninformative**, §28.1). At RB the point estimate
is **negative**: the inputs make it worse, and 3/10 folds improve. §D's null was not an artefact
of one specification. It replicates in a properly specified projection model, on a wider panel,
with reliability-gated features, three estimators and ten folds.

**What the projection is actually doing is fixing two known defects of μ̂:**

1. **μ̂ is uncalibrated.** Its regression slope on realised PPG is **0.667 (WR) / 0.605 (RB)**
   (`results/sectionW1_residuals.csv`) — it is an unshrunk sample mean of a noisy quantity, so
   it over-disperses. Recalibration alone recovers 4.5% (WR) and 7.1% (RB) of RMSE. Inside
   eq. (7) this defect is *partly* already handled: B shrinks μ̂ toward m(ADP). That is why the
   raw-scale gains collapse when moved into θ*.
2. **μ̂ is age-blind.** This is not a new claim — §11 recorded it as "a residual +1.1 μ̂ bias for
   old-career players persists in later folds — an *age* effect (μ̂ has no age curve in it), the
   strongest documented argument for the next iteration." §W1 confirms it out of sample: adding
   an age spline to a calibrated μ̂ moves RB from 4.1703 to 4.0765, which is **the whole of the
   RB result**. Age and experience are near-collinear here and are interchangeable in the fit;
   draft pick is weaker but non-trivial at RB (4.1117).

**Where it lands downstream.** Eq. (7) is where the arm actually acts, and B ∈ [.56, .84] damps
everything:

- **RB: real.** θ* 3.7842 → 3.7018 (ridge_P1), gain +0.607, p = .0070, MDE 0.550, obs/MDE 1.10,
  and hier_P1 +0.572, p = .0040. RB rows also improve on the temporal holdout (3.6441 → 3.6146).
- **WR: uninformative null.** +0.158, p = .48 against MDE 0.678, obs/MDE 0.23. The projection is
  better than μ̂ on the raw scale, but once B has already pulled WRs most of the way to the
  market price, the residual improvement is below what ten folds can resolve. It is not evidence
  that nothing is there; it is evidence the test cannot see it.

**Forecast encompassing** (`results/sectionW1_encompassing.csv`), regressing realised PPG on the
LOSO predictions jointly, HC0 standard errors:

| pos | β(μ̂) alone | β(ŷ) alone | β(μ̂ \| ŷ) | β(ŷ \| μ̂) | β(m̂ \| ŷ) | β(ŷ \| m̂) |
|---|---|---|---|---|---|---|
| WR | 0.667 | 0.999 | 0.254 (.095) | **0.683 (.133)** | 0.586 (.089) | **0.435 (.101)** |
| RB | 0.605 | 0.951 | 0.219 (.096) | **0.663 (.141)** | 0.746 (.071) | **0.272 (.088)** |

ŷ has slope ≈ 1 alone (properly calibrated by construction) and **encompasses μ̂**: μ̂'s
coefficient falls to 0.22–0.25 once ŷ is in. ŷ also retains a positive, several-SE coefficient
alongside the market price m̂. **That last column is deliberately not framed as an edge claim** —
an edge claim requires the §25 FDR family plus a temporal holdout, and the operational version
of it (θ* vs the market) is exactly the eq. (7) test above, which is significant at RB and
uninformative at WR.

## W1.R5 L2.1 — the age curve applied (`results/sectionW1_age.csv`)

**A1, §H's era-3 relative curve applied multiplicatively**, ŷ·f₃(age)/f₃(age−1) (mean ratio
0.991 WR / 0.977 RB; minimum 0.693 / 0.892):

| pos | base arm | RMSE base | RMSE ×f₃ | mean gain | p | folds | θ* base | θ* adj | eq7 gain | eq7 p |
|---|---|---|---|---|---|---|---|---|---|---|
| WR | μ̂ | 3.7760 | **3.6186** | +1.176 | **.0079** | 9/10 | 3.4036 | 3.3756 | +0.194 | .083 |
| WR | μ̂-cal | 3.6059 | 3.5466 | +0.442 | .134 | 7/10 | 3.3967 | 3.3893 | +0.055 | .56 |
| WR | ridge P1 | 3.5072 | 3.5912 | **−0.582** | .041 | **1/10** | 3.3800 | 3.3933 | −0.087 | .33 |
| RB | μ̂ | 4.4909 | **4.3415** | +1.312 | **.0044** | 9/10 | 3.7842 | 3.7368 | +0.355 | **.010** |
| RB | μ̂-cal | 4.1708 | 4.0774 | +0.752 | .121 | 8/10 | 3.7420 | 3.7145 | +0.201 | .143 |
| RB | ridge P1 | 4.0712 | 4.1188 | −0.410 | .319 | 3/10 | 3.7018 | 3.6975 | +0.028 | .82 |

Three findings, in order of importance:

1. **Applied to μ̂, §H's curve is a large, significant, 9/10-fold improvement at both positions**
   — +1.18 WR / +1.31 RB, and it survives into eq. (7) at RB (+0.355, p = .010). This is the
   §11 age defect being repaired by an externally estimated curve that never saw this panel.
   It is a *structural correction*, not an edge claim; §H5's null (the market prices age
   correctly) is untouched, because the comparison here is against μ̂, not against ADP.
2. **Applied to the projection, it is significantly harmful at WR (1/10 folds, p = .041) and
   negative at RB.** Double counting: the projection already carries an age spline and an
   experience term. Correctly diagnosed, not tuned away.
3. **Estimating age inside the panel dominates importing §H's curve.** μ̂+age-spline reaches
   3.5471 / 4.0765 versus μ̂×f₃'s 3.6186 / 4.3415, and adding §H's curve *as a feature* to the
   full projection changes nothing (WR 3.5106 vs 3.5072; RB 4.0777 vs 4.0712). The externally
   estimated curve is a good correction to a curve-free estimator and a redundant one to a
   model that fits its own.

**L2.1 verdict: ADOPT the age correction, in the form of an age term inside the projection
(A2), not as an external multiplier on top of it (A1).** If WS2 keeps μ̂ as the data arm rather
than the projection, then A1 — μ̂×f₃(age)/f₃(age−1) — is the adoptable form and it is worth
+0.355 PPG² at RB inside eq. (7) on its own.

## W1.R6 L2.2 — availability (`results/sectionW1_availability.csv`, `sectionW1_ppsw.csv`)

**Stage 1 — does prior availability predict next-season availability?** Ridge on
{recency-weighted availability, last-season availability, career availability, n prior seasons,
G_last, weighted G, gap since last season}, LOSO, versus the training-fold constant:

| pos | n | RMSE const | RMSE model | out-of-sample R² | r(pred, actual) | mean gain | p | MDE₈₀ | obs/MDE | folds |
|---|---|---|---|---|---|---|---|---|---|---|
| WR | 569 | 0.1873 | 0.1836 | **0.039** | .192 | +0.0014 | **.042** | 0.0019 | 0.75 | 8/10 |
| RB | 490 | 0.2089 | 0.2070 | **0.018** | .118 | +0.0007 | .205 | 0.0016 | 0.43 | 8/10 |

**WR clears the pre-registered bar; RB does not.** Availability is a stable *trait* (§A, ICC .36)
and still a nearly unforecastable *outcome*: 3.9% of next-season availability variance at WR,
1.8% at RB, against a residual SD of 0.18–0.21 on a [0,1] scale. §A and §W1.R6 are not in
tension — an ICC of .36 on a repeated measure is entirely compatible with an R² of .04 on a
single future draw, because most of the trait's variance is swamped by the season-specific
injury draw.

**Stage 2 — points per scheduled week**, against the incumbent $\hat\mu^{\text{ppsw}}$
(recency-weighted mean of prior per-scheduled-week rates), and against its **calibrated**
version, which is the honest control for exactly the reason W1.R4 gives:

| pos | arm | RMSE | vs raw incumbent (gain, p) | vs **calibrated** incumbent (gain, p, MDE) |
|---|---|---|---|---|
| WR | incumbent calibrated | 4.0952 | +2.03, .012 | — |
| WR | μ̂ × avail (naive, no model) | 4.3720 | −0.33, .13 | −2.36, **.0085** |
| WR | μ̂ × âvail | 4.1789 | +1.34, .11 | −0.68, .26 |
| WR | ridge_P1 × âvail | 4.1022 | +1.98, .059 | −0.05, .90 |
| WR | ridge_P1 fitted directly on ppsw | 4.1120 | +1.89, .054 | −0.14, .80 |
| RB | incumbent calibrated | 4.6747 | +3.87, .0058 | — |
| RB | μ̂ × âvail | 4.7702 | +3.05, .0021 | −0.83, .34 |
| RB | ridge_P1 × âvail | 4.5460 | +5.02, .0018 | +1.14, .045 |
| RB | ridge_P1 fitted directly on ppsw | 4.5647 | +4.87, .0020 | +1.00, .036 |

**L2.2 verdict: REJECT the availability model as an input; the second target is available but
carries no validated availability content.** At WR, where stage 1 passed, nothing beats a
recalibrated per-scheduled-week mean. At RB the two positive results (+1.14, +1.00) are on the
position whose **stage-1 gate failed**, so under the pre-registered two-stage rule they may not
be used; and they are in any case attributable to the projection, since ridge_P1 already
contains `avail_wtd` and `G_last` as features and μ̂ × âvail does not beat the control at either
position. The naive multiplier μ̂ × (prior availability) is **significantly worse** than the
control at WR (−2.36, p = .0085), which is an independent quantitative vindication of the
owner's refusal of a post-hoc expected-games multiplier.

**What is delivered anyway, without an availability claim:** `ppsw` is a well-defined second
target and a projection fitted *directly* on it (RMSE 4.1120 WR / 4.5647 RB) is the clean way
to produce it, because it needs no multiplier at all. WS2's replacement-units defect (L5.1) is a
question about which basis the *whole board* is on; W1 supplies both bases and takes no view.

## W1.R7 The §P interaction as precision, pre-registered and rejected (`sectionW1_precision.csv`)

W1.10 pre-registered the §43.5 D1/D2 variants. Harness check first: the incumbent-arm numbers
reproduce §43.5 exactly (WR D1 +0.260 p = .117; WR D2 +0.197 p = .099; RB D1 +0.370 p = .092;
RB D2 +0.235 p = .037).

| pos | arm | variant | n treated | RMSE base | RMSE var | gain | p | MDE₈₀ | folds | holdout | BH |
|---|---|---|---|---|---|---|---|---|---|---|---|
| WR | μ̂ | D1 anchor G<12 | 180 | 3.4011 | 3.3621 | +0.260 | .117 | 0.472 | 5/10 | survives | no |
| WR | μ̂ | D2 B′ | 180 | 3.4011 | 3.3718 | +0.197 | .099 | 0.337 | 7/10 | survives | no |
| WR | ridge P1 | D1 | 180 | 3.3800 | 3.3801 | −0.003 | .96 | 0.154 | 5/10 | survives | no |
| WR | ridge P1 | D2 | 180 | 3.3800 | 3.3788 | +0.009 | .73 | 0.079 | 6/10 | survives | no |
| RB | μ̂ | D1 | 214 | 3.7909 | 3.7437 | +0.370 | .092 | 0.616 | 7/10 | survives | no |
| RB | μ̂ | D2 | 214 | 3.7909 | 3.7606 | +0.235 | **.037** | 0.302 | 8/10 | survives | no |
| RB | ridge P1 | D1 | 214 | 3.7209 | 3.7091 | +0.091 | .33 | 0.276 | 6/10 | survives | no |
| RB | ridge P1 | D2 | 214 | 3.7209 | 3.7153 | +0.043 | .28 | 0.117 | 7/10 | survives | no |

**REJECTED, and the reason is a finding.** Under BH at q = .10 over the declared 8-test family
nothing survives (the smallest p, RB D2 at .037, needs .0125). More informative than the
multiplicity failure: **the adjustment goes to zero once the projection replaces μ̂** — +0.235 →
+0.043 at RB, +0.197 → +0.009 at WR. The projection carries `G_last`, `G_wtd` and `avail_wtd`
as features and discounts a short prior season *in the estimate*, so there is nothing left for a
precision correction to do. §43.5 concluded "what §P's finding implies is a change to B, not to
μ̂." The sharper statement after §W1 is: **it implies a change to B only if the data arm is μ̂.**
The §P interaction is a symptom of an estimator that ignores how many games it was computed
from, not a fact about precision that any estimator must encode.

## W1.R8 Residuals and declared sensitivities (`results/sectionW1_residuals.csv`)

| pos | arm | mean | SD | skew | kurtosis | Shapiro p | slope of \|e\| on fitted | calibration slope |
|---|---|---|---|---|---|---|---|---|
| WR | μ̂ | −0.409 | 3.757 | −0.034 | 0.396 | .18 | +0.011 | **0.667** |
| WR | μ̂-cal | −0.005 | 3.609 | 0.011 | −0.013 | .85 | +0.017 | 0.977 |
| WR | ridge P1 | −0.002 | 3.510 | 0.145 | −0.115 | .37 | −0.009 | 0.999 |
| RB | μ̂ | −0.308 | 4.485 | 0.426 | 0.721 | .0006 | +0.011 | **0.605** |
| RB | μ̂-cal | 0.008 | 4.175 | 0.476 | 0.368 | .0002 | +0.009 | 0.984 |
| RB | ridge P1 | −0.006 | 4.075 | 0.416 | 0.027 | .0003 | +0.049 | 0.951 |

WR residuals are effectively Gaussian and homoskedastic. RB residuals stay right-skewed (0.42)
with mild variance growth in the fitted value — expected for PPR and unchanged from §2's
diagnosis. **log(1+Y) sensitivity, run because of that skew:** fitting and predicting on
log1p and back-transforming is *worse* on the natural scale at both positions — WR 3.5446 vs
3.5163, RB 4.1456 vs 4.0758, DM gains −0.215 (p = .55) and −0.548 (p = .24). Identity link
retained, as §2 concluded for the same reason.

**Elastic net** (`results/sectionW1_elasticnet.csv`), the sensitivity declared in W1.5, over
l1_ratio ∈ {0.1, 0.5, 0.9} with α on an inner grouped CV:

| pos | l1 | RMSE | gain vs μ̂ | p | MDE₈₀ | gain vs μ̂-cal | p |
|---|---|---|---|---|---|---|---|
| WR | 0.1 | 3.5116 | +1.953 | .020 | 2.186 | +0.666 | .104 |
| WR | 0.5 | 3.4879 | +2.116 | .011 | 2.102 | +0.829 | **.034** |
| WR | 0.9 | 3.4870 | +2.126 | .011 | 2.088 | +0.839 | **.025** |
| RB | 0.1 | 4.0657 | +3.596 | .0003 | 1.942 | +0.834 | .230 |
| RB | 0.5 | 4.0722 | +3.549 | .0002 | 1.840 | +0.788 | .230 |
| RB | 0.9 | 4.0843 | +3.448 | .0001 | 1.682 | +0.687 | .209 |

Indistinguishable from ridge (WR 3.5072, RB 4.0712) at every mixing weight — the sparser fits
are marginally better at WR and marginally worse at RB, all inside their MDEs. The choice of
regulariser is not load-bearing, which is itself consistent with W1.R4: the signal is in three
or four terms, not in a subtle weighting of thirty.

**Gate sensitivity.** Switching the reliability gate off (admitting yards/target, YAC/rec,
TD/target, EPA/target, ypc) leaves WR essentially unchanged (ridge_P1 3.5059 vs 3.5072) and
makes RB slightly *better* (4.0615 vs 4.0712) — a difference far inside its MDE. The gate is
not doing damage and it is not doing much work either, which is what one expects when the
rejected stats are noise: ridge sets their coefficients near zero anyway. The gate stays,
because "it would have been fine this time" is not a reason to remove a screen.

## W1.R9 Fitted model and coefficients (`results/sectionW1_coefficients.csv`)

Standardised ridge coefficients, mean over the 10 LOSO folds, median α in the caption. Positive
= raises projected PPG.

**WR, P1 (α = 178):** μ̂ **+0.665**, target share (full) +0.334, WOPR +0.301, touches/g +0.296,
targets/g +0.292, **age spline −0.286**, prior-team pass yards/g +0.249, usage index +0.249,
RACR (wins.) +0.242, air-yards share +0.223, **experience −0.188**, weighted games −0.139,
prior-team TD/g +0.134.

**RB, P1 (α = 18):** μ̂ **+1.911**, **log draft pick −0.805**, air-yards share +0.757,
**experience −0.531**, G_last +0.510, **age spline −0.492**, aDOT −0.453, catch rate +0.289,
carries/g +0.280, prior-team targets/g +0.256, weighted games −0.205, WOPR +0.202.

Two things to read here. First, the RB coefficient on μ̂ exceeding 1 with a low α is the model
*undoing* μ̂'s 0.605 calibration slope and then re-shrinking through the penalty — the
calibration story in the coefficients. Second, `log_draft_pick` at −0.805 is the largest
non-μ̂ term at RB, which together with age and experience is the structural block that W1.R4
shows carries the entire RB result.

## W1.R10 Deliverable: 2026 projections

`results/sectionW1_projection_2026.csv` — 80 WR / 61 RB from the 2026-08-24 FFC pool with at
least one prior season, with `mu_hat`, `mu_cal`, `proj_ridge_P0`, `proj_ridge_P1`, age,
experience, G_last and weighted availability. Rows without prior history are omitted by design:
eq. (7) gives them B = 1 and the market arm owns them. Corr(projection, μ̂) = .92 WR / .88 RB;
mean |projection − calibrated μ̂| = 0.84 WR / 1.11 RB, i.e. the projection moves players about
one point per game relative to a recalibrated mean, and the largest moves are the oldest
players — which is the age term, working as W1.R4 says it works.

**Caveat for WS2, stated explicitly:** `mu_hat` in that file is computed over all prior games
rather than under the §0 inclusion rule, so it differs from the pipeline's μ̂ by a few
hundredths. Use the pipeline's μ̂; the projection columns are the deliverable.

---

# W1 VERDICT

**On the question the round was built to answer — REJECT.** Expected PPG cannot be projected
from measurable inputs better than it can be summarised from past output. On top of a
calibrated, age-aware μ̂, the entire usage/volume/efficiency/environment apparatus is worth
**+0.038 RMSE at WR (p = .40, obs/MDE 0.28 — uninformative)** and **−0.035 at RB (p = .23, wrong
sign)**. Tier B's advanced layer adds nothing on top of that. §D's null replicates.

**On the arm that was pre-registered — ADOPT, in a specific and limited sense.** ridge_P1 and
hier_P1 satisfy all four parts of the W1.7 rule at both positions: lower RMSE, DM p < .05, BH
survival in the 12-test family, and a temporal holdout win. Inside eq. (7), which is where it
acts, the improvement is **real at RB (+0.607 PPG², p = .0070, obs/MDE 1.10, holdout 3.6441 →
3.6146)** and an **uninformative null at WR (+0.158, p = .48, MDE 0.678)**.

**The honest reading, and the recommendation to WS2.** What is being adopted is not a projection
engine; it is a **recalibrated, age-aware μ̂ wearing a projection engine's clothes.** Two
consequences follow:

1. WS2 should take the L1 arm as adopted **at RB**, where it clears every screen and moves the
   downstream posterior. At WR it should be adopted only on the strength of the raw-scale and
   BH evidence, with the eq. (7) null recorded beside it as uninformative rather than
   supportive.
2. The parsimonious alternative — **calibrate μ̂ and add an age term, and stop** — reaches
   3.5478 (WR) / **4.0341 (RB)**, the best RB number in the entire round, with three parameters
   instead of thirty. That specification was *not* pre-registered; it was found by decomposing a
   result. Per this project's own standing rule it is therefore **not adopted here**, and is
   recorded as the round-10 pre-registration candidate, exactly as §43.5 recorded D1/D2.

**Three findings that outlive the verdict:**
- The historical ADP panel's `team` field is an end-of-season label. Anything treating it as
  preseason information is leaking (worth 0.02–0.05 RMSE here; possibly more elsewhere).
- §11's flagged age defect in μ̂ is confirmed out of sample and is now measured: it is the
  single largest correctable error in the incumbent data arm, and at RB it is essentially the
  *only* one.
- §P's ≥12-games interaction is not a fact about precision. It vanishes when the data arm knows
  how many games it was computed from, which means it was a symptom of μ̂'s blindness to G_last
  rather than a structural property of the posterior.

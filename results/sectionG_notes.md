# §G lab notes — RB universe full refit, and the August ADP refresh

Executed 2026-08-09 against `EDA_PLAN4.md` (pre-registered same day, before any round-4
fitting). Rules unchanged: fit as specified, no tuning toward expectations, no named-player
anchors inside any pipeline, anomalies get chased, adoption only on the pre-specified LOSO
evidence. Scripts: `20_wr_board_refresh_aug.py`, `21_wr_refresh_downstream_aug.py`,
`22_rb_g1_g2.py`, `23_rb_g3_market_prior.py`, `24_rb_g4_g5_g6.py`.

---

# Part 1 — Reproduction check, then the August ADP refresh

## 1.0 Reproduction of the frozen v3 board (done first, before anything else)

Scripts `01, 06, 07, 08, 09, 10, 11, 17, 18` were re-run against the **July** inputs. All
fourteen round-1 result files plus the four round-2/3 files compared **byte-identical**
(`cmp`), including `valuation_2026_final.csv`, `loso_scorecard.csv`, `loso_scorecard3.csv`,
`edge_teammate.csv`, `teammate_coherence_2026.csv`. `valuation_2026_v3.csv`'s `V_final_v3`
column matches the re-run `V_final` to `max |diff| = 0.0`, and `rank_v3 == rank_final`.

**Everything reported below as movement is therefore attributable to the ADP refresh and to
nothing else.** No July raw file and no v1/v2/v3 result CSV was overwritten; all refreshed
outputs are new dated files.

## 1.1 What the refresh can and cannot touch

The July raw pull is `n = 1,737` drafts (Jul 6–13); the August pull is `n = 5,187` drafts
(Aug 1–8). Only two things in the pipeline read the 2026 board: the modeling universe, and
the ADP values that enter `m_iso(ADP)`. Every hyperparameter is estimated on data the
refresh does not touch:

| input | fit on | changed by refresh? |
|---|---|---|
| `m_iso(·)`, `tau^2(tier)` | historical 2015–2024 ADP panel (script 07) | no — asserted by hash |
| `sigma^2(tier)` | all WRs' 2014–2025 game logs (script 06) | no — asserted by hash |
| `mu_hat`, `n_eff` | each board WR's own 2014–2025 game logs | only for the 2 new members |
| edge terms | none survived FDR + holdout (script 09) | no |

Verified empirically: for all 28 continuing players the decomposition column
`d_from_muhat` is **exactly 0.000** — μ̂ and n_eff are unchanged, as they must be. The whole
board delta is the ADP channel.

## 1.2 Board composition

**In:** DJ Moore (BUF), Courtland Sutton (DEN). **Out:** DK Metcalf, Christian Watson.
30/30 matched to `gsis_id`. §G1-analogue exclusions on the refreshed universe: 2,119 REG
rows, 36 excluded at targets ≤ 1 (1.70%), mean PPR of excluded 1.88 — same structure as
round 1's 1.9% / 1.9 PPR.

Board tiers 2026: 27 vet, 3 soph (McMillan, Egbuka, Burden), 0 rookies. FFC-August team
field vs the (July-16) Sleeper dump: **0/30 disagreements**, so §G0's team source is
unaffected.

## 1.3 Result — the refreshed board

Full file `results/valuation_2026_wr_20260809.csv`; per-player decomposition
`results/wr_board_refresh_delta_20260809.csv`; refreshed consistency table
`results/consistency_table_20260809.csv`. Top of the board and the movers:

| θ* rank (Aug) | WR | ADP Jul → Aug | θ* Jul → Aug | Δθ* | Δrank |
|---|---|---|---|---|---|
| 1 | Puka Nacua | 2.8 → 2.9 | 20.20 → 20.20 | 0.00 | 0 |
| 2 | Ja'Marr Chase | 3.9 → 3.9 | 19.84 → 19.84 | 0.00 | 0 |
| 3 | Amon-Ra St. Brown | 7.6 → 7.0 | 18.63 → 18.88 | +0.25 | 0 |
| 4 | Jaxon Smith-Njigba | 6.1 → 5.9 | 18.09 → 18.09 | 0.00 | 0 |
| **5** | **Rashee Rice** | **27.3 → 11.4** | 16.36 → **17.64** | **+1.28** | **+2** |
| 6 | CeeDee Lamb | 10.6 → 11.3 | 17.85 → 17.56 | −0.29 | −1 |
| **7** | **Drake London** | **12.6 → 10.6** | 16.13 → **17.01** | **+0.88** | **+2** |
| 9 | Justin Jefferson | 10.0 → 13.3 | 16.93 → **16.07** | **−0.87** | **−3** |
| 14 | Malik Nabers | 43.6 → 33.9 | 14.44 → 14.85 | +0.41 | +3 |
| 20 | Emeka Egbuka | 45.5 → 33.7 | 13.50 → 14.06 | +0.56 | +3 |
| 25 | DJ Moore | — → 51.3 | — → 13.19 | new | new |
| 26 | Mike Evans | 51.2 → 60.0 | 13.79 → **13.04** | **−0.76** | −6 |
| 27 | Rome Odunze | 54.9 → 49.7 | 12.11 → 12.96 | +0.84 | +2 |
| 28 | Courtland Sutton | — → 59.8 | — → 12.80 | new | new |
| 29 | Luther Burden III | 45.2 → 53.4 | 13.21 → 12.60 | −0.60 | −3 |

RMS Δθ* over the 28 continuing players = **0.469 PPG**, max |Δrank| = 6 (Evans), Spearman
between the July and August θ* on those 28 = **0.977**. The board is stable; the refresh is
not a re-ranking event.

The two entrants both land in the bottom third and both are *market-led*: DJ Moore θ* 13.19
vs m(ADP) 13.89 (μ̂ = 12.27 pulls him down), Sutton θ* 12.80 vs m(ADP) 12.54 (μ̂ = 13.13
pulls him up). Neither displaces anyone above θ* rank 25.

**Rice is the story of the refresh.** His ADP moved 15.9 slots (27.3 → 11.4), which is
+2.06 PPG of prior mean; at B = 0.622 that is +1.28 PPG of posterior. He was round 1's
biggest riser vs the market (+8 ranks); the August market has now come to him, and his
Δ-vs-market collapses from +8 to +2. **The single largest disagreement in the v3 board has
been priced away by the market in three weeks.** Adams' +8 has not: he sits at θ* rank 11 vs
ADP rank 20 (+9), essentially unchanged.

### Anomaly chased: 12 of 28 players moved **exactly** zero despite ADP moving

`m_iso` is a step function with **18 unique levels** over ADP 1.4–75, boundaries at ADP
1.4 / 2.6 / 5.6 / 7.6 / 11.4 / 12.1 / 12.9 / 18.9 / 26.2 / 32.6 / 34.5 / 42.1 / 51.9 / 53.9 /
55.2 / 68.1 / 75. Plateaus are 1–13 ADP slots wide, so a player whose ADP moves within one
plateau gets **identically** the same prior mean and hence the same posterior. Mean |ADP
move| among the 12 zero-Δ players is 1.66 slots vs 6.24 among the 16 that moved — the split
is exactly plateau-width. This is a property of the isotonic estimator, not a bug: the board
is robust to sub-plateau ADP noise and jumps discretely at boundaries.

The corollary is worth stating because it is uncomfortable: **Tee Higgins fell 12.8 ADP
slots (26.4 → 39.2) and lost only 0.32 PPG**, because the isotonic curve is nearly flat
across that stretch (14.882 → 14.318 over 16 slots). Historically, ADP 26 and ADP 39 have
been worth almost the same PPG for a WR. The market prior carries far less information in
the WR2/WR3 range than the ADP ordering suggests.

## 1.4 Downstream layers (`21_wr_refresh_downstream_aug.py`)

§E and §F's *tests* are estimated on the historical panel and are unchanged by a 2026 ADP
refresh (both reproduced byte-identically). §E arm (vii) remains **not adopted**
(DM vs (ii) p = .439), so the board value is the unadjusted posterior. The §F PPG↔targets
map (a = −1.3089, b = 1.9423) and the historical top-2 WR TS benchmark (n = 384, p90 = .456,
p95 = .476) both reproduce exactly.

The §F 2026 *duo* table does change, because membership changed — **a sixth duo appears,
DEN Waddle + Sutton**:

| team | duo | implied TS sum (Aug) | pct of realized top-2 sums |
|---|---|---|---|
| LA | Nacua + Adams | .560 | 99.7 |
| DET | St. Brown + J. Williams | .525 | 99.5 |
| CIN | Chase + Higgins | .507 | 98.4 |
| DAL | Lamb + Pickens | .495 | 97.1 |
| CHI | Odunze + Burden | .430 | 82.8 |
| **DEN** | **Waddle + Sutton** | **.410** | **73.7** |

The Rams pair is unchanged at .560 and remains the only duo above the like-for-like
(implied-vs-implied, 94.9th pct) reference from the frozen §F run. The new DEN duo is
unremarkable. §F2 was a full null historically, so per the fixed decision rule no
constraint arm is run and none of this moves a valuation.

---

# Part 2 — §G RB universe, full refit

**Nothing is reused from WR.** Every variance component, reliability weight, threshold and
market curve below is estimated on RB data.

## §G1 Inclusion rule

Regular season; drop player-games with **touches = carries + targets ≤ 1**.

| | WR (§0) | RB (§G1) |
|---|---|---|
| population | all WRs 2014–2025 REG | all RBs 2014–2025 REG |
| player-games | — | 17,863 |
| excluded | 1.9% | **3,072 = 17.20%** |
| mean PPR of excluded | 1.9 | **0.454** |

**This is the first RB/WR structural difference and it is an order of magnitude.** Nearly
one in five RB game rows is a near-zero-touch appearance worth 0.45 PPR — the committee /
RB3 / special-teams-only structure of the position. On the 30-man board itself only 0.99%
of rows are excluded (mean PPR 0.29), because the board is top-of-market. All RB rates
below are therefore "given participation" in a much stronger sense than for WR, and the
missed-participation mass is correspondingly larger; that is exactly why §G5 matters.

**Boom/bust thresholds, frozen before any fit** from the *positional* distribution of the
14,791 qualified RB player-games (not the board, so board selection cannot enter): pooled
p75 = **13.8** and p25 = **3.2** PPR (median 7.6, mean 9.54). WR used 20 / 8.

**Relevance gate for the location-scale sample**, also fixed before fitting: WR §3 gated at
mean targets/game ≥ 3, which retains 82.5% of WR player-seasons with ≥1 included game. The
retention-matched RB value on the touches/game distribution is 4.35, rounded to the nearest
integer as WR's was, giving **touches/game ≥ 4** (retains 84.9%). A ≥ 8 touches/game
"feature back" cut is reported as a sensitivity, not as an alternative headline.

## §G2 Variance components

### G2a Per-player consistency (`results/consistency_table_rb.csv`)

Script 01's `build_table` is imported and re-used with the RB thresholds patched in, so the
estimator is literally the round-1 estimator. 28 of the 30 board RBs have NFL rows.

| | WR board | RB board |
|---|---|---|
| median pooled σ̂_W | 7.75 | **7.99** |
| median naive v (n ≥ 4) | 5.46 | **9.31** |
| median untruncated τ̂²_B (n ≥ 4) | −0.08 | **+5.95** |
| players with n ≥ 4 seasons | 20 | 17 |
| of those, untruncated τ̂²_B < 0 | 10 (50%) | **4 (24%)** |

**Direct answer to the pre-registered anomaly question: τ̂²_B does *not* truncate at zero for
most RBs.** The WR headline finding — "for most established WRs, movement in season means is
nearly all averaging noise" — **does not transfer**. Game-level scatter is essentially
identical across the two positions (σ̂_W ≈ 7.8 vs 8.0), but the *naive* between-season
variance is 70% larger for RB and the eq.-(3) correction does not eat it: RB season levels
genuinely move. Largest τ̂²_B are Kyren Williams 52.1, McCaffrey 29.8, Henry 23.6, Chase
Brown 22.2, Rico Dowdle 22.4 — i.e. backfield-share reallocations and career arcs, the RB
analogue of what §1 called "arcs" for Adams/Collins, but far more common.

EB Beta moment fits on the board: boom (α, β) = (5.251, 5.840), bust (0.775, 11.058).
(WR: boom (6.11, 20.20), bust (4.31, 13.23).) Note the boom prior mean is ~0.47 because the
threshold is a *positional* p75 and the board is the top 30 — by construction these players
boom far more often than a random qualified RB game.

### G2b Crossed decomposition `Y_isg = μ + a_i + b_is + c_{t(i,s),s} + ε` (`variance_components_rb.csv`)

Headline spec, 2021–2025, exclusions on — the identical spec ladder script 02 ran for WR.

| | σ̂²_P (player) | σ̂²_S (player×season) | σ̂²_T (team×season) | σ̂²_G (game) | ρ_max (G=17) |
|---|---|---|---|---|---|
| **WR REML** | 5.48 | 2.48 | 1.20 | 69.93 | 0.413 |
| **RB REML** | **7.06** | **5.78** | **0.00** (boundary) | **63.50** | **0.426** |
| WR MoM | 5.33 | 3.18 | +0.58 | 69.82 | — |
| RB MoM | 6.67 | 7.70 | **−1.68** | 63.36 | — |

Share of *season-mean* variance:

| | stable skill | next-year context (S+T) | irreducible σ²_G/17 |
|---|---|---|---|
| WR (total 13.27) | 41.3% | 27.7% | 31.0% |
| **RB (total 16.58)** | **42.6%** | **34.9%** | **22.5%** |

Readings:

1. **RB season means are more variable overall** (16.58 vs 13.27 PPG²) and a *larger share*
   of that is next-year context: σ̂²_S is **2.3×** the WR value. Committee/role churn is a
   real, large variance component, not a story.
2. **Single-game noise is lower for RB in absolute terms** (63.5 vs 69.9) — carries are a
   floor mechanism; WR scoring is more big-play driven.
3. ρ_max is **essentially the same for both positions (0.426 vs 0.413)**, stable 0.39–0.47
   across the sensitivity ladder except log1p (0.33). The predictability *ceiling* does not
   differ by position; what differs is where the unpredictable part lives.
4. Residual skew +0.79 (WR +0.78) — identical right-skew; log1p over-corrects to −1.49 (WR
   −0.99), so the identity scale is kept, exactly as in round 1.

**Anomaly chased: σ̂²_T is negative for RB.** REML truncates it at the boundary (0.0,
"MLE may be on the boundary" warning fired, `Random effects covariance is singular`); MoM,
which is not constrained, returns **−1.68**. Diagnosis, by direct measurement of teammate
cross-products on the 6 team-seasons carrying two board RBs (DAL 2023, DET 2023/24/25, JAX
2025, NE 2025):

| teammate pair type | n pairs | mean cross-product | correlation |
|---|---|---|---|
| same team-season, **same week** | 88 | **−2.463** | **−0.075** |
| same team-season, different week | 1,414 | −1.629 | −0.036 |

The sign is consistently negative and is *more* negative contemporaneously. This is the
backfield constraint: two RBs on one team split a roughly fixed carry pool, so their game
scores are competitively, not commonly, determined. A variance-component model cannot
represent that — a shared random effect is necessarily a *positive* intraclass covariance —
so the parameter hits the boundary. **σ²_T is not merely weakly identified for RB (as it was
for WR, 20 team-seasons); it has the wrong sign for the model.** Honest resolution: with 6
team-seasons and 88 same-week pairs the estimate is not distinguishable from zero at this
resolution, so the model is fit as pre-specified and what is identified is the **sum**
σ²_S + σ²_T = 5.78 (REML) / 6.02 (MoM). The negative sign is recorded as the finding, and
it is the *right* sign for the §J off-diagonal: RB teammates belong in Σ with a negative
covariance, and this is the measured basis for it.

**Anomaly chased: adjacent-season persistence.** Same estimator as §2's WR check, 2021–2025:

| | lag-1 same-player season-mean covariance | Var(season mean) | ratio |
|---|---|---|---|
| WR board | 8.174 | 14.341 | **0.570** |
| RB board | 4.318 | 17.654 | **0.245** |

An RB's season mean carries **less than half** the WR's information about his next season
mean. This single number is the mechanism behind the §G6 verdict below.

### G2c Heteroskedasticity by experience tier (`heteroskedasticity_rb.csv`, `sigma2_by_tier_rb.csv`)

Gamma GLM (log link, dispersion 2), SEs clustered by player-season; 14,028 games,
1,408 player-seasons (265 rookie / 237 soph / 906 vet).

| tier | RB multiplier vs vet [95% CI] | WR multiplier |
|---|---|---|
| rookie | **0.892** [0.793, 1.004] | 0.844 |
| sophomore | **1.105** [0.990, 1.234] | 0.921 |

Neither marginal CI excludes 1 on its own (RB rookie's upper limit is 1.004, sophomore's
lower limit is 0.990); the evidence is in the **joint** test and in the sign pattern.

Joint Wald χ²(2) = 8.65, p = .013. **The RB pattern is not the WR pattern**: WR had both
young tiers *below* vet; RB has rookies below and **sophomores above**. σ̂²(tier) =
**35.37 / 43.82 / 39.64** (rookie / soph / vet) vs WR's 36.42 / 39.73 / 43.14 — the RB
ordering is non-monotone in experience and the *sophomore* cell is the noisy one.

Chased with the WR mechanism (a level effect): variance scales as σ² ∝ μ^**1.387**
(se .036) for RB, almost exactly WR's μ^1.41. But controlling for level does **not** collapse
the RB tier effects the way it did for WR: rookie → 1.080 (p = .156, now *above* vet) and
sophomore → **1.129 (p = .022)**. So RB sophomores are more volatile per unit of scoring
level, and it survives the level control. The ≥ 8 touches/game sensitivity reproduces it
(soph 1.128 raw, 1.134 level-adjusted, p = .041). Face-valid mechanism: year 2 is when an
RB's backfield share is actively contested — but that is interpretation, not tested here.
Spec unchanged; σ̂²(tier) used exactly as estimated.

## §G3 Market prior (`market_prior_rb.csv`, `tier_variances_rb.csv`, `market_prior_iso_knots_rb.csv`)

Panel: top-30 RBs by FFC PPR 12-team ADP each year 2015–2024, joined to realized same-season
PPG under §G1. **300/300 rows matched** after one documented data-quality fix to the
script-07 normalizer: FFC writes initials unspaced ("CJ Anderson") where nflverse writes
them spaced ("C.J. Anderson"), which the normalizer turned into `cj anderson` vs
`c j anderson`. Added a symmetric whitespace-stripped comparison; it caught exactly 4 rows
(C.J. Anderson ×3, T.J. Yeldon ×1). No player is named in the pipeline logic.

**14 rows fall below the pre-registered ≥ 4-game fit floor** (vs 9 for WR), including three
0-game seasons. n = 286 in the fit.

Tier composition — **the structural difference that matters most for §G4**:

| | rookie | soph | vet |
|---|---|---|---|
| WR panel rows (in_fit) | **4** | 36 | 251 |
| **RB panel rows (in_fit)** | **25** | 46 | 215 |

RB boards routinely carry rookies; WR boards almost never did. The RB rookie cell is
**actually identified**, which is why §G4's full-shrinkage-to-market for the two 2026
rookies rests on an estimated prior variance rather than the WR situation's n = 4 fiction.

Fits: OLS `PPG = 22.24 − 2.587·log(ADP)` (se 0.274, R² = 0.312; year-FE slope −2.613).
Isotonic monotone-decreasing in log ADP: **16 unique levels, 19.40 PPG at the top of the
board down to 10.41 at ADP 80**. In-sample RMSE 3.617 (isotonic) vs 3.747 (OLS).

τ̂²(tier) = Var(realized − m̂_iso), bootstrap 95% CIs (4,000 reps):

| tier | n | τ̂²_iso | 95% CI | WR |
|---|---|---|---|---|
| rookie | 25 | **11.68** | [6.23, 16.40] | 24.54 (n = 4) |
| soph | 46 | **14.66** | [10.24, 18.59] | 7.85 |
| vet | 215 | **12.99** | [10.59, 15.48] | 11.25 |

The pre-registered ordering rookie > soph > vet **fails again, differently**: for RB the
three tiers are statistically indistinguishable (~12–15) with sophomores nominally highest.
Used exactly as estimated; no ordering imposed. Notably the RB rookie τ² is *lower* than the
vet τ² and less than half WR's rookie value — with a real n behind it, the market prices RB
rookies about as accurately as it prices RB veterans.

**Anomaly chased: is the raw ADP→PPG relation monotone?** Decile means:

| ADP decile mean | 3.2 | 8.1 | 13.7 | 20.2 | 28.1 | **34.6** | 42.8 | 51.2 | 59.7 | 71.2 |
|---|---|---|---|---|---|---|---|---|---|---|
| realized PPG | 19.12 | 17.13 | 16.40 | 15.02 | **13.44** | **13.82** | 11.91 | 11.50 | 11.37 | 11.04 |

There is a genuine **reversal between deciles 5 and 6** (ADP ≈ 28 → 35 is worth *more*, not
less). The isotonic estimator flattens it into a single plateau at 13.39 / 13.01 spanning
ADP 27.7–41.0. n = 28 per decile, so the reversal is well within noise; the monotone
constraint is doing exactly its job and no ad-hoc handling was applied. Second observation:
the RB curve is **much flatter at the bottom** than the WR curve (10.4 at ADP 80 vs WR's
8.7 at ADP 75) — RB replacement level inside the top 30 is high.

π and Σ diagonal for §J exported to `results/sectionJ_pi_sigma_rb.csv` (per-player
`pi_ppg` = fitted isotonic value at his 2026 ADP slot, `sigma_diag` = τ̂²(tier)). The
off-diagonal basis is the measured negative RB teammate covariance in §G2b.

## §G4 Thin-data players (`valuation_rb_2026.csv`)

Board tiers 2026: 22 vet / 6 soph / 2 rookie.

- **2 players with zero NFL rows** — Jeremiyah Love (ARI, ADP 25.6), Jadarian Price (SEA,
  ADP 64.5). n_eff = 0 ⇒ B = 1 ⇒ θ* = m_iso(ADP) exactly, i.e. the pure market arm, with
  posterior SD = √τ̂²(rookie) = 3.42. Flagged on the board as
  `no NFL rows: full shrinkage to market`.
- **6 single-season players** (all 2025 rookies): Jeanty, Hampton, Skattebo, Judkins, Tuten,
  Henderson. n_eff = 1 ⇒ V = 43.82 ⇒ B = **0.749**, the hardest shrinkage on the board.
  Flagged `single season: n_eff = 1`.

Round 1's finding is *not* repeated verbatim here: for WR, "n = 4 rookies across ten ADP
boards carries no information" was the honest verdict. For **RB the rookie cell has n = 25**
and τ̂²(rookie) = 11.68 [6.23, 16.40] is estimated, not invented. The full shrinkage to
market for Love and Price is therefore a *calibrated* statement, not a fallback.

## §G5 Availability (`availability_table_rb.csv`)

G = REG games with touches ≥ 2; M = 16 (< 2021) / 17; gate touches/game ≥ 4. 1,408 gated
player-seasons, 426 players.

| | WR (§A) | RB (§G5) |
|---|---|---|
| mean p̂ | 0.604 | **0.607** |
| YoY r of p̂ [player-bootstrap CI] | .422 [.363, .477] | **.369 [.301, .437]** |
| between-player SD of p_i (beta-binomial MoM) | 0.295 | **0.296** |
| ICC-analogue ρ | 0.364 | **0.368** |
| H0 "binomial + age" | p < .001 | null σ²_p mean 0.0001, 95th 0.0010; observed 0.0878 ⇒ **p < .001** |

**Availability structure is, to three decimals, the same for RBs as for WRs.** That is a
result worth stating plainly: the intuition that RBs are more fragile is not visible in
games-played-given-relevance. Game-level logistic (763 player-seasons, 275 clusters):
age **−0.0506/yr** (z = −2.59, p = .0095), prior-season games **+0.0939** (z = 10.6),
two-seasons-prior games **+0.0494** (z = 5.34) — same shape as WR's, slightly steeper age
term.

## §G6 LOSO 2015–2024 (`loso_scorecard_rb.csv`, `loso_predictions_rb.csv`)

Everything refit per fold: m_{−Y}, τ²_{−Y}(tier), σ²_{−Y}(tier), and every player's
μ̂/n_eff rebuilt from seasons strictly before Y (h = 1). `weekly_raw` now reaches 1999, so
unlike the WR run there is **no left-truncation artifact** — the round-1 "2015 fold" caveat
does not apply here. 286 in_fit eval rows, 27 with B = 1 (no prior data).

| target | arm | RMSE | mean Spearman | DM vs market-only, t(9) | p |
|---|---|---|---|---|---|
| PPG | (i) ADP-only m̂(ADP) | 3.9047 | .5234 | — | — |
| PPG | **(ii) blind posterior θ*** | **3.8378** | .5230 | **+0.766** | **.4635** |
| pts/scheduled wk | (i-SV) ADP-only refit on ppsw | 4.6934 | .4311 | — | — |
| pts/scheduled wk | **(iii) SV = θ*·Ê[G]/M** | **4.6578** | .4212 | **+0.399** | **.6990** |
| ppsw, all 300 rows | (iii) SV | 5.1841 | — | −0.009 | .9928 |

**VERDICT — the pre-specified honesty clause fires. Neither RB data arm beats market-only.**
Arm (ii) improves pooled RMSE by 1.7% but wins only **5/10 folds** and is nowhere near the
p < 0.10 adoption bar. Arm (iii) is worse still and is exactly null on the full 300 rows.
Per §G, **the 2026 RB board is market-anchored: `board_value = m_iso(ADP)`**, and we do not
go looking for a different arm. The value we add on RBs lives entirely in §J.

Contrast with WR, where arm (ii) gave RMSE 3.564 → 3.463 at DM p = .025.

### Why it fails — chased, and the answer is not "small sample"

| | WR | RB |
|---|---|---|
| mean yearly loss differential (ii) − (i) | **+0.695** | **+0.488** |
| SD across the 10 folds | **0.819** | **2.015** |
| folds improved | 7/10 | 5/10 |
| DM t | +2.684 | +0.766 |

The *mean* gain is 70% of WR's. The **across-fold SD is 2.5× larger**, and that is what kills
it. With SD = 2.015, the minimum yearly gain detectable at 80% power on 10 folds is ≈ 1.84
PPG²; the observed 0.488 is roughly a quarter of that. Drop-one-fold: p ranges .206
(excl 2020) to .862 (excl 2023) — **no single fold is driving the null**, and the arm never
approaches significance under any deletion. This is reported as-is; it is not a
"nearly significant" result.

The fold variance traces to a specific, position-specific event structure. The three
worst-hit rows in the worst fold and the three best in the best fold:

- **2020 (loss −2.58):** Le'Veon Bell (μ̂ 17.2 → realized 6.8), Todd Gurley (19.0 → 10.9),
  Mark Ingram (15.1 → 5.3). Three simultaneous veteran role/health collapses that the market
  had partially priced (ADP 29–43) and per-game history had not.
- **2023 (gain +3.96):** Alvin Kamara (ADP 69, m̂ 9.5, μ̂ 17.2, realized 17.9), James Conner
  (ADP 66, realized 15.5), David Montgomery (ADP 77, realized 14.8). Three veterans the
  market wrote off and whose own history knew better.
- **2021 (loss −1.50):** McCaffrey μ̂ 28.1 → realized 18.2.

That is the **RB cliff** in both directions: the data arm systematically wins when the market
over-discounts an aging producer, and loses catastrophically when a veteran's role actually
evaporates within one offseason. The two are the same phenomenon with opposite realizations,
they are roughly equal in magnitude, and they net to nothing — with enormous fold-level
variance on the way.

Structurally the same fact is the σ̂²_S = 5.78 and the 0.245 adjacent-season persistence in
§G2b: **an RB's own recency-weighted history is simply a much weaker signal about his next
season than a WR's is**, while eq. (7) weights it at 1 − B ≈ 0.47 (RB vets), slightly *more*
weight than WR vets get (1 − B ≈ 0.43). A structural note, recorded for a future
pre-registration and **explicitly not acted on here**: eq. (7)'s likelihood variance
V = σ²(tier)/n_eff measures how noisily μ̂ estimates *this* player's past level; it contains
no term for how far his *next* season's level will move from that. For WR that omission is
immaterial (median τ̂²_B ≈ 0); for RB the omitted term is ≈ 6 PPG², so V understates the
predictive variance of μ̂ by roughly 40% and B is too small. Testing a corrected V is a
round-5 item to be pre-registered, not a repair applied post-hoc to a result we did not
like.

### §G5 → §G6 coherence

Availability is a genuine, strongly-detected trait for RBs (ρ = .368, p < .001) and yet the
SV arm is null (p = .699). No contradiction: round 2 established that the WR SV win was
"θ* **plus** correct scaling to expected participation", with the differential
"he specifically will miss games" component not separately significant. For RB the
decomposition reproduces that: SV_const (fold-mean availability) DM t = +0.233 (p = .821),
and SV vs SV_const head-to-head t = +0.273 (p = .791) — no player-specific availability
signal either. With no θ* edge to scale, there is nothing for the participation factor to
amplify. The **pre-registered expectation of nothing was correct**, and it was recorded
before the fold loop ran.

---

## Files written by round 4 §G (nothing overwritten)

| file | contents |
|---|---|
| `results/valuation_2026_wr_20260809.csv` | refreshed WR board, August ADP |
| `results/consistency_table_20260809.csv` | §1 table for the refreshed WR universe |
| `results/wr_board_refresh_delta_20260809.csv` | per-player θ* decomposition vs frozen v3 |
| `results/sectionE_2026_20260809.csv`, `teammate_coherence_2026_20260809.csv` | restated round-3 layers |
| `data/players/wr_top30_weekly_20260809.csv` | game logs for the refreshed universe |
| `results/consistency_table_rb.csv` | §G2a |
| `results/variance_components_rb.csv` | §G2b |
| `results/heteroskedasticity_rb.csv`, `sigma2_by_tier_rb.csv` | §G2c |
| `results/market_prior_rb.csv`, `tier_variances_rb.csv`, `market_prior_iso_knots_rb.csv` | §G3 |
| `results/sectionJ_pi_sigma_rb.csv` | π, Σ-diagonal export for §J |
| `results/availability_table_rb.csv` | §G5 |
| `results/loso_predictions_rb.csv`, `loso_scorecard_rb.csv` | §G6 |
| `results/valuation_rb_2026.csv` | §G4 board (market-anchored per the honesty clause) |

## Open items / unresolved

1. **σ²_T for RB is negative and the model cannot hold it.** Measured, explained, reported;
   the 6-team-season sample cannot say whether −1.68 is real or noise. The §J off-diagonal
   should use it; a proper estimate needs the full RB population, not the board.
2. **Sophomore RB excess volatility survives the level control** (1.129, p = .022) where the
   WR analogue collapsed to 1.0. Mechanism untested.
3. Minor bookkeeping discrepancy noticed while building the WR/RB comparison: REPORT.md §3
   states "8 of 20 veterans have negative untruncated τ̂²_B"; recomputing from the frozen
   `consistency_table.csv` on `n_seasons >= 4` gives **10 of 20**. The frozen file is
   unchanged and the qualitative claim stands; the count in the prose is off by two.

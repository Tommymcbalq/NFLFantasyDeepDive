# The Model

*A preseason valuation and draft model for a 10-team PPR league.
Complete specification: what it computes, why each piece is there, and what it refuses to do.*

**Companion documents.** `REPORT.md` is the rolling derivation — how every component was arrived
at, including the ones that failed. `PROCESS.md` is the narrative log. `EDA_PLAN*.md` are the
pre-registrations, committed before their results. This document describes the model **as it
stands**, not the path to it.

---

## 0. The problem, and the one idea

Average draft position is a market price — the pooled judgement of thousands of drafters. Beating it
outright is hard, so the model does not start by trying. It treats **ADP as a prior**, treats a
player's **own history as data**, and combines them the way Bayes says to: weighting each by how much
it actually knows.

Every weight in that combination is **estimated from data, never chosen**. How noisy a game is (§2),
whether that noise varies by experience (§3), how much history a player effectively has (§1), what an
ADP slot has historically been worth and how widely outcomes spread around it (§6.1). Only then does
the model ask whether anything else — age, usage, team environment, schedule — predicts the market's
*errors*, under multiple-testing discipline and a temporal holdout.

**Ten independent attempts to find such an edge have failed.** That is the central empirical finding
of the project, and it shapes everything below: the model's job is not to out-predict the market
player by player. It is to (a) be right about *relative* value once positional replacement is
handled correctly, (b) give discretion a disciplined channel, and (c) convert a board into decisions.

---

## 1. Notation

| symbol | meaning |
|---|---|
| i | player |
| s | season within player i's career |
| Y_isg | PPR points, player i, season s, game g |
| Ȳ_is | season mean points per game |
| μ̂_i | recency-weighted mean of season means, half-life 1 season |
| μ\*_i | **calibrated, age-corrected** μ̂ — the data arm (§3) |
| n_eff | effective sample size of the recency weighting |
| σ²_W | within-season, game-to-game variance |
| τ²_B | between-season variance of a player's true level |
| A_i | ADP |
| m(·) | fitted ADP → points-per-game curve (§4) |
| τ²(e) | outcome variance around m(·), by experience tier e |
| V_i | variance of the data arm = σ²(e)/n_eff |
| B_i | shrinkage weight toward the market = V/(V + τ²) |
| θ\*_i | posterior expected PPG (§5) |
| π, Σ | Black–Litterman prior mean and covariance (§6) |
| R_p | replacement level at position p (§7) |
| λ | floor weight (§8) |

All analysis is **per game**, never per season: 17-game seasons from 2021, 16 before, COVID in 2020.

---

## 2. Data

- **Game logs** — nflverse weekly player stats, **1999–2025**, every player. 9,546 WR/RB
  player-seasons in the age panel alone.
- **ADP** — FantasyFootballCalculator PPR, ten historical boards 2015–2024 (666 WR / 603 RB
  player-seasons) plus the current board. *Provenance note (§35): FFC's `teams` parameter is inert —
  every value of it returns one identical pool. The project's earlier "12-team vs 10-team ADP"
  language was empty and is withdrawn. ESPN's stored 2023/24 ADP is **hindsight-contaminated** and is
  not used (§47.3).*
- **Advanced stats** — 316 documented columns: PFR (YBC, YAC, aDOT, broken tackles, drop rate), Next
  Gen Stats (separation, cushion, YAC-over-expected, 8+ box rate, CPOE, time to throw), snap counts,
  play-by-play, FTN charting, ESPN QBR.
- **Draft logs** — Sleeper's public API gives full pick-by-pick draft data with slots and traded-pick
  ownership.

**Game-inclusion rule**, fixed from aggregate distributions before any fitting: drop player-games
with targets ≤ 1 (WR/TE) or touches ≤ 1 (RB) or pass attempts ≤ 5 (QB). Consequence stated up front:
**all rates are conditional on participation**, and availability is handled separately (§8).

---

## 3. Layer 1 — the data arm, μ\*

**Intuition.** A player's own history is informative but two things distort it. First, a raw mean is
*over-dispersed*: it exaggerates how far a player sits from the pack, because a single season mean is
itself a noisy estimate. Second, it ignores that a 32-year-old and a 25-year-old should carry that
history forward differently.

    μ*_i = a_f + b_f·μ̂_i + g_f(age_i)                                     (3.1)

- **μ̂** is the recency-weighted mean of season means, half-life 1 season. It beat seven alternatives
  in §S — median, 20% trimmed, Huber, p60, role-gated, slope-adjusted and usage-implied — and the
  ordering was informative: robust estimators got *monotonically worse the more tail they discarded*.
  **The boom weeks are signal.** (§37's finding that dispersion does not persist is about predicting
  a player's *shape*, not about estimating his *level*.)
- **a_f, b_f** are calibration coefficients fitted **inside each training fold** and applied out of
  sample. The raw slope is **0.667 (WR) / 0.605 (RB)** — μ̂ needs shrinking toward the positional
  mean by a third.
- **g_f(age)** is §H's per-era age curve entering as the log-ratio f(age)/f(age−1), so it corrects
  the *transition* rather than re-levelling the player.

**Why age is here and not treated as an edge.** §H tested whether the aging curve has moved later in
calendar time (the popular claim) and found the opposite: WR cliff 31.05 → 29.35 → **28.05** across
eras, corroborated by a career-exit hazard rising from .231 to **.377** at age 30. §H5 then found the
market *prices age correctly* — no edge. But the curve is estimated and can be **applied**: this is a
within-model structural correction, not a claim to beat ADP. At RB it is essentially the only
correctable error in the data arm.

---

## 4. Layer 2 — the market prior, m(ADP)

**The move that makes everything else possible: don't theorise a price-to-points conversion,
measure it.** For every player on ten historical boards we know what the market charged (A) and what
he scored (Ȳ). Estimate the conditional mean

    m(a) = E[ Ȳ | A = a ]                                                  (4.1)

*Among all the players history priced at slot a, what did they actually average?* That is the entire
content of "market-implied value" — a fact about what prices have been worth, not a theory about what
they mean.

**Isotonic, not linear.** The only structure imposed is the one thing near-definitional about a
price: `a₁ < a₂ ⟹ m(a₁) ≥ m(a₂)`. A better draft price implies a **weakly** better player. Isotonic
regression finds the monotone curve minimising Σ(Ȳ − m(A))², solved by pool-adjacent-violators. It is
the minimal-assumption estimator here and it earns its place empirically (RMSE 3.32 against a
log-linear fit's 3.40).

**The flat steps are information, not noise.** Where the data cannot establish that one price range
outproduces the next, PAVA pools them into one level rather than inventing a distinction. The 2026
WR curve has 18 levels across ADP 1.4–75, and the WR2/WR3 stretch is nearly flat: m falls only
14.88 → 14.32 across ADP 26 → 42. **Read value gaps, not rank gaps** — a one-point disagreement can
move a player nine ranks in that region.

Also estimated: **τ²(e)**, the spread of realised outcomes around the curve, by experience tier. The
expected ordering rookie > soph > vet *failed* and was used as estimated, not imposed.

---

## 5. Layer 3 — the posterior

With prior θ ~ N(m(A), τ²(e)) and likelihood μ\* | θ ~ N(θ, V), completing the square:

    θ*_i = (1 − B_i)·μ*_i + B_i·m(A_i),      B_i = V_i / (V_i + τ²_i)      (5.1)

A precision-weighted average. No data (n_eff → 0) ⟹ B → 1 and **the price is the estimate**; long
recent history ⟹ B → 0. Every ingredient is estimated: μ\*/n_eff (§1), σ²(tier) (§3), m(·) and
τ²(tier) (§6.1). Observed on the 2026 board: B ∈ [0.56, 0.84] — always market-leaning, as it must be
with 1–3 effective seasons per player.

*(Derived but inert: if a projection arm were ever adopted, (5.1) generalises to a three-way GLS on
a common mean with Ψ = Cov(residuals) − (σ²_W/Ḡ)·J — a rank-one subtraction, because the realised
season is common to all three residuals. It reduces to (5.1) exactly when the third source is
dropped. WS1 rejected the projection, so the two-way blend stands.)*

---

## 6. Layer 4 — discretion, as Black–Litterman

**Why this shape.** Black and Litterman's contribution was not a better optimiser but a better
**prior**: rather than asking an investor for expected returns, invert the market to recover the
returns that would rationalise observed prices, then let subjective views enter as explicit
statements with declared confidence. The same pathology motivates it here, and §21 measures it:
**the entire market-implied spread from WR1 to WR30 is 7.0 PPG against a per-player uncertainty of
1.7–2.9 PPG.** The market's ordering is weakly identified by its own implied uncertainty, so small
input changes reorder large parts of the board.

- **π** = θ\*, the frozen statistical board.
- **Σ** = uncertainty about *true* value: Σ_ii = τ²(e) − σ̂²_W/Ḡ, floored at 25%. The subtraction
  matters — the residual spread of a realised season contains both uncertainty about θ and the noise
  of measuring θ from ~17 games, and BL wants the first only.
- **Off-diagonal: measured, then set to zero.** Same-team residual correlation is **r = +0.016**
  over 71 pairs, cluster-boot CI [−0.321, +0.320], indistinguishable from a random-pair null
  (p = .69). Recorded as a **power limitation, not a refutation** — a share-constraint ρ ≈ −0.3 sits
  inside that CI.
- **Views** are rows of P with magnitude q: absolute (one weight of 1) or relative (weights summing
  to zero). **Ω is declared, never fitted** — Ω_kk = (c·sd_prior(k))², scale fixed before any view
  was written: low 2.0, medium 1.0, high 0.5. "Medium" therefore means *as uncertain as the market
  is about the same quantity*.

    θ̄ = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹π + PᵀΩ⁻¹q]                          (6.1)
    θ̄ − π = M PᵀΩ⁻¹(q − Pπ)                                              (6.2)

(6.2) decomposes the shift **exactly and additively by view**, so each view's contribution to each
player is a reported column. That is what makes the layer auditable rather than a black box.

**Typed views.** A view is `player` or `structural`. A structural view applies to a group and is
implemented as the Ω→0 limit of |group| absolute views sharing a common offset — under diagonal Σ
that is exactly a flat shift on the group and exactly zero elsewhere, asserted numerically each run.

**δ_RB = 1.40** is the one structural view. Its size comes from two revealed-preference inequalities
made days apart (McCaffrey over Amon-Ra ⟹ ≥1.292; Barkley over Rice ⟹ ≥1.401), so 1.40 is the
**smallest** value consistent with both. It encodes a stated fact about this league's meta —
**11 RB against 5 WR in the first 16 picks, versus 8/8 at public ADP** — and is kept **out of the
column scored in January**, because it is a preference premium and not a points forecast.

**Every view is logged dated with its magnitude, confidence and rationale before the season**, so it
can be scored afterwards. Separating the layers is what makes that possible; blended into the fit, a
subjective input can never be evaluated again.

---

## 7. Layer 5 — positional value

    VORP_i = value_i − R_{p(i)}                                            (7.1)

**Why subtract anything.** A receiver at 15.0 PPG and a back at 15.0 PPG are not equally useful,
because the *alternative* differs. Subtracting replacement makes positions comparable.

**Replacement is a PPG order statistic among players with ≥8 games**, at the position counts this
league actually drafts (2026: 44 RB / 63 WR / 14 TE / 19 QB in the top 140):

| | QB | RB | TE | WR |
|---|---|---|---|---|
| R_p | 15.17 | 7.47 | 9.79 | 7.92 |

**Two errors were made here and both are instructive.** Values are points per game *played* while
replacement was computed as season total ÷ 17, i.e. per *scheduled* week — different quantities,
subtracted from one another, and because availability differs by position the error does **not**
cancel in the cross-position contrast that VORP exists to produce. The naive repair (rank by total,
read PPG off it) fixes the units and **breaks the identification**: it selects toward short high-rate
seasons, and one such observation (2024's WR64 was Diggs, 121.9 points in 8 games) blew the WR−RB gap
out to 2.06. The valued quantity is PPG, so replacement must be an order statistic *of PPG*.

**This layer does more work than everything else combined.** Ablation: remove it and the board's rank
correlation with the full model falls to **.768**; remove δ_RB, .963; remove all 37 player views,
.998; remove the empirical-Bayes arm, **.9996**.

---

## 8. Layer 6 — floor

    final_i = VORP_i + λ·( floor_i − floor_ref(p) ),   λ = 0.10            (8.1)

**floor** is the 25th percentile of PPR points over **scheduled** weeks — a missed game counts as the
zero it is — with documented suspensions removed from the denominator, blanked below 34 eligible
weeks, and thin-data players taking a neutral zero rather than a penalty.

**Why scheduled weeks.** Computed over games *played*, McCaffrey's floor is 16.30 and his bust rate
0.02 — the best on the board. Over scheduled weeks it is **0.00** and 0.33, because he has played
68.5% of them over three seasons. The first number is an artifact of only counting the games a player
showed up for.

**Why λ is small.** Floor is a tiebreak, not a driver: §37 found location persists (r(mean) = .69,
r(p25) = .63, r(bust) = .70) but **dispersion does not** (r(IQR) = .19), and regressing next season's
p90 on this season's mean *and* p90 gives a coefficient on p90 of −0.066. **Once last season's mean is
known, last season's ceiling adds nothing.** So the layer may inform floor, never ceiling.

---

## 9. Steps 8–10 — the decision, which is not the model's

**Availability is a declared input, not a model output.** Simulated survival curves produced 84% for
a player the owner says is 0%, and 1% for one whose belief is explicitly bimodal — "he falls to me or
he goes right before." A normal-CDF survival model **cannot represent a bimodal belief**, and no
calibration repairs that. The owner observes his league; the model observes ten boards of public ADP.

So `data/drafts/availability_priors.csv` is supplied by the owner and `scripts/74_decide.py`
**refuses to run without it** rather than substituting a guess. Same discipline as Ω — declared,
never fitted — applied to the quantity the owner knows best.

**Lineup-marginal value.** A player who would not start is worth **zero**, whatever his board rank:
Δ_p = max(0, v − c_p), where c_p is the starter he displaces — zero before a slot is filled,
discontinuous after. In a live case a QB with raw value 7.57 had marginal value **1.00**.

**The wait-or-take rule.** Order available players at position p by marginal value, let δ_i be
consecutive gaps and N_p the number removed before your next pick:

    W_p = Σ_i E[ δ_i · 1(N_p ≥ i) ]                                        (9.1)

The cost of waiting is the **survival-weighted sum of the tier gaps**. Decay enters only through
indicators bounded in [0,1]; **steps enter linearly and unbounded.**

> **Flat tiers are safe to wait on however fast they drain. Steps are not, however slowly.**

Demonstrated live: WR drained twelve times faster than QB at near-identical best-available value, yet
waiting cost 0.119 on WR against **1.148** on QB.

---

## 10. What the model refuses to do

Each of these was built, tested under pre-registration, and rejected. They are listed because a model
is defined as much by what it declines to assert.

| | result |
|---|---|
| **Project value from usage/efficiency inputs** | WR +0.249 (p = .404, below its own MDE), RB **−0.295**. Three estimators, gated features, ten folds. §W1 |
| **Price availability into value** | The naive μ̂ × availability multiplier is **significantly worse than doing nothing** (−2.36, p = .0085). Predicting next-season availability gives out-of-sample R² of 0.039/0.018 |
| **Adjust μ̂ for situation change** | Team change (−1.0 PPG) and vacated targets (+1.9) are *real* within-player, but adjusting μ̂ fails LOSO (p = .44) — ADP already carries the move, so it double-counts |
| **Team environment / Vegas win totals** | Worth +0.251 PPG per win; **ADP charges +0.194 of it (77%, p = .0085)**; residual +0.057 (p = .77) |
| **Schedule strength** | 0/16 survive BH, 0/16 beat the holdout. Bounded near zero by the near-zero-sum structure of a schedule |
| **Conversion patterns by draft cost** | Positionally flat: elite-RB minus elite-WR hit rate −1.1 pp (p = .87) |
| **Offensive environment reshaping the tail** | Raw effect is hindsight; a top-12 RB's own TDs are 18% of his team's points. Leave-own-player-out collapses it 95–197% |
| **Beat ADP on ordering** | A wash. What loses money is **roster mix**; what wins is the **room's bias** — the plain ADP strategy scores **+169 season points against a biased room** versus against ADP opponents |

**Ten independent edge tests, ten nulls.** The single durable data edge is a WR's own recent
production, and only for established starters: worth **+1.101 of face value with a full prior season
and +0.042 without**.

---

## 11. Honest limitations

- **ρ_max ≈ 0.41.** §2's variance decomposition caps how much of next-season PPG is forecastable
  from any preseason information. Most of what happens is not predictable, and the model's job is to
  capture the minority that is.
- **Ten seasons, ten clusters.** Every LOSO test runs on 10 folds. Minimum detectable effects are
  large (0.3–2.0 PPG depending on the design) and several nulls are explicitly *uninformative* rather
  than evidence of absence. Every null in §10 is reported with its MDE.
- **One draft.** The behavioural layer's parameters come from 78 opponent choices in a single mock.
  The τ-persistence pre-test — estimate a manager's temperature on one draft, again on another,
  correlate — **cannot be run at n = 1** and remains the pre-condition for any per-manager modelling.
- **δ_RB is not falsifiable before January**, and is deliberately kept out of the scored column.
- **The WR replacement level is the weakest number in §7** — the 2024 identification leaned on one
  8-game observation.

---

## 12. Running it

    python3 scripts/fetch_data.py                    # raw pulls, cached
    python3 scripts/70_build_board.py --mu-star      # the board, one pass, every layer a column
    python3 scripts/74_decide.py --pick 16 --next 25 --roster "..."   # needs your odds

`results/board_2026_v2.csv` carries every layer as a named column — μ̂, μ\*, π_market, θ\*, each
view's contribution, replacement, VORP, floor, final — so any number can be traced to its source and
any layer switched off.

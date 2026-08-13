# EDA — WR Valuation: Full Specification, v2 (2026-07-14)

Principle: **no named-player anchors, no tuning toward expected results.** Every estimate is
computed blind from the data; fit the model as specified and report what comes out. Any
intuition about a specific player ("X should look consistent") is a prediction to compare with
the output afterward — never an input to spec choice, never a debugging trigger, and never a
reason to refit. A surprising result is a finding, not a bug.

## Notation (used everywhere below)

| symbol | meaning |
|---|---|
| i = 1..N | player (N = 30) |
| s = 1..n_i | season within player i's career |
| g = 1..G_is | game within season s |
| Y_isg | PPR points, player i, season s, game g |
| Ȳ_is = (1/G_is) Σ_g Y_isg | season-mean PPG |
| Ȳ_i· = (1/n_i) Σ_s Ȳ_is | career mean of season means (unweighted) |
| e_is ∈ {0,1,2+} | years of NFL experience entering season s |
| t(i,s) | team of player i in season s |
| E[·], Var(·) | expectation/variance over the assumed generating process |

All analysis is per-game (17-game seasons from 2021, 16 before; never per-season totals).

---

## §1 Consistency profiles

### 1.1 Generating model for one player (index i suppressed)

Two levels of randomness, explicitly separated:

    season level:  θ_s = μ + δ_s,          E[δ_s] = 0,  Var(δ_s) = τ²_B
    game level:    Y_sg = θ_s + ε_sg,      E[ε_sg] = 0, Var(ε_sg) = σ²_W,  ε ⟂ δ, iid across g

- τ²_B ("between"): true year-to-year movement in the player's PPG level — role changes,
  QB changes, health. This is "per-year consistency."
- σ²_W ("within"): game-to-game scatter around that season's level. "Per-game consistency."

### 1.2 Why the naive between-season variance is biased, with the derivation

The observable season mean is the true level plus averaging noise:

    Ȳ_s = θ_s + ε̄_s,   ε̄_s = (1/G_s) Σ_g ε_sg   ⟹   Var(ε̄_s) = σ²_W / G_s
    ⟹   Var(Ȳ_s) = τ²_B + σ²_W/G_s                                            (1)

The naive estimate of year-to-year volatility is the sample variance of season means:

    v = (1/(n−1)) Σ_s (Ȳ_s − Ȳ·)²

Take its expectation. For independent Ȳ_s with common mean μ and Var(Ȳ_s) = V_s, using
E[Σ_s(Ȳ_s − Ȳ·)²] = Σ_s Var(Ȳ_s) − n·Var(Ȳ·) and Var(Ȳ·) = (1/n²)Σ_s V_s:

    E[Σ_s(Ȳ_s − Ȳ·)²] = Σ_s V_s − (1/n) Σ_s V_s = (1 − 1/n) Σ_s V_s

    ⟹  E[v] = (1/n) Σ_s V_s  =  τ²_B + (σ²_W/n) Σ_s (1/G_s)                    (2)

So v estimates τ²_B **plus** a positive term that grows when G_s is small. That is the precise
sense in which low-game seasons masquerade as year-to-year volatility: an 8-game injury season
contributes σ²_W/8 of pure averaging noise to v, and the naive number attributes it to the
player being "inconsistent year to year."

**Magnitude check with realistic numbers.** A typical WR1 has game-to-game SD around 7 PPG
(σ²_W ≈ 49). A 17-game season adds 49/17 ≈ 2.9 to Var(Ȳ_s); an 8-game season adds 49/8 ≈ 6.1.
If the player's true year-to-year variance is τ²_B = 4 (SD 2 PPG), the naive v averages ≈ 4 + 3
to 4 + 6 — i.e., naive year-to-year SD ≈ 2.7–3.2 vs a true 2.0. A 35–60% overstatement, worst
for exactly the injury-prone players where the distinction matters.

### 1.3 The corrected estimator

Estimate σ²_W unbiasedly from within-season scatter (each season's sample variance has
E[s²_s] = σ²_W; pool with df weights):

    s²_s = (1/(G_s−1)) Σ_g (Y_sg − Ȳ_s)²,      σ̂²_W = Σ_s (G_s−1) s²_s / Σ_s (G_s−1)

Then invert (2) — plug σ̂²_W in and solve for τ²_B:

    τ̂²_B = max{ 0,  v − σ̂²_W · (1/n) Σ_s (1/G_s) }                            (3)

E[τ̂²_B] = τ²_B before the truncation at 0 (truncation adds small positive bias when τ²_B ≈ 0;
report untruncated values alongside). If within-season variance visibly differs by season,
replace σ̂²_W/G_s with s²_s/G_s season by season in (3); expectation argument is identical.
Caveat: with n = 4–6 seasons v itself is noisy (χ² with n−1 df ⟹ SD of v is ≈ √(2/(n−1))·E[v]),
so τ̂²_B is reported with that uncertainty and only for players with n ≥ 4.

### 1.4 Level, shape, and rates

**Recency-weighted level.** With S_i the most recent season and half-life h (default 1,
sensitivity over h ∈ {0.5, 1, 2, ∞}):

    w_is = 2^{−(S_i − s)/h},    μ̂_i = Σ_s w_is Ȳ_is / Σ_s w_is

Its sampling variance, if Var(Ȳ_is) ≈ V for all s: Var(μ̂_i) = V · Σw² / (Σw)² = V / n_eff,
defining the effective number of seasons n_eff,i = (Σ_s w_is)² / Σ_s w²_is — the data-precision
input the §3 shrinkage needs.

**Shape.** Empirical quantiles of the game distribution: floor q_i(0.25), ceiling q_i(0.90).

**Boom/bust rates with empirical-Bayes stabilization.** Boom = 1{Y > 20}, bust = 1{Y < 8}.
Player i has k_i booms in m_i games; a raw rate k/m is noisy for short careers. Model:

    k_i | p_i ~ Binomial(m_i, p_i),    p_i ~ Beta(α, β) across players

Method-of-moments for (α, β): with p̂_i = k_i/m_i, the law of total variance gives

    Var(p̂_i) = Var(p_i) + E[ p_i(1−p_i)/m_i ]

so estimate μ_p = mean(p̂_i), subtract the average binomial noise p̄(1−p̄)·mean(1/m_i) from the
sample variance of p̂_i to get V̂ar(p_i), then α+β = μ_p(1−μ_p)/V̂ar(p_i) − 1, α = μ_p(α+β).
Report posterior means (k_i + α)/(m_i + α + β): short careers pulled toward the group rate,
long careers essentially untouched.

**§1 deliverable:** 30 × {μ̂, n_eff, σ̂_W, τ̂_B (n≥4), CV = σ̂_W/μ̂, q(.25), q(.90), boom, bust}.
No target values for any row.

---

## §2 Variance-component ANOVA

### 2.1 Model

    Y_isg = μ + a_i + b_is + c_{t(i,s),s} + ε_isg                              (4)
    a_i ~ N(0, σ²_P),  b_is ~ N(0, σ²_S),  c_ts ~ N(0, σ²_T),  ε_isg ~ N(0, σ²_G)

all mutually independent. a = stable player skill; b = player×season deviation (that year's
role/health/fit); c = team×season environment (shared by teammates); ε = game noise.
Estimation: REML (mixed model y = Xβ + Z_P u_P + Z_S u_S + Z_T u_T + ε). Primary window
2021–2025; 2014–2025 as sensitivity with season fixed effects.

Note c is indexed by team×season, so the same franchise in different years is an independent
draw — "team continuity" across years is not modeled here; it's a §5–6 covariate question.

### 2.2 Implied covariances (read directly off (4))

Cov(Y, Y′) = sum of variances of the random terms the two observations share:

    same player, same season, g ≠ g′:  σ²_P + σ²_S + σ²_T
    same player, different seasons:    σ²_P
    teammates, same season:            σ²_T
    otherwise:                         0

### 2.3 The predictability ceiling, derived

Season means under (4), with G games (balanced for clarity):

    Ȳ_is = μ + a_i + b_is + c_{t,s} + ε̄_is
    Var(Ȳ_is) = σ²_P + σ²_S + σ²_T + σ²_G/G
    Cov(Ȳ_is, Ȳ_i,s+1) = Var(a_i) = σ²_P        (only a_i appears in both)

    ⟹  ρ_max = Corr(Ȳ_is, Ȳ_i,s+1) = σ²_P / (σ²_P + σ²_S + σ²_T + σ²_G/G)     (5)

Interpretation: a preseason forecast built ONLY from a player's own history can correlate with
next season's realized PPG at most ρ_max (achieved by the best linear predictor); its R² is
capped at ρ²_max. The gap — σ²_S + σ²_T — is next year's context, which is what covariates
(§5) and market information (§6) attack. σ²_G/G is irreducible.

### 2.4 BLUP shrinkage of a career mean (the ANOVA-side regularizer)

Best linear predictor of a_i from n_i observed season means (balanced case):

    E[a_i | Ȳ_i1..Ȳ_in] = κ_i (Ȳ_i· − μ),    κ_i = σ²_P / (σ²_P + W/n_i),
    W = σ²_S + σ²_T + σ²_G/G

(from Cov(a_i, Ȳ_i·) = σ²_P and Var(Ȳ_i·) = σ²_P + W/n_i; κ = Cov/Var.) Small n_i ⟹ κ → 0:
short careers are automatically pulled to the population mean. §3 upgrades "population mean"
to the ADP-implied prior.

---

## §3 Non-constant variance by experience (location-scale)

### 3.1 Specification

    Y_isg = μ_is + ε_isg,  ε_isg ~ N(0, σ²_is),
    log σ²_is = γ₀ + γ₁·1{e_is = 0} + γ₂·1{e_is = 1}                           (6)

H₀: γ₁ = γ₂ = 0. exp(γ̂₁) is the multiplicative rookie variance inflation.

### 3.2 Estimation route A — Harvey's log-squared-residual regression

Stage 1: estimate μ_is (player-season means suffice). Residual e = Y − μ̂ ≈ σ_is Z, Z ~ N(0,1).

    log e² = log σ²_is + log Z²

log Z² has known moments: Z² ~ χ²₁ and E[log χ²_k] = ψ(k/2) + log 2 with
ψ(½) = −γ_E − 2 log 2, hence

    E[log Z²] = −γ_E − log 2 ≈ −1.2704,      Var[log Z²] = ψ′(½) = π²/2 ≈ 4.93

So the regression log e²_isg = (γ₀ − 1.2704) + γ₁·1{rookie} + γ₂·1{soph} + η, with η mean-zero,
gives consistent OLS estimates of γ₁, γ₂ (intercept absorbs the offset). Cluster SEs by
player×season.

### 3.3 Estimation route B — gamma GLM (kept as the estimate)

Under normality, e²/σ² ~ χ²₁ = Gamma(shape ½, scale 2), so E[e²] = σ² and Var[e²] = 2σ⁴ —
i.e., e² follows a gamma-family GLM with log link and dispersion 2. Fit e² ~ experience dummies
with that family; iterating (i) weighted estimation of μ with weights 1/σ̂² and (ii) this
variance regression is exactly full ML for model (6). Route A is the fast consistency check.

### 3.4 The posterior that uses these numbers, derived

Two more inputs join: m(ADP_i), the market-implied PPG (fitted in §6), and τ²(e), the
between-player spread of realized PPG around market price within an experience tier (also §6).
Let θ_i = player i's true next-season PPG.

    prior:       θ_i ~ N( m(ADP_i), τ²(e_i) )
    likelihood:  μ̂_i | θ_i ~ N( θ_i, V_i ),    V_i = σ̂²(e_i) / n_eff,i   (from §1.4)

Posterior ∝ exp{ −(μ̂−θ)²/2V − (θ−m)²/2τ² }. Collect terms in θ (complete the square):

    coefficient of θ²:  −½ (1/V + 1/τ²)
    coefficient of θ:      μ̂/V + m/τ²

    ⟹  θ | data ~ N( θ*, (1/V + 1/τ²)^{-1} ),
        θ* = (μ̂/V + m/τ²) / (1/V + 1/τ²) = (1−B) μ̂ + B m,   B = V/(V + τ²)    (7)

Precision-weighted average: B → 1 as V → ∞ (rookie, n_eff ≈ 0 ⟹ price is the estimate);
B → 0 as n_eff grows (veteran ⟹ his own games dominate). γ̂'s set σ²(e); §6 sets m(·), τ²(e);
n_eff comes from §1. Every shrinkage weight is estimated, none set by hand, and none of it
references any player by name.

---

## §4 Stat reliability — the gate for covariates

True-score model for any candidate stat X (target share, aDOT, ...): X_is = T_is + u_is with
Var(T) = σ²_T, Var(u) = σ²_u, T ⟂ u. Reliability ρ_X = σ²_T/(σ²_T + σ²_u).

### 4.1 Split-half with Spearman–Brown, derived

Split a season odd/even weeks; each half-season stat is X_A = T + u_A, X_B = T + u_B with
independent errors of variance σ²_h.

    r_half = Cov(X_A, X_B) / √(Var X_A · Var X_B) = σ²_T / (σ²_T + σ²_h)

The full-season stat is the average of the halves: X = (X_A + X_B)/2 = T + (u_A + u_B)/2, so
its error variance is σ²_h/2, and its reliability is

    ρ_full = σ²_T / (σ²_T + σ²_h/2)

Substitute σ²_h = σ²_T (1 − r_half)/r_half from the first equation:

    ρ_full = σ²_T / ( σ²_T + σ²_T(1−r_half)/(2 r_half) ) = 2 r_half / (1 + r_half)   (8)

### 4.2 Year-over-year vs reliability

    r_YoY = Corr(X_is, X_i,s+1) = ρ_X · φ_X,   φ_X = Corr(T_s, T_{s+1})

(under stationary variances). So r_YoY confounds measurement noise with true role change;
comparing r_YoY against ρ_full from (8) separates them: ρ_full high & r_YoY low ⟹ the stat is
measured well but the underlying role moves; both low ⟹ the stat is mostly noise at season
resolution.

Prediction rule for survivors (regression to the mean): E[X_{s+1}|X_s] = μ_X + r_YoY (X_s − μ_X).

Screened stats: target share, air-yards share, WOPR, aDOT = rec_air_yards/targets, RACR,
yards/target, TD/target, receiving EPA/game, PPG. Admission rule to §5–6: ρ_full ≥ 0.5 AND
r_YoY materially > 0 (bootstrap CI excluding 0). Expectations (to be checked, not enforced):
usage stats sticky, TD/target near noise.

---

## §5 Covariate structure

### 5.1 Age curve and the age–period–cohort identity

For player i in season with calendar year y: age_is = A_i + e_is and y = Y_i + e_is, where
A_i = entry age and Y_i = entry year are constants per player. Subtracting:

    age_is − y = A_i − Y_i = constant within player

So once player intercepts absorb (A_i, Y_i), the regressors {age, experience, calendar year}
are perfectly collinear — any linear trend can be attributed to age OR era OR experience
arbitrarily. Only nonlinear structure is identified. Spec:

    Y_isg = f(age_is) + δ_y + a_i + ε_isg,   f = natural cubic spline (df 4), a_i ~ N(0, σ²_P)

Interpretation restricted to the SHAPE of f (where the peak is, how steep the decline), with
the linear-trend caveat stated. Secondary: interact f with an aDOT-profile indicator (do
high-aDOT receivers decline earlier) — identified because it's a difference of shapes.

### 5.2 Team-environment elasticity (within-player identification)

Join team-week stats; with player×season fixed effects α_is:

    log(1 + Y_isg) = α_is + β₁ log(PassAtt_{t,s,g}) + β₂ PassEPA_{t,s,g} + ε_isg

By Frisch–Waugh–Lovell, β̂ equals OLS on variables demeaned within player-season — i.e., β₁ is
identified purely from game-to-game volume swings within the same player-season, uncontaminated
by "good players are on passing teams" selection. β₁ ≈ elasticity of production to team volume.
Cluster by team-week (teammates share the game shock).

### 5.3 Archetype, clustered not labeled

Feature vector per player-season z_is = (aDOT, target share, YAC/rec, TD/target, slot indicator
from ngs_position), standardized. Gaussian mixture, k by BIC. Then across clusters test:
- mean PPG: one-way ANOVA, F = MS_between/MS_within;
- variance (the more interesting question): Levene's test — ANOVA on |Y_isg − median_cluster| —
  chosen over Bartlett because PPR points are heavy-tailed and Bartlett assumes normality.

---

## §6 Market calibration

Historical panel: FFC ADP 2015–2025 (top-30 WRs each year) joined to realized next-season PPG;
~330 player-season rows.

### 6.1 Market prior curve and tier variances

    PPG_{i,s+1} = m(ADP_is) + R_is,    m = monotone-decreasing spline in log ADP

(log because value per pick is convex in draft position; monotonicity by constrained least
squares, with plain OLS-on-log as the simple fallback). Then

    τ̂²(e) = Var( R_is | e_is = e )

computed per experience tier — this is the prior variance the §3.4 posterior uses. Expectation
to test, not assume: τ²(rookie) > τ²(soph) > τ²(vet).

### 6.2 Efficiency test

Under market efficiency no preseason-observable predicts the residual:

    R_is = Z_is'β + u_is,    H₀: β = 0

Z = {age, team-change indicator, archetype, prior-year target share, rookie × prior-year team
pass EPA, veteran × team change, age × aDOT profile} — all entries restricted to §4 survivors
or fixed demographics. SEs clustered by season (draft-year common shocks); HC3 robustness.

### 6.3 Multiple-testing discipline

(a) Benjamini–Hochberg FDR at q = 0.10 across the interaction set; (b) the binding test is
temporal: fit on 2015–2022, evaluate on 2023–2025. In-sample-only contingencies are discarded.

### 6.4 Final valuation

Inverse-variance combination of the model projection ŷ_i (variance v_m, from LOSO residuals)
and the market m(ADP_i) (variance τ²(e_i)):

    V_i = ( ŷ_i/v_m + m(ADP_i)/τ² ) / ( 1/v_m + 1/τ² )

— formula (7) again with the model projection as the "data" arm. One empirical-Bayes pipeline;
hyperparameters {σ²(e), τ²(e), m(·), n_eff} all estimated.

---

## §7 Validation

Leave-one-season-out on the panel vs the ADP-only baseline m̂(ADP): (1) RMSE on next-season
PPG; (2) Spearman ρ within each year's top-30 board; (3) paired Diebold–Mariano-type test on
squared-error differences, clustered by season. "Beats blind ADP" = (3) significant.

## Deliverables
1. §1 table (30 players × level/variance/shape/rate columns, with n_eff and caveats)
2. Variance components (σ̂²_P, σ̂²_S, σ̂²_T, σ̂²_G) + ρ_max from (5)
3. γ̂₁, γ̂₂ with CIs — experience variance multipliers exp(γ̂)
4. Reliability table per stat: r_half → ρ_full via (8), r_YoY, admit/reject
5. f̂(age) shape, β̂₁ elasticity, archetype ANOVA + Levene
6. m̂(ADP), τ̂²(e), FDR-surviving edge coefficients
7. LOSO scorecard vs blind ADP

Data prerequisites: historical FFC ADP loop (for §3.4/§6); nflverse snap counts if per-route
denominators are wanted in §4–5.

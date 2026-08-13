# §2 notes — variance-component decomposition (2026-07-14)

Model (EDA_PLAN eq. 4): Y_isg = μ + a_i + b_is + c_{team×season} + ε_isg, PPR per game.
Data: REG games, §0 exclusions (targets ≥ 2) unless stated. Headline window 2021–2025:
n = 1,655 games, 30 players, 114 player-seasons, 95 team-seasons.

## Estimators — both run, both reported
1. **REML** — statsmodels MixedLM, single constant group, `vc_formula` = one variance component
   per factor (player, player-season, team-season). Converged on all four specs
   (lbfgs → powell). **Headline numbers are the REML column.**
2. **MoM covariance matching** — mean cross-products by pair type per plan §2.2
   (within player-season ⇒ σ²_P+σ²_S+σ²_T; same player cross-season ⇒ σ²_P; teammates within
   team-season ⇒ σ²_T; total variance ⇒ sum + σ²_G), solved linearly. No distributional
   assumptions; serves as the robustness check.

## Headline (2021–2025, exclusions, REML)

| component | σ̂² | ICC |
|---|---|---|
| player σ²_P | 5.48 | 0.069 |
| player-season σ²_S | 2.48 | 0.031 |
| team-season σ²_T | 1.20 | 0.015 |
| game σ²_G | 69.93 | 0.884 |

**ρ_max (eq. 5, G=17) = 0.413** ⇒ R² ceiling for a history-only preseason forecast ≈ 0.17.
Var(Ȳ_season) = 13.27 PPG²; of that, stable skill 5.48 (41%), next-year context σ²_S+σ²_T = 3.68
(28%), irreducible averaging noise σ²_G/17 = 4.11 (31%). Game-to-game noise dominates the raw
observation (88% of single-game variance): per-game PPR is mostly noise around a fairly stable
player level.

MoM cross-check on the headline agrees: 5.33 / 3.18 / 0.58 / 69.82, ρ_max 0.404. The only
material MoM–REML gap is the σ²_S vs σ²_T split (see identification note below); their sum
matches (3.76 vs 3.68).

## Sensitivities (full grid in variance_components.csv)

| spec | estimator | σ²_P | σ²_S | σ²_T | σ²_G | ρ_max |
|---|---|---|---|---|---|---|
| headline 21–25 excl | REML | 5.48 | 2.48 | 1.20 | 69.93 | 0.413 |
| headline 21–25 excl | MoM | 5.33 | 3.18 | 0.58 | 69.82 | 0.404 |
| (a) 14–25, season FE | REML | 5.32 | 2.78 | 1.55 | 70.42 | 0.386 |
| (a) 14–25, season FE | MoM | 3.01 | 5.00 | 0.05 | 70.35 | 0.247 |
| (b) log(1+Y), 21–25 | REML | 0.0373 | 0.0103 | 0.0110 | 0.403 | 0.454 |
| (b) log(1+Y), 21–25 | MoM | 0.0331 | 0.0222 | 0.0001 | 0.404 | 0.418 |
| (c) 21–25, no excl | REML | 6.05 | 2.67 | 1.42 | 71.17 | 0.422 |
| (c) 21–25, no excl | MoM | 5.65 | 3.79 | 0.36 | 71.18 | 0.404 |

ρ_max is stable at 0.39–0.45 across every REML spec and both response scales; the §0 exclusions
move nothing materially (they mostly trim σ²_G and slightly raise σ²_P).

## Anomalies chased

1. **MoM collapses on the 2014–2025 window (σ²_P 3.0, ρ_max 0.25) while REML barely moves.**
   Traced by computing the same-player cross-season mean product **by season lag**:
   lag 1: +7.36, lag 2: +5.22, lag 3: +2.20, lag 4: +1.01, lag 5+: negative (−1.8 to −7.4).
   The exchangeability assumption behind a constant a_i fails over a decade — career arcs
   (rise-peak-decline) make far-apart seasons anticorrelated around the career mean. MoM pools
   all lags (long careers contribute many long-lag pairs), dragging its "σ²_P" toward the
   long-lag covariance; REML's compound-symmetry likelihood is dominated by the denser short-lag
   information. Neither is "wrong"; they estimate different functionals under misspecification.
   The 5-year headline window largely avoids this (max lag 4), which is why headline MoM and
   REML agree.
   **Corollary finding:** even within 2021–2025 the lag-1 cross-season covariance (7.74) exceeds
   σ̂²_P (5.48) — b_is has positive year-over-year persistence (role continuity) rather than
   being independent across seasons. Under eq. (5) the one-season-ahead ceiling uses
   Cov(Ȳ_s,Ȳ_{s+1}) = σ²_P; plugging the *empirical* lag-1 covariance into the same ratio gives
   ≈ 7.74/13.27 ≈ 0.58. So 0.413 is the ceiling for the *permanent-skill-only* predictor, and
   an adjacent-season predictor that also carries persistent role information could reach ~0.58.
   Reported as a diagnostic; headline stays as pre-specified (eq. 5 with REML σ²_P). This is
   precisely the gap §5 covariates (role/team continuity) are meant to close.

2. **σ²_T is weakly identified.** Only 20 team-seasons in 2021–2025 contain ≥2 of the 30 WRs
   (1/3/4/5/7 by year), giving 4,128 teammate game-pairs from a handful of clusters; the two
   estimators split σ²_S+σ²_T differently (MoM puts nearly all of it in σ²_S) while agreeing on
   the sum. Zero co-rostered pairs exist among these 30 players before 2021, which is why the
   teammate-pair count is identical in the 2014–2025 window (verified — not a bug). Conclusion:
   σ²_T ≈ 0.5–1.5 PPG² (ICC ≤ 1.5%) but with wide uncertainty; the top-30-only sample is not
   informative about team effects. Does not affect ρ_max (enters numerator nowhere, denominator
   as part of a well-identified sum).

3. **Residual diagnostics (REML conditional residuals), reported not fixed:**
   headline skew +0.78, excess kurtosis +0.70, normal-QQ correlation 0.982 — the expected right
   tail of PPR points; mild, driven by big games not outlier rows. log(1+Y) *overcorrects*
   (skew −0.99, kurtosis +1.58, QQ r 0.974) because the reception floor of PPR makes the left
   tail short; the identity scale is the better-behaved of the two, supporting the identity-
   scale headline. Variance components as shares (ICCs) are nearly identical on both scales.

## Files
- `results/variance_components.csv` — all specs × both estimators, components, ICCs, ρ_max,
  residual diagnostics.
- Script: `scripts/02_section2_variance_components.py` (rerunnable; prints convergence flags).

# §3 notes — location-scale heteroskedasticity by experience (2026-07-14)

## Pre-specified sample (decided before estimation)

Experience-tier variance inflation is treated as a property of the WR position, so the
primary sample is **all WRs** in `weekly_raw` 2014–2025 passing the §1 inclusion rule
(REG only, targets ≤ 1 excluded), restricted to player-seasons with season-average
targets/game ≥ 3 (fantasy-relevant). Top-30-only is a sensitivity, not the headline.

Counts: 29,315 WR rows → 27,996 REG → 20,431 after targets>1 → **19,096 game rows**
(1,926 player-seasons of 2,334) after the mean-targets≥3 screen. 0 rows lacked
`rookie_season`; 0 negative-experience rows. Tier composition (game rows / player-seasons):
rookie 2,520/293, soph 2,968/308, vet (2+) 13,608/1,325.

Stage 1: e_isg = Y_isg − player-season mean (as specified; note this makes E[e²] =
σ²(1−1/G_s), a ~6% downward bias common to all tiers — does not affect tier contrasts).

## Estimates (γ's are log-variance offsets vs vet; mult = exp(γ̂) = variance multiplier)

| sample | route | γ̂₁ rookie (95% CI) | mult | γ̂₂ soph (95% CI) | mult |
|---|---|---|---|---|---|
| primary 2014–25 | A Harvey | −0.148 (−0.281, −0.015) | 0.862 | −0.063 (−0.176, +0.050) | 0.939 |
| primary 2014–25 | **B gamma GLM (headline)** | **−0.169 (−0.279, −0.060)** | **0.844** | **−0.082 (−0.172, +0.007)** | **0.921** |
| 2021–25 window | A | −0.111 (−0.318, +0.096) | 0.895 | −0.082 (−0.263, +0.098) | 0.921 |
| 2021–25 window | B | −0.157 (−0.341, +0.026) | 0.854 | −0.056 (−0.199, +0.086) | 0.945 |
| top-30 sensitivity | A | −0.175 (−0.465, +0.115) | 0.839 | −0.230 (−0.560, +0.101) | 0.795 |
| top-30 sensitivity | B | −0.269 (−0.500, −0.038) | 0.764 | −0.154 (−0.345, +0.036) | 0.857 |
| linear e = 0..5+ | A | slope +0.031/yr (+0.008, +0.055) | ×1.032/yr | | |
| linear e = 0..5+ | B | slope +0.030/yr (+0.011, +0.049) | ×1.030/yr | | |

Route A dropped 154 rows with e² < 1e-6 (exact zero residuals, G_s=1 seasons and ties);
Route B uses all rows. Route B fit with dispersion fixed at 2 (χ²₁ theory); freely
estimated Pearson scale = 3.48 (primary), i.e., e² is over-dispersed relative to the
normal-χ²₁ benchmark — consistent with right-skewed PPR points; CIs use the fixed-2 scale
with player×season clustering, which is what the plan pre-specified. Wald H₀: γ₁=γ₂=0
rejected, χ²(2) = 10.7, p = .005 — but in the **opposite direction from the folk
expectation**.

Headline σ̂²(tier), Route B primary (= raw mean e² by tier, as GLM with saturated tier
dummies must reproduce): rookie **36.42**, soph **39.73**, vet **43.14** PPG²
(`results/sigma2_by_tier.csv`).

## Anomaly chased: rookies have LOWER raw game variance than vets

Direction is stable across window, route, and sample (all six tier-dummy γ̂'s ≤ 0). Why:

1. **It is a level effect.** Mean PPG by tier: rookie 9.29, soph 9.89, vet 10.86 (top-30
   sample: 12.8 / 14.0 / 16.3). Game variance scales strongly with level — diagnostic
   regression (NOT the spec, run only to explain): log e² on log(player-season mean)
   gives slope ≈ 1.41 (gamma GLM) — near σ² ∝ μ^1.4.
2. **Conditional on level, tier effects vanish.** Diagnostic gamma GLM e² ~ rookie + soph
   + log μ_ps: rookie mult 1.011 (se .033), soph 1.051 (se .031) — both ≈ 1, n.s.; same
   in the top-30 sample. So rookies are exactly as noisy as vets *relative to their own
   level*; their lower ADP-tier production level mechanically gives lower absolute PPG
   variance. The relative-variance ordering (mean e²/μ²: rookie .425, soph .431, vet
   .390) actually leans the other way but is not the specified estimand.
3. No spec change: model (6) conditions on tier only, headline kept as pre-registered.
   The "rookie uncertainty" everyone intuits is **between-player** uncertainty (who the
   player is), which lives in τ²(e) (§6.1) and in n_eff — not in game-level σ². §6.1
   tests that directly.

Caveat for §3.4: V_i = σ̂²(tier)/n_eff uses a tier-constant σ², which understates V for
high-μ̂ players and overstates it for low-μ̂ players within tier (σ² ∝ μ^1.4). This is the
pre-registered formula; noted, not altered.

## §3.4 posterior valuation of the 2026 board (covariate-free / blind)

Eq. (7) applied to the 30-WR 2026 board (`results/valuation_2026_blind.csv`,
`scripts/08_valuation_blind.py`). Inputs all estimated upstream: μ̂ (h=1) and n_eff from
§1; σ̂²(tier) Route B above; m̂_iso(ADP) and τ̂²(tier) from §6.1. Tier at 2026 =
2026 − rookie_season: 27 vet, 3 soph, 0 rookie (verified: every board player has NFL
games, so the B=1 branch is unused). m̂ evaluated by linear interpolation between
isotonic thresholds — identical to sklearn's `predict` (max abs diff 2e-15); board ADP
range 2.8–60.1 sits inside the fitted 1.2–75.

Structural readouts (findings, not adjustments):
- B ∈ [0.56, 0.84] for all 30: the blind posterior is market-dominated everywhere,
  because n_eff at h=1 is only 1.8–3.0 and σ̂²(tier) ≈ 36–43 PPG² ⟹ V ≈ 14–24 vs
  τ² ≈ 7.9–11.3. Sophs (n_eff = 1, τ²_soph = 7.9) get B = 0.835.
- Biggest risers vs ADP (Δ = adp_rank − θ*-rank): Rice +8, Adams +8, Evans +6,
  Nabers +5. Post-hoc decomposition: each has strong recent per-game production
  (Rice 21.6/18.8 PPG in short 2024/2025 stints; Nabers 18.2 rookie PPG before a
  4-game 2025) while the market prices information the blind model cannot see —
  availability/age/legal risk. τ̂²(vet) itself excludes total-season-loss risk (§6.1
  fit floor), so the blind model systematically over-ranks per-game-good,
  availability-risky players. Expected consequence of covariate-freeness; documented,
  not patched (§6.2 covariates are the pre-specified home).
- Biggest fallers: Waddle −4, Olave/Flowers/McConkey/Smith/Burden −3 — all cases where
  μ̂ sits below the ADP curve and B < 1 lets the games data drag θ* under the market.

## Files
- `results/heteroskedasticity.csv` — all estimates above.
- `results/sigma2_by_tier.csv` — headline σ̂²(tier) for the §3.4 posterior.
- `results/valuation_2026_blind.csv` — §3.4 blind posterior board.
- Scripts: `scripts/06_section3_heteroskedasticity.py`, `scripts/08_valuation_blind.py`.

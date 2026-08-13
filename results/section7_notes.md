# §7 notes — leave-one-season-out validation (2026-07-15, verification pass)

## Spec

For each board year Y ∈ 2015-2024, hold out Y entirely and rebuild every input without it:
m_{-Y} (isotonic, monotone decreasing in log ADP, in-fit rows of the other 9 years),
τ²_{-Y}(tier) (variance of the leave-Y-out isotonic residuals by tier), σ²_{-Y}(tier)
(§3 primary sample, all WRs 2014-2025 with seasons ≠ Y), and per-player μ̂/n_eff from
weekly data STRICTLY before Y (h = 1 recency weights, §1 inclusion rule). Posterior θ*
per eq. (7); zero-prior players get B = 1 (θ* = m̂; 4 rows, all the panel's rookies).
Evaluation on in-fit rows (n = 291). Candidates: (i) m̂_{-Y}(ADP); (ii) blind θ*;
(iii) θ* with prior mean m̂_{-Y} + Z'β̂_{-Y} on the §6.2 FDR survivors (already
temporal-holdout FAILURES — carried as the pre-specified candidate, not a live model).

**Leakage audit (verification pass, checked hard):** m, τ², σ² all exclude Y; μ̂/n_eff use
seasons < Y only; no 2026-board membership enters anywhere (the eval universe is the
historical boards). One leakage bug WAS found and fixed: predictor (iii)'s β_{-Y} was fit
on `resid_iso` — residuals from the §6.1 full-sample isotonic (fit including year Y) —
instead of the leave-Y-out residuals computed two lines earlier. Fixed to use the
leave-Y-out residuals; effect confined to (iii): RMSE 6.609 → 6.469, DM p .351 → .353.
Predictors (i)/(ii) and the DM headline are bit-identical before and after.

## Scorecard (post-fix; `results/loso_scorecard.csv`)

| predictor | RMSE | mean within-yr Spearman | DM vs (i) |
|---|---|---|---|
| (i) ADP-only m̂(ADP) | 3.564 | .461 | — |
| (ii) blind posterior θ* | 3.463 | .467 | t = +2.68 (9 df), **p = 0.025** |
| (iii) θ* + FDR edge terms | 6.469 | .460 | t = −0.98, p = .35 |

**Headline: the blind shrinkage posterior beats blind ADP out of sample** (2.8% RMSE,
significant at the pre-registered paired test). (ii) beats (i) in 7 of 10 folds. Adding
the FDR-passing-but-holdout-failing edge terms is much worse — the multiple-testing +
temporal-holdout discipline did its job.

## DM test implementation (verified)

Paired squared-error differences d = (y − ŷ_i)² − (y − ŷ_ii)², averaged within each of
the 10 year-clusters; t = mean(d̄_yr)/(sd(d̄_yr)/√10) referred to a **t with 9 df**
(`stats.t.sf(|t|, df=9)`), not a normal — correct for 10 clusters. Verified in code and
by recomputation from `loso_predictions.csv`.

## The 2015-fold anomaly, traced

Per-fold metrics (predictors i / ii; n_trunc = board players whose careers began before
the 2014 data window):

| fold | RMSE i | RMSE ii | ρ i | ρ ii | RMSE iii | n_trunc |
|---|---|---|---|---|---|---|
| 2015 | 4.68 | 4.53 | .24 | .30 | **17.6** | 18 |
| 2016 | 3.16 | 3.18 | .45 | .43 | 3.2 | 18 |
| 2017-24 | 3.06–3.92 | 2.44–3.74 | — | — | 2.4–3.7 | 0–18 |

1. **The anomaly is entirely in predictor (iii), not (ii).** With 2015 held out, the only
   rookies left to fit β_{-Y} are Harrison/Nabers 2024, whose new-team EPA z-scores
   (−1.18, −1.31) are nearly collinear with the rookie dummy → β̂_rookie ≈ −79,
   β̂_rook×epa ≈ −63 (a degenerate 2-point fit). Applied to the 2015 rookies:
   Agholor (epa_z +0.30) → θ_edge = −85 PPG, Cooper → +38. That one fold contributes a
   yearly mean loss diff of −288 vs (i) and is the whole reason (iii)'s pooled RMSE is
   6.5 not ~3.5. Another face of the same n = 4 rookie cell already condemned in §6.2/6.3;
   no action beyond reporting — (iii) was already a dead candidate.
2. **Predictor (ii) is fine in 2015** — it beats (i) there (RMSE 4.53 vs 4.68, yearly
   mean loss diff +1.37, Spearman .30 vs .24). 2015 is every predictor's hardest fold
   (both RMSEs highest of any year), but that is fold difficulty, not model failure.
3. **Left-truncation quantified.** 18/30 of the 2015 board have pre-2014 careers, so
   μ̂ = one season (2014) and n_eff = 1.0 exactly (rising to ~2.9 by 2020+). The
   edge-of-window "inflated-confidence n_eff" hypothesis is REFUTED in direction:
   truncation *deflates* n_eff (1.0 vs the ~2.2-2.6 full-history veterans get later),
   so V = σ²/n_eff is larger and B shrinks these players HARDER toward the market
   (B ≈ 0.81) — the conservative direction. Truncated μ̂ is biased high in 2015-16
   (+1.52 PPG vs −0.75 for non-truncated, Welch t = 2.14, p = .04) — a single recent
   season overstates an aging veteran's next-year level — but the heavy shrinkage
   attenuates it: posterior bias in 2015 is −0.26 (truncated) vs the market's −0.53.
   A milder version of the same +1.1 PPG μ̂ bias persists for pre-2014-career players
   in 2017-24 folds with full n_eff — i.e., it is substantially an age-decline effect
   (older cohorts decline; recency-weighted history has no age curve), not purely a
   window artifact. Consistent with §5's age curve; noted as a candidate refinement,
   not patched.

## Sensitivity: excluding the edge-of-window folds (headline unchanged, reported)

DM (ii) vs (i), recomputed on fold subsets:

| folds | k | RMSE i | RMSE ii | t | p |
|---|---|---|---|---|---|
| all 10 | 10 | 3.564 | 3.463 | +2.68 | .025 |
| excl 2015 | 9 | 3.412 | 3.318 | +2.24 | .056 |
| excl 2016 | 9 | 3.603 | 3.491 | +2.91 | .020 |
| excl 2015-16 | 8 | 3.440 | 3.334 | +2.42 | .046 |

The RMSE advantage of (ii) is stable (~2.7-3.1%) in every subset; dropping 2015 alone
moves p to .056 because 2015 is a fold that *favors* (ii) (+1.37 loss diff) and a df is
lost — not because truncation flatters the posterior. Headline retained as pre-registered
(all 10 folds); sensitivity reported.

## Files
- `results/loso_scorecard.csv` — the 3-row scorecard.
- `results/loso_predictions.csv` — 291 per-row predictions (m̂, μ̂, n_eff, B, θ*, θ_edge).
- Script: `scripts/10_section7_loso.py`. Verified 2026-07-15: pre-fix rerun byte-identical
  to the CSVs on disk; post-fix rerun changes `theta_edge` only.

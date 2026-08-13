# §6.2–6.3 notes — market-efficiency (edge) regression + gates (2026-07-15, verification pass)

## Spec (pre-registered in the script header before fitting)

R = realized PPG − m̂_iso(ADP) (from §6.1 panel, in_fit rows, n = 291 → 285 after
covariate-missingness drops) regressed on 10 preseason-knowable Z terms: age, team_change,
ts_prior, adot_prior, wopr_prior, games_prior (pre-registered addition motivated by the
§3.4 blind-run availability diagnostic, added BEFORE this regression ran), rookie,
vet_x_change, agec_x_adotc, rook_x_epa. All ratio stats are §4 reliability survivors.
`no_prior_vet` enters as a nuisance control, not an FDR family member.

- Missing-data rules fixed before fitting: rookie prior-year ratio stats imputed at the
  non-rookie fit mean (rookie dummy absorbs the constant); non-rookie zero-prior-games
  keeps games_prior = 0 (true availability value) + nuisance indicator.
- Inference: OLS, SEs clustered by season (10 clusters, use_t → t with 9 df); HC3 as
  robustness. BH-FDR q = 0.10 across the 10 clustered p-values.
- Binding temporal gate (§6.3, window amended 2023-2025 → 2023-2024, no 2025 ADP at
  source — documented in section6a notes): refit survivors on 2015-2022 in-fit rows
  (n = 226), compare holdout MSE on 2023-2024 (n = 59) vs the zero prediction (market
  efficiency). Survival in §6.4 requires BOTH gates.

## Results

**Nothing survives both gates.** Two terms pass FDR:

| term | β (clustered) | p_cluster | p_HC3 | holdout MSE vs zero 9.463 |
|---|---|---|---|---|
| rookie | −6.87 | ~0 | ~0 | 10.132 — FAILS |
| rook_x_epa | −4.04 | .0002 | .006 | 9.618 — FAILS |
| joint | — | — | — | 9.694 — FAILS |

Everything else n.s. under clustered SEs: age −0.124 (p = .16), adot_prior −0.242
(p = .20), games_prior −0.023 (p = .72), ts_prior/wopr_prior (collinear pair, VIF ≈ 23,
both n.s.), agec_x_adotc (p = .42), team_change/vet_x_change (p = .99). R² = 0.051.

## Anomalies chased

1. **team_change ≡ vet_x_change on the fit sample** (found, not assumed): no sophomore in
   the 300-row panel ever changed teams — rookie-contract selection into top-30 boards.
   Only their sum is identified (+0.009); pinv splits the beta arbitrarily. The joint Wald
   runs on the 9 identified terms.
2. **The rookie t-stats are a few-treated-clusters artifact.** All rookie information
   lives in n = 4 rows in 2 of 10 season clusters (2015: Cooper/Agholor; 2024:
   Harrison/Nabers). Cluster-robust (and HC3 with n = 4) inference is unreliable there —
   which is precisely why the temporal gate is binding, and both rookie terms fail it.
   The interpretable joint test on the 7 non-rookie terms: clustered F = 2.72, p = .082;
   HC3 F = 8.38, p = .30 — market efficiency not rejected on the veteran side.
3. The 9-constraint joint Wald (F = 2650, p ≈ 0) sits at the cluster-covariance rank
   boundary (G − 1 = 9) and is driven by the degenerate rookie terms; reported with that
   caveat, not used for any decision.

## Design note (verification pass)

The §6.3 holdout keeps m̂_iso fixed at its full-sample (2015-2024) fit; both arms (Z-model
and zero prediction) use the same residual definition, so the gate isolates Z's
incremental out-of-sample signal rather than re-testing the curve. The strict
no-full-sample-fit validation of the whole pipeline is §7 LOSO, where m is refit per fold.

## Verdict

Within this pre-specified Z and panel, no systematic market mispricing was found; the
final valuation (§6.4) is therefore the blind posterior, unmodified. The FDR-passing
rookie terms are an n = 4 artifact — confirmed independently by their LOSO blow-up
(section7 notes).

## Files
- `results/edge_panel.csv` — panel + covariates (reused by scripts 10/11).
- `results/edge_regression.csv` — betas, clustered/HC3 SEs, p's, VIF, fdr_survivor,
  holdout_improves, final_survivor (all False).
- `results/edge_holdout.csv` — holdout MSEs and per-year loss diffs.
- Script: `scripts/09_section6b_edge.py`. Verified 2026-07-15: reruns byte-identical.

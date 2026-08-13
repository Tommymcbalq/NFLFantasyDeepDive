# §1 notes — consistency table (2026-07-14)

## §0 game-inclusion rule (pre-specified from aggregate distributions only)

Decided before computing any player-level quantity, from the pooled distribution of
targets/target_share across all 2,033 regular-season player-games:

- **Keep `season_type == 'REG'` only** (118 POST rows dropped; fantasy scoring universe and
  cross-season comparability).
- **Exclude player-games with `targets <= 1`.** Basis: the pooled target-count distribution has
  a smooth body beginning at 2–3 targets; the `targets <= 1` tail is 39/2,033 rows (1.9%) with
  mean PPR 1.9 and mean target_share 0.038 vs. a population median target share of 0.24 —
  consistent with a non-participation mixture component (injured-early exits, decoy snaps,
  near-inactive rows). No snap-count column exists in this table, so targets is the available
  participation proxy. Rule is denominator-free and simple.

**Counts:** 2,033 REG rows → 39 excluded → 1,994 analyzed. Exclusions concentrate exactly where
expected: 7 games for one player's post-ACL rookie year (decoy usage), 5 for another's two
injury-riddled seasons, and 1–3 scattered early-exit games for 16 others.

Headline table = with exclusions (`consistency_table.csv`); sensitivity = REG-only, no
exclusions (`consistency_table_noexcl.csv`).

## Estimator choices (as pre-specified in EDA_PLAN §1)

- μ̂: recency-weighted mean of season means, weights 2^{−(S_i−s)/h} on **season-year** distance
  (handles any skipped years); headline h=1, sensitivities h ∈ {0.5, 2, ∞} as columns.
- n_eff = (Σw)²/Σw² at h=1.
- σ̂²_W: df-weighted pool of season sample variances (seasons with G_s=1 contribute 0 df).
- τ̂²_B: eq. (3), v − σ̂²_W·(1/n)Σ(1/G_s); truncated (reported as SD, `tau_B`) and untruncated
  (`tau2_B_untrunc_reported`), veterans n≥4 only; `naive_v` and the correction term are both
  in the table so the size of the correction is visible (it is 3–10 PPG², i.e., 30–170% of
  naive_v for most rows — the bias is first-order, not a footnote).
- q25/q90: empirical quantiles over all included career games, pooled across seasons.
- Boom P(Y>20)/bust P(Y<8): EB beta, method-of-moments across the 30 players.
  Headline hyperparameters: boom (α, β) = (6.11, 20.20), bust (4.31, 13.23). Between-player
  variance is well above binomial noise for both rates (Var(p̂)=.0106 vs noise .0041 boom;
  .0142 vs .0042 bust), so shrinkage is moderate; only the ≤17-game careers move much.

## Anomalies chased

1. **Cluster of negative untruncated τ²_B among veterans** (8 of 20 n≥4 players, range −0.2 to
   −3.2). Under τ²_B=0, SD(v) ≈ √(2/(n−1))·correction; every negative value has |z| < 1
   (max −0.88). So the negatives are exactly what an unbiased estimator produces when the true
   between-season variance is ≈ 0 — for most established WRs, essentially all observed
   season-mean movement is game-level averaging noise. Contributing second-order effect:
   pooled lag-1 within-season autocorrelation is slightly negative (−0.086, se 0.021, 134
   player-seasons), so Var(ε̄_s) is ~10–15% below σ²_W/G and the correction over-subtracts by
   ~0.3–0.8 PPG²; this pushes small-τ² players slightly negative but changes no conclusion.
   Headline kept as specified.

2. **Two veterans with large τ̂²_B (~22 PPG², SD ≈ 4.7)**, z ≈ 8–9 above noise: both are
   genuine career-arc cases — season means march from single digits in early low-role years to
   ~18–25 at peak (and back down for the 12-season career). Interpretation caveat: the model
   treats seasons as exchangeable draws around a constant μ, so a monotone career trajectory
   loads into τ²_B. For these players τ̂_B is "career non-stationarity," not iid year-to-year
   wobble. Noted, not refit (the §5 age curve is the pre-specified home for trend).

3. **Exclusion-rule sensitivity — biggest movers, traced to games:**
   - One n=3 player's naive_v jumps 7.5→18.0 with exclusions: a 4-game injury season loses its
     targets≤1 exit game, so the season mean is computed from 3 full games (16.2→21.6),
     widening season-mean spread. n<4 so τ_B not reported either way.
   - One n=4 player's τ²_untrunc flips +5.4→−0.3: two injury-exit games removed from an 8-game
     season raise that season's mean 9.6→12.5, aligning it with his other seasons. The naive
     number was booking early exits as year-to-year volatility — precisely the failure mode
     the rule targets.
   - The post-ACL rookie-year player: τ²_untrunc 24.7→9.2 after 4 decoy games are removed from
     a 6-game rowed season (season mean 2.5→7.6). Residual caveat: that season retains G=2.
   - All other rows move by <0.5 PPG in μ̂ and <1 PPG² in τ²; boom/bust EB rates move ≤0.05.
     Bust rates fall mechanically under exclusions (the removed games are all busts); the
     headline bust rate is "bust given participation," which is the intended estimand.

4. **Rookies (n=1):** τ̂_B undefined, n_eff=1, EB shrinkage does most of the work on their
   boom/bust (e.g., a 13-game rookie's raw boom .077 → EB .18). As designed.

## Files
- `results/consistency_table.csv` — headline, sorted by μ̂ (h=1) descending, computed blind.
- `results/consistency_table_noexcl.csv` — no-exclusion sensitivity.
- Script: `scripts/01_section1_consistency.py` (rerunnable).

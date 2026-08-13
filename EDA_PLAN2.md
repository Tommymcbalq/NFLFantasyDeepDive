# EDA Round 2 — Availability, Situation Change, and a Better Data Arm
### Pre-registered 2026-07-15, before any round-2 fitting. Same governing rules as round 1:
no tuning to expectations, no named-player anchors, anomalies chased to explanations,
market-edge claims require BH-FDR (q=0.10) AND temporal holdout, validation is LOSO.

Notation continues from EDA_PLAN.md / REPORT.md. New quantities defined below.
Round-1 outputs are frozen inputs: θ*, m̂(·), τ̂²(e), σ̂²(e), f̂(age), reliability verdicts.

## New data (§D0 — grab before analysis)
- nflverse `snap_counts/snap_counts_{2012..2025}.csv` — participation ground truth.
- nflverse `injuries/injuries_{2014..2025}.csv` — weekly injury reports/designations.
- Derived tables (from existing weekly data):
  - **primary QB** per team-season = passer with most attempts in that team's games; QB-change
    indicator per player-season = primary QB differs from prior season (or team changed).
  - **vacated targets** for team t entering season s = share of team t's season-(s−1) targets
    thrown to players not on t's roster in season s (roster = appears in ≥1 game for t in s).
All raw pulls cached under data/, never overwritten; report row counts + season coverage.

## §A — Availability as a modeled outcome (not a conditioning event)
Everything in round 1 is per-game-given-participation. Model participation itself.

- **Definition.** G_is = games participated (targets ≥ 2 rule; snap-count ≥ 25% as
  sensitivity); M_is = team's scheduled games (16/17). Availability rate p̂_is = G_is/M_is.
- **Is "injury-prone" a stable trait?** Hierarchical beta-binomial:
  G_is ~ BetaBin(M_is, α_i, β_i) with player-level p_i ~ Beta(a, b). Estimate the
  between-player variance of p_i (method of moments as in round-1 §1.4, or ML); report the
  ICC-analogue and the YoY correlation of p̂ with player-bootstrap CI. H0: no stable trait
  (pure binomial + age). Also logistic model: game-level participation ~ age + prior-season
  games + prior-2 games, cluster by player.
- **Value integration.** Season value per scheduled week: SV_i = θ*_i · Ê[G_i]/17, with
  Ê[G_i] from the availability model (age + prior participation). New LOSO arm (iv):
  predict realized TOTAL season PPG-per-scheduled-week (realized points / M) with
  SV vs (i) m̂(ADP) re-fit on that target. Same DM protocol. This is the fair fight the
  market has been winning by default.

## §B — Situation change: does context turnover degrade carryover, and does the market misprice it?
- **B1 carryover degradation.** For gate-admitted stats (target share, WOPR, aDOT): regress
  X_{i,s+1} on X_is × {no change, team change, QB change (same team)} — interaction on the
  slope. H0: carryover slope equal across regimes. All WRs 2015–2025, cluster by player.
- **B2 level shift.** Within-player: ΔPPG_{s→s+1} on change indicators + situation-quality
  covariates of the NEW situation: prior-year team pass attempts (z), prior-year team pass
  EPA (z), vacated target share. Cluster by season.
- **B3 market test (edge protocol).** On the 2015–2024 ADP panel: R = market residual
  regressed on {team change × vacated targets, team change × new-team prior pass EPA,
  QB change}. Pre-specified family; BH-FDR q=0.10; temporal holdout 2015–2022 → 2023–2024.
  Survivors (if any) enter the posterior prior mean; a null is a reported result.

## §C — Age-detrended data arm (fixes the verified +1.1 PPG bias for old careers)
μ̂ᵃ_i = recency-weighted mean of age-adjusted season means:
Ȳᵃ_is = Ȳ_is − [f̂(age_is) − f̂(age_i,2026)], f̂ from round-1 §5 (shape-only; season FE
handled as in that fit). Rebuild θ*ᵃ with identical B weights. LOSO arm (v). No refitting of
f̂ inside folds is allowed to peek at the held-out year (refit f̂ per fold on ≤Y−1 data).

## §D — Usage-based projection of the data arm (chasing the 0.41→0.58 ceiling gap)
Replace raw-PPG history with a projection through the *reliable* stats:
PPG_{i,s+1} = g(target share_is, WOPR_is, aDOT_is, team pass attempts_ts, age) fit by ridge
(λ by CV within training folds) on all WR player-seasons. Data arm becomes ŷ_i = ĝ(x_i,2025)
with V from training-fold residual variance; posterior re-formed per eq. (7). LOSO arm (vi).

## Validation & reporting
- LOSO scorecard extended: arms (i) ADP-only, (ii) blind θ* [frozen round-1], (iv) SV,
  (v) θ*ᵃ, (vi) usage posterior, and (v+vi) combined. Per-game arms score on realized PPG;
  availability arm (iv) scores on points per scheduled week vs a re-fit market baseline —
  never compare across different targets. DM clustered by year, t(9 df), pre-registered.
- Every section writes results/*.csv + results/section{A,B,C,D}_notes.md in house style.
- Figures appended to results/figures/ (fig13+); REPORT.md gets a Round 2 part when done.

# §A notes — availability as a modeled outcome (2026-07-15, round 2)

## §D0 data audit (my share)

- **Snap counts** — `data/snap_counts/snap_counts_{2014..2025}.csv` from nflverse-data
  releases (URL pattern verified on 2014 before bulk pull). 12/12 seasons, 287,654 REG
  rows total (~22.9k/yr through 2020, ~25.3k/yr from 2021; weeks 1–17 pre-2021, 1–18
  after — consistent with the schedule change). Key = `pfr_player_id`; joins to
  `gsis_id` via `players_meta.pfr_id` (90.1% of meta rows have a pfr_id; **98.5% of
  fantasy-relevant WR player-seasons matched** to snap data — the join is not a
  bottleneck for this universe). `offense_pct` is a 0–1 fraction.
- **Injuries** — `data/injuries/injuries_{2014..2025}.csv`. 12/12 seasons, 65,866 rows
  (4.9k–6.0k REG rows/yr). Carries `gsis_id` directly — no crosswalk needed. Not used
  in §A's models (participation is measured from targets/snaps); cached for §B+.
- Plan asks for snap_counts 2012–2013 too; **not pulled** — the analysis window and
  weekly data start at 2014, so 2012–13 would join to nothing. Deviation noted.

## Definitions (as pre-registered)

G_is = REG games with targets ≥ 2 (round-1 §0 participation rule); M_is = 16 (<2021) /
17 (≥2021); p̂ = G/M. Universe: all WRs 2014–2025 passing the round-1 fantasy-relevance
filter (season mean targets ≥ 3 over included games): **1,926 player-seasons, 595
players** (2,334 before the filter). Sensitivity: G_snap = games with offense_pct ≥ 25%.
corr(p̂, p̂_snap) = 0.947; the snap definition adds +0.68 games on average (games played
with a real snap share but <2 targets), mean p̂ 0.604 vs 0.653.

**Interpretation caveat stated up front:** with the targets-≥2 definition, p̂ is
"fantasy-relevant participation," which mixes health with role. The snap sensitivity
and the high-usage restriction below bound how much is which.

## Is injury-proneness a stable trait?

- **(a) YoY correlation.** r = **0.422** across 1,213 consecutive-season pairs (371
  players); player-bootstrap 95% CI **[0.363, 0.477]**. Snap definition: r = 0.356.
- **(b) Beta-binomial MoM** (round-1 §1.4 style noise subtraction): p̄ = 0.604,
  Var(p̂) = 0.0963 across player-seasons, mean binomial noise ≈ 0.0092 →
  σ̂²_p = **0.0871** (SD 0.295 in availability-rate units). ICC-analogue
  ρ = σ²_p / (p̄(1−p̄)) = **0.364** (snap: 0.383). Test vs H0 "pure binomial + age"
  (parametric bootstrap: G ~ Bin(M, p̂(age)), age + age² logistic null, 1,000 sims):
  null σ²_p mean 0.0003, 95th pct 0.0011 — observed is ~80× the null's upper tail,
  **p < 0.001 (0/1000)**. H0 decisively rejected.
- **(c) Game-level logistic** (participation ~ age + G_{s−1} + G_{s−2}; 1,117
  player-seasons ≥ 2 years into career, 18,447 game rows, SEs clustered on 379
  players):

  | term | coef | se | z | p |
  |---|---|---|---|---|
  | const | 0.475 | 0.377 | 1.26 | .21 |
  | age | −0.055 | 0.014 | −3.85 | .0001 |
  | G_{s−1} | 0.112 | 0.007 | 14.92 | <1e−4 |
  | G_{s−2} | 0.044 | 0.007 | 6.33 | <1e−4 |

  Prior participation predicts participation two seasons out, over and above age; a
  17-game vs 8-game prior season is an odds ratio of ≈ e^{0.112·9} ≈ 2.7.

**Verdict:** yes, availability is a stable player trait far in excess of binomial +
age — but see anomaly 1: much of the stability in this universe is *role* persistence.

## LOSO arm (iv) — SV = θ*(fold-honest) × Ê[G]/M

Target: realized total REG PPR points / M (points per scheduled week). Availability
model per fold: binomial GLM G/M ~ age + G_{s−1} + G_{s−2} + missing-lag indicators,
fit on the 2015–2024 panel training years (all 300 rows, including <4-game seasons —
those are exactly the availability signal). θ* per row taken from the frozen
`results/loso_predictions.csv`; for the 9 non-in_fit rows the script-10 fold machinery
was replicated (fits unchanged) and verified to reproduce the frozen θ* to **max
|diff| = 3.6e−15** on all 291 overlapping rows before use. Baseline (i) = isotonic
m(log ADP) **re-fit on the same points-per-scheduled-week target** within each fold.

| eval set | arm | RMSE | mean Spearman | DM vs (i), t(9) | p |
|---|---|---|---|---|---|
| in_fit (291, primary) | (i) ADP-only refit | 4.346 | 0.396 | — | — |
| in_fit (291, primary) | (iv) SV | **4.230** | 0.408 | **+3.53** | **.006** |
| all 300 (sensitivity) | (i) ADP-only refit | 4.672 | 0.414 | — | — |
| all 300 (sensitivity) | (iv) SV | 4.581 | 0.415 | +1.51 | .16 |

Yearly loss diffs favor SV in 8/10 folds (primary). p_avail is compressed (mean 0.834,
sd 0.053) and its correlation with realized G/M is only 0.120 — see anomaly 1.

## Anomalies chased

1. **Is the SV win availability signal or a scale fix?** Replacing each player's
   p_avail with the fold mean (SV_const = θ* × p̄_fold, zero cross-sectional
   availability information): RMSE 4.252, DM vs (i) t = +1.92 (p = .087). SV vs
   SV_const head-to-head: **t = +0.46, p = .66**. So arm (iv)'s significant win over
   the market decomposes as: most of the RMSE gap comes from θ* itself plus the level
   rescaling to the per-scheduled-week target; player-specific availability adds a
   small further improvement (4.252 → 4.230) that is *directionally* positive and
   needed to push DM significance from .087 to .006, but is not itself significant.
   Honest headline: **SV beats the re-fit market baseline; differential availability
   prediction is not separately certified.** Mechanically sensible: among top-30-ADP
   WRs, prior-G varies little, so the model's p_avail has sd 0.053 against realized
   availability noise of sd ≈ 0.09 from binomial variation alone.
2. **Trait = health or role?** Restricting YoY pairs to seasons with mean targets ≥ 6
   on both sides (established starters, where targets ≥ 2 ≈ played): r drops from
   0.422 to **0.158** (414 pairs; same under snap definition, 0.146). Part of the drop
   is range restriction (mean p̂ 0.785 vs 0.604, variance compressed), but the sign is
   clear: the strong stability in the full universe is substantially *role*
   persistence (fringe players staying fringe), and pure health-proneness among
   established starters is a much weaker (though still positive) trait. This is
   consistent with anomaly 1's finding that availability adds little on a top-30 board.
3. **All-300 sensitivity volatility.** The all-rows DM (t = 1.51, n.s.) is driven by
   wipe-out seasons neither model can see coming: 2016 (+4.65 mean loss diff) is
   Keenan Allen (1 game) and Josh Gordon (0 games) landing on SV's side because its
   predictions sit slightly lower; 2019 (−1.86) is Antonio Brown (1 game, non-injury)
   and A.J. Green (0 games) landing against it. With realized outcomes of ~0 the fold
   result is a coin flip on which model happened to predict less; correctly reported
   as noise, not signal.
4. **Negative age coefficient with G-lags in the model** (−0.055/yr): survives the
   lag controls, i.e., aging predicts participation decline beyond last year's games —
   consistent with the round-1 §5 age findings; no conflict.

## Deviations from the plan text

- Ê[G]/**M** used instead of Ê[G]/**17** in SV: the scoring target is points per
  *scheduled* week and M = 16 for the 2015–2020 folds; a hard 17 would misalign those
  folds by 6%. Identical for 2021+ (and for the 2026 board).
- Availability GLM includes missing-lag indicators (rookies/no-history rows would
  otherwise be scored as G=0 histories, i.e., near-zero availability — visibly wrong).
- snap_counts 2012–2013 not downloaded (outside the 2014+ analysis/join window).

## Files

- `results/availability_table.csv` — 1,926 relevant WR player-seasons: G, M, p̂,
  mean_tgt, G_snap, p̂_snap, age.
- `results/loso_availability.csv` — 300 panel rows: θ*, p_avail, SV, re-fit market
  baseline, realized points per scheduled week. Round-1 loso files untouched.
- Script: `scripts/13_availability.py` (rerunnable; includes the data audit, all
  Part-2 tests, the LOSO arm, and the anomaly-chase decompositions).

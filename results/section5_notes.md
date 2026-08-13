# §5 notes — covariate structure (2026-07-14)

All parts use the §0 game-inclusion rule (REG only, targets ≥ 2) on ALL position == WR
players 2014–2025 (pre-specified: these are properties of the position, not of our 30;
top-30-only is a sensitivity). Team quantities joined from `stats_team_week_{year}.csv`
on season+week+team.

## (a) Age curve

Spec: Y_isg = f(age) + δ_year + a_i + ε; f = natural cubic spline df 4 (patsy `cr`), season
fixed effects, player random intercepts (MixedLM, REML). Sample: player-seasons with ≥ 8
included games and ≥ 3 targets/game → 16,566 games, 1,262 player-seasons, 383 players.
Age = (Sept 1 of season − birth_date)/365.25; no missing birth dates. Grid restricted to the
1st–99th age percentiles (21.2–34.4).

**APC caveat (stated per plan §5.1):** age − year = entry age − entry year is constant within
player, so with player intercepts any *linear* trend is attributable to age OR era OR
experience arbitrarily. Only the SHAPE of f (peak location, curvature) is interpreted.

**Shape (primary):** peak at **25.8** (f̂ = 10.54 PPG on the average-season scale);
decline per year −0.42 at 28, −0.71 at 30, −0.92 at 32; f̂(34) − f̂(peak) ≈ −5.3 PPG.
Rise from 22 to peak is ≈ +1.9 PPG. Note the sample's raw mean PPG of age-30+ seasons equals
the overall mean (10.70 vs 10.69) — the decline only appears once player intercepts absorb
"only good players survive to old qualifying seasons," which is exactly what the spec is for.

**aDOT interaction (secondary):** median split of season aDOT at 10.93. Peaks nearly
identical (25.8 low vs 26.1 high); declines diverge after 30: at 32, −1.14/yr (high-aDOT) vs
−0.82/yr (low-aDOT). Joint Wald on the 4 spline×aDOT terms: χ²(4) = 7.53, **p = 0.11** —
direction consistent with "downfield receivers decline faster," not significant at 5% in the
primary sample. (Top-30 sensitivity: χ²(4) = 13.65, p = 0.009, same ordering — but see the
survivorship caveat below before crediting it.)

**Top-30 sensitivity:** peak 26.0 but much flatter decline (−0.19/−0.32/−0.18 at 28/30/32).
Chased: this is survivorship/selection, not a property of stars. The top-30 universe is
conditioned on **2026 ADP** — i.e., on still being valued after the sampled seasons — so the
old seasons that made it into the sample are disproportionately the good ones (6 player-seasons
at age ≥ 30, 4.5% of the sample, averaging 15.5 PPG vs 15.0 overall; primary sample has 12.3%
old seasons). Conditioning on future status is exactly the leak the pre-specified primary
sample avoids; the primary curve is the position's age curve. Top-30 grid also ends at 32
(no support beyond), so its "decline at 32" is a boundary estimate.

Random-effects note: σ̂²_player ≈ 9.6 with residual ≈ 60 (primary); "Random effects covariance
is singular" warnings during optimizer exploration were transient; final fits converged
(lbfgs primary, nelder-mead top-30 — reported by the script).

## (b) Team-environment elasticity

Spec (plan §5.2, FWL): log(1+Y_isg) demeaned within player×season regressed on demeaned
log(team pass attempts) and team passing EPA; SEs clustered by team-week. Sample: all included
WR player-games with ≥ 2 games in the player-season (20,159 games, 6,300 team-week clusters).

| sample | β₁ (log att) | 95% CI | β₂ (pass EPA) | 95% CI |
|---|---|---|---|---|
| 2014–2025 | **0.550** | [0.514, 0.587] | **0.0166** | [0.0158, 0.0174] |
| 2021–2025 | 0.538 | [0.485, 0.591] | 0.0169 | [0.0157, 0.0181] |

Window-stable. Within player-season the two regressors are nearly orthogonal
(r = −0.067), so the coefficients are separately identified. Magnitudes: a 1-SD within-season
swing in log attempts (0.21) moves log(1+PPR) by 0.117; a 1-SD swing in team passing EPA
(9.7 points) moves it by 0.162 — the efficiency channel is slightly stronger than the volume
channel per SD of game-to-game variation. β₁ ≈ 0.55 < 1: production is inelastic to volume
swings (extra attempts spread across the route tree and come disproportionately in
trailing/garbage script).

Interpretation caveats (reported, spec kept as pre-registered): (i) reflection — the player's
own catches are part of team passing EPA (and his targets part of attempts), so β₂ partially
prices his own good games; treat as descriptive environment elasticity, not causal; (ii) FWL
demeaning uses no df correction for the absorbed player-season means — with 6,300 clusters the
effect on SEs is negligible; (iii) negative-PPR games (n = 27) clipped at 0 before
log1p.

## (c) Archetype clustering

Features per player-season with ≥ 8 included games (1,288 rows): aDOT, target share
(Σtargets/Σteam attempts), YAC/reception, TD/target, slot indicator (ngs_position == SLOT_WR).
Z-scored; Gaussian mixture (full covariance, n_init 10, fixed seed), k by BIC over 2–6.

**k\* = 3** (BIC: 103.5 / **−111.5** / −83.7 / −18.4 / 58.9 for k = 2..6). Profiles:

| cluster | n | aDOT | tgt share | YAC/rec | TD/tgt | slot | mean PPG | game SD | CV |
|---|---|---|---|---|---|---|---|---|---|
| 0 downfield secondary | 344 | 12.3 | .137 | 4.8 | .066 | 0 | 9.1 | 7.12 | .76 |
| 1 high-volume primary | 787 | 10.7 | .191 | 3.9 | .041 | 0 | 11.2 | 8.16 | .70 |
| 2 labeled slot | 157 | 8.9 | .175 | 4.5 | .046 | 1 | 10.8 | 7.83 | .70 |

- **ANOVA on player-season mean PPG: F = 30.4, p = 1.2e-13** — levels differ (primary/slot
  clusters ≈ 2 PPG above the downfield-secondary cluster).
- **Levene (game-level ANOVA on |Y − cluster median|): W = 36.4, p = 1.8e-16** — variances
  differ too. Ordering of absolute SD tracks the mean, but not proportionally: the downfield
  cluster has the *highest CV* (0.76 vs 0.70) — per point of expected production it is the
  boomiest/bustiest, consistent with TD-dependent deep usage. Levene on log(1+Y)
  (scale-robust variant) stays significant: W = 13.9, p = 1e-6.

**Measurement caveat chased (important):** `ngs_position` is missing for 60.2% of qualifying
player-seasons and the missingness is era-structured — coverage is decent for careers ending
2021–2025 (60–93%) but ≈ 3% for players active into 2026, i.e. essentially **all currently
active players are unlabeled**, including all 30 of our WRs. So slot = 0 for the top-30 mapping
is a missing-label artifact, not a fact, and cluster 2 is "labeled slot," a subset of true slot
seasons. Relatedly, the binary with near-zero within-cluster variance dominates the mixture
likelihood (headline BIC goes negative because of that near-degenerate dimension), so the k
selection is partly slot-definitional.

**Robustness variant (reported, headline unchanged):** GMM on the 4 continuous features only.
BIC again picks k = 3 (weakly: 14180.5/14172.9/14218.2). Structure reorganizes into
volume/style clusters with slot share ≈ 12% in each: (i) high-volume primaries (n = 526,
tgt share .229, PPG 14.0), (ii) low-volume YAC types (n = 355, YAC 5.4, TD/tgt .035, PPG 8.2),
(iii) low-volume low-YAC TD-dependent (n = 407, YAC 3.3, TD/tgt .059, PPG 8.2). Mean-PPG ANOVA
F = 504 — clusters separate PPG far more sharply without the slot label. Top-30 recent seasons:
23 primaries / 4 YAC / 3 TD-dependent.

**Top-30 mapping (headline clusters, most recent qualifying season; descriptive only, not an
input to anything):** 24 of 30 in cluster 1 (high-volume primary) — Nacua, Smith-Njigba,
St. Brown, Chase, Rice, Nabers (2024), Pickens, London, Olave, Lamb, A.J. Brown, Collins,
G. Wilson (2024), Flowers, Waddle, McMillan, Metcalf, Odunze, D. Smith, Jefferson, Egbuka,
McLaurin, McConkey, Evans; 6 in cluster 0 (downfield secondary) — Adams, Higgins, J. Williams,
Watson, Pierce, Burden; 0 in cluster 2 (unlabelable — see caveat). Full rows in
`archetypes.csv` (`is_top30_recent` flag).

## Files
- `results/age_curve.csv` — age grid × {primary, top30_only} × {all, low_adot, high_adot},
  f̂, SE, 95% CI (average-season scale).
- `results/elasticity.csv` — β₁, β₂, SEs, CIs, sample sizes, both windows.
- `results/archetypes.csv` — 1,288 player-seasons: features, headline cluster, PPG,
  `is_top30_recent`.
- Script: `scripts/05_section5_covariates.py` (rerunnable; seed fixed; prints convergence
  method per fit).

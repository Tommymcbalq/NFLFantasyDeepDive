# WR Valuation — Process Log & Explanation

End-to-end record of what was done, why, how, what came out, and where every number lives.
Executed 2026-07-13/14. The statistical protocol with full derivations is `EDA_PLAN.md`; this
document is the narrative companion — read them together. Equation numbers below refer to
EDA_PLAN.md.

**Governing principles (set before any fitting):**
1. No named-player anchors, no tuning toward expected results. Fit as pre-specified, report
   what comes out. Surprises are findings, not bugs.
2. Every spec and inclusion rule stated *before* looking at player-level output.
3. Claims of edge must survive multiple-testing control (BH-FDR) AND a temporal holdout.
4. Validation is leave-one-season-out; in-sample fit is never evidence.

---

## Step 1 — Data acquisition (`scripts/fetch_data.py`, `scripts/03_fetch_ffc_adp_historical.py`)

**What:** Three no-auth sources, all raw pulls cached under `data/` and never overwritten.
- **ADP** — FantasyFootballCalculator API, PPR, 12-team. Current 2026 board (1,737 real drafts,
  July 6–13 2026) → `data/adp/adp_ppr_2026_all.csv`; top 30 WRs → `wr_top30_adp_2026.csv`.
  Historical boards 2015–2024 → `data/adp/historical/`. **2024 is the last historical year
  available — the API has no 2025 ADP under any format probed. Nothing was imputed.**
- **Player game logs** — nflverse weekly stats, seasons 2014–2025, all players (raw cache
  `data/players/weekly_raw/`, ~19k rows/season, 148 columns: targets, receptions, yards, TDs,
  air yards, YAC, target_share, air_yards_share, WOPR, RACR, EPA, fantasy_points_ppr).
  The 30 board WRs' full careers extracted to `data/players/wr_top30_weekly.csv` (2,151 rows)
  and per-player files in `by_player/`. Coverage verified: the two oldest careers (Adams,
  Evans) begin in 2014, which is their actual rookie year (checked against draft_year).
- **Team weekly stats** — nflverse, all 32 teams, 2014–2025 → `data/teams/` (pass attempts,
  passing/rushing yards, EPA, CPOE per team-game).
- **Metadata** — nflverse players table (birth date, draft capital, rookie season, NGS
  position) → `data/meta/`. All 30 WRs matched on `gsis_id` (the join key everywhere).

**Why these sources:** flat files / open APIs, reproducible, and joinable on a stable ID.

## Step 2 — Protocol pre-registration (`EDA_PLAN.md`)

The full statistical plan — estimators, derivations, admission rules, holdout design — was
written and frozen before any analysis ran. Key design idea: **ADP is the market prior**;
players with thin data get shrunk toward it via an empirical-Bayes posterior (eq. 7) whose
weights are all *estimated* (game-level variance by experience tier, between-player variance
around market price by tier, effective seasons of data per player). The EDA both characterizes
the variance structure and estimates those hyperparameters.

## Step 3 — §0 Game-inclusion rule (pre-specified from aggregates only)

**Rule:** regular season only; exclude player-games with **targets ≤ 1**.
**Why:** the pooled target distribution shows a separated tail at ≤1 targets (1.9% of rows,
mean 1.9 PPR points, target share 0.038 vs population median 0.24) — a non-participation
mixture (injured-early / decoy / near-inactive). No snap counts exist in this table, so
targets is the participation proxy. Decided from aggregate distributions before any
player-level number was computed. Counts: 2,033 regular-season rows → 39 excluded.
Consequence made explicit: bust rates are "bust *given participation*"; missed-game risk is
handled separately (it surfaced later — see Step 9). Details: `results/section1_notes.md`.

## Step 4 — §1 Consistency profiles (`scripts/01_...`, `results/consistency_table.csv`)

**What:** per player — recency-weighted PPG level μ̂ (half-life 1 season; 0.5/2/∞ as
sensitivity), effective seasons n_eff, pooled within-season SD σ̂_W, and between-season
variance τ̂²_B **corrected for averaging noise** per eq. (3): the naive variance of season
means overstates year-to-year movement by σ²_W/G per season, which is exactly why injury-
shortened seasons masquerade as inconsistency. Plus floor q(.25)/ceiling q(.90) and
empirical-Bayes-stabilized boom (>20) / bust (<8) rates.

**Key results:**
- The eq.-3 correction is first-order: 30–170% of the naive between-season variance for most
  veterans. Both naive and corrected values are in the table so the correction is visible.
- 8 of 20 veterans have *negative* untruncated τ̂²_B — all within 1 SE of zero, i.e., the
  unbiased estimator behaving exactly as it should when true year-to-year variance ≈ 0.
  **Finding: for most established WRs, movement in season means is nearly all averaging noise.**
- The two large τ̂²_B values are career *arcs* (long trends), which the exchangeable-season
  model books as between-season variance; trend belongs to the age curve (Step 7).
- Blind top of the table: Nacua 21.3, Chase 20.2, St. Brown 19.6 (St. Brown with the lowest
  within-season SD of the elite tier — emergent, not tuned).

## Step 5 — §2 Variance decomposition (`scripts/02_...`, `results/variance_components.csv`)

**What:** REML mixed model (eq. 4) on 2021–2025 player-games:
Y = μ + player + player×season + team×season + game noise. Method-of-moments estimates
reported alongside; sensitivities: 2014–2025 with season FE, log(1+Y), with/without exclusions.

**Key results (headline REML):** σ̂²_P = 5.48, σ̂²_S = 2.48, σ̂²_T = 1.20, σ̂²_G = 69.93 —
**game noise is 88% of single-game variance**. Predictability ceiling (eq. 5):
**ρ_max = 0.41** (stable 0.39–0.45 across every sensitivity) ⟹ a history-only preseason
forecast is capped at R² ≈ 0.17. Of season-mean variance: 41% stable skill, 28% next-year
context, 31% irreducible.
- Anomaly chased: lag-1 same-player covariance (7.74) > σ̂²_P ⟹ player×season deviations
  persist year to year; an *adjacent-season* predictor's ceiling is ≈ 0.58. The 0.41→0.58 gap
  is the room covariates/market information have.
- σ²_T is weakly identified (only 20 team-seasons with ≥2 board WRs); estimators disagree on
  the σ²_S/σ²_T split but agree on the sum.
- Residuals right-skewed (+0.78) as expected; log1p overcorrects — identity scale kept.

## Step 6 — §4 Reliability gate (`scripts/04_...`, `results/reliability_table.csv`)

**What:** decides which stats are allowed to be covariates. Split-half (odd/even weeks,
Spearman–Brown to full-season, eq. 8) and year-over-year correlation with player-bootstrap
CIs, on ALL WRs 2014–2025 (reliability is a property of the stat, not of our 30 players).
Pre-registered admission rule: ρ_full ≥ 0.5 AND YoY CI excludes 0.

**Verdicts:** ADMIT target_share (.90/.70), air_yards_share (.87/.71), WOPR (.89/.71),
aDOT (.81/.67), PPG (.80/.64); RACR marginal (winsorized form only — its raw CI was blown up
by a handful of gadget seasons with aDOT < 2). **REJECT yards/target (.33), TD/target (.24),
receiving EPA/game (.45)** — TD rate is luck at season resolution, now with numbers attached.
Top-30-only reliabilities are lower — traced to range restriction (signal SD compresses,
noise doesn't), not a data problem.

## Step 7 — §5 Covariates (`scripts/05_...`, `results/age_curve.csv`, `elasticity.csv`, `archetypes.csv`)

- **Age curve** (spline df 4 + season FE + player RE; 383 WRs; shape identified, linear trend
  not — age–period–cohort caveat per plan §5.1): **peak 25.8**, decline −0.42 PPG/yr at 28,
  −0.71 at 30, −0.92 at 32. "High-aDOT receivers decline steeper" is directional (−1.14 vs
  −0.82/yr at 32) but p = 0.11 in the honest sample; the top-30-only version (p = .009) was
  rejected as survivorship-contaminated (conditions on making a 2026 board = future info).
- **Volume elasticity** (within player-season via FWL, so no selection): β₁ = **0.55**
  [0.51, 0.59] on log team pass attempts; team-EPA channel slightly stronger per SD
  (reflection caveat: own production is inside team EPA — upper bound).
- **Archetypes:** GMM, BIC picks k=3; mean PPG differs (F = 30.4) and — more importantly —
  **variance differs (Levene p ≈ 2e-16)**; downfield cluster most volatile per unit of level.
  Major catch: the NGS slot label is ~0% populated for active players, so the "slot" cluster
  is label-definitional; **slot label banned downstream**. Continuous-only robustness
  clustering reproduces k=3.

## Step 8 — §3 Heteroskedasticity by experience (`scripts/06_...`, `results/heteroskedasticity.csv`)

**What:** location-scale model — does game-level residual variance differ by experience tier?
Gamma-GLM headline (E[e²] = σ², dispersion 2), Harvey log-e² regression as check; all WRs
2014–2025, ≥3 targets/game seasons.

**Finding — the pre-stated expectation FAILED, informatively.** Rookie variance multiplier
**0.844** (CI 0.72–0.94): rookies are *less* volatile game-to-game than veterans, jointly
significant (p = .005). Chased to a level effect: variance scales ≈ σ² ∝ μ^1.4, and
controlling for scoring level the tier multipliers collapse to ~1.0. **Rookie uncertainty is
between-player (which rookie did you draft?), not game-level** — it lives in τ² and n_eff,
which is exactly where the shrinkage design already puts it. Spec unchanged; reported as-is.

## Step 9 — §6.1 Market prior (`scripts/07_...`, `results/market_prior.csv`, `tier_variances.csv`)

**What:** historical panel — top-30 WRs by ADP each year 2015–2024 joined to realized
same-season PPG (300/300 name-matches after three documented identity resolutions, including
catching a silent match to a 1990s player of the same name via an impossible 0-game season).
Fit m(ADP) = isotonic monotone-decreasing curve in log ADP (~20.5 PPG at the top to 8.7 at
ADP 75; beats OLS RMSE 3.32 vs 3.40). τ̂² of residuals by tier: rookie 24.5 (n = 4 — nearly
unidentified; boards almost never contained rookies), soph 7.9, vet 11.3.
**The expected ordering rookie > soph > vet FAILED** (soph < vet, n.s., Levene p = .39;
traced to veteran injury/age-cliff seasons vs durability-selected sophomores). Used as
estimated — no ordering imposed.

## Step 10 — §3.4 Blind valuation (`scripts/08_...`, `results/valuation_2026_blind.csv`)

**What:** eq.-7 posterior for the 2026 board: θ* = (1−B)·μ̂ + B·m(ADP), with B = V/(V+τ²),
V = σ̂²(tier)/n_eff. Covariate-free, zero manual adjustments.
**Results:** B ∈ [0.56, 0.84] — market-dominated, as it should be with only ~2–3 effective
seasons per player. Risers vs ADP: Rice +8, Adams +8, Evans +6, Nabers +5; fallers: Waddle −4.
Honest diagnostic: the risers share per-game strength with availability risk — PPG-given-
participation can't see missed games, the market can. That observation motivated adding
prior-season games played to the Step-11 covariate set, registered *before* that regression ran.

## Step 11 — §6.2 Edge regression (`scripts/09_...`, `results/edge_regression.csv`, `edge_holdout.csv`)

**What:** regress market residuals R = realized PPG − m̂(ADP) on preseason-knowable
covariates, restricted to Step-6 survivors + demographics + the pre-registered availability
term. Survival requires BOTH BH-FDR (q = 0.10, clustered-by-season SEs) AND improving
squared-error prediction on the 2023–2024 temporal holdout (fit 2015–2022; the plan's
2023–2025 was amended to 2023–2024 because no 2025 ADP exists).

**Result: NOTHING survives.** Two terms pass FDR — rookie (β ≈ −6.9) and rookie × new-team
pass EPA — but both rest on the n = 4 rookie cell and both *fail* the holdout (no
out-of-sample improvement). Age, team change, prior target share/aDOT/WOPR, prior games
played: all n.s. **Within this covariate set and panel, no systematic market mispricing was
found. Reported plainly; no fishing beyond the pre-specified Z.**

## Step 12 — §7 LOSO validation (`scripts/10_...`, `results/loso_scorecard.csv`)

Leave-one-season-out over 2015–2024; for each held-out year, every hyperparameter (m, τ², σ²)
refit without it and every player's μ̂/n_eff rebuilt from strictly-prior data.

| predictor | RMSE | mean within-yr Spearman | DM test vs ADP-only |
|---|---|---|---|
| (i) ADP-only m̂(ADP) | 3.564 | .461 | — |
| (ii) blind posterior θ* | 3.463 | .467 | t = 2.68, **p = 0.025** |
| (iii) θ* + FDR edge terms | 6.469 | .460 | worse (p = .35) |

**Headline: the blind shrinkage posterior beats blind ADP out of sample**
(≈ 2.8% RMSE, significant at the pre-registered paired test) — the gain comes from the
empirical-Bayes blend of a player's own recency-weighted history with the market curve, not
from any clever covariate. Adding the FDR-passing-but-holdout-failing edge terms makes
predictions much worse — the multiple-testing + holdout discipline did its job.

## Step 13 — §6.4 Final 2026 valuation (`scripts/11_...`, `results/valuation_2026_final.csv`)

Since no edge term survived, **final = blind posterior**, stated per-row in the file.
Top 5: Nacua 20.2, Chase 19.8, St. Brown 18.6, JSN 18.1, Lamb 17.8. Posterior SD ≈ 2.5–2.8
PPG on every row — the honest error bar from Step 5's variance structure.

---

## Status & open items

- **Steps 11–13 VERIFIED (2026-07-15):** scripts 09–11 rerun byte-identical; DM test
  confirmed t(9 df) on the 10 yearly mean loss diffs. The "2015-fold anomaly" traced: it
  lives entirely in the dead candidate (iii) — a degenerate 2-rookie β_{-2015} fit (the
  same n = 4 rookie cell condemned in Step 11) — plus one leakage bug found and fixed in
  (iii)'s training target (full-sample instead of leave-Y-out isotonic residuals; RMSE(iii)
  6.609 → 6.469, predictors (i)/(ii) and the DM headline bit-identical). Predictor (ii)
  actually *wins* the 2015 fold; left-truncated careers get μ̂ biased +1.5 PPG but
  *deflated* n_eff (=1) shrinks them harder to market — conservative, not overconfident.
  Excluding 2015/2016 folds leaves the DM verdict intact (p = .056/.020/.046 for
  excl-2015/excl-2016/excl-both). Details: `results/section7_notes.md`, `section6b_notes.md`.
- 2025 ADP missing at FFC — an alternate source would extend the panel and holdout by a year.
- Candidate next analyses: durability/games-played as a *modeled outcome* (not just a
  covariate); the adjacent-season ceiling (0.58) says persistent player×season structure is
  forecastable in principle — target-share-based projection of μ rather than raw PPG history;
  rookie valuation needs a different data design entirely (n = 4 rookies in ten years of
  ADP boards carries no information).

## File map

| layer | files |
|---|---|
| protocol | `EDA_PLAN.md` (math), `PROCESS.md` (this), `CLAUDE.md` (project brief) |
| raw data | `data/adp/`, `data/players/weekly_raw/`, `data/teams/`, `data/meta/` |
| derived data | `data/players/wr_top30_weekly.csv`, `by_player/`, `data/meta/wr_top30_meta.csv` |
| scripts (rerunnable, in order) | `scripts/fetch_data.py`, `01`–`11` |
| results + per-step notes | `results/*.csv`, `results/section{1,2,3,4,5,6a,6b,7}_notes.md` |

---

## Round 2 (2026-07-15) — availability, situation change, better data arms

Pre-registered in `EDA_PLAN2.md`; executed by three parallel researchers; round-1 arm (ii)
reproduced to machine precision before any comparison. Full results in REPORT.md Part II and
`results/section{A,B,C,D}_notes.md`. One-line outcomes:
- **§A**: availability is a stable trait (ICC ≈ .36, p < .001; partly role persistence);
  SV = θ*·Ê[G]/M beats ADP-only on points per scheduled week (DM p = .006) — the gain is
  θ* + participation scaling, not differential injury prediction.
- **§B**: team change costs −1.0 PPG, vacated targets pay incumbents +1.9 (both real,
  within-player); the market prices all of it (B3 edge test: full null).
- **§C**: age-detrended μ̂ fixes the old-career bias it targeted but nets to a tie (p = .983).
- **§D**: usage-projection arm underperforms on a market-selected board (p = .168, worse).
- **Board unchanged**: `valuation_2026_v2.csv` restates round 1 with the verdict recorded.
New data: `data/snap_counts/`, `data/injuries/` (2014–2025), `data/derived/` situation tables.
Scripts 13–16; new results CSVs per section.

## Round 3 (2026-07-16) — 2026 context + teammate coherence

Pre-registered in `EDA_PLAN3.md`. New data: Sleeper current teams (`data/sleeper/`), 2026
vacated shares (`data/derived/vacated_2026.csv`). Board movers: A.J. Brown→NE, Waddle→DEN,
Evans→SF. Outcomes: §E context-adjusted data arm NOT adopted (DM vs (ii) p=.44 — the market
arm already carries the move at weight B; adjusting μ̂ double-counts). §F: implied duo target
shares flag LAR Nacua+Adams at the 94.9th pct of like-for-like sums (real risk flag, lands on
Adams); F2 market test null → F3 constraint arm not run per the fixed decision rule. Board
unchanged (`valuation_2026_v3.csv`). Scripts 17–18; REPORT.md Part III.

## Round 4 (2026-08-09) — RB universe, the aging curve across eras, Vegas context, views overlay

Pre-registered in `EDA_PLAN4.md` (with one dated in-flight amendment, §I3, recorded after §I1
sourcing and before any §I3 fitting). Executed by four researchers in parallel. New data:
ADP refreshed to the Aug 1–8 pull (n = 5,187 drafts vs July's 1,737, saved dated); weekly player
stats and team-week stats extended back to **1999**; RB top-30 universe; `derived/age_panel_long.csv`
(9,546 WR/RB player-seasons, 1999–2025); `data/vegas/` win totals 2015–2025 + 2026 board.

**Reproduction gate first.** Scripts 01, 06–11, 17, 18 re-run on the July inputs reproduce all 19
round-1–3 result CSVs byte-identically, and `V_final_v3` to max |diff| = 0.0. Nothing was
overwritten; every round-4 output is a new dated or position-suffixed file.

One-line outcomes:

- **ADP refresh (WR).** Board stable: Spearman(July θ*, Aug θ*) = .977, RMS Δθ* = 0.469, max
  |Δrank| = 6. DJ Moore (BUF) and Sutton in; Metcalf and Watson out. **Round 1's largest
  disagreement collapsed on its own** — Rice went ADP 27.3 → 11.4 and our edge fell from +8 to
  +2; the market came to him in three weeks. Adams' +8 did not move (now +9). 12 of 28 players
  moved *exactly* zero: the isotonic prior has 18 levels over ADP 1.4–75 and the WR2/WR3 stretch
  is nearly flat — Higgins fell 12.8 ADP slots and lost 0.32 PPG.
- **§G RB universe, full refit.** Every component re-estimated, nothing borrowed from WR.
  Touches ≤ 1 excludes 17.2% of RB player-games (WR: 1.9%). σ²_S is **2.3× WR's**; adjacent-season
  persistence of season means **.245 (RB) vs .570 (WR)**; game-level noise is *lower* for RB.
  τ̂²_B does not truncate at zero for RBs (4/17 negative vs WR 10/20). **LOSO: neither the data
  arm (DM p = .464) nor the availability arm (p = .699) beats market-only — the honesty clause
  fires and the RB board is market-anchored.** Chased: mean fold gain is 70% of WR's but the
  across-fold SD is 2.5× larger; MDE ≈ 1.85 makes the test ~4× short. The RB cliff cuts both
  ways (2020 Bell/Gurley/Ingram collapses vs 2023 Kamara/Conner/Montgomery surprises) and nets
  to zero. Availability is as real a trait for RB as WR (ICC .368 vs .364) but there is no θ*
  edge for it to scale.
- **§H aging — the pre-registered hypothesis is rejected, in the opposite direction.** A data
  defect was found and repaired first (nflverse `targets` is degenerate 2003–2008, which had
  silently emptied WR era 1 to 1999–2002; repaired via receptions × ρ_pos, raw retained, and the
  defective panel gives the same conclusions). **The drop-off age has not moved later.** WR cliff
  31.05 → 29.35 → 28.05 across eras, RB 28.65 → 28.30 → 26.95; era×age interaction not rejected
  (WR p = .185, RB p = .329). The one contrast clearing its CI is **WR steepening**: the 28→32
  slope is 2.2× steeper now (p = .033). The **exit hazard corroborates**: WR h(30) .231 → .259 →
  .377 (p = .007), RB h(30) .313 → .410 → .474 (p = .033) — careers end *earlier*, not later, and
  H2 and H3b agreeing is what the plan required. The pre-registered smooth check is weakly
  identified under player FE (cohort collinearity) and is reported as a failed check with cause.
- **§H4 workload carryover is mean reversion.** The pre-specified spec gives 200→350 prior
  touches ⇒ Δr = −0.513, but corr(prior touches, prior r) = .854 and prior r enters Δr with
  coefficient −1; controlling for it gives **+0.002 (−0.105, +0.117)**, and the placebo (*next*
  season's touches predicting *this* season's change) is +0.073 (p < .05), which no causal story
  permits. An informative null (±0.11), not an underpowered one.
- **§I Vegas.** Sourcing split: closing win totals obtained for 2015–2025 (32 teams, no gaps;
  validated against an independent nflverse capture at 181/191 exact, MAD 0.026 wins) — gate
  passed. Season point totals and player props are **not obtainable** historically (props begin
  2023-05-03); the tempting per-game `total_line` substitute is in-season data whose sum is
  look-ahead and was barred. **§I3 null** (all p ≥ .52, joint F p = .81, holdout fails). The
  decomposition is the finding: team quality is worth **+0.251 PPG per win** and **ADP already
  charges +0.194 of it (77%, p = .0085)**; the residual +0.057 is noise. Power bound: rules out
  any win-total channel worth more than ~26% of a residual SD per SD of team quality.
- **Joint FDR across the round-4 family** ({H5} ∪ {I3}, m = 11, BH q = .10): the single survivor
  is RB age² (p = .0023 vs threshold .00909). It **fails the second binding screen** — temporal
  holdout 14.611 vs 13.664 — and is fragile (HC3 p = .312, Huber p = .329, year-by-year sign
  flips in 2019/2020, joint Walds on the cluster-covariance rank boundary). Both screens are
  required, so **no age arm enters LOSO**; the curve informs §J only, labelled unvalidated.
- **§J views overlay built and validated.** π = the isotonic ADP→PPG curve at each slot
  (12.54–19.56 PPG, a 7.0-point total spread); Σ diagonal = tier residual variance minus the
  per-game sampling component σ̂²_W/Ḡ (σ_true 1.68–2.88). The **teammate off-diagonal was
  estimated and set to zero**: r = +0.016 over 71 same-team board pairs, cluster-boot CI
  [−0.321, +0.320], vs a random-within-year null of −0.041 (p = .69) — a power limitation, not a
  refutation, since the share-constraint prediction of ρ ≈ −0.3 sits inside that CI. Eight J4
  synthetic assertions pass, including that a view leaks to nobody under diagonal Σ and transmits
  *downward* to a teammate under a negative off-diagonal. No real views entered yet.

**Reconciliation.** §H5 built its own RB isotonic curve before §G3 existed. The two agree at
corr = .9997, mean |Δ| = 0.040 PPG, max 0.230, on identical PPG inputs; the difference is 296 vs
300 rows of inclusion. **§G3's `market_prior_rb.csv` is canonical**; `rb_market_prior.csv` is
retained as the independent check it accidentally provides.

**Corrected.** REPORT.md §3 said "8 of 20 veterans have negative untruncated τ̂²_B"; the frozen
`consistency_table.csv` always said **10**. Transcription error in the write-up, not the estimate;
fixed in place with a dated note.

**Open items carried to round 5.**
1. Eq. (7)'s V = σ²(tier)/n_eff has no term for how far next season's level moves from μ̂. For WR
   that is ≈ 0; **for RB it is ≈ 6 PPG², so V understates μ̂'s predictive variance by ~40% and B
   is too small.** Flagged by the §G researcher and deliberately *not* acted on post hoc — it is a
   round-5 pre-registration item.
2. σ²_T is negative for RB (MoM −1.68, REML pinned at 0 with a singularity warning), diagnosed to
   the backfield share constraint (teammate cross-products −2.46 same-week over 88 pairs) but
   resting on only 6 two-board-RB team-seasons. Right sign for the §J off-diagonal; needs the full
   RB population, not the board.
3. Sophomore RB excess volatility survives the level control (p = .022) where the WR analogue
   collapsed to 1.0. Mechanism untested.
4. 2026 win totals were stored with over prices only, so they cannot be de-vigged on the same
   footing as the history (a 4.5% overround was assumed and flagged). Re-pull the unders if §J
   leans on them.
5. Script numbering collided across parallel researchers (20–24 used by three streams; filenames
   differ so nothing was overwritten). Renumber before round 5.

## Round 4b (2026-08-09) — schedule strength

Pre-registered as `EDA_PLAN4.md` §K after §K0 sourcing and before fitting, as a **new FDR family**
(16 tests, BH q = .10) — the round-4 family {H5, I3} was closed, corrected and reported, so it was
not reopened. New data: `data/schedule/sos_history_2015_2026.csv` (12 seasons × 32 teams, zero
missing, four measure families × three windows, all preseason-knowable), `sos_2026.csv`.
Build validated against CBS (Spearman 1.00), an independent LeagueStation recompute (Pearson .967)
and Sharp Football (.88); a franchise-abbreviation trap (STL/SD/OAK vs LA/LAC/LV) that silently
dropped 128 opponent-games in 2015–2019 was found and fixed, then verified (291/291 and 286/286
rows joined, zero missing cells).

**Verdict: null. 0/16 survive BH, 0/16 beat the holdout, no arm enters LOSO.** Full write-up in
REPORT.md §28. Three things worth carrying:
1. **§K1's pre-registered power prediction was falsified** — predicted >10× underpowering, realized
   5.3×. The ceiling (0.061 PPG/SD) reproduced exactly; the error was importing §I3's MDE, which is
   invariant to SD(x) and therefore not transplantable between features with different error
   structures. Season-centering orthogonalizes x to season dummies, annihilating the season-common
   component of R and putting the cluster SE *below* iid (0.323 vs 0.613) — ~2.7× more precise than
   the design the number came from. Recorded as falsified, not restated.
2. **A false positive was caught and withdrawn**: contemporaneous positional SOS gave +0.76 PPG/SD,
   p = .013, but a defence's realized FPA contains the outcome player's own points — an i-specific
   inflation that does not average across opponents. Leave-own-team-out collapses it to −0.08/+0.03.
3. **The positional nulls are uninformative about matchups**, and are labelled as such. The
   attenuation chain (persistence ~0.25, then season aggregation) caps any lagged positional effect
   at ~0.14–0.21 PPG/SD, below every MDE in the family. §K bounds a preseason season-aggregate; it
   says nothing about in-season weekly matchup value.

Scripts: `25_sectionK_schedule_edge.py`. Results: `sectionK_notes.md`, `edge_schedule.csv`,
`edge_schedule_decomposition.csv`, `edge_schedule_clairvoyant.csv`, `schedule_sources.md`.

## Round 5 (2026-08-09) — §L positional conversion by draft cost

Pre-registered in `EDA_PLAN5.md` with the owner's hypothesis recorded verbatim; new FDR family
(8 tests, BH q=.10) plus holdout 2015–21 → 2022–24. **0/8 survive BH** (min raw p = .079 vs
threshold .0125); holdout sign holds 2/4, both on a disqualified definition and both against the
hypothesis. Full write-up REPORT.md §29.

- **(a) elite RB converts better than elite WR — not supported.** R1–2 top-12 conversion RB 53.7%
  (n=123) vs WR 52.5% (n=101); gap −0.011, p = .866, **MDE 0.190** so a true 10 pp edge would often
  be missed — no support for the direction, not proof of equality. In the owner's 10-team frame the
  point estimate runs 6 pp the *other* way (RB 53.2 vs WR 59.3).
- **(b) trending up — uninformative, not absence.** Logit slope +0.065/yr (p = .332) against
  **MDE ≈ 5 pp/yr ≈ 50 pp over the window**; pre-specified split +3.0 pp (p = .813, MDE 50.5 pp).
  MDE computed for this design, not transplanted (§28's lesson). Must be read against the RB cost
  cycle: **RBs were 2.1 picks cheaper in 2022–24**, the direction that inflates a hit rate for free.
- **(c) mid-round WRs the better buy — not supported; the informative null.** Position×bin
  interaction p = .964. Gaps by bin +1.2/−3.4/+1.0/−1.9/−0.4 pp. Under the value definition the WR
  edge is *largest* in R3–4, wrong shape before any test.
- **Mechanical catches, all favouring the hypothesis had they survived:** the value-return
  definition is positional by construction (at matched positional finish, WRs out-score RBs by
  +11.7→+42.3 raw points across tiers, so a shared threshold measures PPR volume, not beating
  price); a Panthers DE matched onto the 2015 WR board scoring a 0-point "top-12 hit", and Jordan
  Matthews tagged TE in 2015–16 — both broke the 12-slot budget (2015 showed 14 WR hits); and the
  mixed-model check was reporting the QB dummy as the RB effect via first-string term selection.
- **The one non-null structure, and it is not conversion:** at R1–2, RBs play 13.18 games vs WRs
  14.28 while PPG is near-identical (16.43 vs 16.89). The −20.7 pts/season total gap decomposes to
  **−18.6 from games, −6.1 from PPG** — ~90% of the elite-RB shortfall is availability, which §A
  already models. §L2's gap shrinks −0.148 → −0.070 on switching to PPG-given-participation.

**Contribution to round 6 (draft strategy):** the conversion input is positionally flat, so any
RB-vs-WR draft preference must come from scarcity/replacement-level weighting under the owner's
actual 10-team, 2-flex settings — not from RBs justifying their price more often.

Scripts `26_sectionL_conversion.py`, `27_sectionL_figures.py`; results `sectionL_notes.md`,
`conversion_rates.csv`, `conversion_tests.csv`.

## Round 7 (2026-08-18) — §S: the prior's era assumption, and two data repairs

Triggered by the owner's pushback on three claims, two of which were wrong.

**Repairs.** (1) FFC *does* now serve 2025 ADP (249 players, 8,470 drafts, window 08-25→09-01);
§35's "no 2025 data" was true at the July pull and false now. Panel 300→330 rows, 291→321 in fit,
10→11 folds, 30/30 matched. (2) Every historical FFC window is a late-Aug/Sep window; the 2026
board in use was a **July 6–13** window — a six-to-eight-week one-directional information mismatch.
Re-anchored to the 08-11→08-18 pull (6,809 drafts). Both additive; round-1 artifacts untouched.

**§S — the asymmetry.** The likelihood μ̂ is recency-weighted (h=1 season); the prior m(·) and
τ²(tier) pooled all seasons with **equal weight**. Never justified, only inherited. Tested.
- S1 in-sample: season trend in residuals −0.055 PPG/yr (p=.38 HC3, p=.51 clustered, 11 clusters);
  pre/post-2022 split −0.677 PPG (p=.068); 17-game-era split −0.437 (p=.239). Last three seasons all
  negative and worsening (2023 −0.04, 2024 −0.59, 2025 −1.27); board mean PPG 15.76 (2018) → 13.88
  (2025). Directional, underpowered, **not acted on**.
- S2 out-of-sample, 11-fold LOSO, weights w = 2^{-|t−Y|/h} (distance to the *evaluated* season —
  the hypothesis is local stationarity, and for the live 2026 board it collapses to pure recency).
  **All six weighted arms beat their flat counterparts on both RMSE and Spearman**, smoothly in h.
  θ*_{h=4} passes the pre-registered rule (RMSE 3.4334 vs 3.4671, DM p=.049); θ*_{h=2} is the RMSE
  minimum (3.4196, p=.097). **BH across the 3-arm family: 0/3 at q=.10, 2/3 at q=.20.**
  **Verdict: promising, not certified.** Board stays flat-prior. Confirmatory single-h test (h=2)
  pre-registered on the RB/TE/QB panels or 2026 as a genuine holdout.

**Corrections to the record.** §E is not "context is null" — it tested a *mechanical* two-coefficient
arm, and its double-counting mechanism applies only to information the market has already absorbed.
A hand-priced view has never been tested; §J is its home. Also stated explicitly for the first time:
every isotonic fit is **within-position**, and cross-positional value is decided in §M/§O from roster
demand under the 10-team settings, never from ADP.

**Data-hygiene rulings.** `times_drafted` is not a precision weight (erratic at the top of every
board). Historical files are immutable (2015 re-pull byte-identical, max |ΔADP| = 0.0).

**Market moved hard July→August**: Herbert +31.0, Kraft +30.4, Mahomes +24.5, LaPorta +17.0 against
Stafford −23.2, Brooks −20.1, C. Williams −17.2, DJ Moore(BUF) −16.5, Nabers −15.8, Rice −12.8;
new to top 120 on new teams: K. Walker KC 23.4, Diggs WAS 93.7, Deebo SF 110.8. Awaiting the owner's
injury/camp reports to attribute the moves.

Scripts `49_refit11_market_prior.py`, `50_prior_era_weighting.py`, `51_loso11_prior_weighting.py`;
results `market_prior_11yr.csv`, `tier_variances_11yr.csv`, `prior_era_weighting.csv`,
`prior_residual_by_season.csv`, `loso11_scorecard.csv`, `loso11_predictions.csv`,
`adp_movement_2026_jul_aug.csv`.

### Round 7 addendum (same day) — a join bug, a source-lag measurement, and the owner's call

- **Bug, mine.** The first movement table joined on raw `name`, not the project's `norm_name`.
  FFC renamed `Kenneth Walker III` → `Kenneth Walker` between pulls, so he was falsely flagged
  "new to the board". He was 22.6 on KC in July and is 23.4 now — **he did not move**. Genuine new
  arrivals in the top 120 are only Diggs (WAS 93.7) and Deebo (SF 110.8); zero team changes
  between pulls. Rebuilt in `52_adp_movement_fixed.py` / `adp_movement_2026_fixed.csv`.
- **Kraft was not traded.** ACL tear 2025-11-02; activated off PUP 2026-07-31; expected to play
  Week 1 with reps limited "probably until halfway through the season". The +30.4 ADP fade is that.
- **FFC lag quantified.** FFC serves a **7-day trailing window** — structurally ~3.5 days stale by
  construction, which is material in a market that moved 30 slots in six weeks. Pulled ESPN live
  (`adp_espn_2026_20260818.csv`, 300 rows) as a same-day cross-check. Spearman .952 on 121 matched
  top-150 players, but the disagreements are large and concentrated: Tyson FFC 99 / ESPN 198,
  Godwin 84/158, Harvey 86/152, Goff 107/171, Diggs 98/154, Warren 64/117, Pollard 70/119.
  **Kraft: FFC rank 106 vs ESPN 124** — FFC is still 18 slots too expensive on him, the lag in the
  exact direction predicted. Saved `adp_ffc_vs_espn_20260818.csv`.
- **Owner call: use the recency-weighted prior.** Recorded as a decision on a judgment margin
  (BH clears q=.20 not q=.10), not as a certified edge. `53_board_2026_recency.py` rebuilds the WR
  board with h=2 on the 08-11→08-18 window. τ² falls (vet 11.24→9.67, rookie 19.22→14.01) so B
  drops ~.02 — slightly less deference to the market. Effect on the board is small: mean |Δθ*|
  0.310 PPG, max 0.780 (Odunze −0.78, Rice −0.69, DJ Moore −0.68, Jefferson +0.50), 13/30 ranks
  move, max move 2 slots. `board_2026_recency_h2.csv`.
- **Not a staleness issue, stated once:** the LOSO panel is historical by necessity — a forecast
  cannot be validated on a season that has not happened. What must be fresh is the *applied* ADP
  and the *weighting* of the fitted prior. Both now are.

## Round 7c (2026-08-18) — camp-refreshed views on a re-anchored board

Board rebuilt (`55`): WR+RB top 30 on the 08-11→08-18 window, both priors refit on **11 seasons**
with **h=2 recency weighting** (RB refit in `54` so the two positions share a panel and a weighting).
Views v2: 31 total, **8 updated/new**. Targets picked by the §39.7 source-gap residual, not by feel.

Research findings that moved prices: **Hubbard hamstring/MRI/week-to-week** (→ new q 8.80, and the
mechanism behind Brooks' −20.1 rise); **Pittsburgh is a true 1A/1B**, McCarthy "four-down backs"
both, alternating first-team reps (Warren and Dowdle faded symmetrically to 9.80 each; they land
0.01 apart post-overlay, the intended check); **Pollard is the confirmed lead**, so his trim is
smaller and low-confidence — explicitly not a committee fade; **DeVonta Smith hamstring**,
**Burden groin/out for preseason**, **Egbuka minor toe**; **Nabers 9/10 practices, team drills in a
red non-contact jersey, Week 1 target but no confirmed date** → faded to 14.00 vs π 14.88 on
ramp-up risk, not on the knee.

**Structural limitation surfaced, not worked around:** the strongest research findings — Price SEA,
Golden GB, Tate TEN, Hunter Henry NE, Dart, Nix — all sit in the 60–160 ADP band and **cannot be
expressed on a 60-player board**. Sutton and Johnston were retired for the same reason (price moved
them out of the top 30), theses intact. Extending §J onto §P's deep board is the next step.

Also recorded: a fade relative to a *prior view* can still be a raise relative to π (DeVonta Smith
q 15.69→14.90 vs π 14.11 shows +0.16). Shift tables must be read alongside π.

Scripts `54`–`56`; results `board_2026_pi_sigma_h2.csv`, `views_2026_v2.csv`,
`board_2026_v2_with_views.csv`, `adp_source_gap_residual.csv`.

## Round 7d (2026-08-18) — owner's Aug 11-18 digest lands; views v3

`fantasy_news_aug1118_2026.md` (owner-supplied, independently fact-checked, ✅/⚠️ tagged).
**15 views changed** (`views_2026_v3.csv`, 39 total), including one falsification of mine and one
reversal of a fade I had made hours earlier on thinner information.

- **FALSIFIED — McCaffrey.** v04_cmc read "injury fear overpriced; healthy all camp". Digest: still
  not practicing ~a week out, "tightness", no diagnosis or timeline, Shanahan "could be a month" if
  he plays through, contract hold-in theory floated. 19.80 -> **17.20, low**. He falls #2 -> #3.
- **REVERSED — DeVonta Smith.** My own 08-18 fade (15.69 -> 14.90/low) was made off a hamstring
  report with no return date. Digest: **returned to practice Aug 17, arrow up.** Restored to
  **15.50/medium**; he moves +0.69 instead of +0.16.
- **DOWNGRADED — Hampton**, 16.30/HIGH -> **14.80/medium**. The old view's core claim (competition
  evaporated) is contradicted: McDaniel signalled a committee, "Hampton, Mitchell, Vidal all play,
  hot hand". This had been the board's largest positive shift (+1.36 -> +0.10); rank 17 -> 30.
- Also cut: Skattebo (Tracy a true co-starter, not a bell-cow), Love (high ankle sprain, Week 1
  uncertain), Judkins (Browns QB battle suppresses all pass-catcher value), Rice (knee rust).
  New fades: Breece Hall (groin Aug 17, Allen took 2 red-zone TDs), Tuten ("Tuten OR Rodriguez"
  committee), Jameson Williams (shoulder), Henderson/Stevenson (symmetric NE committee pair).
  New raises: **Olave** (Tyson ~2 months / short-term IR, 4yr/$132M, sole target hog) — the cleanest
  in the digest; Irving (fully cleared); Chase Brown (grip reinforced).

**The §39.7 source-gap method validated on its two largest residuals.** FFC had Jordyn Tyson at
rank 99 vs ESPN 198, and RJ Harvey at 86.5 vs ESPN 152. The digest confirms **Tyson is ~2 months
out** (hamstring, possible short-term IR) and **Jonah Coleman is pushing Harvey**. In both cases
ESPN was right and FFC's 7-day trailing window was lagging a real event. This is the first direct
confirmation that the residualised cross-source gap identifies news FFC has not absorbed.

**Data conflict logged, unresolved:** the digest keeps **Deebo Samuel as a Commander** (consistent
across its sources, and it declined a lone claim SF re-signed him); FFC's 08-18 board lists him
**SF**. He is off the top-30 board so nothing downstream depends on it, but FFC's team field is
suspect here. Also confirmed by the digest and already correct in our data: Kenneth Walker III is
a Chief.

Scripts `57_run_views_v3.py`; results `views_2026_v3.csv`, `board_2026_v3_with_views.csv`.

## Round 7e (2026-08-18) — the process fix: views are owner-authored

The owner rejected two of my 08-18b magnitudes and, more importantly, the fact that I was setting
magnitudes at all. **Both objections checked out numerically before being acted on.**

- **Hampton restored to 16.30/high** (θ̄ 15.96, rank 17). My cut to 14.80/medium read McDaniel's
  "Hampton, Mitchell, Vidal all play, hot hand" as a ceiling cut. The owner reads it as coachspeak
  that does not change the role. That judgement was mine, not his; withdrawn.
- **Olave cut to 16.00/low** (θ̄ 15.35, rank 23). Two binding owner constraints: Tyson's ~2-month
  absence is **reported, not confirmed**, so it cannot be priced as fact; and a WR view must not
  put a receiver above the RBs going in round 2. My 16.40/medium violated the second — θ̄ 15.80 vs
  Barkley 15.72. Verified fixed: Olave now sits below Barkley (15.72) and below Hampton (15.96).

**Standing rule adopted, and it supersedes prior practice:** Claude brings dated evidence, π, and
the mechanical leverage; **the owner sets q and confidence.** Nothing enters the board on a
magnitude Claude chose. Measured leverage of the overlay, useful when posing the choice:
**≈80% of (q − π) at high confidence, ≈50% at medium, ≈20% at low.** Confidence is the lever for
how *confirmed* an item is, not how much one likes it — which is exactly the distinction the Olave
correction turned on. Owner-set values carry `OWNER-SET <date>` in the rationale and state where a
Claude judgement was overridden.

This is a design correction, not a preference: §J specifies views as the owner's priced opinions
with Ω *declared, not fitted*. Authoring q from my own research made them my projections wearing
the layer's clothes, and they were wrong in both directions on the same day.

## Round 7f (2026-08-18) — Chase/Burrow tested, London faded, and the flat-top defect named

**Owner hypothesis tested before pricing: Chase with vs without Burrow.** Split his game logs on
Burrow starts (>=10 attempts). **2025: 20.12 PPG in 8 games with Burrow vs 19.08 in 8 without - a
gap of 1.04**, against 2023's 18.97/12.17 gap of 6.80. **Not supported:** the 2025 dip is not mainly
a backup-QB artifact; Chase was down with Burrow on the field (20.12 vs 23.71 in 2024). Rebuilding
mu-hat on Burrow-only games gives 20.86 and pi 19.11 - rank 5, not the owner's "fourth at worst".
Recorded as a falsified hypothesis, and the adjustment was **not applied to Chase alone**: doing so
would be tuning toward a named player. "QB-availability-adjusted mu-hat" is logged as a candidate
**board-wide arm** requiring LOSO validation - distinct from the failed §E context arm, because §E
pushed a population-average effect into the data arm and double-counted the market, whereas this
fixes a within-player measurement problem (mu-hat contaminated by backup-QB games).

**Owner-set views (both q and confidence supplied by the owner, per the round-7e rule):**
- **Ja'Marr Chase 19.30/high** -> theta 19.22, **rank 3**. Explicitly an override of the recency
  signal, not a data correction; scoreable - if Chase repeats ~19.6 PPG the view is wrong.
- **Drake London 14.50/high** -> theta 15.05, **rank 24** from pi 17.23. Largest disagreement with
  the market on the board: ADP 10.2 against a model rank of 24. Context: Atlanta's QB job is an
  unresolved Tua/Penix competition under new HC Stefanski, Pitts is coming off an All-Pro season,
  Branch/Dotson/Zaccheaus add competition. q sits just below his own mu-hat of 15.66.

**Structural defect named, and it caused BOTH complaints.** m(ADP) is **identical at 18.08 for every
WR from ADP 3.1 to 10.2** - the isotonic curve is flat across the top of the board, so price
separates nobody there and mu-hat does all the work. That is why London (mu-hat 15.66, lowest in the
top nine) was carried to 9th on price, and why Chase fell to 6th on a single soft season. Both
complaints are symptoms of the same thing. Fixing the top-of-board resolution is now the highest-value
open modelling item, ahead of extending §J to the deep board.

## Round 7g (2026-08-18) — §T: QB-adjusted mu-hat tested board-wide and REJECTED

Built from the owner's Burrow question, generalised to an estimator hypothesis: is mu-hat biased by
games the team's primary QB missed? Pre-registered definitions (primary QB = most REG attempts in
the team-season; starter game = that QB with >=10 attempts; <4 starter games falls back to the
unadjusted mean). Only the mu-hat input changes - sigma^2, tau^2, m(.), B and the inclusion rule are
identical, so the comparison is clean.

Coverage: 84.9% of player-games are already starter-QB games; 36.1% of season rows fell back; mean
lift on the rest +0.073 PPG. **11-fold LOSO: RMSE 3.4432 vs the incumbent's 3.4196, Spearman .4642
vs .4729, DM t = -1.642 (p = .132). Worse on both metrics. Rejected on both prongs of the
pre-registered rule.**

**The finding is the rejection.** The counter-argument recorded before testing won: conditioning on
the starter playing estimates E[PPG | QB healthy], but the forecast target includes whatever QB
absence actually occurs. QB-absence games carry real predictive information - either the situation
recurs (thin backup rooms, injury-prone starters persist) or the player is revealing a genuine
dependency. **Unadjusted mu-hat is the right estimand, not a contaminated measurement.**

Consequence, stated plainly: **Chase's rank-6 was not a measurement artifact.** mu-hat was correct
and the owner's disagreement belongs in a view, which is where it is (v40_chase, 19.30/high,
owner-set, scoreable). Also: the Burrow split itself did not support the original claim - 2025 was
20.12 with vs 19.08 without, a 1.04 gap, against 2023's 6.80.

**Still open and NOT explained by this:** m(ADP) = 18.08 for every WR from ADP 3.1 to 10.2. The flat
isotonic top remains the live defect - it is why one soft season moved Chase four places and why
London rode price to 9th on a mu-hat of 15.66.

Script `58_qb_adjusted_mu.py`; results `sectionT_qbadj_loso.csv`. REPORT.md §41.

## Round 7h (2026-08-18) — §U: pair continuity as a PRECISION adjustment. Also rejected.

Owner's critique of §T accepted: a generic "did the primary QB play" flag cannot express a pair
interaction (Chase/Burrow 61 shared games vs Nabers/Dart 1). His argument implies the correction
belongs in the precision, not the mean: n_eff_adj = n_eff*(lam + (1-lam)*share). mu-hat untouched.
Reference QB = the team's incumbent (primary QB in Y-1) - owner's rule, and strictly
preseason-knowable, so no hindsight about in-season QB injuries.

Continuity is real: top-20 WRs for 2026 run Rice/Mahomes 1.00, Nacua/Stafford .98, Chase/Burrow .65,
Nabers/Dart .09, and Jefferson, A.J. Brown, London, Garrett Wilson at .00. 5 of 20 below .25.

**Result: rejected. RMSE 3.4228 / 3.4275 / 3.4365 for lam .50/.25/.00 vs incumbent 3.4196; all DM
statistics NEGATIVE; 0/3 survive BH q=.10.** Degradation is MONOTONE in lam - the harder a broken
pairing is discounted, the worse the forecast. That ordering is the informative part: a mistuned
magnitude gives noise around zero, an ordered decline says the direction is wrong.

**Mechanism check on the 38 treated rows (share<.25): MAE of mu-hat 3.15 vs m(ADP) 3.37.** On exactly
the players whose pairing broke, their own history beats the market price. The hypothesis needs the
opposite. Finding: **receiver production is more portable across QB changes than the pair-interaction
story predicts** - consistent with §B3 (the market prices situation changes) but adding that pricing
them is not the same as pricing them better than the player's own record.

Honest caveat: n=38 treated, not decisive. What lifts it above "underpowered" is three independent
signals agreeing - every arm loses, the loss is ordered in lam, and the subgroup MAE contradicts the
mechanism head-on.

**What survives of the critique:** the specification objection to §T stands - a generic flag cannot
express a pair interaction, and §T should not have been the only test. What fails is the consequence.
Mean route (§T) and precision route (§U) both reject, in opposite corners of the estimator.
`qb_continuity_2026.csv` is retained as the right input to a VIEW, where the owner prices a specific
pairing, not to an arm that prices all of them identically.

Script `59_sectionU_pair_continuity.py`; results `sectionU_scorecard.csv`,
`sectionU_continuity_loso.csv`, `qb_continuity_2026.csv`. REPORT.md §42.

## Round 7b (2026-08-18) — §S1-§S4: the mean put on trial, and the board rebuilt in one pass

**The gap this closed.** Nine arms had been tested across six rounds and all had proposed something
to *add to* the data arm. None had touched mu-hat itself. EDA_PLAN7 put eight candidate summaries
of a player's history head-to-head, replacing ONLY mu-hat inside eq. (7).

**Operational pre-spec written before fitting** (results/sectionS_notes.md): the game-level arms
carry weight w_s/G_s per game, which makes their weighted MEAN algebraically identical to the
incumbent, so the arms differ in the functional and nothing else. Harness validated first - it
reproduces §P4's published LOSO numbers to four decimals.

**VERDICT: no replacement.** Nothing clears p<.05 AND an RMSE improvement AND the temporal holdout.
The only BH rejection is p60 at WR, p=.0010, in the WRONG direction, 0/10 folds. Ties go to the
incumbent by the pre-fixed rule. Nulls reported as uninformative per §S3: realised MDEs 0.28-1.12
against observed effects 0.01-0.92, and arms 3/4/8 sit inside their own MDE.

**Two pre-registered expectations, one confirmed and one FALSIFIED.**
- Candidate 6 (mu-hat over seasons with >=12 games) was the one with a mechanism and it came out
  significantly WORSE (RB -0.513, p=.047). Chased it. §P's interaction re-estimates here at +1.098
  vs -0.026 - the finding is real. Candidate 6 is the wrong operationalisation: on the rows it
  targets, deleting partial seasons moves mu-hat UP by +0.45 (WR) / +1.33 (RB), because seasons are
  partial through injury and injury seasons are low-scoring. It is the same error §T rejected -
  conditioning away the games where things went wrong. It also damages 182 WR rows the mechanism
  says nothing about, by making old mu-hats staler.
- Candidates 2-5 were expected to be indistinguishable because §37 found dispersion doesn't persist.
  WRONG. De-biasing them (recentring on the training-fold offset) makes WR losses SHARPER, monotone
  in how much tail is discarded: Huber -0.14 (p=.0006) < trimmed -0.27 < median/p60 -0.69. The boom
  weeks are signal. The expectation conflated predicting shape with estimating level.

**What §P's finding actually implies is a change to B, not mu-hat.** D1/D2 shrinkage variants go the
right way at both positions (RB D2 +0.235, p=.037). NOT ADOPTED - post-hoc, two variants, no family
control. Logged as the round-8 pre-registration candidate. Nothing on the board reflects it.

**§S4 rebuild.** scripts/50_build_board.py is now the only board builder: one pass, raw inputs,
every layer a named column. Reproduces the previous board to machine precision (final: 5.3e-15).
Replacement recomputed rather than hardcoded from the 12-team top-140 composition (63WR/44RB/19QB/
14TE -> 6.382/6.209/12.103/7.866); floor recomputed from raw and asserted against floor_scheduled;
views asserted to be applied exactly once by five checks including a double-application probe.

Scripts 59, 59b, 50; results sectionS_*.csv, board_2026.csv. REPORT.md §43 (§39 was taken).

## Round 7i (2026-08-24) — §V: the flat top tested; board re-anchored to 08-17→08-24

**§V — is the plateau an artifact or a finding?** m(ADP) was 18.08 for every WR from ADP 3.1 to 10.2.
Tested four monotone fits in the 11-fold harness, h=2 throughout, nothing else changed: isotonic
(incumbent), **centred isotonic regression** (Oron & Flournoy - the standard plateau remedy, collapses
each plateau to its weight-centroid and interpolates, no new tuning parameter), OLS on log ADP, and a
50/50 blend.

**All three candidates beat the incumbent on BOTH RMSE and Spearman** (CIR 3.4093/.4800 vs
3.4196/.4729) - three for three, the same coherence pattern as §39.5 - but every DM p is large
(.45-.82) and **0/3 survive BH q=.10. Not adopted.**

**The decisive diagnostic: restricted to ADP<=12, breaking the plateau does NOT help.** RMSE iso
4.0381, CIR 4.0420, OLS 4.0927. The small pooled gain comes from elsewhere on the curve. Direct test
of flatness: within ADP<=12 the OLS slope on log ADP is **-0.539 (p=.585)**, Spearman -0.033 (p=.816),
bin means non-monotone (18.93 / 19.19 / 17.94 / 17.16 / 18.59); outside, ADP>12 gives **-2.433
(p=5e-08)**.

**Power computed for THIS design, not transplanted (§28's lesson): SE 0.980, 95% CI [-2.46, +1.38],
MDE 2.75. The rest-of-board slope -2.433 lies INSIDE the CI.** So the test cannot distinguish "flat at
the top" from "as steep as elsewhere" - the point estimate is 4.5x flatter but n=51 will not carry the
stronger claim. Recorded as such, not as a demonstration of flatness.

**Consequence, which inverts the framing:** at the top of the board the market prior is uninformative
about ordering, so mu-hat and views carrying the ordering there is the correct division of labour, not
a bug. **Chase's rank-6 was never a curve defect** - §T ruled out the mu-hat side, §U the precision
side, §V the m(.) side. The disagreement is an opinion and is recorded as one. CIR retained as the
pre-registered candidate for a single-arm confirmatory test.

**Board re-anchored** to the 08-17→08-24 window (7,658 drafts) - `board_2026_v4_0824.csv`. Top 5
unchanged: Gibbs, Bijan, Chase, McCaffrey, Nacua. Largest ADP moves since 08-18: Breece Hall +5.5
(groin), Warren -4.7, Stevenson -3.5, Odunze +2.8, Egbuka -2.1, Hampton +2.0. **Two views fell off the
top-30 board on price movement: Alec Pierce and Chuba Hubbard** - theses intact, queued with Price,
Golden, Tate, Henry, Sutton, Johnston for the deep-board extension.

Scripts `60_sectionV_flat_top.py`, `61_board_0824.py`; results `sectionV_scorecard.csv`,
`sectionV_flattop_loso.csv`, `board_2026_v4_0824.csv`. REPORT.md §43.

## Round 9 (2026-08-24/25) — the stack rebuilt as one architecture

Pre-registered in `EDA_PLAN9.md`; three researchers on separable layers (WS1 projection, WS2
valuation stack, WS3 draft engine). Reader-facing summary: `MODEL.md`. Full derivations:
REPORT.md §46–48.

**WS1 — projection from inputs: REJECTED, ninth null.** Nested test on a calibrated age-aware μ̂:
WR +0.249 (p=.404, MDE 0.895 — uninformative), RB **−0.295**. Trees lose everywhere as
pre-registered. Two real findings inside the null, both defects in the incumbent: **μ̂ is
over-dispersed (calibration slope 0.667 WR / 0.605 RB)** and **age belongs inside μ̂** (+1.18/+1.31,
9/10 folds) — the latter being essentially the only correctable error in the RB data arm. The
parsimonious `calibrated μ̂ + age` spec was found by decomposition not pre-registration, so it is
NOT adopted; round-10 candidate. Also: availability-as-input rejected (out-of-sample R² 0.039/0.018;
the naive μ̂×availability multiplier is **significantly worse than nothing**, −2.36 p=.0085,
vindicating the owner's refusal), §P's ≥12-games interaction **withdrawn** (collapses to +0.043 once
a projection replaces μ̂ — it was a symptom of μ̂ ignoring age/games), and a leak found in shared
data: **FFC historical ADP `team` is an end-of-season label** (matches final team 88%, week-1 7%).

**WS2 — a units defect that was doing real damage.** Values are points per game *played*;
replacement was season total ÷ 17, i.e. per *scheduled* week. Not cancelling in the cross-position
contrast, which is the only thing VORP is for. The naive repair (rank by total, read PPG) breaks
identification — it selects toward short high-rate seasons; 2024 WR64 was Diggs at 121.9 pts in 8
games. **Adopted: PPG rank among players with ≥8 games** → QB 15.17 / RB 7.47 / TE 9.79 / WR 7.92.
Board is insensitive to the threshold (Spearman ≥.995 across g∈[4,12]) while the rejected estimand
moves it more — identification dominates tuning. δ_RB implemented as a **structural view** (Ω→0
limit of group absolute views, asserted numerically), sized 1.40 as the smallest value consistent
with both revealed-preference inequalities, and kept **out of the January-scoreable column**. ESPN
historical ADP **rejected as hindsight-contaminated** — it drives FFC to zero against outcomes
(p=.881/.129), which a genuine preseason price cannot do. Ablation: **positional replacement does
almost all the work** (Spearman .768 without it) and the EB arm least (.9996), since §P4 restricts
it to 30 of 204 players.

**WS3 — where the edge actually is.** Flat-vs-step formalised as
W_p = Σ E[δ_i·1(N_p≥i)] — steps enter linearly and unbounded, decay only through bounded indicators.
§M's verdict **hardens**: 0/4 beat ADP and all four now lose significantly; mechanism chased to
roster mix (best-available-by-VORP builds 2 RB / 10 WR; constrain to 4 RB by round 7 and the gap
vanishes, −4.2 ± 15.9). *The board's ordering is a wash with ADP; its roster mix was the problem.*
And the sharpest result: **the ADP null scores +169 season points against the biased room** versus
against ADP opponents — the edge is the room's bias, not the board.

**The architectural change.** Simulated availability is removed from the pipeline. It produced 84%
for a player the owner says is 0%, and 1% for one whose belief is explicitly bimodal — a normal-CDF
survival model cannot represent "falls to me or goes right before." Availability is now a **declared
input** (`data/drafts/availability_priors.csv`), and `scripts/74_decide.py` refuses to run without it
rather than substituting a guess. Same discipline as Ω in §26. Steps 1–7 assert value; steps 8–10
take availability as given and compute the decision.

**Live-draft infrastructure added:** Sleeper's public API (`api.sleeper.app/v1/draft/{id}/picks`)
gives the full draft with slots and ownership, so no scraping is needed —
`data/drafts/league_draft_2026_sleeper.csv` (150 picks, mock). Owner prediction calibration recorded
in `data/drafts/prediction_calibration_2026.md`: **20/20 set recall over picks 7–26, mean slot error
1.40** — the set is near-deterministic and only the order is noisy, which is the strongest evidence
yet for §R's viability and simultaneously the reason its simulated survival curves were not usable.

**Carried to round 10:** (1) `calibrated μ̂ + age` as a pre-registered spec; (2) the WR replacement
identification is still the weakest number in L5 — 2024 leaned on one 8-game observation; (3)
τ-persistence pre-test remains unrunnable at n = 1 draft.

---

## Round 10 — the data arm rebuilt (§X, 2026-08-25)

Pre-registered in `EDA_PLAN10.md`, operational definitions in `results/sectionX_notes.md` PART 1
before the first fit. One change to the value layer, carried over from round 9's "not adopted,
because we found it by decomposition" list.

**What was done.** μ̂ was replaced by `mu_star = a + b*mu_hat + c*log[f(age)/f(age-1)]`, all three
coefficients fitted per LOSO training fold, and §H's era-3 age curve **refitted inside each fold**
rather than taken published (§W1's version let the held-out season into the curve). The
re-implementation reproduces `age_curve_era.csv` to 3e-16 and every §W1 component number to four
decimals before anything new was read.

**Result: ADOPT.** LOSO 3.776 → 3.548 (WR, p = .012, 9/10 folds) and 4.491 → 4.092 (RB, p = .0004,
9/10); the genuine 2015–21 → 2022–24 holdout — coefficients *and* curve fitted on the early
window — improves at both positions. The pre-specified overlap held almost exactly: the
combination is worth 71% (WR) / 82% (RB) of the sum of its parts, against a prediction made from
§W1's numbers that was right to 0.02 and 0.12 PPG².

**The finding that matters more than the verdict.** The gain does not survive the blend at WR:
+0.101 inside eq. (7) against its own MDE of 0.626. Chased to mechanism — 73% of the correction
points *toward* the market price, which is the move eq. (7) already makes through B, so the
posterior has pre-empted it. corr(Δ, residual) falls 0.328 → 0.084 at WR but only 0.409 → 0.187 at
RB, which is why RB survives (+0.475, p = .007). And §P4's arm rule sends μ* to WR ADP ≤ 30 only:
**the correction is delivered where it is undetectable and withheld where it is detectable.**

**Board.** `70_build_board.py --mu-star`, off by default so the incumbent still reproduces to
5.3e-15 with all new layers disabled. Thirty players move. Over-dispersed players regress (Nacua
−2.47, Chase −2.19, St. Brown −2.01 on the calibration term), old players fall (Adams −1.87 and
Evans −1.58 on the age term; −10 and −8 board places), young thin-history players rise (Burden,
Odunze, Egbuka). Jefferson and A.J. Brown leave the top 24; Love and Skattebo enter. The apparent
QB movement is the known top-70 floor-reference instability (§47.4), zero under the positional
reference.

**Carried to round 11:** (1) **§P4's arm rule** — with μ*, RB's posterior no longer sits *behind*
the market (−0.354 → +0.122) but does not beat it either (p = .673); revisiting the rule is now
the sharpest open question, and it must be pre-registered, not slipped in; (2) RB's age loading c
is the least determined quantity in the spec (16.55 ± 3.11 across folds); (3) WR replacement
identification and the τ-persistence pre-test, both unchanged from round 9.

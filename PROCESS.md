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

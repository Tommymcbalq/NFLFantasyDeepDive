# EDA Round 4 — RB universe, the aging curve across eras, market context, and a views overlay
### Pre-registered 2026-08-09, before any round-4 fitting. Rules unchanged: no tuning toward
expected results, no named-player anchors anywhere inside the pipeline, anomalies get chased,
arm adoption only on the pre-specified LOSO evidence (DM vs the frozen arm, clustered by year),
and any claim that we beat the market needs FDR control **and** a temporal holdout.

## Motivation (stated before fitting)

Four gaps, in the order they bind:

1. **Universe.** The board is WR-only. Half the discretionary questions in play are RBs, and an
   RB has no market-implied prior to sit against until the pipeline is refit for the position.
   RB is also where the aging question is loudest — and where per-game variance, committee risk
   and durability structure should differ most from WR. Assuming WR variance components carry
   over would be an untested assumption; we estimate them fresh.
2. **Aging.** Round-2 §C tested age as a *detrend of μ̂* and netted a tie (p = .983). It never
   tested the hypothesis that the age–production profile has **shifted later in calendar time**.
   That is not estimable on 2014–2025 alone; the panel now runs 1999–2025 (9,546 WR/RB
   player-seasons, age known for 100%), which gives ~three eras of leverage.
3. **Market context beyond ADP.** Sportsbook win totals and team point totals are a second,
   independently-formed market price on team environment. Whether they carry anything ADP does
   not is an open question — and an unanswerable one without a decade of *historical* closing
   numbers. Sourcing is therefore a prerequisite step with its own go/no-go, not an assumption.
4. **Discretionary views.** There is currently no disciplined channel for subjective input. The
   Black–Litterman construction is the right shape: our §6.1 isotonic ADP→PPG map with its
   estimated outcome spread already *is* a market-implied prior π with uncertainty Σ, obtained by
   reverse-engineering price. Views enter as (P, q, Ω) and never touch the statistical layer.

## §G0 New data (pulled 2026-08-09, `scripts/fetch_data_r4.py`)

- ADP refresh: FFC PPR 12-team, drafts 2026-08-01→08, n = 5,187 (July pull was n = 1,737).
  Saved dated (`adp_ppr_2026_all_20260809.csv`); July raw untouched so rounds 1–3 still reproduce.
- `adp/rb_top30_adp_2026.csv`, `meta/rb_top30_meta.csv` (28/30 matched to weekly rows; the two
  misses are 2026 rookies with no NFL data — handled under §G4), `players/rb_top30_weekly.csv`.
- `players/weekly_raw/` extended to **1999–2025**; `teams/` likewise.
- `derived/age_panel_long.csv` — WR/RB player-season panel 1999–2025 with age on a fixed Sept-1
  reference date, experience, draft capital, games, targets, carries, touches, PPG.

**Board churn to carry forward as a finding, not a nuisance:** vs the July board the WR universe
gained DJ Moore (BUF) and Courtland Sutton, lost Metcalf and Watson; Rice +15.9 ADP, Egbuka
+11.8, Nabers +9.7, Higgins −12.8, McConkey −8.9, Evans −8.8. The §6.1 map, θ*, and the board all
rerun on the August pull. Round-1–3 numbers are restated, not overwritten.

---

## §G RB universe — full refit, nothing assumed from WR

Mirror scripts 01–11 with position as a parameter. Every variance component, reliability weight,
and market-prior curve is **re-estimated on RB data**; no WR estimate is reused.

- **G1 Inclusion rule**, fixed now from aggregate distributions only, no player inspection:
  regular season; drop player-games with **touches (carries + targets) ≤ 1**. Report the excluded
  fraction and its mean PPR, exactly as §0 did for WR. All rates are *given participation*.
- **G2 Variance components** (§1–§3 analogues): σ̂²_W within-season, τ̂²_B via the same
  bias-inversion (eq. 3), recency-weighted μ̂ with h = 1 and n_eff, EB-stabilized boom/bust rates
  with RB-specific thresholds set from the **positional** PPG distribution (not WR's 20/8) —
  thresholds fixed as the pooled RB p75/p25 of qualified player-games before any fitting.
  Heteroskedasticity by experience tier refit for RB.
- **G3 Market prior** (§6.1 analogue): isotonic regression of realized PPG on ADP slot over the
  RB panel 2015–2024 (≈60 RBs/board/year, top-30 modeling universe = 300 player-seasons), with
  the outcome spread around the curve estimated the same way. This yields RB π and Σ for §J.
- **G4 Thin-data players.** 2026 rookies (2/30) and one-season players (6/30) have no usable
  history. They take the pure market arm (n_eff = 0 ⇒ full shrinkage to the ADP-implied value)
  and are **flagged as such on the board**. Round 1's finding stands: n = 4 rookies across ten
  ADP boards carries no information, and we do not pretend otherwise.
- **G5 Availability** (§A analogue): RB games-played ICC and the SV = θ*·Ê[G]/M scoring rule,
  refit for RB. Pre-registered expectation of *nothing*: we do not assume the WR result transfers.
- **G6 LOSO** on 2015–2024 RB boards, arms (i) market-only, (ii) market+data EB, (iii) the §A
  availability-scaled variant. Adoption vs (i) by DM, clustered by year, t(9 df), p < 0.10 **and**
  RMSE improvement. Outputs: `results/sectionG_notes.md`, `loso_scorecard_rb.csv`,
  `valuation_rb_2026.csv`.

**Pre-specified honesty clause:** if RB LOSO shows the data arm does *not* beat market-only —
plausible, given committee volatility and shorter careers — that is the result, the RB board is
market-anchored, and the value we add on RBs lives entirely in §J. We do not go looking for a
different arm until one beats (i).

---

## §H The aging curve, and whether it has moved

Panel: `age_panel_long.csv`, 1999–2025, WR and RB fit **separately**. Qualification (fixed now):
player-season with ≥ 8 games and ≥ 40 touches, and the same for the adjacent season when a
transition is used. Report qualified-N by era before any model.

**Outcome.** Relative PPG: r_is = PPG_is / (mean PPG among qualified players at that position in
season s). This removes league-wide scoring/period effects **by construction**, which we need,
because with player fixed effects age and period are exactly collinear within player — the
age–period–cohort problem. Stated up front: on the raw scale, any league-wide time trend loads
onto the linear component of the age curve and the era comparison would be uninterpretable. The
relative scale is what makes §H answerable at all. Absolute-PPG versions are reported as a
sensitivity, explicitly labeled as confounded.

- **H1 Age profile.** r_is = α_i + f(age_is) + ε_is, player fixed effects, f a natural cubic
  spline (knots at the age quintiles of the qualified panel, fixed before fitting). Cluster-robust
  by player; cluster bootstrap on player for all CIs.
- **H2 Era interaction — the actual hypothesis.** Eras fixed now as **1999–2007 / 2008–2016 /
  2017–2025** (equal thirds of the panel, chosen on calendar arithmetic alone). Fit
  f_e(age) per era; test H0: no era × age interaction by cluster-bootstrap F. Report per era, with
  CIs: peak age; age at which the curve first falls 10% below peak ("the cliff"); slope from 28→32.
  A smooth alternative (age spline × centered season, continuous) is fit as a robustness check.
  **Both directions are findings.** If the cliff has not moved, it has not moved.
- **H3 Selection.** Observed old players are survivors, so H1/H2 are attenuated and — worse — the
  *selection itself* may have changed by era, which could manufacture an apparent shift. Two
  guards, both pre-registered: (a) balanced-cohort refit restricted to players with ≥ 6 qualified
  seasons; (b) a **career-exit hazard** model — discrete-time hazard of last qualified season on
  age spline × era. If the drop-off age genuinely moved later, exit ages move later too, and the
  hazard is far less exposed to the survivorship artifact than the production curve is. H2 and H3b
  agreeing is the evidence; H2 alone is not.
- **H4 Workload carryover (RB).** Δr_{i,s} on prior-season touches (linear + quadratic, and a
  ≥ 350-touch indicator fixed now), controlling for f(age) and player FE. Tests whether the
  post-heavy-load decline that gets *attributed* to age is separable from age.
- **H5 Does the market price any of it?** On the 2015–2024 ADP panel (WR and RB), regress the
  market residual R on {age, age², era interaction, prior-season touches for RB}. FDR q = 0.10
  across the round-4 test family, plus temporal holdout 2015–22 → 2023–24.
  **Decision rule, fixed now:** an age arm enters the LOSO harness *only* if H5 survives both
  screens. If H5 is null — the market already prices age correctly, on either the old curve or the
  new one — then the H1–H3 curve informs the §J views layer only, and is labeled unvalidated
  there. No age adjustment enters the statistical board on the strength of H2 alone.

Outputs: `results/sectionH_notes.md`, `age_curve_era.csv`, `exit_hazard.csv`, figures.

---

## §I Market context (Vegas) — sourcing first, modeling only if history exists

- **I1 Source scout, no modeling.** Determine what is obtainable without paid auth for
  **2015–2024**: closing team win totals, team season point totals, and (if they exist that far
  back) player receiving/rushing props and games-played props. Report per source: seasons covered,
  whether the number is opening or closing, licence/ToS, and reproducibility.
- **I2 Go/no-go, fixed now.** ≥ 8 historical seasons of a consistent team-level number ⇒ §I3 runs.
  Fewer ⇒ **stop**; pull 2026 numbers for the views layer only, labeled unbacktested, and say so
  in the report. We do not adopt an arm we cannot validate, however sensible it looks.
- **I3 (conditional) Edge test.** Regress the market residual R on team-context surprise — the
  posted team total relative to the prior season's realized total, and the change in posted total
  — under the full protocol (FDR + holdout).

  **Amendment recorded 2026-08-09, after §I1 and before any §I3 fitting.** §I1 returned a split
  verdict: closing **win totals** are available for 2015–2025 (32 teams × 11 seasons, no gaps;
  validated against nflverse 2015–2020 at 181/191 exact, MAD 0.026 wins) — so the gate is passed.
  Season **point totals are not obtainable** for the historical window (the market is a sporadic
  novelty; the tempting `nflverse games.csv` substitute is per-game in-season data whose sum is a
  look-ahead quantity no August drafter could see, so it would leak the season into the
  prediction). Player props do not exist before 2023 at all. §I3 therefore runs on **win totals
  substituted for point totals** as the team-environment measure, with the surprise defined as
  (posted win total) − (prior-season realized wins) and the change in posted total year over year;
  lines are de-vigged from the paired over/under prices rather than taken at the raw half-win.
  The 2026 numbers come from a *different* source than the historical series (Covers populates a
  season only retrospectively), so the two are not treated as one continuous series and the 2026
  values are not used to fit anything. Point totals for 2026 exist only as a flagged third-party
  proxy and are §J input only, never a fitted feature. Only on survival does a context arm enter LOSO with
  the standard adoption rule. Note in advance: ADP is formed by drafters who can read the same
  win totals, so the prior is that this is **already priced**; a null is the expected result and
  is publishable as such.
- **I4** Regardless of I2, record the 2026 team totals for every board team as a §J input.

Outputs: `results/sectionI_sources.md`, and conditionally `sectionI_notes.md`.

---

## §J Black–Litterman views overlay — strictly downstream, fully audited

The statistical board is **frozen** before this section runs. §J produces a *second* set of
numbers alongside it; it never feeds back into any fit, LOSO score, or edge test.

- **J1 π and Σ.** π = the fitted §6.1/§G3 isotonic ADP→PPG value at each player's slot. Σ = the
  estimated covariance of true value around that curve: diagonal from the §6.1 outcome spread
  (per position, per experience tier where §3 says it differs), **off-diagonal from the §F
  teammate structure** — same-team players' true PPG are negatively correlated through the
  target-share/touch-share constraint, and §F already measured the implied-share machinery to
  quantify it. Team-environment correlation (same team, positive, via §I4 if available) is the
  second off-diagonal block. Every entry is estimated, not hand-set; if a block cannot be
  estimated it is set to zero and that choice is reported.
- **J2 Views.** Rows of P: absolute (one entry = 1) or relative (entries sum to 0, e.g. "A over B
  by q PPG"). Each view is logged with q, the confidence used to set the Ω diagonal, the stated
  rationale, and its date. Ω is set from a declared confidence scale, not fitted.
- **J3 Posterior.** θ̄ = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹[(τΣ)⁻¹π + PᵀΩ⁻¹q], with τ reported and a sensitivity
  sweep over it. Deliverables per player: π, θ̄, the shift, and a decomposition attributing the
  shift to each view — so a view that moves a player 2 PPG is visible as such.
- **J4 Machinery validation before any real view.** Synthetic views only: a zero view must be a
  no-op; an infinite-confidence absolute view must pin the posterior to q; a relative view must
  move the pair in opposite directions and must leak into teammates through the off-diagonal in
  the direction Σ implies. These are asserted as tests, not eyeballed.
- **J5 Scoring.** Every view is written to `results/views_2026.csv` with its magnitude and
  confidence **before** the season, so views can be scored against outcomes afterwards. This is
  the mechanism that keeps subjective input honest over time; it is the whole point of separating
  the layers rather than blending them into the fit.

Outputs: `scripts/19_bl_overlay.py`, `results/sectionJ_notes.md`, `views_2026.csv`,
`board_2026_with_views.csv` (statistical column and posterior column side by side, always both).

---

## Round-4 deliverables

| item | file |
|---|---|
| pre-registration | `EDA_PLAN4.md` (this) |
| data | `scripts/fetch_data_r4.py`, `data/derived/age_panel_long.csv`, RB tables |
| RB pipeline | `scripts/2x_rb_*.py`, `results/sectionG_notes.md`, `valuation_rb_2026.csv` |
| aging | `results/sectionH_notes.md`, `age_curve_era.csv`, `exit_hazard.csv` |
| market context | `results/sectionI_sources.md` (+ conditional `sectionI_notes.md`) |
| views overlay | `scripts/19_bl_overlay.py`, `results/sectionJ_notes.md`, `views_2026.csv` |
| write-up | REPORT.md Part IV; PROCESS.md round-4 log |

Multiple-testing family for round 4 = {H5 tests, I3 tests}. FDR q = 0.10 across that family,
holdout 2015–22 → 2023–24 for anything that survives.

---

## §K Schedule strength — pre-registered 2026-08-09, after §K0 sourcing, before any fitting

**Family declaration, stated first.** The round-4 FDR family {§H5, §I3} is **closed** — its 11
tests were fixed in advance, corrected, and reported. §K is a **new, separately declared family**
with its own BH correction at q = 0.10. Retroactively expanding a closed family after seeing its
results would invalidate the correction already applied; a new family is the honest construction.

**§K0 outcome (sourcing, complete).** 12 seasons × 32 teams, zero missing, in
`data/schedule/sos_history_2015_2026.csv`: mean opponent preseason win total (`sos_vegas`), mean
opponent prior-season win % (`sos_prior_wpct`), mean opponent prior-season PPR allowed to WRs and
to RBs (`sos_wr_fpa`, `sos_rb_fpa`), each in full-season, weeks-1–14 and weeks-15–17 windows. All
are preseason-knowable: the grid is public in May, and every quality weight is either a preseason
market number already held or a lagged prior-season realized number. `spread_line`/`total_line`
remain barred. Build validated against CBS (Spearman 1.00), an independent LeagueStation recompute
(Pearson 0.967) and Sharp Football (0.88). Franchise-abbreviation drift (STL/SD/OAK vs LA/LAC/LV)
was silently dropping 128 opponent-games in 2015–2019; fixed.

**§K1 Effect-size ceiling, computed before testing and recorded here as a prediction.** Schedules
are near zero-sum: within-season SD of `sos_vegas` is 0.245 wins. At §I3's measured +0.251 PPG per
win, a 1-SD full-season schedule swing is worth **≈0.06 PPG** — against SD(R) = 3.32 and §I3's MDE
of 0.87 PPG per SD. **The full-season test is predicted to be underpowered by more than an order of
magnitude.** It is run anyway, for the record and because the prediction is itself falsifiable, but
a null there carries almost no information and will be reported as uninformative rather than as
evidence of absence.

**§K2 The live hypothesis.** Weeks-15–17 SOS has 4× the dispersion (0.99 vs 0.245 wins) — three
games do not average out — and a season-long ADP has weak reason to price a playoff-window effect.
This is the only sub-hypothesis with a plausible power profile, and it is designated the primary
test **now**, before fitting.

**§K3 Specification.** Market residual R on each SOS measure, one measure per test, WR and RB
panels separately, 2015–2024, SEs clustered by season (t, 9 df), HC3 alongside. Measures are
**not blended** — §K0 established they are near-orthogonal (team-quality vs positional ≈ 0.00;
season vs playoff 0.12; market-implied vs prior-year 0.52), so averaging them would produce a
composite measuring nothing. Report the pre-test MDE alongside every p-value.

**§K4 Decision rule, fixed now.** BH q = 0.10 within the §K family, **plus** the temporal holdout
2015–22 → 2023–24. Both screens required, as always. Only on surviving both does a schedule arm
enter LOSO with the standard adoption rule (DM vs the frozen arm, clustered by year, p < 0.10 and
RMSE improvement). Given §K1, the anticipated result is null.

**§K5 Recorded caveat.** The premise of positional SOS is weak on our own data: WR points-allowed
persists year over year at only ~0.25 and was *negative* in the last two transitions. Positional
tests are run as specified, but a null there is consistent with the measure carrying little
year-over-year signal in the first place, and will be reported that way rather than as evidence
that matchups do not matter.

Outputs: `results/sectionK_notes.md`, `results/edge_schedule.csv`, rerunnable script.

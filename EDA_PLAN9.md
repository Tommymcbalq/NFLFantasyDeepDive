# EDA Round 9 — The full stack, rebuilt as one architecture
### Pre-registered 2026-08-24. Rules unchanged: no tuning toward expected results, no named-player
### anchors inside any fitted layer, anomalies chased, adoption only on pre-specified LOSO evidence,
### FDR + temporal holdout for any market-edge claim.

## Why rebuild

Eight rounds have produced a board that works but was *accreted*: each layer was bolted onto the
last, adjustments live in three different scripts, and the base layer is thinner than the machinery
sitting on top of it. Specifically, **Layer 1 is currently just "market price blended with a
recency-weighted mean."** There is no projection from measurable inputs — usage, volume,
efficiency, environment — anywhere in the stack. §D tested a ridge on usage covariates and failed;
§S tested seven alternative summaries of history and none beat the mean. But a *projection model*
has never been built, only summaries of past output.

The owner has now supplied the missing pieces on the discretionary side: a stated expected draft
order (28 players deep), a positional RB premium implied by two independent revealed preferences
(δ_RB ∈ [1.29, 1.40]), and per-player availability beliefs for his own picks. The architecture
below puts all of it in one pipeline with each layer separable, testable, and individually
disable-able.

## The stack

    L0  data                weekly logs, advanced stats, multi-source ADP, context/news
    L1  projection          measurable inputs -> expected PPG               [WS1]
    L2  structural          age, situation change, availability             [WS1]
    L3  market anchoring    isotonic ADP->points prior; EB shrinkage        [WS2]
    L4  discretion          Black-Litterman views (P, q, Omega)             [WS2]
    L5  positional value    replacement, VORP, delta_RB, floor              [WS2]
    L6  draft value         VONA on LINEUP-MARGINAL value, not raw          [WS3]
    L7  contingency         expected order, survival, decision trees, sim   [WS3]

**Governing principle, fixed now:** each layer must justify itself against the layer below. A layer
is adopted only if it improves out-of-sample prediction (L1–L3) or expected roster value (L5–L7)
under the project's existing screens. **Layers that fail are retained in the write-up as nulls and
disabled in the pipeline**, not silently dropped.

---

## WS1 — The projection engine (L1 + L2)

**The question never asked:** can expected PPG be projected from *inputs* better than it can be
summarised from *past output*?

**L1.1 Design.** Predict next-season PPG from preseason-knowable inputs only. Candidate feature
blocks, fixed now:
- **Volume**: prior-season target share, air-yards share, route participation, carry share,
  goal-line and third-down usage, snap share. (RB/WR/TE differ; QB uses attempts and rush volume.)
- **Efficiency**: YAC/rec, aDOT, YBC/att, broken tackles, RACR, CPOE — but **only those passing the
  §4 reliability gate**, re-run on the wider panel. This is the existing discipline and it binds.
- **Environment**: team pass volume, pace, offensive line proxies, QB continuity.
- **Structure**: age (from §H's per-era curves), experience, draft capital.

**L1.2 Estimator.** Fit and compare, all LOSO on the same folds as §7: (a) regularised linear
(ridge/elastic-net, standardised), (b) gradient-boosted trees, (c) a hierarchical model with
position-level partial pooling. **Pre-specified expectation: the linear model wins or ties.** The
sample is ~600 player-seasons per position with a ρ_max of ~0.41 (§2); tree ensembles will overfit
that. If a tree model wins, chase why before believing it.

**L1.3 The binding comparison.** Projection must beat **μ̂, the recency-weighted mean** — the §S
incumbent — head to head, DM clustered by year. Beating a naive baseline is not enough. Report the
realised MDE beside every p-value (§28.1) and label an underpowered null as uninformative.

**L2.1 Age applied individually.** §H found the aging curve has not moved later and that the market
prices age correctly (§H5 null). But the curve itself is estimated and can be *applied* even when
it carries no market edge: a 32-year-old's projection should regress differently from a 25-year-old's.
Test whether applying f_e(age) to the projection improves it. **This is not an edge claim** — it is
a within-model structural correction, and it is the honest version of "Derrick is Derrick."

**L2.2 Availability as a modelled input, not an output.** §A found availability is a stable trait
(ICC .36). The floor metric built this session (p25 over *scheduled* weeks, suspensions excluded)
is descriptive. Test whether prior availability predicts next-season availability well enough to
enter the projection — and if it does, project **points per scheduled week** as a second target
alongside PPG, reporting both. The owner has ruled out an expected-games multiplier applied post
hoc; this is different, and it must be validated before use.

**Deliverables:** `results/sectionW1_notes.md`, a fitted projection with per-position coefficients
and reliability-gate membership, LOSO scorecard vs μ̂, and a clear ADOPT/REJECT.

---

## WS2 — The valuation stack (L3 + L4 + L5)

**L3.1 Market prior.** Refit the isotonic ADP→points curve on the deep panel (already validated in
§P: exact below ADP 40, support to ~171). If WS1 produces an adopted projection, the EB posterior
becomes a three-way combination — projection, own history, market price — and **the weights must be
estimated, not chosen**, extending eq. (7) with the precision of each source. Derive it properly.

**L3.2 Multi-source ADP.** §Q established FFC's `teams` parameter is inert (one pool, mislabelled)
and that ESPN retains 2023/2024/2026 only. Build the **source-translation map** §Q recorded as the
correct construction: a monotone rank→rank transform per source fitted on overlapping seasons,
translating foreign ADP into FFC-equivalent so the existing curve is never evaluated off its
support. Do not refit the curve on a foreign pool.

**L4 Views.** The BL layer (§26) stands. One change: **views must be typed** — player-specific
(current behaviour) versus *structural*, where a structural view applies to a group. Implement
**δ_RB as a structural view** rather than a pile of player views, with its magnitude taken from the
owner's revealed preferences (1.29 from CMC-over-Amon-Ra, 1.40 from Saquon-over-Rice) and logged
with that derivation. Ω declared, never fitted, as always.

**L5.1 Replacement.** Recompute per position from the *current* ADP composition (2026: 44 RB / 63 WR
/ 14 TE / 19 QB in the top 140), recency-weighted. **Resolve the units defect found this session:**
player values are points per game *played*; replacement is computed as season total ÷ 17, i.e. per
*scheduled* week. These are different quantities and are currently being subtracted from each other.
Fix by putting both on the same basis, and report the effect (RB +0.74, WR +2.63, TE +1.74, QB +3.17
if moved to per-game-played). Note the WR figure is contaminated by ranking on season total and then
reading PPG — a player who misses half a season can rank 64th by total with a high per-game rate — so
the identification of *who* replacement is needs specifying before the number is trusted.

**L5.2 Floor.** λ = 0.10 against a positional reference, as calibrated. Floors are computed over
scheduled weeks with documented suspensions excluded and blanked below 34 eligible weeks. Keep.

**Deliverables:** `scripts/70_build_board.py` as the single builder replacing `50_build_board.py`,
`results/board_2026_v2.csv` with every layer a named column, `results/sectionW2_notes.md`, and a
layer-ablation table showing what each contributes.

---

## WS3 — The draft engine (L6 + L7)

**L6.1 Lineup-marginal value.** §R established that raw VONA is the wrong quantity — a player who
would not start adds zero regardless of board rank. All draft-time value must be **marginal to the
starting lineup** given current roster state (1QB/2RB/2WR/1TE/2FLEX/1DST). Generalise this so it
holds at every pick, not just the one §R evaluated.

**L6.2 The flat-versus-step principle**, discovered in §R and worth generalising: *flat tiers are
safe to wait on however fast they drain; steps are not, however slowly.* Formalise it — for each
position compute both the decay rate (how fast the tier empties) and the step size (the gap to the
next tier) and show that the wait/take decision depends on the second, not the first.

**L7.1 Expected order from the owner's beliefs, not from ADP.** The owner's stated 28-player order
is the prior for opponent behaviour; the mock draft calibrated its noise (mean |slot error| 1.40
over picks 7–26, 20/20 set recall). Beyond the stated block, fall back to translated ADP. Accept
owner-supplied per-player availability overrides (e.g. "Drake London 0% in the 3rd, Olave ~20%") as
explicit constraints on the survival distribution rather than as model output.

**L7.2 Contingency trees.** For each of the owner's picks, the branch structure: which player at
what probability, conditional on who survived. Report as a decision plan he can carry, not a
simulation summary.

**L7.3 Strategy comparison.** Evaluate whole strategies end to end on expected final starting-lineup
value — including counterfactuals the owner names (e.g. "we could have not taken Caleb"). §M's
finding that no pick sequence beat drafting the board assumed ADP-drafting opponents and a board
without δ_RB or lineup-marginal value; re-run it with all three and report whether the verdict holds.

**Deliverables:** `scripts/71_draft_engine.py`, `results/sectionW3_notes.md`, contingency tables per
pick, strategy comparison with standard errors.

---

## Cross-cutting rules

1. **One pipeline.** After this round there is exactly one board builder. No adjustment may live
   anywhere else.
2. **Every layer separable.** Each contributes a named column and can be switched off; the ablation
   table is a required deliverable.
3. **Nothing fitted on 2026 outcomes.** Obviously. But also: nothing fitted on the owner's own picks
   when modelling opponents (§R excluded his 9 for this reason).
4. **The owner's discretion enters only at L4 and L7.** Layers 1–3 and 5–6 never see a named player
   preference. This is what keeps the views scoreable in January.
5. **Report MDEs.** Any null must state what it could have detected.

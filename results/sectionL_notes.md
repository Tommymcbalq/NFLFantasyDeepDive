# §L — Positional conversion rates by draft cost

Executed against `EDA_PLAN5.md` §L0–§L6, pre-registered 2026-08-09 before any round-5 fitting.
Scripts: `scripts/26_sectionL_conversion.py` (analysis), `scripts/27_sectionL_figures.py` (figures).
Outputs: `results/conversion_rates.csv`, `results/conversion_tests.csv`,
`results/sectionL_panel.csv`, `results/sectionL_costtrend.csv`, `results/sectionL_holdout.csv`,
`results/figures/sectionL_*.png`.

**Headline: all eight pre-registered family tests are null under BH q = 0.10 (smallest raw
p = .079 against a threshold of .0125), and the holdout confirms nothing. Claims (a), (b), (c) are
not supported. Two of the three nulls are informative; §L3's is not — it is underpowered by a
factor the design cannot fix.** The most valuable product of §L is a *definitional* finding: the
pre-registered "value-return" hit definition is mechanically positional (§D7) and cannot be read
as a cross-position value measure.

---

## §L0 Construction, and one deviation stated

- Boards: `data/adp/historical/adp_ppr_20{15..24}.csv`, all QB/RB/WR/TE rows (DEF/PK dropped).
  **1,694 board rows, 10 seasons, 0 unmatched, 0 unresolved ambiguous names.**
- Outcomes: `data/players/weekly_raw/`, REG only, **every player in the league**, so a drafted
  player can be displaced by an undrafted breakout. Verified: undrafted players take on average
  0.5 of the 12 RB slots and 0.3 of the 12 WR slots per season (e.g. Puka Nacua and Kyren Williams
  both displace a drafted player in 2023). Slot budgets reconcile exactly — drafted top-12
  finishers average 11.5 (RB) and 11.7 (WR) out of 12.
- Two outcome definitions, both reported throughout: **(T) season total PPR** (primary; a drafted
  player with no REG appearance scores 0 and counts as a miss — 17 such rows) and **(P) PPG given
  ≥ 4 games**. Under (P) the sub-4-game players leave the *denominator*, which is the whole point
  of carrying both (§L6 confound 1).
- **Deviation, stated:** the project's standing WR inclusion rule is REG ∧ `targets > 1`. That is a
  WR-usage filter and is meaningless for RBs, so a "game" here is an appearance (a REG row).
  Applying the WR rule to RBs would have made the two positions non-comparable, which is the entire
  object of §L.

### Two build defects caught before any test was read

1. **A defensive end scored as a WR hit.** "Charles Johnson" on the 2015 WR board (ADP 61.6) matched
   the Panthers DE of the same name: 0.0 fantasy points, but ranked **2nd among DEs**, so the
   pre-fix code recorded a *top-12 hit* for a WR drafted in R5–6. Matching is now restricted to
   offensive skill positions.
2. **Position drift moved board WRs into the TE pool.** nflverse tags Jordan Matthews TE in
   2015–16; ranked among TEs he was a top-12 "WR hit" in both seasons. Fixed by ranking on an
   **effective position** — for a drafted player the *board* position governs, since that is the
   roster slot the pick buys — and recomputing league-wide ranks on it.

Together these inflated WR top-12 hits above the hard ceiling of 12 per season (2015 showed 14,
2024 showed 13). Post-fix every season respects the budget. Both are the same class of error as
§28.3's withdrawn false positive: a number that is arithmetically impossible if you check it
against a conservation constraint.

---

## §L1 Conversion rates — 12-team frame, pooled 2015–2024, Wilson 95%

`hit = top-12 positional finish, season totals` (the clean cross-position measure):

| bin | RB rate (n) | Wilson | WR rate (n) | Wilson |
|---|---|---|---|---|
| R1–2 | **53.7%** (123) | [44.9, 62.3] | **52.5%** (101) | [42.8, 62.0] |
| R3–4 | 24.4% (90) | [16.7, 34.3] | 27.8% (115) | [20.5, 36.6] |
| R5–6 | 12.3% (81) | [ 7.0, 20.9] | 11.3% (97) | [ 6.5, 19.1] |
| R7–8 | 8.7% (92) | [ 4.5, 16.2] | 10.6% (85) | [ 5.7, 18.8] |
| R9+ | 4.1% (217) | [ 2.2, 7.7] | 4.5% (268) | [ 2.6, 7.6] |

`hit = ≥ median season total of the same (season, bin), all positions` — **see §D7 before using
this row: it is mechanically positional**:

| bin | RB rate (n) | WR rate (n) |
|---|---|---|
| R1–2 | 43.1% (123) | 57.4% (101) |
| R3–4 | 34.4% (90) | 60.0% (115) |
| R5–6 | 37.0% (81) | 52.6% (97) |
| R7–8 | 41.3% (92) | 47.1% (85) |
| R9+ | 34.1% (217) | 46.3% (268) |

PPG-given-participation versions and the 2018–2024 window are in `conversion_rates.csv`
(`window` column); the per-season cells are stored there too and, per §L1's pre-specification,
**are not interpreted** — a cell holds 6–15 players, SE ≈ 13 pp.

**10-team frame (the owner's actual league).** R1–2 = picks 1–20 rather than 1–24, so the elite bin
is smaller and richer. Top-12 conversion: RB **53.2%** (n=109) vs WR **59.3%** (n=81); R3–4 RB 32.5%
(77) vs WR 26.3% (95). Value-return: R1–2 RB 41.3% vs WR 59.3%. **The RB-over-WR ordering the
hypothesis asserts does not appear in either frame; in the 10-team frame the top-12 point estimate
runs 6 pp the other way.** Every strategy sentence below names its frame.

---

## §L2 Claim (a): elite RB converts better than elite WR — **not supported**

Season-clustered mean of per-season differences, t with 9 df, R1–2 cell (12-team frame).

| hit definition | RB − WR | SE | 95% CI | p | **MDE** |
|---|---|---|---|---|---|
| top-12 finish (total) | **−0.011** | 0.060 | [−0.147, +0.126] | .866 | 0.190 |
| value-return (total) | **−0.148** | 0.075 | [−0.318, +0.021] | .079 | 0.236 |
| top-12 finish (PPG\|≥4g) | −0.017 | 0.054 | — | .760 | 0.169 |
| value-return (PPG\|≥4g) | −0.070 | 0.072 | — | .357 | 0.228 |

Robustness — LPM with a season random intercept, RB/WR only (`family = 0`): top-12 **+0.008**
(SE 0.068, p = .905), value-return **−0.143** (SE 0.067, p = .032). Both agree in magnitude with the
clustered-t primary; the value-return p is smaller because the random-effects model pools the
within-season variance rather than paying for 9 df, which is the usual efficiency-vs-robustness
trade and is why the clustered t is the pre-registered primary. It does not change the BH outcome
and, per §D7, the value-return estimate should not be believed regardless.

*(A defect caught in this cell: the first version of the robustness code selected the first
`pos_adp` term by string match, which is the **QB** dummy, and reported +0.226 for what was labelled
the RB effect. The R1–2 bin contains 8 QBs and 10 TEs. Corrected to select the RB contrast and to
restrict the sample to RB/WR — the third mechanical error §L caught, and the third that would have
pointed in the hypothesis's favour.)*

**Verdict.** Elite RB and elite WR convert to a top-12 positional finish at statistically
indistinguishable rates: 53.7% vs 52.5%, a 1.1 pp gap against an MDE of 19 pp. The point estimate
is *negative* on all four definitions — the direction opposite to the hypothesis on every one.
This null is only partly informative: the MDE of 19 pp means a true RB advantage of, say, 12 pp
would have been missed more often than not. What can be said is that **the data give no support at
all for the claimed direction, and the largest RB advantage consistent with the top-12 CI is
+12.6 pp while the largest WR advantage consistent with it is +14.7 pp** — i.e. the evidence, such
as it is, is symmetric and slightly favours WR.

The family's smallest p-value (.079, value-return) points **against** the hypothesis, not for it.
It does not survive BH, and §D7 shows it should not be believed even if it had.

---

## §L3 Claim (b): elite-RB conversion is trending up — **uninformative, not evidence of absence**

Logit of hit on year (linear), RB R1–2 cell only, season-clustered SEs with t(9).

| hit definition | slope/yr (log-odds) | cluster SE | 95% CI | p | **MDE (this design)** |
|---|---|---|---|---|---|
| top-12 (total) | +0.065 | 0.063 | [−0.078, +0.208] | .332 | **0.199 log-odds/yr** |
| value-return (total) | +0.014 | 0.055 | [−0.110, +0.139] | .801 | 0.174 |

Secondary, the one pre-specified split (2022–24 vs 2015–21), Welch t on season rates:

| hit definition | 2022–24 | 2015–21 | diff | p | **MDE** |
|---|---|---|---|---|---|
| top-12 | 54.8% | 51.7% | **+3.0 pp** | .813 | **50.5 pp** |
| value-return | 42.5% | 43.5% | −1.0 pp | .918 | 36.8 pp |

**The MDE is computed for this design**, per §28's lesson that an MDE cannot be transplanted across
designs with different error structures: it is derived from the realized season-clustered SE of the
slope on 10 clusters, not borrowed from §I3 or §K. At p ≈ 0.5 an MDE of 0.199 log-odds/yr is about
**5 pp per year — 50 pp over the ten-year window**. The split test's MDE of **50.5 pp** is worse
still: with 3 seasons against 7 and a between-season SD of ~13 pp, only a shift from a coin flip to
near-certainty would register.

**Verdict: uninformative.** The point estimates lean positive (+0.065 log-odds/yr ≈ +1.6 pp/yr at
the observed base rate) but the design cannot distinguish that from zero, and it could not have
distinguished a trend three times that size either. **This is labelled an underpowered null, not
evidence of absence.** No breakpoint other than the pre-specified one was examined, and none should
be: the season series (2015→2024: 27%, 50%, 58%, 64%, 58%, 47%, 57%, 64%, 33%, 67%) has a 2023
trough deep enough that almost any "recent years" story can be told by choosing where to cut.

### §L6 confound 3 — RB draft cost over the same window (mandatory context)

Without this the trend is uninterpretable, so it is reported whether or not it helps:

| year | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean ADP of RB1–10 | 9.8 | 13.5 | 9.9 | 6.8 | 7.6 | 6.4 | 6.8 | 7.4 | 14.2 | 10.8 |
| RBs drafted in R1–2 | 11 | 10 | 12 | 14 | 12 | 15 | 14 | 14 | 9 | 12 |

Linear trend in RB cost: **+0.001 ADP slots/yr, p = .997 — flat.** But flat *on a line* conceals a
clear cycle: elite RBs were cheap in 2015–16, at their most expensive 2018–2022 (mean ADP 6.4–7.6),
and cheapened sharply in 2023–24. **Across the pre-specified split, RBs were 2.1 picks cheaper in
2022–24 (mean ADP 10.81) than in 2015–21 (8.69).** So the +3.0 pp "improvement" in the split is
measured against a *cheaper* elite RB — the direction that inflates a hit rate mechanically. The
+3.0 pp is therefore, if anything, an over-statement of the underlying change, and it was not
significant to begin with.

Two further checks on the same confound run *against* the cheapness story: the correlation between
elite-RB cost and elite-RB conversion is **−0.43** (higher ADP = cheaper ↔ *lower* hit rate), and
the number of RBs taken in R1–2 correlates **+0.54** with their hit rate. The market appears to
push RBs up the board in years when elite RBs are in fact good, rather than conversion rising
because RBs got cheap. Neither correlation is significant on 10 points; both are reported because
the confound obliges it.

---

## §L4 Claim (c): mid-round WRs are the better buy — **not supported, tested as an interaction**

Logit `hit ~ position × bin`, season-clustered, joint Wald on the 4 interaction terms (the
actionable quantity: does the WR-minus-RB gap change with draft cost?). A linear-in-bin
parameterisation gives one interpretable slope and its MDE.

| hit definition | joint Wald p (4 df) | linear-in-bin RB×bin | SE | p | MDE |
|---|---|---|---|---|---|
| top-12 (total) | **.964** | −0.050 | 0.139 | .727 | 0.436 |
| value-return (total) | **.317** | +0.076 | 0.108 | .502 | 0.339 |
| top-12 (PPG\|≥4g) | — | −0.007 | 0.124 | .954 | — |
| value-return (PPG\|≥4g) | — | −0.026 | 0.088 | .778 | — |

**Verdict.** The WR-minus-RB conversion gap is flat in draft cost. Under the clean top-12 definition
the gap is +1.2, −3.4, +1.0, −1.9, −0.4 pp across R1–2 → R9+ — noise around zero, with a joint
Wald p of .96. Under the value-return definition the WR advantage is *largest in R3–4* (+25.6 pp),
not in R5–8 (+15.6, +5.8 pp), which is the wrong shape for "rounds five through eight" even before
the test.

This null is stronger than its p-value suggests, and the reason is §D7: **the value-return
definition's mechanical bias grows with depth** (WRs out-score RBs by +11.7 raw points at the same
positional rank in tier 1, rising to +42.3 by tier 4), so it should manufacture exactly the
increasing WR advantage claim (c) predicts. It does not appear. A mechanism that would fabricate
the claim is present in the measure and the interaction is still flat.

---

## §D7 — the value-return definition is mechanically positional (the round-4b-style catch)

§L2 pre-specified value-return as "a value-return definition that does not privilege positional
scarcity." **It backfires: by refusing to privilege scarcity it privileges raw PPR volume, which is
a positional constant.** Mean season total PPR at matched league-wide positional finish rank:

| finish tier | RB | WR | WR − RB |
|---|---|---|---|
| 1–12 | 274.9 | 286.6 | **+11.7** |
| 13–24 | 194.9 | 222.3 | **+27.5** |
| 25–36 | 152.3 | 187.8 | **+35.4** |
| 37–48 | 115.5 | 157.8 | **+42.3** |

PPR pays a point per reception and WRs catch more, so an equally-ranked WR banks more raw points
than an equally-ranked RB at every tier, increasingly so with depth. A threshold expressed in raw
points that is shared across positions therefore favours WRs by construction.

Post-hoc confirmation (**explicitly not in the FDR family, reported as exploratory**): pooling all
bins and controlling for within-bin price, the RB effect is **−0.602 log-odds (SE 0.148, p = .003)**
on value-return but **−0.134 (SE 0.137, p = .354)** on top-12 finish. A large, "significant"
position effect that exists only under the contaminated definition and vanishes under the clean one.
Within-bin price composition does *not* explain it — RBs are if anything slightly *more* expensive
inside R1–2 (mean ADP 11.1 vs 13.2), i.e. the confound runs the wrong way to rescue it.

**Consequence:** the value-return rows in `conversion_rates.csv` are retained (they were
pre-registered and are reported as pre-registered) but **must not be read as evidence that WRs are
better value than RBs at a given price.** They measure PPR volume. §L6's confound 2 said this in
advance in words — "a 40% RB hit rate and a 40% WR hit rate are not equally useful" — and D7 is the
quantitative version. The cross-position comparison that survives inspection is the top-12
positional-finish definition, and on that definition every §L test is flatly null.

Two further mechanical checks, both clean:

- **Slot-budget ceiling.** Exactly 12 top-12 slots exist per position per season, so a position
  drafted more heavily early has a lower attainable hit rate. The ceiling binds mildly at R1–2
  (mean 12.3 RBs drafted → ceiling 0.94; 10.1 WRs → 0.97) and hard at R9+ (0.57 RB / 0.46 WR).
  The R1–2 comparison is therefore *slightly* biased against RB — the direction that would hurt the
  hypothesis — but by ~3 pp, an order of magnitude below the MDE. The empirical slope of RB hit
  rate on the number of RBs drafted early is **+0.037 (p = .11)**, i.e. positive, the opposite sign
  to the ceiling, so the ceiling is not driving anything.
- **QB/TE contamination of the bin median.** QBs are 3% of R1–2 but 19% of R9+ and score far more
  raw points (median 240–321 vs 108–232), so they push the value-return threshold up in late bins.
  This shifts RB and WR *identically within a bin*, so the interaction is unaffected; recomputing
  the threshold from WR+RB only moves §L2 from −0.148 (p = .079) to −0.124 (p = .108) and §L4 from
  +0.076 (p = .502) to +0.073 (p = .461). No conclusion changes.

---

## The finding that survives: the gap between the two outcome definitions

§L0 predicted that the total-vs-PPG gap would itself be a finding, and it is the only clearly
non-null structure §L produced. At R1–2, decomposing the RB-minus-WR gap in mean season points:

| | RB | WR |
|---|---|---|
| mean games played | **13.18** | **14.28** |
| mean PPG | 16.43 | 16.89 |
| mean season total | 222.8 | 243.5 |
| share playing < 4 games | 6.5% | 2.0% |

Total gap **−20.7 points/season**, of which the **games channel is −18.6** and the PPG channel only
−6.1 (residual +4.0 from the Jensen term). **Roughly 90% of the elite-RB shortfall in what a
drafter actually accrues is availability, not per-game production.** Consistently, §L2's RB−WR gap
shrinks from −0.148 to −0.070 when the outcome switches from totals to PPG-given-participation, and
the top-12 gap is ~0 under both.

This is descriptive, not a family test, and it is not new information — §A already established
availability as a stable, modellable trait, and the RB availability penalty is well known. It is
recorded because it is the correct reading of the total-vs-PPG gap the plan required, and because
it locates whatever positional difference exists in *games*, where the project already has a model,
rather than in *conversion*, where §L finds none.

---

## §L5 Decision — screens and holdout

**Family declared before fitting** (recorded here as executed): the confirmatory family is the
tests on the **primary outcome definition (season totals)**, per §L0's designation — §L2 × 2 hit
definitions, §L3 linear trend × 2, §L3 pre-specified split × 2, §L4 interaction × 2 = **8 tests**.
PPG-given-participation versions, the mixed-model robustness checks, the WR+RB-median sensitivity
and the §D7 post-hoc regression are reported as **sensitivities with `family = 0`** and are not
corrected — none is used to support a claim.

BH at q = 0.10 over the 8:

| rank | test | hit def | estimate | p | BH threshold | reject |
|---|---|---|---|---|---|---|
| 1 | §L2 | value_T | −0.148 | .079 | .0125 | no |
| 2 | §L4 | value_T | +0.076 | .317 | .0250 | no |
| 3 | §L3 | top12_T | +0.065 | .332 | .0375 | no |
| 4 | §L3 | value_T | +0.014 | .801 | .0500 | no |
| 5 | §L3split | top12_T | +0.030 | .813 | .0625 | no |
| 6 | §L2 | top12_T | −0.011 | .866 | .0750 | no |
| 7 | §L3split | value_T | −0.010 | .918 | .0875 | no |
| 8 | §L4 | top12_T | −0.050 | .964 | .1000 | no |

**0 of 8 survive.**

**Temporal holdout, 2015–21 → 2022–24** (`sectionL_holdout.csv`): sign stability is 2 of 4.

| quantity | fit 2015–21 | holdout 2022–24 | sign held |
|---|---|---|---|
| §L2 top-12 RB−WR | +0.011 | −0.061 | no |
| §L2 value RB−WR | −0.151 | −0.143 | yes |
| §L4 top-12 interaction | +0.032 | −0.253 | no |
| §L4 value interaction | +0.066 | +0.097 | yes |

The two that hold are both on the definition §D7 disqualifies, and both point away from the
hypothesis. Nothing enters the round-6 strategy work as a positional prior.

**Closed families untouched:** {§H5, §I3} and {§K} were not reopened or re-corrected.

---

## What §L establishes, stated precisely

1. **Claim (a) is not supported.** Conditional on being drafted in the first two rounds, RBs and
   WRs reach a top-12 positional finish at 53.7% vs 52.5% — a 1.1 pp difference against a 19 pp
   MDE, with the point estimate on the hypothesis's *wrong side* under all four definitions and in
   both league frames. Elite conversion at both positions is, to the precision available, the coin
   flip the owner attributes only to WRs. What §L cannot do is rule out a moderate RB edge of
   ~10 pp.
2. **Claim (b) is untestable at this sample size, and is reported as such.** The MDE is ~5 pp/yr
   (50 pp over the window); the split test's MDE is 50 pp. The point estimate leans up, and the
   confound obliges the note that elite RBs were 2.1 picks *cheaper* in 2022–24 than before, which
   is the direction that raises a hit rate for free. **Uninformative, not absence.**
3. **Claim (c) is not supported, and this null is the informative one.** The WR-minus-RB gap does
   not vary with draft cost (joint Wald p = .96 on the clean definition), even though the
   contaminated definition contains a mechanism that grows with depth and should have manufactured
   the claimed pattern.
4. **A tier-level pattern was hypothesised precisely because ADP is formed player by player and is
   under no pressure to correct it. It is nevertheless absent.** After six player-level nulls, §L
   adds a seventh at the tier level: the market's structure conditional on (position, cost) is as
   efficient as its structure conditional on player. That was §L5's anticipated result, and it is
   what happened.
5. **The one real positional difference is availability, not conversion** — ~90% of the elite-RB
   deficit in accrued points. It belongs to the availability model, not to a draft-strategy rule.

## Unexplained / open

- Nothing in §L is left unexplained. Every anomaly encountered was chased to a mechanism: the
  DE-name collision, the TE position drift, the QB contamination of the bin median, the slot-budget
  ceiling, and the value-return definition's positional bias.
- One thing §L deliberately does **not** answer: whether elite RBs are the better *pick* in the
  owner's 10-team, 2-flex league. Conversion is not value. A 53% RB hit and a 53% WR hit are not
  worth the same, because replacement level differs by position. That calculation needs the owner's
  actual roster settings and belongs to round 6 — §L's contribution to it is that the *conversion*
  input to that calculation is positionally flat, so any RB-vs-WR preference must come from the
  scarcity weighting, not from RBs justifying their price more often.

---

# §L-EXT — cumulative finish tiers (24 / 36), added on request after §L closed

Same panel, same league-wide ranks, same name resolution — only the finish threshold changes.
`hit24` = positional finish ≤ 24, `hit36` = ≤ 36, both outcome definitions, both frames, and a
pooled **2022–2024** window added to `conversion_rates.csv`. The CSVs now carry `tier`
(`top12`/`top24`/`top36`/`value`) and `outcome_def` (`total`/`ppg_ge4`) columns; nothing was
overwritten.

**Multiplicity:** the original 8-test §L family is **closed** and is neither reopened nor
re-corrected. §L-EXT is a **new declared family, `family = 2`**, BH q = 0.10, comprising the tests
on the primary outcome definition (season totals): §L2 × {hit24, hit36} + §L4 × {hit24, hit36} =
**4 tests**. PPG-given-participation versions and every cell-level comparison are `family = 0`
sensitivities/descriptives. Raw p-values are reported throughout.

## Budget integrity check — one violation, diagnosed

Asserting ≤ 12 / ≤ 24 / ≤ 36 drafted hits per position per season across all 60 position-seasons:
**one violation, WR 2024 at tier 12 (13 hits).** Cause is not a data defect: **Ladd McConkey and
Jerry Jeudy both finished on exactly 240.90 points**, an exact tie at rank 12 under `method="min"`,
so both are counted. This is arithmetically correct under the pre-registered "finish 1–12" rule,
affects one cell, and both players sit in the same bin (R9+), so breaking the tie the other way
moves WR R9+ 2024 by one hit in 26. **The rank rule was not changed after seeing results.** Tiers
24 and 36 have zero violations (max 24 and 34 respectively).

## The ceiling — it does not bind per-bin, but the *joint* budget nearly saturates

The coordinator's concern was that with 64–79 drafted players per position at R9+ the 24/36 slots
would constrain achievable rates. That figure is the **3-season pooled n**; the constraint operates
**per season**, where R9+ holds 21.7 RB / 26.8 WR. So the per-bin ceilings are:

| tier | R1–2 | R3–4 | R5–6 | R7–8 | R9+ |
|---|---|---|---|---|---|
| ≤12 ceiling (RB / WR) | .94 / .97 | 1.0 / .98 | 1.0 / 1.0 | 1.0 / 1.0 | **.57 / .46** |
| ≤24 ceiling | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | .98 / .88 |
| ≤36 ceiling | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |

**In no cell, at any tier, is the realized rate within 10 pp of its ceiling** — minimum slack across
all 30 cells is 17.4 pp (RB R1–2 at tier 36: ceiling 1.00, realized .826). The ceiling loosens as
the tier widens, which is the opposite of the concern: at ≤36 no bin is constrained at all.
Diagnostic saved to `results/sectionL_ceiling.csv`.

**But a different budget constraint does nearly bind, and it matters for reading the table.**
Drafted players occupy essentially the whole positional budget every season:

| tier | RB slots used | WR slots used |
|---|---|---|
| ≤12 | 11.5 / 12 (95.8%) | 11.7 / 12 (97.5%) |
| ≤24 | 22.7 / 24 (94.6%) | 22.3 / 24 (92.9%) |
| ≤36 | 32.0 / 36 (88.9%) | 31.1 / 36 (86.4%) |

So the **across-bin profile is a near-fixed pie**: a bin's rate can only rise if another bin's
falls. That is why the monotone decline across bins is not "conversion skill" — it is an allocation.
**The across-position contrast inside a bin is not constrained**, because RB and WR have separate
12/24/36 budgets. This is the precise sense in which §L2 and §L4 remain interpretable while the raw
bin profile is not.

## Conversion by bin and tier — 12-team frame, season totals

**Pooled 2015–2024** (n identical across tiers within a cell; rates cumulative):

| bin | n RB / WR | ≤12 RB / WR | ≤24 RB / WR | ≤36 RB / WR |
|---|---|---|---|---|
| R1–2 | 123 / 101 | .537 / .525 | .724 / .683 | .829 / .812 |
| R3–4 | 90 / 115 | .244 / .278 | .578 / .600 | .722 / .722 |
| R5–6 | 81 / 97 | .123 / .113 | .333 / .309 | .506 / .485 |
| R7–8 | 92 / 85 | .087 / .106 | **.304 / .176** | **.489 / .353** |
| R9+ | 217 / 268 | .041 / .045 | .143 / .149 | .309 / .257 |

**Pooled 2022–2024** (the window the owner is asking about):

| bin | n RB / WR | ≤12 RB / WR | ≤24 RB / WR | ≤36 RB / WR |
|---|---|---|---|---|
| R1–2 | 35 / 34 | .571 / .588 | .743 / .735 | .886 / .882 |
| R3–4 | 24 / 31 | **.333 / .161** | .708 / .581 | .833 / .710 |
| R5–6 | 24 / 34 | .125 / .059 | .333 / .235 | .458 / .500 |
| R7–8 | 25 / 26 | .120 / .154 | .400 / .231 | .560 / .423 |
| R9+ | 64 / 79 | .016 / .063 | .141 / .152 | .375 / .228 |

**10-team frame, pooled 2015–2024** — R1–2 ≤12 RB .532 / WR .593; ≤24 .743 / .691; ≤36 .835 / .827.
**10-team, 2022–2024** — R1–2 ≤12 .586 / .621; R3–4 ≤12 .375 / .200; R7–8 ≤24 .480 / .273.
Wilson bounds for every cell are in `conversion_rates.csv`. Figures:
`results/figures/sectionL_tiers_2015_2024.png`, `sectionL_tiers_2022_2024.png`.

## §L2 and §L4 re-run at hit24 / hit36 — still null

R1–2 position gap, season-clustered t (9 df):

| hit definition | RB − WR | SE | 95% CI | raw p | MDE |
|---|---|---|---|---|---|
| ≤24 (total) | **+0.026** | 0.063 | [−0.117, +0.169] | .690 | 0.199 |
| ≤36 (total) | **+0.022** | 0.044 | [−0.078, +0.121] | .634 | 0.139 |
| ≤24 (PPG\|≥4g) | +0.038 | 0.041 | — | .384 | 0.129 |
| ≤36 (PPG\|≥4g) | +0.041 | 0.029 | — | .188 | 0.091 |

Position × bin interaction:

| hit definition | joint Wald p (4 df) | linear-in-bin RB×bin | SE | p | MDE |
|---|---|---|---|---|---|
| ≤24 (total) | .304 | +0.006 | 0.116 | .958 | 0.365 |
| ≤36 (total) | .810 | +0.068 | 0.101 | .520 | 0.319 |
| ≤24 (PPG\|≥4g) | .615 | −0.093 | 0.077 | .259 | 0.244 |
| ≤36 (PPG\|≥4g) | .635 | −0.064 | 0.093 | .509 | 0.292 |

**BH q = 0.10 over the 4-test §L-EXT family: 0 of 4 (smallest raw p = .304).**

Note the tier gradient in the §L2 estimates: the RB−WR gap is −0.011 at ≤12, **+0.026** at ≤24,
**+0.022** at ≤36 — it crosses zero and turns weakly positive as the tier widens, and the MDE
tightens from 0.190 to 0.139 because wide-tier rates are less variable across seasons. The sign
flip is well inside noise (all three CIs contain zero and each other), but the direction is worth
recording: whatever small edge elite RBs have is in *avoiding a bust* (finishing ≤ 24 or ≤ 36), not
in *hitting the top 12*. That is consistent with the availability finding — an RB who stays healthy
lands somewhere useful — and inconsistent with the "elite RBs justify their ADP more" framing,
which is a top-tier claim.

## The R3–4, 2022–2024 cell — the owner is acting on it, so here is the anatomy

**Verdict: not distinguishable from noise, and not consistent across the three seasons.**

Top-12, R3–4, by season:

| season | RB | WR | gap |
|---|---|---|---|
| 2022 | 1/9 = .111 | 2/11 = .182 | **−0.071** |
| 2023 | 4/8 = .500 | 2/10 = .200 | +0.300 |
| 2024 | 3/7 = .429 | 1/10 = .100 | +0.329 |
| **pooled** | **8/24 = .333** [.180, .533] | **5/31 = .161** [.071, .326] | **+0.172** |

**Fisher exact p = .202** (χ² p = .136). The Wilson intervals overlap across most of their range.
**2022 runs the opposite way**, so this is two seasons of three, not a stable regime — and each
season's cell holds 7–11 players per position, an SE of ~16 pp, so the per-season numbers carry
almost no information individually. **Fragility: the entire pooled gap is 4.1 players.** If four of
the 24 mid-round RBs had finished RB13 instead of RB12 the gap vanishes. This is exactly the size
of perturbation a single season's injury luck produces.

It attenuates at the wider tiers rather than strengthening:

| tier | RB | WR | gap | Fisher p | seasons positive |
|---|---|---|---|---|---|
| ≤12 | 8/24 = .333 | 5/31 = .161 | +0.172 | .202 | 2 of 3 |
| ≤24 | 17/24 = .708 | 18/31 = .581 | +0.128 | .403 | 2 of 3 |
| ≤36 | 20/24 = .833 | 22/31 = .710 | +0.124 | .349 | 2 of 3 |

2022 is negative at all three tiers (−.071, −.081, −.152); 2024 is the strongest at all three.

**Why it does not contradict the pooled interaction (p = .96).** It is one cell of 15
(5 bins × 3 tiers, ×2 positions) selected *after* seeing the table, in a 3-season window, against a
decade-long test. The pooled decade R3–4 gap at ≤12 is **−3.4 pp** (RB .244 vs WR .278, n = 90/115)
— the opposite sign. A 3-year subwindow moving ±20 pp against a decade mean is what a 16 pp
standard error looks like. **It is a post-hoc cell, it is in `conversion_tests.csv` with
`family = 0`, and it is not evidence.**

## One more post-hoc pattern, reported because it is the largest in the extension

The widest same-position gap the tier extension surfaces is **not** R3–4 but **R7–8 at ≤24 over the
full decade: RB .304 (28/92) vs WR .176 (15/85), +12.8 pp**, positive in **7 of 10 seasons**,
season-clustered t p = **.103** (Fisher on pooled counts p = .055). At ≤36 it is +13.6 pp
(.489 vs .353), also 7 of 10 seasons, p = .186.

There is a plausible mechanism — RB touches are concentrated on one back, so a late-round RB who
inherits a job jumps straight into RB2 territory, whereas late-round WR value is spread over more
bodies — and it is the shape a "handcuff/contingent value" story would predict. **It is still
post-hoc**: one cell of ~30 scanned after the fact, p = .10 under the correct error structure, and
it is *not* what the owner's hypothesis claims (his claim is about rounds 1 and 5–8 for WRs, not
round 7–8 for RBs). It is recorded as a candidate for a future pre-registered test, not as a
finding. **Nothing here changes the §L verdict or enters any model.**

## Extension summary

- 0 of 4 new-family tests survive BH q = 0.10; the closed 8-test family is untouched.
- Widening the tier from ≤12 to ≤24 to ≤36 does not create a positional edge at R1–2; the estimate
  drifts from −1.1 pp to +2.6 pp to +2.2 pp, all inside a 14–20 pp MDE.
- The position × bin interaction is null at every tier (joint Wald p = .30, .81 on totals).
- The per-bin slot ceiling never binds (≥ 17 pp slack everywhere); the joint positional budget is
  ~87–98% saturated, which is why the across-bin profile must be read as an allocation and only the
  within-bin position contrast is interpretable.
- The R3–4 2022–24 gap the owner is acting on is +17.2 pp, p = .202, driven by 2023–24 with 2022
  reversed, worth 4.1 players, and opposite in sign to the pooled decade. **Not distinguishable
  from noise at n = 24 vs 31.**

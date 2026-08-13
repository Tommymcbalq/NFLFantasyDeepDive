# §O — TE and QB valuation, refit from nothing

Executed against `EDA_PLAN6.md` §O, pre-registered 2026-08-09 before any §O fitting.
Structure mirrors §G (the RB refit) exactly; **no variance component, threshold, gate,
market curve or shrinkage weight is inherited from the WR or RB pipelines.**

Scripts: `scripts/34_sectionO_o1_o3.py` (§O1/§O2/§O3), `35_sectionO_o4_market_prior.py`
(§O4), `36_sectionO_o5_loso.py` (§O5 + the 2026 boards),
`37_sectionO_o6_o7_vorp.py` (§O6/§O7), `38_sectionO_chases.py` (anomaly chases),
`39_sectionO_contested_baseline.py` (§O7-R: the corrected primary baseline + TE-cliff test).

Outputs: `sectionO_universe_2026.csv`, `consistency_table_{te,qb}.csv`,
`variance_components_{te,qb}.csv`, `heteroskedasticity_{te,qb}.csv`,
`sigma2_by_tier_{te,qb}.csv`, `sectionO_o3_prediction.csv`,
`sectionO_qb_rush_decomp.csv`, `market_prior_{te,qb}.csv`,
`tier_variances_{te,qb}.csv`, `market_prior_iso_knots_{te,qb}.csv`,
`sectionO_iso_diagnostics.csv`, `sectionO_qb_resid_rush.csv`,
`loso_scorecard_{te,qb}.csv`, `loso_predictions_{te,qb}.csv`,
`valuation_te_2026.csv`, `valuation_qb_2026.csv`, `sectionO_shrinkage.csv`,
`vorp_all_positions.csv`, `sectionO_board_2026_vorp.csv`,
`sectionO_premium_by_baseline.csv`, `sectionO_chases.csv`,
`sectionO_contested_baseline.csv`, `sectionO_premium_contested.csv`,
`sectionO_vorp_contested.csv`, `sectionO_te_cliff.csv`.

---

## Headline

1. **Both positions are market-anchored.** The pre-registered data arm fails LOSO at both:
   TE DM t = +0.446, p = .667; QB DM t = +0.542, p = .601. The §O5 honesty clause fires
   twice and `board_value = m(ADP)` for both boards. No further arms were tried. The 2026
   TE and QB boards therefore **reproduce the ADP ordering exactly** — every
   `delta_rank_vs_adp` is 0.

2. **The §O3 pre-registered prediction is FALSIFIED on its operational half, and it is
   recorded as falsified.** QB PPG *is* the least noisy per game in scale-free terms
   (CV 0.435 against WR .562 / RB .533 / TE .594) — but the operational claim, "QB μ̂ is
   more reliable so B shrinks *less* toward market", is wrong: realized mean B is
   **QB 0.683, the highest of the four positions** (RB .603, TE .629, WR .658).

3. **The "elite-TE premium" is not elite.** Cut into *disjoint* positional-rank bands
   rather than nested ones, the TE premium against R_real is +18.1 (TE1–3), +14.7 (TE4–6),
   +21.1 (TE7–12) and **+36.9 (TE13+, p < .001, 10/10 seasons)** — it does not decay, it
   *grows* toward the cheap end. The decay contrast is −3.0 ± 13.6 (p = .83). What §M
   measured as an elite-TE effect is a **replacement-level level-shift**, and the
   actionable version of it is a *late*-TE argument, not an elite-TE argument.

4. **QB carries no premium at any band under any baseline that matters**, confirming §M.

5. **BINDING CORRECTION to §O7's baseline (see §O7 below).** The weekly-foresight
   baseline `R_week` is incoherent for a contested pool and is demoted to a labelled upper
   bound. The primary is now `R_cont`, a contested, no-weekly-foresight,
   hoarding-adjusted replacement. **Under it the elite-TE premium is −27.1 ± 5.5 points
   (p < .001, 10/10 seasons) and the elite-QB premium is +1.3 (p = .93).** The mechanism
   is bench hoarding, not foresight: opponents roster 5.03 RB and 5.85 WR but only 1.37
   TE, so good TEs stay free while good RBs do not. Against the starting-slot baselines
   (`R_exp`, `R_real`) the TE premium is +12.7 to +18.5. Both families are reported.

6. **The owner's TE-cliff claim is strongly true in *identifiability* and not in points.**
   The top-5 ADP TEs deliver a top-5 TE finish 58% of the time against 13% for TE6–12 —
   a gap of **+0.451 (p < .0001, 10/10 seasons)**, five times RB's +0.091 and the sharpest
   positional result in §O. But that identifiability carries **no VORP premium** under the
   primary baseline, which is what a correctly-priced position looks like.

---

## §O1 Universe and the join §L never tested

Top 24 TE and top 24 QB by 2026 ADP (`adp_ppr_2026_all_20260809.csv`; the board carries 25
TE and 30 QB, so "top 24" is essentially the whole rosterable population). Historical panel
= every TE/QB on the FFC board 2015–2024.

**Join verification.** The §L join (`26_sectionL_conversion.py`, three validated fixes),
lifted via `sectionM_common.build_panel`: **187 TE and 238 QB board rows 2015–2024, 0 null
player ids, 0 unmatched, 0 unresolved ambiguities.** 2026 universe: 47 of 48 matched to
weekly rows; 1 TE (a rookie with no NFL rows) matched via `players_meta` and carries
n_eff = 0 ⇒ B = 1 ⇒ full shrinkage to market, flagged.

**Cross-era caveat**: per-game throughout; 16→17 games in 2021 and COVID 2020 left in
unadjusted, as everywhere in this project.

## §O2 Inclusion rule — fixed by the plan, and the exclusions are very different

| | rule | player-games 2014–25 | excluded | % | mean PPR of excluded |
|---|---|---|---|---|---|
| **TE** | targets ≤ 1 | 14,014 | 5,034 | **35.92%** | **1.302** |
| **QB** | pass attempts ≤ 5 | 7,480 | 821 | **10.98%** | **0.533** |
| WR (§0) | targets ≤ 1 | — | — | 1.9% | 1.9 |
| RB (§G1) | touches ≤ 1 | — | — | 17.2% | 0.454 |

Restricted to *board* players the exclusions collapse to **6.50% (TE, mean 1.82)** and
**3.02% (QB, mean 1.27)**. So the 36% figure at TE is almost entirely the blocking-TE
population, not the drafted one — TE is the position whose raw player pool is most diluted
by non-receiving bodies. QB's 11% is the backup/mop-up mixture the plan predicted.

Frozen positional boom/bust thresholds (p75/p25 of qualified games): **TE 11.1 / 3.4**,
**QB 20.9 / 9.7** (WR 20/8, RB 13.8/3.2).

Relevance gates, retention-matched to WR's 82.5% as §G did: **TE 2 targets/game**
(non-binding — the participation filter already guarantees it; the median-usage
sensitivity gate of 3 retains 63.4% and is the informative one) and **QB 21
attempts/game** (retains 82.9%; sensitivity 31, retains 49.0%).

## §O3 Variance components

Crossed decomposition Y_isg = μ + a_i + b_is + c_{team×season} + ε, headline 2021–2025 with
exclusions, MoM covariance-matching (REML agrees; both in the CSVs).

| | σ²_P | σ²_S | σ²_T | σ²_G | ρ_max | median σ_W | median CV |
|---|---|---|---|---|---|---|---|
| WR | 5.33 | 3.18 | 0.58 | 69.82 | .404 | 7.75 | .562 |
| RB | 6.67 | 7.70 | −1.68 | 63.36 | .406 | 7.99 | .533 |
| **TE** | **3.92** | **0.74** | **1.02** | **45.30** | **.470** | **6.40** | **.594** |
| **QB** | **4.74** | *(see below)* | *(see below)* | **58.22** | **.455** | **7.86** | **.435** |

**An identification failure at QB, chased to its mechanism.** The QB 4-way MoM returns
σ²_S = **−17.66** with σ²_T = **+19.92**. This is not a bug and not an estimation accident:
there is **one starting QB per team-season**, so the player-season and team-season
partitions are nearly the same partition and only their *sum* is identified. The sum is
+2.26, and the identified 3-way (player / player-season / game) decomposition returns
σ²_S = **2.26** exactly. REML "solves" it only by hitting the σ²_S ≥ 0 boundary and
splitting 0.18/3.23 — an arbitrary split of a non-identified quantity. **The QB headline is
therefore the 3-way model: σ²_P = 4.74, σ²_S = 2.26, σ²_G = 58.22, ρ_max = .455**, and any
team-environment component at QB is inseparable from the QB himself. TE has 695 teammate
pairs and is properly identified; QB has 194, all from starter changes.

**Season-to-season movement.** Adjacent-season persistence of season means:
**TE 0.629**, WR 0.570, **QB 0.389**, RB 0.245. TE is the *most* persistent position in the
project. Consistently, untruncated τ̂²_B (eq. 3) is negative in only 3/14 TEs (median
+3.02) and 3/16 QBs (median +4.67), against WR's 10/20 (median −0.08) and RB's 4/17.

**§O3c heteroskedasticity by experience tier.**
σ²(tier): TE rookie 22.44 / soph 23.86 / vet 27.99 (Wald p = .015);
QB rookie 43.88 / soph 49.58 / vet 47.91 (**Wald p = .474, a null**).
The TE tier effect is entirely a level effect: log e² on log μ has slope **1.732**, and
controlling for level the rookie multiplier *flips* to 1.169 (p = .031). Same mechanism WR
had. σ²(tier) is used as estimated, with no ordering imposed.

### Pre-declared anomaly chase: is the QB tier structure experience, or rushing?

**Rushing.** The experience tiers are null (Wald p = .474); rush volume is not.
Per-game variance by player-season rush-carry tercile:

| tercile | n | carries/g | mean e² | mean PPG |
|---|---|---|---|---|
| low | 219 | 1.37 | 35.33 | 12.60 |
| mid | 217 | 2.82 | 41.41 | 14.44 |
| **high** | 218 | **5.71** | **44.01** | **16.58** |

Gamma GLM e² ~ tier + rush tercile: rushing-high **×1.244 [1.126, 1.374], p < 10⁻⁴**,
while rookie ×0.879 (p = .111) and soph ×0.989 (p = .871). Continuous:
**+4.9% variance per carry/game (p = .001)**. **The QB variance axis is rushing volume, not
experience** — the pre-registered σ²(tier) is kept as specified, and the rushing
decomposition is recorded as a diagnostic, not swapped in.

This carries into §O4: the QB market-prior residual is also a rushing residual. Mean
isotonic residual by same-season rush quartile: **−1.16 / −0.14 / −0.11 / +1.41**;
Spearman(carries/g, residual) = **+0.308 (p = 2.3×10⁻⁶)**; regression **+0.426 PPG per
carry/game (year-clustered p = .0002, R² = .108)**. The six largest QB residuals in the
decade are Lamar 2019, Mahomes 2018, Dak 2020, Allen 2020, Newton 2015, Rodgers 2020.
**This is contemporaneous, not preseason-knowable** — rush volume is realized, so it is a
decomposition of the residual and *not* an edge. Turning it into a forecast would require a
*lagged* rushing feature, which is a new arm; §O5's honesty clause bars adding one here.
Recorded as the obvious next pre-registration.

## §O4 Market prior

Isotonic PPG on log ADP, monotone decreasing, per position; OLS on log ADP as reference;
< 4-game rows flagged and excluded from the fit and from τ².

| | n panel | in fit | OLS intercept | OLS slope | R² | RMSE OLS | RMSE iso | iso levels | m range |
|---|---|---|---|---|---|---|---|---|---|
| TE | 187 | 179 | 21.91 | −2.497 (.333) | .257 | 2.792 | **2.641** | 13 | 16.85 → 4.40 |
| QB | 238 | 227 | 31.15 | −3.017 (.379) | .201 | 3.218 | **3.099** | 12 | 21.83 → 15.31 |

Isotonic beats the line at both, as it did at WR (3.32 vs 3.40).

τ²(tier), as estimated, **no ordering imposed** (and the expected ordering fails at both,
as it did at WR and RB):

| tier | TE τ² | [95%] | QB τ² | [95%] |
|---|---|---|---|---|
| rookie | 10.68 | [1.90, 15.96] | 10.17 | [4.56, 13.80] |
| soph | 8.98 | [4.10, 14.01] | **17.02** | [9.14, 25.14] |
| vet | 6.70 | [5.21, 8.24] | 8.54 | [6.69, 10.57] |

### Pre-declared anomaly chase: does one elite TE distort the isotonic fit?

**No — and the reason is worth stating, because it is the opposite of the expectation.**
The six largest TE residuals are *cheap* TEs who overperformed (Reed 2015 at ADP 158.4,
z = 3.05; Kittle 2018 at ADP 133.0; Andrews 2021 at ADP 52.8), not an elite TE at the top.
The top isotonic level (m = 16.85, ADP 5.9–19.9) is supported by **8 observations** whose
PPG span 13.9–20.9, so no single season carries it. Dropping the largest residual and
refitting moves m by at most **0.197 PPG anywhere and 0.000 in the top ADP quintile**.
**Isotonic's pooling is exactly the property that immunises it**: a Kelce-type season lands
in a level shared with seven others rather than becoming its own knot. The trap is real for
a smooth or a per-player fit; it does not bite here. Same check at QB: max |Δm| = 0.137.

Raw decile means show 3 (TE) and 4 (QB) adjacent reversals that isotonic flattens — the
monotonicity constraint is doing real work at both positions.

## §O5 LOSO — the honesty clause fires at both positions

Everything refit per fold: m_{−Y}, τ²_{−Y}(tier), σ²_{−Y}(tier); μ̂/n_eff from seasons
strictly before Y (h = 1, weekly_raw back to 1999). Adoption required DM p < 0.10 **and**
RMSE improvement.

| pos | arm | RMSE | mean Spearman | DM t | p | folds improved | drop-one-fold p range |
|---|---|---|---|---|---|---|---|
| TE | (i) market m(ADP) | 2.8264 | .4449 | — | — | — | — |
| TE | (ii) blind θ* | 2.7984 | .4656 | +0.446 | **.667** | 6/10 | .335–.978 |
| QB | (i) market m(ADP) | 3.2869 | .4306 | — | — | — | — |
| QB | (ii) blind θ* | 3.2536 | .4644 | +0.542 | **.601** | 8/10 | .216–.899 |

**Neither adopted. Both boards are market-anchored: `board_value = m(ADP)`.**
Per-fold gains: TE +0.154 (across-fold SD 1.092), QB +0.177 (SD 1.033), against WR
+0.695/0.819 and RB +0.488/2.015. As at RB, this is a **statement about power, not about
the positions being unpredictable**: the point estimates improve RMSE at both positions and
QB wins 8 of 10 folds, but the per-fold gain is a quarter of WR's against comparable
dispersion. No single fold drives either result.

### The §O3 prediction, settled — and falsified

The plan predicted: *"QB PPG should be far less noisy per game than WR/RB … which if true
means QB μ̂ is more reliable and B should shrink less toward market."* Both halves tested.

**Half 1 — is QB less noisy per game? Split verdict.** In absolute σ²_G, QB (58.22) is
below WR (69.82) and RB (63.36) — but **TE (45.30) is far below QB**, so the implied story
("the position without touch-share volatility is the quiet one") is not what the data say;
the quietest position is the one that scores least. Scale-free, the prediction lands
cleanly: **CV = σ_W/μ is 0.435 at QB against .533 RB, .562 WR, .594 TE**, and σ_G/μ̄ is
0.426 vs .538/.568/.619. QB games really are the most predictable *relative to what a QB
scores*.

**Half 2 — does B shrink less toward market at QB? No. The opposite.**

| pos | mean B (has data) | mean σ²(tier) | mean τ² | mean V = σ²/n_eff | σ²/τ² |
|---|---|---|---|---|---|
| RB | **0.603** | 41.51 | 13.03 | — | 3.19 |
| TE | 0.629 | 26.17 | 7.05 | 12.40 | 3.71 |
| WR | 0.658 | ~43.1 | ~11.3 | — | 3.81 |
| **QB** | **0.683** | **46.29** | **9.48** | **21.75** | **4.88** |

**QB shrinks toward market more than any other position.** The mechanism is a scale error
in the prediction, and it is the same error §28 recorded about transplanted MDEs.
Shrinkage is governed by V/(V+τ²) with **both terms in points², not in CV units**. QB's
per-game variance is large in points² *because QB scores 18 PPG*; but τ², the spread of
outcomes around the market price, does **not** scale up proportionally (QB vet τ² = 8.54
against RB 13.03). So the ratio σ²/τ² — the only thing B cares about — is **worst at QB**.
A position can simultaneously have the most reliable per-game signal *relative to its own
level* and the least reliable one *relative to the prior it is being blended with*. The
prediction conflated those two, and it is recorded as falsified rather than restated.

### The 2026 boards

Because both arms failed, `board_value = m(ADP)` and **the model board is the ADP board**;
`delta_rank_vs_adp = 0` for all 48 players. θ* is still reported in the CSVs as a
not-adopted diagnostic (its largest disagreements: Kittle θ* 12.00 vs m 10.06 at ADP 121;
McBride 14.48 vs 13.48; Hurts 18.59 vs 17.63; Mahomes 18.36 vs 17.63).

`valuation_te_2026.csv` (top): McBride 13.48 PPG (ADP 35.3) · Bowers 12.89 (40.7) ·
Loveland 11.63 (60.8) · Warren 11.63 (65.8) · Pitts 10.35 (78.5) · Fannin 10.35 (81.5) ·
LaPorta 10.35 (87.6) · Kelce 10.35 (99.6) · Goedert 10.35 (105.3) · Kraft 10.06 (110.6).
Note the long flat step: **ADP 74.6–110.5 is a single level at 10.35–10.06 PPG**, so seven
TEs are worth the same to two decimal places. Read the value gap, not the rank gap (§21).

`valuation_qb_2026.csv` (top): Allen 21.53 PPG (ADP 26.7) · Maye / Jackson / Burrow /
Prescott 19.52 (49.6–67.3) · Daniels / Hurts / Purdy / Stafford / Lawrence / Mahomes /
Dart / Goff / Williams / Herbert 17.63 (71.3–108.5). **Ten QBs share one isotonic level.**
Thin-data flags: 1 TE with no NFL rows (B = 1), 4 TE and 2 QB single-season (n_eff = 1).

## §O6 VORP under the owner's league, on one scale with RB/WR

Demand, 10-team: 10 QB, 10 TE, 20 RB + 20 WR locked, 20 FLEX allocated by realized usage
(3.5 RB / 16.5 WR) ⇒ effective 23.5 RB, 36.5 WR. 12-team: 12/12/28.8/43.2.

**The frame gap, named:** the ADP is FFC **12-team**; the owner's league is **10-team**.
Every number below carries its frame. Opponent ordering comes from the 12-team board;
demand and replacement come from the 10-team structure.

Replacement (season-total PPR, means 2015–2024), all three baselines:

| frame | pos | D | R_exp | R_real | R_week | bracket width |
|---|---|---|---|---|---|---|
| 10 | QB | 10 | 244.8 | 281.0 | 334.7 | **89.9** |
| 10 | TE | 10 | 130.8 | 147.5 | 201.3 | **70.5** |
| 10 | RB | 23.5 | 145.9 | 170.1 | 195.8 | 49.9 |
| 10 | WR | 36.5 | 148.4 | 169.8 | 194.4 | 45.9 |
| 12 | QB | 12 | 241.4 | 260.8 | 311.6 | 70.1 |
| 12 | TE | 12 | 129.4 | 137.4 | 180.7 | 51.3 |
| 12 | RB | 28.8 | 136.2 | 153.7 | 166.7 | 30.5 |
| 12 | WR | 43.2 | 140.9 | 153.4 | 166.4 | 25.4 |

### VORP by 10-team draft round, four positions, baseline R_real (`vorp_all_positions.csv`)

| round | QB | RB | TE | WR |
|---|---|---|---|---|
| 1 | — | 67.9 | **83.9** | **87.8** |
| 2 | −14.2 | 43.5 | **97.1** | 74.4 |
| 3 | 47.2 | 11.7 | 33.2 | 42.5 |
| 4 | 50.2 | 10.2 | 38.6 | 34.0 |
| 5 | 9.8 | −13.3 | **40.3** | 26.6 |
| 6 | −24.0 | −32.2 | −11.2 | 0.6 |
| 7 | 9.6 | −39.7 | 25.5 | −11.2 |
| 8 | −34.4 | −20.8 | −24.0 | −10.9 |
| 9 | −27.0 | −59.7 | −27.3 | −22.8 |
| 10 | −61.2 | −34.7 | 2.3 | −25.1 |
| 11 | −52.0 | −53.1 | −8.2 | −32.0 |
| 12 | 0.2 | −58.4 | −28.7 | −30.8 |
| 13 | −55.4 | −65.4 | −4.7 | −49.8 |
| 14 | −78.9 | −88.7 | −30.2 | −28.0 |

Same table under R_exp and R_week, both frames, and by positional ADP rank, is in
`vorp_all_positions.csv` with season-clustered SEs. **Cell SEs are 10–50 points on n = 10
seasons; individual cells are not interpretable and are not interpreted.** The shape is.

Baseline-dependent summary of which position leads each 10-team round:

- `R_exp`: WR, TE, QB, QB, TE, WR, QB, WR, QB, TE, TE, QB, TE, WR
- `R_real`: WR, TE, QB, QB, TE, WR, TE, WR, WR, TE, TE, QB, TE, WR
- `R_week`: **WR at 12 of 14 rounds** (TE at R7 and R13)

**Under a streaming baseline the entire cross-positional argument collapses to "take a
WR"**, which is the single most important sentence in §O6.

## §O7 Every VORP against every baseline — and a BINDING CORRECTION to which one is primary

### The correction, recorded before the numbers

§O7 as pre-registered used three baselines and rested its headline on `R_real`. The owner
raised a methodological objection to the widest of them, `R_week`, which is **accepted as
binding**: R_week is not merely optimistic, it is **incoherent for a contested pool**. It
assumes simultaneously (a) perfect weekly foresight about which streamer will hit and
(b) exclusive access to the wire. In a 10-team league nine other managers draw on the same
pool and cannot all hold the best free TE. Differencing a drafted player's realized value
against R_week compares two quantities computed under mutually incompatible assumptions.

**R_week is therefore demoted to a labelled upper bound on what streaming could
theoretically return to one manager if nobody competed and he guessed right every week. It
is retained and reported. No recommendation rests on it.** Correspondingly, §M's finding
that the TE premium goes to −10.2 against R_week is **an artifact of an infeasible
counterfactual, not evidence against elite TE**, and is reported as such.

**This is a post-hoc change of primary baseline and is flagged as one.** It is made on an
a-priori coherence argument about the counterfactual, not because a number was unwelcome —
and the record shows it did not in fact flatter the conclusion it was expected to rescue.

### The contested, no-foresight baseline, computed

Two contested constructions were built rather than one, because "the best player not
rostered by another team" requires saying *how deep teams roster*, and the answer is not
free. Per-team carry is taken from **§M2's own draft simulation of this exact league**
(20,000 drafts, mean S0 roster: **1.75 QB / 5.03 RB / 5.85 WR / 1.37 TE**), not chosen.

- **`R_cont` (PRIMARY, hoarding-adjusted).** The (⌈10·N_p⌉ + 1)-th best player at position
  p by **realized season total over the full pool, undrafted included**. Contested by
  construction — exactly ⌈10·N_p⌉ bodies are held league-wide, so every manager can
  simultaneously be assumed access to one of that quality. No weekly foresight: it is a
  season-total order statistic and never asks which week to start whom. It generalises the
  textbook VORP replacement from *starting* demand (which is `R_real`) to *rostered*
  demand, which is what a waiver wire actually clears.
- **`R_cont_blind` (strict no-information variant).** Roster by preseason expectation, then
  take the next identifiable board player blind. Reported, but **downward-biased at RB/WR**:
  10 × 5.85 rostered WRs nearly exhausts a 66-player FFC board, so its "free WR" is the
  board's tail rather than the wire's best. That asymmetry is an artifact of board length.
- **`R_wire_best`** (top-3 free by realized total) is a *maximum* order statistic, not a
  replacement level, and is reported only to bound the other side.

Season-total PPR, means 2015–2024:

| frame | pos | R_exp | R_cont_blind | **R_cont** | R_real | R_wire_best | *R_week (bound)* |
|---|---|---|---|---|---|---|---|
| 10 | QB | 244.8 | 197.5 | **223.1** | 281.0 | 290.3 | *334.7* |
| 10 | RB | 145.9 | 83.3 | **92.8** | 170.1 | 198.7 | *195.8* |
| 10 | TE | 130.8 | 112.5 | **131.0** | 147.5 | 184.2 | *201.3* |
| 10 | WR | 148.4 | 127.3 | **120.9** | 169.8 | 216.5 | *194.4* |
| 12 | QB | 241.4 | 220.7 | 198.6 | 260.8 | 279.1 | *311.6* |
| 12 | RB | 136.2 | 76.7 | 72.0 | 153.7 | 171.3 | *166.7* |
| 12 | TE | 129.4 | 123.9 | 120.5 | 137.4 | 181.6 | *180.7* |
| 12 | WR | 140.9 | 77.8 | 101.6 | 153.4 | 209.5 | *166.4* |

Carry sensitivity at TE, 10-team: R_cont = 147.5 (carry 1.0) → **131.0 (carry 1.37)** →
109.0 (carry 2.0) → 80.3 (carry 3.0). At QB: 281.0 → **223.1** → 207.5 → 115.6.

**12-team `R_cont_blind` is not computable** — 12 × 5.85 exceeds the board's WR supply in
9 of 10 seasons — and is reported as exhausted, not imputed.

### The premium under every baseline (`sectionO_premium_contested.csv`, `sectionO_premium_by_baseline.csv`)

TE1–5 and QB1–5 over the mean RB/WR within ±6 ADP picks, 10-team frame, season-clustered
t(9):

| baseline | what it assumes | TE1–5 | p | QB1–5 | p |
|---|---|---|---|---|---|
| R_exp | never touch the roster | +12.7 | .03 | +19.0 | .19 |
| R_cont_blind | contested, zero information | −13.5 | .21 | +21.9 | .21 |
| **R_cont (PRIMARY)** | **contested, season info, hoarding-adjusted** | **−27.1** | **<.001** | **+1.3** | **.93** |
| R_real | contested at *starting* demand | +18.5 | .002 | +5.3 | .69 |
| R_wire_best | best free agent (a maximum) | +19.0 | .01 | +34.0 | .05 |
| *R_week* | *infeasible upper bound* | *−10.2* | *.07* | *−23.4* | *.11* |

**Under the primary baseline the elite-TE premium is −27.1 ± 5.5 points a season
(p < .001, negative in 10 of 10 seasons), and there is no elite-QB premium (+1.3, p = .93).**
The disjoint bands under R_cont are −27.0 (TE1–3), −31.7 (TE4–6), −25.3 (TE7–12), −8.1
(TE13+) — again flat, again least bad at the cheap end.

**Chased to the arithmetic, because the sign flip between R_cont and R_real is 45.6 points
and needs a mechanism.** Moving from starting demand to rostered demand lowers each
position's bar by R_real − R_cont: **TE 16.5, QB 57.9, RB 77.3, WR 48.9.** The TE bar
barely moves because teams hold only 1.37 TEs, so the 15th-best TE is nearly the 11th-best
TE. The RB bar collapses because teams hold 5.03 RBs, so the free RB is the 51st-best. The
whole premium is that difference: a drafted RB/WR is far better than what is free at his
position, while a drafted TE is only slightly better than what is free at his.

**So the owner's objection removes the infeasible baseline but does not rescue the
elite-TE premium — a coherent, feasible, contested streaming baseline is *less* favourable
to TE than R_week was, not more.** The streaming argument against paying for TE survives
the removal of clairvoyance, and it survives it because of *bench hoarding*, not foresight:
opponents simply do not hold TEs, so good TEs stay free.

**The honest bracket, stated once.** The TE premium is +12.7 to +19.0 against baselines
that price a *starting slot* (R_exp, R_real, R_wire_best) and −13.5 to −27.1 against
baselines that price *what is actually free* (R_cont_blind, R_cont). Both families are
coherent; they answer different questions, and the difference between them is entirely
roster depth at TE relative to RB/WR. **The recommendation below rests on `R_cont`**, per
the correction. A reader who believes his league hoards TEs (carry ≥ 2) should read the
carry sensitivity, where R_TE falls to 109.0 and the premium moves back toward zero.

**Caveat on the primary, stated plainly:** N_p is endogenous. It was measured from
simulated opponents who draft by ADP; a league that drafted differently would roster
differently, and R_cont would move. R_real does not have this problem, which is why it is
kept alongside rather than discarded.

### VORP by 10-team draft round under the primary baseline (`sectionO_vorp_contested.csv`)

Best position by round under R_cont: **R1 = TE, R2 = WR, R3–R5 = QB, R6 = WR, R7–R9 = QB,
R10 = TE, R11–R12 = QB, R13 = TE, R14 = WR.** Under R_real the same sequence is
WR/TE/QB/QB/TE/WR/TE/WR/WR/TE/TE/QB/TE/WR. **Cell SEs are 10–50 points on 10 seasons;
the sequence is illustrative and is not a pick-by-pick instruction.** The one stable
statement across baselines is that **QB is the position whose relative value is most
concentrated in the middle rounds (3–9) and lowest at the very top.**

### The owner's TE-cliff claim, tested (`sectionO_te_cliff.csv`)

*Claim: outside roughly four TEs, essentially every TE is a streaming-grade asset.*

**(a) Realized season-total gaps — directionally supported, not decisive.** Mean season
total by realized positional rank, and the drop from rank 4 to rank 5 against the average
per-rank drop over ranks 5→12:

| pos | rank1 | rank4 | rank12 | drop 4→5 | mean step 5→12 | ratio |
|---|---|---|---|---|---|---|
| **TE** | 269.3 | 198.4 | 118.3 | **19.1** | 8.7 | **2.19×** |
| QB | 398.9 | 333.5 | 255.9 | 14.6 | 9.0 | 1.62× |
| RB | 384.2 | 295.0 | 218.5 | 14.4 | 8.9 | 1.62× |
| WR | 368.9 | 299.4 | 244.4 | 9.4 | 6.5 | 1.43× |

TE has the sharpest post-rank-4 step of the four positions on both the absolute and the
relative measure. But the per-season SD of that step is **19.6 against a mean of 19.1**,
so it is not separable from QB's or RB's. **On realized ranks the cliff is real in sign,
weak in evidence.** On *ADP* ranks — the basis a drafter can act on — the TE rank-4→5 gap
is **−2.3** (TE5 outscored TE4 on average): there is no cliff at the price you pay.

**(b) Identifiability — this is where the claim is strongly true, and it is the sharpest
positional result in §O.** P(the ADP-rank-k player finishes top-5 at his position),
contrasting ADP ranks 1–5 against 6–12, season-clustered t(9):

| pos | P(top-5 \| ADP 1–5) − P(top-5 \| ADP 6–12) | se | p | seasons positive |
|---|---|---|---|---|
| **TE** | **+0.451** | 0.060 | **<.0001** | **10/10** |
| WR | +0.289 | 0.090 | .011 | 9/10 |
| QB | +0.191 | 0.085 | .051 | 7/10 |
| RB | +0.091 | 0.090 | .334 | 6/10 |

Per ADP rank at TE: P(top-5 finish) = .80, .40, .80, .50, .40 for ranks 1–5, then
.10, .10, .30, .20, .10, .00, .10 for ranks 6–12. **The market identifies which TEs will
be top-5 far better than it identifies which RBs will be — the top-5 priced TEs deliver a
top-5 TE season 58% of the time against 13% for TE6–12, a gap five times RB's.** The
owner's "roughly four (in 2026: McBride, Bowers, Loveland, Warren)" is a fair reading of
where that group ends; the data put the break at 5 rather than 4, and it is a break in
*identifiability*, not in points.

**(c) The reconciliation, which is the substantive answer.** The top-5 ADP TEs are a
genuinely distinct and reliably identifiable group — **and that is exactly why they carry
no VORP premium under the primary baseline.** High identifiability with no premium is what
a correctly-priced position looks like, and it is the same verdict every edge test in this
project has returned. The elite TEs are real; they are not cheap.

### Chase: the "elite-TE premium" is not elite (`sectionO_chases.csv`)

§O7's nested tiers (TE1–3 +18.1, TE1–5 +18.5, TE1–12 +18.7 at R_real) are nearly identical,
which nested tiers cannot distinguish from a flat profile. Cut into **disjoint** bands:

| baseline | TE1–3 | TE4–6 | TE7–12 | **TE13+** |
|---|---|---|---|---|
| R_exp | +12.3 (p .14) | +8.9 (.43) | +15.3 (.20) | **+31.0 (p<.001, 10/10)** |
| **R_real** | +18.1 (.06) | +14.7 (.20) | +21.1 (.06) | **+36.9 (p<.001, 10/10)** |
| R_week | −10.5 (.22) | −14.1 (.24) | −7.6 (.40) | **+8.2 (.14)** |

Formal decay contrast, band(1–3) − band(7–12): **−3.0 ± 13.6, p = .83** (and −2.3 ± 24.6
at QB). **There is no decay. The premium is largest at the cheapest TEs.** Mechanically
this is exactly what a level shift predicts: R_TE = 147.5 is a much lower bar than
R_RB/R_WR ≈ 170, so *every* TE clears more of his positional bar than the RB/WR at the same
price does — and by the most where the RB/WR at that price is furthest below his own bar,
i.e. in the last rounds. **What this changes for the owner: the finding supports taking a
TE cheaply, not taking one early.** §M's elite-TE framing survives only as arithmetic about
replacement level, not as a claim about elite players; §M's steepness test (+39.0, p = .188,
MDE 76.7) was underpowered and this disjoint decomposition is the sharper read of the same
data. The QB bands are null everywhere except QB4–6 at R_exp (+34.0, p = .07), which does
not survive the move to R_real (+20.1, p = .12).

### Chase: is the 2026 board priced where the historical premium was measured?

**No, at TE — and the direction matters.**

| | historical ADP, median (range) | 2026 | seasons cheaper than 2026 |
|---|---|---|---|
| TE1 | 16.8 (5.9–24.4) | **35.3** | **0/10** |
| TE3 | 41.8 (28.5–51.5) | 60.8 | 0/10 |
| TE5 | 58.4 (50.4–64.3) | 78.5 | 0/10 |
| QB1 | 21.6 (16.9–34.5) | 26.7 | 2/10 |
| QB5 | 61.9 (43.2–69.2) | 67.3 | 1/10 |

**The 2026 market prices elite TE more cheaply than in any of the ten seasons the premium
was estimated on** — TE1 costs pick 35 rather than pick 17. Exploratory sensitivity
(post-hoc; the §O7 headline is not replaced): restricting the historical panel to the ADP
window the 2026 board occupies (TE 28–94) gives a premium of **+14.2 ± 7.3 (p = .08)** at
R_real, attenuated from +18.5 but the same sign — consistent with a level shift that does
not depend on where in the board the TE sits. QB in its window (21–81) gives +7.1 (p = .40),
still null.

### The 2026 forward VORP board (`sectionO_board_2026_vorp.csv`)

Per §M1's pre-specification the forward read uses the **model board**, never realized
outcomes. Values: WR θ*, RB `board_value`, and TE/QB `board_value` = m(ADP), each × the
position's mean games 2015–2024. The board's own replacement is an **R_exp** object by
construction; the R_real and R_week columns add the historical position-specific bracket
gaps (10-team: QB +36.2/+89.9, RB +24.2/+49.9, TE +16.7/+70.5, WR +21.4/+45.9). **That
transfer is a modelling choice and is stated, not hidden.**

2026 replacement, 10-team: R_exp QB 244.7 / RB 138.0 / WR 153.7 / TE 133.1;
R_real (transferred) 280.9 / 162.2 / 175.1 / 149.8.

Top of the 2026 VORP board, 10-team frame, **baseline R_real**: Nacua +98.1, Chase +93.3,
Gibbs / B. Robinson +85.0, St. Brown +80.2, McCaffrey +77.3, Smith-Njigba +69.5,
Rice +63.5, Lamb +62.4, London +55.0, Taylor / Achane / C. Brown +51.6.
**The best TE, McBride, is +28.5 (rank 24); the best QB, Allen, is +17.9.**

TE and QB against the RB/WR going within ±6 picks, 10-team frame, **all baselines**
(the `R_cont` column is the primary; it adds the historical R_cont − R_exp gap to the
board's own R_exp, the same transfer used for R_real and R_week):

| player | ADP | R_exp | **R_cont (primary)** | R_real | *R_week* |
|---|---|---|---|---|---|
| McBride (TE) | 35.3 | +6.1 | **−33.1** | +12.0 | *−16.8* |
| Bowers (TE) | 40.7 | +6.5 | **−32.7** | +12.5 | *−16.3* |
| Loveland (TE) | 60.8 | +13.0 | **−26.2** | +18.9 | *−9.9* |
| Warren (TE) | 65.8 | +15.6 | **−23.9** | +21.5 | *−7.3* |
| Allen (QB) | 26.7 | +11.8 | **−8.0** | −1.5 | *−30.0* |
| Maye (QB) | 49.6 | +8.1 | **−5.4** | −5.8 | *−34.7* |
| Jackson (QB) | 52.8 | +11.7 | **−3.8** | −2.1 | *−30.8* |
| Burrow (QB) | 57.8 | +15.9 | **−0.9** | +2.3 | *−26.4* |

Top of the 2026 board by **VORP_R_cont**, 10-team frame: Gibbs / B. Robinson +162.3,
McCaffrey +154.7, Nacua +147.1, Chase +142.2, St. Brown +129.1, Taylor / Achane /
C. Brown +129.0, Smith-Njigba +118.5. **The hoarding-adjusted baseline promotes RB to the
top of the board**, for the same reason it demotes TE: the free-RB bar is the lowest of any
position (92.8 season points) while the free-TE bar is nearly the starting bar. Under
`R_real` the same board is led by Nacua +98.1 and Chase +93.3 with Gibbs/Robinson at +85.0.
**That RB-vs-WR reordering is a direct consequence of the baseline change and should be
read with §M4(3)'s finding attached — §M's 2-FLEX arbitrage argument used `R_real`, and
under `R_real` WR leads RB at all 14 rounds.** §O does not re-litigate that comparison; it
notes that the corrected baseline moves it, and that the movement is a roster-depth
artifact of exactly the kind §M4(3) warned about.

Note also that the 2026 board reproduces the §O7 chase inside itself under both baselines:
the TE premium is *least bad at the highest ADP* (−33.1 at McBride, −23.9 at Warren under
R_cont; +12.0 → +21.5 under R_real), which is the level-shift signature, not a steepness
signature. Josh Allen's headline +75.9 VORP in §M was an R_exp number; his premium over the
RB/WR at his price is **−8.0 (R_cont)** or **−1.5 (R_real)**.

## Decision, and what §O does not license

1. **Both TE and QB are market-anchored.** Draft them in ADP order. The §O5 honesty clause
   fired twice and no substitute arm was sought.
2. **Against the primary contested baseline `R_cont`, do not pay up for TE.** The premium
   over the RB/WR at the same price is −27.1 ± 5.5 (p < .001, negative in 10 of 10
   seasons), and it is least negative at the cheap end of the TE board (−8.1 for TE13+).
   The mechanism is that opponents hold 1.37 TEs and 5.03 RBs, so the free-TE bar (131.0
   season points) sits close to the starting bar while the free-RB bar (92.8) collapses.
3. **This reverses the §M recommendation, and the reversal is due to a change of baseline,
   not to new data.** Against `R_real` — starting demand, the baseline §M used — the TE
   premium is +18.5 (p = .002). Both numbers are correct; they price different things. A
   reader who intends to carry a second TE, or who expects his league to hoard TEs, should
   use `R_real` and the carry sensitivity. **This is stated rather than resolved because
   the data cannot resolve it: the choice is about what counterfactual the owner faces.**
4. **No elite-QB premium under any baseline that is both contested and foresight-free**
   (R_cont +1.3 p = .93; R_real +5.3 p = .69). Allen at ADP 26.7 is priced at or above
   what either supports.
5. **The elite TEs are real and reliably identifiable — the market knows it.**
   +0.451 in P(top-5 finish) between ADP ranks 1–5 and 6–12, 10/10 seasons. High
   identifiability with zero premium is the signature of correct pricing, and it is
   consistent with all seven prior null edge tests.
6. **Power.** §O7's MDEs are 13–15 points at TE1–5 and 40–45 at QB1–5 on 10 season
   clusters. §O could not have detected a QB premium smaller than about two points a game.
   The TE result under R_cont clears its own MDE; the QB nulls do not bound much.
7. **Not licensed:** any claim of player-level edge at TE or QB (LOSO says none); any use
   of the QB rushing-residual relation as a forecast (it is contemporaneous, not
   preseason-knowable); any conclusion resting on `R_week` (infeasible); and any transfer
   of the §O6/§O7 numbers to a league with different starting requirements or different
   rostering depth — the whole apparatus is a function of D_p and N_p.
8. **Known weakness of the primary.** N_p is endogenous: it was measured from simulated
   opponents drafting by ADP. A league that drafted differently would roster differently
   and R_cont would move. R_real is kept alongside precisely because it does not have this
   problem.

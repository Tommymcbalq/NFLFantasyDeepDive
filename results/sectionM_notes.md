# §M — Positional scarcity and draft sequencing under the owner's actual league

Executed against `EDA_PLAN6.md` §M1–§M5, pre-registered 2026-08-09 before any round-6 fitting.

Scripts: `scripts/sectionM_common.py` (shared board→player join), `28_sectionM_scarcity.py`
(§M1/§M4), `28b_sectionM_diagnostics.py`, `28c_sectionM_replacement_bracket.py`,
`29_sectionM_draftsim.py` (§M2/§M3; `SECTIONM_FRAME=12` for the frame sensitivity),
`30_sectionM_2026_board.py`, `31_sectionM_figures.py`.

Outputs: `vorp_curves.csv`, `replacement_levels.csv`, `sectionM_player_vorp.csv`,
`sectionM_premium_tests.csv`, `sectionM_premium_by_baseline.csv`,
`sectionM_replacement_bracket.csv`, `sectionM_rbwr_cross.csv`, `sectionM_diag_te.csv`,
`sectionM_diag_flex.csv`, `sectionM_adp_value_curve.csv`, `strategy_backtest.csv`,
`strategy_by_slot.csv`, `strategy_distribution.csv`, `sim_calibration.csv`,
`*_12team.csv`, `sectionM_board_2026.csv`, `results/figures/sectionM_*.png`.

---

**Headline: no strategy beats "draft the board". 0 of 5 pre-registered comparisons survive
BH q = 0.10 (smallest raw p = .195), in either league frame. The §M5 honesty clause fires and
the recommendation is S0.**

**The one structural finding §M does produce is not a strategy but a definition: replacement
level is a *bracket*, not a number, and the bracket is 71 points wide at TE and 90 at QB against
46–50 at WR/RB. Every elite-TE and elite-QB claim in the literature is a claim about where in
that bracket you sit, not a claim about players.**

**Second structural finding: in a 2-FLEX(RB/WR) league the flex arbitrages RB and WR replacement
level to a common cutoff — realized R_RB = 170.1 vs R_WR = 169.8 season PPR, a 0.3-point gap. RB
"scarcity" as a draft argument is an artifact of assuming the flex splits evenly between the
positions. It does not: 16.5 of the 20 league-wide flex slots go to WRs.**

---

## §M0 Build, and the join §L did not test

The §L board→player join (`26_sectionL_conversion.py`, carrying its three validated fixes —
skill-position restriction, board-position-governs, ambiguity resolution by activity) was lifted
into `sectionM_common.py` and generalised from {RB, WR} to {QB, RB, WR, TE}.

**Verified for QB and TE, which §L did not test: 1,694 board rows across 2015–2024
(666 WR / 603 RB / 238 QB / 187 TE), 0 unmatched, 0 unresolved ambiguous names, 0 null player
ids.** The join therefore holds at all four positions without modification.

**Scope limitations, stated not hidden.** DST and kickers are out of scope (no kicker in this
league; a fixed DST is assigned identically to every team so it cannot differentiate strategies).
There is no waiver wire, no trades and no in-season acquisition in the simulation — §M5's
"simulation is not evidence about the world" clause applies with particular force here, and §M1's
replacement bracket quantifies exactly how much that omission is worth.

**Cross-era note.** Weeks 1–17 are used in every season: pre-2021 the NFL regular season is 17
weeks with byes, post-2021 it is 18. Weeks 15–17 are therefore the fantasy playoff window in both
eras, but pre-2021 week 17 was the *final* week and starters are rested in it, which depresses the
15–17 window in the early folds. 2020 (COVID) is left in and not adjusted.

---

## §M1 Replacement level, both ways

Demand, 10-team: 10 QB, 10 TE, 20 RB + 20 WR locked, 20 FLEX allocated to RB/WR by realized usage.
Realized usage, pooled 2015–2024: **3.5 RB / 16.5 WR** (12-team: 4.8 / 19.2). So effective demand
is **23.5 RB and 36.5 WR**.

| frame | pos | demand | R static (season) | R static (ppg) | R marginal (per week) |
|---|---|---|---|---|---|
| 10 | QB | 10.0 | 281.0 | 16.53 | 19.69 |
| 10 | RB | 23.5 | 170.1 | 10.01 | 11.52 |
| 10 | WR | 36.5 | 169.8 | 9.99 | 11.43 |
| 10 | TE | 10.0 | 147.5 | 8.67 | 11.84 |
| 12 | QB | 12.0 | 260.8 | 15.34 | 18.33 |
| 12 | RB | 28.8 | 153.7 | 9.04 | 9.80 |
| 12 | WR | 43.2 | 153.4 | 9.03 | 9.79 |
| 12 | TE | 12.0 | 137.4 | 8.08 | 10.63 |

### The replacement bracket — the finding that organises everything else

The two pre-registered definitions differ by more than a rounding. A third — the (D+1)-th best
player by *preseason expectation* (the LOSO isotonic ADP→points curve), i.e. what you can plan to
draft with no in-season management — completes an ordered bracket, since each step adds
information:

    R_exp  ≤  R_real  ≤  R_week
  (draft only)  (season foresight)  (weekly foresight)

10-team demand, season-total PPR, means over 2015–2024:

| pos | R_exp | R_real | R_week | bracket width |
|---|---|---|---|---|
| QB | 244.8 | 281.0 | 334.7 | **89.9** |
| TE | 130.8 | 147.5 | 201.3 | **70.5** |
| RB | 145.9 | 170.1 | 195.8 | 49.9 |
| WR | 148.4 | 169.8 | 194.4 | 46.0 |

**The streaming half of the bracket (R_week − R_real) is +53.7 at QB and +53.9 at TE against
+25.7 / +24.5 at RB/WR — a factor of two.** The mechanism is not subtle: the 10th-best TE *in a
given week* is a much better player than the 10th-best TE *on the season*, because a different TE
occupies that slot every week, and the shallower and more volatile the position the more weekly
churn buys. This is why the two pre-registered VORP measures disagree about TE, and it is the
diagnosed cause of the §M1 anomaly below.

`results/figures/sectionM_replacement_bracket.png`.

### VORP curves by draft slot (10-team frame, realized season totals)

Mean realized VORP by the player's positional rank in preseason ADP (`vorp_curves.csv` carries
both frames, both measures, and both units — positional ADP rank and draft round):

| posrank | QB | RB | WR | TE |
|---|---|---|---|---|
| 1 | 9.5 | 40.0 | 106.3 | **79.7** |
| 2 | 51.2 | 45.1 | 116.7 | 32.4 |
| 3 | −10.1 | 85.8 | 63.4 | 35.0 |
| 4 | 27.0 | 89.2 | 103.8 | 7.9 |
| 5 | −1.4 | 7.5 | 74.1 | 10.2 |
| 6 | −3.2 | 95.5 | 70.5 | −13.0 |
| 8 | −56.0 | 20.5 | 48.9 | −2.2 |
| 12 | −30.6 | 32.8 | 40.9 | −20.1 |

Cell SEs are 14–46 points on n = 10 seasons, so **individual cells carry almost no information**
and are not interpreted. The interpretable object is the shape, which §M4 tests directly.

---

## §M4(1) The elite-TE question — the curve *is* steeper at the top, but not significantly, and
## the sign of the premium depends entirely on the bracket

Two reads, both pre-specified.

**(a) Steepness.** Mean VORP(positional ADP rank 1–3) − mean VORP(rank 4–8), per season, then TE
minus the RB/WR average, season-clustered t(9):

| measure | QB | RB | WR | TE | TE − mean(RB,WR) | SE | p | MDE | seasons + |
|---|---|---|---|---|---|---|---|---|---|
| season totals | 24.7 | −2.7 | 24.9 | **50.0** | **+39.0** | 27.4 | .188 | 76.7 | 8/10 |
| weekly-optimal | 11.0 | 9.6 | 24.5 | 32.5 | +15.5 | 14.4 | .311 | 40.5 | 7/10 |

The TE curve is the steepest at the top on both measures and in 8 of 10 seasons, but the design
cannot distinguish +39 from zero: the MDE is 77 points. **Underpowered, directionally supportive,
not evidence.** *This steepness statistic is exactly frame-invariant* — R_p is a level shift that
cancels inside a within-position difference — so the 10-team and 12-team numbers are identical to
the last digit. Frame changes the *premium*, never the *shape*.

**(b) Premium over the RB/WR you could have taken instead** (mean VORP of RB/WR board rows within
±6 ADP picks), season-clustered t(9), 10-team frame:

| tier | baseline | premium | SE | p | MDE | seasons + |
|---|---|---|---|---|---|---|
| TE1–3 | R_real | **+18.1** | 8.35 | .058 | 23.4 | 8/10 |
| TE1–5 | R_exp (draft-only) | **+12.7** | 5.00 | .032 | 14.0 | 7/10 |
| TE1–5 | R_real (season foresight) | **+18.5** | 4.31 | **.002** | 12.1 | 9/10 |
| TE1–5 | R_week (weekly foresight) | **−10.2** | 4.96 | .073 | 13.9 | 3/10 |
| TE1–12 | R_real | +18.7 | 4.44 | .002 | 12.5 | 9/10 |

**Verdict on the owner's elite-TE hypothesis.** Against a replacement you must *draft* and hold,
an elite TE is worth roughly **+13 to +19 season points** over the RB/WR going at the same ADP —
about 1 point per game, real but small, and positive in 8–9 of 10 seasons. Against a replacement
you can *stream with weekly foresight*, the premium is **−10**. Both numbers are correct; they
answer different questions. A 10-team league with 14 roster spots and a live waiver wire sits
between them, closer to the first. **The premium is real, it is one point a game, and it is
entirely contingent on not streaming the position.**

## §M4(2) The elite-QB question — no premium, in either frame, under any baseline

| tier | baseline | premium | SE | p | MDE | seasons + |
|---|---|---|---|---|---|---|
| QB1–3 | R_real | −4.5 | 20.3 | .830 | 57.0 | 7/10 |
| QB1–5 | R_exp | +19.0 | 13.4 | .190 | 37.5 | 6/10 |
| QB1–5 | R_real | +5.3 | 12.8 | .694 | 35.9 | 6/10 |
| QB1–5 | R_week | −23.4 | 13.0 | .105 | 36.4 | 4/10 |
| QB1–12 | R_real | +2.8 | 8.6 | .756 | 24.1 | 5/10 |

The QB VORP curve is **flat and near zero across the whole top of the board** (+9.5, +51.2, −10.1,
+27.0, −1.4, −3.2 at positional ranks 1–6), the steepness test returns +13.6 ± 27.1 (p = .63), and
the largest MDE in the project's history (57 points at QB1–3) sits on top of it. **No elite-QB
premium is detectable, and the point estimate is negative at the very top.** Mechanism: with 10
starting QBs and ~30 rosterable ones, R_QB = 281 season points is the highest replacement level of
any position, and elite QB scoring is not far enough above it to pay for the draft cost.

## §M4(3) Where the RB/WR curves cross — they do not, and the reason is the flex

RB − WR mean VORP by 10-team draft round (season totals, season-clustered SE):

| round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RB − WR | −19.4 | −36.0 | −27.7 | −28.3 | −33.9 | −32.1 | −26.1 | −13.2 | −41.8 | −16.0 | −24.6 | −55.4 |
| SE | 25.6 | 21.6 | 21.3 | 10.8 | 22.1 | 23.7 | 11.2 | 15.5 | 15.3 | 23.6 | 14.5 | 18.2 |

**WR is ahead at every one of the 14 rounds, in both frames, on both VORP measures. The curves
never cross, so there is no draft cost at which the scarcity argument for RB starts paying.**
Individual rounds are not corrected for multiplicity and are `family = 0`; the statement that
survives is the sign pattern, 14 of 14.

**The mechanism, chased to the arithmetic.** With the flex allocated as it is actually used,
R_RB = 170.1 and R_WR = 169.8 — a 0.3-point gap. That is not a coincidence: **a FLEX slot open to
both positions is an arbitrage, and it forces the two replacement levels to a common cutoff.** Once
they are equal, the RB-vs-WR VORP comparison collapses to a raw-points comparison, and §D7 already
established that a WR banks more raw PPR than an RB at every matched positional rank (+11.7 at
tier 1 rising to +42.3 at tier 4). Force the flex 50/50 by fiat and everything reverses:

| flex allocation | R_RB | R_WR | gap | RB − WR VORP, round 1 |
|---|---|---|---|---|
| realized (3.5 / 16.5) | 170.1 | 169.8 | **+0.3** | **−19.9** |
| 50/50 by fiat (10 / 10) | 150.2 | 186.9 | −36.7 | **+16.0** |
| no flex at all (0 / 0) | 185.5 | 215.1 | −29.6 | +5.6 |

**So the entire "elite RB scarcity premium" is a statement about who fills the flex.** A
robustness check that this is not an artifact of counting undrafted breakouts: restricting the
pool to the 140 players a 10-team, 14-round draft actually rosters gives flex 5.4 RB / 14.6 WR and
R_RB 150.3 vs R_WR 150.8 — still equalised, still WR-dominated.

This is the calculation §L explicitly deferred to round 6. §L found conversion positionally flat
and said any RB preference must come from scarcity weighting. **§M measures the scarcity weighting
and finds it is approximately zero in a 2-FLEX league.**

## §M4(4) Is any of this a 12-team artifact? No — and the elite-TE premium runs the other way

| quantity | 10-team | 12-team |
|---|---|---|
| TE1–5 premium (R_real) | **+18.5** (p = .002) | **+12.4** (p = .029) |
| QB1–5 premium (R_real) | +5.3 (p = .694) | +9.1 (p = .541) |
| TE steepness vs RB/WR | +39.0 (p = .188) | +39.0 — identical by construction |
| RB − WR crossing round | none (WR ahead 14/14) | none (WR ahead 13/14; round 7 is +0.1) |
| strategy backtest | 0 of 5 survive BH | 0 of 5 survive BH |

**The elite-TE premium is *larger* in the owner's 10-team league than in the 12-team frame the ADP
comes from, not smaller.** Mechanism: cutting demand from 12 to 10 raises R_TE by 10 points but
raises R_RB/R_WR by 16–17, because the RB/WR replacement sits 29–43 players deep on a curve that is
still falling there while the TE replacement sits 10–12 deep on one that has already flattened.
**A 10-team owner reading 12-team ADP is therefore under-, not over-paying for the elite TE
relative to what his league structure justifies.** The QB conclusion is null in both frames. The
frame gap changes no conclusion in §M; it changes the size of the TE number by ~6 points a season.

---

## §M2/§M3 The draft-simulation backtest

**Design, fixed before any strategy was evaluated** (full statement in the script docstring):
10 teams, snake, 14 rounds of QB/RB/WR/TE = 140 picks. Each of the 9 opponents draws a private
board `v_{t,i} = ADP_i + ε_{t,i}`, `ε ~ N(0, stdev_i)` with `stdev_i` the FFC across-draft SD, and
takes the best available player subject to positional need (cap 2 QB / 2 TE, no second QB/TE while
a mandatory slot is unmet, hard force-fill when picks remaining equal mandatory slots remaining).
N = 200 drafts per (season, slot) × 10 slots × 10 seasons, **common random numbers across
strategies**. Scoring is the weekly optimal 1QB/2RB/2WR/1TE/2FLEX lineup on realized weekly PPR;
a player with no game that week cannot be started, so bench depth pays automatically.

**Opponent-model calibration (checked, reported, not tuned).** Simulated SD of a player's draft
slot averages **4.95 against the FFC stdev of 6.92** (ratio 0.79), with **corr = 0.951** across
players, and the mean absolute displacement of a player's simulated draft slot from his ADP is
**2.42 picks**. So the model reproduces ~79% of observed dispersion and its cross-player pattern
almost exactly, and mapping a 12-team ADP ordering onto a 10-team draft displaces players by only
about 2.4 picks on average. The 21% under-dispersion is a known consequence of the roster-need
constraints truncating the tails; it was left as pre-specified.

**Sanity level:** our S0 team (the exact consensus board, no noise) averages 1735.6 points weeks
1–14 against 1700.3 for the nine noisy opponents. Drafting the consensus board beats drafting a
noisy version of it by ~35 points a season. That is the size of the prize this whole exercise is
competing against.

### The declared family — 5 tests, `pts14`, BH q = 0.10

New declared family. **{§H5, §I3}, {§K}, {§L} and {§L-EXT} were not reopened or re-corrected.**

| rank | strategy | mean Δ vs S0 (wk 1–14) | SE | 95% CI | raw p | BH thresh | reject | MDE | seasons + | slots + |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | S5 VORP-greedy | **−24.5** | 17.5 | [−64.1, +15.1] | .195 | .020 | no | 49.1 | 3/10 | 2/10 |
| 2 | S2 RB-first | **−12.5** | 10.3 | [−35.8, +10.9] | .258 | .040 | no | 28.9 | 4/10 | 1/10 |
| 3 | S3 zero-RB | **−11.5** | 12.0 | [−38.6, +15.6] | .361 | .060 | no | 33.6 | 3/10 | 1/10 |
| 4 | S4 elite-TE | **+3.3** | 6.7 | [−11.9, +18.5] | .636 | .080 | no | 18.8 | 4/10 | **9/10** |
| 5 | S1 model board | **−5.7** | 12.2 | [−33.2, +21.8] | .649 | .100 | no | 34.1 | 6/10 | 3/10 |

**0 of 5 survive.** Secondary windows (`family = 0`): weeks 15–17 gives the same ordering with S5
at −12.7 (p = .022, BH threshold .020 — still not rejected) and S4 at −8.0 (p = .102); weeks 1–17
gives S5 −37.2 (p = .086), everything else p ≥ .33.

**12-team frame sensitivity** (`*_12team.csv`, 12 teams × 12 rounds — the FFC board cannot fill
12 × 14 picks in 2022): identical conclusion, **0 of 5**, smallest raw p = .049 (S5) against a BH
threshold of .020, and S4 is again the only positive mean (+3.7, p = .724).

### The full distribution, and where points and win probability disagree

Weeks 1–14, pooled over 200 × 10 × 10 = 20,000 drafts per strategy:

| strategy | mean | SD | p05 | p25 | median | p75 | p95 | mean rank | P(top-4) | P(1st) |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 draft the board | 1735.6 | 162.7 | 1468 | 1628 | 1735 | 1842 | 2002 | 4.97 | .476 | .133 |
| S1 model board | 1729.9 | **153.9** | 1475 | 1626 | 1732 | 1833 | 1982 | 5.04 | .460 | .115 |
| S2 RB-first | 1723.2 | **164.4** | 1452 | 1614 | 1722 | 1831 | 1993 | 5.17 | .445 | .120 |
| S3 zero-RB | 1724.1 | 158.3 | 1464 | 1618 | 1725 | 1830 | 1985 | 5.12 | .448 | .118 |
| S4 elite-TE | **1738.9** | 166.1 | 1470 | 1628 | 1737 | 1848 | **2018** | **4.93** | **.482** | **.138** |
| S5 VORP-greedy | 1711.1 | 161.7 | 1436 | 1607 | 1717 | 1821 | 1967 | 5.33 | .421 | .108 |

**The spread inside a strategy is 10 times the spread between strategies.** The interquartile
range of a single strategy is ~215 points; the largest gap between any two strategy means is 28.

**Where the two objectives diverge, as §M5 required.** Mean points and P(top-4) rank the six
identically. **P(finishing 1st) does not:** S1 is 3rd on mean points and 3rd on playoff odds but
**5th on P(win)**, and S2 is 5th on mean points but **3rd on P(win)**. The reason is variance, and
it is visible in the table: S1 has the *lowest* SD of any strategy (153.9) and S2 the *highest*
(164.4). Shrinking a board toward a posterior mean — which is exactly what our model does —
compresses the outcome distribution, and a compressed distribution wins fewer 10-team leagues even
when its mean is unchanged. **In a winner-take-most league, our own model's variance reduction is
a cost, not a benefit.** None of these differences is significant (DM on P(1st): S1 −0.018,
p = .232; S2 −0.013, p = .343), so this is a *direction*, recorded because the plan required the
comparison, not a finding.

### What each strategy actually did — mean roster composition

| strategy | QB | RB | WR | TE |
|---|---|---|---|---|
| S0 | 1.75 | 5.03 | 5.85 | 1.37 |
| S1 | 1.64 | **3.93** | **7.17** | 1.27 |
| S2 | 1.75 | 5.88 | 5.00 | 1.37 |
| S3 | 1.70 | 3.46 | 7.54 | 1.30 |
| S4 | 1.89 | 4.96 | 5.67 | **1.49** |
| S5 | 1.79 | 3.37 | 7.48 | 1.36 |

**S1's composition shift is an emergent property of the permutation and is worth flagging.**
S1 differs from S0 only by re-ordering the modelled WRs among the ADP slots they already occupy
— a permutation that is position-neutral by construction. Yet S1 ends with 1.1 fewer RBs. The
reason is availability asymmetry: a WR the model *promotes* jumps ahead of RBs and we take him;
a WR the model *demotes* falls behind those RBs and is gone by the time we come back. A
mean-preserving re-ranking is not roster-composition-preserving in a sequential draft with
competitors. Nothing is wrong with the code; this is a real feature of sequencing.

### The two anomalies chased to mechanisms

**(i) The elite-TE premium changes sign between the two pre-registered VORP measures.**
Decomposed 2×2 over {static R, marginal R} × {linear, positive-part}, TE1–5 premium, 10-team:

| cell | premium | SE | p |
|---|---|---|---|
| linear × static R | **+19.3** | 4.25 | .001 |
| positive-part × static R | +7.9 | 2.17 | .005 |
| linear × marginal R | **−9.4** | 4.84 | .084 |
| positive-part × marginal R | −5.5 | 2.68 | .072 |

**The flip is the replacement baseline, not the positive part**: holding the truncation fixed,
switching R moves the estimate by −28.7 (linear) and −13.4 (positive part); holding R fixed, the
truncation moves it by −11.4 and +3.9. And the magnitude is predicted exactly by the bracket: TE's
streaming gain is +3.17 pts/week against RB/WR's +1.47, so the extra baseline charged to TE is
(3.17 − 1.47) × 17 = **+28.9 points**, against an observed shift of 28.7. Fully accounted for.

**(ii) S5 as first implemented lost by 120 points, and that was a bug.** The pre-registered rule
is "maximise marginal VORP over current roster need". The first implementation's lineup valuer
placed a drafted player into a starting slot unconditionally, so drafting anyone below replacement
scored a *negative* marginal value. Because the ADP→expected-points curve puts WRs above RBs at
every ADP while the flex has already equalised their replacement levels, the least-negative pick
was always a WR: S5 produced **1.03 QB / 1.96 RB / 10.01 WR / 1.00 TE** rosters, taking its two
mandatory RBs in rounds 13–14, and lost 120.4 points (p < .001). A starting slot is worth
`max(assigned player, replacement)` — a freely available replacement-level body can always be
started instead — and with that corrected S5 loses **24.5 points (p = .195)** with sane rosters.
**This is recorded as an implementation defect fixed after seeing the result, in the same class as
§L's three caught defects and §28.3's withdrawn false positive. It is a bug fix, not a strategy
change: the family remains the five pre-registered strategies and no strategy was added.** The
buggy run's numbers are reported above so the correction can be audited.

**(iii) Why the corrected S5 still loses — one hypothesis tested and rejected, one supported.**

*Rejected: bench depth.* The natural story is that a greedy on expected season points assigns zero
marginal value to depth (in expectation nobody is injured), while the §M3 weekly scorer pays for
it. **The diagnostic scorer refutes this.** S5 loses **−35.8** points on ex-post-best-8 season
totals against **−24.5** on the weekly lineup, i.e. weekly scoring *helps* S5 relative to a static
one. Depth is not the mechanism.

*Rejected: the R_exp/R_real scale mismatch.* S5 maximises an objective in *expected* points against
a replacement estimated as a *realized* order statistic, which §M1's bracket shows is 16–36 points
too high and by different amounts per position. Rerunning S5 with the expected-board replacement
(`SECTIONM_S5REP=exp`, `family = 0` diagnostic, `*_s5exp.csv`) puts objective and baseline on one
scale: the result is **−31.3** (p = .189), slightly *worse*, with roster composition essentially
unchanged (3.37 RB / 7.39 WR). **Hypothesis rejected.**

*Supported: roster value is concave in positional count, so per-player VORP superiority does not
license buying more bodies.* Only 2 WR + up to 2 FLEX start, so the 6th and 7th WR add almost
nothing. Within S0 alone, demeaning by (season, slot), mean points by number of RBs drafted:

| RBs drafted | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| mean points vs (season, slot) mean | −41.3 | +10.2 | **+14.4** | +13.1 | −0.5 | −14.2 | −43.9 |

The optimum is 4–5 RB and the falloff is steep and symmetric. **S5 (3.37 RB) and S3 (3.46) sit on
the WR-tilted side, S2 (5.88) on the RB-tilted side, and all three lose; S0's 5.03 sits at the
peak.** The within-S0 curve is observational and `nRB` is endogenous there, but the strategies
supply the experimental variation — they are assigned against identical opponent draws — and they
land where the curve predicts.

**This reconciles §M4(3) with §M2, which otherwise look contradictory.** WR VORP exceeds RB VORP at
every round, yet no WR-tilted strategy wins. There is no contradiction: VORP-by-round compares the
*average RB* to the *average WR* taken at that cost, which is a statement about which player to
prefer *at the margin* — not a statement about how many bodies to own. Roster value is concave in
count, and **the market's positional mix is already at the peak of that curve.**

### Across-slot spread — the usability screen §M5 demanded

Mean Δ vs S0 by our draft slot, weeks 1–14:

| strategy | min | max | range | mean | slots positive | P(top-4) range |
|---|---|---|---|---|---|---|
| S1 model board | −31.1 | +11.3 | 42.4 | −5.7 | 3/10 | .124 |
| S2 RB-first | −27.9 | +17.9 | 45.8 | −12.5 | 1/10 | .141 |
| S3 zero-RB | −28.8 | +22.4 | 51.2 | −11.5 | 1/10 | .140 |
| S4 elite-TE | **−7.2** | +11.2 | **18.4** | +3.3 | **9/10** | **.054** |
| S5 VORP-greedy | −48.8 | +33.8 | **82.6** | −24.5 | 2/10 | .219 |

**S2 and S3 are mirror images that both lose**: RB-first is worth +17.9 from pick 1 and −26.0 from
pick 4; zero-RB is +22.4 from pick 1 and −28.8 from pick 4. Neither is usable as a rule because
neither works from more than one seat. **S5's 82.6-point range makes it the least usable strategy
in the set even after the bug fix.** S4 is the only strategy whose across-slot range (18.4) is
smaller than a single strategy's inter-quartile noise, and the only one positive at more than 3 of
10 slots — but its mean gain is 3.3 points against an MDE of 18.8, so this is a statement about
*consistency*, not about detectable gain.

**Why S4 is the only non-negative strategy, and why that still is not a recommendation.** S4's
per-season gain correlates **+0.674** with the mean realized VORP of that season's top-3 TEs by
ADP (2018: +48.3 when TE1–3 averaged +104.2 VORP; 2016: −10.5 when they averaged +5.3). So the
elite-TE strategy pays exactly in the seasons when the elite TEs are good — which is not
forecastable ex ante, and which §L established the market prices correctly on average. S4 is a
variance-increasing bet on a positionally steep curve, not an edge.

---

## §M5 Decision, and the honesty clauses honoured explicitly

1. **No strategy beats "draft the board". The recommendation is S0.** 0 of 5 comparisons survive
   BH q = 0.10 in the 10-team frame, 0 of 5 in the 12-team frame, and 0 of 5 on either of the two
   secondary scoring windows. Four of five point estimates are *negative*. This was the plan's
   stated likely outcome and it is what happened.
2. **The simulation is evidence about our board interacting with a modelled opponent, not about
   the world.** What it licenses: statements about how a fixed sequencing rule performs against
   nine ADP-with-noise drafters under this league's roster constraints, with no waivers, trades,
   in-season management, DST or kicker. What it does not license: any claim that RB-first or
   zero-RB "does not work" against *human* opponents, who deviate from ADP in correlated and
   strategic ways this model does not contain; or any claim about the value of positions once a
   waiver wire exists — §M1's bracket shows that assumption alone is worth 71 points a season at TE
   and 90 at QB, three to four times the largest strategy difference measured.
3. **The across-slot spread is reported and it disqualifies the two strategies with the largest
   mean effects.** S2 and S3 each win from pick 1 and lose from the middle; S5 has an 82.6-point
   range. Only S4 is slot-robust, and its mean gain is a quarter of its MDE.
4. **The full distribution is reported and the two objectives are compared directly.** Mean points
   and playoff probability agree on all six strategies. Mean points and P(finishing 1st) do not:
   S1 is 3rd and 5th, S2 is 5th and 3rd, and the reason is that S1 has the lowest outcome variance
   and S2 the highest. **In a 10-team league the objective is winning, and a lower-variance board
   is a worse instrument for winning at the same mean.** No difference is significant.

---

## The 2026 forward read (`sectionM_board_2026.csv`)

Per §M1's pre-specification the 2026 recommendation uses the **model board**, never realized
outcomes: base value `E[season total | position, ADP]` from the 2015–2024 isotonic curve, with the
modelled universes overridden by WR θ* and RB `board_value` (which is m(ADP) — REPORT §23 adopted
no RB arm), and replacement taken at the 10-team demand with the realized flex split.

Replacement, 10-team: **QB 242.1 / RB 138.0 / WR 153.7 / TE 129.6** season points. Note these sit
at the *draft-only* (R_exp) end of the bracket by construction, because a board of expectations
has no order-statistic selection in it — so **the QB and TE premiums below are the upper end of
the range §M1 brackets, and should be discounted toward the R_real column before acting.**

Top of the 2026 VORP board, 10-team frame: Nacua (+119.5), Chase (+114.6), Gibbs / B. Robinson
(+109.2), St. Brown (+101.6), McCaffrey (+101.5), Smith-Njigba (+90.9), Rice (+84.9), Lamb (+83.8),
London (+76.4), **Josh Allen (+75.9 at ADP 26.7)**, Taylor / Achane / C. Brown (+75.8).
McBride and Bowers are the top TEs at +56.7, a **+17.6 and +25.7** premium over the RB/WR going
within ±6 picks of them.

**Read with the §M4 verdicts attached**, and this is the practical translation:
- the TE premium is the one real positional effect §M found, it is worth about a point a game, it
  is *larger* in the owner's 10-team league than the 12-team ADP implies, and it is contingent on
  not streaming the position;
- the QB premium on this board is the R_exp end of a 90-point bracket and the historical test
  against R_real is +5.3 ± 12.8 — **treat the Allen number as an upper bound, not a signal**;
- there is no RB scarcity argument in this league at any draft cost, because the flex arbitrages
  it away;
- and none of it beat drafting the board in simulation.

---

## Unexplained / open

- **Four anomalies arose; three are fully explained, one is explained with a caveat.** The TE
  premium sign flip (replacement baseline; predicted magnitude 28.9 vs observed 28.7), S5's
  120-point loss (a lineup-valuer defect), and S1's roster-composition drift (availability
  asymmetry under a mean-preserving re-ranking) are closed. **The corrected S5's residual −24.5 is
  attributed to roster-mix concavity, and that attribution rests partly on an observational
  within-S0 curve where the roster mix is endogenous.** Two rival explanations (bench depth, the
  R_exp/R_real scale mismatch) were tested and rejected by re-running the simulation, so the
  attribution is not merely the last story standing — but it is a −24.5 ± 17.5 effect (p = .195),
  and a quantity that is not significantly different from zero does not strictly require an
  explanation at all. **Recorded as diagnosed-but-not-proven.**
- **Two things §M deliberately does not answer.** (1) The value of *in-season* management. The
  replacement bracket says this is the largest single lever measured anywhere in the project —
  larger at QB and TE than every strategy difference combined — and the simulation contains none of
  it. That is the obvious next pre-registration. (2) Whether human opponents' deviations from ADP
  are exploitable; the noise model is deliberately unstrategic, and a strategic opponent model
  would need to be pre-registered before it could be believed.
- **One power statement, computed for this design and not transplanted** (per §28's lesson). The
  §M2 MDEs range from 18.8 (S4) to 49.1 (S5) points per season on 10 season clusters, against a
  within-strategy SD of ~163. §M2 could not have detected a real sequencing edge smaller than
  about **1.3 points per game**, and the sampling noise in a single season is ~10 times that. A
  null here bounds the size of any sequencing edge; it does not prove one is absent.

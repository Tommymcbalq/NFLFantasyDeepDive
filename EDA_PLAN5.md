# EDA Round 5 — Positional conversion rates by draft cost
### Pre-registered 2026-08-09, before any round-5 fitting. Rules unchanged: no tuning toward
expected results, anomalies chased, adoption only on pre-specified evidence, and any claim of a
exploitable pattern needs FDR control **and** a temporal holdout.

## Motivation

Every test through round 4b asked the same question — *does some preseason variable predict the
market's error for a given player?* — and returned null six times (§6.2, §B3, §E, §F2, §I3, §K).
§L asks a structurally different question.

A drafter does not buy a player in isolation; he buys **a slot at a position**. Even if ADP is
individually unbiased, the *distribution* of outcomes conditional on (position, draft cost) may
differ in ways a drafter can act on — because the actionable choice at pick 14 is "RB or WR", not
"this player or his true value". ADP is formed player by player; a tier-level pattern is not
something an individual price is under pressure to correct.

The project owner's stated hypothesis, recorded verbatim before testing so it can be scored:

> "Top-tier — the top ten running backs — achieving and justifying their ADP more in recent
> years. It's been trending up in maybe the last two to three years, more than wide receivers.
> But wide receivers at the top level are almost a coin flip … I'm pretty sure you're better off
> just stacking running backs in the first round and then taking wide receivers rounds five
> through eight, or four through eight."

Three separable claims: (a) elite-RB conversion exceeds elite-WR conversion; (b) elite-RB
conversion has been **trending up** recently; (c) mid-round WRs convert well enough to be the
better buy there. Each is tested separately. The owner has said he is confident (a)–(c) are true
from observation; that is precisely why they get a pre-registered test rather than a replication.

## §L0 Data and construction (no modelling)

- **Boards:** `data/adp/historical/adp_ppr_20{15..24}.csv` — full FFC PPR 12-team boards, all
  positions (~200 players/year: ~60–68 WR, ~53–65 RB, ~20–23 QB, ~20 TE).
- **Outcomes:** season totals and PPG for **every** player from `data/players/weekly_raw/`
  (regular season only), so positional finish ranks are computed against the whole league, not
  against the board. A player drafted at WR20 who finishes WR8 must be able to displace an
  undrafted breakout.
- **Two outcome definitions, both reported** — they answer different questions and the gap between
  them *is* a finding:
  - **total PPR points** — what a drafter actually accrues; includes missed games. This is the
    primary definition, because availability is part of what a draft pick buys.
  - **PPG given participation** (≥ 4 games, per the project's standing floor) — isolates per-game
    production. §A established availability is a stable trait, so the total-vs-PPG gap is
    interpretable rather than noise.
- **Finish tiers**, fixed now per the owner's specification: WR1 = positional finish 1–12,
  WR2 = 13–24, WR3 = 25–36; identically for RB. QB/TE computed for the round-6 strategy work but
  not tested here.
- **Cost bins**, fixed now: ADP rounds in the **12-team** frame of the ADP source
  (round = ceil(ADP/12)), grouped as R1–2, R3–4, R5–6, R7–8, R9+. **The owner's league is
  10-team**; the same table is reported in a 10-team frame (round = ceil(ADP/10)) and any
  strategy statement must name which frame it is in. This gap is stated, never silently elided.

## §L1 Conversion rates (descriptive, with honest uncertainty)

For each (position, cost bin, season): P(finish in tier T | drafted in bin b). Report with
**Wilson intervals**, and report the cell n on every number. Pooled 2015–2024 and, separately,
the owner's stated 7-year window 2018–2024.

Pre-specified caution: a cost bin holds ~12–24 players per position per season. A single season's
rate has a standard error of roughly 10–14 points. **Single-season rates will not be interpreted
as signal**, and no claim rests on one.

## §L2 Claim (a) — does elite RB convert better than elite WR?

Test P(hit | RB, R1–2) vs P(hit | WR, R1–2), where "hit" is pre-defined two ways and both are
reported: (i) finishing in the top-12 of the position, (ii) returning at least the points of the
median player drafted in the same bin regardless of position (a value-return definition that does
not privilege positional scarcity). Two-proportion test with season-clustered inference
(10 clusters, t with 9 df), plus a logistic regression with season random effects as a robustness
check.

## §L3 Claim (b) — is elite-RB conversion trending up?

**This is the claim most exposed to reading noise, and is pre-registered accordingly.** Logistic
regression of hit on season (linear in year), RB R1–2 cell only, cluster-robust by season;
report the slope, its CI, **and the pre-test MDE**. With ~10 seasons × ~15 players the design is
weak, and §K's lesson applies: report the MDE next to the p-value and label an underpowered null
as uninformative rather than as evidence of absence.

Explicitly pre-specified: no data-driven choice of breakpoint. "The last two to three years" is
**not** a hypothesis a 10-point series can test without the breakpoint being chosen post hoc, so
the linear-trend test is primary; a fixed 2022–2024 vs 2015–2021 split is reported as secondary
with its multiplicity counted, and no other split is examined.

## §L4 Claim (c) — where is the best place to buy WRs?

Conversion by cost bin for WR, and the same for RB, tested for a **position × bin interaction**
(the actionable quantity: does the WR-minus-RB conversion gap change with draft cost?). This is
the statistically correct form of "stack RBs early, take WRs in rounds 4–8" — the claim is about
an interaction, not two separate main effects.

## §L5 Screens and decision rule

New FDR family, declared now, comprising every test in §L2–§L4 at **BH q = 0.10**, and a
**temporal holdout**: fit on 2015–2021, check the pattern holds on 2022–2024. Both binding, as
always. The {H5, I3} and {K} families remain closed and are not reopened.

**What adoption would even mean here.** A surviving pattern does *not* enter θ* — it is not a
player-level valuation term. It would enter the round-6 draft-strategy work as an empirically
grounded positional prior, and would be reported with its effect size in points, not as a rule of
thumb. A null means the market's tier-level structure is as efficient as its player-level
structure, which after six null edge tests is the anticipated result.

## §L6 Recorded confounds, before results

- **Survivorship in finish ranks:** a player who misses the season still occupies a draft slot but
  cannot finish top-12. This is *correctly* counted as a miss under the total-points definition
  and *incorrectly* excluded under PPG-given-participation. Hence both definitions.
- **Positional scarcity is not conversion.** RB and WR finish tiers are not equally valuable in a
  2-flex league; a 40% RB hit rate and a 40% WR hit rate are not equally useful. §L measures
  conversion only; the value weighting belongs to round 6, under the owner's actual settings.
- **Era effects in the RB market:** the 2015–2024 window spans a zero-RB fashion cycle. A trend in
  conversion may reflect the *market* changing (RBs being drafted more cheaply, hence converting
  more easily) rather than RBs improving. §L3 must therefore also report the trend in RB draft
  *cost* over the same window, or a rising hit rate cannot be interpreted.

## Outputs

`results/sectionL_notes.md`, `conversion_rates.csv` (position × bin × season × outcome definition,
with n and Wilson bounds), `conversion_tests.csv`, figures, rerunnable script.

## Carried from round 4, still open (not addressed by §L)

1. Eq. (7)'s V has no term for how far next season's level moves from μ̂: ≈ 0 for WR, ≈ 6 PPG² for
   RB, so V understates predictive variance by ~40% and B is too small on the RB side.
2. RB σ²_T negative (backfield constraint; only 6 two-board-RB team-seasons).
3. Sophomore RB excess volatility survives the level control where the WR analogue did not.
4. 2026 win totals stored without under-prices, so not de-viggable on the same footing as history.
5. Script numbering collided across parallel researchers in round 4; renumber.

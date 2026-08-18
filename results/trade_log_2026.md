# Trade log — 2026 draft

Operational record of every trade evaluated, with the numbers that drove the call.
Valued by `scripts/47_trade_eval.py` on two scales: **generic** (the isotonic market curve at each
slot — what the counterparty sees, and the owner's primary lens) and **yours** (same board after the
28 logged views in `views_2026.csv`). Picks are round.pick in a 10-team snake, overall = (r−1)·10+p.

FAAB is **not modelled** — no in-season waiver valuation exists in this project. Where a deal
includes FAAB it is flagged and excluded from the net, so every net below understates the deal.

---

## T1 — SharksGeneralManager, 1.07 package (2026-08-09) — REJECTED by counterparty

| | picks |
|---|---|
| get | 1.07, 3.07, 5.07, 6.04, 7.07, 10.04 + $25 FAAB |
| give | 2.06, 3.05, 4.06, 5.05, 7.05, 9.05 |

**Net generic +3.57 · yours +3.76.** Structure: rounds 3/5/7 are near-identical swaps two slots
later; the real trade is **2.06 + 4.06 + 9.05 → 1.07 + 6.04 + 10.04** — giving up round 2 and round 4
to buy a second first-rounder. Correct direction in a 10-team league where only 7 starters score:
concentrating value at the top beats spreading it.

Availability at the resulting pick set (20k sims, sharp-league noise ×0.45): **CMC 82% at 5,
Amon-Ra 72% at 7** — roughly a 60% joint outcome for both.

**The cost, and it is real:** the round-4 pick disappears and the gap runs 7 → 27. Davante Adams
(ADP 41.1, our WR12, the board's single biggest edge at +18 vs ADP) goes from **96% available at 36
to 18% at 47**. Higgins, McMillan, Nabers and Egbuka die in the same gap. The trade pays a whole
ADP-30-to-45 tier for a second first.

Rejected by the counterparty. Recorded because first offers rarely clear and the structure is the
template for the next one.

---

## T2 — SharksGeneralManager, pick-swap (2026-08-14) — three versions

Same shape throughout: swap five picks two slots each way, plus $25 FAAB.

| version | change from v1 | generic | yours |
|---|---|---|---|
| v1 | 5 picks each way | **+1.67** | **+2.23** |
| v2 | added 9.07 + 12.04 for 10.06 + 11.05 | +1.01 | +1.57 |
| **v3 (final)** | dropped 12.04 and 11.05 | **+1.67** | **+2.23** |

**v2 was the worse deal despite moving more picks.** The added leg received 0.76 + (−0.13) and sent
0.76 + 0.52 — **−0.65**. Pick 12.04 (overall 114) is past the point where the board values a pick
*below replacement*; it read as generosity and was a cost. v3 removed it, leaving 9.07-for-10.06 at
0.76 vs 0.76, an exact wash.

**Where the +1.67 actually comes from — one place, not five.** Picks 54 and 56 straddle an isotonic
cliff: 2.99 vs 1.72, **1.27 points across two slots**. At 54 you reach the DJ Moore (52.6) / Lamar
(52.7) / Burden (52.9) cluster; at 56 they are gone and it is Pierce and Henderson. Every other swap
in the deal is worth 0.1–0.5. Volume in a trade is not value.

**Tail analysis — the reason to take it** (6,000 sharp-league sims, best-available VORP by slot):

| swap | RB median | **RB p10** |
|---|---|---|
| R3: give 25 → get 27 | 0.00 | **−0.42** |
| R4: give 36 → get 34 | 0.00 | **+0.50** |
| R5: give 45 → get 47 | −0.18 | 0.00 |
| R6: give 56 → get 54 | 0.00 | **+0.25** |
| R8: give 76 → get 74 | 0.00 | 0.00 |

**Medians barely move; the entire effect is in the tail.** Two slots at the median gets a
near-identical player — two slots in the bad 10% of drafts is where a run empties a tier and you
fall past its edge. Net downside improves ~+0.33, better at three of five picks. WR tails are flat
at every swap (0.00): receivers are deep enough here that two picks never crosses a tier edge.

**Verdict: accept v3.** Plus $25 FAAB, which is almost certainly worth more than the 1.27-point
round-6 cliff and which we cannot price.

---

## Standing method notes

- **Evaluate on generic first.** If a trade only works on our board, it is a bet on 28 views, not on
  price. T2 is +1.67 generic, so it does not depend on being right about anything.
- **Read the tail, not the median.** Slot changes of 1–3 picks almost never move the median and
  routinely move p10 by 0.3–0.5. That is the whole content of a small pick swap.
- **Count the legs.** Added picks late in a deal are where value quietly leaves; anything past ~ADP
  110 is at or below replacement on this board and is not compensation.
- **FAAB is unmodelled and excluded.** Every net here understates a deal that includes it.

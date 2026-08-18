# Trade log — 2026 draft

Operational record. Valued by `scripts/47_trade_eval.py` on two scales: **generic** (the isotonic
market curve at each slot — what the counterparty sees, and the primary lens) and **yours** (same
board after the views in `views_2026.csv`). Picks are round.pick in a 10-team snake,
overall = (r−1)·10+p. **FAAB is not modelled**, so any net below understates a deal containing it.

---

## The trade — SharksGeneralManager pick swap (2026-08-14)

Negotiated across three offers; the third is the one that stands.

| | picks |
|---|---|
| get | 3.07, 4.04, 5.07, 6.04, 8.04, 9.07 + $25 FAAB |
| give | 3.05, 4.06, 5.05, 6.06, 8.06, 10.06 |

**Net generic +1.67 · yours +2.23**, plus the FAAB.

**Offer history — volume is not value.** Offer 2 added 9.07 + 12.04 for 10.06 + 11.05 and was
**worse** (+1.01 / +1.57): that leg received 0.76 + (−0.13) and sent 0.76 + 0.52, a **−0.65**. Pick
12.04 (overall 114) sits past the point where the board values a pick *below replacement* — it read
as generosity and was a cost. Offer 3 dropped it, leaving 9.07-for-10.06 at 0.76 vs 0.76, an exact
wash, so offer 3 ≡ offer 1 in value.

**Where the +1.67 comes from — one place, not six.** Picks 54 and 56 straddle an isotonic cliff:
2.99 vs 1.72, **1.27 points across two slots**. At 54 you reach the DJ Moore (52.6) / Lamar (52.7) /
Burden (52.9) cluster; at 56 they are gone and it is Pierce and Henderson. Every other swap in the
deal is worth 0.1–0.5.

**Tail analysis — the reason to take it** (6,000 sharp-league sims, best-available VORP by slot):

| swap | RB median | **RB p10** |
|---|---|---|
| R3: give 25 → get 27 | 0.00 | **−0.42** |
| R4: give 36 → get 34 | 0.00 | **+0.50** |
| R5: give 45 → get 47 | −0.18 | 0.00 |
| R6: give 56 → get 54 | 0.00 | **+0.25** |
| R8: give 76 → get 74 | 0.00 | 0.00 |

Medians barely move; the entire effect is in the tail. Two slots at the median gets a near-identical
player — two slots in the bad 10% of drafts is where a run empties a tier and you fall past its edge.
Net downside improves ~+0.33, better at three of five picks. WR tails are flat at every swap:
receivers are deep enough here that two picks never crosses a tier edge.

---

## Rejected proposal (not a trade) — 1.07 package, 2026-08-09

Recorded only because the structure is the template for any future move up.

Get 1.07, 3.07, 5.07, 6.04, 7.07, 10.04 + $25 FAAB · give 2.06, 3.05, 4.06, 5.05, 7.05, 9.05.
**Net generic +3.57 · yours +3.76.** Rounds 3/5/7 were near-identical swaps two slots later; the real
content was **2.06 + 4.06 + 9.05 → 1.07 + 6.04 + 10.04** — paying round 2 and round 4 for a second
first-rounder. Right direction in a 10-team league where only 7 starters score. Simulated
availability at the resulting pick set: **CMC 82% at pick 5, Amon-Ra 72% at pick 7** (~60% joint).

**The cost, quantified:** the round-4 pick vanishes and the gap runs 7 → 27. Davante Adams (ADP 41.1,
our WR12, the board's biggest edge at +18 vs ADP) drops from **96% available at 36 to 18% at 47**;
Higgins, McMillan, Nabers and Egbuka die in the same gap. The deal pays a whole ADP-30-to-45 tier for
a second first. Declined by the counterparty.

---

## Standing method notes

- **Evaluate on generic first.** If a deal only works on our board it is a bet on the views, not on
  price. The trade above is +1.67 generic, so it does not depend on being right about anything.
- **Read the tail, not the median.** Slot changes of 1–3 picks almost never move the median and
  routinely move p10 by 0.3–0.5. That is the whole content of a small pick swap.
- **Count the legs.** Added picks late in a negotiation are where value quietly leaves; anything past
  ~ADP 110 is at or below replacement and is not compensation.
- **FAAB is unmodelled and excluded** from every net here.

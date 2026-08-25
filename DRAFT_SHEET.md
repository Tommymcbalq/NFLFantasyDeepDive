# Draft sheet — 2026, slot 1.05

Picks (post-trade): **5, 16, 27, 34, 47, 54**, then 65, 74, 85, 87, 105, 116, 125, 136, 145.
Board: `results/board_2026_v5_mustar.csv`. Room: mock board of 2026-08-25, blended with
your stated order, loose at Henry / Nico / Olave / Nabers / London.

---

## Pick 5 — the one real judgement call

**80%: 1.04 takes Nacua → McCaffrey is there (88% of the time).**

| | board value | model's paired verdict |
|---|---|---|
| McCaffrey | **12.32** | |
| Amon-Ra St. Brown | 11.43 | **Amon-Ra by 0.33 PPG** (wins 98% of paired sims) |

McCaffrey is worth **+0.89 more in isolation**, yet the model still wants Amon-Ra. The reason
is not that Amon-Ra is better. It is that you can only ever *start* 4 RBs (RB, RB, FLEX, FLEX),
and the RB pool at your later picks — Henry 9.82, Etienne 9.75, Jacobs 7.97, Javonte 7.85 — is
deeper than the WR pool there (Rice 9.11, then McMillan 6.81, Adams 6.35). Opening with a WR
lets you absorb **four** of those RBs; opening with McCaffrey lets you absorb three and forces a
second, worse WR later.

**Treat this as a coin flip, not a recommendation.** 0.33 PPG sits far inside per-player
posterior SD of 1.3–1.8. And the argument depends entirely on the later RB pool staying deep —
which is precisely the thing you say your room *doesn't* do. If RBs fly in rounds 3–5 the way
you've described, that WR-first edge inverts and McCaffrey is correct. This is an L4 call.

**20%: 1.04 takes McCaffrey → Nacua is there.** Same logic, and cleaner: take Amon-Ra (11.43)
over Nacua (10.96) if both are up; the gap is small but points the same way as the structure.

---

## Pick 16

| | |
|---|---|
| **Derrick Henry** | 51% — the pick |
| Rashee Rice | 33% |
| Saquon Barkley | 8% |

Henry 9.82 vs Saquon 9.78 is a dead heat on value; Henry wins on floor (gap **+4.05** vs
**+1.10**). Saquon is there 72% of the time and the model still takes Henry — but at a 0.04 PPG
gap, take whichever you actually want.

**Fall-through answers — if he drops to you at 16, do you take him?**

| player | there | take |
|---|---|---|
| Saquon | 72% | 8% (coin flip on merit) |
| Justin Jefferson | 87% | **0%** |
| Ashton Jeanty | 47% | **0%** |
| Brock Bowers | 99% | **0%** |
| CeeDee Lamb | 4% | **0%** |

**So: no.** CeeDee at 8.88, Jefferson at 7.56, Jeanty at 8.47, Bowers at 3.46 are all below
Henry/Rice. If 2.02 passes on CeeDee, **let him go** — that is not your value.

---

## Pick 27

| | |
|---|---|
| **Rashee Rice** if there (46%) | taken **99%** |
| **Travis Etienne** otherwise | 44% |

Rice reaches 27 less than half the time and reaches 34 essentially never. Etienne reaches 34
about **81%** of the time. Take the one who cannot come back.

Nico Collins (there 39%) → take **0%**. Chris Olave (there 55%) → take **0%**. Do not.

---

## Pick 34

| | |
|---|---|
| **Travis Etienne** if there | 48% |
| **Josh Jacobs** otherwise (~100% there) | 50% |

---

## Picks 47 / 54 — weakest part of the sheet

47: Javonte Williams 37% | Skattebo 26% | McMillan 14% | Adams 7%
54: DJ Moore 21% | Judkins 17% | Swift 14% | Adams 13%

**Caveat: your mock only ran to pick 39.** Everything past that is ADP with σ = 9, so these
are the least reliable rows here. If you have any read on rounds 5–6 in your room, this is
where it pays.

---

## What the model refuses to do

- No expected-games multipliers (tested: significantly **worse** than nothing, −2.36, p = .0085).
- No availability guessing — the room order is declared by you, not simulated from ADP.
- δ_RB = 1.40 is carried **on your authority** as a statement about your league's meta, and is
  excluded from the January-scored column.

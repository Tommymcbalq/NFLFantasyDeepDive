# Owner draft-order prediction vs realised — 2026 league draft

Recorded 2026-08-24, mid-draft. The owner stated, from memory, the expected pick order for
overall picks 7–26 (from 1.07 through his own 3rd-round turn). Scored against the realised
order captured in `league_draft_2026.csv`.

| metric | value |
|---|---|
| set overlap | **20 / 20 (100%)** — no false positives, no misses |
| exact slot correct | 8 / 20 |
| mean absolute slot error | **1.40** |
| median slot error | 1.0 |
| within 2 slots | 14 / 20 |
| max slot error | 5 (Jeremiyah Love, called 18, went 23) |

**Why this matters for §R (the behavioral draft simulator).** The result decomposes cleanly into
the two things a Plackett–Luce / conditional-logit model separates:

1. **The choice set is near-deterministic.** Perfect set recall over a 20-pick window means the
   underlying player worths `w_j` are well-identified in this league — the managers agree on *who*
   belongs in the window.
2. **Only the ordering within it is stochastic**, and weakly so. A mean slot error of 1.40 implies a
   **low softmax temperature τ** — a chalky room, in the model's terms.

This is the strongest evidence to date that §R is worth building: the quantity the model exists to
produce is **P(player available at pick k)**, and that is governed by set membership far more than
by exact ordering. It is also a *calibration* datapoint of the kind §R's validation plan calls for
(calibration, not accuracy), obtained before any model was fitted.

**Caveats.** One observation, one owner, one draft — this is an anecdote about τ, not an estimate.
It is also recall stated *after* the picks occurred rather than a logged ex-ante forecast, so it is
not a clean out-of-sample prediction; treat it as an upper bound on achievable accuracy. The
τ-persistence pre-test in `EDA_PLAN.md`-successor §R still governs whether per-manager parameters
are estimable at all.

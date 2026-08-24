# The model — 10 steps

Two halves. **Steps 1–7 are the model's job**: what a player is worth. **Steps 8–10 are yours**:
when he is gettable, and therefore what to do. The model never guesses availability — that is the
one input the owner observes better than any fit, and every attempt to simulate it has produced
false precision (Saquon at 1% when the truth is bimodal).

---

## VALUE — the model's half

**1. μ̂ — what he has done.** Recency-weighted mean of season means, half-life 1 season.
Beat seven alternatives in §S (median, trimmed, Huber, p60, role-gated, slope, usage-implied);
none replaced it. **Calibrated**: raw μ̂ is over-dispersed, slope 0.667 WR / 0.605 RB (§W1).

**2. Age.** §H's per-era curve applied to μ̂: +1.18 WR / +1.31 RB, 9/10 folds, p < .01. This is a
structural correction, not an edge claim — §H5 found the market prices age correctly. It is the
largest single correctable error in the data arm, and at RB it is essentially the only one.

**3. m(ADP) — what he costs.** Isotonic regression of realised PPG on log ADP, ten boards
2015–2024, 666 WR / 603 RB. Monotone only; the flat steps are real information about where the
market cannot distinguish prices.

**4. θ\* — combine them.** θ\* = (1−B)·μ̂ + B·m(ADP), with **B = V/(V+τ²) estimated, not chosen**.
Extends to a three-way GLS if a projection is ever adopted (derived in §W2; WS1 rejected the
projection, so the two-way blend stands).

**5. Views — your discretion on value.** Black–Litterman: π = θ\*, Σ estimated, views as (P, q, Ω)
with **Ω declared, never fitted**. Posterior shift decomposes exactly by view, so every one is
scoreable in January. 37 player views + δ_RB as a structural view.

**6. Replacement.** VORP = value − replacement(position), replacement = PPG rank among players with
≥8 games, at the position counts your league actually drafts (44 RB / 63 WR / 14 TE / 19 QB in the
top 140). Does more work than every other layer combined (ablation: Spearman .77 without it).

**7. Floor.** λ = 0.10 × (p25 over *scheduled* weeks − positional reference). Suspensions out of the
denominator, blank below 34 weeks, thin-data players neutral.

→ **`board_2026_v2.csv`.** One number per player. This is the only thing the model asserts.

---

## DECISION — your half

**8. Availability — YOUR odds.** For each player you care about, P(available at each of your picks).
Supplied by you in `data/drafts/availability_priors.csv`, not simulated. Bimodal is fine and
expected: "falls to me or goes right before" is a real shape a normal-CDF model cannot express.

**9. Lineup-marginal value.** A player who would not start is worth **zero**, whatever his board
value. Marginal = max(0, value − the starter he would replace) given roster state
(1QB/2RB/2WR/1TE/2FLEX/1DST).

**10. The pick.** Take the player maximising *marginal value now* minus *expected marginal value at
your next pick*, using your odds from step 8. Formally the cost of waiting is

    W_p = sum_i E[ delta_i * 1(N_p >= i) ]

the survival-weighted sum of tier gaps: **steps enter linearly and unbounded, decay only through
indicators bounded in [0,1]**. So *flat tiers are safe to wait on however fast they drain; steps are
not, however slowly.*

---

## What this model does not do

- **Does not project from usage/efficiency inputs.** Tested properly in §W1 across three estimators
  and a gated feature set: WR +0.249 (p = .404, uninformative), RB **−0.295**. Ninth null.
- **Does not price availability into value.** No expected-games multiplier — tested and
  *significantly worse* than doing nothing (−2.36, p = .0085).
- **Does not claim to beat ADP on ordering.** §M/§W3: the board's ordering is a wash with ADP; what
  loses money is roster mix, and what wins is the room's bias — the ADP null scores +169 season
  points against a biased room versus against ADP opponents.
- **Does not simulate who your league takes.** Step 8 is yours.

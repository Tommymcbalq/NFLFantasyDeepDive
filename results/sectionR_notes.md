# §R — Behavioral draft model, fitted on the live 2026 league draft

*Pre-registration: `EDA_PLAN8.md` §R (2026-08-24). Spec: `fantasy_draft_model.md`.
Binding constraints: REPORT.md §38(1) scale identification, §38(2) τ-persistence pre-test.*

## §R0 What was fixed before fitting

Model set (declared before any coefficient was looked at):

- **M0** `U_j = β_v v_j`
- **M1** `U_j = β_v v_j + Σ_p β_p · need_{m,p,t} · 1[pos(j)=p]`  — §R1 exactly as written
- **M2** M1 + positional intercepts α_p (WR = reference) + β_flex · flexneed · 1[pos ∈ RB,WR]

`need_{m,p,t} = max(0, req_p − n_{m,p,t})` with req = QB 1, RB 2, WR 2, TE 1;
`flexneed = max(0, 2 − [(n_RB−2)_+ + (n_WR−2)_+])`.
M2 was declared because the OLS baseline it must beat contains exactly a positional
offset, and under M1 the need terms are identically zero for a manager who has filled
his base starters — which is most of the room by round 9, precisely the horizon of
interest. Selection criterion, also declared in advance: **leave-one-pick-out mean
predictive log-likelihood**, not in-sample fit and not hit rate.

**τ = 1, β_v free** (§38(1)). Only U/τ is identified. The reported temperature is 1/β_v.

**No per-manager layer.** §38(2)'s persistence pre-test requires two drafts; we have one.
League-mean parameters only, no affinity spike-and-slab, no meta×profile.

**The owner's 9 picks are excluded from the likelihood** (they still remove players from
the pool and still update his roster state). We are modelling *opponent* behaviour in
order to predict what falls to him; he drafts against 37 logged personal views, so his
picks are drawn from a different DGP. Exclusion is preferred to a dummy because a dummy
on the owner would be a pure intercept shift in a softmax over an unchanging choice set —
it cannot represent a different ranking, only a different noise level, and with 9
observations it would be estimated on nothing. n = 78 opponent choice observations.

**Off-board picks.** After suffix-normalised name matching (`Chris Godwin` → `Chris Godwin Jr.`,
`James Cook` → `James Cook III`), **0 of 87** realised picks fell outside our 204-player
universe. The forward simulation nonetheless needs a non-zero rate, because the remaining
63 picks must absorb ten defences that the board does not contain. The raw rate 0/87 is
replaced by its Jeffreys posterior mean λ = 0.0057: at each simulated pick, with probability
λ the manager takes someone not on our board and drains nothing. Because that is an
extrapolation and not an estimate, a sensitivity at λ = 0.15 — a room that has started on
defences — is reported below.


## §R1–R2 Fitted parameters

**Selected model: M2R** (by the pre-declared LOPO criterion).

### M2R
| parameter | estimate | SE (obs. info) | SE (cluster, G=10) | z |
|---|---|---|---|---|
| `v` | +1.0922 | 0.0872 | 0.0630 | +12.52 |
| `need_QB` | +2.8823 | 1.8007 | 0.7528 | +1.60 |
| `need_RB` | +0.6744 | 0.3748 | 0.3501 | +1.80 |
| `need_TE` | +2.0430 | 1.0792 | 0.9529 | +1.89 |
| `need_WR` | +1.0142 | 0.3630 | 0.3331 | +2.79 |
| `int_QB` | -3.4056 | 1.7866 | 0.5736 | -1.91 |
| `int_RB` | +1.6621 | 0.4045 | 0.4304 | +4.11 |
| `int_TE` | +1.9746 | 1.1025 | 0.9841 | +1.79 |
| `flex` | -0.4876 | 0.4234 | 0.4744 | -1.15 |

**Implied temperature 1/β_v = 0.9156 (SE 0.0731, delta method).**
Read it as: a value gap of 0.92 board points between two available players is one
logit unit of preference. The board's undrafted pool spans about 6.1 points from the
top to the 10th percentile, so this is a **chalky room** — consistent with, though not a
confirmation of, the 1.40-slot mean error recorded in `prediction_calibration_2026.md`.

### M1 (the §R1 specification as written)
| parameter | estimate | SE (obs. info) | SE (cluster, G=10) | z |
|---|---|---|---|---|
| `v` | +0.9974 | 0.0811 | 0.0808 | +12.30 |
| `need_QB` | -0.4596 | 0.4708 | 0.5043 | -0.98 |
| `need_RB` | +1.1859 | 0.3447 | 0.2898 | +3.44 |
| `need_TE` | +3.8187 | 0.5616 | 0.5769 | +6.80 |
| `need_WR` | +0.5103 | 0.3252 | 0.3158 | +1.57 |

### M0 (value only)
| parameter | estimate | SE (obs. info) | SE (cluster, G=10) | z |
|---|---|---|---|---|
| `v` | +0.8288 | 0.0677 | 0.0562 | +12.24 |

Standard errors are reported two ways. The observed-information SEs assume the picks are
independent conditional on the state; the manager-clustered SEs (G = 10) allow arbitrary
dependence within a manager. **With 10 clusters the clustered SEs are themselves noisy and
biased downward-in-coverage**; they are reported for honesty, not because they are better.
Every one of these numbers is estimated from a single realisation of a single draft.


## §R3 Validation — calibration first, accuracy second

Leave-one-pick-out over the 78 opponent picks: refit without pick t, predict the full
distribution over A_t, score the realised choice.

| model | LOPO mean log-lik | SE | top-1 | top-5 |
|---|---|---|---|---|
| UNIF | -5.0699 | 0.0180 | — | — |
| OLS | -3.4190 | 0.1438 | 0.090 | 0.321 |
| M0 | -3.6281 | 0.1567 | 0.115 | 0.436 |
| M1 | -3.3736 | 0.1471 | 0.141 | 0.423 |
| M2 | -3.3834 | 0.2343 | 0.141 | 0.474 |
| M2R | -3.2424 | 0.1426 | 0.141 | 0.474 |

**Paired comparison against the OLS baseline** (per-pick differences in held-out log-lik,
paired t on 78 picks):

- M0: Δ = -0.2091 ± 0.1718  (t = -1.22)
- M1: Δ = +0.0453 ± 0.1595  (t = +0.28)
- M2: Δ = +0.0356 ± 0.2488  (t = +0.14)
- M2R: Δ = +0.1766 ± 0.1718  (t = +1.03)

The OLS baseline is converted into a comparable pick model in the only honest way: its
fitted regression gives each player a predicted pick position π̂_j with residual SD σ, so
`P(j chosen at t) ∝ φ((t − π̂_j)/σ)` renormalised over A_t. Both models are then scored on
the same quantity — the held-out probability mass placed on the player actually taken.
The reproduced OLS fit is R² = 0.832, residual SD 10.62 picks, positional offsets
(picks earlier than a WR at the same ADP rank) TE 11.9, QB 3.4, RB 1.9.

### Pick-level calibration (leave-one-pick-out, all (pick, candidate) pairs)

| predicted-probability bucket | n | mean predicted | realised frequency | ±SE |
|---|---|---|---|---|
| [0.0, 0.005) | 10250 | 0.0008 | 0.0003 | 0.0002 |
| [0.005, 0.01) | 785 | 0.0071 | 0.0076 | 0.0031 |
| [0.01, 0.02) | 647 | 0.0141 | 0.0216 | 0.0057 |
| [0.02, 0.05) | 581 | 0.0314 | 0.0361 | 0.0077 |
| [0.05, 0.1) | 208 | 0.0672 | 0.0962 | 0.0204 |
| [0.1, 0.2) | 53 | 0.1345 | 0.1132 | 0.0435 |
| [0.2, 0.4) | 29 | 0.2781 | 0.1379 | 0.0640 |
| [0.4, 0.7) | 15 | 0.4966 | 0.2000 | 0.1033 |
| [0.7, 1.0) | 1 | 0.7642 | 1.0000 | 0.0000 |

### Survival calibration — the quantity the tool actually emits

This is the test that matters, and it is a genuine temporal holdout: at each anchor pick
t₀ ∈ {31, 41, 51, 61, 71} the model is **refitted using only picks before t₀**, then run
forward by Monte Carlo (2,000 runs) for 15 picks. Each still-available player gets a
predicted P(survive), scored against whether he actually did.

| predicted survival bucket | n | mean predicted | realised | ±SE |
|---|---|---|---|---|
| [0.0, 0.1) | 6 | 0.012 | 0.167 | 0.152 |
| [0.1, 0.3) | 9 | 0.182 | 0.111 | 0.105 |
| [0.3, 0.5) | 20 | 0.398 | 0.450 | 0.111 |
| [0.5, 0.7) | 40 | 0.619 | 0.500 | 0.079 |
| [0.7, 0.9) | 118 | 0.815 | 0.797 | 0.037 |
| [0.9, 0.98) | 200 | 0.951 | 0.970 | 0.012 |
| [0.98, 1.0) | 377 | 0.994 | 0.997 | 0.003 |

Brier score 0.0558 against a base-rate Brier of 0.0879 — **skill +0.365**.

**Read the pick-level table honestly.** The model is well calibrated across the mass of the
distribution (every bucket below p = 0.2 matches within about one SE) but **overconfident in
its confident tail**: the [0.2, 0.4) bucket predicts 0.28 and realises 0.14 (n = 29), and
[0.4, 0.7) predicts 0.50 and realises 0.20 (n = 15). Those cell counts are far too small to
estimate a miscalibration slope, but the direction is consistent and it has a mechanical
cause — a softmax over 100+ alternatives with a single global temperature concentrates too
much mass on the modal candidate when the board has an obvious top name. The practical
consequence for the tool is that **P(gone) for the single most obvious next pick is likely
overstated, so survival probabilities for the very top of the board should be read as a
lower bound.** The aggregate survival calibration below, which is what VONA actually
consumes, is much better behaved.

Top-1 and top-5 hit rates are reported above as **secondary diagnostics and explicitly not
as the adoption criterion** (`fantasy_draft_model.md`: "validate by calibration, not
accuracy"). A model can win top-1 by always naming the consensus next man and still be
useless for survival probabilities, which are what VONA consumes.


## §R2b The QB separation problem — the one anomaly worth chasing

M2 as first fitted returned `int_QB` = −15.3 with a standard error of **419**. That is the
signature of quasi-complete separation, and the data say exactly why:

- In **22 of 87** realised picks, the highest-*value* player on our board was a QB.
- A QB was actually taken in **1** of them (Josh Allen at pick 40).
- Of the **5** such picks made by a manager who *already had a QB*, **0** took a QB.

Zero events out of 5 exposures drives the "QB, already filled" coefficient to −∞ while
`need_QB` runs to +∞ to keep the QB-needy combination finite. The MLE does not exist; the
likelihood is monotone along that ridge.

This matters far beyond a standard error. Our board's `final` is VORP-scaled against a
replacement QB, which makes **Dak Prescott (v = 6.75) the single most valuable available
player at pick 88**, ahead of every WR and RB. Under M1 — which has no way to say "a manager
with a QB does not take a QB" — every one of the seven QB-filled managers is modelled as
wanting Dak more than anyone else, and the simulator drains the QB shelf in a way the room
demonstrably does not. **M1 is misspecified in precisely the region that drives the pick-88
recommendation**, and that was visible from the descriptive table above before any VONA was
computed.

The remedy is the textbook one for separation and it was chosen for that reason, not for its
answer: **M2R = M2 with a weakly-informative ridge N(0, 5²) on every coefficient except β_v**
(β_v is left unpenalised because it carries the scale normalisation τ = 1). This is the
penalised-likelihood / Firth-type fix: it renders the mode finite and the curvature
interpretable without changing the sign or the ordering of anything that was identified.
M2R was added to the model set on 2026-08-24 **after** seeing SE = 419 and **before** any
player-level VONA output was inspected, and it is scored on the same pre-declared LOPO
criterion as everything else. Because the choice of specification is doing real work here,
**every §R4 output below is reported under both the selected model and under M1, the §R1
specification exactly as pre-registered.** Where they disagree, that disagreement is the
finding.

For reference, the room takes the maximum-value available board player on only
**10.3%** of picks — so "the room drafts our board" is false in general, not just for QBs.

## §R4 Outputs

Owner roster after 9 picks: {'QB': 1, 'RB': 3, 'TE': 0, 'WR': 5}. Starters required 1QB/2RB/2WR/1TE/2FLEX/1DST.
**Open starting slots: TE and DST.** DST is not on the 204-player board and no DST has been
taken in 87 picks, so its scarcity risk is zero and its lineup contribution is a constant
that cancels across every strategy compared below; it is scheduled for the last pick.

### Positional run risk, 88 → 105

| position | E[gone] | P(≥1) | P(≥2) | P(≥3) | P(≥5) |
|---|---|---|---|---|---|
| QB | 1.95 | 0.955 | 0.682 | 0.255 | 0.004 |
| RB | 4.20 | 0.999 | 0.981 | 0.898 | 0.401 |
| TE | 2.72 | 0.987 | 0.859 | 0.552 | 0.070 |
| WR | 8.04 | 1.000 | 1.000 | 1.000 | 0.993 |

### VONA

| spec | decision | next | pos | best now | player | E[best next] | SE | sd | p10 | p90 | **VONA** | P(top survives) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M2R | 88 | 105 | QB | 6.751 | Dak Prescott | 6.214 | 0.005 | 0.491 | 5.748 | 6.751 | **+0.537** | 0.449 |
| M2R | 88 | 105 | RB | 4.422 | Rico Dowdle | 3.416 | 0.006 | 0.498 | 2.882 | 4.422 | **+1.006** | 0.181 |
| M2R | 88 | 105 | TE | 2.372 | Travis Kelce | 2.125 | 0.003 | 0.279 | 1.785 | 2.372 | **+0.247** | 0.552 |
| M2R | 88 | 105 | WR | 6.366 | Wan'Dale Robinson | 5.294 | 0.007 | 0.606 | 4.713 | 6.292 | **+1.073** | 0.075 |
| M2R | 105 | 116 | QB | 6.212 | (simulated) | 6.052 | 0.009 | 0.539 | 5.621 | 6.751 | **+0.160** | 0.763 |
| M2R | 105 | 116 | RB | 3.411 | (simulated) | 3.053 | 0.007 | 0.428 | 2.652 | 3.272 | **+0.358** | 0.465 |
| M2R | 105 | 116 | TE | 2.126 | (simulated) | 1.969 | 0.005 | 0.333 | 1.616 | 2.372 | **+0.157** | 0.648 |
| M2R | 105 | 116 | WR | 5.291 | (simulated) | 4.742 | 0.007 | 0.441 | 4.216 | 5.229 | **+0.549** | 0.353 |
| M1 | 88 | 105 | QB | 6.751 | Dak Prescott | 5.694 | 0.004 | 0.356 | 5.453 | 5.814 | **+1.057** | 0.055 |
| M1 | 88 | 105 | RB | 4.422 | Rico Dowdle | 4.079 | 0.006 | 0.528 | 3.272 | 4.422 | **+0.343** | 0.703 |
| M1 | 88 | 105 | TE | 2.372 | Travis Kelce | 2.223 | 0.003 | 0.236 | 1.863 | 2.372 | **+0.149** | 0.715 |
| M1 | 88 | 105 | WR | 6.366 | Wan'Dale Robinson | 5.393 | 0.007 | 0.617 | 4.713 | 6.292 | **+0.973** | 0.100 |

### Marginal starting-lineup VONA — the decision-relevant version

Raw-value VONA above answers "how much board value decays at each position". It is *not*
the decision quantity for a manager who already holds eight RB/WR: what matters is how much
a player would add to the **starting lineup** (1QB/2RB/2WR/1TE/2FLEX), which for a
positionally-saturated roster is far less than his board value. Owner's current holdings by
position, in board value: QB: [5.500854847552748]; RB: [12.176922423207584, 9.633204356032396, 5.964104229073328]; TE: []; WR: [8.419248047021613, 8.33498281382771, 7.72482466317768, 6.005679518107883, 3.787655271368103] — current lineup total 57.796.

| position | best now | marginal now | E[marginal at 105] | SE | sd | **lineup VONA** |
|---|---|---|---|---|---|---|
| QB | Dak Prescott | +1.250 | +0.709 | 0.008 | 0.486 | **+0.540** |
| WR | Wan'Dale Robinson | +0.361 | +0.049 | 0.002 | 0.116 | **+0.312** |
| TE | Travis Kelce | +2.372 | +2.123 | 0.004 | 0.279 | **+0.249** |
| RB | Rico Dowdle | +0.000 | +0.000 | 0.000 | 0.000 | **+0.000** |

### Breakeven on the QB call

The QB-vs-TE decision rests on the board's gap between the best available QB and the
incumbent starter, 1.250 points, against a per-player board posterior SD of 1.2-1.8. The
honest question is therefore not the point estimate but how wrong the board would have to
be to flip the decision. Shifting every available QB's evaluated value down by c (opponent
behaviour untouched, so the survival curves do not move):

| c | marginal now | E[marginal at 105] | lineup VONA(QB) | beats TE's 0.249? |
|---|---|---|---|---|
| 0.00 | +1.250 | +0.727 | +0.522 | yes |
| 0.25 | +1.000 | +0.486 | +0.514 | yes |
| 0.50 | +0.750 | +0.346 | +0.403 | yes |
| 0.75 | +0.500 | +0.231 | +0.269 | yes |
| 1.00 | +0.250 | +0.115 | +0.134 | **NO** |
| 1.25 | +0.000 | +0.000 | +0.000 | **NO** |

The QB call survives a downward shift of the whole QB shelf up to c between 0.75 and 1.00;
it flips somewhere in that bracket. Equivalently, the true gap between the best available
QB and the incumbent would have to be below roughly 0.4-0.5 points per game rather than the
board's 1.25. That is a margin of about 0.8 points, inside one board posterior SD, so the
recommendation is directional and should be read as such.


### Expected starting-lineup value by first action at pick 88

Owner follows a marginal-starting-lineup-value greedy rule at [105, 116, 125, 136, 145]; opponents follow
the fitted logit; parameters redrawn from N(θ̂, V̂) each run; 1,500 runs each.

| first action | E[lineup] under M2R | SE | E[lineup] under M1 | SE | modal player |
|---|---|---|---|---|---|
| greedy | 60.925 | 0.012 | 60.448 | 0.007 | Travis Kelce |
| TE | 60.925 | 0.012 | 60.448 | 0.007 | Travis Kelce |
| RB | 60.536 | 0.014 | 60.071 | 0.007 | Rico Dowdle |
| WR | 60.890 | 0.014 | 60.424 | 0.007 | Wan'Dale Robinson |
| QB | 61.170 | 0.007 | 61.266 | 0.006 | Dak Prescott |

### Survival curve excerpt — top 15 available by board value

| player | pos | v | P(avail at 105) | P(avail at 116) | P(avail at 125) |
|---|---|---|---|---|---|
| Dak Prescott | QB | 6.75 | 0.444 | 0.326 | 0.194 |
| Wan'Dale Robinson | WR | 6.37 | 0.074 | 0.008 | 0.001 |
| Jakobi Meyers | WR | 6.29 | 0.091 | 0.011 | 0.002 |
| Quentin Johnston | WR | 5.85 | 0.210 | 0.042 | 0.009 |
| Patrick Mahomes | QB | 5.81 | 0.727 | 0.623 | 0.493 |
| Bo Nix | QB | 5.75 | 0.745 | 0.638 | 0.514 |
| Jared Goff | QB | 5.62 | 0.774 | 0.682 | 0.572 |
| Jaxson Dart | QB | 5.53 | 0.785 | 0.690 | 0.579 |
| Matthew Stafford | QB | 5.49 | 0.806 | 0.717 | 0.609 |
| Justin Herbert | QB | 5.45 | 0.810 | 0.721 | 0.610 |
| Trevor Lawrence | QB | 5.33 | 0.826 | 0.734 | 0.635 |
| Alec Pierce | WR | 5.23 | 0.433 | 0.169 | 0.065 |
| Stefon Diggs | WR | 4.75 | 0.605 | 0.348 | 0.186 |
| Josh Downs | WR | 4.71 | 0.629 | 0.353 | 0.196 |
| Brock Purdy | QB | 4.57 | 0.918 | 0.870 | 0.810 |

### Sensitivity to the off-board hazard

λ = 0.006 (estimated) vs λ = 0.15 (a room that starts taking DSTs). Mean absolute change in
P(available at 105) across undrafted players: 0.0213; max 0.0837.


## §R5 Recommendation, and what it is worth

**Take the QB at pick 88.** Both the selected specification and the pre-registered §R1
specification rank a QB first, by +0.245 and +0.818 expected starting-lineup points
respectively over the next-best action (TE). Spec-robustness is the reason to believe it;
the magnitude is not large.

The logic, stated so it can be checked: the owner's roster is **positionally saturated at
RB/WR** — eight of them, six starting slots, and the best available RB adds exactly **0.000**
to his starting lineup. Only three positions can add anything, and the ranking is not by
who is best but by **how fast the marginal upgrade decays**:

- **TE** is the position he *needs*, and it is the position it is **cheapest to wait on**. The
  tier is flat by construction (the isotonic prior returns steps), so losing the top name
  costs 0.249 points, not a cliff — even though the room drafts TEs 12 picks earlier
  than their ADP-implied slot and 8 have already gone.
- **QB** decays fastest (0.540) despite only 7 QBs being gone, because the upgrade over the
  incumbent is a step function: there is one QB left worth a real upgrade and P(he survives
  to 105) is a coin flip.
- **WR** decays 0.312 but from a marginal base of only 0.361 — the ninth WR barely improves a
  lineup that already starts six RB/WR.

So the sequence the simulator recommends is **QB now, TE at 105, DST last**, with the
remaining picks going to whatever has the largest marginal starting-lineup value at the time.
P(the top TE survives to 105) = 0.55, and P(at least one of the four flat-tier TEs
survives) is essentially 1 given only 2 TEs are expected to go in the window.

**What this recommendation is worth.** The gap between the best and second-best action is
0.245 lineup points. A single player's board posterior SD is 1.2–1.8. The breakeven
analysis above shows the call flips if the true QB upgrade is overstated by ~0.8 points.
Every parameter comes from **one draft, n = 78 opponent choices**, and the model does not beat
the OLS positional baseline at conventional significance (Δ = +0.177 ± 0.172 nats,
t = 1.03). This is a decision aid with a directional answer, not a settled result.

## §R6 Does the behavioural model beat the positional correction?

**On the point estimate yes, at conventional significance no.** The selected conditional
logit gains +0.1766 nats per held-out pick over the ADP+position OLS baseline, with a paired
standard error of 0.1718 (t = 1.03, 78 paired picks). The §R1 specification as
pre-registered gains +0.0453 ± 0.1595 — indistinguishable from the baseline.

That is a real finding and it should be stated as such rather than buried: **most of what a
behavioural draft model knows about this room is already captured by "shift TEs 12 picks
earlier, QBs 3, RBs 2, and add Gaussian noise of 11 picks."** The conditional logit adds
three things the OLS cannot: it is a *proper distribution over the available set* (so it can
be simulated forward without an ad-hoc renormalisation), it is *roster-state dependent* (the
need terms are jointly significant and `need_WR` and `int_RB` are individually so), and it
correctly refuses to draft a second QB. The first of those is why it is worth building even
at parity on log-likelihood; the third is why M1 is not the version to ship.

This also speaks to §38(3): §M found that no pick sequence beat drafting the board against
ADP-drafting opponents, so any edge here must come from opponents being *predictably biased*.
The bias is real and measurable — the room takes the maximum-value board player only 10.3% of
the time, and the positional intercepts (`int_RB` +1.66, `int_TE` +1.97, `int_QB` -3.41 against a
WR baseline) say exactly where. Whether exploiting it is worth more than its estimation
error, on one draft, is not established.


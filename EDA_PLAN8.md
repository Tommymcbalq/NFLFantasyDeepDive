# EDA Round 8 — §R: the behavioral draft model, fitted on real league data
### Pre-registered 2026-08-24, mid-draft, before any fitting. Spec: `fantasy_draft_model.md`.

## What changed: the blocker is gone

§38 recorded strand 3 as SPECIFIED-NOT-BUILT, blocked on draft logs. **We now have one**:
`data/drafts/league_draft_2026.csv`, 87 picks from the owner's live 10-team league with team slots
and traded-pick ownership resolved. Plus a calibration datapoint
(`prediction_calibration_2026.md`): the owner's stated order for picks 7–26 achieved **20/20 set
recall, mean slot error 1.40**.

## The honest constraint, stated first

**One draft cannot support per-manager parameters.** §38's binding pre-condition was a
τ-persistence pre-test — estimate τ_m on one draft, again on another, correlate — and with a single
draft that test cannot be run. Therefore:

> **§R is built with league-mean parameters only. No per-manager layer, no spike-and-slab affinity
> term, no meta×profile interaction.** Those require multiple drafts and are deferred.

This follows the spec's own caution that most of the edge is in (a) the tier-cliff value curve,
(b) temperature, and (c) hand-set priors — not in rich per-manager estimation.

## §R1 The pick model

Conditional logit over the available set, exactly the Plackett–Luce factorisation:

    P(pick = j | A_t, roster_m,t) = exp(U_jt / tau) / sum_{k in A_t} exp(U_kt / tau)
    U_jt = beta_v * v_j  +  sum_p beta_pos_p * need_{m,p,t} * 1[pos(j)=p]

- **v_j** = our board's `final` value — already tiered, since the isotonic prior produces flat steps.
  This is the spec's "tier cliff" requirement, satisfied by construction rather than by refitting.
- **need** = a function of the picking manager's current roster against the 10-team starter
  requirement (1QB/2RB/2WR/1TE/2FLEX/1DST). Roster states are reconstructible pick-by-pick from
  the log.
- **Scale identification, per §38(1):** τ is FIXED AT 1 and β_v estimated. Only U/τ is identified,
  so estimating both is a ridge; the reported "temperature" is then 1/β_v.

## §R2 Fitting

Each of the 87 picks is one choice observation over the set actually available at that moment.
Reconstruct A_t and roster states from the log, fit by maximum likelihood. Report β_v, the
positional need coefficients, and the implied temperature with standard errors.

**Baseline to beat:** the ADP + positional-offset OLS already fitted this session (R² = 0.831,
residual SD 10.4 picks). §R must beat it on held-out picks or it is not worth its complexity.

## §R3 Validation — calibration, not accuracy

Per the spec. Leave-one-pick-out over the 87 observations: refit without pick t, predict the
distribution over A_t, and record the predicted probability of the player actually taken. Then:
- **Calibration curve**: bucket predicted survival probabilities and check realised frequencies.
  Things called 70% should happen ~70% of the time.
- Report top-1 and top-5 hit rate as secondary, explicitly NOT as the adoption criterion.

## §R4 Outputs — the part that is actually used

1. **Survival curves** P(player available at pick k) for every undrafted player, by Monte Carlo
   forward simulation (≥5,000 runs) rather than a normal-CDF approximation.
2. **Positional run risk**: P(≥ n players at position p go before pick k).
3. **VONA** — Value Over Next Available. For each position, the best available now minus the
   expected best available at the owner's *next* pick. This is the quantity that decides whether to
   take a position now or wait, and it is the deliverable.
4. A **recommendation for the current pick** that maximises expected starting-lineup value over the
   remaining schedule, subject to filling 1QB/2RB/2WR/1TE/2FLEX/1DST.

## §R5 Honesty clauses

- The owner is mid-draft. **A working tool beats a complete one.** If the full conditional logit
  cannot be fitted and validated cleanly, ship the survival curves and VONA off the already-fitted
  OLS baseline and say so.
- **n = 87 picks from one draft.** Every parameter is an estimate from a single realisation. Report
  standard errors and do not present point estimates as settled.
- The owner's own picks are in the fitting data and he does not draft to league consensus. Either
  exclude his 9 picks or include a dummy; state which and why.
- Nothing here touches `board_2026.csv`. §R consumes the board; it does not modify it.

Outputs: `scripts/60_draft_model.py`, `results/sectionR_notes.md`, `results/survival_curves.csv`,
`results/vona.csv`, and REPORT.md §44.

# EDA Round 6 — Positional scarcity and draft strategy under the owner's actual league
### Pre-registered 2026-08-09, before any round-6 fitting. Rules unchanged.

## Motivation

Six edge tests have returned null (§6.2, §B3, §E, §F2, §I3, §K) and §L found conversion rates
positionally flat. Together these say the market's *player-level* pricing is hard to beat. But
every one of those tests asked "is this player mispriced?" — none asked **"given a correct board,
what is the best sequence of picks?"**

Those are different problems. A perfectly ranked board still has to be converted into a roster
under constraints: you pick once per round, opponents remove players between your picks, and only
starting slots score. §L established the input: *any RB-vs-WR preference must come from scarcity
weighting, not from RBs justifying their price more often.* §M measures that weighting.

**The league (fixed, from the owner):** 10 teams, PPR, starters **1 QB, 2 RB, 2 WR, 1 TE, 2 FLEX
(RB/WR only), 1 DST**, no kicker. So league-wide starting demand is **40 RB/WR, 10 QB, 10 TE**.

**The frame gap, stated once and carried everywhere:** our ADP is FFC **12-team**. The owner's
league is **10-team**. Opponent behaviour must be modelled from the 12-team ADP *ordering* (which
is the best available consensus of how players are valued) while roster demand and replacement
level come from the **10-team** structure. Every result names its frame. This is a real limitation,
not a formality: 12-team ADP embeds 12-team scarcity, which inflates the relative price of scarce
positions versus what a 10-team league should pay.

## §M1 Replacement level and VORP (descriptive, per season 2015–2024)

For each season, from realized outcomes:

- **Starter demand** per position as above. Replacement level R_p is pre-defined two ways, both
  reported because they answer different questions:
  - **static**: the (D_p + 1)-th best player at position p, D_p = league-wide starting demand,
    with FLEX demand allocated to RB/WR in proportion to realized flex usage;
  - **marginal/last-starter**: the worst player who actually started a given week, averaged.
- **VORP_i = points_i − R_{p(i)}**, computed on season totals **and** on weekly optimal-lineup
  points (see §M3), both PPR.
- Report the **VORP curve by draft slot** per position: this is the object that answers "is elite
  TE worth a premium?" — the question is whether the TE curve is *steeper at the top* than the
  RB/WR curves, not whether TEs score more.

Pre-specified: replacement level is computed from **realized** outcomes for the historical
backtest (that is the correct counterfactual for evaluating a completed draft), and from the
**model board** for the 2026 recommendation. These are different uses and are never mixed.

## §M2 Strategy backtest by draft simulation

The core test. For each season 2015–2024 and each of the 10 draft slots:

1. Opponents draft by ADP order with a pre-specified noise model — pick ~ ADP + ε, ε calibrated
   from the *observed* SD of ADP in the FFC data (the `stdev` column), truncated to available
   players, with a simple positional-need constraint (no team drafts a 3rd QB, etc.). The noise
   model is fixed before any strategy is evaluated.
2. Our team follows strategy S.
3. Realized roster is scored on that season's actual outcomes (§M3).
4. Repeat over N = 200 simulated drafts per (season, slot) to average out opponent noise.

**Strategies, all fixed now** — no strategy may be added after seeing results:
- **S0 best-available by ADP** (the null: draft the market's board)
- **S1 best-available by our model board** (θ*/board_value from §G and the WR pipeline)
- **S2 RB-first** (RB with first two picks, then best available)
- **S3 zero-RB** (no RB before round 5, then best available)
- **S4 elite-TE** (TE within the first three rounds, then best available)
- **S5 VORP-greedy** (maximise marginal VORP over current roster need at every pick)

Comparison is **S_k vs S0**, per-season mean starting-lineup points, Diebold-Mariano clustered by
season, t(9 df) — the same standard every arm in this project has had to meet. BH q = 0.10 across
the five comparisons (new declared family; {H5,I3}, {K}, {L} and {L-EXT} stay closed).

## §M3 Scoring a roster honestly

Season totals are the wrong scorer for a lineup problem: you start the best available players
*each week*, and a player who misses six games is replaced. So the primary scorer is the
**weekly optimal starting lineup** — for each week, fill 1QB/2RB/2WR/1TE/2FLEX from the roster by
that week's realized points, sum over weeks 1–17 (regular season; the fantasy regular season is
weeks 1–14 with 15–17 as playoffs, and both windows are reported).

Bench depth therefore has value automatically, and no separate injury adjustment is needed — this
is the same reasoning that made §A's availability work matter, now applied at roster level.
DST and streaming are **out of scope**: a fixed DST is assigned to every strategy identically so it
cannot differentiate them. This is stated as a limitation, not hidden.

## §M4 The specific questions to answer

Each of these is a pre-specified read of §M1–§M3, not a new test:
1. What is the VORP of the TE1/TE2/TE3 tier under 10-team, 1-TE, no-flex-TE rules, relative to the
   RB/WR available at the same ADP? (The owner's elite-TE hypothesis.)
2. Same for QB.
3. Where does the RB/WR VORP curve cross — i.e. at what draft cost does the scarcity argument for
   RB stop paying?
4. Does the answer to (3) differ between the 10-team and 12-team frames? If elite-TE/QB premiums
   are a 12-team artifact, the owner should know that before drafting.

## §M5 Decision rule and honesty clauses

- A strategy is **recommended** only if it beats S0 under the §M2 screens. Otherwise the
  recommendation is "draft the board", and that is a legitimate and likely outcome.
- **Simulation is not evidence about the world**, only about the interaction of our board with a
  modelled opponent. If S1 beats S0 that partly reflects our board being right; if the board is no
  better than ADP (which §L and six null tests suggest for player-level value) then S1 ≈ S0 is the
  expected result and the interesting comparisons are S2–S5, which are about *sequencing*.
- Report the **spread across draft slots**: a strategy that wins on average but loses badly from
  pick 1 is not a usable recommendation.
- Report per-strategy variance, not just the mean. In a 10-team league the objective is arguably
  P(make playoffs) or P(win), not expected points; **report the full distribution of season
  outcomes** and note explicitly that mean points and win probability can rank strategies
  differently.

## Outputs

`results/sectionM_notes.md`, `vorp_curves.csv`, `strategy_backtest.csv`, figures,
`scripts/28_sectionM_scarcity.py`, `29_sectionM_draftsim.py`. Then REPORT.md §29–30 with the full
derivation per the documentation rule.

---

## §N RB tier finishes and offensive environment — pre-registered 2026-08-09, before fitting

**Question (owner's).** How often does an RB who finishes top-12 (and 13–24) play in a top-10
offence — and, the actionable version, does playing in a *projected* top-10 offence raise the
probability of a top-12 RB finish?

**Why this is not a repeat of §I3.** §I3 asked whether team environment predicts the market's
*errors* in PPG and found it ~77% priced. §N asks a different, conditional question: P(tier finish
| environment), which is about the *shape* of the outcome distribution rather than its mean, and is
the quantity a drafter actually faces. A variable can be fully priced in expectation and still
change the probability of the tail outcome that wins a fantasy league. Both a confirmation and a
null are informative.

**Design.**
- Panel 2015–2024 (2025 outcomes usable descriptively; no 2025 ADP exists).
- Offence rank: (i) *realized* team points scored, rank 1–32; (ii) *projected*, proxied by the
  preseason closing win total already held in `data/vegas/team_win_totals_2015_2025_covers.csv`,
  which is preseason-knowable. Both reported; (ii) is the actionable one and (i) the descriptive one.
- Outcome: positional finish tier (≤12, 13–24, 25–36), season totals and PPG-given-participation.
- **Base-rate discipline, fixed now:** the raw share of RB1s on top-10 offences is not
  interpretable on its own, because good offences also attract better RBs and higher draft capital.
  Primary quantity is therefore P(top-12 finish | projected top-10 offence, **draft cost bin**)
  versus the same off a top-10 offence — i.e. conditional on price, which is what a drafter chooses
  between. Report the unconditional version too, clearly labelled as confounded.
- Same for WR, as the comparison the owner's RB claim implies.
- Report cell n and Wilson intervals everywhere; single-season rates are not signal (§L).

**Screens.** New declared family: BH q = 0.10 over the pre-specified contrasts (RB ≤12, RB ≤24,
WR ≤12, WR ≤24, × projected/realized), plus temporal holdout 2015–21 → 2022–24. Closed families
({H5,I3}, {K}, {L}, {L-EXT}, {M}) are not reopened. Nothing enters θ*; a survivor would inform the
views layer and the draft-strategy read only.

Outputs: `results/sectionN_notes.md`, `rb_tier_environment.csv`, rerunnable script.

---

## §O TE and QB valuation — pre-registered 2026-08-09, before any fitting

**Motivation.** The board covers WR (rounds 1–4) and RB (§G) but not TE or QB, so two of the
owner's seven starting slots have no model behind them. §M established the two facts that make
this worth doing properly rather than by eye: in the owner's **10-team** league an elite TE is
worth **+12.7 to +18.5 points/season** over the RB/WR available at the same ADP (p = .002/.032) —
and the premium is *larger* at 10 teams than 12, because cutting demand from 12 to 10 raises the
RB/WR replacement bar faster than the TE bar — while **elite QB carries no premium at any
baseline** (QB1–5 premium +5.3 ± 12.8, p = .69; replacement level 281 season points, the highest
of any position). §O turns those aggregate findings into per-player values.

**Design: mirror §G exactly, nothing assumed from WR or RB.**
- **O1 Universe.** Top 24 TE and top 24 QB by 2026 ADP (`adp_ppr_2026_all_20260809.csv`);
  historical boards 2015–2024 carry ~20 TE and ~20–23 QB per year, so the panel is ~200
  player-seasons per position — comparable to the RB panel and subject to the same power limits.
- **O2 Inclusion rule**, fixed now from aggregate distributions only: TE drops player-games with
  **targets ≤ 1**; QB drops games with **pass attempts ≤ 5** (a non-participation mixture:
  injury-exits and mop-up relief). Report the excluded fraction and its mean PPR for each.
- **O3 Variance components**, re-estimated per position: σ̂²_W, τ̂²_B via the eq.-3 bias inversion,
  recency-weighted μ̂ at h = 1 with n_eff, heteroskedasticity by experience tier. **Pre-registered
  expectation of difference, not similarity:** QB PPG should be far less noisy per game than WR/RB
  (no touch-share volatility, ~35 attempts every week), which if true means QB μ̂ is *more*
  reliable and B should shrink less toward market. That is a prediction, recorded now.
- **O4 Market prior.** Isotonic PPG-on-ADP per position, with τ²(tier) — giving π and Σ for §J.
- **O5 LOSO**, arms (i) market-only and (ii) market+data EB, DM clustered by year, adoption at
  p < 0.10 **and** RMSE improvement. Same honesty clause as §G: if the data arm does not beat
  market-only, the position is market-anchored and we do not hunt for an arm that wins.
- **O6 Replacement level and VORP under the owner's actual league** (10 teams; 1 QB, 2 RB, 2 WR,
  1 TE, 2 FLEX RB/WR-only, 1 DST; no kicker). Starting demand is 10 QB and 10 TE. Report VORP by
  draft slot for TE and QB **alongside** the RB/WR curves already computed in §M1, on one scale,
  so the cross-positional comparison the owner needs at each pick is direct. Report both the
  10-team and 12-team frames; the ADP source is 12-team and that gap must be named.
- **O7 Streaming baseline.** §M found replacement level is a *bracket*, not a number, and that the
  bracket is widest exactly at the shallow positions — **90 points at QB and 71 at TE against
  50/46 at RB/WR**. So the TE and QB premiums are only real net of what streaming would have got
  you. Report every VORP against all three baselines (draft-only, season-foresight,
  weekly-foresight) and state which one the recommendation uses.

**DST** is out of scope by the owner's instruction and §M's finding that a fixed DST cannot
differentiate strategies.

Outputs: `results/sectionO_notes.md`, `valuation_te_2026.csv`, `valuation_qb_2026.csv`,
`market_prior_te.csv`, `market_prior_qb.csv`, `vorp_all_positions.csv`, rerunnable scripts.

---

## §P Deeper universe — pre-registered 2026-08-11, before any refit

**Motivation.** The board is the top 30 WR and top 30 RB by ADP, which stops at ADP 55.5 and 72.2.
The owner drafts through round 14 (his picks run to overall 136), so more than half his roster is
made outside the modelled universe — and the two players he has raised most recently, Quentin
Johnston (ADP 95.2) and Marvin Harrison Jr. (63.9), both sit outside it. Late-round WR upside and
RB handcuffs are where the remaining roster decisions are, and they are currently unmodelled.

**§P1 Universe.** Extend to **top 60 WR and top 50 RB** by the 2026-08-09 ADP board (covering
roughly ADP ≤ 130), plus any RB identified as a handcuff under §P3 regardless of ADP.

**§P2 Refit, not extrapolate.** The §6.1/§G3 isotonic curves were fitted on top-30 panels and must
not be evaluated outside their fitted support. Refit m(·) and τ²(tier) on the **full historical
board panel** — the 2015–2024 FFC boards already carry ~57–68 WR and ~53–65 RB per year, so the
wider universe is supported by real data rather than extrapolation. Report how much the top-30
region of the curve moves when the tail is added; if it moves materially, the existing board is
restated and the change documented, since a curve is a fit and adding data changes it.
- Pre-specified expectation: the isotonic fit will have **more, flatter steps in the tail**, where
  outcomes are noisier and ADP less informative. Deep-tail values should be read as tier
  membership, not as rankings.
- **Thin-data discipline unchanged** (§G4): players with no NFL rows take the pure market arm at
  B = 1 and are flagged. The owner has stated he will generally not draft rookies, so rookie rows
  exist for completeness and are labelled, not tuned.

**§P3 Handcuff identification** — descriptive, no new model. For each 2026 team, rank RBs by
projected role using 2025 carries and snap share plus current depth; label the primary backup to
each of the top-30 board RBs as that back's handcuff, and report the **conditional** value: what
the handcuff's market price implies if the starter's touches transfer. Report the historical
transfer rate honestly — how much of a lead back's share the primary backup actually absorbed on
injury, 2015–2024 — rather than assuming full inheritance. This is the number that decides whether
a handcuff is worth a pick and it is estimable from the panel we already have.

**§P4 No new edge tests.** §P widens the universe; it does not test any hypothesis. The closed FDR
families stay closed. Adoption rules are unchanged: WR keeps arm (ii), RB stays market-anchored per
§G6 unless the wider panel changes the LOSO verdict — which must be re-run, since the RB LOSO was
powered on 300 player-seasons and the wider panel roughly doubles it. **If the RB data arm now
beats market-only on the wider panel, that is a finding and the honesty clause cuts both ways.**

Outputs: `results/valuation_wr60_2026.csv`, `valuation_rb50_2026.csv`, `handcuff_table_2026.csv`,
`results/sectionP_notes.md`, rerunnable scripts.

# WR Preseason Valuation Model

> **Scope flag: `CASHMODEL/` is NOT part of this project.** It is a separate, independently
> tracked repo (game-level win-probability model) that happens to live in this directory. It
> shares no data, no code and no conclusions with the fantasy valuation model. Ignore it for
> anything in this file, `REPORT.md`, `PROCESS.md` or the `EDA_PLAN*.md` series.

**Start here:** `REPORT.md` — the complete self-contained report (notation, intuition,
derivations, models, tests, all numbers, figures, final 2026 board). `PROCESS.md` — narrative
log of each step. `EDA_PLAN.md` — the pre-registered protocol. Never tune toward expected
results or named players; surprises are findings.

Goal: build our own preseason value model for fantasy receivers (PPR), extracting signal
beyond blind ADP. Starts with WRs; extended to RB, TE, QB.

**The project has three strands, and the intended output is a written paper covering all three
(see REPORT.md §38):**
1. **Value the player** — variance decomposition, reliability-gated covariates, a market prior
   reverse-engineered from ADP, and an empirical-Bayes posterior whose blending weight is
   *estimated, not chosen*. Validated LOSO against the market. Eight edge tests, eight nulls.
2. **A Black-Litterman-inspired views layer** (§26) — π from the isotonic ADP→points curve, Σ
   estimated, views as (P, q, Ω) with Ω declared not fitted, posterior shift decomposed by view.
   Sits strictly downstream of a frozen board; every view is logged dated and scoreable.
3. **A behavioral draft simulator** (§R, SPECIFIED NOT BUILT — `fantasy_draft_model.md`) —
   conditional-logit/Plackett-Luce pick choice, per-manager parameters shrunk toward an
   ADP-anchored league mean, Monte Carlo forward to survival curves and VONA. Blocked on Sleeper
   draft logs. Must run the τ-persistence pre-test first.

## Modeling intent
- ADP is the market prior. Thin-data players (rookies, year-2) get shrunk toward ADP-implied
  value or a group mean (empirical Bayes / partial pooling).
- Explicit covariates of interest: per-game variance, year-to-year variance, receiver archetype
  (slot/outside/X), age, team offense environment (pass attempts, pace, QB), plus discretionary
  context. ANOVA / variance-component decompositions and regressions on these covariates.
- Recency-weight veteran careers (e.g. Adams: Raiders/Jets years matter less than Rams fit + age).
- Cross-era caveat: 16-game seasons pre-2021, COVID 2020. Work per-game, not per-season.

## Data (all pulled 2026-07-13, sources need no auth)
- `data/adp/` — FantasyFootballCalculator PPR ADP, 12-team, 2026
  (`https://fantasyfootballcalculator.com/api/v1/adp/ppr?teams=12&year=2026`).
  `wr_top30_adp_2026.csv` is the modeling universe.
- `data/players/weekly_raw/` — full nflverse weekly player stats, seasons 2014–2025 (raw cache,
  every player; ~7MB/season). From nflverse-data release `stats_player`.
- `data/players/wr_top30_weekly.csv` — full-career game logs for the 30 WRs (2,151 rows, 148 cols:
  targets, receptions, yards, TDs, air yards, YAC, target_share, air_yards_share, WOPR, RACR,
  EPA, fantasy_points_ppr...). Also split per player in `data/players/by_player/`.
- `data/teams/stats_team_week_{2014..2025}.csv` — league-wide team weekly stats (attempts,
  passing/rushing yards, EPA, CPOE, pace inputs), all 32 teams, every season in window.
- `data/meta/wr_top30_meta.csv` — birth date, draft capital, rookie season, NGS position per WR.
  Joined on `gsis_id`; all 30/30 matched.
- Re-pull / extend via `python3 scripts/fetch_data.py` (caches raw files, won't re-download).

## Agents (.claude/agents/)
- `quant-researcher` — executes the EDA/modeling protocol (EDA_PLAN.md); stubborn about
  explaining anomalies, strict about validation/selection protocol, never tunes to expectations
- `player-context-researcher` — offseason moves, depth charts, archetypes, QB situations

(Data pulls don't need an agent — `scripts/fetch_data.py` handles it.)

## Conventions
- Join key across tables: `gsis_id` (fall back to normalized names, report mismatches).
- Never overwrite raw pulls; derived tables live separately.
- Target variable: PPR points per game (`fantasy_points_ppr`).

## Documentation rule (binding)
REPORT.md is a **rolling derivation** of how the model was arrived at, not a results dump.
Whenever a component is added that changes how players are valued or ranked — a new arm, the
Black–Litterman views layer, schedule strength, anything — its full derivation is added to
REPORT.md at the same time, in the same style as the existing sections:

- **notation defined explicitly** before it is used, in the symbol table;
- **the intuition stated in plain language** — what the thing is doing and why, before the algebra;
- **the derivation shown**, not summarized or asserted;
- **the design decisions justified** — why this estimator and not the obvious alternative;
- **nothing left implicit.** If a step "follows", show why it follows.

The reader is a strong stats major who wants to follow the reasoning end to end, not to
reverse-engineer it from formulas. Density is not rigor. If a passage can only be understood by
someone who already knows what it says, it is not finished. This applies to components that are
*tested and rejected* too — a null with its derivation is part of how the model was arrived at.

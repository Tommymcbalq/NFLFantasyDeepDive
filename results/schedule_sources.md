# Schedule strength — source scout and feasibility (§I-style)

Scouted 2026-08-09. **No modelling, no edge test in this document.** Follows the §I1/§I2
template: establish whether the question can be *asked properly*, then apply the pre-registered
go/no-go. Every number below was either fetched from a named URL on 2026-08-09 or computed here
from a named local file. Nothing is asserted from a site's self-description.

---

## Verdict against the §I2 go/no-go line

> ≥ 8 historical seasons of a consistently-constructible, preseason-knowable measure ⇒ an edge
> test may run. Fewer ⇒ views-layer only.

**GO — comfortably, and on four independent measure families, not one.**

| measure | seasons built | teams/season | missing | preseason-knowable |
|---|---|---|---|---|
| `sos_vegas` — mean opponent preseason win total | 2015–2026 (12) | 32 | 0 | yes |
| `sos_prior_wpct` — mean opponent prior-season win % | 2015–2026 (12) | 32 | 0 | yes |
| `sos_wr_fpa` — mean opponent prior-season PPR allowed to WRs | 2015–2026 (12) | 32 | 0 | yes |
| `sos_rb_fpa` — same, RBs | 2015–2026 (12) | 32 | 0 | yes |

Plus a weeks-15–17 (fantasy-playoff) variant of each, also 12 × 32, also complete. The 2015–2024
LOSO window is fully covered, with 2025 and 2026 available on the same construction.

This is a *better* position than §I ended in. §I passed the gate on one series (win totals) and
failed on two (point totals, player props). Here nothing fails, because the binding input —
the schedule grid — is a published fact months before the season, and every quality weight
applied to it is either a preseason market number we already hold or a prior-season realized
number.

**The honest prior is unchanged and should be restated before anything is fitted.** §I found team
quality is worth +0.251 PPG per win and that ADP already charges +0.194 of it (77 %, p = .0085),
leaving an insignificant residual. Schedule strength is the same channel viewed through the
opponent rather than the team, it is published in May and discussed continuously through August,
and the drafters who set ADP read the same rankings we just scraped. A null is the expected
result. Passing the sourcing gate means only that the test can be run — see the effect-size
ceiling in §4, which independently suggests there is very little room for a null to be wrong.

---

## 1. What was built, and why it is not look-ahead

`scripts/fetch_sos.py` and `scripts/fetch_sos_positional.py` →
**`data/schedule/sos_history_2015_2026.csv`** (384 rows = 12 seasons × 32 teams, 19 columns).

Inputs, all of which existed before the relevant Week 1:

- **Schedule grid** — `data/teams/games_nflverse_20260809.csv`, `game_type == "REG"`, using
  **only** `season / week / home_team / away_team`. `spread_line` and `total_line` are **barred**,
  exactly as in §I: they are per-game in-season lines, and any season aggregate of them is a
  quantity no August drafter could see.
- **Preseason win totals** — `data/vegas/team_win_totals_2015_2025_covers.csv` (closing, "as of"
  stamps in the first ~10 days of September) for 2015–2025, and the DraftKings column of
  `data/vegas/team_totals_2026.csv` (2026-08-08) for 2026. Both are §I artefacts already vetted.
- **Prior-season realized records** — recomputed from the grid's own scores, then **lagged one
  season**, so season *t*'s feature uses season *t−1*'s finished results.
- **Prior-season positional defence** — `data/players/weekly_raw/stats_player_week_{2014..2025}.csv`,
  PPR points scored by all WRs (resp. RBs) against each defence, per game, averaged, then
  **lagged one season** and mapped onto season *t*'s opponents.

The look-ahead trap is avoided structurally: the only within-season object used is the opponent
*identity*, which is public in May. Every weight attached to an opponent is dated strictly before
kickoff. This is the distinction that killed point totals in §I and it is respected here.

Two join hazards, both found and fixed rather than assumed away:

- Covers uses full franchise names including relocation-era ones (St Louis Rams, San Diego
  Chargers, Oakland Raiders, three Washington names); the grid uses era-correct abbreviations.
  Resolved against the set of abbreviations actually present in that season's grid, with an
  assertion rather than a silent fallback.
- **nflverse player stats normalise franchises to their *current* abbreviation for all seasons**
  (2014 files contain `LA`, `LAC`, `LV`), while the schedule grid uses the era-correct one
  (`STL`, `SD`, `OAK`). Left unhandled this silently dropped 128 opponent-games concentrated in
  2015–2019 — i.e. it would have quietly corrupted the early half of the LOSO window. Now
  `STL→LA`, `SD→LAC`, `OAK→LV` on the opponent key; missing count is **0** for every season.

### External validation of the build

Two independent checks, both strong, neither tuned:

- `sos_prior_wpct` for 2026 vs **CBS Sports'** published opponent-win-% table: **Spearman = 1.00**
  across all 32 teams. Exact rank replication of an independently computed public number.
- `sos_wr_fpa` for 2026 vs a **LeagueStation** API recompute done independently in full PPR:
  **Pearson 0.967, Spearman 0.943**, mean absolute difference 1.07 PPR points/game (season-long);
  **Spearman 0.98** on the weeks-15–17 variant. Residual gap is expected — LeagueStation's season
  column spans W1–18 and ours W1–17.
- `sos_vegas` for 2026 vs **Sharp Football's** published rank order: **Spearman 0.88**. Sharp
  states the same construction (opponent 2026 Vegas win totals); the gap is book choice (our DK
  column vs their undisclosed source) and week window.

---

## 2. The 2026 sources — and the fact that they are not the same statistic

**`data/schedule/sos_2026.csv`** — 479 rows, long format, one row per (team, source, metric), with
`source`, `metric`, `basis_year`, `definition` and `preseason_knowable` on **every row**. Long
format is deliberate: these measures are incompatible and stacking them keeps that visible.
Averaging them would be a category error — see the correlation matrix below for how badly.

| source | metric | what it actually measures | basis year | n |
|---|---|---|---|---|
| computed (ours) | `mean_opp_preseason_win_total` | mean opponent 2026 Vegas win total; higher = harder | 2026 DK lines, 2026-08-08 | 32 |
| computed (ours) | `mean_opp_preseason_win_total_w15_17` | same, weeks 15–17 only | 2026 DK lines | 32 |
| computed (ours) | `mean_opp_prior_season_win_pct` | mean opponent 2025 realized win % | 2025 records | 32 |
| Sharp Football | `sos_rank` | rank only (1 = easiest), from opponent 2026 Vegas win totals; **no numeric value published** | 2026 win totals | 32 |
| CBS Sports (2026-05-12) | `opp_win_pct_2025` | combined 2025 win % of the 17 opponents | 2025 records | 32 |
| LeagueStation (recomputed full PPR) | `wr_ppr_pts_allowed_season` | mean full-PPR points the 2026 opponents allowed to WRs in 2025; **higher = easier** | 2025 defensive FPA | 32 |
| LeagueStation | `wr_ppr_pts_allowed_w15_17` | same, weeks 15–17 | 2025 | 32 |
| LeagueStation | `def_wr_ppr_allowed_pg_2025`, `def_rb_..._2025` | the *ingredient* — each defence's own 2025 allowance. Not an SOS. | 2025 | 32 each |
| FantasyNerds | `fn_sos_score_wr`, `fn_sos_score_rb` | undisclosed proprietary score, higher = easier | **unstated** | 31 / 32 |
| Sharp Football (fantasy) | `sharp_{season,playoff}_{passing,rushing}_rank` | rank only, opponent efficiency "with game script removed"; passing/rushing, not WR/RB | 2025 | 32 each |

Two data-quality flags carried on the rows themselves: **FantasyNerds is missing Miami entirely**
(31 of 32 teams) and publishes no methodology or basis year, so it is marked
`preseason_knowable = UNCLEAR`; **Sharp publishes ranks only**, no numeric values, so it cannot
support a magnitude test.

### Sources checked and rejected

- **FantasyPros** (`/nfl/strength-of-schedule.php?position=WR`) — the data is embedded in
  `window.FP.reportConfig` but the server renders only **8 of 32 rows** (ARI–CLE) behind a
  registration fence. Browser/Googlebot UAs, session cookies, `&export=xls` and
  `api.fantasypros.com/v2/json/nfl/2026/sos` (403, "Missing Authentication Token") all still
  returned 8 rows. Metric is 1–5 stars from opponent fantasy points allowed to the position. Not
  usable without an account.
- **RotoWire** — QB free, **WR/RB subscriber-only**. Its weeks-15–17 UI filter ignores GET
  params, so even the free position could not be windowed.
- **DraftSharks**, **RosterWatch**, **Establish The Run** — paywalled, no numbers rendered.
  RosterWatch is worth noting as the only source with an explicit fantasy-playoff window built
  from *projected* (not prior-year) matchup grades.
- **FFToday** — its "current" positional SOS grid **still shows the 2025 schedule as of
  2026-08-09** (verified: it lists ARI Week 1 @ NO, which is 2025; real 2026 Week 1 is ARI @ LAC).
  Do not scrape it for 2026. Its *historical* value is real but redundant for us — see §3.
- **FFToolbox, FantasyAlarm, Yahoo** — tables did not render (JS/gated); no numbers taken.
- **ESPN** — the article renders and states the method ("last season's results") and the playoff
  finding narratively (easiest W15–17: WAS, NO, ARI; hardest: PHI, SEA, SF), but two extraction
  passes disagreed with each other, so **no ESPN per-team numbers are recorded**. Treated as
  unverified rather than guessed.
- **PFF, Sumer Sports, nfelo** — no 2026 *positional fantasy* SOS found. nfelo/Sumer publish
  power-rating-based opponent strength, which is a team-quality construct already covered by
  `sos_vegas`, not a positional one.

### Where they disagree — materially

Spearman rank correlations, all oriented **higher = harder**, n = 32 (FantasyNerds n = 31):

| | mine (vegas) | mine (W15–17) | prior wpct | CBS | Sharp | LS WR | LS WR W15–17 | FN WR |
|---|---|---|---|---|---|---|---|---|
| **mine (vegas)** | 1.00 | 0.12 | 0.52 | 0.52 | **0.88** | 0.08 | −0.12 | 0.25 |
| **mine (W15–17)** | 0.12 | 1.00 | 0.15 | 0.15 | 0.17 | 0.03 | 0.27 | 0.14 |
| **prior wpct** | 0.52 | 0.15 | 1.00 | **1.00** | 0.50 | 0.59 | 0.34 | 0.16 |
| **Sharp** | 0.88 | 0.17 | 0.50 | 0.50 | 1.00 | −0.00 | −0.09 | 0.10 |
| **LS WR** | 0.08 | 0.03 | 0.59 | 0.59 | −0.00 | 1.00 | 0.49 | 0.16 |

Three disagreements matter, and each is a reason not to blend:

1. **Market-implied vs prior-year team quality correlate only 0.52.** They are different
   statistics. Biggest 2026 splits: **KC** ranks 8th-easiest on opponent win totals but
   28th-hardest on 2025 records; **HOU** is the mirror (27th vs 7th); likewise ATL, DAL, NE, GB —
   all ≥ 15 rank places apart. This is the market disagreeing with last year's standings about
   who is good, which is precisely the information a Vegas-based measure adds.
2. **Positional SOS is roughly orthogonal to team-quality SOS** (LS WR vs Sharp = −0.00; vs our
   Vegas measure = 0.08). Historically the same: within-season Spearman of `sos_wr_fpa` against
   `sos_vegas` averages about **−0.10** across 2015–2024 and ranges from −0.60 to +0.43, flipping
   sign year to year. Good teams are not reliably good against WRs. If schedule strength is ever
   tested, the positional variant is a *separate hypothesis*, not a refinement of the team one.
3. **Season-long and playoff-weeks SOS are nearly unrelated** (0.12 for our Vegas measure, 0.27
   for LeagueStation's WR measure). The cleanest illustration in the 2026 data: **PIT is 31st for
   the season on WR-FPA SOS but 7th over weeks 15–17; PHI is 1st for the season and 29th in the
   playoffs.** Any claim about "an easy schedule" is meaningless without stating the window.
4. **WR and RB defensive quality are largely unrelated.** In 2025, DEN allowed the fewest RB
   points (17.18/g, 32nd) but was mid-pack against WRs (27.32/g, 27th-softest); CIN was the
   softest RB defence (29.20/g) and the 2nd-toughest WR defence (26.18/g). A single "defence
   rating" would misprice both.

---

## 3. Historical positional SOS — rebuilt, not sourced

**No public source publishes a preseason positional-SOS archive.** Every SOS page found is
current-year only. FFToday's `&Season=YYYY` positional grid does return historical values, but
they are computed from *that same season's realized* allowances — in-sample, look-ahead, and
therefore exactly the disqualifying property §I identified. It is unusable as a preseason
predictor no matter how convenient.

This is a non-issue, because the ingredient is already on disk. `data/players/weekly_raw/` holds
nflverse weekly player stats back to 1999. Summing PPR by (defence, week) for `position == "WR"`,
averaging per game, lagging one season and mapping onto the next season's grid reproduces the
LeagueStation construction — validated above at Spearman 0.94 on 2026 — for **every season
2015–2026 with 32 teams and zero gaps**. The same one-line change gives RB.

**A caveat that should be recorded before any fitting, because it bears on the prior.** The whole
premise of positional SOS is that defensive quality persists year to year. Measured directly on
our own data, WR PPR allowed per game correlates across consecutive seasons at:

| 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|
| .31 | .15 | .46 | .41 | .21 | .40 | .24 | .30 | .20 | **−.05** | **−.07** |

Mean ≈ 0.25, and the two most recent transitions are **negative**. So a prior-year positional SOS
carries roughly a quarter of its apparent signal into the season it is meant to predict, and
recently rather less. Sharp, RosterWatch and Yahoo all volunteer the same point about their own
products. This is a reason to expect a null independent of the ADP-pricing argument, and it must
not be treated as a reason to reweight or reconstruct the feature until it looks better.

---

## 4. Effect-size ceiling — computed before any test, so it cannot be rationalised after

Mean within-season dispersion of each feature, 2015–2024:

| feature | mean within-season SD | mean within-season range |
|---|---|---|
| `sos_vegas` (opponent win totals) | **0.245 wins** | 1.02 wins |
| `sos_prior_wpct` | 0.034 | 0.123 |
| `sos_vegas_w15_17` | 0.988 wins | 3.90 |
| `sos_wr_fpa` (PPR/g allowed to WRs) | 0.977 | 3.89 |
| `sos_wr_fpa_w15_17` | 2.211 | 9.24 |

Schedules are close to zero-sum within a season, and the numbers show it. A full-season schedule
separates the luckiest from the unluckiest team by **about one win of average opponent quality**,
one standard deviation being a quarter of a win.

Carrying §I's estimate across — team quality is worth +0.251 PPG per win — a 1-SD full-season
schedule swing is worth at most **≈ 0.06 PPG**, and that is a generous ceiling in two ways: the
0.251 coefficient is for a receiver's *own* team quality, whereas schedule acts through the
opponent's *defensive* quality, a fraction of team quality; and §I already showed ADP charges
77 % of the team-quality effect. The residual room here is a small fraction of a small number.

The positional feature has more nominal spread (≈ 1.0 PPR/g per SD at the whole-WR-room level),
but it must be discounted by the ≈ 0.25 year-over-year persistence above and then split across a
WR room, which lands it in the same neighbourhood — order 0.1 PPG for a single receiver.

One further note that constrains any test design: `sos_vegas` has a year-over-year
team-level autocorrelation of only **0.153**. A team's schedule difficulty is close to
independent across seasons, which is good for a panel test (little serial dependence to model)
but also means there is no persistent "easy-schedule team" effect to be arbitraged.

The weeks-15–17 variants are the exception worth flagging: their dispersion is **4× the
full-season dispersion** (0.99 vs 0.245 wins; 2.21 vs 0.98 PPR/g). Three games do not average out.
If any part of schedule strength has room to matter, the arithmetic says it is the playoff window
— and that is also the part least likely to be priced into a season-long ADP, since ADP values a
full season. That is the most promising sub-hypothesis, and it is stated here *before* any test.

---

## 5. What a conditional test may and may not use

**May use** (2015–2024, 32 teams, complete, preseason-knowable):
`sos_vegas`, `sos_prior_wpct`, `sos_wr_fpa`, `sos_rb_fpa`, and the `_w15_17` and `_w1_14` variants
of each, plus the within-season z-scored forms already in the file. Test under the full §I3
protocol — FDR across the family, plus holdout. Note the family is now several correlated
hypotheses; the multiplicity correction must count them all, including the playoff-window
variants, and the family must be declared before the first fit.

**May not use:**
- Any SOS built from the season's own realized results (FFToday's historical grids; any
  "opponent's final record this year" rating). Look-ahead.
- Any season aggregate of `spread_line` / `total_line` from `games.csv`. Barred in §I, barred here.
- FantasyNerds (no methodology, no basis year, MIA missing) and Sharp (ranks only) as *fitted*
  features. They are 2026 corroboration only.
- The 2026 values to fit anything. As in §I, the 2026 win-total source (DraftKings, VegasInsider
  board) differs from the historical source (Covers), so the two are not one continuous series.

**Restated prior:** ADP is set by drafters reading the same May-published schedule rankings we
just scraped. §I's finding — 77 % of team quality already in the price, residual insignificant —
plus the ~0.06 PPG ceiling and the ~0.25 defensive persistence all point the same way. **A null
is the anticipated result and is publishable as such.** The one place the arithmetic leaves room
is the fantasy-playoff window, and only because three games do not average out.

---

## Artefacts written

| path | contents |
|---|---|
| `/Users/thomasmcnamee/NFL/data/schedule/sos_2026.csv` | 479 rows. 2026 SOS from all retrievable sources, long format, per-row source / metric / definition / basis year / preseason-knowable flag. |
| `/Users/thomasmcnamee/NFL/data/schedule/sos_history_2015_2026.csv` | 384 rows (12 × 32). The backtestable panel: four measure families × three week windows, plus within-season z-scores. |
| `/Users/thomasmcnamee/NFL/scripts/fetch_sos.py` | Builds the win-total and prior-record SOS history from the grid. |
| `/Users/thomasmcnamee/NFL/scripts/fetch_sos_positional.py` | Adds the WR/RB prior-year-FPA SOS columns from `data/players/weekly_raw/`. |
| `/Users/thomasmcnamee/NFL/scripts/build_sos_2026.py` | Assembles `sos_2026.csv` from the scraped 2026 sources; every literal is inline and attributed. |

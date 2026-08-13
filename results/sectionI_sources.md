# §I1 / §I4 — Vegas market context: source scout and 2026 pull

Scouted 2026-08-09. **No modelling in this document.** Every claim below was verified by an
actual fetch; retrieved artefacts are named. Nothing is asserted from a site's self-description.

---

## Verdict against the pre-registered §I2 go/no-go

> ≥ 8 historical seasons of a consistent team-level number ⇒ §I3 runs.

**GO — for team season win totals only.**
Eleven consecutive seasons (2015–2025), 32 teams each, 352 rows, retrieved and saved. Two
independent sources agree on the overlap to within 0.026 wins mean absolute difference.

**NO-GO — for team season point totals and for player season props.** Neither is obtainable for
2015–2024 without paid authentication, because in the point-total case the market largely does not
exist as an archived season-long future, and in the player-prop case the archive genuinely does
not go back that far at any price we can reach for free. §I4 therefore records 2026 point context
as an explicitly labelled **unbacktested proxy**.

---

## 1. Team season win totals — GO (11 seasons)

### 1a. Primary: Covers.com SportsOddsHistory archive

- **URL pattern:** `https://www.covers.com/sportsoddshistory/nfl-win/?y={season}&sa=nfl&t=win`
  (`sportsoddshistory.com` 301-redirects here; the old domain was absorbed by Covers.)
- **Seasons actually retrieved:** 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025
  — **32 teams every season, no gaps**. This is 10/10 of the 2015–2024 LOSO window plus 2025.
- **Opening or closing:** effectively **closing**. Each page carries its own "As of" stamp, and the
  stamps are season-specific and land in the first ~10 days of September, i.e. at or immediately
  before Week 1, which is when the season win-total market closes:

  | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | Sep 7 | Sep 5 | Sep 4 | Sep 3 | Sep 5 | Sep 10 | Sep 9 | Sep 8 | *(no stamp)* | Sep 5 | Sep 4 |

  The differing dates confirm these are per-season captures, not one number templated across years.
  Caveat to carry: 2023 has no visible stamp; treat that season's provenance as one notch weaker.
- **Book:** **not disclosed.** The page names no sportsbook and gives no consensus methodology.
  This is the single real weakness of the source. Mitigated by the cross-check in 1c.
- **Fields captured:** team, win total, over odds, under odds, week the bet settled, actual wins,
  over/under result. The two-sided odds are present, so a de-vigged fair line is computable
  downstream rather than having to treat the raw half-win number as the market's point estimate.
- **Programmatic access:** plain `GET`, no auth, no JS execution needed — the table is server-
  rendered into the HTML. Parsed with a regex over the single `<table>`. One request per season.
- **ToS / robots:** `covers.com/robots.txt` (fetched 2026-08-09, HTTP 200) contains **no
  `Disallow` covering `/sportsoddshistory/`**; the disallow list targets forum, redirect and
  admin endpoints. So the crawl is robots-permitted. There is no open-data licence, though — this
  is a scrape of a commercial site. Fine for internal research; **do not redistribute the raw
  table** or republish it as a dataset.
- **Reproducibility:** high but not guaranteed. Single-vendor dependency on a page layout that
  could change. The pull is cached to disk so the analysis is reproducible even if the page moves.

### 1b. Cross-check source: nflverse/nfldata `win_totals.csv`

- **URL:** `https://raw.githubusercontent.com/nflverse/nfldata/master/data/win_totals.csv`
  (fetched, HTTP 200, 12,811 bytes).
- **Seasons covered: 2003–2020 only, 32 teams each (576 rows). It stops at 2020 and has not been
  extended.** Verified by counting rows per season, not by reading the docs.
- Columns `season, team, line, over_odds, under_odds`. Opening/closing status is not documented.
- **On its own this is a NO-GO**: it supplies only 6 of the 10 seasons in the 2015–2024 window.
  Its value here is as an independent second opinion, not as the primary.
- Repo metadata declares **no SPDX licence** (GitHub API `license: null`). Publicly distributed and
  conventionally freely used, but there is no explicit grant — worth knowing before redistribution.
- Note: `nfldata/DATASETS.md` does **not** list `win_totals.csv` at all. The file exists anyway.
  Another reason to verify by fetching rather than by reading documentation.

### 1c. Agreement between the two independent sources (the real evidence for "consistent")

Joined on (season, team) over the 2015–2020 overlap, with full franchise-relocation name mapping
(OAK/LV, SD/LAC, STL/LA, and the three Washington names):

- 191 matched pairs (of 192; the one miss is 2015 St. Louis, a name-string artefact, not a
  missing number).
- **181 / 191 exact agreement (94.8 %).**
- Mean absolute difference **0.026 wins**; maximum difference **0.5 wins**.

Ten half-win disagreements across six seasons is exactly what you would expect from two captures of
the same market taken at different moments or at different books. It is strong evidence that both
are measuring one real, stable quantity, and it is the basis for calling the series "consistent" in
the §I2 sense. It does **not** resolve which book, and it does not fully pin down open-vs-close.

### 1d. Sources checked and rejected

| Source | Why rejected |
|---|---|
| **the-odds-api.com historical tier** | Historical snapshots begin **2020-06-06**; player props and alternate lines only from **2023-05-03**. Explicitly **paid plans only** — the free key does not reach the historical endpoints at all, and historical queries cost 10 credits per region per market. Even if paid, the start date alone fails the 2015–2024 requirement. Clean negative. |
| **nfeloapp.com win totals** | Claims coverage back to 2003, which would be excellent, but the page discloses **neither book nor open/close**, offers **no CSV, API or export**, and renders client-side. Not reproducibly pullable. |
| **sportsbettingdime.com** | Self-described coverage from 2018 only — 7 seasons of the window, **below the threshold** on its own. |
| **Pro-Football-Reference** | Carries realized results and no betting lines. Not an odds source. |
| **sportingcharts.com "team totals"** | Despite the name, these are *realized* points-for/points-against averages, not posted futures. Not market data. |
| **Kaggle / assorted GitHub NFL repos** (hvpkod/NFL-Data, nflscrapR-data, fivethirtyeight/nfl-elo-game, ali-ce/datasets) | Searched; none carries preseason season-long win totals. Play-by-play, rosters, Elo and results only. |

---

## 2. Team season point totals — NO-GO for history

Two distinct things get called "team season point total", and only one of them exists as a market:

1. **A posted season-long over/under on points a team scores.** DraftKings does put this up in some
   years (Sharp Football's 2025-season article lists all 32 DK numbers, e.g. Bills 457.5, Browns
   325.5, published 2025-07-21, DraftKings sourced). But it is a **sporadically offered novelty
   market, not a staple**, and critically **no archive of it for 2015–2024 was found anywhere**.
   Coverage found: essentially 2025 via one secondary article. **0–1 of the 10 required seasons.**

2. **An *implied* team total derived arithmetically from game spreads and totals.** This is what
   the fantasy tools (Sharp Football, First Down Studio, RotoWire) actually publish. It is a
   derived quantity, not a posted line.

For (2) the history *is* obtainable and is already essentially in hand:
`nflverse/nfldata data/games.csv` (fetched, HTTP 200, 7,548 rows) carries **`spread_line` and
`total_line` on 100 % of games for every season 2013–2025** — verified by counting non-null values
per season: 267/267 for 2013–2019, 269/269 in 2020, and 284–285/284–285 for 2021–2025.

**But this cannot serve §I3 as a preseason covariate, and I want to be blunt about why.** Those are
per-game lines set during the season with knowledge of injuries, results and form. Summing them
into a season figure builds a look-ahead quantity that no drafter could have seen in August. Using
it as a preseason feature would leak the season into the prediction and would inflate any edge
test. It is available; it is not usable for this purpose. It would be legitimate only for a
different, explicitly in-season question.

**Verdict: NO-GO.** No posted season point-total series exists for 2015–2024, and the only
back-fillable substitute is contaminated by look-ahead.

---

## 3. Player season props (receiving yards, rushing yards, games played) — NO-GO, decisively

The suspicion in the brief is correct, and the negative is clean.

- **the-odds-api**, the most likely commercial route: historical player-prop markets begin
  **2023-05-03**. That is **zero** of the 2015–2024 seasons, and it is paid-only regardless.
- No public archive, repo, or Kaggle dataset of preseason season-long player props for the
  2010s surfaced. Searches returned realized player statistics (PFR, NFL.com, TeamRankings)
  mislabelled as "props" by content sites — the actual posted lines are not archived.
- Structural reason, worth recording: season-long player props were a thin, low-limit sideshow
  before the post-2018 US legalisation wave, and the books that offered them did not publish
  archives. The data was largely never retained in public form. This is not a paywall problem
  that money would solve for the mid-2010s; the record does not exist.

**Verdict: NO-GO.** Player props cannot enter any backtested arm. If they are ever wanted, they
are a 2023-onward, paid, forward-looking input only.

---

## §I4 — 2026 team totals (recorded regardless of the above)

Written to **`/Users/thomasmcnamee/NFL/data/vegas/team_totals_2026.csv`** — 32 rows, one per team,
with a per-column source string on every row.

Columns: `season, team, win_total_consensus, win_total_dk, dk_over_odds, win_total_src,
implied_ppg, implied_season_points, points_src, points_is_proxy`.

**Win totals — VegasInsider multi-book board** (`vegasinsider.com/nfl/odds/win-totals/`), page
last updated **2026-08-08**, retrieved 2026-08-09. Shows BetMGM, DraftKings, Caesars and Rivers
side by side. All 32 teams present. Two fields are stored deliberately:

- `win_total_consensus` — VegasInsider's consensus figure. Six teams are **not** unanimous across
  books and are stored as a range: JAX 8.5–9.5, CLE 5.5–6.5, CAR 6.5–7.5, LV 5.5–6.5, ARI 3.5–4.5,
  MIA 3.5–4.5. Stored as text so the disagreement is visible rather than silently averaged away.
- `win_total_dk` + `dk_over_odds` — the DraftKings line and its over price. **Prefer this pair for
  any modelling**: it is one book, internally consistent across all 32 teams, and the price lets
  you de-vig. The raw half-win number is misleading on its own where the juice is heavy — e.g.
  LA and BAL both sit at DK 10.5 but priced o10.5 −210 and −150 respectively, so their true
  expectations differ by well over half a win.

Top of the 2026 board (DK line, over price): BAL 10.5 (−150), LA 10.5 (−210), BUF 10.5 (−120),
DET 10.5 (−115), SEA 10.5 (−115), KC 10.5 (+115), PHI 10.5 (+115), SF 10.5 (+125), GB 10.5 (+140),
LAC 10.5 (+130), NE 10.5 (+115). Bottom: ARI 4.5, MIA 4.5, NYJ 5.5, LV 5.5, ATL 6.5, CAR 6.5,
CLE 6.5, TEN 6.5.

**Point totals — PROXY, flagged as such** (`points_is_proxy = TRUE` on every row). No posted 2026
season point-total future was found for all 32 teams. Used instead: **First Down Studio season
implied totals** (`firstdown.studio/implied-totals/season`), "powered by Vegas lines", updated
**2026-08-08**, retrieved 2026-08-09 — season-long implied points per game for all 32 teams,
stored as `implied_ppg`, with `implied_season_points = implied_ppg × 17` as a convenience.

Range: LA 26.6 ppg at the top, through BUF/DET 26.1, CIN 26.0, BAL 25.9, DAL 25.7, down to
CLE 18.7, NYJ 18.5, ARI 18.4. Corroborated independently by RotoWire's 2026 implied-scoring rank
order (Lions, Bengals, Ravens, Rams, Cowboys at the top; Browns, Cardinals, Jets at the bottom) —
same teams at both ends, minor rank permutation, consistent with two derivations off slightly
different line snapshots.

Two caveats to carry into §J. First, this is derived from **lookahead** full-season game lines, so
it is a genuine August-available quantity — unlike the in-season `total_line` discussed in §2 —
but it is a third party's arithmetic off an undisclosed line set, not a posted market. Second, it
is **unbacktested by construction**, since §2 established there is no historical series of it. Per
§I2 it may inform the views layer only, and must be labelled as such in the report.

Note: `covers.com/sportsoddshistory/nfl-win/?y=2026` returns an **empty table** — the archive is
populated retrospectively, so 2026 will only become available there after the season. The current-
year and historical pulls are necessarily different sources; do not expect them to be identical
series.

---

## Artefacts written

| Path | Contents |
|---|---|
| `/Users/thomasmcnamee/NFL/data/vegas/team_totals_2026.csv` | §I4 deliverable. 32 teams × 2026 win total (consensus + DK + price) and implied points, per-row source strings. |
| `/Users/thomasmcnamee/NFL/data/vegas/team_win_totals_2015_2025_covers.csv` | 352 rows. Primary history: season, team, win_total, over, under, actual_wins, asof. |
| `/Users/thomasmcnamee/NFL/data/vegas/team_win_totals_2003_2020_nflverse.csv` | 576 rows. Independent cross-check, 2003–2020. |
| `/Users/thomasmcnamee/NFL/scripts/fetch_vegas.py` | Re-runs the Covers historical pull from scratch. |

---

## What §I3 may and may not use

- **May use:** team season win total, 2015–2024, all 32 teams, closing, from the Covers archive,
  with the nflverse file as a 2015–2020 integrity check. Odds are present, so de-vig before use.
- **May not use:** any team season point total as a backtested feature; any player season prop as
  a backtested feature; any season aggregate of in-season `total_line` / `spread_line`.
- **Unchanged prior, restated:** §I3's own pre-registration says ADP is set by drafters reading the
  same win totals, so this is expected to be already priced and a null is the anticipated result.
  Passing the sourcing gate is not evidence of an edge — it only means the test can be run.

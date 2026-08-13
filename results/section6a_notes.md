# §6.1 notes — market prior curve + tier variances (2026-07-14)

## Panel

FFC PPR ADP, top-30 WRs by ADP per year, joined to realized **same-season** PPG under the
§1 inclusion rule (REG only, targets ≤ 1 excluded). **Window is 2015–2024 (10 years, 300
rows), not 2015–2025: 2025 ADP is unavailable at the source** — documented deviation from
the plan's "2015–2025".

### Name → gsis_id matching (every non-exact case reported; nothing silently dropped)
Normalization: lowercase, punctuation stripped, Jr/Sr/II/III/IV(/V) suffixes stripped.
- **300/300 matched.** Zero unmatched after three documented resolutions:
- `Hollywood Brown` (2020, 2022) → Marquise Brown: FFC nickname; one-entry alias table in
  the script (identity resolution, not a modeling choice).
- `Jordan Matthews` (2015) → 00-0031299: meta retro-tags his career position TE; matched
  via any-position fallback restricted to players active in the ADP year. The realized-PPG
  side is therefore built **without a position filter** (join is by gsis_id from the WR
  board; nflverse weekly also tags him TE in 2015, which had silently zeroed his games).
- `Charles Johnson` (2015) → `Charles D. Johnson` 00-0030113: meta middle initial; matched
  by first+last token among WRs active in 2015. (Exact-match fallback previously grabbed
  the 1994–2002 WR of the same name — caught because a 6-game 2015 season showed 0 games.)

### Fit sample (pre-registered: ≥ 4 included games)
9 of 300 rows dropped from the fit (kept in the panel with `in_fit=False` and games noted):
4 zero-game seasons (Gordon 2016, Robinson 2017, Green 2019, Thomas 2021 — suspension/
injury lost years), 5 with 1–3 games (Allen 2016, Decker 2016, Brown 2019, Sutton 2020,
Thomas 2022). All vet-tier; realized-PPG conditional on ≥4 games is the estimand, so
τ̂²(vet) **excludes total-season-loss risk** — stated as a caveat, not patched.
Fit n = 291.

## m(ADP)

- **Baseline OLS:** PPG = 22.57 − 2.315·log(ADP), HC3 se(slope) 0.273, R² = 0.225,
  n = 291. Year-FE sensitivity: slope −2.515 (se 0.280) — same story; pooled kept.
- **Headline: isotonic regression, monotone decreasing in ADP** (fit on log ADP; 18
  distinct step levels, fitted range 20.49 PPG at ADP≈1–2 down to 8.66 at ADP 75; fit-
  sample ADP range 1.2–75.0, so the 2026 board (ADP 2.8–60.1) needs no extrapolation).
  In-sample RMSE 3.32 vs OLS 3.40 (not a selection criterion; isotonic is headline by
  pre-specification). Knots saved to `results/market_prior_iso_knots.csv`.

## τ̂²(e) — residual variance around m̂_iso by tier at ADP year

| tier | n (fit) | τ̂²_iso | boot 95% CI | τ̂²_ols |
|---|---|---|---|---|
| rookie | 4 | 24.5 | (1.7, 35.1) | 28.1 |
| soph | 36 | 7.9 | (5.0, 11.0) | 8.2 |
| vet | 251 | 11.3 | (9.3, 13.2) | 11.5 |

**Ordering verdict: τ²(rookie) > τ²(soph) > τ²(vet) FAILS** — observed rookie > vet >
soph. Tested, not assumed, as the plan requires:
- Rookie > vet direction agrees with intuition but rests on **n = 4** (top-30 preseason
  WR boards almost never contain rookies: Cooper '15, Agholor '15, Harrison '24,
  Nabers '24). Bootstrap CI (1.7, 35.1) — essentially unidentified. Levene rookie-vs-vet
  p = 0.27. Mean rookie residual −2.4 PPG (market paid top-30 prices, delivery averaged
  below the curve), but with n = 4 this is descriptive only.
- Soph < vet is the surprise; Levene soph-vs-vet p = 0.39 — **not significant**, so the
  data are consistent with soph ≈ vet. Chased anyway: (i) vet τ² is inflated by
  injury-shortened realizations — vet τ² by games bucket: 4–9 games 16.0 (n=29), 10–13
  games 10.3, 14+ games 9.7; sophs on these boards almost never have short seasons (min
  9 games, mean 14.6) — survivorship into a top-30 board after year 1 selects durable
  usage; (ii) the biggest vet negative residuals are age/post-injury cliffs (2015 Dez
  −8.0, 2017 Pryor −7.7, 2021 Robinson −7.4, 2017 Nelson −7.1) — a real feature of vet
  risk, not a data problem.
- Per protocol the posterior uses τ̂²(tier) **as estimated** — no monotonicity enforced,
  no hand adjustment. Consequence worth stating: sophs get the *tightest* prior around
  the market curve.

## Files
- `results/market_prior.csv` — full 300-row panel: ADP, rank, gsis, tier, games, ppg,
  m_ols, m_iso, residuals, in_fit.
- `results/tier_variances.csv` — the τ̂² table.
- `results/market_prior_iso_knots.csv` — isotonic step function.
- Script: `scripts/07_section6a_market_prior.py`.

# §F notes — teammate coherence (2026-07-16)

Pre-registration = EDA_PLAN3.md; measurement mechanics fixed in the docstring of
`scripts/18_teammate_coherence.py` before running.

## F1 — implied target share and duo sums

Map: fold-fit OLS PPG = a + b·(TS · team att/gm) on gated WR-seasons 2014–2025
(full-sample a = −1.309, b = +1.942; x = expected targets/game). Inversion:
implied_TS = (θ* − a)/(b · entering-team prior-season att/gm). Historical θ* = the
frozen LOSO values, recomputed for all 300 board rows with script 10's machinery
(in_fit rows asserted identical to `loso_predictions.csv`, max |diff| 3.6e−15).

Historical benchmark (realized top-2 WR TS sums, 384 team-seasons 2014–2025):
mean 0.366, p90 **0.456**, p95 **0.476**, max 0.565.

**Headline F1 finding: the implied measurement runs hot by construction.** 61.9% of
the 63 historical board-duo team-years have implied sums above the realized p90 —
but those same team-years *realized* a mean top-2 sum of 0.391 (65th percentile;
only 19.4% above p90). The gap is the inversion booking θ*'s efficiency component
(stars score more per target than the average gated WR) as volume. So "above p90 on
implied TS" is mostly measurement inflation, not genuine incoherence — the fair
reference for a 2026 duo is the historical *implied* duo-sum distribution, added as
a column after this chase. Caveat: 4 of the 63 "duos" are triples (NE 2019,
LA 2019, TB 2021, HOU 2024) and top the implied table; they are excluded from the
pairs-only implied reference.

## F2 — edge test (binding protocol): full null

R = resid_iso on 291 in_fit rows 2015–2024; family exactly as pre-registered
{teammate_on_board, duo implied-TS sum centered (own implied TS when solo — the
non-degenerate extension, fixed in the docstring), interaction}; season-clustered
t(9); BH-FDR q = 0.10; holdout 2015–22 → 2023–24 (`results/edge_teammate.csv`):

| term | β | t(9) | p | FDR | final |
|---|---|---|---|---|---|
| teammate_on_board | +0.39 | 0.53 | .609 | fail | **no** |
| duo_sum (centered) | −2.71 | −0.29 | .781 | fail | **no** |
| interaction | +2.05 | 0.17 | .867 | fail | **no** |

No term is anywhere near significance; for the record the full family also fails
the holdout (MSE 9.721 vs zero 9.336). Sharing a board with a teammate — even at
jointly "infeasible" implied volume — carries zero residual information. Coherent
with F1's decomposition: the apparent infeasibility is measurement, and to the
extent duo structure is real, the market prices it (Chase/Higgins-type duos have
appeared on boards every year without systematic joint disappointment).

## F3 — not run

Pre-specified decision rule (EDA_PLAN3.md): the constraint arm (viii) runs only if
F2 shows *unpriced* infeasibility (a final survivor). F2 is a full null, so **F3 is
not run** — a proportional-scaling constraint could only add noise to a residual
the market already handles.

## 2026 descriptive output (`results/teammate_coherence_2026.csv`)

Duos verified from the FFC ADP team column and confirmed identical on Sleeper
current teams (0 disagreements; the 3 movers — A.J. Brown NE, Waddle DEN, Evans SF
— create no new duos):

| team | duo | implied TS sum | pct of realized top-2 sums | pct of hist implied duo sums |
|---|---|---|---|---|
| LA | Nacua + Adams | 0.560 | 99.7 (> p95) | 94.9 |
| DET | St. Brown + J. Williams | 0.521 | 99.0 (> p95) | 83.1 |
| CIN | Chase + Higgins | 0.512 | 98.4 (> p95) | 76.3 |
| DAL | Lamb + Pickens | 0.507 | 98.4 (> p95) | 76.3 |
| CHI | Burden + Odunze | 0.426 | 81.3 | 30.5 |

Read with F1's calibration: all four flagged duos are extreme vs *realized* sums but
ordinary-to-high vs what priced board duos have always implied. Only LA (95th pct of
implied duo sums, in the company of 2019 NE / 2024 HOU triples and 2024 MIA) is
unusual even by duo standards — and F2 says even that carries no tradable signal.
Realized-space intuition: 2022–24 CIN realized 0.480–0.555 top-2 sums, so 0.512 is
attainable; LA's 0.560 implied would need a top-3 all-time realized outcome.

## Files
- `results/teammate_coherence_2026.csv`, `results/edge_teammate.csv`
- `scripts/18_teammate_coherence.py`

## Deviations from plan text
- duo_sum for solo players = own implied TS (keeps the pre-specified 3-term family
  non-degenerate; fixed in the docstring before fitting).
- R = full-fit resid_iso (round-2 B3 precedent) rather than fold residuals.
- The pairs-only implied-duo-sum percentile column was added after the F1 anomaly
  chase (descriptive context only; the pre-registered realized-distribution
  percentile is reported unchanged).
- 3 of 300 historical board rows lack team_now (round-1 known gap) → teammate = 0.

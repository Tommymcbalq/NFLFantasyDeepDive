# §G0/§E notes — Sleeper 2026 context + context-adjusted data arm (2026-07-16)

Pre-registration = EDA_PLAN3.md; operational details fixed in the docstring of
`scripts/17_context_arm.py` before running.

## §G0 — Sleeper data and 2026 derived context

Raw dump: `data/sleeper/players_nfl_2026.json` (pulled 2026-07-16, 12,200 records,
14.6 MB, cached, never overwritten). Sleeper `gsis_id` exists on 3,893 records (only
~29% of 2025 participants matched via gsis — the field is sparse for recent players,
and values carry a leading space, stripped). Matching cascade: (1) stripped gsis_id;
(2) normalized name + position (suffixes dropped); ambiguity resolved by
has-team/active preference; (3) tertiary rule added after the first match report:
unique first-3-chars first-name prefix among same-(last name, position) records.
The tertiary rule was a data-quality fix motivated by two false "departures"
(Josh/Joshua Palmer BUF, Mitchell/Mitch Tinsley CIN — CIN board-relevant); it caught
15 players; a few of its non-skill-position matches look dubious (e.g. Dane
Jackson → Dan Jackson DET) but all such cases carry zero 2025 target mass, so
vacated shares are unaffected. **Coverage: 0 unmatched among the 108 fantasy-relevant
2025 WRs (tpg ≥ 3); 30/30 board players matched.** Team codes normalized
(Sleeper LAR→LA, OAK→LV; FFC LAR→LA).

**Movers on the 2026 board (3):** A.J. Brown PHI→NE, Jaylen Waddle MIA→DEN,
Mike Evans TB→SF. Cross-check vs the FFC ADP team column: **0 disagreements** on all
30 (FFC already reflects the moves).

**`data/derived/vacated_2026.csv`:** league-mean 2026 vacated share **0.258** vs
historical mean 0.294 (historical season means ranged 0.251–0.334) — low-normal,
sane. Unmatched target mass ≤ 0.082/team (mean 0.011), reported per team as a bias
check; the two largest (JAX .082, ARI .050) are not entering teams of any board
player. Extremes face-valid: MIA 0.530 (Hill + Waddle both gone), WAS 0.523, PIT
0.479 (Metcalf's own entering team keeps him: he is a stayer), GB 0.402; DEN 0.021
and LA 0.026 are full-retention rooms — note both are entering teams of board
players (Waddle→DEN gets essentially no vacated boost; Nacua/Adams sit on a
no-turnover offense).

## §E — arm (vii), context-adjusted data arm

B2 panel rebuilt from scratch and replicated exactly: n = 958, β_tc = −1.0040,
β_vac = +1.9216 (round-2: −1.00 / +1.92). **Replication bug caught and fixed before
scoring:** a first draft joined the gated `situation_change.csv` to define s+1
change flags, silently imposing the tpg ≥ 3 gate on the *outcome* season (n fell to
784, β_tc −0.44) — flags rebuilt ungated per script 14's construction.

Fold-refit betas on ≤ Y−1 pairs (plain OLS, point estimates only):

| Y | n_train | β_tc | β_vac |
|---|---|---|---|
| 2015 | 0 | 0 (no data) | 0 |
| 2016 | 0 | 0 (no data) | 0 |
| 2017 | 90 | **+1.218** | +2.222 |
| 2018 | 183 | −0.484 | +0.661 |
| 2019 | 280 | −0.537 | −0.273 |
| 2020 | 376 | −0.390 | +0.407 |
| 2021 | 469 | −0.195 | +0.902 |
| 2022 | 570 | −0.587 | +1.855 |
| 2023 | 665 | −0.671 | +1.807 |
| 2024 | 766 | −0.746 | +1.817 |

β_tc stabilizes negative (−0.2 to −0.75) once ≥2 training seasons exist but never
reaches the full-sample −1.0; β_vac wanders −0.27 to +2.2 and only settles ≈ +1.8
from 2022. The B2 effects are real but *slowly identified* — half the folds apply
materially mis-estimated coefficients.

**Scorecard (`results/loso_scorecard3.csv`), arms (i)/(ii) reproduced bit-exact
(asserted) before (vii) was scored:**

| arm | RMSE | Spearman | DM vs (i) | DM vs (ii) |
|---|---|---|---|---|
| (i) ADP-only | 3.5636 | .4610 | — | p = .025 (worse) |
| (ii) blind θ* | 3.4631 | .4667 | t = +2.68, p = .025 | — |
| (vii) θ*ᶜ context-adjusted | 3.4724 | .4549 | t = +2.72, p = .024 | **t = −0.81, p = .439** |

**Verdict: NOT adopted** (rule: DM vs (ii) p < .10 AND RMSE improvement; it fails
both — RMSE is *worse* by 0.009 and the DM points the wrong way).

Anomalies chased:
1. **The 2017 fold is the single largest loss** (yearly mean loss diff −0.687).
   Its β_tc = +1.22 has the wrong sign — fit on one training season (s+1 = 2016,
   n = 90). Its four movers (B. Marshall→NYG, Pryor→WAS, Cooks→NE, Jeffery→PHI, all
   entering high-vacated teams, so doubly boosted +0.48 to +0.60) all underperformed:
   mover-level loss diffs −7.7, −6.2, −3.3, −3.1. Small-sample fold noise, honestly
   propagated.
2. **Excluding 2017, movers are a wash** (mean per-row loss diff −0.11 over 20 mover
   rows; −0.94 including 2017). Only 24 mover rows exist in ten years of boards —
   top-30 boards rarely carry movers, and B3 (round 2) already showed the market
   prices moves (its point estimates said movers are, if anything, slightly
   *over*priced). Shifting μ̂ down −β_tc on a market-selected mover re-penalizes what
   ADP already discounted. This is the *expected* outcome given the B3 null, now
   confirmed in loss space.
3. **The 2024 fold gain (+0.34) is not a mover story** (1 mover, d = +0.10): it comes
   from vacated-centering nudging down μ̂ on low-turnover teams (MIA .124, SF .066)
   whose stars collapsed for unrelated reasons (Hill, Aiyuk, Deebo, Cooper, Waddle)
   — right direction, wrong mechanism; no reason to expect it to repeat.

**2026 board for the record (`results/sectionE_2026.csv`, labeled NOT adopted).**
Full-sample betas (β_tc −1.004, β_vac +1.922, mean vacated 0.283). Largest moves vs
the blind board: Waddle −0.64 PPG (mover into DEN's 0.021-vacated room: both terms
negative, drops 25→27), A.J. Brown −0.44 (8→9), Evans −0.42 (20→23), McLaurin +0.20
(22→20, WAS 0.523 vacated), Metcalf +0.16 (27→26). None of this enters
`valuation_2026_v3.csv`.

## Files
- `data/derived/vacated_2026.csv` (new; historical `vacated_targets.csv` untouched)
- `results/loso_scorecard3.csv`, `results/loso_predictions3.csv`
- `results/sectionE_2026.csv` (not-adopted board, for the record)
- `scripts/17_context_arm.py`

## Deviations from plan text
- Tertiary name-match rule added after seeing the first unmatched report (documented
  above and in the script docstring; a matching fix, not a model refit).
- Folds 2015/2016 carry β = 0 (the B2 panel's earliest outcome season is 2016; no
  ≤ Y−1 data exists) — arm (vii) ≡ arm (ii) there; stated rather than back-filled.
- B2 fold refits use plain OLS (only point estimates enter the adjustment; the
  round-2 clustered inference is not re-run per fold).

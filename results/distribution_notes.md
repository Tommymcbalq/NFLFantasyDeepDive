# §Q — per-player game-level distribution layer (all four positions)

**Status: descriptive infrastructure.** Nothing in this section is a hypothesis test, nothing
opens a new FDR family, nothing enters θ\* or any board value. Every threshold, inclusion rule
and estimator is *reused* from an upstream pre-registered section; none is invented here. The
one regression reported (§Q7) is labelled descriptive and is not interpreted inferentially.

Purpose: the board gives one number per player. It hides shape. At ADP 88–105 four WRs price
within ~1.7 PPG of each other and have nothing else in common — for a bench pick the upper tail
is what is being bought, for a weekly starter the floor is, and one mean cannot express either.

Scripts: `scripts/44_player_distributions.py` (table), `scripts/45_distribution_figures.py`
(figures + pair search). Both rerunnable, both take zero arguments.

Outputs:

| file | content |
|---|---|
| `results/player_distributions.csv` | 1,416 rows = 204 players × window; 74 columns |
| `results/player_distributions_eb_params.csv` | fitted Beta(α, β) per position × window × rate × fit-universe |
| `results/player_distribution_shape_persistence.csv` | consecutive-season pairs for §Q7 |
| `results/player_distribution_pairs.csv` | every same-position pair with \|Δboard_value\| ≤ 0.15 |
| `results/player_distribution_pairs_nearadp.csv` | the same, further restricted to \|ΔADP\| ≤ 12 |
| `results/figures/dist_tier_wr_adp84_110.png` | the worked-example tier |
| `results/figures/dist_tier_wr_adp128_162.png` | WR isotonic plateau — identical board value |
| `results/figures/dist_tier_rb_adp128_200.png` | RB isotonic plateau — identical board value |
| `results/figures/dist_eb_shrinkage.png` | raw → EB movement vs sample size, per position |

---

## §Q1 Universe

204 players, none imputed, none hand-added:

| source | n | position |
|---|---|---|
| `results/board_2026_full.csv` | 88 | WR |
| `results/board_2026_full.csv` | 68 | RB |
| `results/valuation_te_2026.csv` (§O) | 24 | TE |
| `results/valuation_qb_2026.csv` (§O) | 24 | QB |

`board_value` is `value_final` for WR/RB and `board_value` for TE/QB, carried through unchanged
as the point estimate this layer sits next to.

**Identity resolution.** TE/QB boards already carry `gsis_id`. WR/RB are matched by normalised
display name against the weekly panel, reusing §M's `norm_name`/`collapse_initials` and §P's
matcher verbatim. All 204 resolve. One note the run prints: **Travis Hunter** carries the weekly
position label `CB`, so the offensive-position screen inherited from §P misses him; the matcher
falls back to name alone and logs it. Without that fallback he would have been silently dropped
as "no NFL rows" despite having a rowed 2025 season — worth knowing for any future two-way
player.

## §Q2 Inclusion rules — inherited, not chosen here

Regular season only (`season_type == 'REG'`), seasons 2014–2025 (every board player's rookie
season is ≥ 2014, so `career` is a true career for all 204).

| position | rule | source |
|---|---|---|
| WR | drop player-games with `targets ≤ 1` | §0 |
| TE | drop player-games with `targets ≤ 1` | §O2 |
| RB | drop player-games with `touches = carries + targets ≤ 1` | §G1 |
| QB | drop player-games with `attempts ≤ 5` | §O2 |

All rates are therefore **conditional on participation**, in the strong sense §G1 documented
(17.2% of RB player-games and 35.9% of TE player-games league-wide fail the filter). The
non-participation mass is not in these distributions; availability lives in §13/`availability_table*`.

*Known edge case, documented not fixed:* the WR filter is target-based, so a WR game with ≤ 1
target but real rush usage is dropped. Across 2023–2025 that is **8 of 7,397 WR games**
(2 of them one player's). The rule is frozen upstream; changing it here to accommodate a
hybrid usage profile would be exactly the kind of post-hoc redefinition the protocol forbids.

## §Q3 Boom / bust thresholds — every one stated with its derivation

| position | boom | bust | derivation |
|---|---|---|---|
| WR | > 20.0 | < 8.0 | §1, **fixed a priori by the plan**; round numbers, not estimated from data |
| RB | > 13.8 | < 3.2 | §G1, pooled p75 / p25 of all 14,791 qualified RB player-games 2014–25 (positional population, *not* the board, so board selection cannot enter) |
| TE | > 11.1 | < 3.4 | §O2, same construction on 8,980 qualified TE player-games |
| QB | > 20.9 | < 9.7 | §O2, same construction on 6,659 qualified QB player-games |

The script **recomputes** the RB/TE/QB p75/p25 from the population at every run and prints a
MATCH / DRIFT tag against the frozen value. Current run: RB 13.80/3.20 MATCH, TE 11.10/3.40
MATCH, QB 20.90/9.72 MATCH. WR is printed for reference only — the population p75/p25 are
14.30/4.10, so the a-priori 20/8 is a deliberately *stricter* boom bar and a *looser* bust bar
than the positional quartiles. That asymmetry is inherited from §1 and is why WR boom rates
(prior mean 0.16) are not comparable in level to RB/TE/QB boom rates (0.34–0.41). **Rates are
comparable within a position, never across positions.**

## §Q4 Windows

One row per player × window. `window` takes:

| value | definition |
|---|---|
| `career` | every included game 2014–2025 |
| `last3` | seasons 2023–2025 — the owner's stated preference for established players, and the window the advanced layer uses |
| `recent` | 2025 only |
| `season_YYYY` | each individual season the player has ≥ 1 included game in |

The `season_YYYY` rows exist so that *a single outlier year is visible rather than averaged
away*: read a player's pooled `last3` shape next to his three season rows and it is immediately
apparent whether the pooled ceiling is a standing property or one season's work.

## §Q5 Sample-size discipline (non-negotiable)

- `n_games` is on every row.
- **Quantiles (`min`, `p10`, `p25`, `median`, `p75`, `p90`, `max`, `iqr`, `mad`, `skew`) are
  emitted as null whenever `n_games < 8`.** Verified: 0 rows violate this in either direction.
- Rookies and 2026 draftees with no NFL rows would get one flagged row per pooled window with
  every statistic null and `no_nfl_rows = True`. **Nothing is ever imputed.** (After the
  two-way-player fix in §Q1, the current 204-player universe has no such rows.)
- `season_YYYY` rows are simply absent for seasons a player did not play.
- Raw rates are printed at any `n_games ≥ 1` — that is what the EB column exists to protect —
  but always beside `n_games` and `thin_flag`.

## §Q6 Empirical-Bayes stabilisation of the rate stats

Model (§1.4, unchanged):

    k_i | p_i ~ Binomial(m_i, p_i),   p_i ~ Beta(α, β) across players
    Var(p̂_i) = Var(p_i) + E[p_i(1−p_i)/m_i]
    ⟹  V̂ar(p) = Var(p̂) − p̄(1−p̄)·mean(1/m),   α+β = p̄(1−p̄)/V̂ar(p) − 1,   α = p̄(α+β)
    reported value = posterior mean (k_i + α)/(m_i + α + β)

**Refit per position AND per window.** m_i differs by window, so the binomial-noise term being
subtracted differs; carrying a career-window prior into a single-season column would understate
shrinkage exactly where it matters most. Fitted on rows with `m_i ≥ 8` only (the moment
estimator is dominated by the noise correction at tiny m), then applied to every row including
the thin ones.

**Two fit universes are reported.** `boom_eb` / `bust_eb` (headline) fit (α, β) on the *board*,
which is the project convention (§1, §G, §O all fit on the board) and the right reference class
— the owner is choosing among drafted players, not among all NFL bodies. `boom_eb_pop` /
`bust_eb_pop` fit on the *full positional population* (all players with m ≥ 8 in that window) as
the range-restriction sensitivity. The contrast is itself informative: board priors are both
higher-mean and much *tighter* (prior n₀ of 25–46 at TE/QB vs 9–12 in the population), i.e.
**within the drafted tier players are far more alike than across the position**, so board-fit EB
shrinks harder. Both columns ship; neither is tuned.

Fitted parameters, board fit, pooled windows:

| pos | window | rate | n_fit | α | β | prior mean | prior n₀ | Var(p̂) | binom noise | V̂ar(p) |
|---|---|---|---|---|---|---|---|---|---|---|
| WR | career | boom | 72 | 2.436 | 12.771 | 0.160 | 15.21 | .0118 | .0035 | .0083 |
| WR | career | bust | 72 | 2.870 | 4.972 | 0.366 | 7.84 | .0324 | .0061 | .0262 |
| WR | last3 | boom | 72 | 2.471 | 13.063 | 0.159 | 15.53 | .0124 | .0043 | .0081 |
| WR | last3 | bust | 72 | 2.679 | 4.622 | 0.367 | 7.30 | .0355 | .0075 | .0280 |
| WR | recent | boom | 66 | 1.328 | 7.426 | 0.152 | 8.75 | .0229 | .0097 | .0132 |
| WR | recent | bust | 66 | 3.238 | 5.239 | 0.382 | 8.48 | .0428 | .0178 | .0249 |
| RB | career | boom | 58 | 1.939 | 3.714 | 0.343 | 5.65 | .0406 | .0067 | .0339 |
| RB | career | bust | 58 | 1.007 | 5.358 | 0.158 | 6.37 | .0220 | .0040 | .0181 |
| RB | last3 | boom | 58 | 1.816 | 3.369 | 0.350 | 5.18 | .0449 | .0081 | .0368 |
| RB | last3 | bust | 58 | 0.889 | 5.057 | 0.150 | 5.95 | .0228 | .0045 | .0183 |
| RB | recent | boom | 56 | 1.616 | 3.089 | 0.344 | 4.70 | .0556 | .0161 | .0395 |
| RB | recent | bust | 56 | 0.842 | 4.868 | 0.147 | 5.71 | .0277 | .0090 | .0187 |
| TE | career | boom | 23 | 10.513 | 15.297 | 0.407 | 25.81 | .0153 | .0063 | .0090 |
| TE | career | bust | 23 | 3.108 | 27.955 | 0.100 | 31.06 | .0052 | .0023 | .0028 |
| TE | last3 | boom | 23 | 10.038 | 14.949 | 0.402 | 24.99 | .0171 | .0078 | .0092 |
| TE | last3 | bust | 23 | 4.264 | 42.248 | 0.092 | 46.51 | .0045 | .0027 | .0018 |
| TE | recent | boom | 23 | 11.031 | 15.539 | 0.415 | 26.57 | .0269 | .0181 | .0088 |
| TE | recent | bust | 23 | 3.020 | 33.602 | 0.083 | 36.62 | .0076 | .0056 | .0020 |
| QB | career | boom | 24 | 9.842 | 18.400 | 0.349 | 28.24 | .0139 | .0061 | .0078 |
| QB | career | bust | 24 | 4.675 | 24.747 | 0.159 | 29.42 | .0080 | .0036 | .0044 |
| QB | last3 | boom | 23 | 12.235 | 22.330 | 0.354 | 34.56 | .0137 | .0072 | .0064 |
| QB | last3 | bust | 23 | 6.370 | 36.562 | 0.148 | 42.93 | .0069 | .0040 | .0029 |
| QB | **recent** | **boom** | 21 | **3098.9** | **5429.1** | 0.363 | **8528** | .0165 | .0165 | **~0** |
| QB | recent | bust | 21 | 5.745 | 30.292 | 0.159 | 36.04 | .0132 | .0096 | .0036 |

`eb_degenerate` flags any cell where V̂ar(p) < 0.2 × the binomial noise it had to subtract — the
prior swamps the data and everyone in the cell collapses to the pool rate. Flagged cells:
**QB × recent (and QB × season_2025)**, plus TE season_2020/2021 and WR season_2017/2019 where
the board contains too few players with rows in those years.

### The QB × 2025 boom-rate degeneracy — chased, and it is real

`boom_eb` for every 2025 QB comes back ≈ 0.363, regardless of raw rate (Josh Allen raw .563 →
.364; Baker Mayfield raw .176 → .363). That is a 100% shrinkage and it demands an explanation
before the column is shipped.

Parametric bootstrap of the moment estimator under H₀ Var(p) = 0, resampling k_i ~ Bin(m_i, p̄):

| window | universe | n | p̄ | Var(p̂) | noise | V̂ar(p) | null SD | z |
|---|---|---|---|---|---|---|---|---|
| 2025 | board QBs | 21 | .363 | .0165 | .0165 | +.00003 | .0052 | **+0.01** |
| 2025 | all QBs m≥8 | 36 | .289 | .0206 | .0166 | +.0040 | .0040 | **+1.01** |
| 2024 | all QBs m≥8 | 36 | .263 | .0425 | .0154 | +.0271 | .0037 | +7.33 |
| 2023 | all QBs | 35 | .251 | .0294 | .0148 | +.0147 | .0036 | +4.14 |
| 2022 | all QBs | 33 | .226 | .0404 | .0135 | +.0269 | .0033 | +8.06 |
| 2021 | all QBs | 31 | .292 | .0335 | .0141 | +.0195 | .0035 | +5.50 |

So it is **not** board range-restriction (the all-QB fit degenerates almost as hard) and it is
**not** a coding error: it is specific to 2025. The corroborating quantity is the §1 corrected
between-player SD of QB season means:

| season | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | **2025** |
|---|---|---|---|---|---|---|---|---|
| corrected between-QB SD (PPG) | 2.92 | 3.56 | 3.50 | 3.32 | 3.51 | 3.40 | 3.25 | **2.42** |
| pooled σ̂_W (PPG) | 7.90 | 7.53 | 7.42 | 7.26 | 6.81 | 7.18 | 7.48 | **7.71** |

2025 has the lowest between-QB separation and near the highest game-level scatter in the
8-season window. The EB is doing precisely its job: **a single 2025 season carries essentially
no information about which QB booms more often**, so it returns the pool rate. Read the QB
`recent` boom column as uninformative by construction and use the `last3` column
(prior n₀ = 34.6, V̂ar(p) = .0064, not degenerate); `boom_eb_pop` is also available and does
retain spread. Reported as found; nothing was refit to make the column look better.

### How much shrinkage moved the extreme raw rates

Mean and max |EB − raw| on the boom rate, and the compression of the raw range:

| pos | window | mean \|Δ\| | max \|Δ\| | raw range | EB range |
|---|---|---|---|---|---|
| WR | last3 | .031 | .141 | .00–.48 | .06–.39 |
| RB | last3 | .033 | .253 | .00–.84 | .06–.78 |
| TE | last3 | .042 | .101 | .12–.65 | .22–.56 |
| QB | last3 | .045 | .115 | .17–.63 | .25–.52 |
| WR | recent | .052 | .182 | .00–.62 | .05–.46 |
| RB | recent | .054 | .167 | .00–.94 | .08–.80 |
| TE | recent | .078 | .249 | .12–.82 | .32–.57 |
| QB | recent | .120 | .363 | .00–.59 | .36–.36 (degenerate) |

The pattern is the intended one: long samples barely move, short samples move a lot, and the
0.00 and 1.00 raw rates — which is what a 2–5 game sample produces — never survive. The largest
single moves on `last3` are all 2–5 game samples (boom .00 → .25, .18, .14; bust 1.00 → .50).

## §Q7 Does the *shape* persist year to year? (descriptive)

474 consecutive-season pairs among board players, 2019–2025, both years with n ≥ 8. Pearson
correlations between season t and t+1 (`results/player_distribution_shape_persistence.csv`):

| pos | n | r(mean) | r(p25) | r(p90) | r(IQR) | r(SD) | r(bust) | r(boom) |
|---|---|---|---|---|---|---|---|---|
| QB | 79 | +0.476 | +0.403 | +0.379 | **−0.146** | +0.184 | +0.285 | +0.388 |
| RB | 136 | +0.649 | +0.586 | +0.498 | +0.308 | +0.298 | +0.529 | +0.590 |
| TE | 71 | +0.653 | +0.463 | +0.595 | +0.259 | +0.438 | +0.213 | +0.581 |
| WR | 188 | +0.592 | +0.483 | +0.481 | +0.166 | +0.238 | +0.545 | +0.458 |
| ALL | 474 | +0.689 | +0.629 | +0.554 | +0.194 | +0.294 | +0.702 | +0.696 |

**Location persists; dispersion does not.** And the sharper version — regress next season's
ceiling on this season's *mean* and this season's *ceiling* (OLS, HC3; descriptive, no p-value
is interpreted and this is not in any FDR family):

| pos | n | b(mean_t) | b(p90_t) | R² |
|---|---|---|---|---|
| QB | 79 | +0.419 (0.360) | +0.089 (0.279) | .165 |
| RB | 136 | +1.149 (0.297) | −0.272 (0.205) | .323 |
| TE | 71 | +0.782 (0.315) | +0.149 (0.180) | .401 |
| WR | 188 | +0.860 (0.239) | −0.045 (0.148) | .281 |
| ALL | 474 | +0.916 (0.128) | −0.066 (0.089) | .370 |

Once last season's mean is known, last season's ceiling adds **nothing** to next season's
ceiling, in every position. The honest reading of this whole layer follows from that: **the
distribution columns describe what a player's games looked like; they are not a forecast of next
year's shape.** Their forward-looking content travels almost entirely through the mean, which is
already in the board value. What they are for is telling the owner what *kind* of asset the
recent past was — and whether the pooled number is one season's work — not for ranking players
on ceiling.

## §Q8 Partial-season flag (§P)

`partial_flag` / `n_seasons_partial` / `share_games_partial` mark player-seasons with **< 12
games played** (raw REG appearances, pre-exclusion). §P found the data arm's deviation from the
market prior is worth ĉ = **+1.101** when μ̂ was earned in a full prior season and ĉ = **+0.042**
when it was not (interaction p = .0010, WR). A per-game rate earned in six games was earned in a
role the player may not hold. On the `last3` window the flag fires for 28/88 WR, 21/68 RB,
10/24 TE, 8/24 QB. Read every quantile on a flagged row as *conditional on the role he had*.

## §Q9 Advanced-layer join

The recent-3 (2023–2025) advanced profile from `data/derived/adv_*_recent3.csv` is joined onto
every row of the player, so the distribution sits next to the usage that produced it. It is a
*recent-3* profile on all windows including `career` — read it as current context, never as a
description of the career window. Share-based columns use `target_share_full` /
`carry_share_full` / `air_yards_share_full` (full-team-games denominator); the default
`target_share` uses an active-games denominator and sums to 1.36 per team-season, so it is not
used anywhere here. Coverage: QB 100%, TE 96%, RB 88%, WR 85% — the gaps are players with no
2023–2025 rows.

Joined columns (all prefixed `adv_`): pass catchers — `target_share`, `air_yards_share`, `adot`
(PFR) and `adot_nflverse`, `yac_per_rec`, `rz_target_share` (share of own team's red-zone
targets), `snap_share`, `targets_pg`, `wopr`, `catch_rate`, `separation` (NGS),
`deep_target_rate`, `games`. Backs — `carry_share`, `target_share`, `snap_share`, `touches_pg`,
`ybc_per_att`, `yac_per_att`, `box8_rate` (NGS ≥8 defenders), `ryoe_per_att`, `gl5_carries_pg`,
`goal_to_go_pg`, `yac_per_rec`, `adot`, `explosive_run_rate`, `games`. QBs — `attempts_pg`,
`epa_per_dropback`, `cpoe`, `adot`, `rush_share_of_ppr`, `designed_rushes_pg`, `gl5_carries_pg`,
`aggressiveness`, `sack_rate`, `games`.

## §Q10 Column dictionary — `results/player_distributions.csv`

| column | definition |
|---|---|
| `gsis_id`, `player`, `pos`, `team`, `adp` | identity; `adp` is 2026 FFC PPR 12-team |
| `adp_rank_pos` | rank within position by ADP |
| `board_value` | the point estimate this layer contextualises (§P board for WR/RB, §O for TE/QB) |
| `window` | `career` / `last3` / `recent` / `season_YYYY` (see §Q4) |
| `seasons_in_window` | pipe-delimited list of seasons contributing included games |
| `n_seasons`, `n_games` | seasons and **included** games in the window |
| `n_games_played` | raw REG appearances in the window, pre-exclusion |
| `n_seasons_partial` | seasons in the window with < 12 games played (§Q8) |
| `share_games_partial` | share of `n_games_played` sitting inside those seasons |
| `partial_flag`, `thin_flag`, `no_nfl_rows` | booleans; `thin_flag` = `n_games < 8` |
| `mean` | mean PPR per included game |
| `sd` | SD of per-game PPR (n ≥ 2) |
| `sigma_W` | df-weighted pool of within-season variances, √·  — the §1 within-season per-game SD; equals `sd` on single-season windows, and is the version to use on pooled windows because it does not absorb between-season level movement |
| `cv` | `sigma_W / mean` (the §1 CV) |
| `cv_sd` | `sd / mean` |
| `min`, `p10`, `p25`, `median`, `p75`, `p90`, `max` | empirical quantiles of per-game PPR; **null when `n_games < 8`** |
| `iqr` | `p75 − p25` |
| `mad` | 1.4826 × median absolute deviation — outlier-resistant scale |
| `skew` | sample skewness; PPR is right-skewed so `sd`/`cv` are pulled by ceiling games while `iqr`/`mad` are not. (Quantiles are equivariant under log(1+Y), so a log sensitivity reproduces them exactly; only the moment columns would change — that is why the robust companions ship instead of a separate log table.) |
| `boom_thresh`, `bust_thresh` | the position's thresholds (§Q3), on every row so no lookup is needed |
| `k_boom`, `k_bust` | counts of games over / under threshold |
| `boom_raw`, `bust_raw` | `k/m` |
| `boom_eb`, `bust_eb` | **headline** EB posterior means, board-fitted prior (§Q6) |
| `boom_eb_pop`, `bust_eb_pop` | EB posterior means, population-fitted prior (sensitivity) |
| `eb_boom_prior_n`, `eb_bust_prior_n` | α+β of the board prior for that cell — the number of pseudo-games the prior is worth, i.e. the shrinkage strength |
| `eb_degenerate` | prior swamps data in that position × window cell (§Q6) |
| `adv_*` | recent-3 advanced profile (§Q9) |

## §Q11 Figures

`dist_tier_*.png` — one row per player in an ADP band, sorted by ADP: thin bar p10–p90, thick
bar p25–p75, white tick at the median, grey dots the individual 2023–2025 games, and a
right-hand column of p25 / p90 / EB bust / EB boom. Board value and n are in the row label, so
"same price, different shape" is readable directly off the y-axis. Tiers are chosen by ADP band
(the worked-example band, plus the two isotonic plateaus where board value is literally constant
across the tier); no player name entered the tier choice.

`dist_eb_shrinkage.png` — raw rate (orange) and EB posterior (blue) against n_games, one panel
per position, with a segment joining each player's pair. It is the picture of the sample-size
discipline: the leftmost points move the whole way to the prior, the rightmost barely move, and
the QB panel is visibly the degenerate one.

# §P lab notes — deeper universe: top-60 WR, top-50 RB, and RB handcuffs

Executed 2026-08-11 against `EDA_PLAN6.md §P` (pre-registered the same day, before any
§P fitting). Rules unchanged: fit as specified, report what comes out, no named-player
anchors in any pipeline, no new edge tests, closed FDR families stay closed.

Scripts: `40_sectionP_market_prior_deep.py`, `41_sectionP_loso_deep.py`,
`41b_sectionP_loso_decomp.py`, `42_sectionP_boards.py`,
`42b_sectionP_tail_hitrates.py`, `43_sectionP_handcuffs.py`.

---

## §P2 The refit — and how little the top-30 region moves

**Panel.** Every WR (resp. RB) row on the FFC PPR 12-team boards 2015–2024, not the top-30
truncation of §6.1/§G3. **666 WR rows** (58–74/yr) and **603 RB rows** (53–65/yr), joined
via the validated `sectionM_common.build_panel` — **0 unmatched, 0 ambiguous**. Fitted
support now reaches ADP **170.9 (WR)** and **174.0 (RB)**, so the 2026 top-60 WR
(ADP ≤ 133.6) and top-50 RB (ADP ≤ 160.2) universes sit **inside** it.

The pre-registered ≥4-included-game fit floor removes 4.2% of WR rows (7 with 0 games) and
5.0% of RB rows (8 with 0 games). Survivorship it induces is mild and non-monotone in ADP
(WR deciles 1.5%–7.6%; RB 1.7%–8.3%), and the labelled all-rows sensitivity (0-game → PPG 0)
moves the curve by ≤0.53 PPG anywhere on either position. Floor kept as pre-registered.

### Fits

| | WR deep | RB deep |
|---|---|---|
| OLS | `PPG = 24.365 − 2.913·log ADP` (se .158, R² .386, n 638) | `23.507 − 3.073·log ADP` (se .176, R² .441, n 573) |
| isotonic levels | 28 | 22 |
| range | 20.49 → 7.74 PPG | 19.40 → 7.48 PPG |
| in-sample RMSE iso / OLS | 3.256 / 3.349 | 3.557 / 3.676 |

Curve on a grid (PPG):

| ADP | 5 | 10 | 20 | 30 | 40 | 60 | 80 | 100 | 130 | 160 |
|---|---|---|---|---|---|---|---|---|---|---|
| **WR** | 19.14 | 17.94 | 15.40 | 14.88 | 14.32 | 12.85 | 12.20 | 10.82 | 10.17 | 8.41 |
| **RB** | 18.80 | 16.78 | 14.43 | 13.42 | 13.01 | 11.36 | 9.52 | 9.48 | 7.49 | 7.49 |

**Pre-registered expectation confirmed**: the tail is flatter and coarser. RB top-30 rows
span 16 levels over 9.88 PPG; tail rows span 8 levels over 3.44 PPG. WR: 17 levels over
8.05 PPG vs 14 over 5.12. Deep-tail values are tier membership, not rankings.

### How much does the top-30 region move? Essentially not at all — and that is provable

Movement of the refit curve against the frozen top-30 curve, over the frozen curve's own
fitted support:

| ADP band | WR mean Δ | WR RMS Δ | WR max\|Δ\| | RB mean Δ | RB RMS Δ | RB max\|Δ\| |
|---|---|---|---|---|---|---|
| ≤ 5 | 0.000 | 0.000 | 0.000 | −0.000 | 0.000 | 0.000 |
| 5–10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 10–20 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 20–30 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 30–40 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 40–55 | +0.017 | 0.055 | 0.221 | 0.000 | 0.000 | 0.000 |
| 55–75 | **+0.706** | 1.044 | **3.782** | +0.053 | 0.071 | 0.096 |
| 75–90 | — | — | — | −0.233 | 0.520 | 0.891 |

The zeros are exact, not rounded, and they are a property of PAVA rather than luck: adding
observations strictly to the right of a point changes the isotonic fit at that point **only
if the added data forces a pooled block to merge across it**. Left of ADP ≈ 40 no merge is
forced, so the fit is bit-identical. **All the movement is at the right edge of the old
fitted support**, which is exactly where a terminal block is estimated with no neighbours
to its right and is therefore least trustworthy.

**Restatement of the frozen 30-man boards** (recomputing θ* with the deep m(·) and deep
τ²(tier), σ²(tier) unchanged):

| | mean Δm | RMS Δm | max\|Δm\| | RMS Δθ* | max\|Δrank\| | Spearman(old, new θ*) |
|---|---|---|---|---|---|---|
| WR30 | +0.031 | 0.098 | 0.310 | 0.063 | **0** | **1.0000** |
| RB30 | +0.019 | 0.043 | 0.096 | 0.039 | **1** | 0.9991 |

Largest single move on either board: **Alec Pierce +0.205 PPG**; the RB board has two
adjacent one-slot swaps (Barkley/Walker, Hall/Brown). **The existing boards are not
restated in any way that changes a decision.** Files
`results/sectionP_top30_restatement_{wr,rb}.csv`.

**The consequential correction is not to the top-30 fit, it is to what happens beyond it.**
The old WR curve's terminal level was **8.656 PPG at ADP 75**; the deep fit puts that region
at **12.44**, and at ADP 95 gives **10.82** where the old curve, clipped, would have said
**8.66**. The old RB curve ended at **10.41 at ADP 80**; the deep fit says **9.52** there and
falls to **7.49** by ADP 130. So extrapolating the frozen curves would have been **~2.2 PPG
too pessimistic on deep WRs and ~2.9 PPG too optimistic on deep RBs** — opposite signs, both
material. This is precisely the error §P2 was written to prevent.

### τ²(tier)

| tier | WR old (top-30) | WR deep | RB old (top-30) | RB deep |
|---|---|---|---|---|
| rookie | 24.54 (n=4) | **11.42** (n=67) [8.08, 14.68] | 11.68 (n=25) | **13.88** (n=81) [9.61, 19.11] |
| soph | 7.85 (n=36) | **9.39** (n=100) [7.31, 11.41] | 14.66 (n=46) | **15.49** (n=98) [11.00, 20.50] |
| vet | 11.25 (n=251) | **10.78** (n=471) [9.44, 12.13] | 12.99 (n=215) | **11.75** (n=394) [10.03, 13.41] |

The WR rookie cell — an `n = 4` fiction in round 1, honestly labelled as such at the time —
is now identified on **67 rows**, and it collapses from 24.5 to 11.4. The pre-registered
ordering rookie > soph > vet still fails on both positions (WR: soph < vet < rookie; RB:
vet < rookie < soph). Used exactly as estimated; no ordering imposed. Deep-vs-shallow τ² is
flat (WR 10.86 vs 10.44 by ADP rank ≤30 / >30; RB 13.14 vs 12.25) — **the market is not
measurably less accurate deep than it is at the top**, in variance terms.

---

## §P4 The RB LOSO on the wider panel — and an anomaly on the WR side

Same arms, same adoption rule (DM t(9), p < 0.10 **and** RMSE improvement), everything refit
per fold. Power is stated with one explicit formula applied identically everywhere:
`MDE(80%, two-sided α = .10) = (t_{.95,9} + t_{.80,9})·SD_folds/√10`.

| panel | n | arm | RMSE base → arm | folds+ | mean gain | SD | DM t | p | MDE | obs/MDE | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RB top-30 (§G6, restated) | 286 | (ii) θ* | 3.905 → 3.838 | 5/10 | +0.488 | 2.015 | +0.766 | .464 | 1.731 | 0.28 | not adopted |
| **RB wide** | **573** | **(ii) θ\*** | **3.751 → 3.791** | **5/10** | **−0.313** | **0.958** | **−1.034** | **.328** | **0.823** | **−0.38** | **not adopted** |
| RB wide | 573 | (iii) SV | 4.291 → 4.346 | 3/10 | −0.498 | 1.274 | −1.236 | .248 | 1.095 | −0.45 | not adopted |
| WR wide | 638 | (ii) θ* | 3.435 → 3.401 | 5/10 | +0.246 | 0.608 | +1.279 | .233 | 0.522 | 0.47 | not adopted |
| WR wide | 638 | (iii) SV | 4.043 → 3.951 | 8/10 | +0.717 | 0.680 | +3.331 | **.0088** | 0.584 | 1.23 | **adopted** |

**VERDICT on the RB question §P4 was written to settle: the extra data did not close the
gap — it moved the point estimate the wrong way.** The across-fold SD falls from 2.015 to
0.958, MDE from 1.73 to 0.82 PPG², i.e. **2.1× the power**, and under that power the
estimated gain is **negative**. §G6's honesty clause therefore fires again, now with the
underpowering objection removed: **the RB board stays market-anchored.** This is a stronger
null than §G6's, not a repeat of it.

### The anomaly: WR arm (ii) loses significance on the wide panel — chased

§7 adopted WR arm (ii) at p = .025 on the top-30 panel. On the wide panel the same arm
returns p = .233. Decomposing the wide-panel fold differentials by ADP-rank stratum
(`41b_sectionP_loso_decomp.py`; a stratification of an already-run result, not a new test):

| stratum | WR n | WR mean gain | WR t | WR p | RB n | RB mean gain | RB t | RB p |
|---|---|---|---|---|---|---|---|---|
| all | 638 | +0.246 | +1.28 | .233 | 573 | −0.313 | −1.03 | .328 |
| **rank ≤ 30** | 290 | **+0.618** | **+2.59** | **.029** | 286 | +0.408 | +0.74 | .480 |
| rank 31–45 | 146 | +0.354 | +0.90 | .390 | 144 | −0.573 | −1.09 | .305 |
| **rank > 45** | 202 | **−0.471** | −1.57 | .151 | 143 | **−1.472** | **−3.20** | **.011** |

The top-30 subset of the wide panel reproduces §7 almost exactly (+0.618 vs +0.695,
p = .029 vs .025), so the loss of significance is **dilution by the tail, not a failure of
the original result**. The RB tail is worse: the data arm is significantly *harmful* beyond
ADP-rank 45.

Regressing realized PPG on `m̂` and on the deviation `θ* − m̂`, cluster(year) — the
coefficient `c` on the deviation is 1 if the deviation is exactly warranted and 0 if it is
noise:

| | WR c (rank ≤30) | WR c (rank >30) | RB c (rank ≤30) | RB c (rank >30) |
|---|---|---|---|---|
| ĉ (se) | **+1.275** (0.238) | +0.420 (0.300) | +0.693 (0.249) | **−0.137** (0.289) |
| H₀: c = 1 | p = .249 | p = **.053** | p = .217 | p = **.0001** |

**Mechanism, identified.** Interacting the deviation with whether the player played ≥ 12
games in the prior season (rows with prior data only):

| | ĉ, partial prior season | ĉ, full prior season | interaction (p) |
|---|---|---|---|
| WR | **+0.042** (0.302) | **+1.101** | +1.059 (**.0010**) |
| RB | −0.066 (0.232) | +0.528 | +0.594 (.0217) |

The data arm's deviation is worth **exactly what it claims (c ≈ 1.10) when μ̂ was earned in
a full-season role, and worth literally nothing (c ≈ 0.04) when it was not.** Cell-level
fold gains make it concrete: WR rank >45 with a partial prior season → **−2.743 per fold**;
WR rank >45 with a full prior season → **+0.277**. The share of rows with a partial prior
season rises from 11.9% (WR) / 20.4% (RB) inside the top 30 to 30.3% (WR) / 44.0% (RB) beyond
rank 45, which is why the failure concentrates in the tail. In a horse race the prior-role term survives for WR
(dev×full +0.871, p = .009) with a residual depth effect (dev×tail −0.775, p = .086); for RB
the depth term dominates (dev×tail −0.792, p = .044).

Interpretation: μ̂ estimates *points per game in the role he had*, and in the tail that role
is frequently not the role being priced — a spot-starter's fill-in rate, a rookie's
late-season surge. The market prices the projected role. **This is recorded as a candidate
for a future pre-registration (a role-conditioned likelihood variance, extending the §G6
open item on V) and is explicitly NOT acted on here.** No stratum-dependent adoption rule
was invented from it.

### What the boards therefore use

Each verdict is applied to the universe its test was run on, so no post-hoc rule is needed:

- **WR ADP-rank 1–30**: arm (ii), `board_value = θ*`. Warrant: §7 (p = .025) plus its
  wide-panel restatement (p = .029). Unchanged.
- **WR ADP-rank 31–60**: **market-anchored**, `board_value = m_deep(ADP)`. Warrant: the
  pre-specified wide-panel LOSO does not adopt arm (ii) (p = .233). θ* is carried as a
  diagnostic column; it is not a value.
- **RB ADP-rank 1–50**: **market-anchored**, `board_value = m_deep(ADP)` (§G6 + above).

This creates a visible seam at WR rank 30 (e.g. Alec Pierce's θ* 12.02 sits below Metcalf's
market 12.85). The seam is real and is not smoothed: two universes, two pre-specified tests,
two verdicts. **Within the market-anchored region board_value is a monotone function of ADP,
so tail ranks are ADP ranks by construction and carry no information.** Read the level, not
the rank.

---

## §P1 The boards

`results/valuation_wr60_2026.csv` (60 players, ADP 2.9–133.6; 52 vet / 4 soph / 4 rookie;
0 with zero NFL rows among veterans, 4 rookies at B = 1 and flagged) and
`results/valuation_rb50_2026.csv` (50 players, ADP 1.6–160.2; 2 rookies at B = 1).
Descriptive 2025 advanced columns are appended and are **not** inputs to `board_value`:
`target_share_full` (never the active-games `target_share`, which sums to 1.36/team-season),
red-zone and inside-10 target share of team, air-yards share, snap share, and the team's
2026 vacated target share. `routes_proxy` rates are omitted — the proxy counts blocking TEs
and protecting backs and is biased low across archetypes.

### What a late-round pick is actually worth (descriptive; `42b`)

Outcome distribution of the deep panel by ADP bucket. Sub-floor rows count as **misses**;
positional finish is computed among all players at the position with ≥ 8 included games that
season, so it is era-neutral. Wilson 95% intervals.

**WR**

| ADP bucket | n | PPG p10/p50/**p90** | P(top-12) | P(top-24) | P(bust) |
|---|---|---|---|---|---|
| 0–12 | 44 | 12.6 / 18.6 / **23.5** | .73 [.58,.84] | .84 [.71,.92] | .00 |
| 24–36 | 53 | 11.1 / 14.6 / **17.7** | .32 [.21,.45] | .66 [.53,.77] | .04 |
| 36–60 | 110 | 9.2 / 13.6 / **18.0** | .25 [.17,.33] | .45 [.36,.55] | .07 |
| 60–84 | 87 | 8.7 / 12.2 / **17.3** | .10 [.06,.19] | .24 [.16,.34] | .14 |
| **84–110** | 106 | 6.3 / 10.7 / **15.0** | .04 [.01,.09] | **.19 [.13,.27]** | .25 |
| 110–145 | 121 | 6.3 / 10.0 / **14.6** | .03 [.01,.08] | .14 [.09,.21] | .33 |

**RB**

| ADP bucket | n | PPG p10/p50/**p90** | P(top-12) | P(top-24) | P(bust) |
|---|---|---|---|---|---|
| 0–12 | 71 | 13.0 / 16.8 / **24.1** | .59 [.48,.70] | .82 [.71,.89] | .07 |
| 36–60 | 82 | 7.3 / 12.1 / **16.6** | .15 [.09,.24] | .40 [.30,.51] | .18 |
| 60–84 | 84 | 6.2 / 10.9 / **15.2** | .08 [.04,.16] | .29 [.20,.39] | .27 |
| **84–110** | 89 | 4.2 / 9.1 / **14.6** | .06 [.02,.12] | .20 [.13,.30] | **.39** |
| 110–145 | 99 | 3.5 / 7.8 / **13.2** | .05 [.02,.11] | .14 [.09,.22] | .56 |

At the same price the two positions have **almost identical top-24 hit rates but very
different floors**: at ADP 84–110, WR busts 25% of the time and RB 39%; by ADP 110–145 it is
33% vs 56%. Since a bench player's value is an option on the upside and a bust costs a
roster slot, this is the cleanest quantitative support for preferring late-round WR over
late-round RB — and it is a *price-level* statement, not a claim about any player.

Everything in this subsection is descriptive. No covariate is used to predict anything; the
red-zone-share/vacated-share story is **not tested here by design** (§P4 opens no families),
and testing it would require a pre-registration.

---

## §P3 Handcuffs: the transfer rate, which is worse than the folklore

Panel 2015–2024. Lead back L = the RB with most carries in a team-season, qualified at
≥ 8 present weeks and ≥ 10 carries/present game. Present = a row with touches ≥ 2. Eligible
team-seasons need ≥ 2 absent and ≥ 4 present weeks. **296 qualified lead-back team-seasons;
137 eligible.**

**Outage base rate.** P(≥1 missed week) = **.689**, P(≥2) = **.463**, P(≥4) = **.203**, mean
1.89 missed weeks. (Conditioning note: a back who misses so much that he loses the carry
lead is not counted as a qualified lead back, so this *understates* outage risk.)

**The identification problem, measured.** The ex-ante primary backup (top non-lead carrier
in the weeks the starter played — the back you could actually draft) is the same man as the
ex-post inheritor in only **64.2%** of eligible team-seasons. **Better than one time in
three, you handcuff the wrong player.**

**Transfer rates**, as a share of the lead back's own per-game workload. Accounting identity
`T_backup + T_rest − T_volume = 1` holds to 4.4e-16.

| resource | who | n | mean | q10 | q25 | **median** | q75 | q90 | P(T>.5) | P(T<.25) |
|---|---|---|---|---|---|---|---|---|---|---|
| carries | **ex-ante backup** | 137 | .321 | −.132 | .067 | **.309** | .571 | .765 | **.299** | **.416** |
| carries | ex-post inheritor | 137 | .594 | .268 | .405 | **.571** | .757 | .978 | .584 | .080 |
| carries | rest of the room | 137 | .565 | .105 | .247 | **.503** | .816 | 1.060 | — | — |
| carries | team RB volume Δ | 137 | −.115 | −.436 | −.281 | **−.132** | .005 | .246 | — | — |
| targets | ex-ante backup | 137 | .463 | −.403 | −.027 | .331 | .886 | 1.372 | .401 | .438 |
| targets | ex-post inheritor | 137 | .776 | .025 | .250 | .581 | 1.023 | 1.496 | .555 | .248 |
| inside-10 (2018+) | ex-ante backup | 101 | .439 | −.250 | −.062 | .339 | .756 | 1.134 | .436 | .436 |
| inside-10 (2018+) | ex-post inheritor | 101 | .704 | −.071 | .286 | .615 | 1.000 | 1.619 | .614 | .218 |

**The distribution is the finding.** The median identifiable handcuff absorbs **31% of the
carries**, not 100%. He gets less than a quarter of them **41.6%** of the time and more than
half **only 29.9%** of the time; the SD (0.364) is larger than the mean. The **rest of the
room takes a larger median share (50%) than the designated backup does**, and total RB carry
volume *falls* 13% when the lead back is out — teams throw more and spread the work rather
than promoting one man. Goal-line work transfers slightly better than carries (median .339)
and is the most variable. Sensitivities: injury-confirmed absences (≥50% of the absent weeks
on the report, n = 74) give a *higher* mean, .406 vs .321, i.e. genuine injuries transfer
more than benchings; restricting to ≥ 4 absent weeks (n = 60) changes nothing (mean .306).

**In fantasy points**, for the 126 team-seasons where both conditional means exist: the
ex-ante backup averaged **12.77 PPG (SD 6.44)** while the starter was out, up from 7.58 with
him in — a lift of **+5.19 mean / +4.31 median**. But **P(≥12 PPG, i.e. genuinely startable)
is only .476**, P(≥15) = .373, and **P(<8, unusable) = .238**. Conditional value model:

```
ppg_bk_out = 3.438 + 0.6195·ppg_bk_in + 0.3354·ppg_lead_in       cluster(season)
             (2.818)   (0.0761, p<.0001)  (0.1890, p = .076)
R² = .176, residual SD = 5.89 PPG, n = 126; residual skew +0.56, kurt +0.57
```

Note R² = .176 and residual SD 5.89 against a mean of 12.77: **the conditional value of a
handcuff is barely predictable.** The point estimates in the table below should be read with
their p10–p90 band, which is ~15 PPG wide for every player. Startability probabilities use
the **empirical** residual distribution, not a normal (skew +0.56).

The handcuff's own standing usage is what carries the prediction (t = 8.14); the size of the
role above him is marginal (t = 1.78). That is the same message as the transfer distribution
from a different direction: **you are mostly buying the backup's existing role, not an
option on the starter's.**

### 2026 handcuff table (`results/handcuff_table_2026.csv`)

Handcuff = the same-team RB ranked next by (Sleeper 2026 depth-chart order, then 2025
carries) for each top-30 ADP board RB. Top of the table by conditional value:

| starter (ADP) | handcuff (ADP) | cond. PPG if starter out | p10–p90 | P(startable ≥12) | standalone |
|---|---|---|---|---|---|
| Bucky Irving (42.8) | **Kenny Gainwell (89.6)** | 16.15 | 9.5–24.3 | .75 | 9.48 |
| Rico Dowdle (71.3) | **Jaylen Warren (67.2)** | 16.11 | 9.5–24.3 | .75 | 10.93 |
| Cam Skattebo (30.6) | **Tyrone Tracy (156.1)** | 15.43 | 8.8–23.6 | .71 | 7.49 |
| TreVeyon Henderson (55.7) | **Rhamondre Stevenson (72.2)** | 15.42 | 8.8–23.6 | .71 | 10.93 |
| Jahmyr Gibbs (1.6) | **Isiah Pacheco (148.9)** | 14.84 | 8.2–23.0 | .64 | 7.49 |
| Jadarian Price (64.5) | **Zach Charbonnet (139.1)** | 14.13 | 7.5–22.3 | .62 | 7.49 |
| Travis Etienne (37.1) | **Alvin Kamara (132.8)** | 14.12 | 7.5–22.3 | .62 | 7.49 |
| D'Andre Swift (44.4) | **Kyle Monangai (100.1)** | 13.58 | 7.0–21.8 | .52 | 9.48 |
| De'Von Achane (9.4) | Jaylen Wright (—) | 13.52 | 6.9–21.7 | .52 | — |
| Kyren Williams (32.2) | **Blake Corum (121.7)** | 13.09 | 6.5–21.3 | .52 | 8.52 |
| … | … | … | … | … | … |
| Josh Jacobs (24.6) | MarShawn Lloyd (—) | 8.74 | 2.1–16.9 | .25 | — |
| Ashton Jeanty (14.1) | Mike Washington (162.2) | 8.27 | 1.7–16.5 | .23 | — (rookie) |

Flags carried in the file: `committee_flag` (Warren/Dowdle and Henderson/Stevenson are
committees, not starter/handcuff pairs — both appear twice, in both directions);
`out_of_support` (McCaffrey→Jordan James, Taylor→DJ Giddens, Jacobs→MarShawn Lloyd — the
backup has essentially no 2025 usage, below the panel's 5th percentile of `ppg_bk_in`, so
the regression is extrapolating and the number is the intercept talking); `rookie_flag`.

**The decision arithmetic the table implies.** Even the best 2026 handcuff needs
P(starter misses ≥2 weeks) ≈ .46 × P(this back is the actual inheritor) ≈ .64 ×
P(startable | inheriting) ≈ .48, which is **≈ 14%** for a single roster slot held all
season. Handcuffs whose conditional value comes mostly from their *own* standing role —
Gainwell, Warren, Stevenson, Monangai, Charbonnet — are the ones that survive that
arithmetic, because they hold value in the 54% of seasons where the starter stays healthy.
Pure lottery-ticket handcuffs behind a workhorse (Pacheco, Tracy, Corum, Washington) are
paying the full 14% price.

---

## Files written by §P (nothing overwritten)

| file | contents |
|---|---|
| `results/market_prior_{wr,rb}_deep.csv` | §P2 deep panels with fitted m and residuals |
| `results/market_prior_iso_knots_{wr,rb}_deep.csv` | refit isotonic step functions |
| `results/tier_variances_{wr,rb}_deep.csv` | τ²(tier) with bootstrap CIs |
| `results/sectionP_curve_movement.csv` | old vs deep curve on the frozen support |
| `results/sectionP_top30_restatement_{wr,rb}.csv` | per-player Δm, Δθ*, Δrank |
| `results/sectionP_floor_survivorship_{wr,rb}.csv` | fit-floor incidence by ADP decile |
| `results/loso_scorecard_deep.csv`, `loso_predictions_deep.csv` | §P4 wide-panel LOSO |
| `results/sectionP_loso_decomp.csv` | the stratum/role decomposition of §P4 |
| `results/valuation_wr60_2026.csv`, `valuation_rb50_2026.csv` | §P1 boards |
| `results/sectionP_tail_hitrates.csv` | outcome distribution by ADP bucket |
| `results/sectionP_transfer_rates.csv` | per-team-season handcuff transfer panel |
| `results/sectionP_transfer_distribution.csv` | the distribution table above |
| `results/handcuff_table_2026.csv` | §P3 2026 handcuffs |

## Open items

1. **Role-conditioned likelihood variance.** The `dev×full-prior-season` interaction
   (WR +1.06, p = .001) is the sharpest structural result in §P and it generalises §G6's
   open item: eq. (7)'s `V = σ²(tier)/n_eff` measures how noisily μ̂ estimates the player's
   *past* level and contains no term for role migration. A `V` inflated when the prior
   season was partial is the obvious pre-registration. Not applied here.
2. **The WR rank-30 seam** in `board_value` is an artifact of two universes having two
   verdicts. It disappears if a future round validates (or rejects) arm (ii) on a single
   panel spanning both.
3. **Handcuff identification is the binding constraint, not valuation.** 35.8% of the time
   the ex-ante backup is not the inheritor; no amount of better valuation of the wrong
   player helps. Depth-chart source (Sleeper, pulled July 16) is the weakest input in §P.
4. Inside-10 transfer rates rest on pbp 2018+ (101 team-seasons); the 2015–2017 rows are
   carries/targets only.

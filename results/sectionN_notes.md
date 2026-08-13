# §N — RB/WR tier finishes and offensive environment

Run 2026-08-09 18:25 | script `scripts/30_sectionN_tier_environment.py` (pre-registration in its docstring)

## 0. Build

- de-vig scale s = 0.16200 win-rate = 2.754 wins/17 (recomputed here, matches §I3).
- projected-offence proxy: de-vigged win total wt17, range 3.23-13.13; ranked 1-32 within season.
- realized offence: team points/game, range 13.9-35.3; ranked 1-32 within season.
- corr(projected wt17, realized points/game) within season: mean 0.429 (range 0.257-0.672). Agreement of the two top-10 sets: 0.669 of team-seasons.
- §L panel reused unchanged: 1694 board rows; board-team join 1691/1694 matched, 3 missing ([{'year': 2015, 'name': 'Sammy Watkins', 'pos_adp': 'WR'}, {'year': 2015, 'name': 'Fred Jackson', 'pos_adp': 'RB'}, {'year': 2015, 'name': 'Percy Harvin', 'pos_adp': 'WR'}]).
- modelling universe: 1266 drafted RB/WR board rows, 2015-2024 (RB 602, WR 664).

## §N1 The unconditional shares the owner asked for — **CONFOUNDED, read §N2**

P(the player's offence was top-10 | he finished in this tier), all finishers 2015-2024 (drafted or not), realized primary team. Base rate if tiers were spread evenly over teams = 10/32 = 31.2%.

| position | finish tier | n | on realized top-10 offence | Wilson 95% | on *projected* top-10 offence | Wilson 95% |
|---|---|---|---|---|---|---|
| RB | 1-12 | 120 | **47.5%** | [38.8%, 56.4%] | **31.7%** | [24.0%, 40.4%] |
| RB | 13-24 | 120 | **34.2%** | [26.3%, 43.0%] | **30.0%** | [22.5%, 38.7%] |
| RB | 25-36 | 120 | **23.3%** | [16.7%, 31.7%] | **33.3%** | [25.5%, 42.2%] |
| RB | 37-48 | 120 | **30.0%** | [22.5%, 38.7%] | **32.5%** | [24.8%, 41.3%] |
| WR | 1-12 | 121 | **51.2%** | [42.4%, 60.0%] | **40.5%** | [32.2%, 49.4%] |
| WR | 13-24 | 119 | **33.6%** | [25.8%, 42.5%] | **36.1%** | [28.1%, 45.1%] |
| WR | 25-36 | 120 | **30.0%** | [22.5%, 38.7%] | **30.8%** | [23.3%, 39.6%] |
| WR | 37-48 | 120 | **30.0%** | [22.5%, 38.7%] | **26.7%** | [19.6%, 35.2%] |

Per-season spread of the RB 1-12 realized share (recorded because §L requires it, **not** to be read as signal): 2015:25%, 2016:67%, 2017:50%, 2018:58%, 2019:50%, 2020:42%, 2021:50%, 2022:33%, 2023:50%, 2024:50%

## §N3 Conversion by cost bin and projected environment (the primary object)

**projected top-10 offence, finish top12** (rate [Wilson] (n); gap in pp)

| position | bin | on top-10 | off top-10 | gap (pp) |
|---|---|---|---|---|
| RB | R1-2 | 47.1% [34%,60%] (51) | 58.3% [47%,69%] (72) | **-11.3** |
| RB | R3-4 | 21.4% [10%,40%] (28) | 25.8% [17%,38%] (62) | **-4.4** |
| RB | R5-6 | 3.8% [1%,19%] (26) | 16.4% [9%,28%] (55) | **-12.5** |
| RB | R7-8 | 15.6% [7%,32%] (32) | 5.0% [2%,14%] (60) | **+10.6** |
| RB | R9+ | 3.1% [1%,11%] (65) | 4.6% [2%,9%] (151) | **-1.6** |
| WR | R1-2 | 52.1% [38%,66%] (48) | 52.8% [40%,66%] (53) | **-0.7** |
| WR | R3-4 | 26.9% [17%,40%] (52) | 28.6% [19%,41%] (63) | **-1.6** |
| WR | R5-6 | 16.1% [7%,33%] (31) | 9.2% [4%,19%] (65) | **+6.9** |
| WR | R7-8 | 7.4% [2%,23%] (27) | 12.1% [6%,23%] (58) | **-4.7** |
| WR | R9+ | 2.4% [1%,8%] (83) | 5.4% [3%,10%] (184) | **-3.0** |

**projected top-10 offence, finish top24** (rate [Wilson] (n); gap in pp)

| position | bin | on top-10 | off top-10 | gap (pp) |
|---|---|---|---|---|
| RB | R1-2 | 62.7% [49%,75%] (51) | 79.2% [68%,87%] (72) | **-16.4** |
| RB | R3-4 | 42.9% [27%,61%] (28) | 64.5% [52%,75%] (62) | **-21.7** |
| RB | R5-6 | 26.9% [14%,46%] (26) | 36.4% [25%,50%] (55) | **-9.4** |
| RB | R7-8 | 40.6% [26%,58%] (32) | 25.0% [16%,37%] (60) | **+15.6** |
| RB | R9+ | 12.3% [6%,22%] (65) | 15.2% [10%,22%] (151) | **-2.9** |
| WR | R1-2 | 68.8% [55%,80%] (48) | 67.9% [55%,79%] (53) | **+0.8** |
| WR | R3-4 | 59.6% [46%,72%] (52) | 60.3% [48%,71%] (63) | **-0.7** |
| WR | R5-6 | 38.7% [24%,56%] (31) | 26.2% [17%,38%] (65) | **+12.6** |
| WR | R7-8 | 18.5% [8%,37%] (27) | 17.2% [10%,29%] (58) | **+1.3** |
| WR | R9+ | 6.0% [3%,13%] (83) | 19.0% [14%,25%] (184) | **-13.0** |

**realized top-10 offence, finish top12** (rate [Wilson] (n); gap in pp)

| position | bin | on top-10 | off top-10 | gap (pp) |
|---|---|---|---|---|
| RB | R1-2 | 68.9% [54%,80%] (45) | 44.9% [34%,56%] (78) | **+24.0** |
| RB | R3-4 | 34.6% [19%,54%] (26) | 20.3% [12%,32%] (64) | **+14.3** |
| RB | R5-6 | 16.7% [7%,36%] (24) | 10.5% [5%,21%] (57) | **+6.1** |
| RB | R7-8 | 14.6% [7%,28%] (41) | 3.9% [1%,13%] (51) | **+10.7** |
| RB | R9+ | 8.1% [4%,17%] (74) | 2.1% [1%,6%] (142) | **+6.0** |
| WR | R1-2 | 64.6% [50%,77%] (48) | 41.5% [29%,55%] (53) | **+23.1** |
| WR | R3-4 | 37.8% [24%,54%] (37) | 23.1% [15%,34%] (78) | **+14.8** |
| WR | R5-6 | 14.8% [6%,32%] (27) | 10.1% [5%,19%] (69) | **+4.7** |
| WR | R7-8 | 20.0% [9%,39%] (25) | 6.7% [3%,16%] (60) | **+13.3** |
| WR | R9+ | 6.8% [3%,14%] (88) | 3.4% [2%,7%] (179) | **+3.5** |

**realized top-10 offence, finish top24** (rate [Wilson] (n); gap in pp)

| position | bin | on top-10 | off top-10 | gap (pp) |
|---|---|---|---|---|
| RB | R1-2 | 80.0% [66%,89%] (45) | 67.9% [57%,77%] (78) | **+12.1** |
| RB | R3-4 | 57.7% [39%,74%] (26) | 57.8% [46%,69%] (64) | **-0.1** |
| RB | R5-6 | 50.0% [31%,69%] (24) | 26.3% [17%,39%] (57) | **+23.7** |
| RB | R7-8 | 39.0% [26%,54%] (41) | 23.5% [14%,37%] (51) | **+15.5** |
| RB | R9+ | 18.9% [12%,29%] (74) | 12.0% [8%,18%] (142) | **+6.9** |
| WR | R1-2 | 79.2% [66%,88%] (48) | 58.5% [45%,71%] (53) | **+20.7** |
| WR | R3-4 | 70.3% [54%,83%] (37) | 55.1% [44%,66%] (78) | **+15.1** |
| WR | R5-6 | 40.7% [25%,59%] (27) | 26.1% [17%,38%] (69) | **+14.7** |
| WR | R7-8 | 24.0% [11%,43%] (25) | 15.0% [8%,26%] (60) | **+9.0** |
| WR | R9+ | 17.0% [11%,26%] (88) | 14.0% [10%,20%] (179) | **+3.1** |

## §N2 Why the unconditional number is not interpretable — the confound, measured

| position | environment | mean ADP on top-10 | mean ADP off top-10 | share of board rows on top-10 | share in R1-2 on top-10 |
|---|---|---|---|---|---|
| RB | projected | 71.5 | 77.9 | 33.6% | 41.5% |
| RB | realized | 77.7 | 74.7 | 34.9% | 36.6% |
| WR | projected | 72.9 | 85.0 | 36.3% | 47.5% |
| WR | realized | 76.0 | 82.9 | 33.9% | 47.5% |

A team-season supplies 31.2% of teams; if environment were orthogonal to price the shares above would all sit at 31.2% and the two mean ADPs would coincide.

## §N4 Primary tests — cost-conditioned, logit `hit ~ top10 + C(bin12)`, cluster(season), t(9)

| # | pos | tier | environment | beta (log-odds) | cluster SE | 95% CI | raw p | AME (pp) | MH stratified RD (pp) | MDE (log-odds) | MDE (pp) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | RB | top12 | projected | **-0.307** | 0.286 | [-0.955, +0.341] | 0.3113 | -3.6 | -3.7 | 0.901 | 13.9 |
| 2 | RB | top12 | realized | **+0.968** | 0.274 | [+0.349, +1.587] | 0.0063 | +11.8 | +11.7 | 0.861 | 13.3 |
| 3 | RB | top24 | projected | **-0.365** | 0.153 | [-0.711, -0.019] | 0.0410 | -6.6 | -6.6 | 0.482 | 11.3 |
| 4 | RB | top24 | realized | **+0.578** | 0.198 | [+0.130, +1.026] | 0.0171 | +10.6 | +10.6 | 0.623 | 14.6 |
| 5 | WR | top12 | projected | **-0.099** | 0.328 | [-0.841, +0.643] | 0.7703 | -1.1 | -1.2 | 1.032 | 15.0 |
| 6 | WR | top12 | realized | **+0.814** | 0.231 | [+0.291, +1.337] | 0.0065 | +9.9 | +10.1 | 0.727 | 10.5 |
| 7 | WR | top24 | projected | **-0.173** | 0.207 | [-0.640, +0.295] | 0.4253 | -3.0 | -3.0 | 0.650 | 14.5 |
| 8 | WR | top24 | realized | **+0.584** | 0.170 | [+0.199, +0.970] | 0.0075 | +10.4 | +10.4 | 0.536 | 11.9 |

## §N5 Unconditional versions of the same contrasts — **confounded, family = 0**

| pos | tier | environment | beta (no bin FE) | SE | raw p | AME (pp) | cost-conditioned AME (pp) | attributable to price |
|---|---|---|---|---|---|---|---|---|
| RB | top12 | projected | -0.028 | 0.242 | 0.9090 | -0.4 | -3.6 | +3.1 pp |
| RB | top12 | realized | +0.719 | 0.238 | 0.0146 | +11.6 | +11.8 | -0.2 pp |
| RB | top24 | projected | -0.133 | 0.116 | 0.2825 | -3.1 | -6.6 | +3.5 pp |
| RB | top24 | realized | +0.426 | 0.191 | 0.0528 | +10.1 | +10.6 | -0.5 pp |
| WR | top12 | projected | +0.244 | 0.287 | 0.4181 | +3.6 | -1.1 | +4.8 pp |
| WR | top12 | realized | +0.891 | 0.204 | 0.0018 | +13.7 | +9.9 | +3.8 pp |
| WR | top24 | projected | +0.158 | 0.160 | 0.3486 | +3.5 | -3.0 | +6.5 pp |
| WR | top24 | realized | +0.614 | 0.164 | 0.0046 | +14.0 | +10.4 | +3.6 pp |

## §N6 Sensitivities (family = 0)

| pos | tier | env | spec | estimate | SE | p |
|---|---|---|---|---|---|---|
| RB | top12 | projected | log(ADP) control instead of bins | -0.350 | 0.252 | 0.1992 |
| RB | top12 | projected | bins + season FE | -0.310 | 0.297 | 0.3230 |
| RB | top12 | projected | continuous wt17 (per win) + bins | +0.021 | 0.083 | 0.8026 |
| RB | top12 | projected | bins, R1-4 only | -0.389 | 0.349 | 0.2937 |
| RB | top12 | realized | log(ADP) control instead of bins | +0.919 | 0.298 | 0.0131 |
| RB | top12 | realized | bins + season FE | +0.995 | 0.282 | 0.0064 |
| RB | top12 | realized | continuous realized team PPG + bins | +0.130 | 0.029 | 0.0014 |
| RB | top12 | realized | bins, R1-4 only | +0.902 | 0.339 | 0.0260 |
| RB | top24 | projected | log(ADP) control instead of bins | -0.375 | 0.151 | 0.0346 |
| RB | top24 | projected | bins + season FE | -0.352 | 0.160 | 0.0557 |
| RB | top24 | projected | continuous wt17 (per win) + bins | -0.027 | 0.050 | 0.6039 |
| RB | top24 | projected | bins, R1-4 only | -0.845 | 0.329 | 0.0304 |
| RB | top24 | realized | log(ADP) control instead of bins | +0.539 | 0.207 | 0.0287 |
| RB | top24 | realized | bins + season FE | +0.597 | 0.206 | 0.0177 |
| RB | top24 | realized | continuous realized team PPG + bins | +0.062 | 0.018 | 0.0069 |
| RB | top24 | realized | bins, R1-4 only | +0.341 | 0.268 | 0.2352 |
| WR | top12 | projected | log(ADP) control instead of bins | -0.095 | 0.346 | 0.7888 |
| WR | top12 | projected | bins + season FE | -0.108 | 0.333 | 0.7526 |
| WR | top12 | projected | continuous wt17 (per win) + bins | +0.039 | 0.086 | 0.6620 |
| WR | top12 | projected | bins, R1-4 only | -0.055 | 0.388 | 0.8907 |
| WR | top12 | realized | log(ADP) control instead of bins | +0.851 | 0.233 | 0.0053 |
| WR | top12 | realized | bins + season FE | +0.825 | 0.235 | 0.0066 |
| WR | top12 | realized | continuous realized team PPG + bins | +0.123 | 0.038 | 0.0096 |
| WR | top12 | realized | bins, R1-4 only | +0.832 | 0.220 | 0.0044 |
| WR | top24 | projected | log(ADP) control instead of bins | -0.143 | 0.199 | 0.4921 |
| WR | top24 | projected | bins + season FE | -0.174 | 0.212 | 0.4331 |
| WR | top24 | projected | continuous wt17 (per win) + bins | -0.073 | 0.050 | 0.1834 |
| WR | top24 | projected | bins, R1-4 only | +0.001 | 0.291 | 0.9983 |
| WR | top24 | realized | log(ADP) control instead of bins | +0.550 | 0.176 | 0.0122 |
| WR | top24 | realized | bins + season FE | +0.579 | 0.170 | 0.0077 |
| WR | top24 | realized | continuous realized team PPG + bins | +0.070 | 0.016 | 0.0018 |
| WR | top24 | realized | bins, R1-4 only | +0.816 | 0.197 | 0.0025 |

## §N7 BH q = 0.10 over the 8-test §N family

| rank | pos | tier | env | beta | raw p | BH threshold | reject |
|---|---|---|---|---|---|---|---|
| 1 | RB | top12 | realized | +0.968 | 0.0063 | 0.0125 | **YES** |
| 2 | WR | top12 | realized | +0.814 | 0.0065 | 0.0250 | **YES** |
| 3 | WR | top24 | realized | +0.584 | 0.0075 | 0.0375 | **YES** |
| 4 | RB | top24 | realized | +0.578 | 0.0171 | 0.0500 | **YES** |
| 5 | RB | top24 | projected | -0.365 | 0.0410 | 0.0625 | **YES** |
| 6 | RB | top12 | projected | -0.307 | 0.3113 | 0.0750 | no |
| 7 | WR | top24 | projected | -0.173 | 0.4253 | 0.0875 | no |
| 8 | WR | top12 | projected | -0.099 | 0.7703 | 0.1000 | no |

**5 of 8 survive** (smallest raw p = 0.0063 against a threshold of 0.0125).

## §N8 Temporal holdout 2015-2021 -> 2022-2024

| pos | tier | env | fit 2015-21 beta | holdout 2022-24 beta | sign held | holdout AME (pp) |
|---|---|---|---|---|---|---|
| RB | top12 | projected | -0.682 | +0.517 | **no** | +6.2 |
| RB | top12 | realized | +1.027 | +0.849 | yes | +10.0 |
| RB | top24 | projected | -0.416 | -0.278 | yes | -4.9 |
| RB | top24 | realized | +0.370 | +1.163 | yes | +20.0 |
| WR | top12 | projected | +0.185 | -0.922 | **no** | -9.3 |
| WR | top12 | realized | +0.955 | +0.344 | yes | +3.9 |
| WR | top24 | projected | -0.057 | -0.440 | yes | -7.4 |
| WR | top24 | realized | +0.443 | +0.902 | yes | +15.8 |

Sign stability: **6 of 8**.

## §N9 Diagnostics and mechanism

**(a) Where does any unconditional gap come from — price composition or production?**

- RB, projected: on top-10 mean ADP 71.5, mean finish rank 42.4, PPG 11.16, games 12.05; off top-10 ADP 77.9, rank 38.4, PPG 11.00, games 13.08.
- WR, projected: on top-10 mean ADP 72.9, mean finish rank 47.8, PPG 12.38, games 13.47; off top-10 ADP 85.0, rank 48.1, PPG 11.81, games 13.57.

**(b) Consistency with §I3 (mean channel) on this panel** — OLS of season PPG on the top-10 indicator with bin FE, cluster(season):

- RB: -0.178 PPG (SE 0.292, p = 0.5430), n = 575
- WR: -0.310 PPG (SE 0.445, p = 0.4854), n = 642

**(c) Tail vs mean — the §N hypothesis stated as a variance question.** SD of season PPG within cost bin, on vs off a projected top-10 offence (bin-weighted pooled SD):

- RB: SD on top-10 4.10 (n=190) vs off 3.67 (n=385); F = 1.252, p = 0.0685 (family = 0, descriptive)
- WR: SD on top-10 3.60 (n=233) vs off 3.47 (n=409); F = 1.075, p = 0.5279 (family = 0, descriptive)

**(d) §25.3 decomposition applied to the tier outcome.** Slope of each channel on the top-10 indicator (linear-probability form so the identity is exact):

- RB top-12: beta_realized -0.4 pp = beta_priced +3.2 pp + beta_residual -3.7 pp (SE 3.5, p = 0.3013); priced share undefined: beta_realized is ~0, so the ratio is meaningless
- WR top-12: beta_realized +3.6 pp = beta_priced +4.8 pp + beta_residual -1.2 pp (SE 4.0, p = 0.7722); priced share 133%

---

## §N10 Anomaly chasing

**(A) The realized arm contains the outcome — leave-own-player-out rebuild.**

A drafted RB/WR who finishes top-12 scores his team's touchdowns, so `team points scored` is not exogenous to his own finish. Own-TD points as a share of team points:

- RB: mean own-TD points 34.4 of 380 team points = 9.1%; among top-12 finishers 72.8 pts = 18.0%.
- WR: mean own-TD points 29.8 of 380 team points = 7.8%; among top-12 finishers 56.1 pts = 13.8%.

- reclassified by the leave-own-out rank: 171 of 1266 rows (13.5%) change top-10 status.

| pos | tier | original beta | leave-own-out beta | SE | p | AME (pp) | collapse |
|---|---|---|---|---|---|---|---|
| RB | top12 | +0.968 | **-0.301** | 0.368 | 0.4353 | -3.5 | 131% |
| RB | top24 | +0.578 | **-0.560** | 0.255 | 0.0558 | -10.0 | 197% |
| WR | top12 | +0.814 | **+0.040** | 0.244 | 0.8733 | +0.5 | 95% |
| WR | top24 | +0.584 | **-0.176** | 0.190 | 0.3781 | -3.0 | 130% |

**(B) The projected effect is negative for RB — decomposed into games and PPG, within cost bin.**

| pos | quantity | on projected top-10 | off | difference | cluster-t p |
|---|---|---|---|---|---|
| RB | games played | 12.05 | 13.08 | **-1.09** | 0.0035 |
| RB | PPG | >=1 game | 11.41 | 11.05 | **-0.18** | 0.5430 |
| RB | season total PPR | 143.24 | 148.09 | **-12.43** | 0.0004 |
| WR | games played | 13.47 | 13.57 | **-0.27** | 0.3396 |
| WR | PPG | >=1 game | 12.41 | 11.88 | **-0.31** | 0.4854 |
| WR | season total PPR | 169.18 | 164.51 | **-8.07** | 0.3395 |

Availability sub-check — is the games gap late-season rest? Mean REG appearances by week window, RB only, within cost bin (OLS with bin FE, cluster(season)):

| pos | quantity | on top-10 | off | difference | p |
|---|---|---|---|---|---|
| RB | all REG weeks | 12.05 | 13.08 | **-1.089** | 0.0035 |
| RB | weeks 1 to n-2 | 10.81 | 11.70 | **-0.949** | 0.0032 |
| RB | final 2 weeks | 1.25 | 1.38 | **-0.140** | 0.0413 |
| WR | all REG weeks | 13.47 | 13.57 | **-0.274** | 0.3396 |
| WR | weeks 1 to n-2 | 12.01 | 12.11 | **-0.247** | 0.3478 |
| WR | final 2 weeks | 1.46 | 1.47 | **-0.027** | 0.7339 |

Competition sub-check — how many RB/WR does a projected top-10 offence put on the board, and how concentrated is its usage?

- RB: drafted per team-season, projected top-10 2.02 vs off 1.85 (Welch p = 0.0460)
- WR: drafted per team-season, projected top-10 2.43 vs off 1.99 (Welch p = 0.0000)

Indicator-vs-continuous — the top-10 dummy and the continuous de-vigged win total disagree, so the relation is not monotone in projected quality. Hit rates by projected-rank tercile, within cost bin (bin-demeaned, RB):

- RB top12, bin-demeaned rate (pp vs bin mean): proj 1-11 -2.4 (n=202), proj 12-21 +3.7 (n=200), proj 22-32 -1.3 (n=200)
- RB top24, bin-demeaned rate (pp vs bin mean): proj 1-11 -4.4 (n=202), proj 12-21 +1.9 (n=200), proj 22-32 +2.5 (n=200)
- WR top12, bin-demeaned rate (pp vs bin mean): proj 1-11 -0.7 (n=241), proj 12-21 +1.4 (n=209), proj 22-32 -0.5 (n=214)
- WR top24, bin-demeaned rate (pp vs bin mean): proj 1-11 -1.9 (n=241), proj 12-21 -0.8 (n=209), proj 22-32 +2.9 (n=214)

Per-season stability of the RB projected contrast (bin-stratified risk difference, pp) — §L's rule that single-season rates are not signal applies; shown for spread:

- RB top12: 2015:-24, 2016:+2, 2017:-10, 2018:-3, 2019:-4, 2020:-3, 2021:-11, 2022:+0, 2023:+13, 2024:+2 | mean -3.7, negative in 6/10 seasons
- RB top24: 2015:-17, 2016:-14, 2017:-2, 2018:-10, 2019:-4, 2020:-1, 2021:-10, 2022:-30, 2023:+6, 2024:+2 | mean -8.0, negative in 8/10 seasons

**(C) Where the RB availability gap lives — by cost bin, and is it price?**

| pos | bin | games on top-10 (n) | games off (n) | diff | mean ADP on | mean ADP off |
|---|---|---|---|---|---|---|
| RB | R1-2 | 12.39 (51) | 13.74 (72) | **-1.34** | 10.6 | 11.5 |
| RB | R3-4 | 12.25 (28) | 13.34 (62) | **-1.09** | 35.7 | 36.0 |
| RB | R5-6 | 10.69 (26) | 12.91 (55) | **-2.22** | 59.3 | 60.0 |
| RB | R7-8 | 13.66 (32) | 12.75 (60) | **+0.91** | 84.2 | 84.8 |
| RB | R9+ | 11.46 (65) | 12.86 (151) | **-1.40** | 133.3 | 130.6 |
| WR | R1-2 | 14.08 (48) | 14.45 (53) | **-0.37** | 12.9 | 13.4 |
| WR | R3-4 | 14.17 (52) | 14.59 (63) | **-0.41** | 36.2 | 36.7 |
| WR | R5-6 | 13.26 (31) | 12.20 (65) | **+1.06** | 58.1 | 60.0 |
| WR | R7-8 | 12.63 (27) | 13.45 (58) | **-0.82** | 83.7 | 84.4 |
| WR | R9+ | 13.02 (83) | 13.50 (184) | **-0.48** | 132.7 | 131.1 |

Mediation check — add realized games played to the RB top-24 projected model. If the effect is the availability channel it should vanish (games is a POST-TREATMENT control, so this is diagnostic only, never a causal estimate):

- RB top-24: without games -0.365 (p = 0.0410) -> with games -0.070 (p = 0.8003); games coefficient +0.549 (p = 0.0000)
- WR top-24: without games -0.173 (p = 0.4253) -> with games -0.037 (p = 0.8192); games coefficient +0.663 (p = 0.0000)

**(D) Multiplicity, re-accounted.** The four REALIZED contrasts are mechanically contaminated (A) and are withdrawn, exactly as §28.3 withdrew its clairvoyant positional-SOS false positive. Two accountings are reported; the family declared before fitting is the first, and the second is what it becomes once the contaminated arm is replaced by its leave-own-out rebuild.

| rank | pos | tier | env | beta | raw p | BH threshold | reject |
|---|---|---|---|---|---|---|---|
| 1 | RB | top24 | projected | -0.365 | 0.0410 | 0.0125 | no |
| 2 | RB | top24 | realized (leave-own-out) | -0.560 | 0.0558 | 0.0250 | no |
| 3 | RB | top12 | projected | -0.307 | 0.3113 | 0.0375 | no |
| 4 | WR | top24 | realized (leave-own-out) | -0.176 | 0.3781 | 0.0500 | no |
| 5 | WR | top24 | projected | -0.173 | 0.4253 | 0.0625 | no |
| 6 | RB | top12 | realized (leave-own-out) | -0.301 | 0.4353 | 0.0750 | no |
| 7 | WR | top12 | projected | -0.099 | 0.7703 | 0.0875 | no |
| 8 | WR | top12 | realized (leave-own-out) | +0.040 | 0.8733 | 0.1000 | no |

**0 of 8 survive** in the de-contaminated family.

Projected arm alone (4 tests, the only preseason-knowable ones): RB top24 p=0.0410 vs thr 0.0250; RB top12 p=0.3113 vs thr 0.0500; WR top24 p=0.4253 vs thr 0.0750; WR top12 p=0.7703 vs thr 0.1000 -> 0 survive.

**This matters for reading §N7.** RB top-24 projected (p = .0410) cleared BH in the as-declared family only because the four contaminated realized tests occupied ranks 1-4 and lifted its threshold to .0625. Against the four preseason-knowable tests alone its threshold is .025 and it does not clear. It is reported as a suggestive, uncorrected signal, not as a survivor.

**(E) Is the RB availability deficit a roster ROLE effect or an injury effect?** Split each team-season's drafted RBs into the LEAD back (lowest ADP on that team) and the rest. A committee/depth mechanism should show the deficit mainly in the non-lead backs; an injury mechanism should not care about role.

| RB role | games on top-10 (n) | games off (n) | diff | bin-FE p | top-24 rate on | off |
|---|---|---|---|---|---|---|
| lead back | 12.10 (100) | 13.21 (216) | **-1.32** | 0.0604 | 51.0% | 55.6% |
| non-lead | 12.01 (102) | 12.93 (184) | **-0.91** | 0.0700 | 20.6% | 19.0% |

- Controlling for how many RBs the team has on the board, the top-10 games effect moves -1.09 -> -0.84 (p = 0.0315); the backfield-depth coefficient is -0.93 games per extra drafted RB (p = 0.0001).


**(F) Is the RB games deficit an age / experience composition effect?** Contenders sign veterans; veterans miss more games. Checked because it is the obvious alternative to a role story.

- RB age: on projected top-10 25.58 vs off 25.36; within-bin difference +0.259 (p = 0.2356), n = 602
- RB exp: on projected top-10 3.42 vs off 3.19; within-bin difference +0.231 (p = 0.3246), n = 602
- games effect with age, experience and backfield depth all controlled: -1.089 (p = 0.0035) -> -0.816 (p = 0.0352), n = 602

**Honest statement of what is and is not explained.** The negative RB tier effect is *located* precisely -- it is entirely a games-played channel, with per-game production flat -- and roughly a quarter of the games gap is accounted for by projected top-10 offences carrying deeper backfields (-0.93 games per extra drafted RB, p = .0001). The remaining ~0.8 games is not explained by role, age or experience, and is not late-season rest. Given it is a p = .04 estimate that does not clear its own multiplicity screen, the residual is most plausibly noise; it is recorded as open rather than given a story.
## §N11 What §N establishes

1. **The owner's unconditional number, as asked:** 47.5% of RB1-12 finishers 2015-2024 played in a realized top-10 scoring offence (Wilson [38.8, 56.4], n = 120) against a 31.2% even-spread base rate; RB13-24 is 34.2% [26.3, 43.0]; WR1-12 is 51.2% [42.4, 60.0]. So the raw pattern the owner has in mind is real and is not RB-specific: WRs show it at least as strongly.
2. **But it is a realized, not a projected, fact, and it is not actionable.** Using the preseason-knowable projection instead, the RB1-12 share falls to 31.7% [24.0, 40.4] -- indistinguishable from the 31.2% base rate. The entire gap is hindsight: teams are top-10 offences partly *because* their RB finished top-12.
3. **Quantified: the realized arm is mechanically contaminated.** Own touchdowns are 18.0% of team points for a top-12 RB. Rebuilt leave-own-player-out, all four realized contrasts collapse by 95-197% and none is significant. Same defect class as §28.3; the four BH survivors are withdrawn.
4. **Cost-conditioned, a projected top-10 offence does NOT raise P(top-12).** RB -3.6 pp (p = .31, MDE 13.9 pp), WR -1.1 pp (p = .77, MDE 15.0 pp). Both point estimates are negative.
5. **The one suggestive signal points the wrong way and resolves to availability.** RB top-24 on a projected top-10 offence is -6.6 pp (p = .041, negative in 8 of 10 seasons, sign held in holdout). It is fully mediated by games played: RBs on projected top-10 offences play 1.09 fewer games at the same draft cost (p = .0035), PPG is flat (-0.18, p = .54), and controlling for games the tier effect goes to -0.070 (p = .80). It does not clear BH against the four preseason-knowable tests.
6. **Consistency with §I3.** The mean channel is null here too (RB -0.18 PPG, p = .54; WR -0.31, p = .49, cost-conditioned), as §I3's 77%-priced result predicts. §N's separate question -- does environment reshape the tail at fixed price? -- answers no for the upper tail and weakly *negative* for RBs at the bust threshold.

**Nothing enters theta*.** No contrast survives the de-contaminated family; the availability finding is a restatement, at team level, of what §A and §L already own.

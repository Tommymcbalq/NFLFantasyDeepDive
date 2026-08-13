"""§P1 support — what a late-round pick has historically been worth, descriptively.

The deep boards are market-anchored in the tail (§P4), so board_value there is a monotone
function of ADP and the tail RANKING carries no information beyond ADP.  The decision-
relevant objects in that region are therefore not ranks but (a) the tier level of the
isotonic curve and (b) the OUTCOME DISTRIBUTION at that price -- specifically its upper
tail, since a round-10 pick is bought for upside, not for its mean.

This is a descriptive read of the §P2 panel.  It is NOT an edge test: no covariate is used
to predict anything, no family is opened, nothing enters board_value.  The closed FDR
families stay closed.

Quantities, per ADP bucket, per position:
  - quantiles of realized PPG (p10/25/50/75/90) -- the p90 is the "upside" number;
  - P(positional PPG finish in the top 12 / 24 / 36), where the finish rank is computed
    among all players at that position with >= 8 included games that season, so the
    measure is era-neutral (16- vs 17-game seasons cannot move it);
  - P(bust) = P(fewer than 4 included games OR PPG < 8).
  Sub-floor rows (< 4 included games) are counted as MISSES throughout -- excluding them
  would inflate every tail rate by survivorship.  Wilson 95% intervals on every rate.

Output: results/sectionP_tail_hitrates.csv
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

warnings.simplefilter("ignore")
ROOT = Path("/Users/thomasmcnamee/NFL")
YEARS = list(range(2015, 2025))
BUCKETS = [0, 12, 24, 36, 60, 84, 110, 145]

COLS = ["player_id", "position", "season", "season_type", "targets", "carries",
        "fantasy_points_ppr"]
frames = []
for y in YEARS:
    d = pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{y}.csv",
                    usecols=COLS, low_memory=False)
    frames.append(d[d.season_type == "REG"])
wk = pd.concat(frames, ignore_index=True)
wk["touches"] = wk.carries.fillna(0) + wk.targets.fillna(0)

out = []
for pos, inc in [("WR", wk[(wk.position == "WR") & (wk.targets.fillna(0) >= 2)]),
                 ("RB", wk[(wk.position == "RB") & (wk.touches >= 2)])]:
    # league-wide positional finish table, >= 8 included games
    lg = (inc.groupby(["player_id", "season"])
          .agg(G=("fantasy_points_ppr", "size"),
               ppg=("fantasy_points_ppr", "mean")).reset_index())
    lg = lg[lg.G >= 8].copy()
    lg["fin"] = lg.groupby("season").ppg.rank(ascending=False, method="min")
    fin = lg.set_index(["player_id", "season"]).fin

    p = pd.read_csv(ROOT / f"results/market_prior_{pos.lower()}_deep.csv")
    p["finish"] = [fin.get((g, y), np.nan) for g, y in zip(p.pid, p.year)]
    p["bucket"] = pd.cut(p.adp, BUCKETS)
    print("\n" + "=" * 74 + f"\n{pos}: outcome distribution by ADP bucket "
          f"(n = {len(p)} board rows 2015-2024)\n" + "=" * 74)
    for bk, s in p.groupby("bucket", observed=True):
        n = len(s)
        ok = s[s.in_fit]
        q = ok.ppg.quantile([.10, .25, .50, .75, .90])
        r = dict(pos=pos, bucket=str(bk), n=n, n_in_fit=len(ok),
                 mean_adp=s.adp.mean(), mean_ppg=ok.ppg.mean(),
                 p10=q.loc[.10], p25=q.loc[.25], p50=q.loc[.50],
                 p75=q.loc[.75], p90=q.loc[.90])
        line = (f"  ADP {str(bk):>12}  n {n:3d}  mean ADP {s.adp.mean():6.1f}  "
                f"PPG p10/50/90 {q.loc[.10]:5.1f}/{q.loc[.50]:5.1f}/{q.loc[.90]:5.1f}")
        for k, thr in [("top12", 12), ("top24", 24), ("top36", 36)]:
            h = int((s.finish <= thr).sum())          # NaN finish counts as a miss
            lo, hi = proportion_confint(h, n, 0.05, method="wilson")
            r[f"P_{k}"], r[f"P_{k}_lo"], r[f"P_{k}_hi"] = h / n, lo, hi
            line += f"  P({k}) {h/n:.2f}[{lo:.2f},{hi:.2f}]"
        bust = int(((~s.in_fit) | (s.ppg < 8)).sum())
        lo, hi = proportion_confint(bust, n, 0.05, method="wilson")
        r["P_bust"], r["P_bust_lo"], r["P_bust_hi"] = bust / n, lo, hi
        line += f"  P(bust) {bust/n:.2f}"
        print(line)
        out.append(r)

pd.DataFrame(out).to_csv(ROOT / "results/sectionP_tail_hitrates.csv", index=False)
print("\nwrote results/sectionP_tail_hitrates.csv")

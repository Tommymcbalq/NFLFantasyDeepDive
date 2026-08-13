"""§M — the forward 2026 read: replacement levels and VORP from the MODEL board.

§M1 pre-specifies that the historical backtest uses realized replacement while the 2026
recommendation uses the model board.  These are never mixed.

Value scale for 2026 (preseason-knowable only):
  base   E[season total | position, ADP] from the isotonic curve fit on 2015-2024
  override for the modelled universes: WR theta* (results/valuation_2026_wr_*.csv) and
  RB board_value (results/valuation_rb_2026.csv), both PPG, converted to a season total
  by the position's mean games-played over 2015-2024 at the same ADP decile.
Replacement is then the (D_p + 1)-th best on that board, D_p from the 10-team structure
with the flex split taken from realized 2015-2024 usage; the 12-team frame is reported
alongside.

Output: results/sectionM_board_2026.csv
Rerun: python3 scripts/30_sectionM_2026_board.py
"""
import sys, os, glob
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sectionM_common as C

ROOT = C.ROOT
hist = pd.read_csv(f"{ROOT}/results/sectionM_player_vorp.csv")
b26 = pd.read_csv(f"{ROOT}/data/adp/adp_ppr_2026_all_20260809.csv")
b26 = b26[b26.position.isin(C.POSNS)].copy().sort_values("adp").reset_index(drop=True)

# base curve: E[season total | pos, ADP] on the full decade
b26["E_total"] = np.nan
for p in C.POSNS:
    f = hist[hist.pos == p]
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(f.adp, f.total)
    m = b26.position == p
    b26.loc[m, "E_total"] = iso.predict(b26.loc[m, "adp"])
# games scale: mean games by position over the decade, used to put a PPG model value on
# the same season-total scale as the base curve
gmean = hist.groupby("pos").games.mean()
print("mean games played 2015-2024 by position:\n", gmean.round(2).to_string())

wr = pd.read_csv(sorted(glob.glob(f"{ROOT}/results/valuation_2026_wr_*.csv"))[-1])
rb = pd.read_csv(f"{ROOT}/results/valuation_rb_2026.csv")
mv = {}
for _, r in wr.iterrows():
    mv[C.collapse_initials(C.norm_name(r.player))] = ("WR", r.theta_star * gmean["WR"])
for _, r in rb.iterrows():
    mv[C.collapse_initials(C.norm_name(r.player))] = ("RB", r.board_value * gmean["RB"])
b26["nname"] = b26.name.map(C.norm_name).replace(C.ALIASES).map(C.collapse_initials)
b26["model_value"] = [mv[n][1] if (n in mv and mv[n][0] == p) else np.nan
                      for n, p in zip(b26.nname, b26.position)]
print(f"model-board overrides matched: {b26.model_value.notna().sum()} of 60")
miss = [n for n in mv if n not in set(b26.nname)]
print("unmatched model-board players:", miss)
b26["value"] = b26.model_value.fillna(b26.E_total)

FR = {10: dict(QB=10, TE=10, RB=20, WR=20, FLEX=20), 12: dict(QB=12, TE=12, RB=24, WR=24, FLEX=24)}
flex_share = (pd.read_csv(f"{ROOT}/results/sectionM_diag_flex.csv")
              .query("alloc=='realized'").groupby("frame")[["flex_RB", "flex_WR"]].mean())
print("\nrealized flex split 2015-2024 (used for 2026 demand):\n", flex_share.round(2).to_string())

out = []
for F, d in FR.items():
    fRB = int(round(flex_share.loc[F, "flex_RB"]))
    D = {"QB": d["QB"], "TE": d["TE"], "RB": d["RB"] + fRB, "WR": d["WR"] + d["FLEX"] - fRB}
    R = {}
    for p in C.POSNS:
        v = np.sort(b26[b26.position == p].value.values)[::-1]
        R[p] = v[D[p]] if len(v) > D[p] else v[-1]
    print(f"\nframe {F}-team  demand {D}\n  replacement (season pts) "
          f"{ {k: round(v,1) for k,v in R.items()} }")
    t = b26.copy()
    t["frame"] = F
    t["demand"] = t.position.map(D)
    t["R"] = t.position.map(R)
    t["VORP"] = t.value - t.R
    t["vorp_rank"] = t.VORP.rank(ascending=False, method="min").astype(int)
    out.append(t)
B = pd.concat(out, ignore_index=True)
B.to_csv(f"{ROOT}/results/sectionM_board_2026.csv", index=False)

for F in (10, 12):
    t = B[B.frame == F].sort_values("VORP", ascending=False).head(30)
    print(f"\n=== 2026 top-30 by VORP, {F}-team frame ===")
    print(t[["vorp_rank", "name", "position", "team", "adp", "value", "R", "VORP"]]
          .to_string(index=False, float_format=lambda x: f"{x:.1f}"))
for F in (10, 12):
    t = B[B.frame == F]
    print(f"\n-- {F}-team: best TE and QB by VORP, with their ADP neighbourhood --")
    for p in ("TE", "QB"):
        g = t[t.position == p].nlargest(4, "VORP")
        for _, r in g.iterrows():
            nb = t[(t.position.isin(["RB", "WR"])) & ((t.adp - r.adp).abs() <= 6)]
            print(f"  {p} {r['name']:22s} adp {r.adp:5.1f}  VORP {r.VORP:6.1f}   "
                  f"mean RB/WR VORP within +-6 picks: {nb.VORP.mean():6.1f}  "
                  f"(n={len(nb)})  premium {r.VORP - nb.VORP.mean():+6.1f}")

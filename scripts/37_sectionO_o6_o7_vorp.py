"""§O6 replacement level and VORP under the owner's league / §O7 streaming baseline.

EDA_PLAN6.md §O6-§O7.  Puts TE and QB on ONE scale with the RB/WR curves from §M1, in
BOTH league frames, against ALL THREE replacement baselines from §M1's bracket.

League (fixed): 10 teams, PPR, 1QB / 2RB / 2WR / 1TE / 2FLEX(RB-WR) / 1DST, no kicker.
Starting demand 10 QB and 10 TE.  DST out of scope.
FRAME GAP, named on every number: the ADP source is FFC 12-team; the owner's league is
10-team.  Both frames are reported and every table carries its frame.

Baselines (§M1's bracket, re-used unchanged; §O7 requires all three):
  R_exp   (D+1)-th best by PRESEASON EXPECTATION (LOSO isotonic ADP->points).  Draft-only.
  R_real  (D+1)-th best by REALIZED season total.  Season foresight.
  R_week  worst weekly starter under league-optimal weekly allocation x 17.  Weekly
          foresight = clairvoyant streaming.
R_exp <= R_real <= R_week by construction.

Part A (historical, 2015-2024): VORP by draft slot, four positions, three baselines, two
frames -> results/vorp_all_positions.csv.
Part B (2026 forward): the model board.  Per §M1's pre-specification the forward read uses
the MODEL board, never realized outcomes.  TE and QB values come from §O5's boards, which
the §O5 honesty clause left MARKET-ANCHORED (board_value = m(ADP)); WR uses theta* and RB
uses its §G board_value, exactly as §M's 2026 read did.  The board's own replacement is an
R_exp object by construction (a board of expectations contains no order-statistic
selection), so the R_real and R_week columns are formed by adding the historical
position-specific bracket gaps.  That transfer is stated, not hidden.

Outputs: results/vorp_all_positions.csv, results/sectionO_board_2026_vorp.csv,
         results/sectionO_premium_by_baseline.csv
Rerun: python3 scripts/37_sectionO_o6_o7_vorp.py
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sectionM_common as MC  # noqa: E402

ROOT = MC.ROOT
YEARS = MC.YEARS
POSNS = MC.POSNS
BASES = ["R_exp", "R_real", "R_week"]

vp = pd.read_csv(f"{ROOT}/results/sectionM_player_vorp.csv")
BR = pd.read_csv(f"{ROOT}/results/sectionM_replacement_bracket.csv")

print("=" * 78)
print("§O6/§O7  replacement bracket (from §M1), season-total PPR, means 2015-2024")
print("=" * 78)
print(BR.groupby(["frame", "pos"])[["demand"] + BASES].mean().round(1).to_string())
g = BR.groupby(["frame", "pos"])[BASES].mean()
g["bracket_width"] = g.R_week - g.R_exp
print("\nbracket width (R_week - R_exp):")
print(g.bracket_width.round(1).to_string())

# ================================================================= Part A
rows = []
for F in (10, 12):
    Rm = {(r.year, r.pos): {b: getattr(r, b) for b in BASES}
          for r in BR[BR.frame == F].itertuples()}
    v = vp.copy()
    for b in BASES:
        v[b] = [r.total - Rm[(r.year, r.pos)][b] for r in v.itertuples()]
    for unit, col, cap in (("positional_adp_rank", "posrank_adp", 24),
                           ("draft_round", f"round{F}", 14)):
        for (p, s), grp in v.groupby(["pos", col]):
            if s > cap:
                continue
            for b in BASES:
                x = grp[b].values
                # season-clustered SE on the per-season mean
                per = grp.groupby("year")[b].mean()
                rows.append(dict(frame=F, unit=unit, pos=p, slot=int(s), baseline=b,
                                 n=len(x), n_seasons=len(per), mean=x.mean(),
                                 se_season=per.std(ddof=1) / np.sqrt(len(per))
                                 if len(per) > 1 else np.nan,
                                 median=float(np.median(x))))
A = pd.DataFrame(rows)
A.to_csv(f"{ROOT}/results/vorp_all_positions.csv", index=False)

for F in (10, 12):
    for b in BASES:
        t = A[(A.frame == F) & (A.unit == "draft_round") & (A.baseline == b)]
        pt = t.pivot_table(index="slot", columns="pos", values="mean")
        print(f"\n=== VORP by {F}-team draft round, baseline {b} (season totals) ===")
        print(pt.round(1).to_string())

print("\n=== VORP by positional ADP rank, 10-team, baseline R_real ===")
t = A[(A.frame == 10) & (A.unit == "positional_adp_rank") & (A.baseline == "R_real")]
print(t.pivot_table(index="slot", columns="pos", values="mean").head(12).round(1).to_string())

# ---- the cross-positional read the owner needs: at a given round, who is best? ----
print("\n=== §O6 deliverable: best position by mean VORP at each 10-team draft round ===")
for b in BASES:
    t = A[(A.frame == 10) & (A.unit == "draft_round") & (A.baseline == b)]
    pt = t.pivot_table(index="slot", columns="pos", values="mean")
    best = pt.idxmax(axis=1)
    print(f"  {b:7s}: " + " ".join(f"R{int(i)}={best[i]}" for i in pt.index if i <= 14))

# ================================================================= §O7 premium tests
out = []
for F in (10, 12):
    Rm = {(r.year, r.pos): {b: getattr(r, b) for b in BASES}
          for r in BR[BR.frame == F].itertuples()}
    for b in BASES:
        v = vp.copy()
        v["V"] = [r.total - Rm[(r.year, r.pos)][b] for r in v.itertuples()]
        for p in ("TE", "QB"):
            for kmax in (3, 5, 12):
                per = []
                for y in YEARS:
                    gy = v[v.year == y]
                    tg = gy[(gy.pos == p) & (gy.posrank_adp <= kmax)]
                    ds = []
                    for t_ in tg.itertuples():
                        nb = gy[(gy.pos.isin(["RB", "WR"]))
                                & ((gy.adp - t_.adp).abs() <= 6)]
                        if len(nb) >= 2:
                            ds.append(t_.V - nb.V.mean())
                    if ds:
                        per.append(np.mean(ds))
                a = np.array(per)
                se = a.std(ddof=1) / np.sqrt(len(a))
                out.append(dict(frame=F, baseline=b, pos=p, tier=f"{p}1-{kmax}",
                                n_seasons=len(a), mean_premium=a.mean(), se=se,
                                t=a.mean() / se,
                                p=2 * (1 - stats.t.cdf(abs(a.mean() / se), len(a) - 1)),
                                mde=2.802 * se, seasons_positive=int((a > 0).sum())))
O = pd.DataFrame(out)
O.to_csv(f"{ROOT}/results/sectionO_premium_by_baseline.csv", index=False)
print("\n=== §O7 TE/QB premium over the RB/WR at the same ADP (+-6 picks), "
      "every baseline, both frames ===")
print(O.round(2).to_string(index=False))

# ================================================================= Part B: 2026 board
print("\n" + "=" * 78)
print("§O6 forward read — the 2026 board, one scale, both frames, three baselines")
print("=" * 78)
b26 = pd.read_csv(f"{ROOT}/data/adp/adp_ppr_2026_all_20260809.csv")
b26 = b26[b26.position.isin(POSNS)].copy().sort_values("adp").reset_index(drop=True)

for p in POSNS:
    f = vp[vp.pos == p]
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(f.adp, f.total)
    b26.loc[b26.position == p, "E_total"] = iso.predict(b26.loc[b26.position == p, "adp"])
gmean = vp.groupby("pos").games.mean()
print("mean games played 2015-2024 by position:", gmean.round(2).to_dict())

# model-board overrides
mv = {}
wr = pd.read_csv(sorted(glob.glob(f"{ROOT}/results/valuation_2026_wr_*.csv"))[-1])
rb = pd.read_csv(f"{ROOT}/results/valuation_rb_2026.csv")
te = pd.read_csv(f"{ROOT}/results/valuation_te_2026.csv")
qb = pd.read_csv(f"{ROOT}/results/valuation_qb_2026.csv")
for _, r in wr.iterrows():
    mv[MC.collapse_initials(MC.norm_name(r.player))] = ("WR", r.theta_star * gmean["WR"])
for _, r in rb.iterrows():
    mv[MC.collapse_initials(MC.norm_name(r.player))] = ("RB", r.board_value * gmean["RB"])
for _, r in te.iterrows():
    mv[MC.collapse_initials(MC.norm_name(r.player))] = ("TE", r.board_value * gmean["TE"])
for _, r in qb.iterrows():
    mv[MC.collapse_initials(MC.norm_name(r.player))] = ("QB", r.board_value * gmean["QB"])
b26["nname"] = b26.name.map(MC.norm_name).replace(MC.ALIASES).map(MC.collapse_initials)
b26["model_value"] = [mv[n][1] if (n in mv and mv[n][0] == p) else np.nan
                      for n, p in zip(b26.nname, b26.position)]
print(f"model-board overrides matched: {b26.model_value.notna().sum()} "
      f"(WR {len(wr)} + RB {len(rb)} + TE {len(te)} + QB {len(qb)} = "
      f"{len(wr)+len(rb)+len(te)+len(qb)})")
missing = [n for n in mv if n not in set(b26.nname)]
print("unmatched model-board players:", missing)
b26["value"] = b26.model_value.fillna(b26.E_total)

FR = {10: dict(QB=10, TE=10, RB=20, WR=20, FLEX=20),
      12: dict(QB=12, TE=12, RB=24, WR=24, FLEX=24)}
flex_share = (pd.read_csv(f"{ROOT}/results/sectionM_diag_flex.csv")
              .query("alloc=='realized'").groupby("frame")[["flex_RB", "flex_WR"]].mean())
gap = BR.groupby(["frame", "pos"])[BASES].mean()
gap["d_real"] = gap.R_real - gap.R_exp
gap["d_week"] = gap.R_week - gap.R_exp
print("\nhistorical bracket gaps added to the 2026 board's R_exp "
      "(the transfer, stated explicitly):")
print(gap[["d_real", "d_week"]].round(1).to_string())

outB = []
for F, d in FR.items():
    fRB = int(round(flex_share.loc[F, "flex_RB"]))
    D = {"QB": d["QB"], "TE": d["TE"], "RB": d["RB"] + fRB,
         "WR": d["WR"] + d["FLEX"] - fRB}
    Rexp = {}
    for p in POSNS:
        v = np.sort(b26[b26.position == p].value.values)[::-1]
        Rexp[p] = v[D[p]] if len(v) > D[p] else v[-1]
    print(f"\nframe {F}-team  demand {D}")
    print(f"  R_exp (board)  { {k: round(v,1) for k,v in Rexp.items()} }")
    t = b26.copy(); t["frame"] = F
    t["demand"] = t.position.map(D)
    for b in BASES:
        add = {p: (0.0 if b == "R_exp" else gap.loc[(F, p), "d_real" if b == "R_real"
                                                    else "d_week"]) for p in POSNS}
        t[b] = t.position.map({p: Rexp[p] + add[p] for p in POSNS})
        t[f"VORP_{b}"] = t.value - t[b]
    print("  R_real (transferred)",
          {p: round(t[t.position == p][BASES[1]].iat[0], 1) for p in POSNS})
    print("  R_week (transferred)",
          {p: round(t[t.position == p][BASES[2]].iat[0], 1) for p in POSNS})
    outB.append(t)
B26 = pd.concat(outB, ignore_index=True)
B26["vorp_rank_real"] = B26.groupby("frame").VORP_R_real.rank(ascending=False,
                                                              method="min").astype(int)
B26.to_csv(f"{ROOT}/results/sectionO_board_2026_vorp.csv", index=False)

for F in (10, 12):
    t = B26[B26.frame == F]
    print(f"\n=== 2026 top-25 by VORP (baseline R_real), {F}-team frame ===")
    print(t.nsmallest(25, "vorp_rank_real")[
        ["vorp_rank_real", "name", "position", "team", "adp", "value",
         "VORP_R_exp", "VORP_R_real", "VORP_R_week"]]
        .to_string(index=False, float_format=lambda x: f"{x:.1f}"))

for F in (10, 12):
    t = B26[B26.frame == F]
    print(f"\n-- {F}-team frame: 2026 TE and QB vs the RB/WR at the same ADP (+-6) --")
    for p in ("TE", "QB"):
        for _, r in t[t.position == p].nlargest(4, "VORP_R_real").iterrows():
            nb = t[(t.position.isin(["RB", "WR"])) & ((t.adp - r.adp).abs() <= 6)]
            s = f"  {p} {r['name']:20s} adp {r.adp:5.1f} | "
            for b in BASES:
                s += (f"{b}: {r['VORP_' + b]:+6.1f} vs {nb['VORP_' + b].mean():+6.1f} "
                      f"(prem {r['VORP_' + b] - nb['VORP_' + b].mean():+6.1f})  ")
            print(s + f"[n_nb={len(nb)}]")

print("\nwrote vorp_all_positions.csv, sectionO_board_2026_vorp.csv, "
      "sectionO_premium_by_baseline.csv")

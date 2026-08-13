"""§M1 — the replacement bracket, and why every elite-QB/TE claim depends on it.

Three baselines for the same position, same demand D_p, same seasons.  They differ only
in how much in-season information the replacement player is assumed to have:

  R_exp   (D_p+1)-th best by PRESEASON EXPECTATION (LOSO isotonic on ADP).  What you can
          plan to draft.  No in-season management.
  R_real  (D_p+1)-th best by REALIZED season total.  Perfect preseason foresight, or a
          season's worth of waiver-wire hindsight.
  R_week  worst weekly starter under a league-optimal weekly allocation x 17.  Perfect
          WEEKLY foresight (clairvoyant streaming).

R_exp <= R_real <= R_week by construction (each adds information).  The WIDTH of the
bracket is position-specific, and it is the whole content of the elite-TE / elite-QB
question: a premium computed against R_exp can vanish against R_week.

Output: results/sectionM_replacement_bracket.csv
Rerun: python3 scripts/28c_sectionM_replacement_bracket.py
"""
import sys, os
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sectionM_common as C

ROOT = C.ROOT
rep = pd.read_csv(f"{ROOT}/results/replacement_levels.csv")
avc = pd.read_csv(f"{ROOT}/results/sectionM_adp_value_curve.csv")
vp = pd.read_csv(f"{ROOT}/results/sectionM_player_vorp.csv")

rows = []
for F in (10, 12):
    for y in C.YEARS:
        r = rep[(rep.frame == F) & (rep.year == y)].set_index("pos")
        for p in C.POSNS:
            D = int(r.loc[p, "demand"])
            e = np.sort(avc[(avc.year == y) & (avc.pos == p)].E_total.values)[::-1]
            rows.append(dict(frame=F, year=y, pos=p, demand=D,
                             R_exp=e[D] if len(e) > D else e[-1],
                             R_real=r.loc[p, "R_static_total"],
                             R_week=r.loc[p, "R_marg_season17"]))
B = pd.DataFrame(rows)
B["gap_exp_real"] = B.R_real - B.R_exp
B["gap_real_week"] = B.R_week - B.R_real
B.to_csv(f"{ROOT}/results/sectionM_replacement_bracket.csv", index=False)
print("=== replacement bracket, season means (season-total PPR) ===")
print(B.groupby(["frame", "pos"])[["demand", "R_exp", "R_real", "R_week",
                                   "gap_exp_real", "gap_real_week"]].mean().round(1).to_string())

# the premium under each baseline
print("\n=== TE1-5 / QB1-5 premium over RB/WR at the same ADP, under each baseline ===")
print("(uses each player's REALIZED season total minus the stated replacement)")
out = []
for F in (10, 12):
    for base in ("R_exp", "R_real", "R_week"):
        Rm = {(r.year, r.pos): getattr(r, base) for r in B[B.frame == F].itertuples()}
        v = vp.copy()
        v["V"] = [r.total - Rm[(r.year, r.pos)] for r in v.itertuples()]
        for p in ("TE", "QB"):
            per = []
            for y in C.YEARS:
                g = v[v.year == y]
                tg = g[(g.pos == p) & (g.posrank_adp <= 5)]
                ds = [t.V - g[(g.pos.isin(["RB", "WR"])) & ((g.adp - t.adp).abs() <= 6)].V.mean()
                      for t in tg.itertuples()
                      if len(g[(g.pos.isin(["RB", "WR"])) & ((g.adp - t.adp).abs() <= 6)]) >= 2]
                if ds:
                    per.append(np.mean(ds))
            a = np.array(per)
            se = a.std(ddof=1) / np.sqrt(len(a))
            out.append(dict(frame=F, baseline=base, pos=p, mean_premium=a.mean(), se=se,
                            t=a.mean() / se, p=2 * (1 - stats.t.cdf(abs(a.mean() / se), 9)),
                            mde=2.802 * se, seasons_positive=int((a > 0).sum())))
O = pd.DataFrame(out)
O.to_csv(f"{ROOT}/results/sectionM_premium_by_baseline.csv", index=False)
print(O.round(2).to_string(index=False))

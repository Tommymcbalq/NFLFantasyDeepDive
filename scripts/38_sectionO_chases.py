"""§O anomaly chases, run after §O6/§O7 and reported whether or not they flatter it.

Chase 1 — IS THE "ELITE-TE PREMIUM" ELITE?  §O7 returns TE1-3 +18.1, TE1-5 +18.5 and
TE1-12 +18.7 against R_real: essentially identical.  A premium that does not decay with
positional rank is not a statement about elite TEs; it is a level shift, i.e. a statement
about R_TE relative to R_RB/R_WR.  Tested by cutting the premium into DISJOINT rank bands
(1-3, 4-6, 7-12, 13+) instead of nested ones, which is the only way to see decay.

Chase 2 — SUPPORT.  The historical elite-TE premium was measured at the ADPs the market
charged 2015-2024.  The 2026 board prices its TE1 at 35.3.  If the historical TE1-5 sat at
much shorter ADPs, the premium is being read outside the price region where it was
measured.  Reported as an exploratory sensitivity; the §O7 headline is NOT replaced.

Chase 3 — same two for QB.

Output: results/sectionO_chases.csv
Rerun: python3 scripts/38_sectionO_chases.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sectionM_common as MC  # noqa: E402

ROOT = MC.ROOT
YEARS = MC.YEARS
BASES = ["R_exp", "R_real", "R_week"]
vp = pd.read_csv(f"{ROOT}/results/sectionM_player_vorp.csv")
BR = pd.read_csv(f"{ROOT}/results/sectionM_replacement_bracket.csv")
uni = pd.read_csv(f"{ROOT}/results/sectionO_universe_2026.csv")


def premium(v, p, sel, adp_lo=None, adp_hi=None):
    per = []
    for y in YEARS:
        gy = v[v.year == y]
        tg = gy[(gy.pos == p) & sel(gy)]
        if adp_lo is not None:
            tg = tg[(tg.adp >= adp_lo) & (tg.adp <= adp_hi)]
        ds = []
        for t in tg.itertuples():
            nb = gy[(gy.pos.isin(["RB", "WR"])) & ((gy.adp - t.adp).abs() <= 6)]
            if len(nb) >= 2:
                ds.append(t.V - nb.V.mean())
        if ds:
            per.append(np.mean(ds))
    a = np.array(per)
    if len(a) < 3:
        return None
    se = a.std(ddof=1) / np.sqrt(len(a))
    return dict(n_seasons=len(a), mean=a.mean(), se=se, t=a.mean() / se,
                p=2 * (1 - stats.t.cdf(abs(a.mean() / se), len(a) - 1)),
                mde=2.802 * se, seasons_positive=int((a > 0).sum()))


rows = []
BANDS = [("1-3", 1, 3), ("4-6", 4, 6), ("7-12", 7, 12), ("13+", 13, 99)]
for F in (10, 12):
    Rm = {(r.year, r.pos): {b: getattr(r, b) for b in BASES}
          for r in BR[BR.frame == F].itertuples()}
    for b in BASES:
        v = vp.copy()
        v["V"] = [r.total - Rm[(r.year, r.pos)][b] for r in v.itertuples()]
        for p in ("TE", "QB"):
            for lbl, lo, hi in BANDS:
                r = premium(v, p, lambda g, lo=lo, hi=hi:
                            (g.posrank_adp >= lo) & (g.posrank_adp <= hi))
                if r:
                    rows.append(dict(chase="1_disjoint_bands", frame=F, baseline=b,
                                     pos=p, band=lbl, **r))
C = pd.DataFrame(rows)

print("=" * 78)
print("CHASE 1 — DISJOINT positional-rank bands.  A real 'elite' premium must DECAY")
print("across these bands; a flat profile means the effect is a replacement-level shift.")
print("=" * 78)
for b in BASES:
    print(f"\n--- baseline {b}, 10-team frame ---")
    t = C[(C.frame == 10) & (C.baseline == b) & (C.chase == "1_disjoint_bands")]
    print(t[["pos", "band", "n_seasons", "mean", "se", "p", "mde",
             "seasons_positive"]].round(2).to_string(index=False))

# formal decay test: TE band 1-3 minus band 7-12, season-clustered
print("\n--- formal decay contrast: band(1-3) - band(7-12), season-clustered t(9) ---")
dec = []
for F in (10, 12):
    Rm = {(r.year, r.pos): {b: getattr(r, b) for b in BASES}
          for r in BR[BR.frame == F].itertuples()}
    for b in BASES:
        v = vp.copy()
        v["V"] = [r.total - Rm[(r.year, r.pos)][b] for r in v.itertuples()]
        for p in ("TE", "QB"):
            per = []
            for y in YEARS:
                gy = v[v.year == y]
                vals = {}
                for lbl, lo, hi in (("top", 1, 3), ("mid", 7, 12)):
                    tg = gy[(gy.pos == p) & (gy.posrank_adp >= lo) & (gy.posrank_adp <= hi)]
                    ds = [t.V - gy[(gy.pos.isin(["RB", "WR"]))
                                   & ((gy.adp - t.adp).abs() <= 6)].V.mean()
                          for t in tg.itertuples()
                          if len(gy[(gy.pos.isin(["RB", "WR"]))
                                    & ((gy.adp - t.adp).abs() <= 6)]) >= 2]
                    vals[lbl] = np.mean(ds) if ds else np.nan
                if not any(np.isnan(list(vals.values()))):
                    per.append(vals["top"] - vals["mid"])
            a = np.array(per)
            se = a.std(ddof=1) / np.sqrt(len(a))
            dec.append(dict(chase="1_decay_contrast", frame=F, baseline=b, pos=p,
                            band="top(1-3)-mid(7-12)", n_seasons=len(a), mean=a.mean(),
                            se=se, t=a.mean() / se,
                            p=2 * (1 - stats.t.cdf(abs(a.mean() / se), len(a) - 1)),
                            mde=2.802 * se, seasons_positive=int((a > 0).sum())))
D = pd.DataFrame(dec)
print(D[D.frame == 10][["baseline", "pos", "mean", "se", "p", "mde",
                        "seasons_positive"]].round(2).to_string(index=False))

# ------------------------------------------------------------------ chase 2: support
print("\n" + "=" * 78)
print("CHASE 2 — is the 2026 board priced where the historical premium was measured?")
print("=" * 78)
sup = []
for p in ("TE", "QB"):
    h = vp[vp.pos == p]
    for k in (1, 3, 5):
        hist = h[h.posrank_adp <= k].groupby("year").adp.max()
        u26 = uni[uni.pos == p].nsmallest(k, "adp").adp.max()
        print(f"[{p}] historical ADP of the {k}th-best-priced {p}: "
              f"median {hist.median():.1f}, range {hist.min():.1f}..{hist.max():.1f}; "
              f"2026 = {u26:.1f}")
        sup.append(dict(chase="2_support", pos=p, k=k, hist_median=hist.median(),
                        hist_min=hist.min(), hist_max=hist.max(), adp_2026=u26,
                        seasons_cheaper_than_2026=int((hist > u26).sum())))
        print(f"      seasons in which the market priced its {p}{k} MORE cheaply "
              f"than 2026 does: {int((hist > u26).sum())}/10")
S = pd.DataFrame(sup)

# exploratory: restrict the historical premium to the 2026-like ADP window
print("\nexploratory sensitivity — historical premium restricted to the ADP window the")
print("2026 board actually occupies (post-hoc; the §O7 headline is unchanged):")
win = {}
for p in ("TE", "QB"):
    u = uni[uni.pos == p].nsmallest(5, "adp")
    win[p] = (u.adp.min() * 0.8, u.adp.max() * 1.2)
    print(f"  {p} window: ADP {win[p][0]:.0f}..{win[p][1]:.0f} "
          f"(2026 {p}1-5 span {u.adp.min():.1f}..{u.adp.max():.1f})")
wrows = []
for F in (10, 12):
    Rm = {(r.year, r.pos): {b: getattr(r, b) for b in BASES}
          for r in BR[BR.frame == F].itertuples()}
    for b in BASES:
        v = vp.copy()
        v["V"] = [r.total - Rm[(r.year, r.pos)][b] for r in v.itertuples()]
        for p in ("TE", "QB"):
            r = premium(v, p, lambda g: g.posrank_adp <= 99,
                        adp_lo=win[p][0], adp_hi=win[p][1])
            if r:
                wrows.append(dict(chase="2_adp_window", frame=F, baseline=b, pos=p,
                                  band=f"adp {win[p][0]:.0f}-{win[p][1]:.0f}", **r))
W = pd.DataFrame(wrows)
print(W[W.frame == 10][["baseline", "pos", "band", "n_seasons", "mean", "se", "p",
                        "mde", "seasons_positive"]].round(2).to_string(index=False))

pd.concat([C, D, W, S], ignore_index=True).to_csv(f"{ROOT}/results/sectionO_chases.csv",
                                                  index=False)
print("\nwrote sectionO_chases.csv")

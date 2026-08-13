"""§M1 diagnostics — chasing two things that must be explained before §M2 is read:

  (A) the elite-TE premium changes SIGN between vorp_total (static replacement, linear)
      and vorp_weekly (marginal replacement, positive part).  Decomposed 2x2:
      {static R, marginal R} x {linear, positive-part}.
  (B) the RB/WR VORP curves never cross.  Is that the FLEX arbitrage (a 2-flex league
      forces R_RB = R_WR at the common flex cutoff) or something else?  Sensitivity:
      flex allocated 50/50 by fiat instead of by realized usage.

Outputs: results/sectionM_diag_te.csv, results/sectionM_diag_flex.csv
Rerun: python3 scripts/28b_sectionM_diagnostics.py
"""
import sys, os
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sectionM_common as C

ROOT = C.ROOT
YEARS = C.YEARS

wk = C.load_weekly()
panel, _, _ = C.build_panel(wk)
season_pos = (wk.groupby(["player_id", "season"]).position
                .agg(lambda s: s.mode().iat[0] if len(s.mode()) else np.nan)
                .rename("pos_nfl").reset_index())
tot = (wk.groupby(["player_id", "season"])
         .agg(total=("fantasy_points_ppr", "sum"), games=("fantasy_points_ppr", "size"))
         .reset_index().merge(season_pos, on=["player_id", "season"]))
bp = panel.rename(columns={"pid": "player_id", "year": "season"})[
    ["player_id", "season", "pos"]].drop_duplicates(["player_id", "season"])
tot = tot.merge(bp.rename(columns={"pos": "pos_board"}), on=["player_id", "season"], how="left")
tot["pos_eff"] = tot.pos_board.fillna(tot.pos_nfl)
tot = tot[tot.pos_eff.isin(C.POSNS)]

rep = pd.read_csv(f"{ROOT}/results/replacement_levels.csv")
Rst = {(r.year, r.frame, r.pos): r.R_static_total for r in rep.itertuples()}
Rmg = {(r.year, r.frame, r.pos): r.R_marg_week for r in rep.itertuples()}

wkmat = {y: C.weekly_matrix(wk, y, tot[tot.season == y].player_id.tolist()) for y in YEARS}

# ---------------------------------------------------------- (A) 2x2 decomposition
rows = []
for r in panel.itertuples():
    m = wkmat[r.year]
    v = m.loc[r.pid].values if r.pid in m.index else np.full(17, np.nan)
    v = np.nan_to_num(v, nan=0.0)
    d = dict(year=r.year, pos=r.pos, adp=r.adp, name=r.name, total=v.sum(),
             n_active=int((v > 0).sum()))
    for F in (10, 12):
        rs, rm = Rst[(r.year, F, r.pos)] / 17.0, Rmg[(r.year, F, r.pos)]
        d[f"lin_static_{F}"] = float((v - rs).sum())
        d[f"pos_static_{F}"] = float(np.maximum(v - rs, 0).sum())
        d[f"lin_marg_{F}"] = float((v - rm).sum())
        d[f"pos_marg_{F}"] = float(np.maximum(v - rm, 0).sum())
    rows.append(d)
D = pd.DataFrame(rows)
D["posrank_adp"] = D.groupby(["year", "pos"]).adp.rank(method="first").astype(int)

out = []
for F in (10, 12):
    for cell in ("lin_static", "pos_static", "lin_marg", "pos_marg"):
        col = f"{cell}_{F}"
        for p in ("TE", "QB"):
            per = []
            for y in YEARS:
                g = D[D.year == y]
                tg = g[(g.pos == p) & (g.posrank_adp <= 5)]
                ds = []
                for t in tg.itertuples():
                    nb = g[(g.pos.isin(["RB", "WR"])) & ((g.adp - t.adp).abs() <= 6)]
                    if len(nb) >= 2:
                        ds.append(getattr(t, col) - nb[col].mean())
                if ds:
                    per.append(np.mean(ds))
            a = np.array(per)
            se = a.std(ddof=1) / np.sqrt(len(a))
            out.append(dict(frame=F, cell=cell, pos=p, tier=f"{p}1-5", mean=a.mean(),
                            se=se, t=a.mean() / se, p_val=2 * (1 - stats.t.cdf(abs(a.mean() / se), 9)),
                            seasons_pos=int((a > 0).sum())))
dte = pd.DataFrame(out)
dte.to_csv(f"{ROOT}/results/sectionM_diag_te.csv", index=False)
print("=== (A) TE1-5 / QB1-5 premium over RB/WR at same ADP: 2x2 decomposition ===")
print("rows differ ONLY in replacement baseline (static season-rate vs marginal weekly)")
print("and in whether sub-replacement weeks count negatively (lin) or are benched (pos).\n")
print(dte.round(2).to_string(index=False))

print("\n--- replacement levels per game, mean over seasons ---")
z = rep.groupby(["frame", "pos"])[["R_static_ppg", "R_marg_week"]].mean()
z["gap"] = z.R_marg_week - z.R_static_ppg
print(z.round(2).to_string())

# weekly churn: how much does the identity of the top-D player change week to week?
print("\n--- streaming gain: mean weekly Dth-best minus season-rate Dth-best (10-team) ---")
print("(the marginal definition assumes clairvoyant weekly streaming from the WHOLE pool)")
for p in C.POSNS:
    a = rep[(rep.frame == 10) & (rep.pos == p)]
    print(f"  {p}: static {a.R_static_ppg.mean():.2f}  marginal {a.R_marg_week.mean():.2f} "
          f"  streaming gain {a.R_marg_week.mean() - a.R_static_ppg.mean():+.2f} pts/wk")

# ---------------------------------------------------------- (B) flex arbitrage
fl = []
for y in YEARS:
    t = tot[tot.season == y]
    srt = {p: np.sort(t[t.pos_eff == p].total.values)[::-1] for p in C.POSNS}
    for F, base, nfl in ((10, 20, 20), (12, 24, 24)):
        rb, wr = srt["RB"], srt["WR"]
        rest = np.concatenate([np.stack([rb[base:], np.zeros(len(rb) - base)], 1),
                               np.stack([wr[base:], np.ones(len(wr) - base)], 1)])
        rest = rest[np.argsort(-rest[:, 0])][:nfl]
        fRB = int((rest[:, 1] == 0).sum())
        fl.append(dict(year=y, frame=F, alloc="realized", flex_RB=fRB, flex_WR=nfl - fRB,
                       R_RB=rb[base + fRB], R_WR=wr[base + nfl - fRB]))
        h = nfl // 2
        fl.append(dict(year=y, frame=F, alloc="50/50", flex_RB=h, flex_WR=h,
                       R_RB=rb[base + h], R_WR=wr[base + h]))
        fl.append(dict(year=y, frame=F, alloc="no_flex", flex_RB=0, flex_WR=0,
                       R_RB=rb[base], R_WR=wr[base]))
F2 = pd.DataFrame(fl)
F2["R_gap"] = F2.R_RB - F2.R_WR
F2.to_csv(f"{ROOT}/results/sectionM_diag_flex.csv", index=False)
print("\n=== (B) flex allocation and the RB/WR replacement gap (season totals) ===")
print(F2.groupby(["frame", "alloc"])[["flex_RB", "flex_WR", "R_RB", "R_WR", "R_gap"]]
      .mean().round(1).to_string())

# RB-WR VORP by round under each allocation, 10-team
print("\n--- RB - WR mean VORP by 10-team round, under three flex allocations ---")
pl = panel.merge(tot.rename(columns={"player_id": "pid", "season": "year"})[
    ["pid", "year", "total"]], on=["pid", "year"], how="left")
pl["total"] = pl.total.fillna(0.0)
pl["rnd"] = np.ceil(pl.adp / 10).astype(int)
tab = {}
for alloc in ("realized", "50/50", "no_flex"):
    Rm = {(r.year, "RB"): r.R_RB for r in F2[(F2.frame == 10) & (F2.alloc == alloc)].itertuples()}
    Rm.update({(r.year, "WR"): r.R_WR for r in F2[(F2.frame == 10) & (F2.alloc == alloc)].itertuples()})
    v = pl[pl.pos.isin(["RB", "WR"])].copy()
    v["vorp"] = [r.total - Rm[(r.year, r.pos)] for r in v.itertuples()]
    g = v[v.rnd <= 14].groupby(["rnd", "pos"]).vorp.mean().unstack()
    tab[alloc] = (g["RB"] - g["WR"]).round(1)
print(pd.DataFrame(tab).to_string())

# weekly-measure version of the cross
print("\n--- RB - WR mean VORP by 10-team round, WEEKLY positive-part measure ---")
D["rnd"] = np.ceil(D.adp / 10).astype(int)
v = D[(D.pos.isin(["RB", "WR"])) & (D.rnd <= 14)]
g = v.groupby(["rnd", "pos"])["pos_marg_10"].mean().unstack()
g2 = v.groupby(["rnd", "pos"])["pos_static_10"].mean().unstack()
print(pd.DataFrame({"pos_marg": (g["RB"] - g["WR"]).round(1),
                    "pos_static": (g2["RB"] - g2["WR"]).round(1)}).to_string())

# availability channel: games played by round
print("\n--- mean active weeks by round (the §L availability channel) ---")
print(D[D.rnd <= 10].groupby(["rnd", "pos"]).n_active.mean().unstack().round(1).to_string())
D.to_csv(f"{ROOT}/results/sectionM_diag_player.csv", index=False)
print("\ndone.")

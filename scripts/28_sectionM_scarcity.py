"""§M1 / §M4 — replacement level, VORP curves by draft slot, elite-TE / elite-QB reads.

Pre-registered in EDA_PLAN6.md §M1 and §M4. All quantities computed from *realized*
outcomes 2015-2024 (the correct counterfactual for evaluating a completed draft).

Frames, carried everywhere:
  10-team (the owner's league): demand 10 QB, 10 TE, 20 RB + 20 WR locked + 20 FLEX(RB/WR)
  12-team (the frame FFC ADP is drawn from): 12 QB, 12 TE, 24 + 24 + 24 FLEX

Replacement definitions (both pre-specified, both reported):
  static   : the (D_p + 1)-th best season total at position p, FLEX demand allocated to
             RB/WR by realized flex usage (top-20/24 of each locked, flex to the best of
             the remainder).
  marginal : the worst player actually started in a given week under a league-optimal
             weekly allocation over the whole player pool, averaged over weeks.

VORP definitions:
  vorp_total  = season total PPR  -  R_p^static
  vorp_weekly = sum_w max(pts_iw - R_{p,w}^marginal, 0)   [a week the player misses
                contributes 0, not a negative: you start a replacement instead. This is
                the single-player version of the §M3 weekly-optimal-lineup scorer.]

Outputs: results/vorp_curves.csv, results/replacement_levels.csv,
         results/sectionM_premium_tests.csv, results/sectionM_adp_value_curve.csv
Rerun: python3 scripts/28_sectionM_scarcity.py
"""
import sys, os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sectionM_common as C

ROOT = C.ROOT
YEARS = C.YEARS
FRAMES = {
    10: dict(nQB=10, nTE=10, nRB=20, nWR=20, nFLEX=20),
    12: dict(nQB=12, nTE=12, nRB=24, nWR=24, nFLEX=24),
}

print("loading weekly ...", flush=True)
wk = C.load_weekly()
panel, unmatched, ambig = C.build_panel(wk)
print(f"board rows {len(panel)}  unmatched {len(unmatched)}  ambiguous {len(ambig)}")
print(panel.pos.value_counts().to_dict())

# ---------------------------------------------------------------- season totals
# effective position: for a drafted player the BOARD position governs (the roster slot
# the pick buys); for everyone else nflverse position.  Same rule as §L.
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

# ---------------------------------------------------------------- replacement levels
rep_rows = []
static_R = {}     # (year, frame, pos) -> season-total replacement
flexmix = []
for y in YEARS:
    t = tot[tot.season == y]
    for F, d in FRAMES.items():
        srt = {p: np.sort(t[t.pos_eff == p].total.values)[::-1] for p in C.POSNS}
        # flex allocation from realized usage
        rb, wr = srt["RB"], srt["WR"]
        rest = np.concatenate([
            np.stack([rb[d["nRB"]:], np.zeros(len(rb) - d["nRB"])], 1),
            np.stack([wr[d["nWR"]:], np.ones(len(wr) - d["nWR"])], 1)])
        rest = rest[np.argsort(-rest[:, 0])][:d["nFLEX"]]
        fRB = int((rest[:, 1] == 0).sum()); fWR = int((rest[:, 1] == 1).sum())
        flexmix.append(dict(year=y, frame=F, flex_RB=fRB, flex_WR=fWR))
        D = {"QB": d["nQB"], "TE": d["nTE"], "RB": d["nRB"] + fRB, "WR": d["nWR"] + fWR}
        for p in C.POSNS:
            R = srt[p][D[p]] if len(srt[p]) > D[p] else 0.0
            static_R[(y, F, p)] = R
            rep_rows.append(dict(year=y, frame=F, pos=p, demand=D[p],
                                 R_static_total=R, R_static_ppg=R / 17.0))

# marginal / last-starter: league-optimal weekly allocation over the whole pool
pool = wk[wk.player_id.isin(set(tot.player_id))][
    ["player_id", "season", "week", "fantasy_points_ppr"]].merge(
    tot[["player_id", "season", "pos_eff"]], on=["player_id", "season"], how="left")
marg_R = {}
for y in YEARS:
    for F, d in FRAMES.items():
        got = {p: [] for p in C.POSNS}
        for w in range(1, 18):
            s = pool[(pool.season == y) & (pool.week == w)]
            if not len(s):
                continue
            v = {p: np.sort(s[s.pos_eff == p].fantasy_points_ppr.values)[::-1]
                 for p in C.POSNS}
            for p in ("QB", "TE"):
                n = d["nQB"] if p == "QB" else d["nTE"]
                if len(v[p]) >= n:
                    got[p].append(v[p][n - 1])
            rb, wr = v["RB"], v["WR"]
            lb, lw = d["nRB"], d["nWR"]
            if len(rb) <= lb or len(wr) <= lw:
                continue
            rest = np.concatenate([
                np.stack([rb[lb:], np.zeros(len(rb) - lb)], 1),
                np.stack([wr[lw:], np.ones(len(wr) - lw)], 1)])
            rest = rest[np.argsort(-rest[:, 0])][:d["nFLEX"]]
            # worst RB started = min over locked-RB and flex-RB slots
            fr = rest[rest[:, 1] == 0, 0]; fw = rest[rest[:, 1] == 1, 0]
            got["RB"].append(min(rb[lb - 1], fr.min()) if len(fr) else rb[lb - 1])
            got["WR"].append(min(wr[lw - 1], fw.min()) if len(fw) else wr[lw - 1])
        for p in C.POSNS:
            marg_R[(y, F, p)] = float(np.mean(got[p]))
for r in rep_rows:
    r["R_marg_week"] = marg_R[(r["year"], r["frame"], r["pos"])]
    r["R_marg_season17"] = r["R_marg_week"] * 17
rep = pd.DataFrame(rep_rows)
rep.to_csv(f"{ROOT}/results/replacement_levels.csv", index=False)
print("\nreplacement levels, season means:")
print(rep.groupby(["frame", "pos"])[
    ["demand", "R_static_total", "R_static_ppg", "R_marg_week"]].mean().round(2).to_string())
print("\nflex mix (mean over seasons):")
print(pd.DataFrame(flexmix).groupby("frame")[["flex_RB", "flex_WR"]].mean().round(2).to_string())

# ---------------------------------------------------------------- per-player VORP
wkmat = {}
for y in YEARS:
    pids = tot[tot.season == y].player_id.tolist()
    wkmat[y] = C.weekly_matrix(wk, y, pids)

vp = panel.merge(tot.rename(columns={"player_id": "pid", "season": "year"})[
    ["pid", "year", "total", "games"]], on=["pid", "year"], how="left")
vp["total"] = vp.total.fillna(0.0)
vp["games"] = vp.games.fillna(0).astype(int)
for F in FRAMES:
    vp[f"vorp_total_{F}"] = [r.total - static_R[(r.year, F, r.pos)] for r in vp.itertuples()]
    vw = []
    for r in vp.itertuples():
        m = wkmat[r.year]
        v = m.loc[r.pid].values if r.pid in m.index else np.full(17, np.nan)
        v = np.nan_to_num(v, nan=0.0)
        vw.append(float(np.maximum(v - marg_R[(r.year, F, r.pos)], 0).sum()))
    vp[f"vorp_weekly_{F}"] = vw
vp["round10"] = np.ceil(vp.adp / 10).astype(int)
vp["round12"] = np.ceil(vp.adp / 12).astype(int)
vp["posrank_adp"] = vp.groupby(["year", "pos"]).adp.rank(method="first").astype(int)
vp.to_csv(f"{ROOT}/results/sectionM_player_vorp.csv", index=False)

# ---------------------------------------------------------------- VORP curves
curves = []
for F in FRAMES:
    rc = f"round{F}"
    for (p, r), g in vp.groupby(["pos", rc]):
        if r > 15:
            continue
        for meas in ("total", "weekly"):
            x = g[f"vorp_{meas}_{F}"].values
            curves.append(dict(frame=F, unit="draft_round", pos=p, slot=r, n=len(x),
                               measure=meas, mean=x.mean(), sd=x.std(ddof=1) if len(x) > 1 else np.nan,
                               se=x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
                               median=np.median(x)))
    for (p, k), g in vp.groupby(["pos", "posrank_adp"]):
        if k > 30:
            continue
        for meas in ("total", "weekly"):
            x = g[f"vorp_{meas}_{F}"].values
            curves.append(dict(frame=F, unit="positional_adp_rank", pos=p, slot=k, n=len(x),
                               measure=meas, mean=x.mean(), sd=x.std(ddof=1) if len(x) > 1 else np.nan,
                               se=x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
                               median=np.median(x)))
cur = pd.DataFrame(curves)
cur.to_csv(f"{ROOT}/results/vorp_curves.csv", index=False)

print("\n=== VORP by positional ADP rank (season totals, 10-team frame) ===")
pt = cur[(cur.frame == 10) & (cur.unit == "positional_adp_rank") & (cur.measure == "total")]
print(pt.pivot_table(index="slot", columns="pos", values="mean").head(14).round(1).to_string())
print("\n=== VORP by positional ADP rank (weekly-optimal, 10-team frame) ===")
pw = cur[(cur.frame == 10) & (cur.unit == "positional_adp_rank") & (cur.measure == "weekly")]
print(pw.pivot_table(index="slot", columns="pos", values="mean").head(14).round(1).to_string())
print("\n=== VORP by 10-team draft round (season totals) ===")
pr = cur[(cur.frame == 10) & (cur.unit == "draft_round") & (cur.measure == "total")]
print(pr.pivot_table(index="slot", columns="pos", values="mean").round(1).to_string())

# ---------------------------------------------------------------- §M4 premium tests
# For each drafted QB/TE, the premium over the RB/WR you could have taken instead:
#   premium_i = VORP_i - mean VORP of RB/WR board rows with |adp - adp_i| <= W
# season-clustered t(9) on the per-season mean.
res = []
for F in FRAMES:
    for meas in ("total", "weekly"):
        col = f"vorp_{meas}_{F}"
        for p in ("TE", "QB"):
            for kmax, lbl in ((3, f"{p}1-3"), (5, f"{p}1-5"), (12, f"{p}1-12")):
                per_season = []
                for y in YEARS:
                    g = vp[(vp.year == y)]
                    tgt = g[(g.pos == p) & (g.posrank_adp <= kmax)]
                    alt = g[g.pos.isin(["RB", "WR"])]
                    ds = []
                    for r in tgt.itertuples():
                        nb = alt[(alt.adp - r.adp).abs() <= 6]
                        if len(nb) >= 2:
                            ds.append(getattr(r, col) - nb[col].mean())
                    if ds:
                        per_season.append(np.mean(ds))
                a = np.array(per_season)
                t, pv = stats.ttest_1samp(a, 0)
                se = a.std(ddof=1) / np.sqrt(len(a))
                res.append(dict(frame=F, measure=meas, tier=lbl, n_seasons=len(a),
                                mean_premium=a.mean(), se=se, t=t, p=pv,
                                ci_lo=a.mean() - 2.262 * se, ci_hi=a.mean() + 2.262 * se,
                                mde=2.802 * se, seasons_positive=int((a > 0).sum())))
prem = pd.DataFrame(res)
prem.to_csv(f"{ROOT}/results/sectionM_premium_tests.csv", index=False)
print("\n=== §M4(1)/(2) elite-TE and elite-QB premium vs RB/WR at the same ADP (+-6 picks) ===")
print(prem[prem.measure == "total"].round(2).to_string(index=False))
print(prem[prem.measure == "weekly"].round(2).to_string(index=False))

# ---------------------------------------------------------------- §M4(3) RB/WR crossing
cross = []
for F in FRAMES:
    for meas in ("total", "weekly"):
        col = f"vorp_{meas}_{F}"
        for r in range(1, 15):
            g = vp[vp[f"round{F}"] == r]
            a = g[g.pos == "RB"][col].values
            b = g[g.pos == "WR"][col].values
            if len(a) < 5 or len(b) < 5:
                continue
            # season-clustered difference of round means
            ds = []
            for y in YEARS:
                gy = g[g.year == y]
                x, z = gy[gy.pos == "RB"][col], gy[gy.pos == "WR"][col]
                if len(x) and len(z):
                    ds.append(x.mean() - z.mean())
            ds = np.array(ds)
            se = ds.std(ddof=1) / np.sqrt(len(ds)) if len(ds) > 1 else np.nan
            cross.append(dict(frame=F, measure=meas, rnd=r, n_RB=len(a), n_WR=len(b),
                              RB=a.mean(), WR=b.mean(), diff=ds.mean(), se=se,
                              t=ds.mean() / se if se else np.nan,
                              p=2 * (1 - stats.t.cdf(abs(ds.mean() / se), len(ds) - 1)) if se else np.nan,
                              mde=2.802 * se if se else np.nan, n_seasons=len(ds)))
crs = pd.DataFrame(cross)
crs.to_csv(f"{ROOT}/results/sectionM_rbwr_cross.csv", index=False)
print("\n=== §M4(3) RB - WR VORP by draft round ===")
for F in FRAMES:
    print(f"-- frame {F}-team, season totals --")
    print(crs[(crs.frame == F) & (crs.measure == "total")].round(1).to_string(index=False))

# ---------------------------------------------------------------- LOSO ADP->value curve
# Preseason-knowable expected season points given (position, ADP), fit leave-one-season-out
# with a monotone-decreasing isotonic regression. Used by S5 (VORP-greedy) in §M2 and as the
# static value input; never fit on the season it is applied to.
avc = []
for y in YEARS:
    fit = vp[vp.year != y]
    for p in C.POSNS:
        f = fit[fit.pos == p]
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(f.adp, f.total)
        g = vp[(vp.year == y) & (vp.pos == p)]
        avc.append(pd.DataFrame(dict(year=y, pos=p, pid=g.pid, name=g.name, adp=g.adp,
                                     E_total=iso.predict(g.adp))))
avc = pd.concat(avc, ignore_index=True)
avc.to_csv(f"{ROOT}/results/sectionM_adp_value_curve.csv", index=False)
print("\nLOSO ADP->E[season total] curve rows:", len(avc))
print(avc.groupby("pos").E_total.describe()[["mean", "min", "max"]].round(1).to_string())
print("\ndone.")

"""§O1 universe / §O2 inclusion rule / §O3 variance components — TE and QB.

EDA_PLAN6.md §O, pre-registered 2026-08-09.  Mirrors §G (scripts/22_rb_g1_g2.py) exactly,
position swapped.  NOTHING is reused from the WR or RB fits: every variance component,
threshold and gate below is re-estimated on TE data and on QB data separately.

PRE-REGISTERED CHOICES (all fixed by the plan or from AGGREGATE distributions only):

  §O1 universe    top 24 TE and top 24 QB by 2026 ADP (adp_ppr_2026_all_20260809.csv);
                  historical panel = every TE / QB on the FFC board 2015-2024.
                  Board->player join lifted from sectionM_common.py, which is
                  26_sectionL_conversion.py's join carrying its three validated fixes.

  §O2 inclusion   FIXED BY THE PLAN.  TE drops player-games with targets <= 1;
                  QB drops player-games with pass attempts <= 5.  Excluded fraction and
                  its mean PPR reported for each, exactly as §G1 did for RB.

  boom/bust       positional p75 / p25 of PPR over ALL qualified player-games 2014-2025
  thresholds      (the POSITIONAL distribution, not the board), computed and frozen
                  before any rate is fitted.

  relevance gate  the §G rule: match WR's retention rate (82.5% of player-seasons with
                  >= 1 included game) on this position's own usage distribution, rounded
                  to the nearest integer as WR's and RB's were.

  windows         headline crossed decomposition on 2021-2025 (17-game era), with
                  2014-2025+season-FE, log1p and no-exclusion sensitivities — the identical
                  spec ladder scripts 02 and 22 ran, so WR/RB/TE/QB are like-for-like.

Estimators
  §O3a per-player: pooled within-season sigma^2_W (df-weighted), naive between-season v,
       bias-inverted tau^2_B per eq. (3), recency-weighted mu_hat (h=1) with
       n_eff = (sum w)^2 / sum w^2.  script 01's build_table is IMPORTED with the
       positional boom/bust thresholds patched in.
  §O3b crossed decomposition Y_isg = mu + a_i + b_is + c_{team x season} + eps by REML
       (MixedLM, single constant group, vc_formula per factor) and by method-of-moments
       covariance matching.  rho_max per eq. (5).
  §O3c location-scale by experience tier: Gamma GLM (log link, dispersion 2) headline +
       Harvey log e^2 check, SEs clustered by player-season.

  ANOMALY CHASE (pre-declared here, before fitting): QB fantasy points contain rushing, so
  a rushing QB's variance structure differs in kind from a pocket passer's.  §O3c therefore
  additionally decomposes the QB tier structure against rush volume, to establish whether
  any tier effect is about experience or about rushing.

Outputs: results/consistency_table_te.csv, consistency_table_qb.csv,
         variance_components_te.csv, variance_components_qb.csv,
         heteroskedasticity_te.csv, heteroskedasticity_qb.csv,
         sigma2_by_tier_te.csv, sigma2_by_tier_qb.csv,
         sectionO_qb_rush_decomp.csv, sectionO_universe_2026.csv
Rerun: python3 scripts/34_sectionO_o1_o3.py
"""
import importlib.util
import re
import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sectionM_common as MC  # noqa: E402  (alias MC: "C" would shadow patsy C())

ROOT = Path("/Users/thomasmcnamee/NFL")
YEARS = range(2014, 2026)
G_SEASON = 17
WR_RETENTION = 0.825          # the WR gate's retention rate, matched per position

spec = importlib.util.spec_from_file_location(
    "s01", ROOT / "scripts" / "01_section1_consistency.py")
s01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s01)

COLS = ["player_id", "player_display_name", "position", "season", "week",
        "season_type", "team", "targets", "carries", "attempts", "rushing_yards",
        "passing_yards", "fantasy_points_ppr"]
raw = pd.concat([pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{y}.csv",
                             usecols=COLS, low_memory=False) for y in YEARS],
                ignore_index=True)
raw = raw[raw.season_type == "REG"].copy()
raw["attempts"] = raw.attempts.fillna(0)
raw["carries"] = raw.carries.fillna(0)
raw["targets"] = raw.targets.fillna(0)

CFG = {
    "TE": dict(usecol="targets", cut=1, gatecol="targets"),
    "QB": dict(usecol="attempts", cut=5, gatecol="attempts"),
}

# ================================================================= §O1 universe
print("=" * 78)
print("§O1  2026 universe — top 24 TE and top 24 QB by ADP")
print("=" * 78)
adp26 = pd.read_csv(ROOT / "data/adp/adp_ppr_2026_all_20260809.csv")
meta = pd.read_csv(ROOT / "data/meta/players_meta.csv", low_memory=False)
meta = meta.dropna(subset=["gsis_id"]).copy()
meta["nname"] = meta.display_name.map(MC.norm_name).map(MC.collapse_initials)

pdir_src = pd.concat(
    [raw[["player_id", "player_display_name", "position", "season"]]],
    ignore_index=True)
pdir_src["nname"] = pdir_src.player_display_name.map(MC.norm_name).map(MC.collapse_initials)
pdir = (pdir_src.groupby("player_id")
        .agg(nname=("nname", lambda s: s.iat[0]),
             pos=("position", lambda s: s.mode().iat[0]),
             last=("season", "max"), nrows=("season", "size")).reset_index())

uni_rows = []
for P in ("TE", "QB"):
    b = adp26[adp26.position == P].sort_values("adp").head(24).copy()
    b["pos_adp_rank"] = range(1, len(b) + 1)
    b["nname"] = b.name.map(MC.norm_name).replace(MC.ALIASES).map(MC.collapse_initials)
    for _, r in b.iterrows():
        c = pdir[(pdir.nname == r.nname) & (pdir.pos.isin(MC.SKILL))]
        if len(c) > 1:
            c2 = c[c.pos == P]
            c = c2 if len(c2) else c
        if len(c) > 1:
            c = c.sort_values(["last", "nrows"]).tail(1)
        gid = c.player_id.iat[0] if len(c) else None
        src = "weekly"
        if gid is None:
            m = meta[(meta.nname == r.nname) & (meta.position == P)]
            if len(m):
                gid, src = m.gsis_id.iat[0], "meta"
            else:
                src = "NONE(rookie/no NFL rows)"
        uni_rows.append(dict(pos=P, pos_adp_rank=r.pos_adp_rank, name=r["name"],
                             team=r.team, adp=r.adp, stdev=r.stdev, gsis_id=gid,
                             match_source=src))
uni = pd.DataFrame(uni_rows)
uni = uni.merge(meta[["gsis_id", "rookie_season", "birth_date"]], on="gsis_id", how="left")
uni.to_csv(ROOT / "results/sectionO_universe_2026.csv", index=False)
print(uni.groupby(["pos", "match_source"]).size().to_dict())
print("no NFL rows (pure market arm):",
      uni.loc[uni.gsis_id.isna(), ["pos", "name"]].values.tolist())

# ================================================================= §O2 inclusion
print("\n" + "=" * 78)
print("§O2  inclusion rule — excluded fraction and its mean PPR")
print("=" * 78)
POS_DATA, THRESH, GATE, GATE_S = {}, {}, {}, {}
for P in ("TE", "QB"):
    cf = CFG[P]
    d = raw[raw.position == P].copy()
    ex = d[d[cf["usecol"]] <= cf["cut"]]
    q = d[d[cf["usecol"]] > cf["cut"]]
    print(f"\n[{P}] all {P} REG player-games 2014-2025: {len(d)}")
    print(f"  excluded ({cf['usecol']} <= {cf['cut']}): {len(ex)} "
          f"({len(ex)/len(d):.3%}), mean PPR {ex.fantasy_points_ppr.mean():.3f}")
    print(f"  qualified: {len(q)}, mean PPR {q.fantasy_points_ppr.mean():.3f}; "
          f"p25 {q.fantasy_points_ppr.quantile(.25):.2f} "
          f"p50 {q.fantasy_points_ppr.quantile(.50):.2f} "
          f"p75 {q.fantasy_points_ppr.quantile(.75):.2f}")
    THRESH[P] = (round(float(q.fantasy_points_ppr.quantile(.75)), 1),
                 round(float(q.fantasy_points_ppr.quantile(.25)), 1))
    print(f"  FROZEN {P} boom/bust thresholds: > {THRESH[P][0]} / < {THRESH[P][1]}"
          f"   (WR 20/8, RB 13.8/3.2)")
    # board players only
    bg = uni[(uni.pos == P)].gsis_id.dropna()
    bd = d[d.player_id.isin(set(bg))]
    bex = bd[bd[cf["usecol"]] <= cf["cut"]]
    print(f"  board-{P} rows {len(bd)}; excluded {len(bex)} ({len(bex)/max(len(bd),1):.3%}), "
          f"mean PPR {bex.fantasy_points_ppr.mean():.3f}")
    # relevance gate, retention-matched to WR's 82.5%
    ps = q.groupby(["player_id", "season"])[cf["gatecol"]].mean().rename("upg").reset_index()
    gval = float(np.quantile(ps.upg, 1 - WR_RETENTION))
    GATE[P] = round(gval)
    # sensitivity gate = the 50%-retention (median-usage) cut.  Stated as a general rule
    # so it is well-defined at both positions; RB's "2x primary" is not, because QB pass
    # attempts have a hard ceiling near 40/game.
    GATE_S[P] = round(float(np.quantile(ps.upg, 0.50)))
    print(f"  relevance gate: retention-matched value {gval:.2f} {cf['gatecol']}/game "
          f"-> rounded gate {GATE[P]} (retains {(ps.upg >= GATE[P]).mean():.1%}); "
          f"sensitivity gate (median usage) {GATE_S[P]} "
          f"(retains {(ps.upg >= GATE_S[P]).mean():.1%})")
    POS_DATA[P] = dict(all=d, qual=q)

# ================================================================= §O3a consistency
print("\n" + "=" * 78)
print("§O3a  per-player consistency — 2026 board universes")
print("=" * 78)


def build_pos(df, boom, bust):
    src = (ROOT / "scripts" / "01_section1_consistency.py").read_text()
    src = src.replace("(y > 20).sum()", f"(y > {boom}).sum()")
    src = src.replace("(y < 8).sum()", f"(y < {bust}).sum()")
    ns = {"__file__": str(ROOT / "scripts" / "01_section1_consistency.py")}
    exec(compile(src.split("def main()")[0], "s01_pos", "exec"), ns)
    return ns["build_table"](df)


CT = {}
for P in ("TE", "QB"):
    cf = CFG[P]
    bg = set(uni[uni.pos == P].gsis_id.dropna())
    bd = POS_DATA[P]["qual"]
    bd = bd[bd.player_id.isin(bg)].rename(columns={"player_id": "gsis_id"}).copy()
    ct = build_pos(bd, *THRESH[P])
    ct.round(4).to_csv(ROOT / f"results/consistency_table_{P.lower()}.csv", index=False)
    CT[P] = ct
    cols = ["player", "n_seasons", "n_games", "mu_hat", "n_eff", "sigma_W", "naive_v",
            "tau2_B_untrunc", "tau_B", "cv", "q25", "q90", "boom_eb", "bust_eb"]
    print(f"\n[{P}] {len(ct)} players with NFL rows")
    print(ct[cols].round(3).to_string(index=False))
    vets = ct[ct.n_seasons >= 4]
    print(f"  n>=4-season {P}s: {len(vets)}; untruncated tau^2_B < 0 in "
          f"{(vets.tau2_B_untrunc < 0).sum()} ({(vets.tau2_B_untrunc < 0).mean():.0%}); "
          f"median tau^2_B_untrunc {vets.tau2_B_untrunc.median():.2f}")
    print(f"  pooled sigma_W: median {ct.sigma_W.median():.2f}  "
          f"(WR board median 7.86)")

# ================================================================= §O3b crossed VC
print("\n" + "=" * 78)
print("§O3b  crossed variance components")
print("=" * 78)


def pair_sum(y, groups):
    s = pd.Series(y).groupby(np.asarray(groups))
    g_sum, g_sq, g_n = s.sum(), s.apply(lambda v: (v ** 2).sum()), s.size()
    return float(((g_sum ** 2 - g_sq) / 2).sum()), float((g_n * (g_n - 1) / 2).sum())


def mom(d, season_fe=False):
    y = d.yv.values.astype(float)
    y = y - (d.groupby("season").yv.transform("mean").values if season_fe else y.mean())
    var_y = float(np.mean(y ** 2))
    sp_ps, np_ps = pair_sum(y, d.ps)
    sp_p, np_p = pair_sum(y, d.pid)
    sp_ts, np_ts = pair_sum(y, d.ts)
    sp_pts, np_pts = pair_sum(y, d.pts)
    c_ps = sp_ps / np_ps
    s2P = (sp_p - sp_ps) / (np_p - np_ps)
    n_mate = np_ts - np_pts
    s2T = (sp_ts - sp_pts) / n_mate if n_mate > 0 else np.nan
    s2S = c_ps - s2P - (s2T if n_mate > 0 else 0.0)
    return dict(s2P=s2P, s2S=s2S, s2T=s2T, s2G=var_y - c_ps, n_teammate_pairs=n_mate)


def reml(d, season_fe=False):
    d = d.copy(); d["const_grp"] = 1
    vc = {"player": "0 + C(pid)", "player_season": "0 + C(ps)", "team_season": "0 + C(ts)"}
    r = smf.mixedlm("yv ~ C(season)" if season_fe else "yv ~ 1", d,
                    groups="const_grp", vc_formula=vc).fit(
        reml=True, method=["lbfgs", "powell"], maxiter=2000)
    o = dict(zip(sorted(vc), r.vcomp))
    return dict(s2P=o["player"], s2S=o["player_season"], s2T=o["team_season"],
                s2G=r.scale, converged=bool(r.converged),
                resid_skew=float(stats.skew(np.asarray(r.resid))))


def mom3(d, season_fe=False):
    """3-way player / player-season / game.  ALWAYS identified.  Needed because at QB the
    4-way is not: there is one starting QB per team-season, so the player-season and
    team-season factors are nearly the same partition and sigma^2_S / sigma^2_T are not
    separately identified (this shows up as a large negative s2S and a large s2T)."""
    y = d.yv.values.astype(float)
    y = y - (d.groupby("season").yv.transform("mean").values if season_fe else y.mean())
    var_y = float(np.mean(y ** 2))
    sp_ps, np_ps = pair_sum(y, d.ps)
    sp_p, np_p = pair_sum(y, d.pid)
    c_ps = sp_ps / np_ps
    s2P = (sp_p - sp_ps) / (np_p - np_ps)
    return dict(s2P=s2P, s2S=c_ps - s2P, s2T=0.0, s2G=var_y - c_ps, n_teammate_pairs=np.nan)


def reml3(d, season_fe=False):
    d = d.copy(); d["const_grp"] = 1
    vc = {"player": "0 + C(pid)", "player_season": "0 + C(ps)"}
    r = smf.mixedlm("yv ~ C(season)" if season_fe else "yv ~ 1", d,
                    groups="const_grp", vc_formula=vc).fit(
        reml=True, method=["lbfgs", "powell"], maxiter=2000)
    o = dict(zip(sorted(vc), r.vcomp))
    return dict(s2P=o["player"], s2S=o["player_season"], s2T=0.0, s2G=r.scale,
                converged=bool(r.converged),
                resid_skew=float(stats.skew(np.asarray(r.resid))))


def summ(c):
    tot = c["s2P"] + c["s2S"] + c["s2T"] + c["s2G"]
    o = dict(c)
    for k in ["s2P", "s2S", "s2T", "s2G"]:
        o[f"icc_{k[2:]}"] = c[k] / tot
    o["rho_max"] = c["s2P"] / (c["s2P"] + c["s2S"] + c["s2T"] + c["s2G"] / G_SEASON)
    return o


SPECS = [("headline_2021_2025_excl", (2021, 2025), True, False, False),
         ("sens_a_2014_2025_seasonFE", (2014, 2025), True, False, True),
         ("sens_b_log1p_2021_2025", (2021, 2025), True, True, False),
         ("sens_c_no_exclusions", (2021, 2025), False, False, False)]

VC = {}
for P in ("TE", "QB"):
    cf = CFG[P]
    bg = set(uni[uni.pos == P].gsis_id.dropna())
    brd = POS_DATA[P]["all"]
    brd = brd[brd.player_id.isin(bg)].copy()

    def prep(years, exclusions=True, log=False, _brd=brd, _cf=cf):
        d = _brd[_brd.season.between(*years)].copy()
        if exclusions:
            d = d[d[_cf["usecol"]] > _cf["cut"]]
        d = d.dropna(subset=["team"])
        d["pid"] = d.player_id
        d["ps"] = d.player_id + "_" + d.season.astype(str)
        d["ts"] = d.team + "_" + d.season.astype(str)
        d["pts"] = d.pid + "_" + d.ts
        d["yv"] = np.log1p(d.fantasy_points_ppr.clip(lower=-0.99)) if log \
            else d.fantasy_points_ppr
        return d.reset_index(drop=True)

    rows = []
    print(f"\n---- {P} ----")
    for name, yrs, excl, log, sfe in SPECS:
        d = prep(yrs, excl, log)
        m0 = dict(spec=name, n_obs=len(d), n_players=d.pid.nunique(),
                  n_player_seasons=d.ps.nunique(), n_team_seasons=d.ts.nunique())
        m = summ(mom(d, sfe)); rows.append({**m0, "estimator": "MoM_covmatch", **m})
        print(f"[{name}] MoM : " + ", ".join(f"{k}={v:.4g}" for k, v in m.items()))
        try:
            r = summ(reml(d, sfe)); rows.append({**m0, "estimator": "REML_MixedLM", **r})
            print(f"[{name}] REML: " + ", ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in r.items()))
        except Exception as e:
            print(f"[{name}] REML FAILED: {e}")
            rows.append({**m0, "estimator": "REML_MixedLM", "converged": False})
        m3 = summ(mom3(d, sfe)); rows.append({**m0, "estimator": "MoM_3way_noTeam", **m3})
        print(f"[{name}] MoM3: " + ", ".join(f"{k}={v:.4g}" for k, v in m3.items()))
        try:
            r3 = summ(reml3(d, sfe))
            rows.append({**m0, "estimator": "REML_3way_noTeam", **r3})
            print(f"[{name}] RML3: " + ", ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in r3.items()))
        except Exception as e:
            print(f"[{name}] REML3 FAILED: {e}")
    vcp = pd.DataFrame(rows)
    vcp.round(5).to_csv(ROOT / f"results/variance_components_{P.lower()}.csv", index=False)
    VC[P] = vcp

    d = prep((2021, 2025), True, False)
    y = d.fantasy_points_ppr.values - d.fantasy_points_ppr.mean()
    sm_ = pd.DataFrame({"pid": d.pid, "season": d.season, "y": y})
    seas = sm_.groupby(["pid", "season"]).y.mean().reset_index()
    mrg = seas.merge(seas.assign(season=seas.season - 1), on=["pid", "season"],
                     suffixes=("", "_next"))
    r_adj = np.corrcoef(mrg.y, mrg.y_next)[0, 1] if len(mrg) > 2 else np.nan
    print(f"lag-1 same-player season-mean cov {np.cov(mrg.y, mrg.y_next)[0,1]:.3f}, "
          f"corr {r_adj:.3f} (n={len(mrg)} pairs); Var(season mean) {seas.y.var(ddof=1):.3f}")

# ------------- §O3 PRE-REGISTERED PREDICTION TEST -------------
# EDA_PLAN6 §O3 recorded, before fitting: "QB PPG should be far less noisy per game than
# WR/RB (no touch-share volatility, ~35 attempts every week), which if true means QB mu_hat
# is *more* reliable and B should shrink less toward market."
# Two readings are possible and BOTH are reported, because they can disagree:
#   (a) ABSOLUTE per-game variance sigma^2_G  — what "less noisy per game" says literally;
#   (b) SCALE-FREE noise CV = sigma_W / mu     — noise relative to the level being measured,
#       which is what actually drives reliability, since V = sigma^2/n_eff is compared to
#       tau^2 measured on the same PPG scale.
# The operational half of the prediction (does B shrink less?) is settled in §O5's LOSO.
print("\n" + "-" * 78)
print("§O3 pre-registered prediction — is QB PPG less noisy per game than WR/RB?")
print("-" * 78)
pred_rows = []
_ct = {"WR": pd.read_csv(ROOT / "results/consistency_table.csv"),
       "RB": pd.read_csv(ROOT / "results/consistency_table_rb.csv"),
       "TE": CT["TE"], "QB": CT["QB"]}
_vc = {"WR": pd.read_csv(ROOT / "results/variance_components.csv"),
       "RB": pd.read_csv(ROOT / "results/variance_components_rb.csv"),
       "TE": VC["TE"], "QB": VC["QB"]}
for p in ("WR", "RB", "TE", "QB"):
    t = _ct[p]
    v = _vc[p]
    h = v[(v.spec == "headline_2021_2025_excl") & (v.estimator == "MoM_covmatch")].iloc[0]
    h3 = v[(v.spec == "headline_2021_2025_excl") & (v.estimator == "MoM_3way_noTeam")]
    pred_rows.append(dict(pos=p, board_mean_mu=t.mu_hat.mean(),
                          median_sigma_W=t.sigma_W.median(),
                          median_cv=t.cv.median(),
                          s2G_4way=h.s2G, s2P_4way=h.s2P, s2S_4way=h.s2S, s2T_4way=h.s2T,
                          rho_max_4way=h.rho_max,
                          s2G_3way=h3.s2G.iat[0] if len(h3) else np.nan,
                          rho_max_3way=h3.rho_max.iat[0] if len(h3) else np.nan,
                          cv_g=np.sqrt(h.s2G) / t.mu_hat.mean()))
PR = pd.DataFrame(pred_rows)
PR.to_csv(ROOT / "results/sectionO_o3_prediction.csv", index=False)
print(PR.round(3).to_string(index=False))

# ================================================================= §O3c heteroskedasticity
print("\n" + "=" * 78)
print("§O3c  location-scale by experience tier")
print("=" * 78)
mmeta = pd.read_csv(ROOT / "data/meta/players_meta.csv", low_memory=False,
                    usecols=["gsis_id", "rookie_season"])

for P in ("TE", "QB"):
    cf = CFG[P]
    h = POS_DATA[P]["qual"].copy()
    ps = (h.groupby(["player_id", "season"])
          .agg(G=("fantasy_points_ppr", "size"), use=(cf["gatecol"], "mean"),
               mu_ps=("fantasy_points_ppr", "mean"),
               rush=("carries", "mean")).reset_index())

    def het(gate, label, _h=h, _ps=ps):
        keep = _ps[_ps.use >= gate]
        d = _h.merge(keep[["player_id", "season", "mu_ps", "rush"]],
                     on=["player_id", "season"])
        d = d.merge(mmeta.rename(columns={"gsis_id": "player_id"}), on="player_id",
                    how="left").dropna(subset=["rookie_season"])
        d["exp"] = (d.season - d.rookie_season).astype(int)
        d = d[d.exp >= 0].copy()
        d["rookie"] = (d.exp == 0).astype(int); d["soph"] = (d.exp == 1).astype(int)
        d["ps_id"] = d.player_id + "_" + d.season.astype(str)
        d["e2"] = (d.fantasy_points_ppr - d.mu_ps) ** 2
        out = []
        dA = d[d.e2 >= 1e-6]
        mA = smf.ols("np.log(e2) ~ rookie + soph", data=dA).fit(
            cov_type="cluster", cov_kwds={"groups": dA.ps_id})
        X = sm.add_constant(d[["rookie", "soph"]])
        mB = sm.GLM(d.e2, X, family=sm.families.Gamma(link=sm.families.links.Log())).fit(
            scale=2.0, cov_type="cluster", cov_kwds={"groups": d.ps_id})
        for route, m, n in [("A_harvey", mA, len(dA)), ("B_gammaGLM", mB, len(d))]:
            ci = m.conf_int()
            for t in ["rookie", "soph"]:
                out.append(dict(pos=P, sample=label, route=route, term=t,
                                est=m.params[t], se=m.bse[t], ci_lo=ci.loc[t, 0],
                                ci_hi=ci.loc[t, 1], mult=np.exp(m.params[t]),
                                mult_lo=np.exp(ci.loc[t, 0]), mult_hi=np.exp(ci.loc[t, 1]),
                                n=n, n_player_seasons=d.ps_id.nunique()))
        print(f"\n[{P} {label}] games {len(d)}, player-seasons {d.ps_id.nunique()} "
              f"(rookie {d[d.rookie==1].ps_id.nunique()}, soph {d[d.soph==1].ps_id.nunique()}, "
              f"vet {d[(d.rookie==0)&(d.soph==0)].ps_id.nunique()})")
        print(f"  gamma-GLM mult: rookie {np.exp(mB.params['rookie']):.3f} "
              f"[{np.exp(mB.conf_int().loc['rookie',0]):.3f},"
              f"{np.exp(mB.conf_int().loc['rookie',1]):.3f}], "
              f"soph {np.exp(mB.params['soph']):.3f} "
              f"[{np.exp(mB.conf_int().loc['soph',0]):.3f},"
              f"{np.exp(mB.conf_int().loc['soph',1]):.3f}]")
        print("  Wald g1=g2=0:", mB.wald_test("rookie = 0, soph = 0", scalar=True))
        print("  raw mean e2 by tier:",
              d.groupby(d.exp.clip(upper=2)).e2.mean().round(2).to_dict())
        dl = d[(d.mu_ps > 0.5) & (d.e2 >= 1e-6)]
        ml = smf.ols("np.log(e2) ~ np.log(mu_ps) + rookie + soph", data=dl).fit(
            cov_type="cluster", cov_kwds={"groups": dl.ps_id})
        print(f"  log e2 ~ log mu: slope {ml.params['np.log(mu_ps)']:.3f} "
              f"(se {ml.bse['np.log(mu_ps)']:.3f}); tier mults controlling level: "
              f"rookie {np.exp(ml.params['rookie']):.3f} (p {ml.pvalues['rookie']:.3f}), "
              f"soph {np.exp(ml.params['soph']):.3f} (p {ml.pvalues['soph']:.3f})")
        return out, mB, d

    rows, mB_main, dmain = het(GATE[P], f"primary_gate_{GATE[P]:g}")
    rows2, _, _ = het(GATE_S[P], f"sens_gate_median_{GATE_S[P]:g}")
    pd.DataFrame(rows + rows2).to_csv(
        ROOT / f"results/heteroskedasticity_{P.lower()}.csv", index=False)
    g0 = mB_main.params["const"]
    sig2 = pd.DataFrame(dict(tier=["rookie", "soph", "vet"],
                             sigma2=[np.exp(g0 + mB_main.params["rookie"]),
                                     np.exp(g0 + mB_main.params["soph"]), np.exp(g0)]))
    sig2.to_csv(ROOT / f"results/sigma2_by_tier_{P.lower()}.csv", index=False)
    print(f"\nsigma^2(tier) {P}:", sig2.set_index("tier").sigma2.round(3).to_dict(),
          " (WR 36.4/39.7/43.1; RB from sigma2_by_tier_rb.csv)")

    # ---------- pre-declared anomaly chase: is the QB tier structure about rushing? ----------
    if P == "QB":
        print("\n" + "-" * 78)
        print("ANOMALY CHASE (pre-declared): QB tier structure — experience or rushing?")
        print("-" * 78)
        d = dmain.copy()
        d["rush_pg"] = d.rush
        # rush volume tercile of the player-season
        pss = d.groupby("ps_id").agg(rush=("rush_pg", "first"),
                                     e2=("e2", "mean"), mu=("mu_ps", "first"),
                                     exp=("exp", "first")).reset_index()
        pss["rush_tercile"] = pd.qcut(pss.rush, 3, labels=["low", "mid", "high"])
        print("player-season mean e2 and mean PPG by rush-carry tercile:")
        print(pss.groupby("rush_tercile", observed=True)
              .agg(n=("e2", "size"), mean_rush_pg=("rush", "mean"),
                   mean_e2=("e2", "mean"), mean_ppg=("mu", "mean")).round(2).to_string())
        d = d.merge(pss[["ps_id", "rush_tercile"]], on="ps_id")
        d["rhigh"] = (d.rush_tercile == "high").astype(int)
        d["rmid"] = (d.rush_tercile == "mid").astype(int)
        X = sm.add_constant(d[["rookie", "soph", "rmid", "rhigh"]])
        m2 = sm.GLM(d.e2, X, family=sm.families.Gamma(link=sm.families.links.Log())).fit(
            scale=2.0, cov_type="cluster", cov_kwds={"groups": d.ps_id})
        print("\nGamma GLM e2 ~ tier + rush tercile (mults):")
        ci2 = m2.conf_int()
        qrows = []
        for t in ["rookie", "soph", "rmid", "rhigh"]:
            print(f"  {t:7s} {np.exp(m2.params[t]):.3f} "
                  f"[{np.exp(ci2.loc[t,0]):.3f},{np.exp(ci2.loc[t,1]):.3f}]  "
                  f"p {m2.pvalues[t]:.4f}")
            qrows.append(dict(model="e2 ~ tier + rush_tercile", term=t,
                              mult=np.exp(m2.params[t]), lo=np.exp(ci2.loc[t, 0]),
                              hi=np.exp(ci2.loc[t, 1]), p=m2.pvalues[t]))
        # continuous version + does rush explain the LEVEL, not just the scale?
        dl = d[(d.mu_ps > 0.5) & (d.e2 >= 1e-6)]
        m3 = smf.ols("np.log(e2) ~ rush_pg + rookie + soph", data=dl).fit(
            cov_type="cluster", cov_kwds={"groups": dl.ps_id})
        print(f"\n  log e2 ~ rush/game: {m3.params['rush_pg']:+.4f} per carry/game "
              f"(se {m3.bse['rush_pg']:.4f}, p {m3.pvalues['rush_pg']:.4f}); "
              f"tier mults now rookie {np.exp(m3.params['rookie']):.3f}, "
              f"soph {np.exp(m3.params['soph']):.3f}")
        qrows.append(dict(model="log e2 ~ rush_pg + tier", term="rush_pg",
                          mult=np.exp(m3.params["rush_pg"]), lo=np.nan, hi=np.nan,
                          p=m3.pvalues["rush_pg"]))
        # and the between-season component: do rushers move more year to year?
        ct = CT["QB"].copy()
        rmap = (POS_DATA["QB"]["qual"].groupby("player_id").carries.mean())
        ct["rush_pg"] = ct.gsis_id.map(rmap)
        cc = ct[ct.n_seasons >= 4].dropna(subset=["tau2_B_untrunc", "rush_pg"])
        if len(cc) >= 5:
            rr = stats.spearmanr(cc.rush_pg, cc.tau2_B_untrunc)
            rw = stats.spearmanr(cc.rush_pg, cc.sigma_W)
            print(f"  board QBs n>=4 seasons (n={len(cc)}): Spearman(rush/g, tau^2_B) "
                  f"= {rr.statistic:+.3f} (p {rr.pvalue:.3f}); "
                  f"Spearman(rush/g, sigma_W) = {rw.statistic:+.3f} (p {rw.pvalue:.3f})")
            qrows.append(dict(model="board tau2_B vs rush", term="spearman",
                              mult=rr.statistic, lo=np.nan, hi=np.nan, p=rr.pvalue))
            qrows.append(dict(model="board sigma_W vs rush", term="spearman",
                              mult=rw.statistic, lo=np.nan, hi=np.nan, p=rw.pvalue))
        pd.DataFrame(qrows).to_csv(ROOT / "results/sectionO_qb_rush_decomp.csv", index=False)

print("\ndone — wrote consistency_table_{te,qb}.csv, variance_components_{te,qb}.csv, "
      "heteroskedasticity_{te,qb}.csv, sigma2_by_tier_{te,qb}.csv, "
      "sectionO_qb_rush_decomp.csv, sectionO_universe_2026.csv")

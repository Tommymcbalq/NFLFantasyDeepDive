"""§O4 market prior — isotonic ADP -> PPG on the 2015-2024 TE and QB panels.

The §6.1 / §21 construction, re-estimated per position.  No WR or RB quantity is reused.

PRE-REGISTERED (mirrors scripts 07 and 23 exactly, position swapped):
  - panel: every TE / QB on the FFC PPR 12-team board each year 2015-2024 (the boards
    carry ~15-22 TE and ~20-27 QB; §O1's "top 24" is therefore all of them), joined to
    realized SAME-season PPG under the §O2 inclusion rule
    (REG; TE targets >= 2, QB attempts >= 6).
  - board -> player join: sectionM_common.build_panel, i.e. 26_sectionL_conversion.py's
    validated join.  §M verified it for QB/TE at 0 unmatched; re-verified here.
  - rows with < 4 included games stay in the panel (flagged) but are excluded from the
    m(.) fit and from the tau^2 residual variances.  Same floor as WR and RB.
  - m(ADP): isotonic, monotone DECREASING in log ADP (headline); OLS on log ADP as the
    reference fit.
  - tau^2(tier) = Var(realized PPG - m_iso(ADP)) by experience tier at the ADP year
    (rookie / soph / vet), reported AS ESTIMATED with NO ordering imposed.

ANOMALY CHASE, pre-declared before fitting (the §O brief's second trap): TE is the
position where one player can sit many SDs above the curve, which would distort an
isotonic fit at the top.  Diagnostics run and reported regardless of what they show:
  (a) the raw decile means, so any reversal isotonic has to flatten is visible;
  (b) the standardised top-of-board residuals, to locate any such player;
  (c) a leave-the-largest-residual-out refit of m(.), reported as a DIAGNOSTIC ONLY —
      the headline m(.) is the full-sample pre-registered fit and is not replaced.

Outputs: results/market_prior_te.csv, market_prior_qb.csv,
         tier_variances_te.csv, tier_variances_qb.csv,
         market_prior_iso_knots_te.csv, market_prior_iso_knots_qb.csv,
         sectionO_iso_diagnostics.csv
Rerun: python3 scripts/35_sectionO_o4_market_prior.py
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.isotonic import IsotonicRegression

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sectionM_common as MC  # noqa: E402

ROOT = Path("/Users/thomasmcnamee/NFL")
YEARS = list(range(2015, 2025))
CUT = {"TE": ("targets", 1), "QB": ("attempts", 5)}

# ---------------------------------------------------------------- realized PPG
COLS = ["player_id", "season", "season_type", "position", "targets", "attempts",
        "fantasy_points_ppr"]
frames = []
for y in YEARS:
    d = pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{y}.csv",
                    usecols=COLS, low_memory=False)
    frames.append(d[d.season_type == "REG"])
wk_all = pd.concat(frames, ignore_index=True)
wk_all["targets"] = wk_all.targets.fillna(0)
wk_all["attempts"] = wk_all.attempts.fillna(0)

meta = pd.read_csv(ROOT / "data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "rookie_season"]).dropna(subset=["gsis_id"])

# ---------------------------------------------------------------- board panel
print("building board panel (sectionM_common join) ...", flush=True)
wkj = MC.load_weekly()
panel_all, unmatched, ambig = MC.build_panel(wkj)
print(f"board rows {len(panel_all)}; unmatched {len(unmatched)}; ambiguous {len(ambig)}")
print("by position:", panel_all.pos.value_counts().to_dict())
for P in ("TE", "QB"):
    sub = panel_all[panel_all.pos == P]
    print(f"  {P}: {len(sub)} rows, {sub.pid.isna().sum()} null ids, "
          f"per-year counts {sub.groupby('year').size().to_dict()}")
assert panel_all[panel_all.pos.isin(["TE", "QB"])].pid.isna().sum() == 0, \
    "§O0 join verification failed for TE/QB"
print("JOIN VERIFIED for TE and QB: 0 null player ids.")

diag_rows = []
for P in ("TE", "QB"):
    col, cut = CUT[P]
    print("\n" + "=" * 78)
    print(f"§O4  market prior — {P}")
    print("=" * 78)
    inc = wk_all[(wk_all[col] > cut)]
    real = (inc.groupby(["player_id", "season"])
            .agg(games=("fantasy_points_ppr", "size"),
                 ppg=("fantasy_points_ppr", "mean")).reset_index()
            .rename(columns={"player_id": "gsis_id"}))

    p = panel_all[panel_all.pos == P].rename(columns={"pid": "gsis_id"}).copy()
    p["adp_rank"] = p.groupby("year").adp.rank(method="first").astype(int)
    p = p.merge(real, left_on=["gsis_id", "year"], right_on=["gsis_id", "season"],
                how="left").drop(columns=["season"])
    p = p.merge(meta, on="gsis_id", how="left")
    p["games"] = p.games.fillna(0).astype(int)
    p["exp"] = p.year - p.rookie_season
    p["tier"] = np.select([p.exp == 0, p.exp == 1], ["rookie", "soph"], "vet")
    p.loc[p.rookie_season.isna(), "tier"] = "vet"
    p["in_fit"] = p.games >= 4

    print(f"panel rows: {len(p)}; 0 included games {(p.games == 0).sum()}; "
          f"1-3 games {((p.games > 0) & (p.games < 4)).sum()}; "
          f"dropped from fit {(~p.in_fit).sum()}")
    print("tier counts (all rows):", p.tier.value_counts().to_dict())
    print("tier counts (in_fit):", p[p.in_fit].tier.value_counts().to_dict())

    fit = p[p.in_fit].copy()
    ols = smf.ols("ppg ~ np.log(adp)", data=fit).fit(cov_type="HC3")
    print(f"\nOLS ppg ~ log(adp): intercept {ols.params['Intercept']:.3f}, slope "
          f"{ols.params['np.log(adp)']:.3f} (se {ols.bse['np.log(adp)']:.3f}), "
          f"R2 {ols.rsquared:.3f}, n {int(ols.nobs)}")
    ols_fe = smf.ols("ppg ~ np.log(adp) + C(year)", data=fit).fit(cov_type="HC3")
    print(f"year-FE slope: {ols_fe.params['np.log(adp)']:.3f}")

    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(np.log(fit.adp.values), fit.ppg.values)
    p["m_ols"] = ols.params["Intercept"] + ols.params["np.log(adp)"] * np.log(p.adp)
    p["m_iso"] = iso.predict(np.log(p.adp.values))
    p["resid_iso"] = p.ppg - p.m_iso
    p["resid_ols"] = p.ppg - p.m_ols
    r_iso = np.sqrt((p.loc[p.in_fit, "resid_iso"] ** 2).mean())
    r_ols = np.sqrt((p.loc[p.in_fit, "resid_ols"] ** 2).mean())
    print(f"in-sample RMSE: OLS {r_ols:.3f}, isotonic {r_iso:.3f}")

    tv = (p[p.in_fit].groupby("tier")
          .agg(n=("resid_iso", "size"),
               tau2_iso=("resid_iso", lambda x: x.var(ddof=1)),
               tau2_ols=("resid_ols", lambda x: x.var(ddof=1)),
               mean_resid_iso=("resid_iso", "mean"))
          .reindex(["rookie", "soph", "vet"]).reset_index())
    rng = np.random.default_rng(40823)
    for t in ["rookie", "soph", "vet"]:
        r = p.loc[p.in_fit & (p.tier == t), "resid_iso"].values
        if len(r) >= 2:
            bs = [np.var(rng.choice(r, len(r), replace=True), ddof=1) for _ in range(4000)]
            tv.loc[tv.tier == t, ["tau2_lo", "tau2_hi"]] = np.percentile(bs, [2.5, 97.5])
    tv.to_csv(ROOT / f"results/tier_variances_{P.lower()}.csv", index=False)
    print(f"\ntau^2 by tier ({P}), as estimated, no ordering imposed:")
    print(tv.round(3).to_string(index=False))

    p.to_csv(ROOT / f"results/market_prior_{P.lower()}.csv", index=False)
    knots = pd.DataFrame({"log_adp": iso.X_thresholds_, "m": iso.y_thresholds_})
    knots.to_csv(ROOT / f"results/market_prior_iso_knots_{P.lower()}.csv", index=False)
    lv = np.unique(np.round(iso.y_thresholds_, 9))
    print(f"\nisotonic: {len(lv)} unique levels, m from {iso.y_thresholds_.max():.2f} to "
          f"{iso.y_thresholds_.min():.2f}; ADP fit range "
          f"{fit.adp.min():.1f}..{fit.adp.max():.1f}")
    kk = knots.assign(adp=np.exp(knots.log_adp))
    print("step boundaries (ADP -> level):")
    print(kk[kk.m.diff().fillna(0) != 0][["adp", "m"]].round(3).to_string(index=False))

    # ------------------------------------------------ (a) raw decile monotonicity
    fit["bin"] = pd.qcut(fit.adp, 10, duplicates="drop")
    bb = fit.groupby("bin", observed=True).agg(n=("ppg", "size"), adp=("adp", "mean"),
                                               ppg=("ppg", "mean"))
    print("\nraw decile means (isotonic must flatten any reversal here):")
    print(bb.round(3).to_string())
    nrev = int((bb.ppg.diff().dropna() > 0).sum())
    print(f"reversals among 9 adjacent decile steps: {nrev}")

    # --------------------------- (b)/(c) elite-outlier diagnostic, DIAGNOSTIC ONLY
    print("\n--- pre-declared anomaly check: does one player distort the top of m(.)? ---")
    f2 = p[p.in_fit].copy()
    sd = f2.resid_iso.std(ddof=1)
    f2["z"] = f2.resid_iso / sd
    top = f2.nlargest(6, "z")[["year", "name", "adp", "adp_rank", "ppg", "m_iso",
                               "resid_iso", "z"]]
    print(f"residual SD {sd:.3f}; six largest standardised residuals:")
    print(top.round(2).to_string(index=False))
    # how many observations sit in the top isotonic level, and what is it made of?
    topm = f2.m_iso.max()
    tl = f2[np.isclose(f2.m_iso, topm)]
    print(f"top isotonic level m = {topm:.2f}, supported by n = {len(tl)} rows "
          f"(ADP {tl.adp.min():.1f}..{tl.adp.max():.1f}); their ppg "
          f"{np.sort(tl.ppg.values).round(1).tolist()}")
    # leave-out refit
    drop_idx = f2.z.idxmax()
    f3 = f2.drop(index=drop_idx)
    iso2 = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso2.fit(np.log(f3.adp.values), f3.ppg.values)
    m2 = iso2.predict(np.log(f2.adp.values))
    dlt = m2 - f2.m_iso.values
    hi = f2.adp <= np.percentile(f2.adp, 20)
    print(f"leave-one-out (largest residual: {f2.loc[drop_idx,'year']} "
          f"{f2.loc[drop_idx,'name']}, z = {f2.loc[drop_idx,'z']:.2f}) refit of m(.): "
          f"max |delta m| {np.abs(dlt).max():.3f} PPG, mean |delta| over the top ADP "
          f"quintile {np.abs(dlt[hi.values]).mean():.3f}, top level "
          f"{iso2.y_thresholds_.max():.2f} vs {topm:.2f}")
    diag_rows.append(dict(pos=P, resid_sd=sd, top_level=topm, n_top_level=len(tl),
                          top_z=float(f2.z.max()),
                          top_player_year=int(f2.loc[drop_idx, "year"]),
                          decile_reversals=nrev,
                          loo_max_abs_delta_m=float(np.abs(dlt).max()),
                          loo_mean_abs_delta_top_quintile=float(np.abs(dlt[hi.values]).mean()),
                          loo_top_level=float(iso2.y_thresholds_.max()),
                          rmse_iso=float(r_iso), rmse_ols=float(r_ols)))

# ---------------------------------------------------------------------------------
# Follow-up chase, declared after §O3c found the QB variance axis is RUSHING not
# experience: is the right tail of the QB market-prior residual also rushing?
# (Descriptive decomposition of an already-computed residual; no model is refit.)
# ---------------------------------------------------------------------------------
print("\n" + "=" * 78)
print("chase: is the QB market-prior residual a rushing residual?")
print("=" * 78)
qb = pd.read_csv(ROOT / "results/market_prior_qb.csv")
rc = pd.concat([pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{y}.csv",
                            usecols=["player_id", "season", "season_type", "carries",
                                     "rushing_yards", "rushing_tds", "attempts"],
                            low_memory=False) for y in YEARS], ignore_index=True)
rc = rc[rc.season_type == "REG"]
rc = rc[rc.attempts.fillna(0) > 5]
rs = (rc.groupby(["player_id", "season"])
      .agg(rush_pg=("carries", "mean"), ryd_pg=("rushing_yards", "mean"),
           rtd=("rushing_tds", "sum")).reset_index()
      .rename(columns={"player_id": "gsis_id", "season": "year"}))
qb = qb.merge(rs, on=["gsis_id", "year"], how="left")
q = qb[qb.in_fit].dropna(subset=["rush_pg"]).copy()
q["rush_q"] = pd.qcut(q.rush_pg, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
print("mean isotonic residual by same-season rush-carry quartile:")
print(q.groupby("rush_q", observed=True)
      .agg(n=("resid_iso", "size"), rush_pg=("rush_pg", "mean"),
           mean_resid=("resid_iso", "mean"), sd_resid=("resid_iso", "std"),
           mean_ppg=("ppg", "mean"), mean_adp=("adp", "mean")).round(2).to_string())
from scipy import stats as _st  # noqa: E402
sp = _st.spearmanr(q.rush_pg, q.resid_iso)
print(f"Spearman(rush carries/g, residual) = {sp.statistic:+.3f} (p {sp.pvalue:.2e}), "
      f"n = {len(q)}")
big = q.nlargest(15, "resid_iso")
print(f"of the 15 largest positive residuals, {int((big.rush_pg >= q.rush_pg.median()).sum())}"
      f"/15 are above-median rushers (rush/g median {q.rush_pg.median():.2f})")
ols_r = smf.ols("resid_iso ~ rush_pg", data=q).fit(
    cov_type="cluster", cov_kwds={"groups": q.year})
print(f"resid ~ rush/g: {ols_r.params['rush_pg']:+.3f} PPG per carry/game "
      f"(year-clustered se {ols_r.bse['rush_pg']:.3f}, p {ols_r.pvalues['rush_pg']:.4f}), "
      f"R2 {ols_r.rsquared:.3f}")
q[["year", "name", "adp", "ppg", "m_iso", "resid_iso", "rush_pg", "ryd_pg", "rtd"]] \
    .to_csv(ROOT / "results/sectionO_qb_resid_rush.csv", index=False)

# ---------------------------------------------------------------------------------
# Cross-era support check: does the 2026 board sit inside the ADP range m(.) was fit on?
# ---------------------------------------------------------------------------------
print("\n" + "=" * 78)
print("support check: 2026 board ADP vs the historical ADP range m(.) is fit on")
print("=" * 78)
uni = pd.read_csv(ROOT / "results/sectionO_universe_2026.csv")
for P in ("TE", "QB"):
    pp = pd.read_csv(ROOT / f"results/market_prior_{P.lower()}.csv")
    f = pp[pp.in_fit]
    u = uni[uni.pos == P]
    print(f"[{P}] historical in-fit ADP {f.adp.min():.1f}..{f.adp.max():.1f}; "
          f"2026 board ADP {u.adp.min():.1f}..{u.adp.max():.1f}; "
          f"2026 rows below the historical min: {int((u.adp < f.adp.min()).sum())}")
    print(f"      historical rows at ADP <= {u.adp.min():.0f} (the 2026 top): "
          f"{int((f.adp <= u.adp.min()).sum())} of {len(f)}; "
          f"their mean ppg {f[f.adp <= u.adp.min()].ppg.mean():.2f}")
    yb = f.groupby("year").adp.min()
    print("      best-priced {} by year (ADP of the position's #1): {}".format(
        P, yb.round(1).to_dict()))

pd.DataFrame(diag_rows).to_csv(ROOT / "results/sectionO_iso_diagnostics.csv", index=False)
print("\nwrote market_prior_{te,qb}.csv, tier_variances_{te,qb}.csv, "
      "market_prior_iso_knots_{te,qb}.csv, sectionO_iso_diagnostics.csv")

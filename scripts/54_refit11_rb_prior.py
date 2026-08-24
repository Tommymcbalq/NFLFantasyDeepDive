"""§G3 RB market prior — isotonic ADP -> PPG on the 2015-2024 RB panel, plus the
outcome spread around it.  This is the §6.1 analogue, re-estimated on RB data;
no WR quantity is reused.  It also exports the (pi, Sigma) inputs the §J overlay
will need.

PRE-REGISTERED (mirrors script 07 exactly, position swapped):
  - panel: top-30 RBs by FFC PPR 12-team ADP each year 2015-2024 (2025 ADP does
    not exist at the source), joined to realized SAME-season PPG under the §G1
    inclusion rule (REG, touches = carries+targets >= 2).
  - name -> gsis_id by the script-07 normalization (lowercase, punctuation and
    Jr/Sr/II/III/IV/V suffixes stripped), restricted to RBs active that season,
    with the same documented fallback ladder; every unmatched name reported.
  - rows with < 4 included games stay in the panel (flagged) but are excluded
    from the m(.) fit and from the tau^2 residual variances.  Same floor as WR.
  - m(ADP): isotonic, monotone DECREASING in log ADP (headline); OLS on log ADP
    as the reference fit.
  - tau^2(tier) = Var(realized PPG - m_iso(ADP)) by experience tier at the ADP
    year (rookie / soph / vet), reported as estimated with NO ordering imposed.

Outputs: results/market_prior_rb_11yr.csv, results/tier_variances_rb_11yr.csv,
         results/market_prior_iso_knots_rb_11yr.csv, results/sectionJ_pi_sigma_rb_11yr.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/Users/thomasmcnamee/NFL")
YEARS = range(2015, 2026)
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_name(s):
    s = re.sub(r"[^a-z ]", "", str(s).lower().replace(".", " ")
               .replace("-", " ").replace("'", ""))
    return " ".join(t for t in s.split() if t not in SUFFIXES)


meta = pd.read_csv(ROOT / "data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "display_name", "position", "rookie_season",
                            "last_season"]).dropna(subset=["gsis_id"])
meta["nname"] = meta.display_name.map(norm_name)
meta_rb = meta[meta.position == "RB"]

# realized PPG under the §G1 rule
frames = []
for y in YEARS:
    df = pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{y}.csv",
                     usecols=["player_id", "season", "season_type", "targets",
                              "carries", "fantasy_points_ppr"], low_memory=False)
    df = df[df.season_type == "REG"].copy()
    df["touches"] = df.carries.fillna(0) + df.targets.fillna(0)
    frames.append(df[df.touches >= 2])
wk = pd.concat(frames, ignore_index=True)
real = (wk.groupby(["player_id", "season"])
        .agg(games=("fantasy_points_ppr", "size"), ppg=("fantasy_points_ppr", "mean"))
        .reset_index().rename(columns={"player_id": "gsis_id"}))

rows, unmatched = [], []
for y in YEARS:
    adp = pd.read_csv(ROOT / f"data/adp/historical/adp_ppr_{y}.csv")
    adp = adp[adp.position == "RB"].sort_values("adp").head(30).copy()
    adp["adp_rank"] = range(1, len(adp) + 1)
    adp["nname"] = adp.name.map(norm_name)
    cand = meta_rb[(meta_rb.rookie_season <= y) & (meta_rb.last_season >= y)]
    for _, r in adp.iterrows():
        hits = cand[cand.nname == r.nname]
        if len(hits) == 0:
            # fallback 0 (data-quality fix, documented; symmetric, no player named
            # in code): FFC writes initials unspaced ("CJ Anderson") where the
            # nflverse table writes them spaced ("C.J. Anderson"), which the
            # script-07 normalizer turns into "cj anderson" vs "c j anderson".
            # Compare with all whitespace removed.  Caught 4 rows on the RB panel.
            sq = r.nname.replace(" ", "")
            hits = cand[cand.nname.str.replace(" ", "", regex=False) == sq]
            if len(hits):
                print(f"INITIALS match {y} {r['name']} -> {list(hits.display_name)}")
        if len(hits) == 0:
            fl = r.nname.split()[0] + " " + r.nname.split()[-1]
            hits = cand[cand.nname.map(lambda s: (s.split()[0] + " " + s.split()[-1])
                                       if s else "") == fl]
            if len(hits):
                print(f"FIRST+LAST match {y} {r['name']} -> {list(hits.display_name)}")
        if len(hits) == 0:
            hits = meta[(meta.nname == r.nname) & (meta.rookie_season <= y)
                        & (meta.last_season >= y)]
            if len(hits):
                print(f"POSITION-DRIFT match {y} {r['name']}: {list(hits.position)}")
        if len(hits) == 0:
            hits = meta_rb[meta_rb.nname == r.nname]
            if len(hits):
                print(f"OUT-OF-WINDOW match {y} {r['name']}: "
                      f"{list(zip(hits.rookie_season, hits.last_season))}")
        if len(hits) == 0:
            unmatched.append((y, r["name"]))
            continue
        if len(hits) > 1:
            print(f"AMBIGUOUS {y} {r['name']}: {list(hits.gsis_id)}")
            act = hits[(hits.rookie_season <= y) & (hits.last_season >= y)]
            hits = act if len(act) == 1 else hits.sort_values("rookie_season").tail(1)
        h = hits.iloc[0]
        rows.append(dict(year=y, name=r["name"], gsis_id=h.gsis_id, adp=r.adp,
                         adp_rank=r.adp_rank, rookie_season=int(h.rookie_season),
                         exp=y - int(h.rookie_season)))

panel = pd.DataFrame(rows).merge(
    real, left_on=["gsis_id", "year"], right_on=["gsis_id", "season"],
    how="left").drop(columns=["season"])
panel["games"] = panel.games.fillna(0).astype(int)
panel["tier"] = np.select([panel.exp == 0, panel.exp == 1], ["rookie", "soph"], "vet")
panel["in_fit"] = panel.games >= 4

print(f"\npanel rows: {len(panel)} (target 300); unmatched: {unmatched}")
print(f"0 included games: {(panel.games == 0).sum()}; 1-3: "
      f"{((panel.games > 0) & (panel.games < 4)).sum()}; dropped from fit: "
      f"{(~panel.in_fit).sum()}")
print(panel[~panel.in_fit][["year", "name", "adp", "games", "ppg", "tier"]]
      .to_string(index=False))
print("\ntier counts (all rows):", panel.tier.value_counts().to_dict())
print("tier counts (in_fit) :", panel[panel.in_fit].tier.value_counts().to_dict())

fit = panel[panel.in_fit].copy()
ols = smf.ols("ppg ~ np.log(adp)", data=fit).fit(cov_type="HC3")
print(f"\nOLS ppg ~ log(adp): intercept {ols.params['Intercept']:.3f}, "
      f"slope {ols.params['np.log(adp)']:.3f} (se {ols.bse['np.log(adp)']:.3f}), "
      f"R2 {ols.rsquared:.3f}, n {int(ols.nobs)}")
ols_fe = smf.ols("ppg ~ np.log(adp) + C(year)", data=fit).fit(cov_type="HC3")
print(f"year-FE slope: {ols_fe.params['np.log(adp)']:.3f}")

iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
iso.fit(np.log(fit.adp.values), fit.ppg.values)
panel["m_ols"] = ols.params["Intercept"] + ols.params["np.log(adp)"] * np.log(panel.adp)
panel["m_iso"] = iso.predict(np.log(panel.adp.values))
panel["resid_iso"] = panel.ppg - panel.m_iso
panel["resid_ols"] = panel.ppg - panel.m_ols
print(f"in-sample RMSE: OLS {np.sqrt((panel.loc[panel.in_fit,'resid_ols']**2).mean()):.3f}, "
      f"isotonic {np.sqrt((panel.loc[panel.in_fit,'resid_iso']**2).mean()):.3f}")

tv = (panel[panel.in_fit].groupby("tier")
      .agg(n=("resid_iso", "size"),
           tau2_iso=("resid_iso", lambda x: x.var(ddof=1)),
           tau2_ols=("resid_ols", lambda x: x.var(ddof=1)),
           mean_resid_iso=("resid_iso", "mean"))
      .reindex(["rookie", "soph", "vet"]).reset_index())
# bootstrap CIs for tau^2 (player-year rows, 4000 reps, seeded)
rng = np.random.default_rng(40823)
for t in ["rookie", "soph", "vet"]:
    r = panel.loc[panel.in_fit & (panel.tier == t), "resid_iso"].values
    if len(r) >= 2:
        bs = [np.var(rng.choice(r, len(r), replace=True), ddof=1) for _ in range(4000)]
        tv.loc[tv.tier == t, ["tau2_lo", "tau2_hi"]] = np.percentile(bs, [2.5, 97.5])
tv.to_csv(ROOT / "results/tier_variances_rb_11yr.csv", index=False)
print("\ntau^2 by tier (RB):")
print(tv.round(3).to_string(index=False))
o = tv.set_index("tier").tau2_iso
print(f"rookie > soph > vet? {o['rookie'] > o['soph'] > o['vet']}  "
      f"(WR: 24.5 / 7.9 / 11.3, ordering also failed)")

panel.to_csv(ROOT / "results/market_prior_rb_11yr.csv", index=False)
knots = pd.DataFrame({"log_adp": iso.X_thresholds_, "m": iso.y_thresholds_})
knots.to_csv(ROOT / "results/market_prior_iso_knots_rb_11yr.csv", index=False)
lv = np.unique(np.round(iso.y_thresholds_, 9))
print(f"\nisotonic: {len(lv)} unique levels, m from {iso.y_thresholds_.max():.2f} "
      f"to {iso.y_thresholds_.min():.2f}; ADP fit range "
      f"{fit.adp.min():.1f}..{fit.adp.max():.1f}")
kk = knots.assign(adp=np.exp(knots.log_adp))
print("step boundaries (ADP -> level):")
print(kk[kk.m.diff().fillna(0) != 0][["adp", "m"]].round(3).to_string(index=False))

# monotonicity diagnostic: is the RAW binned relation monotone, or does isotonic
# have to flatten a reversal?  (pre-specified as an anomaly check, not a refit)
fit["bin"] = pd.qcut(fit.adp, 10, duplicates="drop")
bb = fit.groupby("bin", observed=True).agg(n=("ppg", "size"), adp=("adp", "mean"),
                                           ppg=("ppg", "mean"))
print("\nraw decile means (isotonic must flatten any reversal here):")
print(bb.round(3).to_string())

# ---- (pi, Sigma) export for §J ----
pi = pd.read_csv(ROOT / "data/adp/rb_top30_adp_2026.csv")[["rb_adp_rank", "name", "team", "adp"]]
mt = pd.read_csv(ROOT / "data/meta/rb_top30_meta.csv")[["name", "gsis_id", "rookie_season"]]
pi = pi.merge(mt, on="name", how="left")
pi["exp_2026"] = 2026 - pi.rookie_season
pi["tier"] = np.select([pi.exp_2026 == 0, pi.exp_2026 == 1], ["rookie", "soph"], "vet")
pi.loc[pi.rookie_season.isna(), "tier"] = "rookie"
pi["pi_ppg"] = np.interp(np.log(pi.adp), knots.log_adp, knots.m)
pi["sigma_diag"] = pi.tier.map(tv.set_index("tier").tau2_iso)
pi["position"] = "RB"
pi.to_csv(ROOT / "results/sectionJ_pi_sigma_rb_11yr.csv", index=False)
print("\nwrote market_prior_rb.csv, tier_variances_rb.csv, "
      "market_prior_iso_knots_rb.csv, sectionJ_pi_sigma_rb.csv")

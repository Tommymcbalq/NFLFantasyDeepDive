"""§H5 — does the ADP market price age (and, for RB, prior-season workload)?

Panel: FFC PPR ADP 2015-2024 (2025 ADP unavailable at source, as documented in §6.1),
top-30 by ADP per position per year.
  WR: market residual R = resid_iso from results/edge_panel.csv (§6.1 isotonic ADP->PPG).
  RB: the §G3 analogue built here identically -- isotonic regression of realized PPG on
      log ADP, monotone decreasing, fit on rows with >= 4 included games (games under the
      §G1 rule: REG, carries + targets > 1). R = PPG - m_iso(ADP).

Spec, as pre-registered: R ~ age + age^2 + era + era x age + era x age^2 (+ prior-season
touches for RB). Age centred at the position panel mean (affects nothing but conditioning).
"era" on a 2015-2024 ADP panel can only contrast 2015-16 (era 2) with 2017-24 (era 3):
era_late = 1{year >= 2017}. That limitation is stated, not worked around.
Inference: OLS, SEs clustered by season (10 clusters, use_t -> t with 9 df), exactly as
§6.2. HC3 reported as robustness.
Temporal holdout: refit on 2015-2022, compare holdout MSE on 2023-2024 against the zero
prediction (market efficiency), same construction as §6.3.

MULTIPLE TESTING: round-4's FDR family is {H5 tests, I3 tests} and §I3 is being run
separately, so RAW uncorrected p-values are the reported quantity here. A within-§H5 BH
correction is shown as PROVISIONAL only; the binding decision is made against the joint
{H5, I3} correction at consolidation.

Outputs: results/sectionH_h5.csv, results/rb_market_prior.csv
"""
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression

ROOT = "/Users/thomasmcnamee/NFL"
YEARS = range(2015, 2025)
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
ALIASES = {"hollywood brown": "marquise brown"}


def log(*a):
    print(*a, flush=True)


def norm_name(s):
    s = re.sub(r"[^a-z ]", "",
               str(s).lower().replace(".", " ").replace("-", " ").replace("'", ""))
    return " ".join(t for t in s.split() if t not in SUFFIXES)


# ---------------------------------------------------------------- RB market prior (§G3)
meta = pd.read_csv(f"{ROOT}/data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "display_name", "position", "rookie_season",
                            "last_season", "birth_date"])
meta = meta.dropna(subset=["gsis_id"]).copy()
meta["nname"] = meta.display_name.map(norm_name)
meta_rb = meta[meta.position == "RB"]

frames = []
for y in YEARS:
    df = pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
                     usecols=["player_id", "season", "season_type", "targets", "carries",
                              "fantasy_points_ppr"], low_memory=False)
    d = df[(df.season_type == "REG") &
           (df.targets.fillna(0) + df.carries.fillna(0) > 1)]
    frames.append(d)
wk = pd.concat(frames, ignore_index=True)
real = (wk.groupby(["player_id", "season"])
        .agg(games=("fantasy_points_ppr", "size"), ppg=("fantasy_points_ppr", "mean"))
        .reset_index().rename(columns={"player_id": "gsis_id"}))

rows, unmatched = [], []
for y in YEARS:
    adp = pd.read_csv(f"{ROOT}/data/adp/historical/adp_ppr_{y}.csv")
    adp = adp[adp.position == "RB"].sort_values("adp").head(30).copy()
    adp["adp_rank"] = range(1, len(adp) + 1)
    adp["nname"] = adp.name.map(norm_name).replace(ALIASES)
    cand = meta_rb[(meta_rb.rookie_season <= y) & (meta_rb.last_season >= y)]
    for _, r in adp.iterrows():
        hits = cand[cand.nname == r.nname]
        if len(hits) == 0:
            fl = r.nname.split()[0] + " " + r.nname.split()[-1]
            hits = cand[cand.nname.map(lambda s: (s.split()[0] + " " + s.split()[-1])
                                       if s else "") == fl]
            if len(hits):
                log(f"  FIRST+LAST match {y} {r['name']} -> {list(hits.display_name)}")
        if len(hits) == 0:
            hits = meta[(meta.nname == r.nname) & (meta.rookie_season <= y)
                        & (meta.last_season >= y)]
            if len(hits):
                log(f"  POSITION-DRIFT match {y} {r['name']}: {list(hits.position)}")
        if len(hits) == 0:
            hits = meta_rb[meta_rb.nname == r.nname]
            if len(hits):
                log(f"  OUT-OF-WINDOW match {y} {r['name']}")
        if len(hits) == 0:
            unmatched.append((y, r["name"]))
            continue
        if len(hits) > 1:
            act = hits[(hits.rookie_season <= y) & (hits.last_season >= y)]
            log(f"  AMBIGUOUS {y} {r['name']}: {len(hits)} rows; keeping active-in-{y}")
            hits = act if len(act) == 1 else hits.tail(1)
        h = hits.iloc[0]
        rows.append(dict(year=y, name=r["name"], gsis_id=h.gsis_id, adp=r.adp,
                         adp_rank=r.adp_rank, rookie_season=int(h.rookie_season),
                         exp=y - int(h.rookie_season), birth_date=h.birth_date))

rb = pd.DataFrame(rows).merge(real, left_on=["gsis_id", "year"],
                              right_on=["gsis_id", "season"], how="left").drop(columns="season")
rb["games"] = rb.games.fillna(0).astype(int)
rb["in_fit"] = rb.games >= 4
log(f"RB ADP panel rows {len(rb)} (target 300); unmatched {unmatched}; "
    f"dropped from fit (<4 games) {(~rb.in_fit).sum()}")
fit = rb[rb.in_fit]
iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(
    np.log(fit.adp.values), fit.ppg.values)
rb["m_iso"] = iso.predict(np.log(rb.adp.values))
rb["resid_iso"] = rb.ppg - rb.m_iso
log(f"RB isotonic m(ADP): fit n={len(fit)}, range "
    f"{rb.m_iso.max():.2f} (ADP {rb.adp.min():.1f}) -> {rb.m_iso.min():.2f} "
    f"(ADP {rb.adp.max():.1f}); in-sample RMSE {np.sqrt((fit.ppg-iso.predict(np.log(fit.adp))).pow(2).mean()):.3f}")
rb.to_csv(f"{ROOT}/results/rb_market_prior.csv", index=False)

# ---------------------------------------------------------------- assemble H5 panel
wr = pd.read_csv(f"{ROOT}/results/edge_panel.csv")
wr = wr[["year", "gsis_id", "name", "adp", "exp", "games", "ppg", "in_fit",
         "resid_iso", "age"]].copy()
wr["position"] = "WR"

rb["age"] = (pd.Timestamp("2000-09-01").normalize() and
             (pd.to_datetime(rb.year.astype(str) + "-09-01") -
              pd.to_datetime(rb.birth_date)).dt.days / 365.25)
rb["position"] = "RB"
h5 = pd.concat([wr, rb[["year", "gsis_id", "name", "adp", "exp", "games", "ppg",
                        "in_fit", "resid_iso", "age", "position"]]], ignore_index=True)

# prior-season touches (repaired panel; 0 + indicator when there is no prior season)
pan = pd.read_csv(f"{ROOT}/data/derived/age_panel_long_repaired.csv",
                  usecols=["gsis_id", "season", "touches"])
pan["year"] = pan.season + 1
h5 = h5.merge(pan[["gsis_id", "year", "touches"]].rename(columns={"touches": "prior_touches"}),
              on=["gsis_id", "year"], how="left")
h5["no_prior"] = h5.prior_touches.isna().astype(float)
h5["prior_touches"] = h5.prior_touches.fillna(0.0)
h5 = h5[h5.in_fit].copy()
log(f"\nH5 fit rows: {h5.groupby('position').size().to_dict()}; "
    f"no prior season: {int(h5.no_prior.sum())}")

# ---------------------------------------------------------------- H5 regressions
res_rows = []
for pos in ["WR", "RB"]:
    d = h5[h5.position == pos].dropna(subset=["age", "resid_iso"]).copy()
    a0 = d.age.mean()
    d["ac"] = d.age - a0
    d["ac2"] = d.ac ** 2
    d["late"] = (d.year >= 2017).astype(float)
    d["ac_late"] = d.ac * d.late
    d["ac2_late"] = d.ac2 * d.late
    terms = ["ac", "ac2", "late", "ac_late", "ac2_late"]
    if pos == "RB":
        d["pt"] = d.prior_touches / 100.0
        terms += ["pt", "no_prior"]
    X = sm.add_constant(d[terms].to_numpy())
    m = sm.OLS(d.resid_iso.values, X).fit(cov_type="cluster",
                                          cov_kwds={"groups": d.year.values}, use_t=True)
    mh = sm.OLS(d.resid_iso.values, X).fit(cov_type="HC3")
    log(f"\n{pos}: n={len(d)}, mean age {a0:.2f}, years 2015-16 n={int((d.late==0).sum())} / "
        f"2017-24 n={int((d.late==1).sum())}; R²={m.rsquared:.4f}")
    for i, t in enumerate(terms, start=1):
        log(f"  {t:10s} β={m.params[i]:+.4f}  se_cl={m.bse[i]:.4f}  "
            f"p_raw_cluster={m.pvalues[i]:.4f}   p_HC3={mh.pvalues[i]:.4f}")
        res_rows.append(dict(position=pos, term=t, beta=m.params[i], se_cluster=m.bse[i],
                             p_raw_cluster=m.pvalues[i], p_HC3=mh.pvalues[i],
                             family="H5", n=len(d)))
    # joint tests
    def joint(names, label):
        R = np.zeros((len(names), X.shape[1]))
        for j, nm in enumerate(names):
            R[j, terms.index(nm) + 1] = 1
        w = m.wald_test(R, scalar=True)
        log(f"  JOINT {label}: F={float(w.statistic):.3f}, p_raw={float(w.pvalue):.4f}")
        res_rows.append(dict(position=pos, term=f"JOINT {label}", beta=np.nan,
                             se_cluster=np.nan, p_raw_cluster=float(w.pvalue),
                             p_HC3=np.nan, family="H5", n=len(d)))
        return float(w.pvalue)
    joint(["ac", "ac2"], "age (level)")
    joint(["ac_late", "ac2_late"], "age x era")
    joint([t for t in terms if t != "no_prior"], "all H5 terms")

    # ---- temporal holdout 2015-2022 -> 2023-2024 ----
    tr, te = d[d.year <= 2022], d[d.year >= 2023]
    Xtr = sm.add_constant(tr[terms].to_numpy())
    Xte = sm.add_constant(te[terms].to_numpy(), has_constant="add")
    mt = sm.OLS(tr.resid_iso.values, Xtr).fit()
    pred = Xte @ mt.params
    mse_m, mse_0 = np.mean((te.resid_iso.values - pred)**2), np.mean(te.resid_iso.values**2)
    log(f"  HOLDOUT 2023-24 (n={len(te)}): model MSE {mse_m:.3f} vs zero {mse_0:.3f}  "
        f"-> {'IMPROVES' if mse_m < mse_0 else 'FAILS'}")
    # age-only holdout variant (no era interaction, the more parsimonious arm)
    t2 = ["ac", "ac2"] + (["pt", "no_prior"] if pos == "RB" else [])
    m2 = sm.OLS(tr.resid_iso.values, sm.add_constant(tr[t2].to_numpy())).fit()
    p2 = sm.add_constant(te[t2].to_numpy(), has_constant="add") @ m2.params
    log(f"  HOLDOUT age-only variant: MSE {np.mean((te.resid_iso.values-p2)**2):.3f} "
        f"vs zero {mse_0:.3f}")
    res_rows.append(dict(position=pos, term="HOLDOUT full", beta=mse_m, se_cluster=mse_0,
                         p_raw_cluster=np.nan, p_HC3=np.nan, family="holdout", n=len(te)))
    res_rows.append(dict(position=pos, term="HOLDOUT age-only",
                         beta=float(np.mean((te.resid_iso.values-p2)**2)), se_cluster=mse_0,
                         p_raw_cluster=np.nan, p_HC3=np.nan, family="holdout", n=len(te)))

out = pd.DataFrame(res_rows)
# PROVISIONAL within-H5 BH (the binding correction is joint with §I3, applied elsewhere)
mask = out.family.eq("H5") & out.term.str.startswith(("ac", "pt", "late", "no_prior"))
p = out.loc[mask, "p_raw_cluster"].values
order = np.argsort(p)
bh = np.empty_like(p)
bh[order] = np.minimum.accumulate((p[order] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
out.loc[mask, "bh_q_provisional_H5_only"] = bh
log("\nPROVISIONAL within-§H5 BH (NOT the binding correction; joint {H5, I3} FDR applies):")
log(out.loc[mask, ["position", "term", "p_raw_cluster", "bh_q_provisional_H5_only"]]
    .sort_values("p_raw_cluster").to_string(index=False))
out.to_csv(f"{ROOT}/results/sectionH_h5.csv", index=False)
log("\nwrote results/sectionH_h5.csv, results/rb_market_prior.csv")

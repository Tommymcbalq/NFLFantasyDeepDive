"""§6.1 market prior curve m(ADP) + tier variances tau^2(e) (EDA_PLAN.md §6.1).

Panel: FFC PPR ADP 2015-2024 (2025 unavailable at source), top-30 WRs by ADP each
year, joined to realized SAME-season PPG (weekly_raw, §1 inclusion rule: REG,
targets > 1). Name -> gsis_id via players_meta with normalization (lowercase, strip
punctuation, strip Jr/Sr/II/III/IV suffixes); every unmatched name reported.
Pre-registered: rows with < 4 included games are kept in the panel (flagged) but
excluded from the m(.) fit and from tau^2 residual variances.

Fits: OLS PPG ~ log(ADP) (baseline; year-FE sensitivity); isotonic regression,
monotone DECREASING in ADP (headline). tau^2(e) = Var(PPG - m_iso(ADP)) by
experience tier at the ADP year (rookie/soph/vet via rookie_season).

Outputs: results/market_prior_11yr.csv, results/tier_variances_11yr.csv
"""
import re
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.isotonic import IsotonicRegression

ROOT = "/Users/thomasmcnamee/NFL"
YEARS = range(2015, 2026)   # 2025 ADP pulled 2026-08-18 (window 08-25 -> 09-01, 8470 drafts)
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
# identity resolution only (documented in section6a_notes.md): FFC nickname -> meta name
ALIASES = {"hollywood brown": "marquise brown"}

def norm_name(s):
    s = re.sub(r"[^a-z ]", "", str(s).lower().replace(".", " ").replace("-", " ").replace("'", ""))
    toks = [t for t in s.split() if t not in SUFFIXES]
    return " ".join(toks)

# ---------- name -> gsis lookup (WRs active in the given year) ----------
meta = pd.read_csv(f"{ROOT}/data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "display_name", "position", "rookie_season",
                            "last_season"])
meta = meta.dropna(subset=["gsis_id"]).copy()
meta["nname"] = meta.display_name.map(norm_name)
meta_wr = meta[meta.position == "WR"]

# ---------- realized PPG per player-season under the inclusion rule ----------
frames = []
for y in YEARS:
    df = pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
                     usecols=["player_id", "position", "season", "season_type",
                              "targets", "fantasy_points_ppr"], low_memory=False)
    # no position filter here: the join is by gsis_id from the WR ADP board, and
    # nflverse retro-tags some career positions (e.g. a 2015 board WR appears as TE)
    frames.append(df[(df.season_type == "REG") & (df.targets > 1)])
wk = pd.concat(frames, ignore_index=True)
real = (wk.groupby(["player_id", "season"])
          .agg(games=("fantasy_points_ppr", "size"), ppg=("fantasy_points_ppr", "mean"))
          .reset_index().rename(columns={"player_id": "gsis_id"}))

# ---------- build panel ----------
rows, unmatched = [], []
for y in YEARS:
    adp = pd.read_csv(f"{ROOT}/data/adp/historical/adp_ppr_{y}.csv")
    adp = adp[adp.position == "WR"].sort_values("adp").head(30).copy()
    adp["adp_rank"] = range(1, len(adp) + 1)
    adp["nname"] = adp.name.map(norm_name).replace(ALIASES)
    cand = meta_wr[(meta_wr.rookie_season <= y) & (meta_wr.last_season >= y)]
    for _, r in adp.iterrows():
        hits = cand[cand.nname == r.nname]
        if len(hits) == 0:
            # fall back 1: first+last token match among WRs active in y
            # (handles meta middle initials, e.g. "Charles D. Johnson")
            fl = r.nname.split()[0] + " " + r.nname.split()[-1]
            hits = cand[cand.nname.map(
                lambda s: (s.split()[0] + " " + s.split()[-1]) if s else "") == fl]
            if len(hits):
                print(f"FIRST+LAST match {y} {r['name']} -> "
                      f"{list(hits.display_name)}")
        if len(hits) == 0:
            # fall back 2: any position, active in year y (meta position drift,
            # e.g. a WR later reclassified TE); the row came from a WR ADP board
            hits = meta[(meta.nname == r.nname) & (meta.rookie_season <= y)
                        & (meta.last_season >= y)]
            if len(hits):
                print(f"POSITION-DRIFT match {y} {r['name']}: meta position "
                      f"{list(hits.position)}")
        if len(hits) == 0:
            # fall back 3: WR ever (meta season bounds off by a year)
            hits = meta_wr[meta_wr.nname == r.nname]
            if len(hits):
                print(f"OUT-OF-WINDOW match {y} {r['name']}: windows "
                      f"{list(zip(hits.rookie_season, hits.last_season))}")
        if len(hits) == 0:
            unmatched.append((y, r["name"]))
            continue
        if len(hits) > 1:
            # disambiguate by activity window closest to y; report
            hits = hits.sort_values("rookie_season")
            print(f"AMBIGUOUS {y} {r['name']}: {len(hits)} meta rows "
                  f"{list(hits.gsis_id)}; keeping the one active in {y}")
            act = hits[(hits.rookie_season <= y) & (hits.last_season >= y)]
            hits = act if len(act) == 1 else hits.tail(1)
        h = hits.iloc[0]
        rows.append(dict(year=y, name=r["name"], gsis_id=h.gsis_id, adp=r.adp,
                         adp_rank=r.adp_rank, rookie_season=int(h.rookie_season),
                         exp=y - int(h.rookie_season)))

panel = pd.DataFrame(rows)
panel = panel.merge(real, left_on=["gsis_id", "year"], right_on=["gsis_id", "season"],
                    how="left").drop(columns=["season"])
panel["games"] = panel.games.fillna(0).astype(int)
panel["tier"] = np.select([panel.exp == 0, panel.exp == 1], ["rookie", "soph"], "vet")
panel["in_fit"] = panel.games >= 4          # pre-registered fit requirement

print(f"\npanel rows: {len(panel)} (target {10*30}); unmatched: {unmatched}")
print(f"rows with 0 included games: {(panel.games==0).sum()}; "
      f"1-3 games: {((panel.games>0)&(panel.games<4)).sum()}; "
      f"dropped from fit (<4 games): {(~panel.in_fit).sum()}")
print(panel[~panel.in_fit][["year", "name", "adp", "games", "ppg", "tier"]].to_string(index=False))

fit = panel[panel.in_fit].copy()

# ---------- m(ADP) ----------
# baseline OLS on log ADP
ols = smf.ols("ppg ~ np.log(adp)", data=fit).fit(cov_type="HC3")
print("\nOLS ppg ~ log(adp):")
print(pd.DataFrame({"est": ols.params, "se": ols.bse}).round(3))
print(f"R2 = {ols.rsquared:.3f}, n = {int(ols.nobs)}")
# year-FE sensitivity
ols_fe = smf.ols("ppg ~ np.log(adp) + C(year)", data=fit).fit(cov_type="HC3")
print(f"year-FE slope: {ols_fe.params['np.log(adp)']:.3f} "
      f"(se {ols_fe.bse['np.log(adp)']:.3f}) vs pooled {ols.params['np.log(adp)']:.3f}")

# headline: isotonic, monotone decreasing in ADP (fit on log ADP, same ordering)
iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
iso.fit(np.log(fit.adp.values), fit.ppg.values)

panel["m_ols"] = ols.params["Intercept"] + ols.params["np.log(adp)"] * np.log(panel.adp)
panel["m_iso"] = iso.predict(np.log(panel.adp.values))
panel["resid_iso"] = panel.ppg - panel.m_iso
panel["resid_ols"] = panel.ppg - panel.m_ols

rmse_ols = np.sqrt((fit.ppg - fit.m_ols.reindex(fit.index) if False else
                    (panel.loc[panel.in_fit, "resid_ols"]**2)).mean())
rmse_iso = np.sqrt((panel.loc[panel.in_fit, "resid_iso"]**2).mean())
print(f"\nin-sample RMSE: OLS {rmse_ols:.3f}, isotonic {rmse_iso:.3f} "
      f"(isotonic has many free levels; not a model-selection criterion)")

# ---------- tau^2 by tier (headline residuals = isotonic) ----------
tv = (panel[panel.in_fit].groupby("tier")
      .agg(n=("resid_iso", "size"),
           tau2_iso=("resid_iso", lambda x: x.var(ddof=1)),
           tau2_ols=("resid_ols", lambda x: x.var(ddof=1)),
           mean_resid_iso=("resid_iso", "mean"))
      .reindex(["rookie", "soph", "vet"]).reset_index())
tv.to_csv(f"{ROOT}/results/tier_variances_11yr.csv", index=False)
print("\ntau^2 by tier:")
print(tv.round(3).to_string(index=False))
ordering = tv.set_index("tier").tau2_iso
print(f"\ntau2(rookie) > tau2(soph) > tau2(vet)? "
      f"{ordering['rookie'] > ordering['soph'] > ordering['vet']}  "
      f"(rookie {ordering['rookie']:.1f}, soph {ordering['soph']:.1f}, vet {ordering['vet']:.1f})")

panel.to_csv(f"{ROOT}/results/market_prior_11yr.csv", index=False)

# save isotonic knots for downstream use (step function: x=log adp, y=fitted)
knots = pd.DataFrame({"log_adp": iso.X_thresholds_, "m": iso.y_thresholds_})
knots.to_csv(f"{ROOT}/results/market_prior_iso_knots_11yr.csv", index=False)
print(f"\nisotonic step levels: {len(np.unique(iso.y_thresholds_))} unique; "
      f"range m: {iso.y_thresholds_.min():.2f}..{iso.y_thresholds_.max():.2f}; "
      f"ADP range fit: {fit.adp.min():.1f}..{fit.adp.max():.1f}")

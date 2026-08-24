"""§S1 — Is the era-flat market prior defensible?

The asymmetry being tested: the LIKELIHOOD mu-hat is recency-weighted (season half-life
h = 1 yr, script 01), while the PRIOR m(ADP) and tau^2(tier) pool 2015-2025 with EQUAL
season weight (script 07/49).  That asymmetry was never justified, only inherited.

Pre-registered, in this order, on the 11-season panel (results/market_prior_11yr.csv):
  S1a  season trend in isotonic residuals: OLS r ~ (year - 2020), HC3.  A non-zero slope
       means the pooled curve is systematically mislevelled in recent years.
  S1b  pre/post 2022 residual means (the owner's split), Welch t-test.
  S1c  17-game era split (<=2020 vs >=2021), Welch t-test.  Distinct from S1b: the
       schedule change is a mechanism, the 2022 split is not.
  S1d  season-weighted refit, half-lives h in {1, 2, 4, inf}: refit OLS and isotonic with
       sample weights w = 2^{-(2025 - year)/h}; report the curve level shift at the ADP
       deciles.  DIAGNOSTIC ONLY — adoption requires beating flat out of sample (script 51).
Nothing here is adopted on the basis of an in-sample fit.
"""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from sklearn.isotonic import IsotonicRegression

ROOT = "/Users/thomasmcnamee/NFL"
p = pd.read_csv(f"{ROOT}/results/market_prior_11yr.csv")
fit = p[p.in_fit].copy()
fit["c_year"] = fit.year - 2020

print(f"panel in-fit rows: {len(fit)}, seasons {fit.year.min()}-{fit.year.max()}")
print("\nper-season mean isotonic residual (PPG above/below the pooled curve):")
bys = fit.groupby("year").agg(n=("resid_iso","size"), mean_r=("resid_iso","mean"),
                              sd=("resid_iso","std"), mean_ppg=("ppg","mean"))
bys["se"] = bys.sd/np.sqrt(bys.n)
print(bys.round(3).to_string())

print("\n--- S1a season trend in residuals ---")
m = smf.ols("resid_iso ~ c_year", data=fit).fit(cov_type="HC3")
print(f"slope {m.params['c_year']:+.4f} PPG/yr  se {m.bse['c_year']:.4f}  "
      f"p {m.pvalues['c_year']:.4f}  95% CI [{m.conf_int().loc['c_year',0]:+.3f}, "
      f"{m.conf_int().loc['c_year',1]:+.3f}]")
# cluster on season: 11 clusters, the honest SE for a season-level trend
mc = smf.ols("resid_iso ~ c_year", data=fit).fit(cov_type="cluster",
                                                 cov_kwds={"groups": fit.year})
print(f"season-clustered: slope {mc.params['c_year']:+.4f} se {mc.bse['c_year']:.4f} "
      f"p {mc.pvalues['c_year']:.4f}  (11 clusters)")

from scipy import stats
def split(name, mask, lab_a, lab_b):
    a, b = fit.resid_iso[mask], fit.resid_iso[~mask]
    t, pv = stats.ttest_ind(a, b, equal_var=False)
    print(f"{name}: {lab_a} mean {a.mean():+.3f} (n={len(a)}) vs {lab_b} "
          f"{b.mean():+.3f} (n={len(b)}); diff {a.mean()-b.mean():+.3f} "
          f"Welch t={t:.2f} p={pv:.4f}")

print("\n--- S1b/S1c era splits ---")
split("S1b 2022 split", fit.year >= 2022, ">=2022", "<=2021")
split("S1c 17-game era", fit.year >= 2021, ">=2021", "<=2020")

print("\n--- S1d season-weighted refit: curve level at ADP deciles ---")
x = np.log(fit.adp.values); y = fit.ppg.values
grid = np.percentile(fit.adp, [5,10,25,50,75,90,95])
out = {"adp": np.round(grid,1)}
for h, tag in [(np.inf,"flat"), (4.0,"h4"), (2.0,"h2"), (1.0,"h1")]:
    w = np.ones(len(fit)) if np.isinf(h) else 2.0**(-(2025-fit.year.values)/h)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(x, y, sample_weight=w)
    out[tag] = np.round(iso.predict(np.log(grid)), 2)
    o = smf.wls("ppg ~ np.log(adp)", data=fit, weights=w).fit()
    n_eff = w.sum()**2/(w**2).sum()
    print(f"  {tag:>4}: OLS slope {o.params['np.log(adp)']:+.3f} "
          f"int {o.params['Intercept']:.2f}  n_eff_seasons "
          f"{ (w.sum()**2/(w**2).sum()) / (len(fit)/fit.year.nunique()):.2f}")
tab = pd.DataFrame(out)
tab["h1_minus_flat"] = (tab.h1 - tab.flat).round(2)
print(tab.to_string(index=False))
tab.to_csv(f"{ROOT}/results/prior_era_weighting.csv", index=False)
bys.to_csv(f"{ROOT}/results/prior_residual_by_season.csv")

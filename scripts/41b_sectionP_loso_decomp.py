"""§P4 diagnostic — WHY the wide-panel LOSO differs from the top-30 LOSO.

This is a decomposition of an already-run, pre-specified result, not a new test.  No arm
is added, no threshold is moved, nothing is refit to a different spec.  Two anomalies to
explain, both from 41_sectionP_loso_deep.py:

  A) RB arm (ii) mean fold gain flips  +0.488 (top-30, §G6)  ->  -0.313 (wide panel),
     with 2.1x better power.  The extra data did not close the gap; it moved the point
     estimate the wrong way.
  B) WR arm (ii) mean fold gain falls  +0.695 (top-30, §7)   ->  +0.246 (wide panel),
     p .025 -> .233.

Decompositions run (all on the SINGLE wide-panel fold structure already produced, so the
estimator is held fixed and only the evaluation stratum varies):
  1. loss differential by ADP-rank stratum (<=30 vs >30) and by ADP decile;
  2. by n_eff bucket (thin vs thick history) and by B;
  3. the same, restricted to rows with prior data (B < 1), since B == 1 rows contribute
     exactly zero to the differential and dilute it;
  4. sign decomposition: where does theta* deviate from m_hat, and is the deviation
     right or wrong on average (regression of realized ppg on the deviation).

Output: results/sectionP_loso_decomp.csv
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.simplefilter("ignore")
ROOT = Path("/Users/thomasmcnamee/NFL")
pred = pd.read_csv(ROOT / "results/loso_predictions_deep.csv")
pred["adp_rank"] = pred.groupby(["pos", "year"]).adp.rank(method="first").astype(int)
fit = pred[pred.in_fit].copy()
fit["dev"] = fit.theta_star - fit.m_hat


def dmstat(df, tgt="ppg", base="m_hat", cand="theta_star"):
    d = (df[tgt] - df[base]) ** 2 - (df[tgt] - df[cand]) ** 2
    dy = d.groupby(df.year).mean()
    if len(dy) < 3 or dy.std(ddof=1) == 0:
        return np.nan, np.nan, dy
    t = float(dy.mean() / (dy.std(ddof=1) / np.sqrt(len(dy))))
    return t, float(2 * stats.t.sf(abs(t), len(dy) - 1)), dy


rows = []
for pos in ["WR", "RB"]:
    f = fit[fit.pos == pos]
    print("\n" + "=" * 72 + f"\n{pos}\n" + "=" * 72)

    print("\n--- 1. by ADP-rank stratum (all in_fit rows) ---")
    for lab, sub in [("all", f), ("adp_rank <= 30", f[f.adp_rank <= 30]),
                     ("adp_rank 31-45", f[f.adp_rank.between(31, 45)]),
                     ("adp_rank > 45", f[f.adp_rank > 45])]:
        t, p, dy = dmstat(sub)
        print(f"  {lab:>16}: n {len(sub):4d}  mean gain {dy.mean():+.3f}  "
              f"SD {dy.std(ddof=1):.3f}  folds+ {int((dy>0).sum())}/10  "
              f"t {t:+.3f}  p {p:.4f}  RMSE {np.sqrt(((sub.ppg-sub.m_hat)**2).mean()):.3f}"
              f" -> {np.sqrt(((sub.ppg-sub.theta_star)**2).mean()):.3f}")
        rows.append(dict(pos=pos, cut="adp_rank", level=lab, n=len(sub),
                         mean_gain=dy.mean(), sd=dy.std(ddof=1),
                         folds_pos=int((dy > 0).sum()), t=t, p=p))

    print("\n--- 2. by prior-data depth (rows with B<1 only) ---")
    fb = f[f.B < 1].copy()
    print(f"  rows with prior data: {len(fb)}/{len(f)}  "
          f"(B==1 contributes exactly 0 to the differential)")
    t, p, dy = dmstat(fb)
    print(f"  {'B<1 pooled':>16}: mean gain {dy.mean():+.3f}  t {t:+.3f}  p {p:.4f}")
    rows.append(dict(pos=pos, cut="prior", level="B<1 pooled", n=len(fb),
                     mean_gain=dy.mean(), sd=dy.std(ddof=1),
                     folds_pos=int((dy > 0).sum()), t=t, p=p))
    fb["neff_b"] = pd.cut(fb.n_eff, [0, 1.01, 2.01, 3.01, 99],
                          labels=["n_eff<=1", "1-2", "2-3", ">3"])
    for lv, sub in fb.groupby("neff_b", observed=True):
        t, p, dy = dmstat(sub)
        print(f"  {str(lv):>16}: n {len(sub):4d}  mean gain {dy.mean():+.3f}  "
              f"t {t:+.3f}  p {p:.4f}  mean B {sub.B.mean():.3f}")
        rows.append(dict(pos=pos, cut="n_eff", level=str(lv), n=len(sub),
                         mean_gain=dy.mean(), sd=dy.std(ddof=1),
                         folds_pos=int((dy > 0).sum()), t=t, p=p))

    print("\n--- 3. cross: ADP stratum x prior data ---")
    for lab, sub in [("<=30 & B<1", fb[fb.adp_rank <= 30]),
                     (">30 & B<1", fb[fb.adp_rank > 30])]:
        t, p, dy = dmstat(sub)
        print(f"  {lab:>16}: n {len(sub):4d}  mean gain {dy.mean():+.3f}  "
              f"SD {dy.std(ddof=1):.3f}  folds+ {int((dy>0).sum())}/10  "
              f"t {t:+.3f}  p {p:.4f}")
        rows.append(dict(pos=pos, cut="cross", level=lab, n=len(sub),
                         mean_gain=dy.mean(), sd=dy.std(ddof=1),
                         folds_pos=int((dy > 0).sum()), t=t, p=p))

    print("\n--- 4. is the deviation theta*-m_hat informative? "
          "ppg = a + b*m_hat + c*dev, cluster(year) ---")
    for lab, sub in [("all", f), ("adp_rank<=30", f[f.adp_rank <= 30]),
                     ("adp_rank>30", f[f.adp_rank > 30])]:
        X = sm.add_constant(sub[["m_hat", "dev"]])
        r = sm.OLS(sub.ppg, X).fit(cov_type="cluster",
                                   cov_kwds={"groups": sub.year})
        print(f"  {lab:>14}: b(m_hat) {r.params['m_hat']:+.3f} "
              f"({r.bse['m_hat']:.3f})   c(dev) {r.params['dev']:+.3f} "
              f"({r.bse['dev']:.3f}, p {r.pvalues['dev']:.4f})   "
              f"SD(dev) {sub.dev.std():.2f}")
        rows.append(dict(pos=pos, cut="dev_reg", level=lab, n=len(sub),
                         mean_gain=r.params["dev"], sd=r.bse["dev"],
                         folds_pos=np.nan, t=r.tvalues["dev"],
                         p=r.pvalues["dev"]))
        # c == 1 would mean the deviation is fully warranted; c == 0 that it is noise
        w = (r.params["dev"] - 1) / r.bse["dev"]
        print(f"                  H0: c = 1 (deviation fully warranted) z = {w:+.2f}, "
              f"p = {2*stats.norm.sf(abs(w)):.4f}")

pd.DataFrame(rows).to_csv(ROOT / "results/sectionP_loso_decomp.csv", index=False)
print("\nwrote results/sectionP_loso_decomp.csv")

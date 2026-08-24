"""§S1 follow-up — the three diagnostics that explain the bake-off result.

These are POST-HOC.  None of them is an adoption candidate and none may be adopted under
the §S2 rule; they exist to answer "why is the number what it is", which is the standing
obligation when a result contradicts a pre-registered expectation.  Every one of them was
run only after the pre-registered bake-off had been completed and recorded.

  D-A  candidate 6 decomposed on the rows it actually changes, split on whether the
       player's MOST RECENT prior season was full (>= 12 games) -- the §P interaction
       variable.  Also re-estimates the §P interaction itself on this harness.
  D-B  the robust arms (2-5) recentred by their training-fold mean offset against mu_1,
       to separate a LEVEL bias from a loss of information.
  D-C  the correction §P's finding actually implies, which is a change to the SHRINKAGE
       and not to mu_hat:  D1 market-anchor a row whose last prior season was partial;
       D2 shrink in proportion to games played.

Output: results/sectionS_diagnostics.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/Users/thomasmcnamee/NFL")
ARMS = ["a2_median", "a3_trim20", "a4_huber", "a5_p60", "a6_stable"]


def dmg(f, base, cand):
    dsq = (f.ppg - f[base]) ** 2 - (f.ppg - f[cand]) ** 2
    dy = dsq.groupby(f.year).mean()
    t = float(dy.mean() / (dy.std(ddof=1) / np.sqrt(len(dy))))
    return (float(dy.mean()), t, float(2 * stats.t.sf(abs(t), len(dy) - 1)),
            float(dy.std(ddof=1)))


def mde(sd, n=10, alpha=0.05):
    return (stats.t.ppf(1 - alpha / 2, n - 1) + stats.t.ppf(0.80, n - 1)) * sd / np.sqrt(n)


pred = pd.read_csv(ROOT / "results/sectionS_predictions.csv")
rows = []
for pos in ["WR", "RB"]:
    pan = (pd.read_csv(ROOT / f"results/market_prior_{pos.lower()}_deep.csv")
           .rename(columns={"pid": "gsis_id"}))
    f = pred[pred.pos == pos].merge(pan[["gsis_id", "year", "in_fit"]],
                                    on=["gsis_id", "year"])
    f = f[f.in_fit].copy()
    r0 = float(np.sqrt(((f.ppg - f.th_a1_mean) ** 2).mean()))
    f["changed"] = (f.a6_stable - f.a1_mean).abs() > 1e-9
    f["partial_last"] = (f._G_last < 12) & (f._n_eff > 0)

    # ---------------- D-A: candidate 6 decomposition
    for lab, sub in [("all rows", f), ("rows candidate 6 changes", f[f.changed]),
                     ("  changed & last season partial (<12g)",
                      f[f.changed & f.partial_last]),
                     ("  changed & last season full (>=12g)",
                      f[f.changed & ~f.partial_last]),
                     ("rows candidate 6 leaves alone", f[~f.changed])]:
        if len(sub) < 5:
            continue
        g, t, p, sd = dmg(sub, "th_a1_mean", "th_a6_stable")
        rows.append(dict(pos=pos, block="D-A candidate 6 decomposition", item=lab,
                         n=len(sub), mean_gain=g, dm_t=t, dm_p=p, mde80=mde(sd),
                         mean_delta_mu=float((sub.a6_stable - sub.a1_mean).mean()),
                         mean_abs_delta_mu=float((sub.a6_stable - sub.a1_mean).abs().mean())))
    # the §P interaction, re-estimated here
    for tag, sub in [("G_last >= 12", f[(f._n_eff > 0) & (f._G_last >= 12)]),
                     ("G_last < 12", f[(f._n_eff > 0) & (f._G_last < 12)])]:
        X = np.c_[np.ones(len(sub)), sub.m_hat, sub.th_a1_mean - sub.m_hat]
        beta = np.linalg.lstsq(X, sub.ppg, rcond=None)[0]
        rows.append(dict(pos=pos, block="D-A §P interaction re-estimated",
                         item=f"coef on (theta*-m) | {tag}", n=len(sub),
                         mean_gain=float(beta[2])))

    # ---------------- D-B: recentred robust arms
    for a in ARMS:
        f["adj"] = np.nan
        for Y in sorted(f.year.unique()):
            tr = f[(f.year != Y) & (f._n_eff > 0)]
            off = float((tr[a] - tr.a1_mean).mean())
            f.loc[f.year == Y, "adj"] = f.loc[f.year == Y, a] - off
        f["th_adj"] = np.where(f._n_eff == 0, f.m_hat,
                               (1 - f.B) * f.adj.fillna(0) + f.B * f.m_hat)
        g, t, p, sd = dmg(f, "th_a1_mean", "th_adj")
        rows.append(dict(pos=pos, block="D-B recentred (level bias removed)", item=a,
                         n=len(f), rmse_incumbent=r0,
                         rmse_arm=float(np.sqrt(((f.ppg - f.th_adj) ** 2).mean())),
                         mean_gain=g, dm_t=t, dm_p=p, mde80=mde(sd),
                         mean_delta_mu=float((f[f._n_eff > 0][a]
                                              - f[f._n_eff > 0].a1_mean).mean())))

    # ---------------- D-C: the shrinkage correction §P actually implies
    part = (f._G_last < 12) & (f._n_eff > 0)
    f["th_D1"] = np.where(part, f.m_hat, f.th_a1_mean)
    k = np.clip(f._G_last / 12.0, 0, 1)
    Bp = 1 - (1 - f.B) * k
    f["th_D2"] = np.where(f._n_eff == 0, f.m_hat,
                          (1 - Bp) * f.a1_mean.fillna(0) + Bp * f.m_hat)
    for nm, c in [("D1 market-anchor rows whose last season < 12g", "th_D1"),
                  ("D2 shrink in proportion to games played", "th_D2")]:
        g, t, p, sd = dmg(f, "th_a1_mean", c)
        rows.append(dict(pos=pos, block="D-C shrinkage correction (NOT ADOPTABLE)",
                         item=nm, n=len(f), n_treated=int(part.sum()),
                         rmse_incumbent=r0,
                         rmse_arm=float(np.sqrt(((f.ppg - f[c]) ** 2).mean())),
                         mean_gain=g, dm_t=t, dm_p=p, mde80=mde(sd)))

out = pd.DataFrame(rows)
out.to_csv(ROOT / "results/sectionS_diagnostics.csv", index=False)
pd.set_option("display.width", 250)
for blk, g in out.groupby("block", sort=False):
    print(f"\n=== {blk} ===")
    print(g.drop(columns=["block"]).round(4).to_string(index=False))

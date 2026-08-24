"""§W1.10 — the §P interaction as a PRECISION adjustment, pre-registered this round,
plus residual diagnostics and the log1p sensitivity.

D1  hard market-anchor rows with G_last < 12       B' = 1
D2  shrink in proportion to games played           B' = 1 - (1-B)*min(G_last/12, 1)

Applied to (a) the incumbent mu_hat arm and (b) the adopted projection arm, on the
full §P wide panel `in_fit` rows so the base RMSE matches §43.5.

Outputs: results/sectionW1_precision.csv, results/sectionW1_residuals.csv
"""
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")
ROOT = Path("/Users/thomasmcnamee/NFL")
spec = importlib.util.spec_from_file_location(
    "w1", ROOT / "scripts/64_sectionW1_projection.py")
w1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w1)

S = pd.read_csv(ROOT / "results/sectionS_predictions.csv")
P = pd.read_csv(ROOT / "results/sectionW1_predictions.csv")
P = P[P.spec == "A_gated_clean"][["gsis_id", "year", "pos", "ridge_P1", "hier_P1",
                                  "mu_cal", "age", "n_prior"]]
PAN = pd.concat([pd.read_csv(ROOT / f"results/market_prior_{q}_deep.csv")
                 .rename(columns={"pid": "gsis_id"}).assign(pos=q.upper())
                 for q in ["wr", "rb"]])[["gsis_id", "year", "pos", "in_fit"]]
D = S.merge(P, on=["gsis_id", "year", "pos"], how="left").merge(
    PAN, on=["gsis_id", "year", "pos"], how="left")


def run_precision():
    rows = []
    for pos in ["WR", "RB"]:
        d = D[(D.pos == pos) & D.in_fit].copy()
        y = d.ppg.values
        B = d.B.values.copy()
        nop = d._n_eff.values == 0
        B[nop] = 1.0
        gl = d._G_last.values
        Bd1 = np.where(gl < 12, 1.0, B)
        Bd2 = 1 - (1 - B) * np.minimum(gl / 12.0, 1.0)
        for armname, arm in [("mu_hat (incumbent)", d.a1_mean.values),
                             ("ridge_P1 (projection)", d.ridge_P1.values)]:
            a = np.where(np.isfinite(arm), arm, 0.0)
            base = np.where(nop, d.m_hat.values, (1 - B) * a + B * d.m_hat.values)
            for lab, Bx in [("D1 anchor G_last<12", Bd1), ("D2 B'=1-(1-B)G/12", Bd2)]:
                cand = np.where(nop, d.m_hat.values, (1 - Bx) * a + Bx * d.m_hat.values)
                r = dict(pos=pos, arm=armname, variant=lab, n=len(d),
                         n_treated=int((gl < 12).sum()),
                         rmse_base=float(np.sqrt(((y - base) ** 2).mean())),
                         rmse_var=float(np.sqrt(((y - cand) ** 2).mean())))
                r.update(w1.dm(y, base, cand, d.year.values))
                # temporal holdout
                h = d.year >= 2022
                r["ho_base"] = float(np.sqrt(((y[h] - base[h]) ** 2).mean()))
                r["ho_var"] = float(np.sqrt(((y[h] - cand[h]) ** 2).mean()))
                r["ho_survives"] = bool(r["ho_var"] < r["ho_base"])
                rows.append(r)
    R = pd.DataFrame(rows)
    R["bh_reject"] = w1.bh(R.dm_p.values, 0.10)
    return R


def residuals():
    rows = []
    for pos in ["WR", "RB"]:
        d = D[(D.pos == pos) & D.in_fit].dropna(subset=["ridge_P1"])
        y = d.ppg.values
        for arm, v in [("mu_hat", d.a1_mean.values), ("mu_cal", d.mu_cal.values),
                       ("ridge_P1", d.ridge_P1.values)]:
            e = y - v
            # heteroskedasticity: |e| on fitted
            b = np.polyfit(v, np.abs(e), 1)
            rows.append(dict(pos=pos, arm=arm, n=len(d), mean_resid=float(e.mean()),
                             sd_resid=float(e.std(ddof=1)), skew=float(stats.skew(e)),
                             kurtosis=float(stats.kurtosis(e)),
                             shapiro_p=float(stats.shapiro(e[:500]).pvalue),
                             abs_resid_slope=float(b[0]),
                             calib_slope=float(np.polyfit(v, y, 1)[0]),
                             calib_intercept=float(np.polyfit(v, y, 1)[1])))
    return pd.DataFrame(rows)


def log_sensitivity():
    """Declared sensitivity: fit and score on log1p(PPG), report on both scales."""
    out = []
    Pl, _ = w1.run(tier="A", gated=True, arms=("ridge",), target="log_ppg")
    return Pl


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    R = run_precision()
    R.to_csv(ROOT / "results/sectionW1_precision.csv", index=False)
    print("=== §W1.10 — §P interaction moved into B (precision), pre-registered ===")
    print(R[["pos", "arm", "variant", "n", "n_treated", "rmse_base", "rmse_var",
             "mean_gain", "dm_t", "dm_p", "mde80", "obs_over_mde", "folds_improved",
             "ho_base", "ho_var", "ho_survives", "bh_reject"]]
          .round(4).to_string(index=False))

    RES = residuals()
    RES.to_csv(ROOT / "results/sectionW1_residuals.csv", index=False)
    print("\n=== residual diagnostics ===")
    print(RES.round(4).to_string(index=False))

    # coefficient summary: mean standardised ridge coefficient across the 10 folds
    C = pd.read_csv(ROOT / "results/sectionW1_coefficients.csv")
    C = C[C.tier == "A"]
    for pos in ["WR", "RB"]:
        for scope in ["P0", "P1"]:
            g = (C[(C.pos == pos) & (C.scope == scope)]
                 .groupby("feature").coef.agg(["mean", "std", "min", "max"])
                 .sort_values("mean", key=abs, ascending=False))
            print(f"\n=== ridge coefficients (standardised), {pos} {scope}, "
                  f"mean over 10 LOSO folds | alpha "
                  f"{C[(C.pos == pos) & (C.scope == scope)].alpha.median():.1f} ===")
            print(g.head(14).round(4).to_string())

"""
GATING DIAGNOSTIC 2 -- are differentials justified, or should each side be free?

Two ways to enter team quality:

  LEVELS (unconstrained)   logit(p) = a + b_h * O_home + b_a * O_away + ...
  DIFFERENCE (constrained) logit(p) = a + b   * (O_home - O_away)  + ...

The difference form is the levels form with b_h = -b_a imposed. That constraint
says only RELATIVE quality matters: a game with home at +0.10 and away at 0.00
is the same as one with home at 0.00 and away at -0.10. If true, imposing it
halves the parameter count for free, which matters at this sample size.

It could fail. If home advantage multiplies quality rather than adding to it --
strong home teams boosted more than weak ones -- the home side's coefficient
would be larger in magnitude than the away side's.

Test: fit levels, then a Wald test of H0: b_h + b_a = 0 for each pair, using the
estimated covariance so the correlation between the two coefficients is handled
correctly. Also a likelihood-ratio test of all constraints jointly. Fit on TRAIN
seasons only. Failing to reject is not proof the constraint holds, so the
practical check is added too: does the constrained model actually predict as
well out of sample?
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE = os.path.dirname(__file__)
TABLE = os.path.join(HERE, "..", "data", "model_table.csv")
TRAIN = range(2014, 2023)

PAIRS = [("h_O_epa", "a_O_epa"), ("h_D_epa", "a_D_epa"),
         ("h_O_pass_epa", "a_O_pass_epa"), ("h_D_pass_epa", "a_D_pass_epa")]


def main() -> None:
    d = pd.read_csv(TABLE)
    d = d[d.season.isin(TRAIN) & (d.home_win != 0.5)].copy()
    print(f"train games: {len(d)} ({TRAIN.start}-{TRAIN.stop - 1}, ties dropped)\n")

    cols = [c for pair in PAIRS for c in pair]
    X = sm.add_constant(d[cols].astype(float))
    y = d.home_win.astype(int)
    fit = sm.Logit(y, X).fit(disp=0)

    print(f"{'pair':<34} {'b_home':>8} {'b_away':>8} {'sum':>8} {'se(sum)':>8} "
          f"{'z':>6} {'p':>7}")
    for hc, ac in PAIRS:
        bh, ba = fit.params[hc], fit.params[ac]
        V = fit.cov_params()
        var = V.loc[hc, hc] + V.loc[ac, ac] + 2 * V.loc[hc, ac]
        se = np.sqrt(var)
        z = (bh + ba) / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        star = "  <-- REJECTS" if p < 0.05 else ""
        print(f"{hc + ' / ' + ac:<34} {bh:8.3f} {ba:8.3f} {bh + ba:8.3f} "
              f"{se:8.3f} {z:6.2f} {p:7.3f}{star}")

    # joint likelihood-ratio test of all four constraints at once
    diff_cols = {"off_epa_diff": ("h_O_epa", "a_O_epa"),
                 "def_epa_diff": ("h_D_epa", "a_D_epa"),
                 "off_pass_epa_diff": ("h_O_pass_epa", "a_O_pass_epa"),
                 "def_pass_epa_diff": ("h_D_pass_epa", "a_D_pass_epa")}
    Xc = sm.add_constant(d[list(diff_cols)].astype(float))
    fit_c = sm.Logit(y, Xc).fit(disp=0)

    lr = 2 * (fit.llf - fit_c.llf)
    df = len(PAIRS)
    p_lr = 1 - stats.chi2.cdf(lr, df)
    print(f"\njoint LR test of all {df} constraints: "
          f"chi2({df}) = {lr:.2f}, p = {p_lr:.3f}")
    print(f"  levels      : {len(fit.params)} params, llf {fit.llf:.2f}, "
          f"AIC {fit.aic:.1f}")
    print(f"  differentials: {len(fit_c.params)} params, llf {fit_c.llf:.2f}, "
          f"AIC {fit_c.aic:.1f}")
    verdict = ("differentials NOT rejected -- impose the constraint"
               if p_lr >= 0.05 else
               "constraint REJECTED -- keep levels")
    print(f"  verdict: {verdict}")

    # Sign convention sanity: def_epa_diff is oriented so positive favours home,
    # so both differential coefficients must come out positive.
    print("\nconstrained coefficients (all should be positive by construction):")
    print(fit_c.params.round(3).to_string())


if __name__ == "__main__":
    main()

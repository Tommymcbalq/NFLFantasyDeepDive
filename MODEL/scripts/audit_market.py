"""AUDIT 1 -- is the market benchmark fair, and is log loss the right objective?

Constructs five market benchmarks with increasing amounts of "hindsight
calibration" and scores them all on the same games:

  M0  spread -> Phi(spread/13.5)                 zero fitted parameters
  M1  spread -> Phi(spread/sigma_hat)            1 param, sigma from TRAIN margins
  M2  logit(home_win) ~ spread, fit on TRAIN     2 params fitted on TRAIN  (current benchmark)
  M3  moneyline, multiplicative devig            0 params, market's own probability
  M4  moneyline, Shin / power / additive devig   0 params, alternative vig models

and the model (spec B) against each. Also: Brier vs log loss, Murphy
decomposition, reliability tables.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from audit_common import (load, standardise, scores, paired_boot, logloss, brier,
                          devig, spread_to_prob, calibration_table, murphy,
                          TRAIN, VALID, HOLDOUT, SPEC_B, RESULTS)

OUT = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def main():
    d = load()
    d = standardise(d, SPEC_B, TRAIN)
    d["blk"] = d.season * 100 + d.week

    # ---------------- market probability constructions ----------------
    d["m0_fixed135"] = spread_to_prob(d.spread_home, 13.5)
    sd_margin = d.loc[d.season.isin(TRAIN), "margin"].std()
    # sigma for the *conditional* margin given spread, not the marginal sd
    tr = d[d.season.isin(TRAIN)]
    resid_sd = (tr.margin - tr.spread_home).std()
    d["m1_fixed_hat"] = spread_to_prob(d.spread_home, resid_sd)
    mkt_fit = sm.Logit(tr.home_win, sm.add_constant(tr[["spread_home"]])).fit(disp=0)
    d["m2_logit_spread"] = mkt_fit.predict(sm.add_constant(d[["spread_home"]]))
    for meth, col in [("multiplicative", "m3_ml_mult"), ("shin", "m4_ml_shin"),
                      ("power", "m4_ml_power"), ("additive", "m4_ml_add")]:
        d[col] = devig(d.home_moneyline, d.away_moneyline, meth)

    p(f"train margin sd = {sd_margin:.2f}; sd(margin - spread) on TRAIN = {resid_sd:.2f}")
    p(f"market logit fit: intercept {mkt_fit.params['const']:+.4f}, "
      f"spread {mkt_fit.params['spread_home']:.4f} "
      f"(implied probit-equivalent sigma ~ "
      f"{1/ (mkt_fit.params['spread_home']/1.7011):.2f})")
    ml_na = d.m3_ml_mult.isna().sum()
    p(f"moneyline missing on {ml_na} of {len(d)} games")
    ov = 1/ ( (d.home_moneyline.abs()*0+1) )  # placeholder
    from audit_common import american_to_dec
    over = 1/american_to_dec(d.home_moneyline) + 1/american_to_dec(d.away_moneyline)
    p(f"median moneyline overround: {np.nanmedian(over):.4f} "
      f"(mean {np.nanmean(over):.4f})")

    # ---------------- model ----------------
    fit = sm.Logit(tr.home_win, sm.add_constant(tr[SPEC_B])).fit(disp=0)
    d["model_specB"] = fit.predict(sm.add_constant(d[SPEC_B]))

    cols = ["model_specB", "m0_fixed135", "m1_fixed_hat", "m2_logit_spread",
            "m3_ml_mult", "m4_ml_shin", "m4_ml_power", "m4_ml_add"]

    rows = []
    for split, seasons in [("train", TRAIN), ("valid", VALID), ("holdout", HOLDOUT)]:
        s = d[d.season.isin(seasons)].dropna(subset=cols)
        for c in cols:
            rows.append({"split": split, "model": c, **scores(s.home_win, s[c])})
    tab = pd.DataFrame(rows)
    p("\n" + "=" * 92)
    p("MARKET BENCHMARK CONSTRUCTIONS (complete-case: games with moneyline)")
    p("=" * 92)
    for split in ["train", "valid", "holdout"]:
        p(f"\n--- {split} ---")
        p(tab[tab.split == split].set_index("model")[
            ["n", "logloss", "brier", "acc", "auc"]].round(4).to_string())
    tab.to_csv(os.path.join(RESULTS, "audit_market_benchmarks.csv"), index=False)

    # ---------------- honest gaps, holdout, blocked bootstrap ----------------
    h = d[d.season.isin(HOLDOUT)].dropna(subset=cols)
    p("\n" + "=" * 92)
    p("HOLDOUT GAP: model spec B minus each market construction (positive = model WORSE)")
    p("blocked bootstrap by season-week, 4000 reps")
    p("=" * 92)
    grows = []
    for c in cols[1:]:
        for mname, mf in [("logloss", logloss), ("brier", brier)]:
            pt, lo, hi, _ = paired_boot(h.home_win.values, h.model_specB.values,
                                        h[c].values, metric=mf, block=h.blk.values)
            grows.append({"benchmark": c, "metric": mname, "diff": pt,
                          "lo": lo, "hi": hi})
    g = pd.DataFrame(grows)
    p(g.pivot(index="benchmark", columns="metric",
              values=["diff", "lo", "hi"]).round(4).to_string())
    g.to_csv(os.path.join(RESULTS, "audit_market_gaps.csv"), index=False)

    # spread-based vs moneyline-based market: which is the better market?
    p("\n--- market vs market (holdout): which market representation is best? ---")
    for a, b in [("m2_logit_spread", "m3_ml_mult"), ("m0_fixed135", "m3_ml_mult"),
                 ("m2_logit_spread", "m0_fixed135"), ("m3_ml_mult", "m4_ml_shin")]:
        pt, lo, hi, _ = paired_boot(h.home_win.values, h[a].values, h[b].values,
                                    block=h.blk.values)
        p(f"  {a:18s} - {b:16s} logloss {pt:+.4f}  [{lo:+.4f},{hi:+.4f}]")

    # ---------------- does the metric change conclusions? ----------------
    p("\n" + "=" * 92)
    p("METRIC SENSITIVITY: rank of each model under log loss vs Brier vs acc (holdout)")
    p("=" * 92)
    hh = tab[tab.split == "holdout"].set_index("model")
    r = pd.DataFrame({"logloss_rank": hh.logloss.rank(), "brier_rank": hh.brier.rank(),
                      "acc_rank": hh.acc.rank(ascending=False),
                      "auc_rank": hh.auc.rank(ascending=False)})
    p(r.round(1).to_string())

    p("\nMurphy (Brier) decomposition, holdout:")
    mrows = []
    for c in ["model_specB", "m2_logit_spread", "m3_ml_mult"]:
        mrows.append({"model": c, **murphy(h.home_win, h[c])})
    md = pd.DataFrame(mrows)
    md["skill_vs_climatology"] = md.RES - md.REL
    p(md.round(4).to_string(index=False))

    # ---------------- reliability ----------------
    p("\n" + "=" * 92)
    p("RELIABILITY (holdout 2024-25)")
    p("=" * 92)
    for c in ["model_specB", "m2_logit_spread", "m3_ml_mult"]:
        p(f"\n-- {c} --")
        p(calibration_table(h.home_win, h[c]).round(3).to_string(index=False))

    # calibration-in-the-large / slope on holdout (Cox recalibration)
    p("\nCox recalibration on holdout: logit(y) ~ a + b*logit(p). "
      "Well-calibrated => a=0, b=1.")
    for c in ["model_specB", "m2_logit_spread", "m3_ml_mult"]:
        lp = np.log(np.clip(h[c], 1e-6, 1 - 1e-6) / (1 - np.clip(h[c], 1e-6, 1 - 1e-6)))
        f = sm.Logit(h.home_win.values, sm.add_constant(lp.values)).fit(disp=0)
        p(f"  {c:18s} a={f.params[0]:+.3f} (se {f.bse[0]:.3f})  "
          f"b={f.params[1]:.3f} (se {f.bse[1]:.3f})")

    # ---------------- how much of the gap is discrimination vs calibration? -----
    p("\n" + "=" * 92)
    p("Decompose the holdout gap: refit each model's calibration ON THE HOLDOUT")
    p("(oracle recalibration -- upper bound on what calibration repair could buy)")
    p("=" * 92)
    for c in ["model_specB", "m2_logit_spread", "m3_ml_mult"]:
        lp = np.log(np.clip(h[c], 1e-6, 1 - 1e-6) / (1 - np.clip(h[c], 1e-6, 1 - 1e-6)))
        f = sm.Logit(h.home_win.values, sm.add_constant(lp.values)).fit(disp=0)
        pr = f.predict(sm.add_constant(lp.values))
        p(f"  {c:18s} as-is {logloss(h.home_win, h[c]):.4f} -> "
          f"oracle-recalibrated {logloss(h.home_win, pr):.4f}")

    # ---------------- combination: does the model add to the market? -----------
    p("\n" + "=" * 92)
    p("DOES THE MODEL ADD ANYTHING TO THE MARKET? logit(y) ~ spread + model features")
    p("fit on TRAIN, scored on holdout. This is the only question that matters if")
    p("the market is available at prediction time.")
    p("=" * 92)
    trc = d[d.season.isin(TRAIN)].dropna(subset=cols)
    f_sp = sm.Logit(trc.home_win, sm.add_constant(trc[["spread_home"]])).fit(disp=0)
    f_both = sm.Logit(trc.home_win,
                      sm.add_constant(trc[["spread_home"] + SPEC_B])).fit(disp=0)
    p(f_both.summary2().tables[1].round(4).to_string())
    p_sp = f_sp.predict(sm.add_constant(h[["spread_home"]]))
    p_bo = f_both.predict(sm.add_constant(h[["spread_home"] + SPEC_B]))
    pt, lo, hi, _ = paired_boot(h.home_win.values, p_bo.values, p_sp.values,
                                block=h.blk.values)
    p(f"\nholdout logloss  spread-only {logloss(h.home_win, p_sp):.4f}   "
      f"spread+features {logloss(h.home_win, p_bo):.4f}")
    p(f"difference (neg = features help): {pt:+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]")

    with open(os.path.join(RESULTS, "audit_market.txt"), "w") as fh:
        fh.write("\n".join(OUT))


if __name__ == "__main__":
    main()

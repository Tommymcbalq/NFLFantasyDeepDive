r"""
Audit 8: decompose the estimator into its two independent design choices and
test them separately.

audit_oppadj.py found the four estimators ordered

    roll 0.6353  <  v1adj 0.6358  <<  v2adj 0.6397  <  v2raw 0.6402

which does NOT line up with "opponent adjustment on/off" (roll and v2raw are both
off; v1adj and v2adj are both on). It lines up with the MEMORY profile: roll and
v1adj both use the v1 memory (half-life 20 weeks, offseason discount gamma=0.70,
two-season lookback), while v2adj and v2raw use the v2 memory (a carried-over
prior scaled by kappa plus a 2-week within-season half-life).

So the design is a 2x2 and the previous comparison confounded the cells:

                       opponent adjustment
                        OFF          ON
        v1 memory      v1raw        v1adj
        v2 memory      v2raw        v2adj

v1raw was never built. This script builds it, reads the 2x2 as a factorial, and
then sweeps the one hyperparameter the cells disagree about -- the within-season
half-life H -- on the proper rolling-origin CV rather than on the two-season
inner split that tune_v2.py used (544 games, noise floor ~0.002, which is the
same size as the effect being selected on).

Also sweeps the ridge penalty to answer whether the amount of shrinkage
interacts with the value of the adjustment: an over-shrunk opponent block is a
no-op by construction, so a null at one lambda is not a null everywhere.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audit_quality as AQ
import audit_model_table as AMT
from audit_oppadj import CV_SEASONS, blocked_boot, cv_predict, ll, fold_signs

RES = os.path.join(HERE, "..", "results")
CORE = ["off_epa_noto_diff", "def_epa_noto_diff", "net_to_rate_diff"]
SUBSET = {k: AQ.METRICS[k] for k in ("epa_noto", "to_rate", "epa")}

out_lines = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out_lines.append(s)


def make_table(panel, est, **over):
    """Build quality + differential table for one estimator configuration."""
    saved = {k: getattr(AQ, k) for k in over}
    full = AQ.METRICS
    try:
        AQ.METRICS = SUBSET
        AMT.METRICS = SUBSET
        for k, v in over.items():
            setattr(AQ, k, v)
        q = AQ.build(panel, est)
        tmp = os.path.join(HERE, "..", "data", "_tmp_quality.csv")
        q.to_csv(tmp, index=False)
        qpath = os.path.join(HERE, "..", "data", "team_quality__tmp.csv")
        os.replace(tmp, qpath)
        d = AMT.build("_tmp")
    finally:
        AQ.METRICS = full
        AMT.METRICS = full
        for k, v in saved.items():
            setattr(AQ, k, v)
    return d[d.home_win != 0.5].reset_index(drop=True)


def score(d, label, ref=None):
    y, p, b, w = cv_predict(d, CORE)
    row = {"config": label, "cv_logloss": ll(y, p),
           "acc": float(((p > 0.5) == (y == 1)).mean())}
    if ref is not None:
        m, lo, hi, pv = blocked_boot(y, p, ref[1], b)
        fs = fold_signs(y, p, ref[1], b // 100)
        row.update({"delta": m, "lo": lo, "hi": hi, "boot_p": pv,
                    "folds_neg": int((fs < 0).sum())})
    return row, (y, p, b, w)


def main():
    panel = AQ.load_panel()

    P("=" * 78)
    P("A. THE 2x2: opponent adjustment x memory profile")
    P("   rolling-origin CV 2017-2023, n=1832, blocked bootstrap by season-week")
    P("=" * 78)
    cells = {}
    for est, label in [("v1adj", "v1 memory + ADJ"), ("v1raw", "v1 memory + raw"),
                       ("v2adj", "v2 memory + ADJ"), ("v2raw", "v2 memory + raw")]:
        d = make_table(panel, est)
        r, pred = score(d, label)
        cells[est] = (r, pred, d)
        P(f"  {label:20s} CV log loss {r['cv_logloss']:.4f}  acc {r['acc']:.4f}")

    P("\n  main effects and interaction (blocked bootstrap):")
    for a, b, label in [
            ("v1adj", "v1raw", "adjustment | v1 memory"),
            ("v2adj", "v2raw", "adjustment | v2 memory"),
            ("v1adj", "v2adj", "v1 memory | adjustment on"),
            ("v1raw", "v2raw", "v1 memory | adjustment off")]:
        ya, pa, bl, _ = cells[a][1]
        _, pb, _, _ = cells[b][1]
        m, lo, hi, pv = blocked_boot(ya, pa, pb, bl)
        fs = fold_signs(ya, pa, pb, bl // 100)
        P(f"    {label:28s} {m:+.4f} [{lo:+.4f},{hi:+.4f}] p={pv:.3f} "
          f"folds_neg={int((fs < 0).sum())}/7")

    # ---------------- within-season half-life sweep -----------------------
    P("\n" + "=" * 78)
    P("B. WITHIN-SEASON HALF-LIFE H (v2 estimator, adjustment ON)")
    P("   tune_v2.py picked H=2 on a 544-game inner split whose noise floor")
    P("   (~0.002) is the size of the effect. Re-swept on the 1832-game CV.")
    P("=" * 78)
    ref = None
    rows = []
    for H in [1.0, 2.0, 4.0, 6.0, 10.0, 16.0, 24.0, 100.0]:
        d = make_table(panel, "v2adj", H_WITHIN=H)
        r, pred = score(d, f"H_within={H:g}", ref)
        if ref is None:
            ref = pred
        rows.append(r)
        P(f"  H={H:6.1f}  ll={r['cv_logloss']:.4f}  acc={r['acc']:.4f}"
          + ("" if "delta" in r is None else
             f"  vs H=1: {r.get('delta', float('nan')):+.4f} "
             f"[{r.get('lo', float('nan')):+.4f},{r.get('hi', float('nan')):+.4f}]"))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "audit_halflife_sweep.csv"),
                              index=False)

    # ---------------- kappa sweep -----------------------------------------
    P("\n" + "=" * 78)
    P("C. PRIOR CARRYOVER kappa (how much of last season survives)")
    P("=" * 78)
    rows = []
    for K in [0.0, 0.3, 0.6, 0.9, 1.0]:
        d = make_table(panel, "v2adj", KAPPA=K)
        r, _ = score(d, f"kappa={K:g}")
        rows.append(r)
        P(f"  kappa={K:.2f}  ll={r['cv_logloss']:.4f}  acc={r['acc']:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(RES, "audit_kappa_sweep.csv"), index=False)

    # ---------------- shrinkage x adjustment interaction -------------------
    P("\n" + "=" * 78)
    P("D. DOES THE AMOUNT OF SHRINKAGE CHANGE THE VALUE OF THE ADJUSTMENT?")
    P("   an over-shrunk opponent block is a no-op by construction, so a null")
    P("   at one lambda is not a null at every lambda")
    P("=" * 78)
    P(f"  {'lambda':>8s} {'ADJ':>9s} {'raw':>9s} {'adj-raw':>9s} {'95% CI':>22s}")
    rows = []
    for lam in [2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 200.0]:
        da = make_table(panel, "v2adj", LAM_UPDATE=lam, LAM_PRIOR=lam)
        dr = make_table(panel, "v2raw", LAM_UPDATE=lam, LAM_PRIOR=lam)
        ra, pa = score(da, "adj")
        rr, pr = score(dr, "raw")
        m, lo, hi, pv = blocked_boot(pa[0], pa[1], pr[1], pa[2])
        P(f"  {lam:8.1f} {ra['cv_logloss']:9.4f} {rr['cv_logloss']:9.4f} "
          f"{m:+9.4f}   [{lo:+.4f},{hi:+.4f}] p={pv:.3f}")
        rows.append({"lam": lam, "adj": ra["cv_logloss"], "raw": rr["cv_logloss"],
                     "delta": m, "lo": lo, "hi": hi, "boot_p": pv})
    pd.DataFrame(rows).to_csv(os.path.join(RES, "audit_lambda_sweep.csv"), index=False)

    with open(os.path.join(RES, "audit_memory_2x2.txt"), "w") as f:
        f.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()

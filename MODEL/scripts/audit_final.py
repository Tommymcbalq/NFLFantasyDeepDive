r"""
Audit 10: (i) is the opponent adjustment harmful EARLY in a season, at full
power; (ii) the final feature recommendation under the pre-registered adoption
rule, cross-checked across estimators.

(i) The week-bucket win-model comparison in audit_oppadj.py had ~440 games per
bucket and could not resolve anything. The encompassing regression has 6302
team-games, so re-running it by week bucket is the powered version of the same
question. If opponent ratings are too thin early to be trusted, the coefficient
b2 on the correction should be near zero or negative in weeks 1-4 and rise later.

(ii) Feature adoption rule, pre-registered and applied without exception:
    pooled rolling-CV delta <= -0.005 vs the incumbent
    AND blocked-bootstrap 95% CI upper bound < 0
    AND negative in >= 6 of 7 folds
    AND stable coherent sign with VIF < 3.
Rolling-origin CV 2017-2023, n=1832, blocked bootstrap by season-week.
2024-2025 is not read by this file.
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audit_quality as AQ
from audit_oppadj import CV_SEASONS, blocked_boot, cv_predict, ll, fold_signs, load_table

RES = os.path.join(HERE, "..", "results")
INCUMBENT = ["off_epa_noto_diff", "def_epa_noto_diff", "net_to_rate_diff"]

out_lines = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out_lines.append(s)


def bh(p, q=0.10):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    ok = p[o] <= q * np.arange(1, n + 1) / n
    k = np.max(np.nonzero(ok)[0]) + 1 if ok.any() else 0
    keep = np.zeros(n, bool)
    keep[o[:k]] = True
    return keep


# ---------------------------------------------------------------------------
def part_i():
    P("=" * 78)
    P("(i) IS THE ADJUSTMENT HARMFUL EARLY? -- encompassing test by week bucket")
    P("    y_realised ~ b1*O_raw + b2*(O_adj - O_raw), clustered by game.")
    P("    b2 < 0 early would be the signature of thin opponent ratings")
    P("    injecting error. b2 ~ b1 means the correction is correctly scaled.")
    P("=" * 78)
    panel = AQ.load_panel()
    adj = pd.read_csv(os.path.join(HERE, "..", "data", "team_quality_v2adj.csv"))
    raw = pd.read_csv(os.path.join(HERE, "..", "data", "team_quality_v2raw.csv"))
    key = ["season", "week", "team"]
    q = adj.merge(raw, on=key, suffixes=("_adj", "_raw"))
    pan = panel[(panel.season_type == "REG") & (panel.season >= 2014)]

    rows = []
    for m in ["epa_noto", "epa"]:
        for side, L in (("off", "O"), ("def", "D")):
            d = pan[["game_id", "season", "week", "team", f"{side}_{m}"]].merge(
                q[key + [f"{L}_{m}_adj", f"{L}_{m}_raw"]], on=key).dropna()
            d["corr_term"] = d[f"{L}_{m}_adj"] - d[f"{L}_{m}_raw"]
            for name, lo_w, hi_w in [("wk 1-4", 1, 4), ("wk 5-9", 5, 9),
                                     ("wk 10-18", 10, 18)]:
                s = d[d.week.between(lo_w, hi_w)]
                X = sm.add_constant(s[[f"{L}_{m}_raw", "corr_term"]])
                f = sm.OLS(s[f"{side}_{m}"], X).fit(
                    cov_type="cluster", cov_kwds={"groups": s.game_id})
                ci = f.conf_int().loc["corr_term"]
                rows.append({"metric": m, "side": L, "bucket": name, "n": len(s),
                             "b_raw": f.params[f"{L}_{m}_raw"],
                             "b_adj": f.params["corr_term"],
                             "t_adj": f.tvalues["corr_term"],
                             "lo": ci[0], "hi": ci[1]})
    r = pd.DataFrame(rows)
    P(r.round(3).to_string(index=False))
    r.to_csv(os.path.join(RES, "audit_oppadj_by_week.csv"), index=False)
    P("\n  sd of the correction, by week bucket (is it even estimable early?)")
    for m in ["epa_noto"]:
        for L in ("O", "D"):
            c = q[f"{L}_{m}_adj"] - q[f"{L}_{m}_raw"]
            g = c.groupby(pd.cut(q.week, [0, 4, 9, 18],
                                 labels=["wk1-4", "wk5-9", "wk10-18"]),
                          observed=True).std()
            P(f"    {L}_{m}: " + "  ".join(f"{k}={v:.4f}" for k, v in g.items()))


# ---------------------------------------------------------------------------
def part_ii():
    P("\n" + "=" * 78)
    P("(ii) FINAL FEATURE RECOMMENDATION")
    P("     estimators were shown indistinguishable, so the search is run on the")
    P("     SIMPLEST one (roll: plain weighted rolling mean) and cross-checked on")
    P("     v1adj and v2adj. A feature that only survives under one estimator is")
    P("     not a feature.")
    P("=" * 78)
    NET = [f"net_{m}_diff" for m in AQ.METRICS]

    for est in ("roll", "v1adj", "v2adj"):
        d = load_table(est)
        y0, p0, b0, _ = cv_predict(d, INCUMBENT)
        P(f"\n--- {est} --- incumbent spec B: CV log loss {ll(y0, p0):.4f}  "
          f"acc {((p0 > 0.5) == (y0 == 1)).mean():.4f}")

        # candidate single-feature and small specs
        cands = {
            "spec B (incumbent)": INCUMBENT,
            "net epa_noto only": ["net_epa_noto_diff"],
            "net epa only": ["net_epa_diff"],
            "net pts_per_drive only": ["net_pts_per_drive_diff"],
            "off/def epa_noto (no TO)": ["off_epa_noto_diff", "def_epa_noto_diff"],
            "off/def epa (no TO)": ["off_epa_diff", "def_epa_diff"],
            "off/def pts_per_drive": ["off_pts_per_drive_diff", "def_pts_per_drive_diff"],
            "off/def pts_per_drive + TO": ["off_pts_per_drive_diff",
                                           "def_pts_per_drive_diff",
                                           "net_to_rate_diff"],
            "net epa_noto + net TO": ["net_epa_noto_diff", "net_to_rate_diff"],
        }
        rows = []
        for name, cols in cands.items():
            y, p, b, _ = cv_predict(d, cols)
            m, lo, hi, pv = blocked_boot(y, p, p0, b)
            fs = fold_signs(y, p, p0, b // 100)
            rows.append({"spec": name, "k": len(cols), "cv_logloss": ll(y, p),
                         "acc": float(((p > 0.5) == (y == 1)).mean()),
                         "delta_vs_B": m, "lo": lo, "hi": hi,
                         "folds_neg": int((fs < 0).sum())})
        rr = pd.DataFrame(rows).sort_values("cv_logloss")
        P(rr.round(4).to_string(index=False))
        rr.to_csv(os.path.join(RES, f"audit_final_specs_{est}.csv"), index=False)

    # ---- incremental battery beyond incumbent, on the simplest estimator ----
    P("\n" + "=" * 78)
    P("INCREMENTAL BATTERY: anything that adds to spec B (estimator = roll)")
    P("BH-FDR q=0.10 over the whole family, then the adoption rule")
    P("=" * 78)
    d = load_table("roll").dropna(subset=NET).reset_index(drop=True)
    y0, p0, b0, _ = cv_predict(d, INCUMBENT)
    P(f"incumbent on this subset: {ll(y0, p0):.4f}")
    rows = []
    for c in NET:
        if c in INCUMBENT:
            continue
        y, p, b, _ = cv_predict(d, INCUMBENT + [c])
        m, lo, hi, pv = blocked_boot(y, p, p0, b)
        fs = fold_signs(y, p, p0, b // 100)
        rows.append({"added": c, "cv_logloss": ll(y, p), "delta": m, "lo": lo,
                     "hi": hi, "boot_p": pv, "folds_neg": int((fs < 0).sum())})
    r = pd.DataFrame(rows).sort_values("delta")
    r["bh_q10"] = bh(r.boot_p.to_numpy())
    r["adopt"] = (r.delta <= -0.005) & (r.hi < 0) & (r.folds_neg >= 6)
    P(r.head(15).round(4).to_string(index=False))
    P(f"\nsurviving BH-FDR q=0.10 : {r[r.bh_q10].added.tolist() or 'NONE'}")
    P(f"passing the adoption rule: {r[r.adopt].added.tolist() or 'NONE'}")
    r.to_csv(os.path.join(RES, "audit_final_incremental.csv"), index=False)

    # ---- VIF and coefficient stability of the incumbent --------------------
    P("\n" + "=" * 78)
    P("COLLINEARITY AND SIGN COHERENCE OF THE INCUMBENT (estimator = roll)")
    P("=" * 78)
    tr = d[d.season < 2024]
    Z = ((tr[INCUMBENT] - tr[INCUMBENT].mean()) / tr[INCUMBENT].std())
    Xv = sm.add_constant(Z).to_numpy()
    for i, c in enumerate(INCUMBENT):
        P(f"  VIF {c:24s} {variance_inflation_factor(Xv, i + 1):.3f}")
    P("\n  per-fold standardised coefficients (sign stability):")
    coefs = []
    for s in CV_SEASONS:
        t = d[d.season < s]
        mu, sd = t[INCUMBENT].mean(), t[INCUMBENT].std()
        X = sm.add_constant((t[INCUMBENT] - mu) / sd, has_constant="add")
        f = sm.Logit(t.home_win.astype(int), X).fit(disp=0)
        coefs.append({"fold": s, **{c: f.params[c] for c in INCUMBENT}})
    cf = pd.DataFrame(coefs)
    P(cf.round(4).to_string(index=False))


def main():
    part_i()
    part_ii()
    with open(os.path.join(RES, "audit_final.txt"), "w") as f:
        f.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()

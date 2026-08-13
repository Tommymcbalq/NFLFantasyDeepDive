"""AUDIT 2 -- model form. All selection done on rolling-origin CV over 2014-2023.
The 2024-25 holdout is NOT touched here.

(a) Is the logit link with linear-in-features adequate?
    - Pregibon/Stukel link test: add eta^2 (and eta^3) to the fitted linear predictor.
    - Natural splines / quadratics on each feature and on the quality index.
    - Box-Tidwell-type check via fractional powers of the signed feature.
(b) Margin model: OLS (and Huber) on margin, mapped to P(win) via Phi(mu/s).
    Discards less information than the binary logit; standard result worth testing.
(c) Regularisation: L2 logit at k=3 and on the wide feature set, lambda by inner CV.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, HuberRegressor

from audit_common import (load, scores, logloss, brier, paired_boot, TRAIN,
                          VALID, HOLDOUT, SPEC_B, RESULTS)

OUT = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.append(s)

CV_TARGETS = list(range(2017, 2024))      # rolling origin: fit <s, predict s
WIDE = ["off_epa_noto_diff", "def_epa_noto_diff", "to_margin_diff",
        "off_pass_epa_diff", "off_rush_epa_diff", "def_pass_epa_diff",
        "def_rush_epa_diff", "off_sr_diff", "def_sr_diff", "off_cpoe_diff",
        "def_cpoe_diff", "off_pass_oe_diff", "pace_diff", "pace_sum",
        "div_game", "is_neutral", "rest_short_diff", "rest_mini_bye_diff",
        "rest_bye_diff", "off_early_epa_diff", "def_early_epa_diff",
        "pass_mismatch_diff"]


def zfit(tr, te, cols):
    mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1.0)
    return (tr[cols] - mu) / sd, (te[cols] - mu) / sd


# ---------------------------------------------------------------- learners
def learn_logit(tr, te, cols, **kw):
    Xtr, Xte = zfit(tr, te, cols)
    f = sm.Logit(tr.home_win.values, sm.add_constant(Xtr.values)).fit(disp=0)
    return f.predict(sm.add_constant(Xte.values, has_constant="add"))


def learn_ridge(tr, te, cols, C=1.0):
    Xtr, Xte = zfit(tr, te, cols)
    m = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=2000)
    m.fit(Xtr.values, tr.home_win.values)
    return m.predict_proba(Xte.values)[:, 1]


def learn_margin_ols(tr, te, cols, sigma=None):
    Xtr, Xte = zfit(tr, te, cols)
    f = sm.OLS(tr.margin.values, sm.add_constant(Xtr.values)).fit()
    mu_tr = f.predict(sm.add_constant(Xtr.values))
    s = np.std(tr.margin.values - mu_tr, ddof=len(cols) + 1) if sigma is None else sigma
    mu = f.predict(sm.add_constant(Xte.values, has_constant="add"))
    return norm.cdf(mu / s)


def learn_margin_ols_fitsig(tr, te, cols):
    """Same but sigma chosen to minimise TRAIN log loss (1 extra param)."""
    Xtr, Xte = zfit(tr, te, cols)
    f = sm.OLS(tr.margin.values, sm.add_constant(Xtr.values)).fit()
    mu_tr = f.predict(sm.add_constant(Xtr.values))
    grid = np.arange(8.0, 22.01, 0.25)
    lls = [logloss(tr.home_win.values, norm.cdf(mu_tr / s)) for s in grid]
    s = grid[int(np.argmin(lls))]
    mu = f.predict(sm.add_constant(Xte.values, has_constant="add"))
    return norm.cdf(mu / s), s


def learn_margin_huber(tr, te, cols):
    Xtr, Xte = zfit(tr, te, cols)
    m = HuberRegressor(epsilon=1.35, alpha=1e-6, max_iter=1000)
    m.fit(Xtr.values, tr.margin.values)
    mu_tr = m.predict(Xtr.values)
    grid = np.arange(8.0, 22.01, 0.25)
    s = grid[int(np.argmin([logloss(tr.home_win.values, norm.cdf(mu_tr / g))
                            for g in grid]))]
    return norm.cdf(m.predict(Xte.values) / s)


def learn_margin_logistic_link(tr, te, cols):
    """OLS on margin, but map with a logistic CDF (fat tails) instead of normal."""
    Xtr, Xte = zfit(tr, te, cols)
    f = sm.OLS(tr.margin.values, sm.add_constant(Xtr.values)).fit()
    mu_tr = f.predict(sm.add_constant(Xtr.values))
    grid = np.arange(4.0, 14.01, 0.1)
    s = grid[int(np.argmin([logloss(tr.home_win.values, 1/(1+np.exp(-mu_tr/g)))
                            for g in grid]))]
    mu = f.predict(sm.add_constant(Xte.values, has_constant="add"))
    return 1 / (1 + np.exp(-mu / s))


def learn_probit(tr, te, cols):
    Xtr, Xte = zfit(tr, te, cols)
    f = sm.Probit(tr.home_win.values, sm.add_constant(Xtr.values)).fit(disp=0)
    return f.predict(sm.add_constant(Xte.values, has_constant="add"))


def learn_poly(tr, te, cols, deg=2):
    Xtr, Xte = zfit(tr, te, cols)
    for c in cols:
        Xtr[c + "_sq"] = Xtr[c] ** 2
        Xte[c + "_sq"] = Xte[c] ** 2
    f = sm.Logit(tr.home_win.values, sm.add_constant(Xtr.values)).fit(disp=0)
    return f.predict(sm.add_constant(Xte.values, has_constant="add"))


def learn_spline_index(tr, te, cols, df=4):
    """Fit linear logit, then re-fit a natural cubic spline in the linear index."""
    Xtr, Xte = zfit(tr, te, cols)
    f = sm.Logit(tr.home_win.values, sm.add_constant(Xtr.values)).fit(disp=0)
    eta_tr = np.asarray(sm.add_constant(Xtr.values) @ f.params)
    eta_te = np.asarray(sm.add_constant(Xte.values, has_constant="add") @ f.params)
    kn = np.quantile(eta_tr, np.linspace(0.05, 0.95, df + 1))
    def basis(x):
        B = [x]
        for k in kn[1:-1]:
            B.append(np.maximum(x - k, 0) ** 3 - np.maximum(x - kn[-1], 0) ** 3)
        return np.column_stack(B)
    g = sm.Logit(tr.home_win.values, sm.add_constant(basis(eta_tr))).fit(disp=0)
    return g.predict(sm.add_constant(basis(eta_te), has_constant="add"))


# ---------------------------------------------------------------- CV driver
def rolling(d, learner, cols, targets=CV_TARGETS, **kw):
    ys, ps, ss = [], [], []
    for s in targets:
        tr = d[d.season < s]
        te = d[d.season == s]
        out = learner(tr, te, cols, **kw)
        if isinstance(out, tuple):
            out = out[0]
        ys.append(te.home_win.values); ps.append(np.asarray(out))
        ss.append(te.season.values * 100 + te.week.values)
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(ss)


def main():
    d = load().sort_values(["season", "week"]).reset_index(drop=True)
    d["blk"] = d.season * 100 + d.week

    # ================= (a) link / linearity diagnostics on TRAIN =================
    tr = d[d.season.isin(TRAIN)]
    Xtr = (tr[SPEC_B] - tr[SPEC_B].mean()) / tr[SPEC_B].std()
    f0 = sm.Logit(tr.home_win.values, sm.add_constant(Xtr.values)).fit(disp=0)
    eta = np.asarray(sm.add_constant(Xtr.values) @ f0.params)
    p("=" * 92)
    p("(a) LINK AND LINEARITY DIAGNOSTICS  (TRAIN 2014-2022, n=%d)" % len(tr))
    p("=" * 92)
    # Pregibon link test
    lt = sm.Logit(tr.home_win.values,
                  np.column_stack([np.ones(len(tr)), eta, eta ** 2])).fit(disp=0)
    p(f"Pregibon link test: coef on eta^2 = {lt.params[2]:+.4f} "
      f"(z={lt.tvalues[2]:+.2f}, p={lt.pvalues[2]:.3f})")
    lt3 = sm.Logit(tr.home_win.values,
                   np.column_stack([np.ones(len(tr)), eta, eta**2, eta**3])).fit(disp=0)
    p(f"  + eta^3 = {lt3.params[3]:+.4f} (z={lt3.tvalues[3]:+.2f}, p={lt3.pvalues[3]:.3f})")
    p(f"  LR test linear vs cubic-in-eta: chi2={2*(lt3.llf-f0.llf):.2f} on 2 df, "
      f"p={1-__import__('scipy.stats',fromlist=['chi2']).chi2.cdf(2*(lt3.llf-f0.llf),2):.3f}")

    # per-feature quadratic
    Xq = Xtr.copy()
    for c in SPEC_B:
        Xq[c + "_sq"] = Xq[c] ** 2
    fq = sm.Logit(tr.home_win.values, sm.add_constant(Xq.values)).fit(disp=0)
    p("\nquadratic terms added per feature (standardised):")
    for i, c in enumerate(SPEC_B):
        j = 1 + len(SPEC_B) + i
        p(f"  {c+'^2':26s} coef {fq.params[j]:+.4f}  z {fq.tvalues[j]:+.2f}  "
          f"p {fq.pvalues[j]:.3f}")
    from scipy.stats import chi2 as _chi2
    p(f"  joint LR vs linear: chi2={2*(fq.llf-f0.llf):.2f} on 3 df, "
      f"p={1-_chi2.cdf(2*(fq.llf-f0.llf),3):.3f}")

    # Box-Tidwell style: signed fractional powers |x|^g * sign(x)
    p("\nBox-Tidwell-style signed-power scan (train log-lik; g=1 is linear):")
    for g in [0.6, 0.8, 1.0, 1.25, 1.5]:
        Xg = np.sign(Xtr.values) * np.abs(Xtr.values) ** g
        fg = sm.Logit(tr.home_win.values, sm.add_constant(Xg)).fit(disp=0)
        p(f"  g={g:<5} llf={fg.llf:.2f}  (linear llf={f0.llf:.2f})")

    # empirical shape: win rate vs eta decile
    dd = tr.assign(eta=eta)
    dec = pd.qcut(dd.eta, 10)
    tabb = dd.groupby(dec, observed=True).agg(
        n=("home_win", "size"), eta=("eta", "mean"),
        emp=("home_win", "mean"))
    tabb["fitted"] = 1 / (1 + np.exp(-tabb.eta))
    tabb["se"] = np.sqrt(tabb.emp * (1 - tabb.emp) / tabb.n)
    p("\nempirical win rate by decile of the linear index:")
    p(tabb.round(3).to_string())

    # ================= CV comparison of model forms =================
    p("\n" + "=" * 92)
    p("MODEL FORM COMPARISON -- rolling-origin CV, fit on all seasons < s,")
    p("predict season s, s = 2017..2023 (pooled n below). Holdout untouched.")
    p("=" * 92)
    forms = {
        "logit linear (spec B)":      (learn_logit, SPEC_B, {}),
        "probit linear":              (learn_probit, SPEC_B, {}),
        "logit + quadratics":         (learn_poly, SPEC_B, {}),
        "logit + spline in index":    (learn_spline_index, SPEC_B, {}),
        "MARGIN ols -> Phi(mu/s_res)": (learn_margin_ols, SPEC_B, {}),
        "MARGIN ols -> Phi(mu/s_fit)": (learn_margin_ols_fitsig, SPEC_B, {}),
        "MARGIN ols -> logistic cdf":  (learn_margin_logistic_link, SPEC_B, {}),
        "MARGIN huber -> Phi":         (learn_margin_huber, SPEC_B, {}),
        "ridge C=1 (spec B)":          (learn_ridge, SPEC_B, {"C": 1.0}),
        "ridge C=0.1 (spec B)":        (learn_ridge, SPEC_B, {"C": 0.1}),
        "logit WIDE (22 feat)":        (learn_logit, WIDE, {}),
        "ridge C=1 WIDE":              (learn_ridge, WIDE, {"C": 1.0}),
        "ridge C=0.1 WIDE":            (learn_ridge, WIDE, {"C": 0.1}),
        "ridge C=0.03 WIDE":           (learn_ridge, WIDE, {"C": 0.03}),
        "MARGIN ols WIDE -> Phi":      (learn_margin_ols_fitsig, WIDE, {}),
    }
    preds = {}
    rows = []
    for name, (fn, cols, kw) in forms.items():
        y, pr, blk = rolling(d, fn, cols, **kw)
        preds[name] = (y, pr, blk)
        rows.append({"form": name, **scores(y, pr)})
    tabf = pd.DataFrame(rows).set_index("form")
    p(tabf.round(4).to_string())
    tabf.to_csv(os.path.join(RESULTS, "audit_form_cv.csv"))

    base = "logit linear (spec B)"
    yb, pb, blk = preds[base]
    p(f"\npaired blocked bootstrap vs '{base}' (negative = better than baseline):")
    brows = []
    for name in forms:
        if name == base:
            continue
        y, pr, _ = preds[name]
        pt, lo, hi, _ = paired_boot(y, pr, pb, block=blk, B=3000)
        flag = "" if (lo < 0 < hi) else "  *"
        brows.append({"form": name, "d_logloss": pt, "lo": lo, "hi": hi, "sig": flag})
        p(f"  {name:30s} {pt:+.4f}  [{lo:+.4f},{hi:+.4f}]{flag}")
    pd.DataFrame(brows).to_csv(os.path.join(RESULTS, "audit_form_boot.csv"), index=False)

    # sigma actually chosen by the margin model
    p("\nsigma chosen by the margin model in each CV fold (train-optimal):")
    for s in CV_TARGETS:
        _, sg = learn_margin_ols_fitsig(d[d.season < s], d[d.season == s], SPEC_B)
        p(f"  predict {s}: sigma = {sg:.2f}")

    # ================= ridge lambda path =================
    p("\n" + "=" * 92)
    p("(c) REGULARISATION PATH (rolling-origin CV log loss)")
    p("=" * 92)
    prows = []
    for cols, label in [(SPEC_B, "specB k=3"), (WIDE, "wide k=%d" % len(WIDE))]:
        for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 1e6]:
            y, pr, _ = rolling(d, learn_ridge, cols, C=C)
            prows.append({"set": label, "C": C, "logloss": logloss(y, pr),
                          "brier": brier(y, pr)})
    pp = pd.DataFrame(prows)
    p(pp.pivot(index="C", columns="set", values="logloss").round(4).to_string())
    pp.to_csv(os.path.join(RESULTS, "audit_ridge_path.csv"), index=False)

    with open(os.path.join(RESULTS, "audit_form.txt"), "w") as fh:
        fh.write("\n".join(OUT))


if __name__ == "__main__":
    main()

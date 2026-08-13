"""
Fit the win-probability logit and make every feature earn its place.

Split (forward-chaining, never random k-fold -- a random split would let a
Week 14 game inform a Week 3 prediction from the same season):

    TRAIN 2014-2022   VALIDATION 2023   HOLDOUT 2024-2025

Specifications are compared on VALIDATION. The holdout is scored once at the end
and is not used to choose anything.

Metrics: log loss and Brier are primary, because this is a probability model and
calibration is the product -- accuracy throws away the difference between a 51%
call and a 95% call. Accuracy and AUC are reported as secondary.

The benchmark that matters is the closing spread. A logit of the home win on the
closing spread, fit on the same training years, is what the market knows. Beating
that is the real bar; matching it is a good result. The spread is deliberately
NOT a feature -- it would swamp everything and make every other coefficient
uninterpretable.

Features are standardised on TRAIN statistics only, so coefficients are directly
comparable across features and the validation/holdout years contribute nothing
to the scaling.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

HERE = os.path.dirname(__file__)
TABLE = os.path.join(HERE, "..", "data", "model_table.csv")
RESULTS = os.path.join(HERE, "..", "results")

TRAIN = range(2014, 2023)
VALID = [2023]
HOLDOUT = [2024, 2025]

REST = ["rest_short_diff", "rest_mini_bye_diff", "rest_bye_diff"]
CONTEXT = ["is_neutral", "season_c"]

# Overall EPA and its own pass/rush components cannot both be in a model -- the
# split IS the whole, so they are alternative parameterisations, not additions.
SPECS = {
    "core (your four)":       ["off_epa_diff", "def_epa_diff", "pace_diff"] + REST,
    "+ context":              ["off_epa_diff", "def_epa_diff", "pace_diff"] + REST + CONTEXT,
    "pass/rush split":        ["off_pass_epa_diff", "off_rush_epa_diff",
                               "def_pass_epa_diff", "def_rush_epa_diff",
                               "pace_diff"] + REST + CONTEXT,
    "+ success rate":         ["off_pass_epa_diff", "off_rush_epa_diff",
                               "def_pass_epa_diff", "def_rush_epa_diff",
                               "off_sr_diff", "def_sr_diff",
                               "pace_diff"] + REST + CONTEXT,
    "+ tier2 (cpoe/pass_oe/to)": ["off_pass_epa_diff", "off_rush_epa_diff",
                               "def_pass_epa_diff", "def_rush_epa_diff",
                               "off_sr_diff", "def_sr_diff",
                               "off_cpoe_diff", "def_cpoe_diff",
                               "off_pass_oe_diff", "off_to_rate_diff",
                               "def_to_rate_diff",
                               "pace_diff"] + REST + CONTEXT,
    "+ interactions":         ["off_pass_epa_diff", "off_rush_epa_diff",
                               "def_pass_epa_diff", "def_rush_epa_diff",
                               "off_sr_diff", "def_sr_diff",
                               "off_cpoe_diff", "def_cpoe_diff",
                               "off_pass_oe_diff", "off_to_rate_diff",
                               "def_to_rate_diff", "pace_diff",
                               "pace_sum", "pace_x_quality",
                               "div_x_quality", "pass_mismatch_diff",
                               "div_game"] + REST + CONTEXT,
}


def load() -> pd.DataFrame:
    d = pd.read_csv(TABLE)
    return d[d.home_win != 0.5].copy()      # 15 ties in 12 seasons; drop


def standardise(d: pd.DataFrame, cols: list) -> pd.DataFrame:
    tr = d[d.season.isin(TRAIN)]
    out = d.copy()
    for c in cols:
        mu, sd = tr[c].mean(), tr[c].std()
        out[c] = (d[c] - mu) / (sd if sd > 0 else 1.0)
    return out


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"logloss": log_loss(y, p), "brier": brier_score_loss(y, p),
            "acc": ((p > 0.5) == (y == 1)).mean(), "auc": roc_auc_score(y, p)}


def fit_spec(d: pd.DataFrame, cols: list):
    tr = d[d.season.isin(TRAIN)]
    X = sm.add_constant(tr[cols].astype(float), has_constant="add")
    return sm.Logit(tr.home_win.astype(int), X).fit(disp=0)

def predict(fit, d: pd.DataFrame, cols: list) -> np.ndarray:
    X = sm.add_constant(d[cols].astype(float), has_constant="add")
    return fit.predict(X[fit.params.index])


def split_scores(fit, d: pd.DataFrame, cols: list) -> dict:
    out = {}
    for name, seasons in [("train", list(TRAIN)), ("valid", VALID), ("holdout", HOLDOUT)]:
        s = d[d.season.isin(seasons)]
        if s.empty:
            continue
        out[name] = evaluate(s.home_win.astype(int).to_numpy(), predict(fit, s, cols))
    return out


def main() -> None:
    d = load()
    all_cols = sorted({c for cols in SPECS.values() for c in cols})
    d = standardise(d, all_cols)
    print(f"games {len(d)}  train {d.season.isin(TRAIN).sum()}  "
          f"valid {d.season.isin(VALID).sum()}  holdout {d.season.isin(HOLDOUT).sum()}\n")

    # ---------- benchmarks ----------
    rows = []
    base_rate = d[d.season.isin(TRAIN)].home_win.mean()
    for name, seasons in [("train", list(TRAIN)), ("valid", VALID), ("holdout", HOLDOUT)]:
        s = d[d.season.isin(seasons)]
        rows.append({"model": "base rate (HFA only)", "split": name,
                     **evaluate(s.home_win.astype(int).to_numpy(),
                                np.full(len(s), base_rate))})

    mkt = fit_spec(d, ["spread_home"])
    for name, sc in split_scores(mkt, d, ["spread_home"]).items():
        rows.append({"model": "MARKET (closing spread)", "split": name, **sc})

    # ---------- specifications ----------
    fits = {}
    for spec, cols in SPECS.items():
        fit = fit_spec(d, cols)
        fits[spec] = (fit, cols)
        for name, sc in split_scores(fit, d, cols).items():
            rows.append({"model": spec, "split": name, **sc})

    res = pd.DataFrame(rows)
    piv = res.pivot_table(index="model", columns="split",
                          values=["logloss", "brier", "acc"])
    order = ["base rate (HFA only)", "MARKET (closing spread)"] + list(SPECS)
    print("=" * 88)
    print("MODEL COMPARISON  (log loss primary; lower is better)")
    print("=" * 88)
    show = res.pivot(index="model", columns="split", values="logloss").reindex(order)
    show = show[["train", "valid", "holdout"]]
    show["acc_valid"] = res[res.split == "valid"].set_index("model").acc.reindex(order)
    show["acc_hold"] = res[res.split == "holdout"].set_index("model").acc.reindex(order)
    print(show.round(4).to_string())

    res.to_csv(os.path.join(RESULTS, "model_comparison.csv"), index=False)

    # ---------- per-feature value ----------
    best_spec = "+ interactions"
    fit, cols = fits[best_spec]
    print("\n" + "=" * 88)
    print(f"PER-FEATURE VALUE  (spec: {best_spec})")
    print("univariate = that feature alone; marginal = validation log loss increase")
    print("when it is dropped from the full model. Positive marginal = it earns its place.")
    print("=" * 88)

    val = d[d.season.isin(VALID)]
    yv = val.home_win.astype(int).to_numpy()
    full_ll = log_loss(yv, np.clip(predict(fit, val, cols), 1e-6, 1 - 1e-6))

    frows = []
    for c in cols:
        uni = fit_spec(d, [c])
        uni_ll = log_loss(yv, np.clip(predict(uni, val, [c]), 1e-6, 1 - 1e-6))
        rest_cols = [x for x in cols if x != c]
        drop = fit_spec(d, rest_cols)
        drop_ll = log_loss(yv, np.clip(predict(drop, val, rest_cols), 1e-6, 1 - 1e-6))
        frows.append({"feature": c, "coef": fit.params[c], "z": fit.tvalues[c],
                      "p": fit.pvalues[c], "univar_ll": uni_ll,
                      "marginal_ll_gain": drop_ll - full_ll})
    f = pd.DataFrame(frows).sort_values("marginal_ll_gain", ascending=False)
    f["earns_place"] = np.where(f.marginal_ll_gain > 0, "yes", "no")
    print(f.round(4).to_string(index=False))
    f.to_csv(os.path.join(RESULTS, "feature_value.csv"), index=False)

    print(f"\nfull-model validation log loss: {full_ll:.4f}")
    print("coefficients are on standardised features, so magnitudes are comparable.")


if __name__ == "__main__":
    main()

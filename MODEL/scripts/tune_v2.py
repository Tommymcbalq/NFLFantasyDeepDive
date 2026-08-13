"""
Tune the v2 weighting directly on WIN-PROBABILITY LOG LOSS.

v1 tuned the feature-construction hyperparameters against one-step-ahead MSE on
the team metric itself. That is a proxy: predicting next week's team EPA well is
not the same objective as predicting who wins. Since the goal here is explicitly
the best log-loss win-probability model, the hyperparameters are chosen against
that end objective instead.

Hyperparameters searched
    kappa       how much of last season carries into this one (0 = full reset)
    lam_update  how much current-season evidence it takes to move off the prior
                (small = react fast to this season, large = stay near the prior)
    h_within    recency half-life in weeks INSIDE the current season

Selection protocol (nested, so nothing leaks):
    inner-train  2014-2020   fit the logit
    inner-valid  2021-2022   choose hyperparameters on this log loss
    valid        2023        untouched during selection
    holdout      2024-2025   scored once at the very end

The validation and holdout years play no part in choosing anything.
"""
import itertools
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import log_loss

from build_features import METRICS, load_panel
from build_features_v2 import fit_prior, fit_week

HERE = os.path.dirname(__file__)
GAMES = os.path.join(HERE, "..", "data", "games", "games.csv")
TEAM_FIXES = {"STL": "LA", "SD": "LAC", "OAK": "LV", "LAR": "LA", "JAC": "JAX"}

INNER_TRAIN = range(2014, 2021)
INNER_VALID = [2021, 2022]

USE_METRICS = ["epa", "epa_noto", "to_rate"]
KAPPAS = [0.0, 0.3, 0.5, 0.7, 0.9]
LAM_UPDATES = [5, 10, 20, 40, 80]
H_WITHINS = [4, 8, 16, 999]      # 999 = effectively no within-season decay


def quality(panel, teams, seasons, kappa, lam_update, h_within):
    rows = []
    for season in seasons:
        priors = {m: fit_prior(panel, m, teams, season) for m in USE_METRICS}
        for week in sorted(panel[panel.season == season].week.unique()):
            est = {m: fit_week(panel, m, teams, season, week, priors[m],
                               kappa=kappa, lam_update=lam_update,
                               h_within=h_within) for m in USE_METRICS}
            for i, t in enumerate(teams):
                r = {"season": season, "week": week, "team": t}
                for m in USE_METRICS:
                    r[f"O_{m}"], r[f"D_{m}"] = est[m][0][i], est[m][1][i]
                rows.append(r)
    return pd.DataFrame(rows)


def to_games(g, q):
    home = q.add_prefix("h_").rename(columns={"h_season": "season",
                                              "h_week": "week", "h_team": "home_team"})
    away = q.add_prefix("a_").rename(columns={"a_season": "season",
                                              "a_week": "week", "a_team": "away_team"})
    d = g.merge(home, on=["season", "week", "home_team"]).merge(
        away, on=["season", "week", "away_team"])
    for m in USE_METRICS:
        d[f"off_{m}_diff"] = d[f"h_O_{m}"] - d[f"a_O_{m}"]
        d[f"def_{m}_diff"] = d[f"a_D_{m}"] - d[f"h_D_{m}"]
    d["to_margin_diff"] = -(d.off_to_rate_diff + d.def_to_rate_diff)
    return d


def score(d, cols):
    tr, va = d[d.season.isin(INNER_TRAIN)], d[d.season.isin(INNER_VALID)]
    mu, sd = tr[cols].mean(), tr[cols].std()
    Xt = sm.add_constant((tr[cols] - mu) / sd, has_constant="add")
    Xv = sm.add_constant((va[cols] - mu) / sd, has_constant="add")
    fit = sm.Logit(tr.home_win.astype(int), Xt).fit(disp=0)
    return log_loss(va.home_win.astype(int), np.clip(fit.predict(Xv), 1e-6, 1 - 1e-6))


def main():
    panel = load_panel()
    teams = np.sort(panel.team.dropna().unique())
    g = pd.read_csv(GAMES)
    g = g[(g.game_type == "REG") & g.result.notna()
          & g.season.isin(list(INNER_TRAIN) + INNER_VALID)].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_FIXES)
    g["home_win"] = np.where(g.result > 0, 1.0, np.where(g.result < 0, 0.0, 0.5))
    g = g[g.home_win != 0.5]
    seasons = sorted(g.season.unique())

    SPECS = {"BASE (off+def EPA)": ["off_epa_diff", "def_epa_diff"],
             "B (TO-neutral + TO margin)": ["off_epa_noto_diff",
                                            "def_epa_noto_diff", "to_margin_diff"]}
    rows = []
    for kappa, lam, h in itertools.product(KAPPAS, LAM_UPDATES, H_WITHINS):
        q = quality(panel, teams, seasons, kappa, lam, h)
        d = to_games(g, q)
        for name, cols in SPECS.items():
            rows.append({"spec": name, "kappa": kappa, "lam_update": lam,
                         "h_within": h, "inner_valid_ll": score(d, cols)})
        print(f"kappa={kappa} lam={lam} h={h}  " + "  ".join(
            f"{r['spec'].split()[0]}={r['inner_valid_ll']:.4f}" for r in rows[-2:]),
            flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(HERE, "..", "results", "v2_grid.csv"), index=False)
    print("\n" + "=" * 70)
    for name in SPECS:
        s = res[res.spec == name].sort_values("inner_valid_ll")
        print(f"\n{name}  -- best 5 by inner-validation log loss")
        print(s.head(5).to_string(index=False))
        b = s.iloc[0]
        edge = [k for k, gr in [("kappa", KAPPAS), ("lam_update", LAM_UPDATES),
                                ("h_within", H_WITHINS)]
                if b[k] in (gr[0], gr[-1])]
        if edge:
            print(f"  NOTE: at grid edge for {edge}")


if __name__ == "__main__":
    main()

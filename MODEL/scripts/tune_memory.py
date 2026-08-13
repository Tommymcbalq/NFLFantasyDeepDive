"""
Tune the MEMORY of the team-quality estimator: how fast it forgets.

Two hyperparameters control this and both were left as guesses in the first
build, which is how a Week 8 estimate ended up drawing 56% of its weight from
prior seasons with almost no recency gradient inside the current one:

    half_life  weeks for a game's weight to halve. Long half-life = a stable,
               slow estimate that ignores recent change; short = responsive but
               noisy, since a handful of games is a small sample.
    gamma      multiplicative discount per offseason crossed. Low gamma = "last
               year barely counts, rosters and coordinators turned over"; high
               gamma = "team quality is persistent across years".

These trade off directly against lambda: a short memory means fewer effective
observations per team, which needs MORE shrinkage to stay stable. So they are
searched JOINTLY with lambda rather than one at a time.

Criterion is identical to tune_lambda.py -- one-step-ahead play-weighted
predictive MSE, TRAIN SEASONS ONLY. Scored on offensive EPA and passing EPA,
the two metrics that carry the model.
"""
import itertools

import numpy as np
import pandas as pd

from build_features import LOOKBACK_SEASONS, METRICS, _fit_week, load_panel

TRAIN_SEASONS = range(2014, 2023)

HALF_LIVES = [3, 5, 8, 12, 20, 32]
GAMMAS = [0.2, 0.4, 0.6, 0.8]
LAMBDAS = [10, 20, 40, 80]
SCORE_METRICS = ["epa", "pass_epa"]


def score(panel, teams, metric, lam, half_life, gamma) -> float:
    idx = {t: i for i, t in enumerate(teams)}
    ycol, wcol = f"off_{metric}", f"off_{METRICS[metric]}"
    se, sw = 0.0, 0.0
    keys = (panel[panel.season.isin(TRAIN_SEASONS)][["season", "week", "t"]]
            .drop_duplicates().sort_values(["season", "week"]))

    for season, week, t_target in keys.itertuples(index=False):
        prior = panel[(panel.t < t_target) & (panel.season >= season - LOOKBACK_SEASONS)]
        cur = panel[(panel.t == t_target) & panel[ycol].notna() & panel[wcol].notna()]
        if prior.empty or cur.empty:
            continue
        O, D, mu, hfa = _fit_week(prior, metric, teams, lam, t_target, season,
                                  half_life=half_life, gamma=gamma)
        if np.isnan(mu):
            continue
        ti = cur.team.map(idx).to_numpy()
        oi = cur.opponent.map(idx).to_numpy()
        ok = ~(pd.isna(ti) | pd.isna(oi))
        c = cur[ok]
        yhat = mu + O[ti[ok].astype(int)] + D[oi[ok].astype(int)] + \
            hfa * c.is_home.to_numpy(float)
        w = c[wcol].to_numpy(float)
        r = c[ycol].to_numpy(float) - yhat
        good = np.isfinite(r) & np.isfinite(w)
        se += float((w[good] * r[good] ** 2).sum())
        sw += float(w[good].sum())
    return se / sw if sw else np.nan


def main() -> None:
    panel = load_panel()
    teams = np.sort(panel.team.dropna().unique())
    print(f"joint search over half_life x gamma x lambda, train "
          f"{TRAIN_SEASONS.start}-{TRAIN_SEASONS.stop - 1}\n")

    for metric in SCORE_METRICS:
        rows = []
        for hl, gm, lam in itertools.product(HALF_LIVES, GAMMAS, LAMBDAS):
            rows.append({"half_life": hl, "gamma": gm, "lam": lam,
                         "mse": score(panel, teams, metric, lam, hl, gm)})
        df = pd.DataFrame(rows)
        best = df.loc[df.mse.idxmin()]
        df["rel"] = df.mse / best.mse

        print(f"===== {metric} =====")
        print(f"BEST: half_life={best.half_life:g} gamma={best.gamma:g} "
              f"lambda={best.lam:g}  mse={best.mse:.6g}")
        edge = []
        if best.half_life in (HALF_LIVES[0], HALF_LIVES[-1]): edge.append("half_life")
        if best.gamma in (GAMMAS[0], GAMMAS[-1]): edge.append("gamma")
        if best.lam in (LAMBDAS[0], LAMBDAS[-1]): edge.append("lambda")
        if edge:
            print(f"  WARNING: at grid edge for {', '.join(edge)} -- extend the grid")

        print("  relative MSE by (half_life, gamma), at each one's best lambda:")
        piv = df.groupby(["half_life", "gamma"]).rel.min().unstack()
        print(piv.round(4).to_string())
        print(f"  old default (half_life=20, gamma=0.7) sat here; relative cost of "
              f"the guess vs best:")
        near = df[(df.half_life == 20) & (df.gamma == 0.8)].rel.min()
        print(f"    half_life=20, gamma=0.8 -> {near:.4f}\n")
        df.to_csv(f"../results/memory_grid_{metric}.csv", index=False)


if __name__ == "__main__":
    main()

"""
Pick the ridge penalty per metric by out-of-sample predictive accuracy.

lambda controls how hard team effects are shrunk toward the league mean. It is
the one hyperparameter that materially changes the features, so it is chosen,
not guessed -- and chosen on TRAIN SEASONS ONLY so the validation and holdout
years stay clean.

Criterion: fit O, D, mu, hfa on everything strictly before week w, then predict
the team-game values actually observed in week w:

    yhat_i = mu + O_team(i) + D_opp(i) + hfa * home_i

and score by play-weighted MSE, pooled over every train (season, week). This is
one-step-ahead predictive validation, which is what the features are for. It also
has a clean Bayesian reading: minimising out-of-sample squared error over lambda
is an empirical-Bayes estimate of sigma^2 / tau^2, i.e. the model is inferring
how much genuine spread in team quality exists relative to week-to-week noise.

Under-shrinking makes the estimates chase noise; over-shrinking flattens every
team toward average and throws away the signal. The minimum trades those off.
"""
import numpy as np
import pandas as pd

from build_features import (LOOKBACK_SEASONS, METRICS, WEEKS_PER_SEASON,
                            _fit_week, load_panel)

TRAIN_SEASONS = range(2014, 2023)          # 2023 = validation, 2024-25 = holdout
GRID = [2.5, 5, 10, 20, 40, 80, 160, 320]


def score(panel: pd.DataFrame, teams: np.ndarray, metric: str, lam: float) -> float:
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
        O, D, mu, hfa = _fit_week(prior, metric, teams, lam, t_target, season)
        if np.isnan(mu):
            continue
        ti = cur.team.map(idx).to_numpy()
        oi = cur.opponent.map(idx).to_numpy()
        ok = ~(pd.isna(ti) | pd.isna(oi))
        ti, oi = ti[ok].astype(int), oi[ok].astype(int)
        c = cur[ok]
        yhat = mu + O[ti] + D[oi] + hfa * c.is_home.to_numpy(float)
        w = c[wcol].to_numpy(float)
        r = c[ycol].to_numpy(float) - yhat
        good = np.isfinite(r) & np.isfinite(w)
        se += float((w[good] * r[good] ** 2).sum())
        sw += float(w[good].sum())
    return se / sw if sw else np.nan


def main() -> None:
    panel = load_panel()
    teams = np.sort(panel.team.dropna().unique())
    print(f"tuning on train seasons {TRAIN_SEASONS.start}-{TRAIN_SEASONS.stop - 1}\n")

    best = {}
    for metric in METRICS:
        scores = {lam: score(panel, teams, metric, lam) for lam in GRID}
        lo = min(scores, key=scores.get)
        best[metric] = lo
        rel = {k: f"{v / scores[lo]:.4f}" for k, v in scores.items()}
        flag = " <-- AT GRID EDGE" if lo in (GRID[0], GRID[-1]) else ""
        print(f"{metric:14s} best lambda={lo:<6g} mse={scores[lo]:.6g}{flag}")
        print(f"{'':14s} rel mse by lambda: {rel}")

    print("\nLAMBDA = {")
    for m, l in best.items():
        print(f'    "{m}": {float(l)},')
    print("}")


if __name__ == "__main__":
    main()

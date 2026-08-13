"""
EXPERIMENT: should last season's prior decay as the current season progresses?

The claim under test: "recent data, i.e. this season, is the most important --
blanket truth." In v2 the prior is scaled by a single constant kappa that does
not depend on the week, so last season keeps the same nominal pull in Week 17 as
in Week 1. It only loses influence indirectly, as current-season evidence
accumulates and the ridge deviation term grows.

If the claim is right, giving the prior an explicit decay in week should help:

    kappa(w) = kappa_0 * 0.5 ** ((w - 1) / W_half)

W_half is the number of weeks for last season's influence to halve. W_half large
= v2's current behaviour (no explicit decay). W_half small = the prior is
essentially discarded a few weeks into the season.

This also matters for a reason beyond weighting. With no personnel data, a
mid-season change -- a quarterback going down, a coordinator fired -- can only be
expressed through how fast the estimator abandons what it believed before. If
the model cannot know WHO is playing, then decaying the prior is the only channel
through which "this team is not the team it was" can enter at all.

Nested selection, identical to tune_v2: fit 2014-2020, select on 2021-2022.
2023 and 2024-25 take no part.
"""
import itertools
import os

import numpy as np
import pandas as pd

from build_features import METRICS, load_panel
from build_features_v2 import _build_X, _design, _ridge, fit_prior
from tune_v2 import (GAMES, INNER_TRAIN, INNER_VALID, TEAM_FIXES, USE_METRICS,
                     score, to_games)

HERE = os.path.dirname(__file__)


def fit_week_decay(panel, metric, teams, season, week, prior,
                   kappa0, w_half, lam, h_within):
    """v2 stage-2 update, but the prior's weight decays with the week."""
    P_O, P_D, mu_p, hfa_p = prior
    idx = {t: i for i, t in enumerate(teams)}
    ycol, wcol = f"off_{metric}", f"off_{METRICS[metric]}"
    n_teams = len(teams)

    kappa = kappa0 * (0.5 ** ((week - 1) / w_half)) if w_half < 900 else kappa0
    base_O, base_D = kappa * P_O, kappa * P_D

    d = panel[(panel.season == season) & (panel.week < week)
              & panel[ycol].notna() & panel[wcol].notna() & (panel[wcol] > 0)]
    if d.empty:
        return base_O, base_D, mu_p, hfa_p
    d, ti, oi = _design(d, teams, idx)
    resid = d[ycol].to_numpy(float) - (base_O[ti] + base_D[oi])
    w = d[wcol].to_numpy(float) * np.exp(
        -np.log(2) * (week - d.week.to_numpy(float)) / h_within)
    w = w / w.mean()
    X = _build_X(len(d), ti, oi, d.is_home.to_numpy(float), n_teams)
    b = _ridge(X, resid, w, lam, n_teams)
    return (base_O + b[1:1 + n_teams],
            base_D + b[1 + n_teams:1 + 2 * n_teams], b[0], b[-1])


def quality(panel, teams, seasons, kappa0, w_half, lam, h_within):
    rows = []
    for season in seasons:
        priors = {m: fit_prior(panel, m, teams, season) for m in USE_METRICS}
        for week in sorted(panel[panel.season == season].week.unique()):
            est = {m: fit_week_decay(panel, m, teams, season, week, priors[m],
                                     kappa0, w_half, lam, h_within)
                   for m in USE_METRICS}
            for i, t in enumerate(teams):
                r = {"season": season, "week": week, "team": t}
                for m in USE_METRICS:
                    r[f"O_{m}"], r[f"D_{m}"] = est[m][0][i], est[m][1][i]
                rows.append(r)
    return pd.DataFrame(rows)


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

    cols = ["off_epa_noto_diff", "def_epa_noto_diff", "to_margin_diff"]
    rows = []
    for k0, wh, h in itertools.product([0.9, 1.0], [2, 4, 6, 10, 999], [2, 4, 8]):
        q = quality(panel, teams, seasons, k0, wh, 10.0, h)
        rows.append({"kappa0": k0, "w_half": wh, "h_within": h,
                     "ll": score(to_games(g, q), cols)})
        print(f"  kappa0={k0} w_half={wh:<4} h_within={h}  ll={rows[-1]['ll']:.4f}",
              flush=True)

    res = pd.DataFrame(rows).sort_values("ll")
    res.to_csv(os.path.join(HERE, "..", "results", "prior_decay_grid.csv"),
               index=False)
    print("\nbest 6 by inner-validation log loss:")
    print(res.head(6).to_string(index=False))
    nodecay = res[res.w_half == 999].ll.min()
    best = res.iloc[0]
    print(f"\nno explicit decay (w_half=999) best: {nodecay:.4f}")
    print(f"with decay best:                      {best.ll:.4f} "
          f"(w_half={best.w_half:g})")
    print("VERDICT:", "decay helps" if best.ll < nodecay - 1e-4
          else "no evidence decay helps -- v2's constant kappa is adequate")


if __name__ == "__main__":
    main()

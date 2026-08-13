"""
Team quality, v2: last season is a PRIOR, not just older data.

What was wrong with v1
----------------------
v1 threw current-season and prior-season team-games into one pool and down-
weighted the old ones by exp(-age/H) * gamma^(seasons crossed). The relative
influence of last season was then controlled only by how many weeks had elapsed,
and the tuned optimum (H=20 weeks over an 18-week season) meant almost no decay
at all. A Week 8 estimate drew 56% of its weight from prior seasons, and Week 1
of the current season counted as much as Week 7. The estimator never really
handed over to the current season.

The v2 structure
----------------
Two stages, which is the natural shape for "start from what we knew, update as
evidence arrives".

STAGE 1 (once per season). Fit team quality on the PREVIOUS seasons only:

    y_i = mu + P^O_team(i) + P^D_opp(i) + h * home_i

with season s-1 at full weight and s-2 down-weighted by rho. Ridge-penalised, so
P is already shrunk toward the league mean. This is the preseason prior.

Carryover. Rosters, coordinators and schemes turn over, so last season's
estimate is not fully believable about this year's team. The prior is scaled by
kappa in [0,1] before use: kappa=1 says team quality carries over intact,
kappa=0 says every team resets to league average. kappa is tuned, not assumed.

STAGE 2 (every week). Fit ONLY current-season games, as a deviation from the
carried-over prior. Subtract the prior's prediction from each observation and
fit what is left:

    y_i - kappa*(P^O_team(i) + P^D_opp(i))  =  mu + d_team(i) + e_opp(i) + h*home_i

penalising d and e. The final estimate is

    O_t = kappa * P^O_t + d_t          D_t = kappa * P^D_t + e_t

Why this gives exactly the behaviour we want
--------------------------------------------
d_t is a ridge estimate from current-season games only, so its magnitude is
governed by how much current-season evidence exists. In Week 1 there is none,
d_t = 0, and the estimate IS the carried-over prior. By Week 6 a team has ~400
plays and d_t has moved partway. By Week 15 current-season data dominates and
the prior is largely overwritten. The handover is automatic and is driven by
accumulating sample size rather than by a fixed decay constant -- which is the
normal-normal posterior mean, with lambda_update playing the role of the
noise-to-signal ratio.

This is exactly the "global prior, then trust this season more as it progresses"
behaviour, and unlike v1 it cannot be defeated by a long half-life.

Within-season recency is kept as a secondary weight (half-life H_within) so that
a Week 12 game still counts for more than a Week 2 game, but it now operates
only INSIDE the current season, where it means what it says.

What this still does not know
-----------------------------
Nothing here is aware of who is playing. A quarterback injury is a discrete
regime change; no weighting scheme over team-level box scores can represent it,
and tune_memory.py confirmed that forgetting faster to chase such breaks makes
predictions strictly worse everywhere else. That gap needs personnel data, not
a better decay.
"""
import os

import numpy as np
import pandas as pd

from build_features import (LAMBDA, METRICS, WEEKS_PER_SEASON, load_panel,
                            verify_no_leakage)

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "data", "team_quality_v2.csv")

# Chosen by tune_v2.py against WIN-PROBABILITY log loss on a nested inner split
# (fit 2014-2020, select on 2021-2022). 2023 and 2024-25 played no part.
# The optima are interior; several nearby settings sit within ~0.001 log loss of
# each other, so these are a robust choice from a flat region rather than a
# sharp peak, and should not be read as precisely identified.
RHO = 0.45          # weight of season s-2 relative to s-1 in the prior
KAPPA = 0.90        # how much of last season's estimate carries into this one
LAM_PRIOR = 20.0    # ridge for the stage-1 prior fit
H_WITHIN = 2.0      # recency half-life (weeks) INSIDE the current season
# NOTE the contrast with v1, which tuned to a 20-week half-life. That long memory
# was an ARTEFACT of conflating "prior season" with "old data": shortening it in
# v1 also discarded the prior-season games that were the only thing stabilising
# the estimate. Once last season is carried explicitly as a prior, the current
# season's memory can be aggressive, and the data clearly prefers it that way.
LAM_UPDATE = {m: 10.0 for m in LAMBDA}


def _design(d, teams, idx):
    ti = d.team.map(idx).to_numpy()
    oi = d.opponent.map(idx).to_numpy()
    ok = ~(pd.isna(ti) | pd.isna(oi))
    return d[ok], ti[ok].astype(int), oi[ok].astype(int)


def _ridge(X, y, w, lam, n_teams):
    Xw = X * w[:, None]
    A = X.T @ Xw
    b = Xw.T @ y
    pen = np.full(X.shape[1], lam)
    pen[0] = 0.0       # mu unpenalised: fixes the additive non-identifiability
    pen[-1] = 0.0      # home-field unpenalised
    return np.linalg.solve(A + np.diag(pen), b)


def _build_X(n, ti, oi, home, n_teams):
    X = np.zeros((n, 1 + 2 * n_teams + 1))
    X[:, 0] = 1.0
    X[np.arange(n), 1 + ti] = 1.0
    X[np.arange(n), 1 + n_teams + oi] = 1.0
    X[:, -1] = home
    return X


def fit_prior(panel, metric, teams, season, rho=RHO, lam=LAM_PRIOR):
    """Stage 1: team quality from PREVIOUS seasons only."""
    idx = {t: i for i, t in enumerate(teams)}
    ycol, wcol = f"off_{metric}", f"off_{METRICS[metric]}"
    n_teams = len(teams)
    d = panel[(panel.season.isin([season - 1, season - 2]))
              & panel[ycol].notna() & panel[wcol].notna() & (panel[wcol] > 0)]
    if len(d) < n_teams:
        z = np.zeros(n_teams)
        return z, z, np.nan, np.nan
    d, ti, oi = _design(d, teams, idx)
    w = d[wcol].to_numpy(float) * np.where(d.season.to_numpy() == season - 1, 1.0, rho)
    w = w / w.mean()
    X = _build_X(len(d), ti, oi, d.is_home.to_numpy(float), n_teams)
    b = _ridge(X, d[ycol].to_numpy(float), w, lam, n_teams)
    return b[1:1 + n_teams], b[1 + n_teams:1 + 2 * n_teams], b[0], b[-1]


def fit_week(panel, metric, teams, season, week, prior, kappa=KAPPA,
             lam_update=None, h_within=H_WITHIN):
    """Stage 2: update the prior using current-season games before `week`."""
    P_O, P_D, mu_p, hfa_p = prior
    idx = {t: i for i, t in enumerate(teams)}
    ycol, wcol = f"off_{metric}", f"off_{METRICS[metric]}"
    n_teams = len(teams)
    lam = LAM_UPDATE[metric] if lam_update is None else lam_update

    base_O, base_D = kappa * P_O, kappa * P_D
    d = panel[(panel.season == season) & (panel.week < week)
              & panel[ycol].notna() & panel[wcol].notna() & (panel[wcol] > 0)]
    if d.empty:
        # Week 1: no current-season evidence at all, so the estimate IS the
        # carried-over prior. This is the correct behaviour, not a fallback.
        return base_O, base_D, mu_p, hfa_p

    d, ti, oi = _design(d, teams, idx)
    # residual after removing what the carried-over prior already explains
    resid = d[ycol].to_numpy(float) - (base_O[ti] + base_D[oi])
    w = d[wcol].to_numpy(float) * np.exp(
        -np.log(2) * (week - d.week.to_numpy(float)) / h_within)
    w = w / w.mean()
    X = _build_X(len(d), ti, oi, d.is_home.to_numpy(float), n_teams)
    b = _ridge(X, resid, w, lam, n_teams)
    return (base_O + b[1:1 + n_teams], base_D + b[1 + n_teams:1 + 2 * n_teams],
            b[0], b[-1])


def build(panel):
    teams = np.sort(panel.team.dropna().unique())
    out = []
    for season in sorted(panel.season.unique()):
        if season - 2 < panel.season.min():
            continue
        priors = {m: fit_prior(panel, m, teams, season) for m in METRICS}
        for week in sorted(panel[panel.season == season].week.unique()):
            est = {m: fit_week(panel, m, teams, season, week, priors[m])
                   for m in METRICS}
            for i, team in enumerate(teams):
                r = {"season": season, "week": week, "team": team}
                for m in METRICS:
                    O, D, _, _ = est[m]
                    r[f"O_{m}"], r[f"D_{m}"] = O[i], D[i]
                    r[f"prior_O_{m}"] = KAPPA * priors[m][0][i]
                out.append(r)
    return pd.DataFrame(out)


def main():
    panel = load_panel()
    teams = np.sort(panel.team.dropna().unique())
    q = build(panel)
    q.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {q.shape}")

    # How fast does the estimate actually hand over from prior to current season?
    s = q[q.season == 2024]
    print("\nhandover: mean |O_epa - prior| by week (2024), i.e. how far the")
    print("estimate has moved off the preseason prior:")
    m = s.groupby("week").apply(
        lambda x: np.abs(x.O_epa - x.prior_O_epa).mean(), include_groups=False)
    print(m.loc[[1, 2, 4, 6, 8, 12, 17]].round(4).to_string())


if __name__ == "__main__":
    main()

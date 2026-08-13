"""
Turn the realized team-game panel into LEAK-FREE team quality estimates.

The estimator
-------------
For a metric m (say offensive EPA/play), every team-game is one observation:

    y_i  =  mu  +  O_{team(i)}  +  D_{opp(i)}  +  h * home_i  +  e_i

O_t is how much team t's offence adds to a neutral baseline; D_o is how much
opponent o's defence adds to whoever it faces (negative = good defence); h is
home-field. Fitting O and D jointly is what does the opponent adjustment: a team
whose raw EPA is high only because it played weak defences gets that credited to
the D terms instead of its own O.

The coefficients are fit by RIDGE, penalising O and D but not mu or h. This is
not an afterthought -- it is the shrinkage. Ridge on a group of exchangeable
effects is exactly the posterior mean of a normal-normal hierarchical model with
O_t ~ N(0, tau^2), where lambda = sigma^2 / tau^2. So a team with three games of
data is automatically pulled hard toward the league mean, and a team with thirty
is barely moved -- no separate shrinkage step, and no ad hoc "minimum games"
cutoff. Leaving mu and h unpenalised also fixes the additive non-identifiability
(you could add c to every O and subtract it from mu), so the O's and D's come
out centred on zero by construction.

Leakage control
---------------
For a game in week w of season s, the fit uses ONLY team-games strictly earlier
than (s, w). The cutoff is applied to the observation set before the fit, not to
the output, so there is no path by which the current game's own result can reach
its own features. This is asserted in verify_no_leakage().

Weighting
---------
Each observation is weighted by

    plays_i  *  exp(-(T - t_i) / H)  *  gamma^(season boundaries crossed)

- plays_i: a 70-play game is a more precise read on a team than a 40-play game.
- exponential recency with half-life H: teams drift within a season.
- gamma: an extra discount for crossing an offseason, since rosters, coordinators
  and schemes turn over. This is what lets Week 1 lean on last season without
  pretending last season is as informative as last month.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
PANEL = os.path.join(HERE, "..", "data", "team_game_panel.csv")
OUT = os.path.join(HERE, "..", "data", "team_quality.csv")

# metric -> column of play counts used as the precision weight
METRICS = {
    "epa":          "plays",
    "pass_epa":     "pass_plays",
    "rush_epa":     "rush_plays",
    "sr":           "plays",
    "pass_sr":      "pass_plays",
    "rush_sr":      "rush_plays",
    "cpoe":         "pass_plays",
    "pass_oe":      "plays",
    "sec_per_play": "neutral_plays",
    "to_rate":      "plays",
    # turnover-neutral: "how well did they move the ball", with the turnover
    # result removed so turnover margin can enter as an independent signal
    "epa_noto":     "plays",
    "sr_noto":      "plays",
    # early downs only (1st/2nd) -- less situation-dependent than all downs
    "early_epa":    "early_plays",
    "early_sr":     "early_plays",
    # turnover components split by luck content: fumbles COMMITTED is the skill
    # part (recovery is ~a coin flip), interceptions are less luck-driven
    "fumble_rate":  "plays",
    "int_rate":     "pass_plays",
}

# Ridge strength per metric. Larger = more shrinkage toward league mean.
# Chosen by tune_lambda.py: one-step-ahead predictive MSE on TRAIN seasons
# (2014-2022) only. All minima are interior to the grid. The ordering is a
# result, not an assumption: turnover rate wants the heaviest shrinkage (least
# stable week to week), rushing EPA more than passing EPA, and pass-rate-over-
# expected the least -- scheme identity persists where efficiency does not.
LAMBDA = {
    "epa": 20.0,
    "pass_epa": 20.0,
    "rush_epa": 40.0,
    "sr": 20.0,
    "pass_sr": 20.0,
    "rush_sr": 20.0,
    "cpoe": 20.0,
    "pass_oe": 10.0,
    "sec_per_play": 20.0,
    "to_rate": 80.0,
    "epa_noto": 20.0,
    "sr_noto": 20.0,
    "early_epa": 20.0,
    "early_sr": 20.0,
    "fumble_rate": 80.0,
    "int_rate": 80.0,
}

HALF_LIFE_WEEKS = 20.0    # recency half-life
SEASON_GAMMA = 0.70       # discount per offseason crossed
LOOKBACK_SEASONS = 2      # how far back to consider at all
WEEKS_PER_SEASON = 23     # absolute-week index spacing (incl. playoffs)


def load_panel() -> pd.DataFrame:
    p = pd.read_csv(PANEL)
    for side in ("off", "def"):
        p[f"{side}_to_rate"] = p[f"{side}_turnovers"] / p[f"{side}_plays"]
        p[f"{side}_fumble_rate"] = p[f"{side}_fumbles"] / p[f"{side}_plays"]
        p[f"{side}_int_rate"] = p[f"{side}_ints"] / p[f"{side}_pass_plays"]
    p["t"] = p.season * WEEKS_PER_SEASON + p.week
    return p


def _fit_week(obs: pd.DataFrame, metric: str, teams: np.ndarray,
              lam: float, t_target: int, season_target: int,
              half_life: float = None, gamma: float = None) -> tuple:
    """Weighted ridge for one metric at one (season, week) cutoff.

    half_life / gamma default to the module constants; tune_memory.py overrides
    them to search the memory profile.

    Returns (O, D, mu, hfa) with O and D indexed like `teams`.
    """
    half_life = HALF_LIFE_WEEKS if half_life is None else half_life
    gamma = SEASON_GAMMA if gamma is None else gamma
    ycol, wcol = f"off_{metric}", f"off_{METRICS[metric]}"
    d = obs[obs[ycol].notna() & obs[wcol].notna() & (obs[wcol] > 0)]
    n_teams = len(teams)
    if len(d) < n_teams:                      # not enough to identify anything
        return (np.full(n_teams, np.nan), np.full(n_teams, np.nan), np.nan, np.nan)

    idx = {t: i for i, t in enumerate(teams)}
    ti = d.team.map(idx).to_numpy()
    oi = d.opponent.map(idx).to_numpy()
    ok = ~(pd.isna(ti) | pd.isna(oi))
    d, ti, oi = d[ok], ti[ok].astype(int), oi[ok].astype(int)

    y = d[ycol].to_numpy(float)
    n = len(y)

    # weights: precision * recency * offseason discount
    recency = np.exp(-np.log(2) * (t_target - d.t.to_numpy(float)) / half_life)
    crossings = np.maximum(season_target - d.season.to_numpy(), 0)
    w = d[wcol].to_numpy(float) * recency * (gamma ** crossings)
    w = w / w.mean()

    # design: [mu | O (n_teams) | D (n_teams) | hfa]
    p = 1 + 2 * n_teams + 1
    X = np.zeros((n, p))
    X[:, 0] = 1.0
    X[np.arange(n), 1 + ti] = 1.0
    X[np.arange(n), 1 + n_teams + oi] = 1.0
    X[:, -1] = d.is_home.to_numpy(float)

    Xw = X * w[:, None]
    A = X.T @ Xw
    b = Xw.T @ y
    pen = np.full(p, lam)
    pen[0] = 0.0        # mu unpenalised -> fixes non-identifiability
    pen[-1] = 0.0       # home-field unpenalised
    beta = np.linalg.solve(A + np.diag(pen), b)

    return beta[1:1 + n_teams], beta[1 + n_teams:1 + 2 * n_teams], beta[0], beta[-1]


def build(panel: pd.DataFrame) -> pd.DataFrame:
    teams = np.sort(panel.team.dropna().unique())
    rows = []
    keys = (panel[["season", "week", "t"]].drop_duplicates()
                 .sort_values(["season", "week"]))

    for season, week, t_target in keys.itertuples(index=False):
        prior = panel[(panel.t < t_target)
                      & (panel.season >= season - LOOKBACK_SEASONS)]
        if prior.empty:
            continue
        rec = {"season": season, "week": week}
        for metric in METRICS:
            O, D, mu, hfa = _fit_week(prior, metric, teams,
                                      LAMBDA[metric], t_target, season)
            rec[f"__O_{metric}"] = O
            rec[f"__D_{metric}"] = D
            rec[f"__mu_{metric}"] = mu
            rec[f"__hfa_{metric}"] = hfa
        # sample maturity: team-games of current-season data behind these numbers
        rec["prior_games_season"] = int(
            prior[prior.season == season].groupby("team").size().median()
            if (prior.season == season).any() else 0)
        rows.append(rec)

    # explode the per-team arrays into long form
    out = []
    for rec in rows:
        for i, team in enumerate(teams):
            r = {"season": rec["season"], "week": rec["week"], "team": team,
                 "prior_games_season": rec["prior_games_season"]}
            for metric in METRICS:
                r[f"O_{metric}"] = rec[f"__O_{metric}"][i]
                r[f"D_{metric}"] = rec[f"__D_{metric}"][i]
            out.append(r)
    return pd.DataFrame(out)


def verify_no_leakage(panel: pd.DataFrame, teams: np.ndarray) -> None:
    """Assert a team's own result cannot influence its own features.

    Perturb one team-game's outcome to an absurd value, refit the week that
    game falls in, and confirm the estimates are bit-identical. If the cutoff
    were off by one week this fails loudly.
    """
    target = panel[(panel.season == 2019) & (panel.week == 8)].iloc[0]
    t_target = target.t
    prior = panel[(panel.t < t_target) & (panel.season >= 2017)]
    O1, _, _, _ = _fit_week(prior, "epa", teams, LAMBDA["epa"], t_target, 2019)

    poisoned = panel.copy()
    mask = (poisoned.season == 2019) & (poisoned.week >= 8)
    poisoned.loc[mask, "off_epa"] = 99.0
    prior2 = poisoned[(poisoned.t < t_target) & (poisoned.season >= 2017)]
    O2, _, _, _ = _fit_week(prior2, "epa", teams, LAMBDA["epa"], t_target, 2019)

    assert np.allclose(O1, O2, equal_nan=True), "LEAKAGE: current/future weeks reached the fit"
    print("leakage check passed: week-8 2019 features unchanged when weeks >=8 are poisoned")


def main() -> None:
    panel = load_panel()
    teams = np.sort(panel.team.dropna().unique())
    print(f"panel: {panel.shape}, {len(teams)} teams, "
          f"seasons {panel.season.min()}-{panel.season.max()}")
    verify_no_leakage(panel, teams)
    q = build(panel)
    q.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {q.shape}")


if __name__ == "__main__":
    main()

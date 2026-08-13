"""
Audit 4: leak-free team-quality estimates for the EXTENDED metric set, under
three estimators, so that opponent adjustment and the two-stage structure can be
tested against their own alternatives rather than assumed.

    v2adj  two-stage prior/update, WITH opponent (D) block   [current pipeline]
    v2raw  two-stage prior/update, WITHOUT opponent block    [no opp adjustment]
    v1adj  single pooled weighted ridge, WITH opponent block [previous pipeline]

v2raw is the honest control for "is opponent adjustment earning its keep": it is
the identical estimator -- same weights, same ridge, same prior/update split --
with only the opponent dummies removed, so any difference is attributable to the
adjustment and not to the shrinkage or the memory profile.

ORIENTATION
-----------
Every metric carries an explicit sign s: +1 if a HIGHER offensive value is better
FOR THE OFFENCE, -1 if it is worse (turnovers, sacks taken, three-and-outs,
starting field position measured as distance-to-goal, penalties). Differentials
are then built as

    off_diff = s * (O_home - O_away)      def_diff = s * (D_away - D_home)

which is positive-favours-home for EVERY metric, by construction rather than by
assertion. Ambiguous metrics (pace, EPA dispersion) are given s=+1 and flagged;
their direction is a question for the coefficient, not for the construction.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, "..", "data", "team_game_panel_ext.csv")
OUTDIR = os.path.join(HERE, "..", "data")

# metric -> (weight column, sign s)
METRICS = {
    # --- core efficiency -------------------------------------------------
    "epa":              ("plays", +1),
    "epa_noto":         ("plays", +1),
    "sr":               ("plays", +1),
    "sr_noto":          ("plays", +1),
    # --- play-type / down splits ----------------------------------------
    "pass_epa":         ("pass_plays", +1),
    "rush_epa":         ("rush_plays", +1),
    "dropback_epa":     ("dropbacks", +1),
    "early_epa":        ("early_plays", +1),
    "early_pass_epa":   ("early_pass_plays", +1),
    "early_rush_epa":   ("early_rush_plays", +1),
    "third_epa":        ("third_plays", +1),
    "third_conv":       ("third_plays", +1),
    # --- distribution tails and shape -----------------------------------
    "expl_rate":        ("plays", +1),
    "pass_expl_rate":   ("pass_plays", +1),
    "rush_expl_rate":   ("rush_plays", +1),
    "disaster_rate":    ("plays", -1),
    "epa_sd":           ("plays", +1),      # ambiguous sign
    "first_down_rate":  ("plays", +1),
    # --- pressure / trenches --------------------------------------------
    "sack_rate":        ("dropbacks", -1),
    "qb_hit_rate":      ("dropbacks", -1),
    "tfl_rate":         ("plays", -1),
    # --- passing detail --------------------------------------------------
    "cpoe":             ("pass_plays", +1),
    "air_epa":          ("pass_plays", +1),
    "yac_epa":          ("pass_plays", +1),
    "pass_oe":          ("plays", +1),      # tendency, ambiguous sign
    # --- turnovers -------------------------------------------------------
    "to_rate":          ("plays", -1),
    "to_all_rate":      ("plays", -1),
    "int_rate":         ("pass_plays", -1),
    "fumble_rate":      ("plays", -1),
    "fumble_lost_rate": ("plays", -1),
    # --- drive level -----------------------------------------------------
    "pts_per_drive":    ("drives", +1),
    "td_per_drive":     ("drives", +1),
    "score_rate":       ("drives", +1),
    "three_out_rate":   ("drives", -1),
    "start_yl100":      ("drives", -1),
    "plays_per_drive":  ("drives", +1),
    "rz_rate":          ("drives", +1),
    "rz_td_rate":       ("rz_drives", +1),
    "series_conv":      ("series", +1),
    # --- other -----------------------------------------------------------
    "pen_rate":         ("snaps", -1),
    "pen_yds_rate":     ("snaps", -1),
    "st_epa":           ("st_plays", +1),
    "sec_per_play":     ("neutral_plays", +1),   # ambiguous sign
}

RHO = 0.45
KAPPA = 0.90
LAM_PRIOR = 20.0
LAM_UPDATE = 10.0
H_WITHIN = 2.0
# v1 settings
V1_HALF_LIFE = 20.0
V1_GAMMA = 0.70
V1_LOOKBACK = 2
V1_LAMBDA = 20.0
WEEKS_PER_SEASON = 23


def load_panel() -> pd.DataFrame:
    p = pd.read_csv(PANEL)
    p["snaps"] = p.off_plays + p.def_plays
    for side, other in (("off", "def"), ("def", "off")):
        p[f"{side}_to_rate"] = p[f"{side}_giveaways"] / p[f"{side}_plays"]
        p[f"{side}_to_all_rate"] = p[f"{side}_to_all"] / p[f"{side}_plays"]
        p[f"{side}_int_rate"] = p[f"{side}_ints"] / p[f"{side}_pass_plays"]
        p[f"{side}_fumble_rate"] = p[f"{side}_fumbles"] / p[f"{side}_plays"]
        p[f"{side}_fumble_lost_rate"] = p[f"{side}_fumbles_lost"] / p[f"{side}_plays"]
    # penalties are charged to a team regardless of which unit was on the field,
    # so the "offensive" value is the team's own rate and the mirror is filled by
    # the opponent's; the ridge D-block then measures penalties INDUCED.
    p["off_pen_rate"] = p.pen_n / p.snaps
    p["off_pen_yds_rate"] = p.pen_yds / p.snaps
    mirror = p[["game_id", "team", "off_pen_rate", "off_pen_yds_rate"]].rename(
        columns={"team": "opponent", "off_pen_rate": "def_pen_rate",
                 "off_pen_yds_rate": "def_pen_yds_rate"})
    p = p.merge(mirror, on=["game_id", "opponent"], how="left")
    p["off_snaps"] = p["snaps"]
    p["def_snaps"] = p["snaps"]
    p["t"] = p.season * WEEKS_PER_SEASON + p.week
    return p


# ---------------------------------------------------------------------------
def _design(d, idx, with_opp):
    ti = d.team.map(idx).to_numpy()
    oi = d.opponent.map(idx).to_numpy()
    ok = ~(pd.isna(ti) | pd.isna(oi))
    d = d[ok]
    ti, oi = ti[ok].astype(int), oi[ok].astype(int)
    n, k = len(d), len(idx)
    ncol = 1 + k + (k if with_opp else 0) + 1
    X = np.zeros((n, ncol))
    X[:, 0] = 1.0
    X[np.arange(n), 1 + ti] = 1.0
    if with_opp:
        X[np.arange(n), 1 + k + oi] = 1.0
    X[:, -1] = d.is_home.to_numpy(float)
    return d, ti, oi, X


def _ridge(X, y, w, lam):
    Xw = X * w[:, None]
    A = X.T @ Xw
    b = Xw.T @ y
    pen = np.full(X.shape[1], lam)
    pen[0] = 0.0
    pen[-1] = 0.0
    return np.linalg.solve(A + np.diag(pen), b)


def _split(beta, k, with_opp):
    O = beta[1:1 + k]
    D = beta[1 + k:1 + 2 * k] if with_opp else np.zeros(k)
    return O, D


def fit_prior(panel, metric, wcol, teams, idx, season, with_opp, side="off"):
    k = len(teams)
    ycol = f"{side}_{metric}"
    wc = f"{side}_{wcol}"
    d = panel[panel.season.isin([season - 1, season - 2])
              & panel[ycol].notna() & panel[wc].notna() & (panel[wc] > 0)]
    if len(d) < k:
        return np.zeros(k), np.zeros(k)
    d, ti, oi, X = _design(d, idx, with_opp)
    w = d[wc].to_numpy(float) * np.where(d.season.to_numpy() == season - 1, 1.0, RHO)
    w = w / w.mean()
    b = _ridge(X, d[ycol].to_numpy(float), w, LAM_PRIOR)
    return _split(b, k, with_opp)


def fit_week_v2(panel, metric, wcol, teams, idx, season, week, prior, with_opp,
                side="off"):
    k = len(teams)
    ycol, wc = f"{side}_{metric}", f"{side}_{wcol}"
    P_O, P_D = prior
    base_O, base_D = KAPPA * P_O, KAPPA * P_D
    d = panel[(panel.season == season) & (panel.week < week)
              & panel[ycol].notna() & panel[wc].notna() & (panel[wc] > 0)]
    if d.empty:
        return base_O, base_D
    d, ti, oi, X = _design(d, idx, with_opp)
    resid = d[ycol].to_numpy(float) - (base_O[ti] + base_D[oi])
    w = d[wc].to_numpy(float) * np.exp(
        -np.log(2) * (week - d.week.to_numpy(float)) / H_WITHIN)
    w = w / w.mean()
    b = _ridge(X, resid, w, LAM_UPDATE)
    dO, dD = _split(b, k, with_opp)
    return base_O + dO, base_D + dD


def fit_week_v1(panel, metric, wcol, teams, idx, season, week, with_opp,
                side="off"):
    k = len(teams)
    ycol, wc = f"{side}_{metric}", f"{side}_{wcol}"
    t_target = season * WEEKS_PER_SEASON + week
    d = panel[(panel.t < t_target) & (panel.season >= season - V1_LOOKBACK)
              & panel[ycol].notna() & panel[wc].notna() & (panel[wc] > 0)]
    if len(d) < k:
        return np.zeros(k), np.zeros(k)
    d, ti, oi, X = _design(d, idx, with_opp)
    recency = np.exp(-np.log(2) * (t_target - d.t.to_numpy(float)) / V1_HALF_LIFE)
    crossings = np.maximum(season - d.season.to_numpy(), 0)
    w = d[wc].to_numpy(float) * recency * (V1_GAMMA ** crossings)
    w = w / w.mean()
    b = _ridge(X, d[ycol].to_numpy(float), w, V1_LAMBDA)
    return _split(b, k, with_opp)


def build(panel, estimator, first_season=2014):
    teams = np.sort(panel.team.dropna().unique())
    idx = {t: i for i, t in enumerate(teams)}
    with_opp = estimator in ("v2adj", "v1adj")
    two_stage = estimator.startswith("v2")
    # Without an opponent block the D-block is identically zero, so the defensive
    # rating has to come from a SECOND fit on the team's own def_ columns (what
    # it allowed). That is the honest no-adjustment control: identical shrinkage
    # and memory, opponent identity simply ignored on both sides.
    sides = ["off"] if with_opp else ["off", "def"]
    rows = []
    for season in sorted(s for s in panel.season.unique() if s >= first_season):
        if two_stage:
            priors = {(m, sd): fit_prior(panel, m, wc, teams, idx, season,
                                         with_opp, sd)
                      for m, (wc, _) in METRICS.items() for sd in sides}
        weeks = sorted(panel[(panel.season == season)
                             & (panel.season_type == "REG")].week.unique())
        for week in weeks:
            est = {}
            for m, (wc, _) in METRICS.items():
                parts = {}
                for sd in sides:
                    if two_stage:
                        parts[sd] = fit_week_v2(panel, m, wc, teams, idx, season,
                                                week, priors[(m, sd)], with_opp, sd)
                    else:
                        parts[sd] = fit_week_v1(panel, m, wc, teams, idx, season,
                                                week, with_opp, sd)
                if with_opp:
                    est[m] = parts["off"]
                else:
                    est[m] = (parts["off"][0], parts["def"][0])
            for i, team in enumerate(teams):
                r = {"season": season, "week": week, "team": team}
                for m in METRICS:
                    r[f"O_{m}"], r[f"D_{m}"] = est[m][0][i], est[m][1][i]
                rows.append(r)
    return pd.DataFrame(rows)


def main():
    panel = load_panel()
    print(f"panel {panel.shape}, seasons {panel.season.min()}-{panel.season.max()}")
    for est in ("v2adj", "v2raw", "v1adj"):
        q = build(panel, est)
        out = os.path.join(OUTDIR, f"team_quality_{est}.csv")
        q.to_csv(out, index=False)
        print(f"wrote {out}: {q.shape}", flush=True)


if __name__ == "__main__":
    main()

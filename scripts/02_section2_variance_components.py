#!/usr/bin/env python3
"""
Section 2 of EDA_PLAN.md: variance-component decomposition
    Y_isg = mu + a_i + b_is + c_{team x season} + eps
Estimators:
  1) REML via statsmodels MixedLM, single constant group, vc_formula per factor.
  2) Method-of-moments covariance matching, straight off EDA_PLAN 2.2:
       mean cross-product, same player same season (g != g'):  s2P + s2S + s2T
       mean cross-product, same player different seasons:      s2P
       mean cross-product, teammates same team-season:         s2T
       variance of Y (demeaned):                               s2P + s2S + s2T + s2G
     (y demeaned by grand mean; by season means in the season-FE spec.)

Specs: headline 2021-2025 with the section-0 exclusions (REG, targets>=2);
sensitivities (a) 2014-2025 with season fixed effects, (b) log(1+Y), (c) no exclusions.
rho_max per eq. (5) with G = 17.

Outputs: results/variance_components.csv, diagnostics printed for section2_notes.md.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Y = "fantasy_points_ppr"
G_SEASON = 17


def load(years, exclusions=True):
    df = pd.read_csv(ROOT / "data/players/wr_top30_weekly.csv")
    df = df[(df["season_type"] == "REG") & df["season"].between(*years)].copy()
    if exclusions:
        df = df[df["targets"] >= 2]
    df["pid"] = df["gsis_id"]
    df["ps"] = df["gsis_id"] + "_" + df["season"].astype(str)
    df["ts"] = df["team"] + "_" + df["season"].astype(str)
    df["pts"] = df["pid"] + "_" + df["ts"]  # player x team x season (trade-safe)
    return df.reset_index(drop=True)


def pair_sum(y, groups):
    """Sum over unordered within-group pairs of y_j*y_k, and pair count."""
    s = pd.Series(y).groupby(groups)
    g_sum, g_sq, g_n = s.sum(), s.apply(lambda v: (v**2).sum()), s.size()
    return float(((g_sum**2 - g_sq) / 2).sum()), float((g_n * (g_n - 1) / 2).sum())


def mom_fit(df, yvar, season_fe=False):
    y = df[yvar].values.astype(float)
    if season_fe:
        y = y - df.groupby("season")[yvar].transform("mean").values
    else:
        y = y - y.mean()
    var_y = float(np.mean(y**2))

    sp_ps, np_ps = pair_sum(y, df["ps"])          # same player, same season
    sp_p, np_p = pair_sum(y, df["pid"])           # same player, any season
    sp_ts, np_ts = pair_sum(y, df["ts"])          # same team-season, any player
    sp_pts, np_pts = pair_sum(y, df["pts"])       # same player, same team-season

    c_within_ps = sp_ps / np_ps                                  # s2P + s2S + s2T
    c_cross = (sp_p - sp_ps) / (np_p - np_ps)                    # s2P
    n_mate = np_ts - np_pts
    c_mate = (sp_ts - sp_pts) / n_mate if n_mate > 0 else np.nan  # s2T

    s2P = c_cross
    s2T = c_mate
    s2S = c_within_ps - s2P - s2T
    s2G = var_y - c_within_ps
    return dict(s2P=s2P, s2S=s2S, s2T=s2T, s2G=s2G, n_teammate_pairs=n_mate)


def reml_fit(df, yvar, season_fe=False):
    d = df.copy()
    d["y"] = d[yvar]
    d["const_grp"] = 1
    fixed = "y ~ C(season)" if season_fe else "y ~ 1"
    vc = {"player": "0 + C(pid)", "player_season": "0 + C(ps)", "team_season": "0 + C(ts)"}
    m = smf.mixedlm(fixed, d, groups="const_grp", vc_formula=vc)
    r = m.fit(reml=True, method=["lbfgs", "powell"], maxiter=2000)
    s2 = r.scale
    vcs = r.vcomp  # order follows sorted(vc keys): player, player_season, team_season
    keys = sorted(vc)
    out = dict(zip([f"vc_{k}" for k in keys], vcs))
    res = np.asarray(r.resid)
    return dict(
        s2P=out["vc_player"], s2S=out["vc_player_season"], s2T=out["vc_team_season"], s2G=s2,
        converged=bool(r.converged),
        resid_skew=float(stats.skew(res)), resid_kurt=float(stats.kurtosis(res)),
        resid_qq_r=float(np.corrcoef(np.sort(res), stats.norm.ppf(
            (np.arange(1, len(res) + 1) - 0.375) / (len(res) + 0.25)))[0, 1]),
    )


def summarize(comp):
    tot = comp["s2P"] + comp["s2S"] + comp["s2T"] + comp["s2G"]
    out = dict(comp)
    for k in ["s2P", "s2S", "s2T", "s2G"]:
        out[f"icc_{k[2:]}"] = comp[k] / tot
    out["rho_max"] = comp["s2P"] / (comp["s2P"] + comp["s2S"] + comp["s2T"] + comp["s2G"] / G_SEASON)
    return out


def main():
    specs = [
        ("headline_2021_2025_excl", dict(years=(2021, 2025), exclusions=True), Y, False),
        ("sens_a_2014_2025_seasonFE", dict(years=(2014, 2025), exclusions=True), Y, True),
        ("sens_b_log1p_2021_2025", dict(years=(2021, 2025), exclusions=True), "logy", False),
        ("sens_c_no_exclusions", dict(years=(2021, 2025), exclusions=False), Y, False),
    ]
    rows = []
    for name, loadkw, yvar, sfe in specs:
        df = load(**loadkw)
        if yvar == "logy":
            df["logy"] = np.log1p(df[Y].clip(lower=-0.99))
        meta = dict(spec=name, n_obs=len(df), n_players=df.pid.nunique(),
                    n_player_seasons=df.ps.nunique(), n_team_seasons=df.ts.nunique())
        mom = summarize(mom_fit(df, yvar, season_fe=sfe))
        rows.append({**meta, "estimator": "MoM_covmatch", **mom})
        print(f"[{name}] MoM: " + ", ".join(f"{k}={v:.4g}" for k, v in mom.items()))
        try:
            reml = summarize(reml_fit(df, yvar, season_fe=sfe))
            rows.append({**meta, "estimator": "REML_MixedLM", **reml})
            print(f"[{name}] REML: " + ", ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in reml.items()))
        except Exception as e:
            print(f"[{name}] REML FAILED: {e}")
            rows.append({**meta, "estimator": "REML_MixedLM", "converged": False, "error": str(e)})

    out = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    out.round(5).to_csv(ROOT / "results/variance_components.csv", index=False)
    print("\nwrote results/variance_components.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

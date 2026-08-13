#!/usr/bin/env python3
"""
Section 5 of EDA_PLAN.md: covariate structure.

(a) Age curve:  Y_isg = f(age) + delta_year + a_i + eps, f = natural cubic spline df 4
    (patsy cr), season fixed effects, player random intercepts (MixedLM, REML).
    Sample (pre-specified): ALL WR player-seasons 2014-2025 with >= 8 included games and
    >= 3 targets/game season average, under the S0 game-inclusion rule (REG, targets >= 2).
    Age from birth_date at season start (Sept 1). Interpretation restricted to SHAPE
    (APC identity: age = entry age + experience, year = entry year + experience, so any
    linear trend is age/era/experience-arbitrary once player intercepts are in).
    Secondary: spline x high/low-aDOT indicator (split at sample median of season aDOT).
    Sensitivity: top-30 universe only.

(b) Team-environment elasticity (FWL within player-season):
    log(1+Y_isg) = alpha_{player x season} + b1 log(team pass att) + b2 team pass EPA + eps
    demeaned within player-season, OLS, SEs clustered by team-week.
    Sample: all WR player-games passing the S0 rule, 2014-2025 (player-seasons need >= 2
    included games to contribute within variation). Sensitivity: 2021-2025.

(c) Archetype clustering: per player-season (>= 8 included games) features
    {aDOT, target_share, YAC/reception, TD/target, slot indicator (ngs_position == SLOT_WR)},
    z-scored; Gaussian mixture, k chosen by BIC over k = 2..6 (fixed seed, n_init = 10).
    Across clusters: one-way ANOVA on player-season mean PPG; Levene (ANOVA on
    |Y_isg - cluster median|) for variance differences. Descriptive mapping of the 30
    top-30 WRs' most recent qualifying season to clusters.

Outputs: results/age_curve.csv, results/elasticity.csv, results/archetypes.csv, console log.
"""
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sps
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YEARS = range(2014, 2026)
SEED = 20260714


# ---------------------------------------------------------------- data
def load_games() -> pd.DataFrame:
    frames = []
    for y in YEARS:
        df = pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{y}.csv",
                         low_memory=False,
                         usecols=["player_id", "player_display_name", "position",
                                  "season", "week", "season_type", "team",
                                  "targets", "receptions", "receiving_yards",
                                  "receiving_tds", "receiving_air_yards",
                                  "receiving_yards_after_catch",
                                  "fantasy_points_ppr"])
        df = df[(df["position"] == "WR") & (df["season_type"] == "REG")
                & (df["targets"] >= 2)]
        tm = pd.read_csv(ROOT / f"data/teams/stats_team_week_{y}.csv",
                         usecols=["season", "week", "team", "season_type",
                                  "attempts", "passing_epa"])
        tm = tm[tm["season_type"] == "REG"].rename(
            columns={"attempts": "team_pass_att", "passing_epa": "team_pass_epa"})
        df = df.merge(tm.drop(columns="season_type"),
                      on=["season", "week", "team"], how="left", validate="m:1")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    assert out["team_pass_att"].notna().all()
    return out


def season_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per player-season aggregates on included games."""
    g = df.groupby(["player_id", "season"])
    t = g.agg(n_games=("targets", "size"), tot_targets=("targets", "sum"),
              tot_rec=("receptions", "sum"), tot_yds=("receiving_yards", "sum"),
              tot_tds=("receiving_tds", "sum"), tot_ray=("receiving_air_yards", "sum"),
              tot_yac=("receiving_yards_after_catch", "sum"),
              tot_team_att=("team_pass_att", "sum"),
              ppg=("fantasy_points_ppr", "mean"),
              name=("player_display_name", "first")).reset_index()
    t["tpg"] = t["tot_targets"] / t["n_games"]
    t["adot"] = t["tot_ray"] / t["tot_targets"]
    t["target_share"] = t["tot_targets"] / t["tot_team_att"]
    t["yac_per_rec"] = np.where(t["tot_rec"] > 0, t["tot_yac"] / t["tot_rec"], np.nan)
    t["td_per_target"] = t["tot_tds"] / t["tot_targets"]
    return t


# ---------------------------------------------------------------- (a) age curve
def robust_fit(model):
    """MixedLM fit with optimizer fallbacks (line-search excursions can hit a
    singular GLS solve); reports which method converged."""
    for meth in ["lbfgs", "bfgs", "cg", "nm", "powell"]:
        try:
            res = model.fit(reml=True, method=[meth], maxiter=2000)
            if res.converged:
                print(f"    (converged: {meth})")
                return res
        except np.linalg.LinAlgError:
            continue
    raise RuntimeError("no optimizer converged")


def curve_from_fit(res, design_info, grid_ages, seasons, extra=None):
    """f-hat on an age grid, season effects averaged over the sample seasons.
    Returns (f, se) using the fixed-effects covariance block."""
    k = len(res.fe_params)
    cov = np.asarray(res.cov_params())[:k, :k]
    beta = np.asarray(res.fe_params)
    f, se = [], []
    for a in grid_ages:
        new = pd.DataFrame({"age": a, "season": seasons})
        if extra:
            for c, v in extra.items():
                new[c] = v
        X = np.asarray(patsy.build_design_matrices([design_info], new)[0])
        xbar = X.mean(axis=0)
        f.append(float(xbar @ beta))
        se.append(float(np.sqrt(xbar @ cov @ xbar)))
    return np.array(f), np.array(se)


def decline(grid, f, at):
    """Centered 1-year change f(a+.5) - f(a-.5) by linear interpolation on the grid."""
    return float(np.interp(at + 0.5, grid, f) - np.interp(at - 0.5, grid, f))


def fit_age_curve(games, seas_ok, meta, sample_tag, out_rows):
    d = games.merge(seas_ok[["player_id", "season", "adot"]],
                    on=["player_id", "season"], how="inner")
    d = d.merge(meta[["gsis_id", "birth_date"]], left_on="player_id",
                right_on="gsis_id", how="left")
    n_nobd = d["birth_date"].isna().groupby(d["player_id"]).first().sum()
    d = d[d["birth_date"].notna()].copy()
    d["age"] = (pd.to_datetime(d["season"].astype(str) + "-09-01")
                - pd.to_datetime(d["birth_date"])).dt.days / 365.25
    d["y"] = d["fantasy_points_ppr"]
    lo, hi = d["age"].quantile([0.01, 0.99])
    grid = np.arange(np.floor(lo * 10) / 10, hi + 1e-9, 0.1)
    seasons = sorted(d["season"].unique())
    print(f"[age:{sample_tag}] rows={len(d)} players={d['player_id'].nunique()} "
          f"player-seasons={len(d.groupby(['player_id','season']))} "
          f"age range used {lo:.1f}-{hi:.1f}; players missing birth_date: {n_nobd}")

    m = smf.mixedlm("y ~ cr(age, df=4) + C(season)", data=d, groups=d["player_id"])
    res = robust_fit(m)
    f, se = curve_from_fit(res, m.data.design_info, grid, seasons)
    for a, fh, s in zip(grid, f, se):
        out_rows.append(dict(sample=sample_tag, group="all", age=round(a, 1),
                             f_hat=fh, se=s, lo=fh - 1.96 * s, hi=fh + 1.96 * s))
    peak = grid[np.argmax(f)]
    print(f"  peak age {peak:.1f}; decline/yr at 28: {decline(grid, f, 28):+.2f}, "
          f"30: {decline(grid, f, 30):+.2f}, 32: {decline(grid, f, 32):+.2f} PPG")

    # secondary: spline x high/low aDOT (median split on season aDOT, this sample)
    med = seas_ok["adot"].median() if sample_tag == "primary" else \
        seas_ok.loc[seas_ok.set_index(["player_id", "season"]).index.isin(
            d.set_index(["player_id", "season"]).index), "adot"].median()
    d["hi_adot"] = (d["adot"] >= med).astype(int)
    m2 = smf.mixedlm("y ~ cr(age, df=4) * C(hi_adot) + C(season)", data=d,
                     groups=d["player_id"])
    res2 = robust_fit(m2)
    inter = [n for n in res2.fe_params.index if ":" in n and "hi_adot" in n]
    idx = [list(res2.fe_params.index).index(n) for n in inter]
    kfe = len(res2.fe_params)
    cov_fe = np.asarray(res2.cov_params())[:kfe, :kfe]
    bI = np.asarray(res2.fe_params)[idx]
    VI = cov_fe[np.ix_(idx, idx)]
    chi2 = float(bI @ np.linalg.solve(VI, bI))
    pw = float(sps.chi2.sf(chi2, df=len(inter)))
    print(f"  aDOT-median split at {med:.2f}; joint Wald on {len(inter)} spline x hi_adot "
          f"terms: chi2={chi2:.2f} (df {len(inter)}), p={pw:.4f}")
    for tag, val in [("low_adot", 0), ("high_adot", 1)]:
        fg, sg = curve_from_fit(res2, m2.data.design_info, grid, seasons,
                                extra={"hi_adot": val})
        for a, fh, s in zip(grid, fg, sg):
            out_rows.append(dict(sample=sample_tag, group=tag, age=round(a, 1),
                                 f_hat=fh, se=s, lo=fh - 1.96 * s, hi=fh + 1.96 * s))
        pk = grid[np.argmax(fg)]
        print(f"    {tag}: peak {pk:.1f}; decline/yr 28 {decline(grid, fg, 28):+.2f}, "
              f"30 {decline(grid, fg, 30):+.2f}, 32 {decline(grid, fg, 32):+.2f}")
    return res


# ---------------------------------------------------------------- (b) elasticity
def fit_elasticity(games, sample_tag):
    d = games.copy()
    d = d[d["team_pass_att"] > 0]
    d["y"] = np.log1p(d["fantasy_points_ppr"].clip(lower=0))
    n_neg = (d["fantasy_points_ppr"] < 0).sum()
    d["x1"] = np.log(d["team_pass_att"])
    d["x2"] = d["team_pass_epa"]
    ps = d.groupby(["player_id", "season"])
    d = d[ps["week"].transform("size") >= 2].copy()
    for c in ["y", "x1", "x2"]:
        d[c + "_dm"] = d[c] - d.groupby(["player_id", "season"])[c].transform("mean")
    d["cluster"] = (d["season"].astype(str) + "_" + d["week"].astype(str)
                    + "_" + d["team"])
    X = sm.add_constant(d[["x1_dm", "x2_dm"]])
    fit = sm.OLS(d["y_dm"], X).fit(cov_type="cluster",
                                   cov_kwds={"groups": d["cluster"]})
    ci = fit.conf_int()
    row = dict(sample=sample_tag, n_games=len(d),
               n_player_seasons=int(d.groupby(["player_id", "season"]).ngroups),
               n_clusters=int(d["cluster"].nunique()),
               beta1_logatt=fit.params["x1_dm"], se1=fit.bse["x1_dm"],
               b1_lo=ci.loc["x1_dm", 0], b1_hi=ci.loc["x1_dm", 1],
               beta2_passepa=fit.params["x2_dm"], se2=fit.bse["x2_dm"],
               b2_lo=ci.loc["x2_dm", 0], b2_hi=ci.loc["x2_dm", 1],
               n_negative_ppr_clipped=int(n_neg))
    print(f"[elast:{sample_tag}] n={row['n_games']} clusters={row['n_clusters']} "
          f"b1={row['beta1_logatt']:.3f} [{row['b1_lo']:.3f},{row['b1_hi']:.3f}] "
          f"b2={row['beta2_passepa']:.4f} [{row['b2_lo']:.4f},{row['b2_hi']:.4f}]")
    return row, fit, d


# ---------------------------------------------------------------- (c) archetypes
def fit_archetypes(games, seas, meta, top30):
    feats = ["adot", "target_share", "yac_per_rec", "td_per_target", "slot"]
    t = seas[seas["n_games"] >= 8].copy()
    t = t.merge(meta[["gsis_id", "ngs_position"]], left_on="player_id",
                right_on="gsis_id", how="left")
    t["slot"] = (t["ngs_position"] == "SLOT_WR").astype(int)
    n_unlabeled = t["ngs_position"].isna().sum()
    t = t.dropna(subset=["adot", "target_share", "yac_per_rec", "td_per_target"])
    print(f"[arch] player-seasons: {len(t)}; ngs_position missing (slot set 0): "
          f"{n_unlabeled} ({n_unlabeled/len(t):.1%})")

    Z = StandardScaler().fit_transform(t[feats].to_numpy(float))
    bics = {}
    for k in range(2, 7):
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             n_init=10, random_state=SEED).fit(Z)
        bics[k] = gm.bic(Z)
    k_star = min(bics, key=bics.get)
    print("  BIC by k:", {k: round(v, 1) for k, v in bics.items()}, "-> k =", k_star)
    gm = GaussianMixture(n_components=k_star, covariance_type="full",
                         n_init=10, random_state=SEED).fit(Z)
    t["cluster"] = gm.predict(Z)

    # cluster profiles
    prof = t.groupby("cluster")[feats + ["ppg", "n_games"]].mean()
    prof["n"] = t.groupby("cluster").size()
    print(prof.round(3).to_string())

    # ANOVA on player-season mean PPG
    groups = [g["ppg"].to_numpy() for _, g in t.groupby("cluster")]
    F, pF = sps.f_oneway(*groups)
    # Levene per plan: game-level ANOVA on |Y - cluster median of game PPG|
    gm_games = games.merge(t[["player_id", "season", "cluster"]],
                           on=["player_id", "season"], how="inner")
    med = gm_games.groupby("cluster")["fantasy_points_ppr"].median()
    gm_games["absdev"] = (gm_games["fantasy_points_ppr"]
                          - gm_games["cluster"].map(med)).abs()
    lev_groups = [g["absdev"].to_numpy() for _, g in gm_games.groupby("cluster")]
    L, pL = sps.f_oneway(*lev_groups)
    sd_by_c = gm_games.groupby("cluster")["fantasy_points_ppr"].std()
    print(f"  ANOVA mean PPG: F={F:.2f} p={pF:.2e}; "
          f"Levene (game level): W={L:.2f} p={pL:.2e}")
    print("  game-PPR SD by cluster:", sd_by_c.round(2).to_dict())

    # descriptive mapping: top-30 most recent qualifying season
    t30 = t[t["player_id"].isin(top30)]
    recent = t30.sort_values("season").groupby("player_id").tail(1)
    t["is_top30_recent"] = t.set_index(["player_id", "season"]).index.isin(
        recent.set_index(["player_id", "season"]).index)
    missing = set(top30) - set(t30["player_id"])
    if missing:
        print(f"  top-30 with NO qualifying season: {sorted(missing)}")
    return t, dict(k=k_star, bics=bics, F=F, pF=pF, L=L, pL=pL,
                   profiles=prof, recent=recent)


# ---------------------------------------------------------------- main
def main():
    np.random.seed(SEED)
    games = load_games()
    meta = pd.read_csv(ROOT / "data/meta/players_meta.csv", low_memory=False)
    top30 = pd.read_csv(ROOT / "data/meta/wr_top30_meta.csv")["gsis_id"].tolist()
    seas = season_table(games)

    # (a) age curve — primary sample: >=8 games and >=3 targets/game
    seas_ok = seas[(seas["n_games"] >= 8) & (seas["tpg"] >= 3)]
    rows = []
    fit_age_curve(games, seas_ok, meta, "primary", rows)
    g30 = games[games["player_id"].isin(top30)]
    fit_age_curve(g30, seas_ok[seas_ok["player_id"].isin(top30)], meta,
                  "top30_only", rows)
    pd.DataFrame(rows).round(4).to_csv(ROOT / "results/age_curve.csv", index=False)

    # (b) elasticity
    erows = []
    erows.append(fit_elasticity(games, "primary_2014_2025")[0])
    erows.append(fit_elasticity(games[games["season"] >= 2021], "window_2021_2025")[0])
    pd.DataFrame(erows).round(5).to_csv(ROOT / "results/elasticity.csv", index=False)

    # (c) archetypes
    t, info = fit_archetypes(games, seas, meta, top30)
    keep = ["player_id", "name", "season", "n_games", "adot", "target_share",
            "yac_per_rec", "td_per_target", "slot", "ppg", "cluster",
            "is_top30_recent"]
    t[keep].round(4).to_csv(ROOT / "results/archetypes.csv", index=False)
    print("\nTop-30 mapping (most recent qualifying season):")
    print(info["recent"].merge(t[["player_id", "season", "cluster"]],
                               on=["player_id", "season"])
          [["name", "season", "cluster_x", "adot", "target_share", "yac_per_rec",
            "td_per_target", "slot", "ppg"]]
          .rename(columns={"cluster_x": "cluster"})
          .sort_values(["cluster", "ppg"], ascending=[True, False])
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()

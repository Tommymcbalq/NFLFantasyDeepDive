"""EDA_PLAN2.md §D — usage-based projection data arm (LOSO arm (vi)) +
extended scorecard for arms (i), (ii), (v), (vi), (v+vi) and the 2026 board v2.

Arm (vi), per plan: within each LOSO training fold (data <= Y-1), ridge
regression of PPG_{s+1} on {target_share_s, WOPR_s, aDOT_s, team pass
attempts_ts (z), age} over all WR player-seasons with >= 3 targets/game
(features from season s, outcome season s+1, BOTH <= Y-1). lambda by CV inside
the fold (5-fold grouped by player, so the same player never straddles a CV
split). Data arm y_hat_i = ridge prediction from the board player's
season-(Y-1) stats; V = training-fold residual variance (in-sample; the
grouped-CV residual variance is reported alongside as the optimism check);
posterior per eq. (7) with the fold's m_hat / tau^2(tier):
    theta(vi) = (1 - B_r) y_hat + B_r m_hat,  B_r = V_r / (V_r + tau^2).
Board players with no qualifying season-(Y-1) stats fall back to arm (ii)'s
mu_hat posterior (documented, counted); zero-history players to m_hat (B = 1).

Arm (v+vi) — age-detrend applied to the usage arm (implementation choice stated
before fitting): the linear `age` ridge feature is REPLACED by the fold's
spline f_hat from §C (scripts/15, refit on <= Y-1 only): the ridge is fit on
the age-detrended outcome T = PPG_{s+1} - f_hat(age_{s+1}) with features
{ts, WOPR, aDOT, att_z}, and the prediction re-adds the spline at the player's
prediction-year age: y_hat^a = g_hat(x_{Y-1}) + f_hat(age_{i,Y}). Fallback for
no-usage players is arm (v)'s mu_hat^a posterior.

Definitions (identical to round-1 §5 conventions):
  target_share_s   = tot targets / tot team pass attempts over included games
  air-yards share  = tot rec air yards / tot team air yards (same games)
  WOPR_s           = 1.5 * target_share + 0.7 * air-yards share
  aDOT_s           = tot rec air yards / tot targets
  team pass att z  = team season attempts/game, z-scored across teams within s
  age              = age at Sept 1 of the PREDICTED season (s+1 / Y)
Inclusion: S0 (REG, targets >= 2); qualifying season = mean targets/game >= 3.

Scorecard (results/loso_scorecard2.csv): arms (i), (ii) reproduced identically
to round 1 (asserted), (v) from scripts/15 partial, (vi), (v+vi). Metrics:
pooled RMSE, mean within-year Spearman, DM vs (i) AND vs (ii) (paired
squared-error differences, clustered by year, t on 10 yearly means, 9 df).

2026 board v2 (results/valuation_2026_v2.csv): recomputed under the best arm
with DM-vs-(ii) support (pre-specified rule: t > 0 and p < 0.05 vs (ii); ties
broken by pooled RMSE). If no arm qualifies, the file restates the round-1
final values with a column saying so.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

ROOT = "/Users/thomasmcnamee/NFL"
YEARS = list(range(2015, 2025))
ALPHAS = np.logspace(-2, 4, 31)
FEATS = ["target_share", "wopr", "adot", "att_z", "age_pred"]
FEATS_A = ["target_share", "wopr", "adot", "att_z"]   # (v+vi): spline handles age

panel = pd.read_csv(f"{ROOT}/results/edge_panel.csv")
partC = pd.read_csv(f"{ROOT}/results/sectionC_partial.csv")
curves = pd.read_csv(f"{ROOT}/results/age_curve_folds.csv")
meta = pd.read_csv(f"{ROOT}/data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "birth_date"])

# ---------------- weekly player + team data, S0 rule ----------------
pl_frames, tm_frames = [], []
for y in range(2014, 2026):
    df = pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
                     usecols=["player_id", "position", "season", "week",
                              "season_type", "team", "targets",
                              "receiving_air_yards", "fantasy_points_ppr"],
                     low_memory=False)
    pl_frames.append(df[(df.season_type == "REG") & (df.targets > 1)])
    tm = pd.read_csv(f"{ROOT}/data/teams/stats_team_week_{y}.csv",
                     usecols=["season", "week", "team", "season_type",
                              "attempts", "passing_air_yards"])
    tm_frames.append(tm[tm.season_type == "REG"].drop(columns="season_type"))
wk = pd.concat(pl_frames, ignore_index=True)
tmw = pd.concat(tm_frames, ignore_index=True).rename(
    columns={"attempts": "tm_att", "passing_air_yards": "tm_ay"})
wk = wk.merge(tmw, on=["season", "week", "team"], how="left", validate="m:1")
assert wk.tm_att.notna().all()

# team season pass attempts/game -> z within season
tse = (tmw.groupby(["season", "team"]).tm_att.mean().rename("tm_att_pg")
       .reset_index())
tse["att_z"] = tse.groupby("season").tm_att_pg.transform(
    lambda x: (x - x.mean()) / x.std(ddof=0))

# WR season table (usage features + ppg)
w = wk[wk.position == "WR"]
seas = (w.groupby(["player_id", "season"])
        .agg(n_games=("targets", "size"), tot_tgt=("targets", "sum"),
             tot_ray=("receiving_air_yards", "sum"),
             tot_tm_att=("tm_att", "sum"), tot_tm_ay=("tm_ay", "sum"),
             ppg=("fantasy_points_ppr", "mean"),
             team_mode=("team", lambda x: x.mode().iat[0])).reset_index())
seas["mean_tgt"] = seas.tot_tgt / seas.n_games
seas["target_share"] = seas.tot_tgt / seas.tot_tm_att
seas["ays"] = seas.tot_ray / seas.tot_tm_ay
seas["wopr"] = 1.5 * seas.target_share + 0.7 * seas.ays
seas["adot"] = seas.tot_ray / seas.tot_tgt
seas = seas.merge(tse[["season", "team", "att_z"]],
                  left_on=["season", "team_mode"], right_on=["season", "team"],
                  how="left").drop(columns="team")
seas = seas.merge(meta.rename(columns={"gsis_id": "player_id"}),
                  on="player_id", how="left")
seas["age_s"] = (pd.to_datetime(seas.season.astype(str) + "-09-01")
                 - pd.to_datetime(seas.birth_date)).dt.days / 365.25

qual = seas[seas.mean_tgt >= 3.0].copy()
nxt = seas[["player_id", "season", "ppg"]].rename(
    columns={"season": "season_next", "ppg": "ppg_next"})
qual["season_next"] = qual.season + 1
pairs = qual.merge(nxt, on=["player_id", "season_next"], how="inner")
pairs["age_pred"] = pairs.age_s + 1.0            # age at the predicted season
n_noage = int(pairs.age_pred.isna().sum())
pairs = pairs.dropna(subset=FEATS + ["ppg_next"]).reset_index(drop=True)
print(f"training pairs (all years): {len(pairs)} "
      f"({pairs.player_id.nunique()} players); dropped for missing "
      f"birth_date: {n_noage}")
att_rate = len(pairs) / (qual.season < 2025).sum()
print(f"attrition note: {att_rate:.1%} of qualifying seasons <=2024 have a "
      f"season s+1 with included games (the YoY sample conditions on survival)")

# sigma^2 sample (identical to scripts/10) for the fallback arm reproduction
# -> not needed: fallback values come straight from sectionC_partial (asserted
#    identical to round-1 loso_predictions in scripts/15).


def f_fold(Y):
    c = curves[curves.fold == Y]
    grid, fh = c.age.values, c.f_hat.values
    lo, hi = c.lo_clamp.iat[0], c.hi_clamp.iat[0]
    return lambda a: np.interp(np.clip(a, lo, hi), grid, fh)


def fit_ridge(X, y, groups):
    mu, sd = X.mean(0), X.std(0, ddof=0)
    Z = (X - mu) / sd
    cv = list(GroupKFold(5).split(Z, y, groups))
    r = RidgeCV(alphas=ALPHAS, cv=cv).fit(Z, y)
    resid = y - r.predict(Z)
    # grouped-CV residual variance (optimism check)
    cv_res = np.empty_like(y)
    for tr_i, te_i in cv:
        rr = RidgeCV(alphas=[r.alpha_]).fit(Z[tr_i], y[tr_i])
        cv_res[te_i] = y[te_i] - rr.predict(Z[te_i])
    return r, mu, sd, float(resid.var(ddof=1)), float(cv_res.var(ddof=1))


# ---------------- LOSO loop ----------------
preds, coef_rows, fold_rows = [], [], []
for Y in YEARS:
    tr_panel = panel[(panel.year != Y) & panel.in_fit].copy()
    ev = partC[partC.year == Y].copy()          # carries m_hat, tau2, arm ii/v

    # ridge training data: s and s+1 both <= Y-1
    d = pairs[pairs.season_next <= Y - 1]
    fY = f_fold(Y)

    if len(d) < 50:
        # DATA EDGE (fold 2015): weekly data starts 2014, so no YoY pair has
        # both seasons <= 2014. Usage arm undefined -> whole fold falls back
        # to arm (ii) [(v) for the detrended variant]. Documented.
        ev["yhat_usage"] = np.nan
        ev["yhat_usage_a"] = np.nan
        ev["theta_usage"] = ev.theta_star
        ev["theta_usage_a"] = ev.theta_star_a
        ev["usage_fallback"] = True
        fold_rows.append(dict(Y=Y, n_eval=len(ev), n_fallback=len(ev),
                              n_train=len(d), alpha=np.nan, V_train=np.nan,
                              V_cv=np.nan, B_vet=np.nan))
        preds.append(ev)
        continue

    # arm (vi): raw target, linear age feature
    X = d[FEATS].values
    r6, mu6, sd6, V6, V6cv = fit_ridge(X, d.ppg_next.values,
                                       d.player_id.values)
    # arm (v+vi): age-detrended target, no age feature, spline re-added
    T = d.ppg_next.values - fY(d.age_pred.values)
    Xa = d[FEATS_A].values
    r56, mu56, sd56, V56, V56cv = fit_ridge(Xa, T, d.player_id.values)

    coef_rows.append(dict(fold=Y, n_train=len(d), alpha=r6.alpha_,
                          **dict(zip(FEATS, r6.coef_)), V_train=V6, V_cv=V6cv,
                          alpha_a=r56.alpha_,
                          **{k + "_a": v for k, v in zip(FEATS_A, r56.coef_)},
                          V_train_a=V56, V_cv_a=V56cv))

    # board players' season-(Y-1) usage
    bx = qual[qual.season == Y - 1][["player_id"] + FEATS_A + ["age_s"]]
    ev = ev.merge(bx.rename(columns={"player_id": "gsis_id"}),
                  on="gsis_id", how="left")
    has_u = ev.target_share.notna() & ev.age.notna()
    Xe = ev.loc[has_u, FEATS_A].values
    age_e = ev.loc[has_u, "age"].values          # age at year Y (panel column)
    yhat6 = r6.predict((np.column_stack([Xe, age_e]) - mu6) / sd6)
    yhat56 = r56.predict((Xe - mu56) / sd56) + fY(age_e)
    ev["yhat_usage"] = np.nan
    ev.loc[has_u, "yhat_usage"] = yhat6
    ev["yhat_usage_a"] = np.nan
    ev.loc[has_u, "yhat_usage_a"] = yhat56

    B6 = V6 / (V6 + ev.tau2)
    B56 = V56 / (V56 + ev.tau2)
    # posterior; fallback: arm (ii) theta* for (vi), arm (v) theta*_a for (v+vi)
    ev["theta_usage"] = np.where(has_u, (1 - B6) * ev.yhat_usage.fillna(0)
                                 + B6 * ev.m_hat, ev.theta_star)
    ev["theta_usage_a"] = np.where(has_u, (1 - B56) * ev.yhat_usage_a.fillna(0)
                                   + B56 * ev.m_hat, ev.theta_star_a)
    ev["usage_fallback"] = ~has_u
    tau2_vet = (ev.loc[ev.tier == "vet", "tau2"].iloc[0]
                if (ev.tier == "vet").any() else np.nan)
    fold_rows.append(dict(Y=Y, n_eval=len(ev), n_fallback=int((~has_u).sum()),
                          n_train=len(d), alpha=r6.alpha_,
                          V_train=round(V6, 2), V_cv=round(V6cv, 2),
                          B_vet=round(float(V6 / (V6 + tau2_vet)), 3)))
    preds.append(ev)

pred = pd.concat(preds, ignore_index=True)
coefs = pd.DataFrame(coef_rows)
print("\n=== ridge coefficients by fold (standardized features, arm vi) ===")
print(coefs[["fold", "n_train", "alpha"] + FEATS + ["V_train", "V_cv"]]
      .round(3).to_string(index=False))
print("\n=== arm (v+vi) coefficients (detrended target) ===")
print(coefs[["fold", "alpha_a"] + [f + "_a" for f in FEATS_A]
            + ["V_train_a", "V_cv_a"]].round(3).to_string(index=False))
print("\n=== fold diagnostics ===")
print(pd.DataFrame(fold_rows).to_string(index=False))
coefs.round(4).to_csv(f"{ROOT}/results/usage_ridge_coefs.csv", index=False)

# ---------------- extended scorecard ----------------
ARMS = {"(i) ADP-only m_hat": "m_hat",
        "(ii) blind theta*": "theta_star",
        "(v) age-detrended theta*_a": "theta_star_a",
        "(vi) usage posterior": "theta_usage",
        "(v+vi) usage + age-detrend": "theta_usage_a"}

# reproduction guard (arms i, ii identical to round 1)
old = pd.read_csv(f"{ROOT}/results/loso_predictions.csv")
chk = pred.merge(old[["year", "gsis_id", "theta_star", "m_hat"]],
                 on=["year", "gsis_id"], suffixes=("", "_r1"))
assert np.allclose(chk.theta_star, chk.theta_star_r1)
assert np.allclose(chk.m_hat, chk.m_hat_r1)


def dm(col, base):
    d = (pred.ppg - pred[base]) ** 2 - (pred.ppg - pred[col]) ** 2  # >0 better
    dyr = d.groupby(pred.year).mean()
    t = float(dyr.mean() / (dyr.std(ddof=1) / np.sqrt(len(dyr))))
    return t, float(2 * stats.t.sf(abs(t), df=len(dyr) - 1))


rows = []
for label, col in ARMS.items():
    err = pred.ppg - pred[col]
    rmse = float(np.sqrt((err ** 2).mean()))
    rho = pred.groupby("year").apply(
        lambda g: stats.spearmanr(g[col], g.ppg).statistic, include_groups=False)
    ti, pi = dm(col, "m_hat") if col != "m_hat" else (np.nan, np.nan)
    tii, pii = dm(col, "theta_star") if col != "theta_star" else (np.nan, np.nan)
    rows.append(dict(predictor=label, rmse=rmse, mean_spearman=float(rho.mean()),
                     dm_t_vs_adp=ti, dm_p_vs_adp=pi,
                     dm_t_vs_blind=tii, dm_p_vs_blind=pii))
    print(f"\n{label}: RMSE {rmse:.4f}, mean Spearman {rho.mean():.4f}"
          + (f"; DM vs (i): t={ti:+.2f} p={pi:.3f}" if not np.isnan(ti) else "")
          + (f"; DM vs (ii): t={tii:+.2f} p={pii:.3f}" if not np.isnan(tii) else ""))

sc = pd.DataFrame(rows)
assert abs(sc.rmse[0] - 3.564) < 5e-4 and abs(sc.rmse[1] - 3.463) < 5e-4
sc.to_csv(f"{ROOT}/results/loso_scorecard2.csv", index=False)
keep = ["year", "name", "gsis_id", "adp", "tier", "age", "rookie_season",
        "games", "ppg", "m_hat", "mu_hat", "n_eff", "B", "theta_star",
        "mu_hat_a", "theta_star_a", "yhat_usage", "theta_usage",
        "yhat_usage_a", "theta_usage_a", "usage_fallback"]
pred[keep].to_csv(f"{ROOT}/results/loso_predictions2.csv", index=False)

# per-fold RMSE by arm (anomaly chasing)
pf = pred.groupby("year").apply(
    lambda g: pd.Series({c: np.sqrt(((g.ppg - g[c]) ** 2).mean())
                         for c in ARMS.values()}), include_groups=False)
print("\n=== per-fold RMSE ===")
print(pf.round(3).to_string())

# ---------------- 2026 board v2 ----------------
support = sc[(sc.dm_t_vs_blind > 0) & (sc.dm_p_vs_blind < 0.05)]
if len(support):
    best = support.sort_values("rmse").iloc[0]
    print(f"\nbest DM-vs-(ii)-supported arm: {best.predictor}")
else:
    best = None
    print("\nNO arm improves on (ii) at the pre-registered DM threshold "
          "(t>0, p<.05 vs blind theta*) — 2026 board v2 restates round-1 values")

final_r1 = pd.read_csv(f"{ROOT}/results/valuation_2026_final.csv")
c26 = pd.read_csv(f"{ROOT}/results/sectionC_2026.csv")

if best is None:
    out = final_r1.copy()
    out["V_final_v2"] = out.V_final
    out["round2_verdict"] = ("unchanged: neither (v) nor (vi) beat the blind "
                             "posterior in LOSO (DM vs (ii) n.s.)")
elif "(v)" == best.predictor.split()[0]:
    out = final_r1.merge(c26[["player", "mu_hat_a", "theta_star_a"]], on="player")
    out["V_final_v2"] = out.theta_star_a
    out["round2_verdict"] = "arm (v) age-detrended posterior"
else:
    # usage arm on the 2026 board: ridge on all pairs (<=2025), 2025 usage
    knots = pd.read_csv(f"{ROOT}/results/market_prior_iso_knots.csv")
    tau2f = pd.read_csv(f"{ROOT}/results/tier_variances.csv").set_index("tier").tau2_iso
    d = pairs  # all data
    use_a = "(v+vi)" in best.predictor
    c_full = None
    if use_a:
        # full-sample f_hat comes from scripts/15's full fit via sectionC_2026
        raise SystemExit("implement (v+vi) 2026 branch if selected")
    r6, mu6, sd6, V6, _ = fit_ridge(d[FEATS].values, d.ppg_next.values,
                                    d.player_id.values)
    b = c26.merge(qual[qual.season == 2025][["player_id"] + FEATS_A],
                  left_on="gsis_id", right_on="player_id", how="left")
    hu = b.target_share.notna()
    Xe = np.column_stack([b.loc[hu, FEATS_A].values, b.loc[hu, "age_2026"].values])
    b["yhat"] = np.nan
    b.loc[hu, "yhat"] = r6.predict((Xe - mu6) / sd6)
    tau2_b = b.tier.map(tau2f)
    B6 = V6 / (V6 + tau2_b)
    b["V_final_v2"] = np.where(hu, (1 - B6) * b.yhat.fillna(0) + B6 * b.m_adp,
                               b.theta_star)   # fallback arm (ii)
    out = final_r1.merge(b[["player", "yhat", "V_final_v2"]], on="player")
    out["round2_verdict"] = f"arm {best.predictor}"

out = out.sort_values("V_final_v2", ascending=False).reset_index(drop=True)
out["rank_v2"] = out.index + 1
out.to_csv(f"{ROOT}/results/valuation_2026_v2.csv", index=False)
print(f"wrote results/valuation_2026_v2.csv "
      f"({'restated round-1' if best is None else best.predictor})")

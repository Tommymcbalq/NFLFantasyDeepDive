"""EDA_PLAN2.md §C — age-detrended data arm (LOSO arm (v)).

mu_hat^a_i = recency-weighted (h=1) mean of age-adjusted season means
    Ybar^a_is = Ybar_is - [ f_hat(age_is) - f_hat(age_i at prediction year) ]
theta*^a rebuilt with the IDENTICAL B weights as round-1 arm (ii) (same V, tau^2,
n_eff, m_hat) — only the likelihood mean changes.

CRITICAL leakage rule (pre-registered): inside each LOSO fold Y, f_hat is refit
on data <= Y-1 ONLY. Estimator choice, stated up front: the round-1 §5 spec is
MixedLM (cr(age, df=4) + season FE + player random intercept). Refit 10x on
game-level data MixedLM is slow/fragile (round 1 needed optimizer fallbacks),
so per the plan's explicit allowance we use the computationally lighter
FIXED-EFFECTS version: OLS of Y_isg on cr(age, df=4) + C(season) with player
fixed effects absorbed by within-player demeaning (FWL). Only the SHAPE of f is
identified either way (APC collinearity, round-1 §5/§7a), and the detrend uses
only differences f(a1) - f(a2), which are shape-only. The full-data FE curve is
compared against the round-1 MixedLM curve (results/age_curve.csv) as a check.

Fold-specific fit sample = round-1 §5(a) primary sample restricted to seasons
<= Y-1: all WR player-seasons with >= 8 included games (S0 rule: REG,
targets >= 2) and >= 3 targets/game, birth date present. Fold Y = 2015 has a
single training season (2014): within-player demeaning leaves no age variation,
so that fold falls back to pooled OLS (cross-sectional identification) —
documented, and flagged in the fold table.

Ages are clamped to the fold's [1st, 99th] pct training age range before
evaluating f_hat (no spline extrapolation). Players with missing birth date get
zero adjustment (mu^a = mu), counted.

Also (a) reproduces round-1 arms (i)/(ii) and ASSERTS RMSE 3.564 / 3.463 before
proceeding; (b) reports the mu_hat bias by career-start cohort before/after
detrending (target: the verified +1.1 PPG bias, pre-2014-career players,
2017-24 folds); (c) fits the full-sample (2014-2025) f_hat and produces the
2026-board age-detrended values for downstream use.

Outputs:
  results/sectionC_partial.csv   per-row fold predictions incl. mu_hat_a, theta_star_a
  results/age_curve_folds.csv    per-fold f_hat on the age grid (for script 16)
  results/sectionC_2026.csv      2026 board mu_hat_a / theta_star_a (full-sample f_hat)
Scorecard integration (arm (v) + DM tests) happens in scripts/16.
"""
import numpy as np
import pandas as pd
from patsy import dmatrix, build_design_matrices
from scipy import stats
from sklearn.isotonic import IsotonicRegression

ROOT = "/Users/thomasmcnamee/NFL"
YEARS = list(range(2015, 2025))
GRID = np.round(np.arange(21.0, 36.0 + 1e-9, 0.1), 1)

panel = pd.read_csv(f"{ROOT}/results/edge_panel.csv")

# ---------------- weekly data (2014-2025), S0 rule ----------------
wk_frames = []
for y in range(2014, 2026):
    df = pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
                     usecols=["player_id", "position", "season", "season_type",
                              "targets", "fantasy_points_ppr"], low_memory=False)
    wk_frames.append(df[(df.season_type == "REG") & (df.targets > 1)])
wk = pd.concat(wk_frames, ignore_index=True)

meta = pd.read_csv(f"{ROOT}/data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "rookie_season", "birth_date"])

# season means for panel players (mu_hat inputs)
sm_all = (wk[wk.player_id.isin(panel.gsis_id.unique())]
          .groupby(["player_id", "season"])
          .agg(G=("fantasy_points_ppr", "size"),
               ybar=("fantasy_points_ppr", "mean")).reset_index())
sm_all = sm_all.merge(meta[["gsis_id", "birth_date"]].rename(
    columns={"gsis_id": "player_id"}), on="player_id", how="left")
sm_all["age"] = (pd.to_datetime(sm_all.season.astype(str) + "-09-01")
                 - pd.to_datetime(sm_all.birth_date)).dt.days / 365.25

# §3 primary sample for sigma^2(tier) — identical to scripts/10
wr = wk[wk.position == "WR"].copy()
ps = (wr.groupby(["player_id", "season"])
        .agg(mean_tgt=("targets", "mean"), mu_ps=("fantasy_points_ppr", "mean"),
             n_games=("targets", "size"))
        .reset_index())
wr = wr.merge(ps[ps.mean_tgt >= 3.0][["player_id", "season", "mu_ps"]],
              on=["player_id", "season"], how="inner")
wr = wr.merge(meta.rename(columns={"gsis_id": "player_id"}), on="player_id",
              how="left").dropna(subset=["rookie_season"])
wr["e2"] = (wr.fantasy_points_ppr - wr.mu_ps) ** 2
wr["exp"] = wr.season - wr.rookie_season
wr["tier"] = np.select([wr.exp == 0, wr.exp == 1], ["rookie", "soph"], "vet")

# §5(a) age-curve fit sample: all WR games in qualifying player-seasons
q = ps[(ps.n_games >= 8) & (ps.mean_tgt >= 3.0)][["player_id", "season"]]
ag = wk[wk.position == "WR"].merge(q, on=["player_id", "season"], how="inner")
ag = ag.merge(meta[["gsis_id", "birth_date"]].rename(
    columns={"gsis_id": "player_id"}), on="player_id", how="left")
n_nobd_fit = ag.birth_date.isna().groupby(ag.player_id).first().sum()
ag = ag.dropna(subset=["birth_date"]).copy()
ag["age"] = (pd.to_datetime(ag.season.astype(str) + "-09-01")
             - pd.to_datetime(ag.birth_date)).dt.days / 365.25
ag["y"] = ag.fantasy_points_ppr
print(f"age-curve fit sample: {len(ag)} games, {ag.player_id.nunique()} players, "
      f"{ag.groupby(['player_id','season']).ngroups} player-seasons; "
      f"players dropped for missing birth_date: {int(n_nobd_fit)}")


def fit_age_fe(upto):
    """FE (within-player) OLS spline fit on seasons <= upto. Returns
    (f_on_grid, lo, hi, mode): f evaluated on GRID (shape-only, arbitrary level),
    clamp range [lo, hi] = 1st/99th pct of training ages."""
    d = ag[ag.season <= upto].reset_index(drop=True)
    lo, hi = d.age.quantile([0.01, 0.99])
    n_seas = d.season.nunique()
    X = dmatrix("cr(age, df=4) + C(season)", d, return_type="dataframe")
    di = X.design_info
    cr_cols = [c for c in X.columns if c.startswith("cr(")]
    if n_seas >= 2:
        Xv = X.drop(columns="Intercept")
        key = d.player_id.values
        Xd = Xv - Xv.groupby(key).transform("mean")
        yd = (d.y - d.groupby("player_id").y.transform("mean")).values
        beta, *_ = np.linalg.lstsq(Xd.values, yd, rcond=None)
        bmap = dict(zip(Xv.columns, beta))
        mode = "player-FE within OLS"
    else:
        beta, *_ = np.linalg.lstsq(X.values, d.y.values, rcond=None)
        bmap = dict(zip(X.columns, beta))
        mode = "pooled OLS (1 training season — no within-player age variation)"
    b_cr = np.array([bmap[c] for c in cr_cols])
    new = pd.DataFrame({"age": np.clip(GRID, lo, hi), "season": d.season.min()})
    Xn = pd.DataFrame(np.asarray(build_design_matrices([di], new)[0]),
                      columns=di.column_names)
    f_grid = Xn[cr_cols].values @ b_cr
    return f_grid, float(lo), float(hi), mode


def f_eval(f_grid, lo, hi, ages):
    return np.interp(np.clip(ages, lo, hi), GRID, f_grid)


def decline(f_grid, at):
    return float(np.interp(at + .5, GRID, f_grid) - np.interp(at - .5, GRID, f_grid))


def mu_neff_before(gsis, Y, f_grid=None, lo=None, hi=None, age_pred=None):
    """h=1 recency-weighted mu_hat and n_eff from seasons < Y.
    If f_grid given, age-detrend each season mean to age_pred first."""
    h = sm_all[(sm_all.player_id == gsis) & (sm_all.season < Y)]
    if len(h) == 0:
        return np.nan, 0.0
    yb = h.ybar.values.copy()
    if f_grid is not None and age_pred is not None and h.age.notna().all():
        adj = f_eval(f_grid, lo, hi, h.age.values) - f_eval(f_grid, lo, hi, age_pred)
        yb = yb - adj
    S = h.season.max()
    w = 2.0 ** (-(S - h.season.values))
    return float((w * yb).sum() / w.sum()), float(w.sum() ** 2 / (w ** 2).sum())


# ---------------- sanity: full-data FE curve vs round-1 MixedLM curve ----------
f_full, lo_full, hi_full, mode_full = fit_age_fe(2025)
r1 = pd.read_csv(f"{ROOT}/results/age_curve.csv")
r1 = r1[(r1["sample"] == "primary") & (r1.group == "all")]
f_r1 = np.interp(GRID, r1.age.values, r1.f_hat.values)
# compare shapes: center both at age 26
c_fe = f_full - np.interp(26, GRID, f_full)
c_r1 = f_r1 - np.interp(26, GRID, f_r1)
msk = (GRID >= max(lo_full, r1.age.min())) & (GRID <= min(hi_full, r1.age.max()))
print(f"\nfull-sample FE curve ({mode_full}): peak {GRID[np.argmax(f_full)]:.1f}; "
      f"decline/yr 28 {decline(f_full,28):+.2f}, 30 {decline(f_full,30):+.2f}, "
      f"32 {decline(f_full,32):+.2f}")
print(f"round-1 MixedLM curve:            peak {r1.age.values[np.argmax(r1.f_hat.values)]:.1f}; "
      f"shape max |FE - MixedLM| on common support (centered at 26): "
      f"{np.max(np.abs(c_fe[msk] - c_r1[msk])):.2f} PPG")

# ---------------- LOSO loop ----------------
preds, fold_rows, curve_rows = [], [], []
for Y in YEARS:
    tr = panel[(panel.year != Y) & panel.in_fit].copy()
    ev = panel[(panel.year == Y) & panel.in_fit].copy()

    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(np.log(tr.adp.values), tr.ppg.values)
    tr["r"] = tr.ppg - iso.predict(np.log(tr.adp.values))
    ev["m_hat"] = iso.predict(np.log(ev.adp.values))
    tau2 = tr.groupby("tier").r.var(ddof=1)
    ev["tau2"] = ev.tier.map(tau2).fillna(tr.r.var(ddof=1))
    sig2 = wr[wr.season != Y].groupby("tier").e2.mean()
    ev["sig2"] = ev.tier.map(sig2)

    # fold f_hat on data <= Y-1 ONLY
    f_g, lo, hi, mode = fit_age_fe(Y - 1)
    for a, fv in zip(GRID, f_g):
        curve_rows.append(dict(fold=Y, age=a, f_hat=fv, lo_clamp=lo, hi_clamp=hi))

    mu, ne, mua = [], [], []
    for gsis, age_pred in zip(ev.gsis_id, ev.age):
        m0, n0 = mu_neff_before(gsis, Y)
        ma, _ = mu_neff_before(gsis, Y, f_g, lo, hi, age_pred)
        mu.append(m0); ne.append(n0); mua.append(ma)
    ev["mu_hat"], ev["n_eff"], ev["mu_hat_a"] = mu, ne, mua
    no_prior = ev.n_eff == 0

    with np.errstate(divide="ignore"):
        V = ev.sig2 / ev.n_eff
    B = np.where(no_prior, 1.0, V / (V + ev.tau2))
    ev["B"] = B
    ev["theta_star"] = np.where(no_prior, ev.m_hat,
                                (1 - B) * ev.mu_hat.fillna(0) + B * ev.m_hat)
    ev["theta_star_a"] = np.where(no_prior, ev.m_hat,
                                  (1 - B) * ev.mu_hat_a.fillna(0) + B * ev.m_hat)
    fold_rows.append(dict(Y=Y, n_eval=len(ev), fit_mode=mode, age_lo=round(lo, 1),
                          age_hi=round(hi, 1),
                          dec28=round(decline(f_g, 28), 2),
                          dec30=round(decline(f_g, 30), 2),
                          dec32=round(decline(f_g, 32), 2),
                          n_noprior=int(no_prior.sum())))
    preds.append(ev[["year", "name", "gsis_id", "adp", "tier", "age",
                     "rookie_season", "games", "ppg", "m_hat", "tau2", "sig2",
                     "mu_hat", "n_eff", "B", "theta_star", "mu_hat_a",
                     "theta_star_a"]])

pred = pd.concat(preds, ignore_index=True)
print("\n=== fold f_hat diagnostics ===")
print(pd.DataFrame(fold_rows).to_string(index=False))

# ---------------- reproduction assert (round-1 arms i, ii) ----------------
rmse_i = float(np.sqrt(((pred.ppg - pred.m_hat) ** 2).mean()))
rmse_ii = float(np.sqrt(((pred.ppg - pred.theta_star) ** 2).mean()))
print(f"\nreproduction: RMSE (i) {rmse_i:.4f} (target 3.564), "
      f"(ii) {rmse_ii:.4f} (target 3.463)")
old = pd.read_csv(f"{ROOT}/results/loso_predictions.csv")
chk = pred.merge(old[["year", "gsis_id", "theta_star"]], on=["year", "gsis_id"],
                 suffixes=("", "_r1"))
assert abs(rmse_i - 3.564) < 5e-4 and abs(rmse_ii - 3.463) < 5e-4, "repro FAILED"
assert np.allclose(chk.theta_star, chk.theta_star_r1), "theta* rows differ from round 1"
print("reproduction PASSED (row-level theta* identical to round 1)")

# ---------------- arm (v) quick metrics ----------------
rmse_v = float(np.sqrt(((pred.ppg - pred.theta_star_a) ** 2).mean()))
print(f"arm (v) theta*_a RMSE: {rmse_v:.4f}")

# ---------------- cohort bias tables ----------------
pred["cohort"] = np.where(pred.rookie_season < 2014, "career_start<2014",
                          "career_start>=2014")
pred["exp_bucket"] = pd.cut(pred.year - pred.rookie_season, [-.1, 2, 5, 8, 30],
                            labels=["exp0-2", "exp3-5", "exp6-8", "exp9+"])
has_mu = pred.n_eff > 0


def bias_tab(by):
    g = pred[has_mu].groupby(by, observed=True)
    return pd.DataFrame({
        "n": g.size(),
        "bias_mu": g.apply(lambda d: (d.mu_hat - d.ppg).mean(), include_groups=False),
        "bias_mu_a": g.apply(lambda d: (d.mu_hat_a - d.ppg).mean(), include_groups=False),
        "bias_theta": g.apply(lambda d: (d.theta_star - d.ppg).mean(), include_groups=False),
        "bias_theta_a": g.apply(lambda d: (d.theta_star_a - d.ppg).mean(), include_groups=False),
    }).round(3)


print("\n=== bias by career-start cohort (all folds, rows with prior data) ===")
print(bias_tab("cohort").to_string())
old_1724 = pred[has_mu & (pred.year >= 2017) & (pred.rookie_season < 2014)]
print(f"\ntarget anomaly — pre-2014 careers, 2017-24 folds (n={len(old_1724)}): "
      f"mu bias {(old_1724.mu_hat - old_1724.ppg).mean():+.2f} -> "
      f"mu_a bias {(old_1724.mu_hat_a - old_1724.ppg).mean():+.2f}; "
      f"theta* bias {(old_1724.theta_star - old_1724.ppg).mean():+.2f} -> "
      f"theta*_a bias {(old_1724.theta_star_a - old_1724.ppg).mean():+.2f}")
print("\n=== bias by experience bucket ===")
print(bias_tab("exp_bucket").to_string())
print("\n=== bias by cohort x fold-era ===")
pred["era"] = np.where(pred.year <= 2016, "2015-16", "2017-24")
print(bias_tab(["era", "cohort"]).to_string())

# largest individual moves (theta*_a - theta*)
pred["d_theta"] = pred.theta_star_a - pred.theta_star
mv = pred[has_mu].reindex(pred[has_mu].d_theta.abs().sort_values(ascending=False).index)
print("\n=== 12 largest |theta*_a - theta*| moves ===")
print(mv[["year", "name", "age", "rookie_season", "mu_hat", "mu_hat_a", "B",
          "theta_star", "theta_star_a", "d_theta", "ppg"]].head(12)
      .round(2).to_string(index=False))

# ---------------- 2026 board (full-sample f_hat, all data <= 2025) ------------
blind = pd.read_csv(f"{ROOT}/results/valuation_2026_blind.csv")
w30 = pd.read_csv(f"{ROOT}/data/meta/wr_top30_meta.csv")[["gsis_id", "birth_date"]]
blind = blind.merge(pd.read_csv(f"{ROOT}/results/consistency_table.csv")
                    [["gsis_id", "player"]], on="player", how="left")
blind = blind.merge(w30, on="gsis_id", how="left", validate="1:1")
assert blind.birth_date.notna().all()
blind["age_2026"] = (pd.Timestamp("2026-09-01")
                     - pd.to_datetime(blind.birth_date)).dt.days / 365.25
mua26 = []
for gsis, a26 in zip(blind.gsis_id, blind.age_2026):
    ma, _ = mu_neff_before(gsis, 2026, f_full, lo_full, hi_full, a26)
    mua26.append(ma)
blind["mu_hat_a"] = mua26
blind["theta_star_a"] = (1 - blind.B) * blind.mu_hat_a + blind.B * blind.m_adp
out26 = blind[["player", "gsis_id", "adp", "adp_rank", "tier", "age_2026",
               "mu_hat", "mu_hat_a", "n_eff", "m_adp", "B", "theta_star",
               "theta_star_a", "post_SD"]]
out26.to_csv(f"{ROOT}/results/sectionC_2026.csv", index=False)
print("\n2026 board, age-detrended (largest moves):")
t = out26.assign(d=out26.theta_star_a - out26.theta_star)
print(t.reindex(t.d.abs().sort_values(ascending=False).index)
      [["player", "age_2026", "mu_hat", "mu_hat_a", "theta_star", "theta_star_a", "d"]]
      .head(10).round(2).to_string(index=False))

pred.drop(columns=["d_theta"]).to_csv(f"{ROOT}/results/sectionC_partial.csv",
                                      index=False)
pd.DataFrame(curve_rows).round(4).to_csv(f"{ROOT}/results/age_curve_folds.csv",
                                         index=False)
print("\nwrote results/sectionC_partial.csv, age_curve_folds.csv, sectionC_2026.csv")

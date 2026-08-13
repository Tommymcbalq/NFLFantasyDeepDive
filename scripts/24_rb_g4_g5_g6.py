"""§G4 thin-data board / §G5 RB availability + SV rule / §G6 RB LOSO (EDA_PLAN4.md).

Everything RB, everything re-estimated.  No WR number enters any fit.

PRE-REGISTERED (fixed before any RB fold was scored):

  §G5  availability.  G_is = REG games with touches >= 2 (the §G1 participation
       definition); M = 16 (season < 2021) / 17 (>= 2021).  Relevance gate =
       the §G2c gate, touches/game >= 4.  ICC-analogue rho = sigma^2_p /
       (pbar(1-pbar)) with sigma^2_p the beta-binomial method-of-moments
       between-player variance (observed Var(p_hat) net of binomial noise);
       H0 "pure binomial + age" by parametric bootstrap, 1000 sims.  YoY
       correlation of p_hat with a player bootstrap (4000 reps).
       Pre-registered expectation: NOTHING.  The WR result is not assumed to
       transfer, and a null here is a publishable result.

  §G6  LOSO over the 2015-2024 RB boards.  For each held-out year Y everything
       is refit on the other nine: m_{-Y}(ADP) isotonic decreasing in log ADP;
       tau^2_{-Y}(tier) from training isotonic residuals; sigma^2_{-Y}(tier)
       from all gated RB games in seasons != Y.  mu_hat/n_eff are rebuilt from
       weekly_raw seasons STRICTLY BEFORE Y (h = 1).  weekly_raw now reaches
       1999, so unlike the WR run there is no left-truncation at 2014; with
       h = 1 the extra depth is second-order (weight 2^-k) but it is used
       because it exists.  Zero prior included games => n_eff = 0 => B = 1.

       Arms on the PPG target:      (i) m_{-Y}(ADP)      (ii) blind posterior theta*
       Arms on the points-per-scheduled-week target, exactly the round-2 §A
       construction:                (i-SV) ADP-only isotonic REFIT on ppsw
                                    (iii) SV = theta* x E[G]/M
       (iii) cannot be DM'd against (i) because it predicts a different target,
       so it is scored against its own refit ADP-only baseline (i-SV), which is
       what round 2 did.  E[G]/M from a binomial GLM G/M ~ age + G1 + G2 +
       miss1 + miss2 fit on training-fold rows only.

       ADOPTION RULE (fixed now): an arm is adopted over its market-only
       baseline iff DM t-test on yearly mean squared-error differentials,
       t(9 df), gives p < 0.10 AND pooled RMSE improves.  HONESTY CLAUSE
       (EDA_PLAN4 §G): if the data arm does not beat market-only, that is the
       result; the RB board is market-anchored and we do not go hunting for a
       different arm.

  §G4  2026 RB board.  Posterior eq. (7) with RB inputs throughout.  The two
       2026 rookies have zero NFL rows => n_eff = 0 => B = 1 => theta* =
       m_iso(ADP), i.e. the pure market arm, and are FLAGGED.  The six
       one-season players get n_eff = 1 and are flagged as thin.

Outputs: results/availability_table_rb.csv, results/loso_predictions_rb.csv,
         results/loso_scorecard_rb.csv, results/valuation_rb_2026.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.isotonic import IsotonicRegression

RNG = np.random.default_rng(20260809)
ROOT = Path("/Users/thomasmcnamee/NFL")
YEARS = list(range(2015, 2025))
GATE = 4.0

# ------------------------------------------------------------- weekly data
COLS = ["player_id", "player_display_name", "position", "season", "week",
        "season_type", "targets", "carries", "fantasy_points_ppr"]
frames = []
for y in range(1999, 2026):
    d = pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{y}.csv",
                    usecols=lambda c: c in COLS or c == "gsis_id", low_memory=False)
    if "gsis_id" in d.columns and "player_id" not in d.columns:
        d = d.rename(columns={"gsis_id": "player_id"})
    frames.append(d[d.season_type == "REG"])
wk = pd.concat(frames, ignore_index=True)
wk["touches"] = wk.carries.fillna(0) + wk.targets.fillna(0)
inc = wk[wk.touches >= 2]
print(f"weekly REG rows 1999-2025: {len(wk)}; included (touches>=2): {len(inc)}")

meta = pd.read_csv(ROOT / "data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "birth_date", "rookie_season"])

# ============================================================ §G5 availability
print("\n" + "=" * 72 + "\n§G5  RB availability\n" + "=" * 72)
rbinc = inc[inc.position == "RB"]
ps = (rbinc.groupby(["player_id", "season"])
      .agg(G=("touches", "size"), mtch=("touches", "mean"),
           name=("player_display_name", "first")).reset_index())
ps["M"] = np.where(ps.season >= 2021, 17, 16)
ps["p_hat"] = ps.G / ps.M
ps = ps.merge(meta.rename(columns={"gsis_id": "player_id"}), on="player_id", how="left")
ps["age"] = ps.season - pd.to_datetime(ps.birth_date, errors="coerce").dt.year
rel = ps[(ps.mtch >= GATE) & ps.season.between(2014, 2025)].copy()
print(f"RB player-seasons 2014-25 with >=1 included game: "
      f"{len(ps[ps.season.between(2014,2025)])}; gated (touches/g >= {GATE:g}): "
      f"{len(rel)} over {rel.player_id.nunique()} players")
print(f"p_hat: mean {rel.p_hat.mean():.3f}, sd {rel.p_hat.std():.3f}, "
      f"p10/p50/p90 {rel.p_hat.quantile([.1,.5,.9]).round(3).tolist()}")
rel.to_csv(ROOT / "results/availability_table_rb.csv", index=False)

d = rel[["player_id", "season", "p_hat"]]
d1 = d.copy(); d1["season"] += 1
pairs = d.merge(d1, on=["player_id", "season"], suffixes=("_cur", "_prev"))
r_obs = np.corrcoef(pairs.p_hat_prev, pairs.p_hat_cur)[0, 1]
uniq = pairs.player_id.unique()
grp = {p: pairs.loc[pairs.player_id == p, ["p_hat_prev", "p_hat_cur"]].values
       for p in uniq}
bs = []
for _ in range(4000):
    arr = np.vstack([grp[p] for p in RNG.choice(uniq, len(uniq), replace=True)])
    bs.append(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"(a) YoY corr of p_hat: r = {r_obs:.3f} on {len(pairs)} pairs "
      f"({len(uniq)} players); player-bootstrap 95% CI [{lo:.3f}, {hi:.3f}]  "
      f"(WR: .422 [.363,.477])")


def mom_sigma2(df):
    p = df.p_hat.values; M = df.M.values
    pbar = df.G.sum() / M.sum()
    v = p.var(ddof=1); c = (1.0 / M).mean()
    s2 = (v - pbar * (1 - pbar) * c) / (1 - c)
    return pbar, v, s2, s2 / (pbar * (1 - pbar))


pbar, v_obs, s2_obs, icc_obs = mom_sigma2(rel)
print(f"(b) MoM: pbar {pbar:.3f}, Var(p_hat) {v_obs:.4f}, sigma^2_p {s2_obs:.4f} "
      f"(SD {np.sqrt(max(s2_obs,0)):.3f}), ICC-analogue rho = {icc_obs:.3f}  "
      f"(WR: SD .295, rho .364)")
nd = rel.dropna(subset=["age"]).copy()
X0 = sm.add_constant(nd[["age"]].assign(age2=nd.age ** 2))
g0 = sm.GLM(np.c_[nd.G, nd.M - nd.G], X0, family=sm.families.Binomial()).fit()
p0 = g0.predict(X0)
sims = np.array([mom_sigma2(nd.assign(G=(gg := RNG.binomial(nd.M.values, p0.values)),
                                      p_hat=gg / nd.M.values))[2]
                 for _ in range(1000)])
print(f"    H0 binomial+age: null sigma^2_p mean {sims.mean():.4f}, 95th "
      f"{np.percentile(sims,95):.4f}; observed {s2_obs:.4f} -> "
      f"p = {(sims >= s2_obs).mean():.4f}")

gl = rel.dropna(subset=["age", "rookie_season"]).copy()
allG = ps[["player_id", "season", "G"]]
for k in (1, 2):
    lag = allG.copy(); lag["season"] += k
    gl = gl.merge(lag.rename(columns={"G": f"G_lag{k}"}),
                  on=["player_id", "season"], how="left")
gl = gl[(gl.season >= 2016) & (gl.season >= gl.rookie_season + 2)].copy()
gl[["G_lag1", "G_lag2"]] = gl[["G_lag1", "G_lag2"]].fillna(0)
rw = gl.loc[gl.index.repeat(gl.M)].copy()
rw["y"] = (rw.groupby(level=0).cumcount() < rw.G).astype(int)
lg = sm.Logit(rw.y, sm.add_constant(rw[["age", "G_lag1", "G_lag2"]])).fit(
    disp=0, cov_type="cluster", cov_kwds={"groups": rw.player_id.values})
print(f"(c) game-level logistic, {len(gl)} player-seasons, "
      f"{gl.player_id.nunique()} clusters")
print(pd.DataFrame({"coef": lg.params, "se": lg.bse, "z": lg.tvalues,
                    "p": lg.pvalues}).round(4).to_string())

# ============================================================ §G6 LOSO
print("\n" + "=" * 72 + "\n§G6  RB LOSO 2015-2024\n" + "=" * 72)
panel = pd.read_csv(ROOT / "results/market_prior_rb.csv")
panel = panel.merge(meta.rename(columns={"gsis_id": "gsis_id"}), on="gsis_id",
                    how="left", suffixes=("", "_m"))
panel["age"] = ((pd.to_datetime(panel.year.astype(str) + "-09-01")
                 - pd.to_datetime(panel.birth_date, errors="coerce")).dt.days / 365.25)
print(f"panel {len(panel)} rows, missing age {panel.age.isna().sum()}")
panel["age"] = panel.age.fillna(panel.age.median())

tot = (wk.groupby(["player_id", "season"]).fantasy_points_ppr.sum()
       .rename("total_pts").reset_index())
panel = panel.merge(tot.rename(columns={"player_id": "gsis_id", "season": "year"}),
                    on=["gsis_id", "year"], how="left")
panel["total_pts"] = panel.total_pts.fillna(0.0)
panel["M"] = np.where(panel.year >= 2021, 17, 16)
panel["ppsw"] = panel.total_pts / panel.M

gcnt = inc.groupby(["player_id", "season"]).size().rename("G").reset_index()
gmap = {(p, s): g for p, s, g in gcnt.itertuples(index=False)}
panel["G_cur"] = [gmap.get((g, y), 0) for g, y in zip(panel.gsis_id, panel.year)]
for k in (1, 2):
    panel[f"G{k}"] = [gmap.get((g, y - k), np.nan)
                      for g, y in zip(panel.gsis_id, panel.year)]
    panel[f"miss{k}"] = panel[f"G{k}"].isna().astype(int)
    panel[f"G{k}"] = panel[f"G{k}"].fillna(0)
AV = ["age", "G1", "G2", "miss1", "miss2"]

# season means for mu_hat, and the gated all-RB sample for sigma^2(tier)
sm_all = (inc[inc.player_id.isin(panel.gsis_id.unique())]
          .groupby(["player_id", "season"])
          .agg(ybar=("fantasy_points_ppr", "mean")).reset_index())
sm_idx = {p: g[["season", "ybar"]].values for p, g in sm_all.groupby("player_id")}
rbs = rbinc.merge(ps[ps.mtch >= GATE][["player_id", "season"]],
                  on=["player_id", "season"])
mu_ps = rbs.groupby(["player_id", "season"]).fantasy_points_ppr.transform("mean")
rbs = rbs.assign(e2=(rbs.fantasy_points_ppr - mu_ps) ** 2)
rbs = rbs.merge(meta.rename(columns={"gsis_id": "player_id"}),
                on="player_id", how="left").dropna(subset=["rookie_season"])
rbs["exp"] = rbs.season - rbs.rookie_season
rbs["tier"] = np.select([rbs.exp == 0, rbs.exp == 1], ["rookie", "soph"], "vet")


def mu_neff_before(g, Y):
    a = sm_idx.get(g)
    if a is None:
        return np.nan, 0.0
    a = a[a[:, 0] < Y]
    if len(a) == 0:
        return np.nan, 0.0
    w = 2.0 ** (-(a[:, 0].max() - a[:, 0]) / 1.0)
    return float((w * a[:, 1]).sum() / w.sum()), float(w.sum() ** 2 / (w ** 2).sum())


preds, notes = [], []
for Y in YEARS:
    tr = panel[(panel.year != Y) & panel.in_fit].copy()
    ev = panel[panel.year == Y].copy()
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(np.log(tr.adp.values), tr.ppg.values)
    tr["r"] = tr.ppg - iso.predict(np.log(tr.adp.values))
    ev["m_hat"] = iso.predict(np.log(ev.adp.values))
    tau2 = tr.groupby("tier").r.var(ddof=1)
    ev["tau2"] = ev.tier.map(tau2).fillna(tr.r.var(ddof=1))
    sig2 = rbs[rbs.season != Y].groupby("tier").e2.mean()
    ev["sig2"] = ev.tier.map(sig2)
    mn = [mu_neff_before(g, Y) for g in ev.gsis_id]
    ev["mu_hat"] = [t[0] for t in mn]; ev["n_eff"] = [t[1] for t in mn]
    nop = ev.n_eff == 0
    with np.errstate(divide="ignore"):
        V = ev.sig2 / ev.n_eff
    B = np.where(nop, 1.0, V / (V + ev.tau2))
    ev["B"] = B
    ev["theta_star"] = np.where(nop, ev.m_hat,
                                (1 - B) * ev.mu_hat.fillna(0) + B * ev.m_hat)
    trA = panel[panel.year != Y]
    Xa = sm.add_constant(trA[AV], has_constant="add")
    gA = sm.GLM(np.c_[trA.G_cur, trA.M - trA.G_cur], Xa,
                family=sm.families.Binomial()).fit()
    ev["p_avail"] = gA.predict(sm.add_constant(ev[AV], has_constant="add")[Xa.columns])
    isoB = IsotonicRegression(increasing=False, out_of_bounds="clip")
    isoB.fit(np.log(trA.adp.values), trA.ppsw.values)
    ev["m_hat_ppsw"] = isoB.predict(np.log(ev.adp.values))
    ev["SV"] = ev.theta_star * ev.p_avail
    notes.append(dict(Y=Y, n_eval=int(ev.in_fit.sum()), n_noprior=int(nop.sum()),
                      n_rookie=int((ev.tier == "rookie").sum()),
                      tau2_rookie=tau2.get("rookie", np.nan),
                      tau2_soph=tau2.get("soph", np.nan),
                      tau2_vet=tau2.get("vet", np.nan),
                      sig2_vet=sig2.get("vet", np.nan), mean_B=float(np.mean(B))))
    preds.append(ev)

pred = pd.concat(preds, ignore_index=True)
print("\n=== fold diagnostics ===")
print(pd.DataFrame(notes).round(3).to_string(index=False))
print(f"\ntotal rows {len(pred)}, in_fit {pred.in_fit.sum()}, "
      f"B==1 (no prior data) {(pred.B == 1).sum()}")


def dm(df, target, base, cand):
    dsq = (df[target] - df[base]) ** 2 - (df[target] - df[cand]) ** 2
    dy = dsq.groupby(df.year).mean()
    t = float(dy.mean() / (dy.std(ddof=1) / np.sqrt(len(dy))))
    return t, float(2 * stats.t.sf(abs(t), df=len(dy) - 1)), dy


rows = []
fit = pred[pred.in_fit]
print("\n--- target: realized PPG (in_fit rows) ---")
for nm, col in [("(i) ADP-only m_hat", "m_hat"), ("(ii) blind theta*", "theta_star")]:
    err = fit.ppg - fit[col]
    rmse = float(np.sqrt((err ** 2).mean()))
    rho = fit.groupby("year").apply(
        lambda g: stats.spearmanr(g[col], g.ppg).statistic, include_groups=False)
    t = p = np.nan
    if col != "m_hat":
        t, p, dy = dm(fit, "ppg", "m_hat", col)
        print(f"  yearly mean loss diff (>0 = arm better): {dy.round(3).to_dict()}")
        print(f"  folds improved: {(dy > 0).sum()}/10")
    rows.append(dict(target="ppg", predictor=nm, rmse=rmse,
                     mean_spearman=float(rho.mean()), dm_t_vs_market=t,
                     dm_p_vs_market=p, n=len(fit)))
    print(f"  {nm}: RMSE {rmse:.4f}, Spearman {rho.mean():.4f}"
          + ("" if np.isnan(t) else f", DM t {t:+.3f}, p {p:.4f}"))

print("\n--- target: points per scheduled week (in_fit rows) ---")
for nm, col in [("(i-SV) ADP-only refit on ppsw", "m_hat_ppsw"),
                ("(iii) SV = theta* x E[G]/M", "SV")]:
    err = fit.ppsw - fit[col]
    rmse = float(np.sqrt((err ** 2).mean()))
    rho = fit.groupby("year").apply(
        lambda g: stats.spearmanr(g[col], g.ppsw).statistic, include_groups=False)
    t = p = np.nan
    if col != "m_hat_ppsw":
        t, p, dy = dm(fit, "ppsw", "m_hat_ppsw", col)
        print(f"  yearly mean loss diff: {dy.round(3).to_dict()}")
        print(f"  folds improved: {(dy > 0).sum()}/10")
    rows.append(dict(target="ppsw", predictor=nm, rmse=rmse,
                     mean_spearman=float(rho.mean()), dm_t_vs_market=t,
                     dm_p_vs_market=p, n=len(fit)))
    print(f"  {nm}: RMSE {rmse:.4f}, Spearman {rho.mean():.4f}"
          + ("" if np.isnan(t) else f", DM t {t:+.3f}, p {p:.4f}"))

# sensitivity: all 300 rows (includes wipe-out seasons) on ppsw
al = pred.copy()
t_all, p_all, _ = dm(al, "ppsw", "m_hat_ppsw", "SV")
print(f"\nsensitivity, all {len(al)} rows, ppsw: (iii) DM t {t_all:+.3f}, p {p_all:.4f}")
rows.append(dict(target="ppsw_all300", predictor="(iii) SV",
                 rmse=float(np.sqrt(((al.ppsw - al.SV) ** 2).mean())),
                 mean_spearman=np.nan, dm_t_vs_market=t_all, dm_p_vs_market=p_all,
                 n=len(al)))
# decomposition of any SV effect: fold-constant availability
pred["SV_const"] = pred.theta_star * pred.groupby("year").p_avail.transform("mean")
fit = pred[pred.in_fit]
tc, pc, _ = dm(fit, "ppsw", "m_hat_ppsw", "SV_const")
print(f"decomposition: SV_const (fold-mean availability) DM t {tc:+.3f}, p {pc:.4f}")
d2 = (fit.ppsw - fit.SV_const) ** 2 - (fit.ppsw - fit.SV) ** 2
dy2 = d2.groupby(fit.year).mean()
t2 = float(dy2.mean() / (dy2.std(ddof=1) / np.sqrt(len(dy2))))
print(f"            SV vs SV_const head-to-head t {t2:+.3f}, "
      f"p {2*stats.t.sf(abs(t2), 9):.4f}")

sc = pd.DataFrame(rows)
sc.to_csv(ROOT / "results/loso_scorecard_rb.csv", index=False)
pred.drop(columns=["birth_date"]).to_csv(ROOT / "results/loso_predictions_rb.csv",
                                         index=False)

# adoption verdicts
ii = sc[(sc.target == "ppg") & sc.predictor.str.startswith("(ii)")].iloc[0]
i0 = sc[(sc.target == "ppg") & sc.predictor.str.startswith("(i)")].iloc[0]
iii = sc[(sc.target == "ppsw") & sc.predictor.str.startswith("(iii)")].iloc[0]
isv = sc[(sc.target == "ppsw") & sc.predictor.str.startswith("(i-SV)")].iloc[0]
adopt_ii = bool((ii.dm_p_vs_market < 0.10) and (ii.rmse < i0.rmse))
adopt_iii = bool((iii.dm_p_vs_market < 0.10) and (iii.rmse < isv.rmse))
print(f"\nADOPTION: arm (ii) blind theta* -> {'ADOPTED' if adopt_ii else 'NOT ADOPTED'} "
      f"(p {ii.dm_p_vs_market:.4f}, RMSE {ii.rmse:.4f} vs {i0.rmse:.4f})")
print(f"ADOPTION: arm (iii) SV        -> {'ADOPTED' if adopt_iii else 'NOT ADOPTED'} "
      f"(p {iii.dm_p_vs_market:.4f}, RMSE {iii.rmse:.4f} vs {isv.rmse:.4f})")

# ============================================================ §G4 board
print("\n" + "=" * 72 + "\n§G4  2026 RB board\n" + "=" * 72)
brd = pd.read_csv(ROOT / "data/meta/rb_top30_meta.csv")
ct = pd.read_csv(ROOT / "results/consistency_table_rb.csv")[
    ["gsis_id", "player", "mu_hat", "n_eff", "n_games", "n_seasons"]]
sig2t = pd.read_csv(ROOT / "results/sigma2_by_tier_rb.csv").set_index("tier").sigma2
tau2t = pd.read_csv(ROOT / "results/tier_variances_rb.csv").set_index("tier").tau2_iso
knots = pd.read_csv(ROOT / "results/market_prior_iso_knots_rb.csv")

b = brd[["rb_adp_rank", "name", "team", "adp", "gsis_id", "rookie_season"]].merge(
    ct, on="gsis_id", how="left")
b["exp_2026"] = 2026 - b.rookie_season
b["tier"] = np.select([b.exp_2026 == 0, b.exp_2026 == 1], ["rookie", "soph"], "vet")
b.loc[b.rookie_season.isna(), "tier"] = "rookie"
b["n_eff"] = b.n_eff.fillna(0.0)
b["n_seasons"] = b.n_seasons.fillna(0).astype(int)
b["player"] = b.player.fillna(b.name)
print("tier counts:", b.tier.value_counts().to_dict())
print("zero-NFL-row players (pure market arm):",
      b.loc[b.n_eff == 0, "name"].tolist())
print("single-season players (thin):", b.loc[b.n_seasons == 1, "name"].tolist())

b["m_adp"] = np.interp(np.log(b.adp), knots.log_adp, knots.m)
b["tau2"] = b.tier.map(tau2t)
b["sigma2_tier"] = b.tier.map(sig2t)
with np.errstate(divide="ignore"):
    b["V"] = b.sigma2_tier / b.n_eff
nop = b.n_eff == 0
b["B"] = np.where(nop, 1.0, b.V / (b.V + b.tau2))
b["theta_star"] = np.where(nop, b.m_adp, (1 - b.B) * b.mu_hat.fillna(0) + b.B * b.m_adp)
b["post_SD"] = np.where(nop, np.sqrt(b.tau2), np.sqrt(1.0 / (1.0 / b.V + 1.0 / b.tau2)))
b["thin_data_flag"] = np.where(nop, "no NFL rows: full shrinkage to market",
                               np.where(b.n_seasons == 1, "single season: n_eff = 1", ""))

# availability scaling column (reported regardless; adopted only if §G6 says so)
gA = sm.GLM(np.c_[panel.G_cur, panel.M - panel.G_cur],
            sm.add_constant(panel[AV], has_constant="add"),
            family=sm.families.Binomial()).fit()
bd = pd.to_datetime(brd.set_index("gsis_id").birth_date, errors="coerce")
b["age"] = ((pd.Timestamp("2026-09-01") - b.gsis_id.map(bd)).dt.days / 365.25)
b["age"] = b.age.fillna(panel.age.median())
for k in (1, 2):
    b[f"G{k}"] = [gmap.get((g, 2026 - k), np.nan) for g in b.gsis_id]
    b[f"miss{k}"] = b[f"G{k}"].isna().astype(int)
    b[f"G{k}"] = b[f"G{k}"].fillna(0)
b["p_avail"] = gA.predict(sm.add_constant(b[AV], has_constant="add")[
    sm.add_constant(panel[AV], has_constant="add").columns])
b["SV"] = b.theta_star * b.p_avail
b["arm_iii_adopted"] = adopt_iii
b["arm_ii_adopted"] = adopt_ii
b["board_value"] = b.theta_star if adopt_ii else b.m_adp

b = b.sort_values("board_value", ascending=False).reset_index(drop=True)
b["rank_model"] = b.index + 1
b["delta_rank_vs_adp"] = b.rb_adp_rank - b.rank_model
cols = ["rank_model", "player", "team", "adp", "rb_adp_rank", "tier", "n_seasons",
        "mu_hat", "n_eff", "V", "m_adp", "tau2", "B", "theta_star", "post_SD",
        "p_avail", "SV", "board_value", "delta_rank_vs_adp", "thin_data_flag",
        "arm_ii_adopted", "arm_iii_adopted"]
b[cols].to_csv(ROOT / "results/valuation_rb_2026.csv", index=False)
print()
print(b[cols].round(3).to_string(index=False))
print("\nwrote availability_table_rb.csv, loso_predictions_rb.csv, "
      "loso_scorecard_rb.csv, valuation_rb_2026.csv")

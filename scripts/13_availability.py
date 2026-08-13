"""S A (EDA_PLAN2.md) - Availability as a modeled outcome. Round 2.

Part 0  Data audit: snap_counts / injuries downloads (row counts, coverage,
        join keys). Injuries join on gsis_id directly; snap counts join on
        pfr_player_id -> players_meta.pfr_id -> gsis_id.
Part 1  Player-season availability table, all WRs 2014-2025 with the round-1
        fantasy-relevance filter (season mean targets >= 3 over included
        games, i.e. REG rows with targets > 1 as in S3):
          G_is = REG games with targets >= 2; M_is = 16 (<2021) / 17 (>=2021);
          p_hat = G/M. Snap sensitivity: G_snap = REG games with
          offense_pct >= 0.25 (where the player joins to snap data).
Part 2  Is injury-proneness a stable trait?
        (a) YoY corr of p_hat within player, player-bootstrap CI (4000 reps);
        (b) beta-binomial method of moments (round-1 S1.4 style):
            Var(p_hat) across player-seasons minus mean binomial noise
            -> sigma2_p (between-player variance of p_i), ICC-analogue
            rho = sigma2_p / (pbar(1-pbar)); test vs H0 "pure binomial +
            age" by parametric bootstrap (simulate G ~ Bin(M, p(age)) from a
            null age-only logistic, recompute sigma2_p, 1000 sims);
        (c) game-level logistic: participation ~ age + G_{s-1} + G_{s-2},
            SEs clustered by player (season-level covariates, Bernoulli rows).
Part 3  LOSO arm (iv). For each held-out year Y in 2015-2024:
        - availability model (binomial GLM: G/M ~ age + G1 + G2 + miss1 +
          miss2) fit on the panel training years != Y -> p_avail for year-Y
          board players; E[G]/M = p_avail.
          NOTE the plan writes E[G]/17; since the scoring target is
          points per SCHEDULED week (total points / M) and M = 16 for
          2015-2020 boards, we use E[G]/M = p_avail (identical for 17-game
          seasons; using a hard 17 would misalign the pre-2021 folds).
        - SV = theta*_{-Y} x p_avail, theta* taken from the frozen round-1
          fold-honest results/loso_predictions.csv (291 in_fit rows). For the
          9 non-in_fit rows (< 4 realized games) we REPLICATE the script-10
          fold machinery to obtain theta* (fits unchanged - still trained on
          in_fit training rows only) and verify the replication reproduces
          loso_predictions.csv exactly before using it. Sensitivity only.
        - baseline: ADP-only isotonic (decreasing in log ADP) RE-FIT on the
          points-per-scheduled-week target within each training fold.
        - score on realized total REG PPR points / M; DM clustered by year,
          t(9 df). Primary eval = the 291 round-1 in_fit rows; sensitivity =
          all 300 panel rows (includes the injury wipe-out seasons).
Outputs: results/availability_table.csv, results/loso_availability.csv,
         results/sectionA_notes.md (written by hand from this log).
Round-1 loso files are NOT touched.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.isotonic import IsotonicRegression

RNG = np.random.default_rng(20260715)
ROOT = "/Users/thomasmcnamee/NFL"
YEARS = list(range(2014, 2026))
PANEL_YEARS = list(range(2015, 2025))

# ---------------------------------------------------------------- Part 0
print("=" * 72, "\nPART 0 - data audit\n", "=" * 72, sep="")
snap_frames, inj_frames = [], []
for y in YEARS:
    s = pd.read_csv(f"{ROOT}/data/snap_counts/snap_counts_{y}.csv",
                    low_memory=False)
    j = pd.read_csv(f"{ROOT}/data/injuries/injuries_{y}.csv", low_memory=False)
    snap_frames.append(s)
    inj_frames.append(j)
    print(f"{y}: snap_counts {len(s):>6} rows (REG {(s.game_type=='REG').sum():>6}, "
          f"weeks {s[s.game_type=='REG'].week.min()}-{s[s.game_type=='REG'].week.max()}) | "
          f"injuries {len(j):>6} rows (REG {(j.game_type=='REG').sum():>6})")
snap = pd.concat(snap_frames, ignore_index=True)
inj = pd.concat(inj_frames, ignore_index=True)
snap = snap[snap.game_type == "REG"]
print(f"total REG snap rows {len(snap)}, injury rows {len(inj)}")

meta = pd.read_csv(f"{ROOT}/data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "pfr_id", "birth_date", "rookie_season",
                            "position"])

# ---------------------------------------------------------------- Part 1
print("\n" + "=" * 72, "\nPART 1 - availability table\n", "=" * 72, sep="")
wk_frames = []
for y in YEARS:
    df = pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
                     usecols=["player_id", "player_display_name", "position",
                              "season", "week", "season_type", "targets",
                              "fantasy_points_ppr"], low_memory=False)
    wk_frames.append(df[df.season_type == "REG"])
wk = pd.concat(wk_frames, ignore_index=True)
wr = wk[wk.position == "WR"].copy()

# G_is: games with targets >= 2; relevance: mean targets >= 3 over those games
inc = wr[wr.targets > 1]
ps = (inc.groupby(["player_id", "season"])
      .agg(G=("targets", "size"), mean_tgt=("targets", "mean"),
           name=("player_display_name", "first")).reset_index())
ps["M"] = np.where(ps.season >= 2021, 17, 16)
ps["p_hat"] = ps.G / ps.M
ps = ps.merge(meta.rename(columns={"gsis_id": "player_id"}),
              on="player_id", how="left")
ps["age"] = ps.season - pd.to_datetime(ps.birth_date).dt.year  # season-year age
rel = ps[ps.mean_tgt >= 3.0].copy()
print(f"WR player-seasons (>=1 included game): {len(ps)}; "
      f"fantasy-relevant (mean tgt >= 3): {len(rel)}; "
      f"players: {rel.player_id.nunique()}")
print(f"p_hat: mean {rel.p_hat.mean():.3f}, sd {rel.p_hat.std():.3f}, "
      f"p10/p50/p90 {rel.p_hat.quantile([.1,.5,.9]).round(3).tolist()}")
print("G > M rows (sanity):", int((rel.G > rel.M).sum()))

# --- snap-based sensitivity: G_snap = games with offense_pct >= 0.25
snap_wr = snap.merge(meta[["gsis_id", "pfr_id"]].dropna()
                     .rename(columns={"pfr_id": "pfr_player_id"}),
                     on="pfr_player_id", how="inner")
gs = (snap_wr[snap_wr.offense_pct >= 0.25]
      .groupby(["gsis_id", "season"]).size().rename("G_snap").reset_index())
rel = rel.merge(gs.rename(columns={"gsis_id": "player_id"}),
                on=["player_id", "season"], how="left")
matched = rel.player_id.isin(
    meta.dropna(subset=["pfr_id"]).gsis_id) & rel.G_snap.notna()
rel["p_hat_snap"] = rel.G_snap / rel.M
print(f"snap join: {rel.G_snap.notna().mean():.3%} of relevant player-seasons "
      f"have snap data; corr(p_hat, p_hat_snap) = "
      f"{rel[['p_hat','p_hat_snap']].corr().iloc[0,1]:.3f}")
print("snap def minus target def, G diff:",
      (rel.G_snap - rel.G).describe().round(2).to_dict())

out_cols = ["player_id", "name", "season", "age", "G", "M", "p_hat",
            "mean_tgt", "G_snap", "p_hat_snap", "rookie_season"]
rel[out_cols].sort_values(["player_id", "season"]).to_csv(
    f"{ROOT}/results/availability_table.csv", index=False)
print("wrote results/availability_table.csv")

# ---------------------------------------------------------------- Part 2
print("\n" + "=" * 72, "\nPART 2 - is injury-proneness a stable trait?\n",
      "=" * 72, sep="")

# (a) YoY correlation of p_hat, consecutive relevant seasons, player bootstrap
d = rel[["player_id", "season", "p_hat", "p_hat_snap"]].copy()
d1 = d.copy(); d1["season"] += 1
pairs = d.merge(d1, on=["player_id", "season"], suffixes=("_cur", "_prev"))
r_obs = pairs[["p_hat_prev", "p_hat_cur"]].corr().iloc[0, 1]
pid = pairs.player_id.values
uniq = pairs.player_id.unique()
boots = []
grp = {p: pairs[pairs.player_id == p][["p_hat_prev", "p_hat_cur"]].values
       for p in uniq}
for _ in range(4000):
    samp = RNG.choice(uniq, size=len(uniq), replace=True)
    arr = np.vstack([grp[p] for p in samp])
    boots.append(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"(a) YoY corr of p_hat: r = {r_obs:.3f} on {len(pairs)} consecutive-"
      f"season pairs ({len(uniq)} players); player-bootstrap 95% CI "
      f"[{lo:.3f}, {hi:.3f}]")
snap_pairs = pairs.dropna(subset=["p_hat_snap_prev", "p_hat_snap_cur"])
r_snap = snap_pairs[["p_hat_snap_prev", "p_hat_snap_cur"]].corr().iloc[0, 1]
print(f"    snap-definition sensitivity: r = {r_snap:.3f} "
      f"({len(snap_pairs)} pairs)")
# anomaly chase: is stability role persistence or health? restrict to
# high-usage seasons on BOTH sides (mean_tgt >= 6 in s-1 and s)
mt = rel[["player_id", "season", "mean_tgt"]]
mt1 = mt.copy(); mt1["season"] += 1
hp = (pairs.merge(mt, on=["player_id", "season"])
      .merge(mt1, on=["player_id", "season"], suffixes=("_cur", "_prev")))
hp = hp[(hp.mean_tgt_cur >= 6) & (hp.mean_tgt_prev >= 6)]
print(f"    high-usage-only (mean_tgt >= 6 both seasons): r = "
      f"{np.corrcoef(hp.p_hat_prev, hp.p_hat_cur)[0,1]:.3f} "
      f"({len(hp)} pairs; range-restricted - mean p_hat "
      f"{rel[rel.mean_tgt>=6].p_hat.mean():.3f} vs {rel.p_hat.mean():.3f})")

# (b) beta-binomial method of moments across player-seasons
def mom_sigma2(df, pcol="p_hat"):
    p = df[pcol].values; M = df.M.values
    pbar = (df.G if pcol == "p_hat" else df.G_snap).sum() / M.sum()
    v = p.var(ddof=1)
    c = (1.0 / M).mean()
    # Var(p_hat) = sigma2_p + (pbar(1-pbar) - sigma2_p)*E[1/M]
    s2 = (v - pbar * (1 - pbar) * c) / (1 - c)
    return pbar, v, s2, s2 / (pbar * (1 - pbar))

pbar, v_obs, s2_obs, icc_obs = mom_sigma2(rel)
print(f"(b) MoM: pbar = {pbar:.3f}; Var(p_hat) = {v_obs:.4f}; "
      f"binomial-noise-corrected between-player var sigma2_p = {s2_obs:.4f} "
      f"(SD {np.sqrt(max(s2_obs,0)):.3f}); ICC-analogue rho = {icc_obs:.3f}")
sb = rel.dropna(subset=["G_snap"])
print("    snap sensitivity: sigma2_p = %.4f, rho = %.3f"
      % mom_sigma2(sb, "p_hat_snap")[2:])

# H0 test: pure binomial + age. Null p(age) from binomial GLM, simulate.
null_d = rel.dropna(subset=["age"]).copy()
X0 = sm.add_constant(null_d[["age"]].assign(age2=null_d.age ** 2))
glm0 = sm.GLM(np.c_[null_d.G, null_d.M - null_d.G], X0,
              family=sm.families.Binomial()).fit()
p0 = glm0.predict(X0)
sims = []
for _ in range(1000):
    g = RNG.binomial(null_d.M.values, p0.values)
    dd = null_d.assign(G=g, p_hat=g / null_d.M.values)
    sims.append(mom_sigma2(dd)[2])
sims = np.array(sims)
pval = (sims >= s2_obs).mean()
print(f"    H0 (binomial + age): null sigma2_p mean {sims.mean():.4f}, "
      f"95th pct {np.percentile(sims,95):.4f}; observed {s2_obs:.4f} "
      f"-> p = {pval:.4f} ({(sims >= s2_obs).sum()}/1000)")

# (c) game-level logistic with two lags of participation, cluster by player
gl = rel.dropna(subset=["age", "rookie_season"]).copy()
allG = ps[["player_id", "season", "G"]]  # lags from ALL WR seasons (no filter)
for k in (1, 2):
    lag = allG.copy(); lag["season"] += k
    gl = gl.merge(lag.rename(columns={"G": f"G_lag{k}"}),
                  on=["player_id", "season"], how="left")
gl = gl[(gl.season >= 2016) & (gl.season >= gl.rookie_season + 2)].copy()
gl[["G_lag1", "G_lag2"]] = gl[["G_lag1", "G_lag2"]].fillna(0)
rows = gl.loc[gl.index.repeat(gl.M)].copy()
rows["y"] = (rows.groupby(level=0).cumcount() < rows.G).astype(int)
Xc = sm.add_constant(rows[["age", "G_lag1", "G_lag2"]])
logit = sm.Logit(rows.y, Xc).fit(disp=0,
                                 cov_kwds={"groups": rows.player_id.values},
                                 cov_type="cluster")
print(f"(c) game-level logistic, {len(gl)} player-seasons -> {len(rows)} "
      f"game rows, {gl.player_id.nunique()} player clusters")
print(pd.DataFrame({"coef": logit.params, "se": logit.bse,
                    "z": logit.tvalues, "p": logit.pvalues}).round(4))

# ---------------------------------------------------------------- Part 3
print("\n" + "=" * 72, "\nPART 3 - LOSO arm (iv): SV = theta* x E[G]/M\n",
      "=" * 72, sep="")
panel = pd.read_csv(f"{ROOT}/results/edge_panel.csv")   # 300 rows, has age
lp = pd.read_csv(f"{ROOT}/results/loso_predictions.csv")

# outcome: total REG PPR points (ALL games incl. targets<=1) / M
tot = (wk.groupby(["player_id", "season"]).fantasy_points_ppr.sum()
       .rename("total_pts").reset_index())
panel = panel.merge(tot.rename(columns={"player_id": "gsis_id",
                                        "season": "year"}),
                    on=["gsis_id", "year"], how="left")
panel["total_pts"] = panel.total_pts.fillna(0.0)   # 0-game seasons score 0
panel["M"] = np.where(panel.year >= 2021, 17, 16)
panel["ppsw"] = panel.total_pts / panel.M

# availability covariates for panel rows: G in year, lags from all-WR table
allG_any = (wk[wk.targets > 1].groupby(["player_id", "season"])
            .size().rename("G").reset_index())     # any position (panel = WRs)
def g_of(gsis, season):
    m = allG_any[(allG_any.player_id == gsis) & (allG_any.season == season)]
    return int(m.G.iloc[0]) if len(m) else np.nan
panel["G_cur"] = [g_of(g, y) for g, y in zip(panel.gsis_id, panel.year)]
panel["G_cur"] = panel.G_cur.fillna(0)
panel["G1"] = [g_of(g, y - 1) for g, y in zip(panel.gsis_id, panel.year)]
panel["G2"] = [g_of(g, y - 2) for g, y in zip(panel.gsis_id, panel.year)]
panel["miss1"] = panel.G1.isna().astype(int)
panel["miss2"] = panel.G2.isna().astype(int)
panel[["G1", "G2"]] = panel[["G1", "G2"]].fillna(0)
AV_COLS = ["age", "G1", "G2", "miss1", "miss2"]
panel["age"] = panel.age.fillna(panel.age.median())

# ---- replicate script-10 theta* machinery to cover the 9 non-in_fit rows
wk14 = wk[wk.targets > 1]
sm_all = (wk14[wk14.player_id.isin(panel.gsis_id.unique())]
          .groupby(["player_id", "season"])
          .agg(G=("fantasy_points_ppr", "size"),
               ybar=("fantasy_points_ppr", "mean")).reset_index())
wrS = wk14[wk14.position == "WR"].copy()
psS = (wrS.groupby(["player_id", "season"])
       .agg(mean_tgt=("targets", "mean"),
            mu_ps=("fantasy_points_ppr", "mean")).reset_index())
wrS = wrS.merge(psS[psS.mean_tgt >= 3.0], on=["player_id", "season"],
                how="inner")
wrS = wrS.merge(meta[["gsis_id", "rookie_season"]].dropna()
                .rename(columns={"gsis_id": "player_id"}),
                on="player_id", how="left").dropna(subset=["rookie_season"])
wrS["e2"] = (wrS.fantasy_points_ppr - wrS.mu_ps) ** 2
wrS["exp"] = wrS.season - wrS.rookie_season
wrS["tier"] = np.select([wrS.exp == 0, wrS.exp == 1], ["rookie", "soph"], "vet")

def mu_neff_before(gsis, Y):
    h = sm_all[(sm_all.player_id == gsis) & (sm_all.season < Y)]
    if len(h) == 0:
        return np.nan, 0.0
    S = h.season.max()
    w = 2.0 ** (-(S - h.season.values) / 1.0)
    return (float((w * h.ybar.values).sum() / w.sum()),
            float(w.sum() ** 2 / (w ** 2).sum()))

ev_frames = []
for Y in PANEL_YEARS:
    tr = panel[(panel.year != Y) & panel.in_fit].copy()
    ev = panel[panel.year == Y].copy()          # ALL rows, incl. non-in_fit
    # theta* machinery (fits identical to script 10: in_fit training rows)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(np.log(tr.adp.values), tr.ppg.values)
    ev["m_hat_ppg"] = iso.predict(np.log(ev.adp.values))
    tr["r"] = tr.ppg - iso.predict(np.log(tr.adp.values))
    tau2 = tr.groupby("tier").r.var(ddof=1)
    ev["tau2"] = ev.tier.map(tau2).fillna(tr.r.var(ddof=1))
    sig2 = wrS[wrS.season != Y].groupby("tier").e2.mean()
    ev["sig2"] = ev.tier.map(sig2)
    mn = ev.gsis_id.map(lambda g: mu_neff_before(g, Y))
    ev["mu_hat"] = [t[0] for t in mn]
    ev["n_eff"] = [t[1] for t in mn]
    nop = ev.n_eff == 0
    with np.errstate(divide="ignore"):
        V = ev.sig2 / ev.n_eff
    B = np.where(nop, 1.0, V / (V + ev.tau2))
    ev["theta_star_rep"] = np.where(nop, ev.m_hat_ppg,
                                    (1 - B) * ev.mu_hat.fillna(0)
                                    + B * ev.m_hat_ppg)
    # availability model on training-fold panel rows (all 270, not in_fit-only)
    trA = panel[panel.year != Y]
    Xa = sm.add_constant(trA[AV_COLS], has_constant="add")
    glmA = sm.GLM(np.c_[trA.G_cur, trA.M - trA.G_cur], Xa,
                  family=sm.families.Binomial()).fit()
    Xev = sm.add_constant(ev[AV_COLS], has_constant="add")[Xa.columns]
    ev["p_avail"] = glmA.predict(Xev)
    # baseline: ADP-only isotonic RE-FIT on points-per-scheduled-week target
    isoB = IsotonicRegression(increasing=False, out_of_bounds="clip")
    trB = panel[panel.year != Y]
    isoB.fit(np.log(trB.adp.values), trB.ppsw.values)
    ev["m_hat_ppsw"] = isoB.predict(np.log(ev.adp.values))
    ev_frames.append(ev)

evall = pd.concat(ev_frames, ignore_index=True)
# verify replication reproduces frozen round-1 theta* on the 291 in_fit rows
chk = evall.merge(lp[["year", "gsis_id", "theta_star"]],
                  on=["year", "gsis_id"], how="inner")
mad = (chk.theta_star_rep - chk.theta_star).abs().max()
print(f"theta* replication check: {len(chk)} rows, max |diff| = {mad:.2e}")
assert mad < 1e-9, "theta* replication does not match frozen round-1 output"
evall["theta_star"] = evall.theta_star_rep
evall["SV"] = evall.theta_star * evall.p_avail

def score(df, label):
    out = {}
    for nm, col in [("(i) ADP-only refit", "m_hat_ppsw"), ("(iv) SV", "SV")]:
        err = df.ppsw - df[col]
        rmse = float(np.sqrt((err ** 2).mean()))
        rho = df.groupby("year").apply(
            lambda g: stats.spearmanr(g[col], g.ppsw).statistic,
            include_groups=False)
        if col == "m_hat_ppsw":
            dm_t = dm_p = np.nan
        else:
            dsq = (df.ppsw - df.m_hat_ppsw) ** 2 - err ** 2
            dyr = dsq.groupby(df.year).mean()
            dm_t = float(dyr.mean() / (dyr.std(ddof=1) / np.sqrt(len(dyr))))
            dm_p = float(2 * stats.t.sf(abs(dm_t), df=len(dyr) - 1))
            print(f"  [{label}] yearly mean loss diff (>0 = SV better): "
                  f"{dyr.round(3).to_dict()}")
        print(f"  [{label}] {nm}: RMSE {rmse:.4f}, mean Spearman "
              f"{rho.mean():.4f}" +
              ("" if np.isnan(dm_t) else
               f", DM t = {dm_t:+.3f} (9 df), p = {dm_p:.4f}"))
        out[nm] = dict(rmse=rmse, spearman=float(rho.mean()),
                       dm_t=dm_t, dm_p=dm_p)
    return out

print("\n--- primary eval: 291 in_fit rows (round-1 protocol) ---")
score(evall[evall.in_fit], "in_fit")
print("\n--- sensitivity: all 300 panel rows (incl. injury wipe-outs) ---")
score(evall, "all")

# descriptive: does p_avail itself carry signal beyond ADP?
print(f"\np_avail: mean {evall.p_avail.mean():.3f}, sd {evall.p_avail.std():.3f}; "
      f"corr(p_avail, realized G/M) = "
      f"{np.corrcoef(evall.p_avail, evall.G_cur/evall.M)[0,1]:.3f}")

# anomaly chase: how much of the SV win is player-specific availability vs
# the level rescaling theta* -> points-per-scheduled-week? Replace p_avail
# with its fold mean (no cross-sectional availability info) and re-score.
evall["SV_const"] = evall.theta_star * evall.groupby("year").p_avail.transform("mean")
fit = evall[evall.in_fit]
for col, nm in [("SV_const", "SV_const (fold-mean availability)")]:
    err = fit.ppsw - fit[col]
    dsq = (fit.ppsw - fit.m_hat_ppsw) ** 2 - err ** 2
    dyr = dsq.groupby(fit.year).mean()
    t = dyr.mean() / (dyr.std(ddof=1) / np.sqrt(len(dyr)))
    print(f"decomposition [in_fit] {nm}: RMSE {np.sqrt((err**2).mean()):.4f}, "
          f"DM vs (i) t = {t:+.3f}, p = {2*stats.t.sf(abs(t), 9):.4f}")
d2 = (fit.ppsw - fit.SV_const) ** 2 - (fit.ppsw - fit.SV) ** 2
dy2 = d2.groupby(fit.year).mean()
t2 = dy2.mean() / (dy2.std(ddof=1) / np.sqrt(len(dy2)))
print(f"decomposition: SV vs SV_const head-to-head (>0 = player-specific "
      f"availability helps): t = {t2:+.3f}, p = {2*stats.t.sf(abs(t2), 9):.4f}")

keep = ["year", "name", "gsis_id", "adp", "tier", "in_fit", "M", "G_cur",
        "total_pts", "ppsw", "theta_star", "p_avail", "SV", "m_hat_ppsw",
        "G1", "G2", "miss1", "miss2", "age"]
evall[keep].to_csv(f"{ROOT}/results/loso_availability.csv", index=False)
print("\nwrote results/loso_availability.csv")

"""§7 leave-one-season-out validation (EDA_PLAN.md §7).

For each board year Y in 2015-2024: hold out year Y entirely; refit on the
other 9 years
  - m_{-Y}(ADP): isotonic, monotone decreasing in log ADP, in_fit rows only;
  - tau^2_{-Y}(tier): var of isotonic residuals by tier (in_fit rows);
  - sigma^2_{-Y}(tier): §3 primary sample (ALL WRs 2014-2025, REG, targets>1,
    player-season mean targets >= 3), seasons != Y, mean squared deviation
    from the player-season mean, by tier (Route-B-with-saturated-dummies
    equals raw mean e^2 by tier, so this is the §3 headline estimator);
then for every player on year-Y's board compute mu_hat and n_eff from
weekly_raw STRICTLY BEFORE year Y (h=1 recency weights on season-year
distance, §1 inclusion rule), and form the posterior theta* per eq (7).
Players with zero prior included games get B = 1 (theta* = m_hat).

DATA LIMITATION (documented): weekly_raw starts 2014, so careers are
truncated at 2014 when building mu_hat for early boards (only 1 prior season
available for Y=2015). h=1 recency weighting makes this second-order for
Y >= 2017. Any non-rookie with no included games in 2014..Y-1 is treated as
no-prior-data (B=1); counts reported.

Candidate predictors of realized year-Y PPG, evaluated on in_fit rows:
  (i)   baseline: m_{-Y}(ADP)
  (ii)  blind posterior theta*
  (iii) theta* with prior mean m_{-Y}(ADP) + Z'beta_{-Y}, beta refit on years
        != Y with the §6.2 FDR-surviving terms (rookie, rook_x_epa). NOTE:
        these terms already FAILED the binding temporal holdout (§6.3); (iii)
        is reported as the pre-specified candidate, not as a live model.
Metrics: pooled RMSE, mean within-year Spearman rho, and a paired
Diebold-Mariano-type test on squared-error differences vs (i), clustered by
year (t on the 10 yearly mean loss differentials, 9 df).

Outputs: results/loso_scorecard.csv, results/loso_predictions.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.isotonic import IsotonicRegression

ROOT = "/Users/thomasmcnamee/NFL"
YEARS = list(range(2015, 2025))
FDR_TERMS = ["rookie", "rook_x_epa"]     # §6.2 FDR survivors (failed temporal)

panel = pd.read_csv(f"{ROOT}/results/edge_panel.csv")   # market panel + Z covs

# ---------------- weekly data (2014-2025) ----------------
wk_frames = []
for y in range(2014, 2026):
    df = pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
                     usecols=["player_id", "position", "season", "season_type",
                              "targets", "fantasy_points_ppr"], low_memory=False)
    wk_frames.append(df[(df.season_type == "REG") & (df.targets > 1)])
wk = pd.concat(wk_frames, ignore_index=True)

# season means for every panel player (mu_hat inputs), all WR rows for sigma^2
sm_all = (wk[wk.player_id.isin(panel.gsis_id.unique())]
          .groupby(["player_id", "season"])
          .agg(G=("fantasy_points_ppr", "size"),
               ybar=("fantasy_points_ppr", "mean")).reset_index())

# §3 primary sample for sigma^2(tier): all WRs, mean tgt >= 3
wr = wk[wk.position == "WR"].copy()
ps = (wr.groupby(["player_id", "season"])
        .agg(mean_tgt=("targets", "mean"), mu_ps=("fantasy_points_ppr", "mean"))
        .reset_index())
wr = wr.merge(ps[ps.mean_tgt >= 3.0], on=["player_id", "season"], how="inner")
meta = pd.read_csv(f"{ROOT}/data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "rookie_season"]).dropna()
wr = wr.merge(meta.rename(columns={"gsis_id": "player_id"}), on="player_id",
              how="left").dropna(subset=["rookie_season"])
wr["e2"] = (wr.fantasy_points_ppr - wr.mu_ps) ** 2
wr["exp"] = wr.season - wr.rookie_season
wr["tier"] = np.select([wr.exp == 0, wr.exp == 1], ["rookie", "soph"], "vet")

def mu_neff_before(gsis, Y):
    """Recency-weighted mu_hat and n_eff from seasons strictly before Y (h=1)."""
    h = sm_all[(sm_all.player_id == gsis) & (sm_all.season < Y)]
    if len(h) == 0:
        return np.nan, 0.0
    S = h.season.max()
    w = 2.0 ** (-(S - h.season.values) / 1.0)
    return float((w * h.ybar.values).sum() / w.sum()), float(w.sum() ** 2 / (w ** 2).sum())

# ---------------- LOSO loop ----------------
preds, fold_notes = [], []
for Y in YEARS:
    tr = panel[(panel.year != Y) & panel.in_fit].copy()
    ev = panel[(panel.year == Y) & panel.in_fit].copy()

    # m_{-Y}
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(np.log(tr.adp.values), tr.ppg.values)
    tr["r"] = tr.ppg - iso.predict(np.log(tr.adp.values))
    ev["m_hat"] = iso.predict(np.log(ev.adp.values))

    # tau^2_{-Y}(tier); fall back to pooled var if a tier has < 2 rows
    tau2 = tr.groupby("tier").r.var(ddof=1)
    tau2_pooled = tr.r.var(ddof=1)
    ev["tau2"] = ev.tier.map(tau2).fillna(tau2_pooled)

    # sigma^2_{-Y}(tier)
    sig2 = wr[wr.season != Y].groupby("tier").e2.mean()
    ev["sig2"] = ev.tier.map(sig2)

    # mu_hat, n_eff strictly before Y
    mn = ev.gsis_id.map(lambda g: mu_neff_before(g, Y))
    ev["mu_hat"] = [t[0] for t in mn]
    ev["n_eff"] = [t[1] for t in mn]
    no_prior = ev.n_eff == 0
    n_noprior_vet = int((no_prior & (ev.exp > 0)).sum())

    # posterior
    with np.errstate(divide="ignore"):
        V = ev.sig2 / ev.n_eff
    B = np.where(no_prior, 1.0, V / (V + ev.tau2))
    ev["B"] = B
    ev["theta_star"] = np.where(no_prior, ev.m_hat,
                                (1 - B) * ev.mu_hat.fillna(0) + B * ev.m_hat)

    # (iii) edge-adjusted prior mean: beta_{-Y} on FDR survivors.
    # Target is tr["r"] — residuals from the LEAVE-Y-OUT isotonic fit above —
    # not the panel's resid_iso (whose curve was fit on all 10 years incl. Y;
    # using it leaked year-Y outcomes into beta_{-Y}. Bug found+fixed in the
    # verification pass; effect confined to predictor (iii), see section7 notes).
    Xtr = sm.add_constant(tr[FDR_TERMS])
    fb = sm.OLS(tr["r"], Xtr).fit()
    Xev = sm.add_constant(ev[FDR_TERMS])[Xtr.columns]
    ev["m_edge"] = ev.m_hat + fb.predict(Xev)
    ev["theta_edge"] = np.where(no_prior, ev.m_edge,
                                (1 - B) * ev.mu_hat.fillna(0) + B * ev.m_edge)

    fold_notes.append(dict(Y=Y, n_eval=len(ev), n_noprior=int(no_prior.sum()),
                           n_noprior_vet=n_noprior_vet,
                           tau2_soph=tau2.get("soph", np.nan),
                           tau2_vet=tau2.get("vet", np.nan),
                           sig2_vet=sig2.get("vet", np.nan),
                           beta_rookie=fb.params["rookie"],
                           beta_rxepa=fb.params["rook_x_epa"]))
    preds.append(ev[["year", "name", "gsis_id", "adp", "tier", "games", "ppg",
                     "m_hat", "mu_hat", "n_eff", "B", "theta_star", "theta_edge"]])

pred = pd.concat(preds, ignore_index=True)
fn = pd.DataFrame(fold_notes)
print("=== fold diagnostics ===")
print(fn.round(3).to_string(index=False))
print(f"\ntotal eval rows: {len(pred)}; no-prior-data rows (B=1): "
      f"{(pred.B == 1).sum()}")

# ---------------- scorecard ----------------
CANDS = {"(i) ADP-only m_hat": "m_hat",
         "(ii) blind theta*": "theta_star",
         "(iii) theta* + FDR edge terms": "theta_edge"}

rows = []
for label, col in CANDS.items():
    err = pred.ppg - pred[col]
    rmse = float(np.sqrt((err ** 2).mean()))
    rho = pred.groupby("year").apply(
        lambda g: stats.spearmanr(g[col], g.ppg).statistic,
        include_groups=False)
    # DM-type paired test vs baseline (i), clustered by year
    if col == "m_hat":
        dm_t, dm_p = np.nan, np.nan
    else:
        dsq = (pred.ppg - pred.m_hat) ** 2 - err ** 2   # >0 = candidate better
        dyr = dsq.groupby(pred.year).mean()
        dm_t = float(dyr.mean() / (dyr.std(ddof=1) / np.sqrt(len(dyr))))
        dm_p = float(2 * stats.t.sf(abs(dm_t), df=len(dyr) - 1))
    rows.append(dict(predictor=label, rmse=rmse, mean_spearman=float(rho.mean()),
                     dm_t_vs_adp=dm_t, dm_p_vs_adp=dm_p))
    print(f"\n{label}: RMSE {rmse:.4f}, mean within-year Spearman {rho.mean():.4f}")
    print("  by year rho:", rho.round(3).to_dict())
    if col != "m_hat":
        print(f"  DM vs (i): t = {dm_t:+.3f} (9 df), p = {dm_p:.4f}; "
              f"yearly mean loss diff: {dyr.round(3).to_dict()}")

# (ii) vs (iii) head-to-head (descriptive)
d23 = (pred.ppg - pred.theta_star) ** 2 - (pred.ppg - pred.theta_edge) ** 2
dyr23 = d23.groupby(pred.year).mean()
t23 = float(dyr23.mean() / (dyr23.std(ddof=1) / np.sqrt(len(dyr23))))
print(f"\n(iii) vs (ii): mean loss diff {d23.mean():+.4f} "
      f"(>0 = edge helps), clustered t = {t23:+.3f}, "
      f"p = {2 * stats.t.sf(abs(t23), df=9):.4f}")

sc = pd.DataFrame(rows)
sc.to_csv(f"{ROOT}/results/loso_scorecard.csv", index=False)
pred.to_csv(f"{ROOT}/results/loso_predictions.csv", index=False)
print("\nwrote results/loso_scorecard.csv, loso_predictions.csv")

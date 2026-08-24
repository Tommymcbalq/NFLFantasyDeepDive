"""§S2 — 11-fold LOSO: does a recency-weighted market prior beat the era-flat one?

Extends §7 to 11 folds (2015-2025 board years, 2025 ADP added 2026-08-18) and adds the
season-weighted prior as a candidate arm.

PRE-REGISTERED before running:
  Weights.  For held-out year Y, training season t gets w = 2^{-|t - Y| / h}.  Distance to
  the EVALUATED season, not to 2025.  Rationale, stated before seeing results: the era
  hypothesis is that m(.) is locally stationary in calendar time, so the informative
  training seasons are the ones NEAR Y in both directions.  Weighting toward 2025 inside a
  fold would instead ask "do late seasons predict early ones", which is not the hypothesis
  and is not the 2026 use case.  For the live 2026 board Y=2026 lies beyond every training
  season, so |t - Y| collapses to pure recency and the two coincide there.
  tau^2(tier) uses the SAME weights (weighted residual variance) — a recency-weighted mean
  with an era-flat spread would be internally inconsistent.
  sigma^2(tier) is left flat: it is a within-player game-level variance estimated on the
  full WR population, not a market quantity, and no era hypothesis was raised about it.

  Arms:  (i)  m_hat flat            [baseline, as §7]
         (ii) theta* flat           [the current model]
         (iii-v) m_hat  h = 4, 2, 1
         (vi-viii) theta* h = 4, 2, 1
  ADOPTION RULE, fixed now: an arm replaces (ii) only if it BOTH lowers pooled RMSE and
  gives DM t > 0 with p < .05 vs (ii), clustered on the 11 folds.  In-sample fit from
  script 50 is not evidence and does not enter.
"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

ROOT = "/Users/thomasmcnamee/NFL"
YEARS = list(range(2015, 2026))
panel = pd.read_csv(f"{ROOT}/results/market_prior_11yr.csv")

wk_frames = []
for y in range(2014, 2026):
    df = pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
                     usecols=["player_id","position","season","season_type",
                              "targets","fantasy_points_ppr"], low_memory=False)
    wk_frames.append(df[(df.season_type=="REG") & (df.targets>1)])
wk = pd.concat(wk_frames, ignore_index=True)

sm_all = (wk[wk.player_id.isin(panel.gsis_id.unique())].groupby(["player_id","season"])
          .agg(G=("fantasy_points_ppr","size"), ybar=("fantasy_points_ppr","mean")).reset_index())

wr = wk[wk.position=="WR"].copy()
ps = (wr.groupby(["player_id","season"])
        .agg(mean_tgt=("targets","mean"), mu_ps=("fantasy_points_ppr","mean")).reset_index())
wr = wr.merge(ps[ps.mean_tgt>=3.0], on=["player_id","season"], how="inner")
meta = pd.read_csv(f"{ROOT}/data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id","rookie_season"]).dropna()
wr = wr.merge(meta.rename(columns={"gsis_id":"player_id"}), on="player_id", how="left").dropna(subset=["rookie_season"])
wr["e2"] = (wr.fantasy_points_ppr - wr.mu_ps)**2
wr["exp"] = wr.season - wr.rookie_season
wr["tier"] = np.select([wr.exp==0, wr.exp==1], ["rookie","soph"], "vet")

def mu_neff_before(gsis, Y):
    h = sm_all[(sm_all.player_id==gsis) & (sm_all.season<Y)]
    if len(h)==0: return np.nan, 0.0
    S = h.season.max(); w = 2.0**(-(S-h.season.values)/1.0)
    return float((w*h.ybar.values).sum()/w.sum()), float(w.sum()**2/(w**2).sum())

def wvar(x, w):
    mu = np.sum(w*x)/np.sum(w)
    return float(np.sum(w*(x-mu)**2) / (np.sum(w) - np.sum(w**2)/np.sum(w)))

HS = [(np.inf,"flat"), (4.0,"h4"), (2.0,"h2"), (1.0,"h1")]
preds = []
for Y in YEARS:
    tr = panel[(panel.year!=Y) & panel.in_fit].copy()
    ev = panel[(panel.year==Y) & panel.in_fit].copy()
    sig2 = wr[wr.season!=Y].groupby("tier").e2.mean()
    ev["sig2"] = ev.tier.map(sig2)
    mn = ev.gsis_id.map(lambda g: mu_neff_before(g, Y))
    ev["mu_hat"] = [t[0] for t in mn]; ev["n_eff"] = [t[1] for t in mn]
    no_prior = ev.n_eff==0

    for h, tag in HS:
        w = np.ones(len(tr)) if np.isinf(h) else 2.0**(-np.abs(tr.year.values-Y)/h)
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso.fit(np.log(tr.adp.values), tr.ppg.values, sample_weight=w)
        r = tr.ppg.values - iso.predict(np.log(tr.adp.values))
        m_hat = iso.predict(np.log(ev.adp.values))
        tau2 = {}
        for t_ in ["rookie","soph","vet"]:
            msk = (tr.tier==t_).values
            tau2[t_] = wvar(r[msk], w[msk]) if msk.sum()>=2 else np.nan
        pooled = wvar(r, w)
        tv = ev.tier.map(tau2).fillna(pooled).values
        with np.errstate(divide="ignore"):
            V = (ev.sig2/ev.n_eff).values
        B = np.where(no_prior, 1.0, V/(V+tv))
        ev[f"m_{tag}"] = m_hat
        ev[f"th_{tag}"] = np.where(no_prior, m_hat, (1-B)*ev.mu_hat.fillna(0).values + B*m_hat)
        if tag=="flat": ev["B_flat"]=B
    preds.append(ev)

pred = pd.concat(preds, ignore_index=True)
print(f"eval rows {len(pred)}, folds {pred.year.nunique()}, no-prior (B=1) {(pred.B_flat==1).sum()}")

def dm(col, ref):
    d = (pred.ppg-pred[ref])**2 - (pred.ppg-pred[col])**2
    dy = d.groupby(pred.year).mean()
    t = dy.mean()/(dy.std(ddof=1)/np.sqrt(len(dy)))
    return float(t), float(2*stats.t.sf(abs(t), df=len(dy)-1))

rows=[]
for tag_label, col in [("(i) m_hat flat","m_flat"),("(ii) theta* flat","th_flat"),
                       ("(iii) m_hat h4","m_h4"),("(iv) m_hat h2","m_h2"),("(v) m_hat h1","m_h1"),
                       ("(vi) theta* h4","th_h4"),("(vii) theta* h2","th_h2"),("(viii) theta* h1","th_h1")]:
    err = pred.ppg-pred[col]
    rmse = float(np.sqrt((err**2).mean()))
    rho = pred.groupby("year").apply(lambda g: stats.spearmanr(g[col], g.ppg).statistic, include_groups=False).mean()
    t_i,p_i = dm(col,"m_flat") if col!="m_flat" else (np.nan,np.nan)
    t_ii,p_ii = dm(col,"th_flat") if col!="th_flat" else (np.nan,np.nan)
    rows.append(dict(arm=tag_label, rmse=round(rmse,4), spearman=round(float(rho),4),
                     dm_t_vs_i=round(t_i,3) if t_i==t_i else np.nan,
                     dm_p_vs_i=round(p_i,4) if p_i==p_i else np.nan,
                     dm_t_vs_ii=round(t_ii,3) if t_ii==t_ii else np.nan,
                     dm_p_vs_ii=round(p_ii,4) if p_ii==p_ii else np.nan))
sc = pd.DataFrame(rows)
print("\n=== 11-fold LOSO scorecard ===")
print(sc.to_string(index=False))
sc.to_csv(f"{ROOT}/results/loso11_scorecard.csv", index=False)
pred.to_csv(f"{ROOT}/results/loso11_predictions.csv", index=False)

best = sc[sc.arm.str.contains("theta")].sort_values("rmse").iloc[0]
print(f"\nADOPTION CHECK vs (ii): best theta arm = {best.arm}, RMSE {best.rmse} "
      f"vs flat {sc[sc.arm=='(ii) theta* flat'].rmse.iloc[0]}")

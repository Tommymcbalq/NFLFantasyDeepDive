"""§O5 LOSO validation + the 2026 TE and QB boards (EDA_PLAN6.md §O).

Mirrors §G6 (scripts/24_rb_g4_g5_g6.py) exactly, position swapped.  Nothing is inherited
from the WR or RB pipelines: m(.), tau^2(tier), sigma^2(tier) and mu_hat are all refit
per position and, inside the LOSO, per fold.

PRE-REGISTERED (fixed before any fold was scored; §O5 of EDA_PLAN6.md):

  LOSO over the 2015-2024 boards.  For each held-out year Y everything is refit on the
  other nine: m_{-Y}(ADP) isotonic decreasing in log ADP; tau^2_{-Y}(tier) from training
  isotonic residuals; sigma^2_{-Y}(tier) from all gated position games in seasons != Y.
  mu_hat / n_eff are rebuilt from weekly_raw seasons STRICTLY BEFORE Y (recency half-life
  h = 1).  weekly_raw reaches 1999 so there is no left-truncation.  Zero prior included
  games => n_eff = 0 => B = 1 => theta* = m(ADP).

  Arms:  (i)  market-only  m_{-Y}(ADP)
         (ii) market + data EB posterior theta* = (1-B) mu_hat + B m,  B = V/(V+tau^2)

  ADOPTION RULE: arm (ii) is adopted over arm (i) iff the Diebold-Mariano t-test on yearly
  mean squared-error differentials, t(9 df), gives p < 0.10 AND pooled RMSE improves.

  HONESTY CLAUSE (EDA_PLAN6 §O5, binding): if the data arm does not beat market-only, the
  position is market-anchored and we do NOT go hunting for an arm that wins.  Only the two
  pre-registered arms are scored.

  §O3's operational prediction is settled here: "QB mu_hat is more reliable and B should
  shrink LESS toward market."  Realized mean B is reported per position against the WR and
  RB values from the existing LOSO scorecards.

Outputs: results/loso_predictions_te.csv, loso_predictions_qb.csv,
         results/loso_scorecard_te.csv, loso_scorecard_qb.csv,
         results/valuation_te_2026.csv, valuation_qb_2026.csv,
         results/sectionO_shrinkage.csv
Rerun: python3 scripts/36_sectionO_o5_loso.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

warnings.simplefilter("ignore")
ROOT = Path("/Users/thomasmcnamee/NFL")
YEARS = list(range(2015, 2025))
CUT = {"TE": ("targets", 1), "QB": ("attempts", 5)}
GATE = {"TE": ("targets", 2), "QB": ("attempts", 21)}   # §O3c relevance gates

COLS = ["player_id", "position", "season", "season_type", "targets", "attempts",
        "carries", "fantasy_points_ppr"]
frames = []
for y in range(1999, 2026):
    d = pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{y}.csv",
                    usecols=lambda c: c in COLS, low_memory=False)
    frames.append(d[d.season_type == "REG"])
wk = pd.concat(frames, ignore_index=True)
for c in ("targets", "attempts", "carries"):
    wk[c] = wk[c].fillna(0)
print(f"weekly REG rows 1999-2025: {len(wk)}")

meta = pd.read_csv(ROOT / "data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "birth_date", "rookie_season"])
uni = pd.read_csv(ROOT / "results/sectionO_universe_2026.csv")

shrink_rows, verdicts = [], {}
for P in ("TE", "QB"):
    ucol, ucut = CUT[P]
    gcol, gval = GATE[P]
    print("\n" + "=" * 78)
    print(f"§O5  LOSO 2015-2024 — {P}")
    print("=" * 78)

    inc = wk[(wk.position == P) & (wk[ucol] > ucut)]
    # gated sample for sigma^2(tier), the §O3c relevance gate
    ps = inc.groupby(["player_id", "season"])[gcol].mean().rename("upg").reset_index()
    gat = inc.merge(ps[ps.upg >= gval][["player_id", "season"]],
                    on=["player_id", "season"])
    mu_ps = gat.groupby(["player_id", "season"]).fantasy_points_ppr.transform("mean")
    gat = gat.assign(e2=(gat.fantasy_points_ppr - mu_ps) ** 2)
    gat = gat.merge(meta.rename(columns={"gsis_id": "player_id"}), on="player_id",
                    how="left").dropna(subset=["rookie_season"])
    gat["exp"] = gat.season - gat.rookie_season
    gat["tier"] = np.select([gat.exp == 0, gat.exp == 1], ["rookie", "soph"], "vet")

    # per-player per-season means for mu_hat (all history, 1999+)
    sm_all = (inc.groupby(["player_id", "season"])
              .agg(ybar=("fantasy_points_ppr", "mean")).reset_index())
    sm_idx = {p: g[["season", "ybar"]].values for p, g in sm_all.groupby("player_id")}

    def mu_neff_before(g, Y):
        a = sm_idx.get(g)
        if a is None:
            return np.nan, 0.0
        a = a[a[:, 0] < Y]
        if len(a) == 0:
            return np.nan, 0.0
        w = 2.0 ** (-(a[:, 0].max() - a[:, 0]) / 1.0)
        return float((w * a[:, 1]).sum() / w.sum()), float(w.sum() ** 2 / (w ** 2).sum())

    panel = pd.read_csv(ROOT / f"results/market_prior_{P.lower()}.csv")
    print(f"panel {len(panel)} rows, in_fit {int(panel.in_fit.sum())}")

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
        sig2 = gat[gat.season != Y].groupby("tier").e2.mean()
        ev["sig2"] = ev.tier.map(sig2).fillna(gat[gat.season != Y].e2.mean())
        mn = [mu_neff_before(g, Y) for g in ev.gsis_id]
        ev["mu_hat"] = [t[0] for t in mn]
        ev["n_eff"] = [t[1] for t in mn]
        nop = ev.n_eff == 0
        with np.errstate(divide="ignore"):
            V = ev.sig2 / ev.n_eff
        B = np.where(nop, 1.0, V / (V + ev.tau2))
        ev["V"] = V
        ev["B"] = B
        ev["theta_star"] = np.where(nop, ev.m_hat,
                                    (1 - B) * ev.mu_hat.fillna(0) + B * ev.m_hat)
        notes.append(dict(Y=Y, n_eval=int(ev.in_fit.sum()), n_noprior=int(nop.sum()),
                          n_rookie=int((ev.tier == "rookie").sum()),
                          tau2_rookie=tau2.get("rookie", np.nan),
                          tau2_soph=tau2.get("soph", np.nan),
                          tau2_vet=tau2.get("vet", np.nan),
                          sig2_vet=sig2.get("vet", np.nan),
                          mean_B=float(np.mean(B)),
                          mean_B_hasdata=float(np.mean(B[~nop.values]))))
        preds.append(ev)

    pred = pd.concat(preds, ignore_index=True)
    print("\n=== fold diagnostics ===")
    print(pd.DataFrame(notes).round(3).to_string(index=False))
    print(f"total rows {len(pred)}, in_fit {int(pred.in_fit.sum())}, "
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
        dy = None
        if col != "m_hat":
            t, p, dy = dm(fit, "ppg", "m_hat", col)
            print(f"  yearly mean loss diff (>0 = arm better): {dy.round(3).to_dict()}")
            print(f"  folds improved: {(dy > 0).sum()}/10")
        rows.append(dict(pos=P, target="ppg", predictor=nm, rmse=rmse,
                         mean_spearman=float(rho.mean()), dm_t_vs_market=t,
                         dm_p_vs_market=p, n=len(fit),
                         mde=2.802 * (dy.std(ddof=1) / np.sqrt(len(dy))) if dy is not None else np.nan))
        print(f"  {nm}: RMSE {rmse:.4f}, Spearman {rho.mean():.4f}"
              + ("" if np.isnan(t) else f", DM t {t:+.3f}, p {p:.4f}"))

    # drop-one-fold stability of the DM p (the §G6 diagnostic), reported either way
    t_full, p_full, dyf = dm(fit, "ppg", "m_hat", "theta_star")
    dps = []
    for y in YEARS:
        d2 = dyf.drop(index=y)
        tt = d2.mean() / (d2.std(ddof=1) / np.sqrt(len(d2)))
        dps.append((y, float(2 * stats.t.sf(abs(tt), len(d2) - 1))))
    print(f"  drop-one-fold p range: {min(x[1] for x in dps):.3f}"
          f"..{max(x[1] for x in dps):.3f}")
    print(f"  per-fold gain: mean {dyf.mean():+.3f}, across-fold SD {dyf.std(ddof=1):.3f} "
          f"(WR +0.695/0.819, RB +0.488/2.015)")

    sc = pd.DataFrame(rows)
    sc.to_csv(ROOT / f"results/loso_scorecard_{P.lower()}.csv", index=False)
    pred.to_csv(ROOT / f"results/loso_predictions_{P.lower()}.csv", index=False)

    i0 = sc[sc.predictor.str.startswith("(i)")].iloc[0]
    ii = sc[sc.predictor.str.startswith("(ii)")].iloc[0]
    adopt = bool((ii.dm_p_vs_market < 0.10) and (ii.rmse < i0.rmse))
    verdicts[P] = adopt
    print(f"\nADOPTION [{P}]: arm (ii) blind theta* -> "
          f"{'ADOPTED' if adopt else 'NOT ADOPTED'} "
          f"(p {ii.dm_p_vs_market:.4f}, RMSE {ii.rmse:.4f} vs {i0.rmse:.4f})")
    if not adopt:
        print(f"  §O5 honesty clause fires: the {P} board is MARKET-ANCHORED, "
              f"board_value = m(ADP).  No further arms are tried.")

    shrink_rows.append(dict(pos=P, mean_B_all=float(pred.B.mean()),
                            mean_B_hasdata=float(pred.loc[pred.B < 1, "B"].mean()),
                            median_B_hasdata=float(pred.loc[pred.B < 1, "B"].median()),
                            mean_V=float(pred.loc[np.isfinite(pred.V), "V"].mean()),
                            mean_tau2=float(pred.tau2.mean()),
                            mean_sig2=float(pred.sig2.mean()),
                            mean_neff=float(pred.n_eff.mean()),
                            frac_no_prior=float((pred.B == 1).mean()),
                            rmse_market=float(i0.rmse), rmse_theta=float(ii.rmse),
                            dm_p=float(ii.dm_p_vs_market), adopted=adopt))

# --------------------------------------------------------- §O3 operational prediction
print("\n" + "=" * 78)
print("§O3 prediction, operational half: does B shrink LESS toward market at QB?")
print("=" * 78)
for p, f in [("WR", "loso_predictions3.csv"), ("RB", "loso_predictions_rb.csv")]:
    try:
        d = pd.read_csv(ROOT / f"results/{f}")
        bc = "B" if "B" in d.columns else [c for c in d.columns if c.lower() == "b"][0]
        shrink_rows.append(dict(pos=p, mean_B_all=float(d[bc].mean()),
                                mean_B_hasdata=float(d.loc[d[bc] < 1, bc].mean()),
                                median_B_hasdata=float(d.loc[d[bc] < 1, bc].median()),
                                mean_V=np.nan,
                                mean_tau2=float(d.tau2.mean()) if "tau2" in d else np.nan,
                                mean_sig2=float(d.sig2.mean()) if "sig2" in d else np.nan,
                                mean_neff=float(d.n_eff.mean()) if "n_eff" in d else np.nan,
                                frac_no_prior=float((d[bc] == 1).mean()),
                                rmse_market=np.nan, rmse_theta=np.nan, dm_p=np.nan,
                                adopted=np.nan))
    except Exception as e:
        print(f"  ({p} comparison unavailable: {e})")
SH = pd.DataFrame(shrink_rows)
SH.to_csv(ROOT / "results/sectionO_shrinkage.csv", index=False)
print(SH.round(4).to_string(index=False))
print("\nB is the weight on the MARKET.  Lower B = the data arm gets more weight = "
      "mu_hat is more reliable relative to the prior.")

# ================================================================= 2026 boards
for P in ("TE", "QB"):
    print("\n" + "=" * 78)
    print(f"2026 {P} board")
    print("=" * 78)
    ucol, ucut = CUT[P]
    ct = pd.read_csv(ROOT / f"results/consistency_table_{P.lower()}.csv")[
        ["gsis_id", "player", "mu_hat", "n_eff", "n_games", "n_seasons", "sigma_W",
         "boom_eb", "bust_eb"]]
    sig2t = pd.read_csv(ROOT / f"results/sigma2_by_tier_{P.lower()}.csv") \
        .set_index("tier").sigma2
    tau2t = pd.read_csv(ROOT / f"results/tier_variances_{P.lower()}.csv") \
        .set_index("tier").tau2_iso
    knots = pd.read_csv(ROOT / f"results/market_prior_iso_knots_{P.lower()}.csv")

    b = uni[uni.pos == P].copy().merge(ct, on="gsis_id", how="left")
    b["exp_2026"] = 2026 - b.rookie_season
    b["tier"] = np.select([b.exp_2026 == 0, b.exp_2026 == 1], ["rookie", "soph"], "vet")
    b.loc[b.rookie_season.isna(), "tier"] = "rookie"
    b["n_eff"] = b.n_eff.fillna(0.0)
    b["n_seasons"] = b.n_seasons.fillna(0).astype(int)
    b["player"] = b.player.fillna(b.name)
    print("tier counts:", b.tier.value_counts().to_dict())
    print("zero-NFL-row players (pure market arm):", b.loc[b.n_eff == 0, "name"].tolist())
    print("single-season players (thin):", b.loc[b.n_seasons == 1, "name"].tolist())

    b["m_adp"] = np.interp(np.log(b.adp), knots.log_adp, knots.m)
    b["tau2"] = b.tier.map(tau2t)
    b["sigma2_tier"] = b.tier.map(sig2t)
    with np.errstate(divide="ignore"):
        b["V"] = b.sigma2_tier / b.n_eff
    nop = b.n_eff == 0
    b["B"] = np.where(nop, 1.0, b.V / (b.V + b.tau2))
    b["theta_star"] = np.where(nop, b.m_adp,
                               (1 - b.B) * b.mu_hat.fillna(0) + b.B * b.m_adp)
    b["post_SD"] = np.where(nop, np.sqrt(b.tau2),
                            np.sqrt(1.0 / (1.0 / b.V + 1.0 / b.tau2)))
    b["thin_data_flag"] = np.where(nop, "no NFL rows: full shrinkage to market",
                                   np.where(b.n_seasons == 1, "single season: n_eff = 1",
                                            ""))
    b["arm_ii_adopted"] = verdicts[P]
    b["board_value"] = b.theta_star if verdicts[P] else b.m_adp
    b = b.sort_values(["board_value", "adp"], ascending=[False, True]).reset_index(drop=True)
    b["rank_model"] = b.index + 1
    b["delta_rank_vs_adp"] = b.pos_adp_rank - b.rank_model
    cols = ["rank_model", "player", "team", "adp", "pos_adp_rank", "tier", "n_seasons",
            "mu_hat", "n_eff", "sigma_W", "V", "m_adp", "tau2", "B", "theta_star",
            "post_SD", "board_value", "delta_rank_vs_adp", "thin_data_flag",
            "arm_ii_adopted", "gsis_id"]
    b[cols].to_csv(ROOT / f"results/valuation_{P.lower()}_2026.csv", index=False)
    print()
    print(b[cols[:-1]].round(3).to_string(index=False))

print("\nwrote loso_scorecard_{te,qb}.csv, loso_predictions_{te,qb}.csv, "
      "valuation_{te,qb}_2026.csv, sectionO_shrinkage.csv")

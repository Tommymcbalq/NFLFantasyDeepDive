"""§S1/§S2 — the mu_hat bake-off.

Eight candidate summaries of a player's history replace ONLY mu_hat inside eq. (7).
B, V, tau2(tier), sigma2(tier), m(.), the inclusion rule and the fold structure are
identical across arms.  LOSO 2015-2024 on the §P wide board panels, so results are
directly comparable to §7 (adp_rank <= 30 stratum) and §P (full wide panel).

Operational definitions are pre-registered in results/sectionS_notes.md, written before
this script was run.  Nothing here is tuned.

Outputs: results/sectionS_bakeoff.csv        one row per (pos, arm, panel)
         results/sectionS_predictions.csv    per-row LOSO predictions, all arms
         results/sectionS_holdout.csv        temporal holdout 2015-21 -> 2022-24
         results/sectionS_sensitivity.csv    declared sensitivities
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
GATE = {"WR": 3.0, "RB": 4.0}
HL = 1.0                      # recency half-life of the incumbent, frozen
MIN_G_STABLE = 12             # candidate 6 threshold, from §P

# ------------------------------------------------------------------ weekly data
COLS = ["player_id", "position", "season", "week", "season_type", "team",
        "targets", "carries", "fantasy_points_ppr"]
frames = []
for y in range(1999, 2026):
    d = pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{y}.csv",
                    usecols=lambda c: c in COLS, low_memory=False)
    frames.append(d[d.season_type == "REG"])
wk = pd.concat(frames, ignore_index=True)
wk["touches"] = wk.carries.fillna(0) + wk.targets.fillna(0)
INC = {"WR": wk[wk.targets.fillna(0) >= 2], "RB": wk[wk.touches >= 2]}
RELCOL = {"WR": "targets", "RB": "touches"}
VOLCOL = {"WR": "targets", "RB": "touches"}
meta = pd.read_csv(ROOT / "data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "birth_date", "rookie_season"])

# team volume per game, for candidate 8
_tv = wk.groupby(["team", "season"]).agg(tgt=("targets", "sum"),
                                         tch=("touches", "sum")).reset_index()
_tg = wk.groupby(["team", "season"]).week.nunique().rename("gm").reset_index()
TEAMVOL = _tv.merge(_tg, on=["team", "season"])
TEAMVOL["tgt_pg"] = TEAMVOL.tgt / TEAMVOL.gm
TEAMVOL["tch_pg"] = TEAMVOL.tch / TEAMVOL.gm


# ------------------------------------------------------------------ estimators
def wquantile(x, w, p):
    """Right-continuous weighted inverse CDF."""
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w) / w.sum()
    j = np.searchsorted(c, p, side="left")
    return float(x[min(j, len(x) - 1)])


def wtrimmed(x, w, alpha=0.20):
    """Mean after dropping alpha of the WEIGHT from each tail."""
    o = np.argsort(x)
    x, w = x[o], w[o].astype(float)
    W = w.sum()
    lo, hi = alpha * W, (1 - alpha) * W
    cum = np.concatenate([[0.0], np.cumsum(w)])
    keep = np.minimum(cum[1:], hi) - np.maximum(cum[:-1], lo)
    keep = np.clip(keep, 0, None)
    if keep.sum() <= 0:
        return float(np.average(x, weights=w))
    return float(np.average(x, weights=keep))


def whuber(x, w, c=1.345, iters=100, tol=1e-10):
    mu = float(np.average(x, weights=w))
    for _ in range(iters):
        s = wquantile(np.abs(x - mu), w, 0.5) / 0.6745
        if not np.isfinite(s) or s <= 1e-12:
            return float(np.average(x, weights=w))
        r = (x - mu) / s
        psi = np.clip(r, -c, c)
        new = mu + s * np.average(psi, weights=w)
        if abs(new - mu) < tol:
            return float(new)
        mu = new
    return float(mu)


def summaries(seasons, ybars, Gs, games_y, games_s, weighted=True):
    """Return dict of arm -> mu_hat for one player-year.  seasons/ybars/Gs are the
    per-season arrays (already filtered to seasons < Y); games_y/games_s are the
    flat game-level PPR values and their season labels."""
    S = seasons.max()
    w = 2.0 ** (-(S - seasons) / HL)
    mu1 = float((w * ybars).sum() / w.sum())
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())

    gw = np.ones(len(games_y))
    if weighted:
        wmap = dict(zip(seasons, w))
        gmap = dict(zip(seasons, Gs))
        gw = np.array([wmap[s] / gmap[s] for s in games_s], dtype=float)

    out = {"a1_mean": mu1,
           "a2_median": wquantile(games_y, gw, 0.50),
           "a3_trim20": wtrimmed(games_y, gw, 0.20),
           "a4_huber": whuber(games_y, gw),
           "a5_p60": wquantile(games_y, gw, 0.60)}

    q = Gs >= MIN_G_STABLE
    if q.any():
        sq, yq = seasons[q], ybars[q]
        wq = 2.0 ** (-(sq.max() - sq) / HL)
        out["a6_stable"] = float((wq * yq).sum() / wq.sum())
        out["a6_neff_alt"] = float(wq.sum() ** 2 / (wq ** 2).sum())
    else:
        out["a6_stable"] = mu1
        out["a6_neff_alt"] = n_eff
    out["_n_eff"] = n_eff
    out["_n_qual"] = int(q.sum())
    out["_G_last"] = float(Gs[np.argmax(seasons)])
    # slope: latest minus second-latest season mean, per season gap
    if len(seasons) >= 2:
        o = np.argsort(seasons)
        out["_d"] = float(ybars[o][-1] - ybars[o][-2])
    else:
        out["_d"] = 0.0
    return out


ARMS = ["a1_mean", "a2_median", "a3_trim20", "a4_huber", "a5_p60", "a6_stable",
        "a7_slope", "a8_usage"]
LABEL = {"a1_mean": "1 recency-wtd mean (incumbent)", "a2_median": "2 median (game)",
         "a3_trim20": "3 trimmed mean 20% (game)", "a4_huber": "4 Huber c=1.345 (game)",
         "a5_p60": "5 p60 (game)", "a6_stable": "6 seasons with G>=12 only",
         "a7_slope": "7 slope-adjusted level", "a8_usage": "8 usage-implied mean"}


def dm(df, base, cand, target="ppg"):
    dsq = (df[target] - df[base]) ** 2 - (df[target] - df[cand]) ** 2
    dy = dsq.groupby(df.year).mean()
    t = float(dy.mean() / (dy.std(ddof=1) / np.sqrt(len(dy))))
    return t, float(2 * stats.t.sf(abs(t), df=len(dy) - 1)), dy


def mde(sd, n=10, alpha=0.05):
    return (stats.t.ppf(1 - alpha / 2, n - 1) + stats.t.ppf(0.80, n - 1)) * sd / np.sqrt(n)


def bh(pvals, q=0.10):
    p = np.asarray(pvals, dtype=float)
    o = np.argsort(p)
    m = len(p)
    thr = q * (np.arange(1, m + 1)) / m
    passed = p[o] <= thr
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(m, bool)
    if k:
        rej[o[:k]] = True
    return rej, thr[np.argsort(o)]


# ------------------------------------------------------------------ main loop
def run(pos, panel_file, weighted=True):
    inc = INC[pos]
    panel = pd.read_csv(ROOT / panel_file).rename(columns={"pid": "gsis_id"})

    # sigma2(tier) source: full positional population, gated (identical to script 41)
    pop = inc[inc.position == pos]
    ps = (pop.groupby(["player_id", "season"])
          .agg(G=("fantasy_points_ppr", "size"), rel=(RELCOL[pos], "mean")).reset_index())
    gated = ps[ps.rel >= GATE[pos]][["player_id", "season"]]
    g2 = pop.merge(gated, on=["player_id", "season"])
    mu_ps = g2.groupby(["player_id", "season"]).fantasy_points_ppr.transform("mean")
    g2 = g2.assign(e2=(g2.fantasy_points_ppr - mu_ps) ** 2)
    g2 = g2.merge(meta.rename(columns={"gsis_id": "player_id"}), on="player_id",
                  how="left").dropna(subset=["rookie_season"])
    g2["exp"] = g2.season - g2.rookie_season
    g2["tier"] = np.select([g2.exp == 0, g2.exp == 1], ["rookie", "soph"], "vet")

    # per-player game lists (panel players only)
    sub = inc[inc.player_id.isin(panel.gsis_id.unique())]
    GL = {p: (g.season.values, g.fantasy_points_ppr.values)
          for p, g in sub.groupby("player_id")}
    SM = {p: g for p, g in
          sub.groupby(["player_id", "season"])
          .agg(ybar=("fantasy_points_ppr", "mean"),
               G=("fantasy_points_ppr", "size")).reset_index().groupby("player_id")}

    # usage covariate: player's share of team volume x team volume/game, per season
    vcol = VOLCOL[pos]
    pv = (sub.groupby(["player_id", "season", "team"])[vcol].sum().rename("pv")
          .reset_index())
    pv = pv.merge(TEAMVOL[["team", "season", "tgt", "tch", "tgt_pg", "tch_pg"]],
                  on=["team", "season"], how="left")
    tot_c, pg_c = ("tgt", "tgt_pg") if pos == "WR" else ("tch", "tch_pg")
    pv["x"] = (pv.pv / pv[tot_c]) * pv[pg_c]
    pv = pv.groupby(["player_id", "season"], as_index=False).x.sum()
    XMAP = {(p, s): x for p, s, x in pv.itertuples(index=False)}

    rows = []
    for _, r in panel.iterrows():
        g, Y = r.gsis_id, r.year
        rec = {"gsis_id": g, "year": Y}
        if g not in GL:
            rec.update({a: np.nan for a in ARMS[:6]})
            rec.update({"_n_eff": 0.0, "_d": 0.0, "_n_qual": 0, "_G_last": 0.0,
                        "a6_neff_alt": 0.0, "x_usage": np.nan})
            rows.append(rec)
            continue
        s_all, y_all = GL[g]
        m = s_all < Y
        sm = SM[g]
        sm = sm[sm.season < Y]
        if not m.any() or len(sm) == 0:
            rec.update({a: np.nan for a in ARMS[:6]})
            rec.update({"_n_eff": 0.0, "_d": 0.0, "_n_qual": 0, "_G_last": 0.0,
                        "a6_neff_alt": 0.0, "x_usage": np.nan})
            rows.append(rec)
            continue
        rec.update(summaries(sm.season.values, sm.ybar.values, sm.G.values,
                             y_all[m], s_all[m], weighted=weighted))
        rec["x_usage"] = XMAP.get((g, int(sm.season.max())), np.nan)
        rows.append(rec)
    S = pd.DataFrame(rows)
    panel = pd.concat([panel.reset_index(drop=True), S.drop(columns=["gsis_id", "year"])],
                      axis=1)

    preds = []
    for Y in YEARS:
        tr = panel[(panel.year != Y) & panel.in_fit].copy()
        ev = panel[panel.year == Y].copy()
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso.fit(np.log(tr.adp.values), tr.ppg.values)
        tr["m_hat"] = iso.predict(np.log(tr.adp.values))
        tr["r"] = tr.ppg - tr.m_hat
        ev["m_hat"] = iso.predict(np.log(ev.adp.values))
        tau2 = tr.groupby("tier").r.var(ddof=1)
        ev["tau2"] = ev.tier.map(tau2).fillna(tr.r.var(ddof=1))
        sig2 = g2[g2.season != Y].groupby("tier").e2.mean()
        ev["sig2"] = ev.tier.map(sig2)

        nop = ev._n_eff == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            V = ev.sig2 / ev._n_eff
        B = np.where(nop, 1.0, V / (V + ev.tau2))
        ev["B"] = B

        # --- arm 7: within-fold OLS ppg ~ 1 + mu_1 + d, fitted on training rows
        trf = tr[tr._n_eff > 0]
        X = np.c_[np.ones(len(trf)), trf.a1_mean.values, trf._d.values]
        beta7 = np.linalg.lstsq(X, trf.ppg.values, rcond=None)[0]
        ev["a7_slope"] = (beta7[0] + beta7[1] * ev.a1_mean.fillna(0)
                          + beta7[2] * ev._d.fillna(0))

        # --- arm 8: within-fold OLS ppg ~ 1 + x
        xmu = float(trf.x_usage.mean())
        X8 = np.c_[np.ones(len(trf)), trf.x_usage.fillna(xmu).values]
        beta8 = np.linalg.lstsq(X8, trf.ppg.values, rcond=None)[0]
        ev["a8_usage"] = beta8[0] + beta8[1] * ev.x_usage.fillna(xmu)

        for a in ARMS:
            ev[f"th_{a}"] = np.where(nop, ev.m_hat,
                                     (1 - B) * ev[a].fillna(0) + B * ev.m_hat)
        # sensitivity: arm 6 with n_eff recomputed on the qualifying subset
        with np.errstate(divide="ignore", invalid="ignore"):
            V6 = ev.sig2 / ev.a6_neff_alt
        B6 = np.where(nop, 1.0, V6 / (V6 + ev.tau2))
        ev["th_a6_neffalt"] = np.where(nop, ev.m_hat,
                                       (1 - B6) * ev.a6_stable.fillna(0) + B6 * ev.m_hat)
        ev["beta7_d"] = beta7[2]
        preds.append(ev)

    pred = pd.concat(preds, ignore_index=True)
    pred["pos"] = pos
    return pred


def score(fit, panel_label, pos, arms=ARMS[1:]):
    base = "th_a1_mean"
    rb_ = float(np.sqrt(((fit.ppg - fit[base]) ** 2).mean()))
    rows = []
    for a in arms:
        c = f"th_{a}"
        rc = float(np.sqrt(((fit.ppg - fit[c]) ** 2).mean()))
        t, p, dy = dm(fit, base, c)
        sd = float(dy.std(ddof=1))
        rows.append(dict(pos=pos, panel=panel_label, arm=a, label=LABEL.get(a, a),
                         n=len(fit), rmse_incumbent=rb_, rmse_arm=rc,
                         d_rmse=rc - rb_, folds_improved=int((dy > 0).sum()),
                         mean_gain=float(dy.mean()), sd_folds=sd,
                         dm_t=t, dm_p=p, mde80=mde(sd),
                         obs_over_mde=float(dy.mean() / mde(sd)),
                         spearman_arm=float(fit.groupby("year").apply(
                             lambda g: stats.spearmanr(g[c], g.ppg).statistic).mean()),
                         spearman_incumbent=float(fit.groupby("year").apply(
                             lambda g: stats.spearmanr(g[base], g.ppg).statistic).mean())))
    return pd.DataFrame(rows)


# ============================================================== run
if __name__ == "__main__":
    PANEL = {"WR": "results/market_prior_wr_deep.csv",
             "RB": "results/market_prior_rb_deep.csv"}
    allpred, allscore, sens = [], [], []
    for pos in ["WR", "RB"]:
        pr = run(pos, PANEL[pos], weighted=True)
        allpred.append(pr)
        fit = pr[pr.in_fit]
        allscore.append(score(fit, "wide (all board rows)", pos))
        allscore.append(score(fit[fit.adp_rank <= 30], "top-30 stratum (§7-comparable)", pos))
        # declared sensitivity 1: arm 6 with n_eff recomputed
        t, p, dy = dm(fit, "th_a1_mean", "th_a6_neffalt")
        sens.append(dict(pos=pos, which="a6 with n_eff recomputed on qualifying seasons",
                         rmse_incumbent=float(np.sqrt(((fit.ppg - fit.th_a1_mean) ** 2).mean())),
                         rmse_arm=float(np.sqrt(((fit.ppg - fit.th_a6_neffalt) ** 2).mean())),
                         mean_gain=float(dy.mean()), dm_t=t, dm_p=p,
                         mde80=mde(float(dy.std(ddof=1)))))
        # declared sensitivity 2: unweighted pooled games for arms 2-5
        pu = run(pos, PANEL[pos], weighted=False)
        fu = pu[pu.in_fit]
        su = score(fu, "wide, UNWEIGHTED pooled games (sensitivity)", pos,
                   arms=["a2_median", "a3_trim20", "a4_huber", "a5_p60"])
        sens += su.to_dict("records")

    SC = pd.concat(allscore, ignore_index=True)
    PR = pd.concat(allpred, ignore_index=True)

    # BH within the declared family of 7 challengers, per position, primary panel
    SC["bh_reject"] = False
    SC["bh_thresh"] = np.nan
    for pos in ["WR", "RB"]:
        m = (SC.pos == pos) & (SC.panel == "wide (all board rows)")
        rej, thr = bh(SC.loc[m, "dm_p"].values, q=0.10)
        SC.loc[m, "bh_reject"] = rej
        SC.loc[m, "bh_thresh"] = thr
    # pooled 14-test BH, reported as robustness
    m = SC.panel == "wide (all board rows)"
    rej14, thr14 = bh(SC.loc[m, "dm_p"].values, q=0.10)
    SC["bh_reject_pooled14"] = False
    SC["bh_thresh_pooled14"] = np.nan
    SC.loc[m, "bh_reject_pooled14"] = rej14
    SC.loc[m, "bh_thresh_pooled14"] = thr14

    # ---------------------------------------------------- temporal holdout
    hrows = []
    for pos in ["WR", "RB"]:
        pr = PR[(PR.pos == pos) & PR.in_fit]
        ho = pr[pr.year >= 2022]
        base = float(np.sqrt(((ho.ppg - ho.th_a1_mean) ** 2).mean()))
        for a in ARMS[1:]:
            r = float(np.sqrt(((ho.ppg - ho[f"th_{a}"]) ** 2).mean()))
            dsq = ((ho.ppg - ho.th_a1_mean) ** 2 - (ho.ppg - ho[f"th_{a}"]) ** 2)
            dy = dsq.groupby(ho.year).mean()
            t = float(dy.mean() / (dy.std(ddof=1) / np.sqrt(len(dy))))
            hrows.append(dict(pos=pos, arm=a, label=LABEL[a], n=len(ho),
                              rmse_incumbent=base, rmse_arm=r, d_rmse=r - base,
                              mean_gain=float(dy.mean()), dm_t=t,
                              dm_p=float(2 * stats.t.sf(abs(t), df=len(dy) - 1)),
                              survives=bool(r < base)))
    HO = pd.DataFrame(hrows)

    SC.to_csv(ROOT / "results/sectionS_bakeoff.csv", index=False)
    HO.to_csv(ROOT / "results/sectionS_holdout.csv", index=False)
    pd.DataFrame(sens).to_csv(ROOT / "results/sectionS_sensitivity.csv", index=False)
    keep = (["gsis_id", "name", "year", "pos", "adp", "adp_rank", "tier", "ppg",
             "m_hat", "B", "_n_eff", "_n_qual", "_G_last", "_d", "x_usage"]
            + ARMS + [f"th_{a}" for a in ARMS] + ["th_a6_neffalt"])
    PR[[c for c in keep if c in PR.columns]].to_csv(
        ROOT / "results/sectionS_predictions.csv", index=False)

    pd.set_option("display.width", 250)
    for pos in ["WR", "RB"]:
        for pan in SC.panel.unique():
            s = SC[(SC.pos == pos) & (SC.panel == pan)]
            if not len(s):
                continue
            print(f"\n=== {pos} | {pan} | n={s.n.iat[0]} | "
                  f"incumbent RMSE {s.rmse_incumbent.iat[0]:.4f} ===")
            print(s[["label", "rmse_arm", "d_rmse", "mean_gain", "sd_folds", "dm_t",
                     "dm_p", "mde80", "obs_over_mde", "folds_improved", "bh_reject"]]
                  .round(4).to_string(index=False))
    print("\n=== temporal holdout 2022-24 ===")
    print(HO.round(4).to_string(index=False))
    print("\n=== sensitivities ===")
    print(pd.DataFrame(sens).round(4).to_string(index=False))

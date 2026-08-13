"""§P4 — LOSO re-run on the WIDE board panel (EDA_PLAN6.md §P4).

The RB LOSO (§G6) failed at p = .464 on 286 in-fit top-30 rows and was diagnosed as
underpowered (mean fold gain 0.488 PPG^2, across-fold SD 2.015).  §P widens the panel to
the full historical boards (573 in-fit RB rows, 638 WR), roughly double.  §P4 pre-registers
a RE-RUN OF THE SAME ARMS on the wider panel -- no new arms, no hunting:

    target PPG    (i)  m_{-Y}(ADP)                 market-only baseline
                  (ii) blind posterior theta*      data arm
    target ppsw   (i-SV) ADP-only isotonic refit on points/scheduled week
                  (iii)  SV = theta* x E[G]/M

Everything is refit inside each fold: m_{-Y}, tau^2_{-Y}(tier), sigma^2_{-Y}(tier), and
each player's mu_hat/n_eff from weekly_raw seasons STRICTLY BEFORE Y (recency half-life
h = 1).  Adoption rule unchanged: DM t-test on yearly mean squared-error differentials,
t(9 df), p < 0.10 AND pooled RMSE improves.

POWER.  Reported for every arm with one explicit formula, applied identically to the
§G6 top-30 panel and to the wide panel so the two are comparable:
    MDE(80% power, two-sided alpha = 0.10, the adoption bar)
        = (t_{0.95,9} + t_{0.80,9}) * SD_folds / sqrt(10).
The §G6 top-30 RB run is recomputed here under the same formula for the comparison.

WR is run alongside RB with the identical estimator.  WR adoption already stands from §7;
the wide-panel WR run is confirmatory/descriptive and is labelled as such.

Outputs: results/loso_scorecard_deep.csv, results/loso_predictions_deep.csv,
         results/sectionP_loso_power.csv
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.isotonic import IsotonicRegression

warnings.simplefilter("ignore")
ROOT = Path("/Users/thomasmcnamee/NFL")
YEARS = list(range(2015, 2025))
GATE = {"WR": 3.0, "RB": 4.0}          # §3 / §G2c relevance gates, unchanged
AV = ["age", "G1", "G2", "miss1", "miss2"]

# ------------------------------------------------------------------ weekly data
COLS = ["player_id", "position", "season", "week", "season_type",
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
meta = pd.read_csv(ROOT / "data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "birth_date", "rookie_season"])
tot = (wk.groupby(["player_id", "season"]).fantasy_points_ppr.sum()
       .rename("total_pts").reset_index())


def dm(df, target, base, cand):
    dsq = (df[target] - df[base]) ** 2 - (df[target] - df[cand]) ** 2
    dy = dsq.groupby(df.year).mean()
    t = float(dy.mean() / (dy.std(ddof=1) / np.sqrt(len(dy))))
    return t, float(2 * stats.t.sf(abs(t), df=len(dy) - 1)), dy


def mde(sd, n=10, alpha=0.10):
    return (stats.t.ppf(1 - alpha / 2, n - 1) + stats.t.ppf(0.80, n - 1)) * sd / np.sqrt(n)


def run(pos, panel_file, label):
    print("\n" + "=" * 76 + f"\n{label}\n" + "=" * 76)
    inc = INC[pos]
    panel = pd.read_csv(ROOT / panel_file)
    panel = panel.rename(columns={"pid": "gsis_id"}) if "pid" in panel else panel
    panel = panel.merge(meta[["gsis_id", "birth_date"]], on="gsis_id", how="left")
    panel["age"] = ((pd.to_datetime(panel.year.astype(str) + "-09-01")
                     - pd.to_datetime(panel.birth_date, errors="coerce")).dt.days / 365.25)
    panel["age"] = panel.age.fillna(panel.age.median())
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

    # mu_hat source: all seasons of each panel player's included games
    sm_all = (inc[inc.player_id.isin(panel.gsis_id.unique())]
              .groupby(["player_id", "season"])
              .agg(ybar=("fantasy_points_ppr", "mean")).reset_index())
    sm_idx = {p: g[["season", "ybar"]].values for p, g in sm_all.groupby("player_id")}

    # sigma^2(tier) source: full position population, gated
    pop = inc[inc.position == pos]
    ps = (pop.groupby(["player_id", "season"])
          .agg(G=("fantasy_points_ppr", "size"),
               rel=(RELCOL[pos], "mean")).reset_index())
    gated = ps[ps.rel >= GATE[pos]][["player_id", "season"]]
    g2 = pop.merge(gated, on=["player_id", "season"])
    mu_ps = g2.groupby(["player_id", "season"]).fantasy_points_ppr.transform("mean")
    g2 = g2.assign(e2=(g2.fantasy_points_ppr - mu_ps) ** 2)
    g2 = g2.merge(meta.rename(columns={"gsis_id": "player_id"}),
                  on="player_id", how="left").dropna(subset=["rookie_season"])
    g2["exp"] = g2.season - g2.rookie_season
    g2["tier"] = np.select([g2.exp == 0, g2.exp == 1], ["rookie", "soph"], "vet")

    def mu_neff_before(g, Y):
        a = sm_idx.get(g)
        if a is None:
            return np.nan, 0.0
        a = a[a[:, 0] < Y]
        if len(a) == 0:
            return np.nan, 0.0
        w = 2.0 ** (-(a[:, 0].max() - a[:, 0]) / 1.0)
        return float((w * a[:, 1]).sum() / w.sum()), float(w.sum() ** 2 / (w ** 2).sum())

    preds = []
    for Y in YEARS:
        tr = panel[(panel.year != Y) & panel.in_fit].copy()
        ev = panel[panel.year == Y].copy()
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso.fit(np.log(tr.adp.values), tr.ppg.values)
        tr["r"] = tr.ppg - iso.predict(np.log(tr.adp.values))
        ev["m_hat"] = iso.predict(np.log(ev.adp.values))
        tau2 = tr.groupby("tier").r.var(ddof=1)
        ev["tau2"] = ev.tier.map(tau2).fillna(tr.r.var(ddof=1))
        sig2 = g2[g2.season != Y].groupby("tier").e2.mean()
        ev["sig2"] = ev.tier.map(sig2)
        mn = [mu_neff_before(g, Y) for g in ev.gsis_id]
        ev["mu_hat"] = [t[0] for t in mn]
        ev["n_eff"] = [t[1] for t in mn]
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
        ev["p_avail"] = gA.predict(sm.add_constant(ev[AV],
                                                   has_constant="add")[Xa.columns])
        isoB = IsotonicRegression(increasing=False, out_of_bounds="clip")
        isoB.fit(np.log(trA.adp.values), trA.ppsw.values)
        ev["m_hat_ppsw"] = isoB.predict(np.log(ev.adp.values))
        ev["SV"] = ev.theta_star * ev.p_avail
        preds.append(ev)

    pred = pd.concat(preds, ignore_index=True)
    pred["pos"] = pos
    fit = pred[pred.in_fit]
    print(f"eval rows {len(pred)}, in_fit {len(fit)}, "
          f"B==1 (no prior data) {(fit.B == 1).sum()}")

    rows = []
    for tgt, base, cand, nm in [("ppg", "m_hat", "theta_star", "(ii) blind theta*"),
                                ("ppsw", "m_hat_ppsw", "SV", "(iii) SV")]:
        rb_ = float(np.sqrt(((fit[tgt] - fit[base]) ** 2).mean()))
        rc = float(np.sqrt(((fit[tgt] - fit[cand]) ** 2).mean()))
        t, p, dy = dm(fit, tgt, base, cand)
        sd = float(dy.std(ddof=1))
        m80 = mde(sd)
        rho_b = fit.groupby("year").apply(
            lambda g: stats.spearmanr(g[base], g[tgt]).statistic, include_groups=False).mean()
        rho_c = fit.groupby("year").apply(
            lambda g: stats.spearmanr(g[cand], g[tgt]).statistic, include_groups=False).mean()
        print(f"\n  target {tgt}: baseline {base} RMSE {rb_:.4f} (Spearman {rho_b:.4f})")
        print(f"                {nm} RMSE {rc:.4f} (Spearman {rho_c:.4f})")
        print(f"    yearly loss diff: {dy.round(3).to_dict()}")
        print(f"    folds improved {int((dy>0).sum())}/10, mean {dy.mean():+.3f}, "
              f"SD {sd:.3f}")
        print(f"    DM t({len(dy)-1}) = {t:+.3f}, p = {p:.4f};  "
              f"MDE(80%, a=.10) = {m80:.3f} PPG^2  -> observed/MDE = {dy.mean()/m80:.2f}")
        adopt = bool(p < 0.10 and rc < rb_)
        print(f"    ADOPTION: {'ADOPTED' if adopt else 'NOT ADOPTED'}")
        rows.append(dict(pos=pos, panel=label, target=tgt, arm=nm,
                         rmse_base=rb_, rmse_arm=rc, mean_spearman_base=rho_b,
                         mean_spearman_arm=rho_c, folds_improved=int((dy > 0).sum()),
                         mean_gain=float(dy.mean()), sd_folds=sd, dm_t=t, dm_p=p,
                         mde80=m80, n=len(fit), adopted=adopt))
    return pred, pd.DataFrame(rows)


out_p, out_s = [], []
for pos, f, lab in [("RB", "results/market_prior_rb_deep.csv", "RB wide panel (all board rows)"),
                    ("WR", "results/market_prior_wr_deep.csv", "WR wide panel (all board rows)")]:
    p_, s_ = run(pos, f, lab)
    out_p.append(p_)
    out_s.append(s_)

# ------------------------------------------ restate §G6 top-30 power, same formula
print("\n" + "=" * 76 + "\n§G6 top-30 RB run, MDE recomputed under the SAME formula\n"
      + "=" * 76)
old = pd.read_csv(ROOT / "results/loso_predictions_rb.csv")
of = old[old.in_fit]
extra = []
for tgt, base, cand, nm in [("ppg", "m_hat", "theta_star", "(ii) blind theta*"),
                            ("ppsw", "m_hat_ppsw", "SV", "(iii) SV")]:
    t, p, dy = dm(of, tgt, base, cand)
    sd = float(dy.std(ddof=1))
    print(f"  {tgt} {nm}: mean {dy.mean():+.3f}, SD {sd:.3f}, t {t:+.3f}, p {p:.4f}, "
          f"MDE {mde(sd):.3f}, observed/MDE {dy.mean()/mde(sd):.2f}")
    extra.append(dict(pos="RB", panel="RB top-30 panel (§G6, restated)", target=tgt,
                      arm=nm, rmse_base=float(np.sqrt(((of[tgt]-of[base])**2).mean())),
                      rmse_arm=float(np.sqrt(((of[tgt]-of[cand])**2).mean())),
                      mean_spearman_base=np.nan, mean_spearman_arm=np.nan,
                      folds_improved=int((dy > 0).sum()), mean_gain=float(dy.mean()),
                      sd_folds=sd, dm_t=t, dm_p=p, mde80=mde(sd), n=len(of),
                      adopted=bool(p < 0.10)))

sc = pd.concat(out_s + [pd.DataFrame(extra)], ignore_index=True)
sc.to_csv(ROOT / "results/loso_scorecard_deep.csv", index=False)
pd.concat(out_p, ignore_index=True).drop(columns=["birth_date"]).to_csv(
    ROOT / "results/loso_predictions_deep.csv", index=False)
print("\n" + sc.round(4).to_string(index=False))
print("\nwrote loso_scorecard_deep.csv, loso_predictions_deep.csv")

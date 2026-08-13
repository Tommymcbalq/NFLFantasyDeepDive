"""AUDIT 3+4 -- validation protocol, selection noise, stability, era drift,
and the 2023 anomaly.

Nothing here touches 2024-25 except to REPORT the already-known season-level
scores of the existing spec (no selection is done on it).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from audit_common import (load, scores, logloss, brier, paired_boot, devig,
                          TRAIN, VALID, HOLDOUT, SPEC_B, RESULTS, calibration_table)

OUT = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.append(s)

CV_TARGETS = list(range(2017, 2024))


def zfit(tr, te, cols):
    mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1.0)
    return (tr[cols] - mu) / sd, (te[cols] - mu) / sd


def logit_pred(tr, te, cols):
    Xtr, Xte = zfit(tr, te, cols)
    f = sm.Logit(tr.home_win.values, sm.add_constant(Xtr.values)).fit(disp=0)
    return f.predict(sm.add_constant(Xte.values, has_constant="add")), f


def main():
    d = load().sort_values(["season", "week"]).reset_index(drop=True)
    d["blk"] = d.season * 100 + d.week
    d["p_mkt"] = devig(d.home_moneyline, d.away_moneyline)
    d.loc[d.p_mkt.isna(), "p_mkt"] = norm.cdf(d.loc[d.p_mkt.isna(), "spread_home"] / 13.0)

    # ============ 1. SELECTION NOISE: how big must a gain be? ============
    p("=" * 92)
    p("SELECTION NOISE -- the noise floor for log-loss differences")
    p("=" * 92)
    rng = np.random.default_rng(11)
    tr = d[d.season.isin(TRAIN)].copy()
    va = d[d.season.isin(VALID)].copy()

    for nnoise, label in [(1, "1 pure-noise feature"), (5, "5 pure-noise features")]:
        gains_v, gains_cv = [], []
        for rep in range(400):
            dd = d.copy()
            ncols = []
            for j in range(nnoise):
                c = f"noise{j}"
                dd[c] = rng.standard_normal(len(dd))
                ncols.append(c)
            # validation-season evaluation
            p0, _ = logit_pred(dd[dd.season.isin(TRAIN)], dd[dd.season.isin(VALID)], SPEC_B)
            p1, _ = logit_pred(dd[dd.season.isin(TRAIN)], dd[dd.season.isin(VALID)],
                               SPEC_B + ncols)
            gains_v.append(logloss(va.home_win.values, p0) - logloss(va.home_win.values, p1))
            if rep < 120:   # rolling CV is slower
                ys, a, b = [], [], []
                for s in CV_TARGETS:
                    trf, tef = dd[dd.season < s], dd[dd.season == s]
                    q0, _ = logit_pred(trf, tef, SPEC_B)
                    q1, _ = logit_pred(trf, tef, SPEC_B + ncols)
                    ys.append(tef.home_win.values); a.append(q0); b.append(q1)
                ys = np.concatenate(ys)
                gains_cv.append(logloss(ys, np.concatenate(a)) - logloss(ys, np.concatenate(b)))
        gv, gc = np.array(gains_v), np.array(gains_cv)
        p(f"\n{label}: log-loss 'improvement' from features that are known to be worthless")
        p(f"  single-season VALID (n=272):   mean {gv.mean():+.4f}  sd {gv.std():.4f}  "
          f"95th pct {np.percentile(gv,95):+.4f}  max {gv.max():+.4f}")
        p(f"  rolling CV 2017-23 (n=1832):   mean {gc.mean():+.4f}  sd {gc.std():.4f}  "
          f"95th pct {np.percentile(gc,95):+.4f}  max {gc.max():+.4f}")
        p(f"  P(noise feature 'wins' on VALID) = {(gv>0).mean():.2f}; "
          f"on rolling CV = {(gc>0).mean():.2f}")

    # sampling sd of a log-loss difference at fixed n, from a real pair
    p("\nsampling sd of a PAIRED log-loss difference (blocked bootstrap, real model pair")
    p("spec B vs spec B + pace_diff), at three evaluation sizes:")
    for label, seasons in [("2023 only (272)", VALID), ("2024-25 (543)", HOLDOUT),
                           ("rolling CV 2017-23 (1832)", CV_TARGETS)]:
        ys, a, b, blks = [], [], [], []
        for s in seasons:
            trf, tef = d[d.season < s], d[d.season == s]
            if len(trf) == 0:
                continue
            q0, _ = logit_pred(trf, tef, SPEC_B)
            q1, _ = logit_pred(trf, tef, SPEC_B + ["pace_diff"])
            ys.append(tef.home_win.values); a.append(q0); b.append(q1)
            blks.append(tef.blk.values)
        ys, a, b, blks = (np.concatenate(x) for x in (ys, a, b, blks))
        pt, lo, hi, dist = paired_boot(ys, b, a, block=blks, B=3000)
        p(f"  {label:28s} diff {pt:+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]  "
          f"half-width {(hi-lo)/2:.4f}")

    # ============ 2. SEASON-BY-SEASON: is 2023 anomalous? ============
    p("\n" + "=" * 92)
    p("SEASON-BY-SEASON DIAGNOSIS (the 2023 question)")
    p("=" * 92)
    rows = []
    for s in range(2015, 2026):
        te = d[d.season == s]
        trf = d[d.season < s]
        pr, _ = logit_pred(trf, te, SPEC_B)
        y = te.home_win.values
        base = trf.home_win.mean()
        rows.append({
            "season": s, "n": len(te),
            "model_ll": logloss(y, pr),
            "market_ll": logloss(y, te.p_mkt.values),
            "baserate_ll": logloss(y, np.full(len(te), base)),
            "entropy": logloss(y, np.full(len(te), y.mean())),
            "home_wr": y.mean(),
            "mkt_fav_wr": float((( te.p_mkt > .5) == (y == 1)).mean()),
            "mean_abs_spread": te.spread_home.abs().mean(),
            "mean_abs_margin": te.margin.abs().mean(),
            "close_games_pct": float((te.margin.abs() <= 7).mean()),
            "mean_pmkt_conf": float(np.abs(te.p_mkt - .5).mean()),
        })
    ss = pd.DataFrame(rows)
    ss["model_minus_mkt"] = ss.model_ll - ss.market_ll
    ss["mkt_minus_entropy"] = ss.market_ll - ss.entropy
    p(ss.round(4).to_string(index=False))
    ss.to_csv(os.path.join(RESULTS, "audit_season_scores.csv"), index=False)

    p("\nRead: 'entropy' is the log loss of the season's OWN home-win base rate --")
    p("the irreducible floor for a model that knows nothing but the season outcome mix.")
    p("'model_minus_mkt' is the model's skill deficit; if 2023 is anomalous FOR THE MODEL,")
    p("that column should spike in 2023. If 2023 is just a high-entropy season, market_ll")
    p("rises too and the deficit column does not move.")
    p(f"\nmodel_minus_mkt: mean {ss.model_minus_mkt.mean():.4f}, "
      f"sd {ss.model_minus_mkt.std():.4f}, 2023 = {ss.loc[ss.season==2023,'model_minus_mkt'].iat[0]:.4f} "
      f"(z = {(ss.loc[ss.season==2023,'model_minus_mkt'].iat[0]-ss.model_minus_mkt.mean())/ss.model_minus_mkt.std():+.2f})")
    p(f"market_ll:       mean {ss.market_ll.mean():.4f}, sd {ss.market_ll.std():.4f}, "
      f"2023 = {ss.loc[ss.season==2023,'market_ll'].iat[0]:.4f} "
      f"(z = {(ss.loc[ss.season==2023,'market_ll'].iat[0]-ss.market_ll.mean())/ss.market_ll.std():+.2f})")
    p(f"correlation across seasons: model_ll vs market_ll = "
      f"{ss.model_ll.corr(ss.market_ll):.3f}")

    # decompose 2023 model deficit: which weeks / which games
    p("\n2023 within-season breakdown (model spec B, trained on <2023):")
    te23 = d[d.season == 2023].copy()
    pr23, _ = logit_pred(d[d.season < 2023], te23, SPEC_B)
    te23["p"] = pr23
    te23["ll"] = -(te23.home_win * np.log(te23.p) + (1 - te23.home_win) * np.log(1 - te23.p))
    te23["ll_mkt"] = -(te23.home_win * np.log(te23.p_mkt) +
                       (1 - te23.home_win) * np.log(1 - te23.p_mkt))
    q = te23.groupby(pd.cut(te23.week, [0, 4, 8, 12, 17, 22])).agg(
        n=("ll", "size"), model=("ll", "mean"), market=("ll_mkt", "mean"))
    p(q.round(4).to_string())
    p("\nworst 10 games for the model in 2023 (and what the market said):")
    w = te23.nlargest(10, "ll")[["game_id", "home_team", "away_team", "week", "p",
                                 "p_mkt", "home_win", "margin", "ll", "ll_mkt"]]
    p(w.round(3).to_string(index=False))
    p(f"\n2023 model log loss {te23.ll.mean():.4f}; excluding the 10 worst games "
      f"{te23.nsmallest(len(te23)-10,'ll').ll.mean():.4f}")
    p(f"2023 market log loss {te23.ll_mkt.mean():.4f}; same 10 games excluded "
      f"{te23[~te23.game_id.isin(w.game_id)].ll_mkt.mean():.4f}")

    p("\n2023 calibration of the model:")
    p(calibration_table(te23.home_win, te23.p).round(3).to_string(index=False))

    # ============ 3. COEFFICIENT STABILITY / ERA DRIFT ============
    p("\n" + "=" * 92)
    p("COEFFICIENT STABILITY AND ERA DRIFT")
    p("=" * 92)
    mu, sd = d.loc[d.season.isin(TRAIN), SPEC_B].mean(), d.loc[d.season.isin(TRAIN), SPEC_B].std()
    Z = (d[SPEC_B] - mu) / sd
    crows = []
    for s in range(2014, 2026):
        m = d.season == s
        f = sm.Logit(d.home_win[m].values, sm.add_constant(Z[m].values)).fit(disp=0)
        crows.append({"season": s, "n": int(m.sum()), "const": f.params[0],
                      **{c: f.params[i + 1] for i, c in enumerate(SPEC_B)},
                      **{c + "_se": f.bse[i + 1] for i, c in enumerate(SPEC_B)},
                      "const_se": f.bse[0]})
    cs = pd.DataFrame(crows)
    p(cs.round(3).to_string(index=False))
    cs.to_csv(os.path.join(RESULTS, "audit_coef_by_season.csv"), index=False)

    p("\nheterogeneity test per coefficient (Cochran Q across 12 season fits):")
    from scipy.stats import chi2
    for c in ["const"] + SPEC_B:
        b, se = cs[c].values, cs[c + "_se"].values
        w = 1 / se ** 2
        bbar = (w * b).sum() / w.sum()
        Q = (w * (b - bbar) ** 2).sum()
        pv = 1 - chi2.cdf(Q, len(b) - 1)
        I2 = max(0.0, (Q - (len(b) - 1)) / Q)
        p(f"  {c:20s} pooled {bbar:+.3f}  Q={Q:.1f} (11 df) p={pv:.3f}  I^2={I2:.2f}")

    p("\nHFA drift: home win rate and fitted intercept by season, plus a trend test")
    hfa = d.groupby("season").agg(home_wr=("home_win", "mean"), n=("home_win", "size"),
                                  mean_spread=("spread_home", "mean"),
                                  mean_margin=("margin", "mean"))
    p(hfa.round(4).to_string())
    x = sm.add_constant(hfa.index.values.astype(float) - 2019.5)
    tfit = sm.OLS(hfa.mean_margin.values, x).fit()
    p(f"  mean home margin trend: {tfit.params[1]:+.3f} pts/season "
      f"(se {tfit.bse[1]:.3f}, p={tfit.pvalues[1]:.3f})")
    tfit2 = sm.OLS(hfa.home_wr.values, x).fit()
    p(f"  home win rate trend:    {tfit2.params[1]:+.4f}/season "
      f"(se {tfit2.bse[1]:.4f}, p={tfit2.pvalues[1]:.3f})")
    p("  2020 (no crowds): home margin %.2f vs %.2f in all other seasons"
      % (d[d.season == 2020].margin.mean(), d[d.season != 2020].margin.mean()))

    # does an explicit season_c or 2020 dummy help out of sample?
    p("\ndoes modelling the drift help? rolling-origin CV 2017-23:")
    d2 = d.copy()
    variants = {"spec B": SPEC_B, "spec B + season_c": SPEC_B + ["season_c"],
                "spec B + is_neutral": SPEC_B + ["is_neutral"],
                "spec B + div_game": SPEC_B + ["div_game"]}
    store = {}
    for name, cols in variants.items():
        ys, ps, bl = [], [], []
        for s in CV_TARGETS:
            q, _ = logit_pred(d2[d2.season < s], d2[d2.season == s], cols)
            ys.append(d2[d2.season == s].home_win.values); ps.append(q)
            bl.append(d2[d2.season == s].blk.values)
        ys, ps, bl = (np.concatenate(z) for z in (ys, ps, bl))
        store[name] = (ys, ps, bl)
        p(f"  {name:24s} CV logloss {logloss(ys, ps):.4f}")
    ybase, pbase, blbase = store["spec B"]
    for name in variants:
        if name == "spec B":
            continue
        y, q, bl = store[name]
        pt, lo, hi, _ = paired_boot(y, q, pbase, block=bl, B=3000)
        p(f"    {name:24s} vs spec B: {pt:+.4f} [{lo:+.4f},{hi:+.4f}]")

    # --- training window / recency weighting: is old data stale? ---
    p("\ntraining window and recency weighting (rolling-origin CV 2017-23):")
    def wlogit(tr, te, cols, hl=None, window=None):
        if window is not None:
            tr = tr[tr.season >= tr.season.max() - window + 1]
        Xtr, Xte = zfit(tr, te, cols)
        w = np.ones(len(tr)) if hl is None else \
            0.5 ** ((tr.season.max() - tr.season.values) / hl)
        f = sm.GLM(tr.home_win.values, sm.add_constant(Xtr.values),
                   family=sm.families.Binomial(), freq_weights=w).fit()
        return f.predict(sm.add_constant(Xte.values, has_constant="add"))
    wstore = {}
    for name, kw in [("all seasons, unweighted", {}),
                     ("half-life 6 seasons", {"hl": 6}),
                     ("half-life 3 seasons", {"hl": 3}),
                     ("half-life 1.5 seasons", {"hl": 1.5}),
                     ("last 5 seasons only", {"window": 5}),
                     ("last 3 seasons only", {"window": 3})]:
        ys, ps, bl = [], [], []
        for s in CV_TARGETS:
            q = wlogit(d[d.season < s], d[d.season == s], SPEC_B, **kw)
            ys.append(d[d.season == s].home_win.values); ps.append(np.asarray(q))
            bl.append(d[d.season == s].blk.values)
        ys, ps, bl = (np.concatenate(z) for z in (ys, ps, bl))
        wstore[name] = (ys, ps, bl)
        p(f"  {name:26s} CV logloss {logloss(ys, ps):.4f}")
    yb, pb, blb = wstore["all seasons, unweighted"]
    for name in wstore:
        if name == "all seasons, unweighted":
            continue
        y, q, bl = wstore[name]
        pt, lo, hi, _ = paired_boot(y, q, pb, block=bl, B=3000)
        p(f"    {name:26s} vs unweighted: {pt:+.4f} [{lo:+.4f},{hi:+.4f}]")

    # ============ 4. INFLUENCE ============
    p("\n" + "=" * 92)
    p("INFLUENTIAL OBSERVATIONS (train fit 2014-2022)")
    p("=" * 92)
    trm = d.season.isin(TRAIN)
    f = sm.Logit(d.home_win[trm].values, sm.add_constant(Z[trm].values)).fit(disp=0)
    infl = f.get_influence()
    cook = infl.cooks_distance[0]
    lev = infl.hat_matrix_diag
    p(f"max Cook's D {cook.max():.4f} (4/n = {4/trm.sum():.5f}); "
      f"{(cook > 4/trm.sum()).sum()} obs above 4/n")
    idx = np.argsort(-cook)[:8]
    sub = d[trm].iloc[idx][["game_id", "season", "week", "home_win", "margin",
                            "spread_home"] + SPEC_B].copy()
    sub["cooksD"] = cook[idx]; sub["leverage"] = lev[idx]
    p(sub.round(3).to_string(index=False))
    # refit dropping top 1%
    keep = cook < np.quantile(cook, 0.99)
    f2 = sm.Logit(d.home_win[trm].values[keep],
                  sm.add_constant(Z[trm].values[keep])).fit(disp=0)
    p("\ncoefficients with/without the most influential 1% of train games:")
    for i, c in enumerate(["const"] + SPEC_B):
        p(f"  {c:20s} full {f.params[i]:+.4f}  trimmed {f2.params[i]:+.4f}  "
          f"(shift {f2.params[i]-f.params[i]:+.4f}, se {f.bse[i]:.4f})")

    with open(os.path.join(RESULTS, "audit_validation.txt"), "w") as fh:
        fh.write("\n".join(OUT))


if __name__ == "__main__":
    main()

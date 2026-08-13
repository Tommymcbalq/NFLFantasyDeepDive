"""AUDIT 5 -- selection noise between genuinely different specs, rank stability
of the 272-game validation season, the winner's-curse magnitude, and the final
v1 fit scored ONCE on 2024-25.

The candidate menu below is fixed before any holdout number is looked at.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr, norm

from audit_common import (load, scores, logloss, brier, paired_boot, devig,
                          calibration_table, murphy,
                          TRAIN, VALID, HOLDOUT, SPEC_B, RESULTS)

OUT = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.append(s)

CV_TARGETS = list(range(2017, 2024))

REST = ["rest_short_diff", "rest_mini_bye_diff", "rest_bye_diff"]
CANDIDATES = {
    "A epa only":        ["off_epa_diff", "def_epa_diff"],
    "B specB":           SPEC_B,
    "C B + pace":        SPEC_B + ["pace_diff"],
    "D B + rest":        SPEC_B + REST,
    "E B + sr":          SPEC_B + ["off_sr_diff", "def_sr_diff"],
    "F pass/rush split": ["off_pass_epa_diff", "off_rush_epa_diff",
                          "def_pass_epa_diff", "def_rush_epa_diff", "to_margin_diff"],
    "G B + cpoe":        SPEC_B + ["off_cpoe_diff", "def_cpoe_diff"],
    "H B + early-down":  SPEC_B + ["off_early_epa_diff", "def_early_epa_diff"],
    "I 22-feature wide": ["off_epa_noto_diff", "def_epa_noto_diff", "to_margin_diff",
                          "off_pass_epa_diff", "off_rush_epa_diff", "def_pass_epa_diff",
                          "def_rush_epa_diff", "off_sr_diff", "def_sr_diff",
                          "off_cpoe_diff", "def_cpoe_diff", "off_pass_oe_diff",
                          "pace_diff", "pace_sum", "div_game", "is_neutral"] + REST +
                         ["off_early_epa_diff", "def_early_epa_diff", "pass_mismatch_diff"],
    "J B + neutral":     SPEC_B + ["is_neutral"],
    "K B no turnovers":  ["off_epa_noto_diff", "def_epa_noto_diff"],
}


def zfit(tr, te, cols):
    mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1.0)
    return (tr[cols] - mu) / sd, (te[cols] - mu) / sd


def pred(tr, te, cols):
    Xtr, Xte = zfit(tr, te, cols)
    f = sm.Logit(tr.home_win.values, sm.add_constant(Xtr.values)).fit(disp=0)
    return f.predict(sm.add_constant(Xte.values, has_constant="add")), f


def rolling(d, cols, targets=CV_TARGETS):
    ys, ps, bl = [], [], []
    for s in targets:
        te = d[d.season == s]
        q, _ = pred(d[d.season < s], te, cols)
        ys.append(te.home_win.values); ps.append(np.asarray(q)); bl.append(te.blk.values)
    return (np.concatenate(ys), np.concatenate(ps), np.concatenate(bl))


def main():
    d = load().sort_values(["season", "week"]).reset_index(drop=True)
    d["blk"] = d.season * 100 + d.week
    d["p_mkt"] = devig(d.home_moneyline, d.away_moneyline)
    d.loc[d.p_mkt.isna(), "p_mkt"] = norm.cdf(d.loc[d.p_mkt.isna(), "spread_home"] / 13.0)

    # ---- score every candidate on VALID-only and on rolling CV ----
    rows, cv_preds = [], {}
    for name, cols in CANDIDATES.items():
        pv, _ = pred(d[d.season.isin(TRAIN)], d[d.season.isin(VALID)], cols)
        yv = d[d.season.isin(VALID)].home_win.values
        y, q, bl = rolling(d, cols)
        cv_preds[name] = (y, q, bl)
        rows.append({"spec": name, "k": len(cols),
                     "valid2023_ll": logloss(yv, pv), "cv_ll": logloss(y, q),
                     "cv_brier": brier(y, q)})
    t = pd.DataFrame(rows).set_index("spec")
    t["valid_rank"] = t.valid2023_ll.rank()
    t["cv_rank"] = t.cv_ll.rank()
    p("=" * 92)
    p("CANDIDATE MENU: single-season VALID vs 7-fold rolling-origin CV")
    p("=" * 92)
    p(t.round(4).to_string())
    rho = spearmanr(t.valid2023_ll, t.cv_ll)
    p(f"\nSpearman rank correlation VALID-2023 vs rolling CV: rho = {rho.statistic:.3f} "
      f"(p={rho.pvalue:.3f})")
    p(f"VALID winner: {t.valid2023_ll.idxmin()};  CV winner: {t.cv_ll.idxmin()}")
    t.to_csv(os.path.join(RESULTS, "audit_candidates.csv"))

    # ---- how noisy is a 272-game comparison between two DIFFERENT specs? ----
    p("\n" + "=" * 92)
    p("SELECTION NOISE BETWEEN NON-NESTED SPECS")
    p("blocked-bootstrap 95% CI half-width for each spec vs spec B, at n=272 (2023)")
    p("and at n=1832 (rolling CV). This is the real 'how big must a gain be' number.")
    p("=" * 92)
    yv = d[d.season.isin(VALID)].home_win.values
    blv = d[d.season.isin(VALID)].blk.values
    pB_v, _ = pred(d[d.season.isin(TRAIN)], d[d.season.isin(VALID)], SPEC_B)
    yb, pb, blb = cv_preds["B specB"]
    nrows = []
    for name, cols in CANDIDATES.items():
        if name == "B specB":
            continue
        pv, _ = pred(d[d.season.isin(TRAIN)], d[d.season.isin(VALID)], cols)
        a, lo, hi, _ = paired_boot(yv, pv, pB_v, block=blv, B=3000)
        y, q, bl = cv_preds[name]
        a2, lo2, hi2, _ = paired_boot(y, q, pb, block=bl, B=3000)
        nrows.append({"spec": name, "valid_d": a, "valid_hw": (hi - lo) / 2,
                      "cv_d": a2, "cv_lo": lo2, "cv_hi": hi2, "cv_hw": (hi2 - lo2) / 2})
        p(f"  {name:20s} VALID {a:+.4f} (+-{(hi-lo)/2:.4f})   "
          f"CV {a2:+.4f} [{lo2:+.4f},{hi2:+.4f}]")
    nn = pd.DataFrame(nrows)
    p(f"\nmedian 95% CI half-width: VALID (272) {nn.valid_hw.median():.4f}   "
      f"CV (1832) {nn.cv_hw.median():.4f}")
    p(f"implied sd of a null difference: VALID ~{nn.valid_hw.median()/1.96:.4f}, "
      f"CV ~{nn.cv_hw.median()/1.96:.4f}")
    nn.to_csv(os.path.join(RESULTS, "audit_selection_noise.csv"), index=False)

    # ---- winner's curse: expected optimism of best-of-K on 272 games ----
    p("\nwinner's curse: with K=%d candidates whose true differences are ~0 and a "
      "null sd of %.4f," % (len(CANDIDATES), nn.valid_hw.median() / 1.96))
    sd = nn.valid_hw.median() / 1.96
    rng = np.random.default_rng(3)
    sim = rng.standard_normal((20000, len(CANDIDATES))) * sd
    p(f"  E[min] = {sim.min(axis=1).mean():+.4f}  -- the apparent gain of the "
      f"'best' spec on a single 272-game season is ~{-sim.min(axis=1).mean():.4f} "
      "of pure selection optimism.")
    sim2 = rng.standard_normal((20000, len(CANDIDATES))) * (nn.cv_hw.median() / 1.96)
    p(f"  same on the 1832-game rolling CV: E[min] = {sim2.min(axis=1).mean():+.4f}")

    # ---- LOSO sensitivity (features are leak-free within season) ----
    p("\nleave-one-season-out sensitivity (fit all seasons 2014-2023 except s, predict s;")
    p("uses future seasons for coefficients, so it is a sensitivity check only):")
    for name in ["B specB", "I 22-feature wide", "K B no turnovers"]:
        cols = CANDIDATES[name]
        ys, ps = [], []
        pool = d[d.season <= 2023]
        for s in range(2014, 2024):
            te = pool[pool.season == s]
            q, _ = pred(pool[pool.season != s], te, cols)
            ys.append(te.home_win.values); ps.append(np.asarray(q))
        p(f"  {name:20s} LOSO logloss {logloss(np.concatenate(ys), np.concatenate(ps)):.4f}")

    # =================================================================
    # FINAL v1: chosen by rolling CV only. Scored once on 2024-25.
    # =================================================================
    final_name = t.cv_ll.idxmin()
    p("\n" + "=" * 92)
    p(f"FINAL v1 -- spec chosen by rolling-origin CV: {final_name}")
    p("scored ONCE on the 2024-25 holdout")
    p("=" * 92)
    # pre-registered rule: prefer the smallest spec within 1 CV-noise sd of the best
    cvbest = t.cv_ll.min()
    within = t[t.cv_ll <= cvbest + nn.cv_hw.median() / 1.96].sort_values("k")
    p("specs within one CV-null-sd of the best (parsimony rule picks the top row):")
    p(within[["k", "cv_ll", "valid2023_ll"]].round(4).to_string())
    final_name = within.index[0]
    cols = CANDIDATES[final_name]
    p(f"\n=> v1 = {final_name}: {cols}")

    fitpool = d[d.season <= 2023]
    ph, f = pred(fitpool, d[d.season.isin(HOLDOUT)], cols)
    h = d[d.season.isin(HOLDOUT)].copy()
    h["p"] = np.asarray(ph)
    p("\ncoefficients (fit 2014-2023, standardised features):")
    p(pd.DataFrame({"coef": f.params, "se": f.bse, "z": f.tvalues},
                   index=["const"] + cols).round(4).to_string())

    p("\nholdout scores:")
    hs = pd.DataFrame([
        {"model": "v1 model", **scores(h.home_win, h.p)},
        {"model": "market (devigged moneyline)", **scores(h.home_win, h.p_mkt)},
        {"model": "base rate 2014-23", **scores(
            h.home_win, np.full(len(h), fitpool.home_win.mean()))},
    ]).set_index("model")
    p(hs.round(4).to_string())
    for s in HOLDOUT:
        m = h.season == s
        p(f"  {s}: model {logloss(h.home_win[m], h.p[m]):.4f}  "
          f"market {logloss(h.home_win[m], h.p_mkt[m]):.4f}  n={m.sum()}")

    for mname, mf in [("logloss", logloss), ("brier", brier)]:
        pt, lo, hi, _ = paired_boot(h.home_win.values, h.p.values, h.p_mkt.values,
                                    metric=mf, block=h.blk.values, B=5000)
        p(f"gap to market ({mname}): {pt:+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]")
    pt, lo, hi, _ = paired_boot(h.home_win.values, h.p.values,
                                np.full(len(h), fitpool.home_win.mean()),
                                block=h.blk.values, B=5000)
    p(f"gap to base rate (logloss): {pt:+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]")

    p("\nholdout calibration of v1:")
    p(calibration_table(h.home_win, h.p).round(3).to_string(index=False))
    lp = np.log(h.p / (1 - h.p))
    cx = sm.Logit(h.home_win.values, sm.add_constant(lp.values)).fit(disp=0)
    p(f"Cox recalibration: a={cx.params[0]:+.3f} (se {cx.bse[0]:.3f}), "
      f"b={cx.params[1]:.3f} (se {cx.bse[1]:.3f})")
    p("Murphy: " + str({k: round(v, 4) for k, v in murphy(h.home_win, h.p).items()}))

    h[["game_id", "season", "week", "home_team", "away_team", "home_win", "p",
       "p_mkt", "spread_home"]].to_csv(
        os.path.join(RESULTS, "audit_v1_holdout_predictions.csv"), index=False)

    with open(os.path.join(RESULTS, "audit_selection.txt"), "w") as fh:
        fh.write("\n".join(OUT))


if __name__ == "__main__":
    main()

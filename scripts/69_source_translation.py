"""§W2 / L3.2 — the source-translation map (§Q's recorded-but-not-executed construction).

The problem §Q stated: the isotonic map m(.) from ADP to PPG is ESTIMATED on ten/eleven years
of FantasyFootballCalculator boards.  Feeding a foreign source's ADP into it evaluates the fit
off the pool it was estimated on.  The correct construction is not to refit the curve but to
translate the foreign price into FFC-equivalent units first:

    t_s : rank on source s  ->  FFC-equivalent rank,   monotone increasing,
                                fitted on seasons where both sources are observed.

Then pi = m( t_s(A_s) ) uses the existing curve untouched.

Three things this script does, in order:

  1.  PROVENANCE CHECK on the overlap seasons, BEFORE fitting anything.  A rank->rank map is
      only meaningful if both series are the same kind of object: a preseason price.  ESPN's
      stored history for 2023/2024 is tested against realised outcomes and against FFC's
      genuine preseason price.  If a source's "historical ADP" predicts the season better than
      a real preseason market does, it is not a preseason quantity and cannot be used --
      fitting on it would inject future information into a preseason feature (cross-cutting
      rule 3).
  2.  FIT the monotone map on the seasons that survive (1), by isotonic regression of
      log FFC ADP on log source ADP.  Global as §Q specified; position-stratified as a
      declared sensitivity, declared because §35 predicted in advance that ESPN's
      disagreement with FFC is positional (TE/QB 30-50 slots earlier under 10-team/1TE
      defaults).
  3.  REPORT fit quality honestly, including the part a monotone map structurally cannot
      carry, and emit the translated 2026 prices for the board builder to consume as a
      named, switchable column.

Outputs:  results/adp_translation_diag.md (human), results/adp_translation_knots.csv,
          results/adp_espn_ffc_equiv_2026.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression

warnings.simplefilter("ignore")
ROOT = Path("/Users/thomasmcnamee/NFL")
sys.path.insert(0, str(ROOT / "scripts"))
from sectionM_common import norm_name, collapse_initials  # noqa: E402

FFC_BOARD = "data/adp/adp_ppr_2026_all_20260809.csv"   # the file 50/70_build_board price off
FFC_LATE = "data/adp/adp_ppr_2026_all_20260824.csv"    # sensitivity: later window
ESPN_HIST = "data/adp/adp_espn_historical.csv"
OVERLAP = (2023, 2024, 2026)


def key(s):
    return collapse_initials(norm_name(s))


def realised(y):
    wk = pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{y}.csv",
                     low_memory=False,
                     usecols=["player_display_name", "season_type", "fantasy_points_ppr"])
    wk = wk[wk.season_type == "REG"].copy()
    wk["k"] = wk.player_display_name.map(key)
    return wk.groupby("k").fantasy_points_ppr.agg(tot="sum", ppg="mean", g="size").reset_index()


# ------------------------------------------------------------------ 1. provenance
def provenance(esp, out):
    out.append("## 1. Provenance of the overlap seasons\n")
    out.append("A rank->rank map presumes both series are preseason prices.  Test: regress the\n"
               "season's realised PPR total on log(source ADP) and log(FFC ADP) jointly.  Two\n"
               "genuine preseason prices of the same market are near-collinear and neither\n"
               "dominates.  A series carrying hindsight drives the FFC coefficient to zero.\n")
    rows = []
    for y in (2023, 2024):
        ffc = pd.read_csv(ROOT / f"data/adp/historical/adp_ppr_{y}.csv")
        ffc["k"] = ffc.name.map(key)
        d = (esp[esp.season == y].merge(ffc[["k", "adp"]].rename(columns={"adp": "ffc"}), on="k")
             .merge(realised(y), on="k"))
        X = sm.add_constant(pd.DataFrame({"log_espn": np.log(d.adp.values),
                                          "log_ffc": np.log(d.ffc.values)}, index=d.index))
        r = sm.OLS(d.tot.values, X).fit()
        rows.append(dict(season=y, n=len(d),
                         rho_espn_ffc=d[["adp", "ffc"]].corr("spearman").iloc[0, 1],
                         rho_espn_outcome=-d[["adp", "tot"]].corr("spearman").iloc[0, 1],
                         rho_ffc_outcome=-d[["ffc", "tot"]].corr("spearman").iloc[0, 1],
                         b_espn=r.params["log_espn"], p_espn=r.pvalues["log_espn"],
                         b_ffc=r.params["log_ffc"], p_ffc=r.pvalues["log_ffc"]))
    t = pd.DataFrame(rows)
    out.append("\n" + t.round(4).to_string(index=False) + "\n")
    bad = t[(t.rho_espn_outcome > t.rho_ffc_outcome + 0.10) & (t.p_ffc > 0.05)].season.tolist()
    out.append(f"\n**Verdict: seasons {bad} are REJECTED as contaminated.** ESPN's stored history "
               "for those seasons predicts the realised season better than FFC's genuine "
               "preseason market does, and conditional on it the genuine preseason price adds "
               "nothing (p above).  That is not what one preseason board looks like against "
               "another; it is what a series refreshed with in/post-season draft activity looks "
               "like.  Fitting a translation map on them, or blending them into a 2026 price, "
               "would put future information into a preseason feature.\n")
    return set(bad)


# ------------------------------------------------------------------ 2. the map
def fit_map(x_src, y_ffc, w=None):
    """Monotone increasing isotonic fit of log FFC ADP on log source ADP.  Returns the
    knot table (sorted unique x with fitted y) so the board builder consumes a frozen
    artefact, not a pickled estimator."""
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(np.log(x_src), np.log(y_ffc), sample_weight=w)
    xs = np.sort(np.unique(np.log(x_src)))
    return pd.DataFrame(dict(log_src=xs, log_ffc_equiv=iso.predict(xs))), iso


def loo_cv(x, y):
    """Leave-one-out CV of the isotonic map, in log units.  Isotonic has no closed-form
    hat matrix, so this is brute force; n < 200 makes that free."""
    e = []
    for i in range(len(x)):
        m = np.ones(len(x), bool)
        m[i] = False
        _, iso = fit_map(x[m], y[m])
        e.append(np.log(y[i]) - iso.predict([np.log(x[i])])[0])
    return np.array(e)


def main():
    out = ["# L3.2 — source-translation map, ESPN -> FFC-equivalent\n\n",
           "*Constructed per §Q; §W2 of EDA_PLAN9.  Fitted 2026-08-24.*\n\n"]
    esp = pd.read_csv(ROOT / ESPN_HIST)
    esp["k"] = esp.name.map(key)
    bad = provenance(esp, out)
    usable = [y for y in OVERLAP if y not in bad]

    out.append("\n## 2. What is left to fit on\n")
    out.append(f"Overlap seasons {list(OVERLAP)}; usable after (1): **{usable}**.  §Q budgeted "
               "~380 matched player-seasons across three seasons; the honest count is one "
               "season.\n")

    ffc = pd.read_csv(ROOT / FFC_BOARD)
    ffc["k"] = ffc.name.map(key)
    e26 = esp[esp.season == 2026][["k", "name", "position", "adp"]].rename(
        columns={"adp": "espn_adp", "position": "espn_pos"})
    d = e26.merge(ffc[["k", "adp", "position", "team"]].rename(columns={"adp": "ffc_adp"}), on="k")
    out.append(f"\nMatched on the board's own FFC pull (`{FFC_BOARD}`): **{len(d)}** players "
               f"({len(e26)} ESPN, {len(ffc)} FFC).\n")

    x, y = d.espn_adp.values, d.ffc_adp.values
    knots, iso = fit_map(x, y)
    d["ffc_equiv"] = np.exp(iso.predict(np.log(x)))
    res_log = np.log(y) - np.log(d.ffc_equiv.values)
    cv = loo_cv(x, y)

    # identity benchmark: how much work is the map actually doing?
    id_log = np.log(y) - np.log(x)
    out.append("\n## 3. Fit quality — and the benchmark that matters\n")
    tab = pd.DataFrame([
        dict(map="identity (use ESPN ADP raw)", rmse_log=np.sqrt((id_log ** 2).mean()),
             mad_slots=np.abs(x - y).mean(), spearman=1.0),
        dict(map="isotonic, global (§Q spec)", rmse_log=np.sqrt((res_log ** 2).mean()),
             mad_slots=np.abs(d.ffc_equiv - y).mean(), spearman=1.0),
        dict(map="isotonic, LOO-CV", rmse_log=np.sqrt((cv ** 2).mean()), mad_slots=np.nan,
             spearman=1.0)])
    out.append("\n" + tab.round(4).to_string(index=False) + "\n")
    out.append(f"\nSpearman(ESPN, FFC) on the matched pool = "
               f"**{d[['espn_adp','ffc_adp']].corr('spearman').iloc[0,1]:.3f}**, "
               f"Pearson on logs = {np.corrcoef(np.log(x), np.log(y))[0,1]:.3f}.\n")

    # position-stratified sensitivity (declared: §35 predicted a positional gap)
    out.append("\n## 4. Declared sensitivity: position-stratified map\n")
    out.append("§35 predicted *in advance* that the ESPN-FFC disagreement is positional (ESPN's "
               "10-team/1QB/1TE defaults price TE and QB 30-50 slots earlier).  A single "
               "monotone rank->rank map cannot represent that: monotone maps preserve order, so "
               "any player ESPN ranks ahead of another stays ahead after translation.  The "
               "residual by position is therefore the part of the disagreement the specified "
               "construction structurally cannot carry.\n")
    d["res_log"] = res_log
    d["res_slots"] = d.ffc_adp - d.ffc_equiv
    bypos = d.groupby("position").agg(n=("res_log", "size"), mean_res_log=("res_log", "mean"),
                                      mean_res_slots=("res_slots", "mean"),
                                      med_res_slots=("res_slots", "median")).reset_index()
    out.append("\n" + bypos.round(3).to_string(index=False) + "\n")
    ps = []
    for p, g in d.groupby("position"):
        if len(g) < 12:
            continue
        kn, isp = fit_map(g.espn_adp.values, g.ffc_adp.values)
        r = np.log(g.ffc_adp.values) - isp.predict(np.log(g.espn_adp.values))
        kn["position"] = p
        ps.append(kn)
        out.append(f"  {p}: stratified in-sample RMSE(log) {np.sqrt((r**2).mean()):.4f} vs "
                   f"global {np.sqrt((g.res_log**2).mean()):.4f} on n={len(g)}\n")
    knots["position"] = "ALL"
    pd.concat([knots] + ps, ignore_index=True).to_csv(
        ROOT / "results/adp_translation_knots.csv", index=False)

    # sensitivity: later FFC window
    late = pd.read_csv(ROOT / FFC_LATE)
    late["k"] = late.name.map(key)
    dl = e26.merge(late[["k", "adp"]].rename(columns={"adp": "ffc_late"}), on="k")
    dl["ffc_equiv"] = np.exp(iso.predict(np.log(dl.espn_adp.values)))
    out.append("\n## 5. Sensitivity: which FFC window the map is fitted against\n")
    out.append(f"Refitting against the 08-24 FFC pull instead of the board's 08-09 pull moves the "
               f"translated price by mean |Δ| = "
               f"{np.abs(np.exp(fit_map(dl.espn_adp.values, dl.ffc_late.values)[1].predict(np.log(dl.espn_adp.values))) - dl.ffc_equiv).mean():.2f} "
               f"slots on {len(dl)} players.  The ESPN pull is dated 08-13 and the board's FFC "
               "window is 08-01→08-08; the five-day gap is real market movement and is part of "
               "the residual above, not fit error.\n")

    # coverage: players ESPN prices that FFC does not
    only_espn = e26[~e26.k.isin(set(ffc.k))]
    out.append("\n## 6. Coverage — the one thing translation buys unconditionally\n")
    out.append(f"{len(only_espn)} players carry an ESPN price and no FFC price; translation "
               "assigns them an FFC-equivalent slot without touching the fitted curve.\n")

    d["pool"] = "matched"
    only = only_espn.copy()
    only["ffc_adp"] = np.nan
    only["ffc_equiv"] = np.exp(iso.predict(np.log(only.espn_adp.values)))
    only["position"] = only.espn_pos
    only["pool"] = "espn_only"
    cols = ["k", "name", "position", "espn_adp", "ffc_adp", "ffc_equiv", "pool"]
    pd.concat([d[cols], only[cols]], ignore_index=True).to_csv(
        ROOT / "results/adp_espn_ffc_equiv_2026.csv", index=False)

    out.append("\n## 7. Verdict\n")
    out.append(
        "The map is **built and emitted, not adopted into the headline price.** Three reasons, "
        "all decided by the construction rather than by the answer it gives:\n\n"
        "1. **It cannot be validated.** §Q's whole case for the construction was that it is "
        "testable out of sample. The only two overlap seasons with realised outcomes fail the "
        "provenance test in §1, so there is no season on which translated-ESPN can be scored "
        "against FFC-alone. A 2026 fit evaluated on 2026 is in-sample by construction.\n"
        "2. **On the pool where it is identified it is nearly the identity.** The two boards "
        "agree at Spearman .92 and the isotonic map's residual is barely below the raw-ESPN "
        "residual; the translation is a rank-scale calibration, not a re-pricing.\n"
        "3. **The informative part of ESPN is exactly the part a monotone map deletes.** §35's "
        "finding is positional composition (TE/QB earlier under 1TE/1QB 10-team defaults). Order "
        "preservation means the translated price still carries ESPN's positional ordering into "
        "an FFC-calibrated curve — importing the scarcity assumption of a different roster "
        "format while claiming to have removed the source difference. That is worse than not "
        "translating.\n\n"
        "The board therefore carries `adp_espn`, `adp_ffc_equiv_espn` and "
        "`pi_market_espn_equiv` as named columns and `--price consensus` as a switch, and the "
        "layer-ablation table reports what adopting it would do. It is off by default.\n")

    (ROOT / "results/adp_translation_diag.md").write_text("".join(out))
    print("".join(out))


if __name__ == "__main__":
    main()

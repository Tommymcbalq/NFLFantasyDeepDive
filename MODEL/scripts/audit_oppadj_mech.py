r"""
Audit 9: WHY does opponent adjustment do nothing? Two rival explanations, and a
test that separates them.

  H1 "it is tiny"    NFL schedules are close to balanced (six divisional games
                     plus rotating blocks), so the spread in strength of
                     schedule is small next to the spread in team quality. The
                     adjustment barely moves any rating, so it can neither help
                     nor hurt.
  H2 "it is noise"   The adjustment moves ratings materially, but the movement
                     is driven by opponent ratings that are themselves badly
                     estimated, so it injects error.

These have different signatures:
  - H1 predicts sd(O_adj - O_raw) small relative to sd(O_raw), and the
    adjustment term uncorrelated with anything because it is nearly constant.
  - H2 predicts a large adjustment term that is NEGATIVELY useful -- in a
    forecast-encompassing regression of realised future performance on O_raw and
    on the adjustment, the adjustment gets a coefficient of zero (pure noise) or
    a wrong-signed one (actively harmful).

The encompassing test
---------------------
For every team-game, both O_raw and O_adj are known strictly before kickoff.
Regress the team's REALISED value in that game on the two leak-free predictors:

    y_realised = a + b1 * O_raw + b2 * (O_adj - O_raw) + e

b2 is what the adjustment buys. If opponent adjustment recovers true quality
that the raw mean misses, b2 should be positive and comparable to b1. If the
adjustment is noise, b2 = 0. If it is anti-informative, b2 < 0. Standard errors
are clustered by game, since the two teams in a game share a realisation.

This runs on ~10,000 team-games rather than 1,832 games, so it has far more power
than the downstream win model and can detect an effect the win model cannot.

Also reported: the reliability of the adjustment itself (does a team's estimated
strength-of-schedule correction in the first half of a season predict its own
correction in the second half?), and the raw dispersion of schedule strength.
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audit_quality as AQ

RES = os.path.join(HERE, "..", "results")
METRICS_TESTED = ["epa_noto", "epa", "to_rate", "pts_per_drive"]

out_lines = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out_lines.append(s)


def main():
    panel = AQ.load_panel()
    adj = pd.read_csv(os.path.join(HERE, "..", "data", "team_quality_v2adj.csv"))
    raw = pd.read_csv(os.path.join(HERE, "..", "data", "team_quality_v2raw.csv"))
    key = ["season", "week", "team"]
    q = adj.merge(raw, on=key, suffixes=("_adj", "_raw"))

    P("=" * 78)
    P("A. HOW BIG IS THE ADJUSTMENT?")
    P("   sd of the correction relative to the sd of the rating it corrects")
    P("=" * 78)
    P(f"  {'metric':16s} {'sd(O_raw)':>10s} {'sd(adj-raw)':>12s} {'ratio':>7s} "
      f"{'corr(raw,adj)':>14s}")
    for m in METRICS_TESTED:
        a, r = q[f"O_{m}_adj"], q[f"O_{m}_raw"]
        d = a - r
        P(f"  O_{m:14s} {r.std():10.4f} {d.std():12.4f} "
          f"{d.std()/r.std():7.3f} {np.corrcoef(a, r)[0,1]:14.4f}")
    for m in METRICS_TESTED:
        a, r = q[f"D_{m}_adj"], q[f"D_{m}_raw"]
        d = a - r
        P(f"  D_{m:14s} {r.std():10.4f} {d.std():12.4f} "
          f"{d.std()/r.std():7.3f} {np.corrcoef(a, r)[0,1]:14.4f}")

    # ---------------- forecast encompassing --------------------------------
    P("\n" + "=" * 78)
    P("B. FORECAST ENCOMPASSING: does the correction predict anything?")
    P("   y_realised ~ b1*O_raw + b2*(O_adj - O_raw), SEs clustered by game")
    P("   b2 ~ b1  -> the adjustment recovers real quality")
    P("   b2 ~ 0   -> the adjustment is pure noise")
    P("   b2 < 0   -> the adjustment is anti-informative")
    P("=" * 78)
    pan = panel[(panel.season_type == "REG") & (panel.season >= 2014)]
    rows = []
    for m in METRICS_TESTED:
        for side, letter in (("off", "O"), ("def", "D")):
            d = pan[["game_id", "season", "week", "team", f"{side}_{m}"]].merge(
                q[key + [f"{letter}_{m}_adj", f"{letter}_{m}_raw"]], on=key,
                how="inner").dropna()
            if len(d) < 500:
                continue
            d["corr_term"] = d[f"{letter}_{m}_adj"] - d[f"{letter}_{m}_raw"]
            X = sm.add_constant(d[[f"{letter}_{m}_raw", "corr_term"]])
            fit = sm.OLS(d[f"{side}_{m}"], X).fit(
                cov_type="cluster", cov_kwds={"groups": d.game_id})
            b1 = fit.params[f"{letter}_{m}_raw"]
            b2 = fit.params["corr_term"]
            t2 = fit.tvalues["corr_term"]
            ci = fit.conf_int().loc["corr_term"]
            rows.append({"metric": m, "side": letter, "n": len(d),
                         "b_raw": b1, "t_raw": fit.tvalues[f"{letter}_{m}_raw"],
                         "b_adjustment": b2, "t_adjustment": t2,
                         "lo": ci[0], "hi": ci[1], "R2": fit.rsquared})
    r = pd.DataFrame(rows)
    P(r.round(4).to_string(index=False))
    r.to_csv(os.path.join(RES, "audit_oppadj_encompassing.csv"), index=False)

    P("\n  Wald test b2 = b1 (does the correction carry the same weight as the")
    P("  rating it corrects, as it would if it were recovering true quality?)")
    for m in ["epa_noto", "epa"]:
        for side, letter in (("off", "O"), ("def", "D")):
            d = pan[["game_id", "season", "week", "team", f"{side}_{m}"]].merge(
                q[key + [f"{letter}_{m}_adj", f"{letter}_{m}_raw"]], on=key,
                how="inner").dropna()
            d["corr_term"] = d[f"{letter}_{m}_adj"] - d[f"{letter}_{m}_raw"]
            X = sm.add_constant(d[[f"{letter}_{m}_raw", "corr_term"]])
            fit = sm.OLS(d[f"{side}_{m}"], X).fit(
                cov_type="cluster", cov_kwds={"groups": d.game_id})
            w = fit.t_test("corr_term - " + f"{letter}_{m}_raw = 0")
            P(f"    {letter}_{m:10s} b2-b1 = {float(np.ravel(w.effect)[0]):+.4f} "
              f"t={float(np.ravel(w.tvalue)[0]):+.2f} p={float(np.ravel(w.pvalue)[0]):.4f}")

    # ---------------- reliability of the correction ------------------------
    P("\n" + "=" * 78)
    P("C. IS THE CORRECTION ITSELF RELIABLE?")
    P("   correlation between a team's mean correction in weeks 2-9 and in")
    P("   weeks 10-18 of the SAME season. A real strength-of-schedule effect")
    P("   should persist within a season; noise should not.")
    P("=" * 78)
    for m in ["epa_noto", "epa"]:
        q["corr_O"] = q[f"O_{m}_adj"] - q[f"O_{m}_raw"]
        h1 = q[q.week.between(2, 9)].groupby(["season", "team"]).corr_O.mean()
        h2 = q[q.week.between(10, 18)].groupby(["season", "team"]).corr_O.mean()
        j = pd.concat([h1.rename("h1"), h2.rename("h2")], axis=1).dropna()
        # comparison: reliability of the RAW rating over the same split
        q["raw_O"] = q[f"O_{m}_raw"]
        r1 = q[q.week.between(2, 9)].groupby(["season", "team"]).raw_O.mean()
        r2 = q[q.week.between(10, 18)].groupby(["season", "team"]).raw_O.mean()
        jr = pd.concat([r1.rename("h1"), r2.rename("h2")], axis=1).dropna()
        P(f"  {m:12s} corr(correction h1, h2) = {j.h1.corr(j.h2):+.3f}   "
          f"corr(raw rating h1, h2) = {jr.h1.corr(jr.h2):+.3f}   n={len(j)}")

    # ---------------- how much does schedule strength actually vary? -------
    P("\n" + "=" * 78)
    P("D. HOW UNBALANCED ARE NFL SCHEDULES, REALLY?")
    P("   sd across teams of the season-mean opponent quality, in the same")
    P("   units as team quality itself")
    P("=" * 78)
    for m in ["epa_noto", "epa"]:
        end = q[q.week == q.groupby("season").week.transform("max")]
        opp = pan.merge(end[key + [f"O_{m}_adj", f"D_{m}_adj"]].rename(
            columns={"team": "opponent"}), on=["season", "opponent"], how="inner")
        sos = opp.groupby(["season", "team"])[f"D_{m}_adj"].mean()
        team = end.set_index(["season", "team"])[f"O_{m}_adj"]
        P(f"  {m:12s} sd(team offence quality) = {team.std():.4f}   "
          f"sd(season-mean opponent defence) = {sos.std():.4f}   "
          f"ratio = {sos.std()/team.std():.3f}")

    with open(os.path.join(RES, "audit_oppadj_mech.txt"), "w") as f:
        f.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()

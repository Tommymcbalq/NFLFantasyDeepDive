"""
Audit 2: general look-ahead test, far stronger than verify_no_leakage().

Two independent tests, run over a grid of (season, week) cells and ALL metrics.

TEST 1 -- POISON. Replace every panel observation at or after the target
(season, week) with absurd values, recompute the quality estimate for that cell,
and require bit-identical output. This is the existing test generalised from one
cell / one metric to many cells / all metrics, and it poisons the *whole* row
(every metric column and every weight column), not just off_epa.

TEST 2 -- TRUNCATE (gold standard). Delete every observation at or after the
target from the panel entirely and rebuild. If any code path reads future rows --
even to compute a normalisation constant, a team list, or a season aggregate --
the estimate must move. Poisoning cannot catch a leak through a *count*;
truncation can.

Both are run against v1 (build_features) and v2 (build_features_v2).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_features as v1
import build_features_v2 as v2

RES = os.path.join(HERE, "..", "results")

CELLS = [(2015, 1), (2015, 9), (2017, 3), (2018, 14), (2019, 8), (2020, 1),
         (2021, 6), (2022, 11), (2023, 2), (2024, 17), (2025, 5)]

NUMERIC_POISON = 999.0


def poison(panel, t_target):
    """Corrupt every numeric column of every row at or after the cutoff."""
    q = panel.copy()
    mask = q.t >= t_target
    num = [c for c in q.columns
           if q[c].dtype.kind in "if" and c not in
           ("season", "week", "t", "is_home")]
    q.loc[mask, num] = NUMERIC_POISON
    return q


def truncate(panel, t_target):
    return panel[panel.t < t_target].copy()


def v1_cell(panel, metric, teams, season, week, t_target):
    prior = panel[(panel.t < t_target) & (panel.season >= season - v1.LOOKBACK_SEASONS)]
    if prior.empty:
        return None
    return v1._fit_week(prior, metric, teams, v1.LAMBDA[metric], t_target, season)


def v2_cell(panel, metric, teams, season, week, t_target):
    prior = v2.fit_prior(panel, metric, teams, season)
    return v2.fit_week(panel, metric, teams, season, week, prior)


def compare(a, b):
    if a is None or b is None:
        return a is None and b is None, np.nan
    dmax = 0.0
    for x, y in zip(a, b):
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        if x.shape != y.shape:
            return False, np.inf
        both_nan = np.isnan(x) & np.isnan(y)
        d = np.abs(np.where(both_nan, 0.0, x - y))
        if np.any(np.isnan(d)):
            return False, np.inf
        dmax = max(dmax, float(np.nanmax(d)) if d.size else 0.0)
    return dmax == 0.0, dmax


def main():
    panel = v1.load_panel()
    teams = np.sort(panel.team.dropna().unique())

    rows = []
    for season, week in CELLS:
        t_target = season * v1.WEEKS_PER_SEASON + week
        pois = poison(panel, t_target)
        trunc = truncate(panel, t_target)
        for name, fn in [("v1", v1_cell), ("v2", v2_cell)]:
            for metric in v1.METRICS:
                base = fn(panel, metric, teams, season, week, t_target)
                for label, alt_panel in [("poison", pois), ("truncate", trunc)]:
                    try:
                        alt = fn(alt_panel, metric, teams, season, week, t_target)
                    except Exception as e:                       # noqa: BLE001
                        rows.append({"version": name, "season": season, "week": week,
                                     "metric": metric, "test": label, "identical": False,
                                     "max_abs_diff": np.inf, "note": repr(e)[:60]})
                        continue
                    ident, dmax = compare(base, alt)
                    rows.append({"version": name, "season": season, "week": week,
                                 "metric": metric, "test": label,
                                 "identical": ident, "max_abs_diff": dmax, "note": ""})

    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(RES, "audit_leakage.csv"), index=False)

    print("=" * 78)
    print("LEAKAGE AUDIT")
    print(f"cells tested: {len(CELLS)}   metrics: {len(v1.METRICS)}   "
          f"total comparisons: {len(r)}")
    print("=" * 78)
    print(r.groupby(["version", "test"]).identical.agg(["sum", "size"]).to_string())
    fails = r[~r.identical]
    if len(fails):
        print("\n!!! FAILURES !!!")
        print(fails.to_string(index=False))
    else:
        print("\nno leakage detected: every estimate is bit-identical under both "
              "poisoning and truncation of all data at or after the cutoff.")

    # ---- separate check: does the v2 PRIOR depend only on s-1, s-2? ----
    print("\n" + "=" * 78)
    print("V2 PRIOR WINDOW CHECK (prior for season s must ignore all seasons "
          "outside {s-1, s-2})")
    print("=" * 78)
    bad = 0
    for season in [2018, 2022, 2024]:
        base = v2.fit_prior(panel, "epa", teams, season)
        q = panel.copy()
        m = ~q.season.isin([season - 1, season - 2])
        num = [c for c in q.columns if q[c].dtype.kind in "if"
               and c not in ("season", "week", "t", "is_home")]
        q.loc[m, num] = NUMERIC_POISON
        alt = v2.fit_prior(q, "epa", teams, season)
        ident, dmax = compare(base, alt)
        print(f"  season {season}: identical={ident} max_diff={dmax:.3e}")
        bad += (not ident)
    print("  PASS" if bad == 0 else "  FAIL")

    # ---- ridge identification / centring ----
    print("\n" + "=" * 78)
    print("RIDGE IDENTIFICATION: are O and D actually centred at zero?")
    print("mu unpenalised is claimed to fix additive non-identifiability.")
    print("=" * 78)
    for season, week in [(2019, 8), (2023, 12), (2024, 4)]:
        t_target = season * v1.WEEKS_PER_SEASON + week
        O, D, mu, hfa = v1_cell(panel, "epa", teams, season, week, t_target)
        print(f"  v1 {season}w{week}: mean(O)={O.mean():+.5f} mean(D)={D.mean():+.5f} "
              f"mu={mu:+.4f} hfa={hfa:+.4f} sd(O)={O.std():.4f} sd(D)={D.std():.4f}")
    for season, week in [(2019, 8), (2023, 12), (2024, 4)]:
        O, D, mu, hfa = v2_cell(panel, "epa", teams, season, week, None)
        print(f"  v2 {season}w{week}: mean(O)={O.mean():+.5f} mean(D)={D.mean():+.5f} "
              f"sd(O)={O.std():.4f} sd(D)={D.std():.4f}")


if __name__ == "__main__":
    main()

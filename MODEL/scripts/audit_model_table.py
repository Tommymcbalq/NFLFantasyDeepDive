"""
Audit 5: game-level differential tables, orientation-correct by construction.

For each estimator (v2adj / v2raw / v1adj) writes
data/audit_model_table_{est}.csv with, for every metric m,

    off_{m}_diff = s_m * (O_home - O_away)
    def_{m}_diff = s_m * (D_away  - D_home)
    net_{m}_diff = off_{m}_diff + def_{m}_diff

all oriented POSITIVE FAVOURS HOME. s_m is the metric's orientation sign from
audit_quality.METRICS. This fixes the defect in build_model_table.py, whose
docstring asserts the positive-favours-home convention but applies it only to
metrics where a higher offensive value is good -- leaving off_to_rate_diff,
off_sack_rate_diff etc. pointing the wrong way. (The turnover COMPOSITES in that
file are correct: to_margin_diff = -(off + def) is algebraically identical to the
s=-1 construction here. It is the raw per-metric diffs that are mis-oriented, and
those are what any feature search would sweep over.)
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from audit_quality import METRICS

GAMES = os.path.join(HERE, "..", "data", "games", "games.csv")
TEAM_FIXES = {"STL": "LA", "SD": "LAC", "OAK": "LV", "LAR": "LA", "JAC": "JAX"}
FIRST_SEASON = 2014


def build(est: str) -> pd.DataFrame:
    g = pd.read_csv(GAMES)
    g = g[(g.game_type == "REG") & g.result.notna()
          & (g.season >= FIRST_SEASON)].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_FIXES)

    q = pd.read_csv(os.path.join(HERE, "..", "data", f"team_quality_{est}.csv"))
    home = q.add_prefix("h_").rename(columns={"h_season": "season", "h_week": "week",
                                              "h_team": "home_team"})
    away = q.add_prefix("a_").rename(columns={"a_season": "season", "a_week": "week",
                                              "a_team": "away_team"})
    n0 = len(g)
    g = g.merge(home, on=["season", "week", "home_team"], how="inner")
    g = g.merge(away, on=["season", "week", "away_team"], how="inner")
    assert len(g) == n0, f"{n0 - len(g)} games lost in merge"

    cols = []
    for m, (_, s) in METRICS.items():
        g[f"off_{m}_diff"] = s * (g[f"h_O_{m}"] - g[f"a_O_{m}"])
        g[f"def_{m}_diff"] = s * (g[f"a_D_{m}"] - g[f"h_D_{m}"])
        g[f"net_{m}_diff"] = g[f"off_{m}_diff"] + g[f"def_{m}_diff"]
        cols += [f"off_{m}_diff", f"def_{m}_diff", f"net_{m}_diff"]

    g["home_win"] = np.where(g.result > 0, 1.0, np.where(g.result < 0, 0.0, 0.5))
    g["margin"] = g.result
    g["spread_home"] = g.spread_line
    g["div_game"] = g.div_game.fillna(0).astype(int)
    g["is_neutral"] = (g.location != "Home").astype(int)

    keep = ["game_id", "season", "week", "home_team", "away_team", "home_win",
            "margin", "spread_home", "total_line", "div_game", "is_neutral"] + cols
    return g[keep].copy()


def main():
    for est in ("v2adj", "v2raw", "v1adj"):
        d = build(est)
        out = os.path.join(HERE, "..", "data", f"audit_model_table_{est}.csv")
        d.to_csv(out, index=False)
        print(f"{est}: {d.shape} -> {out}")
    print(f"\nseasons {d.season.min()}-{d.season.max()}, "
          f"home win rate {d.home_win.mean():.4f}, ties {(d.home_win == 0.5).sum()}")


if __name__ == "__main__":
    main()

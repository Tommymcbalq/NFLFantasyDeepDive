"""
Schedule-strength construction, §I-style sourcing scout (written 2026-08-09).

Builds PRESEASON-KNOWABLE strength-of-schedule measures for 2015-2026 from two
inputs that are both available before Week 1:

  (a) the schedule grid  -- data/teams/games_nflverse_20260809.csv, REG games only,
      home_team/away_team/week/season ONLY. spread_line/total_line are BARRED
      (in-season, look-ahead).
  (b) preseason team win totals -- data/vegas/team_win_totals_2015_2025_covers.csv
      (closing, ~Sep 1-10 stamps) for 2015-2025, and
      data/vegas/team_totals_2026.csv (DraftKings, 2026-08-08) for 2026.

Measures produced, per team-season:
  sos_vegas       mean opponent preseason win total, all REG opponents
  sos_vegas_p1_14 same, weeks 1-14   (fantasy regular season)
  sos_vegas_p15_17 same, weeks 15-17 (fantasy playoffs)
  sos_prior_wins  mean opponent PRIOR-SEASON realized wins (no market input at all;
                  constructible back to 2000 from the grid alone)
  n_opp           number of REG opponents (bye-aware)

Nothing here uses any realized outcome of the season being described.
"""
import pandas as pd, numpy as np, os

ROOT = "/Users/thomasmcnamee/NFL"
GAMES = f"{ROOT}/data/teams/games_nflverse_20260809.csv"
WT_HIST = f"{ROOT}/data/vegas/team_win_totals_2015_2025_covers.csv"
WT_2026 = f"{ROOT}/data/vegas/team_totals_2026.csv"
OUT = f"{ROOT}/data/schedule"

# Covers uses full franchise names, incl. relocation-era names. Map to the
# nflverse abbreviation used in that same season's grid.
NAME2ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    # relocation / rename families -- the grid abbreviation is season-dependent,
    # so these resolve against the set of abbreviations present that season.
    "Oakland Raiders": "OAK", "Las Vegas Raiders": "LV",
    "San Diego Chargers": "SD", "Los Angeles Chargers": "LAC",
    "St Louis Rams": "STL", "St. Louis Rams": "STL", "Los Angeles Rams": "LA",
    "Washington Redskins": "WAS", "Washington Football Team": "WAS",
    "Washington Commanders": "WAS",
}


def load_grid():
    g = pd.read_csv(GAMES, usecols=["season", "game_type", "week", "home_team", "away_team"])
    g = g[(g.game_type == "REG") & (g.season.between(2000, 2026))].copy()
    # long form: one row per team per game
    a = g.rename(columns={"home_team": "team", "away_team": "opp"})[["season", "week", "team", "opp"]]
    b = g.rename(columns={"away_team": "team", "home_team": "opp"})[["season", "week", "team", "opp"]]
    return pd.concat([a, b], ignore_index=True)


def load_win_totals(grid):
    """Preseason win total per (season, abbr), 2015-2026."""
    h = pd.read_csv(WT_HIST)
    h["abbr"] = h.team.map(NAME2ABBR)
    assert h.abbr.notna().all(), h[h.abbr.isna()].team.unique()
    h = h[["season", "abbr", "win_total"]]

    n = pd.read_csv(WT_2026)
    n = n.rename(columns={"team": "abbr", "win_total_dk": "win_total"})[["season", "abbr", "win_total"]]

    wt = pd.concat([h, n], ignore_index=True)

    # Reconcile relocation abbreviations against what the grid actually used that season.
    valid = grid.groupby("season").team.apply(lambda s: set(s)).to_dict()
    fix = {("OAK", "LV"), ("LV", "OAK"), ("SD", "LAC"), ("LAC", "SD"),
           ("STL", "LA"), ("LA", "STL"), ("LA", "LAR"), ("LAR", "LA")}
    rows = []
    for _, r in wt.iterrows():
        s, ab = int(r.season), r.abbr
        vs = valid.get(s, set())
        if ab not in vs:
            alt = [b for (a_, b) in fix if a_ == ab and b in vs]
            assert alt, f"cannot resolve {ab} in {s}"
            ab = alt[0]
        rows.append((s, ab, r.win_total))
    return pd.DataFrame(rows, columns=["season", "abbr", "win_total"])


def prior_realized_wins(games_path):
    """Realized regular-season wins per team-season, from scores. Used only as the
    PRIOR season's value, so it is known in August."""
    g = pd.read_csv(games_path, usecols=["season", "game_type", "home_team", "away_team",
                                         "home_score", "away_score"])
    g = g[(g.game_type == "REG") & g.home_score.notna()].copy()
    rec = []
    for _, r in g.iterrows():
        hw = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
        rec.append((r.season, r.home_team, hw))
        rec.append((r.season, r.away_team, 1.0 - hw))
    d = pd.DataFrame(rec, columns=["season", "abbr", "w"])
    out = d.groupby(["season", "abbr"], as_index=False).agg(wins=("w", "sum"), gp=("w", "size"))
    out["win_pct"] = out.wins / out.gp
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    grid = load_grid()
    wt = load_win_totals(grid)
    rw = prior_realized_wins(GAMES)

    # opponent preseason win total
    g = grid.merge(wt.rename(columns={"abbr": "opp", "win_total": "opp_wt"}),
                   on=["season", "opp"], how="left")
    # opponent prior-season win pct (prior season, so shift season by 1)
    rwp = rw.copy()
    rwp["season"] = rwp.season + 1
    g = g.merge(rwp.rename(columns={"abbr": "opp", "win_pct": "opp_prior_wpct",
                                    "wins": "opp_prior_wins"})[
                    ["season", "opp", "opp_prior_wpct", "opp_prior_wins"]],
                on=["season", "opp"], how="left")

    g = g[g.season.between(2015, 2026)]
    miss = g[g.season.between(2015, 2026) & g.opp_wt.isna()]
    print("missing opponent win totals:", len(miss), miss.season.unique() if len(miss) else "")

    def agg(sub, suffix):
        return sub.groupby(["season", "team"], as_index=False).agg(**{
            f"sos_vegas{suffix}": ("opp_wt", "mean"),
            f"sos_prior_wpct{suffix}": ("opp_prior_wpct", "mean"),
            f"n_opp{suffix}": ("opp", "size")})

    full = agg(g, "")
    fps = agg(g[g.week <= 14], "_w1_14")
    plf = agg(g[g.week.between(15, 17)], "_w15_17")

    out = full.merge(fps, on=["season", "team"]).merge(plf, on=["season", "team"])
    # league-centered z-score within season (schedules are zero-sum-ish within a year)
    for c in ["sos_vegas", "sos_prior_wpct", "sos_vegas_w15_17"]:
        out[c + "_z"] = out.groupby("season")[c].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    out["source"] = ("computed: nflverse games.csv REG grid x preseason win totals "
                     "(Covers closing 2015-2025; DraftKings 2026-08-08 for 2026); "
                     "built 2026-08-09 by scripts/fetch_sos.py")
    out = out.sort_values(["season", "team"])
    p = f"{OUT}/sos_history_2015_2026.csv"
    out.to_csv(p, index=False)
    print("wrote", p, out.shape)
    print(out.groupby("season").agg(n=("team", "size"), sd_vegas=("sos_vegas", "std"),
                                    sd_prior=("sos_prior_wpct", "std")).to_string())
    return out


if __name__ == "__main__":
    main()

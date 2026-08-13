"""Assemble data/schedule/sos_2026.csv — long format, one row per (team, source, metric).
These are NOT the same statistic, so they are stacked, never averaged. Built 2026-08-09.
"""
import pandas as pd

ROOT = "/Users/thomasmcnamee/NFL"
rows = []


def add(source, metric, basis, definition, knowable, d, ranked_low_is_easy=True):
    """d: dict team -> value (or list of teams in rank order if metric endswith '_rank')."""
    if isinstance(d, list):
        d = {t: i + 1 for i, t in enumerate(d)}
    for t, v in d.items():
        rows.append(dict(season=2026, team=t, source=source, metric=metric, value=v,
                         basis_year=basis, definition=definition,
                         preseason_knowable=knowable))


# --- 1. Computed in-house (reproducible, and the one with a historical twin) ---
h = pd.read_csv(f"{ROOT}/data/schedule/sos_history_2015_2026.csv")
n = h[h.season == 2026]
add("computed/fetch_sos.py", "mean_opp_preseason_win_total", "2026 DK win totals (2026-08-08)",
    "Mean over the 17 regular-season opponents of that opponent's preseason Vegas win total. "
    "Higher = harder. Same construction as Sharp Football's headline SOS.", "YES",
    dict(zip(n.team, n.sos_vegas.round(4))))
add("computed/fetch_sos.py", "mean_opp_preseason_win_total_w15_17", "2026 DK win totals (2026-08-08)",
    "Same, restricted to weeks 15-17 (fantasy playoffs). Higher = harder.", "YES",
    dict(zip(n.team, n.sos_vegas_w15_17.round(4))))
add("computed/fetch_sos.py", "mean_opp_prior_season_win_pct", "2025 realized records",
    "Mean over 2026 opponents of that opponent's 2025 realized regular-season win pct. "
    "Higher = harder. No market input; constructible from the grid alone back to 2000.", "YES",
    dict(zip(n.team, n.sos_prior_wpct.round(4))))

# --- 2. Sharp Football Analysis, headline team SOS (rank only, 1 = easiest) ---
sharp = ["DET", "CIN", "NO", "NYJ", "BAL", "CLE", "SF", "PHI", "IND", "NE", "DEN", "KC", "LV",
         "BUF", "MIN", "TEN", "PIT", "JAX", "ATL", "LAC", "NYG", "HOU", "SEA", "GB", "TB",
         "CHI", "LA", "DAL", "WAS", "CAR", "MIA", "ARI"]
add("sharpfootballanalysis.com/analysis/nfl-strength-of-schedule/ (2026-08-02, retr. 2026-08-09)",
    "sos_rank", "2026 Vegas win totals",
    "Rank 1 = easiest. Explicitly built from 2026 Vegas projected win totals of opponents, "
    "not prior-year records. No numeric value published on the page.", "YES", sharp)

# --- 3. CBS Sports, opponent win pct off 2025 records ---
cbs = {"CHI": .550, "MIA": .542, "ARI": .538, "GB": .538, "KC": .536, "NE": .531, "LV": .529,
       "BUF": .528, "LAC": .522, "CAR": .521, "MIN": .519, "NYJ": .517, "LA": .516, "SEA": .514,
       "DEN": .512, "WAS": .502, "NYG": .498, "SF": .497, "PIT": .495, "DAL": .493, "TB": .491,
       "JAX": .490, "PHI": .481, "BAL": .479, "TEN": .476, "HOU": .474, "DET": .467, "ATL": .465,
       "IND": .465, "CIN": .450, "NO": .434, "CLE": .429}
add("cbssports.com/nfl/news/2026-nfl-strength-of-schedule/ (2026-05-12, retr. 2026-08-09)",
    "opp_win_pct_2025", "2025 realized records",
    "Combined 2025 win pct of the 17 2026 opponents. Higher = harder.", "YES", cbs)

# --- 4. LeagueStation: positional, prior-year fantasy points allowed x 2026 schedule ---
# Recomputed by us in full PPR from their unauthenticated API; their scoring was reproduced
# exactly against their published QB numbers first.
ls_wr = {"PHI": 34.40, "TEN": 33.46, "CLE": 33.29, "NYG": 33.28, "HOU": 33.18, "CIN": 33.04,
         "JAX": 32.79, "MIN": 32.76, "DAL": 32.64, "NO": 32.38, "TB": 32.38, "SEA": 32.27,
         "ARI": 31.95, "WAS": 31.93, "IND": 31.77, "ATL": 31.56, "LA": 31.56, "GB": 31.50,
         "SF": 31.47, "DET": 31.21, "BUF": 31.19, "CAR": 31.07, "DEN": 31.06, "BAL": 31.03,
         "LAC": 30.86, "NE": 30.78, "CHI": 30.77, "MIA": 30.72, "NYJ": 30.66, "KC": 30.62,
         "PIT": 30.34, "LV": 29.77}
ls_wr_p = {"CLE": 36.52, "TEN": 36.48, "NYG": 34.81, "MIN": 34.62, "CIN": 34.52, "JAX": 34.35,
           "PIT": 34.08, "DAL": 33.85, "LA": 33.45, "NO": 32.88, "TB": 32.75, "ATL": 32.05,
           "CHI": 31.97, "DET": 31.94, "LV": 31.61, "BUF": 31.60, "GB": 31.29, "ARI": 31.08,
           "HOU": 30.77, "LAC": 30.72, "WAS": 30.63, "CAR": 30.48, "SEA": 30.48, "KC": 30.46,
           "BAL": 30.33, "DEN": 30.33, "IND": 29.85, "MIA": 29.15, "PHI": 29.08, "NE": 28.91,
           "NYJ": 28.51, "SF": 28.15}
LSDEF = ("Mean full-PPR fantasy points allowed to the position by the 2026 opponents, using "
         "each defense's 2025 realized per-game allowance. Higher = easier. Recomputed by us "
         "in full PPR from leaguestation.com/api/draft-kit/strength-of-schedule (no auth); "
         "their half-PPR scoring was reproduced exactly against their published QB values first.")
add("leaguestation.com API, recomputed full-PPR (retr. 2026-08-09)", "wr_ppr_pts_allowed_season",
    "2025 defensive FPA", LSDEF, "YES", ls_wr)
add("leaguestation.com API, recomputed full-PPR (retr. 2026-08-09)", "wr_ppr_pts_allowed_w15_17",
    "2025 defensive FPA", LSDEF + " Weeks 15-17 only.", "YES", ls_wr_p)

# 2025 raw defensive allowance (the ingredient, not an SOS) — kept for auditability
ls_def25_wr = {"DAL": 40.66, "PIT": 37.73, "IND": 37.62, "BAL": 37.49, "CHI": 36.91, "DET": 36.68,
               "TEN": 36.29, "WAS": 36.02, "ATL": 35.28, "LA": 34.99, "NYG": 34.44, "LV": 34.09,
               "SF": 33.34, "GB": 32.19, "TB": 32.15, "JAX": 32.12, "ARI": 31.22, "NYJ": 31.17,
               "MIA": 30.58, "NE": 29.84, "CAR": 28.45, "KC": 28.25, "LAC": 28.21, "PHI": 27.99,
               "NO": 27.97, "SEA": 27.53, "DEN": 27.32, "CLE": 27.08, "BUF": 27.05, "HOU": 26.37,
               "CIN": 26.18, "MIN": 24.48}
add("leaguestation.com API, recomputed full-PPR (retr. 2026-08-09)", "def_wr_ppr_allowed_pg_2025",
    "2025 realized", "That DEFENSE's own 2025 full-PPR WR points allowed per game. This is the "
    "input to the SOS above, not an SOS itself. Higher = softer defense.", "YES", ls_def25_wr)

ls_def25_rb = {"CIN": 29.20, "NYJ": 28.88, "ARI": 28.09, "NYG": 26.72, "WAS": 26.59, "DAL": 26.31,
               "MIA": 25.92, "BUF": 24.96, "CAR": 24.91, "SF": 23.99, "PHI": 23.68, "LV": 23.57,
               "BAL": 23.42, "TB": 22.88, "CLE": 22.43, "CHI": 22.31, "GB": 22.22, "TEN": 22.06,
               "ATL": 22.03, "NO": 21.63, "LA": 21.38, "IND": 20.53, "MIN": 20.46, "HOU": 20.06,
               "NE": 19.85, "KC": 19.71, "DET": 19.66, "PIT": 19.41, "LAC": 19.06, "SEA": 18.87,
               "JAX": 18.81, "DEN": 17.18}
add("leaguestation.com API, recomputed full-PPR (retr. 2026-08-09)", "def_rb_ppr_allowed_pg_2025",
    "2025 realized", "That DEFENSE's own 2025 full-PPR RB points allowed per game. Higher = "
    "softer defense. Note the WR and RB orderings are largely unrelated — DEN is 32nd vs RB "
    "and 27th-softest vs WR; CIN is 1st vs RB and 31st vs WR.", "YES", ls_def25_rb)

# --- 5. FantasyNerds proprietary positional SOS score (higher = easier) ---
fn_wr = {"GB": 360, "NO": 336, "ATL": 335, "DET": 324, "TB": 321, "CHI": 318, "MIN": 317,
         "NYJ": 310, "SEA": 304, "PHI": 297, "CIN": 296, "IND": 290, "WAS": 283, "DEN": 282,
         "BUF": 278, "CLE": 270, "CAR": 269, "LAC": 268, "NYG": 267, "HOU": 266, "ARI": 264,
         "SF": 260, "DAL": 256, "BAL": 256, "NE": 255, "LA": 250, "JAX": 249, "PIT": 248,
         "LV": 241, "TEN": 221, "KC": 217}  # MIA absent from source
add("fantasynerds.com/nfl/strength-of-schedule/WR (retr. 2026-08-09)", "fn_sos_score_wr",
    "undisclosed", "Proprietary undisclosed 'SOS Score', higher = easier; appears to be a sum "
    "of 17 weekly matchup grades. Methodology unpublished, basis year unstated. MIA is MISSING "
    "from the source page (31 of 32 teams).", "UNCLEAR", fn_wr)
fn_rb = {"SF": 339, "KC": 327, "NE": 318, "DEN": 316, "LAC": 313, "BUF": 308, "TEN": 304,
         "NYJ": 302, "LV": 298, "NO": 298, "LA": 297, "DET": 294, "CAR": 284, "SEA": 277,
         "WAS": 276, "CIN": 273, "TB": 273, "JAX": 272, "PIT": 272, "CHI": 271, "DAL": 270,
         "MIA": 266, "IND": 265, "BAL": 261, "GB": 261, "ARI": 257, "HOU": 253, "PHI": 252,
         "CLE": 252, "MIN": 251, "NYG": 250, "ATL": 226}
add("fantasynerds.com/nfl/strength-of-schedule/RB (retr. 2026-08-09)", "fn_sos_score_rb",
    "undisclosed", "Same proprietary score for RB, higher = easier.", "UNCLEAR", fn_rb)

# --- 6. Sharp Football fantasy SOS, rank only, 1 = easiest ---
SH = "sharpfootballanalysis.com/fantasy/strength-of-schedule-fantasy-football/ (retr. 2026-08-09)"
SHDEF = ("Rank only, 1 = easiest; no numeric value published. Sharp describes it as opponent "
         "efficiency 'with game script taken out', basis = last season (2025). Passing/rushing, "
         "not WR/RB specifically. 'Playoff' = weeks 15-17.")
add(SH, "sharp_season_passing_rank", "2025", SHDEF, "YES",
    ["PHI", "MIN", "SEA", "DET", "LA", "ATL", "LAC", "CLE", "NYG", "MIA", "DEN", "NYJ", "BUF",
     "CIN", "KC", "JAX", "GB", "NO", "TEN", "IND", "HOU", "ARI", "DAL", "BAL", "SF", "LV", "TB",
     "NE", "WAS", "PIT", "CHI", "CAR"])
add(SH, "sharp_playoff_passing_rank", "2025", SHDEF, "YES",
    ["MIN", "ATL", "LAC", "ARI", "NO", "JAX", "LA", "PIT", "IND", "LV", "BUF", "KC", "TEN", "CIN",
     "CLE", "GB", "NE", "NYJ", "DEN", "CAR", "NYG", "BAL", "DET", "TB", "CHI", "PHI", "MIA", "SF",
     "DAL", "WAS", "SEA", "HOU"])
add(SH, "sharp_season_rushing_rank", "2025", SHDEF, "YES",
    ["DET", "NO", "SEA", "LA", "MIN", "TB", "PHI", "GB", "WAS", "MIA", "ATL", "HOU", "BAL", "CHI",
     "NYJ", "DAL", "NYG", "IND", "JAX", "SF", "CLE", "KC", "NE", "TEN", "DEN", "ARI", "CAR", "PIT",
     "BUF", "CIN", "LV", "LAC"])
add(SH, "sharp_playoff_rushing_rank", "2025", SHDEF, "YES",
    ["DET", "CHI", "MIA", "NO", "PIT", "JAX", "MIN", "IND", "LAC", "DEN", "NYG", "GB", "KC", "SEA",
     "ATL", "BUF", "CLE", "NYJ", "CIN", "LV", "LA", "SF", "WAS", "BAL", "DAL", "ARI", "CAR", "HOU",
     "TB", "TEN", "PHI", "NE"])

out = pd.DataFrame(rows)
p = f"{ROOT}/data/schedule/sos_2026.csv"
out.to_csv(p, index=False)
print("wrote", p, out.shape)
print(out.groupby(["source", "metric"]).size().to_string())

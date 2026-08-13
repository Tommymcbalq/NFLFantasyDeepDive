"""Pull the raw data for the WR preseason valuation model.

Outputs (all under data/):
  adp/adp_ppr_2026_all.csv          - full 2026 PPR ADP board (FFC, 12-team)
  adp/wr_top30_adp_2026.csv         - top 30 WRs by ADP
  players/weekly_raw/               - full nflverse weekly player stats per season (raw cache)
  players/wr_top30_weekly.csv       - all weekly game logs for the 30 WRs, full careers
  players/by_player/<slug>.csv      - per-player game logs
  teams/team_week_<year>.csv        - league-wide team weekly stats per season
  meta/players_meta.csv             - nflverse player metadata (birth date, draft, entry year)
  meta/wr_top30_meta.csv            - metadata for just our 30 WRs
"""

import io
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"

# Seasons: oldest plausible top-30 WR career start is 2014 (Adams/Evans draft class).
PLAYER_SEASONS = range(2014, 2026)   # 2014..2025 inclusive
TEAM_SEASONS = range(2014, 2026)     # team files are small; grab the same window

session = requests.Session()
session.headers["User-Agent"] = "wr-valuation-project/0.1"


def get_csv(url: str, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)
    r = session.get(url, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content), low_memory=False)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"[.'\-]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def slug(name: str) -> str:
    return norm_name(name).replace(" ", "_")


# ---------------------------------------------------------------- 1. ADP
print("== 1. ADP (FFC, PPR, 12-team, 2026) ==")
r = session.get(
    "https://fantasyfootballcalculator.com/api/v1/adp/ppr",
    params={"teams": 12, "year": 2026},
    timeout=60,
)
r.raise_for_status()
adp_json = r.json()
adp = pd.DataFrame(adp_json["players"])
adp["pulled_meta"] = json.dumps(adp_json["meta"])
(DATA / "adp").mkdir(parents=True, exist_ok=True)
adp.to_csv(DATA / "adp" / "adp_ppr_2026_all.csv", index=False)

wr30 = adp[adp["position"] == "WR"].nsmallest(30, "adp").reset_index(drop=True)
wr30.insert(0, "wr_adp_rank", wr30.index + 1)
wr30.to_csv(DATA / "adp" / "wr_top30_adp_2026.csv", index=False)
print(wr30[["wr_adp_rank", "name", "team", "adp", "adp_formatted", "stdev", "times_drafted"]]
      .to_string(index=False))

wr30["norm"] = wr30["name"].map(norm_name)

# ------------------------------------------------- 2. player metadata
print("\n== 2. nflverse player metadata ==")
meta = get_csv(f"{NFLVERSE}/players/players.csv", DATA / "meta" / "players_meta.csv")
meta_wr = meta[(meta["position"] == "WR") & (meta["last_season"] >= 2024)].copy()
meta_wr["norm"] = meta_wr["display_name"].map(norm_name)

matched = wr30.merge(
    meta_wr[["norm", "gsis_id", "display_name", "birth_date", "rookie_season",
             "last_season", "years_of_experience", "draft_year", "draft_round",
             "draft_pick", "draft_team", "height", "weight", "college_name",
             "ngs_position"]],
    on="norm", how="left",
)
unmatched = matched[matched["gsis_id"].isna()]
if len(unmatched):
    print("!! unmatched vs nflverse metadata:", unmatched["name"].tolist())
matched.to_csv(DATA / "meta" / "wr_top30_meta.csv", index=False)
print(f"metadata matched for {matched['gsis_id'].notna().sum()}/30 WRs")

gsis_ids = set(matched["gsis_id"].dropna())

# --------------------------------------------- 3. weekly player stats
print("\n== 3. weekly player stats (nflverse), seasons "
      f"{PLAYER_SEASONS.start}-{PLAYER_SEASONS.stop - 1} ==")
frames = []
for yr in PLAYER_SEASONS:
    df = get_csv(f"{NFLVERSE}/stats_player/stats_player_week_{yr}.csv",
                 DATA / "players" / "weekly_raw" / f"stats_player_week_{yr}.csv")
    id_col = "player_id" if "player_id" in df.columns else "gsis_id"
    sub = df[df[id_col].isin(gsis_ids)].copy()
    sub["gsis_id"] = sub[id_col]
    frames.append(sub)
    print(f"  {yr}: {len(df):>6} rows total, {len(sub):>4} rows for our WRs")

weekly = pd.concat(frames, ignore_index=True)
weekly = weekly.merge(matched[["gsis_id", "name", "wr_adp_rank"]], on="gsis_id", how="left")
weekly.to_csv(DATA / "players" / "wr_top30_weekly.csv", index=False)

by_dir = DATA / "players" / "by_player"
by_dir.mkdir(parents=True, exist_ok=True)
for name, grp in weekly.groupby("name"):
    grp.to_csv(by_dir / f"{slug(name)}.csv", index=False)

# Coverage check: anyone whose first season in our window is 2014 might predate it.
cov = (weekly.groupby("name")["season"]
       .agg(first="min", last="max", games="count")
       .sort_values("first"))
print("\nper-player coverage (seasons, game rows):")
print(cov.to_string())
edge = cov[cov["first"] == PLAYER_SEASONS.start]
if len(edge):
    print(f"!! first-season == {PLAYER_SEASONS.start} (check career actually starts here):",
          edge.index.tolist())
missing = set(matched.loc[matched["gsis_id"].notna(), "name"]) - set(cov.index)
if missing:
    print("!! matched WRs with NO weekly rows:", sorted(missing))

# --------------------------------------------------- 4. team weekly stats
print("\n== 4. team weekly stats (nflverse) ==")
for yr in TEAM_SEASONS:
    df = get_csv(f"{NFLVERSE}/stats_team/stats_team_week_{yr}.csv",
                 DATA / "teams" / f"stats_team_week_{yr}.csv")
    print(f"  {yr}: {len(df)} team-week rows, {df['team'].nunique()} teams")

print("\ndone.")

"""Round-4 data pull: RB universe + the long (1999-) panel for the age/era work.

Additive only — never touches round-1..3 raw files. Outputs under data/:
  adp/rb_top30_adp_2026.csv         - top 30 RBs by the 2026-08-09 ADP pull
  meta/rb_top30_meta.csv            - metadata for those RBs (gsis_id join)
  players/rb_top30_weekly.csv       - full-career game logs for the 30 RBs
  players/by_player_rb/<slug>.csv   - per-player game logs
  players/weekly_raw/               - extended back to 1999 (raw cache, all players)
  teams/stats_team_week_<yr>.csv    - extended back to 1999
  derived/age_panel_long.csv        - WR+RB player-season panel 1999-2025 with age

The August ADP board (adp_ppr_2026_all_20260809.csv) is the round-4 universe source;
the July board is left in place for reproducing rounds 1-3.
"""

import io
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"

LONG_SEASONS = range(1999, 2026)     # full nflverse stats_player coverage
ADP_BOARD = DATA / "adp" / "adp_ppr_2026_all_20260809.csv"

session = requests.Session()
session.headers["User-Agent"] = "wr-valuation-project/0.4"


def get_csv(url: str, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)
    r = session.get(url, timeout=300)
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


# ------------------------------------------------------- 1. RB universe
print("== 1. RB top-30 from the 2026-08-09 ADP board ==")
adp = pd.read_csv(ADP_BOARD)
rb30 = adp[adp["position"] == "RB"].nsmallest(30, "adp").reset_index(drop=True)
rb30.insert(0, "rb_adp_rank", rb30.index + 1)
rb30.to_csv(DATA / "adp" / "rb_top30_adp_2026.csv", index=False)
print(rb30[["rb_adp_rank", "name", "team", "adp", "stdev", "times_drafted"]].to_string(index=False))
rb30["norm"] = rb30["name"].map(norm_name)

# ---------------------------------------------------- 2. RB metadata
print("\n== 2. metadata join ==")
meta = get_csv(f"{NFLVERSE}/players/players.csv", DATA / "meta" / "players_meta.csv")
meta_rb = meta[(meta["position"] == "RB") & (meta["last_season"] >= 2024)].copy()
meta_rb["norm"] = meta_rb["display_name"].map(norm_name)
meta_rb = meta_rb.sort_values("last_season", ascending=False).drop_duplicates("norm")

cols = ["norm", "gsis_id", "display_name", "birth_date", "rookie_season", "last_season",
        "years_of_experience", "draft_year", "draft_round", "draft_pick", "draft_team",
        "height", "weight", "college_name", "ngs_position"]
matched = rb30.merge(meta_rb[cols], on="norm", how="left")
unmatched = matched[matched["gsis_id"].isna()]
if len(unmatched):
    print("!! unmatched vs nflverse metadata:", unmatched["name"].tolist())
matched.to_csv(DATA / "meta" / "rb_top30_meta.csv", index=False)
print(f"metadata matched for {matched['gsis_id'].notna().sum()}/30 RBs")
rb_ids = set(matched["gsis_id"].dropna())

# --------------------------- 3. weekly stats, extended back to 1999
print(f"\n== 3. weekly player stats {LONG_SEASONS.start}-{LONG_SEASONS.stop - 1} ==")
rb_frames, panel_frames = [], []
for yr in LONG_SEASONS:
    df = get_csv(f"{NFLVERSE}/stats_player/stats_player_week_{yr}.csv",
                 DATA / "players" / "weekly_raw" / f"stats_player_week_{yr}.csv")
    id_col = "player_id" if "player_id" in df.columns else "gsis_id"
    df = df.rename(columns={id_col: "gsis_id"})
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]

    sub = df[df["gsis_id"].isin(rb_ids)].copy()
    rb_frames.append(sub)

    pos_col = next((c for c in ("position", "player_position") if c in df.columns), None)
    keep = df[df[pos_col].isin(["WR", "RB"])] if pos_col else df
    panel_cols = [c for c in ["gsis_id", "player_display_name", "player_name", "season",
                              "week", "team", "recent_team", pos_col, "targets",
                              "receptions", "receiving_yards", "receiving_tds", "carries",
                              "rushing_yards", "rushing_tds", "fantasy_points_ppr"]
                  if c and c in keep.columns]
    panel_frames.append(keep[panel_cols].copy())
    print(f"  {yr}: {len(df):>6} rows, {len(sub):>4} for board RBs, {len(keep):>5} WR/RB rows")

rb_weekly = pd.concat(rb_frames, ignore_index=True)
rb_weekly = rb_weekly.merge(matched[["gsis_id", "name", "rb_adp_rank"]], on="gsis_id", how="left")
rb_weekly.to_csv(DATA / "players" / "rb_top30_weekly.csv", index=False)

by_dir = DATA / "players" / "by_player_rb"
by_dir.mkdir(parents=True, exist_ok=True)
for name, grp in rb_weekly.groupby("name"):
    grp.to_csv(by_dir / f"{slug(name)}.csv", index=False)

cov = (rb_weekly.groupby("name")["season"]
       .agg(first="min", last="max", games="count").sort_values("first"))
print("\nboard-RB coverage (first/last season, game rows):")
print(cov.to_string())
missing = set(matched.loc[matched["gsis_id"].notna(), "name"]) - set(cov.index)
if missing:
    print("!! matched RBs with NO weekly rows:", sorted(missing))

# ------------------------------------------- 4. long age panel (WR+RB)
print("\n== 4. long WR/RB player-season panel with age ==")
panel = pd.concat(panel_frames, ignore_index=True)
pos_col = next(c for c in ("position", "player_position") if c in panel.columns)
name_col = "player_display_name" if "player_display_name" in panel.columns else "player_name"

agg = (panel.groupby(["gsis_id", "season"])
       .agg(name=(name_col, "first"),
            position=(pos_col, "first"),
            games=("week", "nunique"),
            targets=("targets", "sum"),
            receptions=("receptions", "sum"),
            rec_yards=("receiving_yards", "sum"),
            carries=("carries", "sum") if "carries" in panel.columns else ("targets", "sum"),
            rush_yards=("rushing_yards", "sum") if "rushing_yards" in panel.columns else ("targets", "sum"),
            ppr=("fantasy_points_ppr", "sum"))
       .reset_index())
agg["ppg"] = agg["ppr"] / agg["games"]
agg["touches"] = agg["targets"].fillna(0) + agg.get("carries", 0).fillna(0)

bd = meta[["gsis_id", "birth_date", "rookie_season", "draft_year", "draft_round",
           "draft_pick", "display_name"]].drop_duplicates("gsis_id")
agg = agg.merge(bd, on="gsis_id", how="left")
agg["birth_date"] = pd.to_datetime(agg["birth_date"], errors="coerce")
# Age on Sept 1 of the season year — a fixed within-season reference point.
ref = pd.to_datetime(agg["season"].astype(str) + "-09-01")
agg["age"] = (ref - agg["birth_date"]).dt.days / 365.25
agg["exp"] = agg["season"] - agg["rookie_season"]

(DATA / "derived").mkdir(parents=True, exist_ok=True)
agg.to_csv(DATA / "derived" / "age_panel_long.csv", index=False)
print(f"panel: {len(agg)} player-seasons, {agg['gsis_id'].nunique()} players, "
      f"seasons {agg['season'].min()}-{agg['season'].max()}, "
      f"age known for {agg['age'].notna().mean():.1%}")
print(agg.groupby("position")["ppg"].describe().to_string())

# ------------------------------------------------ 5. team weekly, extended
print("\n== 5. team weekly stats, extended ==")
for yr in LONG_SEASONS:
    p = DATA / "teams" / f"stats_team_week_{yr}.csv"
    if p.exists():
        continue
    df = get_csv(f"{NFLVERSE}/stats_team/stats_team_week_{yr}.csv", p)
    print(f"  {yr}: {len(df)} team-week rows")

print("\ndone.")

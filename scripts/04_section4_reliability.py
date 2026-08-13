#!/usr/bin/env python3
"""
Section 4 of EDA_PLAN.md: stat reliability gate.

Pre-specified (before any results):
  - Primary sample: ALL players with position == 'WR' in data/players/weekly_raw/
    stats_player_week_{2014..2025}.csv, REG games only, player-games with targets <= 1
    excluded (the §0 game-inclusion rule from section1_notes.md, reused verbatim).
  - Split-half: within each WR player-season with >= 10 included games, each stat is
    computed separately on odd vs even calendar weeks; ratio stats are computed as
    (sum numerator)/(sum denominator) per half, never means of per-game ratios.
    r_half = Pearson correlation across player-seasons; rho_full = 2 r/(1+r) (eq. 8).
  - Year-over-year: season-level stat (same sum/sum construction over all included
    games) for consecutive seasons within player, both seasons >= 8 included games.
    r_YoY = Pearson correlation across pairs.
  - Bootstrap 95% CIs: resample PLAYERS with replacement (cluster bootstrap), 2000 reps,
    percentile intervals, for both rho_full and r_YoY.
  - Admission rule: rho_full >= 0.5 AND r_YoY bootstrap CI excludes 0.
  - Sensitivities (reported, not gating): top-30 universe only; 2021-2025 window only.

Stats screened: target_share, air_yards_share, wopr, aDOT, racr, yards/target,
TD/target, receiving_epa per game, PPR PPG.

Share denominators (team pass attempts, team passing air yards) come from
data/teams/stats_team_week_{year}.csv joined on season+week+team, so that the
sum/sum construction is exact rather than a mean of per-game shares.

Outputs: results/reliability_table.csv (long format: sample x stat), console log.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YEARS = range(2014, 2026)
RNG = np.random.default_rng(20260714)
N_BOOT = 2000

STATS = ["target_share", "air_yards_share", "wopr", "adot", "racr",
         "yards_per_target", "td_per_target", "rec_epa_pg", "ppr_ppg"]


def load_games() -> pd.DataFrame:
    """All WR regular-season player-games passing the §0 inclusion rule, with team
    pass attempts / team air yards joined for exact share denominators."""
    frames = []
    for y in YEARS:
        df = pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{y}.csv",
                         low_memory=False,
                         usecols=["player_id", "player_display_name", "position",
                                  "season", "week", "season_type", "team",
                                  "targets", "receptions", "receiving_yards",
                                  "receiving_tds", "receiving_air_yards",
                                  "receiving_yards_after_catch", "receiving_epa",
                                  "fantasy_points_ppr"])
        df = df[(df["position"] == "WR") & (df["season_type"] == "REG")
                & (df["targets"] >= 2)]
        tm = pd.read_csv(ROOT / f"data/teams/stats_team_week_{y}.csv",
                         usecols=["season", "week", "team", "season_type",
                                  "attempts", "passing_air_yards"])
        tm = tm[tm["season_type"] == "REG"].rename(
            columns={"attempts": "team_pass_att",
                     "passing_air_yards": "team_pass_air_yards"})
        df = df.merge(tm.drop(columns="season_type"),
                      on=["season", "week", "team"], how="left", validate="m:1")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    assert out["team_pass_att"].notna().all(), "unmatched team-week rows"
    return out


def agg_stats(g: pd.DataFrame) -> dict:
    """Stat vector on a set of games: ratio stats as sum/sum, per-game stats as means.
    Degenerate denominators -> NaN (pairwise-dropped downstream)."""
    tgt = g["targets"].sum()
    ray = g["receiving_air_yards"].sum()
    ts = g["targets"].sum() / g["team_pass_att"].sum()
    ays = ray / g["team_pass_air_yards"].sum() if g["team_pass_air_yards"].sum() > 0 else np.nan
    return dict(
        target_share=ts,
        air_yards_share=ays,
        wopr=1.5 * ts + 0.7 * ays if not np.isnan(ays) else np.nan,
        adot=ray / tgt if tgt > 0 else np.nan,
        racr=g["receiving_yards"].sum() / ray if ray > 0 else np.nan,
        yards_per_target=g["receiving_yards"].sum() / tgt if tgt > 0 else np.nan,
        td_per_target=g["receiving_tds"].sum() / tgt if tgt > 0 else np.nan,
        rec_epa_pg=g["receiving_epa"].mean(),
        ppr_ppg=g["fantasy_points_ppr"].mean(),
        n_games=len(g),
    )


def build_half_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per player-season with >= 10 included games: stat_A (odd weeks),
    stat_B (even weeks)."""
    rows = []
    for (pid, season), g in df.groupby(["player_id", "season"]):
        if len(g) < 10:
            continue
        odd, even = g[g["week"] % 2 == 1], g[g["week"] % 2 == 0]
        a, b = agg_stats(odd), agg_stats(even)
        row = {"player_id": pid, "season": season,
               "n_odd": len(odd), "n_even": len(even)}
        for s in STATS:
            row[f"{s}_A"], row[f"{s}_B"] = a[s], b[s]
        rows.append(row)
    return pd.DataFrame(rows)


def build_season_stats(df: pd.DataFrame, min_games: int = 8) -> pd.DataFrame:
    rows = []
    for (pid, season), g in df.groupby(["player_id", "season"]):
        if len(g) < min_games:
            continue
        d = agg_stats(g)
        d.update(player_id=pid, season=season)
        rows.append(d)
    return pd.DataFrame(rows)


def build_yoy_pairs(seas: pd.DataFrame) -> pd.DataFrame:
    nxt = seas.copy()
    nxt["season"] = nxt["season"] - 1  # align season s+1 onto season s
    m = seas.merge(nxt, on=["player_id", "season"], suffixes=("_s", "_s1"))
    return m


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def spearman_brown(r: float) -> float:
    return 2 * r / (1 + r) if not np.isnan(r) else np.nan


def boot_ci(pair_df: pd.DataFrame, cols: tuple, transform=None,
            n_boot: int = N_BOOT) -> tuple:
    """Cluster bootstrap over players of a correlation between two columns."""
    groups = {pid: g[list(cols)].to_numpy(float)
              for pid, g in pair_df.groupby("player_id")}
    pids = np.array(list(groups.keys()))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        take = RNG.choice(len(pids), size=len(pids), replace=True)
        arr = np.vstack([groups[pids[i]] for i in take])
        r = pearson(arr[:, 0], arr[:, 1])
        stats[b] = transform(r) if transform else r
    stats = stats[~np.isnan(stats)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def analyze(df: pd.DataFrame, sample: str) -> pd.DataFrame:
    halves = build_half_pairs(df)
    seas = build_season_stats(df, min_games=8)
    yoy = build_yoy_pairs(seas)
    print(f"[{sample}] split-half player-seasons: {len(halves)} "
          f"({halves['player_id'].nunique()} players); "
          f"YoY pairs: {len(yoy)} ({yoy['player_id'].nunique()} players)")
    rows = []
    for s in STATS:
        r_half = pearson(halves[f"{s}_A"].to_numpy(float),
                         halves[f"{s}_B"].to_numpy(float))
        rho_full = spearman_brown(r_half)
        rho_lo, rho_hi = boot_ci(halves, (f"{s}_A", f"{s}_B"),
                                 transform=spearman_brown)
        r_yoy = pearson(yoy[f"{s}_s"].to_numpy(float),
                        yoy[f"{s}_s1"].to_numpy(float))
        yoy_lo, yoy_hi = boot_ci(yoy, (f"{s}_s", f"{s}_s1"))
        n_half = int((~(halves[f"{s}_A"].isna() | halves[f"{s}_B"].isna())).sum())
        n_yoy = int((~(yoy[f"{s}_s"].isna() | yoy[f"{s}_s1"].isna())).sum())
        admit = (rho_full >= 0.5) and (yoy_lo > 0 or yoy_hi < 0)
        rows.append(dict(sample=sample, stat=s, n_half_pairs=n_half,
                         r_half=r_half, rho_full=rho_full,
                         rho_full_lo=rho_lo, rho_full_hi=rho_hi,
                         n_yoy_pairs=n_yoy, r_yoy=r_yoy,
                         r_yoy_lo=yoy_lo, r_yoy_hi=yoy_hi,
                         verdict=("ADMIT" if admit else "REJECT")
                                 if sample == "primary" else ""))
    return pd.DataFrame(rows)


def main():
    df = load_games()
    print(f"included WR player-games 2014-2025: {len(df)}; "
          f"players: {df['player_id'].nunique()}")

    top30 = pd.read_csv(ROOT / "data/meta/wr_top30_meta.csv")["gsis_id"].tolist()

    out = pd.concat([
        analyze(df, "primary"),
        analyze(df[df["player_id"].isin(top30)], "top30_only"),
        analyze(df[df["season"] >= 2021], "window_2021_2025"),
    ], ignore_index=True)

    out.round(4).to_csv(ROOT / "results/reliability_table.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

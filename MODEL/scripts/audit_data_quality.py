"""
Audit 1: panel data quality, merge integrity, missingness, outliers, era breaks.

Read-only. Writes results/audit_data_quality.txt
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, "..", "data", "team_game_panel.csv")
GAMES = os.path.join(HERE, "..", "data", "games", "games.csv")
MT = os.path.join(HERE, "..", "data", "model_table_v2.csv")
RES = os.path.join(HERE, "..", "results")

TEAM_FIXES = {"STL": "LA", "SD": "LAC", "OAK": "LV", "LAR": "LA", "JAC": "JAX"}

out = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)


def main():
    p = pd.read_csv(PANEL)
    g = pd.read_csv(GAMES)

    P("=" * 78)
    P("A. PANEL / SCHEDULE RECONCILIATION")
    P("=" * 78)
    gg = g[(g.game_type == "REG") & g.result.notna() & (g.season >= 2006)].copy()
    for c in ("home_team", "away_team"):
        gg[c] = gg[c].replace(TEAM_FIXES)
    preg = p[p.season_type == "REG"]
    P(f"schedule REG games with result 2006-2025 : {len(gg)}")
    P(f"panel REG games                          : {preg.game_id.nunique()}")
    missing = set(gg.game_id) - set(preg.game_id)
    extra = set(preg.game_id) - set(gg.game_id)
    P(f"in schedule not in panel: {len(missing)}  {sorted(missing)[:10]}")
    P(f"in panel not in schedule: {len(extra)}  {sorted(extra)[:10]}")

    # rows per game
    rpg = p.groupby("game_id").size().value_counts().to_dict()
    P(f"rows per game distribution: {rpg}")
    bad = p.groupby("game_id").size()
    P(f"games without exactly 2 rows: {list(bad[bad != 2].index)[:10]}")

    # team identity check: is_home / opponent consistency
    chk = p.merge(p[["game_id", "team", "off_plays", "off_epa"]],
                  left_on=["game_id", "opponent"], right_on=["game_id", "team"],
                  suffixes=("", "_opp"))
    d1 = (chk.def_plays - chk.off_plays_opp).abs()
    d2 = (chk.def_epa - chk.off_epa_opp).abs()
    P(f"\nmirror check: max |def_plays - opp off_plays| = {d1.max()}")
    P(f"mirror check: max |def_epa   - opp off_epa  | = {d2.max():.3e}")
    P("(should be exactly 0: a team's defence is its opponent's offence)")

    P("")
    P("=" * 78)
    P("B. MISSINGNESS (panel, REG only)")
    P("=" * 78)
    miss = preg.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    P(miss.to_string() if len(miss) else "none")
    for c in ["off_sec_per_play", "off_cpoe", "off_rush_epa", "off_pass_epa"]:
        m = preg[preg[c].isna()]
        if len(m):
            P(f"\n{c}: {len(m)} missing; by season {m.groupby('season').size().to_dict()}")

    P("")
    P("=" * 78)
    P("C. CONSTANT / DEGENERATE COLUMNS (within season)")
    P("=" * 78)
    for c in p.columns:
        if p[c].dtype.kind not in "if":
            continue
        nun = p.groupby("season")[c].nunique()
        if nun.max() <= 2:
            P(f"  !! {c}: max distinct values within a season = {nun.max()}")

    P("")
    P("=" * 78)
    P("D. DISTRIBUTIONS / OUTLIERS (REG)")
    P("=" * 78)
    cols = ["off_plays", "off_epa", "off_sr", "off_epa_noto", "off_early_epa",
            "off_pass_epa", "off_rush_epa", "off_cpoe", "off_pass_oe",
            "off_sec_per_play", "off_neutral_plays", "off_turnovers"]
    P(preg[cols].describe(percentiles=[.001, .01, .5, .99, .999]).T.round(3).to_string())

    P("\nlowest off_plays team-games:")
    P(preg.nsmallest(5, "off_plays")[["game_id", "team", "off_plays", "off_epa"]].to_string(index=False))
    P("\nextreme off_epa:")
    P(preg.nsmallest(3, "off_epa")[["game_id", "team", "off_plays", "off_epa"]].to_string(index=False))
    P(preg.nlargest(3, "off_epa")[["game_id", "team", "off_plays", "off_epa"]].to_string(index=False))

    P("")
    P("=" * 78)
    P("E. ERA / SEASON-LEVEL DRIFT")
    P("=" * 78)
    era = preg.groupby("season").agg(
        n=("team", "size"),
        plays=("off_plays", "mean"),
        epa=("off_epa", "mean"),
        sr=("off_sr", "mean"),
        secpp=("off_sec_per_play", "mean"),
        neutral=("off_neutral_plays", "mean"),
        pass_oe=("off_pass_oe", "mean"),
        to=("off_turnovers", "mean"),
    ).round(3)
    P(era.to_string())
    P("\nmax week by season (17-game era from 2021):")
    P(preg.groupby("season").week.max().to_dict())

    P("")
    P("=" * 78)
    P("F. MODEL TABLE MERGE")
    P("=" * 78)
    if os.path.exists(MT):
        m = pd.read_csv(MT)
        P(f"model_table_v2 rows {len(m)}, seasons {m.season.min()}-{m.season.max()}")
        sched = gg[gg.season >= m.season.min()]
        P(f"schedule REG games same window: {len(sched)}")
        lost = set(sched.game_id) - set(m.game_id)
        P(f"games lost in model table: {len(lost)}")
        if lost:
            ls = gg[gg.game_id.isin(lost)]
            P(ls.groupby(["season", "week"]).size().to_string())
        P(f"\nhome_win mean: {m.home_win.mean():.4f}; ties (0.5): {(m.home_win == 0.5).sum()}")
        P(f"missing spread_home: {m.spread_home.isna().sum()}")
        nas = m.isna().sum()
        nas = nas[nas > 0]
        P("NaN columns in model table:\n" + (nas.to_string() if len(nas) else "none"))

    with open(os.path.join(RES, "audit_data_quality.txt"), "w") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

"""§W1.4 — the §4 reliability gate, re-run on the wider panel.

Split-half (odd/even weeks) -> Spearman-Brown rho_full; year-over-year r with a
player-clustered bootstrap.  Admission: rho_full >= 0.5 AND r_YoY 95% CI excludes 0.
Season-level advanced stats cannot be split within season; they are gated on r_YoY
alone and flagged (declared exception, results/sectionW1_notes.md W1.4).

Outputs: results/sectionW1_gate.csv
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
ROOT = Path("/Users/thomasmcnamee/NFL")
RNG = np.random.default_rng(20260824)
NBOOT = 2000

WCOLS = ["player_id", "position", "season", "week", "season_type", "team",
         "targets", "receptions", "carries", "receiving_yards", "receiving_tds",
         "receiving_air_yards", "receiving_yards_after_catch", "receiving_epa",
         "rushing_yards", "rushing_tds", "fantasy_points_ppr"]

frames = []
for y in range(2006, 2026):
    d = pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{y}.csv",
                    usecols=lambda c: c in WCOLS, low_memory=False)
    frames.append(d[d.season_type == "REG"])
wk = pd.concat(frames, ignore_index=True)
for c in ["targets", "receptions", "carries", "receiving_yards", "receiving_tds",
          "receiving_air_yards", "receiving_yards_after_catch", "receiving_epa",
          "rushing_yards", "rushing_tds"]:
    wk[c] = wk[c].fillna(0.0)
wk["touches"] = wk.carries + wk.targets

# team-week denominators (FULL team basis: sum over all team games)
tm = (wk.groupby(["team", "season", "week"])
      .agg(tm_tgt=("targets", "sum"), tm_ay=("receiving_air_yards", "sum"),
           tm_car=("carries", "sum")).reset_index())
wk = wk.merge(tm, on=["team", "season", "week"], how="left")

# ---------------------------------------------------------------- stat builders
def agg(df):
    """Aggregate a set of player-weeks into the candidate rate stats."""
    g = df.groupby(["player_id", "season"])
    a = g.agg(G=("fantasy_points_ppr", "size"),
              tgt=("targets", "sum"), rec=("receptions", "sum"),
              car=("carries", "sum"), tch=("touches", "sum"),
              ry=("receiving_yards", "sum"), rtd=("receiving_tds", "sum"),
              ay=("receiving_air_yards", "sum"),
              yac=("receiving_yards_after_catch", "sum"),
              repa=("receiving_epa", "sum"),
              rushy=("rushing_yards", "sum"), rushtd=("rushing_tds", "sum"),
              ppr=("fantasy_points_ppr", "sum"),
              tm_tgt=("tm_tgt", "sum"), tm_ay=("tm_ay", "sum"),
              tm_car=("tm_car", "sum")).reset_index()
    e = 1e-9
    out = pd.DataFrame({"player_id": a.player_id, "season": a.season, "G": a.G})
    out["targets_pg"] = a.tgt / a.G
    out["rec_pg"] = a.rec / a.G
    out["carries_pg"] = a.car / a.G
    out["touches_pg"] = a.tch / a.G
    out["rush_yards_pg"] = a.rushy / a.G
    out["rec_yards_pg"] = a.ry / a.G
    out["ppr_pg"] = a.ppr / a.G
    out["target_share_full"] = a.tgt / (a.tm_tgt + e)
    out["air_yards_share_full"] = a.ay / (a.tm_ay + e)
    out["carry_share_full"] = a.car / (a.tm_car + e)
    out["wopr_full"] = 1.5 * out.target_share_full + 0.7 * out.air_yards_share_full
    out["ypt"] = a.ry / (a.tgt + e)
    out["ypr"] = a.ry / (a.rec + e)
    out["adot"] = a.ay / (a.tgt + e)
    out["yac_per_rec"] = a.yac / (a.rec + e)
    out["td_per_tgt"] = a.rtd / (a.tgt + e)
    out["catch_rate"] = a.rec / (a.tgt + e)
    out["racr_w"] = np.clip(a.ry / np.where(a.ay.abs() < 1, np.nan, a.ay), 0, 3)
    out["rec_epa_per_tgt"] = a.repa / (a.tgt + e)
    out["ypc"] = a.rushy / (a.car + e)
    out["rush_td_per_car"] = a.rushtd / (a.car + e)
    return out


STATS_WR = ["targets_pg", "rec_pg", "rec_yards_pg", "target_share_full",
            "air_yards_share_full", "wopr_full", "ppr_pg", "ypt", "ypr", "adot",
            "yac_per_rec", "td_per_tgt", "catch_rate", "racr_w", "rec_epa_per_tgt"]
STATS_RB = ["carries_pg", "touches_pg", "targets_pg", "rec_pg", "rush_yards_pg",
            "carry_share_full", "target_share_full", "ppr_pg", "ypc",
            "rush_td_per_car", "ypt", "catch_rate", "rec_epa_per_tgt"]

SCREEN = {"WR": ("targets", 3.0), "TE": ("targets", 2.0), "RB": ("touches", 4.0)}


def gate_basic(pos, seasons, stats):
    p = wk[(wk.position == pos) & wk.season.isin(seasons)]
    relc, thr = SCREEN[pos]
    full = agg(p)
    rel = p.groupby(["player_id", "season"])[relc].mean().rename("rel").reset_index()
    full = full.merge(rel, on=["player_id", "season"])
    qual = full[(full.rel >= thr) & (full.G >= 8)][["player_id", "season"]]
    full = full.merge(qual, on=["player_id", "season"])

    odd = agg(p[p.week % 2 == 1]).merge(qual, on=["player_id", "season"])
    evn = agg(p[p.week % 2 == 0]).merge(qual, on=["player_id", "season"])
    odd = odd[odd.G >= 3]
    evn = evn[evn.G >= 3]
    H = odd.merge(evn, on=["player_id", "season"], suffixes=("_a", "_b"))

    nxt = full.copy()
    nxt["season"] = nxt.season - 1
    Y = full.merge(nxt, on=["player_id", "season"], suffixes=("", "_n"))

    rows = []
    for s in stats:
        a, b = H[f"{s}_a"].values, H[f"{s}_b"].values
        m = np.isfinite(a) & np.isfinite(b)
        rh = float(np.corrcoef(a[m], b[m])[0, 1])
        rfull = 2 * rh / (1 + rh) if rh > -1 else np.nan

        x, y = Y[s].values, Y[f"{s}_n"].values
        pid = Y.player_id.values
        m2 = np.isfinite(x) & np.isfinite(y)
        x, y, pid = x[m2], y[m2], pid[m2]
        ryoy = float(np.corrcoef(x, y)[0, 1])
        # player-clustered bootstrap
        players = np.unique(pid)
        idx = {p_: np.where(pid == p_)[0] for p_ in players}
        bs = np.empty(NBOOT)
        for k in range(NBOOT):
            draw = RNG.choice(players, size=len(players), replace=True)
            sel = np.concatenate([idx[d] for d in draw])
            bs[k] = np.corrcoef(x[sel], y[sel])[0, 1]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(pos=pos, stat=s, kind="weekly", n_half=int(m.sum()),
                         n_yoy=int(m2.sum()), r_half=rh, rho_full=rfull,
                         r_yoy=ryoy, yoy_lo=lo, yoy_hi=hi,
                         admit=bool((rfull >= 0.5) and (lo > 0 or hi < 0)),
                         gate="rho_full>=.5 & YoY CI excl 0"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- advanced (season-level)
ADV_WR = ["snap_share", "pass_snap_share", "routes_proxy_pg", "tprr_proxy",
          "yprr_proxy", "rz_targets_pg", "i10_targets_pg", "ez_targets_pg",
          "third_down_targets_pg", "deep_targets_pg", "deep_target_rate",
          "rz_target_share_of_own", "ngs_avg_separation", "ngs_avg_cushion",
          "ngs_avg_yac_above_expectation", "ngs_percent_share_of_intended_air_yards",
          "pfr_adot", "pfr_ybc_per_rec", "pfr_yac_per_rec", "pfr_drop_pct",
          "pfr_rec_per_broken_tackle", "target_epa"]
ADV_RB = ["snap_share", "pass_snap_share", "run_snap_share", "routes_proxy",
          "tprr_proxy", "gl5_carries_pg", "gl10_carries_pg", "third_down_carries_pg",
          "explosive_run_rate", "stuffed_rate", "short_yd_conv_rate",
          "rush_epa_per_att", "ngs_ryoe_per_att", "ngs_efficiency",
          "ngs_percent_attempts_gte_eight_defenders", "pfr_ybc_per_att",
          "pfr_yac_per_att", "pfr_att_per_broken_tackle", "opportunity_pg",
          "rz_targets_pg", "target_epa"]


def gate_adv(pos, file, stats):
    d = pd.read_csv(ROOT / file, low_memory=False)
    d = d[(d.position == pos) & (d.games >= 8)]
    relc, thr = SCREEN[pos]
    if pos == "RB":
        d = d[(d.touches / d.games) >= thr]
    else:
        d = d[(d.targets / d.games) >= thr]
    nxt = d.copy()
    nxt["season"] = nxt.season - 1
    Y = d.merge(nxt, on=["player_id", "season"], suffixes=("", "_n"))
    rows = []
    for s in stats:
        if s not in d.columns:
            continue
        x, y = Y[s].values.astype(float), Y[f"{s}_n"].values.astype(float)
        pid = Y.player_id.values
        m = np.isfinite(x) & np.isfinite(y)
        x, y, pid = x[m], y[m], pid[m]
        if len(x) < 40 or np.std(x) == 0:
            rows.append(dict(pos=pos, stat=s, kind="season", n_half=0, n_yoy=int(len(x)),
                             r_half=np.nan, rho_full=np.nan, r_yoy=np.nan,
                             yoy_lo=np.nan, yoy_hi=np.nan, admit=False,
                             gate="insufficient n"))
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        players = np.unique(pid)
        idx = {p_: np.where(pid == p_)[0] for p_ in players}
        bs = np.empty(NBOOT)
        for k in range(NBOOT):
            draw = RNG.choice(players, size=len(players), replace=True)
            sel = np.concatenate([idx[dd] for dd in draw])
            bs[k] = np.corrcoef(x[sel], y[sel])[0, 1]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(pos=pos, stat=s, kind="season", n_half=0, n_yoy=int(len(x)),
                         r_half=np.nan, rho_full=np.nan, r_yoy=r, yoy_lo=lo, yoy_hi=hi,
                         admit=bool(lo > 0 or hi < 0),
                         gate="YoY CI excl 0 only (declared exception)"))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = []
    # harness check: §4 replication window, WR 2014-2025
    rep = gate_basic("WR", list(range(2014, 2026)), STATS_WR)
    rep["window"] = "2014-2025 (§4 replication)"
    out.append(rep)
    for pos, stats in [("WR", STATS_WR), ("TE", STATS_WR), ("RB", STATS_RB)]:
        g = gate_basic(pos, list(range(2006, 2026)), stats)
        g["window"] = "2006-2025 (wide)"
        out.append(g)
    out.append(gate_adv("WR", "data/derived/adv_wr_te.csv", ADV_WR).assign(
        window="2018-2025 (advanced)"))
    out.append(gate_adv("TE", "data/derived/adv_wr_te.csv", ADV_WR).assign(
        window="2018-2025 (advanced)"))
    out.append(gate_adv("RB", "data/derived/adv_rb.csv", ADV_RB).assign(
        window="2018-2025 (advanced)"))
    G = pd.concat(out, ignore_index=True)
    G.to_csv(ROOT / "results/sectionW1_gate.csv", index=False)
    pd.set_option("display.width", 200)
    for w in G.window.unique():
        for pos in G[G.window == w].pos.unique():
            s = G[(G.window == w) & (G.pos == pos)]
            print(f"\n=== {pos} | {w} | n_yoy={s.n_yoy.max()} ===")
            print(s[["stat", "rho_full", "r_yoy", "yoy_lo", "yoy_hi", "admit"]]
                  .round(3).to_string(index=False))

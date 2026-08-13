"""§M shared build: board->player join (reused from §L, extended to QB/TE) and the
weekly realized-points matrix.

The join logic is lifted verbatim from scripts/26_sectionL_conversion.py (which carries
three validated bug fixes: skill-position restriction, board-position-governs, ambiguity
resolution by activity) and only generalised from {RB,WR} to {QB,RB,WR,TE}.

Rerun: imported by 28_sectionM_scarcity.py and 29_sectionM_draftsim.py
"""
import re
import warnings
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
ROOT = "/Users/thomasmcnamee/NFL"
YEARS = list(range(2015, 2025))
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
ALIASES = {"hollywood brown": "marquise brown", "joshua palmer": "josh palmer"}
SKILL = {"QB", "RB", "WR", "TE", "FB", "HB"}
POSNS = ["QB", "RB", "WR", "TE"]


def norm_name(s):
    s = re.sub(r"[^a-z ]", "",
               str(s).lower().replace(".", " ").replace("-", " ").replace("'", ""))
    return " ".join(t for t in s.split() if t not in SUFFIXES)


def collapse_initials(n):
    toks = n.split()
    out, buf = [], ""
    for t in toks:
        if len(t) == 1:
            buf += t
        else:
            if buf:
                out.append(buf)
                buf = ""
            out.append(t)
    if buf:
        out.append(buf)
    return " ".join(out)


def load_weekly():
    frames = []
    for y in range(2014, 2026):
        df = pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
                         low_memory=False,
                         usecols=["player_id", "player_display_name", "position",
                                  "season", "week", "season_type", "fantasy_points_ppr"])
        frames.append(df[df.season_type == "REG"])
    wk = pd.concat(frames, ignore_index=True)
    return wk.dropna(subset=["player_id"])


def build_panel(wk):
    """Board rows 2015-2024, all of QB/RB/WR/TE, joined to gsis player_id."""
    wk = wk.copy()
    wk["nname"] = wk.player_display_name.map(norm_name)
    wk["nname_c"] = wk.nname.map(collapse_initials)
    pdir = (wk.groupby("player_id")
              .agg(nname=("nname", lambda s: s.iat[0]),
                   nname_c=("nname_c", lambda s: s.iat[0]),
                   pos=("position", lambda s: s.mode().iat[0])).reset_index())
    seasons_present = wk.groupby(["player_id", "season"]).size().rename("nrows").reset_index()
    meta = pd.read_csv(f"{ROOT}/data/meta/players_meta.csv", low_memory=False,
                       usecols=["gsis_id", "display_name", "position"]).dropna(subset=["gsis_id"])
    meta["nname"] = meta.display_name.map(norm_name).map(collapse_initials)

    rows, unmatched, ambig = [], [], []
    for y in YEARS:
        adp = pd.read_csv(f"{ROOT}/data/adp/historical/adp_ppr_{y}.csv")
        adp = adp[adp.position.isin(POSNS)].copy()
        adp["nname"] = adp.name.map(norm_name).replace(ALIASES).map(collapse_initials)
        pres = seasons_present[seasons_present.season.between(y - 1, y + 1)]
        for _, r in adp.iterrows():
            c = pdir[(pdir.nname_c == r.nname) | (pdir.nname == r.nname)]
            c = c[c.pos.isin(SKILL)]
            if len(c) > 1:
                c2 = c[c.pos == r.position]
                c = c2 if len(c2) else c
            if len(c) > 1:
                c2 = c[c.player_id.isin(pres.player_id)]
                c = c2 if len(c2) else c
            if len(c) > 1:
                n = (pres[pres.player_id.isin(c.player_id)]
                     .groupby("player_id").nrows.sum())
                ambig.append((y, r["name"], r.position, list(c.player_id)))
                c = c[c.player_id == n.idxmax()]
            base = dict(year=y, name=r["name"], pos=r.position, adp=r.adp,
                        stdev=r.stdev, team=r.team)
            if len(c) == 0:
                m = meta[(meta.nname == r.nname) & (meta.position == r.position)]
                if len(m):
                    rows.append({**base, "pid": m.gsis_id.iat[0]})
                    continue
                unmatched.append((y, r["name"], r.position))
                rows.append({**base, "pid": None})
                continue
            rows.append({**base, "pid": c.player_id.iat[0]})
    panel = pd.DataFrame(rows)
    return panel, unmatched, ambig


def weekly_matrix(wk, year, pids):
    """(n_players x 17) matrix of realized PPR by week; NaN = did not appear."""
    sub = wk[(wk.season == year) & (wk.player_id.isin(set(pids)))]
    m = (sub.pivot_table(index="player_id", columns="week",
                         values="fantasy_points_ppr", aggfunc="sum"))
    m = m.reindex(index=pids, columns=range(1, 18))
    return m

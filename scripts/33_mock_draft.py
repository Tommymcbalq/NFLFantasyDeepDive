"""Interactive 10-team snake mock draft from slot 5, scored on our board.

Opponents pick by ADP with noise (the §M-calibrated model) plus crude roster needs.
State persists in results/mock_draft_state.json so the draft can run across turns.

  python3 scripts/33_mock_draft.py --reset
  python3 scripts/33_mock_draft.py --advance          # sim opponents to my next pick, show board
  python3 scripts/33_mock_draft.py --take "Player"    # make my pick, then advance
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
STATE = R / "mock_draft_state.json"
TEAMS, MY_SLOT, ROUNDS = 10, 5, 14
LIMITS = {"QB": 2, "RB": 6, "WR": 6, "TE": 2, "PK": 1, "DEF": 1}


def my_picks():
    out = []
    for r in range(ROUNDS):
        p = r * TEAMS + (MY_SLOT if r % 2 == 0 else TEAMS - MY_SLOT + 1)
        out.append(p)
    return out


def load_pool():
    a = pd.read_csv(ROOT / "data/adp/adp_ppr_2026_all_20260809.csv")
    a = a[a.position.isin(["RB", "WR", "TE", "QB"])].copy()
    b = pd.read_csv(R / "board_2026_with_views.csv")[["name", "theta_bar"]]
    a = a.merge(b, on="name", how="left")
    a["stdev"] = a["stdev"].clip(lower=0.5)
    return a.sort_values("adp").reset_index(drop=True)


def state_load():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"picks": []}   # list of {pick, team, player, position}


def state_save(s):
    STATE.write_text(json.dumps(s, indent=1))


def sim_to(s, pool, target_pick, rng):
    taken = {p["player"] for p in s["picks"]}
    rosters = {}
    for p in s["picks"]:
        rosters.setdefault(p["team"], []).append(p["position"])
    nxt = len(s["picks"]) + 1
    while nxt < target_pick:
        rnd = (nxt - 1) // TEAMS
        idx = (nxt - 1) % TEAMS
        team = idx + 1 if rnd % 2 == 0 else TEAMS - idx
        avail = pool[~pool.name.isin(taken)]
        need = rosters.get(team, [])
        ok = avail[[need.count(p) < LIMITS.get(p, 3) for p in avail.position]]
        if not len(ok):
            ok = avail
        # ADP + noise, take the minimum draw
        draw = rng.normal(ok.adp.to_numpy(), ok.stdev.to_numpy())
        pick = ok.iloc[int(np.argmin(draw))]
        taken.add(pick["name"])
        rosters.setdefault(team, []).append(pick["position"])
        s["picks"].append({"pick": nxt, "team": int(team), "player": pick["name"],
                           "position": pick["position"]})
        nxt += 1
    return s


def show(s, pool, k=14):
    taken = {p["player"] for p in s["picks"]}
    avail = pool[~pool.name.isin(taken)].dropna(subset=["theta_bar"])
    avail = avail.sort_values("theta_bar", ascending=False)
    nxt = len(s["picks"]) + 1
    mine = [p for p in s["picks"] if p["team"] == MY_SLOT]
    print(f"\n=== ON THE CLOCK: pick {nxt} (round {(nxt-1)//TEAMS+1}) ===")
    if mine:
        print("my roster: " + ", ".join(f"{p['player']} ({p['position']})" for p in mine))
    recent = s["picks"][-9:]
    if recent:
        print("just went:  " + ", ".join(f"{p['player']}" for p in recent))
    print("\nbest available on our board:")
    v = avail.head(k)[["name", "position", "team", "adp", "theta_bar"]]
    v = v.rename(columns={"theta_bar": "value"})
    print(v.round(2).to_string(index=False))
    for pos in ("RB", "WR"):
        top = avail[avail.position == pos].head(4)
        print(f"  top {pos}: " + ", ".join(f"{r['name']} {r.theta_bar:.2f}" for _, r in top.iterrows()))
    # What the market says is best available, regardless of our board
    print("\nbest available by ADP (what the room is likely to take):")
    m = avail.sort_values("adp").head(10)[["name", "position", "adp", "theta_bar"]]
    m = m.rename(columns={"theta_bar": "our_value"})
    m["vs_board"] = ["+" if v >= avail.theta_bar.quantile(.75) else "" for v in m.our_value]
    print(m.round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--take")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    pool = load_pool()
    if a.reset and STATE.exists():
        STATE.unlink()
    s = state_load()
    if a.take:
        hits = pool[pool.name.str.lower().str.contains(a.take.lower(), regex=False)]
        if len(hits) != 1:
            raise SystemExit(f"ambiguous/unknown: {a.take} -> {list(hits.name)}")
        p = hits.iloc[0]
        s["picks"].append({"pick": len(s["picks"]) + 1, "team": MY_SLOT,
                           "player": p["name"], "position": p["position"]})
        print(f"YOU TAKE: {p['name']} ({p['position']}, ADP {p.adp})")
    nxt_mine = next(x for x in my_picks() if x > len(s["picks"]))
    s = sim_to(s, pool, nxt_mine, rng)
    state_save(s)
    show(s, pool)

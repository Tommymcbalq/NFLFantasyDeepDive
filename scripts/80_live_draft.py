"""LIVE draft assistant — polls the Sleeper draft and recomputes the board between picks.

Sleeper has no websocket, so this polls GET /v1/draft/<id>/picks. The documented limit is
1000 calls/min; at one call every few seconds we are ~4 orders of magnitude under it.

What it does each refresh:
  1. pull the picks made so far, mark those players gone
  2. work out who is on the clock and how many picks until YOUR next one
  3. re-simulate the remaining draft over the UNDRAFTED pool only, so availability is
     conditional on what has actually happened rather than on a pre-draft prior
  4. print your tiers, best available, and survival to your next pick

Availability here is P(rank among the undrafted >= picks_until_my_next_pick). That
conditioning is the whole point: a run at a position collapses the survival curve for the
rest of it immediately, which a pre-computed table cannot show.

Usage:
    python3 scripts/80_live_draft.py                 # live, refreshes until you quit
    python3 scripts/80_live_draft.py --once          # single snapshot
    python3 scripts/80_live_draft.py --draft <id>    # override draft id
"""
import argparse, json, sys, time, urllib.request
import numpy as np, pandas as pd

DRAFT_ID = "1389519993699328000"          # 2026 real draft, verified pre_draft
LEAGUE_ID = "1389519993686753280"
ME = "timmy2tufff"
BOARD = "results/board_LIVE_sleeper.csv"
OWNERSHIP = "data/drafts/pick_ownership_2026.csv"
N_SIM = 4000


def get(url):
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.load(r)


def load():
    b = pd.read_csv(BOARD)
    b['sleeper_id'] = b.sleeper_id.astype('Int64').astype(str)
    own = pd.read_csv(OWNERSHIP)
    return b, own


def availability(b, gone_mask, picks_until, n=N_SIM, seed=0):
    """P(still there when my next pick arrives), conditional on who is already gone.

    Only the undrafted pool is ranked. A player survives if fewer than `picks_until` other
    undrafted players are taken ahead of him.
    """
    live = ~gone_mask
    mu = b.mu_cal.to_numpy(float)[live]
    sig = b.sig_cal.to_numpy(float)[live]
    out = np.zeros(len(b))
    if picks_until <= 0 or live.sum() == 0:
        out[live] = 1.0
        return out
    rng = np.random.default_rng(seed)
    hit = np.zeros(live.sum())
    for _ in range(n):
        rk = np.argsort(np.argsort(rng.normal(mu, sig)))
        hit += rk >= picks_until
    out[live] = hit / n
    return out


def snapshot(b, own, draft_id, replay=None):
    picks = get(f"https://api.sleeper.app/v1/draft/{draft_id}/picks")
    if replay is not None:
        # rehearsal: truncate a completed draft to its first `replay` picks so the
        # conditional logic can be checked against a board state we already know.
        picks = sorted(picks, key=lambda x: x['pick_no'])[:replay]
    made = {str(p['player_id']) for p in picks if p.get('player_id')}
    n_made = len(picks)
    gone = b.sleeper_id.isin(made).to_numpy()

    mine = own[own.owner == ME].overall.tolist()
    nxt = next((p for p in mine if p > n_made), None)
    on_clock = own[own.overall == n_made + 1]
    who = on_clock.owner.iloc[0] if len(on_clock) else "-"

    print("=" * 74)
    print(f"picks made: {n_made}   on the clock: pick {n_made+1} -> {who}")
    if nxt is None:
        print("no picks left for you")
        return
    until = nxt - 1 - n_made
    rnd = own[own.overall == nxt].iloc[0]
    print(f"YOUR NEXT: overall {nxt} ({rnd.rnd}.{rnd.pick_in_rnd:02d})   {until} picks away")
    print("=" * 74)

    P = availability(b, gone, until)
    b = b.assign(p_now=P, gone=gone)
    live = b[~b.gone].copy()

    if until == 0:
        print("\n*** YOU ARE ON THE CLOCK ***")
        c = live[live.position.isin(['RB', 'WR'])].sort_values(['tier', 'order', 'BOARD'])
        print("\nBEST AVAILABLE BY YOUR TIERS")
        for _, r in c.head(10).iterrows():
            print(f"   {r.position}{int(r.tier)}  {r['name'][:22]:<23} board#{int(r.BOARD):<4}")
        tq = live[live.position.isin(['TE', 'QB'])].sort_values('BOARD').head(3)
        if len(tq):
            print("   market TE/QB: " + ", ".join(f"{r['name']} ({r.position})" for _, r in tq.iterrows()))
        return

    print("\nWILL THEY LAST TO YOUR PICK?   (survival, conditional on picks already made)")
    c = live[live.position.isin(['RB', 'WR']) & (live.tier <= 5)]
    c = c.sort_values(['tier', 'order', 'BOARD']).head(14)
    for _, r in c.iterrows():
        bar = "#" * int(round(r.p_now * 20))
        flag = "  <-- WILL NOT LAST" if r.p_now < .35 else ""
        print(f"   {r.position}{int(r.tier)}  {r['name'][:22]:<23}{r.p_now:>5.0%} |{bar:<20}|{flag}")

    # a tier is 'about to break' if its remaining members are unlikely to survive together
    print("\nTIER STATUS")
    for pos in ('RB', 'WR'):
        for t in sorted(live[live.position == pos].tier.dropna().unique()):
            g = live[(live.position == pos) & (live.tier == t)]
            if len(g) == 0 or t > 5:
                continue
            exp = g.p_now.sum()
            print(f"   {pos}{int(t)}: {len(g)} left, expect {exp:.1f} to reach you"
                  + ("   *** EMPTIES BEFORE YOU ***" if exp < 0.8 else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draft', default=DRAFT_ID)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--every', type=int, default=8)
    ap.add_argument('--replay', type=int, default=None)
    a = ap.parse_args()
    b, own = load()
    while True:
        try:
            snapshot(b, own, a.draft, a.replay)
        except Exception as e:
            print(f"  [poll failed: {e}]", file=sys.stderr)
        if a.once:
            break
        time.sleep(a.every)


if __name__ == '__main__':
    main()

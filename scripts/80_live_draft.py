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
import argparse, json, re, sys, time, urllib.request
import numpy as np, pandas as pd

DRAFT_ID = "1389519993699328000"          # 2026 real draft, verified pre_draft
LEAGUE_ID = "1389519993686753280"
ME = "timmy2tufff"
BOARD = "results/board_LIVE_sleeper.csv"
OWNERSHIP = "data/drafts/pick_ownership_2026.csv"
N_SIM = 4000



# ------------------------------------------------------------------ display
C = dict(rst="\033[0m", b="\033[1m", dim="\033[2m",
         red="\033[38;5;203m", grn="\033[38;5;114m", yel="\033[38;5;221m",
         blu="\033[38;5;75m",  mag="\033[38;5;177m", orn="\033[38;5;215m",
         cyn="\033[38;5;80m",  gry="\033[38;5;245m", wht="\033[38;5;255m",
         bgblu="\033[48;5;24m", bggrn="\033[48;5;22m", bgred="\033[48;5;52m")
POSC = {'RB': C['grn'], 'WR': C['blu'], 'TE': C['orn'], 'QB': C['mag']}
W = 78

def _pc(p):
    return C['grn'] if p >= .70 else (C['yel'] if p >= .35 else C['red'])

_ANSI = re.compile(r"\033\[[0-9;]*m")
def vlen(s):
    """visible width: ANSI escapes occupy no columns, so len() over-counts and
    every box border drawn from it comes out short."""
    return len(_ANSI.sub("", s))

def box_top(t=""):
    t = f" {t} " if t else ""
    pad = W - 2 - vlen(t)
    print(f"{C['cyn']}\u250c{t}{'\u2500'*max(pad,0)}\u2510{C['rst']}")

def box_bot():
    print(f"{C['cyn']}\u2514{'\u2500'*(W-2)}\u2518{C['rst']}")

def line(s="", raw_len=None):
    n = raw_len if raw_len is not None else len(s)
    print(f"{C['cyn']}\u2502{C['rst']}{s}{' '*max(W-2-n,0)}{C['cyn']}\u2502{C['rst']}")

def bar(p, width=22):
    fill = int(round(p*width))
    return f"{_pc(p)}{'\u2588'*fill}{C['gry']}{'\u2591'*(width-fill)}{C['rst']}"

def tag(pos, tier):
    t = f"{pos}{int(tier)}" if tier == tier else pos
    return f"{POSC.get(pos,C['wht'])}{C['b']}{t:<4}{C['rst']}"


def get(url):
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.load(r)


def resolve_draft(league_id, fallback):
    """Find the league's live draft. If the draft is recreated its id changes, so trusting a
    hard-coded constant is a real failure mode on the night."""
    try:
        ds = get(f"https://api.sleeper.app/v1/league/{league_id}/drafts")
        live = [d for d in ds if d.get('status') in ('drafting', 'paused', 'pre_draft')]
        if live:
            d = sorted(live, key=lambda x: x.get('created', 0))[-1]
            if d['draft_id'] != fallback:
                print(f"  [resolved draft_id {d['draft_id']} (status {d['status']})]")
            return d['draft_id']
    except Exception as e:
        print(f"  [could not resolve draft id, using default: {e}]", file=sys.stderr)
    return fallback


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


def snapshot(b, own, draft_id, replay=None, last_n=[-1]):
    picks = get(f"https://api.sleeper.app/v1/draft/{draft_id}/picks")
    if replay is not None:
        picks = sorted(picks, key=lambda x: x['pick_no'])[:replay]
    made = {str(p['player_id']) for p in picks if p.get('player_id')}
    n_made = len(picks)
    if n_made == last_n[0]:
        return
    last_n[0] = n_made
    print("\033[2J\033[H", end="")
    gone = b.sleeper_id.isin(made).to_numpy()

    mine_picks = own[own.owner == ME].overall.tolist()
    nxt = next((p for p in mine_picks if p > n_made), None)
    oc = own[own.overall == n_made + 1]
    who = oc.owner.iloc[0] if len(oc) else "-"
    is_me = (who == ME)

    # ---------------- header
    ttl = "ON THE CLOCK" if is_me else f"pick {n_made+1}"
    bg = C['bggrn'] if is_me else C['bgblu']
    hdr = f" DRAFT  \u2502  {n_made} made  \u2502  {ttl}: {who} "
    print(f"{bg}{C['b']}{C['wht']}{hdr}{' '*max(W-len(hdr),0)}{C['rst']}")

    if nxt is None:
        print(f"{C['gry']}  no picks left for you{C['rst']}"); return
    until = nxt - 1 - n_made
    rr = own[own.overall == nxt].iloc[0]
    col = C['grn'] if until == 0 else (C['yel'] if until <= 3 else C['gry'])
    print(f"  {C['b']}your next{C['rst']} {C['wht']}{rr.rnd}.{rr.pick_in_rnd:02d}{C['rst']}"
          f" (overall {nxt})   {col}{until} picks away{C['rst']}")

    # ---------------- recent + roster
    recent = sorted(picks, key=lambda x: x['pick_no'])[-6:]
    if recent:
        r = "  ".join(f"{C['gry']}{p['pick_no']}{C['rst']} "
                      f"{(p.get('metadata') or {}).get('last_name','')}" for p in recent)
        print(f"  {C['dim']}last:{C['rst']} {r}")
    id2 = dict(zip(b.sleeper_id, zip(b.name, b.position)))
    mine_ids = [str(p['player_id']) for p in picks if p.get('player_id')
                and own[own.overall == p['pick_no']].owner.eq(ME).any()]
    if mine_ids:
        r = "  ".join(f"{POSC.get(id2.get(i,('','?'))[1],C['wht'])}{id2.get(i,(i,''))[0]}{C['rst']}"
                      for i in mine_ids)
        print(f"  {C['b']}yours:{C['rst']} {r}")
    print()

    P = availability(b, gone, until)
    b = b.assign(p_now=P, gone=gone)
    live = b[~b.gone].copy()

    def inj(r):
        st = str(r.get('injury_status') or '')
        if st in ('IR', 'PUP', 'Out', 'Doubtful'):
            return f" {C['red']}{C['b']}[{st}]{C['rst']}"
        if st == 'Questionable':
            bp = str(r.get('injury_body_part') or '')
            return f" {C['yel']}[Q{'' if bp in ('Undisclosed','nan','') else ' '+bp[:12]}]{C['rst']}"
        return ""

    if until == 0:
        box_top(f"{C['b']}BEST AVAILABLE \u2014 YOUR TIERS{C['rst']}")
        c = live[live.position.isin(['RB','WR'])].sort_values(['tier','order','BOARD']).head(11)
        for _, r in c.iterrows():
            nm = r['name'][:24]
            body = (f"  {tag(r.position,r.tier)} {C['wht']}{nm:<25}{C['rst']}"
                    f"{C['gry']}board#{int(r.BOARD):<5}{C['rst']}{inj(r)}")
            print(f"{C['cyn']}\u2502{C['rst']}{body}{' '*max(W-2-vlen(body),0)}{C['cyn']}\u2502{C['rst']}")
        box_bot()
        tq = live[live.position.isin(['TE','QB'])].sort_values('BOARD').head(4)
        if len(tq):
            print(f"  {C['dim']}market TE/QB:{C['rst']} " + "  ".join(
                f"{POSC[r.position]}{r['name']}{C['rst']}" for _, r in tq.iterrows()))
        print(f"\n  {C['bggrn']}{C['b']}{C['wht']}  >>> PICK NOW  <<<  {C['rst']}\a")
        return

    box_top(f"{C['b']}SURVIVAL TO YOUR PICK{C['rst']}")
    c = live[live.position.isin(['RB','WR']) & (live.tier <= 5)]
    c = c.sort_values(['tier','order','BOARD']).head(13)
    for _, r in c.iterrows():
        nm = r['name'][:22]
        warn = f" {C['red']}{C['b']}GONE{C['rst']}" if r.p_now < .35 else ""
        body = (f"  {tag(r.position,r.tier)} {C['wht']}{nm:<23}{C['rst']}"
                f"{_pc(r.p_now)}{r.p_now*100:3.0f}%{C['rst']} {bar(r.p_now)}{warn}")
        print(f"{C['cyn']}\u2502{C['rst']}{body}{' '*max(W-2-vlen(body),0)}{C['cyn']}\u2502{C['rst']}")
    box_bot()

    print(f"\n  {C['b']}TIERS{C['rst']}")
    for pos in ('RB','WR'):
        parts = []
        for t in sorted(live[live.position == pos].tier.dropna().unique()):
            if t > 5: continue
            g = live[(live.position == pos) & (live.tier == t)]
            e = g.p_now.sum()
            col = C['red'] if e < 0.8 else (C['yel'] if e < 2 else C['grn'])
            parts.append(f"{col}{pos}{int(t)}:{len(g)}({e:.1f}){C['rst']}")
        if parts:
            print(f"    {POSC[pos]}{pos}{C['rst']}  " + "  ".join(parts))
    print(f"    {C['dim']}n(expected to reach you); red = tier empties first{C['rst']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draft', default=DRAFT_ID)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--every', type=int, default=8)
    ap.add_argument('--replay', type=int, default=None)
    a = ap.parse_args()
    b, own = load()
    if a.draft == DRAFT_ID and a.replay is None:
        a.draft = resolve_draft(LEAGUE_ID, DRAFT_ID)
    while True:
        try:
            snapshot(b, own, a.draft, a.replay)
        except Exception as e:
            print(f"  [poll failed: {e}]", file=sys.stderr)
        if a.once:
            break
        try:
            time.sleep(a.every)
        except KeyboardInterrupt:
            print("\nstopped."); break


if __name__ == '__main__':
    main()

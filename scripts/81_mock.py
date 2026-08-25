"""Mock draft driver — same panel as the live tool, but the room is simulated.

Stateless by design so it can be driven one pick at a time in conversation: pass the picks
you have made so far and it replays the room deterministically (fixed seed per mock) up to
your next pick, then renders.

    python3 scripts/81_mock.py                                   # opens at pick 5
    python3 scripts/81_mock.py --picks "Christian McCaffrey"     # -> your pick 16
    python3 scripts/81_mock.py --picks "A,B,C" --seed 3          # a different room

The room drafts by the calibrated slot scores (mu_cal, sig_cal), i.e. the same distribution
the live availability numbers come from -- so a mock here is a draw from exactly the belief
we would be pricing against on the night.
"""
import argparse, importlib.util, sys
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location("live", "scripts/80_live_draft.py")
L = importlib.util.module_from_spec(spec)
sys.modules["live"] = L
spec.loader.exec_module(L)
C, POSC, W = L.C, L.POSC, L.W

MY = [5, 16, 27, 34, 47, 54, 65, 74, 85, 87, 105, 116, 125, 136, 145]

# The owner's pick-5 belief is STRUCTURAL, not a marginal: picks 1-3 are Gibbs / Bijan /
# Chase in some order, and pick 4 is Nacua (80%) or McCaffrey (20%). Independent normal
# slot scores cannot represent "exactly these three, then one of those two" -- they let
# Bijan or Chase drift down to 5, which he has ruled out. So the opening is scripted.
OPENING = ['Jahmyr Gibbs', 'Bijan Robinson', "Ja'Marr Chase"]
P_NACUA_AT_4 = 0.80


def render(b, chosen, a):
    name = b.name.to_numpy()
    idx = {n.lower(): i for i, n in enumerate(name)}
    mu = b.mu_cal.to_numpy(float)
    sig = b.sig_cal.to_numpy(float)
    print("\033[2J\033[H", end="")
    rng = np.random.default_rng(a.seed)
    order = np.argsort(rng.normal(mu, sig))
    taken = np.zeros(len(b), bool)
    for i in chosen:
        taken[i] = True

    forced = {}
    op = list(OPENING)
    rng.shuffle(op)
    for k, nm in enumerate(op, start=1):
        forced[k] = idx[nm.lower()]
    forced[4] = idx['puka nacua'] if rng.random() < P_NACUA_AT_4 else idx['christian mccaffrey']

    last = MY[min(len(chosen), len(MY) - 1)]
    ptr = 0
    room = []
    mine_at = {}
    for pk in range(1, last + 1):
        if pk in MY:
            j = MY.index(pk)
            if j < len(chosen):
                mine_at[pk] = chosen[j]
            continue
        if pk in forced:
            i = forced[pk]
            if taken[i]:
                continue
        else:
            while ptr < len(order) and taken[order[ptr]]:
                ptr += 1
            if ptr >= len(order):
                break
            i = order[ptr]
        taken[i] = True
        room.append((pk, i))

    n_made = last - 1
    until = 0
    P = L.availability(b, taken, until)
    b2 = b.assign(p_now=P, gone=taken)
    live = b2[~b2.gone]

    hdr = f" MOCK (seed {a.seed})  │  {n_made} made  │  ON THE CLOCK: you "
    print(f"{C['bggrn']}{C['b']}{C['wht']}{hdr}{' '*max(W-len(hdr),0)}{C['rst']}")
    rr = {5: '1.05', 16: '2.06', 27: '3.07', 34: '4.04', 47: '5.07', 54: '6.04', 65: '7.05',
          74: '8.04', 85: '9.05', 87: '9.07', 105: '11.05', 116: '12.06', 125: '13.05',
          136: '14.06', 145: '15.05'}
    print(f"  {C['b']}your pick{C['rst']} {C['wht']}{rr.get(last,'?')}{C['rst']} (overall {last})")
    if room:
        r = "  ".join(f"{C['gry']}{p}{C['rst']} {name[i].split()[-1]}" for p, i in room[-7:])
        print(f"  {C['dim']}last:{C['rst']} {r}")
    if mine_at:
        r = "  ".join(f"{POSC.get(b.position.iat[i],C['wht'])}{name[i]}{C['rst']}"
                      for i in mine_at.values())
        print(f"  {C['b']}yours:{C['rst']} {r}")
    print()

    def inj(r):
        st = str(r.get('injury_status') or '')
        if st in ('IR', 'PUP', 'Out', 'Doubtful'):
            return f" {C['red']}{C['b']}[{st}]{C['rst']}"
        if st == 'Questionable':
            bp = str(r.get('injury_body_part') or '')
            return f" {C['yel']}[Q{'' if bp in ('Undisclosed','nan','','None') else ' '+bp[:12]}]{C['rst']}"
        return ""

    L.box_top(f"{C['b']}BEST AVAILABLE — YOUR TIERS{C['rst']}")
    c = live[live.position.isin(['RB', 'WR'])].sort_values(['tier', 'order', 'BOARD']).head(11)
    for _, r in c.iterrows():
        nm = r['name'][:24]
        body = (f"  {L.tag(r.position, r.tier)} {C['wht']}{nm:<25}{C['rst']}"
                f"{C['gry']}board#{int(r.BOARD):<5}{C['rst']}{inj(r)}")
        print(f"{C['cyn']}│{C['rst']}{body}{' '*max(W-2-L.vlen(body),0)}{C['cyn']}│{C['rst']}")
    L.box_bot()
    tq = live[live.position.isin(['TE', 'QB'])].sort_values('BOARD').head(4)
    if len(tq):
        print(f"  {C['dim']}market TE/QB:{C['rst']} " + "  ".join(
            f"{POSC[r.position]}{r['name']}{C['rst']}" for _, r in tq.iterrows()))

    nxt = MY[len(chosen) + 1] if len(chosen) + 1 < len(MY) else None
    if nxt:
        Pn = L.availability(b, taken, nxt - last)
        b3 = b.assign(p_now=Pn, gone=taken)
        lv = b3[~b3.gone & b3.position.isin(['RB', 'WR']) & (b3.tier <= 5)]
        print(f"\n  {C['b']}TIERS{C['rst']} {C['dim']}(expected to reach your next pick, {nxt}){C['rst']}")
        for pos in ('RB', 'WR'):
            parts = []
            for t in sorted(lv[lv.position == pos].tier.dropna().unique()):
                g = lv[(lv.position == pos) & (lv.tier == t)]
                e = g.p_now.sum()
                col = C['red'] if e < 0.8 else (C['yel'] if e < 2 else C['grn'])
                parts.append(f"{col}{pos}{int(t)}:{len(g)}({e:.1f}){C['rst']}")
            if parts:
                print(f"    {POSC[pos]}{pos}{C['rst']}  " + "  ".join(parts))
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--picks', default='')
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()
    b, _ = L.load()
    name = b.name.to_numpy()
    idx = {n.lower(): i for i, n in enumerate(name)}

    chosen = []
    for p in [x.strip() for x in a.picks.split(',') if x.strip()]:
        hit = [i for n, i in idx.items() if p.lower() in n]
        if hit:
            chosen.append(hit[0])

    while len(chosen) < len(MY):
        render(b, chosen, a)
        try:
            raw = input(f"\n  {C['b']}take:{C['rst']} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  bye"); return
        if not raw:
            continue
        if raw.lower() in ('q', 'quit', 'exit'):
            print("  bye"); return
        if raw.lower() == 'undo':
            if chosen: chosen.pop()
            continue
        hit = [i for n, i in idx.items() if raw.lower() in n and i not in chosen]
        if not hit:
            input(f"  {C['red']}no match for '{raw}' - enter to retry{C['rst']}")
            continue
        if len(hit) > 1:
            exact = [i for i in hit if name[i].lower().startswith(raw.lower())]
            hit = exact or hit
        chosen.append(hit[0])
        print(f"  {C['grn']}-> {name[hit[0]]}{C['rst']}")

    print(f"\n{C['b']}FINAL ROSTER{C['rst']}")
    for pk, i in zip(MY, chosen):
        print(f"   {pk:>4}  {POSC.get(b.position.iat[i],C['wht'])}{name[i]}{C['rst']}"
              f" {C['gry']}({b.position.iat[i]}){C['rst']}")


if __name__ == '__main__':
    main()

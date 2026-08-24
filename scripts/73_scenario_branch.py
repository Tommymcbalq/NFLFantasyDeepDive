"""Branch-conditional draft plan.

A news event moves TWO things that must be modelled separately:
  - the player's VALUE   (what he is worth)
  - the room's PRICE     (when he comes off the board)
They do not move by the same amount, and the gap between them IS the decision.
If the room drops him further than his value falls, he becomes a buy; if less, avoid.

Usage: python3 scripts/73_scenario_branch.py
"""
import numpy as np, pandas as pd

SLOT, TEAMS, ROUNDS, N = 5, 10, 6, 20000
SIG_STATED = 1.40*np.sqrt(np.pi/2)   # E|err| = 1.40 over the owner's stated block
SIG_TAIL   = 9.0
STATED     = 28

b = pd.read_csv('results/expected_order_2026.csv')
v2 = pd.read_csv('results/board_2026_v2.csv')
def key(n): return str(n).lower().replace('.','').replace("'",'').replace(' jr','').replace(' iii','').strip()
b['k']=b.name.map(key); v2['k']=v2.name.map(key)
b = b.drop(columns=['final']).merge(v2[['k','final']],on='k',how='left')
b['final']=b.final.fillna(-9.0)

def my_picks(slot):
    return [r*TEAMS + (slot if r%2==0 else TEAMS-slot+1) for r in range(ROUNDS)]

def run(val_shift=None, pick_shift=None, seed=7):
    """val_shift/pick_shift: {player_name: delta}. pick_shift +k = falls k picks later."""
    val = b.final.to_numpy(float).copy()
    mu  = b.exp_pick.to_numpy(float).copy()
    for nm,d in (val_shift or {}).items():
        i=b.index[b.name==nm]; val[i]+=d
    for nm,d in (pick_shift or {}).items():
        i=b.index[b.name==nm]; mu[i]+=d
    sig=np.where(b.exp_pick.to_numpy()<=STATED, SIG_STATED, SIG_TAIL)
    rng=np.random.default_rng(seed); MY=my_picks(SLOT)
    got={k:[] for k in MY}
    name=b.name.to_numpy()
    for _ in range(N):
        draw=rng.normal(mu,sig); order=np.argsort(draw)
        taken=np.zeros(len(b),bool); ptr=0
        for pk in range(1,ROUNDS*TEAMS+1):
            if pk in MY:
                free=np.flatnonzero(~taken)
                i=free[np.argmax(val[free])]
                got[pk].append(name[i])
            else:
                while ptr<len(order) and taken[order[ptr]]: ptr+=1
                if ptr>=len(order): break
                i=order[ptr]
            taken[i]=True
    return got

# Jeanty currently carries the -1.24 Aug-23 ankle view (14.54 vs 15.78 undiscounted).
# Scenario A must REMOVE it; the others deepen it. Price shifts are separate.
SCEN = {
 "A. CLEARED - low ankle, plays Wk1":        ({'Ashton Jeanty':+1.24}, {'Ashton Jeanty': -1}),
 "B. MINOR - misses 1-2, room shrugs":       ({'Ashton Jeanty': 0.00}, {'Ashton Jeanty': +4}),
 "C. HIGH ANKLE - 4-6 wks, room fades hard": ({'Ashton Jeanty':-1.50}, {'Ashton Jeanty':+18}),
 "D. HIGH ANKLE but room OVERREACTS":        ({'Ashton Jeanty':-1.50}, {'Ashton Jeanty':+30}),
}
for lab,(vs,ps) in SCEN.items():
    got=run(vs,ps)
    print(f"\n=== {lab} ===")
    for pk in my_picks(SLOT)[:3]:
        c=pd.Series(got[pk]).value_counts(normalize=True).head(3)
        line=" | ".join(f"{n} {p*100:.0f}%" for n,p in c.items())
        print(f"   pick {pk:>2}: {line}")

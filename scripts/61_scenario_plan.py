"""Contingency plan for the first N rounds.

Opponents draft to the OWNER'S stated expected order (results/expected_order_2026.csv) with
noise calibrated from the mock: mean absolute slot error 1.40 over picks 7-26, so sigma is set
so E|error| = 1.40 in the stated block and widens beyond it where he has no stated view.

The owner picks the highest `final` on his own board among survivors — his board decides,
the model only supplies availability.
"""
import numpy as np, pandas as pd, sys

SLOT=int(sys.argv[1]) if len(sys.argv)>1 else 5
TEAMS=10; ROUNDS=int(sys.argv[2]) if len(sys.argv)>2 else 6
N=20000
SIG_STATED=1.40*np.sqrt(np.pi/2)   # sigma giving E|N(0,sigma)| = 1.40
SIG_TAIL=9.0

def my_picks(slot,rounds):
    out=[]
    for r in range(rounds):
        out.append(r*TEAMS + (slot if r%2==0 else TEAMS-slot+1))
    return out

b=pd.read_csv('results/expected_order_2026.csv')
STATED=28
mu=b.exp_pick.to_numpy(float)
sig=np.where(mu<=STATED, SIG_STATED, SIG_TAIL)
val=b.final.fillna(-9).to_numpy(float); name=b.name.to_numpy(); pos=b.position.to_numpy()
MY=my_picks(SLOT,ROUNDS)
rng=np.random.default_rng(3)
got={k:[] for k in MY}; avail={k:np.zeros(len(b)) for k in MY}

for _ in range(N):
    draw=rng.normal(mu,sig)
    order=np.argsort(draw)
    taken=np.zeros(len(b),bool)
    ptr=0
    for pk in range(1,ROUNDS*TEAMS+1):
        if pk in MY:
            free=~taken
            avail[pk]+=free
            i=np.flatnonzero(free)[np.argmax(val[free])]
            got[pk].append(name[i])
        else:
            while ptr<len(order) and taken[order[ptr]]: ptr+=1
            if ptr>=len(order): break
            i=order[ptr]
        taken[i]=True

print(f"=== CONTINGENCY PLAN — slot 1.{SLOT:02d}, first {ROUNDS} rounds, {N} sims ===")
print("your board picks; percentages are how often that branch happens\n")
for pk in MY:
    r=pk//TEAMS+1 if pk%TEAMS else pk//TEAMS
    c=pd.Series(got[pk]).value_counts(normalize=True)
    a=pd.Series(avail[pk]/N,index=name)
    print(f"--- PICK {pk} (round {r}) ---")
    for nm,p in c.head(5).items():
        row=b[b.name==nm].iloc[0]
        print(f"   {p*100:>5.1f}%  {nm:<24}{row.position}  val {row.final:5.2f}   (available {a[nm]*100:.0f}% of the time)")
    print()

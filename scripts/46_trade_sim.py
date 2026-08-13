"""Monte-Carlo the post-trade draft with positional-run dynamics.

Opponents pick ADP + noise, but with RUN MOMENTUM: after a stretch where a position has
gone heavily, the next pickers over-weight it.  Runs are the thing point estimates miss --
independent draws never produce the "six RBs in eight picks" that actually happens.
"""
import sys, numpy as np, pandas as pd

TEAMS=10; ROUNDS=14
MY=[5,7,27,47,54,56,67,76,94,96,105,116,125,136]
NEED={'QB':(1,2),'RB':(2,6),'WR':(2,6),'TE':(1,2)}   # (starters, max)
MOM=float(sys.argv[1]) if len(sys.argv)>1 else 1.6   # run strength
NSIM=int(sys.argv[2]) if len(sys.argv)>2 else 3000

a=pd.read_csv('data/adp/adp_ppr_2026_all_20260809.csv')
a=a[a.position.isin(['RB','WR','TE','QB'])].nsmallest(160,'adp').reset_index(drop=True)
b=pd.read_csv('results/board_2026_overall_vorp.csv')[['name','vorp_real']]
a=a.merge(b,on='name',how='left'); a['vorp_real']=a.vorp_real.fillna(a.vorp_real.min())
adp=a.adp.to_numpy(); sd=np.clip(a.stdev.to_numpy(),.5,None)
pos=a.position.to_numpy(); val=a.vorp_real.to_numpy(); nms=a.name.to_numpy()
rng=np.random.default_rng(20260811)

def run_one():
    taken=np.zeros(len(a),bool); myroster=[]; got={}
    recent=[]
    for pk in range(1,ROUNDS*TEAMS+1):
        if pk in MY:
            # my rule: best available by board, respecting roster maxima
            cnt={p:sum(1 for x in myroster if x==p) for p in NEED}
            ok=~taken & np.array([cnt.get(p,0)<NEED.get(p,(0,99))[1] for p in pos])
            # must fill starters by the end
            if not ok.any(): ok=~taken
            i=np.flatnonzero(ok)[np.argmax(val[ok])]
            myroster.append(pos[i]); got[pk]=(nms[i],pos[i],val[i])
        else:
            w=np.where(taken,-1e9,-rng.normal(adp,sd))
            if recent:                       # run momentum
                c={}
                for p in recent[-6:]: c[p]=c.get(p,0)+1
                for p,k in c.items():
                    if k>=3: w[pos==p]+= MOM*(k-2)
            i=int(np.argmax(w))
        taken[i]=True; recent.append(pos[i])
    return got, sum(v for _,_,v in got.values()), myroster

rows=[]; tot=[]
for s in range(NSIM):
    got,t,ros=run_one(); tot.append(t)
    for pk,(n,p,v) in got.items(): rows.append((s,pk,n,p,v))
d=pd.DataFrame(rows,columns=['sim','pick','name','pos','vorp'])
tot=np.array(tot)
print(f"run momentum={MOM}, {NSIM} sims")
print(f"roster VORP (starters+bench, 14 picks): p10 {np.percentile(tot,10):.1f}  "
      f"p50 {np.percentile(tot,50):.1f}  p90 {np.percentile(tot,90):.1f}\n")
for pk in MY[:8]:
    s=d[d.pick==pk]
    top=s.name.value_counts(normalize=True).head(4)
    mix=s.pos.value_counts(normalize=True)
    print(f"pick {pk:>3}: median VORP {s.vorp.median():5.2f}  "
          f"[{s.vorp.quantile(.1):.2f}–{s.vorp.quantile(.9):.2f}]   "
          + "  ".join(f"{n} {p:.0%}" for n,p in top.items()))
    print(f"          pos mix: " + " ".join(f"{k} {v:.0%}" for k,v in mix.items()))

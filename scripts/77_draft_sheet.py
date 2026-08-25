"""Branch-conditional draft sheet for the live 2026 draft.

Three things this does that 76 did not:

1. ROOT BRANCH AT 1.04. The owner reports the manager at 1.04 takes Nacua ~80%
   of the time and McCaffrey the other ~20%. Whichever he leaves is at 1.05.
   Those are different rosters (RB-first vs WR-first) so the whole downstream
   build is re-optimised inside each branch, not patched afterwards.

2. SURVIVAL-AWARE SELECTION. 76 chose the largest immediate lineup-marginal
   gain. That is myopic: it took Etienne at 27 (gain 9.75) over Rice (9.11)
   even though Rice reaches pick 34 with prob ~0.01 and Etienne with ~0.81.
   Here each candidate is scored by ONE-STEP LOOKAHEAD -- take him, then roll
   the rest of the draft forward greedily over R freshly-drawn room orders and
   average the final starting-lineup value. Waiting is priced automatically.

3. FALL-THROUGH QUERIES. "If 2.02 does not take CeeDee, do we take him?"
   answered by injecting the player into the available set at a given pick and
   asking whether the lookahead score picks him.

Usage: python3 scripts/77_draft_sheet.py
"""
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

TEAMS      = 10
N_OUTER    = 400
R_LOOK     = 8          # inner rollouts per candidate
N_CAND     = 6          # candidates scored per pick
SIG_STATED = 1.40*np.sqrt(np.pi/2)
SIG_TAIL   = 9.0
STATED     = 39         # mock board observed complete through pick 39
P_NACUA_04 = 0.80       # owner's read on the 1.04 manager

OWN = [(1,5),(2,6),(7,5),(9,5),(11,5),(12,6),(13,5),(14,6),(15,5)]
ACQ = [(3,7),(4,4),(5,7),(6,4),(8,4),(9,7)]
MY  = sorted((r-1)*TEAMS+p for r,p in OWN+ACQ)[:6]

SLOTS = [{'RB'},{'RB'},{'WR'},{'WR'},{'TE'},{'RB','WR'},{'RB','WR'}]

# Where the owner says the order is NOT rigid. Everything else keeps SIG_STATED.
LOOSE = {"Derrick Henry": 3.0, "Nico Collins": 3.0, "Chris Olave": 3.0,
         "Malik Nabers": 3.0, "Drake London": 3.0}

def key(n):
    return (str(n).lower().replace('.','').replace("'",'')
            .replace(' jr','').replace(' iii','').replace(' ii','').strip())

def lineup_value(vals, poss):
    if not len(vals): return 0.0
    C = np.zeros((len(vals), len(SLOTS)))
    for i,(v,p) in enumerate(zip(vals,poss)):
        for s,ok in enumerate(SLOTS):
            if p in ok: C[i,s] = v
    r,c = linear_sum_assignment(-C)
    return float(C[r,c].sum())

def load():
    b  = pd.read_csv('results/expected_order_2026_v2.csv')
    bd = pd.read_csv('results/board_2026_v5_mustar.csv')
    b['k']=b.name.map(key); bd['k']=bd.name.map(key)
    b = b.drop(columns=[c for c in ('final','position') if c in b.columns]).merge(
        bd[['k','final','position']], on='k', how='left')
    b['final']=b.final.fillna(-9.0); b['position']=b.position.fillna('WR')
    return b

class Sim:
    def __init__(s, b):
        s.val=b.final.to_numpy(float); s.pos=b.position.to_numpy()
        s.name=b.name.to_numpy(); s.mu=b.exp_pick.to_numpy(float)
        s.sig=np.where(s.mu<=STATED, SIG_STATED, SIG_TAIL)
        for nm,sg in LOOSE.items():
            i=np.flatnonzero(s.name==nm)
            if len(i): s.sig[i[0]]=sg
        s.idx={n:i for i,n in enumerate(s.name)}

    def top_by_pos(s, free):
        out=[]
        for p in ('RB','WR','TE','QB'):
            c=free[s.pos[free]==p]
            if len(c): out.append(int(c[int(np.argmax(s.val[c]))]))
        return out

    def rollout(s, taken, hv, hp, from_pick, order):
        """Greedy completion of my remaining picks over ONE given room order.

        The order is passed in rather than drawn here so that every candidate at
        a pick is evaluated against the SAME futures (common random numbers).
        Without that, candidate scores differ by independent simulation noise of
        roughly the same size as the real gaps (~0.5 PPG), and the argmax mostly
        selects the luckiest draw rather than the best player.
        """
        tk=taken.copy(); hv=list(hv); hp=list(hp)
        ptr=0
        for pk in range(from_pick, MY[-1]+1):
            if pk in MY:
                free=np.flatnonzero(~tk)
                base=lineup_value(hv,hp); bi,bg=free[0],-9e9
                for j in s.top_by_pos(free):
                    g=lineup_value(hv+[s.val[j]],hp+[s.pos[j]])-base
                    if g>bg: bg,bi=g,j
                i=bi; hv.append(s.val[i]); hp.append(s.pos[i])
            else:
                while ptr<len(order) and tk[order[ptr]]: ptr+=1
                if ptr>=len(order): break
                i=order[ptr]
            tk[i]=True
        return lineup_value(hv,hp)

    def choose(s, taken, hv, hp, pk, rng, extra=None):
        """One-step-lookahead choice at pick pk. Returns (index, {cand: score})."""
        free=np.flatnonzero(~taken)
        base=lineup_value(hv,hp)
        cands=s.top_by_pos(free)
        gains={j: lineup_value(hv+[s.val[j]],hp+[s.pos[j]])-base for j in free}
        cands+=[j for j in sorted(free,key=lambda j:-gains[j])[:N_CAND] if j not in cands]
        if extra is not None and extra not in cands: cands.append(extra)
        futures=[np.argsort(rng.normal(s.mu,s.sig)) for _ in range(R_LOOK)]
        sc={}
        for j in cands:
            tk=taken.copy(); tk[j]=True
            sc[j]=np.mean([s.rollout(tk,hv+[s.val[j]],hp+[s.pos[j]],pk+1,o)
                           for o in futures])
        best=max(sc,key=sc.get)
        return best, sc

    def run(s, branch, queries, n=N_OUTER, seed=5):
        """branch: 'A' = Nacua taken at 1.04 (CMC free); 'B' = CMC taken at 1.04."""
        rng=np.random.default_rng(seed)
        got={p:[] for p in MY}
        qres={pk:{nm:[0,0] for nm in q} for pk,q in queries.items()}
        for _ in range(n):
            d=rng.normal(s.mu,s.sig); order=np.argsort(d)
            taken=np.zeros(len(s.val),bool); ptr=0; hv,hp=[],[]
            gone = "Puka Nacua" if branch=='A' else "Christian McCaffrey"
            for pk in range(1, MY[-1]+1):
                if pk == 4:
                    # 1.04 is a FORCED room pick, not a player removed on top of
                    # the room's four picks. Pre-marking him taken while still
                    # letting the room pick at 1.04 removed five players before
                    # pick 5 and pushed McCaffrey off the board spuriously.
                    i = s.idx[gone]
                    taken[i] = True
                    continue
                if pk in MY:
                    for nm in queries.get(pk,[]):
                        qi=s.idx[nm]
                        if taken[qi]: continue
                        tk=taken.copy()
                        j,_=s.choose(tk,hv,hp,pk,rng,extra=qi)
                        qres[pk][nm][1]+=1; qres[pk][nm][0]+= (j==qi)
                    i,_=s.choose(taken,hv,hp,pk,rng)
                    got[pk].append(s.name[i]); hv.append(s.val[i]); hp.append(s.pos[i])
                else:
                    while ptr<len(order) and taken[order[ptr]]: ptr+=1
                    if ptr>=len(order): break
                    i=order[ptr]
                taken[i]=True
        return got, qres

def main():
    b=load(); s=Sim(b)
    QUERIES={16:["CeeDee Lamb","Saquon Barkley","Ashton Jeanty","Justin Jefferson","Brock Bowers"],
             27:["Rashee Rice","Travis Etienne Jr.","Nico Collins","Chris Olave"]}
    for br,lab,w in (('A','1.04 takes NACUA  -> McCaffrey is yours',P_NACUA_04),
                     ('B','1.04 takes McCAFFREY -> Nacua is yours',1-P_NACUA_04)):
        print(f"\n{'='*66}\nBRANCH {br} ({w:.0%}):  {lab}\n{'='*66}")
        got,q=s.run(br,QUERIES)
        for pk in MY:
            c=pd.Series(got[pk]).value_counts(normalize=True).head(4)
            line=" | ".join(f"{n} {f:.0%}" for n,f in c.items())
            print(f"  pick {pk:>2}:  {line}")
        print("\n  fall-through (P(take him | he is there)):")
        for pk,d in q.items():
            for nm,(hit,tot) in d.items():
                if tot: print(f"     pick {pk:>2}  {nm:<22} there in {tot/N_OUTER:5.0%} of sims -> take {hit/tot:5.0%}")

if __name__=='__main__':
    main()

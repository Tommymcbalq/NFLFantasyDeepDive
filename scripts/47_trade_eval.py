"""Evaluate a trade on two scales at once.

  GENERIC : the market curve at each slot / each player's ADP-implied value.  This is what
            the other manager sees, and per the owner it is the primary lens.
  YOURS   : the same board after the 28 logged views (results/views_2026.csv).

Picks are given as round.pick in a 10-team snake (e.g. 2.06) or as an overall number.
Players are given by name.

  python3 scripts/47_trade_eval.py --get "1.07,Davante Adams" --give "2.06,4.06"
"""
import argparse, numpy as np, pandas as pd

R='results/'
f=pd.read_csv(R+'board_2026_full.csv')[['name','team','adp','position','pi','value_final']]
v=pd.read_csv(R+'sectionO_board_2026_vorp.csv'); Rr=v[v.frame==10].groupby('position').R_real.first()/17
te=pd.read_csv(R+'valuation_te_2026.csv'); qb=pd.read_csv(R+'valuation_qb_2026.csv')
def _n(d,p):
    n=[c for c in ['player','name'] if c in d.columns][0]
    vc=[c for c in ['board_value','theta_star','m_adp'] if c in d.columns][0]
    o=d[[n,'adp',vc]].rename(columns={n:'name',vc:'pi'}); o['position']=p; o['value_final']=o.pi
    o['team']=None; return o
A=pd.concat([f,_n(te,'TE'),_n(qb,'QB')],ignore_index=True).dropna(subset=['pi'])
A['gen']=A.pi-A.position.map(Rr)                 # generic / market
A['mine']=A.value_final-A.position.map(Rr)       # after views
SLOT=A.sort_values('adp').reset_index(drop=True)
SLOT['slot']=np.arange(1,len(SLOT)+1)

def parse(tok):
    tok=tok.strip()
    if not tok: return None
    if tok.replace('.','',1).isdigit():
        if '.' in tok:
            r,p=tok.split('.'); ov=(int(r)-1)*10+int(p)
        else: ov=int(tok)
        row=SLOT[SLOT.slot==ov]
        if not len(row): return (tok,'pick',np.nan,np.nan,'-')
        r=row.iloc[0]; return (f"{tok} (ov {ov})",'pick',r.gen,r.mine,r['name'])
    hit=A[A.name.str.lower().str.contains(tok.lower(),regex=False)]
    if len(hit)!=1: raise SystemExit(f"ambiguous/unknown: {tok} -> {list(hit.name)}")
    r=hit.iloc[0]; return (r['name'],'player',r.gen,r.mine,f"ADP {r.adp}")

def side(items,lbl):
    g=m=0; print(lbl)
    for t in items:
        x=parse(t)
        if x is None: continue
        n,k,gv,mv,note=x; g+=0 if np.isnan(gv) else gv; m+=0 if np.isnan(mv) else mv
        print(f"   {n:<24} {k:<7} generic {gv:6.2f}   yours {mv:6.2f}   [{note}]")
    print(f"   {'TOTAL':<24} {'':<7} generic {g:6.2f}   yours {m:6.2f}\n")
    return g,m

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--get',default=''); ap.add_argument('--give',default='')
    a=ap.parse_args()
    ga,ma=side([x for x in a.get.split(',') if x.strip()],'YOU GET:')
    gb,mb=side([x for x in a.give.split(',') if x.strip()],'YOU GIVE:')
    print(f"NET   generic {ga-gb:+.2f}   yours {ma-mb:+.2f}")

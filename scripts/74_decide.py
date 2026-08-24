"""Steps 8-10: your availability odds -> the pick.

No simulation of your league. You supply P(available) in
data/drafts/availability_priors.csv; this computes lineup-marginal value and the
cost of waiting, and ranks the options.

  python3 scripts/74_decide.py --pick 16 --next 25 --roster "Christian McCaffrey"
"""
import argparse, numpy as np, pandas as pd

STARTERS={'QB':1,'RB':2,'WR':2,'TE':1,'FLEX':2}
def key(n): return str(n).lower().replace('.','').replace("'",'').replace(' jr','').replace(' iii','').strip()

def marginal(board, roster, cand):
    """value the candidate ADDS to the best starting lineup."""
    def lineup(names):
        d={p:sorted([board[n]['val'] for n in names if board[n]['pos']==p],reverse=True)
           for p in ['QB','RB','WR','TE']}
        s=sum(d['QB'][:1])+sum(d['TE'][:1])+sum(d['RB'][:2])+sum(d['WR'][:2])
        flex=sorted(d['RB'][2:]+d['WR'][2:],reverse=True)[:2]
        return s+sum(flex)
    return lineup(roster+[cand])-lineup(roster)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--pick',type=int,required=True); ap.add_argument('--next',type=int,required=True)
    ap.add_argument('--roster',default='')
    a=ap.parse_args()
    b=pd.read_csv('results/board_2026_v2.csv')
    board={r['name']:{'val':r.final,'pos':r.position} for _,r in b.iterrows() if pd.notna(r.final)}
    pri=pd.read_csv('data/drafts/availability_priors.csv')
    roster=[x.strip() for x in a.roster.split(',') if x.strip()]
    now=pri[(pri.pick==a.pick)&pri.p_available.notna()]
    nxt=pri[(pri.pick==a.next)&pri.p_available.notna()]
    if not len(now):
        print(f"No odds supplied for pick {a.pick}. Fill p_available in "
              f"data/drafts/availability_priors.csv and rerun.")
        print("\nCandidates on the board around that pick, for reference:")
        c=b.sort_values('final',ascending=False).head(24)
        print(c[['name','position','adp','final']].round(2).to_string(index=False))
        raise SystemExit
    rows=[]
    # expected marginal available at the NEXT pick, given your odds there
    exp_next={}
    for p in ['QB','RB','WR','TE']:
        cands=[(r.player,r.p_available) for _,r in nxt.iterrows()
               if board.get(r.player,{}).get('pos')==p]
        # E[best marginal] treating your odds as independent, best-first
        cands.sort(key=lambda t:-marginal(board,roster,t[0]) if t[0] in board else 0)
        e=0.0; miss=1.0
        for nm,q in cands:
            m=marginal(board,roster,nm); e+=miss*q*m; miss*=(1-q)
        exp_next[p]=e
    for _,r in now.iterrows():
        if r.player not in board: continue
        m=marginal(board,roster,r.player); pos=board[r.player]['pos']
        rows.append((r.player,pos,board[r.player]['val'],r.p_available,m,
                     exp_next.get(pos,0.0),m-exp_next.get(pos,0.0)))
    d=pd.DataFrame(rows,columns=['player','pos','board_val','p_avail','marginal_now',
                                 'E_marginal_next','wait_cost']).sort_values('wait_cost',ascending=False)
    print(f"=== PICK {a.pick} (next pick {a.next}) ===")
    print(f"roster: {roster or 'empty'}\n")
    print(d.round(2).to_string(index=False))
    print("\nrank by wait_cost: how much you lose by NOT taking him now.")

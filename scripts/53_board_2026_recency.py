"""§S4 — 2026 WR board on the 08-11->08-18 ADP window, RECENCY-WEIGHTED prior (h=2).

Owner decision (2026-08-18): use the recency-weighted prior. §S2 evidence: all six weighted
arms beat their flat counterparts on RMSE and Spearman; theta*_h4 DM p=.049 vs flat, h2 is the
RMSE minimum (3.4196 vs 3.4671). BH across the 3-arm family clears at q=.20, not q=.10 — so this
is an owner call on a judgment margin, recorded as such, not a certified edge.

For the live board Y=2026 lies beyond every training season, so w_t = 2^{-(2026-t)/2}:
2025 .71, 2024 .50, 2023 .35, 2022 .25, ... 2015 .02.  Effective seasons ~3.2 of 11.
Likelihood mu_hat keeps its h=1 season half-life (script 01). sigma^2(tier) flat, as pre-registered.
"""
import re, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
ROOT="/Users/thomasmcnamee/NFL"; H=2.0
SUF={"jr","sr","ii","iii","iv","v"}
def nn(s):
    s=re.sub(r"[^a-z ]","",str(s).lower().replace("."," ").replace("-"," ").replace("'",""))
    return " ".join(t for t in s.split() if t not in SUF)

panel=pd.read_csv(f"{ROOT}/results/market_prior_11yr.csv"); fit=panel[panel.in_fit].copy()
w=2.0**(-(2026-fit.year.values)/H)
print(f"training seasons 2015-2025, weights {w.min():.3f}..{w.max():.3f}, "
      f"n_eff seasons {(w.sum()**2/(w**2).sum())/(len(fit)/11):.2f} of 11")

iso_f=IsotonicRegression(increasing=False,out_of_bounds="clip"); iso_f.fit(np.log(fit.adp),fit.ppg)
iso_r=IsotonicRegression(increasing=False,out_of_bounds="clip"); iso_r.fit(np.log(fit.adp),fit.ppg,sample_weight=w)
def wvar(x,ww):
    mu=np.sum(ww*x)/np.sum(ww); return float(np.sum(ww*(x-mu)**2)/(np.sum(ww)-np.sum(ww**2)/np.sum(ww)))
r=fit.ppg.values-iso_r.predict(np.log(fit.adp.values))
tau2={t:wvar(r[(fit.tier==t).values],w[(fit.tier==t).values]) for t in ["rookie","soph","vet"]}
rf=fit.ppg.values-iso_f.predict(np.log(fit.adp.values))
tau2f=fit.assign(rf=rf).groupby("tier").rf.var(ddof=1).to_dict()
print("tau2 recency:",{k:round(v,2) for k,v in tau2.items()}," flat:",{k:round(v,2) for k,v in tau2f.items()})

wk=[]
for y in range(2014,2026):
    d=pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
        usecols=["player_id","position","season","season_type","targets","fantasy_points_ppr"],low_memory=False)
    wk.append(d[(d.season_type=="REG")&(d.targets>1)])
wk=pd.concat(wk,ignore_index=True)
wr=wk[wk.position=="WR"].copy()
ps=wr.groupby(["player_id","season"]).agg(mt=("targets","mean"),mu=("fantasy_points_ppr","mean")).reset_index()
wr=wr.merge(ps[ps.mt>=3.0],on=["player_id","season"])
meta=pd.read_csv(f"{ROOT}/data/meta/players_meta.csv",low_memory=False,
                 usecols=["gsis_id","display_name","position","rookie_season","last_season"]).dropna(subset=["gsis_id"])
meta["k"]=meta.display_name.map(nn)
wr=wr.merge(meta[["gsis_id","rookie_season"]].rename(columns={"gsis_id":"player_id"}),on="player_id").dropna(subset=["rookie_season"])
wr["e2"]=(wr.fantasy_points_ppr-wr.mu)**2
wr["tier"]=np.select([wr.season-wr.rookie_season==0,wr.season-wr.rookie_season==1],["rookie","soph"],"vet")
sig2=wr.groupby("tier").e2.mean().to_dict()

seas=wk.groupby(["player_id","season"]).agg(G=("fantasy_points_ppr","size"),ybar=("fantasy_points_ppr","mean")).reset_index()
brd=pd.read_csv(f"{ROOT}/data/adp/adp_ppr_2026_all_20260818.csv")
brd=brd[brd.position=="WR"].nsmallest(30,"adp").copy(); brd["k"]=brd.name.map(nn)
mw=meta[meta.position=="WR"]
rows=[]
for _,b in brd.iterrows():
    h=mw[mw.k==b.k]
    if not len(h): print("UNMATCHED:",b["name"]); continue
    h=h.sort_values("last_season").iloc[-1]
    hist=seas[(seas.player_id==h.gsis_id)&(seas.season<2026)]
    exp=2026-int(h.rookie_season); tier="rookie" if exp==0 else ("soph" if exp==1 else "vet")
    if len(hist)==0:
        mu_hat,n_eff=np.nan,0.0
    else:
        S=hist.season.max(); ww=2.0**(-(S-hist.season.values)/1.0)
        mu_hat=float((ww*hist.ybar.values).sum()/ww.sum()); n_eff=float(ww.sum()**2/(ww**2).sum())
    out={}
    for tag,isoo,tv in [("flat",iso_f,tau2f),("rec",iso_r,tau2)]:
        m_hat=float(isoo.predict([np.log(b.adp)])[0])
        if n_eff==0: B,th=1.0,m_hat
        else:
            V=sig2[tier]/n_eff; B=V/(V+tv[tier]); th=(1-B)*mu_hat+B*m_hat
        out[f"m_{tag}"],out[f"B_{tag}"],out[f"theta_{tag}"]=m_hat,B,th
    rows.append(dict(name=b["name"],team=b.team,adp=b.adp,stdev=b.stdev,tier=tier,exp=exp,
                     mu_hat=mu_hat,n_eff=n_eff,**out))
d=pd.DataFrame(rows)
d["delta"]=d.theta_rec-d.theta_flat
d["rank_flat"]=d.theta_flat.rank(ascending=False).astype(int)
d["rank_rec"]=d.theta_rec.rank(ascending=False).astype(int)
d["rank_move"]=d.rank_flat-d.rank_rec
d=d.sort_values("theta_rec",ascending=False)
print("\n=== 2026 WR BOARD — recency-weighted prior (h=2), ADP window 08-11 -> 08-18 ===")
print(d[["rank_rec","name","team","adp","tier","mu_hat","n_eff","B_rec","theta_rec","theta_flat","delta","rank_move"]]
      .round(2).to_string(index=False))
d.to_csv(f"{ROOT}/results/board_2026_recency_h2.csv",index=False)
print(f"\nmean |theta shift| {d.delta.abs().mean():.3f} PPG; max {d.delta.abs().max():.3f}; "
      f"rank changes {(d.rank_move!=0).sum()}/30, max move {d.rank_move.abs().max()}")

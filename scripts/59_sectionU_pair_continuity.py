"""§U — QB-pair continuity as a PRECISION adjustment. Board-wide, LOSO-tested.

ORIGIN (owner, 2026-08-18). §T failed because it was generic: a binary "did the primary QB play"
pools every QB-receiver pair as exchangeable. The owner's objection is that the pair is the object -
Chase/Burrow is 61 shared games of accumulated interaction; Nabers/Dart is ONE. A level adjustment
to mu-hat cannot express that. His argument implies the correction belongs in the PRECISION: if a
player's history was accumulated with a QB he no longer has, that history is weaker evidence about
next season, so n_eff should fall, B should rise, and he should shrink harder toward the market.
mu-hat's LEVEL is untouched. This is mechanically disjoint from §T and §T's null does not bear on it.

REFERENCE QB, and why this one (pre-registered). For player i entering season Y, the reference QB is
the primary QB of i's season-Y TEAM in season Y-1 - the incumbent. Chosen for two reasons: it is the
owner's stated rule for unresolved competitions, and it is STRICTLY PRESEASON-KNOWABLE. Using season
Y's realised primary QB would import hindsight about in-season QB injuries, which is exactly the
information a preseason model must not have.

  share_i,Y = recency-weighted (h=1) fraction of i's pre-Y games played with the reference QB.
  n_eff_adj = n_eff * (lam + (1-lam) * share)
Family: lam in {0, 0.25, 0.50}; lam = 1 IS the incumbent (no adjustment). At share=1 every arm
reduces to the incumbent, so the arms differ only in how hard a broken pairing is discounted.
Everything else - mu-hat, sigma^2, tau^2, m(.), the h=2 recency prior, the inclusion rule - is
identical to the incumbent.

ADOPTION RULE fixed before running: an arm replaces the incumbent only if it BOTH lowers pooled RMSE
AND gives DM t > 0, p < .05 on 11 folds, AND survives BH at q=.10 across the 3-arm family. The BH
requirement is carried over from §39.5, where a single p=.049 was NOT treated as sufficient.
"""
import numpy as np, pandas as pd, re
from scipy import stats
from sklearn.isotonic import IsotonicRegression
ROOT="/Users/thomasmcnamee/NFL"; H=2.0; YEARS=list(range(2015,2026))
SUF={"jr","sr","ii","iii","iv","v"}
def nn(s):
    s=re.sub(r"[^a-z ]","",str(s).lower().replace("."," ").replace("-"," ").replace("'",""))
    return " ".join(t for t in s.split() if t not in SUF)
panel=pd.read_csv(f"{ROOT}/results/market_prior_11yr.csv")
cols=["player_id","position","season","week","team","season_type","targets","attempts","fantasy_points_ppr"]
wk=[]
for y in range(2014,2026):
    d=pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",usecols=cols,low_memory=False)
    wk.append(d[d.season_type=="REG"])
wk=pd.concat(wk,ignore_index=True)

qbw=wk[(wk.position=="QB")&(wk.attempts>=10)][["season","week","team","player_id"]].rename(columns={"player_id":"qb_id"}).drop_duplicates(["season","week","team"])
# incumbent QB of a team entering season Y = that team's most-attempting QB in Y-1
prim=(wk[(wk.position=="QB")].groupby(["season","team","player_id"]).attempts.sum().reset_index()
        .sort_values("attempts",ascending=False).drop_duplicates(["season","team"]))
incumbent={(s+1,t):q for s,t,q in zip(prim.season,prim.team,prim.player_id)}

rec=wk[wk.targets>1].merge(qbw,on=["season","week","team"],how="left")
# player's team per board-year, from the FFC board itself
team_by={}
for y in YEARS:
    a=pd.read_csv(f"{ROOT}/data/adp/historical/adp_ppr_{y}.csv")
    for _,r in a[a.position=="WR"].iterrows(): team_by[(y,nn(r["name"]))]=r.team
name_of=dict(zip(panel.gsis_id,panel.name))

def share_and_neff(gsis,Y):
    h=rec[(rec.player_id==gsis)&(rec.season<Y)]
    if len(h)==0: return np.nan,0.0,np.nan
    S=h.season.max(); w=2.0**(-(S-h.season.values)/1.0)
    tm=team_by.get((Y,nn(name_of.get(gsis,""))))
    ref=incumbent.get((Y,tm))
    sh=float(w[(h.qb_id==ref).values].sum()/w.sum()) if ref is not None else np.nan
    g=h.groupby("season").fantasy_points_ppr.agg(["size","mean"])
    ws=2.0**(-(g.index.max()-g.index.values)/1.0)
    mu=float((ws*g["mean"].values).sum()/ws.sum()); ne=float(ws.sum()**2/(ws**2).sum())
    return mu,ne,sh

def wvar(x,w):
    m=np.sum(w*x)/np.sum(w); return float(np.sum(w*(x-m)**2)/(np.sum(w)-np.sum(w**2)/np.sum(w)))
LAMS=[("inc",None),("lam50",0.50),("lam25",0.25),("lam00",0.00)]
preds=[]
wr=wk[(wk.position=="WR")&(wk.targets>1)].copy()
ps=wr.groupby(["player_id","season"]).agg(mt=("targets","mean"),mu=("fantasy_points_ppr","mean")).reset_index()
wr=wr.merge(ps[ps.mt>=3.0],on=["player_id","season"])
mt=pd.read_csv(f"{ROOT}/data/meta/players_meta.csv",low_memory=False,usecols=["gsis_id","rookie_season"]).dropna()
wr=wr.merge(mt.rename(columns={"gsis_id":"player_id"}),on="player_id").dropna(subset=["rookie_season"])
wr["e2"]=(wr.fantasy_points_ppr-wr.mu)**2
wr["tier"]=np.select([wr.season-wr.rookie_season==0,wr.season-wr.rookie_season==1],["rookie","soph"],"vet")

for Y in YEARS:
    tr=panel[(panel.year!=Y)&panel.in_fit].copy(); ev=panel[(panel.year==Y)&panel.in_fit].copy()
    w=2.0**(-np.abs(tr.year.values-Y)/H)
    iso=IsotonicRegression(increasing=False,out_of_bounds="clip"); iso.fit(np.log(tr.adp),tr.ppg,sample_weight=w)
    r=tr.ppg.values-iso.predict(np.log(tr.adp.values))
    tau2={t:(wvar(r[(tr.tier==t).values],w[(tr.tier==t).values]) if (tr.tier==t).sum()>=2 else np.nan) for t in ["rookie","soph","vet"]}
    ev["m_hat"]=iso.predict(np.log(ev.adp.values)); ev["tau2"]=ev.tier.map(tau2).fillna(wvar(r,w))
    ev["sig2"]=ev.tier.map(wr[wr.season!=Y].groupby("tier").e2.mean())
    tri=ev.gsis_id.map(lambda g: share_and_neff(g,Y))
    ev["mu_hat"]=[t[0] for t in tri]; ev["n_eff"]=[t[1] for t in tri]; ev["share"]=[t[2] for t in tri]
    for tag,lam in LAMS:
        ne=ev.n_eff.values if lam is None else ev.n_eff.values*(lam+(1-lam)*ev.share.fillna(1.0).values)
        with np.errstate(divide="ignore",invalid="ignore"): V=ev.sig2.values/ne
        B=np.where(ne<=0,1.0,V/(V+ev.tau2.values))
        ev[f"th_{tag}"]=np.where(ne<=0,ev.m_hat,(1-B)*np.nan_to_num(ev.mu_hat.values)+B*ev.m_hat)
    preds.append(ev)
pred=pd.concat(preds,ignore_index=True)
print(f"rows {len(pred)}; share computable {pred.share.notna().mean():.1%}; mean share {pred.share.mean():.3f}; "
      f"share<0.25 on {(pred.share<0.25).mean():.1%} of rows")
res=[]
for tag,lam in LAMS:
    c=f"th_{tag}"; rmse=float(np.sqrt(((pred.ppg-pred[c])**2).mean()))
    rho=float(pred.groupby("year").apply(lambda g: stats.spearmanr(g[c],g.ppg).statistic,include_groups=False).mean())
    if tag=="inc": t=p=np.nan
    else:
        d=(pred.ppg-pred.th_inc)**2-(pred.ppg-pred[c])**2; dy=d.groupby(pred.year).mean()
        t=float(dy.mean()/(dy.std(ddof=1)/np.sqrt(len(dy)))); p=float(2*stats.t.sf(abs(t),df=len(dy)-1))
    res.append(dict(arm=tag,lam=lam,rmse=round(rmse,4),spearman=round(rho,4),
                    dm_t=round(t,3) if t==t else np.nan,dm_p=round(p,4) if p==p else np.nan))
R=pd.DataFrame(res); print("\n=== §U scorecard (11 folds) ===\n"+R.to_string(index=False))
q=R.dropna(subset=["dm_p"]).sort_values("dm_p"); m=len(q)
print("\nBH q=.10:",[f"{a}(p={pp:.4f} vs {(i+1)/m*0.10:.4f})" for i,(a,pp) in enumerate(zip(q.arm,q.dm_p))])
print("survivors:",[a for i,(a,pp) in enumerate(zip(q.arm,q.dm_p)) if pp<=(i+1)/m*0.10] or "NONE")
pred.to_csv(f"{ROOT}/results/sectionU_continuity_loso.csv",index=False); R.to_csv(f"{ROOT}/results/sectionU_scorecard.csv",index=False)

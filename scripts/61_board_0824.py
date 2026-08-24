"""§S5 — combined WR+RB 2026 board: 11-season panels, recency prior h=2, ADP window 08-11->08-18.
Produces pi and Sigma-diagonal for the §J views overlay. Mirrors script 53 (WR) for RB.
Sigma per eq (26.2): tier residual variance minus per-game sampling component, floored at 25%.
"""
import re, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
ROOT="/Users/thomasmcnamee/NFL"; H=2.0
SUF={"jr","sr","ii","iii","iv","v"}
def nn(s):
    s=re.sub(r"[^a-z ]","",str(s).lower().replace("."," ").replace("-"," ").replace("'",""))
    return " ".join(t for t in s.split() if t not in SUF)
def wvar(x,w):
    mu=np.sum(w*x)/np.sum(w); return float(np.sum(w*(x-mu)**2)/(np.sum(w)-np.sum(w**2)/np.sum(w)))

wk=[]
for y in range(2014,2026):
    d=pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
        usecols=["player_id","position","season","season_type","targets","carries","fantasy_points_ppr"],low_memory=False)
    wk.append(d[d.season_type=="REG"])
wk=pd.concat(wk,ignore_index=True)
meta=pd.read_csv(f"{ROOT}/data/meta/players_meta.csv",low_memory=False,
    usecols=["gsis_id","display_name","position","rookie_season","last_season"]).dropna(subset=["gsis_id"])
meta["k"]=meta.display_name.map(nn)
brd_all=pd.read_csv(f"{ROOT}/data/adp/adp_ppr_2026_all_20260824.csv")

rows=[]
for POS,panel_f,inc in [("WR","market_prior_11yr.csv","targets"),("RB","market_prior_rb_11yr.csv","carries")]:
    fit=pd.read_csv(f"{ROOT}/results/{panel_f}"); fit=fit[fit.in_fit]
    w=2.0**(-(2026-fit.year.values)/H)
    iso=IsotonicRegression(increasing=False,out_of_bounds="clip"); iso.fit(np.log(fit.adp),fit.ppg,sample_weight=w)
    r=fit.ppg.values-iso.predict(np.log(fit.adp.values))
    tau2={t:wvar(r[(fit.tier==t).values],w[(fit.tier==t).values]) for t in ["rookie","soph","vet"]}
    inc_rule=(wk[inc]>1) if POS=="WR" else (wk[inc]>=3)
    sub=wk[inc_rule]
    pos_rows=wk[(wk.position==POS)]
    ps=pos_rows.groupby(["player_id","season"]).agg(mt=((inc),"mean"),mu=("fantasy_points_ppr","mean")).reset_index()
    flo=3.0 if POS=="WR" else 5.0
    pr=pos_rows.merge(ps[ps.mt>=flo],on=["player_id","season"])
    pr=pr.merge(meta[["gsis_id","rookie_season"]].rename(columns={"gsis_id":"player_id"}),on="player_id").dropna(subset=["rookie_season"])
    pr["e2"]=(pr.fantasy_points_ppr-pr.mu)**2
    pr["tier"]=np.select([pr.season-pr.rookie_season==0,pr.season-pr.rookie_season==1],["rookie","soph"],"vet")
    sig2=pr.groupby("tier").e2.mean().to_dict()
    gpm=pr.groupby(["player_id","season"]).size().mean()
    seas=sub.groupby(["player_id","season"]).agg(G=("fantasy_points_ppr","size"),ybar=("fantasy_points_ppr","mean")).reset_index()
    b=brd_all[brd_all.position==POS].nsmallest(30,"adp").copy(); b["k"]=b.name.map(nn)
    mp=meta[meta.position==POS]
    for _,x in b.iterrows():
        h=mp[mp.k==x.k]
        if not len(h): print("UNMATCHED",POS,x["name"]); continue
        h=h.sort_values("last_season").iloc[-1]
        hist=seas[(seas.player_id==h.gsis_id)&(seas.season<2026)]
        exp=2026-int(h.rookie_season); tier="rookie" if exp==0 else ("soph" if exp==1 else "vet")
        if len(hist)==0: mu_hat,n_eff=np.nan,0.0
        else:
            S=hist.season.max(); ww=2.0**(-(S-hist.season.values)/1.0)
            mu_hat=float((ww*hist.ybar.values).sum()/ww.sum()); n_eff=float(ww.sum()**2/(ww**2).sum())
        m_hat=float(iso.predict([np.log(x.adp)])[0])
        if n_eff==0: B,th=1.0,m_hat
        else:
            V=sig2[tier]/n_eff; B=V/(V+tau2[tier]); th=(1-B)*mu_hat+B*m_hat
        sig_theta=max(tau2[tier]-sig2[tier]/gpm, 0.25*tau2[tier])
        rows.append(dict(name=x["name"],team=x.team,position=POS,adp=x.adp,stdev=x.stdev,tier=tier,
                         mu_hat=mu_hat,n_eff=n_eff,m_hat=m_hat,B=B,pi=th,sig=sig_theta))
d=pd.DataFrame(rows).sort_values("pi",ascending=False)
d.to_csv(f"{ROOT}/results/board_2026_pi_sigma_h2_0824.csv",index=False)
print(d[["name","team","position","adp","tier","mu_hat","B","pi","sig"]].round(2).to_string(index=False))

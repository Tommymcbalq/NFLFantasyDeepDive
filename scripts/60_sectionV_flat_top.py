"""§V — The flat top of m(ADP): centred isotonic regression vs the plateau fit.

THE DEFECT. The incumbent m(.) is isotonic regression on log ADP. Isotonic's fitted values are a
STEP function: wherever the pool-adjacent-violators algorithm merges a block, every player in that
block receives the identical fitted value. On the 2026 WR board this collapses the entire top:
m(ADP) = 18.08 for every WR from ADP 3.1 to 10.2. Price separates nobody there, so mu-hat carries
the whole ordering unaided - which is why one soft season moved a player four places, and why a
receiver with the lowest mu-hat in the top nine was carried to 9th on price alone.

Is the plateau a FINDING or an ARTIFACT? Both are possible a priori:
  (a) genuine - the market really cannot distinguish the top few picks, and PPG really is flat there;
  (b) artifact - isotonic is a step estimator fitted on ~30 players/season, and the top decile is
      thin, so PAVA merges blocks that a smoother monotone estimator would separate.
This is decided out of sample, not by inspection.

CANDIDATES (all monotone decreasing in log ADP; the constraint is never relaxed):
  (i)   iso   - incumbent isotonic step function
  (ii)  cir   - CENTRED isotonic regression (Oron & Flournoy 2017). The standard fix for exactly
                this artifact: each isotonic plateau is collapsed to its weight-centroid x, and the
                fit interpolates linearly between centroids. Still monotone, but strictly decreasing
                within a former plateau. No new tuning parameter.
  (iii) ols   - PPG ~ log ADP, strictly monotone by construction, maximally smooth (the §6.1
                reference fit, promoted to a candidate here).
  (iv)  half  - 0.5*iso + 0.5*cir, a shrinkage between step and interpolation.
All are fitted with the SAME h=2 season weights and evaluated in the SAME 11-fold LOSO harness.
Nothing else in the estimator changes.

ADOPTION RULE fixed before running: a candidate replaces (i) only if it lowers pooled RMSE AND gives
DM t > 0 with p < .05 vs (i) on 11 folds AND survives BH q=.10 across the 3-candidate family.
"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
ROOT="/Users/thomasmcnamee/NFL"; H=2.0; YEARS=list(range(2015,2026))
panel=pd.read_csv(f"{ROOT}/results/market_prior_11yr.csv")

def cir_fit(x,y,w):
    """Centred isotonic regression: collapse each isotonic plateau to its weight-centroid."""
    iso=IsotonicRegression(increasing=False,out_of_bounds="clip"); iso.fit(x,y,sample_weight=w)
    f=iso.predict(x)
    o=np.argsort(x); xs,fs,ws=x[o],f[o],w[o]
    cx,cy=[],[]
    i=0
    while i<len(xs):
        j=i
        while j+1<len(xs) and np.isclose(fs[j+1],fs[i]): j+=1
        blk=slice(i,j+1); wt=ws[blk]
        cx.append(np.sum(wt*xs[blk])/np.sum(wt)); cy.append(fs[i])
        i=j+1
    cx,cy=np.array(cx),np.array(cy)
    if len(cx)<2: return lambda q: np.full_like(np.asarray(q,float),cy[0] if len(cy) else np.mean(y))
    return lambda q: np.interp(np.asarray(q,float),cx,cy[::-1][::-1] if False else cy)

def wvar(x,w):
    m=np.sum(w*x)/np.sum(w); return float(np.sum(w*(x-m)**2)/(np.sum(w)-np.sum(w**2)/np.sum(w)))

wk=[]
for y in range(2014,2026):
    d=pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
        usecols=["player_id","position","season","season_type","targets","fantasy_points_ppr"],low_memory=False)
    wk.append(d[(d.season_type=="REG")&(d.targets>1)])
wk=pd.concat(wk,ignore_index=True)
seas=wk.groupby(["player_id","season"]).agg(G=("fantasy_points_ppr","size"),ybar=("fantasy_points_ppr","mean")).reset_index()
wr=wk[wk.position=="WR"].copy()
ps=wr.groupby(["player_id","season"]).agg(mt=("targets","mean"),mu=("fantasy_points_ppr","mean")).reset_index()
wr=wr.merge(ps[ps.mt>=3.0],on=["player_id","season"])
mt=pd.read_csv(f"{ROOT}/data/meta/players_meta.csv",low_memory=False,usecols=["gsis_id","rookie_season"]).dropna()
wr=wr.merge(mt.rename(columns={"gsis_id":"player_id"}),on="player_id").dropna(subset=["rookie_season"])
wr["e2"]=(wr.fantasy_points_ppr-wr.mu)**2
wr["tier"]=np.select([wr.season-wr.rookie_season==0,wr.season-wr.rookie_season==1],["rookie","soph"],"vet")

def mu_neff(g,Y):
    h=seas[(seas.player_id==g)&(seas.season<Y)]
    if len(h)==0: return np.nan,0.0
    S=h.season.max(); w=2.0**(-(S-h.season.values)/1.0)
    return float((w*h.ybar.values).sum()/w.sum()), float(w.sum()**2/(w**2).sum())

preds=[]
for Y in YEARS:
    tr=panel[(panel.year!=Y)&panel.in_fit].copy(); ev=panel[(panel.year==Y)&panel.in_fit].copy()
    w=2.0**(-np.abs(tr.year.values-Y)/H)
    x,yv=np.log(tr.adp.values),tr.ppg.values; xe=np.log(ev.adp.values)
    iso=IsotonicRegression(increasing=False,out_of_bounds="clip"); iso.fit(x,yv,sample_weight=w)
    cir=cir_fit(x,yv,w)
    sw=w/w.sum()*len(w)
    b1,b0=np.polyfit(x,yv,1,w=np.sqrt(sw))
    M={"iso":iso.predict(xe),"cir":cir(xe),"ols":b0+b1*xe}
    M["half"]=0.5*M["iso"]+0.5*M["cir"]
    R={"iso":yv-iso.predict(x),"cir":yv-cir(x),"ols":yv-(b0+b1*x)}
    R["half"]=yv-(0.5*iso.predict(x)+0.5*cir(x))
    mn=ev.gsis_id.map(lambda g: mu_neff(g,Y))
    ev["mu_hat"]=[t[0] for t in mn]; ev["n_eff"]=[t[1] for t in mn]
    ev["sig2"]=ev.tier.map(wr[wr.season!=Y].groupby("tier").e2.mean())
    for tag in ["iso","cir","ols","half"]:
        r=R[tag]
        tau2={t:(wvar(r[(tr.tier==t).values],w[(tr.tier==t).values]) if (tr.tier==t).sum()>=2 else np.nan) for t in ["rookie","soph","vet"]}
        tv=ev.tier.map(tau2).fillna(wvar(r,w)).values
        with np.errstate(divide="ignore",invalid="ignore"): V=ev.sig2.values/ev.n_eff.values
        B=np.where(ev.n_eff.values<=0,1.0,V/(V+tv))
        ev[f"m_{tag}"]=M[tag]
        ev[f"th_{tag}"]=np.where(ev.n_eff.values<=0,M[tag],(1-B)*np.nan_to_num(ev.mu_hat.values)+B*M[tag])
    preds.append(ev)
pred=pd.concat(preds,ignore_index=True)

print("=== how flat is the top? fitted m(.) at the sharp end, full-sample h=2 fit ===")
fit=panel[panel.in_fit]; w=2.0**(-(2026-fit.year.values)/H)
x,yv=np.log(fit.adp.values),fit.ppg.values
iso=IsotonicRegression(increasing=False,out_of_bounds="clip"); iso.fit(x,yv,sample_weight=w); cir=cir_fit(x,yv,w)
sw=w/w.sum()*len(w); b1,b0=np.polyfit(x,yv,1,w=np.sqrt(sw))
g=np.array([3.1,3.9,5.4,6.4,10.2,10.7,14.5,21.9])
print(pd.DataFrame({"adp":g,"iso":iso.predict(np.log(g)).round(2),"cir":np.round(cir(np.log(g)),2),
                    "ols":np.round(b0+b1*np.log(g),2)}).to_string(index=False))

print("\n=== §V scorecard (11 folds) ===")
rows=[]
for tag,lab in [("iso","(i)   isotonic [incumbent]"),("cir","(ii)  centred isotonic"),
                ("ols","(iii) OLS log ADP"),("half","(iv)  0.5*iso + 0.5*cir")]:
    c=f"th_{tag}"; rmse=float(np.sqrt(((pred.ppg-pred[c])**2).mean()))
    rho=float(pred.groupby("year").apply(lambda gg: stats.spearmanr(gg[c],gg.ppg).statistic,include_groups=False).mean())
    if tag=="iso": t=p=np.nan
    else:
        d=(pred.ppg-pred.th_iso)**2-(pred.ppg-pred[c])**2; dy=d.groupby(pred.year).mean()
        t=float(dy.mean()/(dy.std(ddof=1)/np.sqrt(len(dy)))); p=float(2*stats.t.sf(abs(t),df=len(dy)-1))
    rows.append(dict(arm=lab,rmse=round(rmse,4),spearman=round(rho,4),
                     dm_t=round(t,3) if t==t else np.nan,dm_p=round(p,4) if p==p else np.nan))
R=pd.DataFrame(rows); print(R.to_string(index=False))
q=R.dropna(subset=["dm_p"]).sort_values("dm_p"); m=len(q)
surv=[a for i,(a,pp) in enumerate(zip(q.arm,q.dm_p)) if pp<=(i+1)/m*0.10]
print("\nBH q=.10 survivors:",surv or "NONE")
top=pred[pred.adp<=12]
print(f"\n--- on the flat region only (ADP<=12, n={len(top)}) ---")
for tag in ["iso","cir","ols","half"]:
    print(f"  th_{tag:<5} RMSE {np.sqrt(((top.ppg-top[f'th_{tag}'])**2).mean()):.4f}")
pred.to_csv(f"{ROOT}/results/sectionV_flattop_loso.csv",index=False); R.to_csv(f"{ROOT}/results/sectionV_scorecard.csv",index=False)

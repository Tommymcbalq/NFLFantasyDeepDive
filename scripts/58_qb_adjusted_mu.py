"""§T — QB-availability-adjusted mu-hat: a board-wide arm, LOSO-tested.

ORIGIN. The owner asked to see Ja'Marr Chase with and without Burrow (2026-08-18). The split did
not support the specific claim (2025: 20.12 PPG with vs 19.08 without, gap 1.04), but it raised a
general and previously untested question: mu-hat is a recency-weighted mean over ALL a player's
games, including games his team's primary QB missed. If those games are systematically depressed,
mu-hat is a biased estimate of the player's value under normal conditions.

WHY THIS IS NOT §E. §E pushed a POPULATION-AVERAGE mover effect into the data arm and failed
because m(ADP) already carried it - double counting. This is a WITHIN-PLAYER MEASUREMENT
correction: it changes what mu-hat estimates, not what is added to it. No market quantity is
touched, so the double-counting mechanism does not apply.

THE COUNTER-ARGUMENT, STATED BEFORE TESTING. Conditioning on the starter playing estimates
E[PPG | QB healthy], but the forecast target is next season's realised PPG, which INCLUDES whatever
QB absence occurs. So the arm trades noise reduction against an optimism bias, and it is not obvious
a priori which wins. That is exactly what the LOSO test decides. If the arm loses, the finding is
that QB-absence games carry real predictive information about a player's future, not that they are
noise.

DEFINITIONS (pre-registered).
  primary QB of a team-season = the QB with the most REG-season pass attempts for that team.
  a "starter game" for team t in week w = the primary QB recorded >= 10 attempts in it.
  mu_qb = the same h=1 recency-weighted mean as script 01, but each season's mean is taken over
          starter games only. Seasons with < 4 starter games fall back to the unadjusted season
          mean (flagged), so the arm never invents a season from 1-2 games.
  Everything else - sigma^2(tier), tau^2(tier), m(.), B, the inclusion rule - is IDENTICAL to the
  incumbent. Only the mu-hat input changes, so the comparison is clean.

ARMS on the 11-fold WR panel (2015-2025), recency prior h=2 throughout:
  (i)   m_hat            market only
  (ii)  theta*  (mu)     the incumbent blind posterior
  (ix)  theta*  (mu_qb)  the candidate
ADOPTION RULE, fixed before running: (ix) replaces (ii) only if it BOTH lowers pooled RMSE AND
gives DM t > 0 with p < .05 vs (ii), clustered on the 11 folds. No player-level inspection enters
the decision.
"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
ROOT="/Users/thomasmcnamee/NFL"; H=2.0
YEARS=list(range(2015,2026))
panel=pd.read_csv(f"{ROOT}/results/market_prior_11yr.csv")

cols=["player_id","player_display_name","position","season","week","team","season_type","targets","attempts","fantasy_points_ppr"]
wk=[]
for y in range(2014,2026):
    d=pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",usecols=cols,low_memory=False)
    wk.append(d[d.season_type=="REG"])
wk=pd.concat(wk,ignore_index=True)

# ---- primary QB per team-season, and the set of starter games ----
qb=wk[(wk.position=="QB")&(wk.attempts.notna())]
prim=(qb.groupby(["season","team","player_id"]).attempts.sum().reset_index()
        .sort_values("attempts",ascending=False).drop_duplicates(["season","team"]))
prim=prim.rename(columns={"player_id":"qb_id"})[["season","team","qb_id"]]
qs=qb.merge(prim,left_on=["season","team","player_id"],right_on=["season","team","qb_id"])
starter=set(zip(qs[qs.attempts>=10].season, qs[qs.attempts>=10].team, qs[qs.attempts>=10].week))
print(f"team-seasons with an identified primary QB: {len(prim)}; starter games: {len(starter)}")

pl=wk[wk.targets>1].copy()
pl["starter_game"]=[ (s,t,w) in starter for s,t,w in zip(pl.season,pl.team,pl.week) ]
print(f"player-games: {len(pl)}, of which starter-QB games: {pl.starter_game.mean():.1%}")

seas=(pl.groupby(["player_id","season"])
        .agg(G=("fantasy_points_ppr","size"), ybar=("fantasy_points_ppr","mean")).reset_index())
seas_qb=(pl[pl.starter_game].groupby(["player_id","season"])
        .agg(Gq=("fantasy_points_ppr","size"), ybar_qb=("fantasy_points_ppr","mean")).reset_index())
S=seas.merge(seas_qb,on=["player_id","season"],how="left")
S["fallback"]=~(S.Gq>=4)
S["ybar_eff"]=np.where(S.fallback, S.ybar, S.ybar_qb)
print(f"season rows: {len(S)}, fell back to unadjusted (<4 starter games): {S.fallback.mean():.1%}")
print(f"mean lift from QB-adjustment on non-fallback rows: "
      f"{(S.loc[~S.fallback,'ybar_qb']-S.loc[~S.fallback,'ybar']).mean():+.3f} PPG")

# sigma^2(tier), unchanged from the incumbent
wr=wk[(wk.position=="WR")&(wk.targets>1)].copy()
ps=wr.groupby(["player_id","season"]).agg(mt=("targets","mean"),mu=("fantasy_points_ppr","mean")).reset_index()
wr=wr.merge(ps[ps.mt>=3.0],on=["player_id","season"])
meta=pd.read_csv(f"{ROOT}/data/meta/players_meta.csv",low_memory=False,usecols=["gsis_id","rookie_season"]).dropna()
wr=wr.merge(meta.rename(columns={"gsis_id":"player_id"}),on="player_id").dropna(subset=["rookie_season"])
wr["e2"]=(wr.fantasy_points_ppr-wr.mu)**2
wr["tier"]=np.select([wr.season-wr.rookie_season==0,wr.season-wr.rookie_season==1],["rookie","soph"],"vet")

def wvar(x,w):
    mu=np.sum(w*x)/np.sum(w); return float(np.sum(w*(x-mu)**2)/(np.sum(w)-np.sum(w**2)/np.sum(w)))
def mu_neff(gsis,Y,col):
    h=S[(S.player_id==gsis)&(S.season<Y)]
    if len(h)==0: return np.nan,0.0
    m=h.season.max(); w=2.0**(-(m-h.season.values)/1.0)
    return float((w*h[col].values).sum()/w.sum()), float(w.sum()**2/(w**2).sum())

preds=[]
for Y in YEARS:
    tr=panel[(panel.year!=Y)&panel.in_fit].copy(); ev=panel[(panel.year==Y)&panel.in_fit].copy()
    w=2.0**(-np.abs(tr.year.values-Y)/H)
    iso=IsotonicRegression(increasing=False,out_of_bounds="clip"); iso.fit(np.log(tr.adp),tr.ppg,sample_weight=w)
    r=tr.ppg.values-iso.predict(np.log(tr.adp.values))
    tau2={t:(wvar(r[(tr.tier==t).values],w[(tr.tier==t).values]) if (tr.tier==t).sum()>=2 else np.nan) for t in ["rookie","soph","vet"]}
    pooled=wvar(r,w)
    ev["m_hat"]=iso.predict(np.log(ev.adp.values))
    ev["tau2"]=ev.tier.map(tau2).fillna(pooled)
    ev["sig2"]=ev.tier.map(wr[wr.season!=Y].groupby("tier").e2.mean())
    for tag,col in [("mu","ybar"),("qb","ybar_eff")]:
        mn=ev.gsis_id.map(lambda g: mu_neff(g,Y,col))
        mh=np.array([t[0] for t in mn]); ne=np.array([t[1] for t in mn])
        with np.errstate(divide="ignore"): V=ev.sig2.values/ne
        B=np.where(ne==0,1.0,V/(V+ev.tau2.values))
        ev[f"th_{tag}"]=np.where(ne==0,ev.m_hat,(1-B)*np.nan_to_num(mh)+B*ev.m_hat)
    preds.append(ev)
pred=pd.concat(preds,ignore_index=True)

def dm(a,b):
    d=(pred.ppg-pred[b])**2-(pred.ppg-pred[a])**2
    dy=d.groupby(pred.year).mean(); t=dy.mean()/(dy.std(ddof=1)/np.sqrt(len(dy)))
    return float(t), float(2*stats.t.sf(abs(t),df=len(dy)-1))
print(f"\n=== §T scorecard, 11 folds, {len(pred)} rows ===")
for lab,c in [("(i)  m_hat","m_hat"),("(ii) theta* mu  [incumbent]","th_mu"),("(ix) theta* mu_qb [candidate]","th_qb")]:
    rmse=float(np.sqrt(((pred.ppg-pred[c])**2).mean()))
    rho=pred.groupby("year").apply(lambda g: stats.spearmanr(g[c],g.ppg).statistic,include_groups=False).mean()
    extra=""
    if c=="th_qb":
        t,p=dm("th_qb","th_mu"); extra=f"   DM vs (ii): t={t:+.3f}, p={p:.4f}"
    print(f"{lab:<32} RMSE {rmse:.4f}   Spearman {rho:.4f}{extra}")
pred.to_csv(f"{ROOT}/results/sectionT_qbadj_loso.csv",index=False)

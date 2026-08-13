"""Positional (vs-WR, vs-RB) preseason-knowable SOS: season t-1 defensive PPR allowed
per game, mapped onto season t's schedule grid. Appended to sos_history_2015_2026.csv."""
import pandas as pd, numpy as np
ROOT="/Users/thomasmcnamee/NFL"
# 1. defensive PPR allowed per game, by season, by position
recs=[]
for yr in range(2014,2026):
    d=pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{yr}.csv",low_memory=False)
    d=d[(d.season_type=='REG')&d.position.isin(['WR','RB'])]
    for pos in ['WR','RB']:
        s=d[d.position==pos].groupby(['opponent_team','week']).fantasy_points_ppr.sum()
        m=s.groupby('opponent_team').mean()
        for tm,v in m.items(): recs.append((yr,tm,pos,v))
dfa=pd.DataFrame(recs,columns=['season','abbr','pos','fpa_pg'])
print('def-FPA seasons',sorted(dfa.season.unique()),'teams/season',dfa.groupby(['season','pos']).size().unique())

# 2. schedule grid long
g=pd.read_csv(f"{ROOT}/data/teams/games_nflverse_20260809.csv",usecols=['season','game_type','week','home_team','away_team'])
g=g[(g.game_type=='REG')&g.season.between(2015,2026)]
a=g.rename(columns={'home_team':'team','away_team':'opp'})[['season','week','team','opp']]
b=g.rename(columns={'away_team':'team','home_team':'opp'})[['season','week','team','opp']]
L=pd.concat([a,b],ignore_index=True)
# nflverse player stats normalise franchises to their CURRENT abbreviation for all seasons;
# the schedule grid uses the era-correct one. Normalise the grid's OPPONENT key to match.
NORM={'STL':'LA','SD':'LAC','OAK':'LV','LAR':'LA'}
L['opp']=L.opp.replace(NORM)

out=None
for pos in ['WR','RB']:
    p=dfa[dfa.pos==pos].copy(); p['season']=p.season+1          # PRIOR season -> known in August
    m=L.merge(p.rename(columns={'abbr':'opp','fpa_pg':'opp_fpa'})[['season','opp','opp_fpa']],
              on=['season','opp'],how='left')
    print(pos,'missing opp FPA:',m.opp_fpa.isna().sum(),
          m[m.opp_fpa.isna()].groupby('season').size().to_dict())
    full=m.groupby(['season','team'],as_index=False).opp_fpa.mean().rename(columns={'opp_fpa':f'sos_{pos.lower()}_fpa'})
    plf =m[m.week.between(15,17)].groupby(['season','team'],as_index=False).opp_fpa.mean().rename(columns={'opp_fpa':f'sos_{pos.lower()}_fpa_w15_17'})
    j=full.merge(plf,on=['season','team'])
    out=j if out is None else out.merge(j,on=['season','team'])

h=pd.read_csv(f"{ROOT}/data/schedule/sos_history_2015_2026.csv")
h=h.drop(columns=[c for c in h.columns if c.startswith(('sos_wr_fpa','sos_rb_fpa'))],errors='ignore')
h=h.merge(out,on=['season','team'],how='left')
h.to_csv(f"{ROOT}/data/schedule/sos_history_2015_2026.csv",index=False)
print('final',h.shape, h.columns.tolist())
print(h.groupby('season')[['sos_wr_fpa','sos_wr_fpa_w15_17','sos_rb_fpa']].agg(['count','std']).round(3).to_string())
# how different is positional SOS from team-quality SOS?
from scipy.stats import spearmanr
hh=h[h.season.between(2015,2024)]
print('\nwithin-season Spearman, sos_wr_fpa vs sos_vegas:')
print(hh.groupby('season').apply(lambda x: spearmanr(x.sos_wr_fpa,x.sos_vegas).statistic,include_groups=False).round(2).to_string())
print('\nyear-over-year persistence of a defense FPA vs WR (the whole premise):')
w=dfa[dfa.pos=='WR'].pivot(index='abbr',columns='season',values='fpa_pg')
print(pd.Series({y: w[y].corr(w[y-1]) for y in range(2015,2026)}).round(3).to_string())

"""§S3 — ADP movement Jul 13 -> Aug 18 2026, joined on NORMALIZED names.

Supersedes the raw-`name` join used in the first pass, which produced false
"new to the board" flags (FFC renamed 'Kenneth Walker III' -> 'Kenneth Walker'
between pulls).  Uses script 07's norm_name, the project's standing convention.
Also joins ESPN live ranks to measure the FFC trailing-window lag per player.
"""
import re, pandas as pd, numpy as np
SUF={"jr","sr","ii","iii","iv","v"}
def nn(s):
    s=re.sub(r"[^a-z ]","",str(s).lower().replace("."," ").replace("-"," ").replace("'",""))
    return " ".join(t for t in s.split() if t not in SUF)

jul=pd.read_csv('data/adp/adp_ppr_2026_all.csv')[['name','position','team','adp']]
aug8=pd.read_csv('data/adp/adp_ppr_2026_all_20260809.csv')[['name','adp']]
aug18=pd.read_csv('data/adp/adp_ppr_2026_all_20260818.csv')[['name','position','team','adp','stdev','times_drafted']]
esp=pd.read_csv('data/adp/adp_espn_2026_20260818.csv')
for d in (jul,aug8,aug18,esp): d['k']=d.name.map(nn)
jul=jul.rename(columns={'adp':'adp_jul','team':'team_jul'}).drop_duplicates('k')
aug8=aug8.rename(columns={'adp':'adp_aug08'}).drop_duplicates('k')
aug18=aug18.rename(columns={'adp':'adp_aug18'}).drop_duplicates('k')

m=(aug18.merge(jul[['k','adp_jul','team_jul']],on='k',how='outer')
        .merge(aug8[['k','adp_aug08']],on='k',how='left')
        .merge(esp[['k','espn_rank']],on='k',how='left'))
m['d_jul_aug18']=m.adp_aug18-m.adp_jul
m['ffc_rank']=m.adp_aug18.rank()
m['espn_gap']=m.ffc_rank-m.espn_rank            # +ve = FFC still cheap (lagging a rise)
m['team_change']=(m.team.notna()&m.team_jul.notna()&(m.team!=m.team_jul))
top=m[m.adp_aug18<=120]

print(f"rows {len(m)}; matched both windows {m.adp_jul.notna().sum()}")
print("\n=== genuinely NEW to top 120 since July (normalized join) ===")
print(top[top.adp_jul.isna()][['name','position','team','adp_aug18']].to_string(index=False))
print("\n=== team changes between pulls ===")
print(m[m.team_change][['name','position','team_jul','team','adp_jul','adp_aug18','d_jul_aug18']].to_string(index=False))
print("\n=== biggest RISERS (top 120) ===")
print(top.nsmallest(12,'d_jul_aug18')[['name','position','team','adp_jul','adp_aug08','adp_aug18','d_jul_aug18','espn_rank','espn_gap']].to_string(index=False))
print("\n=== biggest FALLERS (top 120) ===")
print(top.nlargest(12,'d_jul_aug18')[['name','position','team','adp_jul','adp_aug08','adp_aug18','d_jul_aug18','espn_rank','espn_gap']].to_string(index=False))
print("\n=== FFC most out of step with ESPN today (|gap|, top 120) ===")
t2=top.dropna(subset=['espn_gap']).copy(); t2['ag']=t2.espn_gap.abs()
print(t2.nlargest(12,'ag')[['name','position','team','adp_aug18','ffc_rank','espn_rank','espn_gap']].to_string(index=False))
m.to_csv('results/adp_movement_2026_fixed.csv',index=False)

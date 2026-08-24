"""§S6 — §J overlay on the refreshed board: 11-season panels, recency prior h=2,
ADP window 08-11->08-18, views_2026_v2.csv (31 views, 8 updated/new 2026-08-18)."""
import sys; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import import_module
bl=import_module("19_bl_overlay")
R=Path("/Users/thomasmcnamee/NFL/results"); TAU=0.5
board=pd.read_csv(R/"board_2026_pi_sigma_h2.csv")
views=pd.read_csv(R/"views_2026_v2.csv")
missing=set(views.player)-set(board.name)
if missing: raise KeyError(f"views off board: {sorted(missing)}")
names=board.name.tolist(); pi=board.pi.to_numpy(float); Sigma=np.diag(board.sig.to_numpy(float))
P,q,Om,ids=bl.build_views(views,names,pi,Sigma,TAU)
theta,M,contrib=bl.posterior(pi,Sigma,P,q,Om,TAU)
out=board.copy(); out["theta_bar"]=theta; out["shift"]=theta-pi; out["post_SD"]=np.sqrt(np.diag(M))
out=out.sort_values("theta_bar",ascending=False).reset_index(drop=True)
out["rank"]=range(1,len(out)+1)
out["mkt_rank"]=out.adp.rank().astype(int)
out.to_csv(R/"board_2026_v2_with_views.csv",index=False)
pd.set_option("display.width",200)
print(out[["rank","name","team","position","adp","pi","theta_bar","shift","post_SD"]].round(2).to_string(index=False))
print("\n=== largest view-driven shifts ===")
s=out.reindex(out["shift"].abs().sort_values(ascending=False).index)
print(s.head(14)[["name","team","position","adp","pi","theta_bar","shift"]].round(2).to_string(index=False))

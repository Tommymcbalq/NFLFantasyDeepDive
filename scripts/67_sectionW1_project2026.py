"""§W1 — fit the projection on 2015-2024 and produce 2026 preseason projections.

Fits on ALL panel rows (2015-2024, in_fit, n_eff>0), then applies to the 2026 WR/RB
ADP pool with features built from seasons < 2026.  This is an INPUT for WS2; it writes
only results/sectionW1_projection_2026.csv and touches no board file.

Environment uses the PRIOR-SEASON team, matching the fitted model exactly (see the leak
note in script 64: the historical panel's `team` field is an end-of-season label, so the
fitted model could never use a forward-looking team).  Rows with no prior history are
dropped -- eq. (7) gives them B = 1 and the market arm owns them.

NOTE for WS2: `mu_hat` here is computed over ALL prior games, not under the §0 inclusion
rule, so it will differ by ~0.01-0.05 PPG from the pipeline's mu_hat.  Use the pipeline's.
"""
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
ROOT = Path("/Users/thomasmcnamee/NFL")
sp = importlib.util.spec_from_file_location(
    "w1", ROOT / "scripts/64_sectionW1_projection.py")
w1 = importlib.util.module_from_spec(sp)
sp.loader.exec_module(w1)
sp2 = importlib.util.spec_from_file_location(
    "fb", ROOT / "scripts/63_sectionW1_features.py")
fb = importlib.util.module_from_spec(sp2)
sp2.loader.exec_module(fb)

ADP = pd.read_csv(ROOT / "data/adp/adp_ppr_2026_all_20260824.csv")
ADP.columns = [c.lower() for c in ADP.columns]
NAMECOL = "name" if "name" in ADP.columns else "player"
POSCOL = "position" if "position" in ADP.columns else "pos"


def build_2026(pos):
    a = ADP[ADP[POSCOL].astype(str).str.upper() == pos].copy()
    a["year"] = 2026
    a["pos"] = pos
    a = a.rename(columns={NAMECOL: "name"})
    # resolve gsis_id by normalised name among players active in 2024/2025
    meta = pd.read_csv(ROOT / "data/meta/players_meta.csv", low_memory=False)
    act = fb.PS[fb.PS.season >= 2023].player_id.unique()
    meta = meta[meta.gsis_id.isin(act)]

    def norm(s):
        return (s.astype(str).str.lower().str.replace(r"[^a-z ]", "", regex=True)
                .str.replace(r"\s+(jr|sr|ii|iii|iv|v)$", "", regex=True).str.strip())
    meta["k"] = norm(meta.display_name)
    a["k"] = norm(a.name)
    a = a.drop_duplicates("k")
    a = a.merge(meta.drop_duplicates("k")[["k", "gsis_id"]], on="k", how="left")
    unmatched = a[a.gsis_id.isna()][["name"]].assign(pos=pos)
    unmatched.to_csv(ROOT / f"data/derived/_w1_2026_unmatched_{pos}.csv", index=False)
    a = a.dropna(subset=["gsis_id"])
    a["team"] = a.get("team", pd.Series(index=a.index, dtype=object))
    a["in_fit"] = True
    a["adp"] = a["adp"].astype(float)
    a["adp_rank"] = a.adp.rank(method="first")
    a["ppg"] = np.nan
    a["games"] = np.nan
    a["tier"] = "vet"
    tmp = ROOT / f"data/derived/_w1_panel2026_{pos}.csv"
    a[["year", "name", "pos", "adp", "team", "gsis_id", "games", "ppg", "tier",
       "in_fit", "adp_rank"]].rename(columns={"gsis_id": "pid"}).to_csv(tmp, index=False)
    d = fb.build(str(tmp.relative_to(ROOT)), pos)
    tmp.unlink()
    return d


if __name__ == "__main__":
    F = w1.load("A")
    fit = F[F.in_fit & (F.n_eff > 0)].copy()
    out = []
    for pos in ["WR", "RB"]:
        n26 = build_2026(pos)
        # mu_hat on the 2026 rows: recency-weighted mean of season means, inclusion rule
        n26 = n26.rename(columns={"h_ppg": "_h_ppg"})
        n26["mu_hat"] = n26["_h_ppg"]
        tr = fit[fit.pos == pos]
        for scope in ["P0", "P1"]:
            cols = [c for c in w1.featlist("A", scope, pos, True) if c in tr.columns]
            miss = [c for c in cols if c not in n26.columns]
            for c in miss:
                n26[c] = np.nan
            Xtr, Xev, names = w1.design(tr, n26, cols)
            y = tr.ppg.values
            a, _ = w1.grouped_cv_alpha(Xtr, y, tr.year.values)
            c_, m0 = w1.ridge_fit(Xtr, y, a)
            n26[f"proj_ridge_{scope}"] = m0 + Xev @ c_
            if scope == "P1":
                n26["alpha"] = a
        b = np.polyfit(tr.mu_hat.values, tr.ppg.values, 1)
        n26["mu_cal"] = b[1] + b[0] * n26.mu_hat.values
        out.append(n26[["gsis_id", "name", "pos", "adp", "adp_rank", "age", "expr",
                        "n_prior", "G_last", "avail_wtd", "mu_hat", "mu_cal",
                        "proj_ridge_P0", "proj_ridge_P1", "alpha"]])
    O = pd.concat(out, ignore_index=True).sort_values(["pos", "adp"])
    O = O[O.n_prior > 0]   # no history -> the market arm owns them (eq. 7, B = 1)
    O.to_csv(ROOT / "results/sectionW1_projection_2026.csv", index=False)
    print(f"wrote {len(O)} rows; unmatched gsis_id: {O.gsis_id.isna().sum()}")
    pd.set_option("display.width", 220)
    for pos in ["WR", "RB"]:
        print(f"\n=== {pos} top 15 by ADP ===")
        print(O[O.pos == pos].head(15).round(3).to_string(index=False))

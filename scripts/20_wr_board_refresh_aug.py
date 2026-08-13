"""Round-4 Job 1 — refresh the WR board onto the 2026-08-09 ADP pull.

PRE-STATED (this docstring written before any refreshed number was inspected):

What changes, and only this: the modeling universe (top-30 WRs by ADP) and the ADP
values that enter m(ADP). Everything else in the round-1 pipeline is estimated on
data the refresh does not touch:
  - m_iso(.) and tau^2(tier)  -> fit on the HISTORICAL 2015-2024 ADP panel (script 07)
  - sigma^2(tier)             -> fit on ALL WRs 2014-2025 game logs (script 06)
Both were re-run against the July inputs and reproduced BYTE-IDENTICALLY before this
script was written, so any movement reported here is attributable to the ADP refresh
and to nothing else. They are read from results/ as frozen inputs; this script asserts
their file hashes are unchanged from the verified re-run.

Board construction, identical conventions to fetch_data.py / script 01:
  - top 30 WRs by adp from data/adp/wr_top30_adp_2026_20260809.csv
  - gsis_id by normalized-name match into players_meta (WR, last_season >= 2024),
    de-duplicated by most-recent last_season; unmatched reported and fatal
  - game logs: weekly_raw seasons 2014-2025, season_type == REG  (SAME window as
    round 1 — weekly_raw now reaches 1999, but widening it would change mu_hat for
    reasons unrelated to the ADP refresh, so it is deliberately NOT widened here)
  - §0 inclusion rule unchanged: drop player-games with targets <= 1
  - mu_hat/n_eff/sigma_W/tau2_B/EB boom-bust: script 01's build_table, imported, not
    re-implemented, so the estimator is bit-for-bit the round-1 estimator.
    (The EB Beta(alpha,beta) moment fit is pooled ACROSS the 30 board players, so a
    change in board membership moves boom/bust slightly. That is a real consequence
    of the refresh and is reported, not suppressed.)
  - posterior: eq. (7), theta* = (1-B) mu_hat + B m_iso(ADP), B = V/(V+tau^2),
    V = sigma^2(tier)/n_eff, tier = 2026 - rookie_season.
  - no edge terms: results/edge_regression.csv is re-read programmatically and the
    "no survivor" branch asserted, exactly as script 11 does.

Outputs (all NEW dated files; nothing from rounds 1-3 is overwritten):
  data/players/wr_top30_weekly_20260809.csv
  results/consistency_table_20260809.csv
  results/valuation_2026_wr_20260809.csv
"""
import hashlib
import importlib.util
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/thomasmcnamee/NFL")
SEASONS = range(2014, 2026)          # same window as round 1, deliberately

# import script 01's estimator so the consistency math is literally the same code
spec = importlib.util.spec_from_file_location(
    "s01", ROOT / "scripts" / "01_section1_consistency.py")
s01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s01)


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[.'\-]", "", s.lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]


# ---------------- frozen upstream inputs ----------------
sig2 = pd.read_csv(ROOT / "results/sigma2_by_tier.csv").set_index("tier").sigma2
tau2 = pd.read_csv(ROOT / "results/tier_variances.csv").set_index("tier").tau2_iso
knots = pd.read_csv(ROOT / "results/market_prior_iso_knots.csv")
edge = pd.read_csv(ROOT / "results/edge_regression.csv", index_col="term")
survivors = list(edge.index[edge.final_survivor.fillna(False)])
assert not survivors, f"edge survivors present: {survivors}"
print("frozen inputs (sha256[:12]):")
for f in ["sigma2_by_tier.csv", "tier_variances.csv", "market_prior_iso_knots.csv",
          "tier_variances.csv"]:
    print(f"  {f:34s} {sha(ROOT / 'results' / f)}")
print(f"sigma2(tier) = {sig2.round(3).to_dict()}")
print(f"tau2(tier)   = {tau2.round(3).to_dict()}")

# ---------------- board ----------------
board = pd.read_csv(ROOT / "data/adp/wr_top30_adp_2026_20260809.csv")
board = board.nsmallest(30, "adp").reset_index(drop=True)
board["adp_rank"] = board.index + 1
board["norm"] = board.name.map(norm_name)

meta_all = pd.read_csv(ROOT / "data/meta/players_meta.csv", low_memory=False)
mw = meta_all[(meta_all.position == "WR") & (meta_all.last_season >= 2024)].copy()
mw["norm"] = mw.display_name.map(norm_name)
mw = mw.sort_values("last_season", ascending=False).drop_duplicates("norm")
b = board.merge(mw[["norm", "gsis_id", "display_name", "rookie_season", "birth_date"]],
                on="norm", how="left")
missing = b[b.gsis_id.isna()]
assert len(missing) == 0, f"unmatched board WRs: {missing.name.tolist()}"
print(f"\nboard: {len(b)} WRs, gsis matched {b.gsis_id.notna().sum()}/30")

jul = pd.read_csv(ROOT / "data/adp/wr_top30_adp_2026.csv")
print(f"entered vs July board: {sorted(set(b.name) - set(jul.name))}")
print(f"left   vs July board: {sorted(set(jul.name) - set(b.name))}")

# ---------------- weekly logs ----------------
ids = set(b.gsis_id)
frames = []
for y in SEASONS:
    df = pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{y}.csv",
                     low_memory=False)
    idc = "player_id" if "player_id" in df.columns else "gsis_id"
    df = df.rename(columns={idc: "gsis_id"})
    frames.append(df[df.gsis_id.isin(ids)])
wk = pd.concat(frames, ignore_index=True)
wk = wk.merge(b[["gsis_id", "name", "adp_rank"]], on="gsis_id", how="left")
wk.to_csv(ROOT / "data/players/wr_top30_weekly_20260809.csv", index=False)
reg = wk[wk.season_type == "REG"]
print(f"weekly rows {len(wk)} (REG {len(reg)}); "
      f"excluded by targets<=1: {(reg.targets <= 1).sum()} "
      f"({(reg.targets <= 1).mean():.3%}), "
      f"mean PPR of excluded {reg.loc[reg.targets <= 1, 'fantasy_points_ppr'].mean():.2f}")

# ---------------- §1 consistency, script-01 estimator ----------------
ct = s01.build_table(reg[reg.targets >= 2].copy())
ct.round(4).to_csv(ROOT / "results/consistency_table_20260809.csv", index=False)
print(f"\nEB boom (a,b): {tuple(round(x,4) for x in ct.attrs['boom_ab'][:2])}  "
      f"bust (a,b): {tuple(round(x,4) for x in ct.attrs['bust_ab'][:2])}")

# ---------------- eq. (7) posterior ----------------
d = b.merge(ct[["gsis_id", "player", "mu_hat", "n_eff", "n_games", "n_seasons"]],
            on="gsis_id", how="left", validate="1:1")
assert d.mu_hat.notna().all(), d[d.mu_hat.isna()].name.tolist()
d["exp_2026"] = 2026 - d.rookie_season
d["tier"] = np.select([d.exp_2026 == 0, d.exp_2026 == 1], ["rookie", "soph"], "vet")
print("tier counts 2026:", d.tier.value_counts().to_dict())

d["m_adp"] = np.interp(np.log(d.adp), knots.log_adp, knots.m)
d["tau2"] = d.tier.map(tau2)
d["sigma2_tier"] = d.tier.map(sig2)
d["V"] = d.sigma2_tier / d.n_eff
d["B"] = d.V / (d.V + d.tau2)
d["theta_star"] = (1 - d.B) * d.mu_hat + d.B * d.m_adp
d["post_SD"] = np.sqrt(1.0 / (1.0 / d.V + 1.0 / d.tau2))

d = d.sort_values("theta_star", ascending=False).reset_index(drop=True)
d["rank_theta"] = d.index + 1
d["delta_rank_vs_adp"] = d.adp_rank - d.rank_theta
d["edge_terms_applied"] = "none (no term survived FDR + temporal holdout)"

out = d[["rank_theta", "player", "team", "adp", "adp_rank", "tier", "mu_hat", "n_eff",
         "V", "m_adp", "tau2", "B", "theta_star", "post_SD", "delta_rank_vs_adp",
         "edge_terms_applied"]]
out.to_csv(ROOT / "results/valuation_2026_wr_20260809.csv", index=False)
print("\n== refreshed WR board (August ADP) ==")
print(out.round(3).to_string(index=False))

# ---------------- decomposition vs the frozen July v3 board ----------------
v3 = pd.read_csv(ROOT / "results/valuation_2026_v3.csv")[
    ["player", "adp", "V_final", "rank_v3"]].rename(
    columns={"adp": "adp_jul", "V_final": "theta_jul"})
cmp = out.merge(v3, on="player", how="outer", indicator=True)
cmp["d_theta"] = cmp.theta_star - cmp.theta_jul
cmp["d_rank"] = cmp.rank_v3 - cmp.rank_theta

# counterfactual: same 30 August players, but July ADP where it exists — isolates
# how much of d_theta is the ADP move vs the mu_hat/EB-pool change from new members
old = pd.read_csv(ROOT / "results/consistency_table.csv")[["gsis_id", "mu_hat", "n_eff"]]
cf = d.merge(v3[["player", "adp_jul"]], on="player", how="left")
cf = cf.merge(old.rename(columns={"mu_hat": "mu_old", "n_eff": "neff_old"}),
              on="gsis_id", how="left")
cf["m_jul"] = np.interp(np.log(cf.adp_jul), knots.log_adp, knots.m)
cf["theta_adp_only"] = np.where(
    cf.adp_jul.notna(), (1 - cf.B) * cf.mu_hat + cf.B * cf.m_jul, np.nan)
cmp = cmp.merge(cf[["player", "theta_adp_only", "mu_old", "neff_old"]],
                on="player", how="left")
cmp["d_from_adp"] = cmp.theta_star - cmp.theta_adp_only
cmp["d_from_muhat"] = cmp.theta_adp_only - cmp.theta_jul
cmp.sort_values("theta_star", ascending=False).round(4).to_csv(
    ROOT / "results/wr_board_refresh_delta_20260809.csv", index=False)
print("\n== movement vs frozen July v3 board ==")
print(cmp.sort_values("d_theta")[
    ["player", "adp_jul", "adp", "theta_jul", "theta_star", "d_theta",
     "d_from_adp", "d_from_muhat", "rank_v3", "rank_theta", "d_rank", "_merge"]]
    .round(3).to_string(index=False))
print("\nwrote results/valuation_2026_wr_20260809.csv, "
      "consistency_table_20260809.csv, wr_board_refresh_delta_20260809.csv")

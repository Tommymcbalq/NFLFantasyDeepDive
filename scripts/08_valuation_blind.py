"""§3.4 posterior valuation of the 2026 top-30 WR board — COVARIATE-FREE / BLIND.

Pre-registered formula (EDA_PLAN.md eq. 7), every input estimated upstream:
    prior:      theta_i ~ N( m_iso(ADP_i), tau^2(tier_i) )        [scripts/07, §6.1]
    likelihood: mu_hat_i ~ N( theta_i, V_i ),  V_i = sigma^2(tier_i)/n_eff_i
                sigma^2(tier) from Route B gamma GLM              [scripts/06, §3]
                mu_hat (h=1), n_eff from consistency_table        [scripts/01, §1]
    posterior:  theta* = (1-B) mu_hat + B m,  B = V/(V+tau^2)
                post_SD = sqrt( (1/V + 1/tau^2)^(-1) )
Tier at 2026 = 2026 - rookie_season (players_meta). Rookies with no NFL games
would get B = 1 (verified none on this board). No manual adjustments of any kind.

Output: results/valuation_2026_blind.csv (ranked by theta*).
"""
import numpy as np
import pandas as pd

ROOT = "/Users/thomasmcnamee/NFL"

board = pd.read_csv(f"{ROOT}/data/adp/wr_top30_adp_2026.csv")[
    ["wr_adp_rank", "name", "adp"]].rename(columns={"wr_adp_rank": "adp_rank"})
ct = pd.read_csv(f"{ROOT}/results/consistency_table.csv")[
    ["gsis_id", "player", "mu_hat", "n_eff", "n_games", "n_seasons"]]
meta = pd.read_csv(f"{ROOT}/data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "rookie_season"])
sig2 = pd.read_csv(f"{ROOT}/results/sigma2_by_tier.csv").set_index("tier").sigma2
tau2 = pd.read_csv(f"{ROOT}/results/tier_variances.csv").set_index("tier").tau2_iso
knots = pd.read_csv(f"{ROOT}/results/market_prior_iso_knots.csv")

d = board.merge(ct, left_on="name", right_on="player", how="left", validate="1:1")
assert d.gsis_id.notna().all(), "board/consistency-table name mismatch"
d = d.merge(meta, on="gsis_id", how="left")
d["exp_2026"] = 2026 - d.rookie_season
d["tier"] = np.select([d.exp_2026 == 0, d.exp_2026 == 1], ["rookie", "soph"], "vet")

# verify: no true rookies (0 NFL games) on the board
zero_g = d[(d.n_games == 0) | d.n_games.isna()]
print(f"players with 0 NFL games: {len(zero_g)} (B=1 branch unused)"
      if len(zero_g) == 0 else zero_g)
print("tier counts 2026:", d.tier.value_counts().to_dict())

# m(ADP): interpolate the isotonic step function on log ADP
d["m_adp"] = np.interp(np.log(d.adp), knots.log_adp, knots.m)

d["tau2"] = d.tier.map(tau2)
d["sigma2_tier"] = d.tier.map(sig2)
d["V"] = d.sigma2_tier / d.n_eff
d["B"] = d.V / (d.V + d.tau2)
d["theta_star"] = (1 - d.B) * d.mu_hat + d.B * d.m_adp
d["post_SD"] = np.sqrt(1.0 / (1.0 / d.V + 1.0 / d.tau2))

d = d.sort_values("theta_star", ascending=False).reset_index(drop=True)
d["rank_theta"] = d.index + 1
d["delta_rank_vs_adp"] = d.adp_rank - d.rank_theta   # + = riser vs market

out = d[["rank_theta", "player", "adp", "adp_rank", "tier", "mu_hat", "n_eff",
         "V", "m_adp", "tau2", "B", "theta_star", "post_SD", "delta_rank_vs_adp"]]
out.to_csv(f"{ROOT}/results/valuation_2026_blind.csv", index=False)
print(out.round(3).to_string(index=False))

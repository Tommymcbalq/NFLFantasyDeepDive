"""§6.4 final 2026 valuation (EDA_PLAN.md §6.4).

V_i per eq (7) with prior mean m_iso(ADP_i) PLUS any §6.2/§6.3 edge terms that
survived BOTH the BH-FDR gate (q=0.10, season-clustered) and the binding
temporal holdout (fit 2015-2022, evaluate 2023-2024) — read programmatically
from results/edge_regression.csv (final_survivor). Variances (sigma^2(tier),
tau^2(tier)), mu_hat, n_eff identical to the blind run (scripts/08).

OUTCOME (§6.2): no term survived both gates (rookie and rook_x_epa passed FDR
via degenerate few-treated-cluster inference on n=4 rookies, then FAILED the
temporal holdout and blew up in LOSO). Therefore the final valuation IS the
blind posterior: V = theta*, stated plainly, no adjustment invented.

Output: results/valuation_2026_final.csv
"""
import numpy as np
import pandas as pd

ROOT = "/Users/thomasmcnamee/NFL"

edge = pd.read_csv(f"{ROOT}/results/edge_regression.csv", index_col="term")
survivors = list(edge.index[edge.final_survivor.fillna(False)])
print(f"edge terms surviving FDR + temporal holdout: {survivors or 'NONE'}")

blind = pd.read_csv(f"{ROOT}/results/valuation_2026_blind.csv")

if survivors:
    raise SystemExit(
        "survivors present — implement the prior-mean adjustment "
        f"m + Z'beta for {survivors} before writing the final board")

out = blind.rename(columns={"theta_star": "theta_star_blind"}).copy()
out["V_final"] = out.theta_star_blind
out = out.sort_values("V_final", ascending=False).reset_index(drop=True)
out["rank_final"] = out.index + 1
out["delta_rank_vs_adp"] = out.adp_rank - out.rank_final
# blind rank == final rank by construction when no survivors
out["delta_rank_vs_blind"] = out.rank_theta - out.rank_final
out["edge_terms_applied"] = "none (no term survived FDR + temporal holdout)"

cols = ["rank_final", "player", "adp", "adp_rank", "tier", "theta_star_blind",
        "V_final", "post_SD", "delta_rank_vs_adp", "delta_rank_vs_blind",
        "edge_terms_applied"]
out[cols].to_csv(f"{ROOT}/results/valuation_2026_final.csv", index=False)
print(out[cols].round(3).to_string(index=False))
print("\nfinal == blind posterior (no surviving edge terms); "
      "wrote results/valuation_2026_final.csv")

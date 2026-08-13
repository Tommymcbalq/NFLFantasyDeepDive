"""Apply the §J Black-Litterman views overlay to the combined 2026 WR+RB board.

WR pi comes from the refreshed August board's m_adp column (the fitted isotonic curve at
each slot); RB pi and Sigma come from §G3's export. Sigma diagonals follow eq. (26.2):
tier residual variance minus the per-game sampling component, floored at 25%.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
bl = import_module("19_bl_overlay")

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
TAU = 0.5

# ---- WR side
wr = pd.read_csv(R / "valuation_2026_wr_20260809.csv").rename(columns={"player": "name"})
tv = pd.read_csv(R / "tier_variances.csv").set_index("tier")["tau2_iso"]
cons = pd.read_csv(R / "consistency_table.csv")
sw = dict(zip(cons["player"], cons["sigma_W"] ** 2 / (cons["n_games"] / cons["n_seasons"])))
med = np.median(list(sw.values()))
# Prior = the FROZEN STATISTICAL BOARD (theta*, which already blends market prior and the
# player's own history at the EB weight B), with its own posterior variance. Views update
# that, not the raw market curve -- using m_adp here would discard the data arm entirely.
wr["pi"] = wr["theta_star"]
wr["sig"] = wr["post_SD"] ** 2
wr["position"] = "WR"

# ---- RB side
rb = pd.read_csv(R / "sectionJ_pi_sigma_rb.csv").rename(columns={"pi_ppg": "pi", "sigma_diag": "sig"})
rb["adp_rank"] = rb["rb_adp_rank"]

board = pd.concat([wr[["name", "team", "adp", "position", "pi", "sig"]],
                   rb[["name", "team", "adp", "position", "pi", "sig"]]], ignore_index=True)

views = pd.read_csv(R / "views_2026.csv")
missing = set(views.player) - set(board.name)
if missing:
    raise KeyError(f"views reference players not on the board: {sorted(missing)}")

names = board.name.tolist()
pi = board.pi.to_numpy(float)
Sigma = np.diag(board.sig.to_numpy(float))          # off-diagonal = 0, see §J1c

P, q, Om, ids = bl.build_views(views, names, pi, Sigma, TAU)
theta, M, contrib = bl.posterior(pi, Sigma, P, q, Om, TAU)

out = board.copy()
out["pi"] = pi
out["theta_bar"] = theta
out["shift"] = theta - pi
out["post_SD"] = np.sqrt(np.diag(M))
for t in (0.25, 1.0):
    P2, q2, Om2, _ = bl.build_views(views, names, pi, Sigma, t)
    th2, _, _ = bl.posterior(pi, Sigma, P2, q2, Om2, t)
    out[f"theta_tau{t}"] = th2
out = out.sort_values("theta_bar", ascending=False).reset_index(drop=True)
out.insert(0, "overall_rank", out.index + 1)
out.to_csv(R / "board_2026_with_views.csv", index=False)

moved = out[out["shift"].abs() > 1e-9]
print(f"tau={TAU}, {len(views.view_id.unique())} views, {len(moved)} players moved\n")
print(moved[["overall_rank", "name", "position", "adp", "pi", "theta_bar", "shift"]]
      .round(2).to_string(index=False))
print("\n--- full board, top 40 by posterior ---")
print(out.head(40)[["overall_rank", "name", "position", "adp", "pi", "theta_bar"]]
      .round(2).to_string(index=False))

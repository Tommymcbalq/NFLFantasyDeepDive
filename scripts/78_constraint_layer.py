"""Owner constraint layer: declared availability, confidence-weighted, calibrated.

The owner states P(player available at one of his picks). Those beliefs are the input --
they are NOT fitted, and they override the ADP-derived room order wherever they exist.

Two mechanisms:

  HARD ZEROS   players the owner says simply will not reach him. Removed before the pick.

  SOFT TARGETS a stated probability p at pick k, plus a confidence. A player's draft slot is
               Normal(mu, sigma). Confidence sets sigma (high confidence -> tight, so the
               owner's number is trusted almost literally; low confidence -> wide, so the
               belief is shrunk toward the ADP order). Then
                     P(slot >= k) = p   =>   mu = k - sigma * Phi^{-1}(1 - p)
               is the STARTING point only. Because availability is decided by RANK among all
               simultaneous draws rather than by the marginal, that closed form does not
               deliver p. So mu is then calibrated by fixed point until the SIMULATED
               availability equals the owner's stated number. Without that step the layer
               quietly reports something other than what he declared.

Usage: python3 scripts/78_constraint_layer.py
"""
import numpy as np, pandas as pd
from scipy.stats import norm

BOARD = 'results/board_LIVE.csv'
N     = 12000

# confidence -> sigma of the player's draft slot. Lower sigma = the owner's number is taken
# more literally; higher sigma = shrunk back toward the ADP-implied ordering.
SIG = {'high': 1.6, 'med': 3.0, 'low': 5.5}

# (player, owner's pick, stated P(available), confidence)
SOFT = [
    ('Christian McCaffrey',  5, 0.80, 'high'),   # owner: strict
    ('Puka Nacua',           5, 0.20, 'high'),
    ('Justin Jefferson',    16, 0.99, 'high'),   # owner: 100
    ('Saquon Barkley',      16, 0.70, 'high'),
    ('Ashton Jeanty',       16, 0.25, 'med'),
    ("De'Von Achane",       16, 0.15, 'med'),
    ('CeeDee Lamb',         16, 0.10, 'low'),
    ('Rashee Rice',         27, 0.70, 'high'),
    ('Malik Nabers',        27, 0.20, 'high'),
    ('Chris Olave',         27, 0.10, 'high'),
    ('Nico Collins',        27, 0.02, 'high'),
    ('Travis Etienne Jr.',  34, 0.72, 'high'),
    ('Kyren Williams',      34, 0.80, 'med'),    # owner: a bit lower than the ADP order says
    ('Garrett Wilson',      34, 0.70, 'low'),    # owner: goes earlier than ADP, low confidence
    ('Luther Burden III',   34, 0.70, 'low'),    # same
]

# Owner says these never reach him. Everything the room order already puts far in front of a
# pick is handled by the order itself; this list is for explicit owner overrides.
# Pick 5 is McCaffrey-or-Nacua and nothing else: owner declared 80/20 with no third
# outcome. Gibbs/Bijan/Chase were left on default dispersion and could therefore drift
# down to pick 5 in a draw, which is not a possibility he wants spent time on.
HARD_ZERO = ['Derrick Henry', 'George Pickens', 'Kenneth Walker', 'Chase Brown',
             'Omarion Hampton', 'Jeremiyah Love',
             'Jahmyr Gibbs', 'Bijan Robinson', "Ja'Marr Chase"]

MY = [5, 16, 27, 34, 47, 54, 65, 74]


def load():
    m = pd.read_csv(BOARD)
    m['ep'] = m.exp_pick.fillna(999.0)
    return m


def simulate(m, mu, sig, n=N, seed=7):
    """Return P(available) at each of the owner's picks, for every player."""
    rng = np.random.default_rng(seed)
    hit = np.zeros((len(m), len(MY)))
    for _ in range(n):
        rk = np.argsort(np.argsort(rng.normal(mu, sig)))
        for j, pk in enumerate(MY):
            hit[:, j] += rk >= pk - 1
    return hit / n


def main():
    m = load()
    name = m.name.to_numpy()
    IX = {n: i for i, n in enumerate(name)}
    mu = m.ep.to_numpy(float).copy()
    sig = np.where(mu <= 39, 1.40 * np.sqrt(np.pi / 2), 9.0)

    # A hard zero means "gone before my pick", NOT "gone first overall". Pinning them to
    # slot 1 piled six players at the front and displaced everyone else ~6 picks later,
    # inflating availability board-wide (it had Jonathan Taylor 76% to reach pick 16).
    # All six already sit ahead of the relevant pick in the room order, so the correct
    # operation is only to remove the upper tail where they survive: keep mu, shrink sigma.
    for p in HARD_ZERO:
        if p in IX:
            sig[IX[p]] = 0.30

    targets = []
    for p, pk, prob, conf in SOFT:
        if p not in IX:
            print(f"  !! {p} not on board"); continue
        i = IX[p]
        sig[i] = SIG[conf]
        prob = min(max(prob, 0.01), 0.99)          # Phi^{-1} is infinite at 0 and 1
        mu[i] = pk - sig[i] * norm.ppf(1 - prob)   # closed-form seed
        targets.append((i, MY.index(pk), min(max(prob,0.01),0.99), p))

    # Fixed-point calibration: nudge mu until simulated availability hits the stated number.
    for it in range(40):
        P = simulate(m, mu, sig, n=3000, seed=100 + it)
        err = max(abs(P[i, j] - q) for i, j, q, _ in targets)
        if err < 0.012:
            break
        for i, j, q, _ in targets:
            mu[i] -= 2.2 * (P[i, j] - q)           # too available -> push him EARLIER (lower mu)
    P = simulate(m, mu, sig, n=N, seed=99)
    print(f"calibration converged in {it+1} passes, max error {err:.3f}\n")
    print(f"{'player':<21}{'you said':>9}{'layer gives':>13}")
    for i, j, q, p in targets:
        print(f"{p:<21}{q:>9.0%}{P[i,j]:>13.0%}")

    m['mu_cal'] = mu; m['sig_cal'] = sig
    for j, pk in enumerate(MY):
        m[f'p_avail_{pk}'] = P[:, j]
    m.to_csv('results/board_LIVE_constrained.csv', index=False)

    print("\n\n=========== WHAT THAT IMPLIES AT EACH PICK ===========")
    for j, pk in enumerate(MY[:6]):
        c = m[(m[f'p_avail_{pk}'] > .04) & (m.tier <= 5) &
              (m.position.isin(['RB', 'WR']))].copy()
        c = c.sort_values(['tier', 'order', 'BOARD']).head(8)
        print(f"\n--- pick {pk} " + "-" * 40)
        for _, r in c.iterrows():
            print(f"    {r.position}{int(r.tier)}  {r['name'][:21]:<22}"
                  f"P(there) {r[f'p_avail_{pk}']:>5.0%}")


if __name__ == '__main__':
    main()

"""Draft plan on the OWNER'S ACTUAL POST-TRADE PICK SCHEDULE.

Earlier planning scripts hard-coded the default slot-5 snake (5,16,25,36,45,56).
The owner traded: gave 3.05/4.06/5.05/6.06/8.06/10.06, received 3.07/4.04/5.07/6.04/8.04/9.07.
Real first six picks are therefore 5, 16, 27, 34, 47, 54.

Board = v5 (mu* adopted). Selection is LINEUP-MARGINAL (max-weight assignment into
the 10-team starting lineup 2RB/2WR/1TE/2FLEX(RB-WR)/1QB/1DST), never raw VORP --
raw VORP builds a 2RB/10WR roster (REPORT.md section 48.3).

Room model: owner's stated 28-player order with sigma = 1.40*sqrt(pi/2) (E|err| = 1.40),
ADP with sigma = 9.0 beyond it.

Usage: python3 scripts/76_plan_real_picks.py
"""
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

TEAMS, N = 10, 6000
SIG_STATED = 1.40*np.sqrt(np.pi/2)
SIG_TAIL   = 9.0
STATED     = 39   # board observed complete through pick 39

OWN = [(1,5),(2,6),(7,5),(9,5),(11,5),(12,6),(13,5),(14,6),(15,5)]
ACQ = [(3,7),(4,4),(5,7),(6,4),(8,4),(9,7)]
MY  = sorted((r-1)*TEAMS+p for r,p in OWN+ACQ)[:6]

# starting lineup: which positions each slot accepts
SLOTS = [{'RB'},{'RB'},{'WR'},{'WR'},{'TE'},{'RB','WR'},{'RB','WR'}]

def key(n):
    return (str(n).lower().replace('.','').replace("'",'')
            .replace(' jr','').replace(' iii','').replace(' ii','').strip())

def lineup_value(vals, poss):
    """Max-weight assignment of held players into SLOTS (transversal matroid).

    NOTE: an earlier version enumerated permutations of the HELD players and
    assigned them to SLOTS[0..k-1]. That silently made the FLEX slots
    unreachable until the roster was 6 deep, so every RB after the second
    scored a marginal gain of exactly 0. Solve the assignment properly.
    """
    if not len(vals):
        return 0.0
    C = np.zeros((len(vals), len(SLOTS)))
    for i, (v, p) in enumerate(zip(vals, poss)):
        for s, ok in enumerate(SLOTS):
            C[i, s] = v if p in ok else 0.0
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].sum())

def main():
    b  = pd.read_csv('results/expected_order_2026_v2.csv')
    bd = pd.read_csv('results/board_2026_v5_mustar.csv')
    b['k'] = b.name.map(key); bd['k'] = bd.name.map(key)
    b = b.drop(columns=[c for c in ('final','position') if c in b.columns]).merge(
        bd[['k','final','position']], on='k', how='left')
    b['final'] = b.final.fillna(-9.0)
    b['position'] = b.position.fillna('WR')

    val  = b.final.to_numpy(float)
    pos  = b.position.to_numpy()
    name = b.name.to_numpy()
    mu   = b.exp_pick.to_numpy(float)
    sig  = np.where(mu <= STATED, SIG_STATED, SIG_TAIL)

    rng  = np.random.default_rng(7)
    got  = {k: [] for k in MY}
    surv = {k: np.zeros(len(b)) for k in MY}

    for _ in range(N):
        draw  = rng.normal(mu, sig)
        order = np.argsort(draw)
        taken = np.zeros(len(b), bool)
        ptr = 0
        held_v, held_p = [], []
        for pk in range(1, MY[-1]+1):
            if pk in MY:
                free = np.flatnonzero(~taken)
                surv[pk][free] += 1
                # Marginal gain is monotone in value for a fixed position, so the
                # best candidate at each position is that position's top free player.
                # Reduces the search from |free| lineup solves to <= 4.
                base = lineup_value(held_v, held_p)
                best_i, best_g = free[0], -np.inf
                for p in ('RB', 'WR', 'TE', 'QB'):
                    cand = free[pos[free] == p]
                    if not len(cand): continue
                    j = cand[int(np.argmax(val[cand]))]
                    g = lineup_value(held_v+[val[j]], held_p+[p]) - base
                    if g > best_g: best_g, best_i = g, j
                i = best_i
                got[pk].append(name[i])
                held_v.append(val[i]); held_p.append(pos[i])
            else:
                while ptr < len(order) and taken[order[ptr]]: ptr += 1
                if ptr >= len(order): break
                i = order[ptr]
            taken[i] = True

    print(f"picks: {MY}   (post-trade; default slot-5 would be [5,16,25,36,45,56])\n")
    for pk in MY:
        c = pd.Series(got[pk]).value_counts(normalize=True).head(5)
        print(f"--- pick {pk} ---")
        for nm, f in c.items():
            p = b.loc[b.name == nm, 'position'].iloc[0]
            print(f"    {f:5.1%}  {nm} ({p})")
        s = pd.Series(surv[pk]/N, index=name).sort_values(ascending=False)
        s = s[(s > .05) & (s < .95)]
        if len(s):
            print("    live coin-flips: " + ", ".join(f"{n} {v:.0%}" for n, v in s.head(6).items()))
        print()

if __name__ == '__main__':
    main()

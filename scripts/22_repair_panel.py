"""Repair an upstream nflverse data defect that silently deleted six era-1 WR seasons.

Found while running §H (not anticipated in EDA_PLAN4.md): in the nflverse weekly release,
`targets` is degenerate (league-wide sum ~0 while receptions and receiving yards are
intact) for seasons 2003-2008, at every position. targets/receptions by season:

    1999-2002   WR 1.84 1.83 1.79 1.77   RB 1.40 1.40 1.37 1.36
    2003-2008   WR 0.00 ... 0.00         RB 0.00 ... 0.01        <-- defect
    2009-2025   WR 1.74 ... 1.60         RB 1.37 ... 1.28

Consequence for §H as pre-registered: the qualification rule is >= 8 games AND >= 40
touches (carries + targets), so ZERO WRs qualified in 2003-2008. WR "era 1 (1999-2007)"
was in fact 1999-2002 only, and era 2 lost 2008. The era comparison -- the entire §H
hypothesis -- would have rested on four seasons of WR data at one end.

Repair (deterministic, computed without reference to the outcome, applied identically to
both positions): for the six defective seasons, targets_hat = receptions * rho_p, where
rho_p is the position's targets/receptions ratio pooled over the eight nearest clean
seasons (1999-2002, 2009-2012). PPR points are computed from receptions, not targets, so
the OUTCOME is untouched; only the qualification screen changes.

Raw data is never overwritten: writes data/derived/age_panel_long_repaired.csv with
targets/touches replaced and an `imputed_targets` flag column.
"""
import numpy as np
import pandas as pd

ROOT = "/Users/thomasmcnamee/NFL"
BAD = list(range(2003, 2009))
CLEAN = [1999, 2000, 2001, 2002, 2009, 2010, 2011, 2012]

p = pd.read_csv(f"{ROOT}/data/derived/age_panel_long.csv")
rho = (p[p.season.isin(CLEAN)].groupby("position")
       .apply(lambda g: g.targets.sum() / g.receptions.sum(), include_groups=False))
print("targets/receptions ratio from the eight nearest clean seasons:")
print(rho.round(4).to_string())

p["imputed_targets"] = p.season.isin(BAD)
mask = p.imputed_targets
p.loc[mask, "targets"] = np.round(
    p.loc[mask, "receptions"].values * p.loc[mask, "position"].map(rho).values).astype(int)
p["touches"] = p.carries + p.targets

q = (p.games >= 8) & (p.touches >= 40)
print("\nqualified player-seasons by season after repair:")
print(p.assign(q=q).groupby(["season", "position"]).q.sum().unstack(1).to_string())
p.to_csv(f"{ROOT}/data/derived/age_panel_long_repaired.csv", index=False)
print("\nwrote data/derived/age_panel_long_repaired.csv")

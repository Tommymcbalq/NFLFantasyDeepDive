#!/usr/bin/env python3
"""§Q figures — distribution spread inside an ADP tier, and what EB shrinkage does.

Descriptive only.  Tiers are chosen by ADP band, not by which players make the point;
the "most striking" pair search below is a mechanical ranking over every same-position
pair whose board values are within 0.15 PPG, with no name entering the selection rule.

Palette: dataviz reference categorical slots 1 (blue #2a78d6) and 2 (orange #eb6834),
which validate all-pairs; everything else is recessive ink.  One axis per panel.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
ROOT = Path("/Users/thomasmcnamee/NFL")
FIG = ROOT / "results/figures"
FIG.mkdir(exist_ok=True)
Y = "fantasy_points_ppr"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#b8b7b2", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                     "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})

d = pd.read_csv(ROOT / "results/player_distributions.csv")

# game-level points for the strip overlay (same inclusion rules as the table)
COLS = ["player_id", "position", "season", "season_type", "targets", "carries",
        "attempts", Y]
wk = pd.concat([pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{y}.csv",
                            usecols=lambda c: c in COLS, low_memory=False)
                for y in range(2023, 2026)], ignore_index=True)
wk = wk[wk.season_type == "REG"]
for c in ("targets", "carries", "attempts"):
    wk[c] = wk[c].fillna(0.0)
wk["touches"] = wk.carries + wk.targets
POSMASK = {"WR": wk.targets >= 2, "TE": wk.targets >= 2, "RB": wk.touches >= 2,
           "QB": wk.attempts >= 6}


def games(pid, pos, lo=2023, hi=2025):
    g = wk[(wk.player_id == pid) & POSMASK[pos] & (wk.season >= lo) & (wk.season <= hi)]
    return g[Y].values


def tier_panel(ax, sub, pos, rng):
    """Horizontal p10-p90 range with a p25-p75 core, median tick, and the raw games."""
    sub = sub.sort_values("adp").reset_index(drop=True)
    ys = np.arange(len(sub))[::-1]
    rs = np.random.default_rng(7)
    allpts = {r.gsis_id: games(r.gsis_id, pos) for _, r in sub.iterrows()}
    xhi = max(np.concatenate(list(allpts.values())).max() + 2, 30)
    xtext = xhi + 1
    for y, (_, r) in zip(ys, sub.iterrows()):
        pts = allpts[r.gsis_id]
        ax.scatter(pts, y + rs.uniform(-.17, .17, len(pts)), s=11, color=MUTED,
                   alpha=.75, linewidths=0, zorder=1)
        ax.plot([r.p10, r.p90], [y, y], color=BLUE, lw=2, alpha=.45,
                solid_capstyle="round", zorder=2)
        ax.plot([r.p25, r.p75], [y, y], color=BLUE, lw=7, alpha=.95,
                solid_capstyle="round", zorder=3)
        ax.plot([r["median"]], [y], marker="|", ms=13, mew=2.2, color=SURF, zorder=4)
        ax.text(xtext, y, f"{r.p25:5.1f}     {r.p90:5.1f}     {r.bust_eb:.2f}"
                          f"     {r.boom_eb:.2f}", va="center", fontsize=7.5, color=INK2,
                family="monospace", clip_on=False)
    ax.text(xtext, len(sub) - .35, " p25      p90     bust    boom", fontsize=7.5,
            color=INK2, family="monospace", va="center", clip_on=False)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r.player}\nADP {r.adp:.0f} · value {r.board_value:.2f} "
                        f"· n={int(r.n_games)}" for _, r in sub.iterrows()], fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("PPR points per game, 2023–2025 (each dot one game)")
    ax.grid(axis="x", color=MUTED, alpha=.35, lw=.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(-3, xhi)
    ax.set_ylim(-.7, len(sub) - .1)
    ax.set_xticks([t for t in ax.get_xticks() if 0 <= t <= xhi])
    ax.set_title(f"{pos}, ADP {rng[0]}–{rng[1]}: same price, different shape",
                 loc="left", fontsize=11, color=INK, pad=8)


TIERS = [("WR", (84, 110), "dist_tier_wr_adp84_110.png"),
         ("WR", (128, 162), "dist_tier_wr_adp128_162.png"),
         ("RB", (128, 200), "dist_tier_rb_adp128_200.png")]
for pos, rng, fn in TIERS:
    sub = d[(d.window == "last3") & (d.pos == pos) & (d.n_games >= 8)
            & d.adp.between(*rng)]
    if len(sub) > 11:
        sub = sub.nsmallest(11, "adp")
    fig, ax = plt.subplots(figsize=(10.4, .52 * len(sub) + 1.8))
    tier_panel(ax, sub, pos, rng)
    fig.text(.012, .015, "bars: p25–p75 (thick) inside p10–p90 (thin); white tick = median. "
             "Rates are EB-shrunk. Sorted by ADP.", fontsize=7, color=INK2)
    fig.subplots_adjust(left=.175, right=.775, top=.88,
                        bottom=.30 / (.52 * len(sub) + 1.8) + .07)
    fig.savefig(FIG / fn, dpi=170)
    plt.close(fig)
    print("wrote", fn, f"({len(sub)} players)")

# ---------------------------------------------------------------- EB shrinkage figure
fig, axes = plt.subplots(1, 4, figsize=(13, 3.7), sharey=True)
for ax, pos in zip(axes, ["WR", "RB", "TE", "QB"]):
    s = d[(d.window == "last3") & (d.pos == pos) & d.boom_eb.notna()]
    for _, r in s.iterrows():
        ax.plot([r.n_games, r.n_games], [r.boom_raw, r.boom_eb], color=MUTED, lw=.8,
                zorder=1)
    ax.scatter(s.n_games, s.boom_raw, s=16, color=ORANGE, linewidths=0, zorder=2,
               label="raw k/m")
    ax.scatter(s.n_games, s.boom_eb, s=16, color=BLUE, linewidths=0, zorder=3,
               label="EB posterior mean")
    pn = s.eb_boom_prior_n.iloc[0]
    ax.axhline(s.boom_eb.iloc[0] * 0 + (s.k_boom.sum() / s.n_games.sum()), color=INK2,
               lw=.8, ls=(0, (4, 3)))
    ax.set_title(f"{pos}   boom > {s.boom_thresh.iloc[0]:g} PPR   prior n₀={pn:.1f}",
                 loc="left", fontsize=9.5)
    ax.set_xlabel("games in 2023–2025 window")
    ax.grid(color=MUTED, alpha=.35, lw=.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("P(boom) per game")
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
fig.suptitle("Empirical-Bayes stabilisation of the boom rate: short samples are pulled to "
             "the positional prior", x=.008, ha="left", fontsize=11)
fig.text(.008, .015, "Dashed line = pooled board rate. Vertical segment = the distance "
         "shrinkage moved the player. Beta(α,β) refit per position on this window.",
         fontsize=7, color=INK2)
fig.tight_layout(rect=(0, .045, 1, .93))
fig.savefig(FIG / "dist_eb_shrinkage.png", dpi=170)
plt.close(fig)
print("wrote dist_eb_shrinkage.png")

# ---------------------------------------------------------------- same-price pair search
print("\nmechanical same-price / different-shape search "
      "(last3, n>=8, |Δboard_value| <= 0.15):")
r = d[(d.window == "last3") & (d.n_games >= 8) & d.p90.notna()].copy()
rows = []
for pos, g in r.groupby("pos"):
    a = g.reset_index(drop=True)
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            x, y = a.iloc[i], a.iloc[j]
            if abs(x.board_value - y.board_value) > 0.15:
                continue
            rows.append(dict(pos=pos, a=x.player, b=y.player,
                             adp_a=x.adp, adp_b=y.adp,
                             val_a=x.board_value, val_b=y.board_value,
                             d_p25=x.p25 - y.p25, d_p90=x.p90 - y.p90,
                             d_bust=x.bust_eb - y.bust_eb, d_boom=x.boom_eb - y.boom_eb,
                             shape_dist=abs(x.p25 - y.p25) + abs(x.p90 - y.p90),
                             p25_a=x.p25, p25_b=y.p25, p90_a=x.p90, p90_b=y.p90,
                             bust_a=x.bust_eb, bust_b=y.bust_eb,
                             boom_a=x.boom_eb, boom_b=y.boom_eb,
                             n_a=x.n_games, n_b=y.n_games))
pairs = pd.DataFrame(rows).sort_values("shape_dist", ascending=False)
pairs.to_csv(ROOT / "results/player_distribution_pairs.csv", index=False)
cols = ["pos", "a", "b", "adp_a", "adp_b", "val_a", "val_b", "p25_a", "p25_b",
        "p90_a", "p90_b", "bust_a", "bust_b", "shape_dist"]
print(pairs.head(12)[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

print("\nsame ranking restricted to genuinely competing picks (|ΔADP| <= 12), so the "
      "pair is a real either/or at the same slot:")
near = pairs[(pairs.adp_a - pairs.adp_b).abs() <= 12]
print(near.head(15)[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
near.to_csv(ROOT / "results/player_distribution_pairs_nearadp.csv", index=False)

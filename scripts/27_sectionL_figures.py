"""§L figures — reads results/conversion_rates.csv, sectionL_panel.csv,
sectionL_costtrend.csv; writes results/figures/sectionL_*.png.

Palette: project's validated light-mode set (12_report_figures.py). Two series
(RB, WR) -> BLUE / RED, validated: CVD ΔE 21.6 protan, contrast PASS.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/Users/thomasmcnamee/NFL")
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
FIG.mkdir(exist_ok=True)
BLUE, RED = "#2a78d6", "#e34948"
INK, SEC, MUTED, GRID, SURF, BASE = ("#0b0b0b", "#52514e", "#898781",
                                     "#e1e0d9", "#fcfcfb", "#c3c2b7")
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "text.color": INK,
    "axes.labelcolor": SEC, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
BINS = ["R1-2", "R3-4", "R5-6", "R7-8", "R9+"]
COL = {"RB": BLUE, "WR": RED}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


rates = pd.read_csv(RES / "conversion_rates.csv")
panel = pd.read_csv(RES / "sectionL_panel.csv")
cost = pd.read_csv(RES / "sectionL_costtrend.csv")

# --- F1: conversion by cost bin, both hit definitions, both frames -----------
for frame, ftxt in [("12team", "12-team frame (ADP source)"),
                    ("10team", "10-team frame (owner's league)")]:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
    for ax, hd, ttl in zip(axes, ["top12_T", "value_T"],
                           ["hit = top-12 positional finish (season total)",
                            "hit = ≥ median total of the bin (all positions)"]):
        d = rates[(rates.frame == frame) & (rates.window == "2015-2024")
                  & (rates.hitdef == hd)]
        for j, pos in enumerate(["RB", "WR"]):
            s = d[d.pos == pos].set_index("bin").reindex(BINS)
            x = np.arange(len(BINS)) + (j - .5) * .30
            ax.errorbar(x, s.rate, yerr=[s.rate - s.wilson_lo, s.wilson_hi - s.rate],
                        fmt="o", ms=7, lw=2, capsize=3, color=COL[pos], label=pos,
                        markeredgecolor=SURF, markeredgewidth=1.5, zorder=3)
            for xi, r, hi, n in zip(x, s.rate, s.wilson_hi, s.n):
                ax.annotate(f"{r:.0%}", (xi, hi), textcoords="offset points",
                            xytext=(0, 5), ha="center", fontsize=8, color=SEC)
                ax.annotate(f"{int(n)}", (xi, -0.045), ha="center", va="top",
                            fontsize=7.5, color=MUTED)
        ax.annotate("n:", (-0.72, -0.045), ha="center", va="top", fontsize=7.5,
                    color=MUTED)
        ax.set_xlim(-0.85, len(BINS) - 0.15)
        ax.set_xticks(range(len(BINS)))
        ax.set_xticklabels(BINS)
        ax.set_title(ttl, fontsize=9.5)
        ax.set_xlabel("draft cost bin")
    axes[0].set_ylabel("conversion rate (Wilson 95%)")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].set_ylim(-0.12, 1.0)
    fig.suptitle(f"§L1 conversion by position and draft cost, 2015–2024 — {ftxt}",
                 fontsize=11, fontweight="bold", color=INK, y=1.02)
    save(fig, f"sectionL_conversion_{frame}.png")

# --- F2: RB R1-2 by season, against the RB draft-cost trend (no dual axis) ---
fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.2), sharex=True,
                         gridspec_kw={"height_ratios": [1.25, 1]})
ax = axes[0]
for pos in ["RB", "WR"]:
    r = []
    for y in sorted(panel.year.unique()):
        c = panel[(panel.year == y) & (panel.pos_adp == pos) & (panel.bin12 == "R1-2")]
        r.append((y, c.hit_pos_T.mean(), len(c)))
    r = pd.DataFrame(r, columns=["y", "p", "n"])
    ax.plot(r.y, r.p, "-o", color=COL[pos], lw=2, ms=7, label=f"{pos} R1–2",
            markeredgecolor=SURF, markeredgewidth=1.5, zorder=3)
ax.set_ylabel("P(top-12 finish)")
ax.set_ylim(0, 1)
ax.legend(frameon=False, ncol=2)
ax.set_title("§L3 elite conversion by season — single-season rates are NOT signal "
             "(n ≈ 10–15/cell, SE ≈ 13 pp)", fontsize=9.5)
ax = axes[1]
ax.plot(cost.year, cost.rb_adp_top10, "-o", color=BLUE, lw=2, ms=7,
        markeredgecolor=SURF, markeredgewidth=1.5, label="mean ADP of RB1–10")
ax.plot(cost.year, cost.wr_adp_top10, "-o", color=RED, lw=2, ms=7,
        markeredgecolor=SURF, markeredgewidth=1.5, label="mean ADP of WR1–10")
ax.invert_yaxis()
ax.set_ylabel("mean ADP (lower = costlier)")
ax.set_xlabel("season")
ax.legend(frameon=False, ncol=2)
ax.set_title("§L6 confound: the price of elite RBs over the same window", fontsize=9.5)
save(fig, "sectionL_trend.png")

# --- F3: the total-vs-PPG gap = availability --------------------------------
fig, ax = plt.subplots(figsize=(7.4, 4.2))
d = rates[(rates.frame == "12team") & (rates.window == "2015-2024")
          & (rates.hitdef.isin(["value_T", "value_P"]))]
w = 0.34
for j, pos in enumerate(["RB", "WR"]):
    for k, (hd, alpha, lab) in enumerate([("value_T", 1.0, "season total"),
                                          ("value_P", 0.45, "PPG | ≥4 games")]):
        s = d[(d.pos == pos) & (d.hitdef == hd)].set_index("bin").reindex(BINS)
        x = np.arange(len(BINS)) + (j - .5) * w + (k - .5) * (w / 2.2)
        ax.bar(x, s.rate, width=w / 2.4, color=COL[pos], alpha=alpha, zorder=3,
               edgecolor=SURF, linewidth=2,
               label=f"{pos} — {lab}")
ax.set_xticks(range(len(BINS)))
ax.set_xticklabels(BINS)
ax.set_ylabel("value-return conversion")
ax.set_xlabel("draft cost bin (12-team)")
ax.legend(frameon=False, fontsize=8.5, ncol=2)
ax.set_title("§L0 the two outcome definitions: the RB gap that closes when missed "
             "games are removed", fontsize=9.5)
save(fig, "sectionL_outcome_definitions.png")


# --- F4: cumulative tiers 12/24/36 by cost bin (extension) ------------------
for win in ["2015-2024", "2022-2024"]:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.3), sharey=True)
    for ax, tier, ttl in zip(axes, ["top12", "top24", "top36"],
                             ["finish ≤ 12", "finish ≤ 24", "finish ≤ 36"]):
        d = rates[(rates.frame == "12team") & (rates.window == win)
                  & (rates.tier == tier) & (rates.outcome_def == "total")]
        for j, pos in enumerate(["RB", "WR"]):
            sr = d[d.pos == pos].set_index("bin").reindex(BINS)
            x = np.arange(len(BINS)) + (j - .5) * .30
            ax.errorbar(x, sr.rate, yerr=[sr.rate - sr.wilson_lo, sr.wilson_hi - sr.rate],
                        fmt="o", ms=7, lw=2, capsize=3, color=COL[pos], label=pos,
                        markeredgecolor=SURF, markeredgewidth=1.5, zorder=3)
            for xi, r_, hi, n in zip(x, sr.rate, sr.wilson_hi, sr.n):
                ax.annotate(f"{r_:.0%}", (xi, hi), textcoords="offset points",
                            xytext=(0, 5), ha="center", fontsize=8, color=SEC)
                ax.annotate(f"{int(n)}", (xi, -0.055), ha="center", va="top",
                            fontsize=7.5, color=MUTED)
        ax.annotate("n:", (-0.72, -0.055), ha="center", va="top", fontsize=7.5,
                    color=MUTED)
        ax.set_xlim(-0.85, len(BINS) - 0.15)
        ax.set_xticks(range(len(BINS)))
        ax.set_xticklabels(BINS)
        ax.set_title(ttl, fontsize=9.5)
        ax.set_xlabel("draft cost bin")
    axes[0].set_ylabel("conversion (season totals, Wilson 95%)")
    axes[0].set_ylim(-0.14, 1.05)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle(f"§L-EXT cumulative finish tiers by position and draft cost, {win} "
                 "— 12-team frame", fontsize=11, fontweight="bold", color=INK, y=1.02)
    save(fig, f"sectionL_tiers_{win.replace('-', '_')}.png")

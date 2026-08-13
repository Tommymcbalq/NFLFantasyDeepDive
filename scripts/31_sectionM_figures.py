"""§M figures. Reads results/vorp_curves.csv, sectionM_replacement_bracket.csv,
sectionM_premium_by_baseline.csv, strategy_distribution.csv, strategy_by_slot.csv;
writes results/figures/sectionM_*.png.

Palette: the project's validated light-mode categorical set (12_report_figures.py),
re-validated for this use with scripts/validate_palette.js --mode light:
  lightness band PASS, chroma floor PASS, CVD separation PASS (worst adjacent
  #eda100<->#1baf7a dE 9.1 protan), normal-vision floor PASS (dE 22.9),
  contrast WARN on the two lightest -> every series is direct-labelled or legended,
  which is the required relief.
Four positions and six strategies are categorical (identity); hues are assigned in a
fixed order and never cycled or reused across figures.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/Users/thomasmcnamee/NFL")
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
FIG.mkdir(exist_ok=True)
BLUE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"
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
PCOL = {"RB": BLUE, "WR": RED, "TE": AQUA, "QB": VIOLET}
SCOL = {"S0": INK, "S1": BLUE, "S2": RED, "S3": YELLOW, "S4": AQUA, "S5": VIOLET}
SLAB = {"S0": "S0 draft the board", "S1": "S1 model board", "S2": "S2 RB-first",
        "S3": "S3 zero-RB", "S4": "S4 elite-TE", "S5": "S5 VORP-greedy"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


cur = pd.read_csv(RES / "vorp_curves.csv")

# --- F1: VORP by positional ADP rank, both frames ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
for ax, F in zip(axes, (10, 12)):
    d = cur[(cur.frame == F) & (cur.unit == "positional_adp_rank") & (cur.measure == "total")]
    for p in ["WR", "RB", "TE", "QB"]:
        s = d[(d.pos == p) & (d.slot <= 14)].sort_values("slot")
        ax.plot(s.slot, s["mean"], "-o", color=PCOL[p], lw=2, ms=6,
                markeredgecolor=SURF, markeredgewidth=1.4, label=p, zorder=3)
        ax.annotate(p, (s.slot.iloc[0], s["mean"].iloc[0]), xytext=(-6, 6),
                    textcoords="offset points", color=SEC, fontsize=9, ha="right")
    ax.axhline(0, color=BASE, lw=1.2, zorder=1)
    ax.set_title(f"{F}-team demand" + ("  (owner's league)" if F == 10 else "  (FFC ADP frame)"))
    ax.set_xlabel("player's rank at his position by preseason ADP")
axes[0].set_ylabel("mean realized VORP, season PPR")
axes[0].legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle("VORP by positional draft rank, 2015–2024 realized outcomes",
             fontweight="bold", y=1.02)
save(fig, "sectionM_vorp_by_posrank.png")

# --- F2: RB vs WR by draft round, and the flex-allocation sensitivity --------
fl = pd.read_csv(RES / "sectionM_diag_flex.csv")
crs = pd.read_csv(RES / "sectionM_rbwr_cross.csv")
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
ax = axes[0]
d = crs[(crs.frame == 10) & (crs.measure == "total")]
ax.errorbar(d.rnd, d["diff"], yerr=1.96 * d.se, fmt="o", color=BLUE, lw=2, ms=7,
            capsize=3, markeredgecolor=SURF, markeredgewidth=1.4, zorder=3)
ax.axhline(0, color=RED, lw=1.6, zorder=2)
ax.set_xlabel("10-team draft round"); ax.set_ylabel("RB − WR mean VORP (season PPR)")
ax.set_title("RB minus WR VORP never crosses zero\n(realized flex allocation)")
ax.annotate("WR ahead at every round", (7, d["diff"].min() * 0.55), color=SEC, fontsize=9)
ax = axes[1]
labs = ["realized\n(3.5 RB / 16.5 WR)", "50/50 by fiat\n(10 / 10)", "no flex\n(0 / 0)"]
g = fl[fl.frame == 10].groupby("alloc")[["R_RB", "R_WR"]].mean().reindex(
    ["realized", "50/50", "no_flex"])
x = np.arange(3)
ax.bar(x - .17, g.R_RB, .32, color=BLUE, label="RB replacement")
ax.bar(x + .17, g.R_WR, .32, color=RED, label="WR replacement")
for xi, (a, b) in enumerate(zip(g.R_RB, g.R_WR)):
    ax.annotate(f"{a:.0f}", (xi - .17, a), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=9, color=SEC)
    ax.annotate(f"{b:.0f}", (xi + .17, b), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=9, color=SEC)
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8.5)
ax.set_ylabel("replacement level, season PPR")
ax.set_title("The 2-FLEX rule equalises RB and WR\nreplacement level (gap 0.3 pts)")
ax.legend(frameon=False, fontsize=9)
save(fig, "sectionM_rbwr_flex.png")

# --- F3: the replacement bracket --------------------------------------------
B = pd.read_csv(RES / "sectionM_replacement_bracket.csv")
fig, ax = plt.subplots(figsize=(7.6, 4.4))
g = B[B.frame == 10].groupby("pos")[["R_exp", "R_real", "R_week"]].mean().reindex(
    ["QB", "RB", "WR", "TE"])
y = np.arange(4)
for i, p in enumerate(g.index):
    ax.plot([g.R_exp[p], g.R_week[p]], [i, i], color=PCOL[p], lw=8, alpha=.30,
            solid_capstyle="butt", zorder=2)
    for col, mk, lab in ((g.R_exp[p], "o", "no management"),
                         (g.R_real[p], "s", "season foresight"),
                         (g.R_week[p], "D", "weekly foresight")):
        ax.plot(col, i, mk, color=PCOL[p], ms=9, markeredgecolor=SURF,
                markeredgewidth=1.6, zorder=3)
    ax.annotate(f"{g.R_week[p]-g.R_exp[p]:.0f} pt bracket", (g.R_week[p], i),
                xytext=(8, 0), textcoords="offset points", va="center",
                fontsize=9, color=SEC)
ax.set_yticks(y); ax.set_yticklabels(g.index)
ax.set_xlabel("replacement level, season PPR (10-team demand)")
ax.set_title("Replacement level is a bracket, not a number\n"
             "circle = draft-only · square = season foresight · diamond = weekly foresight")
ax.set_xlim(110, 380)
save(fig, "sectionM_replacement_bracket.png")

# --- F4: TE / QB premium under each baseline --------------------------------
P = pd.read_csv(RES / "sectionM_premium_by_baseline.csv")
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
order = ["R_exp", "R_real", "R_week"]
xl = ["draft-only\n(no management)", "season\nforesight", "weekly\nforesight"]
for ax, F in zip(axes, (10, 12)):
    for j, p in enumerate(["TE", "QB"]):
        s = P[(P.frame == F) & (P.pos == p)].set_index("baseline").reindex(order)
        x = np.arange(3) + (j - .5) * .22
        ax.errorbar(x, s.mean_premium, yerr=1.96 * s.se, fmt="o", color=PCOL[p],
                    lw=2, ms=8, capsize=3, markeredgecolor=SURF, markeredgewidth=1.5,
                    label=f"{p}1–5", zorder=3)
        for xi, v in zip(x, s.mean_premium):
            ax.annotate(f"{v:+.0f}", (xi, v), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8.5, color=SEC)
    ax.axhline(0, color=BASE, lw=1.4)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(xl, fontsize=9)
    ax.set_title(f"{F}-team demand")
axes[0].set_ylabel("premium over RB/WR at the same ADP\n(season PPR, ±6 picks)")
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("The elite-TE and elite-QB premium is a statement about replacement level",
             fontweight="bold", y=1.02)
save(fig, "sectionM_premium_baselines.png")

# --- F5: strategy backtest --------------------------------------------------
R = pd.read_csv(RES / "strategy_distribution.csv")
bt = pd.read_csv(RES / "strategy_backtest.csv")
fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6))
ax = axes[0]
d = bt[bt.metric == "pts14"].set_index("strat").reindex(["S1", "S2", "S3", "S4", "S5"])
y = np.arange(5)[::-1]
ax.errorbar(d.mean_diff, y, xerr=[d.mean_diff - d.ci_lo, d.ci_hi - d.mean_diff],
            fmt="o", ms=9, lw=2, capsize=3, color=BLUE,
            markeredgecolor=SURF, markeredgewidth=1.5, zorder=3)
ax.axvline(0, color=RED, lw=1.6, zorder=2)
for yi, (s, r) in zip(y, d.iterrows()):
    ax.annotate(f"p = {r.p:.2f}", (r.ci_hi, yi), xytext=(7, 0),
                textcoords="offset points", va="center", fontsize=9, color=SEC)
ax.set_yticks(y); ax.set_yticklabels([SLAB[s] for s in d.index], fontsize=9)
ax.set_xlabel("mean season points vs S0, weeks 1–14")
ax.set_title("0 of 5 survive BH q = 0.10\n(95% CI, season-clustered t(9))")
ax = axes[1]
sl = R.pivot_table(index="slot", columns="strat", values="pts14")
sl = sl.sub(sl["S0"], axis=0)
for s in ["S1", "S2", "S3", "S4", "S5"]:
    ax.plot(sl.index, sl[s], "-o", color=SCOL[s], lw=2, ms=5,
            markeredgecolor=SURF, markeredgewidth=1.2, label=SLAB[s], zorder=3)
ax.axhline(0, color=INK, lw=1.6, zorder=2)
ax.set_xlabel("our draft slot (1–10)"); ax.set_ylabel("mean points vs S0, weeks 1–14")
ax.set_title("Across-slot spread\n(a strategy must work from every seat)")
ax.legend(frameon=False, fontsize=8, ncol=1, loc="lower left")
ax = axes[2]
for s in ["S0", "S2", "S4", "S5"]:
    v = R[R.strat == s].pts14
    ax.hist(v, bins=60, histtype="step", lw=2, color=SCOL[s], label=SLAB[s], density=True)
ax.set_xlabel("season points, weeks 1–14")
ax.set_title("Full outcome distribution\n(the spread dwarfs every mean difference)")
ax.legend(frameon=False, fontsize=8)
ax.set_yticks([])
save(fig, "sectionM_strategy_backtest.png")

# --- F6: mean points vs win probability -------------------------------------
g = R.groupby("strat").agg(mean=("pts14", "mean"), sd=("pts14", "std"),
                           pf=("playoff", "mean"), ch=("champ", "mean"))
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
for ax, (col, lab) in zip(axes, [("pf", "P(top-4 of 10)"), ("ch", "P(finish 1st)")]):
    for s in g.index:
        ax.plot(g["mean"][s], g[col][s], "o", ms=11, color=SCOL[s],
                markeredgecolor=SURF, markeredgewidth=1.8, zorder=3)
        ax.annotate(s, (g["mean"][s], g[col][s]), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=9, color=SEC)
    ax.set_xlabel("mean season points, weeks 1–14"); ax.set_ylabel(lab)
axes[0].set_title("Points and playoff odds agree")
axes[1].set_title("Points and win odds do not:\nS1 is 3rd on points, 5th on P(win)")
save(fig, "sectionM_points_vs_winprob.png")
print("done.")

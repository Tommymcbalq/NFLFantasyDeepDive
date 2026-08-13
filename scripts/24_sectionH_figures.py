"""§H figures. Reads the results/*.csv written by scripts 20-23; no fitting here."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/thomasmcnamee/NFL"
FIG = f"{ROOT}/results/figures"
ERAS = ["1999-2007", "2008-2016", "2017-2025"]
C = {"1999-2007": "#2a78d6", "2008-2016": "#eb6834", "2017-2025": "#1baf7a"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b2"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False})


def grid(ax):
    ax.grid(True, color=MUTED, lw=.5, alpha=.45)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- 1. era curves
cur = pd.read_csv(f"{ROOT}/results/age_curve_era.csv")
fea = pd.read_csv(f"{ROOT}/results/age_curve_features.csv")
for pos in ["WR", "RB"]:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    d = cur[(cur.position == pos) & (cur.outcome == "relative")]
    f = fea[(fea.position == pos) & (fea.outcome == "relative")]
    for e in ERAS:
        s = d[d.era == e].sort_values("age")
        row = f[f.era == e].iloc[0]
        lo, hi = row.peak - 6, 34.5
        s = s[(s.age >= s.age.min()) & (s.age <= 34.5)]
        ax.fill_between(s.age, s.lo, s.hi, color=C[e], alpha=.10, lw=0)
        ax.plot(s.age, s.fit, color=C[e], lw=2, label=e, zorder=3)
        pk = s[np.isclose(s.age, row.peak)]
        if len(pk):
            ax.plot(row.peak, pk.fit.iloc[0], "o", ms=7, color=C[e],
                    mec="#fcfcfb", mew=1.5, zorder=4)
        cl = s[np.isclose(s.age, row.cliff)]
        if len(cl):
            ax.plot(row.cliff, cl.fit.iloc[0], "v", ms=8, color=C[e],
                    mec="#fcfcfb", mew=1.2, zorder=4)
        last = s.iloc[-1]
        ax.annotate(e, (last.age, last.fit), xytext=(5, 0), textcoords="offset points",
                    color=INK2, fontsize=8.5, va="center")
    ax.set_xlabel("age (Sept 1 of the season)")
    ax.set_ylabel("relative PPG (1.0 = qualified positional mean that season)")
    ax.set_title(f"{pos} age profile by era — player-FE natural cubic spline\n"
                 "circle = peak, triangle = cliff (10% below peak); bands = 95% "
                 "cluster-bootstrap on player", fontsize=10, loc="left")
    ax.set_xlim(21, 36.5)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5, labelcolor=INK2)
    grid(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG}/sectionH_age_curve_{pos}.png", dpi=170)
    plt.close(fig)

# ---------------------------------------------------------------- 2. exit hazard
hz = pd.read_csv(f"{ROOT}/results/exit_hazard.csv")
fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharey=True)
for ax, pos in zip(axes, ["WR", "RB"]):
    for e in ERAS:
        s = hz[(hz.position == pos) & (hz.era == e) & hz.in_support].sort_values("age")
        s = s[s.age <= 34]
        ax.plot(s.age, s.hazard, color=C[e], lw=2, label=e)
        ax.annotate(e, (s.age.iloc[-1], s.hazard.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", color=INK2, fontsize=8, va="center")
    ax.set_title(pos, fontsize=10, loc="left")
    ax.set_xlabel("age")
    grid(ax)
    ax.set_xlim(21, 36.5)
axes[0].set_ylabel("P(this is the last qualified season)")
axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
fig.suptitle("Career-exit hazard by era — discrete-time logit, age spline × era, "
             "cluster-robust by player", fontsize=10, x=.01, ha="left")
fig.tight_layout()
fig.savefig(f"{FIG}/sectionH_exit_hazard.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 3. the data defect
raw = pd.read_csv(f"{ROOT}/data/derived/age_panel_long.csv",
                  usecols=["season", "position", "games", "touches"])
rep = pd.read_csv(f"{ROOT}/data/derived/age_panel_long_repaired.csv",
                  usecols=["season", "position", "games", "touches"])
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
for i, (df, ttl) in enumerate([(raw, "as pulled (nflverse targets defect)"),
                               (rep, "after repair")]):
    q = df[(df.games >= 8) & (df.touches >= 40)]
    for j, pos in enumerate(["WR", "RB"]):
        s = q[q.position == pos].groupby("season").size().reindex(range(1999, 2026),
                                                                  fill_value=0)
        axes[i].plot(s.index, s.values, lw=2, color=["#2a78d6", "#eb6834"][j], label=pos)
        axes[i].annotate(pos, (s.index[-1], s.values[-1]), xytext=(4, 0),
                         textcoords="offset points", color=INK2, fontsize=8, va="center")
    axes[i].set_title(ttl, fontsize=9.5, loc="left")
    axes[i].set_xlabel("season")
    axes[i].set_ylim(0, 120)
    grid(axes[i])
axes[0].set_ylabel("qualified player-seasons (≥8 g, ≥40 touches)")
axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower left")
fig.suptitle("Data defect found during §H: nflverse `targets` is ~0 league-wide for "
             "2003–2008, deleting six era-1 WR seasons", fontsize=10, x=.01, ha="left")
fig.tight_layout()
fig.savefig(f"{FIG}/sectionH_data_defect.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 4. H4 decomposition
h4 = pd.read_csv(f"{ROOT}/results/h4_workload_decomp.csv")
h4 = h4[h4.marginal_200_350.notna()]
fig, ax = plt.subplots(figsize=(7.6, 3.4))
y = np.arange(len(h4))[::-1]
ax.errorbar(h4.marginal_200_350, y,
            xerr=[h4.marginal_200_350 - h4.marginal_lo, h4.marginal_hi - h4.marginal_200_350],
            fmt="o", ms=8, lw=2, color="#2a78d6", mec="#fcfcfb", mew=1.2, capsize=0)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(y)
ax.set_yticklabels(h4.spec, fontsize=8.5)
for v, yy in zip(h4.marginal_200_350, y):
    ax.annotate(f"{v:+.3f}", (v, yy), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=8, color=INK2)
ax.set_xlabel("implied Δ relative PPG from a 200 → 350 prior-season touch load")
ax.set_title("§H4 RB workload carryover: the pre-registered effect is mean reversion\n"
             "(bars = 95% cluster bootstrap on player)", fontsize=10, loc="left")
grid(ax)
fig.tight_layout()
fig.savefig(f"{FIG}/sectionH_h4_workload.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 5. H5 market residual
wr = pd.read_csv(f"{ROOT}/results/edge_panel.csv")
wr = wr[wr.in_fit][["year", "age", "resid_iso"]].assign(position="WR")
rb = pd.read_csv(f"{ROOT}/results/rb_market_prior.csv")
rb = rb[rb.in_fit].copy()
rb["age"] = (pd.to_datetime(rb.year.astype(str) + "-09-01")
             - pd.to_datetime(rb.birth_date)).dt.days / 365.25
rb = rb[["year", "age", "resid_iso"]].assign(position="RB")
fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharey=True)
for ax, (df, pos, col) in zip(axes, [(wr, "WR", "#2a78d6"), (rb, "RB", "#eb6834")]):
    d = df.dropna()
    ax.scatter(d.age, d.resid_iso, s=16, color=col, alpha=.45, lw=0)
    a0 = d.age.mean()
    X = np.column_stack([np.ones(len(d)), d.age - a0, (d.age - a0)**2])
    b = np.linalg.lstsq(X, d.resid_iso, rcond=None)[0]
    g = np.linspace(d.age.min(), d.age.max(), 100)
    ax.plot(g, b[0] + b[1]*(g-a0) + b[2]*(g-a0)**2, color=INK, lw=2)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_title(f"{pos}  (n={len(d)})", fontsize=10, loc="left")
    ax.set_xlabel("age at the ADP year")
    grid(ax)
axes[0].set_ylabel("market residual: realized PPG − m̂_iso(ADP)")
fig.suptitle("§H5 — market residual vs age, 2015–2024 top-30 boards "
             "(quadratic fit; neither position survives the temporal holdout)",
             fontsize=10, x=.01, ha="left")
fig.tight_layout()
fig.savefig(f"{FIG}/sectionH_h5_residual_age.png", dpi=170)
plt.close(fig)
print("wrote 6 figures to results/figures/")

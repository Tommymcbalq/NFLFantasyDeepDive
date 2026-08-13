"""Report figures for REPORT.md — reads results/*.csv, writes results/figures/*.png.

Palette: validated light-mode reference set (see dataviz method) — categorical
blue/aqua/yellow/violet/red assigned in fixed order; ink/grid/surface tokens below.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)

BLUE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"
INK, SEC, MUTED, GRID, SURF, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "text.color": INK, "axes.labelcolor": SEC, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- fig 1: consistency scatter
ct = pd.read_csv(RES / "consistency_table.csv")
fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.scatter(ct["mu_hat"], ct["sigma_W"], s=48, color=BLUE, alpha=0.85, zorder=3,
           edgecolors=SURF, linewidths=1.5)
# CV isolines
mus = np.linspace(9, 22.5, 50)
for cvv in (0.45, 0.60, 0.75):
    ax.plot(mus, cvv * mus, color=GRID, lw=1, zorder=1)
    ax.annotate(f"CV {cvv}", (22.5, cvv * 22.5), color=MUTED, fontsize=8, va="center")
label = ["Puka Nacua", "Ja'Marr Chase", "Amon-Ra St. Brown", "Rashee Rice",
         "Mike Evans", "Alec Pierce", "Emeka Egbuka", "Davante Adams", "Nico Collins"]
for _, r in ct[ct["player"].isin(label)].iterrows():
    ax.annotate(r["player"], (r["mu_hat"], r["sigma_W"]), xytext=(5, 5),
                textcoords="offset points", fontsize=8, color=SEC)
ax.set_xlabel("recency-weighted PPG level  μ̂  (h = 1)")
ax.set_ylabel("within-season SD  σ̂_W  (PPG)")
ax.set_title("§1  Level vs per-game volatility — top-30 WRs, 2026 board")
save(fig, "fig01_consistency_scatter.png")

# ------------------------------------------------- fig 2: naive vs corrected between-season var
vets = ct[ct["n_seasons"] >= 4].sort_values("naive_v", ascending=True)
fig, ax = plt.subplots(figsize=(7.2, 6.0))
y = np.arange(len(vets))
tau2 = vets["tau2_B_untrunc"].fillna(vets["naive_v"] - vets["small_sample_corr"])
for yi, (n, t) in enumerate(zip(vets["naive_v"], tau2)):
    ax.plot([t, n], [yi, yi], color=GRID, lw=2, zorder=1)
ax.scatter(vets["naive_v"], y, s=42, color=MUTED, zorder=3, label="naive  v (variance of season means)")
ax.scatter(tau2, y, s=42, color=BLUE, zorder=4, label="corrected  τ̂²_B  (eq. 3, untruncated)")
ax.axvline(0, color=BASE, lw=1)
ax.set_yticks(y, vets["player"], fontsize=8.5)
ax.set_xlabel("between-season variance of PPG (PPG²)")
ax.set_title("§1  The eq.-3 correction — averaging noise removed from year-to-year variance")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
save(fig, "fig02_between_season_correction.png")

# ---------------------------------------------------------------- fig 3: variance components
vc = pd.read_csv(RES / "variance_components.csv")
reml = vc[vc["estimator"].str.contains("REML", case=False, na=False)].copy()
labels = {"headline_2021_2025_excl": "2021–25 (headline)",
          "sens_a_2014_2025_seasonFE": "2014–25 + season FE",
          "sens_b_log1p_2021_2025": "log(1+Y), 2021–25",
          "sens_c_no_exclusions": "2021–25, no exclusions"}
reml["lab"] = reml["spec"].map(labels).fillna(reml["spec"])
fig, ax = plt.subplots(figsize=(7.6, 3.4))
y = np.arange(len(reml))
left = np.zeros(len(reml))
parts = [("icc_P", "player  σ²_P", BLUE), ("icc_S", "player×season  σ²_S", AQUA),
         ("icc_T", "team×season  σ²_T", YELLOW), ("icc_G", "game noise  σ²_G", GRID)]
for col, lab, c in parts:
    v = reml[col].values
    ax.barh(y, v, left=left, height=0.55, color=c, label=lab,
            edgecolor=SURF, linewidth=2)
    left += v
for yi, r in enumerate(reml.itertuples()):
    ax.annotate(f"ρ_max = {r.rho_max:.2f}", (1.005, yi), va="center", fontsize=8.5, color=SEC)
ax.set_xlim(0, 1.12)
ax.set_yticks(y, reml["lab"], fontsize=9)
ax.set_xlabel("share of single-game PPR variance (ICC)")
ax.set_title("§2  Variance decomposition — game noise dominates; skill ceiling ρ_max ≈ 0.41")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28))
ax.grid(axis="y", visible=False)
save(fig, "fig03_variance_components.png")

# ---------------------------------------------------------------- fig 4: reliability gate
rel = pd.read_csv(RES / "reliability_table.csv")
rel = rel[rel["sample"] == "primary"].copy().sort_values("rho_full")
fig, ax = plt.subplots(figsize=(7.2, 4.4))
y = np.arange(len(rel))
for yi, r in enumerate(rel.itertuples()):
    ax.plot([r.r_yoy, r.rho_full], [yi, yi], color=GRID, lw=2, zorder=1)
ax.scatter(rel["rho_full"], y, s=48, color=BLUE, zorder=3, label="ρ_full  (split-half → Spearman–Brown)")
ax.scatter(rel["r_yoy"], y, s=48, color=AQUA, zorder=3, label="r_YoY  (year-over-year)")
ax.axvline(0.5, color=RED, lw=1.2, ls=(0, (4, 3)))
ax.annotate("admission floor ρ_full ≥ 0.5", (0.5, len(rel) - 0.3), color=RED, fontsize=8,
            rotation=90, va="top", ha="right")
names = [f"{s}   ({v})" for s, v in zip(rel["stat"], rel["verdict"])]
ax.set_yticks(y, names, fontsize=9)
ax.set_xlabel("reliability / stability correlation")
ax.set_title("§4  Which stats are signal — the covariate admission gate")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
save(fig, "fig04_reliability_gate.png")

# ---------------------------------------------------------------- fig 5: age curve
age = pd.read_csv(RES / "age_curve.csv")
prim = age[(age["sample"] == "primary")]
groups = prim["group"].unique().tolist()
fig, ax = plt.subplots(figsize=(7.2, 4.4))
al = prim[prim["group"] == "all"]
ax.fill_between(al["age"], al["lo"], al["hi"], color=BLUE, alpha=0.15, linewidth=0)
ax.plot(al["age"], al["f_hat"], color=BLUE, lw=2, label="all WRs (spline df 4, 95% CI)")
for g, c in (("high_adot", RED), ("low_adot", AQUA)):
    if g in groups:
        gg = prim[prim["group"] == g]
        ax.plot(gg["age"], gg["f_hat"], color=c, lw=1.6, ls=(0, (5, 3)),
                label=f"{g.replace('_', ' ')} (p = .11 interaction)")
pk = al.loc[al["f_hat"].idxmax()]
ax.scatter([pk["age"]], [pk["f_hat"]], color=INK, s=30, zorder=5)
ax.annotate(f"peak ≈ {pk['age']:.1f}", (pk["age"], pk["f_hat"]), xytext=(8, 6),
            textcoords="offset points", fontsize=9, color=INK)
ax.set_xlabel("age at Sept 1")
ax.set_ylabel("f̂(age): PPG relative shape")
ax.set_title("§5  Age curve — shape identified (APC caveat: linear trend is not)")
ax.legend(frameon=False, fontsize=8.5)
save(fig, "fig05_age_curve.png")

# ---------------------------------------------------------------- fig 6: hetero multipliers
fig, ax = plt.subplots(figsize=(6.4, 3.2))
tiers = ["rookie", "sophomore", "veteran (ref)"]
mult = [0.844, 0.921, 1.0]
lo = [np.exp(-0.279), np.exp(-0.172), 1.0]
hi = [np.exp(-0.060), np.exp(0.007), 1.0]
y = np.arange(3)
ax.hlines(y[:2], lo[:2], hi[:2], color=BLUE, lw=2)
ax.scatter(mult[:2], y[:2], s=60, color=BLUE, zorder=3)
ax.scatter([1.0], [2], s=60, color=MUTED, zorder=3)
ax.axvline(1.0, color=BASE, lw=1)
ax.set_yticks(y, tiers)
ax.set_xlabel("game-level variance multiplier  exp(γ̂)  vs veteran, 95% CI")
ax.set_title("§3  Rookie game-variance inflation: REFUTED (level effect, σ² ∝ μ^1.4)")
save(fig, "fig06_variance_multipliers.png")

# ---------------------------------------------------------------- fig 7: market prior
mp = pd.read_csv(RES / "market_prior.csv")
knots = pd.read_csv(RES / "market_prior_iso_knots.csv")
fit = mp[mp["in_fit"] == True]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.scatter(fit["adp"], fit["ppg"], s=22, color=MUTED, alpha=0.55, zorder=2,
           label="realized PPG, 2015–2024 boards (n = 291)")
xs = np.exp(knots["log_adp"])
ax.step(xs, knots["m"], where="post", color=BLUE, lw=2.2, zorder=4,
        label="isotonic m̂(ADP) — headline prior")
xg = np.linspace(np.log(1.2), np.log(75), 100)
ax.plot(np.exp(xg), 22.57 - 2.315 * xg, color=RED, lw=1.5, ls=(0, (5, 3)), zorder=3,
        label="OLS: 22.57 − 2.32·log ADP")
ax.set_xscale("log")
ax.set_xticks([2, 5, 10, 20, 40, 75], [2, 5, 10, 20, 40, 75])
ax.set_xlabel("preseason ADP (log scale)")
ax.set_ylabel("realized PPR points per game")
ax.set_title("§6.1  The market prior — what an ADP has historically been worth")
ax.legend(frameon=False, fontsize=8.5)
save(fig, "fig07_market_prior.png")

# ---------------------------------------------------------------- fig 8: tier variances
fig, ax = plt.subplots(figsize=(6.4, 3.2))
tiers = ["rookie (n=4)", "sophomore (n=36)", "veteran (n=251)"]
tau = [24.5, 7.9, 11.3]
lo, hi = [1.7, 5.0, 9.3], [35.1, 11.0, 13.2]
y = np.arange(3)
ax.hlines(y, lo, hi, color=BLUE, lw=2)
ax.scatter(tau, y, s=60, color=BLUE, zorder=3)
ax.set_yticks(y, tiers)
ax.set_xlabel("τ̂²(tier): variance of realized PPG around m̂(ADP), bootstrap 95% CI")
ax.set_title("§6.1  Prior spread by tier — rookie>soph>vet ordering FAILED (rookie cell n=4)")
save(fig, "fig08_tier_variances.png")

# ---------------------------------------------------------------- fig 9: shrinkage map
vb = pd.read_csv(RES / "valuation_2026_blind.csv")
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for t, c in (("vet", BLUE), ("soph", YELLOW)):
    sub = vb[vb["tier"] == t]
    ax.scatter(sub["n_eff"], sub["B"], s=52, color=c, alpha=0.9, zorder=3,
               edgecolors=SURF, linewidths=1.4, label=f"{t} ({len(sub)})")
for _, r in vb[vb["n_eff"] <= 1.0].iterrows():
    ax.annotate(r["player"].split()[-1], (r["n_eff"], r["B"]), xytext=(6, -3),
                textcoords="offset points", fontsize=8, color=SEC)
ax.set_xlabel("effective seasons of data  n_eff  (h = 1)")
ax.set_ylabel("weight on market prior  B = V/(V+τ²)")
ax.set_title("§3.4  Estimated shrinkage — thinner history ⟹ closer to the ADP prior")
ax.legend(frameon=False, fontsize=8.5)
save(fig, "fig09_shrinkage_weights.png")

# ---------------------------------------------------------------- fig 10: valuation dumbbell
vf = pd.read_csv(RES / "valuation_2026_final.csv").sort_values("rank_final", ascending=False)
fig, ax = plt.subplots(figsize=(7.4, 8.2))
y = np.arange(len(vf))
for yi, r in enumerate(vf.itertuples()):
    c = AQUA if r.adp_rank > r.rank_final else (RED if r.adp_rank < r.rank_final else GRID)
    ax.plot([r.adp_rank, r.rank_final], [yi, yi], color=c, lw=2, alpha=0.75, zorder=2)
ax.scatter(vf["adp_rank"], y, s=40, facecolors=SURF, edgecolors=MUTED, linewidths=1.4,
           zorder=3, label="market rank (ADP)")
ax.scatter(vf["rank_final"], y, s=46, color=BLUE, zorder=4, label="model rank (θ*)")
ax.set_yticks(y, [f"{r.player}  ({r.theta_star_blind:.1f})" for r in vf.itertuples()], fontsize=8.5)
ax.set_xlabel("WR rank (1 = best)")
ax.invert_xaxis()
ax.set_title("§6.4  Final 2026 board — model θ* vs market ADP rank\n(green = model higher than market, red = lower)")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
save(fig, "fig10_valuation_dumbbell.png")

# ---------------------------------------------------------------- fig 11: LOSO per-fold
lp = pd.read_csv(RES / "loso_predictions.csv")
rows = []
for yr, g in lp.groupby("year"):
    rows.append({"year": yr,
                 "adp": np.sqrt(np.mean((g["ppg"] - g["m_hat"]) ** 2)),
                 "blind": np.sqrt(np.mean((g["ppg"] - g["theta_star"]) ** 2))})
pf = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(7.2, 4.2))
x = np.arange(len(pf))
w = 0.36
ax.bar(x - w / 2, pf["adp"], w, color=MUTED, label="ADP-only  m̂(ADP)", edgecolor=SURF)
ax.bar(x + w / 2, pf["blind"], w, color=BLUE, label="blind posterior  θ*", edgecolor=SURF)
ax.set_xticks(x, pf["year"], fontsize=9)
ax.set_ylabel("out-of-fold RMSE (PPG)")
ax.set_title("§7  Leave-one-season-out — θ* beats blind ADP in 7/10 folds  (DM p = .025)")
ax.legend(frameon=False, fontsize=8.5)
ax.grid(axis="x", visible=False)
better = (pf["blind"] < pf["adp"]).sum()
print(f"blind better in {better}/10 folds")
save(fig, "fig11_loso_per_fold.png")

# ---------------------------------------------------------------- fig 12: edge coefficients
er = pd.read_csv(RES / "edge_regression.csv")
er = er[er["term"] != "const"].copy()
er["t"] = er["t_cluster"]
er = er.sort_values("t")
fig, ax = plt.subplots(figsize=(7.2, 4.6))
y = np.arange(len(er))
colors = [YELLOW if f and not s else BLUE for f, s in zip(er["fdr_survivor"], er["final_survivor"].fillna(False))]
ax.hlines(y, 0, er["t"], color=GRID, lw=2, zorder=1)
ax.scatter(er["t"], y, s=52, color=colors, zorder=3)
for tcrit in (-2.26, 2.26):
    ax.axvline(tcrit, color=RED, lw=1, ls=(0, (4, 3)))
ax.axvline(0, color=BASE, lw=1)
ax.set_yticks(y, er["term"], fontsize=9)
ax.set_xlabel("cluster-robust t (9 df); dashed = |t| = 2.26 (p = .05)")
ax.set_title("§6.2  Edge regression on market residuals — nothing survives FDR + holdout\n(yellow = passed FDR, failed 2023–24 holdout)")
save(fig, "fig12_edge_regression.png")

print("\nall figures in", FIG)

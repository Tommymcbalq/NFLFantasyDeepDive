"""§P2 — Deep-universe market prior REFIT for WR and RB (EDA_PLAN6.md §P).

PRE-SPECIFIED BEFORE ANY FIT (mirrors §6.1 / §G3 exactly; only the panel widens):

  Panel.  ALL rows at the position on each FFC PPR 12-team board 2015-2024 (not the
          top-30 truncation used in §6.1/§G3).  Boards carry 58-74 WR and 53-65 RB per
          year, so the fitted support reaches ADP ~170 and the 2026 top-60 WR (ADP<=136)
          / top-50 RB (ADP<=160) universes are INSIDE it.  Board position governs the
          inclusion rule.  Join: the validated sectionM_common.build_panel (lifted from
          26_sectionL_conversion.py).

  Realized outcome.  Same-season PPG, REG only, under the position-specific §1/§G1
          participation rule: WR keeps games with targets >= 2; RB keeps games with
          carries+targets >= 2.

  Fit floor.  Pre-registered >= 4 included games.  Rows below the floor stay in the panel,
          flagged, and are excluded from m(.) and tau^2 -- UNCHANGED from round 1.  Because
          the floor bites much harder in the tail than in the top 30, the survivorship it
          induces is measured (fraction below floor by ADP decile) and a labelled
          sensitivity fit is run on ALL rows with ppg=0 imputed for 0-game seasons.
          The pre-registered floor version is the headline; the sensitivity is reported
          whether or not it flatters it.

  m(.).   Isotonic regression, monotone DECREASING in log ADP (headline); OLS on log ADP
          for reference.  tau^2(e) = Var(ppg - m_iso) by tier (rookie/soph/vet at the ADP
          year) with 4000-rep bootstrap CIs.

  Movement test (the §P2 deliverable).  The frozen top-30 curves
          (market_prior_iso_knots.csv / _rb.csv) are evaluated against the refit curves on
          (a) the frozen fitted ADP support, (b) the 2026 top-30 board ADPs.  Reported as
          mean/RMS/max |Delta m| in PPG and as the implied Delta theta* on the existing
          boards.  Nothing is tuned to make the movement small.

Outputs: results/market_prior_wr_deep.csv, market_prior_rb_deep.csv,
         market_prior_iso_knots_wr_deep.csv, market_prior_iso_knots_rb_deep.csv,
         tier_variances_wr_deep.csv, tier_variances_rb_deep.csv,
         results/sectionP_curve_movement.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.isotonic import IsotonicRegression

warnings.simplefilter("ignore")
ROOT = Path("/Users/thomasmcnamee/NFL")
sys.path.insert(0, str(ROOT / "scripts"))
from sectionM_common import build_panel, load_weekly, norm_name, collapse_initials  # noqa

RNG = np.random.default_rng(20260811)
YEARS = list(range(2015, 2025))

# ---------------------------------------------------------------- realized PPG
COLS = ["player_id", "position", "season", "week", "season_type",
        "targets", "carries", "fantasy_points_ppr"]
frames = []
for y in YEARS:
    d = pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{y}.csv",
                    usecols=COLS, low_memory=False)
    frames.append(d[d.season_type == "REG"])
wkr = pd.concat(frames, ignore_index=True)
wkr["touches"] = wkr.carries.fillna(0) + wkr.targets.fillna(0)

real_wr = (wkr[wkr.targets.fillna(0) >= 2].groupby(["player_id", "season"])
           .agg(games=("fantasy_points_ppr", "size"),
                ppg=("fantasy_points_ppr", "mean")).reset_index())
real_rb = (wkr[wkr.touches >= 2].groupby(["player_id", "season"])
           .agg(games=("fantasy_points_ppr", "size"),
                ppg=("fantasy_points_ppr", "mean")).reset_index())

# ---------------------------------------------------------------- board panel
wk_join = load_weekly()
panel_all, unmatched, ambig = build_panel(wk_join)
print(f"board rows 2015-2024 (QB/RB/WR/TE): {len(panel_all)}; "
      f"unmatched {len(unmatched)}; ambiguous resolved {len(ambig)}")

meta = pd.read_csv(ROOT / "data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "rookie_season"]).dropna(subset=["gsis_id"])


def build(pos, real):
    p = panel_all[panel_all.pos == pos].copy()
    p = p.dropna(subset=["pid"])
    p = p.merge(real.rename(columns={"player_id": "pid", "season": "year"}),
                on=["pid", "year"], how="left")
    p["games"] = p.games.fillna(0).astype(int)
    p = p.merge(meta.rename(columns={"gsis_id": "pid"}), on="pid", how="left")
    p["exp"] = p.year - p.rookie_season
    p["tier"] = np.select([p.exp == 0, p.exp == 1], ["rookie", "soph"], "vet")
    p.loc[p.rookie_season.isna(), "tier"] = "vet"
    p["in_fit"] = p.games >= 4
    p["adp_rank"] = p.groupby("year").adp.rank(method="first").astype(int)
    return p.reset_index(drop=True)


def fit_iso(df):
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(np.log(df.adp.values), df.ppg.values)
    return iso


def boot_tau2(res, n=4000):
    v = res.values
    if len(v) < 3:
        return np.nan, np.nan
    b = [np.var(RNG.choice(v, len(v), replace=True), ddof=1) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def run(pos, real, tag, old_knots_file, old_tier_file):
    print("\n" + "=" * 74 + f"\n{pos} deep panel\n" + "=" * 74)
    p = build(pos, real)
    n_board = p.groupby("year").size()
    print(f"panel rows {len(p)}  (per year {n_board.min()}-{n_board.max()}); "
          f"unmatched pid dropped: {(panel_all.pos == pos).sum() - len(p)}")
    print(f"below 4-game floor: {(~p.in_fit).sum()} "
          f"({100*(~p.in_fit).mean():.1f}%), of which 0 games "
          f"{(p.games == 0).sum()}")
    p["adp_dec"] = pd.qcut(p.adp, 10, labels=False) + 1
    surv = (p.groupby("adp_dec")
            .agg(n=("in_fit", "size"), adp=("adp", "mean"),
                 frac_below_floor=("in_fit", lambda s: 1 - s.mean()),
                 ppg_fit=("ppg", "mean")).round(3))
    print("\nsurvivorship of the 4-game floor by ADP decile:")
    print(surv.to_string())

    fit = p[p.in_fit].copy()
    ols = smf.ols("ppg ~ np.log(adp)", data=fit).fit(cov_type="HC3")
    print(f"\nOLS ppg ~ log(adp): {ols.params['Intercept']:.3f} "
          f"{ols.params['np.log(adp)']:+.3f}*log(adp)  "
          f"(se {ols.bse['np.log(adp)']:.3f}, R2 {ols.rsquared:.3f}, n {int(ols.nobs)})")

    iso = fit_iso(fit)
    p["m_iso_deep"] = iso.predict(np.log(p.adp.values))
    p["resid_iso"] = p.ppg - p.m_iso_deep
    lev = np.unique(np.round(iso.y_thresholds_, 6))
    print(f"isotonic: {len(lev)} unique levels, "
          f"{iso.y_thresholds_.max():.2f} -> {iso.y_thresholds_.min():.2f} PPG over "
          f"ADP {fit.adp.min():.1f}-{fit.adp.max():.1f}")
    rm_i = float(np.sqrt((fit.ppg - iso.predict(np.log(fit.adp.values))).pow(2).mean()))
    rm_o = float(np.sqrt(ols.resid.pow(2).mean()))
    print(f"in-sample RMSE: isotonic {rm_i:.3f}, OLS {rm_o:.3f}")

    # step-density in tail vs top: pre-registered expectation was flatter tail
    grid = np.array([5, 10, 20, 30, 40, 60, 80, 100, 130, 160], float)
    gv = iso.predict(np.log(grid))
    print("\nfitted curve on a grid:")
    print(pd.DataFrame({"adp": grid, "m": gv.round(2)}).to_string(index=False))
    top = fit[fit.adp_rank <= 30].copy()
    tail = fit[fit.adp_rank > 30].copy()
    top["m_iso_deep"] = iso.predict(np.log(top.adp.values))
    tail["m_iso_deep"] = iso.predict(np.log(tail.adp.values))
    nl_top = len(np.unique(np.round(top.m_iso_deep, 6)))
    nl_tail = len(np.unique(np.round(tail.m_iso_deep, 6)))
    print(f"unique levels spanned: top-30 rows {nl_top} over "
          f"{top.m_iso_deep.max()-top.m_iso_deep.min():.2f} PPG; "
          f"tail rows {nl_tail} over "
          f"{tail.m_iso_deep.max()-tail.m_iso_deep.min():.2f} PPG")

    tv = (p[p.in_fit].groupby("tier")
          .agg(n=("resid_iso", "size"),
               tau2_iso=("resid_iso", lambda x: x.var(ddof=1)),
               mean_resid=("resid_iso", "mean"))
          .reindex(["rookie", "soph", "vet"]).reset_index())
    lo, hi = zip(*[boot_tau2(p[p.in_fit & (p.tier == t)].resid_iso)
                   for t in tv.tier])
    tv["tau2_lo"], tv["tau2_hi"] = lo, hi
    print("\ntau^2 by tier (deep panel):")
    print(tv.round(3).to_string(index=False))
    # tail-vs-top tau^2, descriptive: is the market noisier deep?
    for lab, sub in [("adp_rank<=30", top), ("adp_rank>30", tail)]:
        print(f"  tau^2 all-tier {lab}: {sub.eval('ppg - m_iso_deep').var(ddof=1):.2f} "
              f"(n={len(sub)})")

    # ---- sensitivity: all rows, 0-game seasons imputed ppg = 0
    alt = p.copy()
    alt["ppg_s"] = alt.ppg.fillna(0.0)
    isoS = IsotonicRegression(increasing=False, out_of_bounds="clip")
    isoS.fit(np.log(alt.adp.values), alt.ppg_s.values)
    print("\nSENSITIVITY (all rows, 0-game -> ppg 0):")
    print(pd.DataFrame({"adp": grid, "m_floor": gv.round(2),
                        "m_allrows": isoS.predict(np.log(grid)).round(2)}
                       ).to_string(index=False))

    # ---- movement of the top-30 region vs the frozen curve
    ok = pd.read_csv(ROOT / old_knots_file)
    old_lo, old_hi = np.exp(ok.log_adp.min()), np.exp(ok.log_adp.max())

    def m_old(a):
        return np.interp(np.log(a), ok.log_adp, ok.m)

    supp = np.exp(np.linspace(np.log(old_lo), np.log(old_hi), 400))
    d = iso.predict(np.log(supp)) - m_old(supp)
    print(f"\nMOVEMENT on frozen support ADP {old_lo:.1f}-{old_hi:.1f}: "
          f"mean {d.mean():+.3f}, RMS {np.sqrt((d**2).mean()):.3f}, "
          f"max|.| {np.abs(d).max():.3f} PPG")
    mv = pd.DataFrame({"pos": pos, "adp": supp, "m_old": m_old(supp),
                       "m_deep": iso.predict(np.log(supp)),
                       "delta": d})

    otv = pd.read_csv(ROOT / old_tier_file)[["tier", "n", "tau2_iso"]]
    print("\ntau^2 old (top-30 panel) vs deep:")
    print(otv.merge(tv[["tier", "n", "tau2_iso"]], on="tier",
                    suffixes=("_old", "_deep")).round(2).to_string(index=False))

    p.to_csv(ROOT / f"results/market_prior_{tag}_deep.csv", index=False)
    pd.DataFrame({"log_adp": iso.X_thresholds_, "m": iso.y_thresholds_}).to_csv(
        ROOT / f"results/market_prior_iso_knots_{tag}_deep.csv", index=False)
    tv.to_csv(ROOT / f"results/tier_variances_{tag}_deep.csv", index=False)
    surv.to_csv(ROOT / f"results/sectionP_floor_survivorship_{tag}.csv")
    return p, iso, mv


p_wr, iso_wr, mv_wr = run("WR", real_wr, "wr",
                          "results/market_prior_iso_knots.csv",
                          "results/tier_variances.csv")
p_rb, iso_rb, mv_rb = run("RB", real_rb, "rb",
                          "results/market_prior_iso_knots_rb.csv",
                          "results/tier_variances_rb.csv")
pd.concat([mv_wr, mv_rb]).to_csv(ROOT / "results/sectionP_curve_movement.csv",
                                 index=False)

# ------------------------------------------------- Delta theta* on frozen boards
print("\n" + "=" * 74 + "\nImplied restatement of the frozen 30-man boards\n" + "=" * 74)
for tag, brdf, isod, oldk, sig_f, tv_f in [
        ("WR", "results/valuation_2026_wr_20260809.csv", iso_wr,
         "results/market_prior_iso_knots.csv", "results/sigma2_by_tier.csv",
         "results/tier_variances_wr_deep.csv"),
        ("RB", "results/valuation_rb_2026.csv", iso_rb,
         "results/market_prior_iso_knots_rb.csv", "results/sigma2_by_tier_rb.csv",
         "results/tier_variances_rb_deep.csv")]:
    b = pd.read_csv(ROOT / brdf)
    ok = pd.read_csv(ROOT / oldk)
    b["m_old"] = np.interp(np.log(b.adp), ok.log_adp, ok.m)
    b["m_deep"] = isod.predict(np.log(b.adp.values))
    b["d_m"] = b.m_deep - b.m_old
    sig = pd.read_csv(ROOT / sig_f).set_index("tier").sigma2
    tau_new = pd.read_csv(ROOT / tv_f).set_index("tier").tau2_iso
    ne = b.n_eff.fillna(0.0)
    with np.errstate(divide="ignore"):
        V = b.tier.map(sig) / ne
    Bn = np.where(ne == 0, 1.0, V / (V + b.tier.map(tau_new)))
    th_new = np.where(ne == 0, b.m_deep,
                      (1 - Bn) * b.mu_hat.fillna(0) + Bn * b.m_deep)
    b["theta_deep"] = th_new
    old_theta = b.theta_star
    b["d_theta"] = b.theta_deep - old_theta
    b["rank_old"] = old_theta.rank(ascending=False).astype(int)
    b["rank_new"] = pd.Series(th_new).rank(ascending=False).astype(int)
    b["d_rank"] = b.rank_old - b.rank_new
    print(f"\n{tag}: mean d_m {b.d_m.mean():+.3f}, RMS d_m "
          f"{np.sqrt((b.d_m**2).mean()):.3f}, max|d_m| {b.d_m.abs().max():.3f}; "
          f"RMS d_theta* {np.sqrt((b.d_theta**2).mean()):.3f}, "
          f"max|d_rank| {b.d_rank.abs().max()}, "
          f"Spearman(old,new theta*) "
          f"{b.theta_deep.corr(old_theta, method='spearman'):.4f}")
    nm = "player" if "player" in b.columns else "name"
    print(b.sort_values("d_theta")[[nm, "adp", "tier", "m_old", "m_deep", "d_m",
                                    "theta_star", "theta_deep", "d_theta",
                                    "d_rank"]].round(3).to_string(index=False))
    b.to_csv(ROOT / f"results/sectionP_top30_restatement_{tag.lower()}.csv",
             index=False)
print("\ndone")

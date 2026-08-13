"""§P1/§P2 — the deep 2026 boards: top-60 WR and top-50 RB.

VALUATION RULE (decided from the §P4 LOSO verdicts, each applied to the universe it was
estimated on -- no stratum-dependent rule is invented):

  WR ADP-rank 1-30 : arm (ii), board_value = theta*.  Warrant = §7 (top-30 panel,
                     DM p = .025) and the wide-panel restatement of the same top-30 rows
                     (mean fold gain +0.618, t = +2.59, p = .029).
  WR ADP-rank 31-60: MARKET-ANCHORED, board_value = m_deep(ADP).  Warrant = the
                     pre-specified wide-panel LOSO, which returns p = .233 and therefore
                     does NOT adopt arm (ii) on the wider universe.  theta* is still
                     reported as a diagnostic column; it is not the board value.
  RB ADP-rank 1-50 : MARKET-ANCHORED, board_value = m_deep(ADP), per §G6 and its
                     wide-panel re-run (mean gain -0.313, p = .328, 2.1x the power).

m(.) and tau^2(tier) come from the §P2 deep refit; sigma^2(tier) is unchanged (it was
already estimated on the full positional population, not on the board).  mu_hat/n_eff are
the §1 estimator (recency half-life h = 1) on every included game the player has.
Thin-data discipline per §G4: zero NFL rows => n_eff = 0 => B = 1 => pure market arm,
flagged.  Rookies are labelled, not tuned.

Descriptive context columns (2025 season, from the advanced layer) are appended for the
tail players and are explicitly NOT inputs to board_value: target_share_full (the
full-team-games denominator; the default target_share sums to 1.36/team-season), red-zone
and inside-10 target share of team, air-yards share, snap share, and the 2026 vacated
target share of the player's team.  routes_proxy-based rates are omitted because the proxy
counts blocking TEs and protecting backs and is biased low across archetypes.

Outputs: results/valuation_wr60_2026.csv, results/valuation_rb50_2026.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression  # noqa: F401  (knots used directly)

warnings.simplefilter("ignore")
ROOT = Path("/Users/thomasmcnamee/NFL")
sys.path.insert(0, str(ROOT / "scripts"))
from sectionM_common import norm_name, collapse_initials  # noqa

ALIASES = {"hollywood brown": "marquise brown", "joshua palmer": "josh palmer"}

# ------------------------------------------------------------------ weekly data
COLS = ["player_id", "player_display_name", "position", "season", "week",
        "season_type", "targets", "carries", "fantasy_points_ppr"]
frames = []
for y in range(1999, 2026):
    d = pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{y}.csv",
                    usecols=lambda c: c in COLS, low_memory=False)
    frames.append(d[d.season_type == "REG"])
wk = pd.concat(frames, ignore_index=True)
wk["touches"] = wk.carries.fillna(0) + wk.targets.fillna(0)
wk["nname"] = wk.player_display_name.map(norm_name).map(collapse_initials)

pdir = (wk.groupby("player_id")
        .agg(nname=("nname", "first"), last=("season", "max"),
             pos=("position", lambda s: s.mode().iat[0] if len(s.mode()) else "NA")
             ).reset_index())
meta = pd.read_csv(ROOT / "data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "display_name", "position", "rookie_season",
                            "birth_date"]).dropna(subset=["gsis_id"])
meta["nname"] = meta.display_name.map(norm_name).map(collapse_initials)

adp26 = pd.read_csv(ROOT / "data/adp/adp_ppr_2026_all_20260809.csv")


def match(nm, pos):
    n = collapse_initials(norm_name(nm))
    n = ALIASES.get(n, n)
    c = pdir[(pdir.nname == n) & (pdir.pos.isin({"WR", "TE", "RB", "FB", "HB"}))]
    if len(c) > 1:
        c2 = c[c.pos == pos]
        c = c2 if len(c2) else c
    if len(c) > 1:
        c = c.sort_values("last").tail(1)
    if len(c):
        return c.player_id.iat[0]
    m = meta[(meta.nname == n) & (meta.position == pos)]
    return m.gsis_id.iat[0] if len(m) else None


def build(pos, n_top, inc_mask, knots_f, tau_f, sig_f, out_f, arm_rule):
    print("\n" + "=" * 76 + f"\n{pos} top-{n_top} 2026 board\n" + "=" * 76)
    b = adp26[adp26.position == pos].sort_values("adp").head(n_top).copy()
    b["adp_rank"] = range(1, len(b) + 1)
    b["gsis_id"] = [match(n, pos) for n in b.name]
    print(f"universe {len(b)} players, ADP {b.adp.min():.1f}-{b.adp.max():.1f}; "
          f"no NFL rows: {b.gsis_id.isna().sum()} "
          f"({b.loc[b.gsis_id.isna(), 'name'].tolist()})")

    inc = wk[inc_mask(wk)]
    seas = (inc[inc.player_id.isin(set(b.gsis_id.dropna()))]
            .groupby(["player_id", "season"])
            .agg(ybar=("fantasy_points_ppr", "mean"),
                 G=("fantasy_points_ppr", "size")).reset_index())
    mus = {}
    for pid, g in seas.groupby("player_id"):
        s = g.season.values
        w = 2.0 ** (-(s.max() - s) / 1.0)
        mus[pid] = (float((w * g.ybar.values).sum() / w.sum()),
                    float(w.sum() ** 2 / (w ** 2).sum()),
                    int(len(s)), int(g.G.sum()))
    b["mu_hat"] = [mus.get(g, (np.nan,))[0] if g in mus else np.nan for g in b.gsis_id]
    b["n_eff"] = [mus[g][1] if g in mus else 0.0 for g in b.gsis_id]
    b["n_seasons"] = [mus[g][2] if g in mus else 0 for g in b.gsis_id]
    b["n_games"] = [mus[g][3] if g in mus else 0 for g in b.gsis_id]

    mm = meta.set_index("gsis_id")
    b["rookie_season"] = b.gsis_id.map(mm.rookie_season)
    b["exp"] = 2026 - b.rookie_season
    b["tier"] = np.select([b.exp == 0, b.exp == 1], ["rookie", "soph"], "vet")
    b.loc[b.rookie_season.isna(), "tier"] = "rookie"
    print("tier counts:", b.tier.value_counts().to_dict())

    kn = pd.read_csv(ROOT / knots_f)
    b["m_deep"] = np.interp(np.log(b.adp), kn.log_adp, kn.m)
    tau = pd.read_csv(ROOT / tau_f).set_index("tier").tau2_iso
    sig = pd.read_csv(ROOT / sig_f).set_index("tier").sigma2
    b["tau2"] = b.tier.map(tau)
    b["sigma2_tier"] = b.tier.map(sig)
    with np.errstate(divide="ignore"):
        b["V"] = b.sigma2_tier / b.n_eff
    nop = b.n_eff == 0
    b["B"] = np.where(nop, 1.0, b.V / (b.V + b.tau2))
    b["theta_star"] = np.where(nop, b.m_deep,
                               (1 - b.B) * b.mu_hat.fillna(0) + b.B * b.m_deep)
    b["post_SD"] = np.where(nop, np.sqrt(b.tau2),
                            np.sqrt(1.0 / (1.0 / b.V + 1.0 / b.tau2)))
    b["thin_data_flag"] = np.where(
        nop, "no NFL rows: full shrinkage to market",
        np.where(b.n_seasons == 1, "single season: n_eff = 1", ""))
    b["arm"] = arm_rule(b)
    b["board_value"] = np.where(b.arm == "theta_star", b.theta_star, b.m_deep)
    return b


def add_context(b, adv_f, extra):
    a = pd.read_csv(ROOT / adv_f, low_memory=False)
    a25 = a[a.season == 2025].copy()
    # team red-zone / inside-10 target totals from BOTH receiver files
    aw = pd.read_csv(ROOT / "data/derived/adv_wr_te.csv", low_memory=False)
    ar = pd.read_csv(ROOT / "data/derived/adv_rb.csv", low_memory=False)
    tm = (pd.concat([aw[aw.season == 2025][["team", "rz_targets", "i10_targets"]],
                     ar[ar.season == 2025][["team", "rz_targets", "i10_targets"]]])
          .groupby("team").sum().rename(columns={"rz_targets": "tm_rz",
                                                 "i10_targets": "tm_i10"}))
    a25 = a25.join(tm, on="team")
    a25["rz_tgt_share_2025"] = a25.rz_targets / a25.tm_rz
    a25["i10_tgt_share_2025"] = a25.i10_targets / a25.tm_i10
    cols = ["player_id", "team", "games", "ppr_pg", "target_share_full",
            "air_yards_share_full", "snap_share", "rz_tgt_share_2025",
            "i10_tgt_share_2025"] + extra
    cols = [c for c in cols if c in a25.columns]
    a25 = a25[cols].rename(columns={"player_id": "gsis_id", "team": "team_2025",
                                    "games": "g_2025", "ppr_pg": "ppg_2025",
                                    "target_share_full": "tgt_share_2025",
                                    "air_yards_share_full": "ay_share_2025",
                                    "snap_share": "snap_share_2025"})
    a25 = a25.groupby("gsis_id", as_index=False).first()
    b = b.merge(a25, on="gsis_id", how="left")
    vac = pd.read_csv(ROOT / "data/derived/vacated_2026.csv")[["team", "vacated_share"]]
    # nflverse writes the Rams as LA, FFC as LAR; normalise before the join
    vac["team"] = vac.team.replace({"LA": "LAR"})
    return b.merge(vac.rename(columns={"vacated_share": "team_vacated_share_2026"}),
                   on="team", how="left")


# --------------------------------------------------------------------- WR60
wr = build("WR", 60, lambda w: w.targets.fillna(0) >= 2,
           "results/market_prior_iso_knots_wr_deep.csv",
           "results/tier_variances_wr_deep.csv",
           "results/sigma2_by_tier.csv", None,
           lambda d: np.where(d.adp_rank <= 30, "theta_star", "m_deep (market)"))
wr = add_context(wr, "data/derived/adv_wr_te.csv",
                 ["targets", "rec_tds", "deep_target_rate", "wopr"])
wr = wr.sort_values("board_value", ascending=False).reset_index(drop=True)
wr["rank_model"] = wr.index + 1
wr["delta_rank_vs_adp"] = wr.adp_rank - wr.rank_model
COLS_WR = ["rank_model", "name", "team", "adp", "adp_rank", "tier", "n_seasons",
           "n_games", "mu_hat", "n_eff", "m_deep", "tau2", "B", "theta_star",
           "post_SD", "arm", "board_value", "delta_rank_vs_adp", "thin_data_flag",
           "team_2025", "g_2025", "ppg_2025", "targets", "tgt_share_2025",
           "ay_share_2025", "snap_share_2025", "rz_tgt_share_2025",
           "i10_tgt_share_2025", "rec_tds", "team_vacated_share_2026"]
wr[[c for c in COLS_WR if c in wr.columns]].to_csv(
    ROOT / "results/valuation_wr60_2026.csv", index=False)
print("\n" + wr[[c for c in COLS_WR if c in wr.columns]].round(3).to_string(index=False))

# --------------------------------------------------------------------- RB50
rb = build("RB", 50, lambda w: w.touches >= 2,
           "results/market_prior_iso_knots_rb_deep.csv",
           "results/tier_variances_rb_deep.csv",
           "results/sigma2_by_tier_rb.csv", None,
           lambda d: np.full(len(d), "m_deep (market)"))
rb = add_context(rb, "data/derived/adv_rb.csv",
                 ["carries", "carry_share_full", "touches_pg", "gl5_carries",
                  "gl10_carries", "targets"])
rb = rb.sort_values("board_value", ascending=False).reset_index(drop=True)
rb["rank_model"] = rb.index + 1
rb["delta_rank_vs_adp"] = rb.adp_rank - rb.rank_model
COLS_RB = ["rank_model", "name", "team", "adp", "adp_rank", "tier", "n_seasons",
           "n_games", "mu_hat", "n_eff", "m_deep", "tau2", "B", "theta_star",
           "post_SD", "arm", "board_value", "delta_rank_vs_adp", "thin_data_flag",
           "team_2025", "g_2025", "ppg_2025", "carries", "carry_share_full",
           "touches_pg", "gl10_carries", "targets", "tgt_share_2025",
           "snap_share_2025"]
rb[[c for c in COLS_RB if c in rb.columns]].to_csv(
    ROOT / "results/valuation_rb50_2026.csv", index=False)
print("\n" + rb[[c for c in COLS_RB if c in rb.columns]].round(3).to_string(index=False))
print("\nwrote valuation_wr60_2026.csv, valuation_rb50_2026.csv")

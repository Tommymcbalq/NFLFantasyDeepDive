#!/usr/bin/env python3
"""§Q — per-player game-level distribution layer, all four positions.

DESCRIPTIVE INFRASTRUCTURE ONLY.  Nothing here is a hypothesis test, nothing enters any
FDR family, and nothing feeds theta*.  The purpose is the shape context an owner reads
next to a point estimate: two players priced identically can have completely different
per-game distributions, and one mean cannot express that.

Everything below reuses frozen upstream choices; no threshold, gate or estimator is
invented here.

  inclusion rules (§0 / §G1 / §O2, all frozen before this script existed)
      WR, TE : drop player-games with targets <= 1
      RB     : drop player-games with touches = carries + targets <= 1
      QB     : drop player-games with pass attempts <= 5
      regular season only, everywhere.

  boom / bust thresholds
      WR 20 / 8       -- §1, fixed a priori by the plan (not data-derived)
      RB 13.8 / 3.2   -- §G1, pooled p75/p25 of all qualified RB player-games 2014-25
      TE 11.1 / 3.4   -- §O2, same construction on TE
      QB 20.9 / 9.7   -- §O2, same construction on QB
  The RB/TE/QB numbers are recomputed here from the population and asserted against the
  frozen values, so drift is detected rather than silently absorbed.

  EB stabilisation of the rate stats (§1.4)
      k_i | p_i ~ Bin(m_i, p_i),  p_i ~ Beta(alpha, beta),  (alpha, beta) by moments:
      Var(p_hat) = Var(p) + E[p(1-p)/m]  =>  subtract the average binomial noise from the
      cross-player sample variance of p_hat, then alpha+beta = mu(1-mu)/Var(p) - 1.
      REFIT PER POSITION AND PER WINDOW on the board universe (m_i differs by window, so
      the noise term differs).  Fitted on rows with m_i >= 8 only -- the moment estimator
      is dominated by the noise correction when m is tiny -- then applied to every row.

  sample-size discipline
      quantiles are emitted as null when n_games < 8.  Never imputed.  Players with no NFL
      rows (rookies, 2026 draftees) get one flagged row per window with all statistics null.

  partial-season flag (§P)
      any player-season with < 12 games played is flagged.  §P found the data arm's
      deviation from market is worth c = +1.101 when mu_hat was earned in a full prior
      season and c = +0.042 when it was not -- a per-game rate earned in 6 games was
      earned in a role the player may not hold.  Pooled windows carry the count and the
      share of games sitting inside such seasons.

Windows (one row per player x window):
      career      -- every included game the player has
      last3       -- seasons 2023-2025 (matches the advanced layer's recent-3 window)
      recent      -- 2025 only
      season_YYYY -- every individual season, so a single outlier year is visible rather
                     than averaged into the pooled shape

Outputs:
      results/player_distributions.csv
      results/figures/dist_tier_*.png, results/figures/dist_eb_shrinkage.png
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
ROOT = Path("/Users/thomasmcnamee/NFL")
sys.path.insert(0, str(ROOT / "scripts"))
from sectionM_common import norm_name, collapse_initials  # noqa: E402

Y = "fantasy_points_ppr"
ALIASES = {"hollywood brown": "marquise brown", "joshua palmer": "josh palmer"}

# frozen upstream thresholds -- position: (boom, bust, source)
THRESH = {
    "WR": (20.0, 8.0, "§1, fixed a priori by the plan"),
    "RB": (13.8, 3.2, "§G1, pooled positional p75/p25 of qualified games"),
    "TE": (11.1, 3.4, "§O2, pooled positional p75/p25 of qualified games"),
    "QB": (20.9, 9.7, "§O2, pooled positional p75/p25 of qualified games"),
}
MIN_N = 8          # refuse to print a quantile below this
EB_FIT_MIN = 8     # rows entering the moment fit for (alpha, beta)
FULL_SEASON = 12   # §P partial-season cut

# ------------------------------------------------------------------ weekly panel
COLS = ["player_id", "player_display_name", "position", "season", "week", "season_type",
        "team", "targets", "carries", "attempts", Y]
frames = []
for yr in range(2014, 2026):
    d = pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{yr}.csv",
                    usecols=lambda c: c in COLS, low_memory=False)
    frames.append(d[d.season_type == "REG"])
wk = pd.concat(frames, ignore_index=True)
for c in ("targets", "carries", "attempts"):
    wk[c] = wk[c].fillna(0.0)
wk["touches"] = wk.carries + wk.targets
wk["nname"] = wk.player_display_name.map(norm_name).map(collapse_initials)


def included(df, pos):
    if pos in ("WR", "TE"):
        return df.targets >= 2
    if pos == "RB":
        return df.touches >= 2
    if pos == "QB":
        return df.attempts >= 6
    raise ValueError(pos)


# ---- verify the frozen thresholds still reproduce off the population -------------
print("threshold reproduction (population 2014-2025, qualified games):")
for pos, (bm, bs, src) in THRESH.items():
    pool = wk[wk.position == pos]
    pool = pool[included(pool, pos)]
    p75, p25 = np.percentile(pool[Y], [75, 25])
    tag = "fixed a priori (not data-derived)" if pos == "WR" else \
          ("MATCH" if (abs(p75 - bm) < .06 and abs(p25 - bs) < .06) else "*** DRIFT ***")
    print(f"  {pos}: n={len(pool):6d}  p75={p75:6.2f} p25={p25:5.2f}   "
          f"frozen {bm}/{bs}  [{tag}]")

# ------------------------------------------------------------------ the universe
pdir = (wk.groupby("player_id")
        .agg(nname=("nname", "first"), last=("season", "max"),
             pos=("position", lambda s: s.mode().iat[0] if len(s.mode()) else "NA"))
        .reset_index())
meta = pd.read_csv(ROOT / "data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "display_name", "position", "rookie_season"]
                   ).dropna(subset=["gsis_id"])
meta["nname"] = meta.display_name.map(norm_name).map(collapse_initials)


def match(nm, pos):
    n = collapse_initials(norm_name(nm))
    n = ALIASES.get(n, n)
    c = pdir[(pdir.nname == n) & (pdir.pos.isin({"WR", "TE", "RB", "FB", "HB", "QB"}))]
    if len(c) > 1:
        c2 = c[c.pos == pos]
        c = c2 if len(c2) else c
    if len(c) > 1:
        c = c.sort_values("last").tail(1)
    if len(c):
        return c.player_id.iat[0]
    m = meta[(meta.nname == n) & (meta.position == pos)]
    if len(m):
        return m.gsis_id.iat[0]
    # two-way players carry a defensive position label in the weekly file, so the
    # offensive-position screen above misses them.  Fall back to name alone.
    c = pdir[pdir.nname == n]
    if len(c):
        c = c.sort_values("last").tail(1)
        print(f"  [note] '{nm}' matched only without the offensive-position screen "
              f"(weekly position label '{c.pos.iat[0]}')")
        return c.player_id.iat[0]
    return None


bd = pd.read_csv(ROOT / "results/board_2026_full.csv")
uni = bd[["name", "team", "adp", "position", "value_final"]].rename(
    columns={"position": "pos", "value_final": "board_value"})
for f, pos in [("valuation_te_2026.csv", "TE"), ("valuation_qb_2026.csv", "QB")]:
    d = pd.read_csv(ROOT / "results" / f)
    uni = pd.concat([uni, pd.DataFrame(dict(
        name=d.player, team=d.team, adp=d.adp, pos=pos, board_value=d.board_value,
        gsis_id=d.gsis_id))], ignore_index=True)
need = uni.gsis_id.isna() if "gsis_id" in uni else pd.Series(True, index=uni.index)
uni.loc[need, "gsis_id"] = [match(n, p) for n, p in
                            zip(uni.loc[need, "name"], uni.loc[need, "pos"])]
uni["adp_rank_pos"] = uni.groupby("pos").adp.rank(method="first").astype(int)
print(f"\nuniverse {len(uni)} players: {uni.pos.value_counts().to_dict()}")
nom = uni[uni.gsis_id.isna()]
print(f"no NFL rows ({len(nom)}): {sorted(nom.name)}")

ids = set(uni.gsis_id.dropna())
panel = wk[wk.player_id.isin(ids)].merge(
    uni[["gsis_id", "pos"]].dropna().rename(columns={"gsis_id": "player_id"}),
    on="player_id", how="left")
panel["inc"] = False
for pos in THRESH:
    m = panel.pos == pos
    panel.loc[m, "inc"] = included(panel[m], pos).values

# games *played* per player-season (all REG rows, pre-exclusion) -> §P flag
gp = (panel.groupby(["player_id", "season"]).size()
      .rename("g_played").reset_index())
gp["partial"] = gp.g_played < FULL_SEASON

# ------------------------------------------------------------------ per-window stats
WINDOWS = [("career", None), ("last3", (2023, 2025)), ("recent", (2025, 2025))] + \
          [(f"season_{s}", (s, s)) for s in range(2014, 2026)]
QS = dict(p10=.10, p25=.25, median=.50, p75=.75, p90=.90)


def window_row(g, gpp, pos, wname):
    """g = included games in window; gpp = games-played table restricted to window."""
    bm, bs = THRESH[pos][0], THRESH[pos][1]
    n = len(g)
    seas = g.groupby("season")[Y].agg(["mean", "var", "size"])
    row = dict(window=wname, n_seasons=int(len(seas)), n_games=int(n),
               n_games_played=int(gpp.g_played.sum()) if len(gpp) else 0,
               seasons_in_window="|".join(str(s) for s in sorted(seas.index)),
               n_seasons_partial=int(gpp.partial.sum()) if len(gpp) else 0,
               share_games_partial=(float(gpp.loc[gpp.partial, "g_played"].sum()
                                          / gpp.g_played.sum()) if len(gpp) else np.nan),
               boom_thresh=bm, bust_thresh=bs)
    if n == 0:
        return row
    y = g[Y].values
    row["mean"] = float(y.mean())
    row["k_boom"] = int((y > bm).sum())
    row["k_bust"] = int((y < bs).sum())
    row["boom_raw"] = row["k_boom"] / n
    row["bust_raw"] = row["k_bust"] / n
    if n >= 2:
        row["sd"] = float(y.std(ddof=1))
    dfree = (seas["size"] - 1).values
    if dfree.sum() > 0:
        s2 = float((dfree[dfree > 0] * seas["var"].values[dfree > 0]).sum() / dfree.sum())
        row["sigma_W"] = float(np.sqrt(s2))
    if n >= MIN_N:                                   # sample-size discipline
        row["min"] = float(y.min())
        row["max"] = float(y.max())
        for k, q in QS.items():
            row[k] = float(np.quantile(y, q))
        # robust dispersion + shape.  PPR is right-skewed, so sd/cv are pulled by the
        # ceiling games; iqr and mad are the outlier-resistant companions, and skew says
        # how much of the mean is coming from the upper tail.  (Quantiles themselves are
        # equivariant under log(1+Y), so a log sensitivity would reproduce them exactly;
        # only the moment-based columns would change.)
        row["iqr"] = row["p75"] - row["p25"]
        row["mad"] = float(1.4826 * np.median(np.abs(y - np.median(y))))
        row["skew"] = float(pd.Series(y).skew())
    return row


rows = []
for _, p in uni.iterrows():
    base = dict(gsis_id=p.gsis_id, player=p["name"], pos=p.pos, team=p.team, adp=p.adp,
                adp_rank_pos=p.adp_rank_pos, board_value=p.board_value)
    if pd.isna(p.gsis_id):
        for wname, _rng in WINDOWS[:3]:
            rows.append({**base, "window": wname, "n_games": 0, "n_seasons": 0,
                         "no_nfl_rows": True,
                         "boom_thresh": THRESH[p.pos][0], "bust_thresh": THRESH[p.pos][1]})
        continue
    pg = panel[(panel.player_id == p.gsis_id) & panel.inc]
    pgall = gp[gp.player_id == p.gsis_id]
    for wname, rng in WINDOWS:
        if rng is None:
            g, gpp = pg, pgall
        else:
            g = pg[(pg.season >= rng[0]) & (pg.season <= rng[1])]
            gpp = pgall[(pgall.season >= rng[0]) & (pgall.season <= rng[1])]
        if wname.startswith("season_") and len(g) == 0:
            continue                                  # don't row seasons he didn't play
        rows.append({**base, "no_nfl_rows": False,
                     **window_row(g, gpp, p.pos, wname)})
out = pd.DataFrame(rows)
out["cv"] = out.sigma_W / out["mean"]
out["cv_sd"] = out.sd / out["mean"]
out["thin_flag"] = out.n_games < MIN_N
out["partial_flag"] = out.n_seasons_partial > 0

# ------------------------------------------------------------------ EB stabilisation
def eb_beta(k, m):
    """§1.4 method of moments.  Returns (alpha, beta, mu_p, var_phat, noise)."""
    p = k / m
    mu = p.mean()
    vph = p.var(ddof=1)
    noise = mu * (1 - mu) * np.mean(1.0 / m)
    vp = vph - noise
    degenerate = vp <= 0
    if degenerate:
        vp = 1e-6
    ab = max(mu * (1 - mu) / vp - 1.0, 1e-6)
    return mu * ab, ab - mu * ab, mu, vph, noise, degenerate


eb_log = []
for rate in ("boom", "bust"):
    for suf in ("_eb", "_eb_pop"):
        out[f"{rate}{suf}"] = np.nan
for c in ("eb_boom_prior_n", "eb_bust_prior_n"):
    out[c] = np.nan
out["eb_degenerate"] = False

# population reference fit: every player at the position with m_i >= 8 in the window,
# not just the board.  Board fits are the headline (project convention, §1/§G/§O all fit
# on the board because the board is the reference class an owner is choosing within);
# the population fit is the range-restriction sensitivity.
pop = wk[wk.position.isin(THRESH)].copy()
pop["inc_p"] = False
for pos in THRESH:
    m = pop.position == pos
    pop.loc[m, "inc_p"] = included(pop[m], pos).values
pop = pop[pop.inc_p]


def pop_counts(pos, rng):
    d = pop[pop.position == pos]
    if rng is not None:
        d = d[(d.season >= rng[0]) & (d.season <= rng[1])]
    bm, bs = THRESH[pos][0], THRESH[pos][1]
    g = d.groupby("player_id")[Y].agg(m="size",
                                      k_boom=lambda s: (s > bm).sum(),
                                      k_bust=lambda s: (s < bs).sum())
    return g[g.m >= EB_FIT_MIN]


WRNG = dict(WINDOWS)
for (pos, wname), grp in out.groupby(["pos", "window"], observed=True):
    fit = grp[(grp.n_games >= EB_FIT_MIN) & grp.k_boom.notna()]
    if len(fit) < 5:
        continue
    pf = pop_counts(pos, WRNG[wname])
    idx = grp.index[grp.k_boom.notna()]
    for rate in ("boom", "bust"):
        a, b, mu, vph, noise, deg = eb_beta(fit[f"k_{rate}"].values.astype(float),
                                            fit.n_games.values.astype(float))
        out.loc[idx, f"{rate}_eb"] = ((out.loc[idx, f"k_{rate}"] + a)
                                      / (out.loc[idx, "n_games"] + a + b))
        out.loc[idx, f"eb_{rate}_prior_n"] = a + b
        # near-degenerate: the between-player signal the moment estimator recovers is a
        # small fraction of the binomial noise it had to subtract, so the prior swamps the
        # data and every player in the cell collapses to the pool rate.  Diagnostic
        # labelling rule, not an inferential test.
        if deg or (vph - noise) < 0.2 * noise:
            out.loc[grp.index, "eb_degenerate"] = True
        ap, bp, mup, vphp, noisep, degp = eb_beta(pf[f"k_{rate}"].values.astype(float),
                                                  pf["m"].values.astype(float))
        out.loc[idx, f"{rate}_eb_pop"] = ((out.loc[idx, f"k_{rate}"] + ap)
                                          / (out.loc[idx, "n_games"] + ap + bp))
        eb_log.append(dict(pos=pos, window=wname, rate=rate, fit="board", n_fit=len(fit),
                           alpha=a, beta=b, prior_mean=a / (a + b), prior_n=a + b,
                           mu_phat=mu, var_phat=vph, binom_noise=noise,
                           var_p=max(vph - noise, 0.0), degenerate=deg))
        eb_log.append(dict(pos=pos, window=wname, rate=rate, fit="population",
                           n_fit=len(pf), alpha=ap, beta=bp, prior_mean=ap / (ap + bp),
                           prior_n=ap + bp, mu_phat=mup, var_phat=vphp,
                           binom_noise=noisep, var_p=max(vphp - noisep, 0.0),
                           degenerate=degp))
eb = pd.DataFrame(eb_log)
eb.to_csv(ROOT / "results/player_distributions_eb_params.csv", index=False)

# ------------------------------------------------------------------ advanced join
ADV = {
    ("WR", "TE"): ("adv_wr_te_recent3.csv", dict(
        adv_games="games", adv_target_share="target_share_full",
        adv_air_yards_share="air_yards_share_full", adv_adot="pfr_adot",
        adv_adot_nflverse="adot_nflverse", adv_yac_per_rec="pfr_yac_per_rec",
        adv_rz_target_share="rz_target_share_of_own", adv_snap_share="snap_share",
        adv_targets_pg="targets_pg", adv_wopr="wopr", adv_catch_rate="catch_rate",
        adv_separation="ngs_avg_separation", adv_deep_target_rate="deep_target_rate")),
    ("RB",): ("adv_rb_recent3.csv", dict(
        adv_games="games", adv_carry_share="carry_share_full",
        adv_target_share="target_share_full", adv_snap_share="snap_share",
        adv_touches_pg="touches_pg", adv_ybc_per_att="pfr_ybc_per_att",
        adv_yac_per_att="pfr_yac_per_att",
        adv_box8_rate="ngs_percent_attempts_gte_eight_defenders",
        adv_ryoe_per_att="ngs_ryoe_per_att", adv_gl5_carries_pg="gl5_carries_pg",
        adv_goal_to_go_pg="goal_to_go_carries_pg", adv_yac_per_rec="pfr_yac_per_rec",
        adv_adot="pfr_adot", adv_explosive_run_rate="explosive_run_rate")),
    ("QB",): ("adv_qb_recent3.csv", dict(
        adv_games="games", adv_attempts_pg="attempts_pg",
        adv_epa_per_dropback="epa_per_dropback", adv_cpoe="cpoe", adv_adot="adot",
        adv_rush_share_of_ppr="rush_share_of_ppr",
        adv_designed_rushes_pg="designed_rushes_pg", adv_gl5_carries_pg="gl5_carries_pg",
        adv_aggressiveness="ngs_aggressiveness", adv_sack_rate="sack_rate",
        adv_snap_share=None, adv_target_share=None)),
}
adv_all = []
for poss, (fn, cmap) in ADV.items():
    a = pd.read_csv(ROOT / "data/derived" / fn)
    keep = {k: v for k, v in cmap.items() if v is not None and v in a.columns}
    t = a[["player_id"] + list(keep.values())].rename(
        columns={v: k for k, v in keep.items()})
    t["_pos_ok"] = ",".join(poss)
    adv_all.append(t)
adv = pd.concat(adv_all, ignore_index=True).drop_duplicates("player_id", keep="first")
out = out.merge(adv.drop(columns=["_pos_ok"]).rename(columns={"player_id": "gsis_id"}),
                on="gsis_id", how="left")

# ------------------------------------------------------------------ write
front = ["gsis_id", "player", "pos", "team", "adp", "adp_rank_pos", "board_value",
         "window", "seasons_in_window", "n_seasons", "n_games", "n_games_played",
         "n_seasons_partial", "share_games_partial", "partial_flag", "thin_flag",
         "no_nfl_rows", "mean", "sd", "sigma_W", "cv", "cv_sd",
         "min", "p10", "p25", "median", "p75", "p90", "max", "iqr", "mad", "skew",
         "boom_thresh", "bust_thresh", "k_boom", "k_bust",
         "boom_raw", "boom_eb", "boom_eb_pop", "bust_raw", "bust_eb", "bust_eb_pop",
         "eb_boom_prior_n", "eb_bust_prior_n", "eb_degenerate"]
out = out[front + [c for c in out.columns if c not in front]]
wo = pd.CategoricalDtype(["career", "last3", "recent"] +
                         [f"season_{s}" for s in range(2025, 2013, -1)], ordered=True)
out["window"] = out.window.astype(wo)
out = out.sort_values(["pos", "adp", "window"]).reset_index(drop=True)
out.to_csv(ROOT / "results/player_distributions.csv", index=False)
print(f"\nwrote results/player_distributions.csv  {out.shape}")
print(out.groupby("pos").window.value_counts().unstack().iloc[:, :3])

# ------------------------------------------------------------------ EB movement report
print("\nEB parameters (pooled windows):")
print(eb[eb.window.isin(["career", "last3", "recent"])]
      [["pos", "window", "rate", "fit", "n_fit", "alpha", "beta", "prior_mean", "prior_n",
        "var_phat", "binom_noise", "var_p", "degenerate"]]
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
for w in ("last3", "recent"):
    d = out[(out.window == w) & out.boom_eb.notna()].copy()
    d["mv_boom"] = (d.boom_eb - d.boom_raw).abs()
    d["mv_bust"] = (d.bust_eb - d.bust_raw).abs()
    print(f"\n[{w}] largest EB movement (|eb - raw|), boom:")
    print(d.nlargest(8, "mv_boom")[["player", "pos", "n_games", "boom_raw", "boom_eb",
                                    "bust_raw", "bust_eb"]].to_string(index=False))

# ------------------------------------------------------------------ shape persistence
# DESCRIPTIVE DIAGNOSTIC, NOT A TEST.  No p-values are interpreted, nothing enters an FDR
# family.  Question: does a player's *shape* persist year to year, or is the pooled shape
# being written by one season?  Consecutive-season pairs among board players, both years
# with n >= 8 included games.
import statsmodels.api as sm  # noqa: E402

sea = out[out.window.astype(str).str.startswith("season_") & (out.n_games >= MIN_N)].copy()
sea["season"] = sea.window.astype(str).str[7:].astype(int)
sea = sea[sea.season >= 2019]
pr = []
for pid, g in sea.groupby("gsis_id"):
    g = g.sort_values("season")
    for a, b in zip(g.itertuples(), g.iloc[1:].itertuples()):
        if b.season - a.season == 1:
            pr.append(dict(pos=a.pos, gsis_id=pid, season=a.season,
                           mean_t=getattr(a, "mean"), mean_t1=getattr(b, "mean"),
                           p25_t=a.p25, p25_t1=b.p25, p90_t=a.p90, p90_t1=b.p90,
                           iqr_t=a.iqr, iqr_t1=b.iqr, sd_t=a.sd, sd_t1=b.sd,
                           bust_t=a.bust_eb, bust_t1=b.bust_eb,
                           boom_t=a.boom_eb, boom_t1=b.boom_eb))
pr = pd.DataFrame(pr)
pr.to_csv(ROOT / "results/player_distribution_shape_persistence.csv", index=False)
print(f"\nshape persistence (descriptive), {len(pr)} consecutive-season pairs 2019-2025, "
      f"n>=8 both years:")
hdr = ("pos", "n", "r(mean)", "r(p25)", "r(p90)", "r(IQR)", "r(SD)", "r(bust)", "r(boom)")
print("{:<5}{:>5}".format(*hdr[:2]) + "".join(f"{h:>10}" for h in hdr[2:]))
for pos, g in list(pr.groupby("pos")) + [("ALL", pr)]:
    def rr(a, b):
        return np.corrcoef(g[a], g[b])[0, 1]
    print(f"{pos:<5}{len(g):>5}" + "".join(
        f"{rr(a + '_t', a + '_t1'):>+10.3f}"
        for a in ("mean", "p25", "p90", "iqr", "sd", "bust", "boom")))
print("\nceiling beyond the mean:  p90_{t+1} ~ 1 + mean_t + p90_t   (OLS, HC3)")
for pos, g in list(pr.groupby("pos")) + [("ALL", pr)]:
    X = sm.add_constant(g[["mean_t", "p90_t"]].values)
    m = sm.OLS(g.p90_t1.values, X).fit(cov_type="HC3")
    print(f"  {pos:<4} n={len(g):>4}  b_mean={m.params[1]:+.3f} ({m.bse[1]:.3f})   "
          f"b_p90={m.params[2]:+.3f} ({m.bse[2]:.3f})   R2={m.rsquared:.3f}")

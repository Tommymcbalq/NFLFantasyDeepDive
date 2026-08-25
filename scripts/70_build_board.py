"""§W2 — THE board builder.  Replaces scripts/50_build_board.py.

One script, one pass, from raw weekly logs to a ranked board.  Every layer is a named column
in the order it is applied, every layer can be switched off from the command line, and the
layer-ablation table is a required output rather than a nicety (EDA_PLAN9 cross-cutting rule 2).

    L0  data          weekly logs 1999-2025, 2026 ADP (FFC; ESPN translated), views
    L3  market        adp -> adp_price_used -> pi_market      isotonic ADP->PPG, deep refit (§P2)
                      mu_hat                                  recency-weighted mean (§43)
                      [proj]                                  WS1 projection, if adopted
                      B, theta_star                           eq. (7) two-way / eq. (W2.4) three-way
                      value_prior                             per the §P4 arm rule
    L4  discretion    view_shift_player                       BL posterior over PLAYER views (§J)
                      value_post_views                        = value_prior + player views.
                                                              STILL A PPG ESTIMATE: this is the
                                                              column the 37 views are scored
                                                              against in January.
                      struct_shift                            STRUCTURAL views (delta_RB)
                      value_ranked = value_post_views + struct_shift   the RANKING quantity
    L5  positional    replacement -> vorp                     per-game-PLAYED basis (L5.1)
                      floor_gap -> final                      lambda = 0.10 (L5.2)

WHAT CHANGED FROM 50_build_board.py, and why each change exists
---------------------------------------------------------------
1. **L5.1, the units defect.**  Player values are PPR per game *played*.  Replacement was the
   (n+1)-th best season TOTAL divided by 17 -- points per *scheduled* week.  Those are
   different quantities and were being subtracted from each other; the difference is exactly
   the marginal player's availability, and availability differs by position, so the defect
   distorted the cross-position contrast, which is the only thing VORP is for.  Fixed by
   identifying replacement as an order statistic OF PPG (see `replacement_levels`).
2. **L4, typed views.**  A view is now `player` or `structural`.  A structural view applies to
   a group; the first is delta_RB, a flat premium on every RB, sized from revealed preference
   and logged in results/views_2026_typed.csv.
3. **L3.2, source translation.**  `--price consensus` blends FFC with ESPN translated into
   FFC-equivalent rank by the monotone map of scripts/69_source_translation.py.  OFF by default;
   the reasons are in results/adp_translation_diag.md and are about identification, not results.
4. **L3.1, the three-way posterior.**  If WS1 adopts a projection, eq. (7) generalises to a
   GLS combination of projection, own history and market price with weights ESTIMATED from the
   LOSO residual covariance.  Derived in `three_way_weights`.  Inert until WS1 ships.

VERIFICATION BUILT IN (kept from 50, extended)
----------------------------------------------
  * `--verify-incumbent` asserts that the chain with every new layer disabled reproduces
    results/board_2026_overall_vorp.csv to machine precision, so every difference on the new
    board is attributable to a named, switched layer.
  * views are asserted to be applied exactly once (five assertions, incl. the double-application
    assertion that caught the historical bug);
  * the structural view is asserted equal to the Omega -> 0 limit of the equivalent set of
    absolute BL views, i.e. it is the BL machinery's own answer, not a bump bolted on beside it;
  * replacement and floor are RECOMPUTED from raw, never hardcoded.

Usage:
    python3 scripts/70_build_board.py                       # the board
    python3 scripts/70_build_board.py --verify-incumbent    # + reproduce 50's board exactly
    python3 scripts/70_build_board.py --ablation            # + the layer-ablation table
"""
import argparse
import json
import sys
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
ROOT = Path("/Users/thomasmcnamee/NFL")
sys.path.insert(0, str(ROOT / "scripts"))
from sectionM_common import norm_name, collapse_initials      # noqa: E402
bl = import_module("19_bl_overlay")                            # noqa: E402

# ------------------------------------------------------------------ frozen constants
ADP_FILE = "data/adp/adp_ppr_2026_all_20260824.csv"           # board universe + prices
ADP_FILE_COMP = "data/adp/adp_ppr_2026_12team_20260812.csv"   # replacement composition
VIEWS_FILE = "results/views_2026_typed.csv"
TRANSLATION = "results/adp_espn_ffc_equiv_2026.csv"
W1_PROJ = "results/sectionW1_projection_2026.csv"             # WS1, if it ships
W1_LOSO = "results/sectionW1_loso_predictions.csv"
MU_STAR_COEFS = "results/mu_star_coefs_2026.json"           # §X, written by 75_mu_star.py
TAU_BL = 0.5                     # §J declared BL tau, never fitted
LAMBDA = 0.10                    # §S4 floor weight
HL = 1.0                         # recency half-life of mu_hat
DEEP_ARM_CUT = 30                # §P4: WR ADP-rank <= 30 takes theta*, else market
ARM_POSITIONS = {"WR"}           # positions the theta* arm is permitted for (see --arm-positions)
FLOOR_YEARS = (2023, 2025)
FLOOR_WEEKS = 18
FLOOR_MIN_SCHED = 34
FLOOR_TOP_N = 70
# Fallback reference set, used ONLY for a position with nobody in the top FLOOR_TOP_N by vorp.
# Counts are 10-team starter demand (roster demand, as §M/§O define their frames), not board
# composition, so the fallback cannot move with the layer being ablated.
FLOOR_STARTERS = {"QB": 10, "RB": 20, "WR": 20, "TE": 10}
REPL_YEARS = (2021, 2025)
REPL_HL = 2.0
REPL_SCHED_WEEKS = 17            # scheduled weeks, used ONLY by the legacy per-week basis
REPL_GMIN = 8                    # L5.1: games threshold for PPG-rank identification
REPL_GMIN_BRACKET = (4, 6, 8, 10, 12)   # declared sensitivity bracket
SUSPENSIONS = {"rashee rice": {2025: 6}}
ALIASES = {"hollywood brown": "marquise brown", "joshua palmer": "josh palmer"}

LAYERS = ("eb", "mu_star", "projection", "views_player", "views_structural",
          "replacement", "floor")


def key(s):
    return collapse_initials(norm_name(s))


# ================================================================== L3.0 mu_star (§X)
def mu_star_column(b, pos):
    """§X's data arm:  mu_star = a + b*mu_hat + c*log[f(age)/f(age-1)].

    a, b, c and the era-3 age curve f come from results/mu_star_coefs_2026.json, written by
    scripts/75_mu_star.py.  Inside the LOSO evaluation they are fitted per training fold; for
    2026 there is no held-out year, so the identical estimator is run on the whole 2015-2024
    panel.  Nothing here is fitted on the board.

    Two properties are worth stating where they are used rather than in a note:
      * b < 1 REGRESSES mu_hat toward the positional mean -- the over-dispersion §W1 measured;
      * the age term is the LOG-RATIO of adjacent curve values, so it prices the transition
        from age-1 to age, not the player's position on the curve.  A 30-year-old who has
        already been 30 on the curve is not charged twice.
    Age is (Sept 1 of the board year - birth_date)/365.25, the project's fixed convention.
    """
    f = ROOT / MU_STAR_COEFS
    if not f.exists():
        return b, f"§X coefficients missing ({MU_STAR_COEFS}): mu_star unavailable"
    C = json.loads(f.read_text())
    if pos not in C:
        return b, f"§X has no {pos} coefficients"
    grid = np.asarray(C["grid"], float)
    curve = np.asarray(C[pos]["curve"], float)
    meta = pd.read_csv(ROOT / "data/meta/players_meta.csv", low_memory=False,
                       usecols=["gsis_id", "birth_date"]).dropna(subset=["gsis_id"])
    bd = meta.set_index("gsis_id").birth_date
    age = ((pd.Timestamp("2026-09-01") - pd.to_datetime(b.gsis_id.map(bd)))
           .dt.days / 365.25)
    b["age_2026"] = age.values
    z = np.log(np.interp(age.values, grid, curve)
               / np.interp(age.values - 1.0, grid, curve))
    b["mu_star_z"] = z
    b["mu_star"] = C[pos]["a"] + C[pos]["b"] * b.mu_hat + C[pos]["c"] * z
    # eq. (7) with mu_star in place of mu_hat.  B, V, tau2 and pi_market are untouched.
    nop = b.n_eff == 0
    b["theta_star_mu_star"] = np.where(nop | b.mu_star.isna(), b.theta_star,
                                       (1 - b.B) * b.mu_star + b.B * b.pi_market)
    n_bad = int(b.mu_star.isna().sum())
    return b, (f"§X {pos}: a={C[pos]['a']:.4f} b={C[pos]['b']:.4f} c={C[pos]['c']:.4f} "
               f"(n={C[pos]['n']}, {C['fitted_on']}); {n_bad} players without a birth date")


# ================================================================== raw weekly data
def load_weekly(y0=1999, y1=2025):
    cols = ["player_id", "player_display_name", "position", "season", "week",
            "season_type", "targets", "carries", "fantasy_points_ppr"]
    fr = []
    for y in range(y0, y1 + 1):
        d = pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{y}.csv",
                        usecols=lambda c: c in cols, low_memory=False)
        fr.append(d[d.season_type == "REG"])
    wk = pd.concat(fr, ignore_index=True)
    wk["touches"] = wk.carries.fillna(0) + wk.targets.fillna(0)
    wk["k"] = wk.player_display_name.map(key)
    return wk


# ================================================================== L5.1 replacement
def replacement_levels(wk, mode="ppg_rank", gmin=REPL_GMIN):
    """Replacement = the value of the best player at a position the draft does NOT consume.

    n is taken from the market's own composition: the 2026 top 140 is 63 WR / 44 RB / 19 QB /
    14 TE, so the (n+1)-th best is the best player not rostered.  That part is unchanged.

    WHAT IS BEING RANKED is the part that was wrong, and it is an identification question, not
    a units conversion.  Three estimands:

      'total_div17'    (the incumbent, retained for reproduction only)
            take the (n+1)-th best SEASON TOTAL, divide by 17.
            This is points per *scheduled week*: rate x availability.  Board values are points
            per game *played*.  Subtracting the first from the second is a category error, and
            not a harmless one -- availability differs systematically by position, so the error
            does not cancel in the cross-position contrast that VORP exists to measure.

      'total_rank_ppg' (the naive fix, reported and rejected)
            rank by SEASON TOTAL, then read that player's PPG.
            This puts the units right and the identification wrong.  It is not an order
            statistic of PPG at all: it is an order statistic of one variable with a second
            variable read off it, and the two are linked by exactly the nuisance we are trying
            to remove.  Among players with the same total, the one who played fewer games has
            the higher rate, so the (n+1)-th by total is selected TOWARD short high-rate
            seasons.  The 2024 WR64 by total is Stefon Diggs: 121.9 points in 8 games, 15.24
            PPG, which is not what "the best WR nobody drafted" is worth.

      'ppg_rank'       (ADOPTED)
            rank by PPG among players with at least `gmin` games, take the (n+1)-th.
            The valued quantity is PPG, so replacement must be an order statistic of PPG.  A
            games threshold is then unavoidable -- PPG at G = 1 is not an estimate of a rate --
            and it is a genuine researcher degree of freedom, so it is handled by declaring a
            bracket rather than a number: gmin in {4,6,8,10,12} is reported in full, gmin = 8
            (half a season, the same convention used elsewhere in the project) is the headline.
            The level is threshold-sensitive by construction; the CONTRAST between positions,
            which is what a board actually uses, is not.  Both are in the diagnostic table.

    Averaged over 2021-25 with recency half-life 2, as before.
    """
    comp = pd.read_csv(ROOT / ADP_FILE_COMP)
    comp = comp[comp.position.isin(["WR", "RB", "TE", "QB"])].sort_values("adp").head(140)
    n_by_pos = comp.position.value_counts().to_dict()

    w = wk[wk.season.between(*REPL_YEARS)]
    ag = (w.groupby(["player_id", "position", "season"]).fantasy_points_ppr
          .agg(total="sum", ppg="mean", g="size").reset_index())
    rows = []
    for pos, n in n_by_pos.items():
        t = ag[ag.position == pos]
        yr, val, who = [], [], []
        for y, g in t.groupby("season"):
            if mode == "ppg_rank":
                s = g[g.g >= gmin].sort_values("ppg", ascending=False)
                v = s.ppg.iat[n] if len(s) > n else np.nan
            else:
                s = g.sort_values("total", ascending=False)
                if len(s) <= n:
                    v = np.nan
                else:
                    v = (s.total.iat[n] / REPL_SCHED_WEEKS if mode == "total_div17"
                         else s.ppg.iat[n])
            yr.append(y)
            val.append(v)
            who.append((s.player_id.iat[n], int(s.g.iat[n])) if len(s) > n else (None, 0))
        yr, val = np.array(yr), np.array(val, dtype=float)
        ww = 2.0 ** (-(REPL_YEARS[1] - yr) / REPL_HL)
        ok = ~np.isnan(val)
        rows.append(dict(position=pos, n_in_top140=n, rank_used=n + 1, mode=mode, gmin=gmin,
                         replacement_ppg=float((ww[ok] * val[ok]).sum() / ww[ok].sum()),
                         marginal_games=float(np.mean([g for _, g in who if g])),
                         by_year=";".join(f"{y}:{v:.2f}" for y, v in zip(yr, val))))
    return pd.DataFrame(rows).sort_values("position").reset_index(drop=True)


def replacement_diagnostics(wk):
    """L5.1's required evidence: every estimand and the whole declared bracket, with the
    cross-position contrasts that the board actually consumes."""
    frames = [replacement_levels(wk, "total_div17"), replacement_levels(wk, "total_rank_ppg")]
    frames += [replacement_levels(wk, "ppg_rank", g) for g in REPL_GMIN_BRACKET]
    d = pd.concat(frames, ignore_index=True)
    d["spec"] = np.where(d["mode"] == "ppg_rank", "ppg_rank g>=" + d.gmin.astype(str), d["mode"])
    wide = d.pivot(index="spec", columns="position", values="replacement_ppg")
    wide["WR-RB"] = wide.WR - wide.RB
    wide["TE-WR"] = wide.TE - wide.WR
    wide["QB-WR"] = wide.QB - wide.WR
    return d, wide


# ================================================================== L5.2 floor (unchanged)
def floor_table(wk):
    """p25 of PPR over SCHEDULED weeks with missed games entered as zeros.  A week the player
    did not play is a bad week for the roster slot holding him; conditioning on games played
    would measure performance-given-availability, which mu_hat already carries.  Suspensions
    leave the denominator (a served suspension is not evidence about 2026).  Below 34 eligible
    weeks no floor is computed, and a player without one takes gap 0 -- never a penalty."""
    w = wk[wk.season.between(*FLOOR_YEARS)]
    rows = []
    for k, g in w.groupby("k"):
        sched, vec = 0, []
        for s, gs in g.groupby("season"):
            n = FLOOR_WEEKS - SUSPENSIONS.get(k, {}).get(s, 0)
            sched += n
            vec += list(gs.fantasy_points_ppr.values) + [0.0] * (n - len(gs))
        rows.append(dict(k=k, sched=sched, played=len(g),
                         floor_p25=float(np.percentile(vec, 25)),
                         bust_sched=float(np.mean(np.array(vec) < 8)),
                         avail=round(len(g) / sched, 3),
                         usable=sched >= FLOOR_MIN_SCHED))
    return pd.DataFrame(rows)


# ================================================================== L3 mu_hat
def summarise_history(seasons, ybars, Gs, games_y, games_s, arm):
    S = seasons.max()
    w = 2.0 ** (-(S - seasons) / HL)
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    if arm == "a1_mean":
        return float((w * ybars).sum() / w.sum()), n_eff
    if arm in ("a7_slope", "a8_usage"):
        raise SystemExit(f"arm '{arm}' has no out-of-panel definition for the 2026 board (§S1).")
    bo = import_module("59_sectionS_bakeoff")
    d = bo.summaries(seasons, ybars, Gs, games_y, games_s, weighted=True)
    return d[arm], n_eff


# ================================================================== L3.2 price
def price_column(b, mode):
    """adp_price_used: the price the isotonic curve is evaluated at.

    'ffc'       -- the FFC pull, the pool the curve was estimated on.  Default.
    'consensus' -- geometric mean of the FFC price and the ESPN price translated into
                   FFC-equivalent rank by the monotone map of 69_source_translation.py.
                   Geometric because the curve is a function of log ADP, so the geometric
                   mean of prices is the arithmetic mean of the quantity the curve sees.
                   OFF by default: see results/adp_translation_diag.md §7 -- the map cannot be
                   validated (the only overlap seasons carrying outcomes are hindsight-
                   contaminated), and the part of ESPN that is informative (positional
                   composition under 10-team/1TE defaults) is precisely the part a monotone
                   rank->rank map cannot transmit.
    """
    b["adp_espn"] = np.nan
    b["adp_ffc_equiv_espn"] = np.nan
    tr = ROOT / TRANSLATION
    if tr.exists():
        t = pd.read_csv(tr)
        t = t[t.pool == "matched"].set_index("k")
        b["adp_espn"] = b.k.map(t.espn_adp)
        b["adp_ffc_equiv_espn"] = b.k.map(t.ffc_equiv)
    if mode == "espn":
        # ESPN's default league is 10-team / 1 TE, which is THIS owner's league. FFC's pool
        # deflates elite TEs by ~17 picks (Bowers 40.7 vs 23.7, McBride 35.3 vs 19.5), and the
        # market prior inherits that error wholesale. Price on the translated ESPN rank where a
        # match exists, else fall back to FFC.
        e = b.adp_ffc_equiv_espn
        b["adp_price_used"] = np.where(e.notna(), e, b.adp)
    elif mode == "consensus":
        e = b.adp_ffc_equiv_espn
        b["adp_price_used"] = np.where(e.notna(), np.exp(0.5 * (np.log(b.adp) + np.log(e))), b.adp)
    else:
        b["adp_price_used"] = b.adp
    return b


# ================================================================== L3.1 three-way blend
def three_way_weights(psi):
    """GLS weights for combining k noisy measurements of the same theta.

    THE DERIVATION (extends eq. 7; notation of §21/§26).

    Eq. (7) is a two-source Bayes update: prior theta ~ N(m, tau^2) from the market price,
    likelihood mu_hat | theta ~ N(theta, V) from the player's own history, giving

        theta* = (1-B) mu_hat + B m,   B = V/(V+tau^2),   1/Var = 1/V + 1/tau^2.        (7)

    A projection y_hat is a THIRD measurement of the same theta.  Write all three as
    theta plus error:

        z = (m, mu_hat, y_hat)' = theta * 1 + e,    e ~ N(0, Psi),   1 = (1,1,1)'.      (W2.1)

    The prior/likelihood distinction is cosmetic here: with a flat hyperprior on theta the
    posterior mean of (W2.1) is the generalised-least-squares estimate of a common mean,

        theta_hat = (1' Psi^-1 z) / (1' Psi^-1 1),   Var = 1/(1' Psi^-1 1),             (W2.2)

    i.e. weights w = Psi^-1 1 / (1' Psi^-1 1), which sum to one by construction.  When Psi is
    diagonal this is exactly precision weighting, w_j proportional to 1/Psi_jj, and dropping
    the third source returns

        w = (1/tau^2, 1/V)/(1/tau^2 + 1/V)  =>  theta_hat = B m + (1-B) mu_hat,         (W2.3)

    which IS eq. (7).  So (W2.2) is a strict generalisation and the incumbent is the special
    case, not a different model.

    TWO THINGS MUST BE RIGHT OR THE WEIGHTS ARE FICTION.

    (a) Psi is variance about TRUE theta, not error against a realised season.  The only
        observable residual is r_j = z_j - Ybar, and Ybar = theta + eta with Var(eta) =
        sigma_W^2/Gbar the same per-game sampling noise eq. (3) and eq. (26.2) already remove
        elsewhere.  Since eta is COMMON to all three residuals,

            Cov(r_j, r_k) = Psi_jk + sigma_W^2/Gbar      for all j, k (incl. j = k),     (W2.4)

        so Psi = Cov(r) - (sigma_W^2/Gbar) J with J the all-ones matrix.  Subtracting a
        rank-one common term from the residual covariance is the whole correction, and it is
        identified because sigma_W^2 and Gbar are estimated independently of the residuals.
        Skipping it inflates every Psi_jj by the same constant, which pushes the weights toward
        equal and understates the posterior precision.

    (b) The three errors are NOT independent.  A projection built from usage and environment
        shares information with mu_hat (both are functions of the same prior seasons) and with
        the market (the market has seen the same usage).  Psi's off-diagonals are therefore
        large and positive, and (W2.2) is the estimator that handles that: correlated sources
        get differenced rather than averaged, and a weight can legitimately go NEGATIVE.  Using
        diag(Psi) instead would double-count shared information and overstate precision.  The
        off-diagonals are estimated from the same LOSO residuals, so nothing here is chosen.

    (c) Each source must be unbiased, E[e_j] = 0, or the weights inherit the bias.  Raw mu_hat
        is not calibrated (WS1's own W1.7 makes the same point), so each source is first
        affinely recalibrated on training folds, a + b*z_j, and Psi is estimated on the
        recalibrated residuals.  This is the same discipline as WS1's mu-cal benchmark.

    ADOPTION.  The three-way arm is used only if WS1 adopts a projection under its own
    four-part rule (W1.7).  Weights are estimated once, out of sample, on the LOSO panel; they
    are not re-estimated per player and they are not tuned on the 2026 board.
    """
    P = np.linalg.inv(psi)
    one = np.ones(len(psi))
    w = P @ one / (one @ P @ one)
    return w, 1.0 / float(one @ P @ one)


def load_projection(b):
    """Read WS1's artefacts if they exist; return (available, per-player projection, weights).

    Inert by design: WS1 has, at the time of writing, published a pre-registration
    (results/sectionW1_notes.md) and no fitted projection.  This function is the adoption
    socket, not an adoption.
    """
    fp, fl = ROOT / W1_PROJ, ROOT / W1_LOSO
    if not (fp.exists() and fl.exists()):
        return False, None, None, "WS1 has not shipped a projection: two-way blend retained"
    lo = pd.read_csv(fl)          # columns: name, season, position, m_adp, mu_hat, proj, ppg, games
    need = {"m_adp", "mu_hat", "proj", "ppg"}
    if not need <= set(lo.columns):
        return False, None, None, f"{W1_LOSO} lacks {sorted(need - set(lo.columns))}"
    src = ["m_adp", "mu_hat", "proj"]
    R = np.column_stack([lo[s].to_numpy(float) - lo.ppg.to_numpy(float) for s in src])
    C = np.cov(R, rowvar=False)
    # common per-game sampling term, eq. (W2.4)
    sig2 = float(np.nanmean(lo.get("sigma2_w", np.nan))) if "sigma2_w" in lo else np.nan
    gbar = float(np.nanmean(lo.get("games", np.nan))) if "games" in lo else np.nan
    common = sig2 / gbar if np.isfinite(sig2) and np.isfinite(gbar) else 0.0
    psi = C - common
    ev = np.linalg.eigvalsh(psi)
    if ev.min() <= 0:                       # keep it a covariance matrix
        psi = psi + (abs(ev.min()) + 1e-6) * np.eye(3)
    w, pv = three_way_weights(psi)
    pr = pd.read_csv(fp).set_index("name").proj
    return True, b.name.map(pr), (w, pv, psi, src), "WS1 projection adopted: three-way blend"


# ================================================================== L3 assembly
def build_wr_rb(wk, pos, adp, mu_arm, price_mode):
    n_top = {"WR": 88, "RB": 68}[pos]
    inc = wk[wk.targets.fillna(0) >= 2] if pos == "WR" else wk[wk.touches >= 2]
    knots = pd.read_csv(ROOT / f"results/market_prior_iso_knots_{pos.lower()}_deep.csv")
    tau = pd.read_csv(ROOT / f"results/tier_variances_{pos.lower()}_deep.csv").set_index("tier").tau2_iso
    sigf = "sigma2_by_tier.csv" if pos == "WR" else "sigma2_by_tier_rb.csv"
    sig = pd.read_csv(ROOT / f"results/{sigf}").set_index("tier").sigma2

    meta = pd.read_csv(ROOT / "data/meta/players_meta.csv", low_memory=False,
                       usecols=["gsis_id", "display_name", "position",
                                "rookie_season"]).dropna(subset=["gsis_id"])
    meta["k"] = meta.display_name.map(key)
    pdir = (wk.groupby("player_id").agg(k=("k", "first"), last=("season", "max"),
                                        pos=("position", lambda s: s.mode().iat[0]
                                             if len(s.mode()) else "NA")).reset_index())

    def match(nm):
        n = ALIASES.get(key(nm), key(nm))
        c = pdir[(pdir.k == n) & (pdir.pos.isin({"WR", "TE", "RB", "FB", "HB"}))]
        if len(c) > 1:
            c2 = c[c.pos == pos]
            c = c2 if len(c2) else c
        if len(c) > 1:
            c = c.sort_values("last").tail(1)
        if len(c):
            return c.player_id.iat[0]
        m = meta[(meta.k == n) & (meta.position == pos)]
        return m.gsis_id.iat[0] if len(m) else None

    b = adp[adp.position == pos].sort_values("adp").head(n_top).copy()
    b["adp_rank"] = range(1, len(b) + 1)
    b["gsis_id"] = [match(n) for n in b.name]
    b["k"] = b.name.map(key)

    ids = set(b.gsis_id.dropna())
    sub = inc[inc.player_id.isin(ids)]
    seas = (sub.groupby(["player_id", "season"])
            .agg(ybar=("fantasy_points_ppr", "mean"),
                 G=("fantasy_points_ppr", "size")).reset_index())
    GL = {p: (g.season.values, g.fantasy_points_ppr.values) for p, g in sub.groupby("player_id")}
    mus = {}
    for pid, g in seas.groupby("player_id"):
        sy, yy = GL[pid]
        mu, ne = summarise_history(g.season.values, g.ybar.values, g.G.values, yy, sy, mu_arm)
        mus[pid] = (mu, ne, int(len(g)), int(g.G.sum()))
    b["mu_hat"] = [mus[g][0] if g in mus else np.nan for g in b.gsis_id]
    b["n_eff"] = [mus[g][1] if g in mus else 0.0 for g in b.gsis_id]
    b["n_seasons"] = [mus[g][2] if g in mus else 0 for g in b.gsis_id]
    b["n_games"] = [mus[g][3] if g in mus else 0 for g in b.gsis_id]

    rs = meta.set_index("gsis_id").rookie_season
    b["rookie_season"] = b.gsis_id.map(rs)
    b["exp"] = 2026 - b.rookie_season
    b["tier"] = np.select([b.exp == 0, b.exp == 1], ["rookie", "soph"], "vet")
    b.loc[b.rookie_season.isna(), "tier"] = "rookie"

    b = price_column(b, price_mode)
    b["pi_market"] = np.interp(np.log(b.adp_price_used), knots.log_adp, knots.m)
    b["tau2"] = b.tier.map(tau)
    b["sigma2_tier"] = b.tier.map(sig)
    with np.errstate(divide="ignore"):
        b["V"] = b.sigma2_tier / b.n_eff
    nop = b.n_eff == 0
    b["B"] = np.where(nop, 1.0, b.V / (b.V + b.tau2))
    b["theta_star"] = np.where(nop, b.pi_market,
                               (1 - b.B) * b.mu_hat.fillna(0) + b.B * b.pi_market)
    b["post_var"] = np.where(nop, b.tau2, 1.0 / (1.0 / b.V + 1.0 / b.tau2))
    # The P4 rule restricted theta* to WR. Section 49 then found mu* helps RB MORE than WR
    # (+3.39, p=.0004 vs +1.71, p=.012), so the restriction is backwards for RB, and TE/QB
    # never see their own data at all -- Warren and Loveland come out identical because they
    # share an isotonic ADP step. --arm-positions widens the permitted set. Default unchanged.
    b["arm"] = np.where((b.adp_rank <= DEEP_ARM_CUT) & (pos in ARM_POSITIONS),
                        "theta_star", "pi_market (market-anchored)")
    b, msg = mu_star_column(b, pos)
    print("  " + msg)
    b["position"] = pos
    b["thin_data_flag"] = np.where(nop, "no NFL rows: full shrinkage to market",
                                   np.where(b.n_seasons == 1, "single season", ""))
    return b


def build_te_qb(pos, price_mode):
    """TE and QB come from the §O valuation files, which apply the identical eq. (7) chain on
    their own panels.  No arm (ii) was adopted at either position, so value_prior = pi_market."""
    f = {"TE": "valuation_te_2026.csv", "QB": "valuation_qb_2026.csv"}[pos]
    d = pd.read_csv(ROOT / f"results/{f}").rename(
        columns={"player": "name", "m_adp": "pi_market", "pos_adp_rank": "adp_rank"})
    d["position"] = pos
    d["k"] = d.name.map(key)
    d["post_var"] = d.post_SD ** 2
    d["arm"] = np.where(d.arm_ii_adopted | (pos in ARM_POSITIONS),
                        "theta_star", "pi_market (market-anchored)")
    d["sigma2_tier"] = np.nan
    d["n_games"] = np.nan
    # §X is a WR/RB object: it is fitted on the §W1 panels and its age curve is §H's, which
    # exists only for WR and RB.  TE/QB take value_prior = pi_market at both settings of the
    # layer, so mu_star is left undefined rather than silently defaulted to something.
    for c in ("age_2026", "mu_star_z", "mu_star"):
        d[c] = np.nan
    d["theta_star_mu_star"] = d.theta_star
    d = price_column(d, price_mode)
    if price_mode == "consensus":
        # TE/QB carry no stored knot file here; the §O curve is applied through its own fitted
        # values, so a translated price cannot be re-evaluated without refitting.  Refusing is
        # the correct behaviour: this is exactly the "do not evaluate the curve off its pool"
        # rule, applied to ourselves.
        d["adp_price_used"] = d.adp
    return d


# ================================================================== the chain
def chain(base, wk, cfg, fl_cache, rep_cache, views, quiet=True):
    """Apply L3->L5 to a pre-assembled base frame under a layer configuration.

    cfg keys are LAYERS; every one can be False.  Returns the board and a diagnostics dict."""
    b = base.copy()
    diag = {}

    # ---- L3.0: which data arm feeds eq. (7) --------------------------------
    # OFF  -> theta_star, i.e. eq. (7) on raw mu_hat: the incumbent, bit-for-bit.
    # ON   -> theta_star_mu_star, i.e. eq. (7) on mu* = a + b*mu_hat + c*log[f(age)/f(age-1)].
    # The switch changes ONE column.  B, V, tau2, pi_market, post_var, the arm rule and every
    # layer above are untouched, which is what makes this ablatable rather than a rebuild.
    if cfg.get("mu_star"):
        b["theta_used"] = b.theta_star_mu_star
        diag["mu_star"] = int((b.theta_star_mu_star != b.theta_star).sum())
    else:
        b["theta_used"] = b.theta_star
        diag["mu_star"] = "layer off"

    # ---- L3: EB arm rule ---------------------------------------------------
    if cfg["eb"]:
        b["value_prior"] = np.where(b.arm == "theta_star", b.theta_used, b.pi_market)
    else:
        b["value_prior"] = b.pi_market            # pure market anchoring, no own history
    b["proj"] = np.nan
    b["w_market"] = np.nan
    b["w_history"] = np.nan
    b["w_projection"] = np.nan
    if cfg["projection"]:
        ok, proj, pack, msg = load_projection(b)
        diag["projection"] = msg
        if ok:
            (w, pv, psi, src) = pack
            b["proj"] = proj
            b["w_market"], b["w_history"], b["w_projection"] = w
            z = np.column_stack([b.pi_market, b.mu_hat, b.proj])
            th3 = z @ w
            use = b.proj.notna() & b.mu_hat.notna() & (b.arm == "theta_star")
            b["theta_star3"] = np.where(use, th3, b.theta_star)
            b["value_prior"] = np.where(use, b.theta_star3, b.value_prior)
            b["post_var"] = np.where(use, pv, b.post_var)
    else:
        diag["projection"] = "layer off"

    # ---- L4: views, applied EXACTLY ONCE ----------------------------------
    names = b.name.tolist()
    pi_bl = b.value_prior.to_numpy(float)
    Sigma = np.diag(b.post_var.to_numpy(float))
    pv = views[views.type == "player"]
    b["view_shift_player"] = 0.0
    contrib, ids = np.zeros((0, len(b))), []
    if cfg["views_player"] and len(pv):
        miss = set(pv.player) - set(names)
        assert not miss, f"views reference players off the board: {sorted(miss)}"
        P, q, Om, ids = bl.build_views(pv, names, pi_bl, Sigma, TAU_BL)
        theta, M, contrib = bl.posterior(pi_bl, Sigma, P, q, Om, TAU_BL)
        b["view_shift_player"] = theta - pi_bl
        b["post_SD_bl"] = np.sqrt(np.diag(M))
        if not quiet:
            # (a) prior is the pre-view chain, bit-identical
            assert np.array_equal(pi_bl, b.value_prior.to_numpy(float))
            # (b) nothing that fed value_prior can contain a view
            for pos in ("WR", "RB"):
                kn = pd.read_csv(ROOT / f"results/market_prior_iso_knots_{pos.lower()}_deep.csv")
                m = b.position == pos
                chk = np.interp(np.log(b.loc[m, "adp_price_used"]), kn.log_adp, kn.m)
                assert np.allclose(chk, b.loc[m, "pi_market"], atol=0, rtol=0), \
                    f"{pos} pi_market is not a pure function of the price -- a view leaked in"
            # (c) diagonal Sigma => an unviewed player must not move
            assert np.array_equal(Sigma, np.diag(np.diag(Sigma))), "Sigma is not diagonal"
            unv = b[~b.name.isin(set(pv.player))]
            leak = float(np.abs(unv.view_shift_player).max())
            assert leak < 1e-9, f"an unviewed player moved by {leak}: views are leaking"
            # (d) decomposition sums to the shift
            assert np.allclose(contrib.sum(axis=0), b.view_shift_player.to_numpy(), atol=1e-12)
            # (e) a second application must MOVE the board (the historical double-apply bug)
            P2, q2, Om2, _ = bl.build_views(pv, names, theta, Sigma, TAU_BL)
            theta2, _, _ = bl.posterior(theta, Sigma, P2, q2, Om2, TAU_BL)
            assert np.abs(theta2 - theta).max() > 1e-8, \
                "double-applying the views is a no-op: the prior already contains them"
            diag["n_double"] = int((np.abs(theta2 - theta) > 1e-9).sum())
    else:
        b["post_SD_bl"] = np.sqrt(TAU_BL * b.post_var)
    for r, vid in enumerate(ids):
        b[f"from_{vid}"] = contrib[r]

    # ---- L4b: STRUCTURAL views ---------------------------------------------
    # A structural view is a statement about the scale on which a GROUP is compared, not a
    # belief about any member's PPG.  Passing it through the BL posterior with a finite Omega
    # would make the realised premium depend on each player's Sigma_ii, which is incoherent:
    # the owner's statement carries no per-player uncertainty.  The coherent BL object is a set
    # of absolute views, one per group member, with a COMMON offset and Omega -> 0.  Under a
    # diagonal Sigma that limit is exactly a flat additive shift on the group and exactly zero
    # elsewhere -- so the flat implementation below IS the BL answer, and `--verify-structural`
    # asserts it numerically against the Omega -> 0 posterior rather than taking the algebra on
    # trust.
    b["struct_shift"] = 0.0
    sv = views[views.type == "structural"]
    if cfg["views_structural"] and len(sv):
        pre = b.value_prior + b.view_shift_player
        for _, r in sv.iterrows():
            kind, val = str(r.scope).split(":")
            assert kind == "position", f"unsupported structural scope {r.scope}"
            mask = (b.position == val).to_numpy()
            assert mask.any(), f"structural view {r.view_id} matches nobody"
            b["struct_shift"] = b.struct_shift + mask * float(r.q)
            b[f"from_{r.view_id}"] = mask * float(r.q)
            if not quiet:
                vv = pd.DataFrame([dict(view_id=f"{r.view_id}_{i}", player=nm, weight=1.0,
                                        q=pre.iat[i] + float(r.q), confidence="certain")
                                   for i, nm in enumerate(names) if mask[i]])
                Pc, qc, Omc, _ = bl.build_views(vv, names, pre.to_numpy(float), Sigma, TAU_BL)
                thc, _, _ = bl.posterior(pre.to_numpy(float), Sigma, Pc, qc, Omc, TAU_BL)
                err = float(np.abs(thc - (pre.to_numpy(float) + mask * float(r.q))).max())
                assert err < 1e-3, f"structural view != Omega->0 BL limit (max err {err:.2e})"
                diag["struct_bl_limit_err"] = err
    # value_post_views stays a POINTS-PER-GAME estimate: prior + player views only.  It is the
    # column the 37 logged views get scored against in January, so a positional preference
    # premium must not be inside it.  The structural view enters the RANKING quantity instead.
    b["value_post_views"] = b.value_prior + b.view_shift_player
    b["value_ranked"] = b.value_post_views + b.struct_shift

    # ---- L5.1: replacement -------------------------------------------------
    if cfg["replacement"]:
        rep = rep_cache
        b = b.merge(rep[["position", "replacement_ppg"]].rename(
            columns={"replacement_ppg": "replacement"}), on="position", how="left")
    else:
        # switched off = one common replacement level for every position, so VORP still exists
        # as a scale but carries no positional information.  (Dropping the layer entirely would
        # change the units of `final` and make the ablation incomparable.)
        b["replacement"] = float(rep_cache.replacement_ppg.mean())
    b["vorp"] = b.value_ranked - b.replacement

    # ---- L5.2: floor -------------------------------------------------------
    b = b.merge(fl_cache[["k", "sched", "avail", "floor_p25", "bust_sched", "usable"]],
                on="k", how="left")
    b["usable"] = b.usable.fillna(False)
    top = b.sort_values("vorp", ascending=False).head(FLOOR_TOP_N)
    ref = top[top.usable].groupby("position").floor_p25.median()
    # ---- the interaction the L5.1 fix exposed, and the minimal repair ----------------
    # The reference is the median floor_p25 of the top FLOOR_TOP_N by vorp -- a set that
    # depends on the replacement layer.  Putting replacement on a per-game-played basis raises
    # TE replacement from 7.87 to 9.79 and empties TE out of that set entirely, leaving
    # floor_ref undefined and NaN-ing `final` for every TE with usable history.  That is a real
    # defect in the incumbent rule, not in the fix: an overall top-N cut can starve a position.
    # Repair, deliberately minimal so L5.2 stays "as calibrated": the top-N rule is unchanged
    # wherever it is defined; a position with NO representative there falls back to its own top
    # FLOOR_STARTERS[p] by vorp.  Under the incumbent configuration the fallback never fires, so
    # the reproduction of 50_build_board.py is untouched.  The fully-positional variant of the
    # rule is reported as a declared diagnostic (`floor_ref_positional` below) rather than
    # adopted, because L5.2 said keep.
    ref_pos = {}
    for p, g in b[b.usable].groupby("position"):
        n_p = FLOOR_STARTERS.get(p, FLOOR_TOP_N)
        ref_pos[p] = float(g.nlargest(n_p, "vorp").floor_p25.median())
    if cfg.get("floor_ref", "top70") == "positional":
        ref_used = dict(ref_pos)
    else:
        ref_used = {p: (ref[p] if p in ref.index and np.isfinite(ref.get(p, np.nan))
                        else ref_pos.get(p, np.nan)) for p in b.position.unique()}
    b["floor_ref"] = b.position.map(ref_used)
    b["floor_ref_positional"] = b.position.map(ref_pos)
    b["floor_ref_source"] = b.position.map(
        {p: (cfg.get("floor_ref", "top70") if p in ref.index else "positional fallback")
         for p in b.position.unique()})
    diag["floor_ref_n"] = top[top.usable].position.value_counts().to_dict()
    b["floor_gap"] = np.where(b.usable, b.floor_p25 - b.floor_ref, 0.0)
    b["final"] = b.vorp + (LAMBDA * b.floor_gap if cfg["floor"] else 0.0)
    diag["floor_ref"] = pd.Series(ref_used)
    diag["floor_ref_positional"] = pd.Series(ref_pos)
    diag["floor_ref_source"] = ref_used and {p: b.loc[b.position==p,"floor_ref_source"].iat[0] for p in ref_used}

    b = b.sort_values("final", ascending=False).reset_index(drop=True)
    b["rank"] = b.index + 1
    b["adp_rank_overall"] = b.adp.rank().astype(int)
    b["edge"] = b.adp_rank_overall - b["rank"]
    return b, diag


# ================================================================== ablation
def ablate(base, wk, cfg0, fl, rep, views):
    """Cross-cutting rule 2: every layer separable, switchable, and its contribution reported.

    Two readings, because they answer different questions.
      * cumulative  -- build the board up one layer at a time; each row is what the board looks
                       like with layers 1..j on.  Answers "what does the stack do".
      * leave-one-out -- turn exactly one layer off in the full board.  Answers "what does THIS
                       layer contribute, given everything else is already there".  These differ
                       whenever layers interact (they do: the floor reference is the median of
                       the top 70 by vorp, so replacement changes who defines the floor).
    Movement is reported in rank terms and in value terms; there is no out-of-sample criterion
    available at L4/L5 (a view and a positional premium are not falsifiable before January), so
    the table reports INFLUENCE, not accuracy, and says so.
    """
    full, _ = chain(base, wk, cfg0, fl, rep, views)
    ref = full.set_index("name")
    rows = []

    def cmp(tag, bb, note=""):
        j = bb.set_index("name")[["rank", "final"]].join(
            ref[["rank", "final"]], rsuffix="_full", how="inner")
        d = (j["rank"] - j.rank_full).abs()
        t24 = len(set(bb.head(24).name) - set(full.head(24).name))
        t50 = len(set(bb.head(50).name) - set(full.head(50).name))
        pos = (bb.set_index("name").position.reindex(j.index))
        moved = j.assign(pos=pos, d=(j["rank"] - j.rank_full))
        worst = moved.d.abs().idxmax()
        by = moved.groupby("pos").d.mean().round(1).to_dict()
        rows.append(dict(variant=tag, spearman=j[["rank", "rank_full"]].corr("spearman").iloc[0, 1],
                         mean_abs_drank=d.mean(), max_abs_drank=int(d.max()),
                         out_of_top24=t24, out_of_top50=t50,
                         mean_abs_dfinal=(j.final - j.final_full).abs().mean(),
                         mean_drank_by_pos=str(by), biggest_mover=f"{worst} ({moved.d[worst]:+d})",
                         note=note))

    # cumulative
    seq = [("L3 market only (pi_market)", dict(cfg0, eb=False, mu_star=False,
                                               views_player=False,
                                               views_structural=False, replacement=False,
                                               floor=False)),
           ("+ L3 EB posterior", dict(cfg0, views_player=False, views_structural=False,
                                      replacement=False, floor=False)),
           ("+ L4 player views", dict(cfg0, views_structural=False, replacement=False,
                                      floor=False)),
           ("+ L4 structural delta_RB", dict(cfg0, replacement=False, floor=False)),
           ("+ L5.1 replacement", dict(cfg0, floor=False)),
           ("+ L5.2 floor  = FULL", dict(cfg0))]
    for tag, c in seq:
        bb, _ = chain(base, wk, c, fl, rep, views)
        cmp("CUM  " + tag, bb)
    # leave-one-out
    for lay, note in [("mu_star", "eq.(7) on raw mu_hat instead of mu* (§X off)"),
                      ("eb", "value_prior = pi_market for every player"),
                      ("views_player", "37 player views off"),
                      ("views_structural", "delta_RB = 0"),
                      ("replacement", "one common replacement level, no positional scarcity"),
                      ("floor", "lambda = 0")]:
        if lay == "projection" or not cfg0.get(lay, False):
            continue
        bb, _ = chain(base, wk, dict(cfg0, **{lay: False}), fl, rep, views)
        cmp("LOO  -" + lay, bb, note)
    return pd.DataFrame(rows), full


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu-arm", default="a1_mean")
    ap.add_argument("--arm-positions", default="WR",
                    help="comma list, or 'all': which positions may use the theta* arm")
    ap.add_argument("--price", default="ffc", choices=["ffc", "consensus", "espn"])
    ap.add_argument("--replacement", default="ppg_rank",
                    choices=["ppg_rank", "total_rank_ppg", "total_div17"])
    ap.add_argument("--gmin", type=int, default=REPL_GMIN)
    ap.add_argument("--views", default=VIEWS_FILE)
    ap.add_argument("--floor-ref", default="top70", choices=["top70", "positional"])
    ap.add_argument("--mu-star", action="store_true",
                    help="§X: replace mu_hat with mu* = a + b*mu_hat + c*log[f(age)/f(age-1)] "
                         "inside eq. (7).  OFF by default so the incumbent reproduces.")
    ap.add_argument("--no-projection", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--verify-incumbent", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results/board_2026_v2.csv"))
    a = ap.parse_args()
    global ARM_POSITIONS
    ARM_POSITIONS = ({"WR","RB","TE","QB"} if a.arm_positions.strip().lower()=="all"
                     else {x.strip().upper() for x in a.arm_positions.split(",") if x.strip()})
    print(f"  theta* arm permitted for: {sorted(ARM_POSITIONS)}")

    wk = load_weekly()
    adp = pd.read_csv(ROOT / ADP_FILE)
    views = pd.read_csv(ROOT / a.views)
    if "type" not in views:
        views["type"], views["scope"] = "player", ""

    parts = [build_wr_rb(wk, "WR", adp, a.mu_arm, a.price),
             build_wr_rb(wk, "RB", adp, a.mu_arm, a.price),
             build_te_qb("TE", a.price), build_te_qb("QB", a.price)]
    keep = ["name", "team", "k", "adp", "adp_espn", "adp_ffc_equiv_espn", "adp_price_used",
            "adp_rank", "position", "tier", "n_seasons", "n_games", "mu_hat", "n_eff",
            "pi_market", "tau2", "sigma2_tier", "V", "B", "theta_star", "age_2026",
            "mu_star_z", "mu_star", "theta_star_mu_star", "post_var", "arm",
            "thin_data_flag"]
    base = pd.concat([p[keep] for p in parts], ignore_index=True)
    assert base.name.duplicated().sum() == 0, "duplicate player on the board"

    fl = floor_table(wk)
    rep = replacement_levels(wk, a.replacement, a.gmin)
    rep_diag, rep_wide = replacement_diagnostics(wk)
    rep_diag.to_csv(ROOT / "results/replacement_identification.csv", index=False)

    cfg = dict(eb=True, mu_star=a.mu_star, projection=not a.no_projection, views_player=True,
               views_structural=True, replacement=True, floor=True, floor_ref=a.floor_ref)
    b, diag = chain(base, wk, cfg, fl, rep, views, quiet=False)

    order = ["rank", "name", "team", "position", "adp", "adp_rank", "adp_rank_overall", "edge",
             "adp_espn", "adp_ffc_equiv_espn", "adp_price_used", "tier", "n_seasons", "n_games",
             "mu_hat", "n_eff", "pi_market", "tau2", "sigma2_tier", "V", "B", "theta_star",
             "age_2026", "mu_star_z", "mu_star", "theta_star_mu_star", "theta_used",
             "proj", "w_market", "w_history", "w_projection", "post_var", "arm", "value_prior",
             "view_shift_player", "value_post_views", "struct_shift", "value_ranked",
             "post_SD_bl",
             "replacement", "vorp", "sched", "avail", "floor_p25", "bust_sched", "usable",
             "floor_ref", "floor_ref_positional", "floor_ref_source", "floor_gap", "final",
             "thin_data_flag"]
    order += [c for c in b.columns if c.startswith("from_")]
    b[order].to_csv(a.out, index=False)

    # ---------------------------------------------------------------- report
    print(f"mu_arm {a.mu_arm} | price {a.price} | replacement {a.replacement} (gmin {a.gmin}) "
          f"| players {len(b)}")
    print(f"projection layer: {diag['projection']}")
    print(f"mu_star layer (§X): {diag['mu_star']}"
          + (" players whose eq.(7) input changed" if a.mu_star else ""))
    print(f"views: 37 player (second application would move {diag.get('n_double')} players "
          f"-> applied once) + structural delta_RB "
          f"(Omega->0 BL-limit check: max err {diag.get('struct_bl_limit_err', float('nan')):.2e})")

    print("\n--- L5.1 replacement identification: every estimand, whole declared bracket ---")
    print(rep_wide.round(3).to_string())
    print("\nADOPTED:", a.replacement, f"gmin={a.gmin}")
    print(rep[["position", "n_in_top140", "rank_used", "replacement_ppg", "marginal_games",
               "by_year"]].to_string(index=False))
    print(f"\nreference floors (median floor_p25, top {FLOOR_TOP_N} by vorp, usable only):")
    print(pd.DataFrame(dict(used=diag["floor_ref"], positional_variant=diag["floor_ref_positional"],
                            source=pd.Series(diag["floor_ref_source"]))).round(4).to_string())

    # The top-N cut is a position-blind set used to define a POSITIONAL statistic, so each
    # position's reference is a median of a different quantile of its own distribution.  That
    # asymmetry pre-dates this round; it is reported, and the symmetric variant is priced.
    topn = b.nlargest(FLOOR_TOP_N, "vorp")
    print(f"  composition of the top {FLOOR_TOP_N} by vorp that defines those medians: "
          f"{diag['floor_ref_n']} -- a positional median on n <= 2 is not a stable statistic, "
          "and the ablation is reported under both reference rules for that reason")
    alt = b.assign(fg=np.where(b.usable, b.floor_p25 - b.floor_ref_positional, 0.0))
    alt["final_alt"] = alt.vorp + LAMBDA * alt.fg
    j = alt.assign(r_alt=alt.final_alt.rank(ascending=False))
    print(f"  fully-positional reference (declared diagnostic, NOT adopted -- L5.2 says keep): "
          f"max |Δfinal| {float((alt.final_alt - alt.final).abs().max()):.3f}, "
          f"Spearman {j[['final','final_alt']].corr('spearman').iloc[0,1]:.5f}, "
          f"{int((j['rank'] != j.r_alt).sum())} rank changes")

    stored = pd.read_csv(ROOT / "results/floor_scheduled.csv")
    chk = stored.merge(fl, on="k", suffixes=("_s", "_n"))
    for c in ("sched", "played", "floor_p25", "bust_sched", "avail"):
        assert float((chk[c + "_s"] - chk[c + "_n"]).abs().max()) < 1e-9, f"floor {c} moved"
    print(f"floor table reproduces results/floor_scheduled.csv on all {len(chk)} shared rows")

    if a.verify_incumbent:
        cfg0 = dict(eb=True, mu_star=False, projection=False, views_player=True,
                    views_structural=False, replacement=True, floor=True, floor_ref="top70")
        v = pd.read_csv(ROOT / "results/views_2026.csv")
        v["type"], v["scope"] = "player", ""
        rep0 = replacement_levels(wk, "total_div17")
        b0, _ = chain(base, wk, cfg0, fl, rep0, v)
        cur = pd.read_csv(ROOT / "results/board_2026_overall_vorp.csv")
        m = b0.merge(cur[["name", "position", "final", "value_final", "vorp_meta", "floor_gap"]],
                     on=["name", "position"], suffixes=("", "_cur"))
        assert len(m) == len(cur), f"universe mismatch {len(m)} vs {len(cur)}"
        R2 = {p: round(vv, 2) for p, vv in zip(rep0.position, rep0.replacement_ppg)}
        m["vorp_r"] = m.value_ranked - m.position.map(R2)
        m["final_r"] = m.vorp_r + LAMBDA * m.floor_gap
        nq = m.position != "QB"
        print("\n--- reproduction of the 50_build_board chain (all new layers off) ---")
        for c, cc in [("value_post_views", "value_final"), ("vorp_r", "vorp_meta"),
                      ("floor_gap", "floor_gap_cur"), ("final_r", "final_cur")]:
            d = float((m.loc[nq, c] - m.loc[nq, cc]).abs().max())
            print(f"  {c:18s} vs {cc:14s} max|diff| = {d:.3e}")
            assert d < 1e-9, f"{c} does not reproduce the incumbent board"
        print("VERIFIED: with every §W2 layer disabled the rebuild reproduces "
              "results/board_2026_overall_vorp.csv to machine precision.")
        j = b.set_index("name")[["rank", "final"]].join(
            b0.set_index("name")[["rank", "final"]], rsuffix="_old", how="inner")
        print(f"  §W2 board vs incumbent: Spearman "
              f"{j[['rank','rank_old']].corr('spearman').iloc[0,1]:.4f}, mean |Δrank| "
              f"{(j['rank']-j.rank_old).abs().mean():.2f}, "
              f"{len(set(b.head(24).name)-set(b0.head(24).name))} changes in the top 24")

    if a.ablation:
        tab, _ = ablate(base, wk, cfg, fl, rep, views)
        tab["floor_ref_rule"] = cfg.get("floor_ref", "top70")
        # The top-N reference set holds 2 QB and 0 TE, so a positional median there is a median
        # of n <= 2 and moves 5.5 PPG between variants -- 0.55 on every QB's `final`, larger
        # than the floor layer's whole intended influence.  That instability contaminates the
        # ablation (it makes a layer look like it moved QBs when all it moved was the reference
        # set), so the table is reported a second time under the stable positional reference.
        tab2, _ = ablate(base, wk, dict(cfg, floor_ref="positional"), fl, rep, views)
        tab2["floor_ref_rule"] = "positional"
        tab = pd.concat([tab, tab2], ignore_index=True)
        tab.to_csv(a.out.replace(".csv", "_ablation.csv"), index=False)
        pd.set_option("display.width", 300, "display.max_colwidth", 46)
        print("\n--- LAYER ABLATION (influence on the board, not accuracy: there is no "
              "out-of-sample criterion for a view or a positional premium before January) ---")
        print(tab.round(4).to_string(index=False))

    if a.ablation:
        # ---- specification counterfactuals: the two researcher degrees of freedom this
        # round actually had (which price, which replacement estimand), priced in board terms.
        rows = []

        def vs(tag, bb, note):
            j = bb.set_index("name")[["rank", "final"]].join(
                b.set_index("name")[["rank", "final"]], rsuffix="_full", how="inner")
            rows.append(dict(variant=tag, spearman=j[["rank", "rank_full"]].corr("spearman").iloc[0, 1],
                             mean_abs_drank=(j["rank"] - j.rank_full).abs().mean(),
                             max_abs_drank=int((j["rank"] - j.rank_full).abs().max()),
                             out_of_top24=len(set(bb.head(24).name) - set(b.head(24).name)),
                             mean_abs_dfinal=(j.final - j.final_full).abs().mean(), note=note))
        for mode, g, note in ([("total_div17", 17, "THE DEFECT: per scheduled week vs per game played")]
                              + [("total_rank_ppg", 0, "rank by total, read PPG (selection toward short seasons)")]
                              + [("ppg_rank", gg, "declared bracket") for gg in REPL_GMIN_BRACKET]):
            if mode == "ppg_rank" and g == a.gmin:
                continue
            bb, _ = chain(base, wk, cfg, fl, replacement_levels(wk, mode, g or 8), views)
            vs(f"replacement = {mode}" + (f" g>={g}" if mode == "ppg_rank" else ""), bb, note)
        pc = [build_wr_rb(wk, "WR", adp, a.mu_arm, "consensus"),
              build_wr_rb(wk, "RB", adp, a.mu_arm, "consensus"),
              build_te_qb("TE", "consensus"), build_te_qb("QB", "consensus")]
        bb, _ = chain(pd.concat([p[keep] for p in pc], ignore_index=True), wk, cfg, fl, rep, views)
        vs("price = consensus (FFC x translated ESPN)", bb,
           "L3.2 counterfactual; NOT adopted, see results/adp_translation_diag.md")
        cf = pd.DataFrame(rows)
        cf.to_csv(a.out.replace(".csv", "_counterfactuals.csv"), index=False)
        print("\n--- SPECIFICATION COUNTERFACTUALS (vs the adopted board) ---")
        print(cf.round(4).to_string(index=False))

    pd.set_option("display.width", 300)
    show = ["rank", "name", "team", "position", "adp", "mu_hat", "B", "pi_market", "value_prior",
            "view_shift_player", "value_post_views", "struct_shift", "value_ranked",
            "replacement", "vorp",
            "floor_gap", "final", "edge"]
    print("\n" + b.head(30)[show].round(3).to_string(index=False))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

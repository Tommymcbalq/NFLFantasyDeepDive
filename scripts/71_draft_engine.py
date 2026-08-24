#!/usr/bin/env python3
"""
WS3 / L6-L7 -- the draft engine.   Pre-registration: EDA_PLAN9.md, WS3.

  L6.1  lineup-marginal value, generalised to EVERY pick and roster state.
  L6.2  the flat-versus-step principle, formalised as an exact identity (W3.4) and
        verified numerically against the simulation.
  L7.1  expected opponent order from the OWNER'S stated beliefs, mock-calibrated noise,
        and owner-supplied availability overrides as HARD CONSTRAINTS on the survival
        distribution (implemented as calibrated latent shifts, not table edits).
  L7.2  contingency trees per pick.
  L7.3  strategy comparison, prospective arm.  The retrospective sec.M re-run on real
        2015-2024 outcomes lives in scripts/72_sectionW3_m_rerun.py.

Parameterised: --board, --slot, --teams, --rounds, --overrides, --delta-rb.
WS2 will emit board_2026_v2.csv; repoint with --board.

Writes only sectionW3_* outputs.  Never touches board_2026.csv / views_2026.csv /
sectionW1_* / sectionW2_*.

Rerun:  python3 scripts/71_draft_engine.py
"""
import os, sys, json, argparse, re, time
import numpy as np
import pandas as pd

ROOT = "/Users/thomasmcnamee/NFL"
RES = os.path.join(ROOT, "results")

ap = argparse.ArgumentParser()
ap.add_argument("--board", default=os.path.join(RES, "board_2026.csv"))
ap.add_argument("--order", default=os.path.join(RES, "expected_order_2026.csv"))
ap.add_argument("--overrides", default=os.path.join(ROOT, "data/drafts/owner_availability_overrides_2026.csv"))
ap.add_argument("--slot", type=int, default=5)
ap.add_argument("--teams", type=int, default=10)
ap.add_argument("--rounds", type=int, default=14)
ap.add_argument("--delta-rb", type=float, default=1.345)
ap.add_argument("--nsim", type=int, default=4000)
ap.add_argument("--nsim-strat", type=int, default=1500)
ap.add_argument("--nsim-cal", type=int, default=1200)
ap.add_argument("--tree-picks", type=int, default=6)
ap.add_argument("--prefix", default="sectionW3")
ap.add_argument("--seed", type=int, default=20260824)
A = ap.parse_args()

TEAMS, ROUNDS, SLOT = A.teams, A.rounds, A.slot
NPICKS = TEAMS * ROUNDS
POS = ["QB", "RB", "TE", "WR"]
PI = {q: i for i, q in enumerate(POS)}
WEEKS = 17

MAE_MOCK = 1.40
SIG_STATED = MAE_MOCK * np.sqrt(np.pi / 2.0)
STATED_DEPTH = 28

SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?")
def nkey(s):
    s = str(s).lower().replace(".", "").replace("'", "").replace("-", " ")
    s = SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()

MY = [r * TEAMS + (SLOT if r % 2 == 0 else TEAMS - SLOT + 1) for r in range(ROUNDS)]

# --------------------------------------------------------------------------- data
board = pd.read_csv(A.board)
board["key"] = board["name"].map(nkey)
board = board[board["position"].isin(POS)].drop_duplicates("key").reset_index(drop=True)

order = pd.read_csv(A.order)
order["key"] = order["name"].map(nkey)
omap = order.drop_duplicates("key").set_index("key")["exp_pick"].to_dict()

adp_files = sorted(f for f in os.listdir(os.path.join(ROOT, "data/adp"))
                   if f.startswith("adp_ppr_2026_all_") and f.endswith(".csv"))
adp = pd.read_csv(os.path.join(ROOT, "data/adp", adp_files[-1]))
adp["key"] = adp["name"].map(nkey)
sdmap = adp.drop_duplicates("key").set_index("key")["stdev"].to_dict()

N = len(board)
NAME = board["name"].to_numpy()
KEY = board["key"].to_numpy()
PARR = board["position"].to_numpy()
PIDX = np.array([PI[p] for p in PARR])
VAL = board["final"].to_numpy(float)
KIDX = {k: i for i, k in enumerate(KEY)}

# outcome-model inputs, all taken from the board (nothing new is fitted here)
SDPOST = board["post_SD_bl"].to_numpy(float)
SDPOST = np.where(np.isnan(SDPOST), np.nanmedian(SDPOST), SDPOST)
AVAILP = board["avail"].to_numpy(float)
for q in POS:
    m = (PARR == q) & np.isnan(AVAILP)
    AVAILP[m] = np.nanmedian(board.loc[PARR == q, "avail"])
AVAILP = np.clip(np.where(np.isnan(AVAILP), 0.80, AVAILP), 0.30, 0.99)
S2T = board["sigma2_tier"].to_numpy(float)
for q in POS:                       # QB/TE have no tier variance on the board
    m = (PARR == q) & np.isnan(S2T)
    S2T[m] = np.nanmedian(S2T) if np.isnan(np.nanmedian(board.loc[PARR == q, "sigma2_tier"])) \
        else np.nanmedian(board.loc[PARR == q, "sigma2_tier"])
S2T = np.where(np.isnan(S2T), np.nanmedian(S2T), S2T)
SDGAME = np.sqrt(S2T)

adp_rank_fallback = board["adp"].rank(method="first").to_numpy(float)
MU = np.array([omap.get(k, np.nan) for k in KEY])
MU = np.where(np.isnan(MU), adp_rank_fallback, MU)
SD_FFC = np.array([sdmap.get(k, np.nan) for k in KEY])
SD_FFC = np.where(np.isnan(SD_FFC), np.nanmedian(SD_FFC), SD_FFC)
SIG = np.where(MU <= STATED_DEPTH, SIG_STATED, SD_FFC)

# The mock's moment is a mean |SLOT error|, and a slot is a RANK, not a latent draw: ranks
# are order statistics of a jointly-drawn vector, so the realised |rank error| is smaller
# than sigma*sqrt(2/pi).  Setting sigma = 1.40*sqrt(pi/2) therefore UNDER-disperses the
# simulated order (realised MAE 1.10 against the target 1.40).  Fix by matching the moment
# that was actually measured: scale sigma inside the stated block until the simulated mean
# |rank - mu| equals 1.40.  This is a moment match against an external calibration datum,
# declared before any survival or strategy output was read, not a fit to any result.
def _sim_mae(scale, nsim=400, seed=11):
    """mean |realised slot - stated slot| WITHIN a draft, then averaged over drafts --
    the same statistic the mock reported.  (Averaging ranks first and differencing after
    measures bias, not error, and is nearly zero for any symmetric noise.)"""
    r = np.random.default_rng(seed)
    sg = np.where(MU <= STATED_DEPTH, SIG_STATED * scale, SD_FFC)
    m = MU <= STATED_DEPTH
    tot = 0.0
    for _ in range(nsim):
        z = MU + r.standard_normal(N) * sg
        rk = np.argsort(np.argsort(z)) + 1
        tot += np.mean(np.abs(rk - MU)[m])
    return float(tot / nsim)
_lo, _hi = 0.5, 8.0
for _ in range(22):
    _mid = 0.5 * (_lo + _hi)
    if _sim_mae(_mid) < MAE_MOCK: _lo = _mid
    else: _hi = _mid
SCALE_STATED = 0.5 * (_lo + _hi)
SIG = np.where(MU <= STATED_DEPTH, SIG_STATED * SCALE_STATED, SD_FFC)
print(f"[order] sigma-scale solved so simulated mean |slot error| = {MAE_MOCK}: "
      f"scale={SCALE_STATED:.3f}, sigma_stated={SIG_STATED*SCALE_STATED:.3f} "
      f"(naive {SIG_STATED:.3f}); achieved MAE {_sim_mae(SCALE_STATED):.3f}")

print(f"[data] {A.board}: {N} players "
      f"({(PARR=='QB').sum()}QB/{(PARR=='RB').sum()}RB/{(PARR=='WR').sum()}WR/{(PARR=='TE').sum()}TE)")
print(f"[league] {TEAMS} teams, slot {SLOT}, {ROUNDS} board rounds; owner picks {MY}")
print(f"[order] sigma_stated={SIG_STATED:.3f}; median sigma_tail={np.median(SIG[MU>STATED_DEPTH]):.2f}")

# ===========================================================================
# L6.1  THE LINEUP OPERATOR, AND MARGINAL VALUE IN CLOSED FORM
# ===========================================================================
# Starting lineup 1QB/2RB/2WR/1TE/2FLEX(RB-WR)/1DST.  DST is identical across strategies
# and cancels, so it is dropped and the draft is ROUNDS rounds of QB/RB/WR/TE.
#
# Board values are VORP-scaled (PPG above positional replacement), so an unfilled slot is
# worth exactly 0 -- a freely-available replacement body fills it -- and a sub-replacement
# player is never started.  Hence a max(.,0) floor on every slot.  sec.R's operator had no
# floor; it makes no difference on a saturated roster but it does at an empty one, where
# without it a negative-value TE would appear to REDUCE lineup value.  Stated as a fix.
#
# CLOSED FORM FOR THE MARGINAL.  The slot structure is a transversal matroid (QB, TE,
# 2 RB, 2 WR, 2 FLEX open to RB/WR), so the optimal lineup is greedy and the value of
# adding one player of value v at position p is
#
#       Delta_p(v ; R) = max(0, v - c_p(R))                                    (W3.2)
#
# where c_p(R) is the DISPLACEMENT THRESHOLD: the weakest current lineup entrant that
# adding a p can cascade out, floored at 0.  For QB and TE that is the incumbent starter;
# for RB and WR it is the weakest of the six RB/WR-block entrants, because an added RB
# pushes an incumbent down the RB slots and into the flex, which is shared with WR.
# (W3.2) makes the marginal O(1) instead of an O(n log n) re-solve, which is what makes
# rollout and the flat-vs-step identity affordable.

def thresholds(cnt_vals):
    """cnt_vals: dict pos -> list of values on the roster.  Returns (c_QB, c_RB, c_TE, c_WR)."""
    qb = sorted(cnt_vals["QB"], reverse=True)
    te = sorted(cnt_vals["TE"], reverse=True)
    rb = sorted(cnt_vals["RB"], reverse=True)
    wr = sorted(cnt_vals["WR"], reverse=True)
    c_qb = max(qb[0], 0.0) if qb else 0.0
    c_te = max(te[0], 0.0) if te else 0.0
    block = rb[:2] + wr[:2]
    flexpool = sorted(rb[2:] + wr[2:], reverse=True)[:2]
    entrants = block + flexpool
    nslots = 6
    if len(entrants) < nslots:
        c_bl = 0.0                       # an empty RB/WR slot exists -> displaces nothing
    else:
        c_bl = max(min(entrants), 0.0)
    return np.array([c_qb, c_bl, c_te, c_bl])

def lineup_value(cnt_vals):
    qb = sorted(cnt_vals["QB"], reverse=True)
    te = sorted(cnt_vals["TE"], reverse=True)
    rb = sorted(cnt_vals["RB"], reverse=True)
    wr = sorted(cnt_vals["WR"], reverse=True)
    tot = max(qb[0], 0.0) if qb else 0.0
    tot += max(te[0], 0.0) if te else 0.0
    tot += sum(max(x, 0.0) for x in rb[:2]) + sum(max(x, 0.0) for x in wr[:2])
    tot += sum(max(x, 0.0) for x in sorted(rb[2:] + wr[2:], reverse=True)[:2])
    return tot

def marg_all(cnt_vals, ev):
    """vector of lineup-marginal values for every board player at roster cnt_vals."""
    c = thresholds(cnt_vals)
    return np.maximum(0.0, ev - c[PIDX])

# ---- roster legality (sec.M's rule, applied to the OWNER as well as opponents) ----
# "Draft the board" cannot mean "take the highest number 14 times" -- a roster that never
# takes a TE cannot field a lineup.  sec.M's S0 was best-available SUBJECT TO legality and
# the same constraint must bind here or the null strategy is a straw man.
MAXQB, MAXTE = 2, 2
def legal_pos(n, picks_left):
    """n = counts array [QB,RB,TE,WR]; returns boolean over the 4 positions."""
    ok = np.ones(4, bool)
    if n[0] >= MAXQB: ok[0] = False
    if n[2] >= MAXTE: ok[2] = False
    need_q = max(0, 1 - n[0]); need_t = max(0, 1 - n[2])
    need_r = max(0, 2 - n[1]); need_w = max(0, 2 - n[3])
    filled_flex = max(0, n[1] - 2) + max(0, n[3] - 2)
    need_f = max(0, 2 - filled_flex)
    need = need_q + need_t + need_r + need_w + need_f
    if need >= picks_left:                      # force-fill
        ok[:] = False
        if need_q: ok[0] = True
        if need_t: ok[2] = True
        if need_r or need_f: ok[1] = True
        if need_w or need_f: ok[3] = True
        if not ok.any(): ok[:] = True
    else:
        if n[0] >= 1 and need > 0: ok[0] = False
        if n[2] >= 1 and need > 0: ok[2] = False
    return ok

# ===========================================================================
# THE OUTCOME MODEL (the scorer) -- and why it is not the greedy's own objective
# ===========================================================================
# A prospective strategy comparison scored on E[lineup value under the board] is CIRCULAR:
# the lineup-marginal greedy maximises exactly that quantity at every step, so it wins by
# construction and the comparison is uninformative.  sec.M avoided this by scoring rosters
# on REALISED weekly points with a weekly-optimal lineup, which gives bench depth genuine
# value.  The prospective analogue, using only quantities already on the board:
#
#   season rate      X_j = final_j + post_SD_bl_j * z_j ,      z_j ~ N(0,1)
#   weekly presence  a_{j,w} ~ Bernoulli(avail_j)              (sec.A: availability is a
#                                                               measured, stable trait)
#   weekly lineup    each week, fill 1QB/2RB/2WR/1TE/2FLEX from the PRESENT roster,
#                    every slot floored at 0 (replacement is free)
#   score            mean over the 17 weeks, so numbers stay in sec.R's PPG units
#
# Bench depth is now worth something because starters miss weeks, and the greedy's
# objective (expected value of TODAY's best lineup) is no longer the scoring function.
# SENSITIVITY (--game-noise): add e_{j,w} ~ N(0, sigma2_tier_j) so week-to-week scoring
# noise also rewards depth.  It is reported separately because sigma2_tier is missing for
# QB and TE on the board and has to be imputed, so it is an assumption, not a measurement.

def draw_shocks(rng, game_noise):
    Z = rng.standard_normal(N)
    X = VAL + SDPOST * Z
    AV = rng.random((N, WEEKS)) < AVAILP[:, None]
    E = rng.standard_normal((N, WEEKS)) * SDGAME[:, None] if game_noise else None
    return X, AV, E

NEG = -1e9
def score_roster(idx, X, AV, E):
    """weekly-optimal starting lineup, averaged over WEEKS."""
    if len(idx) == 0:
        return 0.0
    W = np.repeat(X[idx][:, None], WEEKS, axis=1)
    if E is not None:
        W = W + E[idx]
    W = np.where(AV[idx], W, NEG)
    p = PIDX[idx]
    def blk(k):
        r = W[p == k]
        return -np.sort(-r, axis=0) if len(r) else np.zeros((0, WEEKS))
    qb, rb, te, wr = blk(0), blk(1), blk(2), blk(3)
    def take(Xb, n):
        got = Xb[:n] if len(Xb) >= n else np.vstack([Xb, np.full((n - len(Xb), WEEKS), NEG)])
        return got, (Xb[n:] if len(Xb) > n else Xb[:0])
    q1, _ = take(qb, 1); t1, _ = take(te, 1)
    r2, rrest = take(rb, 2); w2, wrest = take(wr, 2)
    fp = np.vstack([rrest, wrest]) if (len(rrest) + len(wrest)) else np.zeros((0, WEEKS))
    if len(fp): fp = -np.sort(-fp, axis=0)
    f2, _ = take(fp, 2)
    allst = np.vstack([q1, t1, r2, w2, f2])
    allst = np.maximum(allst, 0.0)          # replacement floor: never start below it
    return float(allst.sum() / WEEKS)

# ===========================================================================
# L7.1  OPPONENT ORDER MODEL + OWNER AVAILABILITY OVERRIDES
# ===========================================================================
# Thurstonian noisy-order model, the form pre-registered in EDA_PLAN9 L7.1:
#
#       z_j = mu_j + d_j + eps_j ,   eps_j ~ N(0, sigma_j^2),   opponents pick ascending z
#
# mu_j  = the owner's stated expected pick inside his 28-deep block; translated ADP beyond.
# sigma_j = SIG_STATED inside the block, fixed by the mock's mean |slot error| = 1.40 via
#           E|N(0,s)| = s*sqrt(2/pi); FFC across-draft slot SD beyond it.  Neither is tuned.
# d_j   = override shift, zero unless the owner has stated a belief about that player.
#
# WHY THE SHIFT AND NOT A TABLE EDIT.  Survival events are competing risks on one shared
# pool: exactly one player leaves per pick.  Setting S_London(25) := 0 in an output table
# leaves the pick that took him unspent and is not a draft.  So the owner's belief is
# imposed on the GENERATIVE model: solve for d_j such that the simulated marginal equals
# his stated one,
#
#       S_j(k ; d_j) = q_j                                                     (W3.1)
#
# S is monotone increasing in d (moving a player later in the queue can only raise his
# survival), so the root is unique and bisection converges; common random numbers are
# reused across the bisection so S(.;d) is deterministic in d.  Overrides interact --
# forcing London out early spends an opponent pick that would have taken somebody else --
# so the roots are cycled to a fixed point and the ACHIEVED marginals are reported next to
# the targets.
#
# REJECTED ALTERNATIVE: importance-weight paths by the indicator that all overrides hold.
# Unbiased, but with these five constraints under 2% of paths qualify and the ESS
# collapses.  The cost of the shift instead is that it matches the stated MARGINALS
# exactly while altering the model's joint dependence among overridden players.

def load_overrides(path):
    if not os.path.exists(path):
        print(f"[override] none at {path}")
        return pd.DataFrame(columns=["name", "pick", "p_available", "source", "key"])
    o = pd.read_csv(path)
    o["key"] = o["name"].map(nkey)
    miss = o[~o["key"].isin(KIDX)]
    if len(miss):
        print("[override] NOT ON BOARD, ignored:", list(miss["name"]))
    return o[o["key"].isin(KIDX)].reset_index(drop=True)

OV = load_overrides(A.overrides)

def opp_order_snapshots(eps, dshift, picks):
    """availability masks at the owner's picks, AFTER OPPONENT REMOVALS ONLY.

    The owner's own picks remove nobody here, deliberately: a contingency tree answers
    'who is on the board when I am on the clock', and he knows what he took.  Conditioning
    survival on his own hypothetical choices would change what his stated overrides mean.
    """
    z = MU + dshift + eps
    seq = np.argsort(z, kind="stable")
    taken = np.zeros(N, bool)
    ptr = 0
    out = {}
    want = set(picks)
    for t in range(1, NPICKS + 1):
        if t in MY:
            if t in want:
                out[t] = ~taken.copy()
            continue
        while ptr < N and taken[seq[ptr]]:
            ptr += 1
        if ptr >= N:
            continue
        taken[seq[ptr]] = True
    return out

def survival(eps_mat, dshift, picks):
    acc = {k: np.zeros(N) for k in picks}
    for s in range(eps_mat.shape[0]):
        snaps = opp_order_snapshots(eps_mat[s], dshift, picks)
        for k in picks:
            acc[k] += snaps[k]
    return {k: acc[k] / eps_mat.shape[0] for k in picks}

rng = np.random.default_rng(A.seed)
EPS_CAL = rng.standard_normal((A.nsim_cal, N)) * SIG

def calibrate_overrides(ov, n_cycles=5, tol=0.004):
    d = np.zeros(N)
    if len(ov) == 0:
        return d, pd.DataFrame()
    picks = sorted(set(int(x) for x in ov["pick"]))
    for cyc in range(n_cycles):
        moved = 0.0
        for _, r in ov.iterrows():
            j, k = KIDX[r["key"]], int(r["pick"])
            q = float(np.clip(r["p_available"], 0.002, 0.998))
            lo, hi, prev = -300.0, 300.0, d[j]
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                dt = d.copy(); dt[j] = mid
                if survival(EPS_CAL, dt, [k])[k][j] < q:
                    lo = mid
                else:
                    hi = mid
            d[j] = 0.5 * (lo + hi)
            moved = max(moved, abs(d[j] - prev))
        print(f"[override] cycle {cyc}: max |d| move {moved:.3f}")
        if moved < tol:
            break
    S = survival(EPS_CAL, d, picks)
    rows = [dict(name=NAME[KIDX[r["key"]]], position=PARR[KIDX[r["key"]]], pick=int(r["pick"]),
                 mu_base=MU[KIDX[r["key"]]], d_shift=d[KIDX[r["key"]]],
                 mu_shifted=MU[KIDX[r["key"]]] + d[KIDX[r["key"]]],
                 p_target=float(r["p_available"]),
                 p_achieved=float(S[int(r["pick"])][KIDX[r["key"]]]),
                 source=r.get("source", "")) for _, r in ov.iterrows()]
    return d, pd.DataFrame(rows)

picks_ov = sorted(set(int(x) for x in OV["pick"])) if len(OV) else []
S_FREE0 = survival(EPS_CAL, np.zeros(N), sorted(set(MY[:A.tree_picks]) | set(picks_ov)))
t0 = time.time()
DSHIFT, OVTAB = calibrate_overrides(OV)
if len(OVTAB):
    OVTAB["p_model_unconstrained"] = [S_FREE0[int(r.pick)][KIDX[nkey(r.name)]] for r in OVTAB.itertuples()]
    OVTAB = OVTAB[["name", "position", "pick", "mu_base", "p_model_unconstrained",
                   "p_target", "p_achieved", "d_shift", "mu_shifted", "source"]]
    print(f"[override] calibrated in {time.time()-t0:.0f}s")
    print(OVTAB.drop(columns=["source"]).round(3).to_string(index=False))
    OVTAB.to_csv(os.path.join(RES, f"{A.prefix}_overrides.csv"), index=False)

EPS = rng.standard_normal((A.nsim, N)) * SIG
SURV = survival(EPS, DSHIFT, MY)
SURV_FREE = survival(EPS, np.zeros(N), MY)
sc = pd.DataFrame({"name": NAME, "position": PARR, "value": VAL, "mu": MU, "sigma": SIG,
                   "d_shift": DSHIFT})
for k in MY:
    sc[f"S_{k}"] = SURV[k]; sc[f"Sfree_{k}"] = SURV_FREE[k]
sc.sort_values("value", ascending=False).to_csv(os.path.join(RES, f"{A.prefix}_survival.csv"), index=False)

# noise diagnostic -- does the order model reproduce the dispersion it was calibrated to?
def slot_dispersion(nsim=800):
    S = np.zeros((nsim, N))
    for s in range(nsim):
        z = MU + DSHIFT + rng.standard_normal(N) * SIG
        S[s] = np.argsort(np.argsort(z)) + 1
    return S.mean(0), S.std(0, ddof=1)
SIM_MU, SIM_SD = slot_dispersion()
mst = MU <= STATED_DEPTH
mtl = (MU > STATED_DEPTH) & (MU <= 140)
DIAG_NOISE = dict(sim_mae_stated=_sim_mae(SCALE_STATED, nsim=400, seed=999),
                  sim_bias_stated=float(np.mean(np.abs(SIM_MU[mst] - MU[mst]))),
                  sim_sd_stated=float(SIM_SD[mst].mean()), target_sd_stated=float(SIG_STATED),
                  sim_sd_tail=float(SIM_SD[mtl].mean()), ffc_sd_tail=float(SD_FFC[mtl].mean()),
                  corr_sd_tail=float(np.corrcoef(SIM_SD[mtl], SD_FFC[mtl])[0, 1]))
print("[noise]", {k: round(v, 3) for k, v in DIAG_NOISE.items()})

# ===========================================================================
# POLICIES
# ===========================================================================
def eval_board(drb):
    ev = VAL.copy(); ev[PARR == "RB"] += drb
    return ev
EV0 = VAL.copy()
EVD = eval_board(A.delta_rb)
# sec.M's null S0 is "best available by ADP", not by our board.  Reproduce it exactly:
# a strictly ADP-ordered evaluation (subject to the same roster-legality rule).
EV_ADP = -board["adp"].rank(method="first").to_numpy(float)

def pol_board(mask, held, cnt, ev, picks_left):
    ok = legal_pos(cnt, picks_left)
    m = mask & ok[PIDX]
    if not m.any(): m = mask
    idx = np.flatnonzero(m)
    return int(idx[np.argmax(ev[idx])]) if len(idx) else None

def pol_marginal(mask, held, cnt, ev, picks_left):
    ok = legal_pos(cnt, picks_left)
    m = mask & ok[PIDX]
    if not m.any(): m = mask
    idx = np.flatnonzero(m)
    if not len(idx): return None
    mg = marg_all(held, ev)[idx]
    return int(idx[np.argmax(mg + 1e-4 * ev[idx])])

def run_draft(eps, policy, ev, first_forced=None, ban=frozenset(), qb_not_before=0,
              record_states=None, min_rb_first6=0):
    z = MU + DSHIFT + eps
    seq = np.argsort(z, kind="stable")
    taken = np.zeros(N, bool)
    for b in ban: taken[b] = True
    mask = ~taken
    ptr = 0
    held = {q: [] for q in POS}
    cnt = np.zeros(4, int)
    mine = []
    for t in range(1, NPICKS + 1):
        if t in MY:
            i = MY.index(t)
            if record_states is not None and i in record_states:
                record_states[i].append(({q: list(held[q]) for q in POS}, mask.copy(),
                                         cnt.copy(), list(mine)))
            m = mask.copy()
            if qb_not_before and (i + 1) < qb_not_before:
                m2 = m & (PARR != "QB")
                if m2.any(): m = m2
            if first_forced is not None and t == MY[0]:
                m2 = m & (PARR == first_forced)
                if m2.any(): m = m2
            if min_rb_first6 and i < 6:
                short = min_rb_first6 - cnt[PI["RB"]]
                if short > 0 and (6 - i) <= short:
                    m2 = m & (PARR == "RB")
                    if m2.any(): m = m2
            k = policy(m, held, cnt, ev, ROUNDS - i)
            if k is None: continue
            taken[k] = True; mask[k] = False
            held[PARR[k]].append(ev[k]); cnt[PIDX[k]] += 1
            mine.append(k)
            continue
        while ptr < N and taken[seq[ptr]]: ptr += 1
        if ptr >= N: continue
        taken[seq[ptr]] = True; mask[seq[ptr]] = False
    return mine

# ===========================================================================
# L6.1 (cont.)  RAW vs MYOPIC-MARGINAL vs ROLLOUT, AT EVERY PICK
# ===========================================================================
# The myopic marginal Delta_j(R) is one step.  The decision-theoretic object is
#
#   V_t(R) = max_j E[ score(final roster) | take j now, play policy pi thereafter ]  (W3.3)
#
# approximated by ROLLOUT: take j, then play myopic-greedy to the end against the same
# simulated opponents, score with the outcome model above, average.  Rollout is one step of
# policy improvement, so in expectation it is no worse than the policy it rolls out.
# Reported alongside raw and myopic so the SIZE of each correction is visible.

def l61_table(nstate=60, nroll=40, game_noise=False):
    npk = A.tree_picks
    states = {i: [] for i in range(npk)}
    epsm = rng.standard_normal((max(nstate, 1), N)) * SIG
    for s in range(nstate):
        run_draft(epsm[s], pol_board, EV0, record_states=states)
    rows = []
    for i in range(npk):
        t = MY[i]
        samp = states[i][:nroll]
        for q in POS:
            raw, myo, roll, base = [], [], [], []
            for si, (held, mask, cnt, pre) in enumerate(samp):
                qi = np.flatnonzero(mask & (PARR == q))
                if len(qi) == 0: continue
                kb = int(qi[np.argmax(EV0[qi])])
                raw.append(EV0[kb])
                myo.append(float(marg_all(held, EV0)[kb]))
                # rollout: forced first pick kb at this state, then myopic to the end
                rr = np.random.default_rng(A.seed + 7919 * i + si)
                e2 = rr.standard_normal(N) * SIG
                X, AV, E = draw_shocks(rr, game_noise)
                mine_w = _rollout(mask, held, cnt, i, kb, e2, EV0)
                mine_wo = _rollout(mask, held, cnt, i, None, e2, EV0)
                roll.append(score_roster(np.array(pre + mine_w, int), X, AV, E))
                base.append(score_roster(np.array(pre + mine_wo, int), X, AV, E))
            rows.append(dict(pick=t, round=i + 1, position=q, n=len(raw),
                             raw_best=np.mean(raw) if raw else np.nan,
                             myopic_marginal=np.mean(myo) if myo else np.nan,
                             rollout_gain=(np.mean(np.array(roll) - np.array(base))
                                           if roll else np.nan),
                             rollout_se=(np.std(np.array(roll) - np.array(base), ddof=1) /
                                         np.sqrt(len(roll)) if len(roll) > 1 else np.nan)))
    return pd.DataFrame(rows)

def _rollout(mask0, held0, cnt0, i0, force_k, eps, ev):
    """continue the draft from owner pick i0 (optionally forcing force_k), myopic after."""
    z = MU + DSHIFT + eps
    seq = np.argsort(z, kind="stable")
    mask = mask0.copy()
    held = {q: list(held0[q]) for q in POS}
    cnt = cnt0.copy()
    mine = []
    taken = ~mask
    ptr = 0
    start = MY[i0]
    for t in range(start, NPICKS + 1):
        if t in MY:
            i = MY.index(t)
            if t == start and force_k is not None:
                k = force_k
            else:
                k = pol_marginal(mask, held, cnt, ev, ROUNDS - i)
            if k is None: continue
            taken[k] = True; mask[k] = False
            held[PARR[k]].append(ev[k]); cnt[PIDX[k]] += 1
            mine.append(k)
            continue
        while ptr < N and taken[seq[ptr]]: ptr += 1
        if ptr >= N: continue
        taken[seq[ptr]] = True; mask[seq[ptr]] = False
    return mine

print("\n[L6.1] raw / myopic-marginal / rollout, by pick and position ...")
L61 = l61_table()
L61.to_csv(os.path.join(RES, f"{A.prefix}_lineup_marginal.csv"), index=False)
print(L61.round(3).to_string(index=False))

# ===========================================================================
# L6.2  THE FLAT-VERSUS-STEP PRINCIPLE, FORMALISED
# ===========================================================================
# At the owner's pick t0 with next pick t1, order the AVAILABLE position-p players by
# lineup-marginal value m_(1) >= m_(2) >= ... and write the tier gaps
#       delta_i = m_(i) - m_(i+1) >= 0.
# Let N_p = # of position-p players removed between t0 and t1 (the drain).  The cost of
# waiting is exactly lineup-VONA,
#       W_p = m_(1) - E[ max_{j available at t1, pos=p} m_j ].                  (W3.3)
# If the room removes p-players from the top of the p-list downward, the best p at t1 is
# m_(1+N_p), and telescoping,   m_(1) - m_(1+N_p) = sum_{i<=N_p} delta_i,  so
#
#       W_p = sum_{i>=1} delta_i * Pr(N_p >= i).                                (W3.4)
#
# THE COST OF WAITING IS THE SURVIVAL-WEIGHTED SUM OF THE TIER GAPS.  Decay enters only
# through Pr(N_p >= i), which lives in [0,1] and is therefore BOUNDED.  Steps enter
# linearly and are UNBOUNDED.  Two tight corollaries:
#     flat:  delta_i <= e for all reachable i   =>   W_p <= e * E[N_p]
#     step:  W_p >= delta_1 * Pr(N_p >= 1)
# So the wait/take decision is FIRST-ORDER in step size and only BOUNDED in decay rate.
# (W3.4) holds exactly under order consistency; real rooms violate it (sec.R: the max-value
# available player is taken 10.3% of the time), so both sides are computed from the same
# simulation and the residual is reported rather than assumed away.

def flat_vs_step(nsim=1200, depth=10):
    rows = []
    epsm = rng.standard_normal((nsim, N)) * SIG
    for i in range(len(MY) - 1):
        t0, t1 = MY[i], MY[i + 1]
        acc = {q: dict(m1=[], W=[], Nd=[], gaps=np.zeros(depth), pge=np.zeros(depth),
                       ident=0.0, n=0) for q in POS}
        for s in range(nsim):
            z = MU + DSHIFT + epsm[s]
            seq = np.argsort(z, kind="stable")
            taken = np.zeros(N, bool); mask = ~taken; ptr = 0
            held = {q: [] for q in POS}; cnt = np.zeros(4, int)
            snap = None
            for t in range(1, t1 + 1):
                if t in MY:
                    ii = MY.index(t)
                    if t == t0:
                        # snapshot, then the owner takes NOTHING: W_p is the cost of
                        # WAITING on p, so his own removal must not be counted in the
                        # drain.  From t0 on he is a spectator for this measurement.
                        snap = ({q: list(held[q]) for q in POS}, mask.copy())
                        continue
                    if t > t0:
                        continue
                    k = pol_board(mask, held, cnt, EV0, ROUNDS - ii)
                    if k is None: continue
                    taken[k] = True; mask[k] = False
                    held[PARR[k]].append(EV0[k]); cnt[PIDX[k]] += 1
                    continue
                while ptr < N and taken[seq[ptr]]: ptr += 1
                if ptr >= N: continue
                taken[seq[ptr]] = True; mask[seq[ptr]] = False
            held0, mask0 = snap
            mg = marg_all(held0, EV0)          # marginals w.r.t. the SAME roster state
            for q in POS:
                sel0 = mask0 & (PARR == q)
                if not sel0.any(): continue
                mv = np.sort(mg[sel0])[::-1]
                mv = np.concatenate([mv, np.zeros(depth + 1)])
                acc[q]["m1"].append(mv[0])
                gj = mv[:depth] - mv[1:depth + 1]
                acc[q]["gaps"] += gj
                sel1 = mask & (PARR == q)
                acc[q]["W"].append(mg[sel1].max() if sel1.any() else 0.0)
                nd = int((sel0 & ~mask).sum())
                acc[q]["Nd"].append(nd)
                ind = (nd >= np.arange(1, depth + 1))
                acc[q]["pge"] += ind
                # (W3.4) evaluated INSIDE the sim so Cov(delta_i, 1[N>=i]) is retained;
                # averaging the two factors separately would drop it and the residual
                # would no longer isolate order violation.
                acc[q]["ident"] += float(np.sum(gj * ind))
                acc[q]["n"] += 1
        for q in POS:
            a = acc[q]
            if a["n"] == 0: continue
            gaps = a["gaps"] / a["n"]; prob = a["pge"] / a["n"]
            m1 = float(np.mean(a["m1"])); En = float(np.mean(a["Nd"]))
            W = m1 - float(np.mean(a["W"]))
            rows.append(dict(from_pick=t0, to_pick=t1, position=q,
                             best_marginal_now=m1,
                             delta1=float(gaps[0]), delta2=float(gaps[1]),
                             mean_gap_top4=float(gaps[:4].mean()),
                             E_drain=En, drain_rate=En / (t1 - t0), p_ge1=float(prob[0]),
                             W_cost_of_waiting=W,
                             W_identity=a["ident"] / a["n"],
                             W_identity_naive=float(np.sum(gaps * prob)),
                             residual=W - a["ident"] / a["n"],
                             se_W=float(np.std(a["W"], ddof=1) / np.sqrt(a["n"]))))
    return pd.DataFrame(rows)

print("\n[L6.2] flat-versus-step ...")
FVS = flat_vs_step()
FVS.to_csv(os.path.join(RES, f"{A.prefix}_flat_vs_step.csv"), index=False)
print(FVS.round(3).to_string(index=False))

import numpy.linalg as la
def demo_regression(df):
    d = df.dropna(subset=["W_cost_of_waiting", "delta1", "E_drain"])
    y = d["W_cost_of_waiting"].to_numpy(); out = {}
    for spec, cols in [("step_only", ["delta1"]), ("decay_only", ["E_drain"]),
                       ("both", ["delta1", "E_drain"])]:
        X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy() for c in cols])
        bh, *_ = la.lstsq(X, y, rcond=None)
        e = y - X @ bh
        s2 = e @ e / max(len(d) - X.shape[1], 1)
        se = np.sqrt(np.diag(s2 * la.inv(X.T @ X)))
        out[spec] = dict(n=len(d), r2=round(float(1 - e.var() / y.var()), 3),
                         coef={c: round(float(v), 4) for c, v in zip(["const"] + cols, bh)},
                         se={c: round(float(v), 4) for c, v in zip(["const"] + cols, se)})
    return out
DEMO = demo_regression(FVS)
print("[L6.2] cost-of-waiting regressions:", json.dumps(DEMO))

# ===========================================================================
# L7.2  CONTINGENCY TREES
# ===========================================================================
# The owner's policy is deterministic given availability, so the branch structure is a
# decision LIST: rank candidates by his policy value; branch i fires when 1..i-1 are gone
# and i is there.  That is a plan he can carry, not a simulation summary.

def contingency(policy, ev, label, nsim=None, npk=None):
    nsim = nsim or A.nsim_strat
    npk = npk or A.tree_picks
    epsm = rng.standard_normal((nsim, N)) * SIG
    got = {i: [] for i in range(npk)}
    av = {i: np.zeros(N) for i in range(npk)}
    cond = {}
    for s in range(nsim):
        z = MU + DSHIFT + epsm[s]
        seq = np.argsort(z, kind="stable")
        taken = np.zeros(N, bool); mask = ~taken; ptr = 0
        held = {q: [] for q in POS}; cnt = np.zeros(4, int); first = None
        for t in range(1, MY[npk - 1] + 1):
            if t in MY:
                i = MY.index(t); av[i] += mask
                k = policy(mask, held, cnt, ev, ROUNDS - i)
                if k is None: continue
                got[i].append(k)
                if i == 0: first = k
                if i == 1: cond.setdefault(first, []).append(k)
                taken[k] = True; mask[k] = False
                held[PARR[k]].append(ev[k]); cnt[PIDX[k]] += 1
                continue
            while ptr < N and taken[seq[ptr]]: ptr += 1
            if ptr >= N: continue
            taken[seq[ptr]] = True; mask[seq[ptr]] = False
    rows = []
    for i in range(npk):
        vc = pd.Series(got[i]).value_counts(normalize=True); cum = 0.0
        for k, p in vc.items():
            cum += p
            rows.append(dict(policy=label, pick=MY[i], round=i + 1, name=NAME[k],
                             position=PARR[k], board_value=VAL[k], eval_value=ev[k],
                             p_branch=float(p), p_cumulative=float(cum),
                             p_available=float(av[i][k] / nsim), nsim=nsim))
    cnd = []
    for k1, lst in cond.items():
        vc = pd.Series(lst).value_counts(normalize=True)
        for k2, p in vc.items():
            cnd.append(dict(policy=label, first=NAME[k1], then=NAME[k2], then_pos=PARR[k2],
                            p_given_first=float(p), n_first=len(lst)))
    return pd.DataFrame(rows), pd.DataFrame(cnd)

POLICIES = {"board": (pol_board, EV0), "board+dRB": (pol_board, EVD),
            "lineup_marginal": (pol_marginal, EV0), "lineup_marginal+dRB": (pol_marginal, EVD)}
TR, CD = zip(*[contingency(p, e, lab) for lab, (p, e) in POLICIES.items()])
TREE = pd.concat(TR, ignore_index=True); COND = pd.concat(CD, ignore_index=True)
TREE.to_csv(os.path.join(RES, f"{A.prefix}_contingency.csv"), index=False)
COND.to_csv(os.path.join(RES, f"{A.prefix}_contingency_conditional.csv"), index=False)
print("\n[L7.2] contingency, policy = board+dRB (branches >= 2%):")
print(TREE[(TREE.policy == "board+dRB") & (TREE.p_branch >= 0.02)].round(3).to_string(index=False))

# ===========================================================================
# L7.3  STRATEGY COMPARISON (prospective arm)
# ===========================================================================
# Common random numbers: every strategy sees the SAME opponent-order draws AND the same
# player-level outcome shocks, so the paired difference against S0 is far less noisy than
# the difference of two means.  The reported SE is the SE of that paired difference.
#
# SCOPE OF THE SE -- stated because it is easy to over-read.  It is Monte-Carlo error of
# ONE model.  It contains outcome uncertainty (board posterior SD and availability) and
# opponent-order uncertainty, but NOT: board bias, error in the fitted opponent model
# (n = 78 choices, one draft), or error in the owner's stated order.  It is not a sampling
# distribution over drafts that could happen; it is one over runs of this model.

def strategies():
    # S0 is sec.M's null: best available by ADP subject to roster legality.
    S = {"S0_adp": dict(policy=pol_board, ev=EV_ADP),
         "S1_board": dict(policy=pol_board, ev=EV0),
         "S2_board_dRB": dict(policy=pol_board, ev=EVD),
         "S3_lineup_marginal": dict(policy=pol_marginal, ev=EV0),
         "S4_lineup_marginal_dRB": dict(policy=pol_marginal, ev=EVD)}
    for q in POS:
        S[f"S5_first_{q}"] = dict(policy=pol_board, ev=EV0, first_forced=q)
    S["S6_noQB_before_R8"] = dict(policy=pol_board, ev=EV0, qb_not_before=8)
    S["S7_noQB_before_R12"] = dict(policy=pol_board, ev=EV0, qb_not_before=12)
    for k in (2, 3, 4, 5):
        S[f"S8_minRB{k}_in6"] = dict(policy=pol_board, ev=EV0, min_rb_first6=k)
        S[f"S9_minRB{k}_marg"] = dict(policy=pol_marginal, ev=EV0, min_rb_first6=k)
    return S

def evaluate(strats, nsim=None, ban=frozenset(), tag="", game_noise=False, seed=None):
    nsim = nsim or A.nsim_strat
    rr = np.random.default_rng(seed if seed is not None else A.seed + 99)
    tot = {k: np.zeros(nsim) for k in strats}
    mixc = {k: np.zeros(4) for k in strats}
    for s in range(nsim):
        eps = rr.standard_normal(N) * SIG
        X, AV, E = draw_shocks(rr, game_noise)
        for k, cfg in strats.items():
            mine = run_draft(eps, cfg["policy"], cfg["ev"],
                             first_forced=cfg.get("first_forced"), ban=ban,
                             qb_not_before=cfg.get("qb_not_before", 0),
                             min_rb_first6=cfg.get("min_rb_first6", 0))
            # SCORED ON THE TRUE BOARD, never on the tilted evaluation board: delta_RB is a
            # preference used to CHOOSE, not a claim about points.
            tot[k][s] = score_roster(np.array(mine, int), X, AV, E)
            mixc[k] += np.bincount(PIDX[np.array(mine, int)], minlength=4)
    base = tot[list(strats)[0]]
    rows = []
    for k in strats:
        d = tot[k] - base
        se = d.std(ddof=1) / np.sqrt(nsim)
        rows.append(dict(tag=tag, game_noise=game_noise, strategy=k,
                         E_lineup_ppg=tot[k].mean(), sd_outcome=tot[k].std(ddof=1),
                         se_level=tot[k].std(ddof=1) / np.sqrt(nsim),
                         delta_vs_S0=d.mean(), se_paired=se,
                         t=d.mean() / se if se > 0 else 0.0,
                         p_beats_S0=float((d > 0).mean()),
                         mix=",".join(f"{q}{mixc[k][PI[q]]/nsim:.1f}" for q in POS), nsim=nsim))
    return pd.DataFrame(rows), tot

print("\n[L7.3] strategy comparison (prospective, availability-only scorer) ...")
STR, TOT = evaluate(strategies(), tag="base", game_noise=False)
print(STR.round(3).to_string(index=False))
print("\n[L7.3] sensitivity: + week-to-week scoring noise (sigma2_tier, imputed at QB/TE)")
STRG, _ = evaluate(strategies(), tag="base", game_noise=True, nsim=max(600, A.nsim_strat // 2))
print(STRG.round(3).to_string(index=False))

CF = []
for cn in ["Caleb Williams"]:
    if nkey(cn) not in KIDX: continue
    s, _ = evaluate({k: v for k, v in strategies().items()
                     if k in ("S0_adp", "S1_board", "S2_board_dRB",
                              "S4_lineup_marginal_dRB", "S6_noQB_before_R8")},
                    nsim=max(600, A.nsim_strat // 2), ban=frozenset([KIDX[nkey(cn)]]),
                    tag=f"ban:{cn}")
    CF.append(s)
if CF:
    CFD = pd.concat(CF, ignore_index=True)
    print("\n[L7.3] counterfactual -- player removed from the universe entirely:")
    print(CFD.round(3).to_string(index=False))
else:
    CFD = pd.DataFrame()

SENS = []
for drb in [0.0, 0.5, 1.0, 1.29, 1.345, 1.40, 1.75, 2.0, 2.5, 3.0]:
    s, _ = evaluate({"S1_board": dict(policy=pol_board, ev=EV0),
                     "S2_board_dRB": dict(policy=pol_board, ev=eval_board(drb))},
                    nsim=max(800, A.nsim_strat // 2), tag=f"dRB={drb}")
    SENS.append(s[s.strategy == "S2_board_dRB"].assign(delta_rb=drb))
SENSD = pd.concat(SENS, ignore_index=True)
SENSD.to_csv(os.path.join(RES, f"{A.prefix}_delta_rb_sensitivity.csv"), index=False)
print("\n[L7.3] delta_RB sensitivity (vs S0 on the same CRN):")
print(SENSD[["delta_rb", "E_lineup_ppg", "delta_vs_S0", "se_paired", "t"]].round(3).to_string(index=False))

pd.concat([STR, STRG] + ([CFD] if len(CFD) else []), ignore_index=True) \
  .to_csv(os.path.join(RES, f"{A.prefix}_strategy.csv"), index=False)
json.dump(dict(noise=DIAG_NOISE, demo_regression=DEMO, my_picks=MY, slot=SLOT, teams=TEAMS,
               rounds=ROUNDS, delta_rb=A.delta_rb, board=A.board, nsim=A.nsim,
               nsim_strat=A.nsim_strat, sig_stated=SIG_STATED, stated_depth=STATED_DEPTH),
          open(os.path.join(RES, f"{A.prefix}_params.json"), "w"), indent=1)
print(f"\n[out] {A.prefix}_{{survival,overrides,lineup_marginal,flat_vs_step,contingency,"
      f"contingency_conditional,strategy,delta_rb_sensitivity,params}}")

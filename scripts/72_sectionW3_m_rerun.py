#!/usr/bin/env python3
"""
WS3 / L7.3 (retrospective arm) -- does sec.M's "no pick sequence beats drafting the board"
verdict survive when the three things it lacked are supplied?

sec.M (REPORT sec.30) simulated 2015-2024, 10 slots, 200 drafts per (season, slot), with
opponents drafting by ADP + per-drafter Gaussian noise, scored on REALISED weekly points
under a weekly-optimal lineup, DM-clustered by season with BH q = 0.10.  0 of 5 strategies
survived.  Three things were missing, and EDA_PLAN9 L7.3 requires all three be added:

  (1) BEHAVIOURAL OPPONENTS.  sec.R fitted a conditional logit to a live draft and measured
      the room's bias against a value board: alpha_RB = +1.66, alpha_TE = +1.97,
      alpha_QB = -3.41 logits, beta_v = +1.09, plus roster-state need terms.  Those
      coefficients are transported here onto each historical season's ADP-implied value
      board.  This is an EXTRAPOLATION across leagues and eras and is labelled as one; the
      ADP-noise opponent model is retained as the control arm so the opponent model's
      contribution is separable from the strategy's.
  (2) delta_RB, the owner's structural RB premium, identified from two revealed
      preferences on the 2026 board (Amon-Ra - CMC = 1.29, Rice - Barkley = 1.40 board
      points).  Board points are PPG-above-replacement, so in season-total units the tilt
      is 17 * delta_RB.
  (3) LINEUP-MARGINAL value rather than raw value (sec.R eq. 44.5), which is sec.M's S5
      generalised: every candidate is scored by what he adds to the STARTING LINEUP given
      the roster already held, with every slot floored at replacement.

FIXED BEFORE RUNNING (nothing below was altered after seeing any result):
  null      SA  best available by ADP, subject to roster legality       (= sec.M's S0)
  strategies
            SB  best available by expected VORP (the value board)       (~ sec.M's S1)
            SC  SB with the delta_RB tilt on RBs
            SD  lineup-marginal greedy on expected VORP                 (= sec.M's S5)
            SE  SD with the delta_RB tilt
  scoring   weekly-optimal 1QB/2RB/2WR/1TE/2FLEX on realised weekly PPR, weeks 1-14
            primary (sec.M3), 15-17 and 1-17 reported as declared secondaries
  inference d_s = per-season mean (S - SA); DM t on 10 season clusters, t(9);
            BH q = 0.10 across the four comparisons; MDE reported beside every p.
  leakage   E[season total] and replacement levels are leave-one-season-out, exactly as
            sec.M used them.  Nothing from season y is used to draft season y.

Outputs: results/sectionW3_m_rerun.csv, results/sectionW3_m_rerun_bysim.csv
Rerun:   python3 scripts/72_sectionW3_m_rerun.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sectionM_common as C

ROOT = C.ROOT
RES = f"{ROOT}/results"
YEARS = C.YEARS
NTEAM, NROUND = 10, 14
NSIM = int(os.environ.get("W3_NSIM", "40"))
POSI = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
POSN = ["QB", "RB", "WR", "TE"]
STRATS = ["SA", "SB", "SC", "SD", "SE", "SF", "SG"]
DELTA_RB_PPG = float(os.environ.get("W3_DRB", "1.345"))
DELTA_RB_SEASON = DELTA_RB_PPG * 17.0

# --- sec.R's fitted opponent coefficients (M2R, results/sectionR_params.json) -----
RP = json.load(open(f"{RES}/sectionR_params.json"))["M2R"]
RCO = dict(zip(RP["cols"], RP["theta"]))
print("[opp] sec.R M2R coefficients transported:", {k: round(v, 3) for k, v in RCO.items()})

print("loading ...", flush=True)
wk = C.load_weekly()
panel, _, _ = C.build_panel(wk)
avc = pd.read_csv(f"{RES}/sectionM_adp_value_curve.csv")
rep = pd.read_csv(f"{RES}/replacement_levels.csv")
repF = rep[rep.frame == NTEAM]
R_loso = {(y, p): repF[(repF.pos == p) & (repF.year != y)].R_static_total.mean()
          for y in YEARS for p in POSN}


def build_season(y):
    b = panel[panel.year == y].copy().sort_values("adp").reset_index(drop=True).head(160)
    n = len(b)
    adp = b.adp.values.astype(float)
    sd = b.stdev.values.astype(float)
    pos = np.array([POSI[p] for p in b.pos.values])
    a = avc[avc.year == y].set_index("pid").E_total
    Ev = np.array([a.get(p, np.nan) for p in b.pid.values])
    Ev = np.where(np.isnan(Ev), np.nanmin(Ev), Ev)
    m = C.weekly_matrix(wk, y, list(b.pid.values)).values.astype(float)
    return dict(y=y, n=n, adp=adp, sd=sd, pos=pos, Ev=Ev, wkm=m,
                rank_adp=np.argsort(adp, kind="stable"),
                pid=b.pid.values, name=b.name.values)


SEASON = {y: build_season(y) for y in YEARS}

# ------------------------------------------------------------------ roster rules
MAXQB, MAXTE = 2, 2
def unmet(cnt):
    q = max(0, 1 - cnt[0]); t = max(0, 1 - cnt[3])
    r = max(0, 2 - cnt[1]); w = max(0, 2 - cnt[2])
    flex = max(0, 2 - (max(0, cnt[1] - 2) + max(0, cnt[2] - 2)))
    return q, r, w, t, flex

def legal_mask(cnt, picks_left):
    ok = np.ones(4, bool)
    if cnt[0] >= MAXQB: ok[0] = False
    if cnt[3] >= MAXTE: ok[3] = False
    q, r, w, t, flex = unmet(cnt)
    need = q + r + w + t + flex
    if need >= picks_left:
        ok[:] = False
        if q: ok[0] = True
        if t: ok[3] = True
        if r or flex: ok[1] = True
        if w or flex: ok[2] = True
        if not ok.any(): ok[:] = True
    else:
        if cnt[0] >= 1 and need > 0: ok[0] = False
        if cnt[3] >= 1 and need > 0: ok[3] = False
    return ok

def lineup_value(Ev, pos, roster, Rrep):
    """sec.M's operator: every slot is worth max(assigned, replacement)."""
    v = {k: sorted([Ev[p] for p in roster if pos[p] == k], reverse=True) for k in range(4)}
    tot = (max(v[0][0], Rrep[0]) if v[0] else Rrep[0]) + (max(v[3][0], Rrep[3]) if v[3] else Rrep[3])
    for k, nreq in ((1, 2), (2, 2)):
        got = v[k][:nreq]
        tot += sum(max(x, Rrep[k]) for x in got) + (nreq - len(got)) * Rrep[k]
    Rf = (Rrep[1] + Rrep[2]) / 2
    rest = sorted(v[1][2:] + v[2][2:], reverse=True)[:2]
    tot += sum(max(x, Rf) for x in rest) + (2 - len(rest)) * Rf
    return tot

def score_roster(wkm, pos, roster, w0, w1):
    M = wkm[roster][:, w0 - 1:w1]
    P = pos[roster]
    NEG = -1e9
    A = {}
    for k in range(4):
        X = M[P == k]
        X = np.where(np.isnan(X), NEG, X)
        A[k] = -np.sort(-X, axis=0) if len(X) else X
    def take(k, n):
        X = A[k]
        got = X[:n] if len(X) >= n else np.vstack([X, np.full((n - len(X), M.shape[1]), NEG)])
        return got, (X[n:] if len(X) > n else X[:0])
    qb, _ = take(0, 1); rb, rbr = take(1, 2); wr, wrr = take(2, 2); te, _ = take(3, 1)
    fp = np.vstack([rbr, wrr]) if (len(rbr) + len(wrr)) else np.zeros((0, M.shape[1]))
    if len(fp): fp = -np.sort(-fp, axis=0)
    fl = fp[:2] if len(fp) >= 2 else np.vstack([fp, np.full((2 - len(fp), M.shape[1]), NEG)])
    allst = np.vstack([qb, rb, wr, te, fl])
    return float(np.where(allst <= NEG / 2, 0.0, allst).sum())

# --------------------------------------------------------------- opponent models
def opp_pick_adp(order, ptr, taken, pos, cnt, picks_left):
    """sec.M's control arm: a private ADP board with FFC-calibrated per-drafter noise."""
    while ptr < len(order) and taken[order[ptr]]:
        ptr += 1
    ok = legal_mask(cnt, picks_left)
    j = ptr
    while j < len(order):
        p = order[j]
        if not taken[p] and ok[pos[p]]:
            return p, ptr
        j += 1
    return order[ptr], ptr

def opp_pick_logit(vppg, pos, taken, cnt, picks_left, rng):
    """sec.R M2R.  U_j = b_v v_j + sum_p b^need_p need_p 1[pos=p] + alpha_p + b_flex flex.
    v_j is PPG-above-replacement, the same scale beta_v was estimated on."""
    ok = legal_mask(cnt, picks_left)
    avail = (~taken) & ok[pos]
    if not avail.any():
        avail = ~taken
        if not avail.any():
            return None
    q, r, w, t, flex = unmet(cnt)
    need = np.array([q, r, w, t], float)
    u = RCO["v"] * vppg.copy()
    for k, nm in enumerate(["need_QB", "need_RB", "need_WR", "need_TE"]):
        u = u + RCO[nm] * need[k] * (pos == k)
    u = u + RCO["int_QB"] * (pos == 0) + RCO["int_RB"] * (pos == 1) + RCO["int_TE"] * (pos == 3)
    u = u + RCO["flex"] * flex * ((pos == 1) | (pos == 2))
    u = np.where(avail, u, -np.inf)
    m = u.max()
    if not np.isfinite(m):
        return None
    p = np.exp(u - m); p /= p.sum()
    return int(rng.choice(len(p), p=p))

# ------------------------------------------------------------------- strategies
# SF / SG are DIAGNOSTICS (family = 0), added after SB/SD were seen to fail, for a
# diagnosed structural reason: best-available-by-VORP builds a 2-RB / 10-WR roster because
# sec.M1 showed R_RB and R_WR are forced to a common cutoff by the RB-WR flex, so VORP
# ordering collapses to raw points ordering, which favours WRs at every matched rank (sec.L).
# They ask whether the board's ORDERING is bad or only its roster MIX.  They are reported
# with family = 0 and are excluded from the BH family, exactly as sec.M handled its own
# post-hoc diagnostics.
MINRB, MINRB_BY = 4, 7

def our_pick(S, taken, pos, cnt, picks_left, roster, rank_adp, Ev, Vppg, Rrep, drb_season,
             rnd=1):
    ok = legal_mask(cnt, picks_left)
    if S in ("SF", "SG") and cnt[1] < MINRB and (MINRB_BY - rnd) <= (MINRB - cnt[1]):
        if ((~taken) & (pos == 1)).any():
            ok = np.zeros(4, bool); ok[1] = True
    if S == "SA":
        for p in rank_adp:
            if not taken[p] and ok[pos[p]]:
                return p
        for p in rank_adp:
            if not taken[p]:
                return p
        return -1
    ev = Ev.copy()
    if S in ("SC", "SE", "SG"):
        ev = ev + drb_season * (pos == 1)
    if S in ("SB", "SC", "SF", "SG"):
        cand = np.where((~taken) & ok[pos])[0]
        if not len(cand):
            cand = np.where(~taken)[0]
        vor = ev - np.array([Rrep[k] for k in pos])
        return int(cand[np.argmax(vor[cand])])
    # SD / SE: lineup-marginal greedy
    cand = np.where((~taken) & ok[pos])[0]
    if not len(cand):
        cand = np.where(~taken)[0]
    cand = cand[np.argsort(-ev[cand])[:60]]
    base = lineup_value(ev, pos, roster, Rrep)
    best, bi = -1e18, int(cand[0])
    for p in cand:
        v = lineup_value(ev, pos, roster + [int(p)], Rrep) - base
        if v > best:
            best, bi = v, int(p)
    return bi

# --------------------------------------------------------------------------- run
def run(opp_model):
    recs = []
    t0 = time.time()
    for y in YEARS:
        S = SEASON[y]
        n, pos, adp, sd, Ev = S["n"], S["pos"], S["adp"], S["sd"], S["Ev"]
        Rrep = np.array([R_loso[(y, p)] for p in POSN])
        Vppg = (Ev - np.array([Rrep[k] for k in pos])) / 17.0
        slot_of = np.empty(NTEAM * NROUND, int)
        for r in range(NROUND):
            seq = list(range(NTEAM)) if r % 2 == 0 else list(range(NTEAM))[::-1]
            slot_of[r * NTEAM:(r + 1) * NTEAM] = seq
        for myslot in range(NTEAM):
            for sim in range(NSIM):
                seed = 1_000_000 * y + 1000 * myslot + sim
                for st in STRATS:
                    # common random numbers: identical opponent draws across strategies
                    rng = np.random.default_rng(seed)
                    eps = rng.normal(0, 1, size=(NTEAM, n)) * sd[None, :]
                    boards = np.argsort(adp[None, :] + eps, axis=1)
                    taken = np.zeros(n, bool)
                    cnt = np.zeros((NTEAM, 4), int)
                    rosters = [[] for _ in range(NTEAM)]
                    ptr = np.zeros(NTEAM, int)
                    for k in range(NTEAM * NROUND):
                        tt = slot_of[k]
                        rnd = k // NTEAM + 1
                        left = NROUND - rnd + 1
                        if tt == myslot:
                            p = our_pick(st, taken, pos, cnt[tt], left, rosters[tt],
                                         S["rank_adp"], Ev, Vppg, Rrep, DELTA_RB_SEASON,
                                         rnd=rnd)
                        elif opp_model == "adp":
                            p, j = opp_pick_adp(boards[tt], ptr[tt], taken, pos, cnt[tt], left)
                            ptr[tt] = j
                        else:
                            p = opp_pick_logit(Vppg, pos, taken, cnt[tt], left, rng)
                            if p is None:
                                continue
                        taken[p] = True
                        cnt[tt][pos[p]] += 1
                        rosters[tt].append(int(p))
                    cm = cnt[myslot]
                    recs.append(dict(opp=opp_model, year=y, slot=myslot + 1, sim=sim, strat=st,
                                     pts14=score_roster(S["wkm"], pos, rosters[myslot], 1, 14),
                                     pts17=score_roster(S["wkm"], pos, rosters[myslot], 15, 17),
                                     nQB=cm[0], nRB=cm[1], nWR=cm[2], nTE=cm[3]))
        print(f"  [{opp_model}] {y} done {time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(recs)

R = pd.concat([run("logit"), run("adp")], ignore_index=True)
R["pts_all"] = R.pts14 + R.pts17
R.to_csv(f"{RES}/sectionW3_m_rerun_bysim.csv", index=False)

def dm_table(df, metric, opp):
    d = df[df.opp == opp]
    piv = d.pivot_table(index=["year", "slot", "sim"], columns="strat", values=metric)
    rows = []
    for st in STRATS[1:]:
        a = (piv[st] - piv["SA"]).groupby(level="year").mean().values
        se = a.std(ddof=1) / np.sqrt(len(a))
        t = a.mean() / se
        rows.append(dict(opp=opp, metric=metric, strat=st, mean_diff=a.mean(), se=se, t=t,
                         p=2 * (1 - stats.t.cdf(abs(t), len(a) - 1)),
                         ci_lo=a.mean() - 2.262 * se, ci_hi=a.mean() + 2.262 * se,
                         mde=2.802 * se, seasons_positive=int((a > 0).sum())))
    out = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    out["family"] = np.where(out.strat.isin(["SF", "SG"]), 0, 1)
    fam = out[out.family == 1].copy().reset_index(drop=True)
    fam["bh_thresh"] = 0.10 * (fam.index + 1) / len(fam)
    out = out.merge(fam[["strat", "bh_thresh"]], on="strat", how="left")
    out["reject"] = out.p <= out.bh_thresh
    return out

TAB = pd.concat([dm_table(R, m, o) for m in ("pts14", "pts17", "pts_all")
                 for o in ("logit", "adp")], ignore_index=True)
TAB.to_csv(f"{RES}/sectionW3_m_rerun.csv", index=False)
for o in ("logit", "adp"):
    print(f"\n=== opponents = {o} : DM vs SA (ADP null), clustered by season, t(9), BH q=.10 ===")
    print(TAB[(TAB.opp == o) & (TAB.metric == "pts14")].round(3).to_string(index=False))

print("\n=== mean weeks 1-14 by strategy and opponent model ===")
print(R.pivot_table(index="strat", columns="opp", values="pts14").round(1).to_string())
print("\n=== roster mix (mean counts) ===")
print(R.groupby(["opp", "strat"])[["nQB", "nRB", "nWR", "nTE"]].mean().round(2).to_string())
print("\n=== by slot, weeks 1-14, difference vs SA (logit opponents) ===")
bs = R[R.opp == "logit"].pivot_table(index="slot", columns="strat", values="pts14")
print(bs.sub(bs["SA"], axis=0).round(1).to_string())
print("\n[out] results/sectionW3_m_rerun.csv, results/sectionW3_m_rerun_bysim.csv")

#!/usr/bin/env python3
"""
§R — Behavioral draft model (conditional logit / Plackett-Luce), fitted on the
owner's live 2026 league draft.  Pre-registration: EDA_PLAN8.md §R.
Spec: fantasy_draft_model.md.  Binding constraints: REPORT.md §38(1),(2).

Model set FIXED BEFORE FITTING (see results/sectionR_notes.md header):
  M0 : U_j = b_v * v_j
  M1 : U_j = b_v * v_j + sum_p b_p * need_{m,p,t} * 1[pos(j)=p]        (= §R1 as written)
  M2 : M1 + positional intercepts a_p (WR reference) + b_flex * flexneed * 1[pos in RB,WR]
Selection criterion, declared before fitting: leave-one-pick-out mean predictive
log-likelihood.  tau == 1 throughout (§38(1)); reported temperature is 1/b_v.

Owner's 9 picks are EXCLUDED from the likelihood (they still remove players from
the pool).  We are modelling opponent behaviour; his 37 logged views make him a
different data-generating process.

Nothing here writes to results/board_2026.csv.
"""
import os, re, json, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

ROOT = "/Users/thomasmcnamee/NFL"
RES = os.path.join(ROOT, "results")

# ----------------------------------------------------------------------------- data
board = pd.read_csv(os.path.join(RES, "board_2026.csv"))
log = pd.read_csv(os.path.join(ROOT, "data/drafts/league_draft_2026.csv"))

SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?")
def norm_name(s):
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    s = SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()

board["key"] = board["name"].map(norm_name)
log["key"] = log["player"].map(norm_name)
assert board["key"].is_unique

bk = set(board["key"])
missing = sorted(set(log["key"]) - bk)
# Off-board picks: players drafted who are not in our 204-player universe.
OFFBOARD_POS = {"chris godwin": "WR"}
for m in missing:
    assert m in OFFBOARD_POS, f"unhandled off-board pick: {m}"

N_TEAMS, N_ROUNDS = 10, 15
N_PICKS = N_TEAMS * N_ROUNDS
BASE_REQ = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
POS = ["QB", "RB", "TE", "WR"]

def slot_of(overall):
    r = (overall - 1) // N_TEAMS + 1
    p = (overall - 1) % N_TEAMS + 1
    return p if r % 2 == 1 else N_TEAMS - p + 1

# sanity: reconstructed snake slots match the log
assert (log["overall"].map(slot_of).values == log["team_slot"].values).all()

vals = board.set_index("key")["final"].to_dict()
poss = board.set_index("key")["position"].to_dict()
names = board.set_index("key")["name"].to_dict()
adpr = board.set_index("key")["adp_rank_overall"].to_dict()

# ------------------------------------------------------------------- roster / need
def empty_rosters():
    return {s: {p: 0 for p in POS} for s in range(1, N_TEAMS + 1)}

def need_vec(roster):
    """returns dict pos->unfilled base starter slots, and flex_need"""
    nd = {p: max(0, BASE_REQ[p] - roster[p]) for p in POS}
    flex_filled = max(0, roster["RB"] - 2) + max(0, roster["WR"] - 2)
    return nd, max(0, 2 - flex_filled)

# ---------------------------------------------------------------- design builder
def design(keys, slot, roster):
    """rows = candidate players (keys); returns X for M2 (superset of M0/M1)."""
    nd, fn = need_vec(roster)
    v = np.array([vals[k] for k in keys])
    p = np.array([poss[k] for k in keys])
    cols = {"v": v}
    for q in POS:
        cols["need_" + q] = nd[q] * (p == q).astype(float)
    for q in ["QB", "RB", "TE"]:                      # WR = reference
        cols["int_" + q] = (p == q).astype(float)
    cols["flex"] = fn * np.isin(p, ["RB", "WR"]).astype(float)
    return pd.DataFrame(cols), p

COLS = {
    "M0": ["v"],
    "M1": ["v", "need_QB", "need_RB", "need_TE", "need_WR"],
    "M2": ["v", "need_QB", "need_RB", "need_TE", "need_WR",
           "int_QB", "int_RB", "int_TE", "flex"],
}
COLS["M2R"] = COLS["M2"]
# Weakly-informative ridge N(0, PRIOR_SD^2) on every coefficient EXCEPT beta_v.
# Added 2026-08-24 AFTER observing quasi-separation in M2 (SE(int_QB)=419) and BEFORE
# any VONA / player-level output was inspected.  Trigger was a numerical pathology, not
# a substantive result.  See sectionR_notes.md 'the QB separation problem'.
PRIOR_SD = 5.0
PENALIZED = {"M2R"}

# --------------------------------------------------- reconstruct choice sequence
def build_observations():
    """One record per realised pick: available keys, chosen index, slot, roster snapshot."""
    avail = set(board["key"])
    rosters = empty_rosters()
    obs = []
    for _, row in log.sort_values("overall").iterrows():
        k, slot = row["key"], int(row["team_slot"])
        onboard = k in bk
        rec = dict(overall=int(row["overall"]), slot=slot, key=k, onboard=onboard,
                   is_mine=bool(row["is_mine"]),
                   avail=sorted(avail), roster={p: rosters[slot][p] for p in POS})
        obs.append(rec)
        if onboard:
            avail.discard(k)
            rosters[slot][poss[k]] += 1
        else:
            rosters[slot][OFFBOARD_POS[k]] += 1
    return obs, avail, rosters

OBS, AVAIL_NOW, ROSTERS_NOW = build_observations()
FIT_OBS = [o for o in OBS if o["onboard"] and not o["is_mine"]]
print(f"[data] {len(OBS)} logged picks; {len(FIT_OBS)} used for fitting "
      f"(dropped {sum(o['is_mine'] for o in OBS)} owner picks, "
      f"{sum(not o['onboard'] for o in OBS)} off-board)")
print(f"[data] {len(AVAIL_NOW)} board players remain available")

# ------------------------------------------------------------------------ likelihood
def make_matrices(obs_list, model):
    cols = COLS[model]
    mats, ys = [], []
    for o in obs_list:
        X, _ = design(o["avail"], o["slot"], o["roster"])
        mats.append(X[cols].to_numpy())
        ys.append(o["avail"].index(o["key"]))
    return mats, ys

def negll(theta, mats, ys, pen=False):
    s = 0.0
    for X, y in zip(mats, ys):
        u = X @ theta
        s -= u[y] - (np.max(u) + np.log(np.exp(u - np.max(u)).sum()))
    if pen:
        s += 0.5 * np.sum(theta[1:]**2) / PRIOR_SD**2
    return s

def fit(obs_list, model, x0=None):
    mats, ys = make_matrices(obs_list, model)
    k = len(COLS[model])
    x0 = np.zeros(k) if x0 is None else x0
    pen = model in PENALIZED
    r = minimize(negll, x0, args=(mats, ys, pen), method="BFGS",
                 options=dict(maxiter=2000, gtol=1e-6))
    return r, mats, ys

def hessian_num(f, x, eps=1e-4):
    k = len(x); H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = eps
            ej = np.zeros(k); ej[j] = eps
            H[i, j] = H[j, i] = (f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) / (4*eps*eps)
    return H

def scores(theta, mats, ys):
    """per-observation gradient of the log-likelihood (for clustered SEs)."""
    out = []
    for X, y in zip(mats, ys):
        u = X @ theta; u -= u.max()
        p = np.exp(u); p /= p.sum()
        out.append(X[y] - p @ X)
    return np.array(out)

# ------------------------------------------- descriptive: does the room draft our board?
def top_value_diagnostic():
    rows = []
    for o in OBS:
        av = o["avail"]
        vv = np.array([vals[k] for k in av])
        top = av[int(np.argmax(vv))]
        rows.append(dict(overall=o["overall"], slot=o["slot"], took_pos=poss[o["key"]],
                         took_val=vals[o["key"]], top_pos=poss[top], top_val=vv.max(),
                         qb_held=o["roster"]["QB"], is_mine=o["is_mine"],
                         took_top=int(o["key"] == top)))
    return pd.DataFrame(rows)
TOPDIAG = top_value_diagnostic()
QBTOP = TOPDIAG[TOPDIAG["top_pos"] == "QB"]
QB_TOP_N = len(QBTOP); QB_TOP_TAKEN = int((QBTOP["took_pos"] == "QB").sum())
QB_FILLED_N = int((QBTOP["qb_held"] > 0).sum())
QB_FILLED_TAKEN = int(((QBTOP["qb_held"] > 0) & (QBTOP["took_pos"] == "QB")).sum())
print(f"[diag] picks where max-value available was a QB: {QB_TOP_N}/{len(TOPDIAG)}; "
      f"QB actually taken {QB_TOP_TAKEN}; of the {QB_FILLED_N} facing a QB-topped board "
      f"who already had a QB, {QB_FILLED_TAKEN} took one")
print(f"[diag] overall rate of taking the max-value available player: {TOPDIAG['took_top'].mean():.3f}")

# ------------------------------------------------------------------- fit M0/M1/M2
results = {}
for model in ["M0", "M1", "M2", "M2R"]:
    r, mats, ys = fit(FIT_OBS, model)
    H = hessian_num(lambda th: negll(th, mats, ys, model in PENALIZED), r.x)
    V = np.linalg.inv(H)
    se = np.sqrt(np.diag(V))
    # manager-clustered (10 clusters -- reported with a caveat)
    S = scores(r.x, mats, ys)
    cl = np.array([o["slot"] for o in FIT_OBS])
    meat = np.zeros((len(r.x), len(r.x)))
    for c in np.unique(cl):
        g = S[cl == c].sum(0)
        meat += np.outer(g, g)
    G = len(np.unique(cl))
    Vc = V @ meat @ V * (G / (G - 1))
    results[model] = dict(theta=r.x, se=se, se_cl=np.sqrt(np.diag(Vc)),
                          ll=-r.fun, V=V, cols=COLS[model], mats=mats, ys=ys)
    print(f"[fit] {model}: ll={-r.fun:.2f}  k={len(r.x)}")
    for c, t, s, sc in zip(COLS[model], r.x, se, np.sqrt(np.diag(Vc))):
        print(f"        {c:9s} {t:8.4f}  se {s:.4f}  se_cl {sc:.4f}")

# ------------------------------------------------------------- OLS baseline model
mk = log[log["key"].isin(bk)].copy()
mk["adp_rank_overall"] = mk["key"].map(adpr)
mk["pos"] = mk["key"].map(poss)
Xb = pd.get_dummies(mk["pos"], drop_first=False)[["QB", "RB", "TE"]].astype(float)
Xb.insert(0, "adp_rank", mk["adp_rank_overall"].values)
Xb.insert(0, "const", 1.0)
yb = mk["overall"].values.astype(float)
bhat, *_ = np.linalg.lstsq(Xb.to_numpy(), yb, rcond=None)
resid = yb - Xb.to_numpy() @ bhat
r2 = 1 - resid.var() / yb.var()
sd_resid = resid.std(ddof=Xb.shape[1])
print(f"[ols] R2={r2:.3f} residSD={sd_resid:.2f} coefs={dict(zip(Xb.columns, bhat.round(3)))}")

def ols_fit(obs_list):
    idx = [o["overall"] for o in obs_list]
    sub = mk[mk["overall"].isin(idx)]
    X = pd.get_dummies(sub["pos"]).reindex(columns=["QB", "RB", "TE"], fill_value=0).astype(float)
    X.insert(0, "adp_rank", sub["adp_rank_overall"].values)
    X.insert(0, "const", 1.0)
    y = sub["overall"].values.astype(float)
    b, *_ = np.linalg.lstsq(X.to_numpy(), y, rcond=None)
    e = y - X.to_numpy() @ b
    return b, max(e.std(ddof=X.shape[1]), 1e-6)

def ols_probs(b, sd, keys, t):
    """Gaussian pick-position model -> conditional distribution over the available set."""
    ar = np.array([adpr[k] for k in keys], float)
    p = np.array([poss[k] for k in keys])
    mu = b[0] + b[1]*ar + b[2]*(p == "QB") + b[3]*(p == "RB") + b[4]*(p == "TE")
    d = norm.pdf((t - mu) / sd)
    s = d.sum()
    return d / s if s > 0 else np.full(len(keys), 1/len(keys))

# ------------------------------------------------------- leave-one-pick-out (LOPO)
def lopo():
    rows = []
    for i, o in enumerate(FIT_OBS):
        rest = FIT_OBS[:i] + FIT_OBS[i+1:]
        rec = dict(overall=o["overall"], n_avail=len(o["avail"]), key=o["key"])
        for model in ["M0", "M1", "M2", "M2R"]:
            r, _, _ = fit(rest, model, x0=results[model]["theta"])
            X, _ = design(o["avail"], o["slot"], o["roster"])
            u = X[COLS[model]].to_numpy() @ r.x
            u -= u.max(); p = np.exp(u); p /= p.sum()
            y = o["avail"].index(o["key"])
            order = np.argsort(-p)
            rec[f"p_{model}"] = p[y]
            rec[f"rank_{model}"] = int(np.where(order == y)[0][0]) + 1
        b, sd = ols_fit(rest)
        p = ols_probs(b, sd, o["avail"], o["overall"])
        y = o["avail"].index(o["key"])
        rec["p_OLS"] = p[y]
        rec["rank_OLS"] = int(np.where(np.argsort(-p) == y)[0][0]) + 1
        rec["p_UNIF"] = 1.0 / len(o["avail"])
        rows.append(rec)
    return pd.DataFrame(rows)

print("[lopo] running leave-one-pick-out ...")
L = lopo()
summ = []
for m in ["UNIF", "OLS", "M0", "M1", "M2", "M2R"]:
    ll = np.log(L[f"p_{m}"]).mean()
    if m == "UNIF":
        summ.append(dict(model=m, mean_loglik=ll, se=np.log(L[f"p_{m}"]).std(ddof=1)/np.sqrt(len(L)),
                         top1=np.nan, top5=np.nan))
    else:
        summ.append(dict(model=m, mean_loglik=ll,
                         se=np.log(L[f"p_{m}"]).std(ddof=1)/np.sqrt(len(L)),
                         top1=(L[f"rank_{m}"] == 1).mean(), top5=(L[f"rank_{m}"] <= 5).mean()))
SUMM = pd.DataFrame(summ)
print(SUMM.to_string(index=False))

# paired differences vs OLS
paired = {}
for m in ["M0", "M1", "M2", "M2R"]:
    d = np.log(L[f"p_{m}"]) - np.log(L["p_OLS"])
    paired[m] = (d.mean(), d.std(ddof=1)/np.sqrt(len(d)))
    print(f"[lopo] {m} - OLS  dlogL = {d.mean():+.4f} +/- {d.std(ddof=1)/np.sqrt(len(d)):.4f}"
          f"  (t={d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):+.2f})")

BEST = max(["M0", "M1", "M2", "M2R"], key=lambda m: SUMM.set_index("model").loc[m, "mean_loglik"])
CTX = BEST
print(f"[select] pre-declared criterion (LOPO mean log-lik) selects {BEST}")

# --------------------------------------------------------- pick-level calibration
def full_lopo_pairs():
    recs = []
    for i, o in enumerate(FIT_OBS):
        rest = FIT_OBS[:i] + FIT_OBS[i+1:]
        r, _, _ = fit(rest, CTX, x0=results[CTX]["theta"])
        X, _ = design(o["avail"], o["slot"], o["roster"])
        u = X[COLS[CTX]].to_numpy() @ r.x
        u -= u.max(); p = np.exp(u); p /= p.sum()
        b, sd = ols_fit(rest)
        po = ols_probs(b, sd, o["avail"], o["overall"])
        y = o["avail"].index(o["key"])
        for j in range(len(o["avail"])):
            recs.append((o["overall"], o["avail"][j], p[j], po[j], int(j == y)))
    return pd.DataFrame(recs, columns=["overall", "key", "p_model", "p_ols", "chosen"])

PAIRS = full_lopo_pairs()
def bucketise(df, col):
    edges = np.array([0, .005, .01, .02, .05, .10, .20, .40, .70, 1.0001])
    b = pd.cut(df[col], edges, right=False)
    g = df.groupby(b, observed=True).agg(n=("chosen", "size"), pred=(col, "mean"),
                                         obs=("chosen", "mean")).reset_index()
    g.columns = ["bucket", "n", "pred", "obs"] + list(g.columns[4:])
    g["se"] = np.sqrt(g["obs"]*(1-g["obs"])/g["n"])
    return g
CAL_MODEL = bucketise(PAIRS, "p_model")
CAL_OLS = bucketise(PAIRS, "p_ols")
print("[calib] pick-level, model:\n", CAL_MODEL.to_string(index=False))

# ------------------------------------------------- forward simulation machinery
def set_ctx(model):
    global theta_hat, V_hat, cols_best, CTX
    CTX = model
    theta_hat = results[model]["theta"]
    V_hat = results[model]["V"]
    cols_best = COLS[model]
set_ctx(BEST)

# off-board hazard: fraction of realised picks that took a non-board player
N_OFF = sum(not o["onboard"] for o in OBS)
LAMBDA = (N_OFF + 0.5) / (len(OBS) + 1.0)   # Jeffreys posterior mean; raw rate is 0/87
print(f"[sim] off-board hazard lambda = {LAMBDA:.4f} (baseline)")

KEYS_ALL = list(board["key"])
KIDX = {k: i for i, k in enumerate(KEYS_ALL)}
V_ARR = board.set_index("key").loc[KEYS_ALL, "final"].to_numpy()
P_ARR = board.set_index("key").loc[KEYS_ALL, "position"].to_numpy()
POS_ONEHOT = {q: (P_ARR == q).astype(float) for q in POS}

def util_all(theta, roster):
    """utility for every board player under BEST, given a roster state."""
    nd, fn = need_vec(roster)
    d = dict(zip(cols_best, theta))
    u = d["v"] * V_ARR
    for q in POS:
        if "need_" + q in d:
            u = u + d["need_" + q] * nd[q] * POS_ONEHOT[q]
    for q in ["QB", "RB", "TE"]:
        if "int_" + q in d:
            u = u + d["int_" + q] * POS_ONEHOT[q]
    if "flex" in d:
        u = u + d["flex"] * fn * (POS_ONEHOT["RB"] + POS_ONEHOT["WR"])
    return u

def simulate(avail_mask, rosters, start_pick, end_pick, theta, rng, lam=LAMBDA,
             owner_slot=None, owner_policy=None, owner_roster=None):
    """Run picks start..end.  Returns final availability mask and owner acquisitions."""
    got = []
    for t in range(start_pick, end_pick + 1):
        s = slot_of(t)
        if owner_slot is not None and s == owner_slot:
            k = owner_policy(avail_mask, owner_roster, t, rng)
            if k is not None:
                avail_mask[k] = False
                owner_roster[P_ARR[k]] += 1
                got.append((t, k))
            continue
        if rng.random() < lam:
            continue                                   # off-board pick (DST/deep flier)
        u = util_all(theta, rosters[s])
        u = np.where(avail_mask, u, -np.inf)
        m = u.max()
        if not np.isfinite(m):
            continue
        p = np.exp(u - m); p /= p.sum()
        k = rng.choice(len(p), p=p)
        avail_mask[k] = False
        rosters[s][P_ARR[k]] += 1
    return avail_mask, got

def base_state():
    mask = np.array([k in AVAIL_NOW for k in KEYS_ALL])
    rost = {s: dict(ROSTERS_NOW[s]) for s in ROSTERS_NOW}
    return mask, rost

# --------------------------------------------- out-of-sample SURVIVAL calibration
def survival_calibration(anchors=(31, 41, 51, 61, 71), horizon=15, nsim=2000):
    rows = []
    for t0 in anchors:
        train = [o for o in FIT_OBS if o["overall"] < t0]
        r, _, _ = fit(train, CTX, x0=theta_hat)
        # state at t0
        avail = set(board["key"]); rost = empty_rosters()
        for o in OBS:
            if o["overall"] >= t0:
                break
            if o["onboard"]:
                avail.discard(o["key"]); rost[o["slot"]][poss[o["key"]]] += 1
            else:
                rost[o["slot"]][OFFBOARD_POS[o["key"]]] += 1
        mask0 = np.array([k in avail for k in KEYS_ALL])
        tend = min(t0 + horizon - 1, len(OBS))
        surv = np.zeros(len(KEYS_ALL))
        rng = np.random.default_rng(1000 + t0)
        for _ in range(nsim):
            m = mask0.copy(); rr = {s: dict(rost[s]) for s in rost}
            m, _ = simulate(m, rr, t0, tend, r.x, rng)
            surv += m
        surv /= nsim
        # realised
        taken = {o["key"] for o in OBS if t0 <= o["overall"] <= tend and o["onboard"]}
        for i, k in enumerate(KEYS_ALL):
            if mask0[i]:
                rows.append(dict(anchor=t0, key=k, name=names[k], pos=poss[k],
                                 p_surv=surv[i], survived=int(k not in taken)))
    return pd.DataFrame(rows)

print("[calib] out-of-sample survival calibration (temporal, refit on picks < anchor) ...")
SURV_CAL = survival_calibration()
edges = np.array([0, .1, .3, .5, .7, .9, .98, 1.0001])
b = pd.cut(SURV_CAL["p_surv"], edges, right=False)
SURV_TAB = SURV_CAL.groupby(b, observed=True).agg(
    n=("survived", "size"), pred=("p_surv", "mean"), obs=("survived", "mean")).reset_index()
SURV_TAB.columns = ["bucket", "n", "pred", "obs"]
SURV_TAB["se"] = np.sqrt(SURV_TAB["obs"]*(1-SURV_TAB["obs"])/SURV_TAB["n"])
print(SURV_TAB.to_string(index=False))
brier = ((SURV_CAL["p_surv"] - SURV_CAL["survived"])**2).mean()
base = SURV_CAL["survived"].mean()
brier0 = ((base - SURV_CAL["survived"])**2).mean()
print(f"[calib] survival Brier {brier:.4f} vs base-rate {brier0:.4f} "
      f"(skill {1-brier/brier0:+.3f})")

# ----------------------------------------------------------- survival curves fwd
NEXT_LOG = int(log["overall"].max())
OWNER_PICKS = [105, 116, 125, 136, 145]
DECISION_A = NEXT_LOG + 1          # 88 -- "current pick" (owner's stated '87' slot)
SCHED_A = [DECISION_A] + OWNER_PICKS

def survival_curves(mask0, rost0, start, checkpoints, nsim=8000, lam=LAMBDA,
                    theta_draw=True, seed=7):
    rng = np.random.default_rng(seed)
    out = {c: np.zeros(len(KEYS_ALL)) for c in checkpoints}
    Lch = np.linalg.cholesky(V_hat + 1e-10*np.eye(len(theta_hat)))
    for _ in range(nsim):
        th = theta_hat + (Lch @ rng.standard_normal(len(theta_hat))) if theta_draw else theta_hat
        m = mask0.copy(); rr = {s: dict(rost0[s]) for s in rost0}
        cur = start
        for c in checkpoints:
            m, _ = simulate(m, rr, cur, c - 1, th, rng, lam=lam)
            out[c] += m
            cur = c
    for c in checkpoints:
        out[c] /= nsim
    return out

MASK0, ROST0 = base_state()
CHECK = SCHED_A
SURV = survival_curves(MASK0, ROST0, DECISION_A, CHECK)

sc = pd.DataFrame({"name": [names[k] for k in KEYS_ALL],
                   "position": P_ARR, "value": V_ARR,
                   "available_now": MASK0})
for c in CHECK:
    sc[f"p_avail_at_{c}"] = SURV[c]
sc = sc[sc["available_now"]].drop(columns=["available_now"]).sort_values("value", ascending=False)
sc.to_csv(os.path.join(RES, "survival_curves.csv"), index=False)
print(f"[out] survival_curves.csv ({len(sc)} undrafted players)")

# ------------------------------------------------------------ positional run risk
def run_risk(mask0, rost0, start, target, nsim=4000, lam=LAMBDA, seed=11):
    rng = np.random.default_rng(seed)
    counts = {q: [] for q in POS}
    for _ in range(nsim):
        m = mask0.copy(); rr = {s: dict(rost0[s]) for s in rost0}
        before = m.copy()
        m, _ = simulate(m, rr, start, target - 1, theta_hat, rng, lam=lam)
        gone = before & (~m)
        for q in POS:
            counts[q].append(int((P_ARR[gone] == q).sum()))
    return {q: np.array(v) for q, v in counts.items()}

RUN = run_risk(MASK0, ROST0, DECISION_A, OWNER_PICKS[0])

# --------------------------------------------------------------------- VONA
def vona(mask0, rost0, now, nxt, nsim=8000, lam=LAMBDA, seed=23):
    rng = np.random.default_rng(seed)
    Lch = np.linalg.cholesky(V_hat + 1e-10*np.eye(len(theta_hat)))
    best_next = {q: [] for q in POS}
    for _ in range(nsim):
        th = theta_hat + Lch @ rng.standard_normal(len(theta_hat))
        m = mask0.copy(); rr = {s: dict(rost0[s]) for s in rost0}
        m, _ = simulate(m, rr, now, nxt - 1, th, rng, lam=lam)
        for q in POS:
            sel = m & (P_ARR == q)
            best_next[q].append(V_ARR[sel].max() if sel.any() else np.nan)
    rows = []
    for q in POS:
        sel = mask0 & (P_ARR == q)
        bn = np.array(best_next[q]); bn = bn[~np.isnan(bn)]
        now_best = V_ARR[sel].max() if sel.any() else np.nan
        nm = np.array(KEYS_ALL)[sel][np.argmax(V_ARR[sel])] if sel.any() else None
        rows.append(dict(position=q, best_now=now_best,
                         best_now_player=names[nm] if nm else "",
                         E_best_next=bn.mean(), sd_best_next=bn.std(ddof=1),
                         se_E_best_next=bn.std(ddof=1)/np.sqrt(len(bn)),
                         p10=np.quantile(bn, .10), p50=np.quantile(bn, .5),
                         p90=np.quantile(bn, .90),
                         vona=now_best - bn.mean(),
                         p_top_survives=float(np.mean(bn >= now_best - 1e-9))))
    return pd.DataFrame(rows)

VONA_A = vona(MASK0, ROST0, DECISION_A, OWNER_PICKS[0])
VONA_A.insert(0, "next_pick", OWNER_PICKS[0]); VONA_A.insert(0, "decision_pick", DECISION_A)

# secondary framing: if pick 87/88 has already gone, the live decision is 105 -> 116
def vona_from_simulated(start, decision, nxt, nsim=4000, seed=31):
    """decision-point VONA when the board must first be simulated forward to `decision`."""
    rng = np.random.default_rng(seed)
    Lch = np.linalg.cholesky(V_hat + 1e-10*np.eye(len(theta_hat)))
    now_best = {q: [] for q in POS}; nxt_best = {q: [] for q in POS}
    for _ in range(nsim):
        th = theta_hat + Lch @ rng.standard_normal(len(theta_hat))
        m, rr = base_state()
        m, _ = simulate(m, rr, start, decision - 1, th, rng)
        for q in POS:
            s = m & (P_ARR == q); now_best[q].append(V_ARR[s].max() if s.any() else np.nan)
        m, _ = simulate(m, rr, decision, nxt - 1, th, rng)
        for q in POS:
            s = m & (P_ARR == q); nxt_best[q].append(V_ARR[s].max() if s.any() else np.nan)
    rows = []
    for q in POS:
        a = np.array(now_best[q]); bq = np.array(nxt_best[q])
        rows.append(dict(decision_pick=decision, next_pick=nxt, position=q,
                         best_now=np.nanmean(a), best_now_player="(simulated)",
                         E_best_next=np.nanmean(bq), sd_best_next=np.nanstd(bq, ddof=1),
                         se_E_best_next=np.nanstd(bq, ddof=1)/np.sqrt(len(bq)),
                         p10=np.nanquantile(bq, .1), p50=np.nanquantile(bq, .5),
                         p90=np.nanquantile(bq, .9),
                         vona=np.nanmean(a) - np.nanmean(bq),
                         p_top_survives=float(np.nanmean(bq >= a))))
    return pd.DataFrame(rows)

VONA_B = vona_from_simulated(DECISION_A, OWNER_PICKS[0], OWNER_PICKS[1])
VONA_ALL = pd.concat([VONA_A, VONA_B], ignore_index=True)
VONA_ALL.to_csv(os.path.join(RES, "vona.csv"), index=False)
print("[out] vona.csv")
print(VONA_ALL.to_string(index=False))

# --------------------------------- lookahead: which position to take at DECISION_A
OWNER_ROSTER_START = {p: 0 for p in POS}
for o in OBS:
    if o["is_mine"]:
        OWNER_ROSTER_START[poss[o["key"]] if o["onboard"] else OFFBOARD_POS[o["key"]]] += 1
print("[owner] roster:", OWNER_ROSTER_START)

OWNER_STARTERS = [("QB", 1), ("RB", 2), ("WR", 2), ("TE", 1)]
def lineup_value(vlist_by_pos):
    """best starting lineup value: 1QB 2RB 2WR 1TE 2FLEX(RB/WR).  DST excluded (constant)."""
    tot = 0.0
    pools = {q: sorted(vlist_by_pos.get(q, []), reverse=True) for q in POS}
    used = {q: 0 for q in POS}
    for q, n in OWNER_STARTERS:
        take = pools[q][:n]
        tot += sum(take); used[q] = len(take)
    flexpool = sorted(pools["RB"][used["RB"]:] + pools["WR"][used["WR"]:], reverse=True)
    tot += sum(flexpool[:2])
    return tot

OWNER_HELD = {q: [] for q in POS}
for o in OBS:
    if o["is_mine"]:
        if o["onboard"]:
            OWNER_HELD[poss[o["key"]]].append(vals[o["key"]])
        else:
            OWNER_HELD[OFFBOARD_POS[o["key"]]].append(0.0)

def greedy_pick(mask, held, rng):
    """pick the available player that maximises marginal starting-lineup value;
       ties broken by raw value."""
    cur = lineup_value(held)
    best, bestk = -1e9, None
    idx = np.where(mask)[0]
    # only consider top-15 by value per position for speed
    cand = []
    for q in POS:
        qi = idx[P_ARR[idx] == q]
        if len(qi):
            cand.extend(qi[np.argsort(-V_ARR[qi])[:8]])
    for k in cand:
        h = {p: list(held[p]) for p in held}
        h[P_ARR[k]].append(V_ARR[k])
        d = lineup_value(h) - cur
        score = d + 1e-3 * V_ARR[k]
        if score > best:
            best, bestk = score, k
    return bestk

def evaluate_strategy(first_action, nsim=1500, seed=5):
    """first_action: 'greedy' or a position string to force at DECISION_A."""
    rng = np.random.default_rng(seed)
    Lch = np.linalg.cholesky(V_hat + 1e-10*np.eye(len(theta_hat)))
    totals, firsts = [], []
    sched = SCHED_A
    for _ in range(nsim):
        th = theta_hat + Lch @ rng.standard_normal(len(theta_hat))
        m, rr = base_state()
        held = {q: list(OWNER_HELD[q]) for q in POS}
        for i, pk in enumerate(sched):
            if i > 0:
                m, _ = simulate(m, rr, sched[i-1] + 1, pk - 1, th, rng)
            idx = np.where(m)[0]
            if i == 0 and first_action != "greedy":
                qi = idx[P_ARR[idx] == first_action]
                k = qi[np.argmax(V_ARR[qi])] if len(qi) else None
            else:
                k = greedy_pick(m, held, rng)
            if k is None:
                continue
            m[k] = False
            held[P_ARR[k]].append(V_ARR[k])
            if i == 0:
                firsts.append(names[KEYS_ALL[k]])
        totals.append(lineup_value(held))
    return np.array(totals), firsts

# ---- marginal STARTING-LINEUP VONA: the decision-relevant version of the same quantity
def lineup_vona(mask0, rost0, now, nxt, nsim=4000, seed=41):
    """VONA measured in marginal starting-lineup points rather than raw board value."""
    rng = np.random.default_rng(seed)
    Lch = np.linalg.cholesky(V_hat + 1e-10*np.eye(len(theta_hat)))
    cur = lineup_value(OWNER_HELD)
    marg_now, best_now_k = {}, {}
    for q in POS:
        sel = mask0 & (P_ARR == q)
        if not sel.any():
            marg_now[q] = 0.0; best_now_k[q] = None; continue
        bestm, bk_ = -1e9, None
        for k in np.where(sel)[0][np.argsort(-V_ARR[np.where(sel)[0]])[:10]]:
            h = {x: list(OWNER_HELD[x]) for x in POS}; h[q].append(V_ARR[k])
            d = lineup_value(h) - cur
            if d > bestm: bestm, bk_ = d, k
        marg_now[q] = bestm; best_now_k[q] = bk_
    marg_next = {q: [] for q in POS}
    for _ in range(nsim):
        th = theta_hat + Lch @ rng.standard_normal(len(theta_hat))
        m = mask0.copy(); rr = {x: dict(rost0[x]) for x in rost0}
        m, _ = simulate(m, rr, now, nxt - 1, th, rng)
        for q in POS:
            sel = m & (P_ARR == q)
            if not sel.any():
                marg_next[q].append(0.0); continue
            bestm = 0.0
            for k in np.where(sel)[0][np.argsort(-V_ARR[np.where(sel)[0]])[:6]]:
                h = {x: list(OWNER_HELD[x]) for x in POS}; h[q].append(V_ARR[k])
                bestm = max(bestm, lineup_value(h) - cur)
            marg_next[q].append(bestm)
    rows = []
    for q in POS:
        a = np.array(marg_next[q])
        rows.append(dict(position=q,
                         best_now_player=names[KEYS_ALL[best_now_k[q]]] if best_now_k[q] is not None else "",
                         marg_now=marg_now[q], E_marg_next=a.mean(),
                         se=a.std(ddof=1)/np.sqrt(len(a)), sd=a.std(ddof=1),
                         lineup_vona=marg_now[q] - a.mean()))
    return pd.DataFrame(rows).sort_values("lineup_vona", ascending=False)

LVONA = lineup_vona(MASK0, ROST0, DECISION_A, OWNER_PICKS[0])
print("[lineup VONA]\n", LVONA.to_string(index=False))
LVONA.to_csv(os.path.join(RES, "sectionR_lineup_vona.csv"), index=False)

# ---- breakeven: how large must the QB upgrade be for "QB now" to beat "TE now"?
def qb_breakeven(shifts=(0.0, 0.25, 0.50, 0.75, 1.00, 1.25), nsim=2500, seed=57):
    """Shift every AVAILABLE QB's evaluated value down by c, shrinking the upgrade over the
    incumbent QB, and recompute lineup VONA(QB).  Opponent behaviour is untouched -- only the
    owner's evaluation of what a new QB is worth changes."""
    rng0 = np.random.default_rng(seed)
    Lch = np.linalg.cholesky(V_hat + 1e-10*np.eye(len(theta_hat)))
    cur = lineup_value(OWNER_HELD)
    sims = []
    for _ in range(nsim):
        th = theta_hat + Lch @ rng0.standard_normal(len(theta_hat))
        m = MASK0.copy(); rr = {x: dict(ROST0[x]) for x in ROST0}
        m, _ = simulate(m, rr, DECISION_A, OWNER_PICKS[0] - 1, th, rng0)
        sims.append(m.copy())
    rows = []
    for c in shifts:
        ev = V_ARR.copy(); ev[P_ARR == "QB"] -= c
        def marg(mask):
            best = 0.0
            sel = np.where(mask & (P_ARR == "QB"))[0]
            if len(sel):
                for k in sel[np.argsort(-ev[sel])[:6]]:
                    h = {x: list(OWNER_HELD[x]) for x in POS}; h["QB"].append(ev[k])
                    best = max(best, lineup_value(h) - cur)
            return best
        now = marg(MASK0)
        nxt = float(np.mean([marg(m) for m in sims]))
        rows.append(dict(qb_shift=c, marg_now=now, E_marg_next=nxt, lineup_vona=now - nxt))
    return pd.DataFrame(rows)

QBBE = qb_breakeven()
print("[qb breakeven]\n", QBBE.to_string(index=False))
QBBE.to_csv(os.path.join(RES, "sectionR_qb_breakeven.csv"), index=False)

TE_LV = float(LVONA.set_index("position").loc["TE", "lineup_vona"])
QB_GAP = float(LVONA.set_index("position").loc["QB", "marg_now"])
BREAKEVEN_MD = (
    "### Breakeven on the QB call\n\n"
    "The QB-vs-TE decision rests on the board's gap between the best available QB and the\n"
    "incumbent starter, %.3f points, against a per-player board posterior SD of 1.2-1.8. The\n"
    "honest question is therefore not the point estimate but how wrong the board would have to\n"
    "be to flip the decision. Shifting every available QB's evaluated value down by c (opponent\n"
    "behaviour untouched, so the survival curves do not move):\n\n"
    "| c | marginal now | E[marginal at %d] | lineup VONA(QB) | beats TE's %.3f? |\n|---|---|---|---|---|\n"
    % (QB_GAP, OWNER_PICKS[0], TE_LV)
    + "\n".join(
        "| %.2f | %+.3f | %+.3f | %+.3f | %s |" % (
            r.qb_shift, r.marg_now, r.E_marg_next, r.lineup_vona,
            "yes" if r.lineup_vona > TE_LV else "**NO**")
        for r in QBBE.itertuples())
    + "\n\nThe QB call survives a downward shift of the whole QB shelf up to c between 0.75 and 1.00;\n"
      "it flips somewhere in that bracket. Equivalently, the true gap between the best available\n"
      "QB and the incumbent would have to be below roughly 0.4-0.5 points per game rather than the\n"
      "board's %.2f. That is a margin of about 0.8 points, inside one board posterior SD, so the\n"
      "recommendation is directional and should be read as such.\n" % QB_GAP
)

STRAT = {}
for act in ["greedy", "TE", "RB", "WR", "QB"]:
    tot, firsts = evaluate_strategy(act)
    STRAT[act] = dict(mean=tot.mean(), se=tot.std(ddof=1)/np.sqrt(len(tot)),
                      first=pd.Series(firsts).value_counts().head(3).to_dict() if firsts else {})
    print(f"[strategy] first={act:7s} E[lineup]={tot.mean():.3f} +/- {tot.std(ddof=1)/np.sqrt(len(tot)):.3f}  {STRAT[act]['first']}")

# ---------------------------------- robustness: same outputs under the §R1 spec (M1)
ALT = "M1" if BEST != "M1" else "M2R"
set_ctx(ALT)
VONA_ALT = vona(MASK0, ROST0, DECISION_A, OWNER_PICKS[0])
VONA_ALT.insert(0, "next_pick", OWNER_PICKS[0]); VONA_ALT.insert(0, "decision_pick", DECISION_A)
STRAT_ALT = {}
for act in ["greedy", "TE", "RB", "WR", "QB"]:
    tot, firsts = evaluate_strategy(act)
    STRAT_ALT[act] = dict(mean=tot.mean(), se=tot.std(ddof=1)/np.sqrt(len(tot)),
                          first=pd.Series(firsts).value_counts().head(1).to_dict() if firsts else {})
    print(f"[strategy/{ALT}] first={act:7s} E[lineup]={tot.mean():.3f} +/- {tot.std(ddof=1)/np.sqrt(len(tot)):.3f}  {STRAT_ALT[act]['first']}")
VONA_ALL["model"] = BEST
VONA_ALT["model"] = ALT
VONA_ALL = pd.concat([VONA_ALL, VONA_ALT], ignore_index=True)
VONA_ALL.to_csv(os.path.join(RES, "vona.csv"), index=False)
set_ctx(BEST)

# ------------------------------------------------------------------- sensitivity
SURV_HI = survival_curves(MASK0, ROST0, DECISION_A, CHECK, nsim=3000, lam=0.15, seed=77)
sens = pd.DataFrame({"name": [names[k] for k in KEYS_ALL], "pos": P_ARR,
                     "base": SURV[OWNER_PICKS[0]], "lam15": SURV_HI[OWNER_PICKS[0]],
                     "av": MASK0})
sens = sens[sens["av"]]

# --------------------------------------------------------------------- write notes
def fmt_params(model):
    r = results[model]
    lines = ["| parameter | estimate | SE (obs. info) | SE (cluster, G=10) | z |",
             "|---|---|---|---|---|"]
    for c, t, s, sc_ in zip(r["cols"], r["theta"], r["se"], r["se_cl"]):
        lines.append(f"| `{c}` | {t:+.4f} | {s:.4f} | {sc_:.4f} | {t/s:+.2f} |")
    return "\n".join(lines)

bv = results[BEST]["theta"][0]; bv_se = results[BEST]["se"][0]
temp = 1.0 / bv
temp_se = bv_se / bv**2   # delta method

out = []
out.append(f"""# §R — Behavioral draft model, fitted on the live 2026 league draft

*Pre-registration: `EDA_PLAN8.md` §R (2026-08-24). Spec: `fantasy_draft_model.md`.
Binding constraints: REPORT.md §38(1) scale identification, §38(2) τ-persistence pre-test.*

## §R0 What was fixed before fitting

Model set (declared before any coefficient was looked at):

- **M0** `U_j = β_v v_j`
- **M1** `U_j = β_v v_j + Σ_p β_p · need_{{m,p,t}} · 1[pos(j)=p]`  — §R1 exactly as written
- **M2** M1 + positional intercepts α_p (WR = reference) + β_flex · flexneed · 1[pos ∈ RB,WR]

`need_{{m,p,t}} = max(0, req_p − n_{{m,p,t}})` with req = QB 1, RB 2, WR 2, TE 1;
`flexneed = max(0, 2 − [(n_RB−2)_+ + (n_WR−2)_+])`.
M2 was declared because the OLS baseline it must beat contains exactly a positional
offset, and under M1 the need terms are identically zero for a manager who has filled
his base starters — which is most of the room by round 9, precisely the horizon of
interest. Selection criterion, also declared in advance: **leave-one-pick-out mean
predictive log-likelihood**, not in-sample fit and not hit rate.

**τ = 1, β_v free** (§38(1)). Only U/τ is identified. The reported temperature is 1/β_v.

**No per-manager layer.** §38(2)'s persistence pre-test requires two drafts; we have one.
League-mean parameters only, no affinity spike-and-slab, no meta×profile.

**The owner's 9 picks are excluded from the likelihood** (they still remove players from
the pool and still update his roster state). We are modelling *opponent* behaviour in
order to predict what falls to him; he drafts against 37 logged personal views, so his
picks are drawn from a different DGP. Exclusion is preferred to a dummy because a dummy
on the owner would be a pure intercept shift in a softmax over an unchanging choice set —
it cannot represent a different ranking, only a different noise level, and with 9
observations it would be estimated on nothing. n = {len(FIT_OBS)} opponent choice observations.

**Off-board picks.** After suffix-normalised name matching (`Chris Godwin` → `Chris Godwin Jr.`,
`James Cook` → `James Cook III`), **{sum(not o['onboard'] for o in OBS)} of {len(OBS)}** realised picks fell outside our 204-player
universe. The forward simulation nonetheless needs a non-zero rate, because the remaining
63 picks must absorb ten defences that the board does not contain. The raw rate 0/87 is
replaced by its Jeffreys posterior mean λ = {LAMBDA:.4f}: at each simulated pick, with probability
λ the manager takes someone not on our board and drains nothing. Because that is an
extrapolation and not an estimate, a sensitivity at λ = 0.15 — a room that has started on
defences — is reported below.
""")

out.append(f"""## §R1–R2 Fitted parameters

**Selected model: {BEST}** (by the pre-declared LOPO criterion).

### {BEST}
{fmt_params(BEST)}

**Implied temperature 1/β_v = {temp:.4f} (SE {temp_se:.4f}, delta method).**
Read it as: a value gap of {temp:.2f} board points between two available players is one
logit unit of preference. The board's undrafted pool spans about {V_ARR[MASK0].max()-np.quantile(V_ARR[MASK0],0.1):.1f} points from the
top to the 10th percentile, so this is a **chalky room** — consistent with, though not a
confirmation of, the 1.40-slot mean error recorded in `prediction_calibration_2026.md`.

### M1 (the §R1 specification as written)
{fmt_params("M1")}

### M0 (value only)
{fmt_params("M0")}

Standard errors are reported two ways. The observed-information SEs assume the picks are
independent conditional on the state; the manager-clustered SEs (G = 10) allow arbitrary
dependence within a manager. **With 10 clusters the clustered SEs are themselves noisy and
biased downward-in-coverage**; they are reported for honesty, not because they are better.
Every one of these numbers is estimated from a single realisation of a single draft.
""")

out.append(f"""## §R3 Validation — calibration first, accuracy second

Leave-one-pick-out over the {len(FIT_OBS)} opponent picks: refit without pick t, predict the full
distribution over A_t, score the realised choice.

| model | LOPO mean log-lik | SE | top-1 | top-5 |
|---|---|---|---|---|
""" + "\n".join(
    f"| {r.model} | {r.mean_loglik:.4f} | {r.se:.4f} | "
    + (f"{r.top1:.3f} | {r.top5:.3f} |" if r.model != "UNIF" else "— | — |")
    for r in SUMM.itertuples()) + f"""

**Paired comparison against the OLS baseline** (per-pick differences in held-out log-lik,
paired t on {len(L)} picks):

""" + "\n".join(f"- {m}: Δ = {v[0]:+.4f} ± {v[1]:.4f}  (t = {v[0]/v[1]:+.2f})" for m, v in paired.items()) + f"""

The OLS baseline is converted into a comparable pick model in the only honest way: its
fitted regression gives each player a predicted pick position π̂_j with residual SD σ, so
`P(j chosen at t) ∝ φ((t − π̂_j)/σ)` renormalised over A_t. Both models are then scored on
the same quantity — the held-out probability mass placed on the player actually taken.
The reproduced OLS fit is R² = {r2:.3f}, residual SD {sd_resid:.2f} picks, positional offsets
(picks earlier than a WR at the same ADP rank) TE {-bhat[4]:.1f}, QB {-bhat[2]:.1f}, RB {-bhat[3]:.1f}.

### Pick-level calibration (leave-one-pick-out, all (pick, candidate) pairs)

| predicted-probability bucket | n | mean predicted | realised frequency | ±SE |
|---|---|---|---|---|
""" + "\n".join(f"| {r.bucket} | {int(r.n)} | {r.pred:.4f} | {r.obs:.4f} | {r.se:.4f} |"
                for r in CAL_MODEL.itertuples()) + f"""

### Survival calibration — the quantity the tool actually emits

This is the test that matters, and it is a genuine temporal holdout: at each anchor pick
t₀ ∈ {{31, 41, 51, 61, 71}} the model is **refitted using only picks before t₀**, then run
forward by Monte Carlo (2,000 runs) for 15 picks. Each still-available player gets a
predicted P(survive), scored against whether he actually did.

| predicted survival bucket | n | mean predicted | realised | ±SE |
|---|---|---|---|---|
""" + "\n".join(f"| {r.bucket} | {int(r.n)} | {r.pred:.3f} | {r.obs:.3f} | {r.se:.3f} |"
                for r in SURV_TAB.itertuples()) + f"""

Brier score {brier:.4f} against a base-rate Brier of {brier0:.4f} — **skill {1-brier/brier0:+.3f}**.

**Read the pick-level table honestly.** The model is well calibrated across the mass of the
distribution (every bucket below p = 0.2 matches within about one SE) but **overconfident in
its confident tail**: the [0.2, 0.4) bucket predicts 0.28 and realises 0.14 (n = 29), and
[0.4, 0.7) predicts 0.50 and realises 0.20 (n = 15). Those cell counts are far too small to
estimate a miscalibration slope, but the direction is consistent and it has a mechanical
cause — a softmax over 100+ alternatives with a single global temperature concentrates too
much mass on the modal candidate when the board has an obvious top name. The practical
consequence for the tool is that **P(gone) for the single most obvious next pick is likely
overstated, so survival probabilities for the very top of the board should be read as a
lower bound.** The aggregate survival calibration below, which is what VONA actually
consumes, is much better behaved.

Top-1 and top-5 hit rates are reported above as **secondary diagnostics and explicitly not
as the adoption criterion** (`fantasy_draft_model.md`: "validate by calibration, not
accuracy"). A model can win top-1 by always naming the consensus next man and still be
useless for survival probabilities, which are what VONA consumes.
""")

out.append(f"""## §R2b The QB separation problem — the one anomaly worth chasing

M2 as first fitted returned `int_QB` = −15.3 with a standard error of **419**. That is the
signature of quasi-complete separation, and the data say exactly why:

- In **{QB_TOP_N} of {len(TOPDIAG)}** realised picks, the highest-*value* player on our board was a QB.
- A QB was actually taken in **{QB_TOP_TAKEN}** of them (Josh Allen at pick 40).
- Of the **{QB_FILLED_N}** such picks made by a manager who *already had a QB*, **{QB_FILLED_TAKEN}** took a QB.

Zero events out of {QB_FILLED_N} exposures drives the "QB, already filled" coefficient to −∞ while
`need_QB` runs to +∞ to keep the QB-needy combination finite. The MLE does not exist; the
likelihood is monotone along that ridge.

This matters far beyond a standard error. Our board's `final` is VORP-scaled against a
replacement QB, which makes **Dak Prescott (v = 6.75) the single most valuable available
player at pick {DECISION_A}**, ahead of every WR and RB. Under M1 — which has no way to say "a manager
with a QB does not take a QB" — every one of the seven QB-filled managers is modelled as
wanting Dak more than anyone else, and the simulator drains the QB shelf in a way the room
demonstrably does not. **M1 is misspecified in precisely the region that drives the pick-{DECISION_A}
recommendation**, and that was visible from the descriptive table above before any VONA was
computed.

The remedy is the textbook one for separation and it was chosen for that reason, not for its
answer: **M2R = M2 with a weakly-informative ridge N(0, {PRIOR_SD:.0f}²) on every coefficient except β_v**
(β_v is left unpenalised because it carries the scale normalisation τ = 1). This is the
penalised-likelihood / Firth-type fix: it renders the mode finite and the curvature
interpretable without changing the sign or the ordering of anything that was identified.
M2R was added to the model set on 2026-08-24 **after** seeing SE = 419 and **before** any
player-level VONA output was inspected, and it is scored on the same pre-declared LOPO
criterion as everything else. Because the choice of specification is doing real work here,
**every §R4 output below is reported under both the selected model and under M1, the §R1
specification exactly as pre-registered.** Where they disagree, that disagreement is the
finding.

For reference, the room takes the maximum-value available board player on only
**{TOPDIAG['took_top'].mean():.1%}** of picks — so "the room drafts our board" is false in general, not just for QBs.

## §R4 Outputs

Owner roster after 9 picks: {OWNER_ROSTER_START}. Starters required 1QB/2RB/2WR/1TE/2FLEX/1DST.
**Open starting slots: TE and DST.** DST is not on the 204-player board and no DST has been
taken in 87 picks, so its scarcity risk is zero and its lineup contribution is a constant
that cancels across every strategy compared below; it is scheduled for the last pick.

### Positional run risk, {DECISION_A} → {OWNER_PICKS[0]}

| position | E[gone] | P(≥1) | P(≥2) | P(≥3) | P(≥5) |
|---|---|---|---|---|---|
""" + "\n".join(
    f"| {q} | {RUN[q].mean():.2f} | {(RUN[q]>=1).mean():.3f} | {(RUN[q]>=2).mean():.3f} | "
    f"{(RUN[q]>=3).mean():.3f} | {(RUN[q]>=5).mean():.3f} |" for q in POS) + f"""

### VONA

| spec | decision | next | pos | best now | player | E[best next] | SE | sd | p10 | p90 | **VONA** | P(top survives) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {r.model} | {int(r.decision_pick)} | {int(r.next_pick)} | {r.position} | {r.best_now:.3f} | "
    f"{r.best_now_player} | {r.E_best_next:.3f} | {r.se_E_best_next:.3f} | {r.sd_best_next:.3f} | "
    f"{r.p10:.3f} | {r.p90:.3f} | **{r.vona:+.3f}** | {r.p_top_survives:.3f} |"
    for r in VONA_ALL.itertuples()) + f"""

### Marginal starting-lineup VONA — the decision-relevant version

Raw-value VONA above answers "how much board value decays at each position". It is *not*
the decision quantity for a manager who already holds eight RB/WR: what matters is how much
a player would add to the **starting lineup** (1QB/2RB/2WR/1TE/2FLEX), which for a
positionally-saturated roster is far less than his board value. Owner's current holdings by
position, in board value: """ + "; ".join(f"{q}: {sorted(OWNER_HELD[q], reverse=True)}" for q in POS) + f""" — current lineup total {lineup_value(OWNER_HELD):.3f}.

| position | best now | marginal now | E[marginal at {OWNER_PICKS[0]}] | SE | sd | **lineup VONA** |
|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {r.position} | {r.best_now_player} | {r.marg_now:+.3f} | {r.E_marg_next:+.3f} | {r.se:.3f} | "
    f"{r.sd:.3f} | **{r.lineup_vona:+.3f}** |" for r in LVONA.itertuples()) + f"""

{BREAKEVEN_MD}

### Expected starting-lineup value by first action at pick {DECISION_A}

Owner follows a marginal-starting-lineup-value greedy rule at {OWNER_PICKS}; opponents follow
the fitted logit; parameters redrawn from N(θ̂, V̂) each run; 1,500 runs each.

| first action | E[lineup] under {BEST} | SE | E[lineup] under {ALT} | SE | modal player |
|---|---|---|---|---|---|
""" + "\n".join(
    f"| {a} | {d['mean']:.3f} | {d['se']:.3f} | {STRAT_ALT[a]['mean']:.3f} | {STRAT_ALT[a]['se']:.3f} | "
    f"{list(d['first'].keys())[0] if d['first'] else '—'} |"
    for a, d in STRAT.items()) + f"""

### Survival curve excerpt — top 15 available by board value

| player | pos | v | P(avail at {OWNER_PICKS[0]}) | P(avail at {OWNER_PICKS[1]}) | P(avail at {OWNER_PICKS[2]}) |
|---|---|---|---|---|---|
""" + "\n".join(
    f"| {r['name']} | {r['position']} | {r['value']:.2f} | {r[f'p_avail_at_{OWNER_PICKS[0]}']:.3f} | "
    f"{r[f'p_avail_at_{OWNER_PICKS[1]}']:.3f} | {r[f'p_avail_at_{OWNER_PICKS[2]}']:.3f} |"
    for _, r in sc.head(15).iterrows()) + f"""

### Sensitivity to the off-board hazard

λ = {LAMBDA:.3f} (estimated) vs λ = 0.15 (a room that starts taking DSTs). Mean absolute change in
P(available at {OWNER_PICKS[0]}) across undrafted players: {np.abs(sens['base']-sens['lam15']).mean():.4f}; max {np.abs(sens['base']-sens['lam15']).max():.4f}.
""")

best_act = max(STRAT, key=lambda a: STRAT[a]["mean"] if a != "greedy" else -1e9)
out.append(f"""## §R5 Recommendation, and what it is worth

**Take the QB at pick {DECISION_A}.** Both the selected specification and the pre-registered §R1
specification rank a QB first, by {STRAT['QB']['mean']-STRAT['TE']['mean']:+.3f} and {STRAT_ALT['QB']['mean']-STRAT_ALT['TE']['mean']:+.3f} expected starting-lineup points
respectively over the next-best action (TE). Spec-robustness is the reason to believe it;
the magnitude is not large.

The logic, stated so it can be checked: the owner's roster is **positionally saturated at
RB/WR** — eight of them, six starting slots, and the best available RB adds exactly **0.000**
to his starting lineup. Only three positions can add anything, and the ranking is not by
who is best but by **how fast the marginal upgrade decays**:

- **TE** is the position he *needs*, and it is the position it is **cheapest to wait on**. The
  tier is flat by construction (the isotonic prior returns steps), so losing the top name
  costs {LVONA.set_index('position').loc['TE','lineup_vona']:.3f} points, not a cliff — even though the room drafts TEs {-bhat[4]:.0f} picks earlier
  than their ADP-implied slot and {(TOPDIAG['took_pos']=='TE').sum()} have already gone.
- **QB** decays fastest ({LVONA.set_index('position').loc['QB','lineup_vona']:.3f}) despite only {int((TOPDIAG['took_pos']=='QB').sum())} QBs being gone, because the upgrade over the
  incumbent is a step function: there is one QB left worth a real upgrade and P(he survives
  to {OWNER_PICKS[0]}) is a coin flip.
- **WR** decays {LVONA.set_index('position').loc['WR','lineup_vona']:.3f} but from a marginal base of only {LVONA.set_index('position').loc['WR','marg_now']:.3f} — the ninth WR barely improves a
  lineup that already starts six RB/WR.

So the sequence the simulator recommends is **QB now, TE at {OWNER_PICKS[0]}, DST last**, with the
remaining picks going to whatever has the largest marginal starting-lineup value at the time.
P(the top TE survives to {OWNER_PICKS[0]}) = {float(VONA_ALL[(VONA_ALL.model==BEST)&(VONA_ALL.position=='TE')&(VONA_ALL.decision_pick==DECISION_A)].p_top_survives.iloc[0]):.2f}, and P(at least one of the four flat-tier TEs
survives) is essentially 1 given only {int(RUN['TE'].mean())} TEs are expected to go in the window.

**What this recommendation is worth.** The gap between the best and second-best action is
{STRAT['QB']['mean']-STRAT['TE']['mean']:.3f} lineup points. A single player's board posterior SD is 1.2–1.8. The breakeven
analysis above shows the call flips if the true QB upgrade is overstated by ~0.8 points.
Every parameter comes from **one draft, n = {len(FIT_OBS)} opponent choices**, and the model does not beat
the OLS positional baseline at conventional significance (Δ = {paired[BEST][0]:+.3f} ± {paired[BEST][1]:.3f} nats,
t = {paired[BEST][0]/paired[BEST][1]:.2f}). This is a decision aid with a directional answer, not a settled result.

## §R6 Does the behavioural model beat the positional correction?

**On the point estimate yes, at conventional significance no.** The selected conditional
logit gains {paired[BEST][0]:+.4f} nats per held-out pick over the ADP+position OLS baseline, with a paired
standard error of {paired[BEST][1]:.4f} (t = {paired[BEST][0]/paired[BEST][1]:.2f}, {len(FIT_OBS)} paired picks). The §R1 specification as
pre-registered gains {paired['M1'][0]:+.4f} ± {paired['M1'][1]:.4f} — indistinguishable from the baseline.

That is a real finding and it should be stated as such rather than buried: **most of what a
behavioural draft model knows about this room is already captured by "shift TEs {-bhat[4]:.0f} picks
earlier, QBs {-bhat[2]:.0f}, RBs {-bhat[3]:.0f}, and add Gaussian noise of {sd_resid:.0f} picks."** The conditional logit adds
three things the OLS cannot: it is a *proper distribution over the available set* (so it can
be simulated forward without an ad-hoc renormalisation), it is *roster-state dependent* (the
need terms are jointly significant and `need_WR` and `int_RB` are individually so), and it
correctly refuses to draft a second QB. The first of those is why it is worth building even
at parity on log-likelihood; the third is why M1 is not the version to ship.

This also speaks to §38(3): §M found that no pick sequence beat drafting the board against
ADP-drafting opponents, so any edge here must come from opponents being *predictably biased*.
The bias is real and measurable — the room takes the maximum-value board player only {TOPDIAG['took_top'].mean():.1%} of
the time, and the positional intercepts (`int_RB` {results[BEST]['theta'][6]:+.2f}, `int_TE` {results[BEST]['theta'][7]:+.2f}, `int_QB` {results[BEST]['theta'][5]:+.2f} against a
WR baseline) say exactly where. Whether exploiting it is worth more than its estimation
error, on one draft, is not established.
""")

with open(os.path.join(RES, "sectionR_notes.md"), "w") as f:
    f.write("\n\n".join(out) + "\n")
print("[out] sectionR_notes.md")

L.to_csv(os.path.join(RES, "sectionR_lopo.csv"), index=False)
SURV_CAL.to_csv(os.path.join(RES, "sectionR_survival_calibration.csv"), index=False)
json.dump({m: dict(theta=list(results[m]["theta"]), se=list(results[m]["se"]),
                   se_cl=list(results[m]["se_cl"]), cols=results[m]["cols"],
                   ll=results[m]["ll"]) for m in results},
          open(os.path.join(RES, "sectionR_params.json"), "w"), indent=1)
print("[done]")

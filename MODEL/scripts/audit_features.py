r"""
Audit 6: feature-space dimensionality, feature selection, and whether opponent
adjustment / the two-stage prior earn their complexity.

PRE-REGISTERED PROTOCOL (fixed before any result below was inspected)
=====================================================================
Selection data. Expanding-window forward chaining inside the TRAIN block only:
for each season s in 2016..2022, fit the logit on seasons 2014..s-1 and predict
season s. Pool the seven held-out seasons. That is ~1870 genuinely out-of-sample
games, versus the 272 of a single validation season -- the small validation
season is what produced the earlier overfitting error, so it is not used to
select anything here.

Criterion. Pooled out-of-sample log loss over the seven held-out seasons.
Accuracy and Brier are reported as secondary and never break a tie.

Uncertainty. Paired bootstrap over games, 2000 resamples, on the DIFFERENCE in
log loss between two specs. A difference whose 95% interval covers 0 is declared
noise regardless of its point estimate. On ~1870 games the noise floor on a
log-loss difference is roughly +-0.006, so anything under ~0.005 is not a result.

Multiple testing. The "does candidate X add beyond the EPA core" battery is one
family of ~40 tests; Benjamini-Hochberg FDR at q=0.10 is applied across it and
reported alongside the raw p-values.

Confirmation. 2023 (VALID) is scored ONCE for the final recommended spec against
the incumbent. 2024-2025 (HOLDOUT) is scored ONCE at the very end, and no
decision in this script depends on it.

Dimensionality. Horn parallel analysis on the TRAIN correlation matrix of the
net differentials (eigenvalues compared against the 95th percentile of
eigenvalues from column-permuted data, which destroys correlation but preserves
marginals), plus a forward-chaining sequential-PC logit to count how many
components carry PREDICTIVE, not merely covariance, dimensions.
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import log_loss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from audit_quality import METRICS

RES = os.path.join(HERE, "..", "results")
CV_SEASONS = list(range(2016, 2023))
TRAIN_ALL = list(range(2014, 2023))
VALID = [2023]
HOLDOUT = [2024, 2025]
RNG = np.random.default_rng(20260810)
NBOOT = 2000

NET = [f"net_{m}_diff" for m in METRICS]
OFFDEF = [f"{s}_{m}_diff" for m in METRICS for s in ("off", "def")]

out_lines = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out_lines.append(s)


# ---------------------------------------------------------------------------
def load(est="v2adj"):
    d = pd.read_csv(os.path.join(HERE, "..", "data", f"audit_model_table_{est}.csv"))
    return d[d.home_win != 0.5].reset_index(drop=True)


def zscore(d, cols, ref_seasons):
    ref = d[d.season.isin(ref_seasons)]
    out = d.copy()
    for c in cols:
        mu, sd = ref[c].mean(), ref[c].std()
        out[c] = (d[c] - mu) / (sd if sd > 0 else 1.0)
    return out


def logit_fit(tr, cols):
    X = sm.add_constant(tr[cols].astype(float), has_constant="add")
    return sm.Logit(tr.home_win.astype(int), X).fit(disp=0, maxiter=200)


def cv_predict(d, cols, seasons=CV_SEASONS):
    """Expanding-window forward chaining. Returns (y, p, season) pooled."""
    ys, ps, ss = [], [], []
    for s in seasons:
        tr = d[d.season < s]
        te = d[d.season == s]
        if tr.empty or te.empty:
            continue
        z = zscore(d, cols, list(range(2014, s)))
        trz, tez = z[z.season < s], z[z.season == s]
        f = logit_fit(trz, cols)
        X = sm.add_constant(tez[cols].astype(float), has_constant="add")
        ps.append(np.clip(f.predict(X[f.params.index]).to_numpy(), 1e-6, 1 - 1e-6))
        ys.append(te.home_win.astype(int).to_numpy())
        ss.append(te.season.to_numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(ss)


def ll(y, p):
    return float(log_loss(y, p, labels=[0, 1]))


def acc(y, p):
    return float(((p > 0.5) == (y == 1)).mean())


def boot_diff(y, pa, pb, nboot=NBOOT):
    """Paired bootstrap on log-loss difference (a - b). Negative = a better."""
    la = -(y * np.log(pa) + (1 - y) * np.log(1 - pa))
    lb = -(y * np.log(pb) + (1 - y) * np.log(1 - pb))
    dif = la - lb
    n = len(dif)
    idx = RNG.integers(0, n, size=(nboot, n))
    bs = dif[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    # two-sided bootstrap p: proportion of resamples on the wrong side of 0
    p = 2 * min((bs >= 0).mean(), (bs <= 0).mean())
    return float(dif.mean()), float(lo), float(hi), float(min(p, 1.0))


def bh(pvals, q=0.10):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1)) / n
    passed = p[order] <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    keep = np.zeros(n, bool)
    keep[order[:k]] = True
    return keep


# ===========================================================================
def section_dimensionality(d):
    P("\n" + "=" * 78)
    P("A. HOW MANY REAL DIMENSIONS ARE IN THE FEATURE SPACE?")
    P("=" * 78)
    tr = d[d.season.isin(TRAIN_ALL)]
    X = tr[NET].to_numpy(float)
    X = X[~np.isnan(X).any(axis=1)]
    Z = (X - X.mean(0)) / X.std(0)
    C = np.corrcoef(Z, rowvar=False)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]

    # Horn parallel analysis: permute each column independently
    sims = np.empty((200, len(ev)))
    for b in range(200):
        Zp = np.column_stack([RNG.permutation(Z[:, j]) for j in range(Z.shape[1])])
        sims[b] = np.sort(np.linalg.eigvalsh(np.corrcoef(Zp, rowvar=False)))[::-1]
    crit = np.percentile(sims, 95, axis=0)
    n_horn = int(np.sum(ev > crit))

    P(f"{len(NET)} net differentials, {len(X)} train games")
    P(f"variance explained by PC1..PC6: "
      f"{np.round(100 * ev[:6] / ev.sum(), 1).tolist()} %")
    P(f"cumulative:                     "
      f"{np.round(100 * np.cumsum(ev[:6]) / ev.sum(), 1).tolist()} %")
    P(f"Horn parallel analysis retains {n_horn} components "
      f"(eigenvalue > 95th pct of permuted null)")
    P("  eigen vs null (first 8): "
      + ", ".join(f"{e:.2f}/{c:.2f}" for e, c in zip(ev[:8], crit[:8])))

    # loadings of PC1
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    V = V[:, order]
    l1 = pd.Series(V[:, 0], index=NET).sort_values(key=np.abs, ascending=False)
    P("\nPC1 loadings (|top 12|):")
    P(l1.head(12).round(3).to_string())
    l2 = pd.Series(V[:, 1], index=NET).sort_values(key=np.abs, ascending=False)
    P("\nPC2 loadings (|top 10|):")
    P(l2.head(10).round(3).to_string())
    l3 = pd.Series(V[:, 2], index=NET).sort_values(key=np.abs, ascending=False)
    P("\nPC3 loadings (|top 10|):")
    P(l3.head(10).round(3).to_string())

    # predictive dimensionality: sequential PCs under forward-chaining CV
    P("\nPREDICTIVE dimensionality: cumulative PCs as logit features (CV log loss)")
    P("PCs are refit inside each CV fold on training seasons only.")
    dd = d.dropna(subset=NET).reset_index(drop=True)
    rows = []
    prev_p = None
    for k in range(1, 11):
        ys, ps, ss = [], [], []
        for s in CV_SEASONS:
            trf = dd[dd.season < s]
            tef = dd[dd.season == s]
            mu, sd = trf[NET].mean(), trf[NET].std()
            A = ((trf[NET] - mu) / sd).to_numpy()
            B = ((tef[NET] - mu) / sd).to_numpy()
            _, _, Vt = np.linalg.svd(A - A.mean(0), full_matrices=False)
            W = Vt[:k].T
            trp = pd.DataFrame(A @ W, columns=[f"pc{i}" for i in range(k)])
            trp["home_win"] = trf.home_win.to_numpy()
            tep = pd.DataFrame(B @ W, columns=[f"pc{i}" for i in range(k)])
            f = logit_fit(trp, list(trp.columns[:k]))
            Xt = sm.add_constant(tep, has_constant="add")
            ps.append(np.clip(f.predict(Xt[f.params.index]).to_numpy(), 1e-6, 1 - 1e-6))
            ys.append(tef.home_win.astype(int).to_numpy())
        y = np.concatenate(ys); p = np.concatenate(ps)
        row = {"n_pc": k, "cv_logloss": ll(y, p), "cv_acc": acc(y, p)}
        if prev_p is not None:
            m, lo, hi, pv = boot_diff(y, p, prev_p)
            row.update({"d_vs_prev": m, "lo": lo, "hi": hi, "boot_p": pv})
        rows.append(row)
        prev_p = p
    r = pd.DataFrame(rows)
    P(r.round(4).to_string(index=False))
    r.to_csv(os.path.join(RES, "audit_pc_dimensionality.csv"), index=False)
    return n_horn


def section_univariate(d):
    P("\n" + "=" * 78)
    P("B. UNIVARIATE SCREEN (each net differential alone, forward-chaining CV)")
    P("=" * 78)
    rows = []
    base_y, base_p, _ = cv_predict(d, [])
    P(f"intercept-only CV log loss = {ll(base_y, base_p):.4f}")
    for c in NET:
        dd = d.dropna(subset=[c])
        y, p, _ = cv_predict(dd, [c])
        y0, p0, _ = cv_predict(dd, [])
        m, lo, hi, pv = boot_diff(y, p, p0)
        rows.append({"feature": c, "cv_logloss": ll(y, p), "cv_acc": acc(y, p),
                     "d_vs_null": m, "lo": lo, "hi": hi, "boot_p": pv})
    r = pd.DataFrame(rows).sort_values("cv_logloss")
    r.to_csv(os.path.join(RES, "audit_univariate.csv"), index=False)
    P(r.round(4).to_string(index=False))
    return r


def section_incremental(d, core):
    P("\n" + "=" * 78)
    P(f"C. INCREMENTAL VALUE BEYOND THE CORE  {core}")
    P("Does any candidate add out-of-sample information once the EPA core is in?")
    P("=" * 78)
    dd = d.dropna(subset=NET).reset_index(drop=True)
    y0, p0, _ = cv_predict(dd, core)
    P(f"core CV log loss = {ll(y0, p0):.4f}   acc = {acc(y0, p0):.4f}")
    rows = []
    for c in NET:
        if c in core:
            continue
        y, p, _ = cv_predict(dd, core + [c])
        m, lo, hi, pv = boot_diff(y, p, p0)
        rows.append({"added": c, "cv_logloss": ll(y, p), "delta": m,
                     "lo": lo, "hi": hi, "boot_p": pv})
    r = pd.DataFrame(rows).sort_values("delta")
    r["bh_q10"] = bh(r.boot_p.to_numpy(), 0.10)
    r.to_csv(os.path.join(RES, "audit_incremental.csv"), index=False)
    P(r.round(4).to_string(index=False))
    P(f"\nsurviving BH-FDR q=0.10 across {len(r)} candidates: "
      f"{r[r.bh_q10].added.tolist() or 'NONE'}")
    return r


def section_forward(d, pool, max_k=6):
    P("\n" + "=" * 78)
    P("D. GREEDY FORWARD SELECTION ON FORWARD-CHAINING CV")
    P("stop when the best remaining addition is not significant by paired bootstrap")
    P("=" * 78)
    dd = d.dropna(subset=pool).reset_index(drop=True)
    chosen, hist = [], []
    y_prev, p_prev, _ = cv_predict(dd, [])
    cur = ll(y_prev, p_prev)
    P(f"start (intercept only): {cur:.4f}")
    for step in range(max_k):
        best = None
        for c in pool:
            if c in chosen:
                continue
            y, p, _ = cv_predict(dd, chosen + [c])
            v = ll(y, p)
            if best is None or v < best[1]:
                best = (c, v, p, y)
        c, v, p, y = best
        m, lo, hi, pv = boot_diff(y, p, p_prev)
        sig = hi < 0
        P(f"step {step+1}: + {c:28s} ll={v:.4f}  delta={m:+.4f} "
          f"[{lo:+.4f},{hi:+.4f}] p={pv:.3f} {'KEEP' if sig else 'STOP (noise)'}")
        hist.append({"step": step + 1, "feature": c, "cv_logloss": v,
                     "delta": m, "lo": lo, "hi": hi, "boot_p": pv, "kept": sig})
        if not sig:
            break
        chosen.append(c)
        p_prev, y_prev, cur = p, y, v
    pd.DataFrame(hist).to_csv(os.path.join(RES, "audit_forward_selection.csv"),
                              index=False)
    P(f"\nselected: {chosen}")
    return chosen


def section_estimators(core):
    P("\n" + "=" * 78)
    P("E. IS OPPONENT ADJUSTMENT / THE TWO-STAGE PRIOR EARNING ITS COMPLEXITY?")
    P("same features, same CV, three estimators of the same latent quantity")
    P("=" * 78)
    preds = {}
    for est in ("v2adj", "v2raw", "v1adj"):
        d = load(est).dropna(subset=core).reset_index(drop=True)
        y, p, _ = cv_predict(d, core)
        preds[est] = (y, p)
        P(f"  {est}: CV log loss {ll(y, p):.4f}   acc {acc(y, p):.4f}")
    P("")
    for a, b, label in [("v2adj", "v2raw", "opponent adjustment (adj - raw)"),
                        ("v2adj", "v1adj", "two-stage prior (v2 - v1)")]:
        ya, pa = preds[a]
        yb, pb = preds[b]
        n = min(len(pa), len(pb))
        m, lo, hi, pv = boot_diff(ya[:n], pa[:n], pb[:n])
        verdict = "REAL" if hi < 0 else ("HARMFUL" if lo > 0 else "NOISE")
        P(f"  {label}: {m:+.4f} [{lo:+.4f},{hi:+.4f}] p={pv:.3f}  -> {verdict}")
    return preds


def section_offdef(d, core):
    P("\n" + "=" * 78)
    P("F. NET vs SEPARATE OFF/DEF (is collapsing to a net differential justified?)")
    P("=" * 78)
    dd = d.dropna(subset=NET).reset_index(drop=True)
    y0, p0, _ = cv_predict(dd, core)
    split = []
    for c in core:
        m = c.replace("net_", "").replace("_diff", "")
        split += [f"off_{m}_diff", f"def_{m}_diff"]
    y1, p1, _ = cv_predict(dd, split)
    m, lo, hi, pv = boot_diff(y1, p1, p0)
    P(f"  net form   : {ll(y0, p0):.4f}")
    P(f"  split form : {ll(y1, p1):.4f}")
    P(f"  split - net: {m:+.4f} [{lo:+.4f},{hi:+.4f}] p={pv:.3f}")
    P("  (negative would mean the split is better; a CI covering 0 says the")
    P("   equal-weights restriction O and D costs nothing)")


def main():
    d = load("v2adj")
    P(f"games {len(d)} (ties dropped), seasons {d.season.min()}-{d.season.max()}")
    P(f"CV seasons {CV_SEASONS[0]}-{CV_SEASONS[-1]}, "
      f"{d.season.isin(CV_SEASONS).sum()} held-out games")

    n_horn = section_dimensionality(d)
    uni = section_univariate(d)

    core = ["net_epa_noto_diff", "net_to_rate_diff"]
    section_incremental(d, core)
    chosen = section_forward(d, NET, max_k=6)
    section_estimators(["net_epa_noto_diff", "net_to_rate_diff"])
    section_offdef(d, ["net_epa_noto_diff", "net_to_rate_diff"])

    with open(os.path.join(RES, "audit_features.txt"), "w") as f:
        f.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()

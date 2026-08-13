"""Shared helpers for the audit scripts (audit_*.py).

Everything here is deliberately dependency-light: load the model table, join the
market lines from data/games/games.csv, and provide scoring / bootstrap tools.

Conventions
-----------
spread_home : closing spread, HOME-FAVOURED POSITIVE (verified identical to
              nflverse `spread_line`, max abs diff 0.0).
margin      : home_score - away_score.
home_win    : 1 / 0, with 0.5 for the 11 ties -- ties are dropped everywhere.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
TABLE = os.path.join(ROOT, "data", "model_table_v2.csv")
GAMES = os.path.join(ROOT, "data", "games", "games.csv")
RESULTS = os.path.join(ROOT, "results")

TRAIN = list(range(2014, 2023))
VALID = [2023]
HOLDOUT = [2024, 2025]
ALL_SEASONS = list(range(2014, 2026))

SPEC_B = ["off_epa_noto_diff", "def_epa_noto_diff", "to_margin_diff"]


# ----------------------------------------------------------------- data
def load(drop_ties: bool = True) -> pd.DataFrame:
    d = pd.read_csv(TABLE)
    g = pd.read_csv(GAMES)[
        ["game_id", "home_moneyline", "away_moneyline", "home_score",
         "away_score", "roof", "location", "temp", "wind"]
    ]
    d = d.merge(g, on="game_id", how="left")
    if drop_ties:
        d = d[d.home_win != 0.5].copy()
    d["home_win"] = d.home_win.astype(int)
    return d.reset_index(drop=True)


def standardise(d: pd.DataFrame, cols, fit_seasons) -> pd.DataFrame:
    """z-score `cols` using moments from `fit_seasons` only."""
    tr = d[d.season.isin(fit_seasons)]
    out = d.copy()
    for c in cols:
        mu, sd = tr[c].mean(), tr[c].std()
        out[c] = (d[c] - mu) / (sd if sd > 0 else 1.0)
    return out


# ------------------------------------------------------------ moneyline
def american_to_dec(ml):
    ml = np.asarray(ml, dtype=float)
    return np.where(ml > 0, 1 + ml / 100.0, 1 + 100.0 / np.abs(ml))


def devig(ml_home, ml_away, method="multiplicative"):
    """Return de-vigged P(home win) from the two American moneylines.

    multiplicative : q_i / sum(q) -- proportional, the usual default.
    additive       : q_i - (sum(q)-1)/2 -- splits the overround equally.
    power          : solve q_i^k so that sum = 1 (k>1); favourite-longshot aware.
    shin           : Shin (1993) insider-trading model, solves for z.
    """
    dh, da = american_to_dec(ml_home), american_to_dec(ml_away)
    qh, qa = 1.0 / dh, 1.0 / da
    s = qh + qa
    if method == "multiplicative":
        return qh / s
    if method == "additive":
        return qh - (s - 1.0) / 2.0
    if method == "power":
        out = np.empty_like(qh)
        for i in range(len(qh)):
            f = lambda k: qh[i] ** k + qa[i] ** k - 1.0
            try:
                k = brentq(f, 0.5, 5.0)
                out[i] = qh[i] ** k
            except Exception:
                out[i] = qh[i] / s[i]
        return out
    if method == "shin":
        out = np.empty_like(qh)
        for i in range(len(qh)):
            def f(z):
                den = 2.0 - z
                p = [(np.sqrt(z * z + 4 * (1 - z) * q * q / s[i]) - z) / den
                     for q in (qh[i], qa[i])]
                return p[0] + p[1] - 1.0
            try:
                z = brentq(f, 1e-9, 0.35)
                den = 2.0 - z
                out[i] = (np.sqrt(z * z + 4 * (1 - z) * qh[i] ** 2 / s[i]) - z) / den
            except Exception:
                out[i] = qh[i] / s[i]
        return out
    raise ValueError(method)


def spread_to_prob(spread_home, sigma=13.5):
    """Fixed normal-CDF map, no fitting: P(home win) = Phi(spread / sigma)."""
    return norm.cdf(np.asarray(spread_home, dtype=float) / sigma)


# --------------------------------------------------------------- scoring
def logloss(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def scores(y, p):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    from sklearn.metrics import roc_auc_score
    return {"n": len(y), "logloss": logloss(y, p), "brier": brier(y, p),
            "acc": float(((p > 0.5) == (y == 1)).mean()),
            "auc": float(roc_auc_score(y, p))}


def paired_boot(y, p_a, p_b, metric=logloss, B=4000, seed=0, block=None):
    """Bootstrap CI for metric(a) - metric(b), paired on games.

    `block`: array of group labels (e.g. season-week) for a blocked bootstrap;
    resampling whole blocks preserves within-week dependence.
    """
    rng = np.random.default_rng(seed)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    n = len(y)
    if block is None:
        idx_pool = [np.array([i]) for i in range(n)]
    else:
        block = np.asarray(block)
        idx_pool = [np.where(block == b)[0] for b in np.unique(block)]
    K = len(idx_pool)
    diffs = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, K, K)
        idx = np.concatenate([idx_pool[j] for j in pick])
        diffs[b] = metric(y[idx], p_a[idx]) - metric(y[idx], p_b[idx])
    point = metric(y, p_a) - metric(y, p_b)
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), diffs


def calibration_table(y, p, bins=None):
    if bins is None:
        bins = np.array([0, .2, .3, .4, .5, .6, .7, .8, 1.0])
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.digitize(p, bins) - 1
    rows = []
    for k in range(len(bins) - 1):
        m = idx == k
        if m.sum() == 0:
            continue
        rows.append({"bin": f"[{bins[k]:.2f},{bins[k+1]:.2f})", "n": int(m.sum()),
                     "p_mean": p[m].mean(), "y_mean": y[m].mean(),
                     "gap": y[m].mean() - p[m].mean()})
    return pd.DataFrame(rows)


def murphy(y, p):
    """Brier decomposition: REL - RES + UNC (10 equal-count bins)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    n = len(y)
    ybar = y.mean()
    qs = np.quantile(p, np.linspace(0, 1, 11))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    idx = np.digitize(p, qs) - 1
    rel = res = 0.0
    for k in range(10):
        m = idx == k
        if m.sum() == 0:
            continue
        w = m.sum() / n
        rel += w * (p[m].mean() - y[m].mean()) ** 2
        res += w * (y[m].mean() - ybar) ** 2
    return {"REL": rel, "RES": res, "UNC": ybar * (1 - ybar),
            "brier": brier(y, p)}

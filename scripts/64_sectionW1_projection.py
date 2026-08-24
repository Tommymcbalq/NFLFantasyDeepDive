"""§W1 (L1) — the projection engine, LOSO against mu_hat.

Estimators: (a) ridge (alpha by grouped inner CV on training folds only)
            (b) HistGradientBoosting, hyperparameters declared and NOT tuned
            (c) hierarchical partial pooling across WR/RB (shared block + penalised
                position-deviation block, kappa by the same inner CV)
Scopes:     P0 inputs only (no function of own past PPR points)
            P1 inputs + mu_hat
Benchmarks: mu_hat (binding), mu_cal = OLS ppg ~ 1 + mu_hat on training folds
Also:       eq.(7) substitution, theta* = (1-B)*yhat + B*m_hat

Pre-registration: results/sectionW1_notes.md (written before any fit).
Outputs: results/sectionW1_loso.csv, sectionW1_predictions.csv,
         sectionW1_coefficients.csv, sectionW1_holdout.csv
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.simplefilter("ignore")
ROOT = Path("/Users/thomasmcnamee/NFL")
YEARS = list(range(2015, 2025))
ALPHAS = np.logspace(-2, 4, 25)
KAPPAS = [0.05, 0.2, 1.0, 5.0]
RNG = np.random.default_rng(20260824)

# ------------------------------------------------------------------ feature sets
OPP = ["h_targets_pg", "h_carries_pg", "h_touches_pg", "h_air_yards_pg",
       "h_target_share_full", "h_air_yards_share_full", "h_carry_share_full",
       "h_wopr_full", "h_u_rec", "h_u_rush", "h_u_tot", "G_last", "G_wtd",
       "avail_wtd"]
EFF_GATED = ["h_adot", "h_ypr", "h_catch_rate", "h_racr_w"]      # §4 gate PASS
EFF_REJECTED = ["h_ypt", "h_yac_per_rec", "h_td_per_tgt",
                "h_rec_epa_per_tgt", "h_ypc"]                    # gate FAIL, sens only
# LEAK NOTE (found 2026-08-24, before adoption): the FFC historical ADP `team` field is an
# END-OF-SEASON label, not a draft-day label -- among the 42 panel rows whose player played
# for two teams in year Y it matches his FINAL team 88% of the time and his week-1 team 7%.
# So env_* (year-Y team's prior offence) and team_change are contaminated for traded players.
# The clean, strictly preseason-knowable environment is the PRIOR-SEASON team's offence.
ENVF = ["oldenv_tm_pass_att_pg", "oldenv_tm_tgt_pg", "oldenv_tm_ay_pg",
        "oldenv_tm_car_pg", "oldenv_tm_plays_pg", "oldenv_tm_pass_rate",
        "oldenv_tm_td_pg", "oldenv_tm_pass_yds_pg"]
ENVF_LEAKY = ["env_tm_pass_att_pg", "env_tm_tgt_pg", "env_tm_ay_pg", "env_tm_car_pg",
              "env_tm_plays_pg", "env_tm_pass_rate", "env_tm_td_pg",
              "env_tm_pass_yds_pg", "team_change"]
STRUCT = ["expr", "log_draft_pick"]          # age enters as a spline, built per fold

ADV_WR = ["snap_share", "pass_snap_share", "routes_proxy_pg", "tprr_proxy",
          "yprr_proxy", "rz_targets_pg", "i10_targets_pg", "ez_targets_pg",
          "third_down_targets_pg", "deep_targets_pg", "deep_target_rate",
          "rz_target_share_of_own", "ngs_avg_separation", "ngs_avg_cushion",
          "ngs_avg_yac_above_expectation", "ngs_percent_share_of_intended_air_yards",
          "pfr_adot", "pfr_ybc_per_rec", "pfr_yac_per_rec", "pfr_drop_pct",
          "pfr_rec_per_broken_tackle", "target_epa"]
ADV_RB = ["snap_share", "pass_snap_share", "run_snap_share", "routes_proxy",
          "tprr_proxy", "gl5_carries_pg", "gl10_carries_pg", "third_down_carries_pg",
          "explosive_run_rate", "rush_epa_per_att", "ngs_ryoe_per_att",
          "ngs_efficiency", "ngs_percent_attempts_gte_eight_defenders",
          "pfr_ybc_per_att", "pfr_yac_per_att", "pfr_att_per_broken_tackle",
          "opportunity_pg", "rz_targets_pg"]
TEAMCTX = ["neutral_proe", "neutral_sec_per_play", "off_epa", "rz_td_rate",
           "off_plays_pg", "pass_rate"]


# ------------------------------------------------------------------ helpers
def ns_basis(x, knots, bknots):
    """Natural cubic spline basis (truncated-power, Harrell parameterisation)."""
    x = np.asarray(x, float)
    k = np.concatenate([[bknots[0]], knots, [bknots[1]]])
    K = len(k)
    out = [x]
    for j in range(K - 2):
        def d(idx):
            return (np.clip(x - k[idx], 0, None) ** 3 - np.clip(x - k[-1], 0, None) ** 3
                    * (k[-2] - k[idx]) / (k[-2] - k[-1])
                    - np.clip(x - k[-2], 0, None) ** 3
                    * (k[-1] - k[idx]) / (k[-1] - k[-2])) / (k[-1] - k[0]) ** 2
        out.append(d(j))
    return np.column_stack(out)


def ridge_fit(X, y, alpha):
    n, p = X.shape
    A = X.T @ X + alpha * np.eye(p)
    b = X.T @ (y - y.mean())
    coef = np.linalg.solve(A, b)
    return coef, y.mean()


def grouped_cv_alpha(X, y, groups, alphas=ALPHAS, k=5):
    gs = np.unique(groups)
    RNG.shuffle(gs)
    folds = np.array_split(gs, min(k, len(gs)))
    err = np.zeros(len(alphas))
    for f in folds:
        m = np.isin(groups, f)
        Xtr, ytr, Xte, yte = X[~m], y[~m], X[m], y[m]
        for i, a in enumerate(alphas):
            c, m0 = ridge_fit(Xtr, ytr, a)
            err[i] += ((yte - (m0 + Xte @ c)) ** 2).sum()
    return alphas[int(np.argmin(err))], err


def dm(y, p_base, p_cand, year):
    d = (y - p_base) ** 2 - (y - p_cand) ** 2
    dy = pd.Series(d).groupby(pd.Series(year).values).mean()
    t = float(dy.mean() / (dy.std(ddof=1) / np.sqrt(len(dy))))
    p = float(2 * stats.t.sf(abs(t), df=len(dy) - 1))
    sd = float(dy.std(ddof=1))
    K = len(dy)
    m = (stats.t.ppf(0.975, K - 1) + stats.t.ppf(0.80, K - 1)) * sd / np.sqrt(K)
    return dict(mean_gain=float(dy.mean()), dm_t=t, dm_p=p, mde80=m,
                folds_improved=int((dy > 0).sum()), n_folds=K,
                obs_over_mde=float(dy.mean() / m) if m else np.nan)


def bh(p, q=0.10):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    thr = q * np.arange(1, m + 1) / m
    ok = p[o] <= thr
    kk = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    r = np.zeros(m, bool)
    if kk:
        r[o[:kk]] = True
    return r


# ------------------------------------------------------------------ data
def load(tier="A"):
    fr = []
    for pos in ["WR", "RB"]:
        d = pd.read_csv(ROOT / f"data/derived/w1_features_{pos}.csv", low_memory=False)
        d["pos"] = pos
        fr.append(d)
    F = pd.concat(fr, ignore_index=True)
    S = pd.read_csv(ROOT / "results/sectionS_predictions.csv")
    S = S[["gsis_id", "year", "pos", "a1_mean", "_n_eff", "_G_last", "m_hat", "B",
           "th_a1_mean"]].rename(columns={"a1_mean": "mu_hat", "_n_eff": "n_eff",
                                          "_G_last": "G_last_inc"})
    F = F.drop(columns=[c for c in ["n_eff", "mu_hat", "G_last_inc"] if c in F.columns])
    F = F.merge(S, on=["gsis_id", "year", "pos"], how="left")
    if tier == "B":
        aw = pd.read_csv(ROOT / "data/derived/adv_wr_te.csv", low_memory=False)
        ar = pd.read_csv(ROOT / "data/derived/adv_rb.csv", low_memory=False)
        tc = pd.read_csv(ROOT / "data/derived/team_context.csv", low_memory=False)
        aw = aw[["player_id", "season"] + [c for c in ADV_WR if c in aw.columns]]
        ar = ar[["player_id", "season"] + [c for c in ADV_RB if c in ar.columns]]
        adv = pd.concat([aw, ar], ignore_index=True)
        adv = adv.groupby(["player_id", "season"], as_index=False).mean(numeric_only=True)
        adv = adv.rename(columns={"player_id": "gsis_id"})
        adv["year"] = adv.season + 1
        F = F.merge(adv.drop(columns=["season"]), on=["gsis_id", "year"], how="left")
        tc = tc[["season", "team"] + [c for c in TEAMCTX if c in tc.columns]]
        tc["year"] = tc.season + 1
        tc = tc.rename(columns={c: "tc_" + c for c in TEAMCTX})
        F = F.merge(tc.drop(columns=["season"]), on=["team", "year"], how="left")
    return F


def featlist(tier, scope, pos, gated=True, blocks=("opp", "eff", "env", "struct"),
             leaky_env=False):
    f = []
    if "opp" in blocks:
        f += list(OPP)
    if "eff" in blocks:
        f += list(EFF_GATED) if gated else list(EFF_GATED) + list(EFF_REJECTED)
    if "env" in blocks:
        f += list(ENVF_LEAKY) if leaky_env else list(ENVF)
    if "struct" in blocks:
        f += list(STRUCT)
    if "struct_exp" in blocks:
        f += ["expr"]
    if "struct_dp" in blocks:
        f += ["log_draft_pick"]
    if tier == "B" and "adv" in blocks:
        f += [c for c in (ADV_WR if pos == "WR" else ADV_RB)]
        f += ["tc_" + c for c in TEAMCTX]
    if scope == "P1":
        f = f + ["mu_hat"]
    return f


def design(tr, ev, cols, use_age=True):
    """Standardise on training moments, median-impute on training medians,
    append an age natural spline with knots at training-fold quintiles."""
    med = tr[cols].median()
    Xtr = tr[cols].fillna(med).values.astype(float)
    Xev = ev[cols].fillna(med).values.astype(float)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    Xtr, Xev = (Xtr - mu) / sd, (Xev - mu) / sd
    if not use_age:
        return Xtr, Xev, list(cols)
    kn = np.quantile(tr.age.dropna(), [0.25, 0.50, 0.75])
    bk = np.quantile(tr.age.dropna(), [0.05, 0.95])
    a_tr = ns_basis(tr.age.fillna(tr.age.median()).values, kn, bk)
    a_ev = ns_basis(ev.age.fillna(tr.age.median()).values, kn, bk)
    am, asd = a_tr.mean(0), a_tr.std(0)
    asd[asd < 1e-9] = 1.0
    a_tr, a_ev = (a_tr - am) / asd, (a_ev - am) / asd
    names = list(cols) + [f"age_s{i}" for i in range(a_tr.shape[1])]
    return np.c_[Xtr, a_tr], np.c_[Xev, a_ev], names


GBT_KW = dict(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=20,
              l2_regularization=1.0, early_stopping=False, random_state=0)


def run(tier="A", gated=True, target="ppg", label="",
        blocks=("opp", "eff", "env", "struct", "adv"), leaky_env=False,
        arms=("ridge", "gbt", "hier"), use_age=True, extra=()):
    F = load(tier)
    years = YEARS if tier == "A" else [y for y in YEARS if y >= 2019]
    use = F[F.in_fit & (F.n_eff > 0) & F.year.isin(years)].copy()
    preds, coefs = [], []

    for Y in years:
        tr_all = use[use.year != Y]
        ev_all = use[use.year == Y]
        # ---- per-position models (a) ridge, (b) GBT
        for pos in ["WR", "RB"]:
            tr = tr_all[tr_all.pos == pos]
            ev = ev_all[ev_all.pos == pos].copy()
            if not len(ev):
                continue
            for scope in ["P0", "P1"]:
                cols = featlist(tier, scope, pos, gated, blocks, leaky_env) + list(extra)
                cols = [c for c in cols if c in tr.columns]
                Xtr, Xev, names = design(tr, ev, cols, use_age)
                ytr = tr[target].values
                a, _ = grouped_cv_alpha(Xtr, ytr, tr.year.values)
                c, m0 = ridge_fit(Xtr, ytr, a)
                ev[f"ridge_{scope}"] = m0 + Xev @ c
                coefs.append(pd.DataFrame(dict(year=Y, pos=pos, tier=tier, scope=scope,
                                               feature=names, coef=c, alpha=a)))
                if "gbt" in arms:
                    g = HistGradientBoostingRegressor(**GBT_KW).fit(Xtr, ytr)
                    ev[f"gbt_{scope}"] = g.predict(Xev)
            # calibrated mu_hat benchmark
            b = np.polyfit(tr.mu_hat.values, tr[target].values, 1)
            ev["mu_cal"] = b[1] + b[0] * ev.mu_hat.values
            preds.append(ev)

        # ---- (c) hierarchical, pooled across positions
        if "hier" not in arms:
            continue
        for scope in ["P0", "P1"]:
            cols = featlist(tier, scope, "WR", gated, blocks, leaky_env) + list(extra)
            cols = [c for c in cols if c in tr_all.columns]
            Xtr, Xev, names = design(tr_all, ev_all, cols, use_age)
            isrb_tr = (tr_all.pos == "RB").values.astype(float)[:, None]
            isrb_ev = (ev_all.pos == "RB").values.astype(float)[:, None]
            ytr = tr_all[target].values
            best = (None, np.inf)
            for k in KAPPAS:
                Ztr = np.c_[Xtr, isrb_tr, np.sqrt(k) * Xtr * isrb_tr]
                a, e = grouped_cv_alpha(Ztr, ytr, tr_all.year.values)
                if e.min() < best[1]:
                    best = ((k, a), e.min())
            k, a = best[0]
            Ztr = np.c_[Xtr, isrb_tr, np.sqrt(k) * Xtr * isrb_tr]
            Zev = np.c_[Xev, isrb_ev, np.sqrt(k) * Xev * isrb_ev]
            c, m0 = ridge_fit(Ztr, ytr, a)
            ev_all = ev_all.copy()
            ev_all[f"hier_{scope}"] = m0 + Zev @ c
            ev_all[f"hier_{scope}_kappa"] = k
        h = ev_all[["gsis_id", "year", "pos", "hier_P0", "hier_P1",
                    "hier_P0_kappa", "hier_P1_kappa"]]
        preds[-2] = preds[-2].merge(h, on=["gsis_id", "year", "pos"], how="left")
        preds[-1] = preds[-1].merge(h, on=["gsis_id", "year", "pos"], how="left")

    P = pd.concat(preds, ignore_index=True)
    P["tier"] = tier
    P["gated"] = gated
    P["target"] = target
    C = pd.concat(coefs, ignore_index=True)
    return P, C


ARMS = ["ridge_P0", "ridge_P1", "gbt_P0", "gbt_P1", "hier_P0", "hier_P1"]


def score(P, target="ppg", label=""):
    rows = []
    for pos in ["WR", "RB"]:
        d = P[P.pos == pos]
        y = d[target].values
        base = d.mu_hat.values
        rb_ = float(np.sqrt(((y - base) ** 2).mean()))
        rc_ = float(np.sqrt(((y - d.mu_cal.values) ** 2).mean()))
        r = dict(pos=pos, arm="mu_cal (control)", n=len(d), rmse_mu=rb_,
                 rmse_arm=rc_, d_rmse=rc_ - rb_, label=label)
        r.update(dm(y, base, d.mu_cal.values, d.year.values))
        r["spearman"] = float(d.groupby("year").apply(
            lambda g: stats.spearmanr(g.mu_cal, g[target]).statistic).mean())
        r["spearman_mu"] = float(d.groupby("year").apply(
            lambda g: stats.spearmanr(g.mu_hat, g[target]).statistic).mean())
        rows.append(r)
        for a in ARMS:
            if a not in d:
                continue
            pa = d[a].values
            ra = float(np.sqrt(((y - pa) ** 2).mean()))
            r = dict(pos=pos, arm=a, n=len(d), rmse_mu=rb_, rmse_arm=ra,
                     d_rmse=ra - rb_, label=label)
            r.update(dm(y, base, pa, d.year.values))
            v = dm(y, d.mu_cal.values, pa, d.year.values)
            r["gain_vs_mucal"] = v["mean_gain"]
            r["p_vs_mucal"] = v["dm_p"]
            r["spearman"] = float(d.groupby("year").apply(
                lambda g: stats.spearmanr(g[a], g[target]).statistic).mean())
            r["spearman_mu"] = float(d.groupby("year").apply(
                lambda g: stats.spearmanr(g.mu_hat, g[target]).statistic).mean())
            # eq (7) substitution
            th_a = (1 - d.B.values) * pa + d.B.values * d.m_hat.values
            th_b = (1 - d.B.values) * base + d.B.values * d.m_hat.values
            r["rmse_theta_mu"] = float(np.sqrt(((y - th_b) ** 2).mean()))
            r["rmse_theta_arm"] = float(np.sqrt(((y - th_a) ** 2).mean()))
            e7 = dm(y, th_b, th_a, d.year.values)
            r["eq7_gain"] = e7["mean_gain"]
            r["eq7_p"] = e7["dm_p"]
            r["eq7_mde"] = e7["mde80"]
            rows.append(r)
    return pd.DataFrame(rows)


def holdout(P, target="ppg", label=""):
    """Temporal holdout: rows 2022-2024 only, from the same LOSO predictions."""
    rows = []
    for pos in ["WR", "RB"]:
        d = P[(P.pos == pos) & (P.year >= 2022)]
        if not len(d):
            continue
        y = d[target].values
        b = float(np.sqrt(((y - d.mu_hat.values) ** 2).mean()))
        c = float(np.sqrt(((y - d.mu_cal.values) ** 2).mean()))
        for a in ["mu_cal"] + [x for x in ARMS if x in d]:
            r = float(np.sqrt(((y - d[a].values) ** 2).mean()))
            th_a = (1 - d.B.values) * d[a].values + d.B.values * d.m_hat.values
            th_b = (1 - d.B.values) * d.mu_hat.values + d.B.values * d.m_hat.values
            rows.append(dict(pos=pos, arm=a, label=label, n=len(d), rmse_mu=b,
                             rmse_mucal=c, rmse_arm=r, beats_mu=bool(r < b),
                             beats_mucal=bool(r < c),
                             rmse_theta_mu=float(np.sqrt(((y - th_b) ** 2).mean())),
                             rmse_theta_arm=float(np.sqrt(((y - th_a) ** 2).mean()))))
    return pd.DataFrame(rows)


def encompass(P, arm="ridge_P1", target="ppg"):
    """Forecast-encompassing: does yhat carry information mu_hat does not, and how much
    of it is already in the market price m_hat?"""
    rows = []
    for pos in ["WR", "RB"]:
        d = P[P.pos == pos].dropna(subset=[arm, "mu_hat", "m_hat"])
        y = d[target].values
        def ols(cols):
            X = np.c_[np.ones(len(d)), d[cols].values]
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            r = y - X @ b
            n, k = X.shape
            XtXi = np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(XtXi @ (X.T @ np.diag(r ** 2) @ X) @ XtXi))  # HC0
            return b, se, 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()
        b1, s1, r1 = ols(["mu_hat"])
        b2, s2, r2 = ols([arm])
        b3, s3, r3 = ols(["mu_hat", arm])
        b4, s4, r4 = ols(["m_hat", arm])
        rows.append(dict(pos=pos, arm=arm, n=len(d),
                         beta_mu_alone=b1[1], R2_mu=r1,
                         beta_yhat_alone=b2[1], R2_yhat=r2,
                         beta_mu_joint=b3[1], se_mu_joint=s3[1],
                         beta_yhat_joint=b3[2], se_yhat_joint=s3[2], R2_joint=r3,
                         beta_mhat=b4[1], se_mhat=s4[1],
                         beta_yhat_vs_mhat=b4[2], se_yhat_vs_mhat=s4[2], R2_mkt=r4,
                         corr_yhat_mhat=float(np.corrcoef(d[arm], d.m_hat)[0, 1]),
                         corr_yhat_mu=float(np.corrcoef(d[arm], d.mu_hat)[0, 1])))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out_s, out_p, out_c, out_h = [], [], [], []

    # ---------------- headline: tier A, gated, clean environment
    P, C = run(tier="A", gated=True)
    P["spec"] = "A_gated_clean"
    out_p.append(P); out_c.append(C)
    out_s.append(score(P, label="tier A | gated | clean env | PPG"))
    out_h.append(holdout(P, label="tier A | gated | clean env"))
    ENC = encompass(P, "ridge_P1")
    ENC0 = encompass(P, "ridge_P0")

    # ---------------- tier B (advanced layer, 6 folds)
    PB, CB = run(tier="B", gated=True)
    PB["spec"] = "B_gated_clean"
    out_p.append(PB); out_c.append(CB)
    out_s.append(score(PB, label="tier B | gated | clean env | PPG"))
    out_h.append(holdout(PB, label="tier B | gated | clean env"))

    # ---------------- declared sensitivities
    PU, _ = run(tier="A", gated=False)
    PU["spec"] = "A_ungated_clean"
    out_p.append(PU)
    out_s.append(score(PU, label="tier A | UNGATED (sens) | clean env | PPG"))

    PL, _ = run(tier="A", gated=True, leaky_env=True)
    PL["spec"] = "A_gated_LEAKY"
    out_p.append(PL)
    out_s.append(score(PL, label="tier A | gated | LEAKY env (quantifies the leak)"))

    # ---------------- block ablation, ridge only
    ABL = []
    BLOCKS = {"all": ("opp", "eff", "env", "struct"),
              "-opportunity": ("eff", "env", "struct"),
              "-efficiency": ("opp", "env", "struct"),
              "-environment": ("opp", "eff", "struct"),
              "-structure": ("opp", "eff", "env"),
              "opportunity only": ("opp",),
              "structure only": ("struct",)}
    for name, bl in BLOCKS.items():
        pa, _ = run(tier="A", gated=True, blocks=bl, arms=("ridge",))
        sc = score(pa, label=f"ablation: {name}")
        sc = sc[sc.arm.isin(["ridge_P0", "ridge_P1"])]
        sc["block"] = name
        ABL.append(sc)
    ABL = pd.concat(ABL, ignore_index=True)

    S = pd.concat(out_s, ignore_index=True)
    m = S.label.str.contains("tier A \\| gated \\| clean") & (S.arm != "mu_cal (control)")
    S["bh_reject_12"] = False
    S.loc[m, "bh_reject_12"] = bh(S.loc[m, "dm_p"].values, 0.10)
    for pos in ["WR", "RB"]:
        mm = m & (S.pos == pos)
        S.loc[mm, "bh_reject_6"] = bh(S.loc[mm, "dm_p"].values, 0.10)

    S.to_csv(ROOT / "results/sectionW1_loso.csv", index=False)
    ABL.to_csv(ROOT / "results/sectionW1_ablation.csv", index=False)
    pd.concat(out_h, ignore_index=True).to_csv(ROOT / "results/sectionW1_holdout.csv",
                                               index=False)
    pd.concat([ENC, ENC0], ignore_index=True).to_csv(
        ROOT / "results/sectionW1_encompassing.csv", index=False)
    pd.concat(out_p, ignore_index=True).to_csv(
        ROOT / "results/sectionW1_predictions.csv", index=False)
    pd.concat(out_c, ignore_index=True).to_csv(
        ROOT / "results/sectionW1_coefficients.csv", index=False)

    pd.set_option("display.width", 260)
    for lab in S.label.unique():
        for pos in ["WR", "RB"]:
            s = S[(S.label == lab) & (S.pos == pos)]
            if not len(s):
                continue
            print(f"\n=== {pos} | {lab} | n={s.n.iat[0]} | mu_hat RMSE "
                  f"{s.rmse_mu.iat[0]:.4f} | folds {s.n_folds.iat[0]} ===")
            print(s[["arm", "rmse_arm", "d_rmse", "mean_gain", "dm_t", "dm_p", "mde80",
                     "obs_over_mde", "folds_improved", "spearman", "spearman_mu",
                     "gain_vs_mucal", "p_vs_mucal", "eq7_gain", "eq7_p", "bh_reject_12"]]
                  .round(4).to_string(index=False))
    print("\n=== BLOCK ABLATION (ridge, tier A, gated, clean env) ===")
    print(ABL[["block", "pos", "arm", "rmse_arm", "mean_gain", "dm_p", "mde80",
               "gain_vs_mucal", "p_vs_mucal", "eq7_gain", "eq7_p"]]
          .round(4).to_string(index=False))
    print("\n=== TEMPORAL HOLDOUT 2022-24 ===")
    print(pd.concat(out_h, ignore_index=True).round(4).to_string(index=False))
    print("\n=== FORECAST ENCOMPASSING ===")
    print(pd.concat([ENC, ENC0], ignore_index=True).round(4).to_string(index=False))

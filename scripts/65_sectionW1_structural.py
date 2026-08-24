"""§W1 (L2) — structural layer: age applied (L2.1) and availability (L2.2).

L2.1  A1  multiplicative carry-forward of §H's era-3 relative age curve:
          yhat_age = yhat * f3(age_Y) / f3(age_Y-1)
      A2  the age spline estimated inside the projection vs no age term at all,
          and the §H external curve added as a feature.
      Plus a finer decomposition of the structure block (age / experience / draft).

L2.2  Stage 1: does prior availability predict next-season availability?
      Stage 2 (only if stage 1 clears): points per SCHEDULED week as a second target.

Pre-registration: results/sectionW1_notes.md W1.8, W1.9.
Outputs: results/sectionW1_age.csv, sectionW1_availability.csv, sectionW1_ppsw.csv
"""
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")
ROOT = Path("/Users/thomasmcnamee/NFL")
spec = importlib.util.spec_from_file_location(
    "w1", ROOT / "scripts/64_sectionW1_projection.py")
w1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w1)

# ------------------------------------------------------------------ §H age curve
AC = pd.read_csv(ROOT / "results/age_curve_era.csv")
AC = AC[(AC.outcome == "relative") & (AC.era == "2017-2025")]
CURVE = {p: (g.age.values, g.fit.values) for p, g in AC.groupby("position")}


def f_age(pos, a):
    x, y = CURVE[pos]
    return np.interp(np.asarray(a, float), x, y)


def age_ratio(pos, age):
    """Expected relative-production multiplier for one more year of age."""
    return f_age(pos, age) / f_age(pos, np.asarray(age, float) - 1.0)


# ================================================================== L2.1
def l21():
    P = pd.read_csv(ROOT / "results/sectionW1_predictions.csv")
    P = P[P.spec == "A_gated_clean"]
    rows = []
    for pos in ["WR", "RB"]:
        d = P[P.pos == pos].dropna(subset=["age"]).copy()
        y = d.ppg.values
        d["r_age"] = age_ratio(pos, d.age.values)
        for arm in ["mu_hat", "mu_cal", "ridge_P0", "ridge_P1", "hier_P1"]:
            base = d[arm].values
            adj = base * d.r_age.values
            r = dict(pos=pos, arm=arm, variant="A1 x f3(age)/f3(age-1)", n=len(d),
                     rmse_base=float(np.sqrt(((y - base) ** 2).mean())),
                     rmse_adj=float(np.sqrt(((y - adj) ** 2).mean())),
                     mean_ratio=float(d.r_age.mean()),
                     min_ratio=float(d.r_age.min()), max_ratio=float(d.r_age.max()))
            r.update(w1.dm(y, base, adj, d.year.values))
            # inside eq (7)
            th_b = (1 - d.B.values) * base + d.B.values * d.m_hat.values
            th_a = (1 - d.B.values) * adj + d.B.values * d.m_hat.values
            e = w1.dm(y, th_b, th_a, d.year.values)
            r["eq7_gain"], r["eq7_p"], r["eq7_mde"] = e["mean_gain"], e["dm_p"], e["mde80"]
            r["rmse_theta_base"] = float(np.sqrt(((y - th_b) ** 2).mean()))
            r["rmse_theta_adj"] = float(np.sqrt(((y - th_a) ** 2).mean()))
            rows.append(r)
    return pd.DataFrame(rows)


def l21_design():
    """A2: age spline in-design vs no age term; and §H's external curve as a feature."""
    out = []
    P_noage, _ = w1.run(tier="A", gated=True, arms=("ridge",), use_age=False)
    s = w1.score(P_noage, label="A2: projection WITHOUT any age term")
    out.append(s[s.arm.isin(["ridge_P0", "ridge_P1"])])
    # §H curve value as an extra feature (level, not ratio)
    F = pd.read_csv(ROOT / "data/derived/w1_features_WR.csv", low_memory=False)
    for pos in ["WR", "RB"]:
        f = pd.read_csv(ROOT / f"data/derived/w1_features_{pos}.csv", low_memory=False)
        f["f_age_H"] = f_age(pos, f.age.values)
        f["f_age_H_ratio"] = age_ratio(pos, f.age.values)
        f.to_csv(ROOT / f"data/derived/w1_features_{pos}.csv", index=False)
    P_ext, _ = w1.run(tier="A", gated=True, arms=("ridge",),
                      extra=("f_age_H", "f_age_H_ratio"))
    s = w1.score(P_ext, label="A2: + §H external curve as feature")
    out.append(s[s.arm.isin(["ridge_P0", "ridge_P1"])])
    # finer structure decomposition (P1 = mu_hat + one structural variable at a time)
    for name, kw in [("mu_hat + age spline only", dict(blocks=(), use_age=True)),
                     ("mu_hat + experience only", dict(blocks=("struct_exp",), use_age=False)),
                     ("mu_hat + draft pick only", dict(blocks=("struct_dp",), use_age=False)),
                     ("mu_hat alone (ridge)", dict(blocks=(), use_age=False))]:
        pa, _ = w1.run(tier="A", gated=True, arms=("ridge",), **kw)
        s = w1.score(pa, label=f"A2 decomposition: {name}")
        out.append(s[s.arm == "ridge_P1"])
    return pd.concat(out, ignore_index=True)


# ================================================================== L2.2
def l22_stage1():
    F = w1.load("A")
    F = F[F.in_fit & (F.n_prior > 0)].copy()
    COLS = ["avail_wtd", "avail_last", "avail_career", "n_prior", "G_last", "G_wtd",
            "gap_since_last"]
    rows, preds = [], []
    for pos in ["WR", "RB"]:
        d = F[F.pos == pos].copy()
        d["pred"] = np.nan
        d["const"] = np.nan
        for Y in range(2015, 2025):
            tr, ev = d[d.year != Y], d[d.year == Y]
            if not len(ev):
                continue
            Xtr, Xev, names = w1.design(tr, ev, COLS)
            ytr = tr.avail_y.values
            a, _ = w1.grouped_cv_alpha(Xtr, ytr, tr.year.values)
            c, m0 = w1.ridge_fit(Xtr, ytr, a)
            d.loc[ev.index, "pred"] = np.clip(m0 + Xev @ c, 0.05, 1.0)
            d.loc[ev.index, "const"] = ytr.mean()
        y = d.avail_y.values
        r = dict(pos=pos, n=len(d), target="avail = G/S",
                 rmse_const=float(np.sqrt(((y - d.const.values) ** 2).mean())),
                 rmse_model=float(np.sqrt(((y - d.pred.values) ** 2).mean())),
                 r_pred=float(np.corrcoef(d.pred, y)[0, 1]),
                 R2_oos=float(1 - ((y - d.pred.values) ** 2).sum()
                              / ((y - d.const.values) ** 2).sum()),
                 sd_avail=float(np.std(y)), mean_avail=float(np.mean(y)))
        r.update(w1.dm(y, d.const.values, d.pred.values, d.year.values))
        rows.append(r)
        preds.append(d[["gsis_id", "year", "pos", "avail_y", "pred", "const",
                        "ppg", "ppsw", "mu_hat", "ppsw_hat", "avail_wtd", "B",
                        "m_hat", "sched", "games"]])
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def l22_stage2(av):
    """Second target: points per SCHEDULED week."""
    P = pd.read_csv(ROOT / "results/sectionW1_predictions.csv")
    P = P[P.spec == "A_gated_clean"]
    A = av[["gsis_id", "year", "pos", "pred"]].rename(columns={"pred": "avail_pred"})
    P = P.merge(A, on=["gsis_id", "year", "pos"], how="left")
    # direct projection of ppsw, same design
    P_direct, _ = w1.run(tier="A", gated=True, arms=("ridge",), target="ppsw")
    rows = []
    for pos in ["WR", "RB"]:
        d = P[P.pos == pos].dropna(subset=["avail_pred", "ppsw_hat"]).copy()
        y = d.ppsw.values
        base = d.ppsw_hat.values                       # incumbent: recency-wtd mean ppsw
        # calibrated incumbent control, fitted per LOSO fold (same logic as mu_cal)
        cal = np.full(len(d), np.nan)
        for Y in sorted(d.year.unique()):
            tr, ev = d[d.year != Y], d[d.year == Y]
            b = np.polyfit(tr.ppsw_hat.values, tr.ppsw.values, 1)
            cal[(d.year == Y).values] = b[1] + b[0] * ev.ppsw_hat.values
        cands = {"ppsw_hat CALIBRATED (control)": cal,
                 "mu_hat x avail_wtd (naive, no model)":
                     d.mu_hat.values * d.avail_wtd.values,
                 "mu_hat x avail_pred": d.mu_hat.values * d.avail_pred.values,
                 "ridge_P1 x avail_pred": d.ridge_P1.values * d.avail_pred.values,
                 "ridge_P1 (PPG, unscaled)": d.ridge_P1.values,
                 "mu_hat (PPG, unscaled)": d.mu_hat.values}
        dd = P_direct[P_direct.pos == pos].set_index(["gsis_id", "year"])
        j = d.set_index(["gsis_id", "year"]).join(dd[["ridge_P1"]], rsuffix="_dir")
        cands["ridge_P1 fit DIRECTLY on ppsw"] = j.ridge_P1_dir.values
        for k, v in cands.items():
            ok = np.isfinite(v)
            r = dict(pos=pos, arm=k, n=int(ok.sum()), target="ppsw",
                     rmse_incumbent=float(np.sqrt(((y[ok] - base[ok]) ** 2).mean())),
                     rmse_arm=float(np.sqrt(((y[ok] - v[ok]) ** 2).mean())))
            r.update(w1.dm(y[ok], base[ok], v[ok], d.year.values[ok]))
            c2 = w1.dm(y[ok], cal[ok], v[ok], d.year.values[ok])
            r["gain_vs_cal"] = c2["mean_gain"]
            r["p_vs_cal"] = c2["dm_p"]
            r["mde_vs_cal"] = c2["mde80"]
            rows.append(r)
    return pd.DataFrame(rows)


def nested_increment():
    """The decisive nested test: does the measurable-input apparatus add anything
    ON TOP of a calibrated, age-aware mu_hat?"""
    base, _ = w1.run(tier="A", gated=True, arms=("ridge",), blocks=("struct",))
    full, _ = w1.run(tier="A", gated=True, arms=("ridge",))
    rows = []
    for pos in ["WR", "RB"]:
        b = base[base.pos == pos].set_index(["gsis_id", "year"])
        f = full[full.pos == pos].set_index(["gsis_id", "year"])
        j = b[["ppg", "ridge_P1", "mu_hat", "B", "m_hat"]].join(
            f[["ridge_P1"]], rsuffix="_full").dropna().reset_index()
        y = j.ppg.values
        r = dict(pos=pos, n=len(j),
                 rmse_base_structonly=float(np.sqrt(((y - j.ridge_P1) ** 2).mean())),
                 rmse_full=float(np.sqrt(((y - j.ridge_P1_full) ** 2).mean())))
        r.update(w1.dm(y, j.ridge_P1.values, j.ridge_P1_full.values, j.year.values))
        th_b = (1 - j.B.values) * j.ridge_P1.values + j.B.values * j.m_hat.values
        th_f = (1 - j.B.values) * j.ridge_P1_full.values + j.B.values * j.m_hat.values
        e = w1.dm(y, th_b, th_f, j.year.values)
        r["eq7_gain"], r["eq7_p"], r["eq7_mde"] = e["mean_gain"], e["dm_p"], e["mde80"]
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 260)
    A1 = l21()
    A1.to_csv(ROOT / "results/sectionW1_age.csv", index=False)
    print("=== L2.1 A1 — §H era-3 age curve applied multiplicatively ===")
    print(A1[["pos", "arm", "n", "rmse_base", "rmse_adj", "mean_gain", "dm_p", "mde80",
              "folds_improved", "rmse_theta_base", "rmse_theta_adj", "eq7_gain",
              "eq7_p", "mean_ratio", "min_ratio"]].round(4).to_string(index=False))

    A2 = l21_design()
    A2.to_csv(ROOT / "results/sectionW1_age_design.csv", index=False)
    print("\n=== L2.1 A2 — age inside the design ===")
    print(A2[["label", "pos", "arm", "rmse_mu", "rmse_arm", "mean_gain", "dm_p",
              "mde80", "gain_vs_mucal", "p_vs_mucal", "eq7_gain", "eq7_p"]]
          .round(4).to_string(index=False))

    S1, AV = l22_stage1()
    S1.to_csv(ROOT / "results/sectionW1_availability.csv", index=False)
    print("\n=== L2.2 stage 1 — does prior availability predict next-season availability? ===")
    print(S1.round(4).to_string(index=False))

    NI = nested_increment()
    NI.to_csv(ROOT / "results/sectionW1_nested.csv", index=False)
    print("\n=== NESTED: inputs on top of (mu_hat + age + exp + draft) ===")
    print(NI.round(4).to_string(index=False))

    S2 = l22_stage2(AV)
    S2.to_csv(ROOT / "results/sectionW1_ppsw.csv", index=False)
    print("\n=== L2.2 stage 2 — points per SCHEDULED week ===")
    print(S2.round(4).to_string(index=False))

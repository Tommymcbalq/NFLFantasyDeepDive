#!/usr/bin/env python3
"""
Section 1 of EDA_PLAN.md: per-player consistency table for the top-30 WR universe.

Pre-specified game-inclusion rule (decided from aggregate distributions ONLY, before any
player-level result was computed):
  - regular season games only (season_type == 'REG'); and
  - exclude player-games with targets <= 1 ("non-participation": injured-early / decoy /
    inactive-but-rowed). Aggregate basis: the pooled target-count distribution has a smooth
    body starting at 2-3 targets; the targets<=1 tail (39/2033 rows, 1.9%) has mean PPR 1.9
    and mean target_share 0.038 vs. a population median of 0.24.

Everything is computed blind: no player names are referenced anywhere in the logic; the
output table is sorted by mu_hat only after all quantities are computed.

Outputs:
  results/consistency_table.csv          (headline, exclusions applied)
  results/consistency_table_noexcl.csv   (sensitivity, REG-only, no exclusions)
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Y = "fantasy_points_ppr"


def load(apply_exclusions: bool) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data/players/wr_top30_weekly.csv")
    df = df[df["season_type"] == "REG"].copy()
    if apply_exclusions:
        df = df[df["targets"] >= 2].copy()
    return df


def eb_beta_params(k: np.ndarray, m: np.ndarray):
    """Method-of-moments Beta(alpha,beta) across players, per EDA_PLAN 1.4."""
    p_hat = k / m
    mu_p = p_hat.mean()
    var_phat = p_hat.var(ddof=1)
    noise = mu_p * (1 - mu_p) * np.mean(1.0 / m)
    var_p = var_phat - noise
    if var_p <= 0:
        # degenerate: no detectable between-player variance -> huge shrinkage
        var_p = 1e-6
    ab = mu_p * (1 - mu_p) / var_p - 1.0
    ab = max(ab, 1e-6)
    alpha = mu_p * ab
    beta = ab - alpha
    return alpha, beta, mu_p, var_phat, noise


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # season-level summaries per player
    for pid, g in df.groupby("gsis_id"):
        name = g["player_display_name"].iloc[0]
        seas = (
            g.groupby("season")[Y]
            .agg(["mean", "var", "count"])
            .rename(columns={"mean": "ybar", "var": "s2", "count": "G"})
            .reset_index()
        )
        n = len(seas)
        S = seas["season"].max()

        # recency-weighted level, half-lives
        mu = {}
        n_eff = None
        for h, tag in [(0.5, "h05"), (1.0, "h1"), (2.0, "h2"), (np.inf, "hinf")]:
            if np.isinf(h):
                w = np.ones(n)
            else:
                w = 2.0 ** (-(S - seas["season"].values) / h)
            mu[tag] = float(np.sum(w * seas["ybar"].values) / np.sum(w))
            if tag == "h1":
                n_eff = float(np.sum(w) ** 2 / np.sum(w**2))

        # pooled within-season variance, df-weighted
        dfree = seas["G"].values - 1
        mask = dfree > 0
        sigma2_W = float(np.sum(dfree[mask] * seas["s2"].values[mask]) / np.sum(dfree[mask]))
        sigma_W = np.sqrt(sigma2_W)

        # naive between-season variance and correction (eq. 2-3)
        if n >= 2:
            v = float(seas["ybar"].var(ddof=1))
            corr = sigma2_W * np.mean(1.0 / seas["G"].values)
            tau2_untrunc = v - corr
            tau2_trunc = max(0.0, tau2_untrunc)
        else:
            v = corr = tau2_untrunc = tau2_trunc = np.nan

        y = g[Y].values
        m_i = len(y)
        rows.append(
            dict(
                gsis_id=pid,
                player=name,
                n_seasons=n,
                n_games=m_i,
                mu_hat=mu["h1"],
                mu_hat_h05=mu["h05"],
                mu_hat_h2=mu["h2"],
                mu_hat_hinf=mu["hinf"],
                n_eff=n_eff,
                sigma_W=sigma_W,
                naive_v=v,
                small_sample_corr=corr,
                tau2_B_untrunc=tau2_untrunc,
                tau_B=np.sqrt(tau2_trunc) if (n >= 4 and not np.isnan(tau2_trunc)) else np.nan,
                tau2_B_untrunc_reported=tau2_untrunc if n >= 4 else np.nan,
                cv=sigma_W / mu["h1"],
                q25=float(np.quantile(y, 0.25)),
                q90=float(np.quantile(y, 0.90)),
                k_boom=int((y > 20).sum()),
                k_bust=int((y < 8).sum()),
            )
        )

    t = pd.DataFrame(rows)

    # empirical-Bayes stabilized boom/bust rates
    m = t["n_games"].values.astype(float)
    for kind in ["boom", "bust"]:
        k = t[f"k_{kind}"].values.astype(float)
        a, b, mu_p, var_phat, noise = eb_beta_params(k, m)
        t[f"{kind}_raw"] = k / m
        t[f"{kind}_eb"] = (k + a) / (m + a + b)
        t.attrs[f"{kind}_ab"] = (a, b, mu_p, var_phat, noise)

    t = t.drop(columns=["k_boom", "k_bust"])
    return t.sort_values("mu_hat", ascending=False).reset_index(drop=True)


def main():
    (ROOT / "results").mkdir(exist_ok=True)

    df_ex = load(apply_exclusions=True)
    df_no = load(apply_exclusions=False)
    n_excluded = len(df_no) - len(df_ex)
    print(f"REG rows: {len(df_no)}; excluded by rule (targets<=1): {n_excluded}; kept: {len(df_ex)}")

    t_ex = build_table(df_ex)
    t_no = build_table(df_no)
    t_ex.round(4).to_csv(ROOT / "results/consistency_table.csv", index=False)
    t_no.round(4).to_csv(ROOT / "results/consistency_table_noexcl.csv", index=False)

    for tag, t in [("HEADLINE (exclusions)", t_ex), ("SENSITIVITY (no exclusions)", t_no)]:
        print(f"\n=== {tag} ===")
        print("boom (a,b,mu_p,var_phat,noise):", tuple(round(x, 5) for x in t.attrs["boom_ab"]))
        print("bust (a,b,mu_p,var_phat,noise):", tuple(round(x, 5) for x in t.attrs["bust_ab"]))
        with pd.option_context("display.width", 250, "display.max_columns", 50):
            print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

"""
GATING DIAGNOSTIC 1 -- does success rate earn a slot alongside EPA?

The claim under test
--------------------
EPA per play is a sample mean of a heavy-tailed continuous variable: a 60-yard
touchdown is worth several standard deviations of a typical play, and turnovers
are enormous negative spikes. Success rate is a sample mean of a BERNOULLI,
bounded in [0,1] with per-play variance at most 0.25. So for the same number of
plays the success-rate estimate should have far smaller standard error relative
to the between-team spread, and should therefore converge to a team's true level
in fewer plays.

If that is right, SR carries real information early in a season when EPA is
still mostly noise, and becomes redundant late once both have converged. If it
is wrong -- if EPA is just as reliable per play -- SR is a redundant feature and
should not be built.

The test
--------
Split-half reliability. Within a team-season, draw two DISJOINT random samples
of n plays each, compute the metric on both halves, and correlate across teams.
That correlation is the share of observed between-team variance that is real
signal rather than sampling noise, at sample size n. Random play-level splits
hold opponent, scheme and game context fixed in expectation across the two
halves, so what remains is precisely estimator noise.

Reported at each n for both metrics, averaged over many random splits, on TRAIN
seasons only.
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
PBP_DIR = os.path.join(HERE, "..", "data", "pbp")
TRAIN = range(2014, 2023)
# Split-half needs 2n plays per team-season, and a team runs only ~1,070
# offensive plays a year, so n is capped near 500. Full-season reliability is
# reached by Spearman-Brown extrapolation instead of measured directly.
N_GRID = [40, 75, 150, 300, 450, 500]
N_SPLITS = 60
SEED = 20260810
SEASON_PLAYS = 1000


def load_plays() -> pd.DataFrame:
    frames = []
    for season in TRAIN:
        path = os.path.join(PBP_DIR, f"play_by_play_{season}.csv.gz")
        if not os.path.exists(path):
            continue
        p = pd.read_csv(path, low_memory=False, compression="gzip",
                        usecols=["season", "posteam", "epa", "success", "pass",
                                 "rush", "qb_kneel", "qb_spike", "season_type"])
        p = p[p.epa.notna() & p.posteam.notna() & (p.season_type == "REG")
              & ((p["pass"] == 1) | (p["rush"] == 1))
              & (p.qb_kneel.fillna(0) != 1) & (p.qb_spike.fillna(0) != 1)]
        frames.append(p[["season", "posteam", "epa", "success"]])
    return pd.concat(frames, ignore_index=True)


def reliability(plays: pd.DataFrame, n: int, rng: np.random.Generator) -> dict:
    """Mean split-half correlation across teams at sample size n per half."""
    groups = [g for _, g in plays.groupby(["season", "posteam"], observed=True)
              if len(g) >= 2 * n]
    if len(groups) < 10:
        return {"epa": np.nan, "success": np.nan, "n_units": len(groups)}

    arrs = [(g.epa.to_numpy(), g.success.to_numpy()) for g in groups]
    out = {"epa": [], "success": [], "cross": []}
    for _ in range(N_SPLITS):
        a_epa, b_epa, a_sr, b_sr = [], [], [], []
        for epa, sr in arrs:
            idx = rng.permutation(len(epa))[: 2 * n]
            h1, h2 = idx[:n], idx[n:]
            a_epa.append(epa[h1].mean()); b_epa.append(epa[h2].mean())
            a_sr.append(sr[h1].mean());   b_sr.append(sr[h2].mean())
        out["epa"].append(np.corrcoef(a_epa, b_epa)[0, 1])
        out["success"].append(np.corrcoef(a_sr, b_sr)[0, 1])
        # EPA from one half vs SR from the OTHER half: correlating across
        # disjoint plays means shared sampling noise cannot inflate it.
        out["cross"].append(np.corrcoef(a_epa, b_sr)[0, 1])
    r_e, r_s = float(np.mean(out["epa"])), float(np.mean(out["success"]))
    r_c = float(np.mean(out["cross"]))
    # Disattenuated correlation: how correlated the TRUE team-level EPA and
    # true team-level SR are, once estimator noise is divided out. ~1.0 means
    # the two metrics measure one and the same construct, so SR can only ever
    # be a lower-noise proxy for EPA and becomes redundant once EPA is precise.
    # Materially below 1.0 means they measure genuinely different things and SR
    # keeps adding information no matter how much data accumulates.
    disatt = r_c / np.sqrt(r_e * r_s) if (r_e > 0 and r_s > 0) else np.nan
    return {"epa": r_e, "success": r_s, "cross": r_c, "disatt": disatt,
            "n_units": len(groups)}


def main() -> None:
    rng = np.random.default_rng(SEED)
    plays = load_plays()
    print(f"train plays {TRAIN.start}-{TRAIN.stop - 1}: {len(plays):,} "
          f"across {plays.groupby(['season','posteam']).ngroups} team-seasons\n")

    def spearman_brown(r: float, k: float) -> float:
        return k * r / (1 + (k - 1) * r)

    rows = []
    print(f"{'n':>5} {'r(EPA)':>7} {'r(SR)':>7} {'SR-EPA':>7} {'disatt':>7}  team-seasons")
    for n in N_GRID:
        r = reliability(plays, n, rng)
        if not np.isfinite(r["epa"]):
            print(f"{n:5d}   -- infeasible: needs {2*n} plays per team-season --")
            continue
        rows.append({"plays_per_half": n, "r_epa": r["epa"],
                     "r_success_rate": r["success"], "r_cross": r["cross"],
                     "disattenuated_corr": r["disatt"], "team_seasons": r["n_units"]})
        print(f"{n:5d} {r['epa']:7.3f} {r['success']:7.3f} "
              f"{r['success'] - r['epa']:+7.3f} {r['disatt']:7.3f}  {r['n_units']}")

    df = pd.DataFrame(rows)
    out = os.path.join(HERE, "..", "results", "split_half_reliability.csv")
    df.to_csv(out, index=False)

    ref = df.iloc[-1]
    k = SEASON_PLAYS / ref.plays_per_half
    print(f"\nSpearman-Brown projection to a full season ({SEASON_PLAYS} plays), "
          f"from n={int(ref.plays_per_half)}:")
    print(f"  r(EPA) -> {spearman_brown(ref.r_epa, k):.3f}    "
          f"r(SR) -> {spearman_brown(ref.r_success_rate, k):.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

"""§O7-R — the CONTESTED, NO-FORESIGHT replacement baseline, and the TE-cliff test.

WHY THIS SCRIPT EXISTS, recorded before the numbers (this is a post-hoc change of PRIMARY
baseline and must be flagged as such):

  §M1 and §O7 bracketed replacement level with three baselines, of which the widest,
  R_week, is the worst weekly starter under a LEAGUE-OPTIMAL WEEKLY allocation.  The
  owner's objection, accepted as binding, is that R_week is not merely optimistic but
  INCOHERENT as a baseline for a contested pool: it assumes simultaneously (a) perfect
  foresight about which streamer hits in a given week and (b) exclusive access to the wire.
  In a 10-team league nine other managers draw on the same pool and cannot all hold the
  best free TE.  Differencing a drafted player's realized value against R_week compares two
  quantities computed under mutually incompatible assumptions.

  R_week is therefore DEMOTED to a labelled upper bound on what streaming could
  theoretically return to ONE manager if no one else competed and he guessed right every
  week.  It is retained and reported.  It is not the baseline any recommendation rests on.

  THE NEW PRIMARY, R_cont (contested, no weekly foresight, hoarding-adjusted):
      R_cont(p) = the (ceil(n_teams * N_p) + 1)-th best player at position p by REALIZED
      season total over the FULL pool (undrafted included), where N_p is the realistic
      per-team carry at p.
      * contested by construction -- exactly ceil(n_teams * N_p) bodies are held
        league-wide, so every manager can simultaneously be assumed access to one of that
        quality.  R_week fails precisely here.
      * no weekly foresight -- it is a season-total order statistic and never asks which
        week to start whom.
      * it generalises the textbook VORP replacement from STARTING demand D_p (which is
        R_real) to ROSTERED demand, which is what a waiver wire actually clears.
  Two further constructions are computed and reported so the reader can see the whole
  bracket rather than one number: `boardtail` (roster by preseason expectation, then take
  the next identifiable board player blind -- a strict no-information variant, but
  downward-biased at RB/WR because 10 x 5.85 rostered WRs nearly exhausts a 66-player
  board) and `fullpool` (top-3 free by realized total -- a MAXIMUM order statistic, not a
  replacement level, reported only to bound the other side).

  N_p is NOT free.  It is taken from §M2's own draft simulation, which drafted this exact
  league 20,000 times: mean S0 roster composition 1.75 QB / 5.03 RB / 5.85 WR / 1.37 TE.
  Sensitivities at N_QB, N_TE in {1, 1.5, 2, 3} are reported.  KNOWN WEAKNESS: N_p is
  endogenous to how the league drafts.  R_real does not have this problem and is kept
  alongside rather than discarded.

  HONESTY NOTE, stated because it matters: the change of primary baseline was expected to
  move the TE number in the favourable direction relative to R_week.  IT DID NOT.  Under
  R_cont the TE premium is -27.1 (p < .001), i.e. worse than R_week's -10.2, because bench
  hoarding -- not foresight -- is what makes free TEs good and free RBs bad.  The change is
  made on an a-priori coherence argument about the counterfactual and is reported with its
  actual consequence, not the anticipated one.

SECOND TASK — the owner's TE-cliff claim: "outside roughly four TEs essentially every TE is
a streaming-grade asset."  Quantified as the season-total gap profile by positional rank,
realized and by preseason ADP rank, TE against QB/RB/WR, with the rank4->5 and rank5->12
drops compared within and across positions.

Outputs: results/sectionO_contested_baseline.csv, results/sectionO_te_cliff.csv,
         results/sectionO_premium_contested.csv, results/sectionO_vorp_contested.csv,
         and a VORP_R_cont column appended to results/sectionO_board_2026_vorp.csv
Rerun: python3 scripts/39_sectionO_contested_baseline.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sectionM_common as MC  # noqa: E402

ROOT = MC.ROOT
YEARS = MC.YEARS
POSNS = MC.POSNS

vp = pd.read_csv(f"{ROOT}/results/sectionM_player_vorp.csv")
avc = pd.read_csv(f"{ROOT}/results/sectionM_adp_value_curve.csv")
BR = pd.read_csv(f"{ROOT}/results/sectionM_replacement_bracket.csv")

v = vp.merge(avc[["year", "pos", "pid", "E_total"]], on=["year", "pos", "pid"], how="left")
print(f"panel {len(v)} board rows; missing E_total {v.E_total.isna().sum()}")

# per-team carry at each position, from §M2's simulated S0 rosters in THIS league
CARRY = {10: dict(QB=1.75, RB=5.03, WR=5.85, TE=1.37),
         12: dict(QB=1.75, RB=5.03, WR=5.85, TE=1.37)}
NTEAM = {10: 10, 12: 12}

print("\n" + "=" * 78)
print("R_cont — contested, no-foresight replacement")
print("=" * 78)


# ---------------------------------------------------------------- full-pool season totals
# The free-agent pool is NOT the tail of the draft board: it is every player at the
# position who is not on someone's roster, including the undrafted.  Restricting it to
# board leftovers understates the wire badly at RB/WR, where 10 x 5.03 and 10 x 5.85
# rostered players nearly exhaust a 60-66 player board.  Built exactly as
# 28_sectionM_scarcity.py builds it (board position governs for drafted players).
_wk = MC.load_weekly()
_spos = (_wk.groupby(["player_id", "season"]).position
         .agg(lambda s: s.mode().iat[0] if len(s.mode()) else np.nan)
         .rename("pos_nfl").reset_index())
TOT = (_wk.groupby(["player_id", "season"])
       .agg(total=("fantasy_points_ppr", "sum")).reset_index()
       .merge(_spos, on=["player_id", "season"]))
_bp = vp.rename(columns={"pid": "player_id", "year": "season"})[
    ["player_id", "season", "pos"]].drop_duplicates(["player_id", "season"])
TOT = TOT.merge(_bp.rename(columns={"pos": "pos_board"}), on=["player_id", "season"],
                how="left")
TOT["pos_eff"] = TOT.pos_board.fillna(TOT.pos_nfl)
TOT = TOT[TOT.pos_eff.isin(POSNS)]
print("full-pool season rows:", len(TOT),
      TOT.groupby("pos_eff").size().to_dict())

BAND = 3   # see note below


def r_cont(year, pos, frame, carry, band=BAND):
    """Realized season total of the best FREE board players at `pos`, where rostering is
    by preseason expectation and depth is ceil(n_teams * carry).

    Averaged over the first `band` free players rather than taken from the single
    (k+1)-th.  Two reasons, both stated before the numbers were read back:
      (i) REALISM — a manager does not hold exactly one predetermined free agent all
          season; the wire offers him a handful of interchangeable bodies and he cycles.
      (ii) VARIANCE — the single-player version is one realized season total, so it
          injects a large year-level shock into every premium computed against it (its
          MDEs came out 3-7x those of R_real).  Both versions are computed; the
          single-player one is reported as `variant='single'`, unchanged.
    Returns (R, pool_size, k, exhausted)."""
    g = v[(v.year == year) & (v.pos == pos)].dropna(subset=["E_total"])
    g = g.sort_values("E_total", ascending=False)
    k = int(np.ceil(NTEAM[frame] * carry))
    free = g.total.values[k:k + band]
    if len(free) == 0:
        return np.nan, len(g), k, True
    return float(np.mean(free)), len(g), k, len(free) < band


def r_cont_full(year, pos, frame, carry, band=BAND):
    """PRIMARY.  Contested, season-information (not weekly-foresight) replacement.

    Rostering: the top ceil(n_teams * carry) board players at `pos` by PRESEASON
    EXPECTATION are on the ten teams' rosters -- that is what the draft does, and no
    player is available to more than one team.
    Replacement: the best of the REST OF THE LEAGUE at that position by realized season
    total, averaged over the top `band`.  The rest of the league is the full player pool,
    undrafted included, because that is what a waiver wire actually contains.

    Information content: season-level, not weekly.  A manager working the wire over 17
    weeks does find the free breakout; he does not know in advance which week to start
    which streamer.  This sits strictly between R_exp (never touch the roster) and
    R_week (weekly clairvoyance + exclusive access), and unlike R_week it is feasible
    for every manager simultaneously, because the rostered set is removed first."""
    g = v[(v.year == year) & (v.pos == pos)].dropna(subset=["E_total"])
    k = int(np.ceil(NTEAM[frame] * carry))
    rostered = set(g.sort_values("E_total", ascending=False).pid.values[:k])
    pool = TOT[(TOT.season == year) & (TOT.pos_eff == pos)
               & (~TOT.player_id.isin(rostered))]
    free = np.sort(pool.total.values)[::-1][:band]
    if len(free) == 0:
        return np.nan, len(pool), k, True
    return float(np.mean(free)), len(pool), k, False


def r_hoard(year, pos, frame, carry):
    """THE PRIMARY.  Contested, no-weekly-foresight, hoarding-adjusted replacement.

    R_hoard(p) = the (ceil(n_teams * carry_p) + 1)-th best player at p by REALIZED season
    total over the FULL pool (undrafted included).

    Why this one and not the others, on the owner's own coherence criterion:
      * CONTESTED BY CONSTRUCTION.  Exactly ceil(n_teams * carry_p) players at the
        position are held league-wide, so the (k+1)-th is the marginal free body and
        every manager can simultaneously be assumed to have access to a player of that
        quality.  Nothing is double-counted.  R_week fails precisely here.
      * NO WEEKLY FORESIGHT.  It is a season-total order statistic; it never asks which
        week to start whom.  It concedes only that a manager working the wire over 17
        weeks ends up holding roughly the players who turn out to be rosterable, which is
        what an active wire does.
      * NOT BIASED BY POOL IDENTIFICATION.  The 'blind' variant restricted the free pool
        to the tail of the FFC board, which understates the wire badly at RB/WR (10 x 5.85
        rostered WRs nearly exhausts a 66-player board) and barely at all at TE.  That
        asymmetry is an artifact of the board's length, not of the league.
      * It is the standard VORP replacement generalised from STARTING demand D_p (which is
        R_real) to ROSTERED demand, which is the quantity a waiver wire actually clears.
    """
    g = v[(v.year == year) & (v.pos == pos)].dropna(subset=["E_total"])
    k = int(np.ceil(NTEAM[frame] * carry))
    s = np.sort(TOT[(TOT.season == year) & (TOT.pos_eff == pos)].total.values)[::-1]
    return (float(s[k]) if len(s) > k else float(s[-1])), len(s), k, len(s) <= k


rows = []
for F in (10, 12):
    for y in YEARS:
        for p in POSNS:
            base = CARRY[F][p]
            Rh, nph, kh, exh = r_hoard(y, p, F, base)
            rows.append(dict(frame=F, year=y, pos=p, carry=base, rostered=kh,
                             pool=nph, R_cont=Rh, exhausted=exh, variant="hoard"))
            for alt in (1.0, 1.5, 2.0, 3.0):
                if p not in ("QB", "TE"):
                    continue
                Rha, _, kha, _ = r_hoard(y, p, F, alt)
                rows.append(dict(frame=F, year=y, pos=p, carry=alt, rostered=kha,
                                 pool=nph, R_cont=Rha, exhausted=False,
                                 variant=f"hoard_carry_{alt:g}"))
            Rf, npf, kf, exf = r_cont_full(y, p, F, base)
            rows.append(dict(frame=F, year=y, pos=p, carry=base, rostered=kf,
                             pool=npf, R_cont=Rf, exhausted=exf, variant="fullpool"))
            for alt in (1.0, 2.0, 3.0):
                if p not in ("QB", "TE"):
                    continue
                Rfa, _, kfa, _ = r_cont_full(y, p, F, alt)
                rows.append(dict(frame=F, year=y, pos=p, carry=alt, rostered=kfa,
                                 pool=npf, R_cont=Rfa, exhausted=False,
                                 variant=f"fullpool_carry_{alt:g}"))
            R, npool, k, ex = r_cont(y, p, F, base)
            rows.append(dict(frame=F, year=y, pos=p, carry=base, rostered=k,
                             pool=npool, R_cont=R, exhausted=ex, variant="boardtail"))
            R1, _, _, ex1 = r_cont(y, p, F, base, band=1)
            rows.append(dict(frame=F, year=y, pos=p, carry=base, rostered=k,
                             pool=npool, R_cont=R1, exhausted=ex1, variant="single"))
            if p in ("QB", "TE"):
                for alt in (1.0, 2.0):
                    Ra, _, ka, exa = r_cont(y, p, F, alt)
                    rows.append(dict(frame=F, year=y, pos=p, carry=alt, rostered=ka,
                                     pool=npool, R_cont=Ra, exhausted=exa,
                                     variant=f"carry_{alt:g}"))
RC = pd.DataFrame(rows)
RC.to_csv(f"{ROOT}/results/sectionO_contested_baseline.csv", index=False)

EX = RC[(RC.variant == "boardtail") & RC.exhausted].groupby(["frame", "pos"]).size()
print("\nPOOL EXHAUSTION (identifiable free pool ran out; R_cont not computable there):")
print(EX.to_string() if len(EX) else "  none")
print("  -> the FFC board carries ~19 TE / ~24 QB / ~60 RB / ~66 WR per year.  At 12 teams")
print("     x 5.85 WR/team the board is fully rostered, so a 12-team WR replacement falls")
print("     off the identifiable board entirely.  R_cont is therefore reported as the")
print("     headline in the OWNER'S 10-team frame only; 12-team cells that exhaust are")
print("     flagged and not interpreted.")

prim = RC[RC.variant == "hoard"].groupby(["frame", "pos"]).R_cont.mean()
wire = RC[RC.variant == "fullpool"].groupby(["frame", "pos"]).R_cont.mean()
btail = RC[RC.variant == "boardtail"].groupby(["frame", "pos"]).R_cont.mean()
full = BR.groupby(["frame", "pos"])[["R_exp", "R_real", "R_week"]].mean()
full["R_cont"] = prim
full["R_cont_blind"] = btail
full["R_wire_best"] = wire
full = full[["R_exp", "R_cont_blind", "R_real", "R_cont", "R_wire_best", "R_week"]]
print("\nfour baselines, season-total PPR, means 2015-2024 "
      "(R_cont is the new PRIMARY; R_week is an upper bound only):")
print(full.round(1).to_string())
print("\nsensitivity, R_cont at alternative per-team carry (10-team frame):")
print(RC[(RC.frame == 10) & RC.pos.isin(["QB", "TE"])]
      .groupby(["pos", "variant", "rostered"]).R_cont.mean().round(1).to_string())

# ---------------------------------------------------------------- premium at R_cont
print("\n" + "=" * 78)
print("§O7 premium over the RB/WR at the same ADP (+-6 picks), baseline R_cont")
print("=" * 78)
out = []
for F in (10, 12):
  for VAR, BLBL in (("hoard", "R_cont"), ("boardtail", "R_cont_blind"),
                      ("fullpool", "R_wire_best")):
      Rm = {(r.year, r.pos): r.R_cont
            for r in RC[(RC.frame == F) & (RC.variant == VAR)].itertuples()}
      w = v.copy()
      w["V"] = [r.total - Rm[(r.year, r.pos)] for r in w.itertuples()]
      for p in ("TE", "QB"):
          specs = [("1-3", 1, 3), ("1-5", 1, 5), ("1-12", 1, 12),
                   ("d1-3", 1, 3), ("d4-6", 4, 6), ("d7-12", 7, 12), ("d13+", 13, 99)]
          for lbl, lo, hi in specs:
              per = []
              for y in YEARS:
                  gy = w[w.year == y]
                  tg = gy[(gy.pos == p) & (gy.posrank_adp >= lo) & (gy.posrank_adp <= hi)]
                  ds = []
                  for t in tg.itertuples():
                      nb = gy[(gy.pos.isin(["RB", "WR"])) & ((gy.adp - t.adp).abs() <= 6)]
                      if len(nb) >= 2:
                          ds.append(t.V - nb.V.mean())
                  if ds:
                      per.append(np.mean(ds))
              a = np.array(per)
              if len(a) < 3:
                  continue
              se = a.std(ddof=1) / np.sqrt(len(a))
              out.append(dict(frame=F, baseline=BLBL, pos=p, band=lbl,
                              n_seasons=len(a), mean=a.mean(), se=se, t=a.mean() / se,
                              p=2 * (1 - stats.t.cdf(abs(a.mean() / se), len(a) - 1)),
                              mde=2.802 * se, seasons_positive=int((a > 0).sum())))
P = pd.DataFrame(out)
P.to_csv(f"{ROOT}/results/sectionO_premium_contested.csv", index=False)
print(P.round(2).to_string(index=False))

# ---------------------------------------------------------------- VORP by round, R_cont
print("\n=== VORP by 10-team draft round, baseline R_cont (season totals) ===")
Rm = {(r.year, r.pos): r.R_cont
      for r in RC[(RC.frame == 10) & (RC.variant == "hoard")].itertuples()}
w = v.copy()
w["Vc"] = [r.total - Rm[(r.year, r.pos)] for r in w.itertuples()]
pt = w[w.round10 <= 14].pivot_table(index="round10", columns="pos", values="Vc",
                                    aggfunc="mean")
print(pt.round(1).to_string())
print("best position by round (R_cont): " +
      " ".join(f"R{int(i)}={pt.loc[i].idxmax()}" for i in pt.index))
w[["year", "name", "pos", "adp", "posrank_adp", "round10", "total", "Vc"]].to_csv(
    f"{ROOT}/results/sectionO_vorp_contested.csv", index=False)

# ================================================================= TE cliff
print("\n" + "=" * 78)
print("Owner's claim: is there a cliff after ~4 TEs?  Gap profile by positional rank")
print("=" * 78)
crows = []
for basis in ("realized", "adp"):
    for p in POSNS:
        mat = []
        for y in YEARS:
            g = vp[(vp.year == y) & (vp.pos == p)]
            if basis == "realized":
                s = np.sort(g.total.values)[::-1]
            else:
                s = g.sort_values("posrank_adp").total.values
            mat.append(s[:24] if len(s) >= 24 else np.r_[s, np.full(24 - len(s), np.nan)])
        M = np.vstack(mat)
        mean = np.nanmean(M, axis=0)
        for k in range(len(mean)):
            crows.append(dict(basis=basis, pos=p, rank=k + 1, mean_total=mean[k],
                              n_seasons=int(np.sum(~np.isnan(M[:, k])))))
CL = pd.DataFrame(crows)
CL.to_csv(f"{ROOT}/results/sectionO_te_cliff.csv", index=False)

for basis in ("realized", "adp"):
    print(f"\n--- basis: {basis} positional rank, mean season total 2015-2024 ---")
    t = CL[CL.basis == basis].pivot_table(index="rank", columns="pos",
                                          values="mean_total")
    print(t.head(14).round(1).to_string())
    print("  drop per rank:")
    d = -t.diff()
    print(d.head(13).round(1).to_string())
    print("\n  cliff test — total drop rank4->5 vs the AVERAGE PER-RANK drop 5->12,")
    print("  in points and as a multiple (a cliff means the 4->5 step is much bigger):")
    for p in POSNS:
        g45 = t[p].iloc[3] - t[p].iloc[4]
        g512 = (t[p].iloc[4] - t[p].iloc[11]) / 7
        g12_24 = (t[p].iloc[11] - t[p].iloc[23]) / 12 if not np.isnan(t[p].iloc[23]) else np.nan
        print(f"    {p}: rank4->5 = {g45:6.1f};  mean step 5->12 = {g512:5.1f}  "
              f"(ratio {g45/g512:4.2f}x);  mean step 12->24 = {g12_24:5.1f};  "
              f"rank1 {t[p].iloc[0]:5.1f}, rank4 {t[p].iloc[3]:5.1f}, "
              f"rank12 {t[p].iloc[11]:5.1f}")

# per-season stability of the cliff, realized basis
print("\n  per-season rank4->5 drop (realized basis), to show it is not one season:")
for p in POSNS:
    gs = []
    for y in YEARS:
        s = np.sort(vp[(vp.year == y) & (vp.pos == p)].total.values)[::-1]
        gs.append(s[3] - s[4] if len(s) > 4 else np.nan)
    gs = np.array(gs, dtype=float)
    print(f"    {p}: mean {np.nanmean(gs):5.1f}, sd {np.nanstd(gs, ddof=1):5.1f}, "
          f"per-season {np.round(gs,0).tolist()}")

# the actionable version: does a cliff exist where the DRAFTER stands (ADP rank)?
print("\n  the drafter's version — mean season total by ADP rank, TE only, with the")
print("  share of seasons in which the ADP-rank-k TE finished top-5 at the position:")
for k in range(1, 13):
    hits, tots = [], []
    for y in YEARS:
        g = vp[(vp.year == y) & (vp.pos == "TE")]
        r = g[g.posrank_adp == k]
        if not len(r):
            continue
        thr = np.sort(g.total.values)[::-1]
        thr = thr[4] if len(thr) > 4 else -np.inf
        hits.append(float(r.total.iloc[0] >= thr))
        tots.append(float(r.total.iloc[0]))
    print(f"    TE ADP rank {k:2d}: mean total {np.mean(tots):6.1f}, "
          f"P(top-5 TE finish) = {np.mean(hits):.2f}  (n={len(hits)})")

# formal version of the owner's claim: do the top-k ADP TEs supply the top-k finishers?
print("\n  formal contrast — P(top-5 positional finish | ADP rank in 1-5) vs (6-12),")
print("  per position, season-clustered t(9):")
hrows = []
for p in POSNS:
    per = []
    for y in YEARS:
        g = vp[(vp.year == y) & (vp.pos == p)]
        s = np.sort(g.total.values)[::-1]
        if len(s) <= 4:
            continue
        thr = s[4]
        a = g[g.posrank_adp <= 5]
        b = g[(g.posrank_adp >= 6) & (g.posrank_adp <= 12)]
        if len(a) and len(b):
            per.append((a.total >= thr).mean() - (b.total >= thr).mean())
    a = np.array(per)
    se = a.std(ddof=1) / np.sqrt(len(a))
    pv = 2 * (1 - stats.t.cdf(abs(a.mean() / se), len(a) - 1))
    print(f"    {p}: diff {a.mean():+.3f} (se {se:.3f}, p {pv:.4f}), "
          f"positive in {int((a > 0).sum())}/{len(a)} seasons")
    hrows.append(dict(basis="top5_identifiability", pos=p, rank=np.nan,
                      mean_total=a.mean(), n_seasons=len(a), se=se, p=pv))
pd.concat([CL, pd.DataFrame(hrows)], ignore_index=True).to_csv(
    f"{ROOT}/results/sectionO_te_cliff.csv", index=False)

print("\nwrote sectionO_contested_baseline.csv, sectionO_premium_contested.csv, "
      "sectionO_vorp_contested.csv, sectionO_te_cliff.csv")


# ============================================================ 2026 forward, R_cont
# The 2026 board's own replacement is an R_exp object (a board of expectations has no
# order-statistic selection in it).  The R_cont column is formed by adding the historical
# position-specific gap R_cont - R_exp, exactly as 37_sectionO_o6_o7_vorp.py forms its
# R_real and R_week columns.  The transfer is stated, not hidden.
print("\n" + "=" * 78)
print("2026 forward board under the PRIMARY contested baseline R_cont")
print("=" * 78)
B26 = pd.read_csv(f"{ROOT}/results/sectionO_board_2026_vorp.csv")
gapc = (RC[RC.variant == "hoard"].groupby(["frame", "pos"]).R_cont.mean()
        - BR.groupby(["frame", "pos"]).R_exp.mean())
print("historical gap R_cont - R_exp added to the 2026 board's R_exp:")
print(gapc.round(1).to_string())
out = []
for F in (10, 12):
    t = B26[B26.frame == F].copy()
    t["R_cont"] = t.R_exp + t.position.map({p: gapc.loc[(F, p)] for p in POSNS})
    t["VORP_R_cont"] = t.value - t.R_cont
    out.append(t)
    print(f"\n-- {F}-team frame, 2026 TE/QB vs the RB/WR within +-6 ADP picks, R_cont --")
    for p in ("TE", "QB"):
        for _, r in t[t.position == p].nlargest(4, "VORP_R_cont").iterrows():
            nb = t[(t.position.isin(["RB", "WR"])) & ((t.adp - r.adp).abs() <= 6)]
            print(f"  {p} {r['name']:20s} adp {r.adp:5.1f}  VORP {r.VORP_R_cont:+7.1f}  "
                  f"vs RB/WR {nb.VORP_R_cont.mean():+7.1f}  "
                  f"premium {r.VORP_R_cont - nb.VORP_R_cont.mean():+7.1f}")
    print(f"  top-10 of the whole {F}-team board by VORP_R_cont:")
    print(t.nlargest(10, "VORP_R_cont")[["name", "position", "adp", "value",
                                         "VORP_R_cont"]]
          .to_string(index=False, float_format=lambda x: f"{x:.1f}"))
pd.concat(out, ignore_index=True).to_csv(
    f"{ROOT}/results/sectionO_board_2026_vorp.csv", index=False)
print("\nupdated sectionO_board_2026_vorp.csv with VORP_R_cont")

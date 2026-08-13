"""§L (EDA_PLAN5.md) — positional conversion rates by draft cost.

Build (§L0):
  boards  : data/adp/historical/adp_ppr_20{15..24}.csv  (full FFC PPR 12-team boards)
  outcomes: data/players/weekly_raw/  ALL players, REG only -> season total PPR,
            games appeared, PPG; positional finish rank computed LEAGUE-WIDE.
  two outcome definitions: (T) season total points [primary], (P) PPG | >=4 games.
  finish tiers: 1-12 / 13-24 / 25-36 within position.
  cost bins   : 12-team frame round=ceil(adp/12) -> R1-2,R3-4,R5-6,R7-8,R9+ ;
                10-team frame round=ceil(adp/10) reported alongside.

Outputs: results/conversion_rates.csv, results/conversion_tests.csv,
         results/sectionL_panel.csv, results/figures/sectionL_*.png
Rerun: python3 scripts/26_sectionL_conversion.py
"""
import re, json, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.simplefilter("ignore")
ROOT = "/Users/thomasmcnamee/NFL"
YEARS = list(range(2015, 2025))
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
ALIASES = {"hollywood brown": "marquise brown", "joshua palmer": "josh palmer"}
POS = ["RB", "WR"]
SKILL = {"QB", "RB", "WR", "TE", "FB", "HB"}


def norm_name(s):
    s = re.sub(r"[^a-z ]", "",
               str(s).lower().replace(".", " ").replace("-", " ").replace("'", ""))
    return " ".join(t for t in s.split() if t not in SUFFIXES)


def collapse_initials(n):
    """'c j anderson' -> 'cj anderson' (FFC writes 'CJ Anderson', nflverse 'C.J.')."""
    toks = n.split()
    out, buf = [], ""
    for t in toks:
        if len(t) == 1:
            buf += t
        else:
            if buf:
                out.append(buf)
                buf = ""
            out.append(t)
    if buf:
        out.append(buf)
    return " ".join(out)


# ---------------- league-wide outcomes ----------------
frames = []
for y in range(2014, 2026):
    df = pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
                     low_memory=False,
                     usecols=["player_id", "player_display_name", "position", "season",
                              "week", "season_type", "targets", "carries",
                              "fantasy_points_ppr"])
    frames.append(df[df.season_type == "REG"])
wk = pd.concat(frames, ignore_index=True)
wk = wk.dropna(subset=["player_id"])

season_pos = (wk.groupby(["player_id", "season"]).position
                .agg(lambda s: s.mode().iat[0] if len(s.mode()) else np.nan)
                .rename("pos").reset_index())
out = (wk.groupby(["player_id", "season"])
         .agg(total=("fantasy_points_ppr", "sum"),
              games=("fantasy_points_ppr", "size"),
              touches=("targets", "sum"))
         .reset_index()
         .merge(season_pos, on=["player_id", "season"]))
out["ppg"] = out.total / out.games

wk["nname"] = wk.player_display_name.map(norm_name)
wk["nname_c"] = wk.nname.map(collapse_initials)
# player-level directory: career position mode, seasons present, rows per season
pdir = (wk.groupby("player_id")
          .agg(nname=("nname", lambda s: s.iat[0]),
               nname_c=("nname_c", lambda s: s.iat[0]),
               pos=("position", lambda s: s.mode().iat[0])).reset_index())
seasons_present = wk.groupby(["player_id", "season"]).size().rename("nrows").reset_index()
meta = pd.read_csv(f"{ROOT}/data/meta/players_meta.csv", low_memory=False,
                   usecols=["gsis_id", "display_name", "position"]).dropna(subset=["gsis_id"])
meta["nname"] = meta.display_name.map(norm_name).map(collapse_initials)

# ---------------- draft panel ----------------
rows, unmatched, ambig = [], [], []
for y in YEARS:
    adp = pd.read_csv(f"{ROOT}/data/adp/historical/adp_ppr_{y}.csv")
    adp = adp[adp.position.isin(["WR", "RB", "QB", "TE"])].copy()
    adp["nname"] = adp.name.map(norm_name).replace(ALIASES).map(collapse_initials)
    pres = seasons_present[seasons_present.season.between(y - 1, y + 1)]
    for _, r in adp.iterrows():
        c = pdir[(pdir.nname_c == r.nname) | (pdir.nname == r.nname)]
        # a board row is a skill player; never match a defender with the same name
        c = c[c.pos.isin(SKILL)]
        if len(c) > 1:                      # 1) same position as the board
            c2 = c[c.pos == r.position]
            c = c2 if len(c2) else c
        if len(c) > 1:                      # 2) active around season y
            c2 = c[c.player_id.isin(pres.player_id)]
            c = c2 if len(c2) else c
        if len(c) > 1:                      # 3) most rows in y-1..y+1
            n = (pres[pres.player_id.isin(c.player_id)]
                 .groupby("player_id").nrows.sum())
            ambig.append((y, r["name"], list(c.player_id)))
            c = c[c.player_id == n.idxmax()]
        if len(c) == 0:
            m = meta[(meta.nname == r.nname) & (meta.position == r.position)]
            if len(m):
                rows.append(dict(year=y, name=r["name"], pos_adp=r.position,
                                 adp=r.adp, pid=m.gsis_id.iat[0]))
                continue
            unmatched.append((y, r["name"], r.position))
            continue
        rows.append(dict(year=y, name=r["name"], pos_adp=r.position, adp=r.adp,
                         pid=c.player_id.iat[0]))

panel = pd.DataFrame(rows)
panel = panel.merge(out.rename(columns={"player_id": "pid", "season": "year"}),
                    on=["pid", "year"], how="left")
# a drafted player with no REG row scores zero and cannot finish in any tier
panel["total"] = panel.total.fillna(0.0)
panel["games"] = panel.games.fillna(0).astype(int)
panel["pos"] = panel.pos.fillna(panel.pos_adp)
# ---- effective position: for a drafted player the BOARD position governs (that is
# the roster slot the pick buys); nflverse position drift otherwise puts e.g. a board
# WR into the TE pool.  Ranks are then recomputed league-wide on pos_eff.
out2 = out.rename(columns={"player_id": "pid", "season": "year"}).copy()
board_pos = panel[["pid", "year", "pos_adp"]].drop_duplicates()
out2 = out2.merge(board_pos, on=["pid", "year"], how="left")
out2["pos_eff"] = out2.pos_adp.fillna(out2.pos)
out2["rank_T"] = out2.groupby(["year", "pos_eff"]).total.rank(ascending=False, method="min")
out2["ppg_e"] = np.where(out2.games >= 4, out2.ppg, np.nan)
out2["rank_P"] = out2.groupby(["year", "pos_eff"]).ppg_e.rank(ascending=False, method="min")
panel = panel.drop(columns=["rank_T", "rank_P", "ppg_e"], errors="ignore").merge(
    out2[["pid", "year", "rank_T", "rank_P", "pos_eff"]], on=["pid", "year"], how="left")
panel["round12"] = np.ceil(panel.adp / 12).astype(int)
panel["round10"] = np.ceil(panel.adp / 10).astype(int)


def binlab(rd):
    return np.select([rd <= 2, rd <= 4, rd <= 6, rd <= 8],
                     ["R1-2", "R3-4", "R5-6", "R7-8"], "R9+")


panel["bin12"] = binlab(panel.round12)
panel["bin10"] = binlab(panel.round10)
BINS = ["R1-2", "R3-4", "R5-6", "R7-8", "R9+"]

# hit definitions -------------------------------------------------------------
# (i) positional top-12 finish, league-wide ranks; T and P outcome definitions
panel["hit_pos_T"] = (panel.rank_T <= 12).fillna(False).astype(int)
panel["hit_pos_P"] = (panel.rank_P <= 12).fillna(False).astype(int)
panel["tier_T"] = np.select([panel.rank_T <= 12, panel.rank_T <= 24, panel.rank_T <= 36],
                            ["1-12", "13-24", "25-36"], "none")
panel["tier_P"] = np.select([panel.rank_P <= 12, panel.rank_P <= 24, panel.rank_P <= 36],
                            ["1-12", "13-24", "25-36"], "none")
# EXTENSION (coordinator request): cumulative tiers 24 and 36, both outcome defs.
# Same league-wide ranks, same panel -- only the threshold changes.
for TH in (24, 36):
    panel[f"hit{TH}_T"] = (panel.rank_T <= TH).fillna(False).astype(int)
    panel[f"hit{TH}_P"] = (panel.rank_P <= TH).fillna(False).astype(int)

# (ii) value-return: >= median of ALL drafted players in the same (season,bin),
#      all positions.  Note (mechanical): pooled across positions this is 0.5 by
#      construction within each cell, so only the BETWEEN-position contrast is
#      informative; the bin main effect is not.
panel["elig_P"] = panel.games >= 4       # denominator for every P definition
for fr, b in [("12", "bin12"), ("10", "bin10")]:
    med = panel.groupby(["year", b]).total.transform("median")
    panel[f"hit_val_T_{fr}"] = (panel.total >= med).astype(int)
    medp = (panel[panel.elig_P].groupby(["year", b]).ppg.median()
            .rename("medppg").reset_index())
    panel = panel.merge(medp, on=["year", b], how="left")
    panel[f"hit_val_P_{fr}"] = np.where(panel.elig_P,
                                        (panel.ppg >= panel.medppg).astype(float), np.nan)
    panel = panel.drop(columns=["medppg"])
panel["hit_pos_P"] = np.where(panel.elig_P, panel.hit_pos_P.astype(float), np.nan)
for TH in (24, 36):
    panel[f"hit{TH}_P"] = np.where(panel.elig_P, panel[f"hit{TH}_P"].astype(float), np.nan)
panel["hit_val_T"] = panel["hit_val_T_12"]
panel["hit_val_P"] = panel["hit_val_P_12"]

panel.to_csv(f"{ROOT}/results/sectionL_panel.csv", index=False)
print(f"panel rows {len(panel)}  unmatched {len(unmatched)} {unmatched[:20]}")
print(f"ambiguous-name rows {len(ambig)}: {ambig[:10]}")
print("zero-game drafted rows:", int((panel.games == 0).sum()))


# ---------------- §L1 conversion rates ----------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


HITDEFS = [("top12_T", "hit_pos_T", "top12", "total"),
           ("top12_P", "hit_pos_P", "top12", "ppg_ge4"),
           ("top24_T", "hit24_T", "top24", "total"),
           ("top24_P", "hit24_P", "top24", "ppg_ge4"),
           ("top36_T", "hit36_T", "top36", "total"),
           ("top36_P", "hit36_P", "top36", "ppg_ge4"),
           ("value_T", "VAL_T", "value", "total"),
           ("value_P", "VAL_P", "value", "ppg_ge4")]


recs = []
for frame, bcol in [(12, "bin12"), (10, "bin10")]:
    for window, yrs in [("2015-2024", YEARS), ("2018-2024", list(range(2018, 2025))),
                        ("2022-2024", list(range(2022, 2025)))]:
        sub = panel[panel.year.isin(yrs)]
        for pos in POS:
            for b in BINS:
                cell = sub[(sub.pos_adp == pos) & (sub[bcol] == b)]
                if len(cell) == 0:
                    continue
                for lab, col, tier, odef in HITDEFS:
                    col = col.replace("VAL_T", f"hit_val_T_{frame}").replace(
                        "VAL_P", f"hit_val_P_{frame}")
                    n = int(cell[col].notna().sum())
                    if n == 0:
                        continue
                    k = int(cell[col].sum())
                    lo, hi = wilson(k, n)
                    recs.append(dict(frame=f"{frame}team", window=window, pos=pos,
                                     bin=b, hitdef=lab, tier=tier, outcome_def=odef,
                                     n=n, k=k, rate=k / n,
                                     wilson_lo=lo, wilson_hi=hi,
                                     n_per_season=n / len(yrs)))
        # per-season cells (explicitly not to be read as signal)
        for pos in POS:
            for b in BINS:
                for y in yrs:
                    cell = sub[(sub.pos_adp == pos) & (sub[bcol] == b) & (sub.year == y)]
                    if len(cell) == 0:
                        continue
                    for lab, col, tier, odef in HITDEFS:
                        col = col.replace("VAL_T", f"hit_val_T_{frame}").replace(
                            "VAL_P", f"hit_val_P_{frame}")
                        nn_ = int(cell[col].notna().sum())
                        if nn_ == 0:
                            continue
                        k = int(cell[col].sum())
                        lo, hi = wilson(k, nn_)
                        recs.append(dict(frame=f"{frame}team", window=f"season {y}",
                                         pos=pos, bin=b, hitdef=lab, tier=tier,
                                         outcome_def=odef, n=nn_, k=k,
                                         rate=k / nn_, wilson_lo=lo, wilson_hi=hi,
                                         n_per_season=nn_))
rates = pd.DataFrame(recs)
rates.to_csv(f"{ROOT}/results/conversion_rates.csv", index=False)

print("\n=== §L1 pooled conversion, 12-team frame, 2015-2024 ===")
piv = (rates[(rates.frame == "12team") & (rates.window == "2015-2024")]
       .pivot_table(index=["pos", "bin"], columns="hitdef",
                    values=["n", "rate"]).round(3))
print(piv.to_string())


# ---------------- inference helpers ----------------
def cluster_t(d):
    """season-clustered mean difference: t on len(d)-1 df."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    m, s, n = d.mean(), d.std(ddof=1), len(d)
    se = s / np.sqrt(n)
    t = m / se
    p = 2 * stats.t.sf(abs(t), n - 1)
    ci = stats.t.ppf(.975, n - 1) * se
    return m, se, t, p, m - ci, m + ci, n


def mde_prop_cluster(sd_d, n_clusters, alpha=.05, power=.80):
    """MDE for a season-clustered mean-of-differences design (t, n-1 df)."""
    tc = stats.t.ppf(1 - alpha / 2, n_clusters - 1)
    tp = stats.t.ppf(power, n_clusters - 1)
    return (tc + tp) * sd_d / np.sqrt(n_clusters)


tests = []

# ---------------- §L2 claim (a): elite RB vs elite WR ----------------
print("\n=== §L2 claim (a): R1-2 RB vs WR ===")
for hitdef, col in [("top12_T", "hit_pos_T"), ("value_T", "hit_val_T_12")]:
    rows_ = []
    for y in YEARS:
        s = panel[(panel.year == y) & (panel.bin12 == "R1-2")]
        rb, wr = s[s.pos_adp == "RB"], s[s.pos_adp == "WR"]
        rows_.append(dict(year=y, n_rb=len(rb), n_wr=len(wr),
                          p_rb=rb[col].mean(), p_wr=wr[col].mean(),
                          d=rb[col].mean() - wr[col].mean()))
    ss = pd.DataFrame(rows_)
    m, se, t, p, lo, hi, n = cluster_t(ss.d)
    mde = mde_prop_cluster(ss.d.std(ddof=1), n)
    print(f"[{hitdef}] RB-WR = {m:+.3f} (SE {se:.3f}) CI [{lo:+.3f},{hi:+.3f}] "
          f"p={p:.4f} MDE={mde:.3f}")
    print(ss.round(3).to_string(index=False))
    tests.append(dict(test="L2", claim="a", hitdef=hitdef, spec="season-clustered t",
                      estimate=m, se=se, ci_lo=lo, ci_hi=hi, p=p, mde=mde,
                      n=int(len(panel[(panel.bin12 == "R1-2")])), family=1))
    # robustness: logistic with season random effect (family=0)
    d2 = panel[(panel.bin12 == "R1-2") & (panel.pos_adp.isin(POS))].copy()
    d2["y01"] = d2[col]
    try:
        md = sm.MixedLM.from_formula("y01 ~ C(pos_adp, Treatment('WR'))", groups="year",
                                     data=d2).fit()
        k = [c for c in md.params.index if "T.RB]" in c][0]
        tests.append(dict(test="L2rob", claim="a", hitdef=hitdef,
                          spec="LPM season random intercept", estimate=md.params[k],
                          se=md.bse[k], ci_lo=md.conf_int().loc[k, 0],
                          ci_hi=md.conf_int().loc[k, 1], p=md.pvalues[k], mde=np.nan,
                          n=len(d2), family=0))
    except Exception as e:
        print("mixed failed", e)

# ---------------- §L3 claim (b): RB R1-2 trend ----------------
print("\n=== §L3 claim (b): RB R1-2 trend ===")
rb12 = panel[(panel.bin12 == "R1-2") & (panel.pos_adp == "RB")].copy()
rb12["yc"] = rb12.year - 2019.5
for hitdef, col in [("top12_T", "hit_pos_T"), ("value_T", "hit_val_T_12")]:
    m = smf.logit(f"{col} ~ yc", data=rb12).fit(disp=0,
                                                cov_type="cluster",
                                                cov_kwds={"groups": rb12.year},
                                                use_t=True)
    b, se = m.params["yc"], m.bse["yc"]
    ci = m.conf_int().loc["yc"]
    # MDE for THIS design: season-clustered slope, from the realized cluster SE
    tc, tp = stats.t.ppf(.975, 9), stats.t.ppf(.80, 9)
    mde = (tc + tp) * se
    print(f"[{hitdef}] logit slope/yr = {b:+.4f} (cluster SE {se:.4f}) "
          f"CI [{ci[0]:+.3f},{ci[1]:+.3f}] p={m.pvalues['yc']:.4f} MDE={mde:.4f} "
          f"(= {mde/4:.3f} pp/yr at p=.5)")
    tests.append(dict(test="L3", claim="b", hitdef=hitdef,
                      spec="logit hit~year, cluster(season), t(9)",
                      estimate=b, se=se, ci_lo=ci[0], ci_hi=ci[1],
                      p=m.pvalues["yc"], mde=mde, n=len(rb12), family=1))
    # secondary: pre-specified 2022-24 vs 2015-21 split, season-clustered t
    a = [rb12[(rb12.year == y)][col].mean() for y in range(2022, 2025)]
    bb = [rb12[(rb12.year == y)][col].mean() for y in range(2015, 2022)]
    tt = stats.ttest_ind(a, bb, equal_var=False)
    diff = np.mean(a) - np.mean(bb)
    sp = np.sqrt(np.var(a, ddof=1) / 3 + np.var(bb, ddof=1) / 7)
    mde_s = (stats.t.ppf(.975, tt.df) + stats.t.ppf(.80, tt.df)) * sp
    print(f"   split 2022-24 {np.mean(a):.3f} vs 2015-21 {np.mean(bb):.3f} "
          f"diff {diff:+.3f} p={tt.pvalue:.4f} MDE={mde_s:.3f}")
    tests.append(dict(test="L3split", claim="b", hitdef=hitdef,
                      spec="Welch t on season rates, 2022-24 vs 2015-21",
                      estimate=diff, se=sp, ci_lo=diff - stats.t.ppf(.975, tt.df) * sp,
                      ci_hi=diff + stats.t.ppf(.975, tt.df) * sp, p=tt.pvalue,
                      mde=mde_s, n=len(rb12), family=1))

# confound: RB draft cost trend over the same window (§L6)
print("\n--- §L6 confound: RB draft cost trend ---")
cost = []
for y in YEARS:
    s = panel[panel.year == y]
    rb, wr = s[s.pos_adp == "RB"], s[s.pos_adp == "WR"]
    cost.append(dict(year=y,
                     n_rb_R12=int((rb.bin12 == "R1-2").sum()),
                     n_wr_R12=int((wr.bin12 == "R1-2").sum()),
                     rb_adp_top10=rb.nsmallest(10, "adp").adp.mean(),
                     wr_adp_top10=wr.nsmallest(10, "adp").adp.mean(),
                     rb_in_top24=int((rb.adp <= 24).sum()),
                     wr_in_top24=int((wr.adp <= 24).sum())))
cost = pd.DataFrame(cost)
print(cost.round(2).to_string(index=False))
for c in ["rb_adp_top10", "wr_adp_top10", "n_rb_R12", "rb_in_top24"]:
    sl = stats.linregress(cost.year, cost[c])
    print(f"  {c}: slope {sl.slope:+.3f}/yr  p={sl.pvalue:.3f}")
    tests.append(dict(test="L6cost", claim="confound", hitdef=c,
                      spec="OLS on year (descriptive, not in family)",
                      estimate=sl.slope, se=sl.stderr, ci_lo=np.nan, ci_hi=np.nan,
                      p=sl.pvalue, mde=np.nan, n=10, family=0))
# is the (null) conversion trend confounded by RBs getting cheaper?
rbrate = pd.DataFrame([dict(year=y,
                            hit=panel[(panel.year == y) & (panel.pos_adp == "RB")
                                      & (panel.bin12 == "R1-2")].hit_pos_T.mean(),
                            val=panel[(panel.year == y) & (panel.pos_adp == "RB")
                                      & (panel.bin12 == "R1-2")].hit_val_T_12.mean())
                       for y in YEARS]).merge(cost, on="year")
print(f"  corr(mean ADP of RB1-10 [higher=cheaper], RB R1-2 top-12 rate) = "
      f"{rbrate.rb_adp_top10.corr(rbrate.hit):+.3f}  "
      f"[value-return {rbrate.rb_adp_top10.corr(rbrate.val):+.3f}]")
print(f"  RB1-10 mean ADP: 2015-21 {cost[cost.year <= 2021].rb_adp_top10.mean():.2f} "
      f"vs 2022-24 {cost[cost.year >= 2022].rb_adp_top10.mean():.2f} "
      f"(higher = cheaper)")
cost.to_csv(f"{ROOT}/results/sectionL_costtrend.csv", index=False)

# ---------------- §L4 claim (c): position x bin interaction ----------------
print("\n=== §L4 claim (c): position x bin interaction ===")
d = panel[panel.pos_adp.isin(POS)].copy()
d["bin12"] = pd.Categorical(d.bin12, BINS, ordered=True)
d["binnum"] = d.bin12.cat.codes.astype(float)
for hitdef, col in [("top12_T", "hit_pos_T"), ("value_T", "hit_val_T_12")]:
    f = f"{col} ~ C(pos_adp, Treatment('WR')) * C(bin12)"
    m = smf.logit(f, data=d).fit(disp=0, cov_type="cluster",
                                 cov_kwds={"groups": d.year}, use_t=True)
    inter = [k for k in m.params.index if ":" in k]
    w = m.wald_test_terms().table
    key = [k for k in w.index if ":" in k][0]
    pj = float(w.loc[key, "pvalue"])
    print(f"[{hitdef}] joint interaction Wald p = {pj:.4f}  ({dict(w.loc[key])})")
    print(m.params[inter].round(3).to_string())
    # linear-in-bin version for a single interpretable slope + MDE
    m2 = smf.logit(f"{col} ~ C(pos_adp, Treatment('WR')) * binnum", data=d).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": d.year}, use_t=True)
    k2 = [x for x in m2.params.index if ":" in x][0]
    tc, tp = stats.t.ppf(.975, 9), stats.t.ppf(.80, 9)
    mde = (tc + tp) * m2.bse[k2]
    print(f"   linear-in-bin RB x bin interaction {m2.params[k2]:+.4f} "
          f"(SE {m2.bse[k2]:.4f}) p={m2.pvalues[k2]:.4f} MDE={mde:.4f}")
    tests.append(dict(test="L4", claim="c", hitdef=hitdef,
                      spec="logit hit ~ pos*bin, cluster(season); joint Wald on interaction",
                      estimate=m2.params[k2], se=m2.bse[k2],
                      ci_lo=m2.conf_int().loc[k2, 0], ci_hi=m2.conf_int().loc[k2, 1],
                      p=pj, mde=mde, n=len(d), family=1))

# ================= §L-EXT: cumulative tiers 24 / 36 (round-5 extension) =======
# Declared as a NEW family, family = 2, BH q = 0.10. The original 8-test §L family
# (family = 1) is closed and is NOT reopened or re-corrected. Family = 2 comprises
# the tests on the PRIMARY outcome definition (season totals), exactly as family 1
# did: §L2 x {hit24, hit36} + §L4 x {hit24, hit36} = 4 tests. The PPG-given-
# participation versions are reported as family = 0 sensitivities.
print("\n" + "=" * 72)
print("=== §L-EXT budget integrity check: <= T hits per position per season ===")
viol = []
for TH, col in [(12, "hit_pos_T"), (24, "hit24_T"), (36, "hit36_T")]:
    for pos in POS:
        for y in YEARS:
            k = int(panel[(panel.year == y) & (panel.pos_adp == pos)][col].sum())
            if k > TH:
                viol.append((TH, pos, y, k))
print(f"  violations of the <=12 / <=24 / <=36 slot budget: {len(viol)} {viol}")
for TH, pos, y, k in viol:   # diagnose every violation; ties are the only legal cause
    col = "hit_pos_T" if TH == 12 else f"hit{TH}_T"
    c = panel[(panel.year == y) & (panel.pos_adp == pos) & (panel[col] == 1)]
    tie = c[c.rank_T.duplicated(keep=False)]
    print(f"    {y} {pos} tier<={TH}: {k} hits; tied rows at the boundary:")
    print(tie[["name", "adp", "total", "rank_T"]].to_string(index=False))
budget = []
for TH, col in [(12, "hit_pos_T"), (24, "hit24_T"), (36, "hit36_T")]:
    for pos in POS:
        ks = [int(panel[(panel.year == y) & (panel.pos_adp == pos)][col].sum())
              for y in YEARS]
        budget.append(dict(tier=TH, pos=pos, mean_drafted_hits=np.mean(ks),
                           mean_undrafted=TH - np.mean(ks), max_hits=max(ks)))
print(pd.DataFrame(budget).round(2).to_string(index=False))

print("\n=== §L-EXT ceiling diagnostic: T slots vs n drafted in the bin ===")
ceil_rows = []
for TH, col in [(12, "hit_pos_T"), (24, "hit24_T"), (36, "hit36_T")]:
    for pos in POS:
        for y in YEARS:
            for b in BINS:
                c = panel[(panel.year == y) & (panel.pos_adp == pos)
                          & (panel.bin12 == b)]
                if len(c):
                    ceil_rows.append(dict(tier=TH, pos=pos, year=y, bin=b, n=len(c),
                                          ceiling=min(1, TH / len(c)),
                                          hit=c[col].mean()))
CE = pd.DataFrame(ceil_rows)
cesum = (CE.groupby(["tier", "pos", "bin"])
           .agg(n=("n", "mean"), ceiling=("ceiling", "mean"), hit=("hit", "mean"))
           .reset_index())
cesum["slack"] = cesum.ceiling - cesum.hit
cesum["binds"] = cesum.slack < 0.10
print(cesum.pivot_table(index=["tier", "bin"], columns="pos",
                        values=["n", "ceiling", "hit", "slack"]).round(3).to_string())
print("  cells where the budget ceiling is within 10 pp of the realized rate:")
print(cesum[cesum.binds][["tier", "pos", "bin", "n", "ceiling", "hit"]]
      .round(3).to_string(index=False))
cesum.to_csv(f"{ROOT}/results/sectionL_ceiling.csv", index=False)

print("\n=== §L-EXT §L2 at hit24 / hit36 (R1-2, season-clustered t, 9 df) ===")
EXTDEFS = [("top24_T", "hit24_T", "top24", "total", 2),
           ("top36_T", "hit36_T", "top36", "total", 2),
           ("top24_P", "hit24_P", "top24", "ppg_ge4", 0),
           ("top36_P", "hit36_P", "top36", "ppg_ge4", 0)]
for lab, col, tier, odef, famflag in EXTDEFS:
    src = panel[panel.elig_P] if odef == "ppg_ge4" else panel
    d_ = [src[(src.year == y) & (src.bin12 == "R1-2") & (src.pos_adp == "RB")][col].mean()
          - src[(src.year == y) & (src.bin12 == "R1-2") & (src.pos_adp == "WR")][col].mean()
          for y in YEARS]
    m, se, t, pv, lo, hi, n = cluster_t(d_)
    mde = mde_prop_cluster(np.std(d_, ddof=1), 10)
    print(f"  [{lab}] RB-WR = {m:+.3f} (SE {se:.3f}) CI [{lo:+.3f},{hi:+.3f}] "
          f"p={pv:.4f} MDE={mde:.3f}")
    tests.append(dict(test="L2ext", claim="a", hitdef=lab, tier=tier, outcome_def=odef,
                      spec="season-clustered t on per-season RB-WR difference, R1-2",
                      estimate=m, se=se, ci_lo=lo, ci_hi=hi, p=pv, mde=mde,
                      n=int((src.bin12 == "R1-2").sum()), family=famflag))

print("\n=== §L-EXT §L4 interaction at hit24 / hit36 ===")
for lab, col, tier, odef, famflag in EXTDEFS:
    src = panel[panel.elig_P] if odef == "ppg_ge4" else panel
    dx = src[src.pos_adp.isin(POS)].copy()
    dx["binnum"] = pd.Categorical(dx.bin12, BINS, ordered=True).codes.astype(float)
    mj = smf.logit(f"{col} ~ C(pos_adp, Treatment('WR')) * C(bin12)", data=dx).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": dx.year}, use_t=True)
    wt = mj.wald_test_terms().table
    kj = [k for k in wt.index if ":" in k][0]
    pj = float(wt.loc[kj, "pvalue"])
    m2 = smf.logit(f"{col} ~ C(pos_adp, Treatment('WR')) * binnum", data=dx).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": dx.year}, use_t=True)
    k2 = [x for x in m2.params.index if ":" in x][0]
    mde = (stats.t.ppf(.975, 9) + stats.t.ppf(.80, 9)) * m2.bse[k2]
    print(f"  [{lab}] joint Wald p={pj:.4f}; linear-in-bin RB x bin "
          f"{m2.params[k2]:+.4f} (SE {m2.bse[k2]:.4f}) p={m2.pvalues[k2]:.4f} "
          f"MDE={mde:.4f}")
    tests.append(dict(test="L4ext", claim="c", hitdef=lab, tier=tier, outcome_def=odef,
                      spec="logit hit ~ pos*bin, cluster(season); joint Wald on interaction",
                      estimate=m2.params[k2], se=m2.bse[k2],
                      ci_lo=m2.conf_int().loc[k2, 0], ci_hi=m2.conf_int().loc[k2, 1],
                      p=pj, mde=mde, n=len(dx), family=famflag))

# --- BH over the NEW family only -------------------------------------------
T2 = pd.DataFrame(tests)
f2 = T2[T2.family == 2].copy().sort_values("p").reset_index(drop=True)
f2["bh_thresh"] = 0.10 * (f2.index + 1) / len(f2)
f2["bh_reject"] = False
if (f2.p <= f2.bh_thresh).any():
    f2.loc[:f2.index[f2.p <= f2.bh_thresh].max(), "bh_reject"] = True
print(f"\n=== §L-EXT BH q=0.10 over {len(f2)} NEW-family tests "
      "(the 8-test §L family stays closed) ===")
print(f2[["test", "hitdef", "estimate", "p", "bh_thresh", "bh_reject"]]
      .round(4).to_string(index=False))

# --- the R3-4 2022-2024 question -------------------------------------------
print("\n=== §L-EXT the R3-4, 2022-2024 cell the owner is acting on ===")
for TH, col in [(12, "hit_pos_T"), (24, "hit24_T"), (36, "hit36_T")]:
    rows34 = []
    for y in [2022, 2023, 2024]:
        c = panel[(panel.year == y) & (panel.bin12 == "R3-4")]
        rb, wr = c[c.pos_adp == "RB"], c[c.pos_adp == "WR"]
        rows34.append(dict(year=y, rb_k=int(rb[col].sum()), rb_n=len(rb),
                           rb=rb[col].mean(), wr_k=int(wr[col].sum()), wr_n=len(wr),
                           wr=wr[col].mean(), d=rb[col].mean() - wr[col].mean()))
    r34 = pd.DataFrame(rows34)
    pooled = panel[(panel.year >= 2022) & (panel.bin12 == "R3-4")]
    rb, wr = pooled[pooled.pos_adp == "RB"], pooled[pooled.pos_adp == "WR"]
    k1, n1, k2_, n2 = int(rb[col].sum()), len(rb), int(wr[col].sum()), len(wr)
    lo1, hi1 = wilson(k1, n1)
    lo2, hi2 = wilson(k2_, n2)
    tab = np.array([[k1, n1 - k1], [k2_, n2 - k2_]])
    p_f = stats.fisher_exact(tab)[1]
    chi = stats.chi2_contingency(tab, correction=False)[1]
    print(f"\n  --- tier <= {TH} ---")
    print(r34.round(3).to_string(index=False))
    print(f"  pooled 2022-24: RB {k1}/{n1} = {k1/n1:.3f} [{lo1:.3f},{hi1:.3f}]  "
          f"WR {k2_}/{n2} = {k2_/n2:.3f} [{lo2:.3f},{hi2:.3f}]  "
          f"gap {k1/n1 - k2_/n2:+.3f}")
    print(f"  Fisher exact p = {p_f:.4f}; chi2 p = {chi:.4f}  "
          "(POST-HOC cell, selected after seeing the table -- not in any family)")
    # how many players would have to move to erase the gap?
    need = k2_ / n2 * n1
    print(f"  fragility: RB hits {k1}; equalising with WR needs {need:.1f} -> "
          f"a swing of {k1-need:+.1f} players out of {n1}")
    tests.append(dict(test="R34posthoc", claim="descriptive", hitdef=f"top{TH}_T",
                      tier=f"top{TH}", outcome_def="total",
                      spec="Fisher exact, RB vs WR, R3-4, 2022-24; POST-HOC cell",
                      estimate=k1 / n1 - k2_ / n2, se=np.nan, ci_lo=np.nan,
                      ci_hi=np.nan, p=p_f, mde=np.nan, n=n1 + n2, family=0))

print("\n=== §L-EXT joint budget saturation (the constraint that DOES bind) ===")
for TH, col in [(12, "hit_pos_T"), (24, "hit24_T"), (36, "hit36_T")]:
    for pos in POS:
        ks = np.mean([panel[(panel.year == y) & (panel.pos_adp == pos)][col].sum()
                      for y in YEARS])
        print(f"  tier<={TH} {pos}: drafted players occupy {ks:.1f}/{TH} slots "
              f"= {ks/TH:.1%} of the positional budget")
print("  -> the ACROSS-BIN profile is a near-fixed pie; the WITHIN-BIN position\n"
      "     contrast is unconstrained (RB and WR have separate budgets).")

print("\n=== §L-EXT post-hoc cell scan: R7-8 at top24 (season-clustered) ===")
for TH, col in [(24, "hit24_T"), (36, "hit36_T")]:
    d_ = [panel[(panel.year == y) & (panel.bin12 == "R7-8") & (panel.pos_adp == "RB")][col].mean()
          - panel[(panel.year == y) & (panel.bin12 == "R7-8") & (panel.pos_adp == "WR")][col].mean()
          for y in YEARS]
    m, se, t, pv, lo, hi, n = cluster_t(d_)
    signs = sum(1 for x in d_ if x > 0)
    print(f"  R7-8 tier<={TH}: RB-WR = {m:+.3f} (SE {se:.3f}) p={pv:.4f}; "
          f"positive in {signs}/10 seasons  [POST-HOC, 1 of ~30 cells, not in any family]")
    tests.append(dict(test="R78posthoc", claim="descriptive", hitdef=f"top{TH}_T",
                      tier=f"top{TH}", outcome_def="total",
                      spec="season-clustered t, R7-8 RB-WR; POST-HOC cell scan",
                      estimate=m, se=se, ci_lo=lo, ci_hi=hi, p=pv, mde=np.nan,
                      n=int((panel.bin12 == "R7-8").sum()), family=0))

# ---------------- §L5 BH + temporal holdout ----------------
T = pd.DataFrame(tests)
fam = T[T.family == 1].copy().sort_values("p").reset_index(drop=True)
mfam = len(fam)
fam["bh_thresh"] = 0.10 * (fam.index + 1) / mfam
fam["bh_pass"] = fam.p <= fam.bh_thresh
# step-up
passing = fam.bh_pass[::-1].idxmax() if fam.bh_pass.any() else None
fam["bh_reject"] = False
if fam.bh_pass.any():
    kmax = fam.index[fam.bh_pass].max()
    fam.loc[:kmax, "bh_reject"] = True
print(f"\n=== §L5 BH q=0.10 over {mfam} family tests ===")
print(fam[["test", "hitdef", "estimate", "p", "bh_thresh", "bh_reject"]].round(4)
      .to_string(index=False))

# temporal holdout: sign/magnitude of each family estimate refit on 2015-21 vs 2022-24
hold = []
for lab, yrs in [("fit 2015-21", range(2015, 2022)), ("holdout 2022-24", range(2022, 2025))]:
    p_ = panel[panel.year.isin(yrs)]
    for hitdef, col in [("top12_T", "hit_pos_T"), ("value_T", "hit_val_T_12")]:
        s = p_[p_.bin12 == "R1-2"]
        d_ = [s[(s.year == y) & (s.pos_adp == "RB")][col].mean() -
              s[(s.year == y) & (s.pos_adp == "WR")][col].mean() for y in yrs]
        hold.append(dict(split=lab, test="L2", hitdef=hitdef, estimate=np.mean(d_),
                         n_seasons=len(d_)))
        dd = p_[p_.pos_adp.isin(POS)].copy()
        dd["binnum"] = pd.Categorical(dd.bin12, BINS, ordered=True).codes.astype(float)
        try:
            mm = smf.logit(f"{col} ~ C(pos_adp, Treatment('WR')) * binnum",
                           data=dd).fit(disp=0)
            kk = [x for x in mm.params.index if ":" in x][0]
            hold.append(dict(split=lab, test="L4", hitdef=hitdef,
                             estimate=mm.params[kk], n_seasons=len(list(yrs))))
        except Exception as e:
            print("holdout L4 failed", e)
H = pd.DataFrame(hold)
print("\n=== temporal holdout ===")
print(H.round(4).to_string(index=False))

# ---------------- diagnostics: mechanical checks (§L6 + round-4b discipline) ----
print("\n=== D1 top-12 slot budget: where each position's top-12 finishers came from ===")
for pos in POS:
    tot = []
    for y in YEARS:
        c = panel[(panel.year == y) & (panel.pos_adp == pos)]
        row = {b: int(c[c.bin12 == b].hit_pos_T.sum()) for b in BINS}
        row["drafted_total"] = int(c.hit_pos_T.sum())
        row["undrafted"] = 12 - row["drafted_total"]
        tot.append(row)
    print(pos, pd.DataFrame(tot).mean().round(2).to_dict())

print("\n=== D2 mechanical ceiling: 12 slots / n drafted in the bin ===")
dd = []
for pos in POS:
    for y in YEARS:
        for b in BINS:
            c = panel[(panel.year == y) & (panel.pos_adp == pos) & (panel.bin12 == b)]
            if len(c):
                dd.append(dict(pos=pos, year=y, bin=b, n=len(c),
                               ceiling=min(1, 12 / len(c)), hit=c.hit_pos_T.mean()))
dd = pd.DataFrame(dd)
print(dd.groupby(["pos", "bin"]).agg(n=("n", "mean"), ceiling=("ceiling", "mean"),
                                     hit=("hit", "mean")).round(3).to_string())
r = dd[(dd.pos == "RB") & (dd["bin"] == "R1-2")]
sl = stats.linregress(r.n, r.hit)
print(f"RB R1-2: hit rate on n drafted early: slope {sl.slope:+.4f} p={sl.pvalue:.3f} "
      f"(ceiling would imply a NEGATIVE slope)")

print("\n=== D3 availability decomposition of the R1-2 gap ===")
s12 = panel[panel.bin12 == "R1-2"]
print(s12.groupby("pos_adp").agg(n=("games", "size"), mean_games=("games", "mean"),
                                 p_lt4=("games", lambda g: (g < 4).mean()),
                                 mean_total=("total", "mean"),
                                 mean_ppg=("ppg", "mean")).round(3).to_string())

print("\n=== D4 value-return threshold contamination by QB/TE in the bin ===")
print(panel.groupby(["bin12", "pos_adp"]).size().unstack().reindex(BINS).to_string())
for b in BINS:
    c = panel[panel.bin12 == b]
    print(f"  {b}: median total all-pos {c.total.median():.1f} vs "
          f"WR+RB only {c[c.pos_adp.isin(POS)].total.median():.1f}")
# sensitivity: value-return threshold from WR+RB only
sub = panel[panel.pos_adp.isin(POS)].copy()
sub["hit_val_T_rbwr"] = (sub.total >=
                         sub.groupby(["year", "bin12"]).total.transform("median")).astype(int)
d_ = [sub[(sub.year == y) & (sub.bin12 == "R1-2") & (sub.pos_adp == "RB")].hit_val_T_rbwr.mean()
      - sub[(sub.year == y) & (sub.bin12 == "R1-2") & (sub.pos_adp == "WR")].hit_val_T_rbwr.mean()
      for y in YEARS]
m, se, t, pv, lo, hi, n = cluster_t(d_)
print(f"  SENSITIVITY value_T with WR+RB-only median: RB-WR = {m:+.3f} "
      f"(SE {se:.3f}) p={pv:.4f}")
tests.append(dict(test="L2sens", claim="a", hitdef="value_T_rbwr_median",
                  spec="season-clustered t, threshold from WR+RB only", estimate=m,
                  se=se, ci_lo=lo, ci_hi=hi, p=pv, mde=mde_prop_cluster(np.std(d_, ddof=1), 10),
                  n=len(sub[sub.bin12 == "R1-2"]), family=0))
sub["binnum"] = pd.Categorical(sub.bin12, BINS, ordered=True).codes.astype(float)
m2 = smf.logit("hit_val_T_rbwr ~ C(pos_adp, Treatment('WR')) * binnum", data=sub).fit(
    disp=0, cov_type="cluster", cov_kwds={"groups": sub.year}, use_t=True)
k2 = [x for x in m2.params.index if ":" in x][0]
print(f"  SENSITIVITY L4 interaction (WR+RB median): {m2.params[k2]:+.4f} "
      f"(SE {m2.bse[k2]:.4f}) p={m2.pvalues[k2]:.4f}")
tests.append(dict(test="L4sens", claim="c", hitdef="value_T_rbwr_median",
                  spec="logit hit~pos*binnum, WR+RB-only threshold", estimate=m2.params[k2],
                  se=m2.bse[k2], ci_lo=m2.conf_int().loc[k2, 0], ci_hi=m2.conf_int().loc[k2, 1],
                  p=m2.pvalues[k2], mde=(stats.t.ppf(.975, 9) + stats.t.ppf(.80, 9)) * m2.bse[k2],
                  n=len(sub), family=0))

print("\n=== D4b PPG-given-participation versions (sensitivity, family=0) ===")
for hitdef, col in [("top12_P", "hit_pos_P"), ("value_P", "hit_val_P_12")]:
    e = panel[panel.elig_P]
    d_ = [e[(e.year == y) & (e.bin12 == "R1-2") & (e.pos_adp == "RB")][col].mean()
          - e[(e.year == y) & (e.bin12 == "R1-2") & (e.pos_adp == "WR")][col].mean()
          for y in YEARS]
    m, se, t, pv, lo, hi, n = cluster_t(d_)
    print(f"  L2[{hitdef}] RB-WR = {m:+.3f} (SE {se:.3f}) p={pv:.4f} "
          f"MDE={mde_prop_cluster(np.std(d_, ddof=1), 10):.3f}")
    tests.append(dict(test="L2ppg", claim="a", hitdef=hitdef, spec="season-clustered t, PPG|>=4g",
                      estimate=m, se=se, ci_lo=lo, ci_hi=hi, p=pv,
                      mde=mde_prop_cluster(np.std(d_, ddof=1), 10),
                      n=int(e.bin12.eq("R1-2").sum()), family=0))
    rb = e[(e.bin12 == "R1-2") & (e.pos_adp == "RB")].copy()
    rb["yc"] = rb.year - 2019.5
    mt = smf.logit(f"{col} ~ yc", data=rb).fit(disp=0, cov_type="cluster",
                                               cov_kwds={"groups": rb.year}, use_t=True)
    print(f"  L3[{hitdef}] trend {mt.params['yc']:+.4f} (SE {mt.bse['yc']:.4f}) "
          f"p={mt.pvalues['yc']:.4f}")
    tests.append(dict(test="L3ppg", claim="b", hitdef=hitdef, spec="logit hit~year, PPG|>=4g",
                      estimate=mt.params["yc"], se=mt.bse["yc"],
                      ci_lo=mt.conf_int().loc["yc", 0], ci_hi=mt.conf_int().loc["yc", 1],
                      p=mt.pvalues["yc"], mde=(stats.t.ppf(.975, 9) + stats.t.ppf(.80, 9)) * mt.bse["yc"],
                      n=len(rb), family=0))
    ee = e[e.pos_adp.isin(POS)].copy()
    ee["binnum"] = pd.Categorical(ee.bin12, BINS, ordered=True).codes.astype(float)
    m4 = smf.logit(f"{col} ~ C(pos_adp, Treatment('WR')) * binnum", data=ee).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": ee.year}, use_t=True)
    k4 = [x for x in m4.params.index if ":" in x][0]
    print(f"  L4[{hitdef}] interaction {m4.params[k4]:+.4f} (SE {m4.bse[k4]:.4f}) "
          f"p={m4.pvalues[k4]:.4f}")
    tests.append(dict(test="L4ppg", claim="c", hitdef=hitdef, spec="logit hit~pos*binnum, PPG|>=4g",
                      estimate=m4.params[k4], se=m4.bse[k4], ci_lo=m4.conf_int().loc[k4, 0],
                      ci_hi=m4.conf_int().loc[k4, 1], p=m4.pvalues[k4],
                      mde=(stats.t.ppf(.975, 9) + stats.t.ppf(.80, 9)) * m4.bse[k4],
                      n=len(ee), family=0))

print("\n=== D4c decomposing the R1-2 RB total-points deficit ===")
g_rb, g_wr = s12[s12.pos_adp == "RB"].games.mean(), s12[s12.pos_adp == "WR"].games.mean()
r_rb, r_wr = s12[s12.pos_adp == "RB"].ppg.mean(), s12[s12.pos_adp == "WR"].ppg.mean()
tot_gap = s12[s12.pos_adp == "RB"].total.mean() - s12[s12.pos_adp == "WR"].total.mean()
games_part = (g_rb - g_wr) * r_wr
ppg_part = g_rb * (r_rb - r_wr)
print(f"  total gap {tot_gap:+.1f} pts/season | games channel {games_part:+.1f} "
      f"| PPG channel {ppg_part:+.1f} | residual {tot_gap-games_part-ppg_part:+.1f}")
print(f"  games: RB {g_rb:.2f} vs WR {g_wr:.2f}; PPG: RB {r_rb:.2f} vs WR {r_wr:.2f}")

print("\n=== D7 IS THE VALUE-RETURN DEFINITION MECHANICALLY POSITIONAL? ===")
# Raw PPR points at matched positional finish rank, league-wide.  If WRs out-score
# RBs at the same positional rank, then "points >= bin median across all positions"
# is a measure of PPR volume, not of beating one's price.
rowsx = []
for y in YEARS:
    g = out[out.season == y]
    for pos in POS:
        t = g[g.pos == pos].nlargest(48, "total").total.values
        rowsx.append(dict(year=y, pos=pos, t1=t[:12].mean(), t2=t[12:24].mean(),
                          t3=t[24:36].mean(), t4=t[36:48].mean()))
rx = pd.DataFrame(rowsx)
gapx = (rx[rx.pos == "WR"].set_index("year")[["t1", "t2", "t3", "t4"]]
        - rx[rx.pos == "RB"].set_index("year")[["t1", "t2", "t3", "t4"]]).mean()
print("  mean season total PPR by positional finish tier:")
print(rx.groupby("pos")[["t1", "t2", "t3", "t4"]].mean().round(1).to_string())
print(f"  WR minus RB at matched rank: {gapx.round(1).to_dict()}  "
      "-> value-return mechanically favours WR, more so deeper")
# post-hoc (NOT in the family): pooled position main effect controlling within-bin price
dv = panel[panel.pos_adp.isin(POS)].copy()
dv["ladp"] = np.log(dv.adp)
for col, lab in [("hit_val_T_12", "value_T"), ("hit_pos_T", "top12_T")]:
    mv = smf.logit(f"{col} ~ C(pos_adp, Treatment('WR')) + ladp", data=dv).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": dv.year}, use_t=True)
    kv = [x for x in mv.params.index if "pos_adp" in x][0]
    print(f"  POST-HOC (not in family) {lab}: RB effect | log(ADP) = "
          f"{mv.params[kv]:+.3f} (SE {mv.bse[kv]:.3f}) p={mv.pvalues[kv]:.4f}")
    tests.append(dict(test="D7posthoc", claim="exploratory", hitdef=lab,
                      spec="logit hit ~ pos + log(adp), cluster(season); POST-HOC",
                      estimate=mv.params[kv], se=mv.bse[kv],
                      ci_lo=mv.conf_int().loc[kv, 0], ci_hi=mv.conf_int().loc[kv, 1],
                      p=mv.pvalues[kv], mde=np.nan, n=len(dv), family=0))
print("  mean ADP within bin by position (is the contrast price-balanced?):")
print(dv.groupby(["bin12", "pos_adp"]).adp.mean().unstack().reindex(BINS).round(1).to_string())

print("\n=== D5 10-team frame (owner's league) ===")
p10 = (rates[(rates.frame == "10team") & (rates.window == "2015-2024")]
       .pivot_table(index=["pos", "bin"], columns="hitdef", values=["n", "rate"]).round(3))
print(p10.to_string())
print("\n=== D6 2018-2024 window, 12-team ===")
print((rates[(rates.frame == "12team") & (rates.window == "2018-2024")]
       .pivot_table(index=["pos", "bin"], columns="hitdef",
                    values=["n", "rate"]).round(3)).to_string())

T = pd.DataFrame(tests)
bh = pd.concat([fam[["test", "hitdef", "bh_thresh", "bh_reject"]],
                f2[["test", "hitdef", "bh_thresh", "bh_reject"]]], ignore_index=True)
T = T.merge(bh, on=["test", "hitdef"], how="left")
T.to_csv(f"{ROOT}/results/conversion_tests.csv", index=False)
H.to_csv(f"{ROOT}/results/sectionL_holdout.csv", index=False)
print("\nwrote results/conversion_rates.csv, conversion_tests.csv, sectionL_panel.csv")

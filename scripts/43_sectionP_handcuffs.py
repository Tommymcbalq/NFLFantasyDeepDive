"""§P3 — Handcuff transfer rates (2015-2024 panel) and the 2026 handcuff table.

DESCRIPTIVE. No new model, no hypothesis test, nothing enters an FDR family.
Every rule below is fixed before any transfer rate is computed.

DEFINITIONS
  Team-week universe: REG weeks in which the team appears in weekly_raw (byes excluded
  automatically).  A player is PRESENT in a team-week iff he has a row with
  touches = carries + targets >= 2 (the §G1 participation rule); otherwise ABSENT.

  Lead back L(team, season): the RB with the most carries in that team-season, subject to
  a role qualification fixed from aggregate distributions -- >= 8 present weeks AND
  >= 10 carries per present game.  (Without a qualification, "lead back" on a full
  committee is not a meaningful object and the transfer denominator is tiny.)

  Eligible team-seasons: >= 2 absent weeks AND >= 4 present weeks for L, so both
  conditional means are estimable.

  Two backup definitions, both reported, because the gap between them IS the handcuff risk:
    EX-ANTE  backup = the non-lead RB with the most carries in L's PRESENT weeks.
             This is the back a drafter could actually identify in advance.
    EX-POST  inheritor = the non-lead RB with the most carries in L's ABSENT weeks.
             Only knowable after the fact; the upper bound on handcuff value.

  Transfer rate, per resource r in {carries, targets, inside-10 carries}:
        T_r = ( x_bk^out - x_bk^in ) / x_L^in
  all quantities per game.  Denominator is the lead back's own per-game usage while
  present -- the workload actually up for grabs.  Complementary shares:
        T_rest = ( sum_{j != L, bk} x_j^out - x_j^in ) / x_L^in
        T_leak = ( sum_{all RB} x^out - sum_{all RB} x^in ) / x_L^in     (volume change)
  and the accounting identity  T_bk + T_rest = T_leak + 1  holds exactly.

  Absence definition: PRIMARY = any absence (what a fantasy owner faces, whatever the
  cause).  SENSITIVITY = absences confirmed by the weekly injury report
  (report_status in {Out, Doubtful} or practice status "Did Not Participate" with a
  named injury), reported separately.

  Inside-10 carries come from play-by-play, available 2018+ only, so that resource is
  estimated on the 2018-2024 sub-panel and labelled.

CONDITIONAL VALUE for 2026.  Fitted on the same panel:
        ppg_bk^out = a + b * ppg_bk^in + c * ppg_L^in + e,   OLS, cluster(season)
  i.e. what the ex-ante backup actually scored per game while the starter was out, given
  his own standing usage and the size of the role above him.  Reported with the residual
  SD and the empirical quantiles of ppg_bk^out, because the spread -- not the mean -- is
  what decides whether a handcuff pick is worth a roster slot.

2026 handcuffs: for each RB on the top-30 2026 ADP board, the same-team RB with the next
best projected role, ranked by (Sleeper depth-chart order, 2025 carries).  Standalone value
= the §P2 deep market curve at his own ADP (or the curve's floor level if unranked).

Outputs: results/handcuff_table_2026.csv, results/sectionP_transfer_rates.csv
"""
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.simplefilter("ignore")
ROOT = Path("/Users/thomasmcnamee/NFL")
YEARS = list(range(2015, 2025))
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm(s):
    s = re.sub(r"[^a-z ]", "", str(s).lower().replace(".", " ")
               .replace("-", " ").replace("'", ""))
    return " ".join(t for t in s.split() if t not in SUFFIXES)


# ------------------------------------------------------------------- weekly RB
COLS = ["player_id", "player_display_name", "position", "team", "season", "week",
        "season_type", "targets", "carries", "fantasy_points_ppr"]
frames = []
for y in range(2015, 2026):
    d = pd.read_csv(ROOT / f"data/players/weekly_raw/stats_player_week_{y}.csv",
                    usecols=lambda c: c in COLS, low_memory=False)
    frames.append(d[d.season_type == "REG"])
wk = pd.concat(frames, ignore_index=True)
wk["touches"] = wk.carries.fillna(0) + wk.targets.fillna(0)
teamweeks = wk[["team", "season", "week"]].drop_duplicates()
rb = wk[(wk.position == "RB") & (wk.season.isin(YEARS))].copy()
print(f"RB player-weeks 2015-2024: {len(rb)}; team-weeks: "
      f"{len(teamweeks[teamweeks.season.isin(YEARS)])}")

# ------------------------------------------------------------- inside-10 carries
gl = []
for y in range(2018, 2025):
    p = pd.read_parquet(ROOT / f"data/advanced/pbp/play_by_play_{y}.parquet",
                        columns=["season", "week", "season_type", "posteam",
                                 "rusher_player_id", "rush_attempt", "yardline_100"])
    p = p[(p.season_type == "REG") & (p.rush_attempt == 1) & (p.yardline_100 <= 10)]
    gl.append(p.groupby(["season", "week", "rusher_player_id"]).size()
              .rename("gl10").reset_index())
gl = pd.concat(gl).rename(columns={"rusher_player_id": "player_id"})
rb = rb.merge(gl, on=["player_id", "season", "week"], how="left")
rb["gl10"] = rb.gl10.fillna(0.0)
rb.loc[rb.season < 2018, "gl10"] = np.nan
print(f"inside-10 carries joined for {int(rb.gl10.notna().sum())} RB weeks (2018+)")

# ------------------------------------------------------------------ injury flags
inj = []
for y in YEARS:
    d = pd.read_csv(ROOT / f"data/injuries/injuries_{y}.csv", low_memory=False)
    inj.append(d[d.game_type == "REG"][["season", "week", "gsis_id", "report_status",
                                        "practice_status", "report_primary_injury"]])
inj = pd.concat(inj)
inj["hurt"] = (inj.report_status.isin(["Out", "Doubtful"])
               | (inj.practice_status.eq("Did Not Participate In Practice")
                  & inj.report_primary_injury.notna()))
hurt = set(map(tuple, inj[inj.hurt][["gsis_id", "season", "week"]].values))
print(f"injury-report Out/Doubtful/DNP-with-injury player-weeks: {len(hurt)}")

# =========================================================== build the transfer panel
present = rb[rb.touches >= 2]
ts_car = present.groupby(["team", "season", "player_id"]).carries.sum()
recs, ppgrows = [], []
for (tm, ss), tw in teamweeks[teamweeks.season.isin(YEARS)].groupby(["team", "season"]):
    weeks = set(tw.week)
    room = present[(present.team == tm) & (present.season == ss)]
    if room.empty:
        continue
    tot = room.groupby("player_id").carries.sum().sort_values(ascending=False)
    L = tot.index[0]
    lg = room[room.player_id == L]
    if len(lg) < 8 or lg.carries.mean() < 10:
        continue
    wpres = set(lg.week)
    wabs = weeks - wpres
    if len(wabs) < 2 or len(wpres) < 4:
        continue
    others = [p for p in tot.index if p != L]
    if not others:
        continue
    sub = room[room.player_id.isin(others)]
    cin = sub[sub.week.isin(wpres)].groupby("player_id").carries.sum() / len(wpres)
    cout = sub[sub.week.isin(wabs)].groupby("player_id").carries.sum() / len(wabs)
    cin = cin.reindex(others).fillna(0.0)
    cout = cout.reindex(others).fillna(0.0)
    bk_ante = cin.idxmax()
    bk_post = cout.idxmax()
    rec = dict(team=tm, season=ss, lead=L,
               lead_name=lg.player_display_name.iat[0],
               n_pres=len(wpres), n_abs=len(wabs),
               inj_frac=np.mean([(L, ss, w) in hurt for w in wabs]),
               bk_ante=bk_ante, bk_post=bk_post,
               same_backup=int(bk_ante == bk_post),
               bk_ante_name=room[room.player_id == bk_ante].player_display_name.iat[0],
               bk_post_name=room[room.player_id == bk_post].player_display_name.iat[0])
    for res, col in [("car", "carries"), ("tgt", "targets"), ("gl10", "gl10")]:
        if col == "gl10" and ss < 2018:
            for k in ["den", "ante", "post", "rest", "leak"]:
                rec[f"{res}_{k}"] = np.nan
            continue
        den = lg[col].mean()
        ain = sub[sub.week.isin(wpres)].groupby("player_id")[col].sum() / len(wpres)
        aout = sub[sub.week.isin(wabs)].groupby("player_id")[col].sum() / len(wabs)
        ain = ain.reindex(others).fillna(0.0)
        aout = aout.reindex(others).fillna(0.0)
        allin = den + ain.sum()
        allout = aout.sum()
        rec[f"{res}_den"] = den
        rec[f"{res}_ante"] = (aout[bk_ante] - ain[bk_ante]) / den if den > 0 else np.nan
        rec[f"{res}_post"] = (aout[bk_post] - ain[bk_post]) / den if den > 0 else np.nan
        rest = [p for p in others if p != bk_ante]
        rec[f"{res}_rest"] = ((aout[rest].sum() - ain[rest].sum()) / den
                              if den > 0 else np.nan)
        rec[f"{res}_leak"] = (allout - allin) / den if den > 0 else np.nan
    # PPG of the ex-ante backup in / out, and the lead back's own PPG in
    bg = sub[sub.player_id == bk_ante]
    rec["ppg_bk_in"] = bg[bg.week.isin(wpres)].fantasy_points_ppr.mean()
    rec["ppg_bk_out"] = bg[bg.week.isin(wabs)].fantasy_points_ppr.mean()
    rec["g_bk_out"] = int(bg.week.isin(wabs).sum())
    rec["ppg_lead_in"] = lg.fantasy_points_ppr.mean()
    recs.append(rec)

T = pd.DataFrame(recs)
T.to_csv(ROOT / "results/sectionP_transfer_rates.csv", index=False)

# ---- base rate: how often does a qualified lead back miss games at all?
qual = []
for (tm, ss), tw in teamweeks[teamweeks.season.isin(YEARS)].groupby(["team", "season"]):
    room = present[(present.team == tm) & (present.season == ss)]
    if room.empty:
        continue
    tot = room.groupby("player_id").carries.sum().sort_values(ascending=False)
    lg = room[room.player_id == tot.index[0]]
    if len(lg) < 8 or lg.carries.mean() < 10:
        continue
    qual.append(dict(team=tm, season=ss, n_abs=len(set(tw.week) - set(lg.week))))
Q = pd.DataFrame(qual)
print(f"\nOUTAGE BASE RATE: {len(Q)} qualified lead-back team-seasons 2015-2024; "
      f"P(>=1 missed week) = {(Q.n_abs >= 1).mean():.3f}, "
      f"P(>=2) = {(Q.n_abs >= 2).mean():.3f}, P(>=4) = {(Q.n_abs >= 4).mean():.3f}, "
      f"mean missed weeks {Q.n_abs.mean():.2f}")
print("  (conditioning note: a back who misses so much that he loses the carry lead "
      "is not counted as a qualified lead back, so this understates outage risk)")
print(f"\neligible team-seasons: {len(T)}  ({T.season.nunique()} seasons, "
      f"{T.team.nunique()} teams); mean absent weeks {T.n_abs.mean():.2f}")
print(f"absences confirmed by injury report: mean fraction {T.inj_frac.mean():.3f}")
print(f"ex-ante backup == ex-post inheritor in {T.same_backup.mean()*100:.1f}% "
      f"of team-seasons ({T.same_backup.sum()}/{len(T)})")

print("\n" + "=" * 76 + "\nTRANSFER RATE DISTRIBUTIONS (share of the lead back's "
      "per-game workload)\n" + "=" * 76)
qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
rows = []
for res, lab, sub in [("car", "carries", T), ("tgt", "targets", T),
                      ("gl10", "inside-10 carries (2018+)", T[T.season >= 2018])]:
    for who, nm in [("ante", "EX-ANTE backup"), ("post", "EX-POST inheritor"),
                    ("rest", "rest of the room"), ("leak", "team RB volume change")]:
        v = sub[f"{res}_{who}"].dropna()
        r = dict(resource=lab, who=nm, n=len(v), mean=v.mean(), sd=v.std(),
                 **{f"q{int(q*100)}": v.quantile(q) for q in qs})
        if who in ("ante", "post"):
            r["P(T>0.5)"] = (v > 0.5).mean()
            r["P(T>0.75)"] = (v > 0.75).mean()
            r["P(T<0.25)"] = (v < 0.25).mean()
        rows.append(r)
D = pd.DataFrame(rows)
print(D.round(3).to_string(index=False))
D.to_csv(ROOT / "results/sectionP_transfer_distribution.csv", index=False)

print("\naccounting identity check  T_ante + T_rest - T_leak (should be 1.000): "
      f"{(T.car_ante + T.car_rest - T.car_leak).mean():.4f} "
      f"(max dev {np.abs(T.car_ante + T.car_rest - T.car_leak - 1).max():.2e})")

print("\nsensitivity: team-seasons where >= 50% of absences are injury-confirmed")
inj_sub = T[T.inj_frac >= 0.5]
print(f"  n = {len(inj_sub)}; carries EX-ANTE mean {inj_sub.car_ante.mean():.3f}, "
      f"median {inj_sub.car_ante.median():.3f}  "
      f"(all: {T.car_ante.mean():.3f} / {T.car_ante.median():.3f})")
print("sensitivity: >= 4 absent weeks (a real multi-week outage, not a one-off)")
lng = T[T.n_abs >= 4]
print(f"  n = {len(lng)}; carries EX-ANTE mean {lng.car_ante.mean():.3f}, "
      f"median {lng.car_ante.median():.3f}")

print("\n" + "=" * 76 + "\nWHAT THE BACKUP ACTUALLY SCORED WHILE THE STARTER WAS OUT\n"
      + "=" * 76)
V = T.dropna(subset=["ppg_bk_out", "ppg_bk_in", "ppg_lead_in"]).copy()
print(f"n = {len(V)} team-seasons; ppg_bk_out quantiles "
      f"{V.ppg_bk_out.quantile([.1,.25,.5,.75,.9]).round(2).to_dict()}; "
      f"mean {V.ppg_bk_out.mean():.2f}, SD {V.ppg_bk_out.std():.2f}")
print(f"  ppg_bk_in  mean {V.ppg_bk_in.mean():.2f}; lift out-in "
      f"mean {(V.ppg_bk_out - V.ppg_bk_in).mean():+.2f}, "
      f"median {(V.ppg_bk_out - V.ppg_bk_in).median():+.2f}")
print(f"  P(ppg_bk_out >= 12, i.e. startable) = {(V.ppg_bk_out >= 12).mean():.3f}; "
      f">= 15 = {(V.ppg_bk_out >= 15).mean():.3f}; "
      f"< 8 (unusable) = {(V.ppg_bk_out < 8).mean():.3f}")
X = sm.add_constant(V[["ppg_bk_in", "ppg_lead_in"]])
mod = sm.OLS(V.ppg_bk_out, X).fit(cov_type="cluster",
                                  cov_kwds={"groups": V.season})
print("\nppg_bk_out ~ ppg_bk_in + ppg_lead_in (cluster by season):")
print(pd.DataFrame({"coef": mod.params, "se": mod.bse, "t": mod.tvalues,
                    "p": mod.pvalues}).round(4).to_string())
sig = float(np.sqrt(mod.mse_resid))
print(f"R2 = {mod.rsquared:.3f}, residual SD = {sig:.2f} PPG, n = {int(mod.nobs)}")
res = mod.resid
print(f"residual skew {res.skew():.2f}, kurtosis {res.kurtosis():.2f}; "
      f"log1p sensitivity slope on ppg_bk_in: "
      f"{sm.OLS(np.log1p(V.ppg_bk_out), X).fit().params['ppg_bk_in']:.4f}")

# =========================================================== 2026 handcuff table
print("\n" + "=" * 76 + "\n2026 HANDCUFF TABLE\n" + "=" * 76)
sl = json.load(open(ROOT / "data/sleeper/players_nfl_2026.json"))
sr = pd.DataFrame([{"name": v.get("full_name"), "team": v.get("team"),
                    "pos": v.get("position"), "dco": v.get("depth_chart_order"),
                    "active": v.get("active"), "yexp": v.get("years_exp")}
                   for v in sl.values()])
sr = sr[(sr.pos == "RB") & sr.team.notna() & (sr.active == True)].copy()
sr["team"] = sr.team.replace({"LA": "LAR"})
sr["nname"] = sr.name.map(norm)

adp26 = pd.read_csv(ROOT / "data/adp/adp_ppr_2026_all_20260809.csv")
adp_rb = adp26[adp26.position == "RB"].sort_values("adp").copy()
adp_rb["nname"] = adp_rb.name.map(norm)
adp_map = dict(zip(adp_rb.nname, adp_rb.adp))
board30 = adp_rb.head(30)

c25 = (wk[(wk.season == 2025) & (wk.position == "RB")]
       .groupby("player_display_name")
       .agg(car25=("carries", "sum"), tgt25=("targets", "sum"),
            g25=("touches", lambda s: (s >= 2).sum()),
            ppg25=("fantasy_points_ppr", "mean")).reset_index())
c25["nname"] = c25.player_display_name.map(norm)
cmap = c25.set_index("nname")

knots = pd.read_csv(ROOT / "results/market_prior_iso_knots_rb_deep.csv")
floor_m = knots.m.min()


def curve(a):
    return float(np.interp(np.log(a), knots.log_adp, knots.m)) if a == a else floor_m


rows = []
for _, s in board30.iterrows():
    mates = sr[(sr.team == s.team) & (sr.nname != s.nname)].copy()
    if mates.empty:
        continue
    mates["car25"] = mates.nname.map(cmap.car25).fillna(0.0)
    mates["adp"] = mates.nname.map(adp_map)
    mates["dco"] = mates.dco.fillna(99)
    mates = mates.sort_values(["dco", "car25"], ascending=[True, False])
    h = mates.iloc[0]
    hin = cmap.reindex([h.nname]).iloc[0]
    sin = cmap.reindex([s.nname]).iloc[0]
    ppg_bk_in = hin.ppg25 if hin.ppg25 == hin.ppg25 else 0.0
    ppg_lead_in = sin.ppg25 if sin.ppg25 == sin.ppg25 else curve(s.adp)
    cond = float(mod.params["const"] + mod.params["ppg_bk_in"] * ppg_bk_in
                 + mod.params["ppg_lead_in"] * ppg_lead_in)
    # empirical residual distribution (skew 0.56 -> do not assume normal)
    draws = cond + res.values
    # out-of-support flag: every ex-ante backup in the fit panel was his team's top
    # non-lead carrier, so ppg_bk_in below the panel's 5th pct is an extrapolation
    oos = ppg_bk_in < V.ppg_bk_in.quantile(0.05)
    rows.append(dict(
        starter=s["name"], team=s.team, starter_adp=s.adp,
        starter_board_value=curve(s.adp), starter_ppg_2025=sin.ppg25,
        handcuff=h["name"], handcuff_adp=h.adp,
        handcuff_depth_order=None if h.dco == 99 else int(h.dco),
        handcuff_car_2025=h.car25, handcuff_ppg_2025=hin.ppg25,
        handcuff_g_2025=hin.g25,
        standalone_value=curve(h.adp) if h.adp == h.adp else np.nan,
        conditional_ppg_if_starter_out=cond,
        conditional_sd=sig,
        p_startable_12=float((draws >= 12).mean()),
        p_rb1_15=float((draws >= 15).mean()),
        cond_p10=float(np.percentile(draws, 10)),
        cond_p90=float(np.percentile(draws, 90)),
        cond_minus_standalone=cond - (curve(h.adp) if h.adp == h.adp else floor_m),
        out_of_support=int(oos),
        committee_flag=int(h.adp == h.adp and h.adp <= board30.adp.max()),
        rookie_flag="rookie (owner excludes)" if h.yexp == 0 else ""))
H = pd.DataFrame(rows)
# expected transfer applied to the starter's own 2025 carry load, for reference
H["starter_carries_2025"] = H.starter.map(lambda n: cmap.reindex([norm(n)]).car25.iat[0])
H["med_carry_transfer"] = T.car_ante.median()
H = H.sort_values("conditional_ppg_if_starter_out", ascending=False).reset_index(drop=True)
H.to_csv(ROOT / "results/handcuff_table_2026.csv", index=False)
print(H.round(3).to_string(index=False))
print("\nwrote handcuff_table_2026.csv, sectionP_transfer_rates.csv, "
      "sectionP_transfer_distribution.csv")

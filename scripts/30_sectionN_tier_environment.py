"""§N (EDA_PLAN6.md) — RB/WR tier finishes and offensive environment.

PRE-REGISTRATION (EDA_PLAN6.md §N, 2026-08-09, restated here as executed)
------------------------------------------------------------------------
Question: how often does a top-12 (and 13-24) RB play in a top-10 offence, and does
playing in a PROJECTED top-10 offence raise P(top-12 finish)?

Design decisions fixed BEFORE any result was inspected:

  Panel      : the validated §L board->player join (results/sectionL_panel.csv, built by
               scripts/26_sectionL_conversion.py; carries the three §L bug fixes: DE name
               collision, TE position drift, slot-budget check).  Board team is taken from
               the same FFC boards, joined on (year, name, position) -- exact, since panel
               rows ARE board rows.
  Environment:
    (ii) PROJECTED (actionable, preseason-knowable) = de-vigged preseason closing win total
         (wt17) computed with the §I3 estimator (scripts/24_sectionI3_vegas_edge.py
         devig_rate), ranked 1-32 within season.  top10_proj = rank <= 10.  Team = BOARD
         team (what the drafter knows in August).
    (i)  REALIZED (descriptive) = team points scored per game, REG only, ranked 1-32 within
         season.  top10_real = rank <= 10.  Team = player's realized primary team (most REG
         appearances) for the descriptive shares; board team for the drafted-panel tests, so
         that PROJ and REAL arms differ only in the environment measure, not in the join.
  Outcome    : league-wide positional finish rank on season-total PPR (the §L primary,
               `rank_T`, computed on effective position).  hit12 = rank<=12, hit24 = rank<=24
               (cumulative).  Band 13-24 reported separately because the owner asked for it.
  Cost bins  : §L's 12-team-frame bins R1-2 / R3-4 / R5-6 / R7-8 / R9+ (bin12).

  PRIMARY QUANTITY (the pre-specified one): P(hit | top10 offence, COST BIN) vs P(hit | not,
  cost bin).  Estimator: logit  hit ~ top10 + C(bin12), fit separately per position, SEs
  clustered by season, t with 9 df.  Effect size reported as the average marginal effect
  (risk difference, pp) and as a Cochran-Mantel-Haenszel bin-stratified risk difference.
  The UNCONDITIONAL version (no bin FE) is reported too and is LABELLED CONFOUNDED: good
  offences attract better and more expensive backs, so the raw share is not interpretable.

  FAMILY (new, declared before fitting): 8 tests =
      {RB, WR} x {hit12, hit24} x {projected, realized}, each the cost-bin-conditioned
      logit coefficient on top10.  BH q = 0.10.  Everything else (unconditional versions,
      continuous-wt17 versions, log(ADP) control, year FE, PPG outcome, per-cell rates,
      descriptive shares) is family = 0 and is not corrected.
      Closed families {H5,I3}, {K}, {L}, {L-EXT}, {M} are NOT reopened.
  MDE        : computed FOR THIS DESIGN from the realized cluster SE
               (t_.975,9 + t_.80,9) * SE, per §28's rule that MDEs are not transplantable.
               Reported on the log-odds scale and converted to pp at the cell base rate.
  HOLDOUT    : fit 2015-2021, evaluate 2022-2024; sign stability + direction.

Outputs: results/rb_tier_environment.csv, results/sectionN_tests.csv,
         results/sectionN_shares.csv, results/sectionN_notes.md
Rerun:   python3 scripts/30_sectionN_tier_environment.py
"""
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import norm
from scipy.optimize import brentq

warnings.simplefilter("ignore")
ROOT = "/Users/thomasmcnamee/NFL"
YEARS = list(range(2015, 2025))
GAMES = {y: (16 if y <= 2020 else 17) for y in range(2003, 2027)}
BINS = ["R1-2", "R3-4", "R5-6", "R7-8", "R9+"]
POS = ["RB", "WR"]
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


# ============================ 1. environment ================================
def amer_to_prob(o):
    o = np.asarray(o, dtype=float)
    return np.where(o > 0, 100.0 / (o + 100.0), -o / (-o + 100.0))


def devig_rate(L_rate, p_fair, s, integer_line, G):
    """§I3 estimator: win RATE implied by a de-vigged two-way price."""
    p_fair = min(max(p_fair, 1e-6), 1 - 1e-6)
    if not integer_line:
        return L_rate + s * norm.ppf(p_fair)
    hi, lo = L_rate + 0.5 / G, L_rate - 0.5 / G

    def f(mu):
        po = norm.sf(hi, loc=mu, scale=s)
        pu = norm.cdf(lo, loc=mu, scale=s)
        return po / (po + pu) - p_fair
    return brentq(f, L_rate - 6 * s, L_rate + 6 * s, xtol=1e-10)


NAME2FR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
    "Oakland Raiders": "LV", "Las Vegas Raiders": "LV",
    "San Diego Chargers": "LAC", "Los Angeles Chargers": "LAC",
    "St Louis Rams": "LA", "Los Angeles Rams": "LA",
    "Washington Redskins": "WAS", "Washington Football Team": "WAS",
    "Washington Commanders": "WAS",
}
# any era/board abbrev -> franchise code
ABBR2FR = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "LVR": "LV", "WSH": "WAS"}


def to_fr(s):
    return pd.Series(s).replace(ABBR2FR)


cov = pd.read_csv(f"{ROOT}/data/vegas/team_win_totals_2015_2025_covers.csv")
cov["fr"] = cov.team.map(NAME2FR)
assert cov.fr.notna().all()
assert cov.groupby("season").fr.nunique().eq(32).all()

h = cov[cov.season.between(2015, 2024)].copy()
h["G"] = h.season.map(GAMES)
S_RATE = float((h.actual_wins / h.G - h.win_total / h.G).std(ddof=1))

V = cov[cov.season.isin(YEARS)].copy()
V["G"] = V.season.map(GAMES)
V["L_rate"] = V.win_total / V.G
po, pu = amer_to_prob(V.over.values), amer_to_prob(V.under.values)
V["p_fair_over"] = po / (po + pu)
V["mu_rate"] = [devig_rate(lr, pf, S_RATE, (wt % 1 == 0), g)
                for lr, pf, wt, g in zip(V.L_rate, V.p_fair_over, V.win_total, V.G)]
V["wt17"] = V.mu_rate * 17
V["proj_rank"] = V.groupby("season").wt17.rank(ascending=False, method="first")
V["top10_proj"] = (V.proj_rank <= 10).astype(int)
ENVP = V[["season", "fr", "wt17", "proj_rank", "top10_proj"]].rename(columns={"season": "year"})

# realized offence: team points scored per game, REG only
g = pd.read_csv(f"{ROOT}/data/teams/games_nflverse_20260809.csv", low_memory=False)
g = g[(g.game_type == "REG") & (g.season.isin(YEARS))]
sc = pd.concat([
    g[["season", "home_team", "home_score"]].rename(
        columns={"home_team": "tm", "home_score": "pts"}),
    g[["season", "away_team", "away_score"]].rename(
        columns={"away_team": "tm", "away_score": "pts"})])
sc["fr"] = to_fr(sc.tm)
R = (sc.groupby(["season", "fr"]).pts.agg(["sum", "size"]).reset_index()
       .rename(columns={"sum": "pts_tot", "size": "gms"}))
R["ppg_team"] = R.pts_tot / R.gms
R["real_rank"] = R.groupby("season").ppg_team.rank(ascending=False, method="first")
R["top10_real"] = (R.real_rank <= 10).astype(int)
assert R.groupby("season").fr.nunique().eq(32).all(), "32 franchises per season required"
ENVR = R[["season", "fr", "ppg_team", "real_rank", "top10_real"]].rename(
    columns={"season": "year"})

say("# §N — RB/WR tier finishes and offensive environment")
say("")
say("Run", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    "| script `scripts/30_sectionN_tier_environment.py` (pre-registration in its docstring)")
say("")
say("## 0. Build")
say("")
say(f"- de-vig scale s = {S_RATE:.5f} win-rate = {S_RATE*17:.3f} wins/17 "
    f"(recomputed here, matches §I3).")
say(f"- projected-offence proxy: de-vigged win total wt17, range "
    f"{ENVP.wt17.min():.2f}-{ENVP.wt17.max():.2f}; ranked 1-32 within season.")
say(f"- realized offence: team points/game, range {ENVR.ppg_team.min():.1f}-"
    f"{ENVR.ppg_team.max():.1f}; ranked 1-32 within season.")
rho = (ENVP.merge(ENVR, on=["year", "fr"])
       .groupby("year").apply(lambda d: d.wt17.corr(d.ppg_team)))
say(f"- corr(projected wt17, realized points/game) within season: mean "
    f"{rho.mean():.3f} (range {rho.min():.3f}-{rho.max():.3f}). Agreement of the two "
    f"top-10 sets: {(ENVP.merge(ENVR,on=['year','fr']).eval('top10_proj==top10_real')).mean():.3f} "
    "of team-seasons.")

# ============================ 2. panel ======================================
panel = pd.read_csv(f"{ROOT}/results/sectionL_panel.csv", low_memory=False)
brd = []
for y in YEARS:
    a = pd.read_csv(f"{ROOT}/data/adp/historical/adp_ppr_{y}.csv")
    a = a[a.position.isin(["QB", "RB", "WR", "TE"])]
    brd.append(a.assign(year=y)[["year", "name", "position", "adp", "team"]]
               .rename(columns={"position": "pos_adp", "team": "board_team"}))
brd = pd.concat(brd, ignore_index=True)
n0 = len(panel)
panel = panel.merge(brd, on=["year", "name", "pos_adp", "adp"], how="left")
assert len(panel) == n0, "board join changed row count"
panel["fr"] = to_fr(panel.board_team)
say(f"- §L panel reused unchanged: {n0} board rows; board-team join "
    f"{panel.fr.notna().sum()}/{n0} matched, {panel.fr.isna().sum()} missing "
    f"({panel[panel.fr.isna()][['year','name','pos_adp']].to_dict('records')}).")

d = panel[panel.pos_adp.isin(POS) & panel.fr.notna()].copy()
d = d.merge(ENVP, on=["year", "fr"], how="left").merge(ENVR, on=["year", "fr"], how="left")
assert d.top10_proj.notna().all() and d.top10_real.notna().all()
d["hit12"] = d.hit_pos_T.astype(int)
d["hit24"] = d.hit24_T.astype(int)
d["band1324"] = ((d.rank_T > 12) & (d.rank_T <= 24)).fillna(False).astype(int)
d["bin12"] = pd.Categorical(d.bin12, BINS, ordered=True)
d["ladp"] = np.log(d.adp)
say(f"- modelling universe: {len(d)} drafted RB/WR board rows, 2015-2024 "
    f"(RB {int((d.pos_adp=='RB').sum())}, WR {int((d.pos_adp=='WR').sum())}).")
say("")

# ============================ 3. the owner's unconditional question =========
# Among ALL players who FINISH in a tier (drafted or not), what share were on a
# top-10 offence?  This is P(top-10 offence | tier finish) -- the reverse conditional,
# and the number the owner asked for.  CONFOUNDED (see §N2).
frames = []
for y in range(2014, 2026):
    w = pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
                    low_memory=False,
                    usecols=["player_id", "player_display_name", "position", "season",
                             "week", "season_type", "team", "fantasy_points_ppr"])
    frames.append(w[w.season_type == "REG"])
wk = pd.concat(frames, ignore_index=True).dropna(subset=["player_id"])
wk = wk[wk.season.isin(YEARS)]
wk["fr"] = to_fr(wk.team)
tot = (wk.groupby(["player_id", "season"])
         .agg(total=("fantasy_points_ppr", "sum"), games=("fantasy_points_ppr", "size"))
         .reset_index())
prim = (wk.groupby(["player_id", "season", "fr"]).size().rename("n").reset_index()
          .sort_values("n", ascending=False)
          .drop_duplicates(["player_id", "season"])[["player_id", "season", "fr"]])
pmode = (wk.groupby(["player_id", "season"]).position
           .agg(lambda s: s.mode().iat[0] if len(s.mode()) else np.nan)
           .rename("pos").reset_index())
# effective position, exactly as §L: a drafted player's BOARD position governs
bp = panel[["pid", "year", "pos_adp"]].dropna().drop_duplicates().rename(
    columns={"pid": "player_id", "year": "season"})
allp = tot.merge(pmode, on=["player_id", "season"]).merge(
    prim, on=["player_id", "season"]).merge(bp, on=["player_id", "season"], how="left")
allp["pos_eff"] = allp.pos_adp.fillna(allp.pos)
allp["rank_T"] = allp.groupby(["season", "pos_eff"]).total.rank(ascending=False, method="min")
allp = allp.rename(columns={"season": "year"}).merge(
    ENVP, on=["year", "fr"], how="left").merge(ENVR, on=["year", "fr"], how="left")


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p, dd = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / dd
    hh = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    return (max(0., c - hh), min(1., c + hh))


say("## §N1 The unconditional shares the owner asked for — **CONFOUNDED, read §N2**")
say("")
say("P(the player's offence was top-10 | he finished in this tier), all finishers "
    "2015-2024 (drafted or not), realized primary team. Base rate if tiers were spread "
    "evenly over teams = 10/32 = 31.2%.")
say("")
say("| position | finish tier | n | on realized top-10 offence | Wilson 95% | "
    "on *projected* top-10 offence | Wilson 95% |")
say("|---|---|---|---|---|---|---|")
shares = []
for pos in POS:
    for lab, lo_, hi_ in [("1-12", 1, 12), ("13-24", 13, 24), ("25-36", 25, 36),
                          ("37-48", 37, 48)]:
        c = allp[(allp.pos_eff == pos) & (allp.rank_T >= lo_) & (allp.rank_T <= hi_)]
        c = c.dropna(subset=["top10_real", "top10_proj"])
        n = len(c)
        kr, kp = int(c.top10_real.sum()), int(c.top10_proj.sum())
        lr, hr = wilson(kr, n)
        lp, hp = wilson(kp, n)
        say(f"| {pos} | {lab} | {n} | **{kr/n:.1%}** | [{lr:.1%}, {hr:.1%}] | "
            f"**{kp/n:.1%}** | [{lp:.1%}, {hp:.1%}] |")
        shares.append(dict(pos=pos, tier=lab, n=n, k_real=kr, rate_real=kr / n,
                           wilson_real_lo=lr, wilson_real_hi=hr, k_proj=kp,
                           rate_proj=kp / n, wilson_proj_lo=lp, wilson_proj_hi=hp))
SH = pd.DataFrame(shares)
SH.to_csv(f"{ROOT}/results/sectionN_shares.csv", index=False)
say("")
# per-season spread (per §L: single-season rates are not signal)
say("Per-season spread of the RB 1-12 realized share (recorded because §L requires it, "
    "**not** to be read as signal): " +
    ", ".join(f"{y}:{allp[(allp.pos_eff=='RB')&(allp.rank_T<=12)&(allp.year==y)].top10_real.mean():.0%}"
              for y in YEARS))
say("")

# ============================ 4. cell tables ================================
say("## §N3 Conversion by cost bin and projected environment (the primary object)")
say("")
cells = []
for envlab, envcol in [("projected", "top10_proj"), ("realized", "top10_real")]:
    for pos in POS:
        for b in BINS:
            for t10 in (1, 0):
                c = d[(d.pos_adp == pos) & (d.bin12 == b) & (d[envcol] == t10)]
                if len(c) == 0:
                    continue
                for tier, col in [("top12", "hit12"), ("top24", "hit24"),
                                  ("band13_24", "band1324")]:
                    n, k = len(c), int(c[col].sum())
                    lo_, hi_ = wilson(k, n)
                    cells.append(dict(env=envlab, pos=pos, bin=b, top10=t10, tier=tier,
                                      n=n, k=k, rate=k / n, wilson_lo=lo_, wilson_hi=hi_,
                                      mean_adp=c.adp.mean()))
CELLS = pd.DataFrame(cells)
CELLS.to_csv(f"{ROOT}/results/rb_tier_environment.csv", index=False)

for envlab in ["projected", "realized"]:
    for tier in ["top12", "top24"]:
        say(f"**{envlab} top-10 offence, finish {tier}** "
            f"(rate [Wilson] (n); gap in pp)")
        say("")
        say("| position | bin | on top-10 | off top-10 | gap (pp) |")
        say("|---|---|---|---|---|")
        for pos in POS:
            for b in BINS:
                a = CELLS[(CELLS.env == envlab) & (CELLS.pos == pos) & (CELLS["bin"] == b)
                          & (CELLS.tier == tier)]
                r1 = a[a.top10 == 1]
                r0 = a[a.top10 == 0]
                if not len(r1) or not len(r0):
                    continue
                r1, r0 = r1.iloc[0], r0.iloc[0]
                say(f"| {pos} | {b} | {r1.rate:.1%} [{r1.wilson_lo:.0%},{r1.wilson_hi:.0%}] "
                    f"({r1.n}) | {r0.rate:.1%} [{r0.wilson_lo:.0%},{r0.wilson_hi:.0%}] "
                    f"({r0.n}) | **{100*(r1.rate-r0.rate):+.1f}** |")
        say("")

# ============================ 5. the confound, quantified ===================
say("## §N2 Why the unconditional number is not interpretable — the confound, measured")
say("")
say("| position | environment | mean ADP on top-10 | mean ADP off top-10 | "
    "share of board rows on top-10 | share in R1-2 on top-10 |")
say("|---|---|---|---|---|---|")
for pos in POS:
    for envlab, envcol in [("projected", "top10_proj"), ("realized", "top10_real")]:
        c = d[d.pos_adp == pos]
        a1, a0 = c[c[envcol] == 1], c[c[envcol] == 0]
        say(f"| {pos} | {envlab} | {a1.adp.mean():.1f} | {a0.adp.mean():.1f} | "
            f"{len(a1)/len(c):.1%} | "
            f"{(c[c.bin12=='R1-2'][envcol]).mean():.1%} |")
say("")
say("A team-season supplies 31.2% of teams; if environment were orthogonal to price the "
    "shares above would all sit at 31.2% and the two mean ADPs would coincide.")
say("")

# ============================ 6. family tests ===============================
def cl_logit(data, col, rhs):
    m = smf.logit(f"{col} ~ {rhs}", data=data).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": data.year}, use_t=True)
    return m


def ame(m, data, key):
    """average marginal effect of the binary `key` = risk difference in pp."""
    d1, d0 = data.copy(), data.copy()
    d1[key], d0[key] = 1, 0
    return float(m.predict(d1).mean() - m.predict(d0).mean())


def mh_rd(data, envcol, col):
    """Cochran-Mantel-Haenszel style bin-stratified risk difference (weights n1*n0/n)."""
    num = den = 0.0
    for b in BINS:
        s = data[data.bin12 == b]
        s1, s0 = s[s[envcol] == 1], s[s[envcol] == 0]
        if len(s1) == 0 or len(s0) == 0:
            continue
        w = len(s1) * len(s0) / (len(s1) + len(s0))
        num += w * (s1[col].mean() - s0[col].mean())
        den += w
    return num / den if den else np.nan


TC, TP = stats.t.ppf(.975, 9), stats.t.ppf(.80, 9)
tests = []
say("## §N4 Primary tests — cost-conditioned, logit `hit ~ top10 + C(bin12)`, "
    "cluster(season), t(9)")
say("")
say("| # | pos | tier | environment | beta (log-odds) | cluster SE | 95% CI | raw p | "
    "AME (pp) | MH stratified RD (pp) | MDE (log-odds) | MDE (pp) |")
say("|---|---|---|---|---|---|---|---|---|---|---|---|")
i = 0
for pos in POS:
    for tier, col in [("top12", "hit12"), ("top24", "hit24")]:
        for envlab, envcol in [("projected", "top10_proj"), ("realized", "top10_real")]:
            i += 1
            s = d[d.pos_adp == pos].copy()
            m = cl_logit(s, col, f"{envcol} + C(bin12)")
            b_, se = m.params[envcol], m.bse[envcol]
            ci = m.conf_int().loc[envcol]
            p = m.pvalues[envcol]
            a = ame(m, s, envcol)
            mh = mh_rd(s, envcol, col)
            mde = (TC + TP) * se
            base = s[col].mean()
            mde_pp = mde * base * (1 - base)     # delta-method at the base rate
            say(f"| {i} | {pos} | {tier} | {envlab} | **{b_:+.3f}** | {se:.3f} | "
                f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {p:.4f} | {100*a:+.1f} | {100*mh:+.1f} | "
                f"{mde:.3f} | {100*mde_pp:.1f} |")
            tests.append(dict(test="N4", pos=pos, tier=tier, env=envlab,
                              spec="logit hit ~ top10 + C(bin12), cluster(season), t(9)",
                              estimate=b_, se=se, ci_lo=ci[0], ci_hi=ci[1], p=p,
                              ame_pp=100 * a, mh_rd_pp=100 * mh, mde=mde,
                              mde_pp=100 * mde_pp, n=len(s), family=1))
say("")

# unconditional (confounded) versions, family = 0
say("## §N5 Unconditional versions of the same contrasts — **confounded, family = 0**")
say("")
say("| pos | tier | environment | beta (no bin FE) | SE | raw p | AME (pp) | "
    "cost-conditioned AME (pp) | attributable to price |")
say("|---|---|---|---|---|---|---|---|---|")
for pos in POS:
    for tier, col in [("top12", "hit12"), ("top24", "hit24")]:
        for envlab, envcol in [("projected", "top10_proj"), ("realized", "top10_real")]:
            s = d[d.pos_adp == pos].copy()
            m0 = cl_logit(s, col, envcol)
            a0 = ame(m0, s, envcol)
            a1 = [t for t in tests if t["pos"] == pos and t["tier"] == tier
                  and t["env"] == envlab][0]["ame_pp"] / 100
            say(f"| {pos} | {tier} | {envlab} | {m0.params[envcol]:+.3f} | "
                f"{m0.bse[envcol]:.3f} | {m0.pvalues[envcol]:.4f} | {100*a0:+.1f} | "
                f"{100*a1:+.1f} | {100*(a0-a1):+.1f} pp |")
            tests.append(dict(test="N5uncond", pos=pos, tier=tier, env=envlab,
                              spec="logit hit ~ top10 (NO cost control) - CONFOUNDED",
                              estimate=m0.params[envcol], se=m0.bse[envcol],
                              ci_lo=m0.conf_int().loc[envcol, 0],
                              ci_hi=m0.conf_int().loc[envcol, 1], p=m0.pvalues[envcol],
                              ame_pp=100 * a0, mh_rd_pp=np.nan, mde=np.nan, mde_pp=np.nan,
                              n=len(s), family=0))
say("")

# sensitivities, family = 0
say("## §N6 Sensitivities (family = 0)")
say("")
say("| pos | tier | env | spec | estimate | SE | p |")
say("|---|---|---|---|---|---|---|")
SENS = [("log(ADP) control instead of bins", "{e} + ladp"),
        ("bins + season FE", "{e} + C(bin12) + C(year)"),
        ("continuous wt17 (per win) + bins", "wt17 + C(bin12)"),
        ("continuous realized team PPG + bins", "ppg_team + C(bin12)"),
        ("bins, R1-4 only", "{e} + C(bin12)")]
for pos in POS:
    for tier, col in [("top12", "hit12"), ("top24", "hit24")]:
        for envlab, envcol in [("projected", "top10_proj"), ("realized", "top10_real")]:
            for lab, rhs in SENS:
                if lab.startswith("continuous wt17") and envlab != "projected":
                    continue
                if lab.startswith("continuous realized") and envlab != "realized":
                    continue
                s = d[d.pos_adp == pos].copy()
                if "R1-4 only" in lab:
                    s = s[s.bin12.isin(["R1-2", "R3-4"])]
                    s["bin12"] = s.bin12.cat.remove_unused_categories()
                key = envcol if "{e}" in rhs else rhs.split(" ")[0]
                try:
                    m = cl_logit(s, col, rhs.format(e=envcol))
                except Exception as e:
                    say(f"| {pos} | {tier} | {envlab} | {lab} | FAILED {e} | | |")
                    continue
                say(f"| {pos} | {tier} | {envlab} | {lab} | {m.params[key]:+.3f} | "
                    f"{m.bse[key]:.3f} | {m.pvalues[key]:.4f} |")
                tests.append(dict(test="N6sens", pos=pos, tier=tier, env=envlab,
                                  spec=lab, estimate=m.params[key], se=m.bse[key],
                                  ci_lo=m.conf_int().loc[key, 0],
                                  ci_hi=m.conf_int().loc[key, 1], p=m.pvalues[key],
                                  ame_pp=np.nan, mh_rd_pp=np.nan, mde=np.nan,
                                  mde_pp=np.nan, n=len(s), family=0))
say("")

# ============================ 7. BH + holdout ===============================
T = pd.DataFrame(tests)
fam = T[T.family == 1].copy().sort_values("p").reset_index(drop=True)
fam["bh_thresh"] = 0.10 * (fam.index + 1) / len(fam)
fam["bh_reject"] = False
if (fam.p <= fam.bh_thresh).any():
    fam.loc[:fam.index[fam.p <= fam.bh_thresh].max(), "bh_reject"] = True
say(f"## §N7 BH q = 0.10 over the {len(fam)}-test §N family")
say("")
say("| rank | pos | tier | env | beta | raw p | BH threshold | reject |")
say("|---|---|---|---|---|---|---|---|")
for j, r in fam.iterrows():
    say(f"| {j+1} | {r.pos} | {r.tier} | {r.env} | {r.estimate:+.3f} | {r.p:.4f} | "
        f"{r.bh_thresh:.4f} | {'**YES**' if r.bh_reject else 'no'} |")
say("")
say(f"**{int(fam.bh_reject.sum())} of {len(fam)} survive** "
    f"(smallest raw p = {fam.p.min():.4f} against a threshold of {0.10/len(fam):.4f}).")
say("")

say("## §N8 Temporal holdout 2015-2021 -> 2022-2024")
say("")
say("| pos | tier | env | fit 2015-21 beta | holdout 2022-24 beta | sign held | "
    "holdout AME (pp) |")
say("|---|---|---|---|---|---|---|")
hold = []
for pos in POS:
    for tier, col in [("top12", "hit12"), ("top24", "hit24")]:
        for envlab, envcol in [("projected", "top10_proj"), ("realized", "top10_real")]:
            s = d[d.pos_adp == pos]
            row = {}
            for lab, yrs in [("fit", range(2015, 2022)), ("hold", range(2022, 2025))]:
                ss = s[s.year.isin(yrs)].copy()
                ss["bin12"] = ss.bin12.cat.remove_unused_categories()
                m = smf.logit(f"{col} ~ {envcol} + C(bin12)", data=ss).fit(disp=0)
                row[lab] = m.params[envcol]
                if lab == "hold":
                    row["ame"] = ame(m, ss, envcol)
            ok = np.sign(row["fit"]) == np.sign(row["hold"])
            say(f"| {pos} | {tier} | {envlab} | {row['fit']:+.3f} | {row['hold']:+.3f} | "
                f"{'yes' if ok else '**no**'} | {100*row['ame']:+.1f} |")
            hold.append(dict(pos=pos, tier=tier, env=envlab, fit=row["fit"],
                             hold=row["hold"], sign_held=bool(ok),
                             hold_ame_pp=100 * row["ame"]))
H = pd.DataFrame(hold)
say("")
say(f"Sign stability: **{int(H.sign_held.sum())} of {len(H)}**.")
say("")

T = T.merge(fam[["test", "pos", "tier", "env", "bh_thresh", "bh_reject"]],
            on=["test", "pos", "tier", "env"], how="left")
T.to_csv(f"{ROOT}/results/sectionN_tests.csv", index=False)
H.to_csv(f"{ROOT}/results/sectionN_holdout.csv", index=False)

# ============================ 8. anomaly chasing ============================
say("## §N9 Diagnostics and mechanism")
say("")
# (a) does environment predict finish through workload or through efficiency?
say("**(a) Where does any unconditional gap come from — price composition or production?**")
say("")
for pos in POS:
    s = d[d.pos_adp == pos]
    for envcol, envlab in [("top10_proj", "projected")]:
        s1, s0 = s[s[envcol] == 1], s[s[envcol] == 0]
        say(f"- {pos}, {envlab}: on top-10 mean ADP {s1.adp.mean():.1f}, "
            f"mean finish rank {s1.rank_T.mean():.1f}, PPG {s1.ppg.mean():.2f}, "
            f"games {s1.games.mean():.2f}; off top-10 ADP {s0.adp.mean():.1f}, "
            f"rank {s0.rank_T.mean():.1f}, PPG {s0.ppg.mean():.2f}, "
            f"games {s0.games.mean():.2f}.")
say("")
# (b) §I3 consistency check: the mean channel on this panel
say("**(b) Consistency with §I3 (mean channel) on this panel** — OLS of season PPG on the "
    "top-10 indicator with bin FE, cluster(season):")
say("")
for pos in POS:
    s = d[d.pos_adp == pos].copy()
    s = s[s.games >= 4]
    m = smf.ols("ppg ~ top10_proj + C(bin12)", data=s).fit(
        cov_type="cluster", cov_kwds={"groups": s.year})
    say(f"- {pos}: {m.params['top10_proj']:+.3f} PPG (SE {m.bse['top10_proj']:.3f}, "
        f"p = {m.pvalues['top10_proj']:.4f}), n = {len(s)}")
say("")
# (c) tail vs mean: does the environment change the SPREAD of outcomes at fixed price?
say("**(c) Tail vs mean — the §N hypothesis stated as a variance question.** SD of season "
    "PPG within cost bin, on vs off a projected top-10 offence (bin-weighted pooled SD):")
say("")
for pos in POS:
    s = d[(d.pos_adp == pos) & (d.games >= 4)].copy()
    s["r"] = s.ppg - s.groupby(["bin12", "top10_proj"]).ppg.transform("mean")
    sd1 = s[s.top10_proj == 1].r.std(ddof=1)
    sd0 = s[s.top10_proj == 0].r.std(ddof=1)
    n1, n0 = (s.top10_proj == 1).sum(), (s.top10_proj == 0).sum()
    F = sd1 ** 2 / sd0 ** 2
    pF = 2 * min(stats.f.cdf(F, n1 - 1, n0 - 1), stats.f.sf(F, n1 - 1, n0 - 1))
    say(f"- {pos}: SD on top-10 {sd1:.2f} (n={n1}) vs off {sd0:.2f} (n={n0}); "
        f"F = {F:.3f}, p = {pF:.4f} (family = 0, descriptive)")
say("")
# (d) how much of the offence signal is already in the price?  §25.3 decomposition
say("**(d) §25.3 decomposition applied to the tier outcome.** Slope of each channel on "
    "the top-10 indicator (linear-probability form so the identity is exact):")
say("")
for pos in POS:
    s = d[d.pos_adp == pos].copy()
    # beta_realized: hit on top10, no price control
    br = smf.ols("hit12 ~ top10_proj", data=s).fit(
        cov_type="cluster", cov_kwds={"groups": s.year}).params["top10_proj"]
    # beta_priced: the market's own implied hit rate.  Use the bin-mean hit rate as the
    # price-implied probability (the market's tier expectation at that cost).
    s["phat_price"] = s.groupby("bin12").hit12.transform("mean")
    bp = smf.ols("phat_price ~ top10_proj", data=s).fit(
        cov_type="cluster", cov_kwds={"groups": s.year}).params["top10_proj"]
    s["resid"] = s.hit12 - s.phat_price
    bres = smf.ols("resid ~ top10_proj", data=s).fit(
        cov_type="cluster", cov_kwds={"groups": s.year})
    share = (f"priced share {bp/br:.0%}" if abs(br) > 0.02 else
             "priced share undefined: beta_realized is ~0, so the ratio is meaningless")
    say(f"- {pos} top-12: beta_realized {100*br:+.1f} pp = beta_priced {100*bp:+.1f} pp "
        f"+ beta_residual {100*bres.params['top10_proj']:+.1f} pp "
        f"(SE {100*bres.bse['top10_proj']:.1f}, p = {bres.pvalues['top10_proj']:.4f}); "
        f"{share}")
say("")

with open(f"{ROOT}/results/sectionN_notes.md", "w") as f:
    f.write("\n".join(OUT) + "\n")
print("\nwrote results/sectionN_notes.md, rb_tier_environment.csv, sectionN_tests.csv, "
      "sectionN_shares.csv, sectionN_holdout.csv")

# ===================== 9. ANOMALY CHASING (all family = 0) ==================
# Two results demand a mechanism before they can be reported as findings:
#   (A) the REALIZED arm is large and significant on all four contrasts -- but a
#       player's own touchdowns ARE his team's points, so `team points scored`
#       contains the outcome.  This is the §28.3 defect class exactly.  Rebuild the
#       realized rank LEAVE-OWN-PLAYER-OUT and see what survives.
#   (B) the PROJECTED arm is NEGATIVE for RB and survives BH at top-24 -- the
#       opposite of the owner's hypothesis.  Decompose it.
say("---")
say("")
say("## §N10 Anomaly chasing")
say("")

# ---------- (A) leave-own-player-out realized offence ----------
sc2 = []
for y in YEARS:
    w = pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
                    low_memory=False,
                    usecols=["player_id", "season", "week", "season_type", "team",
                             "rushing_tds", "receiving_tds", "rushing_2pt_conversions",
                             "receiving_2pt_conversions"])
    sc2.append(w[w.season_type == "REG"])
sc2 = pd.concat(sc2, ignore_index=True).fillna({"rushing_tds": 0, "receiving_tds": 0,
                                                "rushing_2pt_conversions": 0,
                                                "receiving_2pt_conversions": 0})
sc2["own_pts"] = 6 * (sc2.rushing_tds + sc2.receiving_tds) + \
                 2 * (sc2.rushing_2pt_conversions + sc2.receiving_2pt_conversions)
own = (sc2.groupby(["player_id", "season"]).own_pts.sum().rename("own_pts")
         .reset_index().rename(columns={"season": "year"}))
d = d.merge(own.rename(columns={"player_id": "pid"}), on=["pid", "year"], how="left")
d["own_pts"] = d.own_pts.fillna(0.0)

Rt = R.rename(columns={"season": "year"})[["year", "fr", "pts_tot", "gms"]]
d = d.merge(Rt, on=["year", "fr"], how="left")
say("**(A) The realized arm contains the outcome — leave-own-player-out rebuild.**")
say("")
say("A drafted RB/WR who finishes top-12 scores his team's touchdowns, so `team points "
    "scored` is not exogenous to his own finish. Own-TD points as a share of team points:")
say("")
for pos in POS:
    s = d[d.pos_adp == pos]
    say(f"- {pos}: mean own-TD points {s.own_pts.mean():.1f} of {s.pts_tot.mean():.0f} "
        f"team points = {100*(s.own_pts/s.pts_tot).mean():.1f}%; among top-12 finishers "
        f"{s[s.hit12==1].own_pts.mean():.1f} pts = "
        f"{100*(s[s.hit12==1].own_pts/s[s.hit12==1].pts_tot).mean():.1f}%.")
say("")
# rebuild ranks per (year, player) with that player's own points removed from HIS team
loo = []
for (y), grp in d.groupby("year"):
    base = Rt[Rt.year == y].set_index("fr")
    for idx, r in grp.iterrows():
        t = base.copy()
        t.loc[r.fr, "pts_tot"] = t.loc[r.fr, "pts_tot"] - r.own_pts
        ppg_ = t.pts_tot / t.gms
        rk = ppg_.rank(ascending=False, method="first")
        loo.append((idx, int(rk.loc[r.fr] <= 10)))
loo = pd.Series(dict(loo), name="top10_real_loo")
d["top10_real_loo"] = loo
say(f"- reclassified by the leave-own-out rank: "
    f"{int((d.top10_real_loo != d.top10_real).sum())} of {len(d)} rows "
    f"({100*(d.top10_real_loo != d.top10_real).mean():.1f}%) change top-10 status.")
say("")
say("| pos | tier | original beta | leave-own-out beta | SE | p | AME (pp) | collapse |")
say("|---|---|---|---|---|---|---|---|")
for pos in POS:
    for tier, col in [("top12", "hit12"), ("top24", "hit24")]:
        s = d[d.pos_adp == pos].copy()
        m = cl_logit(s, col, "top10_real_loo + C(bin12)")
        b_ = m.params["top10_real_loo"]
        orig = [t for t in tests if t["test"] == "N4" and t["pos"] == pos
                and t["tier"] == tier and t["env"] == "realized"][0]["estimate"]
        say(f"| {pos} | {tier} | {orig:+.3f} | **{b_:+.3f}** | "
            f"{m.bse['top10_real_loo']:.3f} | {m.pvalues['top10_real_loo']:.4f} | "
            f"{100*ame(m, s, 'top10_real_loo'):+.1f} | {100*(1-b_/orig):.0f}% |")
        tests.append(dict(test="N10loo", pos=pos, tier=tier, env="realized_loo",
                          spec="logit hit ~ top10_real_LEAVE-OWN-OUT + C(bin12), cluster(season)",
                          estimate=b_, se=m.bse["top10_real_loo"],
                          ci_lo=m.conf_int().loc["top10_real_loo", 0],
                          ci_hi=m.conf_int().loc["top10_real_loo", 1],
                          p=m.pvalues["top10_real_loo"], ame_pp=100 * ame(m, s, "top10_real_loo"),
                          mh_rd_pp=np.nan, mde=np.nan, mde_pp=np.nan, n=len(s), family=0))
say("")

# ---------- (B) decompose the projected effect: games vs PPG ----------
say("**(B) The projected effect is negative for RB — decomposed into games and PPG, "
    "within cost bin.**")
say("")
say("| pos | quantity | on projected top-10 | off | difference | cluster-t p |")
say("|---|---|---|---|---|---|")
for pos in POS:
    s = d[d.pos_adp == pos].copy()
    for q, lab in [("games", "games played"), ("ppg", "PPG | >=1 game"),
                   ("total", "season total PPR")]:
        ss = s.dropna(subset=[q])
        if q == "ppg":
            ss = ss[ss.games >= 4]
        m = smf.ols(f"{q} ~ top10_proj + C(bin12)", data=ss).fit(
            cov_type="cluster", cov_kwds={"groups": ss.year})
        say(f"| {pos} | {lab} | {ss[ss.top10_proj==1][q].mean():.2f} | "
            f"{ss[ss.top10_proj==0][q].mean():.2f} | "
            f"**{m.params['top10_proj']:+.2f}** | {m.pvalues['top10_proj']:.4f} |")
say("")

# week-18 rest hypothesis: good teams clinch and sit starters
say("Availability sub-check — is the games gap late-season rest? Mean REG appearances "
    "by week window, RB only, within cost bin (OLS with bin FE, cluster(season)):")
say("")
apps = []
for y in YEARS:
    w = pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
                    low_memory=False,
                    usecols=["player_id", "season", "week", "season_type"])
    apps.append(w[w.season_type == "REG"])
apps = pd.concat(apps, ignore_index=True).rename(columns={"season": "year",
                                                          "player_id": "pid"})
LAST = {y: (17 if y <= 2020 else 18) for y in YEARS}
apps["is_last"] = apps.apply(lambda r: r.week >= LAST[r.year], axis=1)
apps["is_last2"] = apps.apply(lambda r: r.week >= LAST[r.year] - 1, axis=1)
ag = (apps.groupby(["pid", "year"])
          .agg(g_all=("week", "size"), g_last=("is_last", "sum"),
               g_last2=("is_last2", "sum")).reset_index())
ag["g_ex_last2"] = ag.g_all - ag.g_last2
d = d.merge(ag, on=["pid", "year"], how="left")
for c in ["g_all", "g_last", "g_last2", "g_ex_last2"]:
    d[c] = d[c].fillna(0)
say("| pos | quantity | on top-10 | off | difference | p |")
say("|---|---|---|---|---|---|")
for pos in POS:
    s = d[d.pos_adp == pos].copy()
    for q, lab in [("g_all", "all REG weeks"), ("g_ex_last2", "weeks 1 to n-2"),
                   ("g_last2", "final 2 weeks")]:
        m = smf.ols(f"{q} ~ top10_proj + C(bin12)", data=s).fit(
            cov_type="cluster", cov_kwds={"groups": s.year})
        say(f"| {pos} | {lab} | {s[s.top10_proj==1][q].mean():.2f} | "
            f"{s[s.top10_proj==0][q].mean():.2f} | **{m.params['top10_proj']:+.3f}** | "
            f"{m.pvalues['top10_proj']:.4f} |")
say("")

# backfield / target competition on good offences
say("Competition sub-check — how many RB/WR does a projected top-10 offence put on the "
    "board, and how concentrated is its usage?")
say("")
comp = (d.groupby(["year", "fr", "pos_adp"]).size().rename("n_drafted").reset_index()
          .merge(ENVP, on=["year", "fr"]))
for pos in POS:
    c = comp[comp.pos_adp == pos]
    say(f"- {pos}: drafted per team-season, projected top-10 {c[c.top10_proj==1].n_drafted.mean():.2f} "
        f"vs off {c[c.top10_proj==0].n_drafted.mean():.2f} "
        f"(Welch p = {stats.ttest_ind(c[c.top10_proj==1].n_drafted, c[c.top10_proj==0].n_drafted, equal_var=False).pvalue:.4f})")
# usage concentration: share of team carries (RB) / targets (WR) taken by the top man
say("")

# indicator vs continuous discrepancy
say("Indicator-vs-continuous — the top-10 dummy and the continuous de-vigged win total "
    "disagree, so the relation is not monotone in projected quality. Hit rates by "
    "projected-rank tercile, within cost bin (bin-demeaned, RB):")
say("")
for pos in POS:
    s = d[d.pos_adp == pos].copy()
    s["terc"] = pd.qcut(s.proj_rank, 3, labels=["proj 1-11", "proj 12-21", "proj 22-32"])
    for col, tier in [("hit12", "top12"), ("hit24", "top24")]:
        s["adj"] = s[col] - s.groupby("bin12")[col].transform("mean")
        g_ = s.groupby("terc").adj.mean() * 100
        n_ = s.groupby("terc").size()
        say(f"- {pos} {tier}, bin-demeaned rate (pp vs bin mean): " +
            ", ".join(f"{k} {v:+.1f} (n={n_[k]})" for k, v in g_.items()))
say("")

# per-season stability of the RB projected effect
say("Per-season stability of the RB projected contrast (bin-stratified risk difference, "
    "pp) — §L's rule that single-season rates are not signal applies; shown for spread:")
say("")
for tier, col in [("top12", "hit12"), ("top24", "hit24")]:
    vals = []
    for y in YEARS:
        s = d[(d.pos_adp == "RB") & (d.year == y)]
        vals.append(mh_rd(s, "top10_proj", col) * 100)
    say(f"- RB {tier}: " + ", ".join(f"{y}:{v:+.0f}" for y, v in zip(YEARS, vals)) +
        f" | mean {np.nanmean(vals):+.1f}, negative in "
        f"{int(np.sum(np.array(vals) < 0))}/10 seasons")
say("")

T2 = pd.DataFrame(tests)
T2 = T2.merge(fam[["test", "pos", "tier", "env", "bh_thresh", "bh_reject"]],
              on=["test", "pos", "tier", "env"], how="left")
T2.to_csv(f"{ROOT}/results/sectionN_tests.csv", index=False)
d.to_csv(f"{ROOT}/results/sectionN_panel.csv", index=False)
with open(f"{ROOT}/results/sectionN_notes.md", "w") as f:
    f.write("\n".join(OUT) + "\n")
print("appendix written")

# ---------- (C) is the availability channel a within-bin composition effect? ----
say("**(C) Where the RB availability gap lives — by cost bin, and is it price?**")
say("")
say("| pos | bin | games on top-10 (n) | games off (n) | diff | mean ADP on | mean ADP off |")
say("|---|---|---|---|---|---|---|")
for pos in POS:
    for b in BINS:
        s1 = d[(d.pos_adp == pos) & (d.bin12 == b) & (d.top10_proj == 1)]
        s0 = d[(d.pos_adp == pos) & (d.bin12 == b) & (d.top10_proj == 0)]
        say(f"| {pos} | {b} | {s1.games.mean():.2f} ({len(s1)}) | "
            f"{s0.games.mean():.2f} ({len(s0)}) | **{s1.games.mean()-s0.games.mean():+.2f}** | "
            f"{s1.adp.mean():.1f} | {s0.adp.mean():.1f} |")
say("")

# mediation: does controlling for games kill the negative hit24 effect?
say("Mediation check — add realized games played to the RB top-24 projected model. If the "
    "effect is the availability channel it should vanish (games is a POST-TREATMENT "
    "control, so this is diagnostic only, never a causal estimate):")
say("")
for pos in POS:
    s = d[d.pos_adp == pos].copy()
    m1 = cl_logit(s, "hit24", "top10_proj + C(bin12)")
    m2 = cl_logit(s, "hit24", "top10_proj + games + C(bin12)")
    say(f"- {pos} top-24: without games {m1.params['top10_proj']:+.3f} "
        f"(p = {m1.pvalues['top10_proj']:.4f}) -> with games "
        f"{m2.params['top10_proj']:+.3f} (p = {m2.pvalues['top10_proj']:.4f}); "
        f"games coefficient {m2.params['games']:+.3f} (p = {m2.pvalues['games']:.4f})")
say("")

# ---------- (D) multiplicity re-accounting after the mechanical withdrawal ----
say("**(D) Multiplicity, re-accounted.** The four REALIZED contrasts are mechanically "
    "contaminated (A) and are withdrawn, exactly as §28.3 withdrew its clairvoyant "
    "positional-SOS false positive. Two accountings are reported; the family declared "
    "before fitting is the first, and the second is what it becomes once the contaminated "
    "arm is replaced by its leave-own-out rebuild.")
say("")
fam_as_declared = fam[["pos", "tier", "env", "estimate", "p"]].copy()
alt = []
for t in tests:
    if t["test"] == "N4" and t["env"] == "projected":
        alt.append((t["pos"], t["tier"], "projected", t["estimate"], t["p"]))
    if t["test"] == "N10loo":
        alt.append((t["pos"], t["tier"], "realized (leave-own-out)", t["estimate"], t["p"]))
A = pd.DataFrame(alt, columns=["pos", "tier", "env", "estimate", "p"]).sort_values(
    "p").reset_index(drop=True)
A["bh_thresh"] = 0.10 * (A.index + 1) / len(A)
A["bh_reject"] = False
if (A.p <= A.bh_thresh).any():
    A.loc[:A.index[A.p <= A.bh_thresh].max(), "bh_reject"] = True
say("| rank | pos | tier | env | beta | raw p | BH threshold | reject |")
say("|---|---|---|---|---|---|---|---|")
for j, r in A.iterrows():
    say(f"| {j+1} | {r.pos} | {r.tier} | {r.env} | {r.estimate:+.3f} | {r.p:.4f} | "
        f"{r.bh_thresh:.4f} | {'**YES**' if r.bh_reject else 'no'} |")
say("")
say(f"**{int(A.bh_reject.sum())} of {len(A)} survive** in the de-contaminated family.")
say("")
# and the projected-only family
P4 = fam[fam.env == "projected"].sort_values("p").reset_index(drop=True)
P4["thr"] = 0.10 * (P4.index + 1) / len(P4)
say("Projected arm alone (4 tests, the only preseason-knowable ones): " +
    "; ".join(f"{r.pos} {r.tier} p={r.p:.4f} vs thr {r.thr:.4f}" for _, r in P4.iterrows()) +
    f" -> {int((P4.p <= P4.thr).sum())} survive.")
say("")
say("**This matters for reading §N7.** RB top-24 projected (p = .0410) cleared BH in the "
    "as-declared family only because the four contaminated realized tests occupied ranks "
    "1-4 and lifted its threshold to .0625. Against the four preseason-knowable tests "
    "alone its threshold is .025 and it does not clear. It is reported as a suggestive, "
    "uncorrected signal, not as a survivor.")
say("")
A.to_csv(f"{ROOT}/results/sectionN_bh_decontaminated.csv", index=False)

T3 = pd.DataFrame(tests)
T3 = T3.merge(fam[["test", "pos", "tier", "env", "bh_thresh", "bh_reject"]],
              on=["test", "pos", "tier", "env"], how="left")
T3.to_csv(f"{ROOT}/results/sectionN_tests.csv", index=False)
with open(f"{ROOT}/results/sectionN_notes.md", "w") as f:
    f.write("\n".join(OUT) + "\n")
print("part C/D written")

# ---------- (E) is the RB games deficit role (committee) or injury? ----------
say("**(E) Is the RB availability deficit a roster ROLE effect or an injury effect?** "
    "Split each team-season's drafted RBs into the LEAD back (lowest ADP on that team) "
    "and the rest. A committee/depth mechanism should show the deficit mainly in the "
    "non-lead backs; an injury mechanism should not care about role.")
say("")
d["rb_rank_on_team"] = d[d.pos_adp == "RB"].groupby(["year", "fr"]).adp.rank(method="first")
d["is_lead"] = (d.rb_rank_on_team == 1)
say("| RB role | games on top-10 (n) | games off (n) | diff | bin-FE p | "
    "top-24 rate on | off |")
say("|---|---|---|---|---|---|---|")
for lab, mask in [("lead back", d.is_lead == True), ("non-lead", d.is_lead == False)]:
    s = d[(d.pos_adp == "RB") & mask].copy()
    s["bin12"] = s.bin12.cat.remove_unused_categories()
    m = smf.ols("games ~ top10_proj + C(bin12)", data=s).fit(
        cov_type="cluster", cov_kwds={"groups": s.year})
    s1, s0 = s[s.top10_proj == 1], s[s.top10_proj == 0]
    say(f"| {lab} | {s1.games.mean():.2f} ({len(s1)}) | {s0.games.mean():.2f} ({len(s0)}) "
        f"| **{m.params['top10_proj']:+.2f}** | {m.pvalues['top10_proj']:.4f} | "
        f"{s1.hit24.mean():.1%} | {s0.hit24.mean():.1%} |")
# and the number of RBs the team carries, as the direct committee measure
nrb = (d[d.pos_adp == "RB"].groupby(["year", "fr"]).size().rename("n_rb_drafted")
       .reset_index())
d = d.merge(nrb, on=["year", "fr"], how="left")
s = d[d.pos_adp == "RB"].copy()
m = smf.ols("games ~ top10_proj + n_rb_drafted + C(bin12)", data=s).fit(
    cov_type="cluster", cov_kwds={"groups": s.year})
say("")
say(f"- Controlling for how many RBs the team has on the board, the top-10 games effect "
    f"moves {-1.089:+.2f} -> {m.params['top10_proj']:+.2f} "
    f"(p = {m.pvalues['top10_proj']:.4f}); the backfield-depth coefficient is "
    f"{m.params['n_rb_drafted']:+.2f} games per extra drafted RB "
    f"(p = {m.pvalues['n_rb_drafted']:.4f}).")
say("")

# ---------- (F) is the games deficit an AGE / experience composition effect? ----
mt = pd.read_csv(f"{ROOT}/data/meta/players_meta.csv", low_memory=False,
                 usecols=["gsis_id", "birth_date", "rookie_season", "draft_round"])
mt = mt.dropna(subset=["gsis_id"]).rename(columns={"gsis_id": "pid"})
d = d.merge(mt, on="pid", how="left")
d["age"] = d.year - pd.to_datetime(d.birth_date, errors="coerce").dt.year
d["exp"] = d.year - d.rookie_season
OUT.append("")
say("**(F) Is the RB games deficit an age / experience composition effect?** "
    "Contenders sign veterans; veterans miss more games. Checked because it is the "
    "obvious alternative to a role story.")
say("")
s = d[d.pos_adp == "RB"].copy()
for q in ["age", "exp"]:
    ss = s.dropna(subset=[q])
    m = smf.ols(f"{q} ~ top10_proj + C(bin12)", data=ss).fit(
        cov_type="cluster", cov_kwds={"groups": ss.year})
    say(f"- RB {q}: on projected top-10 {ss[ss.top10_proj==1][q].mean():.2f} vs off "
        f"{ss[ss.top10_proj==0][q].mean():.2f}; within-bin difference "
        f"{m.params['top10_proj']:+.3f} (p = {m.pvalues['top10_proj']:.4f}), n = {len(ss)}")
ss = s.dropna(subset=["age", "exp"])
m0 = smf.ols("games ~ top10_proj + C(bin12)", data=ss).fit(
    cov_type="cluster", cov_kwds={"groups": ss.year})
m1 = smf.ols("games ~ top10_proj + age + exp + n_rb_drafted + C(bin12)", data=ss).fit(
    cov_type="cluster", cov_kwds={"groups": ss.year})
say(f"- games effect with age, experience and backfield depth all controlled: "
    f"{m0.params['top10_proj']:+.3f} (p = {m0.pvalues['top10_proj']:.4f}) -> "
    f"{m1.params['top10_proj']:+.3f} (p = {m1.pvalues['top10_proj']:.4f}), n = {len(ss)}")
say("")
say("**Honest statement of what is and is not explained.** The negative RB tier effect is "
    "*located* precisely -- it is entirely a games-played channel, with per-game production "
    "flat -- and roughly a quarter of the games gap is accounted for by projected top-10 "
    "offences carrying deeper backfields (-0.93 games per extra drafted RB, p = .0001). "
    "The remaining ~0.8 games is not explained by role, age or experience, and is not "
    "late-season rest. Given it is a p = .04 estimate that does not clear its own "
    "multiplicity screen, the residual is most plausibly noise; it is recorded as open "
    "rather than given a story.")

T5 = pd.DataFrame(tests)
T5 = T5.merge(fam[["test", "pos", "tier", "env", "bh_thresh", "bh_reject"]],
              on=["test", "pos", "tier", "env"], how="left")
T5.to_csv(f"{ROOT}/results/sectionN_tests.csv", index=False)
d.to_csv(f"{ROOT}/results/sectionN_panel.csv", index=False)
with open(f"{ROOT}/results/sectionN_notes.md", "w") as f:
    f.write("\n".join(OUT) + "\n")
print("done F")

say("## §N11 What §N establishes")
say("")
say("1. **The owner's unconditional number, as asked:** 47.5% of RB1-12 finishers "
    "2015-2024 played in a realized top-10 scoring offence (Wilson [38.8, 56.4], n = 120) "
    "against a 31.2% even-spread base rate; RB13-24 is 34.2% [26.3, 43.0]; WR1-12 is "
    "51.2% [42.4, 60.0]. So the raw pattern the owner has in mind is real and is not "
    "RB-specific: WRs show it at least as strongly.")
say("2. **But it is a realized, not a projected, fact, and it is not actionable.** Using "
    "the preseason-knowable projection instead, the RB1-12 share falls to 31.7% "
    "[24.0, 40.4] -- indistinguishable from the 31.2% base rate. The entire gap is "
    "hindsight: teams are top-10 offences partly *because* their RB finished top-12.")
say("3. **Quantified: the realized arm is mechanically contaminated.** Own touchdowns are "
    "18.0% of team points for a top-12 RB. Rebuilt leave-own-player-out, all four "
    "realized contrasts collapse by 95-197% and none is significant. Same defect class as "
    "§28.3; the four BH survivors are withdrawn.")
say("4. **Cost-conditioned, a projected top-10 offence does NOT raise P(top-12).** RB "
    "-3.6 pp (p = .31, MDE 13.9 pp), WR -1.1 pp (p = .77, MDE 15.0 pp). Both point "
    "estimates are negative.")
say("5. **The one suggestive signal points the wrong way and resolves to availability.** "
    "RB top-24 on a projected top-10 offence is -6.6 pp (p = .041, negative in 8 of 10 "
    "seasons, sign held in holdout). It is fully mediated by games played: RBs on "
    "projected top-10 offences play 1.09 fewer games at the same draft cost (p = .0035), "
    "PPG is flat (-0.18, p = .54), and controlling for games the tier effect goes to "
    "-0.070 (p = .80). It does not clear BH against the four preseason-knowable tests.")
say("6. **Consistency with §I3.** The mean channel is null here too (RB -0.18 PPG, "
    "p = .54; WR -0.31, p = .49, cost-conditioned), as §I3's 77%-priced result predicts. "
    "§N's separate question -- does environment reshape the tail at fixed price? -- "
    "answers no for the upper tail and weakly *negative* for RBs at the bust threshold.")
say("")
say("**Nothing enters theta*.** No contrast survives the de-contaminated family; the "
    "availability finding is a restatement, at team level, of what §A and §L already own.")


T6 = pd.DataFrame(tests)
T6 = T6.merge(fam[["test", "pos", "tier", "env", "bh_thresh", "bh_reject"]],
              on=["test", "pos", "tier", "env"], how="left")
T6.to_csv(f"{ROOT}/results/sectionN_tests.csv", index=False)
d.to_csv(f"{ROOT}/results/sectionN_panel.csv", index=False)
with open(f"{ROOT}/results/sectionN_notes.md", "w") as f:
    f.write("\n".join(OUT) + "\n")
print("§N complete")

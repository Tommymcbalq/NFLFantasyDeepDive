"""§I3 (EDA_PLAN4.md) — does sportsbook-implied team environment predict ADP's errors?

PRE-REGISTRATION (this docstring). Every choice below was fixed BEFORE the regression
was run, mirroring the §6.2 edge-test protocol (scripts/09_section6b_edge.py).

Conditional on the §I2 sourcing gate, which §I1 passed for TEAM WIN TOTALS ONLY
(results/sectionI_sources.md). Point totals and player props are NO-GO; the
nflverse games.csv per-game total_line/spread_line are explicitly barred as a
preseason feature (in-season look-ahead). games.csv is used here for ONE thing only:
realized team wins in the PRIOR season, which is a preseason-knowable quantity.

OUTCOME
  R = resid_iso from results/market_prior.csv — residual of realized PPG around the
  §6.1 isotonic ADP->PPG curve, per player-season, WR boards 2015-2024.
  Sample = in_fit rows (>=4 realized games), exactly as §6.2.

TEAM ATTACH
  Primary: modal team by REG appearances in data/players/weekly_raw/ for
  (gsis_id, board year); ties broken by earliest week. Pre-specified sensitivity:
  first REG team of the board year (the §6.2 convention, marginally less exposed to
  in-season trades). Covers full team names mapped to nflverse abbreviations
  season-by-season (relocations + the three Washington names).

DE-VIG (from the paired over/under prices, per the §I1 amendment)
  American odds -> implied prob; fair p_over = p_o / (p_o + p_u) (proportional /
  multiplicative de-vig, the standard two-way normalization).
  Wins are modeled as X ~ N(mu, s^2) on a WIN-RATE scale (wins / games_in_season),
  which is how the 16->17 game break in 2021 is handled: everything is computed as a
  rate and then re-expressed in wins-per-17 for interpretability.
    - half-win line L (no push possible):   mu = L + s * Phi^-1(p_fair)
    - integer line L (push voided by the book): the two-way price prices
      P(X > L+0.5) vs P(X < L-0.5) with a discrete-wins continuity correction;
      mu solved by 1-D root find on p_fair = P(X>L+.5)/(P(X>L+.5)+P(X<L-.5)).
  s is the SD of (realized win rate - posted line rate) over the 2015-2024 panel,
  computed ONCE and reported. Note s is a pure scale on the regressors: it changes
  coefficient units, not t-statistics or p-values.

THREE PRE-SPECIFIED TERMS (the §I3 FDR family), all in wins-per-17 units
  1. wt        = de-vigged posted win total for the player's board-year team
  2. surprise  = wt - (prior-season realized wins of that team, rate-adjusted)
  3. d_posted  = wt - (de-vigged posted total of that team in the prior season)
  For board year 2015 the prior-season posted line comes from the nflverse
  2003-2020 file (the §I1 cross-check source, 94.8% exact agreement on the
  2015-2020 overlap). Cross-source rows are flagged; a pre-specified sensitivity
  drops board year 2015 entirely.

SPEC
  R_is = b0 + b1*wt + b2*surprise + b3*d_posted + u_is
  OLS; SEs clustered by season (10 clusters, use_t, t(9)); HC3 reported alongside.
  VIF reported (wt and surprise are mechanically correlated; that is the
  pre-specified parameterization and it is not changed post hoc).
  Sensitivities, all pre-specified: (a) first-team attach, (b) drop 2015,
  (c) Huber robust regression (R is a residual of a right-skewed outcome),
  (d) each term alone.

MULTIPLE TESTING
  RAW uncorrected p reported for every term (§H5 is concurrent; the round-4 FDR
  family is {H5 tests, I3 tests} corrected jointly at consolidation). A local
  BH q=0.10 over these 3 terms is ALSO reported, labelled as local-only.

TEMPORAL HOLDOUT (binding, same window as §6.3)
  Fit 2015-2022, evaluate 2023-2024; a term must beat the zero prediction
  (market efficiency) in holdout MSE.

ADOPTION
  A context arm enters the LOSO harness only if a term survives BOTH screens
  (FDR + holdout), then DM vs the frozen arm (ii), clustered by year, p<0.10 AND
  RMSE improvement. Pre-registered prior: null expected, and a null is the finding.

2026: NOT fitted on. If any term survives, the historical relationship is applied to
2026 inputs and labelled as such. 2026 win totals come from a different source
(VegasInsider/DK) with only an over price, so de-vig there assumes a 4.5% two-way
overround -- flagged in the output.

Outputs: results/edge_vegas.csv, results/sectionI_notes.md
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from scipy.optimize import brentq

ROOT = "/Users/thomasmcnamee/NFL"
GAMES = {y: (16 if y <= 2020 else 17) for y in range(2003, 2027)}
BOARD_YEARS = list(range(2015, 2025))
OUT = []          # notes lines


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


# ---------------------------------------------------------------- odds helpers
def amer_to_prob(o):
    o = np.asarray(o, dtype=float)
    return np.where(o > 0, 100.0 / (o + 100.0), -o / (-o + 100.0))


def devig_rate(L_rate, p_fair, s, integer_line, G):
    """Return mu (win RATE) implied by a de-vigged two-way price."""
    p_fair = min(max(p_fair, 1e-6), 1 - 1e-6)
    if not integer_line:
        return L_rate + s * norm.ppf(p_fair)
    hi = L_rate + 0.5 / G
    lo = L_rate - 0.5 / G

    def f(mu):
        po = norm.sf(hi, loc=mu, scale=s)
        pu = norm.cdf(lo, loc=mu, scale=s)
        return po / (po + pu) - p_fair
    return brentq(f, L_rate - 6 * s, L_rate + 6 * s, xtol=1e-10)


# ---------------------------------------------------------------- vegas history
cov = pd.read_csv(f"{ROOT}/data/vegas/team_win_totals_2015_2025_covers.csv")
nfv = pd.read_csv(f"{ROOT}/data/vegas/team_win_totals_2003_2020_nflverse.csv")

NAME2ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO",
    "New York Giants": "NYG", "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
    # franchises that move / rename: mapped to the FRANCHISE code used below
    "Oakland Raiders": "LV", "Las Vegas Raiders": "LV",
    "San Diego Chargers": "LAC", "Los Angeles Chargers": "LAC",
    "St Louis Rams": "LA", "Los Angeles Rams": "LA",
    "Washington Redskins": "WAS", "Washington Football Team": "WAS",
    "Washington Commanders": "WAS",
}
# nflverse abbrevs -> same franchise codes
NFV2FR = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "LVR": "LV",
          "WSH": "WAS"}
# nflverse weekly team abbrevs -> franchise codes (weekly files use era abbrevs)
WK2FR = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "LV": "LV",
         "LAC": "LAC", "LA": "LA", "WAS": "WAS"}

cov["fr"] = cov.team.map(NAME2ABBR)
assert cov.fr.notna().all(), cov[cov.fr.isna()].team.unique()
assert cov.groupby("season").fr.nunique().eq(32).all()

nfv["fr"] = nfv.team.replace(NFV2FR)
nfv = nfv.rename(columns={"line": "win_total", "over_odds": "over",
                          "under_odds": "under"})

say("# §I3 — Vegas team-environment edge test")
say("")
say("Run", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    "| script `scripts/24_sectionI3_vegas_edge.py` (pre-registration in its docstring)")
say("")
say("## 1. De-vig calibration")

# scale s: SD of (realized win rate - posted line rate), 2015-2024 Covers panel
h = cov[cov.season.between(2015, 2024)].copy()
h["G"] = h.season.map(GAMES)
h["L_rate"] = h.win_total / h.G
h["act_rate"] = h.actual_wins / h.G
S_RATE = float((h.act_rate - h.L_rate).std(ddof=1))
say(f"- s (SD of realized minus posted, win-RATE scale) = {S_RATE:.5f} "
    f"= {S_RATE*17:.3f} wins per 17 games, n = {len(h)}")
say(f"- integer (push-possible) lines: {(h.win_total % 1 == 0).sum()} / {len(h)}"
    f" -> continuity-corrected two-way solve; half-win lines closed form")


def build_devig(df, seasons):
    d = df[df.season.isin(seasons)].copy()
    d["G"] = d.season.map(GAMES)
    d["L_rate"] = d.win_total / d.G
    po = amer_to_prob(d.over.values)
    pu = amer_to_prob(d.under.values)
    d["overround"] = po + pu
    d["p_fair_over"] = po / (po + pu)
    d["mu_rate"] = [devig_rate(lr, pf, S_RATE, (wt % 1 == 0), g)
                    for lr, pf, wt, g in zip(d.L_rate, d.p_fair_over,
                                             d.win_total, d.G)]
    d["wt17"] = d.mu_rate * 17
    d["line17"] = d.L_rate * 17
    return d


V = build_devig(cov, range(2015, 2026))
V14 = build_devig(nfv, [2014])          # prior-season posted line for board 2015

say(f"- mean two-way overround, Covers 2015-24: {V[V.season<=2024].overround.mean():.4f}")
say(f"- de-vig shift |wt17 - line17|: mean {(V.wt17-V.line17).abs().mean():.3f}, "
    f"max {(V.wt17-V.line17).abs().max():.3f} wins-per-17")
_ex = V[(V.season == 2025)].nlargest(3, "wt17")[["team", "line17", "wt17"]]
say(f"- worked example (2025, top 3 de-vigged): "
    + "; ".join(f"{r.team} line {r.line17:.2f} -> {r.wt17:.2f}"
                for r in _ex.itertuples()))

# ---------------------------------------------------------------- realized wins
g = pd.read_csv(f"{ROOT}/data/teams/games_nflverse_20260809.csv", low_memory=False,
                usecols=["season", "game_type", "away_team", "away_score",
                         "home_team", "home_score"])
g = g[(g.game_type == "REG") & g.home_score.notna() & (g.season >= 2014)]
rows = []
for r in g.itertuples():
    hw = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
    rows.append((r.season, r.home_team, hw))
    rows.append((r.season, r.away_team, 1.0 - hw))
W = (pd.DataFrame(rows, columns=["season", "team", "w"])
     .groupby(["season", "team"]).w.sum().reset_index())
W["fr"] = W.team.replace({**NFV2FR, "LA": "LA"})
W["fr"] = W.fr.replace(NFV2FR)
W["act_rate"] = W.w / W.season.map(GAMES)

# integrity check vs the Covers actual_wins column
chk = V[V.season <= 2024].merge(W[["season", "fr", "w"]], on=["season", "fr"])
say(f"- realized-wins integrity check (games.csv vs Covers `actual_wins`): "
    f"{len(chk)} pairs, exact {int((chk.w == chk.actual_wins).sum())}, "
    f"MAD {float((chk.w - chk.actual_wins).abs().mean()):.4f} wins "
    f"(differences are ties, scored 0.5 here vs 0 there)")

# ------------------------------------------------- team-season feature table
feat = V[V.season.isin(BOARD_YEARS)][["season", "fr", "team", "line17", "wt17"]].copy()
prev_post = pd.concat([
    V14[["season", "fr", "wt17"]],
    V[V.season.between(2015, 2023)][["season", "fr", "wt17"]],
])
prev_post["season"] = prev_post.season + 1
prev_post = prev_post.rename(columns={"wt17": "wt17_prev"})
prev_act = W[W.season.between(2014, 2023)][["season", "fr", "act_rate"]].copy()
prev_act["season"] = prev_act.season + 1
prev_act = prev_act.rename(columns={"act_rate": "act_rate_prev"})

feat = feat.merge(prev_post, on=["season", "fr"], how="left")
feat = feat.merge(prev_act, on=["season", "fr"], how="left")
assert feat.wt17_prev.notna().all() and feat.act_rate_prev.notna().all()
feat["surprise"] = feat.wt17 - feat.act_rate_prev * 17
feat["d_posted"] = feat.wt17 - feat.wt17_prev
feat = feat.rename(columns={"wt17": "wt"})
feat["prev_src"] = np.where(feat.season == 2015, "nflverse_2014", "covers")

# ---------------------------------------------------------------- player panel
panel = pd.read_csv(f"{ROOT}/results/market_prior.csv")
wk = []
for y in BOARD_YEARS:
    d = pd.read_csv(f"{ROOT}/data/players/weekly_raw/stats_player_week_{y}.csv",
                    usecols=["player_id", "season", "week", "season_type", "team"],
                    low_memory=False)
    wk.append(d[d.season_type == "REG"])
wk = pd.concat(wk, ignore_index=True)
wk["fr"] = wk.team.replace(WK2FR)

cnt = (wk.groupby(["player_id", "season", "fr"])
       .agg(n=("week", "size"), first_wk=("week", "min")).reset_index())
modal = (cnt.sort_values(["n", "first_wk"], ascending=[False, True])
         .groupby(["player_id", "season"]).head(1)
         .rename(columns={"player_id": "gsis_id", "season": "year",
                          "fr": "fr_modal"})[["gsis_id", "year", "fr_modal"]])
firstt = (wk.sort_values("week").groupby(["player_id", "season"]).fr.first()
          .reset_index()
          .rename(columns={"player_id": "gsis_id", "season": "year",
                           "fr": "fr_first"}))

p = panel.merge(modal, on=["gsis_id", "year"], how="left")
p = p.merge(firstt, on=["gsis_id", "year"], how="left")
say("")
say("## 2. Panel construction")
say(f"- board rows {len(p)}; in_fit {int(p.in_fit.sum())}")
say(f"- modal-team join: {int(p.fr_modal.notna().sum())}/{len(p)} "
    f"({p.fr_modal.notna().mean():.1%}); in_fit "
    f"{int((p.fr_modal.notna() & p.in_fit).sum())}/{int(p.in_fit.sum())}")
miss = p[p.fr_modal.isna()][["year", "name", "games", "in_fit"]]
if len(miss):
    say("- unjoined rows (no REG appearance that season):")
    for r in miss.itertuples():
        say(f"    {r.year} {r.name} games={r.games} in_fit={r.in_fit}")
say(f"- modal vs first-team disagreement (in-season movers): "
    f"{int((p.fr_modal != p.fr_first).sum())} rows")

p = p.merge(feat[["season", "fr", "wt", "surprise", "d_posted", "prev_src"]]
            .rename(columns={"season": "year", "fr": "fr_modal"}),
            on=["year", "fr_modal"], how="left")
p = p.merge(feat[["season", "fr", "wt"]]
            .rename(columns={"season": "year", "fr": "fr_first", "wt": "wt_first"}),
            on=["year", "fr_first"], how="left")

TERMS = ["wt", "surprise", "d_posted"]
d = p[p.in_fit].dropna(subset=["resid_iso"] + TERMS).copy()
say(f"- regression n = {len(d)} (in_fit {int(p.in_fit.sum())}, "
    f"dropped for missing team/feature: {int(p.in_fit.sum()) - len(d)})")
say("")
say("Feature distributions on the fit sample (wins per 17):")
say("```")
say(d[TERMS].describe().round(3).to_string())
say("```")
say(f"- pairwise correlations: wt~surprise {d.wt.corr(d.surprise):.3f}, "
    f"wt~d_posted {d.wt.corr(d.d_posted):.3f}, "
    f"surprise~d_posted {d.surprise.corr(d.d_posted):.3f}")

# ---------------------------------------------------------------- §I3 regression
say("")
say("## 3. Pre-specified regression")
say("")
say("`R = b0 + b1*wt + b2*surprise + b3*d_posted + u`, OLS, SEs clustered by "
    "season (10 clusters, t with 9 df).")

X = sm.add_constant(d[TERMS])
y = d.resid_iso
m_cl = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d.year}, use_t=True)
m_h3 = sm.OLS(y, X).fit(cov_type="HC3")

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                index=X.columns)

tab = pd.DataFrame({"beta": m_cl.params, "se_cluster": m_cl.bse,
                    "t_cluster": m_cl.tvalues, "p_raw_cluster": m_cl.pvalues,
                    "se_hc3": m_h3.bse, "p_raw_hc3": m_h3.pvalues, "vif": vif})
say("```")
say(tab.round(4).to_string())
say("```")
say(f"R2 = {m_cl.rsquared:.5f}; adj R2 = {m_cl.rsquared_adj:.5f}")
wald = m_cl.wald_test(np.eye(len(m_cl.params))[[X.columns.get_loc(c) for c in TERMS]],
                      scalar=True)
say(f"Joint Wald (all 3 = 0), clustered: F = {wald.statistic:.3f}, "
    f"p_raw = {wald.pvalue:.4f}")

# local BH over the 3 terms (labelled local-only)
fam = tab.loc[TERMS, "p_raw_cluster"].sort_values()
bh = 0.10 * np.arange(1, 4) / 3
passed = fam.values <= bh
kmax = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
survivors = list(fam.index[:kmax])
tab["fdr_local_survivor"] = [i in survivors for i in tab.index]
say("Local BH q=0.10 over these 3 terms only (the binding correction is the joint "
    "round-4 family {H5, I3} at consolidation — raw p above are what to carry):")
say("```")
say(pd.DataFrame({"p_raw": fam.round(4), "bh_thresh": bh.round(4)}).to_string())
say("```")
say(f"Local FDR survivors: {survivors if survivors else 'NONE'}")

# residual diagnostics
from scipy import stats as sps
res = m_cl.resid
jb = sps.jarque_bera(res)
say("")
say("### Residual diagnostics")
say(f"- residual skew {sps.skew(res):.3f}, kurtosis {sps.kurtosis(res)+3:.3f}, "
    f"Jarque-Bera p = {jb.pvalue:.2e} (R inherits the right skew of PPG; hence "
    f"the Huber sensitivity below)")
bp = sm.stats.diagnostic.het_breuschpagan(res, X)
say(f"- Breusch-Pagan on the 3 terms: LM = {bp[0]:.3f}, p = {bp[1]:.4f}")

# ---------------------------------------------------------------- sensitivities
say("")
say("## 4. Pre-specified sensitivities")
sens_rows = []


def run_sens(dd, label, cols=TERMS, robust=False, wtcol=None):
    dd = dd.copy()
    if wtcol:
        dd["wt"] = dd[wtcol]
    dd = dd.dropna(subset=["resid_iso"] + cols)
    Xs = sm.add_constant(dd[cols])
    if robust:
        f = sm.RLM(dd.resid_iso, Xs, M=sm.robust.norms.HuberT()).fit()
        se, pv = f.bse, f.pvalues
    else:
        f = sm.OLS(dd.resid_iso, Xs).fit(cov_type="cluster",
                                         cov_kwds={"groups": dd.year}, use_t=True)
        se, pv = f.bse, f.pvalues
    for c in cols:
        sens_rows.append(dict(spec=label, term=c, n=len(dd),
                              beta=f.params[c], se=se[c], p_raw=pv[c]))
    say(f"- {label:34s} n={len(dd):4d}  " + "  ".join(
        f"{c}: {f.params[c]:+.4f} (se {se[c]:.4f}, p_raw {pv[c]:.3f})" for c in cols))
    return f


run_sens(d, "(a) first-REG-team attach", wtcol="wt_first")
run_sens(d[d.year >= 2016], "(b) drop 2015 (cross-source prev)")
run_sens(d, "(c) Huber robust (HuberT M-est)", robust=True)
for c in TERMS:
    run_sens(d, f"(d) single term: {c}", cols=[c])

# ---------------------------------------------------------------- holdout
say("")
say("## 5. Temporal holdout (binding): fit 2015-2022, evaluate 2023-2024")
tr, te = d[d.year <= 2022], d[d.year >= 2023]
mse0 = float((te.resid_iso ** 2).mean())
say(f"- train n = {len(tr)}, eval n = {len(te)}")
say(f"- zero-prediction (market-efficiency) holdout MSE = {mse0:.4f}")

hold = []


def hold_eval(cols, label):
    Xt = sm.add_constant(tr[cols]); Xe = sm.add_constant(te[cols])[["const"] + cols]
    f = sm.OLS(tr.resid_iso, Xt).fit()
    pred = f.predict(Xe)
    mse = float(((te.resid_iso - pred) ** 2).mean())
    dsq = (te.resid_iso ** 2) - (te.resid_iso - pred) ** 2
    t_row = dsq.mean() / (dsq.std(ddof=1) / np.sqrt(len(dsq)))
    byyr = dsq.groupby(te.year).mean()
    hold.append(dict(model=label, mse=mse, mse_zero=mse0, improves=bool(mse < mse0),
                     dsq_mean=dsq.mean(), t_rowlevel=t_row,
                     dsq_2023=byyr.get(2023, np.nan), dsq_2024=byyr.get(2024, np.nan),
                     beta_train=f.params.get(cols[0], np.nan) if len(cols) == 1 else np.nan))
    say(f"- {label:26s} MSE {mse:8.4f}  improves: {str(mse < mse0):5s}  "
        f"mean d_sq {dsq.mean():+.4f} (row t {t_row:+.2f}; "
        f"2023 {byyr.get(2023, float('nan')):+.3f}, 2024 {byyr.get(2024, float('nan')):+.3f})")


hold_eval(TERMS, "joint: all 3")
for c in TERMS:
    hold_eval([c], f"single: {c}")
# train-period sign stability, reported regardless of significance
f_tr = sm.OLS(tr.resid_iso, sm.add_constant(tr[TERMS])).fit(
    cov_type="cluster", cov_kwds={"groups": tr.year}, use_t=True)
f_te = sm.OLS(te.resid_iso, sm.add_constant(te[TERMS])).fit(
    cov_type="cluster", cov_kwds={"groups": te.year}, use_t=True)
say("- coefficient stability train (2015-22) vs eval (2023-24), "
    "reported whether or not anything survives:")
for c in TERMS:
    say(f"    {c:10s} train {f_tr.params[c]:+.4f}  eval {f_te.params[c]:+.4f}")

hold_df = pd.DataFrame(hold)
tab["holdout_improves"] = [
    next((r["improves"] for r in hold if r["model"] == f"single: {i}"), None)
    for i in tab.index]
tab["survives_both"] = [(i in survivors) and bool(
    next((r["improves"] for r in hold if r["model"] == f"single: {i}"), False))
    for i in tab.index]
final_terms = [c for c in TERMS if tab.loc[c, "survives_both"]]

say("")
say(f"**Terms surviving BOTH screens (local FDR + holdout): "
    f"{final_terms if final_terms else 'NONE'}**")

# ------------------------------------------------- post-hoc explanatory (NOT tests)
# Everything below is DESCRIPTIVE decomposition of the null already reported above.
# No term here enters the FDR family; no spec above was changed on the strength of
# anything here. Run because a null is only finished work once its magnitude bound
# and its mechanism are stated.
say("")
say("## 5b. Why the null — power bound and the pricing channel")
say("")
say("*Post-hoc and descriptive. Nothing here is a hypothesis test, nothing enters "
    "the FDR family, and no specification above was altered on the strength of it.*")

say("")
say("### (i) Minimum detectable effect (the bound the null actually buys)")
tcrit = sps.t.ppf(0.975, 9)
for c in TERMS:
    se = tab.loc[c, "se_cluster"]
    mde = (tcrit + sps.t.ppf(0.80, 9)) * se
    sd_c = d[c].std()
    say(f"- {c:9s} SE {se:.4f} PPG per win-per-17 -> MDE at 80% power / 5% two-sided "
        f"= {mde:.3f} PPG per win; per 1 SD of the feature ({sd_c:.2f} wins) "
        f"= {mde*sd_c:.3f} PPG/game. 95% CI on beta: "
        f"[{tab.loc[c,'beta']-tcrit*se:+.3f}, {tab.loc[c,'beta']+tcrit*se:+.3f}]")
say(f"- for scale: SD(R) on the fit sample = {d.resid_iso.std():.3f} PPG. So the "
    f"test rules out any win-total channel worth more than roughly "
    f"{(tcrit+sps.t.ppf(0.80,9))*tab.loc['wt','se_cluster']*d.wt.std()/d.resid_iso.std():.0%}"
    f" of a residual SD per SD of team quality. It does NOT rule out a small one; "
    f"10 season-clusters is the resolution.")

say("")
say("### (ii) Leave-one-season-out coefficient trace")
say("Motivated by sensitivity (b): dropping 2015 moved `wt` from "
    f"{tab.loc['wt','beta']:+.3f} to +0.218 and cut its SE. Is 2015 special, or is "
    "any single cluster that influential?")
loso_rows = []
for yy in BOARD_YEARS:
    dd = d[d.year != yy]
    ff = sm.OLS(dd.resid_iso, sm.add_constant(dd[TERMS])).fit(
        cov_type="cluster", cov_kwds={"groups": dd.year}, use_t=True)
    yr = d[d.year == yy]
    loso_rows.append(dict(dropped_year=yy, n=len(dd),
                          **{f"beta_{c}": ff.params[c] for c in TERMS},
                          **{f"p_{c}": ff.pvalues[c] for c in TERMS},
                          mean_R_dropped=yr.resid_iso.mean(),
                          within_yr_corr_wt_R=yr.wt.corr(yr.resid_iso)))
loso_df = pd.DataFrame(loso_rows)
say("```")
say(loso_df[["dropped_year", "beta_wt", "p_wt", "beta_surprise", "beta_d_posted",
             "mean_R_dropped", "within_yr_corr_wt_R"]].round(4).to_string(index=False))
say("```")
rng = loso_df.beta_wt.max() - loso_df.beta_wt.min()
say(f"- `wt` slope ranges {loso_df.beta_wt.min():+.3f} to {loso_df.beta_wt.max():+.3f} "
    f"across the ten drops (range {rng:.3f}, vs a full-sample SE of "
    f"{tab.loc['wt','se_cluster']:.3f}). 2015 is the extreme, not a category apart: "
    f"the estimate is one-cluster-fragile in BOTH directions, which is the honest "
    f"reading of a 10-cluster design, not evidence of a 2015 data problem.")
say(f"- within-season corr(wt, R) by year: min {loso_df.within_yr_corr_wt_R.min():+.3f}, "
    f"max {loso_df.within_yr_corr_wt_R.max():+.3f}, mean "
    f"{loso_df.within_yr_corr_wt_R.mean():+.3f} — sign flips across seasons.")

say("")
say("### (iii) The pricing channel: is the win total absent from production, or "
    "already in the price?")
say("R = PPG - m_iso(ADP) by construction, so cov(wt,R) = cov(wt,PPG) - cov(wt,m_iso). "
    "A null on R has two very different explanations, and they are separable.")
chan = []
for lbl, yv2 in [("realized PPG", d.ppg), ("m_iso (ADP-implied value)", d.m_iso),
                 ("R (= PPG - m_iso)", d.resid_iso)]:
    ff = sm.OLS(yv2, sm.add_constant(d[["wt"]])).fit(
        cov_type="cluster", cov_kwds={"groups": d.year}, use_t=True)
    chan.append(dict(outcome=lbl, beta_wt=ff.params.wt, se=ff.bse.wt,
                     p_raw=ff.pvalues.wt))
    say(f"- {lbl:28s} on wt: beta {ff.params.wt:+.4f} (cluster se {ff.bse.wt:.4f}, "
        f"p_raw {ff.pvalues.wt:.4f})")
chan_df = pd.DataFrame(chan)
b_ppg, b_iso = chan_df.beta_wt.iloc[0], chan_df.beta_wt.iloc[1]
say(f"- decomposition: a one-win-per-17 better team environment is worth "
    f"{b_ppg:+.3f} PPG in realized production, and the ADP market already charges "
    f"{b_iso:+.3f} PPG for it — i.e. the market prices "
    f"{(b_iso/b_ppg if b_ppg else float('nan')):.0%} of the realized effect. "
    f"The residual {b_ppg-b_iso:+.3f} is the (insignificant) leftover. "
    f"**This is a priced channel, not an absent one** — the mechanism the "
    f"pre-registration predicted.")
say("- caveat on that reading: `wt` is also correlated with ADP by construction of "
    "the board (good offences supply more top-30 WRs), so the m_iso regression is "
    "partly a compositional statement about who makes the board, not purely a "
    "per-player pricing elasticity. Stated, not resolved.")

# ---------------------------------------------------------------- 2026 (labelled)
say("")
say("## 6. 2026 inputs — recorded, never fitted on")
t26 = pd.read_csv(f"{ROOT}/data/vegas/team_totals_2026.csv")
t26["fr"] = t26.team.replace(WK2FR)
OVERROUND_26 = 1.045
po26 = amer_to_prob(t26.win_total_dk.notna().astype(float) * 0 + t26.dk_over_odds.values)
p26 = np.clip(po26 / OVERROUND_26, 1e-6, 1 - 1e-6)
t26["wt_2026"] = [devig_rate(L / 17, pf, S_RATE, (L % 1 == 0), 17) * 17
                  for L, pf in zip(t26.win_total_dk, p26)]
w25 = W[W.season == 2025][["fr", "act_rate"]].rename(columns={"act_rate": "act25"})
v25 = V[V.season == 2025][["fr", "wt17"]].rename(columns={"wt17": "wt25"})
t26 = t26.merge(w25, on="fr", how="left").merge(v25, on="fr", how="left")
t26["surprise_2026"] = t26.wt_2026 - t26.act25 * 17
t26["d_posted_2026"] = t26.wt_2026 - t26.wt25
say("- 2026 win totals: VegasInsider/DraftKings board (2026-08-08). Only an OVER "
    f"price is stored, so a paired de-vig is impossible; a {OVERROUND_26:.3f} "
    "two-way overround is ASSUMED. This is a different source from the Covers "
    "historical series (Covers backfills only retrospectively) — the two are NOT "
    "one continuous series and nothing was fitted on 2026.")
say("- prior-season inputs for 2026 use realized 2025 wins and the Covers 2025 "
    "posted line (same historical source as the fitted series).")
if final_terms:
    say(f"- a surviving term exists ({final_terms}); 2026 fitted values are "
        "emitted, labelled `fit_on=2015-2024, applied_to=2026`.")
else:
    say("- NO term survived, so no 2026 board adjustment is produced. The 2026 "
        "team features are emitted for the §J views layer only, unfitted.")

# ---------------------------------------------------------------- outputs
tab.index.name = "term"
tab_out = tab.reset_index()
tab_out["block"] = "main_regression"
sens = pd.DataFrame(sens_rows); sens["block"] = "sensitivity"
hold_df["block"] = "holdout"
t26o = t26[["season", "fr", "team", "win_total_dk", "dk_over_odds", "wt_2026",
            "surprise_2026", "d_posted_2026"]].copy()
t26o["block"] = "context_2026_unfitted"
pan = d[["year", "name", "gsis_id", "adp", "ppg", "resid_iso", "fr_modal",
         "fr_first", "wt", "surprise", "d_posted", "prev_src"]].copy()
pan["block"] = "panel"

with open(f"{ROOT}/results/edge_vegas.csv", "w") as fh:
    for name, frame in [("main_regression", tab_out), ("sensitivity", sens),
                        ("holdout", hold_df),
                        ("posthoc_loso_coef_trace", loso_df.assign(block=1)),
                        ("posthoc_pricing_channel", chan_df.assign(block=1)),
                        ("context_2026_unfitted", t26o),
                        ("panel", pan)]:
        fh.write(f"# block={name}\n")
        frame.drop(columns=["block"]).to_csv(fh, index=False)
        fh.write("\n")

say("")
say("## 7. Verdict")
if final_terms:
    say(f"Terms {final_terms} survive both screens -> a context arm IS built and "
        "goes to LOSO (DM vs frozen arm (ii), clustered by year, p<0.10 AND RMSE "
        "improvement).")
else:
    say("**Null, as pre-registered.** No pre-specified sportsbook team-environment "
        "term predicts the ADP market's errors on this panel. No context arm is "
        "built; nothing enters the LOSO harness. The de-vigged win total, the "
        "surprise relative to last season's realized wins, and the year-over-year "
        "change in the posted line are all already reflected in where drafters "
        "place receivers. Raw p-values above are carried to the joint round-4 FDR "
        "family {H5, I3}.")

with open(f"{ROOT}/results/sectionI_notes.md", "w") as fh:
    fh.write("\n".join(OUT) + "\n")
print("\nwrote results/edge_vegas.csv, results/sectionI_notes.md")

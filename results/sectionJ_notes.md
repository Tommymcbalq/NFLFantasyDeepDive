# §J — Black–Litterman views overlay: machinery, estimated inputs, validation

Executed 2026-08-09 against the frozen v3 board (July ADP). Re-runs against the August board
once §G lands. Code: `scripts/19_bl_overlay.py`. Pre-registration: `EDA_PLAN4.md` §J.

## J1a — π, the market-implied prior

π is the §6.1 isotonic ADP→PPG curve evaluated at each board slot: price in, implied expected
value out. This is the reverse-engineering step the BL construction rests on, and we already had
it — it was fitted in round 1 for a different purpose.

Range on the 30-WR board: **π ∈ [12.54, 19.56] PPG**, i.e. the entire market-implied spread from
WR1 to WR30 is **7.0 PPG**. Note this for §J2 calibration: it means a stated view of 2 PPG is
worth roughly nine ADP slots at the top of the board and rather more in the flat tail. Views
should be authored against that scale, not against raw intuition about points.

## J1b — Σ diagonal

The residual spread of a *realised* season around the isotonic curve contains two things: genuine
uncertainty about θ, and the noise of estimating θ from ~17 games. BL wants the first only, so

    Σ_ii = τ²_iso(tier of i) − σ̂²_W,i / Ḡ_i,     floored at 0.25·τ²_iso

with τ²_iso the tier residual variance (`tier_variances.csv`, §6.1) and σ̂²_W,i the per-game
variance from §1. Resulting **σ_true ∈ [1.68, 2.88] PPG** per player.

Read against J1a: the per-player uncertainty is ~2.4 PPG against a 7.0 PPG total spread across
the whole board. The market's own ordering is, by its own implied uncertainty, weakly identified —
which is precisely the Merton/BL pathology that motivates the construction. Small view magnitudes
move a lot of rank.

## J1c — Σ off-diagonal: teammate block **estimated, and set to zero**

The share constraint predicts a *negative* correlation between same-team players' true values.
Estimated directly on the historical panel (2015–2024 top-30 WR boards, residuals from the
isotonic curve, demeaned within year to strip season-level shocks):

| quantity | value |
|---|---|
| same-team board pairs | 71, across all 10 years |
| same-team residual correlation | **r = +0.016** |
| cluster-bootstrap CI (on year) | **[−0.321, +0.320]** |
| random within-year pair null | mean −0.041, 95% band [−0.270, +0.202] |
| one-sided p (null ≤ observed) | 0.69 |

Indistinguishable from zero *and* from the null. Per the §J1 rule ("if a block cannot be estimated
it is set to zero and that choice is reported"), the block is zero.

**This is a power limitation, not a refutation.** A true ρ ≈ −0.3 — roughly what a binding
share constraint would imply — sits comfortably inside the CI. 71 pairs cannot separate it. The
finding is consistent with round 3's §F2 null (the market prices duo structure) but is not
evidence *for* independence. `TEAMMATE_RHO` in the script runs a declared non-zero block as a
sensitivity; it is 0.0 by default and any non-zero run must be labelled as a declared assumption,
never as an estimate.

## J2 — view specification

Rows of P: absolute (one weight of 1) or relative (weights summing to 0). Each view carries
q (magnitude, PPG), a confidence level, a rationale, and a date. Ω is **declared, never fitted**:
Ω_kk = (c · sd_prior(view k))², sd_prior(k) = √(p_k' τΣ p_k), with the scale fixed before any view
was written — low 2.0, medium 1.0, high 0.5, certain 1e-6. So "medium" means: this view is about
as uncertain as the prior is about the same quantity.

## J3 — posterior and attribution

    θ̄ = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹π + PᵀΩ⁻¹q]

The shift decomposes exactly by view, θ̄ − π = M PᵀΩ⁻¹(q − Pπ) with M = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹, so
each view's contribution to each player is a separate reported column. τ = 0.5 default, with the
board also emitted at τ ∈ {0.25, 1.0}.

## J4 — machinery validation (synthetic views only, run before any real view)

All eight asserted, all passing:

1. empty view set is an exact no-op;
2. a view stating exactly the prior is a no-op **at every confidence level**;
3. `certain` pins the posterior to q (|θ̄ − q| < 1e-3);
4. shift is strictly monotone in confidence (low < medium < high);
5. a relative view moves the pair in opposite directions, and is exactly zero-sum under equal
   prior variance;
6. with diagonal Σ, no view leaks to a non-viewed player — and with a −0.3 off-diagonal, a
   view on one teammate transmits *downward* to the other, in the direction Σ implies;
7. the per-view decomposition sums exactly to the total shift;
8. the posterior covariance is symmetric positive definite.

Tests 5/6 together are the ones that matter: they prove the overlay respects the covariance
structure rather than just shifting individual numbers, which is the whole reason for using the
BL form instead of an ad-hoc bump.

## Status

Machinery validated and π/Σ built; `results/board_2026_with_views.csv` currently carries the π and
σ_true columns with no views applied. Real views are pending — they get logged to
`results/views_2026.csv` with magnitude, confidence, rationale and date **before** the season, so
they can be scored afterwards. That pre-commitment is the mechanism that keeps the subjective
layer honest, and it is why this layer is kept strictly separate from the statistical board
rather than blended into the fit.

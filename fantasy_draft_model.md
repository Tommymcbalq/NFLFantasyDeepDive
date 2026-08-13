# A Behavioral Draft Simulator for a Competitive Fantasy League

*Base model specification — ADP shrinkage prior + a per-manager behavioral posterior, run forward as a Monte Carlo draft simulation.*

---

## The core idea

Model the draft as a **sequential discrete-choice simulation**. Every pick is one manager choosing one player out of the currently-available pool. If you can write down a good probability for a single pick — "given the board and this manager's tendencies, here's the distribution over who they take" — then you run that forward, pick by pick, thousands of times. The thing you actually want ("who comes back to me at 2.06") is not a formula; it's a **survival curve** that falls out of the Monte Carlo.

ADP is not the model. ADP is the **anchor you shrink toward** when you don't have enough signal about a specific manager. The behavioral layer is a set of small, interpretable *deviations* from that anchor.

---

## One pick = a softmax over utility

For manager $m$ picking at time $t$, with available set $A_t$, give each player $j$ a utility and pick proportional to its exponential:

$$
P(\text{pick}=j \mid A_t, \text{roster}_{m,t}) = \frac{\exp(U_{mjt}/\tau_m)}{\sum_{k\in A_t}\exp(U_{mkt}/\tau_m)}
$$

$$
U_{mjt} = \underbrace{\beta^{\text{ADP}}_m \, v_j}_{\text{consensus value}} \;+\; \underbrace{\sum_p \beta^{\text{pos}}_{m,p}\, \text{need}_{m,p,t}\,\mathbb{1}[\text{pos}(j)=p]}_{\text{roster construction}} \;+\; \underbrace{\gamma_m\, a_{m,j}}_{\text{"their guys"}} \;+\; \underbrace{\delta^\top (\text{meta}_t \otimes \text{profile}_j)}_{\text{meta}\times\text{player interaction}} \;+\; \underbrace{\rho^\top r_{m,j,t}}_{\text{redundancy}}
$$

### Term by term

**$v_j$ — the global prior.** Convert ADP into a latent "consensus value," not raw rank. Raw rank is linear; real boards have **tier cliffs**. Fit a monotone value curve (or use projected points-above-replacement) so the drop from "top-3 RB tier" to "next tier" is a real gap. This is what makes RBs "fly off" — once the tier empties, the need term spikes for everyone at once and you get a run.

**$\tau_m$ — the manager temperature.** The single most important behavioral dial. Low $\tau$ = chalky, drafts almost exactly their internal ranking; high $\tau$ = noisy, reachy, disrupts ADP. This one scalar captures "who stays on meta vs who reaches."

**$\beta^{\text{pos}}_{m,p} \cdot \text{need}$ — roster construction, state-dependent.** `need` is a function of the manager's *current* roster (0 RBs rostered in round 2 → huge RB need). The hard league rule ("must take an RB in the first two rounds") is this need term going to $+\infty$ when the constraint is about to bind — mechanically, mask non-RBs (or add a dominating bonus) on that manager's round-2 pick if they're still RB-empty.

**$\gamma_m a_{m,j}$ — "their guys."** Affinity. This is where the polarization lives (see below).

**meta ⊗ profile — the transferable part.** You *cannot* learn "manager X likes player Y" across years because the player pool changes every season. What transfers is the interaction with player **features**: age, breakout-ADP-delta (is the market moving up on him?), "revered/hyped" score, projected variance/upside. So you learn "manager X reaches on young high-upside guys in a WR-thin meta," which generalizes to next year's crop.

**redundancy $\rho^\top r$ — substitution structure.** The "wouldn't take Brock if you took Genty" rule. A penalty on player $j$ that grows with roster overlap/redundancy (same position already filled, correlated bye, or a hand-coded "these two don't coexist" adjacency). Roster-state-dependent, so it only fires after the conflicting piece is on the roster.

---

## The shrinkage: hierarchical priors

You have ~9 friends and only a handful of past drafts each. You cannot freely estimate rich per-manager parameters — you'll overfit noise. So **partial pooling**:

$$
\beta_m \sim \mathcal{N}(\beta_{\text{league}}, \Sigma), \qquad \tau_m \sim \text{LogNormal}(\mu_\tau, \sigma_\tau)
$$

Managers with lots of consistent history get their own estimates; sparse/erratic managers get pulled toward the league mean. And $\beta_{\text{league}}^{\text{ADP}}$ is anchored so that the league-average behavior reproduces the ADP ordering. **That is the "shrink to ADP" mechanism** — it's the hyperprior, not a separate step.

Keep each manager down to a few interpretable dials: ADP-weight, temperature, one or two position leans, a reach loading, and a short "guys" list. More than that and you're estimating hallucinations.

---

## The polarization ("definitely will or definitely won't")

For a manager's target, the pick isn't a smooth probability — it's bimodal. If their guy is on the board they take him near-certainly; otherwise it's like the player doesn't exist to them. Model affinity as **spike-and-slab**:

$$
a_{m,j} = z_{m,j}\cdot s_{m,j}, \qquad z_{m,j}\sim \text{Bernoulli}(\pi_m)
$$

$z=1$ means "this is one of manager $m$'s guys" (large positive slab); $z=0$ means ordinary. Conditional on $z=1$ and availability, the softmax collapses to "take him." That reproduces the polarized behavior — and your side-knowledge ("he's been hyping a WR at 1.10") enters cleanly as a **prior** on $z_{m,j}=1$ for this year's draft, not a hack.

---

## Fitting it

Each historical pick is one choice observation over the set that was actually available at that moment — a **conditional logit** likelihood (the same math as McFadden discrete choice / a Plackett–Luce ranking). Reconstruct each past draft board pick-by-pick, compute the available set and roster states at each step, and fit the whole hierarchy with MCMC (Stan or PyMC). Small data + strong priors is the right regime here; you want the *posterior over parameters*, not point estimates, because that uncertainty is half the value.

---

## Running it forward — the part you actually want

Once fitted, to answer "who's coming back to me at 2.06":

1. Draw a parameter set from the posterior (propagates your uncertainty about how each friend drafts).
2. Simulate every pick between now and your next one using the softmax above, updating rosters/needs as you go.
3. Record, for each player, whether they survived to your pick.
4. Repeat ~5–10k times.

Outputs:

- **Survival probability** $P(\text{available at pick } k)$ per player — a smooth curve down the board.
- **Positional run risk** — e.g. $P(\ge 3 \text{ RBs go before } 2.06)$.
- **Value Over Next Available (VONA)** — for each position, compare the best player likely there *now* vs the expected best player still there at your *next* pick. If the RB survival curve falls off a cliff before your 3rd-round pick but WR is flat, take the RB now and wait on WR. This quantifies the "I already took an anchor RB in round 1, so what's the dynamic" reasoning.

---

## Two honest cautions

**The binding constraint is data, not math.** A handful of drafts per friend means you should resist making the per-manager model fancy. Most of your edge comes from (a) the tier-cliff value curve, (b) manager temperature, and (c) your hand-set "guys" priors for this specific year.

**Validate by calibration, not accuracy.** Do leave-one-draft-out: refit without one past draft, predict it, and check that your survival curves are *calibrated* (things you said were 70% to be gone were gone ~70% of the time). Calibration is what makes the 2.06 answer trustworthy.

---

## Symbol reference

| Symbol | Meaning |
|---|---|
| $A_t$ | Players still available at pick $t$ |
| $U_{mjt}$ | Utility of player $j$ to manager $m$ at pick $t$ |
| $\tau_m$ | Manager temperature (low = chalky, high = reachy) |
| $v_j$ | Consensus value from ADP (tiered, not linear) |
| $\text{need}_{m,p,t}$ | Positional need given manager's current roster |
| $a_{m,j}$ | Affinity — "their guys" (spike-and-slab) |
| $z_{m,j}$ | Latent indicator that $j$ is one of $m$'s targets |
| $\beta_{\text{league}}$ | League-mean coefficients (shrinkage target ≈ ADP) |

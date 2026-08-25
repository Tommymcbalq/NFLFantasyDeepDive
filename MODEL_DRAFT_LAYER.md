# §50 — The owner constraint layer and the draft simulation

*Belongs in REPORT.md as §50; kept standalone for reading during the draft.*

---

## 50.1 Notation

| symbol | meaning |
|---|---|
| $\mathcal{P}$ | the player universe on the board; a player is indexed $j$ |
| $\mathcal{K}$ | the owner's own picks, $\mathcal{K}=\{5,16,27,34,47,54,65,74,\dots\}$; a pick is $k$ |
| $S_j\in\mathbb{R}$ | the **slot score** of player $j$ — a latent continuous quantity whose *rank* decides when he comes off the board |
| $R_j$ | the **realised draft position** of $j$: the rank of $S_j$ among $\{S_1,\dots,S_{|\mathcal P|}\}$ |
| $A_j(k)$ | **availability**: $A_j(k)=\Pr(R_j\ge k)$, the chance $j$ is still there at pick $k$ |
| $\hat\pi_j$ | the owner's **declared** availability for $j$ at his pick $k_j$ |
| $c_j$ | confidence class of that declaration, $c_j\in\{\text{high},\text{med},\text{low}\}$ |
| $\sigma(c)$ | dispersion assigned to class $c$: $1.6,\;3.0,\;5.5$ |
| $\mu_j,\ \sigma_j$ | location and dispersion of $j$'s slot distribution |
| $e_j$ | the **room-order position** of $j$ — from the owner's mock board for picks $\le 39$, and the FFC/ESPN average ADP beyond it |
| $\Phi$ | the standard normal CDF |
| $T(j),\ o(j)$ | the owner's **tier** for $j$ and his order *within* that tier |
| $\eta$ | calibration step size, $\eta=2.2$ |

---

## 50.2 The idea in plain language

A draft is a ranking, not a set of independent events. Saying "Saquon is 70% to reach pick 16"
is a statement about **where he falls relative to everyone else**, so a model of it has to be a
model of the whole ordering — you cannot just attach a probability to Saquon and leave the rest
of the board alone, because whoever *doesn't* get taken in front of him has to be taken
somewhere, and that displaces other players.

So each player gets a latent score $S_j$. Sorting the scores gives a draft. Draw many times and
count how often each player survives to each of the owner's picks. That is the simulation.

The owner's beliefs enter as constraints on $S_j$. Two things have to be true of them:

1. **A declared number must come out the far end unchanged.** If the owner says 70%, the
   simulation must report 70%, not 61%.
2. **Confidence must do real work.** A belief held firmly should override the ADP ordering; a
   belief held loosely should be pulled back toward it. That is shrinkage, and it is the same
   logic as the empirical-Bayes layer in §3: *confidence is precision*, and precision is what
   decides how far the posterior moves.

---

## 50.3 The model

Each player's slot score is normal and independent of the others:

$$S_j \sim \mathcal{N}(\mu_j,\ \sigma_j^2), \qquad R_j = \operatorname{rank}\big(S_j;\ \{S_i\}_{i\in\mathcal P}\big).$$

**Default parameters** (no owner belief about $j$):

$$\mu_j = e_j, \qquad \sigma_j = \begin{cases}1.40\sqrt{\pi/2}\approx 1.75 & e_j \le 39 \quad\text{(inside the observed mock)}\\[2pt] 9.0 & e_j > 39 \quad\text{(ADP only)}\end{cases}$$

The two regimes are the point. Inside the mock the owner has *seen* the ordering, so dispersion
is small and set from his own stated mean absolute error of 1.4 picks — for a half-normal,
$\mathbb{E}|X|=\sigma\sqrt{2/\pi}$, so $\mathbb{E}|X|=1.4 \Rightarrow \sigma=1.4\sqrt{\pi/2}$.
Beyond the mock there is no observation, only national ADP from a room that is sharper than
national ADP, so dispersion is large.

**Owner-declared parameters.** For a player with a declaration $(\hat\pi_j, c_j)$ at pick $k_j$:

$$\sigma_j = \sigma(c_j), \qquad \mu_j \ \text{chosen so that}\ A_j(k_j)=\hat\pi_j .$$

This is where confidence becomes shrinkage. With $\sigma_j$ small, $S_j$ is tightly concentrated
about $\mu_j$, so $j$'s position is decided almost entirely by the declaration and barely at all
by the common noise — the owner's number is taken literally. With $\sigma_j$ large, $j$'s
realised rank is dominated by noise shared with everyone else, and he drifts back toward where
the ADP ordering would have put him. **Low confidence shrinks the belief toward the market;
high confidence does not.**

---

## 50.4 Why $\mu_j$ has to be calibrated rather than solved

The obvious move is to invert the normal directly. Requiring $\Pr(S_j \ge k)=\hat\pi_j$:

$$\Pr(S_j\ge k)=1-\Phi\!\left(\frac{k-\mu_j}{\sigma_j}\right)=\hat\pi_j
\;\Longrightarrow\;
\frac{k-\mu_j}{\sigma_j}=\Phi^{-1}(1-\hat\pi_j)
\;\Longrightarrow\;
\boxed{\ \mu_j = k-\sigma_j\,\Phi^{-1}(1-\hat\pi_j)\ }$$

Sanity check: at $\hat\pi_j=\tfrac12$, $\Phi^{-1}(\tfrac12)=0$ and $\mu_j=k$ — a player expected
exactly at your pick is a coin flip. At $\hat\pi_j=0.7$, $\Phi^{-1}(0.3)=-0.524$, so
$\mu_j = k+0.524\sigma_j$ — more available means drafted later. Both correct.

**But this solves the wrong equation.** Availability is not $\Pr(S_j\ge k)$; it is
$\Pr(R_j\ge k)$, and $R_j$ is a rank among $|\mathcal P|$ *simultaneous* draws. The map from the
marginal distribution of $S_j$ to the distribution of its rank depends on the parameters of
**every other player**. There is no closed form, and the discrepancy is not small — it is
exactly the displacement effect described in §50.2.

So the closed form is used only as a starting point, and $\mu_j$ is then driven to the target by
a fixed point. Writing $\hat A^{(t)}_j(k)$ for the Monte Carlo availability at iteration $t$:

$$\mu_j^{(t+1)} \;=\; \mu_j^{(t)} \;-\; \eta\left(\hat A^{(t)}_j(k)-\hat\pi_j\right)$$

iterated until $\max_j\big|\hat A_j(k)-\hat\pi_j\big| < 0.012$.

**The sign is the whole content of that line.** If simulated availability *exceeds* the declared
value, the player is surviving too often, so he must be made to go **earlier**, which means
$\mu_j$ must **decrease**. Getting this backwards makes the iteration diverge, and it diverges
to the flattering answer — every declared player reads 100% available — which is why it has to
be stated explicitly rather than left to inspection.

---

## 50.5 Hard zeros

A hard zero is a player the owner states will never reach him. The naive implementation sets
$\mu_j=1$: taken first overall, hence never available.

**That is wrong, and wrong in a way that corrupts the whole board.** A hard zero means "gone
before *my* pick", not "gone before *everyone's*". Pinning six players to slot 1 piles them at
the front of the ordering and displaces every other player about six picks later — in the build
where this was live it reported Jonathan Taylor 76% to reach pick 16, which is absurd for a
player the room takes eighth.

The correct operation is narrower. Every hard zero already sits *ahead* of the relevant pick in
$e_j$; the only reason the simulation ever hands him to the owner is the upper tail of $S_j$. So
keep $\mu_j=e_j$ and shrink the tail:

$$\mu_j = e_j, \qquad \sigma_j = 0.5 .$$

The player stays in his natural slot, displaces nobody, and his survival probability goes to
zero because the dispersion that produced it is gone.

This also resolves an apparent contradiction in the owner's own statements. He said "Henry is
0", and separately that Henry goes at 2.10/3.01 — picks 20–21. Both are true: Henry is $0\%$ at
pick **27** and $100\%$ at pick **16**. A blanket hard zero would have destroyed a guaranteed
fallback.

---

## 50.6 The strategy layer on top

Availability answers *who is there*. The owner's tiers answer *who to take*. Each RB and WR
carries an owner tier $T(j)$ and a within-tier order $o(j)$; the pick rule is lexicographic:

$$j^\star(k) \;=\; \arg\min_{j\ \text{available at}\ k}\ \big(T(j),\ o(j)\big).$$

The quantity that actually drives strategy is **tier exhaustion** — the pick at which the last
member of a tier is taken:

$$X_T \;=\; \max\{\,R_j \;:\; T(j)=T\,\}.$$

This matters because of the flat-vs-step result already derived in §V: within a tier the owner
has declared the players indistinguishable, so choosing among them is worth nothing, while the
gap *between* tiers is worth a step. Value therefore accrues at the moment a tier empties, not
continuously down the board. Concretely: RB2 exhausts at pick 16 and WR2 at 17, so pick 16 is a
choice between the last member of each — and no model can break a tie the owner has declared to
be a tie.

---

## 50.7 What this replaces, and what it does not do

It replaces per-player Gaussian noise around ADP, which encodes "player at ADP 19 goes at 19."
That was wrong for the reason the owner gave: Kenneth Walker does not reach pick 16 because a
specific manager wants him *and* because that roster will not double up at RB — an affinity term
and a state-dependent need term, neither of which is noise.

**It does not model those terms.** The full specification — conditional logit over the available
set, per-manager temperature, spike-and-slab affinity, roster-need — is in
`fantasy_draft_model.md` and is still unbuilt. It was blocked on draft logs needed to *fit* the
per-manager parameters. That blocker is now gone for a different reason: the owner **declares**
the availabilities the fitted model would have produced. This layer consumes the output of that
model without containing it, which is honest but strictly weaker — it can only speak about
players the owner has an opinion on, and it says nothing about *why* a player falls.

Beyond pick 34 the owner has declared nothing, so the board there runs on the FFC/ESPN average
with $\sigma=9$. That is the weakest region and it is where a real behavioural model would earn
its keep.

---

## 50.8 Calibration achieved

Sixteen declarations, all reproduced within 3 points; convergence in ≤40 passes.

| player | pick | declared | delivered |
|---|---|---|---|
| Christian McCaffrey | 5 | 80% | 80% |
| Puka Nacua | 5 | 20% | 20% |
| Ja'Marr Chase | 5 | 0% | 1% |
| Justin Jefferson | 16 | 100% | 99% |
| Saquon Barkley | 16 | 70% | 70% |
| Ashton Jeanty | 16 | 25% | 25% |
| De'Von Achane | 16 | 15% | 14% |
| CeeDee Lamb | 16 | 10% | 10% |
| Rashee Rice | 27 | 70% | 71% |
| Malik Nabers | 27 | 20% | 20% |
| Chris Olave | 27 | 10% | 10% |
| Nico Collins | 27 | 2% | 3% |
| Travis Etienne Jr. | 34 | 72% | 73% |
| Kyren Williams | 34 | 80% | 80% |
| Garrett Wilson | 34 | 70% | 69% |
| Luther Burden III | 34 | 70% | 70% |

$\hat\pi$ is clamped to $[0.01,0.99]$ because $\Phi^{-1}$ is infinite at the endpoints; a
declared 0 or 100 is therefore delivered as 1% or 99%, which is within the Monte Carlo error of
the statement anyway.

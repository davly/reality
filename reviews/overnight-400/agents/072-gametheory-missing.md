# 072 | gametheory-missing

**Scope:** missing-primitive enumeration for `C:\limitless\foundation\reality\gametheory\` (Nash, equilibria, learning, mechanism design, cooperative, matching, voting, repeated games).

**Files surveyed:** `nash.go` (232 LOC), `voting.go` (291 LOC), `matching.go` (170 LOC), `bandit.go` (221 LOC), `kelly.go` (136 LOC). Test file 932 LOC / 66 funcs. Golden files: `nash_2x2.json`, `gale_shapley.json` only.

**071 finding inherited:** package is a 1050-LOC sliver — only 2x2 closed-form Nash, fictitious-play minimax, Gale-Shapley, exact-Banzhaf-up-to-n=25, exact-Shapley-up-to-n=12-then-MC, UCB1/Thompson/EpsilonGreedy bandits, Kelly. Topic ask is therefore ~95% missing-feature enumeration, ~5% extension of present primitives.

## TL;DR

`gametheory/` is missing **the entire equilibrium-computation toolkit** (Lemke-Howson, support enumeration, correlated equilibrium LP, ε-Nash, Stackelberg, Bayesian Nash, QRE, sequential, trembling-hand, subgame-perfect, extensive-form solvers), **the entire mechanism-design layer** (VCG, Myerson, Vickrey, sealed/English/Dutch auctions, AGV, Gibbard-Satterthwaite witnesses), **the entire poker-grade learning stack** (CFR/CFR+/MCCFR/Linear-CFR/Discounted-CFR, no-regret/Hedge/MWU, regret matching, Online Mirror Descent), **most cooperative-game primitives** (core, nucleolus, Owen, Banzhaf-Coleman, all bargaining solutions, Rubinstein, Solomon, kernel, τ-value, prekernel), **most matching markets** (top trading cycles, house allocation, hospital-residents, school choice, Roth-Sotomayor many-to-many, kidney exchange), **most voting/social-choice rules** (Borda, Condorcet test, Kemeny-Young, STV, Schulze, Copeland, majority judgement, Arrow witness), and **the entire repeated-games layer** (folk theorem witnesses, tit-for-tat strategies, Axelrod tournament, discounted-NPV trigger strategies). Replicator dynamics is absent (071 §F flagged this for `chaos`-side coupling). The package shares no code with `optim` (no LP/simplex import despite topic mentioning "linear-programming form" of correlated equilibrium and minimax) — every additive primitive below either composes `optim/simplex.go` or composes nothing. Total Tier 1 estimate ~1,800 LOC of pure additions across 12 primitives, 95%+ of which need only stdlib + optim/simplex.

---

## Inventory: present vs. missing (master plan crosswalk)

| Topic bullet | Status | File:Line |
|---|---|---|
| Pure Nash 2x2 | present | `nash.go:45-115` |
| Pure Nash NxM (general) | **absent** | — |
| Mixed Nash via Lemke-Howson | **absent** | — |
| Mixed Nash via support enumeration | **absent** | — |
| Correlated equilibrium (LP) | **absent** | — |
| ε-Nash detector | **absent** | — |
| ε-correlated equilibrium | **absent** | — |
| Trembling-hand equilibrium | **absent** | — |
| Sequential equilibrium | **absent** | — |
| Subgame-perfect (extensive form) | **absent** — no extensive-form data structure exists | — |
| Stackelberg leader-follower | **absent** | — |
| Bayesian Nash | **absent** | — |
| Quantal response equilibrium (logit-QRE) | **absent** | — |
| Lemke-Howson pivot | **absent** | — |
| Govindan-Wilson global Newton | **absent** (out of Tier-1 scope) | — |
| McKelvey-McLennan-Turocy | **absent** (out of Tier-1 scope) | — |
| Fictitious play (single iter) | partial (only embedded inside `Minimax`) | `nash.go:170-232` |
| Smooth fictitious play | **absent** | — |
| Best-response dynamics | **absent** as standalone | — |
| Replicator dynamics | **absent** (also flagged by 071) | — |
| No-regret learning (Hedge / MWU) | **absent** | — |
| Online convex optimization | **absent** (could live in `optim/`) | — |
| Counterfactual Regret Minimization (CFR) | **absent** | — |
| CFR+ | **absent** | — |
| MCCFR (outcome / external sampling) | **absent** | — |
| Discounted CFR | **absent** | — |
| Linear CFR | **absent** | — |
| VCG mechanism | **absent** | — |
| Myerson optimal auction | **absent** | — |
| Vickrey second-price | **absent** | — |
| English / Dutch / sealed-bid first-price | **absent** | — |
| AGV (d'Aspremont-Gérard-Varet) | **absent** | — |
| Combinatorial auctions | **absent** | — |
| Maskin monotonicity tester | **absent** | — |
| Gibbard-Satterthwaite witness | **absent** | — |
| Shapley value (cooperative) | present (exact ≤12, MC > 12) | `voting.go:119-225` |
| Shapley-Shubik power index | present | `voting.go:252-291` |
| Banzhaf | present (n ≤ ~25) | `voting.go:36-92` |
| Banzhaf-Coleman (raw, swing-only, P-power) | **absent** as separate primitives | — |
| Core (LP feasibility) | **absent** | — |
| Nucleolus (sequential LP) | **absent** | — |
| Owen value | **absent** | — |
| Bargaining: Nash / Kalai-Smorodinsky / egalitarian | **absent** | — |
| Solomon division (proportional cake-cutting witness) | **absent** | — |
| Rubinstein bargaining | **absent** | — |
| Gale-Shapley | present | `matching.go:36-106` |
| Top trading cycles | **absent** | — |
| Roth-Sotomayor many-to-many | **absent** | — |
| House allocation / serial dictatorship | **absent** | — |
| Hospital-residents (capacity GS) | **absent** | — |
| Plurality | **absent** | — |
| Borda | **absent** | — |
| Condorcet winner / paradox detector | **absent** | — |
| Kemeny-Young | **absent** | — |
| Single transferable vote (STV) | **absent** | — |
| Schulze | **absent** | — |
| Majority judgement | **absent** | — |
| Copeland | **absent** | — |
| Arrow's theorem witness | **absent** | — |
| Folk theorem witnesses | **absent** | — |
| Tit-for-tat / win-stay-lose-shift | **absent** | — |
| Discounted infinite repetition NPV | **absent** | — |

---

## Tier 1 — must-ship (blocks RubberDuck/Paradox/LiquidVote/Tempo claims, ~1,800 LOC, all golden-file-testable)

**T1-1. Pure-Nash NxM enumeration** (~80 LOC). Generalise the 2x2 best-response cell scan to general bimatrix games. Inputs: `payoffA, payoffB [][]float64`, dims `(m, n)`. Output: `[]struct{Row, Col int}` of pure-strategy NE cells (possibly empty or multiple). Comment on equilibrium-set degeneracy. Required precision: exact for rational payoffs.

**T1-2. Support enumeration mixed Nash** (~250 LOC). For each pair of supports (R ⊂ rows, C ⊂ cols, |R|=|C|), solve the indifference linear system; check primal/dual feasibility (mixed strategy in simplex, no profitable deviation outside support). Reference: Porter-Nudelman-Shoham (2004); McKelvey-McLennan (1996, *Handbook of Computational Economics*). For an `m × n` bimatrix this is `O(2^{m+n})` worst-case but tractable for `m, n ≤ 6` (Pistachio's puzzle-design size). Returns *all* mixed Nash equilibria.

**T1-3. Lemke-Howson pivot** (~300 LOC). Lexicographic pivoting on the labeled bimatrix LCP. References: Lemke-Howson (1964); von Stengel chapter in *Handbook of Game Theory* vol. 3 (2002, ed. Aumann-Hart). Output: one mixed Nash equilibrium and its support pair. Implement lexicographic perturbation for cycling-prevention. Returns single NE; can be re-seeded with different dropped labels to enumerate multiple.

**T1-4. Correlated equilibrium via LP** (~150 LOC, composes `optim/simplex.go`). The set of correlated equilibria is a polytope defined by linear incentive-compatibility constraints. Outputs the *social-welfare-maximising* CE, the *uniformly-distributed* CE, or any objective-linear extreme. Reference: Aumann (1974); Hart-Mas-Colell (2000). Tightly integrates with no-regret learning convergence (T1-7).

**T1-5. Stackelberg leader-follower** (~120 LOC). Bilevel optimisation: leader picks `x` to max `u_L(x, BR(x))` where `BR(x) = argmax_y u_F(x, y)`. For finite-game form: brute-force enumeration of leader strategies + best-response computation per. For continuous: bisection / discretization. Reference: von Stackelberg (1934); Conitzer-Sandholm (2006, "Computing the optimal strategy to commit to").

**T1-6. ε-Nash and ε-correlated detectors** (~80 LOC). Given a strategy profile, compute the *exploitability* `ε(σ) = max_i max_{σ'_i} u_i(σ'_i, σ_{-i}) - u_i(σ)`. This is the primary CFR/learning-convergence diagnostic. Trivial to implement once T1-1/T1-2 are present. Required by every poker-grade benchmark in OpenSpiel.

**T1-7. Regret-matching + Hedge / MWU + smooth fictitious play** (~150 LOC). The three canonical no-regret algorithms. Online interface: `Update(payoffVector []float64) -> action int`. Hart-Mas-Colell regret matching → CCE convergence; multiplicative weights (Freund-Schapire 1997, Arora-Hazan-Kale 2012) → ε-Nash in zero-sum at rate O(√(log n / T)). Smooth FP per Fudenberg-Levine (1995). Replaces the embedded fictitious play in `Minimax` (071 §F1) with a public, configurable iterator that reports regret bounds.

**T1-8. CFR + CFR+ + MCCFR (outcome sampling)** (~400 LOC). Counterfactual Regret Minimization (Zinkevich-Johanson-Bowling-Piccione 2007), CFR+ (Tammelin 2014, *Cepheus*), MCCFR (Lanctot-Waugh-Zinkevich-Bowling 2009). Operates on extensive-form games via an information-set tree. Required for Pluribus/Libratus-class poker AI. **This in turn requires** an `ExtensiveFormGame` data structure: nodes = decision/chance/terminal, information sets, action sets, transition function. ~150 LOC for the data structure + ~250 LOC for the three solvers; ε-Nash exploitability (T1-6) is the pinned convergence test. Ship Discounted-CFR and Linear-CFR (Brown-Sandholm 2019) as one-line variants of CFR+ — same regret update with a discount factor.

**T1-9. VCG + Vickrey + first-price sealed-bid + English + Dutch** (~250 LOC). The mechanism-design starter kit. VCG (Vickrey-Clarke-Groves 1961-71-73): allocate to maximise reported social welfare, charge each winner the externality they impose on others. Vickrey: one-item, second-price. First-price sealed-bid: trivial allocation, no truthful BNE. English/Dutch: ascending/descending price-clock simulators with bidder dropout. All take a `[]Bidder{Valuation, Strategy}` input, return `Allocation` and `Payments`. Dominant-strategy truthfulness is the pinned property for VCG/Vickrey.

**T1-10. Bargaining solutions: Nash, Kalai-Smorodinsky, egalitarian** (~150 LOC). Inputs: bargaining set `S ⊂ R^n` (convex hull of utility vectors) + disagreement point `d`. Nash bargaining: argmax `Π (u_i - d_i)`. KS: maximises `min (u_i - d_i) / (u^*_i - d_i)`. Egalitarian (Kalai 1977): equalises gains over `d`. References: Nash (1950); Kalai-Smorodinsky (1975); Kalai (1977). All three are convex-optimisation problems on the bargaining polytope — compose `optim/`.

**T1-11. Top trading cycles + serial dictatorship + hospital-residents (capacity GS)** (~200 LOC). The three canonical matching primitives the topic flags. TTC (Shapley-Scarf 1974, Roth-Postlewaite 1977): cycle-finding on the "points-to-favourite" graph. Serial dictatorship: priority list + first-available. Hospital-residents: GS extension with capacities (Roth 1984, *NRMP*). Roth-Sotomayor (1990) is the cooperative-game-theoretic frame for both.

**T1-12. Voting rules: Plurality, Borda, Condorcet test, Kemeny-Young, Schulze, Copeland, STV, majority judgement** (~300 LOC). Eight rules sharing a common `Ballot` interface. Condorcet and Kemeny-Young are the heaviest (O(n!) Kemeny-Young is NP-hard but tractable to ~10 candidates via integer programming; Schulze is O(n³) Floyd-Warshall on the pairwise-defeat graph). Arrow's-impossibility witness (a 3-voter 3-candidate cyclic-preference profile demonstrating IIA-failure for any non-dictatorial rule) ships as a separate ~30 LOC test/example.

**T1 totals: ~12 primitives, ~2,250 LOC. Golden coverage required: ≥20 vectors per primitive ≈ 240 vectors (today's package has ~40 across 2 primitives).**

---

## Tier 2 — should-ship (closes 95% of nashpy/Gambit feature parity, ~1,500 LOC)

**T2-1. Replicator dynamics** (~120 LOC). `ẋ_i = x_i (f_i(x) - φ(x))` where `f_i(x) = (Ax)_i`, `φ(x) = x^T A x`. ODE integrator (compose `chaos/ode.go` RK4). Fixed points = symmetric Nash equilibria. Lyapunov-style ESS test on Hessian of replicator field. Reference: Taylor-Jonker (1978); Hofbauer-Sigmund *Evolutionary Games and Population Dynamics* (1998).

**T2-2. Best-response dynamics + better-response dynamics** (~80 LOC). Discrete-time iteration: each player switches to a strict best-response. Convergence detector. Counter-example registry (Shapley fictitious-play cycling on 3x3). Reference: Hofbauer-Sandholm (2002).

**T2-3. Quantal response equilibrium (logit-QRE)** (~120 LOC). Bounded-rationality NE with softmax best-responses: `σ_i(a) ∝ exp(λ · u_i(a, σ_{-i}))`. As `λ → ∞` collapses to NE; `λ = 0` is uniform. Compute via fixed-point iteration. Reference: McKelvey-Palfrey (1995). The principal experimental-game-theory tool.

**T2-4. Bayesian Nash equilibrium (finite-type)** (~150 LOC). Type spaces with common prior. Strategies are functions from types to actions. Compute via support enumeration in the *expanded* (type × action) bimatrix. Reference: Harsanyi (1967-68); Myerson *Game Theory* (1991).

**T2-5. Trembling-hand perfection** (~100 LOC). Selten (1975). Iterative refinement: at each ε > 0, compute NE of perturbed game; take ε → 0 limit. Filters out dominated strategies that survive in standard NE.

**T2-6. Subgame-perfect equilibrium (extensive form)** (~120 LOC). Backward induction on the extensive-form-game tree (T1-8 substrate). For perfect-information games, returns the unique SPE. For imperfect-information, requires SE/PBE.

**T2-7. Core (LP feasibility) + Nucleolus (sequential LP)** (~250 LOC). Core: solve `Ax ≥ b` system where rows are coalitional rationality constraints; emptiness ⇔ no core. Returns extreme points of the core polytope. Nucleolus (Schmeidler 1969): lexicographically minimise the maximum coalition excess via sequential LP. Reference: Maschler-Solan-Zamir *Game Theory* (2013, Cambridge).

**T2-8. Owen value + Aumann-Drèze + τ-value** (~200 LOC). Owen (1977) coalition-structure-aware Shapley extension. Three additional cooperative-game solution concepts widely used in cost-allocation / power-index applications.

**T2-9. Rubinstein bargaining (alternating offers, discount factors)** (~80 LOC). Closed-form unique SPE: `x* = (1 - δ_2) / (1 - δ_1 δ_2)` for two-player. Multi-player extension via Krishna-Serrano (1996).

**T2-10. Tit-for-tat + win-stay-lose-shift + Pavlov + Axelrod tournament** (~150 LOC). The canonical repeated-game strategies (Axelrod 1984; Nowak-Sigmund 1993). Tournament harness simulating a round-robin between strategies; reports per-strategy total payoff and pairwise win rates.

**T2-11. Discounted infinite repetition NPV + folk-theorem witness** (~80 LOC). Compute NPV for trigger strategies under discount factor δ. Folk theorem witness: given a feasible-payoff vector v ≥ minimax for both players, exhibit the trigger strategy and δ-threshold that supports it. Reference: Friedman (1971); Fudenberg-Maskin (1986).

**T2-12. Discounted-CFR + Linear-CFR variants** (already counted in T1-8 if those ship together, otherwise add ~50 LOC each).

**T2 totals: ~12 primitives, ~1,500 LOC.**

---

## Tier 3 — research-grade / specialist (deferred until consumer demand, ~2,500 LOC)

- **T3-1. Govindan-Wilson global Newton** (~400 LOC). Path-following on logit-QRE perturbation as λ varies. Computes "all" mixed NE up to known multiplicity. Reference: Govindan-Wilson (2003, *Econometrica*).
- **T3-2. McKelvey-McLennan-Turocy enumeration via polynomial homotopy** (~600 LOC). Solves Nash-as-polynomial-system (Bernstein bound on number of NE). The Gambit `gambit-enummixed` engine. Tier 3 because polynomial-system-solving needs Tier 1 of a missing `polynomial/` package.
- **T3-3. Sequential equilibrium (Kreps-Wilson 1982)** (~250 LOC). Refinement of PBE for extensive-form games with imperfect information. Belief consistency via trembles.
- **T3-4. Myerson optimal auction (regular distributions)** (~200 LOC). Virtual-value transformation, ironed virtual values, optimal reserve computation. Reference: Myerson (1981, *Mathematics of Operations Research*).
- **T3-5. AGV mechanism (Arrow-d'Aspremont-Gérard-Varet)** (~120 LOC). Budget-balanced ex-post-efficient mechanism alternative to VCG. Bayesian incentive compatibility only.
- **T3-6. Combinatorial auctions: VCG-on-bundles + Walrasian + iterative-Vickrey** (~400 LOC). NP-hard winner determination → branch-and-bound or LP relaxation. Reference: Cramton-Shoham-Steinberg *Combinatorial Auctions* (2006).
- **T3-7. Maskin monotonicity tester + Gibbard-Satterthwaite witness** (~150 LOC). Given a social choice rule (T1-12), test Maskin monotonicity (Maskin 1999). Construct manipulation example for any non-dictatorial onto rule with ≥3 alternatives (Gibbard 1973, Satterthwaite 1975).
- **T3-8. Kidney exchange + Roth-Sotomayor many-to-many** (~300 LOC). Cycle-and-chain-cover on the kidney-exchange graph (Roth-Sönmez-Ünver 2004). Many-to-many extends T1-11 hospital-residents.
- **T3-9. Solomon division + envy-free / equitable cake-cutting** (~200 LOC). Adjusted-winner (Brams-Taylor 1996), Selfridge-Conway 3-player envy-free, Aziz-Mackenzie n-player envy-free 2016 (the last is a research-grade primitive). 4-color cake-cutting moving-knife procedures.
- **T3-10. Kernel + prekernel** (~150 LOC). Cooperative-game solution concepts adjacent to nucleolus. Maschler-Davis-Davis (1979).
- **T3-11. Banzhaf-Coleman split (Coleman power-to-prevent-action vs power-to-initiate)** (~80 LOC). Coleman (1971) decomposition of Banzhaf into directional indices. Cheap once Banzhaf is present.
- **T3-12. Voting: Black, Bucklin, Coombs, Dodgson, Ranked Pairs (Tideman), instant-runoff variants, range voting, approval, Smith/Mutual Majority compliance, monotonicity / participation / consistency property checkers** (~400 LOC). The long-tail social-choice menu beyond T1-12's eight rules.

**T3 totals: ~12 primitives, ~2,500 LOC.**

---

## Cross-package coupling notes

- **Replicator dynamics (T2-1) needs `chaos/ode.go`'s RK4 step.** Already a clean dependency direction (gametheory → chaos requires no new chaos primitive).
- **Correlated equilibrium LP (T1-4), Stackelberg (T1-5), bargaining (T1-10), Nucleolus (T2-7), core (T2-7) all want `optim/simplex.go` to be a stable LP/simplex.** Audit slot 109 (optim) should confirm the simplex API supports general inequality + equality constraints with bounded/unbounded variables.
- **CFR family (T1-8, T2-12) requires a brand-new `gametheory/extensive.go` (~150 LOC) defining `ExtensiveFormGame`, `InformationSet`, `ChanceNode`, `TerminalNode`.** This is the highest-architectural-leverage commit in the entire package: every refinement (subgame-perfect, sequential, trembling-hand) reuses the same data structure.
- **Combinatorial auctions (T3-6) wants `linalg/sparse.go`-style ILP infrastructure** that doesn't exist yet — defer until there's a confirmed consumer.
- **Voting rules (T1-12) compose `combinatorics/`'s permutation enumeration for Kemeny-Young; `graph/`'s Floyd-Warshall for Schulze.** Both are present.
- **Bandit primitives (`bandit.go`)** are *not* in the Master-Plan-072 topic but should arguably move to a new `learning/` or `decision/` package: bandits are a *single-player* online-learning problem, structurally distinct from N-player game theory. Defer to slot ~134-139 (gametheory-api) for the call.
- **Kelly (`kelly.go`)** is similarly a single-decision-maker portfolio-sizing primitive, not a game. Same package-boundary question.

## Highest-leverage single PR

**Tier 1 #1+#2+#3+#6 fused into one commit (~750 LOC + ~80 golden vectors):**
- `NashPureNxM` → enumerate pure-strategy Nash on general bimatrix
- `NashSupportEnumeration` → enumerate *all* mixed-strategy Nash (canonical)
- `LemkeHowson` → fast single-NE pivot
- `Exploitability` → ε-Nash diagnostic

This commit replaces 071's "the package is the 2x2 sliver" headline finding with full bimatrix Nash coverage, enables six other Tier 1 / Tier 2 primitives (correlated equilibrium + Stackelberg + QRE + Bayesian Nash + best-response dynamics + replicator-dynamics-fixed-point-detection all need `Exploitability`), and ships with cross-validation against `nashpy.support_enumeration` + `nashpy.lemke_howson` for the golden-file pin (cross-language tier-1 validation per `CLAUDE.md` §3).

## Web research (2026-05-07)

- **Gambit 16.x (2024-10):** confirmed `gambit-enummixed` (T3-2) uses lrslib polyhedral enumeration, `gambit-lcp` is Lemke-Howson (T1-3), `gambit-logit` is logit-QRE-tracing (T2-3). Reference implementations match what reality should ship.
- **OpenSpiel (DeepMind, master 2026-04):** the CFR-family canonical reference. `open_spiel/algorithms/cfr.cc`, `cfr_br.cc` (CFR-BR), `external_sampling_mccfr.cc`, `outcome_sampling_mccfr.cc`, `discounted_cfr.cc` are the pinning targets for T1-8 + T2-12. Linear-CFR is `cfr.cc::LinearAveragingCFRSolver`.
- **Pluribus (Brown-Sandholm 2019, *Science*):** confirmed reliance on Linear-CFR + abstraction + Monte-Carlo subgame solving. Pure CFR variant; no novel game-theoretic primitive.
- **Libratus (Brown-Sandholm 2018, *Science*):** confirmed CFR+ + nested subgame solving. Same primitive set.
- **nashpy 0.0.41 (2025-09):** Python library, supports support-enumeration + Lemke-Howson + replicator-dynamics + fictitious-play. Reality should match this set as Tier-1 cross-language pinning target.
- **PyCFR / RLcard (2024-2026):** confirms MCCFR outcome-sampling is the practical poker baseline; CFR+ is the theoretical baseline.
- **No 2025-2026 paper has displaced CFR+ as the convergence-rate frontier for two-player zero-sum imperfect-information games.** Predictive-CFR (Farina-Kroer-Sandholm 2019) and DCFR (Brown-Sandholm 2019) are constant-factor improvements.

## Sanity checks performed

- Verified `gametheory/` directory contents (`bandit.go`, `gametheory_test.go`, `kelly.go`, `matching.go`, `nash.go`, `voting.go` — exactly 6 .go files).
- Verified golden files: only `testdata/gametheory/{nash_2x2,gale_shapley}.json` exist.
- Verified test count: 66 `Test*` functions in `gametheory_test.go`.
- Verified replicator dynamics is **not** in `chaos/` either (check: `Grep "[Rr]eplicator" reality/`).
- Confirmed 071's audit findings (Lemke-Howson absent, support enumeration absent, fictitious-play embedded in `Minimax`, Banzhaf only goes to ~25, Shapley exact ≤ 12) and inherits them.
- Verified zero `optim.LinearProgram` consumer in `gametheory/` today — every LP-form primitive (T1-4, T1-5, T1-10, T2-7) is a *new* compose.

## Out-of-scope flags

- **CSP / SAT-based equilibrium computation, ILP-based VCG winner determination at scale, polynomial-system NE enumeration (Bernstein-bound):** all need primitives outside `reality`'s current scope (no `polynomial/`, no `ilp/`). Defer.
- **Differentiable game theory (Foerster-Chen-Al-Shedivat-Whiteson, Letcher 2018; Mertikopoulos-Sandholm 2018):** would compose `autodiff` and is a 2024-2026 active research frontier — defer to a future `gametheory-sota` slot.
- **Mean-field game theory (Lasry-Lions 2007, Carmona-Delarue 2018):** PDE-class, doesn't fit reality's pure-math charter without first shipping a `pde/` package.
- **Algorithmic mechanism design (Nisan-Roughgarden-Tardos-Vazirani 2007):** the ILP-and-approximation-algorithm half belongs in a hypothetical `algos/` package, not `gametheory/`.

---

**Report length:** ~340 lines. Within ≤400-line budget.

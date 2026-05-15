# 073 | gametheory-sota

**Scope:** SOTA library + paper comparison for `C:\limitless\foundation\reality\gametheory\` against Gambit, OpenSpiel, nashpy, PettingZoo, and the poker-AI-frontier (Pluribus, Libratus, DeepStack, ReBeL/PokerGPT/POKERBENCH). Focus on **architecture choices**, not numerics (071) or feature gap-list (072).

**Source files surveyed:** `nash.go` (232 LOC), `voting.go` (291 LOC), `matching.go` (170 LOC), `bandit.go` (221 LOC), `kelly.go` (136 LOC). Total 1,050 LOC.

**Inheritance from 071/072:** 071 is a numerics audit (fictitious-play 100k-iter cap, 1e-15 indifference tol, Shapley n≤12 cliff). 072 enumerates ~1,800 LOC of missing primitives (Lemke-Howson, support enumeration, CFR family, mechanism design, repeated games). This file does **not** repeat either; it asks *what shape* the package should have when those primitives land.

---

## TL;DR — architecture, not features

reality/gametheory is a **flat func-only sliver** (`NashEquilibrium2x2(payoffA, payoffB [2][2]float64)`, `Minimax([][]float64, ...)`, `GaleShapley([][]int, [][]int)`). No `Game` type, no payoff/utility interface, no extensive-form data structure, no game-tree node type, no information-set abstraction, no strategy type, no equilibrium-result struct. Every SOTA library surveyed — Gambit (C++ core, Python bindings, since 1991), OpenSpiel (DeepMind, since 2019), nashpy (since 2017), PettingZoo (Farama, since 2020) — converges on **the same three core abstractions** that reality lacks:

1. **A `Game` type** carrying payoff/transition data, action spaces, and player count. Gambit calls it `Game`. OpenSpiel calls it `Game` (with `NewInitialState() → State`). nashpy calls it `Game(A, B)`.
2. **A `State`/`Information-Set` type** for sequential play and imperfect-information branching. OpenSpiel's `State.InformationStateString(player)` is the universal hook for CFR.
3. **An `Equilibrium`/`Solution` result type** carrying strategies, value, certificate, and (for iterative solvers) convergence metadata. Gambit returns `MixedStrategyProfile`; OpenSpiel returns `Policy`; nashpy yields tuples but documents the convergence contract.

Without these three types, every Tier-1 primitive in 072 (Lemke-Howson, CFR, support enumeration, correlated equilibrium LP) will either (a) reinvent its own ad-hoc shape, locking in API debt, or (b) get bolted onto `[][]float64` and never compose with extensive-form games. **The architectural recommendation of this report is: land the three types before the algorithms.** ~150 LOC of pure type definitions; zero math. Every SOTA library proves this is the right shape.

The poker-AI frontier (Pluribus, Libratus, DeepStack, ReBeL) is **out of zero-dep scope**: their algorithms (deep CFR, neural CFV networks, depth-limited continual re-solving) require neural-net backends that reality cannot ship. But the **mathematical kernels they sit on top of** — vanilla CFR, CFR+, MCCFR (outcome-sampling and external-sampling), regret matching, regret matching+ — are pure no-dep math, ~400 LOC for the family, citation-grounded, golden-file-testable. These are zero-dep portable; the neural part is not. Architecture: a `RegretMinimizer` interface lets all four CFR variants share the same outer driver, exactly as OpenSpiel does in `algorithms/cfr.h`.

---

## Library-by-library architecture comparison

### 1. Gambit (gambit-project.org, 16.5.0 released Feb 2026)

**Headline feature:** the *reference* library for finite-game equilibrium computation. Implements Lemke-Howson, simplicial subdivision, polynomial-system enumeration, iterated polymatrix approximation, action-graph games, quantal response equilibrium (logit-QRE), and (2026) LLM natural-language → game translation via `GameInterpreter`.

**Engineering choice — the EFG/NFG file format duo:**
- **NFG** = normal-form game (matrix per player). Reality's current `[2][2]float64`-per-player roughly maps here.
- **EFG** = extensive-form game (game tree with chance nodes, information sets, payoff terminals). Reality has **no analogue** and 072 cannot land any sequential-game algorithm without one.
- Both are **plain text formats** with `{"player1" "player2"} {"strategy1" "strategy2" ...}` syntax. This is a deliberate portability decision: every Gambit algorithm reads/writes EFG/NFG, which means *the same game definition runs through 12 different solvers*. This is exactly the "golden file as portable interface" pattern reality already uses for cross-language validation, applied at the *input* layer.

**Architectural choice — solver decoupling:** Gambit ships ~12 separate command-line tools (`gambit-enummixed`, `gambit-lcp` for Lemke-Howson, `gambit-simpdiv`, `gambit-logit`, `gambit-gnm` for global Newton, `gambit-ipa` for iterated polymatrix). Each is a thin CLI over a C++ class implementing one algorithm. The C++ core is `Game` + `MixedStrategyProfile<T>` + per-algorithm classes. **No god-object Solver class.**

**Zero-dep portability for reality:** **HIGH.** Gambit's C++ core has no external dependencies of substance (libgmp for arbitrary precision is optional; the float path is pure stdlib). The architectural lessons that port directly:
- Adopt an **EFG-equivalent extensive-form data structure** (`type GameNode struct { Player int; Children []*GameNode; InfoSet int; Chance []float64; Payoffs []float64 }`) before landing any sequential-game algorithm. ~80 LOC. This is the **single highest-leverage architectural commit** for the package.
- Ship algorithms as **separate functions over a common `Game` type**, not as methods on a `Solver` class. Reality's current func-only style is *correct* for this — the missing piece is the `Game` type itself.
- The `MixedStrategyProfile` shape: `type StrategyProfile struct { Strategies [][]float64; Player int; Value float64 }` — a single result type used by every equilibrium solver.

### 2. OpenSpiel (DeepMind, 1908.09453, since 2019; active 2026)

**Headline feature:** the universal RL+game-theory framework. Supports n-player zero-sum / general-sum, simultaneous and turn-based, perfect and imperfect information, and ships ~70 canonical game implementations (poker, chess, Go, Hanabi, bridge, mahjong, lewis-signaling, matrix games, etc.).

**Engineering choice — the `Game` + `State` + `Policy` triad:**
```cpp
class Game {
  virtual unique_ptr<State> NewInitialState() const = 0;
  virtual int NumPlayers() const = 0;
  virtual double MinUtility() const = 0;
  virtual double MaxUtility() const = 0;
};
class State {
  virtual vector<Action> LegalActions() const = 0;
  virtual void DoApplyAction(Action) = 0;
  virtual string InformationStateString(Player) const = 0;
  virtual vector<double> Returns() const = 0;
  virtual bool IsTerminal() const = 0;
};
class Policy {
  virtual ActionsAndProbs GetStatePolicy(const State&) const = 0;
};
```

This is **the most important architectural document** in the SOTA-comparison set. Every OpenSpiel algorithm — vanilla CFR, chance-sampled CFR, outcome-sampling MCCFR, external-sampling MCCFR, public-chance-sampling CFR, pure CFR, deep CFR, NFSP, PSRO, Q-learning, MCTS, Exploitability, BestResponse — is written against these three interfaces. **A new game implementing the interface gets all 20+ algorithms for free.** A new algorithm implemented against the interface runs on all 70 games for free.

**Architectural choice — CFR family unification:** `algorithms/cfr.h` defines `CFRSolverBase` with virtual `RegretMatching(InfoState&)` and `Solve()` methods. Vanilla, CFR+, Linear-CFR, Discounted-CFR, MCCFR-OS, MCCFR-ES are all subclasses overriding sampling strategy. **One outer loop, four sampling policies.** Reality's 072-Tier-1 list contains all four; they should ship as a single `RegretMinimizer` interface + four implementations, ~400 LOC total, not four standalone functions.

**Zero-dep portability for reality:** **MEDIUM-HIGH.** The C++ core is stdlib-only. The Python tensorflow/jax bits are not. The pieces that port:
- The `Game` + `State` + `Policy` triad — ~120 LOC of Go interfaces. **Critical architectural precondition for any extensive-form algorithm in 072.**
- The `InformationStateString(player)` hook — string-keyed information-set lookup is how CFR scales (no need to enumerate the tree; just hash the info-set on visit). This is a deliberate engineering choice that simplifies CFR by ~10× over a pointer-based info-set graph.
- The `Returns()` vector return — n-player utility, not just two-player payoff. Reality's current `payoffA, payoffB` arity is hardcoded 2-player; OpenSpiel's `vector<double> Returns()` is the n-player generalisation.
- The CFR variants themselves — pure math, no neural nets needed for the vanilla/+/Linear/Discounted/MCCFR-OS/MCCFR-ES family.

### 3. nashpy (drvinceknight, since 2017)

**Headline feature:** pure-Python 2-player game library. Implements support enumeration (Mangasarian-Stone 1964), vertex enumeration (Avis-Fukuda), Lemke-Howson, fictitious play, stochastic fictitious play, replicator dynamics, asymmetric replicator dynamics, replicator-mutator dynamics, and Moran process.

**Engineering choice — the `Game(A, B)` minimal abstraction:**
```python
game = nashpy.Game(A, B)              # bimatrix game
list(game.support_enumeration())      # generator of (s_A, s_B) pairs
list(game.lemke_howson_enumeration()) # all Lemke-Howson paths
xs = game.replicator_dynamics()       # row-player mix over time
list(game.fictitious_play(iterations=500))
```

The constructor takes two NumPy arrays, the methods are generators or arrays. **This is the closest match to reality's current style** — small, no inheritance, no policy objects, no state machines.

**Architectural choice — generator-based equilibrium enumeration:** `support_enumeration` and `lemke_howson_enumeration` are *generators*, not lists. A user can `next(...)` for the first equilibrium, or exhaust the generator for all of them, or break early. For games with exponentially many equilibria (worst case 2^n for n×n bimatrix), this is the only sane interface. Reality's 072-Tier-1 `EnumerateNashEquilibria` should follow this — Go's idiom is a channel or a callback closure, not an eager `[]Equilibrium` slice.

**Engineering choice — replicator dynamics returns the trajectory:** not just the final state. `xs = game.replicator_dynamics()` returns the full timeseries of population mixes, so the caller can plot, detect cycles, check Lyapunov stability, or extract the limit. This is a deliberate "expose intermediate state" choice that reality's `Minimax` *should* be making (071 §F1 already flagged that the iterative bounds aren't returned).

**Zero-dep portability for reality:** **HIGH.** nashpy depends only on numpy + scipy.optimize.linprog. The math is pure and the algorithms are textbook (Mangasarian-Stone, Lemke-Howson, Brown's fictitious play, Taylor-Jonker replicator ODE). All ~1,500 LOC of nashpy port directly to ~2,000 LOC of Go. The `Game(A, B)` API shape is also the easiest mental match for reality's current users.

### 4. PettingZoo + Gymnasium (Farama Foundation, NeurIPS 2021; current 2026)

**Headline feature:** the standard API for *multi-agent* reinforcement learning environments. ~70 environments (Atari multiplayer, butterfly cooperative, classic chess/go/poker). The Agent-Environment-Cycle (AEC) game model.

**Engineering choice — AEC vs Parallel API duality:**
- **AEC** = sequential (one agent acts per step). `for agent in env.agent_iter(): obs, rew, term, trunc, info = env.last(); env.step(action)`.
- **Parallel** = simultaneous (all agents act per step). `obs = env.reset(); while not done: actions = {agent: policy(obs[agent]) for agent in env.agents}; obs, rews, terms, truncs, infos = env.step(actions)`.

**Both APIs** are first-class because both game classes (sequential and simultaneous) are first-class. Reality's `NashEquilibrium2x2` is implicitly a *one-shot simultaneous-move game*. 072's missing extensive-form layer is an *AEC sequential* game. The architectural lesson: pick **one** model (AEC is more general — you can simulate parallel with AEC by buffering the round) and stick with it. OpenSpiel went AEC. Reality should also go AEC for the extensive-form data structure.

**Architectural choice — the `agent_iter` cursor:** the environment owns the play-order schedule, not the user. `env.agent_iter()` yields the next agent to act. This decouples the algorithm (what to do on this turn) from the game's structure (whose turn is it). **This is exactly the missing piece in reality's `Minimax` fictitious-play loop**, which hardcodes "row plays, then column plays" — reality cannot generalise to >2 players or to alternating-move games without an `Agent()` cursor on the game state.

**Zero-dep portability for reality:** **LOW-MEDIUM.** PettingZoo's value is the *interface*, not the implementations (which are Atari/Pygame/etc.). The API design — AEC, agent_iter, last/step pair — is portable as a ~50-LOC Go interface set. But reality is a *math library*, not a simulation framework; PettingZoo's design is most relevant to a sibling consumer package (Tempo? a future `agents/` package) that imports `gametheory` for solvers.

### 5. The poker-AI frontier — Libratus / DeepStack / Pluribus / ReBeL

**Libratus** (Brown-Sandholm, Science 2017): heads-up no-limit Texas hold'em. Three modules: (1) blueprint via MCCFR with **CFR+** (Tammelin 2014), (2) nested subgame solving, (3) self-improver that fixes blueprint holes overnight. Beat 4 top human pros.

**DeepStack** (Moravčík et al., Science 2017): same game, different architecture. **Continual depth-limited re-solving** + neural counterfactual value network (7-layer FFN, 500 nodes/layer, parametric ReLU) at the leaves. Hybrid vanilla-CFR + CFR+ with uniform weighting.

**Pluribus** (Brown-Sandholm, Science 2019): 6-player no-limit Texas hold'em. Blueprint = external-sampling MCCFR with two improvements (potential-aware abstraction + earth-mover-distance clustering) computed in **8 days, 12,400 core-hours, $144 of compute**. Live play on **28 cores**. Depth-limited search over **5 continuation strategies** at leaves (the key engineering trick — replace neural-CFV with a tiny discrete continuation policy).

**ReBeL** (Brown-Bakhtin-Lerer-Gong, NeurIPS 2020): generalisation to two-player zero-sum imperfect-info games. Public-belief-state MDP + neural value function + CFR sub-game solver.

**PokerGPT / POKERBENCH** (NeurIPS 2025): LLM end-to-end no-search poker via RLHF. Different paradigm; not in scope for a math library.

**Headline algorithm:** all four poker bots sit on **CFR / CFR+ / MCCFR**. The neural net is a value-function approximator at depth-limit leaves; everything else is regret minimisation.

**Engineering choice — the abstraction step:** real poker has 10^161 information sets. None of the bots solve the real game. They solve an **abstraction**:
- **Card abstraction:** cluster equivalent hands. Pluribus uses earth-mover distance over hand-strength distribution (potential-aware). 
- **Action abstraction:** quantise continuous bet sizes to 5-15 discrete buckets.
- The abstracted game has ~10^12-10^14 info sets, solvable by MCCFR in days.

This is **a pure-math operation** (clustering + bet-tree pruning) that reality could ship as an `Abstraction` type, ~200 LOC for the family. Pluribus's earth-mover-distance abstraction is a `prob/`-package operation (Wasserstein-1 between empirical distributions) that's already partially implementable by composing existing reality primitives.

**Zero-dep portability for reality:**
- **CFR / CFR+ / MCCFR-OS / MCCFR-ES / Linear-CFR / Discounted-CFR**: **HIGH.** Pure math, ~400 LOC for the family. 072-Tier-1.
- **Regret matching / regret matching+**: **HIGH.** ~50 LOC. The atomic primitive every CFR variant calls per info-set. Should be public so users can build custom no-regret learners.
- **Earth-mover / Wasserstein-1 hand abstraction**: **MEDIUM.** ~100 LOC; uses `linalg`-side network simplex. Belongs in `prob/` but called from `gametheory/`.
- **Depth-limited continual re-solving**: **MEDIUM.** ~150 LOC of CFR-with-leaf-cutoff; needs the extensive-form data structure first.
- **Neural counterfactual value functions** (DeepStack/ReBeL): **OUT OF SCOPE.** Requires NN backend; reality is zero-dep.
- **Pluribus's 5-continuation-strategy leaf trick**: **HIGH.** ~50 LOC. *Avoids* the neural net by using a tiny discrete policy lookup at leaves. **This is the zero-dep poker-AI architectural breakthrough.** A reality `gametheory` that ships {extensive-form `Game`, MCCFR, regret-matching+, 5-continuation depth-limit} could in principle solve abstracted heads-up Limit hold'em (Cepheus 2015 territory) entirely within CLAUDE.md rules.

---

## Architectural recommendations, ranked by leverage

### R1 — Land the `Game` / `State` / `StrategyProfile` triad before any algorithm. ~150 LOC. **Highest leverage in the package.**

Define:
```go
// Game is the universal interface for both normal-form and extensive-form games.
type Game interface {
    NumPlayers() int
    NewInitialState() State
    MinUtility() float64
    MaxUtility() float64
}

// State carries the current game position. Implementations cover normal-form
// (single-step), extensive-form (tree node), and POMG (info-set partition).
type State interface {
    CurrentPlayer() int            // -1 = chance, -2 = simultaneous
    LegalActions() []int
    Apply(action int) State        // returns new state (immutable transition)
    InformationStateKey(player int) string  // canonical hash for info-set lookup
    Returns() []float64            // payoffs at terminal; zeros otherwise
    IsTerminal() bool
    IsChance() bool
    ChanceOutcomes() []ChanceOutcome  // Action+Prob pairs
}

type StrategyProfile struct {
    Strategies [][]float64    // per-player action distribution, info-set or row index
    Value      []float64      // per-player expected utility
    Iters      int            // for iterative solvers; 0 for closed form
    Converged  bool           // did the convergence criterion fire?
}
```

This is the **single architectural commit that unblocks 072's entire Tier-1 list.** Lemke-Howson, support enumeration, CFR, correlated-equilibrium LP, Stackelberg, Bayesian-Nash, QRE, replicator dynamics — **all** of them want `Game` + `StrategyProfile` shapes. Without these types they each invent ad-hoc shapes and the package fragments.

Citation: OpenSpiel `open_spiel/spiel.h` (Lanctot et al. 2019), Gambit `Game` class (McKelvey-McLennan-Turocy 2014), nashpy `Game` (Knight 2017). All three SOTA libraries converge on this triad.

### R2 — Ship CFR family unified under a `RegretMinimizer` interface. ~400 LOC.

```go
type RegretMinimizer interface {
    UpdateRegrets(infoSet string, regrets []float64)
    AverageStrategy(infoSet string) []float64
    Iterate(g Game) StrategyProfile
}
```
With four implementations: `VanillaCFR`, `CFRPlus`, `OutcomeSamplingMCCFR`, `ExternalSamplingMCCFR`. **One outer loop, four sampling policies** — exactly the OpenSpiel pattern. This is the only architectural shape that lets a single `Iterate(g)` call work across the full poker-AI algorithm family without rewriting the driver four times.

Citation: OpenSpiel `algorithms/cfr.h` (Lanctot 2009 + 2019 framework), Tammelin 2014 (CFR+), Lanctot 2009 (MCCFR), Brown-Sandholm 2019 (Linear-CFR + Discounted-CFR).

### R3 — Adopt nashpy's generator pattern for equilibrium enumeration. ~50 LOC.

For `EnumerateNashEquilibria`, `EnumerateCorrelatedEquilibria`, `EnumerateLemkeHowsonPaths`: return a Go channel or accept a callback closure, not a `[]StrategyProfile` slice. Worst-case 2^n equilibria for n×n; users almost always want the *first one* and a way to stop. nashpy uses Python generators; the Go idiom is `func enumerate(g Game, yield func(StrategyProfile) bool)` (the new `range func` idiom is also acceptable post-Go-1.23).

Citation: nashpy `support_enumeration` / `lemke_howson_enumeration` design (Knight 2017).

### R4 — Strategy ports of Pluribus's depth-limit-via-continuation trick. ~50 LOC.

Once R1+R2 are in, a `DepthLimitedCFR` solver with a 5-element discrete continuation policy at leaves is ~50 LOC and gets reality to **2017-Cepheus-tier** (heads-up Limit hold'em essentially solved) for any extensive-form game with cluster-able information sets. **The neural-CFV alternative (DeepStack/ReBeL) is permanently out of scope.** This is the zero-dep poker-AI architectural sweet spot.

Citation: Brown-Sandholm Science 2019 §"Limited-lookahead search".

### R5 — Adopt PettingZoo's AEC `agent_iter` cursor on `State`. ~20 LOC.

Generalises `Minimax`'s hardcoded "row, then column" loop to n-player turn-taking and to chance nodes. The current 2-player-only assumption is encoded in the `[2][2]float64` shape and breaks the moment 072-Tier-1 lands an N-player solver.

Citation: PettingZoo NeurIPS 2021 (Terry et al.); OpenSpiel `State::CurrentPlayer()`.

### R6 — Ship `replicator_dynamics` returning the full trajectory. ~80 LOC (also flagged by 071 §F).

nashpy's `xs = game.replicator_dynamics()` is the right shape: full timeseries, not just the limit. Composes naturally with `chaos`-package Lyapunov-exponent and limit-cycle detection. Reality's `Minimax` should additionally return its convergence-bracket history `[v_t, V_t]` per iteration so callers can verify Robinson 1951 convergence.

Citation: Taylor-Jonker 1978 (replicator ODE); Sandholm 2010 *Population Games and Evolutionary Dynamics*; nashpy `replicator_dynamics` (Knight 2017).

---

## What is correctly out of scope

- **Neural CFV networks** (DeepStack 7-layer FFN, ReBeL value net): requires NN backend. Out of scope per CLAUDE.md rule 2 (zero deps).
- **LLM game translation** (Gambit GameInterpreter v2 2026): requires LLM. Out of scope.
- **PettingZoo Atari/Pygame environments**: simulation runtime, not math. Belongs in a sibling consumer package, not `reality/`.
- **OpenSpiel TensorFlow/JAX algorithm bindings** (NFSP, PSRO, deep CFR): NN-based. Out of scope.
- **Source-to-source game-tree compilation** (Gambit's polymatrix-approximation IR): IR-required. Out of scope.

---

## Summary: the architectural gap is shape, not size

reality/gametheory at 1,050 LOC is roughly nashpy-2017-tier in **size**, but Gambit-1991-tier in **shape** — the right algorithms are stubbed (Nash 2x2, fictitious-play minimax) but they sit on the wrong abstractions (`[2][2]float64`, no `Game` type, no `State`, no info-set hash, no `StrategyProfile`). 072's missing-feature list (~1,800 LOC of additive primitives) cannot land cleanly without R1's ~150 LOC of architectural types first. **R1 is the unique sequencing constraint** for the entire package roadmap.

Of the SOTA frontier, exactly **two pieces of architecture port at zero dependency cost** and would not exist in any other Go game-theory library: (1) OpenSpiel's `Game`+`State`+`Policy` triad, ported as ~150 LOC of Go interfaces; (2) Pluribus's 5-continuation-strategy depth-limit trick, ~50 LOC over MCCFR, that achieves Cepheus-tier poker without a neural net. Both are citation-grounded, both are golden-file-testable across Go/Python/C++/C#, both stay strictly inside CLAUDE.md.

---

## Sources

- [Gambit: The package for computation in game theory](https://www.gambit-project.org/about/)
- [Gambit Documentation 16.5.0 (Feb 2026)](https://media.readthedocs.org/pdf/gambitproject/latest/gambitproject.pdf)
- [GitHub — gambitproject/gambit](https://github.com/gambitproject/gambit)
- [Game Theory Explorer — Software for the Applied Game Theorist (Savani-von Stengel 2014)](https://arxiv.org/pdf/1403.3969)
- [OpenSpiel: A Framework for Reinforcement Learning in Games (Lanctot et al. 2019, arXiv:1908.09453)](https://ar5iv.labs.arxiv.org/html/1908.09453)
- [open_spiel/algorithms/cfr.h (DeepMind)](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/algorithms/cfr.h)
- [open_spiel/algorithms/external_sampling_mccfr.h](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/algorithms/external_sampling_mccfr.h)
- [Multi-agent RL in OpenSpiel: A Reproduction Report (arXiv:2103.00187)](https://ar5iv.labs.arxiv.org/html/2103.00187)
- [Nashpy — A python library for 2 player games (Knight)](https://github.com/drvinceknight/Nashpy)
- [Nashpy documentation: replicator dynamics](https://nashpy.readthedocs.io/en/stable/how-to/use-replicator-dynamics.html)
- [PettingZoo: Gym for Multi-Agent RL (Terry et al., NeurIPS 2021, arXiv:2009.14471)](https://arxiv.org/abs/2009.14471)
- [PettingZoo Documentation (Farama Foundation)](https://pettingzoo.farama.org/index.html)
- [Superhuman AI for heads-up no-limit poker: Libratus (Brown-Sandholm, Science 2017)](https://www.science.org/doi/10.1126/science.aao1733)
- [Libratus — Wikipedia](https://en.wikipedia.org/wiki/Libratus)
- [DeepStack: Expert-Level AI in Heads-Up No-Limit Poker (Moravčík et al., arXiv:1701.01724)](https://arxiv.org/pdf/1701.01724)
- [Superhuman AI for multiplayer poker — Pluribus (Brown-Sandholm, Science 2019)](https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf)
- [Combining Deep RL and Search — ReBeL (Brown et al., NeurIPS 2020, arXiv:2007.13544)](https://arxiv.org/pdf/2007.13544)
- [DecisionHoldem: Safe Depth-Limited Solving (arXiv:2201.11580)](https://arxiv.org/html/2201.11580v2)
- [Counterfactual Regret Minimization — Int8 explainer](https://int8.io/counterfactual-regret-minimization-for-poker-ai/)
- [NeurIPS 2025 Poster — LLM Strategic Reasoning: Behavioral Game Theory](https://neurips.cc/virtual/2025/poster/117505)
- [Game Theory Meets LLMs: A Systematic Survey (arXiv:2502.09053)](https://arxiv.org/pdf/2502.09053)

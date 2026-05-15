# 074 | gametheory-api

**Scope:** API ergonomics audit of `C:\limitless\foundation\reality\gametheory\` — function signatures, parameter shapes, return-tuple structure, pure-vs-mixed-strategy expressiveness, payoff-matrix conventions, player-count generalisation, and how the surface composes (or fails to compose) across the 11 public functions.

**Source files surveyed:** `nash.go` (232), `voting.go` (291), `matching.go` (170), `bandit.go` (221), `kelly.go` (136), `gametheory_test.go` (932). Total prod 1,050 LOC, 11 exported funcs.

**Inheritance from 071/072/073:**
- 071 = numerics audit (fictitious-play 100k cap, 1e-15 indifference tol, convergence-bracket return). I do **not** repeat numerics; I treat the missing `(gap, epsilon)` return as a *signature-shape* observation.
- 072 = missing-feature gap list (~1,800 LOC of additive primitives: Lemke-Howson, CFR, mechanism design, etc.). I do **not** repeat the feature list; I ask what *shapes* the existing functions should commit to so 072's primitives can land coherently.
- 073 = SOTA library comparison + the "land the `Game`/`State`/`StrategyProfile` triad" architectural recommendation. I do **not** re-derive the triad; I take it as given and answer the *narrower* API question: **what should the existing 11 signatures look like under that triad, and what is the cheapest minimal-`Game` shape that fits the *current* normal-form-only feature set without committing to the full extensive-form interface?**

---

## TL;DR — eleven functions, six different argument shapes

The package's public surface uses **six mutually inconsistent ways to describe a game's structure**. None of them is a `Game` type:

| Function | Argument shape | Player-count assumption | Strategy shape |
|---|---|---|---|
| `NashEquilibrium2x2(payoffA, payoffB [2][2]float64)` | fixed 2×2 array per player | 2-player, 2-strategy, hardcoded | **`[2]float64`** array |
| `Minimax(payoff [][]float64, nRows, nCols int)` | jagged slice + redundant dims | 2-player zero-sum (1 matrix) | **`[]float64`** slice |
| `GaleShapley(proposerPrefs, receiverPrefs [][]int)` | rank lists | 2-side, equal-size | `[]int` matching |
| `IsStableMatching(matching []int, proposerPrefs, receiverPrefs [][]int)` | matching + ranks | same | bool |
| `BanzhafIndex(weights []float64, quota float64)` | weighted-voting tuple | n-player, simple game | `[]float64` index |
| `ShapleyValue(n int, charFunc func([]bool) float64)` | characteristic function | n-player cooperative | `[]float64` |
| `ShapleyValueWeightedVoting(weights []float64, quota float64)` | shortcut wrapping above | n-player, simple game | `[]float64` |
| `KellyFraction(prob, odds float64)` | scalar bet | 1-player decision | scalar |
| `KellyFractionMultiple(probs, odds []float64)` | vectored bets | 1-player | `[]float64` |
| `UCB1(counts []int, rewards []float64, totalPulls int)` | bandit state tuple | 1-player online | int (arm) |
| `ThompsonSampling/EpsilonGreedy(...)` | bandit state + RNG | 1-player online | int (arm) |

There is no shared input type. There is no shared output type. There are **two** different conventions for "probability vector over k strategies" (`[2]float64` vs `[]float64`). There are **three** different conventions for "describe a game's payoffs" (a pair of fixed 2×2 arrays, a single jagged slice with redundant dims, a closure over coalition membership). Even the 2-player normal-form game is described differently by `NashEquilibrium2x2` (bimatrix `(A, B)`) and `Minimax` (zero-sum single `A`, with the column player's payoff being implicit `-A`).

The architectural recommendation in 073 (`Game`/`State`/`StrategyProfile` triad, ~150 LOC of Go interfaces) is the *correct long-term* answer. This audit makes a **narrower, cheaper, more immediate claim**: even *before* the extensive-form `State` interface lands, the package should commit to (a) one normal-form `Game` struct, (b) one `MixedStrategy = []float64` vocabulary, and (c) one `Equilibrium` result struct. ~80 LOC. Strictly subset of 073-R1, and resolves the six-shape inconsistency *without* requiring extensive-form support.

---

## Issue map — eleven specific signature-level problems

### A1. `[2][2]float64` is the wrong shape for the only 2-player bimatrix function

`NashEquilibrium2x2(payoffA, payoffB [2][2]float64)` (`nash.go:45`) uses Go fixed-size arrays. This locks the function to *exactly* 2 strategies per player and prevents it from ever generalising. The natural progression — Nash equilibrium for 2×3, 3×2, 3×3, m×n bimatrix games — cannot reuse this signature; it needs a new `NashEquilibriumBimatrix(A, B [][]float64)` and the 2x2 specialisation becomes a back-compat wrapper. **This is API debt locked in by the type, not the algorithm.** The closed-form 2x2 algorithm itself trivially extends to m×n via support enumeration (Mangasarian-Stone 1964; ~80 LOC) which is in 072's Tier-1 list.

The corresponding **return** types `(stratA, stratB [2]float64)` have the same problem: a `[2]float64` cannot be the same type as a `[]float64` strategy returned by `Minimax`, so a downstream consumer that wants to compute "expected utility under this strategy profile" cannot write a single utility function — it must specialise on the equilibrium-finder it called. **This is the most user-facing single-symbol type bug in the package.**

**Recommendation:** rename `NashEquilibrium2x2` → `NashEquilibriumBimatrix(A, B [][]float64) Equilibrium`. Detect 2×2 internally and dispatch to the closed-form path; fall through to support-enumeration for larger. Strategies return as `[]float64`. The `[2][2]float64` signature stays as a deprecated wrapper for one release for back-compat. Citation: nashpy `Game(A, B)` constructor (Knight 2017); Gambit `MixedStrategyProfile<double>` (McKelvey-McLennan-Turocy).

### A2. `Minimax` redundantly takes `(payoff, nRows, nCols)` when slice has `len`

`Minimax(payoff [][]float64, nRows, nCols int)` (`nash.go:132`). In Go, `len(payoff)` is the row count and `len(payoff[0])` is the column count of the first row. The redundant `nRows, nCols` parameters are an anti-idiom inherited from C-style "pass the array dimensions" and are a footgun: nothing prevents `Minimax(matrix3x4, 5, 6)` from compiling, and the function would silently read out of bounds.

**Recommendation:** drop `nRows, nCols`. Signature becomes `Minimax(payoff [][]float64) Equilibrium` with internal `len()` reads. Validate that all rows have equal length; return zero-value equilibrium with `Err = ErrJaggedMatrix` on mismatch. Verified against test sites: every existing call site in `gametheory_test.go` passes `len(payoff)` and `len(payoff[0])` literally (e.g., `Minimax(payoff, 3, 3)` for rock-paper-scissors at `nash.go:130`), so the change is mechanical.

### A3. Zero-sum vs general-sum convention is implicit and undocumented

`Minimax` takes *one* payoff matrix; `NashEquilibrium2x2` takes *two*. The implicit convention is "Minimax assumes column player's payoff = `-A[i][j]` (zero-sum); NashEquilibrium2x2 takes both matrices independently (general-sum)". This is **never stated in the doc strings**. A user with a general-sum 3×3 game and the doc-comment-driven mental model "Minimax solves m×n games" will hit incorrect results because the zero-sum reduction silently misrepresents their game.

**Recommendation:** doc-comment one line each — `Minimax`: "Two-player zero-sum. Column player payoff is `-payoff`." And `NashEquilibrium2x2`: "Two-player general-sum bimatrix. Column player has its own payoff matrix." Once the `Game` struct exists, these become two methods on `Game`: `g.MinimaxValue()` (only valid if `g.IsZeroSum()`) and `g.NashEquilibrium()` (general-sum). Citation: von Neumann 1928 (zero-sum minimax); Nash 1950 (general-sum bimatrix).

### A4. Equilibrium output is a tuple, not a struct — extension-hostile

Both equilibrium-finders return `(stratA, stratB, value)` triples. Adding a fourth return (e.g., 071's recommended `gap` or `iters` for convergence accountability) is a **breaking API change** because Go has positional return values. Adding a fifth (a list of *all* equilibria, an indicator of pure-vs-mixed, a certificate matrix) compounds the breakage.

**Recommendation:** introduce `type Equilibrium struct { Strategies [][]float64; Value []float64; Iters int; Converged bool; Gap float64; Type EqType }` where `EqType` is `Pure | Mixed | Correlated | Stackelberg`. All future equilibrium-finders return this struct, and adding fields (e.g., `Support []int` for support-enumeration output, `Path []int` for Lemke-Howson pivot path) is non-breaking. **This is the structural prerequisite for landing 072's full Tier-1 list.** Subset of 073-R1's `StrategyProfile` — the same type, viewed from the narrower normal-form angle.

### A5. Pure vs mixed equilibrium output is undifferentiated

`NashEquilibrium2x2` may return either a pure equilibrium (e.g., `[1, 0]` for both players) or a mixed equilibrium (e.g., `[0.6, 0.4]`). The caller cannot tell which **without inspecting the strategy vector** for {0, 1} entries. This is silent; there is no `IsPure() bool` accessor.

For 2x2 games this is benign — a strategy with all probability on one action is unambiguously pure. For larger m×n games (once A1's signature change lands) it is not: a mixed strategy with three actions and a numeric `[0.5000000001, 0.0, 0.4999999999]` over [pure, near-zero, mixed] is ambiguous within floating tolerance. The `Type EqType` field on the `Equilibrium` struct (A4) resolves this once the algorithm-side classification is preserved.

**Recommendation:** the `EqType` enum on the result struct. The classification is **free** at solve time (the algorithm knows whether it took the pure-NE branch at `nash.go:98-106` or the mixed-interior branch at `nash.go:67-73`); it just isn't exposed.

### A6. Number of equilibria: only the *first* is returned

`NashEquilibrium2x2` "returns the first pure NE found" (`nash.go:99`). Coordination games have multiple pure equilibria (the canonical Battle-of-the-Sexes example, hardcoded into `gametheory_test.go:62`). Returning only one **arbitrarily privileges row-then-column iteration order** over the player's actual problem (which equilibrium does the user care about?). Battle-of-the-Sexes has three Nash equilibria (two pure, one mixed); reality returns one of the two pure ones, deterministically by iteration order.

**Recommendation:** ship a sibling `NashEquilibriaAll(A, B [][]float64) []Equilibrium` that enumerates them, and `NashEquilibrium(...)` is a convenience for "the first one" with a doc-comment caveat. Following nashpy's generator pattern (073-R3) is even better — return a Go iter.Seq[Equilibrium] under the new `range func` idiom — because for n×n bimatrix the worst case is 2^n equilibria. Citation: Mangasarian-Stone 1964 (support enumeration); Avis-Fukuda 1992 (vertex enumeration); 073-R3.

### A7. `Minimax` does not honour Nash's theorem (always-an-equilibrium guarantee)

Nash's theorem (1950): every finite game has at least one mixed Nash equilibrium. The fictitious-play loop in `Minimax` runs a fixed `maxIter = 100000` iterations (`nash.go:171`) and returns *whatever the empirical mix happens to be*, even if it has not converged. A user has no signal that the result is converged-to-equilibrium vs early-terminated-at-iteration-cap. (071 §F1 already flagged the missing convergence bracket; this is the API-signature consequence — the caller cannot even *ask* whether the answer is good.)

**Recommendation:** add `Equilibrium.Converged bool` and `Equilibrium.Gap float64` (the v_t / V_t bracket width per Robinson 1951). Caller can check `eq.Converged && eq.Gap < 1e-6` before trusting the answer. Same rec as 071-F1 but expressed at the API-shape level.

### A8. Bandit state is fragmented across (counts, rewards, [successes, failures])

`UCB1(counts, rewards, totalPulls)`. `ThompsonSampling(successes, failures, rng)`. `EpsilonGreedy(rewards, counts, epsilon, rng)`. Three different argument tuples for "the state of a multi-armed bandit". A consumer that wants to A/B-test multiple bandit strategies on the same problem must **maintain three different state representations** in lockstep. Worse: `EpsilonGreedy` and `UCB1` swap the order of `rewards` and `counts`.

**Recommendation:** `type BanditState struct { Counts []int; Rewards []float64; Successes []int; Failures []int; TotalPulls int }`. Each of the three algorithms takes `BanditState` and (where needed) an RNG. The struct is a strict superset of all three current argument lists. **The order-swap between `UCB1(counts, rewards, ...)` and `EpsilonGreedy(rewards, counts, ...)` is a latent bug**: any caller who substitutes one for the other without re-reading the signature gets silently-wrong results. (Verified at `bandit.go:37` and `bandit.go:195`.)

### A9. `ShapleyValue` and `BanzhafIndex` use different game representations for the same problem

`ShapleyValue(n, charFunc func([]bool) float64)` takes a closure. `BanzhafIndex(weights []float64, quota float64)` takes a weighted-voting tuple. **The weighted-voting case is a special case of the cooperative game** (the simple game where v(S) = 1 iff sum >= quota, else 0); reality already implements this composition in `ShapleyValueWeightedVoting`. So weighted voting has a closure-form, a tuple-form, *and* a wrapper that bridges them.

The Banzhaf index, however, only has the tuple-form. There is no `BanzhafIndexCooperative(n, charFunc)` that takes the general cooperative-game closure. The two power indices have asymmetric expressivity for no good mathematical reason.

**Recommendation:** add `BanzhafIndexCooperative(n int, charFunc func([]bool) float64) []float64` for symmetry. Citation: Banzhaf 1965; Shapley 1953. Both are special cases of the broader **probabilistic-value family** (Weber 1988); a future `ProbabilisticValue` that takes a coalition-weight function unifies both. (Out-of-scope here; flagged for 072.)

### A10. RNG is `interface { Float64() float64 }` — accidental, and inconsistent with stdlib

`ThompsonSampling` and `EpsilonGreedy` accept `rng interface{ Float64() float64 }` (`bandit.go:88`, `bandit.go:195`). This is structurally compatible with `*math/rand.Rand` (which has `Float64() float64`) but is *also* compatible with `*math/rand/v2.Rand`, with a `crypto/rand`-backed wrapper, with a deterministic LCG, etc. The structural-typing flexibility is good — until you compare it with `voting.go:191`'s `shapleySampled` which uses an inline-defined LCG `seed*6364136223846793005 + 1442695040888963407` instead of accepting an RNG argument at all.

**Recommendation:** every function in the package that consumes randomness should take **the same RNG interface** as a parameter. The current `interface{ Float64() float64 }` is fine *as a contract* but should be **named** (`type RNG interface{ Float64() float64 }`) and used uniformly. `ShapleyValue` for n>12 should accept an RNG instead of hardcoding the LCG seed. Determinism of golden-file tests is preserved by the caller passing a seeded RNG; reality's role is to expose the seam, not own the seed.

### A11. `KellyFractionMultiple` silently scales — return shape hides it

`KellyFractionMultiple(probs, odds []float64) []float64` (`kelly.go:78`). When the sum of positive Kelly fractions exceeds 1 (overcommitment), the function silently scales them down to sum to 1 (`kelly.go:97-104`). The caller cannot distinguish "all bets at full Kelly, total < 1" from "scaled-down due to overcommitment" without summing the result themselves.

**Recommendation:** return `(fractions []float64, scaledDown bool, scaleFactor float64)` — or, with the Equilibrium-struct refactor, return a `KellyResult` carrying a `ScaleApplied float64` field (1.0 = no scaling). For the single-bet `KellyFraction`, the same applies to the [-1, 1] clamp at `kelly.go:45-50` — the caller should be told if their bet was clamped.

---

## A minimal normal-form `Game` struct — 30 LOC, strict subset of 073-R1

073-R1 recommends a full `Game` interface with `NewInitialState() State`, `MinUtility()`, `MaxUtility()`, etc. — the right answer for extensive-form games. For reality's *current* feature set (no extensive form, no sequential games, no chance nodes), a much simpler struct suffices and resolves the six-shape mess at A1-A4 today:

```go
// NormalFormGame is a simultaneous-move finite game in strategic form.
// Payoffs[player][rowAction][colAction]... is an n-dimensional payoff
// tensor for player. For 2-player games, it is two 2-D matrices.
type NormalFormGame struct {
    NumPlayers int
    NumActions []int          // NumActions[player] = |strategy set of player|
    Payoffs    [][]float64    // flat tensor; Payoffs[p*stride + idx] for player p
}

// IsZeroSum reports whether sum_p payoffs[p][joint] == 0 for all joint actions.
// Cached after first call.
func (g *NormalFormGame) IsZeroSum() bool { ... }

// Bimatrix returns (A, B) for a 2-player game; nil, nil if NumPlayers != 2.
func (g *NormalFormGame) Bimatrix() (A, B [][]float64) { ... }

// Equilibrium is the result of any equilibrium computation.
type Equilibrium struct {
    Strategies [][]float64    // Strategies[player] = mixed strategy
    Value      []float64      // expected utility per player
    Type       EqType         // Pure | Mixed | Correlated | Stackelberg
    Iters      int            // 0 for closed-form; >0 for iterative
    Converged  bool           // for iterative solvers
    Gap        float64        // Robinson-1951 bracket width; 0 for exact
}

type EqType int
const (
    EqPure EqType = iota
    EqMixed
    EqCorrelated
    EqStackelberg
)
```

This is **strictly subset of 073's `Game` interface** (which has `NewInitialState`, `IsTerminal`, etc.). It does not block 073-R1; it is the normal-form-only specialisation that 073-R1 will eventually subsume via a `NormalFormGame` *adapter* implementing the `Game` interface. Adopting this *now* costs ~30 LOC, breaks no existing caller (the old `[2][2]float64` and `[][]float64` signatures stay as wrappers), and resolves issues A1, A2, A3, A4, A5, A7 in one stroke.

---

## Unification target — what eleven functions look like under the struct

| Current | Refactored |
|---|---|
| `NashEquilibrium2x2(payA, payB [2][2]float64) ([2]float64, [2]float64, float64)` | `(*NormalFormGame).NashEquilibrium() Equilibrium` |
| `Minimax(payoff [][]float64, nRows, nCols int) ([]float64, []float64, float64)` | `(*NormalFormGame).MinimaxValue() Equilibrium` (precondition: `IsZeroSum()`) |
| `GaleShapley(propPrefs, recvPrefs [][]int) []int` | unchanged — matching is structurally distinct from games-with-payoffs and a `MatchingMarket` type is overengineering for two-side rank-list inputs. |
| `IsStableMatching(matching []int, prefs, prefs [][]int) bool` | unchanged for the same reason. |
| `BanzhafIndex(weights []float64, quota float64) []float64` | `(*WeightedGame).Banzhaf() []float64` — `WeightedGame{Weights, Quota}` is a 4-LOC struct; or keep current API. |
| `ShapleyValue(n int, charFunc func([]bool) float64) []float64` | `(*CooperativeGame).Shapley() []float64` — `CooperativeGame{N, CharFunc}` likewise. |
| `ShapleyValueWeightedVoting(weights, quota)` | `(*WeightedGame).Shapley() []float64`. |
| `KellyFraction(prob, odds float64) float64` | unchanged — single decision, no game shape needed. |
| `KellyFractionMultiple(probs, odds []float64) ([]float64, bool, float64)` | add the `(scaledDown, scale)` returns per A11. |
| `UCB1 / ThompsonSampling / EpsilonGreedy(state BanditState, ...) int` | unify state struct per A8. |

Counts: 11 → 11 (same surface area, unified shapes). LOC delta: +~80 (struct definitions + back-compat wrappers). Breaking-change footprint: **zero** if back-compat wrappers ship in the same release. **All eleven inconsistencies (A1-A11) close.**

---

## What is correctly out of scope

- **Extensive-form `State` / `InformationSet` / chance-node types** — that is 073-R1 and 072-T1-8. A1's normal-form-only `NormalFormGame` does *not* preclude them; it is the simpler cousin that ships first.
- **Per-algorithm policy classes (`SolverInterface`, `BestResponseSolver`, etc.)** — Gambit and OpenSpiel use these for their plugin architecture; reality's flat-function style is *correct* for a 1,050-LOC math library and should not adopt the heavy class hierarchy. 073-R2's `RegretMinimizer` interface is the one exception, justified by the four-CFR-variants common driver.
- **Bandit move to a separate `learning/` or `decision/` package** — flagged by 072 §epilogue, not an API-shape question, deferred.
- **RNG type unification across the entire `reality` repo** — flagged by A10 within `gametheory/` only; cross-package RNG harmonisation is a different audit slot.

---

## Recommendations, ranked by leverage (within the 074 scope)

1. **Land `NormalFormGame`, `Equilibrium`, `EqType`** (~30 LOC). Resolves A1, A2, A3, A4, A5, A7. Strict subset of 073-R1; ships before extensive-form support.
2. **Fix `Minimax(payoff, nRows, nCols)` redundancy** (A2, ~10 LOC). Dimensions come from `len(payoff)`. Critical for caller correctness.
3. **Generalise `NashEquilibrium2x2` → `NashEquilibriumBimatrix(A, B [][]float64)`** (A1, ~50 LOC with support-enumeration fallback). Drops the `[2][2]float64` lock-in.
4. **Add `Equilibrium.Converged` and `.Gap`** (A7, ~20 LOC). Honours Robinson 1951; addresses 071-F1 at the API-shape level.
5. **Unify `BanditState` struct** (A8, ~20 LOC). Removes the `(rewards, counts)` argument-order inconsistency between `UCB1` and `EpsilonGreedy`.
6. **Enumerate Nash equilibria, not just first** (A6, ~80 LOC; nashpy generator pattern; 073-R3).
7. **Doc-comment zero-sum vs general-sum convention** (A3, ~5 LOC of docs).
8. **Add `BanzhafIndexCooperative(n, charFunc)` for shape symmetry with Shapley** (A9, ~30 LOC).
9. **Name and document the RNG seam** (`type RNG interface{ Float64() float64 }`) (A10, ~5 LOC).
10. **Return `(scaledDown, scaleFactor)` from KellyFractionMultiple** (A11, ~10 LOC).

Total: ~260 LOC, zero breaking changes if existing functions become back-compat wrappers around the new struct API. **Every recommendation here is strict subset of 073-R1 + 073-R3.** This audit's contribution is the *signature-level* taxonomy: which of the eleven existing functions misuse which of the six argument shapes, and the minimum-LOC unification.

---

## Sources

- [nashpy `Game` API design (Knight 2017)](https://nashpy.readthedocs.io/en/stable/text-book/index.html)
- [OpenSpiel `Game`/`State`/`Policy` triad (Lanctot et al. 2019, arXiv:1908.09453)](https://ar5iv.labs.arxiv.org/html/1908.09453)
- [Gambit `MixedStrategyProfile<T>` (McKelvey-McLennan-Turocy 2014)](https://gambitproject.readthedocs.io/en/v16.0.1/strategies.html)
- Mangasarian, O.L. & Stone, H. (1964). "Two-person nonzero-sum games and quadratic programming," J. Math. Anal. Appl. 9(3):348-355. (support-enumeration for bimatrix Nash)
- Avis, D. & Fukuda, K. (1992). "A pivoting algorithm for convex hulls and vertex enumeration of arrangements and polyhedra," Discrete Comput. Geom. 8:295-313.
- Robinson, J. (1951). "An iterative method of solving a game," Annals of Mathematics 54:296-301. (fictitious-play convergence with bracket [v_t, V_t])
- Weber, R.J. (1988). "Probabilistic values for games," in The Shapley Value: Essays in Honor of L.S. Shapley, pp. 101-119. (unifies Banzhaf and Shapley as probabilistic values)
- Tammelin, O. (2014). "Solving large imperfect information games using CFR+," arXiv:1407.5042.
- Brown, G.W. (1951). "Iterative solution of games by fictitious play," in Activity Analysis of Production and Allocation, pp. 374-376.
- 073 (this review): `agents/073-gametheory-sota.md` — the `Game`/`State`/`StrategyProfile` triad recommendation that 074-R1 is a strict subset of.

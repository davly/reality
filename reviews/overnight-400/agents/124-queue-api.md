# 124 — queue: API ergonomics

**Topic:** queue-api — open vs closed networks, transient vs steady-state, result types, topology shape, single- vs multi-class, comparison with sibling `graph` package routing patterns.
**Date:** 2026-05-07
**Agent:** 124 of 400
**Scope:** API surface ergonomics. Slots 121 (numerics), 122 (missing primitives), 123 (SOTA comparison) are the prerequisites; this slot does NOT redo gap-listing or numerical analysis.

---

## Headline

The current `reality/queue` API has **eight functions** with **three structural shape problems** that will compound badly when the missing primitives from 122/123 land:

1. **No closed-network entry point at all** — `JacksonNetwork` is exclusively open-network. There is no place in the type system for a "fixed customer population N" parameter, no `MVA` signature, no convention for whether `N` is a configuration or a query argument. The naming `JacksonNetwork(...)` pre-emptively claims the open-vs-closed-agnostic name; the closed variant has no obvious second home (`ClosedJacksonMVA`? `JacksonClosed`? `MVA`?).
2. **Steady-state is silently assumed everywhere** — every function returns `(L, Lq, W, Wq, ρ)` *steady-state* expectations. No function takes a time argument `t`; no function returns a transient response `L(t)`, `P(N(t)=k | N(0)=n₀)`, or relaxation time `τ`. The package docstring (`basic.go:1-21`) doesn't even say "steady-state" — Pulse/Sentinel consumers calling `MM1` during a load spike are getting the wrong answer and the API hides that.
3. **Five-tuple positional returns that don't extend** — `MM1`, `MMc`, `MM1K` use named-return tuples `(Lq, Wq, L, W, rho [, pLoss])`. Adding any field (multi-class L, blocking probability, departure C², percentile waits, transient `tau`) is a breaking change. Every Tier 1 addition from 122 (M/G/1, M/D/1, M/M/∞, Kingman, M/M/c/c-as-tuple, MVA) wants at least one extra field. The shape choice (tuple vs struct) decided once, repeated nine times, becomes a migration debt the size of the package.

The `graph` sibling next door has a *cleaner* type-level discipline (typed `IntAdjacency` adjacency-list, named `Edge` alias, dedicated `dijkstraItem` priority-queue node, `prev []int` predecessor convention) which `queue` would benefit from copying — but more importantly, `graph` ships **no** routing-matrix-as-`[][]float64` API anywhere, instead using sparse `IntAdjacency` + edge weight maps. `JacksonNetwork`'s dense `routing [][]float64` argument is the *opposite* of how `graph` shapes the same problem.

---

## 1. Open vs closed networks

### Current state
`network.go:44` exports exactly one network function:
```go
func JacksonNetwork(
    lambdaExt []float64,
    routing [][]float64,
    mu []float64,
    servers []int,
) (throughput, utilization, queueLength []float64)
```
This is **open** Jackson — external arrivals `lambdaExt` and routing-out probabilities `< 1` are the only way to specify the network. There is no parameter `N` for fixed customer population. A user wanting closed-network analysis has no entry point at all.

### Problem
122 T1.9 will add closed Jackson via Mean Value Analysis (MVA, Reiser-Lavenberg). That function will have a fundamentally different signature:
```go
func ClosedJacksonMVA(
    visits []float64,           // V_i, relative visit ratios
    serviceTime []float64,      // 1/μ_i
    N int,                      // fixed customer population
) (X, Q, R, U []float64)        // throughput, queue length, response, utilization
```
The two APIs have:
- **Disjoint inputs:** open takes `lambdaExt` (arrival rates, dimension 1/time); closed takes `visits` (visit ratios, dimensionless) and `N` (count, dimensionless).
- **Disjoint outputs:** open returns *steady-state* utilization including possible >1 panic for overload; closed always-stable (finite N, no overload possible) and *throughput* `X` is the primary output (not arrival rate, which is derived via Little).
- **Different "at-capacity" semantics:** open panics if any `ρ_i ≥ 1`; closed asymptotes to `X(N) → min_i (1/(V_i·s_i))` as `N → ∞` and can't fail.

Two functions, two contracts. Fine — but they need a *naming* and *organising* convention.

### Comparison with sibling `graph`
`graph/` has the *exact same* "two structurally-disjoint network problems" pattern: `Dijkstra` (single-source shortest path) vs `FloydWarshall` (all-pairs). It solves it by:
- Two top-level functions, both prefixed by their algorithm name (`Dijkstra`, `FloydWarshall`).
- Different signatures embracing the input-shape difference: `Dijkstra(adj, weights, source) → (dist, prev)` vs `FloydWarshall(n, edges) → distMatrix`.
- No shared "Path" supertype trying to unify them.

Recommend `queue` follow the same pattern — keep `JacksonNetwork` as the open variant (matching `Dijkstra`'s posture as the more-commonly-invoked primitive), add `ClosedJacksonMVA` as a peer rather than as `JacksonNetwork(closed=true, N=...)`. **Do not** introduce a unifying `Network` type; the inputs are too different.

### Naming proposal
Rename in v0.11 (or alias for v1):
| Today | Proposed | Rationale |
|---|---|---|
| `JacksonNetwork` | `OpenJacksonNetwork` | makes the open/closed axis explicit |
| (missing) | `ClosedJacksonMVA` | Reiser-Lavenberg single-class exact MVA |
| (missing) | `ClosedJacksonBardSchweitzer` | approximate MVA for large N |
| (missing) | `BCMPNetwork` | open multi-class product-form (122 T2.11) |

Short alias `Jackson()` is tempting but misleading: in textbooks "Jackson network" *defaults* to open (1957 paper), so `JacksonNetwork → OpenJacksonNetwork` is a defensible explicit-rename even though it's six characters more.

### Open question — should `OpenJacksonNetwork` accept finite-buffer nodes?
The current implementation hardcodes M/M/c per node (infinite buffer). 122 T1.1 adds M/M/c/K. If a per-node `K []int` argument lands, the network becomes mixed-blocking, which is *no longer* product-form Jackson — it's manufacturing-blocking territory and there is no closed-form solution. **Do not** add a `K []int` param to `OpenJacksonNetwork`; require the user to call `BCMPNetwork` (which is *also* not closed-form for finite-K, but at least the name doesn't lie about it) or move to the future DES module from 123.

---

## 2. Steady-state vs transient

### Current state
**Every** function in the package is steady-state. Five locations explicitly require `λ < cμ` (or `A < N` for ErlangC) and panic otherwise:
- `basic.go:64` `MM1`: `lambda >= mu` panics.
- `basic.go:120` `MMc`: `lambda >= c*mu` panics.
- `erlang.go:79` `ErlangC`: `A >= N` panics.
- `network.go:127` `JacksonNetwork`: any node `utilization >= 1` panics.

The package docstring says "queueing theory models and metrics" without saying "steady-state". Functions like `MM1` document `1/(μ-λ)` *as if it were the wait time*, when it is the *limiting* mean wait time as `t → ∞`. For `t` smaller than the relaxation time `τ ≈ 1/(μ−λ)·...`, the actual wait is significantly different — and in transient regimes near `ρ → 1`, the steady-state value is unreachable in any reasonable horizon.

### Why this matters
The Pulse/Sentinel consumer named in `basic.go:5` is a *capacity planning* tool that observes traffic over rolling windows. During a load-spike — exactly when the analysis is most useful — `λ` may briefly exceed `μ`, the system has *not* reached steady state, and the package will literally panic. The right answer is "the queue grew by `(λ − μ)·t` during the spike and will drain over `~τ_drain = (queue_at_end)/(μ − λ_post)` once load subsides", which requires *transient* analysis.

### What's needed (signatures)
```go
// MM1Transient computes P(N(t) = k | N(0) = n0) for an M/M/1 queue.
// Even ρ ≥ 1 is allowed (overloaded transient is a valid query).
//
// Returns a length-(kMax+1) vector of probabilities. Uses the modified
// Bessel function of the first kind I_n(z) per Sharma & Gupta 1982 form
// (closed-form transient distribution).
func MM1Transient(lambda, mu float64, n0, kMax int, t float64) (probs []float64)

// MM1ExpectedTransient computes E[N(t) | N(0) = n0]. Exact closed form.
func MM1ExpectedTransient(lambda, mu float64, n0 float64, t float64) (Lt float64)

// MM1RelaxationTime returns the e-folding time τ such that
// |E[N(t)] - L_∞| ≈ |E[N(0)] - L_∞| · exp(-t/τ). Equals 1/((√μ - √λ)²)
// for stable M/M/1 (Morse 1955). Diverges at ρ = 1.
func MM1RelaxationTime(lambda, mu float64) float64

// MMcTransient via uniformization (Grassmann 1977). General-purpose,
// works for any CTMC with bounded rates; M/M/c is a special case.
// Uses an internal truncation at N_trunc such that P(N(t) > N_trunc) < tol.
func MMcTransient(lambda, mu float64, c, n0, kMax int, t float64) (probs []float64)
```

The transient API has *fundamentally different* invariants than steady-state:
- **No stability gate** — overloaded transient queries are valid.
- **Returns a distribution, not a moment** — for ρ near 1 the variance is large enough that the mean is uninformative.
- **Time argument** — every signature has explicit `t`.
- **Initial-condition argument** — `n0` is mandatory.

The natural API split is *two parallel namespaces*: `queue.MM1` (steady) and `queue.MM1Transient` (time-dependent). Do **not** unify into `queue.MM1(t = math.Inf(1) for steady)` — this conflates two genuinely different mathematical objects (a fixed point of the master equation vs the master equation itself) and forces users to understand uniformization to call the simple steady-state.

### Naming convention
Adopt the suffix `-Transient` for the time-dependent variant of any steady-state function. Mirrors `Erlang*WaitTime` already in the package (which is *also* a steady-state quantity, but at least the name is explicit).

### Relaxation time as a first-class export
`MM1RelaxationTime`, `MMcRelaxationTime` should be exported peers — they answer "is the steady-state result trustworthy at this observation window?" which is the *first* question every consumer should ask before reading `MM1`'s output. None of the textbook-canonical libraries (JMT/LINE/queueing-tool) export this directly; it would be a small differentiating contribution.

---

## 3. Result struct: tuple vs explicit named fields

### Current state — five-tuple positional returns
```go
func MM1(lambda, mu float64) (Lq, Wq, L, W, rho float64)
func MMc(lambda, mu float64, c int) (Lq, Wq, L, W, rho float64)
func MM1K(lambda, mu float64, K int) (Lq, Wq, L, W, rho, pLoss float64)
```
Three return-shape variants already, three different position contracts. `MM1K` adds `pLoss` at position 5 (after `rho`); MMcc would need to add `pLoss` too but at the *same* position, except `MMcc` has `Lq=0, Wq=0` always so the field is degenerate.

### Why tuples are the wrong choice
Five reasons, ranked:

1. **Adding a field is a breaking change.** Go's named-returns are *positional* at the call site. `lq, wq, l, w, rho := MM1(...)` cannot pick up a new sixth return without compile error. Every Tier 1 addition from 122 wants at least one new field — `pLoss` (M/M/c/c, M/M/c/K), `lambdaEff` (M/M/c/K), `Cs2` echoed in result (M/G/1), `varianceLq` (M/G/1 P-K extended), `relaxationTau` — and there's no room.

2. **Field-shape varies between functions but tuple-shape is shared.** `MM1` has 5 outputs, `MM1K` has 6, future M/G/1 has 5+, M/M/c/c has 4 meaningful + 2 zero, MVA has 4 vectors per node. Tuples force the user to remember *which* function has *which* extras at *which* position.

3. **Cross-language portability.** CLAUDE.md's golden-file infrastructure validates Go output against Python/C++/C#. Python returns dicts, C# returns records, C++ returns structs; all three port naturally to a Go *struct*. None ports naturally to a Go positional tuple. The current `MM1` golden file at `testdata/queue/mm1.json` already lists `[Lq, Wq, L, W, rho]` as a positional array (`queue_test.go:111`), which means the *golden format* is already infected by Go's tuple choice — adding a field requires regenerating every golden across every language.

4. **Zero-cost abstraction in Go.** A struct return is zero-allocation when it fits in registers (Go ABI passes ≤2-word structs in regs as of Go 1.17). The proposed `Metrics` struct is 6 × float64 = 48 bytes — passed via stack, *one* memcpy on return, identical perf to tuple-returns when escape-analysis sees the value not heap-promoted. **The perf cost is zero for Pistachio's 60FPS path.**

5. **Field names beat position on readability.** `m.Wq` reads better than `_, wq, _, _, _ := MM1(...)`, especially when the caller wants only one or two fields.

### Recommendation — `Metrics` struct
```go
// Metrics is the canonical steady-state output of all single-station
// analytical queueing functions in this package. Fields are NaN when
// not applicable to the model (e.g., PLoss is NaN for infinite-capacity).
type Metrics struct {
    L     float64 // expected number in system
    Lq    float64 // expected number in queue
    W     float64 // expected time in system
    Wq    float64 // expected wait time in queue
    Rho   float64 // utilization (per-server for multi-server)
    PLoss float64 // probability of arrival loss; NaN if infinite capacity
}

func MM1(lambda, mu float64) Metrics
func MMc(lambda, mu float64, c int) Metrics
func MM1K(lambda, mu float64, K int) Metrics
func MMcc(lambda, mu float64, c int) Metrics            // 122 T1.2
func MMcK(lambda, mu float64, c, K int) Metrics         // 122 T1.1
func MG1(lambda, meanS, c2s float64) Metrics            // 122 T1.3
func MD1(lambda, meanS float64) Metrics                 // 122 T1.4
func MMinf(lambda, mu float64) Metrics                  // 122 T1.5
func KingmanGG1(lambda, mu, c2a, c2s float64) Metrics   // 122 T1.6
```

**Pros:**
- One struct, nine consumers. Forwards-compatible: adding `LambdaEff`, `RelaxationTau`, `Cs2Departure` is non-breaking.
- Field-not-applicable conventions consistent: NaN for unmodelled fields. Matches IEEE 754 semantics for "no-value" without sentinels like `-1`.
- Golden files become objects, not arrays: `{"L": 1.0, "Lq": 0.5, ...}` ports cleanly to Python `dict`/C#/C++.
- Single-line consume:
  ```go
  m := queue.MM1(1, 2)
  fmt.Println(m.W)
  ```
- One field at a time:
  ```go
  if queue.MMc(λ, μ, c).PLoss > 0.01 { /* alert */ }
  ```

**Cons:**
- Migration: every existing call site changes from `_, wq, _, _, _ := MM1(...)` to `m := MM1(...); m.Wq`. Manageable; 8 functions × ~5 call sites in the test suite is one afternoon.
- Cross-language port has to learn Metrics — but they're all object-oriented anyway; this is a *gain* over teaching them the position contract.

### Vector / network-shaped return
For `OpenJacksonNetwork` and `ClosedJacksonMVA`, return per-node *slices* of the same `Metrics` struct:
```go
type NetworkMetrics struct {
    Throughput  []float64
    Nodes       []Metrics
}

func OpenJacksonNetwork(...) NetworkMetrics
func ClosedJacksonMVA(...) NetworkMetrics
```
The field name `Nodes` matches the topology language; `node[i].W` reads exactly. Avoids the current parallel-slice convention `(throughput, utilization, queueLength []float64)` which has the same forwards-compatibility weakness as the tuple problem above.

### Erlang B/C scalar returns — leave as-is
`ErlangB(A, N) float64` and `ErlangC(A, N) float64` are *scalars by definition*. Wrapping them in a struct makes no sense. `ErlangCWaitTime` returns `Wq` only; if a future caller wants the full M/M/c metrics they should call `MMc` or a new `MMccWaitMetrics` (122 T1.2 already proposes this as `MMcc`). **Don't rewrap mathematically-scalar quantities.**

---

## 4. Network topology — matrix vs graph

### Current state
`OpenJacksonNetwork` takes `routing [][]float64` — a dense n×n routing-probability matrix. For a 100-node sparse pipeline (typical microservices topology) this is 80 KB of mostly-zero memory, and the row-sum validation (`network.go:62-72`) is O(n²) per call.

### Sibling `graph` does the opposite
`graph/types.go:7` defines:
```go
type IntAdjacency = map[int][]int
```
Plus shortest-path functions in `graph/shortest.go` accept:
```go
func Dijkstra(adj IntAdjacency, weights map[[2]int]float64, source int)
```
Sparse-by-default. Edge weights as `map[[2]int]float64`, where missing entries mean "no edge". The same primitives — adjacency list + edge-weight map — are *exactly* what queueing-network routing wants:
- Adjacency list: which nodes are reachable from node `i`.
- Edge weights ∈ [0, 1]: routing probabilities `P[i→j]`.

123 § 3 already named this gap: queueing-tool (the Python sibling) uses NetworkX graphs with edge weights = routing probabilities, sparse by default. The reality `graph` package has the same structure. The reality `queue` package does not consume it.

### Recommendation — graph-overload entry point
Add an alternate entry point that accepts the existing reality `graph` types:
```go
import (
    "github.com/davly/reality/graph"
    "github.com/davly/reality/queue"
)

func OpenJacksonNetworkGraph(
    adj graph.IntAdjacency,
    routing map[[2]int]float64,   // routing[[i,j]] = P(node i → node j)
    lambdaExt []float64,
    mu []float64,
    servers []int,
) NetworkMetrics
```

**Implementation:** convert sparse → dense internally (~10 LOC), then call existing `OpenJacksonNetwork`. Total ~30 LOC of adapter, breaks no existing API. Validates row-sums in O(|E|) not O(n²).

**Why this matters for BCMP:** 122 T2.11 lands BCMP networks. BCMP has *per-class* routing, which means `routing` becomes `map[[3]int]float64` keyed by `[class, src, dst]` — sparse-natural, dense-disaster. Getting the graph-overload pattern in place for plain Jackson now sets the precedent for BCMP.

### What about replacing the dense form entirely?
**Don't.** The dense form is correct for fully-connected networks and matches the textbook notation Jackson 1957 uses. Two entry points (dense matrix + graph adjacency) is the right answer; mirrors how `graph` ships both `IntAdjacency` and `[][3]float64` edge-list APIs (`graph/shortest.go:144` `FloydWarshall(n, edges [][3]float64)`).

---

## 5. Single-class vs multi-class

### Current state — single-class only
Every function assumes **all customers are identical**. There is no `class` parameter anywhere. The `routing` matrix is `n × n`, not `K × n × n` (K classes × routing per class).

### What multi-class needs
Multi-class queueing is not just "K independent single-class problems":

1. **Per-class routing.** Customer of class `k` at node `i` goes to node `j` with probability `P_k[i,j]`, possibly *changing class* en route (`P_k,k'[i,j]` for class transitions).
2. **Per-class service rates.** Class `k` at node `i` has service rate `μ_{k,i}`. For BCMP-FCFS this must be class-independent (a constraint on the model); for BCMP-PS it can vary.
3. **Per-class arrivals.** Open networks: `λ_ext` becomes `λ_ext[k, i]`. Closed networks: `N` becomes `N[k]` (population vector).
4. **Per-class metrics.** `Q_{k,i}`, `W_{k,i}`, `X_{k,i}` etc. The `Metrics` struct goes 2-D.

122 T2.2/T2.3 (priorities) and T2.11 (BCMP) are both multi-class. They will need either:
- A second *parallel* API surface (`MM1Priority(...) []Metrics`, `BCMPNetwork(...) [][]Metrics`), or
- A class-parametric extension to the existing API.

### Recommendation — separate API for multi-class
Multi-class is a *different API surface*. Consumers who want single-class should not pay the multi-class price (extra dimensions, vector returns, harder-to-validate inputs). Consumers who want multi-class should not be forced to pretend `K=1` to call the single-class form.

Convention:
```go
// Single-class (today's API + 122 Tier 1):
func MM1(lambda, mu float64) Metrics
func MG1(lambda, meanS, c2s float64) Metrics
func OpenJacksonNetwork(...) NetworkMetrics
func ClosedJacksonMVA(visits []float64, s []float64, N int) NetworkMetrics

// Multi-class peers (future, 122 Tier 2-3):
type ClassMetrics struct {
    Class int
    Metrics            // embedded; Class-specific L/Lq/W/Wq/Rho
}
type MultiClassNetworkMetrics struct {
    ClassMetrics [][]ClassMetrics  // [class][node]
    Throughput   [][]float64       // X[class][node]
}
func MM1Priorities(lambdas, mus []float64) []ClassMetrics              // T2.2
func BCMPOpenNetwork(...) MultiClassNetworkMetrics                     // T2.11
func ClosedJacksonMVAMultiClass(...) MultiClassNetworkMetrics
```

Naming suffixes:
- `-Priorities` — multi-class with priority (Cobham, Kleinrock).
- `-MultiClass` — multi-class without priority (BCMP-style, service-discipline-determined).
- No suffix → single-class.

### Migration path
None of the multi-class API needs to land *now*. Slot 124's recommendation is to **not pre-emptively design the multi-class interface into the single-class signatures**. Specifically:
- Keep `MM1`, `MMc`, `MM1K` returning *scalar* `Metrics`, not `[]Metrics`. A single-class user gets `metrics.W` directly.
- When `MMcPriorities` lands, its return type is a *new* `[]ClassMetrics`, not a generalisation of `MMc`'s type.
- Multi-class users compose the two: call `MMc(λ_total, μ, c)` for the unsegmented baseline, then `MMcPriorities(...)` for the per-class breakdown.

This is the pattern `signal/` follows for windowed FFT (`FFT()` is single-frame; `STFT()` is the multi-frame peer, not a generalisation of `FFT(window=1)`).

---

## 6. Comparison with sibling `graph/` patterns

### Six `graph/` patterns worth importing

1. **Typed adjacency alias** (`graph/types.go:7`: `type IntAdjacency = map[int][]int`). `queue` should alias `type RoutingMatrix = [][]float64` and `type SparseRouting = map[[2]int]float64` — self-documenting at call sites, no runtime cost.
2. **Edge type alias** (`graph/graph.go:13`: `type Edge = [2]string`). Queue equivalent: `type RoutingEdge struct { Src, Dst int; Prob float64 }` — natural input shape for `OpenJacksonNetworkFromEdges([]RoutingEdge, ...)`.
3. **Internal helpers as named types** (`graph/shortest.go:187-204`: `dijkstraItem`, `dijkstraHeap`). `queue` currently has zero internal data types. When uniformization/MAM/MVA recursion lands, workspace structures (`mvaState`, `uniformizerState`) should be lowercase package-private types with explicit field names.
4. **Predecessor / trace-mode returns** (`graph/shortest.go:55` returns `prev []int`). Queue parallel: `OpenJacksonNetwork` could return `iterations int` for diagnostics; `ClosedJacksonMVA` should return per-population-step responses `R_i(n)` for `n = 1..N`, not just the final `n = N` — caller can derive utilisation curves cheaply.
5. **Function-parameter pluggability** (`graph/shortest.go:83` `AStar(... heuristic func(int) float64)`). Queue analogue: time-varying arrivals via `func MM1TransientArbitraryArrival(arrivalRate func(t float64) float64, mu, n0, t float64) (Lt float64)`. Function-parameter is the right shape for M_t/G/c (122 T3.10); struct-parameter is wrong.
6. **Heap-based priority queue** (`graph/shortest.go:184-204` `dijkstraHeap`). The DES module from 123 needs exactly this — a `(time, priority, monotonic_id)` heap. Do **not** write a second heap in `queue/sim`; extract or copy verbatim.

### Three `graph/` anti-patterns to avoid

1. **Function-name overloading by shape** (`Dijkstra` vs `FloydWarshall` share *no* signature commonality). `queue` should prefer suffix discipline (`MM1`, `MM1K`, `MM1Priorities` — model-prefix shared, suffix varies). This is what 122 already proposes implicitly.
2. **Mixed string and integer node labels** (`graph/graph.go:14` Edge=[2]string vs `graph/types.go` IntAdjacency). `graph` carries two label conventions, forcing converters. Pick `int` once for `queue`; consumers wanting string labels wrap with `map[string]int` at the call site.
3. **Module-internal `graphSize` max-index heuristic** (`graph/shortest.go:212`) is fragile for sparse high-index inputs. `queue` already requires explicit `len(lambdaExt)=n` sizing; keep this — explicit sizing is correct for fixed-topology queueing networks.

---

## 7. Migration plan

Four small PRs ordered by leverage / lowest-risk-first:

1. **`Metrics` struct (~150 LOC, breaking, v0.11)** — convert `MM1`/`MMc`/`MM1K` to return `Metrics{L,Lq,W,Wq,Rho,PLoss}`. Migrate golden files to object form. Doing this *after* T1.3 lands forces nine migrations instead of three.
2. **`OpenJacksonNetworkGraph` overload (~30 LOC, non-breaking)** — adapter consuming `(graph.IntAdjacency, map[[2]int]float64)`, forwards to existing dense solve, returns `NetworkMetrics`. Establishes queue↔graph bridge before BCMP makes it mandatory.
3. **Rename `JacksonNetwork → OpenJacksonNetwork` (deprecation alias)** — reserves namespace for `ClosedJacksonMVA` (122 T1.9) and `BCMPNetwork` (122 T2.11).
4. **`MM1Transient` + `MM1RelaxationTime` (~80 LOC, non-breaking)** — closed-form Bessel transient + relaxation time. Establishes `-Transient` suffix convention; uniformization extensions build on top.

Total ~260 LOC additive. Closes every ergonomic gap; package becomes forwards-compatible with 122 Tier 1 + Tier 2 without further API churn. Multi-class convention (Step 5 in the larger plan) is documented-only — reserves `-Priorities`/`-MultiClass` suffixes and `[]ClassMetrics`/`MultiClassNetworkMetrics` return types so 122 T2.2/T2.11 implementers don't invent competing conventions.

---

## 8. Files cited

- `C:/limitless/foundation/reality/queue/basic.go:56` (`MM1` — tuple return)
- `C:/limitless/foundation/reality/queue/basic.go:108` (`MMc` — tuple return)
- `C:/limitless/foundation/reality/queue/basic.go:173` (`MM1K` — extended tuple)
- `C:/limitless/foundation/reality/queue/basic.go:235` (`LittlesLaw` — scalar)
- `C:/limitless/foundation/reality/queue/erlang.go:34` (`ErlangB` — scalar, correct as-is)
- `C:/limitless/foundation/reality/queue/erlang.go:70` (`ErlangC` — scalar)
- `C:/limitless/foundation/reality/queue/erlang.go:100` (`ErlangCWaitTime` — scalar)
- `C:/limitless/foundation/reality/queue/network.go:44` (`JacksonNetwork` — open-only, dense routing)
- `C:/limitless/foundation/reality/queue/metrics.go:26` (`BurstinessIndex`, `OfferedLoad` — scalar)
- `C:/limitless/foundation/reality/queue/queue_test.go:111` (golden-file positional contract: `[Lq, Wq, L, W, rho]`)
- `C:/limitless/foundation/reality/graph/types.go:7` (`IntAdjacency` typed alias — copy-pattern)
- `C:/limitless/foundation/reality/graph/graph.go:13` (`Edge = [2]string` — convention-pattern)
- `C:/limitless/foundation/reality/graph/shortest.go:29` (`Dijkstra` signature — sparse-routing pattern)
- `C:/limitless/foundation/reality/graph/shortest.go:144` (`FloydWarshall` — dense-edge alternative entry pattern)
- `C:/limitless/foundation/reality/graph/shortest.go:187-204` (`dijkstraHeap` — heap pattern for future DES module)

---

## Bottom line

`reality/queue` has eight functions and three API decisions baked in that should not survive contact with the missing-primitives roster from 122/123: **positional-tuple returns** (no room for new fields), **steady-state-only signatures with no transient peer** (panics during exactly the load-spike scenarios that motivate Pulse/Sentinel using the package), and **dense routing matrices** (the opposite of how the sibling `graph` package shapes the same problem). Open-vs-closed network distinction is unaddressed — no `N` parameter, no MVA entry point, no naming convention reserving `OpenJacksonNetwork` vs `ClosedJacksonMVA`. Single-vs-multi-class is similarly unaddressed; recommend *not* generalising single-class signatures and instead reserving the `-Priorities`/`-MultiClass` suffix space for future peer functions. Four small PRs (~260 LOC) — `Metrics` struct, graph-overload Jackson, open-rename, transient pair — close every ergonomic gap and make the 122 Tier 1 / Tier 2 additions land cleanly. The single-most-leverage item is the `Metrics` struct: it's the foundation every other function from 122 inherits, and migrating later means migrating nine functions instead of three.

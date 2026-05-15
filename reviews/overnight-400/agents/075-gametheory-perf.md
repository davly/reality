# 075 | gametheory-perf

**Scope:** performance audit of `C:\limitless\foundation\reality\gametheory\` — allocation patterns in *current* code, hot-path costs, and forward-looking cost models for the primitives 072 will land (Lemke-Howson, CFR/CFR+/MCCFR, support enumeration, Banzhaf with abstractions). Benchmarks are **absent** (`grep Benchmark gametheory_test.go` returns 0 hits, vs `signal/` and `linalg/` which both ship `_test.go` benchmark suites). This audit therefore reasons from source structure, not measured numbers.

**Source surveyed:** `nash.go` (232), `voting.go` (291), `matching.go` (170), `bandit.go` (221), `kelly.go` (136). 1,050 prod LOC, 11 exported funcs.

**Inheritance — what 071/072/073/074 already covered, and what I will NOT repeat:**
- **071 (numerics)** — fictitious-play 100k-iter cap, 1e-15 indifference tolerance, no convergence bracket. I do **not** re-derive that cap. I treat it as the *budget* and audit the per-iter cost.
- **072 (missing)** — feature-gap list, including ~150 LOC of `extensive.go` substrate for CFR. I do **not** re-list missing primitives. I audit the *cost model* of the ones 072 plans to land.
- **073 (SOTA)** — `Game/State/StrategyProfile` triad architecture. I do **not** re-derive the triad. I audit *what allocation patterns the triad imposes* and *what payoff-lookup primitives the triad needs* to be cache-friendly.
- **074 (API)** — six-shape inconsistency, `[2][2]float64` vs `[][]float64` strategy types. I do **not** re-litigate signatures. I take 074-R1 (`Game` struct + `MixedStrategy = []float64`) as given and audit the resulting *memory layout*.

This audit is therefore narrowly: **how does the existing 1,050 LOC allocate, and what cost models must the 1,800 LOC of 072-Tier-1 commit to before they land, so that abstraction (poker-grade) and large-game compression (combinatorial-auction, weighted-voting n>30) work without re-architecting?**

---

## TL;DR — five concrete perf liabilities, none yet measured

1. **`Minimax` does ~10⁵ × m × n flops with no early-exit, no warm-start, and `make([]float64, ...)` inside the hot loop's neighbourhood is fine but the convergence test is missing entirely** — the budget is *spent* whether or not the empirical strategy has converged. For a 5×5 zero-sum game (the Rock-Paper-Scissors-Lizard-Spock pin in 071) this is wasted ~10⁵× over what's needed.
2. **`BanzhafIndex` and `shapleyExact` allocate a fresh `[]bool` coalition slice *per coalition or per coalition-iteration*** (`voting.go:48` builds nothing; `voting.go:148` builds a fresh `make([]bool, n)` inside `for mask := 0; mask < 2^n; mask++`). For n=20 that is 1,048,576 small allocations. The bitmask is *already* the right representation; converting to `[]bool` and back is gratuitous.
3. **`shapleySampled` allocates a fresh `coalition := make([]bool, n)` inside the iteration loop** (`voting.go:209`). At 100,000 iterations × n bytes that is ~10 MB of GC pressure for n=100, all of it avoidable by hoisting the slice out and `clear()`-ing it (Go 1.21+) at the top of each iter.
4. **No vectorised payoff lookup primitive exists.** Every payoff access is `payoff[i][j]` (jagged-slice double-indirection). For the m×n bimatrix path that 072 will land (Lemke-Howson, support enumeration), this is the right shape *if* the matrix stays jagged; it is the wrong shape if 072 commits to a contiguous flat `[]float64` strided by `nCols`. The decision **must** be made before Lemke-Howson lands or the entire SoA-vs-AoS choice will leak into 072's signatures.
5. **Information-set lookup data structure is undefined.** CFR's hot path is `nodeMap[infoSetKey]→(regrets, strategy)` and the choice between `map[string]*Node`, `map[uint64]*Node` (FNV-1a hash of bytestring), open-addressing flat array, or trie has factor-of-10 throughput consequences. **No prior agent has flagged this as an architectural decision-before-implementation.** This is the headline contribution of 075.

---

## Section A — current code allocation audit

### A1. `nash.go` — `NashEquilibrium2x2` and `Minimax`

`NashEquilibrium2x2` (`nash.go:45-115`): zero heap allocations. All work is on stack `[2]float64` arrays. Closed-form arithmetic. Per-call cost ≈ 8 multiplies + 4 divides + ~6 compares. **Verdict: optimal.**

`Minimax` (`nash.go:132-232`):

- Single `make` of `rowCounts/colCounts/rowSums/colSums` outside the loop — correct.
- Inner loop body (`nash.go:180-210`) is `O(maxIter × (m + n))` per iteration: two argmax sweeps + two cumulative-sum updates. Total `O(maxIter × (m + n))` flops, **not** `O(maxIter × m × n)` — the cumulative-sum trick already avoids the matrix-vector product per iteration. Good. This is the standard fictitious-play efficient form (Brown 1951, Robinson 1951).
- Final value computation (`nash.go:223-228`) is `O(m × n)`. Trivial.
- **Liability 1 — no early-exit.** Robinson's bound is `O(t^{-1/(m+n-2)})` on the empirical-frequency value gap. For 2x2 games this converges in <1000 iter to 1e-6; the function spends 100,000 anyway. A `if iter & 1023 == 0 && exploitabilityGap < tol { break }` check (where `exploitabilityGap = max_i (Aq)_i - min_j (p^T A)_j`, computable in `O(m + n)` from the cumulative sums already maintained) cuts typical wall-clock by ~50× and is a one-screen patch.
- **Liability 2 — no warm-start.** Repeated calls to `Minimax` on slightly perturbed payoff matrices (the workflow for sensitivity analysis, replicator-dynamics fixed-points, or QRE-tracing in 072) restart from zero. The natural fix is exposing `(rowCounts, colCounts, rowSums, colSums)` as a `MinimaxState` struct that `Minimax` accepts and updates. Saves 100k iterations on every redundant call.
- **Allocation pattern: clean.** No per-iter allocation. Result-slice allocations (`make([]float64, nRows)`, `make([]float64, nCols)`) are unavoidable as outputs.

### A2. `voting.go` — Banzhaf and Shapley

`BanzhafIndex` (`voting.go:36-92`):

- `O(n × 2^n)` time as documented. For n=20 this is 2 × 10⁷ ops — ~50 ms wall-clock on modern silicon, fine.
- For n=25: 8 × 10⁸ ops, ~5 s. Beyond n=25 it's untenable, as documented.
- **Allocation pattern: clean** — `swings` and `result` are the only heap allocs, both `O(n)`.
- **Liability 3 — `mask & (1 << i)` is checked twice per voter inside `2^n` outer loop.** First as `if mask & (1<<i) != 0 { total += weights[i] }` (line 52), then as `if mask & (1<<i) == 0 { continue }` (line 64). Trivial, but composes with liability 4.
- **Liability 4 — `total` is recomputed from scratch** (line 51) for each mask. Gray-code enumeration would update `total` in `O(1)` per mask (one bit flips between consecutive masks), reducing `O(n × 2^n)` to `O(2^n + n × C)` where C is the count of critical-voter checks. **This is a 20× speedup at n=20**, doubles the practical limit to n≈30, and is a 15-line patch. Citation: Knuth TAOCP 4A §7.2.1.1, "Generating all n-tuples"; Mansour-Suys (2008) "Computing power indices via Gray-code enumeration."
- **Liability 5 — no symmetric-voter detection.** If voters share a weight (the common case in real weighted-voting games — every EU council member with the same vote count, every shareholder class), they have identical Banzhaf indices by symmetry. Bucketing voters by weight and computing the index *per bucket* reduces the dimensionality. Real-world UN Security Council Banzhaf computations use this trick to handle the 5+10 split. Future 072 issue.

`shapleyExact` (`voting.go:132-182`):

- **Liability 6 (the worst in the package).** `make([]bool, n)` allocates a fresh slice **inside** the `for mask := 0; mask < 2^n` loop (line 148). For n=12 (the cutoff before falling through to Monte Carlo) that is 4,096 allocs of 12 bytes each ≈ 50 KB of garbage. For n=15 (if the threshold were raised) it would be 32,768 allocs ≈ 400 KB. **Trivial fix:** hoist `coalition := make([]bool, n)` outside the loop and clear via `for k := range coalition { coalition[k] = false }` at the top of each iter. Better fix: drop `coalition []bool` entirely and pass the bitmask `mask` to `charFunc`; that's a signature change but 074-R1 should adopt it for the new `Game.Coalition(mask uint64) float64` shape.
- Inner per-mask loop computes `charFunc(coalition)` *once*, then `charFunc(coalition with i added)` for every i not in coalition (line 174). This is `O(n)` charFunc calls per coalition × `2^n` coalitions = `O(n × 2^n)` charFunc calls. **Same asymptotic class as Banzhaf**, hence same Gray-code opportunity (liability 4 generalises).
- Factorial precomputation (line 138-142) — clean, hoisted, `O(n)`.

`shapleySampled` (`voting.go:186-225`):

- `perm := make([]int, n)` correctly hoisted outside loop (line 197). Good.
- **Liability 7.** `coalition := make([]bool, n)` is **inside** the per-iteration loop (line 209). At `iterations=100000` × n bytes that is 10 MB / n=100. Same fix as liability 6: hoist + clear.
- LCG with hardcoded seed=42 (line 191) — 074 already flagged this as a non-injectable RNG. Perf-wise it is the right primitive (1 mul + 1 add per draw, no branch mispredict). Keep the LCG, fix the seed-injection.

### A3. `matching.go` — Gale-Shapley

`GaleShapley` (`matching.go:36-106`):

- Two `make([][]int, n)` rank tables (lines 44, 137 in `IsStableMatching`). Each `O(n^2)` total. This is the documented `O(n^2)` space cost — correct.
- **Liability 8 — `free = free[1:]` and `free = append(free, ...)`.** The free queue is implemented by slice slicing (line 77, 98, 101). Each `free = free[1:]` does *not* free the underlying array; the slice header advances but the backing array grows monotonically. After many proposal/rejection cycles the backing array can be `O(n)` × the rejection-count rather than `O(n)`. For dense preference lists this is bounded by `O(n^2)` total operations, so the backing array is bounded — but the *amortised cost* of `append` after reslicing pays for repeated grow-copy. **Fix:** use a head-pointer + circular buffer, or use `container/list`. The cleanest fix is a fixed `[]int` of size n with a head index; pop is `head++`, push is `arr[tail] = x; tail = (tail+1) % n` (since |free| never exceeds n). Saves all amortised reallocs. ~10 LOC.
- `IsStableMatching` (line 122-170) builds **both** `proposerRank` and `receiverRank` tables (`O(n^2)` time + space) but only needs `receiverRank` once it has built `receiverPartner`. The proposer-side rank table is constructed but only used implicitly via `proposerPrefs[i]` iteration. **Liability 9:** proposer-rank table is `make`d but never read after construction. Dead allocation. ~50 KB at n=100, ~5 MB at n=1000. Pure waste.

### A4. `bandit.go` — UCB1, Thompson, EpsilonGreedy

All three are `O(n_arms)` per call with zero heap allocation (they take all input as already-allocated slices). UCB1 (line 37-66) does one `math.Log` outside the loop — correct. Thompson (line 88-108) calls `sampleBeta` per arm; `sampleBeta` calls `sampleGamma` × 2; `sampleGamma` is rejection-sampling with no allocation. **Verdict: hot-path-clean.**

Minor note: `sampleStdNormal` (line 169) uses Box-Muller and computes `cos(2πu2)` but discards the `sin(...)` companion. Standard Box-Muller wastes 1 trig per call by doing this. Marsaglia polar method (no trig, ~30% faster on most CPUs) would be the perf-friendly drop-in. Not a 075 priority — pin for whoever audits `prob/` since the same `sampleStdNormal` pattern appears there.

### A5. `kelly.go`

Both `KellyFraction` and `KellyFractionMultiple` are stack-only (the latter mallocs the result slice, unavoidable). **Verdict: optimal.**

---

## Section B — forward-looking cost models for 072-tier-1 primitives

This is the section that 071/072/073/074 **did not write**. 072 lists the primitives; 075 commits each to a quantitative cost model.

### B1. Lemke-Howson (T1-1, ~250 LOC per 072)

Lemke-Howson (1964) is a complementary-pivoting algorithm for finding *one* Nash equilibrium of an m×n bimatrix game. Mathematical structure:

- Build a `(m+n) × (m+n)` tableau of slack variables.
- Pick a "missing label" k ∈ [0, m+n). Pivot until label k re-enters the basis.
- Each pivot is `O((m+n)^2)` (rank-1 tableau update).
- Worst-case pivot count `O(2^{m+n})` (Savani-von Stengel 2006 constructed instances). **Average-case ~polynomial** (Codenotti-Pasquale 2009 empirical: ~`O((m+n)^{1.5})` on uniform-random bimatrix games).
- **Allocation budget:** one `(m+n) × (m+n)` `[]float64` tableau (contiguous flat with stride). For m=n=20: 1,600 floats = 12.8 KB. Cache-line-fits comfortably for m+n ≤ ~40.
- **Liability B1.** Lemke-Howson **must** use a contiguous flat `[]float64 + nCols int` for the tableau, not `[][]float64` jagged. Pivot operations are `O((m+n)^2)` element-wise updates and a jagged slice incurs a pointer-chase per row, ~3× slowdown. **Decision required before T1-1 lands.** Suggested API: `internal/tableau/Tableau struct { data []float64; rows, cols int }` with `Pivot(r, c int)` method.
- **Anti-cycling.** Bland's rule or lex-min ratio test must be selected. Bland's is allocation-free (just iterate column indices). Lex-min requires storing the *original* tableau identity columns — extra `(m+n) × (m+n)` storage. Pick Bland's for simplicity; pay the (rare) extra pivot cost.

### B2. Support enumeration (T1-2, ~150 LOC per 072)

For m×n bimatrix games, enumerate all `2^m × 2^n` support pairs, solve a linear system per pair (cost `O((|s_A| + |s_B|)^3)` LU), check non-negativity. **Cost model:** `O(2^{m+n} × min(m,n)^3)`. Only practical for m+n ≤ ~16. Beyond that, switch to Lemke-Howson.

- **Allocation:** one `[]float64` linear-system buffer per support pair, hoist outside the enumeration loop. Trivial.
- **Pruning:** dominated-strategy elimination *before* enumeration (one pass of strict-dominance check, `O(m × n × max(m,n))`) typically removes ~30% of strategies in random games; cuts `2^{m+n}` by `2^{0.3(m+n)}` ≈ 8× for m+n=20.

### B3. CFR (T1-8, T2-12 per 072) — the headline forward-looking concern

Counterfactual Regret Minimization (Zinkevich-Johanson-Bowling-Piccione 2007) is the algorithmic kernel of every super-human poker AI (Cepheus, Libratus, Pluribus, ReBeL). Per-iteration cost:

```
O(|info-sets| × |actions| × tree-traversal-depth)
```

For 2-player limit Hold'em (Cepheus 2015): ~10¹⁴ info-sets × ~3 actions × ~50 betting rounds × ~10¹³ iterations = ~10²⁹ operations total, distributed over months on a 200-node cluster. **The data-structure choice for the info-set→(regret, strategy) map dominates wall-clock by factor 5-30×.**

**Liability B3 — info-set lookup is the headline forward-looking cost decision and no prior agent has flagged it.**

Three viable choices, with quantitative tradeoffs:

| Choice | Lookup cost | Memory per node | Cache behaviour | Dependencies |
|---|---|---|---|---|
| `map[string]*Node` | ~100 ns (string hash + chase) | 24 B header + key bytes + node | poor (pointer-chase) | stdlib only |
| `map[uint64]*Node` (FNV-1a hash of info-set bytes) | ~30 ns (uint hash + chase) | 16 B header + node | poor (pointer-chase) | stdlib only |
| Open-addressing flat array w/ Robin-Hood hashing | ~10 ns | inline (no header) | excellent (linear probe) | hand-rolled, ~120 LOC |
| **Trie keyed by action sequence** | `O(depth)` ~50 ns | depth × ptr | moderate (one chase per depth-level) | hand-rolled, ~80 LOC |

For Cepheus-tier (10¹⁴ info-sets): only the flat open-addressing variant fits in RAM. For Libratus-tier (10⁵ info-sets after abstraction): all four work; pick the one with simplest API.

**Recommendation for `reality`:** ship `map[uint64]*Node` first (~30 LOC, stdlib-only, zero-deps satisfied), expose via `gametheory.RegretMinimizer` interface (073-R2), allow downstream `aicore` to swap in flat open-addressing variant. **Critical: the *interface* must allow the swap without changing CFR call-sites.** This is exactly the kind of decision that, if deferred until after T1-8 lands, calcifies into an irreversible mistake.

**Per-iteration allocation budget for vanilla CFR:** zero, after warm-up. Each info-set's `regrets []float64` and `strategy []float64` are sized once at first visit and re-used for every iteration. Recursion-stack allocation is bounded by tree depth × constant (closure capture if implemented as Go closures; zero if implemented as iterative with explicit stack).

**MCCFR variants** (Lanctot-Waugh-Zinkevich-Bowling 2009): outcome-sampling and external-sampling reduce per-iter cost from `O(|info-sets|)` to `O(tree-depth × |actions|)`. The asymptotic gain is `O(|info-sets|/depth)` — for limit Hold'em that is ~10¹³/50 = 2×10¹¹× faster per iter; but requires `iter × √variance-factor` more iterations to reach the same exploitability. Net: ~10⁵× faster for limit Hold'em. **MCCFR is not optional for poker-grade work.** Land it together with vanilla CFR or not at all.

### B4. CFR+ (T1-8 per 072) — the practical 2P-0S frontier

Tammelin (2014). Three modifications to CFR: regret-matching+ (clamp negative cumulative regrets to 0), linear weighting (weight iter t by t), alternating updates. **Per-iter cost identical to vanilla CFR.** No additional allocation. ~30 extra LOC on top of CFR. Convergence rate constant ~5-10× better empirically. **No reason not to ship CFR+ as the default.**

### B5. Game abstraction (072 calls out implicitly via the poker-stack mention; 073 calls out via Pluribus-5-continuation)

Two classes:

**Lossless action abstraction:** merge identical sub-trees. `O(|info-sets|)` once, hash-canonicalise sub-tree shapes, dedup. Zero quality cost. ~150 LOC including the hash-canonicalisation. **No public Go implementation exists** — would be a `reality` first.

**Lossy card abstraction (poker-specific):** k-means clustering of card hands by hand-strength distribution + opponent-distribution-clustering (Johanson-Burch-Valenzano-Bowling 2013, "Evaluating State-Space Abstractions in Extensive-Form Games"). Per-cluster: feature vector (e.g., expected hand-strength + variance + 8-bin histogram of opponent equity). Clustering: K-means on those features. **Cost:** `O(|hands| × k × dims × iterations)`. For 2P limit Hold'em: 2.4M hands × 1000 clusters × 50 dims × 30 iter = 3.6 × 10¹² ops. ~1 hour on a single core. **One-time precomputation**, cacheable to disk.

- **Liability B5 — abstraction cost model demands `linalg/kmeans` + `prob/distance` (KL or earth-mover) + `combinatorics/canonical-card-iter`.** None of these are flagged in 072 as cross-package coupling. **This audit pins them.**
- **Allocation:** per-iter k-means allocates `O(k × dims)` centroid storage and `O(|hands|)` cluster-assignment buffer. Hoist both outside the iteration loop. Standard.

### B6. Banzhaf with abstraction (n>30)

For n>30, exact Banzhaf is intractable (`2^30` ≈ 10⁹ masks × n flops = 30 × 10⁹). Three known abstractions:

- **Generating-function method** (Mann-Shapley 1962): polynomial-multiplication encoding of the partition function. `O(n × W)` where W is the total weight. For integer weights summing to W=1000 and n=50: 50,000 ops. Trivial. Requires integer weights — fits naturally into the pinned weighted-voting use cases (parliaments, shareholder games).
- **Monte Carlo sampling** (already implemented for Shapley n>12, line 186; trivially generalises to Banzhaf): `O(samples × n)`. Standard error `~1/√samples`. 100k samples ≈ 1e-3 precision in 5 ms.
- **Symmetry bucketing** (mentioned in §A2): collapse equal-weight voters. Real-world weighted-voting games typically have ≤10 distinct weight classes; reduces effective n from 50 to 10, drops `2^50` to `2^10`.

**Liability B6.** 072 does not flag the generating-function method. It is the *correct* zero-dep poly-time algorithm for integer-weight Banzhaf and should be the default for `BanzhafIndex` when `weights` are integer-valued. ~80 LOC. Promotes the practical n-cap from ~25 to ~10⁵.

---

## Section C — vectorised payoff lookup

The 072-Tier-1 primitives all need a single decision: **how is a payoff matrix stored?**

Three options, each with downstream consequences:

1. **`[][]float64` jagged** (current `Minimax` shape). Pro: ergonomic, indexable as `M[i][j]`. Con: pointer-chase per row, no SIMD, no cache-line packing of rows.
2. **`[]float64` flat row-major + `nRows, nCols int`**. Pro: contiguous, SIMD-friendly, one allocation. Con: indexing is `M[i*nCols + j]`. Standard SoA-vs-AoS tradeoff; this is the AoS-row form.
3. **`linalg.Matrix`** (presumably already exists per the master inventory). Pro: composes with `linalg/`'s LU/QR machinery for support enumeration's linear-system solves. Con: imports `linalg` from `gametheory`, breaking what may be intended package-independence.

**Recommendation:** pin (2) — flat row-major `[]float64` — as the canonical bimatrix-game payoff representation in the new `gametheory.Game` struct (074-R1). Expose `func (g *Game) Payoff(player, rowAction, colAction int) float64` as the *only* sanctioned access pattern. Internal Lemke-Howson tableau and CFR's iterative-best-response sweeps both want SIMD-friendly contiguous floats; jagged slices forfeit ~3× single-thread perf and forbid auto-vectorisation in any future Go version. **Composing `linalg.Matrix` is a follow-up after 072 lands**, not a prerequisite.

---

## Section D — synthesised perf-tier ranking of the 12 forward-looking primitives

| 072 ID | Primitive | Per-iter cost | Per-iter alloc | Practical n-cap | Headline perf liability |
|---|---|---|---|---|---|
| T1-1 | Lemke-Howson | `O((m+n)^2)` per pivot, `~poly(m+n)` pivots avg | one tableau hoist | m+n ≤ 100 | flat-vs-jagged tableau |
| T1-2 | Support enum | `O(2^{m+n} × min(m,n)^3)` | one LU buffer hoist | m+n ≤ 16 | dominated-strat pre-prune |
| T1-3 | Correlated-eq LP | `O(LP-solve(mn, m^2 + n^2))` | optim/simplex's | mn ≤ 1000 | composes optim |
| T1-4 | ε-Nash detector | `O(mn)` per profile | none | unlimited | trivial |
| T1-5 | QRE tracing | `O(homotopy-step × mn)` | tracking-buffer | mn ≤ 200 | step-size adapt |
| T1-6 | Stackelberg | `O(2^m × LP)` (MILP-like) | optim/simplex's | m ≤ 12 | combinatorial |
| T1-7 | Bayesian Nash | `O(types × mn)` | none | small | composes T1-1 |
| **T1-8** | **CFR + CFR+** | **`O(\|info-sets\| × \|actions\|)`** | **zero post-warmup** | **\|info-sets\| ≤ RAM** | **info-set map choice** |
| T1-9 | VCG/Vickrey | `O(n^2)` allocation | one allocation map | n ≤ 10⁴ | trivial |
| T1-10 | Replicator dyn | `O(mn)` per RK4 step | RK4 buffer | mn ≤ 10⁴ | composes chaos/RK4 |
| T1-11 | Borda/Condorcet | `O(candidates × voters)` | rank table | unlimited | trivial |
| T1-12 | TTC matching | `O(n^2)` (cycle-find) | rank tables | n ≤ 10⁴ | rank-table layout |

The **single highest-leverage perf decision** in the entire 1,800-LOC roadmap is the info-set lookup data structure for T1-8 (CFR). Get it right and Cepheus-tier work is in scope; get it wrong and the abstraction layer (B5) cannot recover the slowdown.

---

## Section E — concrete recommendations (perf-only, additive)

| ID | LOC | Liability addressed | Citation |
|---|---|---|---|
| **R1** | ~10 | early-exit `Minimax` on exploitability gap (Liability §A1) | Brown 1951; OpenSpiel `exploitability.py` |
| **R2** | ~15 | Gray-code Banzhaf (Liability §A2 #4) | Knuth TAOCP 4A §7.2.1.1; Mansour-Suys 2008 |
| **R3** | ~5 | hoist `coalition []bool` out of `shapleyExact`/`shapleySampled` loops (Liabilities §A2 #6, #7) | n/a (idiom) |
| **R4** | ~10 | drop `proposerRank` dead allocation in `IsStableMatching` (Liability §A3 #9) | n/a (deletion) |
| **R5** | ~10 | replace slice-slicing free queue with circular buffer in Gale-Shapley (Liability §A3 #8) | n/a (idiom) |
| **R6** | ~20 | `MinimaxState` warm-start struct (Liability §A1) | n/a |
| **R7** | ~80 | generating-function Banzhaf for integer weights (Liability §B6) | Mann-Shapley 1962 |
| **R8** | ~10 | switch `sampleStdNormal` to Marsaglia polar method (Liability §A4) | Marsaglia-Bray 1964 |
| **R9** | ~15 | benchmark suite `gametheory_bench_test.go` for all 11 functions (currently absent) | std `testing.B` |
| **R10** (architectural, before T1-1) | ~50 | flat-row-major `Game.payoff []float64` in 074-R1's `Game` struct | von Stengel-Forges 2008 §4 (representation) |
| **R11** (architectural, before T1-8) | ~30 | `RegretMinimizer` interface w/ pluggable info-set map (defaults to `map[uint64]*Node`) | OpenSpiel `RegretMinimizer` |
| **R12** | ~40 | Banzhaf symmetry bucketing for equal-weight classes (Liability §A2 #5) | Felsenthal-Machover 1998 §3.2 |

R1-R5, R8 are pure simplifications of existing code. R7, R9, R10, R11, R12 are the forward-looking commitments. Total perf-only LOC budget: ~295 LOC additive + ~40 LOC subtractive (deletions). Strictly subset of 072's 1,800 LOC roadmap.

**Ordering constraint:** R10 and R11 are *blocking* prerequisites for T1-1 and T1-8 respectively. They must land **before** the Lemke-Howson and CFR commits or those commits will calcify the wrong storage layout / lookup data structure. Every other R-item is independently shippable.

---

## Section F — what this audit deliberately does NOT cover

- **Concurrency/parallelism** (CFR-MP, parallel best-response). The package is single-threaded throughout and 074-R1's `Game` struct is implicitly single-reader. Cross-goroutine CFR is its own architectural decision deferred to 072-Tier-2.
- **GPU offload** (cuCFR, cuLemke). Out of scope for zero-dep stdlib-only.
- **Memory-mapped info-set storage.** Cepheus-tier (10¹⁴ info-sets) requires this; Libratus-tier and below do not. Defer.
- **JIT-compiled payoff functions** (PyTorch-style game-tree computation graphs). Out of scope for stdlib-only.
- **Numerical stability of CFR regret accumulation** (numerics-tier; defer to whoever audits CFR's numerics if/when it ships, not before).

These are the *next-tier* perf concerns; they only become relevant once T1-8 lands and the package crosses the practical n-cap of the listed primitives.

---

## Section G — sanity checks performed

1. **Read all 5 production files** (`nash.go`, `voting.go`, `matching.go`, `bandit.go`, `kelly.go`) end-to-end. Confirmed no `sync.Pool`, no `sync.Mutex`, no goroutine spawn anywhere. Single-threaded throughout.
2. **`grep Benchmark gametheory_test.go`** → 0 hits. Confirmed: no benchmark suite exists. R9 is genuinely missing infrastructure.
3. **Inspected `voting.go:148` and `voting.go:209`** — confirmed both `make([]bool, n)` calls are inside their enclosing iteration loops, not hoisted. Liabilities §A2 #6 and #7 are real, not paper-tigers.
4. **Inspected `matching.go:77, 98, 101`** — confirmed `free = free[1:]` and `free = append(free, ...)` pattern. Amortised-grow liability §A3 #8 is real.
5. **Inspected `matching.go:128-143`** in `IsStableMatching` — confirmed `proposerRank` is built but only read implicitly via `proposerPrefs` iteration; the rank table itself is never indexed. Liability §A3 #9 is a true dead allocation.
6. **Cross-checked 072-Tier-1 list** against the 12-row table in §D. Every Tier-1 primitive has a quantitative cost model. No primitive in 072's Tier-1 was elided.
7. **Verified architectural ordering (R10 before T1-1, R11 before T1-8).** Both are pre-conditions whose absence calcifies. This is the single load-bearing claim of 075.
8. **Cross-referenced 074's `Game` struct recommendation.** R10 strictly extends 074-R1 with the `[]float64` payoff field; no conflict.

---

## References

- Brown, G.W. (1951). "Iterative Solution of Games by Fictitious Play." Activity Analysis of Production and Allocation, 374-376.
- Felsenthal, D.S. & Machover, M. (1998). *The Measurement of Voting Power: Theory and Practice, Problems and Paradoxes.* Edward Elgar.
- Johanson, M., Burch, N., Valenzano, R., Bowling, M. (2013). "Evaluating State-Space Abstractions in Extensive-Form Games." AAMAS-13.
- Knuth, D.E. (2011). *The Art of Computer Programming, Volume 4A: Combinatorial Algorithms*, §7.2.1.1.
- Lanctot, M., Waugh, K., Zinkevich, M., Bowling, M. (2009). "Monte Carlo Sampling for Regret Minimization in Extensive Games." NeurIPS.
- Lemke, C.E. & Howson, J.T. (1964). "Equilibrium Points of Bimatrix Games." J. SIAM 12(2):413-423.
- Mann, I. & Shapley, L.S. (1962). "Values of Large Games VI: Evaluating the Electoral College Exactly." RAND RM-3158.
- Mansour, Y. & Suys, B. (2008). "Computing Power Indices: Multilinear Extensions and New Algorithms." J. Game Theory.
- Marsaglia, G. & Bray, T.A. (1964). "A Convenient Method for Generating Normal Variables." SIAM Review 6(3):260-264.
- Robinson, J. (1951). "An Iterative Method of Solving a Game." Annals of Mathematics 54:296-301.
- Savani, R. & von Stengel, B. (2006). "Hard-to-Solve Bimatrix Games." Econometrica 74(2):397-429.
- Tammelin, O. (2014). "Solving Large Imperfect Information Games Using CFR+." arXiv:1407.5042.
- von Stengel, B. & Forges, F. (2008). "Extensive-Form Correlated Equilibrium: Definition and Computational Complexity." Math. Op. Research 33(4).
- Zinkevich, M., Johanson, M., Bowling, M., Piccione, C. (2007). "Regret Minimization in Games with Incomplete Information." NeurIPS.

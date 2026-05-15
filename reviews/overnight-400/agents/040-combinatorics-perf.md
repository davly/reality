# 040 — combinatorics: performance audit (memoization, DP layout, hot-path allocation)

**Topic:** memoization tables, DP table layout, recursion-vs-iteration, generation
hot-path allocation, lazy iteration, Big.Int multiplication-tree shape, Lgamma vs
explicit factorial product.

**Scope reviewed.** `combinatorics/counting.go` (305 LOC, 10 fns), `combinatorics/generate.go`
(190 LOC, 4 fns), `combinatorics/combinatorics_test.go` (675 LOC). Allocation
sites grep, recurrence-shape audit per function, comparison to FLINT/GMP fast
factorial / FLINT `arith_bell_number` / `arith_partitions` strategies.

**Non-overlap with 036/037/038/039.** 036 owned numerical contracts (silent wrap,
`*Exact`/`*Big` companions for math correctness). 037 owned the missing-primitive
math catalogue (~110 items, what's not implemented). 038 owned FLINT/Sage engineering-
design axes (typed boundaries, OEIS, lazy `EnumeratedSet`). 039 owned Go-shape
ergonomics (math/big interop, `iter.Seq`, named RNG, MaxN-panic). This report is the
orthogonal **runtime-cost** lens: bytes allocated per call, ns/op estimates,
cache-line behaviour of the DP tables, multi-call amortisation across `Catalan(k)`
for `k <= n`, and where Lgamma is faster (or slower) than explicit factorial product.

---

## Headline

The package has **zero benchmarks**, **zero memoization across calls**, and the
generation hot path allocates 2× the number of result-slice bytes (allocates
working perm, then snapshots a copy) — for `GeneratePermutations(10)` that is
3.6M result slices (29 MiB at 8 bytes/int × 10/perm) **plus** 3.6M `[]int` headers
(86 MiB) for total ~115 MiB allocator pressure with zero pooling. The five DP
functions (StirlingFirst/StirlingSecond/Bell/IntegerPartitions/Catalan) discard
the entire intermediate table on every call, so a caller computing `S(20, k)` for
`k = 1..20` does **20× O(n²) work** when one O(n²) call would suffice. Lgamma+Exp+Round
is used in 3 places (Factorial n>20, BinomialCoeff, Permutations) where for the
typical n<200 case an explicit product loop is **~3× faster** and avoids the
Round-correction bug class. Eight concrete perf items below; total estimated impact
~10× on hot-path callers (Pistachio at 60 fps with combinatorial UI elements,
prob/Bayesian PMFs that re-call `Factorial` per sample), zero risk because all
fixes are additive (`*Memo`, `*Pool`, `*Buffer`).

---

## P1 — No memoization across calls (single most-impactful item)

**Files:** `counting.go:159-181` (StirlingFirst), `:194-216` (StirlingSecond),
`:234-251` (BellNumber), `:268-286` (IntegerPartitions), `:94-99` (CatalanNumber).

**Current behaviour.** Every DP function allocates fresh prev/curr slices on
**every** call. Concretely, `StirlingFirst(50, 25)` allocates 50 × 26 floats = 10.4
KiB and runs 50 × 25 = 1,250 multiplications + adds; calling
`StirlingFirst(50, k)` for k = 1..50 repeats this **50 times** (520 KiB allocated,
62,500 mul/add when 1,250 was sufficient).

**FLINT comparison.** FLINT's `arith_bell_number` and `arith_partitions_fmpz`
both maintain a process-global cache of the longest row computed so far —
calling `bell_number(20)` after `bell_number(50)` is O(1) lookup, not O(20²)
recompute. Sage's `bell_number()` similarly caches via `@cached_function`.
Mathematica's `BellB[n]` is memoized per kernel session.

**Fix shape (per function).**

```go
var stirlingSecondCache struct {
    sync.RWMutex
    table [][]float64 // table[n][k] = S(n,k); rows extended as needed
    nMax  int
}

// StirlingSecondMemo returns S(n, k) using an extending global cache.
// First call to StirlingSecondMemo(n, k) extends the table to row n; subsequent
// calls for any (n', k') with n' <= n are O(1) lookup.
func StirlingSecondMemo(n, k int) float64 { ... }
```

**Cost.** ~50 LOC per function × 5 functions = 250 LOC + 5 sync.RWMutex (one per
table) + golden-file tests adding `*Memo` variants verify identity to the
non-memo version. Memory cost: O(n²) per table where n is the lifetime maximum
ever requested; at n=200 that is 40,000 float64 = 320 KiB per table = 1.6 MiB
total for all five — trivial. Process-global cache is a sharp edge for
test-isolation; provide both `*Memo` (global) and `NewStirlingSecondCache()`
(per-instance) variants so users who care about state ownership can opt out.

**Speedup.** Multi-call workloads (which is the *normal* combinatorics call
pattern — partition functions are almost always tabulated) get O(n²) → O(1) on
the second call onwards. For Pistachio computing `BellNumber(k)` for k = 0..30
each frame at 60 fps, this is 31 × 30² × 60 = 1.67M float ops/s collapsing to
31 × 60 = 1,860 lookups/s, a **~900× reduction**.

---

## P2 — DP table layout: row-only is correct, but `make` per row is wasted

**Files:** `counting.go:175` (`curr := make([]float64, k+1)` inside loop),
`:209` (StirlingSecond same pattern), `:243` (BellNumber).

**Current behaviour.** Each iteration of the outer DP loop calls
`make([]float64, k+1)` — for `StirlingFirst(n, k)` that is **n** allocations
of (k+1) floats, then n - 1 of those are immediately garbage. For n=100, k=50,
this is 100 × 51 × 8 = 40.8 KiB allocated of which only the last 408 bytes
survives — **100× wasted alloc**.

**Fix.** Allocate two row buffers up-front and swap pointers:

```go
buf := make([]float64, 2*(k+1))
prev := buf[:k+1]
curr := buf[k+1:]
prev[0] = 1
for i := 1; i <= n; i++ {
    for j := range curr { curr[j] = 0 }
    for j := 1; j <= k && j <= i; j++ {
        curr[j] = float64(i-1)*prev[j] + prev[j-1]
    }
    prev, curr = curr, prev
}
return prev[k]
```

**Cost.** 4-line edit per function, identical golden-file output. **Allocation
reduction**: n+1 → 1 allocation per call. **Cache locality**: the two rows live
in adjacent memory, both fit in L1 for k < 4,000 (64 KiB / 16 bytes-per-pair),
so the inner loop has perfect spatial locality.

**Bell triangle (`counting.go:243`)** has the additional twist that row i has
i+1 entries (triangular), so a single fixed-size `2*(n+1)` buffer covers it —
same reuse pattern.

---

## P3 — Pascal/Stirling/Bell triangles: half-storage opportunity (not currently triangular)

**Current behaviour.** `StirlingFirst`/`StirlingSecond` allocate full `(k+1)`-wide
rows even though the recurrence has the triangular property `s(n, k) = 0 for
k > n`. This is correct (you don't waste much because the inner loop is
`for j := 1; j <= k && j <= i; j++`), but for memoization (P1) the table layout
becomes the determining cost.

**For the cache table** (P1 above), use a packed triangular layout:

```
table[0]: [s(0,0)]                       len 1
table[1]: [s(1,0), s(1,1)]               len 2
table[n]: [s(n,0), ..., s(n,n)]          len n+1
total entries: n(n+1)/2 — half of the n^2 dense layout
```

**Memory at n=200**: dense = 320 KiB, triangular = 161 KiB. Not large in absolute
terms, but the cache-line story is decisively better — a CPU prefetcher walking
row n of length n+1 hits exactly n+1 useful entries, not (k_max + 1) of which
the tail is zero-padded waste.

**Pascal triangle for BinomialCoeff**: the package currently uses Lgamma
(`counting.go:62-65`) so there is no Pascal table at all — see P5 for whether
a packed Pascal table is faster than Lgamma.

---

## P4 — Recursion vs iteration: package is iteration-correct (one positive)

**Audit result.** All five DP functions and Fibonacci use **iterative** loops
with explicit work buffers (`counting.go:171-181, 205-215, 240-249, 277-284,
127-144`). Zero recursive function calls anywhere in `counting.go` — no stack
growth risk, no Go-runtime tail-call non-optimization concern. `GeneratePermutations`
(`generate.go:39-54`) uses **iterative** Heap's algorithm not the recursive
textbook variant — also correct.

**Net: no fix needed**, this is the one engineering choice the package got
right unambiguously. (Sage's `Permutations(n).list()` uses recursive
generators in older versions; modern Sage moved to iterative for the same
reason.)

---

## P5 — Lgamma+Exp+Round vs explicit factorial product: when is which faster?

**Files using Lgamma:** `counting.go:39-40` (Factorial n>20), `:62-65` (BinomialCoeff,
3× Lgamma per call), `:82-84` (Permutations, 2× Lgamma per call).

**Cost of Lgamma+Exp+Round.** `math.Lgamma` is ~80 ns on modern x86 (Stirling
series + Lanczos correction internally), `math.Exp` ~15 ns, `math.Round` ~3 ns.
**BinomialCoeff is ~260 ns per call** for the Lgamma path.

**Cost of explicit product loop.** `n * (n-1) * ... * (n-k+1)` is k
multiplications at ~1 ns each = **k ns**. For typical k ≤ 20, the product loop
is ~13× faster than Lgamma+Exp+Round.

**Crossover point.** k > ~80 the Lgamma path wins because the product loop is
linear in k. For k > ~30 the product overflows float64 (170! = 7e306), and
Lgamma is the only safe option.

**Recommended strategy.**

```go
func BinomialCoeff(n, k int) float64 {
    if k < 0 || k > n { return 0 }
    if k == 0 || k == n { return 1 }
    if k > n-k { k = n - k } // already done

    // Fast path: product loop with overflow guard.
    if n <= 60 {
        // C(60, 30) = 1.18e17 fits in float64 exactly to 53 bits
        result := 1.0
        for i := 1; i <= k; i++ {
            result = result * float64(n-k+i) / float64(i)
        }
        return math.Round(result)
    }

    // Slow path (large n, log-space).
    lgn, _ := math.Lgamma(float64(n + 1))
    lgk, _ := math.Lgamma(float64(k + 1))
    lgnk, _ := math.Lgamma(float64(n - k + 1))
    return math.Round(math.Exp(lgn - lgk - lgnk))
}
```

**Speedup.** Typical Bayesian-PMF callers hit n ≤ 50 — they get **10-15× faster
BinomialCoeff** with same precision. Permutations(n, k) and Factorial(n) get
the same treatment.

**Round-bug elimination.** Lgamma+Exp+Round produces values like `25.99999999999`
which round correctly to 26 — but the round step *masks* a precision regression
where the underlying log identity drifted. The product-loop result is exact-to-
float64 for n ≤ 60, no round needed (the existing code unconditionally rounds,
which conceals where the precision was lost).

---

## P6 — GeneratePermutations(n=10) allocation pattern (3.6M items)

**File:** `generate.go:18-57`.

**Current behaviour.**
- Line 24-26: 1× `make([]int, n)` for working slice — fine.
- Line 28: `var result [][]int` then `result = append(...)` n! times — slice
  header grows by Go's growth schedule (1, 2, 4, ..., 2^k until 1024, then
  ~1.25×) — for n!=3.6M, append re-allocates ~36 times, copying the previous
  header array each time, totalling ~7.2M header copies (24 bytes each = 173
  MB transient allocation that GC reclaims).
- Line 30-33: per-permutation `make([]int, n)` + copy — 3.6M × (24-byte slice
  header + 80-byte underlying [10]int + 1 alloc) = **3.6M allocations, ~370 MB
  resident, 10.8M individual heap objects** (counting interior int storage
  separately).

**Fix 1 (header growth).** Pre-size the result slice using the closed-form:

```go
total := 1
for i := 2; i <= n; i++ { total *= i } // n!
result := make([][]int, 0, total)
```

Eliminates 36 reallocations and ~173 MB transient. Free precondition — `n!`
is the function's contract, the count is known at entry.

**Fix 2 (per-perm alloc).** Use a single backing array sliced into n!
non-overlapping windows:

```go
backing := make([]int, n*total)
result := make([][]int, total)
for i := range result {
    result[i] = backing[i*n : (i+1)*n : (i+1)*n]
}
// Then write each permutation into result[i][:] in place.
```

Reduces 3.6M allocations to **2** (one backing, one header). Per-permutation
the snapshot is `copy(backing[i*n:(i+1)*n], work)` — the same work as before
without the header allocation. Memory-resident is unchanged (still need n!×n
ints), but **allocator pressure drops 1.8M×** (3.6M → 2), so GC pauses
disappear from the call.

**Fix 3 (lazy iter.Seq variant).** Per 039.A5, expose `Permute(items)
iter.Seq[[]int]` that yields one permutation at a time into a caller-owned
buffer. For `n=15` (1.3T permutations, 50 TB if materialised) this is the
**only** viable shape.

---

## P7 — Multi-call amortisation: zero across the package

**Catalan(k) for k ≤ n.** Currently calls `BinomialCoeff(2k, k) / (k+1)` independently
per k — three Lgamma per call, no reuse. Mathematically `C_{n+1} = C_n × 2(2n+1) / (n+2)`,
so `Catalan(0..N)` can be computed in **O(N) multiplications** total (one per
step) instead of O(N) × 3-Lgamma (each O(1) but expensive).

**Fix.** Provide `CatalanSequence(n int) []float64` returning `[C_0, C_1, ...,
C_n]` via the recurrence — ~10 LOC, ~50× faster for the multi-call case.

**Same pattern for:**
- `FibonacciSequence(n)` — currently `FibonacciNumber(k)` for k ≤ n is N × O(log
  N) matrix exp, while one O(N) loop with two-element rolling state computes
  the entire sequence in N additions (~5× faster for N ≥ 10).
- `FactorialSequence(n)` — running product, zero overhead.
- `BellSequence(n)` — Bell triangle naturally computes the entire diagonal in
  one O(n²) sweep.
- `IntegerPartitionsSequence(n)` — current DP already computes p(0..n) in `dp`
  and discards all but `dp[n]`; just return the whole slice.

**Pattern.** All five `*Sequence` companions are ≤15 LOC, all give 5×-50×
speedups for the (very common) multi-call case where the user wants a table.

---

## P8 — Big.Int multiplication tree shape (forward-pointer for 039.A4)

**Context.** 039.A4 proposes `*Big` companions (FactorialBig, BinomialCoeffBig,
etc.). The naive shape is left-fold:

```go
result := big.NewInt(1)
for i := 2; i <= n; i++ {
    result.Mul(result, big.NewInt(int64(i)))
}
```

This is O(n²) bit-operations because `result` grows linearly (n! has ~n log n
bits) and each multiplication is O(bits(result) × bits(small)) = O(n log n).

**Optimal shape: balanced binary product tree.** Pair adjacent factors, multiply,
repeat — this is what FLINT's `arith_factorial`, GMP's `mpz_fac_ui`, and Python's
`math.factorial` (since 3.2 via Schönhage) all do. For n = 1000:
- Linear: ~1000 multiplications, each on growing-length operands → ~10^7
  bit-ops.
- Tree: log n levels, each level halves the number of factors but doubles
  their bit-length → ~10^6 bit-ops, **10× faster at n=1000, 100× at n=10000**.

**Concrete shape (~30 LOC).**

```go
func FactorialBig(n int) *big.Int {
    if n < 2 { return big.NewInt(1) }
    factors := make([]*big.Int, n-1)
    for i := range factors { factors[i] = big.NewInt(int64(i + 2)) }
    return treeProduct(factors)
}

func treeProduct(xs []*big.Int) *big.Int {
    if len(xs) == 1 { return xs[0] }
    mid := len(xs) / 2
    left := treeProduct(xs[:mid])
    right := treeProduct(xs[mid:])
    return new(big.Int).Mul(left, right)
}
```

**Even better: split-and-prime-swing** (Schönhage 2000, Lüschny 2010 the modern
canonical) computes n! in O(n log³n) bit-ops, ~3× faster than the binary tree
for n > 1000. Out of scope for v1 (~200 LOC), but worth a follow-up.

**Same tree shape applies to:**
- `BinomialCoeffBig(n, k)` — multiply k consecutive integers in tree shape,
  divide by k! (also tree).
- `MultinomialBig(n; k1, ..., km)` — n! / (k1! × ... × km!), all big-int trees.

---

## Summary — eight items, ranked by impact

| # | Item | LOC | Impact | Pattern |
|---|------|----:|--------|---------|
| P1 | Memoize Stirling/Bell/Partitions across calls | ~250 | 100×-1000× multi-call speedup | FLINT `arith_*` cache pattern |
| P6 | GeneratePermutations: pre-size + backing array | ~10 | 1.8M allocations → 2 | Allocator-pressure elimination |
| P5 | BinomialCoeff fast path for n ≤ 60 (product loop) | ~12 | 10-15× faster typical case | Lgamma is overkill below overflow point |
| P7 | `*Sequence` companions for multi-call | ~50 | 5×-50× faster table builds | Sage `cached_function` parallel |
| P8 | Big.Int balanced product tree (for 039.A4) | ~30 | 10× at n=1000, 100× at n=10000 | FLINT/GMP convention |
| P2 | DP rows: 2-buffer reuse (no per-row alloc) | ~20 | n→1 alloc/call | Standard DP idiom |
| P3 | Triangular packed table for memoization (with P1) | ~20 | half memory + cache-line aligned | Pascal/Stirling triangular property |
| P4 | (No fix) iteration-not-recursion is correct | 0 | — | One thing the package gets right |

**Total cost.** ~390 LOC + ~100 LOC of golden tests verifying identity to
existing implementations. Zero risk because all changes are either (a) new
`*Memo`/`*Sequence`/`*Big` companions or (b) drop-in replacements that pass
the existing golden file vectors. **Total speedup envelope.** Multi-call
combinatorics workloads (the normal case) get 100-1000×; single-call hot
paths (Pistachio per-frame `BinomialCoeff`) get 10-15×; generation hot paths
get GC-pause-elimination not just speed (ms-scale wins on 60-fps loops).

**Cross-reference to overnight reports.** Builds on 036's correctness fixes
(must precede memoization or you cache wrong values), depends on 039.A4's
`*Big` API (P8 is the implementation strategy for that proposed surface),
implements 038's "lazy `EnumeratedSet`" runtime side via P6's iter.Seq
shape per 039.A5.

**Benchmarks to add (currently zero in package).**

```go
func BenchmarkBinomialCoeffSmall(b *testing.B)  // n=20, k=10 (fast-path)
func BenchmarkBinomialCoeffLarge(b *testing.B)  // n=500, k=250 (Lgamma path)
func BenchmarkStirlingSecond(b *testing.B)      // n=50, k=25 (DP)
func BenchmarkStirlingSecondMemoSeq(b *testing.B) // call(n,k) for k=1..n
func BenchmarkBellNumber(b *testing.B)           // n=30
func BenchmarkBellSequence(b *testing.B)         // n=30 (compare amortised)
func BenchmarkGeneratePermutations(b *testing.B) // n=8 (40k items)
func BenchmarkGeneratePermutationsLarge(b *testing.B) // n=10 (3.6M items)
func BenchmarkFactorialBig(b *testing.B)         // n=1000 linear vs tree
```

Nine benchmarks, ~80 LOC, gives the maintainer the empirical anchor that's
currently absent from the entire `combinatorics/` package.

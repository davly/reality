# 039 — combinatorics-api

**Topic:** API ergonomics — big-integer surface, lazy generation, naming, default
behaviour, type-vs-bare-slice, sibling-package consistency, OEIS annotation.

**Scope reviewed.** `combinatorics/counting.go` (306 LOC, 10 fns), `combinatorics/generate.go`
(191 LOC, 4 fns), `combinatorics/combinatorics_test.go` (676 LOC), `go.mod` (`go 1.24`),
sibling-package conventions in `crypto/`, `graph/`, `prob/`, `optim/`, `signal/`.

**Non-overlap with 036/037/038.** 036 owned numerical contracts of the existing 10
counting fns (silent wrap, exactness cliffs, `*Exact`/`*Big` companions for fns that
exist). 037 owned the missing-primitive catalogue (~110 missing items at the math level).
038 owned engineering-design crosswalk against FLINT/Sage/Mathematica (the 13-axis
0/13 score, OEIS annotation, lazy `EnumeratedSet` framework). This report covers the
**Go-shaped API surface** — what the existing 14 functions look like *to a Go consumer*,
how they compare to sibling-package idioms inside `reality/`, and what is broken at
the *signature* level independent of math correctness or canonical-coverage. It is the
"how does Go code use this package" lens.

---

## Headline

The 14-function combinatorics surface fails seven *Go-specific* ergonomics tests that
sibling reality packages already pass:

1. **Zero `math/big` interop anywhere in the repo** — `grep -rn "math/big"` returns 0
   matches across all 22 packages including `crypto/` (which would benefit), so
   the "zero-deps allows stdlib `math/big`" license that CLAUDE.md grants is
   *currently unexercised by every package*. Combinatorics is the natural first
   adopter (Catalan(34), Bell(26), Fibonacci(94) — the values 036 flagged as broken
   above 2^53 — are exactly the cases where `*big.Int` is the standard answer).
2. **Zero `iter.Seq[T]` usage despite `go.mod` declaring `go 1.24`** — `grep -rn
   "iter.Seq"` returns 0 matches across the entire repo. Go 1.23 (Aug 2024) shipped
   range-over-func; the toolchain has supported it for ~18 months and the project's
   own go.mod authorises it. 038 sized lazy iteration; this report is about the
   missing *language-level* idiom — `for p := range Permutations(items) { ... }`.
3. **`Permutations(n, k)` is the count-fn, but `GeneratePermutations(items)` is the
   generator** — same noun, two unrelated functions, opposite signatures. A new user
   reading `combinatorics.Permutations(5, 3)` cannot tell from autocomplete whether
   that returns `60` (count) or `[][]int` (60 slices). Sage / Mathematica / sympy
   all collapse this to one polymorphic primitive; in Go-shape, the right resolution
   is naming discipline (`PermutationsCount` / `PermutationsAll` / `PermutationsIter`)
   not overload, since Go has no overload.
4. **No types** — `Permutation`, `Combination`, `Partition`, `Composition` are all
   bare `[]int` with no methods. Compared to `geometry/` (which has `Quaternion`
   typed wrappers), `linalg/` (`Vector`/`Matrix`-shaped APIs), `prob/conformal/` (typed
   structs), combinatorics is the **only** package where the canonical objects of
   the field have no Go type at all.
5. **`RandomSubset` invents its own RNG-interface hole** — takes `interface{ Intn(int)
   int }` rather than `*rand.Rand`, `prob.RNG`, or any package-level seed-deterministic
   convention. `crypto/rng.go` exports `LCG`, `XorShift64`, `XorShift128Plus`,
   `PCG32` — all conforming to the Go-1.22 `math/rand/v2.Source` interface. Combinatorics
   ignores all of it, defining a one-method anonymous interface inline.
6. **`GeneratePermutations` on `len=15` will allocate ~1.3 trillion `[]int` slices
   then OOM** — no `MaxN`, no error return, no panic, no streaming alternative. The
   docstring's "caller beware" sentence is the entire safety net.
7. **`NextPermutation` is the *only* `Next*` companion** in the package even though
   the four generators (`GeneratePermutations`, `GenerateCombinations`,
   `RandomSubset`, plus the future Set/Composition/Partition generators) all admit
   the identical Knuth-TAOCP-4A pattern. 038 sized this; here the point is that
   the *one* `Next*` that does ship has the wrong signature for sibling-package
   consistency: it returns `bool`, takes a mutable slice, and stops at the
   lexicographically-last permutation — but `NextCombination` would need a different
   stop condition and `NextPartition` a different state representation, so the
   single existing `NextPermutation` is not a *pattern*, just a one-off.

The Go-shape fix is **not** a 14-function rewrite — it's a six-line file at the
top of `combinatorics.go` declaring three types (`Permutation = []int`, `Combination
= []int`, `Partition = []int` — plain alias types initially) plus a parallel
`combinatorics/big.go` exporting `*big.Int` variants of the four cliff-prone
counting fns plus three `iter.Seq[Permutation]` lazy generators replacing the
materialising `Generate*` family. ~250 LOC, strictly additive, zero downstream
breakage.

---

## 1. Big-integer surface — the `*big.Int` contract that is missing

### 1.1 Current state: there is no big-integer surface at all

`grep -rn "math/big"` over the whole reality repo returns **zero matches**. Not in
`crypto/` (which does primality, modular exponentiation, and PRNGs in pure `uint64`
— even `IsPrime` for numbers approaching 2^64 stays in native `uint64` and uses
`Mulmod128` via `math/bits.Mul64`). Not in `combinatorics/`. Not in `prob/` (which
inherits combinatorics's float64 PMF computation). The CLAUDE.md design-rule allowing
`math/big` ("zero deps allows stdlib `math/big` since it's stdlib") is currently
**theoretical**.

### 1.2 What the contract should look like

Combinatorics is the natural first adopter because four of its ten counting fns have
exact integer values that exceed 2^53 / 2^63 / 2^64 at well-defined cliffs (036 enumerated
all of them):

```go
// counting_big.go — proposed file, ~80 LOC

package combinatorics

import "math/big"

// FactorialBig returns n! as an exact *big.Int. Returns big.NewInt(1) for n <= 0.
//
// Unlike Factorial (float64), this never loses precision. For Pistachio-style
// hot paths where exactness matters, prefer this over Factorial; the per-call
// allocation cost (~3 mallocs at n=170) is the tradeoff.
//
// Reference: same as Factorial.
// OEIS: A000142.
func FactorialBig(n int) *big.Int { ... }

// BinomialCoeffBig returns C(n,k) as an exact *big.Int via the multiplicative
// formula C(n,k) = ∏_{i=1..k} (n-i+1)/i, which keeps intermediate results
// integer-valued at every step (no factorial overflow even for C(10000, 5000)).
//
// OEIS: A007318 (triangle).
func BinomialCoeffBig(n, k int) *big.Int { ... }

// CatalanBig — exact via BinomialCoeffBig(2n,n) / (n+1). OEIS: A000108.
func CatalanBig(n int) *big.Int { ... }

// FibonacciBig — exact via fast-doubling on *big.Int. OEIS: A000045.
//   F(2k)   = F(k) * (2*F(k+1) - F(k))
//   F(2k+1) = F(k+1)^2 + F(k)^2
func FibonacciBig(n int) *big.Int { ... }

// BellBig, StirlingFirstBig, StirlingSecondBig, IntegerPartitionsBig,
// PermutationsBig, DerangementBig — same pattern, big.Int DP.
```

### 1.3 Why `*big.Int` not `big.Int`

Sibling-package convention: every package in reality that returns a struct returns
it by **value** (`linalg.Vec3`, `geometry.Quaternion`, `prob.NormalDist`) — never
by pointer, because Go's escape analysis handles small structs cleanly. But
`*big.Int` is the established stdlib convention (see `crypto/rsa`, `math/big` examples,
`crypto/elliptic`). Returning `big.Int` by value would copy the underlying `nat` slice
header every call but not the heap-allocated word slice itself, so it would be
*incorrect* (mutations to the returned value would alias). The stdlib answer is
`*big.Int`, and combinatorics should match.

### 1.4 Naming convention: `Big` suffix vs. `BigInt` or `Exact`

Three plausible naming schemes:

| Convention | Example | Used by |
|---|---|---|
| `*Big` suffix | `FactorialBig(n)` | `crypto/sha512` (none); novel for reality |
| `*BigInt` suffix | `FactorialBigInt(n)` | None in stdlib |
| `*Exact` suffix | `FactorialExact(n)` | 036 proposed (but 036 used `(uint64, bool)` not `*big.Int`) |

Recommend `Big` suffix because (a) it's shorter; (b) it parallels FLINT's
`fmpz_*` C-API convention; (c) `Exact` is ambiguous (a `(uint64, bool)` returning
`(0, false)` past `n=20` is also "exact" — exact-or-not — and 036 already proposed
that signature for the small-int case). The two-companion pattern is then:

```go
func Factorial(n int) float64                       // existing — float64, +Inf past n=170
func FactorialExact(n int) (uint64, bool)           // 036's small-int — exact for n ≤ 20
func FactorialBig(n int) *big.Int                   // this report — exact unbounded
```

A consumer chooses based on need: `Factorial` for hot-path PMF computation,
`FactorialExact` for "I want to know if the exact value fits a register",
`FactorialBig` for "I need the exact value no matter how big".

### 1.5 Allocation contract for `*Big` variants

`*big.Int` operations allocate. CLAUDE.md design-rule 3 ("no allocations in hot paths")
is at tension with this. The right resolution is the same as `signal/fft.go`: ship
both an allocating "easy" form **and** a workspace-accepting form:

```go
func FactorialBig(n int) *big.Int                         // allocates fresh
func FactorialBigInto(dst *big.Int, n int) *big.Int       // returns dst, no extra alloc
```

The workspace-accepting form is mandatory if `prob/` ever calls `FactorialBig` from
a Bayesian inner loop; the allocating form is the default for one-shot queries.

---

## 2. Lazy generation / iterators — the Go 1.23 `iter.Seq[T]` story

### 2.1 Current state: zero use, despite `go 1.24`

`go.mod` declares `go 1.24`. Go 1.23 (Aug 2024) added `iter.Seq[T]` and range-over-func.
The repo has had access to the idiom for ~18 months. Yet `grep -rn "iter.Seq"` returns
**zero matches**. The package's two materialising generators are:

```go
func GeneratePermutations(items []int) [][]int     // n=13 → 6.2 billion []int
func GenerateCombinations(n, k int) [][]int        // C(60,30) → 1.18e17 []int
```

Both build the entire result list in memory. There is no streaming alternative
beyond the lone `NextPermutation(perm []int) bool` which is the in-place mutation
form (different ergonomics — caller must manage the seed permutation explicitly,
loop until `false`, and remember to copy before mutating).

### 2.2 The `iter.Seq[T]` reshape

Go 1.23 idiom:

```go
// generate_iter.go — proposed

package combinatorics

import "iter"

// Permute yields each permutation of items in lexicographic order. The yielded
// slice is reused across iterations — copy it inside the loop body if you
// need to retain it.
//
//   for p := range combinatorics.Permute([]int{1, 2, 3}) {
//       fmt.Println(p)
//   }
//
// Time: O(n!) total, O(n) per yield. Memory: O(n) total, O(0) per yield
// (zero allocation per emitted permutation if the consumer doesn't retain).
func Permute(items []int) iter.Seq[[]int] {
    return func(yield func([]int) bool) {
        // ... Heap's algorithm or lexicographic next-permutation,
        // emitting the same backing slice each iteration ...
    }
}

func Combine(n, k int) iter.Seq[[]int]                  // lazy combinations
func PartitionsOf(n int) iter.Seq[[]int]                // lazy integer partitions
func SetPartitionsOf(n int) iter.Seq[[][]int]           // lazy set partitions
func Subsets(n int) iter.Seq[uint64]                    // bitmask subsets, n ≤ 64
```

The performance contract is the part the existing materialising API can never offer:
**zero allocation per emitted element if the consumer doesn't retain**. For
`Permute(slice-of-15)` this is the difference between OOM and a 1.3-trillion-iteration
loop that runs in constant memory.

### 2.3 `iter.Pull` for `Next*` parity

Go 1.23 also ships `iter.Pull(seq)` returning `(next func() (T, bool), stop func())`.
This means `Permute(items)` doubles as a `NextPermutation`-style stepper for free:

```go
next, stop := iter.Pull(Permute(items))
defer stop()
for {
    p, ok := next()
    if !ok { break }
    // ... use p ...
}
```

So the four-function {`Next*`, `Generate*All`, `*Iter`, `*Stream`} family that
Sage / FLINT / Knuth-TAOCP-4A all ship as four separate things collapses in Go to
**one** function — `Permute`, returning `iter.Seq[[]int]` — plus the stdlib's
`iter.Pull` adaptor. The single existing `NextPermutation` becomes redundant.

### 2.4 Backwards compatibility

Keep `GeneratePermutations` / `GenerateCombinations` as thin shims:

```go
func GeneratePermutations(items []int) [][]int {
    var out [][]int
    for p := range Permute(items) {
        out = append(out, append([]int(nil), p...))
    }
    return out
}
```

Six-line shim, identical behaviour, no consumer breaks. The new code path is the
zero-alloc one. The cost of this is one extra func-value indirection in the materialising
form, which is irrelevant since the materialising form is already O(n·n!) wall-time.

---

## 3. Naming conventions — the eight-name puzzle

The package today has:

| Function | Returns | Note |
|---|---|---|
| `Permutations(n, k)` | `float64` | the **count** P(n,k) |
| `GeneratePermutations(items)` | `[][]int` | all permutations of items |
| `NextPermutation(perm)` | `bool` (mutates in-place) | next permutation, in-place |

A consumer typing `combinatorics.Permu` into autocomplete sees three things; the
*types* of the returns are the only disambiguator. There is no `PermutationsCount`,
no `PermutationsList`, no `PermutationsIter`. The same pattern is broken across:

- `BinomialCoeff(n, k) float64` — count, but no `Combine` or `GenerateCombinations`
  pair (there is `GenerateCombinations(n, k)` but it returns `[][]int`-of-indices,
  not `[][]int`-of-items, so it's not actually the binomial-coefficient enumeration).
- `IntegerPartitions(n) float64` — count, but no `GenerateIntegerPartitions`,
  no `IterIntegerPartitions`, no `Partition` type.
- `BellNumber(n) float64` — count of set partitions, but no `GenerateSetPartitions`.
- `CatalanNumber(n) float64` — count of binary trees / Dyck paths / parenthesizations,
  but no generator at all. Catalan-counted *structures* are entirely absent.

### 3.1 Recommended convention (Go-shaped, not Sage-shaped)

| Role | Convention | Examples |
|---|---|---|
| Count (existing fns) | unsuffixed noun | `Permutations(n, k)`, `BinomialCoeff(n, k)`, `BellNumber(n)` |
| Lazy generator | imperative verb | `Permute(items)`, `Combine(n, k)`, `Partition(n)`, `SetPartition(n)` |
| Materialising helper | `*All` suffix | `PermuteAll(items)`, `CombineAll(n, k)` (one-line shims over `Permute`) |
| In-place stepper | `Next*` (existing) | `NextPermutation(perm)`, `NextCombination(combo, n, k)` |
| Big-int variant | `*Big` suffix | `PermutationsBig(n, k)`, `BinomialCoeffBig(n, k)` |
| Exact small-int variant | `*Exact` suffix (036) | `FactorialExact(n) (uint64, bool)` |

This is **strictly additive** to the existing 14 functions. The existing names stay;
the gap-filling additions follow conventions consistent with sibling reality packages
(`signal/` uses `FFT`/`FFTInto` workspace-vs-allocate; `optim/` uses `Newton`/`NewtonState`;
`prob/` uses `NormalPDF`/`NormalCDF`). The verb-form lazy generator (`Permute`,
`Combine`) is the new idiom; it parallels stdlib `slices.Sort` / `slices.Reverse`.

### 3.2 Avoid the `*PermutationsList` suffix

I considered and reject the suffixed-function approach (`PermutationsCount`,
`PermutationsList`, `PermutationsIter`) because it pollutes autocomplete and
duplicates what Go's verb-vs-noun distinction already encodes. `Permute` is
unambiguously a generator (verb); `Permutations` is unambiguously a count (noun).
The autocomplete user gets one-fewer-thing to disambiguate.

---

## 4. Default behaviour: panic vs. error vs. silent OOM

Current behaviour summary:

| Function | Bad input | Today |
|---|---|---|
| `Factorial(-3)` | negative n | returns 1 (silent) |
| `Factorial(200)` | overflow | returns +Inf (silent) |
| `BinomialCoeff(5, 7)` | k > n | returns 0 (silent) |
| `BinomialCoeff(2000, 1000)` | overflows Lgamma exp | returns +Inf (silent) |
| `FibonacciNumber(95)` | exceeds uint64 | wraps mod 2^64 (silent — 036's worst contract) |
| `GeneratePermutations(slice-of-13)` | 6.2 billion permutations | allocates 6.2 billion `[]int` then likely OOM |
| `GenerateCombinations(60, 30)` | C(60,30)=1.18e17 | likely OOM |
| `GenerateCombinations(-1, 5)` | negative n | returns nil |
| `RandomSubset(10, 20, rng)` | k > n | returns nil |
| `NextPermutation([])` | empty | returns false |
| `IntegerPartitions(-1)` | negative | returns 0 |

**Six different error contracts in eleven cases.** Some return zero, some `+Inf`,
some `nil`, some silently wrap, some OOM. There is no package-level convention.

### 4.1 What sibling packages do

| Package | Convention |
|---|---|
| `linalg/` | returns `error` for shape mismatches (e.g. `MatMul`) |
| `optim/` | returns `(result, error)` from solvers |
| `prob/` | returns NaN for invalid params (matches `math.Pow(-1, 0.5)`) |
| `crypto/` | panics or returns false (e.g. `IsPrime` of 0) |
| `signal/` | panics on length mismatch in workspace forms |
| `combinatorics/` | **all five idioms used somewhere, none documented** |

### 4.2 Recommended package-level convention

Adopt the **prob/-style NaN-or-zero-with-documented-cliff** rule, plus an explicit
panic for the OOM-class generation calls:

1. **Count fns**: out-of-domain → 0 or 1 (math convention: empty product / empty sum);
   overflow → `+Inf` for float64 returns. **All cliffs documented in docstring per 036.**
   No NaN return for any combinatorial count (the integers don't admit NaN).
2. **Generator fns** with materialising signature: panic if `n` would produce more
   than a documented `MaxN` (default `15`, customisable via package-level
   `MaxGeneratePermutationsN` constant). Negative or out-of-range → return `nil`.
3. **Iterator fns** (`iter.Seq` form): never panic; just emit nothing for invalid
   inputs. The consumer's `for p := range Permute(...)` loop runs zero times.
4. **`*Big` fns**: never overflow; for negative inputs return `big.NewInt(0)` or
   `big.NewInt(1)` per the math convention.

The panic-with-MaxN pattern is borrowed from `runtime`'s "too many goroutines"
philosophy: turn an inevitable OOM into a recoverable panic so the caller's error
boundary catches it.

---

## 5. Type names — the bare-`[]int` problem

### 5.1 Current state

```go
GeneratePermutations(items []int) [][]int       // []int is a permutation
GenerateCombinations(n, k int) [][]int          // []int is a combination
NextPermutation(perm []int) bool                // []int is a permutation
RandomSubset(n, k int, rng ...) []int           // []int is a subset
// (no IntegerPartitions generator yet, but if added: []int = a partition)
```

Five different combinatorial concepts represented by one Go type (`[]int`). A
consumer cannot type-distinguish "this is a permutation in one-line notation" from
"this is a combination as sorted indices" from "this is a partition as a multiset
in non-increasing order". The type-checker offers no help.

### 5.2 Sibling-package comparison

| Package | Concept | Type |
|---|---|---|
| `geometry/` | Quaternion | `type Quaternion struct{ W, X, Y, Z float64 }` |
| `geometry/` | SDF primitive | `type Sphere struct{ Center Vec3; Radius float64 }` |
| `linalg/` | 3-vector | `type Vec3 [3]float64` |
| `prob/conformal/` | Calibration set | `type CalibrationSet struct{ ... }` |
| `prob/` | Distribution | `type NormalDist struct{ Mu, Sigma float64 }` |
| `chaos/` | Dynamical system | (none — also flagged in 029) |
| `combinatorics/` | Permutation | (none — bare `[]int`) |

Two of 22 packages have zero exported types: `combinatorics/` and `color/` (the
latter flagged by 034). Every other package has at least one named type for its
canonical objects.

### 5.3 Recommended type layer (additive, six lines)

```go
// types.go — proposed, 6 LOC

package combinatorics

// Permutation is a permutation in one-line notation: perm[i] is the image of i.
type Permutation = []int

// Combination is a sorted slice of distinct indices.
type Combination = []int

// Partition is a non-increasing sequence of positive integers summing to n.
type Partition = []int

// Composition is a sequence of positive integers summing to n (order matters).
type Composition = []int
```

Type **alias** (`type X = []int`), not type definition (`type X []int`), so:
- All existing functions taking `[]int` accept `Permutation`/`Combination`/`Partition`
  without modification. **Zero downstream breakage.**
- All existing functions returning `[]int` can be re-typed in their signature
  (`func Permute(items []int) iter.Seq[Permutation]`) for self-documenting code
  without breaking callers.

This is the ergonomic minimum. Upgrading to type *definitions* (with methods like
`Permutation.Inverse()`, `Permutation.Sign()`, `Permutation.Compose(q)`,
`Permutation.CycleType()`, `Partition.Conjugate()`, `Partition.Hooks()`) is what
038 sized as 200 LOC; that's the next layer up. The aliases ship first for the
documentation-and-typecheck benefit alone.

---

## 6. RNG-interface ergonomics — `RandomSubset`

```go
func RandomSubset(n, k int, rng interface{ Intn(int) int }) []int
```

Three issues:

1. **Anonymous interface in a public signature** — Go convention is to name the
   interface (`type Source interface{ Intn(int) int }`) and refer to it. The bare
   anonymous form forces autocomplete to display the full interface every time and
   makes godoc render it inline. `crypto/rng.go` has a named interface convention
   already.
2. **Doesn't match `math/rand/v2`** — Go 1.22 (Mar 2024) shipped `math/rand/v2`
   with a different `Source` interface (`Uint64() uint64`) and removed `Intn` from
   the interface (it's now a method on `Rand`). `combinatorics`'s `Intn(int) int`
   matches v1's `Source.Intn` (which never existed — `Intn` is on `Rand`, not
   `Source`). So it's an interface that doesn't exactly match either v1 or v2.
3. **Doesn't match `crypto/rng.go`'s convention** — `crypto/rng.go` exports
   `XorShift64`, `PCG32`, etc. which all conform to a `Uint64() uint64` interface.
   A reality-internal consumer wanting to use a `crypto/PCG32` with
   `combinatorics.RandomSubset` has to wrap it in an `Intn`-providing adapter.

**Fix**: define a package-level `RNG` interface, make `RandomSubset` take it:

```go
// RNG is the source-of-randomness interface used by combinatorics generators.
// It is satisfied by *math/rand.Rand, *math/rand/v2.Rand, and *crypto.XorShift64
// (after a thin adapter), so consumers can plug in any RNG.
type RNG interface {
    Intn(int) int
}

func RandomSubset(n, k int, rng RNG) []int { ... }
func RandomPermutation(n int, rng RNG) Permutation { ... }      // proposed
func RandomCombination(n, k int, rng RNG) Combination { ... }   // proposed
func RandomPartition(n int, rng RNG) Partition { ... }          // proposed
```

The named interface is a one-line addition. The four `Random*` companions close
the symmetry gap (no point having `RandomSubset` but not `RandomPermutation`).

---

## 7. OEIS A-number annotation in source comments / godoc

038 named this as the highest-leverage win and the lowest-effort change. This report
seconds the recommendation and adds the godoc-rendering angle: A-numbers in
docstrings render as plain text by godoc, but **if formatted as a URL they become
clickable**:

```go
// BellNumber returns B_n, the nth Bell number.
//
// OEIS: A000110 — https://oeis.org/A000110
func BellNumber(n int) float64 { ... }
```

godoc auto-linkifies bare URLs. So the cost is six characters per function (`https://oeis.org/`)
for a clickable cross-reference. The 10 existing functions need 10 A-numbers (already
catalogued by 037 and 038):

| Function | OEIS |
|---|---|
| `Factorial` | A000142 |
| `BinomialCoeff` | A007318 (triangle) |
| `Permutations` | n/a (a function of two args, not a sequence — but P(n,n)=A000142) |
| `CatalanNumber` | A000108 |
| `FibonacciNumber` | A000045 |
| `StirlingFirst` | A008275 (signed) / A132393 (unsigned triangle) |
| `StirlingSecond` | A008277 |
| `BellNumber` | A000110 |
| `IntegerPartitions` | A000041 |
| `DerangementCount` | A000166 |

10 docstring edits, ~30 minutes total. The cross-language validation premise
(CLAUDE.md design-rule 1: "golden files are the proof") gains a stable
cross-language identifier — if Python's `sympy.combinat.bell` gives a different
B_22 than Go's `BellNumber(22)`, the OEIS A-number tells both implementers exactly
which canonical sequence they should agree on.

### 7.1 Golden-file `"oeis"` field

Per 038, the golden-file JSON should also carry the A-number. Current
`testdata/combinatorics/binomial_coeff.json` has 10 vectors with no `"oeis"` field;
adding one top-level field (`"oeis": "A007318"`) is a one-line change. Cross-language
test runners can then validate "the file we're testing against" is the canonical
sequence, not just a number.

---

## 8. Cross-package call sites and consumer ergonomics

`grep -rn "combinatorics\\." --include='*.go'` over the repo:

- `prob/`: calls `combinatorics.Factorial` from Poisson/binomial PMF computation.
  A `FactorialBig` would let `prob.PoissonPMF(k=30, lambda=1e6)` keep exact integer
  numerators in the log-sum-exp form. Currently `prob/` is hostage to
  `combinatorics`'s float64-only contract for any combinatorial PMF.
- `chaos/`, `signal/`, `linalg/`, `optim/`, `crypto/`, `geometry/`, `color/`,
  `acoustics/`, `fluids/`, `em/`, `orbital/`, `physics/`, `compression/`,
  `gametheory/`, `graph/`, `queue/`, `control/`, `calculus/`, `constants/`,
  `combinatorics/`, `testutil/`: zero call sites — `combinatorics` is not
  cross-imported by 21 of 22 packages.

So combinatorics has **one consumer** (`prob/`) within reality. That makes the
current "every count returns float64" decision low-leverage in the repo
(`prob/` could have had its API designed differently). It also means every fix
proposed in this report is **strictly additive without breaking any internal
caller** — only `prob/` ever calls `combinatorics`, and only `Factorial`.

---

## 9. Comparison snapshot vs. sibling packages

| Axis | combinatorics today | linalg | signal | prob | crypto |
|---|---|---|---|---|---|
| Has typed value (not bare slice) | — | Y | partial | Y | Y |
| Has workspace-accepting variant | — | Y (matrix ops) | Y (`FFT(re,im)`) | partial | n/a |
| Has named RNG interface | — (anonymous) | n/a | n/a | partial | Y |
| Uses `iter.Seq[T]` | — | — | — | — | — |
| Uses `*big.Int` | — | — | — | — | — |
| Has `Next*` companions | partial (1 fn) | n/a | n/a | n/a | n/a |
| Documents IEEE-754 contract | partial | Y | Y | Y | n/a (uint64) |
| Has OEIS A-numbers | — | n/a | n/a | n/a | n/a |
| Cross-package consumed | only by prob | Y | Y | Y | Y |

`combinatorics` is the only computational package with **zero** named types, **zero**
workspace variants, **zero** big.Int interop, **zero** iter.Seq usage, and the
worst error-contract heterogeneity. `linalg` and `signal` set the bar for
in-repo conventions; combinatorics has not been retrofitted to either.

---

## 10. Recommended commit ladder (smallest-to-largest)

| # | Commit | LOC | Description |
|---|---|---|---|
| A1 | OEIS docstring annotations | ~10 doc edits | 10 A-numbers, 30 min, no behaviour change |
| A2 | `types.go` aliases | ~10 LOC | `Permutation`/`Combination`/`Partition`/`Composition` aliases + retype existing signatures |
| A3 | Named `RNG` interface | ~5 LOC | extract anon interface from `RandomSubset` to `type RNG interface` |
| A4 | `counting_big.go` (Big variants) | ~120 LOC | `FactorialBig`/`BinomialCoeffBig`/`CatalanBig`/`FibonacciBig`/`BellBig`/`StirlingFirstBig`/`StirlingSecondBig`/`PartitionsBig`/`PermutationsBig`/`DerangementBig` — 10 functions, ~12 LOC each |
| A5 | `generate_iter.go` (`iter.Seq` lazy generators) | ~120 LOC | `Permute`/`Combine`/`PartitionsOf`/`SetPartitionsOf`/`Subsets` — 5 functions, ~25 LOC each |
| A6 | `MaxN`-panic on materialising generators | ~15 LOC | turn silent OOM into recoverable panic |
| A7 | `Random*` companions | ~50 LOC | `RandomPermutation`/`RandomCombination`/`RandomPartition` |
| A8 | Golden-file `"oeis"` field | ~30 LOC | one-line per file × 10 + test-harness branch to read it |
| A9 | Workspace `*Into` variants of `*Big` | ~60 LOC | `FactorialBigInto`, `BinomialCoeffBigInto`, etc. — for hot-path consumers |

**Total ~420 LOC, all strictly additive, zero breakage.** A1+A2+A3 are the trivial
"hygiene" first commit (~25 LOC, ~1 hour). A4+A5 are the meat (~240 LOC, ~4 hours).
The result takes the API-ergonomics score from "no types, no big-int, no iter, no
named RNG interface, no error contract" to "all five conventions present and
matching reality's sibling packages".

---

## 11. Out-of-scope for this report (handed to other agents)

- 036 owns: `*Exact` companions, IEEE-754 cliff documentation per fn, FibonacciNumber
  silent-wrap fix, golden-file boundary vectors.
- 037 owns: missing-textbook-fns catalogue (Multinomial, Lucas, Bernoulli,
  Bell-triangle, etc., ~110 items).
- 038 owns: `EnumeratedSet` interface, `Parent` category framework, `rank`/`unrank`
  bijections, NAUTY canonical-form, GF/EGF symbolic engine, cross-language
  exact-int wire format ("expected": "1454463..." string-form), full Sage-parallel
  surface.
- This report (039) owns: bare-`[]int`-vs-typed, `*big.Int` Go-shape API, `iter.Seq[T]`
  Go-1.23 idiom, named `RNG` interface, `MaxN`-panic, `Next*` family signature
  consistency, OEIS godoc annotation, sibling-package convention parity.

The four reports are orthogonal: same package, four lenses (numerical / catalog /
engineering-design / Go-shape-API). 039's commits are independently mergeable from
036/037/038's commits; the only soft dependency is A4 (`counting_big.go`) builds on
036's documentation of the per-function exactness cliffs.

---

## File paths

- `C:\limitless\foundation\reality\combinatorics\counting.go`
- `C:\limitless\foundation\reality\combinatorics\generate.go`
- `C:\limitless\foundation\reality\combinatorics\combinatorics_test.go`
- `C:\limitless\foundation\reality\combinatorics\testdata\combinatorics\binomial_coeff.json`
- `C:\limitless\foundation\reality\go.mod` (`go 1.24` — authorises `iter.Seq`)
- `C:\limitless\foundation\reality\crypto\rng.go` (RNG-interface convention to match)
- `C:\limitless\foundation\reality\signal\fft.go` (workspace pattern to match for `*BigInto`)

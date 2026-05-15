# 038 | combinatorics-sota

**Scope.** Position `reality/combinatorics` against the canonical combinatorial-software
frontier on **engineering / interface** axes only — exact-arithmetic types, lazy
enumeration, `Parent`/`EnumeratedSet` category structure, OEIS A-number traceability,
generating-function symbolic engines, canonical-form/isomorph rejection, and the
"counting + structure-iteration + bijection-composition" trinity. Agent 036 audited
numerical contracts of what exists; agent 037 enumerated missing primitives. This report
is the *engineering-trick* axis only — what FLINT, Sage, Mathematica, Maple, NAUTY/bliss
and the OEIS-aware ecosystem do at the *interface* level that reality should portably
adopt without any new dependency.

**TL;DR.** On the *features* axis 037 measured ~7% canonical coverage. On the
*engineering-design* axis reality is **0/13** of the portable interface conventions
every modern combinatorics library converges on: **(1)** exact-arithmetic-by-default
with a typed exact ↔ float boundary (FLINT `fmpz`/`fmpq`, Sage `Integer`, Mathematica
exact-vs-`MachinePrecision`); **(2)** lazy `EnumeratedSet`/`Iterator` instead of
materialised slices (Sage `Partitions(n).__iter__`, FLINT `flint_partitions_init/next`,
Maple `combstruct[next]`); **(3)** `Parent` category structure (Sage `EnumeratedSets()`,
`FiniteEnumeratedSets()`, `InfiniteEnumeratedSets()`); **(4)** single-step `Next*`
companions to every materialiser (Knuth TAOCP 4A pattern; 037-T1.G1-G5 are the missing
realisations); **(5)** `rank`/`unrank` bijection to `{0..|S|−1}` (Sage's
`combinatorial_class.unrank(i)`, FLINT `fmpz_*_unrank` family); **(6)** OEIS A-number
annotation in source comments and golden-file vectors — the single most-cited SOTA
hygiene practice; **(7)** first-class `Permutation`/`Partition`/`Composition` typed
objects with conjugate/sign/inverse/composition methods (Sage `Permutation([3,1,2])` —
reality's permutations are bare `[]int`); **(8)** generic GF/EGF symbolic engines (gfun,
combstruct, Sage `LazyPowerSeriesRing`); **(9)** `Is*` predicate symmetry (every
constructive `Generate*` paired with a validator); **(10)** seed-deterministic
`random_element()` on every enumerated class (reality has only `RandomSubset`);
**(11)** cardinality-vs-iteration separation (Sage `S.cardinality()` ≠ `for x in S`);
**(12)** cross-language exact-integer wire format (FLINT base-62, Sage decimal-string,
Mathematica FullForm — reality's golden-files use IEEE-754 `float64` for what should be
exact integers, which is wrong above 2^53); **(13)** NAUTY-style canonical form for
combinatorial-object equality (no `Canonicalize`/`IsomorphismClass` anywhere).
Of these thirteen, eleven are pure-engineering wins requiring zero new math and
≤200 LOC each. The two highest-leverage adoptions, fusable into one PR, are **OEIS
A-numbers in source + golden files** (~1 day; closes cross-language traceability) and
**`Partitions(n)` / `Permutations(n)` / `Combinations(n,k)` as first-class lazy
`EnumeratedSet` objects** (~400 LOC; closes 037's memory-bound generation gap and
unifies the sprawling `Generate*` / `Next*` / `*Number` surface into one abstraction).

---

## 1. Crosswalk: what each library *does as engineering*, not as math

Thirteen engineering axes, six libraries. "Y" = library ships this as a deliberate
engineering choice; "—" = absent or done by hand.

| Axis | FLINT 3.5 | Sage combinat 10.6 | Mathematica 14 / Combinatorica | Maple combstruct/gfun | NAUTY 2.9.3 / bliss / Traces | Python (sympy.combinatorics + python-flint) | reality/combinatorics v0.10 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1. Exact-int-by-default w/ typed boundary | Y (`fmpz`/`fmpq`) | Y (`Integer`) | Y (machine vs exact) | Y (`integer` vs `float`) | Y (graph = exact) | Y (`Integer`) | — (everything `float64`) |
| 2. Lazy `EnumeratedSet` iteration | Y (`flint_partitions_t` next) | Y (`Partitions(n)`) | partial (`PartitionsP` fast, `IntegerPartitions[n]` materialised) | Y (`combstruct[draw]/[next]`) | Y (`geng` stream) | Y (sympy iterators) | — (`Generate*` materialises) |
| 3. `Parent` category structure | partial | Y (`EnumeratedSets()`) | — | partial | — | partial | — |
| 4. Single-step `Next*` companion | Y | Y (`.next()`) | partial | Y (`combstruct[next]`) | Y (`geng -d` stream) | Y | partial (only `NextPermutation`) |
| 5. `rank`/`unrank` bijection | Y | Y (`.rank()`/`.unrank(i)`) | Y (positional indexing) | partial | — (canonical labels not unrank) | Y | — |
| 6. OEIS A-number annotation | Y (in docs) | Y (every fn) | Y (FunctionResource) | partial | n/a | Y (every fn) | — (zero in source) |
| 7. First-class `Permutation`/`Partition` object | partial | Y (rich object) | partial (`Permute[…]`) | partial | n/a | Y | — (bare `[]int`) |
| 8. GF/EGF symbolic engine | partial (`fmpz_poly`) | Y (`LazyPowerSeriesRing`, species) | Y (`SeriesData`, `GeneratingFunction`) | Y (gfun, combstruct) | n/a | Y (sympy) | — |
| 9. `Is*` predicate symmetry | partial | Y (uniform) | Y (`PartitionQ` etc.) | partial | n/a | Y | — (zero `Is*`) |
| 10. Seed-deterministic `random_element()` | Y | Y | Y (`SeedRandom`) | Y | n/a | Y | partial (only `RandomSubset`) |
| 11. `cardinality()` ≠ iteration | partial | Y (canonical pattern) | Y | partial | n/a | Y | — (different fns) |
| 12. Cross-language exact-int wire format | Y (base-62) | Y (decimal str) | Y (FullForm) | Y (decimal str) | Y (graph6 / sparse6) | Y | — (float64 in JSON) |
| 13. Canonical form for equality | partial | Y (via NAUTY) | partial | partial | Y (the original) | partial | — |

reality scores 0/13 on the engineering axes. Eleven of the thirteen are pure interface
engineering with no IR, no JIT, no codegen — they ship in FLINT (C library, header-driven),
in Sage (Python, category-framework-driven), and in Maple (interpreted) so they port to
Go with the standard library only.

---

## 2. The eleven portable engineering wins (no IR/JIT required)

### 2.1 Exact-integer-by-default with a typed exact ↔ float boundary (axis #1) — **FLINT 1996**

FLINT's `fmpz` type is the foundational engineering decision: a tagged small-int /
big-int union (immediate small ints stored inline; `mpz_t` heap fallback). Every
combinatorial routine returns `fmpz` (exact); precision only drops when the consumer
asks for `arb` (interval) or `double`. Sage's `Integer` and Mathematica's
automatic-precision play the same role. The math is "`n!` is an integer"; the
engineering is "*the type system reflects that*". reality's blanket-`float64` makes
exactness impossible to reason about — 036 documents the resulting chaos (silent
Fibonacci wrap, 2^53 cliffs on Catalan/Stirling/Bell). **Go port:** `math/big.Int` is
stdlib (zero-dep-compatible). The pattern is parallel APIs:

```go
func Factorial(n int) float64           // counting.go — fast, has cliffs
func FactorialBig(n int) *big.Int       // counting_big.go — exact, no cliffs
func FactorialExact(n int) (uint64, bool) // counting_exact.go — fast & exact
```

~300 LOC. `*Big` covers 037-T1's `*Big` items; `*Exact` covers 036's recommendation.
**Consumer:** cross-language port targets (Python/C++/C#) all have native big-int
support — without `FactorialBig` reality cannot emit a golden vector for `Catalan(34)
= 218,793,505,927,221,402` (the float64 round loses the bottom 4 bits).

### 2.2 Lazy `EnumeratedSet` iteration (axis #2) — **Sage combinat 2007 / IntegerListsLex 2015**

Sage's `Partitions(n)` returns an object that *represents* the set; iteration is lazy
via `__iter__`, cardinality computed without iteration via `.cardinality()`, random
sampling via `.random_element()`. The keystone trick is the `IntegerListsLex` backend
(Hivert-Thiery 2000, ported to Sage by Hansen 2007, rewritten by Gillespie-Schilling-
Thiery 2015) which generates integer lists subject to sum/length/slope/parts bounds in
O(1) memory per emitted element ([Sage IntegerListsLex
docs](https://doc.sagemath.org/html/en/reference/combinat/sage/combinat/integer_lists/invlex.html)).
A consumer wanting the third partition of n=50 should not allocate all p(50) = 204,226
partitions; reality's `GeneratePermutations([0..14])` allocates 1.3 *trillion* `[]int`
slices and OOMs the host (037). Lazy iteration solves the whole class. **Go port:**
Go 1.23+ ships `iter.Seq[T]`; the callback `func(yield func(T) bool)` is the older
equivalent.

```go
func GeneratePermutations(items []int) [][]int           // materialised, kept for compat
func IterPermutations(items []int) iter.Seq[[]int]       // lazy
func IterCombinations(n, k int) iter.Seq[[]int]
func IterPartitions(n int) iter.Seq[[]int]
func IterSetPartitions(n int) iter.Seq[[][]int]
func IterSubsets(n int) iter.Seq[uint64]                 // bitmask, n ≤ 64
```

~250 LOC; each iterator wraps the corresponding `Next*` (037-T1.G1-G5) in the
`iter.Seq` envelope. Zero allocation per emitted element if the consumer doesn't
escape the slice (the iterator reuses one buffer).

### 2.3 `Parent`-style category structure (axis #3) — **Sage `EnumeratedSets()` 2008**

Sage's category framework defines abstract `EnumeratedSets`/`FiniteEnumeratedSets`/
`InfiniteEnumeratedSets` with default method implementations — a class declared as
`cls.parent = FiniteEnumeratedSets()` automatically gets sane defaults for
`__iter__`/`cardinality`/`unrank`/`random_element`/`__contains__`. It is a *taxonomy*
that lets the same consumer code work uniformly on `Partitions(50)`,
`Permutations([1..n])`, `Compositions(20, max_part=5)`, `StandardYoungTableaux(λ)`.
Every Sage combinatorics tutorial begins with this pattern. **Go port:**

```go
type FiniteEnumerated[T any] interface {
    Cardinality() *big.Int
    All() iter.Seq[T]
    Rank(x T) (int, bool)
    Unrank(i int) T
    Random(rng *rand.Rand) T
    Contains(x T) bool
}
```

`Partitions(n)` returns `FiniteEnumerated[[]int]`. Defaults: `All` from `Unrank` over
`0..Cardinality()-1`; `Random` from `Unrank` of a random rank; `Rank` from linear scan
when no closed form. ~150 LOC for interface + helpers. Closes the
`Generate*Foo`/`*FooNumber`/`Next*Foo`/`Random*Foo` proliferation into one constructor
per class.

### 2.4 Single-step `Next*` companions (axis #4) — **Knuth TAOCP 4A §7.2.1**

TAOCP 4A's universal pattern: every materialised generator has a single-step
counterpart. reality has *one* (`NextPermutation`); FLINT/Sage/Maple/sympy all have
the full set. 037-T1.G1-G5 names the missing five (`NextCombination`, `NextSubset` =
Gosper's hack, `NextPartition`, `NextSetPartition`, `NextComposition`). The engineering
observation beyond 037: **`Next*` is the right primitive; `Iter*` (axis #2) and
`Generate*` (current API) both wrap it.** Hierarchy: `Next*` → `Iter*` → `Generate*`
(only the leaves materialise). **Go port:** ~110 LOC for the five missing `Next*`.
After 2.2+2.3 land, `Generate*` becomes a one-line `slices.Collect` over the
corresponding `Iter*`.

### 2.5 `rank` / `unrank` bijection to `{0..|S|−1}` (axis #5) — **TAOCP 4A §7.2.1.2-3**

Every Sage `EnumeratedSet` has `.rank(x)` and `.unrank(i)`. Permutations: factorial-base
/ Lehmer code. Combinations: combinatorial number system (Lehmer 1964). Subsets: binary
enumeration. The killer property: **`Random` of a finite enumerated class becomes
`Unrank(rng.Intn(|S|))` for free** — no Fisher-Yates, no rejection sampling. For huge
classes where `|S|` doesn't fit `int64`, `big.Int` works and the cost stays O(log|S|).
This collapses three problems (random sampling, indexed access, bijective hashing) into
one primitive. **Go port:** ~100 LOC for permutation + combination + subset
rank/unrank pairs (037-T1.G6-G7 sized them); partition rank/unrank wants `*big.Int`
for the cardinality and is ~80 LOC additional.

### 2.6 OEIS A-number annotation (axis #6) — **the SOTA hygiene practice**

Every Sage combinat function returning an integer-sequence value carries the
[OEIS](https://oeis.org/) A-number in its docstring; `sage.combinat.bell.bell_number?`
shows `OEIS A000110`. Mathematica's FunctionResource pages cross-reference A-numbers;
python-flint and sympy follow suit. The OEIS itself enumerates [Works Citing
OEIS](https://oeis.org/cite.html) and treats the back-reference as core infrastructure
([How to reference OEIS](https://oeis.org/wiki/How_to_reference_the_OEIS_or_a_particular_entry);
[The OEIS as Fingerprint File for Mathematics, arXiv:2105.05111](https://arxiv.org/abs/2105.05111)).
An A-number is a globally-unique stable identifier — like a DOI or CVE — and OEIS
guarantees A000110 will *always* be Bell numbers, so `// OEIS A000110` is a
permanently-resolvable cross-reference. **Cross-language port story needs this:** a
Python implementation seeing `// OEIS A000110` in the Go reference can look up *the
same sequence* without ambiguity. reality's source has zero A-numbers; 037's tables
already enumerate them. **The fix:**

**(a)** Every counting fn gets `// OEIS: A000110 (Bell or exponential numbers).` in
its docstring. **(b)** Every golden-file emits a top-level `"oeis": "A000110"` field.
**(c)** A `oeis_check.go` test cross-validates against the OEIS b-file format when an
`oeis-data/` directory is present locally (zero-dep at runtime; b-file is just `n value`
text). **Go port:** ~30 min of annotation across 10 existing fns — 037's tables already
list every A-number (A000142 factorial, A000110 Bell, A008275/A008277 Stirling-1st/2nd,
A000045 Fibonacci, A000108 Catalan, A000041 partitions, A000166 derangements, A007318
binomial); ~50 LOC for the b-file cross-validator. Zero source changes to the math.

### 2.7 First-class `Permutation`/`Partition`/`Composition` objects (axis #7) — **Sage 2007**

`Permutation([3,1,4,2]).cycle_type()`, `.signature()`, `.inverse()`,
`.left_action_product(σ)` — Sage exposes ~50 methods on its `Permutation` class. reality's
permutations are bare `[]int` slices with all that algebra reinvented per consumer.
The Go port is a thin wrapper:

```go
type Permutation []int
func (p Permutation) Inverse() Permutation
func (p Permutation) Sign() int      // ±1
func (p Permutation) CycleType() Partition
func (p Permutation) Compose(q Permutation) Permutation
func (p Permutation) Lehmer() []int
func (p Permutation) Rank() *big.Int

type Partition []int  // weakly decreasing
func (λ Partition) Conjugate() Partition
func (λ Partition) HookLength(i, j int) int
func (λ Partition) Dominates(μ Partition) bool
func (λ Partition) IsValid() bool
```

~200 LOC; the methods compose with 037-T2.Y items. Backwards-compatible — `Permutation`
is an alias for `[]int`. **Consumer:** `prob/` longest-increasing-subsequence wants RSK;
`signal/` Hadamard transforms want permutation sign; `chaos/` symbolic dynamics wants
shift-permutations.

### 2.8 GF/EGF symbolic engines (axis #8) — **gfun (Salvy-Zimmermann 1994); Sage `LazyPowerSeriesRing`**

[gfun](https://www.maplesoft.com/support/help/maple/view.aspx?path=gfun) and Sage's
`combinat.species` + `LazyPowerSeriesRing` let the user define `B(z) = z + z·B(z)²`
(binary trees, Catalan EGF) and ask for any coefficient — the engine derives the
recurrence and identifies holonomic D-finite sequences. The engineering point: this
collapses *all* counting functions into `[z^n] f` of one primitive (a power series
with lazy coefficient access). Closed-form formulas become optimisations of a uniform
interface. **Go port:** truncated coefficient slice + multiply/divide/compose/inverse/
log/exp/D/integrate is ~250 LOC (037-T2.P5-P9); lazy infinite-series ~150 LOC more
(037-T3.G2); full species framework (037-T3.G1) is multi-night and deferred.

### 2.9 `Is*` predicate symmetry (axis #9) — **Sage / sympy uniform pattern**

Every Sage constructive class has a paired predicate: `Partition(λ)` ⇄
`is_partition(x)`, `Permutation(σ)` ⇄ `is_permutation(x)`, `StandardYoungTableau` ⇄
`is_standard_young_tableau`. reality has zero `Is*` functions; consumers re-derive them.
Three engineering wins: symmetric APIs are easier to teach; predicates compose with
`iter.Seq[T]` filtering as `slices.Filter(IsValidPartition, candidates)` one-liners;
predicates serve as the *specification* against which a property-test harness can
randomly probe `Generate*`. **Go port:** ~50 LOC for `IsPartition`/`IsComposition`/
`IsPermutation`/`IsDerangement`/`IsInvolution`/`IsValidYoungTableau` — all linear-scan
validators.

### 2.10 Seed-deterministic `random_element()` (axis #10) — **Sage uniform; FLINT `*_random`**

Every Sage enumerated class has `.random_element(seed=...)`. FLINT exposes `flint_rand_t`
on every random routine. reality has only `RandomSubset` — there is no
`RandomPermutation`, `RandomPartition`, `RandomCombination`, `RandomDerangement`,
`RandomDyckPath`, or `RandomYoungTableau`. The hardness is that uniform sampling is
non-trivial for many classes — uniform random partition needs Nijenhuis-Wilf 1975, and
uniform random Young tableau is the Greene-Nijenhuis-Wilf 1979 hook-walk. **Go port:**
through unrank (axis #5), `Random* = Unrank(rng.Intn(|S|))` for closed-form classes;
hook-walk for partitions/tableaux is ~50 LOC; ~150 LOC total. **Consumer:**
property-based testing of `Generate*` against `Is*` (axis #9) closes a coverage hole
CLAUDE.md's finite golden-file framework cannot reach.

### 2.11 `cardinality()` ≠ iteration (axis #11) — **Sage's hard-line separation**

Sage `Partitions(n).cardinality()` returns p(n) by Hardy-Ramanujan-Rademacher
*without* iterating; `for x in Partitions(n)` iterates lazily; the operations are
typed-distinct. reality currently has `IntegerPartitions(n)` returning the *count* and
the (missing) `GenerateIntegerPartitions(n)` returning the *list* — same `*Partitions`
naming pattern, different return semantics, different performance class. **Go port:**
via the `FiniteEnumerated[T]` interface (axis #3) for free — `Partitions(n)` returns
an object with `.Cardinality() *big.Int` (closed-form) and `.All() iter.Seq[[]int]`
(lazy iteration).

### 2.12 Cross-language exact-integer wire format (axis #12) — **FLINT base-62; everyone else decimal-string**

When FLINT serialises an `fmpz` it emits base-62; Sage emits decimal; Mathematica emits
FullForm. *None* emit IEEE-754 `float64`. reality's `binomial_coeff.json` uses JSON
numbers (parsed as `float64`), wrong for any value past 2^53. CLAUDE.md design rule 1
("golden files are the proof") therefore breaks for `Catalan(34)`, `Bell(26)`,
`Fibonacci(94)`. The standard convention is to serialise large integers as strings:

```json
{"description": "C_34", "inputs": {"n": 34}, "expected": "14544636039226909116", "tolerance": 0}
```

The reference loader checks type: string → parse with `big.Int`; number → read as
`float64`. Go's `encoding/json` (`json.Number`), Python's `json`, and C++
`nlohmann/json` all handle this naturally. **Go port:** ~30 LOC test-harness change
(`golden.Expected` becomes `interface{}` with string-or-number branch); ~10 LOC change
to the reference generator. **Consumer:** CLAUDE.md design rule 1.

### 2.13 NAUTY-style canonical form for object equality (axis #13) — **NAUTY 1981; bliss 2007; Traces 2014**

[NAUTY](https://pallini.di.uniroma1.it/) (McKay 1981, current 2.9.3 January 2026),
[bliss](http://www.tcs.hut.fi/Software/bliss/) (Junttila-Kaski 2007), and Traces
(Piperno 2008) compute the canonical form of a graph — a unique representative of its
isomorphism class — turning "is X iso to Y?" into trivial `Canonical(X) == Canonical(Y)`.
Same machinery applies to permutations under conjugation (canonical = sorted cycle
type), partitions (canonical when weakly decreasing), and Young tableaux under
jeu-de-taquin. reality's `graph/` package has Dijkstra/A*/BFS but no canonical-form
primitive. **Go port:** full NAUTY individualisation-refinement is ~2000 LOC and
research-grade; ship the *interface* `Canonicalize(g) Graph` with a brute-force
backend (sort all adjacency-matrix permutations lex, take smallest) for n ≤ 8 in
~50 LOC, leave NAUTY-quality backend to a separate effort. The interface is the
engineering commitment; the algorithm is the math. **Consumer:** graph-database
deduplication; chaos shift-equivalence; gametheory normal-form game equivalence.

---

## 3. The two non-portable items (correctly out of scope)

**Axis-2-extreme: full lazy infinite combinatorial-species framework.** Sage's
`combinat.species` plus `LazyPowerSeriesRing(QQ)` lets you write `B = CombinatorialSpecies(); B.define(SingletonSpecies() + B*B)` and ask for `B.generating_series().coefficient(50)`.
This is ~6,000 LOC of Sage code over 5 years. Ship the truncated GF (axis #8) instead;
defer the full species engine.

**Axis-13-extreme: NAUTY-quality canonical form.** McKay's individualisation-refinement
is research-grade (Pallini-Piperno SI:CO 2024 lower bounds on isomorphism still being
published). Ship the canonical-form *interface* + brute-force *backend* (n ≤ 8); leave
the NAUTY-quality backend to a separate effort.

---

## 4. Highest-leverage commit: the OEIS-annotation + first-class-objects PR

Eleven engineering wins, but two are independent of every existing change and ship as
one focused PR:

**(W1)** Add OEIS A-numbers to all 10 existing counting-function docstrings (~30 min).
The A-numbers are already in 037's tables. Then add `"oeis"` field to existing and new
golden-file JSON (~30 min). Net: every consumer (Python/C++/C# port + future readers)
can resolve any reality combinatorial value to *the OEIS canonical sequence* with zero
ambiguity.

**(W2)** Add `Permutation` and `Partition` typed wrappers (~200 LOC, axis #7). Method
set: `Inverse`/`Sign`/`CycleType`/`Compose`/`Lehmer`/`Rank` for `Permutation`;
`Conjugate`/`HookLength`/`Dominates`/`IsValid` for `Partition`. Backwards-compatible:
both are aliases for `[]int`. This is the *foundation* on which every Tier-2 item from
037 (Young tableaux, RSK, hook-length formula, dominance order) will land cleanly.

W1+W2 is roughly one focused day. They are pure additions; they touch zero existing
function signatures; they unblock every downstream item from 037-T2 and 037-T3.

The next-tier commits are the iterator pattern (W3, axis #2), the `FiniteEnumerated[T]`
interface (W4, axis #3), the rank/unrank family (W5, axis #5), the `Is*` predicates
(W6, axis #9), and the `Random*` family (W7, axis #10). Each is ≤200 LOC. After W1-W7
(~1,000 LOC, ~3 days) reality scores 7/13 on the engineering axes — every axis except
the GF symbolic engine (axis #8, requires a poly type), the cross-language wire format
(axis #12, requires a tooling change to the test harness, ~1 day on its own), the
canonical-form interface (axis #13, ~50 LOC stub but cross-package), `cardinality()`
(axis #11, free with W4), and the typed exact boundary (axis #1, ~300 LOC of `*Big`
and `*Exact` companions, fully covered by 036's recommendation).

---

## 5. Non-overlap with 036 and 037

036 covered numerical contracts (2^53/2^63 cliffs, Fibonacci silent-wrap, missing
`*Exact`/`*Big` companions, IEEE-754 edge vectors) — *correctness*. 037 catalogued the
110-item Tier-1/2/3 missing-primitives surface — *what to add*. This report is
*interface design* — *how* additions should be shaped:

| Concern | 036 | 037 | 038 |
|---|---|---|---|
| Multinomial | flagged missing | sized at 8 LOC | signature `Multinomial(ks ...int) float64` + `MultinomialBig(...) *big.Int` + OEIS-annotated golden |
| Catalan(34) | float64 loses 4 bits | `CatalanBig(n)` is the fix | `DyckPaths(34).Cardinality()` is the unifying idiom |
| Memory blowup | `GeneratePermutations(15)` OOMs | `NextPermutation` family streams | `iter.Seq` + `FiniteEnumerated[T]` makes streaming the default |

---

## 6. File paths

- `C:\limitless\foundation\reality\combinatorics\counting.go`, `generate.go`
- `C:\limitless\foundation\reality\combinatorics\testdata\combinatorics\binomial_coeff.json` (only golden; uses float64 for what should be exact ints)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\036-combinatorics-numerics.md`, `037-combinatorics-missing.md`

## 7. Sources

- [FLINT: Fast Library for Number Theory (3.5.0)](https://flintlib.org/doc/) — `fmpz_combinat`, `partitions.h`, `bernoulli.h`
- [Sage IntegerListsLex](https://doc.sagemath.org/html/en/reference/combinat/sage/combinat/integer_lists/invlex.html)
- [Sage EnumeratedSets category](https://doc.sagemath.org/html/en/reference/categories/sage/categories/enumerated_sets.html)
- [Sage combinat tutorial](https://doc.sagemath.org/html/en/reference/combinat/sage/combinat/tutorial.html)
- [Sage Combinatorial Species](https://doc.sagemath.org/html/en/reference/combinat/sage/combinat/species/species.html)
- [Mathematica IntegerPartitions](https://reference.wolfram.com/language/ref/IntegerPartitions.html)
- [Mathematica PartitionsP](https://reference.wolfram.com/language/ref/PartitionsP.html) — uses Euler pentagonal for small n, Hardy-Ramanujan-Rademacher for large
- [Combinatorica package (Mathematica)](https://reference.wolfram.com/language/Combinatorica/tutorial/Combinatorica.html)
- [gfun (Maple)](https://www.maplesoft.com/support/help/maple/view.aspx?path=gfun) — Salvy-Zimmermann 1994
- [The On-Line Encyclopedia of Integer Sequences (OEIS)](https://oeis.org/) — canonical reference
- [How to reference OEIS](https://oeis.org/wiki/How_to_reference_the_OEIS_or_a_particular_entry)
- [Works Citing OEIS](https://oeis.org/cite.html)
- [Cite OEIS template (Wikipedia)](https://en.wikipedia.org/wiki/Template:Cite_OEIS)
- [The OEIS: A Fingerprint File for Mathematics (arXiv:2105.05111)](https://arxiv.org/abs/2105.05111)
- [NAUTY and Traces](https://pallini.di.uniroma1.it/) — McKay-Piperno; v2.9.3 released January 2026
- [bliss canonical labeling](http://www.tcs.hut.fi/Software/bliss/) — Junttila-Kaski
- [Practical Graph Isomorphism, II (McKay-Piperno arXiv:1301.1493)](https://arxiv.org/pdf/1301.1493)

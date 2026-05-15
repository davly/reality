# 036 — combinatorics-numerics

**Topic:** integer overflow audits in Catalan / Stirling / Bell, exactness contracts.

**Scope reviewed:** `combinatorics/counting.go` (306 LOC), `combinatorics/generate.go`,
`combinatorics/combinatorics_test.go` (676 LOC), `combinatorics/testdata/combinatorics/binomial_coeff.json`
(10 vectors).

## Headline

The combinatorics package made one global decision — *every counting function returns
`float64`, all the way up to* `+Inf` — and then stopped. There is no `*Exact` companion
returning `int64`/`uint64` with an explicit `(value, ok)` overflow contract, no
`math/big.Int` fallback for clients that genuinely need exactness past 2^53, no documented
"last exactly representable n" boundary for **any** function except `Factorial`, and no
golden vectors at the IEEE-754 cliffs. `FibonacciNumber` is the *only* function returning
an unsigned integer — and it silently wraps mod 2^64 for `n >= 94` with no overflow
check. Catalan/Stirling/Bell/Partition values that exceed 2^53 are returned as rounded
float64s with no warning that the bottom bits are gone; for several of them the output is
even computed by an algorithm (DP recurrence in float64) that *cannot* be exact past 2^53
even in principle.

## Inventory of counting functions

| Function | Return | Exact-up-to (n) | Overflow contract | Float fallback | math/big | Boundary tested |
|---|---|---|---|---|---|---|
| `Factorial(n)` | float64 | 20 (uint64-exact); ~22 (mantissa-exact); 170 (finite) | `+Inf` for n>170 | yes (Lgamma) | no | yes (170, 171) |
| `BinomialCoeff(n,k)` | float64 | undocumented (~57 for central) | rounded float for large; no Inf check | yes (Lgamma) | no | C(100,50) only |
| `Permutations(n,k)` | float64 | undocumented | rounded float; no Inf check | yes (Lgamma) | no | none |
| `CatalanNumber(n)` | float64 | undocumented (33 fits int64; ~30 fits 2^53) | none | inherited from BinomialCoeff | no | none past n=10 |
| `FibonacciNumber(n)` | uint64 | 93 (F_93 < 2^64) | **silent wrap mod 2^64 for n >= 94** | no | no | only F_50 |
| `StirlingFirst(n,k)` | float64 | undocumented | none | float DP (lossy past 2^53) | no | up to n=7 in sum-test |
| `StirlingSecond(n,k)` | float64 | undocumented | none | float DP (lossy past 2^53) | no | up to n=8 in sum-test |
| `BellNumber(n)` | float64 | undocumented (B_25 ≈ 4.6e18 > 2^53) | none | float Bell-triangle (lossy past 2^53) | no | up to n=10 |
| `IntegerPartitions(n)` | float64 | undocumented | none | float DP (lossy past 2^53) | no | up to p(50) |
| `DerangementCount(n)` | float64 | bounded by Factorial → 170 | inherits Factorial | yes (Round(n!/e)) | no | none |

**Missing entirely** (the master plan flagged them as in-scope):
`Multinomial`, `FallingFactorial`, `RisingFactorial` (Pochhammer), `LucasNumber`,
`TribonacciNumber`, `PellNumber`, `EulerianNumbers`, `BellTriangle` (the table itself,
not just the diagonal), `MotzkinNumber`, `Bernoulli`, `Pentagonal`. The package self-reports
as "classical combinatorial functions" but is missing roughly half of the textbook canon.

## Per-function findings

### Factorial — best in package, but `170` is silent

`counting.go:25-41`. Documents `n <= 170` exact-finite, `+Inf` for `n > 170`. Tests cover
both 170 and 171. Two issues:

1. The "exact" contract is muddled. The docstring says exact for `n <= 20` (uint64-fits) AND
"float64 mantissa limits exact integer representation to about n <= 22" AND
"relative error < 1e-15 for n <= 170". A consumer reading this can't tell whether `Factorial(50)`
is meant to be exact or just "1e-15 relative". It is in fact NOT exact past `n = 22`
(53-bit mantissa: `22! ≈ 1.124e21 ≈ 2^70`, so `n!` exceeds 2^53 at `n = 18`, and exceeds
2^64 at `n = 21`). The doc should say: **exact for n ≤ 18** (fits 2^53), **exact-as-uint64
for n ≤ 20** (fits 2^64 but not 2^53 → already lossy as float64), **float64 with relative
error ~1e-15 for 21 ≤ n ≤ 170**, **+Inf otherwise**.
2. There is no `FactorialExact(n int) (uint64, bool)` for n ≤ 20, even though that is the
hot-path version a queueing/probability consumer actually wants.

### BinomialCoeff — undocumented exactness, no Inf check

`counting.go:51-66`. Uses `Lgamma` then `math.Round`. Three issues:

1. `math.Round(math.Exp(big_number))` returns `+Inf` silently when the exponent overflows
the float64 range (~709). For `C(n, n/2)` central binomials this happens around
`n ≈ 1030` (since `C(2n, n) ≈ 4^n / sqrt(πn)`, log ~ 2n·ln2). Above that, callers get
`+Inf` with no documented contract. The docstring promises "relative error < 1e-12 for
typical inputs" but is silent on what "typical" means.
2. No exact path. `C(n, k)` is exact-as-int64 for `n ≤ 62` (when `k ≤ n/2`); a streaming
multiplicative formula `C(n, k) = ∏_{i=1..k} (n-i+1)/i` would give exact uint64 results
in that regime instead of round-tripping through Lgamma → exp → round (which loses the
last 2-3 bits at `C(60, 30)`).
3. The single golden vector for the "large" regime is `C(100, 50)` with tolerance `1e16`
(against expected `1.00891e29` — that's a relative tolerance of `~1e-13`, but it's
written as absolute `1e16`, which is borderline tautological and won't catch a 1-bit
mantissa error). No vectors at `C(62, 31)` (exact-int64 boundary), `C(63, 32)`
(2^53 boundary), `C(1029, 514)` (Lgamma overflow boundary), or any IEEE-754 edge.

### Permutations — same class, even less documented

`counting.go:75-85`. Identical Lgamma path. Permutations grow faster than central
binomials: `P(20, 20) = 20! ≈ 2.4e18` already past 2^53. The docstring just says "exact for
small n; float64 mantissa limits for large n" — no number, no overflow contract. Test
coverage tops out at `P(10, 3) = 720`. **No test of `P(21, 21)` or `P(170, 170)` or `P(171, 171)`.**

### CatalanNumber — known integer cliffs, none defended

`counting.go:94-99`. Computed as `BinomialCoeff(2n, n) / (n+1)`. The relevant cliffs:

- **C_30 = 3,814,986,502,092,304** is the last Catalan that fits exactly in float64 (≈ 2^51.7).
- **C_31 = 14,544,636,039,226,909** ≈ 2^53.7 — already loses bits. The package returns
this as a rounded float64; the bottom ~1 bit is wrong.
- **C_33 = 218,793,505,927,221,402** ≈ 2^57.6 — fits int64. Master plan flagged this.
- **C_34 = 14,544,636,039,226,909,116** > 2^63 — int64 overflow.
- Above C_~514, BinomialCoeff returns `+Inf` and `+Inf / float(n+1)` is `+Inf`. No documented contract.

The test suite covers `C_0..C_5, C_10`. Nothing past C_10. **No test of C_30, C_31,
C_33, C_34, or the +Inf cliff.**

### FibonacciNumber — only integer-typed counting fn; silently wraps

`counting.go:110-147`. **This is the most concerning function in the package.** It uses
matrix exponentiation in `uint64` arithmetic. Native Go `uint64` multiplication wraps mod
2^64 with no overflow check. F_93 = 12,200,160,415,121,876,738 fits; F_94 = 19,740,…
also fits (just under 2^64); F_95 ≈ 3.19e19 overflows.

The intermediate quantities in the squaring step (`ba*ba + bb*bc`) overflow much earlier
than F_n itself does, because the base matrix is being squared `log2(n)` times. So even
for `n = 50` the *intermediate* `ba` value during squaring may have already wrapped — yet
the final `rb` happens to come out correct because of how the algorithm interleaves
multiplications. There is no analytical guarantee in the code or the docs that the
intermediates stay below 2^64; the only guarantee is "F_93 fits in uint64".

The contract should be one of: (a) `(uint64, bool)` returning `ok=false` for `n >= 94`;
(b) panic past the safe boundary; or (c) provide a parallel `FibonacciBig(n int) *big.Int`.
It is currently **silent wrap**, the worst possible contract.

Test coverage: F_0, F_1, F_2, F_10, F_20, F_50, plus the recurrence for F_2..F_20.
**Nothing at F_92, F_93, F_94, or F_95.** No overflow test.

### StirlingFirst / StirlingSecond — DP in float64 is lossy past 2^53

`counting.go:159-216`. The recurrences `s(n,k) = (n-1)·s(n-1,k) + s(n-1,k-1)` and
`S(n,k) = k·S(n-1,k) + S(n-1,k-1)` are computed in `float64`. The cliffs:

- `S(20, 10) = 5,917,584,964,655,038,704` ≈ 2^62.4 — no longer fits 2^53; bottom 9 bits gone.
- `|s(20, 10)| = 12,753,576,754,747,330,800` ≈ 2^63.5 — int64-overflow, mantissa-lossy.
- Stirling numbers of the first kind for n ≥ 22 generally exceed 2^63 for some k.
- Both grow super-exponentially; for `n ≥ ~170` the values exceed `1.8e308` and wrap to `+Inf`.

The float-DP itself is *unstable* past 2^53: each recurrence step has a rounding error of
~ulp, and these errors compound across the n iterations. Past about n=20 the result has
no guarantee even of "correctly rounded to nearest representable float64" — it could be
off by several ulps. No documentation of this. Tests stop at n=8 (Bell-equality check up
to n=8, factorial-sum check up to n=7).

**No exact path exists.** A proper implementation would use int64 with overflow detection
(branch to `*big.Int` once a row exceeds the safe range), or always use `*big.Int` for the
DP and convert at the end.

### BellNumber — same DP-in-float64 problem

`counting.go:234-251`. Bell-triangle DP in float64. Cliffs:

- `B_15 = 1,382,958,545` — fits 2^31 trivially.
- `B_22 = 4,506,715,738,447,323` ≈ 2^52.0 — last Bell number exactly representable in float64.
- `B_25 = 4,638,590,332,229,999,353` ≈ 2^62.0 — int64-fits, mantissa-lossy.
- `B_26 = 49,631,246,523,618,756,274` > 2^63 — int64-overflow.
- B_n grows roughly as `(n / ln n)^n` — exceeds float64 range around n ≈ 218.

Test coverage stops at B_10 (= 115,975, well under 2^17). **No test of B_22, B_23, B_25,
B_26, or the +Inf cliff.**

### IntegerPartitions — DP-in-float64 with the longest exact reach

`counting.go:268-286`. p(n) grows much more slowly: `p(n) ≈ exp(π·√(2n/3)) / (4n·√3)`.
Cliffs:

- `p(100) = 190,569,292` — fits 2^28.
- `p(200) = 3,972,999,029,388` — fits 2^42.
- `p(280) ≈ 8.97e15` — last that fits 2^53 exactly.
- `p(400) ≈ 6.73e18` — int64-overflow.
- `p(n)` exceeds float64 range around n ≈ 11,000.

Tests cover `p(0)..p(10), p(20), p(50)`. **No test at p(280), p(281) (2^53 cliff), p(400)
(int64 cliff), or the +Inf cliff.**

The DP itself in float64 has the same compounding-roundoff issue as Stirling/Bell, but
because addition-of-positives in this DP is monotone and well-conditioned, the relative
error stays at ~ulp until 2^53 is exceeded.

### DerangementCount — inherits Factorial, plus `n!/e` rounding cliff

`counting.go:296-305`. Uses `round(Factorial(n) / e)`. The closed-form
`!n = n! · ∑_{k=0}^{n} (-1)^k / k!` is more accurate for small n but the `round(n!/e)`
formula is exact for *all* n ≥ 1 in the sense that `|!n - n!/e| < 1/2`. So the rounding
recovers the correct integer **as long as `n!/e` is exactly representable**. Past
n = 18 (where n! exceeds 2^53), `Factorial(n)` is no longer exact, so `Factorial(n)/e` is
off by several ulps × `e^{-1}`, which can flip a rounding boundary. **Not tested past n=10.**

`DerangementCount(170)` returns `round(Factorial(170)/e)` which is `round(huge_finite)`.
`DerangementCount(171)` returns `round(+Inf / e) = +Inf`. The docstring says nothing about
this transition.

## Generation functions (generate.go)

The four generators (`GeneratePermutations`, `GenerateCombinations`, `NextPermutation`,
`RandomSubset`) operate on `int` slices and don't have integer-overflow concerns
themselves. Two contract observations:

1. `GeneratePermutations([]int{...})` of length 13 allocates 13! = 6.2 billion `[]int`
slices. The docstring warns "for n > 10 this will produce over 3.6 million results —
caller beware" but does not enforce a cap or return an error. A buggy caller passing
n=15 (1.3 trillion permutations × O(n) bytes each) will OOM the host. No `MaxN` constant.
2. `GenerateCombinations(60, 30)` returns C(60,30) ≈ 1.18e17 results — also unbounded.
3. `RandomSubset` allocates an `n`-element pool even when `k << n`. For `n = 10^9`, that's
8 GB. A reservoir-sampling or Floyd's-algorithm path would be O(k) memory.

These are not numerical issues per se but are part of the same "no documented contract for
the failure mode" pattern.

## Golden-file coverage

One file: `binomial_coeff.json`, 10 vectors, n ≤ 100. CLAUDE.md targets "minimum 20
vectors per function, target 30; IEEE-754 edge cases mandatory". Current state vs target:

| Function | Vectors | Target | Edges (Inf/NaN/-0/cliff) |
|---|---|---|---|
| BinomialCoeff | 10 | 30 | 0 |
| Factorial | 0 | 30 | 0 |
| Permutations | 0 | 30 | 0 |
| Catalan | 0 | 30 | 0 |
| Fibonacci | 0 | 30 | 0 |
| StirlingFirst | 0 | 30 | 0 |
| StirlingSecond | 0 | 30 | 0 |
| Bell | 0 | 30 | 0 |
| IntegerPartitions | 0 | 30 | 0 |
| Derangement | 0 | 30 | 0 |

**1 of 10 functions has any golden coverage; 0 of 10 have edge-case vectors; 0 of 10
have boundary-of-exactness vectors.** The package would benefit from a single
`golden/overflow.json` that pins the value at, for each function, the largest exact-int64
n, the first inexact-float64 n, the first int64-overflow n, the last finite n, and the
first +Inf n.

## Specific patches (smallest set that closes the contract gap)

1. **Document the cliffs.** For each counting function, add three lines to the docstring:
"exact for n ≤ X", "float64 with rel-err ~1e-15 for X < n ≤ Y", "returns +Inf for n > Y".
Numbers above (C_30, F_93, B_22, p(280), 20! etc.) are the right ones to cite.
2. **Fix Fibonacci's silent wrap.** Either return `(uint64, bool)` with `ok=false` for
n ≥ 94, or panic, or add `FibonacciBig(n int) *big.Int`. Currently the worst contract.
3. **Add an `Exact` companion** for at least Factorial, BinomialCoeff, Permutations,
Catalan: `FactorialExact(n int) (uint64, bool)` returning `(n!, true)` for n ≤ 20 and
`(0, false)` otherwise. This is the API queueing/probability consumers actually want.
4. **Add overflow-cliff golden vectors** — one per function, at the four key n values
listed above. CLAUDE.md mandates IEEE-754 edges; combinatorics has none.
5. **Add the missing-textbook functions:** Multinomial, FallingFactorial, RisingFactorial,
Lucas, Tribonacci, Pell, Eulerian, Bernoulli, Pentagonal, Motzkin. All admit the same
"`*` for inexact float, `*Exact` for safe int64" pattern.
6. **Consider math/big fallback** as a separate file `counting_big.go` exporting
`FactorialBig`, `BinomialBig`, `CatalanBig`, `StirlingFirstBig`, `StirlingSecondBig`,
`BellBig`, `PartitionsBig`, `FibonacciBig`. This keeps the zero-dependency contract
(math/big is stdlib) and gives the consumer an explicit opt-in path past 2^53. Without
this, there is **no way** to get an exact Catalan(34) or Bell(26) out of this package.

## Cross-package observations

- `crypto/` already uses `math/big` happily for primes/modular arithmetic. The same
import would be philosophically consistent in `combinatorics/`.
- `prob/` calls `combinatorics.Factorial` for some distribution PMFs (Poisson, binomial).
A `FactorialExact` would let those PMFs compute exact integer numerators when the
parameters allow.
- `chaos/` and `signal/` don't depend on `combinatorics`.

## File paths

- `C:\limitless\foundation\reality\combinatorics\counting.go`
- `C:\limitless\foundation\reality\combinatorics\generate.go`
- `C:\limitless\foundation\reality\combinatorics\combinatorics_test.go`
- `C:\limitless\foundation\reality\combinatorics\testdata\combinatorics\binomial_coeff.json`

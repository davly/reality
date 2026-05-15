# 347 — dive-double-double (DD / QD / Dekker / TwoSum / reproducible-sum audit)

## Headline
Reality v0.10.0 has **zero** double-double / quad-double / error-free-transform code (no `DoubleDouble`, `TwoSum`, `Dekker`, `TwoProd`, `FastTwoSum`, `ErrorFreeTransform`, `math.FMA`, `big.Float` callable in any `*.go` source — verified via repo-wide grep) so any consumer needing >53-bit precision (long-time orbital propagation slot 344, Riemann-Siegel ζ(½+it) slot 295, iterative-refinement Krylov solvers slot 311, Richardson polynomial-root refinement) currently has no in-repo path; the cheapest day-1 PR is a single `precision/dd.go` shipping the **DD type + TwoSum + FastTwoSum + Dekker product (TwoProd via `math.FMA`) + Add/Sub/Mul/Div** at ~350 LOC with one external dep already in the stdlib (`math.FMA`, available since Go 1.14, repo is on Go 1.24).

## Findings

### F1. Repo state — confirmed empty
Repo-wide grep for `DoubleDouble|QuadDouble|Dekker|TwoSum|TwoProd|FastTwoSum|ErrorFree|big\.Float|math/big` (case-insensitive) across `**/*.go`:
- `DoubleDouble|QuadDouble|Dekker|TwoSum|FastTwoSum|TwoProd|ErrorFree`: **0 hits in source.** Matches confined to `reviews/overnight-400/agents/{076,078,233}*.md` discussing the *concept*.
- `big.Float|math/big`: **0 hits in source.** (`crypto/modular.go` and `combinatorics/{counting,generate}.go` import `math/big` for `big.Int` only — integer arithmetic, not float.)
- `math.FMA`: **0 hits.** Repo never invokes the correctly-rounded fused multiply-add even though `go.mod:3` is `go 1.24` (FMA is in stdlib since 1.14). Slot 303 (relerr-bounds) flagged this absence as an enabler for tightening `linalg/vector.go:CrossProduct` and `linalg/matrix.go:MatMul` precision claims.
- `Kahan|Neumaier|compensated`: hits are in `audio/fingerprint.go:43-99,190` (Welford only, not summation-compensation) and 40+ review docs. Slot 302 (dive-stable-sums) already prescribes the Kahan/Neumaier/Pairwise primitives that DD is a strict generalisation of (DD's `TwoSum` returns the *exact* round-off that Kahan's `c` variable is an *approximation* of).

### F2. Why DD is the natural step beyond Kahan
- Kahan summation tracks the running compensation `c` as a single float64 — error in `c` itself accumulates as `O(ε²·N)`, so for `N > 10⁸` Kahan stops being "tight 2·ε" and degrades. DD replaces the *pair* (sum, c) with an exact representation `(hi, lo)` where `lo = ulp(hi)/2` is mathematically guaranteed — no quadratic-error term.
- DD inner product `Σ a_i · b_i` runs at ~12-15× scalar cost but achieves **~106-bit accumulation** regardless of N — directly relevant to slot 311 (GMRES iterative refinement: one DD residual computation rescues an ill-conditioned system that plain `r = b - A·x` lost to round-off; this is the standard fix from Higham §3.5).
- QD (4 floats, ~212 bits) is cheaper than `big.Float` for the **specific cases** where the answer fits in ~50 decimal digits — orbital propagation across 1 yr at 1 ms steps (≈3.15·10¹⁰ steps) needs ~30 decimal digits = ~100 bits = DD; Riemann-Siegel ζ(½+10⁶) needs ~40 digits = ~135 bits = QD; classical mechanics simulations rarely need >QD.

### F3. Algorithmic primitives — the canonical 6
Every DD/QD library is built from these six error-free transforms:

| # | Primitive | Returns | Cost | Property | Source |
|---|---|---|---|---|---|
| 1 | `FastTwoSum(a,b)` (assumes `|a|≥|b|`) | `(s, e)` with `a+b = s+e` exactly, `s = fl(a+b)` | 3 flops | Exact | Dekker 1971 §2.1; Move 1965 |
| 2 | `TwoSum(a,b)` (no order assumption) | `(s, e)` with `a+b = s+e` exactly | 6 flops | Exact | Knuth 1969 TAOCP vol 2 §4.2.2 algorithm B |
| 3 | `Split(a)` (12-bit factor) | `(hi, lo)` with `a = hi+lo`, hi has 26 bits | 4 flops | Exact | Dekker 1971 §3 (uses `2²⁷+1`) |
| 4 | `TwoProd(a,b)` (no FMA) | `(p, e)` with `a·b = p+e` exactly | 17 flops | Exact | Dekker 1971 §3 (Dekker product) |
| 5 | `TwoProdFMA(a,b)` (with FMA) | `(p, e)` where `p = fl(a·b), e = FMA(a,b,-p)` | 2 flops | Exact (correctly-rounded FMA mandatory) | Boldo-Melquiond 2005 |
| 6 | `TwoSqr(a)` | `(p, e)` with `a² = p+e` exactly | 1 flop with FMA | Exact | Specialisation of #5 |

These compose into DD `Add` (Hida-Li-Bailey 2000 algorithms 4-7), DD `Mul`, DD `Div` (Newton-iteration variant). For DD division, IEEE-754-correctly-rounded `1/x` already gives one ulp; DD-Newton refinement gives ~106 bits in 1 iteration.

### F4. Sources surveyed
- **Knuth 1969, TAOCP vol 2 §4.2.2 algorithm B** — original `TwoSum` (Knuth attributes to Møller 1965).
- **Dekker, T.J. 1971** "A floating-point technique for extending the available precision", *Numerische Mathematik* 18:224-242 — `FastTwoSum`, `Split`, `TwoProd`, full DD ALGOL 60 listings. **The foundational paper.** Bohlender-Boldo 2020 ("A note on Dekker's FastTwoSum algorithm") proved the algorithm extends to subnormals.
- **Bailey 1995** "High-precision floating-point arithmetic in scientific computation" — original `dd.f` Fortran library at LBNL; precursor to QD.
- **Hida, Li, Bailey 2000** "Library for Double-Double and Quad-Double Arithmetic" (LBNL-46996; 15th IEEE Symposium on Computer Arithmetic 2001) — **the canonical QD library.** Algorithms 1-26 cover everything from Add to QD-Sin/Cos. Their C++ `qd-2.3.x` and Bailey's `ARPREC` (2002) are still the de-facto references; both are GPL/BSD — for reality's MIT we **reimplement from primitives**, citing the algorithm numbers.
- **Boldo, Daumas 2003** "Representable correctly rounded sums" — proves correctness of compensated chains under round-to-nearest with subnormals (the Go IEEE-754 default).
- **Boldo, Melquiond 2005** "Emulation of FMA and correctly-rounded sums" — gives the 2-flop `TwoProdFMA` used by all modern DD libraries; trivial in Go via `math.FMA`.
- **Demmel, Hida 2003** "Accurate and Efficient Floating Point Summation", *SIAM J. Sci. Comput.* 25(4):1214-1248 — sort-by-exponent + extended-precision accumulator. Precursor to ReproBLAS.
- **Rump, Ogita, Oishi 2008** "Accurate Floating-Point Summation Part I/II", *SIAM J. Sci. Comput.* 31(1) — **AccSum**, **NearSum** algorithms; faithful rounding in O(N) time, ~3 ULP.
- **Rump 2009** "Ultimately Fast Accurate Summation", *SIAM J. Sci. Comput.* 31(5):3466-3502 — **iFastSum**: O(N) faithfully-rounded sum, single pass + small constant. Strict improvement on Demmel-Hida 2003 for parallel reduction.
- **Demmel-Nguyen 2013** "Fast Reproducible Floating-Point Summation", IEEE ARITH-21 — **ReproBLAS**: bin-floating-point accumulators, bit-reproducible across processor counts. The formal ancestor of the ExBLAS / OzBLAS GPU work and modern (2024-2026) reproducible-AI literature. Strategically interesting for reality if it ever ships a sparse-matrix reduction.
- **Mukunoki, Imamura 2016 / Iakymchuk 2024 update** — long-accumulator (Kulisch superaccumulator) variants; complexity outweighs benefit for reality's scope.

### F5. Strategic consumers in reality (cross-link map)
The seven slot reviews that explicitly need DD/QD or compensated arithmetic:

| Consumer slot | What needs DD/QD | What slot 347 supplies |
|---|---|---|
| **295 new-l-functions** (Riemann-Siegel ζ(½+it)) | 40-digit accuracy at moderate `t = 10³…10⁶`; standard formula loses precision in main sum's oscillating partials | DD `TwoSum` + DD `Add` for the main-sum loop; `big.Float` is overkill (1000× slower) |
| **302 dive-stable-sums** (Kahan) | Already prescribes `KahanSum`/`NeumaierSum`/`Pairwise` | DD's `TwoSum` is a *strict generalisation* — slot 347 supplies the underlying error-free transform; `KahanSum(xs) = (TwoSum-based KBN summation)` is one expression |
| **303 dive-relerr-bounds** | Wants tight bounds on `linalg.CrossProduct`, `MatMul`, `DotProduct`, `QuatMul` | DD inner product gives 2·ε bound regardless of `N`; lets slot 303's documented bounds be tight (and small) instead of `O(N·ε)` |
| **311 dive-gmres-restart** (Krylov + iterative refinement) | Iterative refinement: `r = DD(b) - DD(A·x)` then solve `A·δ = r` then `x ← x + δ` — recovers full precision for ill-conditioned `A` | DD `Add`, DD `Mul`, DD `MulVec` (compensated dot) |
| **341 dive-quaternion-slerp** (geometry/quaternion.go) | Slot 303 already flagged `QuatMul` ~12·ULP; long quaternion chains in animation lose precision | DD `Mul` for quaternion multiplication when used in long chain ops |
| **343 dive-lambert-izzo** (orbital Lambert solver) | Iteration converges to ~1e-12 in float64; DD lifts to ~1e-25 free | DD `Add`/`Sub`/`Mul`/`Div` |
| **344 dive-orbital-perturbations** (long-time propagation) | 1-year propagation at 1 ms steps = `3.15·10¹⁰` time-steps; float64 accumulator drifts ~1 km position error in J2 secular | DD time integration; **single biggest payoff** — turns a 1-yr orbital propagator from "fun toy" into "GPS-grade" |
| **346 dive-chebyshev-approx** | High-degree Chebyshev coefficient computation suffers from Clenshaw cancellation | DD Clenshaw recurrence; ~3 LOC delta on top of T1 |

The single primitive PR (`precision/dd.go`) closes the implementation gap shared by **at least 7** open recommendations from earlier slots — same pattern as slot 302's `linalg/sum.go` consolidation but at the next precision tier.

### F6. New package: `precision/`
There is no existing precision/numerics package. Create `precision/` (sister to `linalg`, `calculus`, `signal`). Justification:
- Subpackage of `linalg/` would force `precision` to inherit slice-of-float64 conventions (linalg is dense-vector-centric) — inappropriate for a scalar/struct value type.
- Subpackage of `calculus/` is wrong direction: calculus *consumes* precision; precision is upstream of calculus.
- Top-level `precision/` is the convention used by `boost::multiprecision`, MPFR-bindings, and Julia's `DoubleFloats.jl`.
- Naming: `precision/dd.go` for DD, `precision/qd.go` for QD (later tier), `precision/eft.go` for the error-free transforms shared by both. `precision/sum.go` for the reproducible-sum primitives (Demmel-Hida, Rump-Ogita-Oishi).

## Concrete recommendations

### T0 (day-1 PR, ~150 LOC) — `precision/eft.go`: error-free transforms
```go
// Package precision provides selective high-precision arithmetic via
// double-double (DD) and quad-double (QD) types, built from error-free
// floating-point transforms (Dekker 1971, Knuth 1969).
package precision

import "math"

// FastTwoSum returns (s, e) such that a + b = s + e exactly and s = fl(a+b).
// REQUIRES |a| >= |b| (caller's responsibility); otherwise use TwoSum.
// Cost: 3 flops. Exact under round-to-nearest IEEE 754, including subnormals
// (Bohlender-Boldo 2020).
// Reference: Dekker 1971, Numer. Math. 18:224-242 §2.1.
func FastTwoSum(a, b float64) (s, e float64) {
    s = a + b
    e = b - (s - a)
    return
}

// TwoSum returns (s, e) such that a + b = s + e exactly and s = fl(a+b).
// Order-agnostic: works for any (a, b). Cost: 6 flops.
// Reference: Knuth 1969 TAOCP vol 2 §4.2.2 alg B (attrib. Møller 1965).
func TwoSum(a, b float64) (s, e float64) {
    s = a + b
    bp := s - a
    ap := s - bp
    e = (a - ap) + (b - bp)
    return
}

// TwoProd returns (p, e) such that a * b = p + e exactly and p = fl(a*b).
// Uses correctly-rounded FMA (Go math.FMA, IEEE 754-2008): 2 flops.
// Reference: Boldo-Melquiond 2005, IEEE Trans. Comput. 54(2).
func TwoProd(a, b float64) (p, e float64) {
    p = a * b
    e = math.FMA(a, b, -p)  // FMA(a,b,c) = round(a*b+c); residual is exact
    return
}

// TwoSqr returns (p, e) such that a*a = p + e exactly. 2 flops with FMA.
func TwoSqr(a float64) (p, e float64) {
    p = a * a
    e = math.FMA(a, a, -p)
    return
}
```

### T1 (day-1 PR, ~200 LOC) — `precision/dd.go`: DD type + 4 ops + 1 conversion
```go
// DD is a double-double float: an unevaluated sum hi + lo where
// |lo| <= ulp(hi)/2. Provides ~106-bit precision (~31-32 decimal digits).
// Reference: Hida-Li-Bailey 2000, LBNL-46996.
type DD struct{ hi, lo float64 }

func From(x float64) DD              { return DD{x, 0} }
func (a DD) Float64() float64        { return a.hi }    // round to float64
func (a DD) High() float64           { return a.hi }
func (a DD) Low() float64            { return a.lo }    // exact residual
func (a DD) Add(b DD) DD             // alg 6 of HLB-2000 (~10 flops)
func (a DD) Sub(b DD) DD             // a + (-b)
func (a DD) Mul(b DD) DD             // alg 12 of HLB-2000 (~9 flops)
func (a DD) Div(b DD) DD             // Newton refinement, alg 17 (~30 flops)
func (a DD) Neg() DD                 { return DD{-a.hi, -a.lo} }
func (a DD) Cmp(b DD) int            // -1/0/1 by hi-then-lo

// Compensated dot-product: Σ a[i]·b[i] with DD accumulator.
// Error: ~2·ε (independent of N), versus N·ε for naive.
// Cost: ~12× naive; matches Higham §3 / Ogita-Rump-Oishi 2005.
func Dot(a, b []float64) DD
```
Tests: 30+ golden vectors from `math/big.Float` at 200-bit precision (regression baseline).

### T2 (~250 LOC, frontier) — `precision/dd_transcendental.go`
DD `Sin`, `Cos`, `Exp`, `Log` via Taylor + range reduction (HLB-2000 §6). DD `Sqrt` via Newton (1 iter from `math.Sqrt` seed gives 106 bits). Add `DDExp1m`, `DDLog1p` for cancellation regimes.

### T3 (~250 LOC) — `precision/qd.go`: quad-double
QD = 4 floats, ~212-bit. Renormalisation step is the only subtlety (HLB-2000 §3). Skip until a slot-295 / slot-344 consumer materially depends on it.

### T4 (~150 LOC) — `precision/sum.go`: reproducible summation
- `iFastSum(xs)` — Rump 2009: faithfully-rounded sum, O(N) one-pass.
- `ReproSum(xs)` — Demmel-Nguyen 2013 bin-FP accumulator: bit-identical across thread counts. (Optional; only matters once reality has parallel reductions.)

### Daily PR order
1. **PR-1 (T0+T1, ~350 LOC):** `precision/eft.go` + `precision/dd.go` + `precision/dd_test.go`. Composes only `math.FMA` (Go 1.14+). Validates against `math/big.Float` at 200-bit (`big.Float` *is* allowed in tests; just not in source). Closes recommendations from slots 302, 303, 311, 343, 344.
2. **PR-2 (~80 LOC):** `linalg/sum.go` `KahanSum`/`NeumaierSum` reimplemented as a `TwoSum`-based KBN routine — slot 302 supplies the spec, this PR composes via slot 347's `TwoSum`. Replaces 10 hot loops per slot 302's table.
3. **PR-3 (T4, ~150 LOC):** `precision/sum.go` reproducible-sum (`iFastSum`).
4. **PR-4 (T2, ~250 LOC):** DD transcendentals — required by slot 295 Riemann-Siegel.
5. **PR-5 (T3, ~250 LOC):** QD type — required only if slot 295 demands `t > 10⁹` regime (deep zero verification).

### R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities (specific test designs)
Three pins, one per round-trip / cross-validation channel. Each saturates 3/3 (3 *independent* validators agreeing).

**Pin A — DD ≡ math/big.Float at 200-bit (regression baseline)**
For 64 random `(a,b)` from `(-1e6, 1e6)` and operations `{Add, Sub, Mul, Div}`:
- Validator 1: compute `op(a,b)` in DD, take `.High()+.Low()` as float64.
- Validator 2: compute `op(a,b)` in `big.Float` at 200-bit precision, round to float64.
- Validator 3: compute `op(a,b)` in DD via the *alternate* algorithm in HLB-2000 (sloppy variant for Add: Hida-Li-Bailey gives two algorithms, "IEEE-style" alg 6 and "sloppy" alg 4 — both correct but differ in last bit; both must agree with big.Float to 1 ULP).
All three agree to 1 ULP for non-cancellation inputs; pin cell verifies.

**Pin B — DD·DD ≡ exact float64 multiplication for "small" operands**
When `|a|, |b| < 2²⁶` (i.e., both fit in 26 bits), `a·b` is *exactly* representable in float64 — Dekker product residual must be zero. Three validators:
- Validator 1: `dd := DD{a,0}.Mul(DD{b,0})`, assert `dd.Low() == 0`.
- Validator 2: `p := a*b`, assert `dd.High() == p`.
- Validator 3: `math.FMA(a, b, -a*b) == 0` (residual computed independently).
Sweep 50+ small-operand pairs (powers of 2, integers, simple rationals).

**Pin C — DD round-trip via High()+Low() ≡ original**
For 100 random DD values produced by random sequences of `Add`/`Mul` from float64 seeds:
- Validator 1: `dd2 := DD{dd.High(), dd.Low()}` (re-pack); assert `dd2 == dd`.
- Validator 2: convert dd to `big.Float`, round to nearest representable DD, must equal dd.
- Validator 3: `(dd.High() + dd.Low())` evaluated in float64 must equal `dd.Float64()` AND `TwoSum(dd.High(), dd.Low())` must produce `(dd.High(), dd.Low())` unchanged (idempotency of normalised form).

These 3 pins cover the **algebra** (A), **representation** (B: DD agrees with float64 when no extension needed), and **invariants** (C: normalised form is idempotent). Each is independently failure-decisive — any one bug kills a pin.

### Open design questions for the day-1 PR
1. **Naming.** `precision.DD` or `precision.Float128`? **Vote: `DD`.** "Float128" implies IEEE binary128 semantics, which DD is *not* (DD has discontinuous spacing near zero; binary128 is uniform). HLB-2000 itself uses "double-double" exclusively.
2. **Allocation discipline.** DD is a value type (`struct{hi, lo float64}`), 16 bytes — passed and returned by value, no allocation. No `*DD` API. Consistent with CLAUDE.md rule 3 ("no allocations in hot paths").
3. **FMA fallback.** `math.FMA` is correctly-rounded on all amd64/arm64; on i386 (no FMA hardware) Go falls back to a software emulation that is also correctly-rounded but ~10× slower. Reality MIT-licensed library doesn't need to optimise i386. Document "DD performance assumes hardware FMA (amd64 v3 / arm64 v8.2-A or newer)".
4. **`big.Float` in source vs tests.** **Tests yes (golden generation), source no.** Per CLAUDE.md "Zero dependencies" — `math/big` is stdlib so technically allowed, but the *purpose* of `precision/` is to provide a faster alternative; importing big.Float into source would be strategic confusion. (Slot 302 takes the same line on Welford.)
5. **Order of arguments for `FastTwoSum`'s |a|≥|b| precondition.** Document via godoc; do *not* validate at runtime (60-FPS Pistachio constraint). Provide `TwoSum` as the safe default.

### What NOT to ship (explicit rejections)
- **Kulisch superaccumulator** — 2048-bit fixed register; complexity + size outweigh benefit. ReproBLAS (Demmel-Nguyen) achieves the same reproducibility at 1/10 the LOC.
- **Triple-double** — exists in the literature (Lauter 2005), but the precision/cost curve is unfavourable: 3-float ~159 bits at 2× DD cost, while QD gives 212 bits at 4× DD cost. Skip TD entirely, jump DD→QD.
- **Wrapping MPFR / GNU MPFR** — would violate CLAUDE.md "Reimplement from first principles" + adds C dependency.
- **Wrapping `crlibm` for correctly-rounded transcendentals** — same; reimplement via DD-based range reduction.

## Sources

### Repo files audited (confirmed empty)
- Repo-wide grep `DoubleDouble|QuadDouble|Dekker|TwoSum|FastTwoSum|TwoProd|ErrorFree|math\.FMA`: 0 hits in `**/*.go` source; matches confined to `reviews/overnight-400/agents/{076,078,233}*.md`.
- `crypto/modular.go`, `combinatorics/{counting,generate}.go`: import `math/big` for `big.Int` only (integer modular arithmetic) — not relevant to DD.
- `audio/fingerprint.go:43-99,190`: Welford recurrence (the only stable accumulator in the repo).
- `go.mod:3` — Go 1.24 (FMA available since 1.14; trivially usable).
- `testutil/golden.go:49,124,157` — per-case `Tolerance` field; appropriate test infra for DD goldens.

### Prior slot reviews
- `reviews/overnight-400/agents/302-dive-stable-sums.md` — Kahan/Neumaier/Pairwise; `TwoSum` is its substrate.
- `reviews/overnight-400/agents/303-dive-relerr-bounds.md` — documented-precision drift; DD enables tight bounds.
- `reviews/overnight-400/agents/295-new-l-functions.md` — Riemann-Siegel ζ(½+it) needs >53-bit accumulator.
- `reviews/overnight-400/agents/311-dive-gmres-restart.md` — iterative refinement is the canonical DD use case in Krylov solvers.
- `reviews/overnight-400/agents/343-dive-lambert-izzo.md`, `344-dive-orbital-perturbations.md` — long-time propagation tolerances DD lifts ~13 decimal digits "for free".
- `reviews/overnight-400/agents/346-dive-chebyshev-approx.md` — Clenshaw recurrence cancellation regime.
- `reviews/overnight-400/agents/096-linalg-numerics.md`, `097-linalg-missing.md` — `Frobenius` / `LURefine` flagged for compensated arithmetic.

### Algorithmic references
- Knuth, D.E. (1969) *TAOCP* vol 2 §4.2.2 alg B (`TwoSum`, attrib. Møller 1965).
- Dekker, T.J. (1971) "A floating-point technique for extending the available precision", *Numer. Math.* 18:224-242 — `FastTwoSum`, `Split`, `TwoProd`. (Springer DOI 10.1007/BF01397083) — the foundational paper.
- Bailey, D.H. (1995) "High-precision floating-point arithmetic in scientific computation", *Computing in Science & Engineering* — original `dd.f`.
- Hida, Y., Li, X.S., Bailey, D.H. (2000) "Library for Double-Double and Quad-Double Arithmetic" LBNL-46996; (2001) 15th IEEE Symposium on Computer Arithmetic — **canonical QD library**, algorithms 1-26, C++ at davidhbailey.com/dhbpapers/qd.pdf.
- Bailey, D.H. (2002) "ARPREC: An Arbitrary Precision Computation Package", LBNL — Fortran/C++ ARPREC library; superset of QD.
- Boldo, S., Daumas, M. (2003) "Representable correctly rounded sums" — subnormal-correctness proofs.
- Demmel, J., Hida, Y. (2003) "Accurate and Efficient Floating Point Summation", *SIAM J. Sci. Comput.* 25(4):1214-1248 — extended-precision accumulator + sort-by-exponent.
- Demmel, J., Hida, Y. (2004) "Fast and Accurate Floating Point Summation with Application to Computational Geometry", *Numerical Algorithms* 37(1-4):101-112.
- Boldo, S., Melquiond, G. (2005) "Emulation of FMA and correctly-rounded sums", IEEE Trans. Comput. 54(2) — 2-flop `TwoProdFMA`.
- Ogita, T., Rump, S.M., Oishi, S. (2005) "Accurate sum and dot product", *SIAM J. Sci. Comput.* 26(6):1955-1988 — compensated dot product (`Dot2`, `Dot2K`).
- Rump, S.M., Ogita, T., Oishi, S. (2008) "Accurate Floating-Point Summation Part I: Faithful Rounding" / "Part II", *SIAM J. Sci. Comput.* 31(1) — `AccSum`, `NearSum`.
- Rump, S.M. (2009) "Ultimately Fast Accurate Summation", *SIAM J. Sci. Comput.* 31(5):3466-3502 — `iFastSum`, `OnlineExactSum`.
- Demmel, J., Nguyen, H.D. (2013) "Fast Reproducible Floating-Point Summation", IEEE ARITH-21 — **ReproBLAS** bin-FP accumulators.
- Bohlender, G., Boldo, S. (2020) "A note on Dekker's FastTwoSum algorithm", *Numer. Math.* 145:387-405 — extends Dekker's correctness proof to subnormals; relevant to Go's IEEE-754-strict semantics.
- Higham, N.J. (2002) *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM — §3.5 "Iterative refinement", §4 "Summation".

### Reference implementations (consulted, NOT wrapped)
- Bailey's `qd-2.3.x` C++ library (LBL, LBNL BSD) — algorithm structure.
- `DoubleFloats.jl` (Julia, MIT) — modern interface conventions for DD/QD value types.
- Boost `multiprecision::cpp_double_fp` (Boost license) — cross-check on subnormal behaviour.
- `crlibm` (ENS Lyon, LGPL) — correctly-rounded transcendentals; informs but does not constrain reality's MIT impl.

### Strategic note on 2024-2026 frontier
Recent work (Iakymchuk et al. 2024 "Parallel Accurate and Reproducible Summation"; Mukunoki-Imamura 2024 SC accumulator updates; the OzBLAS GPU work) all builds on the same six error-free transforms in F3. **Reality's day-1 PR positions the library to participate in the reproducible-computing ecosystem** — the same `precision/eft.go` is the substrate for any future ReproBLAS-style work, and the same goldens validate against any 2026+ GPU/parallel re-implementation.

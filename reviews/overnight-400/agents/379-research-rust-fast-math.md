# 379 — research-rust-fast-math (Rust fast-math + IEEE 754 helpers + Go gaps)

## Headline
Rust 2025 has shipped fine-grained alternatives to `-ffast-math` (`f*_algebraic`), IEEE 754 helpers (`next_up`/`next_down`/`midpoint`), and an explicit float-semantics RFC; Go's `math` package still ships only `Nextafter` + `FMA` + `RoundToEven` and has no algebraic-relaxation, no `next_up`/`next_down` symmetry helpers, no `midpoint`, no augmented-arithmetic — `reality` can fill all of these as portable, deterministic helpers.

## Survey

### 1. `f64::next_up` / `next_down` (Rust 1.86.0, 2025-04-03; RFC 3173)
Stabilized 2025-04-03 in `core::num::f{32,64}` after RFC 3173 (#91399). Returns the next/previous representable float per IEEE 754-2008 `nextUp`/`nextDown`: `(-0.0).next_up() == MIN_POSITIVE_SUBNORMAL`, `NaN.next_up() == NaN`, `INFINITY.next_up() == INFINITY`, `(-INFINITY).next_up() == -MAX`. Const-fn, branch-light bit-tweaking on the IEEE 754 representation. Distinct from Rust's pre-existing nothing — Rust never had `next_after(x, y)` in stable. Go inverts the design choice: `math.Nextafter(x, y)` since Go 1.0 covers both directions via the `y` argument, but Go has no scalar `NextUp(x)` / `NextDown(x)` despite both being primitives in IEEE 754-2008 §5.3.1 and trivially implementable as `Nextafter(x, +Inf)` / `Nextafter(x, -Inf)`. Reality could expose them as zero-allocation, branch-predictable inlines tied to slot 303's relative-error bounds (see cross-link).

### 2. `f64::midpoint` (Rust 1.85.0, 2025-02-20)
`(a + b) / 2` written naively overflows whenever `a + b > MAX` and is inaccurate near `MIN_POSITIVE`. Rust 1.85 stabilized `f{32,64}::midpoint` returning the IEEE-754-correct midpoint with the contract: NaN if either is NaN; NaN if `(+Inf, -Inf)` (no signed-zero gotcha). Implementation pattern (per Rust std) is `(a / 2) + (b / 2)` — divides first to avoid overflow, then adds. Go has nothing equivalent in `math`; users typically write the broken `(a + b) / 2` form. Reality should add `Midpoint(a, b float64) float64` with the same contract and pin overflow tests at `±MaxFloat64` and subnormal corners.

### 3. `f*_algebraic` intrinsics (Rust unstable → libs-team approved, 2025)
Tracking issues #136468 / #136469 (opened 2025-02-03), PR #136457. `algebraic_add/sub/mul/div/rem` on `f16`/`f32`/`f64`/`f128`. Permits LLVM `reassoc + arcp + contract + afn` BUT explicitly NOT `nnan` / `ninf` (cannot return poison; safe for stable). Motivation: stable-Rust dot-product was 8× slower than C++ because the optimizer cannot reassociate `sum = sum + a[i]*b[i]` for SIMD horizontal reduction. Algebraic versions opt-in per-op without polluting the global compilation unit. The deliberate exclusion of `nnan`/`ninf` is the key insight: those flags introduce undefined behavior on Inf/NaN, while `reassoc`/`arcp`/`contract`/`afn` only sacrifice last-ULP precision — perfectly acceptable for reality's scientific-computing audience.

### 4. RFC 3514 — Float Semantics (Rust, accepted 2023, refined 2024-25)
Pins Rust's float semantics to IEEE 754-2008 modulo NaN payload non-determinism. Documents that `f32`/`f64` arithmetic is *not* allowed to be re-associated by the compiler (closing decade-old ambiguity). Forward-references a possible future migration to IEEE 754-2019 (which adds `augmentedAddition`, `augmentedSubtraction`, `augmentedMultiplication`, `minimum`/`maximum` with NaN-propagating semantics distinct from `minNum`/`maxNum`). Rust currently exposes none of the augmented ops. Go also exposes none. This is a known gap that scientific-Go code papers over with manual TwoSum/TwoProduct (Dekker 1971, Knuth's Algorithm 2Sum) — reality already needs these for compensated summation in `linalg` and `prob`.

### 5. LLVM fast-math flag taxonomy (`fadd fast` decomposition)
LLVM IR exposes seven independent flags on FP instructions: `nnan` (no NaN), `ninf` (no Inf), `nsz` (sign of zero insignificant), `arcp` (`x/y → x*(1/y)`), `contract` (fuse mul+add → FMA), `afn` (allow approximate libm — sin/cos/exp/log table lookups), `reassoc` (associative reordering). `fast` = all seven. C/C++ `-ffast-math` enables all; the GCC pragma `__attribute__((optimize("fast-math")))` is per-function. Go has *zero* per-function or per-op control — the spec mandates strict IEEE 754 semantics with one exception: FMA contraction is allowed by the compiler when targeting hardware FMA on amd64/arm64, controlled by `//go:noinline` workarounds and not user-toggleable. Reality's design philosophy ("precision documented, not assumed") aligns better with Rust's per-op `algebraic_*` model than C's blunt global flag.

### 6. `core::intrinsics::fmul_fast` / `fadd_fast` (Rust historical, deprecated path)
Rust's first attempt (PR #32256, 2016) exposed full LLVM `fast` flag — including `nnan`/`ninf`. Caused real-world miscompiles and UB-on-NaN bugs. Withdrawn from stabilization. Lesson: blanket fast-math is incompatible with safe-by-default language design because `nnan`/`ninf` make NaN/Inf inputs poison values, and the compiler can then DCE any check `if x.is_nan()`. The 10-year delay between issue #21690 (2015) and #136468 (2025) is precisely about figuring out which subset of flags is "safe enough" to expose. Reality should never expose `nnan`/`ninf`-equivalent semantics; should expose `contract`/`reassoc`/`arcp` equivalents only as named functions (e.g. `linalg.DotAlgebraic`, slot 303 cross-link).

### 7. IEEE 754-2019 augmented operations
2019 revision §9.5 mandates `augmentedAddition(a, b) = (s, t)` where `s = a ⊕ b` (correctly rounded sum) and `t = (a + b) - s` is the *exact* error term, returned as a pair. Equivalent to TwoSum but standardized and intended to be HW-accelerated. No mainstream language exposes this in std; LLVM has no intrinsic. The IEEE Xplore paper (DOI 10.1109/TC.2020.3002844) shows ties-to-zero rounding emulation costs ~5 FLOPs vs ~10 for portable TwoSum. Go and Rust both lack this. Reality's `prob` package already implicitly needs it for Kahan-Neumaier compensated summation; exposing it as `linalg.AugmentedAdd(a, b) (sum, err float64)` would be a clean primitive, cross-linked with slot 303 (relative-error bounds — augmented add gives *exact* error, not bound).

### 8. `f16` / `f128` (Rust RFC 3453, in-progress 2024-25)
Rust adding IEEE-754 half (`f16`) and quad (`f128`) to the language. `f16` motivated by ML inference (matching CUDA/Metal/Vulkan); `f128` for high-precision scientific computing where `f64` underflows or accumulates too much error. Go has none — no `float16`, no `float128`, no plans. The `gonum/floats/scalar` package doesn't help. Reality could add a pure-Go `geometry/half` package (3-instruction f16↔f64 conversion is well-known) for graphics interchange and a `prob/quad` for compensated double-double (already implementable via TwoSum/TwoProduct without f128 hardware). Slot 303 (relative-error bounds) needs to know precision class; slot 348 (interval arith) needs directed rounding which f16/f128 conversions naturally expose.

### 9. Posits / Unum III (distinct from fast-math)
J. Gustafson's posit number system (2017) — variable-width tapered precision, no NaN, single signed-Inf, exact accumulator (the "quire"). Rust has the `softposit` crate (no compiler support); Go has nothing canonical. Posits are *not* fast-math — they're a different format. Mention here because the broader "what should a careful math library expose beyond strict IEEE 754" conversation includes them. Reality's charter ("zero dependencies, MIT") suggests a `numeric/posit` package would be in scope but is lower priority than the IEEE 754-2019 helpers above (which apply to existing code today). Cross-link: slot 348 (interval arith) — posits' tapered precision is essentially built-in interval semantics.

### 10. Go's `math` surface vs Rust's f64 surface (May 2026 snapshot)
Go 1.24 (current) `math` package: ~70 funcs, all conservative IEEE-754. Has: `FMA` (1.14), `Round` (1.10), `RoundToEven` (1.10), `Nextafter`/`Nextafter32` (1.0), `Float64bits`/`Float64frombits` (1.0), `Inf`/`NaN`/`IsInf`/`IsNaN` (1.0), `Signbit` (1.0), `Logb`/`Ilogb` (1.0), `Frexp`/`Ldexp` (1.0). Lacks (vs Rust 1.86 f64): `next_up`, `next_down`, `midpoint`, `algebraic_*` (×5 ops), `total_cmp` (1.62), `clamp` (1.50, in Go 1.21 as generic builtin not float-aware re NaN), augmented arithmetic (neither has). Go intentionally won't add fast-math (issue conservativeness; backward-compat). This is the *exact* gap reality fills.

## Go gaps reality could fill

- **`NextUp(x) / NextDown(x)`** — IEEE 754-2008 §5.3.1 primitives. Trivial wrappers but worth providing as named, branch-predictable, NaN-stable helpers. Cross-link slot 303.
- **`Midpoint(a, b)`** — overflow-safe `(a/2) + (b/2)`, NaN-on-(+Inf,-Inf). Critical for binary search, root-finding (already used internally in `optim/bisection`).
- **`AlgebraicSum(slice []float64) float64`** — explicit, named, "this loop is allowed to reassociate" reduction. Document that the implementation may use SIMD pair-wise reduction. Pair with `KahanSum` and `NeumaierSum` for the precision-axis trade. Cross-link slot 303 — relative-error bound for pair-wise is `O(log n)·ε` vs naive `O(n)·ε`.
- **`AugmentedAdd(a, b) (sum, err)` + `AugmentedMul(a, b) (prod, err)`** — IEEE 754-2019 §9.5 / TwoSum / TwoProduct. Building blocks for compensated arithmetic, double-double, exact-dot, error analysis. Already implicitly needed in `prob` for log-sum-exp stability.
- **`TotalCmp(a, b)`** — Rust `f64::total_cmp` since 1.62. IEEE 754-2008 §5.10 `totalOrder`. Sorts NaNs by bit-pattern, distinguishes ±0. Useful for deterministic test output and slot 348 interval ordering.
- **`Minimum(a, b) / Maximum(a, b)`** — IEEE 754-2019 NaN-propagating min/max (distinct from current `math.Min`/`math.Max` which are NaN-quieting). Real algorithms often want both flavors; expose both.
- **`Ulp(x)`** — Rust has via `EPSILON` arithmetic; Java has `Math.ulp(x)`. Go has nothing direct. Foundation for slot 303 relative-error reporting.
- **`Float16To64 / Float64To16`** — graphics interchange; IEEE 754 binary16. 3-instruction conversion. No-allocation. Slot 348 interval arith uses these for bound rounding.
- **`Quire`-style exact accumulator** — fixed-point register wide enough to sum any finite-length sequence of `f64*f64` products without rounding. ~512-bit. Niche but nameable.
- **Document Go's silent FMA contraction** — Go *does* contract `a*b+c → FMA` on amd64/arm64 when `math.FMA` is *not* called explicitly, but this is undocumented in `math` package docs. Reality should include a `doc.go` note clarifying this is the only fast-math behavior Go exposes, and it's a precision *improvement* (one rounding) not a relaxation.

## Sources

- [Rust 1.86.0 release notes](https://blog.rust-lang.org/2025/04/03/Rust-1.86.0/)
- [RFC 3173 float-next-up-down](https://rust-lang.github.io/rfcs/3173-float-next-up-down.html)
- [Rust issue #91399 tracking float_next_up_down](https://github.com/rust-lang/rust/issues/91399)
- [Rust 1.85.0 release notes](https://releases.rs/docs/1.85.0/)
- [internals.rust-lang.org — f64::midpoint discussion](https://internals.rust-lang.org/t/f64-midpoint/23491)
- [Rust issue #21690 — Imprecise FP operations (fast-math)](https://github.com/rust-lang/rust/issues/21690)
- [Rust libs-team ACP #532 — algebraic FP intrinsics](https://github.com/rust-lang/libs-team/issues/532)
- [Rust PR #136457 — expose algebraic FP intrinsics](https://github.com/rust-lang/rust/pull/136457)
- [Rust PR #120718 — algebraic fast-math intrinsics](https://github.com/rust-lang/rust/pull/120718)
- [Rust tracking issue #136468 — algebraic FP operations](https://github.com/rust-lang/rust/issues/136468)
- [Rust tracking issue #136469 — algebraic FP methods](https://github.com/rust-lang/rust/issues/136469)
- [RFC 3514 — float semantics](https://rust-lang.github.io/rfcs/3514-float-semantics.html)
- [RFC 3453 — f16 and f128](https://rust-lang.github.io/rfcs/3453-f16-and-f128.html)
- [LLVM Numerics Blog (2019)](https://blog.llvm.org/2019/03/llvm-numerics-blog.html)
- [LLVM IR fast-math flags reference (LangRef)](https://llvm.org/docs/LangRef.html#fast-math-flags)
- [IEEE 754-2019 augmented arithmetic emulation paper](https://ieeexplore.ieee.org/document/9117154/)
- [Go src/math/nextafter.go](https://go.dev/src/math/nextafter.go)
- [Go issue #42613 — Nextafter ±0 behavior](https://github.com/golang/go/issues/42613)
- [Go pkg.go.dev/math reference](https://pkg.go.dev/math)

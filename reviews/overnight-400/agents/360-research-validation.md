# 360 — research-validation (cross-language float validation infrastructure)

## Headline
Reality's "Go is canonical, Python/C++/C# validate against golden JSON" framework has a latent 2-ulp hazard for transcendentals; the fix is to generate goldens from a correctly-rounded oracle (CORE-MATH or mpmath at high-precision-with-CR-check), encode bit-exact via hex-float `%a`, and reserve TestFloat-3 / IBM FPgen for the IEEE-754 ops layer.

## Survey

### 1. CR-LIBM (LRI Lyon / Inria, ENS-Lyon)
LGPL-2.1. Correctly-rounded binary64 elementary functions (exp, log, log2, log10, sin, cos, tan, sinpi/cospi/tanpi, asin, acos, atan, sinh, cosh, expm1, log1p, exp2, pow-RN). Worst-case bounded by Lefèvre–Muller table-maker's-dilemma search; algorithms have Gappa/Coq proofs (de Dinechin–Lauter–Melquiond). Consume as a C library or via Julia's CRlibm.jl, Rust crlibm, or PyPI `crlibm` wrapper.
**Reality-relevance:** highest-quality oracle for the dozen transcendentals reality cares about, but LGPL infects redistribution if linked. Use it OFFLINE during golden generation only — write hex-float results into JSON fixtures, then ship the fixtures (data, not code) under MIT. This is reality's correctly-rounded ground truth source for transcendentals — superior to mpmath because it is per-function ulp-proven, not heuristic-precision.

### 2. CORE-MATH (Inria — Sibidanov, Zimmermann, Glondu)
**MIT licensed.** Successor in spirit to CR-LIBM; correctly-rounded reference C99 implementations explicitly intended to be embedded. Full binary32 set already done; binary64 set growing (~30 functions integrated upstream into glibc as of 2024). Source: `gitlab.inria.fr/core-math/core-math`. Each function ships standalone C, vetted against MPFR, with worst-case ulp tables.
**Reality-relevance:** **this is reality's right answer.** MIT license matches. Use `core-math` as the binary64 oracle for transcendental golden generation. For each `Sin(x)`, `Exp(x)`, etc., compile the CORE-MATH C reference, compute correctly-rounded `f(x)` for the 30 fixture inputs, write the resulting bit-exact `float64` into JSON. Then Go/Python/C++/C# implementations target THAT value with a per-function ulp tolerance reflecting the platform's libm guarantee (Go: 1 ulp; mainstream libm: 1 ulp; result: ≤1 ulp diff against oracle, not 2).

### 3. RLIBM (Rutgers — Lim, Aanjaneya, Nagarakatte)
MIT-style open source. POPL 2021 + PLDI 2021 + arXiv 2104.04043, 2108.06756, 2111.12852, 2504.07409. Generates polynomials approximating the *correctly-rounded result* directly via LP, not the real value — a different theoretical attack than CR-LIBM. Coverage: 32-bit float (RLIBM-32), bfloat16, posit16, multi-rounding (RLibm-MultiRound 2024). Notably **less binary64 coverage** than CR-LIBM/CORE-MATH at present.
**Reality-relevance:** secondary cross-check oracle for binary32-relevant code paths (none in reality currently — reality is binary64-only). Useful as theoretical backstop: if CORE-MATH and RLIBM agree on `f(x)` for a binary32 down-cast of a binary64 fixture, confidence is doubled. Skip for v1.

### 4. Boldo–Daumas–Melquiond formal proofs (Flocq + Gappa + Coq)
Open source (LGPL Coq libs). Flocq formalizes IEEE-754 in Coq; Gappa discharges interval/ulp goals; Sollya generates polynomial approximations. Used to verify CR-LIBM individual functions, FMA emulation, rounding-to-odd, Kahan summation. Not a runnable test suite — a *proof methodology* for individual function implementations.
**Reality-relevance:** out of scope for an MIT library targeting four languages. But: when reality's algorithm citations claim "1 ulp accuracy on [a,b]" they should cite a Gappa-proved bound (e.g., from the CR-LIBM proofs) rather than asserting it. Pull in Gappa-proved ulp bounds as the per-function tolerance metadata.

### 5. IBM FPgen (Aharon, IBM Haifa)
Closed source generator; **public test vectors freely downloadable** (`research.ibm.com/haifa/projects/verification/fpgen/`). Mirrored at `github.com/sergev/ieee754-test-suite`. ~80,000+ test vectors covering IEEE 754 ops (add, sub, mul, div, sqrt, fma, conversions) + edge cases (Inf, NaN, denormals, signed zero, all rounding modes). Format: ASCII, one test per line, hex-float operands and expected result.
**Reality-relevance:** validates that **the language runtime's IEEE 754 ops** are conformant — pre-condition for reality's tests to mean anything. Reality should run FPgen vectors once per supported (Go, Python, C++, C#) × (platform) matrix in a separate "platform sanity" CI job. Not per-package; once. If a platform fails IEEE-754 conformance, all reality goldens are suspect on that platform.

### 6. Berkeley TestFloat-3 / SoftFloat-3 (John Hauser, UC Berkeley)
**Open source (BSD).** `github.com/ucb-bar/berkeley-testfloat-3`. Compares hardware/library FP against Hauser's bit-exact SoftFloat-3 reference. Covers binary16/32/64/80/128, all five rounding modes, all IEEE 754 ops including FMA. Generator (`testfloat_gen`) emits cases; checker (`testfloat`) diffs against SoftFloat. Used by RISC-V cores, GPU vendors, glibc.
**Reality-relevance:** stronger than FPgen for the 754 ops layer (active maintenance, BSD, generates novel cases). Same role as FPgen — a one-time platform-sanity dependency, not a reality fixture. Crucially **does not cover transcendentals**; it is silent on `sin/cos/exp/log` and reality's actual content. Pair with CORE-MATH for full coverage.

### 7. mpmath (Fredrik Johansson)
**BSD licensed.** Pure-Python arbitrary-precision FP; Sage/SymPy depends on it. **Critical caveat from its own docs**: basic arithmetic in `mp` context is correctly rounded; **higher-level transcendentals are not guaranteed CR** — mpmath bumps working precision and trusts a tolerance. So `mpmath.sin(x, prec=200)` rounded to binary64 is "almost certainly" correctly rounded but not provably so for adversarial inputs near halfway points (the table-maker's-dilemma cases CR-LIBM/CORE-MATH solved).
**Reality-relevance:** convenient as a **secondary** oracle and as the only practical option for functions outside CORE-MATH's coverage (incomplete gamma, Bessel, hypergeometric, etc.). Strategy: for each special function, generate at 200-bit precision, round to binary64, and *also* generate at 400-bit precision; if both round to the same float64, lock that as the golden. If they differ, escalate (use Arb or hand-verify).

### 8. Arb / FLINT (Fredrik Johansson)
LGPL-2.1+. C library for ball arithmetic — every value carries a rigorous error radius. Going up in precision shrinks the ball; if the ball excludes the halfway point, the rounded float64 is provably correctly rounded.
**Reality-relevance:** the **principled** way to use mpmath-style oracles. Wrap special-function golden generation as: bump precision until Arb's ball is contained between two consecutive float64 midpoints; emit that float64. This gives a *proof* of correct rounding for each fixture, not just heuristic. License: LGPL — same offline-generator/MIT-fixture pattern as CR-LIBM.

### 9. Hex-float ASCII (`%a` / C99 / Go `%x`)
Standardized in C99, C++17, IEEE 754. Format: `0x1.fffffffffffffp+1023`. **Bit-exact** round-trip via `printf("%a")` / `strtod` and Go `strconv.FormatFloat(f, 'x', -1, 64)` / `ParseFloat`. Python: `float.hex()` / `float.fromhex()`. C#: `BitConverter.DoubleToInt64Bits` (better — encode raw uint64). **No language standard guarantees decimal round-trip is bit-exact across libm versions** — only 17 significant digits + correct rounding mode in the parser, which decimal-literal parsers historically have gotten wrong.
**Reality-relevance:** **mandatory** for reality's JSON fixtures. Decimal floats in JSON are a cross-language hazard (Go `1.1` and Python `1.1` are bit-equal, but `0.1+0.2` is not, and JSON parsers vary). Reality should encode every float in fixtures as either hex-float string `"0x1.999999999999ap-4"` or as `uint64` raw bits in a `"bits"` field. Go's `math.Float64bits` ⇄ Python's `struct.unpack('<Q', struct.pack('<d', x))` ⇄ C++ `std::bit_cast<uint64_t>(d)` ⇄ C# `BitConverter.DoubleToUInt64Bits` is a 4-line cross-language round-trip. **This is the single highest-leverage fix.**

### 10. Sollya (Inria)
CeCILL-C (LGPL-equivalent). Interactive shell + library for polynomial approximation, ulp-bounded code generation, supnorm computation. Used by CR-LIBM authors to *generate* the polynomial approximations and bound their approximation error.
**Reality-relevance:** not a validator; a generator. Pair with Gappa: if reality wants to ship its OWN libm (it shouldn't — it should call platform libm), Sollya generates the polynomial, Gappa proves the ulp bound. Out of scope until/unless reality reimplements transcendentals.

### 11. FPBench / FPCore
MIT-licensed corpus + S-expression IR. `fpbench.org`. Standardizes how floating-point benchmarks are described so tools (Herbie, Daisy, FPTuner, Gappa, Rosa) can consume them.
**Reality-relevance:** zero direct relevance — FPBench targets *expression-level* optimization (rewrite `sqrt(x+1)-sqrt(x)` for accuracy), not library-function validation. Reality's functions are leaves, not expressions. Worth knowing for context only.

### 12. Herbie / Daisy / Rosa
MIT/BSD. Herbie auto-improves expression accuracy; Daisy bounds round-off error of a program statically. Both consume FPCore.
**Reality-relevance:** could be used to *audit* reality's user-callable wrappers (e.g., does `physics.KineticEnergy(0.5*m*v*v)` have a catastrophic-cancellation pattern Herbie would rewrite?) but not to validate cross-language matches. Park as a future "automated numerical-quality" CI job; not on the validation-infrastructure critical path.

## Reality's cross-language validation strategy

The latent bug in the current spec:

> Go is canonical; Python/C++/C# validate against golden files

If Go's `math.Sin` is ≤1-ulp-accurate and Python's `math.sin` (cpython libm) is ≤1-ulp-accurate, both relative to the *real* value, they can disagree by **up to 2 ulp** on the same input. Reality's per-function tolerance of "1e-11 for transcendentals" papers over this, but it (a) is way looser than necessary and (b) silently degrades to even worse on platforms where libm is 2-ulp-accurate (some embedded glibcs, MSVC pre-2019).

**Recommended fix (in priority order):**

1. **Switch the golden-generator from "Go's `math.X`" to a correctly-rounded oracle.** For functions in CORE-MATH's coverage, use CORE-MATH (MIT, no license tension). For functions outside CORE-MATH (incomplete gamma, Bessel, etc., per slot 359), use Arb-with-CR-check or mpmath at 2× precision. Document this in `testutil/`.

2. **Encode floats bit-exact.** Add a `bits` field (uint64 hex string) to every fixture JSON entry. Decimal is for humans; bits are for tests. This unblocks bit-exact comparison for IEEE-754 ops (which ARE deterministic across languages) and tightens the comparator for transcendentals (now a single-ulp window against a CR oracle instead of a two-ulp window against another libm).

3. **Stratify tolerances by function class:**
   - IEEE-754 ops (add, sub, mul, div, sqrt, fma, comparisons, conversions): **0 ulp** — bit-exact across all four languages, validated via TestFloat-3 / FPgen at the platform layer once.
   - Polynomial / rational reality functions (most of `linalg`, `geometry`, `combinatorics` integer ops): **0 ulp** — assuming careful use of `math.FMA` / accurate dot products.
   - Transcendentals consumed via platform libm: **1 ulp** vs the CORE-MATH golden, on the assumption every supported platform's libm is ≤1 ulp. Document this assumption per-platform; fail loudly when violated.
   - Iterative / accumulating ops (long sums, FFTs, ODEs): **per-function relative tolerance** documented with provenance ("Higham §3.1 forward error bound for n=1024").

4. **Run TestFloat-3 + FPgen as a separate "platform sanity" CI job**, once per (OS × language-runtime) cell. Output is a green/red badge for the platform; reality's actual fixtures only run if green. This isolates platform-IEEE-conformance bugs (rare but real, esp. on Windows ARM, WASM, RISC-V) from reality bugs.

5. **Cross-link to slot 359** (research-correctness, test corpora): slot 359 catalogs *what* to test against (DLMF, CALGO, NIST CAVP, mpmath-as-oracle). Slot 360 (this) covers *how* the result-comparison plumbing works across languages. The two are complementary: 359 tells you which inputs and reference values to use; 360 tells you how to encode them in JSON so that Go, Python, C++, C# all read back the bit-identical bytes.

6. **Future: when reality grows a `complex` package**, redo this analysis — complex transcendentals (`csin`, `clog`, branch cuts) are a separate minefield. CR-LIBM does not cover them; mpmath does but with the same heuristic-precision caveat.

The single highest-leverage change for v1 is **(2) hex-bit encoding in JSON**. Everything else is incremental; (2) closes an entire class of cross-language drift bugs at the parser layer.

## Sources
- [CRlibm at LRI/ENS-Lyon — HAL](https://ens-lyon.hal.science/ensl-01529804/document)
- [CORE-MATH project (Inria, MIT-licensed)](https://core-math.gitlabpages.inria.fr/)
- [CORE-MATH GitLab](https://gitlab.inria.fr/core-math/core-math)
- [RLIBM-32 (Lim, Nagarakatte, PLDI'21)](https://arxiv.org/abs/2104.04043)
- [RLIBM POPL 2021 paper](https://people.cs.rutgers.edu/~sn349/papers/rlibm-popl-2021.pdf)
- [RLibm-MultiRound 2024](https://arxiv.org/abs/2504.07409)
- [Boldo/Melquiond Flocq + Gappa survey](https://guillaume.melquiond.fr/doc/14-mscs.pdf)
- [Daumas–Melquiond Gappa TOMS 2009](https://dl.acm.org/doi/10.1145/1644001.1644003)
- [Berkeley TestFloat-3 (Hauser)](http://www.jhauser.us/arithmetic/TestFloat.html)
- [TestFloat-3 GitHub (UCB-BAR)](https://github.com/ucb-bar/berkeley-testfloat-3)
- [IBM FPgen test suite mirror](https://github.com/sergev/ieee754-test-suite)
- [FPgen paper (HLDVT'03)](https://ieeexplore.ieee.org/document/1252469/)
- [mpmath precision/representation docs](https://mpmath.org/doc/current/technical.html)
- [Lefèvre–Muller Table Maker's Dilemma](https://perso.ens-lyon.fr/jean-michel.muller/Intro-to-TMD.htm)
- [FPBench standard](https://fpbench.org/)
- [GCC hex-float docs](https://gcc.gnu.org/onlinedocs/gcc/Hex-Floats.html)
- [Go math accuracy discussion (golang-dev)](https://groups.google.com/g/golang-dev/c/WpAWBRFD6mI)
- [Go math.Pow accuracy issue #25270](https://github.com/golang/go/issues/25270)

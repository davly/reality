# 394 — meta-cross-language (Python/C++/C# validation parity audit)

## Headline
Cross-language validation in `reality` is **aspirational only**: zero Python/C++/C#
sources exist in the repo, golden JSON encodes floats as decimal literals (no
`bits` field), and the Go-as-canonical generator inherits any libm wobble — so
the four-language parity story in `CLAUDE.md` and `testutil/golden.go` doc is at
this moment unverified marketing.

## Current state of cross-language validation

- **Sources present**: Go only. Glob over the tree finds zero `*.py`, `*.cpp`,
  `*.c`, `*.h`, `*.hpp`, `*.cs` files (verified). No `validators/`, `ports/`,
  `bindings/`, or `scripts/` directory exists. Top-level dirs are 22 Go
  packages plus `docs/`, `info/`, `pkg/`, `reviews/`, `testdata/`, `testutil/`.
- **Golden infrastructure**: `testutil/golden.go:1-95` describes the JSON
  format and load path; `LoadGolden` resolves paths via `runtime.Caller`. Schema
  is `{function, cases:[{description, inputs, expected, tolerance}]}`. No
  `bits` / hex-float field. No `oracle` / `provenance` / `git_sha` field.
- **Golden generation oracle**: nothing in-tree. CLAUDE.md claims "Go generates
  golden files via `math/big` at 256-bit precision"; in practice the JSONs are
  hand-written decimal literals (e.g. `acoustics/sound_speed.json:7`:
  `"expected": 343.26283173742496`). No `gen.go`, no `goldengen` build tag, no
  CORE-MATH/MPFR/Arb shim.
- **Tolerance distribution (sampled)**:
  - Exact / integer-valued: `0` (combinatorics/binomial_coeff.json `Tolerance:0`).
  - Polynomial / rational: `1e-15` (color/srgb_linear, compression/entropy).
  - Trig/transcendental composites: `1e-14` … `1e-10` (acoustics/sound_speed
    `1e-10`, compression mixed `1e-14`/`1e-15`).
  - Iterative algorithms: looser, e.g. CIEDE2000 `0.0001` (color/delta_e2000),
    fft `1e-10`-class.
  - One legitimately suspicious value: `combinatorics/binomial_coeff.json:44`
    sets tolerance `1e16`. That is "any answer within 10^16" — almost
    certainly a typo for `1e-16` or for a rel-tol on a 10^N-magnitude binomial.
    Flag separately to slot 393 (review-everything) or a fix-it slot.
- **Comparator**: `AssertFloat64` (golden.go:103-128) does **absolute**
  difference: `math.Abs(got-expected) <= tolerance`. No relative, no ULP, no
  hex-bit comparator. Special-value handling is correct for ±Inf/NaN; -0 is
  treated as 0, subnormals untreated explicitly.
- **CI matrix**: `go.mod` declares one module; reviews/overnight-400/MASTER_PLAN
  references no GOAMD64-split CI (slot 380 also flagged absent). No GitHub
  Actions or workflow files surfaced via Glob. Per-arch determinism unverified.
- **Net**: golden files exist (≥30 packages, ≥80 JSON files), only Go reads
  them, and nothing in the repo cross-validates. The "Python/C++/C# validate"
  clause in CLAUDE.md is a design promise, not a tested property.

## Latent bugs (cross-link slot 360, slot 380, slot 359)

- **Two-libm 2-ulp drift (slot 360)**. With Go's `math.Sin` ≤1 ulp vs platform
  C `sin(3)` ≤1 ulp, both relative to the real value, they may disagree by up
  to 2 ulp. Reality's transcendental-class tolerance of `1e-10`…`1e-14`
  paper-covers this; a stricter `1 ulp` window (≈ 2.2e-16 × |x|) would expose
  the wobble. The fix slot 360 recommends — switch the oracle to CORE-MATH
  (MIT) and tighten tolerance to ≤1 ulp vs. the correctly-rounded reference —
  has not been applied.
- **Decimal-literal round-trip fragility**. JSON parsers in Go, Python, C++,
  C# all converge on IEEE-754 round-half-to-even when round-tripping 17-digit
  decimal floats — but historically multiple parsers have shipped bugs (Java
  pre-1.5, MSVCRT pre-2010, glibc strtod pre-2.17). With `expected: 343.262...`
  written as decimal, a future C# port using a non-conformant parser would see
  a 1-ulp drift sourced entirely in JSON parsing, not in the math. **Fix**: add
  a `bits` field carrying the `uint64` raw IEEE bits (Go `math.Float64bits`,
  Python `struct.unpack('<Q', struct.pack('<d', x))`, C# `BitConverter.
  DoubleToUInt64Bits`, C++ `std::bit_cast<uint64_t>`). Highest-leverage single
  change per slot 360 §2.
- **Go 1.25 silent FMA fusion at GOAMD64=v3 (slot 380 §3)**. Go 1.25 (Aug
  2025) fuses `a*b + c` to a single VFMADD231SD with one rounding when
  GOAMD64=v3+. That is a **different** float result than `(a*b)+c` with two
  roundings, which is what Go 1.24 emits at v1, what Python's interpreter
  emits, and what most C/C++ compilers emit unless `-ffp-contract=on`. So a
  golden generated under Go 1.24/v1 will mismatch on Go 1.25/v3, and a Go-1.25
  generated golden will mismatch the Python validator. Reality currently has
  no `GOAMD64` setting in `go.mod`, no per-arch CI, and no policy on
  `math.FMA` vs `a*b+c`. linalg dot products, prob/conformal Horner forms,
  signal/fft butterflies, and any Kahan compensated summation are exposed.
  Fix: either explicit `math.FMA(a,b,c)` (one rounding, deterministic across
  arch) or explicit `float64(a*b)+c` (two roundings, defeats fusion). Pick one
  policy per package and document.
- **Subnormals + flush-to-zero**. `AssertFloat64` does not check the FTZ/DAZ
  state of the runtime. C# under x64 on .NET 6+ leaves DAZ off by default;
  Python is ABI-dependent on the underlying C runtime; Go disables FTZ
  (`runtime/asm_amd64.s` initializes MXCSR with FTZ=0). A C++ port linking
  against a library that flips FTZ (some game-engine CRTs, OpenMP under
  `-ffast-math`) would silently zero subnormals — and reality has no fixture
  in the surveyed JSONs that exercises a 1e-310 subnormal input. Per CLAUDE.md
  rule "IEEE 754 edge cases mandatory: +Inf, -Inf, NaN, -0.0, subnormals" this
  is a coverage gap. Sampled `acoustics/sound_speed.json` has zero subnormal
  and zero NaN cases; `combinatorics/binomial_coeff.json` and
  `color/srgb_linear.json` similarly. The mandate is not enforced.
- **The `1e16` outlier** (`combinatorics/binomial_coeff.json:44`). A tolerance
  of 10^16 means any finite float passes. Either the test is degenerate, or
  this is a typo for `1e-16` against a ~10^0 binomial expected value. Flag for
  fix.
- **Test-runtime PRNG drift (slot 380 §1)**. `math/rand` (Go 1) is non-portable
  to other languages. If reality migrates to `math/rand/v2.NewPCG(s1,s2)` the
  PCG64-DXSM stream is reproducible by `numpy.random.Generator(PCG64)` and
  `pcg-cpp`. Currently 19 Go files import legacy `math/rand`; any test that
  uses a random input grid is non-portable. For golden-file generation this is
  a one-time fix; for fixture *inputs* that are randomly drawn, the seed-replay
  contract is invisible cross-language.
- **The 1e-10 transcendental tolerance slot 360 flagged is observed in-tree**:
  `acoustics/sound_speed.json` uses `1e-10`; `compression/shannon_entropy.json`
  uses `1e-14`/`1e-15` (it sums `-p*log2(p)` — log is the only transcendental,
  but the sum amplifies). 1e-10 is ~10^6 ulp at magnitude 343 — vastly looser
  than the 2-ulp drift it nominally guards.

## Recommended roadmap

1. **Stop calling the design "four-language" until it is**. Edit
   `testutil/golden.go:1-24` and `CLAUDE.md` Architecture/Golden-File sections
   to read "Go is canonical; the JSON schema is **designed to be**
   cross-language; Python/C++/C# ports are an open work-item, not a shipped
   feature." Truth-in-advertising. Five-line change.

2. **Add `bits` to the JSON schema** (slot 360 §2 — highest leverage).
   New schema: `{description, inputs, expected, expected_bits?, tolerance,
   tolerance_ulp?}` where `expected_bits` is a 16-char hex `uint64` of
   `math.Float64bits(expected)`. Backward-compatible: when `expected_bits` is
   present, comparator does bit-exact + ULP-distance check; when absent, falls
   back to absolute-tolerance (current behavior). Update `LoadGolden` /
   `AssertFloat64`. Add `tolerance_ulp` for transcendentals (replacing
   `1e-10`-style absolute thresholds). Migrate one package per PR.

3. **Pick the oracle and generate, don't hand-write**. For functions in CORE-MATH
   coverage (sin, cos, exp, log, pow, erf, gamma — most of what reality
   touches): write a `tools/gen-golden/main.go` that compiles CORE-MATH offline
   and emits `{x, expected, expected_bits}` triples for each fixture. For
   functions outside CORE-MATH (Bessel, incomplete gamma, hypergeometric per
   slot 359): use Arb/FLINT with a CR-check (radius excludes float64
   midpoint). Encode the oracle and its commit hash in a top-level
   `provenance` field per file. CORE-MATH is MIT — no license tension.

4. **Stratify tolerances** (slot 360 §3, slot 393 numerical stability):
   - IEEE-754 ops & polynomial/integer functions: `0 ulp` (bit-exact).
   - Platform-libm transcendentals via `math.X`: `1 ulp` vs CR oracle.
   - Iterative / accumulating: `n*epsilon` per documented Higham bound,
     citation in the JSON `provenance` field.
   Audit current thresholds — flag `1e-10` and `1e16` as suspect.

5. **Establish the CI matrix** (slot 380):
   - GOAMD64=v1 vs v3 (FMA fusion).
   - Linux/glibc, macOS/Apple-libm, Windows/UCRT (libm differs).
   - For any cell that fails, fail loud — that's the validation working.
   Until a non-Go validator exists, this matrix at least proves Go-Go bit
   stability across libm variants; it is the Phase-0 prerequisite to any
   Python/C++/C# port.

6. **Pilot a Python validator**. One package, one fixture file, one CI job:
   `validators/python/test_acoustics_sound_speed.py` reads
   `acoustics/testdata/acoustics/sound_speed.json`, calls a
   `validators.python.acoustics.SoundSpeed(...)` reimplementation, asserts
   bit-equal via `bits` field. This proves the JSON schema cross-decodes,
   surfaces parser hazards, and gives a working template before scaling.
   Estimate: 1 dev-day for one pilot; ~30 dev-days to cover all 80 JSONs.

7. **Document the FMA policy** in `CONTEXT.md`. Per package, decide:
   - "uses `math.FMA(a,b,c)` explicitly for determinism" (linalg, signal/fft,
     prob/conformal); or
   - "writes `float64(a*b)+c` to defeat fusion" (calculus, wherever 2-rounding
     parity with C/Python matters); or
   - "non-deterministic across arch — golden tolerance ≥ 1 ulp absorbs it".
   Pick one per package. Today there is no policy.

8. **Cover the IEEE 754 edge cases mandated by CLAUDE.md**. Add at least one
   subnormal input, one NaN input, one ±Inf input, one ±0.0 input per fixture
   file. Auto-grade in CI: a fixture file lacking these four is a CI failure.
   Today, sampled files (sound_speed, srgb_linear, binomial_coeff) lack all
   four; the mandate is on paper only.

9. **TestFloat-3 / FPgen platform-sanity job** (slot 360 §1, §6). Once per
   (OS × runtime) cell, run Berkeley TestFloat-3 (BSD, covers IEEE 754 ops on
   the runtime). If it red-flags the platform, all reality goldens are suspect
   on that platform. This is a one-shot upstream gate, not per-package.

10. **Cross-link to slot 359 corpora**. Slot 359 catalogs *what* reference
    values to use (DLMF, mpmath, SciPy xsref, Boost.Math). Slot 360 covers
    *how* to encode them. Slot 394 (this) covers *that nothing has been
    encoded yet*. The dependency order is: 360-style schema fix first
    (`bits` field), then 359-style oracle ingestion (mpmath/Arb/CORE-MATH),
    then per-language ports. Doing the ports first against the current
    decimal-only schema would be strictly wasted work — they would all need
    redo when the schema gains `bits`.

## Sources

- `C:\limitless\foundation\reality\CLAUDE.md` — Architecture, Golden-File
  Testing Infrastructure, Key Design Rules.
- `C:\limitless\foundation\reality\testutil\golden.go:1-265` — JSON schema,
  comparator, no `bits` field, absolute-difference comparator.
- `C:\limitless\foundation\reality\testutil\testdata\sample_golden.json` —
  reference example showing decimal-only encoding.
- `C:\limitless\foundation\reality\acoustics\testdata\acoustics\sound_speed.json:7`
  — example of decimal-literal expected and `1e-10` transcendental tolerance.
- `C:\limitless\foundation\reality\combinatorics\testdata\combinatorics\binomial_coeff.json:44`
  — anomalous `1e16` tolerance (likely typo).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\360-research-validation.md`
  — CR-LIBM / CORE-MATH / Arb / TestFloat-3 / hex-float schema fix.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\380-research-go-math-extras.md`
  — Go 1.25 GOAMD64=v3 FMA fusion; `math/rand/v2`; per-arch CI matrix.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\359-research-correctness.md`
  — mpmath / DLMF / CALGO / MPFR / Arb / SciPy xsref / Boost.Math corpora.
- [issue 71204: GOAMD64=v3 FMA auto-fusion](https://github.com/golang/go/issues/71204)
- [Go Wiki: Minimum Requirements / GOAMD64 levels](https://go.dev/wiki/MinimumRequirements)
- [Bruce Dawson — Comparing Floating Point Numbers, 2012 Edition](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)
- [Bit Bashing — Comparing Floating-Point Numbers Is Tricky](https://bitbashing.io/comparing-floats.html)
- [google/truth #690 — ULP-based float assertions](https://github.com/google/truth/issues/690)
- [googletest discussion — 5+ ULP cross-machine drift](https://groups.google.com/g/googletestframework/c/7ZoZWlfxpg4)
- [Wikipedia — Unit in the last place](https://en.wikipedia.org/wiki/Unit_in_the_last_place)
- [CORE-MATH (MIT-licensed correctly-rounded oracle)](https://core-math.gitlabpages.inria.fr/)
- [Berkeley TestFloat-3 (BSD, IEEE 754 ops conformance)](https://github.com/ucb-bar/berkeley-testfloat-3)

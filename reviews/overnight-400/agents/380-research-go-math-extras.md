# 380 — research-go-math-extras (Go stdlib math + x/exp + Block-E close-out)

## Headline
Reality (`go 1.24`) is parked on legacy `math/rand`; the priority migrations are `math/rand/v2` (PCG/ChaCha8, since 1.22), `encoding.TextAppender`/`BinaryAppender` interfaces (1.24), and an explicit `math.FMA` policy for hot dot-products under GOAMD64=v3.

## Survey

### 1. math/rand/v2 (Go 1.22, Feb 2024) — strongest "should adopt" signal
Replaces the global Go 1 LCG with PCG64-DXSM (`rand.PCG`) and ChaCha8Rand (`rand.ChaCha8`, default for global funcs). ChaCha8 is a modified DJB ChaCha8 stream cipher (32-byte seed, 8 rounds, ~2.5× ChaCha20 per Aumasson "Too Much Crypto"). API cleanups: `Uint64`, `Uint32`, `IntN`/`UintN` with explicit upper bound, removed `Read` deprecation footgun, `N[T]` generic helper, `ChunkedSeq`. Reality currently imports `math/rand` in 19 files (calculus, changepoint, gametheory, optim, prob/conformal, timeseries/dcc+garch, topology/persistent, etc.). Cross-language golden-file replay requires deterministic seed semantics — `math/rand/v2.NewPCG(seed1, seed2)` is the right primitive because PCG is portable to C++ (pcg-random.org) and Python (numpy.random.Generator(PCG64)). ChaCha8 is also reproducible across languages but encoding seeds is more involved. **Recommendation:** migrate test PRNGs to `rand/v2.PCG`; keep determinism via explicit `New(NewPCG(s1,s2))`. Cross-references slot 376 (PCG XSH-RR finding) — DXSM is the modern variant Go ships, slot 376's "obsolete" critique applies to *XSH-RR*, not DXSM.

### 2. Go 1.24 (Feb 2025) — encoding.TextAppender / BinaryAppender
Three math/big types (`Float`, `Int`, `Rat`) implement `encoding.TextAppender`. `rand/v2.PCG` and `rand/v2.ChaCha8` implement `encoding.BinaryAppender`. Top-level `math/rand.Seed` is now a no-op (revert via `GODEBUG=randseednop=0`). `math` and `math/cmplx` unchanged. Reality's `testutil` golden-file marshalers can drop a `bytes.Buffer` allocation per vector by switching from `MarshalText` to `AppendText` once mass-emitting vectors at 256-bit precision; modest, but free.

### 3. Go 1.25 (Aug 2025) — implicit FMA fusion at GOAMD64=v3
No direct math-package additions. Critical for reality: at `GOAMD64=v3` or higher, the compiler now fuses `a*b + c` into a single VFMADD231SD with one rounding (improved precision *and* speed). Without v3, a guarded `math.FMA` call still uses runtime feature detection. **Implication:** linalg dot products, polynomial evaluation (`prob/conformal`, `timeseries/garch`), Kahan summation residuals, and any Horner-form code in `calculus`/`signal` will silently change rounding when users build with v3. To preserve cross-language float bit-equality (slot 360 finding), reality must either (a) document a v3 build matrix and regenerate goldens per-arch, or (b) defeat fusion explicitly by writing `float64(a*b) + c`. Also: `crypto/sha1` is 2× faster on amd64 (SHA-NI) — relevant if reality grows hash-test fixtures.

### 4. Go 1.26 (Feb 2026) — language `new()` enhancements, no math/* changes
Released 2026-02-10. Standard math packages: zero documented changes per release notes. Notable side-effects: Green Tea GC default-on (less GC tail latency in `optim/genetic` long-running searches), generic-type-constraint relaxations, `crypto/hpke`, `crypto/mlkem/mlkemtest`, `testing/cryptotest`. Cross-link slot 371 (PQ 2026) for `crypto/mlkem` — reality should *not* duplicate ML-KEM but may want to validate against `cryptotest` patterns.

### 5. golang.org/x/exp/constraints — frozen, not promoted
The constraints package was *not* promoted to stdlib alongside `slices` and `maps` (1.21). The Go team's stated reason: in practice most generic code uses `any`/`comparable` and hand-rolled local constraints. `constraints.Float`, `constraints.Integer`, `constraints.Ordered` remain canonical for math libraries but live indefinitely under `golang.org/x/exp` (a dependency). For reality's "zero dependencies" rule this is a hard wall: importing `x/exp/constraints` violates the design rule. **Decision:** define reality-internal constraints in a new `internal/numeric` package (`type Float interface { ~float32 | ~float64 }`, `type Real interface { Float | Integer }`). Slot 357 (research-libs-go) noted reality's USP is pure-stdlib + MIT + zero-dep; pulling x/exp would forfeit one third of that triple.

### 6. golang.org/x/exp/rand — deprecated 2024–2025
Per upstream issue golang/go#71373, `x/exp/rand` is documented as superseded by `math/rand/v2` and scheduled for tag-and-delete (golang/go#61716). Gonum historically used it; gonum's migration is the canonical reference if reality adopts numerical-distribution sampling. No effect on reality today (no x/exp imports), but a reminder: any "we'll just import gonum a little" temptation would surface this dependency.

### 7. math/big.Float / math/big.Rat — stable, mostly inert
Repeatedly-proposed `Float.Pow`, `Float.Exp`, `Float.Ln` (golang/go#14102, open since 2016) remain unimplemented. No 2024–2026 performance overhauls. Reality's `testutil` uses `math/big` at 256-bit precision for golden-file generation — that path is correct and stable. For *runtime* arbitrary precision, `math/big` is slow vs. MPFR/Arb; the `ericlagergren/decimal` library is the typical Go alternative but adds a dependency. Since reality's golden-file path is offline (test-time only), big.Float perf is not on the critical path.

### 8. math.FMA, Round-family, Modf — long-stable primitives
`math.FMA` (1.14, 2020): single-rounding multiply-add, intrinsic on amd64 with FMA, ARM64, riscv64, ppc64. `math.Round`, `math.RoundToEven`, `math.Trunc` (1.10): IEEE 754-2008 rounding modes. `math.Modf`: integer/fractional split. **Audit need:** reality's Kahan-summation code paths and any `a*b+c` accumulators in `linalg`, `signal`, `prob`, `optim/transport` should be reviewed: explicit `math.FMA(a,b,c)` is bit-stable across architectures (compiler always emits one-rounding form), whereas `a*b+c` is bit-stable *only* with explicit `float64(a*b)+c` cast (Go 1.25+ guarantees fusion at v3 otherwise). For golden-file determinism this matters.

### 9. math/cmplx — frozen since 1.0
No changes 1.22→1.26. Notable absences vs. C99: no `cmplx.Acos` *fma*-precision variant, no `csqrt` correctly-rounded guarantee. Reality's `signal/fft` and any complex polynomial roots in `optim` should not over-trust `cmplx.*` for last-bit accuracy. Cross-reference slot 360 (validation) and slot 377 (numerical stability).

### 10. go vet & static-analysis for numerics
`go vet` ships no floating-point-specific analyzers (no NaN-comparison check, no `==` on floats lint, no FMA-vs-non-FMA fusion warning). `nilness` (in `golang.org/x/tools/go/analysis/passes/nilness`) is not numeric. Third-party: `gosec`, `staticcheck`, `golangci-lint` — none target numerical bugs (e.g., catastrophic cancellation, Kahan-broken patterns, `==` on floats, `int → float64` precision loss above 2^53). **Gap:** reality could ship its own `cmd/realityvet` analyzer pass that flags (a) `==` between `float64` operands, (b) `int64 → float64` conversions of values that may exceed 2^53, (c) `math.Sqrt(x*x + y*y)` (should be `math.Hypot`), (d) `math.Log(1+x)` (should be `math.Log1p`). Slot 377 (numerical stability) is the natural home for the rule list.

## Reality slot recommendations

- **prob/, optim/, gametheory/, timeseries/dcc+garch, topology/persistent, calculus/, changepoint/ (test files):** migrate `math/rand` → `math/rand/v2.New(rand.NewPCG(s1,s2))`. Keeps cross-language determinism (PCG64-DXSM is portable). Re-emit affected golden vectors *once*, then PCG-stable forever.
- **testutil/:** when Go floor lifts to 1.24+, switch golden-file emitters from `MarshalText` → `AppendText` on `big.Float`/`big.Rat`/`big.Int` to drop per-vector allocations.
- **go.mod:** stay on `go 1.24` until Q3 2026 (matches "support last two majors" norm); when bumping to 1.25, add a `GOAMD64` build-matrix entry to CI and regenerate any goldens that depend on `a*b+c` rounding.
- **internal/numeric (new):** define reality-local `Float`, `Integer`, `Real`, `Complex` constraints to avoid ever importing `golang.org/x/exp/constraints` (preserves the zero-dep USP, slot 357).
- **linalg/, signal/, prob/, optim/transport/:** audit hot-path `a*b+c` patterns. For bit-deterministic ops, write `math.FMA(a,b,c)` explicitly; for "let the compiler choose" ops, document non-determinism in CONTEXT.md.
- **CI:** add an explicit `GOAMD64=v1` job alongside `v3` (or default) to guarantee goldens match on the lowest-common-denominator amd64 — matters because Go 1.25 silently fuses `a*b+c` at v3+.
- **cmd/realityvet (new optional package):** small analyzer pass for the four numerical-hygiene rules in entry §10. Aligns with reality's "precision documented" Design Rule 5.

## Block-E close-out (slots 351-380)

Block E delivered 30 internet-scoped surveys covering papers (351, 352, 361–366), libraries (353–360, 369, 373–376), standards/constants (367, 368), production cryptography (370, 371), and language-specific math idioms (357, 372, 377–380). Synthesis:

- **USP confirmed (slot 357):** reality's pure-Go + MIT + zero-dep + golden-file triple is genuinely unmatched in the surveyed Go ecosystem; gonum (BSD, depends on x/exp), v8go (cgo), ericlagergren/decimal (single-package) all violate at least one leg. Slot 380 reinforces: importing `x/exp/constraints` would forfeit it.
- **License hazards mapped (354/356/360):** JuMP (MPL2), Eigen (MPL2), CR-LIBM (LGPL) all incompatible with reality's MIT+zero-dep stance. Reimplementation-from-first-principles (Design Rule 6) is non-negotiable, not stylistic.
- **Concrete drift/bug findings (367, 376):** ε₀/μ₀ values lag a CODATA cycle in `constants/`; the PCG XSH-RR variant cited in some docs is obsolete vs. the DXSM variant Go's `math/rand/v2` ships (slot 380 closes that loop — reality's path forward is `rand/v2.PCG`, not a hand-rolled PCG-XSH-RR).
- **Cross-language float-validation latent design bug (360):** golden files at 1e-11 tolerance vs. Go 1.25's silent FMA fusion at GOAMD64=v3 will eventually mis-validate against C++/Python that don't fuse. Slot 380 surfaces the CI fix (per-arch matrix, or explicit `math.FMA`/`float64()` casts).
- **Production-stable PQC + ZK (370, 371):** ML-KEM/ML-DSA/SLH-DSA shipped in Go 1.24/1.26 stdlib; reality should integrate via `crypto/mlkem` rather than reimplement, since Design Rule 6 ("first principles") explicitly excludes cryptography (zero-dep applies to math, not to NIST-mandated crypto kits).
- **10+ algorithmic improvements logged:** Duan SSSP (362), randomized SVD (351), CR-LIBM-style correctly-rounded transcendentals (360), modern sparse formats (375), Pollard-Rho variants (372), HNSW/PQ for `prob/conformal` (361). Each is a concrete sprint-sized adoption.
- **Block-E meta:** internet-research surveys delivered ~30 unique strategic recommendations vs. ~150 implementation-shaped ones from earlier blocks. The ratio is correct: research is sparse-and-strategic by design. Outstanding follow-ups concentrate in: (i) constants drift, (ii) cross-language float-bit policy, (iii) `rand/v2` migration, (iv) `internal/numeric` constraint package, (v) per-arch CI matrix.

## Sources
- [Go 1.24 Release Notes](https://go.dev/doc/go1.24)
- [Go 1.25 Release Notes](https://go.dev/doc/go1.25)
- [Go 1.26 Release Notes](https://go.dev/doc/go1.26)
- [Secure Randomness in Go 1.22 (ChaCha8Rand)](https://go.dev/blog/chacha8rand)
- [Evolving the Go Standard Library with math/rand/v2](https://go.dev/blog/randv2)
- [math/rand/v2 package](https://pkg.go.dev/math/rand/v2)
- [golang.org/x/exp/constraints](https://pkg.go.dev/golang.org/x/exp/constraints)
- [golang.org/x/exp/rand (deprecated)](https://pkg.go.dev/golang.org/x/exp/rand)
- [math/big package](https://pkg.go.dev/math/big)
- [proposal: math/big add Float.Pow/Exp/Ln (#14102)](https://github.com/golang/go/issues/14102)
- [proposal: go/math add generic numeric utilities (#76193)](https://github.com/golang/go/issues/76193)
- [issue 71204: GOAMD64=v3 FMA auto-fusion](https://github.com/golang/go/issues/71204)
- [issue 71373: x/exp/rand → math/rand/v2 migration](https://github.com/golang/go/issues/71373)
- [issue 61716: math/rand/v2 revised API](https://github.com/golang/go/issues/61716)
- [Go Wiki: Minimum Requirements & GOAMD64](https://go.dev/wiki/MinimumRequirements)
- [nilness analyzer (x/tools)](https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/nilness)

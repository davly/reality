# 387 — meta-simd-strategy (SIMD policy across reality packages)

## Headline
Stay pure-Go for v0.x, accept the 4-10× ceiling vs OpenBLAS/Accelerate as the price of zero-deps + cross-compile + golden-file determinism, and pre-commit to a single migration target: Go 1.27+ stdlib `simd` package (currently `GOEXPERIMENT=simd` on amd64 in 1.26) — not Plan 9 assembly, not avo, not cgo.

## Where SIMD matters in reality

Inventory of hot kernels by package, sorted by realistic SIMD payoff. Speedups are typical published numbers for FP64 dense work on Skylake/Zen3+ class hardware; ARM NEON is roughly half on FP64 (128-bit lane only).

| Package | Hot kernel | Pure-Go ceiling | AVX2 (×4 FP64) | AVX-512 (×8 FP64) | Notes |
|---|---|---|---|---|---|
| `linalg` | `Dgemm` (matrix-matrix) | 1× | 4-6× | 6-10× | Largest absolute win; 5-loop blocking dwarfs the SIMD inner-kernel benefit alone |
| `linalg` | `Ddot` / `Daxpy` / `Dnrm2` (Level-1) | 1× | 2.5-3.5× | 3.5-5× | Memory-bound past L2; SIMD wins on hot data |
| `linalg` | `Dgemv` (matrix-vector) | 1× | 3-4× | 4-6× | The kernel `signal`/`prob`/`pca` actually hit |
| `signal` | radix-2 FFT inner butterfly | 1× | 2-3× | 3-4× | FFT is twiddle-table-bound; SIMD helps but cache helps more |
| `signal` | `Convolve` / FIR `Filter` | 1× | 3-4× | 5-6× | Tap-loop is canonical SIMD shape (FMA + reduction) |
| `signal` | `PowerSpectrum` (squared magnitude reduce) | 1× | 3× | 4× | Trivial `r*r+i*i` map-reduce |
| `prob` | log-likelihood sum-reduce (GMM, KDE) | 1× | 2-3× | 3-4× | `log` is expensive scalar; SVML-style vector log is 4× |
| `linalg` | `PCA` (covariance + eigendecomp) | 1× | dominated by `Dgemm` | — | Same as `Dgemm`; no new kernel |
| `geometry` | quaternion mul, SDF batch eval | 1× | 2-4× | 3-6× | Matters only if Pistachio batches at >10⁶/frame |
| `optim` | line-search inner products, gradient `Daxpy` | 1× | 2-3× | 3-4× | Falls out of `linalg/blas` work; not its own target |
| `crypto` | hash, AES-CTR, ChaCha8 | 1× | already in stdlib via SHA-NI / AES-NI / `rand/v2.ChaCha8` | — | `crypto/sha256`, `crypto/aes` already use Plan 9 asm; reality should not duplicate |
| `compression` | Huffman / RLE inner loops | 1× | 1.5-2× | 1.5-2× | Branchy; SIMD payoff small. Skip |
| `color`, `acoustics`, `em`, `fluids`, `orbital`, `queue`, `gametheory`, `combinatorics`, `chaos`, `control` | scalar formulas, ODE step | 1× | 1-1.5× | 1-1.5× | Not vectorizable in any meaningful way. Skip |

**Realistic top-5 SIMD targets** if reality ever leaves the pure-Go lane:
1. `linalg/blas.Dgemm` (only one with >5× ceiling and broad downstream pull-through)
2. `linalg/blas.Dgemv`
3. `signal.Convolve` / FIR filter
4. `linalg/blas.Ddot`+`Daxpy`+`Dnrm2` (one micro-kernel covers all three)
5. `signal.FFT` radix-2 butterfly

That's it. Beyond these five, SIMD is rounding-error against scalar Go.

## Strategy options

### 1. Stay pure Go (current; recommended for v0.x → v1.0)

**What you get:**
- Free FMA fusion at `GOAMD64=v3` (Go 1.25+, single-rounding `a*b+c`); arm64 always-on. This is what slot 380 flagged: reality already gets the only "free" SIMD instruction the compiler emits.
- `crypto/sha256`, `crypto/aes`, `crypto/rand/v2.ChaCha8` already use Plan 9 SIMD assembly inside stdlib — reality benefits transparently.
- `math.Sqrt`, `math.FMA`, `math.Hypot` — single-instruction inlines on amd64/arm64 (SSE2/NEON scalar; not vector but still 1 cycle).
- Bit-identical golden files across linux/darwin/windows × amd64/arm64. Critical: cross-language test vectors (Python, C++, C#) cannot validate against arch-specific reduction ordering. SIMD reductions reorder additions; golden files break.
- `go build` works on every supported `GOOS/GOARCH` (including `js/wasm`, `wasip1/wasm`, riscv64, ppc64le) with zero conditional compilation.

**What you pay:**
- 4-10× off OpenBLAS on `Dgemm` at >256×256.
- 2-4× off vendored FFT on >2K signals.
- Negligible elsewhere (>90% of reality's surface area).

For a "60 FPS small-matrix" downstream profile (slot 374's read of Pistachio's actual workload), the Dgemm gap collapses: 8×8 to 64×64 matmul is L1-bound, not FLOPS-bound, and pure-Go-with-FMA hits 50-70% of OpenBLAS at those sizes. The big gap appears only at sizes reality doesn't realistically use.

### 2. Add `_amd64.s` Plan 9 assembly fastpaths (NOT recommended now)

**Mechanics:**
- Use [`mmcloughlin/avo`](https://github.com/mmcloughlin/avo) as a build-time generator (Go DSL → Plan 9 .s). Avo *generates* sources; the result is committed and builds with stock `go build` — no avo runtime dep. Used by `minio/highwayhash`, `klauspost/reedsolomon`, `zeebo/blake3`, internal Go crypto.
- Runtime dispatch via `golang.org/x/sys/cpu.X86.HasAVX2` / `cpu.ARM64.HasASIMD`. Caveat: `x/sys/cpu` is a non-stdlib dependency — violates reality's zero-dep rule unless reality vendors a 50-line subset (CPUID + ID register read, public-domain-trivial).
- Need `_amd64.s` + `_arm64.s` + `_riscv64.s` + `_ppc64le.s` + `_amd64.go` (Go fallback) + `_other.go` (catch-all) per kernel. That's 6 files × 5 kernels = 30 files of triplicated code minimum.

**Real costs:**
- Plan 9 syntax has no FP register allocator beyond what avo provides; AVX-512 register pressure at the dgemm 4×8 microkernel is a genuine pain (16 ZMM registers required, 32 available, but spill bugs are silent miscompiles).
- AVX-512 frequency licensing on pre-Ice Lake Intel actually slowed mixed workloads — gonum/blas explicitly avoids 512-bit kernels for this reason.
- Golden-file vectors will diverge at ~1 ULP for any reduction (dot product, FFT, GEMM accumulator) because SIMD reorders the addition tree. You then need either (a) per-arch goldens (kills cross-language validation) or (b) deterministic SIMD reductions (kills the speedup).
- ARM SVE is variable-length — Plan 9 has no SVE support at all as of Go 1.26. NEON only. So you write x86 AVX2 + AVX-512, ARM64 NEON, and *nothing* for SVE; performance ceiling on Graviton4/Neoverse-V2 is left on the table indefinitely.
- WASM, riscv64, ppc64le all fall back to Go scalar — not a regression, but the asm work doesn't help them.

**Verdict:** the engineering load (avo build pipeline + 30+ source files + per-arch goldens + ULP-divergence-aware testutil) for at-most-five kernels is wildly disproportionate to the 4-10× speedup delivered to the one downstream consumer (Pistachio) that, by slot 374's reading, is small-matrix-bound and already inside the speedup-collapses-at-small-N regime. **Don't.**

### 3. Wait for Go stdlib SIMD intrinsics (recommended migration target)

**Status as of 2026-05:**
- [Issue golang/go#67520](https://github.com/golang/go/issues/67520): proposal accepted in principle Q4 2024.
- [Issue golang/go#73787](https://github.com/golang/go/issues/73787): `simd/archsimd` landed in Go 1.26 (Feb 2026) under `GOEXPERIMENT=simd`. amd64 only. 128 / 256 / 512-bit vector types, intrinsics that mirror VPADDQ/VFMADD231PD/etc.
- Go 1.27 (planned Aug 2026): expected to add ARM64 NEON intrinsics on `dev.simd` branch; SVE2 (variable-length) and a portable high-level API are explicitly punted post-1.27.
- [Issue golang/go#76175](https://github.com/golang/go/issues/76175): `go vet` CPU-feature check under GOEXPERIMENT=simd, prevents shipping AVX-512 intrinsics in code paths that run on AVX-only chips.

**What it gives reality:**
- No external dep, no avo, no Plan 9. Pure Go syntax, compiler-managed register allocation, integrates with FMA fusion already there.
- A real path to ARM64 NEON without separately maintaining `_arm64.s`.
- A real path to `_other` archs by leaving the scalar fallback unchanged.

**What it doesn't fix:**
- Reduction-order divergence still breaks bit-identical goldens; reality must commit to either (a) per-arch goldens behind a build tag, or (b) explicit Kahan-summed scalar fallback as the *canonical* result and SIMD as opt-in `// +build simd`. **(b) is the right choice** — golden files stay scalar-canonical, performance is opt-in.

**Decision rule:** revisit SIMD adoption when ALL of these are true:
- Go 1.27 stable (Aug 2026) ships `simd` package out of `GOEXPERIMENT`
- ARM64 NEON intrinsics in stdlib
- Reality has a documented benchmark (Pistachio frame-budget) showing pure-Go `Dgemm`/`Convolve` is the actual bottleneck (slot 357 USP discussion suggests it currently isn't)
- Reality's `linalg/blas` interface (slot 374) exists, so the SIMD impl can be a swappable backend not a fork of every kernel

Until then: pure Go.

### 4. CGO to OpenBLAS/MKL/Accelerate (forbidden by zero-dep)

Slot 374 already covers this: define the `linalg/blas` *interface* compatible with `gonum.org/v1/gonum/blas.Float64`, ship a pure-Go reference implementation, and document `linalg/blas.Use(gonumcgo.Implementation{})` as the user-side opt-in for HPC. Reality the *library* stays cgo-free; users compose. This is the answer for users who actually need 10× today — they bring their own BLAS.

## Cross-links

- **Slot 357 (research-libs-go):** reality's USP is "pure-Go, MIT, zero-dep, cross-language golden-file-validated." All three SIMD options 2/3/4 stress one of those three legs. Option 1 (stay pure-Go) preserves the USP exactly. Option 3 (stdlib `simd` once stable) preserves it cleanly. Options 2 and 4 forfeit either zero-dep (avo build pipeline, x/sys/cpu) or cross-language goldens (per-arch ULP divergence).
- **Slot 374 (research-blas-modern):** the pluggable `linalg/blas.Use(impl)` interface is the contract that lets users escape pure-Go's perf ceiling without making reality itself impure. Combined with this slot's recommendation: reality ships pure-Go BLAS reference; users opt into `gonum/blas/cgo` → OpenBLAS/MKL/Accelerate; eventually reality ships a `simd`-backed pure-Go BLAS that closes most of the gap without breaking any rule.
- **Slot 380 (research-go-math-extras):** Go 1.25 FMA fusion at `GOAMD64=v3` is the only "free" SIMD reality currently uses. Caveat from slot 380 still applies: bit-stable goldens require explicit `math.FMA(a,b,c)` or `float64(a*b)+c` to defeat fusion, *or* `GOAMD64=v3`-pinned regen. Reality's golden-file determinism requirement is the binding constraint, not the SIMD ceiling.

## Recommendation

**Codify the policy explicitly** (add to `CLAUDE.md` design rules, after rule 6):

> **Rule 7. SIMD policy: stay pure-Go through v1.0.** SIMD via Plan 9 assembly, avo-generated assembly, or cgo is out-of-scope for reality. Performance gap vs. OpenBLAS/Accelerate is accepted as the price of zero-deps and cross-language golden-file determinism. The migration target for SIMD is Go's stdlib `simd` package (currently `GOEXPERIMENT=simd` in Go 1.26 amd64; expected ARM64 NEON in 1.27). When that lands stable AND a downstream profile justifies it, reality may add SIMD-backed kernels for at most: `linalg/blas.{Dgemm, Dgemv, Ddot/Daxpy/Dnrm2}`, `signal.Convolve`, `signal.FFT` — gated behind a build tag, with the scalar implementation remaining canonical for golden files.

**Concrete actions for v0.10.x → v0.11.0:**
1. Audit `linalg/vector.go` (DotProduct, L2Norm, VectorAdd, VectorSub, VectorScale) — these are the pure-Go references that the future `linalg/blas` interface (slot 374) will implement. Confirm they use `math.FMA` or `float64(a*b)+c` for bit-stability, not bare `a*b+c`. Cross-link slot 380.
2. Add a `// SIMD candidate` doc-comment marker on the five top-5 hot kernels listed above. Future grep target when Go 1.27 ships.
3. Do **not** add `golang.org/x/sys/cpu` or `mmcloughlin/avo` to `go.mod`. Both violate zero-dep.
4. Document explicitly in package-level docs that performance-sensitive consumers should swap in a cgo BLAS backend via the `linalg/blas` interface (slot 374's recommendation).
5. When Go 1.27 stable ships (~Aug 2026), revisit. Until then: this slot is settled.

**The 2-8× perf cost of staying pure-Go is real but bounded:** it manifests on `Dgemm`/`FFT`/`Convolve` at large sizes, which are not reality's primary workload. For everything else (95% of the package surface), pure-Go is within 10-20% of hand-tuned SIMD because the kernels are scalar-shaped (ODE steps, color conversions, queueing formulas, orbital mechanics) and SIMD doesn't apply.

## Sources

- [Go proposal: simd package for intrinsics (golang/go#67520)](https://github.com/golang/go/issues/67520)
- [Go: simd/archsimd architecture-specific intrinsics under GOEXPERIMENT (golang/go#73787)](https://github.com/golang/go/issues/73787)
- [Go: simd CPU feature vet check under GOEXPERIMENT=simd (golang/go#76175)](https://github.com/golang/go/issues/76175)
- [Go proposal: package for SIMD instructions (golang/go#53171)](https://github.com/golang/go/issues/53171)
- [Go 1.26 release notes](https://go.dev/doc/go1.26)
- [golang.org/x/sys/cpu — feature detection package](https://pkg.go.dev/golang.org/x/sys/cpu)
- [mmcloughlin/avo — generate x86 assembly with Go](https://github.com/mmcloughlin/avo)
- [Go Wiki: AVX-512](https://go.dev/wiki/AVX512)
- [Go Assembly Optimization with Plan 9 (DEV)](https://dev.to/nithinbharathwaj/go-assembly-optimization-a-guide-to-high-performance-computing-with-plan-9-3594)
- [Optimising Go with SIMD assembly (Oscar Peace)](https://www.oscarcp.net/blog/optimising-goputer-with-assembly)
- [SIMD instructions in Go (cryptologie.net)](https://www.cryptologie.net/posts/simd-instructions-in-go/)
- [Go assembly complementary reference (Quasilyte)](https://www.quasilyte.dev/blog/post/go-asm-complementary-reference/)
- [Go-nuts: Go viewpoint on SIMD intrinsics](https://groups.google.com/g/golang-nuts/c/I2mTRxIwyQ4)
- [SIMD in Go: An In-Depth Exploration (Tfrain)](https://programmerscareer.com/golang-simd/)
- [Optimizing Go programs by AVX2 using Auto-Vectorization in LLVM (Medium)](https://medium.com/@c_bata_/optimizing-go-by-avx2-using-auto-vectorization-in-llvm-118f7b366969)
- [Accelerating Data Processing with AVX-512: Go vs Rust](https://techbytes.app/posts/avx-512-acceleration-go-rust-tutorial/)
- [Slot 357 — research-libs-go (USP, gonum/gorgonia/gosl positioning)](./357-research-libs-go.md)
- [Slot 374 — research-blas-modern (pluggable BLAS interface)](./374-research-blas-modern.md)
- [Slot 380 — research-go-math-extras (FMA at GOAMD64=v3)](./380-research-go-math-extras.md)

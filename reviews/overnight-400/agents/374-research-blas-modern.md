# 374 — research-blas-modern (BLAS implementations + reality positioning)

## Headline
Reality should NOT vendor or reimplement BLAS — define a small Go-pure `linalg/blas` interface (Level-1 + GEMM/GEMV/TRSM only), borrow gonum's interface shape, and keep dense kernels naive but correct; let downstream consumers swap in OpenBLAS/MKL via cgo if they need throughput.

## Survey

### Reference BLAS (Netlib, Fortran 77)
The spec, not the implementation. Three levels formalized 1979 (Lawson/Hanson/Kincaid/Krogh, Level-1 vector-vector), 1988 (Dongarra/DuCroz/Hammarling/Hanson, Level-2 matrix-vector), 1990 (Dongarra/Duff/DuCroz/Hammarling, Level-3 matrix-matrix). The Fortran 77 reference is intentionally unoptimized — it defines semantics. Public domain. Every other BLAS conforms to its calling conventions (column-major, 1-based indexing, character flags for transpose/uplo/diag/side). Reality should respect Level-1 and Level-2 names (DOT, AXPY, GEMV, TRSV) when defining its interface — this is how scientific Go users will recognize the shape. Reference: ACM TOMS papers; netlib.org/blas.

### OpenBLAS (BSD-3, OpenMathLib)
Goto/Wang fork of GotoBLAS2 1.13. Latest stable 0.3.30 (June 2025): added Intel Arrow Lake auto-detection, AmpereOne (Ampere-1A) support, optimized SBGEMM kernel for ARM Neoverse-V1, GCC 15 / MinGW fixes, GEMM workload-partitioning fixes. Hand-written assembly per-arch (AVX-512, NEON, SVE, RISC-V). License now BSD-3 (was modified BSD). The default scientific-Python wheel BLAS as of 2024+. Performance: typically 8-12× faster than gonum-pure-Go on dgemm at 4kx4k. Lesson for reality: OpenBLAS is the realistic CGo backend if a user needs HPC — design our interface so a `Use(openblasImpl)` swap is one line.

### BLIS (BSD-3, UT Austin / Field G. Van Zee)
The architectural reference design of the modern era. Refactors GotoBLAS as **five loops around a microkernel** — only the microkernel (typically 4×8 or 6×16 register-blocked GEMM inner loop) needs per-arch hand-tuning; packing/blocking loops are portable C. BLIS 1.0 shipped May 2024 — covers x86-64 (Intel/AMD), AArch64 (ARM32, Ampere Altra), Power, RISC-V (SiFive X280). Provides BLAS, BLAS-like (typed) API, and "object" API. Lesson for reality: **the five-loop blocking strategy is the single most portable HPC idea.** If reality ever writes a tuned dgemm, copy BLIS's structure verbatim — register block in microkernel, cache block in outer loops, NC×KC×MC parameters per L1/L2/L3.

### Intel MKL / oneMKL (proprietary, free-as-in-beer)
Closed binary. Gold standard on Intel x86. oneMKL 2025.1 (`INTEL_MKL_VERSION=20250100`): improved AMX `cblas_gemm_s8u8s32` and `cblas_gemm_bf16bf16f32` for INT8/BF16 large matmul; new SYCL `optimize_gemm` for sparse GEMM. SYCL/DPC++ path runs on Intel GPUs. Not vendorable (licensing). Reality cannot use it directly; users link it via cgo if they're on Intel. Lesson: AMX/AVX-512 BF16 paths are where 2024-2026 throughput lives — reality's eventual matmul should at least leave room for a BF16 dispatch.

### AMD AOCL (BSD-3-Clause)
AMD's BLIS fork + add-ons (libFLAME, ScaLAPACK, Sparse, RNG, FFTZ, libMem, libM). AOCL 5.2 (late 2025): Zen4/Zen5 GEMM tuning, AVX-512 kernels, "small/skinny matrix" specialized paths, smarter thread heuristics, FFTZ FFT, Build-It-Yourself customization utility. Because AOCL-BLAS is BSD-3 BLIS-derived, the kernel sources are legally portable. Lesson for reality: AMD's small-matrix tuning is the right benchmark for our use cases (a 60 FPS app rarely does 4k×4k matmul; it does many 8×8 to 64×64 ones). The "small/skinny" path is the one reality cares about.

### Apple Accelerate / vDSP (proprietary, macOS/iOS-bundled)
Free but Apple-only. Hits the M1/M2/M3/M4 AMX (Apple Matrix Extension) coprocessor — exclusive hardware path. Result: dense BLAS/LAPACK 6-14× faster than OpenBLAS for dgemm and 2-4× for factorizations on Apple Silicon. macOS 13.3+ added ILP64 BLAS. vDSP gives FFT/biquad/convolution that overlaps with reality's signal package. Reality cannot vendor it. Lesson: on macOS, a cgo bridge to Accelerate is the only way to access AMX. Keep our interface compatible.

### ATLAS (BSD-3, R. Clint Whaley 2001)
Auto-Tuned Linear Algebra Software. Pioneered empirical autotuning — generate many variants of dgemm, time them on the target machine, pick the fastest. Still maintained on SourceForge / GitHub `math-atlas/math-atlas`, but largely surpassed by OpenBLAS and BLIS for raw performance. Historically important: invented the autotuning paradigm. Lesson for reality: empirical autotuning is over-engineering for a 60 FPS app; static block sizes by ARCH are sufficient.

### cuBLAS / hipBLAS (proprietary / MIT-shim)
GPU. cuBLAS 13.2 = CUDA-runtime BLAS for NVIDIA H100/B200 (>90% peak FLOPs at moderate sizes). hipBLAS = AMD ROCm BLAS-marshalling shim, dispatches to rocBLAS or cuBLAS. MI300X benchmarks: 45-50% peak utilization (heuristic gaps vs cuBLAS). Reality is CPU-only and 60 FPS — GPU BLAS is out-of-scope for reality itself (belongs in a downstream `pistachio-gpu` adapter).

### Eigen (MPL2, headers-only C++)
Template-based dense+sparse linear algebra. Since Eigen 3.3, can dispatch to any F77 BLAS/LAPACK as backend (MKL, Accelerate, OpenBLAS, Netlib) for Dynamic-sized float/double/complex matrices. MPL2 = weak copyleft; usable in closed-source. Lesson for reality: Eigen's expression-template approach (lazy evaluation, no temporaries) is genuinely valuable; Go doesn't have templates so we can't replicate it directly, but we can offer fused ops (`AddScaled`, `MulAdd`, `GemvAdd`) that avoid intermediate allocations. Reality's no-allocation rule already pushes this direction.

### Boost.uBLAS (Boost License, C++ template)
Joerg Walter / Mathias Koch, now David Bellot / Stefan Seefeld. Provides Level 1/2/3 for dense/banded/symmetric/sparse via expression templates. Since 2018 also higher-order tensor outer/inner products. Slow vs. real BLAS — academic-grade; rarely chosen for HPC. Lesson: not a model for reality.

### Gonum BLAS (BSD-3, pure Go + cgo)
The actual incumbent in Go-land. `gonum.org/v1/gonum/blas` defines an interface; default backend `gonum/blas/gonum` is pure Go; alternative `gonum/blas/cgo` wraps Netlib/OpenBLAS. `gonum/mat` then accepts either via `Use()`. Pure-Go is roughly 10× slower than OpenBLAS at 4k×4k dgemm but works on every platform with no toolchain. The old `github.com/gonum/blas` repo is **DEPRECATED** in favor of `gonum.org/v1/gonum/blas`. Lesson for reality: this is the prior art that dictates what Go scientific users expect. Reality's BLAS interface should be a strict subset of gonum's `blas.Float64` interface so adapters are trivial.

## Reality positioning

### Recommendation: define a thin `linalg/blas` Go-pure interface, not a vendored BLAS.

**Do:**
1. Add `linalg/blas/blas.go` defining a minimal interface compatible with `gonum.org/v1/gonum/blas.Float64` for these routines only:
   - L1: `Ddot`, `Daxpy`, `Dnrm2`, `Dscal`, `Dasum`, `Idamax`
   - L2: `Dgemv`, `Dtrsv`, `Dger`, `Dsymv`
   - L3: `Dgemm`, `Dtrsm`, `Dsyrk`
   That's 12 functions — 95% of what `linalg/decompose.go`, `linalg/eigen.go`, `linalg/pca.go`, `signal/*` actually need.
2. Provide a default pure-Go reference implementation (naive triple-loop GEMM, no blocking) — correctness first. Golden-file tests at 1e-11 tolerance for transcendentals, exact for L1.
3. Document at the package level: "For HPC, swap in `gonum.org/v1/gonum/blas/cgo` via `linalg/blas.Use(...)`." Reality stays cgo-free; users who want OpenBLAS opt in.
4. Steal BLIS's five-loop blocking diagram in a comment block on `Dgemm` so a future contributor knows the right shape if they ever optimize.

**Do NOT:**
1. Vendor OpenBLAS/BLIS source — both BSD-3 and legally fine, but breaks zero-deps and pure-Go.
2. Write hand-tuned AVX-512/NEON kernels — out of scope; that's `pistachio-perf` or a cgo backend.
3. Reimplement gonum's full BLAS surface — gonum already did it; we just need an interface our packages can target.

**Why this shape:**
- Reality already has `linalg/decompose.go` and `linalg/eigen.go` (slot 308). They currently call internal vector/matrix ops directly. Routing them through a `blas` interface lets external consumers (aicore, pistachio) swap in tuned BLAS without forking reality.
- 60 FPS use case: typical matrices are 3×3 (rotations), 4×4 (projections), 8×8 (small SDF Jacobians), 64×64 (PCA on small descriptor banks). For these sizes, OpenBLAS overhead (function-call dispatch, packing) often **loses** to a naive Go loop. Reality's pure-Go default is the right default for its actual workload.
- The 10× pure-Go penalty appears at 4k+ — outside reality's scope.

### Cross-links to other reality slots
- **Slot 301 (signal/fft)**: FFT does not need BLAS but shares the "tight inner loop, no allocations" discipline. Same five-loop / cache-blocking lesson applies to large 1D FFT (Stockham vs. radix-2).
- **Slot 308 (linalg/decompose)**: LU/QR/Cholesky already implemented; refactoring them to call `linalg/blas.Dgemm`/`Dtrsm` would be a clean 1-day refactor and unlock external BLAS swap-in.
- **Slot for `linalg/pca.go`**: SVD/PCA is dominated by GEMM cost on large matrices — most likely beneficiary of an external BLAS swap.
- **Slot for `linalg/correlation.go`**: small-matrix; pure-Go is fine forever.
- Possible future slot **`linalg/blas`**: this review's recommendation. ~600 LoC, 12 routines, ~250 golden vectors.

### Microkernel architecture takeaways (BLIS lesson, for the eventual day reality writes a tuned `Dgemm`)
1. **Five loops:** outermost over `JC` blocks of `N` (NC ≈ 4096), then `PC` blocks of `K` (KC ≈ 256), pack `B` panel; then `IC` blocks of `M` (MC ≈ 96), pack `A` panel; then `JR` (NR ≈ 8), then `IR` (MR ≈ 4) — innermost is the microkernel doing `MR×NR` rank-1 updates over `KC`.
2. **Microkernel:** the only arch-specific code. Hand-written FMA loop using all available vector registers. ~50-150 lines per arch.
3. **Block sizes** chosen so the packed `A` block fits in L2, packed `B` panel fits in L3, microkernel `C` block stays in registers.
4. **Packing** matters: gather strided slices into contiguous panels once, reuse many times.
5. Reality's pure-Go `Dgemm` should at least pick MR=4, NR=8, and arrange the inner loop so Go's SSA backend can recognize the FMA pattern. Don't bother with explicit packing until profiled.

## Sources
- [BLIS: Extending BLAS Functionality (SIAM)](https://www.siam.org/publications/siam-news/articles/blis-extending-blas-functionality/)
- [BLIS: A Framework for Rapidly Instantiating BLAS Functionality (Van Zee, ACM TOMS)](https://www.cs.utexas.edu/~flame/pubs/BLISTOMSrev2.pdf)
- [BLIS KernelsHowTo (microkernel docs)](https://github.com/flame/blis/blob/master/docs/KernelsHowTo.md)
- [OpenBLAS 0.3.30 Phoronix release notes (June 2025)](https://www.phoronix.com/news/OpenBLAS-0.3.30)
- [OpenBLAS GitHub (OpenMathLib/OpenBLAS)](https://github.com/OpenMathLib/OpenBLAS)
- [AMD AOCL-BLAS GEMM small matrix tech article (2025)](https://www.amd.com/en/developer/resources/technical-articles/2025/aocl-blas-boosting-gemm-performance-for-small-matrices-.html)
- [AOCL Dense Linear Algebra page](https://www.amd.com/en/developer/aocl/dense.html)
- [Apple Accelerate framework overview](https://developer.apple.com/accelerate/)
- [Apple Accelerate documentation](https://developer.apple.com/documentation/accelerate)
- [Intel oneMKL 2025 release notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/onemkl-release-notes.html)
- [cuBLAS 13.2 documentation](https://docs.nvidia.com/cuda/cublas/)
- [hipBLAS 3.2.0 documentation (ROCm)](https://rocm.docs.amd.com/projects/hipBLAS/en/latest/)
- [Eigen: Using BLAS/LAPACK from Eigen](https://libeigen.gitlab.io/eigen/docs-nightly/TopicUsingBlasLapack.html)
- [Boost uBLAS docs](https://www.boost.org/doc/libs/latest/libs/numeric/ublas/doc/index.html)
- [ATLAS / math-atlas GitHub](https://github.com/math-atlas/math-atlas)
- [Netlib BLAS reference](https://www.netlib.org/blas/)
- [LAPACK Working Note 81 — Levels 1, 2, 3 BLAS](https://www.netlib.org/lapack/lawn81-3.0/node7.html)
- [gonum mat package docs (interface design)](https://pkg.go.dev/gonum.org/v1/gonum/mat)
- [gonum BLAS pure-Go vs OpenBLAS benchmark gist](https://gist.github.com/bluescreen10/b2cc7ed4a8054e64fc994979edd0f2b1)
- [Gonum BLAS DEPRECATED repo notice](https://github.com/gonum/blas)
- [Outperforming cuBLAS on H100 worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)
- [GEMMFIP: Unifying GEMM in BLIS (arXiv 2302.08417)](https://arxiv.org/pdf/2302.08417)

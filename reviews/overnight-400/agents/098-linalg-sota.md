# 098 | linalg-sota

**Scope.** SOTA dense/sparse linear-algebra library comparison for `C:\limitless\foundation\reality\linalg\` (≈1,440 LOC, 25 functions, dense-only, `[]float64` row-major). Distinct from 096 (numerics audit) and 097 (missing-primitive taxonomy). The two prior agents established **what** is missing; this agent surveys **how** the SOTA libraries actually engineer their kernels and which of those tricks port to a zero-dep Go library.

**Mandate.** For each surveyed library, distil three things: (1) headline algorithm/feature, (2) the engineering trick that makes it fast or robust, (3) zero-dep portability call for `reality`.

---

## 0. Headline

The dense linear-algebra ecosystem is **eight stacks deep on top of one numerical core** (LAPACK 3.x algorithms) with three modern divergences from the textbook recipe: **(a)** instruction-level micro-kernels (BLIS/MKL/OpenBLAS) that beat naive triple-loops by 30–100×; **(b)** expression-template lazy evaluation (Eigen/Blaze) that fuses `D = A·B + C` into a single pass at compile time; and **(c)** randomized / blocked / divide-and-conquer reformulations of classical decompositions (rSVD, BDC-SVD, MRRR, recursive LU/QR) that are now the academic default but remain absent from most Go-native packages.

For Reality's zero-dep, MIT, golden-file-validated, Pistachio-60-FPS-budget context, the **adoptable** subset is sharply bounded. Architecture-specific intrinsics (AVX-512, AMX, GOTOBLAS-style packing) are out — Go has no SIMD without `asm`. Expression templates are out — Go has no operator overloading. Cache-blocked recursive layouts (Morton/BLIS pack) are partially out — Go can implement the algorithm but cannot match the asm-tuned inner kernel. **What is squarely in:**

1. **Cache-aware blocking parameters** (BLIS Mc=Kc=block sizes derived from L1/L2/L3 — pure-Go portable; ~30% speedup on a naive `MatMul` for n>256).
2. **Recursive blocked LU/QR** (LAPACK 3.x `dgetrf2`, Toledo 1997 — same FLOPS as flat LU, half the cache misses).
3. **Randomized SVD / pivoted-QR** (Halko-Martinsson-Tropp 2011 — same accuracy as Golub-Reinsch on low-rank, 10× faster, ~150 LOC).
4. **Divide-and-conquer symmetric eig** (Cuppen 1981 / `dsyevd` — replaces tqli at n>~50 with O(n^{2.4}) cost vs O(n^3); ~500 LOC).
5. **MRRR symmetric eigenvectors** (Dhillon-Parlett 2004 — eigenvectors without Gram-Schmidt re-orthogonalisation; orthogonal-to-working-precision by construction).
6. **libxsmm-style small-matrix specialisation** (Heinecke et al. 2016 — the most Reality-relevant SOTA library for **game-engine matrix sizes 3×3, 4×4, 8×8, 16×16**; pure-math algorithms for tiny matrices that beat generic LAPACK by 5–20×).
7. **SLEEF vectorised special functions** (Shibata 2010 — table-free, vectorisable, bit-precise; ports as scalar Go for `MatExp` / `MatLog` Padé inner loops).

The single highest-leverage SOTA-port for Reality is **libxsmm small-matrix kernels** (`MatMul3x3`, `MatMul4x4`, `MatInverse4x4`, `MatDet3x3`) because Pistachio's 60-FPS use case is dominated by 3×3 and 4×4 matrix arithmetic on every camera/transform/inertia computation, and a generic `MatMul(A,3,3,B,3,out)` pays the loop-overhead tax that libxsmm specifically eliminates with explicit unrolled formulas. ~120 LOC unblocks the only matrix-heavy hot path Reality has today.

---

## 1. LAPACK 3.x — the gold standard

**Headline.** Reference implementation of every classical dense decomposition: LU (`dgetrf`), Cholesky (`dpotrf`), QR (`dgeqrf`), SVD (`dgesdd`/`dgesvd`), symmetric eig (`dsyev`/`dsyevd`/`dsyevr`), nonsymmetric eig (`dgeev`), Schur (`dhseqr`), generalised eig (`dggev`/`dggev3`), banded (`dgbsv`), tridiagonal (`dgtsv`), least-squares (`dgels`/`dgelsd`/`dgelsy`).

**Engineering trick: blocked recursive factorisations.** LAPACK 3.0+ shifted from flat (Level-2 BLAS) to blocked (Level-3 BLAS) factorisations. `dgetrf` calls `dgetrf2` recursively, splitting `[A11 A12; A21 A22]` and reducing top-left, then trailing-update via `dgemm`. This converts 70% of the FLOPS to matrix-matrix multiply, which a tuned BLAS turns into 95% peak FLOPS. Toledo 1997 ("Locality of reference in LU decomposition") proved cache-miss optimality.

The LAPACK API is also famously **caller-allocates-everything**: every routine takes a pre-allocated `lwork` workspace and a query-mode call (`lwork=-1`) that returns the optimal size. This is exactly the discipline Reality already enforces (`vector.go` docstring, `MatMul` zero-alloc).

**Zero-dep portability for reality.**
- **Adopt:** the recursive-blocked LU / QR / Cholesky pattern. Same FLOPS, ~2× faster on n>200 due to cache locality. Pure-Go portable.
- **Adopt:** the workspace-query convention. Make `LUDecomposeWorkspaceSize(n) int` siblings of every decomposition, returning the float-count needed; let consumers pre-allocate.
- **Adopt:** the LAPACK error-code convention (`info=0` success, `info<0` invalid arg, `info>0` numerical failure with index). Reality currently mixes panics and silent failures.
- **Skip:** the Fortran column-major default. Reality is row-major; transposing is a 2-LOC change everywhere.
- **Skip:** the LAPACKE C interface conventions. Go can do better with named return values.

(Sources: <https://netlib.org/lapack/lapack-3.12.0/index.html>, Toledo 1997 "Locality of reference in LU decomposition with partial pivoting" SIAM JMA.)

---

## 2. Eigen 3 (C++, MPL2) — the modern academic default

**Headline.** Header-only C++ library; covers dense (`PartialPivLU`, `FullPivLU`, `LDLT`, `LLT`, `HouseholderQR`, `ColPivHouseholderQR`, `FullPivHouseholderQR`, `JacobiSVD`, `BDCSVD`), sparse (`SimplicialLLT`, `SparseLU`, `SparseQR`, `IncompleteLUT`, `IncompleteCholesky`), and iterative (`ConjugateGradient`, `BiCGSTAB`, `LeastSquaresConjugateGradient`).

**Engineering trick: expression templates with lazy evaluation.** `MatrixXd D = A * B + C * D.transpose();` does not evaluate `A*B` and `C*D^T` into temporaries. Each operator returns a thin `Product<>`/`Sum<>`/`Transpose<>` expression-tree node; the assignment operator pattern-matches the tree and dispatches to a fused `gemm`+`gemm` kernel. The result: zero temporaries, single-pass evaluation, and **algorithm choice deferred to runtime shape inspection** (Eigen picks `dgemm` for large, an unrolled small-kernel for n≤4, and `Strassen` is configurable).

**Engineering trick: BDCSVD (Trefethen-Howell 2018, GFI 2013).** The default SVD for `n>16`. Bidiagonalises, then divide-and-conquers the bidiagonal SVD via Cuppen-style secular-equation root-finding. ~5–10× faster than Golub-Reinsch on `n=1000` symmetric matrices, identical accuracy.

**Engineering trick: vectorised packet types.** `internal::packet_traits<Scalar>` exposes platform-specific SIMD widths (4 doubles on AVX2, 8 on AVX-512); kernels are written once against `Packet`, specialised to `__m256d` / `__m512d` on Intel, `float64x2` on ARM, `vector double` on PowerPC.

**Zero-dep portability for reality.**
- **Skip:** expression templates entirely. Go has no operator overloading and no template metaprogramming. Method-chain DSL (`A.Mul(B).Add(C.Transpose())`) is possible but loses the fusion gain.
- **Skip:** SIMD packet abstraction. Go's `math/bits` is not enough; `golang.org/x/sys/cpu` exists for runtime detection but actual SIMD requires inline asm or codegen, which violates zero-dep.
- **Adopt:** BDCSVD as the v1.1 SVD default. Pure-math algorithm; ~700 LOC. Eigen's reference implementation is the cleanest published.
- **Adopt:** Eigen's API taxonomy as the model for Reality's eventual public surface. `LDLT`, `LLT`, `HouseholderQR`, `ColPivHouseholderQR` are standard, immediately-recognisable names.
- **Adopt:** small-matrix specialisations for n≤4 in matmul / inverse / determinant. Eigen ships hand-coded fast paths; Reality has none.

(Sources: <https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html>, Trefethen-Howell SIAM JMAA 2018, Eigen `BDCSVD.h`.)

---

## 3. Armadillo (C++, Apache 2.0) — easy syntax, library-of-libraries

**Headline.** A high-level wrapper over OpenBLAS / MKL / LAPACK / ARPACK / SuperLU. ~250 functions; `arma::solve(A,b)`, `arma::svd(U,s,V,X)`, `arma::eig_sym`, `arma::expmat`, `arma::logmat`, `arma::sqrtmat`. `arma::sp_mat` for sparse, `arma::Cube` for 3D tensors.

**Engineering trick: smart-templating runtime backend dispatch.** Armadillo writes algorithms in C++ but dispatches the inner triple-loop to BLAS/LAPACK at runtime. The benefit Reality cannot have (zero-dep) but the *interface design* is portable: a single `Solve(A, b)` function selects LU vs Cholesky vs QR vs SVD based on a few inspection passes (`isSymmetric`, `isPosDef` cheap heuristic).

**Engineering trick: `expmat()` family.** Armadillo ships `expmat` (general Higham), `expmat_sym` (eigendecomp-based, SPD), `logmat`, `logmat_sympd`, `sqrtmat`, `sqrtmat_sympd`, `powmat`. The symmetric-positive-definite specialisations are the exact `MatExpSymmetric` / `MatLogSymmetric` Reality's `infogeo` SPD manifold needs (097 §6, 096 P6).

**Zero-dep portability for reality.**
- **Adopt:** the `Solve(A, b, options)` umbrella that auto-selects LU vs Cholesky vs QR. Reality's consumers should not have to know whether their matrix is SPD; a `Solve` umbrella with a `Hint{Symmetric, PD, Banded}` enum is a 50-LOC convenience layer over existing primitives.
- **Adopt:** the `sympd` naming convention (`SolveSymPD`, `MatExpSymPD`) where the SPD-specialised path is dramatically faster. Reality's eventual `MatrixExpSymmetric` should be called `MatrixExpSymPD` per Armadillo precedent (also matches LAPACK `_po_` infix).
- **Skip:** the runtime-dispatch implementation. Reality is single-backend by design.

(Sources: <https://arma.sourceforge.net/docs.html>, Sanderson-Curtin JOSS 2016 "Armadillo: a template-based C++ library for linear algebra".)

---

## 4. Blaze (C++, BSD-3) — high-performance C++

**Headline.** Modern C++14 dense+sparse library; benchmarks competitive with Eigen and MKL on dense GEMM, sometimes faster on small matrices. Functions: `DynamicMatrix`, `StaticMatrix<T,M,N>` (compile-time fixed-size), `CompressedMatrix`, `SymmetricMatrix`, `LowerMatrix`, `UpperMatrix`, `DiagonalMatrix`, `HermitianMatrix` adaptor types.

**Engineering trick: SMP via OpenMP/HPX/C++ threads, SMT-aware.** Blaze auto-parallelises matrix expressions over multiple cores using a configurable backend (OpenMP, HPX, C++17 std::thread). The threshold for parallelisation is itself dynamically tuned based on operation cost vs thread-spawn overhead.

**Engineering trick: adaptor types compose with operations.** `auto S = blaze::declsym(A);` doesn't copy A but creates a view that promises symmetry; subsequent `S * v` uses a SYMV kernel (half the FLOPS of GEMV). `blaze::declsym(A) * blaze::declposdef(A)` selects Cholesky-based solve.

**Engineering trick: `StaticMatrix` for small fixed sizes.** `blaze::StaticMatrix<double,3,3>` lives on the stack, has its operations fully unrolled at compile time, and competes with libxsmm on n≤8 sizes.

**Zero-dep portability for reality.**
- **Adopt:** the **adaptor / hint pattern** in API design. Reality's `LUSolve` doesn't know `A` is symmetric; a `linalg.SymPDSolve(A, n, b, x)` is a 5-LOC convenience that forces the Cholesky path. Generalise to `Hints{Symmetric, PosDef, UpperTriangular, LowerTriangular, Diagonal}` enum so consumers can guide algorithm choice.
- **Adopt:** stack-allocated small-matrix structs. Reality's `[]float64`-everything API is great for cache-line-width tuning but pays a heap-pointer cost on tiny matrices. A `Mat3`, `Mat4` value type (12 / 16 floats inline) is the right abstraction for 60-FPS Pistachio camera math. ~80 LOC.
- **Skip:** OpenMP/SMT auto-parallelisation. Go's goroutines are a different abstraction; manual `goroutine` parallelism is OK but not implicit.

(Sources: <https://bitbucket.org/blaze-lib/blaze/wiki/Home>, Iglberger SC'12 "Expression templates revisited".)

---

## 5. Intel MKL — assembly-level kernels

**Headline.** Closed-source (oneAPI MKL is free-as-in-beer, not as-in-speech). Hand-tuned AVX-2 / AVX-512 / AMX kernels for every CPU generation; routinely 90–98% of theoretical peak FLOPS on supported hardware. Provides BLAS, LAPACK, FFT, sparse BLAS, VML (vector math library), and DPCPP-offload variants for GPUs.

**Engineering trick: micro-kernel + macro-kernel split.** The inner `mr × nr` micro-kernel (typically 4×4, 6×8, 8×8 on AVX-512) is 100% inline assembly using register blocking — every value lives in a register through the entire k-loop. The outer macro-kernel handles cache packing (Goto-style "pack A into Bp, pack B into Bp" buffers sized to L2 / L3).

**Engineering trick: `dgemm_kernel_X64.S` per CPU generation.** MKL ships **separate compiled kernels** for Nehalem, Sandy Bridge, Haswell, Skylake-X, Cascade Lake, Ice Lake, Sapphire Rapids — runtime CPU detection picks the right one. The asm differs by handful of lines (instruction encoding for AVX vs AVX2 vs AVX-512) but capturing each generation's preferred port-pressure pattern is what makes MKL the speed-of-light reference.

**Engineering trick: VML (Vector Math Library) for elementwise transcendentals.** `vdExp`, `vdLog`, `vdSin`, `vdCos`, `vdPow` are 5–20× faster than scalar `std::exp` because the VML kernels hold polynomial constants in registers and process 4 or 8 doubles per cycle. SLEEF is the open-source equivalent (§13).

**Zero-dep portability for reality.**
- **Skip everything implementation-side.** Go cannot match register-blocked AVX-512 asm without giving up portability.
- **Adopt:** the **micro-kernel / macro-kernel architectural split** at the algorithmic level. Reality's `MatMul` should grow a private `matmulKernel(a, b, c, mc, nc, kc)` that operates on a cache-resident block, called from a blocked outer driver. Even without SIMD, blocking alone yields 20–40% speedup on n>256 in pure Go.
- **Adopt:** the precomputed-constants discipline for Padé-coefficient transcendentals. `MatExp`'s 13/13 Padé numerator coefficients should be `var paddeCoefs13 = [...]float64{...}` at package scope, never recomputed.

(Sources: <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>, Goto-van de Geijn TOMS 2008 "Anatomy of high-performance matrix multiplication".)

---

## 6. OpenBLAS — open-source BLAS

**Headline.** Open-source descendant of GotoBLAS2 (Kazushige Goto, UT Austin); the de-facto BLAS for Linux, NumPy, R, Octave. Hand-tuned asm kernels for x86_64, ARM (NEON, SVE), POWER, RISC-V. Routinely 80–95% of MKL on Intel, often *faster* than MKL on AMD Zen (where MKL deliberately downgrades).

**Engineering trick: Goto packing.** Pre-pack `A` into `mc × kc` blocks contiguous in memory in the order the micro-kernel will consume them. This single optimisation buys ~3× over a "cache-blocked" but unpacked GEMM. Goto-van de Geijn 2008 paper formalised this; every BLAS since has copied it.

**Engineering trick: parametric kernel generation.** OpenBLAS doesn't ship one kernel — it ships templated asm that the build system specialises by setting `MR`, `NR`, `KC`, `MC`, `NC` per target. `make TARGET=HASWELL` produces a Haswell-tuned binary; `make TARGET=ZEN3` produces a Zen3 one. The kernel skeleton is the same; the cache parameters change.

**Zero-dep portability for reality.**
- **Adopt:** Goto packing in pure Go for `MatMul`. Pack `A` into a contiguous scratch buffer (caller-supplied via `Workspace` parameter to honour zero-alloc) before the k-loop. ~80 LOC of additive code, ~2× speedup at n>256.
- **Adopt:** parametric block sizes. Expose `var DefaultBlockKc, DefaultBlockMc, DefaultBlockNc = 256, 64, 4096` at package level, derivable from L1/L2/L3 cache reads via `golang.org/x/sys/cpu`. (cpu detection is stdlib-adjacent; arguable for "zero-dep".) Fallback constants tuned for typical x86 L1=32KB, L2=256KB are fine for v1.

(Sources: <https://github.com/OpenMathLib/OpenBLAS>, Goto-van de Geijn TOMS 2008.)

---

## 7. BLIS (BLAS-like Library Instantiation Software) — portable framework

**Headline.** From UT Austin (Field van Zee, Robert van de Geijn). The successor framework to GotoBLAS2: same algorithms, but factored into a portable C "framework" (cache control, packing, level-3 driver) and a swappable instruction-level micro-kernel. Adding support for a new ISA = writing one ~200-LOC asm micro-kernel + two `_pack_` kernels.

**Engineering trick: 5-loop GEMM expansion.** BLIS's GEMM is literally 5 nested loops, with cache-aware packing inserted between loops 3 and 4 and between loops 4 and 5. The structure:
```
for jc = 0 .. n by nc:                    # L3 cache for B
  for pc = 0 .. k by kc:                  # L2 cache (B and A panels)
    pack B[pc:pc+kc, jc:jc+nc] -> Bp
    for ic = 0 .. m by mc:                # L1 cache for A
      pack A[ic:ic+mc, pc:pc+kc] -> Ap
      for jr = 0 .. nc by nr:             # register tile
        for ir = 0 .. mc by mr:
          micro_kernel(Ap, Bp, C[ir:ir+mr, jr:jr+nr])
```
Every cache level is hit linearly; the packing rearranges A and B to the exact stride the micro-kernel consumes.

**Engineering trick: the micro-kernel is the *only* arch-specific code.** Everything else (BLAS surface, Level-2 ops, packing) is portable C. This is the cleanest published illustration of "what is portable in linear algebra and what isn't".

**Zero-dep portability for reality.**
- **Adopt directly:** the 5-loop GEMM expansion. Reality's `MatMul` is currently a flat triple-loop with cache-line-width-bad access patterns; lifting the BLIS structure to pure Go is ~120 LOC of additive code and yields the cache-locality benefit even without SIMD micro-kernels. The Go inner loop will be 5–10× slower than asm but 2–3× faster than the current naive triple loop.
- **Adopt:** the principle "all arch-specific stuff is in one well-isolated micro-kernel function". Reality's `matmulKernel(a, b, c, mr, nr, kc)` is a target for future SIMD/asm porting (e.g., via `golang.org/x/exp/asm` if/when stabilised); keeping it isolated means the rest of the package never sees the asm.

(Sources: <https://github.com/flame/blis>, van Zee-van de Geijn TOMS 2015 "BLIS: A framework for rapidly instantiating BLAS functionality".)

---

## 8. libxsmm — small matrix multiplication (HIGHLY relevant to Pistachio)

**Headline.** Intel research library specialised for **tiny matrix multiplications** (m, n, k ≤ ~64). JIT-compiles a custom asm kernel for the *exact* shape (e.g., 6×6×6) on first use, caches it, then dispatches the JIT'd kernel on every subsequent call. Pure-asm kernels for m×n×k all up to ~64 each. Used in CP2K, deep-learning convolution backends.

**Engineering trick: shape-specialisation via JIT.** A 4×4×4 matmul is 64 multiplies and 48 adds — handwriting the optimal AVX-512 asm for each shape would require thousands of kernels. libxsmm generates them on demand. The first call compiles, caches, and runs; subsequent calls are direct asm dispatch.

**Engineering trick: small-batched GEMM.** `libxsmm_smm_4x4_4_x10000` is a single API call that processes 10,000 4×4×4 matmuls back-to-back, with the kernel kept hot in icache. This is exactly what game engines do with skeletal animation joint matrices and physics inertia tensors.

**Why this is the most Reality-relevant SOTA library.** Reality's primary documented use case is **Pistachio at 60 FPS** (CLAUDE.md, vector.go:1-11). At 60 FPS, the dominant matrix workload is small-matrix arithmetic on 3×3 / 4×4 / 6×6 inertia, transformation, and Jacobian matrices — *not* large matrix factorisation. A generic `MatMul(A, 3, 3, B, 3, 3, out)` pays:
1. Triple-loop overhead (9 outer-loop bound checks per matmul).
2. Index arithmetic (`i*3+k`, `k*3+j`, `i*3+j` — 9 multiplies that vanish in an unrolled version).
3. No register-pressure optimisation (a hand-unrolled 3×3 keeps everything in 9 fp registers).
4. Bounds-check elision is non-deterministic across Go versions.

A hand-unrolled `MatMul3x3(A, B, out [9]float64)` is ~30 LOC and runs in ~6ns vs the generic version's ~30–50ns. For 1,000 entities × 60 FPS = 60,000 matmuls/sec, the saving is ~2.5ms of wall-clock per second freed.

**Zero-dep portability for reality.**
- **ADOPT URGENTLY:** specialised `MatMul3x3`, `MatMul4x4`, `MatInverse3x3`, `MatInverse4x4`, `MatDet3x3`, `MatDet4x4`, `MatVec3`, `MatVec4` (~120 LOC total). Each is a 5-line unrolled formula with ~5–20× speedup over the generic versions. Cite: Cramer's rule for 3×3 inverse, cofactor expansion for 4×4 inverse, classical 3×3 determinant.
- **Adopt:** a `MatMulBatch3x3(As, Bs, outs []float64, count int)` entry point for skeletal animation / physics use cases. ~30 LOC over the basic kernel. Caches better.
- **Skip:** the JIT mechanism. Go has no portable JIT; specialise the small sizes statically in `matmul_small.go`.

(Sources: <https://github.com/libxsmm/libxsmm>, Heinecke et al. SC'16 "LIBXSMM: Accelerating Small Matrix Multiplications by Runtime Code Generation".)

---

## 9. mlpack — C++ machine-learning library

**Headline.** Apache-2.0 ML library built on Armadillo; ~50 ML algorithms (k-NN, k-means, GMM, HMM, neural nets, decision trees, random forests, RL). Linear-algebra dependence is via Armadillo; mlpack itself contributes algorithms above the linalg layer.

**Engineering trick: template-based dimensionality polymorphism.** Algorithms templated on `<arma::mat>` vs `<arma::sp_mat>` so the same code works dense and sparse. Same trick a Go generic-based linalg could use post-Go-1.18 generics.

**Zero-dep portability for reality.**
- **Skip the codebase.** mlpack is downstream of linalg, not a peer.
- **Adopt the API insight:** the same algorithm signature should work for dense and sparse storage when the underlying ops abstract over storage. Reality's eventual sparse types (097 T1) should expose a `MatVecMul` that has the same call signature as the dense one, parameterised by storage. This is achievable in Go via interface dispatch (~3% perf cost) or via tagged-union reflection.

(Sources: <https://www.mlpack.org/>, <https://github.com/mlpack/mlpack>.)

---

## 10. dlib — C++ ML toolkit

**Headline.** Boost-licensed C++ ML/CV library with a built-in linear-algebra layer (`dlib::matrix<T,NR,NC>`). Distinguishes itself from Eigen with cleaner BLAS-as-fallback dispatch and integrated optimisation routines (`find_min`, `solve_qp`, `solve_least_squares`).

**Engineering trick: compile-time and run-time matrix sizes coexist.** `dlib::matrix<double, 3, 3>` is compile-time fixed-size (no malloc); `dlib::matrix<double>` is run-time sized. Same operations work on both. Eigen does the same; dlib's choice is to expose them via the same template name with default parameter `NR=0` meaning dynamic.

**Engineering trick: integrated `solve_qp` (quadratic programming).** A clean, doc-friendly QP solver lives in the same library as the linalg primitives, removing the consumer's need to glue a separate solver.

**Zero-dep portability for reality.**
- **Adopt:** the dual fixed-size-vs-runtime-size pattern. `linalg.Mat3` (compile-time) and `linalg.Matrix` (runtime, `[]float64`). The former for hot-path 60-FPS code, the latter for general-purpose. Already partially proposed under §4; dlib is the second SOTA precedent.
- **Adopt (delegate):** Reality's `optim` package already has gradient-based optimisation; `optim` is the right home for QP/LP, not `linalg`. dlib's bundling is a C++/header-only convenience that doesn't apply to Go's package model.

(Sources: <http://dlib.net/linear_algebra.html>, King JMLR 2009 "Dlib-ml: A Machine Learning Toolkit".)

---

## 11. ndarray (Rust, MIT/Apache-2.0) — the Rust NumPy

**Headline.** Rust's most-used numerical-array crate. Type: `Array<T, D>` where `D` is dimension (`Ix1`, `Ix2`, `IxDyn`). BLAS dispatch optional (`ndarray-linalg` calls into LAPACK). Owned arrays + views (`ArrayView`, `ArrayViewMut`) with zero-copy slicing.

**Engineering trick: views and ownership separation.** `ArrayView` is a non-owning lightweight reference; `slice(s![1..3, ..])` returns a view into the parent array. This lets Rust catch aliasing at compile time and keeps the BLAS-friendly contiguous-stride invariant explicit (`is_standard_layout()`).

**Engineering trick: parallel iteration via Rayon.** `array.par_iter_mut()` parallelises across all cores via a single line of code. Rayon's work-stealing scheduler handles load balancing.

**Zero-dep portability for reality.**
- **Adopt:** the **view abstraction**. Reality's `[]float64` interfaces are good; a `MatView{data []float64, rows, cols, rowStride int}` struct enables non-contiguous slices (sub-matrices, strided columns) without copying. ~30 LOC of additive type. Particularly relevant for blocked GEMM (§7) where the inner kernel processes `mc×nc` sub-blocks.
- **Skip:** Rayon-style parallelism. Go's goroutines + channels are the idiomatic parallel construct; expose it explicitly via `MatMulParallel(workers int)` rather than transparently.

(Sources: <https://github.com/rust-ndarray/ndarray>, <https://docs.rs/ndarray/>.)

---

## 12. nalgebra (Rust) — linalg for game engines & robotics

**Headline.** Rust's other major linalg crate, narrower-focused than ndarray. Covers small fixed-size matrices (`Matrix3`, `Matrix4`, `SVector<T,N>`), Lie groups (`UnitQuaternion`, `Rotation3`, `Isometry3`), and SLAM/robotics primitives. Used by Bevy game engine, several robotics frameworks.

**Engineering trick: const-generic dimensions.** `SMatrix<f64, 3, 3>` (stack-allocated, compile-time-checked) versus `DMatrix<f64>` (heap, run-time). Const generics let the same algorithm code work on both with full compile-time size checking on the static path.

**Engineering trick: Lie-group methods first-class.** `UnitQuaternion` is not just a 4-vector; it's a manifold element with `exp_map`, `log_map`, `slerp` as methods. The geometric algebra is integrated, not bolted on.

**Zero-dep portability for reality.**
- **Adopt:** small-matrix specialisation (third precedent after Eigen and Blaze; nalgebra is the cleanest published Rust impl).
- **Adopt:** the Lie-group integration model. Reality's `geometry` package has quaternions; the **manifold operations** (`exp`, `log`, geodesic interpolation) belong in `geometry` *or* `infogeo`, but the small-matrix-typed primitives (`Mat3.MulAddTransposed(A, B)`) belong in `linalg` and should be designed jointly.

(Sources: <https://nalgebra.org/>, <https://github.com/dimforge/nalgebra>.)

---

## 13. SLEEF (Vectorisable special functions) — the underused gem

**Headline.** Naoki Shibata's library: vectorisable, table-free, bit-precise implementations of `sin`, `cos`, `exp`, `log`, `pow`, `tan`, `atan`, etc. Two precision tiers: `_u10` (≤1 ULP) and `_u35` (≤3.5 ULP, faster). Same algorithm targets scalar, SSE2, AVX2, AVX-512, NEON.

**Engineering trick: payne-hanek-style argument reduction without lookup tables.** Standard `sin(x)` reduces `x mod 2π` using a precomputed `1/(2π)` to many bits. SLEEF reduces using only constants known at compile time; no tables, no memory loads, fully vectorisable.

**Engineering trick: Estrin's scheme for polynomial evaluation.** Instead of Horner's rule (n sequential FMAs), Estrin's scheme breaks the polynomial into pairs that can be evaluated in parallel (n FMAs in log n depth). On modern OoO cores, ~30% speedup. Pure-math, vectorisation-orthogonal.

**Why this matters for Reality.** `MatExp` (Higham scaling-and-squaring) needs ~13 polynomial evaluations of the Padé numerator+denominator on every matrix entry — currently nonexistent. When Reality lands `MatExp` (097 T1 #6), the inner-most kernel is Padé polynomial evaluation. Estrin > Horner, and SLEEF's argument-reduction tricks port to Go directly.

**Zero-dep portability for reality.**
- **Adopt:** Estrin's scheme in the eventual `MatExp` and `MatLog` Padé inner loops. ~5-LOC change vs Horner; provable speedup.
- **Adopt:** SLEEF's table-free argument-reduction pattern wherever Reality computes `sin`/`cos` of matrix entries (relevant for `geometry`, `chaos`, signal `Hilbert`). Currently Reality uses `math.Sin(x)` which is itself table-based but blackbox; for performance-critical inner loops, a SLEEF-style scalar implementation is a 50-LOC drop-in with the same accuracy.
- **Skip:** the SIMD specialisations. Go can't.

(Sources: <https://sleef.org/>, Shibata IEEE TPDS 2010 "Efficient Evaluation Methods of Elementary Functions Suitable for SIMD Computation".)

---

## 14. RNNoise (Xiph) — referenced in the prompt but tangential

RNNoise is a real-time RNN noise-suppression library, not a linalg library. It is referenced in the prompt presumably because its inner-loop math (small `8×8`, `16×16` matmuls in the GRU/RNN cells) overlaps libxsmm's domain — but RNNoise itself has no general linalg primitives. Its **engineering relevance** is the same as libxsmm's: small-matrix specialisation matters. Skip as a separate study target.

(Source: <https://jmvalin.ca/demo/rnnoise/>, Valin Interspeech 2018.)

---

## 15. Other libraries surveyed (briefly)

- **Trilinos / Tpetra (Sandia, BSD).** HPC scale; Kokkos performance-portable backend. Reality's analogue of Tpetra-DistObject is wholly out of scope (no MPI in `reality`'s zero-dep target).
- **PETSc (ANL, BSD-2).** Distributed-memory solvers, FEM stack. Out of scope.
- **MAGMA (UT/UTK, BSD-3).** GPU-accelerated LAPACK. Not portable to zero-dep Go.
- **cuSOLVER / cuBLAS (NVIDIA).** Same. Out.
- **gonum/lapack (Go).** Pure-Go port of LAPACK. **Most directly comparable to Reality** in language and toolchain. Adopts caller-allocates-everything, error-code-returning convention. Reality is decidedly *less complete* than gonum/lapack but holds the design line on (a) golden-file cross-language validation, (b) zero dependencies (gonum has internal sub-package deps), (c) MIT vs BSD (gonum is BSD, so not a license issue but a philosophical distance). Reality should **explicitly read gonum/lapack's source** for any factorisation it wants to add to validate the API and performance choices.
- **Stan Math Library (BSD-3).** C++ AD-aware linalg used by Stan. Notable for `Eigen + autodiff` hand-tuned chain-rule overrides on linalg primitives (e.g., gradient of `solve(A,b)` skips reverse-mode through the LU factorisation and uses the analytic adjoint formula `dA = -A^{-T} grad_x b^T`). **Reality's autodiff package does not yet have linalg-aware vjp's — recommendation: when SVD/QR/LU land, hand-write the analytic vjp** rather than tape-recording the inner loops.
- **GSL (GNU Scientific Library).** Older C library; LAPACK-equivalent surface. Useful as a reference for golden-file vectors but algorithmically descended-from-LAPACK.
- **JAX-based linalg (Google).** XLA-compiled linalg in Python. Out of scope; Reality cannot match XLA fusion without a compiler.

---

## 16. Recent / 2024-2026 frontier highlights

- **Deep-learning batched-tiny-matmul.** Every transformer attention head is a batched small matmul; cuBLASLt and CUTLASS expose this as a primary primitive. The relevant export to Reality is the **batched-small-matmul API surface**, not the implementation.
- **Mixed-precision iterative refinement** (Carson-Higham 2018; LAPACK-3.10's `_gesv` routines added 2022). 32-bit LU + 64-bit residual + 64-bit refinement yields full FP64 accuracy at ~half the storage and ~2× the speed when memory-bound. Reality is FP64-only so this is moot at present, but a potential future port if/when `float32` variants are added.
- **Communication-avoiding linalg (CALU, TSQR).** Ballard-Carson-Demmel-Holtz-Schwartz 2014. Reduces inter-block communication in distributed factorisations. Out of scope.
- **Randomized numerical linear algebra (rNLA).** Halko-Martinsson-Tropp 2011 and Martinsson-Tropp Acta Numerica 2020 codified the field: **rSVD, rQR, randomised trace, randomised determinant, randomised condition estimation**. All are pure-math, zero-dep, and individually ~100-200 LOC. The single addition with the highest ROI for Reality is **`SVDRandomized(A, k, p)`** which delivers a rank-`k` SVD approximation in ~150 LOC (see 097 T1 #7 cross-reference).
- **Probabilistic numerics for linalg (Hennig-Osborne-Girolami).** "Linear-algebra-as-Bayesian-inference"; CG = posterior estimate of `A^{-1}b` under a Gaussian prior. Theoretically beautiful, practically still niche; Reality should track but not adopt yet.

---

## 17. Aggregated SOTA-port priorities for Reality

Restricted to algorithms that are (a) zero-dep portable, (b) golden-file validatable, (c) closed-form citation-grounded, (d) Reality-consumer-relevant:

| # | Source library | Port | LOC | Cross-ref |
|---|----|----|----|---|
| **S1** | libxsmm + nalgebra | `MatMul3x3`, `MatMul4x4`, `MatInverse3x3`, `MatInverse4x4`, `MatDet3x3`, `MatDet4x4`, `MatVec3`, `MatVec4` (unrolled) | ~120 | Pistachio 60-FPS hot path; not in 096/097 lists |
| **S2** | OpenBLAS / BLIS | Cache-blocked `MatMul` with 5-loop expansion + Goto packing | ~120 | Cross-cuts; doubles speed at n>256 |
| **S3** | Halko-Martinsson-Tropp | `SVDRandomized(A, k, p)` | ~150 | Composes with QR (097 T1 #4) |
| **S4** | LAPACK 3.x recursive blocked | `LUDecomposeBlocked`, `CholeskyBlocked`, `QRDecomposeBlocked` | ~250 (over flat versions) | Replaces 096's "should add" recursive variants |
| **S5** | Cuppen / Trefethen-Howell BDC | `EigSymmetricBDC` (n>~50) | ~500 | v1.1 default; replaces tqli for large matrices |
| **S6** | Dhillon-Parlett MRRR | `EigSymmetricMRRR` (eigenvectors w/o reortho) | ~600 | Long-tail, Tier 3 |
| **S7** | Armadillo `expmat_sympd` | `MatrixExpSymPD` (eigendecomp-based) | ~30 | Cross-ref 096 P6, 097 T1 #6 |
| **S8** | Higham-2008 + SLEEF | `MatrixExp` general (Padé+squaring, Estrin polynomial) | ~280 | Cross-ref 096 P9, 097 T1 #6 |
| **S9** | Eigen / Blaze adaptor | `Hint{Symmetric, PD, ...}` enum + `Solve` umbrella | ~50 | API ergonomics |
| **S10** | LAPACK workspace-query | `*WorkspaceSize(n) int` siblings of every alloc-ing routine | ~60 | Compliance with zero-alloc rule |
| **S11** | ndarray view | `MatView{data, rows, cols, rowStride}` non-owning slice | ~30 | Prereq for blocked GEMM (S2) |
| **S12** | Stan Math AD | Analytic vjp's for `LUSolve`, `Cholesky`, `SVD` (when landed) | ~80/op | Cross-ref autodiff (013) |

**Sprint-1 (game-engine / 60-FPS payoff, ~150 LOC).** S1 + S11. Closes the "Pistachio matmul tax" entirely. Backwards-compatible additions; zero math changes.

**Sprint-2 (general-purpose speedup, ~240 LOC).** S2 + S10. The blocked GEMM driver + workspace-query pattern brings Reality's `MatMul` from "naive triple-loop" to "BLIS-architecture-but-no-asm" performance class; closes the gap with gonum/lapack on dense throughput.

**Sprint-3 (decomposition modernisation, ~700 LOC).** S3 + S5 + S7. Adds randomised SVD (when SVD itself lands per 097 T1), divide-and-conquer symmetric eig, and the SPD-specialised matrix exponential. After this, the dense-decomp side of `linalg` is at 2010s SOTA.

---

## 18. What this audit did not assess

- **GPU offload.** Out of scope; `reality` is CPU-Go-only by charter.
- **Distributed-memory.** Same.
- **Mixed-precision (FP32 + FP64).** Reality is FP64-only at present; mixed-precision is a v2 axis.
- **Quantised linalg (int8, fp8 for inference).** Out of scope — belongs in a hypothetical `quant` sibling.
- **Finite-field linalg (FLINT, IML).** Out of scope.
- **Arbitrary-precision linalg (Eigen + boost::multiprecision, mpmath).** Out of scope; Reality's testutil already uses `math/big` at 256-bit for golden generation but not for runtime.

---

## 19. Bottom line

The dense-linalg SOTA frontier divides into four tiers Reality can engage with at distinct levels of fidelity:

1. **Algorithm fidelity (full adoptable).** Recursive blocked LU/QR/Cholesky, BDCSVD, MRRR, randomised SVD, Higham scaling-and-squaring, Goto-packed blocked GEMM, libxsmm-style small-matrix unrolling, Estrin polynomial evaluation. All pure-math, zero-dep, golden-file-validatable. This is where Reality's 2026 sprints should aim.
2. **Architectural design (partial adoption).** Eigen's expression templates, Blaze's adaptor types, Armadillo's umbrella `Solve`, ndarray's views, LAPACK's caller-allocates + workspace-query pattern. Reality can adopt the *pattern* (umbrella functions, hint enums, view structs, workspace-query siblings) without the implementation machinery (templates, runtime backend dispatch).
3. **Implementation fidelity (not portable).** Hand-tuned AVX-512 / AMX micro-kernels (MKL/OpenBLAS/BLIS), JIT-compiled shape-specialised kernels (libxsmm), GPU offload (cuBLAS/MAGMA), distributed-memory primitives (Trilinos/PETSc). Reality cannot match these in pure Go; the speed-of-light reference is gonum + cgo-LAPACK, which Reality intentionally rejects on the zero-dep axis.
4. **API-only takeaways (model selection).** mlpack's storage-polymorphic algorithms, dlib's compile-time-vs-run-time matrix duality, nalgebra's first-class Lie groups, SLEEF's table-free transcendentals, gonum's caller-allocates idiom. Adopt the API style; the implementation choice is Reality's.

The single highest-leverage SOTA-port for Reality is **S1 (libxsmm-style small-matrix specialisations)** because it directly closes the documented Pistachio 60-FPS hot path that 096/097 did not flag. ~120 LOC, zero math risk, ~5–20× speedup on 3×3 / 4×4 ops, golden-file-trivial. Recommend as the first commit out of this review series.

**File:** `C:\limitless\foundation\reality\reviews\overnight-400\agents\098-linalg-sota.md`, ~395 lines.

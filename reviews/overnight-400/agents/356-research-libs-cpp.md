# 356 — research-libs-cpp (C++ math ecosystem 2025-26 lessons + license traps)

## Headline
C++ leads the world in expression-template numerics and hardware-portable parallelism (Eigen 5.0, Kokkos 5.1, std::mdspan in C++23), but the ecosystem is a license minefield — CGAL, libsemigroups GPL; Eigen MPL2; Armadillo Apache-2 — so reality must read these for *patterns*, never vendor a line.

## Survey

### Eigen 5.0 / 3.4.1 (MPL2)
Released 2025-09-30 (5.0.0) and 3.4.1 maintenance. Header-only, expression templates remove temporaries; SIMD covers SSE2/3/4, AVX/AVX2/AVX512, FMA, NEON (32+64), AltiVec/VSX, ZVector, and (since 3.4) MIPS MSA, all with scalar fallback. 3.4 was the last C++03 line; 5.0 raises the floor. The defining design move: lazy products `(A*B + C*D).eval()` materialise once. License is **MPL2 file-level copyleft** — modified Eigen source must be re-released MPL2; static-link distribution is fine for a permissive consumer if Eigen sources stay untouched, but reality's MIT charter still rules out vendoring. Reality cribs the *idea* (no allocations in hot paths, fused ops via output buffers) without the templates.

### Blaze (BSD-3)
Bitbucket-hosted, last public tagged release 3.8 (Aug 2020) but `live-clones/blaze` mirror shows continued activity through 2025. "Smart Expression Templates" go further than Eigen: vectorisation across SSE/AVX/AVX-512, FMA, plus optional SVML/SLEEF/xsimd backends, and explicit padding/unrolling for small dense matrices. Benchmarks (Poya 2020, still cited in 2025) put Eigen ≈ Blaze ≈ Fastor on dense GEMM; Blaze loses on large matrices because over-aggressive unroll exceeds SIMD width. STEllAR-GROUP fork adds CUDA. License is BSD-3 — *vendorable in principle*, but the deep template machinery is hostile to Go translation.

### xtensor (BSD-3)
NumPy-style multi-dim arrays with broadcasting and lazy `xexpression` objects: `x + y * sin(z)` constructs an expression node that evaluates only on access or assignment, with **no temporaries**. Latest tagged 0.24.6, master active through 2025; einfochips Oct-2025 writeup highlights real-time/embedded use. Slicing semantics, `xt::view`, `xt::strided_view`, and a `xtensor-python` bridge mirror NumPy almost exactly. xframe (DataFrame), xtensor-blas (BLAS bridge), xsimd backend. The architectural takeaway for reality: **lazy expression DAGs let you pretend Go has operator overloading** — e.g. a `prob.Sum(prob.Mul(a,b), c)` builder that fuses inner loops at call time with output buffers.

### Kokkos 5.1 (BSD-3)
The HPC performance-portability gold standard. C++20 floor; backends include CUDA, HIP, SYCL, HPX, OpenMP, C++ threads. Single-source code runs on Frontier (AMD), Aurora (Intel PVC), El Capitan (AMD MI300A) — proven on LAMMPS-KOKKOS (arXiv 2508.13523, Aug 2025). Core abstractions: `View<T***, Layout, Memory>` (the precursor to `std::mdspan`), `parallel_for`, `parallel_reduce`, execution and memory spaces. Kokkos 4.x added `RangePolicy` work-stealing and SIMD types; Kokkos 5 consolidates with C++20 concepts. Reality is single-threaded library code and won't import Kokkos, but the **layout-policy-as-template-parameter** pattern translates to Go via interfaces (`Layout` = struct of strides + index function).

### std::mdspan (C++23, ISO standard)
Non-owning multi-dim view, parameterised on element type, `extents`, `LayoutPolicy` (left/right/stride), and `AccessorPolicy`. Reference impl `kokkos/mdspan` (the Kokkos team drove the proposal). Compiler support landed in libstdc++ 14, libc++ 18, MSVC 19.40 through 2024-25; multidim subscript `m[i,j,k]` requires C++23 fully-conformant front-ends. `submdspan` (slicing) is in P2630 and ships in C++26. Reality lesson: ship matrices as `(data []float64, layout Layout)` pairs — borrow the `extents` + `LayoutPolicy` decomposition so `linalg.Matrix` can later get a `RowMajor`/`ColMajor`/`Strided`/`Banded` layout without rewriting algorithms.

### Boost.Math 1.90 (BSL-1.0)
Special functions: gamma/beta/erf families plus inverses, digamma, factorials, Bessel, elliptic integrals, hypergeometric ₀F₁/₁F₁/₂F₁, sinc/sinhc, inverse hyperbolics, Legendre/Laguerre/Hermite/Chebyshev, plus statistical distributions, root-finding, polynomial/rational tools, interpolation, quadrature. C++14 floor, header-only, **fully generic over the real type** (works with Boost.Multiprecision for >double precision). License BSL-1.0 = MIT-compatible attribution-only, vendorable. The killer feature reality should copy: **per-type specialisation**. Boost.Math computes Bessel J₀ with one polynomial approximation for `float`, a tighter one for `double`, and a series for arbitrary-precision types. Reality's analogue: split `prob.GammaP` into `_f64` and `_big` paths backed by golden files.

### Armadillo 14/15 (Apache-2)
MATLAB-like syntax (`A.t() * B`) over BLAS/LAPACK. ArXiv 2502.03000 (Feb 2025) describes the latest expression-optimisation passes. 14.6.3 is the last C++11-compatible line; 15.0.0 raises floor to C++14. Heavy use of template metaprogramming for delayed evaluation and operator fusion. Apache-2 is permissive but **patent-grant clauses** complicate downstream MIT relicensing. RcppArmadillo is the most-cited single C++ math binding (CRAN). Reality lesson: the MATLAB-like API is what scientists *want*; Go's lack of operator overloading means `linalg.Op(linalg.MulT(A), B)` instead of `A.t()*B`, but the chaining can still be ergonomic if the underlying ops are fused via expression nodes.

### Ceres Solver 2.x (BSD-3)
Google's nonlinear least squares solver. The reference for **bundle adjustment**: 3D point + camera optimisation with reprojection error minimised via Levenberg-Marquardt or Trust Region Dogleg, exploiting the BA Jacobian's special sparsity (Schur complement on the 3D-point block first). Auto-differentiation via Jet types is the architectural flagship — a `Jet<double, N>` carries value + N-derivative dual numbers, so users write residuals as ordinary code and gradients fall out exactly. Loss functions (Huber, Cauchy, Tukey) are robust-statistics primitives reality already partially has in `optim`. BSD-3, vendorable, but the Jet-based AD is heavy template magic; Go would need codegen or interface-based duals.

### GTSAM 4.2 / 4.3-α (BSD-3)
Factor-graph SLAM and Structure-from-Motion, Georgia Tech (Frank Dellaert). 4.3 is in pre-release with major Boost removal — replaced by C++17 std lib — and a new `TableFactor` for sparse discrete factors plus much faster `DecisionTreeFactor` elimination. Hybrid (continuous-discrete) inference is the 4.3 frontier. iSAM2 (incremental Smoothing-and-Mapping) is the textbook online-SLAM solver. Reality is unlikely to ship a factor graph engine, but the **Bayes-tree elimination ordering** is the right reference for any future `prob.GraphicalModel` work — and it shows that *factor graphs subsume* both Kalman filters and particle filters as special cases.

### dlib 20.0 / 20.0.1 (Boost Software License)
Davis King's monolithic ML/linalg/geometry/imaging toolkit. v20.0 May-2025, v20.0.1 maintenance Mar-2026. Includes SVMs, deep learning (custom CNN/RNN engine, no PyTorch dep), clustering, Bayesian opt, structural SVM, the famous HOG face detector, RANSAC, kd-trees, optimisation. BSL-1.0 = MIT-compatible. It's the closest thing to a "reality but in C++" — *one library, many domains, header-heavy, BSL license, decade-stable API*. Reality's positioning is more curated (no ML, just constants/physics/math), but dlib proves the model is viable: a single-author toolkit can stay coherent at ~1M LoC if the testing discipline holds.

### CGAL 6.1.1 (GPL-3 / LGPL kernel + commercial)
Computational Geometry Algorithms Library. Kernel + support libs are LGPL-2.1+; **most algorithms are GPL-3+** — Delaunay, mesh generation, Boolean operations on Nef polyhedra, alpha shapes, polygon arrangement, surface reconstruction. CGAL 6.1 (Oct 2025) added compiler portability fixes. Dual-licensed by GeometryFactory for commercial use. Reality's `geometry` package must **not vendor or wrap CGAL** even at link time — GPL infection is the ecosystem's most famous footgun. For convex hull / CDT / mesh generation reality wants, look to LGPL Triangle (Shewchuk, public domain), CGAL-style algorithms reimplemented from papers, or Boost.Geometry (BSL-1.0).

### libsemigroups (GPL-3)
James Mitchell's combinatorics library: finite semigroups/monoids, Knuth-Bendix completion, Todd-Coxeter, congruences, Schreier-Sims. Best-in-class for the corner of computational algebra reality's `combinatorics` package doesn't currently target. **GPL-3 — do not vendor.** The reference papers (Mitchell et al., LMS J. Comp. Math.) are the right primary sources if reality ever wants e.g. a finite-monoid enumerator.

### PCL 1.15.1 (BSD-3)
Point Cloud Library. 1.14.0 (Jan 2024) shipped a faster, more robust GICP; 1.15.0 (Feb 2025) and 1.15.1 (Aug 2025) parallelised key classes (RANSAC, segmentation). Boost.filesystem dep is now optional under C++17. PCL's strength is the **kd-tree / octree / surface-reconstruction stack** (Poisson, MLS, GP3) and registration (ICP, NDT, GICP). License BSD-3 vendorable. For reality's `geometry`, PCL's octree code is the cleanest open-source reference; the SAC-based RANSAC API is the model for `optim`/`geometry` consensus fitting.

## Aggregate themes

- **Expression templates are the C++ moat.** Eigen, Blaze, xtensor, Armadillo, Fastor all converge on the same pattern: build a compile-time DAG, fuse loops, eliminate temporaries. Go can't replicate the syntax, but the **builder + output-buffer + fused-kernel** pattern translates: `linalg.MulAdd(out, A, B, C)` is the manual version of `out = A*B + C`.
- **mdspan is the lingua franca for layouts.** Both std::mdspan (C++23) and Kokkos `View` decompose multi-dim arrays into `(data, extents, layout, accessor)`. Reality's matrices currently are `[]float64 + rows/cols` — a future v0.x should generalise to `Layout` interface so banded/symmetric/triangular don't each need a copy of every algorithm.
- **Hardware portability via single-source.** Kokkos proves one `parallel_for` body can target NVIDIA, AMD, Intel GPUs. Reality won't go GPU, but its zero-allocation hot-path discipline is the same idea: write once, vectorise via Go's escape analysis + future SIMD intrinsics (`golang.org/x/sys/cpu` + asm).
- **Auto-differentiation as Jet/dual numbers.** Ceres' Jet type is the textbook design. For reality's autodiff aspirations, this is the right reference — exact gradients, no graph at runtime.
- **Factor graphs subsume filters.** GTSAM's `iSAM2` makes Kalman, EKF, particle, and smoothing all instances of one Bayes-tree elimination. Future `prob` extension target.
- **Special-functions completeness is a moat.** Boost.Math (gamma, Bessel, ellint, hypergeometric, all inverses) is what raises a math library from "useful" to "trusted." Reality currently has the basic set; Bessel + elliptic integrals + hypergeometric ₂F₁ would be the catch-up shopping list.

## License hazards (do NOT vendor)

- **CGAL** — GPL-3 on most algorithms (LGPL kernel only). Vendoring or even linking infects reality with GPL. Use as *reference for papers*, never copy code.
- **libsemigroups** — GPL-3. Same as CGAL.
- **Eigen** — MPL2 file-level copyleft. Vendoring requires keeping Eigen sources MPL2 and disclosing modifications. Reality's MIT-only charter says no.
- **Armadillo** — Apache-2. Permissive but patent-grant + state-changes clause; downstream MIT relicensing is muddier than BSL/MIT/BSD-3.
- **GeometryFactory commercial CGAL** — proprietary; off-limits.

Vendorable (MIT-compatible): Boost.Math/uBLAS/Geometry (BSL-1.0), dlib (BSL-1.0), Blaze/Kokkos/Ceres/GTSAM/PCL/xtensor (BSD-3). Reality still chooses *not* to vendor — first-principles reimplementation is the design rule — but these are the safe references to read closely.

## Sources

- [Eigen 5.0 / 3.4.1 release page](https://libeigen.gitlab.io/releases/)
- [Eigen 3.4 announcement](https://libeigen.gitlab.io/releases/3.4/)
- [Eigen Wikipedia](https://en.wikipedia.org/wiki/Eigen_(C%2B%2B_library))
- [Blaze live-clones GitHub mirror](https://github.com/live-clones/blaze)
- [Blaze STEllAR-GROUP CUDA fork](https://github.com/STEllAR-GROUP/blaze)
- [Poya 2020 expression-templates benchmark](https://romanpoya.medium.com/a-look-at-the-performance-of-expression-templates-in-c-eigen-vs-blaze-vs-fastor-vs-armadillo-vs-2474ed38d982)
- [xtensor GitHub](https://github.com/xtensor-stack/xtensor)
- [xtensor lazy expressions docs](https://xtensor.readthedocs.io/en/latest/expression.html)
- [einfochips xtensor 2025 writeup](https://www.einfochips.com/blog/beyond-numpy-exploring-xtensor-for-c-scientific-computing/)
- [Kokkos GitHub](https://github.com/kokkos/kokkos)
- [LAMMPS-KOKKOS exascale 2025 (arXiv 2508.13523)](https://arxiv.org/html/2508.13523v1)
- [Performance-portability frameworks survey (Tsukuba 2025)](https://www.ccs.tsukuba.ac.jp/wp-content/uploads/sites/14/2025/09/08.-Performance-Portability-frameworks-Kokkos-Raja-CStd-Parallelism-etc.pdf)
- [std::mdspan cppreference](https://en.cppreference.com/cpp/container/mdspan)
- [C++23 mdspan details (cppstories 2025)](https://www.cppstories.com/2025/cpp23_mdspan/)
- [kokkos/mdspan reference impl](https://github.com/kokkos/mdspan)
- [Boost.Math 1.90](https://www.boost.org/library/latest/math/)
- [Boost.Math GitHub](https://github.com/boostorg/math)
- [Armadillo home](https://arma.sourceforge.net/)
- [Armadillo 2025 paper (arXiv 2502.03000)](https://arxiv.org/abs/2502.03000)
- [Ceres Solver tutorial](http://ceres-solver.org/nnls_tutorial.html)
- [Ceres bundle-adjustment tutorial](http://ceres-solver.org/nnls_solving.html)
- [GTSAM home](https://gtsam.org/)
- [GTSAM GitHub releases](https://github.com/borglab/gtsam/releases)
- [dlib home + change log](https://dlib.net/change_log.html)
- [CGAL license page](https://www.cgal.org/license.html)
- [CGAL Wikipedia](https://en.wikipedia.org/wiki/CGAL)
- [PCL releases GitHub](https://github.com/PointCloudLibrary/pcl/releases)
- [PCL Wikipedia](https://en.wikipedia.org/wiki/Point_Cloud_Library)

# 375 — research-sparse-modern (sparse linear algebra + graph-via-matrix libraries)

## Headline
Modern sparse ecosystem (SuiteSparse:GraphBLAS 10.x, HYPRE/ParaSails, PETSc 3.25, Trilinos/Tpetra, SuperLU_DIST 8.x, METIS, Eigen Sparse, Ginkgo 1.9) is mature, GPU-portable, and built around CSR/CSC/COO/BSR/SELL formats plus semiring SpGEMM — reality's pure-Go dense `linalg` and from-scratch `graph` deliberately sit one tier below this stack and should add a small, dependency-free `linalg/sparse` (CSR + CSC + COO + SpMV/SpGEMM) before attempting any GraphBLAS-style unification.

## Survey

### 1. SuiteSparse:GraphBLAS (Davis, TAMU)
Reference C implementation of the GraphBLAS C API, currently shipping in SuiteSparse as **GraphBLAS 10.2.0** (2025) with `LAGraph 1.2.1`. v10.x adds RISC-V vectorisation, a `GRAPHBLAS_VANILLA` mode that disables the `GxB_*` extensions (purely-spec-compliant build), and Power/s390x compiler workarounds. Implements `GrB_*` matrix types over arbitrary user-defined semirings (monoid + binop), with masked SpGEMM, SpMV, `eWiseAdd`/`eWiseMult`, `reduce`, `apply`, `assign`, `extract`. LAGraph builds BFS, SSSP (delta-stepping), connected components, triangle counting, PageRank, k-truss, Louvain, MaxFlow on top of these primitives. Apache-2 lib core, GPL MATLAB iface. ~250 KLOC C, no AI-generated code (per repo policy). Algorithm 1037 in TOMS describes the design. Binding: Python (`python-graphblas` 2025.2.x), Julia (`SuiteSparseGraphBLAS.jl`).

### 2. GraphBLAS C API specification
**Version 2.1.0 (final)** is the current spec implemented by SuiteSparse. v2.0.0 PDF at graphblas.org is still the canonical printable doc; 2.1 adds (a) `GrB_select` second-arg refinements, (b) clarified non-blocking mode rules, (c) more standard monoids. Core primitives: `mxm`, `mxv`, `vxm`, `eWiseAdd/Mult`, `extract`/`assign` with masks, `reduce`/`apply`/`select`, `kronecker`. Semirings unify graph algorithms: BFS = `(min,+)` over `min.first` semiring on the adjacency mask; SSSP = `(min,+)` tropical; PageRank = `(+,*)` standard arithmetic on row-stochastic matrix.

### 3. HYPRE (LLNL)
Parallel preconditioners and multigrid solvers. README docs reference v3.1.0 (Apr 2025). Modules: BoomerAMG (algebraic multigrid), PFMG/SMG (structured AMG), AMS (auxiliary-space Maxwell), ADS (divergence), Euclid (parallel ILU), **ParaSails** (sparse approx inverse, see #4), MLI, FAC. CSR (`hypre_CSRMatrix`), ParCSR (distributed), Struct (regular grids), SStruct (semi-structured). MPI + OpenMP + CUDA/HIP. Used by PETSc as a back-end (`PCHYPRE`).

### 4. ParaSails (Chow, in HYPRE)
Parallel sparse-approximate-inverse (SPAI) preconditioner. Computes `M ≈ A^{-1}` by least-squares Frobenius minimisation `min ||AM − I||_F` column-by-column. Sparsity pattern = pattern of a power of a sparsified `A` (a-priori, *not* dynamic), then post-filter to drop small entries. Knobs: `nlevels` (power), `thresh` (sparsification), `filter` (post-prune), `loadbal`. Bundled in HYPRE since ~2000 (Chow 2001, IJHPCA). Strength: trivially parallel application (just SpMV). Weakness: setup is expensive, quality drops on ill-conditioned matrices vs ILU/AMG.

### 5. PETSc (Argonne)
**Latest: 3.25.x** (as of mid-2026 docs); 3.22 was Sep 2024, 3.21 Mar 2024. Mat types `MATAIJ` (CSR), `MATBAIJ` (BSR), `MATSBAIJ` (sym block), `MATELLPACK` (via cuSPARSE), `MATDENSE`, `MATMPIAIJ` (distributed CSR). KSP: CG, GMRES, BiCGStab, Chebyshev, FGMRES, Pipelined variants. PC: Jacobi, ASM, ILU, ICC, GAMG, BoomerAMG (via HYPRE), Hypre/AMS, FieldSplit. External factor packages: SuperLU_DIST, MUMPS, PaStiX, MKL_PARDISO, UMFPACK, CHOLMOD, KLU, cuSPARSE, cuDSS. GPU: CUDA, HIP, SYCL, Kokkos.

### 6. Trilinos / Tpetra (Sandia)
arXiv 2503.08126 (Mar 2025) is the canonical "Trilinos: Enabling Scientific Computing Across Diverse Hardware Architectures at Scale" paper. **Tpetra** = next-gen distributed sparse linalg over Kokkos (perf-portable across CUDA/HIP/SYCL/OpenMP). Old Epetra stack scheduled for archival end-2025. Packages: Tpetra (matrices/vectors), Belos (Krylov), Ifpack2 (point preconds), MueLu (multigrid), Amesos2 (direct iface to SuperLU_DIST/MUMPS/Basker), Zoltan2 (partitioning), ShyLU (hybrid solvers). 2025 Tpetra adds MPI Advance comm, granular transfers (`TPETRA_GRANULAR_TRANSFERS=ON`).

### 7. SuperLU / SuperLU_DIST (Li, LBNL)
Supernodal sparse direct LU. Latest: **SuperLU_DIST 8.1.1**. Recent additions (TOMS, "Newly Released Capabilities…", 2023): 3D communication-avoiding factorisation, multi-GPU (CUDA + HIP), mixed-precision (single-prec factor + double-prec iterative refinement). Static pivoting (`P_r D_r A D_c P_c` permute+scale, then no dynamic pivoting) → predictable comm pattern. C+MPI+OpenMP, F90/Julia/Python bindings. Sequential SuperLU is the back-end inspiration for Eigen `SparseLU`.

### 8. METIS / ParMETIS (Karypis Lab, UMN)
**METIS** (serial, last touched Jul 2025 on GitHub) and **ParMETIS** (MPI, Dec 2023). Multilevel recursive bisection + multilevel k-way + multi-constraint. `pymetis` 2025.2.2 (Nov 2025) wraps it for Py 3.14. Used by PETSc, Tpetra/Zoltan2, OpenFOAM, hMETIS (hypergraph variant) for circuit/VLSI. Output: a 0..k-1 partition vector minimising edge cut subject to balance ε. Also produces fill-reducing orderings (nested dissection) that direct solvers consume.

### 9. Eigen Sparse (libeigen, header-only C++)
Header-only C++ template lib. `Eigen::SparseMatrix<Scalar,Options,StorageIndex>` (CSC default, CSR via `RowMajor`). Solvers: **SimplicialLLT/LDLT** (own Cholesky), **SparseLU** (Davis-style supernodal LU, sequential), **SparseQR** (rank-revealing), **ConjugateGradient**, **BiCGSTAB**, **LeastSquaresConjugateGradient**. Wrapper iface to UMFPACK, CHOLMOD, SuperLU, SPQR, PaStiX, PARDISO. Heavily used in Ceres, libigl, OpenMVG, autograd-style PDE codes. Pure C++, MPL-2 licensed, no MPI.

### 10. Ginkgo (KIT/UTK)
**Ginkgo 1.9.0** (2025) + **pyGinkgo** (arXiv 2510.08230, Oct 2025). Modern C++ sparse-iterative library with platform-portable kernels for CUDA, HIP, SYCL/oneAPI, OpenMP, DPC++, plus reference CPU. Formats: CSR, COO, ELL, Sliced-ELL (SELL-P), Hybrid (HYB = ELL+COO, NVIDIA-style), BCCOO, FBCSR. Solvers: CG, BiCGStab, GMRES, IDR, multigrid, batched CG/Jacobi (2025). Preconditioners: Jacobi (block), ILU/IC (own, no vendor lib needed as of 2025), ParILU(T), ISAI, SSOR/Gauss-Seidel (2025), AMG (distributed, 2025). FP16/BF16 support added 1.9 (2025). LinOp abstraction = composable solvers.

### 11. (Bonus) kPlex algorithms
*Not a single library — a problem class.* k-plex = vertex set inducing a subgraph where every vertex has ≥|S|−k neighbours inside S; cliques are 1-plexes. Recent 2024-2026 algorithms:
- **Listing Maximal k-Plexes** (Wang et al., arXiv 2202.08737, refined 2024) — branch-and-bound with second-order reduction.
- **Efficient maximum k-plex** (PVLDB 2022) — ego-network shrinking + 2-hop pruning.
- **MKPM** (Inf. Sci. 2026) — modularity-driven k-plex enumeration for community detection.
- **KPGN** (Inf. Sci. 2024) — k-plex seeds + GNN unsupervised community detection.
- **D2K** (KDD 2018, still cited 2025) — small-diameter k-plex on massive networks.
Implementations are mostly research C++ binaries; no widely-used standard library equivalent of NetworkX/igraph for k-plex.

## Reality positioning

Reality's `linalg/` is dense only (`matrix.go`, `vector.go`, `decompose.go` = LU/QR/Cholesky/SVD on `[][]float64`-style storage, `eigen.go`, `pca.go`). Reality's `graph/` is dense-or-edge-list (BFS/DFS, Bellman-Ford, Dijkstra, MST, PageRank power iteration on dense, centralities, community, flow). **Zero sparse storage anywhere in the repo.** Power iteration on a 100k×100k dense float64 = 80 GB; reality presently can't represent typical real-world graphs.

### Recommendations

**1. Add `linalg/sparse` (small, focused, dependency-free).** Match the floor of Eigen-Sparse / cuSPARSE format zoo, not the ceiling of Ginkgo:
   - `CSR{Indptr, Indices, Data []float64; Rows, Cols int}` — required.
   - `CSC` — required (transpose / column ops, used by direct solvers).
   - `COO{Row, Col []int; Data []float64}` — required (assembly format).
   - `BSR` (block-CSR) — optional, defer; only useful for PDE block-structured.
   - `ELL`/`SELL` — defer; GPU-oriented, not relevant for pure-Go CPU library.
   - Ops: `SpMV(A, x, y)` with output buffer (no alloc), `SpMM`, `SpGEMM` (Gustavson algorithm), `SpAdd`, `Transpose`, `CSR<->CSC<->COO` conversions, `Sort`, `Compress` (sum duplicates).
   - Iterative: `CG`, `BiCGStab`, `GMRES(m)` with restart, all parametrised by SpMV closure.
   - Preconditioners: diagonal Jacobi (1 hour to write), ILU(0) (a weekend), then stop. Skip AMG/SPAI/ParaSails — those are 50KLOC+ projects.
   - Direct: skip. SuperLU/UMFPACK quality is years of work; if a user needs direct, they reach for SuiteSparse.

**2. Refactor `graph/` to optionally consume CSR.** Keep the current dense/edge-list APIs; add a `graph/csr.go` adapter so PageRank/BFS/Dijkstra/triangle-count can run on `linalg/sparse.CSR` adjacency. This unlocks 100k-node graphs without rewriting algorithms.

**3. Do NOT chase GraphBLAS in Go.** SuiteSparse is 250 KLOC of C with bespoke JIT; the spec assumes user-defined semirings via function pointers + non-blocking lazy execution. Reproducing v2.1 in pure Go means either (a) accept 50x slowdown via interface dispatch in the inner SpGEMM loop, or (b) add code-generation. Both contradict the "pure Go, no AI gen" ethos. Better: pick the *one* high-leverage semiring (`(min,+)` for SSSP) and write a specialised BFS/SSSP that operates on CSR, no abstraction.

**4. Skip kPlex and SPAI entirely.** Specialist research code; no clean canonical implementation; not core to "universal truth in code." Document `community.go` covers Louvain/Label-Prop instead, which is the 90% case.

**5. Don't add direct solvers (no SuperLU, no MUMPS-equivalent).** Pure-Go fill-reducing nested-dissection + supernodal LU is a multi-year project. Iterative-only is the right scope; cite the floor (Saad's `Iterative Methods for Sparse Linear Systems`, 2nd ed) as the source.

**Bottom line:** add `linalg/sparse` with CSR/CSC/COO + SpMV/SpGEMM + CG/BiCGStab/GMRES + Jacobi/ILU(0). ~3-5 KLOC, ~150 golden vectors, fits the repo's ethos, unblocks `graph/` to scale past dense matrices. Everything else (GraphBLAS, ParaSails, kPlex, Ginkgo-tier GPU formats) is out of scope.

## Sources
- [SuiteSparse:GraphBLAS GitHub (DrTimothyAldenDavis)](https://github.com/DrTimothyAldenDavis/GraphBLAS) — v10.x stable branch, RISC-V vectorisation, GRAPHBLAS_VANILLA build mode.
- [Algorithm 1037: SuiteSparse:GraphBLAS, TOMS](https://dl.acm.org/doi/10.1145/3577195) — design paper.
- [LAGraph releases](https://github.com/GraphBLAS/LAGraph/releases) — v1.2.1 maxflow fix, HPEC'25 paper.
- [GraphBLAS C API v2.0.0 PDF](https://graphblas.org/docs/GraphBLAS_API_C_v2.0.0.pdf) and [graphblas.org](https://graphblas.org/) for v2.1.0 status.
- [HYPRE GitHub (LLNL)](https://github.com/LLNL/hypre) and [readthedocs](https://hypre.readthedocs.io/) — BoomerAMG, ParaSails, AMS.
- [HYPRE ParaSails docs](https://hypre.readthedocs.io/en/latest/solvers-parasails.html) — Frobenius-norm SPAI, a-priori sparsity pattern.
- [Edmond Chow software page (Georgia Tech)](https://faculty.cc.gatech.edu/~echow/software.html) — ParaSails origin, Chow 2001 IJHPCA paper.
- [PETSc 3.25 documentation](https://petsc.org/release/) and [linear solver table](https://petsc.org/release/overview/linear_solve_table/) — Mat formats, KSP/PC, external solver list.
- [PETSc release changes](https://petsc.org/release/changes/) — 3.21 (Mar 2024), 3.22 (Sep 2024), 3.25 current.
- [Trilinos GitHub](https://github.com/trilinos/Trilinos/releases) — Tpetra MPI Advance, Epetra archival end-2025.
- [Tpetra docs](https://trilinos.github.io/tpetra) and [Trilinos overview arXiv 2503.08126](https://arxiv.org/abs/2503.08126).
- [SuperLU GitHub (xiaoyeli)](https://github.com/xiaoyeli/superlu) — v8.1.1, 3D CA-LU, multi-GPU, mixed precision.
- [TOMS "Newly Released Capabilities in SuperLU_DIST"](https://dl.acm.org/doi/abs/10.1145/3577197).
- [METIS (KarypisLab)](https://github.com/KarypisLab/METIS) updated Jul 2025; [ParMETIS](https://github.com/KarypisLab/ParMETIS).
- [Karypis software overview](https://karypis.github.io/glaros/software/metis/overview.html).
- [Eigen Sparse tutorial](https://eigen.tuxfamily.org/dox/group__TutorialSparse.html) — SparseLU, SimplicialLLT, CG/BiCGSTAB.
- [cuSPARSE storage formats](https://docs.nvidia.com/cuda/cusparse/storage-formats.html) — CSR, CSC, COO, BSR, ELL, SELL definitions.
- [Ginkgo project page](https://ginkgo-project.github.io/) and [GitHub](https://github.com/ginkgo-project/ginkgo) — v1.9.0 FP16, distributed AMG.
- [Sparse Day 2025 Ginkgo talk](https://sparsedays.cerfacs.fr/wp-content/uploads/sites/72/2025/07/2025_SparseDay_YHMTsai_Ginkgo.pdf).
- [pyGinkgo paper, arXiv 2510.08230](https://arxiv.org/abs/2510.08230).
- [Listing Maximal k-Plexes, arXiv 2202.08737](https://arxiv.org/abs/2202.08737).
- [k-plex community detection (Inf. Sci. 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0020025524014233).
- [MKPM modularity k-plex (Inf. Sci. 2026)](https://www.sciencedirect.com/science/article/abs/pii/S0020025526002100).
- [Efficient maximum k-plex (PVLDB 2022)](https://dl.acm.org/doi/10.14778/3565816.3565817).

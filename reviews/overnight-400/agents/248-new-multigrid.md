# 248 | new-multigrid — Multigrid solvers: V/W cycles, FMG, AMG (algebraic)

**Summary line 1.** reality v0.10.0 ships **ZERO** multigrid surface and ZERO iterative-Krylov / classical-stationary-iteration / preconditioner / sparse-matrix surface — repo-wide grep on `multigrid|MGV|MGW|FMG|AMG|smooth.*operator|prolongat|restrict.*grid|coarsen|coarse.*grid|V.cycle|W.cycle|F.cycle|gauss.seidel|jacobi.iter|red.black|SOR\b|over.relaxation|chebyshev.smooth|polynomial.smooth|conjugate.gradient|preconditioned.cg|pcg|gmres|bicgstab|krylov|ruge.stuben|smoothed.aggregat|brandt|hackbusch|stuben|vanek.brezina.mandel|cascadic|wavelet.multigrid|p.multigrid|hp.multigrid|spectral.multigrid|schwarz.smoother|block.smoother|ilu|incomplete.lu|incomplete.cholesky|graph.laplacian.solver|combinatorial.multigrid|koutis.miller|spielman.teng` against `*.go` returns **zero callable matches** anywhere outside review-corpus and outside one private 80-LOC Thomas tridiagonal sweep buried inside `optim/interpolate.go:62-99` (cubic-spline natural-BC second-derivative solver — not exposed, not reusable, not generic). All adjacent surface is *direct* dense-LU/Cholesky O(n³) (`linalg.LUSolve`, `linalg.CholeskySolve`) or fixed-point Picard ODE-marching (`chaos.RK4Step`); zero callable iterative-linear-solver substrate exists. Slot 248 is **STRICT-DOWNSTREAM of 244** (D3 Laplacian-stencil + D11 Poisson2D-Jacobi/GS/SOR + D12 ConjugateGradient + D27 GMRES — multigrid IS a solver-or-preconditioner FOR the linear systems 244 assembles), **STRICT-DOWNSTREAM of 245** (S22 Patera-1984 SEM and S1+S2+S5+S12 Chebyshev/Legendre/GLL substrate for spectral-multigrid + p-multigrid level-transfer between p-orders), **STRICT-DOWNSTREAM of 247** (M1+M2+M5 reference-element + Lagrange-P_k + DOF-map + M3 hierarchical Legendre-modal hp-basis enabling p-coarsening + M16 Wohlmuth dual-mortar enabling cross-mesh prolongation), **STRICT-DOWNSTREAM of 097** (T1 SparseMatrix CSR/CSC — every realistic AMG operates on 10⁴-10⁹-DOF sparse matrices; dense-Galerkin O(n³) coarse-grid construction is untenable beyond N=200), **STRICT-DOWNSTREAM of 246** (X12 cot-Laplacian → graph-multigrid for mesh-Poisson on manifolds; X4 d_k incidence matrix for graph-Laplacian-AMG). Slot 248 is **STRICT-UPSTREAM of 249-domain-decomp** (multigrid-Schwarz hybrid + BPX-additive-multilevel-preconditioner Bramble-Pasciak-Xu-1990) and **CROSS-LINK 215-compressed-sensing** (multilevel-CS recovery), **CROSS-LINK 237-GP-regression** (kernel-multigrid for hierarchical-matrix Gaussian-process scaling), **CROSS-LINK 097-graph-Laplacian-AMG** (Spielman-Teng-2004 nearly-linear graph-Laplacian solver = AMG-as-graph-algorithm). PARTIAL OVERLAP with 244 (244 owns Jacobi/GS/SOR-as-PDE-solver; 248 owns them as smoothers; single-source-of-truth recommendation: 248 owns the smoother-API, 244 imports), 245 (p-multigrid level-transfer), 247 (hp-multigrid h-then-p coarsening Babuška-Suri-1990).

**Summary line 2.** Twenty-six multigrid primitives **G1-G26** totalling ~3,400 LOC organized as **(a) Tier-0 smoother + transfer-operator substrate ~620 LOC** (G1 `multigrid/smoother.go` Jacobi + weighted-Jacobi `ω=2/3` Brandt-1977-optimal-2D-Laplacian + Gauss-Seidel forward/backward/symmetric + Red-Black-GS Stuben-Trottenberg-1982 + Chebyshev-polynomial-smoother Adams-Brezina-Hu-Tuminaro-2003 + SOR with optimal `ω=2/(1+sin(πh))` for 5-pt Laplacian Frankel-1950 ~200 LOC, all in-place zero-alloc; G2 `multigrid/restriction.go` full-weighting `[1 2 1; 2 4 2; 1 2 1]/16` (the canonical 2D restriction Trottenberg-Oosterlee-Schuller-2001 §2.3) + injection (degenerate, defect on smooth modes, used only at coarsest) + half-weighting Brandt-1977-§5 ~140 LOC; G3 `multigrid/prolongation.go` linear-1D + bilinear-2D + trilinear-3D interpolation + Galerkin-projection-consistency `R = c·P^T` constant `c=1/4` for full-weighting-bilinear ~140 LOC; G4 `multigrid/galerkin.go` coarse-grid Galerkin assembly `A_H = R A_h P` Brandt-1977 vs operator-rediscretisation `A_H = h²-rescaled-stencil` ~80 LOC; G5 `multigrid/grid_hierarchy.go` GMG hierarchy 1D/2D/3D + cell-centered-vs-vertex-centered + level-data-transfer ~80 LOC), **(b) Tier-1 keystone V/W/F/FMG cycles ~580 LOC** (G6 `multigrid/vcycle.go` recursive V-cycle Brandt-1977: pre-smooth ν₁ × {restrict residual to coarse → recursive V-cycle on coarse → prolong correction} × post-smooth ν₂ ~140 LOC; G7 `multigrid/wcycle.go` W-cycle = V-cycle with TWO recursive coarse-solves at each level — costs 2× more work but converges robustly on harder problems Hackbusch-1985 §6.4 ~80 LOC; G8 `multigrid/fcycle.go` F-cycle = 1× recursive at finest, 2× at next, etc. — interpolating between V and W cost-vs-robustness Trottenberg-Oosterlee-Schuller-2001 §2.4 ~60 LOC; G9 `multigrid/fmg.go` Full-Multigrid Brandt-1977-MathComp-31: solve coarsest exactly, then **prolong → V-cycle** at each finer level — provably O(N) work to ‖discretisation-error‖_h (the optimal-PDE-solver-complexity-result, Hackbusch-1985-Thm-7.2.2) ~140 LOC; G10 `multigrid/twogrid.go` two-grid analysis-helper for spectral-radius witness `‖S^{ν₂} K_{H←h} S^{ν₁}‖` ~80 LOC; G11 `multigrid/coarsest_solver.go` direct-LU at coarsest (calls `linalg.LUSolve`) or fall-back-CG-with-tolerance ~80 LOC), **(c) Tier-2 algebraic multigrid ~720 LOC** (G12 `multigrid/amg/strong.go` Ruge-Stüben-1987 strong-connection criterion `|a_{ij}| ≥ θ max_{k≠i} |a_{ik}|` typically θ=0.25 ~80 LOC; G13 `multigrid/amg/cf_split.go` Ruge-Stüben classical C-F splitting first-pass + second-pass ~160 LOC, the load-bearing-data-structure of classical-AMG; G14 `multigrid/amg/interp.go` Ruge-Stüben direct-interpolation + standard-interpolation `(P x)_i = Σ_{j∈C_i} w_{ij} x_j` weights derived from F-row of A ~140 LOC; G15 `multigrid/amg/sa.go` Smoothed-Aggregation Vaněk-Mandel-Brezina-1996-Computing tentative-prolongator-via-aggregation + Jacobi-smoothed-prolongator `P = (I - ω D⁻¹ A) P̃` ~180 LOC; G16 `multigrid/amg/aggregation.go` greedy aggregation of strong-connection graph (the SA equivalent of CF-split) ~80 LOC; G17 `multigrid/amg/cycle.go` AMG-V-cycle wrapper composing G6 with AMG-built-G2-G3-G4 ~80 LOC), **(d) Tier-3 multigrid as preconditioner + Krylov hybrid ~560 LOC** (G18 `multigrid/pcg.go` Preconditioned-CG with multigrid-V-cycle as preconditioner Concus-Golub-O'Leary-1976 — the workhorse of production-PDE-solvers; ~140 LOC, depends on 244-D12 CG; G19 `multigrid/gmres_mg.go` GMRES preconditioned by AMG for non-symmetric / indefinite ~160 LOC, depends on 244-D27; G20 `multigrid/k_cycle.go` Notay-2008-NLAA Krylov-accelerated K-cycle for hard / non-symmetric problems — replaces V-cycle's recursive call with 2-step-Krylov-acceleration ~120 LOC; G21 `multigrid/cascadic.go` Bornemann-Deuflhard-1996-NumerMath cascadic = FMG-without-recursive-V (only nested-iteration + smoother) provably O(N log N) for elliptic ~80 LOC; G22 `multigrid/bpx.go` Bramble-Pasciak-Xu-1990 additive-multilevel-preconditioner ‖u‖² = Σ_k ‖Q_k u - Q_{k-1} u‖_{A_k}² used as ADDITIVE not MULTIPLICATIVE preconditioner for parallel-friendliness ~80 LOC), **(e) Tier-4 frontier ~620 LOC ⊘ DEFER** (G23 `multigrid/p_mg.go` p-multigrid Rønquist-Patera-1987 / Helenbrook-Mavriplis-Atkins-2003 hp-FE p-coarsening Π^{p+1}_p projector — ⊘DEFER blocked-on-245-S22-SEM and 247-M3 Legendre-modal-hp-basis ~140 LOC; G24 `multigrid/hp_mg.go` hp-multigrid Babuška-Suri-1990 h-then-p hierarchy ~100 LOC ⊘DEFER blocked-on-247; G25 `multigrid/spectral_mg.go` spectral-multigrid Zang-Wong-Hussaini-1982 / Heinrichs-1988 — projection between Chebyshev p-levels ~100 LOC ⊘DEFER blocked-on-245; G26 `multigrid/bootstrap_amg.go` Brandt-Brannick-Kahl-Livshits-2011 bootstrap-AMG learning prolongator from candidate-vectors via test-vector-relaxation ~140 LOC ⊘DEFER frontier-research; G27 `multigrid/wavelet_mg.go` Briggs-Henson-McCormick-2000 wavelet-multigrid §10 + Beylkin-Coifman-Rokhlin-1991 ~100 LOC ⊘DEFER; G28 `multigrid/graph_amg.go` Koutis-Miller-Peng-2010-FOCS Spielman-Teng-2004 nearly-linear graph-Laplacian-solver ~140 LOC ⊘DEFER consumer-pull-zero-currently). **SINGULAR-FOUNDATIONAL G1+G2+G3+G6 smoother+restriction+prolongation+V-cycle ~620 LOC** because every single multigrid algorithm — geometric / algebraic / p / hp / spectral / bootstrap / Krylov-accelerated / cascadic / preconditioner — calls these four primitives as substrate. Zero-dep Go absent everywhere. **SINGULAR-MOAT G15 Smoothed-Aggregation-AMG Vaněk-Mandel-Brezina-1996 ~180 LOC** because SA-AMG is the **production-grade modern AMG** (used in Trilinos-ML, Hypre-PyAMG, Aztec-OO, PETSc-GAMG, MOOSE — every major DOE lab solver) and dramatically outperforms classical Ruge-Stüben on systems-of-PDEs (elasticity, magnetohydrodynamics, Stokes) where the near-null-space contains rigid-body-modes / divergence-free-modes that classical-AMG mishandles. **SINGULAR-CUTTING-EDGE G20 K-cycle Notay-2008 ~120 LOC** because K-cycle (Krylov-accelerated multigrid) is the post-2008 robust-AMG state-of-the-art (deployed in AGMG production solver — Notay's commercial AGMG library is the **gold-standard of black-box AMG performance benchmarks**) and zero-dep Go absent. **SINGULAR-PEDAGOGICAL G6+G9 V-cycle + FMG on Poisson-2D ~280 LOC** the entire-pedagogical-entry-point Briggs-Henson-McCormick-2000 *"A Multigrid Tutorial"* §3-§4 (the canonical reference text) — pin against `-Δu = f` on `[0,1]²` with `u = sin(πx)sin(πy)` exact-solution showing iteration-count INDEPENDENT of grid-resolution `h` (the famous **h-independent-convergence** signature of multigrid: 5-cycles → 1e-10 at every h, FD-CG needs O(N^{1/2}) cycles which scales). **SINGULAR-2024-FRONTIER G26 Bootstrap-AMG Brandt-Brannick-Kahl-Livshits-2011 + G28 Spielman-Teng combinatorial-multigrid ~280 LOC ⊘DEFER** because (a) bootstrap-AMG learns the prolongator via test-vector relaxation — works on Helmholtz / Stokes / non-elliptic problems where strong-connection-AMG fails, and (b) Spielman-Teng-2004 nearly-linear-time graph-Laplacian solver is the post-2010 theoretical-frontier-of-iterative-methods that Koutis-Miller-Peng-2010-FOCS made practical (used in graph-partitioning, ML graph-convolutional-networks, electrical-flow problems). Defer both — research-frontier, consumer-pull near-zero in reality today.

Recommended placement **NEW package `multigrid/`** ~2,640 LOC (G1-G22 minus deferred G23-G28). Subpackage layout:

```
multigrid/
  smoother.go         # G1: Jacobi/wJacobi/GS/RBGS/SymGS/Cheb/SOR
  restriction.go      # G2: full-weighting + half-weighting + injection
  prolongation.go     # G3: linear/bilinear/trilinear interp
  galerkin.go         # G4: coarse-grid R A P assembly
  grid_hierarchy.go   # G5: GMG hierarchy 1D/2D/3D
  vcycle.go           # G6: recursive V-cycle keystone
  wcycle.go           # G7: W-cycle
  fcycle.go           # G8: F-cycle
  fmg.go              # G9: Full Multigrid (the O(N) optimal solver)
  twogrid.go          # G10: two-grid analysis helper
  coarsest_solver.go  # G11: direct LU or fall-back CG
  amg/
    strong.go         # G12: Ruge-Stüben strong-connection
    cf_split.go       # G13: classical C-F splitting
    interp.go         # G14: classical AMG direct/standard interpolation
    sa.go             # G15: Smoothed Aggregation (Vaněk-Mandel-Brezina)
    aggregation.go    # G16: greedy aggregation
    cycle.go          # G17: AMG V-cycle wrapper
  pcg.go              # G18: PCG with MG-preconditioner
  gmres_mg.go         # G19: GMRES with AMG-preconditioner
  k_cycle.go          # G20: Notay-2008 K-cycle (Krylov-accel)
  cascadic.go         # G21: Bornemann-Deuflhard cascadic
  bpx.go              # G22: Bramble-Pasciak-Xu additive multilevel
  # DEFER
  # p_mg.go           # G23: p-multigrid (blocked-on-245)
  # hp_mg.go          # G24: hp-multigrid (blocked-on-247)
  # spectral_mg.go    # G25: spectral-multigrid (blocked-on-245)
  # bootstrap_amg.go  # G26: Brandt-Brannick-Kahl-Livshits frontier
  # wavelet_mg.go     # G27: Briggs-Henson-McCormick §10
  # graph_amg.go      # G28: Spielman-Teng / Koutis-Miller-Peng
```

Rationale for **NEW** `multigrid/` rather than nesting under `pde/multigrid/` per 244 layout: multigrid-as-a-discipline is **NOT exclusively a PDE-solver** — algebraic multigrid (AMG) operates on arbitrary sparse SPD matrices coming from FE-Maxwell, FE-elasticity, statistical-Markov-chains, graph-Laplacians, kernel-regression, image-processing variational models (TV-L2 denoising, optical-flow Horn-Schunck), constraint-Hessian Newton-systems in optimisation — far beyond the canonical PDE-class. AMG-on-graph-Laplacian (G28) is a graph-algorithm. Nesting under `pde/` artificially couples a domain-agnostic linear-system-solver-discipline to PDE-specific consumers. Top-level `multigrid/` matches PETSc-PCMG / hypre-BoomerAMG / Trilinos-ML layout conventions where multigrid is its own domain-agnostic library imported by FE / FV / kernel / image / graph / optimisation consumers without circularity. Sub-package precedent inside reality: `prob/copula/`, `prob/conformal/`, `optim/proximal/`, `optim/transport/`. **Single-source-of-truth concern**: smoothers G1 (Jacobi/GS/SOR) duplicate 244-D11 Poisson2D-Jacobi/GS/SOR — recommended migration: 244-D11 narrows to "Poisson-specific assembly + boundary-condition-handling" and imports `multigrid.JacobiSmoother(A, b, x, ...)` for the relaxation kernel. CG (244-D12) and GMRES (244-D27) live in `linalg/` (canonical-home for general Krylov) and `multigrid/pcg.go` + `multigrid/gmres_mg.go` here are *thin wrappers* composing `linalg.CG` + `multigrid.VCycle` as the preconditioner-`M⁻¹`-callback.

**CANDOR.** Multigrid is **the canonical-O(N)-optimal-PDE-solver** (Hackbusch-1985-Thm-7.2.2 for FMG, Brandt-1977 for V-cycle on Poisson) and the one universal-result-everyone-cites in computational PDE. But multigrid is **also** the most-implementation-detail-dependent algorithm in numerical-linear-algebra: the V-cycle works textbook-perfectly on Poisson-2D-uniform-grid but breaks subtly on (a) Helmholtz with `k > h⁻¹` (smoother stops smoothing oscillatory modes), (b) anisotropic-elliptic (line-smoothers needed, point-smoothers fail), (c) saddle-point Stokes (block-smoothers Vanka-1986 needed), (d) convection-diffusion at high Péclet (semi-coarsening + ILU-smoother needed) — every textbook chapter past §4 of Briggs-Henson-McCormick is "...how to fix the V-cycle when it doesn't work". reality should ship the **Tier-0 + Tier-1 + Tier-2 + Tier-3 substrate (G1-G22 ~2,640 LOC over 5 PRs / 12-15 engineer-days)** that solves the canonical-elliptic-PDE class (Poisson, screened-Poisson, mass-stiffness on FE-grid) and the algebraic generalisation to arbitrary-sparse-SPD via AMG, AND ship multigrid-as-CG-preconditioner (the workhorse usage). Defer Tier-4 frontier (p-multigrid, hp-multigrid, spectral-multigrid, bootstrap-AMG, wavelet-MG, Spielman-Teng) — research-frontier with strict-blocking on 245 + 247 not-yet-shipped. **Cheapest-1-day-shippable**: G1 + G2 + G3 + G6 + G11 ~580 LOC ships smoother + restriction + prolongation + V-cycle + coarsest-LU saturating R-MULTIGRID-H-INDEPENDENCE 4/4 against Poisson-2D `-Δu = 2π² sin(πx)sin(πy)` analytical solution `u = sin(πx)sin(πy)` showing 5-V-cycles → 1e-10 at h={1/16, 1/32, 1/64, 1/128} (the famous h-independent-convergence pin Briggs-§3.1). **Highest-leverage-1-week-unlock**: PR-1 + PR-2 + PR-3 ~1,460 LOC G1-G11 saturates GMG keystone + AMG-classical + AMG-SA — first-class O(N) optimal solver for elliptic-PDE on uniform-grid + arbitrary-SPD-sparse via AMG, simultaneously unlocking 244-elliptic-Poisson-large-scale-N>10⁴ + 247-mortar-saddle-point-iterative-solve + 249-Schwarz-multigrid-hybrid. **20 of 26 primitives unique to this slot** (G1 smoothers shared with 244-D11 stationary-iteration; G18 PCG composes with 244-D12 CG; G19 GMRES-MG composes with 244-D27 GMRES; G23 p-MG blocked-on-245-S22+247-M3; G24 hp-MG blocked-on-247; G28 graph-AMG cross-link 097-graph + 246-X12 cot-Laplacian-eigenmodes).

---

## 0. State at HEAD (2026-05-09, v0.10.0) — verified by direct read

Repo-wide audit for multigrid / V-cycle / W-cycle / F-cycle / FMG / AMG / Ruge-Stüben / smoothed-aggregation / Brandt / Hackbusch / Stüben / Vaněk-Mandel-Brezina / Bramble-Pasciak-Xu / Notay K-cycle / cascadic / Briggs-Henson-McCormick / wavelet-multigrid / p-multigrid / hp-multigrid / spectral-multigrid / Koutis-Miller / Spielman-Teng / Schwarz-block-smoother / Vanka-block-smoother / classical-stationary-iteration (Jacobi, weighted-Jacobi, Gauss-Seidel, Red-Black-GS, SOR) / Krylov (CG, PCG, GMRES, BiCGStab) / preconditioner (ILU, Jacobi-precon, MG-precon) / sparse-matrix surface — **zero callable matches** anywhere in `*.go` files outside review-corpus.

| Surface | Path | Multigrid relevance |
|---|---|---|
| `chaos.RK4Step / EulerStep / SolveODE` | `chaos/ode.go` | Deterministic ODE substrate — **NOT** an iterative-linear-solver; no Jacobi/GS/CG framework |
| `linalg.LUSolve / LUDecomposition` | `linalg/decompose.go` | Dense LU O(n³) — usable as **coarsest-grid solver G11** for V-cycle but unsuitable for fine-grid (which is the entire point of multigrid: AVOID O(n³) at fine grids by recursing to coarse); also not sparse |
| `linalg.Cholesky / CholeskySolve` | `linalg/decompose.go` | SPD direct — usable for coarsest-grid SPD case; not sparse |
| `linalg.QRAlgorithm` | `linalg/eigen.go:20` | Symmetric eigensolver — usable for spectral-radius-of-error-propagation `‖S^{ν₂} K_{H←h} S^{ν₁}‖` two-grid analysis (G10) |
| `linalg.MatVecMul / MatMul` | `linalg/matrix.go` | Dense matrix-vector — usable for coarse-grid Galerkin `R A P` (G4) at small N; NOT sparse-Galerkin |
| `optim/interpolate.go::Thomas-tridiag` | private 80-LOC at lines 62-99 | Cubic-spline natural-BC tridiagonal Thomas — **PRIVATE, NOT EXPOSED**; would be coarsest-grid solver for 1D-Poisson |
| `signal.FFT / IFFT` | `signal/fft.go` | Cooley-Tukey radix-2 — diagonalises Laplacian on torus `Δ → -k²` enabling FFT-direct-solve (alternative to multigrid for periodic-BC; see 245); NOT a multigrid primitive |
| `calculus.NumericalGradient` | `calculus/calculus.go` | First-derivative central-difference — NOT a smoother (smoothers are A⁻¹-approximations not derivatives) |
| `infogeo.LaplacianKernel` | `infogeo/mmd.go` | Exp(-‖x-y‖/σ) MMD-kernel — name-collision with differential-Laplacian; not relevant |
| `pde/laplacian.go D3` (slot 244) | not yet shipped | **STRICT-UPSTREAM** — the `A_h` operator multigrid solves |
| `pde/elliptic/poisson_2d.go D11` (slot 244) | not yet shipped | **STRICT-UPSTREAM** — Jacobi/GS/SOR smoother substrate; single-source-recommendation: 248 owns smoother-API, 244 imports |
| `pde/elliptic/cg.go D12` (slot 244) | not yet shipped | **STRICT-UPSTREAM** — base CG; 248-G18 composes V-cycle as preconditioner |
| `linalg/gmres.go D27` (slot 244) | not yet shipped | **STRICT-UPSTREAM** — base GMRES; 248-G19 composes AMG as preconditioner |
| `linalg/sparse/csr.go T1` (slot 097) | not yet shipped | **STRICT-UPSTREAM** — every realistic AMG operates on sparse-CSR; flagged 097-T1 |
| `spectral/sem.go S22` (slot 245) | not yet shipped | **STRICT-UPSTREAM** for G23 p-multigrid; deferred |
| `spectral/poly.go S1` (slot 245) | not yet shipped | **STRICT-UPSTREAM** for G25 spectral-multigrid Chebyshev-projection |
| `fem/legendre_modal.go M3` (slot 247) | not yet shipped | **STRICT-UPSTREAM** for G23+G24 hp-multigrid Π^{p+1}_p projector; deferred |
| `fem/mortar/dual_basis.go M16` (slot 247) | not yet shipped | **STRICT-UPSTREAM** for cross-mesh-prolongation in non-conforming-grid-multigrid |
| `geometry/dec/cot_laplacian.go X12` (slot 246) | not yet shipped | **STRICT-UPSTREAM** for graph-AMG-on-mesh in G28 |
| `multigrid/` package | -- | **ABSENT** — this slot creates |
| Jacobi / wJacobi / Gauss-Seidel / Red-Black-GS / SOR smoothers | -- | **ABSENT** — G1 ships (single-source-owned by 248, imported by 244-D11) |
| Restriction (full-weighting / half-weighting / injection) | -- | **ABSENT** — G2 ships |
| Prolongation (linear / bilinear / trilinear) | -- | **ABSENT** — G3 ships |
| Coarse-grid Galerkin `A_H = R A_h P` | -- | **ABSENT** — G4 ships |
| GMG hierarchy 1D/2D/3D | -- | **ABSENT** — G5 ships |
| V-cycle / W-cycle / F-cycle | -- | **ABSENT** — G6+G7+G8 ship |
| Full Multigrid (FMG) — the O(N) optimal-solver | -- | **ABSENT** — G9 ships |
| Two-grid analysis helper | -- | **ABSENT** — G10 ships |
| Ruge-Stüben classical AMG (strong / CF-split / interp) | -- | **ABSENT** — G12+G13+G14 ship |
| Smoothed-Aggregation AMG Vaněk-Mandel-Brezina-1996 | -- | **ABSENT** — G15+G16 ship (production-grade keystone) |
| Multigrid-preconditioned CG (Concus-Golub-O'Leary-1976) | -- | **ABSENT** — G18 ships |
| GMRES preconditioned by AMG | -- | **ABSENT** — G19 ships |
| K-cycle Krylov-accelerated (Notay-2008) | -- | **ABSENT** — G20 ships |
| Cascadic multigrid (Bornemann-Deuflhard-1996) | -- | **ABSENT** — G21 ships |
| BPX additive-multilevel-preconditioner (Bramble-Pasciak-Xu-1990) | -- | **ABSENT** — G22 ships |
| p-multigrid / hp-multigrid / spectral-multigrid | -- | **ABSENT** — G23-G25 stub; defer blocked-on-245+247 |
| Bootstrap-AMG (Brandt-Brannick-Kahl-Livshits-2011) | -- | **ABSENT** — G26 stub; defer frontier-research |
| Wavelet-multigrid (Briggs §10) | -- | **ABSENT** — G27 stub; defer |
| Graph-AMG / Spielman-Teng / Koutis-Miller-Peng | -- | **ABSENT** — G28 stub; defer |

**Cross-import edges this slot creates.**
- `multigrid → linalg.LUSolve / Cholesky` (PRESENT) — G11 coarsest-grid direct solve.
- `multigrid → linalg.MatVecMul` (PRESENT) — G4 dense Galerkin assembly fallback at small-N.
- `multigrid → linalg/sparse/` (FUTURE 097-T1) — G4 sparse-Galerkin + G15 SA-prolongator-smoothing on CSR.
- `multigrid → linalg/cg.go` (FUTURE 244-D12) — G18 PCG composes CG with V-cycle as preconditioner.
- `multigrid → linalg/gmres.go` (FUTURE 244-D27) — G19 GMRES-MG composes GMRES with AMG.
- `multigrid → linalg.QRAlgorithm` (PRESENT) — G10 two-grid spectral-radius witness.
- `pde/elliptic → multigrid.VCycle / FMG` (FUTURE 244-D11) — single-source-of-truth migration: 244 imports smoother + V-cycle from 248.
- `fem/mortar → multigrid.AMG` (FUTURE 247-M14) — saddle-point GMRES preconditioned by AMG.
- `geometry/dec → multigrid.AMG` (FUTURE 246-X12) — graph-AMG on cot-Laplacian for mesh-Poisson.
- `prob/spde → multigrid.PCG` (FUTURE 242) — implicit-Euler-step on stochastic-heat-equation needs SPD-PCG-MG.
- `optim → multigrid.AMG` (FUTURE) — interior-point KKT-Hessian solves.

**Strict downstream consumers** of `multigrid/` substrate (already specified or planned):
- `pde/elliptic/poisson_2d.go` (244-D11) — narrows to assembly + BC; imports smoother + V-cycle.
- `fem/mortar/lagrange_mult.go` (247-M14) — saddle-point GMRES + AMG-precon.
- `geometry/dec/cot_laplacian.go` (246-X12) — graph-AMG-on-mesh.
- `domdecomp/schwarz.go` (249) — Schwarz-multigrid hybrid (multiplicative-Schwarz + V-cycle = canonical algorithm in deal.II).
- `prob/spde/heat_implicit.go` (242) — implicit-Euler stochastic-heat needs SPD-elliptic-solve.
- `optim/interior_point.go` (future) — KKT-Hessian SPD-AMG.
- `linalg/sparse/spd_solve.go` (097-T2) — SPD-sparse-direct fallback to AMG when N exceeds dense-LU break-even.

---

## 1. Twenty-six primitives (G1-G26, G27-G28 deferred) detailed

**G1 `multigrid/smoother.go` smoothers ~200 LOC.** All operate in-place on `x[]` given `A` (matrix-vector callback `Av func(x, out []float64)` for matrix-free) and `b[]`. **Jacobi**: `x_i^{k+1} = (b_i - Σ_{j≠i} a_{ij} x_j^k) / a_{ii}` (parallel-friendly; needs separate scratch buffer). **Weighted-Jacobi** `ω=2/3` for Brandt-1977-optimal-2D-Laplacian-smoothing — eigenvalues of `(I - ω D⁻¹ A)` for 5-pt Laplacian have spectral-radius `1/3` on highest-modes (the smoothing-rate that makes V-cycle work). **Gauss-Seidel** forward+backward+symmetric: `x_i^{k+1} = (b_i - Σ_{j<i} a_{ij} x_j^{k+1} - Σ_{j>i} a_{ij} x_j^k) / a_{ii}` (in-place; sequential). **Red-Black-GS** Stüben-Trottenberg-1982: 2-colour partition of grid (chequerboard); update reds first, blacks second — parallel-friendly + GS-strength + Jacobi-parallelism. **Chebyshev-polynomial-smoother** Adams-Brezina-Hu-Tuminaro-2003: polynomial `p(D⁻¹A)` of degree-k where Chebyshev-coefficients minimise sup-norm on `[λ_min, λ_max]` of `D⁻¹A` — optimal-smoothing-rate for given polynomial degree, parallel-only matvecs. **SOR** with optimal `ω = 2/(1 + sin(πh))` for 5-pt Laplacian (Frankel-1950) — accelerates Gauss-Seidel by `O(h⁻¹)` factor; obsolete as standalone-iterative-solver but **excellent smoother**. API `multigrid.JacobiSweep(A, b, x, omega, sweeps int, work []float64)`, `multigrid.GaussSeidelSweep(A, b, x, sweeps, direction int)`, `multigrid.RedBlackGSSweep(A, b, x, redMask []bool, sweeps int)`, `multigrid.ChebyshevSmoother(matvec func([]float64, []float64), b, x []float64, lambdaMin, lambdaMax float64, degree int, work [][]float64)`. Refs: Trottenberg-Oosterlee-Schüller-2001 §2.3; Briggs-Henson-McCormick-2000 §2; Brandt-1977 MathComp 31.

**G2 `multigrid/restriction.go` restriction operators ~140 LOC.** **Full-weighting 2D** stencil `[1 2 1; 2 4 2; 1 2 1]/16`: each coarse-cell receives weighted average of 9 neighbouring fine-cells. **Half-weighting 2D** `[0 1 0; 1 4 1; 0 1 0]/8` (Brandt-1977 §5.2): cheaper, sometimes sufficient. **Injection**: `r_H[i,j] = r_h[2i, 2j]` (only valid at coarsest, biased on smooth modes — DO NOT use for production cycles). **1D linear restriction** stencil `[1 2 1]/4`. **3D trilinear restriction** stencil `[1 2 1; 2 4 2; 1 2 1; 2 4 2; 4 8 4; 2 4 2; 1 2 1; 2 4 2; 1 2 1]/64`. **Galerkin-projection-consistency**: must satisfy `R = c P^T` for variational consistency `(R u_h, v_H)_H = (u_h, P v_H)_h` — full-weighting `c = 1/4` in 2D, `c = 1/8` in 3D. API `multigrid.FullWeighting2D(rh []float64, rH []float64, nh, nH int)` zero-alloc.

**G3 `multigrid/prolongation.go` prolongation operators ~140 LOC.** **Linear-1D**: `u_h[2i] = u_H[i]` direct-injection at coarse-points; `u_h[2i+1] = (u_H[i] + u_H[i+1])/2` linear-interp at fine-points. **Bilinear-2D**: extend tensorially. **Trilinear-3D**. **Variational-consistency** with full-weighting-restriction: `R = (1/4) P^T` in 2D; ensures coarse-grid Galerkin `A_H = R A_h P` is well-defined. API `multigrid.BilinearProlong2D(uH []float64, uh []float64, nH, nh int)`. Note: prolongation operates on **corrections** not solutions — `u_h ← u_h + P e_H`.

**G4 `multigrid/galerkin.go` coarse-grid assembly ~80 LOC.** Two strategies: **Galerkin coarse-grid** `A_H = R A_h P` — automatically variationally-consistent regardless of `A_h` operator; required for AMG (G15-G17) where there is no underlying-PDE structure to rediscretise; cost = sparse-matrix-triple-product. **Operator-rediscretisation** `A_H = h_H²-rescaled-stencil` — cheaper at small N but requires PDE-structure-knowledge; strictly only for GMG with regular grid hierarchies. Rule-of-thumb: GMG = rediscretise (cheaper); AMG = Galerkin (mandatory). API `multigrid.GalerkinCoarsen(A_h_matvec func(in, out []float64), R, P []float64, A_H []float64, nh, nH int)` dense-fallback; `multigrid.GalerkinCoarsenSparse(A_h *sparse.CSR, R, P *sparse.CSR, out *sparse.CSR)` blocked-on-097-T1.

**G5 `multigrid/grid_hierarchy.go` GMG hierarchy ~80 LOC.** `Hierarchy{ levels int, n []int, A []*Operator, R []*Operator, P []*Operator }` storing per-level grid-size + operators. Standard coarsening factor 2 in each spatial dimension (fine-grid `N=2^L+1` vertices supports `L` levels). 1D / 2D / 3D variants. Cell-centred vs vertex-centred coarsening differ at boundary; vertex-centred easier for Dirichlet, cell-centred easier for Neumann.

**G6 `multigrid/vcycle.go` V-cycle ~140 LOC.** **The Brandt-1977 keystone algorithm**:
```
function VCycle(A_h, b, x, level):
    if level == coarsest: x = LUSolve(A_h, b); return x
    x = Smooth(A_h, b, x, ν₁)              # pre-smooth
    r_h = b - A_h x                         # residual
    r_H = Restrict(r_h)                     # restrict to coarse
    e_H = VCycle(A_H, r_H, 0, level+1)      # recurse
    x = x + Prolong(e_H)                    # correct
    x = Smooth(A_h, b, x, ν₂)              # post-smooth
    return x
```
Typical `ν₁ = ν₂ = 2` (Briggs-§3.1). **Spectral-radius bound**: for Poisson-2D + weighted-Jacobi `ω=2/3`, two-grid spectral-radius ≤ 1/3 INDEPENDENT of h — V-cycle inherits this (h-independent-convergence). API `multigrid.VCycle(hierarchy *Hierarchy, b, x []float64, nu1, nu2 int)` recursive.

**G7 `multigrid/wcycle.go` W-cycle ~80 LOC.** Same as V-cycle but recurses **twice** at each non-coarsest level. Cost per cycle: O(N) (geometric sum still converges) but with constant ~2× V-cycle. More robust for: non-symmetric / indefinite operators, jumping-coefficient elliptic, anisotropic problems. Hackbusch-1985-§6.4.

**G8 `multigrid/fcycle.go` F-cycle ~60 LOC.** Recurses 1× at finest, 2× at next-coarse, 1× at next-coarse, 2× at next-coarse, ... — interpolates V (cheap, weak) and W (expensive, robust). Trottenberg-Oosterlee-Schuller-2001 §2.4.

**G9 `multigrid/fmg.go` Full Multigrid ~140 LOC.** **The optimal-O(N)-PDE-solver result**:
```
function FMG(A_finest, b_finest):
    Restrict b_finest → b_coarsest through all levels
    x_coarsest = LUSolve(A_coarsest, b_coarsest)
    for level = coarsest+1 .. finest:
        x_level = ProlongFMG(x_{level-1})  # FMG-prolong (higher-order than VCycle-prolong)
        x_level = VCycle(A_level, b_level, x_level, ν)
    return x_finest
```
Hackbusch-1985-Thm-7.2.2: FMG converges to `‖discretisation-error‖_h` in **O(1) cycles per level** = **O(N) total operations** — provably optimal-asymptotic-PDE-solver. The reason multigrid is *the* canonical-result of computational-PDE. **FMG-prolongation** uses higher-order interpolation (cubic) than V-cycle's bilinear-prolong because we prolong solutions (where smoothness matters) not corrections (where machine-precision-prolong is enough). API `multigrid.FullMultigrid(hierarchy *Hierarchy, b []float64, x []float64, nu int)`.

**G10 `multigrid/twogrid.go` two-grid analysis ~80 LOC.** Helper computing two-grid error-propagation operator `K = S^{ν₂} (I - P A_H⁻¹ R A_h) S^{ν₁}` and its spectral-radius via `linalg.QRAlgorithm`. Diagnostic-tool not production-solver. Validates that V-cycle convergence-factor `ρ(K) < 1`. Briggs-§4.4.

**G11 `multigrid/coarsest_solver.go` coarsest-grid solver ~80 LOC.** At coarsest level (typically `N ≤ 100`), solve directly via `linalg.LUSolve` (general) or `linalg.CholeskySolve` (SPD). Alternatively iterative-CG-to-tight-tolerance for large coarsest-grid (sub-optimal — make hierarchy deeper instead). API `multigrid.CoarsestSolveLU(A []float64, b, x []float64, n int)`.

**G12 `multigrid/amg/strong.go` Ruge-Stüben strong-connection ~80 LOC.** **The first-step of classical AMG**: define for each row `i` set of strong-connections `S_i = {j ≠ i : -a_{ij} ≥ θ max_{k≠i, a_{ik}<0} (-a_{ik})}` typically `θ = 0.25`. Strong-connection-graph drives C-F-splitting + interpolation. Ruge-Stüben-1987-MultigridMethods-Frontiers §5.

**G13 `multigrid/amg/cf_split.go` Ruge-Stüben C-F splitting ~160 LOC.** **The data-structure-defining step of classical-AMG.** Two-pass algorithm: (a) **First-pass** — greedy: choose `i` with largest `|S_i^T|` (most-strongly-influences-others) as C-point; mark all `j ∈ S_i^T` as F-points; repeat on remaining undetermined. (b) **Second-pass** — fix interpolation-quality: each F-point must have ≥ 1 C-point in its strong-connection-set, else add a C-point. Output: `C[]bool` and `F[]bool` partitioning indices `0..n-1`. Ruge-Stüben-1987 §5.2.

**G14 `multigrid/amg/interp.go` classical AMG interpolation ~140 LOC.** **Direct interpolation**: for F-point `i`, `(P x)_i = Σ_{j ∈ C_i} w_{ij} x_j` with `w_{ij} = -a_{ij} / (Σ_{k ∈ C_i} a_{ik})` — preserves row-sum (constant-vector-preservation, the load-bearing property for elliptic). **Standard interpolation**: extends to handle F-F strong connections by symmetrising. API `multigrid.RugeStubenInterp(A *sparse.CSR, C, F []bool, P *sparse.CSR)`.

**G15 `multigrid/amg/sa.go` Smoothed Aggregation ~180 LOC.** **The production-grade modern AMG** (Vaněk-Mandel-Brezina-1996-Computing-56). (a) **Aggregation**: greedy partition of strong-connection-graph into disjoint aggregates (G16). (b) **Tentative prolongator** `P̃`: each aggregate gets one column of `P̃`; entries are 1 inside aggregate, 0 outside (or, for systems-of-PDEs, the rigid-body / null-space modes restricted to aggregate). (c) **Jacobi-smoothed prolongator** `P = (I - ω D⁻¹ A) P̃` with `ω = 4/(3 ρ(D⁻¹A))` — improves interpolation-accuracy (textbook-AMG-with-tentative-P-only fails on Poisson; smoothed-P is essential). The `ω` choice ensures smoother is contractive on prolongator-range. **Why SA beats classical**: handles systems-of-PDEs (elasticity 3-DOF/node, Stokes velocity-pressure, MHD) where classical-AMG-on-block-system fails because near-null-space contains rigid-body-modes that strong-connection-criterion misidentifies. Used in Trilinos-ML, hypre-PyAMG, Aztec-OO, PETSc-GAMG, MOOSE.

**G16 `multigrid/amg/aggregation.go` greedy aggregation ~80 LOC.** Simple greedy: pick uncoloured node `i` with largest unaggregated-neighbourhood, form aggregate `{i} ∪ {strong-neighbours of i not yet aggregated}`, mark coloured. Repeat. Final pass: assign isolated F-nodes to nearest aggregate. Average aggregate-size = 6-8 (2D) or 27-30 (3D) typical. Vaněk-Mandel-Brezina-1996 §3.

**G17 `multigrid/amg/cycle.go` AMG V-cycle wrapper ~80 LOC.** Compose G6 V-cycle with AMG-built-G2 (= sparse `R = P^T`), AMG-built-G3 (= `P` from G14 or G15), AMG-built-G4 (= sparse Galerkin `R A_h P`), AMG-built-G11 (LU at coarsest where `n_coarsest < 100`). Setup-phase (G12-G16) is amortised across many solves of the same matrix.

**G18 `multigrid/pcg.go` MG-preconditioned CG ~140 LOC.** Concus-Golub-O'Leary-1976-SISC: for SPD `A`, preconditioner `M⁻¹` symmetric positive-definite, run CG on `M⁻¹ A x = M⁻¹ b`. With `M⁻¹ = VCycle(A, ·, 0)` (V-cycle as an SPD-preconditioner — note: V-cycle with symmetric-GS or weighted-Jacobi smoother + Galerkin coarse-grid IS SPD), PCG inherits CG's monotone-A-norm-error-decrease + V-cycle's h-independent-spectral-equivalence: convergence-factor `(√κ - 1)/(√κ + 1)` with `κ = O(1)` independent of h. **The workhorse production-PDE-solver** for elliptic problems where the matrix is SPD but the problem is hard enough that pure-V-cycle's geometric convergence has too-large constant. API `multigrid.PCGWithMG(A_matvec, b, x []float64, hierarchy *Hierarchy, tol float64, maxiter int)`.

**G19 `multigrid/gmres_mg.go` GMRES preconditioned by AMG ~160 LOC.** Saad-Schultz-1986-GMRES + AMG-V-cycle as right-preconditioner: `A M⁻¹ y = b`, recover `x = M⁻¹ y`. For non-symmetric / indefinite — convection-diffusion at high Pé, advective-Stokes, Helmholtz. Restart parameter `m=20-50`. Note: V-cycle as right-preconditioner does NOT need to be symmetric; can use forward-GS-only smoother. Depends on 244-D27 GMRES.

**G20 `multigrid/k_cycle.go` Notay K-cycle ~120 LOC.** Notay-2008-NumerLinAlgAppl-15: replace V-cycle's recursive call to `VCycle(A_H, r_H)` with `KCycle(A_H, r_H)` defined as 1-2 steps of Krylov-acceleration (CG or flexible-GMRES) preconditioned by V-cycle one-level-deeper. **Why**: V-cycle assumes coarse-grid-correction reduces error by factor `c < 1`; if c ≈ 0.7 (degraded for hard problems), V-cycle stalls. K-cycle's Krylov-step makes coarse-grid-correction *optimal-in-Krylov-space* — robust against any `c < 1`. The post-2008 black-box-AMG-state-of-the-art (Notay's commercial AGMG library uses this).

**G21 `multigrid/cascadic.go` cascadic multigrid ~80 LOC.** Bornemann-Deuflhard-1996-NumerMath-75: FMG-without-the-recursive-V-cycle; only nested-iteration + smoothing at each level. Provably O(N log N) for elliptic. Cheap; works when full V-cycle is overkill.

**G22 `multigrid/bpx.go` BPX additive multilevel preconditioner ~80 LOC.** Bramble-Pasciak-Xu-1990-MathComp-55: `‖u‖²_{BPX} = Σ_k Σ_i |⟨u, ψ_{k,i}⟩|²` with `ψ_{k,i}` the level-k nodal-basis. Used as ADDITIVE preconditioner `M⁻¹ = Σ_k P_k A_k⁻¹ R_k` — parallelisable across levels (V-cycle is multiplicative-sequential). Spectrally-equivalent to A on H¹.

**G23-G28 deferred ⊘**: see Tier-4 in Summary.

---

## 2. Roadmap (5 PRs, ~12-15 engineer-days, ~2,640 LOC for G1-G22)

| PR | Slug | LOC | Days | Saturates | Unblocks |
|---|---|---|---|---|---|
| PR-1 | mg-substrate | 580 | 2.5 | G1+G2+G3+G6+G11 + R-MULTIGRID-H-INDEPENDENCE 4/4 | 244-D11 single-source migration |
| PR-2 | mg-cycles | 360 | 1.5 | G4+G5+G7+G8+G9+G10 + R-FMG-OPTIMALITY 4/4 | 242-spde-implicit-step |
| PR-3 | amg-classical | 460 | 3 | G12+G13+G14+G17 + R-AMG-RS-CONVERGENCE 3/3 | 097-graph-Laplacian (cross-link) |
| PR-4 | amg-sa | 260 | 2 | G15+G16 + R-AMG-SA-ELASTICITY 3/3 | 247-mortar-saddle-point |
| PR-5 | mg-precon | 540 | 3 | G18+G19+G20+G21+G22 + R-PCG-MG-CONVERGENCE 4/4 | 244-CG/GMRES upgrade |

**Cheapest-1-day-shippable**: PR-1 alone (580 LOC) saturates smoother + restriction + prolongation + V-cycle + coarsest-LU on Poisson-2D — the canonical Briggs-§3.1 demonstration.

**Recommended ship-order**: PR-1 → PR-2 → PR-3 → PR-5 → PR-4 (PR-5 before PR-4 because PCG-MG is more pedagogically-canonical than SA-AMG; SA needs more validation on systems-of-PDEs which require 244-elasticity-FE not yet shipped).

## 3. Golden-file pins

- **R-MULTIGRID-H-INDEPENDENCE 4/4**: Poisson-2D `-Δu = 2π² sin(πx)sin(πy)` analytical `u = sin(πx)sin(πy)` on `[0,1]²` Dirichlet — V-cycle with weighted-Jacobi `ω=2/3` ν₁=ν₂=2 converges to 1e-10 in **5 cycles at every** h ∈ {1/16, 1/32, 1/64, 1/128} (the famous h-independent-convergence pin Briggs-Henson-McCormick-2000-§3.1).
- **R-FMG-OPTIMALITY 4/4**: Same Poisson-2D — FMG-V(2,2) achieves discretisation-error `O(h²)` in **single FMG-pass** (1 V-cycle per level after prolongation) at all h (Hackbusch-1985-Thm-7.2.2 closed-form pin); operations-count ≤ 30N proportional to fine-grid DOFs.
- **R-AMG-RS-CONVERGENCE 3/3**: classical Ruge-Stüben on 5-pt Laplacian sparse-matrix (no geometric structure exposed) — convergence-factor ≤ 0.15 per V-cycle for n={64², 128², 256²}.
- **R-AMG-SA-ELASTICITY 3/3**: SA-AMG on plane-strain elasticity 2D (3-DOF-per-node block-system) — converge to 1e-8 in ≤ 15 V-cycles independent of mesh-h — the systems-of-PDEs validation classical-AMG fails. Blocked on 244-D26-P1-2D + elasticity-assembly downstream.
- **R-PCG-MG-CONVERGENCE 4/4**: Poisson-2D + jumping-coefficient `α(x) ∈ {1, 1000}` checkerboard — pure-V-cycle convergence-factor degrades to 0.4; PCG-MG recovers ≤ 0.05 effective-rate; iterations ≤ 12 to 1e-10 at all h.
- **R-K-CYCLE-NOTAY 3/3**: Helmholtz-screened `(Δ + k²) u = f` k=10 — V-cycle stalls (factor > 0.9); K-cycle converges (factor ≤ 0.3). Tests robustness-on-indefinite.
- **R-MG-SMOOTHER-RATE 4/4**: smoothing-rate of weighted-Jacobi `ω=2/3` on 5-pt-Laplacian highest-frequency modes ≤ 1/3 (Brandt-1977 closed-form spectral analysis).

## 4. Cross-slot dependencies

**Strict-upstream dependencies for slot 248:**
- 097-T1 SparseMatrix CSR — without this, G15 SA-prolongator-smoothing `(I - ω D⁻¹A) P̃` is dense and untenable beyond N=10³.
- 244-D3 Laplacian-stencil — defines `A_h` for canonical-Poisson-MG demo; D11 Poisson2D-Jacobi/GS/SOR — single-source-migration: 244 imports smoother from 248.
- 244-D12 ConjugateGradient — G18 PCG composes; G18 depends.
- 244-D27 GMRES — G19 GMRES-MG composes; G19 depends.
- 245-S22 SEM (DEFERRED inside 245) — G23 p-multigrid blocks.
- 247-M3 hierarchical Legendre-modal-hp-basis — G23+G24 hp-multigrid blocks.

**Strict-downstream consumers of slot 248:**
- 244-D11 Poisson2D large-N solve — replaces dense-LU with PCG-MG once N>10⁴ (otherwise dense-Cholesky blows past 1GB at N=64² on diagonal-block dense).
- 247-M14 mortar-saddle-point — GMRES + AMG-precon for indefinite KKT.
- 249-domain-decomp Schwarz — Schwarz-multigrid hybrid is the canonical production-method.
- 242-spde implicit-Euler step — SPD-elliptic at each timestep; PCG-MG.
- 246-X12 cot-Laplacian-on-mesh — graph-AMG (deferred).
- aicore future-physics-engine elastic-collision-PDE substrate.

**Cross-link not strict-dependency:**
- 215-compressed-sensing — multilevel-CS-recovery shares prolongation-as-frame-expansion concept.
- 237-GP-regression — kernel-multigrid for hierarchical-matrix scaling.
- 097-graph-Laplacian-AMG — Spielman-Teng-2004 = graph-algorithm-form-of-AMG.
- 156-persistent-homology — non-related.

**Strict-twin / partial-overlap:**
- 244-PDE-solvers — owns Jacobi/GS/SOR-as-PDE-iterations; 248 owns them as smoothers. Single-source-of-truth: 248-`multigrid/smoother.go` is canonical home, 244-D11 imports `multigrid.JacobiSweep` etc.
- 247-mortar — M16 dual-Lagrange basis enables cross-mesh-prolongation for non-conforming-multigrid (advanced; deferred).
- 245-spectral — S22 SEM enables p-multigrid (deferred).

**Cross-cuts existing single-source-of-truth concerns:**
- `linalg.LUSolve` — used as G11 coarsest-solver; OK.
- Smoother-API in 248 vs PDE-iteration-API in 244 — recommend single-canonical-home in 248, 244 imports.
- CG/GMRES homes in `linalg/` (244-D12 + 244-D27) — `multigrid/pcg.go` + `multigrid/gmres_mg.go` are *thin compositions* not duplications.

## 5. Performance / no-allocations-in-hot-paths

Reality's key-design-rule #3: "No allocations in hot paths. Pistachio calls these at 60 FPS." For multigrid V-cycle at N=128²=16384:
- G1 smoothers: in-place on `x[]`; weighted-Jacobi needs scratch (caller-provided); Red-Black-GS in-place.
- G2 restriction / G3 prolongation: caller pre-allocates `rH[], rh[]` per-level; zero-alloc inside.
- G6 V-cycle: caller allocates `Hierarchy` once (level-0..L data); recursive call uses level-k scratch.
- G15 SA-prolongator-smoothing `(I - ω D⁻¹A) P̃`: setup-phase only (one-time per-matrix); not in hot-path.
- G18 PCG-MG: scalar-products O(N) per-iteration; preconditioner-application is one V-cycle ~30N ops; total ~50N per CG iteration (constant-factor faster than dense matvec which is N²).

## 6. References (canonical)

- Brandt, "Multi-level adaptive solutions to boundary-value problems," **MathComp** 31, 1977. The keystone-paper.
- Hackbusch, **Multi-Grid Methods and Applications**, Springer 1985. The keystone-textbook.
- Briggs-Henson-McCormick, **A Multigrid Tutorial**, 2nd-ed SIAM 2000. The canonical-pedagogical-text; entry-point-of-the-discipline.
- Trottenberg-Oosterlee-Schüller, **Multigrid**, Academic Press 2001. The 600-page-comprehensive-reference.
- Ruge-Stüben, "Algebraic multigrid," in **Multigrid Methods (Frontiers in Applied Math 3)**, SIAM 1987. Classical AMG canonical reference.
- Vaněk-Mandel-Brezina, "Algebraic multigrid by smoothed aggregation for second and fourth order elliptic problems," **Computing** 56, 1996. Smoothed-Aggregation keystone.
- Stüben, "A review of algebraic multigrid," **JCAM** 128, 2001.
- Notay, "An aggregation-based algebraic multigrid method," **ETNA** 37, 2010 + **NLAA** 15, 2008. AGMG / K-cycle.
- Bramble-Pasciak-Xu, "Parallel multilevel preconditioners," **MathComp** 55, 1990. BPX additive-multilevel.
- Bornemann-Deuflhard, "The cascadic multigrid method for elliptic problems," **NumerMath** 75, 1996.
- Concus-Golub-O'Leary, "A generalized conjugate gradient method for the numerical solution of elliptic partial differential equations," in **Sparse Matrix Computations**, Academic Press 1976.
- Frankel, "Convergence rates of iterative treatments of partial differential equations," **MTAC** 4, 1950. SOR-optimal-omega.
- Adams-Brezina-Hu-Tuminaro, "Parallel multigrid smoothing: polynomial versus Gauss-Seidel," **JCompPhys** 188, 2003.
- Brandt-Brannick-Kahl-Livshits, "Bootstrap AMG," **SISC** 33, 2011.
- Spielman-Teng, "Nearly-linear time algorithms for graph partitioning, graph sparsification, and solving linear systems," **STOC** 2004.
- Koutis-Miller-Peng, "Approaching optimality for solving SDD linear systems," **FOCS** 2010.
- Helenbrook-Mavriplis-Atkins, "Analysis of p-multigrid for continuous and discontinuous finite element discretizations," **AIAA** 2003-3989. p-multigrid.
- Babuška-Suri, "The hp version of the finite element method with quasi-uniform meshes," **M2AN** 21, 1987. hp-multigrid setup.
- Zang-Wong-Hussaini, "Spectral multigrid methods for elliptic equations," **JCompPhys** 48, 1982. Spectral-multigrid.
- Beylkin-Coifman-Rokhlin, "Fast wavelet transforms and numerical algorithms I," **CommPureApplMath** 44, 1991. Wavelet-multigrid.

---

End of agent 248 review. ~340 lines.

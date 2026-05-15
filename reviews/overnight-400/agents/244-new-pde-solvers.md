# 244 | new-pde-solvers — PDE finite-difference / FEM scaffolding (1D-3D, BCs)

**Summary line 1.** reality v0.10.0 ships **ZERO** PDE-solver surface — repo-wide grep on `finite.difference|Laplacian.(stencil|operator)|wave.equation|heat.equation|poisson|helmholtz|advection|crank.nicolson|cfl|von.neumann.stability|dirichlet|neumann|robin|periodic.bc|stencil|upwind|godunov|riemann|hll|roe|tvd|muscl|weno|finite.volume|finite.element|galerkin|stiffness.matrix|mass.matrix|fem|fvm|gmres|bicgstab|conjugate.gradient|krylov|multigrid|method.of.lines|adaptive.mesh|amr|yee|stagger.grid|p1.element|p2.element|spectral.element` against `*.go` returns **zero callable matches** (only nominal hits are `infogeo.LaplacianKernel` for the exp(-||x-y||/σ) MMD-kernel — name-collision with the differential operator, `gametheory/nash.go` `von Neumann` for the 1928 minimax theorem — name-collision with the 1947 stability analysis, `chaos/systems.go` line 148 "Boundary conditions: the grid wraps around (torus topology)" — for the Lorenz attractor not for PDE BCs, `autodiff/doc.go` mentioning *finite differences* as the alternative to autodiff not as a PDE method, `infogeo/mmd.go` `LaplacianKernel`, `prob/*` Poisson distribution / Poisson process — name-collision with the Poisson PDE, and `physics/thermo.go` heat-capacity / heat-flux scalar formulas with no spatial discretisation). Closest substrate is `chaos/ode.go` (RK4Step + EulerStep + SolveODE on `dy/dt = f(t, y)` — pure deterministic-ODE engine, no spatial-discretisation surface, no PDE Laplacian, no boundary-condition vocabulary), `signal/fft.go` (Cooley-Tukey radix-2 in-place real FFT — gates pseudo-spectral / Fourier-Galerkin PDE solvers on torus; STRICT upstream for spectral methods owned by 245), `linalg/decompose.go` (LU + QR + Cholesky banded — usable but not specialised; tridiagonal-Thomas O(n) absent — every implicit-Euler / Crank-Nicolson timestep degrades to O(n²) general LU), `linalg/eigen.go` (Householder→TQL symmetric-tridiag eigensolver — gates spectral PDE eigenfunction expansions), `calculus.NumericalDerivative / NumericalGradient` (central-difference 1D/multidim — first-derivative only, no Laplacian / no boundary conditions), `optim/proximal/` (Moreau-Yosida + ISTA/FISTA — usable for variational-PDE TV-regularisation but no PDE-solver surface). **MASTER_PLAN slot 244 is named "PDE finite-difference / FEM scaffolding: 1D-3D, boundary conditions"** and is **STRICT UPSTREAM** of FOUR other Block-C reviews already enumerating their PDE-substrate dependency on this slot: 159-synergy-em-signal **W1+W2** Wave1DFDTD+CFL ~120 LOC (calls "the first PDE solver in the entire repo"), 192-synergy-fluids-control **F-family** ROM-based-flow-control via reduced-order PDE-projection, 219-new-mean-field-games **M1 HJB-FD upwind-Hamiltonian + M2 FP-positivity-preserving ~360 LOC** (named `chaos/pde/upwind.go` substrate), 242-new-spde **P0c Laplacian1D ~80 LOC** (explicit "either ship in slot 244 or inline here"), 243-new-fpe **F0a-F0c grid + upwind + Laplacian + tridiag-Thomas ~440 LOC** (explicit single-source ownership recommendation: F0a-F0c live in `pde/`, slot 244 owns), 245-new-spectral-methods strict-twin (spectral methods are an alternative discretisation; 244 owns FD/FV/FEM, 245 owns Chebyshev/Fourier/RBF/pseudo-spectral), 247-new-mortar-fem strict-downstream (mortar/non-conforming FEM consumes basic FEM substrate), 248-new-multigrid strict-downstream (multigrid is a SOLVER for FD/FE-Laplacian systems built here), 249-new-domain-decomp strict-downstream, 251-new-shape-opt strict-downstream (level-set + phase-field PDEs).

**Summary line 2.** Twenty-eight PDE-substrate primitives **D1-D28** totalling ~4,200 LOC organized as **(a) Tier-0 grid+stencil substrate ~620 LOC** (D1 `pde/grid.go` UniformGrid1D/2D/3D + ghost-cells ~140 LOC, D2 `pde/bc.go` Dirichlet/Neumann/Robin/Periodic/Reflecting/Absorbing ~120 LOC, D3 `pde/laplacian.go` 1D/2D/3D 3-pt/5-pt/9-pt/7-pt/19-pt/27-pt + 4th-order/6th-order high-order stencils ~200 LOC, D4 `pde/divergence.go` D5 `pde/gradient.go` div+grad+curl FD operators ~80 LOC, D6 `linalg/tridiag.go` Thomas O(n) tridiag-solver ~80 LOC), **(b) Tier-1 keystone parabolic+elliptic ~1,100 LOC** (D7 HeatEquation1D explicit-Euler+CFL ~140 LOC, D8 HeatEquation1D-implicit Crank-Nicolson ~160 LOC, D9 HeatEquation2D ADI Peaceman-Rachford ~220 LOC, D10 PoissonEquation1D-Dirichlet tridiag-direct ~100 LOC, D11 PoissonEquation2D 5-point Jacobi/Gauss-Seidel/SOR ~180 LOC, D12 ConjugateGradient SPD-Krylov ~140 LOC, D13 vonNeumannStability symbolic stability-analysis helper ~80 LOC, D14 CFLCondition explicit-step-bound helper ~30 LOC, D15 MethodOfLines time-stepper-coupling-to-chaos.RK4 ~50 LOC), **(c) Tier-2 hyperbolic+advection ~1,180 LOC** (D16 WaveEquation1D-FDTD second-order leapfrog ~140 LOC, D17 WaveEquation2D-FDTD ~180 LOC, D18 AdvectionEquation1D upwind/Lax-Friedrichs/Lax-Wendroff ~160 LOC, D19 AdvectionDiffusionReaction1D operator-splitting ~140 LOC, D20 RiemannSolverHLL Harten-Lax-vanLeer ~140 LOC, D21 RiemannSolverHLLC contact-restoring ~160 LOC, D22 GodunovScheme finite-volume conservation-law ~140 LOC, D23 MUSCL-Hancock 2nd-order TVD slope-limiter ~120 LOC), **(d) Tier-3 high-order+FE ~840 LOC** (D24 WENO5-Jiang-Shu 5th-order non-oscillatory ~180 LOC, D25 FiniteElementP1-1D mass+stiffness assembly + Dirichlet ~220 LOC, D26 FiniteElementP1-2D triangular linear elements + Galerkin assembly ~280 LOC, D27 GMRES Saad-Schultz-1986 nonsymmetric Krylov ~160 LOC, D28 BiCGStab vanderVorst-1992 ~140 LOC), **(e) Tier-3.5 frontier+defer ~460 LOC** (D29 YeeGridEM 3D staggered FDTD-Maxwell ~200 LOC ⊘ DEFER blocked-on-em-vec3, D30 SpectralElementMethod hp-adaptive Patera-1984 ~120 LOC ⊘ DEFER blocked-on-245, D31 DG-Discontinuous-Galerkin Hesthaven-Warburton-2008 ~140 LOC ⊘ DEFER, D32 AMR-Berger-Oliger-1984 quadtree-refinement ~280 LOC ⊘ DEFER, D33 FastMultipoleMethod Greengard-Rokhlin-1987 N-body ~200 LOC ⊘ DEFER blocked-on-274). **SINGULAR-FOUNDATIONAL D1+D2+D3 grid+BC+Laplacian-stencil ~460 LOC** because this is **THE** load-bearing substrate that **EIGHT downstream Block-C reviews explicitly name**: 159-W1 (Wave1DFDTD), 219-M1 (HJB upwind), 219-M2 (FP positivity), 242-P0c (Laplacian1D), 243-F0a-F0c (grid+upwind+Laplacian), 247 (mortar-FEM), 248 (multigrid), 251 (level-set+phase-field). Single 460-LOC commit unblocks ~3,000+ LOC of downstream reviews already specified in agents/. **SINGULAR-MOAT D9 ADI-Peaceman-Rachford-1955 ~220 LOC** because Peaceman-Rachford alternating-direction-implicit is the canonical 2D-parabolic-PDE scheme (every plasma physics / atmospheric-modeling / petroleum-reservoir-simulation code uses it; Wachspress-Habetler-1960 generalisation), unconditionally stable and second-order in time, and no zero-dep Go library ships it (PETSc has it via TS module — 100k+ LOC of MPI/parallel infrastructure). **SINGULAR-PEDAGOGICAL D7 HeatEquation1D-explicit ~140 LOC** because heat-equation is the entire-pedagogical-entry-point of PDE numerics (Strikwerda-2004 §4.1, LeVeque-2007 §1) and the closed-form Gaussian initial-condition `u(x, t) = (4πt)^{-1/2} exp(-x²/(4t))` is the ideal cross-language R-pin against analytical-reference. **SINGULAR-CUTTING-EDGE D24 WENO5-Jiang-Shu-1996 ~180 LOC** because Jiang-Shu-1996-JCompPhys-126 fifth-order weighted-essentially-non-oscillatory is the *production-standard* high-order shock-capturing scheme, used by every modern compressible-flow / astrophysics-MHD / climate-model code (Weinan-E group + Shu group standard textbook reference Shu-2009-Acta-Numerica-18) and zero-dependency Go absent. **SINGULAR-2024-FRONTIER D31 Discontinuous-Galerkin Hesthaven-Warburton-2008 ~140 LOC ⊘ DEFER** because DG is the post-2010-research-frontier high-order method (used in Nek5000, MFEM, deal.II); defer because consumer-demand for DG inside reality is research-only.

Recommended placement **NEW package `pde/`** ~3,800 LOC (D1-D28 minus deferred) + `linalg/tridiag.go` ~80 LOC. Subpackage layout:

```
pde/
  grid.go        # D1: UniformGrid 1D/2D/3D + ghost-cells
  bc.go          # D2: Dirichlet/Neumann/Robin/Periodic/Reflecting/Absorbing
  laplacian.go   # D3: 3-pt/5-pt/7-pt/9-pt/19-pt/27-pt 2nd/4th/6th-order
  divgrad.go     # D4+D5: div, grad, curl FD operators
  cfl.go         # D14: CFL helpers
  vneumann.go    # D13: von Neumann analysis helper
  parabolic/
    heat_explicit.go    # D7
    heat_implicit.go    # D8 Crank-Nicolson
    heat_2d_adi.go      # D9 Peaceman-Rachford
  elliptic/
    poisson_1d.go       # D10
    poisson_2d.go       # D11 Jacobi/GS/SOR
    cg.go               # D12 (also at linalg/cg.go shared)
  hyperbolic/
    wave_1d.go          # D16
    wave_2d.go          # D17
    advection.go        # D18 upwind/LF/LW
    adr.go              # D19
    riemann_hll.go      # D20
    riemann_hllc.go     # D21
    godunov.go          # D22
    muscl.go            # D23
    weno.go             # D24
  fem/
    p1_1d.go            # D25
    p1_2d.go            # D26
  mol.go         # D15 method-of-lines wrapper around chaos.RK4Step
linalg/
  tridiag.go     # D6 Thomas O(n)
  cg.go          # D12 (canonical home; pde/elliptic re-exports)
  gmres.go       # D27 Saad-Schultz-1986
  bicgstab.go    # D28 vanderVorst-1992
```

Rationale: PDE-solvers form a **self-contained subdomain with internal hierarchy** (parabolic / elliptic / hyperbolic — the three canonical PDE classes; FE / FV / FD — the three canonical discretisations; explicit / implicit / IMEX — the three canonical timesteppers). Sub-package precedent: `prob/copula/`, `prob/conformal/`, `optim/proximal/`, `optim/transport/`. Krylov solvers (CG / GMRES / BiCGStab / preconditioned-CG) live in `linalg/` as **canonical home** because they are general SPD/non-symmetric linear-system solvers consumed by FE/FV-PDE assembly (also by RBF interpolation, GP regression, IRLS for sparse-coding) — single-source ownership in `linalg/` not `pde/`.

**CANDOR.** PDE-solvers are foundational to applied math but reality is a **library not a numerical-PDE-platform** (FreeFem++, FEniCS, deal.II, MFEM are the canonical FE platforms with 100k-1M LOC). Reality should ship **the substrate primitives that downstream Block-C reviews require** (D1-D15 ~1,720 LOC saturates 159+219+242+243+248) and stop short of large-scale-FE infrastructure (D26 P1-2D triangular FE is the *boundary* of what makes sense — beyond that we are reimplementing FreeFem++ which violates reality's "zero deps + reimplement from first principles" rule but for a 100k-LOC target). Cheapest-1-day shippable: **D1 + D2 + D3 + D6 + D7 + D14 ~600 LOC** ships UniformGrid1D + boundary-conditions + 1D-Laplacian-stencil + Thomas-tridiag + HeatEquation1D-explicit + CFL-helper saturating R-HEAT-GAUSSIAN-DECAY 4/4 against closed-form Gaussian initial-condition `u(x, t) = (4πt)^{-1/2} exp(-x²/(4t))` to 1e-6 over 1000 timesteps with periodic BC on `[-10, 10]`, and **simultaneously unblocks** P0c-242, F0a-243, M1-219, W1-159 (four downstream slots that already specified D1-D3 as their substrate). Highest-leverage-1-week-unlock: **PR-1 + PR-2 + PR-3 ~1,720 LOC** D1-D15 + Thomas tridiag — gives reality FIRST-CLASS 1D parabolic + 2D parabolic (ADI) + 1D-2D Poisson + CG-Krylov + method-of-lines, completing the substrate four-Block-C-downstreams need. **18 of 28 primitives unique to this slot** (D3 Laplacian shared with 242-P0c + 243-F0c; D6 tridiag shared with 243-F0d; D11 Poisson2D shared with 248-multigrid; D12 CG shared with 248 + 215-compressed-sensing + 237-GP-regression; D27 GMRES shared with 248 + 247-mortar-FEM; D29 Yee shared with 159-em-FDTD).

---

## 0. State at HEAD (2026-05-08, v0.10.0)

Repo-wide audit for PDE / finite-difference-stencil / boundary-condition / Laplacian-operator / wave-heat-poisson-helmholtz / hyperbolic-conservation-law / Riemann-solver / FEM-FVM / mass-stiffness-matrix / Krylov-CG-GMRES / multigrid / Yee-stagger surface — **zero callable matches** anywhere in `*.go` files outside review-corpus.

| Surface | Path | PDE relevance |
|---|---|---|
| `chaos.RK4Step / EulerStep / SolveODE` | `chaos/ode.go` | Deterministic-ODE substrate — **ONLY time-stepper in repo** — gates method-of-lines (D15) and explicit-Euler/RK4 timesteps for D7-D9-D17-D18 |
| `signal.FFT / IFFT` | `signal/fft.go` | Cooley-Tukey real DFT — gates pseudo-spectral / Fourier-Galerkin (slot 245 owns); diagonalises Laplacian on torus `Δ → -k²` |
| `linalg.LUSolve` | `linalg/decompose.go` | General LU O(n³) factor + O(n²) solve — usable for implicit-Euler / Crank-Nicolson but **wasteful** for tridiagonal: **Thomas O(n) absent** (D6 ships) |
| `linalg.Cholesky / CholeskySolve` | `linalg/decompose.go` | SPD-direct solver — usable for elliptic-PDE Poisson with SPD-stiffness; **CG/preconditioned-CG absent** (D12 ships) |
| `linalg.Eigen` | `linalg/eigen.go` | Householder→TQL — gates spectral-eigenfunction expansion (Karhunen-Loève for SPDE-242, Hermite-Galerkin for FPE-243, modal-decomposition for ROM-PDE) |
| `calculus.NumericalDerivative / NumericalGradient` | `calculus/calculus.go` | Central-difference — first-derivative only, no Laplacian, no boundary-handling |
| `infogeo.LaplacianKernel` | `infogeo/mmd.go` | Exp(-||x-y||/σ) MMD-kernel — name-collision with differential operator |
| `gametheory.Minimax` | `gametheory/nash.go` | "von Neumann 1928" minimax theorem — name-collision with 1947 stability-analysis |
| `physics.HeatCapacity / HeatFlux / FourierLaw` | `physics/thermo.go` | Scalar thermodynamic formulas — no spatial discretisation |
| `prob.PoissonPMF / PoissonProcess` | `prob/distributions.go` | Statistical Poisson — name-collision with Poisson PDE |
| `chaos/systems.go::torus topology` | `chaos/systems.go:148` | Lorenz attractor on torus — comment-only "boundary conditions wrap around" — not a PDE BC framework |
| `optim/proximal/` | `optim/proximal/*.go` | Moreau-Yosida envelope, ISTA/FISTA, prox-operators — usable for TV-regularised-PDE variational formulations but no PDE-solver |
| `pde/` package | -- | **ABSENT** — this slot creates |
| FE-mass / FE-stiffness assembly | -- | **ABSENT** — D25/D26 ship; slot 247 extends to mortar/hp |
| Multigrid V/W/FMG/AMG | -- | **ABSENT** — slot 248 owns |
| Domain decomposition Schwarz/FETI/BDDC | -- | **ABSENT** — slot 249 owns |
| Spectral methods Chebyshev/RBF/pseudo-spectral | -- | **ABSENT** — slot 245 owns (Fourier-spectral substrate `signal.FFT` PRESENT) |
| AMR Berger-Oliger / quadtree / octree | -- | **ABSENT** — D32 stub; defer |
| Yee-grid 3D-FDTD-Maxwell | -- | **ABSENT** — D29 stub blocked on `em` Vec3 surface (062-em-missing flagged) |
| D1-D28 PDE primitives | -- | **ALL ABSENT** (28 distinct primitives) |

**Cross-import edges that this slot creates.**
- `pde → chaos.RK4Step / EulerStep` (PRESENT) for method-of-lines time-stepping.
- `pde → linalg.LUSolve / Cholesky` (PRESENT) for direct elliptic-PDE solves on small grids.
- `pde → linalg/tridiag.Thomas` (NEW, D6) for 1D-implicit-PDE timesteps.
- `pde → linalg/cg.ConjugateGradient` (NEW, D12) for SPD-Krylov on Poisson-2D.
- `pde → linalg/gmres.GMRES` (NEW, D27) for non-symmetric Krylov on advection-diffusion.
- `pde/elliptic → linalg.Eigen` (PRESENT) for spectral-Helmholtz eigenvalue problems.
- `pde/hyperbolic → linalg.SolveLinearSystem` (PRESENT) for implicit-shock-capturing schemes.
- `pde → signal.FFT` (PRESENT) for spectral-Galerkin diagonalisation on torus (cross-link 245).
- `pde → calculus.NumericalGradient` (PRESENT) for adjoint-PDE gradient construction.

**Strict downstream consumers** of `pde/` substrate (already specified in agents/):
- `prob/spde/` (slot 242) consumes `pde.Laplacian1D` (D3) — explicit P0c flag.
- `pde/fpe/` (slot 243) consumes `pde.UniformGrid1D` (D1) + upwind (D18) + Laplacian (D3) + Thomas (D6) — explicit F0a-F0d flag.
- `gametheory/mfg/` (slot 219) consumes `pde.upwind` (D18) for HJB-Hamiltonian + `pde.fp_positivity` (custom, derived from D11 + D18) — explicit M1+M2 flag.
- `em/wave/` (slot 159) consumes `pde.WaveEquation1D-FDTD` (D16) + `pde.CFL` (D14) — explicit W1+W2 flag.
- `pde/multigrid/` (slot 248) consumes `pde.Laplacian2D` (D3) + `pde.PoissonEquation2D` (D11) for V/W/FMG cycles.
- `pde/mortar/` (slot 247) consumes `pde.FE-P1-2D` (D26) + GMRES (D27) for non-conforming-element coupling.
- `pde/domdecomp/` (slot 249) consumes `pde.UniformGrid2D` (D1) + GMRES (D27) for Schwarz-iteration.

---

## 1. The 28 PDE primitives

Each entry: capability + reference / composition / LOC / cross-link / blocking-flag.

### Tier-0 — grid+stencil substrate (~620 LOC) — STRICT-UPSTREAM of slots 159, 219, 242, 243, 247, 248, 249, 251

**D1 — `pde/grid.go::UniformGrid1D / UniformGrid2D / UniformGrid3D` ~140 LOC.** Cartesian uniform-spacing grids. `UniformGrid1D{N int, L float64, dx float64, X []float64}` with ghost-cell layout for boundary-handling (Strikwerda-2004 §4.2). `UniformGrid2D{Nx, Ny int, Lx, Ly float64, dx, dy float64}` row-major flat-storage. `UniformGrid3D{Nx, Ny, Nz int, ...}`. Helper methods `Idx(i, j) int` flat-index conversion, `IdxGhost(i, j) int` with halo. Pin: round-trip `Idx(g.Coord(k)) == k` for all k. Reference: LeVeque-2007 §2.1, Strikwerda-2004 §1.4.

**D2 — `pde/bc.go::BoundaryCondition` interface + 6 implementations ~120 LOC.** `BoundaryCondition` interface with `Apply(grid, field []float64)` method (zero-allocation). Six canonical implementations:
- `DirichletBC{Value float64}` — `u(x_boundary) = value` for stationary plates.
- `NeumannBC{Flux float64}` — `∂u/∂n |_boundary = flux` for insulated walls (flux=0).
- `RobinBC{Alpha, Beta, Gamma float64}` — `α·u + β·∂u/∂n = γ` for radiative-boundaries / convective-boundaries.
- `PeriodicBC{}` — `u(0) = u(L)` for torus topology.
- `ReflectingBC{}` — `∂u/∂n |_boundary = 0` for symmetry-axis (special-case Neumann with flux=0; semantically distinct for FPE-density).
- `AbsorbingBC{}` — `u(boundary) = 0` for first-passage-time problems (special-case Dirichlet with value=0; semantically distinct).

Pin: `DirichletBC{Value:1}` applied to `u = 0` everywhere yields `u[0] = u[N-1] = 1`. Cross-link 243-F0a. Reference: Strikwerda-2004 §3.2, Larsson-Thomée-2003 *Partial-Differential-Equations-with-Numerical-Methods* §1.

**D3 — `pde/laplacian.go::Laplacian1D{2,3,4,5,6}_th_order / Laplacian2D / Laplacian3D` ~200 LOC.** Centered finite-difference Laplacian stencils:
- 1D 2nd-order 3-point: `(u_{i+1} − 2u_i + u_{i-1})/Δx²` — error O(Δx²).
- 1D 4th-order 5-point: `(-u_{i+2} + 16u_{i+1} − 30u_i + 16u_{i-1} − u_{i-2})/(12Δx²)` — error O(Δx⁴).
- 1D 6th-order 7-point: `(2u_{i+3} − 27u_{i+2} + 270u_{i+1} − 490u_i + 270u_{i-1} − 27u_{i-2} + 2u_{i-3})/(180Δx²)` — error O(Δx⁶).
- 2D 5-point cross: `(u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} − 4u_{i,j})/Δx²`.
- 2D 9-point compact-Mehrstellen: `(4(u_{i±1,j} + u_{i,j±1}) + (u_{i±1,j±1}) − 20u_{i,j})/(6Δx²)` — 4th-order accuracy with same compact 9-point support (Spotz-Carey-1995).
- 3D 7-point: `(u_{i±1,j,k} + u_{i,j±1,k} + u_{i,j,k±1} − 6u_{i,j,k})/Δx²`.
- 3D 19-point and 27-point compact stencils for 4th-order 3D-Laplacian.

Output writes via `out []float64` (zero-allocation). Boundary cells dispatch to `BoundaryCondition.Apply`. Pin: closed-form eigenfunction `Δ sin(πx/L) = -(π/L)² sin(πx/L)` recovered at machine precision for spectral functions to 1e-12 with fixed-Dirichlet on `[0, L]`. Cross-link 242-P0c, 243-F0c. Reference: Smith-1985 *Numerical-Solution-of-PDEs-Finite-Difference-Methods* §5.2, Spotz-Carey-1995-IntJNumMethEng-38(20).

**D4 — `pde/divergence.go::Divergence2D / Divergence3D` ~40 LOC.** Centered FD divergence operator `∇·F = ∂F_x/∂x + ∂F_y/∂y` on staggered or co-located grid. Used for fluid-flow continuity equations + EM Gauss's-law check.

**D5 — `pde/gradient.go::Gradient2D / Gradient3D / Curl3D` ~40 LOC.** Centered FD gradient `∇φ = (∂φ/∂x, ∂φ/∂y, ∂φ/∂z)` and curl `∇×F`. Used for Poisson-equation post-processing (potential→field).

**D6 — `linalg/tridiag.go::ThomasAlgorithm` ~80 LOC.** Thomas algorithm O(n) tridiag-solver. Standard textbook implementation with diagonal pivot — solves `T·x = b` where T is tridiagonal with sub-diagonal `a`, diagonal `b`, super-diagonal `c`. Modifies inputs in-place for cache-locality. **CRITICAL** for every implicit-Euler / Crank-Nicolson 1D-PDE timestep — currently `linalg.LUSolve` is O(n³) factor + O(n²) solve which is wasteful for tridiagonal which is intrinsically O(n). 1D-implicit-PDE timestep dropped from O(n³) total to O(n) per timestep (1000x speedup at N=1000). Cross-link 243-F0d. Pin: `T·ThomasSolve(T, b)` returns `b` to within 1e-13 for diagonally-dominant T. Reference: Conte-de Boor-1980 *Elementary-Numerical-Analysis* §5.3, Press-1992 *Numerical-Recipes* §2.4.

### Tier-1 — keystone parabolic + elliptic (~1,100 LOC)

**D7 — `pde/parabolic/heat_explicit.go::HeatEquation1D_Explicit(u0, alpha, dt, dx, T, bc) [][]float64` ~140 LOC.** Solves `∂u/∂t = α ∂²u/∂x²` on `[0, L]` with explicit-Euler timestep `u^{n+1}_i = u^n_i + α·dt/dx²·(u^n_{i+1} − 2u^n_i + u^n_{i-1})`. Stability-bound `α·dt/dx² ≤ 1/2` (CFL) verified at entry — returns error if violated. Pin: closed-form Gaussian-fundamental-solution `u(x, t) = (4πt)^{-1/2} exp(-x²/(4αt))` matched to 1e-6 over 1000 timesteps with periodic BC on `[-10, 10]`. **THE CHEAPEST-1-DAY-SHIPPABLE-PRIMITIVE** — saturates R-HEAT-GAUSSIAN-DECAY 4/4 (mode-by-mode amplitude `e^{-αk²t}` + L²-energy decay + maximum-principle `u ≤ ||u_0||_∞` + diffusion-of-Dirac-delta to Gaussian). Reference: Strikwerda-2004 §4.1, LeVeque-2007 §1.1.

**D8 — `pde/parabolic/heat_implicit.go::HeatEquation1D_CrankNicolson(u0, alpha, dt, dx, T, bc) [][]float64` ~160 LOC.** Crank-Nicolson scheme `(I − α·dt/(2dx²)·L)·u^{n+1} = (I + α·dt/(2dx²)·L)·u^n`. Unconditionally stable + second-order accurate in time + space (Crank-Nicolson-1947). Tridiagonal linear system solved via D6 ThomasAlgorithm (O(n) per timestep). Pin: same Gaussian closed-form to 1e-9 (better than D7 due to higher-order accuracy + larger stable timestep). Reference: Crank-Nicolson-1947-PCPS-43(1), Strikwerda-2004 §6.3.

**D9 — `pde/parabolic/heat_2d_adi.go::HeatEquation2D_ADI(u0, alpha, dt, dx, dy, T, bc) [][]float64` ~220 LOC.** **SINGULAR-MOAT.** Peaceman-Rachford-1955 alternating-direction-implicit:
- Half-step 1: `u^{n+1/2}_{ij} − α·dt/(2dx²)·δ²_x u^{n+1/2}_{ij} = u^n_{ij} + α·dt/(2dy²)·δ²_y u^n_{ij}` (implicit in x, explicit in y).
- Half-step 2: `u^{n+1}_{ij} − α·dt/(2dy²)·δ²_y u^{n+1}_{ij} = u^{n+1/2}_{ij} + α·dt/(2dx²)·δ²_x u^{n+1/2}_{ij}` (implicit in y, explicit in x).

Each half-step is N tridiagonal systems (one per row/column) solved via D6 ThomasAlgorithm. Total cost O(N²) per full timestep — **identical to explicit cost but unconditionally stable**. Second-order in time + space. Pin: 2D-Gaussian-fundamental-solution `u(x, y, t) = (4πt)^{-1} exp(-(x²+y²)/(4αt))` to 1e-7. Reference: Peaceman-Rachford-1955-JSIAM-3(1), Wachspress-Habetler-1960-JSIAM-8(2).

**D10 — `pde/elliptic/poisson_1d.go::PoissonEquation1D(f, dx, bc) []float64` ~100 LOC.** Solves `-d²u/dx² = f(x)` on `[0, L]` via direct tridiagonal solve (D6 ThomasAlgorithm). 5-point stencil; Dirichlet/Neumann/Robin/Periodic via D2. Pin: `f(x) = -π²sin(πx)` with `u(0)=u(L)=0` recovers `u(x) = sin(πx)` to 1e-10 at N=1000. Reference: Strikwerda-2004 §3.1.

**D11 — `pde/elliptic/poisson_2d.go::PoissonEquation2D{Jacobi, GaussSeidel, SOR}(f, dx, dy, bc) [][]float64` ~180 LOC.** Solves `-Δu = f(x, y)` on rectangular domain. Three iterative solvers:
- **Jacobi**: `u^{k+1}_{ij} = (u^k_{i±1,j} + u^k_{i,j±1} + dx²·f_{ij}) / 4` — O(N²) per iteration, O(N²) iterations to convergence (poor scaling).
- **Gauss-Seidel**: in-place update — 2x faster than Jacobi.
- **SOR (Successive-Over-Relaxation)**: `u^{k+1} = (1-ω)u^k + ω·u^{GS}` with optimal ω = 2/(1 + sin(π/N)) — O(N²) per iteration, O(N) iterations (Young-1954).

Convergence bound `||u^{k+1} - u^*|| ≤ ρ^k ||u^0 - u^*||` with spectral-radius ρ. Pin: `f(x, y) = -2π²sin(πx)sin(πy)` recovers `u(x, y) = sin(πx)sin(πy)` to 1e-6. Reference: Young-1954-TransAMS-76(1), Hageman-Young-1981 *Applied-Iterative-Methods*.

**D12 — `linalg/cg.go::ConjugateGradient(A_mul func([]float64) []float64, b []float64, tol float64) []float64` ~140 LOC.** Hestenes-Stiefel-1952 conjugate-gradient method for SPD systems. Matrix-free interface: `A_mul` is a closure computing `A·x` (so user can pass dense matrix, sparse matrix, or PDE-Laplacian-stencil-evaluator with no allocation). Convergence in ≤ N iterations exact arithmetic; in practice O(√κ) iterations where κ is condition-number. Used for D11 PoissonEquation2D (alternative to SOR) + slot 248 multigrid + slot 215 compressed-sensing + slot 237 GP-regression. Pin: standard 2D-Poisson on 64×64 grid converges to residual 1e-8 in ≤ 200 iterations. Reference: Hestenes-Stiefel-1952-JResNBS-49(6), Saad-2003 *Iterative-Methods-for-Sparse-Linear-Systems* §6.7.

**D13 — `pde/vneumann.go::VonNeumannStability(scheme, dx, dt, alpha float64) (stable bool, amplification complex128)` ~80 LOC.** Symbolic-numerical von Neumann stability analysis: substitute `u_j^n = G^n e^{ikx_j}` into difference scheme, solve for amplification factor G(k, dx, dt, params), check `|G| ≤ 1` for all k. Helper for users designing custom schemes. Reference: von Neumann-Richtmyer-1950-JApplPhys-21, Charney-Fjørtoft-vonNeumann-1950-Tellus-2.

**D14 — `pde/cfl.go::CFL{Heat, Wave, Advection}(dt, dx, c float64) bool` ~30 LOC.** Closed-form CFL bound checks:
- `CFLHeat`: `α·dt/dx² ≤ 1/2` (parabolic explicit).
- `CFLWave`: `c·dt/dx ≤ 1` (hyperbolic, Courant-Friedrichs-Lewy-1928).
- `CFLAdvection`: `|v|·dt/dx ≤ 1` (advection-explicit upwind).

Cross-link 159-W2. Reference: Courant-Friedrichs-Lewy-1928-MathAnn-100(1).

**D15 — `pde/mol.go::MethodOfLines(spatial_op func([]float64) []float64, u0, T, dt) [][]float64` ~50 LOC.** Method-of-lines wrapper: discretize PDE in space → ODE system → integrate via `chaos.RK4Step` or `chaos.EulerStep`. Single function turns any spatial-operator into a time-evolved PDE solver. Reuses chaos.RK4 substrate (already PRESENT). Reference: Schiesser-1991 *Numerical-Method-of-Lines-Integration-of-PDEs*.

### Tier-2 — hyperbolic + advection (~1,180 LOC)

**D16 — `pde/hyperbolic/wave_1d.go::WaveEquation1D_FDTD(u0, v0, c, dt, dx, T, bc) [][]float64` ~140 LOC.** Solves `∂²u/∂t² = c²∂²u/∂x²`. Second-order leapfrog `u^{n+1}_i = 2u^n_i − u^{n-1}_i + (c·dt/dx)²·(u^n_{i+1} − 2u^n_i + u^n_{i-1})`. CFL `c·dt/dx ≤ 1` (D14). Cross-link 159-W1. Pin: travelling-wave `u(x, t) = sin(k(x - ct))` to 1e-6 over 100 periods. Reference: Yee-1966-IEEETransAntPropag-14(3), Strikwerda-2004 §10.

**D17 — `pde/hyperbolic/wave_2d.go::WaveEquation2D_FDTD(u0, v0, c, dt, dx, dy, T, bc) [][]float64` ~180 LOC.** 2D wave-equation FDTD with 5-point Laplacian. CFL `c·dt·√(1/dx² + 1/dy²) ≤ 1`. Pin: 2D-Gaussian initial-condition spreads as expanding ring (Huygens' principle in 2D — partial — at 2D the wave does NOT have sharp-front-vanishing-tail). Reference: Yee-1966.

**D18 — `pde/hyperbolic/advection.go::Advection1D{Upwind, LaxFriedrichs, LaxWendroff}(u0, v, dt, dx, T, bc)` ~160 LOC.** Solves `∂u/∂t + v·∂u/∂x = 0`. Three schemes:
- **Upwind**: `u^{n+1}_i = u^n_i − v·dt/dx·(u^n_i − u^n_{i-1})` if v>0 — first-order accurate, monotone, diffusive (numerical diffusion ∼ v·dx/2).
- **Lax-Friedrichs**: `u^{n+1}_i = (u^n_{i+1} + u^n_{i-1})/2 − v·dt/(2dx)·(u^n_{i+1} − u^n_{i-1})` — first-order, stable for |v·dt/dx| ≤ 1, large numerical diffusion.
- **Lax-Wendroff**: `u^{n+1}_i = u^n_i − v·dt/(2dx)·(u^n_{i+1} − u^n_{i-1}) + (v·dt)²/(2dx²)·(u^n_{i+1} − 2u^n_i + u^n_{i-1})` — second-order accurate, dispersive (Gibbs oscillations near shocks).

Cross-link 219-M1 (HJB upwind) + 243-F0b. Pin: Square-pulse advection; Lax-Wendroff exhibits Gibbs oscillations as expected. Reference: Lax-Friedrichs-1954-CommPureApplMath-7, Lax-Wendroff-1960-CommPureApplMath-13.

**D19 — `pde/hyperbolic/adr.go::AdvectionDiffusionReaction1D(u0, v, alpha, react, dt, dx, T, bc)` ~140 LOC.** Solves `∂u/∂t + v·∂u/∂x = α·∂²u/∂x² + R(u)`. Strang-splitting: `L^{dt/2} N^{dt} L^{dt/2}` with L = advection-diffusion (linear, implicit), N = reaction (nonlinear, explicit). Used for combustion / chemistry / population-models. Reference: Strang-1968-SIAMJNumerAnal-5(3).

**D20 — `pde/hyperbolic/riemann_hll.go::RiemannSolverHLL` ~140 LOC.** Harten-Lax-vanLeer-1983 approximate Riemann solver for hyperbolic conservation laws. Given left/right states (ρ_L, u_L, p_L) and (ρ_R, u_R, p_R) for Euler-equations, estimate wave-speeds (S_L, S_R), compute HLL flux:
```
F_HLL = { F_L                       if S_L > 0
        { (S_R F_L − S_L F_R + S_L S_R (U_R − U_L)) / (S_R − S_L)  if S_L ≤ 0 ≤ S_R
        { F_R                       if S_R < 0
```
Used in D22 Godunov + D23 MUSCL. Reference: Harten-Lax-vanLeer-1983-SIAMReview-25(1), Toro-2009 *Riemann-Solvers-and-Numerical-Methods-for-Fluid-Dynamics* §10.

**D21 — `pde/hyperbolic/riemann_hllc.go::RiemannSolverHLLC` ~160 LOC.** Toro-Spruce-Speares-1994 HLLC restoring contact discontinuity. Adds intermediate wave-speed S_M for the contact wave, giving 4-state intermediate Riemann fan. Better than HLL for contact-resolution (essential for material-interface tracking, plasma physics). Reference: Toro-Spruce-Speares-1994-ShockWaves-4, Toro-2009 §10.6.

**D22 — `pde/hyperbolic/godunov.go::GodunovScheme(u0, flux, dt, dx, T, riemann_solver)` ~140 LOC.** First-order conservative finite-volume scheme: `u^{n+1}_i = u^n_i − dt/dx·(F_{i+1/2} − F_{i-1/2})` where `F_{i+1/2} = RiemannSolver(u^n_i, u^n_{i+1})`. Monotone, total-variation-diminishing (TVD), preserves conservation exactly. First-order accurate (dispersive). Reference: Godunov-1959-MatSb-47(89), LeVeque-1992 *Numerical-Methods-for-Conservation-Laws* §12.

**D23 — `pde/hyperbolic/muscl.go::MUSCL_Hancock(u0, flux, dt, dx, T, slope_limiter)` ~120 LOC.** vanLeer-1979 Monotone-Upstream-centered-Scheme-for-Conservation-Laws, second-order TVD scheme. Reconstruct linear-piecewise within each cell with slope-limiter (minmod, vanLeer, vanAlbada, superbee) — limit slope to prevent new extrema. Predictor-corrector half-step Hancock-1980. Reference: vanLeer-1979-JCompPhys-32, Toro-2009 §13-14.

### Tier-3 — high-order + finite-element (~840 LOC)

**D24 — `pde/hyperbolic/weno.go::WENO5_JiangShu(u0, flux, dt, dx, T)` ~180 LOC** **CUTTING-EDGE.** Jiang-Shu-1996 5th-order weighted-essentially-non-oscillatory scheme. Three 3rd-order stencils with smoothness-indicators β_k weight the candidate stencils as `ω_k = α_k / Σ α_k` with `α_k = d_k / (β_k + ε)²`. The smoothness-indicators downweight stencils crossing discontinuities, yielding 5th-order accuracy in smooth regions and shock-capturing without oscillations. Used in MFEM, Chombo, Athena++ for compressible flow / astrophysical-MHD. Pin: smooth sin-wave maintains 5th-order convergence; square-wave captures shock without Gibbs. Reference: Jiang-Shu-1996-JCompPhys-126(1), Shu-2009-ActaNumerica-18.

**D25 — `pde/fem/p1_1d.go::FE_P1_1D(grid, f, bc) []float64` ~220 LOC.** P1-linear finite-element method on 1D mesh. Assembly:
- Element mass matrix `M^e = (h/6) [[2, 1], [1, 2]]`.
- Element stiffness matrix `K^e = (1/h) [[1, -1], [-1, 1]]`.
- Global assembly `K_{IJ} = Σ_e K^e_{ij}` via local-to-global element-DOF-mapping.
- Right-hand-side `f_I = Σ_e ∫_e φ_I(x) f(x) dx` via Gauss-quadrature.
- Apply Dirichlet by row-replacement; Neumann via boundary integral; Robin via Robin matrix.
- Solve `K·u = f` via D12 CG (SPD) or D6 Thomas (tridiagonal in 1D).

Pin: solve `-u'' = π² sin(πx), u(0)=u(1)=0` recovers `u(x) = sin(πx)` to 1e-6 at N=100. Reference: Brenner-Scott-2008 *Mathematical-Theory-of-FE-Methods* §3.

**D26 — `pde/fem/p1_2d.go::FE_P1_2D(triangulation, f, bc) []float64` ~280 LOC.** P1-linear FE on triangular mesh. Triangulation input `[][3]int` triangle-vertex-indices + `[][2]float64` vertex-coordinates. Per-triangle assembly:
- Element stiffness `K^e_{ij} = ∫_e ∇φ_i · ∇φ_j dA` analytically `(b_i b_j + c_i c_j)/(4·area)` for linear basis.
- Element mass `M^e_{ij} = ∫_e φ_i φ_j dA` analytically `area/6 · (1 + δ_{ij})`.
- Global assembly via sparse-COO-to-CSR conversion.
- Solve via D27 GMRES (general non-symmetric) or D12 CG (Galerkin-symmetric).

Pin: solve Poisson on unit-square unstructured mesh, compare to D11 5-point FD → matches to 1/N convergence. Reference: Brenner-Scott-2008 §3.7, Hughes-2000 *FE-Method* §1.

**D27 — `linalg/gmres.go::GMRES(A_mul func([]float64) []float64, b []float64, restart int, tol float64) []float64` ~160 LOC.** Saad-Schultz-1986 generalized-minimum-residual method for non-symmetric linear systems. Builds Arnoldi orthonormal basis of Krylov subspace K_m = span{r_0, A r_0, ..., A^{m-1} r_0}, minimizes residual via least-squares on Hessenberg matrix. Restart at m=`restart` iterations to bound memory O(m·N). Used for D26 P1-2D-FEM (when non-symmetric stabilisation), D29 advection-diffusion (non-self-adjoint), slot 248 multigrid as smoother. Pin: convergence to 1e-8 on diagonally-dominant non-symmetric system in ≤ 2N iterations. Reference: Saad-Schultz-1986-SIAMJSciStatComput-7(3), Saad-2003 §6.5.

**D28 — `linalg/bicgstab.go::BiCGStab(A_mul, b, tol) []float64` ~140 LOC.** vanderVorst-1992 BiConjugate-Gradient-Stabilized for non-symmetric linear systems — alternative to GMRES with O(1) memory (no Arnoldi basis storage). More chaotic convergence than GMRES but cheaper per iteration. Used as fallback when GMRES restart-parameter tuning is impractical. Reference: vanderVorst-1992-SIAMJSciStatComput-13(2), Saad-2003 §7.4.

### Tier-3.5 — frontier + DEFER (~460 LOC)

**D29 — `em/wave/yee_3d.go::YeeGrid3D_FDTD ~200 LOC ⊘ DEFER.** Yee-1966 staggered 3D FDTD-Maxwell. Stagger E-fields on cell-edges, B-fields on cell-faces, exact discrete-Maxwell preserved. **Defer until** slot 062-em-missing ships `Vec3` + `complex128` impedance surface (Yee needs 3D vector EM-fields). Recommended placement `em/wave/yee_3d.go` co-shipped with slot 159-W4-W6 EM-FDTD primitives. Reference: Yee-1966-IEEETransAntProp-14, Taflove-Hagness-2005 *Computational-Electrodynamics-FDTD-Method*.

**D30 — `pde/spectral/sem.go::SpectralElementMethod ~120 LOC ⊘ DEFER.** Patera-1984 spectral-element method (hp-adaptive high-order FEM with Lagrange basis at Gauss-Lobatto-Legendre quadrature points). **Blocked on slot 245** (Chebyshev / Legendre / Gauss-Lobatto quadrature substrate). Reference: Patera-1984-JCompPhys-54, Karniadakis-Sherwin-2005 *Spectral-hp-Element-Methods-for-CFD*.

**D31 — `pde/dg/dg_1d.go::DiscontinuousGalerkin ~140 LOC ⊘ DEFER.** Hesthaven-Warburton-2008 nodal-DG: piecewise-polynomial basis discontinuous across element boundaries, fluxes via Riemann-solver (D20-D21). Most flexible high-order scheme (used in Nek5000, MFEM, deal.II for compressible-flow + electromagnetics). Defer: implementation surface large + consumer pull from reality is research-only. Reference: Hesthaven-Warburton-2008 *Nodal-DG-Methods*, Cockburn-Shu-1998-JSciComput-16.

**D32 — `pde/amr/quadtree.go::AdaptiveMeshRefinement ~280 LOC ⊘ DEFER.** Berger-Oliger-1984 patch-based AMR with quadtree (2D) / octree (3D). Refine where error-estimator (Richardson-extrapolation or gradient-based) exceeds threshold. Defer: AMR is the *infrastructural* PDE-frontier with 10k-100k LOC implementations (Chombo, BoxLib, AMReX). Reality cannot ship AMReX-equivalent. Reference: Berger-Oliger-1984-JCompPhys-53, Berger-Colella-1989-JCompPhys-82.

**D33 — `pde/fmm/fmm.go::FastMultipoleMethod ~200 LOC ⊘ DEFER.** Greengard-Rokhlin-1987 fast-multipole method for `O(N) N-body` summation (alternative to direct O(N²)). Used for boundary-integral methods (Laplace / Helmholtz / Stokes equations on complex geometries). Defer to slot 274 randomized-numerics or its own slot — not core PDE substrate. Reference: Greengard-Rokhlin-1987-JCompPhys-73.

---

## 2. Connective tissue — what each new edge buys

| Edge | LOC of glue | What it unlocks |
|---|---|---|
| `pde/ → chaos.RK4Step` | 0 — already callable | Method-of-lines (D15) and explicit-Euler timesteps reuse chaos ODE engine |
| `pde/ → linalg.LUSolve / Cholesky` | 0 — already callable | Direct elliptic-PDE solves on small grids (D10, D25) |
| `pde/ → linalg/tridiag.Thomas` | 80 LOC NEW (D6) | 1D-implicit-PDE timesteps O(n) per step (D8, D10) |
| `pde/ → linalg/cg.ConjugateGradient` | 140 LOC NEW (D12) | SPD-Krylov for Poisson-2D, FEM-stiffness, slot 248 multigrid |
| `pde/ → linalg/gmres.GMRES` | 160 LOC NEW (D27) | Non-symmetric-Krylov for advection-diffusion + non-symmetric-FEM (D26) |
| `pde/ → linalg/bicgstab.BiCGStab` | 140 LOC NEW (D28) | O(1)-memory non-symmetric Krylov fallback |
| `pde/ → linalg.Eigen` | 0 — already callable | Spectral-Helmholtz eigenvalue problems |
| `pde/ → signal.FFT` | 0 — already callable | Fourier-Galerkin diagonalisation on torus (gates spectral-Galerkin variants of D7-D8 + cross-link 245) |
| `pde/ → calculus.NumericalGradient` | 0 — already callable | Adjoint-PDE gradient construction for inverse problems |
| `prob/spde/ → pde.Laplacian1D` | consumer-side 0 (slot 242 P0c) | Stochastic-heat / Allen-Cahn / Burgers / Cahn-Hilliard via FD-Laplacian |
| `pde/fpe/ → pde.{grid, bc, laplacian, upwind}` | consumer-side 0 (slot 243 F0a-F0c) | Fokker-Planck / Klein-Kramers / Black-Scholes-from-FPE substrate |
| `gametheory/mfg/ → pde.upwind + pde.fp_positivity` | consumer-side 0 (slot 219 M1+M2) | HJB-FD upwind-Hamiltonian + FP positivity-preserving for MFG-numerics |
| `em/wave/ → pde.WaveEquation1D-FDTD + pde.CFL` | consumer-side 0 (slot 159 W1+W2) | 1D-EM-wave-propagation, dispersion-relation, group-velocity |
| `pde/multigrid/ → pde.{Laplacian2D, PoissonEquation2D, GMRES}` | slot 248 owns | V/W/FMG multigrid cycles on FE/FD-Laplacian |
| `pde/mortar/ → pde/fem.P1_2D + linalg.GMRES` | slot 247 owns | Non-conforming-element coupling / hp-adaptive |

**Total upstream substrate this slot ships ~3,800 LOC** unblocks **~5,000+ LOC of downstream Block-C reviews already scoped in agents/** (242-P-family, 243-F-family, 219-M-family, 159-W-family, 247, 248, 249, 251).

---

## 3. Cross-package blockers and synergies

| Blocker | Owner | Blocks | LOC est |
|---|---|---|---:|
| `pde/grid.go` UniformGrid 1D/2D/3D (D1) | this slot 244 | 242-P0c, 243-F0a, 219-M1, 159-W1 | 140 |
| `pde/bc.go` 6 BC types (D2) | this slot 244 | 242, 243, 219, 159, 247, 248 | 120 |
| `pde/laplacian.go` 1D-2D-3D-stencils (D3) | this slot 244 | 242-P0c, 243-F0c, 248 | 200 |
| `linalg/tridiag.go` Thomas (D6) | this slot 244 | 243-F0d (single-most-leveraged commit for FPE) | 80 |
| `linalg/cg.go` CG (D12) | this slot 244 | 248-multigrid, 215-compressed-sensing, 237-GP-regression | 140 |
| `linalg/gmres.go` GMRES (D27) | this slot 244 | 247-mortar-FEM, 248-multigrid-smoother | 160 |
| 2D / 3D FFT (slot 245) | 245-spectral | spectral-Galerkin variants of D7-D8, 242-P9 2D-NSE, 242-P10 Phi-4 | (slot 245) |
| Adaptive ODE timestepper RK45 (slot 027-chaos-missing) | 027 | stiff-PDE regimes (D7-D8 with sharp gradients) | (slot 027) |
| `em/Vec3` + `complex128` impedance | 062-em-missing | D29 Yee-grid-3D-FDTD | (slot 062) |
| Sparse-matrix CSR/COO format | 097-linalg-missing | D26 P1-2D FE-assembly (currently dense-matrix fallback) | (slot 097) |

**Cross-link audit.**
- 159-synergy-em-signal: **strict-downstream** — W1 Wave1DFDTD + W2 CFL = D16 + D14 here; W3-W6 EM-FDTD-2D + dispersion-relations build on top.
- 192-synergy-fluids-control: **light-downstream** — ROM-based-flow-control via reduced-order-PDE-projection consumes pde/fem.P1_2D (D26) for snapshot-POD basis.
- 219-new-mean-field-games: **strict-downstream** — M1 HJB-upwind = D18 here; M2 FP-positivity-preserving = D11 + D18 composition.
- 242-new-spde: **strict-downstream** — P0c Laplacian1D = D3 here (242 explicitly says "either ship in slot 244 or inline here"); P2-P6 stochastic-PDEs all consume pde-substrate.
- 243-new-fpe: **strict-downstream** — F0a-F0c grid+upwind+Laplacian = D1+D18+D3 here; F0d Thomas = D6 here. **Single-source-ownership recommendation: F0a-F0c live in `pde/`, slot 244 owns; 243 consumes**.
- 245-new-spectral-methods: **strict-twin** — 244 owns FD/FV/FEM, 245 owns Chebyshev/Fourier/RBF/pseudo-spectral; spectral-Galerkin variants of D7-D8 use signal.FFT (PRESENT) but require Chebyshev / Legendre quadrature owned by 245.
- 246-new-discrete-exterior: **orthogonal** — DEC operators (∂, d, ⋆) on simplicial complexes are an alternative discretisation (Hirani-2003); minor overlap on FE-mass / FE-stiffness viewpoint.
- 247-new-mortar-fem: **strict-downstream** — mortar / non-conforming FEM consumes D26 P1-2D-FE; hp-adaptivity consumes D30 SEM (deferred).
- 248-new-multigrid: **strict-downstream** — multigrid V/W/FMG/AMG cycles SOLVE the linear systems built by D11 + D26; smoothers use D12 CG / D27 GMRES.
- 249-new-domain-decomp: **strict-downstream** — Schwarz / FETI / BDDC consume D1 grid + D27 GMRES.
- 251-new-shape-opt: **strict-downstream** — level-set + phase-field PDEs consume D7 + D11 + D26.

---

## 4. Sequencing and PRs

**PR-1 (1 day, ~600 LOC)** — Tier-0 substrate + cheapest-ship: D1 grid + D2 BC + D3 Laplacian + D6 Thomas-tridiag + D7 HeatEquation1D-explicit + D14 CFL. Saturates R-HEAT-GAUSSIAN-DECAY 4/4 against closed-form Gaussian fundamental solution. Single-day-shippable. **SIMULTANEOUSLY UNBLOCKS** 242-P0c, 243-F0a + F0c + F0d, 219-M1 (partial), 159-W2 — single 600-LOC PR clears P3 of FOUR downstream Block-C reviews.

**PR-2 (2 days, ~620 LOC)** — Tier-1 keystone parabolic + elliptic: D8 Crank-Nicolson + D9 ADI + D10 Poisson-1D + D11 Poisson-2D + D12 CG + D13 vonNeumann + D15 method-of-lines. Saturates 2D-Gaussian-decay + Poisson-eigenfunction R-pins. Unblocks 248-multigrid (CG smoother PRESENT) + 215-compressed-sensing (CG-IRLS).

**PR-3 (2 days, ~860 LOC)** — Tier-2 hyperbolic + advection: D16 Wave1D + D17 Wave2D + D18 advection-3-schemes + D19 ADR + D20 HLL + D22 Godunov + D23 MUSCL. Saturates R-CONSERVATION 4/4 (mass conservation + total-variation-bounds + entropy-condition + Riemann-fan-resolution). Unblocks 159-W1 + 219-M2.

**PR-4 (2 days, ~480 LOC)** — Tier-2.5 high-order + Riemann: D21 HLLC + D24 WENO5. WENO5 is **THE singular-cutting-edge moat** of this slot.

**PR-5 (3 days, ~860 LOC)** — Tier-3 finite-element + Krylov: D25 P1-1D-FE + D26 P1-2D-FE + D27 GMRES + D28 BiCGStab. Unblocks 247-mortar + 248-multigrid + 249-domain-decomp.

**PR-6 deferred** — D29 Yee (blocked-on-em-Vec3) + D30 SEM (blocked-on-245) + D31 DG + D32 AMR + D33 FMM (research-frontier, defer until consumer pull).

**Total ~3,420 LOC across 5 PRs over ~10 engineer-days.** Delivers the production-grade PDE substrate that 8+ downstream Block-C slots require.

---

## 5. Saturation pins (R-pin checklist)

- **R-HEAT-GAUSSIAN-DECAY 4/4** (D7+D8+D9): mode-by-mode amplitude `e^{-αk²t}` 1e-6 + L²-energy decay 1e-6 + maximum-principle `u ≤ ||u_0||_∞` 1e-12 + Dirac→Gaussian fundamental-solution at long times 1e-6.
- **R-POISSON-EIGENFUNCTION 3/3** (D10+D11+D25+D26): `f = sin(πx) sin(πy)` recovers `u = sin(πx)sin(πy)/(2π²)` to 1e-6; FD vs FE agreement to O(h²); CG vs SOR vs direct-solve agreement to 1e-9.
- **R-WAVE-D'ALEMBERT 3/3** (D16+D17): travelling-wave `u(x, t) = sin(k(x − ct))` to 1e-6 over 100 periods; energy-conservation `∫(u_t² + c²u_x²)dx` to 1e-9; CFL violation triggers exponential-growth (anti-pin).
- **R-CONSERVATION 4/4** (D18+D22+D23+D24): mass `Σu_i Δx` constant to 1e-13; total-variation non-increasing for TVD schemes (D22+D23+D24); Riemann-fan resolution; Sod-shock-tube reference solution to 1e-3.
- **R-CFL-STABILITY 3/3** (D14+D13): D7 stable for `α·dt/dx² ≤ 0.5` blows up at 0.55; D16 stable for `c·dt/dx ≤ 1` blows up at 1.05; D18 upwind stable for |v·dt/dx| ≤ 1 (numerical-diffusion not blow-up).
- **R-FE-CONVERGENCE 3/3** (D25+D26): `||u_h − u||_{L²} ≤ C·h²` for P1 elements; mass-matrix eigenvalues real-positive; stiffness-matrix symmetric-PSD.
- **R-KRYLOV-CONVERGENCE 3/3** (D12+D27+D28): 2D-Poisson 64×64 CG converges 1e-8 in ≤ 200 iter; GMRES on shifted-Laplacian non-symmetric converges in ≤ 2N iter; BiCGStab on same with O(1)-memory.
- **R-WENO-ORDER 2/2** (D24): smooth sin-wave 5th-order convergence `||error|| ∼ h^5`; square-wave shock captured without Gibbs (TV non-increasing).
- **R-ADI-ORDER 2/2** (D9): 2D-Gaussian decay 2nd-order in time + space; unconditionally stable for any dt (anti-pin: D7 explicit blows up at large dt; D9 ADI does not).

---

## 6. Differentiation from prior agents in 400-roster

This report is **PDE-discretization-substrate-pure** (FD-stencils + FV-Riemann-solvers + FE-mass+stiffness + Krylov-solvers + boundary-conditions + CFL/von-Neumann-stability) where:
- **159-em-signal** owns EM-wave-as-PDE consumer-side-application (W1-W6 EM-FDTD specifically) — strict-downstream of this slot's D14+D16.
- **219-MFG** owns mean-field-game PDE-system-coupling-and-fixed-point-iteration — strict-downstream of this slot's D3+D11+D18.
- **242-SPDE** owns stochastic-PDE-noise-discretization — strict-downstream of this slot's D3 + signal.FFT.
- **243-FPE** owns Fokker-Planck-PDE positivity-preserving-Chang-Cooper-equilibrium-preserving consumer — strict-downstream of this slot's D1+D3+D6+D18.
- **245-spectral-methods** owns Chebyshev/Legendre/RBF/pseudo-spectral — **strict-twin** of this slot (different discretisation family for the same PDE classes).
- **246-discrete-exterior** owns DEC on simplicial complexes — **orthogonal** discretisation framework.
- **247-mortar-FEM** owns hp-adaptive non-conforming-FEM — strict-downstream of this slot's D25+D26.
- **248-multigrid** owns V/W/FMG/AMG cycles — strict-downstream of this slot's D11+D26+D12+D27.
- **249-domain-decomp** owns Schwarz/FETI/BDDC — strict-downstream of this slot's D1+D27.
- **251-shape-opt** owns level-set+phase-field+SIMP — strict-downstream of this slot's D7+D11+D26.

**18 of 28 primitives unique to this slot** (D1 / D2 / D3 / D6 partially shared with 242-P0c, 243-F0a-F0d; D11 partially shared with 248-multigrid; D12 / D27 shared with 215, 237, 247, 248; D14 / D16 shared with 159; D18 shared with 219-M1, 243-F0b).

---

## 7. CANDOR — what NOT to ship

PDE-numerics is foundational but reality is a **library not a numerical-PDE-platform**. Canonical FE-platforms (FreeFem++, FEniCS, deal.II, MFEM) are 100k-1M LOC of MPI-parallel + adaptive-mesh + hp-FEM + multiphysics infrastructure that reality cannot and should not match. Recommended **ship the substrate primitives D1-D28 ~3,800 LOC** that downstream Block-C reviews require, **stop at P1-FEM-2D** (D26), and **defer**:
- D29 Yee-3D-FDTD until 062-em-missing ships Vec3 substrate.
- D30 SEM until 245-spectral ships Legendre/Gauss-Lobatto.
- D31 DG (research-frontier).
- D32 AMR (10k-100k-LOC infrastructure).
- D33 FMM (its own slot 274 candidate).
- p-FEM, hp-FEM beyond P1+P2 (slot 247 owns).
- Multigrid (slot 248 owns).
- Domain decomposition (slot 249 owns).

Reality's PDE-niche: **provide the canonical zero-dep-Go FD/FV/FE-substrate that EIGHT downstream Block-C reviews need** (159, 192, 219, 242, 243, 247, 248, 249, 251). Beyond that, downstream consumers should integrate with FreeFem++ / FEniCS via JSON-fixture-export — reality does not compete at the deal.II-tier.

**End ~395 lines.**

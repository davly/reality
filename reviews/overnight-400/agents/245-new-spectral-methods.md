# 245 | new-spectral-methods — Spectral methods for PDEs (Chebyshev, Fourier, RBF, pseudo-spectral)

**Summary line 1.** reality v0.10.0 ships the FFT half of one (Fourier) spectral basis and nothing else of a spectral-PDE stack — repo-wide grep on `chebyshev|legendre.poly|hermite.poly|laguerre|jacobi.poly|gauss.lobatto|gauss.radau|spectral.differentiation|cardinal.function|barycentric|pseudospectral|collocation|chebfun|chebyshev.transform|spectral.viscosity|exponential.filter|cesaro.filter|boyd.filter|spectral.element|sem|dgsem|RBF|radial.basis|multiquadric|polyharmonic|RBF.FD|fornberg|partition.of.unity.RBF|ETDRK|exponential.time.differencing|cox.matthews` against `*.go` returns **zero callable matches** outside the autodiff finite-difference doc-mention and signal/window.go which only ships Hann/Hamming/Blackman tapers (NOT spectral filters in the Boyd-2001 §11 sense — Hann-window is for STFT side-lobe-suppression, distinct from Boyd-vandeven-erfc-log filter for de-aliasing spectral-PDE solutions). Closest extant substrate is `signal/fft.go` (Cooley-Tukey radix-2 FFT/IFFT + PowerSpectrum + FFTFrequencies — **the entire Fourier-spectral substrate** that gates pseudo-spectral on `T^d` torus problems; diagonalises Laplacian to `Δ→ -k²`, advection to `∂_x → ik`, biharmonic to `Δ² → k⁴` — already-callable); `calculus.GaussLegendre` (precomputed 2-5 point Gauss-Legendre nodes+weights on `[-1,1]` — gates Galerkin inner-product evaluation but capped at 5 points so unsuitable for spectral-Galerkin where N=64-256 modes is typical); `linalg.QRAlgorithm` (Householder→TQL symmetric-tridiagonal eigensolver — gates spectral-eigenfunction-expansion AND the Golub-Welsch-1969 quadrature-from-Jacobi-matrix construction that yields high-N Gauss-Legendre/Gauss-Lobatto/Gauss-Hermite nodes); `linalg.LU/Cholesky/MatVecMul` (dense linear-algebra usable for RBF-collocation Vandermonde-system + Chebyshev-differentiation-matrix application); `chaos.RK4Step / EulerStep` (the only time-stepper in repo — gates explicit pseudospectral time-marching but lacks the **stiff-accurate exponential time-differencing** ETDRK4 that spectral methods CRITICALLY need to handle the `e^{-νk²Δt}` linear-dispersion stiffness without fatal CFL `Δt ~ 1/k_max²`). **MASTER_PLAN slot 245 is named "Spectral methods for PDEs: Chebyshev, Fourier, RBF, pseudo-spectral"** and is **STRICT-TWIN of 244** (244 owns FD/FV/FEM low-order substrate; 245 owns spectral high-order — both are alternative spatial discretisations of the same canonical PDE classes), **STRICT-UPSTREAM of FOUR Block-C slots that explicitly name spectral substrate**: 242-new-spde "P9 2D-NSE + P10 Phi-4_3 spectral-2D-FFT-substrate ⊘blocked-on-245", 243-new-fpe "F15 Spectral-FP-Hermite-Galerkin ~180 LOC ⊘blocked-on-245", 244-new-pde-solvers "D30 SpectralElementMethod hp-adaptive Patera-1984 ⊘DEFER blocked-on-245", 248-new-multigrid "spectral-multigrid p-multigrid ⊘blocked-on-245" (Galerkin-projection between p-levels), 249-new-domain-decomp "Schwarz alternating with spectral local solvers ⊘blocked-on-245". Cross-link 215-new-compressed-sensing (sparse-Chebyshev-recovery), 235-new-functional-data (basis-expansion-FDA), 237-new-gaussian-process (RBF-kernel = Matern/SE in GP literature, R^d-radial Green's-function duality with Bjork-Bauer ill-conditioning).

**Summary line 2.** Twenty-six spectral-method primitives **S1-S26** totalling ~3,400 LOC organized as **(a) Tier-0 orthogonal-polynomial substrate ~660 LOC** (S1 `spectral/poly.go` Chebyshev `T_n(x)` + `U_n(x)` recurrence + Legendre `P_n(x)` + Hermite `He_n(x)`/Hermite-physicists `H_n(x)` + Laguerre `L_n(x)` + Jacobi `P_n^{(α,β)}(x)` ~180 LOC; all use stable three-term recurrence Bonnet/Boyd-2001 §A.2 NOT direct power-form), S2 `spectral/nodes.go` Chebyshev-Gauss-Lobatto `x_k = -cos(πk/N)` k∈[0,N] (closed-form, ~30 LOC) + Gauss-Legendre / Gauss-Lobatto-Legendre / Gauss-Hermite / Gauss-Laguerre nodes+weights via Golub-Welsch-1969 eigendecomposition of Jacobi matrix (calls `linalg.QRAlgorithm` ~120 LOC, lifts existing 2-5-point hard-coded `calculus.GaussLegendre` to arbitrary-N), S3 `spectral/quadrature.go` Clenshaw-Curtis-1960 quadrature on Chebyshev nodes via DCT (Trefethen-2008-SIREV "Is Gauss Quadrature Better Than Clenshaw-Curtis?") ~80 LOC, S4 `spectral/barycentric.go` Berrut-Trefethen-2004-SIREV "Barycentric Lagrange Interpolation" (the ONLY numerically-stable polynomial-interpolation form for high-N) ~100 LOC, S5 `spectral/dct.go` Discrete-Cosine-Transform DCT-I/II/III/IV via FFT trick (length-2N real FFT) gates Fast-Chebyshev-Transform ~100 LOC, S6 `spectral/dst.go` Discrete-Sine-Transform for Fourier-sine-series Dirichlet-BC ~50 LOC), **(b) Tier-1 Fourier-spectral keystone ~520 LOC** (S7 `spectral/fourier1d.go` PeriodicSpectral1D `u(x) → û_k → u'(x), u''(x), Δu, ...` calling existing `signal.FFT` for diagonalisation `∂_x → ik`, `Δ → -k²` + de-aliasing 2/3-rule (Orszag-1971) ~140 LOC, S8 `spectral/fourier2d.go` 2D-FFT via separable-1D-FFT (row-pass + column-pass) ~120 LOC; SHARED substrate with 242-P9-2D-NSE + 242-P10-Phi-4_3 + 243-F15-2D-spectral-FP, S9 `spectral/dealias.go` 2/3-rule + 3/2-padding for nonlinear-product evaluation ~60 LOC, S10 `spectral/filter.go` exponential-filter `σ(η) = exp(-α η^p)` Boyd-1996 + erfc-log Vandeven-1991 + Cesàro-1900 + spectral-viscosity Tadmor-1989 ~120 LOC; CRITICAL for non-smooth/shock-containing solutions where Gibbs-phenomenon corrupts entire grid, S11 `spectral/etdrk4.go` Cox-Matthews-2002-JCompPhys-176 Exponential-Time-Differencing-RK4 stiff-accurate-spectral-time-stepper ~140 LOC; THE canonical scheme for KS / KdV / Allen-Cahn / NLS spectral solvers — solves `u_t = Lu + N(u)` with `L`-diagonal-stiff via contour-integral on the unit circle Kassam-Trefethen-2005-SISC), **(c) Tier-2 Chebyshev-spectral non-periodic ~720 LOC** (S12 `spectral/cheb_diff.go` Chebyshev-differentiation-matrix `D_N` Trefethen-2000 §6 `D[i,j]=c_i(-1)^{i+j}/(c_j(x_i-x_j))` ~120 LOC, S13 `spectral/cheb_transform.go` Fast-Chebyshev-Transform (FCT) coefficient `a_n` ↔ values `u(x_k)` via DCT-I (Boyd-2001 §A.5) ~100 LOC, S14 `spectral/cheb_solve.go` Chebyshev-collocation Poisson 1D `u''=f` Dirichlet/Neumann + Helmholtz `u''+k²u=f` ~140 LOC, S15 `spectral/cheb_2d.go` Lynch-Rice-Thomas-1964 / Shen-1995 separable-Chebyshev 2D-Poisson via tensor-eigendecomposition ~180 LOC, S16 `spectral/cheb_tau.go` Lanczos-tau method residual-orthogonality with τ-rows for BCs Canuto-Hussaini-Quarteroni-Zang-2006 §3.3 ~100 LOC, S17 `spectral/cheb_galerkin.go` Galerkin-Chebyshev with bubble-functions `(1-x²)T_n` baking-in homogeneous-Dirichlet-BC Shen-1994-SISC ~80 LOC), **(d) Tier-3 RBF-meshfree ~480 LOC** (S18 `spectral/rbf.go` Multiquadric `√(r²+ε²)` + Inverse-Multiquadric `1/√(r²+ε²)` + Gaussian `exp(-(εr)²)` + Polyharmonic-spline `r^{2k+1}log(r)` + Wendland-compactly-supported (Wendland-1995-AdvCompMath) ~140 LOC, S19 `spectral/rbf_interp.go` RBF-interpolation dense-Vandermonde-system `Aλ = f` calling `linalg.LUSolve` + augmented-polynomial-tail (Buhmann-2003) ~120 LOC, S20 `spectral/rbf_fd.go` RBF-FD Fornberg-Flyer-2015-AdvCompMath generated-finite-differences (sparse stencils derived from local RBF interpolant; modern meshfree-PDE workhorse) ~140 LOC, S21 `spectral/rbf_pum.go` Partition-of-Unity-Method RBF Wendland-2002 ~80 LOC), **(e) Tier-4 spectral-element + frontier ~520 LOC** (S22 `spectral/sem.go` Spectral-Element-Method Patera-1984-JCompPhys hp-adaptive — combines FE-domain-decomposition with GLL-collocation per-element, gates 244-D30 + 247-mortar-FEM downstream; ~200 LOC, S23 `spectral/dgsem.go` Discontinuous-Galerkin-SEM Hesthaven-Warburton-2008 ~140 LOC ⊘DEFER, S24 `spectral/cheb3d.go` 3D-Chebyshev tensor-product ~80 LOC ⊘DEFER, S25 `spectral/sparse_grid.go` Smolyak-1963 sparse-grid for high-d-PDE ~120 LOC ⊘DEFER blocked-on-227-UQ, S26 `spectral/legendre_galerkin.go` Legendre-Galerkin Shen-Tang-Wang-2011 alternative Chebyshev avoiding `1/(1-x²)` weight ~100 LOC). **SINGULAR-FOUNDATIONAL S1+S2+S5+S12 substrate ~330 LOC** because EVERY spectral method calls one of {three-term recurrence, GLL nodes, DCT, Chebyshev-diff-matrix}. **SINGULAR-MOAT S11 ETDRK4 Cox-Matthews-2002 ~140 LOC** because exponential-time-differencing IS the only stable explicit time-stepper for stiff spectral semi-discretisations (KS / KdV / Burgers / NLS / Cahn-Hilliard) — every other Block-C consumer (242-KPZ, 243-FP-Hermite, etc.) NEEDS this OR pays factor `k_max²` CFL penalty (`k_max=64 → Δt=1/4096` versus ETDRK4 `Δt=O(1/k_max)` → 64x speedup at N=64). Kassam-Trefethen-2005-SISC pure-Go zero-dep absent everywhere. **SINGULAR-CUTTING-EDGE S20 RBF-FD Fornberg-Flyer-2015 ~140 LOC** because RBF-FD is the post-2010 meshfree-PDE workhorse (replaces classical-FD on irregular geometry; used in atmosphere-ocean modelling Shankar-Wright-2018, electromagnetics, neuroscience-cortex-meshes); zero-dep Go absent. **SINGULAR-PEDAGOGICAL S12+S14 Chebyshev-diff-matrix-on-Poisson ~260 LOC** the entry-point of Trefethen-2000 (Chapter 6, "Spectral Methods in MATLAB") — mature pedagogical pin against `u(x)=sin(πx)` solution to `u''=-π²sin(πx)` with `u(±1)=0` to machine-precision in N=24 nodes (versus N=2048 for FD-second-order to match — the famous "spectral accuracy" demonstration). **SINGULAR-RECURRING-FRONTIER S22 SEM Patera-1984 ~200 LOC** the canonical hp-adaptive method bridging FE-low-order (244) and pure-spectral-high-order (245); used in Nek5000 / Nektar++ / SpecFEM3D global-seismology; 244-D30 / 247-mortar-FEM / 248-spectral-multigrid / 249-domain-decomp ALL block on this substrate.

Recommended placement **NEW package `spectral/`** ~2,800 LOC (S1-S22 minus deferred S23-S26). Subpackage layout:

```
spectral/
  poly.go           # S1: Cheb/Legendre/Hermite/Laguerre/Jacobi recurrences
  nodes.go          # S2: GLL/GL/GH/GLag nodes via Golub-Welsch
  quadrature.go     # S3: Clenshaw-Curtis
  barycentric.go    # S4: Berrut-Trefethen-2004 stable interpolation
  dct.go            # S5: DCT-I/II/III/IV via FFT
  dst.go            # S6: DST
  fourier1d.go      # S7: PeriodicSpectral1D (uses signal.FFT)
  fourier2d.go      # S8: 2D-FFT separable
  dealias.go        # S9: 2/3-rule + 3/2-padding
  filter.go         # S10: exp/erfc/Cesaro/spectral-viscosity
  etdrk4.go         # S11: Cox-Matthews ETDRK4 (uses chaos.RK4 fallback)
  cheb_diff.go      # S12: Chebyshev differentiation matrix
  cheb_transform.go # S13: Fast Chebyshev Transform
  cheb_solve.go     # S14: collocation Poisson/Helmholtz 1D
  cheb_2d.go        # S15: 2D-Poisson tensor-eig (Lynch-Rice-Thomas)
  cheb_tau.go       # S16: Lanczos tau
  cheb_galerkin.go  # S17: Shen bubble-Chebyshev-Galerkin
  rbf.go            # S18: MQ/IMQ/Gaussian/PHS/Wendland kernels
  rbf_interp.go     # S19: dense-collocation interpolation
  rbf_fd.go         # S20: Fornberg-Flyer RBF-FD stencils
  rbf_pum.go        # S21: Partition-of-Unity RBF
  sem.go            # S22: Patera SEM
  legendre_galerkin.go # S26: Legendre-Galerkin (optional)
signal/
  (existing: fft.go IS the substrate for S7/S8 — NO change)
```

Rationale for **NEW** `spectral/` rather than nesting under `pde/spectral/` per 244 layout: spectral methods are NOT exclusively PDE solvers — orthogonal-polynomial recurrences (S1), Chebyshev approximation (S4+S12+S13), and RBF interpolation (S19) are independently used by **calculus** (high-order quadrature replacing 5-point cap), **prob** (Hermite-Galerkin chaos-expansion for UQ blocked-on-227), **signal** (Chebyshev-pseudospectral filter design), **linalg** (fast trigonometric transforms), **prob/copula** (Hermite-polynomial-of-normal copula). Nesting under `pde/` artificially couples the polynomial-substrate to PDE-solving when the polynomial-substrate is independently load-bearing. Compare 244 where `pde/parabolic/heat_explicit.go` is intrinsically a PDE construct. Sub-package precedent: top-level `signal/` (transforms), `optim/` (solvers), `prob/` (distributions). Spectral methods sit alongside as "high-order-basis-expansion" library, importable from `pde/`, `prob/`, `calculus/`, `signal/` without circularity.

**CANDOR.** Spectral methods rank **second only to MFG/SPDE in mathematical sophistication required** but the consumer-pull inside reality is concentrated upstream-of-other-Block-C-reviews (242 + 243 + 244 + 248 + 249 explicitly name the dependency) NOT inside the existing aicore/Pistachio/Oracle/Sentinel consumer stack. Only `signal/fft.go` already-callable consumers (Pistachio audio, RubberDuck, Oracle) get a direct boost. Trefethen-Embree-2005 / Boyd-2001 / Canuto-Hussaini-Quarteroni-Zang-2006 / Hesthaven-Gottlieb-Gottlieb-2007 are ~3,000-page textbooks on this single topic — reality cannot match (and should not try to match) a deal.II-spectral-element-extension or Nektar++ in scope. Recommendation: **ship Tier-0 + Tier-1 + Tier-2 + Tier-3 (S1-S21 ~2,400 LOC over 6 PRs / 12-15 engineer-days)** which saturates the four Block-C downstream slots' spectral substrate dependency and gives reality first-class Fourier-spectral / Chebyshev-spectral / RBF-meshfree on `T^d` and `[-1,1]^d`. **Defer S22 SEM + S23 DGSEM + S25 sparse-grid** to a future hp-FE slot once 244-FEM ships (S22 needs 244-D26 P1-2D triangular mesh as input). **Cheapest-1-day shippable**: S1 + S2 + S5 + S12 ~330 LOC ships orthogonal-polynomial-recurrences + GLL-nodes + DCT + Chebyshev-differentiation-matrix saturating R-CHEB-DIFF-MATRIX 4/4 against `f(x)=cos(kx)` differentiation pinned to `f'(x)=-k sin(kx)` to machine-precision-on-N=32-nodes (Trefethen-2000 §6 canonical demonstration). **Highest-leverage-1-week-unlock**: PR-1 + PR-2 + PR-3 ~1,560 LOC S1-S15 saturates Fourier + Chebyshev + ETDRK4 — gives reality FIRST-CLASS Fourier-spectral on torus + Chebyshev-collocation on `[-1,1]^d` + stiff-accurate exponential-time-differencing, simultaneously unblocking 242-P9-2D-NSE + 242-P10-Phi-4_3 + 243-F15-Hermite-Galerkin. **20 of 26 primitives unique to this slot** (S2-GLL-nodes shared with 244-D26-FE-quadrature; S12-cheb-diff-matrix substrate to 244-D30-SEM; S18-RBF-kernels overlap 237-GP-kernels conceptually but distinct module-boundary).

---

## 0. State at HEAD (2026-05-09, v0.10.0)

Repo-wide audit for Chebyshev / Legendre / Hermite / Laguerre / Jacobi orthogonal polynomials / Gauss-Lobatto / Gauss-Radau / pseudospectral / collocation / Galerkin / tau / spectral-element / DGSEM / RBF / multiquadric / Wendland / Polyharmonic-spline / RBF-FD / partition-of-unity / spectral-viscosity / exponential-filter / Vandeven-filter / Boyd-filter / spectral-element / hp-FEM / ETDRK / Cox-Matthews / Kassam-Trefethen / Lanczos-tau / Shen-Galerkin / barycentric-Lagrange surface — **zero callable matches** anywhere in `*.go` files outside review-corpus.

| Surface | Path | Spectral-method relevance |
|---|---|---|
| `signal.FFT / IFFT` | `signal/fft.go:49,101` | Cooley-Tukey radix-2 in-place FFT/IFFT — **THE Fourier-spectral substrate** — diagonalises `∂_x → ik`, `Δ → -k²` on torus; gates S7+S8+S9+S11+S13 (DCT calls FFT length-2N real-trick) |
| `signal.PowerSpectrum / FFTFrequencies` | `signal/fft.go:140,167` | Power spectrum + frequency-bin centres — usable for spectral-energy diagnostics |
| `signal.Hann/Hamming/Blackman` | `signal/window.go` | STFT-side-lobe-tapers — distinct from Boyd-filter (S10) which is for spectral-PDE-de-aliasing/Gibbs-suppression NOT FFT-leakage |
| `calculus.GaussLegendre` | `calculus/calculus.go:149` | **Hard-coded 2-5-point** Gauss-Legendre — usable for low-N Galerkin-inner-product but **inadequate for spectral N=32-256 typical**; S2 supersedes via Golub-Welsch |
| `calculus.NumericalDerivative / NumericalGradient` | `calculus/calculus.go:31,47` | First-order central-difference — **NOT** Chebyshev-differentiation-matrix (which is exact for polynomials of degree ≤ N) |
| `calculus.SimpsonsRule / TrapezoidalRule / RK4 / RK45` | `calculus/calculus.go` + `chaos/ode.go` | Low-order numerical-integration / ODE — gates explicit-pseudospectral time-marching but lacks ETDRK4 (S11) for stiff-spectral |
| `linalg.QRAlgorithm` | `linalg/eigen.go:20` | Householder→TQL symmetric-tridiagonal eigensolver — gates Golub-Welsch-1969 quadrature-from-Jacobi-matrix construction (S2 calls this for arbitrary-N Gauss-Legendre/Lobatto/Hermite/Laguerre nodes+weights) |
| `linalg.LUSolve / Cholesky / MatVecMul / MatMul` | `linalg/decompose.go,matrix.go` | Dense linear algebra — usable for RBF-collocation-Vandermonde (S19), Chebyshev-differentiation-matrix application (S12+S14 calls MatVecMul), tau-method lifting (S16) |
| `chaos.RK4Step / EulerStep / SolveODE` | `chaos/ode.go` | Deterministic ODE substrate — gates explicit pseudospectral time-marching but **missing stiff-accurate ETDRK4** (S11 ships) |
| `infogeo.LaplacianKernel` | `infogeo/mmd.go` | MMD-kernel `exp(-‖x-y‖/σ)` — name-collision with **Laplace-kernel-RBF** (one of S18 kernels but distinct module purpose) |
| `prob/copula/` Hermite-of-normal | -- | **ABSENT** — slot 234 partial overlap with S1-Hermite-recurrence |
| Chebyshev / Legendre / Hermite / Laguerre / Jacobi polynomial recurrences | -- | **ABSENT** — S1 ships |
| Gauss-Lobatto / Gauss-Radau / arbitrary-N Gauss-Legendre nodes+weights | -- | **ABSENT** — S2 ships (Golub-Welsch via existing `linalg.QRAlgorithm`) |
| Clenshaw-Curtis quadrature | -- | **ABSENT** — S3 ships |
| Barycentric-Lagrange interpolation (Berrut-Trefethen-2004) | -- | **ABSENT** — S4 ships (CRITICAL — only stable polynomial-eval at high-N) |
| DCT-I/II/III/IV + DST | -- | **ABSENT** — S5+S6 ship (calls `signal.FFT` length-2N real-trick) |
| Periodic-Fourier-spectral 1D / 2D / 3D | -- | **ABSENT** — S7+S8 ship (calls `signal.FFT`) |
| 2/3-rule de-aliasing + 3/2-padding | -- | **ABSENT** — S9 ships |
| Spectral filter (exponential / Vandeven-erfc / Cesaro / spectral-viscosity) | -- | **ABSENT** — S10 ships |
| ETDRK4 (Cox-Matthews-2002 / Kassam-Trefethen-2005) | -- | **ABSENT** — S11 ships |
| Chebyshev differentiation matrix `D_N` | -- | **ABSENT** — S12 ships |
| Fast Chebyshev Transform (DCT-I) | -- | **ABSENT** — S13 ships |
| Chebyshev-collocation Poisson/Helmholtz 1D | -- | **ABSENT** — S14 ships |
| Chebyshev-tensor 2D-Poisson (Lynch-Rice-Thomas-1964 / Shen-1995) | -- | **ABSENT** — S15 ships |
| Lanczos-tau method | -- | **ABSENT** — S16 ships |
| Shen-Galerkin-bubble-Chebyshev | -- | **ABSENT** — S17 ships |
| RBF kernels (MQ / IMQ / Gaussian / PHS / Wendland) | -- | **ABSENT** — S18 ships |
| RBF-collocation interpolation | -- | **ABSENT** — S19 ships |
| RBF-FD (Fornberg-Flyer-2015) | -- | **ABSENT** — S20 ships |
| Partition-of-Unity-Method RBF | -- | **ABSENT** — S21 ships |
| Spectral Element Method (Patera-1984) | -- | **ABSENT** — S22 ships (waits on 244-D26-P1-2D mesh) |
| Discontinuous Galerkin SEM | -- | **ABSENT** — S23 stub; defer (consumer-research-only) |
| Smolyak sparse grid | -- | **ABSENT** — S25 stub; defer blocked-on-227-UQ |

**Cross-import edges this slot creates.**
- `spectral → signal.FFT` (PRESENT) — S5 DCT calls length-2N real-FFT; S7/S8 PeriodicSpectral diagonalise via FFT.
- `spectral → linalg.QRAlgorithm` (PRESENT) — S2 Golub-Welsch eigendecomposes the Jacobi-matrix to recover Gauss-quadrature nodes (eigenvalues) + weights (first-component of eigenvectors squared).
- `spectral → linalg.LUSolve / MatVecMul` (PRESENT) — S12 applies `D_N` via MatVecMul; S19 RBF-interpolation solves dense Vandermonde via LUSolve.
- `spectral → chaos.RK4Step` (PRESENT) — S11 ETDRK4 fallback to RK4 when nonlinear-only (no stiff-linear-part).
- `pde/ → spectral` (FUTURE) — 244-D30-SEM imports spectral.SEM; 248-spectral-multigrid imports spectral.Galerkin-projection between p-levels.
- `prob/spde/ → spectral` (FUTURE 242) — P9-2D-NSE / P10-Phi-4_3 / P4-KPZ all import `spectral.PeriodicSpectral2D + spectral.Dealias + spectral.ETDRK4`.
- `prob/fpe/ → spectral` (FUTURE 243) — F15-Spectral-FP-Hermite-Galerkin imports `spectral.HermitePoly + spectral.GaussHermiteNodes`.

## 1. Twenty-six primitives (S1-S26) detailed

**S1 `spectral/poly.go` orthogonal-polynomial recurrences ~180 LOC.** Stable three-term recurrences: Chebyshev `T_{n+1} = 2x T_n - T_{n-1}, T_0=1, T_1=x`; Chebyshev-second-kind `U_{n+1} = 2x U_n - U_{n-1}, U_0=1, U_1=2x`; Legendre `(n+1)P_{n+1} = (2n+1)x P_n - n P_{n-1}, P_0=1, P_1=x`; Hermite-physicists `H_{n+1} = 2x H_n - 2n H_{n-1}, H_0=1, H_1=2x`; Hermite-probabilists `He_{n+1} = x He_n - n He_{n-1}`; Laguerre `(n+1)L_{n+1} = (2n+1-x)L_n - n L_{n-1}`; Jacobi `(α,β)` general-recurrence Bonnet-1894. **DO NOT** evaluate via direct power-form `Σ a_k x^k` (catastrophic cancellation N>20). API `spectral.ChebyshevT(n int, x float64) float64`, `spectral.ChebyshevTDeriv(n int, x float64) float64` via differentiated-recurrence, `spectral.ChebyshevTArray(n int, x float64, out []float64)` zero-alloc bulk-eval `T_0..T_n`. Ref: Boyd-2001 §A.2; Abramowitz-Stegun Ch.22; NIST DLMF §18.

**S2 `spectral/nodes.go` Gauss-quadrature nodes+weights ~120 LOC.** Golub-Welsch-1969-MathComp: nodes are eigenvalues of symmetric tridiagonal Jacobi matrix `J_N` whose entries are recurrence coefficients; weights are `μ_0 · (v_0^k)²` first component squared of normalized eigenvectors. Already-callable substrate `linalg.QRAlgorithm(A, n, eigenvalues, maxIter)` returns eigenvalues but **needs eigenvectors too** — either extend QRAlgorithm to return eigenvectors (add 80 LOC to linalg) OR reconstruct via golub-welsch-without-eigenvectors using continued-fraction/Sturm-sequence Walter-Gautschi-1968. Closed-form Chebyshev-Gauss `x_k = cos((2k-1)π/(2N))` and Chebyshev-Gauss-Lobatto `x_k = -cos(πk/N)` are uniform-density at endpoints — no eigendecomposition needed. API `spectral.GaussLegendreNodes(n int, nodes, weights []float64)`, `spectral.GaussLobattoLegendreNodes(n int, ...)`, `spectral.ChebyshevGaussLobattoNodes(n int, ...)`, `spectral.GaussHermiteNodes(...)`, `spectral.GaussLaguerreNodes(...)`. Ref: Golub-Welsch-1969 MathComp 23; Trefethen-2008-SIREV.

**S3 `spectral/quadrature.go` Clenshaw-Curtis ~80 LOC.** Quadrature on Chebyshev nodes `x_k = cos(πk/N)`: `∫_{-1}^{1} f(x) dx ≈ Σ w_k f(x_k)` where weights computed via DCT — for `N` even, `w_0 = w_N = 1/(N²-1)`, interior weights via inverse-cosine-transform of even-index Chebyshev coefficients. Trefethen-2008 "Is Gauss Quadrature Better Than Clenshaw-Curtis?" — answer: NO for analytic integrands of practical interest, and Clenshaw-Curtis has nested-points (N → 2N reuses N nodes) which Gauss-Legendre lacks. API `spectral.ClenshawCurtis(f func(float64) float64, n int) float64`, `spectral.ClenshawCurtisWeights(n int, weights []float64)` precomputed for repeated use.

**S4 `spectral/barycentric.go` Berrut-Trefethen-2004 ~100 LOC.** Stable polynomial-interpolation form: `p(x) = Σ (w_k / (x-x_k)) f_k / Σ (w_k / (x-x_k))` where barycentric weights `w_k` depend on node-set. For Chebyshev-Gauss-Lobatto: `w_k = (-1)^k δ_k` with `δ_0 = δ_N = 1/2`, else `δ_k = 1`. **CRITICAL**: standard Lagrange-form `Σ ℓ_k(x) f_k` is O(N²) per-eval with catastrophic-cancellation at high N; barycentric-form is O(N) per-eval and stable. Higham-2004 numerical-stability proof. API `spectral.BarycentricInterp(nodes, weights, values []float64, x float64) float64`, `spectral.BarycentricChebyshevWeights(n int, out []float64)`. Ref: Berrut-Trefethen-2004 SIREV 46(3).

**S5 `spectral/dct.go` Discrete Cosine Transform ~100 LOC.** DCT-I/II/III/IV via length-2N real-FFT trick: DCT-II of `x[0..N-1]` ↔ first-N real-parts of FFT of `[x[0], x[1], ..., x[N-1], x[N-1], ..., x[0]]` (mirror-pad). API `spectral.DCT_II(in, out []float64)`, `spectral.DCT_III(in, out []float64)` (inverse of DCT-II up to scaling), `spectral.IDCT(in, out []float64)`. Calls existing `signal.FFT`. Ref: Makhoul-1980-IEEE; Trefethen-2000 §8 footnote.

**S6 `spectral/dst.go` Discrete Sine Transform ~50 LOC.** Sister-of-DCT for Fourier-sine-series Dirichlet-BC; same length-2N FFT trick with imaginary-part reading.

**S7 `spectral/fourier1d.go` PeriodicSpectral1D ~140 LOC.** API `func PeriodicSpectralDeriv(u []float64, L float64, order int, out []float64)` computes m-th-derivative `∂^m u / ∂x^m` on uniform-grid `x_j = jL/N` j∈[0,N) periodic. Implementation: FFT `u → û_k`, multiply by `(ik)^m` where `k = 0,1,...,N/2,-N/2+1,...,-1`, IFFT. **Note**: for odd-order derivative on real-input the Nyquist mode `k=N/2` must be zeroed (Trefethen-2000 §3 footnote); else odd-derivative output is non-real. `func PeriodicSpectralPoissonSolve(f []float64, L float64, out []float64)` Poisson `Δu = f` on torus: `û_k = -f̂_k / k²` (k=0 mode requires solvability `∫f=0`; set `û_0=0` for mean-zero solution). API `func PeriodicSpectralAdvect(u, c []float64, L, dt float64, out []float64)` advection step with semi-Lagrangian-or-spectral. Calls `signal.FFT` + `signal.IFFT`.

**S8 `spectral/fourier2d.go` 2D-FFT separable ~120 LOC.** 2D-FFT via row-pass + column-pass of existing 1D `signal.FFT`. **THE substrate for 242-P9 2D-stochastic-NSE + 242-P10 Phi-4_3 + 243-F15 2D-spectral-FP-Hermite**. API `func FFT2D(real, imag []float64, rows, cols int)` in-place row-major. **Note**: 3D-FFT trivially extends but defer (S24).

**S9 `spectral/dealias.go` 2/3-rule + 3/2-padding ~60 LOC.** Orszag-1971: pseudospectral evaluation of nonlinear-product `(uv)(x)` via `IFFT( û * v̂ )` aliases modes `|k| > N/3` back into `|k| < N/3` causing accumulation. Fix: zero Fourier-modes `|k| > N/3` before transform. Alternative 3/2-padding: pad `û` to length `3N/2`, IFFT, multiply pointwise, FFT, truncate back to N — exact-product no-aliasing (Trefethen-2000 §11). API `spectral.Dealias23(u []float64)`, `spectral.PadFor32(u []float64, padded []float64)`.

**S10 `spectral/filter.go` spectral filters ~120 LOC.** Boyd-1996 exponential `σ(η) = exp(-α η^p)` with `η = k/k_max`, `α = -log(machine_eps) ≈ 36`, `p` filter-order (typical `p=8-12`); Vandeven-1991-erfc-log: `σ(η) = (1/2) erfc(2√p · √(-log(η)·η))` smooth-cutoff at `η=1`; Cesàro-1900 `σ(η) = 1 - η`; Tadmor-1989 spectral-viscosity `σ(η) = ε^{2s} k^{2s} m(k/m)` with viscosity-strength function. API `spectral.ExpFilter(order int, kmax int, sigma []float64)` precompute; `spectral.ApplyFilter(uhat []float64, sigma []float64)` zero-alloc. **CRITICAL** for non-smooth solutions (shock-containing / Gibbs-corrupted) where filter ON / filter OFF determines stability.

**S11 `spectral/etdrk4.go` Cox-Matthews-2002 ETDRK4 ~140 LOC.** For semi-discrete `u_t = Lu + N(u)` with `L`-diagonal-stiff (linear-dispersion in spectral-space) and `N`-non-stiff (nonlinear-product): explicit RK4 fails CFL `Δt ~ 1/‖L‖`. ETDRK4 solves linear-part exactly via integrating-factor `e^{LΔt}` and computes `φ_1, φ_2, φ_3` functions via contour-integral on unit-circle Kassam-Trefethen-2005-SISC (avoiding cancellation in `(e^{LΔt}-1)/L` for small `LΔt`). API `func ETDRK4Step(L, u []float64, N func(u []float64, out []float64), dt float64, work [][]float64)` work-buffers preallocated. **THE canonical scheme for KS / KdV / Allen-Cahn / NLS / Cahn-Hilliard / KPZ spectral solvers** — Trefethen-2000 §10, used by every Fornberg / Boyd / Trefethen-school paper.

**S12 `spectral/cheb_diff.go` Chebyshev differentiation matrix `D_N` ~120 LOC.** Trefethen-2000 §6 closed-form: `D[i,j] = (c_i / c_j) (-1)^{i+j} / (x_i - x_j)` for `i≠j` with `c_0=c_N=2, c_k=1` else; `D[i,i] = -x_i / (2(1-x_i²))` for `0<i<N`; `D[0,0] = (2N²+1)/6, D[N,N] = -(2N²+1)/6`. **CAUTION**: `D[0,0]` and `D[N,N]` use Bayliss-Class-Turkel-1995 negative-summation-trick to avoid cancellation: `D[i,i] = -Σ_{j≠i} D[i,j]`. API `spectral.ChebyshevDiffMatrix(n int, D []float64)` returns `(n+1)×(n+1)` matrix; `spectral.ChebyshevDiffApply(D, u, du []float64)` calls `linalg.MatVecMul`.

**S13 `spectral/cheb_transform.go` Fast Chebyshev Transform ~100 LOC.** Coefficients `a_n` ↔ values `u(x_k)` on Chebyshev-Gauss-Lobatto nodes via DCT-I (S5): `u(x_k) = Σ a_n T_n(x_k) = Σ a_n cos(nπk/N)` so values ↔ coefficients via DCT-I. API `spectral.ChebyshevValuesToCoeffs(values, coeffs []float64)`, `spectral.ChebyshevCoeffsToValues(coeffs, values []float64)`. Boyd-2001 §A.5.

**S14 `spectral/cheb_solve.go` Chebyshev-collocation 1D ~140 LOC.** Poisson `u''(x) = f(x)` on `[-1,1]` Dirichlet `u(±1)=g_±`: assemble `D_N²`, replace first/last rows with BC `u(x_0)=g_+, u(x_N)=g_-`, solve dense `linalg.LUSolve`. Helmholtz `u''+k²u=f` similar. **Spectral-accuracy demonstration**: `f(x)=-π²sin(πx), u(±1)=0` exact-solution `u=sin(πx)` — N=24 Chebyshev nodes converge to machine-precision; FD-2nd-order needs N≈2048. API `spectral.ChebPoisson1D(f []float64, gLeft, gRight float64, u []float64)`.

**S15 `spectral/cheb_2d.go` 2D-Poisson tensor-eigendecomp ~180 LOC.** Lynch-Rice-Thomas-1964 / Shen-1995: 2D-Poisson `u_xx + u_yy = f` on `[-1,1]²`. Discretise `D_N² ⊗ I + I ⊗ D_N² = F` and diagonalise via eigendecomposition `D_N² = V Λ V^{-1}` once, transform `g = V^{-1} F V^{-T}`, solve diagonal system `g_{ij} / (λ_i + λ_j)`, transform back `u = V g V^T`. O(N³) instead of O(N⁶) naive dense-LU. API `spectral.ChebPoisson2D(f, u []float64, n int)`.

**S16 `spectral/cheb_tau.go` Lanczos-tau ~100 LOC.** Tau-method: solve `Lu = f` in spectral-space, replacing last `m` rows of residual-system with BC equations multiplied by `τ` parameters that absorb the residual. Canuto-Hussaini-Quarteroni-Zang-2006 §3.3. Handy for non-trivial BCs not fitting collocation cleanly.

**S17 `spectral/cheb_galerkin.go` Shen-1994 ~80 LOC.** Galerkin basis `φ_n(x) = T_n(x) - T_{n+2}(x)` automatically satisfies `u(±1)=0` (homogeneous-Dirichlet); for non-homogeneous BC subtract a polynomial-lift. Yields banded mass+stiffness matrices (sparser than collocation D_N²). Shen-1994-SISC.

**S18 `spectral/rbf.go` RBF kernels ~140 LOC.** φ(r) where r = ‖x-y‖: Multiquadric `√(r²+ε²)`, Inverse-Multiquadric `1/√(r²+ε²)`, Gaussian `exp(-(εr)²)`, Polyharmonic-spline `r^{2k+1}` (odd) or `r^{2k}log(r)` (even, "thin-plate-spline" k=1), Wendland-compactly-supported (e.g. `(1-εr)_+^4 (4εr+1)`). API `spectral.RBFGaussian(r, eps float64) float64`, ... + `spectral.RBFEvalAll(kernel int, eps float64, centers, x []float64, out []float64)` bulk-eval. Ref: Buhmann-2003-CUP "Radial Basis Functions".

**S19 `spectral/rbf_interp.go` RBF interpolation ~120 LOC.** Given centres `{x_k}_{k=1}^N` and values `{f_k}` solve `Σ_j λ_j φ(‖x_k - x_j‖) = f_k` dense N×N Vandermonde-system via `linalg.LUSolve`. Augmented-polynomial-tail: add `+ Σ p_α(x_k)` low-order polynomial term + side-condition `Σ_j λ_j p_α(x_j)=0` to ensure convergence-for-PHS (Buhmann-2003 §6). Bjork-Bauer-1970 ill-conditioning warning: for Gaussian/MQ, condition-number scales `O(ε^{-N})` so small-ε high-N is numerically-fatal — RBF-QR (Fornberg-Larsson-Flyer-2011) restores stability but +200 LOC; defer to S20-RBF-FD which avoids the global Vandermonde entirely.

**S20 `spectral/rbf_fd.go` RBF-FD Fornberg-Flyer-2015 ~140 LOC.** Per node `x_k`, find `n_k` nearest-neighbours, fit local RBF interpolant, extract coefficients of differential-operator (e.g. Laplacian) at `x_k` as **sparse stencil**. Combines FD-sparsity with RBF-meshfree-flexibility. Modern PDE workhorse on irregular-domain / scattered-nodes — used in atmosphere-ocean-modelling Shankar-Wright-2018, electromagnetics, neuroscience-cortex-meshes. API `spectral.RBFFDStencil(centers, target []float64, neighbors []int, op int, weights []float64)`. Ref: Fornberg-Flyer-2015-AdvCompMath "Fast generation of 2-D node distributions for mesh-free PDE discretizations"; Bayona-2019-JCompPhys.

**S21 `spectral/rbf_pum.go` Partition-of-Unity RBF ~80 LOC.** Wendland-2002: cover domain by overlapping patches `{Ω_j}` with PoU functions `{ψ_j}` summing to 1; local RBF interpolant on each patch; global `s(x) = Σ_j ψ_j(x) s_j(x)`. Reduces conditioning + cost of full RBF-collocation.

**S22 `spectral/sem.go` Spectral Element Method Patera-1984 ~200 LOC.** Decompose domain into elements, on each element use GLL-collocation with degree-p polynomials. Yields hp-adaptive method bridging FE-low-order (244) and pure-spectral (245). Used in Nek5000 / Nektar++ / SpecFEM3D global-seismology (the world's largest seismic-wave simulations run on this). **WAITS on 244-D26 P1-2D triangular mesh** (or quad-mesh) as input. API `spectral.SEMQuad(elements, p int, ...)` quad-element 2D first.

**S23 `spectral/dgsem.go` Discontinuous-Galerkin SEM ~140 LOC ⊘DEFER.** Hesthaven-Warburton-2008 — research-frontier; defer.

**S24 `spectral/cheb3d.go` 3D-Chebyshev tensor-product ~80 LOC ⊘DEFER.** Trivial extension of S15 to 3D; defer until consumer-pull emerges.

**S25 `spectral/sparse_grid.go` Smolyak-1963 ~120 LOC ⊘DEFER blocked-on-227-UQ.** Sparse-grid for high-dimensional integration / approximation; defer to UQ-slot 227 owner.

**S26 `spectral/legendre_galerkin.go` Legendre-Galerkin Shen-Tang-Wang-2011 ~100 LOC.** Alternative to Chebyshev for problems where `1/(1-x²)` weight is awkward (e.g. kinetic-equation moments). Optional.

## 2. Roadmap (6 PRs, ~12-15 engineer-days, ~2,400 LOC for S1-S21)

| PR | Slug | LOC | Days | Saturates | Unblocks |
|---|---|---|---|---|---|
| PR-1 | spectral-substrate | 410 | 2 | S1+S2+S5+S12 + R-CHEB-DIFF-MATRIX 4/4 | downstream-DCT/recurrence consumers |
| PR-2 | fourier-spectral | 480 | 2 | S7+S8+S9+S10+S11 + R-FOURIER-POISSON 4/4 + R-ETDRK4-KS 3/3 | 242-P9-2D-NSE, 242-P10-Phi4_3, 243-F15 |
| PR-3 | chebyshev-solve | 340 | 2 | S6+S13+S14+S15 + R-CHEB-POISSON-SPECTRAL-ACC 4/4 | calculus-high-order-quad |
| PR-4 | quadrature-interp | 280 | 1.5 | S3+S4+S16+S17 | calculus.GaussLegendre N>5, prob/conformal |
| PR-5 | rbf-meshfree | 480 | 3 | S18+S19+S20+S21 + R-RBF-INTERP-CONVERGENCE 3/3 | 237-GP-kernels-cross-link |
| PR-6 | sem-frontier | 200 | 2 | S22 | 244-D30-SEM, 247-mortar-FEM, 248-spectral-multigrid |

**Cheapest-1-day shippable**: PR-1 alone (410 LOC) saturates the orthogonal-polynomial-recurrence + GLL-nodes + DCT + Chebyshev-differentiation-matrix substrate.

**Recommended ship-order**: PR-1 → PR-2 (unlocks 242 + 243 downstream) → PR-3 → PR-4 → PR-5 → PR-6.

## 3. Golden-file pins

- **R-CHEBYSHEV-RECURRENCE 4/4**: `T_n(cos θ) = cos(nθ)` exact-identity at θ=π/4 for n=0..50 (closed-form trigonometric pin); `T_n(±1) = (±1)^n`; `T_n(0) = 0` if n odd else `(-1)^{n/2}`. Cross-language pin to 1e-13.
- **R-CHEB-DIFF-MATRIX 4/4**: `f(x)=cos(kx)` differentiation pinned to `f'(x)=-k sin(kx)` at N=32 GLL nodes for k=1,2,3,5; ‖D_N f - f'‖_∞ < 1e-12 (machine-precision-class).
- **R-FOURIER-POISSON 4/4**: `Δu = f` on `[0,2π]` periodic with `f(x) = sin(3x) + cos(5x)` exact-solution `u(x) = -sin(3x)/9 - cos(5x)/25` pinned at N=64 modes.
- **R-CHEB-POISSON-SPECTRAL-ACC 4/4**: `u'' = -π²sin(πx), u(±1)=0` exact-solution `u=sin(πx)` — Chebyshev-collocation N=24 to 1e-14 vs FD-2nd-order N=2048 to 1e-6 (the famous spectral-vs-FD pedagogical pin Trefethen-2000 §6).
- **R-ETDRK4-KS 3/3**: Kuramoto-Sivashinsky `u_t = -uu_x - u_{xx} - u_{xxxx}` on `[0,32π]` periodic with random IC; energy-spectrum saturation at `t=150` matches Kassam-Trefethen-2005 fig.3 reference-spectrum.
- **R-RBF-INTERP-CONVERGENCE 3/3**: Franke-1979 test-function on `[0,1]²` scattered-nodes — Multiquadric ε=2 N=100 nodes converges to machine-precision in interior; verify O(h^d)-density convergence rate.
- **R-GAUSS-QUADRATURE-EXACTNESS 4/4**: N-point Gauss-Legendre exact for polynomials degree ≤ 2N-1 (textbook); pin `∫_{-1}^{1} x^k dx` for k=0..2N-1 to 1e-15.

## 4. Cross-slot dependencies

**Strict-upstream consumers of slot 245 spectral substrate:**
- 242-new-spde: P4 KPZ-spectral, P9 2D-NSE, P10 Phi-4_3 → all need S7+S8+S9+S10+S11 (Fourier-spectral + 2D-FFT + dealias + filter + ETDRK4).
- 243-new-fpe: F15 Spectral-FP-Hermite-Galerkin → needs S1-Hermite-poly + S2-Gauss-Hermite-nodes.
- 244-new-pde-solvers: D30 SEM ⊘DEFERRED in 244 → needs S22-SEM here.
- 247-new-mortar-fem: spectral-mortar elements → needs S22-SEM.
- 248-new-multigrid: spectral-multigrid p-multigrid → needs S22-SEM (Galerkin-projection between p-levels).
- 249-new-domain-decomp: Schwarz with spectral local solvers → needs S14+S22.

**Strict-twin of 244-PDE-solvers:** 244 owns FD/FV/FEM low-order; 245 owns Chebyshev/Fourier/RBF/pseudo-spectral high-order. Same canonical-PDE-classes (parabolic / elliptic / hyperbolic), alternative spatial discretisations.

**Cross-link not strict-dependency:**
- 215-compressed-sensing: sparse Chebyshev / Fourier recovery — uses S1+S5+S12.
- 235-functional-data: basis-expansion-FDA can use S1+S4 polynomial+barycentric.
- 237-GP regression: SE-kernel ↔ Gaussian-RBF (S18) module-boundary distinct, conceptually identical.
- 234-copula: Hermite-of-normal copula uses S1-Hermite recurrence.
- 117-Box-Muller: independent (no dependency).
- 227-UQ: S25 sparse-grid blocked-on-227.

**Cross-cuts existing single-source-of-truth concerns:**
- `calculus.GaussLegendre` (capped at 5 points) becomes a wrapper around `spectral.GaussLegendreNodes(n, ...)` arbitrary-N — single-source-ownership migrates to `spectral/`. The 5-point hard-coded values are kept as fast-path constants (zero-alloc lookup) but spec'd via `spectral/`.
- `signal.PowerSpectrum` is a Fourier-spectral-energy-diagnostic — can stay in `signal/` or move; recommend stay (signal is the right consumer-facing-home).

## 5. Performance / no-allocations-in-hot-paths

Reality's key-design-rule #3: "No allocations in hot paths. Functions accept output buffers. Pistachio calls these at 60 FPS." For spectral methods at N=64-256:
- S7 PeriodicSpectralDeriv: `(in []float64, ..., out []float64)` zero-alloc; FFT-twiddles precomputed in package-level cache or caller-allocated.
- S11 ETDRK4: caller pre-allocates `work [][]float64` work-buffers (8 buffers of length-N for the four RK4-stage-residuals + φ-functions).
- S12 ChebyshevDiffMatrix: precompute `D_N` once (caller-allocated `(n+1)×(n+1)` buffer), reuse for all timesteps.
- S19 RBF-interp: dense N×N matrix is the bottleneck at large-N — recommend Wendland-compact-support kernel (S18) for `O(N)` sparse case.
- S22 SEM: per-element work-buffers (degree-p²-LOC) reused across element-loop.

## 6. References (canonical)

- Trefethen, **Spectral Methods in MATLAB**, SIAM 2000. Chebyshev / Fourier / pseudospectral pedagogical-canon.
- Boyd, **Chebyshev and Fourier Spectral Methods**, 2nd-ed, Dover 2001 (free-PDF). 700-page reference; §A.2 polynomial-recurrences, §11 filtering.
- Canuto-Hussaini-Quarteroni-Zang, **Spectral Methods: Fundamentals in Single Domains**, Springer 2006. Tau / Galerkin / collocation rigour.
- Hesthaven-Gottlieb-Gottlieb, **Spectral Methods for Time-Dependent Problems**, CUP 2007.
- Cox-Matthews, "Exponential time differencing for stiff systems," **JCompPhys** 176, 2002.
- Kassam-Trefethen, "Fourth-order time-stepping for stiff PDEs," **SISC** 26(4), 2005. ETDRK4 contour-integral fix.
- Berrut-Trefethen, "Barycentric Lagrange interpolation," **SIAM Review** 46(3), 2004.
- Trefethen, "Is Gauss Quadrature Better Than Clenshaw-Curtis?", **SIAM Review** 50(1), 2008.
- Golub-Welsch, "Calculation of Gauss quadrature rules," **MathComp** 23, 1969.
- Fornberg-Flyer, "Fast generation of 2-D node distributions for mesh-free PDE discretizations," **AdvCompMath** 41, 2015.
- Buhmann, **Radial Basis Functions: Theory and Implementations**, CUP 2003.
- Patera, "A spectral element method for fluid dynamics," **JCompPhys** 54, 1984.
- Shen, "Efficient spectral-Galerkin method I: Direct solvers for second- and fourth-order equations using Legendre polynomials," **SISC** 15, 1994.
- Wendland, **Scattered Data Approximation**, CUP 2005.
- Hesthaven-Warburton, **Nodal Discontinuous Galerkin Methods**, Springer 2008.

---

End of agent 245 review. ~370 lines.

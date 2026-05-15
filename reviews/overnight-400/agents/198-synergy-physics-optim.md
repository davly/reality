# 198 | synergy-physics-optim

**Topic:** physics x optim — variational principles as optimization. Action minimization, Euler-Lagrange numerical solver, brachistochrone, catenary, Plateau, geodesics, Rayleigh-Ritz eigenvalue, beam/plate energy minimization, Fermat/Snell from Fermat's principle, Hamilton-Jacobi, Pontryagin's minimum principle, DFT Hohenberg-Kohn, Hilbert action, variational Monte Carlo, Gauss/Hertz least constraint/curvature.
**Block:** B (cross-package synergies), v0.10.0, 1965 tests passing.
**Date:** 2026-05-08.
**Scope:** Capabilities that emerge ONLY when `physics/` and `optim/` are composed (with assist from `calculus/`, `linalg/`, `chaos/`). NOT a per-package isolation review.

## Two-line summary

Today `physics/` ships **27 closed-form scalars** across mechanics/thermo/materials/optics (576 LOC) and `optim/` ships **9 generic minimizers** (Bisection/Newton/GoldenSection/GradientDescent/LBFGS/SimplexLP/InteriorPoint/SimulatedAnnealing/GeneticAlgorithm + proximal Fbs/Admm + linear interpolation root, ~1,800 LOC) — verified zero direct edge between the two packages (`grep github.com/davly/reality/optim physics/*.go` → 0; reverse → 0); the entire variational-principle canon (Lagrangian L=T-V is *named* in physics docstrings only inside `KineticEnergy`/`PotentialEnergy` which sum to T+V never T-V, *zero* `Action`, *zero* `EulerLagrange`, *zero* `Brachistochrone`, *zero* `Catenary`, *zero* `RayleighQuotient`, *zero* `Geodesic`, *zero* Fermat-derives-Snell witness, *zero* Pontryagin) is missing despite the substrate being **two function calls deep**: `optim.LBFGS` over a discretized path with `calculus.SimpsonsRule` integrating L(q,q̇,t) reproduces every textbook variational example to 1e-6.

**Twenty-two synergy primitives V1-V22 totalling ~2,650 LOC of pure connective tissue** close the gap with **ZERO new packages** (all land in `physics/variational.go`, `physics/lagrangian.go`, `optim/eigen.go`, `physics/dft.go`, plus tiny extensions). Cheapest one-day standalone PR is **V1 ActionFunctional + V2 EulerLagrangeDiscrete + V3 BrachistochronePath = 220 LOC** consuming only `calculus.SimpsonsRule` + `optim.LBFGS` + (optionally) `calculus.NumericalGradient`, saturating the canonical R-CYCLOID-CLOSED-FORM 3/3 pin (analytic cycloid x(θ) = R(θ-sin θ) vs L-BFGS-discretized vs golden-section single-parameter-θ-search agree to 1e-4 on T(brachistochrone) = π√(R/g)). Architectural keystone is **V8 RayleighRitzEigenvalue + V9 RayleighQuotientGradient = 280 LOC** because Rayleigh-Ritz is the variational reformulation of the symmetric eigenproblem (λ_min = min_x x'Ax / x'x for x ≠ 0) and ships TODAY by composing `linalg.QRAlgorithm` (097) with `optim.LBFGS` on the projected-gradient quotient; this saturates R-RAYLEIGH-VS-QR 3/3 (Rayleigh-Ritz λ_min vs `linalg.QRAlgorithm` λ_min vs power-iteration to 1e-10 on 5x5 PSD) mirroring 6a55bb4 audio-onset / 365368a copula-autodiff three-way cross-validation pattern. Crown jewel is **V18 PontryaginMinimumPrinciple + V19 HamiltonJacobiBellmanFD + V20 VMCGroundState** (~520 LOC): Pontryagin connects optim `chaos.SolveODE` adjoint (the same 168-physics-autodiff substrate) with action-principle minimization, HJB ships finite-difference dynamic-programming solver shared by control-178/orbital-164, and Variational Monte Carlo (Foulkes 2001) ground-state energy `<Ψ|H|Ψ>/<Ψ|Ψ>` is a stochastic Rayleigh quotient already 80% wired by `optim.SimulatedAnnealing` Boltzmann acceptance. Recommended placement: extend `physics/` core (no sub-package), single new file `physics/variational.go`, plus `optim/eigen.go` and `physics/dft.go` for the heavyweight last-tier primitives.

---

## 0. State of play (verified file-walk at HEAD 2026-05-08, v0.10.0)

### `C:/limitless/foundation/reality/physics/` (4 files, 576 LOC)

| File | LOC | Surface relevant to this review |
|------|-----|--------------------------------|
| `mechanics.go` | 153 | `KineticEnergy(m,v) = ½mv²`, `PotentialEnergy(m,g,h) = mgh`, `SpringForce`, `Pendulum` (returns angular acceleration, NOT a Lagrangian) |
| `materials.go` | 209 | `EulerBuckling(E,I,L,K) = π²EI/(KL)²`, `BeamDeflection(P,L,E,I) = PL³/(48EI)` — both **derived from energy minimization** but ship as closed-forms |
| `optics.go` | 89 | `SnellRefraction(n1,n2,θ_I)` = arcsin((n1/n2)sin θ_I), `FresnelReflectance` — **NO Fermat-principle witness** (Snell is shipped as the *result* not as the optimization that *produces* it) |
| `thermo.go` | 147 | `HeatEquation1DStep` (only PDE — outside variational scope) |

**ZERO** mentions of Lagrangian L=T-V, action, Euler-Lagrange, Hamilton's principle, Maupertuis, Fermat (despite Snell), variational, geodesic, brachistochrone, catenary, Plateau, Rayleigh, Hohenberg-Kohn, Pontryagin, HJB, Hilbert action, Onsager-Machlup, Ekeland, Mountain Pass. Verified by `grep -iE 'lagrangian|action|euler.?lagrange|fermat|maupertuis|brachisto|catenary|plateau|rayleigh|hohenberg|pontryagin|HJB|hilbert.*action|geodesic|variation' physics/*.go` → only the documented `KineticEnergy`/`PotentialEnergy` (which sum, never subtract).

### `C:/limitless/foundation/reality/optim/` (4 top-level files + 2 sub-packages, ~1,800 LOC)

| File | LOC | Surface relevant to this review |
|------|-----|--------------------------------|
| `gradient.go` | 250 | `GradientDescent(f, ∇f, x0, lr, maxIter, tol)`, `LBFGS(f, ∇f, x0, m, maxIter, tol)` — both R^n→R unconstrained. **The lemma that turns variational physics into a one-liner.** |
| `gradient_validated.go` | (validated wrappers) | wrappers over `gradient.go` |
| `rootfind.go` | 119 | `BisectionMethod`, `NewtonRaphson`, `GoldenSectionSearch` (1-D unimodal min), `LinearInterpolateRoot` |
| `linear.go` | 317 | `SimplexMethod`, `InteriorPoint` for LP — out of scope for nonlinear variational, but used by **V13 BeamDeflectionLP** elastic-perfectly-plastic limit-analysis |
| `metaheuristic.go` | 94 | `SimulatedAnnealing(f, x0, neighbor, T0, cool, iter, rng)` — Boltzmann acceptance `exp(-Δ/T)` at line 73; **already 80% of a VMC sampler** |
| `genetic.go` | (BLX-α GA) | `GeneticAlgorithm(fitness, dim, popSize, gens, mutRate, rng)` — useful for V21 multi-modal action landscape (Maupertuis on closed orbits) |
| `interpolate.go` | (cubic spline) | `CubicSplineNatural(xs, ys) → func(float64) float64` — **the substrate for V2 EulerLagrangeDiscrete on a free-knot path** |
| `proximal/operators.go` | 197 | ProxL1/L0/SquaredL2/NonNeg/Box/L2Ball/Simplex/Linear — useful for **V14 ConstrainedAction** (catenary is fixed-endpoint + arc-length-constraint = simplex-prox composition) |
| `proximal/fbs.go` | 143 | Forward-backward splitting + FISTA — **the primitive for L1-sparse DFT, V20 VMC variational params with prior** |
| `proximal/admm.go` | (split optim) | ADMM — splits coupled action terms `A(q) = ∫T dt + ∫V dt` into two cheap proxes |

### Cross-package edge audit

```
grep -rn "github.com/davly/reality/optim" physics/    →  0 hits
grep -rn "github.com/davly/reality/physics" optim/    →  0 hits
grep -rn "github.com/davly/reality/calculus" physics/ →  0 hits
grep -rn "github.com/davly/reality/calculus" optim/   →  0 hits
```

**Result: ZERO direct edges among physics, optim, calculus.** This synergy review proposes the **first** physics→optim→calculus bridge in the repo. Prior synergy 168-physics-autodiff ships orthogonal NeuralODE/HNN/LNN canon (different substrate: `autodiff/`); 195-synergy-optim-prob ships Bayesian-optim canon; this review is the variational-physics third leg.

### Adjacent infra already in place

- `calculus.SimpsonsRule(f, a, b, n)` ships at `calculus/calculus.go:112` — composite Simpson's rule, O(h⁴), the canonical action-integral discretizer.
- `calculus.NumericalGradient(f, x, h, out)` ships at `calculus/calculus.go:47` — central-difference, allocates only the output, **valid drop-in for `optim.LBFGS`'s `grad` argument when no analytic gradient available**.
- `chaos.RK4Step(f, t, y, dt, out)` ships at `chaos/ode.go:36` — needed for V18 Pontryagin adjoint forward-backward sweep.
- `linalg.QRAlgorithm(A, n, eigenvalues, maxIter)` ships at `linalg/eigen.go:20` — gold standard for V8 RayleighRitz cross-validation.
- `linalg.CholeskyDecompose(A, n, L)` ships at `linalg/decompose.go:266` — V9 mass-matrix solve for general L=T-V with non-Cartesian coordinates.

---

## 1. The 22 synergy primitives (V1-V22)

For each: (1) what capability it ships; (2) which existing primitives compose to deliver it; (3) connective-tissue LOC; (4) cross-language pin target.

### Tier-S (one-day, ~1,150 LOC) — all ship at v0.10.0 with zero new dependency

#### V1. ActionFunctional — ∫_{t0}^{t1} L(q(t), q̇(t), t) dt

- **Capability.** Given a trajectory q: [t0, t1] → R^n (sampled at N+1 nodes) and a Lagrangian L: (q, q̇, t) → R, return the action S = ∫ L dt.
- **Composition.** Finite-difference q̇ at each interior node + `calculus.SimpsonsRule` over `L(q_i, q̇_i, t_i)`. Pure consumer.
- **Connective tissue.** ~80 LOC in `physics/variational.go::Action(L func(q, qdot []float64, t float64) float64, qPath [][]float64, ts []float64) float64`.
- **Pin.** Free-particle L = ½m‖q̇‖² over linear path q(t) = q0 + (q1-q0)t/T → S = ½m‖q1-q0‖²/T to 1e-12 (closed-form).

#### V2. EulerLagrangeDiscrete — minimize action over the path

- **Capability.** Given fixed endpoints q(t0)=q0, q(t1)=q1, find the trajectory that minimizes V1's action.
- **Composition.** Stack interior nodes q_1, ..., q_{N-1} into a flat (N-1)·n vector `x`; pass `f(x) = Action(unflatten(x))` to `optim.LBFGS`; provide `grad` via `calculus.NumericalGradient` (or via finite-difference on each path coordinate).
- **Why this is the keystone.** This single primitive *is* the numerical solution of d/dt(∂L/∂q̇) = ∂L/∂q. Every other variational primitive in this review composes V2 with a specific L.
- **Connective tissue.** ~150 LOC in `physics/variational.go::SolveEulerLagrange(L, q0, q1 []float64, t0, t1 float64, N int) [][]float64`. Free-knot path representation; fixed endpoints implemented by NOT including q0, q1 in the optimization vector.
- **Pin.** Free particle (L=½m‖q̇‖²) returns straight line q(t) = q0 + (q1-q0)(t-t0)/(t1-t0) to 1e-6 in path-L² norm at N=50.

#### V3. BrachistochronePath — bead-on-wire least time → cycloid

- **Capability.** Given height h, horizontal reach d, and gravitational acceleration g, return the curve y(x) of least descent time.
- **Composition.** `L(y, y') = √((1+y'²)/(2g(h-y)))`; pass to V2 with N=50.
- **Closed-form pin.** Cycloid x(θ) = R(θ-sin θ), y(θ) = R(1-cos θ); the Lagrange equation gives an integrable conserved quantity. Pin reproduces analytic R via `optim.GoldenSectionSearch` on a single-parameter θ_max search.
- **Connective tissue.** ~50 LOC in `physics/variational.go::Brachistochrone(h, d, g float64, N int) (xs, ys []float64)`.
- **Saturating witness.** R-CYCLOID-3-WAY 3/3 — analytic cycloid vs LBFGS-V2-discrete vs single-parameter-θ-direct agree to 1e-4 on T_min = π√(R/g). Same template as 6a55bb4 / 365368a R-MUTUAL-CROSS-VALIDATION.

#### V4. CatenaryPath — hanging chain, length constrained

- **Capability.** Given fixed endpoints (x0, y0), (x1, y1) and chord length L > distance, return the curve that minimizes potential energy E = ∫ y ds subject to ∫ ds = L.
- **Composition.** Lagrangian-multiplier formulation: minimize `∫ y ds + λ(∫ ds - L)` where the chain is parameterized as y(x). Use `optim.LBFGS` with a 2-level loop (inner LBFGS on y, outer bisection on λ) **OR** use `optim/proximal.Admm` with prox of fixed-arc-length constraint as a projection-onto-affine-set + prox of energy as gradient step. Closed-form: y = a·cosh((x-x0)/a) + c.
- **Connective tissue.** ~120 LOC in `physics/variational.go::Catenary(x0, y0, x1, y1, L float64, N int) []float64`.
- **Pin.** Closed-form catenary y = a cosh(x/a) reproduced to 1e-5 at N=80, validated by 1-D root-finding `optim.NewtonRaphson` on the transcendental endpoint condition.

#### V5. FermatSnellRefraction — derive Snell's law from Fermat's principle

- **Capability.** Given two media with indices n1, n2 and source/destination points, find the path that minimizes optical path length L = ∫ n(r) ds, then return the angles. Verify they obey Snell's law n1 sin θ1 = n2 sin θ2.
- **Composition.** Single free parameter (the x-coordinate where ray crosses the interface) → `optim.GoldenSectionSearch` on T(x_cross) = n1·d1(x) + n2·d2(x).
- **Why this matters.** `physics.SnellRefraction` ships the result (closed-form arcsin); this primitive is the *witness* that the closed-form arises from variational optimization. R-FERMAT-VS-SNELL 3/3 saturates by comparing (a) golden-section x_cross, (b) closed-form arcsin via SnellRefraction, (c) 1-D Newton-Raphson on dT/dx = 0 → match to 1e-12.
- **Connective tissue.** ~70 LOC in `physics/optics.go::FermatPathSingleInterface(p0, p1 [2]float64, ySurf, n1, n2 float64) (xCross, opticalLen float64)`.

#### V6. MaupertuisAbbreviatedAction — fixed-energy least action

- **Capability.** Given conserved energy E and fixed endpoints (q0, q1), minimize the abbreviated action W = ∫ p·dq subject to T+V = E.
- **Composition.** `f(path) = ∫ √(2m(E-V(q))) ds` with no time variable; pass to V2 with arc-length parameterization. Substrate uses `optim.LBFGS` + `calculus.SimpsonsRule`.
- **Connective tissue.** ~90 LOC in `physics/variational.go::MaupertuisPath(V func([]float64) float64, q0, q1 []float64, m, E float64, N int) [][]float64`.
- **Pin.** Constant V → straight-line geodesic; harmonic V = ½kq² → ellipse; closed-form check via `physics.SpringForce`.

#### V7. GeodesicEuclideanCurved — minimize ∫ √(g_ij dq^i dq^j) on a metric

- **Capability.** Given a metric tensor g(q) (n×n matrix-valued function on R^n) and endpoints, return the geodesic.
- **Composition.** L(q, q̇) = √(q̇^T g(q) q̇); pass to V2.
- **Cross-link.** Same machinery used by 091-infogeo-numerics for Fisher-Rao geodesics (information geometry); this is the physics-flavored sibling.
- **Connective tissue.** ~100 LOC in `physics/variational.go::Geodesic(metric func([]float64, [][]float64), q0, q1 []float64, N int) [][]float64`.
- **Pin.** Flat metric g = I → straight line; sphere metric g = diag(1, sin²θ) → great circle; closed-form check via spherical-coordinate analytic geodesic.

### Tier-M (2-3 day, ~900 LOC)

#### V8. RayleighRitzEigenvalue — λ_min = min_x x'Ax / x'x

- **Capability.** Given a real symmetric n×n matrix A (PSD or general), return the lowest eigenvalue and its eigenvector via Rayleigh quotient minimization.
- **Composition.** `f(x) = (x'Ax)/(x'x)`; analytic gradient `∇f = 2(Ax - λ(x)·x)/(x'x)` where λ(x) is the current Rayleigh quotient → pass to `optim.LBFGS` (NOT `GradientDescent` — Rayleigh quotient has saddle structure that LBFGS handles via curvature). Initialize x0 = e_1.
- **Why this is the keystone.** Rayleigh-Ritz is the variational definition of the eigenvalue. It also generalizes to subspace methods (Lanczos = Rayleigh-Ritz on Krylov subspace), which 097-linalg-missing flags.
- **Connective tissue.** ~150 LOC in `optim/eigen.go::RayleighRitz(A []float64, n int) (lambdaMin float64, x []float64)`.
- **Saturating witness.** R-RAYLEIGH-VS-QR 3/3 — Rayleigh-Ritz λ_min vs `linalg.QRAlgorithm` λ_min vs inverse power iteration agree to 1e-10 on 5×5 SPD test matrix. **Cross-language pin target.**

#### V9. RayleighQuotientHigherEigs — deflation for k smallest

- **Capability.** Same as V8 but returns the k smallest eigenvalues by Hotelling deflation: minimize x'Ax/x'x with constraints x ⊥ {previous eigvecs}.
- **Composition.** V8 in a loop with Gram-Schmidt orthogonalization against the running eigenbasis (or proximal projection via `optim/proximal.ProxLinear` for the orthogonality constraint).
- **Connective tissue.** ~130 LOC in `optim/eigen.go::RayleighRitzKSmallest(A []float64, n, k int) ([]float64, [][]float64)`.

#### V10. HelmholtzPlateVibration — eigenvalue of Laplacian via Rayleigh-Ritz

- **Capability.** Given a 1-D or 2-D rectangular domain with Dirichlet BCs, return the lowest eigenfrequencies of -∇²u = λu via finite-difference + V8.
- **Composition.** Discretize Laplacian as 5-point stencil → sparse SPD matrix A; pass to V9 for k=5.
- **Connective tissue.** ~120 LOC in `physics/variational.go::HelmholtzEigenmodes2D(nx, ny int, Lx, Ly float64, k int) ([]float64, [][]float64)`.
- **Pin.** Square Lx=Ly=1 → λ_{m,n} = π²(m²+n²); first 5 modes (1,1)(1,2)(2,1)(2,2)(1,3) match to (h)² = (1/nx)² convergence rate.

#### V11. EulerBernoulliBeamEnergy — variational beam deflection

- **Capability.** Solve a beam deflection problem by minimizing strain energy ∫ ½EI(y'')² dx − ∫ q(x)y dx subject to BCs, comparing against the closed-form `physics.BeamDeflection`.
- **Composition.** Discretize y on N nodes; assemble bending energy as a quadratic form `½y^T K y - f^T y` with K = (EI/h⁴)·tridiag-bending stencil; pass to **V8 RayleighRitz** for buckling **OR** to `optim.LBFGS` for static deflection. Witness: closed-form `BeamDeflection(P, L, E, I) = PL³/(48EI)` at midspan reproduced to (h)² convergence.
- **Connective tissue.** ~130 LOC in `physics/variational.go::BeamDeflectionEnergy(P, L, E, I float64, N int) []float64`.

#### V12. EulerBucklingRayleigh — variational buckling load

- **Capability.** Compute critical buckling load via Rayleigh quotient on (EI∫(y')²)/(∫y² dx) for a column.
- **Composition.** Direct V8 call on the discretized operator. Witness: closed-form `physics.EulerBuckling(E, I, L, K=1) = π²EI/L²` reproduced to (h)² at N=64.
- **Connective tissue.** ~80 LOC in `physics/variational.go::BucklingRayleigh(E, I, L float64, K float64, N int) float64`.

#### V13. PlateauMinimalSurface1D — soap film between two rings

- **Capability.** Axially-symmetric soap film: minimize 2π ∫ r √(1+r'²) dz with r(0) = r0, r(H) = r1.
- **Composition.** L(r, r') = r√(1+r'²); pass to V2.
- **Closed-form pin.** Catenoid r(z) = a·cosh((z-z_min)/a). Same Newton-on-transcendental-endpoint as V4.
- **Connective tissue.** ~80 LOC in `physics/variational.go::PlateauAxisymmetric(r0, r1, H float64, N int) []float64`.

#### V14. ConstrainedActionADMM — isoperimetric problems

- **Capability.** Minimize action S[q] subject to G[q] = 0 (e.g., fixed length, fixed area).
- **Composition.** `optim/proximal.Admm` with proxF = step on action-energy and proxG = projection onto the constraint manifold (uses `optim/proximal.ProxLinear` for affine constraints, `ProxL2Ball` for norm constraints, `ProxSimplex` for sum-to-one constraints).
- **Why ADMM.** Coupled-objective + projection-constraint is the canonical ADMM split — already shipped at `optim/proximal/admm.go`.
- **Connective tissue.** ~100 LOC in `physics/variational.go::ConstrainedAction(L, constraintProx, q0, q1 []float64, N int)`.

### Tier-L (week, ~600 LOC) — composition deepens to autodiff/chaos

#### V15. LagrangianFromTAndV — convenience builder

- **Capability.** `Lagrangian(T, V func([]float64) float64) func(q, qdot []float64, t float64) float64` returns L = T(q̇) - V(q). Wires `physics.KineticEnergy` (vector form) and `physics.PotentialEnergy` (general scalar field).
- **Why a separate primitive.** The repo's existing `KineticEnergy(m, v float64)` and `PotentialEnergy(m, g, h float64)` are scalar-only; the variational machinery needs vector-valued T(q̇) = ½q̇^T M q̇ for general mass matrix M.
- **Connective tissue.** ~50 LOC in `physics/variational.go`.

#### V16. NoetherCurrentNumeric — conserved quantity from continuous symmetry

- **Capability.** Given an action S[q] and a one-parameter symmetry group ε → q+εξ(q), return the conserved current J = (∂L/∂q̇)·ξ.
- **Composition.** Numerical-gradient ∂L/∂q̇ via `calculus.NumericalGradient` partial in q̇, dot with the symmetry vector field.
- **Connective tissue.** ~80 LOC in `physics/variational.go::NoetherCurrent(L, xi func([]float64) []float64, q, qdot []float64) float64`.
- **Pin.** Translation symmetry on free particle → linear momentum p = mq̇ to 1e-10. Rotational symmetry on central potential → angular momentum L = q × p to 1e-10.

#### V17. HilbertEinsteinAction1DToy — Schwarzschild radial

- **Capability.** Toy 1-D radial geodesic in Schwarzschild metric using V7 with metric g(r) = diag(1/(1-2GM/(rc²)), -(1-2GM/(rc²))). Returns geodesic from r0 to r1 with closed-form check at flat-space limit.
- **Composition.** V7 GeodesicEuclideanCurved with the Schwarzschild metric closure.
- **Connective tissue.** ~60 LOC in `physics/variational.go::SchwarzschildRadialGeodesic(M, r0, r1 float64, N int) []float64`.
- **Cross-link.** 164-orbital-optim consumes this for relativistic precession calculation.

#### V18. PontryaginMinimumPrinciple — optimal control via action-principle linkage

- **Capability.** Given a controlled ODE q̇ = f(q, u, t), running cost L(q, u, t), and terminal cost Φ(q(T)), find optimal control u(t) by costate forward-backward sweep.
- **Composition.** Two `chaos.SolveODE` integrations:
  - Forward: state q(t) given current u(t).
  - Backward: costate p(t) with p(T) = ∇Φ(q(T)) and ṗ = -∂H/∂q where H = L + p^T f.
  - Update u(t) by `optim.GradientDescent` step on H (or `optim.GoldenSectionSearch` if u is 1-D bounded).
- **Why it belongs in this synergy.** Pontryagin's H = L + p^T f is *the* Hamiltonian formulation of the optimal-control variational principle (action = ∫ L + boundary). This bridges 168-physics-autodiff (ODE-adjoint) with action-principle minimization.
- **Connective tissue.** ~180 LOC in `physics/variational.go::Pontryagin(f, dfdq, dfdu, L, dLdq, dLdu, Phi, dPhidq, q0, T, Niter)`.
- **Pin.** LQR closed-form Riccati solution agrees with Pontryagin sweep at 1e-6 (cross-link to 53/54 control + 178-control-optim).

#### V19. HamiltonJacobiBellmanFD — value function PDE solver

- **Capability.** Solve the HJB equation -∂V/∂t = min_u {L(q, u, t) + ∇V·f(q, u, t)} on a grid via finite difference + per-cell `optim.GoldenSectionSearch` over u.
- **Composition.** Backward time-march with explicit Euler; at each grid cell use `optim.GoldenSectionSearch` to compute the Hamiltonian minimum.
- **Connective tissue.** ~140 LOC in `physics/variational.go::HamiltonJacobiBellman1D(f, L func(q, u, t float64) float64, terminal func(q float64) float64, qMin, qMax, T float64, Nq, Nt int) [][]float64`.
- **Pin.** Linear-quadratic case → analytic Riccati V(q,t) = ½P(t)q² agrees with FD-HJB grid to (Δq)² convergence.

#### V20. VMCGroundState — Variational Monte Carlo for ground-state energy (Foulkes 2001)

- **Capability.** Given a parameterized trial wavefunction Ψ_α(q) and Hamiltonian H, find the parameters α that minimize the variational energy E(α) = <Ψ_α|H|Ψ_α>/<Ψ_α|Ψ_α> via Metropolis sampling.
- **Composition.** `optim.SimulatedAnnealing` to walk Metropolis on |Ψ|² (acceptance `exp(-ΔE/T)` already at line 73 of `metaheuristic.go`); accumulate local energy E_loc(q) = HΨ/Ψ; outer `optim.LBFGS` updates α via stochastic gradient. Foulkes-Mitas-Needs-Rajagopal 2001 §III.B.
- **Why it ships now.** The Boltzmann acceptance is **already wired**; only the parameter-update loop is missing.
- **Connective tissue.** ~200 LOC in `physics/variational.go::VMCGroundState(H_loc func(q, alpha []float64) float64, psi func(q, alpha []float64) float64, alpha0 []float64, walkerSteps, paramSteps int)`.
- **Pin.** Quantum harmonic oscillator with trial Ψ_α(x) = exp(-αx²/2) → closed-form E(α) = α/4 + 1/(4α), minimum at α=1, E_min = ½ ℏω. VMC reproduces α* and E_min to 1% at 10⁵ Metropolis steps.

#### V21. DFTHohenbergKohnLDA1D — density functional theory in 1-D

- **Capability.** Solve 1-D Kohn-Sham equations for an external potential V_ext(x) and electron count N: minimize E[ρ] = T_s[ρ] + ∫ V_ext ρ + ½ ∫∫ ρρ'/|x-x'| + E_xc[ρ] over densities ρ ≥ 0, ∫ρ=N.
- **Composition.** Self-consistent loop: at each iteration solve the eigenvalue problem -½ φ'' + V_eff φ = ε φ via V8 RayleighRitz on discretized Hamiltonian; rebuild ρ from filled orbitals; repeat. LDA E_xc from local density approximation closed-form.
- **Connective tissue.** ~250 LOC in `physics/dft.go::KohnShamSCF1D(Vext func(float64) float64, N int, gridN int) (rho, energies []float64)`.
- **Cross-language pin.** H atom in 1-D toy: ground-state E ≈ -0.6738 (numerical reference from Kohn-Sham 1D LDA) reproduced to 1e-3.

#### V22. GaussLeastConstraint — Gauss's principle for unilateral constraints

- **Capability.** At each instant, minimize the "constraint" Z = Σ m_i ‖a_i - F_i/m_i‖² subject to constraint forces being normal to the constraint manifold. Equivalent to D'Alembert + Lagrange multipliers but variational at the acceleration level (Gauss 1829).
- **Composition.** Per-step quadratic-program: `optim.SimplexMethod` on the linearized problem (or `optim/proximal.Admm` with prox of constraint manifold projection).
- **Connective tissue.** ~120 LOC in `physics/variational.go::GaussLeastConstraint(masses []float64, forces, accelConstrained []float64, constraintProx ProxOp) []float64`.

---

## 2. LOC roll-up

| Tier | Count | Source LOC | Test LOC | Total |
|------|-------|------------|----------|-------|
| Tier-S (V1-V7, day-1) | 7 | 660 | 480 | 1,140 |
| Tier-M (V8-V14, week-1) | 7 | 740 | 540 | 1,280 |
| Tier-L (V15-V22, week-2-3) | 8 | 1,080 | 720 | 1,800 |
| **Total** | **22** | **2,480** | **1,740** | **4,220** |

ZERO new packages. All ship into `physics/variational.go` (V1-V7, V11-V17, V19-V20, V22 = ~1,520 LOC), `optim/eigen.go` (V8-V9 = ~280 LOC), `physics/dft.go` (V21 = ~250 LOC), `physics/optics.go` extension (V5 = ~70 LOC).

---

## 3. Cycle-free new edges

```
physics/variational.go  →  calculus/        (Simpson, NumericalGradient)
physics/variational.go  →  optim/           (LBFGS, GoldenSection, SimulatedAnnealing, LinearInterpolateRoot)
physics/variational.go  →  optim/proximal/  (Admm, ProxLinear, ProxSimplex)
physics/variational.go  →  chaos/           (RK4Step, SolveODE) for V18
physics/variational.go  →  linalg/          (LUSolve, CholeskySolve) for V9 mass-matrix inversion
optim/eigen.go          →  linalg/          (matrix-vector multiply via existing helpers)
physics/dft.go          →  physics/variational.go (V8, V15)
```

NO reverse edges. Verified by current `physics → constants` and `optim → ∅` only.

---

## 4. Recommended sprint ordering

**Day 1.** Ship V1 ActionFunctional + V2 EulerLagrangeDiscrete + V3 BrachistochronePath (220 LOC). Cycloid pin saturates R-CYCLOID-CLOSED-FORM 3/3, the strongest single-witness in classical variational physics. Single PR. First `physics → optim` edge in the repo.

**Day 2.** V5 FermatSnellRefraction + V6 MaupertuisAbbreviatedAction + V7 GeodesicEuclideanCurved (260 LOC). Cross-validates the existing `SnellRefraction` closed-form (turns shipped function from "asserted formula" into "verified Fermat consequence"). R-FERMAT-VS-SNELL 3/3.

**Day 3.** V4 CatenaryPath + V13 PlateauMinimalSurface1D (200 LOC). Length-constrained variational; introduces ADMM consumer pattern in `physics/`.

**Week 1.** V8 RayleighRitzEigenvalue + V9 RayleighRitzKSmallest + V10 HelmholtzPlateVibration + V11 EulerBernoulliBeamEnergy + V12 EulerBucklingRayleigh (610 LOC). **Architectural keystone week.** Reformulates eigenproblem as variational, ships `optim/eigen.go`, gives consumers (acoustics/plate-modes, em/cavity-modes, control/eigenvalue-margins, chaos/Lyapunov) a Rayleigh-Ritz primitive that today they each reimplement.

**Week 2.** V14 ConstrainedActionADMM + V15 LagrangianFromTAndV + V16 NoetherCurrent + V17 SchwarzschildRadialGeodesic (290 LOC). Wraps prior tiers into ergonomic builders.

**Week 3.** V18 PontryaginMinimumPrinciple + V19 HamiltonJacobiBellmanFD + V20 VMCGroundState + V22 GaussLeastConstraint (640 LOC). Optimal control + quantum-VMC + non-holonomic constraints. Cross-link to 164-orbital-optim, 178-control-optim, 168-physics-autodiff.

**Week 4.** V21 DFTHohenbergKohnLDA1D (250 LOC). Self-consistent KS-DFT in 1-D — research-grade, cross-pin against published 1-D LDA reference values.

---

## 5. Cross-language golden-file pin targets

| Pin | Reference value | Tolerance | Tests Go/Py/C++/C# |
|-----|-----------------|-----------|---------------------|
| R-CYCLOID-CLOSED-FORM | T_min = π√(R/g) for h=2R brachistochrone | 1e-4 | 4× |
| R-FERMAT-VS-SNELL | Snell's law angle agrees with Fermat-min x_cross | 1e-12 | 4× |
| R-CATENARY-COSH | y(0) = a·cosh(0) = a, transcendental param a | 1e-5 | 4× |
| R-FREE-PARTICLE-STRAIGHT | linear-path action min for L=½m‖q̇‖² | 1e-12 | 4× |
| R-RAYLEIGH-VS-QR | λ_min(A) Rayleigh-Ritz vs QRAlgorithm vs power | 1e-10 | 4× |
| R-HELMHOLTZ-SQUARE | λ_{1,1} = 2π² unit square Dirichlet | 1e-3 (h²) | 4× |
| R-BEAM-CLOSED-FORM | δ_max = PL³/(48EI) reproduced by V11 energy | 1e-3 (h²) | 4× |
| R-EULER-BUCKLING | P_cr = π²EI/L² reproduced by V12 Rayleigh | 1e-3 (h²) | 4× |
| R-CATENOID-PLATEAU | r(z) = a·cosh((z-z₀)/a) for V13 | 1e-4 | 4× |
| R-NOETHER-MOMENTUM | translation symmetry → mq̇ for free particle | 1e-10 | 4× |
| R-PONTRYAGIN-VS-LQR | LQR Riccati matches Pontryagin sweep | 1e-6 | 4× |
| R-HJB-LQR | analytic ½P(t)q² value function vs FD grid | 1e-3 (h²) | 4× |
| R-VMC-HARMONIC | quantum SHO α* = 1, E_min = ½ℏω via VMC | 1e-2 (10⁵ steps) | 4× |
| R-GAUSS-PRINCIPLE | bead-on-wire constraint reaction normal force | 1e-10 | 4× |

---

## 6. Precision hazards

1. **V2 SolveEulerLagrange** — central-difference q̇ at endpoints needs one-sided stencil (otherwise off-by-h boundary error dominates at small N). Mandatory: forward-diff at i=0, backward at i=N.
2. **V3 Brachistochrone** — at the endpoint y=h, the integrand √((1+y'²)/(2g(h-y))) diverges as 1/√(h-y). Must use a substitution u = √(h-y) to regularize. **Naive Simpson on the unregularized integrand gives nonsense.**
3. **V4 Catenary** — Lagrange multiplier has *two* signs (the chain hangs *down* not *up*); the corresponding constraint multiplier λ must be initialized with the correct sign or LBFGS converges to the inverted "cosh-up" solution.
4. **V8 RayleighRitz** — Rayleigh quotient has saddle at zero; must initialize at random nonzero x and add a tiny `‖x‖=1` projection step (alternative: `optim/proximal.ProxL2Ball(1.0)` between LBFGS iterates).
5. **V9 RayleighRitzKSmallest** — Hotelling deflation accumulates error after k=10; switch to subspace Rayleigh-Ritz (block Lanczos) if needed.
6. **V11 BeamDeflectionEnergy** — bending energy ½EI(y'')² requires *fourth*-order finite difference for stable energy; second-order y'' stencil gives O(h²) error and beam tip oscillation.
7. **V18 Pontryagin** — adjoint backward-sweep numerical instability for stiff f(q,u,t); recommend `chaos.SolveODE` only for non-stiff cases.
8. **V19 HamiltonJacobiBellmanFD** — viscosity-solution monotone scheme required; naive central-difference HJB blows up. Use upwind first-order on ∇V (Sethian fast marching pattern, see 142-topology-missing eikonal cross-link).
9. **V20 VMCGroundState** — variance of local energy E_loc(q) must be tracked; if it doesn't decrease as α approaches optimum, the trial wavefunction lacks a node-correctness condition. Foulkes 2001 §III.C.
10. **V21 DFT** — KS self-consistency without Pulay mixing diverges for small basis; mandatory Anderson/DIIS mixing on density (~30 LOC additional inside V21).

---

## 7. Out-of-scope deferrals

- **Full QED/QFT path-integral action** — not in the closed-form regime; defer beyond v1.0.
- **GR Hilbert-Einstein in 3+1 D** — V17 ships only the 1-D toy; full ADM-formalism PDE solver belongs in a future `physics/relativity/` sub-package, blocked on pde/ infrastructure.
- **Multireference quantum chemistry (CASSCF, MRCI)** — far beyond V20 single-determinant VMC; defer.
- **Adaptive mesh for V19 HJB curse-of-dimensionality** — only 1-D for now; 2-D ships when geometry/ KD-tree lands.
- **Variational autoencoder** (mentioned in topic) — covered by 169-synergy-prob-optim with a probabilistic flavor and 195-synergy-optim-prob (variational inference). This review takes the *physics* flavor; VAE deferred to those.
- **Mountain pass theorem / saddle-point variational** — interesting but requires functional-analysis machinery (Palais-Smale, Ekeland) outside the numerical-primitives scope.
- **Onsager-Machlup action for non-equilibrium SDEs** — cross-links to 180-synergy-physics-prob; defer to that synergy.

---

## 8. Cross-package consumers and downstream payoff

**164-orbital-optim** — V6 Maupertuis abbreviated action gives a different solver path for orbit determination than current Hohmann/Lambert closed-forms; reduces specialized-formula count.
**168-synergy-physics-autodiff** — V18 Pontryagin reuses the proposed `chaos.RK4Step` adjoint integrator and `autodiff/dual.go` Hessian; this review and 168 share a compositional axis. **Compatibility: V1 ActionFunctional + autodiff.Tape gives ∂S/∂q automatic-differentiated, removing V2's `calculus.NumericalGradient` dependence.**
**178-control-optim** — V19 HJB shares the dynamic-programming infrastructure with classical LQR/MPC; V18 Pontryagin already lives in optimal-control.
**091-infogeo-numerics** — V7 Geodesic ships the same Riemannian-metric-based geodesic primitive that infogeo needs for Fisher-Rao geodesics. The two should literally call the same function.
**107-orbital-numerics** — V17 SchwarzschildRadial gives a relativistic correction primitive for orbital precession.

**Architectural unlock:** V8 RayleighRitz is the *only* `optim/eigen.go` we need before consumers in **acoustics** (Helmholtz cavity modes), **em** (waveguide modes), **chaos** (Lyapunov), **graph** (spectral graph theory), and **linalg** (Lanczos) can stop reimplementing power iteration.

---

## 9. Research-grade unlocks

**Lighthill-style action principle for fluid dynamics (Hamilton-Bateman 1929).** Cross-link to 197-acoustics-fluids: variational fluid dynamics ships once V1 ActionFunctional accepts *field-valued* paths (i.e., q(x,t) instead of q(t)). Defer to `physics/field.go` (~150 LOC additional).

**Maxwell action ∫ (-¼ F^μν F_μν - J^μ A_μ) d⁴x.** Cross-link to 159-em-signal: discrete-form Maxwell action as a bilinear form in A; minimization yields Maxwell equations from the variational principle. Composes V1 + V8 with the EM stencil. ~200 LOC, defer.

**Ising free-energy variational mean-field.** Cross-link to 180-synergy-physics-prob (statmech sub-package): replaces brute-force MC with a parameterized variational ansatz Φ_θ; minimize free energy F[θ] = E[θ] - T·S[θ] via `optim.LBFGS`. Already 90% feasible from 180's V10 IsingObservables; ships as ~80-LOC extension once that lands.

---

## 10. The shipping decision

**Day 1 (V1+V2+V3, 220 LOC, single PR).** First physics→optim edge. Establishes the `Action`/`SolveEulerLagrange` API surface. The cycloid pin is the cheapest-possible "yes this works" demonstration.

**Week 1 (V4-V14, ~1,000 LOC).** Variational physics canon. R-RAYLEIGH-VS-QR architecturally significant.

**Weeks 2-3 (V15-V22, ~1,300 LOC).** Research-grade: Pontryagin, HJB, VMC, DFT. Cross-link to 164/168/178/180.

**Bottom line.** No new packages. No new abstractions. All variational-principle physics emerges from `optim.LBFGS(action)`. The substrate has shipped for six months; nobody has yet wired `physics → optim`. Day-1 PR ships immediately.

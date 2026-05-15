# 243 | new-fpe — Fokker-Planck / Kolmogorov forward-backward / finite-volume schemes

**Summary line 1.** reality v0.10.0 ships **zero** Fokker-Planck / forward-Kolmogorov / backward-Kolmogorov / Klein-Kramers / Chang-Cooper / mass-conservative-finite-volume / first-passage-time / Kramers-escape / stationary-FP / Onsager-Machlup / Wasserstein-gradient-flow surface — repo-wide grep on `Fokker|fokker|Kolmogorov.forward|Kolmogorov.backward|FPE|Chang.?Cooper|Patankar|first.passage|mean.exit|Kramers|Klein.?Kramers|Onsager.?Machlup|stationary.distribution|detailed.balance` against `*.go` files returns **only one tangential hit** (`info/lz/doc.go` mentioning *Kolmogorov complexity* — entirely unrelated; this is algorithmic information theory, not the diffusion-equation Kolmogorov). The closest substrate is the deterministic-ODE engine `chaos/ode.go` (RK4Step + EulerStep + SolveODE on `dy/dt = f(t, y)` — no PDE spatial discretisation, no Laplacian stencil, no upwind operator), `linalg/decompose.go` (LU + Cholesky banded solvers usable for tridiagonal implicit-Euler / Crank-Nicolson timesteps but no Thomas-tridiag specialisation), `linalg/eigen.go` (Householder→TQL symmetric-tridiag eigensolver — gates spectral-FPE eigenfunction expansion), `signal/fft.go` (Cooley-Tukey radix-2 in-place — gates pseudo-spectral / Fourier-Galerkin FPE on torus), `prob/markov.go` (`MarkovSteadyState` power-iteration on stochastic transition matrix — the *closest cousin* of FPE-stationary-density: discrete-time-discrete-state version of the same `pi = pi · P` fixed-point but the continuous-state PDE-spatial-flux machinery is entirely net-new). **Cross-link landscape**: slot 202-new-sde establishes the Itô-SDE primitives (Brownian path, Euler-Maruyama, Milstein) and the FPE is the **Kolmogorov-forward-equation dual** of every SDE in 202 — the two reviews are siblings on the SDE↔PDE Itô-Stratonovich-equivalence axis (Risken-1989 §3 *The Fokker-Planck Equation*); slot 219-new-mean-field-games already names a `chaos/pde/upwind.go` ~180 LOC substrate (M1 HJB-FD upwind-Hamiltonian) and a positivity-preserving FP-scheme (M2 ~280 LOC) as part of the MFG corpus — **direct overlap with this slot's F-primitives**, recommend single-source ownership in `pde/fpe/` consumed by `gametheory/mfg/` rather than parallel implementations; slot 244-new-pde-solvers is the **strict upstream** for 1D/2D/3D Laplacian + boundary-condition substrate (Dirichlet / Neumann / periodic / reflecting / absorbing — all four FPE-relevant), slot 245-new-spectral-methods is **strict upstream** for Hermite / Chebyshev / Fourier-Galerkin spectral-FPE solvers; slot 242-new-spde is **dual** in the orthogonal direction (FPE evolves the *density* p(x, t); SPDE evolves a *random field* u(x, t) — same x-discretisation substrate, different stochastic semantics). **MASTER_PLAN slot 243 names `Fokker-Planck: forward/backward Kolmogorov, finite-volume schemes`**; this report enumerates F1-F22 totalling ~3,400 LOC organized as Tier-0 substrate / Tier-1 keystone / Tier-2 high-demand / Tier-3 frontier.

**Summary line 2.** Twenty-two FPE primitives **F1-F22** totalling ~3,400 LOC split across **(a) Tier-0 substrate ~440 LOC** (F0a `pde/fpe/grid.go` UniformGrid1D + Dirichlet/Neumann/periodic/reflecting/absorbing boundary helpers ~120 LOC, F0b `pde/fpe/upwind.go` first-order upwind drift discretisation ~80 LOC shared with 219-MFG-M1, F0c `pde/fpe/laplacian.go` 1D/2D 5-point/9-point Laplacian ~120 LOC shared with 244, F0d `linalg/tridiag.go` Thomas-algorithm O(n) tridiag solver — currently `linalg.LUSolve` is O(n²) general LU which is wasteful for FPE which is intrinsically tridiagonal in 1D ~80 LOC), **(b) Tier-1 keystone ~1,180 LOC** (F1 ForwardKolmogorov1D explicit-Euler + finite-volume + Chang-Cooper-1970 ~280 LOC, F2 BackwardKolmogorov1D dual operator for expectation propagation / option pricing / mean-exit-time ~240 LOC, F3 StationaryFP1D solve `L*p = 0` for invariant density ~200 LOC, F4 CrankNicolsonFP1D second-order-in-time implicit ~220 LOC, F5 ImplicitEulerFP1D unconditional-stability tridiag-implicit ~120 LOC, F6 IMEX-FP1D explicit-drift+implicit-diffusion ~120 LOC), **(c) Tier-2 high-demand ~1,030 LOC** (F7 ForwardKolmogorov2D ADI-Peaceman-Rachford ~260 LOC, F8 ChangCooper2D positivity+equilibrium-preserving ~180 LOC, F9 KleinKramersFPE phase-space (x, v) — stochastic-Hamiltonian / underdamped Langevin ~280 LOC, F10 MeanExitTime backward-Kolmogorov-Dirichlet `L*u = -1` ~160 LOC, F11 FirstPassageTimeDistribution ~150 LOC), **(d) Tier-3 frontier ~750 LOC** (F12 BlackScholesFromFPE backward-Kolmogorov on lognormal-GBM ~120 LOC, F13 KramersEscapeRate Arrhenius-like asymptotic for double-well metastability ~100 LOC, F14 OnsagerMachlupAction path-integral Lagrangian for FPE ~140 LOC, F15 SpectralFP-Hermite-Galerkin ~180 LOC blocked-on-245, F16 ParticleFP-method-of-characteristics N-particle empirical-density approximation ~140 LOC, F17 EnsembleKalmanFP localized-particle FPE-approximation ~70 LOC), **(e) Tier-3.5 connective ~420 LOC** (F18 WassersteinGradientFlow JKO-step on KL-divergence cross-link 201-OT ~180 LOC, F19 DetailedBalance-check property-test for reversibility ~60 LOC, F20 BoltzmannEquilibrium `p_∞(x) ∝ exp(-V(x)/(k_B T))` for-gradient-systems ~60 LOC, F21 ErgodicityTest mixing-time / spectral-gap diagnosis ~60 LOC, F22 TensorTrainFP high-dim FPE blocked-on-203-tensor-train ~60 LOC stub). **SINGULAR-MOAT F1+F8 Chang-Cooper-1970 finite-volume scheme ~460 LOC** because Chang-Cooper is **THE** canonical positivity-preserving + equilibrium-preserving discretisation for FPE (Chang-Cooper-1970-J.Comput.Phys-6 — every plasma-physics / astrophysics-radiative-transfer / financial-volatility code uses it; Larsen-Levermore-Pomraning-Sanderson-1985 generalisation for high dimensions, Buet-Le-Thanh-2008 for non-uniform grids), and no zero-dependency Go library ships it: scipy has it as `scipy.special` only via numerical-recipe-port, MATLAB has the Chebfun `pdesolve` wrapper, but production-Go has nothing. **SINGULAR-PEDAGOGICAL-MOAT F12 Black-Scholes-from-FPE ~120 LOC** because Black-Scholes-1973 is the **canonical reverse-derivation** illustrating that the celebrated PDE `∂V/∂t + (1/2)σ²S²·∂²V/∂S² + rS·∂V/∂S - rV = 0` is **literally the backward-Kolmogorov equation** for geometric-Brownian-motion under risk-neutral measure — pinning F2 BackwardKolmogorov1D against the closed-form Black-Scholes call/put price (Hull-2017 §17.4) is the ideal cross-language golden-file fixture (analytical N(d1)/N(d2) reference vs FD numerical solve, `‖Δ‖_∞ < 1e-6` for ATM-1y-call). **SINGULAR-2024-FRONTIER F18 Wasserstein-gradient-flow JKO-step ~180 LOC** because Jordan-Kinderlehrer-Otto-1998-SIAM.J.Math.Anal-29 establishes that **FPE = gradient flow of relative entropy on Wasserstein-2 space** (the JKO theorem), making FPE the **dual viewpoint** of slot 201-OT's W_2-substrate; this connects to 201-Sinkhorn (entropic-regularization JKO ↔ Schrödinger-bridge limit Léonard-2014) and to 219-MFG-M17 transport-based-MFG (FPE = forward-equation in MFG-system). **SINGULAR-CROSS-LINK F9 Klein-Kramers ~280 LOC** because Klein-Kramers FPE in phase space (x, v) is the **only** FPE that's *not* a heat-equation-with-drift — it's a **degenerate-parabolic-hypoelliptic** PDE (no v-Laplacian — diffusion only in v, advection in x via v) requiring distinct numerics (Bouchut-1998, splitting schemes). Recommended placement **NEW package `pde/fpe/`** ~3,000 LOC = (`grid.go` ~120 + `upwind.go` ~80 + `laplacian.go` ~120 + `forward_1d.go` ~280 + `backward_1d.go` ~240 + `stationary.go` ~200 + `crank_nicolson.go` ~220 + `implicit_euler.go` ~120 + `imex.go` ~120 + `forward_2d.go` ~260 + `chang_cooper_2d.go` ~180 + `klein_kramers.go` ~280 + `mean_exit_time.go` ~160 + `first_passage.go` ~150 + `black_scholes.go` ~120 + `kramers_escape.go` ~100 + `onsager_machlup.go` ~140 + `spectral_hermite.go` ~180 + `particle_fp.go` ~140 + `ensemble_kalman.go` ~70 + `wasserstein_flow.go` ~180 + `properties.go` ~180 detailed-balance / Boltzmann / ergodicity), plus **shared substrate `linalg/tridiag.go`** ~80 LOC Thomas-algorithm (O(n) tridiag-solver — currently absent, every implicit FPE timestep degrades to O(n²) general LU); rationale for `pde/fpe/` package: FPE is sufficiently large and self-contained (forward + backward + stationary + Klein-Kramers + first-passage + financial + spectral + particle = 22 primitives ~3,000 LOC) to deserve its own subpackage parallel to `prob/copula/` / `prob/conformal/`, and the `pde/` parent will be co-created by slot 244 to host the 1D/2D/3D Laplacian + boundary-handling substrate; placing FPE *under* `pde/` (rather than at top-level or under `prob/`) signals the canonical viewpoint that FPE is a parabolic PDE first and a stochastic-process second, with the stochastic-process semantics (sample paths, Itô integral, Stratonovich correction) living in `prob/sde/` (slot 202). **CANDOR**: FPE-numerics has *substantial* overlap with three other Block-C reviews — slot 202-SDE (FPE is dual to SDE; every Kolmogorov-forward solution is also computable via Monte-Carlo on the dual SDE — F1 vs 202-S1 parity-check), slot 219-MFG (M1 HJB upwind-Hamiltonian + M2 FP positivity-preserving = F2 + F1 in MFG context), slot 244-PDE-solvers (FPE is a special parabolic PDE — Tier-0 substrate fully overlaps). Single-source ownership recommendation: F0a-F0c live in `pde/` (slot 244 owns), F1-F22 live in `pde/fpe/` (this slot owns), 219-MFG consumes both. Cheapest-1-day shippable: **F0a + F0d + F1 + F3 + F19 ~640 LOC** Tier-0 substrate + ForwardKolmogorov1D + StationaryFP1D + DetailedBalance-check on top of existing `linalg.LUSolve` and `chaos.EulerStep` patterns, saturating R-FORWARD-EVOLUTION 3/3 against closed-form OU-process Gaussian-stationary `p_∞(x) = N(0, σ²/(2γ))` for `dX = -γX dt + σ dW`. **18 of 22 primitives unique to this slot** (F0a-F0c shared with 244, F18 cross-links 201-OT, F2 cross-links 219-MFG-M1).

---

## 0. State at HEAD (2026-05-08, v0.10.0)

Repo-wide audit for FPE / forward-Kolmogorov / backward-Kolmogorov / Chang-Cooper / first-passage / Kramers / Klein-Kramers / Onsager-Machlup / detailed-balance / stationary-density surface — verified via Grep on full primitive lexicon:

| Surface | Path | FPE relevance |
|---|---|---|
| `chaos.RK4Step / EulerStep / SolveODE` | `chaos/ode.go` | Time-stepping substrate for explicit-FPE integration via method-of-lines; PRESENT (132 LOC). **No PDE spatial discretisation, no upwind, no Laplacian stencil, no boundary handling.** |
| `chaos.systems` Lorenz / Van-der-Pol / etc | `chaos/systems.go` | Deterministic-ODE chaos test cases; orthogonal to FPE |
| `linalg.LUDecompose / LUSolve` | `linalg/decompose.go` | General LU — usable for tridiag implicit-FPE but O(n²) factor + O(n²) solve where a **Thomas tridiag-solver is O(n)**. PRESENT (general) but tridiag-specialisation is ABSENT. F0d ~80 LOC. |
| `linalg.CholeskyDecompose / CholeskySolve` | `linalg/decompose.go` | Cholesky for SPD systems; usable for symmetric-FPE-discretisation (e.g. self-adjoint FP operator under detailed-balance) |
| `linalg.tridiagonalize / tqli` (private) | `linalg/eigen.go` | Symmetric-tridiag QL eigensolver (private); gates F15 spectral-FPE-Hermite eigendecomposition. **Currently unexported** — would need `EigenSymmetricTridiag(d, e []float64) (eigvals []float64, eigvecs [][]float64)` public surface |
| `signal.FFT / IFFT` | `signal/fft.go` | Cooley-Tukey radix-2; gates F15 pseudo-spectral-FP-Fourier on torus, F18 Fourier-mode initialization |
| `prob.MarkovSteadyState` | `prob/markov.go` | **Discrete-time-discrete-state analog** of F3 StationaryFP — power-iteration on `pi · P = pi`. The continuous-state PDE-flux generalisation (F3) is entirely net-new. |
| `prob/distributions.go` Normal / Beta / Gamma / Exp PDF+CDF | `prob/distributions.go` | Used as *initial conditions* for FPE solves (Gaussian initial density, etc) and *reference equilibria* (Boltzmann = Gaussian for harmonic potential); PRESENT |
| `prob.SampleNormal` Box-Muller | -- | **ABSENT** — keystone-blocker shared with 202/216/217/218/219/220/239/240/241/242 (now ELEVEN+ Block-C reviews); needed for F16 ParticleFP and F17 EnsembleKalman initialisation |
| `optim/transport/sinkhorn.go` Sinkhorn | `optim/transport/sinkhorn.go` | W_2-distance via entropic-regularisation; gates F18 Wasserstein-gradient-flow JKO-step (cross-link 201-OT) |
| `calculus.NumericalDerivative / NumericalGradient` | `calculus/calculus.go` | Differentiation substrate; useful for F2 BackwardKolmogorov drift+diffusion sampling at arbitrary points |
| `physics.thermo.go` | `physics/thermo.go` | Thermodynamics; **does not import** Boltzmann distribution `p ∝ exp(-βH)` (F20 is the FPE-equilibrium statement of this) |
| F1-F22 FPE primitives | -- | **ALL ABSENT** (22 distinct primitives) |

**Cross-import edges.** `pde/fpe/` would consume `linalg.LUSolve` (PRESENT — would prefer Thomas-tridiag F0d), `linalg.CholeskySolve` (PRESENT, F4 Crank-Nicolson SPD case), `linalg.EigenSymmetric` (PRESENT but private tridiag — F15 needs public surface), `signal.FFT` (PRESENT, F15 spectral / F18 Fourier-init), `optim/transport.Sinkhorn` (PRESENT, F18 W_2-JKO), `chaos.EulerStep` (PRESENT, F1 explicit-Euler-time), `prob.MarkovSteadyState` as test-fixture-only (discrete-vs-continuous parity), `prob.SampleNormal` (Box-Muller — ABSENT, F16/F17 only), `prob.NormalPDF/CDF` (PRESENT, F12 Black-Scholes reference). New edges: `pde/fpe → linalg`, `pde/fpe → signal`, `pde/fpe → chaos`, `pde/fpe → optim/transport`, `pde/fpe → prob`. No cycles.

**Cross-package blockers.**

| Blocker | Owner | Blocks F-primitives | LOC est |
|---|---|---|---:|
| `linalg.SolveTridiag` Thomas O(n) | F0d THIS-slot | F4, F5, F6, F7, F8 (every implicit-FPE timestep) | 80 |
| `pde.Laplacian1D` 5-point stencil | 244-new-pde-solvers OR ship-here-as-F0c | F1, F4, F5, F6 | 120 (in slot 244) |
| `pde.Laplacian2D` 9-point stencil | 244-new-pde-solvers OR ship-here-as-F0c | F7, F8, F9 | 200 (in slot 244) |
| `pde.UpwindDrift1D` first-order upwind | F0b THIS-slot SHARED-with-219-MFG-M1 | F1, F2, F4, F5, F6 | 80 (single-source) |
| `pde.BoundaryConditions` Dirichlet/Neumann/periodic/reflecting/absorbing | F0a THIS-slot | All F1-F11 | 120 |
| `prob.SampleNormal` Box-Muller | 117/202 cross-blocker (12+ slots) | F16, F17 only | 50 |
| `linalg.EigenSymmetricTridiag` public | THIS-slot ~10 LOC wrapper around private `tridiagonalize/tqli` | F15 only | 10 |
| `signal.FFT2D` 2D-FFT | 245-new-spectral-methods | F15 (2D spectral) only | 250 (slot 245) |
| `optim/transport.SinkhornDual` for JKO | 201-OT (PRESENT for primal) — JKO-step new ~120 LOC | F18 | 120 (could ship in F18 as helper) |

Total upstream-blocker LOC ~1,030 of which ~50 (Box-Muller) is twelve-times-redundant in the Block-C roster (slot 117 ships it as 1-day delivery). Thomas-tridiag F0d (~80 LOC) is the **single most-leveraged net-new commit** for FPE alone — every implicit-time FPE primitive (F4/F5/F6/F7/F8) drops from O(n²) to O(n) per timestep.

**Cross-link audit.**
- 026-chaos-numerics.md / 027-chaos-missing.md: surveyed Lyapunov-exponent / Lorenz / Mackey-Glass / RK45-adaptive — **FPE not flagged** (chaos package is pointwise-trajectory not density-evolution).
- 117-prob-missing.md: surfaced Sample-API as Box-Muller-keystone — F16/F17 here consume it.
- 161-synergy-control-prob.md: mentions Fokker-Planck for stochastic-control "in passing" — **does not enumerate F-primitives**.
- 198-synergy-physics-optim.md: mentions HJB-Bellman (the dual of FPE under min-max swap) — does not enumerate FPE.
- 201-new-optimal-transport.md: shipped Sinkhorn / W_2 / Wasserstein-barycenter — **F18 Wasserstein-gradient-flow JKO-step is the natural-FPE-consumer of 201's W_2-substrate**; slot 201 mentions JKO once but defers the FPE-side to slot 243 (this report).
- 202-new-sde.md: shipped Brownian-path / Euler-Maruyama / Milstein / MLMC — **direct dual** to this report; FPE = forward-Kolmogorov of every SDE in 202; cross-language golden-file pin: F1-vs-S1 parity (1e6 SDE Monte-Carlo paths → empirical density vs F1 grid-FPE) within 4σ for `dX = -X dt + dW` OU-process at `t=1`.
- 218-new-rough-paths.md: rough-path-lift of Brownian motion is upstream of *some* FPE-with-rough-noise variants but reality won't ship.
- 219-new-mean-field-games.md: **STRICT-OVERLAP** — M1 HJB-FD upwind-Hamiltonian + M2 FP-positivity-preserving = F2 + F1 in MFG context; recommend single-source-ownership (F1+F2 in `pde/fpe/`, M1+M2 in `gametheory/mfg/` consumes them as `import "github.com/davly/reality/pde/fpe"`).
- 242-new-spde.md: dual axis (FPE = density-evolution, SPDE = field-evolution); shared spatial-discretisation substrate `pde.Laplacian1D/2D` (slot 244).
- 244-new-pde-solvers.md: **STRICT-UPSTREAM** for `pde/` package + Laplacian1D/2D/3D + boundary conditions; F0a-F0c live in slot 244, FPE consumes.
- 245-new-spectral-methods.md: STRICT-UPSTREAM for F15 SpectralFP-Hermite-Galerkin and Fourier-Galerkin-on-torus.
- 203-tensor-train (presumed slot, name TBD): F22 high-dim FPE blocked.

---

## 1. The twenty-two FPE primitives (F1–F22)

### Tier-0 substrate (~440 LOC, blocks Tier-1+)

**F0a `pde/fpe/grid.go` — UniformGrid1D + boundary helpers ~120 LOC.** Defines `Grid1D{XMin, XMax, N float64}`, dx := (XMax-XMin)/(N-1), and five boundary-condition encoders: Dirichlet (`p[0]=p[N-1]=0` — natural for absorbing-boundary first-passage problems), Neumann (`dp/dx[0]=dp/dx[N-1]=0` — natural for reflecting-boundary mass-conserving FPE), periodic (`p[N]=p[0]` — natural for FPE-on-torus), reflecting (probability-current `J=0` — strict mass conservation), absorbing (probability vanishes at boundary — first-exit problems). Reference: LeVeque-2007 *Finite Difference Methods for ODEs and PDEs* §2-3, Risken-1989 §6.

**F0b `pde/fpe/upwind.go` — first-order upwind drift discretisation ~80 LOC.** SHARED with slot 219-MFG-M1. Discretises `μ(x)·∂p/∂x` with upwind based on sign of `μ`: `μ⁺·(p_i - p_{i-1})/dx + μ⁻·(p_{i+1} - p_i)/dx`. Mass-conserving, monotone, but only first-order accurate. Reference: LeVeque-2007 §4.

**F0c `pde/fpe/laplacian.go` — 1D 3-point + 2D 5-point Laplacian ~120 LOC.** SHARED with slot 244. Discretises `∂²/∂x²` by central-difference `(p_{i+1} - 2p_i + p_{i-1})/dx²`; 2D 5-point cross-stencil for `Δ = ∂²/∂x² + ∂²/∂y²`. Optional 9-point stencil for higher-order accuracy.

**F0d `linalg/tridiag.go` — Thomas algorithm tridiag-solver O(n) ~80 LOC.** **Currently absent; reality's only banded-solver is general `linalg.LUSolve` which is O(n²) for tridiag.** Implements forward-elimination + back-substitution on `(a_i p_{i-1} + b_i p_i + c_i p_{i+1} = d_i)` system. Cross-package leverage: F4/F5/F6/F7/F8 implicit-FPE all use it; also useful for Crank-Nicolson heat-equation, B-spline interpolation, cubic-spline solvers.

### Tier-1 keystone (~1,180 LOC, the core FPE corpus)

**F1 `pde/fpe/forward_1d.go` ForwardKolmogorov1D ~280 LOC.** Solves `∂p/∂t = -∂(μ·p)/∂x + (1/2)·∂²(σ²·p)/∂x²` (forward-Kolmogorov / Fokker-Planck — note `σ²` *inside* the second derivative, not outside, this is the **Itô-form** vs the Stratonovich `(σ·p)·∂σ/∂x` correction). Three sub-schemes: (a) explicit-Euler-time + central-diff-space + upwind-drift (CFL `dt < dx²/(σ²)` and `dt < dx/|μ|`), (b) finite-volume-conservative form `∂p/∂t = -∂J/∂x` with flux `J = μ·p - (1/2)·∂(σ²·p)/∂x` discretised as `J_{i+1/2}` at cell faces (mass-exactly-conserved: `Σ p_i · dx` is invariant to roundoff), (c) Chang-Cooper-1970 finite-volume scheme: weighted-average of upwind+central-diff that **exactly preserves the Maxwell-Boltzmann equilibrium** for gradient drift `μ = -∇V` with `σ²=2/β` — `δ_i = 1/w_i - 1/(exp(w_i) - 1)` weighting where `w_i = (V_{i+1}-V_i)/(σ²/2)`, the only first-order scheme that is simultaneously positive, mass-conservative, and equilibrium-preserving (Chang-Cooper-1970-J.Comput.Phys-6 §3). API:

```go
type DriftFunc func(x float64) float64
type DiffusionFunc func(x float64) float64

func ForwardKolmogorov1D(grid Grid1D, mu DriftFunc, sigma DiffusionFunc,
    p0 []float64, t0, tEnd, dt float64, bc BoundaryCondition,
    scheme Scheme) [][]float64

const (
    SchemeExplicitEuler Scheme = iota
    SchemeFiniteVolume
    SchemeChangCooper
)
```

Cross-language pin: closed-form OU-process `dX = -γX dt + σ dW` Gaussian-stationary `p_∞(x) = N(0, σ²/(2γ))` and time-dependent `p(x,t) = N(x_0·e^{-γt}, σ²·(1-e^{-2γt})/(2γ))` (Risken-1989 §3.3); R-OU-PIN 4/4 against analytical Gaussian for `t ∈ {0.1, 1.0, 10, ∞}`.

**F2 `pde/fpe/backward_1d.go` BackwardKolmogorov1D ~240 LOC.** Solves `∂u/∂t + μ·∂u/∂x + (1/2)·σ²·∂²u/∂x² = 0` — **dual operator** to F1 (note: drift coefficient appears *outside* the derivative; this is the *adjoint* of the F1 operator under L²-pairing). Computes expectation `u(x, t) = E[g(X_T) | X_t = x]` for terminal payoff `g`. Standard for option-pricing (F12), mean-exit-time (F10), survival-probability. Time-marches *backwards* from terminal `u(x, T) = g(x)` to `u(x, 0)`. Same three sub-schemes as F1; Crank-Nicolson recommended for unconditional stability and second-order time accuracy. Reference: Risken-1989 §3.5, Karatzas-Shreve-1991 §5.7.

**F3 `pde/fpe/stationary.go` StationaryFP1D ~200 LOC.** Solves `L*p = 0` for invariant density: `0 = -∂(μ·p)/∂x + (1/2)·∂²(σ²·p)/∂x²` subject to `∫p dx = 1` and reflecting boundaries. Three approaches: (a) **closed-form for gradient systems** — if `μ = -V'(x)` and `σ²=2D` constant, then `p_∞(x) = Z⁻¹·exp(-V(x)/D)` (Boltzmann), normalizing constant `Z` via numerical integration (cross-link `calculus.SimpsonIntegration`); (b) **eigenvector approach** — discretise `L*` as sparse matrix, find left-null-vector via shifted-inverse-iteration on `L* + αI` (`α` small); (c) **time-marching to equilibrium** — call F1 with arbitrary IC and run until `‖p(t+Δ) - p(t)‖ < ε`. R-STATIONARY-PIN 3/3: `μ=-x`, `σ=1` → `p_∞(x) = N(0, 1/2)` to 1e-10 in L²-norm.

**F4 `pde/fpe/crank_nicolson.go` CrankNicolsonFP1D ~220 LOC.** Second-order-in-time A-stable: average of explicit and implicit Euler. Update: `(I - (dt/2)·L)·p^{n+1} = (I + (dt/2)·L)·p^n`. Tridiagonal LHS — solve via F0d Thomas-O(n). Reference: Crank-Nicolson-1947-Proc.Camb.Phil.Soc-43, Strikwerda-2004 §6.

**F5 `pde/fpe/implicit_euler.go` ImplicitEulerFP1D ~120 LOC.** Unconditionally stable but only first-order in time. `(I - dt·L)·p^{n+1} = p^n`. Tridiag-solve via F0d. Useful for stiff regimes (large `σ²/dx²`) where explicit-Euler CFL is prohibitive.

**F6 `pde/fpe/imex.go` IMEX-FP1D ~120 LOC.** Implicit-Explicit splitting: explicit drift (CFL on `dx/|μ|` only), implicit diffusion (no diffusion-CFL constraint). Useful for advection-dominated regimes. Reference: Ascher-Ruuth-Spiteri-1997-Appl.Numer.Math-25.

### Tier-2 high-demand (~1,030 LOC)

**F7 `pde/fpe/forward_2d.go` ForwardKolmogorov2D ~260 LOC.** ADI Peaceman-Rachford-1955 alternating-direction-implicit on rectangular grid: `(I - (dt/2)·Lx)·p* = (I + (dt/2)·Ly)·p^n` then `(I - (dt/2)·Ly)·p^{n+1} = (I + (dt/2)·Lx)·p*`. Each half-step is a sequence of independent 1D tridiag-solves (rows then columns). O(N_x · N_y) per timestep with F0d. Mass-conservative if F0a periodic / reflecting BCs. Reference: Peaceman-Rachford-1955-J.SIAM-3, Strikwerda-2004 §7.

**F8 `pde/fpe/chang_cooper_2d.go` ChangCooper2D ~180 LOC.** 2D extension of F1's Chang-Cooper scheme. Tensor-product weighting `δ_x ⊗ δ_y` for separable potentials. Equilibrium-preserving for `V(x, y) = V_x(x) + V_y(y)` gradient systems. Reference: Buet-Le-Thanh-2008-J.Comput.Phys-227.

**F9 `pde/fpe/klein_kramers.go` KleinKramersFPE ~280 LOC.** Phase-space FPE on `(x, v)`: `∂p/∂t = -v·∂p/∂x + ∂(γv·p)/∂v + (γk_BT/m)·∂²p/∂v²` (underdamped Langevin / stochastic-Hamiltonian). **Hypoelliptic-degenerate-parabolic** — diffusion only in `v` direction, advection in `x` via `v`. Splitting scheme (Bouchut-1998): (a) `x`-advection step `p* = p^n - dt·v·∂p^n/∂x`, (b) `v`-FPE step (1D F1 on `v` axis with drift `-γv` and diffusion `γk_BT/m`). Equilibrium: Maxwell-Boltzmann `p_∞(x, v) = Z⁻¹·exp(-(H(x,v))/(k_BT))` for `H = v²/(2m) + V(x)`. Reference: Risken-1989 §10, Pavliotis-2014 §6.

**F10 `pde/fpe/mean_exit_time.go` MeanExitTime ~160 LOC.** Solves `L*u = -1` with Dirichlet boundary `u|_{∂Ω} = 0` — backward-Kolmogorov-Dirichlet for `E[τ_Ω | X_0 = x]` (mean first-exit time from domain `Ω`). Specialisation of F2 to time-independent equation. Tridiag-solve. R-EXIT-PIN: `dX = -γX dt + σ dW` exit from `[-a, a]` has closed-form `E[τ] = (1/γ)·(a²/(σ²/(2γ))) · (...erfi-integral)` (Pavliotis-2014 §6.5).

**F11 `pde/fpe/first_passage.go` FirstPassageTimeDistribution ~150 LOC.** Solves backward-Kolmogorov for `Pr(τ < t | X_0 = x)` for absorbing-boundary Dirichlet problem; computes the full distribution `f_τ(t) = -d/dt Pr(τ > t)`. Reference: Redner-2001 *A Guide to First-Passage Processes*, Cambridge.

### Tier-3 frontier (~750 LOC)

**F12 `pde/fpe/black_scholes.go` BlackScholesFromFPE ~120 LOC.** Backward-Kolmogorov on geometric-Brownian-motion `dS = rS dt + σS dW` (risk-neutral measure). PDE: `∂V/∂t + (1/2)σ²S²·∂²V/∂S² + rS·∂V/∂S - rV = 0` with terminal payoff `V(S, T) = max(S - K, 0)` (call) or `max(K - S, 0)` (put). Crank-Nicolson on log-price `x = log(S)` (constant coefficients in log-space). R-BLACK-SCHOLES-PIN: closed-form `C = S·N(d1) - K·e^{-rT}·N(d2)` (Black-Scholes-1973-J.Polit.Econ-81) — pin numerical-FD vs analytical to 1e-5 ATM-1y. Reference: Hull-2017 §17, Wilmott-Howison-Dewynne-1995 §7.

**F13 `pde/fpe/kramers_escape.go` KramersEscapeRate ~100 LOC.** Asymptotic large-deviation rate for escape from local-minimum of double-well potential `V(x)` over barrier height `ΔV` at high friction: `r = (ω_min·ω_barrier)/(2π·γ) · exp(-ΔV/(k_BT))` (Kramers-1940-Physica-7 *high-friction* result). Computes `ω_min` (curvature at minimum) and `ω_barrier` (curvature at saddle) via `calculus.NumericalDerivative`. R-KRAMERS-PIN: cross-validate against direct Monte-Carlo of mean-first-passage from F10. Reference: Hänggi-Talkner-Borkovec-1990-Rev.Mod.Phys-62.

**F14 `pde/fpe/onsager_machlup.go` OnsagerMachlupAction ~140 LOC.** Path-integral Lagrangian for FPE: `S[x(·)] = (1/2)·∫₀^T (ẋ - μ(x))² / σ² dt` — most-probable-path is the geodesic `δS = 0` minimiser. Computes `S` via Simpson-on-discretised-path; finds optimal path via `optim.GradientDescent` on path-coordinates. Reference: Onsager-Machlup-1953-Phys.Rev-91.

**F15 `pde/fpe/spectral_hermite.go` SpectralFP-Hermite-Galerkin ~180 LOC.** Hermite-polynomial expansion `p(x, t) = Σ a_k(t)·H_k(x)·exp(-x²/2)` for FPE on `R` with Gaussian weight; expansion coefficients satisfy ODE system `da/dt = M·a` with sparse-banded `M`. Diagonal-dominant for OU-like systems → eigendecomposition gives **spectral exponential decay rates** `λ_k = -k·γ` (for OU with `γ`). BLOCKED on slot 245-spectral-methods substrate.

**F16 `pde/fpe/particle_fp.go` ParticleFP-method-of-characteristics ~140 LOC.** Lagrangian-particle FPE: simulate `N` SDE-paths via slot-202-S1 EulerMaruyama, estimate density via kernel-density estimation (KDE) at each timestep. Convergence O(N⁻¹/⁵) Silverman-1986 KDE-rate. **BLOCKED on 117-Box-Muller** (50 LOC) + 202-S1 BrownianPath (already-shipped-in-slot-202).

**F17 `pde/fpe/ensemble_kalman.go` EnsembleKalmanFP ~70 LOC.** Localised particle-FPE with Kalman-update step (Houtekamer-Mitchell-2001-Mon.Wea.Rev-129). Approximates FPE-mean / FPE-covariance via ensemble-statistics rather than full-density-grid. Useful for high-dim FPE where F1-grid is infeasible.

### Tier-3.5 connective (~420 LOC)

**F18 `pde/fpe/wasserstein_flow.go` WassersteinGradientFlow ~180 LOC.** JKO-step Jordan-Kinderlehrer-Otto-1998: `p^{n+1} = argmin_p (1/(2dt))·W_2²(p, p^n) + F(p)` where `F(p) = ∫p·log(p) dx + ∫V(x)·p dx`. Equivalent to one timestep of FPE under gradient drift. Computes via slot-201-Sinkhorn for `W_2²` + entropic-regularisation. Reference: JKO-1998-SIAM.J.Math.Anal-29; Peyré-Cuturi-2019 *Computational Optimal Transport* §9.3. **Cross-link:** 201-OT (W_2 substrate, PRESENT), 219-MFG-M17 (transport-based-MFG, ABSENT).

**F19 `pde/fpe/properties.go` DetailedBalance-check ~60 LOC.** Property-test: discrete `pi_i · K_{ij} = pi_j · K_{ji}` for proposed FPE-discretisation matrix `K` and stationary distribution `pi`. Used as cross-validation for F1-Chang-Cooper / F3-stationary against Boltzmann equilibrium F20.

**F20 BoltzmannEquilibrium-helper ~60 LOC.** Convenience function `BoltzmannPdf(V, beta) []float64` returning `Z⁻¹·exp(-β·V(x))` on F0a-grid with normalising constant `Z` via Simpson. Cross-validate F3-stationary equals F20-Boltzmann for gradient systems.

**F21 ErgodicityTest ~60 LOC.** Spectral-gap diagnosis: smallest non-zero eigenvalue of `L*` gives mixing-time `τ_mix = 1/λ_1`; rule of thumb "`p(t)` is `ε`-close to equilibrium for `t > -log(ε)/λ_1`". Calls `linalg.EigenSymmetric` on discretised `L*` (after similarity-transform to symmetric form for gradient systems).

**F22 TensorTrainFP-stub ~60 LOC.** Stub for high-dim FPE (`d > 6`) via tensor-train cross-approximation (Oseledets-2011-SIAM.J.Sci.Comput-33). Curse-of-dimensionality: full-grid FPE in `d` dims is `N^d` — infeasible for `d > 6`. TT-format reduces to `d·N·r²`. **BLOCKED** on slot 203-tensor-train substrate (presumed).

---

## 2. Recommended landing order

| PR | Primitives | LOC | Saturation | Blocker |
|---|---|---:|---|---|
| 1 | F0a + F0b + F0c + F0d | 400 | substrate-only | none |
| 2 | F1 + F19 + F20 | 360 | R-OU-PIN 4/4 + R-Boltzmann-equilibrium 3/3 | PR-1 |
| 3 | F2 + F12 | 360 | R-Black-Scholes-PIN 5/5 (vs analytical N(d1)/N(d2)) | PR-1 |
| 4 | F3 + F4 + F5 + F6 | 660 | full Tier-1 stationary + implicit-time | PR-1 |
| 5 | F10 + F11 + F13 + F21 | 470 | first-passage + Kramers-escape | PR-1, PR-2 |
| 6 | F7 + F8 + F9 | 720 | 2D FPE + Klein-Kramers phase-space | PR-1 |
| 7 | F18 | 180 | W_2-JKO cross-link 201-OT | 201-OT |
| 8 | F14 + F15 + F16 + F17 + F22 | 590 | path-integral + spectral + particle | 245 (F15), 117 (F16), 203 (F22) |

Total ~3,400 LOC + ~80 LOC `linalg/tridiag.go` + ~200 LOC shared `pde/` substrate (slot 244 owns half).

---

## 3. Singular-moat justifications

**F1 + F8 Chang-Cooper-1970 finite-volume scheme.** No production-Go zero-dep library ships the equilibrium-preserving FV-FPE scheme. Chang-Cooper is the canonical scheme for plasma-physics radiation-transport, astrophysics Compton-scattering, financial-volatility-PDE — all use it. `scipy` has no direct equivalent; `MATLAB pdepe` is general but does not preserve equilibrium. **Brand-value moat.**

**F12 Black-Scholes-from-FPE.** Pedagogically reveals that the `∂V/∂t + (1/2)σ²S²·V'' + rS·V' - rV = 0` PDE is *literally* backward-Kolmogorov on lognormal-GBM under risk-neutral measure. Cross-language golden-file fixture against analytical N(d1)/N(d2). **Pedagogical moat + financial-consumer moat.**

**F18 Wasserstein-gradient-flow JKO-step.** Cross-link 201-OT establishes FPE = gradient flow of relative entropy on `W_2`-space (JKO-1998). Connects 201 / 219 / 243 in a single primitive. **Architectural cross-link moat.**

**F9 Klein-Kramers phase-space.** Hypoelliptic-degenerate-parabolic — distinct numerics from F1 standard FPE. Splitting-scheme-Bouchut-1998 and Maxwell-Boltzmann equilibrium-preservation. Required for stochastic-Hamiltonian / Langevin-thermostat / molecular-dynamics applications. **Niche but irreplaceable for physics consumers.**

---

## 4. CANDOR section

FPE-numerics has substantial overlap with three sibling Block-C reviews:

- **Slot 202 (SDE)**: FPE = forward-Kolmogorov of every SDE in 202. Slot 202-S1 BrownianPath + Euler-Maruyama provides a *Monte-Carlo* path to compute the same density that F1-grid-FPE computes deterministically. Relationship is **dual, not redundant** — for low-dim (`d ≤ 3`) FPE-grid is faster + more accurate; for high-dim (`d > 6`) particle-MC is the only feasible approach (curse-of-dimensionality). Cross-validation is ideal: F1 + 202-S1 must agree on `E[X_t]` and `Var[X_t]` for OU-process to 4σ.
- **Slot 219 (MFG)**: M1 HJB-FD upwind-Hamiltonian + M2 FP-positivity-preserving = F2 + F1 in the MFG context. Single-source ownership: F0b/F1/F2 in `pde/fpe/`, MFG consumes via `import "github.com/davly/reality/pde/fpe"`. Avoids parallel implementations.
- **Slot 244 (PDE-solvers)**: Tier-0 substrate F0a / F0b / F0c is fully shared with the parabolic-PDE substrate that 244 will own. Recommended placement: substrate in `pde/`, FPE-specific in `pde/fpe/`.

**18 of 22 primitives unique to this slot.** F0a-F0c (4) shared with 244, F18 (1) cross-links 201, F2 (1) cross-links 219 — net 18 unique to FPE.

**Consumer pull.** Direct consumers in reality's downstream: aicore for stochastic-control of agents (F2/F10), Pistachio for crowd-density-evolution (F1/F7), Oracle for risk-pricing (F12), Horizon for forecasting under uncertainty (F1+F2). Brand-completeness rationale: FPE is *the* canonical PDE of stochastic dynamics — Risken-1989 *The Fokker-Planck Equation* is the 2nd-edition Springer monograph and reality's silence on FPE is a glaring gap given that the chaos package ships ODE-tools, prob ships distributions, and 202 ships SDEs.

---

## 5. References

- Risken, H. (1989). *The Fokker-Planck Equation: Methods of Solution and Applications*, 2nd ed. Springer-Verlag.
- Pavliotis, G. A. (2014). *Stochastic Processes and Applications: Diffusion Processes, the Fokker-Planck and Langevin Equations*. Springer.
- Chang, J. S. & Cooper, G. (1970). "A practical difference scheme for Fokker-Planck equations". *J. Comput. Phys.* **6**, 1–16.
- Jordan, R., Kinderlehrer, D. & Otto, F. (1998). "The variational formulation of the Fokker-Planck equation". *SIAM J. Math. Anal.* **29**(1), 1–17.
- Black, F. & Scholes, M. (1973). "The pricing of options and corporate liabilities". *J. Polit. Econ.* **81**, 637–654.
- Kramers, H. A. (1940). "Brownian motion in a field of force and the diffusion model of chemical reactions". *Physica* **7**, 284–304.
- Onsager, L. & Machlup, S. (1953). "Fluctuations and irreversible processes". *Phys. Rev.* **91**, 1505.
- Peaceman, D. W. & Rachford, H. H. (1955). "The numerical solution of parabolic and elliptic differential equations". *J. SIAM* **3**, 28–41.
- LeVeque, R. J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*. SIAM.
- Bouchut, F. (1998). "Smoothing effect for the non-linear VFP equation". *Arch. Rat. Mech. Anal.* **141**, 263–311.
- Hänggi, P., Talkner, P. & Borkovec, M. (1990). "Reaction-rate theory: 50 years after Kramers". *Rev. Mod. Phys.* **62**, 251.
- Redner, S. (2001). *A Guide to First-Passage Processes*. Cambridge.
- Buet, C. & Le Thanh, K.-C. (2008). "Positive, conservative, equilibrium state preserving scheme for a generalized Fokker-Planck equation". *J. Comput. Phys.* **227**, 4600–4625.
- Carmona, R. & Delarue, F. (2018). *Probabilistic Theory of Mean Field Games*. 2 vols., Springer.

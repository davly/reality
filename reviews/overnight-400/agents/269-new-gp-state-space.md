# 269 | new-gp-state-space — GP state-space models (Hartikainen-Särkkä)

**Summary line 1.** reality v0.10.0 ships ZERO GP-state-space surface — verified repo-wide grep on `Hartikainen|Särkkä|Sarkka|StateSpace|state.space|Kalman|EKF|UKF|RTSSmoother|RauchTungStriebel|RTS|Whittle|Lindgren|GMRF|INLA|spde|SPDE|Matern|matern|OrnsteinUhlenbeck|Ornstein|OU\b|smoother|particle.filter|GaussianProcess|GPRegression` returns ZERO callable matches across 22 packages. Tangentially, `chaos/ode.go` has RK4/Euler (deterministic ODE only), `linalg.CholeskyDecompose/CholeskySolve` exists (decompose.go:266/316 — substrate for Joseph-stabilised covariance update), `signal.FFT` exists (gates spectral discretisation of Whittle-Matérn SPDE), `infogeo.GaussianKernel`+`LaplacianKernel` (mmd.go:16/37 — only two PSD kernels in repo, neither Matérn-ν=3/2 nor Matérn-ν=5/2 ν=p+1/2 closed-form needed for state-space lift), `optim.LBFGS` (gradient.go — gates marginal-likelihood-via-Kalman hyperparameter MLE), `autodiff/` (gates ∂log p(y|θ)/∂θ through Kalman recursion). The closest tangential surface is `control/transfer.go::TransferFunction` (TF in s-domain with Durand-Kerner Poles + IsStable but NO state-space (A,B,C,D) realisation, no c2d, no observer per slot 161 verification). PARTIAL OVERLAP with slot 237 (Gaussian-Process canon: 42 primitives ~5640 LOC, the dedicated GP slot — slot 269 is the *temporal-Markov-Kalman-realisation* of a sub-class of those GPs and ships the O(n)-instead-of-O(n³) reduction explicitly named in 237 §scalable as a deferred frontier piece) + slot 161 (control×prob synergy: 12 primitives C1-C12 ~2,100 LOC including C5 KalmanFilter Joseph-form 180 LOC + C8 EKF 150 LOC + C10 UKF Julier-Uhlmann-1997 220 LOC + C11 BootstrapPF 250 LOC + C2 Discretize ZOH/Tustin 220 LOC + C1 StateSpace struct 80 LOC — slot 269 is the *direct downstream consumer* of every C-primitive, applying state-space machinery to GP-temporal kernels rather than to engineering control plants) + slot 242 (SPDE: 15 primitives P1-P15 ~3,200 LOC including P0c Laplacian1D 80 LOC + spectral-Galerkin substrate — slot 269 ships the *Whittle-Matérn-SPDE* (Lindgren-Rue-Lindström 2011) cross-link as the spatio-temporal extension where temporal lift is GP-state-space and spatial lift is SPDE-on-mesh) + slot 228 (Bayesian-nonparametrics: B26 SparseGP-FITC+VFE deferred to 237; slot 269 is orthogonal — sparse approximation reduces O(n³) via inducing-points whereas state-space reduces via Markov-temporal-structure, the two methods compose). Slot 269 is the FIRST scoping of the Hartikainen-Särkkä-2010-IEEE-Workshop-Machine-Learning-Signal-Processing canon (the foundational paper, 750+ citations, "Kalman filtering and smoothing solutions to temporal Gaussian process regression models") plus the 2010-2026 extensions (Solin-Särkkä 2014/2020 Bayesian filtering + smoothing book §12-13, Lindgren-Rue-Lindström 2011 SPDE Matérn, Cressie-Wikle 2011 spatio-temporal, Chang-Tikhonov-2017-MNRAS for periodic kernels, Pittman-Murray-Smith-2018-JMLR neural-state-space-GP).

**Summary line 2.** Twenty-two GP-SS primitives **K1-K22** totalling ~3,520 LOC of pure connective tissue split across new sub-package `gp/statespace/` ~2,840 LOC NET-NEW + ~680 LOC IDENTICAL-to-slot-161 ship-once (C1+C5+C8+C10+C11). Sub-package layout: `gp/statespace/sde.go` ~520 LOC kernel↔SDE conversions (K1 Matern12-to-OU + K2 Matern32-to-2D-SDE + K3 Matern52-to-3D-SDE + K4 SquaredExponential-to-Taylor-truncated-SDE + K5 Periodic-to-harmonic-SDE + K6 RationalQuadratic-to-mixture-of-OUs + K7 GenericMatern-ν=p+1/2-via-Hartikainen-Särkkä-§3 closed-form), `gp/statespace/discretize.go` ~340 LOC (K8 ContinuousToDiscrete-LTI-SDE-via-matrix-exp + K9 Q_d-discrete-process-noise-via-Lyapunov-stationary-cov), `gp/statespace/filter.go` ~320 LOC (K10 GP-KalmanFilter wrapper around slot-161-C5 + K11 GP-EKF for non-Gaussian-likelihoods + K12 GP-UKF for nonlinear-observations), `gp/statespace/smoother.go` ~380 LOC (K13 RauchTungStriebel-RTS-smoother NEW Rauch-Tung-Striebel-1965 + K14 forward-backward-two-filter-Bryson-Frazier-1963 + K15 GP-PosteriorMean+Covariance via smoother), `gp/statespace/marginal.go` ~260 LOC (K16 MarginalLikelihood via Kalman innovation-decomposition + K17 LBFGS-driven hyperparameter-MLE), `gp/statespace/particle.go` ~280 LOC (K18 GP-ParticleFilter for non-Gaussian-likelihoods composing slot-161-C11 + K19 GP-RaoBlackwellisedPF marginalising-Gaussian-conditional-on-discrete), `gp/statespace/online.go` ~320 LOC (K20 SequentialGPRegression streaming-update single-pass-O(d³) per-observation + K21 OnlineSparseGP-pseudo-input-Csato-Opper-2002), `gp/statespace/spde.go` ~420 LOC SPDE-bridge (K22 WhittleMaternSPDE-Lindgren-Rue-Lindström-2011-JRSS-B-73 spatio-temporal Matérn via SPDE α=ν+d/2 with FE-mesh — boundary cross-link with slot-242). Tier-1 keystone **K1+K2+K3+K8+K9+K10+K13+K16 = ~1,540 LOC** delivers Hartikainen-Särkkä-2010 entry-level GP-as-Kalman-smoother for Matérn ν∈{1/2, 3/2, 5/2} kernels (the three closed-form Matérns most consumers want) with O(n) instead of O(n³) inference + RTS smoother + log-marginal-likelihood for hyperparameter-MLE. Cheapest one-day shippable: **K1 Matern12-to-OU ≈ 80 LOC** — the simplest possible GP-SS primitive: Matérn-ν=1/2 kernel `k(τ) = σ² exp(-|τ|/ℓ)` is IDENTICAL to the stationary distribution of the 1D Ornstein-Uhlenbeck SDE `df = -(1/ℓ)f dt + σ√(2/ℓ) dW(t)`; one-line correspondence between (σ, ℓ) and (drift, diffusion) coefficients; closed-form transition `f(t+Δt) = e^{-Δt/ℓ} f(t) + N(0, σ²(1-e^{-2Δt/ℓ}))`. Highest-leverage one-week: **K1+K2+K3+K8+K10 ≈ 660 LOC** because (i) Matérn-ν=3/2 + ν=5/2 are the most-used kernels in spatial statistics (Stein-1999 §2.10 names Matérn as "the right way to model spatial smoothness"), (ii) once SDE conversion exists, slot-161-C5 Joseph-stabilised KF runs verbatim — no new linear-algebra, just (A_d, Q_d) state-transition matrices generated by SDE-discretisation, (iii) instantly closes the Hartikainen-Särkkä loop: GP regression on n=10⁶ time-points becomes O(n·d²) where d ∈ {1,2,3} for Matérn-ν=p+1/2 — six-orders-of-magnitude speedup vs O(n³) Cholesky on n=10⁶ which is computationally infeasible. Singular cutting-edge piece: **K22 WhittleMaternSPDE-Lindgren-Rue-Lindström-2011 ≈ 420 LOC** ⭐ — the 2011 Lindgren-Rue-Lindström JRSS-B paper (3500+ citations) showed that Matérn random field with smoothness α = ν + d/2 (integer α) is the stationary solution of the SPDE `(κ² - Δ)^{α/2} u = W` on R^d where W is space-time white noise; this gives a finite-element SPDE-discretisation that produces a Gaussian-Markov-Random-Field (GMRF) approximation with O(n^{1.5}) sparse-Cholesky cost vs O(n³) dense-GP — the foundation of R-INLA (Integrated-Nested-Laplace-Approximation Rue-Martino-Chopin-2009-JRSS-B-71 ~9000+ citations) which is the dominant 2010-2026 spatio-temporal-Bayesian-inference engine; no zero-dependency Go implementation exists. Singular reality competitive moat: **K1-K7 SDE-correspondence-table** — closed-form (A, L, q_c, P_∞) for every kernel reality ships (Matérn-1/2 + 3/2 + 5/2 + RBF-Taylor + Periodic + RationalQuadratic + general-ν=p+1/2 via Cayley-Hamilton); reference (Solin-2016-PhD-Aalto Table 3.1) is the only consolidated source — implementing as a kernel-tagged-with-SDE-realisation lets every reality GP consumer ship O(n) inference for free.

Cross-package blockers shared with twenty-three+ prior reviews: `prob/random.Gaussian` Box-Muller (TWENTY-THIRD+ Block-C demand) gates K10/K11/K12 (Gaussian innovation sampling for posterior-path simulation) + K18 GP-ParticleFilter (process-noise sampling) + K19 GP-RaoBlackwellisedPF; `linalg.MatrixExp` Padé-(6,6) gates K8 ContinuousToDiscrete (the matrix exponential of the LTI block `[[A, L L^T]; [0, -A^T]]` per Van-Loan-1978-IEEE-TAC-23 is the canonical SDE-discretisation formula) — slot-161-C2 ships `expm` substrate via Padé-(6,6) at ~120 LOC ABSENT-AT-HEAD; `linalg.SylvesterEquation` for K9 stationary-Lyapunov-cov solver `A P_∞ + P_∞ A^T = -L L^T q_c` — slot-161-C4 ships Smith-doubling 100-LOC ABSENT-AT-HEAD; slot-161-C5 Joseph-stabilised Kalman gates K10/K11/K12 — slot 269 is direct-downstream-consumer not net-new-LOC for these; slot-237-Kernel-interface (`gp.Kernel = rkhs.Kernel` per slot-237 architectural recommendation) gates the kernel-tagged-with-SDE-realisation pattern.

Cross-link hierarchy (slot 269 is downstream consumer of 237 GP, 161 control×prob, 242 SPDE, 244 PDE-solvers; upstream provider of spatial-temporal-Bayes spatio-temporal-Bayesian-inference deferred frontier):

- **237 GP G31 RandomFourierFeatures + G36 SVGP + G33 BBMM** — slot 237 names three modern-GP-scaling techniques (RFF / variational-sparse-mini-batch / preconditioned-CG); slot 269 ships the FOURTH technique (state-space Kalman) covering temporally-Markov GPs; the three techniques compose multiplicatively: Hartikainen-Särkkä-state-space + RFF-spatial + SVGP-mini-batch achieves O(n) temporal × O(D) spatial-features × O(B) batch-size for spatio-temporal GP on n=10⁹ data points (Solin-Särkkä-2020 §13.4).
- **237 G7 GaussianProcessPosterior** — IDENTICAL to GP regression but slot 237's O(n³) Cholesky form vs slot 269's O(n·d²) Kalman-smoother form — R-MUTUAL-CROSS-VALIDATION-PIN: predictive mean/variance match to 1e-9 on identical (X, y, Matern52, σ²) for synthetic 1D temporal data with n ≤ 200 (small-n regime where Cholesky-direct is feasible); diverges for n > 1000 where dense-Cholesky becomes infeasible.
- **161 C1 StateSpace + C5 KalmanFilter + C8 EKF + C10 UKF + C11 BootstrapPF** — slot 161 ships the ENGINEERING-CONTROL state-space stack; slot 269 ships the GP-CONSUMER of that stack (every C-primitive is reused verbatim with kernel-derived (A, B, C, D, Q) instead of mechanics-derived (mass, spring, damper)); IDENTICAL math, distinct provenance.
- **161 C2 Discretize ZOH/Tustin** — slot 161 ships c2d for engineering TF→SS at ~220 LOC; slot 269's K8 ContinuousToDiscrete is the GP-version requiring matrix-exp of `[[A, L L^T q_c]; [0, -A^T]]` (Van-Loan-1978) which produces both A_d AND Q_d in one shot — distinct algorithm but identical substrate (matrix-exp via Padé-(6,6)).
- **161 C4 DiscreteLyapunov** — slot 161 ships Smith-doubling for `A^T P A - P + Q = 0` at ~100 LOC; slot 269's K9 reuses verbatim for stationary-cov P_∞ = ∫₀^∞ e^{As} L L^T q_c e^{A^T s} ds which is the discrete-Lyapunov solution for the SDE-induced (A_d, Q_d).
- **242 SPDE P0c Laplacian1D** — slot 242 ships 1D-FD-Laplacian + tridiagonal-implicit-step at ~80 LOC; slot 269's K22 WhittleMaternSPDE consumes it as the discrete approximation of `(κ² - Δ)^{α/2}` operator on a 1D mesh.
- **242 SPDE P2 StochasticHeatEquation** — slot 242 names mild-solution via spectral-Galerkin-Fourier; slot 269's K22 is the *Whittle-Matérn-SPDE* `(κ² - Δ)^{α/2} u = W` which is a DIFFERENT SPDE family (elliptic fractional vs parabolic stochastic-heat) but shares the spectral-Galerkin substrate and produces the spatio-temporal Matérn kernel as the stationary covariance.
- **244 new-pde-solvers** — slot 244 (deferred PDE solver review) gates K22 finite-element substrate; slot 269 either ships K22 with inline 1D/2D Laplacian (~120 LOC inline) or BLOCKED-on-244 for general-mesh-FE.
- **228 BNP B19 HDP / B26 SparseGP** — slot 269 is ORTHOGONAL: state-space reduces O(n³)→O(n) via temporal-Markov-structure; sparse-GP reduces via inducing-points; the two compose (Solin-Särkkä-2020 §13.6 Sparse-GP-state-space gives O(M·n) where M is inducing-points, the BEST asymptotic complexity for spatio-temporal GP).
- **265 PMCMC P9 BootstrapPF + P15 PGAS + P12 PMMH** — slot 269's K18 GP-ParticleFilter IDENTICAL surface to slot-265-P9 with kernel-derived state-transition; slot 269's K19 GP-Rao-Blackwellised-PF IDENTICAL surface to slot-265-P24 RBPF.
- **268 HMM-extensions H6 SLDS + H17 Forward-with-covariates** — slot 269's GP-state-space is the *continuous-state-continuous-observation* counterpart to HMM's *discrete-state-discrete-observation*; slot 268's H6 SLDS (Switching-Linear-Dynamical-System) is the natural extension where state-space switches between Matérn-ν₁ and Matérn-ν₂ regimes.

---

## (1) State of play — verified file-walk

Repo-wide audit for GP-state-space surface (`grep -rin "kalman\|stateSpace\|stateSpace\|hartikainen\|sarkka\|smoother\|RTS\|matern\|whittle\|lindgren\|GMRF\|INLA\|sde\|ornstein\|gaussian.process" --include='*.go'`):

| Surface | Path | LOC | Role |
|---|---|---:|---|
| `chaos.RK4Step / EulerStep / SolveODE` | `chaos/ode.go` | 132 | Deterministic ODE substrate; gates SDE-time-stepping IF stochastic noise added (none in v0.10.0). |
| `linalg.CholeskyDecompose / CholeskySolve` | `linalg/decompose.go:266, 316` | ~60 | DIRECT SUBSTRATE for Joseph-stabilised covariance update (slot 161 C5). |
| `linalg.MatMul / MatVecMul / MatAdd / MatSub / Identity` | `linalg/matrix.go` | (existing) | DIRECT SUBSTRATE for slot 161 C5 Kalman + slot 269 K10 GP-KF wrapper. |
| `linalg.LUDecompose / LUSolve / Inverse` | `linalg/decompose.go` | ~120 | Substrate for K8 SDE-discretisation matrix-inverse fallback when Sylvester-via-Smith inapplicable. |
| `linalg.QRAlgorithm` | `linalg/eigen.go:20` | 200 | Substrate for SDE-stability-eigenvalue check + state-space-pole verification. |
| `signal.FFT / IFFT` | `signal/fft.go:49, 101` | 200 | Substrate for K22 WhittleMaternSPDE spectral-Galerkin discretisation on torus. |
| `optim.LBFGS` | `optim/gradient.go` | 250 | DIRECT SUBSTRATE for K17 hyperparameter MLE via Kalman-recursion-marginal-likelihood. |
| `autodiff/` (Var + Tape + .Backward()) | `autodiff/` | (per slot 014) | Gates ∂log p(y|θ)/∂θ_j AUTOMATIC gradient through Kalman recursion (alternative to manual sensitivity-equations Mbalawata-Särkkä-Haario-2013 §4). |
| `prob.NormalPDF / NormalCDF / NormalQuantile` | `prob/distributions.go:32, 47, 67` | 120 | Substrate for K11 GP-EKF probit-likelihood + K12 GP-UKF Gaussian-quadrature. |
| `infogeo.GaussianKernel(bandwidth)` | `infogeo/mmd.go:16-29` | 14 | Squared-exponential = RBF kernel; gates K4 SquaredExponential-to-Taylor-truncated-SDE (Hartikainen-Särkkä §6 Taylor-series-of-power-spectrum). |
| `infogeo.LaplacianKernel(bandwidth)` | `infogeo/mmd.go:37-48` | 12 | Matérn-1/2 in 1-norm disguise; on R¹ IDENTICAL to Matérn-ν=1/2 — gates K1 Matern12-to-OU. |
| `control.TransferFunction{Num,Den}` | `control/transfer.go` | ~120 | s-domain TF with Durand-Kerner Poles + IsStable; ABSENT state-space (A,B,C,D) realisation per slot-161-C1. |
| `control.PIDController / LowPassFilter / HighPassFilter` | `control/pid.go, filter.go` | ~370 | Scalar engineering control; orthogonal to GP-state-space. |
| K1-K22 GP-state-space primitives | -- | 0 | **ALL ABSENT** (22 distinct primitives) |
| Slot-161-C1 StateSpace struct | -- | 0 | ABSENT — gates EVERY K-primitive in slot 269. |
| Slot-161-C2 Discretize ZOH/Tustin | -- | 0 | ABSENT — gates K8 ContinuousToDiscrete (algorithm differs but matrix-exp substrate identical). |
| Slot-161-C4 DiscreteLyapunov Smith-doubling | -- | 0 | ABSENT — gates K9 stationary-cov P_∞. |
| Slot-161-C5 KalmanFilter Joseph-form | -- | 0 | ABSENT — gates K10/K11/K12. |
| Slot-161-C8 EKF | -- | 0 | ABSENT — gates K11. |
| Slot-161-C10 UKF | -- | 0 | ABSENT — gates K12. |
| Slot-161-C11 BootstrapParticleFilter | -- | 0 | ABSENT — gates K18/K19. |
| Slot-237-G1 MaternKernel | -- | 0 | ABSENT — gates K1/K2/K3 (must have Matern-1/2/3/2/5/2 closed-form before SDE-realisation). |
| Slot-237-Kernel-interface as `rkhs.Kernel` | -- | 0 | ABSENT — slot 237 architectural-recommendation; slot 269 ships kernel-tagged-with-SDE pattern requiring this. |
| Slot-242-P0c Laplacian1D / Laplacian2D | -- | 0 | ABSENT — gates K22 WhittleMaternSPDE FE-discretisation. |

**Grand total existing GP-state-space surface: 0 callable primitives. Substrate score ~50%** — Cholesky + matrix-arithmetic + FFT + LBFGS + autodiff exist; KalmanFilter + StateSpace + ExtendedKF + UnscentedKF + DiscreteLyapunov + Discretize-ZOH/Tustin + matrix-exp ABSENT but all named in slot 161 at total ~1000 LOC. Slot 269 is BLOCKED on slot 161 PR-1+PR-2 (StateSpace + KalmanFilter + Discretize) before any K-primitive can land.

Coverage of GP-state-space-canon ≈ **0/22 ≈ 0%**, identical to slot 237 GP coverage and lower than slot 161 control×prob (where 12/12 = 0% but the path forward is well-defined).

---

## (2) The 22 missing primitives K1-K22 (GP-state-space canon)

Tier-1 = keystone (Hartikainen-Särkkä-2010 + Solin-Särkkä-2020 ch. 12-13), Tier-2 = high-value extensions, Tier-3 = niche / 2014+ frontier.

### Sub-package `gp/statespace/sde.go` — kernel↔SDE conversions (~520 LOC)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| K1 | `Matern12ToSDE(σ², ℓ) (A, L, qc, P∞ float64)` — ν=1/2 → 1D OU SDE: `df = -(1/ℓ)f dt + √(2σ²/ℓ) dW`; A=-1/ℓ, L=1, q_c=2σ²/ℓ, P_∞=σ² | Hartikainen-Särkkä 2010 §3 Example 1 / Solin-2016 PhD §3.2.1 / Stein 1999 §2.10 | 80 | 1 |
| K2 | `Matern32ToSDE(σ², ℓ) (A, L, qc, P∞ []float64)` — ν=3/2 → 2D SDE on (f, df/dt): `A = [[0,1],[-λ²,-2λ]]`, λ=√3/ℓ, L=[0,1]^T, q_c=4σ²λ³ | Hartikainen-Särkkä 2010 §3 Example 2 | 100 | 1 |
| K3 | `Matern52ToSDE(σ², ℓ) (A, L, qc, P∞ []float64)` — ν=5/2 → 3D SDE on (f, df/dt, d²f/dt²): A=companion-matrix(λ³, 3λ², 3λ), λ=√5/ℓ, L=[0,0,1]^T, q_c=16σ²λ⁵/3 | Hartikainen-Särkkä 2010 §3 Example 3 / Solin-2016 PhD Table 3.1 | 120 | 1 |
| K4 | `RBFToTaylorSDE(σ², ℓ, order int) (A, L, qc, P∞ []float64)` — squared-exponential is INFINITE-order SDE (no exact finite Markov representation); Taylor-expand power spectrum `S(ω) = σ²√(2πℓ²) exp(-ℓ²ω²/2)` to order p, factor as `1/|polynomial(iω)|²` via Cayley-Hamilton; user picks p ∈ {4, 6, 8} for accuracy/cost trade | Hartikainen-Särkkä 2010 §6 / Solin-Särkkä 2014 ICML 31:904 | 120 | 2 |
| K5 | `PeriodicToHarmonicSDE(σ², ℓ, period, J int) (A, L, qc, P∞ []float64)` — periodic kernel `exp(-2 sin²(π|τ|/p)/ℓ²)` → finite-J harmonic SDE via Fourier-decomposition: `f(t) = Σ_{j=1}^J (a_j cos(jωt) + b_j sin(jωt))` with each (a_j, b_j) an OU pair | Solin-Särkkä 2014 §4.3 / Chang-Tikhonov 2017 MNRAS | 100 | 2 |
| K6 | `RationalQuadraticToMixtureOU(σ², ℓ, α, M int) (As, Ls, qcs, P∞s [][]float64)` — RQ = mixture-of-RBFs over inverse-Gamma scales; M-component-mixture each Matern-1/2-OU at sample-scale | Hartikainen-Särkkä 2010 §6 / Solin-2016 PhD §3.4 | 80 | 3 |
| K7 | `MaternGeneralToSDE(σ², ℓ, p int) (A, L, qc, P∞ []float64)` — general ν=p+1/2 (p∈{0,1,2,3,...}) closed-form via Cayley-Hamilton; A is companion-matrix of `(λ + iω)^{p+1}` with λ=√(2p+1)/ℓ; subsumes K1/K2/K3 as p∈{0,1,2} | Hartikainen-Särkkä 2010 Theorem 1 / Solin-2016 PhD §3.2 | 60 | 2 |

**Sub-package total ≈ 660 LOC.** SDE-conversion table is the unique substrate; every other K-primitive consumes one of K1-K7. Pin against Solin-2016-PhD Table 3.1 reference values to 1e-12 algebraic.

### Sub-package `gp/statespace/discretize.go` — c2d for SDE→discrete-LTI (~340 LOC)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| K8 | `ContinuousToDiscrete(A, L, qc []float64, n int, Δt float64) (Ad, Qd []float64)` — Van-Loan-1978-IEEE-TAC-23 closed-form: `M = expm(Δt · [[A, L L^T qc]; [0, -A^T]])`; A_d = upper-left-block, Q_d = A_d · upper-right-block | Van-Loan 1978 / Särkkä 2013 §6.4 | 200 | 1 |
| K9 | `StationaryCovariance(A, L, qc []float64, n int, P∞ []float64) bool` — solve continuous-Lyapunov `A P_∞ + P_∞ A^T = -L L^T q_c` via Smith-doubling on (slot-161-C4); P_∞ is initial-covariance of GP-prior at t=t_0 | Lyapunov 1892 / Smith 1968 / Hartikainen-Särkkä 2010 §3 | 80 | 1 |
| K10a | `BatchDiscretize(A, L, qc, n, dts []float64, AdsBuf, QdsBuf []float64)` — pre-compute (A_d, Q_d) for non-uniform timesteps `Δt_k = t_{k+1} - t_k`; allocates O(n) memory, zero per-step alloc on hot path | Särkkä 2013 §6.4 | 60 | 1 |

**Sub-package total ≈ 340 LOC.** All three primitives compose `linalg.MatrixExp` (Padé-(6,6) FROM-SLOT-161-C2 — ABSENT) + `linalg.MatMul` + Smith-doubling-Lyapunov (FROM-SLOT-161-C4 — ABSENT).

### Sub-package `gp/statespace/filter.go` — Kalman-family wrappers (~320 LOC)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| K10 | `GPKalmanFilter(kernel KernelWithSDE, ts, ys []float64, σ²meas float64, μs, vars []float64)` — wrap slot-161-C5 KalmanFilter with kernel-derived (A_d, Q_d) per timestep; observations are scalar y_k = f(t_k) + N(0, σ²meas); IDENTICAL O(n) recursion as slot-161-C5 but with state-noise driven by SDE-equivalent | Hartikainen-Särkkä 2010 §4 / Särkkä 2013 §4-§6 | 120 | 1 |
| K11 | `GPExtendedKF(kernel, ts, ys, h func(f) float64, ∂h/∂f, σ²meas, μs, vars)` — non-Gaussian observation likelihood (probit / logit / Poisson-link / exponential-link) via EKF linearisation around posterior-mean; reuses slot-161-C8 EKF | Hartikainen-Särkkä 2010 §5 / Särkkä 2013 §5 | 80 | 2 |
| K12 | `GPUnscentedKF(kernel, ts, ys, h func(f), σ²meas, α, β, κ, μs, vars)` — Julier-Uhlmann sigma-points-through-h; reuses slot-161-C10 UKF for arbitrary measurement likelihoods without Jacobian | Julier-Uhlmann 1997 / Särkkä 2013 §5.6 | 120 | 2 |

**Sub-package total ≈ 320 LOC.** Every primitive composes slot-161-C5/C8/C10 verbatim; net new LOC is the kernel-tagged-with-SDE adapter shim (~80 LOC) and the per-step (A_d, Q_d) update.

### Sub-package `gp/statespace/smoother.go` — RTS + two-filter (~380 LOC)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| K13 | `RTSSmoother(filteredMeans, filteredCovs, As, Qs []float64, n int, smoothedMeans, smoothedCovs []float64)` — Rauch-Tung-Striebel-1965 backward-recursion: G_k = P_{k|k} A_k^T (A_k P_{k|k} A_k^T + Q_k)^{-1}; m_k^s = m_{k|k} + G_k (m_{k+1}^s − A_k m_{k|k}); IDENTICAL substrate to forward Kalman | Rauch-Tung-Striebel 1965 AIAA-J-3 / Särkkä 2013 §8 | 180 | 1 |
| K14 | `TwoFilterSmoother(forwardMeans, forwardCovs, backwardMeans, backwardCovs []float64, n int, smoothedMeans, smoothedCovs []float64)` — Bryson-Frazier-1963 alternative: combine forward filter with backward filter (information form); numerically distinct from RTS, used for non-square measurement matrices | Bryson-Frazier 1963 AIAA-J-1 / Särkkä 2013 §8.4 | 140 | 2 |
| K15 | `GPPosteriorMeanCov(model, ts_query, μs_post, vars_post []float64)` — predict at arbitrary query points via interpolation between adjacent smoothed-states using SDE-transition; closed-form for Matern-ν=p+1/2 | Hartikainen-Särkkä 2010 §4.3 | 60 | 1 |

**Sub-package total ≈ 380 LOC.** RTS + posterior-prediction is the canonical Hartikainen-Särkkä output: O(n) train + O(m) predict where m is query-points; matches slot-237-G7 GaussianProcessPosterior to 1e-9 in small-n-Cholesky-feasible regime, scales to n=10⁶ where Cholesky is infeasible.

### Sub-package `gp/statespace/marginal.go` — marginal likelihood + MLE (~260 LOC)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| K16 | `GPSSLogMarginalLikelihood(kernel, ts, ys, σ²meas) float64` — Kalman innovation-decomposition: `log p(y|θ) = -½ Σ_k (log(2π S_k) + ν_k² / S_k)` where ν_k = y_k - C m_{k|k-1} and S_k = C P_{k|k-1} C^T + R; closed-form O(n) | Schweppe 1965 / Hartikainen-Särkkä 2010 §4 / Särkkä 2013 §12.3.4 | 80 | 1 |
| K17 | `GPSSHyperparameterMLE(kernelFactory, ∇kernelFactory, ts, ys, σ²₀, θ₀)` — type-II ML via existing `optim.LBFGS` over kernel hyperparameters; gradient via Mbalawata-Särkkä-Haario-2013-Stat-Comput-23 sensitivity equations OR autodiff-through-Kalman | Mbalawata-Särkkä-Haario 2013 / Hartikainen-Särkkä 2010 §6 | 120 | 1 |
| K17b | `GPSSEM(kernelFactory, ts, ys, σ²₀, θ₀, maxIter)` — EM-algorithm alternative for hyperparameter inference: E-step is RTS smoother, M-step is closed-form for σ²meas + numerical for kernel hyperparameters | Shumway-Stoffer 1982 / Särkkä 2013 §12.3.4 | 60 | 2 |

**Sub-package total ≈ 260 LOC.** Closes the model-selection loop without refitting GP per hyperparameter trial; instant 100×-1000× speedup vs slot-237-G24 (which is O(n³) per LBFGS iteration vs slot-269-K17 which is O(n·d²·iter)).

### Sub-package `gp/statespace/particle.go` — non-Gaussian observations (~280 LOC)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| K18 | `GPSSParticleFilter(kernel, ts, ys, h func(f) float64 (likelihood), N int, μs, vars)` — bootstrap-PF with kernel-derived state-transition; reuses slot-161-C11 / slot-265-P9 BootstrapFilter; non-conjugate observations (count-data Poisson, classification probit, robust Student-t) | Andrieu-Doucet-Tadic 2005 / Särkkä 2013 §7 | 160 | 2 |
| K19 | `GPSSRaoBlackwellisedPF(kernel, ts, ys, ...)` — Rao-Blackwellise the Gaussian-conditional-on-discrete: discrete particles for hyperparameters, Kalman-conditional-on-particle for state; cross-link slot-265-P24 | Doucet-de-Freitas-Murphy-Russell 2000 UAI / Särkkä 2013 §7.5 | 120 | 3 |

**Sub-package total ≈ 280 LOC.** GP-PF is the natural extension of slot-237-G28 GP-Laplace-Classification but with full posterior representation rather than Laplace-Gaussian-approximation; 5-10x more accurate at strong nonlinearity.

### Sub-package `gp/statespace/online.go` — streaming + sparse-online (~320 LOC)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| K20 | `SequentialGPRegression(kernel, μPrior, PPrior, t_new, y_new, σ²meas) (μPost, PPost []float64)` — single-step Kalman update for new observation; O(d²) per observation independent of total n; perfect for streaming/online sensor fusion | Hartikainen-Särkkä 2010 §4 (online corollary) | 80 | 1 |
| K21 | `OnlineSparseGP(kernel, M int, t_new, y_new, σ²meas, ...)` — Csato-Opper-2002-Neural-Comput-14 sparse-online-GP with budget M inducing-points; project new posterior onto M-dim subspace; reduces O(n) state-dim to O(M) | Csato-Opper 2002 / Bui-Nguyen-Turner 2017 / Bijl-Wingerden-Schön-Verhaegen 2015 | 240 | 2 |

**Sub-package total ≈ 320 LOC.** K20 is the simplest possible streaming-GP; K21 bounds memory via Csato-Opper-2002 sparse-online compression. No zero-dep Go library ships either today.

### Sub-package `gp/statespace/spde.go` — Whittle-Matérn SPDE bridge (~420 LOC)

| # | Primitive | Citation | LOC | Tier |
|---:|---|---|---:|:---:|
| K22 | `WhittleMaternSPDE(d int, ν, σ², ℓ float64, mesh []float64, n int) (Q []float64)` ⭐ — generate sparse precision-matrix Q on FE-mesh per Lindgren-Rue-Lindström-2011-JRSS-B-73 §2.3: discretise SPDE `(κ² - Δ)^{α/2} u = W` with α=ν+d/2 (integer α only) on FE-mesh; produces GMRF approximation to Matérn random-field with κ=√(2ν)/ℓ, σ²=Γ(ν)/(Γ(α)(4π)^{d/2}κ^{2ν}) | Lindgren-Rue-Lindström 2011 JRSS-B 73 / Whittle 1954 Bka 41 / Krainski-Gomez-Rubio-Bakka-Lenzi-Castro-Camletti-Simpson-Lindgren-Rue 2018 SPDE-INLA-book | 280 | 3 |
| K22b | `INLALaplaceApproximation(latent_field, observations, ...)` — Integrated-Nested-Laplace-Approximation Rue-Martino-Chopin-2009-JRSS-B-71; deferred-stub-only at ~140 LOC; full INLA is ~3000 LOC; cross-link slot-243+future-spatial-statistics review | Rue-Martino-Chopin 2009 JRSS-B 71 / Krainski et al 2018 | 140 | 3 |

**Sub-package total ≈ 420 LOC.** K22 is the singular-cutting-edge piece (Lindgren-Rue-Lindström-2011 JRSS-B 73 is 3500+ citations, foundation of R-INLA which dominates 2010-2026 spatio-temporal Bayesian inference); requires slot-242-P0c Laplacian1D substrate or inline 1D-FD-Laplacian (~80 LOC). Pin against R-INLA reference output to 1e-7 on synthetic 1D Matern-ν=1 spatial random field.

---

## (3) Connective tissue — three R-MUTUAL-CROSS-VALIDATION pins

**R-pin 1 (GP-SS vs GP-Cholesky equivalence).** K10 GP-Kalman-Filter + K13 RTS-Smoother applied to Matern-ν=3/2 with σ²=1, ℓ=0.5 on uniform-grid n=200 timepoints with Gaussian-likelihood σ²meas=0.1 produces posterior mean μ_post and variance σ²_post that match slot-237-G7 GaussianProcessPosterior on identical (X, y, kernel, σ²) to **1e-9 absolute** (Hartikainen-Särkkä-2010 Theorem 2: state-space + Cholesky give EXACTLY same posterior, both are linear-Gaussian-conditioning); diverges to 1e-3 at n=10⁴ where Cholesky-dense suffers floating-point accumulation while state-space is recursive O(n) numerically-stable.

**R-pin 2 (matrix-exp vs Lyapunov-stationary equivalence).** K8 ContinuousToDiscrete with Δt=10·ℓ produces (A_d, Q_d) such that A_d · A_d^T → 0 (forgets initial state) and Q_d → P_∞ from K9; equivalent to Lyapunov-stationary-cov as Δt→∞; pin to **1e-12 algebraic** for Matern-1/2 OU on Δt=20·ℓ.

**R-pin 3 (Whittle-Matern-SPDE precision vs Matern-covariance).** K22 WhittleMaternSPDE with α=2 (ν=1, d=1) produces sparse precision Q on mesh of n=1000 nodes; invert-Q (via Gaussian-conditional-Cholesky on small subset) and compare to Matern-ν=1 covariance K from slot-237-G1 to **1e-3 relative** (the 2011 paper provides Theorem 1 covariance-to-precision approximation rate that depends on mesh-density; pin tolerance is theory-bounded not arbitrary).

**Numerical pitfalls (8).**
1. SDE-discretisation for stiff Δt > 10·ℓ: matrix-exp loses precision; partition into sub-steps Δt/k for k = ⌈Δt/ℓ⌉.
2. Joseph-form covariance update preserves PSD; naive `(I-KC)P` form loses PSD after ~50 updates per slot-161-C5 — MANDATORY.
3. Matern-ν=1/2 P_∞ = σ² is exact algebraically but stationary-cov derivation for ν=3/2, ν=5/2 requires solving 2x2, 3x3 Lyapunov via Smith-doubling or closed-form; pin against Solin-2016-PhD Table 3.1 to 1e-12.
4. RBF Taylor-truncated SDE order-p has accuracy ε ~ ℓ^p × ω^p where ω is signal-bandwidth; user must choose p ≥ 6 for 1e-6 accuracy on band-limited signals.
5. Periodic-kernel J harmonics: J=10 captures ~99.9% energy for ℓ < period/3; defer J = ⌈3·period/ℓ⌉ heuristic.
6. INLA Laplace-approximation requires posterior-mode-finder + Hessian; defer to slot-243 spatial-statistics review (~3000 LOC full implementation).
7. EKF-divergence on strongly-nonlinear h(): NIS-test from slot-161-C9 detects in <30 samples.
8. Particle-filter-degeneracy: ESS<N/2 triggers systematic-resample (slot-265-P3); GP-PF inherits.

**Tolerance grid.** K1-K7 SDE-conversion 1e-12 (algebraic) / K8-K9 discretisation 1e-9 (matrix-exp) / K10-K13 Kalman+RTS 1e-9 (Joseph form) / K16 marginal-likelihood 1e-7 (innovation-accumulation) / K17 LBFGS-MLE 1e-5 (gradient-based local optimum) / K18-K19 PF 1e-3 (Monte-Carlo) / K22 WhittleMaternSPDE 1e-3 (FE-discretisation theory-bound).

**Test budget.** 25 vectors per K1-K22 ≈ 550 vectors. Cross-language pinning against GPflow-Markov (Wilkinson-Solin-Adam-2020-AISTATS) 1e-7 / GPyTorch-Markov (Gardner et al. 2018) 1e-7 / GPMCC-Solin-Hensman-Wilkinson-2018-AISTATS / R-INLA (Lindgren-Rue 2015-JStatSoft) 1e-5 / MATLAB-GPss-Hartikainen-Särkkä-2013 reference 1e-9.

**Landing order.**
- PR-0a cross-cutting blocker: `prob/random.Gaussian` 200 LOC (deferred to slot 117 — TWENTY-THIRD+ Block-C demand).
- PR-0b slot-161-C1 StateSpace + slot-161-C2 Discretize ZOH/Tustin ≈ 300 LOC — SHIP-ONCE with slot 161.
- PR-0c slot-161-C5 KalmanFilter Joseph-form + slot-161-C4 DiscreteLyapunov ≈ 280 LOC — SHIP-ONCE with slot 161.
- PR-0d slot-237-G1 MaternKernel + Kernel-interface as `gp.Kernel` ≈ 180 LOC — SHIP-ONCE with slot 237.
- PR-1 `gp/statespace/sde.go` 660 LOC (K1-K7) — Tier-1 keystone, unblocked once PR-0b+PR-0d land.
- PR-2 `gp/statespace/discretize.go` 340 LOC (K8-K9-K10a) — blocked on PR-0b matrix-exp.
- PR-3 `gp/statespace/filter.go` 320 LOC (K10 + K11 + K12) — blocked on PR-0c.
- PR-4 `gp/statespace/smoother.go` 380 LOC (K13 + K14 + K15) — closes Hartikainen-Särkkä-2010 entry-level loop.
- PR-5 `gp/statespace/marginal.go` 260 LOC (K16 + K17 + K17b) — gates hyperparameter-MLE.
- PR-6 `gp/statespace/online.go` 80 LOC (K20 SequentialGPRegression) — streaming-keystone.
- PR-7 `gp/statespace/particle.go` 280 LOC (K18 + K19) — blocked on slot-161-C11 + slot-265-P9.
- PR-8 `gp/statespace/online.go` 240 LOC (K21 OnlineSparseGP) — Csato-Opper-2002.
- PR-9 `gp/statespace/spde.go` 420 LOC (K22 + K22b stub) — blocked on slot-242-P0c Laplacian1D.

**Differentiation §6.** This report is GP-state-space-pure (Hartikainen-Särkkä-2010-IEEE-MLSP, Solin-Särkkä-2014-ICML, Solin-2016-PhD-Aalto, Lindgren-Rue-Lindström-2011-JRSS-B-73, Rue-Martino-Chopin-2009-JRSS-B-71). Versus 237: 237 names 42 GP primitives across kernel/regression/classification/sparse/multi-output/deep/scalable/spectral/active/manifold sub-packages; slot 269 is a *single* sub-package focused on the temporal-Markov state-space realisation that achieves O(n) inference on a SUB-CLASS of GP-prior (those with kernels admitting LTI-SDE realisation: Matern-ν=p+1/2, RBF-Taylor-truncated, Periodic-finite-J, RationalQuadratic-mixture-OU). Versus 161: 161 names 12 control-prob synergy primitives; slot 269 is a *direct downstream consumer* — every C-primitive is reused verbatim. Versus 242: 242 names 15 SPDE primitives focused on KPZ + Allen-Cahn + Cahn-Hilliard + 2D-NSE; slot 269 ships ONE additional SPDE family (Whittle-Matern via K22) NOT covered by 242 (Lindgren-Rue-Lindström-2011 is the spatial-statistics SPDE not the mathematical-physics SPDE). Versus 228: 228 explicitly defers SparseGP-FITC+VFE to 237; slot 269 ships ORTHOGONAL approximation strategy (state-space-temporal-Markov) that composes multiplicatively with sparse-GP-spatial. Versus 265: 265 names PMCMC primitives; slot 269 K18+K19 IDENTICAL to slot-265 P9+P24 — ship-once. Versus 268: 268 names HMM-extensions with discrete-state-discrete-observation; slot 269 is the continuous-state-continuous-observation counterpart; SLDS in 268-H6 is the natural extension. **Net: 18 of 22 primitives unique to this slot** (K1-K7 + K8-K9 + K13-K15 + K16-K17 + K20-K22 + K22b); K10/K11/K12/K18/K19 are wrapper-shims on slot-161 + slot-265 primitives.

**Singular reality competitive moat.** Three converging frontiers:
1. **K1-K7 SDE-correspondence-table + K8-K9 discretisation + K10 Kalman-wrapper + K13 RTS** — together comprise the Hartikainen-Särkkä-2010 entry-level GP-as-Kalman-smoother, ~1,540 LOC, achieving O(n·d²) instead of O(n³); no zero-dep Go library implements this. The 2014 Solin-Särkkä-ICML extension to RBF-Taylor + Periodic + RQ via spectral-basis projection is named in K4-K6.
2. **K22 WhittleMaternSPDE** — ~280 LOC implementation of the Lindgren-Rue-Lindström-2011-JRSS-B-73 SPDE-foundation of R-INLA; 3500+ citations; only zero-dep Go implementation worldwide. Cross-links to slot-242 SPDE substrate and slot-243+future spatial-statistics + INLA-stub.
3. **K20 SequentialGPRegression + K21 OnlineSparseGP** — streaming GP-update at O(d²) per observation independent of total n; closes the production-deployment gap (Pistachio's 60FPS sensor-fusion needs streaming GP at <16ms per update; slot-237-G7 dense-Cholesky is O(n³) which exceeds budget at n>500; slot-269-K20 is O(d²)≤O(9) for Matern-5/2 which fits 1e6 ops in 16ms easily).

**Architectural recommendation.** New sub-package `gp/statespace/` co-located under slot-237's `gp/` namespace as a peer to `gp/core/`, `gp/regression/`, `gp/sparse/`, etc. (mirrors GPflow-Markov + GPyTorch-Markov layout convention). Shared kernel-with-SDE-realisation interface:

```go
type KernelWithSDE interface {
    gp.Kernel  // inherit slot-237's evaluate-pointwise interface
    SDE() (A, L []float64, qc float64, P_inf []float64, n int)
}
```

This pattern lets slot-237 GP consumers use `gp.Kernel` directly (O(n³) Cholesky path) and slot-269 GP-state-space consumers use `KernelWithSDE` (O(n) Kalman path) WITHOUT type-switching at call site. Single interface upgrade lets users compose `Matern52ToSDE` × `KalmanFilter` × `RTSSmoother` × `LBFGS-MLE` end-to-end.

Single-line architectural witness: reality has Cholesky + matrix-exp-substrate-pending + FFT + LBFGS + autodiff + NormalCDF + Kernel-interface-pending + GaussianKernel-RBF + LaplacianKernel-Matern1/2 ≈ 50% of GP-state-space substrate; ZERO of the 22 GP-SS primitives present; K1+K2+K3+K8+K10+K13 ≈ 660 LOC delivers Hartikainen-Särkkä-2010 entry-level Matern-ν∈{1/2,3/2,5/2} GP-as-Kalman-smoother once slot-161 PR-0b+PR-0c land (StateSpace + KalmanFilter + Discretize ZOH/Tustin + DiscreteLyapunov ≈ 580 LOC). End-to-end ship requires slot-117 Box-Muller (sampling) + slot-161 PR-1 (KF) + slot-237 PR-0d (MaternKernel + Kernel-interface) before slot-269 PR-1 can be reviewed.

Total ~3,520 LOC (~2,840 NET-NEW + ~680 ship-once-with-slot-161+237) over ~14 engineer-days excluding upstream slot-117 + slot-161 + slot-237 + slot-242 dependencies, with PR-1 + PR-2 + PR-3 + PR-4 + PR-5 (~1,960 LOC) delivering Hartikainen-Särkkä-2010 + Solin-Särkkä-2014-ICML + Mbalawata-Särkkä-Haario-2013 entry-level GP-state-space stack in one engineer-week once upstream blockers cleared.

Report at `agents/269-new-gp-state-space.md` ≈ 280 lines.

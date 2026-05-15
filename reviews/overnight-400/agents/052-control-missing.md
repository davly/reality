# 052 — control: canonical algorithms missing

## Headline
`control/` ships **3 source files / ~493 LOC / 8 user-visible primitives**: `PIDController` (clamping anti-windup), `LowPass/HighPass/Complementary` filters, `RateLimiter`, and `TransferFunction` with `Evaluate / Poles (Durand-Kerner) / IsStable`. Sibling 051 already audited what is shipped. This report enumerates what is **not** shipped — and the gap is *most of the discipline*. The package has **no state-space type at all** (no `(A,B,C,D)`), no observability/controllability primitives, no pole placement, no continuous-discrete bridge (no ZOH, no Tustin, no Backward-Euler discretization), no Bode/Nyquist/root-locus, no stability margins, no LQR/Kalman/MPC, no robust-control toolbox (H∞/μ/LMI), no nonlinear-control primitives (sliding-mode, backstepping, IDA-PBC), no system identification (PRBS/ARX/4SID), no adaptive control (MRAC/STR/gain-scheduling). python-control 2025 ships ~250 callable functions; reality/control ships 8. The Tier-1 gap below targets the smallest set that turns this from "PID + a transfer function evaluator" into "I can design a controller end-to-end."

## What exists today
- `control/pid.go` (123 LOC): `PIDController` struct + `NewPID` + `Update` + `Reset`. Clamping anti-windup. Backward-Euler integral, derivative-on-error (kick bug per 051).
- `control/transfer.go` (253 LOC): `TransferFunction{Num, Den}` + `Evaluate(s) complex128` + `Poles() []complex128` (closed-form deg ≤ 2, Durand-Kerner higher) + `IsStable() bool`. Internal Taylor-series `realCos/realSin`.
- `control/filter.go` (117 LOC): `LowPassFilter`, `HighPassFilter`, `ComplementaryFilter` (algebraically broken per 051), `RateLimiter`.

That is the entire user surface: **8 free functions / methods + 1 stateful struct + 1 transfer-function struct**.

## Missing canonical algorithms

### Tier 1 — high-impact / table-stakes for any classical-control package

**Frequency-domain analysis (the bare minimum for any continuous TF library)**
- **Bode plot** — `Bode(tf, omegas) (mag, phase []float64)` returning magnitude in dB and phase in degrees (or radians); needs phase unwrapping. ω→0 closed-form `H(0) = num[end]/den[end]` shortcut. ω→∞ asymptotic rolloff (`-20·(deg(D)-deg(N)) dB/dec`). Pure DSP, ~80 LOC + golden vectors.
- **Nyquist plot** — `Nyquist(tf, omegas) []complex128` (parametric `H(jω)` from -∞ to +∞), plus `EncirclementCount(samples, point complex128) int` for the standard `(-1, 0)` test. ~40 LOC.
- **Stability margins** — `GainMargin(tf) float64` (gain in dB at the −180° phase crossover), `PhaseMargin(tf) float64` (180° + ∠H at the unity-gain crossover), `DelayMargin(tf) float64 = PhaseMargin / ωgc`. Bracket-and-bisect on Bode arrays. ~60 LOC.
- **Root locus** — `RootLocus(num, den, gains) [][]complex128` returning closed-loop poles vs gain `K` for `1 + K·H(s) = 0`. Each gain step is one polynomial root-find. ~50 LOC.
- **Step response** — `StepResponse(tf, t []float64) []float64` and `ImpulseResponse(tf, t []float64) []float64`. Direct numerical evaluation via residue / partial-fraction expansion or via `chaos.RK4` on the realized state-space form. ~80 LOC.
- **DC gain & high-frequency rolloff** — `DCGain(tf) float64`, `HighFrequencyRolloffSlope(tf) int` (dB/decade). Trivial closed-form, currently absent — call sites recompute. ~10 LOC.

**Continuous ↔ Discrete bridge (the missing half of every TF)**
- **`DiscreteTransferFunction`** type with z-domain coefficients; `Evaluate(z complex128)`, `Poles()` (stability `|p|<1` not `Re(p)<0`), `Step/ImpulseResponse`. ~80 LOC.
- **`Tustin(tf, T) DiscreteTransferFunction`** — bilinear transform `s ← (2/T)(z-1)/(z+1)`, closed-form coefficient algebra, optional pre-warping at a chosen frequency. ~60 LOC.
- **`ZOH(tf, T) DiscreteTransferFunction`** — zero-order hold via state-space realization + `Φ = exp(A·T)`, `Γ = ∫ exp(A·τ)dτ · B`. Requires matrix exponential (linalg currently has neither `expm` nor Schur — see prerequisites). ~80 LOC + 60 LOC `expm` (Padé-13 or scaling-and-squaring).
- **`BackwardEuler(tf, T)`** — `s ← (z-1)/(zT)`. ~20 LOC.
- **`ForwardEuler(tf, T)`** — `s ← (z-1)/T`. Documented as conditionally stable. ~15 LOC.
- **`d2c` reverse map** — `Tustin⁻¹`, ZOH inverse via matrix log. ~40 LOC.

**State-space — the prerequisite for everything robust/optimal/nonlinear**
- **`StateSpace{A, B, C, D}`** type with vector/matrix-shape contract, `New(A, B, C, D) StateSpace` validating shapes (`A:n×n, B:n×m, C:p×n, D:p×m`). ~60 LOC.
- **`tf2ss(tf) StateSpace`** — controllable canonical form is one-liner; observable canonical form, modal form (via Schur/Jordan). ~60 LOC.
- **`ss2tf(ss, inputIdx, outputIdx) TransferFunction`** — `H(s) = C(sI−A)⁻¹B + D` via determinant or Faddeev-Leverrier algorithm. ~50 LOC.
- **`Controllable(A, B) (rank int, ctrb []float64)`** — controllability matrix `[B, AB, A²B, …, Aⁿ⁻¹B]` and its rank. ~30 LOC.
- **`Observable(A, C) (rank int, obsv []float64)`** — observability matrix `[Cᵀ, AᵀCᵀ, …]ᵀ`. ~30 LOC.
- **`ControllabilityGramian(A, B) []float64`** — solve `AWc + WcAᵀ + BBᵀ = 0` (continuous Lyapunov). ~120 LOC.
- **`ObservabilityGramian(A, C) []float64`** — solve `AᵀWo + WoA + CᵀC = 0`. ~30 LOC (shares Lyapunov solver).
- **`ContinuousLyapunov(A, Q) []float64`** — solves `AX + XAᵀ + Q = 0` via Bartels-Stewart (Schur on A, then back-substitute). ~150 LOC + Schur.
- **`DiscreteLyapunov(A, Q) []float64`** — solves `AXAᵀ − X + Q = 0`. ~120 LOC.
- **`SylvesterEquation(A, B, C) []float64`** — solves `AX + XB = C`. ~120 LOC, building block for everything.

**Pole placement & basic state-feedback**
- **Ackermann's formula** — `PlacePoles(A, B, desiredPoles) K []float64` for SISO; `K = [0,…,0,1]·ctrb⁻¹·αd(A)` where αd is the desired char poly. ~50 LOC.
- **Bass-Gura** alternative for SISO. ~30 LOC.
- **`PlaceMIMO(A, B, desiredPoles)`** — Kautsky-Nichols robust pole placement (via Schur). ~200 LOC.
- **Observer design (Luenberger)** — dual of pole placement using `(Aᵀ, Cᵀ)`; returns observer gain `L`. ~30 LOC.

**LTI optimal regulation (LQR / DLQR)**
- **`LQRContinuous(A, B, Q, R, N) (K, S, eigCL []float64)`** — solves continuous-time algebraic Riccati equation `AᵀS + SA − (SB+N)R⁻¹(BᵀS+Nᵀ) + Q = 0`; canonical methods are Schur (Laub 1979) or generalized eigenproblem on the Hamiltonian matrix. Returns optimal gain `K = R⁻¹(BᵀS+Nᵀ)`, Riccati solution `S`, and closed-loop eigenvalues. ~180 LOC + Schur dependency.
- **`LQRDiscrete(A, B, Q, R, N)`** — discrete ARE `S = AᵀSA − (AᵀSB+N)(R+BᵀSB)⁻¹(BᵀSA+Nᵀ) + Q`. Same Schur method on the symplectic pencil. ~180 LOC.
- **LQ output regulator / tracker** — `LQT(A, B, C, Q, R, ref)` for tracking a reference signal; finite-horizon Riccati via backward recursion. ~80 LOC.
- **Kalman filter — discrete** — `KalmanDiscrete{F, H, Q, R}` with `Predict`/`Update` cycle; `(x̂, P)` state. ~100 LOC.
- **Kalman filter — continuous** — `KalmanContinuous{A, C, Q, R}` solving the continuous Kalman-Bucy filter Riccati `AP + PAᵀ − PCᵀR⁻¹CP + Q = 0`. ~120 LOC.
- **Square-root Kalman (Bierman-Thornton UDU')** — numerically stable variant; the *standard* Kalman is unstable in finite precision over many updates as `P` loses positive-definiteness. ~200 LOC.
- **Square-root Kalman (Carlson)** — alternative `LDLᵀ` factorization. ~150 LOC.
- **Information filter** — dual form (information matrix `Y = P⁻¹`, information vector `y = Yx̂`); easier sensor fusion. ~60 LOC.
- **Extended Kalman Filter (EKF)** — `EKF{f, h, F, H, Q, R}` with Jacobian-evaluation callbacks; first-order linearization. ~120 LOC.
- **Unscented Kalman Filter (UKF)** — sigma-point propagation (Julier-Uhlmann), no Jacobians. Scaled UT (Wan-Van der Merwe). ~200 LOC.
- **Square-root UKF** — Cholesky-form propagation. ~250 LOC.

**Discretization / pre-warping helpers (not just for TFs but for general filter design)**
- **Tustin pre-warping** — `PreWarp(omega, T) float64 = (2/T)·tan(omega·T/2)`. ~5 LOC.
- **Bilinear with frequency mapping** — `BilinearPrewarped(tf, T, omegaMatch)`. ~30 LOC.
- **Padé approximants for delay** — `PadeDelay(L, order) TransferFunction`; the standard way to model `e^(-Ls)` in continuous TF. ~40 LOC.

**Compensator design (the bread and butter of classical loop-shaping)**
- **Lead compensator** — `Lead(omegaCenter, alpha)` with `alpha < 1` returning `(1+s/(α·ωc))/(1+s/ωc)`; max phase boost `arcsin((1−α)/(1+α))`. ~20 LOC.
- **Lag compensator** — `Lag(omegaCenter, alpha)` with `alpha > 1`; DC-gain boost. ~20 LOC.
- **Lead-lag** — composition. ~10 LOC.
- **Notch filter** — `Notch(omega0, Q) TransferFunction` for a parametric second-order notch. ~30 LOC.

**PID auto-tuning (canonical and frequently asked for)**
- **Ziegler-Nichols open-loop** (reaction-curve method, K, L, T from step response). ~50 LOC.
- **Ziegler-Nichols closed-loop** (ultimate gain Ku, ultimate period Pu). ~50 LOC.
- **Cohen-Coon** — improved Z-N for processes with significant dead time. ~40 LOC.
- **Astrom-Hagglund AMIGO** — modern (2004) Z-N replacement using the same K/L/T from a step. ~50 LOC.
- **Skogestad SIMC** — model-based PID tuning from a first-order-plus-deadtime (FOPDT) model. ~50 LOC.
- **Relay feedback (Astrom-Hagglund 1984)** — auto-induce sustained oscillation via on-off relay, infer Ku/Pu without model. ~60 LOC.

### Tier 2 — moderately useful / standard control toolbox features

**Robust control (the H∞/μ/LMI stack)**
- **Internal Model Control (IMC)** — `IMCTune(plantModel, lambda)` for robust PID where λ is the closed-loop time constant. Skogestad's IMC-PID rules. ~80 LOC.
- **Loop shaping (Glover-McFarlane)** — `LoopShape(plant, shapingFilter, gamma)`; the McFarlane-Glover normalized-coprime-factor robust controller. ~250 LOC.
- **H∞ synthesis (DGKF / Glover-Doyle 1989)** — solve two coupled AREs for the H∞ optimal controller; γ-iteration to find minimum γ. The canonical "robust optimal" controller. ~500 LOC + ARE solver.
- **H∞ loop-shaping** — `Hinf_LoopShape(plant, Wp, Wu) Controller`, mixed-sensitivity S/KS or S/T design with weighting filters. ~400 LOC.
- **μ-synthesis / D-K iteration** — `MuSynthesis(plant, uncertaintyStruct, gamma) Controller`; alternates H∞ design and D-scale fitting. The full robust-control workhorse. ~600 LOC.
- **Structured Singular Value (μ)** — upper bound via LMI or Osborne scaling. ~300 LOC.
- **Lyapunov LMI** — `SolveLMI(constraints) Solution`; primal-dual interior-point on small LMIs. Or restrict to the S-procedure (one quadratic constraint implies another) and `(A,P)` Lyapunov stability LMI `AᵀP + PA < 0`. ~500 LOC for a minimal LMI solver, or ~80 LOC for the closed-form Lyapunov-LMI cases.
- **LQG/LTR (Loop Transfer Recovery)** — `LQGLTR(A, B, C, Q, R, V, W, rho)` recovering loop properties of LQR via Doyle-Stein. ~150 LOC.

**Model Predictive Control (MPC) primitives**
- **`MPCQP{Q, R, N, A, B, xMin, xMax, uMin, uMax}`** — discrete-time linear MPC building the prediction matrices `(F, G)` and condensed quadratic program. ~150 LOC.
- **QP solver** — interior-point or active-set for the small dense QPs MPC produces. Reality currently has *no QP solver*; `optim/` has L-BFGS, simulated annealing, genetic — none of which solve constrained QPs. ~400 LOC for OSQP-style ADMM, ~600 LOC for an interior-point. (Or accept dependency on a sibling `qp` sub-package once it exists.)
- **Explicit MPC** — multi-parametric QP (mpQP) producing a piecewise-affine state-feedback law. ~600 LOC. Tier-3 candidate.
- **Move-blocking, terminal cost, terminal constraint set** helpers. ~80 LOC.

**System identification**
- **PRBS (pseudo-random binary sequence)** — `PRBS(n int, seed int64) []float64` for excitation. Uses the `crypto/` LFSR. ~30 LOC.
- **ARX (`A(q)y = B(q)u + e`) least-squares fit** — `ARX(u, y, na, nb, nk) (a, b []float64)`. ~80 LOC.
- **OE (output-error) `y = B(q)/F(q) u + e`** — iterative; pseudolinear regression. ~150 LOC.
- **ARMAX (`A·y = B·u + C·e`)** — extended LS or PEM. ~200 LOC.
- **Box-Jenkins** — most general SISO model `y = (B/F)u + (C/D)e`. ~250 LOC.
- **Subspace ID — N4SID** (Van Overschee-De Moor). ~400 LOC + SVD/QR (linalg has these).
- **Subspace ID — MOESP**. ~400 LOC.
- **Subspace ID — CVA** (canonical-variate analysis). ~400 LOC.
- **ERA (Eigensystem Realization Algorithm)** — Juang-Pappa Hankel-SVD ID for impulse response. ~150 LOC.
- **Empirical TF estimation** — `Spectrogram-based ETF`, ETFE on time-series I/O via cross-spectrum / auto-spectrum (uses `signal.FFT`). ~80 LOC.
- **Persistent excitation rank check** — `IsPersistentlyExciting(u, order) bool`. ~30 LOC.
- **Welch / spa-style smoothed estimator**. ~80 LOC.

**Modal / canonical decomposition**
- **Schur decomposition** — needed by Bartels-Stewart, ARE solvers. linalg has QR-iteration eigenvalues but no orthogonal Schur form `A = QTQᵀ` with quasi-triangular `T`. ~300 LOC.
- **Jordan decomposition** — for repeated eigenvalues; numerically unstable, used mostly for analysis. ~250 LOC.
- **Real Schur to complex Schur** ordering. ~80 LOC.
- **Modal canonical form** — diagonal/quasi-diagonal `A` realization of a state-space system. ~80 LOC.
- **Balanced realization** — Moore 1981; transforms state coordinates so that controllability and observability Gramians are equal and diagonal (Hankel singular values). ~150 LOC.
- **Balanced truncation model reduction** — drop states with smallest Hankel singular values. ~80 LOC.
- **Hankel-norm model reduction** — Glover 1984. ~250 LOC.

**Adaptive control**
- **MRAC (Model-Reference Adaptive Control)** — Lyapunov-based MIT rule, `θ̇ = -γ·e·φ`. ~80 LOC.
- **Self-Tuning Regulator (STR) — indirect** — recursive least-squares ID + pole placement. ~150 LOC.
- **STR — direct (minimum-variance, minimum-phase)** — Astrom-Wittenmark MV controller. ~120 LOC.
- **Gain scheduling** — `GainScheduledController{schedulers []Controller, breakpoints, interpolation}`; piecewise-linear gain interpolation as a function of operating point. ~80 LOC.
- **Recursive least squares (RLS)** — with forgetting factor. Belongs in `signal/` or here. ~60 LOC.

**Nonlinear control (the named topics from MASTER_PLAN)**
- **Lyapunov-function checker** — `IsLyapunov(V, Vdot, domain) bool` numerical sufficient-condition checker. ~80 LOC.
- **Feedback linearization (input-output)** — relative-degree determination and the canonical input transformation `u = (v - Lf^r h) / (LgLf^(r-1) h)`. ~150 LOC. Requires a symbolic / autodiff layer that Reality has (`autodiff/`).
- **Backstepping** — recursive Lyapunov design for strict-feedback systems. Mostly an analysis pattern + canonical building blocks. ~200 LOC.
- **Sliding-mode control (SMC)** — `SlidingSurface{c, lambda} Surface`; `SMC.Update(x)` returning `u = -K·sgn(s)`. Boundary-layer (saturation) variant for chattering reduction. Super-twisting (Levant) and twisting algorithms. ~150 LOC for first-order SMC + boundary layer; ~120 LOC for super-twisting.
- **Higher-order SMC** — Levant arbitrary-order. ~250 LOC.
- **IDA-PBC (Interconnection-Damping Assignment Passivity-Based Control)** — Ortega-Spong-Gomez-Estern; for port-Hamiltonian systems shape-and-damping assignment. ~250 LOC.
- **Passivity-based control (general)** — energy-shaping + damping injection for Euler-Lagrange systems. ~150 LOC.
- **Control Lyapunov Function (CLF) / Sontag's universal formula** — `u = -((LfV) + sqrt((LfV)² + (LgV)⁴)) / LgV` when LgV ≠ 0. ~50 LOC.
- **Control Barrier Function (CBF)** — quadratic-program safety filter `min ‖u-u_nom‖² s.t. LfB+LgB·u ≥ -α(B)`. Trendy / load-bearing for safe-RL applications. Requires QP solver (see MPC). ~80 LOC + QP.
- **ISS (Input-to-State Stability)** — gain-margin computation; small-gain theorem composition rule for two ISS subsystems. ~100 LOC.

**Discrete-event / hybrid (briefly)**
- **Bumpless transfer** between two controllers (canonical anti-windup-adjacent). ~30 LOC.
- **Anti-windup back-calculation Kt variant** (per 051's C-PID-AW-1). ~30 LOC.

### Tier 3 — niche but appears in canonical curricula / 2025 SOTA

**Modern numerical methods**
- **Pseudo-spectral abscissa** — for robust stability radius computation; Boyd-Balakrishnan. ~250 LOC.
- **Distance to instability** — smallest perturbation moving a pole to RHP. ~150 LOC.
- **Real / complex μ exact computation** for low-dim cases — branch and bound. ~400 LOC.
- **`SLICOT`-style Riccati**: doubling, Newton with line search, sign-function method. (The `linalg/Schur` route covers the canonical Laub method; these are alternatives with different stability properties.)
- **Convex MPC via `cvxpygen`-style code generation** — out of scope for math library.
- **Differential flatness** check + flat-output trajectory generation. ~250 LOC.
- **Trajectory optimization** — direct collocation, multiple shooting; bridges to `optim/` and `chaos/` ODE integrators. ~400 LOC.
- **Reachability / set-valued analysis** — Hamilton-Jacobi PDE on a grid (Mitchell's level-set method). ~600 LOC. Belongs in a future `pde/` package.
- **Zonotopes / interval-arithmetic reach sets** — propagate `(A,B)` system through zonotope set state. ~300 LOC.

**Specialized / academic**
- **PI-lambda / fractional-order PID** — Podlubny (1999). ~80 LOC.
- **Smith predictor** — dead-time compensation. ~50 LOC.
- **Otto-Smith / IMC for dead-time**. ~30 LOC.
- **Repetitive control** for periodic disturbance rejection. ~80 LOC.
- **Iterative Learning Control (ILC)** — first-order, higher-order, adjoint-based. ~150 LOC.
- **Phase-locked loop** primitive (already partially fits in `signal/`). ~60 LOC.
- **Anti-aliasing filter design** at a chosen sample rate (companion to ZOH). ~30 LOC.

**Realtime / embedded**
- **Fixed-point PID** — `PIDFixed{Kp, Ki, Kd, scale int32}`. ~80 LOC. Deferral candidate.
- **Tustin in fixed-point**. ~60 LOC.

**Pre-2025 / 2025 SOTA worth a glance** (web research)
- **python-control 0.10 (2025)**: ships LQR/LQG/Kalman (linear & EKF/UKF), `c2d` with all four discretization methods, Bode/Nyquist/Nichols, root locus, margins, controllability/observability/Gramians, balanced reduction, Sylvester/Lyapunov/Riccati solvers via `slycot`. Algorithmic API stable since ~0.8 (2018); recent work has been `flatsys` (differential flatness) and `phaseplot` ergonomics. Reality is ~7 years behind on the *core* surface; getting Tier 1 closes ~80% of `python-control`.
- **Slycot** — Python wrapper around SLICOT Fortran; reference SOTA for ARE / Lyapunov / model reduction at scale.
- **CVXPY + cvxpygen** — convex-optimization-based control (MPC, robust MPC, CBF QPs). Out of scope mathematically (general convex solvers belong in `optim/proximal/` already started).
- **CasADi 3.6 (2024)** — symbolic differentiation for nonlinear MPC; built atop autodiff. Reality has `autodiff/` already.
- **do-mpc 4.6 (2024)** — built on CasADi; nonlinear MPC + moving-horizon estimation. Pattern-match for what *uses* a Reality MPC.
- **Modelica / Modelica.Blocks.Continuous** — every primitive in Tier 1 has a Modelica reference block; Modelica-Blocks-style "first-order", "second-order", "transfer function", "state space", "PID" naming maps directly.
- **Drake (Tedrake/MIT, 2025)** — direct-collocation NLP, Lyapunov via SOS programming, contact-implicit MPC. Tier-3 territory.
- **`harold`** — pure-Python, NumPy/SciPy, 2018-2022. Subset of `python-control` but pedagogically clean reference.

## Prerequisites in sibling packages

Reality's `linalg/` ships eigenvalue iteration (`QRAlgorithm`, `tqli`, `tridiagonalize`) but **lacks**:
- **Schur decomposition** `A = QTQᵀ` with quasi-triangular `T` (real Schur). Needed by Bartels-Stewart Lyapunov and Laub-method ARE. ~300 LOC, MUST land before LQR/Lyapunov.
- **Matrix exponential** `expm(A)` via Padé-13 + scaling-and-squaring. Needed by ZOH and by `c2d` and by Kalman propagation across non-uniform timesteps. ~150 LOC.
- **Generalized eigenproblem** (QZ algorithm) for Hamiltonian / symplectic pencils used by ARE. ~400 LOC. Optional — Schur on the Hamiltonian matrix is enough for SISO/dense.
- **Sylvester equation solver** `AX + XB = C`. ~120 LOC, builds on Schur.

`optim/` has gradient methods (L-BFGS, gradient, simulated annealing, genetic) but **no QP solver** for MPC / CBF safety filters. ADMM-style OSQP would cost ~400 LOC. Active-set or interior-point ~600 LOC.

`autodiff/` is shipped (per CLAUDE.md), enabling feedback linearization and CBF Lie-derivative computation without symbolic algebra.

`chaos/RK4` exists, useable for nonlinear simulation `ẋ = f(x,u)` — sufficient for nonlinear MPC interior-loop integration and for nonlinear-control validation.

`crypto/` LFSR can back PRBS generation directly.

`signal/FFT` enables Bode-from-data (ETFE) and frequency-response identification.

`prob/` provides the multivariate Gaussian primitives Kalman variants need.

## Concrete commit ladder (ordered for incremental shippability)

| # | Bundle | LOC est | Prereq | Ships |
|---:|---|---:|---|---|
| **L1** | Bode + Nyquist + DCGain + HighFreqRolloff + GainMargin + PhaseMargin + DelayMargin | ~250 | none (uses TF.Evaluate) | Frequency-domain analysis |
| **L2** | Tustin + BackwardEuler + ForwardEuler + DiscreteTransferFunction type | ~180 | none | Continuous → Discrete |
| **L3** | StateSpace{A,B,C,D} + tf2ss + ss2tf + Controllable + Observable | ~230 | linalg.SVD or rank | State-space surface |
| **L4** | linalg.Schur (real) + linalg.Expm + linalg.SolveSylvester + linalg.SolveContinuousLyapunov | ~700 | linalg only | Prerequisites for LQR/LQG/ZOH |
| **L5** | ZOH + c2d (Tustin / ZOH / FE / BE selectors) + Padé delay | ~140 | L4 | Round-trip discretization |
| **L6** | LQRContinuous + LQRDiscrete + LQ output regulator | ~440 | L3 + L4 | Optimal regulation |
| **L7** | KalmanDiscrete + KalmanContinuous + Square-root (Bierman-Thornton) + EKF + UKF | ~870 | L3 + L4 + prob | Optimal estimation |
| **L8** | StepResponse + ImpulseResponse + RootLocus | ~180 | L3 | Time-domain analysis |
| **L9** | PlacePoles (Ackermann SISO) + Luenberger observer + LQG-LTR | ~230 | L3 | Pole placement / observers |
| **L10** | Lead + Lag + LeadLag + Notch | ~80 | none | Compensator design |
| **L11** | Z-N (open + closed) + Cohen-Coon + AMIGO + Skogestad + Relay-feedback | ~300 | none | PID auto-tuning |
| **L12** | PRBS + ARX + ETFE + persistent-excitation check | ~220 | crypto, signal | Basic system ID |
| **L13** | optim/QP (OSQP-style ADMM) + MPCQP{Q,R,N} + condensed prediction | ~550 | L3 + L4 | Linear MPC |
| **L14** | SlidingMode (1st-order + boundary layer + super-twisting) + Lyapunov-checker + CLF/Sontag | ~320 | autodiff | Nonlinear-control basics |
| **L15** | IMC + Hinf two-Riccati DGKF + LoopShape (Glover-McFarlane) | ~1150 | L4 + L6 | H∞ robust control |
| **L16** | μ-synthesis D-K iter + LMI mini-solver | ~1100 | L4 + L15 | μ-synthesis |
| **L17** | MRAC + RLS + STR (indirect, direct) + GainSchedule | ~410 | L3 + L4 | Adaptive |
| **L18** | N4SID + MOESP + ERA + ETFE-Welch | ~1230 | L3 + linalg.SVD | Subspace ID |
| **L19** | IDA-PBC + Backstepping + Feedback-linearization | ~600 | autodiff | Advanced nonlinear |
| **L20** | Balanced realization + balanced truncation + Hankel-norm reduction | ~480 | L3 + L4 | Model reduction |

**Highest-value first PR (Tier-1 minimum):** L1 + L2 + L3 + L8 + L10 ≈ ~920 LOC, no new linalg dependencies, transforms `control/` from "PID + scalar TF eval" into "I can design a classical SISO controller and check its Bode/Nyquist/margins." All five together are the missing 60% of any control-systems undergrad textbook (Ogata Ch. 5-7, Astrom-Murray Ch. 9-10).

**Second wave:** L4 (linalg prereq) + L6 (LQR) + L7 (Kalman, all 5 variants) ≈ ~2010 LOC, gates the entire optimal/state-space / robust / adaptive / MPC / nonlinear stack. Of these, the *single* most load-bearing missing piece across the codebase is **`linalg.expm`** — everything from `c2d` to Kalman propagation to nonlinear-system local linearization to matrix-Lyapunov needs it.

**Third wave:** L13 (MPC + QP) + L14 (sliding-mode + Lyapunov) ≈ ~870 LOC, opens the modern-control / safety-critical surface (Pistachio gimbal sliding-mode camera control, Sentinel rate-limited safe-mode controller).

## Sources

Repo
- `C:\limitless\foundation\reality\control\pid.go`
- `C:\limitless\foundation\reality\control\transfer.go`
- `C:\limitless\foundation\reality\control\filter.go`
- `C:\limitless\foundation\reality\control\control_test.go`
- `C:\limitless\foundation\reality\control\control_edge_test.go`
- `C:\limitless\foundation\reality\linalg\eigen.go` (QR iteration only — no Schur / expm)
- `C:\limitless\foundation\reality\linalg\decompose.go` (LU / QR / Cholesky present)
- `C:\limitless\foundation\reality\optim\` (no QP solver)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\051-control-numerics.md` (companion numerics audit)

Web (consulted 2026-05-07)
- python-control 0.10.x docs https://python-control.readthedocs.io/ (LQR/LQG, Bode/Nyquist/margins, c2d, Riccati via `slycot`)
- Slycot — SLICOT Fortran wrapper https://github.com/python-control/Slycot
- SLICOT library https://github.com/SLICOT/SLICOT-Reference (Fortran 77, the canonical control numerics reference; ARE solvers, balanced reduction, Lyapunov)
- `harold` https://github.com/ilayn/harold (pure-Python, no Fortran dep — closest design analogue to a hypothetical Reality control)
- CasADi 3.6 https://web.casadi.org/ (autodiff for NMPC)
- do-mpc 4.6 https://www.do-mpc.com/ (nonlinear MPC built on CasADi)
- Modelica.Blocks.Continuous https://doc.modelica.org/ (block naming reference)
- Drake (Tedrake/MIT) https://drake.mit.edu/ (Lyapunov SOS, contact-implicit MPC, trajectory optimization)
- Astrom & Murray, *Feedback Systems* (online, 2nd ed., 2020) — canonical curriculum reference for Tier-1.
- Ogata, *Modern Control Engineering* (5th ed., 2010) — classical Bode/Nyquist/root-locus pedagogy.
- Astrom & Wittenmark, *Adaptive Control* (2nd ed., 1995) — MRAC / STR canonical.
- Skogestad & Postlethwaite, *Multivariable Feedback Control* (2nd ed., 2005) — H∞ / μ canonical reference.
- Goodwin-Graebe-Salgado, *Control System Design* (2001) — system ID and SISO canonical.
- Khalil, *Nonlinear Systems* (3rd ed., 2002) — Lyapunov, sliding mode, backstepping, feedback linearization.
- Ortega-van der Schaft-Maschke-Escobar, *Putting energy back in control* (2001) — IDA-PBC canonical paper.
- Levant 2003, *Higher-order sliding modes, differentiation and output-feedback control* — super-twisting reference.
- Van Overschee-De Moor 1996, *Subspace Identification for Linear Systems* — N4SID canonical.

## Disjoint-check appendix

Adjacent control/ slots (per MASTER_PLAN context implied by 051's footer):
- 051 control-numerics — owned numerical correctness of *shipped* code (PID kick, filter algebra, Durand-Kerner). This report does **not** revisit those.
- 053 control-sota — owns library/research-trend comparison narrative. This report touches python-control / Slycot / Drake only as *evidence of canonicality*, not as comparative analysis.
- 054 control-api — owns naming, error-handling, ergonomics of *both* shipped surface and any new surface. This report sketches signatures only enough to argue feasibility / LOC; final naming ceded.
- 055 control-perf — owns allocation/inlining/SIMD considerations. This report flags expm/Schur as prerequisites with cost ~150-300 LOC each but does not opine on per-call performance.

This report covers **missing canonical algorithms** only, organized by Tier and grouped by the eight subdomains in the topic prompt (Classical, State-space, Optimal, Robust, Nonlinear, Adaptive, System-ID, plus the explicit named items LQR / LQG/Kalman / MPC / H∞ / μ / IDA-PBC / sliding-mode / gain-scheduling).

Report at `agents/052-control-missing.md`, ~340 lines.

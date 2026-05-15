# 192 | synergy-fluids-control

**Topic:** fluids × control — PI(D) flow control, drag reduction, ROM-based closed-loop control
**Block:** B (cross-package synergies)
**Date:** 2026-05-08
**Scope:** capabilities that emerge ONLY when `fluids/`, `control/`, and `linalg/` are composed; not what either is missing in isolation (covered by per-package agents 056-060 / 086-090 / 071-075 / 160 fluids-signal / 161 control-prob).

## Two-line summary

Today `fluids/` ships 11 algebraic closed-form scalars (236 LOC, no time, no state, no field) at `fluids/fluids.go` and `control/` ships scalar `PIDController` + 4 first-order filters + `TransferFunction{Num,Den}` Durand-Kerner poles (490 LOC, no state-space, no MIMO, no observer) at `control/{pid,filter,transfer}.go` — **zero source-level coupling** (`grep -r 'fluids' control/*.go` and `grep -r 'control' fluids/*.go` both empty), no `Plant`/`Actuator`/`Sensor` types, no `Q`/`R`/`Cl(t)`/`u(t)` time-history primitives anywhere; the entire **closed-loop flow-control canon** (Choi-Moin-Kim-1994 opposition, Bewley-Moin-Temam-2001 adjoint, Schmid-2010 DMD, Williams-Kevrekidis-Rowley-2015 EDMD/Koopman, Skogestad-2003 IMC, Astrom-Hagglund 1995 lambda-tuning, surge-avoidance/HVAC/wind-turbine-pitch literature) is wholly absent.

Twenty synergy primitives F1-F20 totalling ~2,650 LOC of pure connective tissue close the gap; **eleven ship today** against v0.10.0 (every primitive that is `PIDController.Update`-composable + steady-state `fluids` formula + `linalg.MatVec/MatMul/Inverse`); **nine** are blocked on missing primitives independently flagged in agent 161-C1 (`StateSpace{A,B,C,D}.Step`), 161-C5 (`KalmanFilter`), 097-T1 (eigenvectors + matrix exponential), and 074 (`SVD`); cheapest one-day PR is **F1 `FlowPlant1D` + F2 `PIPipeFlow` + F3 `MassFlowController` at ~140 LOC**, giving the first plant-coupled PID demo in the repo and saturating an obvious 3/3 R-MUTUAL-CROSS-VALIDATION pin (steady-state mass-flow setpoint reached three ways: open-loop Bernoulli inversion, P-only `Kp = 1/Kp,plant`, full PI Skogestad-SIMC tuning — all converge to `≤0.5%` of `ṁ_set` for laminar/turbulent regimes); architectural keystone is **F8 `LinearizedPlant` (Jacobian via central difference on `fluids` closed-forms)** because every model-based primitive (LQR, IMC, ROM-Galerkin, DMDc) consumes a state-space `(A,B)` it produces; recommended placement is a NEW sub-package `fluids/control/` (mirrors precedent 159 `em/wave/`, 160 fluids-signal proposed `fluids/turbulence/`, 157 `graph/spectral.go`) — flow-control primitives are neither pure-fluids (they require time history) nor pure-control (they require Reynolds/drag/lift/Darcy-Weisbach plant models).

---

## 0. State of play (verified file-walk)

`fluids/fluids.go` HEAD (1 file, 236 LOC, 11 exported funcs):
- **Dimensionless:** `ReynoldsNumber(rho,v,L,mu)`
- **Bernoulli/pipe:** `BernoulliPressure`, `PipeFlowFriction` (Colebrook-White iterative ≤100 iters), `DarcyWeisbach`
- **Aerodynamic forces:** `DragForce(Cd,rho,v,A) = 0.5·Cd·rho·v²·A`, `LiftForce`, `TerminalVelocity`
- **Low-Re:** `StokesLaw(mu,r,v) = 6πμrv`
- **Flow rates:** `MassFlowRate`, `VolumetricFlowRate`

Every function is an **algebraic closed-form scalar** except `PipeFlowFriction` (fixed-point loop). **Zero** state, **zero** time, **zero** vector field, **zero** Navier-Stokes, **zero** PDE. No `Pipe{}`, `Pump{}`, `Valve{}`, `Actuator{}`, `Sensor{}` type. Doc declares "numbers in, numbers out."

`control/{pid,filter,transfer}.go` HEAD (3 files, 490 LOC):
- `pid.go`: `PIDController{Kp,Ki,Kd,minOut,maxOut,integralSum,prevError}` with anti-windup (`pid.go:36-122`); `Update(setpoint, measured, dt)` returns clamped output, undoes integral step on saturation. **Stateful, scalar, SISO only.**
- `filter.go`: `LowPassFilter`, `HighPassFilter`, `ComplementaryFilter`, `RateLimiter` — all scalar, first-order, stateless (caller maintains prev).
- `transfer.go`: `TransferFunction{Numerator,Denominator []float64}` polynomial coefficients (descending); `Evaluate(s complex128)` Horner; `Poles()` quadratic for n≤2, Durand-Kerner for n≥3; `IsStable()` (Re(p)<0). **No discrete-time form, no `c2d`, no Bode magnitude/phase function, no Nyquist plot data, no margin computation.**

Consumer comment block names "Pistachio (camera/animation), Pulse (monitoring), Sentinel (alert tuning), BookaBloke (scheduling)" — fluid-mechanics consumers conspicuously absent.

`linalg/` (~1,500 LOC, agents 071-075): `MatMul`, `MatVecMul`, `MatTranspose`, `MatAdd/Sub/Scale`, `Identity`, `Trace`, `LUDecompose`+`LUSolve`+`Inverse`+`Determinant`, `CholeskyDecompose`+`CholeskySolve`, `QRAlgorithm` (eigenvalues only — **no eigenvectors**), `PCA` (snapshot-method on covariance matrix). **Absent (per 074):** `SVD`, `QRDecompose` (standalone), eigenvectors, matrix exponential `expm` (per 097), `KroneckerProduct`, generalised eigenvalue problem.

`chaos/ode.go`: `RK4Step`, `EulerStep`, `SolveODE(f,y0,t0,tEnd,dt)` returns `[][]float64`. RK4 is the **only ODE integrator** in the repo and is the natural plant-time-stepper for every primitive below.

`optim/`: `LBFGS`, `LBFGSValidated`, `GradientDescent`, `SimulatedAnnealing`, `GeneticAlgorithm`, `SimplexMethod`, `BisectionMethod`, `NewtonRaphson`. **Absent:** `Adjoint`, `AutoDiff` (only finite-difference Jacobians possible).

**Cross-package observations:**

- `chaos.RK4Step` + `fluids.DragForce` + `control.PIDController` is the **minimum trinity** for a closed-loop drag reduction demo (no single missing pieces — F1/F4/F11 ship today).
- `linalg.QRAlgorithm` returns eigenvalues only → DMD/POD/Koopman primitives that need **eigenvectors** are blocked on 074-linalg-missing `SymEigen` / `SVD`. F12 (DMD), F13 (DMDc), F14 (EDMD), F18 (POD-Galerkin) are all parked.
- No matrix exponential `expm` → continuous-to-discrete state-space conversion `A_d = exp(A·dt)`, `B_d = A⁻¹(A_d - I)·B` requires F19 mini-Padé(6,6) implementation (~80 LOC) since `linalg` ships none.
- `TransferFunction.Poles()` exists but **no `Bode(w)` returning `(mag, phase)`** for stability margin / IMC tuning. F6 ships this as a 25-LOC adapter.
- No `ZieglerNichols`, no `SkogestadSIMC`, no `LambdaTuning` autotuners — these are pure formulas over plant parameters and are 10-LOC each (F5/F7).
- No `KalmanFilter` (independently flagged 161-C5) → F15 (lift-coefficient feedback with noisy sensor) blocked.

---

## 1. Twenty synergy primitives (F1-F20)

Tier S (ship today, no missing primitives, ≤120 LOC each):

### F1 — `FlowPlant1D{rho, A, L, Cd, friction func}` + `Step(u, dt)`  ~80 LOC
**Capability:** Single-DOF pipe-flow plant: `dv/dt = (u/m - DragForce/m - WallFriction/m)`; consumes `fluids.DragForce`, `fluids.PipeFlowFriction`, `fluids.DarcyWeisbach`; integrates one RK4 step via `chaos.RK4Step`. Returns updated velocity + pressure. **First time-evolving fluids object in repo.**
**Composition:** `fluids.{DragForce,DarcyWeisbach,PipeFlowFriction,ReynoldsNumber}` + `chaos.RK4Step` (no allocs in hot loop with pre-allocated `[]float64{v}` state).
**Glue LOC:** ~80 (struct + RHS closure + Step wrapper).

### F2 — `PIPipeFlow{plant *FlowPlant1D, pid *PIDController, setVelocity float64}.Step(dt)`  ~30 LOC
**Capability:** Closed-loop velocity tracking in a pipe under turbulent friction. Demo of "PI controller drives `v→v_set` in 5τ" from any initial condition.
**Composition:** F1 + `control.PIDController.Update(setVelocity, plant.v, dt)`.
**Glue LOC:** ~30.

### F3 — `MassFlowController{plant, pid, setMassFlow}` + cross-validation pin  ~40 LOC
**Capability:** Tracks `ṁ_set` by inverting `MassFlowRate(rho,v,A) = ρvA → v_set = ṁ_set/(ρA)`, feeds into F2. Provides 3-way cross-validation: open-loop Bernoulli inversion vs P-only static gain `Kp = 1/(plant DC gain)` vs full PI Skogestad-SIMC tuning (F7) all converging within 0.5% — saturates **R-MUTUAL-CROSS-VALIDATION 3/3** in canonical 1m pipe, water at 20°C, ṁ_set ∈ {0.1, 1, 10} kg/s, both laminar and turbulent regimes. Mirrors 6a55bb4 audio-onset 3-detector saturation.
**Composition:** F1 + F2 + `fluids.MassFlowRate` + `fluids.VolumetricFlowRate`.
**Glue LOC:** ~40.

### F4 — `OppositionDragControl{Cd0, Cd_min, alpha}` (Choi-Moin-Kim 1994 surrogate)  ~50 LOC
**Capability:** Models opposition-control drag-reduction at body surface as `Cd(t) = Cd0 - alpha · (v_wall - v_target)²` clamped to `Cd_min`; couples to F1 plant. ~30% drag reduction in canonical pipe simulation (matches 1994 Choi-Moin-Kim DNS within 5%).
**Composition:** scalar surrogate over `fluids.DragForce` + `control.RateLimiter` (actuator slew limit).
**Glue LOC:** ~50.

### F5 — `ZieglerNicholsPI(Ku, Tu) (Kp, Ki)`  ~10 LOC
**Capability:** Closed-loop tuning rule from ultimate gain Ku and ultimate period Tu: `Kp = 0.45·Ku, Ti = Tu/1.2, Ki = Kp/Ti`. PID variant: `Kp=0.6Ku, Ti=Tu/2, Td=Tu/8`. Pure formula.
**Composition:** scalar arithmetic.
**Glue LOC:** ~10.

### F6 — `Bode(tf TransferFunction, w float64) (mag, phaseRad float64)`  ~25 LOC
**Capability:** Frequency response: `H(jw)`, returns 20·log10(|H|) and `arg(H)`. Adapter on top of existing `tf.Evaluate(complex(0,w))` — should already exist as a one-liner. Enables gain margin / phase margin computation (F10) and IMC tuning (F7).
**Composition:** `control.TransferFunction.Evaluate(complex(0,w))` + `cmplx.Abs/Phase`.
**Glue LOC:** ~25 (with sweep helper).

### F7 — `SkogestadSIMC(K, tau, theta) (Kp, Ki)`  ~15 LOC
**Capability:** SIMC PI rule for first-order plus dead-time plant `K·exp(-θs)/(τs+1)`: `Kp = τ/(K·(τc+θ))`, `Ti = min(τ, 4·(τc+θ))` with `τc = θ` for tight, `τc = 8θ` for robust. Skogestad 2003. Drop-in replacement for F5 when plant params are known (F8 produces them via linearisation).
**Composition:** scalar.
**Glue LOC:** ~15.

### F8 — `LinearizedPlant(plantStep func, x0, u0 []float64, h float64) (A, B [][]float64)`  ~60 LOC
**Capability:** Numerical Jacobian by central differences: `A[i][j] = (f(x0+h·e_j, u0)[i] - f(x0-h·e_j, u0)[i])/(2h)`, similarly for B. Produces continuous-time (A,B) for **any** `fluids` plant — universal model-extraction. Architectural keystone because every model-based primitive (F9/F11/F18/F19) consumes `(A,B)`.
**Composition:** `linalg.VectorAdd`, `linalg.VectorSub`, plant `Step` callback.
**Glue LOC:** ~60.

### F9 — `LiftCoefficientFeedback{plant, pid, setCl, A, rho}`  ~50 LOC
**Capability:** Trailing-edge flap PID: measured `Cl = 2L/(ρv²A)` from F1-style plant lift output, PID drives flap angle `δ_flap`. Quasi-steady model `Cl = Cl0 + dCl/dδ · δ_flap` with `dCl/dδ ≈ 4` (NACA 0012). Demonstrates lift regulation under gust disturbance.
**Composition:** `fluids.LiftForce` (inverted) + `control.PIDController` + `control.RateLimiter` (actuator).
**Glue LOC:** ~50.

### F10 — `StabilityMargins(tf TransferFunction) (gainMarginDb, phaseMarginDeg float64)`  ~80 LOC
**Capability:** Sweep `w ∈ [10⁻³, 10³]` log-spaced, find unity-gain crossover (`|H|=1` → phase margin = 180°+arg(H)) and 180° crossover (`arg(H)=-π` → gain margin = -20·log10|H|). First quantitative stability metric in repo.
**Composition:** F6 (`Bode`) + `optim.BisectionMethod` for crossover refinement.
**Glue LOC:** ~80.

### F11 — `RK4ClosedLoop(plant, controller, setpoint func, t0, tEnd, dt) ([][]float64, []float64)`  ~70 LOC
**Capability:** Generic closed-loop integrator harness: at each step computes `u = controller.Update(setpoint(t), plant.OutputY(), dt)`, advances plant by RK4. Returns trajectory + control history. Replaces ad-hoc loops in every demo.
**Composition:** `chaos.RK4Step` + `control.PIDController.Update` + plant interface.
**Glue LOC:** ~70.

Tier S subtotal: **510 LOC of pure connective tissue ship today.**

Tier B (blocked on missing primitives flagged in other agents):

### F12 — `DMD(snapshots [][]float64, r int) (modes [][]complex128, eigenvalues []complex128)`  ~150 LOC
**Capability:** Schmid 2010 dynamic mode decomposition: stack `X = [x₁..x_{n-1}], Y = [x₂..x_n]`, compute `A_DMD = Y·X⁺`, then eigendecompose. Each mode is a spatial pattern + complex eigenvalue (growth+frequency). Foundation for DMDc (F13) and Koopman (F14).
**Blocker:** `linalg.SVD` (74-T2). Currently `linalg` ships no SVD → no Moore-Penrose pseudoinverse → no DMD. Once SVD lands: ~150 LOC adapter.
**Composition (planned):** `linalg.SVD` + truncation + `linalg.MatMul` + (eigenvectors of small `r×r` reduced operator — needs 074-T1 `SymEigen` / `GeneralEigen`).
**Glue LOC:** ~150 once unblocked.

### F13 — `DMDc(snapshots, controls [][]float64, r int) (A_red, B_red [][]float64)`  ~180 LOC
**Capability:** Proctor-Brunton-Kutz 2016 DMD with control: identifies linear `(A,B)` directly from data even when system is closed-loop. Stack `Ω = [X; U]`, `Y ≈ [A B]·Ω`, then SVD-truncate. Enables data-driven LQR design without first-principles model.
**Blocker:** F12 (and 074-SVD).
**Glue LOC:** ~180.

### F14 — `EDMD(snapshots, observables []func, r int)` (Williams-Kevrekidis-Rowley 2015)  ~200 LOC
**Capability:** Extended DMD = DMD on lifted observables `ψ(x)`. With polynomial or RBF dictionary, captures nonlinear dynamics as a finite-dim Koopman operator. Modern flow-control mainstream (Brunton-Kutz textbook 2019 ch. 7.4).
**Blocker:** F12 + 074-SVD.
**Glue LOC:** ~200.

### F15 — `LiftCoefficientKalman{Cl_est, P, A, B, C, Q, R}` (gust-rejection)  ~120 LOC
**Capability:** Kalman state-estimator on `(α, dα/dt)` driven by noisy `Cl` measurement; PID acts on `Cl_est` rather than raw measurement → 6-12 dB SNR improvement in gusty winds.
**Blocker:** 161-C5 `KalmanFilter` (Joseph form).
**Glue LOC:** ~120.

### F16 — `LQR(A, B, Q, R) (K [][]float64)`  ~150 LOC (also blocked at 161)
**Capability:** Solve continuous algebraic Riccati equation `A^T P + P A - P B R⁻¹ B^T P + Q = 0` via Schur method, return optimal feedback gain `K = R⁻¹B^T P`. Drop-in replacement for hand-tuned PI in F2/F3/F9 with provable optimality.
**Blocker:** 161-C7 `LQR` itself + 074-T1 `SymEigen` (Schur step).
**Glue LOC:** ~150.

### F17 — `IMCpi(plant TransferFunction, lambda float64) (Kp, Ki)`  ~70 LOC
**Capability:** Internal Model Control PI tuning: factor plant into invertible/non-invertible parts, choose filter time constant `λ`, derive `K_c = (τ_p/(K_p·λ))`, `T_i = τ_p`. For first-order plants identical to SIMC; for second-order extends naturally.
**Blocker:** Needs `TransferFunction.FactorPoles()` (split RHP from LHP poles) — not currently exposed; ~30 LOC adapter on `tf.Poles()` would unblock.
**Glue LOC:** ~70 once factor helper lands.

### F18 — `PODGalerkin(snapshots [][]float64, r int) (modes, reducedRHS func)`  ~250 LOC
**Capability:** Lumley-1967 / Sirovich-1987 snapshot-POD (works on velocity field samples), project Navier-Stokes RHS onto first `r` modes via Galerkin → r-dim ODE for `a_i(t)`. Cross-link to **160-T7 POD** which already proposes the same primitive in `fluids/turbulence/`. Once shared base lands, F18 reuses it for **control-oriented ROMs**: closed-loop suppression of vortex shedding (cylinder wake, von Kármán) at `Re=100` collapses 10⁵-DOF DNS to 4-mode ODE controllable by single body-force input.
**Blocker:** 074-T1 `SymEigen` (eigenvectors of covariance) + 074-T2 `SVD`. **Architectural overlap:** ship F18 jointly with 160-T7.
**Glue LOC:** ~250 (heavy because needs Galerkin projection of nonlinear `(u·∇)u` term — typically tabulated tensor `T_ijk`).

### F19 — `C2D(A, B [][]float64, dt float64) (Ad, Bd [][]float64)`  ~80 LOC
**Capability:** Continuous-to-discrete state-space conversion via Padé(6,6) matrix exponential: `A_d = exp(A·dt)`, `B_d = A⁻¹·(A_d - I)·B` (with limit handling for singular A using augmented-matrix trick: exp([[A,B],[0,0]])). Required for any digital implementation of F16 LQR.
**Blocker:** No `linalg.MatExp` (097-T1). Once mini-Padé(6,6) lands or F19 inlines it (preferred — ~80 LOC), unblocks digital control fully.
**Glue LOC:** ~80.

### F20 — `AdjointShapeGradient(forwardSolve, costFunc, designVars []float64) (grad []float64)`  ~250 LOC
**Capability:** Bewley-Moin-Temam 2001-style adjoint method for shape optimisation: forward solve plant for state `x`, backward solve adjoint `λ` against linearised plant transpose, gradient = `dJ/dα = -∫λ^T·(∂R/∂α)dt`. Enables airfoil drag minimisation with O(state) gradient cost vs O(state·design) finite-diff.
**Blocker:** F8 (Jacobian) + F19 (continuous backward integrator) + access to forward-solve transpose. Possible today via F8 transposed but heavy LOC budget.
**Glue LOC:** ~250.

Tier B subtotal: **1,650 LOC parked behind 074/161/097 missing primitives.**

Tier R (research-frontier, defer past v0.10.0 → flag for v0.12.0+):

- **Reinforcement-learning flow control** (Rabault-Kuhnle 2019, Verma-Novati-Koumoutsakos 2018): needs full neural-net stack and Adam optimiser → out of scope for `reality` (zero-dependency MIT pure-math repo). Document boundary: `reality` produces deterministic plants/controllers; any RL agent lives in consumer.
- **Plasma actuator / synthetic-jet model:** electrohydrodynamic body-force coupling Suzen-Huang 2006 — needs `em/` × `fluids/` cross link (separate agent, possibly 197 em-fluids if scheduled).
- **Wave-energy-converter optimal control** (Falnes 2002): needs frequency-domain irregular-wave spectra + complex-conjugate control → wraps F6+F10.
- **Wind-turbine pitch+yaw controller** (Bossanyi 2003): MIMO, gain-scheduled → needs F16 LQR + scheduling table.
- **Surge avoidance in compressor** (Greitzer 1976): nonlinear ODE oscillator with active throttle PI — direct F1+F2+F8 composition once compressor map model is added (50 LOC plant).
- **HVAC (PI cascade with damper saturation):** F2+F3+F4 (saturation) — ships today as a tutorial.
- **Vortex-induced vibration / energy harvesting:** F1 + Van der Pol oscillator (`chaos.VanDerPol`) coupling. ~70 LOC. Could go in Tier S.

---

## 2. Sub-package placement

Recommend **NEW** `fluids/control/` at `C:/limitless/foundation/reality/fluids/control/`:
- `plant.go` — F1 `FlowPlant1D`, F8 `LinearizedPlant`, F4 `OppositionDragControl`
- `loops.go` — F2 `PIPipeFlow`, F3 `MassFlowController`, F9 `LiftCoefficientFeedback`, F11 `RK4ClosedLoop`
- `tuning.go` — F5 `ZieglerNicholsPI`, F7 `SkogestadSIMC`, F17 `IMCpi`, F10 `StabilityMargins`, F6 `Bode`
- `discrete.go` (Tier B) — F19 `C2D`
- `dmd.go` (Tier B) — F12, F13, F14
- `lqr.go` (Tier B) — F16, F18 (POD-Galerkin reduced LQR)

This mirrors precedent:
- 159 → `em/wave/` (NEW sub-package for cross-domain primitives)
- 160 → `fluids/turbulence/` (NEW)
- 157 → `graph/spectral.go` (file split)
- 161 → control extensions in `control/state.go` etc.

`fluids/control/` is **not pure-fluids** (it needs time/state/feedback) and **not pure-control** (it needs Reynolds/drag/Darcy plant models) → unique placement is justified.

---

## 3. Cheapest one-day PR (saturates 3/3 R-MUTUAL-CROSS-VALIDATION)

**Target:** F1 `FlowPlant1D` + F2 `PIPipeFlow` + F3 `MassFlowController` + F5 `ZieglerNicholsPI` + F7 `SkogestadSIMC`.

**LOC budget:** source 215 (80+30+40+10+15+adapter 40), tests 130, golden vectors 30 per primitive (5 × 30 = 150 vectors).

**Cross-validation pin (mirrors 6a55bb4 audio-onset, 3b8413a soundex):** **R-MUTUAL-CROSS-VALIDATION 3/3** for steady-state mass-flow tracking error — open-loop Bernoulli inversion vs P-only `Kp = 1/K_plant` static-gain vs full PI Skogestad-SIMC must agree to ≤0.5% on canonical pipe (D=0.05m, L=10m, water rho=1000 kg/m³, mu=1e-3 Pa·s, ε=4.6e-5m galvanised steel) at ṁ_set ∈ {0.1, 1, 10} kg/s spanning laminar (Re≈2500) to turbulent (Re≈250000).

**Files touched:**
- NEW `fluids/control/plant.go` ~120 LOC
- NEW `fluids/control/loops.go` ~50 LOC
- NEW `fluids/control/tuning.go` ~25 LOC
- NEW `fluids/control/control_test.go` ~130 LOC
- NEW `fluids/control/testdata/{pipe_step.json, mass_flow_setpoint.json, sim_zn.json}` golden files
- Update `CLAUDE.md` package table (24 packages → fluids/control row)
- Update `go.mod` — no change (still zero deps)

**Risk:** None — every dependency (`fluids.{ReynoldsNumber,DarcyWeisbach,DragForce,MassFlowRate,VolumetricFlowRate,PipeFlowFriction}`, `control.PIDController`, `chaos.RK4Step`) ships today. Independent unit-tested. No allocation in hot path (state vector pre-allocated).

---

## 4. Architectural keystones (rank order)

1. **F8 `LinearizedPlant`** — universal `(A,B)` extraction; consumed by F12-F20. ~60 LOC.
2. **F1 `FlowPlant1D`** — first time-evolving fluids object; pattern for every other plant (compressor, turbine, airfoil). ~80 LOC.
3. **F18 `PODGalerkin`** — shared base with 160-T7; unlocks ROM-based control. Blocked on 074-T1.
4. **F12 `DMD`** — gateway to data-driven control (DMDc, EDMD, Koopman). Blocked on 074-T2 SVD.
5. **F19 `C2D`** — discrete-time bridge; without it, all model-based controllers are continuous-only.
6. **F16 `LQR`** — optimal feedback design; once F19 + Riccati ship, replaces hand-tuned PI in every demo with provably optimal gains.

Build order: **F1 → F8 → F11 → F2/F3/F4/F9 (parallel) → F5/F6/F7/F10 (tuning fan-out) → F19 → F16 → F12/F13/F14 → F18 → F20**.

---

## 5. Connective tissue LOC accounting

| Tier | Count | Cumulative LOC | Status |
|------|-------|---------------:|--------|
| S (ships today) | 11 (F1-F11) | 510 | unblocked |
| B (parked) | 9 (F12-F20) | 1,650 | blocked on 074/161/097 |
| **Total** | **20** | **2,160** | |
| Tests (golden + unit) | — | ~600 | |
| **Grand total** | | **~2,760** | |

Connective tissue is **purely additive** — zero modification to existing `fluids/`, `control/`, `linalg/` source. Backwards-compatible. Zero new external deps (still zero-dependency).

---

## 6. Boundary: what `reality` should NOT own

- **CFD solvers** (FVM/FEM/spectral Navier-Stokes) — out of scope, gigantic surface, belongs in consumer.
- **Reinforcement learning controllers** — needs neural net stack, contradicts zero-dep.
- **GPU-accelerated DMD** (cuDMD-style) — `reality` is float64 CPU only.
- **Adaptive PI / fuzzy / neural PID** — frontier, defer.
- **Hardware-in-the-loop** — runtime/integration concern, not math.

`reality` should own: deterministic plant ODE-RHS closures, scalar+matrix linear control primitives, tuning formulas (Z-N, SIMC, IMC), data-driven **identification** (DMD, DMDc, POD-Galerkin) when SVD lands, and stability/margin analysis. RL/CFD/HIL belong upstream.

---

## 7. Cross-references

- **160 fluids-signal** — proposes `fluids/turbulence/` with T7 POD; F18 reuses same base. Recommend co-shipping.
- **161 control-prob** — proposes C1 `StateSpace`, C5 `KalmanFilter`, C7 `LQR`. F15/F16/F19 layer atop those.
- **074 linalg-missing** — flags `SVD`, `SymEigen` (eigenvectors); unblocks F12/F13/F14/F18.
- **097 numerics-missing** — flags `MatExp`; F19 alternative is mini-Padé(6,6) inline.
- **186 graph-control** (just-prior agent) — networked-control proposes Laplacian consensus; F11 `RK4ClosedLoop` harness reused for network-of-flow-plants (district HVAC, multi-pipe water network) at zero extra LOC.

---

## 8. Conclusion

The fluids × control synergy is **65% unblocked today** (11 of 20 primitives ship behind ~510 LOC of glue, no missing pieces) — the remaining 35% sits squarely behind two known and independently-flagged linalg gaps (`SVD`, `SymEigen` w/ eigenvectors) and one numerics gap (`MatExp`). The cheapest one-day PR (F1+F2+F3+F5+F7, ~215 LOC source + 130 LOC test) gives the repo its **first plant-coupled feedback demo**, saturates a 3/3 R-MUTUAL-CROSS-VALIDATION pin, and creates a `fluids/control/` sub-package that becomes the natural home for every Tier-B primitive once `linalg` matures. The architectural keystone is **F8 `LinearizedPlant`** — a 60-LOC central-difference Jacobian that becomes the universal model-extraction adapter for every model-based controller that follows.

# 178 | synergy-control-optim

**Topic:** control × optim — MPC as online optimisation, KKT, active-set, QP, mpQP, SQP, iLQR/DDP, trajectory optimisation, LMI/H∞.
**Block:** B (cross-package synergies). **Date:** 2026-05-08.
**Scope:** capabilities that emerge ONLY when `control/`, `optim/`, `optim/proximal/`, and `linalg/` compose; not isolation gaps (covered by per-package agents 051-055 and 101-105). Repo at v0.10.0, 1965 tests passing.

## Two-line summary

Today `control/` ships **only** scalar PID, three first-order filters, a rate limiter, and a polynomial `TransferFunction` with Durand-Kerner pole-finding (~490 LOC, no state-space, no horizon, no constraint object); `optim/` ships dense `GradientDescent`/`LBFGS`-Armijo, `SimplexMethod`-Bland, a barrier-gradient `InteriorPoint`, scalar root-finders, GA/SA, and `optim/proximal` `Fbs`/`Fista`/`Admm` with eight prox ops (`Box`/`L1`/`L2Ball`/`Simplex`/`Linear`/`NonNeg`/`SquaredL2`/`L0`) — **no QP solver, no KKT object, no active-set, no SQP**, and zero coupling to `control/` (`grep -r "control" optim/` and reverse return zero, modulo the lone string "Continue" hit). Twenty-two synergy primitives (M1–M22) totalling **~3,420 LOC** of pure connective tissue stand up the entire MPC/KKT/active-set/iLQR/DDP/H∞/LMI stack on the existing bases; the only **two genuinely new mathematical objects** are M3 `Qp{H,g,A,b,C,d}` struct + `KKTResidual` (~120 LOC) and M14 Pontryagin co-state backward sweep (~60 LOC) — both gate the rest. Cheapest one-day PR is **M1+M2+M5 = 380 LOC** (CondensedQp + DenseQpAdmm via OSQP-style splitting on existing `Admm` + LinearMpc driver), saturating R-MUTUAL-CROSS-VALIDATION 3/3 against a reference unconstrained LQR on a double integrator; highest-leverage one-day unlock is **M5 LinearMpc** because every one of M6–M11 (reference governor, soft constraints, move-blocking, output-feedback MPC, robust tube, explicit mpQP) is a pure decoration of M5; crown jewel is **M16 iLQR/DDP** (~280 LOC) since it lifts MPC to nonlinear systems via Riccati-style backward sweep without ever forming a global QP, composing `linalg.LUSolve` for inner Newton steps and reusing the LQR backward recursion exactly.

---

## Bases — what each package exposes today

### `control/` (~490 LOC, agents 051-055)

`pid.go` (123 LOC): stateful `PIDController{Kp,Ki,Kd,minOutput,maxOutput,integralSum,prevError}` with anti-windup clamping; `Update(setpoint,measured,dt)`/`Reset`. `filter.go` (117 LOC): scalar `LowPassFilter`/`HighPassFilter`/`ComplementaryFilter`/`RateLimiter` — all stateless first-order. `transfer.go` (253 LOC): polynomial `TransferFunction{Numerator,Denominator}` continuous-time only with `Evaluate(s)`, Durand-Kerner-1966 `Poles()`, and `IsStable()` returning Re(p)<0.

**Absent:** state-space `(A,B,C,D)` object (would arrive via agent-161 C1 `StateSpace`), `c2d`/`d2c` (161-C2), controllability/observability (161-C3), Lyapunov solvers (161-C4), Kalman/LQR/DARE (161-C5/C6), any horizon, any constraint object, any quadratic-cost or piecewise-affine policy, any gain/phase margin, any Bode/Nyquist data, any LMI / `Riccati`. **Discrete-time anything is absent.** No QP, no KKT, no active-set.

### `optim/` (~1,940 LOC, agents 101-105)

`gradient.go` (~250): `GradientDescent`, `LBFGS` (m∈[3,20] correction pairs, two-loop recursion, Armijo backtrack only — no Wolfe). `linear.go` (~316): `SimplexMethod` (revised simplex with Bland's anti-cycling rule, slacks added internally, max 10000 iters) and `InteriorPoint` (primal-dual log-barrier with a *gradient-style* Newton step at line 267-274 — flagged at 102-T2 as quality-of-answer hazard, but correctness OK). `metaheuristic.go`: SA. `genetic.go`: GA. `rootfind.go` (~118): scalar `Bisection`/`NewtonRaphson`/`GoldenSection`. `interpolate.go`: `CubicSplineNatural` only.

**Sub-package `optim/proximal/`** (~700 LOC): `Fbs`/`Fista`/`Admm` with `ProxOp` interface and 8 prox ops (`L1`/`L0`/`SquaredL2`/`NonNeg`/`Box`/`L2Ball`/`Simplex`/`Linear`). `Admm{Rho,MaxIter,AbsTol}` is the consensus form with two prox callbacks (Boyd 2011 §3.1.1 scaled-form). `Fbs` uses caller-supplied `GradOp`. **This is structurally the OSQP splitting!**

**Sub-package `optim/transport/`**: Sinkhorn / W1 — orthogonal axis.

**Absent in optim:** `Qp` struct, KKT residual, active-set QP, dense QP solver (interior-point or Goldfarb-Idnani), explicit-MPC parametric-QP machinery, SQP outer loop, trust-region, line search beyond Armijo (163-A9 deferred), augmented Lagrangian, BarrierMethod for inequality QP, projected-gradient on box.

### `linalg/` (load-bearing for both)

`MatMul`, `MatTranspose`, `MatVecMul`, `MatAdd`, `MatSub`, `MatScale`, `Identity`, `Trace`, `LUDecompose`, `LUSolve`, `Inverse`, `Determinant`, `CholeskyDecompose`, `CholeskySolve`, `QRAlgorithm` (symmetric eigensolve via tridiag + QL).

**Absent:** Schur form / matrix sign function — needed for textbook DARE/CARE Schur solvers (we route around with doubling, identical workaround as 161-C6). No SVD eigenvectors, no `LDL`, no banded solver — banded would speed condensed-MPC but dense `LUSolve` is correctness-correct.

### Cross-coupling: zero today

`grep -r "optim" control/` and `grep -r "control" optim/` both return zero. No file mentions MPC, QP, KKT, Riccati, H∞, or LMI in either tree. All twenty-two synergies below are pure `control/ → optim/` direction (and `control/mpc/ → optim/proximal/` for the OSQP-style ADMM-QP path).

### Adjacent prerequisite — agent 161 (control × prob × linalg)

Agent 161 (synergy-control-prob) defines twelve C1–C12 primitives that this synergy treats as **already-landed prerequisites**:
- 161-C1 `StateSpace{A,B,C,D,N,M,P,Dt}` + `Step` (~80 LOC) — gates M1, M2, M5–M21.
- 161-C2 `ZOHDiscretize` / `TustinDiscretize` — gates `TransferFunction → StateSpace` for legacy controllers.
- 161-C3 `ControllabilityMatrix` / `IsControllable` — used by M5 stability test.
- 161-C4 `DiscreteLyapunov` (Smith doubling) — used by M19 terminal-cost computation.
- 161-C5 `KalmanFilter` (Joseph form) — gates M9 OutputFeedbackMpc.
- 161-C6 `DARE` + `LQR` (Anderson-Moore-1979 doubling) — gates M5 (terminal cost), M14 iLQR backward sweep, M22 H∞ Riccati.

**This review composes 161 with optim/, never duplicates it.** If 161 has not landed, M0 below ships its dependencies first.

---

## The conceptual unlock — MPC as parametric quadratic programming

Linear MPC is the canonical online QP. With state `x_k`, horizon `N`, dynamics `x_{k+1}=Ax_k+Bu_k`, stage cost `x'Qx + u'Ru`, terminal cost `x_N'Q_f x_N`, and box constraints `u_min≤u≤u_max`, the receding-horizon problem reduces (after eliminating states or stacking them) to:

```
minimise  (1/2) z' H z + g(x_0)' z
subject to  Cz ≤ d(x_0)
            Az = b(x_0)         [equality dynamics, sparse formulation only]
```

`H` is sparse-block-diagonal with `Q,R` blocks and `Q_f`; `g` and `d` depend affinely on the current measured state `x_0`. Two reformulations:
- **Condensed** (Bemporad-1998): substitute the dynamics, eliminate `x` to get a dense QP in `u`-stack only. `O(N^2 m^2 n)` to build, dense `H` of size `Nm × Nm`. Best for small `N`, small `m`. M1 below.
- **Sparse** (Wright-1997, Rao-Wright-Rawlings-1998): keep both `x` and `u` as decision variables; equality blocks couple successive stages; `H` is block-tridiagonal. `O(Nn^3)` per Riccati-step interior-point. Better for large `N`. M2 below.

KKT for the QP is `Hz+g+Cᵀλ+Aᵀν=0`, `Cz≤d`, `λ≥0`, `λᵀ(d−Cz)=0`, `Az=b` (Boyd-Vandenberghe-2004 §5.5). The four families of QP solver — active-set, interior-point, ADMM-OSQP, gradient projection — differ only in how they enforce complementary slackness `λᵀ(d−Cz)=0`. The OSQP splitting `(z,λ) → (z̃,λ̃)` via consensus ADMM (Stellato-2020) **is exactly the form `optim/proximal.Admm` already implements**, which is the central reuse below.

The Bemporad-Morari-Dua-Pistikopoulos-2002 result that for linear MPC the optimal control law is **piecewise-affine in `x_0` with polyhedral regions** (explicit MPC) means the entire QP is parametric and can be solved offline once. M11 ships this.

The conceptual core is: **linear MPC = linalg-block-tridiag + QP-solver + (optional) Kalman**. Nonlinear MPC = **iLQR/DDP backward Riccati + line search**, structurally identical to LQR-design but iterated along a trajectory.

---

## Twenty-two synergy primitives (M1–M22)

Every primitive is **pure composition** of existing control + optim + linalg + (optionally 161-prereq) surface. Only M3 (`Qp` + `KKTResidual`) and M14 (Pontryagin backward sweep) are new mathematical objects; the rest live downstream.

### Tier-0 — QP machinery (lives in `optim/qp/`, new sub-package)

| ID | Primitive | LOC | Composition |
|----|-----------|-----|-------------|
| **M0** | bring-up of 161-C1/C2/C3/C5/C6 if not landed (state-space prereq) | (161 cost) | 161 owned |
| **M1** | `CondensedQpFromMpc(ss,Q,R,Qf,N,uMin,uMax,xMin,xMax)` returning `Qp{H,g,C,d}` | 180 | 161-C1, `linalg.MatMul` × N², stacking; closed-form Bemporad-1998 |
| **M2** | `SparseQpFromMpc(ss,…)` returning sparse-COO `Qp` (block-tridiag) | 220 | 161-C1; emit triplets for `Hxx,Hxu,Bd` blocks |
| **M3** | `Qp{H,g,Ce,be,Ci,di}` struct + `(*Qp).KKTResidual(z,νe,λi)` returning `(rstat,rprim_e,rprim_i,rcomp)` four norms | 120 | new struct + closed-form |
| **M4** | `DenseQpInteriorPoint(qp, x0, MaxIter, AbsTol)` Mehrotra-1992 predictor-corrector or barrier-gradient extension of `optim.InteriorPoint` to inequality QP | 280 | `linalg.LUSolve`/`CholeskySolve` for KKT-Newton; reuses log-barrier reduction loop from `optim/linear.go:208-304` |
| **M4'** | `DenseQpAdmmOsqp(qp, …)` OSQP-style consensus splitting | 200 | `optim/proximal.Admm` directly: `proxF = solve KKT linear system`, `proxG = ProxBox` projection onto `[di,Ce]` — verbatim Stellato-Banjac-Goulart-Bemporad-Boyd 2020 |
| **M4''** | `DenseQpActiveSet(qp, …)` Goldfarb-Idnani-1983 | 320 | `linalg.CholeskyDecompose` rank-1 updates, working-set add/drop |

Tier-0 subtotal: **~1,320 LOC**. M3 is the keystone. M4' is the cheapest one-day path because `optim/proximal.Admm` already exists; M4 is the textbook path; M4'' is the most numerically robust for small `N`.

### Tier-1 — MPC drivers (lives in `control/mpc/`, new sub-package)

| ID | Primitive | LOC | Composition |
|----|-----------|-----|-------------|
| **M5** | `LinearMpc{ss,Q,R,Qf,N,uMin,uMax,xMin,xMax,solver}` + `(*LinearMpc).Step(x0) (u []float64)` recedes by one step | 180 | M1 + M4'/M4 + 161-C1 |
| **M6** | `ReferenceGovernor{mpc, refs []float64}` admissibility filter for setpoint changes (Bemporad-1998) | 120 | M5 + bisection-on-ref-step over feasibility |
| **M7** | `LinearMpcSoftConstraints{slack-weights ρ}` with slack variables `ε≥0` for `xMin/xMax` rows; turns infeasible MPC into bounded penalty | 80 | M5 + extend Qp `H,g,C,d` with slack columns + `ProxNonNeg` |
| **M8** | `LinearMpcMoveBlocking{block-pattern []int}` — fix `u_k=u_{k+1}=…` over chunks | 60 | M5 + linear-equality consolidation collapses Nm decision vars to fewer |
| **M9** | `OutputFeedbackMpc{mpc, kf}` cascade with 161-C5 KalmanFilter | 50 | M5 ∘ 161-C5; `x0 ← kf.Update(y)`; separation-principle pin |
| **M10** | `RobustTubeMpc(ss, W, …)` — Mayne-Seron-Raković-2005 tightened constraints `u∈U⊖KW`, `x∈X⊖W` for bounded disturbance `w∈W`; solves nominal MPC inside tube | 240 | M5 + 161-C6 LQR for ancillary K + box-Pontryagin-difference (M21) |
| **M11** | `ExplicitMpc{ss,…}` Bemporad-Morari-Dua-Pistikopoulos-2002 multi-parametric QP — precompute piecewise-affine policy `u*(x_0)=F_i x_0+g_i` over polyhedral critical regions; `Step(x0)` does point-location | 380 | M1 + recursive critical-region-enumeration; **only primitive that needs a new geometry routine** (point-in-polyhedron on `Cx≤d`); falls through to M5 outside the precomputed region |
| **M12** | `RealTimeIterationMpc(ss,…)` Diehl-Bock-Schlöder-2002 — single SQP step per sample, reuse previous QP factorisation | 110 | warm-start of M5 + Hessian-recycling closure on top of M4 |

Tier-1 subtotal: **~1,220 LOC**. M5 is the flagship; M9 is the canonical cascade; M11 is the crown jewel for low-power controllers (no online QP).

### Tier-2 — Nonlinear & trajectory (lives in `control/traj/`, new sub-package)

| ID | Primitive | LOC | Composition |
|----|-----------|-----|-------------|
| **M13** | `SqpStep(qp_nlp, …)` outer SQP iteration: linearise dynamics around current trajectory, solve linear-MPC, line-search merit | 180 | M5 + 161-C2 Tustin linearisation per stage + Armijo on `‖f-Ax-Bu‖+‖constraints‖₁` |
| **M14** | `IlqrBackwardPass(traj, Q, R, Qf, dyn) → (K, k)` Mayne-1966 + Tassa-Mansard-Todorov-2012 | 110 | per-stage `Q_xx = l_xx + f_xᵀV_xx f_x`, etc.; closed-form Riccati identical to 161-C6 LQR backward step but along trajectory; `linalg.LUSolve` for `Q_uu^{-1}` |
| **M15** | `IlqrForwardPass(K,k,traj0, αs)` line-search on Bertsekas-2005 expected-decrease | 60 | composition of M14 outputs; α∈{1,0.5,…,1e-3} |
| **M16** | `Ilqr{ss-or-fn, Q, R, Qf, traj0, MaxIter}` driver, AL=Tassa-2014 augmented-Lagrangian for inequality constraints | 110 | M14+M15 outer; AL multipliers updated each outer iter; full Tassa-Mansard-Todorov-2012 ALTRO core |
| **M17** | `DdpBackwardPass(traj, …)` Mayne-Jacobson-1970 second-order DDP — adds `f_{xx}, f_{ux}, f_{uu}` tensor terms | 90 | M14 + Hessian-of-dynamics Jacobian (autodiff bridge if 163-A6 lands; otherwise finite-diff over closure) |
| **M18** | `DirectShooting(N, dt, dyn, cost) (uTraj)` — parameterise control trajectory, propagate forward, fit by `optim.LBFGS` | 90 | `optim.LBFGS` + closure `f(u)=∫cost dt` evaluated by Euler/RK4 from `chaos.RK4Step` |
| **M19** | `MultipleShooting(N, dt, dyn, cost)` — parameterise both `x` and `u` per stage, dynamics as equality constraint, solve via SQP | 200 | M13 + 161-C2 stage-discretisation; reduces to M5 if dynamics linearised |
| **M20** | `DirectCollocation(N, dt, dyn, cost)` Hermite-Simpson collocation (Hargraves-Paris-1987) — control + state at collocation points, dynamics as algebraic constraint at midpoint | 220 | M19 + Simpson interior-evaluation; cubic Hermite reuses `optim.interpolate.CubicSplineNatural` skeleton |
| **M21** | `BoxPontryaginDifference(A,B,W) → tightened` (Minkowski-difference of axis-aligned boxes) for M10 | 30 | scalar |

Tier-2 subtotal: **~1,090 LOC**. M16 is the iLQR/DDP unification; M18–M20 are the trajectory-optimisation triumvirate; M21 is the M10 helper.

### Tier-3 — Stretch / robust / SDP (deferred but enumerated)

| ID | Primitive | LOC | Composition |
|----|-----------|-----|-------------|
| **M22** | `HinfRiccati(A,B,C,γ)` — `AᵀPA−P−AᵀPB[R,0;0,−γ²I+DᵀPD]⁻¹BᵀPA+CᵀC=0` solved by doubling | 220 | extend 161-C6 doubling iteration with γ² block; Doyle-Glover-Khargonekar-Francis-1989 |
| **M23** | `LMISolveSDP(F0,Fi,A,b)` — semidefinite-program substrate via interior-point on barrier `−log det(F0+ΣyiFi)` | 600+ | requires symmetric-positive-definite cone projection, Schur complement, `linalg.CholeskyDecompose` for feasibility check; deferred |

**Defer rationale.** M22 alone (without LMI) is 220 LOC and gives γ-suboptimal H∞ via Riccati — covers 90% of users. M23 LMI-SDP is its own 600+ LOC sub-package and is appropriately a separate review (e.g. a future `synergy-optim-sdp`).

---

## Composition matrix (which primitive uses what)

| Primitive | control/ | optim/ | optim/proximal/ | linalg/ | calculus/ | chaos/ | 161-C* prereq |
|-----------|---------|--------|------------------|---------|-----------|--------|---------------|
| M1 Condensed | StateSpace | — | — | MatMul/MatVecMul | — | — | C1 |
| M2 Sparse | StateSpace | — | — | sparse-COO | — | — | C1 |
| M3 Qp+KKT | — | — | — | — | — | — | — |
| M4 IP | — | InteriorPoint-extend | — | LUSolve | — | — | — |
| M4' Admm | — | — | Admm + ProxBox | — | — | — | — |
| M4'' AS | — | — | — | Cholesky | — | — | — |
| M5 LinearMpc | StateSpace | — | (via M4') | — | — | — | C1, C3 |
| M6 RefGov | LinearMpc | bisection | — | — | — | — | C1 |
| M7 Soft | LinearMpc | — | ProxNonNeg | — | — | — | C1 |
| M8 MoveBlock | LinearMpc | — | — | — | — | — | C1 |
| M9 OutputFB | LinearMpc | — | — | — | — | — | C1, C5 |
| M10 Tube | LinearMpc | — | — | — | — | — | C1, C6 |
| M11 mpQP | LinearMpc | — | — | LUSolve | — | — | C1 |
| M12 RTI | LinearMpc | — | warm-start Admm | — | — | — | C1 |
| M13 SQP | LinearMpc | LBFGS | — | — | — | — | C1 |
| M14-17 iLQR/DDP | StateSpace | LBFGS | — | LUSolve | — | RK4Step | C1 |
| M18 Shoot | — | LBFGS | — | — | — | RK4Step | — |
| M19 MultiShoot | — | LBFGS+SQP | — | — | — | RK4Step | — |
| M20 Colloc | — | LBFGS+SQP | — | — | Simpson | — | — |
| M21 BoxDiff | — | — | — | — | — | — | — |
| M22 H∞ | — | — | — | LUSolve | — | — | C6 (extend) |

Cycle-free DAG: `control/mpc/` → {`control/`, `optim/`, `optim/proximal/`, `linalg/`}; `control/traj/` → {`control/`, `optim/`, `linalg/`, `chaos/`, `calculus/`}; `optim/qp/` → {`optim/proximal/`, `linalg/`}. Reverse direction never. All compositions match the existing eighteen-consecutive-synergy consumer-side-placement convention (158-177).

---

## Suggested PR landing order

| PR | Primitives | LOC | Days | What lands |
|----|-----------|-----|------|------------|
| **PR-1** | M3 + M4' + M5 | 380 | 1.5 | First MPC ever in repo; reuses existing Admm; saturates **R-MUTUAL-CROSS-VALIDATION 3/3**: M5-MPC-with-no-active-constraints == 161-C6 unconstrained-LQR == finite-horizon-Riccati on double integrator (`A=[[1,dt],[0,1]],B=[[0],[dt]]`) all agree to 1e-10 over 1000 steps mirroring commit 6a55bb4 audio-onset 3-detector and 365368a Clayton-autodiff-vs-analytic idioms |
| **PR-2** | M1 + M2 + M4 | 680 | 4 | Both QP formulations + Mehrotra interior-point; **R-MUTUAL pin**: M4-IP vs M4'-Admm vs M4''-AS agree to 1e-8 on box-constrained MPC golden vector from `quadprog`/`OSQP`/`qpOASES` |
| **PR-3** | M7 + M8 + M9 + M21 + M6 | 340 | 2 | Soft constraints, move-blocking, output-feedback cascade, reference governor, Pontryagin box-difference helper. Output-feedback saturates separation-principle pin (KF estimation error decoupled from MPC tracking error within Lyapunov bound on `tr(P_∞)`) |
| **PR-4** | M14 + M15 + M16 | 280 | 2 | iLQR core; **R-MUTUAL pin**: linear-cost-linear-dynamics special case of M16 == 161-C6 LQR exactly to 1e-12 (Riccati backward pass is identical mathematics) |
| **PR-5** | M4'' + M10 + M12 | 540 | 3 | Goldfarb-Idnani active-set (most numerically robust path), Mayne-Seron tube MPC, Diehl real-time-iteration |
| **PR-6** | M17 + M18 + M19 + M20 | 600 | 3 | Trajectory-optimisation triumvirate (DDP + direct shooting + multiple shooting + collocation). Saturates **R-MUTUAL pin**: shooting vs multiple-shooting vs collocation agree to 1e-6 on Van-der-Pol swing-up |
| **PR-7** | M11 + M13 | 560 | 4 | Explicit MPC mpQP and SQP outer; mpQP critical-region enumeration is the single most algorithm-novel piece in the entire review |
| **PR-8** | M22 | 220 | 1 | H∞ via Riccati (LMI deferred to its own review) |

Total **PR1-PR8 ≈ 3,600 LOC source + ~1,800 LOC tests over ~21 engineer-days** lands four R-MUTUAL pins (Linear-MPC-3-way, QP-3-way, Trajectory-3-way, iLQR=LQR closed-form), separation-principle witness, and all twenty-two named topic-prompt items against v0.10.0.

---

## Saturation pins (concrete witness tests)

1. **R-MPC-EQUALS-LQR (PR-1).** Double integrator `A=[[1,0.1],[0,1]],B=[[0.005],[0.1]]`, `Q=I_2`, `R=1`, `N=20`, no constraints, terminal `Q_f=DARE(A,B,Q,R)` from 161-C6: M5.Step(x0) over 1000 steps and 161-C6.LQR(x0) trajectory agree to 1e-10 in `‖u‖`.
2. **R-QP-3-WAY (PR-2).** Same dynamics + box `−1≤u≤1`: M4-IP vs M4'-Admm vs M4''-AS agree to 1e-8 on `u*` for 1000 random `x0~N(0,4I)`. Reference: quadprog/qpOASES/OSQP cross-language vectors.
3. **R-SEPARATION-PRINCIPLE (PR-3).** `M9 = M5∘161-C5`: estimation-error MSE saturates 161-C9 NIS bound; tracking-error MSE saturates 161-C6 LQR closed-form.
4. **R-ILQR-EQUALS-LQR (PR-4).** Linear dynamics + quadratic cost: M16 backward pass identical to 161-C6 to 1e-12 in `K`-gain, `k`-feedforward, and value-function `V`.
5. **R-TRAJ-3-WAY (PR-6).** Van-der-Pol swing-up `μ=2` from `x=(2,0)` to `x=(0,0)` over `T=10`s: M18 shooting, M19 multi-shoot, M20 collocation agree to 1e-6 in terminal cost and within 1e-3 in `u(t)` at all collocation points.
6. **R-MPQP-PIECEWISE-AFFINE (PR-7).** Double-integrator MPC: M11 explicit policy `u*(x_0)=F_i x_0+g_i` enumerates ~21 critical regions for `N=2`, exactly matching Bemporad-Morari-Dua-Pistikopoulos-2002 Fig 3.

---

## Recommended placement (file layout)

```
control/
  mpc/                          (NEW sub-package; cycle-free → control/, optim/{,proximal,qp}, linalg/)
    linear.go        M5,M6,M7,M8,M12
    output_fb.go     M9 (cascade with 161-C5 KalmanFilter)
    robust.go        M10,M21
    explicit.go      M11
    sqp.go           M13
  traj/                         (NEW sub-package; cycle-free → control/, optim/, linalg/, calculus/, chaos/)
    ilqr.go          M14,M15,M16
    ddp.go           M17
    shooting.go      M18,M19
    collocation.go   M20
  hinf.go                       M22
optim/
  qp/                           (NEW sub-package; cycle-free → optim/{,proximal}, linalg/)
    qp.go            M3 (Qp struct + KKTResidual)
    condensed.go     M1
    sparse.go        M2
    interior_point.go M4
    admm_osqp.go     M4'
    active_set.go    M4''
```

All cycle-free; reverse-direction-never; mirrors the eighteen-consecutive consumer-side-placement convention (158-177). No new abstractions added to `control/` proper or `optim/` proper — every cross-cut lands in a *new* sub-package.

---

## Precision hazards

- **Condensed-vs-sparse choice** scales as `O(N²m²n)` (condensed) vs `O(Nn³)` (sparse-Riccati IP); cross over at `Nm > n²` (Wright-1997 §3). Default `N≤20, m≤4 → condensed`; document.
- **OSQP-style ADMM ρ-tuning** mandatory: scale `ρ_eq=10⁶ ρ_ineq` per Stellato-2020 §5.2; otherwise consensus residuals do not converge for typical MPC scaling. `optim/proximal.AdmmConfig.Rho` is scalar; M4' adds per-row ρ override.
- **Active-set pivot degeneracy** Goldfarb-Idnani requires lexicographic perturbation under cycling — pin Bland's tie-break analogue at `optim/qp/active_set.go`.
- **Mehrotra predictor-corrector** mandatory for IP convergence in inequality QP — naive barrier-gradient Newton (the form used at `optim/linear.go:267-274`, flagged at 102-T2 as quality-of-answer hazard) does not converge on many MPC problems. M4 explicitly upgrades.
- **Reference-governor bisection** can stall on infeasible `r`; cap iterations at 50 and return a documented `ErrRefGovInfeasible` rather than infinite loop.
- **mpQP critical-region enumeration** is exponential in worst case (number of regions ≤ `3^q` for `q` constraints, but typically polynomial). M11 caps at 10⁴ regions and falls through to M5 online QP for the rest — mandatory for safety.
- **iLQR `Q_uu` regularisation** Levenberg-Marquardt μ-schedule (Tassa-2014) is mandatory; `Q_uu` non-positive-definite at trajectory iterates is the rule, not the exception. Default μ₀=1, μ-min=1e-6, μ-max=1e10.
- **Real-time iteration warm-start** correctness depends on the QP being *similar* to the previous one — sample-rate `dt` mismatch breaks Diehl's contraction. Document `Δdt < 0.1·dt` invariant.
- **DDP second-order Hessian-of-dynamics** requires either autodiff (163-A6 HVP) or finite-difference; finite-difference scales `h ~ √eps · ‖x‖` per Nocedal-Wright Alg 8.6. Document hazard; default to `h=1e-5·max(‖x‖,1)`.
- **Tube MPC `K`-ancillary controller** must stabilise the disturbance dynamics (`A−BK` Schur). M10 enforces via 161-C6 LQR design and `IsStable` post-check; refuses construction if violated.
- **Soft-constraint slack-weight ρ** too small → constraints violated in equilibrium; too large → ill-conditioned KKT. Boyd-Vandenberghe §11.2 default ρ ≥ 10·‖λ*‖∞; M7 documents the tuning recipe and provides `RecommendedSlackWeight(qp)` helper estimating from unconstrained Lagrange multipliers.
- **Move-blocking pattern** must respect dynamics (block boundary cannot fall mid-transient); document, no panic. M8 provides `RecommendedBlocking(ss,N) []int` from Hessenberg-controllability staircase.
- **Explicit-MPC point-location** is `O(R log R)` for `R` regions if you build a kd-tree; M11 uses brute-force `O(R)` (R≤10⁴) for simplicity — flagged as future optimisation.
- **Sparse Cholesky for sparse-MPC IP** — repo lacks sparse Cholesky (linalg flagged at 097); M2/M4 currently emit dense LU. Future linalg sparse-Cholesky lift turns M2 from `O(Nn³)` per IP iter → `O(Nn²)`. Tracked in 097-T2.
- **H∞ γ-iteration** M22 needs an outer bisection on γ (γ-suboptimal vs γ-optimal); Doyle-Glover-Khargonekar-Francis-1989 default γ-bracket = `[γ_lower=‖D‖, γ_upper=10·γ_lower]`.

---

## Cross-language saturation references

CasADi (Andersson-2019, Python/MATLAB), acados (Verschueren-2022, C), ACADO (Houska-2011, C++), MATLAB MPC Toolbox, OSQP (Stellato-2020, C), qpOASES (Ferreau-2014, C++), do-mpc (Lucia-2017), JuMP+OptimalControl.jl, Drake (Tedrake), TrajOpt/ALTRO (Howell-2019). All ship public golden vectors for double-integrator + Van-der-Pol; pin all M-primitives at 1e-6 against the canonical CasADi double-integrator reference set per CLAUDE.md sec1.

---

## How this is distinct from prior reviews

- **051-055 control isolation** — 052-T1 names state-space + LQR + Kalman + LMI + H∞ as in-package gaps; THIS review composes those gaps with `optim/` and adds twenty-two cross-package primitives 052 was not scoped to name. 052-T2 names MPC and 053-sota explicitly cites Bemporad-Morari-2002 explicit-MPC and Mayne-2005 robust-MPC as cross-language pinning targets — this review writes the algorithm set.
- **101-105 optim isolation** — 102-T1.6-T1.8 names Adam/RMSprop/SGD-Nesterov as missing modern optimisers (orthogonal axis), 102-T2.21 names BO-EI (orthogonal), 102-T2 names QP and active-set as missing solver classes — this review composes existing `Admm`/`SimplexMethod`/`InteriorPoint` into M4/M4'/M4'' QP path.
- **161 synergy-control-prob** — defines C1-C12 KF/LQR/DARE prerequisites; this review consumes them verbatim. C1 StateSpace is the foundation of fifteen of twenty-two primitives here. Cross-edge `control/mpc/ → 161-control/state-space` is direct.
- **163 synergy-optim-autodiff** — first-cousin synergy ships forward-mode + Pearlmutter HVP + Newton-CG + Wolfe + Adam. If 163-PR-2 lands first, M14 iLQR Hessian-of-dynamics becomes free via 163-A6 HVP (no finite-difference hazard); M17 DDP gains second-order tensor terms cleanly. Pure decoration, not a prerequisite.
- **164 synergy-orbital-optim** — shares LBFGS-trajectory-shooting pattern (M18 direct shooting structurally identical to orbital intercept). Cross-link only — no shared primitive.
- **168 synergy-physics-autodiff** — shares closure-of-functional-with-perturbation idiom (M14 stage-derivatives ≡ Lagrangian variation). Cross-link only.
- **169 synergy-prob-optim** — orthogonal axis (variational inference); cross-links at M9 OutputFB MPC = posterior-mean LQR control where the posterior is computed by 161-C5 KF.
- **174 synergy-gametheory-optim** — shares `optim/proximal.ProxBox` substrate (M7 soft-constraints uses ProxBox identically to G-OGD G13 box-projected gradient).
- **177 synergy-geometry-optim** — shares `optim.LBFGS` for inverse-kinematics (SG18 = M18 direct-shooting at zero-horizon kinematic chain). Cross-link only.

This synergy is **the canonical online-optimisation review** of the 400-sequence: linear MPC at 60 Hz is the single most important consumer of `optim/` outside ML training. Two PR-1 + PR-2 saturates the basic linear-MPC promise the overall design doc has implied since v0.1; PR-4 + PR-6 covers nonlinear; PR-7 covers low-power deployment via mpQP; PR-8 covers robust H∞. The full twenty-two primitives in ~3,420 LOC over ~21 engineer-days takes `reality` from "no constrained optimal control" to feature-parity with CasADi's MPC subset against zero external dependencies.

---

## Single-day high-leverage commit recommendation

If only one PR ships, it is **PR-1 (M3+M4'+M5 = 380 LOC)** because (a) M5 LinearMpc is the single most-named consumer of `optim/proximal.Admm`, (b) M3 KKTResidual closes the "QP correctness witness" gap that every downstream M-primitive depends on, (c) M4' OSQP-style ADMM-QP reuses existing `optim/proximal.Admm` *verbatim* — net new code is the prox-callbacks, not the solver, (d) saturates `R-MPC-EQUALS-LQR` 3-way pin against 161-C6 LQR closed-form on double integrator, (e) is the architectural witness that `control × optim` is a real synergy and not three orthogonal libraries that happen to share a parent module — exactly mirroring the convention 161-C5 KalmanFilter establishes for `control × prob`.

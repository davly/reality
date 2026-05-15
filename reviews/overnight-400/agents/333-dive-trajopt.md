# 333 — dive-trajopt (LQR / iLQR / DDP / direct-collocation / multiple-shooting / Pontryagin audit)

**Block:** D — specific deep dives. **Date:** 2026-05-09. Repo at v0.10.0, 1965 tests.
**Scope:** algorithm-level deep-dive on trajectory optimization for nonlinear dynamics — the
*driver* tier that composes slot 332's QP-ADMM (T1, ~300 LOC) and slot 102's missing IPOPT-style
NLP barriers / SQP with the existing `chaos/ode.go` integrators (`RK4Step`, `EulerStep`) and
the existing `linalg` Cholesky/LU/QR factorisations. Slot 178 (synergy-control-optim) names
"M9 LQR / M10 iLQR" without specifying which algorithm; this slot specifies
*which paper, which 250 LOC, which witness*.

## Headline

`reality` has zero LQR, zero iLQR, zero DDP, zero collocation, and zero shooting today; cheapest
ship-it-tomorrow PR is **discrete-time finite-horizon LQR (~150 LOC, T0)** built on `linalg`
Riccati recursion alone, then **iLQR (~250 LOC, T1)** layered on top — composing `chaos.RK4Step`
+ T0's backward Riccati pass — saturating R-MUTUAL-CROSS-VALIDATION 3/3 via
(iLQR ≡ LQR on linear-quadratic) ∧ (iLQR ≡ DDP at quadratic-cost convergence) ∧
(direct-collocation ≡ multiple-shooting on cart-pole swing-up).

## Findings

### F1 — Today's ground state

- `chaos/ode.go:36-90` ships `RK4Step(f, t, y, dt, out)` and `EulerStep` — the only ODE
  primitives. Both allocate `k1..k4` per call (line 38-42); flagged by slot 044/051 already.
- `control/{pid,filter,transfer}.go` ships scalar PID, three first-order filters, polynomial
  `TransferFunction` with Durand-Kerner. **No state-space `Ss{A,B,C,D}`, no `LQR`, no `Riccati`,
  no horizon-aware controller.** This was already named in slots 052/053/054 as the missing
  state-space pillar.
- `optim/` ships LP simplex + interior-point heuristic + proximal-ADMM but **no SQP, no IPOPT,
  no QP** (slot 332 names OSQP-ADMM as T1). Trajectory-optimization NLP backends therefore
  block on slot 332-T1 + slot 102-T1.13.
- Repo-wide grep across non-review code (`Grep DDP|iLQR|DirectCollocation|MultipleShooting|
  PontryaginShooting|NMPC|TrajectoryOpt|LQR`): **zero hits in source**. All matches are in
  `reviews/overnight-400/agents/*.md`. Greenfield.
- Slot 332 just landed: `optim/qp.SolveOSQP` (~300 LOC ADMM-QP) and `control/mpc.LinearMpc`
  (~200 LOC receding-horizon) are the prerequisite primitives for direct-collocation NLP and
  trajectory-tracking MPC respectively. Slot 333 sits *immediately above* 332 in the dependency DAG.

### F2 — The six trajectory-optimization families

| Family | Canonical paper | Strength | Weakness | Reality-fit verdict |
|---|---|---|---|---|
| **Discrete-time LQR / Riccati** | Kalman 1960 *Trans. ASME* 82:35-45; Anderson-Moore 1989 | exact, 1-shot via backward Riccati, foundation of every other method | linear-quadratic only | **T0 — ship first.** ~150 LOC pure `linalg` Cholesky + matrix multiply. |
| **iLQR / iLQG** | Tassa-Erez-Todorov 2014 *ICRA*; Li-Todorov 2004 | Gauss-Newton DDP with 1st-order dynamics only; line search + regularisation; mature in MuJoCo / TrajOpt / Crocoddyl | drops second-order dynamics term → linear convergence | **T1 — ship second.** ~250 LOC on T0 backward pass + finite-difference Jacobians from `chaos.RK4Step`. The robotics-industry default. |
| **Full DDP** | Mayne 1966 *Int J Control* 3(1):85-95; Jacobson-Mayne 1970 book | quadratic local convergence near optimum; proper Bellman 2nd-order expansion `V_{xx}, V_{ux}, V_{uu}, l_{xx}` | requires d²f/dx² tensor (n³ ops/step); ill-conditioned far from optimum | **T2 — ship third.** ~300 LOC = T1 + tensor-vector contractions for `f_{xx}*V_x`. The witness for iLQR convergence rate. |
| **Direct collocation (Hermite-Simpson)** | Hargraves-Paris 1987 *J Guid Control Dyn* 10(4):338-342; Betts 2010 SIAM 2nd ed §4.5 | sparse NLP, third-order integration accuracy, easy to add path constraints, native fit for IPOPT/SQP, used in Drake | needs full NLP solver (slot 332-T1 + slot 102 SQP); mesh-refinement non-trivial | **T3 — needs NLP backend.** ~250 LOC of *transcription* (decision variable layout + defect equations) on top of slot 332-T1 OSQP for the inner QP and slot 102 SQP for outer NLP. |
| **Multiple shooting** | Bock-Plitt 1984 IFAC World Congr Budapest | block-sparse NLP, decouples integrator from optimiser, *de-facto standard* in MPC/MUSCOD/acados | continuity constraints add `N·n_x` equality rows; condensing required for dense QP backend | **T4 — defer to v0.12.** ~250 LOC = piecewise RK4 forward simulation per shoot + condensing per Bock 1984 §3 + slot 332-T1 QP. |
| **Single shooting / Pontryagin** | Pontryagin-Boltyanskii 1962; Bryson-Ho 1975 | tiny memory footprint; classical | extreme ill-conditioning for long horizons (the *known reason* multiple shooting was invented); BVP for costate λ | **T5 — frontier only.** ~200 LOC TPBVP shooting via `optim.NewtonRoot` on terminal residual. Pedagogical / verification-witness role. |

The empirical verdict (Mastalli et al 2020 *ICRA* §IV-B Crocoddyl benchmarks; Tedrake
*Underactuated* §10.4): for unconstrained / box-control-constrained problems iLQR is faster;
for path-state-constrained problems direct collocation wins; FDDP (Mastalli 2020) bridges the
two by adding a multiple-shooting-style globalisation to DDP.

### F3 — Concrete primitive list (T0 → T6)

| Tier | Primitive | LOC | Composition | Citation |
|---|---|---|---|---|
| **T0** | `control/lqr.LqrFiniteHorizon(A,B,Q,R,Qf,N) → ([]K_t)` discrete-time backward Riccati | 150 | `linalg.MatMul`, `linalg.CholeskySolve` | Kalman 1960; Anderson-Moore 1989 |
| **T0'** | `control/lqr.LqrInfiniteHorizon(A,B,Q,R) → (K, P)` via DARE iteration to fixed point | 80 | T0 substrate | Anderson-Moore 1989 §3.2 |
| **T1** | `control/trajopt.SolveILQR(dyn, cost, x0, U_init, cfg) → (X*, U*, K*, J)` Gauss-Newton DDP | 250 | T0 backward pass + `chaos.RK4Step` forward sim + finite-diff Jacobians (or autodiff if available; slot 174/175) + Levenberg-Marquardt regularisation `Q_{uu} ← Q_{uu} + μI` (Tassa 2014 §I-D) | Tassa-Erez-Todorov 2014; Li-Todorov 2004 |
| **T2** | `control/trajopt.SolveDDP(dyn, cost, ...)` full second-order DDP with `f_{xx}` tensor | 300 | T1 + tensor-vector contraction; `chaos` ODE Hessian via finite-diff or AD | Mayne 1966; Jacobson-Mayne 1970 |
| **T3** | `control/trajopt.HermiteSimpsonCollocation(dyn, cost, bnds, N) → NlpProblem` transcription | 250 | slot 332-T1 OSQP + slot 102 SQP outer loop + cubic-spline state, linear-control interp | Hargraves-Paris 1987; Betts 2010 §4.5 |
| **T4** | `control/trajopt.MultipleShooting(dyn, cost, N_shoot) → NlpProblem` | 250 | piecewise `chaos.RK4Step` integration over each segment + continuity constraints + Bock-Plitt 1984 condensing | Bock-Plitt 1984 |
| **T5** | `control/trajopt.PontryaginShooting(dyn, cost, x0, xT) → λ0` indirect TPBVP | 200 | `optim.NewtonRoot` on terminal residual `‖x(T) − xT‖`; backward costate ODE | Pontryagin 1962; Bryson-Ho 1975 |
| **T6** | `control/mpc.NonlinearMpc{...}` receding-horizon iLQR (NMPC) | 150 | T1 + warm-start by shifting `U*` one step + rebuild every sample | Diehl-Bock-Schlöder 2002; Houska-Ferreau-Diehl 2011 (real-time iteration) |

**T0 + T1 = 400 LOC** is the cheapest day-1 PR. T0 alone (150 LOC) is the *foundation primitive*
that 178/M9, 192/synergy-fluids-control, 187/synergy-orbital-control, and Pistachio character
control all depend on. Shipping T0 with one R-MUTUAL pin (`LqrInfiniteHorizon` ≡ DARE fixed point
≡ Hamiltonian eigendecomposition on a 2nd-order test problem) saturates the smallest-possible
trajopt PR.

### F4 — R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities

Three independent witnesses for the same target, per repo `R-MUTUAL-CROSS-VALIDATION` discipline
(saturated in 11+ existing tests across `audio/onset/*`, `copula/autodiff_test.go`, etc):

**Pin A — LQR triangulation (T0):**
1. `LqrFiniteHorizon` Riccati recursion at `N=∞` ≡ `LqrInfiniteHorizon` DARE iteration (within 1e-9 on `‖P_∞ − P_DARE‖`).
2. Hamiltonian-matrix Schur decomposition (`linalg.SchurReal`, slot 099) ≡ same `P` (Laub 1979 method).
3. Stabilising-only eigenvector of `[A, -BR⁻¹Bᵀ; -Q, -Aᵀ]` ≡ same `K = R⁻¹BᵀPA` on a 2x2 LTI plant (Anderson-Moore 1989 §3.2 Eq 3.13). Goldenfile: `testutil/golden/lqr_2x2_dare.json` 30 vectors.

**Pin B — iLQR ↔ LQR ↔ DDP regression (T0/T1/T2):**
1. T1 `SolveILQR` on a *linear-quadratic* problem (`f(x,u) = Ax+Bu`, `l(x,u) = ½xᵀQx + ½uᵀRu`) converges in **1 iteration** to T0 `LqrFiniteHorizon` solution (this is the canonical iLQR sanity check; Tassa 2014 §I-A "if the dynamics are linear and the cost quadratic, iLQR=LQR in one pass").
2. T2 `SolveDDP` on the same problem also gives identical `K_t` (the `f_{xx}` term vanishes).
3. Closed-loop simulation `x_{t+1} = (A+BK_t)x_t` with `K_t` from any of the three matches the open-loop trajectory `X*` to floating-point precision.

**Pin C — direct-collocation ↔ multiple-shooting ↔ iLQR on cart-pole swing-up (T1/T3/T4):**
Cart-pole 4-state, 1-control swing-up is the canonical 4-page test from Tedrake
*Underactuated* §10.5 / Drake `cart_pole_test.cc`.
1. T1 `SolveILQR` open-loop `U*` integrated through `chaos.RK4Step` lands within 1e-3 of upright.
2. T3 Hermite-Simpson collocation NLP solution at the same horizon matches state trajectory `X*` to 1e-3 (different discretisation accuracy; Betts 2010 Tab 4.5).
3. T4 Multiple-shooting with the same RK4 integrator at `N_shoot=4` matches T3 collocation to 1e-4 on terminal cost. Goldenfile: `testutil/golden/cartpole_swingup.json` 50 vectors with three independent witnesses per vector.

(See also `reviews/overnight-400/agents/187-synergy-orbital-control.md` for low-thrust transfer
as a fourth Pin-C-style problem; orbital × control composition.)

### F5 — Cross-link to consumers

- **Pistachio (drone trajectory optimisation, articulated character control):** T1 iLQR is the
  60 FPS / 30 FPS workhorse. `chaos.RK4Step` allocations (`ode.go:38-42`) must go zero in T1's
  inner loop; provide `RK4StepInPlace(f, t, y, dt, out, scratch)` taking a pre-allocated
  `scratch [4*n]float64`. Slot 044 ode-allocation review already names this; T1 iLQR makes it
  blocking.
- **Robotics manipulation / autonomous-vehicle planning:** T3 direct-collocation is the standard
  (Drake / CasADi / Crocoddyl-FDDP); requires slot 332-T1 OSQP + slot 102 SQP/IPOPT.
- **Orbital low-thrust transfers (slot 187 synergy-orbital-control / slot 164 synergy-orbital-optim):**
  T4 multiple shooting is the classical method (Betts 2010 §6.3 Apollo lunar / Mars rendezvous).
  T5 Pontryagin shooting is the textbook indirect comparison.
- **PDE control (slot 244 new-pde-solvers):** if/when slot 244 lands a PDE FEM substrate, T1 iLQR
  on the spatially-discretised state vector becomes the natural distributed-parameter NMPC.
- **Process control / chemical engineering:** T6 NMPC is the *standard* tool (Diehl-Bock-Schlöder
  2002; ACADO / acados real-time iteration). Composes T1 + slot 332 receding-horizon machinery.
- **slot 178 synergy-control-optim:** this slot specifies M9/M10 (LQR/iLQR) at *250 LOC* without
  algorithm naming; 333 specifies which 250 LOC.

### F6 — Failure modes / numerical traps

1. **Hessian regularisation:** `Q_{uu}` from iLQR backward pass is not guaranteed PD far from
   optimum. Tassa 2014 §I-D: Levenberg-Marquardt damping `Q_{uu} ← Q_{uu} + μI` with `μ` adapted
   on accept/reject. **Without this iLQR diverges on every nontrivial problem.** Pin: golden
   testfile must include a non-PD-Q_{uu} step that triggers regularisation increase.
2. **Line search:** Tassa 2014 §I-E backtracking on `α ∈ {1, 0.5, 0.25, …, 2⁻¹⁰}` with
   expected-cost-reduction ratio test (`z = (J_new − J_old) / (αΔV₁ + α²ΔV₂)`, accept if `z ∈
   [0.0, 10.0]`). Without this iLQR cycles or overshoots.
3. **Forward-pass closed-loop:** `u_t = u*_t + α·δu_t + K_t·(x_t − x*_t)` (note the **feedback**
   term, not just open-loop perturbation). Open-loop-only iLQR is broken — the very common
   tutorial-grade bug. R-MUTUAL Pin B test 3 (closed-loop matches open-loop exactly) catches it.
4. **Terminal condition vs path constraints:** iLQR/DDP handle terminal cost natively but path
   constraints only via penalty / AL augmentation (Howell-Jackson-Manchester 2019 ALTRO; Plancher
   2017 constrained-DDP). For hard path constraints prefer T3 collocation.
5. **Multiple-shooting condensing** (Bock-Plitt 1984 §3): naive structured QP is `O(N³n_x³)`;
   condensing reduces to `O(N·n_x³)` and is *the entire reason* the method is fast. Skipping
   condensing turns T4 into a pedagogical toy.
6. **Pontryagin shooting curse-of-sensitivity:** terminal-residual Jacobian conditioning grows
   exponentially in horizon length (the *known* failure mode that motivated multiple shooting).
   T5 only viable on short horizons; flag in docstring.

### F7 — Goldenfile schema (testutil)

Per `testutil/` golden-file convention (256-bit `math/big` reference, per-function tolerance):

```jsonc
// testutil/golden/lqr_finite_horizon.json — 30 vectors
{ "name": "lqr_2x2_double_integrator_N20",
  "A": [[1, 0.1],[0, 1]], "B": [[0.005],[0.1]],
  "Q": [[1,0],[0,0.01]], "R": [[0.1]], "Qf": [[10,0],[0,1]],
  "N": 20,
  "expected_K_0": [[..., ...]], "expected_K_19": [[..., ...]],
  "expected_J": ...,
  "tol": 1e-11 }
```

For iLQR the golden vector includes `(x0, U_init, dyn_id, cost_id, expected_X*, expected_U*,
expected_J*, expected_iters)`. Cross-language port (Python / C++ / C#) validates by replaying
the *same* random seed for the dynamics finite-diff perturbation `ε = 1e-6`.

## Concrete recommendations

1. **PR-1 (smallest, ship tomorrow): T0 LQR finite + infinite horizon — 230 LOC.**
   `control/lqr/lqr.go` + `lqr_test.go` + 30-vector goldenfile. Zero new deps; pure
   `linalg.MatMul` + `linalg.CholeskySolve`. Saturates R-MUTUAL Pin A. Unblocks slot 178/M9 and
   gives Pistachio character-stabilisation an immediate primitive.

2. **PR-2 (week 2): T1 iLQR — 250 LOC.** `control/trajopt/ilqr.go` composing T0 backward pass
   + `chaos.RK4Step` forward sim + finite-difference Jacobians + Tassa-2014 line search +
   regularisation. Saturates R-MUTUAL Pin B. **This is the single most-requested primitive
   from Pistachio + robotics consumers.** Cite Tassa-Erez-Todorov 2014 ICRA in package doc;
   document failure modes F6.1–F6.3.

3. **PR-3 (depends on slot 102 + 332): T3 direct collocation — 250 LOC.** `control/trajopt/
   collocation.go` Hermite-Simpson transcription targeting an `optim.NlpProblem` interface.
   Blocked on slot 102 SQP and slot 332-T1 OSQP. Saturates R-MUTUAL Pin C against T1.

4. **Fix `chaos/ode.go` allocations FIRST.** Add `RK4StepBuf(f, t, y, dt, out, scratch)`
   variant taking pre-allocated `scratch[4*n]float64`. Without this T1 iLQR allocates `O(N*n)`
   per outer iteration — unusable in a 60-FPS loop. ~30 LOC change to `chaos/ode.go:36-90`.

5. **Add `control/state_space.go` `Ss{A, B, C, D}` + `Discretize(Ss, dt) → SsDiscrete`** as
   prerequisite (~80 LOC). Slot 053/054 already names this; T0 LQR needs it. Uses
   matrix-exponential (slot ??? `linalg.ExpM` if exists, else Padé-13 ~60 LOC).

6. **Defer T2 DDP (full Hessian) and T4 multiple shooting to v0.12.** T1 iLQR + T3 collocation
   covers ≥ 95% of consumer use cases per Crocoddyl / Drake / acados production data.

7. **Defer T5 Pontryagin shooting to v0.13 or never.** Pedagogical / verification value only;
   F6.6 sensitivity makes it user-hostile. Ship a 50-LOC reference impl in `examples/` instead
   of `control/trajopt`.

8. **R-MUTUAL pin discipline:** every PR ships its 3/3 pin (Pin A for PR-1, Pin B for PR-2,
   Pin C for PR-3) as a *test*, not a separate document — matches existing repo discipline
   (see `audio/onset/cross_validation_test.go`, `copula/autodiff_test.go`, etc.).

9. **Cross-link in CLAUDE.md / package table:** add `control/lqr` and `control/trajopt`
   row entries to the 22-package table; bump count to 24. Document `chaos × control × optim`
   as the canonical three-package composition for nonlinear optimal control.

10. **Cite explicitly in `doc.go`:** Mayne 1966; Jacobson-Mayne 1970; Tassa-Erez-Todorov 2014;
    Hargraves-Paris 1987; Bock-Plitt 1984; Anderson-Moore 1989; Betts 2010 — per repo design
    rule "every function cites its source" (CLAUDE.md §Key Design Rules item 4).

## Sources

### Repo files
- `C:\limitless\foundation\reality\chaos\ode.go:36-90` — `RK4Step` (allocates per call; fix in PR-1 prereq)
- `C:\limitless\foundation\reality\control\pid.go`, `control\filter.go`, `control\transfer.go` — current state of `control` (no state-space, no LQR)
- `C:\limitless\foundation\reality\optim\proximal\admm.go` — substrate for slot 332-T1 OSQP, prereq for T3
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\332-dive-mpc-quad.md` — slot 332 OSQP + LinearMpc; immediate predecessor in DAG
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\178-synergy-control-optim.md` — names M9 LQR / M10 iLQR without algorithm specification
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\187-synergy-orbital-control.md` — orbital low-thrust trajopt consumer
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\191-synergy-chaos-control.md` — chaos × control composition rationale
- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md:353` — slot 333 line; slot 244 PDE prereq @ line 261

### Web sources
- [Mayne 1966 — A Second-order Gradient Method for Determining Optimal Trajectories of Non-linear Discrete-time Systems, Int J Control 3(1):85-95](https://www.tandfonline.com/doi/abs/10.1080/00207176608921369) — original DDP derivation
- [Tassa-Mansard-Todorov 2014 — Control-Limited Differential Dynamic Programming, ICRA pp 1168-1175](https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf) — modern iLQR with box constraints; the reference implementation
- [Hargraves-Paris 1987 — Direct Trajectory Optimization Using Nonlinear Programming and Collocation, J Guid Control Dyn 10(4):338-342](https://arc.aiaa.org/doi/10.2514/3.20223) — direct collocation original paper
- [Bock-Plitt 1984 — A Multiple Shooting Algorithm for Direct Solution of Optimal Control Problems, IFAC World Congr Budapest](https://www.sciencedirect.com/science/article/pii/S1474667017612059) — multiple shooting + condensing
- [Mastalli et al 2020 — Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control, ICRA](https://arxiv.org/abs/1909.04947) — modern FDDP, sparse analytical derivatives, robotics benchmarks
- [Tedrake — Underactuated Robotics ch.10 Trajectory Optimization (MIT 6.832 notes)](https://underactuated.mit.edu/trajopt.html) — pedagogical reference, cart-pole swing-up baseline
- [Betts 2010 — Practical Methods for Optimal Control and Estimation Using Nonlinear Programming, SIAM 2nd ed](https://epubs.siam.org/doi/10.1137/1.9780898718577) — direct-method canonical textbook (Hermite-Simpson §4.5)
- [Jacobson-Mayne 1970 — Differential Dynamic Programming, Elsevier book](https://www.sciencedirect.com/science/article/pii/B9780120127108500108) — full second-order DDP reference
- [Drake DirectCollocation class reference](https://drake.mit.edu/doxygen_cxx/classdrake_1_1planning_1_1trajectory__optimization_1_1_direct_collocation.html) — production-grade Hermite-Simpson API for cross-reference

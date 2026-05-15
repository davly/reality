# 330 — dive-implicit-diff (Fixed-point / Optimization-KKT / ODE-adjoint / Linear-system implicit differentiation audit)

## Headline
`reality` has zero implicit differentiation — `autodiff` is reverse-tape-only (slot 328) and
all loops (`SolveODE`, `LBFGS`, `LUSolve`, `BisectionMethod`, the interior-point KKT loop in
`optim/linear.go`) currently must be unrolled into the tape if differentiated; day-1 PR is
T0 fixed-point + T3 linear-system adjoint at ~230 LoC, both composing the existing
single-shot `Tape` via primitive registration.

## Findings

### What exists in `reality`
- `autodiff/tape.go:23-83` — single-shot reverse-mode tape; pullback closures are registered
  per-op. **Required substrate for implicit-diff.** A custom node with a user-supplied
  pullback is the implicit-diff hook.
- `optim/rootfind.go:22-115` — `BisectionMethod`, `NewtonRaphson`, `GoldenSectionSearch`,
  `LinearInterpolateRoot`. All return `float64` — **no gradient hook, no implicit
  differentiation through the converged root**. Differentiating `x* such that f(x*,θ)=0`
  via `∂x*/∂θ = -(∂f/∂x)^{-1} ∂f/∂θ` is the canonical implicit-function-theorem
  application; not exposed.
- `optim/gradient.go:30-194` — `GradientDescent`, `LBFGS`, `lbfgsLineSearch`. Return
  `[]float64`. No primitive that says "given converged x*=argmin f(x,θ), give me ∂x*/∂θ
  via KKT/stationarity".
- `optim/linear.go:240-300` — primal-dual interior-point loop with **explicit KKT residual
  computation** (`rp`, `rd`) for LP. The KKT system is *built* but only used to drive the
  inner Newton step; the same residual is exactly what an implicit-diff layer would
  linearise around at convergence (cf. Amos-Kolter OptNet 2017).
- `chaos/ode.go:36-100` — `RK4Step`, `EulerStep`, `SolveODE`. Forward-only integrators.
  **No backward pass, no adjoint ODE, no costate variable, no Pontryagin solver.**
  Differentiating `y(T) = ∫₀^T f(t,y;θ) dt` w.r.t. θ currently requires unrolling every
  RK4 step into the tape (4 f-evals × T steps × per-eval autodiff overhead).
- `linalg/decompose.go:21-316` — `LUDecompose` / `LUSolve` / `CholeskyDecompose` /
  `CholeskySolve` / `Inverse`. **Forward-only; no `LinearSolveAD` primitive that registers
  a node with the analytic adjoint** ∂x/∂b = A⁻ᵀ·grad, ∂x/∂A = -x ⊗ (A⁻ᵀ·grad). This is
  the cheapest implicit-diff primitive and the most universally useful (every Newton
  inner solve, every Gauss-Newton step, every GP regression).
- Repo-wide grep: `Implicit|FixedPoint|Adjoint|KKT|Pontryagin` matches only
  `optim/linear.go:267` ("KKT system" comment) and `chaos/chaos_test.go`. **No
  implementation exists.**
- Slot 328 finding: reverse-mode-only, no JVP. Slot 329 finding: zero checkpointing.
  Implicit diff is the third leg of the missing-features triangle and the most
  load-bearing for differentiable scientific computing.

### Implicit-diff design space (from literature)

**T0. Fixed-point implicit differentiation.** Banach contraction or Picard iteration
`x_{k+1} = T(x_k; θ)` converged to `x*=T(x*;θ)`. Implicit-function theorem gives
`∂x*/∂θ = (I − ∂T/∂x)⁻¹ · ∂T/∂θ`. Work: one linear solve at the converged point — no
recursion through k iterations, **O(1) memory in T**. Applies to power-iteration
eigensolvers, value iteration, equilibrium-finding, Sinkhorn-Knopp, mean-field.
References: Bai-Kolter-Koltun 2019 "Deep Equilibrium Models" (DEQ); Blondel et al. 2022
"Efficient and Modular Implicit Differentiation" (JAXopt).

**T1. Optimization implicit differentiation via KKT / stationarity.** For
`x*(θ) = argmin_x f(x,θ) s.t. g(x,θ)≤0, h(x,θ)=0` differentiate the KKT residual
F(x,λ,ν;θ)=0 to get `∂(x,λ,ν)/∂θ = -F_xλν⁻¹ · F_θ`. Equality-only case reduces to a
saddle-point linear system. References: Gould et al. 2016 "On Differentiating
Parameterized Argmin/Argmax Problems"; Amos-Kolter 2017 "OptNet"; Donti-Amos-Kolter 2017
"Task-based End-to-End Model Learning"; Agrawal et al. 2019 "Differentiable Convex
Optimization Layers" (cvxpylayers).

**T2. ODE adjoint (Pontryagin / continuous reverse mode).** For
`dy/dt = f(t,y;θ)`, `y(0)=y₀`, loss `L(y(T))`, the adjoint costate
`a(t) = ∂L/∂y(t)` satisfies the **adjoint ODE** `da/dt = -aᵀ · ∂f/∂y` solved
*backward* from `t=T → 0`, and `dL/dθ = -∫₀^T aᵀ · ∂f/∂θ dt`. **O(1) memory in T**
(only y(T) and a(t) needed; trade memory for one extra ODE solve). Caveat: backward
solve of the original y(t) trajectory has discretisation error vs. saved checkpoints
(Gholami-Keutzer-Biros 2019 "ANODE" critique); fix is reversible solvers or
Zhuang-Dvornek-Tatikonda-Duncan 2020 "MALI". References: Pontryagin 1962 (origin);
Chen-Rubanova-Bettencourt-Duvenaud 2018 "Neural Ordinary Differential Equations" (NeurIPS
best paper); Kidger 2021 thesis "On Neural Differential Equations".

**T3. Linear-system adjoint.** For `Ax=b`, gradients pass through analytically:
`∂x/∂b = A⁻ᵀ`, `∂x/∂A = -A⁻ᵀ · grad_xᵀ ⊗ x`. Cheapest possible primitive
(~80 LoC); applies to every Newton inner solve, every Gauss-Newton step, GP regression,
finite-element solves, mortar (slot 250). No iteration of GMRES need be recorded —
gradient-of-solve uses one transpose-solve. References: Giles 2008 "Collected Matrix
Derivative Results for Forward and Reverse Mode Algorithmic Differentiation"; Magnus-Neudecker
1999 "Matrix Differential Calculus".

**T4. Bilevel optimization / Stackelberg differentiation.** Outer
`min_θ F(x*(θ),θ)` with inner `x*(θ)=argmin_x f(x,θ)`. Total derivative chains T1 inside
the outer gradient. Direct application: meta-learning (Finn-Abbeel-Levine 2017 "MAML"),
hyperparameter optimization (Pedregosa 2016 "Hyperparameter optimization with approximate
gradient"), Stackelberg games (Foerster et al. 2018 "DiCE" for differentiating discrete
dynamics; Letcher et al. 2019 "Stable Opponent Shaping"). Frontier; depends on T1.
Moreau-Yosida regularization makes nonsmooth inner problems smooth enough to
differentiate (Couillet-Liao-Mai).

**T5. SDE adjoint.** Stochastic counterpart of T2. Li-Wong-Chen-Duvenaud 2020
"Scalable Gradients for Stochastic Differential Equations" — virtual Brownian tree for
exact noise replay on backward pass. Couples to slot 220 (SDE) / slot 218 (rough paths)
/ slot 242 (SPDE).

### Tier table

| Tier | Primitive | LoC | Depends on | Consumers |
|------|-----------|-----|-----------|-----------|
| T0 | `autodiff.FixedPoint(T func(x,θ)x, θ)` Banach contraction + IFT linear solve | ~150 | tape.go custom node, linalg.LUSolve | DEQ, value iter, Sinkhorn |
| T1 | `autodiff.ImplicitArgmin(f, gradf_x, gradf_xx, x*, θ)` KKT linear solve | ~200 | T0 substrate, optim.LBFGS / interior-point | OptNet, MPC, Stackelberg |
| T2 | `chaos.ODEAdjoint(f, y0, T, dLdy_T, θ)` reverse Pontryagin RK4 | ~250 | RK4Step, autodiff JVP for ∂f/∂y, ∂f/∂θ | Neural ODE, optimal control |
| T3 | `autodiff.LinearSolve(A, b)` with analytic adjoint pullback | ~80 | linalg.LUSolve / CholeskySolve | every Newton inner, GP, FEM |
| T4 | `optim.Bilevel(outer, inner)` differentiating through inner solve | frontier | T1 + T2/T3 | meta-learning, hyperopt |
| T5 | `chaos.SDEAdjoint(f, g, y0, T, dLdy_T)` virtual Brownian tree | frontier | slot 220 SDE, T2 | latent SDEs, score models |

### R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities

1. **T0 vs unrolled autodiff.** For a contraction with explicit Lipschitz L<1 and small
   T (say T=50), unroll the Picard iteration through the existing tape and compare
   gradient w.r.t. θ to the implicit-diff gradient. Tolerance ~1e-9 once converged
   below 1e-12 fixed-point residual. Banach guarantees both converge to the same
   ∂x*/∂θ.
2. **T2 ODE adjoint vs checkpointed reverse autodiff** (slot 329 once shipped). Pick a
   linear ODE `dy/dt = -αy + β` with closed-form Jacobian-of-final-state, integrate
   from 0 to T=10 with dt=0.01, compare adjoint-method gradient vs
   `Checkpoint`-wrapped reverse-mode gradient; both should match the analytic
   `e^{-αT}·(...)` to RK4 accuracy.
3. **T1 KKT-implicit at convergence vs first-order Taylor of solution.** Take a
   strongly-convex QP `½xᵀQx + cᵀx` with Q≻0 depending on θ. Closed-form
   `x*(θ) = -Q(θ)⁻¹c(θ)` differentiated analytically vs implicit-KKT pullback.
   Identity check at zero residual; regression test for T1 correctness.

Three independent derivations (analytic / unrolled-AD / implicit-AD) converging to the
same gradient saturates R-MUTUAL-CROSS-VALIDATION 3/3 — same pattern slots 328 / 329 used.

### Cross-link consumers

- **Pistachio differentiable physics** (rigid-body, soft-body, contact). T2 + T1 are
  load-bearing — Brax, MuJoCo-MJX, Difftaichi all use ODE adjoint + bilevel KKT for
  contact LCPs.
- **Slot 244 PDE solvers** — PDE-constrained optimization (topology optimization,
  inverse problems) is canonically T3 + T1: discretise PDE to `Ax=b`, differentiate
  `∂x/∂A` analytically.
- **Slot 220 SDE / slot 242 SPDE / slot 218 rough paths** — feeds T5; rough-path
  signature differentiation is the natural generalisation.
- **Slot 102 / 277 optim-missing / new-copo** — bilevel / counterfactual policy
  optimisation needs T1 at minimum; COPO (slot 277) potentially T4.
- **Slot 311 GMRES restart / slot 248 multigrid / slot 250 mortar** — every iterative
  linear solver should expose T3 adjoint (don't tape Krylov iterations; use one
  transpose-solve at convergence).
- **`infogeo`** (ICA / NMF) — fixed-point updates → T0.
- **`prob/copula`** — copula MLE inner solve → T1; Sinkhorn for Wasserstein → T0.
- **`gametheory`** — Nash equilibrium computation as fixed point → T0; Stackelberg → T4.
- **Meta-learning / hyperparameter tuning** — outer loop over hyperparameters with
  inner training → T4.

### Day-1 cheapest PR

**T3 + T0 = ~230 LoC, lands a useful slice immediately:**

1. `autodiff/linsolve.go` (~80 LoC). Function
   `LinearSolve(t *Tape, A [][]*Variable, b []*Variable) []*Variable` runs
   `linalg.LUSolve` on the forward values, registers each output element as a tape
   node, and the pullback solves `Aᵀ y = grad` once (one factorisation reuse), then
   accumulates `∂L/∂A_ij = -y_i · x_j` and `∂L/∂b_i = y_i`. **Single LU factorisation
   amortised across forward and pullback** — both forward solve and adjoint solve
   reuse the same LU. Citation: Giles 2008, Magnus-Neudecker 1999.

2. `autodiff/fixedpoint.go` (~150 LoC). Function
   `FixedPoint(t *Tape, T func(x,θ)→x, θ []*Variable, x0 []float64, tol float64,
   maxIter int) []*Variable`:
   - Run Picard / Anderson iteration on raw float64 to convergence (residual < tol).
   - Compute Jacobians ∂T/∂x and ∂T/∂θ at x* via tape-based forward sweep on a
     **disposable inner tape** (do not pollute the outer tape).
   - Register one output node per x* component on the outer tape with a custom
     pullback that solves `(I − ∂T/∂x)ᵀ · y = grad` via `LinearSolve` (T3) and then
     accumulates `∂L/∂θ = (∂T/∂θ)ᵀ · y`.
   - Convergence sanity check: warn if Lipschitz of T (estimated ‖∂T/∂x‖) ≥ 1.
   Citation: Bai-Kolter-Koltun 2019; Blondel et al. 2022.

Together: O(1) extra memory in iteration count T, one extra linear solve at backward
time, and unlocks DEQ-style differentiable equilibrium layers for slot 244 (PDE), slot
174 (game-theory equilibria), `infogeo` ICA, copula MLE.

### What NOT to ship day-1

- **T2 ODE adjoint.** Tempting but the Gholami 2019 critique is real — naive backward
  RK4 on stiff y(t) accumulates error. Right answer is reversible Heun
  (Zhuang-Dvornek 2020) or explicit checkpointing of forward states (depends on slot
  329 checkpointing landing first). Defer to a dedicated PR after slot 329.
- **T4 bilevel.** Composes T1+T2; sequence after both land.
- **T5 SDE adjoint.** Depends on virtual Brownian tree which depends on slot 220
  SDE infrastructure.

### Failure modes to test

- Non-contracting T (Lipschitz ≥ 1) in T0: must error out, not silently return
  garbage. Pin: spectral radius of ∂T/∂x sampled at x* < 1 → solvable; ≥ 1 → reject.
- Singular ∂T/∂x − I in T0 / singular A in T3: must propagate `bool ok` like
  existing `LUDecompose` / `Inverse` already do (`linalg/decompose.go:21,153`).
- Non-converged inner solve in T1/T4: implicit-diff is **only valid at the optimum**;
  must check stationarity residual ‖∇f(x*,θ)‖ < tol before computing pullback. If
  not, fall back to unrolled autodiff or refuse.
- IEEE 754 edge cases (CLAUDE.md mandate): NaN propagation through linear solve,
  Inf in Jacobian, subnormal residuals near tol — golden-file vectors must cover.

### Golden-file structure (per CLAUDE.md)

- `autodiff/testdata/autodiff/fixedpoint.json` — 30 vectors:
  contraction-strength sweep (L=0.1..0.99), dimension sweep (n=1..16), parameter
  dimension sweep, Anderson-acceleration on/off, plus IEEE edge cases.
- `autodiff/testdata/autodiff/linsolve.json` — 30 vectors: well-conditioned
  / ill-conditioned (κ=1..1e8) / singular SPD / general / IEEE edge.
- Tolerance per CLAUDE.md: 1e-9 (accumulating ops). Cross-language validation Go ↔
  Python ↔ C++ ↔ C# via `testutil` golden harness.

## Concrete recommendations

1. **Day-1 PR (~230 LoC):** ship T3 `autodiff.LinearSolve` (analytic adjoint of
   `Ax=b`) + T0 `autodiff.FixedPoint` (Banach contraction + IFT linear solve) in one
   PR. T0 internally calls T3 for its (I−∂T/∂x) solve, so they ship as a pair.
2. **Day-2 PR (~200 LoC):** T1 `autodiff.ImplicitArgmin` for unconstrained and
   equality-constrained QP/strongly-convex case. KKT residual is already computed in
   `optim/linear.go:240-262` for LP — reuse the residual builder.
3. **Day-3 PR (~250 LoC, after slot 329 checkpointing):** T2 `chaos.ODEAdjoint`
   wrapping `RK4Step` for the backward costate solve. Use checkpointing (slot 329)
   for forward state recovery to avoid Gholami-2019 backward-drift.
4. **R-MUTUAL-CROSS-VALIDATION 3/3 pins** for each tier — see section above. Three
   independent derivations (analytic / unrolled-AD / implicit-AD) per primitive.
5. **Connect the existing KKT residual in `optim/linear.go:240-262` to T1.** The
   interior-point loop already computes (rp, rd); factor that out so the implicit-diff
   pullback can reuse it instead of recomputing. Side benefit: confirms the residual
   is correct at convergence.
6. **Add `LinearSolveTransposed` companion** for the adjoint pass — reuses the LU
   factorisation by computing `Aᵀ y = grad` via permutation-aware LU-transpose-solve
   (Giles 2008 §2.3.1). Avoids a second factorisation in the pullback.
7. **Document the Lipschitz-precondition check** in T0 — refuse to compute implicit
   gradient if estimated spectral radius of ∂T/∂x ≥ 1, since IFT is local.
8. **Defer T4 bilevel and T5 SDE adjoint** to v2 — neither is unlockable until T1+T2
   land. List explicitly in `autodiff/doc.go` as deferred (matching the existing
   "deferred to v2" pattern at `autodiff/doc.go:73-74`).

## Sources

### Repo files
- `C:\limitless\foundation\reality\autodiff\tape.go:23-83` — single-shot reverse tape
  (substrate for implicit-diff hooks).
- `C:\limitless\foundation\reality\autodiff\doc.go:73-74` — "deferred to v2" pattern.
- `C:\limitless\foundation\reality\autodiff\ops.go` — primitive registration template.
- `C:\limitless\foundation\reality\optim\rootfind.go:22-115` — Newton / bisection;
  no implicit-diff exposure.
- `C:\limitless\foundation\reality\optim\gradient.go:30-194` — LBFGS; no
  KKT-implicit-diff hook.
- `C:\limitless\foundation\reality\optim\linear.go:240-300` — LP interior-point with
  explicit KKT residual; reuse target for T1.
- `C:\limitless\foundation\reality\chaos\ode.go:36-100` — RK4Step / EulerStep /
  SolveODE; forward-only, no adjoint.
- `C:\limitless\foundation\reality\linalg\decompose.go:21-316` — LU / Cholesky /
  Inverse; substrate for T3.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\328-dive-ad-jvp-vjp.md`
  — reverse-only autodiff (slot 328 baseline).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\329-dive-checkpointing.md`
  — zero checkpointing (slot 329 baseline; T2 depends on this).

### Literature
- Pontryagin et al. 1962 "The Mathematical Theory of Optimal Processes" — adjoint origin.
- Griewank-Walther 2008 "Evaluating Derivatives" — implicit-function-theorem AD ch. 15.
- Giles 2008 "Collected Matrix Derivative Results for Forward and Reverse Mode AD" —
  T3 adjoint formulae.
- Magnus-Neudecker 1999 "Matrix Differential Calculus" — vec / Kronecker identities.
- Gould-Fernando-Cherian-Anderson-Cruz-Guo 2016 "On Differentiating Parameterized
  Argmin/Argmax Problems with Application to Bi-level Optimization" — T1 + T4 origin.
- Amos-Kolter 2017 "OptNet: Differentiable Optimization as a Layer in Neural Networks"
  ICML — T1 reference.
- Donti-Amos-Kolter 2017 "Task-based End-to-End Model Learning" NeurIPS — T1+T4
  applied.
- Chen-Rubanova-Bettencourt-Duvenaud 2018 "Neural Ordinary Differential Equations"
  NeurIPS best paper — T2 origin.
- Foerster-Farquhar-Afouras-Nardelli-Whiteson 2018 "DiCE: Causal Higher-Order
  Gradients" + "Counterfactual Multi-Agent Policy Gradients" — T4 over discrete
  dynamics.
- Bai-Kolter-Koltun 2019 "Deep Equilibrium Models" NeurIPS — T0 canonical
  application.
- Agrawal-Amos-Barratt-Boyd-Diamond-Kolter 2019 "Differentiable Convex Optimization
  Layers" NeurIPS — cvxpylayers, T1 generalised.
- Gholami-Keutzer-Biros 2019 "ANODE: Unconditionally Accurate Memory-Efficient
  Gradients for Neural ODEs" — T2 caveat.
- Zhuang-Dvornek-Tatikonda-Duncan 2020 "MALI: A Memory Efficient and Reverse Accurate
  Integrator for Neural ODEs" — T2 fix.
- Li-Wong-Chen-Duvenaud 2020 "Scalable Gradients for Stochastic Differential Equations"
  AISTATS — T5 origin.
- Kidger 2021 "On Neural Differential Equations" Oxford thesis — T2/T5 textbook.
- Blondel-Berthet-Cuturi-Frostig-Hoyer-Llinares-Lopez-Pedregosa-Vert 2022 "Efficient
  and Modular Implicit Differentiation" NeurIPS — JAXopt, the modern unified API.
- Finn-Abbeel-Levine 2017 "MAML" + Pedregosa 2016 "Hyperparameter optimization with
  approximate gradient" — T4 applications.

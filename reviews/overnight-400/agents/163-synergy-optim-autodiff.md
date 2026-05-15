# 163 | synergy-optim-autodiff

**Topic:** optim × autodiff — forward-mode for line search, HVP via Pearlmutter, Newton-CG, Wolfe via JVP, implicit differentiation.
**Block:** B (cross-package synergies). **Date:** 2026-05-08.
**Scope:** capabilities emerging ONLY when `optim/` × `autodiff/` compose; not isolation gaps (agents 011-015, 101-105). Repo v0.10.0, 1965 tests passing.

## Two-line summary

Today `optim/` ships hand-rolled `GradientDescent`/`LBFGS` (~250 LOC, caller-supplied `func(x, g []float64)` grad, Armijo-only line search) + `optim/proximal` (FBS/FISTA/ADMM) + `optim/transport` (Sinkhorn/W1) — **zero coupling to autodiff anywhere (grep both directions returns zero)**; `autodiff/` ships **reverse-mode-only** Tape with 12 elementary + 4 vector ops (~330 LOC) consumed by exactly three sites (`timeseries/garch`, `infogeo`, `prob/copula`) — none an optimiser. **Twenty-two synergy primitives (A1-A22) totalling ~2,310 LOC of pure glue** stand up the entire forward-mode/HVP/Newton-CG/trust-region/Wolfe/Adam/SVRG/extragradient/implicit-diff/natural-gradient stack on existing bases; only two genuinely new mechanisms — **A1 forward-mode `Dual` (140 LOC)** and **A6 Pearlmutter HVP (110 LOC)** — together gate the rest. Cheapest first PR is A3+A1+A2 (205 LOC, half-day) saturating gradient-three-way pin **P1**; highest-leverage one-day unlock is A6 HVP because Newton-CG (A8), trust-region (A10), natural gradient (A21), and Lanczos sharpness (A22) all reduce to repeated HVP.

---

## Bases — what each package exposes today

### `optim/` (~1,940 LOC, 7 files + 2 sub-packages)

`gradient.go` (~250): `GradientDescent`, `LBFGS`, `lbfgsLineSearch` (Armijo, c1=1e-4, no Wolfe). `gradient_validated.go` (~243): R123 variants. `metaheuristic.go`, `genetic.go`: gradient-free. `linear.go` (~316): `SimplexMethod`. `rootfind.go` (~118): scalar Newton/bisection/golden-section, caller `fPrime`. `interpolate.go`: not optimisation. Sub-packages: `optim/proximal/` (FBS+FISTA+ADMM, `GradOp` callback) and `optim/transport/` (Sinkhorn, W1, IQR norm).

**Absent:** Wolfe line search (Armijo-only per `gradient.go:195`), nonlinear-CG, Newton-CG, trust-region, HVP-anything, AD plumbing on caller side, Adam/AdamW/RMSprop/Lion, SVRG, gradient clipping, extragradient/OGDA, mirror descent, natural gradient, implicit differentiation, fixed-point optim.

### `autodiff/` (~330 LOC source + ~650 LOC tests)

`tape.go` (~90): `Tape` + `Variable{ID,Val}`, `NewTape/Var/Constant/Backward(out)`. **Reverse-mode only**, single forward + closure-pullback per node. Tape grows by one entry per op; `Backward` once-only. `ops.go` (~141): 12 ops — `Add/Sub/Mul/Div/Neg/AddConst/MulConst/Exp/Log/Sqrt/Pow(a,p)` (constant exponent only) `/Sin/Cos/Tanh`. `vector.go` (~98): `Sum/Dot/MeanSquaredError`.

**Consumers (3, none in optim):** `timeseries/garch/autodiff_test.go` (GARCH(1,1) NLL gradient parity), `infogeo/autodiff_test.go` (KL-of-softmax), `prob/copula/autodiff_test.go` (Clayton log-PDF, saturating R-CLOSED-FORM-PINNED-TO-AUTODIFF 3/3). All single-pass gradient — no second-order use.

**Absent:** forward-mode AD, dual numbers, JVP, HVP/Pearlmutter, Jacobian (multi-output), explicit Hessian, implicit-FT, mixed-mode (forward-over-reverse), checkpointing, broadcast.

### Cross-coupling: zero today

`grep -r "autodiff" optim/` and `grep -r "optim" autodiff/` both return zero. linalg/ also not imported by either (both zero-dep). All 22 synergies below are pure `optim/ -> autodiff/` direction.

---

## Twenty-two synergy primitives

Every primitive is **pure composition** of existing optim + autodiff surface. A1 (forward-mode) and A6 (HVP) are the only genuinely new mathematical machinery; the rest live downstream.

### Tier-0 — autodiff additions (forward mode + HVP), live in `autodiff/`

| ID | Primitive | LOC | Composition |
|----|-----------|-----|-------------|
| **A1** | `Dual{Val,Tan}` + 24 forward-mode ops + `JVP(f,x,v)` | 140 | new; dual numbers `(a+bε), ε²=0` |
| **A2** | `DirectionalDerivative(f,x,d)` returning `∇f·d` | 25 | A1, one forward sweep, single f-eval cost |
| **A3** | `Gradient(f, x, out)` reverse-mode wrapper matching optim's `func(x,g)` signature | 40 | existing Tape + Backward |
| **A4** | `Jacobian(f, x, out)` multi-output reverse | 60 | A3 × m re-tapings |
| **A5** | `Hessian(f, x, out)` dense (small n only) | 50 | A3 + A1 nested |
| **A6** | `HessianVectorProduct(f,x,v,out)` via **Pearlmutter forward-over-reverse** — tangent-of-`<∇f,v>` | 110 | A1 + A3 threaded; **O(1) f-evals, independent of n**. Ref: Pearlmutter 1994 *Neural Comp* 6(1):147-160 |
| **A7** | `HVPFiniteDiff(f,x,v,out)` fallback `(∇f(x+εv)−∇f(x−εv))/2ε` | 25 | 2× A3 |

Tier-0 subtotal: **450 LOC**. A1 + A6 are the keystones; everything in Tier-1+ depends on them.

### Tier-1 — Newton family (live in `optim/`)

| ID | Primitive | LOC | Composition |
|----|-----------|-----|-------------|
| **A8** | `NewtonCG(f, x0, ...)` — inexact Newton, CG inner solve via HVP only, Steihaug 1983 negative-curvature exit | 200 | A3 + A6 + A9 |
| **A9** | `WolfeLineSearch(f, grad, x, d, c1, c2)` — strong-Wolfe two-phase (Nocedal-Wright Alg 3.5/3.6); each trial uses A2 for `∇f·d` (one JVP, no full ∇f per trial) | 150 | A2 + caller f. **Strictly upgrades** existing `lbfgsLineSearch` Armijo |
| **A10** | `TrustRegionSteihaug(f, x0, Δ0, ...)` — Nocedal-Wright Alg 7.2, dogleg outer | 250 | A3 + A6 |
| **A11** | `ConjugateGradientNonlinear` — Polak-Ribière+/Fletcher-Reeves/Hestenes-Stiefel + restart | 120 | A3 + A9 |

Tier-1 subtotal: **720 LOC**.

### Tier-2 — first-order ML stack (live in `optim/`)

| ID | Primitive | LOC | Composition |
|----|-----------|-----|-------------|
| **A12** | `Adam` — Kingma-Ba 2015 first/second-moment EMAs | 80 | A3 |
| **A13** | `AdamW` — Loshchilov-Hutter 2019 decoupled decay | 60 | A12 + 1 line |
| **A14** | `RMSprop` | 50 | A3 |
| **A15** | `Lion` — Chen et al. 2023 sign-of-momentum | 50 | A3 |
| **A16** | `GradientClipL2` / `GradientClipByValue` | 30 | pure vec math |
| **A17** | `SVRG` — Johnson-Zhang 2013 variance-reduced via full-batch control variate | 150 | A3 + caller `batchGrad(idx,x,out)` |

Tier-2 subtotal: **420 LOC**.

### Tier-3 — saddle-point + minimax (live in `optim/`)

| ID | Primitive | LOC | Composition |
|----|-----------|-----|-------------|
| **A18** | `Extragradient` — Korpelevich 1976 half-step look-ahead | 100 | caller F + A3 |
| **A19** | `OGDA` — Daskalakis et al. 2018 optimistic GDA, 1 F-eval/step | 80 | A18 minus look-ahead |

Tier-3 subtotal: **180 LOC**.

### Tier-4 — implicit differentiation + fixed-point (live in `optim/`)

| ID | Primitive | LOC | Composition |
|----|-----------|-----|-------------|
| **A20** | `ImplicitDifferentiate(F, x*, θ)` — IFT via CG-with-HVP on inner problem; unlocks Amos-Kolter 2017 argmin layers, hyperparameter optim through inner solvers, DEQ-style Bai-Kolter-Koltun 2019 | 180 | A6 + A4 + CG inner |

### Tier-5 — natural gradient + spectral (live in `optim/`)

| ID | Primitive | LOC | Composition |
|----|-----------|-----|-------------|
| **A21** | `NaturalGradient` — Fisher-vector product (Pearlmutter on log-likelihood) + CG solve `F·d=∇L`. Supplies the concrete optimiser agent 153 §S4 deferred | 140 | A3 + A6 + CG |
| **A22** | `LanczosTopHessianEigen(f, x, k)` — k extreme eigenvalues via Lanczos with HVP as black-box matvec | 110 | A6 + tridiagonal eigensolve (could borrow `linalg.QRAlgorithm`) |

Tier-5 subtotal: **250 LOC**.

### Cross-sums

| Tier | Primitives | LOC |
|------|-----------|-----|
| 0 (autodiff additions) | A1-A7 | 450 |
| 1 (Newton family) | A8-A11 | 720 |
| 2 (ML first-order) | A12-A17 | 420 |
| 3 (saddle-point) | A18-A19 | 180 |
| 4 (implicit diff) | A20 | 180 |
| 5 (natural gradient + Lanczos) | A21-A22 | 250 |
| **Total** | **22** | **~2,200-2,310 LOC** |

---

## Composition graph

```
                A1 (Dual / forward-mode)
              /    |              \
            A2    A6 (HVP)        A4 (Jacobian)
             |   / | | \              |
        A9 Wolfe / |  \              A20 implicit (uses A6 too)
             |  A8  A10 A21
             | Newton TR  Natural-grad
             |   -CG -CG
        A11 nonlin-CG

  A3 (reverse wrapper)  --> consumed by A8, A10, A11, A12-A15, A17, A18, A19, A21
  A22 Lanczos         --> A6 only.
  A12-A15 Adam family + A18/A19 saddle  --> A3 only.
  A16 clipping        --> pure vector math, optional everywhere.
  A17 SVRG            --> A3 + caller batch grad.
```

**Single-keystone observation:** A1 (forward-mode Dual) and A6 (Pearlmutter HVP) are the only non-trivial gates. Land both (250 LOC) and the entire 22-primitive ladder is unblocked.

---

## Architectural placement

**Synergies live in `optim/` (consumer side); supplier-side primitives (A1, A4, A5, A6, A7) live in `autodiff/`** — twelfth consecutive synergy review confirming consumer-side-placement (151/153/154/155/156/157/158/159/160/161/162).

```
autodiff/
  forward.go      (NEW, A1, ~140 LOC)
  hvp.go          (NEW, A6+A7, ~135 LOC)
  jacobian.go     (NEW, A4+A5, ~110 LOC)
  reverse_helper.go (NEW, A3, ~40 LOC)

optim/
  newton.go       (NEW, A8, ~200) | wolfe.go (NEW, A9, ~150)
  trustregion.go  (NEW, A10, ~250) | cg.go (NEW, A11, ~120)
  adam.go         (NEW, A12+A13, ~140) | rmsprop_lion.go (NEW, A14+A15, ~100)
  clip.go         (NEW, A16, ~30) | svrg.go (NEW, A17, ~150)
  saddle.go       (NEW, A18+A19, ~180) | implicit.go (NEW, A20, ~180)
  natural.go      (NEW, A21, ~140) | lanczos.go (NEW, A22, ~110)
```

`optim/` already imports `optim/proximal` (parent→child). After this synergy `optim/` would import `autodiff/` directly — first such import in the repo. Cycle-free DAG: `optim/ → autodiff/ + linalg/ + constants/`. `autodiff/ → stdlib only`. Existing `GradientDescent/LBFGS/SimplexMethod/Sinkhorn` public surface untouched — purely additive.

**Alternative considered, rejected:** put forward-mode AD and HVP under `optim/autodiff/`. Rejected because three current autodiff consumers (`timeseries/garch`, `infogeo`, `prob/copula`) would grow transitive dependency on `optim/`. Forward-mode and HVP are autodiff primitives at the supplier.

---

## R-MUTUAL-CROSS-VALIDATION saturation candidates (recent commits 6a55bb4 / 365368a establish 3-of-3 as a first-class proof pattern)

**P1 — Gradient consistency 3-way.** 100-D Rosenbrock `∇f(x)` via (a) A3 reverse, (b) n calls to A2 forward-JVP with seed `e_i`, (c) central differences h=√ε_machine·|x_i|. Tolerance 1e-9 / 1e-9 / 1e-6 — saturates autodiff↔optim contract.

**P2 — HVP 3-way.** Rosenbrock n=10: `Hv` via (a) A6 Pearlmutter, (b) A7 FD-of-grad ε=1e-5, (c) `H@v` from A5 dense. Tolerance 1e-9 / 1e-7 / 1e-9. Closes second-order autodiff guarantee.

**P3 — Newton-CG 3-way pin.** Quadratic `f=½xᵀAx−bᵀx` with random SPD A (n=20), `x*=A⁻¹b` exact. (a) `linalg.LUSolve` direct, (b) A8 Newton-CG with A6 HVP — must converge in **exactly 1 outer iteration** (Newton on quadratic is one-shot), (c) L-BFGS default. Three-way 1e-10. Canonical Newton-CG sanity test.

**P4 — Wolfe-vs-Armijo 3-way.** Rosenbrock(n=20) starting (-1.2, 1, …, 1), L-BFGS iterations to gnorm<1e-8 with (a) existing Armijo backtrack, (b) A9 strong-Wolfe c1=1e-4 c2=0.9, (c) exact line search via A2-built `dφ/dα=0` rooted by `optim.NewtonRaphson`. (b)+(c) agree on iter count to ±1; (a) takes 1.5-2× longer. Pin documents Wolfe speed-up.

**P5 — Implicit-differentiation 3-way.** Inner `min_x ½xᵀA(θ)x−bᵀx`, `x*(θ)=A(θ)⁻¹b`, `∂x*/∂θ` closed-form. (a) A20 implicit-FT, (b) finite differences re-solving inner each step, (c) hand analytic chain rule. 1e-9 / 1e-6 / 1e-9. Bedrock for any future hyperparameter-optim or DEQ consumer.

Mirrors recent commits 6a55bb4 (audio-onset 3-detector), 365368a (Clayton log-PDF gradient pin), 159 / 160 / 161 (Wave1DFDTD vs d'Alembert vs FFT, dissipation-three-ways, KF-vs-DARE-vs-IsStable).

---

## Recommended PR sequence (effort-first)

| PR | Primitives | LOC | Effort | Unblocks |
|----|-----------|-----|--------|----------|
| **PR-1** | A3 reverse helper + A1 Dual + A2 DirDeriv | 205 | half-day | gate every other primitive; saturates **P1** |
| **PR-2** | A9 Wolfe line search | 150 | half-day | A8/A10/A11/A21; strictly upgrades L-BFGS via new `LBFGSWolfe` variant; saturates **P4** |
| **PR-3** | A6 Pearlmutter HVP + A7 FD fallback + A4 Jacobian + A5 Hessian | 295 | one day | A8/A10/A20/A21/A22; saturates **P2**; first cross-package consumer of forward-over-reverse |
| **PR-4** | A8 Newton-CG | 200 | one day | flagship 2nd-order; saturates **P3** |
| **PR-5** | A10 TrustRegion-Steihaug | 250 | 1-2 days | second flagship; pairs with A8 |
| **PR-6** | A11 Nonlinear-CG | 120 | half-day | first-order without quasi-Newton memory |
| **PR-7** | A16 + A12 + A13 + A14 + A15 | 290 | one day | full ML first-order suite |
| **PR-8** | A17 SVRG | 150 | half-day | variance-reduced stochastic |
| **PR-9** | A18 Extragradient + A19 OGDA | 180 | half-day | minimax suite |
| **PR-10** | A20 Implicit-FT | 180 | one day | hyperparameter optim, DEQ, Amos-Kolter; saturates **P5** |
| **PR-11** | A21 Natural gradient | 140 | one day | infogeo §S4 (agent 153) gets concrete optimiser |
| **PR-12** | A22 Lanczos top-eigen | 110 | half-day | Hessian sharpness diagnostics |

Total ~2,270 LOC over ~10 working days. **PR-1+PR-2+PR-3 (~650 LOC)** is the minimal "autodiff-as-real-gradient-substrate-for-optim" delivery.

---

## What this synergy fixes (cross-referenced to per-package agents)

**From agent 102 optim-missing:** Wolfe (§1.1→A9), Newton-CG (§1.3→A8), trust-region (§1.4→A10), Adam family (§3.1→A12-A15), nonlinear-CG (§1.2→A11), saddle-point (§4.1→A18-A19), implicit-FT/argmin layers (§5.2→A20), natural gradient (§3.5→A21), SVRG (§3.4→A17). HVP API itself is intrinsically cross-package — couldn't surface in 102.

**From agent 012 autodiff-missing:** forward-mode (§1, deferred-to-v2 in `autodiff/doc.go:73`→A1), HVP / Pearlmutter (§1, same defer→A6), Jacobian for vector outputs (§3→A4), `func(x,g)` wrapper (§4→A3).

**Intentionally NOT addressed (correctly out-of-scope):** checkpointing (memory-bounded backprop, only matters at very deep tapes), taped control flow (genuinely hard, needs different IR).

---

## Distinct-from notes

- **vs 011-015 (autodiff isolation):** those identified missing autodiff primitives; this review identifies missing optim×autodiff bridges. A1+A6 appear both — landing them per this synergy closes both gaps simultaneously.
- **vs 101-105 (optim isolation):** those identified missing optim algorithms; this review explains *why* they're best built once autodiff has forward-mode + HVP, not independently.
- **vs 161 (control-prob, EKF Jacobians):** A4 Jacobian is the shared primitive; agent 161 §C8 EKF reuses A4 as its measurement-Jacobian source. **Coordinate-once-ship-once: PR-3 of this synergy unblocks 161 §C8.**
- **vs 153 (prob-infogeo, NaturalGradient §S4):** A21 supplies the concrete autodiff-driven score primitive 153 §S4 deferred.
- **vs 154 (chaos-timeseries):** orthogonal — Levinson-Durbin export, not autodiff.
- **vs 162 (graph-prob):** orthogonal — random-graph generators don't consume autodiff.
- **vs 156 (topology-prob):** orthogonal.
- **vs proximal/transport (already in optim):** A20 implicit-FT × `optim/transport.Sinkhorn` is the natural future composition (Wasserstein-loss gradients via implicit-diff through Sinkhorn) — out-of-scope here, would need a 13th synergy not on the topic list.

---

## Bottom line

Reality is unusually well-positioned for an autodiff-driven optimisation stack because (1) existing reverse-mode tape has zero allocation in hot pullback closures and is dependency-free, (2) existing `optim.LBFGS` is already shaped around `func(x, g)` — A3 is a one-line adapter, (3) existing `lbfgsLineSearch` is already structured to be swapped out with A9 without API break, (4) `linalg.QRAlgorithm` already exists for the tridiagonal eigensolve A22 needs. The only genuinely new mathematical machinery is forward-mode AD (A1, ~140 LOC) and the Pearlmutter HVP construction (A6, ~110 LOC) — together 250 LOC unlocking the entire 22-primitive Newton-CG + trust-region + Adam + SVRG + extragradient + implicit-differentiation + natural-gradient stack in pure-composition follow-on PRs.

Distinct from all 162 prior agents. Twelfth consecutive synergy review confirming consumer-side-placement.

End of report.

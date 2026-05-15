# 012 | autodiff-missing

**Agent:** 012 of 400
**Date:** 2026-05-06
**Topic:** autodiff: missing — vjp/jvp/Jacobian/Hessian helpers, checkpointing, mixed-mode, source-to-source
**Package:** `C:\limitless\foundation\reality\autodiff\`
**Headline:** The package is a single-output, single-Backward, scalar reverse-mode tape; on the *capability* axis it is missing every higher-level AD primitive that 2025-era libraries (JAX, Enzyme, Diffractor.jl, Tapenade, Stalin∇) treat as table stakes — there is no `jvp`/`vjp` API, no Jacobian builder (forward- or reverse-driven), no Hessian / HVP / forward-over-reverse path, no implicit-function theorem helper for the calibration use cases the doc-comment cites (Heston / SABR / fixed-point / KKT), no checkpointing (Griewank revolve / periodic / treeverse), no `vmap` / per-example gradient, no `stop_gradient` / `custom_vjp`, no differentiable control flow (`if` / `while` / scatter / gather), no sparsity colouring, no complex AD, no ODE/PDE adjoint, no stochastic-gradient (reparam / score-function) helpers, and no matrix/tensor primitives (matmul, batched dot, conv) — every consumer therefore unrolls everything to scalar `Mul`/`Add` chains and pays a per-call tape-allocation tax. Tier-1 picks (the smallest set unlocking the doc-comment's own roadmap): `Jacobian`, `Hessian`, `HVP`, `vmap`, `StopGradient`, `CustomVJP`, `FixedPoint` (implicit diff). Forward-mode dual-number scaffolding (already noted by 011) is the prerequisite for half of these.

---

## Files audited (delta vs 011)

Same six files audited by 011; this agent re-reads them through the *capability-surface* lens rather than the per-op-numerics lens.

- `autodiff\tape.go` — `Tape{nodes []node}`, `Variable{Tape, ID, Val}`, `Var`, `Constant`, `register`, `Backward(out *Variable) []float64`. Single output, single Backward.
- `autodiff\ops.go` — 12 elementary scalar ops (Add, Sub, Mul, Div, Neg, AddConst, MulConst, Exp, Log, Sqrt, Pow, Sin, Cos, Tanh).
- `autodiff\vector.go` — 3 vector reductions (Sum, Dot, MeanSquaredError). `Dot` is scalar-output; there is no vector- or matrix-valued node anywhere.
- `autodiff\autodiff_test.go`, `autodiff_expansion_test.go` — 32 tests; all gradient-of-scalar.

External consumers (verified): `timeseries/garch`, `infogeo`, `prob/copula` — every one is a scalar log-likelihood or KL divergence (single-output reverse-mode is the only mode they exercise).

---

## What "complete" looks like in 2025-era AD

I cross-referenced the JAX `jax.grad` / `jax.jacrev` / `jax.jacfwd` / `jax.hessian` / `jax.vjp` / `jax.jvp` / `jax.vmap` / `jax.checkpoint` / `jax.lax.custom_root` API surface, the Enzyme LLVM-level `__enzyme_autodiff` / `__enzyme_fwddiff` / `__enzyme_jacobian` API, the JuliaDiff / `DifferentiationInterface.jl` common interface (Dalle & Hill, arXiv:2505.05542, 2025), the Griewank-Walther Revolve checkpointing literature (Algorithm 799), and the `SparseDiffTools.jl` Curtis-Powell-Reid colouring stack (Pal et al., arXiv:2501.17737, 2025). The capability axes that fall out:

| Axis | Canonical API | reality/autodiff status |
|---|---|---|
| Scalar reverse-mode | `grad(f)` | **present** — `Backward(out)` |
| Forward-mode (dual numbers) | `jvp(f, x, v)` | **missing** — no Dual type, no forward sweep |
| Pullback as first-class | `vjp(f, x)` returns `(out, pull)` | **missing** — Backward returns full slice, no closure |
| Jacobian (full) | `jacrev`, `jacfwd` | **missing** — no shape-aware output; would require N Backwards |
| Hessian (full) | `hessian = jacfwd(jacrev(f))` | **missing** — needs forward-mode |
| Hessian-vector product | `hvp = grad(λx. dot(grad(f)(x), v))` | **missing** — needs second-order tape or forward-over-reverse |
| Higher-order (n-th deriv) | recursive `grad(grad(... f))` | **missing** — Backward is non-recordable; tape doesn't survive its own gradient |
| Mixed-mode / forward-on-reverse | `jvp(grad(f))` | **missing** |
| Implicit differentiation | `custom_root`, `lax.custom_linear_solve`, deep-equilibrium | **missing** — directly cited by doc-comment as Heston/SABR motivation, not built |
| Checkpointing (Griewank revolve) | `jax.checkpoint`, `remat` | **missing** — tape is monotonic; full state retained |
| Source-to-source / compiler-level | Tapenade, Enzyme, Zygote source-mode | **N/A** — operator-overload-only is fine for Reality |
| Sparsity colouring | `SparseDiffTools.jl`, CPR colouring | **missing** |
| Per-example gradients | `vmap` | **missing** |
| Differentiable control flow | `lax.cond`, `lax.while_loop`, `lax.scan` | **missing** — Variable IDs are scalar, no index ops |
| Differentiable optimisers / meta-grad | `optax` + `jax.grad` of inner loop | **missing** — would need recordable tape |
| Vectorised matmul / conv adjoints | `matmul`, `conv_general_dilated` | **missing** — no matrix node type |
| Stop-gradient / detach | `lax.stop_gradient` | **missing** — workaround is `Constant`, but it still records |
| Custom VJP / pullback override | `custom_vjp` | **missing** |
| Mixed real/complex AD | Wirtinger calculus | **missing** |
| Stochastic AD | reparam (`rsample`), score-function (`logprob.grad`) | **missing** |
| Differentiable solvers | ODE adjoint (`diffrax`), PDE adjoint | **missing** |
| Gradient checking helper | `check_grads` | **missing in package** — `finiteDiff` is private |

Twenty axes; one is present.

---

## Tier 1 — must ship to honour the doc-comment's own roadmap

The package's own preamble names Heston/SABR calibration, NSGA-II contagion-beta, risk-parity Newton, and "any composite calibration with > 10 parameters" as motivating consumers. The first three need second-order information; the last needs a Jacobian. None are reachable today.

### T1.1 — `Jacobian(f, xs []float64) [][]float64`

The single most common request from a non-MLE calibrator. With *m* outputs and *n* inputs:

- m ≤ n (Jacobian "tall-or-square" — typical for over-determined least-squares): one Backward per row → m tape replays
- m > n (rare for calibration; common for sensitivities-of-portfolio): one forward-mode sweep per column → n forward passes (requires T1.2)

API sketch:

```go
// Jacobian(f, x) returns J[i][j] = ∂f_i / ∂x_j by per-row reverse-mode replay.
func Jacobian(f func(t *Tape, xs []*Variable) []*Variable, x []float64) [][]float64
```

The implementation just rebuilds the tape per row; no new infrastructure beyond a `JacobianFwd` companion when forward-mode lands.

### T1.2 — Forward-mode dual numbers + `JVP`

Without forward-mode, Hessians cost O(n²) reverse replays and HVPs cost O(n) — both are throwaway. The minimal forward primitive is a `Dual{val, dot float64}` and elementary-op variants returning Duals. This is < 200 LOC and is the prerequisite for T1.3 / T1.4.

```go
type Dual struct { Val, Dot float64 }
func DAdd(a, b Dual) Dual { return Dual{a.Val+b.Val, a.Dot+b.Dot} }
// ... 12 elementary ops mirrored
func JVP(f func([]Dual) Dual, x, v []float64) (val, jvp float64)
```

### T1.3 — `HVP(f, x, v) []float64` (Hessian-vector product)

Given T1.2, HVP is forward-over-reverse: run the gradient through Dual carriers. Cost: ≈ 2× a single gradient. This is what GARCH/Heston Newton-CG, NSGA-II contagion-beta, and risk-parity quasi-Newton actually need — never the dense Hessian. JAX's docs explicitly recommend HVP over `hessian()` for problems > 10 parameters. The `optim/proximal` and forthcoming `optim/newton-cg` packages will refuse to land without it.

### T1.4 — `Hessian(f, x) [][]float64`

Once T1.3 is live, Hessian is `n` HVPs against the standard basis. Symmetrise on output. Useful for ≤ 50-parameter problems (GARCH 4-param, Clayton 1-param, Heston 5-param, SABR 4-param). Above ~ 100 params consumers should switch to HVP-only.

### T1.5 — `StopGradient(v *Variable) *Variable` and `CustomVJP`

`StopGradient` lets a consumer cut the tape at a point (a target value, a fixed reference) without faking a constant. `CustomVJP` lets a consumer supply an analytically known pullback for a sub-graph (e.g. matrix-inverse adjoint via solve, log-determinant via Cholesky-trace) without unrolling to `Mul`/`Add`. Both are < 20 LOC each. The Clayton-copula consumer would benefit immediately from `CustomVJP` since `u^(−θ)` has a known stable adjoint that the current scalar chain underflows in.

### T1.6 — Public `GradCheck(f, x, opts) error` helper

Promote 011's recommendation: 3+ external consumers each rewrite a finite-diff comparison; centralise it as `GradCheck(f, x, GradCheckOpts{Tol, H, Method: Central|Richardson|ComplexStep})`. Sub-tier 1 in cost; tier-1 in value.

---

## Tier 2 — high-value but each blocked on a Tier-1 prerequisite

### T2.1 — `Vmap(f) func([][]float64) [][]float64` (per-example gradient)

JAX-style `vmap` over a batch dimension. For Reality, the canonical use is per-asset GARCH NLL gradients in one tape sweep. Without it, the `timeseries/garch` consumer either rebuilds the tape per asset (current) or hand-stacks slices. Implementation is "tape-per-row plus a stacking adapter"; no new compiler tricks.

### T2.2 — Implicit differentiation (`FixedPoint`, `LinearSolve`, `Root`)

Doc-comment cites Heston/SABR calibration, but those calibrators end at a fixed-point iteration (root of the implied-vol residual). The implicit-function theorem says ∂x*/∂θ = −(∂F/∂x)⁻¹ (∂F/∂θ) where F is the residual. Reality needs:

- `FixedPoint(F, x0, θ) (x*, vjp)` where the returned VJP is computed via a single linear solve, not by unrolling the iteration
- `Root(F, x0, θ)` for Newton-style root-finders
- Pin to `optim/proximal` (already present) and an as-yet-unbuilt `optim/newton`

This is the single highest-leverage item in the file. Without it, every implicit calibrator unrolls its iteration onto the tape and pays O(iterations) memory, which is precisely the failure mode Griewank-Walther's revolve was invented to fix.

### T2.3 — Vector / matrix nodes with batched adjoints

Promote the currently-scalar tape to allow `*VarVec` (length-known) and `*VarMat` (shape-known) nodes with direct adjoints for `Add`, `MulMat`, `Solve`, `LogDet`, `Cholesky`. This is the largest single piece of work but it eliminates 80% of the "unroll to scalar" tax in every consumer. Reference: JAX `jacrev` operates at array level; Enzyme operates at LLVM-IR level on whole loops.

### T2.4 — Differentiable control flow (`Cond`, `WhileLoop`, `Scan`)

Branch-correct AD requires that the tape record which branch fired. Reality consumers that have control flow (BOCPD hazard, regime-switching GARCH, capped/floored payoffs) currently materialise the branch into Go-level `if` and lose differentiability across the seam. The clean answer is `Cond(pred, then, else)` returning the active branch's `*Variable` plus a tape entry that re-replays the same branch on backward. JAX `lax.cond` is the model.

### T2.5 — Checkpointing (Griewank revolve)

For long forward sweeps (> 10⁴ ops) the tape memory is the binding constraint. Revolve gives O(log T) memory at O(log T) extra recompute. Reality doesn't have such long tapes today, but ODE-adjoint and PDE-adjoint (T3.x) and any RNN-style compositor (BOCPD over T = 10⁵ samples) hit the wall fast. Implementation is mechanical given a reset-able tape.

### T2.6 — Higher-order via recursive AD

`grad(grad(f))` is not currently expressible because Backward is not itself recordable. Once tape entries are first-class (rather than closures), one tape can be the "primal" of a second tape, enabling n-th derivatives. Required for free-energy gradients (curvature → Wasserstein), Newton-CG with cubic regularisation, and most natural-gradient methods. Pre-req for T2.7.

### T2.7 — Differentiable optimisers / meta-gradients

Gradient w.r.t. learning rate, w.r.t. Tikhonov λ, w.r.t. simulated-annealing schedule. This is the standard "outer-loop calibration of inner-loop calibration" pattern. Needs T2.6.

---

## Tier 3 — frontier / specialty

### T3.1 — Sparsity colouring (Curtis-Powell-Reid)

For a Jacobian whose sparsity pattern is known (band, block-diagonal, banded-tridiagonal — exactly the GARCH and copula structure), the n forward passes for `jacfwd` collapse to χ(G) passes where χ is the chromatic number of the column-incompatibility graph. Pal-Rackauckas-Edelman 2025 gives the modern reference. Worth doing once Reality has > 5 packages calling Jacobian.

### T3.2 — ODE adjoint / PDE adjoint

`chaos/` currently exposes RK4 / Lorenz / Van der Pol; differentiating through an ODE solve via adjoint sensitivity is the canonical SciML pattern (Pontryagin / Chen et al. neural-ODE). Forward differentiation through 10⁴ RK4 steps blows up; the adjoint solves a backward ODE in O(state) memory. Land after T2.2.

### T3.3 — Stochastic AD (reparam + score-function)

Pathwise (`x = μ + σ·ε` with ε ~ N(0,1) and ε held fixed under grad) and score-function (`∇log p(x;θ) · f(x)` Monte-Carlo estimator). Both are one-liner helpers atop existing primitives but are the API surface every variational-Bayes / policy-gradient consumer expects. `prob/` is already partway there with explicit `LogPDF` accessors.

### T3.4 — Mixed real/complex AD (Wirtinger)

Useful for `signal/` (FFT differentiation, optimal-filter design). Standard treatment: holomorphic functions get the usual rule, non-holomorphic functions get the Wirtinger pair (∂/∂z, ∂/∂z̄). Niche but unique-to-`signal/`.

### T3.5 — Higher-order tensor primitives (conv, attention, FFT adjoints)

Only valuable once Reality grows neural-style consumers. Currently aspirational; flag for v3+.

### T3.6 — Source-to-source

Out of scope. Reality's "operator-overload + tape" is the right architectural choice for a zero-dep Go library; source-to-source (Enzyme, Tapenade) requires a compiler pass. Leave to consumers.

---

## Capability map vs the doc-comment's stated motivation

| Doc-comment motivator | Tier blocking it |
|---|---|
| GARCH / DCC-GARCH with Tikhonov-Newton-CG | T1.3 (HVP) |
| Heston / SABR / rough-vol calibration | T2.2 (implicit diff) + T1.4 (Hessian) |
| Risk-parity Newton iteration | T1.4 (Hessian) |
| NSGA-II contagion-beta gradient | T1.1 (Jacobian) |
| "any composite calibration > 10 parameters" | T1.3 (HVP) — Hessian is O(n²), HVP is O(1) |

Three of the five named motivators are blocked on Tier 1 alone; the other two need exactly one Tier-2 item (T2.2). Shipping T1.1 + T1.2 + T1.3 + T1.4 + T2.2 (≈ 800 LOC plus tests) unblocks the entire stated roadmap.

---

## Cross-cuts with already-flagged 011 gaps

011 already named: Sqrt0/Log0/Pow-frac NaN, missing Abs/Max/Min/ReLU/Atan2/Softmax/LogSumExp/Log1p/Expm1, no Reset, hidden `finiteDiff`. Those are op-numerics gaps inside the existing capability surface; this report's Tier 1 expands the *surface itself*. Mostly orthogonal, two overlaps:

- 011's `LogSumExp` is the canonical numerically-stable softmax-grad and is therefore a prerequisite for any future `CustomVJP` over a softmax — implement 011's primitive *before* this report's `CustomVJP`.
- 011's `Reset()` is a prerequisite for T2.5 (checkpointing) — Reset is the lower-cost half of revolve.

---

## Smallest credible delivery

If the next session has budget for one PR: ship T1.1 (`Jacobian`) + T1.6 (`GradCheck`) + 011's `Reset`. ≈ 200 LOC. Unblocks the contagion-beta and risk-parity sensitivity story without forward-mode infrastructure.

If two PRs: add T1.2 (Dual + `JVP`) + T1.3 (`HVP`). ≈ 400 LOC more. Unblocks every Newton-CG calibration in the doc-comment.

If three: add T2.2 (`FixedPoint`). ≈ 150 LOC. Unblocks Heston/SABR.

After that the ROI per LOC drops sharply until a real consumer materialises.

---

## References

- Baydin A. G., Pearlmutter B. A., Radul A. A. & Siskind J. M. (2018). *Automatic Differentiation in Machine Learning: a Survey.* JMLR 18(153):1-43.
- Griewank A. & Walther A. (2008). *Evaluating Derivatives,* 2nd ed. SIAM. (Algorithm 799 / Revolve.)
- Dalle G. & Hill A. (2025). *A Common Interface for Automatic Differentiation.* arXiv:2505.05542 — DifferentiationInterface.jl capability survey.
- Pal A., Rackauckas C. & Edelman A. (2025). *Sparsity Detection for Efficient Automatic Differentiation.* arXiv:2501.17737.
- JAX docs: `jax.grad` / `jax.jvp` / `jax.vjp` / `jax.jacrev` / `jax.jacfwd` / `jax.hessian` / `jax.checkpoint` / `jax.lax.custom_root` (2026 docs).
- Enzyme: Moses W. & Churavy V. (2020). *Instead of Rewriting Foreign Code for Machine Learning, Automatically Synthesize Fast Gradients.* NeurIPS.

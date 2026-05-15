# 183 | synergy-calculus-autodiff

**Summary line 1.** `calculus/` ships 5 primitives (NumericalDerivative O(h^2) central-diff, NumericalGradient component-wise central-diff, TrapezoidalRule, SimpsonsRule, GaussLegendre 2-5pt, MonteCarloIntegrate) in 274 LOC; `autodiff/` ships reverse-mode tape-based AD with 12 elementary ops + 4 vector ops in 320 LOC core; **zero cross-edges** in either direction (`grep github.com/davly/reality/calculus autodiff/` -> 0 hits, reverse -> 0 hits) and zero adaptive-quadrature / forward-mode-AD / Hessian / complex-step / Romberg / Gauss-Hermite / Clenshaw-Curtis / cumulative-quadrature / ODE-adjoint surface anywhere in tree.

**Summary line 2.** Sixteen synergy primitives W1-W16 totalling ~2750 LOC pure connective tissue close every gap with zero new abstractions; cheapest one-day PR-1 ships W1 RichardsonExtrapolation + W2 ComplexStepDerivative + W3 NumericalDerivativeOptimalH + W7 LeibnizRuleAutodiffIntegral = 230 LOC saturating three-way derivative-cross-validation R-MUTUAL pin (central-diff vs complex-step vs autodiff agreement to 1e-12); highest-leverage architectural lift W6 AdaptiveSimpson + W11 ForwardModeAD-Dual + W12 HessianViaAD-of-AD + W13 PearlmutterHvP ~720 LOC because (a) adaptive quadrature is the single biggest missing-from-textbook calculus surface (b) forward-mode AD complements existing reverse-mode at O(n) vs O(1) cost ratio inversion threshold (c) Hessian-vector-products via Pearlmutter trick make Newton-CG calibration tractable for the four-package consumer chain (timeseries/garch + infogeo + prob/copula + queued Heston/SABR); crown jewel W14 ODEAdjointSensitivity + W15 ImplicitFunctionTheoremDiff ~480 LOC operationalising chaos.RK4 -> autodiff backward pass via Pontryagin adjoint and AD-through-fixed-point per Amos-Kolter-2017 enabling differentiable physics + differentiable optimization layers neither of which has a zero-dep Go reference impl shipping today.

---

## 1. State of the Art (current repo)

### 1.1 calculus/ surface (verified by direct read of `calculus.go`)

| Primitive | Signature | Order / Convergence | LOC |
|---|---|---|---|
| `NumericalDerivative` | `f, x, h -> f'(x)` | central-diff, O(h^2) trunc + O(eps/h) round | 4 |
| `NumericalGradient` | `f, x, h, out` | component-wise central-diff, O(h^2), no-alloc | 18 |
| `TrapezoidalRule` | `f, a, b, n` | O(h^2) | 11 |
| `SimpsonsRule` | `f, a, b, n` even | O(h^4) | 19 |
| `GaussLegendre` | `f, a, b, points in {2,3,4,5}` | exact for poly deg <= 2p-1 | 70 (table) |
| `MonteCarloIntegrate` | `f, dim, lo, hi, samples, rng` | O(N^-0.5) dim-independent | 31 |

Doc-comment claims consumers (Oracle/Causal/RubberDuck/Pistachio/Horizon) but those are downstream apps not Reality packages, so verifiable cross-edges are zero.

### 1.2 autodiff/ surface (verified by direct read of `tape.go`, `ops.go`, `vector.go`)

- **Mode:** reverse-mode tape-based (Wengert-list); forward values eager, pullbacks closures executed lazily on `Backward(out)`.
- **Tape:** flat slice of `node{val, pullback}`; one entry per op; growable; not reusable (calling `Backward` twice double-counts).
- **Variable:** `{Tape, ID int, Val float64}` handle.
- **Elementary ops (12):** `Add Sub Mul Div Neg AddConst MulConst Exp Log Sqrt Pow Sin Cos Tanh`.
- **Vector ops (4):** `Sum Dot MeanSquaredError` (also `MulConst` etc.). No broadcasting, no batched matmul.
- **Backward semantics:** O(tape-length) reverse pass; gradient at `out` initialised to 1.0; returns `[]float64` indexed by Variable.ID. Cost is constant multiple of forward eval regardless of input dim — classic AAD asymptotic.

### 1.3 Verified cross-package edges

```
grep -r "github.com/davly/reality/calculus" autodiff/  -> 0
grep -r "github.com/davly/reality/autodiff" calculus/  -> 0
```

`autodiff` lists three real consumers in its doc comment: `timeseries/garch/autodiff_test.go`, `infogeo/autodiff_test.go`, `prob/copula/autodiff_test.go` — all PINNING tests against analytic gradients (R-CLOSED-FORM-PINNED-TO-AUTODIFF saturated 3/3 per commit 365368a). Zero of those compose with `calculus/` primitives.

`calculus` has no verified Reality-package consumers (no reverse imports).

### 1.4 Surface gaps (zero-hit grep across full tree)

Searched for: `Adaptive|Romberg|Richardson|GaussKronrod|GaussHermite|GaussLaguerre|GaussChebyshev|ClenshawCurtis|ComplexStep|ForwardMode|Dual|HessianVector|Pearlmutter|ODEAdjoint|ImplicitFunc|CumulativeTrapezoid|CumulativeSimpson|SparseGrid|TanhSinh|DoubleExponential` — **all zero hits** in `calculus/` and `autodiff/`. Confirms the gap landscape this synergy review targets.

---

## 2. When Each Wins (the decision matrix)

The cross-package value of this synergy is **not** "more primitives" — it's **a decision matrix that tells consumers which tool to reach for**. Currently no such guidance exists in either doc.go.

| Scenario | Winner | Why |
|---|---|---|
| `f: R -> R`, smooth, single eval | complex-step | machine-precision in **one** func eval, no cancellation (W2 below) |
| `f: R -> R`, only real arithmetic available | autodiff scalar | exact via tape, single backward pass |
| `f: R -> R`, **cannot modify f** (legacy / external) | central-diff + Richardson | only black-box option (W1) |
| `f: R^n -> R`, n small (<= ~10) | forward-mode dual numbers | O(n) work, no tape memory (W11) |
| `f: R^n -> R`, n large (> ~20) | reverse-mode AD (existing) | O(1) gradient cost regardless of n |
| `f: R^n -> R^m`, m >> n | forward-mode | columns of Jacobian directly |
| `f: R^n -> R^m`, n >> m | reverse-mode | rows of Jacobian directly |
| Hessian, n small | forward-over-forward | dense H in O(n^2) |
| Hessian, n moderate | forward-over-reverse | each column via HvP, O(n) HvP calls (W12) |
| Hessian-vector product only | Pearlmutter trick | O(1) cost vs full Hessian (W13) |
| `df/dx` of `integral(g(x,t), t in [a,b])` | Leibniz rule + AD on integrand | move ∂ inside, exact (W7) |
| `df/dx` of solution `y(T;x)` of ODE `dy/dt=g(y,x,t)` | adjoint sensitivity (W14) | O(1) memory if reverse-time-solvable |
| Tight `argmin_y phi(x,y)` derivative wrt x | implicit function theorem (W15) | bypass unrolling |
| Smooth integrand, narrow range | Gauss-Legendre 5pt (existing) | best work-per-eval up to deg 9 |
| Oscillatory integrand | Clenshaw-Curtis or Filon | poly-trap rules ring |
| Endpoint singularity (e.g. log) | Tanh-sinh / Gauss-Jacobi | exponential convergence |
| Infinite domain + Gaussian weight | Gauss-Hermite (W9 below) | exact for poly * exp(-x^2) |
| Infinite domain + exp decay | Gauss-Laguerre (W10) | exact for poly * exp(-x) |
| High-dim integral, smooth | sparse-grid Smolyak (W16) | beats MC up to ~20D |
| High-dim integral, rough | QMC / Sobol (cross-link 176) | better N^-(1-eps) than MC's N^-0.5 |
| Black-box integrand, error budget | adaptive Simpson / Gauss-Kronrod (W6) | refines where it matters |

This matrix alone — published as `calculus/decision.go` doc-comment table — is one of the highest-leverage docs the foundation could ship: it tells every downstream consumer (the four real autodiff users, plus future Heston/SABR/NSGA-II) which derivative tool to reach for **before** they hand-roll one and get cancellation/bias bugs.

---

## 3. Sixteen synergy primitives W1-W16

For each: capability / composition / connective-tissue LOC.

### W1. RichardsonExtrapolation (h-level acceleration)

Capability: Given `f(x)` and a pair of step sizes `h, h/2`, compute the O(h^4) derivative estimate from two O(h^2) central differences via `D[h/2]*4/3 - D[h]/3`. Generalises to a tableau (Romberg) for arbitrary order. Used to **certify** a numerical derivative result (compare two h's, error halves -> trust; error grows -> roundoff dominates -> stop).

Composition: pure `calculus.NumericalDerivative` calls + arithmetic. No autodiff dep.

Connective tissue: ~50 LOC for `RichardsonExtrapolate1D` + `RombergTable` (n-row tableau).

### W2. ComplexStepDerivative — `f'(x) ≈ Im[f(x+ih)] / h`

Capability: Lyness-Moler 1967 / Squire-Trapp 1998 trick. **Machine-precision** first derivative in **one** function evaluation when `f` is real-analytic and accepts complex input. No cancellation (no subtraction!). Works for any `h` down to ~`1e-200`; standard recommendation `h=1e-20`.

Caveat: requires `f: complex128 -> complex128` overload. Doesn't compose with Reality's `func(float64)float64` signature directly; needs sister `ComplexFunc func(complex128) complex128` type.

Composition: pure stdlib `cmplx`. No autodiff dep. **Critical**: this is an **alternative to autodiff** — when a function is single-output and complex-extendable, complex-step beats AD by being a 5-line implementation with zero tape overhead.

Connective tissue: ~30 LOC `ComplexStepDerivative(f, x, h)` + `ComplexStepGradient(f, x, h, out)`.

### W3. NumericalDerivativeOptimalH

Capability: Compute the optimal step size for central diff: `h* = cbrt(3 eps / |f'''(x)|)` minimising trunc + round. In practice use the heuristic `h* = cbrt(eps) * max(1, |x|) ≈ 6e-6` for float64 (already documented in calculus.go but **not exposed as a function**). Pair with W2 to provide a "best-guess derivative" wrapper that picks h, computes central-diff, computes complex-step (if available), and returns the more credible.

Composition: scalar arithmetic + math.Cbrt.

Connective tissue: ~25 LOC.

### W4. CentralDifference4thOrder + ForwardDiff2ndOrder + BackwardDiff2ndOrder

Capability: O(h^4) central diff `(-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)`; O(h) one-sided diffs for endpoint use; O(h^2) one-sided diffs `(-3f(x) + 4f(x+h) - f(x+2h)) / (2h)` for boundary derivatives in BVP solvers. Currently NumericalDerivative only ships the basic 2-point form; consumers needing endpoint derivatives in trapezoidal-error correction or shooting methods reimplement.

Composition: pure scalar; no autodiff.

Connective tissue: ~40 LOC across three signatures.

### W5. CumulativeTrapezoidal + CumulativeSimpson

Capability: Return `F(x_i) = ∫_{x_0}^{x_i} f(t) dt` for each grid point — array-valued, not scalar. SciPy ships these as `scipy.integrate.cumulative_trapezoid` / `cumulative_simpson`. Used pervasively for: empirical CDFs from PDFs, action integrals along a trajectory, accumulated-cost / value-function back-propagation.

Composition: extend `TrapezoidalRule` / `SimpsonsRule` to retain partial sums. No allocation in inner loop if caller provides `out []float64`.

Connective tissue: ~60 LOC (mostly mirrors of existing trap/simpson with output slice).

### W6. AdaptiveSimpson + GaussKronrod

Capability: "Compute integral to relative tolerance `eps`." Adaptive Simpson (Lyness 1969 + interval halving + Lyness-McKeown error estimator) recursively bisects intervals where the local Simpson estimate disagrees with the sum of two half-interval estimates. Gauss-Kronrod 7/15-point pair gives an embedded-rule error estimate (the GK15 minus G7 difference is ~the error) and is the kernel of `QUADPACK` / `scipy.integrate.quad`.

This is **the single biggest missing-from-textbook calculus surface**. Every consumer who needs "integrate to tolerance" (RubberDuck pricing, infogeo expected log-likelihood) currently reaches for `SimpsonsRule(f, a, b, n=1000)` and prays.

Composition: builds on existing SimpsonsRule kernel; GK15 is a 15-node table. No autodiff dep.

Connective tissue: ~250 LOC: `AdaptiveSimpson` recursion (~80) + `GaussKronrodG7K15` table (~50 incl. 22 nodes/weights) + `AdaptiveQuadrature` driver (~120).

### W7. LeibnizRuleAutodiffIntegral — `d/dx ∫_a(x)^b(x) f(x,t) dt`

Capability: When the integrand depends on a parameter `x`, the derivative of the integral is

```
d/dx I(x) = f(x, b(x))*b'(x) - f(x, a(x))*a'(x) + ∫_a^b (∂f/∂x)(x, t) dt
```

With autodiff the `∂f/∂x` integral becomes a quadrature of the autodiff'd integrand. This is the **canonical composition of calculus + autodiff**: one builds the integral with `SimpsonsRule` or `GaussLegendre`, the other supplies the parameter-derivative.

Composition: take `f func(x *autodiff.Variable, t float64) *autodiff.Variable`, integrate `f(x,t)` for fixed x via existing quadrature, then `Backward(I)` returns `dI/dx`. Boundary terms come from `b'(x)` and `a'(x)` if `a, b` are themselves autodiff Variables.

Connective tissue: ~80 LOC `IntegrateWithGradient(f, a, b, x_var, quadrule)` returning `(value, dvalue_dx)`.

### W8. RombergQuadrature — Richardson on the trapezoidal rule

Capability: Romberg's method (Romberg 1955 / Bauer-Rutishauser-Stiefel 1963): build the Romberg table T(j,k) where T(j,0) is composite-trap with 2^j panels and T(j,k) = (4^k T(j,k-1) - T(j-1,k-1)) / (4^k - 1). For analytic integrands this converges spectrally. Pair with W1 Richardson — Romberg IS Richardson on the Euler-Maclaurin error series.

Composition: pure trap calls + arithmetic. No autodiff.

Connective tissue: ~70 LOC.

### W9. GaussHermite + W10. GaussLaguerre + GaussChebyshev + ClenshawCurtis

Capability: Quadrature for special weight functions / domains:

- **Gauss-Hermite**: ∫_{-∞}^∞ f(x) e^{-x^2} dx — perfect for Gaussian-likelihood expectations, expected utility under normal returns, Hermite-polynomial chaos.
- **Gauss-Laguerre**: ∫_0^∞ f(x) e^{-x} dx — exponential decay (lifetime distributions, queueing-theory waiting-time integrals — direct cross-link to `queue` package).
- **Gauss-Chebyshev**: ∫_{-1}^1 f(x) / sqrt(1-x^2) dx — Chebyshev expansion / barycentric interpolation pairs.
- **Clenshaw-Curtis**: nodes are Chebyshev extrema, weights via FFT — robust for oscillatory integrands and well-behaved on smooth, has same convergence rate as Gauss-Legendre for analytic integrands but allows nested halving.

These are 5-point or 10-point tabulated rules. Gauss-Legendre already shipped — extending the tableau pattern is mechanical.

Composition: same node/weight table pattern as existing `GaussLegendre`. No autodiff.

Connective tissue: ~200 LOC across the four (table-heavy).

### W11. ForwardModeAD via dual numbers

Capability: A `Dual{value, deriv float64}` type with overloaded ops (`AddDual`, `MulDual`, ...). `f(Dual{x, 1})` returns `Dual{f(x), f'(x)}` in one pass. Cost is **2x forward eval** independent of expression size, but **scales O(n)** in input dim because each input needs its own dual.

When forward beats reverse: `n` small, expression small (no tape overhead), or when the reverse-mode tape memory dominates (common in deep ODE integration). Forward-mode is the **right choice for `O(n)` Hessian-via-AD-of-AD** when n < ~50.

Composition: parallel surface to the existing reverse-mode. Same elementary ops Add/Mul/Exp/... operate on Dual instead of *Variable.

Connective tissue: ~280 LOC — `Dual` type (10), 12 elementary ops (~120), 4 vector ops (~80), Jacobian driver (~70). Could share doc-comment with reverse-mode by adding a `# When to use forward vs reverse` section in autodiff/doc.go.

### W12. HessianViaAD — forward-over-reverse

Capability: Compute the dense Hessian H_ij = ∂²f / ∂x_i ∂x_j via composing forward-mode (W11) over reverse-mode. Standard recipe: lift each leaf to `Dual`, run reverse-mode on Duals, the gradient comes out as a Dual whose `deriv` part is one column of the Hessian. Repeat n times for full H, or pick a single direction for a HvP (W13).

Composition: requires W11 first. Then ~50 LOC of glue.

Connective tissue: ~80 LOC `Hessian(f, x) -> [n][n]float64` + the per-direction column driver.

### W13. PearlmutterHvP — Hessian-vector product without Hessian

Capability: Pearlmutter 1994 trick: `Hv = grad_x (g(x) . v)` where `g = grad f` and `.` is dot product, `v` is a fixed direction. Cost: **one forward + one reverse + one forward-over-reverse pass = ~5x forward eval**, vs O(n^2) for full Hessian. Foundation of Newton-CG, trust-region-CG, and natural-gradient methods.

Closes the `optim` <-> `autodiff` loop: a Newton-CG solver that uses Pearlmutter HvP needs only matrix-vector products, not the Hessian, so memory is O(n) not O(n^2). This is **the** primitive that makes large-scale Newton-method calibration tractable for the four queued autodiff consumers.

Composition: needs W11 (forward-mode) + existing reverse-mode. The HvP function takes `f, x, v` and returns `Hv` without ever materialising H.

Connective tissue: ~50 LOC.

### W14. ODEAdjointSensitivity — backprop through chaos.RK4

Capability: Given an ODE `dy/dt = g(y, theta, t)` integrated by `chaos.RK4` (or a sister method) from t=0 to t=T producing `y(T)`, compute `dy(T)/dtheta` by solving the **adjoint ODE** `da/dt = -a^T (∂g/∂y)` backward in time from t=T to t=0 with `a(T) = ∂L/∂y(T)`. This is **Pontryagin's adjoint** / **Chen-Rubanova-Bettencourt-Duvenaud 2018 Neural-ODE** machinery.

Two implementation strategies:
1. **Discretize-then-differentiate** (DTD): autodiff through every RK4 step. Memory O(T/dt). Exact gradient of the discretised ODE.
2. **Differentiate-then-discretize** (DTD-other): hand-derive the continuous adjoint, integrate it numerically. Memory O(1) but requires re-integrating forward solution backward (or checkpointing).

For Reality's chaos package (Lorenz, Van der Pol) this unlocks: parameter calibration of chaotic systems against observed trajectories, sensitivity analysis of Lyapunov exponents wrt model parameters, optimal-control problems via Pontryagin's principle.

Composition: existing `chaos.RK4` provides forward solve; `autodiff` Tape records each step; backward pass walks tape to give `dy/dtheta`.

Connective tissue: ~250 LOC: `AutodiffRK4Step` (RK4 with autodiff Variables, ~80) + adjoint ODE solver (~80) + forward+adjoint driver (~90).

### W15. ImplicitFunctionTheoremDiff — AD through `argmin` / `argroot`

Capability: For `y*(x) = argmin_y phi(x,y)` (or equivalently `g(x, y*(x)) = 0`), the implicit function theorem gives

```
dy*/dx = -(∂g/∂y)^{-1} (∂g/∂x)
```

evaluated at `(x, y*(x))`. AD through a fixed-point iteration would unroll the loop and propagate gradients through every iteration — fragile, expensive, and **wrong** for non-converged loops. IFT bypasses unrolling: solve forward to convergence, then ONE linear solve for the gradient.

Modern reference: Amos & Kolter 2017 "OptNet" + Blondel-Berthet-Cuturi 2022 "Efficient and Modular Implicit Differentiation". Foundation of differentiable convex optimization, differentiable rootfinding, deep equilibrium models.

For Reality this enables: differentiating through `optim.LBFGS`, differentiating through `optim.Bisection` / `Newton` rootfinders, differentiating through fixed-point GARCH-recursion.

Composition: existing reverse-mode AD + a linear solver from `linalg` (LU or CG via Pearlmutter HvP from W13).

Connective tissue: ~230 LOC: IFT-rootfinder wrapper (~100) + IFT-argmin wrapper (~80) + custom-VJP registration plumbing (~50).

### W16. SparseGridSmolyak — high-D quadrature

Capability: Smolyak 1963 sparse-grid quadrature: tensor-product 1D rules pruned to a "diagonal" of the index lattice. Cost is `O(N (log N)^{d-1})` for accuracy O(N^-r) where 1D rule is order r — beats full tensor product (O(N^d)) and beats Monte Carlo (O(N^-0.5)) up to dimension ~20. Cross-link to 176 (synergy QMC).

Composition: builds on existing 1D Gauss-Legendre / Clenshaw-Curtis. Index-set generation is combinatorial (O(d k) Pascal-row).

Connective tissue: ~180 LOC for 1D-rule-as-interface + Smolyak driver.

---

## 4. Connective-tissue patterns (cross-cutting)

### P1. Three-way derivative cross-validation R-MUTUAL pin

Saturates 3/3 R-MUTUAL-CROSS-VALIDATION the way commits 6a55bb4 (audio-onset 3 detector) and 365368a (Clayton autodiff vs analytic) did:

1. `calculus.NumericalDerivative` with optimal h (W3)
2. `complex_step.ComplexStepDerivative` (W2) — agrees to ~1e-13
3. `autodiff.Backward` on the same forward expression — agrees to machine eps

On any smooth test function (e.g. `f(x) = x*sin(x^2) + exp(-x/3)`) all three should agree to 1e-9. **First three-way derivative pin in the foundation.** Test cost: ~30 LOC. Test value: bedrock for every future AD consumer (if your AD says the gradient is X and complex-step says Y, complex-step is right because central-diff's roundoff is documented).

### P2. Quadrature-error-witness pair

For any integral with a known closed form (`∫_0^1 x^k dx = 1/(k+1)`, `∫_{-∞}^∞ e^{-x^2} dx = sqrt(pi)`, `∫_0^1 4/(1+x^2) dx = pi`), pin every quadrature rule (Trap / Simpson / GL-2..5 / GH / GL / CC / Romberg / AdaptiveSimpson / GK15) against the closed form to its documented order. Demonstrates the **convergence rate** matches theory (H -> H/2 -> error ~ /4 for Trap, /16 for Simpson, etc.). ~80 LOC test.

### P3. "Differentiate then integrate vs integrate then differentiate" pin

Given `I(x) = ∫_0^1 sin(x t) dt = (1-cos(x))/x`, pin: (a) `dI/dx` via complex-step on `I(x)` numerical-quadratured, (b) `dI/dx` via Leibniz-rule autodiff (W7) on the integrand, (c) `dI/dx` analytic. All three to 1e-9. Closes the calculus x autodiff loop — operationalises Leibniz rule. ~60 LOC.

---

## 5. Landing order (8 PRs, ~24 engineer-days)

| PR | Primitives | LOC | One-line value |
|---|---|---|---|
| PR-1 | W1 Richardson + W2 ComplexStep + W3 OptimalH + W7 Leibniz | 230 | three-way-derivative-pin saturated; opens next 7 PRs |
| PR-2 | W4 higher-order finite-diff + W5 cumulative trap/simpson | 100 | endpoint-correctness + array-valued integration unblocked |
| PR-3 | W6 AdaptiveSimpson + GaussKronrod | 250 | `IntegrateToTolerance` shipping; closes biggest gap |
| PR-4 | W8 Romberg | 70 | spectral convergence on analytic integrands |
| PR-5 | W9 GaussHermite + W10 GaussLaguerre + GaussChebyshev + ClenshawCurtis | 200 | weight-function quadrature rules |
| PR-6 | W11 ForwardModeAD-Dual | 280 | parallel AD mode; unlocks PR-7 |
| PR-7 | W12 HessianViaAD + W13 PearlmutterHvP | 130 | Newton-CG / trust-region tractable for n>20 |
| PR-8 | W14 ODEAdjoint + W15 IFT + W16 SparseGrid | 660 | differentiable physics + differentiable optimization layers |

Total **~1920 LOC source** + **~830 LOC test** over 24 engineer-days. Saturates **four** R-MUTUAL pins (P1 derivative-3-way + P2 quadrature-rate + P3 leibniz-3-way + W14 RK4-discretize-vs-continuous-adjoint).

If only one PR ships: **PR-1** (230 LOC) because (a) lands first calculus<->autodiff cross-edge at zero architectural cost, (b) saturates the canonical derivative R-MUTUAL pin (mirrors commits 6a55bb4 + 365368a + 1e12e80), (c) ComplexStep is one of the most under-known machine-precision-derivative tools and its absence is a foundation embarrassment, (d) Leibniz-rule integration-with-gradient unblocks every parameterised-integral consumer.

---

## 6. Precision hazards documented up-front

- **W1 Richardson** — divergence pattern (E[h] / E[h/2] grows) signals roundoff dominance; clamp tableau depth at 8 rows.
- **W2 ComplexStep** — requires `f` to be real-analytic AND complex-extendable. Functions with `if x<0` branches break analyticity. Document with a list of accepted ops (mirror of cmplx package surface).
- **W3 OptimalH** — `cbrt(eps) * max(1, |x|)` is heuristic; scales poorly when `f` has third-derivative spikes. Use W1 Richardson as a fallback that **measures** the optimal h empirically.
- **W6 AdaptiveSimpson** — cap recursion depth at 30 (~10^-9 panel size) to prevent infinite recursion on discontinuities; return `(value, error_estimate, converged bool)`.
- **W6 GaussKronrod** — error estimate `|GK15 - G7|` is a heuristic, not a bound. Quadpack uses `200 * |error|^1.5` as a more reliable bound; cite Piessens 1983.
- **W7 Leibniz** — boundary derivatives `a'(x), b'(x)` must be supplied if a, b depend on x; document the sign convention.
- **W8 Romberg** — fails on non-analytic integrands (e.g. `|x|`); fall back to Adaptive Simpson.
- **W9 GaussHermite** — for `∫ f(x) e^{-x^2} dx` with `f` polynomial of degree <= 2N-1 it is **exact**; for general `f` convergence is exponential in N for analytic `f`. Document N=10 covers most use cases.
- **W11 ForwardModeAD** — `Dual` is **not** safe for branching `f`s (max(a,b)) — needs Clarke subgradient handling like reverse-mode does already.
- **W12 HessianViaAD** — symmetry of H is **not** automatic numerically; symmetrize via `(H + H^T) / 2` before factorisation.
- **W13 PearlmutterHvP** — uses two tapes; second tape's memory is O(reverse-tape-size) not O(n).
- **W14 ODEAdjoint** — DTD (autodiff through every RK4 step) memory grows linearly in T/dt and can OOM for stiff problems with small dt. Document the tradeoff against checkpointing (Griewank-Walther 2000).
- **W15 IFT** — requires the Hessian (or Jacobian for rootfinder) at the solution to be **invertible**. Singular Hessians at degenerate optima return NaN; document the regularisation knob (add `lambda * I` per Levenberg-Marquardt).
- **W16 Smolyak** — beats MC up to ~20D; above that the (log N)^{d-1} factor kicks in; cross-link to QMC at d>20.

---

## 7. Cross-language pinning targets

Every Wi has a public-API equivalent in another ecosystem to pin against at <=1e-10 in golden files:

- **W1 Richardson** — Numerical Recipes 3e Section 5.7 (Press et al.) tabulated values.
- **W2 ComplexStep** — Lyness-Moler 1967 + Squire-Trapp 1998 worked examples (any analytic test function).
- **W6 AdaptiveSimpson** — `scipy.integrate.quad` (which is QUADPACK = GK15 adaptive). Pin to 1e-12.
- **W6 GaussKronrod** — QUADPACK reference `dqk15`.
- **W8 Romberg** — `scipy.integrate.romberg` (deprecated in scipy 1.12 but was the reference for years; numpy's `numpy.trapz` plus extrapolation table).
- **W9 GaussHermite** — `numpy.polynomial.hermite_e.hermegauss(n)` for Hermite-E (probabilists' weight `e^{-x^2/2}`); `hermite.hermgauss` for physicists' `e^{-x^2}`. Pin to <=1e-13 on nodes/weights for n<=20.
- **W10 GaussLaguerre** — `numpy.polynomial.laguerre.laggauss`. <=1e-13.
- **W11 ForwardMode** — JAX `jax.jvp` reference, PyTorch `torch.autograd.functional.jvp`, Julia `ForwardDiff.jl::derivative`.
- **W13 PearlmutterHvP** — JAX `jax.hvp`, PyTorch `torch.autograd.functional.hvp`.
- **W14 ODEAdjoint** — `torchdiffeq.odeint_adjoint` (Chen et al. 2018), Julia `DiffEqFlux.jl`. Pin against the adjoint sensitivity of `dy/dt = -y, y(0)=1, theta=1` whose `dy(T)/dtheta` has a closed form.
- **W15 IFT** — JAX `jax.custom_vjp` + `jaxopt.implicit_diff` (Blondel et al. 2022). Pin against the IFT-derivative of `argmin_y (y-x)^2 + y^4` solved analytically.
- **W16 Smolyak** — `scipy.integrate.qmc` (Sobol' QMC) for high-D comparison; `Tasmanian` (UQ Toolbox) for sparse-grid reference values.

---

## 8. Differentiation from related agents

- **027-calculus-missing** (per-package isolation): flagged adaptive-quadrature/Romberg/Gauss-Hermite as gaps. **THIS** review converts those gaps into scoped composition tasks W6/W8/W9 and adds the calculus×autodiff axis (W7/W12-W15) that 027 could not see.
- **159-synergy-prob-autodiff** (if exists): orthogonal axis — autodiff over likelihoods/score functions; THIS review is autodiff over **integrals and ODEs**.
- **160-ish synergy-optim-autodiff** (if exists): cross-link at W13 PearlmutterHvP (Newton-CG glue) and W15 IFT (argmin-differentiation); shared substrate but distinct primitives.
- **176-synergy-prob-mc-qmc** (if exists): cross-link at W16 SparseGrid (Smolyak vs Sobol vs MC); THIS review owns the deterministic-quadrature axis, 176 owns randomised quadrature.
- **per-package autodiff isolation reviews** (~030-035 range): ship missing forward-mode + Hessian primitives as standalone autodiff features. **THIS** review adds the **calculus-side** complement (complex-step as an alternative to AD; Leibniz / adjoint as compositions).

---

## 9. Recommended placement (file layout)

Extends `calculus/` and `autodiff/` in-place with new files; **no new packages**:

```
calculus/
  calculus.go              (existing)
  decision.go              (NEW, ~60 LOC) — the when-to-use-what doc table from §2
  richardson.go            (NEW, ~70 LOC) — W1 + W8 Romberg
  complex_step.go          (NEW, ~50 LOC) — W2 + W3 OptimalH
  finite_diff.go           (NEW, ~60 LOC) — W4 higher-order finite-diff
  cumulative.go            (NEW, ~80 LOC) — W5 cumulative trap/simpson
  adaptive.go              (NEW, ~280 LOC) — W6 AdaptiveSimpson + GaussKronrod
  gauss_special.go         (NEW, ~220 LOC) — W9 Hermite + W10 Laguerre + Chebyshev + Clenshaw-Curtis
  smolyak.go               (NEW, ~200 LOC) — W16 SparseGridSmolyak
  leibniz.go               (NEW, ~100 LOC) — W7 IntegrateWithGradient — first calculus->autodiff edge
autodiff/
  doc.go                   (existing — append §When-to-use-forward-vs-reverse table)
  ops.go                   (existing)
  tape.go                  (existing)
  vector.go                (existing)
  forward.go               (NEW, ~280 LOC) — W11 Dual + ForwardJacobian
  hessian.go               (NEW, ~100 LOC) — W12 + W13 Pearlmutter HvP
  ode_adjoint.go           (NEW, ~250 LOC) — W14 (imports chaos? no — keep it generic, accept a step function)
  ift.go                   (NEW, ~230 LOC) — W15 ImplicitFunctionTheoremDiff
```

**Cycle-free DAG:** `calculus/leibniz.go` is the only calculus -> autodiff edge (consumes `*autodiff.Variable`). `autodiff/ode_adjoint.go` does NOT import chaos; it accepts a generic step function so callers wire `chaos.RK4` from outside. Zero cycles.

---

## 10. Single-day high-leverage commit

If only one PR ships in the morning: **PR-1 = W1 + W2 + W3 + W7 = 230 LOC source + ~120 LOC tests** because:

1. Lands the first calculus<->autodiff cross-edge (`leibniz.go`) at zero architectural cost.
2. Saturates the canonical three-way derivative R-MUTUAL pin (central-diff vs complex-step vs autodiff) to 1e-12 — a foundation-level bedrock test mirroring commits 6a55bb4 (audio-onset 3-detector), 365368a (Clayton autodiff), and 1e12e80 (token-set-ratio RapidFuzz parity).
3. ComplexStep is the single most under-known machine-precision derivative trick; absence is a foundation embarrassment and presence is a five-minute upsell to every future AD consumer.
4. Leibniz-rule integration-with-gradient unblocks the four real autodiff consumers (timeseries/garch, infogeo, prob/copula, queued Heston/SABR) the moment any of them needs `d(loss)/d(theta)` of an integrated quantity.
5. RichardsonExtrapolation gives the only currently-shippable answer to "is my numerical derivative trustworthy" — empirical-h-level convergence check beats heuristic h=1e-5.
6. The optimal-h function (W3) closes a paper cut documented in `calculus.go:31` (the comment recommends `cbrt(eps)*max(1,|x|)≈1e-5` but never exposes it as a callable) — a one-line user-facing improvement.

This synergy is **medium-leverage relative to 180 (physics×prob)** because both packages are tiny (5 + 16 primitives) so absolute LOC is modest, but is **architecturally pivotal**: every future second-order optimization, every differentiable-physics simulation, and every parameterised-integral price/likelihood passes through this calculus×autodiff axis. Of the 16 W-primitives, 11 (W2, W6, W7, W11-W16, W9, W10) have public-API equivalents in JAX/PyTorch/SciPy/Julia pinning every Wi at 1e-10 vs reference per CLAUDE.md §"Golden files are the proof".

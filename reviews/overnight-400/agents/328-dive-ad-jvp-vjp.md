# 328 — dive-ad-jvp-vjp (Forward / Reverse / Mixed / Hessian / Jacobian audit)

## Headline
`autodiff` is reverse-mode-only (taped VJP); no forward-mode JVP, no Hessian, no Jacobian
helper, no checkpointing — a clean MVP that needs a forward-mode dual + a Hessian
forward-on-reverse to round out the tradeoff matrix.

## Findings (existing audit)

### What exists

- `autodiff/doc.go:1-99` documents the package as **reverse-mode only** (Wengert tape +
  pullback closures). Forward-mode is not mentioned outside the phrase
  "asymptotic advantage of reverse-mode over forward-mode" (`doc.go:10-12`).
  Hessian-vector products and checkpointing are explicitly listed as **deferred
  to v2** (`doc.go:73-74`).
- `autodiff/tape.go:10-15` — `Tape` is a flat `[]node`. Each `node` holds a forward
  value and a `pullback func(grad float64, gradients []float64)` closure
  (`tape.go:17-20`).
- `autodiff/tape.go:68-83` — `Backward` walks `t.nodes` in reverse, seeds
  `grads[out.ID] = 1.0`, calls each non-nil pullback once. Returns the full gradient
  vector indexed by Variable id. **Single-shot**: doc-comment warns calling Backward
  twice produces undefined gradients (`tape.go:65-67`) — pullbacks accumulate, no
  reset.
- `autodiff/ops.go:1-142` — 12 elementary primitives implemented as
  operator-overloading-flavoured Go functions (since Go has no operator overloading,
  these are free functions taking `*Variable`):
  - Arithmetic: `Add`, `Sub`, `Mul`, `Div`, `Neg`, `AddConst`, `MulConst`
    (`ops.go:6-69`).
  - Transcendentals: `Exp`, `Log`, `Sqrt`, `Pow(a, p)` (constant exponent),
    `Sin`, `Cos`, `Tanh` (`ops.go:75-141`).
  - **Pullbacks correct.** `Mul` captures both `aVal, bVal` by value
    (`ops.go:29`), so the closure does not chase mutated state. Same pattern in
    `Div`, `Pow`, `Log`, `Sin`, `Cos`.
- `autodiff/vector.go:1-99` — vector-level fused ops:
  - `Sum` (`vector.go:13-32`), `Dot` (`vector.go:37-66`),
    `MeanSquaredError` (`vector.go:71-98`).
  - These pre-allocate capture slices (`ids`, `aVals`, `bVals`, `resid`) at register
    time so the pullback itself does not allocate — consistent with CLAUDE.md
    "no allocations in hot paths".
  - Missing fused: vector-add (`xs + ys`), scalar-vector multiply, broadcast,
    sigmoid, softmax, log-sum-exp, dot-self/L2-norm-squared, cross-entropy. Doc
    claims "4 vector ops" (`doc.go:69`) — only 3 ship; vector-add and scalar-vector-mul
    are advertised but absent. Dead promise.

### What is missing (vs the autodiff design space)

- **No forward-mode (JVP).** No `Dual` number, no tangent propagation, no
  `Jvp(f, primals, tangents)` API. For a function `f : R^n → R^m` with
  m >> n, reverse-mode is wasteful (m sweeps); forward-mode would cost n sweeps.
  All current consumers (`garch`, `infogeo`, `copula`) are scalar-output
  log-likelihoods (m=1), which is exactly reverse-mode's sweet spot — but the
  optim package and any future Jacobian consumer (e.g. residuals in
  Levenberg-Marquardt, sensitivities in `control`) will want forward.
- **No Hessian helper.** `prob/copula/autodiff_test.go` and the GARCH calibration
  doc-comment hint at Newton-CG, but Newton-CG needs **Hessian-vector products**
  `H·v`, classically computed via Pearlmutter's "fast exact multiplication by
  the Hessian" trick = forward-mode-on-reverse-mode. Not implemented.
- **No Jacobian helper.** `Jacobian(f, x)` for f: R^n → R^m would require either
  m reverse sweeps (costly when m large) or n forward sweeps. Neither path
  available.
- **No checkpointing.** Tape grows monotonically; for a 10^6-step GARCH MLE the
  tape is a 10^6-element slice. Griewank-Walther's `revolve` (binomial
  checkpointing) gives O(log n) memory at O(log n) recompute cost. Deferred.
- **No reverse-on-reverse (higher-order via taped Backward).** Reverse-mode of
  reverse-mode requires taping the pullback computation itself. Not supported —
  pullbacks are opaque Go closures, not autodiff-traceable. This rules out
  third-derivative tensors via pure reverse-on-reverse.
- **No control-flow taping caveats.** Tape records what the forward branch took;
  if the forward branch is data-dependent (e.g. Heaviside, ReLU at 0), the
  gradient at the kink is silently the one-sided derivative. No subgradient
  story.
- **`Backward` is non-reentrant.** Comment at `tape.go:65-67` says calling twice
  is undefined. In practice this means consumers who want both ∇f and ∇²f·v on
  the same tape must rebuild from scratch — fine for one-shot, costly inside an
  optimiser inner loop.
- **No `Reset()` on `Tape`.** A new tape per gradient evaluation means
  `make([]float64, n)` for the grads slice on every call. The end-to-end linreg
  test (`autodiff_test.go:285-298`) builds 1000 tapes — explicit demonstration of
  the per-iter alloc.

### Numerics / edge cases

- `Log(0)` returns `-Inf`, pullback divides by `aVal` → NaN/Inf in grads. Doc
  says "Caller must ensure a.Val > 0" (`ops.go:84-85`) — punted to caller.
- `Sqrt(0)`: `v = 0`, pullback computes `g / (2*0) = ±Inf` (`ops.go:99-101`).
  Same caller-beware.
- `Div(_, 0)`: forward returns `±Inf`, pullback `g / 0 = ±Inf` and
  `-g*a / 0 = ±Inf`. Same.
- `Pow(0, p)` with `p < 1`: pullback contains `Pow(0, p-1.0) = +Inf`,
  multiplied by `p` gives Inf gradient. Same caller-beware.
- **No NaN propagation tests.** Unlike most reality packages, no IEEE-754
  edge-case golden file. CLAUDE.md mandates this. (Cf.
  `reviews/overnight-400/agents/011-autodiff-numerics.md`.)
- Cross-tape combinator panics correctly enforced (`tape.go:86-90`,
  `vector.go:21-23, 51-53, 83-85`) — good.

### Thread-safety

- Tape construction is **not** safe for concurrent goroutines (doc says so,
  `tape.go:8-9`). Backward is read-mostly but mutates `grads`; one Backward per
  Tape is the contract.

### Allocation profile

- Per-op allocation: one `node` (heap-escaping due to closure), one closure
  capturing scalar floats. For 12 ops × N forward steps, that's ~24N heap
  allocations per gradient. The "no allocations in hot paths" CLAUDE.md rule
  applies to the *pullback* execution (which does not allocate after the
  vector-fused ops pre-capture their slices), not tape construction. Acceptable
  for outer loops, painful for tight Newton-CG.

## Concrete recommendations

1. **Add forward-mode JVP via `Dual` numbers** in `autodiff/forward.go`. A
   `Dual struct { Val, Tan float64 }` with `Add/Mul/.../Sin/Cos` methods costs
   ~150 LoC, zero alloc, no tape. Top-level API:
   `Jvp(f func([]Dual) Dual, primals, tangents []float64) (val, tan float64)`.
   This is **the** missing half of the autodiff package.

2. **Add `Hessian(f, x)` and `HessianVectorProduct(f, x, v)`** in
   `autodiff/hessian.go`, computed forward-on-reverse: seed each input's tangent,
   build a tape over Dual-valued forward, take Backward to harvest a row of H.
   Pearlmutter 1994. R-pattern target: forward-on-reverse Hessian ≡
   reverse-on-forward Hessian to 1e-9 (saturates a fresh
   R-MUTUAL-CROSS-VALIDATION 3/3 once three functions pin both paths).

3. **Add `Jacobian(f, x)` helper** in `autodiff/jacobian.go` that picks the
   cheaper mode based on shape: m forward sweeps if m ≤ n, n reverse sweeps
   otherwise. Single function, ~40 LoC, dispatches to JVP/VJP.

4. **Promise a `Tape.Reset()`** that retains capacity:
   `t.nodes = t.nodes[:0]; for i := range t.nodes { t.nodes[i] = node{} }`.
   Saves one `make([]node, ...)` per Newton step in the linreg-style outer loop
   on `autodiff_test.go:285-298`. Trivial win.

5. **Replace per-op closure capture with explicit op tag + scratch slot.**
   Instead of `pullback func(...)` (heap-allocates a Go closure with captured
   floats), use `tag uint8` + `aux [3]float64` and a switch in Backward. Eliminates
   one heap alloc per forward op. Material for million-step tapes (GARCH MLE).

6. **Wire `optim.GradientDescent` and `optim.LBFGS` to autodiff.** Today
   `optim/gradient.go:30, 81` accept `grad func([]float64, []float64)` — callers
   write the chain rule by hand. Add `optim.WithAutodiff(f func([]autodiff.Variable) *autodiff.Variable)`
   adapter that builds tape, calls Backward, copies into the `g` buffer. Closes
   the loop the doc-comment lists (`autodiff/doc.go:17-25`).

7. **Add log-sum-exp and softmax fused vector ops** (`vector.go`). These appear
   in every ML loss and have a numerically stable closed form
   `lse(x) = m + log(Σ exp(xᵢ - m))` whose pullback is `softmax(x)`. Without a
   fused op, the user composes Exp/Sum/Log and loses the stability + overpays in
   tape entries. Direct demand from `infogeo/autodiff_test.go`'s
   `KL(p || softmax(θ))` consumer (`doc.go:45-50`).

8. **Add an IEEE-754 edge-case golden file** per CLAUDE.md
   ("IEEE 754 edge cases mandatory: +Inf, -Inf, NaN, -0.0, subnormals"). Pin
   gradient behaviour at boundary inputs of every transcendental. Currently zero
   such tests.

9. **Dual-mode cross-validation pin** (R-MUTUAL-CROSS-VALIDATION 3/3): once
   forward mode lands, pin `JVP(f, x, eᵢ) ≡ VJP(f, x, [1])[i]` for three
   functions (analytic + composition + transcendental) at 1e-12. Plus the
   classical `central_finite_diff(f, x, h=1e-6) ≈ autodiff(f, x)` to 1e-7
   regression (already in shape — `autodiff_test.go:155-209` does this for
   reverse mode; extend to forward).

10. **Document the data-dependent-control-flow pitfall.** Tape records the
    branch taken at the forward sample point; differentiating through `if x>0`
    silently returns the one-sided derivative at x=0. Add a paragraph in
    `doc.go` and a test that demonstrates Heaviside / ReLU subgradient choice.

## Day-1 PR (cheapest)

`autodiff/forward.go` adding the `Dual` type + `Jvp` driver + 12 method
receivers mirroring `ops.go`. ~150 LoC, no test infrastructure churn, unlocks
recommendations 2/3/9. Single PR review surface.

## Cross-links to consumers

- `optim/gradient.go:30, 81` (`GradientDescent`, `LBFGS`) — accept hand-rolled
  `grad` callbacks today; the autodiff adapter (rec. 6) is the cleanest
  replacement.
- `prob/copula/autodiff_test.go` — Clayton log-PDF gradient pin
  (`doc.go:51-58`). Demands stable `log` and `pow` already shipped; would also
  benefit from log-sum-exp (rec. 7) for elliptical / Archimax extensions.
- `infogeo/autodiff_test.go` — KL-divergence gradient pin (`doc.go:45-50`).
  Direct customer for fused softmax / log-sum-exp (rec. 7).
- `timeseries/garch/autodiff_test.go` — GARCH(1,1) NLL gradient pin
  (`doc.go:37-44`). For the Newton-CG path called out in `doc.go:19-21`,
  Hessian-vector products (rec. 2) are mandatory.
- `control/` — sensitivities ∂y/∂θ over a transfer-function sweep are exactly
  Jacobian (rec. 3); today computed via finite differences in `control/bode.go`
  (out of scope here).
- Pistachio (downstream consumer per CLAUDE.md depgraph) — differentiable
  rendering wants forward-mode for many-output / few-input pixel shaders.

## Sources

### Repo

- `autodiff/doc.go` (package overview, MVP scope, deferred-to-v2 list)
- `autodiff/tape.go` (Tape, Variable, Backward)
- `autodiff/ops.go` (12 elementary ops + pullbacks)
- `autodiff/vector.go` (Sum, Dot, MSE)
- `autodiff/autodiff_test.go` (18 base tests, FD validation, end-to-end linreg)
- `autodiff/autodiff_expansion_test.go` (15 expansion tests, panics, leaf grad)
- `optim/gradient.go` (GradientDescent, LBFGS — current grad-callback contract)
- `prob/copula/autodiff_test.go`, `infogeo/autodiff_test.go`,
  `timeseries/garch/autodiff_test.go` (3 cross-package consumers)
- Earlier overnight reviews: `reviews/overnight-400/agents/011-autodiff-numerics.md`,
  `012-autodiff-missing.md`, `013-autodiff-sota.md`, `014-autodiff-api.md`,
  `015-autodiff-perf.md` — slot 015 has tape-memory perf analysis;
  `163-synergy-optim-autodiff.md`, `168-synergy-physics-autodiff.md`,
  `183-synergy-calculus-autodiff.md`, `185-synergy-signal-autodiff.md` map
  cross-package opportunities.

### External

- Wengert R. E. (1964). *A simple automatic derivative evaluation program.*
  CACM 7(8):463-464. — original tape-based reverse-mode.
- Linnainmaa S. (1970). *The representation of the cumulative rounding error of
  an algorithm as a Taylor expansion of the local rounding errors.* — first
  modern reverse-mode account.
- Griewank A. & Walther A. (2008). *Evaluating Derivatives: Principles and
  Techniques of Algorithmic Differentiation*, 2nd ed., SIAM. — canonical
  reference; chapters on reverse mode, forward mode, mixed mode, and `revolve`
  checkpointing.
- Pearlmutter B. A. (1994). *Fast exact multiplication by the Hessian.* Neural
  Computation 6(1):147-160. — forward-on-reverse trick for H·v.
- Baydin A. G., Pearlmutter B. A., Radul A. A., Siskind J. M. (2018).
  *Automatic Differentiation in Machine Learning: a Survey.* JMLR 18(153):1-43.
  — modern survey; clarifies operator-overloading vs source-rewriting vs taping.
- JAX (`grad`, `jvp`, `vjp`, `jacfwd`, `jacrev`, `hessian`) — reference API
  surface for what a complete autodiff exposes.
- PyTorch autograd — dynamic-graph operator-overloading reverse mode; same
  shape as reality's tape.

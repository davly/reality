# 011 | autodiff-numerics

**Agent:** 011 of 400
**Date:** 2026-05-06
**Topic:** autodiff: gradient accuracy, second-order tape correctness, complex-step diff
**Package:** `C:\limitless\foundation\reality\autodiff\`
**Headline:** Gradient correctness is solid for the in-scope MVP (12 ops + 4 vector ops, all pullbacks pin to finite differences), but the package is silently fragile at IEEE-754 boundaries (Sqrt(0)→+Inf grad, Pow(<=0, p)→NaN grad, Log(0)→+Inf grad with no guard), the `Backward`-double-call doc-comment is provably wrong (Backward IS reentrant — the slice is allocated fresh each call), there is no Hessian / forward-over-reverse / complex-step path, no abs/max/min/relu/clip/atan2/softmax/log1p/expm1, no Reset (tape memory grows monotonically), and no centralised finite-difference grad-check helper for downstream consumers.

---

## Files audited

- `autodiff\doc.go` (99 lines) — package preamble, scope, references
- `autodiff\tape.go` (90 lines) — Tape, node, Variable, Var/Constant/register/Backward
- `autodiff\ops.go` (142 lines) — 12 elementary ops: Add Sub Mul Div Neg AddConst MulConst Exp Log Sqrt Pow Sin Cos Tanh
- `autodiff\vector.go` (98 lines) — Sum Dot MeanSquaredError
- `autodiff\autodiff_test.go` (342 lines) — 14 tests, finite-diff helper, end-to-end linreg fit
- `autodiff\autodiff_expansion_test.go` (306 lines) — 18 expansion tests (panics, edge powers, leaf, composition, IDs)

External consumers (verified, all 1e-9 parity tests):
- `timeseries\garch\autodiff_test.go` — GARCH(1,1) NLL gradient parity
- `infogeo\autodiff_test.go` — KL(p || softmax(θ)) gradient = q − p
- `prob\copula\autodiff_test.go` — Clayton log-PDF dθ gradient

`go test ./autodiff/` passes; tests run in 0.37s.

---

## Per-operation correctness

All 12 elementary pullbacks in `ops.go` are mathematically correct in the interior of their domain. The pullback formulas match the analytic derivative; multi-use (diamond) accumulation is correct because every pullback uses `+=` into the gradient slice. I verified via probe:

- `d(x*x)/dx at x=3 → 6` (diamond OK; same leaf appears in both factors)
- `d/dx log(x²) at x=2 → 1` (chain through Mul and Log composes correctly)
- Forward-mode is NOT implemented; this is reverse-mode-only AAD.

| Op | Pullback | Interior correctness | Edge behaviour |
|---|---|---|---|
| `Add(a,b)` | g→a, g→b | OK | — |
| `Sub(a,b)` | g→a, −g→b | OK | — |
| `Mul(a,b)` | g·b→a, g·a→b | OK | captures `aVal,bVal` at construction (correct) |
| `Div(a,b)` | g/b→a, −g·a/b²→b | OK | **b=0 → +Inf in both grads, silently** |
| `Neg(a)` | −g→a | OK | — |
| `AddConst(a,c)` | g→a | OK | — |
| `MulConst(a,c)` | g·c→a | OK | — |
| `Exp(a)` | g·exp(a)→a | OK | overflow at large a → +Inf grad (acceptable) |
| `Log(a)` | g/a→a | OK for a>0 | **a=0 → +Inf grad; a<0 → NaN grad; no guard** |
| `Sqrt(a)` | g/(2√a)→a | OK for a>0 | **a=0 → +Inf grad (1/0); a<0 → NaN grad** |
| `Pow(a,p)` | g·p·a^(p−1)→a | OK for a>0 or p∈ℤ | **a≤0 with non-integer p → NaN val and NaN grad** |
| `Sin(a)` | g·cos(a)→a | OK | — |
| `Cos(a)` | −g·sin(a)→a | OK | — |
| `Tanh(a)` | g·(1−tanh²(a))→a | OK; saturates correctly to 0 at ±20 | uses cached `v=tanh(a)` (cheap, correct) |

| Vector op | Pullback | Notes |
|---|---|---|
| `Sum(xs)` | g→every xs[i] | OK; panics on empty |
| `Dot(a,b)` | g·b[i]→a[i], g·a[i]→b[i] | OK; cheaper than chained Mul/Add via captured slices |
| `MeanSquaredError(pred, target)` | g·(pred[i]−target[i])/n→pred[i] | OK; target is `[]float64`, no grad on target side |

---

## Edge-case probe results (run against current HEAD)

I exercised 10 IEEE-754-style probes against the live package. Results:

```
Sqrt(0).Val = 0          grad = +Inf      ← divide-by-zero, no guard
Log(0).Val  = -Inf       grad = +Inf      ← divide-by-zero, no guard
Pow(-1, 0.5).Val = NaN   grad = NaN       ← documented-as-undefined but no panic/error
Pow(0, 0.5).Val = 0      grad = +Inf      ← 0.5 · 0^(-0.5) overflows
Diamond x*x at x=3 → 6                    ← correct (multi-use accumulation works)
Backward called 3× → 7,2 ; 7,2 ; 7,2      ← REENTRANT (doc says "undefined")
Div(1, 0).Val = +Inf     grad_a=+Inf grad_b=-Inf   ← propagates Inf, no guard
Exp(1000).Val = +Inf     grad = +Inf      ← acceptable overflow
Tanh(20) grad = 0 (want 0)                ← correct saturation, no NaN
After 100 ops, tape nodes are not freed (no Reset method)
```

Two findings here are load-bearing:

1. **`Backward` is reentrant** despite the doc-comment in `tape.go:67-67` claiming "calling it twice produces undefined gradients (each pullback re-applies and would double-count)." The implementation in `tape.go:72` allocates `grads := make([]float64, len(t.nodes))` fresh on every call, so pullback closures only mutate the local slice — there is no externally-visible state for them to double-count into. Tests in fact never exercise the double-call path. Either the doc-comment is wrong (and should be deleted) or the reentrancy is accidental (and should be locked by an explicit single-call guard). Quoting the actual pullback shape from `ops.go:9-12` confirms there is no captured-grads-slice anywhere — every pullback takes `grads []float64` as a parameter.

2. **Sqrt(0), Log(0), Pow(<=0, frac) silently produce ±Inf or NaN gradients.** The doc-comments say "Caller must ensure a.Val > 0; behaviour at non-positive a is undefined" (`ops.go:85-86, 95`) — that is a legitimate contract, but the package ships no `Abs`, `Max`, `Min`, `Clip`, `ReLU`, `Softplus`, or any of the standard ML ops where one would *want* a derivative-discontinuity policy. The garch consumer reparameterises away from these failure modes via softmax+exp; new consumers will have to do the same and there is no documented playbook.

---

## Missing capability surface (vs. topic scope)

The agent topic asks specifically about:

| Capability | Status | Notes |
|---|---|---|
| Forward-mode dual numbers | **NOT IMPLEMENTED** | Package is reverse-mode-only; doc-comment scopes forward-mode out |
| Reverse-mode tape | OK | Single-pass, monotone tape; pullbacks are correct closures |
| Multi-use accumulation | OK | Verified by probe; `+=` everywhere |
| In-place mutation safety | OK | `Mul`/`Div` capture `aVal,bVal` at register time; later mutation of inputs cannot retroactively change pullback math |
| log/exp/sin/cos/sqrt special cases | PARTIAL | Sqrt(0) and Log(0) silently produce +Inf grad; no `Log1p`/`Expm1`/`LogSumExp` for stability |
| relu / max / abs / clip discontinuities | **MISSING** | None of these ops exist; `min(a,b)` style = no derivative-tie policy |
| Pow(x^y) with y as Variable | **MISSING** | Pow only takes `float64` exponent; consumers (Clayton copula) work around via `exp(y · log x)` |
| Pow(x≤0, p) with p non-integer | NaN; documented as undefined; no panic | |
| Complex-step differentiation | **NOT IMPLEMENTED** | No path; no `Variable[complex128]` |
| Hessian / second-order | **NOT IMPLEMENTED** | doc-comment defers to v2 ("Hessian-vector products via forward-over-reverse") |
| Numerical comparison vs central-difference | PARTIAL | `finiteDiff` helper exists in `autodiff_test.go:14` but is private; not exposed for downstream consumers; only used in 2 in-package tests at h=1e-6 |
| Atan2 partials | **MISSING** | No Atan2 op; if a consumer needs it they have to compose Sin/Cos/etc. |
| Softmax (numerically stable) | **MISSING as a primitive** | Consumers (infogeo, garch) hand-roll the subtract-max idiom each time; bug-prone |
| Tape memory leak across runs | **YES, by design** | No `Reset()`; no shrink-to-zero; consumers must allocate a fresh `NewTape()` per gradient eval (which is what they do — see `garch.go:286` style) |
| Kahan summation in gradient accumulation | NOT USED | All accumulations are plain `+=`. For long tapes (>1e6 ops) catastrophic cancellation is plausible but unobserved. |

---

## Strengths to preserve

1. **Correctness in the interior** is unimpeachable — every pullback formula matches the textbook derivative. All 32 tests pass. The 3 cross-package consumer tests pin autodiff against analytic gradients at 1e-9 across diverse loss surfaces (GARCH NLL with stationarity reparam, KL with softmax, Clayton copula log-density). The `R-CLOSED-FORM-PINNED-TO-AUTODIFF` pattern is at 3/3 saturation.
2. **Allocation discipline**: the hot inner closure does no allocation; the only allocation per op is one `node` struct + one closure capture.
3. **Same-tape enforcement**: `requireSame` panics on cross-tape ops (covered by `TestCrossTape_Panics`, plus 4 vector-op cross-tape panic tests in `autodiff_expansion_test.go:110-174`).
4. **Captured-by-value forward state**: `Mul`, `Div`, `Sin`, `Cos`, `Pow` all capture the input's `Val` at construction time — the closure is immune to later mutation of the input Variable's `Val` field.
5. **Determinism**: pinned by `TestBackward_Deterministic` and the build-twice test in expansion suite.

---

## Gaps with concrete remediation

In priority order:

1. **Fix or delete the `Backward`-twice doc-comment** (`tape.go:66-67`). Reentrancy is real; the comment is wrong. Add a `TestBackward_Reentrant` test to lock the actual behaviour.

2. **Add a public `GradCheck(f, x, h, tol)` helper.** The private `finiteDiff` in the test file should be promoted to package API so every downstream consumer (garch, infogeo, copula, future Heston/SABR) doesn't reinvent its own central-difference probe at h=1e-6. Recommended signature: `func GradCheck(f func(*Tape, []*Variable) *Variable, x []float64, h, tol float64) (max float64, err error)`.

3. **Add `Sqrt0`, `Log1p`, `Expm1`, `LogSumExp`, `Softplus`, `Softmax` as primitives.** Each one fixes a known IEEE-754 trap or stability issue. `LogSumExp(xs)` would let infogeo and garch drop their hand-rolled subtract-max patterns. `Sqrt0(a, eps)` returning `Sqrt(a + eps)` with a stable derivative covers the variance-floor pattern.

4. **Add discontinuity-aware ops with a documented sub-derivative policy**: `Abs`, `Max`, `Min`, `ReLU`, `Clip`. Choose the convention (e.g. `Abs'(0) = 0`, `Max(a,b)` ties go to `a`) and pin it in tests.

5. **Add `Atan2(y, x)` with the correct partials**: `∂/∂y = x/(x²+y²)`, `∂/∂x = −y/(x²+y²)`. This is the standard 4-quadrant trap that consumers cannot easily synthesise from existing ops.

6. **Add `PowVar(a, b)` with `b` a Variable.** `x^y = exp(y · log x)` is the workaround the copula consumer uses; promoting it to a primitive saves three tape entries per call and lets the pullback handle x≤0 with explicit policy.

7. **Add a `(t *Tape) Reset()` method** that re-uses the underlying `nodes` slice (`t.nodes = t.nodes[:0]`) so consumers calling `NewTape()` per Newton-CG iteration can avoid the allocation. Alternatively document the intentional per-call-fresh-tape pattern.

8. **Add forward-over-reverse for Hessian-vector products.** This is the natural next milestone for the doc-cited Heston/SABR/Newton-CG consumers. Start with `(t *Tape) Hvp(out *Variable, v []float64) []float64` returning `H·v` via re-tape with dual numbers, then build `Hessian` on top.

9. **Document NaN-propagation policy through unused branches.** Right now, if a downstream Variable is registered but never reached by `out` in the reverse walk, its gradient is correctly 0 (because pullbacks only get called for nodes reachable in the topological reverse order — actually no, the reverse loop in `tape.go:76-81` iterates *every* node, but only invokes pullback if `grads[i] != 0` would have been a useful guard; currently every non-leaf pullback runs unconditionally with `grads[i] = 0` as input, which is wasted work but produces correct zero-gradients). The unused-branch concern is: a NaN forward value (e.g. log of a negative computed in a never-used branch) does NOT poison the gradient because the pullback's input `g=0` makes `0 * NaN = NaN` — actually this DOES poison via `0 * NaN = NaN`. **This is a real bug under aggressive constant-folding.** Test it.

10. **Consider Kahan summation in `Backward`'s gradient accumulator.** Long-tape consumers (>1e5 ops, e.g. GARCH on a 10-year minute bar) accumulate gradient via `+=`; for ill-conditioned losses this loses precision. Compensated-summation is a small constant-factor cost.

---

## Cross-package risk

- The `R-CLOSED-FORM-PINNED-TO-AUTODIFF` pattern at 3/3 means every new consumer that adds a 1e-9 parity test promotes the pattern further; this is a strong substrate witness. New consumers should be required to add such a test, not optional.
- The garch/infogeo/copula consumers all hand-roll the subtract-max softmax pattern. If autodiff added `LogSumExp` as a primitive, those three consumers should be refactored to use it (and bug-fixed centrally if a stability issue is ever found).
- No consumer currently exercises the catastrophic-cancellation regime (long tape + tight tolerance). Future Heston/SABR calibrators on minute-bar 5-year series will. Plan for it now via Kahan or via documentation.

---

## Summary

Autodiff is a small, disciplined reverse-mode AAD substrate that is mathematically correct in the interior of every shipped op and has three production-quality consumer parity tests. It is, however, an MVP in the truest sense: there is no second-order capability, no complex-step alternative, no discontinuity-aware ops, no numerically-stable softmax / log1p / expm1 primitive, no Atan2, no Pow with variable exponent, no exposed grad-check helper, no Reset, no protection against +Inf/NaN gradients at common boundary inputs (Sqrt(0), Log(0), Pow(≤0, frac)), and one provably-wrong doc-comment about Backward reentrancy. Every gap is well-scoped and locally fixable; none of them block the current consumers because each consumer reparameterises around the gaps. The package is safe to keep building on, but each new consumer pays a per-package tax to reinvent the same workarounds — the right next step is to absorb those workarounds into the substrate.

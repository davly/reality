# 014 — autodiff: API ergonomics — Var/Grad/Tape naming, broadcasting, mutability

**Agent:** 014 of 400
**Date:** 2026-05-07
**Topic:** API ergonomics review of `C:\limitless\foundation\reality\autodiff\` (per MASTER_PLAN.md)
**Files audited:** `autodiff/doc.go`, `autodiff/tape.go`, `autodiff/ops.go`, `autodiff/vector.go`, `autodiff/autodiff_test.go`, `autodiff/autodiff_expansion_test.go`, plus three consumer call-sites: `timeseries/garch/autodiff_test.go:110-186`, `infogeo/autodiff_test.go:110-176`, `prob/copula/autodiff_test.go:123-156`. Sibling style sampled: `optim/gradient.go:1-80`, `linalg/matrix.go:1-80`. Predecessors 011/012/013 inform but are not duplicated here.

## Headline

reality/autodiff is the *function-style scalar reverse-mode tape with no methods on Variable* — the whole API is `pkg.Op(a, b *Variable) *Variable` plus `tape.Var(v)` / `tape.Backward(out)`, which forces every consumer into deeply nested, comment-only-readable expressions like `autodiff.Add(autodiff.Add(autodiff.Mul(a,b), autodiff.Sin(c)), autodiff.MulConst(t, k))`; the three live consumers (garch, infogeo, copula) all reinvent the same workarounds (manual softmax, `x^θ = exp(θ·log x)`, per-iteration `NewTape()`+full re-registration, hand-pulling gradients via `g[v.ID]`); the `Variable.Tape *Tape` back-pointer + integer ID + `[]float64`-indexed-by-ID gradient return locks in a design where users must cache the IDs of "interesting" leaves before forgetting the rest, the tape cannot be reset, gradients cannot be returned by name, broadcasting and shapes do not exist (no `Vector` or `Matrix` node — `Dot`/`Sum` are scalar-output reductions), error reporting is panic-only (six panic sites, zero error returns, no sentinel `var ErrCrossTape = ...`), Backward is effectively single-shot per tape, and there is no flagship runnable example in `doc.go` — the only end-to-end demonstration lives in `TestEndToEnd_LinearRegressionByGradientDescent` (autodiff_test.go:279-305) which is excellent but not surfaced. The fix-set is small and bounded: add `(*Variable).Add/.Mul/.Sub/.Div/.Neg` methods that wrap the package functions (Go-idiomatic — see `geometry.Quaternion`), introduce `tape.Grad(out, vars ...*Variable) []float64` so consumers stop indexing by `.ID`, add `tape.Reset()` for inner-loop reuse, switch the cross-tape / shape-mismatch panics to typed `ErrCrossTape` / `ErrShapeMismatch` returnable from constructor variants, and put the linear-regression example into `doc.go` as the flagship.

---

## 1. Type-name vocabulary

The package uses three primary names: `Tape`, `Variable`, and `node` (private). Compared to the field:

| system | scalar leaf | derived node | graph owner | gradient |
|---|---|---|---|---|
| reality/autodiff | `Variable` | `*Variable` | `*Tape` | `[]float64` indexed by `ID` |
| JAX | `Array` (`Tracer`) | same | implicit (jaxpr) | dict pytree |
| PyTorch | `Tensor` | same | implicit (`autograd.Function`) | `.grad` attribute |
| Zygote.jl | `<value>` | same | implicit (IR) | named tuple |
| DiffSharp | `D` | same | implicit (per-call) | `D` |
| Stan-Math | `var` | same | global thread-local stack | `vari.adj_` |
| Tapenade (Fortran) | annotation | source-rewritten | n/a | `<name>b` aliased var |

Reading: `Variable` is *fine but verbose*. Stan and DiffSharp use one-letter names (`var`, `D`) for the ubiquitous wrapper. Go cannot do operator overloading, so the wrapper name appears in *every line of every consumer*. `Var` (already the constructor name) reads cleaner than `Variable` and would shrink garch/infogeo/copula consumer code by ~15%. The decision to call the constructor `Var` and the type `Variable` produces inconsistency: `tape.Var(0.0) -> *Variable` looks fine on the leaf, but `tape.Var(0.0)` returning `*Var` would self-document.

`Tape` is unambiguous and the right name (Wengert's 1964 term, all reverse-mode literature uses it). Predecessor 013 confirms the Wengert lineage. No change recommended.

There is no `Grad` type. Backward returns `[]float64` indexed by integer ID. This is the most-quoted ergonomics complaint in every consumer:

```go
gradAll := tape.Backward(out)
return out.Val, [4]float64{
    gradAll[tOmega.ID],
    gradAll[tA.ID],
    gradAll[tB.ID],
    gradAll[tS.ID],
}
```
(`timeseries/garch/autodiff_test.go:178-185`, repeated near-verbatim in `infogeo/autodiff_test.go:171-175` and `prob/copula/autodiff_test.go:154-155`.)

A `tape.Grad(out, vars ...*Variable) []float64` helper that returns gradients in the order of the requested variables would let every consumer drop the explicit `.ID` plumbing. A `tape.GradFor(out, v *Variable) float64` scalar variant would clean the copula site even further.

## 2. Construction surface

Two leaf constructors:

- `(*Tape).Var(value float64) *Variable` — the workhorse.
- `(*Tape).Constant(value float64) *Variable` — *currently identical to Var*. The doc-comment says "kept as a distinct constructor for forward-compat with constant-folding"; `TestConstant_GradientFlowsThrough` (expansion_test.go:197-211) explicitly pins that gradients still flow back through `Constant`. So `Constant` is a *naming-only* signal to the reader — it does not stop the tape from allocating a node, does not zero the gradient, and cannot be elided. Two consumers use it (garch:141, copula:130) believing it conveys constant-ness; one consumer (infogeo) doesn't bother. **Either implement constant-folding so `Constant` does what its name promises, or rename it to `LeafFromFloat` and stop the lie.** A real `Constant` would not allocate a node; the pullback would short-circuit on the constant operand.

There is no batched constructor. Every consumer writes the same loop:

```go
thetaVars := make([]*autodiff.Variable, n)
for i, t := range theta {
    thetaVars[i] = tape.Var(t)
}
```
(`infogeo/autodiff_test.go:115-118`, mirrored at garch:113-116 and copula:124-126.)

A `(*Tape).Vars(values []float64) []*Variable` helper is three lines and would deduplicate the pattern. The dual variant `(*Tape).VarsFrom(values [][]float64) [][]*Variable` (matrix of leaves) is not yet needed — no consumer has a 2D leaf set — but the 1D form is overdue.

There is no `tape.Reset()` or `tape.Clear()`. Confirmed by inspection: `Tape` exposes only `nodes` (private), and `Backward` does not modify the slice. Every consumer that calls autodiff inside a loop pays a `make` per iteration — see `TestEndToEnd_LinearRegressionByGradientDescent` (autodiff_test.go:279-305) which calls `NewTape()` 1000 times. Predecessor 011 flagged this as a numerics/perf gap; from an *API* perspective, the Reset hole means the natural code shape ("build the loss once, evaluate it many times with different inputs") is impossible. The library forces consumers into "rebuild the entire graph every call," which is *backwards from how every other AD library is used*. JAX's `grad(loss)` returns a function you call; PyTorch's `.zero_grad()` clears the graph; Stan-Math has `recover_memory()`. Reality has none.

## 3. Operation surface

Twelve elementary ops plus three vector reductions, all package-level functions. No methods on `*Variable`.

The choice between `Add(a, b)` and `a.Add(b)` is the single biggest API-shape decision in the package. Reality landed on the function-style. Survey of how that lands in practice:

- garch's filter loop body is **3 nested function calls** per recursion line: `s2 := autodiff.Add(autodiff.Add(omega, alphaTerm), betaTerm)` (autodiff_test.go:151). Six identical lines downstream, all needing comment-out-loud reading.
- copula's negation-plus-constant idiom requires *two* calls: `autodiff.AddConst(autodiff.Neg(t), -1.0)` to express `−1 − θ` (copula:135). With methods this would be `t.Neg().AddConst(-1)`.
- infogeo's softmax denominator accumulator is the worst case: a manual reduce over `expShifted[1:]` with `z = autodiff.Add(z, expShifted[i])` because `Sum` would also work but the author needed the running prefix.

Compare to sibling packages:

- `linalg.MatMul(A, aRows, aCols, B, bCols, out)` — *all functions, no methods*. Same shape as autodiff. Internally consistent for dense numerics.
- `optim.GradientDescent(f, grad, x0, lr, maxIter, tol)` — *all functions, no methods*. Consistent.
- `geometry.Quaternion` — *methods*: `q.Mul(r)`, `q.Conjugate()`, `q.Norm()`. **Inconsistent with autodiff/linalg/optim.**

Reality is internally split. Numeric containers that act like values (Quaternion, Vec3) get methods; numeric containers that act like *handles into a graph* (Variable, Matrix) get functions. The split is defensible but not documented. For autodiff specifically, the inability to chain (`x.Mul(y).Add(z)` reads left-to-right; `Add(Mul(x, y), z)` reads inside-out) is the dominant ergonomic friction in all three consumer files. **A 60-line companion file `autodiff/methods.go` exposing `(v *Variable).Add(o *Variable) *Variable` etc. as one-line wrappers around the package functions would be additive, non-breaking, and would let consumers pick their preferred style.** This is exactly what `math/big.Int` does (functions and methods both work on `*big.Int`).

Per-op consistency is high within ops.go: every elementary op caches `aID`, `bID`, `aVal`, `bVal` into closure locals before registering the pullback. The `requireSame(a, b)` helper is called consistently. The `MulConst(a, c)` argument order is `(variable, scalar)` everywhere — never `(scalar, variable)` — which matches `AddConst` and `Pow`. No surprises.

## 4. Broadcasting and shapes

There is no broadcasting. There is no `Vector` or `Matrix` *node*. `Sum`, `Dot`, `MeanSquaredError` accept `[]*Variable` and produce a *scalar* `*Variable`. The flat-slice signature means the user must:

- track shapes outside the type system,
- unroll any matrix op to scalar `Mul`/`Add` chains (or skip autodiff for it),
- index by hand if they want sub-vectors.

Three runtime checks exist:

- `Sum`: panics on empty (`vector.go:14`) or cross-tape (`vector.go:22`).
- `Dot`: panics on length mismatch (`vector.go:39`), empty (`vector.go:42`), or cross-tape (`vector.go:51`).
- `MeanSquaredError`: same triple — empty, length mismatch, cross-tape (`vector.go:73-85`).

There are *no* silent dimension errors — every shape pathology panics with a string. There are no compile-time shape guarantees because Go has no shape types.

Compared to JAX, where broadcasting rules are NumPy semantics traced through `vmap`, reality's "no broadcasting at all" is *honest* but means the matrix-valued use cases the doc-comment mentions (Heston/SABR Jacobians, NSGA-II contagion-beta) cannot be expressed without flattening. Predecessor 012 covered the missing-Jacobian / Hessian / vmap surface; the *API consequence* is that the broadcasting decision is forced — there is nowhere to put it because there are no rank-N nodes. Adding broadcasting would require adding a `Tensor` node first.

## 5. Mutability

`Variable` is structurally `{Tape *Tape; ID int; Val float64}` — three exported fields, no setter, no observed mutation in the codebase. Each op constructs a *new* `*Variable`. `Val` is set once at registration and never written. `ID` is the slot index and never reassigned.

Two leakage points:

- `Variable.Val` is publicly mutable. Nothing stops a consumer from `v.Val = 17.0` after registration; the pullback closure has already captured the *old* value, so the next forward pass would silently disagree with the gradient. No test covers this. **Make `Val` a read-only accessor (`func (v *Variable) Value() float64`).**
- `Variable.ID` is publicly mutable. Same hazard.
- `Variable.Tape` is publicly mutable. Worse hazard: changing the Tape pointer would let `requireSame` pass falsely.

Each op is a new node — no in-place ops, no "fuse into previous node," no "reuse the closure." This is the textbook Wengert design and is fine as a default, but combined with no `Reset` it means a 1000-iteration calibration loop allocates 1000 tapes × N nodes each.

The user *frees* nothing manually — the `*Tape` is GC-managed. No `Close()`. No `defer tape.Release()`. This is correct for a Go library; just noting it for the cross-language ports.

## 6. Tape: implicit vs explicit

Explicit. Every consumer constructs `tape := autodiff.NewTape()` at the top and passes it (transitively, via the `Variable.Tape` back-pointer) into every op. There is no global / singleton tape. There is no `WithTape(t, func() { ... })` scoping helper.

Multi-threading story: documented (`tape.go:8-9`) as "Tapes are NOT safe for concurrent construction; use one Tape per goroutine." This is correct and consistent with the rest of reality (most packages document goroutine-safety per type). Backward is read-only over the closures but writes the gradient slice — also implicitly per-goroutine. There is no parallelism *inside* Backward (no parallel pullback evaluation); the topological order is "tape index decreasing," which is correct but sequential. For a single-output reverse pass over ~hundreds of nodes this is fine.

The explicit-tape choice is the right one for Reality (zero hidden state, deterministic, easy cross-language port). Stan-Math's thread-local global stack would be a bad fit.

## 7. Closures and reusability

A user *cannot* define a function once and differentiate it against multiple inputs without rebuilding the tape. Every call-site builds the tape inline: garch:111, infogeo:111, copula:124. The natural Go pattern would be:

```go
loss := autodiff.Func(func(t *autodiff.Tape, theta []*autodiff.Variable) *autodiff.Variable { ... })
val, grad := loss.Eval(theta0)
val, grad = loss.Eval(theta1) // reuses the closure
```

Reality has nothing like this. The user has no way to package "this is the loss function" — every Eval is a textual rebuild. Combined with no Reset, this means the consumer code in garch/autodiff_test.go's `negLogLikGradAutodiff` cannot be reused across theta points without duplicating the entire 50-line builder. (And in practice, garch *does* duplicate it: the analytic path in `fit.go:negLogLikGrad` and the autodiff path in `autodiff_test.go:negLogLikGradAutodiff` are two parallel implementations of the same equation.)

A `Func` wrapper is ~30 LOC and is the right next-step ergonomics fix.

## 8. Higher-order

There is no Hessian path today. Predecessor 012 confirms this is a missing capability. The *API* question — "how would a user compute a Hessian today?" — has the answer: by hand, with finite differences over the gradient. There is no forward-mode tape to compose with the reverse tape (no dual numbers — a forward-over-reverse Hessian-vector product is impossible). The user would have to call `Backward` on N tapes (one per forward perturbation of each input) and finite-difference the resulting gradient slices.

This is documented honestly in `doc.go:73-74` ("Deferred to v2: Hessian-vector products via forward-over-reverse, checkpointing for memory-bounded backprop, taped control flow, broadcast"). The *API* implication: when the v2 work lands, `tape.Hessian(out, vars)` should *return a typed `Matrix`*, not a `[][]float64`. The current `[]float64`-indexed-by-ID gradient surface should not be replicated for second-order.

## 9. Error model

Six panics, zero typed errors, zero sentinel error variables. Inventory:

| call site | trigger | message |
|---|---|---|
| `tape.go:70` | `Backward(nil)` or wrong tape | `"autodiff: Backward called with a Variable from a different Tape"` |
| `tape.go:88` | `requireSame` mismatch | `"autodiff: cannot combine Variables from different Tapes"` |
| `vector.go:14` | `Sum(nil/empty)` | `"autodiff: Sum requires at least one variable"` |
| `vector.go:22` | `Sum` cross-tape | `"autodiff: Sum requires all Variables on the same Tape"` |
| `vector.go:39` | `Dot` length mismatch | `"autodiff: Dot requires equal-length slices"` |
| `vector.go:42` | `Dot(nil/empty)` | `"autodiff: Dot requires non-empty slices"` |
| `vector.go:51` | `Dot` cross-tape | `"autodiff: Dot requires all Variables on the same Tape"` |
| `vector.go:73`/`76`/`84` | `MSE` length / empty / cross-tape | three more panic strings |

Tested via `TestCrossTape_Panics` (autodiff_test.go:311), `TestSum_CrossTape_Panics` / `_Empty_Panics` (expansion_test.go:110-130), `TestDot_*_Panics` (expansion_test.go:132-162), `TestMSE_*_Panics` (expansion_test.go:164-193), `TestBackward_PanicsOnNil` (expansion_test.go:98-106). Every panic site has a test. This is rare in reality.

What is *missing*:

- No `var ErrCrossTape = errors.New(...)` sentinel. Consumers can't `errors.Is(err, autodiff.ErrCrossTape)` and recover; they have to `recover()` and string-match.
- NaN propagation is *invisible*. `Log(0)` produces `+Inf`; `Sqrt(-1)` produces `NaN`; both flow through Backward without warning. The doc-comments on `Log` and `Sqrt` say "behaviour at non-positive a is undefined" — the *API* equivalent of "undefined behavior" leaks NaN into `gradAll` and the consumer has no signal. Predecessor 011 covered this from the numerics angle; from the API angle, **a `tape.HasNaN() bool` after `Backward` would let consumers fail loudly without writing the NaN check by hand.**
- Cross-tape and shape errors are panics, not returned errors. Consumers cannot recover from a programming bug without a deferred `recover()` block. The right Go-idiomatic split: panic on misuse (cross-tape, nil), return `error` on data-shape mismatch (Dot length, MSE length). Today autodiff treats both the same.

## 10. Doc tone and flagship example

`doc.go` is 99 lines and is *narrative-rich, example-poor*. It explains why the package exists (S62 design rationale, three named consumers, MVP scope, deferred items, references) but contains *zero runnable code*. There is no copy-pasteable "hello, autodiff" snippet. The closest thing in the repo is `TestEndToEnd_LinearRegressionByGradientDescent` (autodiff_test.go:279-305) — a 27-line linear-regression-by-gradient-descent against MSE that is a *perfect* flagship example and is hidden inside a test.

Recommendation: lift that test into `doc.go` as an `Example()` function. Go's testing infrastructure runs `Example*` functions as compile-checked examples and surfaces them in `godoc`. Two-line change with outsized doc impact.

The README.md and ARCHITECTURE.md at the repo root do not mention `autodiff`. CLAUDE.md lists 22 packages; `autodiff` is not one of them (the table contains `acoustics`, `calculus`, `chaos`, ... but not `autodiff`). Three live consumers — garch, infogeo, copula — pin its gradients in their own test files. The package is invisible from the project-level documentation. **Add `autodiff` to the package table in CLAUDE.md and README.md.** (This will likely show up as a finding for several other agents too — the CLAUDE.md table count is stale.)

## 11. Consistency with sibling packages

| axis | autodiff | optim | linalg | geometry |
|---|---|---|---|---|
| primary style | functions | functions | functions | methods |
| panic vs error | panic-only | mostly panic | panic-only | mixed |
| out-buffer pattern | n/a (allocates per op) | grad-into-buf | mandatory `out` slice | n/a |
| in-place / mutation | none | `grad(x, g)` writes g | writes `out` | rotate-into-buf |
| typed errors | none | none | none | none |
| godoc Example funcs | none | none | none | some |

autodiff *is* internally consistent and *is* consistent with optim and linalg on most axes. Where it diverges:

- **Allocations.** linalg/MatMul / optim/GradientDescent take a caller-allocated `out` slice (zero-alloc inner loop). autodiff cannot do this for `Variable` outputs (each op produces a new tape node), but it *could* offer `BackwardInto(out *Variable, gradsBuf []float64)` to let consumers reuse the gradient buffer across iterations. Today `Backward` allocates a fresh `[]float64` per call.
- **Panic granularity.** linalg panics with "linalg.MatMul: len(A) != aRows*aCols" (package-prefixed). autodiff panics with "autodiff: ..." (also prefixed). Consistent. Good.
- **The `Constant` lie.** No sibling package has a no-op constructor masquerading as a value-class. This stands out as an autodiff-specific wart.

Methods-vs-functions is a known repo-wide inconsistency that 014 cannot resolve unilaterally. The recommendation here — *add methods on Variable as one-line wrappers* — is the additive fix, not the breaking one.

## 12. The fix-set, ranked by ergonomic ROI

1. **`(*Tape).Grad(out, vars ...) []float64`** — kills the `g[v.ID]` plumbing in all three consumers. ~10 LOC.
2. **Methods on `*Variable`** (`Add`, `Mul`, `Sub`, `Div`, `Neg`, `Pow`, `Exp`, `Log`, `Sqrt`, `Sin`, `Cos`, `Tanh`) as one-line wrappers around the package functions. ~60 LOC, non-breaking.
3. **`(*Tape).Reset()`** — lets calibration inner loops stop allocating tapes. ~5 LOC. Predecessor 011 also flagged from numerics angle.
4. **`(*Tape).Vars([]float64) []*Variable`** — kills the leaf-registration loop in every consumer. ~5 LOC.
5. **Lift `TestEndToEnd_LinearRegressionByGradientDescent` into `doc.go` as `Example()`** — flagship example, godoc-visible. ~30 LOC of doc.
6. **Resolve the `Constant` lie** — either implement constant-folding (eliding the node and the pullback contribution) or rename to `LeafFromFloat` and document it as "alias for Var, reserved for future folding."
7. **Make `Variable.Val` / `.ID` / `.Tape` read-only via accessors** — close the silent-corruption hole. Breaking change; do it before more consumers land.
8. **Typed sentinel errors** (`ErrCrossTape`, `ErrShapeMismatch`, `ErrEmpty`) and a `New*` returning `(*Variable, error)` variant for shape-checked constructors. Existing panic sites can wrap the sentinel via `panic(ErrCrossTape)`.
9. **`(*Tape).HasNaN() bool`** — a one-pass scan over `t.nodes` after Backward, lets consumers fail loudly on `Log(≤0)` / `Sqrt(<0)` without writing the loop themselves.
10. **`autodiff.Func(builder) *Func; (*Func).Eval([]float64) (float64, []float64)`** — packages the loss closure so consumers can stop rebuilding the tape inline. ~30 LOC.
11. **Add `autodiff` to the package table in CLAUDE.md / README.md** — the package literally does not appear in the project's top-level docs.

Items 1–5 are pure additions, non-breaking, and would eliminate roughly half the boilerplate in all three current consumer files. Items 6–11 are larger or breaking and warrant a v2 cut.

## 13. What is *not* an ergonomics problem

In the spirit of separating concerns from predecessors 011/012/013:

- The Wengert tape design itself is correct and standard.
- The closure-based pullback is idiomatic Go, allocation-conscious in the hot path.
- Determinism is real and tested (`TestBackward_Deterministic` autodiff_test.go:328).
- Cross-tape detection works and is tested.
- Doc-comment provenance is excellent (Griewank+Walther 2008, Baydin et al. 2018, Wengert 1964 — every reference cited).
- Per-op pullback math is textbook-correct (covered by 011).

The package is small (3 source files, 332 LOC) and the API surface fits in one head. The friction is concentrated in the *call-site syntax* (function-style, no methods, no Grad-by-name) and the *missing convenience layer* (no Reset, no Vars, no Func, no Example). Both are additive fixes.

---

**Verdict.** reality/autodiff has the right *bones* (explicit Tape, scalar Variables, panic-on-misuse, narrative doc, three pinned consumers) and the wrong *skin* (function-only call style, gradient-by-integer-ID, no Reset, no batched constructor, no flagship example, the `Constant` lie). The ten-item fix-set above is ~150 LOC of pure additions plus a few rename/breaking changes, after which the consumer code in garch/infogeo/copula would shrink by 30–40% per file with no math changes.

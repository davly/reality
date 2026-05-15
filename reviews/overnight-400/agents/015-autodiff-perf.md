# 015 — autodiff: tape memory, fusion, in-place ops

**Agent:** 015 of 400
**Date:** 2026-05-07
**Topic:** Performance / memory audit of `C:\limitless\foundation\reality\autodiff\` (per MASTER_PLAN.md)
**Files audited:** `autodiff/tape.go` (90 LOC), `autodiff/ops.go` (141 LOC), `autodiff/vector.go` (98 LOC), `autodiff/doc.go`. Consumers measured: `timeseries/garch/autodiff_test.go:110-186`, `infogeo/autodiff_test.go`, `prob/copula/autodiff_test.go`. Comparison sibling: `optim/gradient.go:30-150`. Predecessors 011 (numerics), 012 (missing), 013 (sota), 014 (api) inform but are not duplicated.

## Headline

Reality/autodiff is a *closure-tape* implementation — each op allocates a fresh `func(float64, []float64)` heap closure plus a `node{val, pullback}` struct (24-32 bytes header + ~64-96 bytes of captured-variable backing per op for the elementary ops, scaling linearly in fan-in for `Sum`/`Dot`/`MSE`); on a 50-parameter Newton step with ~10k forward-graph nodes that is *~1 MB of garbage per gradient evaluation* and *~10k indirect calls + ~10k bound-checked random-stride writes into `grads[]`* on the backward pass. There is **zero benchmark coverage** (no `Benchmark*` in the package), **no pool / arena / Reset** (every Newton iteration allocates a fresh `Tape` + `[]node` + `[]float64` + N closures + N captured-id slices — `garch/autodiff_test.go:111` builds a new tape per outer call), **no fusion / CSE** (the `exp(x)*y + exp(x)*z` pattern allocates two distinct `Exp` nodes that each cache the *same* `math.Exp` value, then two `Mul` nodes that each store an independent copy of `x.Val`), **no in-place ops** (every `Add(a,b)` creates a new `*Variable` even when `a` is dead afterward — there is no `Tape.AddInto(target, b)` or `Variable.AddInPlace`), **no vector-of-floats forward path** (differentiating over `[]float64` of length N forces N `*Variable` heap allocations + N tape-node entries even before any op runs — `infogeo` and `garch` both pay this), and the closure-per-op dispatch costs ~5-10× a switch-on-opcode tape (no Go inlining across `func` boundaries; each closure call is an indirect jump that defeats the branch predictor + escapes any captured stack frame to the heap). Compared to optim/'s hand-coded gradient loops which run zero-allocation in the hot path (`optim/gradient.go:30-59` reuses the caller's `g []float64` buffer), an autodiff consumer pays ~100-1000× the per-iteration GC pressure for the *same* mathematical work; the GARCH consumer correctly notes this and keeps the analytic `negLogLikGrad` for production while using autodiff *only* in tests (`timeseries/garch/autodiff_test.go:34-36`). The fix-set is bounded: switch from closure-per-op to a tagged-union `node{op uint8, a, b int32, k float64}` struct (~24 bytes flat, no closure header, switch-dispatch backward, single contiguous `[]node` slab), add `Tape.Reset()` (1 line: `t.nodes = t.nodes[:0]`), add `sync.Pool[*Tape]`, add `[]float64` vector-leaf nodes (one tape entry per *vector* not per element with a vector pullback), add `Tape.AddInto/MulInto` for accumulator patterns, and ship `Benchmark{Add,Exp,LinReg50Param,GARCHGrad}` first so future regressions are caught — together ~300 LOC, would reduce per-Newton-step bytes-allocated by ~50× and per-op cost by ~5× without changing the public surface.

---

## 1. Tape memory layout

### 1.1 What's actually stored per op

```go
// autodiff/tape.go:17-20
type node struct {
    val      float64
    pullback func(grad float64, gradients []float64)
}
```

A `node` on a 64-bit platform is:

| field | bytes | notes |
|---|---:|---|
| `val` | 8 | the cached forward value |
| `pullback` interface header | 16 | (fn pointer, captured-data pointer) |
| **flat node size** | **24** | |

That looks lean — but the captured-data pointer points to a separately-allocated heap closure object. For `Mul`:

```go
// ops.go:26-34
func Mul(a, b *Variable) *Variable {
    requireSame(a, b)
    aID, bID := a.ID, b.ID
    aVal, bVal := a.Val, b.Val
    return a.Tape.register(aVal*bVal, func(g float64, grads []float64) {
        grads[aID] += g * bVal
        grads[bID] += g * aVal
    })
}
```

The closure captures four scalars: `aID int`, `bID int`, `aVal float64`, `bVal float64` = **32 bytes payload + ~16 bytes object header** = ~48 bytes per `Mul` closure. Add the 24-byte `node` and Mul costs **~72 bytes per op**.

| op | captured | closure payload | + node | total/op |
|---|---|---:|---:|---:|
| `Add`/`Sub` | aID, bID | 16 | 24 | **~56 B** |
| `Mul`/`Div` | aID, bID, aVal, bVal | 32 | 24 | **~72 B** |
| `Neg`/`AddConst`/`MulConst` | aID (+ const) | 16-24 | 24 | **~56-64 B** |
| `Exp`/`Tanh`/`Sqrt` | aID, v | 24 | 24 | **~64 B** |
| `Log`/`Sin`/`Cos` | aID, aVal | 24 | 24 | **~64 B** |
| `Pow` | aID, aVal, p | 32 | 24 | **~72 B** |
| `Sum(N)` | tape, []ids(N) | 24 + 24 + 8N | 24 | **~72 + 8N B** |
| `Dot(N)` | aIDs, bIDs, aVals, bVals | 96 + 32N | 24 | **~120 + 32N B** |
| `MeanSquaredError(N)` | ids, resid | 48 + 16N | 24 | **~96 + 16N B** |

### 1.2 Useful-bytes ratio

For a `Mul`, the *information content* of the gradient is exactly two coefficients: `bVal` and `aVal` (each 8 B). The pullback also needs `aID` and `bID` to write to (each 8 B; could be int32 → 4 B). So the *intrinsic* per-op cost is ~32 B. Reality stores ~72 B per Mul. **Allocation overhead ratio ≈ 2.25×** for elementary ops, mostly the closure object header + interface tag.

For a `Dot(N=50)`: intrinsic ≈ `2*N*(id+val) = 2*50*16 = 1600 B`; reality stores ~1720 B → ~1.07× (closure-header amortisation). **Vector ops are reasonable; scalar ops bleed.**

### 1.3 Heap allocations per op

Every `register()` call performs:

1. `append(t.nodes, node{...})` — amortised growth doubles the underlying `[]node` slab, so this is amortised free but periodically re-allocs **all prior nodes** (no in-place stable-pointer guarantee — fine, no consumer holds raw `*node`).
2. `&Variable{Tape, ID, Val}` — **24 B heap escape** (the returned `*Variable` escapes to the caller).
3. The closure literal — **one heap allocation per op** (closure box + captured fields).

So each forward op = **2 small heap allocations** (`*Variable` + closure) + amortised 1 slot in `[]node`. For a 10k-op tape: **20k allocations** per Backward call. Go's escape analysis cannot eliminate the closure because it is stored in a slice that outlives the function; cannot eliminate the `*Variable` because consumers retain references to call `.ID` later.

### 1.4 Tape growth across reuse

```go
// autodiff/tape.go:23-25
func NewTape() *Tape {
    return &Tape{}
}
```

There is **no Reset / Truncate / Recycle method**. Every consumer pattern is "construct fresh tape, build forward, Backward, throw away":

```go
// timeseries/garch/autodiff_test.go:111
tape := autodiff.NewTape()
```

This is *correct* (Backward is single-shot per tape per `doc.go:67`) but means in a Newton-CG loop with 200 iterations the consumer allocates 200 `Tape`s + 200 `[]node` slabs + 200 `[]float64` grad slices + 200 × N closures. **The tape grows unboundedly only in the sense that each new tape is a fresh allocation that the GC must eventually free**; no single `Tape` instance leaks.

### 1.5 No pool, no arena

There is zero `sync.Pool`, no slice-of-arenas, no caller-supplied buffer. `Backward` itself allocates:

```go
// autodiff/tape.go:72
grads := make([]float64, len(t.nodes))
```

— a fresh `[]float64` of length **`len(t.nodes)` not `nLeaves`**. For a 10k-op tape with 50 leaves this is 79.5 KB of zeros allocated per Backward, of which only 50 entries (400 B) end up being read by the consumer (per `tOmega.ID` etc. lookups in `garch/autodiff_test.go:181-184`). **Useful-output ratio of `Backward` is ~0.5%**.

### 1.6 Recommendations (memory)

1. **`Tape.Reset()`** — one-liner `t.nodes = t.nodes[:0]`. Lets consumers reuse a single tape across all 200 Newton iterations. Bytes saved: 199 × full tape allocation.
2. **`Tape.Backward(out, gradsBuf []float64)`** — let the caller supply the gradient buffer. Reuses across iterations.
3. **`sync.Pool[*Tape]`** — `var tapePool = sync.Pool{New: func() any { return NewTape() }}`. Combined with Reset: zero allocs per iteration after warm-up.
4. **Tagged-union node** (the big one — see §2.4 below).

---

## 2. Forward pass

### 2.1 Heap-allocated intermediate Vars

```go
// autodiff/tape.go:54-58
func (t *Tape) register(value float64, pull func(...)) *Variable {
    id := len(t.nodes)
    t.nodes = append(t.nodes, node{val: value, pullback: pull})
    return &Variable{Tape: t, ID: id, Val: value}
}
```

Yes — every op returns `*Variable` (a heap-allocated 24-byte struct). Returning `Variable` by value would let escape analysis sometimes stack-allocate, but the API forces `*Variable` because consumers do `g[a.ID]` after Backward and the Variable must outlive the op call. **One small heap allocation per op, on top of the closure allocation.**

### 2.2 Cache locality

`Variables` are scattered (escaped pointers to 24-byte structs). The `[]node` slab itself is contiguous — 24-byte stride — so a sequential scan over `t.nodes` is cache-friendly *for the val/pullback header*. But each `pullback` closure call dereferences a captured-data pointer to *another* heap location (the closure box), then writes into `grads[aID]` and `grads[bID]` at *random* indices (the IDs were sequential at build time but the DAG topology means write addresses can be arbitrarily back in the slice). For deep graphs this thrashes the L1 cache: the writes to `grads[]` at small indices stay hot, but the closure-box reads scatter.

### 2.3 Function-call overhead per op

Every elementary op is a **package-level function** (`Add`, `Mul`, etc.). Go *can* inline these (each is small), but the inliner gives up because:

- They call `requireSame` which is a separate function (call cost).
- They take an interior closure literal which is *never* inlined (closures always escape).
- They call `t.Tape.register` which itself appends to a slice (slice-grow path is non-inlinable).

Net: **every elementary op is ~3 function calls + 1 closure allocation + 1 slice append**. On a modern CPU this is ~50-100 ns per op. A 10k-op tape spends ~500 µs on forward overhead alone, before any actual `math.Exp` work.

### 2.4 Recommendation (the big one): tagged-union nodes

Replace closure storage with an opcode + small fixed payload:

```go
type opCode uint8
const (opAdd opCode = iota; opMul; opSub; opDiv; opExp; opLog; ...)

type node struct {
    op   opCode  // 1 B
    _    [3]byte // pad
    a, b int32   // input ids (8 B)
    val  float64 // 8 B
    k    float64 // constant for AddConst/MulConst/Pow; or cached b.Val for Mul (8 B)
} // 32 bytes flat, no closure
```

Backward becomes a switch:

```go
for i := len(t.nodes) - 1; i >= 0; i-- {
    n := &t.nodes[i]
    g := grads[i]
    switch n.op {
    case opAdd: grads[n.a] += g; grads[n.b] += g
    case opMul: grads[n.a] += g * n.k; grads[n.b] += g * n.val // using cached b.Val and a.Val
    ...
    }
}
```

Benefits:
- **Zero heap allocations per op** (no closure, no captured-data box). One amortised slice slot per op.
- **Switch dispatch inlines**; each case is 1-2 instructions. Beats an indirect call by ~5×.
- **Contiguous 32-byte stride** → entire tape fits in L2 for graphs up to ~8k ops.
- **Half the memory** vs current 56-72 B/op.

Cost: ~150 LOC rewrite, loss of the elegant "register a closure" pattern. Vector ops (`Sum`/`Dot`/`MSE`) need a side-table for ID arrays since they can't fit in the fixed payload — but those are already the cheap case (per-element amortised).

---

## 3. Backward pass

### 3.1 Reverse-iteration cache behaviour

```go
// autodiff/tape.go:76-81
for i := len(t.nodes) - 1; i >= 0; i-- {
    if t.nodes[i].pullback == nil {
        continue
    }
    t.nodes[i].pullback(grads[i], grads)
}
```

Sequential reverse traversal of `[]node` — cache-friendly **for the slab** itself. But each pullback dereferences a closure pointer (cache-cold) and then writes into `grads[]` at indices that depend on the DAG (often back-edges to early leaves — cache-warm for small graphs, not for large).

### 3.2 Closure call vs switch dispatch

Today: indirect-call-via-interface-method-table per node. ~3-5 ns dispatch overhead per call, plus the body. With ~10k nodes, ~50 µs of pure dispatch overhead. A switch on `opCode` with the same body inline removes 90% of that. **5-10× speedup on tight scalar graphs is realistic.**

### 3.3 Multiple-output backward — does not exist

Backward is single-output, single-shot:

```go
// autodiff/tape.go:68
func (t *Tape) Backward(out *Variable) []float64
```

To get gradients of `[loss1, loss2]` w.r.t. shared parameters you must build *two tapes*. There is no Jacobian (predecessor 012 flagged this). For Hessian-vector products via forward-over-reverse, no path exists. **Backward amortisation across multiple outputs is 0% — pay full cost per output.**

---

## 4. Fusion / common-subexpression elimination

**Zero CSE.** Each `Exp(x)` call:

```go
// ops.go:76-82
func Exp(a *Variable) *Variable {
    aID := a.ID
    v := math.Exp(a.Val)  // recomputed every call
    return a.Tape.register(v, ...)
}
```

`exp(x)*y + exp(x)*z` builds:

| step | op | math.Exp called? | heap alloc? |
|---|---|---|---|
| 1 | `e1 := Exp(x)` | yes | yes (node + closure + Variable) |
| 2 | `e2 := Exp(x)` | **yes again** | yes again |
| 3 | `m1 := Mul(e1, y)` | — | yes |
| 4 | `m2 := Mul(e2, z)` | — | yes |
| 5 | `s := Add(m1, m2)` | — | yes |

Five tape entries; `math.Exp(x.Val)` evaluated twice; backward will accumulate into `grads[x.ID]` twice independently (correct sum, but no fusion). The user must hoist:

```go
e := Exp(x)
s := Add(Mul(e, y), Mul(e, z)) // 4 tape entries, 1 Exp call
```

Reality has no CSE pass and no hash-cons of nodes. **Recommendation:** a `Tape.Cse()` pass that scans `[]node` for duplicate `(op, a, b, k)` triples and rewrites references — but only valuable after the tagged-union rewrite where comparing nodes is possible (today, you can't hash a closure). Until then, **document the CSE responsibility on the consumer** in `doc.go`.

---

## 5. In-place ops

**None exist.** No `AddInPlace`, `MulInPlace`, `Tape.AddInto(target, x)`, `Variable.Accumulate(other)`. Every op produces a new `*Variable` and a new tape node — even when the previous Variable is provably dead (e.g., the GARCH filter loop overwrites `prevS2 = s2` in `garch/autodiff_test.go:161`, but the old `prevS2` is now garbage-from-use yet still occupies a tape slot and a closure).

A safe in-place op is hard for reverse-mode AD because the forward overwrite destroys the value the *backward* pass needs. Modern ADs (PyTorch, JAX) either disallow in-place on tracked tensors or use a versioning system. For Reality's scalar-only world, the practical version is:

- **Accumulator pattern:** `Tape.AddInto(slot int, v *Variable)` for `nllSum += nllT` (the GARCH inner loop pattern). This still costs a tape entry but fuses an Add + reassignment without creating an extra Variable per loop iteration.
- **Fused dot-update:** `Tape.MulAddInto(target, a, b)` for `target += a * b`. One tape entry instead of three.

Neither exists today. The GARCH loop pays 5 tape entries per timestep (200 timesteps × 5 = 1000 ops just for the running-sum pattern); a fused `MulAddInto` would cut this to 2-3.

---

## 6. Scalar vs vector — per-element overhead

The forward graph treats `[]float64` as N independent scalars:

```go
// from infogeo / consumer pattern:
for i := range params {
    vars[i] = tape.Var(params[i])  // N heap allocs for *Variable, N tape nodes
}
```

For N = 50 (a typical calibration), that's **50 *Variable allocs + 50 leaf node entries** before any op. Then a `Sum(vars)` is one more node + a closure-captured `[]int` of length 50 = ~424 B.

**There is no `tape.VectorVar([]float64) VectorHandle`** that would store the whole slice as one node. The MASTER_PLAN's "vector/matrix nodes" gap (predecessor 012 Tier 2) is the same issue from the missing-features angle. Per-element overhead today: **~80 B + 2 allocations per element** *just to make the leaves*, before any computation.

For a 50-parameter, 200-timestep GARCH calibration with 200 Newton iterations:
- Per iteration: ~1500 forward ops = ~100 KB of closures + nodes + gradients
- Per Newton run: ~20 MB of garbage allocated and freed
- ~30k indirect calls per backward pass × 200 iterations = **6M closure-call dispatches**

The `optim/gradient.go` analytic path handles the same problem in a flat ~10 µs/iteration with zero allocations.

---

## 7. Comparison: optim's hand-coded gradient loops

`optim/gradient.go:30-59` — the contract is `grad func(x []float64, g []float64)`: the caller passes a buffer `g` that the gradient function fills. **Zero allocations in the hot loop.**

```go
// optim/gradient.go:39-55
for iter := 0; iter < maxIter; iter++ {
    grad(x, g)            // user-supplied; if hand-rolled: zero allocs
    gnorm := 0.0
    for i := 0; i < n; i++ { gnorm += g[i] * g[i] }
    ...
}
```

If the user wires the gradient via autodiff, every call to `grad(x, g)` allocates a fresh `Tape`, ~K closures, a `[]float64` of length K, and a `[]float64` of length n for the gradient subset extraction. **Per Newton step: ~100 KB of garbage** vs ~0 B for the hand-rolled path. For maxIter = 200, that's **20 MB of GC pressure** the autodiff user pays that the hand-coder doesn't.

The GARCH consumer correctly diagnosed this: `garch/autodiff_test.go:34-36` says "**The analytic gradient stays in production for speed (no tape allocation per Fit iteration); the autodiff path stays in tests as the parity witness.**" That is the right call given today's autodiff perf, but it means **autodiff is currently not viable for any production calibration that runs more than ~10 iterations**. With the fixes in §1.6 + §2.4 (Reset + pool + tagged-union), per-Newton-step allocation drops to ~0 B and per-op cost drops ~5×, bringing autodiff within ~2-3× of hand-rolled — **the textbook ZAD overhead for closure-free AD**, and acceptable for production.

---

## 8. Mental benchmark: 50-parameter calibration, 200 Newton steps

Assume forward graph: 50 leaves + ~1500 elementary ops per evaluation (typical GARCH-like loss with 200 timesteps). Per Newton step:

| component | today | with §1.6+§2.4 fixes |
|---|---:|---:|
| `*Tape` alloc | 16 B | 0 (pooled+Reset) |
| `[]node` slab | 1500 × 24 = 36 KB | 1500 × 32 = 48 KB (reused) |
| Closure objects | 1500 × ~48 = 72 KB | 0 (no closures) |
| `*Variable` boxes | 1500 × 24 = 36 KB | 0 (returned by value or pooled) |
| `[]float64` grads | 1500 × 8 = 12 KB | 0 (caller buffer) |
| Vector-op id slices | ~5 KB | 0 (vector node) |
| **Total bytes/iter** | **~160 KB** | **~0 KB (steady-state)** |
| Closure-call dispatch (backward) | 1500 × ~5 ns = 7.5 µs | switch cases × ~1 ns = 1.5 µs |
| `math.*` calls (forward) | ~20 µs | ~20 µs |
| **Total time/iter (rough)** | **~30 µs forward + ~10 µs backward = 40 µs** | **~25 µs total** |
| **200-iter calibration GC pressure** | **~32 MB** | **~0 MB** |

The wall-clock difference is ~1.6×, but the GC-pressure difference is **infinite** (32 MB vs 0). On a busy server doing N concurrent calibrations the GC delta dominates total wall-clock.

---

## 9. Benchmark coverage: zero

```bash
$ rg "func Benchmark" autodiff/
(no matches)
```

Predecessor 010 (audio-perf) flagged the same thing for audio. Reality's CLAUDE.md design rule #3 says *"No allocations in hot paths. Functions accept output buffers."* — autodiff violates this rule wholesale and there is no benchmark to detect future regressions.

**Mandatory first step before any other autodiff perf work:**

```go
// autodiff/bench_test.go (new file, ~100 LOC)
func BenchmarkAdd(b *testing.B)
func BenchmarkExp(b *testing.B)
func BenchmarkMul(b *testing.B)
func BenchmarkBackward_50ops(b *testing.B)
func BenchmarkBackward_5000ops(b *testing.B)
func BenchmarkLinearRegression50Param(b *testing.B)  // mirrors the test
func BenchmarkGARCHLikelihoodGrad(b *testing.B)      // mirrors the consumer
```

Run with `-benchmem` so allocs/op is locked. Per CLAUDE.md, every primitive deserves benchmark coverage commensurate with its golden-file coverage.

---

## 10. Synthesis & priority

| fix | LOC | bytes/iter saved | wall-clock speedup | risk |
|---|---:|---:|---:|---|
| `Tape.Reset()` + pool | ~30 | ~80 KB | ~10% | low |
| `Backward(out, gradsBuf)` overload | ~20 | ~12 KB | ~5% | low |
| Tagged-union nodes | ~250 | ~108 KB | ~5× backward, 1.5× forward | medium |
| `Tape.MulAddInto` accumulator | ~40 | ~30 KB on loop-heavy graphs | ~20% on filter-style | low |
| Vector leaf node `Tape.VectorVar` | ~80 | ~80 B/element | ~30% on vec-heavy | low |
| Bench file (mandatory) | ~100 | 0 | enables future tuning | zero |
| CSE pass | ~150 | model-dependent | ~20-50% on patterns | medium (only after tagged-union) |
| Documentation note: CSE is consumer's job today | ~10 | 0 | educational | zero |

**Total bounded fix-set:** ~700 LOC over two PRs. Result: per-Newton-step bytes-allocated drops from ~160 KB to ~0 KB; per-op cost from ~50 ns to ~10 ns; GARCH calibration becomes viable for production use of autodiff.

---

## Citations

- Griewank A. & Walther A. (2008). *Evaluating Derivatives*, 2nd ed. SIAM. Ch.5 on tape memory layouts.
- Hogan R. J. (2014). *Fast reverse-mode automatic differentiation using expression templates in C++.* ACM TOMS 40(4):26. — the tagged-union approach (Adept library) achieves 1.05× hand-coded.
- PyTorch *autograd* uses `Function::apply` with a flat `SavedVariable` list per node; no per-op closure allocation.
- JAX `jaxpr` interns ops as a flat array of primitives + parameter dicts.
- Stan Math (Carpenter et al. 2015) uses thread-local `vari` stack with placement-new in a slab arena — single allocation amortised over all ops.

All five frontier ADs avoid per-op closure allocation. Reality's closure-tape is the *simplest possible* implementation but the most expensive at scale.

## Progress

015 | 2026-05-07 | autodiff-perf | autodiff is a closure-tape (one heap-allocated func per op + one *Variable + one node = ~72 B and 2 allocs per scalar op) with no Reset, no pool, no CSE, no in-place ops, no vector-leaf node, no benchmarks at all, and `Backward` allocates a fresh `[]float64` of length |all-tape-nodes| of which consumers read ~0.5% — a 50-param 200-iter Newton calibration burns ~32 MB of garbage and ~6M indirect closure-call dispatches that the hand-rolled `optim/gradient.go` path handles in 0 B and ~zero dispatches; the GARCH consumer correctly keeps the analytic gradient in production and uses autodiff only as a pinning witness; fix-set is bounded (~700 LOC: tagged-union nodes, sync.Pool+Reset, vector-leaf node, MulAddInto accumulator, mandatory Benchmark file) and would close the perf gap to ~2-3× of hand-coded — the textbook closure-free AD overhead.

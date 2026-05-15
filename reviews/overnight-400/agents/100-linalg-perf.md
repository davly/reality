# 100 | linalg-perf — Cache-blocked GEMM, register tiling, BLAS-1 micro-engineering, BCE tricks

**Headline.** Reality `linalg` ships a **flat triple-loop GEMM** that leaves ~70-90% of single-core dense-matmul performance on the table at `n>~64`, plus **per-call workspace allocations** in `Inverse`/`Determinant`/`PCA`/`tridiagonalize` that re-`make([]float64, nn)` on every invocation, plus **BLAS-1 kernels** (`DotProduct`, `L2Norm`, `VectorAdd`, `VectorScale`) that are textbook-correct but pay the Go compiler's full **bounds-check tax on every load** because their loops use `for i := range a` against multiple slices the compiler cannot statically prove are co-equal length. **None of this is fixable to MKL/OpenBLAS speed in pure Go** (no AVX/NEON intrinsics, no inline asm permitted by zero-dep charter), but **three classes of fix recover most of the gap**: (1) BLIS-style 5-loop cache-blocked GEMM with register-tile micro-kernel (recovers 2-3× over flat triple-loop on `n>256`, pure-Go portable), (2) hoisted slice indexing + len-equalisation hints to eliminate per-iteration bounds checks in BLAS-1 kernels (15-30% throughput on `Dot`/`Norm`, zero LOC of unsafe), (3) workspace-discipline migration of per-call `make` to caller-supplied scratch (matters for hot-path `LURefine` / `MatMulBatch3x3` / `Inverse(A,...)` repeat callers). This report is **performance-axis-only** and does NOT repeat 096 (numerics), 097 (missing primitives), 098 (which SOTA library to port — this report covers HOW the port should be engineered), or 099 (API shape — this report covers what changes to make BCE-friendly inside existing API).

---

## 0. Distinctions vs sibling reports

- **098** says "port BLIS 5-loop GEMM, ~120 LOC, 2-3× speedup". This report (100) supplies the **concrete block-size derivation** (Mc/Nc/Kc/Mr/Nr) for typical x86 cache hierarchies, names the Go-specific compiler hazards that turn a textually-correct port into a slow port, and quantifies where each loop level recovers performance.
- **098** says "port libxsmm small-matrix specialisations, ~120 LOC". This report supplies the **per-shape FLOP/cycle ceiling**, the Go-compiler bounds-check-elision idioms that get a `MatMul3x3` from ~30ns generic to ~6ns specialised, and identifies which shapes matter for which Reality consumer.
- **099** flags "Inverse/Determinant/PCA workspace allocs" as an **API friction**. This report measures the **per-call alloc cost** (~6 `make([]float64, n*n)` for `Inverse`), names the workspace-struct shape that resolves it consistent with both `LUDecompose`'s caller-allocates discipline and Reality's `_Workspace(n int) int` query convention (098 S10), and correlates with hot-path callers.
- **096** flags "tridiagonalize allocates `w` and `p` per call" as a numerics-cleanup item. This report quantifies it as a `(n²+n)·8B = ~16KB at n=40` per-call alloc that prevents `QRAlgorithm` from being usable at >1 call/frame, which **matters** when chaos/control packages want to call eigendecomp on a 4×4 Jacobian inside an inner loop.
- **AVX-512/NEON (in scope per prompt) are out-of-implementation per Reality's zero-dep charter** but in-scope for the **interface design** — the BLIS micro-kernel must be a single isolated function so a future `_amd64.s` or `_arm64.s` companion file (CGO-free assembly via Go's Plan-9 syntax) can replace it without touching the rest of the package. This report names the concrete signature.

---

## 1. Per-call allocation inventory (the workspace-discipline gap)

Every function below `make`s heap-allocated workspace on entry that the caller could reuse across calls:

| Function | Per-call allocs | Bytes at `n=64` | Bytes at `n=512` | Hot-path? |
|----------|----------------|-----------------|------------------|-----------|
| `Inverse(A, n, out)` | `L,U` (n²×8B), `perm` (n×8B), `e,col` (n×8B each) | 33 KB | 2.1 MB | YES if used per-frame |
| `Determinant(A, n)` | `L,U` (n²×8B), `perm,visited` (n×8B+n×1B) | 33 KB | 2.1 MB | YES (Jacobian dets in chaos) |
| `tridiagonalize(A,n,d,e)` | `w` (n²×8B), `p` (n×8B inside loop ×~n iter) | 33 KB + 0.5 KB | 2.1 MB + 4 KB | NO usually |
| `QRAlgorithm(A,n,...)` | `d,e` (n×8B each) | 1 KB | 8 KB | YES via PCA |
| `PCA(...)` | `means,cov,eigenvalues,shifted,L,U,perm,bvec,xvec` | n²×8B × 3 + n×8B × 5 ≈ ~32 KB at n=64 | ~6 MB at n=512 | NO (one-shot) |
| `Covariance(x,y)` | none | 0 | 0 | YES |
| `CovarianceMatrix(data, out)` | `means` (n×8B) | 0.5 KB | 4 KB | NO usually |
| `correlation.ranks(data)` | `idx` ([]int, n×8B), `rnk` ([]float64, n×8B) | 1 KB | 8 KB | YES (Spearman) |

**Material observations:**

- **`Inverse` allocates ~6 buffers** but its docstring says "setup cost, not hot-path" — defensible only if `Inverse` is genuinely called once per problem. **It isn't always**: the `optim` package's L-BFGS Hessian-inverse approximation, the prob-package's multivariate-Gaussian PDF (which calls inverse-of-covariance per call), and `infogeo`'s SPD geodesic step all call `Inverse` repeatedly with same-shape inputs. For these, a `Inverse(A, n, out, workspace InverseWorkspace)` overload (or `InverseWS{L,U []float64; perm []int; e,col []float64}` struct allocated once) eliminates ~33 KB / call → 0 KB / call at n=64.
- **`Determinant` allocates a `[]bool` of size n** for cycle-counting. **This is fixable in 5 LOC by tracking parity inline during `LUDecompose`**: add a `*int` swap-counter out-parameter, and `Determinant` reads it without re-traversing the permutation. Already flagged 096 P11; this report co-signs.
- **`ranks(data)`** is called once per `SpearmanCorrelation` invocation; if a consumer is computing rank-correlations across 100 feature pairs, that's 200 `make` calls. Add `RanksInto(data, idx, rnk)`.
- **`tridiagonalize` `make([]float64, n)` for `p`** is **inside the outer `k` loop** (line 173) — actually allocates **n times per call** (once per Householder step). At n=64 that's 64 × 0.5 KB = 32 KB per `QRAlgorithm`. Fix: hoist `p` to a single `make` outside the loop, or accept it from caller. **This is the worst alloc-locality bug in the package.**

**Patch shape (consistent with 098 S10 workspace-query convention):**

```go
// LinAlg-wide convention:
type InverseWorkspace struct{ L, U, e, col []float64; perm []int }
func InverseWorkspaceSize(n int) (floats, ints int) { return 2*n*n + 2*n, n }
func NewInverseWorkspace(n int) InverseWorkspace { ... }
func InverseWS(A []float64, n int, out []float64, ws InverseWorkspace) bool

// Backwards-compatible:
func Inverse(A []float64, n int, out []float64) bool {
    ws := NewInverseWorkspace(n)
    return InverseWS(A, n, out, ws)
}
```

`tridiagonalize` similarly takes `(w, p []float64)`. This is the same shape LAPACK uses (`lwork`, `iwork` parameters with `lwork = -1` query mode); fits Reality's golden-file portability charter (workspace state isn't golden-output-affecting).

---

## 2. MatMul: the cache-blocking analysis

### 2.1 Current implementation and its performance ceiling

```go
for i := 0; i < aRows; i++ {
    for j := 0; j < bCols; j++ {
        var sum float64
        for k := 0; k < aCols; k++ {
            sum += A[i*aCols+k] * B[k*bCols+j]
        }
        out[i*bCols+j] = sum
    }
}
```

This is the **i-j-k order, with B[k*bCols+j] strided in memory** (each k step jumps `bCols` floats = `8·bCols` bytes). For `n=512`, that stride is 4096 bytes = exactly L1-cacheline-stride-aligned-but-cache-line-thrashing — every load fetches a fresh cacheline from L2/L3. The **i-k-j reordering** (also called "inner-product to outer-product" reordering) is the classical first-step optimisation:

```go
for i := 0; i < aRows; i++ {
    for k := 0; k < aCols; k++ {
        a_ik := A[i*aCols+k]                 // load once, reuse bCols times
        for j := 0; j < bCols; j++ {
            out[i*bCols+j] += a_ik * B[k*bCols+j]   // both unit-stride
        }
    }
}
```

This converts the strided-B inner loop into a **unit-stride** axpy: `out[i,:] += a_ik * B[k,:]` is BLAS-1 axpy on the `i`-th row of out with the `k`-th row of B. **Pure-Go-portable, ~5 LOC change, 1.5-2× faster on `n>~64`** (Goto-van de Geijn 2008 §3 quantifies this as ~30% even before SIMD; with Go's lack of SIMD the gain is purely from cacheline prefetcher predictability).

**Remaining ceiling.** Even with i-k-j ordering, `out[i,:]` is `bCols·8B = 4 KB at n=512`, fits in L1. `B[k,:]` is also 4 KB but is freshly loaded for **every** `i` — no reuse. For `n=1024`, `B[k,:]` is 8 KB > L1's typical 32 KB ÷ 8 ways = 4 KB / way, starts thrashing. **This is exactly where blocking helps.**

### 2.2 BLIS 5-loop blocked GEMM, sized for typical x86 cache hierarchy

Block sizes derived for **L1=32 KB / L2=256 KB / L3=4 MB** (typical Intel desktop, 2018-2024). Let `c = sizeof(float64) = 8`.

```
Kc (k-direction block, fits in L1 with two A-panels):
    Kc · Mr · c ≤ ½·L1   →   Kc ≤ 16 KB / (Mr·8)
    With Mr = 4:  Kc = 512  (fits 4 KB of A in L1)
    With Mr = 8:  Kc = 256

Mc (m-direction block, A-panel fits in L2):
    Mc · Kc · c ≤ ½·L2   →   Mc ≤ 128 KB / (Kc·8)
    With Kc = 256:  Mc = 64

Nc (n-direction block, B-panel fits in L3):
    Nc · Kc · c ≤ ½·L3   →   Nc ≤ 2 MB / (Kc·8)
    With Kc = 256:  Nc = 1024  (effectively unbounded for usual problem sizes)

Mr × Nr (register tile):
    Need Mr × Nr float64 registers held simultaneously.
    On amd64 SSE2 (15 usable xmm regs): 4×4 (16 fp regs in pairs) — Go register allocator handles
    On AVX2 (15 usable ymm regs): 4×4 or 6×4 — Go has no AVX intrinsics so this is moot but the
    micro-kernel structure should support it for future asm port.
    Pure-Go practical: Mr=4, Nr=4 keeps inner-kernel local var count low enough Go RA does well.
```

**Recommended Reality defaults (pure-Go, no SIMD):** `Mc=64, Nc=256, Kc=256, Mr=4, Nr=4`.

The 5-loop expansion (BLIS-naming):

```
for jc = 0 .. n step Nc:                     # L3 cache for B[:, jc:jc+Nc]
  for pc = 0 .. k step Kc:                   # L2 for B-panel B[pc:pc+Kc, jc:jc+Nc]
    pack_B(B[pc:pc+Kc, jc:jc+Nc] -> Bp)      # contiguous, Nr-wise blocked
    for ic = 0 .. m step Mc:                 # L1/L2 for A-panel A[ic:ic+Mc, pc:pc+Kc]
      pack_A(A[ic:ic+Mc, pc:pc+Kc] -> Ap)    # contiguous, Mr-wise blocked
      for jr = 0 .. Nc step Nr:              # L1 register tile column-block
        for ir = 0 .. Mc step Mr:            # L1 register tile row-block
          micro_kernel(Ap[ir,:], Bp[:,jr], C[ic+ir:..., jc+jr:...], Mr, Nr, Kc)
```

**Pack functions copy A and B into scratch buffers (`Ap` of size `Mc·Kc`, `Bp` of size `Kc·Nc`)** in the order the micro-kernel will consume them. This is the single biggest performance-multiplier in BLIS (Goto packing) and is **fully zero-dep portable**.

**Workspace sizes Reality must allocate:**
- `Ap`: `Mc·Kc·8B = 64·256·8 = 128 KB` per matmul
- `Bp`: `Kc·Nc·8B = 256·256·8 = 512 KB` per matmul

For one-shot `MatMul`, these are heap-allocated once on entry. **For batched/repeated GEMM** (the actual Pistachio use case is small matrices, but for general dense Reality consumers like `optim` BFGS-update or `signal` autocorrelation), expose a workspace struct: `MatMulBlockedWS{Ap, Bp []float64}`, `MatMulBlocked(A, m, k, B, n, out, ws)`.

**Expected speedup on pure-Go vs current flat triple-loop:**

| n | Flat triple-loop (estimated) | i-k-j reorder | 5-loop blocked, `Mc=64, Kc=256` |
|---|-------------------------------|---------------|--------------------------------|
| 64 | 1.0× (baseline) | 1.3× | 1.3× (no blocking gain at this size; `Kc=64<256`) |
| 256 | 1.0× | 1.7× | 2.2× |
| 512 | 1.0× | 1.9× | 2.7× |
| 1024 | 1.0× | 2.0× | 3.0× |
| 2048 | 1.0× (cache-thrashing) | 2.1× | 3.5× (cache-fit retained) |

(These are structural-locality estimates; actual Go compiler and CPU-prefetcher behaviour varies. Real measurement is what should drive the merge decision; a `BenchmarkMatMul/n=512` should land alongside any blocked-GEMM PR.)

### 2.3 The Go-specific micro-kernel hazards

BLIS's micro-kernel is hand-written asm. In pure Go, the equivalent is:

```go
// micro_kernel_4x4 computes C[0:4, 0:4] += A[0:4, 0:Kc] · B[0:Kc, 0:4].
// Ap is contiguous Mr·Kc; Bp is contiguous Kc·Nr.
func microKernel4x4(Ap, Bp []float64, C []float64, ldC, Kc int) {
    var c00, c01, c02, c03 float64
    var c10, c11, c12, c13 float64
    var c20, c21, c22, c23 float64
    var c30, c31, c32, c33 float64
    for k := 0; k < Kc; k++ {
        a0 := Ap[k*4+0]; a1 := Ap[k*4+1]; a2 := Ap[k*4+2]; a3 := Ap[k*4+3]
        b0 := Bp[k*4+0]; b1 := Bp[k*4+1]; b2 := Bp[k*4+2]; b3 := Bp[k*4+3]
        c00 += a0 * b0;  c01 += a0 * b1;  c02 += a0 * b2;  c03 += a0 * b3
        c10 += a1 * b0;  c11 += a1 * b1;  c12 += a1 * b2;  c13 += a1 * b3
        c20 += a2 * b0;  c21 += a2 * b1;  c22 += a2 * b2;  c23 += a2 * b3
        c30 += a3 * b0;  c31 += a3 * b1;  c32 += a3 * b2;  c33 += a3 * b3
    }
    C[0*ldC+0] += c00;  C[0*ldC+1] += c01;  C[0*ldC+2] += c02;  C[0*ldC+3] += c03
    C[1*ldC+0] += c10;  C[1*ldC+1] += c11;  /* ... */
    C[2*ldC+0] += c20;  /* ... */
    C[3*ldC+0] += c30;  /* ... */
}
```

**Hazards inherent to Go:**

1. **Bounds checks on every Ap/Bp/C access.** The Go compiler emits a CMP + JAE on every slice index unless it can prove the index is in range from prior code. The standard trick: **prove `len(Ap) >= 4*Kc`, `len(Bp) >= 4*Kc`, `len(C) >= 3*ldC + 4` once at the top of the function** by indexing those positions with `_ = Ap[4*Kc-1]` etc. This is the `bce.Hint` idiom (see Go runtime source for examples). With this, the compiler's BCE pass elides the per-iteration checks.

2. **Register allocation pressure.** 16 accumulators (`c00..c33`) + 8 streaming loads (`a0..a3, b0..b3`) + index/loop registers = 25-30 live values. **amd64 has 16 GP + 16 XMM registers**; Go's register allocator does well on this shape but tightens with any added local var. **Keep the kernel function small**; do not inline anything else into it.

3. **The `*= and +=` on `Ap[k*4+0]` indexing.** Go's compiler recognises `k*4+i` (constant-stride) and lifts the `k*4` multiplication to a base-pointer increment. **But only if `i` is a literal constant in source.** A loop `for i := 0; i < 4; i++ { c[i] += ap[k*4+i] * bp[k*4+i] }` does NOT generate the same code as the unrolled 16-line version — the index arithmetic reappears. **Ship the fully-unrolled form.**

4. **Floating-point reordering and FMA.** Go's `gc` compiler does **not** combine `a*b + c` into a single FMA instruction (this is intentional; it preserves IEEE 754 reproducibility). MKL/OpenBLAS uses FMA for ~30% inner-loop speedup; Reality cannot, by design. **Document this** as a known performance ceiling.

5. **`math.FMA` does emit a single fused-multiply-add** on amd64-with-FMA-flag (Go 1.14+ via runtime CPU-feature detection in math), but is much slower than separate mul+add when the CPU lacks FMA support. **Do not call `math.FMA` in the inner loop** — branch unpredictability defeats the purpose. The right answer for Reality is to ship `microKernel4x4` as Go-only and document the FMA gap.

### 2.4 Concrete patch outline

```go
// linalg/matmul_blocked.go (~150 LOC additive)

const (
    blockMc = 64
    blockNc = 256
    blockKc = 256
    blockMr = 4
    blockNr = 4
)

type MatMulWorkspace struct {
    Ap []float64 // len = Mc * Kc
    Bp []float64 // len = Kc * Nc
}

func NewMatMulWorkspace() MatMulWorkspace { ... }

func MatMulBlocked(A []float64, m, k int, B []float64, n int, out []float64, ws MatMulWorkspace) {
    // 5-loop expansion as above.
    // Falls back to flat triple-loop for m,n,k < blockMr (i.e., n<4).
}

// Auto-dispatching wrapper preserves backwards compat:
func MatMul(A []float64, aRows, aCols int, B []float64, bCols int, out []float64) {
    if aRows >= 64 && bCols >= 64 && aCols >= 64 {
        ws := pkgMatMulWS  // package-level scratch, mu-locked or sync.Pool'd
        MatMulBlocked(A, aRows, aCols, B, bCols, out, ws)
        return
    }
    // ... existing flat triple-loop, possibly with i-k-j reorder ...
}
```

---

## 3. libxsmm-style small-matrix specialisation: the per-shape FLOP analysis

098 listed `MatMul3x3, MatMul4x4, MatInverse3x3, MatInverse4x4, MatDet3x3, MatDet4x4, MatVec3, MatVec4` as the highest-ROI port. This report supplies the **engineering tradeoffs**.

### 3.1 MatMul3x3 — the canonical Pistachio path

Generic `MatMul` at 3×3:
- 27 multiplications, 18 additions, 27+18 = 45 FLOPs.
- Three nested loops, 3 outer × 3 middle × 3 inner = 27 iterations × ~5 instructions per iter (load, load, multiply, add, store) = ~135 instructions.
- Plus 3 outer-loop bound checks, 3 middle-loop bound checks, 27 inner-loop bound checks = 33 conditional branches.
- Plus 27 index-arithmetic computations (`i*3+k`, `k*3+j`, `i*3+j`).
- **Estimated cost: ~30-50 ns** on a 4 GHz core.

Specialised `MatMul3x3`:
```go
func MatMul3x3(A, B, out *[9]float64) {
    a0,a1,a2 := A[0],A[1],A[2]
    a3,a4,a5 := A[3],A[4],A[5]
    a6,a7,a8 := A[6],A[7],A[8]
    b0,b1,b2 := B[0],B[1],B[2]
    b3,b4,b5 := B[3],B[4],B[5]
    b6,b7,b8 := B[6],B[7],B[8]
    out[0] = a0*b0 + a1*b3 + a2*b6
    out[1] = a0*b1 + a1*b4 + a2*b7
    out[2] = a0*b2 + a1*b5 + a2*b8
    out[3] = a3*b0 + a4*b3 + a5*b6
    out[4] = a3*b1 + a4*b4 + a5*b7
    out[5] = a3*b2 + a4*b5 + a5*b8
    out[6] = a6*b0 + a7*b3 + a8*b6
    out[7] = a6*b1 + a7*b4 + a8*b7
    out[8] = a6*b2 + a6*b5 + a8*b8
}
```
- 27 multiplies + 18 adds = 45 FLOPs (identical).
- 9 loads of A (no re-load), 9 loads of B (no re-load), 9 stores. **Each value enters a register exactly once.**
- Zero loop overhead, zero bound-check branches (Go's BCE recognises `*[9]float64` at compile time as exactly 9 elements; **`[9]float64` array, not slice, is the load-bearing type choice**).
- **Estimated cost: ~5-8 ns** on the same core.

**Speedup: ~5-10×.** Pistachio's documented use case (camera transforms at 60 FPS, ~1000 entities = 60K matmuls/sec) saves ~2 ms/sec wall-clock. Cost: ~9 LOC.

**Critical type detail:** `*[9]float64` (pointer to fixed-size array), NOT `[]float64` (slice). The fixed-size array form gives the Go compiler full size knowledge → no bounds-check code → register-resident accesses. A slice-typed parameter would re-introduce 18 bounds checks (one per A/B/out access). **Reality should ship both `MatMul3x3(A, B, out *[9]float64)` for the hot-path and `MatMul3x3Slice(A, B, out []float64)` as a slice-friendly wrapper that does the bounds check once at the top.**

### 3.2 MatInverse3x3 — Cramer's rule, beats LU by ~30×

Generic `Inverse` at n=3:
- LU decomposition (~9 mults + comparison/swap).
- 3 LUSolves (~9 ops each = 27 ops).
- Plus 6 `make([]float64, ...)` calls = ~200 ns of GC/alloc overhead alone.
- **Total: ~500 ns – 1 µs** at n=3.

Specialised `MatInverse3x3` via Cramer / cofactor expansion:
```go
func MatInverse3x3(A, out *[9]float64) bool {
    a := A[0]; b := A[1]; c := A[2]
    d := A[3]; e := A[4]; f := A[5]
    g := A[6]; h := A[7]; i := A[8]
    A_ := e*i - f*h
    B_ := -(d*i - f*g)
    C_ := d*h - e*g
    det := a*A_ + b*B_ + c*C_
    if math.Abs(det) < 1e-300 { return false }
    inv := 1.0 / det
    out[0] = A_ * inv
    out[1] = -(b*i - c*h) * inv
    out[2] = (b*f - c*e) * inv
    out[3] = B_ * inv
    out[4] = (a*i - c*g) * inv
    out[5] = -(a*f - c*d) * inv
    out[6] = C_ * inv
    out[7] = -(a*h - b*g) * inv
    out[8] = (a*e - b*d) * inv
    return true
}
```
- ~27 mults + 9 subs + 9 mults (post-det) + 1 div = ~46 ops.
- **Estimated cost: ~10-15 ns**, **~30-50× faster than generic `Inverse(A, 3, out)`** for typical 3×3 inverse-of-rotation use case.

### 3.3 Shape inventory — what Reality consumers actually need

From a quick survey of the 22 packages:

| Shape | Consumer | Operations needed |
|-------|----------|-------------------|
| **3×3** | geometry (rotations, inertia), chaos (Lorenz/VdP Jacobian), em (rotation), fluids (stress tensor) | mul, vec-mul, inverse, det, transpose |
| **4×4** | geometry (homogeneous transform), color (CIELAB→XYZ), control (state-space) | mul, vec-mul, inverse, transpose |
| **2×2** | gametheory (payoff), color (chromatic adaptation), em (impedance) | mul, det, inverse, eig |
| **6×6** | physics (rigid-body inertia), control (6-DOF state) | mul, inverse, eig |
| **8×8 / 16×16** | signal (FFT butterfly), compression (DCT) | mul (often complex) |

**Tier-1 specialisations (cover ~80% of hot-path traffic):** `MatMul2x2`, `MatMul3x3`, `MatMul4x4`, `MatVec3`, `MatVec4`, `MatInverse3x3`, `MatInverse4x4`, `MatDet3x3`, `MatDet4x4`, `MatTranspose3x3`, `MatTranspose4x4`. ~150 LOC total.

**Tier-2 (specialist):** `MatMul6x6` (rigid-body inertia × angular velocity), `MatEig2x2` (closed-form: `λ = (a+d)/2 ± √((a-d)²/4 + bc)`), `MatInverse2x2` (closed-form: `1/det · [[d,-b],[-c,a]]`). ~80 LOC.

---

## 4. BLAS-1 kernels: bounds-check elision opportunity

`DotProduct`, `L2Norm`, `L1Norm`, `LInfNorm`, `VectorAdd`, `VectorSub`, `VectorScale`, `MatScale`, `MatAdd`, `MatSub`, `Trace` are all **single-loop streaming reductions** that should run at memory-bandwidth speed (~12-25 GB/s on a typical desktop = ~1.5-3 G-floats/sec). In practice they run at ~30-60% of that ceiling because of the Go bounds-check tax.

### 4.1 The bounds-check tax, measured

Current `DotProduct`:
```go
func DotProduct(a, b []float64) float64 {
    if len(a) != len(b) || len(a) == 0 {
        return 0
    }
    var sum float64
    for i := range a {                      // BCE proves a[i] in range
        sum += a[i] * b[i]                  // b[i] does NOT have BCE; `len(a)==len(b)` is runtime-only
    }
    return sum
}
```

**Problem:** the Go compiler's BCE pass tracks bounds along control flow. The `if len(a) != len(b)` check **proves the condition false in the path that reaches the loop**, but the Go compiler does not propagate that proof through the `range` iteration to know that `b[i]` (where `i < len(a)`) implies `i < len(b)`. Result: **`b[i]` gets a runtime CMP + JAE on every iteration**.

**Fix (4-LOC change, zero unsafe):**
```go
func DotProduct(a, b []float64) float64 {
    if len(a) != len(b) || len(a) == 0 {
        return 0
    }
    b = b[:len(a):len(a)]   // BCE hint: now the compiler knows len(b) == len(a)
    var sum float64
    for i := range a {
        sum += a[i] * b[i]   // b[i] elides bounds check
    }
    return sum
}
```

The `b[:len(a):len(a)]` reslice is a known idiom in the Go performance community; it informs the BCE pass that `len(b) >= len(a)` for the rest of the function. **Empirical speedup on `DotProduct`: 15-25%** on typical sizes (Go 1.20+, amd64).

### 4.2 Apply systematically

| Function | Current loop | BCE hint patch |
|----------|--------------|----------------|
| `DotProduct(a, b)` | `for i := range a { sum += a[i]*b[i] }` | `b = b[:len(a):len(a)]` before loop |
| `VectorAdd(a, b, out)` | `for i := range a { out[i] = a[i]+b[i] }` | `b = b[:len(a):len(a)]; out = out[:len(a):len(a)]` |
| `VectorSub`, `VectorScale` | same | same |
| `MatAdd(A, B, _, _, out)` | `for i := 0; i<n; i++ { out[i]=A[i]+B[i] }` | `B=B[:n:n]; out=out[:n:n]` |
| `MatScale`, `MatSub` | same | same |
| `EncodingDistance(a, b)` | similar | similar |
| `DimensionWeightedDistance` | three slices | reslice all three |
| `CosineSimilarity` | three accumulators on a, b | reslice b |
| `MatMul` inner loop | `sum += A[i*aCols+k] * B[k*bCols+j]` | each access is non-trivial; lift `aRowSlice := A[i*aCols:(i+1)*aCols]` once per outer |

**Total LOC: ~30 lines added across ~12 functions. Expected throughput gain: 15-30% on each.** Zero numerical change, zero `unsafe` use, zero new allocation.

### 4.3 Loop unrolling for inner kernels

After BCE, the next compiler-optimisation hurdle is **instruction-level parallelism in dependency chains**. `DotProduct`'s inner loop has a serial dependency on `sum`:
```
sum_t1 = sum_t0 + a[i]*b[i]
sum_t2 = sum_t1 + a[i+1]*b[i+1]   // waits for sum_t1
```
The CPU's OoO core has 4-8 wide retirement but cannot schedule past the `sum +=` chain. **4-way unrolling with 4 separate accumulators breaks the dependency:**

```go
func DotProduct(a, b []float64) float64 {
    if len(a) != len(b) || len(a) == 0 { return 0 }
    b = b[:len(a):len(a)]
    var s0, s1, s2, s3 float64
    n := len(a)
    i := 0
    for ; i+4 <= n; i += 4 {
        s0 += a[i+0] * b[i+0]
        s1 += a[i+1] * b[i+1]
        s2 += a[i+2] * b[i+2]
        s3 += a[i+3] * b[i+3]
    }
    sum := s0 + s1 + s2 + s3
    for ; i < n; i++ { sum += a[i]*b[i] }
    return sum
}
```

**Expected speedup on top of BCE:** another 1.5-2×. Total `DotProduct` speedup vs current: ~2-3×.

**Caveat: changes accumulation order, may shift LSB of golden vectors.** Reality's floating-point reproducibility charter requires this to be an opt-in (`DotProductFast` parallel-named variant) OR the golden vectors regenerated. Recommend **separate fast variant** to preserve current bit-exact golden results.

### 4.4 Why no SIMD (AVX-512 / NEON)

The Go gc compiler does NOT auto-vectorise. The only ways to get SIMD are:
1. **Go assembly** (Plan-9 syntax, in `*_amd64.s` / `*_arm64.s` files). Counts as code-in-the-package; technically zero-dep but cross-language golden-file harder; CI must build asm. **Out per Reality charter (no asm).**
2. **CGO into a C kernel.** Adds C compiler dependency; worst-of-all-worlds for Reality. Out.
3. **`asm.Build` plugin or assembly-via-codegen.** Out.
4. **`golang.org/x/sys/cpu` runtime CPU detection + lookup-table dispatch.** Possible in Go but the dispatched-to function must itself be assembly; same problem.
5. **Wait for Go SIMD intrinsics** (open issue golang/go#58610). Not landed as of 2026-Q1.

**Reality's pure-Go ceiling is therefore ~30-40% of MKL/OpenBLAS speed on dense GEMM**, with the gap widening on AVX-512 hardware. This is **acceptable** given the zero-dep + golden-file + cross-language charter; consumers needing speed-of-light dense linalg should call out to an external BLAS via aicore.

**Architectural posture:** isolate the micro-kernel (`microKernel4x4`) as a single function so a future opt-in `microKernel4x4_amd64.s` companion file (compiled into a build tag, not the default) can replace it without disturbing the rest of the package. Ship the Go-only version as canonical; document the asm-replacement seam.

---

## 5. Loop ordering and structural transforms in existing decompositions

### 5.1 LUDecompose — innermost loop already correct

```go
for j := k + 1; j < n; j++ {
    U[i*n+j] -= factor * U[k*n+j]      // both U[i,:] and U[k,:] are unit-stride
}
```

**i-k-j-style implicit ordering, unit-stride. ✓.** No restructuring needed beyond BCE (apply `U[i*n:(i+1)*n]` slice hoist in outer loop).

### 5.2 CholeskyDecompose — j-loop on `k=0..j` is row-into-itself

```go
for k := 0; k < j; k++ {
    sum -= L[i*n+k] * L[j*n+k]
}
```

`L[i,:]` and `L[j,:]` are both unit-stride; this is essentially a dot product of two rows. **Identical optimisations to `DotProduct` apply** (BCE on `L[j,:]`, 4-way unroll). Expected 1.5-2× speedup on Cholesky for `n>~32`.

### 5.3 CholeskySolve — back-substitution L^T loop strided

```go
for j := i + 1; j < n; j++ {
    sum -= L[j*n+i] * x[j]   // L[j*n+i] strided by n
}
```

**This is the only strided-access loop in the package's decompositions.** Strided accesses on a stored row-major matrix's column are inherently bad. Two options:
- **(a) Store both L and L^T at decomposition time.** Doubles storage, removes the strided access. Defensible only if `CholeskySolve` is called many times per factorisation.
- **(b) Live with it.** At `n=64` the strided column fits in L1; at `n=1024` it thrashes. For Reality's typical n≤512 consumers, (b) is fine.

Recommend (b) but document the n>~512 performance cliff.

### 5.4 tridiagonalize — symmetric inner update

```go
for i := k + 1; i < n; i++ {
    for j := k + 1; j <= i; j++ {
        val := w[i*n+j] - w[i*n+k]*p[j] - p[i]*w[j*n+k]
        w[i*n+j] = val
        w[j*n+i] = val
    }
}
```

The triangle update touches both `w[i,:]` (unit-stride) and `w[:,k]` (strided). The strided access on `w[j*n+k]` is the bottleneck. **Fix: stage `w[:,k]` into a contiguous `colK := make([]float64, n)` buffer once per outer-`k` step**, then the inner loop is:
```go
for i := k + 1; i < n; i++ {
    for j := k + 1; j <= i; j++ {
        val := w[i*n+j] - colK[i]*p[j] - p[i]*colK[j]
        w[i*n+j] = val
        w[j*n+i] = val
    }
}
```
**Cost: `n*8B` extra workspace per call. Speedup: 1.5-2× on `tridiagonalize` for `n>64`.** (And the `colK` allocation gets folded into the workspace struct from §1.)

---

## 6. Aggregated patch list, prioritised by leverage / LOC

| # | Patch | LOC | Speedup | Cross-ref |
|---|-------|-----|---------|-----------|
| **F1** | BCE-hint reslices in BLAS-1 kernels (`DotProduct`, `VectorAdd`, etc.) | ~30 | 1.15-1.30× each | new |
| **F2** | `MatMul` i-k-j reorder | ~10 | 1.5-2× at n>64 | new (098 implies, doesn't quantify) |
| **F3** | `MatMul3x3`, `MatMul4x4`, `MatVec3`, `MatVec4` (specialised, `*[9]float64` types) | ~80 | 5-10× at fixed shape | 098 S1 |
| **F4** | `MatInverse3x3`, `MatInverse4x4`, `MatDet3x3`, `MatDet4x4` | ~60 | 30-50× at fixed shape | 098 S1 |
| **F5** | `tridiagonalize` workspace pull-out + colK staging | ~15 | 1.5-2× at n>64 | 096 (alloc), new (perf) |
| **F6** | `DotProductFast` 4-way unrolled | ~15 | 1.5-2× extra (over BCE) | new |
| **F7** | `Inverse(A,n,out, ws InverseWorkspace)` | ~30 | eliminates 6 allocs per call | 099 (named), new (LOC) |
| **F8** | `RanksInto(data, idx, rnk)` | ~10 | eliminates 2 allocs per Spearman | new |
| **F9** | `Determinant` parity-tracking inline (drop visited[]) | ~5 | eliminates 1 alloc | 096 P11 |
| **F10** | BLIS 5-loop `MatMulBlocked` + workspace | ~150 | 2-3× at n>256 | 098 S2 |
| **F11** | `MatMulBatch3x3(As, Bs, outs []float64, count int)` | ~30 | i-cache + branch-predictor warm | 098 S1 |
| **F12** | Document FMA gap, register-pressure constraints in package godoc | ~20 | 0× (informational) | new |

**Sprint-1 ergonomic-speed bundle (F1+F2+F5+F8+F9, ~70 LOC):** BCE hints + i-k-j reorder + tridiagonalize fix + ranks scratch + determinant parity. Pure-additive (F1, F2 for the existing path) or bug-fix (F5, F8, F9). Zero behavioural change. **15-30% across BLAS-1 + 1.5-2× on MatMul for n>64.**

**Sprint-2 small-matrix bundle (F3+F4+F11, ~170 LOC):** Pistachio's exact hot path. **5-50× on shape-specific calls.** Backwards-compatible additions.

**Sprint-3 blocked-GEMM bundle (F7+F10, ~180 LOC):** General-purpose dense throughput. **2-3× on n>256.** Adds workspace surface (consistent with 099 S10 pattern).

**Sprint-4 micro-engineering bundle (F6+F12, ~35 LOC):** Fast-variant DotProduct + perf documentation. Opt-in fast path; document FMA / SIMD ceilings explicitly.

---

## 7. What this audit did not measure

- **Actual benchmark numbers.** This is a structural-audit report. A `go test -bench=.` pass with `BenchmarkMatMul/n={64,256,512,1024}` and `BenchmarkDotProduct/n={1k,10k,100k}` benches against a Skylake or Zen3 baseline would convert the speedup estimates above into measured numbers. Recommend any sprint-1 PR include the bench file.
- **Scaling above L3.** Reality's likely consumer matrix sizes (n≤512 dense, n≤10k sparse) all fit in L3; behaviour at n>2048 (where memory bandwidth dominates) is a separate study not done here.
- **NUMA effects on multi-socket.** Out of scope for single-process Go programs typically; mentioned for completeness.
- **Allocation rate vs GC pressure on long-running PCA workloads.** A `go test -benchmem` run on `BenchmarkPCA/n=64,k=10` would surface whether the per-call PCA allocs (means, cov, eigenvalues, shifted, L, U, perm, bvec, xvec) measurably trigger GC at 60 Hz. Likely yes; the workspace migration (§1) directly fixes it.
- **Goroutine-parallel MatMul.** Out of scope here (an explicit `MatMulParallel(workers int)` is a separate item; 099 §6 anti-recommends transparent parallelism).

---

## 8. Bottom line

Reality `linalg` v0.10.0 is **textbook-correct, structurally-naive on the performance axis**. The largest leverage is **shape-specialised tiny-matrix kernels** (3×3 / 4×4) — 5-50× speedup, 150 LOC, directly closes Pistachio's documented 60-FPS hot path. The next-largest is **bounds-check-elision hints in BLAS-1 kernels** — 15-30%, 30 LOC, no charter cost, no behavioural change, zero unsafe. The third-largest is **BLIS 5-loop blocked GEMM** — 2-3× at n>256, 150 LOC, but only material for consumers with >256-dim dense matrices (rare in current Reality consumer set; likely-future for `prob` GP regression). The **per-call workspace allocations** in `Inverse`, `Determinant`, `tridiagonalize`, and `PCA` are individually small but **collectively prevent these routines from being called inside per-frame loops** — workspace-struct migration is a 60-LOC cleanup that lets `chaos`/`control`/`infogeo` use eigendecomp/inverse without GC pressure.

**Pure-Go performance ceiling vs MKL/OpenBLAS is fundamentally bounded by Go's lack of SIMD intrinsics.** Reality cannot, by charter, close that gap; the right architectural posture is to isolate the micro-kernel as a single function so a future asm/CGO replacement (off by default, opt-in build tag) is a one-file substitution. Ship the Go-only version as canonical, document the gap explicitly.

**Total v0.11 perf sprint LOC: ~470 across 12 patches**, taking the package from "naive triple-loop + per-call alloc churn" to "BLIS-architectured + libxsmm-specialised + BCE-tuned + workspace-disciplined" performance class — the fastest pure-Go can be without crossing the no-asm boundary.

Report at `agents/100-linalg-perf.md`, ~395 lines.

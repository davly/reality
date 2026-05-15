# 099 | linalg-api: Matrix view types, in-place vs copy semantics, broadcasting

**Headline:** Reality `linalg` v0.10.0 ships a *deliberately* type-free, view-free, broadcast-free C-style API — every matrix is a `[]float64` + `(rows, cols)` tuple, every output is caller-allocated, storage is row-major, transpose materialises, no broadcasting, no slicing, no `M.At(i,j)`. This API shape is *correct for the stated charter* (zero-alloc Pistachio 60-FPS hot paths, golden-file portability across Go/Python/C++/C#) but ships seven concrete ergonomic frictions that are individually small (~10-50 LOC each) and collectively define the package's developer-experience ceiling. **The single most-leverage API addition is a `MatView{data, rows, cols, rowStride int}` non-owning slice type (~30 LOC)** — it opens zero-copy sub-matrix access (block decompositions, panel-loop GEMM kernels, training-fold splits without `make+copy`), it is a strict superset of the current `[]float64+rows+cols` triple (current callers pass stride==cols and behave identically), it is the prerequisite for the libxsmm-style 3x3/4x4 specialisations 098 ranked highest-ROI, it is the prerequisite for the cache-blocked GEMM 098 ranked second, and it does *not* commit the package to broadcasting or expression templates — it merely standardises the indexing convention everyone is currently inlining. **Cross-reference vs siblings:** 096 (numerics — norms, cond, refinement), 097 (missing surface — QR/SVD/sparse/Krylov), 098 (SOTA porting — libxsmm/blocked-GEMM/BDCSVD); this report is API-shape-only, no overlap.

---

## 1. Current API shape — eight observations

### 1.1 Matrix type: NONE — flat `[]float64` + `(rows, cols int)` tuple

Every matrix function in the package takes the matrix as **three positional arguments**: the data slice, the row count, and the column count, with the column count omitted when implied by squareness (e.g. LU/Cholesky take `n` only). There is **no `Matrix` struct, no view type, no shape type, no transposed-view marker, no aliasing flag, no row-stride field**.

Examples (from `matrix.go`, `decompose.go`):

```go
MatMul(A []float64, aRows, aCols int, B []float64, bCols int, out []float64)
MatTranspose(A []float64, rows, cols int, out []float64)
LUDecompose(A []float64, n int, L, U []float64, perm []int) bool
CholeskyDecompose(A []float64, n int, L []float64) bool
PCA(data []float64, nSamples, nFeatures, nComponents int, components, explained []float64) float64
```

This is the **gonum/lapack pure-Go convention** without the `mat.Dense` wrapper — closest comparable in the Go ecosystem is the *lower* `gonum/internal/asm/f64` layer, not the user-facing `gonum/mat` layer. It is also the **LAPACK Fortran ABI** without leading-dimension (LDA) parameters.

**Verdict:** *correct given charter.* No struct = no method-call indirection, no GC pressure, no copy semantics surprises, easy to translate to C++/C# golden-file validators (a `[]float64` is a `double*`+`size_t` everywhere). The cost is everything below.

### 1.2 Storage order: row-major (NOT LAPACK column-major)

Every function computes `A[i*cols+j]`. This matches numpy/Eigen-default/Pistachio/Go-idiomatic, **opposes** LAPACK/Fortran/Julia/Matlab/Eigen-`ColMajor`. **Implication for golden-file portability:** Python/NumPy default ('C order') matches; C++ Eigen default matches but `Eigen::ColMajor` doesn't; C# `MathNet.Numerics` defaults to *column-major* (Fortran legacy) — golden vectors must be carefully serialised as flat row-major to round-trip.

**No documentation** of storage order in any function godoc beyond `vector.go:5` package-doc one-liner ("row-major order"). **Recommendation:** every function godoc should state "stored row-major", and every golden-vector JSON should include `"layout": "row-major"` as a self-describing key (~5 LOC fix in `testutil` + a doc-pass).

### 1.3 In-place vs copy: caller-allocates *exclusively*; **no `*Into` / `*New` pair**

Every output buffer is the *last* argument and the caller pre-allocates. There is **no allocating sibling** — no `MatMulNew(A, B) []float64` for one-shot use, no `IdentityNew(n) []float64`, no `MatTransposeNew(A) []float64`. This forces every caller to write the same three-line preamble:

```go
out := make([]float64, aRows*bCols)
linalg.MatMul(A, aRows, aCols, B, bCols, out)
// use out
```

across **every** call site. For the Pistachio 60-FPS hot path this is *correct* (allocation in inner loops would dominate). For the **library-evaluation / one-shot / scripting / test / golden-generation** path this is repetitive boilerplate — and the package **does** have a one-shot path (its golden-file tests, `PCA` allocates internal workspace, `Inverse` allocates `L+U+perm`). **The convention is inconsistent:** `Inverse`, `Determinant`, `PCA`, `QRAlgorithm` all internally allocate workspace (`make([]float64, nn)` etc.) — they are *not* hot-path-zero-alloc despite the docstrings claiming the package is.

**Recommendation:** adopt the **gonum dual-convention** — every primary `MatMul`-style function takes `out` as last arg (current behaviour kept), every primary function gains an allocating sibling `MatMulNew(A, aRows, aCols, B, bCols) []float64` (one-line wrapper over `make+call`, ~5 LOC each, ~12 functions = ~60 LOC). Caller picks: hot-path uses `MatMul(...,out)`, one-shot uses `out := MatMulNew(...)`. Golden tests trivially adopt `New` form (already do `make+call` inline). **This is purely additive, zero behavioural change.**

Alternative naming: `MatMulInto(out, A, ...)` (LAPACK-style out-first) vs `MatMul(A, ..., out)` (current, out-last). Reality picked out-last; gonum picks `dst` as receiver method or first arg. **Don't change the existing convention** (would break callers); just add `New` siblings.

### 1.4 Sub-matrix views: NONE — slicing requires `make+copy`

There is no way to extract `A[1:3, 2:5]` as a 2x3 view onto the parent storage. The only mechanism available is allocate-and-copy:

```go
sub := make([]float64, 2*3)
for i := 0; i < 2; i++ {
    for j := 0; j < 3; j++ {
        sub[i*3+j] = A[(1+i)*aCols + (2+j)]
    }
}
```

inline at every call site. This blocks **every blocked algorithm** (recursive blocked LU/QR/Cholesky 098 S4, BLIS-style 5-loop GEMM 098 S2, panel-factorisation in QR 097 T1 #3, training/test splits in PCA, cross-validation k-fold). **The single most-impactful API addition is `MatView`** — see §2 below.

### 1.5 Transpose: materialises (not lazy)

`MatTranspose(A, rows, cols, out)` writes `out[j*rows+i] = A[i*cols+j]` into a freshly-allocated buffer. There is **no transposed-view** — no `MatT{base, rows, cols}` that flips the index lookup, no `MatMulT(A, B^T)` that consumes a transpose flag. Consequence: `MatMul(A, A^T)` (Gram matrix, every covariance-matrix consumer including `correlation.CovarianceMatrix` and `pca.PCA`) requires materialising `A^T` as a full `rows*cols` allocation.

**Cost analysis:** for the golden-file consumer count (PCA covariance: 1 call/PCA invocation, infrequent), zero. For Pistachio per-frame transforms (`R^T * v` for camera rotation), 9 floats — also zero. **For the BLIS-blocked-GEMM SOTA path (098 S2)**, transpose-on-pack is the *standard* pattern: pack `A` and `B^T` into `mc x kc` panels, the transposition is fused into the pack — so a "transpose flag" on `MatMul` is *the* correct interface. Recommendation: add `MatMulFlags(A, aRows, aCols, transA bool, B, bRows, bCols, transB bool, out)` as the v1.1 signature once blocked GEMM lands; current `MatMul` becomes a 1-line shim. **~30 LOC net.**

### 1.6 Concatenation, stacking: NONE

There is no `MatHStack`, `MatVStack`, `MatStack`, `MatBlock` constructor. Building a block matrix `[[A, B], [C, D]]` requires manual offset arithmetic at the call site. **This is acceptable** — these are rare operations, they always involve allocation (so the zero-alloc charter doesn't apply), and a 4-function suite is ~80 LOC additive. Not high priority but a clean v1.1 add.

### 1.7 Random access: NO `M.At(i,j)` — raw `A[i*cols+j]` everywhere

Every consumer (including the package's own internals — see `decompose.go:50` `U[k*n+k]`, `eigen.go:139` `A[k*n+l]`, `pca.go:177` similar) inlines the row-major index arithmetic. **This is the cost of having no `Matrix` struct.** Pros: zero method-call overhead, fully transparent to the optimiser. Cons: every reader of every consuming function does the index-arithmetic-decoding-in-their-head step; one-character bugs (`i*n+j` vs `j*n+i`) are silent and only catch on golden-file diff; transposing the storage convention (e.g. for a hypothetical `ColMajorMatMul`) requires editing every line.

**Mitigation without sacrificing zero-overhead:** publish a **convention helper** `idxRM(rows, cols, i, j int) int { return i*cols + j }` documented as "compiler will inline; use this in non-hot-path code for readability; raw `i*cols+j` is fine in inner loops". And/or publish a row-iterator helper:

```go
// Row returns a slice aliasing row i of an rows x cols row-major matrix.
func Row(A []float64, rows, cols, i int) []float64 { return A[i*cols : (i+1)*cols] }
```

— this is **already trivially safe** for row-major (rows are contiguous), is `~3 LOC`, eliminates index-arithmetic in 80% of decomposition inner loops, and is **the closest reality can get to a view type without breaking the flat-slice charter**. Strong recommendation: ship `Row` as v0.11.

Note: the **column** equivalent is *not* equally cheap — columns in row-major are non-contiguous, so `Col` would need to be either an `[]float64` copy or a stride-aware iterator (which requires a view type, see §2). This asymmetry is itself an argument for `MatView`.

### 1.8 Broadcasting: NONE

There is no scalar-add-to-matrix, no row-vector-add-to-matrix-rows, no column-vector-add-to-matrix-columns. The package is **deliberately non-broadcasting**: `MatAdd` requires identical-shape operands, `MatScale` is the only "scalar op" form, vector-vs-matrix mixing simply doesn't exist. PCA mean-centering (`pca.go`) hand-rolls the row-broadcast loop inline.

**Verdict:** *correct given charter.* Broadcasting has well-known footguns (numpy's `(N,)` vs `(N,1)` vs `(1,N)` shape-promotion bugs, the worst of which is "right-aligned shape match" silently doing the wrong thing). Cross-language golden-file portability would multiply by 4 (Go/Python/C++/C# all need to agree on shape-promotion rules). **Do not add broadcasting**. Instead, add a **named explicit-broadcast suite** (~8 functions, ~60 LOC):

```go
MatAddRow(A, rows, cols, row, out)   // out[i,j] = A[i,j] + row[j]
MatAddCol(A, rows, cols, col, out)   // out[i,j] = A[i,j] + col[i]
MatAddScalar(A, s, out)              // out[i] = A[i] + s
MatMulRow(A, rows, cols, row, out)   // out[i,j] = A[i,j] * row[j]   (column scaling)
MatMulCol(A, rows, cols, col, out)   // out[i,j] = A[i,j] * col[i]   (row scaling)
// + Sub variants
```

Each is ~7 LOC, the names are self-describing, no shape-promotion magic. This unblocks PCA mean-centering, Z-score normalisation, cosine-similarity-of-rows, k-means centroid update, every common ML preprocessing step. **High ROI relative to LOC.**

---

## 2. The single most-leverage API addition: `MatView`

```go
// MatView is a non-owning view of a row-major matrix. The view may be a
// sub-rectangle of a larger matrix; in that case rowStride > cols and the
// underlying data slice extends beyond what the view exposes.
//
// For a contiguous matrix (no slicing), rowStride == cols.
//
// At(i, j) returns the (i, j) element (0 <= i < rows, 0 <= j < cols).
// All linalg functions accept either a flat []float64 + (rows, cols) triple
// (rowStride implied = cols) or a MatView; the View form is required when
// rowStride != cols (sub-matrices, transposed views, panels).
type MatView struct {
    data      []float64
    rows      int
    cols      int
    rowStride int
}

func NewMatView(data []float64, rows, cols int) MatView { /* rowStride = cols */ }
func (m MatView) At(i, j int) float64                   { return m.data[i*m.rowStride+j] }
func (m MatView) Set(i, j int, v float64)               { m.data[i*m.rowStride+j] = v }
func (m MatView) Sub(r0, c0, r1, c1 int) MatView        { /* zero-copy slice */ }
func (m MatView) Row(i int) []float64                   { /* contiguous, alias */ }
// Col(j) deliberately returns a strided iterator, not a slice — caller copies if needed.
```

**Total: ~30 LOC of struct + helpers.** Then offer **opt-in view-aware variants** of the workhorse routines as `MatMulV(A, B, out MatView)` etc. (each ~20 LOC because the inner loop must use `data[i*rowStride+j]` instead of `data[i*cols+j]` — single-character change in ~6 places per routine). The flat-slice form is **kept** as the canonical zero-alloc path; the View form is **added** for slicing/blocking/panel use cases.

**Why this is highest leverage:**

1. **Strict superset:** `NewMatView(A, rows, cols)` is bit-identical to the flat-slice triple — every existing caller can switch with no behavioural change.
2. **Unblocks 098 S1 (libxsmm small-matrix) directly:** `MatMul3x3(A, B, out MatView)` lets a transform pipeline pass `worldMatrices.Sub(i, 0, i+1, 16)` into the kernel without copy.
3. **Unblocks 098 S2 (BLIS blocked GEMM):** the inner loop is *defined* on strided sub-blocks; without `MatView` the implementation can't be written cleanly.
4. **Unblocks 097 T1 #3 (QR):** Householder reflectors update *trailing sub-matrices*; the canonical loop is "for each panel, reflect (A.Sub(k, k, m, n))".
5. **Cross-language golden file impact: zero.** Golden vectors stay flat-slice; only the *Go API surface* gains the view type.

**Risk:** introducing a struct in a "no-struct" package crosses a charter boundary. Mitigation: keep flat-slice functions canonical and fully tested; the View form is documented as "for blocked/sliced workloads"; the View struct is a *value* type (no pointer, no GC overhead, copies cheaply by value as 4 words = 32 bytes).

---

## 3. Seven concrete API frictions, ranked

| # | Friction | Fix | LOC | Cross-ref |
|---|----------|-----|-----|-----------|
| 1 | No sub-matrix view → blocked algorithms can't be written cleanly | `MatView{data, rows, cols, rowStride}` + `MatMulV` family | ~30 + ~20/fn | 098 S1, S2, S11; 097 T1 #3 |
| 2 | No allocating sibling → `make+call` boilerplate at every call site | Add `MatMulNew`, `IdentityNew`, `MatTransposeNew`, ... | ~5/fn × ~12 = ~60 | n/a |
| 3 | Transpose materialises → Gram matrices double allocation | `MatMulFlags(transA, transB)` v1.1 signature; current `MatMul` becomes shim | ~30 | 098 S2 |
| 4 | No row helper → index arithmetic inlined everywhere | `Row(A, rows, cols, i) []float64` (3-line alias) + doc convention `idxRM` | ~10 | n/a |
| 5 | No explicit broadcast helpers → PCA centering hand-rolled | `MatAddRow/Col/Scalar` + Sub/Mul variants (8 fns) | ~60 | 097 PCA path |
| 6 | No concat/stack → block matrices manual | `MatHStack/VStack/Block` (3 fns) | ~80 | low priority |
| 7 | Storage order undocumented per-function | Doc-pass + golden JSON `"layout"` key | ~10 + JSON schema | 097 cross-lang |

**Sprint-1 ergonomics bundle** (#1+#2+#4+#7, ~140 LOC): unblocks blocked-GEMM/SVD/QR future work, drops boilerplate at every existing call site, ships row-iterator helper, doc-pass on storage order. **Strict additivity, zero behavioural change to any existing function, zero new allocations on any existing path.**

**Sprint-2 ergonomics bundle** (#3+#5, ~90 LOC): transpose-flag GEMM enables blocked-GEMM SOTA path, broadcast helpers unblock common ML preprocessing.

---

## 4. Storage-order specifics (charter-relevant)

Reality is **row-major**. LAPACK is **column-major**. C++ Eigen defaults to **column-major** but accepts `Eigen::RowMajor`. NumPy defaults to **row-major** ('C order'). C# MathNet defaults to **column-major** (Fortran legacy). **For golden-file portability across all four target languages**, the JSON test vectors must be unambiguously row-major, and the consumer adapters must explicitly set order.

Specific portability watch-outs found in current package:

- `linalg/testdata/linalg/matrix_multiply.json` (referenced by `linalg_test.go:27`) presumably stores `A`, `B`, expected `out` as flat arrays; if a downstream C# port reads them through `DenseMatrix.OfArray(values, rows, cols)` (which expects column-major), the test will silently *appear* to pass for square matrices but fail for non-square. **Recommendation:** add `"layout": "row-major"` to every linalg golden JSON header (mechanical 5-LOC change in `testutil` + ~14 JSON file touches).

- `MatTranspose` golden vectors implicitly test layout: if they pass in Go (row-major) but fail in C# (column-major default), the bug is downstream config, not Reality. Cross-port validation should explicitly test a non-square matrix to catch this.

---

## 5. Comparison snapshot vs comparable libraries

| Library | Matrix type | Sub-views | Storage | In-place | Broadcast | Transpose |
|---------|------------|-----------|---------|----------|-----------|-----------|
| reality (current) | `[]float64+r,c` | none | row-major | always | none | materialise |
| gonum/mat | `*mat.Dense` w/ stride | `mat.Slice` | row-major | `Mul(a,b)` and `MulVec` | none | `mat.Transpose` adapter (lazy) |
| numpy | `ndarray` (strides) | `arr[1:3, 2:5]` zero-copy | both ('C'/'F') | `np.add(a,b,out=c)` | yes (right-align rules) | `.T` lazy view |
| Eigen3 C++ | `Matrix<T,M,N>` w/ `Stride<>` | `.block(r,c,h,w)` zero-copy | both | expression templates | broadcast via `colwise()/rowwise()` | `.transpose()` lazy expr |
| Julia | `Array{T,2}` w/ strides | `view(A, 1:3, 2:5)` | column-major | `mul!(C, A, B)` | broadcasting `.+` operator | `A'` lazy adjoint |
| MathNet C# | `Matrix<double>` | `.SubMatrix` (copies) | column-major | several | none | `.Transpose()` materialise |
| LAPACK C ABI | `double*+ld` | by leading-dim arithmetic | column-major | always | none | `'T'`/`'N'` op flags |

**Reality's API shape is closest to LAPACK C ABI** (caller-allocates, no struct, no broadcast) but with **row-major** instead of column-major and **without** the leading-dimension parameter — which is exactly the API gap `MatView` fills (LD ≡ rowStride). This is the *natural target shape* for a zero-dep golden-file-portable Go linalg core; adopting `rowStride` brings reality's API to **exact LAPACK-C parity modulo the row-major flip**, which makes future SOTA porting (098 S4 LAPACK 3.x recursive blocking, S2 BLIS) one-to-one rather than approximate.

---

## 6. What to NOT add

Explicit anti-recommendations (each with rationale):

- **No expression templates / lazy-eval.** Go has no operator overloading; emulating Eigen's `D = A*B + C` fusion would require a builder API (`Expr().Mul(A,B).Add(C).Eval(out)`) that's *more* boilerplate than the direct call. (098 noted same conclusion.)
- **No `interface{}` Matrix abstraction.** Would force every inner loop through a method-call (~3-5x slowdown observed in gonum benchmarks for the interface-dispatch path vs direct slice).
- **No NumPy-style broadcasting.** Shape-promotion footguns + 4-language golden-file consistency cost. Ship explicit `MatAddRow/Col/Scalar` instead (§1.8).
- **No column-major variant.** Picks a fight nobody asked for; row-major matches Go idiom + numpy + Pistachio.
- **No reference-counted shared storage.** Go's GC handles aliasing fine; `MatView` carries a `[]float64` slice header which already aliases backing array correctly.
- **No `Matrix3x3` / `Matrix4x4` distinct types.** Keep dimension-polymorphic; rely on libxsmm-style specialised *functions* (`MatMul3x3`) not specialised *types* — the latter doubles the API surface.
- **No `Vector` type distinct from `[]float64`.** Current package uses `[]float64` for vectors throughout; consistent with Go idiom; consistent with golden-file flat-array serialisation.

---

## 7. Sprint plan (additive only, no behavioural change to existing surface)

**Sprint-1 (~140 LOC, unblocks blocked-algorithm future work):**
1. `MatView{data, rows, cols, rowStride}` + `NewMatView`, `At`, `Set`, `Sub`, `Row` (~30)
2. `MatMulV`, `MatVecMulV`, `MatTransposeV`, `MatAddV`, `MatScaleV` view-aware variants (~120 — 5 fns × ~24 LOC each, mostly `data[i*rowStride+j]` substitution)
3. Allocating siblings: `MatMulNew`, `IdentityNew`, `MatTransposeNew`, `MatAddNew`, `MatScaleNew`, `MatVecMulNew`, `Inverse2`(returns) (~60 — ~8 fns × ~7 LOC)
4. `Row(A, rows, cols, i) []float64` package-level helper (~3)
5. Doc-pass: every function godoc gains "stored row-major" sentence; every golden JSON gains `"layout": "row-major"` header (~10 + schema)

**Sprint-2 (~90 LOC, ergonomic completers):**
6. `MatMulFlags(A, ar, ac, transA, B, br, bc, transB, out)` + 1-line `MatMul` shim (~30)
7. `MatAddRow/Col/Scalar`, `MatSubRow/Col/Scalar`, `MatMulRow/Col` broadcast helpers (~60 — 8 fns × ~7 LOC)

**Sprint-3 (~80 LOC, low-priority completers):**
8. `MatHStack`, `MatVStack`, `MatBlock` concat (~80)

**Total: ~310 LOC strictly additive over the current ~1,440 LOC package, zero existing tests broken, every existing call site unchanged, every existing function unchanged.** All view-form functions can be written by mechanical edit of the flat-form (substitute `i*cols+j` → `i*rowStride+j`) so the implementation cost per function is bounded by a half-page each.

---

## 8. Charter alignment check

Does this proposal violate any of the six design rules in `CLAUDE.md`?

1. **"Golden files are the proof"** — no, view-form and flat-form share golden vectors (vectors are flat regardless of API shape).
2. **"Zero dependencies"** — no, all changes are pure stdlib.
3. **"No allocations in hot paths"** — no, view-form is *less* allocation than flat-form for sliced inputs (no `make+copy` to extract sub-block); allocating `New` siblings are explicit opt-in, not on hot path.
4. **"Every function cites its source"** — preserve docstring discipline; new functions cite Golub-Van Loan / LAPACK Users' Guide.
5. **"Precision documented, not assumed"** — preserve; view-form has identical precision (same arithmetic, different addressing).
6. **"Reimplement from first principles"** — `MatView` is original code, not a wrap of `gonum.Dense`.

**No charter violations.** The view type is the LAPACK leading-dimension parameter rolled into a struct — first-principles construction.

---

## 9. Bottom line

The current `linalg` API is *correct given the zero-alloc/golden-file/Pistachio-60FPS charter*; it ships seven concrete ergonomic frictions of which **`MatView` is by far the highest leverage** because it (a) is a strict superset of the current API, (b) unblocks the three highest-priority items from 098 (libxsmm small-matrix, BLIS blocked GEMM, view abstraction), (c) unblocks two highest-priority items from 097 (QR via panel reflectors, blocked LU), (d) costs ~30 LOC for the type and ~20 LOC per view-aware function variant, (e) crosses no charter boundary, (f) breaks no existing test, (g) brings the package to LAPACK-C-API parity modulo storage flip, (h) adopts the same view abstraction that gonum, NumPy, Eigen, Julia all converge on. **Ship sprint-1 first.** The remaining frictions (no `New` siblings, no transpose flag, no broadcast helpers, no concat) are nice-to-haves whose value is bounded by call-site count, which for a foundation library used through aicore is bounded by a few dozen — so prioritise the structural unlock (views) over the syntactic sugar (siblings, broadcast helpers).

Report at `agents/099-linalg-api.md`, ~390 lines.

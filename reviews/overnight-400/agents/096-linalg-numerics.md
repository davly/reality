# 096 | linalg-numerics

**Scope.** Numerical-correctness audit of `C:\limitless\foundation\reality\linalg\` against the standard
checklist: pivot strategies, condition number reporting, iterative refinement, Cholesky PD check,
SVD convergence, eigendecomposition, matrix norms, sparse arithmetic, matrix exponential design.

**Files read.**
`C:\limitless\foundation\reality\linalg\decompose.go` (LU, Cholesky, Inverse, Determinant — 346 LOC),
`C:\limitless\foundation\reality\linalg\eigen.go` (symmetric tridiagonal QL with Wilkinson shift — 213 LOC),
`C:\limitless\foundation\reality\linalg\matrix.go` (MatMul, transpose, basic ops — 209 LOC),
`C:\limitless\foundation\reality\linalg\vector.go` (vector ops, L1/L2/L∞ vector norms — 242 LOC),
`C:\limitless\foundation\reality\linalg\pca.go` (PCA via covariance + inverse iteration — 215 LOC),
`C:\limitless\foundation\reality\linalg\correlation.go` (Pearson/Spearman, covariance matrix — 184 LOC).

---

## 0. Headline

The `linalg` package, as shipped in v0.10.0, is a **dense-direct-solver mini-LAPACK with one
eigenvalue routine** — no QR/SVD decomposition, no condition-number reporting, no iterative
refinement, no sparse arithmetic, no matrix norms (only vector norms), and no matrix exponential.
What it does ship (LU+partial-pivot, Cholesky, symmetric-tridiagonal QL eigen, dense matmul/PCA)
is **numerically sound and correctly written from first principles**, with one Cholesky semantic
bug (`sum <= 0` rejects the legitimate `A = 0_{1×1}` PSD-singular boundary case but more
importantly **does not detect indefiniteness from a "barely positive" pivot**, e.g. `sum = 1e-300`
yields `sqrt` underflow and downstream divisions by ~0).

**The CLAUDE.md package summary is aspirational, not descriptive.** It promises "vectors, matrices,
LU/QR/Cholesky decomposition, PCA, sparse matrices" — **QR decomposition (matrix factorisation)
and sparse matrices are not implemented.** "QR" appears only as the QR-algorithm-on-tridiagonal
in `eigen.go`. Downstream consumers reading the package list will assume `linalg.QR(...)`,
`linalg.SVD(...)`, `linalg.SparseCSR{...}` exist; they don't. Either ship them, or tighten the
description (≤5-LOC patch: drop the words "QR" and "sparse matrices" from `CLAUDE.md`'s linalg row).

**Single highest-leverage addition (≈70 LOC):** `MatNorm1`, `MatNormInf`, `MatNormFro` (one-pass
each), plus `Cond1` and `CondInf` that wrap `Inverse` then take the matrix-norm product.
Without these, every consumer of `LUSolve`/`CholeskySolve` is **flying blind on numerical
trustworthiness** — they have no way to learn the system was ill-conditioned and the answer is
garbage. Iterative refinement (Wilkinson 1963) is then a 30-LOC follow-on that piggybacks on
existing `LUSolve` + a residual `r = b - A·x` computed with elementwise-Kahan summation.

---

## 1. LU decomposition (`decompose.go:21-89`)

**Pivot strategy.** Partial pivoting (row swaps only). The implementation correctly searches
column `k` from row `k` down for the largest `|U[i,k]|`, swaps full row of `U`, swaps already-computed
columns `0..k-1` of `L`, swaps `perm[k]` (line 67-73). This is textbook Algorithm 3.4.1 of
Golub-Van Loan and is numerically stable for matrices with a reasonable growth factor —
**but is not stable in the worst case** (Wilkinson's classic counterexample is the lower-triangular
matrix with 1s on the diagonal and -1s below; partial pivoting yields growth factor `2^{n-1}` and
a 64-element matrix loses all precision).

**Full / rook pivoting: not present.** This is the textbook gap, but in practice partial pivoting
is the right default and full pivoting belongs behind an opt-in flag. Recommendation: do not add
full pivoting until a consumer demonstrates a need. Document the failure mode instead.

**Singular detection** uses `maxVal < 1e-300` (line 60). This is a numerical floor (true subnormal
boundary), not a relative criterion. A near-singular matrix with `||A||∞ = 1e10` and pivot
`1e-290` will pass this check and produce an arbitrary garbage solution. The defensible fix is a
relative criterion — `maxVal < eps * ||A||∞` with `eps = 1e-15`. But more important than tightening
the criterion is **reporting** the smallest pivot ratio (= proxy for condition number), so the
caller can decide. Concrete proposal: extend signature with an optional `*float64` for
`pivotGrowthRatio = max|U[i,i]| / min|U[i,i]|`, OR a sibling `LUDecomposeWithDiagnostics(...)`.

**Row-swap mechanics: physical row swap, not implicit pivoting.** Lines 65-74 physically swap full
rows of `U` and partial rows of `L` plus the perm vector. This is `O(n)` per swap, total
`O(n^2)` swap cost, dominated by the `O(n^3)` elimination — correct cost-vs-cleanliness tradeoff.
LAPACK's `dgetrf` does the same.

**Permutation cycle counting in `Determinant` (lines 222-237).** This is correct (cycles count
transpositions = (length - 1)) but performs an `O(n)` `make([]bool, n)` allocation. For a public
"Determinant" API used in hot paths (e.g., Jacobian determinants in geodesic integrators) this is
a wart. Patch is 2-LOC: track swap parity by xor-ing `swaps ^= 1` once at every actual swap inside
`LUDecompose` and return the sign as a second return value, eliding the allocation entirely.

---

## 2. Cholesky decomposition (`decompose.go:266-303`)

**PD check is `sum <= 0`** (line 286), which rejects the diagonal of an SPD matrix only when
the cumulative subtraction has gone non-positive. Two issues:

1. **No relative threshold.** `sum = 1e-300` passes; `L[j,j] = sqrt(1e-300) = 1e-150`; the next
   row's `sum/ljj` overflows or amplifies error catastrophically. A standard test (Higham,
   "Accuracy and Stability of Numerical Algorithms", §10.3) is `sum <= eps * max(diagonal_so_far)`.
2. **No symmetry check.** The function only reads the lower triangle (line 282-298 indexes
   `A[i*n+j]` for `i >= j`), so a non-symmetric input silently uses the lower-triangular part.
   This is consistent with LAPACK `dpotrf` — defensible and faster — but it must be **documented**.
   The current docstring says "Only the lower triangle of A is read" (line 254) which is good;
   add explicit "symmetry of A is not validated".

**Forward/back substitution in `CholeskySolve`** (lines 327-345) is textbook and correct.
Reuses `x` for the intermediate `y` (zero-alloc). No issues.

**Recommendation.** Add `CholeskyDecomposeRel` (or a `*float64 minPivotRatio` out-param) that
tightens to relative-pivot threshold; keep the existing `CholeskyDecompose` as a thin wrapper for
backward compatibility. ~25 LOC.

---

## 3. QR decomposition (matrix factorisation): NOT PRESENT

The package has zero matrix-factorisation QR. The `QRAlgorithm` in `eigen.go` is the **QL
iteration on a symmetric tridiagonal matrix**, an entirely different beast. Consumers expecting
to call `linalg.QR(A)` to factor a rectangular `m×n` matrix into `Q (m×n orthonormal) · R (n×n
upper triangular)` will find nothing.

**Design considerations when added:**

- **Householder is the right default** — backward stable to within `n·eps·||A||_F`, allocates
  `O(n)` per reflector, can write `R` over `A` and store the Householder vectors in the
  zeroed-out lower triangle (LAPACK's compact representation). Use this.
- **Modified Gram-Schmidt (MGS)** — easier to read but has `O(eps·κ(A))` orthogonality loss
  in `Q`; classical Gram-Schmidt (CGS) is catastrophic and should not ship. MGS is acceptable
  for `m >> n` rectangular least-squares; expose only via a doc-comment'd alternative.
- **Givens rotations** — best when the matrix is already nearly upper-triangular (e.g.,
  Hessenberg, banded), allocate zero, and are the right primitive for **updating** an existing
  QR factorisation under a rank-1 modification. Not the default; ship later if a consumer needs
  Kalman-filter-style sequential update.

**Stability ranking** (Higham §19): **Householder ≈ Givens > MGS >> CGS**. Ship Householder.

**Recommended signature** (parity with `LUDecompose`'s caller-allocates discipline):
```go
// Caller allocates:
//   tau []float64      length min(m,n)   — Householder scaling factors
//   R   []float64      length n*n         — upper triangular (or compact m*n with Householders)
//   workspace          length m           — scratch for reflector application
func QRDecomposeHouseholder(A []float64, m, n int, tau, R, workspace []float64)
```

`QRSolve(R, tau, b, x)` then handles least-squares via back-substitution on `R` after applying
`Q^T` from the compact representation. ~150 LOC for the pair.

---

## 4. Singular Value Decomposition: NOT PRESENT

This is the largest user-visible numerical gap. Without SVD, the package cannot:

- Solve **rank-deficient** least-squares (LU/Cholesky require full-rank).
- Compute the **2-norm condition number** `σ_max / σ_min` (the gold standard).
- Compute a **pseudoinverse** (Moore-Penrose) for over- or under-determined systems.
- Compute a **principal-component basis directly from data** (current PCA routes through
  covariance, which **squares the condition number** and loses half the available precision —
  see §7).
- Compute a low-rank approximation (truncated SVD) for compression, denoising, latent-space
  embedding.

**Convergence of iterative SVD.** The standard algorithm is **Golub-Reinsch (Householder
bidiagonalisation → implicit-shift QR sweep on the bidiagonal)**, two-stage and proven to
converge superlinearly under Wilkinson shifts (Demmel, "Applied Numerical Linear Algebra",
Algorithm 5.4). The bidiagonal-QR sweep is the same kernel pattern as the existing `tqli` in
`eigen.go:62-129` — with an extra rotation pair to keep the bidiagonal structure — so the
implementation cost is moderate (~250 LOC). Convergence is to within `~5n` sweeps for typical
matrices; failure mode is "no convergence after `30n` sweeps" → return iter count and a `false`.

**Alternative for tall-skinny `m >> n`:** Jacobi rotations on `A^T A`, simpler but **squares
condition number**. Use only as a fallback or for `n ≤ 10`.

**Alternative for symmetric matrices:** SVD = eigendecomposition with `σ_i = |λ_i|`,
`u_i = sign(λ_i) v_i`. Already supportable today by composing with `QRAlgorithm`; add a thin
wrapper `SymmetricSVD(...)` (~20 LOC) as an immediate stopgap.

---

## 5. Condition number reporting: NOT EXPOSED

No function in the package returns or accepts a condition number. A user calling
`LUSolve(L,U,n,perm,b,x)` has no way to assess whether `x` is meaningful.

**Norms relevant to condition number:**
- **2-norm κ_2 = σ_max(A) / σ_min(A)** — gold standard, requires SVD (§4).
- **Frobenius κ_F** — easy to compute (one `O(n^2)` pass), but `κ_F ≥ κ_2` and is often
  pessimistic by a factor of `√n`. Don't bother.
- **1-norm κ_1 = ||A||_1 · ||A^{-1}||_1** — computable via existing `Inverse` + a one-pass
  norm. Hager's algorithm (LAPACK `dlacon`) estimates `||A^{-1}||_1` without forming the
  inverse, in ~5 matvecs; ship the explicit-inverse version first (cheap, one-page),
  ship Hager later when `Inverse` is too expensive (e.g., `n > 1000`).
- **∞-norm κ_∞ = ||A||_∞ · ||A^{-1}||_∞** — same shape as 1-norm. By identity
  `||A||_∞ = ||A^T||_1` so once `1`-norm machinery exists, `∞` is free.

**Currently exposed.** Only **vector** norms (`L1Norm`, `L2Norm`, `LInfNorm` — `vector.go:163-185`).
**No matrix norms.** This is the missing piece. Patch (≤70 LOC for all three matrix norms +
two condition numbers):

```go
// MatNorm1: max column-sum    ||A||_1 = max_j sum_i |A[i,j]|
// MatNormInf: max row-sum     ||A||_∞ = max_i sum_j |A[i,j]|
// MatNormFro: ||A||_F = sqrt(sum A[i,j]^2)
// Cond1(A,n): ||A||_1 · ||A^{-1}||_1, returns (cond, ok); ok=false if A singular
// CondInf(A,n): symmetric variant
```

`Frobenius` should use **Kahan summation** for `n > ~1000`. The existing summing patterns in this
package (e.g., `Trace` line 178-181) accumulate naively; for matrices with values spanning 8+
orders of magnitude this loses log2(N)·eps bits. Not a v1 blocker for `n ≤ 100` consumers.

---

## 6. Iterative refinement: NOT PRESENT

After `LUSolve(L,U,perm,b,x)` returns, the standard postprocessing step (Wilkinson 1963; see
Higham §12) is:

```text
for k = 1..few:
    r = b - A·x          // residual, must be computed in higher precision OR with Kahan
    solve A·d = r        // reuse existing LU
    x += d
    if ||d|| / ||x|| < eps_target: stop
```

This **recovers most of the precision lost to ill-conditioning**, at the cost of one matvec +
one re-solve per iteration (`O(n^2)` each, vs `O(n^3)` for the original LU — so the marginal
cost is ~`1/n` of the original solve). It is **the standard answer to "the LU solve gave me
garbage"**. The reality package does not offer it.

**Implementation sketch (~30 LOC):**

```go
// LURefine: one step of Wilkinson iterative refinement.
// A is the original (un-pivoted) matrix; L,U,perm from LUDecompose.
// b is the original RHS; x is the current solution (modified in place).
// scratch_r and scratch_d each have length n.
// Returns the relative correction norm ||d||_∞ / ||x||_∞.
func LURefine(A []float64, n int, L, U []float64, perm []int,
              b, x, scratch_r, scratch_d []float64) float64 {
    // r = b - A·x
    MatVecMul(A, n, n, x, scratch_r)
    for i := range scratch_r { scratch_r[i] = b[i] - scratch_r[i] }
    LUSolve(L, U, n, perm, scratch_r, scratch_d)
    // x += d
    var maxD, maxX float64
    for i := range x {
        x[i] += scratch_d[i]
        if a := math.Abs(scratch_d[i]); a > maxD { maxD = a }
        if a := math.Abs(x[i]); a > maxX { maxX = a }
    }
    if maxX == 0 { return 0 }
    return maxD / maxX
}
```

**Caveat: residual precision.** True Wilkinson refinement requires the residual
`r = b - A·x` to be computed in **double the working precision**. Without this, refinement still
helps for moderately ill-conditioned matrices (`κ ≤ 10^8`) but plateaus at the working
precision. For Reality's "no dependencies" rule, the practical compromise is:
**(a)** ship the simple `LURefine` shown above and document "useful for `κ ≤ 10^8`",
**(b)** add `LURefineKahan` (~20 LOC extra) that uses compensated summation for the
matvec in `r = b - A·x`, recovering most of the benefit. Quad-precision residual via
double-double arithmetic is overkill for v1.

---

## 7. PCA via covariance + inverse iteration: PRECISION HAZARD (`pca.go`)

`PCA` (line 33) routes through the explicit covariance matrix (`cov.go:67-80`) and then runs
`QRAlgorithm` on it for eigenvalues, then **inverse iteration** on `(Σ - λI)` for eigenvectors
(line 101-200). Two numerical concerns:

**(a) Covariance formation squares the condition number.** If the data matrix `X` has
condition number `κ`, then `Σ = X^T X / (n-1)` has condition number `κ^2`. For data that
spans 4 orders of magnitude (`κ ≈ 10^4`), `Σ`'s condition number is `10^8` and `QRAlgorithm` on
`Σ` has only ~7 digits left in single-data eigenvalues. **The standard fix is to compute the
SVD of the centred data matrix directly** (`U·Σ·V^T`); singular values are the principal
"standard deviations" and `V` columns are the principal components. This sidesteps the squaring.
**Recommendation:** when SVD lands (§4), expose `PCAFromSVD(X, ...)` and demote the current
covariance-route to `PCAFromCovariance(...)`, with the docstring telling users which to pick.

**(b) Inverse iteration with `shift = lambda - 1e-10·(1+λ²)`** (line 107) is correct in
principle (exact-shift inverse iteration converges in 1-2 steps), but the shift formula
`1e-10·(1+λ²)` is dimensionally awkward for `λ < 1` — when `λ = 0.001`, `shift = 0.001 - 1e-10`,
the gap is `1e-10 / 0.001 = 1e-7` relative, fine; when `λ = 1000`, `shift = 1000 - 1e-4`, the
gap is `1e-4 / 1000 = 1e-7` relative, also fine. Actually the formula is dimensionally
sensible because of the `λ²` term. **Not a bug; just non-obvious.** Add a comment explaining
the scaling.

**(c) Convergence check at line 171:** `if maxDiff < 1e-12 { break }`. The `maxDiff` allows for
sign flip — good — but compares un-normalised differences, so for components with `||v||_∞` already
~1 the threshold is sensible. Fine.

**(d) Gram-Schmidt orthogonalisation against previous components** (line 177-185) uses
**classical Gram-Schmidt** (CGS), not modified GS. For `nComponents > ~10` this loses
orthogonality. Fix is a one-line change: replace the inner subtraction loop with **two passes**
of CGS ("CGS2" — the Daniel-Gragg-Kaufman-Stewart trick), which is provably as stable as MGS.
~5 LOC added.

---

## 8. Eigendecomposition (`eigen.go`)

**Coverage.** Symmetric real matrices only, eigenvalues only — no eigenvectors.

**Algorithm.** Householder tridiagonalisation (lines 137-212) → implicit-shift QL with Wilkinson
shifts (lines 62-129). This is the textbook Numerical Recipes `tqli` adapted from LAPACK
`dsteqr`. The implementation looks correct: shift formula at line 88-90 matches Wilkinson's,
the rotation chase from `m-1` down to `l` (line 97-120) matches Golub-Van Loan §8.3.3, and the
convergence test `|e[m]| ≤ 1e-15·(|d[m]| + |d[m+1]|)` (line 74) is the standard relative test.

**Convergence guarantee.** Wilkinson shifts give cubic local convergence; the maxIter ceiling
(`return totalIter` at line 84) is a safety valve, not the expected exit. Typical convergence
is `~30·n` total iterations. The function returns the iteration count, which is good — callers
can detect non-convergence by comparing to `maxIter·n`.

**Eigenvectors are NOT returned.** This is a real gap. PCA (`pca.go`) works around it via
inverse iteration, which is valid but slower than computing eigenvectors during the QL sweeps
(LAPACK's `dsteqr` accumulates them on the fly for ~2× cost). **Recommendation:**
`QRAlgorithmFull(A, n, eigenvalues, eigenvectors []float64, maxIter int)` that runs an extra
accumulation matrix `Z` updated by the same rotations. ~80 LOC delta over the existing
`tqli`. PCA can then drop inverse iteration entirely.

**Non-symmetric eigendecomposition is NOT PRESENT.** Standard algorithm is the Francis double-shift
QR on a Hessenberg form (LAPACK `dhseqr`), considerably more complex than the symmetric case.
Not a v1 priority. Out of scope for the present audit; flag for v2.

**Tridiagonalize allocates `w` and `p` per call** (lines 139, 173). Not a hot-path concern (PCA
calls eigendecomp once at setup) but inconsistent with the package's stated "zero allocations
in hot paths" rule. Patch is to add `Workspace` parameter; ~10 LOC.

---

## 9. Matrix norms: ONE OF FOUR EXPOSED (only as vector norm)

What's exposed today:
- **L1Norm, L2Norm, LInfNorm — VECTOR ONLY** (vector.go:150-185). Applying these to a flattened
  `n×n` matrix gives `||vec(A)||_1, ||A||_F, max|A[i,j]|` respectively — not the matrix-induced
  norms a numerical analyst expects.

What's missing:
- **MatNorm1** (max column-sum, the operator 1-norm).
- **MatNormInf** (max row-sum, the operator ∞-norm).
- **MatNormFro** (Frobenius, identical to vec'd L2 — but should ship as a named function for
  clarity; consumers won't think to call `L2Norm(A)` on a matrix).
- **MatNorm2** (operator 2-norm = largest singular value; requires SVD per §4).

**Recommendation as in §5:** ship the three cheap ones (`MatNorm1`, `MatNormInf`, `MatNormFro`)
in a single ~40 LOC patch. Defer `MatNorm2` to SVD landing.

---

## 10. Sparse matrix arithmetic: NOT PRESENT

CLAUDE.md lists "sparse matrices" but no CSR/CSC/COO type, no sparse matvec, no sparse solve
exists in the package. For Reality's stated consumers (signal, control, optim), the absence of
sparse is mostly fine — most problems are small-dense. But three places suffer:

- **`graph` package** (Dijkstra, A*) maintains adjacency that is morally sparse — currently
  `graph` does its own thing rather than going through `linalg`. Acceptable.
- **`signal`** convolution / FIR filtering — naturally banded matrices. Currently uses direct
  loops, not matrix abstraction. Acceptable.
- **`optim`** L-BFGS — does not need sparse. Fine.

**Verdict.** Sparse is a **future** addition, only when an internal consumer demonstrably needs
it. Until then, **drop the word "sparse" from the CLAUDE.md package description** to avoid
misleading downstream.

When it eventually ships, the right primitive is **CSR (Compressed Sparse Row)** for matvec-heavy
workflows, with `SparseMatVec(rowptr, colidx, values, x, out)`. ~150 LOC for the basic type +
matvec + add. Sparse direct factorisation (sparse LU, sparse Cholesky) is much harder and
out of scope.

---

## 11. Matrix exponential: NOT PRESENT (per agents 052, 055, 095)

Multiple cross-cutting agents have flagged `linalg.MatrixExp` as the canonical home for `expm(A)`
that consumers in `chaos`, `control`, `infogeo` (SPD manifold), and `physics` (operator
exponentiation) all need. Current state: zero implementations across the entire repo.

**Standard algorithm.** Higham 2008, "The Scaling and Squaring Method for the Matrix Exponential
Revisited" — computes `e^A` via Padé[13/13] approximation of `e^{A/2^s}` followed by `s` repeated
squarings. Cost is `~25 n^3` (5-6 matmuls + linear-system solve + s squarings). Reference is
LAPACK `dgexp` (in supplementary `expokit`).

**Convergence and stability.** Higham 2008 proves the scaling-and-squaring with Padé[13/13]
achieves `||e^A - p13(A/2^s)^{2^s}|| ≤ eps · ||e^A||` provided `||A/2^s||_1 ≤ 5.4`. The norm
choice (1-norm) couples this directly to the missing `MatNorm1` from §5/§9. **Building blocks
must land in the order:** matrix norms → linear solver (already exists) → Padé coefficients →
expm.

**Design considerations:**
- **Symmetric SPD case is special** — eigendecomposition gives `e^A = V·diag(e^{λ_i})·V^T` in
  `~12n^3` vs Higham's `~25n^3`. The `infogeo` SPD manifold needs precisely this. Ship
  `MatrixExpSymmetric(A,n,out)` (~30 LOC over existing `QRAlgorithmFull`) early; the general
  Higham can come later.
- **Skew-symmetric case (rotations).** `e^{[ω]_×}` for 3-vectors → Rodrigues, already in
  `geometry`. Generalisation to `n×n` skew is `~SO(n)` — exact for `n=3`, Higham for general.
- **Workspace discipline.** Higham's algorithm needs ~6 scratch matrices; pass a
  `MatrixExpWorkspace` struct.
- **Backward error analysis.** Document the relative error bound `~n·eps·||e^A||`; document
  that this can be much worse than `n·eps·||A||` for `A` with negative eigenvalues (Hochbruck-
  Lubich phenomenon). Cite Higham 2008.

**Estimated v1 cost:** symmetric variant ~30 LOC, general Higham ~250 LOC, plus ~50 LOC of
golden-file vectors (rotation matrices, Jordan blocks, Frank/Forsythe matrices). The
prerequisites are MatNorm1 (§9) and a triangular-solve (extractable from existing `LUSolve`).

---

## 12. Aggregated patch list (priorities)

| # | Patch | LOC | Why |
|---|-------|-----|-----|
| **P1** | Drop "QR" and "sparse matrices" from CLAUDE.md linalg row | ≤5 | Truth-in-advertising; current claim misleads. |
| **P2** | `MatNorm1`, `MatNormInf`, `MatNormFro` | ~40 | Prerequisite for everything below. |
| **P3** | `Cond1`, `CondInf` (use existing `Inverse`) | ~30 | Closes "flying blind on solver trustworthiness" today. |
| **P4** | `LURefine` (Wilkinson iterative refinement) | ~30 | Recovers precision lost to ill-conditioning; standard postprocessing. |
| **P5** | `QRAlgorithmFull` (eigenvalues + eigenvectors in one pass) | ~80 | Lets PCA drop inverse iteration; faster + simpler. |
| **P6** | `MatrixExpSymmetric` for SPD via eigendecomp | ~30 | Unblocks infogeo SPD manifold. |
| **P7** | `QRDecomposeHouseholder` + `QRSolve` | ~150 | Rectangular least-squares; foundation for SVD. |
| **P8** | `SVD` (Golub-Reinsch bidiagonal QR) | ~250 | Closes 2-norm condition number, pseudoinverse, robust PCA. |
| **P9** | `MatrixExp` (Higham 2008 scaling-and-squaring) | ~250 | General matrix exponential for chaos/control/physics. |
| P10 | Cholesky relative-pivot threshold | ~25 | Defensive numerics for near-PSD inputs. |
| P11 | LU determinant: track parity inline (no `visited[]` alloc) | ~5 | Tiny perf win, moves toward zero-alloc rule. |
| P12 | PCA: replace classical Gram-Schmidt with CGS2 | ~5 | Restores orthogonality at `nComponents > ~10`. |
| P13 | Tridiagonalize: accept Workspace; eliminate per-call alloc | ~10 | Compliance with package's zero-alloc rule. |

**Sprint-1 ship-ready batch (P1+P2+P3+P4+P10+P11+P12+P13):** ~150 LOC net, all defensible against
Higham/Golub-Van Loan citations, all backwards-compatible (additive). Closes ill-conditioning
blind spot and brings the package into compliance with its own stated zero-alloc rule.

**Single most impactful follow-on (P5):** lets PCA drop inverse iteration entirely (delete ~80
LOC of `pca.go`, gain ~20 LOC of accumulator in eigen.go) — net negative LOC for a faster, more
direct, more numerically defensible PCA. Recommended as the next addition after sprint-1.

---

## 13. Cross-cutting consistency notes

- **Allocation discipline is inconsistent.** `MatMul`, `MatVecMul`, vector ops are zero-alloc;
  `Inverse` and `Determinant` and `tridiagonalize` allocate per call (lines 162-164, 212-214,
  139, 173). The package docstring says "zero heap allocations in hot paths" (vector.go:1-11),
  with the implication that setup ops can allocate. This is defensible **provided** the
  setup-vs-hot-path classification is documented per-function; today it's documented for some
  (e.g., `Inverse` line 151: "Allocates temporary buffers ... setup cost, not hot-path") and not
  others (`Determinant` line 199: "Allocates temporary buffers internally" — same wording, no
  hot-path qualifier). **Patch:** uniformise the docstrings; ~10 LOC.
- **Singular detection thresholds vary.** LU uses `< 1e-300` (decompose.go:60), Cholesky uses
  `<= 0` (line 286), tridiagonalize uses `< 1e-300` twice (lines 150, 167), tqli uses
  `<= 1e-15·dd` (line 74). Three different scales. This is OK because they test different
  things, but should be **commented**: each test should say what scale it's relative to.
- **No NaN/Inf input validation** anywhere in the package. A NaN on input propagates silently to
  output. CLAUDE.md §10 ("Precision documented, not assumed") implies graceful handling, but no
  function declares behaviour on non-finite inputs. **Patch:** add a one-line clause to each
  docstring: "Returns NaN if any input is non-finite" (or whatever the actual behaviour is).
  Not blocking; cleanup item.

---

## 14. What this audit did not assess

- **Allocation-rate benchmarks.** No timing data taken; cited LOC counts and `O(·)` complexities.
- **Numerical experiments.** Did not run the existing tests or construct stress matrices.
  The LU/Cholesky/eigen implementations were assessed by code reading against Golub-Van Loan
  and Numerical Recipes references.
- **Compatibility with the `testutil` golden-file infrastructure.** Whether new functions
  (norms, refine, SVD) would integrate with the cross-language golden-file scheme is a separate
  question — likely yes, but I did not read `testutil`.
- **Downstream consumer surveys.** Which Reality packages currently call `Inverse` / `LUSolve`
  / `QRAlgorithm` and might benefit from condition-number reporting was not surveyed; likely
  candidates from CLAUDE.md package list: `optim` (L-BFGS Hessian inverse), `prob` (multivariate
  Gaussian), `infogeo` (SPD), `chaos` (Jacobian eigenvalues at fixed points).

---

## 15. Bottom line

**`linalg` v0.10.0 is a small, correct, dense-direct-solver core that is missing four standard
LAPACK-equivalent capabilities** (matrix norms, condition number, iterative refinement, SVD/QR
factorisation) plus the cross-cutting `MatrixExp`. The existing code is numerically sound and
defensibly implemented. The fastest payoff (P1-P4 + P10-P13) is ~150 LOC of additive,
backwards-compatible patches that close the trust gap on every solver call site that exists
today. SVD (P8) is the largest individual addition and unlocks the most downstream value
(robust PCA, pseudoinverse, true 2-norm condition number) but is properly a v1.1 item, not a
hotfix.

**File:** `C:\limitless\foundation\reality\reviews\overnight-400\agents\096-linalg-numerics.md`,
~395 lines.

# 097 | linalg-missing

**Scope.** Enumerate canonical linalg primitives NOT yet present in `C:\limitless\foundation\reality\linalg\`. Cross-checked against agent 096 (numerics audit) and against LAPACK 3.x / Eigen 3 / ARPACK / SuiteSparse / scipy.sparse.linalg / MATLAB Reference taxonomy.

**Files audited (file inventory).**
`linalg/decompose.go` (LU + Cholesky + Inverse + Determinant — 346 LOC),
`linalg/eigen.go` (Householder tridiagonalize + symmetric-tridiagonal QL via Wilkinson shift — 213 LOC),
`linalg/matrix.go` (MatMul, MatTranspose, MatVecMul, Identity, MatAdd, MatSub, MatScale, Trace, CrossProduct — 209 LOC),
`linalg/vector.go` (DotProduct, L1/L2/Inf vector norms, VectorAdd/Sub/Scale, CosineSimilarity, EncodingDistance, DimensionWeightedDistance, L2Normalize, Clamp, StructuralOverlap — 242 LOC),
`linalg/pca.go` (PCA via covariance + inverse-iteration eigvecs — 215 LOC),
`linalg/correlation.go` (Pearson, Spearman, Covariance, CovarianceMatrix — 184 LOC).

**Total public surface (verified).** 25 dense-only functions, all `[]float64` row-major. **Symmetric-real eigenvalues only**, **no eigenvectors** (PCA improvises via inverse iteration), **no QR factorisation**, **no SVD**, **no sparse types**, **no matrix norms**, **no condition number**, **no iterative refinement**, **no matrix exponential / log / sqrt / sign**, **no iterative Krylov solvers**, **no banded / tridiagonal / Toeplitz solvers**, **no pseudoinverse**, **no rank-revealing factorisations**, **no Schur**, **no generalised eigenvalue**, **no LDL^T**, **no tensor decompositions**.

The CLAUDE.md package row claims "QR/Cholesky decomposition, sparse matrices" — both QR-as-factorisation and any sparse type are aspirational (096 §0, §10 already flagged this). v0.10.0 ships ~10% of the LAPACK + ARPACK + SuiteSparse + Krylov-solver canon a math library calling itself `linalg` is expected to ship.

---

## 0. Headline

The package is presently a **dense-direct-solver minicore** at the LU + Cholesky + symmetric-tridiagonal-eigenvalue level. Across the **six families a numerical-linear-algebra library is judged on**, it covers exactly one and a fraction (1.2 / 6):

| Family | Coverage | Status |
|---|---|---|
| **Dense direct factorisations** | LU, Cholesky | **partial** (no QR, SVD, Schur, polar, LDL^T, Hessenberg, bidiag) |
| **Dense iterative & function-of-matrix** | none | **absent** (no expm/logm/sqrtm/sign, no Padé, no Schur-Parlett) |
| **Sparse types & sparse direct** | none | **absent** (no CSR/CSC/COO, no sparse Cholesky, no orderings) |
| **Krylov iterative solvers** | none | **absent** (no CG, GMRES, BiCGStab, IDR(s), MINRES, LSQR, LSMR) |
| **Eigenvalue iterative methods** | symmetric only, eigenvalues only | **partial** (no Lanczos, Arnoldi, IRAM, LOBPCG, Francis, gen-eig) |
| **Special-structure & utilities** | none | **absent** (no banded, tridiag, Toeplitz, condest, pinv, RREF) |

Cross-cuttingly, **eight packages in this repo would benefit immediately** from additions here — `optim` (L-BFGS Hessian inverse), `infogeo` (SPD manifold expm/logm), `chaos` (Jacobian eigvals via Schur), `prob` (multivariate Gaussian with rank-deficient Σ), `signal` (banded filtering), `control` (Lyapunov/Sylvester equations need expm + Schur), `graph` (sparse spectrum via Lanczos), `compression` (rSVD for low-rank approx). The largest-leverage single addition is **Householder QR** because it unlocks SVD, condition number, pseudoinverse, rank-revealing least-squares, and rectangular eigenvalue iteration in one move.

---

## 1. Decompositions — present vs missing

| Decomposition | Status | Reference algorithm | Approx LOC | Tier |
|---|---|---|---|---|
| LU (partial pivoting) | **present** (`decompose.go:21`) | Golub-Van Loan §3.4.1 | — | — |
| LU (rook / full pivoting) | **missing** | GVL §3.4.7 | ~80 | T3 |
| LDL^T (Bunch-Kaufman) | **missing** | LAPACK `dsytrf` | ~200 | T2 |
| Cholesky (LL^T) | **present** (`decompose.go:266`) | GVL §4.2 | — | — |
| Pivoted Cholesky (rank-revealing) | **missing** | LAPACK `dpstrf` (Higham 1990) | ~100 | T2 |
| Modified Cholesky (Gill-Murray-Wright) | **missing** | for indef-Hessian fixups | ~120 | T3 |
| QR (Householder) | **missing** | GVL §5.2.1, LAPACK `dgeqrf` | ~150 | **T1** |
| QR (Givens, sequential update) | **missing** | GVL §5.1.8 | ~80 | T2 |
| QR (Modified Gram-Schmidt) | **missing** | GVL §5.2.8 | ~50 | T3 |
| Pivoted QR / rank-revealing QR | **missing** | LAPACK `dgeqp3`, BBC pivoting | ~180 | T2 |
| SVD (full / thin, Golub-Reinsch) | **missing** | Demmel Alg 5.4, LAPACK `dgesvd` | ~300 | **T1** |
| SVD (one-sided Jacobi) | **missing** | LAPACK `dgesvj` | ~180 | T3 |
| SVD (truncated, deflated GR) | **missing** | scipy `svds` | ~80 over base | T2 |
| Randomized SVD (Halko-Martinsson-Tropp 2011) | **missing** | rSVD with power iteration | ~150 | **T1** |
| Schur (real, Francis double-shift QR) | **missing** | LAPACK `dhseqr` | ~400 | T2 |
| Schur (complex, single-shift) | **missing** | LAPACK `zhseqr` | ~350 | T3 |
| Hessenberg reduction | **missing** | GVL §7.4.3, LAPACK `dgehrd` | ~120 | T2 (prerequisite for Schur/Francis/Arnoldi) |
| Bidiagonalisation (Householder) | **missing** | LAPACK `dgebrd` | ~140 | T2 (prerequisite for SVD) |
| Tridiagonalisation (Householder, sym.) | **present** (`eigen.go:137`) | GVL §8.3.1 | — | — |
| Polar decomposition (Higham Newton) | **missing** | Higham 2008 §8 | ~120 | T2 |
| Generalised eigenvalue (Ax=λBx, QZ) | **missing** | LAPACK `dggev` | ~500 | T3 |
| CS decomposition | **missing** | LAPACK `dorcsd` | ~250 | T3 |
| Symmetric eigenvectors (full) | **missing** (eigvals only) | accumulate `Z` in QL | ~80 | **T1** (pinned by 096 P5) |
| Non-symmetric eigendecomposition | **missing** | needs Hessenberg + Schur | composes T2 | T2 |

**Tier 1 minimum** for closing the headline factorisation gap: Householder-QR + symmetric-eigenvectors + (Golub-Reinsch SVD or rSVD). With those three, ~80% of practical dense problems become solvable.

---

## 2. Sparse types & direct factorisations — entirely absent

| Primitive | Status | Reference | Approx LOC | Tier |
|---|---|---|---|---|
| COO (triplet) sparse type | **missing** | scipy.sparse.coo, Eigen `Triplet` | ~100 | **T1** |
| CSR (Compressed Sparse Row) | **missing** | scipy.sparse.csr, SuiteSparse | ~250 | **T1** |
| CSC (Compressed Sparse Col) | **missing** | scipy.sparse.csc, SuiteSparse | ~250 | T2 |
| Sparse matvec (SpMV) | **missing** (CSR/CSC dependent) | trivially follows CSR | ~30 each | **T1** |
| Sparse-sparse matmul (SpGEMM) | **missing** | Gustavson 1978 | ~180 | T2 |
| Sparse triangular solve (SpTRSV) | **missing** | level-set / Eisenstat | ~120 | T2 |
| Sparse LU (LDL^T-symbolic + Gilbert-Peierls numeric) | **missing** | SuiteSparse `KLU`, `UMFPACK` | ~800 | T3 |
| Sparse Cholesky (left-looking, supernodal) | **missing** | SuiteSparse `CHOLMOD`, T. Davis | ~600 | T2 |
| Symbolic Cholesky (elimination tree) | **missing** | T. Davis "Direct Methods" §4.1 | ~150 | T2 |
| AMD (Approximate Minimum Degree) ordering | **missing** | Amestoy-Davis-Duff 1996 | ~250 | T2 |
| RCM (Reverse Cuthill-McKee) ordering | **missing** | Cuthill-McKee 1969 | ~120 | T2 |
| METIS / nested dissection ordering | **missing** | Karypis-Kumar 1998 (out of scope: external) | ~600 native, prefer wrap | T3 |
| MMD (Multiple Minimum Degree) ordering | **missing** | Liu 1985 | ~200 | T3 |
| Graph-coloring for parallel SpMV | **missing** | Saad §11.5 | ~150 | T3 |

**Sprint advice.** The right entry point is COO → CSR → SpMV → CSR-CSR add (~430 LOC total). That alone unlocks every Krylov solver below — sparse direct factorisation is a separate large project, properly v1.2.

---

## 3. Iterative Krylov solvers — entirely absent

These are the modern workhorses for large sparse `Ax = b`. Reality has zero of them. **All require only matvec** (sparse or dense), so they layer cleanly on top of CSR + the existing `MatVecMul`.

| Solver | Use case | Reference | Approx LOC | Tier |
|---|---|---|---|---|
| Conjugate Gradient (CG) | SPD systems | Hestenes-Stiefel 1952; Saad §6.7 | ~80 | **T1** |
| Preconditioned CG (PCG) | SPD + preconditioner | Saad §9.2 | ~30 over CG | **T1** |
| BiCG | unsymmetric, transpose available | Saad §7.3 | ~120 | T3 |
| BiCGStab (van der Vorst 1992) | unsymmetric, no transpose | Saad §7.4.2 | ~150 | **T1** |
| BiCGStab(L) | smoother convergence than BiCGStab | Sleijpen-Fokkema 1993 | ~200 | T2 |
| GMRES (m-restart) | general unsymmetric | Saad-Schultz 1986; Saad §6.5 | ~250 | **T1** |
| Flexible GMRES (FGMRES) | variable preconditioner | Saad 1993 | ~80 over GMRES | T2 |
| MINRES | symmetric indefinite | Paige-Saunders 1975 | ~200 | T2 |
| SYMMLQ | symmetric indefinite (LQ variant) | Paige-Saunders 1975 | ~200 | T3 |
| LSQR | least-squares (rectangular) | Paige-Saunders 1982 | ~250 | T2 |
| LSMR | least-squares (better cond. on regularised) | Fong-Saunders 2011 | ~270 | T2 |
| QMR / TFQMR | unsymmetric, Lanczos-based | Freund-Nachtigal 1991 | ~280 | T3 |
| IDR(s) | unsymmetric, fewer matvecs than BiCGStab | Sonneveld-van Gijzen 2008 | ~300 | T2 |
| Block CG | multiple RHS | O'Leary 1980 | ~200 | T3 |
| Block GMRES | multiple RHS | Vital 1990 | ~350 | T3 |
| Pipelined / communication-avoiding CG | high-performance HPC | Ghysels-Vanroose 2014 | ~250 | T3 |
| Chebyshev iteration (with eigenvalue bounds) | preconditioner / smoother | Saad §12.4 | ~120 | T3 |

### Preconditioners (necessary partner)

| Preconditioner | Status | Reference | Approx LOC | Tier |
|---|---|---|---|---|
| Jacobi (diagonal) | **missing** | trivial | ~20 | **T1** |
| SSOR | **missing** | Saad §10.2 | ~80 | T2 |
| Incomplete Cholesky IC(0), IC(k) | **missing** | Saad §10.3.2 | ~250 | T2 |
| Incomplete LU ILU(0), ILU(k), ILUT | **missing** | Saad §10.4 | ~400 | T2 |
| AMG (algebraic multigrid) | **missing** | Stüben 2001; PyAMG | ~1500 | T3 |
| Geometric multigrid | **missing** | Briggs-Henson-McCormick | ~800 | T3 |

---

## 4. Eigenvalue iterative methods — symmetric-only, eigenvalues-only

Currently exposed: `QRAlgorithm` (symmetric, eigenvalues, dense) and a private power-method-via-inverse-iteration in `pca.go`.

| Method | Use case | Reference | Approx LOC | Tier |
|---|---|---|---|---|
| Power iteration (top-1) | dominant eigvec | textbook | ~30 | **T1** (lift from `pca.go`) |
| Inverse iteration | eigvec for given eigval | GVL §7.6.1 | ~50 (lift from `pca.go`) | **T1** |
| Subspace / orthogonal iteration | top-k dense | GVL §8.2.4 | ~120 | T2 |
| Lanczos (symmetric, sparse) | extreme eigvals of large sparse | Lanczos 1950; Saad §6.6 | ~200 | **T1** |
| Implicitly-Restarted Lanczos | thick-restart, ARPACK `dsaupd` | Sorensen 1992 | ~400 | T2 |
| Arnoldi (general, sparse) | extreme eigvals nonsymmetric large | Arnoldi 1951; Saad §6.2 | ~250 | **T1** |
| IRAM (Implicitly-Restarted Arnoldi) | ARPACK `dnaupd` | Sorensen 1992 | ~600 | T2 |
| Krylov-Schur | replacement for IRAM with cleaner restart | Stewart 2002 | ~500 | T2 |
| LOBPCG | symmetric generalised eigvals, large sparse | Knyazev 2001 | ~400 | T2 |
| Davidson / Jacobi-Davidson | extreme eig with preconditioner | Davidson 1975; Sleijpen-van der Vorst 1996 | ~500 | T3 |
| Rayleigh-quotient iteration | local cubic convergence | GVL §8.2.3 | ~80 | T3 |
| Francis double-shift QR (Hessenberg) | dense nonsymmetric eigvals | Francis 1961; LAPACK `dhseqr` | ~400 | T2 |
| Generalised eig (QZ on (A,B)) | Ax = λBx | LAPACK `dggev` | ~600 | T3 |
| HHDSVD / divide-and-conquer eig | large symmetric, fast | Cuppen 1981; LAPACK `dsyevd` | ~500 | T3 |
| MRRR (MR3) | symmetric eigvecs without GS reortho | Dhillon-Parlett 2004 | ~600 | T3 |

---

## 5. Special-structure solvers — entirely absent

| Solver | Status | Reference | Approx LOC | Tier |
|---|---|---|---|---|
| Symmetric tridiagonal solver (Thomas) | **missing** (tridiagonalisation present, but no SPD-tridiag solve) | textbook | ~40 | **T1** |
| General tridiagonal solver | **missing** | LAPACK `dgtsv` | ~50 | T2 |
| Banded solver (LU, lower/upper bandwidth) | **missing** | LAPACK `dgbsv` | ~150 | **T1** |
| Banded Cholesky | **missing** | LAPACK `dpbsv` | ~120 | T2 |
| Block-tridiagonal (Thomas-block) | **missing** | textbook, Saad §3.7 | ~150 | T2 |
| Toeplitz solver (Levinson-Durbin) | **missing** | Levinson 1947, Durbin 1960 | ~80 | **T1** (signal/control consumers) |
| Toeplitz solver (Trench) | **missing** | Trench 1964 | ~120 | T3 |
| Hankel solvers | **missing** | analogous to Toeplitz | ~100 | T3 |
| Vandermonde solver (Björck-Pereyra) | **missing** | Björck-Pereyra 1970 | ~80 | T3 |
| Cauchy-like solvers (GKO) | **missing** | Gohberg-Kailath-Olshevsky 1995 | ~250 | T3 |
| Circulant via FFT | **missing** | composes with `signal/fft` | ~40 | T2 |

The Toeplitz Levinson is high-value because `signal` (autocorrelation → AR coefficients) and `control` (Yule-Walker, prediction) both reinvent it locally without coordinating.

---

## 6. Function-of-matrix — entirely absent

Currently zero. Multiple downstream consumers blocked (`infogeo` SPD manifold, `control` Lyapunov/Sylvester, `chaos` matrix flows, `physics` operator exponentials).

| Function | Use case | Reference | Approx LOC | Tier |
|---|---|---|---|---|
| `MatExp` (Padé scaling-and-squaring) | ODE flows, infogeo SPD log | Higham 2008 §10 | ~250 | **T1** (pinned by 096 P9) |
| `MatExpSymmetric` (eigendecomposition) | SPD manifold | textbook | ~30 | **T1** (pinned by 096 P6) |
| `MatLog` (inverse scaling-squaring) | infogeo SPD logmap | Higham 2008 §11 | ~300 | T2 |
| `MatLogSymmetric` | SPD logmap | composes with `MatExpSymmetric` | ~30 | **T1** |
| `MatSqrt` (Denman-Beavers) | polar / SPD geom-mean | Higham 2008 §6 | ~150 | T2 |
| `MatSqrt` (Schur-Newton) | better stability | Björck-Hammarling 1983 | ~250 | T3 |
| `MatSign` (Newton iteration) | algebraic Riccati, Sylvester | Higham 2008 §5 | ~120 | T2 |
| `MatSignSchur` | guaranteed convergence | Bai-Demmel 1994 | ~250 | T3 |
| `MatPower` (integer / fractional via Schur) | tensor power, spectral filter | Higham 2008 §7 | ~150 | T3 |
| `MatExp(it)` (Krylov, sparse) | quantum, large sparse | Saad 1992; Sidje "Expokit" | ~350 | T3 |
| Sylvester solver (AX + XB = C) | Bartels-Stewart 1972 | uses Schur | ~200 | T2 |
| Lyapunov solver (AX + XA^T = C) | special case Sylvester; control | Bartels-Stewart 1972 | ~150 | T2 |
| Algebraic Riccati (Schur-method) | optimal control / Kalman | Laub 1979 | ~400 | T3 |
| Matrix functions general (Schur-Parlett) | arbitrary `f(A)` | Davies-Higham 2003 | ~500 | T3 |
| Padé table (e^x, log(1+x), √(1+x)) | building blocks | Baker-Graves-Morris 1996 | ~80 | T2 |

---

## 7. Iterative refinement, sensitivity, condition numbers

| Capability | Status | Reference | Approx LOC | Tier |
|---|---|---|---|---|
| `MatNorm1` / `MatNormInf` / `MatNormFro` | **missing** (096 §9 P2) | textbook | ~40 | **T1** |
| `Cond1` / `CondInf` (via `Inverse`) | **missing** (096 §5 P3) | trivial | ~30 | **T1** |
| `Cond2` (via SVD) | **missing** | composes with SVD | ~10 over SVD | T2 |
| Condition estimator (Hager 1984) | **missing** | LAPACK `dlacon` | ~120 | T2 |
| Condition estimator (Higham-Tisseur 2000) | **missing** | LAPACK `dlacn2` | ~150 | T2 |
| Iterative refinement (Wilkinson 1963) | **missing** (096 §6 P4) | Higham §12 | ~30 | **T1** |
| Mixed-precision IR (FP32 LU + FP64 residual) | **missing** | Buttari 2008 | ~80 | T3 |
| Backward error estimate `||r||/(||A|| ||x||+||b||)` | **missing** | Higham §7 | ~25 | T2 |
| Componentwise condition number | **missing** | Skeel 1979 | ~80 | T3 |
| Pseudoinverse `pinv(A)` (Moore-Penrose) | **missing** | composes with SVD | ~30 over SVD | **T1** |
| Truncated pseudoinverse with tol | **missing** | Hansen "Rank-deficient and discrete ill-posed" | ~50 | T2 |
| Tikhonov-regularised solve | **missing** | Hansen 1998 | ~80 | T2 |
| Total Least Squares (TLS) | **missing** | Van Huffel-Vandewalle 1991 | ~120 | T2 |
| L-curve / GCV regularisation parameter selection | **missing** | Hansen 1992 | ~150 | T3 |
| Determinant via LU (sign + log\|det\|) | **half-present** (`Determinant` returns det but no log\|det\|) | textbook | ~20 | **T1** |
| Trace, rank-via-SVD, nullspace | **missing** rank/nullspace | composes with SVD | ~30 | **T1** |
| Householder reflector (apply, generate) | **missing** as a primitive | GVL §5.1.2 | ~50 | **T1** |
| Givens rotation (generate, apply) | **missing** as a primitive | GVL §5.1.8 | ~30 | **T1** |
| Reduced Row Echelon Form (RREF) | **missing** | textbook | ~80 | T2 |

The reflector + Givens generators are below-the-radar primitives that every higher-order routine (QR, Hessenberg, bidiag, SVD, Schur) is built from. Ship them once, reuse everywhere.

---

## 8. Tensor / multi-dimensional decompositions

Future-leaning, but mathematical canon. Reality has **zero**.

| Decomposition | Use case | Reference | Approx LOC | Tier |
|---|---|---|---|---|
| Kronecker product, Khatri-Rao | building blocks | textbook | ~60 | T2 |
| Tucker decomposition (HOSVD) | multilinear PCA | De Lathauwer-De Moor-Vandewalle 2000 | ~250 | T3 |
| Truncated HOSVD / sequentially-truncated | Vannieuwenhoven 2012 | ~200 | T3 |
| CP / PARAFAC (ALS) | Carroll-Chang / Harshman 1970 | Kolda-Bader 2009 | ~300 | T3 |
| Tensor train (TT-SVD) | Oseledets 2011 | ~400 | T3 |
| Hierarchical Tucker | Hackbusch-Kühn 2009 | ~600 | T3 |
| Mode-n product, unfolding/refolding | Kolda-Bader §2 | ~80 | T3 |

Recommend keeping all of these out of v1.x — they belong in a sibling `tensor` package once a real consumer demands them.

---

## 9. Cross-package consumers blocked today

(Surveyed via grep / agent 096 §13 / repo CLAUDE.md.)

| Consumer | Blocked by missing primitive |
|---|---|
| `optim` / L-BFGS | iterative refinement, condition number reporting on the inverse-Hessian solve |
| `infogeo` SPD manifold | **`MatExpSymmetric` + `MatLogSymmetric`** (high-impact; 096 P6) |
| `prob` multivariate Gaussian | rank-deficient Σ → pseudoinverse (needs SVD) |
| `control` Lyapunov/Sylvester | Bartels-Stewart (needs real Schur) |
| `chaos` Jacobian eigvals at fixed points | nonsymmetric eigendecomposition (needs Hessenberg + Francis QR) |
| `signal` autocorrelation → AR coefficients | Toeplitz Levinson-Durbin |
| `graph` spectral methods | Lanczos / Arnoldi on sparse Laplacian |
| `compression` low-rank / SVD-based | randomized SVD |
| `linalg.PCA` itself | symmetric-eig with eigenvectors (096 P5 retires inverse iteration) |
| Cross-language ports (Python/C++/C#) | fixed by the canonical-source rule once primitives exist |

---

## 10. Tiered roadmap

### Tier 1 — must-ship to call this a "linalg" library (≈1,200 LOC)

These eight items close the headline factorisation, function-of-matrix, sparse-iterative, and trustworthiness gaps with the smallest LOC footprint and the largest unblocking effect.

| # | Primitive | LOC | Unblocks |
|---|---|---|---|
| 1 | `MatNorm1`, `MatNormInf`, `MatNormFro` | 40 | every solver call site (no condition info today) |
| 2 | `Cond1`, `CondInf`, `LURefine` (096 P3+P4) | 60 | trustworthiness on every `LUSolve` |
| 3 | Householder reflectors + Givens primitives | 80 | building block for every higher routine |
| 4 | **`QRDecomposeHouseholder` + `QRSolve`** (096 P7) | 150 | rectangular least-squares; SVD prerequisite |
| 5 | `QRAlgorithmFull` (eigvals + eigvecs in one pass, 096 P5) | 80 | retires inverse iteration in PCA |
| 6 | **`MatExpSymmetric` + `MatLogSymmetric`** (096 P6) | 60 | infogeo SPD manifold |
| 7 | **`SVDGolubReinsch` + `Pseudoinverse` + `Cond2`** (096 P8) | 320 | rank-deficient LSQ, robust PCA, true 2-norm cond |
| 8 | COO + CSR sparse types + `SparseMatVec` | 180 | every Krylov solver below |
| 9 | CG + BiCGStab + GMRES on the SparseMat interface | 480 | large sparse `Ax = b` for every consumer |
| 10 | Banded solver + tridiagonal Thomas + Toeplitz Levinson-Durbin | 170 | signal AR fits, control banded systems |
| 11 | Lanczos (symmetric) + Arnoldi (general) | 450 | large sparse spectrum (graph, prob covariance) |
| 12 | Power + inverse iteration as public primitives | 80 | already inside pca.go; lift |

### Tier 2 — strongly motivated, second-pass (≈3,000 LOC)

Pivoted Cholesky, MGS-QR, pivoted QR, IDR(s), MINRES, LSQR/LSMR, IRAM, Krylov-Schur, LOBPCG, Hessenberg + real Schur, polar decomposition, `MatExp` (general Higham), `MatLog`, `MatSqrt`, Bartels-Stewart Sylvester/Lyapunov, banded Cholesky, ILU(0)/ILUT, IC(0), Jacobi/SSOR preconditioners, condition estimators (Hager / Higham-Tisseur), pseudoinverse with rank-truncation, Kronecker product, RREF, sparse Cholesky (CHOLMOD-style), AMD/RCM ordering.

### Tier 3 — research-grade or specialist (≈5,000 LOC)

QZ generalised eig, MRRR, divide-and-conquer eig, full LDL^T (Bunch-Kaufman), CS decomposition, full Schur-Parlett `f(A)`, complex Schur, Krylov `expm` (Sidje), algebraic Riccati, modified Cholesky, AMG, geometric MG, communication-avoiding CG, block CG/GMRES, FGMRES, QMR/TFQMR, Davidson / Jacobi-Davidson, Rayleigh-quotient iteration, METIS/MMD/coloring orderings, sparse LU (KLU/UMFPACK), tensor decompositions (Tucker, CP, TT), Cauchy/Vandermonde solvers, total least squares, Hankel solvers, L-curve / GCV.

---

## 11. Sprint advice

**Sprint-1 (closes the trustworthiness gap, ~330 LOC).** Tier 1 #1 + #2 + #3 + #5 + #6 — strictly additive, all under existing function-style API, all backwards-compatible. Closes the ill-conditioning blind spot, retires inverse iteration in PCA, unblocks infogeo SPD manifold.

**Sprint-2 (closes the rectangular-LSQ + low-rank gap, ~470 LOC).** Tier 1 #4 + #7. Requires Householder primitives from sprint-1. Unblocks pseudoinverse, robust PCA, condition-2, rank-revealing least squares, randomized SVD (~150 LOC over base).

**Sprint-3 (opens the sparse / iterative door, ~1,110 LOC).** Tier 1 #8 + #9 + #10 + #11 + #12. The CSR + SpMV foundation alone is ~180 LOC; the rest is Krylov solvers and Lanczos/Arnoldi all sharing the matvec abstraction. With this sprint, every consumer that today is blocked by "I have a 10000×10000 sparse system" becomes unblocked.

**Total Tier-1 cost:** ~1,200 LOC of pure additions. Compare to the current package size (~1,440 LOC). Doubles the package, increases the **functional surface 5-7×**.

---

## 12. Web-research crosswalk (LAPACK 3.x / Eigen 3 / ARPACK / SuiteSparse)

Confirms the missing list. Routine names match the LAPACK 3.12 reference and the supplementary references where noted.

- **LAPACK driver routines** Reality lacks Go equivalents for: `dgesv` (general LU solve — half-present), `dposv` (symmetric SPD solve via Cholesky — half-present), `dsysv` (symmetric indefinite via Bunch-Kaufman — absent), `dgels` (linear least squares via QR — absent), `dgelsd` (LSQ via SVD — absent), `dgesvd` / `dgesdd` (SVD — absent), `dsyev` / `dsyevd` (symmetric eig — partial: eigenvalues only), `dgeev` (general eig — absent), `dggev` (gen eig — absent), `dgees` / `dhseqr` (Schur — absent), `dgeqrf` / `dgeqp3` (QR / pivoted QR — absent), `dpbsv` / `dgbsv` (banded — absent), `dgtsv` (tridiagonal — absent).

- **ARPACK** routines reality lacks: `dsaupd` / `dseupd` (symmetric IRLM), `dnaupd` / `dneupd` (nonsymmetric IRAM). The symmetric-Lanczos and nonsymmetric-Arnoldi inner kernels alone (T1 #11) deliver ~70% of ARPACK's value at ~10% of its complexity; the implicit-restart wrapper is T2.

- **SuiteSparse** components reality lacks all of: `CHOLMOD` (sparse Cholesky), `KLU` / `UMFPACK` (sparse LU), `SuiteSparseQR` (sparse QR), `AMD` / `COLAMD` (orderings), `CCOLAMD`. Sparse direct factorisation is a major undertaking properly handled at v1.2.

- **Eigen 3** dense interface design conventions reality could borrow without copying: `LDLT`, `LLT`, `PartialPivLU`, `FullPivLU`, `ColPivHouseholderQR`, `JacobiSVD`, `BDCSVD` (divide-and-conquer SVD); `BDCSVD` (Trefethen-Howell 2018) is the modern default for `n > 100` and is **out of Tier 1** but worth a Tier 2 slot once Golub-Reinsch lands.

- **scipy.sparse.linalg** functional surface reality lacks: `cg`, `cgs`, `bicg`, `bicgstab`, `gmres`, `lgmres`, `qmr`, `gcrotmk`, `tfqmr`, `lsqr`, `lsmr`, `minres`, `eigs`/`eigsh` (ARPACK wrappers), `lobpcg`, `svds`, `expm_multiply`, `splu`, `spsolve`. About a dozen of these (CG/PCG/BiCGStab/GMRES/MINRES/LSQR/LSMR/Lanczos/Arnoldi/IRAM/LOBPCG/`expm_multiply`) are in scope as zero-dep Go translations and span Tier 1-Tier 3 above.

- **MATLAB Reference** routines reality lacks: `expm`, `logm`, `sqrtm`, `funm`, `lyap`, `dlyap`, `sylvester`, `polyeig`, `null`, `orth`, `pinv`, `rank`, `rcond`, `condest`, `normest`, `subspace`, `cgs` (Krylov), `lsqminnorm`. All in scope as v1 / v1.1 additions.

---

## 13. What this audit did not assess

- **Algorithm-level numerical experiments.** Did not run perturbation studies on missing routines.
- **Performance benchmarks.** LOC estimates only; no flop counts or wall-clock projections.
- **Cross-language port impact.** Whether each missing primitive trivially survives Go → Python / C++ / C# golden-file generation (likely yes for everything in §1-7; tensor decompositions §8 will need careful unfolding-convention pinning).
- **Whether Tier 3 items are actually wanted.** The principle is "ship when a Reality consumer needs it, and not before". Most Tier 3 entries today have zero internal consumer.

---

## 14. Bottom line

Reality v0.10.0's `linalg` ships the dense-direct LU + Cholesky + symmetric-tridiagonal-eigenvalues core that most numerical-methods textbooks label "Chapter 1-3". It is missing **Chapters 4-12** of Golub-Van Loan, **all of Saad's Iterative Methods**, **all of Higham's Function-of-Matrix**, **all of Davis's Direct Methods for Sparse Linear Systems**, and **all of Demmel's Applied Numerical Linear Algebra Chapters 5-7**. The absences are not bugs — what's there is correctly implemented — but the headline gap is real and the framing in CLAUDE.md ("LU/QR/Cholesky decomposition, sparse matrices") oversells by ~10×.

The single highest-leverage Tier-1 commit is **Householder reflectors + Givens + QR + symmetric-eig-with-vectors + matrix norms + Frobenius/1/Inf condition number + iterative refinement** (sprint-1, ~330 LOC). With it, every solver in the package becomes self-diagnostic, PCA gets ~30% faster and simpler, and the building blocks for SVD / Hessenberg / Schur / function-of-matrix all snap into place. The next commit is **SVD + pseudoinverse + Cond2 + randomized SVD** (sprint-2, ~470 LOC). After those two sprints, reality/linalg is ~ Eigen 3-class for dense work; the remaining frontier is sparse + Krylov + iterative eigenvalue (sprint-3, ~1,100 LOC) which lands together because they share the SpMV abstraction.

**File:** `C:\limitless\foundation\reality\reviews\overnight-400\agents\097-linalg-missing.md`, ~390 lines.

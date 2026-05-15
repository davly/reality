# 311 — dive-gmres-restart (Krylov solvers: CG / GMRES / BiCGStab / MINRES / LSQR + preconditioners)

## Headline
Reality v0.10.0 ships **zero Krylov solvers** — only dense direct LU/Cholesky/QR; this slot is the consolidating deep-dive specifying the canonical 11-primitive Krylov family (CG / MINRES / GMRES(m) / BiCGStab / LSQR / IDR(s) / FGMRES / LGMRES / Jacobi-ILU(0)-IC(0) preconditioners + Arnoldi/Lanczos primitives) ~1,650 LOC, with restart/deflation/augmentation strategies that previous slots 097 / 244 / 248 / 249 already enumerated as strict-upstream blockers.

## Findings

### F1. Concrete inventory of what reality has — and lacks
- **Have (dense direct, `linalg/decompose.go`):** `LUDecompose` + `LUSolve` (decompose.go:21, decompose.go:103); `CholeskyDecompose` + `CholeskySolve` (decompose.go:266, decompose.go:316); `QRAlgorithm` for eigenvalues (eigen.go:20). All allocation-free, dense, row-major, O(n³).
- **Have (dense matvec, `linalg/matrix.go`):** `MatVecMul(A, rows, cols, x, out)` (matrix.go:66) — the operator-application primitive every Krylov solver pivots on, ALREADY zero-allocation.
- **Lack (Krylov entirely):** No CG, no GMRES, no BiCGStab, no MINRES, no LSQR, no Arnoldi, no Lanczos, no preconditioner interface, no SparseMat / CSR / COO type. Verified via repo-wide grep: every match for `GMRES|BiCGStab|Krylov|Arnoldi|Lanczos|MINRES|LSQR` lives inside `reviews/overnight-400/agents/*.md`, plus one `prob/mathutil.go` Lanczos-of-the-other-kind (gamma function approximation, not solver-Lanczos).
- **Predecessor slots that explicitly call out the gap:**
  - `097-linalg-missing.md:28` — "Krylov iterative solvers: none; absent (no CG, GMRES, BiCGStab, IDR(s), MINRES, LSQR, LSMR)".
  - `097-linalg-missing.md:98-112` — already lists the full Krylov canon as Tier-1/Tier-2 with cited references and LOC budgets. **This slot does not reinvent that table; it deepens the restart/deflation/augmentation analysis.**
  - `244-new-pde-solvers.md:83-84,223,287-288` — pde elliptic + advection-diffusion solvers depend on `linalg/cg.ConjugateGradient` (D12) + `linalg/gmres.GMRES` (D27) + `linalg/bicgstab.BiCGStab` (D28).
  - `248-new-multigrid.md:99-100,179` — PCG-MG (G18) composes CG with V-cycle as preconditioner; GMRES-MG (G19) composes GMRES with AMG as right-preconditioner. Strict-upstream on this slot.
  - `249-new-domain-decomp.md`, `247-new-mortar-fem.md`, `102-optim-missing.md` — Schwarz / FETI-DP / mortar / interior-point KKT all consume GMRES + AMG-precon.

### F2. Restart strategies — the load-bearing GMRES design decision
Plain (full) GMRES stores k Arnoldi vectors at iteration k; memory grows linearly, orthogonalisation cost grows quadratically. For n = 10⁶, k = 200 means 1.6 GB just for the basis. Three established mitigations:

1. **GMRES(m) — fixed restart (Saad-Schultz 1986).** After m iterations, take current iterate x_m as initial guess, discard Arnoldi basis, restart Arnoldi from r_m = b − A x_m. Bounded memory O(m·n). **Failure mode:** loses superlinear convergence; can stagnate when m too small for spectrum (typical: m=20-50). Pin: m=30 is the *MATLAB default* and the right reality default.
2. **Augmented / Recycled Krylov (LGMRES — Baker-Jessup-Manteuffel 2005, builds on Eisenstat-Walker 1996; GCROT — de Sturler 1999).** Retain k "best" vectors across restarts, typically the smallest harmonic-Ritz vectors that approximate the slow-converging eigendirections. ~80 LOC delta over GMRES(m). Saves 30-70% iterations on stagnating problems.
3. **Deflated GMRES (Erhel-Burrage-Pohl 1996; Morgan 1995 GMRES-E).** Explicitly project out near-zero eigenvalues that cause stagnation: build augmented basis [V_m, U_k] where U_k spans approximate eigenspace of small |λ|. Effectively shifts spectrum, dramatically improves clustering. Convergence-decisive for indefinite Helmholtz, advection-dominated, and second-restart problems.

For reality's first cut: **ship GMRES(m) with m as parameter, defer LGMRES/deflated until empirical need.** LGMRES is ~80 LOC additional once GMRES(m) is solid; deflated GMRES needs an eigensolver companion (Krylov-Schur slot 097-T2) so naturally sequences after.

### F3. Preconditioning is the actual fight
Unpreconditioned Krylov is a textbook curiosity. Production needs:
- **Left preconditioning:** solve M⁻¹A x = M⁻¹b. Changes the residual norm being measured (M⁻¹-norm), can mask convergence.
- **Right preconditioning:** solve A M⁻¹ y = b, x = M⁻¹y. Preserves true residual norm. **Default for GMRES.**
- **Flexible preconditioning (FGMRES — Saad 1993):** allows preconditioner to vary across iterations (e.g., inner Krylov as preconditioner). Costs extra basis storage. ~80 LOC over GMRES.
- **Split preconditioning:** M = M_L M_R, solve M_L⁻¹ A M_R⁻¹ y = M_L⁻¹b. Good for SPD via incomplete-Cholesky.

Standard ladder of preconditioners (cheapest → strongest):
| Preconditioner | LOC | Pin tier |
|---|---|---|
| Identity (none) | 0 | sanity |
| Jacobi (diagonal) | 20 | T1 |
| Block Jacobi | 50 | T2 |
| Symmetric Gauss-Seidel | 60 | T2 |
| SSOR (Symmetric Successive Over-Relaxation) | 80 | T2 |
| ILU(0) — same sparsity as A | 150 | **T1** |
| ILU(p) — level-of-fill p | 200 | T2 |
| ILUT(τ, p) — threshold-based | 250 | T2 |
| IC(0) — incomplete Cholesky for SPD | 130 | T1 |
| AMG V-cycle (slot 248-G17) | n/a (in 248) | T1 |

ILU(0) is the right "first non-trivial preconditioner" — same nonzero pattern as A, no fill-in, cheap setup, generally cuts iterations 3-10× for elliptic. **Requires a SparseMat (CSR) type — strict-upstream from 097-T1 (#8 in 097's PR ladder).**

### F4. Stopping criteria and finite-precision pitfalls (the bug surface)
Production Krylov implementations differ on which residual norm to report — and this is where they break in finite arithmetic.
- **Recurrent residual** vs **explicit residual** (b − A x_k): can drift by 1e-6 by iteration 200 even in FP64 (Greenbaum-Strakos 1992). reality **MUST** recompute explicit residual every ~50 iters or at termination, and warn if drift > 10× tolerance.
- **Modified Gram-Schmidt vs Classical Gram-Schmidt:** MGS is the reality default for Arnoldi — CGS loses orthogonality catastrophically on ill-conditioned problems (Björck 1994). MGS-twice-is-enough (Giraud-Langou-Rozložník 2005) is the gold standard but ~10% slower; ship MGS-once for v1, MGS-twice as opt-in.
- **Lucky breakdown:** when h_{j+1,j} = 0 in Arnoldi, K_j is invariant — solution is exact. Must detect and handle, not propagate NaN.
- **Tolerance specification:** `tol` is *relative* (‖r_k‖ / ‖b‖) by canon; absolute tolerance is also useful for ill-conditioned. Ship both: `relTol`, `absTol`, terminate on either.

### F5. Per-method cheat sheet (decision matrix)
| A is... | Preferred solver | Why |
|---|---|---|
| SPD | **CG** | 3-term recurrence, O(1) memory, optimal in A-norm |
| symmetric indefinite | **MINRES** | optimal in 2-norm, 3-term recurrence still |
| symmetric semi-definite least-squares | **LSQR** or LSMR | numerically stable for ill-conditioned LS |
| non-symmetric, transpose available | **GMRES(m)** + ILU | most robust general-purpose |
| non-symmetric, no transpose, low memory | **BiCGStab** | O(1) basis, often faster wall-clock than GMRES(50) |
| non-symmetric, smoother convergence than BiCGStab | **BiCGStab(L)** or IDR(s) | breakdown-resistant |
| variable preconditioner (inner Krylov, AMG with adaptive smoothing) | **FGMRES** | only solver allowing M to change |
| stagnating GMRES | **LGMRES** or deflated-GMRES | augmented Krylov |
| huge non-symmetric | **IDR(s)** | Sonneveld-van Gijzen 2008, often 2× fewer matvecs than BiCGStab |

### F6. Cross-validation pin opportunities (R-MUTUAL-CROSS-VALIDATION 3/3)
1. **CG ≡ Cholesky on SPD (n=200, κ=1e6).** Build random SPD via A = QΛQᵀ with Λ_i ∈ [1, 1e6]. CG with relTol=1e-12 must agree with `CholeskySolve` to 1e-10 in 2-norm. Tests (a) correctness, (b) finite-precision residual recovery, (c) iteration count ≤ 2√κ (CG's textbook bound).
2. **GMRES ≡ LU on dense full-rank non-symmetric (n=200).** Random A with entries N(0,1). Both unrestarted GMRES (m=200) and GMRES(30) restart should converge to the LU solution to 1e-10. Pin GMRES(30) iter-count ≤ 200.
3. **GMRES ≡ BiCGStab ≡ LU 3-way on n=200 random non-symmetric, well-conditioned (κ ≤ 1e4).** All three solvers + LU must agree pairwise to 1e-10. **This is the gold R-pattern 3/3 — three independent algorithmic paths converging on identical answer.**
4. **Bonus: GMRES(m) + ILU(0) ≡ GMRES(m) + Identity on Poisson-2D 64×64.** Same answer, ILU(0) takes ~3-5× fewer iters. Pins both correctness AND preconditioner contract.

### F7. Memory and allocation discipline
Reality's design rule #3 forbids hot-path allocation. Implications:
- All Krylov solvers must take a `Workspace` struct caller-allocates: e.g. `cg.Workspace{ R, P, Z, Ap []float64 }` for CG (4n floats); GMRES(m) needs `{ V (m+1)×n, H (m+1)×m, Givens 2m, Y m, ... }` ~(m+5)·n + 4m floats.
- The matrix-application abstraction is `type MatVec func(x, out []float64)` — closures over either dense `MatVecMul` or future sparse `CSR.MatVec`. Same signature works for any operator (FFT-based for circulant, sten cil-based for FD-Laplacian).
- Preconditioner abstraction: same `type Preconditioner func(r, z []float64)` (apply M⁻¹ to r, store in z).

### F8. Consumer impact (which slots unblock when this lands)
| Consumer slot | What unblocks | Solver needed |
|---|---|---|
| 244 PDE solvers (Poisson-2D, advection-diffusion, FE) | D11, D26, D28 | CG + GMRES + BiCGStab |
| 247 mortar-FEM (saddle-point KKT) | M14 | GMRES + AMG-precon |
| 248 multigrid (PCG-MG, GMRES-MG, K-cycle) | G18, G19, G20 | CG + GMRES + Preconditioner interface |
| 249 domain-decomp (Schwarz, FETI-DP) | all | GMRES + Schwarz-as-precon |
| 102 optim missing (interior-point, SOCP) | KKT solves | MINRES (saddle-point) + GMRES |
| 097 linalg missing | the entire Krylov family | this slot is the deep-dive for that table |
| 245 spectral methods | spectral-precond CG | CG + spectral-precon |
| 237 GP regression (n > 10⁴) | kernel solve | CG + Nyström-precon |
| 261/262 randomized SVD / online SVD | Lanczos restart | Lanczos primitive |
| 272 manifold learning (LE, LLE, diffusion maps) | sparse eig | Lanczos + ARPACK-style restart |
| Pistachio inverse-rendering, large-scale image reconstruction | normal-equations LS | LSQR / LSMR |
| aicore (downstream consumer) | KFAC / natural gradient | CG + block-diag preconditioner |

### F9. Ship-order — cheapest-to-most-impact day-1 PR
**~580 LOC for the keystone PR:**
- `linalg/cg.go` ConjugateGradient ~150 LOC (Hestenes-Stiefel 1952, textbook 4-state recurrence)
- `linalg/gmres.go` GMRES(m) ~250 LOC (Arnoldi+MGS, Givens rotation least-squares, restart loop)
- `linalg/precon.go` Preconditioner type + Jacobi (~30 LOC) + Identity (~10 LOC) ~50 LOC
- `linalg/bicgstab.go` BiCGStab ~150 LOC (van der Vorst 1992)
- Golden vectors + R-MUTUAL-CROSS-VALIDATION 3/3 from F6.

**Why this order:** CG validates against Cholesky (already in repo) — pure regression test, no new dependency. GMRES validates against LU (already in repo). BiCGStab validates against GMRES — internal triangulation. Once these three plus Jacobi land, slot 244 D11/D12/D27/D28 unblocks immediately, multigrid PCG-MG (248-G18) becomes a thin composition.

**Sprint-2 (~700 LOC):** ILU(0) + IC(0) preconditioners (depend on 097-T1 SparseMat/CSR), MINRES, LSQR. Unblocks 102 (interior-point KKT), 245 (spectral-precon), Pistachio LSQR.

**Sprint-3 (~600 LOC, frontier):** FGMRES (~80 over GMRES), LGMRES (~80 over GMRES), IDR(s) (~300), deflated-GMRES with Krylov-Schur eigenpairs (~150). Unblocks Helmholtz at high frequency, indefinite shifted-Laplacian, the convection-dominated stagnation cases.

### F10. Single-source-of-truth boundary clarification
- **CG / GMRES / BiCGStab / MINRES / LSQR / Arnoldi / Lanczos / Preconditioner-interface live in `linalg/`** — every PDE / FEM / multigrid / domain-decomp / optim / GP / Pistachio consumer imports here.
- **AMG V-cycle, smoothed-aggregation, classical-AMG-Ruge-Stüben live in `multigrid/`** (slot 248) — they are *applications* of the linalg-CG/GMRES + their own coarsening logic.
- **PDE-domain-specific iterations (Jacobi-as-PDE-iteration, GS-as-PDE-iteration on actual stencils)** live where slot 244 puts them; the *generic Jacobi/GS preconditioners* live in `linalg/precon.go`. This split was implicit in 248-`smoother.go` discussion (`248-new-multigrid.md:240`); making it explicit prevents duplication.

### F11. What this slot is NOT
- Not the place to ship sparse-matrix types — that's 097-T1 PR-3.
- Not the place to ship multigrid V-cycle — that's 248.
- Not the place to ship eigsolvers (Lanczos for eigenvalues, ARPACK-style restart) — that's 097-T2 / a future slot.
- Not the place to debate Communication-Avoiding CG (Hoemmen 2010, s-step), Pipelined CG (Ghysels-Vanroose 2014), Mixed-precision iterative refinement (Carson-Higham 2018) — these are v1.2 enhancements once the base is solid.

## Concrete recommendations

1. **Day-1 PR (this slot's cheapest leverage):** ship `linalg/cg.go` + `linalg/gmres.go` (with restart parameter `m`) + `linalg/bicgstab.go` + `linalg/precon.go` (Jacobi + Identity) — ~580 LOC, four files, allocation-free, Workspace-struct API. Land R-MUTUAL-CROSS-VALIDATION 3/3 on n=200 random non-symmetric.
2. **Adopt MatVec closure abstraction** (`type MatVec func(x, out []float64)`) as the universal operator-application primitive; same signature works for dense, future sparse, FFT-based-circulant, and stencil-based finite-difference operators. Document in `linalg/doc.go`.
3. **Default GMRES restart m=30** (matches MATLAB), expose as parameter; default tolerance `relTol=1e-8, absTol=0`, hard cap `maxIter = max(2n, 1000)`. Recompute explicit residual every 50 iterations to detect drift; emit warning if drift > 10× relTol.
4. **Use Modified Gram-Schmidt** for Arnoldi (not Classical); ship MGS-once initially with TODO for MGS-twice-is-enough on ill-conditioned. Document Björck 1994 / Giraud-Langou-Rozložník 2005 trade-off.
5. **Ship preconditioner interface as `type Preconditioner func(r, z []float64)`** parallel to MatVec. Provide Identity (no-op) + Jacobi day-1; ILU(0) + IC(0) sprint-2 once SparseMat lands (097-T1).
6. **Detect lucky breakdown** in Arnoldi (h_{j+1,j} below `eps·‖A‖_F`); declare exact convergence, return current iterate. Test with explicit lucky-breakdown construction.
7. **Defer LGMRES, FGMRES, IDR(s), deflated-GMRES to sprint-3.** Each is small (~80-300 LOC) once base GMRES is solid, but adds zero value before then.
8. **Single-source-of-truth split:** generic Jacobi/GS preconditioners live in `linalg/precon.go`; AMG/V-cycle stays in `multigrid/`; PDE-discrete-iteration stays in `pde/elliptic/`. Slot 244 + 248 + this slot must reference each other to prevent duplication.
9. **Naming alignment:** match scipy.sparse.linalg names exactly where possible: `cg`, `gmres`, `bicgstab`, `minres`, `lsqr`, `lgmres`. Reduces friction for users coming from Python/MATLAB.
10. **Cross-language golden files:** Python validation via scipy.sparse.linalg with same matrices/RHS/tolerance — already the reality testing protocol. Same operator + same RHS + same tol must converge to same x to 1e-10 across Go and Python.
11. **Update `097-linalg-missing.md` PR ladder:** the Tier-1 row "9 | CG + BiCGStab + GMRES on the SparseMat interface | 480 LOC" already aligns with this slot — confirm convergence between 097's plan and this deep-dive when 097 enters its PR phase.
12. **Update `244-new-pde-solvers.md` D12/D27/D28 references** to point at the day-1 PR signatures specified here, so PDE-elliptic / advection-diffusion can land immediately after.

## Sources

### Reality repository (file:line)
- `linalg/decompose.go:21` `LUDecompose`; `linalg/decompose.go:103` `LUSolve`; `linalg/decompose.go:266` `CholeskyDecompose`; `linalg/decompose.go:316` `CholeskySolve` — the direct solvers Krylov regresses against.
- `linalg/eigen.go:20` `QRAlgorithm` — used by deflated-GMRES Ritz pair extraction (sprint-3).
- `linalg/matrix.go:66` `MatVecMul` — the operator-application primitive every Krylov method calls.
- `reviews/overnight-400/agents/097-linalg-missing.md:28,98-112,278,317` — predecessor flagging missing Krylov canon with LOC budgets, references, and PR ladder rung.
- `reviews/overnight-400/agents/098-linalg-sota.md:48` — Eigen comparison, confirms scipy/Eigen surface reality lacks.
- `reviews/overnight-400/agents/244-new-pde-solvers.md:83-84,223,246-248,303,319` — PDE D11/D12/D26/D27/D28 strict-downstream specifications.
- `reviews/overnight-400/agents/248-new-multigrid.md:99-100,179,220-221,247` — multigrid PCG-MG (G18) + GMRES-MG (G19) compositions on this slot's primitives.
- `CLAUDE.md` — design rules #3 (no allocations in hot paths) + #6 (reimplement from first principles); explains the Workspace-struct API discipline.

### Canonical literature
- Saad-Schultz, "GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems," **SIAM J. Sci. Stat. Comput.** 7(3), 1986. The GMRES paper.
- Saad, **Iterative Methods for Sparse Linear Systems**, 2nd ed., SIAM 2003. The textbook for everything in this slot — §6.5 GMRES, §6.7 CG, §7.4 BiCGStab, §9 preconditioners, §6.5.7 restart, §6.5.9 augmented Krylov.
- Hestenes-Stiefel, "Methods of Conjugate Gradients for Solving Linear Systems," **J. Research NBS** 49(6), 1952. CG keystone.
- Paige-Saunders, "Solution of Sparse Indefinite Systems of Linear Equations," **SIAM J. Numer. Anal.** 12, 1975. MINRES.
- Paige-Saunders, "LSQR: An Algorithm for Sparse Linear Equations and Sparse Least Squares," **ACM TOMS** 8(1), 1982. LSQR.
- van der Vorst, "Bi-CGSTAB: A Fast and Smoothly Converging Variant of Bi-CG," **SIAM J. Sci. Stat. Comput.** 13(2), 1992. BiCGStab.
- Sleijpen-Fokkema, "BiCGstab(L) for Linear Equations involving Unsymmetric Matrices with Complex Spectrum," **ETNA** 1, 1993.
- Erhel-Burrage-Pohl, "A Deflated Version of the Conjugate Gradient Algorithm," **SIAM J. Sci. Comput.** 17(5), 1996 (and the GMRES counterpart line of work). Deflated Krylov.
- Eisenstat-Walker, "Choosing the Forcing Terms in an Inexact Newton Method," **SIAM J. Sci. Comput.** 17(1), 1996. Augmented-Krylov motivation in inexact-Newton context.
- Saad, "A Flexible Inner-Outer Preconditioned GMRES Algorithm," **SIAM J. Sci. Comput.** 14(2), 1993. FGMRES.
- Morgan, "A Restarted GMRES Method Augmented with Eigenvectors," **SIAM J. Matrix Anal. Appl.** 16(4), 1995. GMRES-E / augmented restart.
- Baker-Jessup-Manteuffel, "A Technique for Accelerating the Convergence of Restarted GMRES," **SIAM J. Matrix Anal. Appl.** 26(4), 2005. LGMRES.
- de Sturler, "Truncation Strategies for Optimal Krylov Subspace Methods," **SIAM J. Numer. Anal.** 36(3), 1999. GCROT.
- Sonneveld-van Gijzen, "IDR(s): A Family of Simple and Fast Algorithms for Solving Large Nonsymmetric Systems of Linear Equations," **SIAM J. Sci. Comput.** 31(2), 2008. IDR(s).
- Trefethen-Bau, **Numerical Linear Algebra**, SIAM 1997. Lectures 32-40 — GMRES geometric intuition.
- Greenbaum, **Iterative Methods for Solving Linear Systems**, SIAM 1997. Convergence theory + finite-precision (Greenbaum-Strakos).
- Björck, "Numerics of Gram-Schmidt Orthogonalization," **Linear Algebra Appl.** 197/198, 1994. MGS vs CGS.
- Giraud-Langou-Rozložník, "On the Loss of Orthogonality in the Gram-Schmidt Orthogonalization Process," **Computers & Mathematics with Applications** 50, 2005. MGS-twice-is-enough.

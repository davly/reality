# 203 — New Math: Tensor Networks (Block C, slot 3)

**Summary line 1:** reality v0.10.0 ships **zero** tensor-network or higher-order-tensor machinery — repo-wide grep on `tensor|MPS|PEPS|MERA|DMRG|TEBD|HOSVD|Tucker|CANDECOMP|PARAFAC|kronecker|einsum|TT.SVD|tt_cross` returns only review files and one `[][][]float64` accumulator in `chaos/analysis.go`; the closest substrate is `linalg/` (matrices only, NO SVD, NO QR, NO Householder exposed — `tridiagonalize` is private, `QRAlgorithm` returns eigenvalues but not eigenvectors as Q, `PCA` recovers eigenvectors via inverse iteration), `signal/fft.go` (1-D FFT useful for periodic-tensor conjugate), and `autodiff/` (scalar tape only, no batched/tensor ops).
**Summary line 2:** Twenty-four ranked primitives T0a-T20 totalling ~5,400 LOC would establish a `tensor/` package on top of a *blocking* prerequisite — `linalg/svd.go` (~400 LOC Golub-Reinsch + thin/truncated variants); cheapest one-day shippable artifact is T0a SVD itself (unblocks **everything** in this list and ~6 other slots), single-highest-leverage cutting-edge piece is T5 TT-SVD (Oseledets 2011, ~250 LOC) — gives reality an exponential-compression primitive that *no zero-dependency math library ships*, and is the foundational data structure for high-dimensional PDE / quantum-circuit / probabilistic-graphical-model breakthroughs since 2011.

---

## (1) What reality ships today (verified at v0.10.0)

**Tensor machinery: nothing.** Every appearance of "tensor" in a non-review .go file is either a docstring metaphor or absent entirely. Repo-wide hits on `tensor|Tensor|MPS|TT|PEPS|MERA|DMRG|TEBD|hosvd|HOSVD|Tucker|CANDECOMP|PARAFAC|kronecker|Kronecker|einsum|Einsum` return zero matches in production source. The Lighthill stress *tensor* is shipped as a per-point `[3][3]float64` in `acoustics/aero/` (slot 197 plan) but that's a 3×3 rank-2 array, not the higher-order-tensor abstraction this slot is about.

**Critical missing prerequisite — full SVD.** Every tensor decomposition (CP, Tucker, HOSVD, TT-SVD, hierarchical Tucker, MPS canonicalisation, TT-rounding) reduces to an SVD inner loop. reality has:

| Substrate | Status | Gap |
|---|---|---|
| `linalg/decompose.go` | LU + Cholesky only | No SVD, no QR, no rank-revealing decomposition |
| `linalg/eigen.go` | `QRAlgorithm` returns eigenvalues of symmetric matrices via Householder + tqli; **does not return eigenvectors** | Tucker/HOSVD need top-k singular *vectors*, not just values |
| `linalg/pca.go` | `PCA` recovers eigenvectors via inverse iteration on the covariance matrix | Works for symmetric PSD only; cannot truncate-SVD an arbitrary rectangular matrix |
| `linalg/matrix.go` | MatMul, MatTranspose, Identity, Trace, MatAdd, MatScale | No reshape / fold / unfold / Kronecker product |
| `signal/fft.go` | Cooley-Tukey FFT | Available; needed for one TT-application (S6 quantum-FFT-as-MPS) |
| `autodiff/` | Scalar reverse-mode AD over Variables | No batched / matrix / tensor ops; would need augmentation for differentiable tensor decompositions |
| `optim/` | Newton / L-BFGS / simulated annealing | Can drive the alternating least-squares (ALS) inner loops for CP / Tucker / DMRG, but no native ALS substrate |

**No multidimensional array type.** reality's convention is row-major flat `[]float64` with explicit shape arguments. This is fine for matrices (rank-2) and the scattered `[][]float64` jagged arrays in `chaos`, `optim/genetic`, `compression`. For higher-order tensors (rank 3+), repo has *exactly one* `[][][]float64` accumulator (`chaos/analysis.go` recurrence-plot 3-D histogram, used as scratch; not exposed). **A canonical `tensor.Tensor` type with shape + flat storage is the very first artifact this slot must ship.**

**v2 deferral roster from linalg/ — none.** `linalg/decompose.go` ends with LU + Cholesky and does not enumerate SVD as deferred. `linalg/eigen.go` does not enumerate eigenvectors as deferred. `linalg/pca.go` mentions inverse-iteration but does not surface SVD. **This slot surfaces the entire tensor-network corpus and should also flag the `linalg/svd.go` gap to slot 081-linalg-missing and slot 084-linalg-sota.**

---

## (2) What's missing — twenty-four primitives ranked by demand

Demand ranking weights: (a) explicit consumer in CONTEXT.md / aicore / Pistachio downstream, (b) frequency in Cichocki-Mandic-Phan-2015 / Kolda-Bader-2009 / Schollwöck-2011 / Oseledets-2011 review-article corpora, (c) connective-tissue readiness, (d) "no-zero-dep-library-ships-this" cutting-edge score.

### Tier-0 — substrate (~700 LOC, blocks everything below)

#### T0a. `linalg/svd.go` — full and truncated SVD — ~400 LOC ⭐
**Blocks every primitive in this slot and four other slots (autodiff Greeks, optim proximal nuclear-norm, prob covariance estimation, color-PCA whitening).** Golub-Reinsch 1970 algorithm: bidiagonalisation via Householder reflectors + implicit QR sweep on bidiagonal. Three public surfaces:

```go
// Full SVD: A = U·Σ·V^T, A is m×n, U is m×min(m,n), S is min(m,n), V is n×min(m,n).
func SVD(A []float64, m, n int, U, S, V []float64) error

// Thin SVD with rank truncation; returns effective rank r ≤ k.
func SVDTruncated(A []float64, m, n, k int, U, S, V []float64) (int, error)

// Randomized SVD (Halko-Martinsson-Tropp 2011) for m,n ≫ k.
func RandomizedSVD(A []float64, m, n, k, oversample int, rng RNG, U, S, V []float64) error
```

Implementation reuses the existing `tridiagonalize` Householder machinery (currently private in `linalg/eigen.go`); promotes it to `householderReflector(v, beta, tau, n)` and adds the bidiagonalisation phase (alternating left/right Householder). The implicit-QR sweep on the bidiagonal mirrors `tqli` from `eigen.go` with two-sided Givens rotations instead of one-sided.

Cross-substrate parity: pin against numpy.linalg.svd at 1e-12 element-wise on the singular values and 1e-10 on `U·Σ·V^T` reconstruction. Sign convention (column signs of U) deliberately fixed to match LAPACK so cross-language golden files don't drift.

**This is the single highest-priority addition to reality across all of Block C.** Without SVD, Tucker / HOSVD / TT / DMRG / proximal nuclear-norm / robust-PCA / Procrustes / CCA / pseudoinverse all stall.

#### T0b. `tensor/tensor.go` — canonical multi-dim array type — ~150 LOC
The data structure that all of Tier-1+ consumes:

```go
type Tensor struct {
    Data    []float64  // flat row-major (C-order)
    Shape   []int      // tensor mode dimensions
    Strides []int      // for non-contiguous views (defaults to row-major strides)
}

func NewTensor(shape ...int) *Tensor
func (t *Tensor) NDim() int
func (t *Tensor) Size() int  // product of shape
func (t *Tensor) Get(idx ...int) float64
func (t *Tensor) Set(value float64, idx ...int)
func (t *Tensor) Reshape(shape ...int) *Tensor      // zero-copy if contiguous
func (t *Tensor) Unfold(mode int, out []float64) (rows, cols int)  // mode-n unfolding
func (t *Tensor) Fold(mode int, rows, cols int, out *Tensor)       // inverse of Unfold
func (t *Tensor) Permute(perm []int, out *Tensor)
```

**Mode-n unfolding** (matricisation) is *the* fundamental operation that bridges tensor algebra ↔ matrix algebra. T_(n) = the matrix whose rows are the mode-n fibres of T. Tucker, HOSVD, TT-SVD all unfold-then-SVD-then-fold. Convention: Kolda-Bader 2009 §2 ordering (mode-n column index = i_1·I_2·... + i_2·I_3·... + ... excluding mode n).

#### T0c. `tensor/contract.go` — basic contractions — ~150 LOC
Three primitive operations:

```go
// Mode-n product: T ×_n M produces T' with shape ..., M.cols, ... replacing mode n.
func ModeNProduct(t *Tensor, mode int, M []float64, mRows, mCols int, out *Tensor)

// Generalised inner product over named modes (the "contraction core").
func Contract(a, b *Tensor, aModes, bModes []int, out *Tensor)

// Kronecker product (matrix-only, but reused by Tucker reconstruction).
func Kronecker(A []float64, aRows, aCols int, B []float64, bRows, bCols int, out []float64)

// Khatri-Rao (column-wise Kronecker; foundational for CP-ALS).
func KhatriRao(A []float64, aRows, k int, B []float64, bRows int, out []float64)
```

Connective-tissue: reuses `linalg.MatMul` after unfolding for ModeNProduct.

### Tier-1 — high demand, short connective tissue (~1,800 LOC)

#### T1. CP decomposition via ALS (CANDECOMP/PARAFAC, Carroll-Chang 1970, Harshman 1970) — ~300 LOC
Approximates rank-d tensor T ≈ Σ_{r=1}^R λ_r · u_r^(1) ⊗ u_r^(2) ⊗ ... ⊗ u_r^(d) by alternating least squares: cycle over modes 1..d, freeze all but mode-n factor matrix A^(n), solve the linear least-squares problem `T_(n) ≈ A^(n) · (A^(d) ⊙ ... ⊙ A^(n+1) ⊙ A^(n-1) ⊙ ... ⊙ A^(1))^T` where ⊙ is Khatri-Rao.

```go
type CPResult struct {
    Factors []*linalg.Matrix  // d factor matrices, each I_n × R
    Lambda  []float64         // R weights (ℓ²-norm of each rank-1 component)
    FitError float64
    Iterations int
}
func CPALS(T *Tensor, rank int, maxIter int, tol float64) (CPResult, error)
```

Cross-substrate parity: pin against tensorly.decomposition.parafac at 1e-6 on the reconstructed tensor (CP factor matrices are *not* unique up to permutation/scaling, but the reconstruction is). Failure mode: CP is non-convex; ship `CPALSMultistart(T, rank, restarts, ...)` with deterministic seed for reproducibility.

#### T2. Tucker decomposition (HOSVD initialisation + ALS refinement) — ~300 LOC
T ≈ G ×_1 U^(1) ×_2 U^(2) ×_d U^(d) with core tensor G of shape (R_1, ..., R_d) and orthogonal factor matrices U^(n) of shape I_n × R_n. **Two-stage:**

- **HOSVD (De Lathauwer-De Moor-Vandewalle 2000):** non-iterative initialisation. For each mode n, U^(n) ← top-R_n left singular vectors of T_(n). Core G ← T ×_1 U^(1)^T ×_2 ... ×_d U^(d)^T.
- **HOOI (Higher-Order Orthogonal Iteration):** ALS refinement. Cycle: U^(n) ← top-R_n left singular vectors of (T ×_{m≠n} U^(m)^T)_(n). Repeat to convergence.

```go
type TuckerResult struct {
    Core    *Tensor
    Factors []*linalg.Matrix
    Reconstruction func() *Tensor  // lazy
}
func HOSVD(T *Tensor, ranks []int) (TuckerResult, error)        // non-iterative
func HOOI(T *Tensor, ranks []int, maxIter int, tol float64) (TuckerResult, error)  // refined
```

**Truncation-error bound.** Eckart-Young is *not* tight for HOSVD (Tucker is not the best multilinear approximation), but the De Lathauwer 2000 quasi-optimality theorem gives `||T − T_HOSVD||_F ≤ √d · ||T − T_best||_F`. **Pin this bound in a golden-file test.**

#### T3. Higher-Order SVD with truncation bounds (HOSVD-T) — ~150 LOC
Extends T2 with explicit error-controlled truncation. Given target relative error ε, pick ranks R_n adaptively per mode such that `Σ_{r > R_n} (σ_r^(n))² ≤ ε² · Σ_r (σ_r^(n))²`. Returns the chosen ranks. Mirrors Vannieuwenhoven-Vandebril-Meerbergen 2012 truncation strategy.

```go
func HOSVDAdaptive(T *Tensor, epsilon float64) (TuckerResult, []int, error)
```

#### T4. Tensor-Train (TT) representation and basic arithmetic — ~250 LOC
Foundational data structure for Tier-2. A TT tensor of order d is a list of *cores* G_k ∈ R^{r_{k-1} × n_k × r_k} with r_0 = r_d = 1; the tensor entry is

`T(i_1, ..., i_d) = G_1(i_1) · G_2(i_2) · ... · G_d(i_d)` (matrix product of core slices).

```go
type TT struct {
    Cores []*Tensor  // each shape [r_{k-1}, n_k, r_k]
    Ranks []int      // [r_0, r_1, ..., r_d], r_0 = r_d = 1
}
func (tt *TT) ToFull() *Tensor               // exponential blow-up; for testing/small d
func (tt *TT) Element(idx ...int) float64    // efficient single-element lookup
func (tt *TT) Norm() float64                 // ||T||_F via left-orthogonalisation
func (tt *TT) MaxRank() int
func TTAdd(a, b *TT) *TT                     // ranks add: r_k(a+b) = r_k(a) + r_k(b)
func TTHadamard(a, b *TT) *TT                // element-wise; ranks multiply
func TTInnerProduct(a, b *TT) float64
```

Schollwöck 2011 review and Oseledets 2011 paper are the canonical references. Cross-substrate parity: pin via `ToFull` round-trip at 1e-12.

#### T5. TT-SVD constructive algorithm (Oseledets 2011, Theorem 2.2) — ~250 LOC ⭐
**The single highest-leverage cutting-edge tensor primitive.** Given a full tensor T and target tolerance ε (or fixed maximum rank r_max), constructs the TT representation by sequential SVDs of mode-by-mode unfoldings:

```
1. C ← T, reshaped to (n_1) × (n_2 · ... · n_d)
2. For k = 1, ..., d-1:
     [U, Σ, V] ← truncated SVD of C with rank r_k chosen by ε / r_max
     G_k ← reshape(U, [r_{k-1}, n_k, r_k])
     C ← Σ · V^T, reshaped to (r_k · n_{k+1}) × (n_{k+2} · ... · n_d)
3. G_d ← reshape(C, [r_{d-1}, n_d, 1])
```

`||T - T_{TT}||_F² ≤ Σ_k ε_k²` where ε_k is the truncation error at step k — **a globally-controlled error bound that no other tensor decomposition provides**. This is *the* quasi-optimality theorem that makes TT the practical workhorse for high-d compression.

```go
func TTSVD(T *Tensor, epsilon float64) (*TT, error)            // ε-bounded
func TTSVDFixedRank(T *Tensor, maxRank int) (*TT, error)       // r_max-bounded
```

Cross-substrate parity: pin Hilbert-tensor T(i_1,...,i_d) = 1 / (i_1 + ... + i_d) at d=10, n=8 (10⁸ entries) and verify TT representation reduces to ~kilobytes with reconstruction error 1e-8 (Oseledets 2011 Example 4.1). **This single test demonstrates the curse-of-dimensionality break that motivates the entire tensor-network field.**

#### T6. TT-rounding (orthogonalize-then-truncate) — ~200 LOC
After arithmetic operations (TTAdd, TTHadamard) ranks grow; rounding compresses back. Two-pass:

1. **Right-to-left orthogonalisation:** for k = d, d-1, ..., 2: QR-decompose G_k matricised as (r_{k-1}) × (n_k · r_k); replace G_k with Q^T factor; absorb R into G_{k-1}.
2. **Left-to-right SVD truncation:** for k = 1, ..., d-1: SVD on G_k matricised as (r_{k-1} · n_k) × r_k; truncate to ε; absorb singular values into G_{k+1}.

```go
func (tt *TT) Round(epsilon float64) *TT
func (tt *TT) RoundFixedRank(maxRank int) *TT
```

The orthogonalisation pass needs **QR decomposition** (slot 084-linalg-sota gap). Ship a thin Householder QR alongside SVD in T0a (50 LOC extension).

#### T7. TT-cross approximation (Oseledets-Tyrtyshnikov 2010, "DMRG-cross") — ~250 LOC
Sample-based TT construction for tensors that are *not* explicitly stored. Given an oracle `f(i_1, ..., i_d) → float64`, build the TT representation by *sampling* a tiny fraction of entries at adaptively-chosen pivots. Maximum-volume submatrix selection (Goreinov-Tyrtyshnikov-Zamarashkin 2010) drives pivot choice. Reduces from N^d entries to O(d·n·r²·N) samples with quasi-optimal error.

**This is the practical entry point for high-d PDE / parametric quantum chemistry / Bayesian inference.** TT-SVD requires the full tensor (impossible for d > 30); TT-cross builds the same compressed representation from oracle access.

```go
type Oracle func(idx []int) float64
func TTCross(oracle Oracle, shape []int, epsilon float64, maxIter int) (*TT, error)
```

Defer maximum-volume submatrix selection to a separate `linalg/maxvol.go` (50 LOC, Goreinov 2010 algorithm). Connective-tissue: needs the QR from T6.

#### T8. Tensor contraction order optimisation — ~250 LOC
Given a tensor network as a graph (nodes = tensors, edges = shared modes), the cost of contracting it depends on the order: NP-hard in general (Lam-Sadayappan-Wenger 1997). Practical heuristics:

- **Greedy minimum cost** (Markov-Shi 2008): at each step contract the pair with smallest immediate cost; O(V²·E) per step.
- **Optimal exhaustive for V ≤ 10** via dynamic programming on subsets (Pfeifer-Haegeman-Verstraete 2014).
- **Treewidth-bounded** for V ≤ 30 via tree decomposition (Markov-Shi 2008).

```go
type ContractionGraph struct {
    Tensors []*Tensor
    Edges   [][2]int  // pairs of (tensorIdx, modeIdx)
}
func OptimalContractionOrder(g *ContractionGraph) (order [][2]int, cost float64, err error)
func GreedyContractionOrder(g *ContractionGraph) ([][2]int, float64)
```

Connective-tissue: cross-link to slot 080 graph review (junction tree, treewidth bounds). The graph package already ships BFS/DFS/topological sort; extend with a tree-decomposition routine.

#### T9. Einstein summation notation parser — ~150 LOC
Parses the einsum string `"ij,jk->ik"` (matrix multiply) or `"ijk,jl,kl->il"` (3-tensor contraction) and dispatches to T0c primitives. Standard pattern (numpy.einsum / opt_einsum) but kept minimal: no broadcasting, no implicit ellipsis, just labeled-index contraction.

```go
func Einsum(spec string, tensors []*Tensor, out *Tensor) error
```

Drives developer ergonomics for the entire tensor package.

### Tier-2 — high demand, medium connective tissue (~1,500 LOC)

#### T10. DMRG (Density Matrix Renormalization Group, White 1992 / Schollwöck 2011) — ~400 LOC
Variational ground-state finder for 1-D quantum lattice Hamiltonians H represented as a Matrix Product Operator (MPO). Sweep through MPS sites left-to-right and back, at each site solving a small (r·n·r) eigenvalue problem to update the local tensor. Recovers ground-state energy with machine precision for gapped 1-D models in O(N·r³·dim_local³) per sweep.

```go
type MPO struct { Cores []*Tensor }  // similar to TT but with two physical modes per core
func MPSGroundState(h *MPO, mpsRank, sweeps int, tol float64) (*TT, energy float64, err error)
```

**The cornerstone of computational quantum many-body physics since 1992.** Cross-substrate parity: pin Heisenberg-1/2 chain at N=20 against exact diagonalisation. Connective-tissue: needs an iterative eigensolver beyond `QRAlgorithm` — Lanczos or Arnoldi (~150 LOC, slot 084-linalg-sota gap; this slot surfaces it).

#### T11. TEBD (Time-Evolving Block Decimation, Vidal 2003) — ~300 LOC
Real- or imaginary-time evolution of an MPS under a 1-D Hamiltonian H = Σ_i h_{i,i+1} via Trotter-Suzuki decomposition (2nd-order Suzuki 1985, 4th-order Forest-Ruth 1990) into local two-site gates; each gate application + SVD-truncation maintains the MPS form. Connective-tissue: TT machinery (T4) + matrix-exp `linalg.MatExp` (slot 081-linalg-missing gap, ~80 LOC Padé approximation).

#### T12. Hierarchical Tucker decomposition (Hackbusch-Kühn 2009) — ~250 LOC
Generalises Tucker to a *binary tree* of mode groupings: nodes hold transfer matrices, leaves hold mode factors. Unlike TT (which is a chain), HT can be more efficient when mode interactions are hierarchical rather than sequential. Storage O(d·n·r + d·r³). Niche but elegant; ships when a consumer pulls.

#### T13. Block-term decomposition (De Lathauwer 2008) — ~200 LOC
Generalises CP and Tucker: T ≈ Σ_r G_r ×_1 U_r^(1) ×_2 ... ×_d U_r^(d) — sum of *partial* Tuckers. Useful for blind source separation with structured signals (BTD-(L,L,1) for telecommunication channel identification).

#### T14. PEPS (Projected Entangled Pair States, Verstraete-Cirac 2004) — ~250 LOC ⊘ defer
2-D generalisation of MPS: tensors live on a 2-D lattice with virtual bonds along x and y. Exact contraction is #P-hard; practical evaluation uses **boundary MPS** approximation. Drives 2-D quantum many-body simulation.

**Defer until consumer pulls.** PEPS contraction is research-frontier; the implementation is high-LOC (boundary MPS itself is ~400 LOC) and the consumer surface today is empty. Reality should ship the *data structure* + a `PEPSToFull()` for tiny lattices (~150 LOC) so future expansion is unblocked, but skip the boundary-MPS contractor.

#### T15. MERA (Multi-scale Entanglement Renormalization Ansatz, Vidal 2007) — ~150 LOC ⊘ defer
Hierarchical tensor network for critical 1-D systems with logarithmic entanglement scaling. Layered structure of disentanglers + isometries. Variational optimisation requires Riemannian gradient descent on the Stiefel manifold (constrained-isometry).

**Defer.** MERA is a 2008-2014 era research topic with niche usage; ship only the data-structure stub + `MERAEvaluate` for trivial cases. Variational optimisation is high-LOC and demand is low compared to DMRG/TEBD.

### Tier-3 — niche / advanced (~700 LOC)

#### T16. Randomized tensor decompositions (Halko-Martinsson-Tropp 2011 + Battaglino-Ballard-Kolda 2018) — ~250 LOC
Randomized HOSVD: replace each mode-n SVD with a randomized SVD (T0a). For large tensors (n_k ≥ 1000) the speedup is order-of-magnitude. Battaglino-Ballard-Kolda 2018 extends randomized projections to CP-ALS via tensor sketching.

```go
func RandomizedHOSVD(T *Tensor, ranks []int, oversample int, rng RNG) (TuckerResult, error)
func RandomizedCPALS(T *Tensor, rank, sketchSize int, rng RNG) (CPResult, error)
```

#### T17. Sparse tensor formats: COO + CSF + sparse-rank-1-sums — ~150 LOC
Coordinate (COO) and Compressed Sparse Fiber (CSF, Smith-Karypis 2015) layouts for tensors with ≪ 100% density. Critical for graph-mining tensors (Smith-Karypis 2015 SPLATT) and recommender systems.

```go
type COOTensor struct { Indices [][]int; Values []float64; Shape []int }
type CSFTensor struct { ... }  // pointer arrays per mode
func (cooT *COOTensor) MTTKRP(mode int, factors []*linalg.Matrix, out []float64)  // matricised tensor times Khatri-Rao product, the sparse-CP-ALS workhorse
```

#### T18. Quantum circuit simulation as MPS — ~200 LOC
Each qubit ↔ TT mode of dim 2. Single-qubit gates ↔ multiply core by 2×2 matrix. Two-qubit gates ↔ TEBD step (T11). Measurement ↔ marginalisation via TT contraction. Schwarz-Cirac 2007 / Vidal 2003 framework. **Demonstrates that 1-D quantum-circuit simulation is polynomial when entanglement is bounded** — a foundational complexity-theoretic statement. Cross-link to slot 064-zkmark.

#### T19. AMEn / MINRES-TT linear solver (Dolgov-Savostyanov 2014) — ~250 LOC ⊘ defer
Solves A·x = b for A and b in TT form (high-d linear systems with ≥ 10^9 unknowns). Alternating minimisation with rank enrichment (AMEn). The state-of-the-art linear-solver-in-TT-format. Drives high-d PDE / parametric uncertainty quantification.

**Defer until consumer pulls.** Implementation is delicate (rank-enrichment heuristics, restart logic) and reality has no current high-d-PDE consumer.

#### T20. Tensor network for high-d PDE (curse-of-dimensionality break) — ~150 LOC
Galerkin discretisation of d-dimensional Poisson on [0,1]^d gives an N^d × N^d sparse matrix; representing the right-hand side, solution, and operator in TT format reduces storage from N^d to O(d·N·r²) and the time-step from O(N^d) to O(d·N²·r²). The full pipeline uses TT-cross (T7) for the RHS, AMEn (T19) for the solve, TT-rounding (T6) for compression. Defer until T19 ships.

---

## (3) Connective tissue — what each new edge buys

Nine cross-package edges activate once `tensor/` and `linalg/svd.go` land:

| Edge | LOC of glue | What it unlocks |
|---|---|---|
| `linalg/svd.go → linalg/pca.go` | -50 (refactor) | PCA reuses SVD instead of inverse iteration; cleaner, faster, more stable |
| `linalg/svd.go → optim/proximal/` | 50 | Nuclear-norm proximal operator (slot 102 deferral roster) for low-rank matrix completion |
| `linalg/svd.go → autodiff/` | 80 | Differentiable SVD via Townsend-2016 / Wan-Zhang 2019 closed-form gradients; enables backprop through tensor decompositions |
| `tensor/ → linalg/` | 0 | Mode-n products dispatch to existing MatMul; full tensor algebra rests on existing matrix substrate |
| `tensor/ → autodiff/` | 100 | Tensor-valued autodiff Variables; differentiable CP / Tucker / TT-SVD for tensor regression |
| `tensor/ → optim/` | 30 | ALS inner-loops use existing L-BFGS for non-linear refinement (rare, but a stretch goal) |
| `tensor/ → graph/` | 50 | Tensor-network contraction order optimisation uses graph algorithms; cross-link slot 074 graph-missing |
| `tensor/ → signal/` | 0 | FFT-based 1-D MPS quantum-circuit simulation (T18 + signal/fft.go); already callable |
| `tensor/ → topology/persistent/` | 0 — defer | Persistent homology of tensor-rank stratification |

Two **new packages** appear if the full roster ships: `tensor/` (~5,000 LOC) and a substantial extension to `linalg/` (~700 LOC for SVD + QR + Lanczos + maxvol + matrix-exp). No existing package needs an API break.

---

## (4) Three architectural recommendations

**F1. Ship `linalg/svd.go` (T0a) as a separate PR before any tensor primitive.** SVD unblocks **six other cutting-edge slots** (102 optim-missing nuclear-norm proximal, 184 synergy-linalg-prob whitening, 153 prob-sota Bayesian PCA, 084 linalg-sota randomized methods, 168 physics-autodiff differentiable spectral methods, this slot). Promote the private `tridiagonalize` Householder machinery to a public `householderReflector` helper and add bidiagonalisation. **One-week effort. Must land first.** The same PR should add `linalg/qr.go` (~150 LOC) and `linalg/lanczos.go` (~150 LOC) since they reuse the same Householder primitive and unblock T6 + T10.

**F2. Establish `tensor.Tensor` as a thin wrapper over `[]float64` with explicit shape, never as a heavy generic type.** Mirrors reality's existing convention of "flat slice + dimension args" used in `linalg.MatMul(A, aRows, aCols, B, bCols, out)`. Operations are zero-allocation by accepting pre-allocated `out *Tensor`. Avoids Go's generic-type-explosion and aligns with reality's "no allocations in hot paths" rule.

```go
// Idiomatic: caller provides pre-allocated workspace.
func ModeNProduct(t *Tensor, mode int, M []float64, mRows, mCols int, out *Tensor)
```

**F3. Pin tensor-decomposition error bounds via golden files, not just function correctness.** The killer claims of this field are *quantitative*:

- HOSVD quasi-optimality: `||T − T_HOSVD||_F ≤ √d · ||T − T_best||_F` (De Lathauwer 2000)
- TT-SVD error: `||T − T_TT||_F² ≤ Σ_k ε_k²` (Oseledets 2011 Theorem 2.2)
- Hilbert-tensor compression at d=10, n=8: 10⁸ entries → ~kilobytes at 1e-8 reconstruction error (Oseledets 2011 Example 4.1)
- DMRG ground-state energy: machine precision for gapped 1-D Heisenberg N=20 vs exact diagonalisation (Schollwöck 2011)

**These bound-pinning tests are the cross-language parity contract.** Without them an SVD-with-truncation looks "approximately right" but doesn't *prove* the curse-of-dimensionality break that motivates the entire field.

---

## (5) Risks and gotchas

- **G1. SVD numerical instability near zero singular values.** Truncated SVD with relative tolerance 1e-15 is unsafe; use absolute floor `max(σ_max · ε_relative, 1e-300)` (LAPACK convention). Document in T0a.
- **G2. CP-ALS local minima.** CP is non-convex; default to multi-start with `restarts=5` deterministic seeds. Cross-language parity must use the *same* seed sequence.
- **G3. CP rank not lower-semi-continuous.** A rank-R tensor's best-rank-(R-1) approximation may not exist (border rank issue, De Silva-Lim 2008). CP-ALS may diverge as iterates blow up. Ship `MaxFactorNorm` early-stop guard.
- **G4. TT-rank ≠ matrix-rank intuition.** TT-rank is *position-dependent*: r_k is the rank of the matricisation T_(1:k) × (k+1:d). Order of modes matters; bad ordering can blow up TT-rank exponentially. Document and provide `RecommendedModeOrdering(T) []int` heuristic.
- **G5. HOSVD ranks not all-equal.** Each mode has its own truncation; `ranks []int` argument needs len(ranks) == d, not a single int. Common API confusion; surface in error messages.
- **G6. DMRG eigensolver tolerance vs sweep convergence.** Tight inner-eigensolver tolerance (1e-12) is wasteful in early sweeps; loose tolerance (1e-4) misses true ground state in late sweeps. Standard practice: tolerance scales with sweep number. Ship as `dmrg.AdaptiveTolerance` option.
- **G7. Mode-n unfolding convention drift.** Kolda-Bader 2009 §2 uses one ordering; LAPACK / numpy use a different (transposed) convention. Pick *one* and pin it byte-for-byte against numpy.tensorly for cross-language parity. Document loud-fail when consumer mixes conventions.
- **G8. Contraction order optimality is NP-hard.** Don't claim "optimal" without explicit `OptimalContractionOrder` (small V) vs `GreedyContractionOrder` (large V) split. Document the V≤10 / V≤30 / V>30 regime boundaries.

---

## (6) Cross-language parity targets

Eight pinned tests covering the foundational error bounds and decomposition uniqueness:

| Test | Pin | Tolerance | Reference |
|---|---|---|---|
| `TestSVDReconstruction_Random10x8` | `||A − U·Σ·V^T||_F` | 1e-12 | Golub-Reinsch 1970 |
| `TestSVDSingularValues_Hilbert` | σ_i vs analytic | 1e-10 | Hilbert matrix is famous SVD test |
| `TestHOSVDQuasiOptimality_d4n5r2` | error ≤ √4 · best-rank-2 | bound | De Lathauwer 2000 Thm 5.4 |
| `TestTTSVD_HilbertTensor_d10n8` | TT-rank ≤ 8, recon error ≤ 1e-8 | structural + 1e-8 | Oseledets 2011 Example 4.1 |
| `TestTTRoundingPreservesNorm` | `||T_rounded||_F` vs original | 1e-12 | Oseledets 2011 §3 |
| `TestCPALSConvergence_Synthetic_R3` | converges to fit < 1e-6 in 100 iter | reproduces input | Kolda-Bader 2009 Algorithm 3.1 |
| `TestDMRG_Heisenberg_N12` | ground energy vs ED | 1e-10 | Schollwöck 2011 §6.2 |
| `TestEinsum_MatrixMultiply` | `"ij,jk->ik"` matches MatMul | 1e-14 | trivial sanity |

---

## (7) Verdict

**Ship Tier-0 + Tier-1 (~2,500 LOC over 6-8 sprints):**
- Sprint 1: T0a `linalg/svd.go` (400) + `linalg/qr.go` (150) — substrate for *six* slots
- Sprint 2: T0b `tensor/tensor.go` (150) + T0c `tensor/contract.go` (150) — array type + primitives
- Sprint 3: T1 CP-ALS (300) + T2 HOSVD/HOOI (300) — classical higher-order decompositions
- Sprint 4: T3 HOSVD-Adaptive (150) + T4 TT representation (250) — TT data structure
- Sprint 5: T5 TT-SVD (250) ⭐ — Oseledets 2011 flagship
- Sprint 6: T6 TT-rounding (200) + T8 contraction order (250) — TT arithmetic toolkit
- Sprint 7: T7 TT-cross (250) + T9 Einsum (150) — sample-based + ergonomics
- Sprint 8: cross-substrate parity tests + documentation polish

**Defer-but-design Tier-2 (~1,400 LOC, ship when consumer pulls):** T10 DMRG, T11 TEBD, T12 HT, T13 BTD, T18 quantum-circuit MPS.

**Drop until consumer pulls:** T14 PEPS, T15 MERA, T19 AMEn, T20 high-d PDE pipeline. Each is a research-frontier capability whose downstream consumer doesn't exist in reality's stack today.

**Single-highest-leverage 1-day project:** T0a SVD on its own (~400 LOC). Unblocks every primitive in this slot *and* six others. **The single highest-priority addition to reality across all of Block C.**

**Single-highest-leverage cutting-edge piece:** T5 TT-SVD (Oseledets 2011). Order-of-magnitude exponential compression for high-d tensors with provable global error bound, ~250 LOC of orchestration over T0a + T0b + T0c. **The flagship deliverable for this slot.** No off-the-shelf zero-dependency math library ships TT-SVD; tensorly (Python) does, but tensorly depends on numpy/scipy/torch and isn't a library reality can mirror byte-for-byte. TT-SVD is *the* primitive that would let reality claim "we ship the curse-of-dimensionality break", which is a genuine cutting-edge math story Block C exists to surface.

**Cross-slot synergy callouts:**
- Slot 081 linalg-missing (and 084 linalg-sota): SVD/QR/Lanczos must surface there too as flagged gaps. This slot's T0a is the natural shared deliverable.
- Slot 102 optim-missing: nuclear-norm proximal operator (already flagged "needs SVD") immediately lights up once T0a lands.
- Slot 184 synergy-linalg-prob: SRHT randomized projections share the Halko-Martinsson-Tropp 2011 framework with T0a's RandomizedSVD and T16's randomized HOSVD.
- Slot 168 physics-autodiff: differentiable SVD (Townsend 2016) is the entry point for differentiable tensor decompositions.
- Slot 074 graph-missing: tree decomposition / treewidth feeds T8 contraction-order optimisation.

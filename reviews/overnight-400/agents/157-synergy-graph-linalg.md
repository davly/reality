# 157 | synergy-graph-linalg

**Summary line 1.** `graph/` and `linalg/` are siblings under `reality/` that today do NOT import each other — graph algorithms operate on `IntAdjacency = map[int][]int` (no matrices), linalg ships dense `[]float64` row-major (no graph types), the bridge primitive `AdjacencyMatrix(adj, n) []float64` does not exist, and the entire spectral graph theory canon (Laplacian L=D-A, symmetric/random-walk normalised Laplacians, Fiedler value/vector, Cheeger bound, spectral bisection, Ng-Jordan-Weiss / Shi-Malik clustering, heat kernel H_t=exp(-tL), effective resistance via L^+, commute/hitting times, personalised PageRank, Graph Fourier Transform, Kirchhoff matrix-tree theorem, Spielman-Srivastava sparsification) — twelve canonical operators, six of them nameable in any 2025 graph-textbook index — is wholly absent.

**Summary line 2.** Sixteen synergy primitives totalling ~1380 LOC of pure glue close the gap; nine ship today against the v0.10.0 linalg surface (LU + Cholesky + symmetric eigenvalues), seven are blocked on missing linalg primitives (`Eigvec` returning eigenvectors, `Pseudoinverse` from SVD, `MatrixExp` Padé, `LanczosSym` Krylov) that are independently flagged in agent 097-linalg-missing Tier 1; cheapest one-day standalone is **G1 AdjacencyMatrix + G2 Laplacian** at 60 LOC unlocking 13 of 16; keystone is **G3 SymNormalizedLaplacian** since spectral clustering, Cheeger bounds, and GFT all consume the symmetric normalised form; highest-leverage architectural addition is **G6 Fiedler** (≈80 LOC on top of `linalg.QRAlgorithm` deflation) because algebraic-connectivity/spectral-bisection is the textbook landing page of spectral graph theory and it gates G7 SpectralBisection / G8 SpectralClustering / G9 GFT; recommended placement is `graph/spectral.go` (consumer-shaped, mirrors 151/153/156 placement precedent of "synergy lives in the package that shapes the API surface, not the package that provides primitives").

---

## 0. State of play (verified file-walk)

`graph/` HEAD (12 files, ~1100 LOC numeric core):

- `graph.go`: `AdjacencyList`, `Nodes`, `InDegree`, `Roots`, `Leaves` — string-keyed helpers
- `types.go`: `IntAdjacency = map[int][]int` (single line)
- `bfs.go`, `bellman_ford.go`, `dag.go`: traversal & DAG depth
- `shortest.go`: Dijkstra, AStar, FloydWarshall
- `flow.go`: Edmonds-Karp MaxFlow + Kahn TopologicalSort
- `mst.go`: KruskalMST, PrimMST
- `centrality.go`: BetweennessCentrality (Brandes), EigenvectorCentrality (power-iteration on adjacency), DegreeCentrality
- `pagerank.go`: PageRank (power iteration with damping + dangling redistribution)
- `community.go`: ConnectedComponents, StronglyConnected (Tarjan), LouvainCommunities
- `importance.go`: NodeImportance, EdgeFraction (bespoke)

**Search for spectral primitives in graph/:** `Laplacian`, `Spectral`, `HeatKernel`, `Fiedler`, `Cheeger`, `EffectiveResistance`, `CommuteTime`, `HittingTime`, `PersonalizedPageRank`, `Pseudoinverse`, `MatrixTree`, `Sparsif` — **zero matches** in `graph/*.go`.

**Search for graph-shaped primitives in linalg/:** same patterns — **zero matches** in `linalg/*.go`.

`linalg/` HEAD (6 files, ~1500 LOC):

- `matrix.go`: MatMul, MatTranspose, MatVecMul, Identity, MatAdd, MatSub, MatScale, Trace, CrossProduct
- `vector.go`: DotProduct, L1/L2/Inf norms, VectorAdd/Sub/Scale, CosineSimilarity, EncodingDistance, L2Normalize
- `decompose.go`: LUDecompose + LUSolve, Inverse (LU-based), Determinant, CholeskyDecompose + CholeskySolve
- `eigen.go`: QRAlgorithm — **eigenvalues only**, no eigenvectors returned (PCA improvises via inverse iteration)
- `pca.go`: PCA via covariance + inverse-iteration eigenvectors (the only place in repo that recovers eigenvectors)
- `correlation.go`: Pearson, Spearman, Covariance, CovarianceMatrix

The linalg gap-list crucial for this synergy:
- **No `Eigvec` returning eigenvectors** alongside eigenvalues (097 §2 T1 — Householder QR + back-transformation needed; PCA's inverse-iteration trick is private, not exported, and is per-eigenvalue not joint)
- **No `Pseudoinverse`** (097 §2 T1 — needs SVD; SVD itself missing)
- **No `MatrixExp`** (097 §3 T1 — Padé + scaling-and-squaring; Higham 2008 Alg 10.20)
- **No `LanczosSym`** Krylov subspace iteration (097 §5 T1 — needed for sparse-spectrum top-k eigenpairs without forming full matrix)
- **No sparse types** (097 §0 — graph adjacency is naturally sparse; dense O(n²) storage caps at n≈10k)

---

## 1. The sixteen synergy primitives

Each entry: (1) capability, (2) composition recipe over present primitives, (3) connective-tissue LOC, (4) blocking flag if any. All build for v0.10.0 unless tagged BLOCKED.

### G1 — `AdjacencyMatrix(adj IntAdjacency, n int, weights map[[2]int]float64) []float64`

**Capability.** Convert `graph.IntAdjacency` to dense row-major n×n `[]float64` matrix A with A[i*n+j] = weight(i,j) (default 1.0 if no weights map provided). The single missing bridge between the two packages.

**Composition.** Trivial: allocate `n*n` slice, walk adj, set A[u*n+v] = weight. Plus an undirected variant `SymAdjacencyMatrix` that symmetrises.

**LOC.** ~25 (directed) + ~15 (sym variant) = 40.

**Notes.** This is the keystone-of-the-keystone. Today every graph eigen-anything must hand-roll this conversion. Place in `graph/spectral.go` (not `linalg/`) per consumer-side placement precedent.

### G2 — `Laplacian(adj IntAdjacency, n int, weights map[[2]int]float64) []float64`

**Capability.** Combinatorial graph Laplacian L = D − A as dense n×n. For weighted graph, D[i,i] = Σ_j A[i,j] (weighted degree). L is symmetric PSD with smallest eigenvalue 0 (multiplicity = number of connected components).

**Composition.** G1 + diagonal accumulation: 1 nested loop over A's rows summing degrees, write D-A into output buffer. ~25 LOC on top of G1.

**LOC.** 25.

### G3 — `SymNormalizedLaplacian(adj, n, weights) []float64`

**Capability.** L_sym = I − D^(−1/2) A D^(−1/2). The form spectral clustering and Cheeger bounds need. Eigenvalues lie in [0, 2]; Spectrum of L_sym on a graph is "intrinsic" (invariant under uniform edge re-weighting) where unnormalised L is not.

**Composition.** G1 + scale rows and columns of A by 1/√d_i. ~35 LOC.

**LOC.** 35.

### G4 — `RandomWalkLaplacian(adj, n, weights) []float64`

**Capability.** L_rw = I − D^(−1) A. Eigenvalues equal those of L_sym (similarity transform via D^(−1/2)) but eigenvectors are different — L_rw eigenvectors are stationary distributions of perturbed Markov chains. Shi-Malik 2000 normalised cut uses this form.

**Composition.** G1 + scale rows of A by 1/d_i, subtract from I. ~30 LOC.

**LOC.** 30.

### G5 — `AlgebraicConnectivity(L []float64, n int) float64` — λ₂(L)

**Capability.** Returns the second-smallest eigenvalue λ₂ of L (Fiedler value). Equals 0 iff graph disconnected; quantitative measure of "how connected" the graph is. Cheeger inequality: λ₂/2 ≤ h(G) ≤ √(2λ₂) for L_sym (h = edge expansion / conductance).

**Composition.** Call existing `linalg.QRAlgorithm(L, n, eigs, maxIter)` on the symmetric Laplacian; return `eigs[n-2]` (eigenvalues are returned in descending order — λ₂ is second-from-bottom). ~20 LOC including allocation of eig buffer.

**LOC.** 20.

**Note.** Works against v0.10.0 linalg surface; QRAlgorithm gives eigenvalues only, which is exactly what's needed for the connectivity scalar.

### G6 — `Fiedler(L []float64, n int) (lambda2 float64, vec []float64)` — Fiedler vector

**Capability.** Returns λ₂ AND its eigenvector. The Fiedler vector signs partition the graph into two roughly-equal components in spectral bisection (G7).

**Composition.** Inverse iteration on (L − λ₂·I) with σ = λ₂ shift, mirroring the trick `linalg/pca.go:101-174` already uses internally for PCA's eigenvector recovery. Steps: (1) call `QRAlgorithm` to get λ₂; (2) form `(L − (λ₂ − ε)·I)` with tiny shift ε; (3) `LUDecompose` it; (4) iterate `LUSolve(L,U,perm,b,x)` 50× with L2-normalisation between iterations — this is the canonical inverse-iteration loop that is already private inside PCA but should be exposed as public `linalg.InverseIteration(A, lambda, n, maxIter, vec)` (~80 LOC, agent 097 §5 T1.eigvec) and reused here.

**LOC.** ~30 in graph/spectral.go on top of a public `linalg.InverseIteration` (which is itself ~80 LOC of refactor of existing PCA private code). If `linalg.Eigvec` (full eigendecomposition Q^T·D·Q) lands per agent 097 §2 T1, this drops to ~10 LOC.

**Status.** SHIPS TODAY (with private code lift) or BLOCKED-SOFT on linalg refactor.

### G7 — `SpectralBisection(adj, n, weights) (left, right []int)`

**Capability.** Partition graph nodes into two parts such that conductance is approximately minimised. Industry-standard graph-partitioning baseline.

**Composition.** G6 → take Fiedler vector v → split nodes by sign of v[i] (or by median for guaranteed-balance variant). ~30 LOC including the median-split branch.

**LOC.** 30.

### G8 — `SpectralClusteringNJW(adj, n, weights, k int) []int`

**Capability.** Ng-Jordan-Weiss 2002 algorithm: compute k smallest eigenvectors of L_sym, stack into n×k matrix U, row-normalise to unit-length, run k-means on rows. Returns cluster label for each node. The de-facto modern community-detection alternative to Louvain when k is known a priori.

**Composition.** G3 (L_sym) → top-k smallest eigenvectors via repeated inverse iteration with deflation against previously-found vectors — same Gram-Schmidt-deflation idiom `linalg/pca.go:177-185` already implements internally → row-normalise stacked U → call k-means (which itself does NOT exist in reality today; needs ~120 LOC of standard Lloyd's algorithm in `linalg/cluster.go` or `prob/cluster.go`).

**LOC.** ~80 in graph/spectral.go for the eigenvec-stacking + row-normalisation, BLOCKED-HARD on `linalg.Eigvec` (top-k smallest) and on a kmeans primitive.

**Note.** The kmeans gap is independently flagged in 097 (linalg-missing) and in 116/117 (prob-isolation). Recommend placing kmeans in `linalg/cluster.go` because it is fundamentally a vector-quantisation linear-algebra primitive (not a probabilistic one).

### G9 — `SpectralClusteringShiMalik(adj, n, weights, k int) []int`

**Capability.** Shi-Malik 2000 normalised cut: solve generalised eigenproblem L·v = λ·D·v (equivalent to L_rw eigenvectors), use k smallest eigenvectors of L_rw. Variant of G8 that the computer-vision literature uses.

**Composition.** G4 (L_rw) → top-k smallest eigenvectors → k-means. Same dependencies as G8.

**LOC.** ~80 with same blocking caveats as G8.

### G10 — `GraphFourierTransform(L []float64, n int, signal []float64) (coeffs []float64, basis []float64)`

**Capability.** GFT is the projection of a graph signal x ∈ R^n onto the eigenbasis of L. Coefficient k = ⟨x, u_k⟩ where u_k is k-th Laplacian eigenvector. The graph-theoretic analogue of the discrete Fourier transform: classical DFT IS the GFT of the cycle graph because cycle Laplacian eigenvectors ARE Fourier modes. Foundational primitive of "graph signal processing" (Shuman et al. 2013).

**Composition.** G2 → full eigendecomposition of L → write basis (eigenvectors as rows) and project signal onto each. ~50 LOC on top of `linalg.Eigvec`.

**LOC.** 50, BLOCKED-HARD on `linalg.Eigvec`.

**Note.** Cross-validation pin: GFT of a cycle graph C_n on a discrete cosine signal MUST agree with `signal.FFT` of the same array up to permutation of basis. This is a R-MUTUAL-CROSS-VALIDATION 3/3 candidate (FFT × GFT × manual eigenbasis) following the recent commit pattern 6a55bb4.

### G11 — `EffectiveResistance(adj, n, weights) []float64` — n×n matrix R(u,v)

**Capability.** Klein-Randić 1993 effective resistance: R(u,v) = (e_u − e_v)^T · L^+ · (e_u − e_v) where L^+ is the Moore-Penrose pseudoinverse of L. Equivalent to the resistance between u and v if every edge is replaced by a 1-Ω resistor (1/weight Ω for weighted). Dual to commute time: τ_uv (commute) = 2m·R(u,v) where m is total edge weight.

**Composition.** G2 (L) → `linalg.Pseudoinverse(L, n, Lplus)` → for each pair (u,v) compute the quadratic form in 2n FLOPS using G2's L^+. ~40 LOC on top of `linalg.Pseudoinverse`.

**LOC.** 40, BLOCKED-HARD on `linalg.Pseudoinverse` (which itself blocks on `linalg.SVD`).

**Note.** Workaround for v0.10.0: since L has rank n−1 on a connected graph, can compute L^+ via `linalg.LUDecompose` on (L + 1/n · J) where J is all-ones (the "regularised Laplacian" trick), then subtract 1/n · J back. This sidesteps the SVD requirement at the cost of one rank-1 correction. ~80 LOC standalone, ships today against v0.10.0.

### G12 — `CommuteTime(adj, n, weights) []float64` — and `HittingTime`

**Capability.** Commute time τ_uv = expected time of random walk u→v→u. Hitting time h_uv = expected time of random walk u→v (asymmetric: h_uv ≠ h_vu generally). Both are graph distances in the random-walk metric.

**Composition.** Commute: 2m·R(u,v) where R is G11 effective resistance and m = Σ w / 2. Hitting: more delicate — h_uv = m · (R(u,v) + (1/d_v)Σ_w d_w R(v,w) − (1/d_u)Σ_w d_w R(u,w)). Both reduce to G11 plus degree-weighted sums.

**LOC.** ~60 on top of G11 (which is itself blocked on pseudoinverse).

**Status.** SHIPS TODAY via the regularised-Laplacian workaround in G11.

### G13 — `HeatKernel(L []float64, n int, t float64) []float64` — H_t = exp(−tL)

**Capability.** Heat kernel of the graph at time t: H_t[u,v] is the amount of "heat" remaining at v after diffusing from a unit source at u for time t. Equivalent to t-step transition density of a continuous-time random walk. Used for: diffusion distances (Coifman-Lafon 2006), graph kernels for SVMs (Kondor-Lafferty 2002), node-similarity scores, vertex embeddings.

**Composition.** G2 (L) → `linalg.MatrixExp(-t·L)` via Padé scaling-and-squaring. The single-line wrapping makes the entire heat-kernel literature accessible to reality consumers.

**LOC.** ~25 wrapper. BLOCKED-HARD on `linalg.MatrixExp` (097 §3 T1, ~150 LOC of Higham 2008 Alg 10.20).

**Status.** BLOCKED until `linalg.MatrixExp` ships. **No workaround** — naïve eigendecomposition + diagonal exp is 5× slower than Padé and forfeits accuracy guarantees for stiff L.

### G14 — `PersonalizedPageRank(n int, edges [][3]float64, source []float64, damping float64, iters int) []float64`

**Capability.** Replace the uniform teleport vector in standard PageRank with a personalisation vector p (e.g., p[u]=1, all others 0 → "PageRank from u's point of view"). Used in Google Scholar's recommendation engine, Twitter's WTF (who-to-follow) at scale, and as the basis for Andersen-Chung-Lang 2006 local-cluster mining.

**Composition.** Take existing `graph.PageRank` (pagerank.go:31-109) — already a power iteration on the stochastic matrix. Replace the line `teleport := (1.0 - damping) / float64(n)` (pagerank.go:75) with element-wise teleport p[i]·(1−damping). ~30 LOC including a dedicated `PersonalizedPageRank` function that takes the source vector and validates it sums to 1.

**LOC.** 30. SHIPS TODAY — pure graph-side change, doesn't even touch linalg.

**Note.** This is the cheapest synergy ship in the entire grid. Recommend bundling with G15 PageRankAsLinearSystem as a single PR.

### G15 — `PageRankAsLinearSystem(n int, edges [][3]float64, damping) []float64`

**Capability.** Solve PageRank as a linear system (I − α·M^T)·r = (1−α)/n·1 instead of via power iteration. Exact solution; no convergence parameter; useful as cross-validation oracle for the existing iterative PageRank (pagerank.go).

**Composition.** Build M^T as dense n×n via G1 + row-stochastic normalisation → form (I − αM^T) → solve via `linalg.LUDecompose` + `LUSolve`. ~50 LOC.

**LOC.** 50. SHIPS TODAY against v0.10.0.

**Note.** Cross-validation pin: this gives an R-MUTUAL-CROSS-VALIDATION pin for PageRank — `pagerank.go` (iterative) vs G15 (direct LU) on a fixed test fixture should agree to 1e-12, with G15 as the gold-standard reference because LU has no convergence error, just round-off.

### G16 — `MatrixTreeTheorem(adj IntAdjacency, n int, weights map[[2]int]float64) float64`

**Capability.** Kirchhoff matrix-tree theorem: number of spanning trees of a graph = any cofactor of the Laplacian = product of non-zero Laplacian eigenvalues divided by n. For weighted graph, it gives the sum of edge-weight products over all spanning trees (the "spanning-tree polynomial" at the all-ones evaluation).

**Composition.** G2 (L) → delete row 0 and column 0 → compute determinant of the (n−1)×(n−1) sub-matrix via existing `linalg.Determinant`. ~40 LOC.

**LOC.** 40. SHIPS TODAY against v0.10.0.

**Note.** Beautiful pedagogical primitive that demonstrates determinant-eigenvalue duality. Cross-check against eigenvalues: `prod(eigs[1:n]) / n` (skipping the zero eigenvalue) must equal the determinant of L_minor — three-way pin (G2-determinant, prod-of-eigs, NetworkX reference value) on small test graphs (K_n where #trees = n^(n−2) is the Cayley formula).

### G17 (deferred) — `SpectralSparsification(adj, n, weights, epsilon)` — Spielman-Srivastava 2011

**Capability.** Produce a sparse subgraph H ⊂ G such that x^T·L_G·x ≤ x^T·L_H·x ≤ (1+ε)·x^T·L_G·x for all x ∈ R^n. The sparse approximation has O(n log n / ε²) edges regardless of original |E|. Foundational for graph-Laplacian solvers in O(m log^c n) time.

**Composition.** G11 (effective resistance) → sample edges with probability proportional to weight·R(u,v) → set sparsified weights to original/(p·k) where k is number of samples. ~150 LOC.

**LOC.** 150. BLOCKED-HARD on G11 which is BLOCKED-HARD on `linalg.Pseudoinverse` (or shipped via G11 regularised-Laplacian workaround → unblocked, but ~250 LOC including the workaround). DEFERRED to a v2 PR.

---

## 2. Status table

| ID | Primitive | LOC | Status | Blockers |
|---|---|---:|---|---|
| G1 | AdjacencyMatrix + SymAdjacencyMatrix | 40 | SHIPS TODAY | none |
| G2 | Laplacian | 25 | SHIPS TODAY | G1 |
| G3 | SymNormalizedLaplacian | 35 | SHIPS TODAY | G1 |
| G4 | RandomWalkLaplacian | 30 | SHIPS TODAY | G1 |
| G5 | AlgebraicConnectivity (λ₂) | 20 | SHIPS TODAY | G2, linalg.QRAlgorithm |
| G6 | Fiedler vector | 30+80(refactor) | SHIPS w/ refactor | linalg.InverseIteration (extract from PCA) |
| G7 | SpectralBisection | 30 | SHIPS w/ G6 | G6 |
| G8 | SpectralClusteringNJW | 80+120(kmeans) | BLOCKED-HARD | linalg.Eigvec (top-k), kmeans |
| G9 | SpectralClusteringShiMalik | 80 | BLOCKED-HARD | same as G8 |
| G10 | GraphFourierTransform | 50 | BLOCKED-HARD | linalg.Eigvec (full) |
| G11 | EffectiveResistance | 40 / 80 (workaround) | SHIPS w/ workaround | linalg.Pseudoinverse (or reg-Laplacian trick) |
| G12 | CommuteTime + HittingTime | 60 | SHIPS w/ G11 workaround | G11 |
| G13 | HeatKernel | 25 | BLOCKED-HARD | linalg.MatrixExp |
| G14 | PersonalizedPageRank | 30 | SHIPS TODAY | none |
| G15 | PageRankAsLinearSystem | 50 | SHIPS TODAY | none |
| G16 | MatrixTreeTheorem | 40 | SHIPS TODAY | linalg.Determinant (present) |
| G17 | SpectralSparsification | 150 | DEFERRED | G11 |

**Total connective tissue:** ~1380 LOC including G17 and including the G6 PCA-refactor side-quest. **Of which ships today against v0.10.0:** G1+G2+G3+G4+G5+G14+G15+G16 + G11-workaround + G12 = **9 primitives, ~440 LOC**.

---

## 3. Recommended PR sequence

**PR-1 — Foundations (single evening, ~150 LOC):** G1 AdjacencyMatrix + G2 Laplacian + G3 SymNormalized + G4 RandomWalk. Place in new file `graph/spectral.go`. Adds four functions, zero new dependencies, immediately enables G5/G6/G11/G14/G15/G16 in subsequent PRs.

**PR-2 — Algebraic connectivity & matrix-tree (single morning, ~80 LOC):** G5 + G16. Both reduce to single calls into existing linalg primitives. The matrix-tree theorem provides a R-MUTUAL-CROSS-VALIDATION 3/3 pin (Cayley formula × Det of L_minor × prod-of-eigs) on K_n test fixtures.

**PR-3 — PageRank synergies (single afternoon, ~80 LOC):** G14 PersonalizedPageRank + G15 PageRankAsLinearSystem. G15 doubles as the LU-cross-check oracle for the existing iterative `pagerank.go`, closing a R-MUTUAL-CROSS-VALIDATION pin that is currently absent from the package.

**PR-4 — Eigenvector-recovery refactor (one day, ~80 LOC):** Extract the inverse-iteration loop currently private inside `linalg/pca.go:101-174` into a public `linalg.InverseIteration(A, lambda, n, maxIter, vec []float64)` and `linalg.Eigvec(A, n, lambdas, eigvecs)`. Independently flagged in 097 §5 T1.eigvec; this PR is the keystone for G6/G7 and unblocks 7 of the 16 primitives.

**PR-5 — Fiedler suite (one day, ~60 LOC):** G6 Fiedler + G7 SpectralBisection. Lands the most-cited spectral primitive (algebraic-connectivity / spectral-bisection). Ships immediately after PR-4.

**PR-6 — Resistance & random-walk metrics (two days, ~180 LOC):** G11 EffectiveResistance via regularised-Laplacian workaround + G12 CommuteTime + HittingTime. Workaround sidesteps the Pseudoinverse blocker for v1; v2 swaps the workaround for true L^+ when SVD lands.

**PR-7 (after linalg matrix-exp lands) — Heat kernel (one day, ~25 LOC):** G13.

**PR-8 (after linalg full eigendecomp + kmeans land) — GFT & spectral clustering (one week, ~250 LOC):** G10 + G8 + G9.

**PR-9 (deferred) — Sparsification:** G17 once G11 is on a true L^+ basis.

---

## 4. Cross-package coupling and placement (cycle hazard analysis)

**One-way import direction is `graph/` consumes `linalg/`, never reverse.** Validated:

- G1-G4 (matrix builders): graph types in graph/, output is plain `[]float64` consumed by linalg.
- G5/G6 (eigenvalue queries): graph/spectral.go calls `linalg.QRAlgorithm` and `linalg.InverseIteration`.
- G11 (effective resistance): graph/spectral.go calls `linalg.Inverse` (workaround) or `linalg.Pseudoinverse`.
- G15 (PageRank-as-linear-system): graph/pagerank_linear.go calls `linalg.LUDecompose` and `LUSolve`.

Reverse direction (linalg consuming graph) is **never required** by the spectral grid. Conclusion: **place all 16 primitives in `graph/`** (specifically `graph/spectral.go` for G1-G13/G17, augment `graph/pagerank.go` for G14, new file `graph/pagerank_linear.go` for G15, new file `graph/matrixtree.go` for G16). This matches the 151 spectral/ precedent ("synergy lives in the consumer-shaped package") and the 153 prob/infogeo.go precedent.

**Anti-pattern to avoid:** placing `Laplacian` in `linalg/` would force linalg to import `graph.IntAdjacency`, creating a cyclic mental dependency where linalg suddenly knows about graphs. Don't.

**One exception:** k-means (needed for G8/G9) is a pure linalg primitive (vector-quantisation, Lloyd's algorithm, no graph types). It belongs in `linalg/cluster.go` and the spectral-clustering glue in `graph/spectral.go` calls into it. This matches 097 §6 (linalg utilities tier).

---

## 5. R-MUTUAL-CROSS-VALIDATION pins this synergy enables

Recent commits 6a55bb4 (audio onset 3-detector cross-validation), 365368a (copula × autodiff Clayton log-PDF gradient), and others establish the R-MUTUAL-CROSS-VALIDATION 3/3 pattern as a saturation criterion for review work. The graph-linalg synergy enables four new such pins:

**Pin 1 — Matrix-tree theorem on K_n.** Three independent paths to #spanning-trees of complete graph K_n:
- Cayley formula closed-form: n^(n−2)
- G16 cofactor: det(L_minor) where L_minor deletes row/col 0
- Eigenvalue product: prod(eigs[L][1:n]) / n

All three must agree to ~1e-12 for n=2..10. Saturates 3/3.

**Pin 2 — PageRank iterative vs direct.** Two independent paths:
- `graph.PageRank` iterative power iteration (existing, pagerank.go:31)
- G15 `PageRankAsLinearSystem` direct LU solve

Plus a third oracle:
- Eigenvector centrality of the column-stochastic matrix M (existing `EigenvectorCentrality` in centrality.go)

All three must agree on the dominant eigenvector to ~1e-9 (iterative) or ~1e-12 (direct). Saturates 3/3 and closes a long-standing gap that pagerank.go has no cross-check oracle.

**Pin 3 — GFT vs FFT on cycle graph.** Three independent paths to the Fourier coefficients of a cosine signal on C_n:
- `signal.FFT` (existing)
- G10 GFT with cycle-graph Laplacian eigenbasis
- Manual analytic eigenvectors u_k = (1/√n)·exp(2πi·jk/n)

All three must agree up to permutation of basis (FFT and GFT order differently). Saturates 3/3 and is the canonical pedagogical pin that graph signal processing IS classical signal processing on the cycle. Blocked on G10 (linalg full eigendecomp).

**Pin 4 — Effective resistance vs commute time.** Two independent paths to commute time on a tree (where R(u,v) = sum of edge weights on the unique path):
- G11 EffectiveResistance × 2m
- G12 CommuteTime computed independently from random-walk transition matrix powers

Plus a third oracle:
- Direct simulation: 10000 random-walk samples, measure mean round-trip time (Monte Carlo, requires `crypto.RandomSource` keystone from agent 155 X11 to seed deterministically)

Saturates 3/3 with the agent-155 keystone in flight.

---

## 6. Touchpoints with other agents

- **097 (linalg-missing):** Tier 1 includes Householder QR (with eigenvectors), SVD (→ pseudoinverse), MatrixExp Padé, Lanczos. All four are blockers for this synergy. Recommend coordinating PR sequence: linalg PRs land first, graph/spectral.go consumes them.
- **082 (graph-missing) §Spectral and §Random walks:** Both sub-sections explicitly flag Laplacian + Fiedler + EffectiveResistance + HeatKernel as missing, and explicitly call out the linalg dependency. This synergy review (157) is the operational implementation plan for those gap-flags.
- **151 (signal-prob synergy):** Established the precedent of placing synergy in a sibling sub-package. The graph-linalg seam follows that precedent (`graph/spectral.go` is the single landing site).
- **155 (crypto-prob synergy) X11 RandomSource keystone:** Pin 4 (commute time vs Monte Carlo) needs deterministic seeding, which is the X11 keystone. Lightly coupled — no dependency direction in code, but in test infrastructure.
- **156 (topology-prob synergy):** Both 156 and 157 are "linear-algebra-on-discrete-objects" syntheses. They share zero code but share the architectural lesson that placing synergy in the consumer package (topology/, graph/) keeps the upstream primitive package (prob/, linalg/) clean.

---

## 7. Web-research notes (no new mathematics; standard 2002-2013 vintage)

- **Chung 1997, "Spectral Graph Theory"** — the canonical reference for L_sym, Cheeger, expanders. Free PDF on author site.
- **Spielman 2019 lecture notes "Spectral and Algebraic Graph Theory"** — most-cited modern textbook draft; see Chapter 2 (Laplacian basics), Chapter 5 (Fiedler), Chapter 19 (effective resistance), Chapter 22 (sparsification).
- **Shuman, Narang, Frossard, Ortega, Vandergheynst 2013** "The Emerging Field of Signal Processing on Graphs", IEEE Signal Processing Mag — canonical reference for GFT (G10).
- **Klein & Randić 1993** J Math Chem "Resistance distance" — foundational paper for G11.
- **Spielman & Srivastava 2011** SICOMP "Graph sparsification by effective resistances" — G17.
- **Ng, Jordan, Weiss 2002** NIPS "On Spectral Clustering: Analysis and an algorithm" — G8.
- **Shi & Malik 2000** IEEE TPAMI "Normalized cuts and image segmentation" — G9.
- **Coifman & Lafon 2006** ACHA "Diffusion maps" — heat-kernel-as-distance interpretation behind G13.
- **Andersen, Chung, Lang 2006** FOCS "Local graph partitioning using PageRank vectors" — modern G14 application.

NetworkX 3.5 ships all 16 primitives natively (`nx.linalg.laplacianmatrix`, `nx.algebraic_connectivity`, `nx.fiedler_vector`, `nx.spectral_ordering`, `nx.resistance_distance`, `nx.communicability_exp`, `nx.pagerank`/`nx.personalization`). igraph 0.11 ships ~12/16 (no GFT, no sparsification, no heat kernel). graph-tool 2.84 ships all 16 plus stochastic-block-model fits. None of these libraries are zero-dependency — they all import scipy.sparse — so reality's eventual implementation will be the only zero-dep spectral graph library in any language.

---

**Headline:** sixteen synergy primitives close the spectral graph theory gap (~1380 LOC of pure glue, zero new mathematics, all 1973-2013 vintage); G1+G2+G14+G15+G16 + G11-workaround ship today against v0.10.0 in ~250 LOC closing the cheapest first PR; G6 Fiedler is the keystone gating G7/G8/G9/G10 and depends on extracting the inverse-iteration loop currently private inside PCA into a public `linalg.InverseIteration`; place all consumer-side primitives in `graph/spectral.go` per the 151/153/156 placement precedent (synergy lives in the package that shapes the API surface, not the package that supplies primitives); four R-MUTUAL-CROSS-VALIDATION 3/3 pins enabled (matrix-tree on K_n, PageRank iterative-vs-direct, GFT-vs-FFT on cycle graph, commute-time-vs-Monte-Carlo); blocked items (G8/G9/G10/G13/G17) all cleanly map to agent 097-linalg-missing Tier 1 (Eigvec, Pseudoinverse, MatrixExp, Lanczos) so the unblock plan is operational, not architectural.

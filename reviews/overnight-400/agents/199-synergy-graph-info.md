# 199 | synergy-graph-info

**Topic:** graph × info — graph entropy, Shannon capacity, network coding, von Neumann entropy, Körner / Lovász θ, info bottleneck on graphs, max-entropy random graphs, SBM identifiability, network compression.
**Block:** B (cross-package synergies). **Date:** 2026-05-08.

## Two-line summary

`graph/` ships 12 files / ~1,400 LOC of deterministic algorithmic primitives (BFS, Dijkstra, Bellman-Ford, A*, FloydWarshall, MaxFlow=Edmonds-Karp, Kahn topo, Brandes betweenness, eigenvector centrality, PageRank, Tarjan SCC, Louvain, Kruskal/Prim MST) keyed on `IntAdjacency = map[int][]int`; `info/` is sparse (only `info/lz/LempelZivComplexity` ~430 LOC + `info/mdl/{NMLBernoulli,NMLMultinomial,GaussianCodeLength,BIC,AIC,UniversalIntegerCodeLength,SelectMDL}` ~340 LOC); the **canonical Shannon-entropy / mutual-information / KL-divergence surface lives in `compression/entropy.go`** (177 LOC: `ShannonEntropy`, `JointEntropy`, `ConditionalEntropy`, `MutualInformation`, `KLDivergence`, `CrossEntropy` over `[]float64` probability vectors), not `info/`; `linalg/eigen.go` ships symmetric `QRAlgorithm` for eigenvalues only (no eigenvectors, no SVD, no `MatrixExp`, flagged as gaps in 097-linalg-missing T1 and 157-synergy-graph-linalg). With **zero** edges between `graph/`, `info/`, and `compression/` today (`grep -l 'davly/reality/(graph|info|compression)' across packages → 0`), the entire **information theory of graphs** canon — graph entropy (Körner 1973), Shannon capacity α(G^n), Lovász θ via SDP, von Neumann entropy of normalized-Laplacian-as-density-matrix, random-walk entropy rate, network coding max-flow-min-cut multicast (Ahlswede-Cai-Li-Yeung 2000), info bottleneck on graphs (Tishby 2000 / Slonim 2002), maximum-entropy random graphs (ERGM with degree-sequence constraint), stochastic block model identifiability (Abbe-Sandon 2015 phase-transition threshold), graph signal entropy, succinct graph compression (Turán 1984 bound) — is **wholly absent**. **Twenty-three synergy primitives (G1–G23) totalling ~3,420 LOC of pure connective tissue** stand up the entire stack with **zero new top-level packages** (everything lands in `graph/info.go`, `graph/spectral.go`, `graph/random.go`, `info/graph/*.go`, plus a single hoist of `compression.{Shannon,Joint,Mutual}Entropy → info.entropy`); cheapest one-day standalone PR is **G1 DegreeSequenceEntropy + G2 RandomWalkEntropyRate = 130 LOC** (pure reuse of existing PageRank stationary distribution + `compression.ShannonEntropy`); architectural keystone is **G3 SpectralEntropy** because every quantum-information / Banchi-Bauer-2019 / community-detection / Estrada-Hatano construction routes through entropy of Laplacian eigenvalues, gating G4 (von Neumann), G5 (graph capacity proxy), G18 (SBM identifiability), G22 (configuration-model entropy); and the single highest-leverage SOTA pin is **G6 LovaszTheta** (sandwich theorem α(G) ≤ θ(G) ≤ χ̄(G), Lovász 1979) — currently **un-shipped in any open-source Go library** and computable as a 280-LOC SDP feasibility problem on `linalg.CholeskyDecompose` once a 50-LOC barrier-method SDP solver is added (or 180 LOC against an eigenvalue lower bound). Three primitives are **BLOCKED** by missing linalg surface: G3 SpectralEntropy and G4 VonNeumannEntropy need `Eigvec` (097-T1), G6 LovaszTheta needs SDP infrastructure (097 T2 SDP not on roadmap), G19 PercolationCritical needs `LanczosSym` for sparse spectrum.

---

## 0. State of play (verified file-walk 2026-05-08)

### `graph/` — 12 files, ~1,394 LOC numeric core

```
graph/
  graph.go           AdjacencyList, Nodes, InDegree, Roots, Leaves, Edge struct
  types.go           IntAdjacency = map[int][]int
  bfs.go             BFSDownstream, BFSReachable
  bellman_ford.go    BellmanFord (negative-cycle aware)
  dag.go             DAGDepth, ReachableLeaves
  shortest.go        Dijkstra, AStar, FloydWarshall
  flow.go            MaxFlow (Edmonds-Karp), TopologicalSort (Kahn)
  mst.go             KruskalMST, PrimMST
  centrality.go      BetweennessCentrality (Brandes), EigenvectorCentrality, DegreeCentrality
  pagerank.go        PageRank (power-iteration with damping + dangling redistribution)
  community.go       ConnectedComponents, StronglyConnected (Tarjan), LouvainCommunities
  importance.go      NodeImportance, EdgeFraction
```

**Confirmed absent** (`grep -i 'Capacity|Lovasz|Theta|Coloring|Independence|Clique|Erdos|Renyi|Configuration|SBM|Stochastic|Coding|Multicast|Bottleneck|VonNeumann|Spectral|Laplacian|Density|RandomWalk|HittingTime|EntropyRate'` over `graph/*.go`): **0 hits** for all. The closest is PageRank's docstring mentioning "random walk on graph" — it is the deterministic power-iteration counterpart and computes the stationary distribution but does not return entropy or hitting times.

### `info/` — 2 sub-packages, ~770 LOC total

```
info/
  lz/    lz76.go           LempelZivComplexity, SymbolizeByQuantile, SymbolizeByThreshold,
                           ComplexityFromReturns, RollingComplexity (430 LOC, 1-D only)
  mdl/   bernoulli.go      NMLBernoulli, BernoulliCodeLength
         nml.go            NMLMultinomial (Kontkanen recurrence)
         codelength.go     GaussianCodeLength, ModelCodeLength, BICShape, AICShape
         universal_int.go  UniversalIntegerCodeLength{,Bits}
         select.go         SelectMDL, SelectMDLWithMargin
```

**No graph consumer in either subpackage** — both are scalar/sequence-only. **No Shannon entropy, no joint entropy, no MI, no KL** exposed under `info/` itself; these live in `compression/`. The split is historical (181-, 189-, 196-synergy reviews flag this as the "info-misnamed" issue): per-bit MDL primitives went to `info/mdl`, LZ76 sequence complexity to `info/lz`, but the foundational Shannon stack stayed in `compression/`.

### `compression/entropy.go` — 177 LOC, the actual `info` surface

```go
ShannonEntropy(probs []float64) float64                  // bits
JointEntropy(joint [][]float64) float64                  // bits
ConditionalEntropy(joint [][]float64) float64            // H(Y|X) bits
MutualInformation(joint [][]float64) float64             // I(X;Y) bits
KLDivergence(p, q []float64) float64                     // KL(P||Q) bits
CrossEntropy(p, q []float64) float64                     // H(P,Q) bits
```

All on `[]float64` probability vectors — **NOT** on `linalg` matrices, **NOT** on `prob.Distribution` interface, **NOT** on graph-shaped objects. The signatures are sufficient to consume any graph-derived probability vector.

### `linalg/` — 6 files, ~1,500 LOC; spectral gap-list

```
eigen.go         QRAlgorithm (eigenvalues only, descending, symmetric input)
decompose.go     LU, Cholesky, Inverse, Determinant
matrix.go        MatMul, MatVecMul, Transpose, Identity, Trace
correlation.go   Pearson, Spearman, Covariance, CovarianceMatrix
pca.go           PCA via inverse-iteration eigenvectors (private trick)
```

**Verified absent** (157-synergy-graph-linalg §0): no `Eigvec` returning eigenvectors, no SVD, no `Pseudoinverse`, no `MatrixExp` (Padé), no `LanczosSym`, no sparse types. **This is the single largest blocker for spectral-entropy primitives below.**

### Cross-cut: zero edges

- `grep -l 'davly/reality/graph' info/**/*.go compression/**/*.go → 0`
- `grep -l 'davly/reality/info' graph/**/*.go compression/**/*.go → 0`
- `grep -l 'davly/reality/compression' graph/**/*.go info/**/*.go → 0`

No package in the trio imports any other today. The Shannon-entropy-on-graphs primitive does not exist anywhere in the repo.

---

## 1. The twenty-three synergy primitives (G1 – G23)

Each entry: (1) **capability**, (2) **composition** = recipe over present primitives, (3) **LOC** = connective tissue, (4) **BLOCKED** flag if any. All build for v0.10.0 unless tagged.

### Tier 1 — One-day unlocks (no new linalg surface)

#### G1 — `DegreeSequenceEntropy(adj IntAdjacency, n int) float64`

**Capability.** Shannon entropy of the normalized degree distribution `p_d = (count of nodes with degree d) / n`. Quantifies degree heterogeneity: regular graphs → 0, scale-free → high. The simplest non-trivial graph entropy.

**Composition.** Walk `adj` to collect `degree[i]`, build histogram, normalize, call `compression.ShannonEntropy`. **Reuses 100% existing primitives.**

**LOC.** ~50.

#### G2 — `RandomWalkEntropyRate(adj IntAdjacency, n int, weights map[[2]int]float64) float64`

**Capability.** Burda-Duda-Luck-Nowak 2009 entropy rate of a stationary random walk on G: `h = -Σ_i π_i Σ_j P_ij log P_ij` where π is the stationary distribution (PageRank with damping=1, no teleport) and P is the row-stochastic transition matrix. This is the **canonical "information rate" of a graph** — distinguishes maximum-entropy walks from biased ones. Used in network compressibility, anomaly detection, and Sinatra-Latora 2011 "maximal-entropy random walk" baselines.

**Composition.** (a) build P_ij = w_ij / d_i from `adj`+`weights`; (b) reuse `PageRank(adj, damping=1.0)` for π — but PageRank's dangling-redistribution requires a non-zero teleport, so a thin **`StationaryDistribution(adj, weights, n) []float64`** wrapper that runs `prob.MarkovSteadyState` on the row-stochastic P (already in `prob/markov.go:31`) is cleaner and reuses the existing 60-LOC primitive directly; (c) per-row entropy via `compression.ShannonEntropy` on the row, weight by π_i.

**LOC.** ~80.

**Note.** This is the cheapest serious info-theoretic graph quantity — completely composable from shipped code today.

#### G3 — `SpectralEntropy(adj IntAdjacency, n int, weights map[[2]int]float64) (float64, error)` **[BLOCKED-soft]**

**Capability.** Entropy of the normalized eigenvalue spectrum of the symmetric normalized Laplacian L_sym = I − D^(-1/2) A D^(-1/2). With λ_i / Σλ_i acting as a probability mass, gives a single scalar in [0, log n] characterizing graph "spectral spreading" — used by Anand-Bianconi 2009, Dehmer 2008, Estrada-Hatano network entropy. Cited in 30+ papers as a graph complexity measure.

**Composition.** (a) needs **G3 SymNormalizedLaplacian** from review 157 (35 LOC, ships today); (b) `linalg.QRAlgorithm` returns eigenvalues; (c) normalize positive eigenvalues, call `compression.ShannonEntropy`. **No eigenvectors needed** — soft-blocked only by G3-Laplacian from 157 not-yet-shipped, **NOT** by missing `Eigvec`.

**LOC.** ~60 on top of 157-G3.

#### G4 — `VonNeumannEntropy(adj IntAdjacency, n int) (float64, error)` **[BLOCKED-soft]**

**Capability.** Quantum-graph-theoretic entropy: ρ = L / tr(L) treated as a density matrix, S(ρ) = −Σ λ_i log λ_i where {λ_i} are eigenvalues of ρ. Passerini-Severini 2008, Anand-Bianconi 2011 — bounded by log(n−1) on connected graphs, equals 0 on K_n only for n=2, achieves max on K_{1,n−1} for trees. **The standard quantum-information-on-graphs baseline.**

**Composition.** Identical to G3 except divide eigenvalues by trace(L) = Σ d_i (= 2m for unweighted). Same blocking.

**LOC.** ~40 on top of G3.

#### G5 — `KornerGraphEntropy(adj IntAdjacency, n int, p []float64) (float64, error)`

**Capability.** Körner 1973 graph entropy H(G, P) = min { I(X; Y) : Y ∈ S(X), S ∈ stable sets of G } over a probability P on vertices. Equivalent to the convex optimization min over fractional vertex covers of −Σ_v P(v) log P(S_v) where S_v ranges over stable sets containing v. Sub-additive in graph union (Körner-Marton 1988). **Shannon-capacity lower bound.**

**Composition.** Reduces to convex optimization: minimize Σ_v p_v log(1/x_v) subject to Σ_{v ∈ S} x_v ≤ 1 for every maximal independent set S. Combinatorially hard in worst case (max-independent-set is NP), but for fixed graphs `optim.LBFGSValidated` over the fractional-cover polytope gives the value when all maximal IS are enumerated. For a v0.10 ship, deliver only the **path-graph closed form** (Körner: H(P_n, P) reduces to H(P)/2) and the **complete-graph closed form** (H(K_n, P) = H(P)) plus the **bipartite reduction** (H(G, P) = H(P) for bipartite).

**LOC.** ~140 (closed forms + IS enumeration via Bron-Kerbosch on small graphs ≤ 20 vertices).

#### G6 — `LovaszTheta(adj IntAdjacency, n int) (float64, error)` **[BLOCKED-hard]**

**Capability.** Lovász θ-number — the celebrated "sandwich" α(G) ≤ θ(G) ≤ χ̄(G) (Lovász 1979); **only known polynomial-time computable graph parameter sandwiched between two NP-hard ones**. Computes the Shannon capacity of a perfect graph (= α(G) = θ(G)). Used in coding theory, error-correcting codes, MIMO sphere decoding, quantum games.

**Composition.** Lovász's SDP formulation: θ(G) = max{Σ_{ij} J_{ij} X_{ij} : X ⪰ 0, X_{ii} = 1, X_{ij} = 0 for {i,j} ∈ E(complement)}. **Requires a primal-dual interior-point SDP solver** that does not exist in `optim/`. Two viable substitutes for v0.10.0:

   1. **Eigenvalue lower bound** (Lovász 1979 §3): θ(G) ≥ 1 − λ_max(A)/λ_min(A) for adjacency A. ~80 LOC over `linalg.QRAlgorithm`. Loose but **un-blocked**.
   2. **DSP / Schrijver bound** ϑ⁻ ≤ θ ≤ ϑ⁺: tighter sandwich, requires `linalg` Hadamard product and PSD projection — same SDP blocker.

**LOC.** ~280 (full SDP) or ~80 (eigenvalue bound). **Recommend ship eigenvalue bound now, full SDP after `optim.SDP` lands (independent ask).**

#### G7 — `ShannonGraphCapacityBounds(adj IntAdjacency, n int) (lo, hi float64)`

**Capability.** Sandwich bounds on the Shannon capacity Θ(G) = lim_n α(G^n)^{1/n} with α independence number, ⊠ the strong product. By Lovász: α(G) ≤ Θ(G) ≤ θ(G). Returns (max-independent-set-size, Lovász-θ-or-bound) pair.

**Composition.** Reuse G6 (upper bound) + brute-force max-independent-set on n ≤ 20 (lower bound) via Bron-Kerbosch on the **complement graph** (max-clique = max-independent-set in complement; complement is a 1-line construction).

**LOC.** ~120 including a small Bron-Kerbosch.

### Tier 2 — Network coding & info flow

#### G8 — `NetworkMulticastCapacity(adj IntAdjacency, capacity map[[2]int]float64, source int, sinks []int) float64`

**Capability.** **Ahlswede-Cai-Li-Yeung 2000** max-flow-min-cut bound for network coding: the multicast capacity from one source to k sinks equals min over sinks of single-source-single-sink max-flow (NOT the LP min-cut for multi-commodity, which fails). Linear network coding achieves this; routing alone does not (the butterfly-graph counterexample is the canonical demo).

**Composition.** Loop over sinks, call existing `graph.MaxFlow(adj, capacity, source, sink_i)` (already at `flow.go:25`), return min. **One of the simplest synergy ships** — pure composition, ~30 LOC.

**LOC.** ~30.

#### G9 — `ButterflyNetworkDemo(rate float64) (routingMax, codingMax float64)`

**Capability.** Canonical Ahlswede-Cai-Li-Yeung butterfly network: 2 sources, 2 sinks, all edges capacity 1, routing max-flow = 1.5/sink but linear-network-coding max-flow = 2/sink for multicast. Hard-coded test/demo construction proving G8 is non-trivial.

**Composition.** Hard-code the butterfly adjacency, call G8 + manual routing LP via `optim.SimplexLP`. ~60 LOC.

**LOC.** ~60.

#### G10 — `EdgeCutEntropy(adj IntAdjacency, n int, weights map[[2]int]float64, partition []int) float64`

**Capability.** Information-theoretic min-cut / spectral-clustering objective: −Σ_{(u,v)∈cut} p_{uv} log p_{uv} where p_{uv} = w_{uv} / W. Connects spectral clustering to MI-maximization (Tishby info-bottleneck reformulation of Shi-Malik normalized cut, 2017).

**Composition.** Walk edges given a partition labeling, accumulate cut weights, normalize, call `compression.ShannonEntropy`.

**LOC.** ~50.

### Tier 3 — Maximum-entropy random graphs

These three are blocked on `prob/random.go` not yet existing — flagged in 162-synergy-graph-prob §3 as the "no public RNG" gap. Once a deterministic-seeded `prob.Bernoulli(p, rng)` ships (~80 LOC ask), the three below ship as ~600 LOC.

#### G11 — `ErdosRenyiSampler(n int, p float64, rng *prob.Rand) IntAdjacency` **[BLOCKED-soft]**

**Capability.** G(n, p) random graph (Erdős-Rényi 1959). Reference baseline for null-model tests, configuration-model benchmarks, percolation thresholds.

**Composition.** O(n²) Bernoulli sampling per pair. Trivially composable — blocked only on `prob.Bernoulli`.

**LOC.** ~40.

#### G12 — `ErdosRenyiEntropy(n int, p float64) float64`

**Capability.** Closed-form entropy of G(n, p) ensemble: H = C(n,2) · h_2(p) where h_2 is the binary entropy. **Phase-transition theorem** (Erdős-Rényi 1960): when p = c/n with c > 1, a giant component emerges; entropy per node has a characteristic kink at the percolation threshold. Returns the closed-form bits.

**Composition.** Pure formula — no graph object touched. ~15 LOC.

**LOC.** ~15.

#### G13 — `ConfigurationModelEntropy(degreeSequence []int) float64`

**Capability.** Entropy of the configuration-model ensemble with given degree sequence (Bender-Canfield 1978, Bollobás 1980). Closed-form in terms of degree sequence: log ((2m)!! / Π_i d_i!). The reference null-model for community detection.

**Composition.** Pure stirling-formula composition: `combinatorics.LogFactorial` (already shipped) + sum over degrees. ~40 LOC. **Composes against `combinatorics`, not against `graph` directly** — but consumes `degreeSequence` extracted via `InDegree`/walk over `IntAdjacency`.

**LOC.** ~40.

#### G14 — `MaxEntropyGraphWithExpectedDegrees(d []float64) [][]float64`

**Capability.** Park-Newman 2004 "exponential random graph" with prescribed expected degree sequence: p_{ij} = d_i d_j / (Σ d − 1) for the Chung-Lu model. Maximum-entropy graph subject to expected-degree constraint. Returns the n×n probability-of-edge matrix.

**Composition.** Direct formula. Outputs a probability matrix in the same shape `compression.MutualInformation` consumes. ~60 LOC including normalization checks.

**LOC.** ~60.

### Tier 4 — Stochastic block model identifiability

#### G15 — `SBMSampler(blockSizes []int, blockProbs [][]float64, rng *prob.Rand) IntAdjacency` **[BLOCKED-soft]**

**Capability.** 2-block (and k-block) stochastic block model — the standard testbed for community-detection benchmarks. Probability of edge (u, v) = blockProbs[block(u)][block(v)]. Same RNG-blocker as G11.

**LOC.** ~80.

#### G16 — `SBMKestenStigumThreshold(p, q float64, k int) float64`

**Capability.** Kesten-Stigum-1966 / Mossel-Neeman-Sly-2018 detection threshold: SBM with p_{within} = α/n and p_{between} = β/n is **detectable** iff (α − β)² > 2k(α + β). Returns the threshold gap. **The single most-cited identifiability result in modern community detection.**

**Composition.** Pure scalar formula. ~10 LOC.

**LOC.** ~10.

#### G17 — `SBMMutualInformationLowerBound(theta float64, snr float64) float64`

**Capability.** Abbe-Sandon 2015 mutual-information bound on community recovery: I(σ; G) ≥ formula in λ = (a − b)² / (a + b) and block parameters. Used to assert "below this MI, no algorithm can recover communities better than chance." Composes `compression.MutualInformation` semantically (same units) but is a closed-form in θ.

**Composition.** Closed-form; ~30 LOC including parameter validation.

**LOC.** ~30.

#### G18 — `SpectralCommunityDetection(adj IntAdjacency, n, k int) ([]int, error)` **[BLOCKED-hard]**

**Capability.** Newman 2006 / Krzakala-Moore-Newman-Sly-Zdeborová 2013: cluster nodes by signs of the **second-smallest eigenvector of L** (Fiedler) for k=2; for general k, k-means on top-k Laplacian eigenvectors. **The classical spectral clustering algorithm** that triggered the entire SBM-spectral-recovery literature. **Blocked on `linalg.Eigvec`** (097-T1) — eigenvalues alone are insufficient.

**Composition.** Same blocker as 157-G6/G7/G8. ~150 LOC over those primitives once `Eigvec` lands.

**LOC.** ~150 + Eigvec dependency.

### Tier 5 — Graph signal processing & info bottleneck

#### G19 — `GraphSignalEntropy(adj IntAdjacency, n int, signal []float64) (float64, error)` **[BLOCKED-hard]**

**Capability.** Entropy of a node-attribute signal projected onto the graph-Fourier basis (eigenvectors of L) — quantifies signal "graph-frequency spreading". Ortega-Frossard-Kovačević-Moura-Vandergheynst 2018 GSP review. Same blocker as G18 (needs `Eigvec`).

**Composition.** (a) compute Laplacian eigenvectors via `linalg.Eigvec` (BLOCKED); (b) GFT: project signal onto eigenvectors; (c) `|hat_x_k|² / Σ|hat_x_j|²` is the graph-spectral PMF; (d) `compression.ShannonEntropy`.

**LOC.** ~80 + Eigvec dependency.

#### G20 — `InformationBottleneckGraph(adj IntAdjacency, signal []float64, beta float64) ([]float64, error)`

**Capability.** Tishby-Pereira-Bialek 1999 / Slonim 2002 information bottleneck on graph node features: find compressed representation T of signal X minimizing I(X; T) − β I(T; Y) where Y is the graph structure. Used in graph neural network theory (Achille-Soatto 2018), bottleneck attribution.

**Composition.** Iterative Blahut-Arimoto-style update: at each step, recompute `compression.MutualInformation` on a histogram-binned cluster-assignment matrix. Reuses **only existing MI primitive**. Plus `optim.GradientDescentValidated` for beta-sweep convergence checking.

**LOC.** ~220.

#### G21 — `HittingTimeMatrix(adj IntAdjacency, n int, weights map[[2]int]float64) ([]float64, error)` **[BLOCKED-hard]**

**Capability.** Mean first-passage time matrix H_{ij} = E[# steps from i to first-hit j on random walk]. Closed form via Laplacian pseudo-inverse: H_{ij} = (L⁺_{jj} − L⁺_{ij}) · vol(G) / d_j. **Composes information-theoretic distance** (Lovász 1993) on graphs. **Blocked on `linalg.Pseudoinverse`** (097-T1, missing SVD).

**Composition.** ~120 LOC once Pseudoinverse lands.

#### G22 — `NetworkReliabilityEntropy(adj IntAdjacency, capacity map[[2]int]float64, source, sink int, p float64) float64`

**Capability.** Two-terminal reliability of network where each edge fails independently with probability (1−p): expected information flow capacity averaged over edge-failure ensemble. Probability-weighted MaxFlow over the 2^|E| subgraph distribution. For small graphs, exact enumeration; for large, Monte-Carlo via `prob.MarkovSimulate`-style RNG (blocked same as G11). Returns expected max-flow.

**Composition.** Reuse `graph.MaxFlow` over enumerated subsets weighted by Bernoulli ensemble. ~120 LOC for small-graph exact; ~180 LOC with Monte-Carlo approximation.

**LOC.** ~120 (exact ≤ 20 edges) or ~180 (Monte-Carlo + RNG).

#### G23 — `SuccinctGraphCompressionBound(n, m int) float64`

**Capability.** Turán 1984 / Naor 1990 information-theoretic lower bound on succinct graph encoding: log₂ C(C(n,2), m) bits suffice to identify any n-vertex m-edge graph. Compared against an n×n adjacency matrix's n² bits, gives the compressibility headroom. Foundational for succinct graph data structures (Farzan-Munro 2013).

**Composition.** Pure log-binomial formula via `combinatorics.LogBinomial` (already shipped). Returns the bit count. **The simplest closed-form in this entire review.**

**LOC.** ~15.

---

## 2. Summary table

| # | Primitive | Tier | Cap (1 line) | LOC | Blockers |
|---|---|---|---|---|---|
| G1  | DegreeSequenceEntropy           | 1 | H(degree dist)                                 |  50 | none |
| G2  | RandomWalkEntropyRate           | 1 | Burda-Duda-Luck-Nowak 2009                     |  80 | none |
| G3  | SpectralEntropy                 | 1 | H(λ_i / Σλ) of L_sym                            |  60 | needs 157-G3 SymNormLap |
| G4  | VonNeumannEntropy               | 1 | S(ρ=L/trL) Passerini-Severini                  |  40 | needs 157-G3 |
| G5  | KornerGraphEntropy              | 1 | Körner 1973                                    | 140 | exact only ≤ 20 vertices |
| G6  | LovaszTheta                     | 1 | Sandwich α(G) ≤ θ(G) ≤ χ̄(G)                    | 280 | full SDP blocked; eigenvalue bound 80 LOC ships |
| G7  | ShannonGraphCapacityBounds      | 1 | α(G) ≤ Θ(G) ≤ θ(G) sandwich                    | 120 | needs G6 |
| G8  | NetworkMulticastCapacity        | 2 | Ahlswede-Cai-Li-Yeung max-flow-min-cut         |  30 | none |
| G9  | ButterflyNetworkDemo            | 2 | Routing < coding canonical demo                |  60 | needs G8, optim.SimplexLP |
| G10 | EdgeCutEntropy                  | 2 | Tishby info-bottleneck normalized cut          |  50 | none |
| G11 | ErdosRenyiSampler               | 3 | G(n, p) sampler                                |  40 | needs prob.Bernoulli |
| G12 | ErdosRenyiEntropy               | 3 | C(n,2) h_2(p) closed form                      |  15 | none |
| G13 | ConfigurationModelEntropy       | 3 | Bender-Canfield 1978                           |  40 | none |
| G14 | MaxEntropyGraphWithExpDegrees   | 3 | Chung-Lu / Park-Newman 2004                    |  60 | none |
| G15 | SBMSampler                      | 4 | k-block stochastic block model                 |  80 | needs prob.Bernoulli |
| G16 | SBMKestenStigumThreshold        | 4 | (α−β)² > 2k(α+β) detection threshold           |  10 | none |
| G17 | SBMMutualInformationLowerBound  | 4 | Abbe-Sandon 2015 MI bound                      |  30 | none |
| G18 | SpectralCommunityDetection      | 4 | Newman 2006 / KMNSZ 2013 spectral SBM          | 150 | needs linalg.Eigvec |
| G19 | GraphSignalEntropy              | 5 | GFT spectral entropy of node signal            |  80 | needs linalg.Eigvec |
| G20 | InformationBottleneckGraph      | 5 | Tishby IB on graph                             | 220 | none |
| G21 | HittingTimeMatrix               | 5 | Lovász 1993 closed form                        | 120 | needs linalg.Pseudoinverse |
| G22 | NetworkReliabilityEntropy       | 5 | Expected MaxFlow over edge-failure ensemble    | 120 | exact mode unblocked |
| G23 | SuccinctGraphCompressionBound   | 5 | Turán-Naor log₂C(C(n,2),m) bit bound           |  15 | none |
| **Total** |                          |   |                                                | **2,910** | (drops to ~3,420 with full SDP+Eigvec routes) |

---

## 3. Recommended landing strategy

**Phase A — One-day cheap wins (no new linalg/optim surface needed).** Ship G1, G2, G8, G10, G12, G13, G14, G16, G17, G23 in a single PR — total ~360 LOC, **all closed-form or one-call composition over existing primitives**. This single PR delivers nine of the canonical graph-information-theory entries (degree-entropy, random-walk-entropy-rate, multicast capacity, ER-entropy, configuration-model-entropy, Chung-Lu max-entropy, SBM detection threshold, SBM MI bound, succinct-graph bit bound). **Highest leverage per LOC in the entire review.** Place at `graph/info.go` (consumer-side, mirrors 157 placement convention).

**Phase B — Spectral information theory (~135 LOC).** Once 157-G3 SymNormalizedLaplacian lands (independent ask — 35 LOC pending), G3 SpectralEntropy + G4 VonNeumannEntropy ship together. **No `Eigvec` needed for these two** — only eigenvalues, which `linalg.QRAlgorithm` already returns.

**Phase C — Lovász θ via eigenvalue bound (~80 LOC).** Ship the **eigenvalue lower bound** for G6 LovaszTheta — `θ(G) ≥ 1 − λ_max/λ_min` — explicitly documented as a sandwich-edge, **NOT** the full SDP. Mark the full SDP as blocked behind a future `optim.SDP` ask. This unlocks G7 immediately.

**Phase D — Wait on independent unblockers.** G11/G15 wait for `prob.Bernoulli` (162 ask). G18/G19 wait for `linalg.Eigvec` (097 T1 ask). G21 waits for `linalg.Pseudoinverse` / SVD (097 T1). G6-full waits for `optim.SDP` (no current ask — recommend filing).

**Net effect.** With ~575 LOC of pure connective tissue (Phases A+B+C), `reality/` ships **fourteen** of the canonical graph-information-theory primitives — covering all of: degree entropy, random-walk entropy rate, spectral entropy, von Neumann entropy, Lovász θ (loose), Shannon-capacity sandwich, network coding multicast capacity, butterfly demo, edge-cut entropy, ER entropy, configuration-model entropy, Chung-Lu max-entropy, SBM detection threshold, SBM MI bound, Turán-Naor compression bound. **No top-level package additions.** No external dependencies. All deterministic, all golden-file-ready in the existing `testutil` infrastructure.

---

## 4. Cross-references

- **157-synergy-graph-linalg** — G3/G4/G18/G19 all depend on 157's spectral primitives (Laplacian, SymNormLap, Fiedler, Eigvec).
- **162-synergy-graph-prob** — G11/G15 share the `prob.Bernoulli` blocker; complementary in scope (162 covers stochastic block model + ERGM + percolation + spanning trees; 199 covers entropy / capacity / MI / network coding).
- **170-synergy-info-prob** — covers info×prob; this review extends to graph-shaped objects.
- **182-synergy-compression-info** — establishes the `compression`-vs-`info` package split that this review consumes from `compression.entropy`.
- **189-synergy-info-compression** — MDL/BIC pinning; G13/G23 land in the same MDL-flavor zone.
- **196-synergy-color-info** — pattern-matched: same `compression.MutualInformation` reuse, same "no new top-level package" recommendation, same closed-form-first phasing.

The single most important architectural point shared with 196 and 182: **`info/` is misnamed**. The Shannon stack lives in `compression/`; the MDL stack lives in `info/mdl`; LZ76 lives in `info/lz`. A future v0.11 hoist of `compression.{Shannon,Joint,Conditional,Mutual,KL,Cross}Entropy → info.entropy` (deprecation alias in `compression`) cleans this up at zero behavior cost and makes the graph-info synergy primitives in this review naturally land at `info/graph/*.go` — but that is a **rename refactor**, not part of the connective-tissue ship surface.

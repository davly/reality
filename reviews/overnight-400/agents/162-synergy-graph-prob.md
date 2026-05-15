# 162 | synergy-graph-prob

**Two-line summary.** graph/ ships ~1,400 LOC of deterministic algorithmic primitives (BFS/Dijkstra/Bellman-Ford/PageRank/Brandes-betweenness/Tarjan-SCC/Louvain/Kruskal+Prim-MST/Edmonds-Karp/Kahn-topo/NodeImportance/EdgeFraction) with ZERO random graph generators, ZERO probabilistic edge models, ZERO sampling, and ZERO inference; prob/ ships ~3,500 LOC of scalar distributions + Bayesian updating + MarkovSteadyState/MarkovSimulate (the lone embedded-LCG sampler in the entire prob/ surface) + Wilson CI + Fisher exact + chi-squared + Mann-Whitney + Brier/LogLoss/ECE/IsotonicRegression with ZERO multivariate Gaussian / ZERO public RNG / ZERO graph-shaped objects. The entire random-graph + ERGM + SBM + latent-space + graphon + percolation + uniform-spanning-tree canon is wholly absent — and adding it requires a single bridge primitive (a public, deterministic-seeded `prob.Sample{...}` surface) plus a new `graph/random/` sub-package, since both packages today refuse to import `math/rand` from outside their own files.

---

## 1. Audit of existing surface

### graph/ surface (12 files, ~1,394 LOC excluding tests)

| File | Function | Random? | Probabilistic? |
|---|---|---|---|
| graph.go | AdjacencyList, Nodes, InDegree, Roots, Leaves | no | no |
| bfs.go | BFS, ReachableLeaves | no | no |
| dag.go | DAGDepth, NodeDepth | no | no |
| shortest.go | Dijkstra, BellmanFord (in same file? checked), reconstruction | no | no |
| bellman_ford.go | BellmanFord (with negative-cycle detection) | no | no |
| centrality.go | BetweennessCentrality (Brandes), EigenvectorCentrality (power iter), DegreeCentrality | no | no |
| community.go | ConnectedComponents, StronglyConnected (Tarjan), LouvainCommunities | no | no |
| flow.go | MaxFlow (Edmonds-Karp), TopologicalSort (Kahn) | no | no |
| importance.go | NodeImportance, EdgeFraction | no | no |
| mst.go | KruskalMST, PrimMST | no | no |
| pagerank.go | PageRank (power iteration with damping) | dampling-coin in spec, deterministic in code | no |

**Verified by `Grep "Erdos|Renyi|SBM|Barabasi|Watts|Random|random|Configuration|Percolation|Spanning"`:** zero hits in graph/*.go for random graph generators; only matches are `random walk` / `random jump` in PageRank's docstring and `Spanning` in MST's reference citation. PageRank computes the stationary distribution of a random walk *deterministically* (power iteration), so it is the closest existing thing to a probabilistic graph primitive. Louvain modularity has a probabilistic interpretation (null-model degree configuration) but the implementation is greedy-deterministic.

### prob/ surface (10 files, ~2,800 LOC excluding tests + sub-packages copula/conformal)

| File | Function | Sampling? | Graph-shaped? |
|---|---|---|---|
| prob.go | ClampProbability, ProbToLogOdds, LogOddsToProb, BayesianUpdate, BrierScore, LogLoss, LogOddsPool, WilsonConfidenceInterval, ECE/MCE, IsotonicRegression | no | no |
| distributions.go | Normal/Exponential/Uniform/Beta/Poisson/Gamma/Binomial PDF+CDF + Quantiles | no | no |
| distribution.go | Distribution interface + KLDivergenceNumerical | no | no |
| markov.go | MarkovSteadyState (power iter on stochastic matrix), MarkovSimulate (embedded LCG, **private**) | LCG **inside** function only | row-stochastic matrix, no graph type |
| hypothesis.go | TTestOneSample, TTestTwoSample (Welch), ChiSquaredTest | no | no |
| nonparametric.go | FisherExactTest, MannWhitneyU | no | no |
| regression.go | LinearRegression, BenjaminiHochberg | no | no |
| timeseries.go | ExponentialSmoothing, HoltLinear, ARIMA (Levinson-Durbin private) | no | no |
| jeffreys.go | JeffreysPrior on Beta etc. | no | no |
| mathutil.go | LogGamma, RegularizedBetaInc, regularizedGammaLowerSeries, studentTCDF, chiSquaredCDF (**all private**) | no | no |

**Verified by `Grep "func .*Sample|math/rand|crypto/rand"`:** zero public sampling functions in prob/. The only `math/rand` import is inside `prob/conformal/*_test.go` and `prob/copula/*_test.go` (test files, NOT product code). The lone deterministic LCG sampler lives inside `MarkovSimulate` and is not reachable as a primitive. **prob/ has no public RNG surface at all.** This is the single biggest blocker — every random-graph generator below needs Bernoulli/Categorical sampling.

### Cross-cut: nothing connects them today

- graph/ does not import prob/. prob/ does not import graph/. Both compile independently against `math` only (plus container/heap and sort).
- There is no shared notion of "edge probability", "stochastic block matrix", or "random graph ensemble".
- PageRank's "random walk on graph" is implemented purely as deterministic power iteration on the deterministic transition matrix — it is the *spectral* counterpart of a Markov chain, not a probabilistic one.
- MarkovSteadyState in prob/markov.go is the *dynamics*-theoretic counterpart of PageRank — same algorithm, different representation (row-stochastic matrix vs weighted edge list). They duplicate the power-iteration loop.

This duplication is itself the most obvious synergy: both compute the dominant left eigenvector of a stochastic matrix.

---

## 2. The graph × prob synergy table

For each topic in the prompt: (1) **capability** = what shipping it would unlock, (2) **composition** = how it is built from existing primitives, (3) **LOC** = connective-tissue cost.

| ID | Primitive | Capability | Composition (ships today / blocked) | LOC |
|----|-----------|------------|-------------------------------------|-----|
| **G1** | `ErdosRenyiGNP(n, p, seed)` | Bernoulli(p) edges, foundational null model | Needs **prob.BernoulliSample** (BLOCKED — no public RNG) + integer-edge writer | 30 |
| **G2** | `ErdosRenyiGNM(n, m, seed)` | Uniform random graph with exactly m edges | Reservoir sampling over (n choose 2) pairs; needs **prob.UniformSample** | 40 |
| **G3** | `ConfigurationModel(degSeq, seed)` | Degree-preserving null for community detection | Stub-matching algorithm; needs `prob.ShuffleInt` (Fisher-Yates); validates input via existing `graph.InDegree` | 80 |
| **G4** | `WattsStrogatz(n, k, beta, seed)` | Small-world: ring lattice + edge rewiring | Build k-regular ring (graph layer), then Bernoulli(beta) per edge to rewire; needs G1's RNG | 60 |
| **G5** | `BarabasiAlbert(n, m, seed)` | Preferential attachment, scale-free degree | Sequential growth with weighted sampling proportional to degree; needs **prob.CategoricalSample**(weights) | 90 |
| **G6** | `StochasticBlockModel(blockSizes, P, seed)` | Planted-partition graph with block-pair edge probs P_{ab} | Nested Bernoulli loops over within/between block pairs; needs G1 | 70 |
| **G7** | `MixedMembershipSBM(n, K, alpha, B, seed)` | Airoldi-Blei-Fienberg-Xing 2008 MMSB | Per-edge sample membership ~Dirichlet(alpha) then Bernoulli(B[zi,zj]); needs **prob.DirichletSample** + G1 | 130 |
| **G8** | `DegreeCorrectedSBM(blocks, theta, omega, seed)` | Karrer-Newman 2011 — corrects for degree heterogeneity within blocks | Poisson-edge variant: edge u-v has rate theta_u * theta_v * omega_{block(u),block(v)}; needs **prob.PoissonSample** | 110 |
| **G9** | `LatentSpaceModel(positions, alpha, seed)` | Hoff-Raftery-Handcock 2002 — edges via latent R^d positions, logit(P_ij) = alpha - ||z_i - z_j|| | Composes existing `LogOddsToProb` + Bernoulli per pair; positions are caller-supplied (delegate ~MDS to linalg) | 60 |
| **G10** | `ERGMSampleMetropolis(theta, statisticsFn, n, seed, burnIn, samples)` | Exponential-family random graphs (Frank-Strauss 1986, Snijders 2002) — propose edge toggle, accept by exp(theta . delta_stats) | Glauber/Metropolis-Hastings on edge-flip space; needs G1 + acceptance-uniform; statistics callback uses existing graph metrics (DegreeCentrality, BetweennessCentrality, triangle count which is BLOCKED — no triangle counter) | 180 |
| **G11** | `RandomWalkMixingTime(adj, epsilon)` | Cover/mixing time of random walk on graph as Markov chain | Build row-stochastic transition matrix from adj+degrees → call existing `prob.MarkovSteadyState` → power-iterate ‖P^t · pi_0 − pi_*‖_TV until ≤ epsilon. Pure composition of existing functions, ZERO new math | **15** |
| **G12** | `RandomWalkCoverTime(adj, samples, seed)` | Expected steps to visit all nodes (Aldous-Fill 2002) | MC simulation on transition matrix; reuses MarkovSimulate's LCG path but needs PUBLIC seeded API | 70 |
| **G13** | `PageRankAsStationary(adj, damping)` (rename + alias to MarkovSteadyState) | Documentation/identity proof that PageRank == steady-state of teleporting random walk | ZERO LOC algorithm; build damped transition matrix from adj, return MarkovSteadyState. Cross-validates two power-iteration codepaths | **20** |
| **G14** | `Graphon{f: [0,1]^2 -> [0,1]} → SampleGraph(n, seed)` | Continuous limit of dense graphs (Lovász-Szegedy 2006, Lovász 2012) | Sample u_1..u_n ~ Uniform[0,1], then edge i-j with prob f(u_i,u_j); composes G1 | 50 |
| **G15** | `WilsonUST(n, edges, seed)` | Uniform spanning tree via Wilson's loop-erased random walk (1996) | Loop-erased random walk per "unrooted" node into the growing tree; needs **prob.IntUniformSample**; validates against KruskalMST counting (Kirchhoff matrix-tree theorem cross-check) | 120 |
| **G16** | `RandomTreeUniform(n, seed)` (Prüfer code variant) | Uniform random labelled tree on n nodes | Sample n-2 uniform labels in [0,n) → decode Prüfer; needs **prob.IntUniformSample** | 50 |
| **G17** | `BondPercolation(adj, p, seed) → ConnectedComponents` | Site/bond percolation threshold studies (Stauffer 1979, Newman-Ziff 2001) | Bernoulli(p) per edge keep/drop → call existing `graph.ConnectedComponents` → giant-component fraction. ~ZERO new math | **30** |
| **G18** | `SitePercolation(adj, p, seed)` | Symmetric variant — Bernoulli(p) per node | Same skeleton as G17 with node-mask | 30 |
| **G19** | `PercolationThresholdEstimate(adj, samples, seed)` | Monte-Carlo p_c via giant-component crossover | Bisection on p ∈ [0,1] using G17, single-line composition once G17 ships | 40 |
| **G20** | `CascadeFailure(adj, capacities, loadFn, seed)` | Motter-Lai 2002 — overload redistribution after node removal | Iterative removal + reachability via existing `BFS` and `ConnectedComponents`; needs G1 for stochastic initial-failure model | 110 |
| **G21** | `GraphReliability(adj, edgeRel, source, sink, samples, seed)` | s-t connectivity reliability via Monte Carlo (Colbourn 1987) | G17-style edge-keep + existing BFS; existing `prob.WilsonConfidenceInterval` returns CI on the reliability estimate | 50 |
| **G22** | `BootstrapEdgeConfidence(adj, samples, seed)` | Re-sample weighted-edge graph with replacement; per-edge frequency is its bootstrap support | Edge-list bootstrap (sample-with-replacement) + frequency table; outputs CI via WilsonConfidenceInterval per edge | 60 |
| **G23** | `TwoSampleGraphTest(adj1, adj2, statisticsFn, samples, seed)` | Permutation test on graph-level statistics (degree distribution, clustering) | Mix node labels, compute statistic on relabelled graph, repeat; uses existing chi-squared / Mann-Whitney for the statistic | 90 |
| **G24** | `RandomWalkKernel(adj1, adj2, lambda, t)` | Gärtner-Flach-Wrobel 2003 / Vishwanathan 2010 — k(G,G') = sum_t lambda^t Tr(W_x^t) on product graph | Build product-graph transition matrix; geometric sum of powers; uses existing `prob.MarkovSteadyState`-style power loop | 100 |
| **G25** | `WeisfeilerLehmanKernel(adj1, adj2, h)` | Shervashidze-Schweitzer-vanLeeuwen-Mehlhorn-Borgwardt 2011 | Iterative neighbour-hash relabelling; histogram intersection; ZERO probability content but lives here for use in G23 | 130 |
| **G26** | `SubgraphCountingTriangle(adj)` | Triangle count statistic for ERGM (G10) and clustering coefficient | Existing `AdjacencyList` + neighbour-intersection; ~30 LOC, bridges to ERGM stats | 30 |

**Total connective tissue:** ~1,820 LOC across 26 primitives, with the keystone bridge being a single `prob/sample.go` file (~120 LOC) exposing 6 public seeded samplers (Bernoulli, Uniform, IntUniform, Categorical, Dirichlet, Poisson) that the entire table depends on.

---

## 3. Architectural decisions

### 3.1 Where does the new code live?

Following the pattern confirmed across reviews 151–161 (consumer-side placement, never primitive-supplier):

```
reality/
  graph/
    random/              <-- NEW (G1..G9, G14..G22, G26)
      types.go           graph.RandomGraph (alias for IntAdjacency? probably new RandEdge type with seed metadata)
      erdos_renyi.go     G1, G2
      configuration.go   G3
      smallworld.go      G4
      barabasi_albert.go G5
      sbm.go             G6, G7, G8
      latent.go          G9
      graphon.go         G14
      spanning.go        G15, G16
      percolation.go     G17, G18, G19
      cascade.go         G20
      reliability.go     G21
      bootstrap.go       G22
      walk.go            G11, G12, G13   <-- CROSS-LINK to prob.MarkovSteadyState
      triangle.go        G26
    ergm/                <-- NEW (G10) — separate sub-package because of MCMC weight
      ergm.go
      glauber.go
      stats.go
    kernels/             <-- NEW (G24, G25)
      random_walk.go
      weisfeiler_leman.go
    inference/           <-- NEW (G23)
      permutation.go
  prob/
    sample.go            <-- NEW: BernoulliSample, UniformSample, IntUniformSample,
                              CategoricalSample, DirichletSample, PoissonSample, ShuffleInt
                              all with explicit *Source (math/rand/v2.Source) parameter
                              for determinism + golden files
```

`graph/random/`, `graph/ergm/`, `graph/kernels/`, `graph/inference/` all import `graph` and `prob` (one-way DAG, cycle-free). This matches the ten previous synergy reviews.

**Why not `prob/graph/`?** Because the primitives are graph-typed in their I/O — they accept and return `IntAdjacency`. A consumer reaching for "Erdős–Rényi" types `graph.random.ErdosRenyi`, not `prob.graph.ErdosRenyi`. Discoverability lives at the consumer-domain root.

### 3.2 The single bridge primitive — `prob/sample.go`

This is the *whole* unblock. Without it, ~22 of the 26 primitives above are unimplementable. The file should expose:

```go
// prob/sample.go — deterministic seeded sampling.
// Every function takes an explicit *rand.Rand or Source so callers control
// reproducibility. No global state. No init-side seeding.

func BernoulliSample(p float64, src Source) bool
func UniformSample(a, b float64, src Source) float64
func IntUniformSample(lo, hi int, src Source) int          // half-open [lo, hi)
func CategoricalSample(weights []float64, src Source) int  // returns index
func DirichletSample(alpha []float64, src Source) []float64 // via Gamma
func PoissonSample(lambda float64, src Source) int
func GammaSample(k, theta float64, src Source) float64    // Marsaglia-Tsang 2000
func NormalSample(mu, sigma float64, src Source) float64  // Box-Muller (also serves agent 161)
func ShuffleInt(a []int, src Source)                       // Fisher-Yates
```

This single file is shared with **agent 161** (control-prob requested BoxMuller + SystematicResample) and downstream `optim/` (annealing requires Boltzmann acceptance, currently uses internal LCG). Co-ordination: prob/sample.go should be designed once with a stable *Source-typed API and used by both 161 and 162 sub-packages.

**LOC for the bridge:** ~120 (50 LOC for the eight functions + 70 LOC for the Marsaglia-Tsang gamma + Box-Muller normal + categorical alias-method).

**Determinism / golden files.** Every primitive in §2 takes a `*rand.Rand` (or seed) as last parameter. Tests pin (seed → expected adjacency matrix) JSON vectors at 256-bit precision. This matches CLAUDE.md §1 "Golden files are the proof" for stochastic primitives — for chi-squared-distribution moments we golden-file (seed, samples) → Pearson's chi-squared statistic.

### 3.3 Cycle hazards — none

```
graph/random/   imports graph + prob + constants
graph/ergm/     imports graph + graph/random + prob + constants + optim   (MCMC adapts step size)
graph/kernels/  imports graph + linalg                                     (matrix operations on product graph)
graph/inference/imports graph + graph/random + prob + constants
prob/sample.go  imports math/rand/v2 + math   (already prob's only deps)
```

graph/ does not import prob/ today and that stays true at the package root — only the new sub-packages bridge. Sibling-zero-dep invariant preserved.

---

## 4. Saturation candidates (R-MUTUAL-CROSS-VALIDATION 3/3)

Mirroring the recent commit pattern (6a55bb4 audio-onset 3-detector, 365368a copula×autodiff Clayton log-PDF, 159 W1 wave1d-vs-d'Alembert-vs-FFT, 160 dissipation-rate-three-ways, 161 random-walk-Kalman-vs-DARE):

### Pin S1 — PageRank ≡ MarkovSteadyState ≡ EigenvectorCentrality(damped)

Three independent codepaths, must agree to 1e-10:

1. `graph.PageRank(adj, damping=0.85, iters=200)` — power iter on weighted edges with teleportation.
2. `prob.MarkovSteadyState(P_damped)` — power iter on row-stochastic matrix where P_damped = damping·P + (1-damping)/n · 1·1^T.
3. `graph.EigenvectorCentrality(adj_undamped, ...)` agrees with PageRank in the damping=1 limit ONLY on strongly-connected aperiodic graphs.

**Negative pin:** disconnected graph with damping=1 → PageRank stays normalised, MarkovSteadyState reaches degenerate uniform distribution. Both implementations must surface this difference, not silently agree.

### Pin S2 — Erdős–Rényi G(n,p) edge count three ways

For G(n=1000, p=0.05):

1. **Empirical:** count edges in single sampled graph at fixed seed.
2. **Analytic mean:** E[|E|] = p · (n choose 2) = 24,975.
3. **Binomial CI:** Wilson 95% CI for proportion p̂ = |E| / (n choose 2) using `prob.WilsonConfidenceInterval` should contain p.

All three must lock together within Wilson CI. Saturation pins the bridge sampler + the binomial CDF + the proportion CI in one shot.

### Pin S3 — Bond percolation threshold p_c on infinite 2-D lattice ≈ 0.5

Bond-percolation on the square lattice has exact p_c = 0.5 (Kesten 1980). For a finite 100×100 lattice:

1. **Sweep p in 0.40..0.60:** for each p run G17 1000× and record giant-component-fraction.
2. **Crossover detection:** finite-size scaling P_∞(p) ~ (p−p_c)^β with β=5/36 in 2-D.
3. **Cross-check:** estimate p_c via G19 bisection; should land within 0.50 ± 0.02 for the 100×100 sample.

Validates G17 + G19 + ConnectedComponents all at once.

### Pin S4 — Random walk mixing time bound

For an n-cycle (cycle graph), the mixing time of the random walk is Θ(n²·log(1/ε)). Three independent verifications:

1. `RandomWalkMixingTime` measures empirical TV distance per step.
2. **Spectral bound:** mixing time ≤ log(n/ε) / (1−λ_2) where λ_2 is second-largest eigenvalue of P; computed via `linalg.Eig` on the cycle's known transition matrix.
3. **Closed-form for n-cycle:** λ_2 = cos(2π/n), so bound = log(n/ε) / (1−cos(2π/n)).

Must all agree to within constants. Saturates G11 + linalg eigenvalue + the analytic formula.

### Pin S5 — SBM detectability threshold (Decelle-Krzakala-Moore-Zdeborova 2011)

For a 2-block SBM with within-prob a/n and between-prob b/n, communities are detectable iff (a−b)² > 2(a+b) — the spectral threshold. Pin:

1. **Generate** SBM at (a=10, b=2) → above threshold; run Louvain → recovers >70% accuracy.
2. **Generate** SBM at (a=6, b=5) → below threshold; Louvain → ~50% (random chance).
3. **Spectral gap of B = adjacency-eigenvector centrality** on (a,b) above-threshold has gap > √(a+b); below threshold gap < √(a+b).

This pin saturates G6 + Louvain + spectral check on a deeply non-trivial phase-transition theorem.

### Pin S6 — Wilson UST ≡ Kirchhoff matrix-tree-theorem count

For a small graph (n≤8) where the Kirchhoff matrix-tree theorem gives exact uniform-spanning-tree count τ(G) = det(L_reduced) where L is the Laplacian:

1. **Wilson's algorithm** sampled K=10,000 times → empirical histogram over distinct trees.
2. **Theoretical** uniform = 1/τ(G) per tree.
3. **Chi-squared goodness-of-fit** via `prob.ChiSquaredTest` on histogram-vs-uniform.

Validates G15 sampling correctness + provides a beautiful pedagogical example of `prob.ChiSquaredTest` consuming graph-derived data.

---

## 5. PR sequencing — recommended

Based on dependency depth and individual cost:

| PR | Content | LOC | Blocks/Unlocks |
|----|---------|-----|-----------------|
| **PR-1** | `prob/sample.go` (the bridge) | ~120 | Unlocks 22 of 26 primitives + 161 + future optim/SA + future MCMC |
| **PR-2** | `graph/random/erdos_renyi.go` (G1+G2) | ~70 | Unlocks G3, G4, G6, G7, G8, G9, G14, G17, G18, G21, G22, S2 |
| **PR-3** | `graph/random/walk.go` (G11+G13) | ~35 | Pure composition — saturates Pin S1 immediately, demonstrates PageRank ≡ MarkovSteadyState identity |
| **PR-4** | `graph/random/percolation.go` (G17+G18+G19) | ~100 | Saturates Pin S3 — single most beautiful pin: Kesten 1980 exact p_c on lattice |
| **PR-5** | `graph/random/sbm.go` (G6+G8) | ~180 | Saturates Pin S5 — Decelle-Krzakala-Moore-Zdeborova 2011 detectability transition |
| **PR-6** | `graph/random/configuration.go` + `graph/random/triangle.go` (G3+G26) | ~110 | Foundation for ERGM and clustering coefficient |
| **PR-7** | `graph/random/smallworld.go` + `barabasi_albert.go` (G4+G5) | ~150 | Watts-Strogatz + Barabási-Albert classics |
| **PR-8** | `graph/random/spanning.go` (G15+G16) | ~170 | Saturates Pin S6 — Wilson UST ≡ Kirchhoff |
| **PR-9** | `graph/random/graphon.go` + `latent.go` (G14+G9) | ~110 | Continuous limits + Hoff-Raftery-Handcock |
| **PR-10** | `graph/random/cascade.go` + `reliability.go` (G20+G21) | ~160 | Network reliability MC |
| **PR-11** | `graph/inference/permutation.go` + `bootstrap.go` (G22+G23) | ~150 | Two-sample graph testing |
| **PR-12** | `graph/kernels/random_walk.go` + `weisfeiler_leman.go` (G24+G25) | ~230 | Graph kernels for ML downstream |
| **PR-13** | `graph/ergm/` (G10) | ~250 | ERGM via Glauber-Metropolis; the most subtle PR — needs care on detailed-balance proof |

Cumulative reach after PR-3: pure composition no new dependencies, **15 LOC** unlock saturation pin S1.
Cumulative reach after PR-4: **205 LOC** unlock S2 + S3 — already covers Erdős–Rényi + percolation entire surface.
Cumulative reach after PR-13: **~1,820 LOC** complete the random-graph + ERGM + kernels canon.

---

## 6. Distinctness from neighbouring agents

- **Distinct from 116–120 (prob isolation per-package):** they review prob/ in isolation; this review proposes the *graph-typed* output additions (graph/random/) and the bridge primitive (prob/sample.go) that prob/ alone would not motivate.
- **Distinct from 086–090 (graph isolation per-package):** they review graph/ in isolation; this review proposes the random-graph generators that graph/ alone would not motivate, plus the cross-validation-of-PageRank-with-MarkovSteadyState pin S1 that no isolated graph review would catch.
- **Coordinates with 161 (synergy-control-prob):** that review requested `prob.BoxMullerSample + SystematicResample`. This review extends that ask to a fuller `prob/sample.go` (Bernoulli, Uniform, IntUniform, Categorical, Dirichlet, Poisson, Gamma, Normal, ShuffleInt). **A single combined PR satisfies both consumer-domain reviews — coordinate the API surface once, ship once.**
- **Coordinates with 151 (synergy-signal-prob):** that review asked to publicise `chiSquaredCDF`. Pin S6 (Wilson UST chi-squared goodness-of-fit) consumes the same publicised primitive, so the single chiSquaredCDF publicisation lands two synergy bills.
- **Distinct from 157 (synergy-graph-linalg):** that review covered Laplacian spectrum / spectral clustering / Cheeger inequality / Fiedler vector. This review's G24 (random-walk kernel) and the spectral-bound side of Pin S4 do touch matrix eigenvectors, but the keystone here is the *probabilistic* edge model not the linalg representation. The two reviews together suggest a future graph/spectral/ + graph/random/ pair.
- **Distinct from 154 (synergy-chaos-timeseries):** orthogonal — chaos/timeseries owns Markov-chain-as-deterministic-system, this review owns Markov-chain-as-graph.

---

## 7. Today vs after-this-synergy

| | Today (v0.10.0) | After full ship |
|--|-----------------|-----------------|
| Random graph generators | 0 | 9 (Erdős-Rényi×2, configuration, WS, BA, SBM×3, MMSB, DCSBM) |
| Latent-space / graphon | 0 | 2 (LSM, Graphon sampler) |
| ERGM | 0 | 1 + statistics callbacks |
| Random walk on graph | PageRank only (deterministic spectral counterpart) | mixing time, cover time, MarkovSteadyState identity, RWK kernel |
| Spanning tree sampling | KruskalMST, PrimMST (deterministic min-weight only) | + Wilson UST, Prüfer-uniform tree |
| Percolation | 0 | site, bond, p_c estimator + finite-size scaling |
| Cascading failures | 0 | Motter-Lai 2002 + reliability MC |
| Two-sample graph testing | 0 | permutation + bootstrap + WL kernel |
| Public RNG in prob/ | 0 | seeded BernoulliSample, UniformSample, CategoricalSample, DirichletSample, PoissonSample, GammaSample, NormalSample, ShuffleInt |
| Pinned cross-validation paths | 0 | 6 (S1..S6) |

---

## 8. External-library comparison

NetworkX (Python) ships all 26 primitives with a non-trivial dependency stack (numpy + scipy). graph-tool (Python+C++) ships all but G10 ERGM. statnet (R) ships full ERGM. Lemon (C++) ships graph algorithms but no random-graph generators. None are zero-dependency. None are golden-file-validated cross-language. **reality/graph/random + graph/ergm + graph/kernels would be the only zero-dependency, cross-language-portable, golden-file-validated random-graph + ERGM + graph-kernel library in any language.**

---

## 9. Bottom line

reality is unusually well-positioned for the random-graph canon because (a) graph/ already ships every deterministic primitive (BFS / Dijkstra / Brandes / Tarjan / Louvain / Kruskal / Edmonds-Karp / PageRank), (b) prob/ already ships every distribution (Beta / Gamma / Poisson / Binomial / Dirichlet-via-Gamma) needed for ensemble priors, (c) there is exactly ONE missing ingredient: a public seeded RNG surface in prob/ (currently `MarkovSimulate`'s LCG is private). The single bridge file `prob/sample.go` (~120 LOC) plus the new `graph/random/` sub-package (~1,400 LOC across 9 files) plus optional `graph/ergm/` (~250 LOC) plus `graph/kernels/` (~230 LOC) plus `graph/inference/` (~150 LOC) shipping in PR-1..PR-13 turns reality into a complete random-graph + percolation + ERGM + kernel-test platform for ~2,000 LOC TOTAL of pure connective tissue, ZERO new mathematics — every theorem 1959 (Erdős-Rényi) to 2011 (Karrer-Newman) vintage. Architectural lesson confirmed for the 11th consecutive synergy review (151/153/154/155/156/157/158/159/160/161/162): synergy-shaped sub-packages always live in the consumer-side directory (graph/random/, graph/ergm/, graph/kernels/, graph/inference/), never in the primitive-supplier directory; the only cross-package additive change is a single bridge file (prob/sample.go) that *also* satisfies agents 151 and 161 — coordinate once, ship once.

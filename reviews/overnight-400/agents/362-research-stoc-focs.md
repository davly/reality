# 362 — research-stoc-focs (STOC/FOCS 2024-2025 algorithms reality should track)

## Headline
STOC '25 broke Dijkstra's sorting barrier and edge-coloring's near-quadratic ceiling; FOCS '24 sealed Dijkstra universal-optimality, gave optimal quantile sketches, and almost-linear network unreliability — five concrete algorithms reality should mirror in `graph`, `prob`, `signal`, `compression`.

## Top papers (sorted by impact x fit)

### 1. Duan, Mao, Mao, Shu, Yin — "Breaking the Sorting Barrier for Directed SSSP" (STOC 2025, Best Paper)
arXiv:2504.17033. Deterministic O(m log^(2/3) n) SSSP on directed graphs with non-negative real weights in the comparison-addition model. First sub-Dijkstra bound on sparse graphs since 1959; recursively shrinks the frontier by interleaving Dijkstra and Bellman-Ford with a group-insert/extract structure that escapes the Omega(n log n) heap-ordering cost. Multiple open-source impls already exist (Rust DunMaoSSSP, C DMMSY-SSSP). Direct fit for `graph` package. Clean comparison-addition model — matches reality's "no FP tricks, document precision" rule. Practical caveat: large hidden constants; benchmark vs Dijkstra-with-Fibonacci before adopting as default.

### 2. Haeupler, Hladik, Rozhon, Tarjan, Tetek — "Universal Optimality of Dijkstra via Beyond-Worst-Case Heaps" (FOCS 2024, Best Paper)
arXiv:2311.11793. Proves Dijkstra with the right heap is universally optimal — for every fixed graph topology, no comparison-based SSSP algorithm beats it on more than o(1) of weight assignments. Constructs a working-set heap where extract-min costs O(log k) where k = inserts after the extracted item (not total size). First "universal optimality" result for any sequential algorithm. Even simpler variant in Haeupler-Hladik-Rozhon "Simpler Universally Optimal Dijkstra" (ESA 2025). For reality's `graph`: replace pairing/binary heap with this working-set heap to get instance-optimal Dijkstra without changing the outer algorithm.

### 3. Assadi, Behnezhad, Bhattacharya, Costa, Solomon, Zhang — "Vizing's Theorem in Near-Linear Time" (STOC 2025, Best Paper)
arXiv:2410.05240. Randomized O(m log Delta) algorithm computing a (Delta+1)-edge-coloring w.h.p. — closes a 40-year gap (Gabow et al. O(m sqrt(n)), 1985). Vizing's theorem is constructive but his original proof was O(mn); intermediate work hit O~(n^2), O~(mn^(1/3)), O~(mn^(1/4)) before this near-optimal bound. Strong fit for `graph` package: edge coloring is the canonical scheduling/register-allocation primitive. SODA 2025 has a simpler (Delta+1) variant ("Faster Vizing and Near-Vizing", arXiv:2405.13371) suitable for first impl.

### 4. Gupta, Singhal, Wu — "Optimal Quantile Estimation: Beyond the Comparison Model" (FOCS 2024, Best Student Paper)
arXiv:2404.03847. Deterministic streaming sketch estimating any rank to within additive epsilon*n using O(1/epsilon) words (= O(epsilon^-1 (log(epsilon n) + log(epsilon U))) bits). First sketch beating the comparison-based lower bound and first improvement on q-digest in 20 years. Drop-in replacement for `prob.Quantile` streaming variants. Directly competes with t-digest/DDSketch but with provable bounds. Reality should add this alongside existing exact quantile primitives — golden vectors are easy because the sketch is deterministic.

### 5. Cen, Li, Panigrahi — "Network Unreliability in Almost-Linear Time" (STOC 2025)
arXiv:2503.23526. m^(1+o(1))-time FPTAS for the network unreliability problem (probability that a graph disconnects given iid edge failures). Optimal up to subpolynomial factors since reading the graph is Omega(m). Builds on Karger's spanning-tree-packing estimator + the almost-linear max-flow machinery. Fit for `prob` x `graph`: network reliability is the canonical Bayesian-graph primitive. Implementable but heavy — depends on min-cut/spanning-tree-packing infrastructure reality doesn't yet have.

### 6. Ghaffari, Grunau — "Near-Optimal Deterministic Network Decomposition and Ruling Set, and Improved MIS" (FOCS 2024, Best Paper)
arXiv:2410.19516. Deterministic distributed network decomposition in ~O(log^2 n) rounds with O(log n) diameter and O(log n) colors — matches the Omega(log^2 n) lower bound. Ruling-set construction (O(log log n) ruling set in O(log n) rounds) is exponential improvement on Awerbuch-Goldberg-Luby-Plotkin 1989. Mostly relevant to distributed systems, but the sequential analogue (BFS-layered decomposition) is a useful primitive for `graph` connectivity/sparsifiers. Lower priority for reality unless a distributed-graph slot opens.

### 7. Goyal, Harsha, Kumar, Shankar — "Fast List Decoding of Univariate Multiplicity and Folded Reed-Solomon Codes" (FOCS 2024)
Randomized O~(n) list-decoding algorithms for FRS and univariate multiplicity codes — codes that achieve list-decoding capacity (decode (1-r-epsilon) errors at rate r) but historically had only polynomial-time decoders. Reality has no `coding` package yet, but Reed-Solomon decoding is the natural base primitive for any error-correcting addition. Pair with Brakensiek-Gopi-Makam combinatorial result (random RS achieves capacity over linear-sized fields, STOC '24/FOCS '23). If reality grows a `coding` slot, this is the modern decoder to ship.

### 8. Chen, Lian, Mao, Zhang — "An Improved Pseudopolynomial Time Algorithm for Subset Sum" (FOCS 2024)
arXiv:2402.14493. O~(n + sqrt(w*t))-time Subset-Sum where w = max integer, t = target. Beats Bringmann's O~(n+t) (SODA '17) when t >> w. Convolution-based — uses sumset structure + careful FFT. Fit for `combinatorics` and `compression` (knapsack-style coding). Direct port: needs FFT (already in `signal`) and modular convolution.

### 9. Bansal, Cohen-Addad, Prabhu, Saulpic, Schwiegelshohn — "Sensitivity Sampling for k-Means: Worst Case and Stability Optimal Coreset Bounds" (FOCS 2024)
Tight bounds for sensitivity-sampled coresets for k-means: O(k epsilon^-2) points in stable instances, matching lower bounds. Replaces ad-hoc coreset sizes used in current k-means libraries. Fit for `linalg`/`prob`: coreset is the right primitive for "compress n points to a weighted subset preserving objective". Useful when reality adds a clustering submodule (none today).

### 10. Buchbinder, Feldman — "Deterministic Algorithm and Faster Algorithm for Submodular Maximization Subject to a Matroid Constraint" (FOCS 2024)
First deterministic (1-1/e)-approximation for monotone submodular maximization under matroid constraints, plus a faster randomized variant. Resolves a long-standing question about derandomizing the continuous-greedy algorithm. Fit for `optim` (submodular optimization is currently absent); useful primitive for any "select k diverse items maximizing coverage" problem (sensor placement, feature selection).

### 11. Liu — "On Approximate Fully-Dynamic Matching and Online Matrix-Vector Multiplication" (FOCS 2024)
Bridges OMv conjecture and fully dynamic (1-epsilon)-matching: shows OMv-hardness essentially tight for current dynamic matching algorithms. Algorithmic side: O(n^{1.5+o(1)}) total update time for (1-epsilon)-bipartite matching. Combined with FOCS '24 result of similar bounds for dense-graph dynamic matching, gives the modern toolkit. Fit for `graph` if dynamic algorithms become a goal.

### 12. Bernstein, Bhattacharya, Kiss, Saranurak — "Deterministic Dynamic Maximal Matching in Sublinear Update Time" (STOC 2025)
First deterministic fully-dynamic maximal matching in O~(n^{8/9}) amortized update time, breaking the long-standing Omega(n) barrier on dense graphs. Uses edge-degree-constrained subgraph (EDCS) plus monotone Even-Shiloach trees and random walks on directed expanders. Heavy machinery; relevant only if reality adds dynamic graph algorithms.

### 13. Alman, Duan, V. Williams, Xu, Xu, Zhou — "More Asymmetry Yields Faster Matrix Multiplication" (SODA 2025; mentioned for completeness)
arXiv:2404.16349. omega < 2.371339, improving 2.371552. Asymmetric laser method. Not implementable as a practical algorithm (galactic), but the theoretical exponent matters for `linalg` documentation: reality should cite the current best omega in `linalg/README.md` and never claim O(n^3) is optimal. Strassen's O(n^2.807) remains the practical cutoff for hand-rolled implementations.

## Reality slot recommendations

- **`graph` (priority 1):**
  - Implement Duan-Mao-Mao-Shu-Yin SSSP (#1) as a second SSSP variant alongside Dijkstra; benchmark crossover point.
  - Replace internal heap with Haeupler-Tetek working-set heap (#2) to get universal-optimality for free; preserve current Dijkstra outer loop.
  - Add Vizing (Delta+1) edge-coloring (#3) — start with simpler near-Vizing SODA '25 variant; full near-linear later.

- **`prob` (priority 2):**
  - Add Gupta-Singhal-Wu deterministic quantile sketch (#4) as `prob.QuantileSketch` with O(1/epsilon) words. Pair with existing exact quantile primitive.
  - Add Cen-Li-Panigrahi network unreliability FPTAS (#5) once min-cut + spanning-tree-packing land in `graph`.

- **`combinatorics` (priority 3):**
  - Replace existing subset-sum (if any) with Chen et al. O~(n+sqrt(wt)) (#8); reuse `signal.FFT` for convolution.

- **`optim` (priority 3):**
  - Add Buchbinder-Feldman deterministic submodular max under matroid (#10) when a submodular slot opens.

- **New `coding` package (deferred):**
  - Fast list decoding (#7) for Reed-Solomon / Folded-RS as the modern primitive.

- **Documentation hygiene (cheap):**
  - Cite omega < 2.371339 (#13) in `linalg` provenance metadata.
  - Note Vizing's theorem and (Delta+1)-edge-coloring complexity (#3) wherever graph coloring is referenced.

## Sources
- [STOC 2025 Best Paper Award (SIGACT)](https://www.sigact.org/prizes/best_paper.html)
- [STOC Best Paper Award: How to Find the Shortest Path Faster (MPI-INF)](https://www.mpi-inf.mpg.de/news/detail/stoc-best-paper-award-how-to-find-the-shortest-path-faster)
- [Duan et al. arXiv:2504.17033](https://arxiv.org/abs/2504.17033)
- [STOC 2025 Accepted Papers](https://acm-stoc.org/stoc2025/accepted-papers.html)
- [STOC 2025 Proceedings](https://acm-stoc.org/stoc2025/toc.html)
- [Haeupler et al. arXiv:2311.11793](https://arxiv.org/abs/2311.11793)
- [Quanta — Best Way to Traverse a Graph](https://www.quantamagazine.org/computer-scientists-establish-the-best-way-to-traverse-a-graph-20241025/)
- [Assadi et al. arXiv:2410.05240](https://arxiv.org/abs/2410.05240)
- [Faster Vizing and Near-Vizing — arXiv:2405.13371](https://arxiv.org/abs/2405.13371)
- [Gupta-Singhal-Wu arXiv:2404.03847](https://arxiv.org/abs/2404.03847)
- [Cen-Li-Panigrahi arXiv:2503.23526](https://arxiv.org/abs/2503.23526)
- [Ghaffari-Grunau arXiv:2410.19516](https://arxiv.org/abs/2410.19516)
- [Chen et al. Subset Sum arXiv:2402.14493](https://arxiv.org/abs/2402.14493)
- [More Asymmetry Yields Faster Matmul arXiv:2404.16349](https://arxiv.org/abs/2404.16349)
- [FOCS 2024 Accepted Papers](https://focs.computer.org/2024/accepted-papers-for-focs-2024/)
- [FOCS 2025 Accepted Papers](https://focs.computer.org/2025/accepted-papers/)
- [Computational Complexity blog: FOCS 2024](https://blog.computationalcomplexity.org/2024/10/focs-2024.html)
- [INSAIT FOCS '24 best paper note](https://insait.ai/insait-scientists-with-a-breakthrough-in-algorithms-and-a-best-paper-award-at-focs24/)
- [Yang P. Liu publications](https://yangpliu.github.io/research.html)
- [11011110 highlights from FOCS 2024](https://11011110.github.io/blog/2024/11/03/highlights-from-focs.html)

# 363 — research-soda (SODA 2024-2025 discrete algorithms reality should track)

## Headline
SODA 2024-25 advanced dynamic graphs (subpolynomial-update min-cut, fully-dynamic spanners), faster matchings (combinatorial bipartite, Bayesian online ≥0.678), partial-information sorting in optimal comparisons, and sublinear (Δ+1)-coloring via asymmetric palette sparsification — most relevant to `reality/graph`, `linalg`, `prob`, `combinatorics`.

## Top papers

### 1. Fully Dynamic Approximate Minimum Cut in Subpolynomial Time per Operation (SODA 2025)
First fully-dynamic algorithm for (1+o(1))-approximate global minimum cut achieving n^{o(1)} update time, breaking the long-standing polynomial-time-per-update barrier for dynamic min-cut. Builds on expander decomposition and dynamic low-conductance routing. Pairs well with the deterministic edge-connectivity result below in showing min-cut-machinery maturing across both static and dynamic regimes. Reality has no dynamic graphs yet, but this is the canonical reference for any future incremental cut/connectivity API. Relevance: `graph` (dynamic edge-connectivity primitive).

### 2. Deterministic Edge Connectivity and Max Flow Using Subquadratic Cut Queries (SODA 2025)
First deterministic algorithm computing global min-cut on a simple graph using Õ(n^{5/3}) cut queries (sub-quadratic), and Õ(n)-size s-t max-flows in Õ(n^{5/3}) queries. Establishes deterministic versions of expander decomposition and isolating-cuts in the cut-query model. Strong candidate algorithm for any future `graph.MinCut` / `graph.MaxFlow` work since it derandomizes Karger-line techniques. Reality currently lacks max-flow entirely — this is the "right" textbook target for a deterministic baseline. arXiv via SIAM SODA 2025 proceedings doi:10.1137/1.9781611978322.4.

### 3. A Faster Combinatorial Algorithm for Maximum Bipartite Matching (SODA 2024, Chuzhoy–Khanna)
Combinatorial Õ(m·n^{1/4}) algorithm for maximum bipartite matching — first significant combinatorial improvement over Hopcroft-Karp Õ(m·n^{1/2}) in 50 years, without resorting to interior-point/IPM machinery. Builds on weighted push-relabel and length-constrained expander decompositions. Highly relevant: reality could expose a clean `graph.BipartiteMatching` based on this rather than IPM. Code is implementable from first principles (CLAUDE.md rule 6). doi:10.1137/1.9781611977912.79.

### 4. Fully Dynamic Algorithms for Graph Spanners via Low-Diameter Router Decomposition (SODA 2025)
Fully-dynamic algorithm maintaining a (2k-1)-spanner with Õ(n^{1+1/k}) edges under both edge insertions and deletions, against an adaptive adversary, with sublinear amortized update time. Introduces "low-diameter router decomposition" — a primitive likely to appear in many future dynamic-graph results. Spanners are the right space-vs-stretch primitive for distance-oracle work and routing. Relevance: future `graph.Spanner` API; also informs static spanner construction. doi:10.1137/1.9781611978322.23.

### 5. Fast and Simple Sorting Using Partial Information (SODA 2025, Haeupler-Hladík-Iacono-Rozhoň-Tarjan)
Resolves a 1976 problem: deterministic O(m + log T)-time, O(log T)-comparison algorithm for sorting n items given m known pre-existing comparisons (T = number of consistent total orders). Previous best with O(log T) comparisons cost O(n^{2.5}) time. Uses topological sort + heapsort with the right heap (fibonacci-style) + sorted-list search. Modified version solves top-k optimally to constants. Reality's `sequence` package already has fuzzy/string algorithms — this would be a natural addition for partially-ordered-input sort. Direct first-principles port.

### 6. New Approximation Algorithms and Reductions for n-Pairs Shortest Paths and All-Nodes Shortest Cycles (SODA 2025)
New approximation framework for n-PSP and ANSC, improving both running times and approximation factors. Reduces ANSC ≤ n-PSP, plus combinatorial speedups via fast matrix multiplication and 3SUM-style techniques. For reality's `graph` package, n-PSP is the natural intermediate between SSSP (current Dijkstra/A*) and APSP (which reality lacks); this is the modern reference implementation. doi:10.1137/1.9781611978322.177.

### 7. Beyond 2-Approximation for k-Center in Graphs (SODA 2025)
Breaks the long-standing 2-approximation barrier for graph k-Center. Achieves (2-ε, O(1))-bicriteria in time O(m·n + n^{k/2+1}); for k≥10 yields (3/2, 1/2)-bicriteria in n^{k-1+1/(k+1)+o(1)}. Significant because Hochbaum-Shmoys 2-approx (1985) was conjectured tight for plain approximation. Relevance: `graph` clustering/facility-location primitives if reality ever extends `prob` clustering to graph metrics. doi:10.1137/1.9781611978322.6.

### 8. Massively Parallel Minimum Spanning Tree in General Metric Spaces (SODA 2025)
Round-complexity tight bounds for metric MST in MPC: Ω(log n) is unavoidable for exact metric MST (matching folklore), but approximation breaks the bound. Proves (i) ε-dependency optimal, (ii) approximation strictly necessary to beat O(log n) rounds, (iii) metric MST strictly harder than low-dim Euclidean MST. Reality's `graph.MST` is sequential; this is the reference for any future parallel/distributed extension. Also informs the linalg-graph border. doi:10.1137/1.9781611978322.5.

### 9. New Philosopher Inequalities for Online Bayesian Matching, via Pivotal Sampling (SODA 2025)
0.678-approximation for online Bayesian bipartite matching (improves prior 0.652), and 0.685 for vertex-weighted special case. Uses pivotal sampling with discarding to induce strong negative correlations among offline nodes. Yields polytime truthful pricing-based mechanisms with the same ratio. Relevance: `prob` (negative correlation / pivotal sampling primitives), `gametheory` (mechanism-design baseline), `optim` (online Bayesian matching). doi:10.1137/1.9781611978322.98.

### 10. Settling the Pass Complexity of Approximate Matchings in Dynamic Graph Streams (SODA 2025, Assadi et al.)
Tight pass-complexity bounds for approximate maximum matching in dynamic (insert/delete) graph streams using O(n·polylog n) space — semi-streaming model. Settles a question open since Ahn-Guha-McGregor 2012. Establishes the right complexity-class for streaming-matching primitives. Relevance: any future `streaming` package in reality; complementary to current `compression`/`signal` streaming infrastructure. doi:10.1137/1.9781611978322.25.

### 11. Tight Bounds and Phase Transitions for Incremental and Dynamic Retrieval (SODA 2025)
Optimal bounds for incremental + dynamic retrieval data structures (key→value mapping with fast lookup, no membership). Discovers a phase transition: in the incremental setting, space redundancy shrinks from ~n·log log n down to Θ(n) as value size approaches log n bits. Static retrieval is a building block of perfect hash families and minimal perfect hashes. Relevance: `crypto` (perfect-hash construction), `compression` (succinct dictionaries). doi:10.1137/1.9781611978322.135.

### 12. Simple Sublinear Algorithms for (Δ+1) Vertex Coloring via Asymmetric Palette Sparsification (SOSA/SODA-companion 2025)
Asymmetric Palette Sparsification (APST) generalizes Assadi-Chen-Khanna 2019 PST to allow per-vertex list-sizes with only an average bound. Recovers nearly-optimal sublinear, MPC, and streaming (Δ+1)-coloring with much simpler algorithms and analyses. Relevance: any future `graph.Coloring` in reality should follow this "simpler is better" path; matches CLAUDE.md rule 6 (first principles). arXiv:2502.17629.

### 13. A Tight VC-Dimension Analysis of Clustering Coresets with Applications (SODA 2025)
Sharp VC-dimension bounds for k-median/k-means coreset constructions, yielding smaller coresets with rigorous guarantees and improved running-time tradeoffs in clustering. Relevance: directly applicable as a citation for any coreset-based clustering reality may add to `prob`/`linalg` (PCA-coreset hybrids). doi:10.1137/1.9781611978322.162.

### 14. Correlation Clustering and (De)Sparsification (STOC 2025 — companion to SODA streaming work)
Introduces "graph de-sparsification" — recover an unweighted simple graph with approximately the original cut/spectral structure from a sketch. Ports classical correlation-clustering gains to dynamic-streaming, MPC, and distributed models in a black-box way. Strong candidate for any future `graph.Sparsifier`/`graph.CorrelationClustering` in reality. (STOC, but tight thematic fit with SODA 2025 streaming track.)

## Reality slot recommendations
- `graph.MaxFlow` / `graph.MinCut`: prefer the SODA 2025 deterministic cut-query algorithm (#2) as canonical baseline; deterministic, derandomizes Karger.
- `graph.BipartiteMatching` (new): port Chuzhoy-Khanna combinatorial Õ(mn^{1/4}) (#3) — clean first-principles fit, no IPM dependency.
- `graph.APSP` and `graph.AllShortestCycles` (new): cite #6 for current SODA-2025 approximation tradeoffs; reality already has Dijkstra/A* but no APSP.
- `graph.kCenter`: when added, target the (2-ε) bicriteria from #7 instead of stale Hochbaum-Shmoys 2-approx.
- `graph.Spanner` (new): use #4's low-diameter router decomposition as the static recipe (decomposition is reusable).
- `graph.MST`: keep current Kruskal/Prim; document #8's lower bound for any future parallel `graph.ParallelMST`.
- `sequence.SortPartial` (new): port #5 — partial-info sorting in O(m + log T) time. Natural neighbor of existing `sequence.NGramDiceCoefficient`/fuzzy code.
- `prob.OnlineMatching` and `gametheory.MechanismDesign`: cite #9 for negative-correlation/pivotal-sampling primitives and 0.678 ratio.
- `crypto.Retrieval` / `compression.MinimalPerfectHash` (new): cite #11's phase-transition Θ(n) bound when constructing succinct retrieval structures.
- `graph.Coloring` (new): if added, follow APST (#12) — simpler analysis, sublinear-friendly, MPC/streaming portable.
- Future `streaming` package: anchor on #10 pass-complexity result; pair with #14 graph-desparsification primitive for sketching support.

## Sources
- [SIAM SODA 2025 Proceedings](https://epubs.siam.org/doi/book/10.1137/1.9781611978322)
- [SODA 2025 program (Pagh)](https://rasmuspagh.net/sodaprogram/)
- [DBLP SODA 2025](https://dblp.org/db/conf/soda/soda2025.html)
- [Berkeley EECS — three best papers SODA 2024](https://eecs.berkeley.edu/news/berkeley-eecs-wins-three-best-paper-awards-at-soda/)
- [Michigan CSE — sixteen papers SODA 2025](https://cse.engin.umich.edu/stories/sixteen-papers-by-cse-researchers-at-soda-2025)
- [DIMACS — Srivastava SODA 2025 best paper (folded RS codes)](http://dimacs.rutgers.edu/news_archive/shashank-award)
- [Chuzhoy–Khanna SODA 2024 bipartite matching PDF](https://www.cis.upenn.edu/~sanjeev/papers/soda24_bipartite_matching.pdf) (also doi:10.1137/1.9781611977912.79)
- [SODA 2025 Fully Dynamic Approx Min Cut](https://epubs.siam.org/doi/10.1137/1.9781611978322.22)
- [SODA 2025 Fully Dynamic Spanners](https://epubs.siam.org/doi/10.1137/1.9781611978322.23)
- [SODA 2025 Deterministic EC + MaxFlow via Cut Queries](https://doi.org/10.1137/1.9781611978322.4)
- [SODA 2025 Parallel MST in General Metric Spaces](https://epubs.siam.org/doi/10.1137/1.9781611978322.5)
- [SODA 2025 Beyond 2-Approx k-Center](https://epubs.siam.org/doi/10.1137/1.9781611978322.6)
- [SODA 2025 Online Bayesian Matching](https://doi.org/10.1137/1.9781611978322.98)
- [SODA 2025 Approximate Matchings in Dynamic Streams (Assadi et al.)](https://epubs.siam.org/doi/10.1137/1.9781611978322.25)
- [SODA 2025 Incremental/Dynamic Retrieval Phase Transitions](https://doi.org/10.1137/1.9781611978322.135)
- [SODA 2025 Sorting w/ Partial Information](https://doi.org/10.1137/1.9781611978322.134)
- [SODA 2025 n-PSP and ANSC](https://epubs.siam.org/doi/10.1137/1.9781611978322.177)
- [SODA 2025 VC-Dim Coresets](https://epubs.siam.org/doi/10.1137/1.9781611978322.162)
- [Asymmetric Palette Sparsification (arXiv:2502.17629)](https://arxiv.org/abs/2502.17629)
- [Correlation Clustering & (De)Sparsification (STOC 2025)](https://www.researchwithrutgers.com/en/publications/correlation-clustering-and-desparsification-graph-sketches-can-ma/)

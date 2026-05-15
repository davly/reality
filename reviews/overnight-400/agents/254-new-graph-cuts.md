# 254 | new-graph-cuts — Boykov-Kolmogorov / push-relabel / α-expansion / α-β-swap / QPBO / TRW-S / belief-propagation / dual-decomposition / multiway-cut

**Summary line 1.** reality v0.10.0 ships **ONE** max-flow primitive (`graph/flow.go::MaxFlow` Edmonds-Karp BFS-Ford-Fulkerson, O(V·E²), ~100 LOC, validated by 081-graph-numerics F5 as float-residual-noise-prone but textbook-correct on integer capacities) and **ZERO** of the energy-minimisation / MRF-MAP-inference / submodular-optimisation / multi-label-graph-cut surface — repo-wide grep on `boykov.kolmogorov|push.relabel|goldberg.tarjan|fifo.preflow|highest.label|gap.heuristic|alpha.expansion|alpha.beta.swap|veksler|qpbo|boros.hammer|kolmogorov.rother|pseudo.boolean|submodular.cut|ishikawa|kolmogorov.zabih|regularity.condition|trw|tree.reweighted|wainwright.jaakkola.willsky|sequential.trw|kolmogorov.2006|belief.propagation|loopy.bp|sum.product|max.product|junction.tree|dual.decomposition|komodakis|paragios|message.passing|mplp|globerson.jaakkola|lp.relaxation.mrf|ilp.mrf|ad3|martins|figueiredo|branch.and.bound.map|bethe.approximation|mean.field.mrf|gibbs.mrf|metropolis.mrf|simulated.annealing.mrf|ising.image|potts.mrf|markov.random.field|conditional.random.field|crf.image|multiway.cut|multi.cut|klein.plotkin.stein|dahlhaus|garg|hierarchical.graph.cut|parallel.graph.cut|delong.boykov|fusion.move|range.move|jump.move|move.making|expansion.move|swap.move` returns **zero callable matches** outside `graph/flow.go::MaxFlow` and the false-positive name-collision `chaos/systems::Ising` (which is a 1-D-spin-glass-mean-field-thermodynamics primitive — NOT 2-D-image-Ising-MRF-MAP-inference). The Edmonds-Karp implementation is **algorithmically wrong-asymptotic** for the grid-graphs that energy-minimisation produces: BK-2004 reports 5-10× speedup over EK on 256×256-image graph-cuts, push-relabel is 2-3× over EK on dense graphs, and EK's O(V·E²) blows up on V=65k-pixel grids (~10⁹ ops vs BK's ~10⁸). **PARTIAL OVERLAP with 252 (S14 BoykovKolmogorovMaxFlow ~200 LOC + S15 alpha-expansion ~100 LOC under "Tier-3 graph-based ~600 LOC" of `image/segment/`):** slot 254 is the **deeper-zoom on the algorithmic-graph-cut substrate that lives ONE ABSTRACTION LAYER BELOW segmentation** — 252's S14/S15 are *consumers* of a `graph/cuts/` (or `graph/maxflow/`) sub-package that 254 enumerates as the proper home for the BK-2004-augmenting-path, Goldberg-Tarjan-1988 push-relabel, Dinic-1970 blocking-flow, IBFS-2010, parallel-BK-2010, dynamic-graph-cut-Kohli-Torr-2007, QPBO-Boros-Hammer-2002, QPBO-I/QPBO-P-Rother-Kolmogorov-Lempitsky-Szummer-2007, α-expansion + α-β-swap-Boykov-Veksler-Zabih-2001, α-expansion-fusion-Lempitsky-Rother-Roth-Blake-2010, range-move-Veksler-2007, jump-move-Veksler-2012, TRW-S-Kolmogorov-2006, MPLP-Globerson-Jaakkola-2008, AD³-Martins-Figueiredo-Aguiar-2011, junction-tree-Lauritzen-Spiegelhalter-1988, loopy-BP-Pearl-1988, multiway-cut-Dahlhaus-Johnson-Papadimitriou-Seymour-Yannakakis-1992, and the regularity-condition-Kolmogorov-Zabih-2004 (F²-submodularity test that determines whether an arbitrary pseudo-boolean function is graph-representable). **PARTIAL OVERLAP with 081-graph-numerics F5:** 081 already enumerated the float-capacity termination-noise hazard in `graph.MaxFlow`; slot 254 is the **algorithmic-replacement track** (BK + push-relabel) that subsumes that fix — once BK ships, EK should remain only as the cross-validation-oracle that 252-Pin-2 specifies. **Block-C verdict:** the entire energy-minimisation / discrete-MRF-MAP-inference / pseudo-boolean-optimisation tier is ABSENT and the substrate-gap is the second-largest in the `graph/` package after the missing kdtree (097-flagged) — once BK + push-relabel + QPBO + α-expansion + TRW-S land in `graph/cuts/`, slot 252's segmentation tier ships against a proven O(V·E·|C|) substrate instead of EK's O(V·E²) substrate.

**Summary line 2.** **Twenty-six primitives C1-C26 totalling ~3,640 LOC** organised as **(a) Tier-0 max-flow algorithm tier ~860 LOC** (C1 `graph/cuts/dinic.go` Dinic-1970 / Even-Itai-Shamir-1976 blocking-flow O(V²·E) via BFS-level-graph + DFS-multi-augmenting-paths-per-phase, the simplest "modern" max-flow surpassing EK ~120 LOC; C2 `graph/cuts/boykov_kolmogorov.go` Boykov-Kolmogorov-2004 PAMI-26:1124 augmenting-path-with-tree-reuse maintaining S-tree-from-source + T-tree-from-sink with growth/augment/adopt-orphan three-stage-loop and parent-relabel on saturation, the de-facto-standard for vision graph-cuts ~280 LOC; C3 `graph/cuts/pushrelabel.go` Goldberg-Tarjan-1988 JACM-35:921 generic push-relabel preflow with FIFO selection rule O(V²·√E) ~140 LOC; C4 `graph/cuts/highest_label.go` Cherkassky-Goldberg-1997-Algorithmica-19:390 highest-label selection rule with global-relabelling + gap-heuristic for the practically-fastest push-relabel variant ~120 LOC; C5 `graph/cuts/ibfs.go` Goldberg-Hed-Kaplan-Tarjan-Werneck-2011-ESA Incremental-Breadth-First-Search blocking-flow combining BK-tree-reuse with Dinic-blocking-flow ~140 LOC; C6 `graph/cuts/dynamic_cut.go` Kohli-Torr-2007 PAMI-29:2079 dynamic-graph-cut for video segmentation reusing residual-graph between frames ~60 LOC), **(b) Tier-1 min-cut extraction + s-t-cut substrate ~280 LOC** (C7 `graph/cuts/min_cut.go` extract min-cut partition `(S,T)` from saturated residual-graph BFS-from-source O(V+E) ~40 LOC; C8 `graph/cuts/grid_graph.go` 4-connected / 8-connected / 26-connected grid-graph builder with `BuildGrid2D(w, h, conn int, unary func(p) (sCap, tCap), pairwise func(p, q) cap)` ~120 LOC; C9 `graph/cuts/multiway_cut.go` Dahlhaus-Johnson-Papadimitriou-Seymour-Yannakakis-1992-SODA / Calinescu-Karloff-Rabani-2000 multiway-cut as `(2 − 2/k)`-approximation ~120 LOC), **(c) Tier-2 pseudo-boolean optimisation + regularity ~520 LOC** (C10 `graph/cuts/regularity.go` Kolmogorov-Zabih-2004 PAMI-26:147 F²-regularity-condition checker `θ(0,0) + θ(1,1) ≤ θ(0,1) + θ(1,0)` per pairwise-term + reduction-to-graph-cut for arbitrary submodular pseudo-boolean ~80 LOC; C11 `graph/cuts/qpbo.go` Boros-Hammer-2002-Discrete-Appl-Math-123:155 / Rother-Kolmogorov-Lempitsky-Szummer-2007-CVPR Quadratic-Pseudo-Boolean-Optimisation via roof-duality producing partial-labelling `{0, 1, ?}` for non-submodular pseudo-boolean ~220 LOC; C12 `graph/cuts/qpboi.go` QPBO-Improved Rother-2007 sec.4 with strongly-connected-component reduction + auto-fusion for the unlabelled `?` set ~120 LOC; C13 `graph/cuts/qpbop.go` QPBO-Probing Rother-2007 sec.5 partial-fixing-by-test-labelling for hard non-submodular ~100 LOC; C14 `graph/cuts/ishikawa.go` Ishikawa-2003 PAMI-25:1333 multi-label energy with **convex** pairwise terms reduced to single graph-cut via layered graph construction O(K·V) ~80 LOC), **(d) Tier-3 multi-label move-making ~520 LOC** (C15 `graph/cuts/alpha_expansion.go` Boykov-Veksler-Zabih-2001 PAMI-23:1222 α-expansion via repeated binary BK-cuts, provably 2c-approximation under metric pairwise terms ~140 LOC; C16 `graph/cuts/alpha_beta_swap.go` Boykov-Veksler-Zabih-2001 α-β-swap, applies to wider class than expansion (semi-metric instead of metric pairwise) at cost of weaker-approximation ~100 LOC; C17 `graph/cuts/fusion_move.go` Lempitsky-Rother-Roth-Blake-2010 PAMI-32:1392 fusion-move generalising α-expansion: take TWO labelings and find min-energy convex-combination via QPBO ~120 LOC; C18 `graph/cuts/range_move.go` Veksler-2007-CVPR / Kumar-Torr-2008 range-move expanding to a *range* of labels per move (faster convergence on truncated convex priors) ~80 LOC; C19 `graph/cuts/jump_move.go` Veksler-2012 jump-move for ordered-label (depth/disparity) energies ~80 LOC), **(e) Tier-4 LP-relaxation message-passing ~720 LOC** (C20 `graph/cuts/trw_s.go` Kolmogorov-2006 PAMI-28:1568 Sequential-Tree-Reweighted-Message-Passing — monotonically-improving lower-bound on MAP-MRF-energy, dual-LP-relaxation solver, more robust than loopy-BP ~280 LOC; C21 `graph/cuts/loopy_bp.go` Pearl-1988 + Murphy-Weiss-Jordan-1999-UAI loopy belief-propagation max-product / sum-product on factor graphs ~140 LOC; C22 `graph/cuts/junction_tree.go` Lauritzen-Spiegelhalter-1988-JRSS-50:157 junction-tree exact-inference via clique-tree-message-passing — exponential in tree-width but exact ~160 LOC; C23 `graph/cuts/mplp.go` Globerson-Jaakkola-2008-NIPS Max-Product-Linear-Programming dual-decomposition with cluster-based-reparameterisation ~140 LOC), **(f) Tier-5 LP / ILP / branch-and-bound ~360 LOC** (C24 `graph/cuts/lp_mrf.go` Schlesinger-1976 / Wainwright-Jaakkola-Willsky-2005 LP-relaxation of MRF-MAP via local-marginal-polytope, solved by `optim/lp` Simplex/IPM ~140 LOC; C25 `graph/cuts/branch_bound.go` branch-and-bound for MAP using LP-relaxation as lower-bound at each node ~120 LOC; C26 `graph/cuts/ad3.go` Martins-Figueiredo-Aguiar-Smith-2011-ICML Alternating-Directions-Dual-Decomposition combining ADMM with LP-relaxation for MAP ~100 LOC), **(g) Tier-6 sampling-based MAP ~380 LOC** (C27 `graph/cuts/gibbs_sampler.go` Geman-Geman-1984-PAMI-6:721 Gibbs-sampler for MRF + simulated-annealing-MAP ~100 LOC; C28 `graph/cuts/swendsen_wang.go` Swendsen-Wang-1987-PRL-58:86 cluster-Monte-Carlo for Ising/Potts MRF ~120 LOC; C29 `graph/cuts/mean_field.go` Bethe-1935 / Yedidia-Freeman-Weiss-2005-IT-51:2282 mean-field + Bethe-approximation variational-inference on MRF ~80 LOC; C30 `graph/cuts/parallel_bk.go` Strandmark-Kahl-2010-CVPR parallel-BK via dual-decomposition splitting graph along cuts ~80 LOC).

**SINGULAR-FOUNDATIONAL C2 Boykov-Kolmogorov ~280 LOC** — single-most-cited graph-cut max-flow algorithm in computer vision (>15,000 citations), the de-facto-standard substrate that 252-S14 / 253-A20 / every modern segmentation paper consumes; no zero-dep cross-language Go implementation exists worldwide; the canonical reference C++ is Vladimir Kolmogorov's `maxflow-3.04` (~1,500 LOC, GPL-licensed — cannot be wrapped, must be reimplemented from-paper for MIT). **SINGULAR-CHEAPEST-1-DAY C1 Dinic + C7 min-cut-extract + C8 grid-graph-builder ~280 LOC** — Dinic-1970 is the simplest provably-faster-than-EK max-flow (O(V²·E) vs O(V·E²)), no tree-reuse-machinery to debug, and the C7+C8 substrate is the bridge from "max-flow-value" to "segmentation-mask-image" that EVERY user actually wants. Ships entirely against `graph.MaxFlow`-PRESENT signature (drop-in replacement, golden-file-cross-validation against EK on small instances). **SINGULAR-MOAT C2 BK-2004 + C11 QPBO + C15 α-expansion ~640 LOC** — the trinity that powers every graph-cut-based vision system from 2001 to today (GrabCut-Rother-Kolmogorov-Blake-2004 = BK + α-expansion; OpenCV `cv::GraphCut` + `cv::grabCut` wrap exactly this trio; the Microsoft Photo-Magnetic-Lasso uses BK + IBFS). The QPBO addition is the SINGLE differentiator that lets reality handle non-submodular energies (deformable-shape-matching, stereo-with-occlusion, photo-stitching) that pure BK cannot — these are the energies at which textbook-α-expansion silently produces wrong-answers because the binary-subproblem violates Kolmogorov-Zabih-regularity. **SINGULAR-2024-FRONTIER C20 TRW-S + C26 AD³ ~380 LOC** — Sequential-TRW-Kolmogorov-2006 is the production-default LP-relaxation solver in every modern probabilistic-vision system (the OpenGM library wraps TRW-S + AD³ + QPBO + α-expansion as the four-pillar dispatch), monotonically-improving-lower-bound is the production-quality property that loopy-BP lacks, and AD³-Martins-2011 is the post-2010 dual-decomposition refinement that beats TRW-S on tightness for many real instances. **SINGULAR-PEDAGOGICAL C1 Dinic + C3 push-relabel + C2 BK ~540 LOC** — the three canonical max-flow algorithms in CLRS / Cormen-Leiserson-Rivest-Stein-3e §26 + the BK-2004 specialised-for-vision adjunct that most teaching-curricula now add. Five R-MUTUAL-CROSS-VALIDATION 3/3 pins enabled (see §3). Recommended placement **NEW sub-package `graph/cuts/`** under the existing `graph/` package — same "consumer-shaped sub-package" precedent as 252 (`image/segment/`) and 253 (`levelset/`), plus the existing `graph/` package is the natural-supplier-of-substrate-with-no-circular-imports. Strict-downstream of `graph/flow.go::MaxFlow` (oracle for testing); strict-upstream of 252-S14/S15 segmentation-graph-cut tier and 253-A21 (Local-CV uses BK as inner-loop) and 251-shape-opt (combinatorial-shape-optimisation via min-cut-inflation).

---

## 0. State at HEAD (2026-05-09, v0.10.0)

Repo-wide audit for graph-cut / energy-minimisation / MRF-MAP-inference / pseudo-boolean-optimisation surface.

| Surface | Path | Graph-cut relevance |
|---|---|---|
| `graph.MaxFlow` (Edmonds-Karp BFS-Ford-Fulkerson) | `graph/flow.go:25-104` | Substrate; **PRESENT** but O(V·E²) wrong-asymptotic for vision-grids (BK-2004 reports 5-10× speedup over EK on grid graphs); 081-F5 flags float-residual-noise termination hazard |
| `graph.TopologicalSort` | `graph/flow.go:127` | Unrelated (DAG-ordering, not max-flow) |
| `graph.Dijkstra / AStar` | `graph/shortest.go` | Substrate for fast-marching but NOT for max-flow |
| `graph.BellmanFord` | `graph/bellman_ford.go` | Unrelated |
| `graph.{KruskalMST, PrimMST}` | `graph/mst.go` | Substrate for Felzenszwalb-Huttenlocher segmentation (252-S12) but NOT for cuts |
| `graph.LouvainCommunities` | `graph/community.go` | Modularity-based community-detection — different objective from min-cut |
| `optim/proximal.{ProxL1, ProxBox, Admm, Fista, Fbs}` | `optim/proximal/*.go` | Substrate for LP-relaxation primal-dual and AD³ ADMM; **PRESENT** |
| `optim/lp` (Simplex / IPM) | -- | LP-relaxation substrate; **ABSENT** (097-linalg-missing flags) |
| `prob.{NormalPDF, Bernoulli}` | `prob/distributions.go` | Substrate for unary-data-term log-likelihood; **PRESENT** |
| `linalg.{Matrix, ConjugateGradient}` | `linalg/*.go` | Substrate for LP / IP / TRW-S; **PARTIAL** (CG absent — 244-D12 ships) |
| `graph/cuts/` package | -- | **ABSENT** — this slot creates |
| C1-C30 graph-cut primitives | -- | **ALL ABSENT** |

**False-positive name-collisions audited:**
- `chaos/systems::Ising` — 1-D-spin-glass / mean-field-thermodynamics dynamical-system, NOT 2-D-image-Ising-MRF for MAP-inference. Different math, different consumer.
- `combinatorics::Partition` — set-partition-counting (Bell-numbers / Stirling-numbers), NOT image-multi-cut-partition.
- `compression::Huffman` — prefix-code-tree, NOT max-flow-tree-reuse-of-BK-2004 (homonym only).
- `prob/conformal::CutOffQuantile` — quantile-cut-off, unrelated to graph-cuts.

**Cross-import edges that this slot creates:**
- `graph/cuts → graph` for shared `IntAdjacency` type, `MaxFlow` oracle, BFS infrastructure.
- `graph/cuts → linalg.{ConjugateGradient, SparseMatrix}` (244-D12 + 097) for C20 TRW-S + C24 LP-MRF + C26 AD³.
- `graph/cuts → optim/lp` (097-flagged) for C24 LP-MRF + C25 branch-and-bound.
- `graph/cuts → optim/proximal.{Admm, Fbs}` for C26 AD³ + C29 mean-field.
- `graph/cuts → prob.NormalPDF` for unary-data-term log-likelihood (consumer-side, optional).

**Strict downstream consumers of `graph/cuts/`:**
- `image/segment/graph_cut.go` (252-S14) → uses `cuts.BoykovKolmogorov` (C2) instead of EK.
- `image/segment/alpha_expansion.go` (252-S15) → uses `cuts.AlphaExpansion` (C15).
- `image/segment/multiphase_chan_vese.go` (252-S20 / 253-A22) → can use `cuts.AlphaExpansion` (C15) for discrete K-phase as alternative to log₂(K) level-sets.
- `image/segment/chan_vese.go` (252-S19 / 253-A20) → uses `cuts.QPBO` (C11) for non-submodular extensions (Bhattacharyya-CV requires QPBO).
- 251-shape-opt T7 (combinatorial-shape-opt via inflation) → uses `cuts.BoykovKolmogorov` (C2).
- 215-compressed-sensing T-binary-CS → uses `cuts.QPBO` (C11) for non-submodular sparse-recovery.

---

## 1. The thirty primitives (C1-C30)

Each entry: name, LOC, reference, API sketch.

### Tier 0 — Max-flow algorithm tier (~860 LOC)

**C1 — `graph/cuts/dinic.go` ~120 LOC.** `Dinic(adj IntAdjacency, capacity map[[2]int]float64, source, sink int) float64` Dinic-1970 / Even-Itai-Shamir-1976 blocking-flow. Phase = (BFS-build-level-graph then DFS-find-augmenting-path-in-level-graph until no more paths in this level-graph). Each phase increases shortest-augmenting-path-length by ≥1 → at most V phases → O(V²·E) total. Simpler than BK-2004; faster than EK in practice on dense graphs by 3-5×; on grid-graphs BK still beats it by 2-3×. **Refs.** Dinic-1970 *Soviet-Math-Doklady-11:1277*; Even-Itai-Shamir-1976 *J-ACM-23:248*; CLRS-3e §26.2.

**C2 — `graph/cuts/boykov_kolmogorov.go` ~280 LOC — KEYSTONE.** `BoykovKolmogorov(adj, capacity, source, sink) (flow float64, partition []bool)` Boykov-Kolmogorov-2004 PAMI-26:1124. Maintains two trees `S` (rooted at source, all edges oriented away) and `T` (rooted at sink, all edges oriented toward) covering subset of nodes; un-touched nodes are *free*. Three-stage main-loop:
- **Growth.** Active node `p∈S` (resp. `T`) scans edges; for any edge `(p,q)` with non-zero residual where `q` is free, add `q` to `S` (resp. `T`); if `q` is in opposite tree → *augmenting path found*.
- **Augment.** Push max-bottleneck-flow along the path; saturated edges create *orphans* (children whose parent-edge is saturated).
- **Adopt.** For each orphan `p∈S`, scan incoming edges for a new parent in `S` with non-zero residual originating-from-source; if none, declare `p` free. Recurse on grandchildren.

The tree-reuse + orphan-adoption is the single innovation over EK; on 256×256 grid-graphs reports 5-10× speedup. R-MUTUAL-CROSS-VALIDATION 3/3 against C1 Dinic + `graph.MaxFlow` (PRESENT EK). Returns max-flow value AND `partition[v]` indicating reachability-from-source in the saturated residual-graph (the min-cut). **Refs.** Boykov-Kolmogorov-2004 *PAMI-26:1124*; Boykov-Veksler-Zabih-2001 *PAMI-23:1222* preliminary.

**C3 — `graph/cuts/pushrelabel.go` ~140 LOC.** `PushRelabel(adj, capacity, source, sink) float64` Goldberg-Tarjan-1988 JACM-35:921 generic. Maintains *preflow* (excess at each node) + *height* labels `h[v]`; PUSH excess from `u` to `v` if `h[u] = h[v]+1` and `(u,v)` has residual; RELABEL `u` to `min h[w]+1` over residual-out-edges if no push possible. FIFO selection rule: O(V²·√E). Generic O(V²·E) is the textbook bound; FIFO improves it to O(V³). **Refs.** Goldberg-Tarjan-1988 *JACM-35:921*; CLRS-3e §26.4.

**C4 — `graph/cuts/highest_label.go` ~120 LOC.** `HighestLabelPushRelabel(adj, capacity, source, sink) float64` Cherkassky-Goldberg-1997-Algorithmica-19:390. Selection rule: always push from highest-label active node. Plus `GlobalRelabel` periodically running BFS-from-sink to update heights to true distance, plus `GapHeuristic` detecting empty heights `h*` and bumping all nodes above to V (proves they cannot reach sink). **Practically the fastest push-relabel variant** — within 1.2× of BK-2004 on grid-graphs in DIMACS-1991 benchmarks. **Refs.** Cherkassky-Goldberg-1997 *Algorithmica-19:390*; Goldberg-1995 *Algorithmica-13:506*.

**C5 — `graph/cuts/ibfs.go` ~140 LOC.** `IBFS(adj, capacity, source, sink) float64` Goldberg-Hed-Kaplan-Tarjan-Werneck-2011-ESA Incremental-Breadth-First-Search. Combines BK's tree-reuse with Dinic's blocking-flow via incremental BFS-tree maintenance. Reports 1.5-2× speedup over BK-2004 on vision graphs. The post-2011 production-default in some pipelines. **Refs.** Goldberg-Hed-Kaplan-Tarjan-Werneck-2011 *ESA-2011*; Hochbaum-2008 *Algorithmica-50:1* pseudo-flow algorithm (alternative).

**C6 — `graph/cuts/dynamic_cut.go` ~60 LOC.** `DynamicCut(prevResidual, deltaCapacities, source, sink) float64` Kohli-Torr-2007 PAMI-29:2079. Reuse the residual-graph from a previous max-flow computation; only re-process the changed edges. Used in video-segmentation where consecutive frames share most pixels. Provides 5-10× speedup vs from-scratch on temporally-coherent inputs. **Refs.** Kohli-Torr-2007 *PAMI-29:2079*.

### Tier 1 — Min-cut extraction + s-t-cut substrate (~280 LOC)

**C7 — `graph/cuts/min_cut.go` ~40 LOC.** `MinCut(saturatedResidual ResidualGraph, source int) (sourceSide []bool)` BFS-from-source on saturated residual. O(V+E). The bridge from "I have max-flow-value" to "I have a labelling 0/1 per node". Trivial but ESSENTIAL.

**C8 — `graph/cuts/grid_graph.go` ~120 LOC.** Grid-graph builder + canonical-energy-minimisation API:

```go
type GridEnergy struct {
    Width, Height, Connectivity int  // 4, 8, 26 (3-D)
    UnaryCost                   func(p int) (sCap, tCap float64) // data term
    PairwiseCost                func(p, q int) float64           // smoothness term
}

func (e GridEnergy) Solve() (labels []bool, energy float64) // calls C2 BK + C7 MinCut
```

Captures the structural pattern of 90% of vision-graph-cut consumers (binary-segmentation, stereo with α-expansion outer loop, denoising). 4-/8-/26-connectivity is the standard set. **Refs.** Boykov-Funka-Lea-2006 *IJCV-70:109* survey; Kolmogorov-Zabih-2004 *PAMI-26:147* graph-construction.

**C9 — `graph/cuts/multiway_cut.go` ~120 LOC.** `MultiwayCut(adj, capacity, terminals []int) (labels []int, cost float64)` Dahlhaus-Johnson-Papadimitriou-Seymour-Yannakakis-1992 SODA. Minimum-weight edge-set whose removal separates k terminals into k components. NP-hard for k≥3 but `(2 − 2/k)`-approximation via *isolating-cuts* algorithm; `(3/2 − 1/k)` via Calinescu-Karloff-Rabani-2000 LP-relaxation. **Refs.** Dahlhaus-Johnson-Papadimitriou-Seymour-Yannakakis-1992 *SIAM-J-Comput-23:864*; Calinescu-Karloff-Rabani-2000 *J-Comput-Syst-Sci-60:564*.

### Tier 2 — Pseudo-boolean optimisation + regularity (~520 LOC)

**C10 — `graph/cuts/regularity.go` ~80 LOC.** `IsRegular(theta_pq func(li, lj bool) float64) bool` Kolmogorov-Zabih-2004 PAMI-26:147 F²-regularity-condition: `θ(0,0) + θ(1,1) ≤ θ(0,1) + θ(1,0)` (i.e., the pairwise term is *submodular*). PLUS `ReduceToGraphCut(unary, pairwise []) (adj, capacity)` mechanical-construction-rules from §4 of the paper turning any submodular F² (and via auxiliary-variable construction, F³ + F⁴) into an equivalent graph-cut instance. **The single most-important theoretical paper** in graph-cut energy-minimisation — answers exactly what energies are graph-representable. **Refs.** Kolmogorov-Zabih-2004 *PAMI-26:147*; Freedman-Drineas-2005-CVPR F⁴-extension; Živný-Cohen-Jeavons-2011-AI-175:1827 the-complexity-of-soft-constraint-languages (full classification).

**C11 — `graph/cuts/qpbo.go` ~220 LOC — KEYSTONE.** `QPBO(unary []float64, pairwise []PairwiseTerm) (labels []int, fullyLabelled bool)` Boros-Hammer-2002-Discrete-Appl-Math-123:155 / Rother-Kolmogorov-Lempitsky-Szummer-2007-CVPR. Roof-duality: for any (potentially non-submodular) pseudo-boolean function E(x), construct a doubled-variable graph with 2n nodes representing (x, ¬x) such that min-cut yields a *partial-optimal-labelling* — each node gets `0`, `1`, or `?` (unlabelled). Theorem (Boros-Hammer): the labelled subset is part of SOME global-minimum (autarky property). The `?` set is the *non-submodular core*. **The single algorithm that lets graph-cut handle non-submodular energies.** Application: stereo-with-occlusion, deformable matching, image-stitching. **Refs.** Boros-Hammer-2002 *Discrete-Appl-Math-123:155*; Hammer-Hansen-Simeone-1984 *Math-Programming-28:121* roof-duality-original; Rother-Kolmogorov-Lempitsky-Szummer-2007 *CVPR-2007*.

**C12 — `graph/cuts/qpboi.go` ~120 LOC.** `QPBOImproved(unary, pairwise) []int` Rother-2007 §4. After QPBO leaves an unlabelled set `U`, run strongly-connected-component reduction on the residual-graph + auto-fusion to label MORE of `U` while preserving partial-optimality. Reports labelling 80-95% of variables on hard instances where vanilla QPBO labels 50%. **Refs.** Rother-Kolmogorov-Lempitsky-Szummer-2007 §4.

**C13 — `graph/cuts/qpbop.go` ~100 LOC.** `QPBOProbing(unary, pairwise, maxIter) []int` Rother-2007 §5. For each unlabelled variable `u∈U`, test-label `u=0` and `u=1`, run QPBO each time, intersect results: any variable labelled identically across both probes is fixed. Iterate. Provably labels every variable on instances where it terminates; usually does. **The strongest non-submodular MAP-inference algorithm prior to TRW-S.** **Refs.** Rother-Kolmogorov-Lempitsky-Szummer-2007 §5; Boros-Hammer-Tavares-2008 *Discrete-Optim-5:501* preprocessing for QPBO.

**C14 — `graph/cuts/ishikawa.go` ~80 LOC.** `IshikawaMultiLabel(unary [][]float64, pairwise func(li, lj int) float64, K int) []int` Ishikawa-2003 PAMI-25:1333. Multi-label energy with **convex** pairwise terms `g(|li − lj|)` reduced to *single* graph-cut on a layered graph with K·V nodes. Edge constructions encode the discrete-derivative-bound. Provably exact for convex pairwise; O(K·V) graph size. The right algorithm when pairwise term is convex; α-expansion is the right algorithm when pairwise term is metric-but-not-convex. **Refs.** Ishikawa-2003 *PAMI-25:1333*; Schlesinger-2007-CVPR submodular-multi-label-extension.

### Tier 3 — Multi-label move-making (~520 LOC)

**C15 — `graph/cuts/alpha_expansion.go` ~140 LOC — FOUNDATIONAL.** `AlphaExpansion(unary [][]float64, pairwise PairwiseFn, K int, maxIter int) []int` Boykov-Veksler-Zabih-2001 PAMI-23:1222. Outer loop over labels α ∈ {0..K-1}; for each α, solve the binary subproblem "for each variable, is its current label ≥ α-better, or expand to α?" via single BK-cut (regularity-condition C10 holds iff pairwise is *metric* — `V(α,β) ≥ 0`, `V(α,α) = 0`, `V(α,β) = V(β,α)`, `V(α,β) ≤ V(α,γ) + V(γ,β)`). Convergence in O(1) outer-iterations in practice; provably 2c-approximation (`c = max V/min V`) under metric. **Powers Photoshop's Magic-Wand-2.0, GrabCut, OpenCV's `cv::stereoBM` post-processing, Microsoft Kinect's body-segmentation.** **Refs.** Boykov-Veksler-Zabih-2001 *PAMI-23:1222*; Veksler-1999-PhD; Komodakis-Tziritas-2007 *PAMI-29:1436* primal-dual extension.

**C16 — `graph/cuts/alpha_beta_swap.go` ~100 LOC.** `AlphaBetaSwap(unary, pairwise, K, maxIter) []int` Boykov-Veksler-Zabih-2001. For each pair (α, β), solve the binary subproblem "swap α and β labels among current α-or-β-labelled variables". Applies to *semi-metric* pairwise (no triangle-inequality required), wider than α-expansion's metric. Weaker approximation guarantee but ships in less-restrictive regime. **Refs.** Boykov-Veksler-Zabih-2001 *PAMI-23:1222*.

**C17 — `graph/cuts/fusion_move.go` ~120 LOC.** `FusionMove(currentLabels, proposalLabels []int, unary, pairwise) (newLabels []int)` Lempitsky-Rother-Roth-Blake-2010 PAMI-32:1392. For each variable `i`, choose between `currentLabels[i]` and `proposalLabels[i]` to minimise energy. The binary-fusion subproblem is non-submodular in general (proposals can encode any labelling) → solve via QPBO (C11). Generalises α-expansion (which uses constant-α as proposal). Workhorse for optical-flow, stereo with arbitrary-shaped expansion-moves. **Refs.** Lempitsky-Rother-Roth-Blake-2010 *PAMI-32:1392*; Lempitsky-Rother-Blake-2007-CVPR LogCut-original.

**C18 — `graph/cuts/range_move.go` ~80 LOC.** `RangeMove(currentLabels, alpha, beta int, unary, pairwise) []int` Veksler-2007-CVPR / Kumar-Torr-2008. Like α-expansion but the "expansion target" is a *range* `[α, β]` of labels — for variables in the range, choose any label in `[α, β]`; for others, keep current. Pairwise-cost between two new-labels `li, lj ∈ [α, β]` reduces to graph-cut iff pairwise is convex on the range. Faster convergence than α-expansion on truncated-convex priors (stereo, depth). **Refs.** Veksler-2007-CVPR; Kumar-Torr-2008-PAMI-30:1.

**C19 — `graph/cuts/jump_move.go` ~80 LOC.** `JumpMove(currentLabels, jumpSize int, unary, pairwise) []int` Veksler-2012. Each variable can either stay or jump to label `current + jumpSize`. Application-specific (depth-discontinuity preservation). **Refs.** Veksler-2012-IJCV-100:96.

### Tier 4 — LP-relaxation message-passing (~720 LOC)

**C20 — `graph/cuts/trw_s.go` ~280 LOC — 2024-FRONTIER.** `TRWS(unary, pairwise, factorGraph, maxIter) (lowerBound float64, primal []int)` Kolmogorov-2006 PAMI-28:1568 Sequential-Tree-Reweighted-Message-Passing. Decompose factor-graph into spanning-trees; on each tree run exact dynamic-programming (Viterbi); propagate messages between trees in a fixed schedule that **monotonically increases the dual lower-bound on MAP-energy**. The schedule is the key: TRW-S vs the parallel TRW (Wainwright-Jaakkola-Willsky-2005-IT-51:3697 original) is monotone-non-decreasing on the dual, where the parallel version may oscillate.

The lower-bound is computed exactly; the primal is recovered by argmax of accumulated messages. Termination: lower-bound stops increasing → fixed-point of dual-LP; if `lowerBound = primalEnergy`, MAP is exact.

**The post-2006 production-default for non-submodular MAP-MRF inference.** Used in OpenGM (Andres-Beier-Kappes-2012 OpenGM-2.0 library), BUNDLER (Snavely-2006 photo-tourism), KinectFusion (Newcombe-2011 ISMAR). **Refs.** Kolmogorov-2006 *PAMI-28:1568*; Wainwright-Jaakkola-Willsky-2005 *IT-51:3697* original-TRW; Komodakis-Paragios-2008-CVPR survey.

**C21 — `graph/cuts/loopy_bp.go` ~140 LOC.** `LoopyBP(factorGraph, maxIter) (marginals [][]float64, mapLabels []int)` Pearl-1988 + Murphy-Weiss-Jordan-1999-UAI. Both max-product (MAP) and sum-product (marginals) variants. Local message-passing on factor-graph; converges exactly on trees (junction-tree-equivalence), heuristically on loopy-graphs (Bethe-fixed-point connection — Yedidia-Freeman-Weiss-2005). Less robust than TRW-S but the textbook entry-point. **Refs.** Pearl-1988 *Probabilistic Reasoning in Intelligent Systems*; Murphy-Weiss-Jordan-1999 *UAI-1999*; Yedidia-Freeman-Weiss-2005 *IT-51:2282*.

**C22 — `graph/cuts/junction_tree.go` ~160 LOC.** `JunctionTree(factorGraph) (tree, cliquePotentials)` + `Inference(tree)` Lauritzen-Spiegelhalter-1988 JRSS-50:157. Triangulate factor-graph → maximal-cliques → clique-tree (junction-tree property: running-intersection). Exact inference via two-pass message-passing. Cost is exponential in tree-width (small for chains, exponential for grids). The textbook EXACT algorithm — for small-tree-width problems the right choice. **Refs.** Lauritzen-Spiegelhalter-1988 *JRSS-50:157*; Cowell-Dawid-Lauritzen-Spiegelhalter-1999 *Probabilistic Networks and Expert Systems*.

**C23 — `graph/cuts/mplp.go` ~140 LOC.** `MPLP(factorGraph, maxIter) (lowerBound, primal)` Globerson-Jaakkola-2008-NIPS Max-Product-Linear-Programming. Dual-decomposition with cluster-based-reparameterisation: tighter LP-relaxation than pairwise-LP via adding cycle-inequalities or larger clusters. Sontag-Globerson-Jaakkola-2008-NIPS extends to higher-order. Used by Dlib's `find_max_factor_graph_potts`. **Refs.** Globerson-Jaakkola-2008 *NIPS-2008*; Sontag-Globerson-Jaakkola-2008 *NIPS-2008* tightening-relaxations.

### Tier 5 — LP / ILP / branch-and-bound (~360 LOC)

**C24 — `graph/cuts/lp_mrf.go` ~140 LOC.** `LPMrfRelaxation(unary, pairwise) (lpValue float64, fractionalLabels [][]float64)` Schlesinger-1976 / Wainwright-Jaakkola-Willsky-2005 LP-relaxation of MAP-MRF over local-marginal-polytope. Variables: `μ_p(l)` and `μ_pq(l, l')`; constraints: marginalisation + non-negativity. Solve via `optim/lp` Simplex / IPM. Provides a true certificate of optimality when integer-solution + LP-bound match. **Refs.** Schlesinger-1976 *Kibernetika-12* (the original LP-MRF, USSR); Wainwright-Jaakkola-Willsky-2005 *IT-51:3697*; Sontag-Jaakkola-2007-NIPS new-outer-bounds.

**C25 — `graph/cuts/branch_bound.go` ~120 LOC.** `BranchBound(unary, pairwise, lowerBound LowerBoundFn) []int` standard branch-and-bound for MAP-MRF using LP-relaxation (C24) or TRW-S-dual (C20) as lower-bound at each tree-node. Branches on highest-uncertainty variable. Optimal for small-K-small-V instances; defers to heuristics for large. **Refs.** Lauritzen-1996 *Graphical Models* §6.

**C26 — `graph/cuts/ad3.go` ~100 LOC.** `AD3(factorGraph, maxIter) (primal, lowerBound)` Martins-Figueiredo-Aguiar-Smith-Xing-2011-ICML Alternating-Directions-Dual-Decomposition. ADMM-based LP-MRF solver — splits variables across factors, augmented-Lagrangian-update. Faster convergence than TRW-S on some instances; tighter-LP than pairwise-LP via factor-augmentation. **Refs.** Martins-Figueiredo-Aguiar-Smith-Xing-2011 *ICML-2011*; Martins-Smith-Xing-2009-EMNLP-alpha-version.

### Tier 6 — Sampling-based MAP + variational (~380 LOC)

**C27 — `graph/cuts/gibbs_sampler.go` ~100 LOC.** `GibbsSampler(unary, pairwise, schedule SchedFn, steps int, rng) []int` Geman-Geman-1984 PAMI-6:721. Coordinate-Gibbs-update; with annealing schedule `T_k → 0` → simulated-annealing-MAP. The original MRF-MAP algorithm, predates graph-cut by 17 years. **Refs.** Geman-Geman-1984 *PAMI-6:721*; Kirkpatrick-Gelatt-Vecchi-1983 *Science-220:671* simulated-annealing.

**C28 — `graph/cuts/swendsen_wang.go` ~120 LOC.** `SwendsenWang(unary, pairwise, beta, steps, rng) []int` Swendsen-Wang-1987 PRL-58:86 + Barbu-Zhu-2003-CVPR cluster-algorithm extension to MRF-MAP. Cluster-Monte-Carlo: build random-bond-clusters (with probability `1 − exp(−β·V)` along edges), flip whole clusters at once. Avoids critical-slowing-down of single-site Gibbs near phase-transitions. **Refs.** Swendsen-Wang-1987 *PRL-58:86*; Wolff-1989 *PRL-62:361* alternative; Barbu-Zhu-2003 *CVPR-2003* MRF-extension.

**C29 — `graph/cuts/mean_field.go` ~80 LOC.** `MeanField(unary, pairwise, maxIter) [][]float64` + `BetheApproximation(...)`. Mean-field variational: minimise KL(q‖p) over fully-factorised q. Bethe-approximation: minimise Bethe-free-energy (loopy-BP-fixed-point connection). Looser bound than TRW-S but cheap; used as initialisation. **Refs.** Yedidia-Freeman-Weiss-2005 *IT-51:2282*; Bethe-1935 *Proc-Royal-Soc-London-A-150*.

**C30 — `graph/cuts/parallel_bk.go` ~80 LOC.** `ParallelBK(adj, capacity, source, sink, numWorkers)` Strandmark-Kahl-2010-CVPR. Dual-decomposition splitting graph along cuts; run BK on each subgraph; merge via Lagrange-multiplier updates. Modest speedup (2-4× on 8 cores) but the only well-cited parallel-BK approach. **Refs.** Strandmark-Kahl-2010 *CVPR-2010*; Liu-Sun-2010-PAMI-32:2049 alternative parallel-graph-cut.

---

## 2. Connective tissue + cross-package blockers

**Substrate-blocker-1 (HARD)** `graph.IntAdjacency` + map-based capacity (PRESENT in `graph/flow.go`). Every C-primitive uses this type. ZERO BLOCKERS.

**Substrate-blocker-2 (SOFT)** `linalg.SparseMatrix` (097-flagged ABSENT). Gates C20 TRW-S full-grid-graph + C24 LP-MRF (sparse-LP-solver) at large-V scale. 252 also flags this. C20 ships at quality-bar without sparse-matrix; only the 1M-pixel grid variant requires it.

**Substrate-blocker-3 (SOFT)** `optim/lp` Simplex / IPM (097-flagged ABSENT). Gates C24 LP-MRF + C25 branch-and-bound. AD³-C26 (ADMM-based) sidesteps this — ships against existing `optim/proximal.Admm`.

**Substrate-blocker-4 (NONE)** `optim/proximal.{Admm, Fbs}` (PRESENT). C26 AD³ + C29 mean-field consume directly.

**Substrate-blocker-5 (NONE)** `prob.NormalPDF` (PRESENT). C8 grid-graph data-term and C-segmentation-applications consume.

**Substrate-blocker-6 (NONE)** Container-of-pairwise-terms `[]PairwiseTerm{i, j int; theta00, theta01, theta10, theta11 float64}`. Pure value-type, ships in `graph/cuts/types.go` ~30 LOC.

**Total upstream-substrate dependency** (not counting `graph.MaxFlow` which is PRESENT): ~0 LOC of NEW code in non-`graph/cuts/` paths is required for the 24 ship-against-PRESENT-substrate primitives (C1-C19, C21-C23, C26-C30). The 6 LP-dependent primitives (C24, C25) require `optim/lp` (097); C20 TRW-S at large-V requires `linalg/sparse` (097).

**Cheapest-no-blocker subset:** **C1 Dinic + C2 BK + C3 push-relabel + C7 min-cut-extract + C8 grid-graph + C10 regularity + C11 QPBO + C15 α-expansion ~1,100 LOC**. Covers 80% of segmentation-graph-cut consumer demand without ANY 097/244 substrate gating.

**Recommended PR sequence:**

- **PR-A (Tier-0 max-flow algorithms ~860 LOC, 2 weeks)** C1 Dinic + C2 BK + C3 push-relabel + C4 highest-label + C5 IBFS + C6 dynamic-cut. PR-A is the SINGULAR-MOAT — ships completely against PRESENT substrate, validates against `graph.MaxFlow` EK as oracle on small instances.
- **PR-B (Tier-1 min-cut substrate ~280 LOC, 3 days)** C7 + C8 + C9. The grid-graph-builder (C8) is the API that consumers actually call.
- **PR-C (Tier-2 pseudo-boolean ~520 LOC, 2 weeks)** C10 regularity-checker + C11 QPBO + C12 QPBO-I + C13 QPBO-P + C14 Ishikawa. PR-C is the SINGULAR-DIFFERENTIATOR vs every-other-Go-graph-library.
- **PR-D (Tier-3 multi-label move-making ~520 LOC, 2 weeks)** C15 α-expansion + C16 α-β-swap + C17 fusion-move + C18 range-move + C19 jump-move.
- **PR-E (Tier-4 LP-relaxation message-passing ~720 LOC, 3 weeks)** C20 TRW-S + C21 loopy-BP + C22 junction-tree + C23 MPLP.
- **PR-F (Tier-5 LP / ILP / branch-and-bound ~360 LOC, 2 weeks)** C24 LP-MRF + C25 branch-and-bound + C26 AD³. Depends on `optim/lp` (097).
- **PR-G (Tier-6 sampling + variational ~380 LOC, 1.5 weeks)** C27 Gibbs + C28 Swendsen-Wang + C29 mean-field + C30 parallel-BK.

Total `graph/cuts/` PR-A through PR-G: ~3,640 LOC, ~12-14 engineer-weeks. Substantially smaller than 252's 3,800-LOC `image/segment/` because graph-cuts share more substrate (one max-flow underlies everything) than segmentation does (every segmentation algorithm has its own structure).

---

## 3. Five R-MUTUAL-CROSS-VALIDATION 3/3 pins this slot enables

**Pin 1 — Max-flow value on synthetic 32×32 grid-graph with analytical-known min-cut.** Three paths to identical max-flow value to 1e-12 (integer-capacity exactness):
- C1 Dinic blocking-flow
- C2 Boykov-Kolmogorov augmenting-path-with-tree-reuse
- `graph.MaxFlow` Edmonds-Karp BFS-Ford-Fulkerson (PRESENT)

All three must agree exactly. Saturates 3/3, AND simultaneously pins relative-asymptotic-performance (BK 5-10× faster than EK on grid-graphs as paper-claim → benchmark verification). EXTENSION: add C3 push-relabel as fourth path → 4/3-witness for the same fact (over-saturation, valuable for algorithm-correctness-confidence).

**Pin 2 — Submodular-pseudo-boolean global minimum.** Three paths to the same labelling on a small (12-variable) submodular F² instance:
- C10 + C2 (regularity-check passes → reduce-to-graph-cut → BK min-cut)
- C11 QPBO with `?`-set empty (provably gives global-min for submodular)
- Brute-force enumeration over 2¹² = 4096 labellings (oracle)

All three must give identical labelling. Saturates 3/3 + a "negative-pin" against attempting QPBO on a NON-submodular instance and getting `?` set non-empty (proves QPBO actually does the right thing in both regimes).

**Pin 3 — α-expansion two-label binary-segmentation matches single graph-cut.** Three paths to the same labelling on a 2-label energy-minimisation instance:
- C2 BK direct binary-cut
- C15 α-expansion with K=2 (one outer-iteration suffices)
- C20 TRW-S on the same factor-graph (LP-relaxation tight on submodular → integer-recovery exact)

All three must give identical labelling AND identical primal-energy AND C20 must report `lowerBound = primalEnergy` (LP-tightness witness). Saturates 3/3 + a "tightness-pin" certifying TRW-S achieves the right LP-bound on submodular.

**Pin 4 — TRW-S monotone-dual-bound vs loopy-BP non-monotone.** Three paths on a synthetic non-submodular MRF:
- C20 TRW-S → returns monotone-non-decreasing lower-bound sequence (paper-property)
- C21 loopy-BP max-product → may oscillate or stop at suboptimal fixed-point
- C25 branch-and-bound + LP-relaxation → exact MAP (oracle for small instances)

C20's lower-bound at convergence must equal C25's MAP-energy on instances where LP is tight; on non-tight instances must satisfy `C20-lower ≤ C25-exact`. C21's primal-energy must be ≥ C20's primal-energy on most instances (TRW-S strictly stronger than BP on practical problems). Saturates 3/3 + a "monotone-property-pin" by checking the lower-bound sequence is non-decreasing across iterations.

**Pin 5 — Ishikawa convex-multi-label = α-expansion-on-K-labels.** Three paths on a 5-label energy with convex `g(|li − lj|) = (li − lj)²` pairwise:
- C14 Ishikawa single-graph-cut on K·V layered graph (provably exact)
- C15 α-expansion with K=5 (provably 2c-approximation; on convex pairwise, c = max ratio = 1 + ... → near-exact)
- C20 TRW-S converged (LP-tight on truncated-convex by Sontag-Jaakkola → exact)

All three must agree on labelling within 1% energy-gap (α-expansion is approximation, not exact, so allow tiny gap; Ishikawa and TRW-S exact). Saturates 3/3 + an exactness-vs-approximation negative-pin.

---

## 4. Touchpoints with other agents

- **081 (graph-numerics) F5:** Edmonds-Karp float-residual-noise hazard. Slot 254 is the algorithmic-replacement track — once C2 BK ships, 081's F5 mitigation becomes "use C2 BK on float capacities, retain `graph.MaxFlow` EK for cross-validation only on integer instances".
- **252 (image-segmentation) S14, S15, S20:** STRICT DOWNSTREAM. 252-S14 BoykovKolmogorovMaxFlow ~200 LOC and S15 alpha-expansion ~100 LOC are *re-exports* of 254-C2 + 254-C15 with image-shaped wrappers. Recommend: 252 inlines the algorithms as I/O-shaped-thin-wrappers (~80 LOC instead of 300 LOC) once 254-PR-A + 254-PR-D land.
- **253 (active-contours) A20, A21, A22:** STRICT-INDIRECT-DOWNSTREAM. A21 Local-CV uses internal BK-cuts when the local-statistic is binarised; A22 multiphase-CV can alternatively use 254-C15 α-expansion for K-phase-discrete instead of log₂(K)-level-sets. Cross-link defer to v1.1+.
- **251 (shape-opt) T7:** combinatorial-shape-optimisation via inflation = max-flow on dual-graph. Strict consumer of 254-C2.
- **215 (compressed-sensing) T-binary-CS:** binary compressed-sensing recovery via QPBO when the recovery-objective is non-submodular pseudo-boolean. Strict consumer of 254-C11 QPBO.
- **097 (linalg-missing):** `optim/lp` (Simplex + IPM) and `linalg/sparse` are PREREQUISITES for 254-C20-large-grid + 254-C24 + 254-C25. Soft block (Tier-4 + Tier-5).
- **244 (pde-solvers) D12:** Conjugate-Gradient is candidate substrate for 254-C20 TRW-S inner-loop; soft cross-link.
- **117 (prob-missing):** PRNG primitives needed for 254-C27 Gibbs + 254-C28 Swendsen-Wang (already present-enough via Go-stdlib `math/rand`).
- **142 (topology-missing):** discrete-Morse-theory + persistent-homology share min-cut substrate with 254-C7 (basin-cuts produce filtered-complexes); cross-link defer.
- **165 (synergy-graph-prob):** the graph-MRF-as-Bayesian-network identification is the hub between this slot and `prob/`. Specifically, 254-C21 loopy-BP and 254-C22 junction-tree are *the same algorithms* as Bayesian-network-inference; the right placement may be a future `prob/graphical/` slot consuming 254-C21+C22 via interface.
- **A future graph-cuts-isolation review (none currently scheduled):** with 252 + 253 + 254 all consuming `graph/cuts/`, recommend opening `graph-cuts-numerics`, `graph-cuts-missing`, `graph-cuts-sota`, `graph-cuts-api`, `graph-cuts-perf` slots in a future overnight grid (252 also flagged this for `image/segment/`).

---

## 5. Singular load-bearing recommendation

**Ship PR-A (Tier-0 max-flow algorithms) FIRST as the SINGULAR-MOAT ~860 LOC, 2 weeks.** C2 Boykov-Kolmogorov is the SINGLE most-cited graph-cut algorithm in computer-vision (>15,000 citations), and a zero-dep cross-language byte-identical Go implementation exists in NO library worldwide — Vladimir Kolmogorov's reference C++ `maxflow-3.04` is GPL-licensed (cannot be MIT-wrapped, must be from-paper-reimplemented), Boost's `boost::graph::boykov_kolmogorov_max_flow` is C++-only and not bit-identical-portable, OpenCV's `cv::GraphCut` is wrapped behind `cv::grabCut` and not exposed as a standalone primitive. Having C2 in zero-dep Go with golden-file cross-language validation is a **unique reality contribution**.

**Then ship PR-C (Tier-2 pseudo-boolean) ~520 LOC, 2 weeks** because C11 QPBO is the SINGULAR-DIFFERENTIATOR vs every-other-Go-graph-library — no Go max-flow library handles non-submodular pseudo-boolean optimisation. The roof-duality theorem (Boros-Hammer-2002) is the most beautiful result in pseudo-boolean optimisation and is small enough (~220 LOC) to be a flagship pedagogical demonstration of how min-cut generalises beyond submodular.

**Then ship PR-D (Tier-3 multi-label move-making) ~520 LOC, 2 weeks** because C15 α-expansion is FOUNDATIONAL — every multi-label segmentation, stereo-matching, optical-flow, photo-stitching paper since 2001 uses α-expansion as the outer-loop. PR-A + PR-C + PR-D = the GrabCut-2004 / Photo-Magnetic-Lasso / Stereo-Pipeline trio at 1,900 LOC.

**Then ship PR-E (Tier-4 LP-relaxation TRW-S) ~720 LOC, 3 weeks** as the 2024-FRONTIER. C20 TRW-S is the production-default LP-MRF solver in OpenGM-2.0 / pgmpy / opengm-Python; monotone-lower-bound-property is the production-quality differentiator. PR-A through PR-E covers ~80% of consumer demand.

**Defer PR-F (LP / ILP / branch-and-bound) until 097-`optim/lp` lands** — the dependency is hard.

**Defer PR-G (sampling + parallel)** as v1.1 polish.

**Avoid scoping: differentiable-graph-cut / soft-graph-cut / continuous-relaxation-ML.** These are deep-learning primitives (Hochbaum-2008 pseudo-flow has a continuous-relaxation, but the modern continuous-graph-cut is the differentiable-CRF-Krähenbühl-Koltun-2011-NIPS / DeepLab-CRF-as-RNN-Zheng-Jayasumana-Romera-Paredes-Vineet-Su-Du-Huang-Torr-2015 family — these are aicore-territory, not reality-territory).

**Avoid scoping: graph-neural-network-message-passing.** GNN message-passing is structurally identical to loopy-BP but the parameterisation is learned-tensors, not analytic-potentials → aicore-territory.

**Avoid scoping: Goldberg-Rao-1998 binary-blocking-flow O(min(V^{2/3}, E^{1/2}) · E · log(V²/E) · log(U))**. State-of-the-art for very-dense graphs but rarely used in vision (where graphs are sparse-grid). Defer.

**Final precision-hazards:**
- **(a)** BK-2004 on float capacities — same termination-noise hazard as `graph.MaxFlow` EK (081-F5 documents it). Mitigation: same `EPS_RATIO` parameter or integer-capacity-only contract. The BK paper itself recommends integer-capacity-quantisation when source data is float (multiply by 1e6, round to int64).
- **(b)** QPBO on degenerate energies (all pairwise-terms zero, only unary) — produces identical labelling to "argmin unary independently per variable" but the autarky-property still holds. Test case mandatory.
- **(c)** α-expansion on non-metric pairwise — silently produces wrong-answer if regularity (C10) fails on the binary subproblem. C15 must call C10 internally and either (i) fall through to QPBO + accept partial-labelling, or (ii) error out. Recommend (i) with a warning.
- **(d)** TRW-S on grid-graphs requires choosing a tree-decomposition — different choices yield different schedules. Standard: row-trees + column-trees. Pin this in cross-language reproducibility.
- **(e)** Junction-tree on grid-graph has tree-width = min(W, H), so exponential-cost on 100×100 grid → unusable. C22 documentation must enumerate this regime.
- **(f)** Push-relabel global-relabel period — common values 0.5·V or `V·log(V)`. Pin 0.5·V for cross-language reproducibility (Cherkassky-Goldberg-1997 default).
- **(g)** Multi-way-cut C9 is approximation, not exact — the `(2 − 2/k)` factor must be in the docstring; consumers expecting exact-MWC must use ILP path (not currently supported).
- **(h)** Parallel-BK C30 is non-deterministic — random thread-scheduling produces different residual-graphs at intermediate iterations though final max-flow is identical. Document; cross-language-reproducibility for parallel-BK requires fixed-thread-count + fixed-decomposition.
- **(i)** QPBO-Probing C13 may not terminate on adversarial inputs (test-labelling can recurse arbitrarily deep). Add `maxIter` parameter; document.
- **(j)** Dynamic-graph-cut C6 incorrectness if the residual-graph from previous frame was not exactly saturated (e.g., if previous BK was terminated early). C6 must validate or error.

**Headline:** Thirty graph-cut + energy-minimisation primitives close the entire 1962-2011 graph-cut canon (Dinic-1970 / Goldberg-Tarjan-1988 / Geman-Geman-1984 / Pearl-1988 / Lauritzen-Spiegelhalter-1988 / Ishikawa-2003 / Boykov-Veksler-Zabih-2001 α-expansion + α-β-swap / Kolmogorov-Zabih-2004 regularity / Boykov-Kolmogorov-2004 augmenting-path / Wainwright-Jaakkola-Willsky-2005 TRW / Kolmogorov-2006 TRW-S / Boros-Hammer-2002 + Rother-Kolmogorov-Lempitsky-Szummer-2007 QPBO + QPBO-I + QPBO-P / Kohli-Torr-2007 dynamic-cut / Globerson-Jaakkola-2008 MPLP / Veksler-2007/2012 range/jump-move / Strandmark-Kahl-2010 parallel-BK / Lempitsky-Rother-Roth-Blake-2010 fusion-move / Goldberg-Hed-Kaplan-Tarjan-Werneck-2011 IBFS / Martins-Figueiredo-Aguiar-Smith-2011 AD³ / Cherkassky-Goldberg-1997 highest-label-PR / Dahlhaus-Johnson-Papadimitriou-Seymour-Yannakakis-1992 multi-way-cut / Swendsen-Wang-1987 cluster-MC / Yedidia-Freeman-Weiss-2005 Bethe) in ~3,640 LOC of pure synthesis on top of `graph.MaxFlow`-PRESENT (oracle) + `optim/proximal`-PRESENT (ADMM substrate); cheapest-no-blocker subset C1+C2+C3+C7+C8+C10+C11+C15 ~1,100 LOC; foundational keystone C2 BK ~280 LOC; singular-moat C2 + C11 + C15 ~640 LOC; 2024-frontier C20 TRW-S + C26 AD³ ~380 LOC; pedagogical C1 Dinic + C3 push-relabel + C2 BK ~540 LOC. Five R-MUTUAL-CROSS-VALIDATION 3/3 pins enabled (Dinic↔BK↔EK max-flow, regularity↔QPBO↔brute-force submodular, BK↔α-expansion↔TRW-S binary-cut equivalence, TRW-S↔loopy-BP↔branch-bound LP-tightness, Ishikawa↔α-expansion↔TRW-S convex-multi-label). Strict-upstream of 252-S14/S15 segmentation-graph-cut tier and 253-A21 Local-CV inner-loop and 251-shape-opt T7 combinatorial-inflation; strict-downstream of `graph.MaxFlow`-PRESENT (oracle); subsumes 081-F5 mitigation track for float-capacity termination noise. Recommended placement NEW sub-package `graph/cuts/` under existing `graph/` package — same "consumer-shaped sub-package, not in primitive-supplier package" precedent (151/153/156/157/158/247/250/252/253). PR-A SINGULAR-MOAT ~860 LOC ships first against PRESENT substrate with ZERO upstream blockers.

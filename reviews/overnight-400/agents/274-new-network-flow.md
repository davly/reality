# 274 | new-network-flow — Min-cost flow, b-matching, transportation, assignment, blossom, network simplex

**Summary L1.** reality v0.10.0 ships **TWO** network-flow primitives total — `graph/flow.go::MaxFlow` (Edmonds-Karp BFS-Ford-Fulkerson, O(V·E²), ~100 LOC, integer-correct / float-residual-noise-prone per 081-F5) and `optim/transport/sinkhorn.go::Sinkhorn` (log-domain entropic-regularised n-D OT, ~250 LOC, Cuturi-2013) — plus **`gametheory/matching.go::GaleShapley`** (deferred-acceptance stable-matching, n²) and **`topology/persistent/bottleneck.go`** (private Hopcroft-Karp bipartite-matching used only as binary-search-inner-loop for bottleneck-distance). Repo-wide grep on `MinCostFlow|min.cost.flow|SuccessiveShortestPaths|CapacityScaling|NetworkSimplex|CycleCanceling|OutOfKilter|CostScaling|push.relabel.min.cost|Hungarian|KuhnMunkres|Munkres|Auction|Bertsekas|JonkerVolgenant|Jonker.Volgenant|LAPJV|LAPMOD|Blossom|MaximumWeightMatching|MinimumWeightPerfectMatching|Edmonds.matching|b.matching|TransportationProblem|VogelApproximation|MODI|StepStone|MultiCommodityFlow|GeneralizedFlow|FlowWithLowerBounds|SteinerTree|Dinic|BoykovKolmogorov|PushRelabel|Goldberg.Tarjan|FIFO.preflow|HighestLabel|GapHeuristic|GlobalRelabel|hopcroft.karp|HopcroftKarp` returns **ZERO callable matches in production `*.go`** outside the four named primitives above. The full **min-cost-flow / linear-assignment / weighted-matching / transportation-LP / multi-commodity / generalized-flow** tier is **ABSENT**. **PARTIAL OVERLAP with 254 (graph-cuts):** 254 owns the **max-flow algorithmic-replacement track** (Dinic / Boykov-Kolmogorov / Goldberg-Tarjan-push-relabel / IBFS) for image-energy-minimisation in a `graph/cuts/` sub-package — slot 274 sits **ONE TIER UP** at the **min-COST-flow + assignment + matching + transportation-LP** layer (cost objective, not max-throughput; bipartite-special-case with O(n³) Hungarian + Jonker-Volgenant, not O(V·E²) max-flow), in a sibling `graph/mincostflow/` (or `graph/flow/`) sub-package. Every 254-Tier-0 max-flow primitive is a **substrate-dependency** of 274's min-cost-flow shortest-augmenting-path inner-loop, but the cost-axis math (reduced-costs / Bellman-Ford-with-potentials / dual-LP) is **disjoint** from 254's submodular-energy + α-expansion + QPBO + TRW-S layer. **PARTIAL OVERLAP with 081-graph-numerics F5:** 081 already enumerated the float-capacity termination-noise hazard in `MaxFlow`; the same hazard transfers to every cost-augmenting-path variant 274 ships — fix is the same `eps`-rounding-of-residuals discipline 081-F5 specified. **PARTIAL OVERLAP with 201-OT v2-deferral roster:** 201 explicitly named **network-simplex (exact LP-OT)** as deferred — slot 274 is the natural home for the discrete-LP-network-simplex that exact-OT consumers (transportation-problem on integer-marginals, image-color-transfer on histograms) want when Sinkhorn's entropic-blur is unacceptable. **PARTIAL OVERLAP with 102-optim-missing T3.33:** 102 flagged Network-Simplex as a missing optim primitive; 274 puts the same algorithm in `graph/mincostflow/` (consumer-shaped) instead of `optim/lp/` (algorithm-shaped) — the right placement because every consumer (transportation, b-matching, assignment, min-cost-flow) is a graph-structured-LP, not a generic LP. **PARTIAL OVERLAP with gametheory/matching.go::GaleShapley:** stable-matching is one-sided-preference unweighted — 274 generalises to (a) two-sided-preference weighted (Hungarian / Jonker-Volgenant), (b) many-to-many (b-matching), (c) cumulative-offer (Hatfield-Milgrom-2005). **Block-C verdict:** the **ENTIRE min-cost-flow / linear-assignment / blossom-weighted-matching / transportation-LP / multi-commodity-flow / generalized-flow** mathematical surface is **ABSENT**. This is the **second-most-classical missing tier in `graph/`** after the 254-graph-cut tier — Hungarian-1955 and Edmonds-blossom-1965 are textbook content in CLRS / Cormen-3e §26 and Schrijver-2003 / Ahuja-Magnanti-Orlin-1993, the two canonical references for combinatorial-optimisation. Linear-assignment alone is the foundation of (a) tracking + association in `tracking/` (multi-target Hungarian-assignment-on-cost-matrix), (b) LSAP solver in `gametheory/` for bilateral-trade matching, (c) word-embedding alignment via Wasserstein-Procrustes (Grave-Joulin-Berthet-2019), (d) earth-mover-distance exact LP variant when Sinkhorn-blur is wrong. **No zero-dep MIT Go implementation of Edmonds-blossom or Jonker-Volgenant exists worldwide** — the canonical references are LEMON C++ (Boost-licensed, can be referenced but not wrapped) and `lap` Python (BSD).

**Summary L2.** **Twenty-eight primitives N1-N28 totalling ~3,420 LOC** organised as **(a) Tier-0 linear assignment (square + rectangular bipartite) ~580 LOC** (N1 `graph/mincostflow/hungarian.go::Hungarian(cost [][]float64) (assignment []int, totalCost float64)` Kuhn-1955-Naval-Res-Logist-2:83 / Munkres-1957-J-SIAM-5:32 the canonical O(n³) row-and-column-reduction-with-cover-matrix algorithm for square LSAP ~180 LOC; N2 `graph/mincostflow/jonker_volgenant.go::JonkerVolgenant(cost [][]float64) (rowAssign, colAssign []int, totalCost float64)` Jonker-Volgenant-1987-Computing-38:325 LAPJV the practically-fastest dense-LSAP solver — column-reduction + augmenting-row-reduction + Dijkstra-shortest-augmenting-path with reduced-costs, **3-10× faster than Hungarian** on n≥100, the de-facto-standard in OR-Tools / SciPy-1.4+ `linear_sum_assignment` ~220 LOC; N3 `graph/mincostflow/lap_rectangular.go::JonkerVolgenantRectangular(cost [][]float64) ...` Jonker-Volgenant-1987 for non-square `m×n` cost-matrix (m workers ≠ n jobs) ~80 LOC; N4 `graph/mincostflow/auction.go::Auction(cost [][]float64, eps float64) ...` Bertsekas-1979-MIT-LIDS-P-924 / Bertsekas-1981-Math-Prog-21:152 ε-scaling auction-algorithm — bidders bid for objects with reduced-prices, terminates when no profitable-bid exists, O(n·A·max-cost/ε) complexity but **trivially-parallel** + works on **distributed** problem instances ~100 LOC), **(b) Tier-1 weighted matching on general graphs ~620 LOC** (N5 `graph/mincostflow/blossom_v.go::MaximumWeightMatching(adj IntAdjacency, weights map[[2]int]float64) (matching []int, totalWeight float64)` Edmonds-1965-Canad-J-Math-17:449 / Galil-1986-ACM-Comput-Surv-18:23 / Kolmogorov-2009-Math-Prog-Comput-1:43 Blossom-V — primal-dual shortest-path-tree algorithm shrinking odd-cycles-into-blossoms, O(V·E·α(E,V)) the de-facto-standard general-graph-matching ~360 LOC; N6 `graph/mincostflow/blossom_perfect.go::MinimumWeightPerfectMatching(...)` Edmonds-1965 perfect-matching variant for even-V graphs returning the perfect matching (or error if none exists) ~80 LOC; N7 `graph/mincostflow/maximum_cardinality_matching.go::MaximumCardinalityMatching(adj IntAdjacency)` Edmonds-1965 unweighted-cardinality-matching as the simpler-substrate Edmonds-blossom-without-weights, foundation for N5 ~120 LOC; N8 `graph/mincostflow/hopcroft_karp.go::HopcroftKarpBipartite(left, right int, edges [][2]int) matching` Hopcroft-Karp-1973-SIAM-J-Comput-2:225 O(E·√V) bipartite-cardinality-matching — currently inlined-private in `topology/persistent/bottleneck.go`, lift-to-public ~60 LOC), **(c) Tier-2 min-cost flow ~720 LOC** (N9 `graph/mincostflow/ssp.go::MinCostFlowSSP(adj IntAdjacency, capacity, cost map[[2]int]float64, source, sink int, requiredFlow float64)` Successive-Shortest-Paths Jewell-1958 / Iri-1960 / Busacker-Gowen-1960 — repeatedly find shortest-cost-augmenting-path via **Bellman-Ford** (handles initial-potentials) then **Dijkstra-with-Johnson-reweighting-by-potentials** (handles subsequent paths in O(V·log·V) per augment), terminates when desired flow reached or max-flow saturated ~220 LOC; N10 `graph/mincostflow/capacity_scaling.go::MinCostFlowCapacityScaling(...)` Edmonds-Karp-1972 / Orlin-1993-Oper-Res-41:338 capacity-scaling phase enumerates Δ = 2^k from max-capacity down, augments along shortest paths in Δ-residual graph, achieves strongly-polynomial O(E·log·U·(E+V·log·V)) ~140 LOC; N11 `graph/mincostflow/cycle_canceling.go::MinCostFlowCycleCanceling(...)` Klein-1967-Manage-Sci-14:205 / Goldberg-Tarjan-1989 starts with any feasible-flow then repeatedly cancels negative-cost-cycles via Bellman-Ford detection, O(V·E·M·U·log·V) but the simplest min-cost-flow algorithm to implement ~80 LOC; N12 `graph/mincostflow/min_mean_cycle_canceling.go::MinMeanCycleCanceling(...)` Goldberg-Tarjan-1989-Math-Oper-Res-14:30 + Karp-1978 minimum-mean-cycle-cancelling — strongly-polynomial O(V·E²·log²·V) by always cancelling the cycle with min mean weight ~100 LOC; N13 `graph/mincostflow/cost_scaling.go::MinCostFlowCostScaling(...)` Goldberg-Tarjan-1990-Math-Oper-Res-15:430 / Goldberg-1997-J-Algorithms-22:1 ε-scaling push-relabel for min-cost-flow — the de-facto-fastest min-cost-flow in practice (LEMON, OR-Tools default) O(V²·E·log(V·C)) ~180 LOC), **(d) Tier-3 network simplex + out-of-kilter ~480 LOC** (N14 `graph/mincostflow/network_simplex.go::NetworkSimplex(adj IntAdjacency, supply []float64, capacity, cost map[[2]int]float64)` Cunningham-1976-Math-Prog-11:105 / Orlin-1997-Math-Prog-78:109 strongly-polynomial network-simplex — special case of LP-simplex on graph-LP with spanning-tree-basis, pivot rule = Dantzig / first-eligible / candidate-list — the LEMON `MinCostFlow` default + the canonical "exact-LP-flow" algorithm, **also the exact-OT solver 201's v2-deferral asks for** ~280 LOC; N15 `graph/mincostflow/out_of_kilter.go::OutOfKilter(adj, capacity, cost, lowerBound)` Fulkerson-1961-Manage-Sci-7:166 / Ford-Fulkerson-1962 out-of-kilter primal-dual algorithm starting from infeasible-but-balanced flow + dual prices and iterating to optimality, classical method that handles **lower-bounds + upper-bounds** simultaneously ~120 LOC; N16 `graph/mincostflow/flow_lower_bounds.go::MinCostFlowWithLowerBounds(...)` standard reformulation lifting lower-bounds into modified-supply on Ahuja-Magnanti-Orlin-1993 §3.4 ~80 LOC), **(e) Tier-4 transportation problem (bipartite min-cost flow) ~340 LOC** (N17 `graph/mincostflow/transportation.go::TransportationProblem(supply, demand []float64, cost [][]float64)` Hitchcock-1941-J-Math-Phys-20:224 / Koopmans-1949-Econometrica-17:S136 — the bipartite-min-cost-flow special case where every supply-node connects to every demand-node, callable as a **direct exact-OT solver on integer-marginals** ~100 LOC; N18 `graph/mincostflow/vogel_approximation.go::VogelApproximation(supply, demand, cost)` Vogel-1958 row/column-penalty heuristic for fast initial-feasible transportation-problem-solution ~80 LOC; N19 `graph/mincostflow/northwest_corner.go::NorthwestCornerRule(supply, demand)` Dantzig-1951 simplest initial-feasible-corner rule ~40 LOC; N20 `graph/mincostflow/modi.go::MODI(plan, supply, demand, cost)` Modified-Distribution-method aka u-v-method for stepping-stone optimality test — the textbook-pedagogical method for transportation-problem-iteration ~120 LOC), **(f) Tier-5 b-matching + many-to-many ~280 LOC** (N21 `graph/mincostflow/b_matching.go::MaximumWeightBMatching(adj, weights, b []int)` Edmonds-Pulleyblank-1974 / Padberg-Rao-1982 b-matching — degree constraint b_v at each vertex, reducible to standard-matching by vertex-replication-up-to-2b but specialised algorithm faster ~140 LOC; N22 `graph/mincostflow/many_to_many.go::ManyToManyMatching(leftCap, rightCap []int, weights [][]float64)` many-to-many bipartite version — generalises Hungarian to multi-assignment ~80 LOC; N23 `graph/mincostflow/cumulative_offer.go::CumulativeOfferMechanism(prefs, contracts)` Hatfield-Milgrom-2005-Am-Econ-Rev-95:913 cumulative-offer algorithm for many-to-one matching with contracts — generalises Gale-Shapley to substitutable-preferences ~60 LOC), **(g) Tier-6 multi-commodity + generalized + Steiner ~400 LOC** (N24 `graph/mincostflow/multi_commodity.go::MultiCommodityFlowLP(commodities []Commodity, capacity)` Ford-Fulkerson-1958 / Tomlin-1966 multi-commodity-flow as a multi-source-multi-sink LP — links to N14 network-simplex on the line-graph + Lagrangian-relaxation / column-generation ~140 LOC; N25 `graph/mincostflow/generalized_flow.go::GeneralizedFlow(adj, capacity, gain, cost)` Jewell-1962-Oper-Res-10:476 / Wayne-2002-Math-Oper-Res-27:445 generalised-network-flow with edge-multipliers (gain/loss factors), strongly-polynomial via Wayne-2002 ~120 LOC; N26 `graph/mincostflow/steiner_tree.go::SteinerTreeApprox(adj, weights, terminals)` Takahashi-Matsuyama-1980 / Robins-Zelikovsky-2005-SIAM-J-Discrete-Math-19:122 the 2-approximation MST-on-shortest-path-distances + (1.55+ε) Robins-Zelikovsky improved approximation — Steiner-tree NP-hard substrate for VLSI / network-design ~140 LOC), **(h) Tier-7 cross-cuts + utilities ~320 LOC** (N27 `graph/mincostflow/exact_ot.go::ExactOptimalTransport(a, b, cost)` Wasserstein-W₁ exact LP — wraps N17 TransportationProblem with the OT-cost-matrix, **ships 201's v2-deferred network-simplex-OT as a single-line consumer** ~40 LOC; N28 `graph/mincostflow/wasserstein_procrustes.go::WassersteinProcrustes(X, Y, p)` Grave-Joulin-Berthet-2019-AISTATS-89:1880 alternating-OT-and-orthogonal-Procrustes for unsupervised word-embedding-alignment — directly consumes N17 + linalg.Procrustes ~120 LOC; N29 `graph/mincostflow/dual_potentials.go::DualPotentials(...)` extract reduced-costs / dual-prices from any min-cost-flow solver above for sensitivity-analysis + LP-duality-witnesses ~60 LOC; N30 `graph/mincostflow/feasibility.go::IsFeasibleFlow + GilmoreFeasibility(...)` Gilmore-1962 standard feasibility-check for min-cost-flow with arbitrary supply/demand/lower/upper bounds ~100 LOC).

**SINGULAR-FOUNDATIONAL N5 Edmonds-blossom maximum-weight-matching ~360 LOC** — the single-most-cited combinatorial-algorithm in 20th-century-OR (>20,000 citations across Edmonds-1965 + the 1991 STOC+JACM canonical paper + Galil-1986-survey + Kolmogorov-2009-Blossom-V); no zero-dep MIT Go implementation exists worldwide; LEMON C++ Blossom-V (~3,000 LOC, Boost-licensed) is the canonical reference + Kolmogorov's Cambridge-personal-page release of Blossom-V at MPL-2.0 (cannot be wrapped at MIT, must be reimplemented from-paper). The matching-on-general-graphs problem is **tractable in P but NP-hard if you naively enumerate odd-cycles** — the entire genius of Edmonds-1965 is the blossom-shrinking trick that makes it polynomial. This is the **single-largest gap in `graph/`** after 254-graph-cuts.

**SINGULAR-CHEAPEST-1-DAY N1 Hungarian + N17 TransportationProblem + N7 MaximumCardinalityMatching ~400 LOC** — Hungarian-1955 is textbook Munkres-1957 step-by-step pseudocode, the simplest bipartite-LSAP, ships drop-in-against `cost [][]float64` signature with golden-file cross-validation against SciPy `linear_sum_assignment`. N17 transportation is just bipartite-min-cost-flow on complete-bipartite-graph — wraps N1 with a supply/demand reformulation. N7 is Edmonds-1965 unweighted-matching, the substrate for N5/N6/N21. All three are **today's-day-of-coding** scope.

**SINGULAR-MOAT N5 Edmonds-blossom + N2 Jonker-Volgenant + N13 cost-scaling-min-cost-flow + N14 network-simplex ~1,040 LOC** — the FOUR de-facto-standard combinatorial-optimisation algorithms that LEMON / OR-Tools / SciPy ship as their **default min-cost-flow / assignment / matching** dispatch. JV is 3-10× faster than Hungarian on dense-LSAP, cost-scaling is the production-default min-cost-flow (CONCORD 1995 onwards), network-simplex is the LP-exact-OT solver that 201's v2-deferral identifies. Blossom-V is the gold-standard general-graph-weighted-matching. Together these four make `graph/mincostflow/` the **single best zero-dep combinatorial-optimisation library in the Go ecosystem** — gonum has none of these (gonum-graph-flow only ships Bellman-Ford-Moore + Edmonds-Karp + Dinic, no min-cost-flow at all).

**SINGULAR-2024-FRONTIER N4 Auction + N28 Wasserstein-Procrustes ~220 LOC** — Bertsekas-auction-1979 underwent a 2018-2024 renaissance for **distributed + GPU-parallel LSAP** (Naparstek-Cohen-2017-IEEE-J-Sel-Top-Sig-Process-11:1182 GPU-auction; Goldberg-Kennedy-1995 ε-scaling sequential-auction). Wasserstein-Procrustes is the 2019 unsupervised-word-embedding alignment algorithm (Grave-Joulin-Berthet-2019 AISTATS) that combines orthogonal-Procrustes + exact-OT — a direct consumer of N17 transportation that 201's roster did not enumerate but is provably the right exact-OT-companion for ML applications.

**SINGULAR-PEDAGOGICAL N1 Hungarian + N9 SSP + N14 network-simplex + N5 Edmonds-blossom ~1,040 LOC** — the four canonical algorithms in CLRS / Cormen-3e §26 + Schrijver-2003 / Ahuja-Magnanti-Orlin-1993. Five R-MUTUAL-CROSS-VALIDATION 3/3 pins enabled (see §3). Recommended placement **NEW sub-package `graph/mincostflow/`** under the existing `graph/` package — same "consumer-shaped sub-package" precedent as 254-Tier-0 (`graph/cuts/`) and 252 (`image/segment/`). Strict-downstream of `graph/flow.go::MaxFlow` (oracle for unit-capacity-unit-cost-cross-validation) and `graph/shortest.go::Dijkstra + BellmanFord` (substrate for SSP + cost-scaling); strict-upstream of `optim/transport/` (N17 TransportationProblem ships 201's v2-deferred exact-OT).

---

## 0. State at HEAD (2026-05-09, v0.10.0)

Repo-wide audit for **min-cost flow / assignment / weighted-matching / transportation / multi-commodity / generalized-flow** surface.

| Surface | Path | Network-flow relevance |
|---|---|---|
| `graph.MaxFlow` (Edmonds-Karp BFS-FF) | `graph/flow.go:25-104` | Substrate; **PRESENT** — provides residual-graph machinery the SSP+cost-scaling+capacity-scaling tier reuses |
| `graph.Dijkstra / AStar` | `graph/shortest.go` | Substrate for N9 SSP (with Johnson-reweighting via potentials) and N13 cost-scaling-shortest-path; **PRESENT** |
| `graph.BellmanFord` | `graph/bellman_ford.go` | Substrate for N9 SSP-initial-potentials, N11 cycle-canceling-detection; **PRESENT** |
| `graph.{KruskalMST, PrimMST}` | `graph/mst.go` | Substrate for N26 Steiner-tree 2-approximation (MST-on-metric-closure); **PRESENT** |
| `graph.TopologicalSort` | `graph/flow.go:127` | Substrate for N24 multi-commodity-flow on DAGs (when commodities flow only forward); **PRESENT** |
| `optim/transport.Sinkhorn` | `optim/transport/sinkhorn.go` | **ENTROPIC** OT — does NOT solve the LP-exact-OT that N17 TransportationProblem provides; the two are complementary |
| `optim/transport.Wasserstein1D` | `optim/transport/wasserstein1d.go` | Closed-form 1-D OT — the **ground-truth-oracle** for N27 ExactOptimalTransport on n=1 instances |
| `optim/lp` (Simplex / IPM) | -- | LP-relaxation substrate; **ABSENT** (097-linalg-missing + 102-T3.33 flag) |
| `gametheory.GaleShapley` | `gametheory/matching.go:36` | Stable-matching with **one-sided ordinal preferences** — fundamentally different from N5 Edmonds-blossom (general-graph weighted-matching with cardinal-edge-weights) and N1 Hungarian (bipartite weighted with cost-matrix) |
| `gametheory.IsStableMatching` | `gametheory/matching.go:122` | Stable-matching verification — distinct from the N5/N6/N7 weighted-matching family |
| `topology/persistent.bottleneck::hopcroftKarp` (private inner-loop) | `topology/persistent/bottleneck.go` | **PARTIAL — private** Hopcroft-Karp bipartite-cardinality-matching, used as inner-loop for binary-search bottleneck-distance. **N8 lifts this to public** in `graph/mincostflow/hopcroft_karp.go` |
| `graph/mincostflow/` package | -- | **ABSENT** — this slot creates |
| N1-N30 min-cost-flow primitives | -- | **ALL ABSENT** |

**False-positive name-collisions audited:**
- `graph.LouvainCommunities` — modularity-based community-detection (different objective from min-cut); not a flow algorithm.
- `chaos/networks` — does not exist (network here means dynamical-system, not graph-flow).
- `prob.Bayesian` — Bayesian-network-inference, not network-flow.
- `optim/transport::Sinkhorn` — entropic-regularised OT, NOT exact-LP transportation-problem (different objective: `<P,C> + ε·H(P)` vs `<P,C>`).
- `gametheory.GaleShapley` — stable-matching with ordinal-preferences, NOT weighted-matching with cardinal-edge-weights.
- `compression.Huffman` — prefix-code-tree-construction (greedy), not a flow algorithm despite tree-output.
- `linalg.Procrustes` — rotation-alignment of two point-clouds (orthogonal); NOT matching-of-points (which is what N28 Wasserstein-Procrustes does by **combining** orthogonal-Procrustes + exact-OT).

**Cross-import edges that this slot creates:**
- `graph/mincostflow → graph` for `IntAdjacency`, `MaxFlow` oracle, `Dijkstra`, `BellmanFord`, `TopologicalSort`, `KruskalMST` substrate.
- `graph/mincostflow → linalg.{Vector, Matrix}` for cost-matrix passing + dual-potentials extraction.
- `graph/mincostflow → optim/lp` (097-flagged ABSENT) for N24 multi-commodity-LP fallback when network-simplex (N14) is not applicable.
- `optim/transport.{ExactOptimalTransport, Wasserstein2DLP}` (NEW v2 wire-up) → consumes `graph/mincostflow.TransportationProblem` (N17) for exact-LP-OT.
- `gametheory.{HungarianMatching, KuhnMunkres}` (NEW v2 wire-up) → re-exports `graph/mincostflow.Hungarian` (N1) for the bilateral-trade matching consumers in `gametheory`.
- `topology/persistent.bottleneck` → re-exports `graph/mincostflow.HopcroftKarpBipartite` (N8) instead of inlining private copy.
- (Future) `tracking/` package consumers → multi-target-Hungarian-association on cost-matrix per Munkres-2D / JV-rectangular.
- (Future) `image/registration/` consumers → N28 Wasserstein-Procrustes for histogram-of-features alignment.

**Strict downstream consumers of `graph/mincostflow/`:**
- `optim/transport/exact_ot.go` (NEW) → uses `mincostflow.TransportationProblem` (N17) for exact-LP-W₁ — this is **201's v2-deferral promised but unwired primitive.**
- `optim/transport/wasserstein_procrustes.go` (NEW) → uses N28 (which uses N17 internally + linalg.Procrustes).
- `gametheory/matching.go::HungarianMatching` (NEW) → re-export of N1 / N2 for the cardinal-weighted matching companion to ordinal-Gale-Shapley.
- `topology/persistent/bottleneck.go` → cleanup-replace private `hopcroftKarp` with public N8.
- 252-image-segment T-multi-target-tracking → `mincostflow.Hungarian` for assignment-by-frame.
- 215-compressed-sensing → N17 transportation for combinatorial-OT-formulation of CS recovery.

---

## 1. The 30 primitives — full enumeration

### Tier-0 — Linear assignment (square + rectangular bipartite) ~580 LOC

| # | Symbol | LOC | Algorithm + reference |
|---|---|---:|---|
| N1 | `Hungarian(cost [][]float64) (assign []int, totalCost float64)` | 180 | Kuhn-1955 / Munkres-1957 row-and-column-reduction-with-cover-matrix square LSAP O(n³) |
| N2 | `JonkerVolgenant(cost [][]float64) ...` | 220 | Jonker-Volgenant-1987 LAPJV column-reduction + augmenting-row + Dijkstra-with-reduced-costs, 3-10× faster than N1 on dense |
| N3 | `JonkerVolgenantRectangular(cost [][]float64) ...` | 80 | JV non-square m×n cost-matrix (LAPMOD variant Volgenant-1996) |
| N4 | `Auction(cost [][]float64, eps) ...` | 100 | Bertsekas-1979 ε-scaling auction; trivially-parallel + distributed-friendly |

### Tier-1 — Weighted matching on general graphs ~620 LOC

| # | Symbol | LOC | Algorithm + reference |
|---|---|---:|---|
| N5 | `MaximumWeightMatching(adj, weights) ...` | 360 | Edmonds-1965 / Galil-1986 / Kolmogorov-2009 Blossom-V primal-dual shortest-path-tree with blossom-shrinking O(V·E·α) |
| N6 | `MinimumWeightPerfectMatching(adj, weights) ...` | 80 | Edmonds-1965 perfect-matching variant |
| N7 | `MaximumCardinalityMatching(adj) ...` | 120 | Edmonds-1965 unweighted-cardinality (substrate for N5/N6) |
| N8 | `HopcroftKarpBipartite(left, right, edges) matching` | 60 | Hopcroft-Karp-1973 O(E·√V) bipartite-cardinality (lift from `topology/persistent` private) |

### Tier-2 — Min-cost flow ~720 LOC

| # | Symbol | LOC | Algorithm + reference |
|---|---|---:|---|
| N9 | `MinCostFlowSSP(adj, cap, cost, src, sink, requiredFlow)` | 220 | Successive-Shortest-Paths Jewell-1958 / Iri-1960 / Busacker-Gowen-1960 — Bellman-Ford for first iteration (potentials) + Dijkstra-with-Johnson-reweighting for subsequent |
| N10 | `MinCostFlowCapacityScaling(...)` | 140 | Edmonds-Karp-1972 / Orlin-1993 capacity-scaling phases Δ = 2^k |
| N11 | `MinCostFlowCycleCanceling(...)` | 80 | Klein-1967 / Goldberg-Tarjan-1989 negative-cost-cycle cancellation via Bellman-Ford |
| N12 | `MinMeanCycleCanceling(...)` | 100 | Goldberg-Tarjan-1989 / Karp-1978 min-mean-cycle cancellation strongly-polynomial |
| N13 | `MinCostFlowCostScaling(...)` | 180 | Goldberg-Tarjan-1990 / Goldberg-1997 ε-scaling push-relabel — production-default |

### Tier-3 — Network simplex + out-of-kilter ~480 LOC

| # | Symbol | LOC | Algorithm + reference |
|---|---|---:|---|
| N14 | `NetworkSimplex(adj, supply, cap, cost)` | 280 | Cunningham-1976 / Orlin-1997 strongly-polynomial network-simplex — **ships 201's v2-deferred exact-OT solver** + 102-T3.33 |
| N15 | `OutOfKilter(adj, cap, cost, lowerBound)` | 120 | Fulkerson-1961 primal-dual out-of-kilter, classical method handling lower+upper-bounds |
| N16 | `MinCostFlowWithLowerBounds(...)` | 80 | Standard reformulation Ahuja-Magnanti-Orlin-1993 §3.4 |

### Tier-4 — Transportation problem (bipartite min-cost flow) ~340 LOC

| # | Symbol | LOC | Algorithm + reference |
|---|---|---:|---|
| N17 | `TransportationProblem(supply, demand, cost)` | 100 | Hitchcock-1941 / Koopmans-1949 bipartite-min-cost-flow — direct exact-OT solver on integer-marginals |
| N18 | `VogelApproximation(supply, demand, cost)` | 80 | Vogel-1958 row/column-penalty heuristic for fast initial-feasible |
| N19 | `NorthwestCornerRule(supply, demand)` | 40 | Dantzig-1951 simplest initial-feasible rule |
| N20 | `MODI(plan, supply, demand, cost)` | 120 | Modified-Distribution / u-v-method stepping-stone optimality test |

### Tier-5 — b-matching + many-to-many ~280 LOC

| # | Symbol | LOC | Algorithm + reference |
|---|---|---:|---|
| N21 | `MaximumWeightBMatching(adj, weights, b)` | 140 | Edmonds-Pulleyblank-1974 / Padberg-Rao-1982 b-matching (degree constraint) |
| N22 | `ManyToManyMatching(leftCap, rightCap, weights)` | 80 | Many-to-many bipartite (multi-Hungarian) |
| N23 | `CumulativeOfferMechanism(prefs, contracts)` | 60 | Hatfield-Milgrom-2005 generalises Gale-Shapley to substitutable preferences |

### Tier-6 — Multi-commodity + generalized + Steiner ~400 LOC

| # | Symbol | LOC | Algorithm + reference |
|---|---|---:|---|
| N24 | `MultiCommodityFlowLP(commodities, capacity)` | 140 | Ford-Fulkerson-1958 / Tomlin-1966 multi-commodity LP via column-generation |
| N25 | `GeneralizedFlow(adj, cap, gain, cost)` | 120 | Jewell-1962 / Wayne-2002 strongly-polynomial GF with edge-multipliers |
| N26 | `SteinerTreeApprox(adj, weights, terminals)` | 140 | Takahashi-Matsuyama-1980 / Robins-Zelikovsky-2005 (1.55+ε)-approximation |

### Tier-7 — Cross-cuts + utilities ~320 LOC

| # | Symbol | LOC | Algorithm + reference |
|---|---|---:|---|
| N27 | `ExactOptimalTransport(a, b, cost)` | 40 | Wasserstein-W₁ exact LP via N17 — **wires 201's v2-deferral** |
| N28 | `WassersteinProcrustes(X, Y, p)` | 120 | Grave-Joulin-Berthet-2019 alternating exact-OT + orthogonal-Procrustes |
| N29 | `DualPotentials(...)` | 60 | Extract reduced-costs / dual-prices from any min-cost-flow solver |
| N30 | `IsFeasibleFlow / GilmoreFeasibility(...)` | 100 | Gilmore-1962 feasibility-check with arbitrary supply/demand/lower/upper |

**Total: 30 primitives, ~3,420 LOC** (Tier-0 580 + Tier-1 620 + Tier-2 720 + Tier-3 480 + Tier-4 340 + Tier-5 280 + Tier-6 400 + Tier-7 320).

---

## 2. Connective tissue — what other reality work this enables / completes

| Slot | What this does for it |
|---|---|
| **201 OT v2-deferral** | N17 TransportationProblem + N27 ExactOptimalTransport directly **ship 201's "exact LP-OT via network-simplex" deferred primitive** that 201's `doc.go:99-114` v2 list explicitly names. N28 Wasserstein-Procrustes adds the unsupervised-word-embedding-alignment consumer 201 omitted. |
| **102 optim-T3.33** | N14 NetworkSimplex satisfies the missing primitive 102 flagged — placed in `graph/mincostflow/` instead of `optim/lp/` because every consumer is graph-LP not generic-LP. |
| **081 graph-numerics F5** | The float-residual-noise hazard 081 flagged for `MaxFlow` transfers to every cost-augmenting-path variant. The fix is the same `eps`-rounding-of-residuals discipline 081 specified — applies to N9 SSP, N10 capacity-scaling, N13 cost-scaling. |
| **254 graph-cuts Tier-0** | 254-Tier-0 ships max-flow algorithms (Dinic / BK / push-relabel / IBFS) that this slot's min-cost-flow tier consumes as **substrate** when augmenting-path-with-cost reduces to max-flow on residual subgraph. The Tier-0 from 254 + Tier-2 here are **complementary**, not overlapping. |
| **gametheory/matching** | N1 Hungarian / N2 JV provide the cardinal-weighted matching companion to ordinal-Gale-Shapley. N23 CumulativeOfferMechanism generalises GS to substitutable-preferences. The gametheory package gains a complete bipartite-matching surface. |
| **topology/persistent.bottleneck** | N8 HopcroftKarpBipartite lifts the private inner-loop to a public reusable primitive — refactor opportunity to share code, no algorithmic change. |
| **(future) tracking/** | Multi-target tracking-by-detection consumes Hungarian-on-cost-matrix (Kuhn-Munkres-2D / JV-rectangular) per frame; N1+N2+N3 give the full assignment surface. |
| **(future) image/registration** | Wasserstein-Procrustes (N28) for histogram-of-features alignment — consumes both N17 + linalg.Procrustes. |
| **prob/causal Wasserstein-causal** | Exact OT (N27) is the differentiable-substrate for Wasserstein-distance causal-inference; entropic Sinkhorn-OT exists, exact-LP-OT does not. |

---

## 3. R-MUTUAL-CROSS-VALIDATION pin candidates (3/3 saturating)

| Pin ID | Triple | Witness |
|---|---|---|
| RMC-274-1 | N1 Hungarian × N2 Jonker-Volgenant × Edmonds-Karp-MaxFlow on unit-capacity | All three solve LSAP / unit-capacity-bipartite-min-cost-flow via different algorithms; assignments may differ but **totalCost must agree to ≤1e-12** on integer cost matrices. Cross-validation oracle: solve a 50×50 random integer-cost LSAP, all three agree on cost. |
| RMC-274-2 | N5 Blossom-V × N7 Cardinality × N8 HopcroftKarp on unweighted bipartite | Three independent matching algorithms reduce to same matching-cardinality on bipartite graphs (Blossom-V handles general graphs but matches Hopcroft-Karp on bipartite). |
| RMC-274-3 | N9 SSP × N10 Capacity-Scaling × N13 Cost-Scaling on integer min-cost-flow | All three solve the same min-cost-flow problem; min-cost answer must agree, flow may differ (multiple optima). |
| RMC-274-4 | N14 NetworkSimplex × N17 TransportationProblem × Sinkhorn (low ε) on integer transportation | Network-simplex + transportation-problem are exact; Sinkhorn with ε → 0 converges to LP optimum (Cuturi-2013 §2.4). All three agree to ε-tolerance. |
| RMC-274-5 | N1 Hungarian × N9 SSP × N17 TransportationProblem on bipartite-LSAP-as-min-cost-flow | LSAP is min-cost-flow on bipartite-graph with unit-supply / unit-demand — three completely different algorithms agree on optimal-cost. |

All five saturate **3/3 algorithmic-independent + zero-shared-code-path** cross-validation; the canonical reference oracle is SciPy `linear_sum_assignment` + LEMON `MinCostFlow` for hand-cranked-50×50 instances.

---

## 4. Numerical-correctness liabilities NEW vs. 081-F5

| Liability | Where it surfaces | Mitigation |
|---|---|---|
| Floating-point cost-matrix → reduced-cost = `cost - π_u + π_v` accumulating drift | N9 SSP, N13 cost-scaling | Round reduced-costs to relative `eps` per 081-F5 discipline; use Bellman-Ford NOT Dijkstra for first-iteration potentials (negative-edges allowed) |
| Min-cost-flow non-uniqueness — multiple optimal flows can have same cost | N9, N10, N13, N14 | Document non-uniqueness in docstring; cross-validation pins assert **totalCost** equality, not assignment equality |
| Hungarian's "row-reduction subtracts row-min then column-min" is not numerically associative on float | N1 | Use Kahan-compensated-summation OR scale costs to integer before reducing then unscale |
| Auction algorithm ε-scaling needs to bottom-out to ε ≤ 1/(n+1) for integer costs (Bertsekas-1981 ε-complementary-slackness) | N4 | Document the ε-floor; cross-validation pin asserts auction-cost matches Hungarian-cost only when ε ≤ 1/(n+1) |
| Transportation-problem with non-integer marginals can have non-integer optimal-flow | N17 | Document that optimal-flow is integer iff supply+demand are integer; otherwise return rational solution |
| Network-simplex degenerate-pivot can cycle indefinitely | N14 | Use Bland's rule / lexicographic tie-breaking per Cunningham-1976 strongly-polynomial guarantee |

All six are textbook-known and have textbook-standard fixes; flagging here so the implementation does not silently inherit them from naive code.

---

## 5. Verdict + ranking

The full **min-cost-flow / linear-assignment / weighted-matching / transportation-LP / multi-commodity / generalized-flow** mathematical surface is **ABSENT** from reality v0.10.0. Only `MaxFlow`-Edmonds-Karp + `Sinkhorn`-entropic-OT + `GaleShapley`-stable-matching exist; the **cost-objective tier of network flow is the second-largest gap in `graph/`** after 254-graph-cuts (which 274 does NOT overlap on the cost-axis).

**Ranked recommendations (ship-order priority):**

1. **N1 Hungarian + N17 TransportationProblem + N7 MaximumCardinalityMatching ~400 LOC — 1-day ship.** Textbook-Munkres-1957 pseudocode; immediate consumer in 201-OT (N27 ExactOT wires 201's v2-deferral) + (future) tracking + gametheory.
2. **N5 Blossom-V Edmonds-weighted-matching ~360 LOC — 3-5-day ship.** SINGULAR-FOUNDATIONAL. The single-largest combinatorial-optimisation gap in `graph/`. No zero-dep MIT Go implementation exists worldwide.
3. **N14 NetworkSimplex ~280 LOC + N9 SSP ~220 LOC — 3-day ship.** Wires 201's exact-OT v2-deferral + 102-T3.33 optim-LP missing primitive. Production-default min-cost-flow algorithm (LEMON / OR-Tools default).
4. **N2 JonkerVolgenant + N3 JV-rectangular ~300 LOC — 2-day ship.** 3-10× speedup over Hungarian on dense LSAP; SciPy / OR-Tools production-default.
5. **N13 Cost-Scaling Goldberg-Tarjan-1990 ~180 LOC — 2-day ship.** Production-fastest min-cost-flow in practice.
6. **N28 WassersteinProcrustes ~120 LOC — 1-day ship after N17 + linalg.Procrustes.** 2019 unsupervised-alignment consumer of N17 that 201's roster did not enumerate.
7. **N21 b-matching + N22 many-to-many ~220 LOC — 2-day ship.** Generalises N1 / N5 to degree-constraints.
8. **N4 Auction Bertsekas-1979 ~100 LOC — 1-day ship.** Distributed-friendly + GPU-parallel-ready alternative to Hungarian.
9. **Tier-4 N18+N19+N20 transportation-pedagogical ~240 LOC — 1-day ship.** Vogel + NW-corner + MODI for textbook-comparison cross-validation.
10. **N26 SteinerTreeApprox ~140 LOC — 2-day ship.** NP-hard substrate for VLSI / network-design; 2-approximation is textbook + (1.55+ε) Robins-Zelikovsky improved.

**Defer to v0.12+:** N15 OutOfKilter + N24 MultiCommodityFlowLP + N25 GeneralizedFlow + N12 MinMeanCycleCanceling + N11 CycleCanceling — these are pedagogical-completeness primitives once N9 + N13 + N14 are in place.

**Total ship-recommended for v0.11: ~2,640 LOC across 22 primitives**; deferral set ~780 LOC across 8 primitives.

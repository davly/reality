# 281 — new-temporal-graphs (Temporal Graphs / Time-Respecting Paths / Temporal Motifs)

## Headline
reality v0.10.0 ships ZERO temporal-graph surface (edge type is `[2]string`/`[2]int`, no timestamp) — the entire Holme-Saramäki-2012 / Casteigts-2012 / Wu-2014 / Paranjape-2017 / Latapy-2018 / Rozenshtein-2016 canon is greenfield; cheapest day-1 PR is `graph/temporal/` sub-package with `TemporalEdge{u,v,t}` + Wu-2014 earliest-arrival BFS (~300 LOC, zero blockers, additive).

## Findings

### State at HEAD (verified by direct grep on `*.go`)

| Surface | Path | Temporal-graph relevance |
|---|---|---|
| `Edge = [2]string` | `graph/graph.go:14` | Static edge tuple. **NO timestamp field.** Keystone-blocker — every primitive in this slot needs a richer edge type. |
| `IntAdjacency = map[int][]int` | `graph/types.go:7` | Static directed adjacency. NO time-indexed adjacency. |
| `Dijkstra(adj, weights, source)` | `graph/shortest.go:29` | Min-heap shortest-path on **time-invariant** weights. Adapter-style temporal-Dijkstra is straight-line: relax edge `(u,v,t,λ)` only if `t ≥ arrival[u]`, set `arrival[v] = min(arrival[v], t+λ)`. |
| `BellmanFord(adj, weights, source)` | `graph/bellman_ford.go` | Same — substrate for temporal-Bellman-Ford (Dehne-Sevilgen-1996 *DAM* 67:131). |
| `BFS / IterativeBFS` | `graph/bfs.go` | Static BFS — substrate for time-respecting-BFS (one-pass over `(u,v,t)` sorted by `t`). |
| `LouvainCommunities` | `graph/community.go:155` | Static modularity. NO temporal modularity (Mucha-Richardson-Macon-Onnela-Porter-2010 *Science* 328:876 multilayer-modularity ABSENT; slot 280 owns the inference axis). |
| `PageRank, EigenvectorCentrality, Betweenness` | `graph/centrality.go, pagerank.go` | All static — Rozenshtein-Gionis-2016 temporal PageRank ABSENT, Wu-Cheng-2014 temporal-betweenness ABSENT. |
| `MaxFlow` | `graph/flow.go` | Static. Temporal flow networks (Köhler-Langkau-Skutella-2002 *Networks* 39:142) ABSENT. |
| `MinimumSpanningTree` | `graph/mst.go` | Static Kruskal/Prim. Holm-Lichtenberg-Thorup-2001 *J-ACM* 48:723 fully-dynamic-MST ABSENT. |
| `timeseries/{garch,dcc}/` | timeseries pkg | Volatility models — NOT temporal-graph substrate. Useful only as "edges-vs-time" sanity (volatility-clustering ≡ burstiness in temporal-graph edge inter-arrival, Karsai-Kaski-Barabási-Kertész-2012 *Sci-Rep* 2:397). |
| `changepoint/bocpd.go` | changepoint pkg | Adams-MacKay-2007 BOCPD — REUSABLE on edge-rate time-series for regime-change detection on temporal-graph density (link-stream ρ(t)). |
| `Temporal\|TimeRespect\|LinkStream\|EdgeStream\|Snapshot.*Graph\|TimeWindow.*Graph\|TemporalBetween\|TemporalCentrality\|TemporalPageRank\|TemporalMotif` | repo-wide grep | **ALL ABSENT** outside review corpus. |

### Slot boundaries

- **Slot 162** (synergy-graph-prob) = static random-graph **generators** (Erdős-Rényi, SBM, BA, WS).
- **Slot 274** (P-time graph algorithms) = static-graph polynomial algorithms (matching, planar, treewidth).
- **Slot 277** (exact MIP / B&B) = combinatorial optimisation on static graphs.
- **Slot 280** (network generative models) = SBM/DCSBM/MMSB **inference**; explicitly cites Matias-Miele-2017 *JRSS-B* 79:1119 dynamic-SBM as future-work — slot 281 owns the **algorithmic / structural** temporal-graph axis (paths, motifs, centrality), 280 owns the **statistical** temporal-graph axis (block-membership over time).
- **Slot 271** (spectral-clustering) / **273** (spectral-embedding) = static spectral. Multilayer/temporal-spectral (Tang-Lu-Dhillon-2009 community-detection on multi-network) is downstream of 271/273+281.
- **Slot 281 (this) = TEMPORAL/DYNAMIC GRAPHS — edge timestamps as first-class.** Mostly orthogonal to 280; only overlap is dynamic-SBM (which 280 explicitly defers to fitting-axis review).

### Web context (no MIT pure-Go exists)

- **Holme-Saramäki-2012** *Phys-Rep* 519:97 "Temporal networks" — canonical 100+page review. Establishes (u,v,t) link-stream representation, time-respecting paths, temporal motifs.
- **Wu-Cheng-Huang-Ke-Lu-Xu-2014** *PVLDB* 7:721 "Path Problems in Temporal Graphs" — 4 path types: earliest-arrival, latest-departure, fastest, shortest. All O(n+M) where M = #edges. Reference C++ ~800 LOC.
- **Paranjape-Benson-Leskovec-2017** WSDM "Motifs in Temporal Networks" — 3-node-3-edge δ-temporal-motif, 8 canonical patterns, fast counting via 2-node-3-edge decomposition. Reference SNAP C++ (BSD-style, but heavy SNAP dependency) — pure-Go MIT ABSENT.
- **Latapy-Viard-Magnien-2018** *SNAM* 8:61 (arXiv:1710.04073) "Stream Graphs and Link Streams" — node-density d(v), link-density d(uv), clusters, paths re-cast in continuous-time. Reference Python `straph` (LIP6) GPL.
- **Rozenshtein-Gionis-2016** ECML-PKDD "Temporal PageRank" — random-walk on time-respecting walks, online streaming algorithm. Reference Python (polinapolina/temporal-pagerank) MIT but research-grade ~400 LOC.
- **Casteigts-Flocchini-Quattrociocchi-Santoro-2012** *IJPEDS* 27:387 "Time-Varying Graphs" — taxonomy of TVG classes (TVG-R, TVG-B, TVG-F).
- **Kempe-Kleinberg-Kumar-2002** *J-Comp-Sys-Sci* 64:820 "Connectivity and Inference Problems for Temporal Networks" — first proof temporal-min-cut is NP-hard, polynomial reachability.
- **Holm-Lichtenberg-Thorup-2001** *J-ACM* 48:723 — fully-dynamic-connectivity O(log²n) amortised, dynamic-MSF O(log⁴n). Reference C++ (tomtseng/dynamic-connectivity-hdt) MIT ~1500 LOC, depends on Euler-tour-trees.
- **Henzinger-King-1999** *J-ACM* 46:502 — randomised dynamic-connectivity O(log³n). Simpler than HLT but Las-Vegas.
- **Kovanen-Karsai-Kaski-Kertész-Saramäki-2011** *J-Stat-Mech* P11005 — temporal-motif (Δt-window) prior to Paranjape, simpler counting.

## Concrete recommendations

1. **T0 — `graph/temporal/types.go` (~80 LOC, zero deps).** New sub-package. Define:
   ```go
   type TemporalEdge struct { U, V int; T float64; Lambda float64 } // Lambda = traversal duration; 0 = instantaneous (Holme-2012)
   type TemporalGraph struct { N int; Edges []TemporalEdge }       // edges sorted by T (invariant enforced by ctor)
   func NewTemporalGraph(n int, edges []TemporalEdge) *TemporalGraph // sorts in place, dedups
   func (tg *TemporalGraph) Window(t0, t1 float64) *TemporalGraph    // sliding-window (Latapy-2018 link-stream slice)
   func (tg *TemporalGraph) Snapshot(t float64) graph.IntAdjacency   // collapse to static-graph (regression hook)
   ```
   Unblocks every primitive below. Day-1 PR.

2. **T1 — `graph/temporal/paths.go` (~200 LOC).** Wu-2014 four path-types, all single-source O(n + M log M) (M = #edges; sort once, sweep):
   ```go
   func EarliestArrival(tg *TemporalGraph, src int, t0 float64) []float64 // arrival[v] = min t s.t. time-respecting path src→v exists in [t0, ∞)
   func LatestDeparture(tg *TemporalGraph, dst int, tEnd float64) []float64
   func FastestPath(tg *TemporalGraph, src, dst int) (start, end float64)  // minimises end-start
   func ShortestPath(tg *TemporalGraph, src, dst int) (hops int)            // min #temporal-edges
   ```
   Algorithm is heap-based sweep over `Edges` ascending by T (no Dijkstra heap-of-distances needed for earliest-arrival — single pass). **R-MUTUAL-CROSS-VALIDATION 3/3**: (a) one-pass sweep ≡ (b) adapted Dijkstra on time-expanded DAG (one node per (v, t-event) pair) ≡ (c) adapted Bellman-Ford on time-expanded DAG; on n=200 random temporal graph, all three return identical arrival[]. **Regression hook**: when all `T = 0`, EarliestArrival(src) ≡ static BFS-distance from src in the collapsed graph (gates against drift).

3. **T2 — `graph/temporal/connectivity.go` (~120 LOC).** Kempe-Kleinberg-Kumar-2002 temporal reachability. Define temporal-strongly-connected = ∀(u,v) time-respecting path u→v AND v→u (within [t0, t1]). Compute via per-source EarliestArrival → reachability-matrix R[i][j] ∈ {0,1} → run static-Tarjan on R. O(n · (n + M)). NP-hard variants (min-cut) flagged but not implemented.

4. **T3 — `graph/temporal/centrality.go` (~250 LOC).** Wu-Cheng-2014 + Tang-Musolesi-Mascolo-Latora-2010 *SIGCOMM* temporal-betweenness:
   ```go
   func TemporalBetweenness(tg *TemporalGraph) []float64 // σ_uv(w,t) / σ_uv(t) summed over (u,v,t)
   func TemporalCloseness(tg *TemporalGraph) []float64   // 1 / mean earliest-arrival distance
   ```
   O(n · (n + M)) via repeated EarliestArrival/Brandes-style accumulation. Brandes-2001 *J-Math-Soc* 25:163 substrate is the existing static-betweenness in `graph/centrality.go` — refactor to extract `BrandesAccumulate(prev, σ)` as shared kernel between static and temporal.

5. **T4 — `graph/temporal/motifs.go` (~350 LOC).** Paranjape-Benson-Leskovec-2017 δ-temporal-motifs. Implement `Count2N3E(tg, delta)` for 2-node-3-edge baseline (8 canonical patterns) and `Count3N3E(tg, delta)` for 3-node-3-edge (40 canonical patterns over 6 static motifs). Algorithm: per-edge sliding-δ-window enumeration with 2-node decomposition (Algorithm 1 of Paranjape-2017, O(M·d_max·δ) expected). **R-MUTUAL-CROSS-VALIDATION 3/3**: brute-force O(M³) enumeration on n≤20 graph ≡ Algorithm-1-fast ≡ static-motif-count when δ → ∞ AND timestamps collapse (matches existing static-3-cycle count if added in slot 274).

6. **T5 — `graph/temporal/walks.go` + `temporal_pagerank.go` (~280 LOC).** Rozenshtein-Gionis-2016 temporal PageRank. Streaming algorithm: for each edge (u,v,t) in time order, transient PageRank[v] += β · ActiveWalks[u]; ActiveWalks[v] += (1-α) · ActiveWalks[u]; PageRank[u] += 1. Two parameters: damping α (0.85 default), transition-probability β (0.5 default). **R-MUTUAL-CROSS-VALIDATION 3/3**: when timestamps collapse to t=0 AND graph is static-stable, TemporalPageRank → existing static `PageRank()` from `graph/pagerank.go` within tol 1e-9 over n=100 ER graph (regression-pin).

7. **T6 — `graph/temporal/linkstream.go` (~180 LOC).** Latapy-Viard-Magnien-2018 link-stream framework:
   ```go
   func NodeDensity(tg *TemporalGraph, t0, t1 float64) []float64 // d(v) per Latapy §4
   func LinkDensity(tg *TemporalGraph, u, v int, t0, t1 float64) float64 // d(uv)
   func StreamGraphDensity(tg *TemporalGraph, t0, t1 float64) float64    // δ(L)
   func ClusteringCoefficient(tg *TemporalGraph, t0, t1 float64) []float64 // local
   ```
   Pure book-keeping over edge intervals. Continuous-time integration via prefix-sum on event timestamps (no quadrature needed).

8. **T7 — `graph/temporal/snapshot.go` (~140 LOC).** Snapshot-model (Tang-Mascolo-Musolesi-Latora-2010 *Eurosys*): partition `[t_min, t_max]` into K equal-width bins, each bin → static graph. Provides `Snapshots(tg, K) []graph.IntAdjacency` + temporal-stability metric (Jaccard between consecutive snapshots). Cheap; bridges existing static-graph centrality to temporal axis.

9. **T8 — DEFER. Holm-Lichtenberg-Thorup-2001 fully-dynamic-connectivity** (~1500 LOC, depends on Euler-tour-trees / link-cut-trees not yet in repo). Mark as slot 281-FUTURE and emit issue when slot 097 or sibling lands link-cut-trees / Euler-tour-trees substrate. **NOT day-1.**

10. **T9 — DEFER. Multilayer modularity / temporal-Louvain** (Mucha-2010 *Science* 328:876). Cross-cuts slot 280 inference axis. Out of scope; flag for slot 280-extension.

### Cheapest day-1 PR

T0 + T1 alone (`graph/temporal/types.go` + `graph/temporal/paths.go`, ~280 LOC + ~120 LOC test, golden-file vectors at `graph/temporal/testdata/*.json` for n≤30 hand-validated temporal DAGs). Zero blockers, zero new deps, additive sub-package — does not touch existing static-graph code. Locks the `TemporalEdge` struct contract before downstream slots fork.

### Blockers / dependencies

- T1-T7 only need T0 (~80 LOC types + sort).
- T3 wants `BrandesAccumulate` factored out of `graph/centrality.go` — small refactor (~30 LOC), no breaking change to public API.
- T6 link-stream-clustering is mathematically clean but needs careful interval-arithmetic on `[t_start, t_end]` edge durations — recommend instantaneous-only first pass (`Lambda = 0`) then extend.
- T8 truly blocks on Euler-tour-trees / link-cut-trees — emit GitHub issue, do not block this slot's day-1 PR.

## Cross-cutting

- **Slot 280 (new-sbm)** ← TemporalGraph type from T0 is the input substrate for Matias-Miele-2017 dynamic-SBM (which 280 explicitly defers); shipping T0 unblocks 280's "future-work" axis. Snapshot model from T7 is the naive baseline that dynamic-SBM beats.
- **Slot 162 (synergy-graph-prob)** ← Forward generators should grow `TemporalSBM(blocks, P_t, λ)`, `ContactProcess(λ_birth, λ_death)`, `ActivityDriven(a_i, m)` (Perra-Goncalves-Pastor-Satorras-Vespignani-2012 *Sci-Rep* 2:469). These need T0 first.
- **Slot 023 (changepoint-sota)** ← BOCPD on `LinkDensity(tg, t0, t1)` time-series detects regime-shifts in temporal-graph density — direct consumer of T6.
- **Slot 154 (synergy-chaos-timeseries)** ← Burstiness B = (σ-μ)/(σ+μ) on edge inter-arrival times (Karsai-2012) is a chaos×timeseries×temporal-graph triple-synergy.
- **Slot 274 (P-time graph algorithms)** ← Static-motif count is the t→∞ limit of temporal-motif (T4); if 274 ships static-3-cycle / static-FFL count, T4 reuses the canonical-pattern enumeration.
- **Slot 271 / 273 (spectral)** ← Multilayer / supra-adjacency spectral clustering (De Domenico-Solé-Ribalta-Cozzo-Kivelä-Moreno-Porter-Gómez-Arenas-2013 *PRX* 3:041022) is downstream of T7 snapshot model.
- **Pistachio (relationship-evolution graphs)** ← Direct consumer: relationship strength over time = TemporalEdge with Lambda=interaction-duration; T3 temporal-betweenness ranks "bridge" people in a network as it evolves; T5 temporal-PageRank captures concept-drift in attention.
- **aicore (causalmath)** ← Time-respecting paths from T1 are the substrate for time-aware Granger/causal-discovery (Eichler-2007 *J-Econometrics* 137:334).
- **Communication / contact-tracing / epidemic** ← T1 EarliestArrival on a contact-temporal-graph IS the SI-model first-infection time (Holme-2014 *PLOS-CB* 10:e1003142); T6 link-stream density tracks epidemic potential.
- **Financial fraud detection** ← T4 temporal-motifs on transaction graphs detect cycle-patterns (mule-rings, layering) — Paranjape-2017 §6 demonstrated this on Bitcoin.
- **Transportation timetables** ← T1 fastest/earliest-arrival paths are the canonical algorithm for journey-planning (Bast-Delling-Goldberg-Müller-Hannemann-Pajor-Sanders-Wagner-Werneck-2016 *Algorithm-Engineering* 19:1).

## Sources

### Repo files (verified)
- `C:/limitless/foundation/reality/graph/graph.go:14` — `Edge = [2]string` static.
- `C:/limitless/foundation/reality/graph/types.go:7` — `IntAdjacency` static.
- `C:/limitless/foundation/reality/graph/shortest.go:29` — `Dijkstra` static, weights time-invariant.
- `C:/limitless/foundation/reality/graph/bellman_ford.go:42` — static.
- `C:/limitless/foundation/reality/graph/community.go:155` — static Louvain.
- `C:/limitless/foundation/reality/graph/centrality.go, pagerank.go, mst.go, flow.go, bfs.go, dag.go, importance.go` — all static.
- `C:/limitless/foundation/reality/timeseries/{garch,dcc}/` — only volatility models, no graph time-series.
- `C:/limitless/foundation/reality/changepoint/bocpd.go` — Adams-MacKay BOCPD, reusable on edge-rate series.
- `C:/limitless/foundation/reality/reviews/overnight-400/MASTER_PLAN.md:298` — slot 281 line.
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/280-new-sbm.md` — slot 280 defers dynamic-SBM to slot 281.
- Repo-wide grep `Temporal|TimeRespect|LinkStream|EdgeStream|Snapshot.*Graph|TemporalBetween|TemporalCentrality|TemporalPageRank|TemporalMotif` → ALL ABSENT outside review corpus.

### Web sources
- Holme-Saramäki-2012 *Phys-Rep* 519:97 "Temporal networks" — review.
- Wu-Cheng-Huang-Ke-Lu-Xu-2014 *PVLDB* 7:721 "Path Problems in Temporal Graphs" — https://www.vldb.org/pvldb/vol7/p721-wu.pdf
- Paranjape-Benson-Leskovec-2017 WSDM "Motifs in Temporal Networks" — https://arxiv.org/abs/1612.09259, code https://snap.stanford.edu/temporal-motifs/
- Latapy-Viard-Magnien-2018 *SNAM* 8:61 "Stream Graphs and Link Streams" — https://arxiv.org/abs/1710.04073
- Rozenshtein-Gionis-2016 ECML-PKDD "Temporal PageRank" — Springer LNCS 9852:674. Reference impl https://github.com/polinapolina/temporal-pagerank
- Casteigts-Flocchini-Quattrociocchi-Santoro-2012 *IJPEDS* 27:387 "Time-Varying Graphs".
- Kempe-Kleinberg-Kumar-2002 *J-Comp-Sys-Sci* 64:820 "Connectivity and Inference Problems for Temporal Networks".
- Holm-Lichtenberg-Thorup-2001 *J-ACM* 48:723 — fully-dynamic-connectivity, dynamic-MSF poly-log. Reference impl https://github.com/tomtseng/dynamic-connectivity-hdt (MIT).
- Henzinger-King-1999 *J-ACM* 46:502 — randomised dynamic-connectivity.
- Kovanen-Karsai-Kaski-Kertész-Saramäki-2011 *J-Stat-Mech* P11005 — earlier temporal-motif framework.
- Mucha-Richardson-Macon-Onnela-Porter-2010 *Science* 328:876 — multilayer modularity.
- Tang-Musolesi-Mascolo-Latora-2010 *SIGCOMM-WOSN* — temporal-betweenness, temporal-closeness.
- Karsai-Kaski-Barabási-Kertész-2012 *Sci-Rep* 2:397 — burstiness in temporal graphs.
- Perra-Goncalves-Pastor-Satorras-Vespignani-2012 *Sci-Rep* 2:469 — activity-driven temporal-graph generator.
- Matias-Miele-2017 *JRSS-B* 79:1119 — dynamic-SBM (boundary with slot 280).

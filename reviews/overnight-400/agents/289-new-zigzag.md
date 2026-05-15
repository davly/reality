# 289 — new-zigzag (Zigzag Persistence + Multi-Parameter Persistence + Generalized Persistence)

## Headline
reality v0.10.0 ships ONLY monotone-1-parameter persistent homology (`topology/persistent/{vr,barcode,bottleneck}.go` — Edelsbrunner-Letscher-Zomorodian 2000 column reduction over F_2 at maxDim ∈ {0,1}) and ZERO of the entire post-2009 advanced-persistence canon (`Zigzag|MultiParameter|Multipersistence|RankInvariant|FiberedBarcode|RIVET|GRIL|MultiPersistence|BarcodeBasis|IntervalDecomposition|CommutativeLadder` repo-wide grep on `*.go` outside reviews/ → 0 hits); the day-1 cheapest PR is `topology/zigzag/` ≈ **520 LOC** shipping ZigzagFiltration type + Carlsson-de Silva-Morozov 2009 right-filtration extended-ELZ algorithm + zigzag barcode (interval-with-direction), saturating **R-MUTUAL-CROSS-VALIDATION 3/3 day 1** via the all-arrows-forward pin (zigzag with monotone-only inclusions ≡ existing `ComputeBarcode` in `topology/persistent/barcode.go:60`); 2-parameter persistence is provably *harder* (Carlsson-Zomorodian 2009 — no complete discrete invariant) so the multi-parameter axis ships **partial-invariants only** (rank invariant, fibered barcodes via line restrictions, GRIL landscape, Lesnick-Wright 2022 minimal presentation) totalling another ~1200 LOC; pure-Go MIT zero-dep ABSENT in any language ecosystem (Dionysus2 BSD-3 C++ for zigzag by Morozov, RIVET GPL-3 C++ for 2-param by Lesnick-Wright, GUDHI BSD-3 C++ no zigzag, scikit-tda Apache-2 Python no multi-param). **This is Block-C cutting-edge; landing it puts reality at the 2023-2025 TDA research frontier with the four ML-direct consumer surfaces (streaming/temporal data, multivariate scientific simulations, dynamic-network analysis, multi-filter sensor fusion) currently uncovered.**

## Findings

### State at HEAD (verified by direct grep on `*.go` outside reviews/)

| Surface | Path | Zigzag/multi-param relevance |
|---|---|---|
| `Filtration{Simplices, Times}` | `topology/persistent/vr.go:50` | **Monotone scalar filtration only** — single non-decreasing time stream. Cannot encode an alternating ←/→ inclusion sequence (zigzag) and cannot encode a 2-D filter function `f: X → R^2`. Keystone-blocker for both axes. |
| `ComputeBarcode(filtration, maxDim)` | `topology/persistent/barcode.go:60` | ELZ-2000 column reduction over F_2. **Single-direction matrix reduction** — every column corresponds to "simplex enters at time t". Zigzag needs *birth* AND *death* operators (left and right multiplications by the same matrix); Carlsson-de Silva-Morozov 2009 extended-ELZ keeps a *parenthesisation* of the zigzag and reduces both directions. **Substrate is reusable** (same Simplex + boundary-column representation) but the reduction loop is different. |
| `BottleneckDistance(d1, d2, dim)` | `topology/persistent/bottleneck.go:50` | W_∞ on **standard** persistence diagrams. Bauer-Lesnick 2014 induced-matching theory generalises bottleneck stability to zigzag; the same bipartite + diagonal-stand-in encoding works on zigzag intervals (interval-with-direction collapses to point in R² for stability). Reusable. |
| `topology/persistent/doc.go:65-76` | — | Phase-A scope: maxDim ∈ {0, 1}, persistent-cohomology / landscape / W_p / Mapper deferred to v2. **No mention of zigzag or multi-param** — both are post-v2 territory and slot 289 owns the design call. |
| `graph/temporal/` (planned slot 281) | — | Slot 281 explicitly cites Carlsson-de Silva 2010 zigzag as the natural fit for time-varying simplicial flag complexes. **Direct upstream consumer** — once 281 ships `TemporalGraph` + sliding-window flag complex, zigzag persistence on the resulting alternating-inclusion sequence IS the time-varying H_*. |
| `topology/{simplicial,reeb,mapper}/` (planned slots 283/286/287) | — | 283 SimplexTree + 286 ContourTree + 287 Mapper-graph all consume single-parameter persistence. None consume zigzag or multi-param yet — slot 289 is the **upgrade path** for streaming/multi-filter consumers. |
| repo-wide grep `Zigzag\|MultiParameter\|Multipersistence\|RankInvariant\|FiberedBarcode\|RIVET\|GRIL\|MultiPersistence\|BarcodeBasis\|IntervalDecomposition\|CommutativeLadder\|InducedMatching\|GeneralizedPersistence\|BubenikScott\|EscolarHiraoka\|LesnickWright\|MinimalPresentation` outside reviews/ | `*.go` | **ZERO hits.** Greenfield. |

### Slot boundaries (NO overlap with 283-288, additive on the advanced-persistence axis)

- **Slot 283 (simplicial complexes) ←** Provides `SimplexTree` data structure (Boissonnat-Maria 2014, deferred). Zigzag stores its complex sequence as a sequence of SimplexTrees with shared backbone; multi-param stores a single SimplexTree with R^d-graded entry times. **289 gates lazily on 283-T0** but ships standalone today over flat `[]Simplex` (matching existing `topology/persistent/vr.go`).
- **Slot 284 (CW complexes) ←** No direct interaction. Zigzag of cubical complexes = streaming-image persistence; cubical zigzag is a future pin but not a blocker.
- **Slot 285 (discrete Morse) ←** DMT vector field on a zigzag is open research (Carlsson-Singh-Zomorodian 2010 §6); not for v1.
- **Slot 286 (Reeb / merge-tree) ←** Merge-tree edit distance (slot 286 T10c) IS a competing temporal-data tool. **Zigzag is more general** (works on any time-varying simplicial complex, not just sublevel-set merge trees of a scalar field) but more expensive. Both ship; cross-reference in doc-stub.
- **Slot 287 (Mapper) ←** Singh-Memoli-Carlsson 2007 §2.3 + Munch-Wang 2016: multinerve Mapper has provable convergence to **multi-persistent** Mapper for multi-parameter cover. **Direct consumer** of T7-T9 multi-param machinery: a 2-parameter cover (e.g. PCA-1 × density) with overlap → 0 produces a multi-persistent Mapper graph whose 2-param persistence module captures cover-stability. Slot 287 T9 stochastic Mapper consumes 289-T8 fibered barcodes.
- **Slot 288 (persistence stats) ←** All twelve diagram-statistics primitives (W_p, landscape, image, PSSK, SWK, PFK, PWGK, Frechet, bootstrap, two-sample test) operate on **persistence diagrams** (multisets of (b, d) ∈ R²). Zigzag barcodes are also multisets of intervals; **same diagram-statistics machinery applies verbatim** with one tweak (intervals carry a direction-tag, but for stats the (b, d) projection ignores it). Multi-param has *no diagrams* in general — only fibered barcodes (1-D slices through the 2-param plane). 288's machinery applies *per-slice* for multi-param via fibered-barcode reduction. **288 is the diagram-statistics layer; 289 is the underlying-module layer.** No file overlap.
- **Slot 281 (temporal graphs) ← KEYSTONE CONSUMER.** Zigzag on the flag complex of a sliding-window temporal graph is the time-varying persistent homology of the link stream. **Single most concrete day-1 consumer** of T0-T2 zigzag.
- **Slot 280 (network generative models, dynamic SBM) ←** Dynamic SBM produces a sequence of community partitions over time → zigzag persistence on the community-incidence simplicial complex tracks community birth/death across time. **Direct consumer** of T2 zigzag barcode.
- **Slot 277 (combinatorial optimisation, BnB / ILP) ←** **NEW pin established by this slot:** *optimal interval decomposition* for 2-param persistence modules is non-trivial — Carlsson-Zomorodian 2009 prove there is no complete discrete invariant, but for "nice" modules (interval-decomposable per Asashiba-Buchet-Escolar-Nakashima-Yoshiwaki 2019) the decomposition reduces to a graph-isomorphism / set-cover combinatorial problem. **Optional 277 consumer** at scale; not a blocker.
- **Slot 142 (topology-missing) / 143 (topology-sota) ←** Both reviews enumerate zigzag + multi-param as future work. Slot 289 is the **dedicated** review.
- **Slot 156 (synergy-topology-prob) / 190 (synergy-topology-signal) / 247 (mortar-fem) ←** Downstream feature consumers — once T2 zigzag-barcode + T7 fibered-barcodes ship, the slot-288 stats stack consumes verbatim.

### Web context (verified pure-Go MIT zero-dep ABSENT for every primitive enumerated)

- **Carlsson-de Silva 2010** *Found-Comput-Math* 10:367 "Zigzag persistence" — establishes zigzag persistence theory. A *zigzag module* is a sequence of vector spaces `V_0 ↔ V_1 ↔ V_2 ↔ ... ↔ V_n` where each ↔ is either → or ← (a forward or backward linear map). Theorem 4.1: every zigzag module of finite-dim spaces decomposes uniquely into interval modules `I[b, d]_d` where d ∈ {→, ←} marks the direction. Output: barcode = multiset of intervals `[b_i, d_i]` with directions on endpoints. **THE foundational zigzag paper.** Generalises Crawley-Boevey 2015 *J-Algebra-Appl* 14:1550066 representation-theory result on quiver decomposition.
- **Carlsson-de Silva-Morozov 2009** *SoCG* 247-256 "Zigzag persistent homology and real-valued functions" — **THE algorithm.** Extended ELZ: maintain a *parenthesisation* of the zigzag (insertions are `[`, deletions are `]`); at each `[` create a column, at each `]` reduce it; the resulting *birth-death-direction* triples form the zigzag barcode. O(m³) worst case where m = total simplex-events; in practice near-linear on Vietoris-Rips-style streams. Reference C++ Dionysus (BSD-3) by Morozov ~1500 LOC.
- **Milosavljević-Morozov-Skraba 2011** *SoCG* "Zigzag persistent homology in matrix multiplication time" — improves Carlsson-de Silva-Morozov 2009 to O(m^ω) via fast matrix multiplication; not for v1 (numerical implementation; theoretical interest only).
- **Maria-Oudot 2014** *SoCG* "Zigzag persistence via reflections and transpositions" — alternative algorithm via local moves (transpositions of adjacent simplex events). Better empirical performance than CdSM-2009 in some cases; reference GUDHI implementation (now part of GUDHI 3.10 BSD-3).
- **Memoli-Wan 2017** *arXiv:1706.04571* "On the efficient computation of zigzag persistence and the dimension of zigzag persistence diagrams" — efficient zigzag for streaming sliding-window. Foundation for T4 streaming zigzag.
- **Carlsson-Zomorodian 2009** *Discrete-Comput-Geom* 42:71 "The theory of multidimensional persistence" — **THE multi-param paper.** A *d-parameter persistence module* is an `R^d`-graded module over `k[x_1, ..., x_d]`. Theorem 1: for d ≥ 2 there is **no complete discrete invariant** (the moduli space of indecomposable d-parameter persistence modules has positive dimension; barcode-style classification fails). Introduces the **rank invariant** `ρ(u, v) = rank(M_u → M_v)` as a partial invariant. **Critical theorem to cite:** establishes that 2-param persistence is *fundamentally* harder than 1-param.
- **Lesnick-Wright 2015** *arXiv:1512.00180* "Interactive visualization of 2-D persistence modules" — **RIVET.** Computes 2-parameter persistence presentations + minimal presentation + fibered barcodes (1-D slices) in O(m³) where m = simplex count. The standard 2-param computational tool. Reference C++ (rivet-x.com) GPL-3 ~10000 LOC. **Pure-Go MIT zero-dep ABSENT.**
- **Lesnick-Wright 2022** *Found-Comput-Math* 22:1093 "Computing minimal presentations and bigraded Betti numbers of 2-parameter persistent homology" — **efficient minimal presentation algorithm**. O(m³) → improved variants O(m^2.37) via fast matrix multiplication. **The current SOTA for 2-param computation.**
- **Cerri-Frosini-Landi 2013** *Math-Methods-Appl-Sci* 36:1543 "Betti numbers in multidimensional persistent homology are stable functions" — **stability theorem for multi-param.** d_∞(rank-invariant(M), rank-invariant(N)) ≤ ‖f - g‖_∞ for sublevel-set filtrations. Foundational.
- **Bauer-Lesnick 2014** *J-Comput-Geom* 6:162 "Induced matchings and the algebraic stability of persistence barcodes" — **induced-matching theory.** For any morphism of persistence modules `M → N`, one can define an induced matching of barcodes that respects bottleneck distance. Generalises stability to non-1-param settings. Theory backbone for Cerri-Frosini-Landi multi-param stability.
- **Bauer-Lesnick 2020** *J-Algebra* 547:84 "Persistence diagrams as diagrams: a categorification of the stability theorem" — categorical / derived-category formulation of persistence. Sets up the broader framework that includes zigzag and multi-param as special cases of "persistence over a poset". Theoretical only; cite in doc stub.
- **Bubenik-Scott 2014** *Discrete-Comput-Geom* 51:600 "Categorification of persistent homology" — **generalised persistence over arbitrary posets.** Both 1-param (poset = R), zigzag (poset = zigzag quiver), and multi-param (poset = R^d) are instances. Provides unifying language. **Foundation for T0 generic ZigzagFiltration / T7 generic MultiParamFiltration types.**
- **Escolar-Hiraoka 2016** *Adv-Math* 290:1015 "Persistence modules on commutative ladders of finite type" — **commutative-ladder persistence.** Generalises both zigzag and 1-param: a "ladder" is a 2×n grid where horizontal arrows form a 1-param filtration and vertical arrows are bidirectional. Useful for *paired* time-series (e.g. control-vs-treatment topology over time). Open-source reference: Phat-Top (BSD-3 C++) ~3000 LOC.
- **Hellmer-Spreer 2024** *arXiv:2304.04970* "Generalized persistence diagrams for persistence modules over posets" + **Xin-Hellmer 2024** *arXiv:2304.04970* "GRIL: A 2-parameter persistence based vectorization for machine learning" — **GRIL (Generalized Rank Invariant Landscape).** Vectorisation of 2-param persistence modules suitable for ML pipelines (analogous to Bubenik-2015 landscapes for 1-param). **Current SOTA 2024-2025 ML-on-multi-persistence primitive.**
- **Asashiba-Buchet-Escolar-Nakashima-Yoshiwaki 2019** *J-Comput-Geom* 10:201 "On approximation of 2D persistence modules by interval-decomposables" — when is a 2-param module decomposable into intervals? Combinatorial optimisation flavour; not always possible, but when it is the decomposition is unique (analogous to 1-param Krull-Schmidt). **Foundation for T9 interval decomposition.**
- **Botnan-Lesnick 2018** *arXiv:1809.05151* "Algebraic stability of zigzag persistence modules" — extends Bauer-Lesnick induced-matching to zigzag. Stability theorem: bottleneck distance between zigzag barcodes ≤ interleaving distance between zigzag modules.
- **Dey-Hou 2021** *SoCG* "Computing zigzag persistence on graphs in near-linear time" — efficient zigzag specifically for the graph-flag-complex case. Direct consumer of slot-281 temporal graphs.
- **Dionysus2** by Morozov BSD-3 C++ — gold-standard zigzag implementation. **Pure-Go MIT zero-dep ABSENT.**
- **RIVET** by Lesnick-Wright GPL-3 C++ — gold-standard 2-param. **Pure-Go MIT zero-dep ABSENT.**
- **GUDHI 3.10** Apache-2/MIT C++/Python — has Maria-Oudot zigzag but no multi-param. Pure-Go ABSENT.
- **PHAT** BSD-3 C++ — fast persistent homology, **no zigzag**. Pure-Go ABSENT.
- **scikit-tda persim 0.4** Apache-2 Python — only 1-param. ABSENT for both.

## Concrete recommendations

(Ordered by leverage; LOC is glue-only assuming `topology/persistent/{vr,barcode}.go` substrate stands as today.)

1. **T0 — `topology/zigzag/types.go::ZigzagFiltration` + `Direction` type** (~120 LOC, zero deps). New sub-package. Define:
   ```go
   type Direction int8 // Forward = +1, Backward = -1
   type ZigzagEvent struct {
       Simplex   persistent.Simplex
       Direction Direction // Forward = inclusion (simplex added); Backward = deletion (simplex removed)
       Time      float64   // event timestamp; non-decreasing
   }
   type ZigzagFiltration struct { Events []ZigzagEvent }
   func NewZigzagFiltration(events []ZigzagEvent) (*ZigzagFiltration, error) // validates: closed under faces at every state
   func (z *ZigzagFiltration) StateAt(t float64) []persistent.Simplex // current complex at time t
   func (z *ZigzagFiltration) ToFiltration() (persistent.Filtration, error) // returns nil error iff all-Forward; pin substrate
   ```
   The `ToFiltration` accessor IS the **all-arrows-forward equivalence** that powers Pin #1 below. **Day-1 PR; zero blockers.**

2. **T1 — `topology/zigzag/algorithm.go::ComputeZigzagBarcode(z *ZigzagFiltration, maxDim int) ([]ZigzagBar, error)`** (~280 LOC). Carlsson-de Silva-Morozov 2009 extended-ELZ. Algorithm:
   - Maintain a parenthesisation of the zigzag: at each `Forward` event, push a column onto the matrix-reduction stack; at each `Backward` event, pop the corresponding column and reduce both above and below the popped row.
   - Output: list of `ZigzagBar{Dim, Birth, Death, BirthDirection, DeathDirection}` where the Birth/DeathDirection track which of the four interval-types `[b, d]`, `(b, d]`, `[b, d)`, `(b, d)` the bar corresponds to.
   - Composes verbatim against `persistent.Simplex` and the existing F_2 boundary-column substrate from `topology/persistent/barcode.go:74-100` (the boundary-column builder is reusable; only the reduction loop changes).
   - Complexity: O(m³) worst-case in m = total events; O(m²) on streaming sliding-window typical.
   - Closes the keystone gap. Composes Phase-A `ComputeBarcode` substrate.

3. **T2 — `topology/zigzag/types.go::ZigzagBar` + `topology/zigzag/barcode.go` accessors** (~120 LOC). Zigzag-specific accessors: `IsClosed()` (both endpoints `Forward` direction), `IsHalfOpen()`, `MidPoint()`, `ToStandardBar() (persistent.Bar, bool)` (lossy projection that drops direction-tags; second return is true iff the bar is closed-closed = standard form). The `ToStandardBar()` accessor IS the projection used in **Pin #1** to compare against the existing 1-param barcode pipeline.

4. **T3 — `topology/zigzag/zigzag_test.go::TestZigzag_AllForwardEqualsELZ`** (~80 LOC). **R-MUTUAL-CROSS-VALIDATION 3/3 saturation Pin #1.** Take any standard `Filtration` from `topology/persistent/vr.go::VietorisRipsComplex(points, r, dim)` (across the existing barcode_test.go fixtures: equidistant-4-point H_0, cyclic-4-point H_1, plus 30 random Vietoris-Rips fixtures). Convert each to a zigzag with all events `Forward` via `ZigzagFiltration` constructor on the simplex sequence. Assert:
   - **Path A:** `persistent.ComputeBarcode(filtration, dim)` (existing).
   - **Path B:** `zigzag.ComputeZigzagBarcode(z_forward, dim)` then map each closed-closed bar via `ToStandardBar()`.
   - **Path C:** brute-force matrix-reduction via direct ELZ on the zigzag-as-sparse-matrix representation (test-only, n ≤ 10 simplices).
   All three agree to *exact* equality on Bar.Dim/Birth/Death (no tolerance — F_2 arithmetic is exact). **Day-1 PR. The first R-MUTUAL pin in `topology/zigzag/`.**

5. **T4 — `topology/zigzag/streaming.go::SlidingWindowZigzag(stream <-chan TemporalEvent, w float64) *ZigzagFiltration`** (~150 LOC). Memoli-Wan 2017 streaming zigzag construction. Consumes a stream of `(simplex, time)` events; maintains a sliding window of width `w`; emits a zigzag with `Forward` event per simplex-entry-into-window and `Backward` event per simplex-exit-from-window. **Direct consumer of slot-281 `TemporalGraph`** — once 281 ships `TemporalEdge{u,v,t}` + flag-complex constructor, this primitive turns the time-varying flag complex into a zigzag filtration that T1 reduces to a barcode of "topology that lived during the window". The keystone consumer surface for streaming TDA.

6. **T5 — `topology/zigzag/temporal_graph.go::ZigzagFromTemporalGraph(tg *graph.TemporalGraph, w float64, maxDim int) (*ZigzagFiltration, error)`** (~100 LOC, gates on slot-281). Convenience wrapper: build sliding-window flag complex of `tg`, feed into `SlidingWindowZigzag`. **Direct slot-281 consumer.** Not on the day-1 critical path; ships in week-1.

7. **T6 — `topology/multipersist/types.go::MultiParamFiltration` + `RankInvariant`** (~180 LOC, zero deps). New sub-package for the multi-param axis.
   ```go
   type MultiParamSimplex struct {
       Simplex persistent.Simplex
       Birth   []float64 // entry time in R^d; len(Birth) = d
   }
   type MultiParamFiltration struct {
       Dim      int                // d ≥ 1
       Simplices []MultiParamSimplex
   }
   func (m *MultiParamFiltration) RestrictToLine(origin, direction []float64) (persistent.Filtration, error)
   func (m *MultiParamFiltration) RankInvariant(grid [][]float64) [][]int // ρ(u, v) = rank(H_*(M_u) → H_*(M_v)) on grid points
   ```
   Carlsson-Zomorodian 2009. The `RankInvariant` is the canonical *partial* invariant; the `RestrictToLine` projection IS the basis of fibered barcodes (T8 below). **Day-1 PR for the multi-param axis.**

8. **T7 — `topology/multipersist/fibered.go::FiberedBarcode(m *MultiParamFiltration, origin, direction []float64, maxDim int) ([]persistent.Bar, error)`** (~80 LOC). Lesnick-Wright 2015 RIVET-style 1-D slice through the multi-param plane. Composes `m.RestrictToLine(origin, direction)` then existing `persistent.ComputeBarcode`. **R-MUTUAL-CROSS-VALIDATION 3/3 Pin #2:** when the multi-param filtration is **degenerate-to-1-param** (all simplices share entry time on `direction[1..d-1]`), FiberedBarcode along `direction = e_0` ≡ standard `ComputeBarcode` on the projected 1-D filtration. **Critical regression-pin** — multi-param restricted to a single line ≡ standard 1-param persistence on that filtration.

9. **T8 — `topology/multipersist/presentation.go::MinimalPresentation(m *MultiParamFiltration, maxDim int) (*Presentation2D, error)`** (~350 LOC, **the most expensive primitive in the slot but the SOTA hook**). Lesnick-Wright 2022 efficient O(m³) algorithm. Output: a `Presentation2D{Generators, Relations}` where Generators are pairs `(simplex, R²-position)` and Relations encode the homotopy structure. **R-MUTUAL-CROSS-VALIDATION 3/3 Pin #3:** rank invariant computed from the minimal presentation ≡ rank invariant computed by direct O(grid^2 · m³) rank-on-each-morphism from T6. Three paths: (A) T6 direct rank-on-each-morphism, (B) T8 presentation + reconstruct rank invariant from generators, (C) brute-force rank computation on tiny fixtures (n ≤ 8 simplices). All three agree exactly on integer rank values. **The cutting-edge primitive that puts reality at the 2022-2025 frontier.**

10. **T9 — `topology/multipersist/landscape.go::GRIL(m *MultiParamFiltration, dim, k int, grid [][]float64) [][]float64`** (~220 LOC). Hellmer-Xin 2024 Generalized Rank Invariant Landscape. Computes a vectorised landscape-style feature on the multi-param rank invariant suitable for ML pipelines. Composes T6 RankInvariant + closed-form Hellmer-2024 §3 formula. **The 2024-2025 SOTA ML-on-multi-persistence primitive.**

11. **T10 — `topology/multipersist/decomposition.go::IntervalDecomposition(m *MultiParamFiltration, maxDim int) ([]IntervalSummand, error)`** (~250 LOC). Asashiba-Buchet-Escolar-Nakashima-Yoshiwaki 2019. Returns the unique interval decomposition when it exists; returns `ErrNotIntervalDecomposable` with a witness when not. Required theorem: when M is interval-decomposable, decomposition is unique up to permutation (Krull-Schmidt for multi-param). **Optional consumer of slot-277 BnB** at scale: optimal interval decomposition for non-decomposable modules can be formulated as a combinatorial optimisation (find the interval-decomposable module nearest in interleaving distance).

12. **T11 — `topology/zigzag/ladder.go::CommutativeLadderPersistence(top, bottom *persistent.Filtration, maps []Map) ([]LadderBar, error)`** (~200 LOC). Escolar-Hiraoka 2016 commutative-ladder persistence on 2×n ladders (one filtration on top, one on bottom, vertical maps connecting them). Subsumes both 1-param (only top ladder) and zigzag (alternating verticals). Useful for **paired time-series** (control-vs-treatment topology over time). Lower priority than T0-T9; ships in week-2.

13. **T12 — `topology/zigzag/induced_matching.go::InducedMatching(d1, d2 []ZigzagBar, ...) BarMatching`** (~150 LOC). Bauer-Lesnick 2014. Constructs the induced matching of zigzag barcodes from a morphism of zigzag modules. Used to prove and machine-check **stability theorem**: bottleneck distance between zigzag barcodes ≤ interleaving distance between zigzag modules. **R-MUTUAL pin: stability theorem regression test** — perturb a zigzag input by ε in event-timing, assert d_B(zigzag-barcode-old, zigzag-barcode-new) ≤ ε. Generalises slot-288 T3 stability-regression-test to zigzag.

14. **T13 — `topology/multipersist/stability.go::MultiParamStability_test.go`** (~80 LOC). Cerri-Frosini-Landi 2013. Generate two multi-param filtrations differing by ‖f - g‖_∞ ≤ ε; assert d_∞(rank-invariant-1, rank-invariant-2) ≤ ε. Machine-checkable theorem-pin. Composes T6.

### LOC roll-up

| Tier | Primitive | LOC | Day |
|---|---|---|---|
| T0 | `ZigzagFiltration` + `Direction` types | 120 | 1 |
| T1 | `ComputeZigzagBarcode` (CdSM-2009) | 280 | 1 |
| T2 | `ZigzagBar` accessors | 120 | 1 |
| T3 | All-forward ≡ ELZ regression test | 80 | 1 |
| T4 | `SlidingWindowZigzag` (Memoli-Wan 2017) | 150 | 2-3 |
| T5 | `ZigzagFromTemporalGraph` (slot-281 consumer) | 100 | week-1 |
| T6 | `MultiParamFiltration` + `RankInvariant` | 180 | 2 |
| T7 | `FiberedBarcode` (RIVET-style 1-D slice) | 80 | 2 |
| T8 | `MinimalPresentation` (Lesnick-Wright 2022) | 350 | week-2 |
| T9 | `GRIL` landscape (Hellmer-Xin 2024) | 220 | week-2 |
| T10 | `IntervalDecomposition` (ABENY 2019) | 250 | week-3 |
| T11 | `CommutativeLadderPersistence` (Escolar-Hiraoka) | 200 | week-3 |
| T12 | `InducedMatching` (Bauer-Lesnick 2014) + zigzag stability test | 150 | week-2 |
| T13 | Multi-param stability regression test | 80 | week-2 |
| | **Total** | **~2360 LOC** | |

### Day-1 cheapest PR

**T0 (120) + T1 (280) + T2 (120) + T3 (80) = ~600 LOC + 30 golden vectors.** Single PR. Ships zigzag persistence end-to-end against the existing `persistent.Filtration` substrate. Saturates **R-MUTUAL-CROSS-VALIDATION 3/3 day-1** via Pin #1 (all-forward zigzag ≡ existing ELZ ≡ brute-force-direct-reduction). **Zero external dependency, zero slot-blocker** — composes verbatim against `topology/persistent/barcode.go:74-100` boundary-column substrate. **Multi-param T6+T7 (~260 LOC) ships in PR #2 day-2** with Pin #2 (FiberedBarcode along degenerate-direction ≡ ComputeBarcode on the projected 1-D filtration).

### R-MUTUAL-CROSS-VALIDATION 3/3 pin matrix

Saturate the recent commit pattern (audio onset 6a55bb4, copula×autodiff 365368a, NGramDice 85a80db):

- **Pin #1 (Day-1, T3):** All-forward zigzag ≡ standard 1-param persistence. THREE paths agree exactly on F_2 over n=30 fixtures: (A) `persistent.ComputeBarcode(f, dim)` (existing ELZ-2000), (B) `zigzag.ComputeZigzagBarcode(zigzag-of-f-as-all-forward, dim)` then `ToStandardBar()`, (C) brute-force direct matrix reduction on n ≤ 10 simplices. **First R-MUTUAL pin in `topology/zigzag/`.** No tolerance — F_2 exact equality.
- **Pin #2 (Day-2, T7):** Multi-param restricted to a single line ≡ standard 1-param persistence on that filtration. THREE paths agree: (A) `multipersist.FiberedBarcode(m, origin, e_0)` then standard barcode, (B) `persistent.ComputeBarcode(m.RestrictToLine(origin, e_0))` directly, (C) Lesnick-Wright RIVET-style direct fibered computation. **The Lesnick-Wright RIVET regression-pin.**
- **Pin #3 (Week-2, T8):** Rank invariant computed from minimal presentation ≡ rank invariant computed by direct rank-on-each-morphism. THREE paths: (A) T6 direct, (B) T8 minimal-presentation + reconstruct, (C) brute-force on n ≤ 8 simplices. Integer rank exact agreement. **The cutting-edge multi-persistence pin.**
- **Pin #4 (Week-2, T12):** Zigzag stability theorem (Botnan-Lesnick 2018). Perturb zigzag-input event timings by ε; assert d_B(zigzag-barcode-old, zigzag-barcode-new) ≤ ε across 100 random perturbations. Single-direction inequality machine-checkable. Generalises slot-288 T3 to zigzag.
- **Pin #5 (Week-2, T13):** Multi-param stability theorem (Cerri-Frosini-Landi 2013). Perturb multi-param filter function by ε; assert d_∞(rank-invariant-1, rank-invariant-2) ≤ ε. Machine-checkable.

## Cross-cutting

- **slot 281 (temporal-graphs) ← KEYSTONE CONSUMER.** T4 `SlidingWindowZigzag` + T5 `ZigzagFromTemporalGraph` are the direct streaming-TDA consumer surface. Zigzag on the flag complex of a sliding-window temporal graph IS the time-varying persistent homology of the link stream. Single-shipment hook.
- **slot 280 (network generative models, dynamic SBM) ← Direct consumer.** Dynamic SBM produces a sequence of community partitions over time → zigzag persistence on the community-incidence simplicial complex tracks community birth/death.
- **slot 286 (Reeb / merge-tree) ← Cross-axis pin.** Merge-tree edit distance (286 T10c) is a competing temporal-data tool. Both ship; cross-reference. Future: zigzag of merge trees (Curry-Hang-Mukherjee-Mémoli-Tatakopoulos 2022).
- **slot 287 (Mapper) ← Direct consumer.** Multinerve Mapper has provable convergence to **multi-persistent** Mapper for multi-parameter cover (Munch-Wang 2016). 287 T9 stochastic Mapper consumes T8 fibered barcodes for stability bounds. 287's `FilterFunction` interface generalises trivially to `f: X → R^d` once T6 ships.
- **slot 283 (simplicial complexes) ← Substrate.** T0/T6 prefer SimplexTree storage once 283 ships; today they ship over flat `[]Simplex`. Lazy gate.
- **slot 288 (persistence stats) ← Diagram-stats consumer.** All twelve 288-statistics primitives apply verbatim to zigzag-bars (drop direction-tag projection) and to fibered barcodes (per-slice). **288 = stats layer; 289 = module-and-decomposition layer.** No file overlap.
- **slot 277 (combinatorial optimisation, BnB / ILP) ← Future consumer.** Optimal interval decomposition for non-decomposable multi-param modules (T10) can be formulated as combinatorial optimisation. Optional 277 consumer; not a blocker.
- **slot 142 (topology-missing) / 143 (topology-sota) ← Both punt advanced persistence here.** Slot 289 is the dedicated review.
- **slot 156 (synergy-topology-prob) / 190 (synergy-topology-signal) / 247 (mortar-fem) ← Downstream feature consumers** for streaming-TDA and multi-filter sensor fusion ML pipelines.
- **slot 097-T1 (linalg-missing, Eigvec) ← NO blocker.** All zigzag and multi-param work is F_2 boundary-matrix reduction + arithmetic on R^d-graded simplex lists. No eigendecomposition needed.
- **Pistachio frame-comparison ← Direct consumer.** Currently uses `BottleneckDistance(today, yesterday)` per `topology/persistent/bottleneck.go:21`. Streaming zigzag (T4) elevates this from "compare two snapshots" to "track topology continuously across the entire frame stream" — a *single* zigzag-barcode summarising scene-topology over a window of N frames, instead of N(N-1)/2 pairwise bottleneck distances.
- **aicore TDA-features-for-LLM-eval ← Direct consumer.** Multi-param sublevel-set persistence on LLM loss-landscapes (multi-filter: loss × gradient-norm) → 2-param persistence module → GRIL feature → input to gating / regularisation. Birdal-Lou-Guibas-Simsekli 2021 generalisation.
- **multivariate scientific simulations (climate models, molecular dynamics) ← Direct consumer.** Multi-parameter filtration: temperature × pressure (climate), or potential × density (molecular). T6+T7+T8+T9 are the entire multi-param pipeline. Frontier ML application.

## Sources

- `topology/persistent/vr.go:50` — `Filtration{Simplices, Times}` monotone-1-param substrate. Cannot encode zigzag or multi-param.
- `topology/persistent/barcode.go:60-100` — ELZ-2000 column reduction; the boundary-column substrate is reusable for T1 zigzag-extended-ELZ.
- `topology/persistent/bottleneck.go:50` — W_∞ on persistence diagrams; reusable for zigzag-stability test (T12) and multi-param stability test (T13) after diagonal-stand-in projection.
- `topology/persistent/doc.go:65-76` — Phase-A scope; no mention of zigzag/multi-param. Slot 289 owns the design call.
- `reviews/overnight-400/agents/281-new-temporal-graphs.md:1-66` — sliding-window temporal-graph substrate; the keystone day-1 consumer of T4-T5.
- `reviews/overnight-400/agents/283-new-simplicial-complexes.md` — SimplexTree gate (lazy).
- `reviews/overnight-400/agents/286-new-reeb.md` — merge-tree edit distance cross-axis tool; zigzag is the more-general competitor.
- `reviews/overnight-400/agents/287-new-mapper.md:21-42` — multinerve Mapper convergence to multi-persistent Mapper; direct consumer of T6-T9.
- `reviews/overnight-400/agents/288-new-persistence-stats.md` — 12 diagram-statistics primitives; apply verbatim to zigzag/fibered barcodes.
- `reviews/overnight-400/agents/277-new-copo.md` — BnB substrate; optional T10 consumer.
- Carlsson, de Silva (2010). *Found. Comput. Math.* 10:367. Zigzag persistence.
- Carlsson, de Silva, Morozov (2009). *SoCG* 247-256. Zigzag persistent homology and real-valued functions. **THE T1 algorithm.**
- Maria, Oudot (2014). *SoCG*. Zigzag persistence via reflections and transpositions.
- Milosavljević, Morozov, Skraba (2011). *SoCG*. Zigzag in matrix multiplication time.
- Memoli, Wan (2017). arXiv:1706.04571. Streaming zigzag.
- Carlsson, Zomorodian (2009). *Discrete Comput. Geom.* 42:71. The theory of multidimensional persistence. **Establishes no-complete-discrete-invariant for d ≥ 2.**
- Lesnick, Wright (2015). arXiv:1512.00180. Interactive visualization of 2-D persistence modules (RIVET).
- Lesnick, Wright (2022). *Found. Comput. Math.* 22:1093. Computing minimal presentations and bigraded Betti numbers of 2-parameter persistent homology.
- Cerri, Frosini, Landi (2013). *Math. Methods Appl. Sci.* 36:1543. Betti numbers in multi-D persistent homology are stable functions.
- Bauer, Lesnick (2014). *J. Comput. Geom.* 6:162. Induced matchings and the algebraic stability of persistence barcodes.
- Bauer, Lesnick (2020). *J. Algebra* 547:84. Persistence diagrams as diagrams: a categorification of the stability theorem.
- Bubenik, Scott (2014). *Discrete Comput. Geom.* 51:600. Categorification of persistent homology (generalised persistence over arbitrary posets).
- Escolar, Hiraoka (2016). *Adv. Math.* 290:1015. Persistence modules on commutative ladders of finite type.
- Hellmer, Spreer (2024). arXiv:2304.04970. Generalized persistence diagrams for persistence modules over posets.
- Xin, Hellmer (2024). GRIL: A 2-parameter persistence based vectorization for machine learning.
- Asashiba, Buchet, Escolar, Nakashima, Yoshiwaki (2019). *J. Comput. Geom.* 10:201. On approximation of 2D persistence modules by interval-decomposables.
- Botnan, Lesnick (2018). arXiv:1809.05151. Algebraic stability of zigzag persistence modules.
- Dey, Hou (2021). *SoCG*. Computing zigzag persistence on graphs in near-linear time.
- Crawley-Boevey (2015). *J. Algebra Appl.* 14:1550066. Decomposition of pointwise finite-dimensional persistence modules.
- Mileyko, Mukherjee, Harer (2011). *Inverse Problems* 27:124007. Probability measures on the space of persistence diagrams (extends to zigzag PD-space).
- Curry, Hang, Mukherjee, Mémoli, Tatakopoulos (2022). Tracking dynamics of persistence via zigzag of merge trees.
- Dionysus2 (Morozov BSD-3 C++) — gold-standard zigzag impl. **Pure-Go MIT zero-dep ABSENT.**
- RIVET (Lesnick-Wright GPL-3 C++) — gold-standard 2-param. **Pure-Go MIT zero-dep ABSENT.**
- GUDHI 3.10 (Apache-2/MIT C++/Python) — Maria-Oudot zigzag, no multi-param. ABSENT.
- PHAT (BSD-3 C++) — fast 1-param, no zigzag. ABSENT.
- scikit-tda persim 0.4 (Apache-2 Python). ABSENT for both.

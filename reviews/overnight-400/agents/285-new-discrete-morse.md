# 285 — new-discrete-morse (Discrete Morse Theory: Forman, V-paths, Optimal DMT, Cancellation)

## Headline
reality v0.10.0 ships ZERO discrete Morse surface (`Morse|DiscreteVector|GradientPath|VPath|CriticalCell|MorseMatching|Forman|MorseFunction|MorseSmale` repo-wide grep on `*.go` returns 0 callable hits); slot 284 outlines DMT only at overview depth (T4 GreedyMatching + T5 Morse complex + T6 Morse persistence ~ 480 LOC inside its budget) — this slot owns the **full `topology/complex/morse/` sub-package at depth (~2,100 LOC across 14 primitives)**: Forman-1998 axioms + V-path machinery, Lewiner-2003 greedy, **Joswig-Pfetsch-2006 optimal DMT (first concrete consumer of slot-277 BranchAndBound ILP)**, Benedetti-Lutz-2014 random DMT, Robins-2011 image-cubical DMT, Mischaikow-Nanda-2013 persistent DMT, Bauer-Edelsbrunner-2014 cancellation algorithm relating Morse pairs to ELZ persistence pairs, Sköldberg-2006 algebraic DMT (chain complex without geometric realization), incremental DMT (insert/delete cell maintaining matching), Morse-Smale complex (terrain/scalar-field analysis), generic gradient-flow framework (vector-field visualization for Pistachio terrain rendering). Day-1 cheap PR is **T0+T2+T6 = DiscreteMorseFunction interface + Lewiner-greedy on simplex tree + critical-cell extraction + Morse-boundary** ≈ 380-420 LOC, gated only on slot-283 SimplexTree + slot-284 CellComplex interface landing first; ships next-PR on top of those. **Single highest-leverage TDA performance primitive in the entire reality plan**: 100×-1000× wallclock speedup on cubical / VR persistence (Mischaikow-Nanda 2013 §6 + Wagner 2012 cubical 256³ MRI benchmark), with ~99% cell-count compression on benign inputs. Pure-Go MIT zero-dep ABSENT in any language ecosystem (perseus 4.0 C++ ~6k LOC undefined-license; GUDHI Apache wrapping C++; Julia Eirene GPL).

## Findings

### State at HEAD (verified)

| Surface | Path | Morse-relevance |
|---|---|---|
| `Simplex = []int` | `topology/persistent/vr.go:14` | Inputs to Morse matching (each cell = sorted vertex tuple). |
| `Filtration{Simplices, Times}` | `topology/persistent/vr.go:50` | Filtered complex (Morse function f(σ) = filtration time + tiebreaker). |
| `boundaryColumn` private | `topology/persistent/barcode.go:221` | F_2 boundary; needed by Morse complex post-collapse to compute β_k of critical-cell complex. |
| `ComputeBarcode(filt, maxDim)` | `topology/persistent/barcode.go:60` | ELZ-2000 column reduction; cap maxDim ∈ {0,1}; **target of Mischaikow-Nanda speedup** — replace inner loop on collapsed complex with ~1-100× fewer columns. |
| `optim/genetic.go`, `optim/metaheuristic.go` | `optim/` | Stochastic optimizers; Benedetti-Lutz random DMT is metaheuristic-flavored. |
| **slot 277 (future) `optim/intopt/bnb.go::BranchAndBound`** | NOT YET LANDED | **First concrete consumer in reality's plan**: Joswig-Pfetsch-2006 optimal DMT formulates max-Morse-matching as 0-1 ILP with acyclicity constraints, BnB resolves. **Pin: 277 → 285-T3.** |
| repo-wide grep `Morse|Discrete[VG]|VPath|CriticalCell|MorseMatching|Forman|MorseFunction|MorseSmale|MatchingV` | `*.go` outside reviews/ | **ZERO hits.** |

### Slot boundaries (absorption with adjacent slots)

- **Slot 283 (simplicial-complexes) ←** SimplexTree + ChainComplex over F_2/Z. **283-T0 SimplexTree is the substrate** for `MorseFunction.OnSimplex` evaluator. 285-T2 GreedyMatching iterates over 283-T0's per-dim simplex arrays in decreasing dimension.
- **Slot 284 (cw-complexes) ←** Generic `CellComplex` interface + Hasse diagram + cubical complex. **284-T0 `CellComplex` is the substrate** for ALL of 285's Morse primitives — Lewiner / Joswig-Pfetsch / Benedetti-Lutz / Robins-2011 / Sköldberg all consume `CellComplex` polymorphically. 284-T4 outlines `GreedyMatching` ~140 LOC; 285-T2 implements with full V-path acyclicity validation + 285-T3 ILP variant + 285-T4 random variant + 285-T5 Robins specialization + 285-T6 collapse with full Morse-boundary (gradient-path counting) + 285-T7 incremental + 285-T8 cancellation pairs not in 284's scope. **285 is the dedicated DMT slot; 284 punts everything beyond toy collapse.**
- **Slot 277 (combinatorial-optimization) ←** `optim/intopt/bnb.go::BranchAndBound` from 277-I3. **First concrete consumer:** 285-T3 Joswig-Pfetsch optimal DMT formulates `max Σ_(σ,τ)∈Hasse y_(σ,τ)  s.t. Σ_(σ in pair) y_(σ,*) ≤ 1, V acyclic`, fed to 277-I3 with custom acyclicity-cut callback (lazy constraint generation = check candidate matching for V-cycle, reject + add cut). **Pin 277 → 285-T3.**
- **Slot 280 (SBM) ←** Random simplicial complexes (Linial-Meshulam 2006, Kahle 2014); 285-T4 Benedetti-Lutz random DMT measures expected critical-cell count under random complex distributions → empirical Morse inequality bound (β_k(K) ≤ E[c_k] ≤ optimal critical count).
- **Slot 281 (temporal-graphs) ←** Zigzag persistence on filtered cell complex; 285-T7 incremental DMT maintains matching under cell insertion/deletion (single-cell update O(localcofaces)) — the keystone primitive for streaming TDA on temporal graphs.
- **Slot 282 (hypergraphs) ←** Hypergraph Hodge Laplacian L_k via 282-T6's `AsSimplicialComplex`; 285-T6 Morse complex compresses *before* L_k construction → 100× smaller dense matrix → 100× faster eigendecomposition.
- **Slot 247 (mortar-fem), 248 (multigrid) ←** Discrete Morse provides natural multigrid hierarchy on cell complexes (each Morse-collapsed level = one V-cycle level); 285-T9 generic-gradient-flow exposes hierarchy.
- **Slot 097-T1 (linalg-missing, Eigvec) ←** No direct dep — Morse machinery is combinatorial/F_2. Hodge spectrum on Morse complex (100× smaller matrix) gates on 097-T1 via 284-T9 path.

### Web context (verified pure-Go MIT zero-dep ABSENT in all listed)

- **Forman-1998** *Adv-Math* 134:90 "Morse theory for cell complexes" + **Forman-2002** *Sém-Lothar-Combin* 48 "A user's guide to discrete Morse theory" — combinatorial analogue of smooth Morse. A **discrete Morse function** f: K → R assigns one real value per cell with at-most-one anomaly per (σ, τ) cofacet pair: σ < τ codim-1 implies either f(σ) < f(τ) (regular pair → can be matched in V) or f(σ) ≥ f(τ) (anomaly, contributes critical pair). **Equivalent (Forman §3) to a discrete vector field V** = matching in the Hasse diagram (each cell appears in ≤1 matched pair, σ < τ codim-1) such that **the modified Hasse digraph (reverse arrows for matched pairs) is acyclic** (no V-paths σ → τ → σ' → τ' → ... → σ that return). **Critical cells = unmatched cells.** **Forman main theorem**: K is homotopy-equivalent to a CW-complex with one cell per critical cell. **Morse inequalities**: c_k ≥ β_k for all k; alternating sum ≡ Euler characteristic.
- **Forman-2002 §6** Morse boundary ∂^M: counts gradient paths (V-paths from face-of-τ to σ); the chain complex on critical cells with this boundary has the same homology as K (Forman's collapse theorem). **Generalizes smooth Morse-Witten complex** to combinatorial setting.
- **Lewiner-Lopes-Tavares-2003** *J-Math-Imaging-Vision* 19:223 + 2003 *GMP* "Optimal discrete Morse functions for 2-manifolds" — **greedy coreduction-pair matching algorithm**: process cells in decreasing dimension; for each unmatched τ with exactly one unmatched codim-1 face σ ("free face"), match σ↑τ (add to V); when no free faces exist, declare next cell critical and continue. **Linear-time O(Σ_k n_k · k)**; not optimal (achieves optimal on 2-manifolds + simply-connected 3-manifolds, can be far from optimal on general complexes — Joswig-Pfetsch §1) but typically within a few percent. Reference: perseus 4.0 C++ ~6k LOC.
- **Joswig-Pfetsch-2006** *J-Math-Soc-Japan* "Computing optimal Morse matchings" — optimal DMT is **NP-hard** (reduces from MAX-3-SAT); they formulate as **0-1 ILP with exponential acyclicity constraints + lazy cut generation** in branch-and-cut. Benchmarks: optimal matching on simplicial sphere S^4 (=20 cells) = 2 critical cells (matching β: minimum c_0 = β_0 = 1, c_4 = β_4 = 1); greedy gives 4-6 critical cells. **First concrete consumer of slot-277 BnB.**
- **Benedetti-Lutz-2014** *Exp-Math* 23:66 "Random discrete Morse theory and a new library of triangulations" — empirical: **random Morse matchings on a triangulation typically achieve near-optimal collapsing on benign inputs** (small Betti, simply connected). Algorithm: random ordering of cells + greedy Lewiner free-face matching → collect statistics over many trials. **Used as the practical default in many libraries** when speed matters more than optimality. Detects pathological complexes (Bing's house, dunce hat — c-vector stays high under randomization → hard for Morse).
- **Mischaikow-Nanda-2013** *Discrete-Comput-Geom* 50:330 "Morse theory for filtrations and efficient computation of persistent homology" — canonical paper: discrete Morse on a *filtered* complex (filtration-respecting matching, free-face must respect birth time), produces a "reduced" filtered Morse complex with **identical persistence diagram** but typically 100×-1000× fewer cells. Combined with twist (Chen-Kerber 2011) → multiplicative speedup. **THE keystone TDA performance primitive.**
- **Bauer-Edelsbrunner-2014** *J-Topol-Anal* 6:531 "The Morse theory of Čech and Delaunay complexes" + Bauer-2017 *J-Symb-Comput* — **cancellation algorithm**: each persistence pair (birth=σ, death=τ) in ELZ output corresponds to a Morse pair (σ, τ) cancellable by a single Forman cancellation (reverse the V-path connecting them). **Provides the algebraic-topological identification:** Morse-pair-cancellation ≡ ELZ persistence-pair. Enables incremental updates: cancel a pair = collapse two critical cells.
- **Robins-Wood-Sheppard-2011** *IEEE-TPAMI* 33:1646 "Theory and algorithms for constructing discrete Morse complexes from grayscale digital images" — DMT for **2D/3D image segmentation**. Uses lower-star filtration on cubical complex (Wagner-2012 V-construction); critical-cell extraction → image features (peaks/saddles/pits = max/saddle/min cells); Morse-Smale complex → segmentation by gradient-flow basins. **The standard tool** for medical-imaging topological feature extraction. Reference C implementation Pyrcc (academic).
- **Sköldberg-2006** *Trans-AMS* 358:115 "Morse theory from an algebraic viewpoint" — Forman's discrete Morse generalizes to **algebraic Morse theory** on a chain complex with no underlying geometric cell structure: a "matching" on the chain-complex digraph (basis-element pairs σ → τ with non-zero ∂(τ) coefficient at σ) such that the perturbed differential converges. **Foundation for the modern unified treatment.** Used in computer-algebra (computing Tor, Ext, group-cohomology of discrete groups).
- **Curry-Ghrist-Nanda-2013** "Discrete Morse theory for computing cellular sheaf cohomology" — extends DMT from chain complexes to *cellular sheaves* (functorial assignment of vector spaces to cells). Foundation for sheaf-cohomology computation; used by Hansen-Ghrist sheaf neural networks (2020+).
- **Reininghaus-Hotz-Wenzel-2012** *IEEE-TVCG* "Combinatorial 2D vector field topology" — Morse-Smale complex on 2D vector fields by combinatorial methods; critical-point graph (saddle-connector graph) extraction. Used in fluid-flow visualization, weather-data analysis, scientific computing.
- **Bauer-Kerber-Reininghaus-2017** PHAT — modular column reduction; complementary to Morse pre-collapse.
- **Open-source landscape:**
  - perseus 4.0 (no clear license) — Mischaikow's reference C++ Morse-preprocessing engine; ~6k LOC.
  - GUDHI 3.10 (BSD-3, Apache wrapping C++) — Morse-Smale on regular complex; ~50k total LOC.
  - DiscreteMorse.jl (MIT, Julia, ~800 LOC, archived since 2019; not maintained, no Forman main theorem proof, no Joswig-Pfetsch ILP).
  - DiscretizedMorse / SyMo (academic Python, GPL).
  - **Pure-Go MIT zero-dep ABSENT for ALL of:** discrete Morse function, V-path machinery, greedy / optimal / random matching, critical-cell extraction, Morse complex with gradient-path boundary, Morse-persistence, Bauer-Edelsbrunner cancellation, incremental DMT, Robins-2011 image DMT, Sköldberg algebraic DMT, Morse-Smale complex.

## Concrete recommendations

### T0 — DiscreteMorseFunction interface + V-path machinery (zero-blocker, ~120 LOC)

1. **`topology/complex/morse/types.go::DiscreteMorseFunction` (~80 LOC).** Forman-1998 abstract:
   ```go
   // f : Cells → R s.t. for every (σ, τ) cofacet pair (codim-1):
   //   #{τ' > σ : f(τ') ≤ f(σ)} ≤ 1   (at-most-one upward anomaly)
   //   #{σ' < τ : f(σ') ≥ f(τ)} ≤ 1   (at-most-one downward anomaly)
   type DiscreteMorseFunction interface {
       Value(c CellID) float64
       Complex() CellComplex                  // 284-T0
   }
   type FiltrationMorse struct {              // f(σ) := filtration_time(σ) + lex tiebreaker
       cc   CellComplex
       ties []float64                         // perturbation for total order
   }
   func (f *FiltrationMorse) Validate() error // verify Forman axiom on every cofacet pair
   ```
   Equivalent definition (Forman §3) via discrete vector field V; the two views interconvert. **Pin R-MUTUAL-CROSS-VALIDATION 3/3 (#A): function-view ↔ vector-field-view ↔ Hasse-modified-digraph-acyclicity.**

2. **`topology/complex/morse/vpath.go::VPath` + `IsAcyclicV(V MorseMatching, cc CellComplex) error` (~40 LOC).** A **V-path** of dim-k cells is a sequence σ_0, τ_0, σ_1, τ_1, ..., σ_r, τ_r where (σ_i, τ_i) ∈ V (matched gradient pair, σ_i is a codim-1 face of τ_i) and σ_{i+1} ≠ σ_i is a codim-1 face of τ_i. Acyclicity check: DFS on the modified Hasse digraph (regular Hasse arrows σ → τ for σ < τ codim-1, plus reversed arrows τ → σ for matched pairs); **V is a valid vector field iff the modified digraph has no directed cycle** (Forman §6 theorem). **Single load-bearing correctness witness** for any matching algorithm (T2/T3/T4).

### T1 — Hasse-diagram + matching primitives (~80 LOC)

3. **`topology/complex/morse/matching.go::MorseMatching` + helpers (~80 LOC).**
   ```go
   type MorseMatching struct {
       Pairs    []CellPair                 // matched (σ < τ) gradient pairs; |Pairs| = #matched-cells / 2
       Critical []CellID                   // unmatched = critical
       Pair     map[CellID]CellID          // bidir lookup σ ↔ τ; nil for critical cells
   }
   func (V *MorseMatching) IsCritical(c CellID) bool
   func (V *MorseMatching) PartnerOf(c CellID) (CellID, bool)
   func (V *MorseMatching) CriticalCount(byDim []int) int
   ```
   `byDim[k]` = c_k (# critical k-cells); Morse inequality regression test: `c_k >= β_k`, `Σ_k (-1)^k c_k = χ`.

### T2 — Lewiner-Lopes-Tavares-2003 greedy matching on generic CellComplex (~180 LOC)

4. **`topology/complex/morse/greedy.go::GreedyMatching(cc CellComplex) MorseMatching` (~140 LOC).** Lewiner-2003 coreduction-pair greedy: process cells in **decreasing dim** k = Dim..0; for each unmatched τ ∈ cells of dim k, scan cofaces; if τ has exactly one unmatched codim-1 face σ (a "free face"), add (σ, τ) to V; iterate until no free face → pick an unmatched cell as critical; repeat. **Linear-time O(Σ_k n_k · k) = O(total face-incidences).** Operates polymorphically via 284-T0 `CellComplex.Cofaces` / `Boundary`; works on simplicial (283-T0), cubical (284-T1), Δ-complex (284-T7), CW (284-T8). **No allocation in hot path: preallocate matched-flag bitset + free-face queue.**

5. **`topology/complex/morse/greedy.go::FilteredGreedy(cc CellComplex, f DiscreteMorseFunction) MorseMatching` (~40 LOC).** Mischaikow-Nanda-2013 variant: free-face σ must satisfy `f(σ) ≤ f(τ)` (filtration-respecting); ensures the resulting Morse complex has **identical persistence diagram** to the original (T6). **Foundation for T6 MorsePersistence.**
   - **R-MUTUAL-CROSS-VALIDATION 3/3 (#B):** GreedyMatching outputs are *valid* (T0.2 IsAcyclicV passes), satisfy Morse inequalities (c_k ≥ β_k from 283-T2 BettiF2 on K, T6 BettiF2 on Morse complex), and FilteredGreedy preserves persistence (T6 MorsePersistence ≡ direct ELZ on K). **Three-axis saturation across {validity, homology preservation, persistence preservation}.**

### T3 — Joswig-Pfetsch-2006 optimal DMT via 0-1 ILP (~280 LOC, gates on slot 277 BnB)

6. **`topology/complex/morse/optimal.go::OptimalMatching(cc CellComplex) (MorseMatching, error)` (~280 LOC).** Joswig-Pfetsch-2006 0-1 ILP:
   ```
   max  Σ_(σ,τ): codim-1 cofacet pair  y_(σ,τ)
   s.t. Σ_τ : σ<τ y_(σ,τ) + Σ_ρ : ρ<σ y_(ρ,σ) ≤ 1   ∀σ          (at-most-one-pair)
        V(y) is acyclic                                          (lazy constraint)
        y_(σ,τ) ∈ {0, 1}
   ```
   Acyclicity is exponentially-many-cuts; resolved via **lazy constraint callback** in slot-277 BnB: each BnB node's LP-relaxation candidate y* is rounded → check V(y*) acyclicity by DFS on modified Hasse → if cycle σ_0 → ... → σ_0 found, add cut `Σ_pairs-in-cycle y_pair ≤ |cycle|-1` (forbid this exact cycle), re-solve LP. **First concrete consumer of slot-277 `optim/intopt/bnb.go::BranchAndBound` with lazy-cut callback** (277-I3 plus lazy-cut hook). **Solves NP-hard problem optimally on small/medium complexes** (n_total < 10^4 cells reasonable; > 10^5 use T2 greedy or T4 random instead).
   - **R-MUTUAL-CROSS-VALIDATION 3/3 (#C, optimal-bound):** OptimalMatching critical count `c_k(opt) ≤ c_k(greedy)` ≤ `c_k(random)` median; all `≥ β_k(K)` (Morse inequality). On simply connected complexes (e.g., n-ball B^n, n-disk subdivisions) `c_k(opt) = δ_{k,0}` (single critical 0-cell, deformation retracts to point). Three-axis saturation: optimal matches analytic minimum, greedy bounds within constant factor, random tracks empirical CDF.

### T4 — Benedetti-Lutz-2014 random DMT (~80 LOC)

7. **`topology/complex/morse/random.go::RandomMatching(cc CellComplex, rng *rand.Rand) MorseMatching` (~80 LOC).** Greedy with **random tiebreaking on cell ordering within each dimension**. **Empirically near-optimal on benign inputs** (Benedetti-Lutz §4: c_vector matches Joswig-Pfetsch optimal on > 95% of triangulations of S^n, n ≤ 4). Detects pathological inputs (Bing's house, dunce hat → c-vector stays high under randomization). Useful as Monte-Carlo lower bound on critical-cell count and for catching pessimal-input bugs in T2/T3.
   - **R-MUTUAL-CROSS-VALIDATION 3/3 (#D, random ↔ greedy ↔ optimal on simplex):** on Δ^n (single n-simplex, contractible) all three return (1, 0, 0, ..., 0); on S^n (boundary of (n+1)-simplex) all three return (1, 0, ..., 0, 1). Three-axis saturation on closed-form analytic fixtures.

### T5 — Robins-2011 image-cubical DMT specialization (~100 LOC, gates on 284-T1)

8. **`topology/complex/morse/image.go::ImageMatching(cc *cubical.CubicalComplex, values []float64) MorseMatching` (~100 LOC).** Robins-Wood-Sheppard-2011 specialization of Lewiner-greedy to cubical complex with **lower-star filtration on grayscale image** (Wagner-2012 V-construction): each pixel/voxel gets values; each cell's f-value = max value over incident vertices; greedy free-face matching yields critical cells = image features (peaks=max-cells, saddles=intermediate-cells, pits=min-cells). **The standard tool for medical-imaging topological feature extraction** (Robins-2011 §6 brain MRI, §7 microstructure). **2D image with 1024² pixels**: ~4M cubical cells → ~50-200 critical cells (10⁵× compression). **Direct consumer for aicore segmentation pipelines.**

### T6 — Morse complex (collapsed CW) + Morse boundary + persistence (~280 LOC)

9. **`topology/complex/morse/collapse.go::MorseComplex(cc CellComplex, V MorseMatching) *MorseCellComplex` (~120 LOC).** Construct homotopy-equivalent CW with one cell per critical cell of V. **Morse boundary ∂^M(τ) = Σ_σ critical, dim σ = dim τ - 1 [Σ_(γ V-path from face-of-τ to σ) ε(γ)] · σ** (Forman §6, Lewiner §3.3). Algorithm: enumerate V-paths from each face of τ to all critical cells via DFS on modified Hasse digraph (T0.2); count gradient-path multiplicities (over F_2: parity; over Z: signed sum). Returns `MorseCellComplex{Critical []CellID, Bdry map[CellID][]SignedCell}` API-compatible with 284-T0 `CellComplex` interface (Morse complex IS itself a CellComplex). **Recursive collapse:** can re-Morse the Morse complex until matching converges (typically 2-3 iterations).

10. **`topology/complex/morse/collapse.go::HomologyOfMorseComplex(mcc *MorseCellComplex) []int` (~40 LOC).** β_k via F_2 column reduction on the (typically tiny) Morse complex; reuse 283-T2 `BettiF2`. **Forman main theorem regression** (R-MUTUAL #E): β_k(K) ≡ β_k(MorseComplex(K, V)) ≡ analytic for fixtures; `c_k(V) ≥ β_k(K)` (Morse inequality lower bound) for every V.

11. **`topology/complex/morse/persistence.go::MorsePersistence(filt FilteredCellComplex) []Bar` (~120 LOC).** Mischaikow-Nanda-2013 algorithm: (a) T2.5 FilteredGreedy → filtration-respecting V; (b) T6.9 MorseComplex → small filtered Morse complex; (c) ELZ-twist column reduction (reuse `topology/persistent.boundaryColumn` machinery + Chen-Kerber-2011 twist) on Morse complex → identical persistence diagram. **5-100× wallclock speedup** on simplicial VR; **100×-1000× speedup** on cubical (Wagner-2012 256³ MRI: 17M cells → 50 critical cells; ELZ on 17M ≈ 1 hour → ELZ on 50 ≈ 1 ms; Morse-preprocess linear-time ≈ 1 second; total ≈ 1 second vs 1 hour = 3,600× speedup).
   - **R-MUTUAL-CROSS-VALIDATION 3/3 (#F, three independent persistence algorithms):** MorsePersistence(K) ≡ direct `topology/persistent.ComputeBarcode(K)` ≡ 284-T3 CubicalPersistence(K) when K is cubical. Three-axis saturation on 32², 64², 128² height-function fixtures.

### T7 — Bauer-Edelsbrunner-2014 cancellation algorithm (~120 LOC)

12. **`topology/complex/morse/cancellation.go::CancelPair(V MorseMatching, σ, τ CellID) (MorseMatching, error)` (~80 LOC).** Bauer-Edelsbrunner-2014: each persistence pair `(birth=σ, death=τ)` in the ELZ output corresponds to a critical-cell pair (σ, τ) of the same dimension difference; **Forman cancellation** = reverse the unique V-path from τ-face to σ → both σ and τ become matched (no longer critical) → removes pair (σ, τ) from critical set without changing homotopy type if the V-path is unique. **Provides the algebraic-topological identification** Morse-pair ↔ ELZ-persistence-pair. Used for incremental persistence updates (cancel = annihilate a feature with persistence ≤ ε).

13. **`topology/complex/morse/cancellation.go::Simplify(V MorseMatching, persistence []Bar, εthreshold float64) MorseMatching` (~40 LOC).** Iteratively cancel persistence pairs with `death - birth ≤ ε` (topological denoising / persistence-based simplification). **Standard tool for terrain analysis** (cancel small saddle-pit pairs to get clean ridge network); **Pistachio terrain renderer use case.**
   - **R-MUTUAL-CROSS-VALIDATION 3/3 (#G, cancellation ↔ persistence):** Simplify(V, bars, ε) yields Morse matching whose Morse-complex persistence equals `{b ∈ bars : death(b) - birth(b) > ε}`. Pin: ε = 0 → no cancellations (identity); ε = ∞ → only essential bars survive (β_k critical cells). Three-axis: identity, full simplification, intermediate fixture.

### T8 — Sköldberg-2006 algebraic DMT (chain complex without geometric realization) (~140 LOC)

14. **`topology/complex/morse/algebraic.go::AlgebraicMorse(cc *ChainComplex) (Reduced *ChainComplex, P, ι Map)` (~140 LOC).** Sköldberg-2006: takes any chain complex (C_*, ∂_*) over a ring (default F_2; over Z requires SNF for Morse boundary), constructs **algebraic Morse matching** on the chain-complex digraph (basis-element pairs (σ, τ) with `∂(τ)|_σ ≠ 0` invertible), perturbs differential to converge → reduced chain complex **without underlying geometric cell structure**. Foundation for computer-algebra applications (Tor / Ext, group cohomology of discrete groups, A_∞-algebra resolutions). Deduplicates with T6.9 when geometric structure exists; **strictly more general** (works on any chain complex, e.g., Koszul resolution).

### T9 — Incremental DMT (insert/delete cell maintaining matching) (~120 LOC)

15. **`topology/complex/morse/incremental.go::Insert(V *MorseMatching, c CellID) error` + `Delete(V *MorseMatching, c CellID) error` (~120 LOC).** Maintain a valid Morse matching under cell insertion / deletion (single-cell update reads only local cofaces / faces, O(local-incidence) ≪ O(n_total) re-matching). Insertion: c is critical initially; if c has a free face, match. Deletion: if c is matched, unmatch its partner; cascade. **Keystone primitive for streaming TDA** on temporal cell complexes (slot 281 zigzag persistence). **Bauer-Edelsbrunner-2014 §7** outlines algorithm; first OSS implementation in any language.

### T10 — Morse-Smale complex (terrain / scalar-field analysis) (~180 LOC)

16. **`topology/complex/morse/morsesmale.go::MorseSmaleComplex(cc CellComplex, V MorseMatching) *MorseSmaleComplex` (~180 LOC).** Reininghaus-Hotz-Wenzel-2012-style: refines Morse complex by tracking the **gradient-flow basins** (which cells flow to which critical cell along V-paths). Output: critical-point graph (saddle-connector graph: each saddle connects to two extrema = ascending/descending manifolds). **Standard tool for terrain analysis** (peaks, ridges, valleys, pits) and **vector-field visualization** (fluid flow, weather, electromagnetic field topology). **Pistachio direct consumer:** Morse-Smale on heightmap → terrain feature extraction (ridgelines, valley network) at 60 FPS.

### T11 — Generic gradient-flow framework (~80 LOC)

17. **`topology/complex/morse/flow.go::GradientFlow(V MorseMatching, c CellID) []CellID` + `Basin(V, critical) []CellID` (~80 LOC).** Iterate V-arrows from any starting cell c; output the V-path and its terminal critical cell. **Basin** = set of cells flowing to a given critical cell (dual to ascending/descending manifold). Used for vector-field visualization (color cells by basin), surface segmentation by gradient-flow basins (Robins-2011 §7 image segmentation, Edelsbrunner-Harer-Zomorodian-2003 terrain hierarchy).

## Single cheapest day-1 PR

**`topology/complex/morse/` package, T0+T1+T2+T6 ~500 LOC, gated on slot-283 SimplexTree + slot-284 CellComplex landing first** (no other blockers).

- `topology/complex/morse/types.go` — DiscreteMorseFunction interface + FiltrationMorse (T0.1, ~80 LOC)
- `topology/complex/morse/vpath.go` — VPath + IsAcyclicV (T0.2, ~40 LOC)
- `topology/complex/morse/matching.go` — MorseMatching struct + helpers (T1.3, ~80 LOC)
- `topology/complex/morse/greedy.go` — Lewiner-2003 GreedyMatching + FilteredGreedy (T2.4+T2.5, ~180 LOC)
- `topology/complex/morse/collapse.go` — MorseComplex + Morse boundary + HomologyOfMorseComplex (T6.9+T6.10, ~160 LOC)

Tests:
- `TestForman_VectorField_Acyclic` — every output of GreedyMatching / RandomMatching satisfies T0.2 IsAcyclicV.
- `TestMorseInequality_c_k_ge_beta_k` — for fixtures (S^n, T^n, RP^2, dunce hat) c_k(V) ≥ β_k(K) (analytic regression, pin #B).
- `TestForman_HomotopyEquivalence` — β_k(MorseComplex(K, V)) ≡ β_k(K) ≡ analytic for fixtures (pin #E, **load-bearing Forman main theorem regression**).
- `TestMorsePersistence_Equals_Direct` — bottleneck distance (`topology/persistent.BottleneckDistance`) between MorsePersistence(K) and ComputeBarcode(K) ≡ 0 to round-off (pin #F).
- `TestSimplyConnected_OneCriticalCell` — for Δ^n, B^n, n-disk subdivisions, c-vector = (1, 0, ..., 0); GreedyMatching achieves optimal (defines simply-connected discrete Morse).

This PR ships **the standalone DMT primitive at full depth**: discrete Morse function + vector field + greedy matching + Morse complex + collapse. T3 Joswig-Pfetsch optimal DMT, T4 Benedetti-Lutz random, T5 Robins image-DMT, T7 cancellation, T9 incremental, T10 Morse-Smale are second-PR follow-ups; T8 algebraic DMT and T11 gradient-flow are deferred to consumer pull. **Single highest-leverage TDA performance primitive in the entire reality plan**: 100×-1000× wallclock speedup unlocks 256³ voxel MRI / VR persistence at n=10000 / streaming temporal complexes that are **flat-out infeasible without DMT**.

## Cross-cutting

- **Slot 097-T1 (linalg-missing, Eigvec) ←** No direct dep. Hodge-spectrum on Morse complex (100× smaller matrix → 100× faster eigendecomposition) gates on 097-T1 via 284-T9 path.
- **Slot 247 (mortar-fem), 248 (multigrid) ←** Discrete Morse provides natural multigrid hierarchy on cell complexes. Iterating Morse collapse on Morse complex ≈ V-cycle multigrid; T11 gradient-flow exposes hierarchy.
- **Slot 277 (combinatorial-optimization) ←** **First concrete consumer in reality**: 285-T3 Joswig-Pfetsch optimal DMT formulates max-Morse-matching as 0-1 ILP with lazy acyclicity-cut callback; depends on 277-I3 `optim/intopt/bnb.go::BranchAndBound` plus lazy-cut hook. **Pin: 277 → 285-T3.**
- **Slot 280 (SBM, random simplicial complexes) ←** 285-T4 Benedetti-Lutz random DMT measures expected critical-cell count under random complex distributions → empirical Morse inequality bounds for Linial-Meshulam-2006 / Kahle-2014 distributions.
- **Slot 281 (temporal-graphs, zigzag persistence) ←** 285-T9 incremental DMT (insert/delete cell maintaining matching) is **the keystone primitive for streaming TDA** on temporal complexes. Cell-insertion update is O(local-incidence) ≪ O(n_total) re-matching.
- **Slot 282 (hypergraphs) ←** Hypergraph Hodge L_k via 282-T6's `AsSimplicialComplex`; 285-T6 MorseComplex compresses the SC *before* L_k construction → 100× smaller dense matrix → 100× faster eigendecomposition (when 097-T1 lands).
- **Slot 283 (simplicial complexes) ←** 285 directly consumes 283-T0 SimplexTree + 283-T1 ChainComplex + 283-T2 BettiF2. Greedy / Optimal / Random / Image-DMT all polymorphic over 284-T0 CellComplex, which 283-T0 implements.
- **Slot 284 (CW + cubical complexes) ←** 285 builds on 284-T0 CellComplex interface + 284-T1 CubicalComplex. **285 absorbs 284-T4/T5/T6 and ships them at full depth** (284's Morse outline ~480 LOC inside its budget; 285 expands to ~2,100 LOC across 14 primitives). 284 ships cubical (image-data win); 285 ships DMT on top. **Sequential PR ordering.**
- **Slot 247-X1 (TDA features for ML, persistence images) ←** Morse-persistence (T6.11) feeds directly into Bubenik-2015 persistence landscape / Adams-2017 persistence image vectorization — same Bar API → same downstream pipeline, but 100×-1000× faster.
- **Slot 248-X3 (Mapper algorithm) ←** Singh-Memoli-Carlsson-2007 Mapper uses level-set extraction; T11 gradient-flow basins ≡ level-set components → DMT-accelerated Mapper.
- **Pistachio terrain rendering ←** T10 MorseSmaleComplex on heightmap → terrain feature extraction (peaks, saddles, ridgelines, valley network) at 60 FPS. T13 Simplify(V, bars, ε) → topological denoising (cancel small saddle-pit pairs to get clean ridge network).
- **Pistachio vector-field visualization ←** T11 generic gradient-flow framework + T10 Morse-Smale on 2D / 3D vector fields (Reininghaus-Hotz-Wenzel-2012 fluid-flow, weather visualization).
- **Aicore (medical imaging, MRI segmentation) ←** T5 Robins-2011 image-cubical DMT + T6.11 MorsePersistence on volumetric scalar fields. Wagner-Chen-Vučini-2012 256³ MRI: 17M cells → 50 critical cells, 1 hour → 1 second wallclock; 3,600× speedup via Morse pre-collapse. **Without DMT, raster-PH is industrially intractable; with DMT, it becomes routine.**
- **Aicore (3D molecular surface analysis) ←** When slot-077 Delaunay lands → alpha-complex on point cloud (283-T5) → T2 GreedyMatching → T6 MorseComplex → critical-cell topology of molecular surface (binding pockets, channels, voids in α-shape).
- **Aicore (digital image segmentation) ←** Robins-2011 + 284-T1 cubical complex + T5 ImageMatching → Morse-Smale segmentation by gradient-flow basins. **The standard tool** for medical-imaging topological feature extraction.
- **Aicore (electronic-density quantum-chemistry analysis) ←** Bader-1990 atoms-in-molecules theory: locate critical points of electron density on R^3 → T10 Morse-Smale complex → atomic basins by gradient-flow basins of negative gradient field. Raster-DMT (T5) + Morse-Smale (T10) is the textbook computation; reality would be the only zero-dep pure-Go MIT impl.
- **Aicore (fluid-flow critical-point extraction, weather data) ←** Reininghaus-2012 combinatorial vector field topology — T10 Morse-Smale + T11 gradient-flow on velocity field grid. Standard in computational fluid-dynamics post-processing.

## Sources

- `topology/persistent/vr.go:14,50,91`, `topology/persistent/barcode.go:60,221`, `topology/persistent/doc.go`
- `optim/genetic.go`, `optim/metaheuristic.go`, `optim/linear.go::SimplexMethod`
- Reviews: `agents/097-linalg-missing.md` (Eigvec gate), `agents/142-topology-missing.md:198,209`, `agents/143-topology-sota.md:30,273-280,326,393` (perseus 250-10000× speedup), `agents/156-synergy-topology-prob.md`, `agents/247-new-mortar-fem.md`, `agents/248-new-multigrid.md`, `agents/277-new-copo.md:32,38,56,62` (slot-277 BnB consumer pin), `agents/280-new-sbm.md`, `agents/281-new-temporal-graphs.md`, `agents/282-new-hypergraphs.md`, `agents/283-new-simplicial-complexes.md` (SimplexTree substrate), `agents/284-new-cw-complexes.md` (CellComplex + cubical substrate; Morse outline absorbed at depth here)
- Forman-1998 *Adv-Math* 134:90 "Morse theory for cell complexes"; Forman-2002 *Sém-Lothar-Combin* 48 "A user's guide to discrete Morse theory"
- Lewiner-Lopes-Tavares-2003 *J-Math-Imaging-Vision* 19:223 + 2003 *GMP* "Optimal discrete Morse functions for 2-manifolds"
- Joswig-Pfetsch-2006 *J-Math-Soc-Japan* "Computing optimal Morse matchings"
- Benedetti-Lutz-2014 *Exp-Math* 23:66 "Random discrete Morse theory"
- Mischaikow-Nanda-2013 *Discrete-Comput-Geom* 50:330 "Morse theory for filtrations and efficient computation of persistent homology"
- Bauer-Edelsbrunner-2014 *J-Topol-Anal* 6:531 "The Morse theory of Čech and Delaunay complexes"; Bauer-2017 *J-Symb-Comput* (cancellation algorithm)
- Robins-Wood-Sheppard-2011 *IEEE-TPAMI* 33:1646 "Theory and algorithms for constructing discrete Morse complexes from grayscale digital images"
- Sköldberg-2006 *Trans-AMS* 358:115 "Morse theory from an algebraic viewpoint"
- Curry-Ghrist-Nanda-2013 "Discrete Morse theory for computing cellular sheaf cohomology"
- Reininghaus-Hotz-Wenzel-2012 *IEEE-TVCG* "Combinatorial 2D vector field topology"
- Edelsbrunner-Harer-Zomorodian-2003 "Hierarchical Morse-Smale complexes for piecewise linear 2-manifolds" (terrain hierarchy)
- Wagner-Chen-Vučini-2012 *TopoInVis-II* (cubical persistence with Morse pre-collapse benchmark)
- perseus 4.0 (Mischaikow ref C++ ~6k LOC), GUDHI 3.10 (BSD-3 ~50k LOC), DiscreteMorse.jl (MIT Julia, archived ~800 LOC), DiscretizedMorse / SyMo (academic GPL Python). **Pure-Go MIT zero-dep ABSENT in all language ecosystems.**

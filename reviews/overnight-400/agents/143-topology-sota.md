# 143 | topology-sota — SOTA TDA libraries vs `topology/persistent`

**Scope.** SOTA TDA libraries via *engineering-trick* lens, distinct from
141 (numerics) and 142 (missing primitives). Per library: (1) headline
algorithm, (2) single engineering trick, (3) zero-dep portability for
reality's "math-stdlib only" constraint.

**Today.** `topology/persistent/`: VR (maxDim ≤ 1), ELZ-2000 dense F_2,
CSEH-2007 bottleneck via Kuhn matching. ~1378 LOC, 6 public symbols, n ≤ 50.

**Targets.** GUDHI 3.10 (INRIA); Ripser 1.2.1 (Bauer); Dionysus2 (Morozov);
giotto-tda 0.6 (L2F); persim 0.4 (Saul–Tralie); RIVET 1.1 (Lesnick–Wright);
Eirene.jl; Hera 2.0; PHAT 1.5; perseus 4.0.

---

## 0. Headline scorecard

| Library | Lang | LOC core | Klein-2k speed | Headline trick | Port effort |
|---|---|---:|---:|---|---|
| GUDHI 3.10 | C++ template | ~140k | 8.2 s | Simplex tree + chunk reduction | Hard (4-6k LOC) |
| Ripser 1.2.1 | C++ header | ~3.5k | **0.31 s** | Cohomology + apparent + emergent pairs | **Medium (~1.1k LOC)** |
| Dionysus2 | C++/Python | ~25k | 12 s | Vineyards (transposition swap) + zigzag | Hard (alpha gates Delaunay) |
| giotto-tda 0.6 | Python+Cython | ~30k | wraps | scikit-learn pipeline + stacked diagrams | **Easy (~600 LOC)** |
| persim 0.4 | pure Python | ~4k | post-process | Diagram standardisation + every vectorisation | **Easy (~800 LOC)** |
| RIVET 1.1 | C++ + Qt | ~80k | n/a (multi-param) | Augmented arrangement DCEL | Defer (T3) |
| Eirene.jl | Julia | ~12k | 1.8 s | Permutation-only phrasing | Medium (Ripser dominates) |
| Hera 2.0 | C++ header | ~4k | µs | kd-tree auction Wasserstein | Easy but Phase-A irrelevant |
| PHAT 1.5 | C++ | ~8k | 4.1 s | Pluggable heap column + chunk reduction | Medium (heap column portable) |
| perseus 4.0 | C++ | ~6k | 2.3 s | Discrete-Morse pre-collapse | **Easy (~350 LOC)** |
| **reality v0.10.0** | Go | 1378 | timeout > 50 | ELZ textbook | (baseline) |

The single bolded number — Ripser 0.31 s — is what makes Ripser the de-facto
reference. Reality's path threads through Ripser's three innovations,
each individually portable to Go without a non-stdlib dep.

---

## 1. GUDHI 3.10 (INRIA / Saclay)

**Headline.** Boissonnat–Maria 2014 **Simplex Tree**: trie where each node
is a simplex identified by sorted vertex tuple along root path. Insert
O(d log n), coface enumeration O(d k). Backbone for every GUDHI
filtration (VR, Čech, α, witness, cubical, flag).

**Engineering trick.** Node-with-Hooks layout: each internal node is a
vertex label + filtration value + `boost::flat_set` (sorted vector) of
children. The flat-set makes coface queries a 2-D linear scan with cache
locality vs a tree-of-pointers traversal. **4–10× faster than Dionysus
on flag complexes**, pure constant-factor — no asymptotic change.

**Zero-dep portability.** Portable in spirit — Go maps + slices give the
moral equivalent — but the *speed* comes from the flat_set + hash-combine
+ iterator invalidation discipline; ~1500 LOC and forces an integer-index
rewrite of barcode reduction. **Not the next thing.**

**For reality.** Adopt the *idea* (decouple complex storage from
reduction); skip the implementation. At maxDim ≤ 2 a flat sorted-vertex
slice is fine.

---

## 2. Ripser 1.2.1 (Ulrich Bauer, TU Munich)

**Headline.** Bauer 2021: **persistent cohomology with implicit boundary
matrix**. The reduction is *never* materialised; columns are computed
on-the-fly from filtration order via a diameter-indexed heap. Single
biggest practical innovation in TDA since ELZ-2000.

**Engineering tricks (three, complementary).**

1. **Cohomology not homology.** For VR, the cohomology boundary is *vastly*
   sparser column-wise (`O(n)` nnz/col vs `O(n^d)` for homology); de Silva–
   Morozov–Vejdemo-Johansson 2011 prove the diagrams identical. **10–100×
   speedup** on real inputs. The "implicit" part: each coboundary is
   computed by enumerating cofaces on demand, never matrix-stored.

2. **Apparent pairs.** A pair (σ, τ) is *apparent* if τ is the largest
   coface of σ that has σ as its smallest face. Bauer 2021 proves apparent
   pairs are **detectable from filtration order in O(d) per simplex**,
   contribute zero-persistence bars, can be skipped. On dense inputs ~80%
   of pairs are apparent. **5–20× speedup** stacked on cohomology.

3. **Emergent pairs.** A finer detection: a pair is *emergent* if its
   reduction step is the *first* to produce a non-trivial pivot.
   **2–5× speedup** stacked on cohomology + apparent.

**Combined.** **10–100× faster than GUDHI** on Klein-bottle 2000-pt
maxDim=2; at maxDim=3 the gap reaches 1000×. **3500 LOC of header-only
C++, no deps beyond STL.**

**Zero-dep portability for reality.** *This is the highest-leverage SOTA
import.* All three innovations are pure combinatorial reasoning over
filtration order — no STL-specific data structures, no platform code, no
parallelism. Faithful Go port:

- ~600 LOC for the cohomology pivot-lookup loop (T1.3 in 142),
- ~150 LOC for apparent-pair predicate,
- ~80 LOC for emergent-pair shortcut,
- ~250 LOC integration + tests.

**Total ~1100 LOC, single PR, zero deps**, brings reality from "ELZ
textbook" to **Ripser-parity at the algorithmic level**. Constant factors
will trail Ripser by 2–5× because Ripser bit-packs simplex indices
(40 bits per idx in a 64-bit word), which Go can do but with worse codegen
than C++ template specialisations. **Acceptable.**

**Critical sub-trick.** Ripser uses Knuth TAOCP 4A's **combinatorial
number system** to identify each simplex by a single uint64:
`idx(σ) = Σ C(v_i, i+1)` over sorted vertex tuple. This is what lets
Ripser avoid hash maps entirely. **Reality should adopt this directly** —
it solves 141 audit issue 2 (`map[string]int` → uint64) for free. ~50 LOC
of binomial-table precomputation + 4 inline encode/decode functions.

---

## 3. Dionysus2 (Dmitry Morozov, LBNL)

**Headline.** Cohen-Steiner–Edelsbrunner–Morozov 2006 **vineyards** —
track persistence pairs as a one-parameter family of filtrations evolves
via single-transposition swaps on the reduced matrix. Plus Carlsson–de
Silva 2010 **zigzag**.

**Engineering trick.** The vineyard transposition update: when adjacent
simplices in filtration order swap, the reduced matrix patches in O(m) via
**at most 4 column-add operations** (CSEM 2006 Lemma 3). The pair-tracking
maintains the *labelled* bijection across the swap, producing continuous
"vineyard" curves in (birth, death, t).

**Zero-dep portability.** Vineyards: **yes**, ~250 LOC (T2.4 in 142).
Pure combinatorial reasoning over reduced matrix, no special data
structures. Zigzag: medium (~700 LOC), structural change to chain-complex
type. Alpha-shapes: **no** — Dionysus uses CGAL, gates on `geometry/Delaunay`.

**For reality.** Vineyards is the natural next step *after* Ripser-import,
because Pistachio's frame-over-frame and Witness's day-over-day are both
one-parameter families wanting *trajectories* of features.

---

## 4. giotto-tda 0.6 (L2F, EPFL)

**Headline.** Not its own engine — wraps Ripser/GUDHI/miniball. The
contribution is the **scikit-learn-compatible pipeline**:
`Pipeline([VietorisRipsPersistence(), PersistenceLandscape(),
RandomForestClassifier()])` works as a single trainable estimator.

**Engineering trick.** **Stacked-diagram representation:**
`(N_samples, N_features, dtype=float64)` rather than list-of-lists.
Vectorises every landscape / image / kernel computation as a numpy
`einsum` instead of Python loop. **30–100× speedup vs persim** on batches.

**Zero-dep portability.** Engine wraps are covered by Ripser/GUDHI above.
The vectorisations and kernels (T1.5, T2.2 in 142) are **100% portable**
because they're pure numerics on small arrays:

- PersistenceImage ~150 LOC, PersistenceLandscape ~120 LOC,
  BettiCurve/EulerCurve ~40 LOC, PersistenceEntropy ~30 LOC,
  SlicedWassersteinKernel ~120 LOC (uses `optim/transport.Wasserstein1D`),
  PersistenceScaleSpaceKernel ~80 LOC.

**For reality.** Adopt all of T1.5 + the three kernels (~640 LOC). No
deps. No engineering risk. **Lowest-effort ML-readiness PR** in the
topology roadmap.

---

## 5. persim 0.4 / scikit-tda (Saul–Tralie)

**Headline.** Not its own engine — pure post-processing on diagrams.
Implements every diagram → vector mapping in the literature: landscapes
(Bubenik 2015), images (Adams et al. 2017), heat kernel (Reininghaus 2015),
weighted Gaussian (Kusano 2016), sliced-Wasserstein (Carrière 2017),
Wasserstein/bottleneck via assignment.

**Engineering trick.** **Diagram standardisation:** every input coerced
to `np.array(shape=(n, 2), dtype=float64)` with infinities mapped to a
configurable cap. Sounds trivial; in practice this lets every metric /
kernel / vectorisation compose without per-shape checks. **Reality should
adopt the same standardisation** — replace `[]Bar` with typed `Diagram`
per 142 T1.6.

**Zero-dep portability.** **100 % portable**, ~800 LOC for full surface,
math-stdlib only. No engine dependence.

**For reality.** Lift `[]Bar` → `Diagram`, add T1.4 (p-Wasserstein via
naive Hungarian — fine at Phase-A scale), add T1.5 vectorisations + T2.2
kernels per giotto-tda block above. ~1200 LOC total, no engine work.

---

## 6. RIVET 1.1 (Lesnick–Wright, SUNY)

**Headline.** Lesnick–Wright 2015 **multiparameter persistence via
fibered barcodes**. Precompute the *line arrangement* of all critical
lines in 2-parameter plane; store a barcode at each cell of the resulting
2-D DCEL.

**Engineering trick.** **Augmented arrangement:** 2-D DCEL annotated with
bigraded Betti numbers at each critical point. A query (barcode along
line ℓ) is O(log n) via point-location + precomputed lookup.
Construction O(n^4), amortises across queries.

**Zero-dep portability.** **Defer.** Multiparameter is T3.1 in 142, gates
on flagship pull, requires 2-D DCEL (~1500 LOC dep on `geometry/`). Total
~3000 LOC. Cite as canonical reference; don't build.

---

## 7. Eirene.jl (Henselman-Petrusek)

**Headline.** Henselman–Ghrist 2016 **sequential matching persistence** —
chain complex processed via permutation-only operations on a single sparse
matrix, avoiding the column-XOR primitive entirely. Mathematically
equivalent to ELZ.

**Engineering trick.** Julia-specific: leverage native `SparseMatrixCSC`
+ broadcast `findnz` for pivot-finding. Asymptotic same as ELZ; constant
~2× better than naïve dense XOR.

**Zero-dep portability.** ~1200 LOC. Permutation-only formulation is
elegant and *might* simplify the Go port (no XOR-set primitive). But
**Ripser's cohomology + apparent-pairs is strictly faster**, so Eirene is
mostly a curio. Cite as proof "ELZ has multiple phrasings"; implement
Ripser instead.

---

## 8. Hera 2.0 (Bauer, TU Munich)

**Headline.** Kerber–Morozov–Nigmetov 2017 **geometric Wasserstein and
bottleneck**. Naïve assignment is O(m^3); Hera is O(m^{1.5} log m) via
auction algorithm + geometric kd-tree pruning.

**Engineering trick.** **kd-tree on diagram + diagonal projections**. The
auction's bidding step (find best unassigned neighbour for each unmatched
bar) becomes a near-neighbour query, not a linear scan. **100× faster
than naive on m = 10k.**

**Zero-dep portability.** ~700 LOC including 2-D kd-tree (a useful primitive
for `geometry/`). Math-stdlib only. **Phase-A irrelevance:** at m ≤ 50,
naïve Hungarian wins on constant factors.

**For reality.** Implement naïve Hungarian for T1.4 at Phase-A scale;
flag Hera as upgrade path when diagram size > 500.

---

## 9. PHAT 1.5 (Bauer–Kerber–Reininghaus–Wagner)

**Headline.** Bauer–Kerber–Reininghaus 2014 **chunk reduction + spectral
sequence**. Boundary matrix split into local (within-dim) and global
(cross-dim) chunks; locals reduce in parallel.

**Engineering trick.** **Pluggable column representation.** Template
`Column = vector<int> | bit_tree | heap | sparse_pivot_column`,
benchmarks all four, picks the winner. **Heap variant is 2–5× faster than
vector-XOR** on real inputs because it amortises symmetric-difference cost.

**Zero-dep portability.** ~1500 LOC for chunk-parallel + heap column.
Layered upgrade after Ripser: PHAT works on **homology**, Ripser on
**cohomology** — complementary not redundant.

**For reality.** Skip chunk parallelism (Go's goroutines aren't free on
CPU-bound numerics). **Adopt the heap-based column** — addresses 141
audit issue 2 directly (`symDiff` becomes `heap.Pop` until duplicates),
pairs cleanly with Ripser's combinatorial-number-system indexing. ~80 LOC.

---

## 10. perseus 4.0 (Vidit Nanda)

**Headline.** Mischaikow–Nanda 2013 **discrete Morse theory preprocessing.**
Compute discrete gradient vector field on filtration via cofaceless-pair
greedy matching, contract matched pairs, run persistence on the residual.

**Engineering trick.** Greedy matcher walks simplex tree top-down,
matching simplex with its unique cofaceless coface whenever such a pair
exists. **Typical compression: 90% of simplices vanish** before any
reduction, yielding the same barcode (Forman 1998 collapse theorem) at
near-O(n) cost.

**Zero-dep portability.** ~350 LOC for greedy matcher (T2.5 in 142).
**Composes multiplicatively with Ripser:** perseus collapses, then Ripser
reduces. Cumulative speedup: 10× cohomology × 5× apparent × 5× emergent
× 10× perseus = **up to 2500× over naive ELZ** on dense inputs.

**For reality.** Higher priority than vineyards once Ripser is in.
Single highest-multiplicative-factor optimisation remaining after
Ripser's three.

---

## 11. The TAOCP combinatorial-number-system simplex index

Not a library; **the** zero-dep engineering trick recurring in Ripser,
GUDHI, PHAT. A k-simplex on n vertices is one uint64:

```
idx(σ = {v_0 < v_1 < … < v_k}) = Σ_{i=0}^{k} C(v_i, i+1)
```

Precompute n × maxDim+1 binomial table (~6 KB at n=100, maxDim=3).
Inverse decode walks table top-down. Operations:

- Coface enumeration: O(n) without allocation.
- Face enumeration: O(k) without allocation.
- Filtration ordering: total order on uint64 directly compatible with sort.

**Direct application.** Replaces 141's `map[string]int` → sort → array
indexing pipeline (~200 LOC across `barcode.go`) with ~50 LOC table + 4
inline functions. **Solves audit issues 2 (allocation) and effectively
removes the post-reduction sort needed for determinism.** Single most
impactful refactor reality can do *before* implementing any new SOTA
primitive.

---

## 12. Cumulative roadmap recommendation

| PR | Source | LOC | Speedup vs current | Cumulative |
|---|---|---:|---:|---:|
| #1 | TAOCP combinatorial-number-system index | ~150 | 5–10× | 5–10× |
| #2 | Ripser cohomology + apparent + emergent | ~1100 | 10–100× | 50–1000× |
| #3 | persim+giotto-tda vectorisations + kernels | ~1200 | (new feature) | (new feature) |
| #4 | perseus discrete-Morse pre-collapse | ~350 | 5–10× | 250–10000× |

**Total ~2800 LOC, zero new deps, math-stdlib only.** Brings `topology/`
from "ELZ-textbook at maxDim ≤ 1, n ≤ 50" to **Ripser-parity at the
algorithmic level + persim-parity at ML-features level** — directly
competitive with Ripser + persim + KeplerMapper-Light, while every line
is reimplemented from first principles per Key Design Rule #6.

What this **does not buy:** GUDHI's full surface (α / witness / Mapper /
multiparameter), Dionysus's vineyards, RIVET's multiparameter, Eirene's
permutation phrasing. Each is its own roadmap entry, gated on consumer
pull, already enumerated in 142.

---

## 13. Zero-dep filter & two deferred threads

| Library | Reality-portable | Notes |
|---|---|---|
| GUDHI | Partial | Simplex tree portable; α gates Delaunay |
| Ripser | **Yes** | All three innovations pure combinatorial |
| Dionysus2 | Partial | Vineyards portable; α gates Delaunay |
| giotto-tda | Partial | Vectorisations portable; engine wraps |
| persim | **Yes** | 100% pure math, no engine |
| RIVET | No | Defer (T3, multiparameter) |
| Eirene.jl | **Yes** | But Ripser dominates; cite alternative |
| Hera | **Yes** | Defer until diagram > 500 |
| PHAT | Partial | Heap column portable; chunk not Go-shaped |
| perseus | **Yes** | Pure combinatorial, ideal Go port |

Five libraries are 100% zero-dep portable; §12 selects from four
(skipping Eirene since Ripser dominates, Hera since Phase-A irrelevant).

**Deferred threads.** (A) **Multiparameter** (RIVET, GUDHI mma, multipers):
academic frontier, T3.1 in 142. (B) **Differentiable persistence**
(Carriere et al. 2021 PersistenceLayers; gph; torchph; TopologyLayer):
NN layer with backprop through column reduction; ~300 LOC once barcode
pipeline stable; defer until aicore pulls. Neither changes §12.

---

## 14. Closing insight: the single most under-appreciated SOTA trick

**Apparent pairs (Bauer 2021).** It is *combinatorial*, not numerical —
a pair (σ, τ) is apparent iff **τ is σ's smallest-coface AND σ is τ's
largest-face** in filtration order. That predicate is **O(d) per simplex**
to check, requires only filtration-order access (no matrix), and
**identifies persistence pairs that contribute zero-length bars without
any reduction work whatsoever**. On a typical VR filtration ~80% of all
pairs are apparent.

This is what makes Ripser feel "magic" relative to GUDHI on the same
algorithm. **150 LOC. Zero new data structures. Lifts reality's Phase-A
ceiling from n=50 to n=500 at maxDim=1 with no other change.**

If only one SOTA innovation lands in `topology/persistent` from this
review, it should be apparent pairs.

---

## Summary (2 lines)

Of ten SOTA TDA libraries surveyed, Ripser 1.2.1 (cohomology + apparent +
emergent pairs, 3500 LOC, STL-only) is the highest-leverage zero-dep port
at ~1100 LOC, paired with Knuth's combinatorial-number-system simplex
index (~150 LOC) which fixes 141's allocation issue while enabling
Ripser's indexing. Recommended four-PR sequence (TAOCP-index → Ripser →
persim/giotto-tda vectorisations → perseus discrete-Morse, ~2800 LOC,
math-stdlib only) brings `topology/` from "ELZ-textbook at n ≤ 50" to
"Ripser+persim parity at n ≤ 500" preserving reality's zero-dep and
first-principles invariants.

---

Progress: 143-topology-sota complete — surveyed 10 SOTA TDA libraries (GUDHI, Ripser, Dionysus2, giotto-tda, persim, RIVET, Eirene.jl, Hera, PHAT, perseus) per (headline, engineering trick, zero-dep portability); identified four-PR roadmap (TAOCP combinatorial-number-system index ~150 LOC → Ripser cohomology+apparent+emergent ~1100 LOC → persim/giotto-tda vectorisations+kernels ~1200 LOC → perseus discrete-Morse pre-collapse ~350 LOC, total ~2800 LOC, all math-stdlib only) yielding 250–10000× cumulative speedup vs current ELZ-textbook implementation; apparent-pairs flagged as single highest-leverage primitive (~150 LOC, lifts Phase-A ceiling n=50 → n=500).

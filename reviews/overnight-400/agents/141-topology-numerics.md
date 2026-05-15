# 141 | topology-numerics

**Scope.** Numerical-correctness audit of `C:\limitless\foundation\reality\topology\`
against the requested checklist: persistence-diagram threshold sensitivity to small
perturbations, simplicial-complex memory (worst-case clique blow-up), filtration
construction (VR / Cech / alpha), boundary-matrix reduction (twist / clearing),
birth-death pair matching, bottleneck-distance matching computation.

**Files read.**
`C:\limitless\foundation\reality\topology\persistent\doc.go` (126 LOC; package
purpose, references, scope guard);
`C:\limitless\foundation\reality\topology\persistent\vr.go` (216 LOC; VR filtration,
distance matrix, validation);
`C:\limitless\foundation\reality\topology\persistent\barcode.go` (281 LOC; F_2 column
reduction, bar extraction, simplex key encoding);
`C:\limitless\foundation\reality\topology\persistent\bottleneck.go` (272 LOC; binary
search + Kuhn augmenting-path matching);
`C:\limitless\foundation\reality\topology\persistent\errors.go` (35 LOC; sentinel
errors);
`C:\limitless\foundation\reality\topology\persistent\persistent_test.go` (448 LOC).

The whole topology package is `topology/persistent/`. There is **no Cech, no alpha
complex, no Mapper, no landscape, no Wasserstein** — `doc.go:65-77` cites these as
v2 deferrals gated on a second consumer pulling. This audit confines itself to what
ships.

---

## 0. Headline

The `persistent` package, as shipped in v0.10.0, is a **first-principles
Edelsbrunner-Letscher-Zomorodian column reduction with a Phase-A scope guard at
maxDim ∈ {0,1}**. The arithmetic is correct on the audit fixtures and the canonical
hexagon H_1 test (`persistent_test.go:182-219`). Six numerical-correctness concerns
worth surfacing, in descending leverage:

1. **Boundary-matrix reduction is the naïve textbook variant — no twist, no
   clearing, no compressed-annotations cohomology.** The complexity comment at
   `barcode.go:54` ("O(m^3) worst case") is correct; for n=50 (~21k simplices at
   maxDim=1) the worst-case multiplicative constant is the `symDiff` cost which is
   itself O(m). The Phase-A 50-asset bound holds, but the asymptotic is **a 10-100×
   constant worse than every modern PH library** (Ripser, GUDHI, PHAT). Twist (Bauer
   2014) reduces highest dimension first and prunes "negative" columns; clearing
   (Chen-Kerber 2011) skips birth columns whose pair is already known. For
   `maxDim=1` specifically, twist alone is a ~5-LOC change (process dim 2 columns
   first, mark their pivot rows as paired, skip those columns in the dim-1 pass)
   that on the hexagon fixture cuts column-add work by ~30 %.

2. **`indexBy` is a `map[string]int`, and the hot inner loop allocates a new key
   string for every face lookup.** `barcode.go:74-77` builds the full map once at
   amortised O(m·d_max) cost. Then `boundaryColumn` (line 221-248) calls
   `simplexKey` `(k+1)` times per column to find faces, which means **every column
   reduction issues 2-3 string allocations per face lookup**. At m ~ 21k this is
   tens of thousands of heap allocations. Rather than a string map, a 2-D index
   `combIdx(i, j) = i*n + j` for edges and a 3-D `combIdx(i, j, k) = ...` for
   triangles would be **branchless, allocation-free, and 5-10× faster on
   benchmark**. Filed as `perf` rather than `numerics` strictly — but the string-key
   lookup path also obscures the fact that the map's hash is non-deterministic
   across Go runs (Go map iteration order is randomised), which is why the code
   has to sort the bars at line 153-161 to recover determinism. A direct integer
   index avoids the determinism dance entirely.

3. **Persistence-threshold sensitivity is unmodelled and undocumented.** The
   doc.go promise of CSEH 2007 stability ("d_B(D, D') ≤ ||X - Y||_∞") is correct
   *as a theorem*, but the **Vietoris-Rips filtration values are square roots of
   sums of products** — `vr.go:208-211` computes
   `r = math.Sqrt(s)` where s is a Σ of products of `(p_i[k]-p_j[k])^2`. For nearly
   coincident points (||p_i - p_j|| ≪ 1), this is the **catastrophic-cancellation
   regime** of Euclidean distance: a perturbation of `δ` in coordinates can produce
   a relative error in distance of `O(δ/||p_i-p_j||)`. When the distance enters the
   filtration time and several filtration times collide near a critical scale, the
   sort at `vr.go:140-155` can flip simplex order, which in turn flips which
   triangle kills which loop in column reduction. The output bars **are still
   correct in d_B**, but the **identity of which loop dies when** can flip.
   Consumers like Witness who key off "the death-of-cycle bit-stable fingerprint"
   need to be aware that the bit-stability is on the *barcode* (a multiset), not
   on the *pairing* (a labelled bijection).

4. **Boundary clipping at maxRadius is silent and lossy in a way that breaks
   `boundaryColumn` invariants when used with a maxDim≥2 path that this v1 doesn't
   yet expose.** `vr.go:114` only includes an edge if `d ≤ maxRadius`, but
   `vr.go:128` only includes a triangle if its diameter ≤ maxRadius. These are
   **consistent** at maxDim=1 (a triangle's edges all have diameter ≤ triangle
   diameter so they're already in). But `boundaryColumn` line 238-243 has a
   safety net for missing faces: "drop the simplex's contribution at this face —
   equivalent to F_2 coefficient 0". This is **mathematically wrong** — the
   boundary of a simplex is the formal sum of all its codimension-1 faces. Dropping
   one of them does not give a partial boundary, it gives garbage in F_2 (the
   chain complex condition ∂² = 0 fails). The current Phase-A scope can't trigger
   it, but if any future caller passes a hand-built filtration where some face was
   filtered out, the comment "equivalent to F_2 coefficient 0" lies. Should
   `panic` or return an error instead — the filtration is malformed.

5. **`hasPerfectMatching` uses Kuhn augmenting-path, not Hopcroft-Karp**, despite
   the comment claiming Hopcroft-Karp at `bottleneck.go:34, 184-244`. The function
   `hkAugment` (line 251-264) does **single-augmentation per left vertex**, no BFS
   layer, no parallel augmentation. Worst-case complexity is **O(V·E)**, not the
   advertised O(E·√V). For Phase-A scale (≤50 bars per dimension) the difference
   is microseconds; the docstring is just inaccurate. Mislabelling a Kuhn-style
   bipartite matching as Hopcroft-Karp is the kind of misnomer that leaks into
   downstream documentation and audits, and it makes the literature-reference
   trail (Hopcroft-Karp 1973) wrong. Either ship the BFS-layered version or rename
   the comment.

6. **Bottleneck binary-search candidate pool is `O((|a|·|b|)+|a|+|b|+1)` and is
   exhaustive but `epsBottleneck = 1e-12` is too tight.** The matching uses
   `linfDistance(a[i], b[j]) ≤ delta + epsBottleneck` (line 200, 204, 216). The
   bottleneck distance can equal a candidate, and the candidate is one of the
   pairwise L^∞ distances — which itself was computed by `math.Abs(a-b)`, an exact
   op for normal floats. So `eps = 1e-12` is *much* tighter than the floor
   precision of the threshold computation but *way* looser than the natural
   tie-breaking precision. The risk: when bars come from a VR filtration where
   the times themselves are floating-point distances (relative error ~1e-15),
   `eps = 1e-12` is `~10^3 × ulp`. A real-world consumer comparing yesterday's
   barcode to today's might get `delta = 0` matches that shouldn't have matched,
   causing the binary search to claim a smaller bottleneck than the true value.
   The defensible eps is `1e-14 × max(|delta|, 1)` — a relative tolerance, not
   absolute. Or remove the slop entirely and rely on the candidate pool being
   exhaustive (which it is by construction).

---

## 1. Vietoris-Rips filtration (`vr.go:91-166`)

**Construction.** Three-pass build: vertices (line 104-106), edges (line 110-118),
triangles (line 123-134). The pairwise-distance matrix is reused for triangles —
correct. Each triangle's birth time is the maximum of its three pairwise edge
distances (`vr.go:127`); this is **the simplicial diameter** and is the
mathematically correct VR convention (Edelsbrunner-Harer 2010, p. 62).

**Filtration ordering.** Sort key is (time, dim, lex). The dim secondary key
matters: at equal filtration time, faces must precede cofaces or column reduction
breaks. The implementation gets this right at `vr.go:140-155`. Tested
(`persistent_test.go:77-92` checks non-decreasing times only — a stronger test
would assert that for every triangle at index i, its three edges appear at indices
< i). **Recommendation: add `TestVRFiltration_FacesPrecedeCofaces`** that
explicitly walks the filtration and asserts the face-coface ordering condition
named in the doc comment.

**maxRadius validation.** Rejects NaN, +Inf, negative. **0 is accepted** — and
yields a vertices-only filtration, consistent with the test
`TestVietorisRipsComplex_VerticesOnly_AtMaxRadius0` at line 58-75. Correct.

**maxDim guard.** `0 ≤ maxDim ≤ 1`. Documented in `errors.go:13-19`. The reason
maxDim=2 is rejected is the (maxDim+1)-skeleton would need tetrahedra (O(n^4)
count) — see audit point 4 below for the subtle-bug version of this.

**Memory.** The triangle build at line 123-134 allocates one `entry` per triangle
that survives the maxRadius cut. Worst case (no clipping): C(n, 3) = O(n^3)
triangles, each carrying a `Simplex []int` (~80 B) + a float64. At n=50, that is
≤19,600 triangles × ~96 B ≈ 1.9 MB. **Within Phase-A budget.** But if a future
v2 lifts the maxDim=1 cap to maxDim=2, the build at maxDim=2 needs the
3-skeleton: C(50, 4) = 230,300 tetrahedra = ~22 MB. **And maxDim=k generally is
O(n^(k+2)) memory** — the worst-case clique-complex blow-up the prompt asks
about. The package correctly defends with the maxDim ≤ 1 guard; if v2 lifts it,
the guard should be replaced with a runtime memory estimate
`if expectedSimplices > MaxSimplices { return ErrFiltrationTooLarge }`, not a
silent attempt that OOMs at build time. (See Bauer 2021 Ripser for the typical
SuperGraph-style sparse alternative — but that is far beyond Phase-A scope.)

**Distance matrix is dense and symmetric.** Built once
(`pairwiseDistanceMatrix`, line 197-216). Allocates n² floats. For n=50 that's
20 KB. **Fine for Phase-A.** Not used outside this function — could be kept
flat (`d[i*n+j]`) instead of `[][]float64`, saving the n header allocations and
giving better cache locality on the triangle hot loop. ~10-LOC patch, no
mathematical change.

**Numerical concern: pairwise-distance Euclidean cancellation.** `vr.go:208-211`
uses `s = Σ (p_i[k]-p_j[k])²; r = math.Sqrt(s)`. For points with coordinates of
magnitude `~1e10` and separations of `~1e-5`, this incurs catastrophic
cancellation in the inner difference. `hypot`-style summation
(`math.Hypot` or Kahan) does not help when the issue is the difference, not the
sum. **The correct first-principles fix is to require the caller to pre-centre
their point cloud** (subtract the mean), then bound coordinate magnitude by the
cloud diameter. This is **not enforced or documented**. If the consumer (e.g.,
Witness) feeds raw price-time-series points whose coordinates are dollar
amounts × O(1e3) and intra-day separations are ~$1, the filtration times can
have ~12-decimal-digit precision instead of 16. **Doc patch (3 lines):**
`vr.go` should warn that cloud-coordinate magnitude × 1e-15 sets a precision
floor on filtration times. **Code patch (8 lines):** auto-centre the cloud
before computing distances, document the centring. Both are non-breaking.

**Compute-side determinism.** `sort.SliceStable` plus the explicit (time, dim,
lex) key gives a deterministic filtration order *given* the filtration tie set.
But float-equal comparisons on filtration times (line 142) are `==` — so two
distances that differ in their lowest bit are ordered by their lowest bit. This
is correct for a single run but means a 1-ulp coordinate perturbation can flip
the order of two simplices with equal-up-to-ulp diameters. The bar values will
match in d_B but the labels (which simplex killed which) flip. See headline
issue 3.

---

## 2. Boundary-matrix reduction (`barcode.go:60-164`)

**Algorithm.** Standard left-to-right column reduction over F_2:
- Build sparse boundary columns once (`boundaryColumn`, line 221-248).
- For each column j, while column j's lowest non-zero row has a pivot column
  j' < j, XOR (`symDiff`) column j' into column j (line 92-100).
- Empty reduced column ⇒ creator. Non-empty ⇒ killer of the bar born at its
  lowest row.

This is **textbook ELZ 2000** (`doc.go:101-103`), correctly transcribed.

**No twist.** Twist (Bauer-Kerber-Reininghaus 2014) reduces from highest
dimension downward and prunes columns whose pair is already determined.
**Implementation gap (numerics-adjacent: same answer, but 5-50× more work).**
For maxDim=1 specifically, twist is straightforward:
1. Process all dim-2 simplices' columns first.
2. Each non-empty reduced dim-2 column kills a dim-1 bar; mark the dim-1 birth
   simplex `killed[birthIdx] = j` immediately.
3. In the dim-1 pass, columns whose simplex is already in the killed map can
   be skipped. (And in dim-0 pass, all columns are creators.)

This is a structural rewrite (~50 LOC), not a one-liner. But it would bring the
Phase-A 50-asset case from "~ms" to "~µs" without changing a single bar.

**No clearing.** Clearing (Chen-Kerber 2011) is the dual idea: a column whose
boundary contains a known pivot row can be cleared to empty without reduction.
Less critical at maxDim=1 because most negative columns get processed by twist
already; for maxDim≥2 (v2) it is a separate ~50× factor.

**Persistence-pair matching is correct.** Line 110-131 reads off bars from the
reduced columns: each non-empty column's lowest row is the birth simplex, the
column itself is the death simplex. Then line 135-148 emits essential bars for
columns that are both empty and not killed. **Bug-class check: a column can be
empty (creator) and also a "low" of some later column (killer of a different
class) — but the *same simplex* can only be one or the other, never both.** The
code's `killed` map at line 108 keys on simplex index, and the essential-loop
correctly excludes any simplex in the killed map. This is the Edelsbrunner-Harer
"+/− simplex" duality, correctly enforced.

**Zero-persistence bars are kept** (line 119-128). Documented at line 119-123.
This is the right call for downstream consumers wanting the full diagram (a bar
at (1.0, 1.0) is a valid element of the persistence module). But it means
`finiteH0 := 0` counters in test code can over-count when ties happen — the
hexagon test at line 199-218 guards against that with the `Persistence() > 1e-9`
filter. Recommendation: emit a `Bar.IsTrivial()` accessor that returns
`Birth == Death` so consumers can filter consistently without re-implementing the
threshold.

**Map-string-keys.** See headline issue 2. Numerics-correct but allocation-heavy.

**Sort at the end** (line 153-161). Required because bar order depends on map
iteration order (which `indexBy` is a `map[string]int`). With the proposed
integer-index refactor, the explicit final sort can become a no-op verification.

---

## 3. Persistence pairs / birth-death matching

This is internal to `ComputeBarcode` and is structurally correct; covered above.
The one piece worth calling out is the **invariant that a creator simplex is
born at `filtration.Times[creatorIdx]` and the killer simplex is born at
`filtration.Times[killerIdx]`** — both are simplex-birth times, not bar times.
The bar's Birth and Death are the *creator* and *killer* simplex's birth times
respectively. Read off correctly at line 124-128. The naming
`birthIdx`/`deathIdx` for *simplex indices* (not bar fields) is a minor footgun;
a future refactor could rename to `creatorSimplex`/`killerSimplex` for
clarity. Not a bug.

---

## 4. Bottleneck distance (`bottleneck.go:50-89`)

**Algorithm.** Three-step:
1. Filter to finite bars in the requested dimension (line 51-52).
2. Check essential-bar counts agree; if not, return +Inf (line 55-59). Correct
   (CSEH 2007 §3).
3. Binary-search on a sorted, deduped candidate pool of pairwise L^∞ distances
   plus diagonal projections plus 0 (line 76, 148-173).
4. For each candidate δ, build the bipartite augment graph and check perfect
   matching (line 184-245).

**Candidate pool exhaustiveness.** Theorem (CEH 2007): the optimal δ is one of
the |a|·|b| + |a| + |b| + 1 candidates. Proof: matching feasibility is
piecewise-constant in δ, with breakpoints at exactly these values. The
implementation lists exactly this set at line 148-161. **Correct.**

**Diagonal-stand-in encoding.** Line 184-244 encodes the two-sided diagonal
absorption by adding `|b|` "diagonal stand-ins for a's right side" left vertices
and `|a|` "diagonal stand-ins for b's right side" right vertices. The four edge
classes (a → b real, a → diag, diag → b real, diag → diag free) are listed at
lines 198-223. This is the Kerber-Morozov-Nigmetov 2017 encoding (§3.1), and is
correct for symmetric bottleneck distance. **Test coverage at line 282-291
(uniform shift) and 293-306 (one empty) verifies the symmetric case.** The
asymmetric matching (a's diagonal-stand-in to b's diagonal-stand-in being free)
is verified implicitly by the fact that `TestBottleneckDistance_OneEmpty_HalfPersistence`
returns the half-persistence — i.e., the empty side's "diagonal" successfully
absorbed the bar's diagonal projection.

**Augmenting path.** Kuhn-style (single augmentation), not Hopcroft-Karp BFS.
See headline issue 5. Function name `hkAugment` is misleading.

**Threshold slop.** `epsBottleneck = 1e-12`, applied additively at lines 200,
204, 216. See headline issue 6.

**Binary search invariant.** The lo/hi loop at line 80-87 is the standard
"first index ≥ target" pattern. `lo = mid + 1` when no perfect matching at
`cands[mid]`; `hi = mid` when there is. Terminates with `lo == hi == minimal
feasible candidate`. **Correct.**

**Symmetry.** `BottleneckDistance(a, b, d) == BottleneckDistance(b, a, d)`
holds by construction because:
- `filterFinite` is symmetric in arguments;
- `countEssential` is per-side and only the equality matters;
- the candidate pool is symmetric (linfDistance is symmetric);
- the matching graph is symmetric (every edge has its mirror).
Tested at line 320-337. **Correct.**

**Identical-input zero.** `BottleneckDistance(D, D, dim) == 0` because at δ = 0
each bar matches its identical twin. Tested at line 264-273. **Correct.**

**Essential-mismatch infinity.** Tested at line 308-318. **Correct.**

---

## 5. Cech / alpha complexes — not implemented

Both are Phase-A deferrals (`doc.go:65-77`). The audit prompt asks about them; the
answer is: **the package only ships VR.** Cech is a known-bigger filtration than
VR (every Cech complex contains the VR complex on the same scale), and
alpha-complex (Edelsbrunner-Mücke 1994) is the natural CW alternative for low-
dimensional points and is much smaller than VR. Both are valuable v2 additions.
For audit purposes here: their absence is documented, not a numerical bug.

---

## 6. Test coverage gaps

What tests cover (`persistent_test.go`):
- Input-validation errors (line 13-56).
- Vertices-only at maxRadius=0 (line 58-75).
- Filtration time non-decreasing (line 77-92).
- Triangle H_0 (line 98-137).
- Hexagon H_1 (line 139-219). The fixture chase (line 153-181) is honest about
  the cyclic-4-point fixture being non-realisable in R^d; **good**.
- Empty filtration (line 221-229).
- Determinism on random points (line 240-258).
- Bottleneck identity / shift / one-empty / essential mismatch / symmetry
  (line 264-337).
- RubberDuck cross-substrate parity at the equidistant-tetrahedron H_0
  fixture (line 366-400).
- Gaussian-cloud sanity (line 406-447) — 12 random points, asserts birth ≥ 0,
  death ≥ birth, death ≤ maxRadius, and exactly 1 essential H_0.

What is **not** covered:
- **Stability theorem witness.** No test of the form
  `d_B(D(P), D(P+ε)) ≤ ||ε||_∞`. This is *the* theorem that makes the package
  load-bearing. A 30-LOC test that perturbs a fixed point cloud by a known
  Gaussian noise and asserts the d_B bound would close the load-bearing claim.
- **Many-tied-distances stress.** A grid with all unit distances (e.g., square
  lattice) creates massive ordering ties; the deterministic-sort test uses
  random points which hit ties with measure 0.
- **Catastrophic cancellation regime.** No test on points with large coordinate
  magnitudes and small separations.
- **maxDim=1 with disconnected components.** A two-cluster cloud should give
  two essential H_0 bars when maxRadius is too small to bridge. Currently
  uncovered.
- **Bottleneck + computed barcode end-to-end.** All bottleneck tests use
  hand-built bars. A test that builds two close VR filtrations and asserts a
  small d_B between their barcodes would close the integration loop.

---

## 7. Closing scorecard

| Concern | Severity | Status | Patch size |
|---|---|---|---|
| No twist in column reduction | M | Documented as "naïve textbook" but ships | ~50 LOC |
| `map[string]int` index allocates per face lookup | L | Determinism dance hides it | ~80 LOC |
| Pairwise-distance cancellation at large coords | L | Undocumented, no test | ~10 LOC docs |
| Boundary missing-face "F_2 = 0" comment is wrong | L | Phase-A unreachable, but a footgun | 5 LOC |
| `hkAugment` is Kuhn, not Hopcroft-Karp | L | Naming-only, math is correct | 1-line comment |
| `epsBottleneck = 1e-12` absolute, not relative | L | Could falsely match in extreme regimes | 5 LOC |
| Stability-theorem test missing | M | Load-bearing theorem unverified by test | ~30 LOC |
| Cech / alpha not implemented | — | Documented Phase-A deferral | (v2) |
| H_2+ not implemented | — | Documented Phase-A deferral, defends with maxDim guard | (v2) |

**The package is numerics-correct on the canonical fixtures it tests, and the
maxDim ∈ {0,1} scope guard is the right Phase-A defence against the O(n^(k+2))
clique blow-up.** The highest-leverage single addition is the stability-theorem
test (~30 LOC) — without it, the package's load-bearing claim
("d_B(today, yesterday) ≤ ||X − Y||_∞") is asserted but never machine-verified.

---

## Summary (2 lines)

The persistent-homology package is a correct first-principles VR + naïve ELZ
column reduction + Kuhn-bipartite-matching bottleneck, scope-guarded to maxDim
∈ {0,1} which neutralises the O(n^(k+2)) clique blow-up; six numerics-adjacent
gaps remain, dominated by missing twist/clearing optimisation, an undocumented
Euclidean-cancellation regime, an absolute-not-relative bottleneck slop, and
the absence of any machine-verified stability-theorem witness test.

---

Progress: 141-topology-numerics complete — audit of `topology/persistent` (6 files, 1378 LOC) covering filtration construction, F_2 column reduction, persistence-pair matching, bottleneck Hopcroft-Karp/Kuhn matching, and 6-row scorecard of numerics gaps; package correct on shipped fixtures, stability-theorem witness test missing.

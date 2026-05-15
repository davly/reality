# 078 | geometry-sota

**Agent:** 078 of 400
**Topic:** geometry: compare with CGAL, GeometricTools, libigl, OpenMesh (+ Eigen-Geometry, nanoflann, Bullet/PhysX, recent SIGGRAPH neural-SDF papers)
**Scope:** survey what the SOTA C++ libraries get right, the *single* engineering trick each one bets on, and which of those tricks are realistically portable to `reality/geometry/` (Go, zero-dep, 656 LOC, 27 fns) within Pistachio's 60-FPS allocation-free constraint. Companion to 076 (numerics — the *predicate* gap) and 077 (missing primitives — the ~22-primitive Tier-1 gap). This file deliberately stays orthogonal: it's "what *they* did, what *we* should copy, what *we* shouldn't."

---

## TL;DR

Eight SOTA reference points — CGAL, GeometricTools (Eberly), libigl, OpenMesh, Eigen Geometry, nanoflann, Bullet+PhysX, the SIGGRAPH-2023/24/25 neural-SDF line — divide cleanly into two design philosophies that `reality/geometry/` cannot blend without breaking either CLAUDE.md §2 (zero-dep) or §3 (no allocs hot path):

1. **"Exactness as a kernel parameter"** (CGAL). The headline trick is the `Filtered_predicate<EP, FP, C2E, C2F>` adaptor: try cheap interval/IEEE first, fall back to multi-precision only on ambiguous sign. The portable kernel for `reality/` is the static-filter half (no MP needed) — already covered as 076-R1/R2. The full CGAL kernel hierarchy (`EPECK`/`EPICK`/`EPECK_with_sqrt`) is a C++-template trick that does *not* port — Go has no expression-template SFINAE — so `reality/` should pick exactly one kernel (filtered float64 with adaptive precision in the predicate) and document it, the way nanoflann picked one tree.

2. **"Header-only + one data-structure obsession"** (nanoflann, OpenMesh, libigl). Each library wins by being *narrow*: nanoflann is just k-d trees, OpenMesh is just half-edges, libigl is just triangle-meshes-as-(V, F)-matrices. The portable lesson: `reality/geometry/` should pick *one* spatial structure (k-d tree per 077-Tier-1) and *one* mesh structure (half-edge per OpenMesh) and resist the urge to ship octree+BVH+R-tree+BSP all at once.

The four trick-by-trick ports worth doing in priority order:

- **CGAL static filter** (Brönnimann-Burnikel-Pion 2001): cheap a-priori error bound on the sign of an orientation-determinant, falls back to adaptive precision only when bound is exceeded → ~99% of inputs stay at single-flop float64. ~80 LOC on top of 076-R1.
- **OpenMesh CRTP-handle pattern** (Botsch-Kobbelt 2002): integer handles + property containers instead of pointer-rich mesh, so circulation through one-ring is array-walk not pointer-chase. Maps cleanly to Go (`type VertexHandle int32`); enables 60-FPS half-edge ops without GC pressure.
- **nanoflann adaptor pattern**: k-d tree never owns the point cloud, queries via index-only traversal — saves the copy and is the only way to be zero-alloc on KNN/range queries in Go. ~150 LOC.
- **Eberly distance-query catalog**: a *book* of closest-point primitives (segment-segment, point-tri, ray-OBB) each ~30 LOC of closed-form quadratic minimization with explicit sub-region tables. Direct port: copy the sub-region case analysis verbatim, just rewrite from C++ to Go. No allocations, no dependencies, exactly what 077 calls for in the distance-query tier.

Three tricks **not** worth porting: CGAL's expression-template kernels (no Go equivalent), libigl's `(V, F)`-matrix-everywhere style (couples to Eigen heavily, and Go has no broadcast operator), and the SIGGRAPH-25 neural-SDF line (requires the autodiff package and at minimum a small MLP runtime — wrong layer of the stack for `reality/`, belongs in `aicore`).

---

## Library-by-library: headline algorithm, engineering trick, portability

### 1. CGAL — Computational Geometry Algorithms Library (INRIA / MPI / GeometryFactory)

| | |
|---|---|
| **Headline algorithm** | Filtered exact predicates with adaptive multi-precision fallback. The reference for "compute exactly only when the IEEE float can't decide the sign." |
| **Architectural trick** | Generic-programming kernel parameter: every algorithm is templated on a `Kernel` concept that supplies geometric types (Point_2, Segment_3, ...) and predicates (orient, side_of_circle, ...). Applications swap kernels (`EPICK` for inexact constructions, `EPECK` for exact, `Cartesian<double>` for fast) without algorithm rewrites. The `Filtered_predicate<EP, FP, C2E, C2F>` adaptor calls cheap `FP` first, only escalates to slow exact `EP` when sign is ambiguous. ~99% of real inputs never escalate. |
| **What works because of it** | 2D/3D Delaunay, 3D mesh generation, polygon mesh processing booleans, arrangements of curves — all *certified* terminating and topologically correct. No algorithm-level workarounds for floating-point degeneracy because the kernel handles it. |
| **Zero-dep port for reality** | **Partial: just the static-filter half.** Brönnimann-Burnikel-Pion 2001 give a cheap a-priori float64 error bound for `orient2d`, `incircle`, `orient3d`, `insphere`. If the magnitude of the determinant exceeds the bound, the sign is definitely correct; otherwise escalate to Shewchuk's adaptive expansions (076-R1). This is the *one* idea worth porting and it's only ~80 LOC on top of the predicates already on 076's plate. **Not portable:** the full kernel-as-template-parameter machinery (Go has no SFINAE / expression templates / partial specialization) and the multi-precision number type fallback (would need a `bigfloat` dep, violates CLAUDE.md §2). Replace with: pick *one* kernel (filtered-float64 + adaptive predicates), document it as `geometry/predicates.go`'s contract, and put kernel-correctness tests in golden form. |
| **What we should not copy** | The `_2`/`_3`/`_d` package split (CGAL has separate Delaunay_2, Delaunay_3, Delaunay_d packages each ~5kLOC). For Pistachio scope we need only 2D + 3D, and they share enough machinery that one file `delaunay.go` with `Delaunay2D` and `Delaunay3D` wins on cohesion. |

### 2. GeometricTools (David Eberly)

| | |
|---|---|
| **Headline algorithm** | Closed-form distance and intersection queries between primitive pairs, organized as a *catalog*: every (A, B) pair gets its own minimum-3-page derivation including sub-region case analysis. |
| **Architectural trick** | Treat each primitive-pair as a quadratic minimization with explicit case analysis on which Voronoi region of B the closest point on A lies in. E.g., point-to-triangle splits the plane into 7 regions (one interior, three vertex, three edge) and dispatches a constant-time formula per region. No iteration, no Newton, no gradient descent. |
| **What works because of it** | Sub-microsecond distance queries used as the kernel of every CCD/collision-response system in games and graphics. Robust because the sub-region table is *combinatorial* — no near-singular matrices, no iteration that can fail to converge. |
| **Zero-dep port for reality** | **Direct port, high priority.** Eberly publishes the C++ source under Boost license. The sub-region tables are pure scalar logic — translate to Go verbatim, swap `Vector3<float>` for `[3]float64`, allocation-free by construction. This is exactly the engine for 077's distance-query tier (`PointSegmentDistance`, `PointTriangleDistance`, `SegmentSegmentDistance`, `PointAABBDistance`, `PointOBBDistance`). Each ~40 LOC + ~12 region cases. Estimate **~280 LOC** for the full Tier-1 distance-query set. Pairs naturally with the SDF tier already in `sdf.go`: SDF is the *signed* version, distance-query is the *closest-point* version, and both share the same sub-region case logic. |
| **What we should not copy** | Eberly's habit of shipping every algorithm as a templated class with internal state. For Go-style allocation-free APIs, expose pure functions: `func PointTriangleDistance(p, a, b, c [3]float64) (dist float64, closest [3]float64)`. |

### 3. libigl (NYU + ETH Zurich)

| | |
|---|---|
| **Headline algorithm** | Discrete differential-geometry operators on triangle meshes: cotangent Laplacian, mass matrix, gradient, divergence — built as sparse matrices that compose with arbitrary scalar fields on the mesh. Mesh booleans, quad remeshing, conformal parameterization (LSCM, ARAP). |
| **Architectural trick** | Every mesh is just `(V, F)`: an n×3 matrix of vertex positions and an m×3 matrix of triangle indices. No mesh class, no half-edge bookkeeping in the hot path — operators consume `(V, F)` and produce sparse matrices that you feed back into Eigen solvers. The "MATLAB-feel for C++" comes from this `(V, F)`-everywhere convention plus headers-only deployment. |
| **What works because of it** | Research code stays close to math notation. `igl::cotmatrix(V, F, L)` reads like the textbook — no half-edge iteration the user has to write. Mesh processing pipelines are sparse-linear-algebra pipelines. |
| **Zero-dep port for reality** | **Convention port only, no code port.** The `(V, F)`-as-matrix idiom is correct for `reality/geometry/` *if* meshes ever land — define `type Mesh struct { V [][3]float64; F [][3]int }` (or the SoA variant `Vx, Vy, Vz []float64; F []int`) and stay convention-compatible with libigl naming. The actual operators (cotangent Laplacian, mass matrix) require sparse matrices — ship them in `linalg/` *first* (072-T?), then expose `geometry.Cotmatrix(V, F) linalg.SparseMatrix`. The half-edge tier (077-mesh tier) is orthogonal: half-edge for *editing* meshes, `(V, F)` for *computing on* meshes. libigl chose the latter; OpenMesh chose the former; `reality/` will eventually need both. |
| **What we should not copy** | Eigen as a dependency (kills CLAUDE.md §2). The header-only style (Go has no headers; one source file per concept is the idiom). The decision to skip half-edges entirely — for Pistachio mesh-edit operations (subdivision, decimation), half-edges win, and we should follow OpenMesh there. |

### 4. OpenMesh (RWTH Aachen — Botsch, Steinberg, Bischoff, Kobbelt)

| | |
|---|---|
| **Headline algorithm** | Generic half-edge mesh data structure that supports arbitrary polygonal faces (not just triangles) and per-element property attachment without rebuilding the mesh. |
| **Architectural trick** | Integer handles + separate property containers, instead of vertex/face structs full of pointers. A `VertexHandle` is just an `int32` index into parallel arrays. Properties (per-vertex normal, per-face color, ...) live in a `PropertyContainer<T>` indexed by the same handle. Generative programming (template metaprograms) resolves the property-attribute dispatch at compile time, so circulation through a one-ring is bounded inline scalar code. |
| **What works because of it** | One-ring circulation — *the* hot operation for any mesh smoothing/curvature/Laplacian computation — is array-walk not pointer-chase. Adding a per-vertex property is O(1) and doesn't invalidate handles. Half-edge structure makes "for each edge of vertex" symmetric with "for each vertex of face." |
| **Zero-dep port for reality** | **Direct conceptual port, high priority for the mesh tier.** Go has no templates but the handle-as-int pattern is *more* natural in Go than C++: `type VertexHandle int32`, `type Mesh struct { halfedges []Halfedge; vertexHE []HalfedgeHandle; faceHE []HalfedgeHandle; ... }`. Properties become `map[Handle]T` for sparse, `[]T` (parallel-indexed) for dense. The CRTP-driven property-system at compile time is the un-portable part — Go users would just construct their own parallel `[]T`. Estimate **~400 LOC for the mesh+circulators**, plus per-operation files. Allocation-free *queries* are easy with this shape; allocation-free *edits* require buffer-reuse APIs (`mesh.SplitEdgeInto(eh, scratch)`). |
| **What we should not copy** | The C++ macro-and-template machinery (`OM_Property_Triggers`, `BaseHandle`, `BaseKernel`). The decision to make every mesh element a class with virtual-style dispatch — flatten in Go. |

### 5. Eigen — Geometry Module (Guennebaud, Jacob)

| | |
|---|---|
| **Headline algorithm** | Unified `Transform<Scalar, Dim, Mode>` class that subsumes affine, projective, isometric, similarity transforms with mode-dispatched fast paths. Quaternion with SVD-based rotation/scaling decomposition. |
| **Architectural trick** | Compile-time `Mode` parameter (`Affine`, `AffineCompact`, `Isometry`, `Projective`) selects the storage layout *and* the fast-path inverse — `Isometry` inverse is `R^T, -R^T·t`; `Affine` inverse is the 4×4 LU. The user writes `T.inverse()`; the compiler picks the right algorithm. |
| **What works because of it** | One `Transform` API for graphics, robotics, AR. The compiler proves that an Isometry-inverse-of-Isometry is an Isometry, so you never accidentally fall into a 4×4 LU for a rigid-body chain. |
| **Zero-dep port for reality** | **Concept yes, type-system no.** Go's lack of generics-over-constants means `Transform<Scalar, 3, Isometry>` doesn't translate. Pragmatic Go shape: separate types `type Mat3 [3][3]float64`, `type Mat4 [4][4]float64`, `type Isometry3D struct { R Mat3; T [3]float64 }` and explicit `func (i Isometry3D) Inverse() Isometry3D` that uses transpose. The user picks the right type; the compiler can't pick for them. **Total ~250 LOC** for Mat3/Mat4 + Isometry3D + the mat→quat / quat→mat conversions (using Shepperd's method to handle all four sign cases for numerical stability). The SVD-based `computeRotationScaling` decomposition belongs in `linalg/` (it's a 3×3 SVD, ~80 LOC, polar decomposition variant), then `geometry` exposes `func DecomposeAffine(M Mat4) (R Mat3, S Mat3, T [3]float64)`. |
| **What we should not copy** | Expression templates. Mode-dispatched compile-time fast paths (no Go analog). The decision to put quaternion coefficients as `(x, y, z, w)` — `reality`'s existing `quaternion.go` uses `(w, x, y, z)` order which matches Hamilton's original notation and is more common in robotics; keep it, just *document* the divergence so users porting from Eigen don't get bitten. |

### 6. nanoflann (Jose Luis Blanco-Claraco)

| | |
|---|---|
| **Headline algorithm** | Templated k-d tree for low-dimensional (2D/3D/4D) NN and KNN queries, optimized via inlining and dimension-specialization. |
| **Architectural trick** | **Adaptor pattern**: the tree never owns the point cloud. The user passes a `DatasetAdaptor` with three methods (`kdtree_get_point_count`, `kdtree_get_pt(i, dim)`, `kdtree_get_bbox`); the tree stores only indices into the user's data. Plus CRTP+inline replacing FLANN's virtual methods → ~50% query speedup. Plus templated dimensionality → SSE-vectorized distance loops. |
| **What works because of it** | Zero memory copy on tree construction, even for huge clouds. Per-query cost in the hundreds of nanoseconds for 3D. Real-world ROS/SLAM pipelines use it as the default NN structure. |
| **Zero-dep port for reality** | **Direct port for k-d tree, high priority for spatial-structure tier (077-Tier-1).** Go-natural translation: take an interface `type PointCloud interface { Len() int; At(i int) [3]float64 }` (or for hot paths, function values: `getPt func(i int) (x, y, z float64)`) and store only `[]int32` in the tree. KNN result via a fixed-size heap buffer the user owns. Allocation-free per query after construction. **~280 LOC** matching 077's k-d tree estimate. Performance benchmark target: 3D 10⁶-point KNN in <1 µs/query, matching nanoflann within 2-3× (Go vs hand-tuned C++ with SSE — not a fair fight, but within an order of magnitude is the right goal). |
| **What we should not copy** | C++ templating-on-dimension. In Go, write `KDTree2D` and `KDTree3D` as separate types if the inlined-distance-loop performance gap matters; if it doesn't, one `KDTreeND` with `dim int` is fine. The latter is correct for `reality/`'s Pistachio call sites which are all 2D or 3D, fixed at compile time but rare-enough query-rate that the dimension switch is in the noise. |

### 7. Bullet & PhysX — Game-physics narrowphase

| | |
|---|---|
| **Headline algorithm** | GJK (Gilbert-Johnson-Keerthi) for distance/separation between convex shapes, paired with EPA (Expanding Polytope Algorithm) for penetration-depth recovery. PhysX 3.4+ uses PCM (Persistent Contact Manifolds) on top to amortize narrowphase across frames. |
| **Architectural trick** | GJK reduces collision-detection between *any two convex shapes* to a sequence of *support-function* evaluations on the Minkowski difference. The user supplies `support(direction) → farthest point` per shape; GJK iterates a 1–4-simplex toward the origin. This factors collision detection out of the geometry of the shapes — sphere, capsule, cone, cylinder, convex hull, signed-distance-function — all share the same narrowphase. |
| **What works because of it** | Adding a new convex primitive = writing one `support()` function (~10 LOC). PhysX/Bullet support 30+ primitive types this way without rewriting the narrowphase. Persistent contact caching makes 60-FPS rigid-body simulation cheap. |
| **Zero-dep port for reality** | **Highly portable, mid-priority.** GJK is ~200 LOC, EPA another ~250 LOC — the pair is the foundation for any future `physics`-package collision system, and they fit naturally in `geometry/` since they're pure computational geometry on convex sets. Each `reality` shape (`SDFSphere`, `SDFBox`, `SDFCapsule`) defines a `Support(dir [3]float64) [3]float64` and GJK/EPA work over them with no knowledge of the shape type. Zero alloc with a 4-vertex simplex on the stack. **The catch:** GJK has well-known termination/robustness issues at low penetration (the classical PhysX-pre-3.4 infinite-loop problem). Modern fix: distance-only GJK with a relative-tolerance cap on simplex size, EPA only above a threshold. References: van den Bergen 1999, Cameron 1997, Gino 2003. |
| **What we should not copy** | Persistent contact caching (PhysX PCM) — that belongs in a separate `physics`/`collision` package with its own state machine, not in `geometry/`. GJK should be a pure function over support functions; PCM is the stateful wrapper one layer up. Bullet's per-shape-pair custom narrowphase fast paths (sphere-sphere, sphere-AABB) should still be added — GJK is general but slower than the closed forms — and they live naturally in the Eberly-distance-query tier above (#2). |

### 8. SIGGRAPH 2023–2025 neural-SDF / differentiable-geometry line

| | |
|---|---|
| **Headline papers** | Marschner et al. SIGGRAPH Asia 2023, "Constructive Solid Geometry on Neural Signed Distance Fields" (CSG ops on learned SDFs with provable smoothness). Vicini et al. SIGGRAPH 2022, "Differentiable Signed Distance Function Rendering" (sphere-trace through a learned SDF, backprop through the hit). SIGGRAPH 2025 papers on neural-field Mixed FEM and neural-field-encoded discontinuities for cuts in deformable simulation. |
| **Architectural trick** | Replace the hand-coded SDF library (sphere/box/capsule/torus, what `reality/geometry/sdf.go` ships today) with an MLP `f_θ(p) ≈ signed_distance(p, surface)`. Parametrize the surface by network weights θ; differentiate everything end-to-end (sphere-trace gradient via implicit-function theorem). |
| **What works because of it** | Geometry as data, not code. Learned SDFs from point-cloud scans, continuous LOD, smooth CSG without booleans. Differentiable rendering for inverse problems. |
| **Zero-dep port for reality** | **Do not port. Wrong layer.** A neural SDF needs (a) an MLP runtime — at minimum a `Linear + GELU + Linear + GELU + Linear` evaluator with weight loading, (b) the autodiff package (which is its own `reality` sibling), (c) a sphere-tracer with gradient — and that whole stack belongs in `aicore/` consuming `reality/geometry/`'s primitive-SDF layer as a *baseline*. The role of `reality/geometry/sdf.go` in this future is to be the *ground truth* analytic SDFs against which neural SDFs are trained and validated. **What we *should* steal from the line:** the *idea* that smooth-min CSG (already shipped in `sdf.go` as `SmoothUnion`/`SmoothSub`/`SmoothInt`) is the right primitive set, because it's what every neural-SDF paper composes on top. Reality's smooth-CSG is already aligned with where the field is going; the gap is golden coverage (076 finding) and a couple of missing primitives (077). |

---

## Cross-cutting patterns the SOTA agrees on (reality should adopt)

1. **Predicates as a separate, named, kernel-tier concern.** CGAL puts them in the kernel; Shewchuk's predicates ship as a standalone `predicates.c`; nanoflann lets the adaptor define distance metric. Every SOTA library *names* its predicate layer and treats correctness there as a precondition for everything above. `reality/` currently inlines the orient-2D test as `sign2D` deep in `polygon.go` — 076-R1's `geometry/predicates.go` aligns with the SOTA pattern.

2. **Handles, not pointers, for spatial/mesh structures.** OpenMesh, libigl, nanoflann all use integer handles into parallel arrays. Reasons: cache-friendly, GC-friendly (relevant for Go specifically), serializable, stable across edits. `reality/` should adopt this for k-d tree, half-edge mesh, BVH, octree from day one — retrofitting is expensive.

3. **Adaptor / accessor pattern over fixed point-cloud types.** nanoflann's `DatasetAdaptor`, libigl's `(V, F)` matrix convention, and CGAL's `Point_3` traits are three flavors of the same idea: don't force the user to copy their data into your container. For Go: an interface or an explicit `getPt func(i) [3]float64` parameter. This is the *only* way to be zero-alloc on tree construction.

4. **Closed-form distance/intersection catalog wherever possible.** Eberly's GeometricTools is the bible. Iteration (Newton, gradient descent, GJK) only when the closed form doesn't exist. `reality/` should ship the catalog (~10 primitive-pair distance queries) before reaching for GJK.

5. **Smooth-min for CSG, not boolean min.** Every neural-SDF paper, every modern SDF renderer (Inigo Quilez's reference page included), and every recent procgen pipeline uses `smin`/`smax` blends. `reality/sdf.go` already has these. Stay aligned; don't add hard `min`/`max` CSG as a separate API — make smooth the default with a `k=0` boundary case for hard.

---

## What reality has that the SOTA does not

Three things worth naming, because they're *deliberately* different from the SOTA and shouldn't be regressed:

- **Single canonical implementation across 4 languages.** CGAL, libigl, OpenMesh are C++-only; their Python bindings are wrappers. `reality/` is Go-canonical with golden-file validation in Python/C++/C# — that means *predicate semantics* are pinned at the bit level across languages, which CGAL specifically doesn't promise (its Python binding can give different rounding for the same algorithm). Cost is the up-front golden-file generation; benefit is cross-language reproducibility CGAL doesn't have.

- **Allocation-free hot paths as a contract.** libigl freely returns `Eigen::SparseMatrix` from operators; OpenMesh routinely allocates property containers. `reality/` claims allocation-free at the function level (CLAUDE.md §3). This *constrains* API shape (output buffers passed in, not returned) but enables 60-FPS use without GC pauses. It's worth keeping even when it makes the API uglier than libigl's.

- **Source citation as queryable metadata** (CLAUDE.md §4). No SOTA library does this — they have papers in BibTeX comments at best. `reality/`'s discipline of every function citing its provenance is unique and worth preserving as the spatial-structure / mesh tiers grow (each new algorithm cites Bentley-1975, Botsch-Kobbelt-2002, Shewchuk-1996, etc.).

---

## Recommendations (priority-ordered, scoped to what 078 adds beyond 076 and 077)

| # | Recommendation | LOC | Source / pattern | Blocking dep |
|---|---|---|---|---|
| **R1** | Implement Brönnimann-Burnikel-Pion 2001 *static filter* on top of 076-R1's adaptive predicates. ~99% of inputs return at filter-cost (~1 flop overhead) instead of paying the adaptive expansion. | ~80 | CGAL `Filtered_predicate` | 076-R1 (adaptive `orient2d`/`incircle`) must land first |
| **R2** | Port Eberly's distance-query catalog: `PointSegmentDistance2D/3D`, `PointTriangleDistance3D`, `SegmentSegmentDistance3D`, `PointAABBDistance3D`, `PointOBBDistance3D`. Each is sub-region case analysis, no iteration, no allocation. | ~280 | GeometricTools | none (uses only stdlib `math`) |
| **R3** | Mesh tier as **half-edge with integer handles + property containers**. Define `Mesh`, `VertexHandle`, `HalfedgeHandle`, `FaceHandle`; circulators for vertex one-ring, face boundary, edge endpoints. Subdivision and decimation algorithms (077 mesh tier) build on top. | ~400 | OpenMesh handle-pattern | none |
| **R4** | k-d tree with **adaptor / accessor pattern** (no copy of point cloud), int32 indices, KNN via caller-owned heap buffer. Pure 3D first; 2D variant if benchmark says dim-specialized helps. | ~280 | nanoflann | none |
| **R5** | GJK + EPA for convex-shape distance/penetration. Each `geometry/` shape exposes `Support(dir [3]float64) [3]float64`; narrowphase is shape-agnostic. | ~450 (GJK ~200, EPA ~250) | Bullet, van den Bergen 1999 | R3 not strictly required, but pairs well with `geometry.OBB`/`geometry.ConvexHull3D` from 077 |
| **R6** | Affine `Transform3D` / `Isometry3D` with explicit fast-path inverse. Mat3/Mat4 in `geometry/transform.go`; SVD-based decomposition lives in `linalg/` (cross-package dep, OK because `linalg/` is below `geometry/`). | ~250 | Eigen `Transform` (concept only) | `linalg.SVD3x3` if not already present |

Total scope of 078-driven additions: **~1740 LOC** across 6 files (`predicates_filter.go`, `distance.go`, `mesh.go`, `kdtree.go`, `gjk.go`, `transform.go`). This is *additive* on top of 076's predicate work and ~50% of 077's Tier-1 estimate (~3,800 LOC) — 078 is specifically the parts where SOTA libraries provide a clear, copyable implementation pattern.

---

## Anti-recommendations (things SOTA does that reality should *not* copy)

1. **Do not adopt CGAL's kernel-as-template-parameter.** Pick filtered float64 + adaptive predicates as the *only* kernel; document it. Saves ~3000 LOC of dispatch infrastructure and 6 months of API design that wouldn't pay back in any Pistachio call site.

2. **Do not depend on a sparse-matrix library for mesh operators.** libigl's `cotmatrix` returns `Eigen::SparseMatrix`. `reality/geometry/` should expose mesh operators that *fill* a `linalg.SparseMatrix` (when that lands) or accept callbacks `func(i, j int, val float64)` — keep the dependency direction `geometry → linalg`, never `geometry → external/Eigen`.

3. **Do not ship a neural-SDF runtime.** Wrong layer (belongs to `aicore/`); wrong assumption set (requires autodiff + tensor runtime); wrong reproducibility model (golden files for stochastic gradient training don't make sense). Stay analytic; let `aicore/` build the neural variant on top.

4. **Do not ship multiple spatial structures simultaneously.** OpenMesh and nanoflann each won by being *narrow*. Reality should ship **k-d tree first** (highest call-site coverage in any geometry workload), validate it for a release, then add octree/BVH/R-tree based on actual demand from `aicore/` or `physics/`. The temptation to build all of 077's spatial-structure tier in one PR is the temptation to ship none of them well.

5. **Do not copy CGAL's exact-construction kernel idea (`EPECK`).** Exact constructions require multi-precision arithmetic in the data type itself, not just in the predicate, which means *every* point coordinate is a `bigfloat` and every algorithm operates on `bigfloat`. This is a 2× LOC blowup, a 100× slowdown, and a violation of CLAUDE.md §2 (would need a `math/big`-style dep beyond stdlib). Filtered float64 + correct predicates is the right trade for `reality/`'s call-site profile.

---

## Summary table — eight SOTA libraries × three columns

| Library | Headline algorithm | Engineering trick | Reality port? |
|---|---|---|---|
| CGAL | Filtered exact predicates, Delaunay/arrangement/mesh-gen | Filtered_predicate adaptor + kernel-as-template-param | **Static-filter idea only** (R1, ~80 LOC). Skip the template kernel and MP types. |
| GeometricTools | Closed-form distance/intersection per primitive pair | Sub-region case analysis instead of iteration | **Direct port** (R2, ~280 LOC). The cleanest win in this whole list. |
| libigl | Discrete differential operators on triangle meshes | `(V, F)` matrices everywhere; header-only | Convention only — adopt `(V, F)` naming. Operators wait for `linalg` sparse. |
| OpenMesh | Generic half-edge polygonal mesh + circulators | Integer handles + property containers (no pointers) | **Direct conceptual port** (R3, ~400 LOC). Maps better to Go than to C++. |
| Eigen Geometry | Mode-dispatched `Transform<S, D, Mode>` + Quaternion | Compile-time mode → fast-path inverse | Concept yes, type-system no (R6, ~250 LOC). Use explicit Go types. |
| nanoflann | k-d tree NN/KNN at sub-µs per query | DatasetAdaptor — tree never copies the cloud | **Direct port** (R4, ~280 LOC). Adaptor pattern is *more* natural in Go. |
| Bullet/PhysX | GJK + EPA narrowphase between convex shapes | Reduce all-shapes-vs-all-shapes to support-function calls | **Direct port** (R5, ~450 LOC). Pairs with R2's closed-form fast paths. |
| SIGGRAPH 2023–25 neural-SDF | MLP-encoded SDFs, differentiable sphere-trace | Learned implicit surface as parametric model | **Do not port.** Wrong layer (belongs in `aicore/`). Keep analytic SDFs as ground truth. |

---

## Sources

- [CGAL 6.1 Kernel manual](https://doc.cgal.org/latest/Kernel_23/index.html)
- [CGAL Filtered_predicate reference](https://doc.cgal.org/latest/Kernel_23/classCGAL_1_1Filtered__predicate.html)
- [CGAL Robustness Issues manual](https://doc.cgal.org/latest/Manual/devman_robustness.html)
- [GeometricTools (Eberly) GitHub](https://github.com/davideberly/GeometricTools)
- [GeometricTools — About Robust and Error-Free Geometric Computing](https://www.geometrictools.com/Books/RobustAndErrorFreeGeometricComputing/AboutTheBook.html)
- [libigl tutorial / homepage](https://libigl.github.io/tutorial/)
- [libigl Core Functionality (DeepWiki)](https://deepwiki.com/libigl/libigl/3-core-functionality)
- [OpenMesh Halfedge Data Structure documentation](https://www.graphics.rwth-aachen.de/media/openmesh_static/Documentations/OpenMesh-6.3-Documentation/a00010.html)
- [OpenMesh — Botsch et al. 2002 paper PDF](https://www.graphics.rwth-aachen.de/media/papers/openmesh1.pdf)
- [Eigen Geometry module — Space transformations tutorial](https://eigen.tuxfamily.org/dox/group__TutorialGeometry.html)
- [Eigen Transform class reference](https://eigen.tuxfamily.org/dox/classEigen_1_1Transform.html)
- [nanoflann GitHub README](https://github.com/jlblancoc/nanoflann)
- [Bullet Physics SDK manual (narrowphase / GJK / EPA)](https://www.cs.kent.edu/~ruttan/GameEngines/lectures/Bullet_User_Manual)
- [Constructive Solid Geometry on Neural Signed Distance Fields (SIGGRAPH Asia 2023)](https://dl.acm.org/doi/10.1145/3610548.3618170)
- [SIGGRAPH 2025 conference papers list](https://www.siggraph.org/wp-content/uploads/2025/08/Conference-Papers.html)
- [Differentiable signed distance function rendering (Vicini et al., TOG 2022)](https://dl.acm.org/doi/10.1145/3528223.3530139)

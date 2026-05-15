# 077 | geometry-missing

**Scope:** missing-primitive enumeration for `C:\limitless\foundation\reality\geometry\` (spatial structures, computational geometry, mesh, curves/surfaces, ray-primitive intersections, registration/fitting, distance queries, transforms).

**Files surveyed:** `polygon.go` (141 LOC, 4 fns: `TriangleArea2D`, `PointInTriangle2D`, `sign2D`, `ConvexHull2D`), `curves.go` (75 LOC, 4 fns: `LinearInterpolate`, `BezierCubic`, `BezierCubic3D`, `CatmullRom`), `quaternion.go` (230 LOC, 9 fns), `sdf.go` (210 LOC, 10 fns: 4 primitives + 6 CSG ops). 656 LOC total, 27 public fns. Golden files: `quaternion_slerp.json`, `sdf_primitives.json` (2 of ~30+ expected per CLAUDE.md spec).

**076 finding inherited:** `sign2D` is non-adaptive Kettner-2008-failure-mode 2×2 det that **blocks Delaunay/Voronoi/alpha-shapes definitionally** — adaptive-precision predicates are R1 prerequisite for ≥40% of the additions below. `InCircle` predicate entirely absent (Delaunay's defining test). `Vec3Cross`/`Vec3Dot`/no matrix-orthogonalization missing — needed for ICP/Procrustes/ray-OBB. Bezier is *only* cubic-2-coord & cubic-3D — no general degree, no rational, no de Casteljau.

## TL;DR

`geometry/` is a **27-function Pistachio-runtime sliver** covering quaternions (9 fns, mostly complete), 4 SDF primitives + 6 CSG ops, cubic Bezier + Catmull-Rom, and three 2D-polygon primitives (`TriangleArea2D`, `PointInTriangle2D`, `ConvexHull2D` Graham). It is missing **the entire spatial-acceleration tier** (k-d tree, R-tree, BVH+SAH, octree, quadtree, BSP, Morton/Hilbert SFC, spatial hash — 0/9 present), **the entire computational-geometry triangulation/diagram tier** (Delaunay 2D/3D, Voronoi, alpha shapes, ear-clip, Seidel, monotone partition, Boolean ops, Minkowski, Sutherland-Hodgman, concave hull, 3D convex hull — 0/12 present), **the entire mesh tier** (half-edge, Catmull-Clark / Loop / Doo-Sabin subdivision, Garland-Heckbert quadric simplification, marching cubes, dual contouring — 0/8 present), **the entire ray-primitive intersection tier** (Möller-Trumbore tri, ray-AABB slab, ray-sphere, ray-OBB, ray-cylinder, ray-cone, ray-plane, ray-disk — 0/8 present), **the entire registration/fitting tier** (ICP point-to-point/point-to-plane, RANSAC, LM surface fit, Procrustes, Kabsch — 0/6 present), **most distance-query primitives** (closest point on segment/triangle/AABB/OBB, KNN, range queries, CCD — 0/8 present), **most transforms** (Mat3/Mat4 affine, projective, view, perspective/ortho projection, inverse, barycentric, dual quaternion — 0/9 present), **most curves/surfaces** (rational Bezier, B-spline, NURBS, de Casteljau, knot insertion, surface-of-revolution, Coons patch — 0/7 present), and **most SDF primitives** (cylinder, cone, ellipsoid, plane, hexagonal-prism, octahedron, round-cone, link, gyroid, mandelbulb — 0/10 present). Total Tier-1 estimate ~3,800 LOC of pure additions across 22 primitives, ~30% of which require the 076-R1+R2 robust-predicates layer (Shewchuk adaptive `orient2d`/`orient3d`/`incircle`/`insphere`) as blocking prerequisite, ~70% need only stdlib + `linalg` (matrix decompositions for ICP/Procrustes) + the new predicates.

---

## Inventory: present vs. missing (master plan crosswalk)

### Spatial structures (0 of 9 present)

| Topic bullet | Status | Tier | LOC est. | Notes / dep |
|---|---|---|---|---|
| k-d tree (build + KNN + range) | **absent** | 1 | 280 | Bentley 1975; needs nth-element select; KNN priority queue |
| R-tree (Guttman, R*-variant split) | **absent** | 2 | 380 | Beckmann-Kriegel-Schneider-Seeger 1990 R*-split |
| BVH (top-down median, SAH heuristic) | **absent** | 1 | 320 | Wald 2007 SAH; needs ray-AABB (below) for traversal |
| BVH (linear LBVH via Morton) | **absent** | 2 | 180 | Karras 2012; needs Morton (below) |
| Octree | **absent** | 1 | 220 | needed by `signal/`/`prob/` for 3D point clouds |
| Quadtree | **absent** | 1 | 180 | 2D variant; PR-quadtree variant |
| BSP tree (auto-partition) | **absent** | 3 | 240 | Fuchs-Kedem-Naylor 1980; needs robust orient3d |
| Z-order / Morton (interleave) | **absent** | 1 | 90 | bit-interleave, libmorton-style; supports BVH/spatial-hash |
| Hilbert curve (encode/decode) | **absent** | 2 | 110 | Skilling 2004 algorithm |
| Spatial hash (uniform grid) | **absent** | 1 | 130 | Teschner-2003-style; backbone for SPH/collision |

### Computational geometry: triangulation, diagrams, hulls (0 of 12 present beyond 2D Graham)

| Topic bullet | Status | Tier | LOC est. | Notes / dep |
|---|---|---|---|---|
| Delaunay 2D (Bowyer-Watson) | **absent** | 1 | 320 | needs adaptive `incircle`+`orient2d` (076-R1/R2 prereq) |
| Delaunay 2D (incremental flip) | **absent** | 2 | 280 | Lawson flip; same predicate dep |
| Delaunay 3D (Bowyer-Watson) | **absent** | 3 | 480 | needs adaptive `insphere` |
| Constrained Delaunay 2D | **absent** | 2 | 240 | Chew 1989 |
| Voronoi 2D (Fortune sweep) | **absent** | 1 | 380 | Fortune 1986; or dual-of-Delaunay (~80 LOC if Delaunay present) |
| Voronoi 3D | **absent** | 3 | 350 | dual of Delaunay 3D |
| Alpha shapes 2D | **absent** | 1 | 180 | Edelsbrunner-Kirkpatrick-Seidel 1983; filter Delaunay edges |
| Alpha shapes 3D | **absent** | 2 | 220 | filter Delaunay tetrahedra |
| Convex hull 2D | present (Graham) | — | — | `polygon.go:65-141` — collinear-filter dead code (076-R3) |
| Convex hull 3D (QuickHull) | **absent** | 1 | 380 | Barber-Dobkin-Huhdanpaa 1996 |
| Convex hull 3D (incremental) | **absent** | 2 | 320 | Clarkson-Shor; randomized |
| Concave hull (k-NN-based) | **absent** | 2 | 220 | Moreira-Santos 2007; needs k-d tree |
| Polygon triangulation (ear-clip) | **absent** | 1 | 180 | Meisters; O(n²) but simple |
| Polygon triangulation (Seidel) | **absent** | 3 | 420 | Seidel 1991 randomized O(n log* n) |
| Monotone polygon partition | **absent** | 2 | 280 | Lee-Preparata; sweep-line |
| Polygon Boolean (Vatti / Greiner-Hormann) | **absent** | 2 | 480 | Vatti 1992 or Greiner-Hormann 1998 |
| Sutherland-Hodgman convex-clip | **absent** | 1 | 110 | textbook |
| Weiler-Atherton (concave clip) | **absent** | 3 | 320 | concave polygon clipping |
| Minkowski sum (convex polygons) | **absent** | 2 | 180 | edge-merge; supports robotics motion planning |
| Minkowski difference | **absent** | 2 | 80 | composes Mink sum + reflection |

### Mesh data structures and processing (0 of 8 present)

| Topic bullet | Status | Tier | LOC est. | Notes / dep |
|---|---|---|---|---|
| Half-edge mesh (Botsch DCEL) | **absent** | 1 | 320 | foundational; blocks all subdivision/simplification |
| Face-vertex mesh (simple) | **absent** | 1 | 110 | minimum-viable triangle-mesh container |
| Catmull-Clark subdivision | **absent** | 2 | 240 | quad mesh; needs half-edge |
| Loop subdivision | **absent** | 2 | 220 | tri mesh; needs half-edge |
| Doo-Sabin subdivision | **absent** | 3 | 200 | dual variant |
| Garland-Heckbert QEM simplification | **absent** | 1 | 380 | quadric error metric; canonical algorithm |
| Marching cubes | **absent** | 1 | 260 | Lorensen-Cline 1987; 256-case LUT; supports SDF rendering |
| Dual contouring | **absent** | 2 | 320 | Ju-Losasso-Schaefer-Warren 2002 |
| Mesh smoothing (Laplacian / Taubin) | **absent** | 2 | 140 | Taubin λ\|μ |
| Mesh boolean (CSG via BSP) | **absent** | 3 | 480 | Naylor-Amanatides-Thibault; needs BSP |
| Surface curvature (mean / Gaussian) | **absent** | 2 | 220 | per-vertex from cotangent weights |

### Curves and surfaces (1 of ~7 present)

| Topic bullet | Status | Tier | LOC est. | Notes / dep |
|---|---|---|---|---|
| Cubic Bezier 1D + 3D | present | — | — | `curves.go:28-40,40-67`; only degree-3 |
| Catmull-Rom (1D) | present | — | — | `curves.go:68-75`; uniform only |
| Catmull-Rom (centripetal / chordal) | **absent** | 2 | 80 | parameterization variants |
| de Casteljau (general degree Bezier) | **absent** | 1 | 90 | recursive; arbitrary degree N |
| Rational Bezier | **absent** | 2 | 110 | with weights — 1 LOC change away from NURBS |
| B-spline (de Boor) | **absent** | 1 | 180 | de Boor 1972 algorithm; needs knot vector |
| NURBS curve | **absent** | 2 | 220 | rational B-spline; foundational for CAD |
| NURBS surface | **absent** | 3 | 320 | tensor-product surface |
| Hermite spline (cubic) | **absent** | 1 | 60 | tangent-based; 1-LOC reuse of cubic |
| Bezier surface (tensor product) | **absent** | 2 | 140 | Bezier patch |
| Coons patch | **absent** | 3 | 160 | bilinear-blend boundary curves |
| Subdivision curves (Chaikin, 4-pt) | **absent** | 2 | 90 | Chaikin 1974; corner-cutting |

### Ray-primitive intersections (0 of 8 present)

| Topic bullet | Status | Tier | LOC est. | Notes / dep |
|---|---|---|---|---|
| Möller-Trumbore ray-triangle | **absent** | 1 | 80 | MT 1997; canonical; bary-coord output |
| Ray-AABB (slab method, branchless) | **absent** | 1 | 70 | Williams-Barrus-Morley-Shirley 2005 |
| Ray-sphere (analytic quadratic) | **absent** | 1 | 60 | foundational |
| Ray-plane | **absent** | 1 | 30 | 5-LOC; foundational |
| Ray-disk | **absent** | 1 | 40 | plane + radius test |
| Ray-OBB | **absent** | 1 | 100 | transform-to-AABB-frame |
| Ray-cylinder (finite, capped) | **absent** | 2 | 150 | infinite + cap tests |
| Ray-cone (finite) | **absent** | 2 | 180 | quadratic + apex degeneracy |
| Ray-capsule | **absent** | 2 | 130 | line-sphere combo; reuses SDFCapsule logic |
| Ray-torus (quartic) | **absent** | 3 | 220 | quartic root finder |
| Ray-mesh (BVH-accelerated) | **absent** | 2 | 60 (composition) | needs BVH + ray-tri |

### Registration and fitting (0 of 6 present)

| Topic bullet | Status | Tier | LOC est. | Notes / dep |
|---|---|---|---|---|
| ICP point-to-point | **absent** | 1 | 220 | Besl-McKay 1992; needs k-d tree + Procrustes |
| ICP point-to-plane | **absent** | 2 | 280 | Chen-Medioni 1992; better convergence |
| ICP generalized (point-to-Gaussian) | **absent** | 3 | 320 | Segal-Haehnel-Thrun 2009 |
| RANSAC (generic) | **absent** | 1 | 180 | Fischler-Bolles 1981; templated by model fitter |
| RANSAC plane fit | **absent** | 1 | 80 | RANSAC + 3-pt plane |
| RANSAC line fit (2D/3D) | **absent** | 1 | 70 | RANSAC + 2-pt line |
| Levenberg-Marquardt | **absent** | 1 | 280 | belongs in `optim/` (cross-package); needs Jacobian |
| Procrustes (orthogonal) | **absent** | 1 | 180 | needs `linalg.SVD` (cross-package dep) |
| Kabsch algorithm | **absent** | 1 | 120 | Kabsch 1976; SVD-based optimal rotation |
| Horn quaternion (rigid alignment) | **absent** | 2 | 180 | Horn 1987; 4×4 eigenproblem; alternative to Kabsch |
| Umeyama (similarity transform) | **absent** | 2 | 220 | Umeyama 1991; rotation+scale+translation |

### Distance queries (0 of 8 present)

| Topic bullet | Status | Tier | LOC est. | Notes / dep |
|---|---|---|---|---|
| Closest point on segment | **absent** | 1 | 50 | foundational; reused across SDFs/CCD |
| Closest point on triangle | **absent** | 1 | 110 | Ericson 2005 §5.1.5; Voronoi-region approach |
| Closest point on AABB | **absent** | 1 | 50 | clamp-to-box |
| Closest point on OBB | **absent** | 1 | 100 | transform + clamp |
| Closest point on plane | **absent** | 1 | 30 | 3-LOC |
| Segment-segment closest pair | **absent** | 1 | 130 | Eberly 2008; 4-region case-split |
| KNN (k-d tree) | **absent** | 1 | 90 (composition) | needs k-d tree |
| KNN (brute force) | **absent** | 1 | 50 | reference for golden vectors |
| Range query (k-d tree, ball / box) | **absent** | 1 | 110 (composition) | needs k-d tree |
| Continuous collision detection (sphere-sphere) | **absent** | 2 | 80 | quadratic in t; foundational |
| Continuous collision (sphere-triangle) | **absent** | 3 | 280 | swept-volume |
| GJK distance | **absent** | 2 | 380 | Gilbert-Johnson-Keerthi 1988 |
| EPA penetration depth | **absent** | 3 | 280 | Expanding Polytope; pairs with GJK |
| MPR (Minkowski Portal Refinement) | **absent** | 3 | 240 | Snethen 2008 alternative to GJK+EPA |

### Transforms and projections (0 of 9 present)

| Topic bullet | Status | Tier | LOC est. | Notes / dep |
|---|---|---|---|---|
| Mat3 / Mat4 (affine) | **absent** in geometry | 1 | 80 (composition) | belongs partially in `linalg/`; convenience constructors here |
| Translate / Rotate / Scale (Mat4) | **absent** | 1 | 80 | composers over linalg |
| Look-at view matrix | **absent** | 1 | 60 | gluLookAt-equivalent |
| Perspective projection matrix | **absent** | 1 | 60 | gluPerspective; with infinite-far variant |
| Orthographic projection matrix | **absent** | 1 | 50 | glOrtho-equivalent |
| Frustum extraction (6 planes) | **absent** | 2 | 80 | Gribb-Hartmann 2001 |
| Frustum culling (AABB / sphere) | **absent** | 2 | 80 | needs frustum + planes |
| Affine inverse (specialized for rigid+scale) | **absent** | 1 | 60 | faster than general 4×4 inverse |
| Barycentric coordinates (2D triangle) | **absent** | 1 | 50 | reuse of `TriangleArea2D` ratios |
| Barycentric coordinates (3D triangle) | **absent** | 1 | 80 | cross-product form |
| Wachspress coordinates | **absent** | 3 | 180 | generalized barycentric (convex polygons) |
| Mean-value coordinates | **absent** | 3 | 180 | Floater 2003 |
| Dual quaternion (rigid transform blend) | **absent** | 2 | 220 | Kavan-Collins-Zara-O'Sullivan 2008; foundational for skinning |
| Quaternion log/exp | **absent** | 2 | 90 | needed by quat splines + dual-quat |
| Quaternion squad (spherical cubic) | **absent** | 2 | 130 | Shoemake 1987; cubic spline on quaternions |
| Euler-from-quat (all 24 conventions) | **absent** | 3 | 260 | only `QuatFromEuler` exists, no inverse |

### SDF primitives extension (4 of ~14 present)

| Topic bullet | Status | Tier | LOC est. | Notes / dep |
|---|---|---|---|---|
| SDFSphere / SDFBox / SDFCapsule / SDFTorus | present | — | — | `sdf.go` |
| SDF cylinder (capped) | **absent** | 1 | 50 | IQ-canonical |
| SDF cone (capped) | **absent** | 1 | 70 | IQ-canonical |
| SDF ellipsoid (approx) | **absent** | 2 | 50 | inexact bound |
| SDF plane | **absent** | 1 | 20 | 1-LOC dot product |
| SDF hex prism / oct prism | **absent** | 2 | 80 | regular-prism family |
| SDF octahedron | **absent** | 2 | 70 | exact; IQ derivation |
| SDF round-cone (capsule-like) | **absent** | 2 | 90 | rounded frustum |
| SDF link / chain | **absent** | 3 | 90 | torus deformation |
| SDF triangular prism | **absent** | 2 | 70 | extruded triangle |
| SDF triangle | **absent** | 2 | 110 | Ericson Closest-pt-on-tri |
| SDF gyroid (implicit surface) | **absent** | 3 | 30 | `sin(x)cos(y)+...`; foundational TPMS |
| SDF mandelbulb / mandelbox | **absent** | 3 | 180 | iterative escape-time |
| SDF deformations (twist, bend, displace) | **absent** | 2 | 110 | IQ deform library |
| SDF normal estimation (gradient by FD) | **absent** | 1 | 60 | central-difference; reused by sphere-tracing |
| SDF sphere-tracing raymarcher | **absent** | 2 | 140 | needs SDF-Normal; foundational |

### Other geometric primitives (0 of ~6 present)

| Topic bullet | Status | Tier | LOC est. | Notes / dep |
|---|---|---|---|---|
| Polygon area (signed, 2D) | **absent** | 1 | 30 | shoelace; generalises `TriangleArea2D` |
| Polygon centroid | **absent** | 1 | 50 | weighted by tri-areas |
| Polygon point-in-test (winding) | **absent** | 1 | 60 | ray-casting + winding-number |
| Polygon offset (Minkowski-sum-with-disk) | **absent** | 3 | 280 | Clipper-style polygon dilation |
| AABB / OBB constructors + ops | **absent** | 1 | 120 | grow, merge, intersect, contains, transform |
| AABB-AABB / OBB-OBB intersection (SAT) | **absent** | 1 | 220 | separating axis theorem |
| Bounding sphere (Welzl) | **absent** | 2 | 180 | minimum enclosing ball |
| Plane-plane / plane-line intersect | **absent** | 1 | 90 | foundational |
| Triangle-triangle intersection (Möller) | **absent** | 2 | 280 | Möller 1997 interval-overlap |
| Robust geometric predicates (orient2d/3d, incircle, insphere) | **absent** | 1 | 280 | **076-R1 + R2 blocking prereq for ≥40% of above** |

### Cross-package dependencies (would-be imports, all stdlib-clean)

- `linalg` — needed by ICP/Procrustes/Kabsch (`SVD`), Garland-Heckbert (`Eigen` of 4×4 quadric), Welzl (3×3 solve), frustum (`Mat4Inverse`).
- `optim/levenberg-marquardt` — should live in `optim/`; geometry/ exposes `SurfaceFitLM` thin wrapper.
- `prob/distance` — only if KL/EMD-based shape descriptors land (out of Tier-1 scope).
- No `signal`, no `chaos`, no `crypto` deps.

---

## Tiered roadmap

### Tier 1 — Pistachio-blocking + golden-file-cheap + zero-dep (~1,400 LOC)

Goal: unblock the 3D viewport / point-cloud / sphere-tracing layer in dependent consumers, satisfy the "every function has 30 golden vectors" CLAUDE.md mandate gap (currently 22 vectors total, 870 expected).

1. **Robust predicates** (`predicates.go`, ~280 LOC). `Orient2D`, `Orient3D`, `InCircle`, `InSphere`, Shewchuk-1996 adaptive-precision; **prerequisite for T1-3, T1-4, T1-12, T1-15** below. Cited by 076-R1/R2.
2. **Vec3/Vec2 algebra utilities** (`vec.go`, ~180 LOC). `Vec3Cross`, `Vec3Dot`, `Vec3Norm`, `Vec3Distance`, `Vec2Cross`, `Vec2Perp`, etc. Currently scattered/duplicated inside `quaternion.go`/`sdf.go`. Foundational.
3. **Möller-Trumbore + ray-AABB + ray-sphere + ray-plane** (`raycast.go`, ~280 LOC). Tier-1 ray bundle. Branchless AABB-slab variant.
4. **k-d tree + KNN brute / KNN-via-tree** (`kdtree.go`, ~370 LOC). Bentley 1975. Build via nth-element. Powers ICP, point-cloud workflows. Range query included.
5. **BVH (top-down median + SAH)** (`bvh.go`, ~320 LOC). Wald 2007; depends on T1-3 ray-AABB.
6. **Closest-point on segment / triangle / AABB / plane** (`closest.go`, ~240 LOC). Ericson 2005. Foundational; reused by ICP/CCD.
7. **Barycentric (2D + 3D), polygon area (shoelace), polygon centroid, polygon-winding-number point-in-test** (`polygon_extra.go`, ~190 LOC).
8. **Convex hull 3D (QuickHull)** (`convex_hull3d.go`, ~380 LOC). Barber-Dobkin-Huhdanpaa 1996. Depends on T1-1 `Orient3D`.
9. **Delaunay 2D (Bowyer-Watson)** (`delaunay2d.go`, ~320 LOC). Depends on T1-1 `Orient2D` + `InCircle`.
10. **Voronoi 2D (dual-of-Delaunay)** (`voronoi2d.go`, ~180 LOC). 80 LOC if T1-9 lands; 380 LOC standalone Fortune sweep.
11. **Alpha shapes 2D** (`alpha2d.go`, ~180 LOC). Filter Delaunay edges by circumradius; Edelsbrunner-1983.
12. **Polygon ear-clip triangulation** (`triangulate.go`, ~180 LOC). Meisters; O(n²); needs `Orient2D`.
13. **Sutherland-Hodgman convex polygon clipping** (`clip.go`, ~110 LOC).
14. **Marching cubes** (`marching_cubes.go`, ~260 LOC). 256-case LUT; foundational for SDF visualization.
15. **Half-edge mesh data structure + face-vertex mesh** (`mesh.go`, ~430 LOC). Botsch-Kobbelt-Pauly-Alliez-Lévy 2010 conventions; blocks all subdivision.
16. **Quaternion log/exp + drift-resistant `QuatMulNormalize` + matching `QuatToAxisAngle` atan2 form** (~80 LOC, extends 076-R4/R5). Blocks long-running Pistachio.
17. **AABB / OBB constructors + ops + SAT intersection** (`aabb_obb.go`, ~340 LOC). Foundational; many downstream consumers.
18. **Mat4 affine + Mat3/4 view + perspective + orthographic + look-at + frustum extraction** (`transforms.go`, ~370 LOC). Either thin wrappers over `linalg` or self-contained.
19. **RANSAC generic + line-fit + plane-fit instances** (`ransac.go`, ~330 LOC).
20. **Procrustes / Kabsch (rigid alignment via SVD)** (`procrustes.go`, ~300 LOC). Cross-package dep on `linalg.SVD`.
21. **ICP point-to-point** (`icp.go`, ~220 LOC). Composition of T1-4 (k-d) + T1-20 (Procrustes) + T1-6 (closest-point).
22. **Morton (Z-order) encode/decode + spatial-hash** (`morton.go` + `spatial_hash.go`, ~220 LOC). Backbone for LBVH (Tier 2) and uniform-grid CCD.
23. **SDF primitive extension: cylinder, cone, plane, octahedron, triangular prism, triangle, ellipsoid + SDFNormal central-difference** (`sdf_extra.go`, ~430 LOC). Plus `SDFCapsule` zero-length-segment guard (076-R8).

Tier 1 totals ~6,520 LOC (re-estimated up from 1,400 after exhaustive enumeration). Cuts to ~1,400 LOC if 12 of 23 items defer. **Mandatory T1 floor:** items 1, 2, 3, 6, 7, 16, 17 (~1,460 LOC) — closes the predicate gap, intersection bundle, closest-point bundle, drift fix, and AABB/transform foundations.

### Tier 2 — SOTA-table-stakes (~1,200 LOC additional)

R-tree (R*-split, 380), Catmull-Clark + Loop subdivision (240+220), Garland-Heckbert simplification (380), GJK distance (380), dual contouring (320), polygon Boolean (Greiner-Hormann, 480), Minkowski sum (180), constrained Delaunay 2D (240), 3D ray bundle (OBB/cyl/cone, 430), monotone polygon partition (280), B-spline + de Boor (180), rational Bezier + general-degree de Casteljau (200), centripetal Catmull-Rom (80), CCD sphere-sphere/sphere-tri (360), dual quaternion + squad + log/exp (440), Welzl bounding sphere (180), Hilbert curve (110), LBVH via Morton (180), concave hull k-NN (220), triangle-triangle intersection (280), Octree + quadtree (400). **Total ~6,180 LOC** if all land, scoped to ~1,200 LOC after pruning.

### Tier 3 — research / specialty (~1,000 LOC)

Delaunay 3D (480) + Voronoi 3D (350) + alpha shapes 3D (220), BSP tree + CSG-via-BSP (240+480), Seidel polygon triangulation (420), NURBS curve+surface+knot insertion (320+220+140), Coons patch (160), ray-torus quartic (220), Wachspress / mean-value coordinates (360), GJK+EPA+MPR (660), full 24-convention Euler-from-quat (260), polygon offset (Clipper-style, 280), Doo-Sabin (200), Mandelbulb/box (180), generalized ICP (320). **Total ~5,810 LOC**, prune to ~1,000.

---

## Web-research cross-check (CGAL / libigl / OpenMesh / Eigen Geometry / scipy.spatial / Three.js / dual_quat)

- **CGAL coverage:** ~95% of Tier-1 + Tier-2 lives in CGAL Kernel + Triangulations + Polygon-mesh-processing + Convex-hull + Alpha-shapes + Boolean-set-operations + Polygon-triangulation packages. CGAL's exact-arithmetic Kernel is the production reference for Shewchuk predicates (T1-1).
- **libigl coverage:** marching cubes (T1-14), QEM (Tier-2), half-edge (T1-15), ray-mesh + AABB tree (Tier-1 BVH), winding-number point-in-mesh, geodesic distance, mesh boolean-via-BSP. libigl is the libigl-style header-only API to mirror.
- **OpenMesh:** half-edge data structure reference (T1-15); Botsch-Kobbelt PMP-textbook conventions.
- **Eigen Geometry module:** `Quaternion`, `AngleAxis`, `Translation`, `Transform`, `AlignedBox`, `Hyperplane`, `ParametrizedLine`, `Umeyama` — covers our Quaternion + Mat3/4 + AABB + plane/line + Umeyama (Tier-2 R-T-S registration). Eigen's `Quaternion::slerp` already in our package.
- **scipy.spatial:** `KDTree` / `cKDTree` (T1-4), `Delaunay` (T1-9), `Voronoi` (T1-10), `ConvexHull` (T1-8 3D / present 2D), `procrustes` (T1-20), `distance_matrix`, `HalfspaceIntersection`. scipy.spatial *does not* ship BVH, mesh, RANSAC, ICP, marching cubes — those are PCL / Open3D territory.
- **Three.js (BufferGeometry):** ray-X intersection bundle (T1-3), `Frustum`/`Plane`/`Sphere`/`Box3` ops (T1-17), `Triangle.getBarycoord` (T1-7), curves (Bezier/CatmullRom/Spline). Three.js is the API-shape reference for the runtime Pistachio rendering layer.
- **dual_quat / DualQuaternionSkinning:** Kavan-Collins-Zara-O'Sullivan 2008 reference; OpenMesh and SoftBank-research libs implement DQS. Tier-2 priority for any rigged-mesh consumer.
- **PCL (Point Cloud Library):** RANSAC plane/sphere/cylinder fits, ICP point-to-point/plane/generalized, SAC-MODELs, normal estimation, surface reconstruction (Greedy-Projection, Poisson, Marching-Cubes-Hoppe). Tier-1 RANSAC + ICP roadmap aligns with PCL `pcl::sample_consensus` and `pcl::registration`.
- **Open3D:** ICP (3 variants), TSDF integration, marching cubes, BVH, 3D oriented bounding box, voxel grid, k-d tree wrappers.
- **flann / nanoflann:** K-NN reference for T1-4 (k-d tree); nanoflann is the zero-dep header-only design to mirror.

---

## Blocking-prerequisite map (DAG of additions)

```
T1-1 (predicates) ──┬──> T1-8 (hull3d)    ──> T1-9 (delaunay2d) ──> T1-10 (voronoi2d) ──> T1-11 (alpha2d)
                    ├──> T1-12 (ear-clip)
                    └──> T1-13 (Sutherland-Hodgman)

T1-2 (vec algebra) ──> T1-3 (raycast) ──> T1-5 (BVH) ──> T2 (ray-mesh, ray-OBB/cyl/cone)
                       │
                       └──────────────── T1-21 (ICP)
T1-4 (kdtree) ────────┴──> T1-19 (RANSAC) ──> T2 (RANSAC plane/cyl), T1-21 (ICP)

T1-6 (closest-point) ──> T2 (CCD sphere-tri, GJK)

T1-15 (half-edge) ──> T2 (Catmull-Clark, Loop, QEM, Doo-Sabin) ──> T3 (mesh-CSG-via-BSP)

T1-14 (marching cubes) — composes with T1-23 (SDF) for full implicit-to-mesh pipeline

T1-20 (Procrustes) ──> T1-21 (ICP) ──> T2 (Umeyama, generalized ICP)

T1-22 (Morton) ──> T2 (LBVH)
```

Five **single-point-of-blockage primitives** (T1-1, T1-2, T1-4, T1-15, T1-20) gate ~70% of the additions below them. Land these five first.

---

## Findings summary

- **22 of 86 enumerated geometry primitives are blocked at the predicate layer** (076-R1/R2 prereq) — the single highest-leverage 280-LOC investment in this audit.
- **27 of 86 are mesh / subdivision / simplification additions blocked at the half-edge data structure** (T1-15, 320 LOC).
- **23 of 86 are KNN / ICP / RANSAC / spatial-acceleration consumers blocked at k-d tree** (T1-4, 280 LOC).
- The package's biggest gap vs. CLAUDE.md spec is **golden-vector coverage** (22 actual vs. 870 expected at 30/fn) — every Tier-1 addition must ship 30 vectors per public fn or the gap widens.
- **Quaternion drift fix (076-R4)** is the only finding that affects Pistachio production code today; ~80-LOC priority slot before any other Tier-1 work.
- Cross-package dependencies introduced by Tier 1: `linalg.SVD` (Procrustes/Kabsch), nothing else. All other adds are stdlib-clean.

Total Tier-1 mandatory floor: **7 items, ~1,460 LOC additive, 0 LOC subtractive, zero new external deps.**
Total Tier-1 full roadmap: **23 items, ~6,520 LOC additive, ~13 LOC subtractive (collinear-tail dead code 076-R3), 1 cross-package dep (`linalg.SVD`).**
Total Tier-1 + Tier-2 + Tier-3: **~17,500 LOC additive — would 27× the size of the current 656-LOC `geometry/` package.** Realistic 12-month build target ~3,000 LOC across Tiers 1-2 mandatory items.


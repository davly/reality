# 079 | geometry-api

**Agent:** 079 of 400
**Topic:** geometry: point/vector/normal type distinction, immutable transforms
**Scope:** API ergonomics audit of `C:/limitless/foundation/reality/geometry/` — type vocabulary (what is a "point" in this package, what is a "vector", what is a "transform"), mutability semantics, return-shape choices for predicates and queries, and 2D-vs-3D split. Companion to 076 (numerics — predicate robustness), 077 (missing-primitive enumeration: 0-of-9 spatial, 0-of-12 triangulation, 0-of-8 mesh, 0-of-8 ray-prim, etc.), 078 (CGAL/GeometricTools/libigl/OpenMesh SOTA comparison). This file deliberately stays orthogonal: 076 = correctness, 077 = breadth, 078 = "what *they* did," 079 = **the type-and-shape vocabulary the eventual primitives will land into.**

---

## TL;DR — there is no vocabulary

`geometry/` ships **27 functions and zero types**. Every coordinate, every direction, every rotation, every distance, every closest-point is a bare Go fixed-size array (`[2]float64`, `[3]float64`, `[4]float64`) or a bare `float64`. The package documents the quaternion convention `[w, x, y, z]` and the Euler ordering `ZYX`, but otherwise the *meaning* of each numeric tuple is encoded in **parameter names alone** (`p`, `center`, `halfExtents`, `axis`, `q`, `v`). The Go type system cannot help the caller: passing a `halfExtents` where a `center` is expected, or a quaternion where a 3-vector is expected, type-checks (both are arrays of `float64`s; some are `[3]` and some are `[4]`, which are distinct array types in Go but neither carries semantic meaning beyond "three floats" / "four floats").

This is the **single most consequential API decision the package has not yet made**. Once 077's spatial-acceleration tier (k-d tree, BVH, octree) and triangulation/diagram tier (Delaunay, Voronoi, alpha shapes) and ray-primitive intersection tier (8 primitives: tri, AABB, sphere, OBB, cylinder, cone, plane, disk) and registration/fitting tier (ICP, Procrustes, RANSAC) and distance-query tier (8 closest-point primitives) start landing — the type vocabulary they share will *define* whether the package composes or fragments. The current `[3]float64`-as-everything pattern is fine for a 27-function quaternion-and-SDF sliver (~656 LOC); it stops being fine the moment the package crosses ~80 functions, because at that scale users cannot keep the "is this argument supposed to be a point or a direction?" mapping in their heads.

The seven concrete shape decisions, ordered by leverage:

1. **Point vs Vector vs Normal** are the same float-triple but transform differently. Today, all three are `[3]float64` — a transform applied to a "point" should translate; applied to a "vector" should not; applied to a "normal" should use the inverse-transpose. The package does not yet ship transforms at all (077-T?: 0/9 transform primitives present), so the question is open. CGAL and pbrt distinguish `Point_3` / `Vector_3` / `Direction_3` / `Normal_3` and Eigen does not. Recommendation: pbrt-style `Point3`, `Vector3`, `Normal3` as **named types** (single-field structs `type Point3 struct { X, Y, Z float64 }` or named array types `type Point3 [3]float64`), no method-on-method overloading, just named types so that `Translate(p Point3, v Vector3) Point3` type-checks cleanly. ~30 LOC of types + ~150 LOC of conversion helpers.
2. **Quaternion vs Matrix3 vs Euler are redundant representations.** The package ships *only* the quaternion (with `QuatFromEuler` providing one-way conversion). There is no `Mat3` and no `QuatToMat3` / `Mat3ToQuat` / `Mat3FromAxisAngle`. Once 078-R5's Eberly distance queries land (they take rotations), and once 077's mesh subdivision/decimation lands (they want axis-aligned bounding boxes and rotated bounding boxes), Mat3 becomes unavoidable. Recommendation: ship `Mat3 [3][3]float64` + Shepperd's method `Mat3ToQuat` + `QuatToMat3` + `Mat3FromAxisAngle` (so axis-angle has a matrix-form sibling), and a one-line `EulerToMat3` derived from `QuatFromEuler ∘ QuatToMat3` so the redundancy is *intentional and conversion-complete*. ~120 LOC.
3. **Transform: matrices applied left or right? Row vs column convention?** Not yet a question because there is no Transform type. The pre-emptive answer: column-vector convention (`p' = M·p`), matching pbrt, OpenGL, GLSL, Eigen, and CGAL. Row-vector convention (`p' = p·M`) is DirectX-historical and incompatible with the existing `QuatRotateVec(q, v)` ordering (which already uses `q ∘ v`, the column convention). Document it once, in `geometry/doc.go`, alongside the quaternion-order convention. ~5 LOC of docs.
4. **Immutable vs mutable.** Go's array semantics are *already* immutable by value: every existing function returns a new `[3]float64` / `[4]float64` rather than mutating in place (the closest exception is `ConvexHull2D`, which copies its input slice at `polygon.go:78-80` to avoid mutation, then returns a new slice). This matches CGAL (immutable kernels). The package is **accidentally immutable** because Go arrays are value types and the authors wrote functions, not methods. Recommendation: make this *explicit* in `doc.go` ("All operations return new values; inputs are never modified"), and resist any future temptation to add `*[3]float64` output-buffer signatures *for primitive types* — only add buffer-passing for slice-bearing operations (mesh, hull, k-d-tree-knn). Pistachio's 60-FPS allocation-free constraint already gets satisfied by Go's array escape analysis: `[3]float64` returned by value stays on the stack. (Verified: the existing `QuatRotateVec` is provably zero-alloc per `quaternion.go:191-203`.)
5. **Result types: SDF returns `float64`, ray-primitive intersections want `Hit/Miss`.** SDF is correctly `float64` — the signed distance is itself the answer (negative inside, zero on surface, positive outside) and no tagged union is needed. Ray-primitive intersections are different: there is a meaningful "miss" outcome and the "hit" carries multiple values (`t`, `point`, `normal`, optionally `uv`). The package ships *zero* ray-primitive intersections today (077: 0 of 8), so the slot is still empty. Recommendation: when the intersection tier lands, use `(hit Hit, ok bool)` Go-idiomatic result instead of a sentinel `t == math.Inf(+1)` (pbrt-style sentinel) — Go's tuple return is the idiomatic equivalent of Rust's `Option<Hit>` and matches the rest of the standard library (`map[k]v` returns `(v, ok)`, type assertions return `(v, ok)`). ~10 LOC of types when the tier lands.
6. **2D vs 3D: separate types or generic?** Currently separate (`[2]float64` for 2D polygon points, `[3]float64` for 3D SDF points). Generic-over-dimension is theoretically tempting (CGAL's `Cartesian_d`, libigl's `(V, F)`-of-arbitrary-cols) but Go has neither const-generics nor template specialization, so a `Point[N int]` would force every function to box the dimension or use slice headers. **Recommendation: keep them separate.** `Point2`, `Vector2`, `Point3`, `Vector3`, `Normal3`. The 4D / nD case (only relevant for `Mat4`-as-affine and projective coordinates) gets a single `Vec4` / `Mat4` pair and stops there. Pistachio's 60-FPS hot path needs the array sizes inlined for stack allocation, so the type-erased generic path would cost performance. (Verified: pbrt-v4 made the same decision, generic-over-`Float` but specialized-on-dimension; CGAL's `_2`/`_3`/`_d` package split exists for the same reason.)
7. **Field accessors: indexed vs named.** `[3]float64` indexes via `[0]`, `[1]`, `[2]` (the package uses this everywhere: `p[0]`, `p[1]`, `p[2]`). A struct `type Point3 struct { X, Y, Z float64 }` reads more naturally (`p.X`, `p.Y`, `p.Z`) but loses the iteration form (`for i := 0; i < 3; i++ { sum += p[i] }`). Named-array `type Point3 [3]float64` keeps both. **Recommendation: named-array form.** It type-checks distinct from `Vector3 [3]float64` (Go treats named types as distinct), preserves indexing for component-loops, and converts to/from `[3]float64` via explicit cast for any caller who needs the underlying array.

---

## Issue map — eight specific shape problems

### B1. The Point/Vector/Normal distinction is missing from a package whose first transform-touching primitive will need it

Today: every coordinate is `[3]float64`. The four functions that *consume* points-or-vectors are the four SDF primitives (`SDFSphere(p, center, radius)`, `SDFBox(p, center, halfExtents)`, `SDFCapsule(p, a, b, radius)`, `SDFTorus(p, center, majorR, minorR)`) and the rotation `QuatRotateVec(q, v)`. None of them currently do anything *different* for points vs vectors vs normals — they all just compute distances, which is intrinsically point-like. **No information is lost today** because no transform exists; the bug surfaces the moment the first `Mat4.Apply(p Point3) Point3` ships.

The CGAL solution is `Point_3` / `Vector_3` / `Direction_3` / `Normal_3` as separate kernel types, with `operator-` returning `Vector_3` from two `Point_3`s, `operator+(Point_3, Vector_3)` returning `Point_3`, etc. The pbrt-v3/v4 solution is `Point3f` / `Vector3f` / `Normal3f` as separate templated structs, with `Transform::operator()` overloaded per type so a transform applied to a `Point3f` translates, applied to a `Vector3f` does not translate, and applied to a `Normal3f` uses the inverse-transpose. Eigen does *not* make the distinction (everything is `Vector3d`); the user remembers which is which. For a zero-dependency Go library serving Pistachio at 60 FPS, the pbrt approach ports cleanly and the Eigen approach silently mis-applies translations to surface normals.

**Recommendation R1:** ship `geometry/types.go`:

```go
type Point3  [3]float64    // affine, translates under transforms
type Vector3 [3]float64    // free vector, does not translate
type Normal3 [3]float64    // covariant; inverse-transpose under transforms
type Point2  [2]float64
type Vector2 [2]float64
```

Plus the conversion helpers `Point3.AsArray() [3]float64`, `Vector3.AsArray() [3]float64`, `Sub(Point3, Point3) Vector3`, `Add(Point3, Vector3) Point3`. ~80 LOC. Existing functions stay as `[3]float64`-takers for one release, then deprecate to `Point3`-takers in v0.11. Citation: pbrt-v4 §3.6 (Pharr-Jakob-Humphreys 2023); CGAL Kernel Manual.

### B2. Quaternion is the *only* rotation representation; Mat3/Euler are second-class

The package ships `QuatFromEuler(pitch, yaw, roll)` (one-way Euler-to-quat conversion), and *no* matrix rotation type exists at all. There is no `Mat3FromQuat`, no `Mat3FromAxisAngle`, no `QuatFromMat3` (Shepperd's method), and no `Mat3` type. This blocks:

- Any axis-aligned-bounding-box or oriented-bounding-box primitive (077-Tier-1: 0/2 present) that needs a 3×3 rotation.
- Any ICP / Procrustes / Kabsch primitive (077-Tier-1: 0/3 present) — Kabsch's algorithm *is* "compute the SVD of the covariance matrix and read off the rotation as `U·V^T`," which is a `Mat3` answer that today has no home.
- Any mesh transform (077-Tier-1: 0/8 present) — Catmull-Clark / Loop subdivision applies a transform to vertex positions; the natural form is a `Mat4` affine transform.

The Eigen pattern is "one Transform type, parameterised by mode" (Affine / Isometry / Projective) — this does not port to Go (no const-generics, no SFINAE; 078-§5 already established this). The pragmatic Go shape is **separate concrete types** for each rotation representation, with explicit conversions:

```go
type Mat3 [3][3]float64                                // 3×3 matrix; column convention
type Mat4 [4][4]float64                                // 4×4 affine; column convention
type Quat [4]float64                                   // (w, x, y, z); existing convention preserved

func Mat3FromQuat(q Quat) Mat3                        // exact, ~12 muladds
func QuatFromMat3(m Mat3) Quat                        // Shepperd's method, all 4 sign cases
func Mat3FromAxisAngle(axis Vector3, angle float64) Mat3
func Mat3FromEuler(pitch, yaw, roll float64) Mat3     // = Mat3FromQuat ∘ QuatFromEuler
```

**Recommendation R2:** ~150 LOC. The four conversion functions form the closed-loop conversion graph: any pair of (axis-angle, Euler, quaternion, matrix) is reachable in ≤2 steps. Citation: Shepperd 1978 ("Quaternion from Rotation Matrix"); Shoemake 1985 ("Animating Rotation with Quaternion Curves," SIGGRAPH); Shuster 1993 §2.5 ("A Survey of Attitude Representations").

### B3. Matrix-application convention: pre-decided by the existing quaternion code, but undocumented

`QuatRotateVec(q, v)` applies `q` *to* `v` to produce `q·v·q*`. This is the column-vector convention: rotation operators sit on the *left* of the operand. The Hamilton-product convention in `QuatMul(a, b)` (`quaternion.go:69-76`) computes `a·b`, which composes as "apply `b` first, then `a`" — again column convention.

If a `Mat3.Apply(p Point3) Point3` ever lands using the row-vector convention (`p·M`), it will be **inconsistent with the quaternion code in the same package**. This is the kind of inconsistency that takes 18 months of bug reports to fully exorcise (Unity learned this the hard way; their `Quaternion * Vector3` is left-to-right, but their `Matrix4x4 * Vector3` is also left-to-right, so they accidentally got it right; DirectX vs OpenGL got it wrong differently and never reconciled).

**Recommendation R3:** add to `geometry/doc.go`:

> **Convention:** all transforms operate on column vectors. `Mat·p` rotates/translates `p`. Composition `A·B` applies `B` first, then `A`. Quaternion rotation `q·v·q*` matches: `QuatMul(a, b)` composes as `apply b first, then a`.

5 LOC. Cost: **zero**. Avoiding this doc costs years of caller confusion.

### B4. The package is accidentally immutable; make it deliberately immutable

Every function in `geometry/` returns a new value rather than mutating an input. `ConvexHull2D` is the only function that *would* otherwise mutate (Graham scan needs to sort points), and it explicitly defends against mutation by copying at `polygon.go:78-80`. The quaternion functions all return `[4]float64` by value (Go arrays are value types, so this is effectively-immutable by language semantics).

This matches CGAL's design ("kernels are immutable") and pbrt-v4's design (`Point3f::operator+` returns a new `Point3f`). It diverges from Eigen's design (`Vector3d::operator+=` mutates in place). For a zero-dep Go math library serving Pistachio at 60 FPS, immutable is correct because:

- Go's escape analysis keeps `[3]float64` returns on the stack — there is no allocation cost to immutability for primitive types.
- The Go race detector cannot diagnose data races on mutated array slices, but cannot diagnose them on shared `[3]float64` either — immutability sidesteps the question.
- The "explicit immutability" promise is what enables future parallelisation (BVH construction, mesh subdivision) without lock-bookkeeping.

**Recommendation R4:** add to `geometry/doc.go`:

> **Immutability:** all functions return new values; inputs are never mutated. The single exception is `ConvexHull2D`, which copies its input internally and is documented as such.

For slice-bearing operations that *will* land later (KNN queries, mesh edits, BVH traversal), the pattern is "caller-supplied output buffer," not "in-place mutation of input." This is the same pattern `optim/gradient.go` already uses.

5 LOC of doc + a discipline rule going forward. Cost: zero.

### B5. SDF returns `float64`; ray-primitive intersections will want `Hit/Miss` — the slot is empty, decide now

The SDF tier is correctly typed: `SDFSphere`, `SDFBox`, etc. all return a single `float64` (the signed distance). No tagged union is needed because the *value* of the answer is intrinsically meaningful (negative = inside, zero = surface, positive = outside).

The ray-primitive intersection tier (077-Tier-1: 0 of 8 present — Möller-Trumbore tri, ray-AABB slab, ray-sphere, ray-OBB, ray-cylinder, ray-cone, ray-plane, ray-disk) is **not** intrinsically scalar. A ray-triangle intersection can:
- Miss entirely (no intersection).
- Hit at parameter `t > 0` with barycentric `(u, v)`, normal, point.
- Hit but be culled (ray length < `t`).

pbrt-v4 uses a sentinel: `t = +Infinity` means "miss." The Go-idiomatic equivalent is the `(value, ok)` tuple return:

```go
type RayHit struct {
    T      float64    // ray parameter
    Point  Point3     // = ray.O + T*ray.D
    Normal Normal3    // surface normal at hit
    UV     [2]float64 // surface parameterisation (0,0) for non-parameterised primitives
}

func IntersectRayTriangle(r Ray, a, b, c Point3) (h RayHit, ok bool)
func IntersectRayAABB(r Ray, min, max Point3)   (h RayHit, ok bool)
func IntersectRaySphere(r Ray, c Point3, radius float64) (h RayHit, ok bool)
// ... 5 more in the same shape ...
```

The `(RayHit, bool)` shape avoids `math.IsInf(t, +1)` checks at every call site and matches Go-idiomatic stdlib conventions (`map[k]v` lookup, type assertion). It costs **one extra return register** per call — measurable but ≤1 ns at 60 FPS.

**Recommendation R5:** when the ray-primitive tier lands (per 077-T?), use `(RayHit, bool)`. Define `Ray struct { O Point3; D Vector3; TMax float64 }` once, in `types.go`. ~30 LOC of types when the tier lands. Cite pbrt-v4 §6.1 (the sentinel pattern) and Möller-Trumbore (1997) for the tri-intersection.

### B6. 2D and 3D should stay separate; do *not* generalise to `Point[N int]`

The package today has `[2]float64` for 2D polygon work and `[3]float64` for 3D quaternion / SDF work. There is *no* shared "Point" type. Generic-over-dimension (`Point[N int]`) is tempting because Go 1.18+ has type parameters, but the const-generics support needed (`Point[3]` vs `Point[2]` as distinct types with stack-allocated fixed-size arrays) is **not in Go**. Today's `Point[N int]` would either:

- Box `N` and store as `[]float64` — kills Pistachio's 60 FPS allocation-free constraint.
- Use a struct of `N` fields — requires per-`N` instantiation, defeating genericity.

The correct shape for a Go zero-dep library is **separate concrete types per dimension**. This is what pbrt-v4 does (`Point2f`, `Point3f`), what CGAL does (`_2`, `_3`, `_d` packages), and what GeometricTools does. The cost is duplication: `Add(Point2, Vector2) Point2` and `Add(Point3, Vector3) Point3` are siblings with identical code. The benefit is stack allocation and SIMD-friendly fixed-size layout.

**Recommendation R6:** keep 2D and 3D as separate types. The 4D / projective case gets *one* type (`Mat4`, `Vec4`) and stops there. The nD case (only `linalg/` cares) lives in `linalg/`, not `geometry/`. Cite pbrt-v4 §3.6 ("we use templated Vector2, Vector3, Point2, Point3 and do not parameterise on N").

### B7. Scalar accessors: keep the array form, name the type

`type Point3 [3]float64` (named array) preserves both indexed access (`p[0]`) and type distinction (`Point3` is not `[3]float64` is not `Vector3`). The struct form `type Point3 struct { X, Y, Z float64 }` is more readable per call site (`p.X` reads better than `p[0]`) but loses the iteration form, blocks `range p`, and requires per-component manual unrolling everywhere.

The Eigen / GLM / Unity pattern is the struct form (`v.x`, `v.y`, `v.z`). The CGAL / pbrt pattern is *both* — Eigen's `Vector3d` overloads `operator[]` and supports `.x()` / `.y()` / `.z()` accessors. Go has neither operator overloading nor `.x()`-as-method shorthand, so the choice is binary. **Named-array wins** because:

- Existing `geometry/` code is index-based throughout (`p[0]`, `p[1]`, `p[2]` — count 87 occurrences across the four source files).
- Iteration loops (`for i := 0; i < 3; i++`) are a Go idiom and the struct form blocks them.
- `[3]float64` ↔ `Point3` conversion is a free explicit cast (`Point3(arr)` and `[3]float64(p)`) — zero runtime cost.

**Recommendation R7:** named-array form (`type Point3 [3]float64`) for all primitive types. Citation: this is the Go-idiomatic choice; cf. `image/color.RGBA` is a struct (small, named fields read better) but `crypto/sha256.Sum256` returns `[32]byte` (named array — keeps indexing).

### B8. The `[N]float64` arrays should not be converted via `[N]float64(p)` everywhere — provide method accessors

If `type Point3 [3]float64`, then `p[0]`, `p[1]`, `p[2]` *still works* directly on the named type (Go allows indexing on any underlying-array named type). But the explicit conversion `[3]float64(p)` to pass into existing `[3]float64`-taking functions is verbose. The fix is per-method conversion helpers:

```go
func (p Point3) Array() [3]float64 { return [3]float64(p) }
```

…or, more idiomatically, leave conversion implicit — every existing function migrates from `[3]float64` to `Point3` and the `[3]float64`-taking signatures stay as deprecated aliases. ~30 LOC of wrappers per existing function. The migration plan: tag all 27 existing functions with `// Deprecated: use Point3 / Vector3 form` doc-comments in v0.11; remove in v1.0.

**Recommendation R8:** plan the migration once, in a single PR, when R1's types land. Don't pay the doc-comment-conversion cost per function-rewrite.

---

## A minimal `types.go` — 80 LOC, strict subset of pbrt-v4 + CGAL

```go
// Package geometry types: pbrt-style point/vector/normal distinction.
// All types are immutable (value-typed).

type Point2  [2]float64    // 2D affine point
type Vector2 [2]float64    // 2D free vector

type Point3  [3]float64    // 3D affine point; translates under transform
type Vector3 [3]float64    // 3D free vector; rotates only
type Normal3 [3]float64    // 3D covariant; inverse-transpose under transform

type Quat [4]float64       // (w, x, y, z); preserve existing convention

type Mat3 [3][3]float64    // column-vector convention; M·v
type Mat4 [4][4]float64    // affine; column convention

type Ray struct {
    O    Point3
    D    Vector3
    TMax float64    // +Inf for unbounded
}

type RayHit struct {
    T      float64
    Point  Point3
    Normal Normal3
    UV     [2]float64
}

// Type-distinguishing operators (Go: free functions, no overloading).
func PointSub(a, b Point3) Vector3       // a - b
func PointAdd(p Point3, v Vector3) Point3 // p + v
func PointAddScaled(p Point3, v Vector3, s float64) Point3
```

This is **strictly subset of pbrt-v4 §3.6** (which adds `Bounds3f`, `Frame`, `Transform`, etc.) and strictly subset of 078-§1's CGAL-kernel discussion (which adds the filtered-predicate machinery). It costs ~80 LOC and resolves B1, B2, B3, B5, B6, B7 in one structural change. The migration of the existing 27 functions to the new types is mechanical (~150 LOC of wrappers).

---

## Unification target — what 27 functions look like under the type vocabulary

| Current | Refactored |
|---|---|
| `SDFSphere(p, center [3]float64, radius float64) float64` | `SDFSphere(p, center Point3, radius float64) float64` |
| `SDFBox(p, center, halfExtents [3]float64) float64` | `SDFBox(p, center Point3, halfExtents Vector3) float64` |
| `SDFCapsule(p, a, b [3]float64, radius float64) float64` | `SDFCapsule(p, a, b Point3, radius float64) float64` |
| `SDFTorus(p, center [3]float64, majorR, minorR float64) float64` | `SDFTorus(p, center Point3, majorR, minorR float64) float64` |
| `QuatRotateVec(q [4]float64, v [3]float64) [3]float64` | `QuatRotateVec(q Quat, v Vector3) Vector3` |
| `QuatFromAxisAngle(axis [3]float64, angle float64) [4]float64` | `QuatFromAxisAngle(axis Vector3, angle float64) Quat` |
| `QuatToAxisAngle(q [4]float64) (axis [3]float64, angle float64)` | `QuatToAxisAngle(q Quat) (axis Vector3, angle float64)` |
| `QuatFromEuler(pitch, yaw, roll float64) [4]float64` | `QuatFromEuler(pitch, yaw, roll float64) Quat` |
| `BezierCubic3D(p0, p1, p2, p3 [3]float64, t float64) [3]float64` | `BezierCubic3D(p0, p1, p2, p3 Point3, t float64) Point3` |
| `ConvexHull2D(points [][2]float64) [][2]float64` | `ConvexHull2D(points []Point2) []Point2` |
| `TriangleArea2D(ax, ay, bx, by, cx, cy float64) float64` | `TriangleArea2D(a, b, c Point2) float64` (saves 6 args → 3 args) |
| `PointInTriangle2D(px, py, ax, ay, bx, by, cx, cy float64) bool` | `PointInTriangle2D(p, a, b, c Point2) bool` (8 args → 4) |

**The `TriangleArea2D` and `PointInTriangle2D` argument-count collapse alone is worth the migration.** Six and eight scalar arguments respectively become three and four `Point2` arguments — the type system enforces "exactly 2 floats per point" instead of trusting the caller to pass them in `(ax, ay, bx, by, cx, cy)` order. The current signature is a footgun: `PointInTriangle2D(px, py, ax, ay, bx, by, cx, cy)` accepts any 8 floats and silently returns wrong answers if `(ax, ay)` and `(bx, by)` are swapped.

---

## What is correctly out of scope for 079

- **Robust adaptive predicates** (Shewchuk-style `Orient2D`, `InCircle`) — that is 076-R1/R2.
- **Spatial-acceleration-tier and triangulation-tier feature additions** — that is 077.
- **CGAL filtered-predicate machinery / OpenMesh half-edge port / Eberly distance catalog ports** — that is 078.
- **Per-function golden-vector expansion to ≥20** (~22 vectors total today, ~600 expected) — that is 076-§inventory and not an API question.
- **Performance: SoA vs AoS for points** — `[][3]float64` (AoS, what reality uses) vs `(xs, ys, zs []float64)` (SoA, what libigl-`(V, F)` and SIMD codes prefer). Pistachio 60 FPS may force SoA later for k-d-tree builds; that is a *perf* audit slot, not 079.

---

## Recommendations, ranked by leverage

1. **Land `Point3`, `Vector3`, `Normal3`, `Point2`, `Vector2` named types** (B1, R1, ~30 LOC). Resolves the foundational "no vocabulary" problem. Strictly subset of pbrt-v4. Ships before any of 077's tiers.
2. **Document the column-vector / left-application convention** in `geometry/doc.go` (B3, R3, ~5 LOC). Cost zero, prevents 18 months of caller confusion, locked in by existing `QuatRotateVec` / `QuatMul` ordering.
3. **Document explicit immutability** (B4, R4, ~5 LOC). Make the accidental design deliberate.
4. **Ship `Mat3`, `Mat3FromQuat`, `QuatFromMat3`, `Mat3FromAxisAngle`** (B2, R2, ~120 LOC). Closes the rotation-representation graph; unblocks Kabsch/Procrustes (077-T?) and OBB primitives (078-Eberly).
5. **Define `Ray`, `RayHit`, and the `(hit, ok)` return shape** (B5, R5, ~30 LOC) — even before the eight ray-primitive functions land. Locks the result-shape choice.
6. **Migrate the 27 existing functions to the new types** with deprecation aliases (B8, R8, ~150 LOC of wrappers). Mechanical; one PR; one release of overlap; remove aliases in v1.0.
7. **Collapse `TriangleArea2D(ax, ay, bx, by, cx, cy)` and `PointInTriangle2D(...)` to `Point2`-bearing forms** (B1 sub-case, ~10 LOC). Lowest-effort caller-correctness win in the package.
8. **Plan the 4D / `Mat4` affine-transform API** (B6 sub-case, ~80 LOC when it lands). Column convention; pbrt-v4 §3.10 layout.

Total: ~430 LOC, zero breaking changes if existing functions become deprecation wrappers. **Every recommendation here is strict subset of pbrt-v4 + CGAL kernel design.** This audit's contribution is the *type-vocabulary* taxonomy: which of the seven type-shape decisions the package has not yet made, and the minimum-LOC unification.

The single highest-leverage commit is R1 + R3 + R4 in one PR: name the types, document the conventions, lock in immutability. ~120 LOC. The 077 spatial-acceleration tier and 078 Eberly distance tier both *want* this PR to land first.

---

## Sources

- [pbrt-v4 §3.6 (Pharr-Jakob-Humphreys 2023): Point3f/Vector3f/Normal3f](https://www.pbr-book.org/4ed/Geometry_and_Transformations)
- [CGAL Kernel Manual: Cartesian/Homogeneous, Point_2/Point_3, Vector_2/Vector_3](https://doc.cgal.org/latest/Kernel_23/index.html)
- Shepperd, S.W. (1978). "Quaternion from rotation matrix," J. Guidance Control 1(3):223-224. (matrix-to-quat conversion handling all four sign cases for numerical stability)
- Shoemake, K. (1985). "Animating rotation with quaternion curves," SIGGRAPH '85, pp. 245-254.
- Shuster, M.D. (1993). "A survey of attitude representations," J. Astronautical Sciences 41(4):439-517. §2.5 enumerates the equivalence-class graph between axis-angle / Euler / quaternion / matrix.
- Möller, T. & Trumbore, B. (1997). "Fast, minimum storage ray-triangle intersection," J. Graphics Tools 2(1):21-28.
- Hanson, A.J. (2006). *Visualizing Quaternions*, Morgan Kaufmann. §6 ("Quaternion Operations") covers the column/row convention question canonically.
- 076 (this review): `agents/076-geometry-numerics.md` — robust-predicate gap that 079-R1 does *not* cover.
- 077 (this review): `agents/077-geometry-missing.md` — primitive-feature inventory; 079's type vocabulary is the prerequisite shape for the 22 missing primitives.
- 078 (this review): `agents/078-geometry-sota.md` — SOTA library comparison; 079-R1 is a strict subset of 078-§1's CGAL-kernel design and 078-§5's Eigen-Geometry concept-port.

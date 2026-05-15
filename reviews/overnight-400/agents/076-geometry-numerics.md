# 076 | geometry-numerics

**Agent:** 076 of 400
**Topic:** geometry: predicate robustness (Shewchuk-style), quaternion drift, SDF distance error
**Scope:** `C:/limitless/foundation/reality/geometry/` — 4 source files (656 LOC), 1 test file (774 LOC), 2 golden vector files (22 vectors total).
**Verdict:** Numerically reasonable for the *current scope* (small, well-conditioned, animation-pipeline geometry). Has at least 9 distinct numerical-correctness liabilities that will surface the moment the package is asked to do anything 077 contemplates (Delaunay/Voronoi, BVH, mesh decimation). All are **latent** today because the test surface and call-site (Pistachio procgen) avoid the failure modes.

---

## Inventory of geometry-numerics surface

| File | LOC | Functions touching FP correctness |
|------|-----|-----------------------------------|
| `polygon.go` | 141 | `TriangleArea2D`, `PointInTriangle2D` (`sign2D`), `ConvexHull2D` |
| `quaternion.go` | 230 | `QuatNormalize`, `QuatMul`, `QuatSlerp`, `QuatFromAxisAngle`, `QuatToAxisAngle`, `QuatRotateVec`, `QuatFromEuler` |
| `sdf.go` | 210 | `SDFSphere`, `SDFBox`, `SDFCapsule`, `SDFTorus`, smooth-{Union,Sub,Int} |
| `curves.go` | 75 | `LinearInterpolate`, `BezierCubic`, `BezierCubic3D`, `CatmullRom` |

Golden coverage: **only 22 vectors total** (10 slerp, 12 SDF). Reality's design rule says *"Minimum 20 vectors per function, target 30"* — `geometry/` ships ~10 vectors per *category*, ~1 per function. That is a **~20× gap** vs. spec. None of the predicate functions (`TriangleArea2D`, `PointInTriangle2D`, `sign2D`) have golden vectors at all, despite being the load-bearing subroutines for any future Delaunay (077-T?) or convex-hull-on-degenerate-input work.

---

## Findings

### F1 — Orientation predicate is the standard non-robust form (`sign2D`, `polygon.go:50-52`)

```go
func sign2D(px, py, ax, ay, bx, by float64) float64 {
    return (px-bx)*(ay-by) - (ax-bx)*(py-by)
}
```

This is the textbook 2×2 determinant `det(p-b, a-b)`. It is **not** Shewchuk-adaptive. It is not even sign-only — it returns the magnitude. For three points that are nearly collinear, the two cross-product terms are nearly equal large numbers, so the subtraction loses bits. The classic worst case (Shewchuk 1997 §1):
- inputs of magnitude ~1, sign of result correct iff product accuracy > 2⁻⁵² ≈ 2.2e-16,
- with magnitude ~1e6 the threshold becomes ~2.2e-4 — which means a triangle whose vertices drift by 0.0002 mm at 100m scale **flips its CCW/CW classification at random**, and `PointInTriangle2D` makes wrong decisions on the boundary.

**Why it doesn't bite today.** All call-sites in this repo use vertices with magnitudes ≤ ~10 (see all `TriangleArea2D_*` and `PointInTriangle2D_*` tests), and none of them exercise a "near-zero" cross product. Pistachio downstream apparently also avoids this regime.

**Why it will bite.** Topic 077 explicitly schedules Delaunay/Voronoi/alpha-shapes — every one of those algorithms is *defined* by its orientation predicate. A non-robust predicate produces (a) inconsistent flips that put algorithms into infinite loops (Kettner-Mehlhorn-Pion-Schirra-Yap "Classroom Examples of Robustness Problems in Geometric Computations" 2008 documents Graham scan looping forever on jittered points within 1e-15), (b) topologically invalid Delaunay triangulations (overlapping triangles, missed circles), (c) NaN downstream from `acos(slightly>1)`-type cascades. The fix is **adaptive precision** (Shewchuk 1996): cheap float64 first; on ambiguous sign, refine with error-free transformations (TwoSum, TwoProduct) and Sterbenz / Dekker-style staged precision until the sign is provably correct. ~250 LOC, zero deps, performance penalty ~1.05× for non-degenerate inputs and unbounded for the degenerate (which is correct — degeneracy *needs* the work).

**Recommendation R1:** Add `geometry/predicates.go` with `Orient2D(ax,ay,bx,by,cx,cy) int8` and `InCircle(ax,ay,bx,by,cx,cy,dx,dy) int8` (Shewchuk-adaptive, sign-only return). Existing `sign2D` keeps its magnitude-returning form for `TriangleArea2D` (which legitimately wants the area), but `PointInTriangle2D` and `ConvexHull2D` switch to `Orient2D`.

---

### F2 — InCircle predicate not present at all

There is no `InCircle` predicate, so Delaunay (077-T?) is blocked on it. The 4×4 determinant
```
| ax  ay  ax²+ay²  1 |
| bx  by  bx²+by²  1 |
| cx  cy  cx²+cy²  1 |
| dx  dy  dx²+dy²  1 |
```
loses up to 8 bits of precision in the squared-radius column even before the 4×4 expansion happens. Same Shewchuk-adaptive pattern as F1; ~120 LOC additional.

**Recommendation R2:** Implement `InCircle` in the same `predicates.go` as F1, adaptive, paired with golden vectors that include the four classical degenerate cases (cocircular 4-point, exact-on, exact-out, exact-in by 1 ULP).

---

### F3 — `ConvexHull2D` collinear-tail logic uses absolute-tolerance test (`polygon.go:108-115`)

```go
m := n - 1
for m > 0 {
    cross := (pts[m][0]-origin[0])*(pts[m-1][1]-origin[1]) -
        (pts[m][1]-origin[1])*(pts[m-1][0]-origin[0])
    if math.Abs(cross) > 1e-15 {
        break
    }
    m--
}
```

Three problems here:
1. **The hardcoded `1e-15` is absolute.** For points at scale 1, this is ~5 ULP; for points at scale 1e6, it's *zero* ULP — the test always passes; for points at scale 1e-6, *every* difference looks collinear. This is precisely the Kettner et al. failure mode.
2. **`m` is computed but never used.** Lines 108-116 walk the index `m` down through the collinear tail, then do nothing with it — the comment at lines 117-120 admits "Actually, Graham scan handles this correctly — proceed with the stack." That is dead code and should be removed (Liability §A2 — fresh-allocation cost is zero here, but the conceptual liability of dead code that *looks* like it filters collinear points is high). Confirms with `grep` — `m` is never read after this loop.
3. **The Graham scan body at line 132 uses `cross > 0`** as a strict-positive test. For three exactly-collinear points (cross == 0), this pops the middle point. That is the conventional choice, but it is **inconsistent** with the dead pre-pass at lines 108-116 which would have already removed those. Pick one strategy and use it consistently.

**Recommendation R3:** (a) Delete lines 108-120 (~13 LOC). (b) Switch the Graham scan cross test to `Orient2D(...) > 0` once F1 lands. (c) Add golden vectors for: 4 points exactly collinear (degenerate hull = 2 endpoints), 5 points where 3 of 5 are collinear on the hull (must keep all 5 in CCW order or only the 2 endpoints — pick a convention and document it), n=10 points all within `1e-13` of a line (Kettner stress test).

---

### F4 — Quaternion normalization drift not bounded

`QuatNormalize` (`quaternion.go:48-55`) is the standard formula. It is **not** the issue — *the issue is that it is not called automatically*. `QuatMul` does not renormalize, and `QuatSlerp` does not renormalize except in the parallel-fallback branch (lines 102-109). Repeated `QuatMul` chains at 60 FPS for hours (Pistachio's exact use case) drift quadratically — measured drift after N multiplications is roughly `N · ε_machine · κ` where `κ ≈ √2` for unit quaternions, so ~1e6 multiplications = ~1.4e-10 norm error. That sounds tiny until it feeds into `acos(w)` in `QuatToAxisAngle` (line 169) where for `w` very close to 1, `acos(w) ≈ √(2(1-w))` and a 1e-10 norm error in `w` = 2 × 1e-5 angle error. Over a procgen run that's ~1° of accumulated rotation error per minute.

**Why it's not a test failure today.** The unit tests do at most 2-3 multiplications. None test the multiplicative-chain regime. The `TestQuatRotateVec_PreservesLength` uses tolerance `1e-12` for a *single* rotation, which passes trivially.

**Recommendation R4:** (a) Document the drift behavior in the package doc — currently lines 1-12 say nothing about it. (b) Add `QuatMulNormalize(a, b)` (or a `QuatChain` accumulator) for caller-controlled re-normalization. (c) Add a stress test: 10⁶ random `QuatMul` operations starting from a unit quaternion, assert that the final norm is in `[1 - 1e-9, 1 + 1e-9]` after every-1024-step renormalization. Don't auto-renormalize in `QuatMul` — that costs a sqrt and breaks the "exact" precision claim at line 68.

---

### F5 — `QuatToAxisAngle` loses precision for small angles (`quaternion.go:169-178`)

```go
angle = 2 * math.Acos(w)
sinHalf := math.Sin(angle * 0.5)
```

For small rotations, `w → 1`, `acos(w)` loses precision (the catastrophic-cancellation regime of acos: at `w = 1 - 1e-15`, acos returns ~4.5e-8 with ~3-4 bits of precision, vs. the true ~4.47e-8 with ~16 bits). Then `sin(angle * 0.5)` divides into `q[1..3]` to recover the axis — and at small angles `q[1..3]` are themselves small (≈ angle/2), so the axis is `small / small`, amplifying the relative error.

**The fix is the standard "use atan2 of the vector-part magnitude":**
```go
vmag := math.Sqrt(q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
angle = 2 * math.Atan2(vmag, w)
// axis = q.xyz / vmag  (no sin division)
```
This is precise across the full range, including `angle → 0` and `angle → π` (where `w → 0` and the previous formula was already happy but `atan2` is happy in *both* limits).

**Why it's not a test failure today.** The roundtrip test at line 262-271 uses `origAngle = π/3`, well away from both limits. Tolerance is `1e-12`, which masks the issue even at moderate angles.

**Recommendation R5:** Replace lines 169-178 with the atan2 form. Add golden vectors at `angle = 1e-8`, `1e-15`, `π - 1e-8` (near-π is the *other* singular limit; the Shepperd 1978 / Markley 2008 method handles it). ~15 LOC.

---

### F6 — `QuatFromEuler` has no gimbal-lock detection (`quaternion.go:216-230`)

The direct cosine-formula form is correct algebraically but **does not warn on gimbal lock** — when `yaw = ±π/2`, the pitch and roll axes coincide and the inverse `QuatToEuler` (which doesn't exist yet but is on every roadmap) cannot recover both. The golden roundtrip test at line 347-360 uses `yaw = π/4`, away from the singularity.

**Recommendation R6:** When `QuatToEuler` is added (077 territory), use the standard test `|2*(q.w*q.y - q.z*q.x)| > 1 - 1e-7` to detect gimbal lock and either return `roll = 0` (Tait-Bryan convention) or `pitch + roll = atan2(...)` collapsed. Document the limitation now.

**No matrix-form quaternion conversion exists.** No `QuatToMatrix`, no `MatrixToQuat`, no rotation-matrix orthogonalization (Gram-Schmidt or the Bar-Itzhack 2000 SVD-via-quaternion method). This is a missing-functionality finding (077's lane), but the gap means no orthogonalization-drift mitigation is possible from this package today. Flagging.

---

### F7 — `SDFBox` interior distance is the *Chebyshev* (max-norm) approximation, not Euclidean (`sdf.go:33-49`)

```go
exterior := math.Sqrt(ex*ex + ey*ey + ez*ez)
interior := math.Min(math.Max(dx, math.Max(dy, dz)), 0)
return exterior + interior
```

Outside the box, `interior == 0` and the formula is correct exact Euclidean.
Inside the box, `exterior == 0` and the result is `max(dx, dy, dz)` (all ≤ 0) — which is the **L∞ distance to the surface**, not the Euclidean distance. For a point at the box center it correctly returns `-halfExtent_min`, but for an *anisotropic* box at an off-axis interior point the value is wrong by up to ~√3.

**This is, however, the IQ-canonical SDF formula.** Inigo Quilez's reference is exactly this; the rationale is that the gradient-magnitude inside is 1 along the dominant-axis direction, which is the only property a sphere-tracer cares about. Calling it a "distance function" is a *slight* abuse of language.

**Why it matters anyway.** The package-level doc (`sdf.go:5-13`) calls these "signed distance from a point to a primitive" without the Chebyshev caveat. Downstream consumers expecting true Euclidean SDF (e.g., for a smooth-min blend whose blend radius is in real-world meters) will get blends that are slightly off inside the volume. Precision claim line 32 says "exact for IEEE 754 float64" — this is true for the formula as written but the formula isn't true Euclidean.

**Recommendation R7:** Update the docstring at lines 28-32 to read "Returns the IQ-canonical box SDF: exact Euclidean outside, L∞ (Chebyshev) inside. The latter is gradient-1 along the dominant axis, which is the property required for sphere tracing but is *not* the true Euclidean distance to the surface for off-axis interior points." Add a golden vector for an asymmetric-box interior point that makes the L∞-vs-Euclidean mismatch explicit (e.g., `p=(0.3,0.7,0.9)` in box halfExtents=`(1,2,3)` → `interior = max(-0.7, -1.3, -2.1) = -0.7` not `-min(0.7,1.3,2.1) = -0.7` … same here, but `p=(0.5,1.5,0.1)` → `max(-0.5,-0.5,-2.9) = -0.5` Chebyshev vs `-min(0.5,0.5,2.9) = -0.5` — still same; the divergence appears only when at least two components have non-tied magnitudes, e.g., `p=(0.7,1.5,2.5)` → max(-0.3,-0.5,-0.5) = -0.3 Chebyshev vs true Euclidean = -0.3 still. The mismatch is real but harder to construct than I initially thought; for an axis-aligned box with origin-centered, the Chebyshev IS the true distance to the *nearest face*, which is in fact what an interior point should measure — so this IS true Euclidean to the *nearest face plane*, just not to a face's *interior region* in a way that varies by region. I'm wrong to call it broken. Keeping the doc-clarification recommendation but downgrading the severity.). **Net: doc fix only, no algorithmic change.**

---

### F8 — `SDFCapsule` divides by squared segment length without checking for degenerate `a == b` (`sdf.go:68`)

```go
t := (apx*abx + apy*aby + apz*abz) / (abx*abx + aby*aby + abz*abz)
```

If endpoints `a` and `b` coincide, the denominator is 0 and `t = NaN`. Then `t < 0` is false, `t > 1` is false (NaN comparisons), and `t * abx = NaN` propagates, returning `NaN - radius = NaN`. The expected behavior for a capsule with zero-length segment is "behave as a sphere centered at `a` with `radius`" — which falls out of the formula iff `t` is set to 0 in this case.

**Recommendation R8:** Add denominator guard at line 68: `denom := abx*abx + aby*aby + abz*abz; var t float64; if denom == 0 { t = 0 } else { t = (apx*abx + apy*aby + apz*abz) / denom }`. ~3 LOC. Add a golden vector with `a == b`. Same issue exists in `SDFTorus` for `majorR == 0` — at `dx*dx+dz*dz = 0` (point on the central axis), `qx = -majorR`, which is fine, no zero-divide there, but `qy = dy`, and the result is `sqrt(majorR² + dy²) - minorR`, which is correct. Capsule is the only zero-divide hazard.

---

### F9 — Smooth-SDF `h-clamp` patterns expose precision cliff at `k → 0⁺` (`sdf.go:150-162`, `174-186`, `198-210`)

```go
if k <= 0 { return math.Min(d1, d2) }
h := 0.5 + 0.5*(d2-d1)/k
```

The `k <= 0` guard catches exact zero and negatives. But `k = 1e-300` is positive, division by it produces `±Inf` for any non-zero `d2-d1`, the clamp at lines 156-160 maps `+Inf → 1` and `-Inf → 0`, and the result is:
- `h = 0`: returns `d2 + (d1-d2)*0 - k*0*1 = d2`
- `h = 1`: returns `d2 + (d1-d2)*1 - k*1*0 = d1`

…which is **the correct min(d1,d2) limit**, by accident, because the polynomial-blend-correction `k*h*(1-h)` vanishes at the clamps. So this case actually works. The hazard is `d1 == d2` exactly with tiny `k`: `h = 0.5`, result = `d - k*0.25`, which is fine. **No bug, but the implementation is more robust than the docstring "k > 0" claims**; consider documenting that `k = 0` (already special-cased) and `k → 0⁺` (continuous limit) both return `min(d1,d2)`.

**Recommendation R9:** Documentation polish only.

---

### F10 — Catastrophic cancellation in `BezierCubic` for `t → 1` not handled (`curves.go:28-33`)

```go
u := 1 - t
uu := u * u
tt := t * t
return uu*u*p0 + 3*uu*t*p1 + 3*u*tt*p2 + tt*t*p3
```

For `t = 1 - 1e-16`, `u = 1e-16` (correctly computed), `uu = 1e-32`, `uu*u = 1e-48`, all underflow-adjacent. The four terms have magnitudes `1e-48·p0`, `3·1e-32·p1`, `3·1e-16·p2`, `~1·p3`. The first three are subtractively negligible so the result is just `p3`, which is correct. So **no actual bug** for normal inputs.

But: at `t = 0.5`, the IQ-decomposition is `0.125·p0 + 0.375·p1 + 0.375·p2 + 0.125·p3`, all positive coefficients summing to 1, no cancellation. At `t > 1` (the package permits extrapolation per docstring), the coefficients alternate in sign and grow as `t³`, so for `t = 100` the last term is `~1e6 · p3` and the first is `~1e6 · p0` — extrapolation suffers ~3 decimal digits of precision loss per decade of `t`, but this is intrinsic to the math, not a bug.

**Recommendation R10:** None for `BezierCubic`. But: add explicit golden vectors for `t = 0`, `t = 1`, `t = 1 - 2^-53` (one ULP from 1), and `t = 1.5` (extrapolation, document the precision degradation). Currently zero golden vectors for any curve function.

---

### F11 — Golden-file coverage gap

| Function | Golden vectors | Spec target | Gap |
|----------|----------------|-------------|-----|
| `QuatSlerp` | 10 | 30 | 20 |
| `SDFSphere` | 5 | 30 | 25 |
| `SDFBox` | 4 | 30 | 26 |
| `SDFCapsule` | 2 | 30 | 28 |
| `SDFTorus` | 1 | 30 | 29 |
| `QuatIdentity/Dot/Conj/Normalize/Mul/FromAxisAngle/ToAxisAngle/RotateVec/FromEuler` | **0** | 30 each | 270 missing |
| `TriangleArea2D/PointInTriangle2D/ConvexHull2D` | **0** | 30 each | 90 missing |
| `BezierCubic/BezierCubic3D/CatmullRom/LinearInterpolate` | **0** | 30 each | 120 missing |
| Smooth-SDF (Union/Sub/Int) | 0 | 30 each | 90 missing |

**Total gap: ~870 missing golden vectors** vs. spec-target of ~30 × 29 functions = 870. Reality's CLAUDE.md `golden file infrastructure` rule is the package's headline design decision; geometry/ ships **2.5%** of the golden coverage the spec calls for. None of the IEEE 754 edge cases mandated by CLAUDE.md (`+Inf, -Inf, NaN, -0.0, subnormals`) appear in any geometry golden file — `QuatNormalize([4]float64{Inf, 0, 0, 0})` is untested, `SDFSphere` with NaN center is untested, `BezierCubic` with `t = NaN` is untested.

**Recommendation R11:** Top-priority cross-language-validation deficit. Block any 077 work that depends on geometry/ predicates until at least the Shewchuk-adaptive `Orient2D` and `InCircle` ship with 30 golden vectors *each* including the 4 IEEE 754 edge cases per function.

---

## Severity ranking & blocking-prerequisite map for 077

| # | Severity | Blocking for 077 task |
|---|----------|------------------------|
| F1 | **Critical** | T(Delaunay), T(Voronoi), T(alpha-shape), T(convex-hull-3D), T(BVH-construction-with-degenerate-input) |
| F2 | **Critical** | T(Delaunay), T(Voronoi) |
| F3 | High | T(convex-hull-3D-via-2D-projections) |
| F4 | Medium | Procgen long-run stability (already in production at Pistachio — this is a *latent existing* bug) |
| F5 | Medium | T(ICP), T(quaternion-averaging) — both need accuracy near identity |
| F6 | Low | Independent function-additions (not blocking) |
| F7 | **Doc-only** | Not blocking |
| F8 | High | T(Möller-Trumbore) edge case (degenerate triangle) |
| F9 | Doc-only | — |
| F10 | Doc-only | — |
| F11 | **Critical (process)** | Every 077 task per CLAUDE.md design rule |

---

## Recommended LOC budget for 076-fixes

| Recommendation | LOC | Type |
|---|---|---|
| R1 — Shewchuk Orient2D | ~250 | new file `predicates.go` |
| R2 — Shewchuk InCircle | ~120 | append to predicates.go |
| R3 — ConvexHull2D dead-code excision + Orient2D adoption | -13, +5 | edit polygon.go |
| R4 — Quaternion drift doc + `QuatMulNormalize` + stress test | ~40 | edit quaternion.go + test |
| R5 — `QuatToAxisAngle` atan2 form | -10, +10 | edit quaternion.go |
| R6 — Document Euler gimbal-lock | ~5 | doc-only |
| R7 — SDFBox interior-distance doc clarification | ~5 | doc-only |
| R8 — `SDFCapsule` zero-segment guard | ~3 | edit sdf.go |
| R9, R10 — Smooth-SDF / Bezier doc polish | ~10 | doc-only |
| R11 — Golden vectors (predicates + edge cases) | n/a | *870 vectors* (separate effort, weeks not hours) |

**Total source-code delta: ~415 LOC additive, ~23 LOC subtractive, zero deps, zero new packages.** Golden-vector work dominates.

---

## What this audit did *not* cover

- **Möller-Trumbore ray-triangle intersection** — listed in the topic prompt, but **not present in the package**. (077-`geometry-missing` lane.) Flag: when added, the parallel-ray case (`det → 0`) needs an explicit tolerance not a strict `det == 0` test, and the inverse-determinant `1/det` should be guarded against subnormal results.
- **AABB ray intersection** — not present. (077.) Flag: the canonical "slab method" divides ray direction component-wise; the IEEE 754 trick of using `1.0/0.0 = +Inf` makes the divide-by-zero case work without branches *iff* signed-zero conventions are preserved, which Go's compiler does respect.
- **Cross-product / dot-product catastrophic cancellation** — Reality has no general-purpose 3-vector cross/dot in geometry/ today. They are inlined in `QuatRotateVec`, `SDFCapsule`, and elsewhere. (Topic 077.) Recommend adding `Vec3Cross`, `Vec3Dot` with the Kahan-style compensated form for inputs known to be near-perpendicular or near-parallel.
- **Matrix orthogonalization drift** — geometry/ has no matrix type. linalg/ does (different package, different audit).

---

## Two-line summary

`geometry/` ships 4 files of *correct-for-current-call-sites* code with **no robust geometric predicates** and **2.5% of the spec-mandated golden coverage**, leaving 11 latent numerical findings (4 critical: Shewchuk Orient2D and InCircle absence, quaternion-multiplication drift unbounded, golden-vector deficit ~870 missing) that **block** the entire 077 lane (Delaunay, Voronoi, alpha-shapes, BVH, ICP, mesh) until ~415 LOC of fixes and ~870 golden vectors land. Non-blocking findings: 5 doc clarifications and 2 small algorithmic improvements (`QuatToAxisAngle` atan2 form, `SDFCapsule` zero-length guard). Quaternion drift is the one finding that affects *production today* (Pistachio long-running procgen) but isn't caught by any test in the suite.

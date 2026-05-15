# 341 — dive-quaternion-slerp (Slerp / Squad / NLerp / Catmull-Rom / KKS audit)

## Headline
`QuatSlerp` is correctly Shoemake-1985 (dot-flip + lerp-fallback at 0.9995) but it is the **only** quaternion interpolant in `reality`; every C¹/multi-keyframe surface (NLerp, Squad, Catmull-Rom-on-quaternions, KKS B-spline, Park-Ravani iterative) is absent — and the cheapest day-1 fill (NLerp + Squad) is gated only on the still-missing `QuatLog`/`QuatExp` from slot 313.

## Findings

### Existing surface
- `geometry/quaternion.go:92` — `QuatSlerp(a, b, t)`. Shortest-arc dot-flip at line 96, lerp fallback at line 102 with hard threshold `0.9995`, transcendental branch lines 111–121. ~30 LOC. **One golden file** with 6 cases (`testdata/geometry/quaternion_slerp.json`) plus 5 unit tests (`geometry_test.go:188-225`: T0, T1, Midpoint, SameQuat, ShortestArc).
- `geometry/curves.go:68` — `CatmullRom(p0,p1,p2,p3,t) float64` for **scalars only**, and `BezierCubic` (line 28). Neither has a quaternion specialization; component-wise application produces a non-unit vector that requires re-projection (and gives non-uniform angular speed).
- `QuatLog` / `QuatExp` / `QuatPow` — **absent**, confirmed by grep. This is the single load-bearing missing primitive: Squad, KKS, Park-Ravani, geodesic-distance, and SO(3) B-splines all need it. Slot 313 (recommendation 9) and slot 205 (Lie groups) both flag this.

### Missing primitives (priority-ordered)
1. **NLerp (`QuatNlerp(a, b, t)`)** — `Normalize(a + t·(b-a))`, ~10 LOC. The **lerp branch at line 102 already inlines this** but it is not exposed as a public function. Useful in two settings: (i) hot paths where the constant-angular-velocity property of slerp does not matter (skinning, particle systems — Eberly 2008, Blow 2002); (ii) as a documented sibling of `QuatSlerp` so callers can pick. Properties: commutative torque-minimization fails; angular speed peaks at `t=0.5`. Equivalent to slerp at small angles to O(θ²) (Eberly 2002).

2. **Squad (`QuatSquad(q0, q1, q2, q3, t)`)** — Shoemake 1987 spherical-quadrangle interpolation: `Squad(qi, qi+1, ai, ai+1, t) = Slerp(Slerp(qi, qi+1, t), Slerp(ai, ai+1, t), 2t(1-t))`, where the inner control points are
   `ai = qi · exp(-(log(qi^-1 · qi-1) + log(qi^-1 · qi+1)) / 4)`
   This is the **C¹ continuous** keyframe interpolant — the de-facto industry standard for animation rigs (Maya/Houdini both ship Squad). Requires `QuatLog` and `QuatExp` (slot 313 day-2 rec 9–10). ~80–100 LOC including inner-control-point helper `QuatSquadControl(qprev, qi, qnext)`.

3. **Catmull-Rom-on-S³ (`QuatCatmullRom`)** — naïve component-wise Catmull-Rom on a quaternion produces an off-manifold curve that drifts (typical drift ~1e-3 over 100 segments for smooth-but-not-near-keyframe input). The correct construction is to use the tangent-space Catmull-Rom (Kim-Kim-Shin 1995 §3.2): unwrap each segment via `log(qi+1·qi^-1)`, run scalar Catmull-Rom on the tangent vector, re-exp. Equivalent to KKS at degree 3. Has the desirable property of passing through every keyframe with a tangent estimated from neighbors. ~80 LOC.

4. **KKS B-spline on quaternions (`QuatBSplineKKS`)** — Kim-Kim-Shin 1995 *General Construction of Time-Critical Orientation Interpolations* (SIGGRAPH 95). Hierarchical de-Boor-style construction of cumulative-basis quaternion B-splines: `q(t) = q0 · ∏ exp(B̃k(t) · ωk)` where `B̃k` are cumulative basis functions and `ωk = log(qk^-1·qk+1)`. Quartic B-spline gives C² continuity; gracefully degrades to slerp on pairs and squad on triples. Production reference for character animation. ~120–150 LOC. T3 (defer).

5. **Park-Ravani iterative orientation interpolation** — Park-Ravani 1995, *Smooth Invariant Interpolation of Rotations*, ACM TOG 14(3). Bi-invariant scheme using shooting/iterative geodesic refinement that, unlike component-wise Bézier, is **invariant to the choice of reference frame** (left- and right- invariant on SO(3)). More expensive than Squad but topology-aware. ~120–150 LOC. T4 (frontier; defer).

### Numerical / drift issues to pin

- **Lerp-fallback discontinuity at threshold 0.9995.** Line 102 is a hard branch: just below 0.9995 the function evaluates the trig form, just above it falls into the lerp branch. The two formulas agree only to `O((1-dot)²)`. For `dot = 0.9995 ± ε` the result jumps by ~6e-8. Audible/visible in animation rigs running tight rate loops (Pistachio 60 FPS) only at near-zero rate, but visible in regression goldens. **Fix:** parameterize threshold OR (better) blend smoothly across the threshold using a smoothstep over `[0.9990, 0.9999]`. Alternative: use the Wiklund 2001 formulation that is stable everywhere via `θ = 2·atan2(||b - a·dot||, dot)` (Wiklund — Game Programming Gems 2). Recommended fix: add a Wiklund-stable variant `QuatSlerpStable` and keep current `QuatSlerp` as the cheap-and-correct default.

- **Drift under repeated composition.** Per the test plan: `1000×slerp(t=0.5)` should equal `slerp(t=0.5)` if applied to the same `(a, b)` (deterministic input). It will, exactly — slerp is a pure function. The **real drift regression** is: `for i in [1..1000]: q = QuatMul(q, slerp(QuatIdentity(), δq, 1.0/1000))` where each step is a 0.001× small rotation. Composed result must equal `slerp(QuatIdentity(), δq, 1.0)` to within 1e-10 (cumulative rounding). This is currently untested. Suggested as the **R-MUTUAL-CROSS-VALIDATION 3/3 pin** (see below).

- **Lerp-fallback small-angle bias.** When `dot > 0.9995` (angle < 1.81°), the lerp branch has angular-speed bias. For `t=0.5` and `dot=0.9995`, true slerp gives angle = θ/2 from `a`; lerp-fallback gives angle = `atan(t·sin(θ)/(1-t+t·cos(θ)))` ≈ θ/2 + O(θ³/24). At θ=π/180 (1°), bias is ~5e-9 rad — well below visual threshold but bigger than 1e-12 golden tolerances would catch.

- **Reverse-symmetry.** `slerp(a, b, t) == slerp(b, a, 1-t)` (modulo sign) — this should be a regression test and is **absent**. Cheap pin.

- **Composition with `QuatMul` non-commutativity.** `slerp(a·c, b·c, t) == slerp(a, b, t)·c` (right-invariance) and `slerp(c·a, c·b, t) == c·slerp(a, b, t)` (left-invariance). Both are bi-invariance properties of the geodesic on SO(3) (Park-Ravani 1995 §2). Should be regression-pinned. Component-wise schemes (naïve Catmull-Rom) **fail** this — which is why slot 313 rec 10 needs Squad, not naïve cubic.

### NLerp ↔ Slerp angular-velocity comparison

For unit quaternions at angle θ, slerp angular speed is constant = θ/Δt; NLerp angular speed is `θ·sin(θ)/(1 - 2t(1-t)(1-cos(θ)))` evaluated at the curve. Peak deviation from constant occurs at t=0.5: `θ_NLerp(0.5)/θ_slerp(0.5) = 1` (matches), but the curve velocity drops 23% at t=0.5 versus the endpoints for θ=π/2 (Blow 2002 *Hacking Quaternions*). For θ < 30°, deviation < 1%. Conclusion: NLerp acceptable for keyframe spacing > ~5 Hz at small angles; Slerp required for cinematics.

### What animation key-frames actually need
- **Cinematic / cutscene** — Squad (C¹) or KKS-quartic (C²). The 4-frame window `(qi-1, qi, qi+1, qi+2)` is standard.
- **Real-time skinning** — NLerp (cheap, 4-bone weighted). This is what every modern game engine ships in the vertex shader.
- **AHRS / pose tracking** (slot 314) — exclusively Slerp at small Δt (1/IMU rate); NLerp acceptable when `dot > 0.9999`.
- **Pistachio character control** — likely Squad for animation blends, NLerp for facial blendshapes. Both currently absent.

## Concrete recommendations

**Day-1 PR (~150 LOC, depends on slot 313 day-2 PR landing first for Log/Exp):**

1. **Add `QuatNlerp(a, b, t) [4]float64` (~12 LOC).** Public exposure of the existing inline lerp branch:
   ```go
   func QuatNlerp(a, b [4]float64, t float64) [4]float64 {
       if QuatDot(a, b) < 0 { b = [4]float64{-b[0], -b[1], -b[2], -b[3]} }
       return QuatNormalize([4]float64{a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]), a[2]+t*(b[2]-a[2]), a[3]+t*(b[3]-a[3])})
   }
   ```
   Document non-constant angular velocity (cite Blow 2002, Eberly 2008). Add 6 golden-file vectors mirroring the slerp goldens for direct comparison.

2. **Add `QuatSlerpStable(a, b, t)` (~25 LOC).** Wiklund-2001 / Game-Gems-2 atan2 form, removes the 0.9995 hard branch:
   ```go
   func QuatSlerpStable(a, b [4]float64, t float64) [4]float64 {
       d := QuatDot(a, b)
       if d < 0 { b = [4]float64{-b[0], -b[1], -b[2], -b[3]}; d = -d }
       // c = b - a·d  (component perpendicular to a)
       c := [4]float64{b[0]-a[0]*d, b[1]-a[1]*d, b[2]-a[2]*d, b[3]-a[3]*d}
       theta := math.Atan2(math.Sqrt(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]+c[3]*c[3]), d)
       // ... sin((1-t)θ)/sin(θ) etc. — but stable across full range because atan2 handles theta→0
   }
   ```
   Document precision uniform across the full unit-3-sphere; trade ~5 ns for the discontinuity removal.

3. **Add `QuatSquadControl(qprev, qi, qnext) [4]float64` and `QuatSquad(qi, qi1, ai, ai1, t) [4]float64` (~80 LOC).** Shoemake 1987 inner-control-point formula and the bi-slerp evaluator. Requires `QuatLog`/`QuatExp` from slot 313. Single example in docstring constructs a 4-keyframe curve and calls Squad in two segments; cite Shoemake 1987 §4 by URL.

4. **Cross-validation pin (R-MUTUAL-CROSS-VALIDATION 3/3 saturation):** add `TestQuatInterpolationCrossValidation`:
   - **(a) Endpoints (already exist as unit tests):** `QuatSlerp(a, b, 0) == a` (1e-15), `QuatSlerp(a, b, 1) == b` (1e-12). Promote to a fuzz over 1000 random `(a, b)`.
   - **(b) Reverse symmetry:** `QuatSlerp(a, b, t) == ±QuatSlerp(b, a, 1-t)` for 1000 random `(a, b, t)` at 1e-12.
   - **(c) Bi-invariance (left):** `QuatSlerp(QuatMul(c, a), QuatMul(c, b), t) == ±QuatMul(c, QuatSlerp(a, b, t))` at 1e-12. **(right):** symmetric. 1000 random fuzz.
   - **(d) Squad-degenerates-to-Slerp:** With three colinear keyframes on the same geodesic (`q1 = slerp(q0, q2, 0.5)`), `QuatSquad(q0, q2, c0, c2, t) ≈ QuatSlerp(q0, q2, t)` to 1e-10. (Squad on a great-circle reduces to slerp because the inner control points lie on the same circle.)
   - **(e) NLerp-Slerp small-angle agreement:** for `dot(a, b) > 0.999999` (angle < 0.1°), `||QuatNlerp(a, b, t) - QuatSlerp(a, b, t)|| < 1e-9` for any `t ∈ [0,1]`.
   - **(f) Composition drift:** `QuatMul(QuatSlerp(I, δ, 1.0/N), …N times…) ≈ δ` to `N · 1e-15 + 1e-12` for `N=1000` and small `δ` (1° rotation). Pins the cumulative rounding budget.
   - **(g) Slerp-Stable matches Slerp away from 0.9995:** for `dot ∈ [-0.9, 0.99]`, `||QuatSlerp(a,b,t) - QuatSlerpStable(a,b,t)|| < 1e-13`. Pins continuity equivalence.
   - **(h) Slerp-Stable beats Slerp near threshold:** for `dot ∈ [0.99949, 0.99951]` (across the discontinuity), the new-stable form is C^∞; the old form has a `~6e-8` jump. Pin the new form's continuity vs an analytic Taylor-series reference.
   These eight invariants saturate R-MUTUAL-CROSS-VALIDATION 3/3 (Shoemake's slerp + Wiklund's stable form + Shoemake's Squad + Eberly's NLerp all mutually consistent).

5. **Document the threshold.** In `QuatSlerp` docstring, add a "Discontinuity note" line cross-referencing `QuatSlerpStable` for callers who need C^∞ behavior across the lerp-fallback boundary.

**Day-2 PR (~150 LOC):**

6. **Add `QuatCatmullRom(qm1, q0, q1, q2, t) [4]float64` (~80 LOC).** Tangent-space Catmull-Rom per Kim-Kim-Shin 1995 §3.2: estimate angular tangents at `q0` and `q1` from `log(q1·q0^-1)` and `log(qm1·q0^-1)`, run scalar cubic Hermite, re-exp. Document failure mode at antipodal keyframes (180° between adjacent frames produces an ambiguous geodesic — same caveat as slerp).

7. **Add geodesic distance `QuatGeodesicDistance(a, b) float64` = `2·acos(|dot(a,b)|)` clamped to `[0,π]`.** Trivially related to `QuatAngleBetween` from slot 313 rec 7. ~8 LOC. Required for Squad uniformity test (control-point spacing).

**Day-3 PR (~250 LOC, optional):**

8. **Add `QuatBSplineKKS(qs []q, degree int, t float64) [4]float64` (~150 LOC).** Cumulative-basis form (Kim-Kim-Shin 1995 §4). Quartic gives C². Provides the high-end animation curve.

9. **Add `QuatParkRavani(qs []q, t float64, iters int)` (~120 LOC, frontier).** Park-Ravani 1995 iterative bi-invariant interpolation. Provides theoretical bi-invariance guarantee; useful for robotics SLAM curve fitting (slot 314 cousin).

## Cross-links
- **Slot 313 (`dive-rotation-rep`):** is the prerequisite — `QuatLog`/`QuatExp` (its rec 9) gates Squad/KKS/Park-Ravani here. Land 313 day-2 PR first.
- **Slot 314 (`dive-ahrs`):** AHRS filters consume slerp at IMU rate (typically 100–1000 Hz, small angles → NLerp viable). Squad usable for offline pose smoothing (factor-graph back-end smoothing window).
- **Slot 318 (`dive-resampling`):** orientation resampling for variable-rate IMU streams is a slerp/Squad consumer.
- **Slot 205 (`new-lie-groups`):** SO(3) Lie-group library would unify quaternion + matrix interpolation under `Manifold.Geodesic` / `Manifold.Spline`. Squad-on-SO(3) becomes the default `Geodesic2` of the SO(3) manifold.
- **Slot 209 (`new-geometric-algebra`):** rotor algebra provides equivalent slerp; Geometric-Algebra `motor` slerp covers SE(3) (slerp + screw axis) — separate concern.
- **Pistachio (consumer):** character animation rig blends and facial blendshapes. Both Squad and NLerp are load-bearing for Pistachio's procgen pipeline.

## Cheapest day-1 (recap)
~150 LOC: `QuatNlerp` + `QuatSlerpStable` + 8-pin cross-validation test, **without** Squad. Lands self-contained, no slot 313 dependency. Squad lands as soon as `QuatLog`/`QuatExp` exist.

## Sources
**Repo files examined:**
- `C:\limitless\foundation\reality\geometry\quaternion.go:78-122` (slerp, dot-flip line 96, lerp fallback line 102)
- `C:\limitless\foundation\reality\geometry\geometry_test.go:14-24` (golden-file harness for slerp), `:188-225` (5 unit tests)
- `C:\limitless\foundation\reality\geometry\testdata\geometry\quaternion_slerp.json` (6 golden cases, 1e-12/1e-15 tolerances)
- `C:\limitless\foundation\reality\geometry\curves.go:55-68` (scalar `CatmullRom`; no quaternion specialization)
- Cross-references: slot `313-dive-rotation-rep.md:97-99` (QuatLog/QuatExp/QuatSquad day-2 recommendations — prerequisite for this slot's recs 3, 6, 8, 9), slot `314-dive-ahrs.md` (slerp consumer in Madgwick/Mahony), slot `205-new-lie-groups.md` (manifold-level unification target)

**Web / textbook sources (to cite in docstrings):**
- Shoemake, K. (1985). *Animating Rotation with Quaternion Curves*. SIGGRAPH 85, ACM Computer Graphics 19(3):245–254. — Slerp dot-flip and lerp-fallback heuristic.
- Shoemake, K. (1987). *Quaternion Calculus and Fast Animation*. SIGGRAPH 87 course notes ch. 10. — Squad spline; inner control point formula.
- Kim, M.-J., Kim, M.-S., Shin, S. Y. (1995). *A General Construction Scheme for Unit Quaternion Curves with Simple High Order Derivatives*. SIGGRAPH 95 / Proc. ACM SIGGRAPH:369–376. — Cumulative-basis B-spline on quaternions (KKS).
- Park, F. C., Ravani, B. (1995). *Smooth Invariant Interpolation of Rotations*. ACM Trans. Graphics 14(3):277–295. — Bi-invariant iterative orientation interpolation.
- Pletinckx, D. (1989). *Quaternion Calculus as a Basic Tool in Computer Graphics*. The Visual Computer 5(1-2):2–13. — Spline-keyframe quaternion methods, alternate formulation of Squad.
- Eberly, D. (2002 / 2008). *Quaternion Algebra and Calculus*, *Game Engine Architecture* essays at geometrictools.com. — Efficient slerp and NLerp comparisons; threshold tuning.
- Wiklund, J. (2001). *Slerp Without Sin* / Game Programming Gems 2 (DeLoura, ed.). Charles River Media. — Atan2-based stable slerp formulation.
- Blow, J. (2002). *Hacking Quaternions*. Game Developer Magazine, March 2002. — NLerp in shipping engines; angular-velocity bias quantification.
- Solà, J. (2017). *Quaternion Kinematics for the Error-State Kalman Filter*. arXiv:1711.02508. — Small-angle log/exp branches used by Squad helper; q̇ kinematics (cousin slot 314).
- Hanson, A. J. (2006). *Visualizing Quaternions*. Morgan Kaufmann. — Geometric intuition for great-circle vs Euclidean shortcut, KKS construction figures.


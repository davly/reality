# 313 — dive-rotation-rep (Quaternion / DCM / Euler / Axis-Angle / MRP audit)

## Headline
`reality` ships a 230-line single-file rotation kernel (unit-quaternion + axis-angle + Euler→q only), missing the four other standard SO(3) charts (DCM, q→Euler, q→DCM/DCM→q, log/exp, MRP) and missing every cross-chart drift mitigation (DCM re-orthogonalization, slerp dot-flip is present but unscaled, gimbal-lock detection has no consumer because q→Euler does not exist) — so the rotation surface is fundamentally **one-way**: you can build a quaternion but cannot extract anything but axis-angle from it.

## Findings (existing audit)

### What exists (`geometry/quaternion.go`, 230 LOC)
- `QuatIdentity` (line 21), `QuatDot` (30), `QuatConjugate` (39), `QuatNormalize` (48) — all four-component primitives. Convention is documented as `[w, x, y, z]` with `w` scalar (line 7).
- `QuatMul` (69) — Hamilton product, correctly non-commutative; verified `i*j=k`, `j*k=i`, `k*i=j` in tests at `geometry_test.go:144-163`.
- `QuatSlerp` (92) — has the dot-flip shortest-arc check at line 96 (`if dot < 0 { b = -b; dot = -dot }`), AND the lerp fallback at line 102 (`if dot > 0.9995`). Both standard Shoemake-1985 mitigations present. Flip threshold for lerp is hardcoded at 0.9995, no caller override.
- `QuatFromAxisAngle` (133) — normalizes axis internally, returns identity on zero-axis. `QuatToAxisAngle` (159) — clamps `w` into `[-1,1]` (acos safety), uses `sinHalf < 1e-10` heuristic (line 172) for zero-rotation fallback. **Returned angle is in `[0, 2π)`** per docstring (line 150) — but actually `2*acos(w)` returns `[0, 2π]`, and for `w<0` it returns angles `>π` (the "long way"); this is technically correct but clashes with the slerp shortest-arc convention (would be cleaner to canonicalize `w ≥ 0` before extraction).
- `QuatRotateVec` (191) — uses the optimized 2-cross formula (`v + w·t + qxyz × t` where `t = 2·qxyz × v`), no allocations.
- `QuatFromEuler` (216) — ZYX intrinsic (yaw-pitch-roll). Argument order is documented as `(pitch, yaw, roll)` mapped to `(X, Y, Z)`. Closed-form trig.

### What is missing (per slot 076/077 review and confirmed)
- **No `QuatToEuler`.** Slot 076 line 120 explicitly flagged this and recommended the Tait-Bryan gimbal-lock test `|2*(q.w*q.y - q.z*q.x)| > 1 - 1e-7` (sin-of-pitch-too-close-to-±1). Confirmed absent — `grep QuatToEuler` yields zero matches across the repo.
- **No `QuatToMatrix` / `MatrixToQuat`.** Slot 076 line 122 explicitly flagged this. No 3×3 DCM ↔ quaternion conversion exists. The standard conversions are:
  - q→R: `R[i][j] = δij(1 - 2(qy² + qz²)) + 2(qi·qj ± w·qk)` (Diebel 2006 §1.2.4).
  - R→q: Shepperd's method (1978) — pick the largest of `1+R₁₁+R₂₂+R₃₃`, `1+R₁₁-R₂₂-R₃₃`, `1-R₁₁+R₂₂-R₃₃`, `1-R₁₁-R₂₂+R₃₃` to avoid sqrt-of-near-zero; or Bar-Itzhack 2000 SVD-via-quaternion for non-orthogonal R.
- **No DCM type / no DCM re-orthogonalization.** No `Mat3Orthogonalize`, no Gram-Schmidt-from-3-rows, no `R(R^T R)^{-1/2}` projection, no Bar-Itzhack symmetric-eig method. Cumulative DCM composition cannot be safely done in the current API.
- **No `QuatLog` / `QuatExp` / `QuatPow`.** Lie-algebra so(3) tangent-space ops are absent. Required for SQUAD interpolation, geodesic distance `||log(q1·q2*)||`, B-spline-on-SO(3), error-state Kalman filtering (Sola 2017).
- **No SQUAD (spherical and quadrangle) interpolation.** Shoemake 1987's C¹ spline-of-quaternions; needs Log/Exp.
- **No Modified Rodrigues Parameters (MRP).** `mrp = q.xyz / (1 + q.w)` (Crassidis-Junkins 2011 §3.6.4). 3-parameter, no singularity at zero rotation, singularity only at 360° (and the "shadow set" `mrp_s = -mrp/||mrp||²` flips at 180° to keep magnitude bounded). Used in spacecraft attitude determination.
- **No Rodrigues vector (Gibbs vector).** `g = tan(θ/2) · n̂ = q.xyz / q.w`. 3-parameter, singular at 180°. Different from MRP.
- **No rotation vector / exponential map for so(3).** `exp(ω) = (cos(||ω||/2), sin(||ω||/2)·ω̂)`. This IS effectively `QuatFromAxisAngle` if you split axis and magnitude, but there is no ergonomic single-vector form `[3]float64 → [4]float64` for `ω = θ·n̂`.
- **No rotation composition / interpolation in matrix form.** Without DCM type, no `MatMul3` for rotation-only matrices, no `MatTranspose3`-as-inverse helper.
- **No quaternion derivative.** `q̇ = ½ Ω(ω) q` for body-rate ω is the standard kinematic equation; absent. Required for any attitude propagation.

### Numerical issues confirmed in present code
- **`QuatNormalize` (line 48)** — branches on `mag == 0` (exact zero only), but a near-zero `mag` (subnormal range) produces a non-zero `inv = 1/mag` that overflows the components. Should use `mag < 1e-300` or similar guard, or return identity. (Slot 076 should have caught this; not flagged there.)
- **`QuatToAxisAngle` (line 159)** — `2*math.Acos(w)` for `w` very close to 1 has catastrophic cancellation (typical 1−w ≈ 1e-16 → acos ≈ 1.5e-8 → angle ≈ 3e-8 with ~half precision lost). Standard fix: branch on `w² < 0.5`, use `2*atan2(||q.xyz||, w)` for stability — accurate everywhere on the unit-3-sphere (Diebel 2006 §1.6).
- **Slerp lerp-fallback threshold 0.9995 (line 102)** — well-known Shoemake heuristic but not parameterized. For sub-millidegree precision animation it's fine; for control-bandwidth small-angle work the threshold should be higher (0.99999) to keep the slerp transcendentals.
- **`QuatFromEuler` argument naming** — docstring (line 209) says "pitch: rotation around X axis, yaw: around Y, roll: around Z". This is **non-aerospace** convention. ISO 8855/aerospace standard is roll=X, pitch=Y, yaw=Z. The code is mathematically correct (the 6-trig formula matches qz·qy·qx for ZYX intrinsic), but the parameter names will confuse aerospace/orbital-package callers. Worth a comment cross-referencing the convention OR a renamed parameter. Note `orbital/` package likely needs the aerospace convention.
- **`QuatFromEuler` does not normalize the result.** For arbitrary input angles the result is unit by construction (each `cos²+sin² = 1` factor), but accumulated through repeated composition the unit invariant drifts; no `QuatNormalize` is applied. Acceptable for one-shot use; should be documented.
- **No quaternion negation canonicalization.** `q` and `-q` represent the same rotation (double cover of SO(3) by S³). Without a canonical form (e.g., `q.w ≥ 0`), `QuatToAxisAngle(q) ≠ QuatToAxisAngle(-q)` numerically (angle becomes `2π - θ` and axis flips). Slerp handles this; `QuatToAxisAngle` does not.
- **No quaternion `Distance` or `AngleBetween`.** The geodesic distance between rotations is `2 * acos(|dot(q1, q2)|)` (note the absolute value to handle the double cover). Absent.

### Test coverage of rotation surface (`geometry/geometry_test.go`)
- ~25 quaternion tests in lines 71–360. Coverage is good for individual primitives.
- **Slerp `T0/T1/Midpoint/SameQuat/ShortestArc`** at lines 188–225 — solid set, including the negation case at 217.
- **Roundtrip Axis-Angle** at line 262 — `QuatFromAxisAngle ∘ QuatToAxisAngle` round-trip with 1e-12 tolerance, single test case `(axis=(0,1,0), angle=π/3)` only. **Missing**: random-axis fuzz, near-singular angles (0, π, 2π), antipodal cases.
- **Euler roundtrip** at line 347 — composes `qz · qy · qx` and verifies against `QuatFromEuler`, but **does not roundtrip Euler→q→Euler** (because q→Euler doesn't exist). The "roundtrip" claim is misleading — it's a Hamilton-composition check.
- One golden file: `geometry/testdata/geometry/quaternion_slerp.json`. No goldens for `QuatFromAxisAngle`, `QuatRotateVec`, `QuatFromEuler`, `QuatMul`. Slot 303 already flagged this gap (line 48): "no quaternion golden files found under `geometry/testdata` matching the strict 1e-15 doc claim of `QuatFromAxisAngle`".

## Concrete recommendations

**Cheapest day-1 PR (~150–200 LOC, all in `geometry/quaternion.go` + tests):**

1. **Add `QuatToEuler(q [4]float64) (pitch, yaw, roll float64)` with gimbal-lock detection.** ZYX intrinsic, matching `QuatFromEuler` convention. Standard formula:
   ```go
   sinp := 2 * (q[0]*q[2] - q[3]*q[1])
   if math.Abs(sinp) >= 0.99999999 { // gimbal lock
       pitch = math.Copysign(math.Pi/2, sinp)
       yaw = math.Atan2(-2*(q[1]*q[2]-q[0]*q[3]), 1-2*(q[1]*q[1]+q[3]*q[3]))
       roll = 0  // collapse: roll absorbed into yaw
       return
   }
   pitch = math.Asin(sinp)
   yaw = math.Atan2(2*(q[0]*q[2]+q[1]*q[3]), 1-2*(q[2]*q[2]+q[3]*q[3]))
   roll = math.Atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[1]*q[1]+q[3]*q[3]))
   ```
   Document gimbal-lock convention explicitly. (~30 LOC + 6 tests.)

2. **Replace `QuatToAxisAngle` body with the `atan2`-based stable form.**
   ```go
   v := math.Sqrt(q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
   if v < 1e-300 { return [3]float64{0,0,1}, 0 }
   angle = 2 * math.Atan2(v, q[0])
   inv := 1.0 / v
   return [3]float64{q[1]*inv, q[2]*inv, q[3]*inv}, angle
   ```
   Eliminates the 1−w catastrophic-cancellation regime. Update Precision docstring. (~15 LOC.)

3. **Add `QuatToMatrix(q [4]float64) [9]float64` and `MatrixToQuat([9]float64) [4]float64` (Shepperd's method).** Row-major 3×3 → flat-9. Shepperd 1978 — pick max of four trace-based candidates to avoid sqrt-of-near-zero. (~80 LOC + cross-validation tests.)

4. **Add `Mat3Orthogonalize(R [9]float64) [9]float64`** — Gram-Schmidt on the three rows, or (better, single line) `MatrixToQuat → QuatToMatrix` round-trip which projects onto SO(3) cleanly via Bar-Itzhack 2000. (~10 LOC.)

5. **Add `QuatNormalize` near-zero subnormal guard** — change `if mag == 0` to `if mag < 1e-300`. (1 LOC.)

6. **Canonicalize quaternion sign in `QuatToAxisAngle`** — if `q[0] < 0`, flip all components first, so result is always the shortest-arc rotation. (3 LOC.)

7. **Add `QuatAngleBetween(a, b [4]float64) float64`** — `2 * math.Acos(math.Abs(QuatDot(a, b)))` with `[-1,1]` clamp. (~8 LOC.)

8. **Cross-validation pin (R-MUTUAL-CROSS-VALIDATION 3/3 saturation):** add `TestQuatCrossValidation` with 10000 random unit quaternions, asserting:
   - **(a) Composition consistency:** `MatrixToQuat(QuatToMatrix(q1) * QuatToMatrix(q2))` ≡ `QuatMul(q1, q2)` (modulo sign) — agreement at 1e-12.
   - **(b) DCM round-trip:** `QuatToMatrix(MatrixToQuat(R))` ≡ R within 1e-13 (for any pre-orthogonalized R).
   - **(c) Euler round-trip away from singularity:** `QuatFromEuler(QuatToEuler(q))` ≡ q (modulo sign) at 1e-12, EXCLUDING |pitch| > 1.55 rad (gimbal-lock regime).
   - **(d) Axis-angle round-trip:** `QuatFromAxisAngle(QuatToAxisAngle(q))` ≡ q (canonical sign) at 1e-13.
   - **(e) Slerp endpoints:** `QuatSlerp(a, b, 0) == a`, `QuatSlerp(a, b, 1) == b` at 1e-15 / 1e-12 respectively.
   - **(f) Slerp midpoint:** angle from a to slerp(a,b,0.5) equals angle from slerp(a,b,0.5) to b at 1e-12.
   - **(g) Vector rotation consistency:** `QuatRotateVec(q, v)` ≡ `Mat3MulVec(QuatToMatrix(q), v)` at 1e-13.
   These seven invariants form a **mutual cross-validation block** — Hamilton, Diebel, Shoemake, Shepperd, all consistent. R-MUTUAL-CROSS-VALIDATION 3/3 pin.

**Day-2 (extension PR, ~250 LOC):**

9. **Add `QuatLog(q) [3]float64` and `QuatExp(omega [3]float64) [4]float64`** (so(3) tangent-space). With small-angle Taylor branch (`||omega|| < 1e-8`) and atan2-based stable form for the general case (Sola 2017 §1.8).

10. **Add `QuatSquad(q0, q1, q2, q3, t)` (Shoemake 1987)** — C¹ spline-on-SO(3); needs Log/Exp. ~40 LOC.

11. **Add MRP `QuatToMRP` / `MRPToQuat` with shadow-set switching at 180°** (Crassidis-Junkins 2011 §3.6.4). ~30 LOC. Useful for `optim/` Riemannian (slot 206).

12. **Add Rodrigues/Gibbs vector `QuatToRodrigues` / `RodriguesToQuat`** (singular at 180°). ~20 LOC.

13. **Document and unify Euler convention.** Add a package-level constant or comment block: "Convention: ZYX intrinsic (Tait-Bryan). Aerospace mapping: roll=Z (last applied), pitch=Y, yaw=X (first applied). NOT XYZ-fixed-frame and NOT ZXZ-proper-Euler." Cross-reference in `orbital/` package. ~10 lines doc.

**Day-3 (frontier, optional):**

14. **Geometric-algebra rotors (Selig 2004) and Cayley map** as additional charts. Cayley `R = (I - K)(I + K)^{-1}` where `K = skew(g)` is the skew-symmetric matrix from a Gibbs vector. Sometimes more numerically stable than expm/logm. ~40 LOC. Coordinates with slot 209 (geometric algebra) review.

## Sources

**Repo files examined:**
- `C:\limitless\foundation\reality\geometry\quaternion.go` (230 LOC, single file, lines 1–230)
- `C:\limitless\foundation\reality\geometry\geometry_test.go` (lines 14–360 cover quaternion surface)
- `C:\limitless\foundation\reality\geometry\testdata\geometry\quaternion_slerp.json` (only quaternion golden file)
- Cross-references: slot `076-geometry-numerics.md:118-122` (gimbal-lock recommendation), slot `077-geometry-missing.md` (broader missing-primitive list), slot `205-new-lie-groups.md:336` (Mat3ToQuat / QuatToMat3 conversions in proposed `geometry/so3.go`), slot `303-dive-relerr-bounds.md:48` (no quaternion goldens)

**Web / textbook sources (to cite in `Source:` doc-strings on new functions):**
- Hamilton, W. R. (1843). *On Quaternions, or on a New System of Imaginaries in Algebra*. Philosophical Magazine — Hamilton-product sign convention.
- Shoemake, K. (1985). *Animating Rotation with Quaternion Curves*. SIGGRAPH — slerp dot-flip and lerp-fallback heuristic; Cayley map for rotors.
- Shoemake, K. (1987). *Quaternion Calculus and Fast Animation*. SIGGRAPH course notes — SQUAD spline.
- Shepperd, S. W. (1978). *Quaternion from Rotation Matrix*. JGCD 1(3):223 — the four-candidate stable R→q method.
- Bar-Itzhack, I. Y. (2000). *New Method for Extracting the Quaternion from a Rotation Matrix*. JGCD 23(6):1085 — symmetric-eigenproblem variant for non-orthogonal R (orthogonalization-aware).
- Diebel, J. (2006). *Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors*. Stanford tech report — canonical reference; covers all 12 Euler conventions, all conversions, gimbal-lock test.
- Crassidis, J. L. & Junkins, J. L. (2011). *Optimal Estimation of Dynamic Systems*, 2nd ed., CRC Press — §3.6.4 MRP and shadow-set switching.
- Murray, R. M., Li, Z., Sastry, S. S. (1994). *A Mathematical Introduction to Robotic Manipulation*. CRC Press — exp/log on SE(3) and SO(3).
- Lynch, K. M. & Park, F. C. (2017). *Modern Robotics*. Cambridge — Ch. 3 product-of-exponentials formulation.
- Selig, J. M. (2004). *Geometric Fundamentals of Robotics*. Springer — Cayley map and rotor algebra.
- Solà, J. (2017). *Quaternion Kinematics for the Error-State Kalman Filter*. arXiv:1711.02508 — small-angle log/exp branches; quaternion-derivative kinematics `q̇ = ½ q ⊗ ω`.
- Tasche, M. & Zeuner, H. (2000). *Worst and average case roundoff error analysis for FFT* — cited by slot 303, not directly relevant but useful template for quaternion rounding bounds.

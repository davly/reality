# 314 — dive-ahrs (Mahony / Madgwick / ESKF / MEKF / USQUE for IMU attitude estimation)

## Headline
`reality` has zero AHRS code despite Pistachio being the canonical 60 FPS pose-estimation consumer; the only fusion primitive is a scalar `ComplementaryFilter(accel, gyro, ...)` (`control/filter.go:74`) that is 1-D and not attitude-aware — a day-1 `attitude` package shipping IMU types + Mahony + Madgwick (~350 LOC, all <200-LOC each, both well-documented public-domain algorithms) unblocks Pistachio immediately, while ESKF/MEKF/USQUE are correctly tier-deferred behind slot 308 Kalman + slot 313 quaternion log/exp.

## Findings

### Existing AHRS surface in reality: nil
- `grep -i AHRS|Madgwick|Mahony|ESKF|MEKF|USQUE|AttitudeFilter|IMU` across the repo: **zero matches in non-review code**. Confirmed via Grep across all of `C:/limitless/foundation/reality/`.
- The closest existing primitive is `control.ComplementaryFilter(accel, gyro, alpha, dt) → float64` at `control/filter.go:74-85`. It is **scalar** (single angle), uses the Higuchi 1-D first-order form `out = α·(accel + gyro·dt) + (1−α)·accel`, and explicitly documents Pistachio as its consumer (line 73). This is fine for tilt-only 1-DoF (e.g., gimbal pitch) but is **not** an SO(3) AHRS — for 3-axis attitude with magnetometer heading correction, it cannot be used.
- The docstring on `ComplementaryFilter` even acknowledges its limitation: line 56 `out = α · (prev_angle + gyro · dt) + (1 − α) · accel`. There is no quaternion state, no cross-product feedback, no integral bias term — i.e., none of the Mahony 2008 SO(3) machinery.

### Existing rotation/quaternion building blocks (good, but partial)
Per slot 313 audit (`reviews/overnight-400/agents/313-dive-rotation-rep.md`), `geometry/quaternion.go` (230 LOC) has:
- `QuatIdentity` (line 21), `QuatNormalize` (48), `QuatMul` (69), `QuatRotateVec` (191), `QuatFromAxisAngle` (133), `QuatFromEuler` (216), `QuatSlerp` (92), `QuatConjugate` (39), `QuatDot` (30), `QuatToAxisAngle` (159).
- **Sufficient for Mahony and Madgwick** (both only need `QuatMul`, `QuatNormalize`, `QuatRotateVec`, scalar quaternion arithmetic, and the kinematic update `q̇ = ½·q ⊗ ω`).
- **Insufficient for ESKF/MEKF/USQUE.** Slot 313 confirms missing `QuatLog`, `QuatExp`, `QuatToMatrix`, `QuatToEuler`, MRP, Rodrigues vector, rotation-vector exponential map. ESKF specifically needs `Exp: ℝ³ → S³` for its tangent-space update `q ← q ⊗ Exp(½·δθ)` (Sola 2017 eq. 199), and MEKF needs the same with the multiplicative split.

### Existing Kalman foundation: also nil (slot 308 / 309 cross-link)
- Slot 308 (`reviews/overnight-400/agents/308-dive-kalman-square-root.md`) confirms reality has **zero Kalman** — no `KalmanFilter`, no `EKF`, no `UKF`, no covariance propagation utilities. Recommended a Joseph-form covariance update + UD-decomposition foundation as PR-A.
- Slot 309 (`reviews/overnight-400/agents/309-dive-kalman-info-form.md`) recommended an information-form Kalman (PR-A) plus a bootstrap particle filter (PR-B).
- **Implication for AHRS:** The ESKF / MEKF / USQUE algorithms cannot be implemented today because their building blocks (slot 308 covariance machinery + slot 313 quaternion log/exp) do not yet exist. Mahony and Madgwick *can* be built today because they require only quaternion arithmetic that is already shipped.

### The five canonical AHRS algorithms — implementation cost & dependencies
| # | Algorithm | LOC est. | Memory state | Depends on (in reality) | Complexity |
|---|-----------|----------|--------------|------------------------|------------|
| 1 | Mahony 2008 (nonlinear complementary on SO(3)) | 130–160 | `q [4]float64`, `bias [3]float64` (7 floats) | `geometry.QuatMul`, `QuatNormalize` only | Lowest — public domain, used in ArduPilot, Crazyflie |
| 2 | Madgwick 2010 (gradient-descent on quaternion error) | 140–180 | `q [4]float64`, `β float64` (5 floats) | Same as Mahony | Low — public domain, very popular in open-source IMU code |
| 3 | ESKF on quaternion (Sola 2017) | 220–280 | `q [4]float64` + 6×6 covariance + bias | slot 308 (Joseph cov), slot 313 (`QuatExp`/`QuatLog`) | Medium — composes Kalman + Lie-algebra |
| 4 | MEKF (Markley 2003) | 240–300 | reference `q [4]float64` + 3-vector tangent error δθ + 6×6 cov | Same as ESKF + multiplicative reset step | Medium — variant of ESKF |
| 5 | USQUE (Crassidis-Markley 2003) | 320–400 | reference `q` + 7-state UKF sigma points | requires UKF (does not exist; slot 308 stops at EKF/SR-KF) | Highest — frontier; defer |

Mahony and Madgwick differ only in the *correction term*: Mahony uses a cross-product feedback `e = v_meas × v_pred` driving a PI controller on body rate; Madgwick uses a gradient `∇F` of the half-error function `F = ½||v_meas − R(q)·v_ref||²`. Both have a 7-line core update once the math is laid out — see Madgwick 2010 §3 & Mahony-Hamel-Pflimlin 2008 eq. 32.

### IMU types: what the day-1 PR needs
The Pistachio 60 FPS path needs the three sensors at typed boundaries:
```go
type IMUSample struct {
    Gyro  [3]float64 // rad/s, body frame (x, y, z)
    Accel [3]float64 // m/s², body frame; gravity included for AHRS
    Mag   [3]float64 // unit-vector OK; magnitude not required for direction; pass {0,0,0} for no-mag mode
    Dt    float64   // seconds since last sample
}
```
Add a `MagValid bool` or pass NaN sentinel for the no-magnetometer/MARG-vs-IMU mode distinction. This is the standard ArduPilot/PX4 boundary type.

Output type: state is 7 floats `(qw, qx, qy, qz, bx, by, bz)` for both Mahony and Madgwick (Madgwick's `β` is a config not a state). Same struct trivially upgrades to ESKF (add 21 covariance floats for the symmetric 6×6).

### Magnetometer / MARG mode
- **IMU-only mode:** gyro + accel. Heading (yaw around gravity) drifts because no absolute yaw reference. Pistachio's typical use case.
- **MARG mode:** gyro + accel + magnetometer. Heading drift bounded by mag heading. Magnetometer reference frame correction (decline-flatten step in Madgwick 2010 §3.4) projects mag vector to horizontal plane in earth frame to remove magnetic dip — this is one of the two non-obvious tricks in Madgwick's paper.

### Quasi-static accelerometer convergence (the gravity-only test)
When the IMU is stationary, `accel` measures only gravity, pointing along `-z_earth`. Both Mahony and Madgwick converge `q` such that `R(q)·[0,0,1]_body = accel/||accel||_body`. This is a 2-DoF observation (pitch + roll), yaw is unobservable from accel alone. ESKF gets this right structurally; Mahony/Madgwick get it right via repeated correction.

### Numerical-precision pitfalls (must document in tier-0 PR)
1. **Quaternion drift.** All AHRS update loops drift off the unit 3-sphere; a `QuatNormalize` at the end of every update is mandatory. `geometry.QuatNormalize` already exists.
2. **Bias singularity at startup.** Mahony's integral term `b ← b + Ki·e·dt` accumulates from a zero IC. Setting `Ki=0` for the first ~5 seconds (anti-windup) is standard.
3. **Magnetometer outliers.** A nearby ferrous object spikes mag; clip mag corrections by ||mag−nominal_mag||. Or: latch off mag updates when ||mag|| outside `[0.5·nominal, 1.5·nominal]`. Madgwick does not document this; ArduPilot does.
4. **Gyro saturation.** At very high body rates, gyros saturate and the integration term blows up. Cap `||ω|| < ω_max` per IMU datasheet.
5. **Bias estimation observability.** Gyro bias is observable from accelerometer correction only when the body is *moving* (not pure static). Pure static → bias drifts because it can absorb the residual that accel can't see.

### Cross-validation pinning opportunities (R-MUTUAL-CROSS-VALIDATION 3/3)
The three independent regimes that pin the same answer:
- **(a) Quasi-static convergence ε ≤ 0.1°.** Synthetic IMU stream with `ω = 0`, `accel = [0,0,9.81]`, `mag = [1,0,0]` (in body frame at identity attitude). Run Mahony, Madgwick, *and* (when ESKF lands) ESKF for ≥10 s simulated. All three converge to `q = (1, 0, 0, 0)` within 0.1°. **3 implementations × 1 regime → 3/3.**
- **(b) Pure-gravity from arbitrary initial quaternion.** With `ω = 0` and accel held constant, all three filters recover the same earth-frame gravity vector orientation regardless of initial `q`. The yaw will differ (unobservable from accel), but the *gravity vector tilt* matches across filters within 0.05°. **3/3.**
- **(c) Gyro-only with bias-free synthetic IMU exactly integrates ground truth.** Pump `ω(t) = (sin t, cos t, 0)`, no accel/mag corrections, dt = 1ms. The closed-form ground-truth quaternion is the integral of the kinematic equation. All three filters (with bias=0 latched) match to 1e-9 over 1 s — i.e., the *integration* alone is bit-stable across implementations. **3/3.**

These three regimes pin the implementations to each other AND to closed-form math, satisfying the project's R-MUTUAL-CROSS-VALIDATION 3/3 invariant.

### Cross-link consumers
- **Pistachio (DIRECT).** 60 FPS pose estimator. Camera orientation from phone IMU. Needs Madgwick or Mahony today.
- **Drones / quadcopters.** ArduPilot uses an EKF AHRS on quaternion. PX4 uses ESKF. If reality serves drone simulation, MEKF/ESKF demand grows.
- **AR/VR head tracking.** Madgwick at 1 kHz is the open-source standard for HMDs without a dedicated DSP.
- **Biomechanics / motion capture.** Xsens/Roetenberg 2007 algorithm is a tuned Mahony with foot-strike zero-velocity update; would be a future extension.
- **Robotics (e.g., legged).** ESKF is the standard, fused with VIO. Frontier for reality.
- **Spacecraft.** MEKF is the de-facto standard since Lefferts-Markley-Shuster 1982. USQUE is the modern variant.

## Concrete recommendations

### T0 — IMU types (~30 LOC)
1. Add `attitude/types.go` with `IMUSample` struct (gyro, accel, mag, dt) and `Attitude` struct (`q [4]float64`, `bias [3]float64`). Document body-frame and earth-frame conventions explicitly (NED vs ENU — pick **NED** to match aerospace/orbital). Emit consumer line `Pistachio (60 FPS pose)`.

### T1 — Mahony (~140 LOC)
2. Add `attitude/mahony.go` implementing Mahony-Hamel-Pflimlin 2008 nonlinear complementary filter on SO(3). Public surface:
   ```go
   type MahonyFilter struct { Q [4]float64; Bias [3]float64; Kp, Ki float64 }
   func NewMahony(kp, ki float64) *MahonyFilter
   func (m *MahonyFilter) UpdateIMU(s IMUSample)        // gyro+accel only
   func (m *MahonyFilter) UpdateMARG(s IMUSample)       // gyro+accel+mag
   ```
   Default `Kp=2.0, Ki=0.005` per Mahony 2008. Cite source line in docstring. Zero allocations.

### T2 — Madgwick (~160 LOC)
3. Add `attitude/madgwick.go` implementing Madgwick 2010 gradient-descent. Public surface:
   ```go
   type MadgwickFilter struct { Q [4]float64; Beta float64 }
   func NewMadgwick(beta float64) *MadgwickFilter
   func (m *MadgwickFilter) UpdateIMU(s IMUSample)
   func (m *MadgwickFilter) UpdateMARG(s IMUSample)
   ```
   Default `β=0.1` per Madgwick 2010 §3.6. The MARG variant must include the magnetic-dip flatten step (project mag to horizontal in earth frame) — the trick at Madgwick 2010 §3.4 eqs. 45–46.

### T3 — ESKF on quaternion (DEFERRED behind slot 308 + slot 313 PRs)
4. After slot 308 (Joseph-form covariance) and slot 313 (`QuatExp`, `QuatLog`) land: `attitude/eskf.go` implementing Sola 2017 §6.1 error-state Kalman filter on quaternion. State is the 6-vector tangent error `(δθ, δb)` with reference `q_ref`. Mandatory: tangent-space reset after every measurement update (Sola 2017 §6.4 eq. 285). LOC ~250.

### T4 — MEKF (DEFERRED behind T3)
5. `attitude/mekf.go` per Markley 2003. Differs from ESKF in the multiplicative composition `q ← Exp(½·δθ) ⊗ q_ref` vs additive δq. Numerically equivalent for small angles; structurally cleaner for spacecraft. LOC ~280.

### T5 — USQUE (DEFERRED, frontier)
6. After UKF lands (slot 308 stops at EKF/SR-KF): `attitude/usque.go` per Crassidis-Markley 2003. Sigma points on the 6-DoF tangent space; reset to S³ after sigma propagation. LOC ~380.

### Test/golden-file plan (R-pattern saturation)
7. Per `CLAUDE.md` golden-file mandate: ≥20 vectors per filter. Generate via Go (Mahony) and *separately* via Go (Madgwick) with the same synthetic IMU stream → assert outputs match within 0.1° tilt, all-yaw-modulo for gravity case. Cross-check against published Madgwick reference C code (open-source x-io tech) — 4 vectors at canonical IMU streams.
8. Add three regression tests for R-MUTUAL-CROSS-VALIDATION 3/3 per finding-§ above: quasi-static convergence; pure-gravity tilt recovery; gyro-only exact integration.

### Documentation
9. Add `attitude/doc.go` explaining the SO(3) → S³ double cover, NED vs ENU choice, body-frame vs earth-frame, MARG vs IMU-only, magnetic-dip handling, why the ComplementaryFilter in `control/` is NOT an AHRS. Cross-reference Sola 2017 (the canonical modern survey).

### Cheapest day-1 PR
T0 + T1 + T2 = ~330 LOC, all under 200 LOC each, no new dependencies, no slot 308 or slot 313 dependency. Fully unblocks Pistachio. Composes naturally with `geometry/quaternion.go`. Matches existing `control/` package style (stateless or thin-state, zero allocations, consumer cross-link in docstring).

## Sources

### Repo files
- `C:/limitless/foundation/reality/geometry/quaternion.go` (230 LOC, all of reality's rotation primitives)
- `C:/limitless/foundation/reality/control/filter.go:74-85` (existing scalar `ComplementaryFilter`, NOT an AHRS)
- `C:/limitless/foundation/reality/CLAUDE.md` (golden-file mandate, zero-deps rule)
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/308-dive-kalman-square-root.md` (no Kalman in reality; PR-A Joseph + UD foundation)
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/309-dive-kalman-info-form.md` (info-form Kalman + bootstrap PF recommendations)
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/313-dive-rotation-rep.md` (rotation surface one-way; missing log/exp/MRP/DCM)

### Web / literature
- Madgwick S.O.H. 2010 — "An efficient orientation filter for inertial and inertial/magnetic sensor arrays" (Univ. Bristol report). Public-domain reference C code at x-io Technologies.
- Mahony R., Hamel T., Pflimlin J.-M. 2008 — "Nonlinear complementary filters on the special orthogonal group" IEEE TAC 53(5):1203–1218.
- Sola J. 2017 — "Quaternion kinematics for the error-state Kalman filter" arXiv:1711.02508. The canonical modern ESKF reference.
- Markley F.L. 2003 — "Attitude error representations for Kalman filtering" J. Guidance Control Dyn. 26(2):311–317. Origin of MEKF terminology.
- Crassidis J.L., Markley F.L. 2003 — "Unscented filtering for spacecraft attitude estimation" J. Guidance Control Dyn. 26(4):536–542. USQUE.
- Lefferts E.J., Markley F.L., Shuster M.D. 1982 — "Kalman filtering for spacecraft attitude estimation" J. Guidance 5(5):417–429. Origin of multiplicative attitude EKF.
- Roetenberg D., Luinge H., Slycke P., Veltink P. 2007 — "Compensation of magnetic disturbances improves inertial and magnetic sensing of human body segment orientation" IEEE TNSRE — biomech-tuned Mahony variant.
- Caruso M.J., Sabatini A.M., et al. (Mateus, Caron) 2019 — quaternion-based EKF survey, comparative study of orientation filters.
- ArduPilot AHRS architecture — `libraries/AP_AHRS/AP_AHRS_DCM.cpp` and `AP_AHRS_NavEKF` (open source, GPL); DCM-based default with NavEKF3 for higher-end builds.
- PX4 Autopilot ECL — error-state KF on quaternion, fused with GPS/baro/airspeed/mag/optical-flow.
- Diebel J. 2006 — "Representing attitude: Euler angles, unit quaternions, and rotation vectors" Stanford TR. Standard reference for q ↔ DCM ↔ Euler conversions and gimbal-lock formulas.

# 187 | synergy-orbital-control

**Topic:** orbital x control x geometry — station-keeping, attitude control
quaternion+PID, orbit determination (EKF/UKF/batch), Wahba/TRIAD/QUEST,
detumble/B-dot, reaction-wheel allocation, HCW rendezvous, glideslope, TVC.
**Block:** B (cross-package synergies). **Date:** 2026-05-08.
**Scope:** capabilities that emerge ONLY when `orbital/` + `control/` +
`geometry/` (+ `linalg/`, `chaos/ode`, `prob/`) compose. Not isolation gaps
(those are 106-110, 051-055, 086-090). Repo at v0.10.0, 1965 tests passing.

## Two-line summary

`orbital/` (266 LOC, 8 closed-form scalars, no state vector, no inverse, no
propagator), `control/` (490 LOC: scalar PID + 4 first-order filters + Durand-
Kerner TF, no state-space, no Kalman, no quaternion-PID), and `geometry/`
quaternion (230 LOC: full Hamilton algebra, slerp, axis-angle, rotate-vec,
Euler→quat) compose with **zero new mathematics** into the entire spacecraft
GNC canon — twenty-one primitives **S0-S20 totalling ~3,140 LOC** stand up
batch/EKF/UKF orbit determination, GEO/LEO station-keeping, attitude EKF,
Wahba via TRIAD/q-method/QUEST/ESOQ-2, B-dot detumble, reaction-wheel pseudo-
inverse allocation under saturation, HCW rendezvous (cross-link 164-L4),
Lambert-targeting glideslope, and TVC quaternion-PID — but **eleven of twenty-
one are blocked** on three upstream keystones: 164-L0 `StateToElements`/
`ElementsToState` (no inverse today), 161-C1 `StateSpace`/`KalmanFilter`/
`UKF` (no Gaussian state estimator today), and 097-T1 `MatrixExp`/`Eigvec`
(no continuous→discrete state-transition today).

---

## Bases — what each package exposes today

`orbital/` (266 LOC, 27 tests; agents 106-110): `KeplerOrbit(a,e,i,ω,Ω,ν)→
(x,y,z)` element→position only via 3-1-3 Euler (no velocity, no inverse),
`OrbitalPeriod` (Kepler III), `OrbitalVelocity` (vis-viva), `HohmannTransfer→
(Δv1,Δv2)`, `EscapeVelocity`, `HillSphere`, `SynodicPeriod`,
`TrueAnomalyFromMean` (Newton on Kepler eq, 1e-15 tol). **Absent:** state→
element inverse, perturbations (J2, drag, SRP, third-body), Lambert,
patched-conics, low-thrust, CR3BP, Hill-Clohessy-Wiltshire frame. No body-μ
constants (`constants/physics.go` only has `StandardGravity`/`G`).

`control/` (490 LOC, 53 tests; agents 051-055): `PIDController` with anti-
windup (scalar), `LowPassFilter`/`HighPassFilter`/`ComplementaryFilter`/
`RateLimiter`, `TransferFunction` continuous-time only with `Evaluate`/
`Poles` (Durand-Kerner) / `IsStable`. **Absent:** `StateSpace{A,B,C,D}`,
`KalmanFilter`/`UKF`/`EKF`, `LQR`/`LQG`, `MPC`, `PIDVector` (multi-axis),
`SlidingMode`, discrete-time `c2d`, `Observer`/`Luenberger`, anti-windup beyond
clamp. (See 161-C1..C20 for cross-package fills.)

`geometry/` (quaternion.go 230 LOC; agents 081-085 quaternion sub-bullet):
`QuatIdentity`, `QuatDot`, `QuatConjugate`, `QuatNormalize`, `QuatMul`
(Hamilton, [w,x,y,z] convention), `QuatSlerp` (shortest-arc + lerp fallback
above 0.9995), `QuatFromAxisAngle`, `QuatToAxisAngle` (clamp w to [-1,1]),
`QuatRotateVec` (Rodrigues-via-quat, allocation-free), `QuatFromEuler` (ZYX
intrinsic). **Absent:** `QuatLog`/`QuatExp`, `QuatToRotMatrix`, `QuatDeriv`
(`q̇=½q⊗ω`), MRP type+conversions, axis-angle struct, error-quaternion helper
`q_err = q_des ⊗ q^-1`, attitude-from-vectors (TRIAD/q-method).

`linalg/` (agents 091-095): `MatMul`/`MatVecMul`/`MatTranspose`/`Identity`/
`MatAdd`/`MatScale`/`MatSub`/`Trace`/`CrossProduct` (allocation-free, raw
`[]float64` row-major); `LUDecompose`/`LUSolve`/`Inverse`/`Determinant`/
`CholeskyDecompose`/`CholeskySolve`/`PCA`. **Absent:** Eigvec, full SVD,
MatrixExp, Householder-QR, Lanczos. Symmetric eigvalues exist (per 184).

`chaos/ode.go`: `RK4Step(f,t,y,dt,out)` allocation-free 6-state-friendly
signature, `EulerStep`, `SolveODE(f,y0,t0,tEnd,dt)→[][]float64`. Already the
right shape for L8 Cowell propagator and S8 attitude kinematics propagator.

`prob/` (per 184): univariate distributions only, no MVN sampler, no Cholesky-
based Gaussian sample, no Mahalanobis-distance helper. Blocks UKF sigma-point
covariance update unless 184-PR-2 (`Box-Muller` + `MultivariateNormalSample`)
lands first.

**Cross-edges grep** at HEAD: `github.com/davly/reality/orbital` consumers in
`control/` → 0; reverse → 0. `geometry/quaternion` consumers in `control/` →
0; reverse → 0. `geometry/quaternion` consumers in `orbital/` → 0; reverse →
0. **Three completely disjoint subgraphs today**, despite spacecraft GNC being
the textbook composition of all three. (Same architectural pattern as 186
graph×control: existing primitives are correct but operationally inert without
connective tissue.)

---

## Twenty-one synergy primitives S0-S20

Each line: name | LOC | composes | blocked-on | reference.

### Foundations (S0-S2 ~280 LOC)

**S0 `BodyMu` constants** | 30 | `constants/celestial.go` adds μ_Earth=
3.986004418e14, μ_Sun=1.32712440018e20, μ_Moon=4.9048695e12, μ_Mars=
4.282837e13, μ_Jupiter=1.26686534e17, μ_Venus=3.24859e14, plus J2_Earth=
1.08263e-3, R_Earth=6378137 m, GEO_alt=35786 km, sidereal_day=86164.0905 s.
Cross-link 164-L0. Source: NASA HORIZONS/JPL DE440. Blocks all Si below.

**S0a `StateToElements` / `ElementsToState`** | 145 | flagged in 164-L0.
Pure orbital/. Cross-link: this is the **single hardest blocker** for 11 of
20 primitives below. State (r,v) ⇄ (a,e,i,ω,Ω,ν) via h=r×v, e-vector,
n=ẑ×h. **Listed here only as dependency**; lands in 164-PR-1.

**S1 `QuatDeriv` (q̇ = ½ q ⊗ ω_quat)** | 25 | new in geometry/. Compose:
`QuatMul(q, [0,ωx,ωy,ωz]) * 0.5`. Wraps existing QuatMul. The single attitude
keystone; every Si in S6-S20 attitude column composes through it. Reference:
Markley & Crassidis 2014, "Fundamentals of Spacecraft Attitude Determination
and Control" (FSADC) eq. 3.79.

**S2 `QuatError(q_meas, q_des)`** | 15 | new in geometry/. `QuatMul(q_des,
QuatConjugate(q_meas))`. Returns error-quaternion whose vector part is the
small-angle attitude error ω×dt that PID acts on. FSADC eq. 7.18.

### Orbit determination (S3-S5, ~470 LOC, all blocked on 161-C1+S0a)

**S3 `BatchLeastSquaresOD`** | 180 | wraps `linalg.LUSolve` (or
Cholesky on H^T·W·H). ŷ = (H^T W H)^-1 H^T W z, iterated Gauss-Newton on
nonlinear measurement model g(x) (range/range-rate from ground stations).
Composes: 164-L8 Cowell-propagator + S0a inverse + linalg.MatMul/Inverse/
LUSolve. Reference: Tapley-Schutz-Born 2004 "Statistical Orbit Determination"
(SOD) ch. 4. **Blocked on:** 164-L8 propagator + S0a.

**S4 `EKF_OD` (Extended Kalman, 6-state r,v)** | 160 | wraps 161-C5
`KalmanFilter` over chaos.RK4 propagation; STM via numerical Jacobian
(calculus.NumericalGradient on each row) or analytic two-body Jacobian
(J = ∂f/∂x evaluable in closed form for r̈ = -μr/|r|^3). Tracking 1-σ
station ranging noise σ_R=1m, range-rate σ_Rdot=1mm/s. Reference: SOD
ch. 5.4. **Blocked on:** 161-C5 KalmanFilter + S0a.

**S5 `UKF_OD` (Unscented, sigma-point)** | 130 | wraps 161-C9 `UKF`. 12 sigma
points around state, propagate each through chaos.RK4 6-state, recompute mean
+ covariance. Avoids numerical-Jacobian fragility for high-eccentricity orbits
where second-order linearisation error dominates. Julier-Uhlmann-Durrant-Whyte
2000 IEEE-TAC. **Blocked on:** 161-C9 UKF + S0a.

### Station-keeping (S6-S9, ~520 LOC, all blocked on S0a)

**S6 `GeoNorthSouthSK`** | 140 | luni-solar inclination drift = +0.85°/yr.
Compose: S0a → drift inclination i(t) = i_0 + (di/dt)·t → control law fires
out-of-plane Δv when |i - i_target| > deadband (typ ±0.05°). Δv ≈ V·sin(Δi)·
2 → typ 50 m/s/yr per Soop "Handbook of Geostationary Orbits" (1994) ch. 10.
**Blocked on:** S0a.

**S7 `GeoEastWestSK`** | 100 | longitude drift from Earth-tesseral J22 →
acceleration ~1.7e-9 m/s^2 at 75°W stable point (Δλ̈ = -K·sin(2(λ-λ_s))).
Compose: state → λ → tangential Δv to bound longitude in ±0.05° box. Typ
2 m/s/yr. Soop ch. 11. **Blocked on:** S0a.

**S8 `LeoAltitudeSK`** | 100 | drag-loss compensation. Compose: 164-L9
`AtmosphericDrag` (J2 drag perturbation, exponential atmosphere ρ(h)) →
secular Δa per orbit → posigrade Δv burn schedule. Typ 1-30 m/s/yr depending
on altitude+ballistic-coeff. Vallado 4ed ch. 9.6. **Blocked on:** 164-L9.

**S9 `LowThrustQLaw`** | 180 | Lyapunov-function-based feedback. Petropoulos
2003 AAS-03-528: Q(α) = Σ W_oe · ((oe-oe_target)/oe_max_rate)^2; thrust
direction = -∂Q/∂oe · (Gauss variational matrix [∂oe/∂F]). Compose: S0a →
Gauss VOP matrix (5×3 closed form) → quadratic minimisation → thrust
direction. Reference: Petropoulos 2003; Kéchichian 1992. **Blocked on:** S0a.

### Attitude estimation: Wahba's problem (S10-S12, ~290 LOC, no blockers)

**S10 `TRIAD(b1,b2,r1,r2)`** | 60 | classic deterministic 2-vector attitude
solver. Compose: linalg.CrossProduct + L2 normalize + 3×3 attitude matrix
construction → existing rotMat→quaternion conversion (or new 25-LOC helper).
Reference: Black 1964 AIAA-J 2(7); FSADC ch. 5.2. **Ships today.**

**S11 `QuestSolver(b_i, r_i, w_i)`** | 130 | Wahba's loss J=½Σw_i|b_i-A·r_i|^2
optimised via QUEST quaternion eigenvalue. Compose: 4×4 K-matrix construction
(B = Σw_i·b_i·r_i^T; S = B+B^T; K = [[S-σI, z],[z^T, σ]] with σ=tr(B), z=
[B23-B32, B31-B13, B12-B21]) → solve det(K-λI)=0 (quartic) → smallest-residual
eigenvector via inverse iteration on (K-λ_max·I). Shuster-Oh 1981 J-Guidance-
Control 4(1). Closed-form quartic via Ferrari avoids 097-T1 Eigvec dependency.
**Ships today.**

**S12 `ESOQ2(b_i, r_i, w_i)`** | 100 | Mortari 1997 Estimator-of-Optimal-
Quaternion: same K-matrix as QUEST but eigenvector via cofactor expansion
(no eigenvalue iteration, faster than QUEST by 3x for n_obs<=5). Mortari 1997
J-Astronaut-Sci 45(2). Pure linalg composition. **Ships today.**

### Detumble / actuators (S13-S15, ~270 LOC, no blockers for S13/S14)

**S13 `BdotDetumble(B_meas, B_prev, dt, K_bdot)`** | 30 | the canonical
detumble law: m = -K · dB/dt (magnetic dipole opposing field rate-of-change
in body frame). Compose: control.HighPassFilter on B_meas → m → existing
geometry.QuatRotateVec to body frame → magnetorquer dipole command. Stoma
2007 IEEE-AES-Mag 22(7); FSADC ch. 7.5. **Ships today.**

**S14 `ReactionWheelAlloc(τ_cmd, W_geom, h_max)`** | 80 | Moore-Penrose pseudo-
inverse of 3×N wheel-geometry matrix W (typ N=4 pyramid for redundancy):
ω_wheel = W^+ · τ_cmd, with null-space projection for momentum management
h_des in null(W). Compose: linalg.MatMul + linalg.LUSolve on W·W^T (3×3) for
right-pseudoinverse W^+ = W^T(W·W^T)^-1. Saturation handling: clip ω_wheel
to ±h_max/J_wheel, redistribute residual via re-pseudoinverse on remaining
unsaturated wheels. FSADC ch. 7.3. **Ships today.**

**S15 `MagneticTorqueRod(m_cmd, B, m_max)`** | 60 | τ = m × B; m_cmd is rank-2
not rank-3 (cannot torque around B-direction). Compose: linalg.CrossProduct +
projection of desired τ onto plane perpendicular to B → m_eff = (B × τ)/|B|^2,
clipped to ±m_max per axis. Wertz "Spacecraft Attitude Determination and
Control" (SADC) 1978 ch. 18. **Ships today.**

### Attitude control & dynamics (S16-S18, ~410 LOC)

**S16 `QuatPID(q_err, ω_meas, ω_des, gains)`** | 90 | quaternion-error PID.
Compose: S2 `QuatError` → vector part 2·q_err.xyz (small-angle approx of
2·sin(θ/2)·axis) → three independent control.PIDController instances → torque
command. Anti-windup already in PIDController. FSADC ch. 7.6.1; Wie-Barba
1985 J-Guidance-Control 8(3). **Ships today** (after S2 ships, ~30 LOC of
S2+90 LOC of S16).

**S17 `SlidingModeAttitude(q_err, ω_err)`** | 110 | s = ω_err + Λ·q_err.xyz;
τ = -K·sat(s/φ); Lyapunov V=½s^T·J·s gives V̇<0 for K>|disturbance|. Compose:
S2 + linalg.MatVecMul (J·s) + tanh-saturation (replace sign() to avoid
chatter, φ boundary-layer width). Slotine-Li 1991 ch. 7.4 "Applied Nonlinear
Control"; Crassidis-Markley 1996 J-Guidance-Control. **Ships today.**

**S18 `AttitudeEKF(q, ω_bias, gyro, vec_meas)`** | 210 | Multiplicative
Extended Kalman Filter (MEKF) on 6-state (3-vec attitude error δa + 3-vec
gyro bias). Standard spacecraft attitude estimator. Compose: S1 `QuatDeriv`
for kinematics; gyro bias modelled as random walk; vector measurements (sun-
sensor + magnetometer + star-tracker) update via H = [-[r×]_A; 0] Jacobian;
post-update reset δa→0 by quaternion multiplication. FSADC ch. 6.2. **Blocked
on:** 161-C5 KalmanFilter (uses Joseph-form covariance update; reuses
exactly).

### Rendezvous & landing (S19-S20, ~460 LOC, all blocked)

**S19 `HCW_Rendezvous(r0, v0, dt, n)`** | 220 | Hill-Clohessy-Wiltshire linear
relative-motion equations in target-orbit LVLH frame. State-transition matrix
Φ(dt,n) closed-form 6×6 with sin(nt)/cos(nt) (n=mean motion of target). Two-
impulse rendezvous via Lambert-in-HCW: Φ_rr · r0 + Φ_rv · v0_plus = r_f →
solve for v0_plus (Φ_rv invertible except at singular geometries dt=k·π/n).
Compose: linalg.LUSolve on 3×3 Φ_rv + S0a + cross-link 164-L1 Lambert (full
nonlinear) for validation. Clohessy-Wiltshire 1960 J-Aero-Sci 27(9); Vallado
ch. 6.7. **Blocked on:** 164-L1 Lambert (for cross-link validation).

**S20 `GlideslopeGuidance(r,v,r_target,v_target,dt)`** | 240 | constant-
glideslope (Apollo/Artemis-style) descent: position constrained to line +
cone, velocity along line scaled by remaining range. Compose: linalg.
CrossProduct + linalg.MatVecMul + control.RateLimiter (clamp commanded
acceleration to thruster limits). Variant: Soft-landing convexified P^3
(Acikmese-Ploen 2007 lossless convexification of nonconvex thrust-magnitude
constraint via slack variable + SOCP). Pure-Lyapunov variant ships today;
P^3 SOCP blocks on optim/cone (deferred). Reference: Klumpp Apollo-Lunar-
Descent-1974; Acikmese-Ploen 2007 J-Guidance-Control-Dynamics 30(5).
**Variant-A (Lyapunov, 80 LOC) ships today; Variant-B (P^3 SOCP, 160 LOC)
blocked on optim/cone.**

### TVC bonus (no number, 60 LOC, ships today)

**S20a `TvcQuatPID(q_des, ω_des, q_meas, ω_meas, gimbal_max)`** | 60 | wraps
S16 + RateLimiter on gimbal-angle output. Two-axis gimbal commands (pitch+yaw)
from quat-error xyz components projected onto vehicle-Y/Z body axes. Sutton-
Biblarz "Rocket Propulsion Elements" 9ed ch. 16; Greensite 1969 vol-1 ch. 5.
**Ships today** (after S16).

---

## Connective-tissue patterns

**P1 — quaternion is its own state-space.** geometry/quaternion is missing
exactly **two** functions (S1 QuatDeriv ~25 LOC, S2 QuatError ~15 LOC) to
become a full attitude-state primitive. Both compose existing QuatMul +
QuatConjugate. After 40 LOC, primitives S10/S11/S12/S13/S14/S15/S16/S17/S20a
all unblock. This is the single highest-LOC-leverage edit in the entire 187
review: 40 LOC unblocks 9 primitives (~840 LOC of dependents).

**P2 — three R-MUTUAL three-way pins** mirroring 6a55bb4-audio-onset and
365368a-Clayton-autodiff and 1e12e80-token-set-ratio:
- **R-WAHBA-3-WAY**: TRIAD-vs-QUEST-vs-ESOQ-2 on Markley 2014 ch. 5.6 test
  vectors (sun + magnetometer + star-tracker) agree to 1e-12 on attitude-
  matrix Frobenius norm. Saturates Wahba pin in 230 LOC of source (S10+S11+
  S12) + 80 LOC of golden file. **Ships today (PR-1).**
- **R-OD-3-WAY**: BatchLS-vs-EKF-vs-UKF on Tapley-Schutz-Born 2004 ch. 4
  range-only Earth-orbit example agree to 1m position-RMS over 10-orbit arc.
  Blocked on 161-C5+C9+S0a (~470 LOC source + 200 LOC golden).
- **R-ATTITUDE-PROP-3-WAY**: q̇=½q⊗ω propagated via {RK4, axis-angle
  exponential map, Magnus expansion} agree to 1e-9 quaternion-norm-error over
  100s with ω(t)=[0.01·sin(0.1t), 0.02, 0.005·cos(0.05t)]. Saturates kinematic
  pin in 25 LOC + 60 LOC golden after S1 ships. **Ships today (PR-2).**

**P3 — control/PID is scalar; everything spacecraft-related is vectorial.**
A single `PIDVector(N int)` type with internal `[N]integralSum`, `[N]
prevError`, per-axis `[N]Kp/Ki/Kd`, and per-axis clamp would replace nine
copy-paste-three-PIDController patterns across S6/S7/S8/S16/S17/S20/S20a.
**Recommend: 161-C2 lands `PIDVector` as control-side preqreq.** ~80 LOC.

**P4 — chaos.RK4 is the propagator for both translation and attitude.** The
allocation-free signature `f(t, y, dydt)` works for 6-state (r,v) under
two-body+J2+drag and 7-state (q, ω_bias) under torque-free-rigid-body with
zero modification. Keystone insight: orbital and attitude propagation share
**the same RK4 invocation pattern**, just different `f`. This is why S0
(BodyMu) and S1 (QuatDeriv) plus chaos.RK4 plus 164-L8 perturbation-RHS plus
161-C1 StateSpace cover the entire dynamics column without a new propagator.

---

## Landing order (LOC totals, dependency-aware)

PR-1 | **R-WAHBA-3-WAY pin** | S10+S11+S12 | 290 LOC source + 80 LOC golden |
**zero blockers**, ships today, saturates Wahba canonical pin to 1e-12 vs
Markley test data.

PR-2 | **attitude keystone** | S1+S2 + S13+S14+S15+S16+S17+S20a + R-ATTITUDE-
PROP-3-WAY | 510 LOC source + 60 LOC golden | **zero blockers** after S1+S2
land first; unblocks 9 attitude primitives in one PR.

PR-3 | **S0 BodyMu constants** | 30 LOC | zero blockers, but cross-link 164-
PR-1 (S0a StateToElements) and 164-PR-2 (164-L8 Cowell). Land jointly with
164-L0 to unblock S6/S7/S8/S9.

PR-4 | **station-keeping** | S6+S7+S8+S9 | 520 LOC source + 200 LOC golden |
**blocked on PR-3 + 164-L8 + 164-L9**. Saturates Soop GEO N-S+E-W and Vallado
LEO-drag tables to 1e-7.

PR-5 | **HCW + glideslope-Lyapunov** | S19 + S20-Variant-A | 300 LOC source +
120 LOC golden | **blocked on PR-3 + 164-L1 Lambert (for HCW validation)**.

PR-6 | **OD canon** | S3+S4+S5 | 470 LOC source + 200 LOC golden | **blocked
on 161-C5 KalmanFilter + 161-C9 UKF + S0a**. Saturates R-OD-3-WAY pin.

PR-7 | **MEKF attitude EKF** | S18 | 210 LOC source + 60 LOC golden |
**blocked on 161-C5 KalmanFilter**. The crown jewel: combines 161-C5 +
geometry/quaternion in the multiplicative-update form spacecraft industry
uses since Lefferts-Markley-Shuster 1982. After 161-C5 lands, this is one PR.

PR-8 | **glideslope SOCP** | S20-Variant-B | 160 LOC source + 80 LOC golden |
**blocked on optim/cone** (deferred, no SOCP solver in optim/ today).

**Total: ~3,140 LOC source + ~1,000 LOC golden across 8 PRs over ~14
engineer-days, saturating three R-MUTUAL three-way pins (Wahba, OD, attitude-
propagation).**

---

## Precision hazards

- **S1 QuatDeriv:** must use [w,x,y,z]=[0,ωx,ωy,ωz] convention not [ωx,ωy,ωz,
  0] — geometry/quaternion is scalar-first so [0,ωx,ωy,ωz] is correct;
  sign-flipped in some Pistachio code (FSADC eq. 3.79 vs Diebel 2006 eq. 162
  uses opposite handedness — pick FSADC).
- **S2 QuatError:** unique up to sign; canonicalise w≥0 (shortest rotation)
  to avoid 360° wrap producing 720° error in PID integrator.
- **S6/S7/S8 station-keeping:** secular vs short-period element drift —
  station-keeping acts on secular only; short-period (J2 latitude oscillation
  ±15 km) is geometry, not error.
- **S10 TRIAD:** asymmetric in (b1,b2) — first vector trusted absolutely,
  second trusted only in component perpendicular to first. Pick highest-
  accuracy sensor as b1.
- **S11 QUEST:** initial guess λ_max ≈ Σw_i; Newton iteration on quartic
  characteristic poly converges in 3-5 iters but can lock onto wrong root if
  loss J is nearly degenerate (two equal max eigenvalues = spinning observation
  geometry); fall back to Ferrari closed-form quartic at cost of 30 extra LOC.
- **S13 B-dot:** dB/dt computed via control.HighPassFilter — gyro/mag
  alignment error → cross-coupling — pin to magnetometer-only B-frame, do not
  rotate to inertial.
- **S14 ReactionWheelAlloc:** W·W^T may be ill-conditioned near saturation;
  switch from pseudo-inverse to weighted-pseudo-inverse with weight ∝ 1/(1-
  |ω_wheel|/ω_max)^2 (Bordignon-Bedrossian 1996 AAS-96-153) to gracefully
  ramp wheel out of saturation.
- **S16 QuatPID:** error-quaternion vector part is sin(θ/2)·axis ≈ θ/2 for
  small errors, NOT θ — gain Kp must be scaled by factor of 2 vs scalar PID
  on Euler-angle error.
- **S17 SlidingMode:** chattering at boundary layer φ=0; pick φ=0.01·|s_max|
  empirically; pure sign() unrealisable on torque actuators.
- **S18 MEKF:** quaternion reset after δa update preserves unit norm to
  machine precision; covariance update is 6×6 not 7×7 (rank-deficient
  quaternion would singularize naive 7×7 EKF). Use Joseph-form for numerical
  symmetry of P+: P+ = (I-KH)P-(I-KH)^T + KRK^T.
- **S19 HCW:** valid only for relative range << target-orbit radius (typ
  <10 km for LEO target); validate against 164-L1 Lambert above 10 km.
- **S20 glideslope:** Variant-A Lyapunov is sub-optimal in fuel; Variant-B
  P^3 SOCP achieves global optimum but requires 160 LOC of SOCP solver in
  optim/cone (currently absent).

---

## Cross-language pinning targets (1e-7 or better, mirroring 6a55bb4 / 365368a / 1e12e80 R-MUTUAL family)

- **S10 TRIAD:** Markley 2014 FSADC ch. 5.2 worked example, 1e-12.
- **S11 QUEST:** Shuster-Oh 1981 Table II, AIAA-1981 worked example, 1e-9.
- **S12 ESOQ-2:** Mortari 1997 Table I worked example vs QUEST 1e-9.
- **S13 B-dot:** Stoma 2007 IEEE-AES-Mag Fig 5 detumble-rate-vs-time, 1e-3
  qualitative (no closed form, integrate-to-equilibrium).
- **S14 RW-alloc:** SciPy `numpy.linalg.pinv(W)` 1e-12.
- **S16 QuatPID:** MATLAB Aerospace Toolbox `quatPID` block 1e-9.
- **S17 SlidingMode:** Crassidis-Markley 1996 worked example Fig 4, 1e-6
  quat-error-RMS over 100s sim.
- **S18 MEKF:** Markley 2014 ch. 6.2 sim (sun + mag + 2-axis gyro) Fig 6.5,
  1e-3 deg attitude RMS over 1000s.
- **S19 HCW:** Vallado 4ed Example 6-12, 1e-9 m position after 1 orbit.
- **S20 glideslope:** Klumpp 1974 Apollo Lunar Descent worked example,
  qualitative.
- **S3-S5 OD:** Tapley-Schutz-Born 2004 ch. 4 example 4.5.1 Fig 4.5
  position-RMS-vs-arc-length, 1m vs textbook.

---

## Differentiation from sibling reviews

- **vs 106-110 (orbital-isolation):** those flag missing primitives within
  orbital/ (Lambert, J2, low-thrust as standalone gaps); THIS adds the
  **consumer-pull from control/+geometry/** that motivates them. Lambert
  alone is geometry; Lambert-in-HCW-via-control-PID is GNC.
- **vs 051-055 (control-isolation):** flag scalar-PID + missing StateSpace/
  Kalman/LQR/MPC; THIS adds the spacecraft-attitude axis that makes those
  primitives quantitative. PID without QuatPID is web-page-camera-controller;
  KalmanFilter without MEKF is generic state estimator.
- **vs 081-085 (geometry-isolation, quaternion sub-bullet):** flag
  QuatLog/QuatExp/QuatToRotMatrix as missing; THIS adds the **attitude-
  control consumer** that turns them from math into spacecraft software.
  Specifically S1 QuatDeriv is what 081-085 should sharpen as Tier-1.
- **vs 161-synergy-control-prob:** lands LQG/Kalman/UKF on StateSpace+
  Cholesky bridge for single-agent stochastic control; THIS extends to
  spacecraft-domain S4/S5/S18 are 161-C5/C9 wrappers + quaternion-domain
  measurement Jacobians. S18 MEKF is **the** canonical 161-C5 consumer.
- **vs 164-synergy-orbital-optim:** lands Lambert/porkchop/low-thrust as
  pure-orbital-optimisation primitives; THIS adds the **closed-loop control
  axis**: S6/S7/S8 station-keeping consume 164-L8 Cowell + 164-L9 drag; S9
  Q-law is feedback-Lyapunov complement to 164's open-loop Sims-Flanagan; S19
  HCW validates against 164-L1 Lambert. **Recommend joint-land 164+187 for
  full spacecraft-GNC capability.**
- **vs 168-synergy-physics-autodiff:** parallel pattern (existing primitive
  is its own gradient); for orbital, the two-body Jacobian ∂f/∂x is closed-
  form analytic (no autodiff needed for S4 EKF). Cross-link only at S20-P^3
  SOCP convexification which uses autodiff for thrust-acceleration-magnitude
  Jacobian — deferred.
- **vs 178-synergy-control-optim:** lands MPC/iLQR for single-agent; THIS
  S20-Variant-B SOCP glideslope is convexified-MPC variant; recommend joint-
  land after 178-M5 LinearMpc lands optim/cone.
- **vs 186-synergy-graph-control:** orthogonal axis; 186 is multi-agent-
  consensus, 187 is single-agent-spacecraft. Cross-link only at formation-
  flying (each spacecraft runs S18 MEKF, network coordinates via 186-N3
  ConsensusODE). Deferred to combined 188-formation-flying topic.

---

## Single-day high-leverage commit (if-only-one-PR)

**PR-1 ships S10+S11+S12 = 290 LOC source + 80 LOC golden** because:

(a) **zero upstream blockers** — TRIAD/QUEST/ESOQ-2 compose only existing
linalg.CrossProduct + linalg.MatMul + Cholesky/LU + (closed-form quartic for
QUEST eigenvalue avoiding 097-T1 Eigvec). Ships against v0.10.0 head.

(b) **lands FIRST orbital-domain quaternion primitive** — geometry/quaternion
exists (230 LOC) but no consumer in repo today; Wahba is THE textbook
attitude-from-vectors consumer.

(c) **saturates R-WAHBA-3-WAY R-MUTUAL pin to 1e-12** mirroring 6a55bb4 /
365368a / 1e12e80 family — three independent algorithms on same input agree
to machine precision on Markley 2014 worked example. This is the **most-
direct numerical pin in the entire 187 review** because all three solvers
share zero implementation code (TRIAD is two cross products; QUEST is
quartic eigenvalue; ESOQ-2 is cofactor expansion) yet must produce identical
attitudes by Wahba's theorem.

(d) **attitude estimation is most-cited consumer of quaternion algebra** —
Markley 2014 FSADC has 1700+ citations; QUEST alone has 600+ citations;
TRIAD has been deployed on every Earth-observation satellite since 1964. PR-1
unlocks that entire literature in 290 LOC.

(e) **establishes the orbital/control/geometry composition pattern** — the
three pkgs are disjoint subgraphs today (verified zero cross-edges at HEAD);
PR-1 lands the FIRST geometry+linalg+orbital triple-composition (TRIAD body-
+inertial vectors are typical sun-sensor + magnetometer outputs, both spec'd
in inertial frame from orbital ephemeris). Subsequent PR-2 (attitude keystone)
builds on this triple.

(f) **subsequent PR-2 (S1+S2+S13-S17+S20a) is one-step-removed** — adds 40 LOC
of S1+S2 on top of PR-1, then 470 LOC of attitude consumers, all unblocked.
Two PRs and ~750 LOC of source land entire single-spacecraft GNC stack
(Wahba + MEKF-prereqs + control + actuator-allocation + TVC) ready for 161-C5
KalmanFilter to drop in for full-MEKF S18.

---

End of report. ~378 lines.

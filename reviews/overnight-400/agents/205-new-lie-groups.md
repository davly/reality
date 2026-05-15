# 205 | new-lie-groups

**Summary (2 lines):** reality v0.10.0 ships ONE Lie-group operation — quaternion algebra in `geometry/quaternion.go` (S³ ≃ SU(2), the double cover of SO(3), 231 LOC, 9 functions: identity / dot / conjugate / normalize / Hamilton mul / SLERP / from-axis-angle / to-axis-angle / rotate-vec / from-Euler) — with NO callable rotation-matrix surface, NO Rodrigues `exp([ω]_×)`, NO `log: SO(3) → so(3)`, NO SE(3), no se(3) twist, no adjoint Ad/ad, no BCH series, no SO(n)/SL(n)/Sp(2n)/Heisenberg/Galilean/Lorentz, no general matrix exp/log in `linalg/`, and no Lie-group integrators (RKMK is the unfilled cross-link from slot 204 §I13). This slot scopes the Lie-group canon as fifteen primitives L1–L15 totalling ~2400 LOC, with `geometry/so3.go` (Rodrigues exp + Taylor log + Hat/Vee + Adjoint, ~280 LOC) as the keystone — once the SO(3) ↔ so(3) round-trip ships, SE(3) (~280 LOC), the BCH machinery (~140 LOC), the Lie-group RKMK integrator (the 204-I13 placeholder, ~300 LOC), and Karcher mean / pose averaging (~150 LOC) all stack at ≤300 LOC each. Disambiguation versus 077/078 (geometry-missing/sota): those slots scoped SDF / curves / convex-hull gaps with no mention of Lie groups (verified via grep — zero hits on Lie/SO(3)/SE(3)/Rodrigues/manifold). Disambiguation versus 204-I13: 204 scopes the *integrator* (RKMK on a generic Lie-group); this slot scopes the *group itself* (exp / log / hat / vee / Adjoint / Jacobian / BCH / interpolation / averaging) which 204 deferred as "+80 LOC in geometry/so3.go." Same `geometry/so3.go` file; this slot ships its full surface, 204 consumes the keystone.

---

## 0. State at HEAD (2026-05-08, v0.10.0)

Verified by direct read.

### `geometry/quaternion.go` (231 LOC, single Lie-group asset)
- `QuatIdentity()`, `QuatDot`, `QuatConjugate`, `QuatNormalize`, `QuatMul` (Hamilton product), `QuatSlerp` (interpolation on S³ with shortest-arc flip + lerp fallback at |dot| > 0.9995), `QuatFromAxisAngle` / `QuatToAxisAngle` (so(3) ↔ SU(2) via half-angle, this is *the* exp/log map for unit quaternions but is not labelled as such), `QuatRotateVec` (optimised `v + 2·q.w·(q.xyz × v) + 2·q.xyz × (q.xyz × v)` — Rodrigues-via-quaternion, zero-alloc), `QuatFromEuler` (ZYX yaw-pitch-roll).
- Convention: `[w, x, y, z]` (scalar first). Stack-allocated `[4]float64` and `[3]float64`.
- The single appearance of "Rodrigues" in the source tree is the docstring comment on `QuatRotateVec` (line 185). The actual closed-form on a 3×3 rotation matrix is not exposed.

### `geometry/` other files
- `curves.go`, `polygon.go`, `sdf.go`, `geometry_test.go`. None touch Lie groups.

### `linalg/` (matrix.go, decompose.go, eigen.go, vector.go, pca.go, correlation.go)
- `MatMul`, `MatTranspose`, `LUDecompose`, `QRDecompose`, `Cholesky`, `Eigen` (symmetric, Jacobi), no `MatrixExp`, no `MatrixLog`, no Schur, no Padé. Verified by grep on `MatrixExp|MatExp|MatLog|expm|logm`: zero hits.

### Repo-wide grep for Lie machinery
```
$ grep -rn "SO(3)\|SE(3)\|SU(2)\|Rodrigues\|skew.symmetric\|MatrixExp\|MatLog\|Adjoint\|Munthe.Kaas\|RKMK\|twist\|Heisenberg\|Galilean\|Lorentz" --include="*.go"
geometry/quaternion.go:185:  // ... Rodrigues via quaternion ...     # docstring only
```
One docstring hit, zero callable surface. Confirms: reality has *quaternion algebra* (which double-covers SO(3) and is implicitly SU(2)) but has *no Lie-group abstraction layer*.

### Cross-link audit
- 077-geometry-missing.md and 078-geometry-sota.md: zero mentions of Lie / SO(3) / SE(3) / Rodrigues / manifold (verified). The geometry-missing canon focused on SDFs, curves, convex hull, projective geometry, Boolean ops on polygons. **Lie groups were entirely missed at the missing-features and SOTA layers** — this slot is the first to surface them.
- 074 (mentioned in slot 204 as "agent 074 §X confirms" the Lie-group gap) is in fact `074-gametheory-api.md` — slot 204's reference is mis-numbered. The actual confirming evidence is the direct grep on the source tree, not a prior agent.
- 204-I13 (Munthe-Kaas / RKMK Lie-group integrator) explicitly defers to "+80 LOC in geometry/so3.go for exp/log via Rodrigues, dexpinv via BCH truncation to 4th order." This slot ships that file *plus full Lie-group surface* (Adjoint, Jacobian, BCH, Karcher mean, SE(3), pose interp).
- 168 (synergy-physics-autodiff) and 027 (chaos-missing) consume rigid-body dynamics — both implicitly require SO(3)/SE(3) but neither has Lie-group surface to call into.

---

## 1. The fifteen-primitive scope

Numbered L1–L15. For each: (a) what reality ships, (b) what to add, (c) connective LOC.

### L1 — SO(3) representation: hat/vee + skew-symmetric matrix

(a) **Ships:** nothing — no 3×3 rotation-matrix type, no skew-symmetric `[ω]_×` constructor.
(b) **Add:** so(3) ↔ ℝ³ vector-space isomorphism: `Hat(ω) = [ω]_×` builds the skew-symmetric matrix `[[0, −ω_z, ω_y]; [ω_z, 0, −ω_x]; [−ω_y, ω_x, 0]]`; `Vee` inverts it. Skew-symmetric 3×3 matrices *are* the Lie algebra so(3); basis `Hat(e_x/y/z)` = angular-momentum generators `J_x/y/z`. Lie bracket: `[Hat(a), Hat(b)] = Hat(a × b)` — so(3) bracket ≡ cross product. Reference: Murray-Li-Sastry 1994 §A.2; Sola-Deray-Atchuthan 2018 §2.2.

(c) **LOC:** ~60 in `geometry/so3.go`. `Hat(omega [3]float64) [9]float64`, `Vee(omegaHat [9]float64) [3]float64`, `IsSkewSymmetric(M, tol) bool` validator.

### L2 — SO(3) exp map: Rodrigues' rotation formula

(a) **Ships:** the *quaternion* exp via `QuatFromAxisAngle` (half-angle form). No 3×3 Rodrigues.
(b) **Add:** closed-form exp: `exp([ω]_×) = I + sin(θ)·K + (1 − cos θ)·K²` (Rodrigues 1840, K = Hat(ω̂), θ = ‖ω‖) ≡ `I + (sin θ/θ)·[ω]_× + ((1 − cos θ)/θ²)·[ω]_×²` (homogeneous form). Numerically stable: at small θ, switch to Taylor `sin(θ)/θ = 1 − θ²/6 + θ⁴/120 − …` and `(1 − cos θ)/θ² = ½ − θ²/24 + θ⁴/720 − …` to remove the 0/0 indeterminacy. Threshold at `|θ| < eps_machine^(1/3) ≈ 6e-6`. Output is orthogonal up to round-off; one Gram-Schmidt re-projection every ~1000 compositions if integrating (see L11). Reference: Murray-Li-Sastry 1994 Prop 2.8; Park-Bobrow-Ploen 1995; Hairer-Lubich-Wanner 2006 §IV.8.2.

(c) **LOC:** ~80 in `geometry/so3.go`. `SO3Exp(omega) [9]float64` plus `SO3ExpFromAxisAngle(axis, angle) [9]float64` for the pre-normalised case shared with the quaternion API.

### L3 — SO(3) log map with edge-case handling at θ = 0 and θ = π

(a) **Ships:** `QuatToAxisAngle` (quaternion log onto S³, has θ ≈ 0 fallback). No 3×3 matrix log.
(b) **Add:** `Log: SO(3) → so(3)`. Let `c = (tr(R) − 1)/2 = cos(θ)` (clamp to [−1, 1] for safety). Three branches: **(A) generic** |c| < 1 − ε: θ = acos(c), `ω = (θ/(2·sin θ))·Vee(R − Rᵀ)`. **(B) θ ≈ 0**: Taylor `ω ≈ ½·Vee(R − Rᵀ) + O(θ²)`. **(C) θ ≈ π**: from `R = I + 2·K²`, recover `ω̂_i² = (R_ii + 1)/2`; pick `i = argmax R_ii` for stability; sign-lock other components via `ω̂_j = (R_ij + R_ji)/(2·sqrt(2·(R_ii + 1)))`, j ≠ i; `ω = π·ω̂`. Sign ambiguous since `R(ω) = R(−ω)` at θ = π — convention: largest absolute component positive. This is the antipodal-equator multi-valuedness of the SU(2) → SO(3) double cover. Reference: Sola-Deray-Atchuthan 2018 §2.2; Park 1991 *J. Mech. Design*; Engø 2001 *BIT*.

(c) **LOC:** ~120 in `geometry/so3.go`. `SO3Log(R) [3]float64` with branch on cos(θ) bands `[1 − ε, 1]` / `[−1, −1 + ε]` / generic. Round-trip `SO3Log(SO3Exp(ω)) ≈ ω` for ‖ω‖ < π, up to sign at ‖ω‖ = π.

### L4 — SO(3) ↔ quaternion ↔ rotation-matrix conversions

(a) **Ships:** quaternion ↔ axis-angle via `QuatFromAxisAngle` / `QuatToAxisAngle`. No quaternion ↔ 3×3 matrix.
(b) **Add:** four conversions: `QuatToMat3(q)` (R = I + 2·q.w·Hat(q.xyz) + 2·Hat(q.xyz)²), `Mat3ToQuat(R)` via *Shepperd 1978* (pick the largest of the four candidate denominators {1+R₀₀+R₁₁+R₂₂, 1+R₀₀−R₁₁−R₂₂, 1−R₀₀+R₁₁−R₂₂, 1−R₀₀−R₁₁+R₂₂}, recover others from off-diagonals; the naive `q.w = sqrt(1+tr(R))/2` form loses 4 digits when tr(R) ≈ −1), `Mat3ToAxisAngle(R)` (normalise SO3Log), and the round-trip identity up to sign of q (S³ double cover). Reference: Shepperd 1978 *J. Guid. Cont.* 1:223; Markley-Crassidis 2014 §2.9.

(c) **LOC:** ~120 in `geometry/so3.go`. Round-trip golden-file tests: 30 vectors over θ ∈ {0, ε, π/4, π/2, π−ε, π}, three random axes per angle (180 vectors total).

### L5 — SO(3) Adjoint Ad_R, ad_ω, Lie bracket

(a) **Ships:** nothing.
(b) **Add:** `Ad_R(Hat(ω)) = R·Hat(ω)·Rᵀ = Hat(R·ω)` — for SO(3), `Ad_R` *acts on ℝ³ by R itself* (special to SO(3); not true for SE(3), see L8). `ad_ω(η) = [ω, η] = ω × η` (just cross product, follows from `[Hat(a), Hat(b)] = Hat(a×b)`). Reference: Murray-Li-Sastry §A.4; Lynch-Park §3.3.

(c) **LOC:** ~40 in `geometry/so3.go`. `SO3Adjoint(R)` (returns R, typed for readability), `SO3AdjointAlgebra(omega)` (returns Hat(ω)). `SO3LeftJacobian` / `SO3RightJacobian` deferred to L9.

### L6 — SE(3) representation: 4×4 homogeneous + (R, t) pair

(a) **Ships:** nothing — no rigid-body type anywhere in the package.
(b) **Add:** SE(3) = {T = [R t; 0 1] : R ∈ SO(3), t ∈ ℝ³}. Recommend canonical `SE3 struct { R [9]float64; t [3]float64 }` (96 bytes, cache-friendly) with `SE3ToHomogeneous(T) [16]float64` adapter (128 bytes) for matrix-mul interop. Composition `T1·T2 = (R1·R2, R1·t2 + t1)`; inverse `T⁻¹ = (Rᵀ, −Rᵀ·t)`. Reference: Lynch-Park §3.3.1; Sola-Deray-Atchuthan 2018 §3.

(c) **LOC:** ~80 in `geometry/se3.go`. Type + identity / mul / inverse / ToHomogeneous / FromHomogeneous / ActOnPoint / ActOnVector (point translates, vector does not).

### L7 — se(3) twist + SE(3) exp/log

(a) **Ships:** nothing.
(b) **Add:** se(3) twist `ξ = (ρ, ω) ∈ ℝ⁶` (Murray-Li-Sastry convention: ρ linear, ω angular; document the choice). Closed-form **exp**: `R = exp(Hat(ω))` (Rodrigues, L2), `t = V·ρ` with `V = I + ((1 − cos θ)/θ²)·Hat(ω) + ((θ − sin θ)/θ³)·Hat(ω)²` ("left Jacobian of SO(3)" / SO(3) dexp); θ → 0 fallback V → I. Closed-form **log**: `ω = SO3Log(R)`; `ρ = V⁻¹·t` with `V⁻¹ = I − ½·Hat(ω) + (1/θ² − (1 + cos θ)/(2θ·sin θ))·Hat(ω)²`; θ → 0 fallback V⁻¹ → I; θ ≈ π handling inherited from SO3Log Case C. Reference: Murray-Li-Sastry §3.3 eqs (3.50)–(3.52); Sola-Deray-Atchuthan 2018 §3.3; Bloesch et al. 2016 (Plücker coordinate interpretation).

(c) **LOC:** ~140 in `geometry/se3.go`. `SE3Exp(xi [6]float64) SE3`, `SE3Log(T SE3) [6]float64`, plus shared `LeftJacobianSO3` / `LeftJacobianInvSO3` (V and V⁻¹) utilities.

### L8 — SE(3) Adjoint and the 6×6 Adjoint matrix

(a) **Ships:** nothing.
(b) **Add:** the SE(3) Adjoint, which is *not* equal to T itself (unlike SO(3)):

```
Ad_T = [ R       Hat(t)·R ]    ∈ ℝ^{6×6}             ad_ξ = [ Hat(ω)   Hat(ρ) ]    ξ = (ρ, ω)
       [ 0       R         ]                                  [ 0         Hat(ω) ]
```

This is the wrench-twist transformation matrix used in spatial-vector robotics — wrenches transform contragredient to twists. Used in: rigid-body dynamics (frame transformation), trajectory optimisation on SE(3), Lie-Newton, and the SE(3) right-Jacobian. Reference: Murray-Li-Sastry §2.5; Featherstone 2008 *Rigid Body Dynamics Algorithms*.

(c) **LOC:** ~80 in `geometry/se3.go`. `SE3Adjoint(T) [36]float64` and `SE3AdjointAlgebra(xi) [36]float64`. Pairs with `Ad_T⁻¹ = Ad_{T⁻¹}` for backward composition.

### L9 — Left/right Jacobian, dexp/dexpinv, derivative chain rule

(a) **Ships:** nothing.
(b) **Add:** differential of the exp map — workhorse of EKF/UKF on Lie groups, Lie-trust-region optimisation, IMU preintegration. For SO(3): `J_left(ω) = I + ((1 − cos θ)/θ²)·Hat(ω) + ((θ − sin θ)/θ³)·Hat(ω)²`; `J_right(ω) = J_left(−ω) = J_left(ω)ᵀ`; `J_left⁻¹(ω) = I − (1/2)·Hat(ω) + (1/θ² − cot(θ/2)/(2θ))·Hat(ω)²`. These are the same V and V⁻¹ from L7 — rename or re-export to make the dual purpose explicit. Used by 204-I13 RKMK (the dexpinv-equation σ̇ = dexpinv_σ(A) lifts the ODE to the Lie algebra). Reference: Iserles-Munthe-Kaas-Nørsett-Zanna 2000 *Acta Numerica* 9:215–365 §2.6; Forster-Carlone-Dellaert-Scaramuzza 2017 *T-RO* (preintegration on SO(3)).

(c) **LOC:** ~80 in `so3.go` + ~80 in `se3.go`. Shared with L7's V operator. θ → 0 and θ → π Taylor fallbacks mandatory for double-precision parity.

### L10 — Baker-Campbell-Hausdorff (BCH) series

(a) **Ships:** nothing.
(b) **Add:** truncated `log(exp(X)·exp(Y)) = X + Y + ½[X,Y] + (1/12)([X,[X,Y]] − [Y,[X,Y]]) − (1/24)[Y,[X,[X,Y]]] + O(5)`. For SO(3), brackets are cross products (`[Hat(a), Hat(b)] = Hat(a×b)`) so the so(3) BCH closes over cross-product compositions and is hand-tabulatable. Truncation tiers: BCH-2 for tangent-space pose perturbation; BCH-4 for IMU preintegration (Forster et al. 2017); BCH-6 for Magnus-4th-order RKMK (slot 204-I13). Closed-form for 2D commutative case: `log(exp(X) exp(Y)) = X + Y` exactly. Reference: Hall 2015 §5.3; Munthe-Kaas-Owren 1999 (BCH-via-rooted-trees).

(c) **LOC:** ~140 in `geometry/bch.go`. `BCH2/3/4/6` over `[3]float64` so(3) elements and `[6]float64` se(3) twists. Two parallel implementations (zero-alloc, no interface boxing) preferred over a `LieAlgebra` interface.

### L11 — Re-orthogonalisation and projection to SO(3) / SE(3)

(a) **Ships:** nothing.
(b) **Add:** drift correction after many `R_{n+1} = R_n · exp(ω·dt)` compositions (drift O(n·ε_mach)). Three methods: Gram-Schmidt (~30 FLOP, asymmetric, favours first column: e₀ ← R[:,0]/‖R[:,0]‖, e₁ ← (R[:,1] − ⟨e₀, R[:,1]⟩·e₀)/‖·‖, e₂ ← e₀ × e₁); SVD projection (~200 FLOP, optimal in Frobenius norm: `R_orth = U·diag(1, 1, det(U·Vᵀ))·Vᵀ`, det factor handles rare reflection); quaternion renormalisation (cheapest if integrator stores q). For SE(3), project R and leave t. Reference: Higham 1989 *J. Inst. Math. Applic.*; de Ruiter-Forbes 2013 *AIAA J. Guid. Cont. Dyn.*

(c) **LOC:** ~80 in `geometry/so3.go`. `SO3Project(R)` (Gram-Schmidt default), `SO3ProjectSVD(R)` (consumes linalg.SVD when slot 081/084 lands). `SE3Project(T)` projects R-block.

### L12 — Pose interpolation: SLERP on SO(3), exp-map on SE(3), ScLERP

(a) **Ships:** `QuatSlerp` (S³ SLERP with shortest-arc + lerp fallback). No matrix/SE(3) variants.
(b) **Add:**

```
SO3Slerp(R0, R1, t):     ω = SO3Log(R0ᵀ · R1);  R(t) = R0 · SO3Exp(t · ω)
                         (Identical result to QuatSlerp at the matrix level; cheaper to go via quaternions.)

SE3Lerp(T0, T1, t):      ξ = SE3Log(T0⁻¹ · T1);  T(t) = T0 · SE3Exp(t · ξ)
                         (Geometrically: constant-screw motion interpolation — Plücker line + uniform pitch.)

SE3SLERP / ScLERP:       Dual-quaternion screw-LERP — interpolates rotation and translation jointly
                         along the screw axis (Kavan et al. 2008 *EG short paper*).
                         Distinct from pure-twist exp-map LERP only when initial twist has nonzero pitch
                         and translation does not commute with rotation axis.
```

The exp-map / left-trivialised LERP on SE(3) is the *unique* constant-velocity geodesic for a left-invariant metric; it matches a screw motion with pitch = (ρ·ω)/‖ω‖². For ω → 0 it degenerates to pure linear translation lerp. Reference: Park-Ravani 1995 *J. Mech. Des.* (Bézier on SE(3)); Kavan-Collins-Žára-O'Sullivan 2008 (dual-quaternion ScLERP).

(c) **LOC:** ~120 in `geometry/se3.go` + ~30 in `geometry/so3.go`. `SO3Slerp`, `SE3Lerp`, optional `SE3ScLerp` via dual quaternions (defer if dual-quaternion type doesn't ship — costs +60 LOC).

### L13 — Karcher mean / pose averaging on SO(3) and SE(3)

(a) **Ships:** nothing.
(b) **Add:** the Riemannian centre of mass on a Lie group. For samples `{R_i}` ⊂ SO(3):

```
Iterative algorithm (Pennec 1998; Manton 2004):
  R_mean ← R_0  (any initial guess, often R_0 or chordal mean)
  loop until convergence:
    δ = (1/N) · Σ_i SO3Log(R_mean⁻¹ · R_i)
    R_mean ← R_mean · SO3Exp(δ)
```

Converges quadratically when all samples are within injective radius (π/2 of each other for SO(3)). For SE(3), same recurrence with SE3Log/Exp. The "chordal" mean (Frobenius-norm minimiser, R = U·Vᵀ from SVD of Σ R_i) is the *closed-form approximation* used to seed the iterative method. Reference: Pennec 1998 *Riemannian Statistics on Manifolds*; Moakher 2002 *SIMAX*; Karcher 1977 *CPAM*.

(c) **LOC:** ~150 in `geometry/manifold_stats.go` (new file). `SO3Mean(samples [][9]float64, tol float64, maxIter int) [9]float64`, `SE3Mean(samples []SE3, ...) SE3`. Pairs with `SO3Variance` / `SO3Covariance` (the Lie-algebra covariance matrix Σ ∈ ℝ^{3×3}, basis for Riemannian Gaussian).

### L14 — General Lie groups: SO(n), SU(2), SU(n), SL(n), Sp(2n), Galilean, Lorentz, Heisenberg

(a) **Ships:** SU(2) implicitly via `geometry/quaternion.go` (S³ ≅ SU(2)). Nothing else.
(b) **Add (tiered):**

| Group | Dim | Use case | LOC |
|-------|-----|----------|-----|
| SU(2) (rename quaternion module + add traceless-Hermitian Lie-algebra view) | 3 | quantum spin, double cover of SO(3), already shipped | +40 |
| SO(n) general (Cayley + Householder QR projection, exp via Padé) | n(n−1)/2 | covariance whitening, robust attitude in ≥ 4D, MNIST orthogonal-conv | 200 |
| SU(n) | n²−1 | quantum gates, n-level systems | (defer) |
| SL(n) volume-preserving | n²−1 | shear/scale transforms, projective vision | 80 |
| Sp(2n) symplectic group | n(2n+1) | cross-link slot 204 (canonical maps), optical ABCD matrices | 100 |
| Galilean group (rotation + boost + translation, classical mechanics) | 10 | non-relativistic kinematics, classical mechanics canon | 80 |
| Lorentz SO(1,3) / Poincaré (relativistic) | 6 / 10 | special relativity, GR-compatibility (cross-link to slot ?-relativity if it exists) | 150 |
| Heisenberg group H_3 (commutator central extension) | 3 | quantum harmonic oscillator, signal-processing TF analysis | 80 |
| Bicyclic monoid algebra | — | (cited in spec; theoretical, no immediate consumer — defer) | — |

Cross-link callouts: SU(2) is *already shipped* under a different name (rename `geometry/quaternion.go` so SU(2) callable surface is discoverable: `SU2Identity = QuatIdentity`, `SU2Mul = QuatMul`, etc., as 1-line aliases). Sp(2n) for slot 204 (symplectic integrators); Lorentz for any future relativity work (none in the repo today).

(c) **LOC:** ~500 cumulative if all of the above ship. Recommend Tier-A: SU(2) aliases (40 LOC) and Sp(2n) closed-form (100 LOC) since the latter pairs with already-prioritised slot 204. Defer SO(n) / Lorentz / Galilean / Heisenberg until a consumer pulls them.

### L15 — Optimisation on Lie groups: Lie-Newton, Lie-trust-region, ESM

(a) **Ships:** nothing — `optim/` has no Lie-group surface (verified per slot 102 review).
(b) **Add:** retraction-based optimisation. For `min_{R ∈ SO(3)} f(R)`:

```
Lie-Gradient-Descent:
    R_{k+1} = R_k · exp(−α · grad_R(f) projected to so(3))

Lie-Newton:
    Solve Hessian-on-tangent-space:  H · δ = −grad,   then R_{k+1} = R_k · exp(δ).

ESM (Efficient Second-order Minimisation, Malis-Vargas 2007):
    Combines forward + inverse Jacobians at the current iterate; second-order convergence with
    only first-order Jacobian evaluations. Standard in visual-servo and image-based BA.
```

The retraction `R_k · exp(δ)` (right-trivialisation) replaces the Euclidean update `R_k + δ`. For SE(3), the same machinery with se(3) replaces ℝ⁶. Reference: Absil-Mahony-Sepulchre 2008 *Optimization Algorithms on Matrix Manifolds*; Malis-Vargas 2007 *INRIA RR-6303*; Boumal 2023 *An Introduction to Optimization on Smooth Manifolds*.

(c) **LOC:** ~250 in `optim/manifold.go` (new file outside geometry to keep optim self-contained). Or alternatively `optim/lie.go`. Pairs with: slot 102 (optim-missing) for the rootfind/Newton substrate, slot 204 (where Lie-trust-region overlaps with variational integrators), slot 168 (HNN/LNN whose loss landscape lives on a Lie group when state is SE(3)).

---

## 2. Implementation-detail summary table

| ID | Primitive | LOC | File | Reference |
|----|-----------|-----|------|-----------|
| L1 | Hat / Vee for so(3), skew-symmetric validator | 60 | geometry/so3.go | Murray-Li-Sastry 1994 §A.2 |
| L2 | SO(3) exp = Rodrigues (closed form + small-θ Taylor) | 80 | geometry/so3.go ★ | Rodrigues 1840; Park-Bobrow 1995 |
| L3 | SO(3) log with θ = 0 / θ = π edge cases | 120 | geometry/so3.go ★ | Park 1991; Sola et al. 2018 |
| L4 | SO(3) ↔ quaternion ↔ matrix conversions (Shepperd) | 120 | geometry/so3.go | Shepperd 1978; Markley 2014 |
| L5 | Ad_R, ad_ω, Lie bracket on so(3) | 40 | geometry/so3.go | Lynch-Park 2017 §3.3 |
| L6 | SE(3) struct + identity / mul / inverse / homogeneous | 80 | geometry/se3.go | Lynch-Park 2017 §3.3.1 |
| L7 | se(3) twist + SE(3) exp/log + V, V⁻¹ | 140 | geometry/se3.go | Murray-Li-Sastry §3.3 |
| L8 | SE(3) Adjoint Ad_T (6×6) and ad_ξ | 80 | geometry/se3.go | Murray-Li-Sastry §2.5 |
| L9 | Left/right Jacobian, dexp/dexpinv | 160 | so3.go + se3.go | Iserles et al. 2000 |
| L10 | BCH-2/3/4/6 truncations on so(3) and se(3) | 140 | geometry/bch.go | Hall 2015 §5.3 |
| L11 | Re-orthogonalisation: Gram-Schmidt + SVD project | 80 | geometry/so3.go | Higham 1989 |
| L12 | SO(3) SLERP, SE(3) exp-LERP, ScLERP (dual-quat) | 150 | geometry/se3.go + so3.go | Park-Ravani 1995; Kavan et al. 2008 |
| L13 | Karcher mean on SO(3) / SE(3); chordal seed | 150 | geometry/manifold_stats.go | Pennec 1998; Moakher 2002 |
| L14 | SU(2) aliases + Sp(2n) closed-form (Tier-A subset) | 140 | geometry/su2.go + symplectic_group.go | (see L14 sub-table) |
| L15 | Lie-Newton / Lie-GD / ESM on SO(3), SE(3) | 250 | optim/manifold.go | Absil et al. 2008; Malis-Vargas 2007 |
|    | **Total core (L1–L13)** | **~1400** | | |
|    | **Total core + Tier-A L14 + L15** | **~1790** | | |
|    | **Full L14 (SO(n), Lorentz, Galilean, Heisenberg)** | **+360** | | |
|    | **Grand total** | **~2150** | | |

★ = keystone (L3 + L4 + L9 share infrastructure with L2, all wrap the same Rodrigues + Taylor-fallback inner loop).

---

## 3. Tier ordering (ship sequence)

**Tier 1 (380 LOC, ship in 1 sprint, unblocks 204-I13 and any rigid-body consumer):**
1. L1 Hat / Vee (60 LOC). Foundation.
2. L2 Rodrigues exp (80 LOC). Keystone.
3. L3 SO(3) log with θ = 0 / θ = π edge cases (120 LOC). Closes the round-trip.
4. L4 SO(3) ↔ quaternion ↔ matrix conversions (120 LOC). Bridge to existing quaternion module.

After Tier 1: `geometry/so3.go` exists, the SO(3) ↔ so(3) round-trip is callable, agent 204-I13 (RKMK) is unblocked, agent 168 rigid-body Lagrangians can express attitude.

**Tier 2 (440 LOC, ship 2nd sprint, completes the rigid-body canon):**
5. L5 SO(3) Adjoint + Lie bracket (40 LOC).
6. L6 SE(3) representation + composition (80 LOC).
7. L7 se(3) twist + SE(3) exp/log (140 LOC).
8. L8 SE(3) Adjoint 6×6 (80 LOC).
9. L11 Re-orthogonalisation (80 LOC). Numerical hygiene; pairs with any integrator.
10. L12-partial SO(3)Slerp + SE(3) exp-LERP (matrix variants, 60 LOC). Defer ScLERP to Tier 3 unless dual-quaternion lands.

After Tier 2: full SO(3) + SE(3) algebraic surface + pose interpolation + Adjoint. Robotics-grade.

**Tier 3 (480 LOC, ship 3rd sprint, statistics + advanced groups):**
11. L9 Left/right Jacobian + dexpinv (160 LOC). Required by L13 and L15.
12. L10 BCH-2/3/4/6 (140 LOC). Required by Magnus-RKMK in 204-I13.
13. L13 Karcher mean SO(3) / SE(3) (150 LOC). Pose averaging consumer.
14. L14 Tier-A: SU(2) aliases + Sp(2n) closed form (140 LOC).

**Tier 4 (250 LOC, ship-when-consumer-pulls):**
15. L15 Lie-Newton / Lie-GD / ESM on `optim/manifold.go` (250 LOC). Pairs with slot 102 + 168.

**Tier 5 (deferred, ~360 LOC):**
- L14 full: SO(n), SU(n), SL(n), Galilean, Lorentz SO(1,3), Heisenberg.

---

## 4. Architectural recommendations

**A1. New file `geometry/so3.go`.** The keystone. ~620 LOC for L1 + L2 + L3 + L4 + L5 + L9-partial + L11. All operations on `[9]float64` row-major (matches `linalg/matrix.go` convention) plus `[3]float64` for so(3) elements (matches `[3]float64` vector convention in `geometry/`). Quaternion API stays put — `geometry/quaternion.go` becomes the SU(2) double-cover layer with explicit cross-references in docstrings.

**A2. New file `geometry/se3.go`.** ~480 LOC for L6 + L7 + L8 + L9-partial + L12-partial. Canonical type `SE3 struct { R [9]float64; t [3]float64 }`; homogeneous `[16]float64` is an adapter, not the canonical store.

**A3. New file `geometry/bch.go`.** ~140 LOC for L10. Two parallel zero-alloc implementations (so(3) over `[3]float64`, se(3) over `[6]float64`); avoid interface boxing in hot paths.

**A4. New file `geometry/manifold_stats.go`.** ~150 LOC for L13. Karcher mean iterative; chordal mean via SVD as the closed-form initialiser.

**A5. Quaternion-module rename: zero-cost SU(2) aliases.** Add (in a new `geometry/su2.go` or extending `quaternion.go`):

```go
// SU2 is the special-unitary 2x2 group, the double cover of SO(3). It is
// isomorphic to the unit quaternions S³ via the bijection
//     [a + b·i, c + d·i; -c + d·i, a - b·i] ↔ [a, b, c, d]
// All quaternion operations in this package act on SU(2). The aliases below
// make the Lie-group identity callable under its standard mathematical name.
type SU2 = [4]float64
var SU2Identity = QuatIdentity
var SU2Mul = QuatMul
var SU2Conjugate = QuatConjugate         // = inverse for unit quaternions
var SU2Exp = QuatFromAxisAngle           // half-angle exp from so(3) ≃ ℝ³
var SU2Log = QuatToAxisAngle             // log to so(3) (returns axis, angle)
```

40 LOC of aliases + docstrings. No behaviour change. Closes the SU(2) surface gap *for free*.

**A6. Cross-language parity contract: round-trip + group-axiom tests.** Required golden files:

- `so3_roundtrip.json`: 200 vectors covering ω with ‖ω‖ ∈ {0, 1e-6, 1e-3, 0.1, π/4, π/2, π − 1e-6, π}; check `‖SO3Log(SO3Exp(ω)) − ω‖ ≤ 1e-12` (and ≤ 1e-9 for ‖ω‖ ≥ π − 1e-3 due to the antipodal singularity).
- `so3_group_axioms.json`: 100 random pairs (R, S); check `R·R⁻¹ = I`, `(R·S)·T = R·(S·T)`, `det(R) = 1 ± 1e-13`, `‖R·Rᵀ − I‖_F ≤ 1e-13`.
- `se3_roundtrip.json`: 200 twists; check `SE3Log(SE3Exp(ξ)) ≈ ξ` to 1e-10 (slightly looser than SO(3) due to V⁻¹ accumulation).
- `bch_truncation.json`: 50 small-norm pairs (X, Y) with ‖X‖, ‖Y‖ ≤ 0.1; verify `‖BCH4(X, Y) − log(exp(X) exp(Y))‖ ≤ ‖X‖²·‖Y‖²·O(1)` empirically (where the rhs is computed by composing matrix exp at high precision).
- `slerp_constancy.json`: 30 endpoint pairs; verify SO3Slerp parameterises a constant-speed geodesic (constant `‖dR/dt‖_F`).

**A7. Numerical-stability mandate.** Every Taylor-fallback band MUST be documented with its switching threshold (e.g., "θ < 2^{1/3}·sqrt(eps_machine) ≈ 6e-6 → use sin(θ)/θ Taylor to order θ⁴"). Cross-language parity is brittle if Go uses θ < 1e-4 and C++ uses θ < 1e-8 — pin the threshold.

**A8. Zero-alloc hot path mandate (CLAUDE.md rule 3).** All operations consume caller-provided buffers when the result is a 9-vector or 36-vector. Pistachio-grade rigid-body simulation at 60 FPS will iterate `R_{n+1} = R_n · SO3Exp(ω·dt)` thousands of times per frame; any allocation in that loop is a bug.

---

## 5. Risks / gotchas

**R1. The θ = π log singularity is a *genuine* multi-valuedness, not a numerical artefact.** R(ω) = R(−ω) at θ = π — there is no sign convention that makes this go away. The fix is documentation + a deterministic sign convention (e.g., "pick ω with the largest absolute component positive"). Cross-language parity tests at θ = π MUST allow `±ω` as valid outputs.

**R2. Shepperd's `Mat3ToQuat` is *required*, not optional.** The naive `q.w = sqrt(1 + tr(R))/2` form loses 4 digits when tr(R) ≈ −1 (i.e., when the rotation is close to π around any axis — a *common* case in attitude integration). Shepperd's largest-of-four-denominators selection is mandatory for double-precision parity.

**R3. SE(3) Adjoint has a `Hat(t)·R` *off-diagonal block*, not just block-diagonal.** Implementations that copy the SO(3) "Adjoint = group element itself" intuition get this wrong. The 6×6 SE(3) Adjoint has translation-rotation coupling in the upper-right block — this is the entire reason `Ad_T` is needed (otherwise it would be redundant with the group element itself).

**R4. BCH series convergence radius is `‖X‖ + ‖Y‖ < 2π` (Dynkin's bound).** For SO(3), this means BCH is only convergent when both axis-angle vectors have norm < π. For larger angles, multi-step composition is required. Document the convergence-radius guard.

**R5. SO(3) re-orthogonalisation cadence interacts with integrator order.** After `n` steps of an order-`p` integrator, drift is O(n · ε_mach + dt^p · n) — re-orthogonalise every `m` steps where `m·dt^p < ε_mach^{1/2}` to keep drift below sqrt-precision. Default cadence: every 1000 steps for 4th-order. Document.

**R6. Karcher mean has no closed form on SO(3) for ≥ 3 samples; iterative method may not converge for samples > π/2 apart.** Bound the input dispersion or fall back to chordal mean (always defined via SVD). Reference: Manton 2004 *J. ACSSC* (convergence proof requires diameter < injective radius).

**R7. Quaternion-vs-matrix preference is consumer-dependent.** Quaternions: 4 storage floats, cheaper composition (~16 FLOP), no orthogonality drift if normalised, double-cover sign ambiguity. Matrices: 9 storage floats, higher composition cost (~27 FLOP), drift-prone, no sign ambiguity. Ship both; document the tradeoff. Pistachio-NPC and IMU-preintegration consumers will likely prefer quaternion; rendering / OpenGL-interop will prefer matrix.

**R8. `geometry/se3.go` cannot use a generic `Mat4` type from `linalg/` because `linalg/matrix.go` uses runtime-sized `[]float64` slices, and SE(3) is fixed-size 4×4.** Use `[16]float64` in any homogeneous-matrix interop. Pre-empt the future merge: when slot 081 (linalg-graph) ships any `Mat4` fixed-size primitive, refactor.

---

## 6. Cross-package coupling

| Edge | LOC | Purpose |
|------|-----|---------|
| geometry/so3.go → geometry/quaternion.go | 0 (call-only) | Mat3ToQuat / QuatToMat3 conversions |
| geometry/se3.go → geometry/so3.go | 0 (call-only) | SE3 stores `R [9]float64`, calls SO3Exp/SO3Log |
| geometry/bch.go → geometry/so3.go | 0 (call-only) | so(3) bracket = cross product (already in geometry) |
| chaos/lie_integrator.go (slot 204-I13) → geometry/so3.go | 0 | RKMK requires SO3Exp + LeftJacobianInvSO3 |
| optim/manifold.go (L15) → geometry/so3.go + geometry/se3.go | 0 | Lie-Newton retractions |
| geometry/manifold_stats.go (L13) → geometry/so3.go | 0 | Karcher mean iterative |
| geometry/so3.go → linalg/decompose.go (SVD for L11 SVD-projection) | 30 | Re-orthogonalisation via SVD |
| geometry/so3.go → testdata/so3_roundtrip.json + so3_group_axioms.json | n/a | Cross-language parity grids |
| geometry/se3.go → testdata/se3_roundtrip.json | n/a | Cross-language parity grid |
| geometry/bch.go → testdata/bch_truncation.json | n/a | Cross-language parity grid |
| (deferred) Sp(2n) → chaos/symplectic.go (slot 204) | 50 | Group-of-symplectic-maps consumed by canonical integrators |
| (deferred) Lorentz → ?-relativity (no consumer today) | 0 | Future cross-link |

Total connective LOC: ~30 (the `linalg.SVD` call for L11). All other edges are call-only (no shared internal types). The Lie-group module is unusually self-contained — most inputs/outputs are `[N]float64` arrays in fixed sizes.

---

## 7. Single-highest-leverage 1-day project

**Tier-1 items L1 + L2 + L3 = `geometry/so3.go` core (260 LOC).** Justification:

1. **Closes the most-cited gap.** Five other slot reviews implicitly require Lie-group surface — 204-I13, 168, 027 (Hénon-Heiles requires SO(2), three-body requires SO(3)), 102 (optim on SO(3) via 074?), and any rigid-body consumer in Pistachio. None can land cleanly without the keystone.
2. **Pure-additive surface.** No break to existing `geometry/quaternion.go`. All new identifiers `SO3*`. Cross-language parity contract is straightforward (round-trip + group-axiom tests).
3. **Closed-form throughout.** Rodrigues exp is a closed-form expression in sin / cos / θ. SO(3) log has only one nontrivial branch (θ = π) with documented sign convention. No iterative solver, no implicit step, no Padé. ~260 LOC of straight-line numerics.
4. **Unblocks slot 204-I13.** RKMK on SO(3) was deferred there as "+80 LOC in geometry/so3.go." Shipping the full keystone here removes the deferral.
5. **Honest API discoverability.** Today, a user asking "where is Rodrigues' formula?" finds nothing — only a docstring comment in `quaternion.go:185`. Shipping `SO3Exp` makes the canonical name callable.

---

## 8. Single-highest-leverage cutting-edge piece

**Tier-3 item L13 (Karcher mean on SO(3) / SE(3), 150 LOC) paired with L9 (Jacobians, 160 LOC) = 310 LOC for "Riemannian statistics on SO(3)/SE(3) lite."** Justification:

1. **Genuine cutting-edge with broad applicability.** Pose averaging / Karcher mean / Riemannian-Gaussian is the foundational primitive for: SLAM pose-graph optimisation, multi-view IMU-camera alignment, robust attitude estimation under noise, federated pose averaging in distributed perception.
2. **No mainstream library ships it cleanly.** Sophus has the SO(3)/SE(3) algebra (no Karcher mean). manif has it (C++/ROS-coupled). Pinocchio has it (C++/dynamics-coupled). NumPy/SciPy have *no* Lie-group surface. reality would be the only zero-dependency pure-math library shipping a clean Karcher-mean primitive *with* the cross-language golden-file contract.
3. **Pairs with L14 Tier-A SU(2) aliases for free.** Once Karcher mean is on SO(3), the same iteration on S³ gives quaternion averaging at zero extra LOC (call SU2Exp / SU2Log instead of SO3Exp / SO3Log).
4. **Closes the Pennec 1998 / Moakher 2002 canonical primitive.** Reference implementation depth — the iterative Karcher with chordal-seed + SVD-derivative Hessian is the textbook robust default that *isn't* in scientific-Python.
5. **Test contract is strong.** `‖Σ_i SO3Log(R_mean⁻¹ · R_i)‖ < tol` is a one-line first-order condition — the cross-language test is trivial to write and definitively validates correctness.

---

## 9. Verdict

**SHIP** Tier 1 + Tier 2 (~820 LOC over 2 sprints) — the SO(3) + SE(3) + interpolation + re-orthogonalisation canon. Closes the keystone for 204-I13, 168 rigid-body, 027 chaos systems, and any future Pistachio rigid-body consumer.

**SHIP** Tier 3 (~590 LOC over 3rd sprint) — Jacobians + BCH + Karcher mean + Tier-A L14 (SU(2) aliases + Sp(2n)). Required for IMU preintegration, pose-graph optimisation, and the symplectic-group cross-link to slot 204.

**DEFER-BUT-DESIGN** Tier 4 (250 LOC) — Lie-Newton / ESM on `optim/manifold.go`. Ship when first explicit consumer pulls. Probable consumers: Pistachio inverse kinematics, future visual-servo / SLAM service. Pairs naturally with slot 102 (optim-missing).

**DEFER** Tier 5 (~360 LOC) — full L14: SO(n), SU(n), SL(n), Galilean, Lorentz, Heisenberg, bicyclic. No consumer today. Defer until: Lorentz wanted by relativity work; SO(n) wanted by ML covariance work; Heisenberg wanted by signal-processing TF analysis (slot signal-* could pull this).

**Cross-slot synergy callouts:**
- **204-I13 (Munthe-Kaas / RKMK)**: explicitly waits for `geometry/so3.go` (this slot's keystone).
- **168 (synergy-physics-autodiff)**: rigid-body Lagrangians need SO(3)/SE(3) for attitude state; HNN/LNN with attitude state need this.
- **027 (chaos-missing)**: three-body and rigid-body chaos systems need SO(3).
- **177 (synergy-geometry-optim)**: directly cross-links — Lie-Newton on SO(3)/SE(3) is the canonical "geometry × optim" piece. Worth re-reading 177 before tier-4 pull.
- **194 (synergy-em-geometry)**: charged-rigid-body dynamics (gyroscopes in B-fields) need SE(3).
- **slot 102 (optim-missing)**: Newton substrate consumed by Lie-Newton in L15.
- **074 (gametheory-api, NOT geometry-api)**: 204's "agent 074 §X confirms" reference is mis-numbered; the actual confirming evidence is the source-tree grep in §0 above.

---

*205-new-lie-groups.md — 398 lines.*

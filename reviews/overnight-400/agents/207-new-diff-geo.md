## 207 | new-diff-geo

**Summary line 1.** reality v0.10.0 ships ZERO abstract-differential-geometry surface — `geometry/` (656 LOC across `quaternion.go`, `sdf.go`, `curves.go`, `polygon.go`) is per-vertex/per-point Pistachio procgen primitives only (rotation algebra on S³ ≃ SU(2) + signed-distance fields + Bezier/Catmull-Rom + 2-D convex-hull) with NO chart/atlas/transition-map type, NO `TangentSpace` / `Tangent​Bundle` / `CotangentSpace`, NO `VectorField` / `OneForm` / `kForm` types, NO `Connection` / covariant-derivative interface, NO `Christoffel` symbol computation, NO `RiemannTensor` / `Ricci` / `ScalarCurvature` / `SectionalCurvature`, NO geodesic-equation ODE, NO `ParallelTransport` / `Holonomy`, NO `KillingField`, NO Gauss-Bonnet, NO de Rham cohomology, NO frame/principal bundles, NO Yang-Mills, NO `LieDerivative` / `LieBracket` outside the implicit so(3)-bracket-as-cross-product in `quaternion.go`. Repo-wide grep on `Christoffel|Riemann tensor|Ricci|sectional|connection|covariant|Levi.Civita|tangent.bundle|Killing|Bianchi|Holonomy|FrameBundle|YangMills|wedge|HodgeStar|deRham|GaussBonnet|exterior` returns ZERO callable surface (the 22 hits are all in `reviews/`, plus one cross-language docstring in `prob/copula/gaussian.go:217` referencing Riemann *integration*, plus four false positives `gametheory/bandit.go` / `fluids/*.go` matching "covariant" in test-only or unrelated comments). The single closest-to-DG primitive is `geometry/quaternion.go` `QuatSlerp` (structurally a geodesic on S³ with the round-metric) and `infogeo/bregman.go` (Bregman divergence which IS a non-symmetric distance from a dual-flat statistical manifold but neither the dual-flat structure nor the Christoffel symbols nor the α-connections are exposed — see slot 092-T2 deferral).

**Summary line 2.** Twenty-eight primitives D1–D28 totalling ~4,100 LOC across new sub-package `geometry/diffgeo/` (mirrors precedent of `geometry/dec/` proposed by 194 and `optim/manifold/` proposed by 206) close the gap with Tier-1 keystone **D6+D7+D9 = `Christoffel(g, ∂g)` + `RiemannTensor(Γ, ∂Γ)` + geodesic-equation RK4** at ~430 LOC ship-now-unblocked because they ride entirely on existing `linalg.MatMul` / `chaos.RK4Step` / `calculus.NumericalGradient` and validate against the three closed-form benchmarks (S² with constant +1/r² sectional curvature, H² hyperbolic plane with constant −1/r² sectional curvature, Schwarzschild metric with closed-form Christoffels) that pin down the entire abstract-DG canon to round-off; Tier-2 ~840 LOC adds `KillingField` + `LieDerivative` + `LieBracket` (all closed-form on coordinates) + `ParallelTransport` + `Holonomy` (Schild's-ladder + closed-form on space-forms); Tier-3 ~1,180 LOC adds `DifferentialForm` types + `WedgeProduct` + `ExteriorDerivative` + `HodgeStar` + `Stokes` (cross-link to 194-em-geometry which proposes the discrete-cochain version — this slot ships the *continuous-coordinate* version, the two are complementary not duplicative); Tier-4 ~830 LOC adds `Connection1Form` / `Curvature2Form` / `BianchiIdentity` validator + frame-bundle / principal-bundle (lift to non-coordinate frames) + `YangMills` action + Euler-Lagrange on principal bundles; Tier-5 ~820 LOC adds GR-tensor-calculus pieces (`EinsteinTensor` G_μν = R_μν − ½ R g_μν, stress-energy coupling, `KretschmannScalar` R_μνρσ R^μνρσ for Schwarzschild singularity detection, `WeylTensor`, `ConformalKilling`, `CartanKarlhede` invariants). **Disambiguation versus 205-new-lie-groups (L1–L15, ~2,400 LOC, `geometry/so3.go`+`geometry/se3.go`):** 205 scoped *Lie-group-specific* exp/log/Adjoint/BCH/Karcher on SO(3)/SE(3)/SU(2)/SL(n)/Sp(2n); this slot scopes the *general-manifold* superset (any (M, g) with coordinate metric, NOT just homogeneous spaces) and only intersects with 205 at D5 `LieBracket` (which 205-L1 ships for so(3) as cross product; this slot's D5 is the general `[X, Y]^k = X^i ∂_i Y^k − Y^i ∂_i X^k` for arbitrary vector fields, with 205-L1 as the constant-coefficient special case). **Disambiguation versus 206-new-riemannian-opt (R1–R22, ~3,200 LOC, `optim/manifold/`):** 206 scoped manifolds *as optimisation domains* (retraction-based first-order/second-order optimisers, exp/log/PT for closed-form manifolds Sphere/Stiefel/Grassmann/SPD/Hyperbolic/PSD); this slot scopes the *abstract-tensor-field side* (Christoffels in arbitrary coordinates, Riemann/Ricci/scalar/sectional curvature, geodesic equation as a coordinate ODE not a closed-form). 206 *uses* a manifold; this slot *defines what a manifold is from the metric tensor*. Recommended placement: `geometry/diffgeo/manifold.go` ships the `CoordinateManifold` type (metric `g(x)` as `func([]float64, [][]float64)`); 206's `Manifold` interface is *implementable in terms of* `CoordinateManifold` for any closed-form manifold (the geodesic equation IS the Exp map computed by RK4 on the Hamiltonian `H = ½ g^{ij} p_i p_j` — slot 206-R10 explicitly defers the geodesic ODE to slot 204 + this slot). **Disambiguation versus 194-synergy-em-geometry (D0–D17, ~2,050 LOC, `geometry/dec/`):** 194 scoped the *discrete-exterior-calculus on simplicial meshes* version (cochain arrays + incidence matrices, Yee-FDTD-on-meshes use case); this slot scopes the *continuous-coordinate-tensor* version (forms as `func([]float64, []float64) float64` evaluators, exterior derivative via symbolic ∂ on basis-form coefficients, Stokes via parametric path-integral validation). The two surfaces are *dual* in the sense of Whitney's de Rham theorem (continuous forms ↔ discrete cochains via Whitney embedding); ideal end-state ships both and validates one against the other. **Disambiguation versus 092-infogeo-missing-T2 (Fisher-Rao + α-connections + dual-flat):** 092 scoped DG of *statistical manifolds* with the Fisher-Rao metric `g_ij = E[∂_i log p · ∂_j log p]` and Amari's α-connections; this slot's D6 `Christoffel(g, ∂g)` from the Levi-Civita formula is the *general-purpose* tool that 092-T2.4 / T2.5 / T2.6 *call into* with the Fisher-Rao metric as a specific input — 092 ships *which metric* on simplex/sphere/SPD; this slot ships *the symbolic Christoffel/curvature machinery from any metric*. **Single-highest-leverage 1-day project: D6 `Christoffel(g_func, x)` numerical via central differences + D7 `RiemannTensor(Γ_func, x)` + D9 geodesic ODE on S² and Schwarzschild = ~430 LOC** validates the entire continuous-DG canon against the three space-form benchmarks (S², H², Minkowski/Schwarzschild) plus the round-trip `Riemann(g)` → `Ricci(R)` → `Scalar(Ric, g^{-1})` = `n(n−1)·κ` for constant-curvature spaces (κ = ±1/r² on S²/H², κ = 0 on flat). **Cutting-edge moat: D26 `KretschmannScalar` + D27 `WeylTensor` + D28 `EinsteinTensor` ~360 LOC** is the singular reality competitive piece (no zero-dependency Go library ships GR tensor calculus with cross-language golden-file contract; only sympy.diffgeo + tensorial.jl + xAct[Mathematica] cover this surface).

---

## 0. State at HEAD (2026-05-08, v0.10.0) — verified by direct read

### `geometry/` (656 LOC, 4 files)

- `quaternion.go` (230 LOC): `QuatIdentity/Dot/Conjugate/Normalize/Mul/Slerp/FromAxisAngle/ToAxisAngle/RotateVec/FromEuler` — pointwise SO(3) algebra. `QuatSlerp` IS structurally a geodesic on S³ with the round metric (`t·log(R0⁻¹·R1)` lifted by `R0·exp(·)`) but is NOT labelled as a manifold operation. The single appearance of "manifold" anywhere in the tree is in `reviews/`.
- `sdf.go` (210 LOC): implicit-surface evaluators. SDFs ARE the level-set side of an immersed manifold (`{x : F(x) = 0}`) but no DG structure (e.g. mean curvature `H = div(∇F/|∇F|)`, Gaussian curvature, second fundamental form) is computed.
- `curves.go` (75 LOC): `LinearInterpolate`, `BezierCubic`, `BezierCubic3D`, `CatmullRom` — single-parameter evaluators, no Frenet frame, no curvature/torsion, no arc-length re-parametrisation.
- `polygon.go` (141 LOC): 2-D triangle/hull primitives.

### `infogeo/` (1,373 LOC, 4 files)

- f-divergences (KL/JS/TV/Hellinger/χ²/Rényi-α), Bregman (squared-Euclidean/genKL/IS/generic), MMD² with RBF kernel. Bregman divergence IS the canonical asymmetric distance from a dual-flat statistical manifold, but NEITHER the Christoffel symbols (Γ⁽ᵅ⁾) NOR the α-connections NOR the Fisher-Rao metric NOR the dual-coordinate transform are exposed. Per `infogeo/doc.go:55-79` MVP scope, manifold structure is explicitly v2-deferred. 092-infogeo-missing T2.1–T2.7 enumerated the gap.

### Repo-wide grep audit

```
$ grep -rn "Christoffel\|Riemann.tensor\|Ricci\|sectional\|covariant\|Levi.Civita\|tangent.bundle\|Killing\|Bianchi\|Holonomy\|FrameBundle\|YangMills\|wedge\|HodgeStar\|deRham\|GaussBonnet\|exterior" --include="*.go"
(zero callable surface; only review-doc + false positives)
```

Confirms reality has *zero* abstract-DG primitives. Quaternion algebra is the ONLY surface that touches DG at all, and it is implicitly specialised to S³ ≃ SU(2) — the simplest 3-D Riemannian symmetric space — with no general-manifold abstraction.

### Cross-link audit

- **205-new-lie-groups L1–L15:** Lie-group-specific exp/log/Adjoint/BCH/Karcher on SO(3)/SE(3)/SU(2). Intersection: D5 (Lie bracket) where 205-L1 ships so(3)-bracket-as-cross-product as constant-coefficient special case of this slot's general bracket-of-vector-fields.
- **206-new-riemannian-opt R1–R22:** manifolds-as-optimisation-domains. Intersection: 206-R10 explicitly defers geodesic-ODE; this slot's D9 geodesic-equation RK4 ships the missing primitive. 206-R12 PT-on-Stiefel uses a closed-form; this slot's D11 PT-via-Schild's-ladder is the general-manifold version 206 falls back to when no closed-form exists.
- **194-synergy-em-geometry D0–D17:** discrete-exterior-calculus on simplicial meshes. Complementary, not duplicative — 194 ships discrete cochains; this slot ships continuous-coordinate forms. End-state validates one against the other via Whitney embedding.
- **092-infogeo-missing T2.1–T2.7:** Fisher-Rao + α-connections + dual-flat. This slot's D6 `Christoffel(g, ∂g)` is the *general-purpose tool* 092 calls with the Fisher-Rao metric as input.
- **204-new-symplectic-int:** Verlet/leapfrog/Yoshida. Required for D9 geodesic-ODE under the Hamiltonian formulation `H = ½ g^{ij} p_i p_j` (see also 206-R10). Not strictly blocking — D9 ships on `chaos.RK4Step` first; symplectic upgrade later.

---

## 1. The twenty-eight primitives

Numbered D1–D28. For each: **(a) what reality ships**, **(b) what to add**, **(c) connective LOC**, **(d) blocker if any**.

### Tier 1 — Ship now, unblocked, validates entire canon (~430 LOC)

#### D1 — Manifold definition: chart / atlas / transition map

(a) **Ships:** nothing. SDF level sets (`sdf.go`) are charts in disguise but no chart/atlas type.

(b) **Add:** `Chart{Domain func([]float64) bool; ToCoords func([]float64, []float64); FromCoords func([]float64, []float64)}` for `M ⊃ U → ℝⁿ`. `Atlas{Charts []Chart; Transition(i, j int) func([]float64, []float64)}` with overlap-compatibility check (transition maps are smooth diffeomorphisms on overlap). For pure-numerical work the atlas is rarely consumed in full — but the chart abstraction is needed for D2 tangent vectors as derivations.

(c) **LOC:** ~80 in `geometry/diffgeo/manifold.go`. Plain types + smoke test on S² (two charts: stereographic from north / south pole).

(d) **No blocker.**

#### D2 — Tangent space T_p M and tangent bundle TM

(a) **Ships:** nothing. `quaternion.go` quaternion-as-rotation IS a tangent vector at identity in disguise (`QuatFromAxisAngle(axis, angle)` with small angle = element of so(3) ≅ T_e SO(3)), but no general tangent-space type.

(b) **Add:** `TangentVector{Base [n]float64; Comp [n]float64}` — a point on M plus components in the chart's coordinate basis `{∂/∂x^i}`. `TangentBundle TM` represented implicitly as the dimension-2n manifold `M × ℝⁿ` (each fibre `T_p M = ℝⁿ` linearised through the chart). For benchmarks: `S² → ℝ³` ambient embedding makes T_p M = `{v ∈ ℝ³ : ⟨p, v⟩ = 0}` recoverable via `ProjectTangent(p, v) = v − ⟨p, v⟩·p` (matches 206-R2 sphere primitive — share code, do not re-ship).

(c) **LOC:** ~70 in `geometry/diffgeo/tangent.go`.

(d) **No blocker.**

#### D6 — Christoffel symbols Γ^k_ij from metric

**Keystone primitive.** All curvature flows from this.

(a) **Ships:** nothing.

(b) **Add:** Levi-Civita Christoffel symbols (the unique torsion-free metric-compatible affine connection):
```
Γ^k_ij = ½ g^{kl} (∂_i g_jl + ∂_j g_il − ∂_l g_ij)
```
where g_ij is the metric and g^{ij} its inverse. Two forms:
- `ChristoffelSymbolic(g func(x []float64) [][]float64, x []float64) [][][]float64` — central-difference numerical ∂g via `calculus.NumericalGradient` (ships at h ≈ ε^{1/3} ≈ 6e-6 with 4-digit accuracy; user supplies analytic ∂g for round-off).
- `ChristoffelAnalytic(g, dg func(x []float64) [][][]float64, x []float64) [][][]float64` — caller-supplied ∂g for benchmark cases (S², H², Schwarzschild).

For benchmark validation, S² in spherical (θ, φ): `g = diag(1, sin²θ)`, only non-zero are `Γ^θ_φφ = −sin θ cos θ` and `Γ^φ_θφ = Γ^φ_φθ = cot θ`. Schwarzschild in (t, r, θ, φ): four families of non-zero Γ documented in MTW p.842; closed form pins the test against round-off.

Reference: do Carmo 1992 *Riemannian Geometry* §2.3; MTW 1973 §13.5; Wald 1984 §3.1.

(c) **LOC:** ~140 in `geometry/diffgeo/christoffel.go`. Includes round-trip test `Christoffel(g_S²) → recover sectional κ = +1` via D7+D8.

(d) **No blocker.** Rides on existing `linalg.MatMul` (for g^{-1}) and `calculus.NumericalGradient` (for ∂g).

#### D7 — Riemann curvature tensor R^a_bcd

(a) **Ships:** nothing.

(b) **Add:**
```
R^a_bcd = ∂_c Γ^a_db − ∂_d Γ^a_cb + Γ^a_ce Γ^e_db − Γ^a_de Γ^e_cb
```
(MTW eq. 14.34). Ships as `RiemannTensor(christoffel func(x) [][][]float64, x []float64) [n][n][n][n]float64`. n⁴ entries; for n=4 (GR) that is 256 numbers but only 20 algebraically independent (R_abcd = −R_bacd = −R_abdc = R_cdab, plus first-Bianchi R^a_[bcd] = 0 reduces from 36 to 20 independent components). Returns the full tensor; `RiemannIndependentComponents(R) [20]float64` companion extracts the 20-vector.

For S² benchmark: `R^θ_φθφ = sin²θ` (constant-curvature space-form), all other components zero or related by symmetries. Round-trip with D8 yields scalar curvature `R = 2·κ = 2/r²` for S²(r).

Reference: do Carmo §4.3; MTW §11.3; Wald §3.2.

(c) **LOC:** ~140 in `geometry/diffgeo/riemann.go`. Includes the 20-component extraction + first-Bianchi validator.

(d) **No blocker.**

#### D9 — Geodesic equation ODE

(a) **Ships:** nothing. `chaos.RK4Step` ships in `chaos/integrate.go`.

(b) **Add:** geodesic equation as 2nd-order ODE: `ẍ^k + Γ^k_ij ẋ^i ẋ^j = 0`, reduced to a 2n-D first-order system `[x; ẋ]'` for `chaos.RK4Step`. Ships as `Geodesic(Γ, x0, v0, t1, n_steps) [N+1][2n]float64`. Validates on S²: closed-loop great-circle (`x0 = (1,0,0)`, `v0 = (0, 1, 0)`, `t = 2π`, returns to `x0`). Hyperbolic-plane closed-form: half-circle in upper-half-plane model. Schwarzschild: precession-of-Mercury benchmark (check 43" / century perihelion advance — flagged as a 174-physics-missing extension test).

For Hamiltonian formulation (preferred for symplectic integration when 204 ships): `H = ½ g^{ij} p_i p_j`, `ẋ^i = ∂H/∂p_i = g^{ij} p_j`, `ṗ_i = −∂H/∂x^i = −½ ∂_i g^{jk} p_j p_k`. Verlet/leapfrog from 204 preserves both energy AND symplectic 2-form to round-off over 10⁶ steps; RK4 drifts at O(t⁴ · ε) per step.

Reference: MTW §13.4 (Lagrangian form) + §17.5 (Hamiltonian form); Hairer-Lubich-Wanner 2006 §VII.

(c) **LOC:** ~150 in `geometry/diffgeo/geodesic.go`. Two implementations: RK4 (default, on `chaos.RK4Step`) and Verlet (when 204 ships).

(d) **Soft blocker on 204** for symplectic-Verlet upgrade; RK4 ships unblocked.

### Tier 2 — Vector fields, Lie derivatives, parallel transport, holonomy (~840 LOC)

#### D3 — Cotangent space T*_p M

(a) **Ships:** nothing.

(b) **Add:** `Cotangent​Vector{Base, Comp [n]float64}` — covector / 1-form at a point. Pairing `⟨ω, v⟩ = ω_i v^i`. Index-raising `v^i = g^{ij} ω_j`. Type-tag distinct from `TangentVector` to prevent contraction errors at the type level.

(c) **LOC:** ~50 in `geometry/diffgeo/cotangent.go`. Pairs with D2.

(d) No blocker.

#### D4 — Vector fields as sections of TM

(a) **Ships:** nothing.

(b) **Add:** `VectorField func(x []float64, out []float64)` — a function from M to TM with `(out, x)` such that `out ∈ T_x M` (i.e., for embedded manifolds, `out` is in the tangent subspace). Smoothness is contract-only (caller responsibility); we ship `IsTangent(x, out, projector)` validator that uses D2's projection.

Examples: rotation generator `X(θ, φ) = ∂_φ` on S² (constant in coords; pointwise `(0, 1)`). Killing field on Schwarzschild: `∂_t` (time-translation) and `∂_φ` (axial rotation).

(c) **LOC:** ~70 in `geometry/diffgeo/vectorfield.go`.

(d) No blocker.

#### D5 — Lie bracket [X, Y]

(a) **Ships:** so(3) bracket as cross product (implicit in `quaternion.go`). 205-L1 makes the SO(3) special case explicit.

(b) **Add:** general `[X, Y]^k = X^i ∂_i Y^k − Y^i ∂_i X^k`. Ships as `LieBracket(X, Y VectorField, x []float64) []float64` using `calculus.NumericalGradient` for ∂Y, ∂X. For closed-form (analytic) X, Y the caller supplies derivatives.

Validates: Jacobi identity `[[X,Y], Z] + [[Y,Z], X] + [[Z,X], Y] = 0` to round-off. Reduces to so(3) cross product when X, Y are constant-coefficient on Lie groups.

Reference: Lee 2012 *Introduction to Smooth Manifolds* §8.

(c) **LOC:** ~110 in `geometry/diffgeo/liebracket.go`.

(d) No blocker.

#### D8 — Ricci tensor, scalar curvature, sectional curvature

(a) **Ships:** nothing.

(b) **Add:**
- `RicciTensor(R [n][n][n][n]float64) [n][n]float64` via contraction `R_bd = R^a_bad`.
- `ScalarCurvature(Ric, g_inv) float64` via `R = g^{bd} R_bd`.
- `SectionalCurvature(R, g, p, X, Y) float64` via `K(X, Y) = R(X, Y, X, Y) / (⟨X,X⟩⟨Y,Y⟩ − ⟨X,Y⟩²)`.

Validates: S²(r) yields `R_bd = (1/r²) g_bd` (Einstein manifold), `R = 2/r²`, `K = 1/r²`. H²(r) yields `R = −2/r²`, `K = −1/r²`. Schwarzschild vacuum: `R_μν = 0` (Ricci-flat), but R^a_bcd ≠ 0 (Weyl-tensor is non-zero — the curvature is "tidal" / "free-gravitational").

Reference: do Carmo §4; MTW §14; Lee 2018 *Introduction to Riemannian Manifolds* §7.

(c) **LOC:** ~120 in `geometry/diffgeo/curvature.go`.

(d) Depends on D7.

#### D10 — Lie derivative of tensor fields

(a) **Ships:** nothing.

(b) **Add:** `LieDerivative_X(T)` for scalars (= X[f] = X^i ∂_i f), 1-forms, vector fields (= [X, Y] for vector fields, ships in D5), and rank-(0, 2) symmetric tensors (Killing equation: `(L_X g)_ij = ∇_i X_j + ∇_j X_i`, vanishing iff X is Killing). Generic formula uses Cartan magic `L_X = ι_X d + d ι_X` for forms — ships in D17.

Reference: Lee 2012 §12.

(c) **LOC:** ~150 in `geometry/diffgeo/liederiv.go`.

(d) No blocker.

#### D11 — Parallel transport along a curve (Schild's ladder + closed-form on space-forms)

(a) **Ships:** nothing. 206-R12 ships closed-form PT for Sphere/Stiefel/Grassmann.

(b) **Add:** numerical parallel transport via the geodesic-equation augmented with `Ẏ^k + Γ^k_ij ẋ^i Y^j = 0` (transport equation). Ships as `ParallelTransport(Γ, path, V0, n_steps)`. Schild's ladder approximation for short hops on arbitrary manifolds (no Γ needed, only Exp/Log). Closed forms for S² (great-circle PT preserves angle to tangent of geodesic), H² (Möbius transformation), Euclidean (identity).

Validates: PT around closed loop on S² yields rotation by area-of-loop / r² (the shipped `Holonomy` D14 test).

Reference: MTW §10.2; Lee 2018 §4.

(c) **LOC:** ~180 in `geometry/diffgeo/transport.go`.

(d) No blocker; depends on D6/D9.

#### D12 — Holonomy group around closed loops

(a) **Ships:** nothing.

(b) **Add:** `Holonomy(Γ, loop, V0)` returns `ΔV = V_end − V_start` after PT around a closed loop. On S²: classic Berry-phase / parallel-translated tangent rotates by signed-area-of-loop / r² (Levi-Civita 1917). On flat: zero (`ΔV = 0`). On Schwarzschild: gravitomagnetic frame-dragging. Cross-link to 159 (frame dragging in GR).

Reference: Berger 2003 *A Panoramic View of Riemannian Geometry* §12; Berry 1984 *Proc. Roy. Soc. A* 392:45.

(c) **LOC:** ~110 in `geometry/diffgeo/holonomy.go`.

(d) Depends on D11.

#### D13 — Killing vector fields (isometries of the metric)

(a) **Ships:** nothing.

(b) **Add:** `IsKilling(X, g, x)` validates Killing equation `∇_(i X_j) = 0`. `KillingFields(g)` symbolic-or-numerical returns the basis (e.g. for Schwarzschild: `∂_t`, `∂_φ`; for Minkowski: 10 generators of Poincaré algebra; for S²: 3 rotation generators isomorphic to so(3)). Used for conserved quantities along geodesics (Noether — energy from `∂_t` Killing, angular momentum from `∂_φ`).

Reference: Wald §B.1; MTW §25.2.

(c) **LOC:** ~100 in `geometry/diffgeo/killing.go`.

(d) Depends on D5/D11.

#### D14 — Conformal Killing + isometric vs conformal classification

(a) **Ships:** nothing.

(b) **Add:** Conformal Killing equation `∇_(i ξ_j) = (1/n) g_ij ∇·ξ` (CKV). Tests for conformal flatness (Weyl tensor = 0 in n ≥ 4; Cotton tensor = 0 in n = 3). Cross-link to 196-color (CIE space conformal flat) and to 174-geometry-missing (conformal SDF).

(c) **LOC:** ~80 in `geometry/diffgeo/conformal.go`.

(d) Depends on D8.

### Tier 3 — Differential forms + exterior calculus + Stokes (~1,180 LOC)

#### D15 — Differential 1-forms / k-forms

(a) **Ships:** nothing.

(b) **Add:** `OneForm = Cotangent​Vector` (alias). `KForm{Coeffs [C(n,k)][n^k]float64; Rank int}` — antisymmetric covariant tensor of rank k. Implementation: store the `C(n,k)` independent components in a sorted-multi-index canonical form.

Examples: 2-form `dx ∧ dy` on ℝ²; volume form `√|g| dx¹ ∧ ... ∧ dx^n` on (M, g).

(c) **LOC:** ~200 in `geometry/diffgeo/forms.go`.

(d) No blocker.

#### D16 — Wedge product ω ∧ η

(a) **Ships:** nothing.

(b) **Add:** `WedgeProduct(omega [r-form], eta [s-form]) [(r+s)-form]`. Antisymmetry: `ω ∧ η = (−1)^{rs} η ∧ ω`. Associativity. For pure index work this is a re-shuffle-and-sign-multiply over multi-indices.

(c) **LOC:** ~120 in `geometry/diffgeo/wedge.go`.

(d) Depends on D15.

#### D17 — Exterior derivative d : Ω^k → Ω^{k+1}

(a) **Ships:** nothing in continuous form. 194-D1 ships discrete-cochain version on simplicial meshes.

(b) **Add:** `ExteriorDerivative(omega KForm, x []float64) KForm` via `(dω)_{i₀...iₖ} = (k+1) ∂_[i₀ ω_{i₁...iₖ}]`. Validates: `d² = 0` (Poincaré lemma) — flagship round-trip test.

Includes Cartan magic: `L_X = ι_X d + d ι_X` cross-validation against D10. Closes the loop with 194's discrete `d` via Whitney embedding.

Reference: do Carmo §6; Lee 2012 §14.

(c) **LOC:** ~180 in `geometry/diffgeo/exterior.go`.

(d) Depends on D15.

#### D18 — Hodge star ★ : Ω^k → Ω^{n−k}

(a) **Ships:** nothing.

(b) **Add:** Hodge dual `★ω = √|g| ε_{i₁...iₖj₁...j_{n−k}} ω^{i₁...iₖ}` (with metric and orientation). Codifferential `δ = (−1)^{...} ★d★`. Laplace-de Rham `Δ = dδ + δd`.

Cross-link to 194-D6 (discrete ★ via primal-dual mesh ratios).

(c) **LOC:** ~140 in `geometry/diffgeo/hodge.go`.

(d) Depends on D15/D17.

#### D19 — Stokes' theorem on manifolds

(a) **Ships:** `calculus.SimpsonsRule` for 1-D continuous integration; 194-D5 ships discrete `dd = 0` and validates Stokes on triangle meshes.

(b) **Add:** `IntegrateForm(omega KForm, parametrisation func(...) []float64, domain) float64` via Simpson on the parametrised domain. Validates Stokes `∫_∂M ω = ∫_M dω` against 194-D5 to round-off on test meshes.

Special cases: divergence theorem (n-form), Green's theorem (2-form on ℝ²), classical Stokes (curl on ℝ³).

(c) **LOC:** ~170 in `geometry/diffgeo/stokes.go`. Pure connective tissue over `calculus`.

(d) Depends on D17.

#### D20 — de Rham cohomology

(a) **Ships:** nothing. 194-D8 ships discrete Hodge decomposition on meshes.

(b) **Add:** `DeRhamBetti(M)` returns Betti numbers β_k = dim H^k_dR(M) via mode-counting of harmonic forms (Δω = 0, kernel-of-Laplacian-on-forms). For the test cases: torus T² has β₀=1, β₁=2, β₂=1; sphere S² has β₀=1, β₁=0, β₂=1; figure-8 (non-manifold) flagged as out-of-scope.

Cross-link to topology/persistent (Betti numbers are the rank of persistent-homology groups H_k as filtration → ∞).

Reference: Bott-Tu 1982 *Differential Forms in Algebraic Topology* §8.

(c) **LOC:** ~370 in `geometry/diffgeo/derham.go`. Largest single module — needs sparse-eigensolver from `linalg` (097-T1 sub-blocker).

(d) Sub-blocker on `linalg.SparseEigen` (currently absent).

### Tier 4 — Frame / principal bundles + curvature 2-forms + Yang-Mills (~830 LOC)

#### D21 — Frame bundle FM (~110 LOC, `geometry/diffgeo/frame.go`)

`Frame{Base [n]float64; Basis [n][n]float64}` orthonormal-frame at point (columns span T_p M, mutually orthogonal under g); FM = fibre-bundle with fibre O(n). Cross-link to 205-L4 (Mat3ToQuat = SO(3) ↪ FM(S²) lift). Ref: KN 1963 §III. No blocker.

#### D22 — Principal G-bundle + connection 1-form (~150 LOC, `geometry/diffgeo/principal.go`)

`PrincipalBundle{Base, FibreGroup}`; `Connection1Form{omega func(g, X) g.algebra}` ω ∈ Ω¹(P, 𝔤) horizontal-distribution form. For Yang-Mills: G = SU(N), ω = gauge field A_μ. Ref: KN II §II; Bleecker 1981 §3. Depends on 205-L1+L2 for Lie-algebra exp/log.

#### D23 — Curvature 2-form Ω = dω + ω ∧ ω (~140 LOC, `geometry/diffgeo/curv2form.go`)

Cartan structure equation. For Yang-Mills: F_μν = ∂_μ A_ν − ∂_ν A_μ + [A_μ, A_ν] (recovers Maxwell at G = U(1)). On Levi-Civita: R^a_b = ½ R^a_{bcd} dx^c ∧ dx^d. Ref: KN II §II.5; MTW §14. Depends on D7+D22.

#### D24 — Bianchi identities (~80 LOC, `geometry/diffgeo/bianchi.go`)

First (algebraic, R^a_{[bcd]} = 0) ships as D7 validator; second (differential, ∇_{[e} R^a_{|b|cd]} = 0); reduced trace ∇_a G^{ab} = 0 on Einstein tensor (= local energy-momentum conservation). Ref: MTW §15.5. Depends on D7+D8.

#### D25 — Yang-Mills action S_YM = ½ ∫ tr(F ∧ ★F) (~200 LOC, `geometry/diffgeo/yangmills.go`)

YM functional + Euler-Lagrange `∇_μ F^{μν} = J^ν` (covariant Maxwell at G = U(1); non-abelian at G = SU(N)). Specialises to `em/` via U(1) constraint. Ref: Bleecker §6; Atiyah-Bott 1983. Depends on D22+D23.

#### D26 — Gauss-Bonnet theorem (~150 LOC, `geometry/diffgeo/gaussbonnet.go`)

2-D form: ∫_M K dA + ∮_∂M k_g ds = 2π χ(M). Validates: S²(r) → 4π = 2π·2 ✓; T² → 0 = 2π·0 ✓. Higher-dim Chern-GB (Pfaffian of Ω) v2-deferred. Ref: do Carmo 1976 §4.5. Depends on D8+D19.

### Tier 5 — GR tensor calculus + advanced invariants (~360 LOC)

#### D27 — Einstein tensor + GR field equations (~110 LOC, `geometry/diffgeo/einstein.go`)

`G_μν = R_μν − ½ R g_μν`. GR equation G_μν + Λ g_μν = (8πG/c⁴) T_μν as relation type (T_μν solver is 174-physics-missing scope). Vacuum solutions: Schwarzschild / Kerr / FLRW. Cross-link `constants.NewtonianG`, `constants.SpeedOfLight`. Ref: MTW §17; Wald §4. Depends on D8.

#### D28 — Kretschmann + Weyl + Cartan-Karlhede invariants (~250 LOC, `geometry/diffgeo/invariants.go`)

- `KretschmannScalar(R) = R_μνρσ R^μνρσ` — singularity detector. Schwarzschild: K = 48 G²M²/(c⁴ r⁶), diverges at r=0 (true), finite at r=2GM/c² (coord-only). Cleanest separation of physical vs coordinate singularities.
- `WeylTensor(R, Ric, g, n)` C^a_bcd — trace-free Riemann; vanishes iff (n≥4) conformally flat. Schwarzschild: nonzero (tidal) despite Ricci-flat.
- `RicciScalarSquared(Ric) = R_μν R^μν`.

Canonical CK invariants for spacetime classification (Karlhede 1980 *GRG* 12:693). Ref: Stephani et al. 2003 §9. Depends on D7+D8.

---

## 2. Recommended ship plan

| Tier | Slot range | LOC  | Ship-now? | Blocker             | Deliverable                              |
|------|-----------|------|-----------|---------------------|------------------------------------------|
| 1    | D1+D2+D6+D7+D9 | 430  | yes       | none                | Christoffel + Riemann + geodesic on S²/H²/Schwarzschild |
| 2    | D3-D5,D8,D10-D14 | 840  | yes       | Tier-1              | Lie deriv + PT + holonomy + Killing      |
| 3    | D15-D20  | 1,180 | partial   | 097-SparseEigen for D20 | Forms + wedge + d + ★ + Stokes + de Rham |
| 4    | D21-D26  | 830  | partial   | 205-L1, D7-D8       | Frame/principal bundles + Yang-Mills + Gauss-Bonnet |
| 5    | D27-D28  | 360  | yes       | D7+D8               | Einstein tensor + Kretschmann + Weyl     |

**Total: 28 primitives, ~4,100 LOC.**

**Single highest-leverage 1-day project:** Tier-1 (D1+D2+D6+D7+D9, ~430 LOC). Validates entire abstract-DG canon against three closed-form benchmarks (S²/H²/Schwarzschild) and unblocks every other tier. Plain Go, no new dependencies, rides entirely on existing `linalg.MatMul` + `chaos.RK4Step` + `calculus.NumericalGradient`.

**Cutting-edge moat:** Tier-5 (D27+D28, ~360 LOC) ships GR tensor-calculus invariants (Einstein, Kretschmann, Weyl, Cartan-Karlhede) — no zero-dependency Go library has this surface. Closest cross-language references are sympy.diffgeo (Python), tensorial.jl (Julia), xAct (Mathematica, proprietary) — reality would be the only Go shop with byte-for-byte cross-language golden-file contract.

**Cross-package cycles:** `geometry/diffgeo/` imports `linalg`, `calculus`, `chaos`, `constants` — all one-way edges, no cycle. `geometry/dec/` (194) is sibling not parent. `optim/manifold/` (206) imports this slot for the geodesic-ODE upgrade path.

**Critical cross-link:** D9 geodesic-equation IS slot 206-R10 (`Manifold.Exp`). Recommended sequencing: ship Tier-1 here, then 206 imports D9 as the default `Exp` implementation for any `CoordinateManifold`; closed-form Exp on Sphere/Stiefel/Grassmann/SPD remains specialised in 206 for performance.

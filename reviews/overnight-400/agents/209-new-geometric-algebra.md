# 209 | new-geometric-algebra

**Summary line 1.** NINTH Block-C cutting-edge-math review and FIRST Clifford / geometric-algebra (GA) scoping in the 400-sequence covering Cl(p,q,r) basis e_i² ∈ {+1,−1,0} / geometric product ab = a·b + a∧b (inner + outer in one operation) / multivector grades 0..n (scalar/vector/bivector/trivector/pseudoscalar) / Cl(3,0) Euclidean GA / Cl(1,3) spacetime algebra (STA) / Cl(4,1) conformal GA (CGA, Hestenes 2001) / rotors R = exp(B/2) for any-dim rotation / reflections via versor sandwich x' = −m x m / quaternions ⊂ Cl(3,0)+ even subalgebra / Pauli matrices = Cl(3,0) / Dirac matrices = Cl(1,3) / outermorphism / dual A* = A I^(−1) / inner / outer / scalar / left+right contractions / regressive (meet) / CGA conformal embedding x → x + ½x²e∞ + e0 / CGA primitives (point/sphere/plane/circle/line all as multivectors) / CGA motors (translation+rotation+dilation+inversion as one rotor) / versor decomposition / geometric calculus ∇ = e^i ∂_i (subsumes grad/div/curl) / spinors as left-ideal elements / Maxwell as ∇F = J (one equation in STA) / robotics rigid motion as single CGA motor: reality v0.10.0 ships **ZERO** GA surface — `geometry/{quaternion,sdf,curves,polygon}.go` total 656 LOC has only quaternion algebra (which IS Cl(3,0)+ via i=e₂e₃ j=e₃e₁ k=e₁e₂ — Hamilton product equals Clifford product on the even subalgebra — but is not labelled as such), and repo-wide grep on `Clifford|Cl\(3|GeometricProduct|MultiVector|Bivector|Rotor|Pseudoscalar|ConformalGA|Versor|Spinor|Outermorphism` returns ZERO callable matches (only false positives `audio/separation/doc.go:64` mechanical "multi-rotor" comment + 13 review-doc-only mentions in `reviews/`). Reality has been shipping a special case of GA on Cl(3,0)+ since v0.1, but the connection is invisible at the API and there is no path to lift quaternions into a higher-dim or different-signature Clifford algebra (no SE(3) motor, no spacetime rotor, no CGA point).

**Summary line 2.** Twenty-two primitives G1–G22 totalling ~3650 LOC across new sub-package `geometry/ga/` (mirrors 194 `geometry/dec/` + 207 `geometry/diffgeo/` + 208 `geometry/extcalc/` four-sibling-precedent — `dec/` discrete cochains, `diffgeo/` Christoffel/Riemann on charts, `extcalc/` continuous Λ^k forms, `ga/` Clifford multivectors over a fixed signature). Tier-1 keystone **G1+G2+G3+G4 = generic `Cl(p,q,r)` `Multivector` + Cayley table + `GeometricProduct` + grade projection ⟨A⟩_k ~580 LOC** ships unblocked and validates against 7 closed-form benchmarks (Cl(0,0)=ℝ, Cl(0,1)=ℂ, Cl(0,2)=ℍ quaternions, Cl(2,0)=Pauli pair, Cl(3,0)=8-dim Euclidean GA, Cl(1,3)=16-dim spacetime, Cl(4,1)=32-dim conformal). Tier-2 ~640 LOC adds wedge ∧, contractions ⌋ ⌊, scalar product, dual A·I^{−1}, reverse Ã, grade involution, Clifford conjugation, outermorphism, regressive. Tier-3 ~720 LOC adds rotor exp/log on bivectors (closed form by sign of B² — three branches Euclidean/Lorentzian/null), versor sandwich x' = ±VxṼ, Cartan-Dieudonné decomposition (every rotor = product of vectors), bivector splitting. Tier-4 ~860 LOC ships three flagship signatures: Cl(3,0) Euclidean (quaternion bridge G17), Cl(1,3) STA (boost+rotation as bivector exp, spacetime split G18), Cl(4,1) CGA (point/sphere/plane/circle/line/motor + similarity transforms G19). Tier-5 ~860 LOC ships geometric calculus ∇ = Σe^i ∂_i (subsumes grad/div/curl), Maxwell as ∇F = J (four equations collapse to one), Pauli + Dirac matrix representations, spinors-as-left-ideals. **Disambiguation:** versus 205-new-lie-groups L1–L15 (matrix-form Lie-group-specific exp/log on SO(3)/SE(3)/SU(2)) — this slot ships the Clifford superset where Spin(p,q) ⊂ Cl(p,q)+, all rotations `exp(B/2)`, SE(3) = single CGA motor not 4×4 matrix split; quaternion bridge is Cl(3,0)+↔ℍ literal. Versus 207-new-diff-geo (coordinate-chart Christoffel/Riemann numerical arrays) — this slot ships coordinate-free ∇ on multivectors; Riemann becomes bivector-valued bivector operator <50 LOC versus 207-D7 256-number array. Versus 208-new-exterior-calculus (antisymmetric k-forms with wedge-only multiplication) — this slot ships full Clifford product ab = a·b + a∧b of which 208's wedge is grade-(r+s) component and inner is grade-|r−s|; recommended re-home: **208-E2 Wedge SHOULD be implemented as `GradeProject(GP(A,B), r+s)`** single source of truth, saves ~120 LOC there. Versus 194-em-geometry (discrete cochain DEC on meshes) — orthogonal axis, both ship. **Single highest-leverage 1-day project: Tier-1 G1+G2+G3+G4 ~580 LOC** saturates R-MUTUAL-CROSS-VALIDATION 3/3 via three independent paths: Cl(3,0)+ even-subalgebra GP byte-equality with `QuatMul`; Cl(2,0) GP byte-equality with Pauli matrix product on 2×2 complex; Cl(0,2) GP byte-equality with quaternion product via (i,j,k)→(e₁,e₂,e₁e₂) isomorphism. **Singular reality competitive moat: G19 CGA + G21 Maxwell-as-∇F=J ~520 LOC** — no zero-dependency Go library ships either surface (closest: clifford.py NumPy-dep, ganja.js browser-only, Gaalop Java code-generator, Garamon C++ template-heavy, Versor C++ wrong-convention, GAlgebra Mathematica proprietary). Reality would be the only Go shop, and only language ecosystem at all, with byte-for-byte cross-language golden-file contract on CGA primitives. Cross-package blockers: `linalg.SchurDecomposition` (currently absent) gates G14 high-dim bivector splitting (Cl(3,0)/Cl(1,3) closed-form fallback ships unblocked); `em/` for G21 Maxwell cross-validation; 207-D7 RiemannTensor for G16 curved-space rotor test (flat case ships).

---

## 0. State at HEAD (2026-05-08, v0.10.0)

`geometry/` 656 LOC src + 774 LOC test, four files. `quaternion.go` (230 LOC, 9 functions: `QuatIdentity/Dot/Conjugate/Normalize/Mul/Slerp/FromAxisAngle/ToAxisAngle/RotateVec/FromEuler`, Hamilton convention `[w,x,y,z]`). **Critical observation:** the Hamilton product `(a₀+a₁i+a₂j+a₃k)(b₀+b₁i+b₂j+b₃k) = (a₀b₀ − a·b) + (a₀b + b₀a + a×b)` IS literally the Clifford product on Cl(3,0)+ with i=e₂e₃, j=e₃e₁, k=e₁e₂. `QuatRotateVec` is the optimised sandwich `xRx̃` formula. `QuatFromAxisAngle` is the Cl(3,0)+ rotor exp via half-angle. `sdf.go` (210 LOC), `curves.go` (75 LOC), `polygon.go` (141 LOC) — none touch GA explicitly, though `TriangleArea2D` IS the magnitude of the bivector `(b−a)∧(c−a)/2` exposed as a det-2 scalar.

`linalg/` ships `MatMul`, `MatTranspose`, `LU/QR/Cholesky/Eigen` — no `MatrixExp`, no `MatrixLog`, no Schur, no Padé. Bivector exp doesn't need generic `MatrixExp` because for any single bivector B with B² scalar, the closed form is one trig (or hyperbolic, or linear) call — see G12.

Repo-wide grep audit: zero callable GA surface. Two false-positive hits (`audio/separation/doc.go:64` mechanical-rotor comment, `timeseries/garch/` test variable name). Quaternion is the only GA-adjacent code in v0.10.0.

Cross-link audit: 205-new-lie-groups (matrix exp/log/Adjoint on SO(3)/SE(3)/SU(2) — this slot reformulates as bivector exp + sandwich, all rotors); 207-new-diff-geo (coordinate Christoffel/Riemann; this slot ships coordinate-free ∇); 208-new-exterior-calculus (antisymmetric k-forms; this slot's wedge subsumes 208's E2 — re-home recommended); 194-em-geometry (discrete cochain DEC on meshes; sibling sub-package, orthogonal axis); 159-em-signal (mentions GA for Maxwell — G21 ships ∇F=J as the single-equation form); 112/107 physics-/orbital-missing (want spinor formulations — G22 ships Cl(3,0)/Cl(1,3) representations).

---

## 1. The twenty-two primitives

### Tier 1 — Multivector + signature + GP + grade projection (~580 LOC, ship-now-unblocked)

**G1 — `Multivector` type + `Signature{P,Q,R}` ~120 LOC** in `geometry/ga/multivector.go`. `Signature{P,Q,R int}`: basis `e_1..e_n` with `n=P+Q+R`, `e_i² = +1` for i ∈ [1,P], `−1` for i ∈ (P,P+Q], `0` for i ∈ (P+Q, n] (degenerate metric needed for projective GA + CGA-at-infinity). `Multivector{Sig; Coeffs []float64}` with `len(Coeffs) = 2^n`, indexed by bitmask: bit i set in mask = blade `e_i` present (mask `0b110` = blade `e₂∧e₃` at offset 6). API: `New(sig)`, `Set(mask,v)`, `Get(mask)`, `Grade()`, `IsBlade()`, `IsHomogeneous(grade)`, `String()`.

**G2 — Cayley table ~140 LOC** in `geometry/ga/cayley.go`. `cayleyTable(sig) [][]TableEntry` precomputes for every (mask_a, mask_b) pair the resulting blade and sign via popcount on bit-reversed AND. O(4^n) precompute, O(1) per lookup. For n ≤ 6 (covers Cl(4,1) CGA = 32 elements, table ~8 KB) fits in L1 cache; for n>6, on-the-fly `swapSign(maskA, maskB)` fallback (+30 LOC).

**G3 — Geometric product `GP(A, B)` — KEYSTONE ~280 LOC** in `geometry/ga/product.go`. `GPDense(A,B)` naive O(2^{2n}); `GPSparse(A,B)` skips zero coefficients, reduces grade-1 × grade-1 to O(n²) matching `a·b + a∧b` exactly. Tests: ungrading round-trip `GP(a,b) = ScalarProduct(a,b) + Wedge(a,b)` for grade-1; quaternion equivalence Cl(3,0)+ GP byte-equal `QuatMul`; Pauli equivalence Cl(3,0) GP byte-equal Pauli σ-matrix product; Dirac equivalence Cl(1,3) GP byte-equal γ-matrix product; associativity `GP(A,GP(B,C)) = GP(GP(A,B),C)` round-off. Refs: Hestenes-Sobczyk 1984 *Clifford Algebra to Geometric Calculus*; Doran-Lasenby 2003 *GA for Physicists* §4; Dorst-Fontijne-Mann 2007 §6.

**G4 — Grade projection ⟨A⟩_k ~40 LOC** in `geometry/ga/grade.go`. `Grade(A,k)` zeroes blades of grade ≠k. `GP(a,b) = Σ_k ⟨GP(a,b)⟩_k`; for grade-1 a,b: `⟨ab⟩_0 = a·b` (inner), `⟨ab⟩_2 = a∧b` (outer). G4 is the dispatcher that turns G3 into specialised inner/outer/contraction operations. Plus `MaxGrade(A)`, `IsGrade(A,k)`.

### Tier 2 — Outer / inner / contractions / dual / reverse / outermorphism / regressive (~640 LOC)

**G5 — Wedge `Wedge(A,B) = ⟨GP(⟨A⟩_r, ⟨B⟩_s)⟩_{r+s}` ~80 LOC** in `geometry/ga/wedge.go`. Antisymmetric on grade-1; `a∧a = 0`. **Re-home recommendation:** 208-E2 `WedgeProduct` should call this as a thin wrapper — single source of truth across the repo.

**G6 — Inner + left/right contractions ~120 LOC** in `geometry/ga/inner.go`. `Inner(A,B) = (GP(A,B)+GP(B,A))/2` Hestenes; `LeftContraction(A,B) = ⟨GP(⟨A⟩_r,⟨B⟩_s)⟩_{s−r}` (lowering); `RightContraction(A,B) = ⟨GP(⟨A⟩_r,⟨B⟩_s)⟩_{r−s}`. For grade-1 vectors: all reduce to metric `a·b = Σ_i sig(i) a_i b_i`. Pin Dorst-Fontijne-Mann convention in docstring.

**G7 — Reverse + grade involution + Clifford conjugation ~80 LOC** in `geometry/ga/involution.go`. `Reverse(A)` reverses blade order, sign `(−1)^{k(k−1)/2}` on grade k; `Involution(A)` flips grade-odd, sign `(−1)^k`; `Conjugate = Reverse ∘ Involution` generalises quaternion conjugate. On Cl(3,0)+: `Conjugate` byte-equals `QuatConjugate`. Useful identity: for unit rotor R, `Reverse(R) = R^{-1}` (no inversion needed).

**G8 — Norm `MagSq(A) = ⟨GP(A,Reverse(A))⟩_0` ~50 LOC** in `geometry/ga/norm.go`. **Caution:** in mixed signature (q>0) MagSq can be negative or zero (null elements). For unit rotor `|R|² = 1`; for null vector in CGA (embedded point) `|x_CGA|² = 0` exactly — pinning test G8'.

**G9 — Dual `Dual(A) = GP(A, PseudoscalarInverse())` ~60 LOC** in `geometry/ga/dual.go`. Swaps grade-k ↔ grade-(n−k). On Cl(3,0): `Dual(e₁) = e₂e₃`, `Dual(e₁e₂) = e₃`. On CGA: dual transforms direct-form ↔ dual-form (sphere-as-vector ↔ sphere-as-trivector), central to CGA primitive duality. Tests: `Dual(Dual(A)) = ±A` with sign by signature parity (cross-language pin per Dorst-Fontijne-Mann §3.5).

**G10 — Outermorphism `OutermorphismLift(linMap [n][n]float64) func(MV) MV` ~110 LOC** in `geometry/ga/outermorphism.go`. Any linear `f:V→V` lifts to grade-preserving algebra map `f̄:Cl→Cl` via `f̄(a₁∧…∧aₖ) = f(a₁)∧…∧f(aₖ)`. The universal mechanism by which a 3×3 rotation matrix acts coherently on bivectors and trivectors. Refs: Hestenes-Sobczyk §3; Dorst et al. §4.

**G11 — Regressive product (meet) `Regressive(A,B) = Dual^{-1}(Wedge(Dual(A), Dual(B)))` ~140 LOC** in `geometry/ga/regressive.go`. De Morgan dual of wedge — wedge is "join" (smallest blade containing A,B), regressive is "meet" (largest blade contained in both). On Cl(3,0): regressive of two parallel planes is the line of intersection. Critical for projective GA + CGA primitive intersections.

### Tier 3 — Rotor exp/log + versor sandwich + Cartan-Dieudonné + bivector split (~720 LOC)

**G12 — Rotor exp `Exp(B)` on grade-2 bivector ~150 LOC** in `geometry/ga/rotor.go`. **Closed form by sign of B², no matrix series:**
- B² = −β² < 0 (Euclidean rotation): `exp(B) = cos β + (B/β) sin β`.
- B² = +β² > 0 (Lorentz boost in STA): `exp(B) = cosh β + (B/β) sinh β`.
- B² = 0 (null bivector, CGA translation/dilation): `exp(B) = 1 + B`.
Three branches, one signature dispatch. Non-blade B (sum of orthogonal planes) falls back to G14 splitting then exp each separately. Tests: round-trip `Log(Exp(B)) = B` round-off; on Cl(3,0)+ `Exp(θ/2 · e₂e₃)` byte-equals `QuatFromAxisAngle({1,0,0},θ)`. `linalg.MatrixExp` NOT needed.

**G13 — Versor sandwich `Apply(R,x) = GP(R, GP(x, Reverse(R)))` ~110 LOC** in `geometry/ga/versor.go`. Tag distinguishes `EvenVersor` (rotor, `+RxR̃`) from `OddVersor` (reflection, `−RxR̃`). Vector-on-vector `Apply(m,x) = −mxm̃` reflects through hyperplane orthogonal to m. **Cartan-Dieudonné theorem:** every rotor decomposes as a product of vectors (= reflections); composition of two reflections = rotation by twice the angle between them. Tests: byte-equality with `QuatRotateVec` on Cl(3,0)+; reflection idempotence `Apply(m,Apply(m,x)) = x`.

**G14 — Bivector splitting `OrthogonalBivectorSplit(B) [k]Bivector` ~160 LOC** in `geometry/ga/bivector_split.go`. Decompose general grade-2 element into orthogonal blades `B = Σ_i β_i e_{a_i}∧e_{b_i}`. Needed for G12 when B is not a single blade (STA boost+rotation; CGA motor = 6-D bivector splitting into translation+rotation pair). Algorithm: spectral decomposition. **Soft blocker on `linalg.SchurDecomposition`** (absent). For Cl(3,0): 3-D bivector space, all blades — trivial. For Cl(1,3) STA: 6-D, splits into 3+3 via observer split (closed-form fallback).

**G15 — Rotor log `Log(R) Bivector` ~120 LOC** in `geometry/ga/log.go`. Closed form on each branch (Euclidean/Lorentzian/null) — same dispatch as G12. **`Slerp(R0,R1,t) = R0 · Exp(t · Log(R0̃·R1))`** generalises quaternion SLERP to any-dim rotor, motor, Lorentz boost — single-line replacement for `geometry.QuatSlerp`, `SE3Slerp` (absent), `LorentzSlerp` (absent). Tests: round-trip `Exp(Log(R)) = R` round-off.

**G16 — Cartan-Dieudonné decomposition ~110 LOC** in `geometry/ga/decompose.go`. Recovers unit vectors {m₁,…,m_k} such that `R = m_k…m_1`. Even k = rotor; odd k = reflection. For Cl(3,0)+ rotor, k=2.

### Tier 4 — Concrete signatures Cl(3,0) + Cl(1,3) STA + Cl(4,1) CGA (~860 LOC)

**G17 — Cl(3,0) Euclidean GA convenience surface ~180 LOC** in `geometry/ga/cl3.go`. `cl3` namespace exposes 8-D algebra as named blades `{1, e1, e2, e3, e23, e31, e12, e123}`; constructors `Vector(x,y,z)`, `Bivector(yz,zx,xy)`, `Pseudoscalar(s)`. **Quaternion bridge:** `QuaternionToMV(q)` and `MVToQuaternion(A)` exact byte-equality round-trip — one of the most pinnable identities in the package. `ApplyToVec(R,x)` byte-equals `QuatRotateVec(R_to_quat, x)`; `Cl3SLERP` byte-equals `QuatSlerp`.

**G18 — Cl(1,3) Spacetime Algebra (Hestenes 1966) ~280 LOC** in `geometry/ga/sta.go`. Signature (1,3,0) = (+,−,−,−) Minkowski (matches Wald/MTW conventions used elsewhere in reality). `EventVector(t,x,y,z)`. `BoostBivector(beta)` Lorentzian bivector with B²>0; `BoostRotor(beta) = Exp(BoostBivector(beta)/2)`. `RotateRotor(axis,θ)` spatial-rotation rotor. General Lorentz transformation = product of boost rotor + rotation rotor in single composition (no 4×4 matrix split). **Spacetime split:** F = E + I·B where I = e₀e₁e₂e₃ is spacetime pseudoscalar — recovers Cl(3,0) elements as STA-relative-to-observer e₀. Round-trip `RelativeFrame(F,observer) ↔ ToSTA(E,B,observer)` byte-equal. Refs: Hestenes 1966 *Space-Time Algebra*; Doran-Lasenby §5; Lasenby-Doran-Gull 1993 *FoP* 23:1295.

**G19 — Cl(4,1) Conformal Geometric Algebra (Hestenes 2001) ~400 LOC** in `geometry/ga/cga.go`. Signature (4,1,0) — adds null basis `e_∞ = e₋+e₊` (point at infinity, e_∞²=0) and `e_0 = (e₋−e₊)/2` (origin, e_0²=0, e_∞·e_0 = −1) to the standard 3-D Euclidean basis. **Six primitives:**
- Conformal point: `ConformalEmbedding(x) = x + ½x²·e_∞ + e_0` — null vector in 5-D ambient. `ConformalProject(P)` recovers Euclidean x.
- Sphere: any unit grade-1 vector with S²>0 IS a sphere; centre+radius extracted by inner with e_∞.
- Plane: unit grade-1 with `P·e_∞ = 0` and P²>0 (Euclidean unit normal + signed distance).
- Line: `P₁∧P₂∧e_∞` (grade-3).
- Circle: `P₁∧P₂∧P₃` (grade-3, dual to a line).
- Point pair: `P₁∧P₂` (grade-2, "edge" primitive).

**Versors / motors:** all four similarity transforms are versors — translation `T = exp(−t·e_∞/2)`, rotation `R = exp(B/2)`, dilation `D = exp(½ log(s)·(e_∞∧e_0))`, inversion-through-origin = unit pseudoscalar. **Motor** = SE(3) rigid motion = product of translation + rotation versor in one multiplication. NO 4×4 matrix. Tests: conformal embedding round-trip; sphere construction → circle as `S₁∧S₂` (sphere intersection); point-on-sphere `P·S = 0` ↔ `|x−c|² = r²`; motor application `MPM̃` byte-equals matrix `Rx+t`. Refs: Hestenes 2001 *Old Wine in New Bottles*; Doran-Lasenby §10; Dorst et al. §13–15.

### Tier 5 — Geometric calculus + Maxwell-as-∇F=J + Pauli/Dirac/spinors (~860 LOC)

**G20 — Geometric calculus ∇ = Σ e^i ∂_i ~250 LOC** in `geometry/ga/calculus.go`. `Nabla(F MultivectorField, x)` for any function `F:ℝⁿ→Cl(p,q)`. Operates grade-by-grade: on grade 0 (scalar f) `∇f = grad(f)`; on grade 1 (vector A) `∇A = (∇·A) + (∇∧A)` — divergence + curl in one operation (grade-0 part = divergence; grade-2 part dualises to standard ∇×A). **The unification:** grad / div / curl / Laplacian / d / δ / ★d★ — all classical or differential-form operators — collapse to specific grade projections of the single ∇ operator. ∇² on grade-0 is the standard Laplacian; on higher grades the Laplace-de Rham operator from 208-E9. Tests: `Grade(Nabla(A),0)` matches `calculus.Divergence(A,x)` to NumericalGradient precision (~6e-6); `Dual(Grade(Nabla(A),2))` matches `calculus.Curl(A,x)`. Depends `calculus.NumericalGradient`. Refs: Hestenes-Sobczyk §2; Doran-Lasenby §6.

**G21 — Maxwell as `∇F = J` (single equation) — pedagogical keystone ~120 LOC** in `geometry/ga/maxwell.go`. `MaxwellSTA(A,x)` returns `F = ∇∧A` (spacetime bivector encoding E and B). `MaxwellResidual(F,J,x)` returns `∇F − J` (zero by Maxwell). Two-line proof: `∇F = ⟨∇F⟩_1 + ⟨∇F⟩_3 = J` decomposes as `⟨∇F⟩_1 = ∇·F = J` (inhomogeneous: Gauss + Ampère) and `⟨∇F⟩_3 = ∇∧F = 0` (homogeneous: Gauss-magnetic + Faraday — automatic since `F = ∇∧A` and `∇∧∇∧A = 0`). **Pedagogical impact:** four PDEs collapse to one. Cross-validation: `MaxwellSTA` ↔ `em/maxwell.go` four-component ↔ 208-E18 differential-forms `dF=0, d★F=★J` — three independent code paths, one identity, byte-for-byte cross-pinned (R-MUTUAL-CROSS-VALIDATION 3/3). Depends G18, G20. Refs: Hestenes 1966; Doran-Lasenby §7; Wald §4.

**G22 — Pauli + Dirac representations + spinors-as-left-ideals ~250 LOC** in `geometry/ga/representations.go`. Two representation maps:
- `PauliRep(A) [2][2]complex128` for A ∈ Cl(3,0): 8 basis blades → {I, σₓ, σ_y, σ_z, iσ_z, …}. `PauliInverse` round-trips. Bridge to standard QM convention.
- `DiracRep(A) [4][4]complex128` for A ∈ Cl(1,3): 16 basis blades → standard γ-matrices (Dirac convention pinned; `DiracToWeyl`/`WeylToDirac` utilities ship). `DiracInverse` round-trips.

**Spinors as left ideals:** Pauli spinors live in left ideal generated by idempotent `(1+e₃)/2`. `SpinorFromColumn([2]complex128)` constructs even-multivector from 2-component complex column; `SpinorToColumn` extracts. Schrödinger / Dirac equations rewrite as multivector equations on left-ideal spinors, fully algebraic, no complex numbers needed. Tests: rep homomorphism `PauliRep(GP(A,B)) = PauliRep(A)·PauliRep(B)`; σ anti-commutation `{σ_i,σ_j} = 2δ_ij·I` recovered as `Inner(e_i,e_j) = δ_ij` in Cl(3,0); γ anti-commutation `{γ_μ,γ_ν} = 2η_μν·I` recovered as Cl(1,3) inner product; spinor ψ rotates as `ψ → R·ψ` (left action) versus vector `x → R·x·R̃` (sandwich) — fundamental pin distinguishing spinor double-cover from vector single-cover. Refs: Hestenes 1966 (Pauli); Lounesto 2001 *Clifford Algebras and Spinors* §8 (Dirac); Doran-Lasenby §8.

---

## 2. Composition graph

```
              G1 Multivector + G2 CayleyTable
                      │
                      ▼
              G3 GeometricProduct (KEYSTONE)
                      │
       ┌──────────────┼─────────────┬──────────────┐
       ▼              ▼             ▼              ▼
   G4 Grade      G5 Wedge       G6 Inner      G7 Reverse/Involution
                 (subsumes      (subsumes              │
                  208-E2)       contractions)          ▼
       │                                          G9 Dual
       ▼                                              │
   G8 Norm                                            │
       └──────────┬───────────────────────────────────┘
                  ▼
           G10 Outermorphism + G11 Regressive
                  ▼
           G12 RotorExp ─── G14 BivectorSplit (soft-blocker linalg.Schur)
                  │              │
                  ▼              ▼
           G13 VersorSandwich  G15 RotorLog ─── G16 Cartan-Dieudonné
                  │
       ┌──────────┼───────────────┐
       ▼          ▼               ▼
   G17 Cl(3,0) G18 Cl(1,3) STA  G19 Cl(4,1) CGA
   (quaternion (Lorentz boost   (point/sphere/plane/circle/line +
    bridge)     + spacetime      motor + similarity transforms)
                split)
                  │
                  ▼
           G20 Geometric calculus ∇  ── G21 Maxwell-as-∇F=J  ←── 208-E18 + em/ cross-validation
                  ▼
           G22 Pauli + Dirac + Spinor representations
```

Tier-1 (G1+G2+G3+G4) ships today and unlocks all other tiers. Quaternion is subsumed at G17.

---

## 3. Cheapest 1-day standalone PR

**G1+G2+G3+G4 = 580 LOC src + ~400 LOC test** delivers the first generic Clifford-algebra primitive in reality and saturates **R-MUTUAL-CROSS-VALIDATION 3/3** at the keystone via three independent paths:
- Path A: Cl(3,0)+ even-subalgebra `GeometricProduct` byte-equality with `QuatMul` over 100 random quaternions.
- Path B: Cl(2,0) `GeometricProduct` byte-equality with Pauli matrix product on 2×2 complex representation over 100 random multivectors.
- Path C: Cl(0,2) `GeometricProduct` byte-equality with quaternion product via (i,j,k)→(e₁,e₂,e₁e₂) isomorphism — independent of path A by signature.

Mirrors recent 6a55bb4 audio-onset 3-detector and 365368a copula×autodiff R-MUTUAL-CROSS-VALIDATION saturation pattern. Standalone PR: zero coupling beyond `linalg.MatMul` (test-only, Pauli matrix product). New sub-package `geometry/ga/`.

---

## 4. Architectural keystone

**G3 GeometricProduct** unifies G5 Wedge (= grade-(r+s) component), G6 Inner (= grade-|r−s|), G7 Reverse (= antiautomorphism), G9 Dual (= product with pseudoscalar inverse), G12 RotorExp (= bivector-power series collapsing to closed form), G13 VersorSandwich (= triple `RxR̃`), G20 GeometricCalculus (= ∇ as left-multiplication on multivector field). Once G3 ships, the entire Clifford / GA / CGA canon is a composition graph away. **Single most leveraged primitive in all of Block-C.**

**Quaternion bridge.** G3 on Cl(3,0)+ even subalgebra IS the Hamilton product. Reality has been shipping a special case of GA since v0.1 — generalising to arbitrary signature costs ~280 LOC and unlocks any-dim rotors, motors, Lorentz boosts, conformal transformations, spinors, Maxwell-as-one-equation. The Cl(3,0)+ ↔ ℍ bridge in G17 is the canonical numerical-precision contract between this slot and existing quaternion code.

---

## 5. Placement and import graph

NEW sub-package `geometry/ga/` (mirrors 194 `geometry/dec/` + 207 `geometry/diffgeo/` + 208 `geometry/extcalc/`). Four sibling sub-packages of `geometry/`: `dec/` discrete cochains, `diffgeo/` Christoffel/Riemann on charts, `extcalc/` continuous Λ^k, `ga/` Clifford multivectors over a fixed signature.

Cycle-free imports:
```
geometry/ga/  →  linalg/    (MatMul, MatVec; Eigen for G14 fallback; Pauli/Dirac in test)
              →  calculus/  (NumericalGradient for G20)
              →  geometry/  (quaternion bridge in G17 test-only)
              →  constants/ (ε₀, μ₀, c for G21 Maxwell)
              →  geometry/extcalc/  (208-E2 wedge cross-validation, test-only)
              →  geometry/diffgeo/  (207-D7 RiemannTensor for G16 curved-space test, test-only)
              →  em/        (test-only — cross-validate G21 against em/Maxwell)
```

Forward consumers: 208 (E2 Wedge → GA wrapper, single source of truth); 205 (SE(3) as CGA motor simplifies 205-L9); robotics-future (rigid motion as motor — singular advantage).

---

## 6. LOC roll-up

| Tier | Primitives | Source LOC | Test LOC | Cumulative src |
|------|-----------|------------|----------|----------------|
| 1 (1-day) | G1, G2, G3, G4 | 580 | 400 | 580 |
| 2 (algebra) | G5–G11 | 640 | 480 | 1220 |
| 3 (rotors) | G12–G16 | 720 | 540 | 1940 |
| 4 (signatures) | G17, G18, G19 | 860 | 660 | 2800 |
| 5 (calculus + physics) | G20, G21, G22 | 860 | 600 | 3660 |

Total: **~3650 LOC src + ~2680 LOC test**. (208-E2 wedge re-implementation as GA wrapper saves ~120 LOC there → net repo growth ~3530 LOC across 209+208.)

---

## 7. Cross-language pinning targets

- **G3 quaternion equivalence:** 100 random quaternions q1,q2 → `MVMul(QuatToMV(q1), QuatToMV(q2))` byte-equals `MVToQuat(QuatMul(q1,q2))` to 1e-15.
- **G3 Pauli equivalence:** 100 random Cl(2,0) MVs A,B → `PauliRep(GP(A,B))` byte-equal `PauliRep(A)·PauliRep(B)` (via `linalg.MatMul` on 2×2 complex) to 1e-14.
- **G3 Dirac equivalence:** 100 random Cl(1,3) MVs → `DiracRep(GP(A,B))` byte-equal Dirac γ-matrix product to 1e-13 (4×4 complex).
- **G3 associativity:** `GP(A,GP(B,C)) − GP(GP(A,B),C)` < 1e-13 for 100 random MVs per signature in {(3,0),(1,3),(4,1),(4,0),(0,4)}.
- **G5 wedge ↔ 208-E2 wedge:** byte-equality `Grade(GP(A,B), r+s) = 208ExtcalcWedge(A_kform, B_kform)` for grade-r A, grade-s B. Mandatory if 208-E2 re-home lands.
- **G12 rotor on Cl(3,0)+ ↔ quaternion exp:** `Exp(θ/2 · (e₂e₃ + α·e₃e₁))` byte-equal `QuatFromAxisAngle({1,α,0} normalised, θ)` after `MVToQuat`.
- **G13 versor sandwich on Cl(3,0)+ ↔ `QuatRotateVec`:** byte-equal round-off on 100 random rotor+vector pairs.
- **G19 CGA conformal embedding round-trip:** `ConformalProject(ConformalEmbedding(x)) = x` to 1e-15 for 100 random Euclidean points.
- **G19 CGA point-on-sphere:** `Inner(P_conformal, S_conformal) = 0` to 1e-14.
- **G19 CGA motor application:** `Apply(motor(R,t), Embed(x)) = Embed(R·x + t)` byte-equal for 100 random (R,t,x) triples.
- **G21 Maxwell-as-one-equation:** `Grade(MaxwellResidual(F,J), 1)` and `Grade(...,3)` byte-equal `em/` four-component formulation to 1e-12 for 100 random (E,B,J,ρ) configurations.

---

## 8. Out-of-scope deferrals

Conformal Killing vectors via CGA (defer to 207-D14). Octonions / Cl(0,7) split-octonions (v2, no clear application within reality scope). Twistors (Penrose) — v2. Higher-dimensional CGA (n>3 ambient Euclidean) — straightforward generalisation of G19 but no immediate consumer; v2. GA-based numerical optimisation (Lasenby et al.) — handled in 206. Symbolic GA-as-code-generator (à la Gaalop) — out of scope, reality is numerical not symbolic.

---

## 9. Precision hazards

- **G2 Cayley sign convention:** swap-count for canonical-ordering depends on strict left-to-right sort; sign error cascades through every product. Mandatory invariant check on construction.
- **G3 sparse/dense numerical equivalence:** `GPSparse` skips zero coefficients but must produce byte-equal output to dense for all-finite inputs (no 1e-17 noise from skipped paths).
- **G6 inner-product convention:** Hestenes vs. Dorst-Fontijne-Mann differ on scalar × multivector. Pin Dorst in docstring; provide both `Inner` (Hestenes) and `Scalar` (Dorst) aliases.
- **G7 reverse vs Clifford conjugation:** sign error on grade-2 is `(−1)^{2·1/2}=−1` for both, but on grade-3 differ: Reverse `(−1)^3=−1`, Conjugate `(−1)^{3·2/2+3}=+1`. Test on Cl(3,0) pseudoscalar.
- **G9 dual sign:** `Dual(Dual(A)) = sign·A` where sign depends on signature parity `s = q − p mod 4`. Pin in docstring: `s=0:+1; s=1:±I; s=2:−1; s=3:∓I`.
- **G12 rotor branch dispatch:** sign of B² distinguishes Euclidean / Lorentzian / null; near-zero B² requires special-case to avoid divide-by-zero. Use ε=1e-14 threshold + Taylor fallback.
- **G18 STA signature:** pin (+,−,−,−) matching Wald 1984 / MTW 1973 conventions used elsewhere in reality (`em/`, `physics/`). Sign flips on Lorentz boost generator B²>0.
- **G19 CGA null basis:** `e_∞·e_0 = −1` (Dorst convention) versus `+1` (Hestenes "stereographic"). Pin Dorst — matches Garamon, ganja.js, Versor.
- **G19 conformal embedding factor:** `½` on `x²·e_∞` term is convention-dependent. Pin Hestenes 2001 `P = e_0 + x + ½x²·e_∞`.
- **G22 Dirac convention:** Dirac (block-diagonal γ⁰), Weyl (block-off-diagonal), Majorana (real). Pin Dirac; ship `DiracToWeyl` / `WeylToDirac` conversion utilities.

---

## 10. Cross-link summary

- **194-em-geometry (DEC):** sibling sub-package, orthogonal axis (discrete cochains versus continuous multivectors). G19-CGA points on a CGA mesh formally distinct from 194 simplicial vertices; cross-link in test only.
- **205-new-lie-groups:** SE(3) as CGA motor (G19) significantly simplifies 205-L9 SE(3)-exp; recommend coordination — 205 ships matrix-form for compatibility, this slot ships motor-form for new code; both round-trip byte-equal.
- **207-new-diff-geo:** 207-D23 curvature 2-form `Ω = dω + ω∧ω` is Lie-algebra-valued 2-form which in GA terms is bivector-valued bivector — G3+G14 simplifies the structure equation to ~50 LOC versus 207-D23's 140 LOC.
- **208-new-exterior-calculus:** 208-E2 Wedge SHOULD be re-implemented as thin wrapper over G3+G4 — single source of truth across the repo. Net 208 LOC reduction ~120.
- **156-topology-persistent:** G19-CGA primitives provide coordinate-free embedding for persistent-homology Vietoris-Rips complexes — natural cross-link.
- **098-linalg-sota:** mentions Clifford for matrix decomposition; this slot supersedes by demonstrating that for SO(3)/SE(3)/SO(p,q) the relevant decompositions are rotor decompositions (G16 Cartan-Dieudonné), no matrix Schur needed.
- **107-orbital-missing / 112-physics-missing:** both want spinor formulations of spinning rigid bodies + classical electron — G22 ships those as Cl(3,0) / Cl(1,3) representations.
- **159-em-signal:** G21 Maxwell-as-∇F=J is the singular-equation form 159 mentions in passing. Co-ship recommended.

---

## 11. Verdict

**SHIP Tier-1 (G1+G2+G3+G4) ~580 LOC** as 1-sprint standalone PR — saturates R-MUTUAL-CROSS-VALIDATION 3/3 against quaternion (Cl(3,0)+) and Pauli (Cl(2,0) and Cl(0,2)). Single most-leveraged 1-day project in Block-C — once G3 ships, the entire Clifford / GA / CGA / STA canon composes over it.

**SHIP Tier-2 (G5–G11) ~640 LOC** as 2nd sprint — algebra surface. Re-home 208-E2 Wedge as wrapper.

**SHIP Tier-3 (G12–G16) ~720 LOC** as 3rd sprint — rotors, sandwich, Cartan-Dieudonné. SLERP generalises to any-dim rotor.

**SHIP Tier-4 (G17–G19) ~860 LOC** as 4th sprint — Cl(3,0) Euclidean (quaternion bridge), Cl(1,3) STA (Lorentz boosts as bivector exp), Cl(4,1) CGA (point/sphere/plane/circle/line/motor). G19 CGA is the singular flagship — robotics + computer-vision + physics-pedagogy unified primitive.

**SHIP Tier-5 (G20–G22) ~860 LOC** as 5th sprint — geometric calculus + Maxwell-as-∇F=J + Pauli/Dirac/spinor representations. **G21 is the singular pedagogical demonstration** — four Maxwell equations collapse to one with byte-for-byte cross-validation against `em/` and 208-E18.

**DEFER**: octonions, twistors, GA-based optimisation, symbolic GA code-generation. All v2.

**Single highest-leverage 1-day project:** G1+G2+G3+G4 ~580 LOC — first generic Clifford algebra in reality, saturates 3/3 cross-validation via quaternion + Pauli + Cl(0,2)→quaternion paths.

**Singular cutting-edge competitive moat:** G19 CGA + G21 Maxwell-as-∇F=J ~520 LOC — no zero-dependency Go library ships either surface. Closest cross-language references all dependency-heavy or proprietary (clifford.py NumPy, ganja.js browser-only, Gaalop Java code-generator, Garamon C++ template-heavy, Versor C++ wrong-convention, GAlgebra Mathematica). Reality would be the only Go shop AND only language ecosystem at all with byte-for-byte cross-language golden-file contract on CGA primitive operations.

Total v0: ~3650 LOC src + 2680 LOC test across Tiers 1–5. Re-homes 208-E2 wedge as GA wrapper saving ~120 LOC there. Three soft blockers (linalg.SchurDecomposition for G14 high-dim splitting, em/ for G21 cross-validation, 207-D7 RiemannTensor for G16 curved test), all unblocked-on-low-dim-fallback.

End of report. 22 primitives, ~3650 LOC src, ~2680 LOC test, one new sub-package `geometry/ga/`, zero cycles. Keystone G3 GeometricProduct collapses Hamilton + Pauli + Dirac + wedge + dot + Maxwell-grad-div-curl into a single dispatch table. Quaternion bridge demonstrates reality has been shipping GA on Cl(3,0)+ since v0.1 — generalising to arbitrary signature unlocks STA + CGA + spinors at <100 LOC each. Singular reality moat: G19 CGA + G21 Maxwell-as-∇F=J ~520 LOC — no other Go library, no other language with byte-for-byte cross-language golden-file contract, has this surface.

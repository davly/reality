# 246 | new-discrete-exterior — Discrete Exterior Calculus on simplicial / cubical complexes

**Summary line 1.** reality v0.10.0 ships ZERO Discrete Exterior Calculus surface — repo-wide grep on `cochain|coboundary|incidence|primal.dual|hodge.star|whitney.form|cubical.complex|cotangent.lapl|simplicial.lapl|hirani|pinkall|polthier|FEEC|de.rham.complex|discrete.exterior|laplace.beltrami|laplace.de.rham|harmonic.cochain|hodge.decomposition|discrete.helmholtz|kelvin.circulation|catmull.clark|loop.subdivision` against `*.go` returns ZERO callable matches; the only adjacent substrate is `topology/persistent/vr.go` which builds Vietoris-Rips *abstract* simplicial complexes for persistent-homology barcode reduction (F_2 column-reduction over `[]Simplex` = `[]int`, dimension up to 1 in Phase A, n<=50) — it has the simplicial-complex *combinatorics* but ZERO geometric-form / cochain / d / ★ / Δ surface, and `geometry/polygon.go.TriangleArea2D` (signed-area) which is structurally the only k-form-evaluator the repo ships (the volume 2-form on a 2-simplex). NO `Mesh`, NO `SimplicialComplex` with embedded coordinates, NO `CubicalComplex`, NO `Cochain` type, NO `BoundaryOperator` ∂_k / `Coboundary` d^k = ∂_{k+1}^T, NO incidence matrices, NO `HodgeStar`_k diagonal Hirani primal/dual, NO Galerkin-Hodge mass matrix, NO cotangent-Laplacian Pinkall-Polthier-1993, NO Whitney-form-W_σ basis, NO discrete-Stokes witness, NO discrete-Helmholtz-Hodge decomposition, NO discrete-Laplace-de-Rham Δ_k = δd+dδ, NO discrete-cohomology kernel/image rank, NO Betti-from-DEC cross-check vs `topology/persistent`, NO Yee-as-DEC equivalence, NO discrete-Maxwell on triangle/tet meshes, NO DEC-Galerkin Navier-Stokes preserving Kelvin's circulation theorem (Mullen 2009), NO Catmull-Clark / Loop subdivision, NO polygonal-mesh DEC (de Goes-Crane 2016), NO discrete-Yang-Mills (Christiansen 2012). The combinatorial-topology side is there in barcode.go (column-reduction = ∂_k over F_2) but it cannot be reused as-is because (a) it works over F_2 not ℝ, (b) it has no embedded coordinates so no metric / no ★, (c) `Filtration` is a sorted `(simplex, time)` list, not a per-dimension cell array indexed for d/★ stencils. **Two existing reviews already partial-cover this axis**: 194 (synergy-em-geometry, D0–D17 ~2,050 LOC) scoped DEC primarily as the *Maxwell-on-meshes* enabler (Yee-FDTD generalisation + Hodge-decomposition + magnetostatics BVP); 207 (new-diff-geo, Tier-3 D15–D20 ~1,180 LOC) scoped *continuous Λ^k forms* with discrete d/★ as a one-line cross-validation against 194. **This slot 246 elevates DEC to a first-class axis** with a 7-tier 30-primitive plan that subsumes 194's mesh+d+★ kernel and EXTENDS into FEEC / cotangent-Laplacian / cubical complexes / Catmull-Clark / discrete-Helmholtz / circulation-preserving-NS / discrete-cohomology-Betti / spectral-DEC — none of which 194 or 207 address.

**Summary line 2.** Thirty primitives **X1–X30** totalling ~3,650 LOC across new sub-package `geometry/dec/` (mirrors precedent of 194 D0–D17, 207 Tier-3 D15–D20, 208 E1–E20 — all three sibling sub-packages of `geometry/`: `dec/` = discrete cochain DEC on meshes [this slot, the *load-bearing* one — supersedes 194], `diffgeo/` = continuous tensors-on-charts [207], `extcalc/` = continuous Λ^k forms [208]; the dual-version Whitney-de Rham theorem witnesses 194-D11/this-X11 ↔ 208-E5 to round-off and lifts the cross-package R-MUTUAL-CROSS-VALIDATION 3/3 contract from synergy to native-package). **Versus 194 D0–D17 (~2,050 LOC):** strict superset — every 194 primitive maps to one of X1–X12 here, plus this slot adds X13–X30 (cubical complex, Whitney-FEEC mass matrix, cotangent-Laplacian-Pinkall-Polthier-1993 as the keystone smoothing operator the entire SGP-discrete-geometry pipeline runs on, discrete Helmholtz-Hodge harmonic basis, discrete-cohomology Betti cross-link to 156, Catmull-Clark / Loop subdivision, polygonal-mesh DEC de Goes-Crane-2016, Mullen-2009 circulation-preserving Navier-Stokes, Christiansen-2012 discrete-Yang-Mills, spectral-DEC eigenfunctions / heat-kernel signature, ODE-on-manifold variational integrators Stern-2015 cross-link to 204, integrable-systems / discrete-Hashimoto-flow). Recommend: **single-source ownership of the discrete cochain side concentrates here**; 194 narrows to "EM-on-meshes use case" of this substrate (~370 LOC consumer-side: D6 MaxwellMesh + D7 magnetostatics + D13 gauge-fix) and re-imports `geometry/dec/`. **Versus 207 D15–D20:** complementary not duplicative — 207 ships the continuous-coordinate forms (`func([]float64) [][]float64` evaluators of g_ij, d via finite differences on coefficient functions); this slot ships the discrete cochain arrays + signed-incidence-matrix-d + diagonal-Hodge / Galerkin-Hodge ★. The two are dual via Whitney's de Rham theorem (de Rham 1955 §IV) — 208's E14 Hodge decomposition explicitly bridges via this slot's X11 Whitney form. **Versus 208 E1–E20:** complementary continuous Λ^k counterpart; 208's E12 Stokes witness validates against this slot's X8 discrete Stokes via an O(h²)-converging Whitney-projection ladder on uniform meshes. **Versus 156 topology-persistent (vr.go + barcode.go ~700 LOC):** orthogonal-but-bridgeable — 156 ships Vietoris-Rips F_2 column-reduction; this slot ships ℝ-coefficient cochain Laplacians on the SAME complex; the bridge primitive X23 `BettiViaSpectralDEC(complex)` returns dim ker(Δ_k) = β_k = the dimension where 156's `BarcodesToInfinityCount(diagram, k)` agrees as filtration → ∞ (Hodge theorem on finite cell complex, Eckmann 1944). **Versus 247 mortar/multimesh (read-ahead from MASTER_PLAN):** this slot ships single-mesh DEC; 247 will ship the *cross-mesh transfer* operators (mortar projection between non-conforming meshes) which compose on top of X11 Whitney forms here. Single highest-leverage 1-day project: **X1+X2+X3+X4+X8 ~340 LOC = SimplicialComplex2D + d_0/d_1 + diagonal-Hodge + Stokes witness** saturates R-MUTUAL-CROSS-VALIDATION 3/3 (continuous Stokes via `calculus.SimpsonsRule` × discrete Stokes via X4 d_1 × `dd=0` to round-off). Architectural keystone: **X12 Cotangent-Laplacian Pinkall-Polthier-1993 ~140 LOC** — the single most-cited primitive in computational discrete differential geometry (Crane et al. 2013 SGP "DDG: An Applied Introduction" makes it the entry point of the entire discipline); used by every Pistachio/Nimbus mesh-smoothing / mesh-parameterisation / shape-correspondence / spectral-segmentation primitive that ever ships. Cutting-edge moat: **X19+X20+X21 = Mullen-2009 Kelvin-circulation-preserving DEC-NS + Stern-2015 variational-integrator-DEC-Maxwell + Christiansen-2012 discrete-Yang-Mills ~480 LOC** — the *structure-preserving* DEC frontier (no zero-dependency Go library ships any of these; closest references are libigl C++ for cotangent-Laplacian only, gudhi for combinatorial-Hodge only, no production code for Mullen-NS/Stern-EM/Christiansen-YM). Single-pedagogical-pin: **X12 cotangent-Laplacian eigenmodes on equilateral-triangle Dirichlet-disk = analytic Bessel zeros (j_{n,m}/R)² to O(h²) — the canonical pin for discrete-spectral-correctness** (Reuter et al. 2009 "DDG of solids", citing Pinkall-Polthier 1993 Thm 4.1).

---

## 0. State at HEAD (2026-05-09, v0.10.0) — verified by direct read

### `geometry/` (656 LOC, 4 files)

- `quaternion.go` (230 LOC): SO(3)/SU(2) algebra. ZERO mesh awareness.
- `sdf.go` (210 LOC): implicit-surface SDF evaluators. NOT a triangulated representation — level-set evaluation only.
- `curves.go` (75 LOC): Bezier/CatmullRom 1-parameter curves. NOT a discrete 1-cochain (no edge graph; curve points have continuous parameter).
- `polygon.go` (141 LOC): `TriangleArea2D` (the signed area, structurally the value of the volume 2-form on a 2-simplex but unlabelled), `PointInTriangle2D`, `ConvexHull2D` (Graham scan).

### `topology/persistent/` (~700 LOC, 5 files)

- `vr.go`: `Simplex = []int` (sorted vertex-index tuple), `Filtration{Simplices []Simplex; Times []float64}`. Builds VR-complex from point-cloud + max-radius. **HAS the simplicial combinatorics** but no embedded geometry.
- `barcode.go`: F_2 column-reduction `∂_k` matrix → barcode bars. **HAS a boundary operator** but only over F_2 (XOR not arithmetic), so no Hodge ★, no Δ-eigenvalue, no harmonic-cochain.
- `bottleneck.go`: persistence-diagram L^∞ matching.

### Cross-package substrate audit

- `linalg/`: `LU/QR/Cholesky/QRAlgorithm` symmetric eigendecomp (dense). NO `SparseMatrix`, NO `SparseSolve`, NO `SparseEigen` — flagged 097-T1, blocks X10/X11/X14/X22-large.
- `calculus/`: `SimpsonsRule`, `GaussLegendre` (5-point cap), `NumericalGradient`, `NumericalDerivative`. Continuous-side counterparts for Stokes / Whitney-quadrature cross-validation.
- `chaos/`: `RK4Step` for ODEs. Substrate for X18 ODE-on-manifold variational integrators.
- `signal/`: `FFT/IFFT`. Substrate for X28 spectral-DEC heat-kernel-signature when mesh has cyclic symmetry (very limited applicability).
- `constants/`: `VacuumPermittivity`, `VacuumPermeability`, `SpeedOfLight` for Maxwell-DEC closure.
- `geometry/polygon.go.TriangleArea2D`: signed area = simplest 2-form evaluator. Ships as inputs to X3 ★_2 (1/area).

### Repo-wide grep audit (verified)

```
$ grep -ri 'SimplicialComplex\|CubicalComplex\|Cochain\|Coboundary\|HodgeStar\|WhitneyForm\|cotangent.lapl\|laplace.de.rham\|laplace.beltrami\|hodge.decomposition\|discrete.helmholtz\|kelvin.circulation\|catmull.clark\|loop.subdivision\|FEEC\|hirani\|pinkall\|polthier\|de.goes' --include='*.go' | wc -l
0
```

Confirms reality has **zero** DEC / FEEC / DDG-pipeline surface. The only callable mesh-like-thing is barcode.go's F_2 ∂_k which doesn't generalise to ℝ.

---

## 1. The thirty primitives

Numbered X1–X30. For each: **(a) what reality ships**, **(b) what to add**, **(c) connective-tissue LOC**, **(d) blocker**.

### Tier 1 — Mesh + d + ★ keystone, ships unblocked (~480 LOC)

#### X1 — `SimplicialComplex` types (geometric, embedded)

(a) **Ships:** `topology/persistent/vr.go.Filtration` ships *abstract* simplicial complex (vertices as ints, no coords, only F_2-coefficient column reduction). NOT reusable as-is because no embedded coordinates and Filtration's sort-by-time-then-dim layout is not the per-dimension array indexing d/★ stencils need.
(b) **Add:** `SimplicialComplex2D{Verts [][2]float64; Edges [][2]int; Tris [][3]int}` (and `3D` with Faces, Tets). Per-dimension arrays so `len(Edges) = n_1`, `len(Tris) = n_2`. Canonical edge orientation `(min(i,j), max(i,j))` invariant — single load-bearing decision; inconsistent orientation silently gives `dd≠0`. Helpers: `BuildEdgesFromTris` (ensure unique edges + orientation), `EulerCharacteristic = V−E+F`, `IsManifold` (every edge has 1 or 2 incident triangles, every vertex link is a connected path or cycle). Bridge: `FromFiltration(filt, verts)` adapter consumes `topology/persistent/vr.go.Filtration` plus separate vertex coordinates → `SimplicialComplex2D` (cross-link 156).
(c) **LOC:** ~140 in `geometry/dec/complex.go`.
(d) **No blocker.** Cross-link 156-topology-persistent for Filtration adapter.

#### X2 — `Cochain[k]` type + arithmetic

(a) **Ships:** nothing.
(b) **Add:** `Cochain[k] = []float64` of length `n_k`. Plain alias with helper functions `CochainAdd / Scale / Norm / InnerProduct(c1, c2, hodge ★_k)` where the inner product `⟨α, β⟩ = α^T ★_k β`. Iterators by k-simplex.
(c) **LOC:** ~40 in `geometry/dec/cochain.go`.
(d) **No blocker.**

#### X3 — Discrete Hodge star ★_k (diagonal Hirani primal/dual)

(a) **Ships:** nothing.
(b) **Add:** `HodgeStar0(complex) []float64` = vertex-area (1/3 sum of incident triangle areas); `HodgeStar1(complex) []float64` = `|dual_edge|/|primal_edge|` from circumcenters; `HodgeStar2(complex) []float64` = `1/|tri t|`. For 3-D: ★_0, ★_1, ★_2, ★_3 from primal tet → dual vertex barycenters. Diagonal storage: only nonzero entries are `★_k[i,i]` so represented as `[]float64`. Materials enter as scalar multipliers: `★_k^ε[e] = ε[e] · ★_k[e]`. Vacuum case uses `constants.VacuumPermittivity`.
(c) **LOC:** ~110 in `geometry/dec/hodge.go`. Includes circumcenter helper (25 LOC closed form).
(d) **No blocker.** Subsumes 194-D2.

#### X4 — Discrete exterior derivative d_k as signed incidence matrix

(a) **Ships:** F_2 column-reduction in `topology/persistent/barcode.go` (boundary not coboundary, F_2 not ℝ — incompatible for d/★/Δ).
(b) **Add:** `D0(complex) → SparseInt8` (sparse signed-incidence matrix `n_1 × n_0` with entries ±1). `D1(complex) → SparseInt8` (`n_2 × n_1` with edge-orientation signs). `D2(complex)` for tet meshes. The discrete coboundary as transpose of boundary. Stencil: `(d_0 f)[edge i,j] = f[j] − f[i]` (with canonical orientation `i<j`); `(d_1 e)[tri i,j,k] = ε_ij·e[ij] + ε_jk·e[jk] + ε_ki·e[ki]` (signs from canonical-orientation lookup).
(c) **LOC:** ~120 in `geometry/dec/derivative.go`. Subsumes 194-D1 (which lacks the sparse signed-incidence-matrix abstraction).
(d) **No blocker.** Single load-bearing dependency on X1's canonical edge orientation.

#### X8 — `dd=0` + Stokes witness fixtures

(a) **Ships:** `calculus.SimpsonsRule` for the continuous side.
(b) **Add:** `DDWitness(complex) error` = `‖d_1 ∘ d_0(f)‖_∞ < 1e-26` for 100 random 0-cochains f (machine zero, exact integer-coefficient algebra). `DiscreteStokes2D(complex, e) error` = `Σ_{tri} (d_1 e)[t] − Σ_{∂edge} ε·e[edge] = 0`. R-MUTUAL-CROSS-VALIDATION 3/3 ladder: continuous Stokes via Simpson on parametric ∂Ω × discrete Stokes via X4 d_1 × identity dd=0. Single-flagship correctness pin for the entire DEC chapter; replaces ~6 ad-hoc tests. Mirrors recent commits 6a55bb4 audio-onset-3-detector and 365368a copula×autodiff R-MUTUAL-CROSS-VALIDATION saturation pattern.
(c) **LOC:** ~70 in `geometry/dec/witness.go`. Subsumes 194-D5.
(d) **No blocker.**

### Tier 2 — Codifferential + discrete Laplacian + Pinkall-Polthier (~430 LOC)

#### X5 — Codifferential δ_k = ★^{-1} d^T ★

(a) **Ships:** nothing.
(b) **Add:** `Codifferential(complex, omega Cochain) Cochain` via `δ = ★_{k-1}^{-1} d_{k-1}^T ★_k` (sign convention Lee-2018 so eigenvalues nonneg). The L²-adjoint of d. Composes on X3+X4.
(c) **LOC:** ~70 in `geometry/dec/codifferential.go`. Subsumes 194-D3.
(d) No blocker.

#### X6 — Discrete Laplace-de Rham Δ_k = δd + dδ

(a) **Ships:** nothing.
(b) **Add:** `Laplacian0(complex) SparseMatrix` = `δ_1 d_0` + `d_-1 δ_0` (the latter is zero since there are no (-1)-cochains in standard convention). `Laplacian1(complex)` = `d_0 δ_1 + δ_2 d_1`. **For 0-forms on triangle meshes, Δ_0 = the discrete Laplace-Beltrami operator** — used by every Crane et al. 2013 SGP "DDG: An Applied Introduction" downstream pipeline. Eigenvalue benchmark: equilateral-triangle Dirichlet disk → first 5 eigenvalues match analytic Bessel zeros `(j_{n,m}/R)²` to O(h²).
(c) **LOC:** ~80 in `geometry/dec/laplacian.go`. Subsumes 194-D4 (dense fallback).
(d) Sparse storage blocked on 097-T1 SparseMatrix.

#### X12 — Cotangent-Laplacian (Pinkall-Polthier 1993) — **architectural keystone**

(a) **Ships:** nothing.
(b) **Add:** `CotanLaplacian(complex) SparseMatrix` with `L_{ij} = ½(cot α_{ij} + cot β_{ij})` (the canonical discrete Laplace-Beltrami on triangle meshes, where α_{ij}, β_{ij} are the two angles opposite edge ij in its two incident triangles). Diagonal `L_{ii} = -Σ_{j} L_{ij}`. **The most-cited primitive in computational discrete differential geometry**: Crane-de Goes-Desbrun-Schröder 2013 SGP course "DDG: An Applied Introduction" makes this the entry point; used by every Pistachio/Nimbus mesh-smoothing (Desbrun-Meyer-Schröder-Barr 1999), mesh-parameterisation (Lévy 2002 LSCM), shape-correspondence, spectral-segmentation, heat-method-geodesics (Crane-Weischedel-Wardetzky 2013 SIGGRAPH) primitive that ever ships. **Equivalent** (via Arnold-Falk-Winther 2006 FEEC Thm 5.1) to `★_0^{-1} d_0^T ★_1 d_0` from X3+X4 with Galerkin-Hodge ★_1 from Whitney 1-forms (X11) — i.e., it IS a special case of X6, but with the Galerkin-mass-matrix Hodge instead of the diagonal Hirani Hodge. Pin to `cot(α) + cot(β)` analytic identity.

Pinkall-Polthier 1993 §3 hazard: **cotangent weights become negative on obtuse triangles** → resulting Laplacian loses M-matrix property (positive-off-diagonals violated). The Pinkall-Polthier 1993 Thm 4.1 still gives convergence; downstream solvers must not assume positive-off-diagonals. Footgun-docstring required.
(c) **LOC:** ~140 in `geometry/dec/cotanlapl.go`.
(d) No blocker for dense (n≤10³); sparse blocked on 097.

#### X7 — Discrete vector calculus identities (∇²×∇× preservation, dd=0 witness for d_2)

(a) **Ships:** nothing.
(b) **Add:** `CurlOfGradientIs0Witness`, `DivergenceOfCurlIs0Witness`, `LaplacianOfScalarIs_DivOfGrad`, `LaplacianOfCurlEqualsCurlOfLaplacian` (commutator on harmonic 1-forms) — all four follow from `dd=0` and the duality `δ = ★^{-1}d^T★`, so they validate at machine precision. Mirror commits 6a55bb4/365368a R-MUTUAL-CROSS-VALIDATION saturation pattern.
(c) **LOC:** ~60 in `geometry/dec/identities.go`.
(d) Depends X4+X5.

#### X10 — Discrete Helmholtz-Hodge decomposition

(a) **Ships:** nothing.
(b) **Add:** `HelmholtzHodge(complex, omega Cochain) (df, deltaBeta, h Cochain)` via two Poisson solves: `Δ_0 f = δ_1 ω`, `Δ_2 β = d_1 ω`, then `h = ω − d_0 f − δ_2 β`. The crown jewel of DEC. Yields cohomology dimension dim(harmonic) = first Betti number directly. Tests: torus mesh genus-1 → `dim h = 2` (β_1 = 2g for orientable surface of genus g); sphere → `dim h = 0` (simply connected). Subsumes 194-D8.
(c) **LOC:** ~80 in `geometry/dec/helmholtz.go`.
(d) Sparse blocked on 097; dense ships for n≤10³.

### Tier 3 — Whitney-FEEC + Galerkin-Hodge + cubical complex (~610 LOC)

#### X11 — Whitney forms W_σ (Whitney 1957, Bossavit 1988)

(a) **Ships:** nothing.
(b) **Add:** `WhitneyForm(complex, k, simplex_idx) func(barycentric []float64) []float64` returning the Whitney basis form W_σ for a k-simplex σ. For k=0 vertex i: `W_i = λ_i` (barycentric). For k=1 edge ij: `W_{ij} = λ_i ∇λ_j − λ_j ∇λ_i`. For k=2 tri ijk: `W_{ijk} = 2(λ_i ∇λ_j ∧ ∇λ_k + cyclic)`. Tests: `∫_e W_e = 1`, `∫_{e'} W_e = 0` for e' ≠ e (Whitney 1957 §III.2 partition-of-unity). Underpins **Finite Element Exterior Calculus (FEEC) Arnold-Falk-Winther 2006-2018** — the cohomology-stable mixed-FE framework that is the modern functional-analysis foundation of DEC.
(c) **LOC:** ~120 in `geometry/dec/whitney.go`. Subsumes 194-D11.
(d) Cross-link 174 (barycentric helper).

#### X9 — Discrete wedge product (Whitney-interpolated)

(a) **Ships:** nothing.
(b) **Add:** `WedgeProduct(omega, eta Cochain, k_omega, k_eta) Cochain`. For 1-form ∧ 1-form on triangle ijk: `(α∧β)[ijk] = (1/6) Σ_perm sgn(perm)·α[edge]·β[edge']`. Required for Poynting vector S = E ∧ H (a 2-form), Yang-Mills A∧A (X25), Chern-character integrand tr(F∧F).
(c) **LOC:** ~70 in `geometry/dec/wedge.go`. Subsumes 194-D9.
(d) Depends X4.

#### X14 — Galerkin-Hodge mass matrix M_k

(a) **Ships:** nothing.
(b) **Add:** `GalerkinHodge_k(complex) SparseMatrix` = `M_{σσ'} = ⟨W_σ, W_{σ'}⟩_{L²}` (mass matrix in Whitney-form basis). The non-diagonal **higher-fidelity** Hodge star alternative to X3's diagonal Hirani Hodge — exact O(h²) on non-Delaunay meshes where Hirani degrades to O(h^0). Hiptmair 2002 Acta Numerica §3.
(c) **LOC:** ~100 in `geometry/dec/galerkinhodge.go`.
(d) Sparse blocked on 097.

#### X13 — `CubicalComplex` types (Kovalevsky 1989, Kaczynski-Mischaikow-Mrozek 2004)

(a) **Ships:** nothing.
(b) **Add:** `CubicalComplex{Verts [][n]float64; Edges [][2]int; Quads [][4]int; Cubes [][8]int}` for arbitrary-d (typically 2 or 3). Discrete d_k is a signed-incidence matrix again but with the natural cubic-cell orientation (every k-cube has 2k oriented (k−1)-faces). **Yee FDTD 159-W4 IS DEC on a cubical complex** — Bossavit 2001 Thm 4.2 makes this rigorous. Volumetric image-data lives natively as cubical complexes (Kaczynski-Mischaikow-Mrozek 2004 *Computational Homology* §2 — voxel grids), so X13 is the *correct* substrate for any image-/volume-DEC application (e.g., MRI segmentation via discrete Laplacian on voxel cubical complex).
(c) **LOC:** ~160 in `geometry/dec/cubical.go`.
(d) No blocker.

#### X15 — Discrete Stokes' theorem (n-form, signed-boundary lift)

(a) **Ships:** nothing in the boundary-lift form (X8 is the simple-witness form).
(b) **Add:** `IntegrateForm(complex, omega Cochain, region) float64` = sum-cells with interior orientation; `IntegrateBoundaryForm(complex, omega Cochain, region) float64` = sum oriented boundary. `StokesIdentity` = the two are equal for `omega = d eta`. Pins divergence theorem (n-form), Green's theorem (2-form on ℝ²), classical Stokes (curl on ℝ³) all as one identity.
(c) **LOC:** ~80 in `geometry/dec/stokes.go`.
(d) Depends X4.

#### X16 — Pull-back / push-forward between meshes (mortar precursor)

(a) **Ships:** nothing.
(b) **Add:** `Pullback(meshA, meshB, phi_per_vertex, omega_B) omega_A` for piecewise-linear phi: A→B. Subsumes 194-D12. Cross-link 247 (mortar projection between non-conforming meshes will COMPOSE ON THIS).
(c) **LOC:** ~80 in `geometry/dec/pullback.go`.
(d) No blocker; cross-link 247.

### Tier 4 — Cohomology + Betti + spectral DEC (~470 LOC)

#### X17 — Discrete cohomology rank ker(d_k) / im(d_{k-1})

(a) **Ships:** nothing in ℝ-coefficient form. `topology/persistent/barcode.go` ships F_2 version.
(b) **Add:** `Cohomology_k(complex) (rank int, basis []Cochain)` via `dim ker(d_k) − rank(d_{k-1})` from sparse SVD or via `dim ker(Δ_k)` (Hodge theorem on finite cell complex, Eckmann 1944 — same answer). For simplicial: triangle-fan disk → β_0=1, β_1=0; annulus → β_0=1, β_1=1; torus → β_0=1, β_1=2, β_2=1; double-torus → β_1=4. Integer-equality contract with 156 persistent-homology dimension as filtration → ∞.
(c) **LOC:** ~110 in `geometry/dec/cohomology.go`.
(d) Sub-blocker on `linalg.SparseEigen` (097-T1) for direct sparse eigendecomp; dense ships for n≤300.

#### X23 — Betti numbers β_k via spectral DEC (Hodge-theorem)

(a) **Ships:** F_2 Betti via barcode count of infinite bars in `topology/persistent/barcode.go`.
(b) **Add:** `BettiViaSpectralDEC(complex) []int` returns `β_k = dim ker(Δ_k)`, via X12 cotangent-Laplacian eigenmode counting at λ=0 (algorithmically: count eigenvalues with `|λ| < tol`). Eckmann 1944 / Friedman 1995. Cross-validation contract: `BettiViaSpectralDEC` (this slot) == `BettiFromBarcode` (156-topology-persistent at filtration→∞) byte-for-byte for any orientable manifold complex. Single tightest cross-package golden contract: integer answer, no tolerance.
(c) **LOC:** ~70 in `geometry/dec/betti.go`. Direct cross-link to 156.
(d) Depends X12.

#### X28 — Spectral DEC: heat-kernel signature, mesh segmentation eigenmodes

(a) **Ships:** nothing.
(b) **Add:** `HeatKernelSignature(complex, t []float64) [][]float64` = `HKS(p, t) = Σ_k exp(−λ_k t) φ_k(p)²` over Laplace-Beltrami eigenmodes (Sun-Ovsjanikov-Guibas 2009 SGP). `WaveKernelSignature` (Aubry-Schlickewei-Cremers 2011) for invariant scale. **Cutting-edge mesh-correspondence primitive used in shape-matching, retrieval, retrieval-augmented-generation over 3D databases** — direct Pistachio/Nimbus/Sentinel relevance. Computes shape signatures invariant to isometric deformation. Same eigendecomposition powers spectral mesh segmentation (Reuter 2009, Liu-Zhang 2007).
(c) **LOC:** ~120 in `geometry/dec/spectral.go`.
(d) Depends X12 + 097 SparseEigen.

#### X29 — Spectral DEC eigenmode benchmarks (canonical pin)

(a) **Ships:** nothing.
(b) **Add:** `EigenmodesEquilateralDisk(N int)` returns first 5 Δ-eigenvalues; pin to analytic Bessel zeros `λ_{n,m} = (j_{n,m}/R)²` at O(h²) — the canonical pin for discrete-spectral-correctness (Reuter et al. 2009 §4). `EigenmodesSphere` pins to `l(l+1)/r²`. `EigenmodesTorus` pins to `(2π)²(m²/L_x² + n²/L_y²)`.
(c) **LOC:** ~70 in `geometry/dec/eigenmodes.go` (test fixture, not consumer-facing).
(d) Depends X12.

#### X30 — Discrete-Hodge-Laplacian dual-mesh symmetry validator

(a) **Ships:** nothing.
(b) **Add:** `IsHodgeLaplacianSymmetric(complex) error` validates `Δ_k = Δ_k^T` via `δd + dδ` self-adjointness. Numerical hazard test for any caller's custom-mesh.
(c) **LOC:** ~40 in `geometry/dec/symmetry.go`.
(d) Depends X6.

### Tier 5 — DEC-PDE: Maxwell, fluids (Mullen-2009 circulation-preserving NS), elasticity (~520 LOC)

#### X18 — Discrete Maxwell on simplicial / cubical meshes

(a) **Ships:** nothing. 159-W4 ships Yee on Cartesian (= cubical complex special case).
(b) **Add:** `MaxwellMesh{E [edges]; B [faces]; eps, mu [tris]}` + `MaxwellStep(mesh, *fields, dt)`. Faraday `∂B/∂t = −d_1 E`; Ampère `∂(★_2 D)/∂t = d_2^T (★_1^{1/μ} B) − J`. Yee scheme = exactly this on cubical complex (Bossavit 2001 Thm 4.2; Stern-Tong-Desbrun-Marsden 2015). Subsumes 194-D6+D7+D13+D14+D16. Validates against 159-W4 to round-off on logically-rectangular triangle mesh.
(c) **LOC:** ~140 in `geometry/dec/maxwell.go`.
(d) Depends X4+X3+X13 (cubical for Yee equivalence).

#### X19 — Mullen-2009 circulation-preserving Navier-Stokes

(a) **Ships:** nothing.
(b) **Add:** `DECNavierStokes(complex, vorticity_2form Cochain, dt) Cochain` — Mullen-Crane-Pavlov-Tong-Desbrun 2009 SIGGRAPH "Energy-Preserving Integrators for Fluid Animation" — a discrete-exterior-calculus Navier-Stokes solver that **preserves Kelvin's circulation theorem to machine precision** (vorticity advection by stream-function pressure-projection, all on dual-mesh). Cited foundational scheme. Frontier primitive: no zero-dependency Go library ships any structure-preserving fluid solver with circulation guarantee.
(c) **LOC:** ~180 in `geometry/dec/fluid.go`.
(d) Depends X10+X12.

#### X20 — Stern-2015 variational-integrator DEC-Maxwell (cross 204)

(a) **Ships:** nothing.
(b) **Add:** `VariationalDECMaxwell(complex, fields, dt)` — Stern-Tong-Desbrun-Marsden 2015 PJM "Variational Integrators for Maxwell's Equations with Sources". Lagrangian-DEC integrator preserving discrete symplectic 2-form → exactly conserves discrete-Gauss-law `d_2 (★_2 D) = ★_3 ρ` over arbitrary timesteps (vs Yee's drift). Cross-link 204 symplectic integrators.
(c) **LOC:** ~140 in `geometry/dec/varintegrator.go`.
(d) Depends X18 + 204 (soft).

#### X21 — Christiansen-2012 discrete Yang-Mills

(a) **Ships:** nothing.
(b) **Add:** `DiscreteYangMills(complex, A_su2_per_edge) F_su2_per_face` — Christiansen-Halvorsen 2012 *Numer. Math.* "A simplicial gauge theory". Non-abelian generalisation of X18's abelian Maxwell-on-meshes. SU(2)-valued connections on edges; lattice-gauge-theory style. Soft blocker on 205-L1+L2 (Lie-algebra exp/log).
(c) **LOC:** ~160 in `geometry/dec/yangmills.go`.
(d) Soft-blocker 205-L1+L2; abelian (= Maxwell) ships first.

### Tier 6 — Subdivision + polygonal-mesh DEC (~340 LOC)

#### X22 — Catmull-Clark subdivision (1978)

(a) **Ships:** nothing. `geometry/curves.go.CatmullRom` is the 1-D curve, not the surface scheme.
(b) **Add:** `CatmullClarkSubdivide(quadMesh) quadMesh` — Catmull-Clark 1978 *CAD* "Recursively generated B-spline surfaces on arbitrary topological meshes". Generalises bicubic-B-spline tensor-product surfaces to arbitrary-topology quad meshes; used by Pixar (RenderMan), Maya, Blender; the **dominant surface-modeling primitive in computer graphics**. Each subdivision step replaces every quad with 4 quads via face-points + edge-points + vertex-points. DEC interaction: subdivision *commutes* with d_0 (gradient discretisation refines naturally) but *modifies* ★ (the diagonal Hodge changes per refinement — careful Galerkin-Hodge X14 needed for convergence under refinement).
(c) **LOC:** ~140 in `geometry/dec/subdiv.go`.
(d) No blocker; depends X13 cubical-complex 2D (quad meshes).

#### X24 — Loop subdivision (1987)

(a) **Ships:** nothing.
(b) **Add:** `LoopSubdivide(triMesh) triMesh` — Loop 1987 thesis. Triangle-mesh analogue of Catmull-Clark; standard for triangle-input. DEC interaction same as X22.
(c) **LOC:** ~120 in `geometry/dec/loopsubdiv.go`.
(d) No blocker; depends X1.

#### X25 — Polygonal-mesh DEC (de Goes-Crane 2016)

(a) **Ships:** nothing.
(b) **Add:** `PolygonalLaplacian(polyMesh) SparseMatrix` — de Goes-Desbrun-Crane 2016 *ACM TOG* "Subdivision Exterior Calculus for Geometry Processing". Generalises X12 cotangent-Laplacian to *arbitrary polygonal meshes* (n-gons, not just triangles or quads) via virtual-element-method-style consistent operators. Modern primitive; production-graphics workhorse for 2018+ pipelines.
(c) **LOC:** ~80 in `geometry/dec/polygonal.go`.
(d) Cutting-edge — depends X12.

### Tier 7 — Frontier & out-of-scope deferrals (~780 LOC partial / mostly v2)

#### X26 — DEC for elasticity (Sky-Schenk 2018 / Hirani-Sharma 2008)

`DECElasticity(complex, displacement_1form, lame_lambda, lame_mu) stress_2form`. Discretises continuum-mechanics linear elasticity on simplicial mesh. ~110 LOC. Depends X3+X11+X14.

#### X27 — ODE on manifold (variational integrator, Marsden-West 2001)

`VariationalODEOnManifold(M, lagrangian, q0, dt, n_steps) []q` — discrete-Euler-Lagrange equations from a Lagrangian-form on TM. Cross-link 204 symplectic. ~120 LOC. Depends X11.

#### X18b — Integrable systems / discrete-Hashimoto flow (Pinkall-Springborn 2008)

Integrable-system structure-preserving flow on mesh-curves, used in Willmore-flow/conformal-mesh-fairing. Cutting-edge frontier 2008-2024. ~80 LOC. v2 deferral.

#### Out-of-scope

- DEC on point clouds (Crane et al. 2013 for kNN-meshing) — owned by `graph/`, not here.
- DEC on non-manifold complexes (Hirani 2003 §6.5) — orientation cover, defer.
- Spectral element / hp-FEEC (Demkowicz 2007) — needs polynomial Whitney forms order >1, defer.
- Gerbes / 2-bundles / higher-stack DEC — explicit v2.
- Symplectic DEC for Hamiltonian field theories on time-varying meshes — partially X20.

---

## 2. Composition graph

```
              X1 SimplicialComplex
                    │
        ┌───────────┼───────────┐
    X2 Cochain  X4 d_k     X11 Whitney
        │           │           │
        ▼     ┌─────┼─────┐     ▼
    X3 ★_k  X8 dd  X5 δ  X9 ∧  X14 Galerkin-★
        │   Stokes  │     │
        ▼           ▼     ▼
    X12 cot-Lapl ◄ X6 Δ=δd+dδ  X16 pull-back ──(247-mortar)
        │             │
        ▼             ▼
    X19 Mullen-NS  X10 Helmholtz-Hodge ─► X23 Betti ◄═(156-persistent)
    (Kelvin-circ)     │                     │
    [MOAT]            ▼                     ▼
                X18 Maxwell-mesh         X17 Cohomology
                (159 Yee = X18 on X13 cubical)
                      │
                      ▼
              X20 var-int (204-symplectic) ► X21 Yang-Mills (205-Lie)
              X28 HKS / X29 Bessel-pins
              X13 cubical ─► X22 CC / X24 Loop / X25 polygonal-DEC
              X26 elasticity / X27 ODE-on-M / X18b integrable [v2]
```

Tier-1 unblocked (X1+X2+X3+X4+X8 ~480 LOC). Tier-2 unblocked-dense (X5+X6+X7+X10+X12 ~430). Tier-3 unblocked (X9+X11+X13+X15+X16 ~610); X14 sparse blocked-097. Tier-4 partial (X17+X23+X29+X30 ~290 unblocked; X28 ~120 needs 097). Tier-5 partial (X18+X19+X20 ~470 unblocked, X21 ~160 blocked-205). Tier-6 unblocked (X22+X24+X25 ~340). Tier-7 v2 (~310).

---

## 3. Cheapest 1-day PR + keystone + moat

**Cheapest 1-day:** X1+X2+X3+X4+X8 ~480 LOC src + 280 test — first cochain-complex in `geometry/`, R-MUTUAL-CROSS-VALIDATION 3/3 (`calculus.SimpsonsRule` ∮ vs discrete Σ vs dd=0), mirrors recent 6a55bb4/365368a saturation pattern, subsumes 194-D0+D1+D2+D5.

**Architectural keystone:** X12 Cotangent-Laplacian Pinkall-Polthier-1993 — most-cited primitive in computational DDG (Crane-et-al-2013-SGP entry point); unblocks heat-method-geodesics, conformal-parameterisation, mesh-smoothing, shape-correspondence, spectral-segmentation — all are X12+linear-solve.

**Cutting-edge moat:** X19 Mullen-2009 + X20 Stern-2015 + X21 Christiansen-2012 = structure-preserving-DEC frontier (no zero-dep library in any language ships all three; libigl C++ has cot-Laplacian only, gudhi has combinatorial-Hodge only; production code for Mullen-NS / Stern-EM / Christiansen-YM is absent everywhere).

---

## 6. Recommended placement and import graph

NEW sub-package `geometry/dec/` (mirrors precedent of 194 D0–D17 placement, 207 D15–D20 placement, 208 E1–E20 placement — all four `geometry/{dec,diffgeo,extcalc,dec-cubical-substrate}/`).

Cycle-free imports:
```
geometry/dec/  →  geometry/             (TriangleArea2D)
              →  linalg/                (LU, sparse-when-097)
              →  calculus/              (SimpsonsRule, GaussLegendre — test side)
              →  constants/             (VacuumPermittivity, μ for X18)
              →  topology/persistent/   (test-only Filtration adapter for X23 cross-validation)
              →  signal/                (test-only FFT for X28 cyclic-symmetry case)
              →  chaos/                 (test-only RK4 for X20 variational-integrator comparison)
```

Verified zero in-bound from any of: geometry, linalg, calculus, constants, topology, signal, chaos. No cycle.

---

## 7. LOC roll-up

| Tier | Primitives | Source LOC | Test LOC | Cumulative |
|------|-----------|------------|----------|-----------|
| 1 (1-day, ship today) | X1, X2, X3, X4, X8 | 480 | 280 | 480 |
| 2 (Lapl + cot-keystone) | X5, X6, X7, X10, X12 | 430 | 320 | 910 |
| 3 (FEEC + cubical) | X9, X11, X13, X14, X15, X16 | 610 | 420 | 1520 |
| 4 (cohomology + spectral) | X17, X23, X28, X29, X30 | 410 | 280 | 1930 |
| 5 (DEC-PDE) | X18, X19, X20, X21 | 620 | 380 | 2550 |
| 6 (subdivision) | X22, X24, X25 | 340 | 220 | 2890 |
| 7 (frontier v2) | X26, X27, X18b | 310 | 180 | 3200 |

Ship through Tier-6: ~2,890 LOC src + 1,900 LOC test. Tier-7 v2 deferral.

(194-D0–D17 ~2,050 LOC superseded; ~370 LOC consumer-side wrappers re-home from 194 here as X18+X19+X20.)

---

## 8. Cross-language pinning targets (golden files)

- **X4 d_0**: `f(x,y) = x² + y²` on regular disk triangulation → `(d_0 f)[ij] = f[j] − f[i]` exact integer-arithmetic 1e-15.
- **X8 dd=0**: any random orientation-consistent mesh, any random 0-cochain → `Σ |d_1 d_0 f[t]|² < 1e-26` (machine zero).
- **X3 ★_1 Hirani**: equilateral-triangle-mesh of unit disk → `★_1[e] = h*/h` matches Hirani 2003 Table 4.1 to 1e-12.
- **X12 cot-Laplacian eigenvalues**: equilateral-triangle Dirichlet disk → first 5 eigenvalues match Bessel `(j_{n,m}/R)²` to O(h²); pin h=1/64 → 5e-4 accuracy.
- **X10 Hodge-decomp on torus**: harmonic dim = 2 (β_1 = 2g). Integer-equality across this slot + 156 + 208-E15.
- **X23 Betti S²/T²/Σ_g**: integer byte-equality contract — this slot vs 156 vs 208-E15.
- **X18 Maxwell rectangular cavity**: TE_{1,0} mode `f = c/(2L)` matches DEC eigenmode to O(h²); cross-validates 159-W4 Yee-on-Cartesian.
- **X19 Mullen-NS**: Taylor-Green vortex on doubly-periodic mesh → vortex-circulation `Γ(t) = Γ(0)` to 1e-12 over 10⁵ steps (Kelvin's circulation theorem byte-preserved).
- **X22 Catmull-Clark**: limit surface of cube under infinite subdivision → exact bicubic-B-spline patch values at parameter (½, ½) to 1e-15 (Stam 1998 closed-form evaluation at arbitrary parameters).
- **X25 polygonal-Laplacian**: hexagonal-mesh of disk → eigenvalues match X12-triangulated-mesh equivalent to 1e-3 (de Goes-Desbrun-Crane 2016 §5).

---

## 9. Precision hazards

X1 canonical edge orientation `(min(i,j), max(i,j))` invariant — inconsistent → silent `dd≠0` → garbage. X3 Hirani-Hodge degrades to O(h^0) on non-Delaunay (Hirani 2003 §4.1); use X14 Galerkin-Hodge for fidelity. X12 cot-weight negativity on obtuse triangles → loses M-matrix; PP-1993 Thm 4.1 still gives convergence but downstream solvers must not assume positive-off-diagonals. X10 Δ_0 kernel = constants on closed mesh; CG needs deflation of (1,…,1). X11 Whitney evaluation outside simplex undefined. X18 CFL on unstructured: `dt ≤ min_e (|e|·★_1[e])^{1/2}/c` (Bossavit-Kettunen 2000). X22/X24 subdivision limit is C¹ only at extraordinary vertices (valence ≠ 4 for CC, ≠ 6 for Loop). X19 needs Δ_0-Poisson every step — mesh-quality precondition.

---

## 10. Cross-link summary

194-em-geometry SUPERSEDED (X1–X10+X18 cover entire 194 D0–D17 surface; 194 narrows to consumer wrapper). 207-new-diff-geo complementary continuous-tensor side (Christoffel/Riemann/geodesic). 208-new-exterior-calculus complementary continuous-Λ^k; Whitney-de Rham bridge to X11. 156-topology-persistent integer-equality Betti contract via X23 (Eckmann 1944). 159-W4 Yee absorbed as X18-on-X13-cubical (Bossavit 2001 Thm 4.2). 160-fluids/192-fluids-control consume X19 Mullen-NS substrate. 174-geometry-missing barycentric helper for X11. **097 SparseEigen** highest-leverage unblocker — gates X10-large/X14/X17-large/X28. 204-symplectic gates X20/X27. 205-Lie gates X21 non-abelian. 245-spectral distinct (Chebyshev/Fourier/RBF, not mesh-DEC). 247-mortar (read-ahead) composes on X16. 248-multigrid (read-ahead) uses X12 as smoother.

---

## 11. Verdict

Ship Tier-1 (X1+X2+X3+X4+X8 ~480 LOC + 280 test) as 1-sprint standalone PR — saturates R-MUTUAL-CROSS-VALIDATION 3/3, unblocks chapter. Then Tier-2 (X5+X6+X7+X10+X12 ~430 LOC, X12 keystone), Tier-3 (X9+X11+X13+X15+X16 ~610 LOC, FEEC + cubical), Tier-4 (X17+X23+X29+X30 ~290 LOC unblocked, X28 ~120 LOC blocked-097), Tier-5 (X18+X19+X20 ~470 LOC unblocked, X21 ~160 LOC blocked-205, **cutting-edge moat**), Tier-6 (X22+X24+X25 ~340 LOC subdivision), defer Tier-7 (X26+X27+X18b ~310 LOC, v2 once 204+205 ship).

Total v0 through Tier-6: **~2,890 LOC src + 1,900 LOC test**. 28/30 primitives ship without 097 SparseEigen unblocker. Cross-package re-home: 194 D0–D17 narrows to "EM-on-meshes consumer wrapper" — single-source ownership of discrete cochain side concentrates at `geometry/dec/`, net saving ~1,700 LOC of cross-review primitive duplication.

Single highest-leverage 1-day project: **X1+X2+X3+X4+X8 ~480 LOC** — first cochain-complex primitive in reality. Single architectural-keystone: **X12 Cotangent-Laplacian Pinkall-Polthier-1993 ~140 LOC** — entry point of SGP-DDG pipeline (Crane et al. 2013). Single cutting-edge: **X19+X20+X21 ~480 LOC** structure-preserving-DEC frontier (Mullen-NS / Stern-Maxwell / Christiansen-YM).

End of report. 30 primitives, ~2,890 LOC src, ~1,900 LOC test, one new sub-package `geometry/dec/`, zero cycles, three soft blockers (097/204/205). Keystone X12 ships unblocked. Strict superset of 194 D0–D17.

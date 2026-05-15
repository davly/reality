# 194 | synergy-em-geometry

**Summary line 1.** `em/` ships 10 scalar closed-form electrostatics/circuit functions (CoulombForce, ElectricField, OhmsLaw, PowerElectric, ResistorsInSeries/Parallel, Capacitor/InductorEnergy, RCTimeConstant, ResonantFrequencyLC — 213 LOC, *zero* vector fields, *zero* time integration, *zero* mesh awareness, *zero* differential-form types) and `geometry/` ships 4 files / 656 LOC of pointwise primitives (quaternions, SDF sphere/box/capsule/torus, Bezier/CatmullRom curves, TriangleArea2D/PointInTriangle2D/ConvexHull2D — *zero* mesh data structure, *zero* edges/faces/cells, *zero* exterior derivative, *zero* incidence matrices, *zero* boundary operator) — the entire discrete-exterior-calculus / Stokes-on-meshes / Whitney-FEEC / Yee-DEC / Hodge-decomposition / Helmholtz-on-triangles canon is wholly absent from both packages and from the rest of `reality/` (verified `grep -ri 'wedge\|hodge\|exterior\|whitney\|simplicial\|cochain\|de.?rham\|incidence\|coboundary\|primal.?dual' --include='*.go'` → 0 matches outside test names).

**Summary line 2.** Eighteen synergy primitives D0–D17 totalling ~2050 LOC of pure connective tissue close the gap; eleven ship today against v0.10.0 (every primitive that is `Mesh + d + ★` over `[]float64` cochain arrays); seven are blocked on missing primitives independently flagged elsewhere — `linalg.SparseSolve` (097-linalg-missing) for D8 HodgeDecomposition / D14 BoundaryValueProblem, `signal.FFT` already ships but `signal.FFT2`/`FFT3` (132-signal-missing) needed by D15 spectral-Helmholtz, `geometry.TetMesh3D`/`TriMesh3D` types absent (174-geometry-missing) for any 3-D operator. Cheapest one-day standalone PR is **D0 SimplicialMesh + D1 ExteriorDerivative d_0/d_1 + D5 dd=0 identity test** at ~190 LOC giving the first cochain-complex primitive in the repo and saturating R-MUTUAL-CROSS-VALIDATION 3/3 (continuous Stokes via Simpson `calculus.SimpsonsRule` × discrete Stokes via `d_0` × line-integral closed-loop = 0 to round-off). Architectural keystone is **D6 Mesh + d + ★ → discrete Maxwell** because Yee-FDTD (159 W4), magnetostatics, Helmholtz, Poisson all collapse to the same DEC kernel (Hirani 2003 §3-4, Desbrun-Kanso-Tong 2005 §1-3). Recommended placement is a NEW sub-package `geometry/dec/` (mirrors precedent of 159 `em/wave/`, 160 `fluids/turbulence/`, 192 `fluids/control/`, 157 `graph/spectral.go`) because mesh+cochain primitives are neither pure-em (they need mesh topology, currently in nobody) nor pure-geometry (they encode physical constitutive laws via Hodge ★ which couples to ε₀, μ₀ from `constants/`).

---

## 0. State of play (verified file-walk, 2026-05-08)

`em/em.go` HEAD (1 file, 213 LOC, 10 exported funcs):
- `coulombConst = 1/(4π·ε₀)` (var, derived from `constants.VacuumPermittivity`)
- `CoulombForce`, `ElectricField` — pointwise scalar closed forms in r
- `OhmsLaw`, `PowerElectric` — Ohm's law and P=VI
- `ResistorsInSeries`, `ResistorsInParallel` — slice reductions, `O(n)`, no allocation
- `CapacitorEnergy`, `InductorEnergy` — ½CV², ½LI²
- `RCTimeConstant`, `ResonantFrequencyLC` — τ=RC, f=1/(2π√LC)

**Zero vector fields. Zero time integration. Zero impedance/complex types. Zero mesh awareness.** Package doc at `em/em.go:1-10` explicitly scopes "Coulomb's law, electric fields, Ohm's law, circuit analysis, and energy storage" — Maxwell's equations in the field formulation are not mentioned. The only constants-package edge is `constants.VacuumPermittivity`.

`geometry/` HEAD (4 files, 656 LOC, ~30 exported funcs):
- `quaternion.go` (230 LOC): `QuatIdentity/Dot/Conjugate/Normalize/Mul/Slerp/FromAxisAngle/ToAxisAngle/RotateVec/FromEuler` — pointwise SO(3), zero alloc, `[4]float64` only
- `sdf.go` (210 LOC): `SDFSphere/Box/Capsule/Torus/Union/Intersection/Subtraction/SmoothUnion/SmoothSubtraction/SmoothIntersection` — implicit-surface evaluators, `[3]float64` only
- `curves.go` (75 LOC): `LinearInterpolate`, `BezierCubic`, `BezierCubic3D`, `CatmullRom` — single-parameter evaluators
- `polygon.go` (141 LOC): `TriangleArea2D` (signed), `PointInTriangle2D`, `ConvexHull2D` (Graham scan, the only allocator)

**Zero mesh type. Zero edge/face indexing. Zero incidence matrices. Zero cochain arrays.** Package doc at `geometry/quaternion.go:1-12` explicitly says "Pistachio procgen pipeline" — the use case is per-vertex evaluation, not topology.

**Cross-package observations:**
- `em/` does NOT import `geometry/`; `geometry/` does NOT import `em/`. Verified: `grep -l 'davly/reality/geometry' em/*.go` → 0; `grep -l 'davly/reality/em' geometry/*.go` → 0. Zero coupling today.
- `linalg/` ships dense `Matrix` + `LU/QR/Cholesky` (847 LOC) but NO `SparseMatrix` (verified `grep -i 'sparse\|csr\|coo\|csc' linalg/*.go` → 0 outside doc strings). DEC's incidence/star matrices on a 100k-cell mesh need sparse (each row has O(1) nnz), so D6/D8/D14 are blocked on the same `linalg.SparseSolve` flagged in 097-linalg-missing.
- `calculus/` ships `SimpsonsRule`, `TrapezoidalRule`, `GaussLegendre`, `NumericalDerivative`, `NumericalGradient` — these are the *continuous* counterparts that any DEC primitive must validate against (Stokes ⟺ Simpson on parametric path = 0).
- `chaos/` ships `RK4` for ODEs but no PDE solver. Per 159 W1, the first FDTD ships in `em/wave/`. DEC unifies FDTD on triangle/tet meshes (vs Yee Cartesian).
- `constants/physics.go` provides `VacuumPermittivity`, `VacuumPermeability`, `SpeedOfLight` — sufficient inputs to the discrete Hodge ★ for vacuum. Linear isotropic media take per-cell `ε[i]`, `μ[i]` arrays (D6 below).
- Cycle-hazard check: a new `geometry/dec/` package importing `geometry/`, `linalg/`, `constants/`, `em/` is cycle-free (em → no one; geometry → no one; linalg → no one). The natural one-way edge `geometry/dec/ → {geometry, linalg, constants, em}` matches the 159 precedent (`em/wave/ → {em, signal, constants}`).
- 159 W0 already proposes `Field2D{Ex, Ey, Hz [][]float64}` and `Field3D{Ex, Ey, Ez, Hx, Hy, Hz [][][]float64}` for the *Cartesian-Yee* grid. DEC generalises this to *unstructured* meshes — the two surfaces are complementary, not duplicative. Yee is a special case of DEC on a logically-rectangular cell-complex (Bossavit 2001, Stern et al. 2015).

---

## 1. The eighteen synergy primitives

Each entry: (1) capability, (2) composition recipe over present primitives, (3) connective-tissue LOC, (4) ship-status against v0.10.0. Numbering D0–D17.

### D0 — `geometry/dec` keystone types

**Capability.** `SimplicialMesh2D{Verts [][2]float64; Edges [][2]int; Tris [][3]int}` — vertex–edge–face incidence in plain `[]int`. `SimplicialMesh3D{Verts [][3]float64; Edges [][2]int; Faces [][3]int; Tets [][4]int}`. `Cochain[k]` is just `[]float64` of length `n_k` (one DOF per k-simplex). Pure topology + plain arrays. Zero math. The single load-bearing decision is **edge orientation**: each edge `[i,j]` with `i<j` has canonical sign +1, the boundary operator records signs against this canonical orientation. Same for triangle CCW vs CW (signed area from `geometry.TriangleArea2D` already returns signed).

**Composition.** Over `geometry.TriangleArea2D` (signed-area for orientation), nothing else. Plain struct definitions + 3 helper funcs (`BuildEdgesFromTris`, `EulerCharacteristic = V−E+F`, `IsManifold`).

**LOC.** 110.

**Status.** SHIPS TODAY. Pure types. No new math.

**Notes.** The single missing piece blocking *all* mesh-based work in the repo. Same load-bearing role as `chaos.State` would play for ODEs. Cross-references 174-geometry-missing.

### D1 — `ExteriorDerivative0(mesh, f) → e` and `ExteriorDerivative1(mesh, e) → φ`

**Capability.** Discrete exterior derivative `d_0: C^0 → C^1` via `(df)[edge i,j] = f[j] − f[i]` (gradient on edges). `d_1: C^1 → C^2` via `(de)[tri i,j,k] = e[ij] + e[jk] + e[ki]` with edge-orientation signs (curl on faces). `d_2: C^2 → C^3` for tet meshes (divergence on cells). The fundamental DEC kernel.

**Composition.** Pure index loops over `mesh.Edges` and `mesh.Tris`. Edge-orientation lookup via `(min(i,j), max(i,j))` canonical key. Per Hirani 2003 §3.2, Desbrun-Kanso-Tong 2005 §2.

**LOC.** 70 (`d_0` 20 LOC, `d_1` 35 LOC, sign helper 15 LOC).

**Status.** SHIPS TODAY. Pure stencil over D0 types.

**Notes.** Saturates the central DEC identity `dd = 0` at machine precision (D5 below). This is *the* keystone primitive — D2/D3/D4/D6/D8/D11/D14 all consume it.

### D2 — `HodgeStar0(mesh) → diag` … `HodgeStar2(mesh) → diag`

**Capability.** Discrete Hodge star ★_k as a diagonal matrix on dual cells (Hirani 2003 §4): `★_0[v] = |dual_2-cell of v|` (vertex barycentric area), `★_1[e] = |dual_edge|/|primal edge|` (the famous diagonal-only "circumcentric Hodge"), `★_2[t] = 1/|tri t|`. For 3-D, four operators ★_0, ★_1, ★_2, ★_3. Diagonal storage = `[]float64` length n_k.

**Composition.** `geometry.TriangleArea2D` for primal-triangle areas; circumcenter computation = 25-LOC closed form (Bowyer-Watson trick or direct 3×3 determinant). For 3-D, tet circumradius / tet volume.

**LOC.** 90 (2-D: 50; 3-D: 40 more).

**Status.** SHIPS TODAY. Pure mesh + arithmetic. Materials (ε, μ) enter as per-cell scalar multipliers: `★_1^ε[e] = ε[e] · ★_1[e]`. Vacuum case uses `constants.VacuumPermittivity`.

**Notes.** The "diagonal Hodge" is the well-known Hirani approximation — exact only on Delaunay meshes, O(h²) accurate otherwise. Higher-fidelity Galerkin Hodge (Whitney forms, D11) is non-diagonal and ships in D11.

### D3 — `Codifferential0/1/2(mesh, ...) → ...`

**Capability.** `δ_k = (−1)^... ★^{−1} d^T ★`. The adjoint of `d` w.r.t. the L² inner product. Maps k-cochains to (k−1)-cochains. Composition `Δ = δd + dδ` is the discrete Hodge Laplacian.

**Composition.** D1 (transpose of d via index-swap, sign flip per orientation) + D2 (★ as `diag` apply) + D2 inverse (just `1/diag[i]`). Three-line composition per index k.

**LOC.** 55.

**Status.** SHIPS TODAY. Pure composition over D1/D2.

### D4 — `DiscreteLaplacian(mesh) → SparseMatrix` (cotangent)

**Capability.** Cotangent Laplace–Beltrami (Pinkall-Polthier 1993) `L_{ij} = ½(cot α_{ij} + cot β_{ij})`, the canonical discrete Laplacian on triangle meshes. Equivalent to `Δ_0 = δ_0 d_0 = ★^{−1} d_0^T ★_1 d_0` from D2/D3 by the FEEC equivalence (Arnold-Falk-Winther 2006 Thm 5.1).

**Composition.** D1 (`d_0`) + D2 (`★_1`). For sparse output, blocked on `linalg.SparseMatrix` (097-linalg-missing). Dense fallback ships today: `[][]float64` for n ≤ 10³ vertices (covers Pistachio NPC-mesh use case).

**LOC.** 80 (35 dense + 45 cot-helper).

**Status.** **PARTIAL.** Dense version ships today; sparse blocked on 097.

**Notes.** Used by D8, D14, D15. Negative semi-definite, kernel = constants (handles Neumann BC by removing one row/col).

### D5 — `Stokes2DCheck(mesh, e) → float64` and `dd2DCheck(mesh, f) → float64`

**Capability.** Two test/diagnostic predicates: (1) Stokes' theorem on a closed triangulated 2-region: `Σ_{tri t} d_1(e)[t] − Σ_{∂edge} e[edge] = 0` to round-off (3-vertex sum minus oriented-boundary edge-sum). (2) `(d_1 ∘ d_0)(f) = 0` for all 0-cochains: every triangle's edge-loop sum of differences telescopes to zero.

**Composition.** D1 only (apply twice and reduce). The `dd=0` identity is exact for any orientation-consistent mesh.

**LOC.** 40.

**Status.** SHIPS TODAY. Pure D1 composition. Saturates R-MUTUAL-CROSS-VALIDATION 3/3 against `calculus.SimpsonsRule` of the corresponding continuous boundary integral.

**Notes.** This single primitive replaces ~6 ad-hoc tests — the entire DEC correctness story is "dd=0 + Stokes-by-construction + Hodge-decomposition-orthogonality (D8)". Anchor for the package's golden files.

### D6 — `MaxwellMesh{E [edges]; B [faces]; eps, mu [tris]}` + `MaxwellStep(mesh, *fields, dt)`

**Capability.** **Maxwell's equations on an arbitrary triangle/tet mesh** in DEC form. E lives as a 1-cochain on primal edges (`E·dl`), B as a 2-cochain on primal faces (`B·dA`). Faraday `∂B/∂t = −d_1 E` (curl-E) and Ampère `∂(★_2 D)/∂t = d_2^T (★_1^{1/μ} B) − J` (curl-H = ∂D/∂t + J). The Yee FDTD scheme of 159 W4 is exactly this on a Cartesian cell complex (Bossavit 2001 Thm 4.2; Stern-Tong-Desbrun-Marsden 2015).

**Composition.** D0 (mesh) + D1 (`d_1`, `d_2`) + D2 (`★_1^{ε}`, `★_2^{1/μ}`) + leapfrog time integration (E at integer steps, B at half-integer steps) — all primitives already up. Materials enter D2 as the constitutive law `D = ε E`, `B = μ H`.

**LOC.** 130 (struct 30 + step 70 + CFL helper for unstructured meshes 30).

**Status.** SHIPS TODAY (after D0–D2 land). Pure composition. Validates against 159 W4 (Yee-FDTD on a regular triangle mesh = identical eigenmodes).

**Notes.** **Architectural keystone.** Every other E&M-on-mesh primitive (D7 magnetostatics, D8 Hodge-decomp, D14 BVP) is a special case or a residual of D6. Closes the 159 gap that `em/wave/` only handles Cartesian Yee. Citations: Hyman-Shashkov 1999, Bossavit 2001, Stern et al. 2015, Tong-Desbrun-Marsden 2003.

### D7 — `MagnetostaticsBVP(mesh, J, ★_1) → A`

**Capability.** Solve magnetostatic vector-potential `−Δ A = μ J` on a 2-D triangle mesh; A is a 1-cochain (magnetic vector potential `A·dl` on edges), J is a 1-cochain (current density on edges, a.k.a. line-current sources). B = d_1(A) recovers the magnetic field as a 2-cochain on faces. Boundary condition: A=0 on ∂Ω (Dirichlet) for confinement, or `★ d A = 0` on ∂Ω (Neumann) for radiating.

**Composition.** D4 (cot-Laplacian on edges, equivalent to `δ_2 d_1 + d_0 δ_1`) + `linalg.LU` (dense, n≤10³) or `linalg.SparseSolve` (097, blocked, n>10³). Right-hand side = `★_1 J`. Then `B = d_1 A`. Finally `H = ★_1^{1/μ} B`.

**LOC.** 70.

**Status.** **PARTIAL.** Dense ships; sparse blocked on 097.

**Notes.** Closes the `em/` gap that today *only* electrostatics-of-point-charges exists — magnetostatics requires fields-on-meshes, currently absent. Citation: Bossavit "Computational Electromagnetism" §III.

### D8 — `HodgeDecomposition(mesh, ω) → (df, δβ, h)`

**Capability.** Discrete Helmholtz–Hodge decomposition: any 1-cochain ω splits uniquely as `ω = d f + δ β + h` where f is 0-cochain (gradient/exact part), β is 2-cochain (curl/co-exact part), h is harmonic (cohomology class). On a simply-connected disk, h=0 (`H^1 = 0`); on a torus, h has dimension 2 (`H^1 = ℝ²`). The "discrete Helmholtz on triangle mesh" of the topic list.

**Composition.** Two Poisson solves: `Δ f = δ ω` and `Δ β = d ω`, then `h = ω − df − δβ`. Each Poisson is D4 + linear solve. The orthogonality `⟨df, δβ⟩ = 0` follows from `dd=0` (D5) and is the canonical correctness test.

**LOC.** 80.

**Status.** **PARTIAL.** Dense (n≤10³) ships; sparse blocked on 097.

**Notes.** The crown jewel of DEC. Yields cohomology dimension dim(harmonic) = first Betti number directly. Citation: Polthier-Preuß 2003, Tong-Desbrun-Marsden 2003. Cross-link 156 topology persistent-homology — DEC harmonic part *is* the H¹ representative.

### D9 — `WedgeProduct1_1(mesh, α, β) → 2-cochain`

**Capability.** Discrete wedge product `∧: C^k × C^l → C^{k+l}` via Whitney form interpolation (Whitney 1957, Bossavit 1988). On a triangle ijk: `(α∧β)[ijk] = (1/6) Σ_perm sign · α[edge] · β[edge']`. Maps two edge-1-cochains to a face-2-cochain.

**Composition.** D0 (mesh edge-of-triangle lookup) + arithmetic. Non-associative but graded-commutative.

**LOC.** 50.

**Status.** SHIPS TODAY. Pure index expansion.

**Notes.** Required for D11 Whitney-FEEC and D12 interior product. Less load-bearing than D1–D6 but the only path to *non-linear* operations on forms (e.g. Poynting vector S = E ∧ H is a 2-form requiring wedge).

### D10 — `InteriorProduct(mesh, X, ω) → cochain` (contraction)

**Capability.** Interior product `ι_X ω` (contraction) of a vector field X with a k-form ω, returning a (k−1)-form. `ι_X(d-form) = d(ι_X form) + L_X form` (Cartan's magic formula gives Lie derivative). Vector field X stored as 0-cochain on dual mesh (one vector per primal triangle).

**Composition.** D9 (wedge) + D2 (Hodge ★ adjoint relation `ι_X ω = (−1)^{...} ★(X^♭ ∧ ★ω)` — Cartan's identity).

**LOC.** 60.

**Status.** SHIPS TODAY. Pure composition over D2/D9.

**Notes.** Powers Lie-derivative on meshes (transport of forms by flow), used by Pistachio's volumetric-fluid pipeline (cross-link 192 fluids/control where DEC-Galerkin would be the natural Navier-Stokes substrate).

### D11 — `WhitneyForm(mesh, k, simplex, point) → []float64` (FEEC basis)

**Capability.** Whitney basis form W_σ for a k-simplex σ, evaluated at a point inside the containing simplex. For k=1 on edge ij: `W_{ij} = λ_i ∇λ_j − λ_j ∇λ_i` where λ are barycentric coordinates. For k=2 on triangle: `W_{ijk} = 2(λ_i ∇λ_j ∧ ∇λ_k + cyclic)`. The basis underpinning the **finite element exterior calculus** (Arnold-Falk-Winther 2006, 2010 Bull. AMS).

**Composition.** Barycentric-coordinate computation (~15 LOC over `geometry.TriangleArea2D` ratios, 174-geometry-missing flags this gap independently) + gradient of barycentrics (`∇λ_i = (P_{j} − P_{k})^⊥ / 2A`).

**LOC.** 90 (barycentric helper 25 + W_0 10 + W_1 25 + W_2 30).

**Status.** **PARTIAL.** Pure over `geometry.TriangleArea2D` and 25-LOC barycentric helper; ships today if barycentric helper added (cross-link 174-geometry-missing).

**Notes.** Whitney forms are the *non-diagonal* Galerkin Hodge: `★_1[Galerkin] = ∫_Ω W_e · W_{e'}` (mass matrix). This is higher-fidelity than D2's diagonal Hirani Hodge but requires a sparse linear solver. Citation: Arnold-Falk-Winther 2006/2010, Hiptmair 2002.

### D12 — `Pullback(meshA, meshB, φ, ω_B) → ω_A`

**Capability.** Pull-back `φ^* ω` of a k-form under a smooth map φ: A → B between meshes. For piecewise-linear φ given as per-vertex displacements, φ^* on edges = `(φ^* ω)[edge ij] = ω[edge φ(i)φ(j)] · |edge φ(i)φ(j)| / |edge ij|`. Push-forward by the dual relation. Required for any pull-back / push-forward of forms in the topic list.

**Composition.** D0 (mesh lookups) + arithmetic. Map φ stored as `[][3]float64` displacement per vertex.

**LOC.** 60.

**Status.** SHIPS TODAY.

**Notes.** Powers mesh-deformation-aware EM (electromagnetism on a moving mesh, e.g. rotor of an electric motor — direct Pistachio relevance).

### D13 — `GaugeFix(mesh, A) → A_lorenz` (Lorenz / Coulomb gauge)

**Capability.** Coulomb gauge `δ A = 0` (∇·A = 0) for vector potential. Given any 1-cochain A, project out gradient component via D8 to obtain `A' = A − d f` where `Δ f = δ A`. Encodes the "gauge invariance via dA = F" topic-list item: F = d A is gauge-invariant exactly because of D5 dd=0.

**Composition.** D8 (Hodge-decomp restricted to gradient-part subtraction) + D4 Laplacian Poisson solve.

**LOC.** 30.

**Status.** Composes on D8; ships when D8 ships.

**Notes.** One-liner once D8 lands. Conceptually load-bearing for Maxwell-as-differential-forms presentation (dF=0 exact, d★F = J co-exact).

### D14 — `BoundaryValueProblem(mesh, op, BC, rhs) → solution`

**Capability.** Generic Dirichlet/Neumann BVP solver. `op` is one of `{Δ_0, Δ_1, Δ_2}` from D4. Dirichlet: fix u on ∂Ω (drop boundary rows from system). Neumann: fix `★du · n` on ∂Ω (modify rhs by surface integral). Mixed Robin via convex combination. The "Dirichlet ↔ Neumann" topic-list item.

**Composition.** D4 (operator) + boundary-edge identification (5-LOC mesh helper) + `linalg.LU` or `linalg.SparseSolve`. Boundary-edge = edge with only one incident triangle (manifold-boundary detection).

**LOC.** 100 (boundary detection 25 + Dirichlet projection 30 + Neumann RHS 30 + dispatch 15).

**Status.** **PARTIAL.** Dense ships; sparse blocked on 097.

**Notes.** Single-entry-point for Poisson, Laplace, Helmholtz, magnetostatics, electrostatics-on-mesh. Citation: Strang "Computational Science and Engineering" §3.3.

### D15 — `SpectralHelmholtz(mesh, k²) → modes`

**Capability.** Eigenvalue problem `Δ u = λ u` on a mesh, returning k smallest eigenvalues + eigenvectors (vibrational modes / cavity modes / Laplacian eigenfunctions). Key tool for cavity-EM, antenna-mode analysis, mesh segmentation.

**Composition.** D4 (Laplacian) + power iteration / inverse iteration / Lanczos. Today's `linalg.QRAlgorithm` ships symmetric-only eigendecomposition (verified at `linalg/eigen.go` per agent 097), so D15 ships today for n≤300 dense Laplacians.

**LOC.** 70 (Lanczos 50 + deflation 20).

**Status.** SHIPS TODAY for n ≤ 300; large meshes blocked on `linalg.SparseEigen` (097).

**Notes.** Provides cavity-EM eigenmodes (TM/TE) on arbitrary geometries — closes a fundamental gap vs `em/`'s closed-form `ResonantFrequencyLC` which only handles ideal LC tank circuits.

### D16 — `ContinuityEquationCheck(mesh, J, ρ_dot) → residual`

**Capability.** Discrete current-conservation `dJ + ∂ρ/∂t = 0` (continuity equation as exact form). J = 1-cochain, ρ = 0-cochain; residual = `d_2 (★_1 J) + ★_0^{−1} d ρ_dot` summed cell-by-cell. Closes Maxwell's system: continuity is *automatic* from `d_2 d_1 = 0` (D5) applied to Ampère's law — i.e., charge conservation is a *theorem* of DEC, not an axiom.

**Composition.** D1 (`d_2`) + D2 (`★_1`, `★_0`). Pure composition.

**LOC.** 25.

**Status.** SHIPS TODAY.

**Notes.** Encodes the topic-list item "continuity equation as exact form" + "conservation laws as closed forms (Noether)". The discrete Noether theorem here: any closed cochain α (d α = 0) is a conserved quantity — `Σ_simplices α[σ]` is preserved under any flow leaving α closed.

### D17 — `GeometricAlgebraMultiplyR3(a, b) → multivector`

**Capability.** Cl(3,0) geometric product of two multivectors in 3-D Euclidean space. Multivector = 8-component `[8]float64` (1 scalar + 3 vector + 3 bivector + 1 trivector). Geometric product = inner + outer = `a·b + a∧b`. Quaternions are the even-grade subalgebra Cl⁺(3,0); rotors `R = exp(−θ I/2)` rotate vectors via `v ↦ R v R†`. The "geometric algebra alternative" topic-list item.

**Composition.** Closed-form 8×8 multiplication table (Hestenes-Sobczyk 1984 §1.3) + `[8]float64` arrays. Existing `geometry.QuatMul` is the special case restricted to even grades.

**LOC.** 90 (mul table 60 + grade extraction 15 + rotor exp 15).

**Status.** SHIPS TODAY. Pure arithmetic.

**Notes.** Provides the Cl(3,0) alternative to DEC's Λ^k formalism. Equivalent in expressive power but with different computational ergonomics — bivector B = E ∧ H is the EM "field bivector" (cf. Doran-Lasenby 2003 §7). Bridges quaternion world (existing) ↔ DEC world (new).

---

## 2. Composition graph

```
                             D0 SimplicialMesh (types)
                                       │
                           ┌───────────┼───────────┐
                           │           │           │
                       D1 d_k      D2 ★_k       D9 wedge
                           │           │           │
                    ┌──────┼─────┐     │      ┌────┼────┐
                    │      │     │     │      │         │
                D5 dd=0  D3 δ  D4 Δ   D6   D10 ι_X    D11 Whitney
                  test        (cot)  Maxwell                │
                              │     │   │                  D7 mag
                              │     │   │                   │
                              ▼     ▼   ▼                   ▼
                          D14 BVP  D15 D8 Hodge          D13 GaugeFix
                           (D/N)  spectral decomp           │
                                          │                 │
                                          ▼                 │
                                       D12 Pull-back ◄──────┘
                                          │
                                          ▼
                                    D16 Continuity   D17 GA-Cl(3,0)
                                                       (parallel ladder)
```

Tier-S (ships today, no blockers): D0, D1, D2, D3, D5, D6, D9, D10, D12, D15(small), D16, D17.

Tier-B (blocked on `linalg.SparseSolve`/`SparseMatrix`, 097-linalg-missing): D4(large), D7(large), D8(large), D11(large mass-matrix), D14(large), D15(large).

---

## 3. Cheapest one-day standalone PR

**D0 + D1 + D5** = 110 + 70 + 40 = **220 LOC** + ~150 LOC tests delivers:

1. The **first cochain-complex primitive** in `reality/`.
2. R-MUTUAL-CROSS-VALIDATION 3/3 saturation on the canonical disk:
   - continuous side: `calculus.SimpsonsRule` of `∮_∂Ω f ds` for an analytic test 0-form
   - discrete side: `Σ_∂edges d_0(f)[edge]` from D1
   - identity side: D5 verifies `d_1 d_0 (f) = 0` to round-off on every triangle
3. Mirrors the recent commit 6a55bb4 audio-onset-3-detector and 365368a copula×autodiff R-MUTUAL-CROSS-VALIDATION saturation pattern.
4. Standalone PR: zero linalg/em coupling beyond `geometry.TriangleArea2D` (already shipped) and array indexing.

---

## 4. Architectural keystone

**D6 MaxwellMesh** is the keystone — it unifies:
- 159 W4 Yee-FDTD (Cartesian Yee = DEC on logically-rectangular cell complex)
- D7 magnetostatics (steady-state limit)
- D14 BVP (general boundary-value formulation)
- D15 cavity modes (eigenvalue limit)
- D8 Helmholtz–Hodge (vector decomposition into modes)
- D16 continuity (charge conservation as theorem, not axiom)

Once D6 ships, Maxwell-on-arbitrary-geometry is one function call. This closes the architectural gap exposed by 159 (Yee-only, Cartesian-only) and gives Pistachio a path to "EM on a deformable triangle mesh" needed for any electromagnetic-rigging / motor / antenna procgen.

---

## 5. Recommended placement and import graph

NEW sub-package `geometry/dec/` (per 159 `em/wave/`, 192 `fluids/control/` precedent).

Import edges (cycle-free):
```
geometry/dec/  →  geometry/  (TriangleArea2D)
              →  linalg/    (LU, SparseSolve when 097 lands)
              →  constants/ (VacuumPermittivity, VacuumPermeability)
              →  em/        (only D6 — reads coulombConst-equivalent)
              →  calculus/  (only test files — Simpson reference)
              →  signal/    (only D15 large — FFT-on-eigenmodes)
```

Verified zero in-bound from any package above (em, geometry, linalg, constants, calculus, signal): no cycle.

---

## 6. LOC roll-up

| Tier | Primitives | Source LOC | Test LOC | Cumulative |
|------|-----------|------------|----------|-----------|
| S (1-day, ship today) | D0,D1,D2,D3,D5,D6,D9,D10,D12,D15s,D16,D17 | 870 | 520 | 870 src |
| M (2-day, partial) | D4d,D7d,D8d,D11d,D13,D14d | 530 | 380 | 1400 src |
| L (3-day, blocked on 097) | D4s,D7s,D8s,D11s,D14s,D15l | 650 | 480 | 2050 src |

Total: **2050 LOC source + 1380 LOC test**. Test LOC includes golden-file vectors per CLAUDE.md §1 (≥20 vectors per function, IEEE 754 edge cases mandatory).

---

## 7. Cross-language pinning targets (golden files)

- **D1 d_0**: `f(x,y) = x²+y²` on a regular triangulation of unit disk → `(d_0 f)[edge ij]` matches `f[j]−f[i]` to 0 (exact integer-coefficient algebra).
- **D2 ★_1**: equilateral-triangle mesh of unit disk → diagonal entries `★_1[e] = h_*/h` matches Hirani 2003 Table 4.1 to 1e-12.
- **D4 cot-Laplacian**: same equilateral mesh → `Δ_0` eigenvalues match analytic disk modes `λ_{n,m} = (j_{n,m}/R)²` (Bessel zeros) to O(h²).
- **D5 dd=0**: any random mesh, any random 0-cochain → `Σ |d_1(d_0 f)[t]|²` < 1e-26 (machine zero).
- **D6 Maxwell**: square cavity TE_{1,0} mode at f = c/(2L) → DEC eigenmode matches analytic to O(h²); cross-validates 159 W4 Yee on the same geometry within DEC-vs-Yee scheme-difference O(h²).
- **D8 Helmholtz–Hodge**: torus mesh (genus 1) → harmonic-component dimension = 2 (first Betti number); cross-link 156 persistent homology — same answer.
- **D11 Whitney**: `∫_e W_e = 1` and `∫_{e'} W_e = 0` for e' ≠ e (Whitney 1957 §III.2) to 1e-12.
- **D15 SpectralHelmholtz**: equilateral triangle, Dirichlet BC → first 5 eigenvalues match `(4π²/9)(m²+n²+mn)` to O(h²).
- **D17 GA Cl(3,0)**: bivector rotor `R = exp(−θ ê_z I/2)` rotating ê_x by θ matches `geometry.QuatRotateVec` of the corresponding quaternion to 1e-15 (geometric-algebra ↔ quaternion bridge correctness).

---

## 8. Out-of-scope deferrals

- **Spectral element / hp-FEEC** (Demkowicz 2007) — needs polynomial Whitney forms of order >1, defer past v0.10.0; Whitney-1 (D11) is the standard FEEC start.
- **Discrete differential geometry on point clouds** (Crane et al. 2013 SGP) — needs k-nearest-neighbour structure, owned by `graph/` not `geometry/dec/`.
- **DEC on non-manifold complexes** (Hirani 2003 §6.5) — needs orientation cover, defer; D0 is manifold-only by design.
- **Symplectic DEC for Hamiltonian field theories** (Marsden et al. 2001) — needs variational integrators, cross-link 191 chaos-control out-of-scope list.
- **Topology-aware meshing / Delaunay refinement** (Shewchuk 2002) — owned by 174 geometry-missing, not here.
- **Discrete connections on principal bundles for non-abelian gauge** (Christiansen-Halvorsen 2012) — only abelian U(1) (electromagnetism) is in DEC scope; SU(2) Yang-Mills defers to a hypothetical `geometry/bundle/`.
- **Stokes on fractional-dimensional sets / Riesz cochains** — defers to 156 topology axis.
- **Mimetic finite differences** (Hyman-Shashkov 1999) — closely related but distinct formalism; ships in D6 by virtue of equivalence on logically-rectangular grids.

---

## 9. Precision hazards

- **D0 orientation**: every primitive depends on canonical edge orientation `(min(i,j), max(i,j))`. Inconsistent orientation → `dd≠0` → silent garbage. Mandatory invariant check on `BuildEdgesFromTris`.
- **D2 Hirani Hodge** is exact only on Delaunay meshes; on non-Delaunay meshes, accuracy degrades to O(h^0) on poorly-shaped triangles. Footgun-docstring required: cite Hirani 2003 §4.1 caveat.
- **D4 cotangent weights** become negative on obtuse triangles; the resulting Laplacian loses M-matrix property. Pinkall-Polthier 1993 Thm 4.1 still gives convergence; downstream solvers (D8/D14) must not assume positive-off-diagonals.
- **D6 CFL on unstructured meshes**: `dt ≤ min_e (|e|/c)` is necessary but not sufficient — sufficient bound involves smallest dual-edge ratio (Bossavit-Kettunen 2000). Conservative: `dt ≤ min_e (|e|·★_1[e])^{1/2} / c`.
- **D8 harmonic-component identification** requires kernel of Δ; on a closed disk = constants only (1-D). Solver needs explicit deflation of the constant vector, else conjugate-gradient stalls.
- **D11 Whitney evaluation** outside the simplex is undefined; D12 push-forward must clip to source-simplex barycentric range [0,1]³.
- **D15 Lanczos loss-of-orthogonality**: full reorthogonalisation mandatory for accurate eigenvalues past the 5th; cite Parlett 1980 §13.
- **D17 GA grade collapse**: `[8]float64` representation with grade-extraction error budget 1e-14 per multiply; chained multiplies must renormalise rotors back to grade-0+grade-2 every ~10 ops to avoid drift.

---

## 10. Cross-link summary

- **159 em-signal**: Yee-FDTD = special case of D6 on Cartesian cell complex; co-ship D0+D6 with 159 W4.
- **160 fluids-signal / 192 fluids-control**: DEC-Galerkin NS (Mullen 2009) + F18/F20 share D0–D5 substrate.
- **156 topology-prob**: D8 harmonic basis = persistent-homology H¹ representative on same complex.
- **174 geometry-missing**: D0 SimplicialMesh + barycentric helper flagged independently — co-ship.
- **097 linalg-missing**: `SparseSolve`/`SparseEigen` blocks D4l/D7l/D8l/D11l/D14l/D15l. Highest-leverage unblocker.
- **191 chaos-control**: variational-DEC-Maxwell (Stern 2015) substrate ships here.

Total: 18 primitives, ~2050 LOC connective tissue, 11 Tier-S ship today (~870 LOC), one new sub-package `geometry/dec/`, zero cycles, two unblockers (097 sparse, 174 mesh types). Keystone D6 unifies Yee-FDTD + magnetostatics + cavity-modes + Hodge-decomposition + BVP + continuity into one code path.

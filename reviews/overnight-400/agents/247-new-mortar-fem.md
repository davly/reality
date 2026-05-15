## 247 | new-mortar-fem — Mortar / non-conforming FEM, hp-adaptivity

**Summary line 1.** reality v0.10.0 ships **ZERO** mortar / non-conforming-FEM / hp-adaptive / DG / IGA / VEM surface — repo-wide grep on `mortar|crouzeix|raviart|nitsche|lagrange.multiplier|interface.coupling|nonconforming|non.matching|cross.point|saddle.point|inf.sup|lbb|babuska.brezzi|discontinuous.galerkin|interior.penalty|sip|nipg|iipg|bassi.rebay|local.dg|hdg|hybrid.discontinuous|hp.adapt|h.refinement|p.refinement|p.enrichment|posteriori|residual.estimat|hierarchical.estimat|gradient.recovery|zienkiewicz.zhu|zz.estimat|dual.weighted|dwr|goal.oriented|anisotropic.mesh|isoparametric|nurbs.basis|isogeometric|iga|t.spline|lr.spline|wachspress|mean.value.coordinat|virtual.element|vem|polygonal.fem|boundary.element|bem|fem.bem.coupling|element.by.element|jump.operator|trace.operator` against `*.go` returns **zero callable matches** anywhere outside review-corpus (only nominal hits are `gametheory_test.go:175 saddle point at (1,0)` for the 1928 minimax theorem name-collision and `changepoint/bocpd.go MapRunLength` posterior MAP name-collision). The closest extant substrate is **none of the FEM type** because slot 244 (`pde/fem/p1_1d.go` D25 + `pde/fem/p1_2d.go` D26 P1 triangular) is itself **NOT YET SHIPPED** — slot 247 is **strict-downstream of 244-D25-D26 ~500 LOC**, of 244-D27 GMRES ~160 LOC (Lagrange-multiplier saddle-point is non-symmetric), of **246-X1-X4 SimplicialComplex2D + d_k + ★_k ~410 LOC** (mortar interface = 1-cochain on shared edge graph, Whitney-projection = X11), of **245-S1-S2-S12 Chebyshev/Legendre/GLL ~270 LOC** (hp-FEM Legendre-shape-functions on reference element + GLL-Lobatto-quadrature), and of **097-T1 SparseMatrix / 097-T2 SparseSolve / 097-T6 SparseEigen** (every realistic mortar problem is sparse — dense-LU is O((n_h+n_H+n_λ)³) which kills any production use beyond N=10³ DOFs). MASTER_PLAN slot 247 is named "Mortar / non-conforming FEM, hp-adaptivity" and is **STRICT-DOWNSTREAM of 244 + 245 + 246**, **STRICT-UPSTREAM of 250-mortar-contact** (slot 250 is contact-mechanics specialisation; reuses the X11 Whitney-mortar-projection here as the gap-function discretisation), **CROSS-LINK 248-multigrid** (hp-multigrid V-cycle prolongation/restriction across p-levels uses M11 cross-mesh-projection here), **CROSS-LINK 249-domain-decomp** (Schwarz with non-overlapping subdomains = mortar; FETI-DP cross-points are exactly M5 here). PARTIAL OVERLAP with 244 (244 owns conforming P1 FEM substrate; 247 extends to non-conforming + hp), 245 (245 owns spectral-element on a single block; 247 ships the cross-element coupling that turns SEM into hp-FEM), 246 (246 owns single-mesh DEC; 247 ships cross-mesh transfer). Cross-link 062-em-missing for FEM-BEM-EM-coupling, 070-fluids-perf for hp-CFD downstream consumer (none in reality today).

**Summary line 2.** Twenty-eight mortar / non-conforming / hp-FEM primitives **M1–M28** totalling ~4,100 LOC organized as **(a) Tier-0 reference-element + shape-function substrate ~720 LOC** (M1 `fem/refelem.go` ReferenceTriangle + ReferenceQuad + ReferenceTet + ReferenceHex + barycentric/affine maps + Jacobian determinants ~160 LOC; M2 `fem/lagrange.go` P_k Lagrange shape functions on reference simplex via Vandermonde + GLL-collocation k=1..8 ~180 LOC, calls `spectral.GaussLobatto` from 245-S2; M3 `fem/legendre_modal.go` modal hierarchical Legendre basis Babuška-Szabó-1991 `φ_k(x) = (P_k(x) - P_{k-2}(x))/√(4k-2)` integrated-Legendre-bubble ~120 LOC, the canonical hp-FEM basis enabling p-enrichment without rebuilding mass matrices; M4 `fem/quadrature.go` Dunavant-1985 triangle-quadrature 1-20 order + Stroud tetrahedral + tensor-product GLL on quads/hexes ~140 LOC, calls `spectral.GaussLegendre` from 245-S2; M5 `fem/dofmap.go` DOF numbering for P_k continuous + face/edge ownership + cross-point flagging — the load-bearing data structure that mortar/hp algorithms hang off ~120 LOC), **(b) Tier-1 non-conforming + DG ~880 LOC** (M6 `fem/cr.go` Crouzeix-Raviart-1973 P1-NC element on triangles (DOFs at edge midpoints, conformity in the mean) + jump operator `[[u]]_e = u^+ - u^-` + average `{{u}}_e = (u^+ + u^-)/2` ~140 LOC; M7 `fem/dg/sip.go` Symmetric Interior Penalty Galerkin Arnold-1982 / Wheeler-1978 — the canonical DG-elliptic with σ_e = penalty/h ~200 LOC; M8 `fem/dg/bassi_rebay.go` Bassi-Rebay-1997 BR1 + BR2 lifting operators for compressible-NS DG ~140 LOC; M9 `fem/dg/ldg.go` Cockburn-Shu-1998 Local-DG flux for advection-diffusion ~120 LOC; M10 `fem/dg/hdg.go` Hybrid-DG Cockburn-Gopalakrishnan-Lazarov-2009-SINUM-47 with hybridised trace + static condensation reducing global system to face-DOFs only ~200 LOC, the post-2009 DG state of the art; M11 `fem/dg/numerical_flux.go` upwind / Lax-Friedrichs / Roe / Riemann-DG flux library ~80 LOC, shares with 244-D20+D21 hyperbolic Riemann solvers), **(c) Tier-2 mortar interface coupling ~860 LOC** (M12 `fem/mortar/interface.go` MortarInterface{master, slave, projection_quad} ~120 LOC — the central data structure pairing two non-matching meshes along a shared geometric interface; M13 `fem/mortar/projection.go` L²-projection master→slave using slave test functions + master trial functions + Whitney-form cross-validation against 246-X11 ~180 LOC, the Bernardi-Maday-Patera-1994 keystone; M14 `fem/mortar/lagrange_mult.go` discrete Lagrange multiplier space Λ_h on slave-skeleton + saddle-point assembly K=[[A B^T];[B 0]] ~160 LOC; M15 `fem/mortar/cross_points.go` cross-point handling — the famously subtle 2D corner where 4+ subdomains meet and the multiplier space must be reduced to preserve inf-sup ~120 LOC, single-source ownership of Wohlmuth-2001-Acta-Numerica-11 dual-Lagrange-basis at corners; M16 `fem/mortar/dual_basis.go` Wohlmuth-1999 biorthogonal dual basis turning M = Σ ψ_i^* φ_j δ_{ij} into a diagonal matrix invertible without inversion ~140 LOC, the production-grade mortar everyone uses since 2000 because it static-condenses to a positive-definite Schur complement; M17 `fem/mortar/3d.go` 3D mortar with tetrahedral interface + edge-cross + vertex-cross handling ~140 LOC; M18 `fem/mortar/conservation.go` mass-conservation + energy-conservation cross-mortar witnesses ~80 LOC, the keystone validation pin), **(d) Tier-3 hp-adaptivity + a-posteriori ~880 LOC** (M19 `fem/hp/hp_basis.go` element-wise variable-p storage + p-distribution-vector ~80 LOC; M20 `fem/estim/residual.go` Babuška-Rheinboldt-1978 residual estimator η_K² = h_K² ‖f + Δu_h‖² + Σ h_e ‖[[∂_n u_h]]‖² ~140 LOC; M21 `fem/estim/zz.go` Zienkiewicz-Zhu-1987 / SPR-superconvergent-patch-recovery ~140 LOC, the post-1987 production-default-estimator everywhere from ANSYS to Abaqus; M22 `fem/estim/hierarchical.go` Bank-Smith-1993 hierarchical estimator using p+1-enrichment-bubble ~120 LOC; M23 `fem/estim/dwr.go` Dual-Weighted-Residual Becker-Rannacher-2001-Acta-Numerica goal-oriented adaptivity solving adjoint problem + per-element error contributions to user-defined functional J(u) ~200 LOC, the post-2001 frontier-of-frontier adaptive method for engineering quantities of interest; M24 `fem/hp/decision.go` h-vs-p decision policy Mavriplis-1989 / Demkowicz-2007 — at each error-flagged element, decide whether to refine h or enrich p based on smoothness indicator (Legendre-coefficient-decay rate) ~100 LOC; M25 `fem/hp/anisotropic.go` anisotropic refinement deciding edge-direction of refinement using Hessian recovery ~100 LOC), **(e) Tier-4 IGA + VEM + BEM frontier ~660 LOC** (M26 `fem/iga/nurbs_basis.go` Hughes-Cottrell-Bazilevs-2005 Isogeometric Analysis NURBS shape functions + Bézier-extraction operators Borden-Scott-Evans-Hughes-2011 ~220 LOC; M27 `fem/vem/vem_2d.go` Beirão-da-Veiga-Brezzi-Cangiani-Manzini-Marini-Russo-2013 Virtual Element Method on polygonal meshes — projection operator Π^∇ + stabilisation S^E + consistency-stability decomposition ~220 LOC, the post-2013 polygon-FEM workhorse; M28 `fem/bem/galerkin_bem.go` Galerkin Boundary Element Method with single + double layer potential + collocation BEM Sauter-Schwab-2010 ~220 LOC). **SINGULAR-FOUNDATIONAL M1+M2+M5 reference-element + shape-functions + DOF-map ~460 LOC** because every other M-primitive — Crouzeix-Raviart, DG-SIP, mortar-projection, hp-Legendre-modal, IGA, VEM, residual estimators — calls them as substrate; cannot ship anything else without these. **SINGULAR-MOAT M16 Wohlmuth-1999 dual-Lagrange-basis ~140 LOC** because biorthogonal dual basis is the *production-grade* mortar method (used in deal.II MeshWorker, MFEM, FreeFEM, Code_Aster) and turns a constrained-saddle-point problem (indefinite, KKT, expensive) into a positive-definite Schur complement (cheap, conjugate-gradient-friendly) — the engineering-defining decision that made mortar viable beyond research codes; no zero-dep Go library anywhere ships this, no Python (FEniCS lacks it natively, only via dolfin-adjoint extensions), and the canonical reference is Wohlmuth's 2001-Acta-Numerica-11 monograph. **SINGULAR-CUTTING-EDGE M27 VEM Beirão-da-Veiga-2013 ~220 LOC** because Virtual Element Method is the post-2013 frontier of polygonal-FEM (replaces P1-triangular + Q1-quad with arbitrary-polygon + arbitrary-polyhedron meshes; used in Voronoi-mesh-CFD, fractured-rock geomechanics, crystal-plasticity, topology-optimisation) and zero-dep Go absent everywhere; the original paper has 4000+ citations and is the discipline-defining reference. **SINGULAR-PEDAGOGICAL M6 Crouzeix-Raviart ~140 LOC** because CR-NC-P1 is the 1973-canonical-pedagogical-entry-point to non-conforming-FEM (Brezzi-Fortin-1991 "Mixed and Hybrid FEM" §III.2 entry-point) and the closed-form analytical-pin against the Stokes-saddle-point inf-sup-test on a unit-square mesh is the textbook validation. **SINGULAR-2024-FRONTIER M23 DWR Becker-Rannacher-2001 + M10 HDG Cockburn-2009 ~400 LOC** because (a) DWR-goal-oriented-adaptivity is THE 2001-2024 method for engineering-quantity-of-interest adaptivity (every modern aerospace / nuclear / climate adaptive code uses it), and (b) HDG hybridisation reduces global linear-system size by O(p) — the crowning post-2009 DG efficiency result, used in Nektar++ and ngsolve. Both zero-dep Go absent everywhere.

Recommended placement **NEW package `fem/`** ~3,800 LOC (M1-M28 minus Tier-3.5 deferred). Subpackage layout:

```
fem/
  refelem.go          # M1: reference triangle/quad/tet/hex + Jacobian
  lagrange.go         # M2: Lagrange P_k shape functions
  legendre_modal.go   # M3: hierarchical Legendre hp basis
  quadrature.go       # M4: Dunavant + Stroud + tensor-GLL
  dofmap.go           # M5: DOF numbering
  cr.go               # M6: Crouzeix-Raviart P1-NC
  dg/
    sip.go            # M7: Symmetric Interior Penalty
    bassi_rebay.go    # M8: BR1+BR2 lifting
    ldg.go            # M9: Local DG
    hdg.go            # M10: Hybrid DG
    numerical_flux.go # M11: upwind/LF/Roe/Riemann
  mortar/
    interface.go      # M12: MortarInterface struct
    projection.go     # M13: L²-projection master↔slave
    lagrange_mult.go  # M14: saddle-point assembly
    cross_points.go   # M15: Wohlmuth-2001 cross-point
    dual_basis.go     # M16: biorthogonal dual basis
    threed.go         # M17: 3D mortar
    conservation.go   # M18: mass+energy witnesses
  hp/
    hp_basis.go       # M19: variable-p storage
    decision.go       # M24: h-vs-p Mavriplis-Demkowicz
    anisotropic.go    # M25: anisotropic refinement
  estim/
    residual.go       # M20: Babuška-Rheinboldt
    zz.go             # M21: Zienkiewicz-Zhu SPR
    hierarchical.go   # M22: Bank-Smith
    dwr.go            # M23: dual-weighted-residual
  iga/
    nurbs_basis.go    # M26: NURBS Bezier-extraction
  vem/
    vem_2d.go         # M27: Virtual Element Method
  bem/
    galerkin_bem.go   # M28: Galerkin BEM
```

Rationale for **NEW** `fem/` rather than nesting under `pde/fem/` per 244 layout: FEM-as-a-discipline outgrows the PDE-solver wrapper once you add non-conforming + hp + DG + IGA + VEM + BEM (the seven canonical FEM-frontier extensions) — the package becomes its own subdomain with its own internal hierarchy (reference-element / shape-function / DOF-map substrate; conforming / non-conforming / DG / hybrid discretisations; h / p / hp / anisotropic adaptivity; residual / ZZ / hierarchical / DWR estimators). Top-level `fem/` allows clean import edges from `pde/` (PDE consumers), `geometry/dec/` (mortar Whitney bridge to 246-X11/X16), `optim/` (shape-optimisation 251), `signal/` (no consumer), and parallel sibling-of-`pde/` placement matches deal.II / MFEM / FreeFEM / FEniCS layout conventions where FEM is its own top-level subdomain. Sub-package precedent inside reality: `prob/copula/`, `prob/conformal/`, `optim/proximal/`, `optim/transport/`, `geometry/dec/` (proposed in 246), `topology/persistent/` (existing). This slot ships ~3,800 LOC across 6 PRs / 18-22 engineer-days; deferred Tier-4 IGA-3D + VEM-3D + BEM-Maxwell ~600 LOC.

**CANDOR.** Mortar / hp-FEM / DG / IGA / VEM ranks as the **highest-LOC most-specialised single Block-C slot in the 240-249 cluster** but the consumer-pull inside reality is **near-zero today** — no aicore / Pistachio / Oracle / Sentinel service uses mortar coupling, hp-adaptivity, or IGA, and there is no PDE substrate (244 must ship first) on which to even *demonstrate* these methods. deal.II is 800k LOC, MFEM is 250k LOC, FEniCS is 500k LOC, FreeFEM is 200k LOC — reality cannot match (and should not try to match) any of these in scope. The case for shipping at all rests on: (1) mathematical-completeness — the Block-C frontier of computational PDE methods named in MASTER_PLAN; (2) downstream-ready when other slots ship — 248-multigrid + 249-domain-decomp + 250-contact + 251-shape-opt all explicitly compose on this; (3) brand-prestige — first zero-dep Go library to ship Wohlmuth-1999 dual-mortar + Beirão-2013 VEM + Becker-Rannacher-2001 DWR. Recommendation: **ship Tier-0 + Tier-1 + Tier-2 (M1-M18 ~2,460 LOC over 4 PRs / 10-12 engineer-days)** which gives reality first-class non-conforming-CR + DG-SIP/BR/LDG/HDG + 2D-mortar-Wohlmuth-dual — the production-grade FEM-frontier substrate. **Defer Tier-3 + Tier-4 (M19-M28 ~1,540 LOC)** as a follow-up unless concrete consumer-demand emerges (e.g., aicore picks up an FE-based plasma physics or topology-optimisation use case). **Cheapest-1-day-shippable**: M1 + M2 + M5 + M6 ~520 LOC ships ReferenceTriangle + Lagrange-P_k + DOF-map + Crouzeix-Raviart-P1-NC saturating R-CR-INF-SUP 3/3 against Stokes-on-unit-square analytical inf-sup constant ≥ 1/√2 (Brezzi-Fortin-1991-§III.2-Thm-2.4 closed-form pin) — but blocked-on-244-D25-D26 (need at least one *conforming* P1 FEM to compare CR against). **Highest-leverage-1-week-unlock**: PR-1 + PR-2 ~1,540 LOC M1-M11 saturating reference-element + Lagrange + Legendre-hp-modal + Dunavant-quadrature + DOF-map + Crouzeix-Raviart + SIP-DG + Bassi-Rebay + LDG + HDG + numerical-flux — the *complete* DG/non-conforming foundation. **22 of 28 primitives unique to this slot** (M2-Lagrange shape functions cross-consumed by 244-D26-P1-2D, M4-Dunavant-quadrature cross-consumed by 244-D26 + 246-X14 Galerkin-Hodge mass, M11-numerical-flux cross-shared with 244-D20+D21 hyperbolic Riemann solvers, M13-projection cross-shared with 246-X11 Whitney-form + X16 mortar-precursor pullback, M19-M25 hp + estimators stand alone here). Six primitives explicitly cross-link other 240-249 slots (M1+M2 → 244-FEM, M3+M4 → 245-spectral-element, M11 → 244-Riemann, M12-M16 → 246-DEC-X11/X16, M19-M25 → 248-multigrid p-coarsening, M27 → 246-X25 polygonal-DEC); 248-multigrid + 249-domain-decomp + 250-contact + 251-shape-opt all explicitly downstream consumers of M12-M18.

---

## 0. State at HEAD (2026-05-09, v0.10.0) — verified by direct read

Repo-wide audit for mortar / non-conforming / hp / DG / IGA / VEM / BEM / FEEC / posteriori-error-estimator / Lagrange-multiplier-FE / Crouzeix-Raviart / Wohlmuth / Bernardi-Maday-Patera / Bassi-Rebay / Cockburn / Hughes-Cottrell-Bazilevs / Beirão-da-Veiga / Wachspress / mean-value-coordinates / Bramble-Pasciak / Demkowicz / Babuška / Zienkiewicz-Zhu / Becker-Rannacher / Hesthaven-Warburton / hp-FEM / hp-DG / spectral-element-coupling / cross-points / dual-Lagrange-basis surface — **zero callable matches** anywhere in `*.go` files outside review-corpus.

| Surface | Path | Mortar/hp-FEM relevance |
|---|---|---|
| `pde/fem/p1_1d.go` D25 (slot 244) | not yet shipped | P1 conforming 1D — STRICT-UPSTREAM for 247 |
| `pde/fem/p1_2d.go` D26 (slot 244) | not yet shipped | P1 conforming 2D triangular — STRICT-UPSTREAM for 247 |
| `linalg/gmres.go` D27 (slot 244) | not yet shipped | non-symmetric Krylov for saddle-point — STRICT-UPSTREAM for M14 |
| `geometry/dec/whitney.go` X11 (slot 246) | not yet shipped | Whitney-form interpolation — bridge for M13 mortar-projection |
| `geometry/dec/complex.go` X1 (slot 246) | not yet shipped | SimplicialComplex2D — substrate for M5 DOF-map |
| `geometry/dec/derivative.go` X4 (slot 246) | not yet shipped | signed-incidence d_k — substrate for M6 CR jump operator |
| `spectral/poly.go` S1 (slot 245) | not yet shipped | Legendre + Jacobi recurrence — substrate for M3 hierarchical-modal |
| `spectral/nodes.go` S2 (slot 245) | not yet shipped | GLL + GL nodes via Golub-Welsch — substrate for M2+M4 |
| `spectral/cheb_diff.go` S12 (slot 245) | not yet shipped | Chebyshev-differentiation-matrix — substrate for SEM coupling |
| `linalg.LUSolve` | `linalg/decompose.go` | Dense direct — usable for small-N FE assembly but unsuitable for sparse mortar saddle-point (M14 needs GMRES from 244-D27) |
| `linalg.QRAlgorithm` | `linalg/eigen.go` | Symmetric eigensolver — usable for inf-sup constant computation (Babuška-Brezzi LBB test in M18 conservation witnesses) |
| `linalg/sparse/` | -- | **ABSENT** — flagged 097-T1; blocks every realistic-N mortar/hp problem (dense-LU O(n³) blows up at N=10⁴ DOFs) |
| `topology/persistent/vr.go.Filtration` | abstract simplicial complex over F_2 | unusable for FE — no embedded coords, no ℝ-arithmetic |
| `geometry/polygon.go.TriangleArea2D` | signed area | usable as the volume 2-form for M1 ReferenceTriangle Jacobian |
| `calculus.GaussLegendre` | 5-point cap | inadequate for hp-FEM — needs Dunavant 1-20-order (M4) and arbitrary-N GLL (245-S2) |
| `chaos.RK4Step` | only time-stepper | usable for M9 LDG advection-diffusion explicit time-marching |
| Crouzeix-Raviart 1973 P1-NC | -- | **ABSENT** — M6 ships |
| Discontinuous Galerkin (SIP / BR / LDG / HDG) | -- | **ABSENT** — M7-M10 ship |
| Mortar interface coupling Bernardi-Maday-Patera-1994 | -- | **ABSENT** — M12-M18 ship |
| Wohlmuth-1999 biorthogonal dual-Lagrange basis | -- | **ABSENT** — M16 ships (production-grade keystone) |
| Cross-points Wohlmuth-2001-Acta-Numerica-11 | -- | **ABSENT** — M15 ships (subtle 2D corner handling) |
| hp-adaptivity decision Mavriplis-1989 / Demkowicz-2007 | -- | **ABSENT** — M24 ships |
| A-posteriori estimators (residual / ZZ / hierarchical / DWR) | -- | **ABSENT** — M20-M23 ship |
| Anisotropic mesh refinement | -- | **ABSENT** — M25 ships |
| Isogeometric Analysis Hughes-Cottrell-Bazilevs-2005 NURBS | -- | **ABSENT** — M26 ships |
| Virtual Element Method Beirão-da-Veiga-2013 | -- | **ABSENT** — M27 ships |
| Galerkin Boundary Element Method | -- | **ABSENT** — M28 ships |
| FEM-BEM coupling | -- | **ABSENT** — deferred (cross-link 062-em) |
| Hierarchical-modal Legendre Babuška-Szabó-1991 hp-basis | -- | **ABSENT** — M3 ships |

**Cross-import edges that this slot creates:**
- `fem → pde/fem` (244-D25-D26, NEW) for conforming P1 baseline
- `fem → spectral` (245-S1-S2-S12, NEW) for hp-Legendre-modal + GLL nodes + Chebyshev-differentiation-matrix
- `fem → geometry/dec` (246-X1-X4-X11-X16, NEW) for SimplicialComplex2D + d_k + Whitney-form bridge
- `fem → linalg/gmres` (244-D27, NEW) for saddle-point Lagrange-multiplier solve
- `fem → linalg/sparse` (097-T1, NEW) for realistic-N
- `fem → calculus.NumericalGradient` (PRESENT) for adjoint problem in M23 DWR
- `fem → chaos.RK4Step` (PRESENT) for M9 LDG explicit time-marching

**Strict downstream consumers** of `fem/` substrate (already specified or anticipated):
- `pde/multigrid/` (slot 248) consumes M19-M25 hp-coarsening for hp-multigrid V-cycle
- `pde/domdecomp/` (slot 249) consumes M12-M16 mortar-interface for non-overlapping Schwarz + FETI-DP cross-points (M15)
- `mortar/contact/` (slot 250) consumes M13 L²-projection for gap-function discretisation in Signorini contact
- `optim/shape/` (slot 251) consumes M20-M25 a-posteriori estimators + M23 DWR for goal-oriented shape derivative

---

## 1. The twenty-eight primitives

For each: **(a) what reality has (zero, modulo 244+245+246 cross-link)**, **(b) what to add (specialised — be candid)**, **(c) connective-tissue LOC**, **(d) blocker**.

### Tier-0 — Reference-element + shape-function substrate (~720 LOC, blocks-on-244+245)

#### M1 — ReferenceTriangle / ReferenceQuad / ReferenceTet / ReferenceHex
(a) Nothing. `geometry/polygon.go.TriangleArea2D` ships the signed-area = volume 2-form on a 2-simplex but is unlabelled and not part of an FE reference-element abstraction.
(b) `fem/refelem.go`: `ReferenceTriangle{}` with vertices {(0,0),(1,0),(0,1)}, barycentric coords λ_i, affine map T_K(ξ)=B_K ξ + b_K from reference to physical, Jacobian determinant det(B_K). Same for Quad{[-1,1]²}, Tet (4-vertex unit), Hex {[-1,1]³}. Helpers: `MapToPhysical(refelem, coeffs, ξ)`, `JacobianAt(refelem, ξ)`, `InverseMap(refelem, x)` for newton-iteration on isoparametric.
(c) ~160 LOC.
(d) **No blocker.**

#### M2 — Lagrange P_k shape functions on reference simplex
(a) Nothing. 244-D25-D26 ship P1 only.
(b) `fem/lagrange.go`: P_k for k=1..8 via Vandermonde-on-GLL-collocation-points solve. `LagrangeBasis(refelem, k, i)(ξ)` returns φ_i(ξ); Lagrange-property φ_i(x_j)=δ_{ij}; gradient via chain rule. Calls `spectral.GaussLobatto(k+1)` (245-S2) for nodal placement.
(c) ~180 LOC.
(d) **Blocker:** 245-S2 GLL nodes.

#### M3 — Hierarchical modal Legendre basis Babuška-Szabó-1991
(a) Nothing.
(b) `fem/legendre_modal.go`: φ_0(x)=(1-x)/2, φ_1(x)=(1+x)/2 (vertex modes); φ_k(x) = (P_k(x)-P_{k-2}(x))/√(4k-2) for k≥2 (integrated-Legendre-bubble, vanishes at ±1). 2D bubble basis on quad via tensor product, on triangle via Karniadakis-Sherwin-1995-warped-tensor-product. The canonical hp-FEM basis: enables p-enrichment by *adding* basis functions without rebuilding mass matrices for existing p-modes (mass matrix is *banded* in modal-Legendre, dense in nodal-Lagrange). Critical for efficient hp.
(c) ~120 LOC.
(d) **Blocker:** 245-S1 Legendre recurrence.

#### M4 — Dunavant + Stroud + tensor-GLL quadrature
(a) `calculus.GaussLegendre` 5-point cap on 1D `[-1,1]`. Inadequate for triangle/tet integration of P_k×P_k products which need exact-integration order 2k.
(b) `fem/quadrature.go`: Dunavant-1985 symmetric triangle quadrature orders 1-20 (look-up table of (λ_1,λ_2,λ_3,w) tuples); Stroud-1971 tetrahedral quadrature orders 1-8; tensor-product GLL on quads via `spectral.GaussLobatto`; tensor-product GL on hexes via `spectral.GaussLegendre`.
(c) ~140 LOC (mostly precomputed Dunavant tables — lifted from Dunavant-1985-IJNME-21 paper Table A).
(d) **Blocker:** 245-S2 GLL nodes for quad/hex.

#### M5 — DOF map + face/edge ownership + cross-point flagging
(a) Nothing. 246-X1 SimplicialComplex2D (when shipped) provides Edges + Tris connectivity but not DOF numbering for P_k continuous (which needs interior-edge + interior-face DOFs ordered consistently across element-sharing).
(b) `fem/dofmap.go`: `DofMap{global_to_local, local_to_global, owner_element, face_ownership}` for continuous-Lagrange-P_k or modal-hierarchical. Cross-point flagging `IsCrossPoint(vertex)` = true if 4+ subdomains share vertex (the Wohlmuth-2001 special case, M15 needs this). Hanging-node tracking for h-refinement (M24).
(c) ~120 LOC.
(d) **Blocker:** 246-X1 SimplicialComplex2D.

### Tier-1 — Non-conforming + DG (~880 LOC)

#### M6 — Crouzeix-Raviart 1973 P1-NC element on triangles
(a) Nothing.
(b) `fem/cr.go`: DOFs at edge midpoints {x_e} (one per edge, total = n_1); shape functions φ_e(x_{e'}) = δ_{ee'}; conformity in the mean (∫_e [[u_h]] = 0 not pointwise); jump operator `Jump(u, edge) = u^+ - u^-` and average `Avg(u, edge) = (u^+ + u^-)/2`. Key application: Stokes velocity-pressure pair (CR-P1-NC velocity × P0 pressure) is inf-sup-stable on quasi-uniform meshes (Brezzi-Fortin-1991-§III.2-Thm-2.4) — saturates R-CR-INF-SUP 3/3 against Stokes-unit-square analytical inf-sup ≥ 1/√2.
(c) ~140 LOC.
(d) **Blocker:** M1 + 246-X1 + 246-X4.

#### M7 — Symmetric Interior Penalty Galerkin Arnold-1982 / Wheeler-1978
(a) Nothing.
(b) `fem/dg/sip.go`: Bilinear form a_h(u,v) = ∫_Ω ∇u·∇v − Σ_{e∈Γ_h} ∫_e {{∂_n u}}[[v]] − Σ_{e∈Γ_h} ∫_e {{∂_n v}}[[u]] + Σ_{e∈Γ_h} (σ_e/h_e) ∫_e [[u]][[v]]. Symmetric (the SIP — alternative is NIPG non-symmetric). Coercivity requires σ_e ≥ σ_min(d, k) — provide `MinPenaltyParameter(d, k)` from Epshteyn-Riviere-2007 sharp bounds.
(c) ~200 LOC.
(d) **Blocker:** M1 + M2 + M5.

#### M8 — Bassi-Rebay 1997 BR1 + BR2 lifting operators
(a) Nothing.
(b) `fem/dg/bassi_rebay.go`: BR1 — replace ∇u_h with R_h(u_h) = ∇u_h + L([[u_h]]) where L is the global lifting operator from face-jumps to volume gradient. BR2 — local lifting δ_e supported only on the two elements sharing edge e. Used in compressible-Navier-Stokes DG (the original Bassi-Rebay-1997 paper). Provides M_h^{-1} as preconditioner.
(c) ~140 LOC.
(d) **Blocker:** M1 + M2 + M5 + M11.

#### M9 — Local DG Cockburn-Shu-1998
(a) Nothing.
(b) `fem/dg/ldg.go`: Mixed formulation u_h, q_h = ∇u_h with element-local elimination of q_h via static condensation. Numerical fluxes from Cockburn-Shu-1998-SINUM-35: q̂ = {{q_h}} − C_{12}[[q_h]] + C_{12}[[u_h]] (the (C_{11},C_{12}) parameter family). Used for advection-diffusion-reaction.
(c) ~120 LOC.
(d) **Blocker:** M1 + M2 + M5 + M11.

#### M10 — Hybrid DG Cockburn-Gopalakrishnan-Lazarov-2009-SINUM-47
(a) Nothing.
(b) `fem/dg/hdg.go`: Introduce hybrid trace λ_h on edges + local DG inside elements; static-condense local DG out to get a global system in λ_h only — global-system-size O(n_face·p) instead of O(n_element·p²). The post-2009 DG state-of-the-art used in Nektar++ and ngsolve. Reduces global LU cost by O(p) — an asymptotically-defining efficiency gain for high p.
(c) ~200 LOC.
(d) **Blocker:** M1 + M2 + M5 + M11 + 244-D27 GMRES.

#### M11 — Numerical flux library
(a) 244-D20 HLL + D21 HLLC + D22 Godunov + D23 MUSCL ship the *finite-volume* Riemann solvers; reuse them.
(b) `fem/dg/numerical_flux.go`: thin wrappers exposing the same Riemann functions with FE-DG signature `Flux(u_left, u_right, n) → ĥ(u_left, u_right, n)`. Upwind / Lax-Friedrichs / Roe / HLL / HLLC for advection / Burgers / Euler / NS.
(c) ~80 LOC (mostly re-exports).
(d) **Blocker:** 244-D20 + D21 + D22 + D23 (or inline mini-version).

### Tier-2 — Mortar interface coupling Bernardi-Maday-Patera-1994 (~860 LOC)

#### M12 — MortarInterface data structure
(a) Nothing.
(b) `fem/mortar/interface.go`: `MortarInterface{Master, Slave, ProjectionQuad, MasterSide, SlaveSide}` pairing two non-matching meshes along a shared geometric interface Γ. The master side carries the trial-function space, slave side the test-function (Lagrange-multiplier) space. Helpers: `BuildFromMeshes(mesh1, mesh2, interface_edges)`, `MasterTraceDof(interface, dof)`, `SlaveTraceDof(interface, dof)`. Cross-link 246-X16 mortar-precursor pullback.
(c) ~120 LOC.
(d) **Blocker:** M5 + 246-X1.

#### M13 — L²-projection master→slave
(a) Nothing.
(b) `fem/mortar/projection.go`: Compute Π: V_master|_Γ → V_slave|_Γ such that ∫_Γ (Π u − u) v_h = 0 ∀v_h ∈ V_slave. Standard mortar projection from Bernardi-Maday-Patera-1994. Mass matrix M^{slave-slave}_{ij} = ∫_Γ φ_i^slave φ_j^slave; coupling matrix C_{ij} = ∫_Γ φ_i^slave φ_j^master; then Π = M^{-1} C. For dual-basis (M16) M is diagonal. Witness via 246-X11 Whitney-form cross-validation (mortar projection of a 1-form should agree with Whitney pullback to round-off).
(c) ~180 LOC.
(d) **Blocker:** M1 + M2 + M4 + M12 + 246-X11.

#### M14 — Discrete Lagrange-multiplier saddle-point assembly
(a) Nothing.
(b) `fem/mortar/lagrange_mult.go`: Saddle-point system K = [[A B^T]; [B 0]] where A is the standard FE stiffness, B is the M13 coupling matrix Bᵢⱼ = ∫_Γ φ_i^master φ_j^Λ − ∫_Γ φ_i^slave φ_j^Λ, and Λ_h ⊂ L²(Γ) is the multiplier space (typically dual-basis from M16). Solve via GMRES (244-D27) or via static-condensation Schur-complement S = A_master − B^T A^{-1} B (positive-definite if dual-basis used).
(c) ~160 LOC.
(d) **Blocker:** M12 + M13 + 244-D27.

#### M15 — Cross-point handling Wohlmuth-2001-Acta-Numerica-11
(a) Nothing.
(b) `fem/mortar/cross_points.go`: At a cross-point x_c where 4+ subdomains meet, the multiplier space must be reduced (one-fewer DOF) to preserve the Babuška-Brezzi inf-sup condition. Detection via M5 cross-point flag; reduction via DOF aggregation: drop one of the cross-point DOFs and replace with a constraint. The famously subtle 2D corner — naïve mortar with 4 subdomains gives a singular saddle-point matrix without this fix. Single-source-ownership of Wohlmuth-2001-AN-11 §3.4 algorithm.
(c) ~120 LOC.
(d) **Blocker:** M5 + M12 + M14.

#### M16 — Wohlmuth-1999 biorthogonal dual-Lagrange basis
(a) Nothing.
(b) `fem/mortar/dual_basis.go`: Build {ψ_i} dual-basis on slave-side such that ∫_Γ ψ_i φ_j = δ_{ij} (biorthogonality). With dual basis, mass matrix M = I diagonal, projection Π = C trivially-applicable, and the saddle-point Schur complement S = A_master − B^T A^{-1} B is positive-definite-symmetric (so CG works, no GMRES needed). The production-grade-mortar everyone has used since 2000 — used in deal.II MeshWorker, MFEM, FreeFEM, Code_Aster. Construction via local 1-element solve `ψ_i = Σ_j a_ij φ_j` with a_ij from local mass-matrix-inversion on the slave-element.
(c) ~140 LOC.
(d) **Blocker:** M1 + M2 + M4 + M12.

#### M17 — 3D mortar
(a) Nothing.
(b) `fem/mortar/threed.go`: 3D mortar with tetrahedral interface + edge-cross + vertex-cross handling. The 3D analog of M15 cross-points has additional edge-cross-points (1D entities where 3+ subdomains meet) needing further multiplier reduction. Wohlmuth-2001-AN-11 §4 generalisation.
(c) ~140 LOC.
(d) **Blocker:** M12-M16 + 246-X1 SimplicialComplex3D.

#### M18 — Mass + energy conservation cross-mortar witnesses
(a) Nothing.
(b) `fem/mortar/conservation.go`: Discrete-mass-conservation `∫_Γ_master u_master = ∫_Γ_slave Π u_master` to round-off (only true if M16 dual-basis used), discrete-energy-conservation `‖u_master‖_Γ ≥ ‖Π u_master‖_Γ` (projection is non-expansive). Provides R-MORTAR-CONSERVATION 3/3 cross-validation pin. Inf-sup constant computation via Babuška-Brezzi LBB test using `linalg.QRAlgorithm` symmetric-eigensolver on Schur-complement `S = B M^{-1} B^T` smallest eigenvalue.
(c) ~80 LOC.
(d) **Blocker:** M12-M16.

### Tier-3 — hp-adaptivity + a-posteriori (~880 LOC)

#### M19 — Variable-p storage
(a) Nothing.
(b) `fem/hp/hp_basis.go`: Element-wise `p[K]` polynomial-order vector, with hp-DOF-map indexing into shared edges/faces using `min(p_K, p_K')` lower-of-neighbours rule (continuity at interfaces). Helpers: `RefineP(elem, +1)`, `CoarsenP(elem, -1)`, `EnforceContinuity(dofmap, p_per_elem)`.
(c) ~80 LOC.
(d) **Blocker:** M3 + M5.

#### M20 — Babuška-Rheinboldt-1978 residual estimator
(a) Nothing.
(b) `fem/estim/residual.go`: Per-element estimator η_K² = h_K² ‖f + Δu_h‖²_K + Σ_{e⊂∂K} h_e ‖[[∂_n u_h]]‖²_e. Global reliability `‖u−u_h‖ ≤ C Σ_K η_K²` (Verfürth-1996 sharp constants). The 1978-canonical-pedagogical-residual-estimator.
(c) ~140 LOC.
(d) **Blocker:** M1-M5 + M6 (jump operator).

#### M21 — Zienkiewicz-Zhu-1987 / SPR superconvergent-patch-recovery
(a) Nothing.
(b) `fem/estim/zz.go`: Recover smoother gradient G_h u_h by polynomial-fit on patches of elements + comparison to ∇u_h: η_K² = ‖G_h u_h − ∇u_h‖²_K. Zienkiewicz-Zhu-1992-IJNME-33 Superconvergent-Patch-Recovery (SPR) variant. Production-default-estimator everywhere from ANSYS to Abaqus, used since 1987 because it works on virtually any FE problem without problem-specific tuning.
(c) ~140 LOC.
(d) **Blocker:** M1-M5.

#### M22 — Bank-Smith-1993 hierarchical estimator
(a) Nothing.
(b) `fem/estim/hierarchical.go`: Solve auxiliary problem in p+1-enrichment-bubble space (M3 enables this with banded mass matrices). Bank-Smith-1993-SISC-14 hierarchical estimator. Tighter than M20-residual for elliptic problems with smooth solutions.
(c) ~120 LOC.
(d) **Blocker:** M3 + M5.

#### M23 — Dual-Weighted-Residual Becker-Rannacher-2001-Acta-Numerica
(a) Nothing.
(b) `fem/estim/dwr.go`: Goal-oriented estimator for user-defined functional J(u). Solve adjoint problem a(v, z_h) = J'(u_h)(v) ∀v on enriched space. Per-element η_K = ρ(u_h)(z_h - I_h z_h)|_K weighted residual where ρ is the primal-residual. The post-2001-frontier-of-frontier adaptive method for engineering quantities of interest (lift, drag, pointwise-stress) used in every modern aerospace / nuclear / climate adaptive code. Cross-link `calculus.NumericalGradient` for J'(u_h) when J is user-supplied non-linear.
(c) ~200 LOC.
(d) **Blocker:** M1-M5 + M19 (p-enrichment for adjoint) + 244-D27 GMRES.

#### M24 — h-vs-p decision policy Mavriplis-1989 / Demkowicz-2007
(a) Nothing.
(b) `fem/hp/decision.go`: At each error-flagged element K, decide h-refine vs p-enrich based on smoothness indicator. Mavriplis-1989: estimate Legendre-coefficient-decay-rate `|a_n| ~ exp(−σ n)`; if σ > σ_smooth threshold (≈ 1) → enrich p, else split h. Demkowicz-2007 isotropic+anisotropic generalisation.
(c) ~100 LOC.
(d) **Blocker:** M3 + M19 + M20-M23.

#### M25 — Anisotropic mesh refinement
(a) Nothing.
(b) `fem/hp/anisotropic.go`: Estimate Hessian H = ∇∇u_h via SPR (M21 reuses). Decide edge-direction of refinement aligning with smallest-eigenvalue-direction of H (refine perpendicular to it — i.e., refine in the steepest-gradient direction). Apel-1999-Verfürth-1996 anisotropic-residual-estimator on stretched meshes.
(c) ~100 LOC.
(d) **Blocker:** M21.

### Tier-4 — IGA + VEM + BEM frontier (~660 LOC)

#### M26 — Isogeometric Analysis Hughes-Cottrell-Bazilevs-2005 NURBS
(a) Nothing.
(b) `fem/iga/nurbs_basis.go`: NURBS shape functions R_i^p(ξ) = N_i^p(ξ) w_i / Σ_j N_j^p(ξ) w_j. Knot-vector storage; B-spline recurrence (Cox-de-Boor); Bézier-extraction operator C_e mapping NURBS → element-local Bernstein-Bézier (Borden-Scott-Evans-Hughes-2011-IJNME-87) for legacy-FE-code compatibility. Refinement: knot-insertion (h), order-elevation (p), k-refinement (knot-insert+order-elevate, the IGA-unique-mode). The exact-CAD-geometry FEM bridge — used in Bazilevs-Calo-Cottrell-Hughes-Reali-Scovazzi-2007 Navier-Stokes.
(c) ~220 LOC.
(d) **Blocker:** M1 + M5.

#### M27 — Virtual Element Method Beirão-da-Veiga-2013
(a) Nothing.
(b) `fem/vem/vem_2d.go`: VEM on arbitrary-polygon meshes (no triangulation needed). For each polygon E: projection operator Π^∇_E mapping virtual-trial-functions to polynomials of degree k; local-stiffness `K_E = (Π^∇)^T A^k Π^∇ + S_E (I − Π^∇)^T (I − Π^∇)` where S_E is the stabilisation (Beirão-2013 dofi-dofi). The post-2013-polygon-FEM workhorse with 4000+ citations: used in Voronoi-mesh-CFD, fractured-rock geomechanics, crystal-plasticity, topology-optimisation. Cross-link 246-X25 de-Goes-Crane-2016 polygonal-DEC.
(c) ~220 LOC.
(d) **Blocker:** M1 + M2 + M4 + M5 + 246-X25.

#### M28 — Galerkin Boundary Element Method
(a) Nothing.
(b) `fem/bem/galerkin_bem.go`: For Laplace BVP u | ∂Ω = g via boundary integral u(x) = ∫_∂Ω G(x,y) ∂_n u(y) dS_y − ∫_∂Ω ∂_n G(x,y) u(y) dS_y where G(x,y)=−1/(2π) log|x−y| (2D) or 1/(4π|x−y|) (3D). Galerkin-discretised via P1 basis on ∂Ω + Sauter-Schwab-2010 singular-quadrature for log-singularity. Single-layer + double-layer potential operators. FEM-BEM coupling deferred (cross-link 062-em-missing).
(c) ~220 LOC.
(d) **Blocker:** M1 + M2 + M4 + 246-X1.

---

## 2. Cross-link audit vs neighbouring slots

### 244 (PDE solvers / FEM scaffolding)
247 is **strict-downstream of 244** — needs D25 P1-FE-1D + D26 P1-FE-2D as the conforming-baseline that CR (M6) and DG (M7-M10) compare against. 244 ships P1; 247 ships P_k + non-conforming + DG + mortar + hp on top of P1.

### 245 (Spectral methods)
247 cross-links 245 for: S1 Legendre-recurrence (M3 hp-modal); S2 GLL-nodes (M2 Lagrange + M4 quadrature); S12 Chebyshev-differentiation-matrix (SEM coupling, deferred). The natural composition `spectral` (single-block high-order) + `fem/mortar` (cross-block coupling) = full hp-SEM.

### 246 (Discrete exterior calculus)
247 strongly cross-links 246 for: X1 SimplicialComplex2D (M5 DOF-map substrate); X4 d_k signed-incidence (M6 CR jump operator); X11 Whitney-form (M13 mortar-projection bridge — **mortar projection ↔ Whitney pullback** is the 246/247 unifying duality, validated to round-off as R-MORTAR-WHITNEY-DUALITY 3/3); X16 mortar-precursor pullback (M12 interface struct).

### 248 (Multigrid)
247 is **strict-upstream of 248** for hp-multigrid p-coarsening — V-cycle prolongation/restriction across p-levels uses M19 hp-basis + M3 hierarchical-modal (banded mass matrices enable cheap p-coarsening). Algebraic multigrid (AMG) on saddle-point mortar systems composes on M14+M16 dual-basis + Schur-complement.

### 249 (Domain decomposition)
247 is **strict-upstream of 249** for non-overlapping Schwarz = mortar (M12-M16) and FETI-DP (Farhat-Lesoinne-LeTallec-Pierson-Rixen-2001) cross-points = M15. The dual-primal-FETI-DP method specifically chose primal cross-point DOFs to avoid the multiplier reduction M15 needs; both algorithms are dual to each other.

### 250 (Mortar contact)
247 is **strict-upstream of 250** for gap-function discretisation in Signorini contact — M13 L²-projection of master-surface to slave-surface gives the discrete gap g_h, and M16 dual-basis enables the KKT-complementarity LCP solver to work efficiently (Hüeber-Wohlmuth-2005-CMAME). Slot 250 is mostly the contact-mechanics specialisation on top of this slot's coupling primitives.

### 251 (Shape optimisation)
247 cross-links 251 via M23 DWR — shape derivative is naturally formulated as a goal-functional J(u, Ω) and goal-oriented adaptivity guides the mesh-deformation. `optim/shape/` will consume M23.

### 156 (Persistent homology) and 097 (Linalg)
247 cross-links 156 only via 246-X1 (mortar interface graph is a 1-cochain on shared edges, validated as F_2-homology cycle in the trivial limit); 247 strongly blocks-on 097-T1 SparseMatrix + 097-T2 SparseSolve for any realistic-N — dense-LU on a 100k-DOF mortar saddle-point is O(10¹⁵) operations, untenable beyond toy meshes.

---

## 3. Reproducible R-pins (mortar-FEM specific)

### R-CR-INF-SUP 3/3
Crouzeix-Raviart × P0-pressure on Stokes-unit-square mesh N×N, N=4..32. Compute inf-sup constant β_h = √(λ_min(B M^{-1} B^T)) via M18 + `linalg.QRAlgorithm` smallest-eigenvalue. Pin β_h ≥ 1/√2 (Brezzi-Fortin-1991-§III.2-Thm-2.4 closed-form lower bound). Cross-validate against analytical inf-sup constant for a uniform-mesh Stokes-saddle-point.

### R-DG-SIP-CONVERGENCE 4/4
SIP-DG (M7) on Poisson `−Δu = f` with `f = 2π² sin(πx) sin(πy)` on unit-square Dirichlet-zero, exact solution `u = sin(πx) sin(πy)`. Pin h-convergence ‖u−u_h‖_{H1} = O(h^p) for p=1,2,3,4 polynomial degree to 1e-4 over h ∈ {1/8, 1/16, 1/32, 1/64}. Saturates DG order-of-accuracy.

### R-MORTAR-CONSERVATION 3/3
Two-mesh mortar setup: mesh A (refined, master) × mesh B (coarse, slave) sharing a common interface Γ. Project `u(x,y) = sin(πx)cos(πy)` from A to B via M13 + M16 dual-basis. Pin (a) ∫_Γ u_A − ∫_Γ Π u_A = 0 to round-off (mass conservation); (b) ‖Π u_A‖_Γ ≤ ‖u_A‖_Γ (non-expansiveness); (c) cross-link 246-X11 Whitney-pullback agreement to round-off.

### R-MORTAR-WHITNEY-DUALITY 3/3 (cross-link 246)
On uniform-mesh refinement chain h_n=h/2^n, Whitney-projection (246-X11) of a smooth 1-form ω onto cochains agrees with mortar-projection (M13) of the corresponding nodal interpolant to O(h²). Cross-validates 246-247 unification.

### R-DWR-EFFICIENCY 4/4
DWR estimator (M23) on Poisson-on-L-shape with goal J(u) = u(x_target) at a corner-singularity-distant interior point. Pin (a) global energy-norm-error vs M20 residual estimator agrees to factor 2 (efficiency); (b) DWR effectivity-index = estimated-J-error / true-J-error ∈ [0.5, 2.0] uniformly under refinement; (c) DWR-guided refinement reduces J-error by factor 100 in ≤ 5 levels vs uniform-refinement needing 8+ levels.

### R-VEM-PATCH-TEST 3/3 (M27)
VEM on Voronoi mesh of unit-square. Patch-test: exact reproduction of polynomials of degree k. Pin (a) for u = x + 2y + 3xy (k=2), VEM solution agrees to round-off; (b) energy-norm-error ‖u − u_h‖_{H1} = O(h^k) on smooth Helmholtz problem; (c) condition-number of K^E independent of polygon-shape (Beirão-Lovadina-Russo-2017).

---

## 4. PR sequencing (6 PRs over 18-22 engineer-days)

**PR-1 ~720 LOC / 4 engineer-days** — Tier-0 substrate M1+M2+M3+M4+M5. Saturates R-CR-INF-SUP 3/3 (with M6 in PR-2). Blocks-on 245-S1+S2 + 246-X1.

**PR-2 ~340 LOC / 2 engineer-days** — Crouzeix-Raviart M6 + numerical flux M11. Saturates R-CR-INF-SUP 3/3 + connects to 244-D20-D23 Riemann library.

**PR-3 ~660 LOC / 3 engineer-days** — DG family M7+M8+M9. SIP + Bassi-Rebay + LDG. Saturates R-DG-SIP-CONVERGENCE 4/4.

**PR-4 ~200 LOC / 1.5 engineer-days** — HDG M10. Post-2009 frontier, blocks-on 244-D27 GMRES.

**PR-5 ~860 LOC / 5 engineer-days** — Mortar M12-M18. Saturates R-MORTAR-CONSERVATION 3/3 + R-MORTAR-WHITNEY-DUALITY 3/3. Single-source-ownership of Wohlmuth-1999/2001 dual-basis + cross-points.

**PR-6 ~880 LOC / 5 engineer-days** — hp + estimators M19-M25. Saturates R-DWR-EFFICIENCY 4/4. Closes Tier-3.

**Defer (no PR)** — Tier-4 M26 IGA + M27 VEM + M28 BEM ~660 LOC. Ship only on concrete consumer demand. R-VEM-PATCH-TEST 3/3 if shipped.

**Cumulative through PR-6 ~3,660 LOC src + ~1,800 LOC test**, saturating five R-pins and unblocking 248 + 249 + 250 + 251.

---

## 5. CANDOR — what to ship vs defer

**Ship (recommended):** PR-1 + PR-2 + PR-3 + PR-5 ~2,580 LOC over 4 PRs / 14-15 engineer-days. Gives reality first-class non-conforming-CR + DG-SIP/BR/LDG + 2D-mortar-Wohlmuth-dual. Saturates R-CR-INF-SUP, R-DG-SIP-CONVERGENCE, R-MORTAR-CONSERVATION, R-MORTAR-WHITNEY-DUALITY. Production-grade FEM-frontier substrate. Unblocks 248 + 249 + 250 explicitly.

**Defer (without concrete consumer):** PR-4 HDG ~200 LOC (post-2009 efficiency, niche), PR-6 hp + estimators ~880 LOC (slot 251 shape-opt would be the natural consumer; ship when 251 begins), Tier-4 IGA + VEM + BEM ~660 LOC (no aicore consumer; mention in MASTER_PLAN as future-Block-D).

**Single-day-shippable cheapest** — M1+M2+M5 ~460 LOC reference-element + Lagrange + DOF-map. Blocks-on 245-S1-S2 (so requires that slot to ship first). Useful as cross-validation of 244-D26 P1-2D triangle Jacobian.

**Highest-leverage architecturally** — PR-5 mortar Wohlmuth-1999-dual-basis ~860 LOC because (a) directly unblocks 248-multigrid + 249-domain-decomp + 250-contact, (b) ships the production-grade-decision (dual-basis, not standard-mortar) that no zero-dep Go library has, (c) cross-validates 246-X11 Whitney-form via R-MORTAR-WHITNEY-DUALITY 3/3 — the discrete-exterior ↔ FE-cross-mesh unifying contract.

**Brand-prestige play** — Beirão-2013-VEM ~220 LOC is a 4000+-citation paper with no zero-dep Go implementation; first-mover advantage. But near-zero consumer-pull inside reality.

---

## 6. Summary of LOC + dependency counts

| Tier | Primitives | LOC src | LOC test | Blocks on |
|---|---|---|---|---|
| 0 substrate | M1-M5 | 720 | 360 | 244-D25-D26, 245-S1-S2, 246-X1 |
| 1 non-conforming + DG | M6-M11 | 880 | 440 | Tier-0, 244-D27 (HDG), 244-D20-D23 (flux) |
| 2 mortar | M12-M18 | 860 | 430 | Tier-0+Tier-1, 246-X11+X16 |
| 3 hp + estimators | M19-M25 | 880 | 440 | Tier-0+Tier-2, calculus.NumericalGradient |
| 4 IGA+VEM+BEM | M26-M28 | 660 | 330 | Tier-0, 246-X25 |
| **Total** | **M1-M28** | **4,000** | **2,000** | -- |
| Recommended ship | PR-1+2+3+5 | 2,580 | 1,290 | -- |

Twenty-eight primitives, four primitives shared with neighbouring slots (M2 → 244-D26, M11 → 244-D20-D23, M13 → 246-X11, M27 → 246-X25), twenty-four primitives unique to this slot.

The mortar-Wohlmuth-1999-dual-basis is the single-most-important production-grade-decision in computational FEM since 1999 and is the architectural keystone of this slot. Ship it.

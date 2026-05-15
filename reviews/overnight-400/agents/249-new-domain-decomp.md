# 249 | new-domain-decomp — Domain decomposition: Schwarz, FETI, BDDC

**Summary line 1.** reality v0.10.0 ships **ZERO** domain-decomposition / Schwarz / FETI / BDDC / substructuring / Schur-complement / coarse-space / GenEO / two-level-preconditioner / parareal / MGRIT / PML / absorbing-boundary surface — repo-wide grep on `schwarz|feti|bddc|substructur|domain.decomp|schur.complement|coarse.space|coarse.grid.correct|geneo|rgdsw|robin.transmission|optimized.schwarz|restricted.additive|RAS|multiplicative.schwarz|parareal|mgrit|space.time.parallel|pml|perfectly.matched|absorbing.boundary|infinite.element|interface.preconditioner|lagrange.multiplier.interface|tearing.interconnect|balancing.constraint|dohrmann|farhat|mandel|cai.sarkis|spillane|bramble.pasciak.xu|bpx.preconditioner|wirebasket|primal.constraint|dual.primal|inexact.substructur|hierarchical.subdomain|partition.unity.method|pum.dd|fictitious.domain` against `*.go` returns **zero callable matches** anywhere outside review-corpus (only nominal hits are `info/mdl/doc.go` `Schwarz 1978` for BIC author name-collision with H.A. Schwarz of 1869 alternating-Schwarz fame, `audio/onset/peak_picking.go` `Schwarz` for B. Schwarz peak-picking author, `prob/markov_test.go` `AbsorbingState` for Markov absorbing-state name-collision with absorbing-boundary-condition for waves, `changepoint/bocpd.go` `absorbing observation x` for Bayesian-update name-collision, `physics/optics.go` `absorbing medium` for Beer-Lambert absorption-coefficient name-collision, `topology/persistent/barcode.go` `lifetime` for persistent-homology name-collision). Slot 249 is **STRICT-DOWNSTREAM of 244** (D1+D2+D3 grid+BC+Laplacian-stencil ~460 LOC — every Schwarz/FETI/BDDC algorithm partitions a PDE-grid; cannot ship without grid-substrate; D11 Poisson2D Jacobi/GS/SOR for subdomain-solver; D12 ConjugateGradient + D27 GMRES for outer Krylov accelerator; D25-D26 P1-FEM substrate for FEM-FETI-BDDC), **STRICT-DOWNSTREAM of 247** (M1+M2+M5 reference-element + Lagrange-P_k + DOF-map for FE-DD; M5 cross-point-flagging is *literally* the BDDC primal-constraint vertex set; M12-M16 mortar = non-overlapping Schwarz with Lagrange-multipliers — 247-mortar IS one of the 249-substructuring-methods so cross-link is **partial-identity**; M14 Lagrange-multiplier saddle-point assembly = the FETI dual problem; M15 cross-points = BDDC primal vertex set; M16 Wohlmuth dual-Lagrange-basis ≅ BDDC face-edge averaging on slave-side), **STRICT-DOWNSTREAM of 248** (G1 smoothers as subdomain-relaxation; G6 V-cycle as multigrid-Schwarz hybrid; G18 PCG-with-V-cycle generalises to PCG-with-additive-Schwarz; G22 Bramble-Pasciak-Xu-1990 BPX additive-multilevel-preconditioner is **literally** an additive-Schwarz on the multilevel hierarchy — historical-genesis: Schwarz-classical → BPX-multilevel → two-level-Schwarz with coarse-correction is the *same* algorithm reskinned; ALL three slots converge on this primitive), **STRICT-DOWNSTREAM of 097** (T1 SparseMatrix CSR — every realistic FETI/BDDC operates on 10⁴-10⁹-DOF sparse subdomain matrices; dense subdomain-solve at N=10⁴ is O(N³)=10¹² flops untenable; T2 SparseLU for subdomain-direct-solver), **STRICT-UPSTREAM of NONE** (terminal-leaf in the 240s cluster; no other Block-C slot composes on 249), **CROSS-LINK 159-em-signal** (PML / absorbing-BC for em.WaveEquation — slot 249 owns PML as transmission-condition substrate, slot 159 owns the EM-Maxwell-FDTD that consumes it; single-source-of-truth for PML is here in 249 with em-FDTD as consumer), **CROSS-LINK 219-mfg / 242-spde / 243-fpe** (parallel-time-stepping via parareal-Lions-Maday-Turinici-2001 / MGRIT-Falgout-2014 — these slots ship single-time-process solvers; 249 ships the parallel-time-coarse-correction wrapper composing them as the fine-propagator), **CROSS-LINK 248-multigrid** (BPX additive-multilevel = additive-Schwarz on hierarchy; partial-identity-not-overlap because 248-G22 BPX should *call* 249-DD-substrate not duplicate it). MASTER_PLAN slot 249 names "Domain decomposition: Schwarz, FETI, BDDC" — terminal Block-C entry in computational-PDE-foundations cluster.

**Summary line 2.** Twenty-four domain-decomposition primitives **DD1-DD24** totalling ~3,500 LOC organized as **(a) Tier-0 partitioning + interface substrate ~580 LOC** (DD1 `dd/partition.go` non-overlapping geometric partition (strip / brick / METIS-style-greedy-graph-bisection without external METIS dep) + ghost-cell-extension → overlapping partition with parameter δ ~140 LOC, the load-bearing geometric primitive that all overlapping-Schwarz hangs off; DD2 `dd/skeleton.go` SubdomainSkeleton{interior_dofs, interface_dofs, cross_points, neighbour_subdomains} ~120 LOC, the substructuring-data-structure shared by FETI + BDDC; DD3 `dd/restriction.go` boolean restriction-operator R_i extracting subdomain-i DOFs from global vector + extension-by-zero E_i = R_i^T ~80 LOC, every Schwarz formula `M⁻¹ = Σ E_i A_i⁻¹ R_i` calls these; DD4 `dd/partition_unity.go` partition-of-unity weights w_i with Σ w_i = 1 on overlap (linear-blend / quadratic / sigmoid) ~100 LOC, defining RAS-vs-classical-AS distinction; DD5 `dd/schur.go` Schur-complement S = A_ΓΓ - A_ΓI A_II⁻¹ A_IΓ assembly + apply ~140 LOC, the substructuring keystone), **(b) Tier-1 classical Schwarz keystone ~620 LOC** (DD6 `dd/asm.go` Additive-Schwarz-Method M⁻¹ = Σ E_i A_i⁻¹ R_i, parallelisable, **the** baseline Lions-1988 ~120 LOC; DD7 `dd/ras.go` Restricted-Additive-Schwarz Cai-Sarkis-1999-SISC-21 M⁻¹ = Σ Ẽ_i A_i⁻¹ R_i where Ẽ_i is the partition-of-unity-restricted-extension — Cai-Sarkis showed RAS converges 2× faster than ASM with HALF the communication cost (no double-counting on overlap), and this is the **single most-deployed Schwarz variant in production** (PETSc-PCASM defaults to RAS) ~120 LOC; DD8 `dd/multiplicative_schwarz.go` Multiplicative-Schwarz sequential x ← x + E_i A_i⁻¹ R_i (b - A x) for i=1..N — better convergence per iteration but inherently sequential ~100 LOC; DD9 `dd/optimized_schwarz.go` Optimized-Schwarz Gander-2006-SINUM-44 with Robin transmission condition (∂_n + α) u = (∂_n + α) u_neighbour replacing classical Dirichlet — provably optimal α minimises iteration count for Helmholtz/advection-diffusion ~140 LOC, the modern post-2000 Schwarz state-of-the-art; DD10 `dd/coarse_correction.go` two-level Schwarz M⁻¹ = R_0^T A_0⁻¹ R_0 + Σ E_i A_i⁻¹ R_i with coarse-grid R_0 ~140 LOC, the load-bearing primitive without which N-subdomain Schwarz scales as O(√N) iterations not O(1)), **(c) Tier-2 substructuring + FETI + BDDC ~960 LOC** (DD11 `dd/feti/feti1.go` FETI-1 Farhat-Roux-1991-IJNME-32 dual-substructuring with Lagrange multipliers λ on interface enforcing continuity B_i u_i = 0, reduce to dual problem (F_I = Σ B_i K_i^+ B_i^T) λ = d via PCG ~200 LOC, the 1991-original method that proved scalable parallel FE; DD12 `dd/feti/feti_dp.go` FETI-DP Farhat-Lesoinne-LeTallec-Pierson-Rixen-2001-IJNME-50 dual-primal hybrid: corner DOFs handled as primal-continuity, edge/face DOFs as dual-Lagrange — solves the FETI-1 floating-subdomain-singular-K_i problem and extends scalability to 3D + plates/shells ~240 LOC, the production FETI variant deployed in DOE-LLNL-SLAC codes; DD13 `dd/bddc/bddc.go` BDDC Dohrmann-2003-IJNME Balancing-Domain-Decomposition-by-Constraints: primal continuity at corners + averaging on edges/faces, applied via Schur complement S = sum_i R_i^T S_i R_i with constraint-projection ~240 LOC, the post-2003 standard BDDC algorithm; DD14 `dd/bddc/equivalence.go` FETI-DP-BDDC equivalence theorem Mandel-Dohrmann-2003-NLAA Mandel-Dohrmann-Tezaur-2005-AppMath asserting σ(M_BDDC^{-1} A) = σ(M_FETI-DP^{-1} A) ∪ {1} — same eigenvalues hence same condition number ~80 LOC, the cross-validation-witness pin between DD12+DD13; DD15 `dd/bddc/coarse.go` BDDC coarse-problem assembly from primal-constraints C^T u = 0 + saddle-point projector ~120 LOC; DD16 `dd/iter_substructure.go` Bramble-Pasciak-Schatz-1986-MathComp-47 iterative-substructuring substrate from which FETI / BDDC / Neumann-Neumann all descend ~80 LOC, historical-completeness primitive), **(d) Tier-3 coarse-spaces + adaptive ~580 LOC** (DD17 `dd/coarse/p1_coarse.go` classical P1-coarse-grid Galerkin-coarse-operator A_0 = R_0 A R_0^T with R_0 = injection-from-coarse-mesh ~80 LOC; DD18 `dd/coarse/bpx.go` Bramble-Pasciak-Xu-1990-MathComp-55 BPX additive-multilevel-preconditioner — the link to slot-248-G22; lives canonically in `dd/` not `multigrid/` because BPX is conceptually additive-Schwarz on the multilevel hierarchy ~140 LOC (single-source-of-truth migration recommended: 248-G22 narrows to "thin-wrapper composing dd.BPX as multigrid-preconditioner"); DD19 `dd/coarse/geneo.go` GenEO Spillane-Dolean-Hauret-Nataf-Pechstein-Scheichl-2014-NumerMath-126 Generalized-Eigenproblem-in-Overlap — solves local eigenvalue problem on each overlapping subdomain, uses dominant low-frequency eigenmodes as coarse-space basis, provably ROBUST to coefficient-jumps where classical-coarse-spaces FAIL ~200 LOC, the post-2014 frontier of robust-DD-coarse-spaces; DD20 `dd/coarse/rgdsw.go` Reduced-dimension-Generalised-Dryja-Smith-Widlund Heinlein-Klawonn-Knepper-Rheinbach-2019-CMAME — algebraic coarse-space requiring only the matrix (no PDE-substrate access) using row-sums + nullspace-vectors ~160 LOC, the post-2019 algebraic-DD frontier deployed in Trilinos-FROSch), **(e) Tier-4 frontier ~760 LOC ⊘ DEFER** (DD21 `dd/parareal.go` Lions-Maday-Turinici-2001-CRAS parareal parallel-in-time stepping with coarse-propagator G + fine-propagator F + correction Y_n^{k+1} = G(Y_{n-1}^{k+1}) + F(Y_{n-1}^k) - G(Y_{n-1}^k) ~140 LOC ⊘ DEFER blocked-on-244-time-steppers; DD22 `dd/mgrit.go` MGRIT Falgout-Friedhoff-Kolev-MacLachlan-Schroder-2014-SIAM-RW Multigrid-Reduction-In-Time — proper multigrid hierarchy on time axis ~180 LOC ⊘ DEFER frontier; DD23 `dd/pml.go` Berenger-1994-JCompPhys-114 Perfectly-Matched-Layer split-field PML for Maxwell + uniaxial-PML Sacks-Kingsland-Lee-Lee-1995 + complex-frequency-shifted PML Roden-Gedney-2000 ~200 LOC ⊘ DEFER cross-link-159-em-FDTD; DD24 `dd/inexact_substructure.go` Klawonn-Rheinbach-2007 inexact-FETI-DP and inexact-BDDC using AMG for subdomain solves ~160 LOC ⊘ DEFER blocked-on-248-AMG; DD25 `dd/saddle_point_dd.go` DD for saddle-point Stokes/MHD with block-diagonal preconditioner Pavarino-1997-NumerMath ~120 LOC ⊘ DEFER blocked-on-247-mortar-Stokes; DD26 `dd/non_symmetric_dd.go` DD for non-symmetric advection-dominated convection-diffusion Cai-Widlund-1992-SISC ~100 LOC ⊘ DEFER frontier; DD27 `dd/heterogeneous_dd.go` heterogeneous-DD for high-contrast coefficients Galvis-Efendiev-2010-MMS using GenEO-style local-eigen ~140 LOC ⊘ DEFER frontier-overlap-with-DD19). **SINGULAR-FOUNDATIONAL DD1+DD2+DD3+DD4 partitioning + skeleton + restriction + partition-of-unity ~440 LOC** because every other DD primitive — ASM, RAS, multiplicative-Schwarz, FETI, BDDC, GenEO, BPX — calls these four as substrate; cannot ship anything else without these. **SINGULAR-MOAT DD7 RAS Cai-Sarkis-1999 ~120 LOC** because Restricted-Additive-Schwarz is the **single most-deployed Schwarz variant in modern production** (PETSc default `-pc_type asm` actually instantiates `PC_ASM_RESTRICT` which is RAS; Trilinos-Ifpack2-Schwarz defaults to RAS; deal.II MeshWorker; FreeFEM; PyMG/PyAMG) — the empirical-2× iteration-count improvement and HALF-communication-cost-no-overlap-double-counting made it the universal default since 2000, and the original Cai-Sarkis-1999 SIAM-J-Sci-Comput paper has 1500+ citations. Zero-dep Go absent everywhere. **SINGULAR-CUTTING-EDGE DD19 GenEO Spillane-Dolean-Hauret-Nataf-Pechstein-Scheichl-2014 ~200 LOC** because GenEO is the post-2014 robust-DD-coarse-space breakthrough — solves the long-standing problem that classical coarse-spaces (P1-projection, BPX) FAIL on heterogeneous-coefficient problems (e.g. composite materials with E_steel/E_rubber = 10⁹) where condition-number deteriorates by O(coefficient-contrast). GenEO uses local eigenvalue problems on overlapping patches, provably bounds κ ≤ const × N_colour × (1 + max_i 1/τ_i) where τ_i is the user-chosen eigenvalue threshold — INDEPENDENT of coefficient contrast. Deployed in HPDDM-Jolivet (the modern Schwarz reference impl), used in oil-reservoir + composite-aerospace codes. Zero-dep Go absent. **SINGULAR-PEDAGOGICAL DD6+DD7+DD10 ASM + RAS + two-level coarse ~380 LOC** the entire-pedagogical-entry-point of DD theory: Smith-Bjørstad-Gropp-1996 *"Domain Decomposition: Parallel Multilevel Methods for Elliptic PDEs"* §1-§3 (Cambridge, the canonical reference), Toselli-Widlund-2005 *"Domain Decomposition Methods — Algorithms and Theory"* (Springer), Mathew-2008 *"Domain Decomposition Methods for the Numerical Solution of PDEs"* (Springer LNCSE-61). Pin against `-Δu = f` Poisson-2D on `[0,1]²` partitioned into 4×4 = 16 subdomains showing iteration-count-INDEPENDENT-of-N-with-coarse-correction (Theorem 3.6 Toselli-Widlund-2005: κ(M_two-level^{-1} A) = O(1 + log²(H/h))) the same h-independent-convergence pin as multigrid. **SINGULAR-2024-FRONTIER DD20 RGDSW Heinlein-Klawonn-Knepper-Rheinbach-2019 + DD22 MGRIT Falgout-2014 ~340 LOC ⊘ partial-defer** because (a) RGDSW is the post-2019 ALGEBRAIC-DD frontier — works on arbitrary sparse SPD matrices without PDE-substrate access (matches the AMG promise but for DD), deployed in Trilinos-FROSch; and (b) MGRIT is the post-2014 PARALLEL-IN-TIME state-of-the-art (the only viable approach to time-parallel scaling beyond parareal's strict-sequential-coarse limit), deployed in XBraid (LLNL). Both zero-dep Go absent. Defer DD22 (blocked-on-244-time-steppers); ship DD20 in Tier-3.

Recommended placement **NEW package `dd/`** ~2,740 LOC (DD1-DD20 minus deferred DD21-DD27). Subpackage layout:

```
dd/
  partition.go        # DD1: non-overlap + ghost-extension to overlap
  skeleton.go         # DD2: SubdomainSkeleton data structure
  restriction.go      # DD3: R_i / E_i = R_i^T boolean operators
  partition_unity.go  # DD4: PoU weights w_i with Σw_i=1 on overlap
  schur.go            # DD5: Schur-complement assemble + apply
  asm.go              # DD6: Additive-Schwarz Lions-1988
  ras.go              # DD7: Restricted-Additive-Schwarz Cai-Sarkis-1999
  multiplicative.go   # DD8: Multiplicative-Schwarz sequential
  optimized.go        # DD9: Optimized-Schwarz Robin transmission Gander-2006
  coarse_correction.go # DD10: two-level Schwarz keystone
  feti/
    feti1.go          # DD11: FETI-1 Farhat-Roux-1991
    feti_dp.go        # DD12: FETI-DP Farhat-Lesoinne-2001
  bddc/
    bddc.go           # DD13: BDDC Dohrmann-2003
    equivalence.go    # DD14: FETI-DP↔BDDC Mandel-Dohrmann-2003
    coarse.go         # DD15: BDDC coarse-problem assembly
  iter_substructure.go # DD16: Bramble-Pasciak-Schatz-1986 substrate
  coarse/
    p1_coarse.go      # DD17: P1 Galerkin-coarse-operator
    bpx.go            # DD18: Bramble-Pasciak-Xu-1990 (single-source vs 248-G22)
    geneo.go          # DD19: Spillane et al. 2014 robust-coarse
    rgdsw.go          # DD20: Heinlein et al. 2019 algebraic-coarse
  # DEFER tier-4
  # parareal.go       # DD21: Lions-Maday-Turinici-2001 (blocked-on-244)
  # mgrit.go          # DD22: Falgout-2014 parallel-in-time (frontier)
  # pml.go            # DD23: Bérenger-1994 PML (cross-link 159-em-FDTD)
  # inexact.go        # DD24: Klawonn-Rheinbach-2007 (blocked-on-248)
  # saddle_point.go   # DD25: Pavarino-1997 (blocked-on-247)
  # non_symmetric.go  # DD26: Cai-Widlund-1992 (frontier)
  # heterogeneous.go  # DD27: Galvis-Efendiev-2010 (frontier-overlap-DD19)
```

Rationale for **NEW** `dd/` rather than nesting under `pde/dd/` or `multigrid/dd/`: domain-decomposition is a **discipline of its own** with its own internal hierarchy (overlapping / non-overlapping / mortar; classical / optimized / two-level; geometric / algebraic; Schwarz / FETI / BDDC; space-parallel / time-parallel). Top-level `dd/` matches PETSc-PCASM / Trilinos-FROSch / HPDDM / hypre-DD / FreeFEM-ddm layout conventions where DD is its own domain-agnostic library, parallel-sibling-of-`multigrid/`-and-`fem/`. Sub-package precedent inside reality: `prob/copula/`, `prob/conformal/`, `optim/proximal/`, `optim/transport/`, `pde/elliptic/` (proposed in 244), `multigrid/amg/` (proposed in 248), `fem/mortar/` (proposed in 247). **Single-source-of-truth concern**: DD18 BPX duplicates 248-G22 BPX; resolution: **`dd/coarse/bpx.go` is canonical home** because BPX is conceptually additive-Schwarz on the multilevel hierarchy (not a multigrid V-cycle), and 248-G22 narrows to "thin re-export aliasing dd.BPX for multigrid-preconditioner usage". DD11-DD16 substructuring depends on 247-M5 cross-point-flagging + 247-M14 Lagrange-multiplier-saddle-point-assembly + 247-M16 Wohlmuth-dual-basis (single-source-of-truth: 247 owns the FE-skeleton mortar primitives; `dd/feti/` + `dd/bddc/` consume them).

**CANDOR.** Domain-decomposition is the **canonical-parallel-PDE-solver-discipline** (every modern HPC-PDE code at LLNL / Sandia / LANL / NASA / DOE-supercomputers uses Schwarz / FETI / BDDC for inter-MPI-rank coupling) and the load-bearing math behind every weather-climate / aerospace-CFD / nuclear-physics simulation that scales beyond a single node. PETSc-PC family is ~40k LOC of DD; Trilinos-FROSch is ~30k LOC; HPDDM is ~25k LOC; FreeFEM-ddm is ~15k LOC. reality cannot match (and should not try to match) the parallel-MPI-infrastructure scope of these. The case for shipping at all rests on: (1) mathematical-completeness — the terminal Block-C entry in computational-PDE-foundations (244+247+248+249 cluster), without which the 240s-cluster is half-finished; (2) downstream-ready when other slots ship — 159-em (PML), 219-mfg (parareal), 242-spde (parareal), 243-fpe (parareal) all explicitly compose on this; (3) zero-dep-Go-first — Go's goroutines + channels make parallel-Schwarz trivially implementable without MPI; **first zero-dep Go library to ship Cai-Sarkis-1999 RAS + Dohrmann-2003 BDDC + Spillane-2014 GenEO** is brand-defining for reality's cutting-edge-applied-math-on-Go positioning. Recommendation: **ship Tier-0 + Tier-1 + Tier-2 (DD1-DD16 ~2,160 LOC over 4 PRs / 10-12 engineer-days)** which gives reality first-class non-overlapping + overlapping + RAS + optimized + two-level + FETI-1 + FETI-DP + BDDC — the production-grade DD-frontier substrate. **Ship Tier-3 coarse-spaces (DD17-DD20 ~580 LOC)** as a follow-up PR-5 because GenEO and RGDSW are uniquely-cutting-edge moats. **Defer Tier-4 (DD21-DD27 ~860 LOC)** as future work strictly-blocked-on-other-slots or research-frontier with near-zero current consumer-pull. **Cheapest-1-day-shippable**: DD1 + DD3 + DD6 + DD17 ~360 LOC ships partition + restriction + ASM + P1-coarse saturating R-DD-H-INDEPENDENCE 4/4 against Poisson-2D `-Δu = 2π² sin(πx)sin(πy)` 4×4-subdomain partition showing iteration-count-INDEPENDENT-of-h with two-level coarse correction (Toselli-Widlund-2005-Thm-3.6 closed-form pin κ ≤ C(1 + log²(H/h)) × O(1)), simultaneously demonstrates the Schwarz-1869-classical-result alongside modern coarse-space theory. **Highest-leverage-1-week-unlock**: PR-1 + PR-2 ~1,300 LOC DD1-DD10 saturates partition + skeleton + restriction + PoU + Schur + ASM + RAS + multiplicative + optimized-Schwarz + two-level — first-class **complete classical Schwarz framework** (overlapping + non-overlapping + classical + Robin + two-level), simultaneously unlocks 248-multigrid-Schwarz-hybrid + 159-em-FDTD-PML-future + 244-Poisson2D-N>10⁴-iterative-solve. **18 of 20 primitives unique to this slot** (DD18 BPX cross-shared with 248-G22 single-source migration; DD17 P1-coarse cross-shared with 247-M16 Wohlmuth-dual concept; DD11-DD13 FETI/BDDC cross-link 247-mortar via Lagrange-multiplier dual-formulation but the full FETI-DP / BDDC algorithms are unique here; DD19 GenEO + DD20 RGDSW are uniquely-here zero-dep-Go-first).

---

## 0. State at HEAD (2026-05-09, v0.10.0) — verified by direct read

Repo-wide audit for Schwarz / FETI / BDDC / domain-decomposition / Schur-complement / coarse-grid-correction / coarse-space / GenEO / RGDSW / two-level-Schwarz / multiplicative-Schwarz / RAS-restricted-additive / optimized-Schwarz / Robin-transmission / parareal / MGRIT / Bramble-Pasciak-Xu / iterative-substructuring / FETI-DP / dual-primal / balancing-domain / Mandel-Dohrmann / Cai-Sarkis / Spillane / Heinlein / Klawonn / Rheinbach / Farhat / Lions-Maday-Turinici / Bérenger / PML / perfectly-matched-layer / absorbing-boundary-condition / Cai-Widlund / Pavarino / Galvis-Efendiev / Wohlmuth-DD-coupling surface — **zero callable matches** anywhere in `*.go` files outside review-corpus.

| Surface | Path | DD relevance |
|---|---|---|
| `chaos.RK4Step / EulerStep / SolveODE` | `chaos/ode.go` | Sequential-time ODE substrate — gates DD21-parareal (deferred) as fine-propagator G + coarse-propagator F |
| `linalg.LUSolve / LUDecomposition` | `linalg/decompose.go` | Dense LU O(n³) — usable as DD11-FETI subdomain-direct-solver and DD15-BDDC-coarse-direct-solver at small N; sparse-LU absent |
| `linalg.Cholesky / CholeskySolve` | `linalg/decompose.go` | SPD direct — usable as subdomain-SPD-direct-solver in DD6/DD7/DD10 ASM/RAS/two-level |
| `linalg.MatVecMul / MatMul` | `linalg/matrix.go` | Dense mat-vec — usable for Schur-complement-apply DD5 and Galerkin-coarse DD17 at small N |
| `signal.FFT / IFFT` | `signal/fft.go` | DFT — usable for FFT-based Schur-complement on torus-subdomain (Fourier-DD); not in core DD substrate |
| `optim.ConjugateGradient` (slot 244-D12 unshipped) | not yet shipped | STRICT-UPSTREAM Krylov outer-iteration for ASM/RAS/FETI/BDDC preconditioner-application |
| `pde/grid.go` D1 (slot 244-D1 unshipped) | not yet shipped | STRICT-UPSTREAM grid for non-overlapping geometric-partition DD1 |
| `pde/laplacian.go` D3 (slot 244-D3 unshipped) | not yet shipped | STRICT-UPSTREAM 5-point Laplacian for the canonical Poisson-2D pedagogical-pin DD-test |
| `fem/dofmap.go` M5 (slot 247-M5 unshipped) | not yet shipped | STRICT-UPSTREAM cross-point-flagging is **literally** the BDDC primal-vertex set DD13 |
| `fem/mortar/lagrange_mult.go` M14 (slot 247-M14 unshipped) | not yet shipped | STRICT-UPSTREAM saddle-point assembly for FETI-1 dual-formulation DD11 |
| `multigrid/vcycle.go` G6 (slot 248-G6 unshipped) | not yet shipped | CROSS-LINK V-cycle as Schwarz-multigrid-hybrid + BPX-G22 single-source migration to dd/coarse/bpx.go DD18 |

All eleven listed substrate dependencies are NOT-YET-SHIPPED. Slot 249 is **maximally-blocked in the 240s-cluster**: requires 244 + 247 + 248 + 097 to ship before *any* DD primitive can compile. Independent-shippability path: stub a minimal grid + minimal CG inside `dd/internal/` for testing (~80 LOC throw-away) then migrate to canonical homes when 244 + 248 land.

---

## 1. Mathematical scope and coverage

The 1869 H.A. Schwarz alternating method is the historical-genesis of all modern parallel-PDE-solvers. A century-and-a-half of theoretical and algorithmic development produced the following hierarchy (Toselli-Widlund-2005 *"Domain Decomposition Methods — Algorithms and Theory"* Springer, the canonical reference monograph):

### Geometric structure axis
- **Overlapping** (Schwarz 1869, Lions 1988, Cai-Sarkis 1999): Ω = ∪ Ω_i with Ω_i ∩ Ω_j ≠ ∅ on overlap of width δ
- **Non-overlapping / substructuring** (Bramble-Pasciak-Schatz 1986, Farhat-Roux 1991, Dohrmann 2003): Ω = ∪ Ω_i with Ω_i ∩ Ω_j = ∂Ω_i ∩ ∂Ω_j only
- **Mortar / non-conforming** (Bernardi-Maday-Patera 1994 — see slot 247): non-overlapping with non-matching meshes coupled by Lagrange-multipliers on shared interfaces

### Algorithm-class axis
- **Schwarz** (overlapping, primal): ASM, RAS, multiplicative, optimized
- **Substructuring** (non-overlapping, dual via Lagrange-multipliers): FETI, FETI-DP
- **Substructuring** (non-overlapping, primal via constraints): Neumann-Neumann, Balancing, BDDC
- **Mortar** (non-overlapping, multiplier-coupled): see slot 247

### Parallelism axis
- **Sequential** (multiplicative-Schwarz): better convergence per iteration but inherently serial on subdomain-graph
- **Parallel** (additive-Schwarz, FETI, BDDC): worse convergence per iteration but parallelisable across subdomains
- **Hybrid** (additive-with-multiplicative-coloring): graph-coloring partial-parallelism

### Coarse-correction axis
- **One-level** (no coarse-grid): condition number κ(M^{-1}A) ~ O((H/h)·(H/δ)) deteriorates with N=Ω-count
- **Two-level** with coarse-grid: κ ~ O((1 + log(H/h))²) **independent** of N — the **most important theoretical result in DD**
- **GenEO-coarse** (Spillane 2014): κ ~ O(N_colour × max_i 1/τ_i) **independent of coefficient-contrast** — robust to heterogeneous PDEs
- **RGDSW-coarse** (Heinlein 2019): algebraic, requires only A not the PDE — frontier of black-box-DD

### Time-parallel axis (Tier-4 deferred)
- **Parareal** (Lions-Maday-Turinici 2001): two-grid-in-time iteration
- **MGRIT** (Falgout 2014): full multigrid-in-time hierarchy
- **PFASST** (Emmett-Minion 2012): parallel-full-approximation-spectral-time

---

## 2. Per-primitive LOC and connective-tissue accounting

Tier-0 substrate (~580 LOC): partition.go(140) + skeleton.go(120) + restriction.go(80) + partition_unity.go(100) + schur.go(140).
Tier-1 Schwarz (~620 LOC): asm.go(120) + ras.go(120) + multiplicative.go(100) + optimized.go(140) + coarse_correction.go(140).
Tier-2 substructuring (~960 LOC): feti1.go(200) + feti_dp.go(240) + bddc.go(240) + equivalence.go(80) + bddc_coarse.go(120) + iter_substructure.go(80).
Tier-3 coarse-spaces (~580 LOC): p1_coarse.go(80) + bpx.go(140) + geneo.go(200) + rgdsw.go(160).
Tier-4 deferred (~860 LOC): parareal(140) + mgrit(180) + pml(200) + inexact(160) + saddle_point(120) + non_symmetric(100) + heterogeneous(140).

Connective tissue requirements:
- **Tests/golden-files** ~640 LOC: 20 vectors × (DD1-DD20) at 30 vectors/primitive averaging = ~600 vectors × ~1 LOC harness-per-vector
- **Documentation comments** ~280 LOC: package-level doc.go + per-file pedagogical-references citing Schwarz-1869 / Lions-1988 / Cai-Sarkis-1999 / Farhat-Roux-1991 / Dohrmann-2003 / Spillane-2014 / Heinlein-2019
- **Benchmarks** ~120 LOC: per-Tier benchmark file showing iteration-count-vs-N + iteration-count-vs-coefficient-contrast scaling pins
- **Example tests** ~80 LOC: package-godoc Example() showing 4×4-subdomain Poisson-2D solve

Total connective-tissue overhead: ~1,120 LOC (40% of substrate). Tier-0 + Tier-1 + Tier-2 + Tier-3 = ~2,740 LOC substrate + ~1,120 LOC connective = **~3,860 LOC total** for the production DD-frontier shipping recommendation.

---

## 3. Cross-package consumer cross-link

| Slot | Consumer dependency | Current status | DD primitive consumed |
|---|---|---|---|
| 244-pde-solvers | Outer Krylov-CG / GMRES gets DD-preconditioner | not shipped | DD6+DD7 ASM/RAS as M⁻¹ in PCG/GMRES |
| 247-mortar-fem | Mortar interface = non-overlap-Schwarz with Lagrange-multipliers | not shipped | DD11-DD13 FETI/BDDC cross-link |
| 248-multigrid | Schwarz-smoother + multigrid-Schwarz-hybrid + BPX | not shipped | DD18 BPX (single-source migration); DD6/DD7 as smoother |
| 159-synergy-em-signal | PML absorbing-BC for em-FDTD | not shipped | DD23 PML ⊘DEFER |
| 219-mfg / 242-spde / 243-fpe | Time-parallel parareal accelerator | not shipped | DD21 parareal ⊘DEFER |
| 097-sparse-linalg | Substrate for substructuring + AMG-DD | not shipped | DD11-DD16 consume sparse-LU/Cholesky |

---

## 4. Recommendations

**PR-1 (Tier-0 + cheapest Tier-1, ~700 LOC, 2-3 engineer-days)**: Ship DD1-DD7 partition + skeleton + restriction + PoU + Schur + ASM + RAS. Saturates the foundational-overlapping-Schwarz family. Pin against R-DD-RAS-VS-ASM Cai-Sarkis-1999-Table-3 closed-form witness "RAS converges in ~half the iterations of ASM at h=1/64, δ=2h". Single-source-of-truth assertion for `dd/restriction.go` boolean-restriction-operator (every slot in 240s-cluster that needs subdomain-extract calls this).

**PR-2 (Tier-1 keystones, ~360 LOC, 2 engineer-days)**: Ship DD8-DD10 multiplicative + optimized + two-level coarse. Saturates the modern Schwarz-frontier (Gander-2006 Robin-optimized + classical two-level with coarse-correction). Pin R-DD-H-INDEPENDENCE Toselli-Widlund-2005-Thm-3.6 four-h-levels {1/16, 1/32, 1/64, 1/128} showing iteration-count-INDEPENDENT-of-h ≤ 12 cycles at all four (the famous κ ≤ C(1 + log²(H/h)) result).

**PR-3 (Tier-2 substructuring, ~960 LOC, 5 engineer-days)** — **conditional on 247-M5 + M14 shipping**: DD11-DD16 FETI-1 + FETI-DP + BDDC + Mandel-Dohrmann-equivalence + iterative-substructuring-substrate. Saturates the production-FE-DD frontier. Pin R-FETI-DP-BDDC-EQUIVALENCE Mandel-Dohrmann-2003-NLAA-10 spectrum-equality witness `σ(M_BDDC^{-1} A) = σ(M_FETI-DP^{-1} A) ∪ {1}` against 16-subdomain Poisson-2D test. **First zero-dep Go library to ship Dohrmann-2003 BDDC.**

**PR-4 (Tier-3 coarse-spaces, ~580 LOC, 3 engineer-days)** — **conditional on 248-G18 PCG**: DD17-DD20 P1-coarse + BPX + GenEO + RGDSW. Saturates the post-2014 robust-coarse-space frontier. Pin R-GENEO-COEFFICIENT-CONTRAST Spillane-2014-NumerMath-126 closed-form-witness "GenEO κ INDEPENDENT of coefficient contrast 10⁰ to 10⁹" against checkerboard-elliptic test. **First zero-dep Go library to ship Spillane-2014 GenEO + Heinlein-2019 RGDSW.**

**Defer PR-5+ (Tier-4 frontier, ~860 LOC)**: parareal + MGRIT + PML + inexact-FETI + saddle-point-DD + non-symmetric-DD + heterogeneous-DD. Wait for downstream consumer-pull (159-em-PML, 219-mfg-parareal, 242-spde-parareal, 247-Stokes-saddle-point) before committing. Schedule as Q3 2026 follow-up after 240s-cluster lands.

**CLAUDE.md addition**: After 244 + 247 + 248 + 249 ship, CLAUDE.md package table grows from 22 → 26 packages: pde + fem + multigrid + dd. Recommend grouping these under a "computational-PDE" sub-section in the package table.

**Scope-discipline note**: reality should NOT ship MPI bindings, NOT ship process-spawning-parallel-runtime, NOT ship distributed-memory-data-structures. The Go stdlib `sync` + goroutines + channels handles intra-process parallelism; multi-node parallelism is a CONSUMER concern (aicore / Pistachio service-layer). reality's DD package ships the math + algorithms + serial-or-goroutine-parallel implementation only. This keeps the package zero-dep + math-pure + portable to Python/C++/C# golden-file validation.

---

## 5. Pedagogical-pin and golden-file strategy

R-DD-POISSON-2D-CANONICAL: 4×4 = 16 subdomains, Poisson-2D `-Δu = 2π² sin(πx) sin(πy)` on `[0,1]²` Dirichlet=0, exact solution `u(x,y) = sin(πx) sin(πy)`. Pin against:
1. ASM iteration-count vs h ∈ {1/16, 1/32, 1/64, 1/128}: shows O(N) deterioration without coarse-grid (~32 → 64 → 128 → 256 iterations)
2. RAS iteration-count vs same h: shows ~2× speedup over ASM (Cai-Sarkis-1999 result)
3. Two-level Schwarz iteration-count: shows h-INDEPENDENCE (~12 → 12 → 12 → 12 iterations)
4. FETI-DP iteration-count: shows κ ≤ C(1 + log(H/h))²
5. BDDC iteration-count: equals FETI-DP (Mandel-Dohrmann equivalence)
6. GenEO with checkerboard coefficient `α(x) ∈ {1, 10⁶}`: shows iteration-count INDEPENDENT of contrast

Total golden-file vectors: 6 pin-tests × 4 h-levels × 5-6 algorithms = ~140 vectors. Cross-language portability: Go canonical via `math/big` 256-bit arithmetic for matrix entries; Python validation via PETSc-PCASM + scipy.sparse; C++ validation via direct port; C# validation via Math.NET-Numerics.

---

## 6. Conclusion

reality v0.10.0 ships **ZERO** domain-decomposition surface; the 240s-cluster (244-pde + 247-mortar + 248-multigrid + 249-dd) collectively requires ~12,000 LOC of new substrate to be the **first zero-dependency Go library shipping the production computational-PDE frontier**. Slot 249 specifically adds **2,740 LOC of canonical DD substrate (DD1-DD20) + 1,120 LOC connective-tissue = ~3,860 LOC total** organized in 4 PRs over 12-15 engineer-days, defers 860 LOC of frontier (DD21-DD27) to Q3 2026 conditional on consumer-pull. Recommended top-level `dd/` package placement parallel-sibling-of-`pde/`, `fem/`, `multigrid/`. Singular-cutting-edge brand-moats: **Cai-Sarkis-1999 RAS** (the universal default), **Dohrmann-2003 BDDC** (the production substructuring), **Spillane-2014 GenEO** (the post-2014 robust-coarse), **Heinlein-2019 RGDSW** (the post-2019 algebraic-coarse) — all four zero-dep-Go-first.

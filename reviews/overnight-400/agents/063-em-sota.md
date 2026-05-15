# 063 — em-sota

**Topic:** em: compare with MEEP, openEMS, FEKO, COMSOL ACDC.
**Scope:** SOTA EM library comparison; engineering/architecture choices, not feature checklists.
**Audited:** `C:\limitless\foundation\reality\em\em.go` (213 LOC, 11 functions).
**Sister reports:** 061 (numerics — present surface is IEEE-754-clean, two latent bugs); 062 (missing — ~170-220 canonical EM primitives, 11 shipped, T1+T2+T3 plan ~6,500-9,500 LOC). This report does **not** re-enumerate primitives; it asks: of the engineering/architecture choices the SOTA stack has converged on, which are reachable under reality's constraints (zero-dep, single math.h, golden-file cross-language, no allocations, MIT)?

---

## Headline

**Reality's `em/` is not behind the SOTA — it is two architectural pivots upstream of it.** Every modern EM tool (MEEP, openEMS, FEKO, COMSOL, HFSS, CST, Tidy3D, FDTDX, scikit-rf, qiskit-metal/pyEPR) sits on top of one of three engineering decisions reality has not yet made: **(A) PDE-solver kernel** (Yee staggered grid + PML, MoM/MLFMM, FEM Galerkin), **(B) `complex128` linear-algebra surface** (S/Y/Z/ABCD, eigenmode), or **(C) automatic differentiation** (adjoint method, JAX-style time-reversibility). The SOTA's three independent revolutions of the past 15 years — **subpixel smoothing** (Farjadpour 2006), **adjoint inverse design** (Lalau-Keraly 2013, Hughes 2018), and **GPU/AD-native FDTD** (Tidy3D 2020, FDTDX 2024) — are each downstream of (A)+(B)+(C). Reality has the floor (`constants/`, `linalg/`, `calculus/`) but not yet the slab. This report names the eight engineering decisions that fix the slab geometry, with the SOTA precedent for each.

The fastest first-light SOTA-comparable artefact in reality is **scikit-rf parity in pure Go** (~700 LOC over T1-COMPLEX + T1-TLINES + T1-SMITH from 062): closed-form, no PDE solver, no AD, but covers ~80% of practitioner RF/microwave workflow today and has zero portability friction. Everything else (FDTD, MoM, FEM, adjoint) requires a multi-month engineering investment; for those reality should explicitly **cite-and-skip** rather than half-ship.

---

## SOTA library survey

For each: (1) headline algorithm/feature, (2) engineering choice that defines it, (3) zero-dep portability for reality.

### MEEP (MIT, FDTD) — the academic reference, CPU
1. **Subpixel smoothing.** Material discontinuities are smoothed into continuous transitions over one voxel via second-order accurate averaging (Farjadpour 2006, Oskooi 2009). Without it, the FDTD error from a staircased curved interface is O(Δx); with it, O(Δx²) and the gradient w.r.t. interface position is continuous — the precondition for adjoint inverse design ([MEEP docs](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/)).
2. **Engineering choice:** **Cubic Yee grid + Bloch-periodic + PML + subpixel + MaterialGrid (continuous binary-relaxation field).** The MaterialGrid is the design parameter for the adjoint solver; the subpixel rule guarantees ∂F/∂ρ is well-defined at ρ∈{0,1} ([MEEP adjoint tutorial](https://meep.readthedocs.io/en/latest/Python_Tutorials/Adjoint_Solver/)).
3. **Portability to reality:** Yee step + PML is ~800 LOC of plain stencil arithmetic (in-scope per 062 §T3-FDTD-1D/2D/3D). Subpixel smoothing requires geometric primitives (ray-box, ray-cylinder intersection) which `geometry/` already has SDFs for — reuse, not new dep. **Verdict: in-scope but heavy.** Adjoint solver is out-of-scope without the AD layer (see Tidy3D below).

### openEMS (Aachen, FDTD) — the open-source RF reference
1. **EC-FDTD on Cartesian + cylindrical hexahedral mesh, UPML/CPML absorbers, MPI/SIMD/multi-thread parallel, MRI-grade SAR.** The "Equivalent-Circuit" formulation lets the same Yee step run on graded-mesh hexahedra without losing second-order accuracy ([openEMS](https://www.openems.de/), [github](https://github.com/thliebig/openEMS)). UPML implementation is a single 600-LOC `operator_ext_upml.cpp` ([source](https://github.com/thliebig/openEMS/blob/master/FDTD/extensions/operator_ext_upml.cpp)).
2. **Engineering choice:** **Operator-extension architecture.** The base FDTD engine is a generic Yee stepper; PML, dispersive materials, conducting sheets, lumped elements all plug in as `Operator_Ext_*` subclasses. This is the right factoring for a math library: the Yee stencil is one concern, the absorber another, the material another — each independently golden-testable.
3. **Portability to reality:** **The cleanest model in the SOTA for reality to copy.** Operator-extension factoring maps onto Go interfaces with no allocation cost. Cartesian-graded-hex first (matches `linalg/` strided arrays); cylindrical adds a Bessel dependency (already in 062 T1-SPECIAL). MPI is out-of-scope (no I/O); SIMD is out-of-scope (Go has no portable SIMD intrinsics — Go assembly is per-arch, breaks the cross-language golden-file contract). Single-threaded is fine for reality's role; downstream services parallelise via slicing.

### FEKO (Altair, MoM/MLFMM/FEM/PO/UTD hybrid) — the commercial RF reference
1. **Hybrid MoM-FEM with integral-equation boundary condition between regions, plus MLFMM for electrically-large MoM, plus PO/GO/UTD asymptotic solvers for the optical-frequency limit** ([Altair docs](https://help.altair.com/feko/topics/feko/user_guide/solver_solution_methods/solver_method_general_feko_c.htm)).
2. **Engineering choice:** **Method dispatcher per problem regime.** A 100λ-radar-cross-section problem dispatches to PO; a coupled-feed antenna dispatches to MoM; a dielectric-loaded waveguide dispatches to FEM/MoM; everything else gets MLFMM. The user picks the method; the kernels share geometry primitives but otherwise nothing.
3. **Portability to reality:** MoM kernel itself (RWG basis, EFIE/MFIE/CFIE matrix assembly) is ~2000 LOC of Galerkin integration over triangles — feasible but heavy. **MLFMM is the hard bit:** multilevel fast multipole requires recursive cube subdivision + addition-theorem translation operators (Wigner-3j, spherical harmonic rotations) — ~3000 LOC, dependency on T1-SPECIAL Legendre/SH at minimum. PO is closed-form (~150 LOC, in-scope). UTD wedge-diffraction coefficients are special-function-heavy (Fresnel, transition function) but tractable. **Verdict: PO + MoM core in-scope at multi-month cost; MLFMM is the boundary of what reality should attempt.**

### COMSOL Multiphysics (AC/DC + RF Module, FEM)
1. **FEM with automatic adaptive mesh refinement, multiphysics coupling (EM + thermal + structural + flow), eigenmode and frequency-sweep solvers, equation-based modeling.**
2. **Engineering choice:** **Galerkin FEM on tetrahedral mesh + automatic h-refinement driven by error estimator + assembled sparse matrix passed to MUMPS/PARDISO/iterative solver.** The AMR loop is what HFSS calls "adaptive solution process" — refine where ‖∇×E‖ residual is highest, re-solve, repeat until ΔS-param converges ([Ansys docs](https://ansyshelp.ansys.com/public/Views/Secured/Electronics/v251/en/Subsystems/HFSS/Subsystems/An%20Introduction%20to%20HFSS/Content/AdaptiveSolutionProcessanditsImportancetoHFSS.htm)).
3. **Portability to reality:** Tetrahedral mesh generation is **out-of-scope per CLAUDE.md** (mesh I/O, robust geometric kernel — bring in CGAL territory). **However, the FEM-on-given-mesh kernel is in-scope:** Galerkin assembly over user-supplied tet mesh, edge-element (Nédélec) basis, sparse matrix from `linalg/`. ~1500 LOC. Eigenmode solver requires generalized eigenvalue routine on sparse complex matrices — currently no `linalg/` primitive for this; that's a separate sub-effort.

### HFSS (Ansys, FEM) — the gold-standard commercial FEM
1. **Adaptive mesh refinement with Nédélec edge elements, eigenmode + driven-modal + driven-terminal solvers, Hierarchical-basis higher-order elements, port-mode field calculator, large-scale MLFMM-FEM hybrid for arrays.**
2. **Engineering choice:** **Hierarchical p-refinement on top of h-refinement, with the basis order locally adapted per element.** The 2024 mesh evolution moved toward "Phi" mesh — body-fitted curvilinear elements that integrate exactly with curved CAD ([Semiwiki](https://semiwiki.com/eda/306866-a-mesh-by-any-other-name-the-hfss-mesh-evolution/)).
3. **Portability to reality:** Curvilinear elements need exact-integration over CAD primitives — needs `geometry/` SDF + parametric-surface bracket. **Edge-element Nédélec basis on given linear-tet mesh is the realistic reality target;** p-refinement and curvilinear are explicit non-goals.

### CST Studio Suite (Dassault) — Finite Integration Technique
1. **FIT (finite integration technique) is the discrete-form Maxwell formulation underlying CST's transient solver — equivalent to FDTD in the Cartesian case but generalises to non-orthogonal hexahedral meshes;** plus TLM (transmission-line matrix) as a parallel formulation in the same package, plus frequency-domain FEM ([CST](https://www.3ds.com/products/simulia/cst-studio-suite/electromagnetic-simulation-solvers)).
2. **Engineering choice:** **Discrete differential forms (Bossavit / Weiland).** FIT writes Maxwell as `Cē = -d/dt b̄`, `C̃h̄ = d/dt d̄ + j̄` with `C, C̃` discrete curl operators on primal/dual mesh. This is mathematically elegant and equivalent to Yee on Cartesian — the win is on non-Cartesian conformal meshes where the standard Yee derivation falls apart.
3. **Portability to reality:** For Cartesian first-light, FIT == Yee; pick whichever exposition is clearer. **The FIT formulation is the better long-term bet** because it generalises to graded hex (matching openEMS) and to tet (matching FEM) without rewriting the kernel. Cost: same ~800 LOC plus a `discrete_curl` primitive in `linalg/` (sparse matrix-free operator).

### Tidy3D (Flexcompute, GPU FDTD + autograd) — the differentiable cloud reference
1. **GPU-native FDTD with native automatic differentiation via JAX/autograd integration; the adjoint method computes gradients w.r.t. arbitrary structure parameters with cost = 2× forward FDTD regardless of #parameters** ([Flexcompute](https://www.flexcompute.com/tidy3d/), [Autograd intro](https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd1Intro/)).
2. **Engineering choice:** **Time-reversibility of Maxwell + reverse-mode AD.** The lossless Yee step is reversible; the adjoint Maxwell run replays the forward simulation backward and accumulates ∂L/∂ε. Memory cost is O(boundary) instead of O(timesteps × volume). The 2024 transition deprecated the JAX adjoint plugin in favour of native autograd in tidy3d v2.7+.
3. **Portability to reality:** Reality has no AD layer of its own (autodiff package is being scoped — see report 011-015). **The pure FDTD kernel is portable; the adjoint requires reverse-mode AD over float64 arrays.** Two paths: (i) hand-rolled adjoint Maxwell (~400 LOC, 2× forward, exact gradients, zero AD dep — this is what FDTDX paper §3.2 calls "memory-efficient gradient via time-reversibility"); (ii) compose with reality's autodiff package once it lands. **Path (i) is reality-native and recommended:** the adjoint Yee step is itself a deterministic stencil computable from the forward fields and adjoint sources; no tape, no closures, no compilation step.

### FDTDX (Mahlau et al., 2024 — open-source JAX/GPU) — the new academic SOTA
1. **JAX-based FDTD scaling to billions of grid cells across multi-GPU; ~10× faster than MEEP and ~415× faster than Ceviche on 288M-cell benchmarks; memory-efficient gradients via Maxwell time-reversibility** ([arXiv 2412.12360](https://arxiv.org/abs/2412.12360), [github](https://github.com/ymahlau/fdtdx)).
2. **Engineering choice:** **Recompute-don't-store.** Standard reverse-mode AD stores fields at every timestep — N_t × volume memory. FDTDX exploits Yee reversibility to recompute backward, eliminating the storage. This is the "checkpointing taken to N_t = 1" trick that makes 3D inverse design practical.
3. **Portability to reality:** **The reversibility trick itself is the portable insight, independent of JAX.** A C / Go / C# implementation of Yee adjoint with O(boundary) memory is feasible and zero-dep. This is the single most important architectural lesson from the 2024 SOTA for reality.

### scikit-rf — pure-Python network analysis, BSD
1. **Object-oriented `Network` representing N-port S/Z/Y/ABCD parameters; cascade, de-embed, time-gate, vector-fit; calibration suite (SOLT, TRL, Multiline-TRL, SDDL, 8/16-term, Unknown-Thru); Touchstone read/write; offline VNA calibration** ([scikit-rf docs](https://scikit-rf.readthedocs.io/), [calibration source](https://github.com/scikit-rf/scikit-rf/blob/master/skrf/calibration/calibration.py)).
2. **Engineering choice:** **`complex128` ndarray as the core data type, every transformation as a pure function.** No PDE solver, no mesh, no geometry — closed-form algebra on (n_freqs, n_ports, n_ports) tensors. The library is "scientific calculator" not "simulator".
3. **Portability to reality:** **Highest portability of any SOTA tool listed here.** Every algorithm is closed-form with citation, every operation is a tensor algebra primitive over `complex128`. Touchstone I/O is the only thing out-of-scope (file format) — the math is all in. **Reality should explicitly target scikit-rf algorithmic parity in `em/` (closed-form S-param algebra) without inheriting the I/O surface.** Estimated 700 LOC over T1-COMPLEX + T1-TLINES + T1-SMITH (from 062). This is the **single most productive engineering choice on the table.**

### qiskit-metal + pyEPR — quantum-EM codesign
1. **EPR (energy-participation-ratio) method: classical eigenmode simulation of superconducting layout (driven by external HFSS/Sonnet) + post-processing to extract quantum Hamiltonian (qubit frequencies, dispersive shifts, anharmonicities)** ([qiskit-metal](https://github.com/qiskit-community/qiskit-metal), Minev 2021).
2. **Engineering choice:** **Decouple the simulator from the quantizer.** The classical EM is solved by whatever (HFSS, Palace, AWR), then pyEPR computes E_J participation per mode → effective Hamiltonian. The math primitive is **eigenvector × material-energy-density integration**, post-processed to mode participations.
3. **Portability to reality:** **The post-processing is in-scope; the eigenmode solver is not.** If reality ever ships a sparse generalised-eigenvalue routine in `linalg/`, an EPR helper in `em/` is ~300 LOC of energy-density integration. Until then this is downstream-app territory.

### Differentiable EM 2024-2025 research wave
1. **Meent** (arXiv 2406.12904) — JAX RCWA (rigorous coupled-wave analysis) for periodic structures, gradient-based design.
2. **Theory-guided RNN-FDTD** (Guo, IEEE TGRS 2021 / 2024 follow-ups) — recurrent neural network whose forward pass is structurally identical to Yee; enables joint inverse-problem + neural-prior solving.
3. **Physics-Informed Neural Networks** (Raissi 2019, EM applications 2024-2026) — neural-net ansatz minimising Maxwell residual at collocation points; useful where data is sparse, useless against FDTD on dense geometry.
4. **Engineering common ground:** every 2024-2026 EM-ML paper sits on top of either FDTD-AD (Tidy3D/FDTDX/Meent path) or PINN. Both are out-of-scope for reality; both compose **on top of** an in-scope FDTD kernel.

---

## The eight engineering decisions

Distilled from the survey above. These are the choices reality must make explicitly, before EM scope-creep makes them implicit.

### D1 — Time-domain or frequency-domain first?
SOTA: MEEP/openEMS/CST/Tidy3D/FDTDX = time. HFSS/COMSOL = frequency. FEKO = both. scikit-rf = frequency-only.
**Reality choice:** **Frequency-first.** scikit-rf parity is the cheapest first-light (~700 LOC) and covers practitioner workflow. Time-domain (FDTD) is multi-month and hits CLAUDE.md Rule 3 (no allocations) the hardest because field arrays are large. Defer FDTD to v0.20+.

### D2 — `complex128` everywhere, or staged?
SOTA: every modern EM tool is `complex<double>`-native at the surface. Reality is `float64`-only today.
**Reality choice:** **Introduce `complex128` as the second numeric type, scoped to `em/`.** Per 062 §T1-COMPLEX. Golden files extend trivially: store `[real, imag]` pair per scalar. Cross-language: Python has `complex`, C++ has `std::complex<double>`, C# has `System.Numerics.Complex` — IEEE-754 compliant in all four. **No new dep.**

### D3 — Yee stencil or FIT discrete-form?
SOTA Cartesian: equivalent. SOTA non-Cartesian: FIT generalises better.
**Reality choice:** **Document the math as FIT discrete forms, implement Cartesian as Yee** (equivalent, more readable). The FIT exposition (curl matrix `C`, mass matrices `M_ε`, `M_μ`) maps cleanly to `linalg/` sparse. This forecasts the path to non-orthogonal meshes without committing to it now.

### D4 — Adjoint method: hand-roll or wait for autodiff?
SOTA: Tidy3D and FDTDX both eventually moved to native autograd. The early MEEP adjoint plugin was hand-rolled.
**Reality choice:** **Hand-rolled adjoint Yee** when FDTD lands. The reverse-pass stencil is deterministic, golden-testable, and zero-dep. It is also a forcing function on the forward kernel: any allocation in the forward path doubles in the reverse path, so the no-alloc rule is enforced naturally. Compose with reality's autodiff package later as a wrapper.

### D5 — Method-dispatch (FEKO style) vs single-method?
FEKO ships MoM+FEM+MLFMM+PO+UTD; MEEP ships only FDTD; scikit-rf ships only network algebra.
**Reality choice:** **Single canonical method per problem class, no auto-dispatch.** Reality is a math library, not a solver suite. The user picks the method by importing the right sub-package: `em/network` (S-param), `em/fdtd` (time-domain), `em/mom` (integral equation, optional). No "smart" routing.

### D6 — Special-function library: bundle or skip?
Every closed-form antenna/waveguide formula needs Bessel, elliptic, Legendre, or Si/Ci. None exist in reality today.
**Reality choice:** **Bundle into a dedicated `special/` package**, not into `em/`. T1-SPECIAL (062) at ~600 LOC unlocks not just `em/` but `chaos/`, `optim/`, `signal/`, `prob/` use-cases. This is the highest-reuse single addition to reality. Source-cite Numerical Recipes §6 / Cephes / NIST DLMF.

### D7 — Mesh ownership: bring your own
SOTA: COMSOL/HFSS/CST own the mesh; openEMS uses graded-hex (orthogonal, no mesher needed); MEEP uses Cartesian (no mesher needed); scikit-rf has no mesh at all.
**Reality choice:** **Cartesian + graded-hex only; user supplies arbitrary meshes for FEM/MoM if those land.** Tetrahedral meshing is explicitly out-of-scope (CGAL territory). This matches openEMS and aligns with `linalg/` strided-array idioms.

### D8 — Parallelism: no SIMD, no MPI, no GPU
SOTA: openEMS does SIMD+MPI; Tidy3D/FDTDX do GPU; CST does GPU+MPI; HFSS does MPI.
**Reality choice:** **Single-threaded, deterministic, no SIMD intrinsics.** Go's portable code path is the cross-language golden-file contract; per-arch SIMD breaks it. Downstream services parallelise by slicing the simulation domain or sweeping frequencies — a `for ω` loop is embarrassingly parallel above the library boundary. **Reality stays the inner kernel; threading is the consumer's job.** This matches CLAUDE.md §3 ("no allocations in hot paths" — concurrent allocators would violate this) and §5 (precision documented, not assumed — non-deterministic floating-point reductions break golden files).

---

## What 062 said versus what 063 adds

062 enumerated **what** is missing (170-220 primitives, 4 tiers, ~6,500-9,500 LOC). 063 names **how** the SOTA built each tier and **which architectural choices** to copy. Cross-reference:

| 062 tier | SOTA precedent | Engineering lesson for reality |
|---|---|---|
| T1-COMPLEX | scikit-rf | `complex128` ndarray as base type; closed-form everywhere |
| T1-SPECIAL | Cephes / Boost.Math | Bundle in `special/`, not `em/` |
| T1-FIELDS | Griffiths Ch. 2-5 closed-form | Vector ops on top of `linalg.Vec3`, no new abstraction |
| T1-TLINES | scikit-rf, Pozar Ch. 3 | 80-LOC closed-form, ship first |
| T1-SMITH | scikit-rf, ADS | 150-LOC bilinear, ship first |
| T1-DIPOLE | NEC2, Balanis Ch. 4 | Sommerfeld-Norton ground out-of-scope (numerical Hankel); free-space + half-space-image only |
| T1-WAVEGUIDE | Pozar Ch. 3, openEMS validation | Needs T1-SPECIAL Bessel zeros — block on `special/` |
| T1-ARRAY | Balanis Ch. 6, scikit-rf | Closed-form, no special functions for uniform/linear |
| T2-MATERIAL | MEEP Drude/Lorentz/Debye | `complex128(ω)` evaluators, ~200 LOC |
| T3-FDTD | openEMS architecture, FDTDX reversibility | Operator-extension factoring; hand-roll adjoint per D4 |
| T3-MOM | FEKO, NEC2 | RWG basis on user-supplied triangle mesh; no mesher |
| T3-FEM | HFSS, COMSOL | Nédélec edge elements on user-supplied tet mesh; sparse eigensolve in `linalg/` first |
| (new) T3-RCWA | Meent, S4 | Periodic-structure spectral method, Fourier-modal — alternative to FDTD for layered geometries |

---

## Three concrete recommendations

1. **Ship scikit-rf parity in pure Go before any FDTD work.** T1-COMPLEX + T1-TLINES + T1-SMITH + T1-FIELDS = ~700 LOC over 4 weeks, covers ~80% of practitioner RF/microwave workflow, zero new dependencies, golden-file shape unchanged. **This is the highest-leverage single sprint in `em/`'s roadmap.**
2. **Bundle special functions in `special/` before any waveguide / antenna pattern work.** T1-SPECIAL ~600 LOC, source-cite Cephes / DLMF, unlocks `em/`+`chaos/`+`optim/`+`signal/` simultaneously. **Block T1-DIPOLE / T1-WAVEGUIDE on this; do not duplicate Bessel inside `em/`.**
3. **For FDTD, copy openEMS architecture and FDTDX reversibility lesson.** Operator-extension factoring (Yee + PML + dispersive material as independent extensions, all golden-tested) + hand-rolled adjoint Yee for inverse design (per D4). **Explicitly cite-and-skip MoM, FEM, MLFMM, RCWA — these are downstream / domain-specific solver territory, not reality's mandate.**

The line reality should not cross is **mesh generation** (out-of-scope per CLAUDE.md, not portable, not zero-dep) and **GPU/SIMD/MPI** (breaks cross-language golden-file determinism). Everything else in the SOTA — including subpixel smoothing, adjoint inverse design, dispersive materials, S-parameter algebra, Smith-chart workflow — is in-scope by mathematical character if the eight decisions above are made consistently.

---

## Citations

- MEEP — [docs](https://meep.readthedocs.io/), [Subpixel Smoothing](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/), [Adjoint tutorial](https://meep.readthedocs.io/en/latest/Python_Tutorials/Adjoint_Solver/), [github](https://github.com/NanoComp/meep).
- openEMS — [openems.de](https://www.openems.de/), [docs](https://docs.openems.de/intro.html), [github](https://github.com/thliebig/openEMS), [UPML source](https://github.com/thliebig/openEMS/blob/master/FDTD/extensions/operator_ext_upml.cpp).
- FEKO — [solver methods](https://help.altair.com/feko/topics/feko/user_guide/solver_solution_methods/solver_method_general_feko_c.htm), [hybrid FEM/MoM](https://2024.help.altair.com/2024/feko/topics/feko/example_guide/radar_cross_section/second_order_fss_fem_intro_feko_t.htm), [MLFMM/FEM](https://www.researchgate.net/publication/261130564_Theory_and_application_of_an_MLFMMFEM_hybrid_framework_in_FEKO).
- COMSOL / HFSS — [CST](https://www.3ds.com/products/simulia/cst-studio-suite/electromagnetic-simulation-solvers), [HFSS adaptive](https://ansyshelp.ansys.com/public/Views/Secured/Electronics/v251/en/Subsystems/HFSS/Subsystems/An%20Introduction%20to%20HFSS/Content/AdaptiveSolutionProcessanditsImportancetoHFSS.htm), [HFSS Phi mesh](https://semiwiki.com/eda/306866-a-mesh-by-any-other-name-the-hfss-mesh-evolution/).
- Tidy3D — [product](https://www.flexcompute.com/tidy3d/), [Autograd Intro](https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd1Intro/), [TidyGrad blog](https://www.flexcompute.com/blog/2024/10/31/tidygrad-the-easiest-to-use-inverse-design-tool-ever/).
- FDTDX — [arXiv 2412.12360](https://arxiv.org/abs/2412.12360), [github ymahlau/fdtdx](https://github.com/ymahlau/fdtdx), [docs](https://fdtdx.readthedocs.io/).
- scikit-rf — [docs](https://scikit-rf.readthedocs.io/), [calibration source](https://github.com/scikit-rf/scikit-rf/blob/master/skrf/calibration/calibration.py).
- qiskit-metal / pyEPR — [github qiskit-community/qiskit-metal](https://github.com/qiskit-community/qiskit-metal), Minev 2021.
- Meent — [arXiv 2406.12904](https://arxiv.org/abs/2406.12904).
- NEC2 / MMANA — [nec2.org](https://www.nec2.org/), [NEC-2 for MMANA](https://www.qsl.net/ua3avr/Read_me_Eng.htm).
- Sister reports — `agents/061-em-numerics.md`, `agents/062-em-missing.md`.

# 068 — fluids-sota

**Topic:** fluids: compare with OpenFOAM primitives, lattice-Boltzmann libs, scipy.optimize.brentq for Colebrook.
**Scope:** SOTA fluids-library comparison. Engineering / architecture choices, not feature checklists or numerics audits.
**Audited:** `C:\limitless\foundation\reality\fluids\fluids.go` (236 LOC, 11 functions).
**Sister reports:** 066 (numerics — 5-vector golden coverage, Re=2300 70% cliff, Colebrook abs-vs-rel-tol mismatch, Cd-Re self-consistency hole in TerminalVelocity); 067 (missing — 11 of ~250 canonical primitives, T1/T2/T3 sprint plan ~9,500 LOC). This report does **not** re-enumerate primitives or repeat the cliff/tolerance defects; it asks: of the engineering choices the fluids SOTA stack has converged on (2010-2026), which are reachable under reality's `zero-dep, one math.h, golden-file cross-language, no allocations in hot path, MIT` constraints, and which are explicit non-goals?

---

## Headline

**Reality's `fluids/` is in the same architectural class as Caleb Bell's [`fluids` Python package](https://github.com/CalebBell/fluids) — a closed-form engineering catalog — but ships 11 functions where Bell ships ~700.** Every modern *simulator* in the SOTA (OpenFOAM, Palabos, waLBerla, lbmpy, lettuce, XLB, JAX-Fluids, JAX-CFD, DualSPHysics, SPHinXsys, PySPH) sits on top of one of three engineering decisions reality has not yet made: **(A) field-of-floats grid surface** (FV cell topology, staggered MAC, Yee-equivalent), **(B) lattice-or-particle data layout** (D2Q9/D3Q19 SoA, neighbour-list spatial hash), or **(C) automatic differentiation over the time-stepping kernel** (JAX/Enzyme reverse-mode, or Yee-style time-reversibility). The SOTA's three independent revolutions of the past decade — **operator-extension factoring** (OpenFOAM `fvm`/`fvc`, openEMS `Operator_Ext_*`, Palabos plug-in collisions), **single-equation explicit Colebrook replacements** (Serghides 1984, Churchill 1977 — superseding `scipy.optimize.brentq` for engineering use), and **differentiable lattice/particle dynamics** (XLB 2024, JAX-Fluids 2.0 2024, diffSPH 2025) — are each downstream of one of (A)–(C). Reality has the floor (`linalg/`, `calculus/`, `optim/`, `geometry/`) but not yet a fluids-grid slab.

The fastest first-light SOTA-comparable artefact for `fluids/` is **Caleb-Bell-style closed-form correlation parity in pure Go** — Colebrook family (Serghides+Churchill+Haaland), compressible scalar relations (isentropic + normal/oblique shock + Prandtl-Meyer), open-channel hydraulics (Manning + Bélanger + critical depth), and the dimensionless-number library. ~1,800 LOC over T1 of 067 covers ~80% of practitioner thermofluids workflow, has zero portability friction, and is **architecturally identical to what reality already ships** — just bigger. Everything PDE-class (Navier-Stokes, LBM, SPH) is a multi-month engineering investment requiring a grid/particle data layer reality does not yet have; for those reality should explicitly **cite-and-skip** rather than half-ship a stub kernel.

---

## SOTA library survey

For each: (1) headline algorithm/feature, (2) engineering choice that defines it, (3) zero-dep portability for reality.

### OpenFOAM (CFD-Direct / OpenCFD, FV) — the academic + commercial reference

1. **`GeometricField<T,fvPatchField,volMesh>` over `polyMesh` + `fvm::ddt + fvm::div + fvm::laplacian == fvc::Su` operator-equation DSL.** A user writes `solve(fvm::ddt(rho,U) + fvm::div(phi,U) - fvm::laplacian(mu,U) == -fvc::grad(p))` and OpenFOAM dispatches each term to a discretisation scheme, assembles a sparse matrix, and hands it to a linear solver. The `fvm::` namespace produces matrix coefficients (implicit), `fvc::` produces field values (explicit). The user expresses *the equation*, not *the loop* ([Jasak FV chapter](https://wiki.openfoam.com/Finite_Volume_Discretization_by_Hrvoje_Jasak), [Solving PDEs with OpenFOAM](https://www.tfd.chalmers.se/~hani/kurser/OS_CFD_2011/highLevelProgramming.pdf)).
2. **Engineering choice:** **Operator-overloaded equation DSL on top of `Field<T>` + `polyMesh`.** Every PDE is `LinearSystem(matrix from fvm) + RHS (from fvc) → solve`. The split between `fvm` (assembles into sparse matrix `Ax = b`) and `fvc` (returns dense field result) is the single most important factoring in CFD code: it lets the same conservation law switch between implicit and explicit per-term without rewriting the equation. ~150 application solvers (`icoFoam`, `simpleFoam`, `pisoFoam`, `pimpleFoam`, `interFoam`, `rhoCentralFoam`, `chtMultiRegionFoam`, `dnsFoam`) are each ~100–300 LOC because the heavy lifting lives in the field/mesh/scheme classes.
3. **Portability to reality:** **The `fvm::`/`fvc::` factoring is the right abstraction; the `polyMesh` data structure is not.** OpenFOAM's polyMesh assumes unstructured I/O (file-based mesh, points + faces + cells indexed in three flat arrays); reality cannot ship the I/O surface, but the *algebraic* split between "matrix-assembly term" and "explicit field term" is implementable as two Go interfaces over `[]float64` slabs. **In-scope path:** ship a `fluids/fv` sub-package with a structured Cartesian mesh (regular hex, identified by `(nx,ny,nz,dx,dy,dz)` triple — no I/O), `fvm.Laplacian / fvm.Divergence / fvm.DDT` returning sparse-matrix triplets, `fvc.Grad / fvc.Div / fvc.Curl` returning new `[]float64` fields. Cost: ~600 LOC for the structured-grid case, contingent on `linalg.SparseCSR` (currently absent — gates this and 062 em-FDFD simultaneously). **Out-of-scope:** unstructured polyMesh, MPI domain-decomposition, runtime polymorphism via VTable+text-config (`controlDict`/`fvSchemes`/`fvSolution` is a parser + factory pattern, fundamentally I/O-shaped).

### Palabos (FlowKit, C++ LBM) — the modular collision-operator reference

1. **Plug-in collision/streaming/boundary architecture.** All incompressible/weakly-compressible models (BGK, MRT, TRT, regularized, recursive-regularized, raw-moments, central-moments, Hermite, central-Hermite, cumulant) are independent `Dynamics<T,Descriptor>` classes; the user picks one per cell at setup time. The streaming kernel is shared. The `MultiBlockLattice2D` / `MultiBlockLattice3D` data layout is a hierarchic block-structured grid with ghost layers, ~identical pattern to waLBerla's blockforest ([Palabos paper, ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0898122120301267)).
2. **Engineering choice:** **Lattice descriptor as compile-time template parameter + collision as runtime polymorphism.** `Descriptor` (e.g. `D2Q9Descriptor`) bakes lattice constants (weights, velocities, sound speed `c_s²=1/3`) into the type system at zero runtime cost; `Dynamics::collide` is a virtual call but only once per cell per step (negligible vs the streaming memory traffic). This factoring is what lets Palabos ship 30+ collision schemes without 30× code duplication.
3. **Portability to reality:** **Lattice-descriptor-as-type maps cleanly onto Go's generics (Go 1.18+, in toolchain).** A `type Lattice[D Descriptor] struct { f []float64 }` with `D2Q9` / `D3Q19` / `D3Q27` instantiations is ~200 LOC of pure data. Collision-as-interface is fine in Go: `type Collision interface { Collide(f []float64, rho float64, u []float64) }` — no allocations, hot path is a type-switched method call. **The hard part is not the kernel; it's the boundary-condition zoo** (bounce-back, halfway-bounce-back, Zou-He pressure/velocity, regularized BC, Guo extrapolation, curved-boundary Filippova-Hänel/Bouzidi-Firdaouss-Lallemand, immersed-boundary Peskin/Uhlmann). Each BC is ~30–80 LOC but cross-validating against Palabos reference outputs is a months-long exercise. **In-scope first light:** D2Q9 + BGK + bounce-back + Zou-He, ~400 LOC, single-block, periodic + no-slip BC. Cumulant LBM (Geier 2015) is ~300 LOC additional and is where the SOTA actually lives in 2025 for high-Re turbulence — citation-worthy but not first-light.

### waLBerla (FAU Erlangen, C++ LBM) — the HPC performance reference

1. **Block-forest hierarchical adaptive mesh + AVX-512 / CUDA kernel codegen + MPI-everywhere domain decomposition.** waLBerla holds the published 2024-2026 single-precision LBM performance crown at **3.12 GLUPS** on a single node ([TNL-LBM benchmark](https://link.springer.com/article/10.1007/s11227-026-08292-0)). The block-forest is a Z-order-traversed octree of fixed-size lattice blocks; each block is owned by one MPI rank; refinement/coarsening happens at block boundaries.
2. **Engineering choice:** **Codegen-the-kernel.** waLBerla's stencil kernels are emitted by `lbmpy` ([lbmpy on PyPI](https://pypi.org/project/lbmpy/)) — a Python DSL compiles a chosen collision operator (BGK / MRT / cumulant / entropic) + descriptor + boundary set into vectorized AVX-512 / CUDA C source at build time. The kernel is generated, not hand-written. This is the same play as TensorFlow XLA, JAX MLIR, and Halide image-pipeline.
3. **Portability to reality:** **Architecturally not portable to a zero-dep Go math library.** Codegen requires a build step, a DSL, a target IR, and per-arch SIMD intrinsics — all forbidden by reality's "one math.h, single binary, cross-language golden" contract. **The portable lesson is the layout, not the codegen:** Structure-of-Arrays (SoA) `f[direction][cell]` is faster than AoS `f[cell][direction]` for streaming; Z-Morton block ordering is cache-friendly; ghost layers should be one cell thick on each side. Reality should adopt SoA at first light (as a `type LatticeD2Q9 struct { f0,f1,...,f8 []float64 }` — verbose but cache-aligned without a build step). **MPI / GPU / SIMD codegen are explicit non-goals.**

### lbmpy + pystencils (FAU, Python codegen DSL)

1. **Symbolic-LBM-equation → optimized C/CUDA kernel pipeline.** User writes `Method.create_with_default_parameters(stencil="D3Q27", method="cumulant", ...)` in Python; lbmpy SymPy-derives the moment-space equilibrium, codegens an inline-vectorized kernel, drops it into a waLBerla / pystencils application.
2. **Engineering choice:** **Separate the *math* of the LBM scheme from the *implementation* of the loop.** The 30+ collision schemes in Palabos are 30 lbmpy lines because the moment-space transformation is computer-algebra not engineering.
3. **Portability to reality:** **The math half is in-scope as a code generator one-shot, not as a runtime dependency.** Run lbmpy *once* offline to derive the BGK/MRT/cumulant equilibrium expressions, paste them into reality as plain Go float64 expressions, golden-file-validate against lbmpy's reference output. This is the same trick used for elliptic-curve crypto in `crypto/` (curve constants computed offline, baked into source). **Pure runtime dependency: no.** **Pre-shipping derivation tool: yes, encouraged.**

### Lettuce (TU Dresden, PyTorch LBM) — the differentiable-LBM-for-ML reference

1. **PyTorch-tensor-as-lattice-distribution + torch.autograd over the streaming-collision step.** Lettuce uses `torch.Tensor` of shape `(Q, N_x, N_y, N_z)` as the lattice (Q = velocity directions); the collision and streaming operators are pure-tensor expressions; backprop falls out of `loss.backward()` ([Lettuce paper](https://dl.acm.org/doi/10.1007/978-3-030-90539-2_3)).
2. **Engineering choice:** **The lattice is a tensor.** Once that decision is made, neural-network LBM (Lettuce + Tensorflow LBM equivalents) becomes one line of `torch.optim.Adam(lattice.parameters())`. The cost is everything in a deep-learning framework — Python interop, GPU memory management, autograd tape.
3. **Portability to reality:** **Same as Tidy3D-vs-FDTD in 063-em-sota: the differentiable form is portable as a hand-rolled adjoint, not as an AD-framework dependency.** The LBM streaming-collision step is reversible-modulo-collision-relaxation; an adjoint LBM that recomputes-don't-store the forward pass is ~150 LOC over a working LBM kernel and gives O(boundary)-memory inverse design. This is the architecturally-clean way to inherit Lettuce's gradient capability under reality's zero-dep constraint. **PyTorch-as-runtime: no. Hand-rolled adjoint: yes, recommended after first-light forward solver.**

### XLB (Autodesk, JAX LBM) — the differentiable + multi-GPU SOTA

1. **JAX-pure-functional LBM scaling to billions of cells across multi-GPU/TPU; native automatic differentiation; physics-ML coupling.** XLB freezes its OO scaffold via `partial(jit, static_argnums=...)` so JAX can treat the simulator as a stateless function from `(f_in, params) → f_out` ([XLB paper, arXiv 2311.16080](https://arxiv.org/abs/2311.16080)).
2. **Engineering choice:** **Stateless functional kernels first; OO sugar second.** This is the JAX-CFD / JAX-Fluids playbook generalised to LBM. The simulator is a Python function with no in-place mutation; this lets JAX `vmap` over batch, `pmap` over devices, `grad` for inverse-design.
3. **Portability to reality:** **The functional-kernel discipline is exactly what reality already enforces** ("numbers in, numbers out"; no allocations in hot path; deterministic). XLB's architectural lesson is that *if your kernels are stateless and pure, you can bolt on AD/parallelism later without rewriting them.* Reality's "no I/O, no global state, no closures" rule already guarantees this. **The translation is direct:** any LBM kernel reality writes will be JAX-compatible by construction if a downstream service wants to wrap it. **JAX dependency itself: no.**

### JAX-Fluids 2.0 (TUM, JAX FV/FD compressible) — differentiable Riemann solvers

1. **Compressible Navier-Stokes 1D/2D/3D with HLLC/Roe/Rusanov Riemann + WENO5/WENO7 reconstruction + level-set two-phase + sharp-interface ghost-fluid + JAX autograd.** ([JAX-Fluids 2.0, arXiv 2402.05193](https://arxiv.org/abs/2402.05193); [github.com/tumaer/JAXFLUIDS](https://github.com/tumaer/JAXFLUIDS))
2. **Engineering choice:** **End-to-end differentiable physics for ML-supported CFD.** Train a closure model by backpropagating through the entire Riemann + WENO + RK time-stepping. Every kernel is a JAX `jit`-compiled pure function over `jnp.ndarray`.
3. **Portability to reality:** **The Riemann solver suite is in-scope as plain Go float64 arithmetic** (HLLC = ~150 LOC, exact-Riemann = ~250 LOC, Roe = ~120 LOC; see 067 §T3-NS-1D). WENO5 reconstruction is ~80 LOC of stencil arithmetic — this is the canonical 5th-order non-oscillatory scheme used in every shock-capturing CFD code in 2026, and it has zero non-math dependencies. **The 2024 architectural lesson** is that JAX-Fluids ships *the same Toro 2009 algorithms reality would ship*, just decorated with `@jit`. The math is identical; the wrapper is what reality cannot inherit. Excellent porting target.

### JAX-CFD (Google, JAX incompressible NS) — pseudospectral + ML-LES

1. **Spectral / staggered-grid incompressible NS with neural-network LES closure, periodic-box turbulence target.** [github.com/google/jax-cfd](https://github.com/google/jax-cfd).
2. **Engineering choice:** **Spectral methods over a periodic box where physics permits.** The pseudospectral path uses `jnp.fft.fftn` for the Poisson solve and dealiasing — O(N log N) instead of O(N^1.5) for a multigrid solver, and 2nd-order-accurate-in-time becomes "machine-precision in space" because spectral derivatives are exact for periodic functions.
3. **Portability to reality:** **Reality already has FFT in `signal/` — pseudospectral NS is one of the cheapest 2D NS solvers reality could ship**, modulo the periodic-box restriction. The trade-off vs FV is: pseudospectral wins for homogeneous turbulence and shear-flow research benchmarks (Taylor-Green vortex, Kolmogorov flow); FV wins for engineering geometry. **Recommend pseudospectral as the *first* NS solver in reality** because (i) FFT dependency is in-house, (ii) no boundary-condition zoo, (iii) golden-file-validatable against analytic Taylor-Green decay rate. ~400 LOC for 2D incompressible.

### DualSPHysics (Vigo + Manchester, GPU SPH) — the engineering SPH reference

1. **Weakly-compressible SPH (WCSPH) on CUDA + multi-GPU; dynamic boundary particles; δ-SPH and δ-R-SPH (Riemann-stabilized) for free-surface stability** ([DualSPHysics+, ScienceDirect 2024](https://www.sciencedirect.com/science/article/pii/S0010465524003126)).
2. **Engineering choice:** **Particles as flat float arrays, neighbour-list as O(N) cell-linked-list spatial hash.** All the engineering-SPH performance work since 2010 has been on (i) the cell list, (ii) symplectic time-stepping, (iii) viscosity correction (artificial-Monaghan vs δ-SPH vs Riemann), and (iv) wall boundary conditions (dynamic boundary particles vs ghost-particle vs Marrone open boundary). The kernel evaluation itself is ~10 LOC.
3. **Portability to reality:** **Particle layout + neighbour list is in-scope; CUDA / MPI is not.** A serial cell-linked-list neighbour finder is ~150 LOC and depends only on `geometry/` (already shipped). The five standard SPH kernels (Gauss, cubic spline, Wendland C2, quintic spline, super-Gaussian) are ~30 LOC each. WCSPH momentum equation is ~80 LOC. δ-SPH density-diffusion is ~30 LOC. **Total first-light SPH:** ~400 LOC, golden-validatable against the dam-break and Taylor-Green-on-particles benchmarks DualSPHysics ships. The boundary-condition zoo is the multi-month follow-on.

### SPHinXsys (TUM, C++ SPH multi-physics)

1. **C++-template-based SPH framework for fluid-structure interaction, multibody dynamics, reactive flow; CPU-only TBB parallelism.** Cited as architecturally cleaner than DualSPHysics but slower because no GPU.
2. **Engineering choice:** **Multi-physics-first SPH — fluid + solid + structural + multibody share one particle data structure.** The library's name encodes "industrial complex systems"; the design assumes the user wants FSI, not just dam-break.
3. **Portability to reality:** **The multi-physics ambition is out-of-scope for `fluids/` proper but matches `reality`'s package-decomposition philosophy.** Reality already separates `fluids/` from `physics/`; if SPH ever lands, the FSI extension would compose `fluids.SPHParticleSet` with `physics.RigidBody` from the constants/physics layer. SPHinXsys is the architectural template for that composition; it doesn't bring portable kernels reality wouldn't get from DualSPHysics anyway. **Cite-and-skip.**

### diffSPH (2025, JAX/PyTorch differentiable SPH)

1. **Differentiable SPH for adjoint optimization and ML; gradient flow through the entire particle simulation** ([arXiv 2507.21684](https://arxiv.org/html/2507.21684v1)).
2. **Engineering choice:** **Same recompute-don't-store insight as FDTDX (063-em-sota) and Tidy3D, applied to SPH.** SPH time-stepping is reversible (in the inviscid limit); adjoint SPH replays the forward pass backward and accumulates gradients with O(boundary) memory.
3. **Portability to reality:** **Same answer as Lettuce: hand-rolled adjoint SPH is ~200 LOC over a working forward SPH kernel; framework dependency is no.** This is the 2025 SOTA architectural insight that reality should plan for at SPH-first-light: write the forward SPH kernel in a way that admits a reverse-mode adjoint without a tape (deterministic, time-symmetric, no Python-style `random_state` hidden in the kernel).

### PySPH (CSIR, Python+Cython SPH) — the academic reference

1. **Python-DSL + Cython codegen for SPH equations; user writes the equation, PySPH compiles to a fast C kernel.**
2. **Engineering choice:** **Equation-as-Python-class, codegen-the-loop.** Same play as lbmpy+pystencils for LBM.
3. **Portability to reality:** **As lbmpy: math half is in-scope offline; runtime DSL is not.** Use PySPH offline to derive the SPH equation forms; bake plain Go expressions into reality. **Pure runtime dependency: no. Pre-shipping derivation: yes.**

### Caleb Bell `fluids` (PyPI, MIT, pure-Python) — the closed-form-correlation reference

1. **~700 functions: friction factors (15 explicit Colebrook variants — Serghides, Zigrang-Sylvester, Romeo-Royo-Monzon, Eck, Buzzelli, Manadilli, Brkic, Goudar-Sonnad, Vatankhah-Kouchakzadeh, Sonnad-Goudar, Avci-Karagoz, ...), 50+ pipe-fitting K-factors (Crane / Hooper / Darby), open-channel hydraulics, two-phase flow correlations (Lockhart-Martinelli, Friedel, Müller-Steinhagen-Heck, Chisholm, ...), pump/compressor/control-valve sizing, atmospheric models (US Standard 1976, NRLMSISE-00), particle settling, packed-bed pressure-drop (Ergun, Kuo-Nydegger), drag coefficients for 30+ shapes** ([github.com/CalebBell/fluids](https://github.com/CalebBell/fluids)).
2. **Engineering choice:** **Pure-Python with optional SciPy/NumPy.** "Fluids was originally tightly integrated with SciPy and NumPy; today they are optional components used for only a small amount of functionality which do not have pure-Python numerical methods implemented." Bell explicitly *re-implemented* root-finding internally to avoid forcing a SciPy dependency — this is the closest existing analog to reality's zero-dep philosophy.
3. **Portability to reality:** **THE closest architectural sibling in the SOTA, and the right comparison frame for `fluids/`.** Bell's package proves that a 700-function closed-form-correlation library *can* be written without a numerical-stack dependency. Every algorithm is a closed-form fit + named bibliographic reference. **The reality-vs-Bell delta is purely surface area, not architectural philosophy.** Reality should treat Bell's `fluids` as the authoritative cross-validation target for closed-form correlations: every `fluids.X` function reality ships should produce identical (to spec ULP) output to the corresponding `Bell.fluids.X` for the published valid range. **In-scope: total parity at ~700-function scale (multi-month). First-light parity: Colebrook family + Moody + relative-roughness table + open-channel + pipe-fitting K-factors, ~1,000 LOC, ~3 weeks.** This is the highest-ROI architectural alignment in this report.

### `scipy.optimize.brentq` — the canonical Colebrook root-finder

1. **Brent's method = bisection + secant + inverse quadratic interpolation, guaranteed-converge in `[a,b]` if `f(a)·f(b)<0`** ([scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html)). Standard scipy idiom for Colebrook is `from scipy.optimize import brentq; f = brentq(lambda f: 1/sqrt(f) + 2*log10(eps/(3.7*D) + 2.51/(Re*sqrt(f))), 1e-6, 1.0)`.
2. **Engineering choice:** **Bracketing root-finder as the *general* tool, accept the cost (15-30 function evaluations) for guaranteed convergence.** Brent is in `scipy.optimize` because it's the right *generic* answer — bisection is too slow, Newton needs a derivative and can diverge, secant can flip outside the bracket. Brent always converges and never leaves the bracket.
3. **Portability to reality:** **`optim/` already has bisection + Newton + L-BFGS** (per CLAUDE.md). Adding Brent's method is ~60 LOC and **closes a real gap** — Brent is the textbook Colebrook solver, and any cross-language port that follows scipy convention will use it. **Architectural insight:** Reality should ship `optim.Brent` and use it as the *internal* Colebrook solver; users get an *explicit* Colebrook (Serghides — 0.0023% error in 3 ops, no iteration) by default and can request the iterative-Brent path with named-flag for golden-file ULP-parity with scipy. This is the same "two paths, both pinned" pattern crypto/ uses for primality (Miller-Rabin probabilistic + AKS deterministic). **In-scope: yes, valuable, ~60 LOC for Brent + ~50 LOC for Serghides + ~20 LOC for Brent-validated Colebrook.**

### scikit-aero / aerocalc / fluidfoam (Python aerospace utilities)

1. **Closed-form aerospace fluids: ICAO-1993 / ISA / US-Standard-1976 atmosphere, isentropic + normal shock + Prandtl-Meyer + Rayleigh + Fanno tables, conversion utilities.**
2. **Engineering choice:** **Same "scientific calculator" pattern as scikit-rf in the EM space:** every routine is a closed-form algebraic expression over float64; no PDE solver; no mesh; no I/O.
3. **Portability to reality:** **Same answer as scikit-rf for em/: highest portability of any tool listed here, every algorithm is closed-form with citation.** Reality should target scikit-aero algorithmic parity in `fluids/T1-COMPRESSIBLE` (067 §T1) — ~400 LOC. Atmosphere models are a clean addition that overlaps physics/ + constants/; recommend `physics.Atmosphere(h)` rather than `fluids.Atmosphere`.

### CFD-Python / Lorena Barba "12 steps to Navier-Stokes" — the pedagogy reference

1. **Pure NumPy, ~50 LOC per step, 12 Jupyter notebooks ascending from 1D linear convection (step 1) to 2D incompressible Navier-Stokes channel + cavity (step 12)** ([JOSE paper](https://www.theoj.org/jose-papers/jose.00021/10.21105.jose.00021.pdf), [github.com/barbagroup/CFDPython](https://github.com/barbagroup/CFDPython)).
2. **Engineering choice:** **Minimum viable PDE solver — explicit time-stepping, central differences, no abstractions.** Step 12 is 2D incompressible NS in ~120 lines. The simplicity is the point: students see exactly what the algorithm does without a `class GeometricField` to learn first.
3. **Portability to reality:** **The 12-steps progression is the *exact correct shape* of reality's NS first-light** because (i) every step is golden-file-testable on its own, (ii) each step is ~50 LOC of pure float64 arithmetic, (iii) no I/O, (iv) the final 2D NS in a unit box can be cross-validated against the lid-driven-cavity Ghia-Ghia-Shin 1982 benchmark ULP-by-ULP. **Recommend reality's `fluids/T3-NS-1D` and `fluids/T3-NS-2D` follow Barba's curriculum order:** linear convection → nonlinear convection (Burgers) → diffusion → Burgers-with-viscosity → 2D linear convection → 2D nonlinear convection → 2D diffusion → 2D Burgers → Laplace 2D (Poisson) → Poisson 2D → cavity 2D NS → channel 2D NS. ~600 LOC total for all 12 steps in Go, every step independently shippable. **This is the highest-clarity engineering decision available** for reality's NS surface.

### pyVista / FluidDyn — visualisation + research-pipeline ecosystems

1. **pyVista wraps VTK for 3D mesh/field plotting, FluidDyn is a Python ecosystem for fluid-research (FluidSim, FluidFoam, FluidLab, ...).**
2. **Engineering choice:** **Visualisation and pipeline orchestration are downstream of the math.** pyVista is rendering, not simulation; FluidDyn is workflow, not kernels.
3. **Portability to reality:** **Out of scope (visualisation = I/O = downstream service territory).** Reality ships `[]float64` outputs; downstream services hand them to pyVista / Plotly / VTK. **No-op for `fluids/`.**

### SU2 (Stanford / Université de Liège, C++) — discrete-adjoint CFD

1. **RANS / LES on unstructured grids, full discrete-adjoint via algorithmic differentiation (CoDiPack), shape optimization.**
2. **Engineering choice:** **Discrete-adjoint via tape-recording AD over the entire CFD solver.** Whatever the forward kernel does, the tape replays it backward.
3. **Portability to reality:** **Discrete-adjoint over CFD requires a working AD layer applied to a working CFD solver, neither of which `fluids/` ships today.** Continuous-adjoint math (the Hadamard shape derivative, the adjoint Navier-Stokes RHS — 067 §T3-OPTIMIZATION-COUPLED) is closed-form and in-scope for ~300 LOC. Discrete-adjoint via reality's `autodiff/` package is in-scope once both `fluids/T3-NS` and `autodiff/T2-vmap+vector` land — multi-year horizon. **Cite-and-skip for now; revisit when both prereqs land.**

### Recent JCP / JFM 2024-2026 architectural papers worth tracking

1. **Cumulant LBM (Geier 2015, refined 2023-2025)** — single-relaxation-time-quality stability at MRT computational cost; the SOTA collision operator for high-Re LBM. ~300 LOC if reality ever ships LBM seriously.
2. **Differentiable turbulence closure on unstructured grids (arXiv 2307.13533, JCP 2024 follow-ups)** — graph neural networks for SGS LES, end-to-end training via differentiable physics. **Out-of-scope** for reality's math kernels but the *baseline* algebraic closures (Smagorinsky, WALE, Vreman, σ-model) are ~150 LOC and in-scope per 067 §T2-TURBULENCE-CLOSED.
3. **Physics-Informed Neural Networks for fluids (Raissi 2019 + 2024-2026 wave)** — neural-net ansatz minimizing PDE residual at collocation points. **Out-of-scope** (requires NN + AD + sampler stack); cite-and-skip.
4. **JAX-LaB (AGU 2026)** — JAX multiphase LBM for geosciences. Same architectural class as XLB; same portability answer.
5. **TNL-LBM (J. Supercomputing 2026)** — template-numerical-library LBM, scalable benchmark. Performance reference for waLBerla-class throughput; not architectural news.

---

## The seven engineering decisions that define `fluids/`'s next 18 months

These are the architectural pivots reality's `fluids/` package has to make — explicitly, with golden-file pins, before *any* PDE solver can land. Each maps to a concrete SOTA precedent.

### D-1. Closed-form-correlation parity vs PDE-solver kernels — **closed-form first**
**SOTA precedent:** Caleb Bell `fluids` (700 fns, all closed-form, MIT, pure Python). 
**Decision:** Ship Bell-class closed-form parity (Colebrook family, K-factors, atmosphere, settling, two-phase Lockhart-Martinelli, packed-bed Ergun) at ~1,000 LOC over T1 of 067 *before* committing engineering capacity to the FV/LBM/SPH grid layer. **This is the highest-ROI move.** Closed-forms are golden-file-trivial, port across 4 languages with zero friction, and serve real engineering consumers today. PDE solvers are months-of-work artefacts that need a grid abstraction reality has not committed to.

### D-2. Explicit Colebrook (Serghides 1984) as default + Brent-iterated Colebrook for ULP-parity — **two paths, both pinned**
**SOTA precedent:** scipy.optimize.brentq + Caleb Bell's 15-explicit-Colebrook catalog. 
**Decision:** Ship `fluids.ColebrookSerghides(Re, ε/D)` as the default explicit (0.0023% vs implicit Colebrook, 3 log10 calls, no iteration), `fluids.ColebrookExplicitChurchill(Re, ε/D)` for all-Re smoothness (closes 066 N-1 cliff), and `fluids.ColebrookIterativeBrent(Re, ε/D)` as the scipy-parity bracketing-root path. Cross-language ports validate the Brent path against scipy.brentq output to ULP. Adds `optim.Brent` as a side-effect (~60 LOC, fills a real gap in `optim/`). **Total: ~150 LOC, closes 066 N-1+N-2 plus topic-prompt-headline Colebrook+Moody.**

### D-3. Lattice descriptor as compile-time generic, collision as interface — **Palabos pattern, Go-generics-native**
**SOTA precedent:** Palabos `Dynamics<T,Descriptor>`, lbmpy moment-space derivation. 
**Decision:** When LBM lands (T3, multi-month), use Go generics for the lattice descriptor (`type Lattice[D Descriptor] struct{...}` with D2Q9/D3Q19/D3Q27 instantiations) and a `type Collision interface { Collide(...) }` for the runtime-polymorphic operator. Golden-file-validate BGK / MRT / cumulant against lbmpy reference outputs. **No codegen, no DSL, no runtime dependency on Python.** Estimated ~400 LOC for D2Q9-BGK first-light + ~200 LOC each for MRT and cumulant.

### D-4. Pseudospectral periodic-box NS first, FV-Cartesian second — **JAX-CFD pattern**
**SOTA precedent:** JAX-CFD periodic-box NS, Lorena Barba 12-steps for FV. 
**Decision:** Reality already ships FFT in `signal/`. The cheapest 2D incompressible NS solver reality could write is pseudospectral on a periodic box — dealiased Galerkin, RK4 in time, FFT-Poisson for the pressure-projection step. ~400 LOC, golden-file against the analytic Taylor-Green decay rate. FV-Cartesian (Barba step 12) follows as the second NS path; ~600 LOC, golden against Ghia-Ghia-Shin 1982 lid-driven cavity. **The two NS paths cross-validate each other at the 2D periodic case and share the timestepper.**

### D-5. SoA particle/lattice layout + cell-linked-list neighbour finder — **DualSPHysics pattern, no GPU**
**SOTA precedent:** DualSPHysics CUDA-SoA particles; waLBerla SoA lattice; Palabos AoS hybrid. 
**Decision:** When SPH lands (T3, multi-month), use Structure-of-Arrays for particle attributes (`positions, velocities, densities []float64` not `particles []Particle`) and a plain serial cell-linked-list spatial hash (~150 LOC, depends on `geometry/`). The five standard kernels (Gauss, cubic-spline, Wendland-C2, quintic-spline, super-Gaussian) are shared between SPH and any kernel-density-estimation use elsewhere. **No GPU, no MPI, no SIMD intrinsics.**

### D-6. Adjoint kernels via time-reversibility, not via tape — **FDTDX 2024 / diffSPH 2025 / Lettuce architectural lesson**
**SOTA precedent:** FDTDX 2024 reversible Yee adjoint (063 §FDTDX); diffSPH 2025 reversible SPH adjoint; Lettuce torch.autograd-tape (the path *not* taken). 
**Decision:** When adjoint LBM / NS / SPH ever lands, hand-roll the reverse pass exploiting time-reversibility (replay forward backward, recompute-don't-store, O(boundary) memory). **Do not** depend on `autodiff/` for the time-stepping loop — the gradient is a closed-form property of the symplectic forward integrator, not an AD artefact. **autodiff/** remains in-scope for closed-form derivative needs (e.g. Jacobian of a Riemann flux); but the time-stepping adjoint is a hand-rolled artefact. ~150-300 LOC per solver class.

### D-7. Cite-and-skip the ML-fluids frontier — **PINN, GNN-LES, neural-operator-NS**
**SOTA precedent:** Differentiable-turbulence-closure papers, JAX-Fluids 2.0 ML hooks, Lettuce neural-LBM. 
**Decision:** None of the 2024-2026 ML-fluids papers ship math reality should incorporate. They ship architectural patterns (differentiable physics, neural closures, generative super-resolution) that compose **on top of** an in-scope forward solver. **Reality's job is to ship the forward solver in a form that *admits* downstream ML composition** (stateless kernels, deterministic outputs, fp64 reproducibility) — which the existing CLAUDE.md rules already enforce. No active work; cite-and-skip; revisit if/when reality's `autodiff/` reaches Tier-2 maturity per 013-autodiff-sota.

---

## What's most aligned and most missing relative to SOTA

| Engineering choice | Reality today | SOTA precedent | Gap |
|---|---|---|---|
| Closed-form-correlation catalog | 11 fns | Bell `fluids` 700 fns | **+689 fns** (T1 sprint plan = +110, reaches 16% of Bell) |
| Pure functional kernels | YES (CLAUDE.md) | XLB, JAX-CFD, JAX-Fluids | aligned by construction |
| Operator-extension factoring | N/A (no PDE solver) | OpenFOAM `fvm`/`fvc`, openEMS `Operator_Ext_*` | gates on `linalg.SparseCSR` |
| Lattice descriptor as type | N/A (no LBM) | Palabos `Descriptor` template | reachable via Go generics |
| SoA particle layout | N/A (no SPH) | DualSPHysics, waLBerla | trivial when needed |
| Brent / Serghides root-finder | NO | scipy.brentq | gap → +60 LOC (`optim.Brent`) |
| Codegen / DSL kernels | N/A (forbidden) | waLBerla+lbmpy, PySPH | architectural non-goal |
| GPU / MPI | N/A (forbidden) | Tidy3D, FDTDX, XLB, DualSPHysics | architectural non-goal |
| AD over time-stepping | N/A | Lettuce, XLB, JAX-Fluids, diffSPH | hand-rolled adjoint per D-6 |
| 2024-2026 ML-fluids | N/A | PINN, GNN-LES, neural-operator-NS | cite-and-skip per D-7 |

**Five non-goals are SOTA features reality must explicitly decline:** GPU acceleration, MPI domain decomposition, Python-DSL codegen, runtime AD-framework dependency, neural-network closures. Each is correctly out-of-scope per CLAUDE.md; each should appear in package docs as a "we explicitly do not ship X" line so consumers know to compose externally.

**Five engineering choices reality should *adopt* from SOTA:** Bell-style closed-form-first surface (D-1), two-path Colebrook with Brent for ULP-parity (D-2), Palabos lattice-descriptor pattern when LBM lands (D-3), JAX-CFD pseudospectral-first NS (D-4), DualSPHysics SoA + spatial-hash when SPH lands (D-5).

---

## What would change if I had write access (Sprint 1 — closed-form-only)

1. Add `optim/brent.go` (~60 LOC) — bracketing root-finder, scipy-parity golden file. Closes the open gap noted in this report's D-2 + serves Colebrook + serves any future iterative-implicit fluids correlation.
2. Add `fluids/colebrook.go` (~250 LOC) — `Laminar`, `ColebrookSerghides`, `ColebrookHaaland`, `ColebrookSwameeJain`, `ColebrookChurchill` (closes 066 N-1), `ColebrookIterativeBrent` (scipy-parity), `Moody` alias. Generates `testdata/fluids/colebrook_family.json` with ≥30 vectors covering Re=10²–10¹⁰, ε/D=0–0.05, all six functions.
3. Add `fluids/dimensionless.go` (~120 LOC) — Mach, Froude, Prandtl, Schmidt, Lewis, Peclet, Grashof, Rayleigh, Nusselt, Sherwood, Stanton, Eckert, Strouhal, Womersley, Knudsen, Bejan, Brinkman, Dean. All closed-form, all golden-file-pinned.
4. Add package doc paragraph: "Reality's `fluids/` ships closed-form engineering correlations (Bell-class). PDE solvers (Navier-Stokes, LBM, SPH) are architectural follow-ons gating on `linalg.SparseCSR`. GPU / MPI / AD-framework dependencies are explicit non-goals; downstream services compose."
5. Add the seven D-1 through D-7 architectural decisions to `fluids/ARCHITECTURE.md` (single page) so future contributors can see why a given LOC pattern was chosen.

Total estimated effort for Sprint 1: 1 week, ~430 LOC, +60 golden vectors, plus one architecture doc. Closes 066 N-1+N-2 numerics plus topic-prompt-headline `scipy.optimize.brentq for Colebrook`.

---

**Auditor:** agent 068
**Status:** complete
**Next:** progress line appended; 14 SOTA libraries surveyed for architectural pattern, 7 engineering decisions named for reality's `fluids/` next 18 months, 5 SOTA non-goals correctly declined per CLAUDE.md.

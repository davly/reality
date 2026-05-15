# 354 — research-libs-julia (Julia/SciML ecosystem 2025-26 lessons)

## Headline
SciML proves a unified composable solver interface (LinearSolve/NonlinearSolve/Optimization/Integrals) is the right architecture for a math library; Julia's TTFX, dynamic dispatch, and source-transform AD are anti-patterns Reality should not import.

## Survey

### DifferentialEquations.jl (SciML, MIT)
The gold-standard ODE/SDE/DAE/DDE/PDE meta-package. Workshop "A Deep Dive Into DifferentialEquations" held July 2025 at JuliaCon by Rackauckas. The major v8 reorganization split out non-ODE solvers into separate packages (OrdinaryDiffEq, StochasticDiffEq, DelayDiffEq, etc.); the meta-package now re-exports a polyalgorithm with adaptive method selection. Architectural lesson for Reality: a single `Solve(problem)` entry point that polyalgorithmically dispatches to the best concrete solver is more user-friendly than 30 named functions, but it requires problem objects (`ODEProblem`, `LinearProblem`) — Reality's current free-function style is fine for primitives but should adopt problem-types for `chaos`, `optim`, and `control` where multiple algorithms compete. Anti-pattern: relying on multiple-dispatch for solver selection — in Go we should make algorithm choice explicit and named.

### ModelingToolkit.jl (SciML, MIT)
Symbolic-numeric acausal modeling, v9 series in 2025. Switched from Unitful.jl to DynamicQuantities.jl as the units backend (lighter, more compile-friendly), exports common `t`/`D` definitions, absorbed `connect`/`Connector` from Symbolics. Index-reduction (Pantelides), Tearing, and automatic Jacobian/Hessian generation from symbolic models. Architectural lesson: symbolic preprocessing of a numerical problem before solving (DAE index reduction, sparsity detection) is hugely valuable and Go cannot replicate it without a CAS — but Reality could ship pre-derived Jacobians as named code paths (e.g., explicit Jacobian for Lorenz, Van der Pol in `chaos/`). Anti-pattern: code generation from symbolic models at runtime — a build-time codegen step would be cleaner if we ever go that route.

### JuMP.jl (jump-dev, MPL-2.0)
Modeling DSL for LP/MIP/conic/SDP/NLP, separate from solver backends via MathOptInterface. 2025 roadmap: nonlinear programming as first-class citizen (done), non-Float64 coefficients, constraint programming, multi-objective. Future: SI units in models, vector user-functions for optimal control. Note license is MPL-2.0 not MIT — still permissive but file-level copyleft. Architectural lesson: clean separation of model definition (problem) from solver (engine) via a stable interface (MOI) is the most important pattern for long-term ecosystem health; Reality's `optim` package should formalize a `Problem`/`Solver` split rather than baking algorithm into function name. Anti-pattern: building a string-based DSL — Go's type system is enough.

### Distributions.jl (JuliaStats, MIT)
Probability distributions, v0.25.123 (Jan 2026), still the canonical PDF/CDF/sampler library after 12 years. Recent fixes: skip parameter checks for internal Normal construction in LogNormal, exported `gradlogpdf`, more efficient MvNormalCanon sampling. Single abstract type tree (`Distribution{F<:VariateForm,S<:ValueSupport}`) with `pdf`, `cdf`, `quantile`, `rand`, `mean`, `var`, `entropy` as the universal interface. Architectural lesson: Reality's `prob` package should expose a tight uniform interface across distributions (the trait-set above) so downstream code can write `dist.PDF(x)` without knowing the family — this is more important than adding more distributions. Anti-pattern: relying on multiple-dispatch overloads of `pdf(d, x)` — Go interfaces achieve the same with explicit method sets, which is preferable for golden-file determinism.

### Flux.jl + Lux.jl (FluxML, MIT)
Flux v0.16.5, Julia 1.10+. Layer-stacking DSL on top of Zygote source-transform AD; 100% pure Julia stack. Lux.jl is the more recent explicit-parameters alternative. Ecosystem: Transformers.jl, Metalhead.jl, FluxTraining.jl, GeometricFlux.jl. Out of scope for Reality (no NN layers in a math foundation), but Flux's "compose small layer types via function composition" pattern is worth noting — it works because Julia's compiler inlines the entire chain. Go cannot replicate this without code generation. Architectural lesson: keep ML out of Reality (it's correctly excluded) and let consumers build NN layers on top of `linalg`. Anti-pattern: implicit-parameter style (Flux pre-Lux) — explicit state passing aligns better with Reality's no-globals rule.

### Zygote.jl vs Enzyme.jl (FluxML / EnzymeAD, MIT)
2025 consensus: Zygote is a Julia-source transformation AD, fast for non-mutating code but fails on mutation; Enzyme operates at LLVM IR level, supports mutation, has an extensible rule system. For 2025: "Reactant.jl + Enzyme.jl" is the recommended best-performance backend; Zygote remains best on pure CPU without Reactant. DifferentiationInterface.jl (juliadiff) provides a unified frontend over 12+ AD backends. Architectural lesson: AD backend choice is hard and ecosystem-fragmenting — Reality's chosen approach (no built-in AD; consumers do their own) avoids the swamp. If Reality ever adds AD primitives, dual-number forward-mode is the only sensible default for a deterministic library. Anti-pattern: source-to-source AD over arbitrary user code — too magical, breaks under refactors.

### Symbolics.jl (JuliaSymbolics, MIT)
v6.57.0 (Oct 2025). Built on SymbolicUtils.jl term-rewriting; supports symbolic differentiation, equation solving, matrix algebra of symbolic expressions, code generation. Underpins ModelingToolkit. Achieves "symbolic expressions interoperate with stdlib functions" through Julia's multiple dispatch — `sin(x::Symbolic)` Just Works. Architectural lesson: a CAS is out of scope for Reality (would violate zero-deps and reimplement-from-first-principles), but Reality can pre-compute symbolic identities at code-write time and hard-code closed-form derivatives (e.g., gradient of every PDF in `prob/`, Jacobian of every Lorenz-family system). Anti-pattern: ad-hoc rewrite rules without a canonical normal form — leads to non-deterministic outputs.

### LinearSolve.jl (SciML, MIT)
v3.72 (early 2026). Unified interface over LU/QR/Cholesky factorizations + KLU/UMFPACK sparse + MKLPardiso + every Krylov package (Krylov.jl preferred over IterativeSolvers.jl/KrylovKit.jl). Polyalgorithm dispatch on matrix structure; caching of symbolic and numerical factorizations across solves with same sparsity. Architectural lesson: this is the single most important pattern Reality should steal — `linalg.Solve(A, b, opts...)` that automatically picks dense LU vs sparse Cholesky vs CG vs GMRES based on A's properties, with explicit override. Reality's current `linalg` exposes individual factorizations — adding a polyalgorithm wrapper would be high value. The caching pattern (re-solve with new b but same A) is also worth importing. Anti-pattern: tying solver choice to call site — should be runtime-dispatched on matrix properties.

### DataInterpolations.jl (SciML, MIT)
1D interpolation library: Linear, Quadratic, Cubic, Akima, Monotonic, BSpline (interpolation + approximation), regularization smoothing splines. Documentation refreshed Jan 2026. Architectural lesson: Reality lacks an `interp` package — this is a gap. The DataInterpolations API (`itp = CubicSpline(u,t); itp(x)`) is callable-object style, which Go can replicate via interface methods. Worth adding `interp` package with 1D linear/cubic/Akima/monotonic. Anti-pattern: returning a closure from a constructor — Go should return a struct with an `Eval(x)` method for golden-file friendliness and zero allocations.

### Manifolds.jl + Manopt.jl (JuliaManifolds, MIT)
Bergmann's group: Manopt.jl optimization on Riemannian manifolds, sister to Matlab Manopt and pymanopt. 2025 work: intrinsic Riemannian proximal gradient (convex and nonconvex preprints, arXiv:2507.16055 and 2506.09775). Built on ManifoldsBase.jl interface — algorithms work on any conforming manifold (Sphere, Stiefel, Grassmann, SPD, hyperbolic, etc.). Architectural lesson: separating abstract manifold structure (exp/log/parallel-transport/inner-product) from optimizer kernels is the right factoring; Reality could add a small `manifold` package (sphere + SPD + Stiefel) usable from `optim`. Out of scope for v0.10 but worth noting for v0.20. Anti-pattern: hardcoding Euclidean assumptions in optimizer code — `optim/` should accept a metric/retraction interface.

### ApproxFun.jl (JuliaApproximation, MIT)
Chebfun port to Julia. Functions are `Fun` objects with `space + coefficients`. Adaptive constructor: pass `exp` and it figures out 14 Chebyshev coefficients give machine precision. Operations (multiply, integrate, differentiate, solve ODE) preserve adaptivity. Architectural lesson: function approximation is a gap in Reality — `approxfn` package with Chebyshev polynomial fit + adaptive degree selection would be a compact valuable addition (~200 LOC). Anti-pattern: representing functions as opaque closures — the `Fun = (basis, coeffs)` representation is more inspectable, golden-file friendly, and allocation-free.

### IntervalArithmetic.jl (JuliaIntervals, MIT)
v0.22+, IEEE 1788-2015 compliant. Validated numerics — every operation produces a rigorous enclosure. Used by IntervalRootFinding.jl and IntervalOptimisation.jl for guaranteed root-finding and global optimization. Doc-built Nov 2025 on Julia 1.12.1. Architectural lesson: rigorous numerics is the ultimate form of golden-file testing — if Reality computed IA bounds for transcendentals at golden-generation time, we could prove "the true value lies in [lo, hi]" rather than picking an epsilon. Worth a `interval` mini-package with directed-rounding add/mul/sin/cos for use in test infrastructure (not hot paths). Anti-pattern: unchecked floating-point accumulation in golden generators — directed-rounding interval arithmetic should bound errors.

### JuliaStats: StatsBase.jl + GLM.jl (JuliaStats, MIT)
StatsBase v0.34.10 (Jan 2026): scalar stats, moments, ranks, covariances, sampling, empirical density. GLM.jl v1.9.0: linear/logistic/Poisson regression with formula DSL via StatsModels. Architectural lesson: clean separation between `StatsBase` (primitive moments/sampling) and `GLM` (modeling) maps directly to Reality's prob/optim split. Reality's `prob` covers the StatsBase territory but lacks regression — a future `glm` or `regression` package would compose `linalg` (QR) + `prob` (link functions) + `optim` (IRLS). Anti-pattern: formula-DSL strings parsed at runtime — Go users should pass design matrices directly.

## Aggregate themes Reality should track

- **Polyalgorithm + caching** (LinearSolve, DifferentialEquations): a unified `Solve(problem, opts)` that dispatches on problem structure and caches expensive setup. Reality should add this for `linalg`, `optim`, and `chaos` where multiple algorithms compete.
- **Problem-types vs free functions**: SciML uses `LinearProblem`/`ODEProblem`/`OptimizationProblem` as the contract. Reality's free-function style is fine for primitives; once we have polyalgorithms, problem-types become inevitable. Adopt for `optim` first.
- **Composable interfaces ≫ monoliths**: SciML's 200 packages share a small set of interfaces (SciMLBase, MOI, ManifoldsBase). Reality's 22 packages should similarly share lightweight interfaces (e.g., a `Function1D` interface used by `interp`, `approxfn`, `calculus`, `optim`).
- **Backend-agnostic frontends harm determinism**: DifferentiationInterface.jl is great for Julia's ergonomics but means user code can produce different results based on installed AD package. Reality's "one canonical implementation per function" rule is correct.
- **Rigorous numerics for golden generation**: IntervalArithmetic-style directed-rounding bounds during golden-file generation would replace ad-hoc epsilon choices with provable enclosures.
- **Symbolic preprocessing pre-baked into code**: We can't ship a CAS, but we can hard-code the products of one (closed-form derivatives, symbolic-simplified Jacobians, sparse patterns) in source.
- **MIT licensing dominates**: SciML is uniformly MIT; JuMP is MPL-2.0 (file-level copyleft, still safe). Reality's MIT choice aligns. Avoid copying any MPL code into Reality.

## Anti-patterns Reality already avoids

- **TTFX / compile-on-import**: 2025 "This Month in Julia" still tracks TTFX as a top concern; PrecompileTools/SnoopCompile mitigate but don't fix. Go's AOT compilation makes this a non-issue.
- **Source-transform AD over arbitrary user code** (Zygote): brittle under refactors, fails on mutation, fragments the ecosystem. Reality has no built-in AD.
- **Multiple-dispatch ambiguity hazards**: SciML constantly fights MethodAmbiguity errors when packages overload each other's types. Go's explicit-method-set interfaces are deterministic.
- **Dynamic typing in numerical kernels**: SciML papers over this with `@inbounds @fastmath` and aggressive specialization. Reality's typed `float64`/`complex128` is golden-file friendly out of the box.
- **Macro-heavy DSLs** (ModelingToolkit `@variables`, `@parameters`, JuMP `@constraint`): powerful but obfuscate the call graph and hurt grep-ability. Reality's plain Go is auditable.
- **Code generation at runtime** (ModelingToolkit emits `RuntimeGeneratedFunctions`): security and reproducibility hazard; build-time codegen (`go generate`) is the right answer if we ever need it.
- **Closure-returning constructors as the only API** (ApproxFun, DataInterpolations): make struct fields visible for golden-file inspection.
- **Wrapping foreign libraries** (LinearSolve wraps MKL, Pardiso, OpenBLAS): Reality's reimplement-from-first-principles rule sidesteps the licensing/build complexity.

## Sources
- [State of the SciML Open Source Software Ecosystem, 2025](https://sciml.ai/news/2025/06/26/state_of_sciml/)
- [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl)
- [JuliaCon 2025 DifferentialEquations workshop](https://sciml.github.io/2025-JuliaCon-DifferentialEquations-Workshop/)
- [ModelingToolkit.jl NEWS](https://github.com/SciML/ModelingToolkit.jl/blob/master/NEWS.md)
- [JuMP.jl roadmap](https://jump.dev/JuMP.jl/stable/developers/roadmap/)
- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
- [Flux.jl](https://github.com/FluxML/Flux.jl)
- [Lux.jl autodiff guide](https://lux.csail.mit.edu/stable/manual/autodiff)
- [DifferentiationInterface.jl](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/)
- [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl)
- [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl)
- [DataInterpolations.jl](https://github.com/SciML/DataInterpolations.jl)
- [ApproxFun.jl](https://github.com/JuliaApproximation/ApproxFun.jl)
- [Manopt.jl](https://github.com/JuliaManifolds/Manopt.jl)
- [IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl)
- [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl)
- [GLM.jl](https://github.com/JuliaStats/GLM.jl)
- [Bergmann arXiv:2507.16055 — Intrinsic Riemannian Proximal Gradient (convex)](https://arxiv.org/abs/2507.16055)
- [Bergmann arXiv:2506.09775 — Intrinsic Riemannian Proximal Gradient (nonconvex)](https://arxiv.org/abs/2506.09775)
- [PrecompileTools.jl (TTFX)](https://github.com/JuliaLang/PrecompileTools.jl)
- [This Month in Julia World — November 2025](https://julialang.org/blog/2025/12/this-month-in-julia-world/index.html)

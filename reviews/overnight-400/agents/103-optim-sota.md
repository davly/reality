# 103 | optim-sota | SOTA optim library comparison

**Topic:** Compare reality/optim with NLopt, scipy.optimize, Optuna, Ax/BoTorch, Optimization.jl, IPOPT, Pyomo, JuMP.jl, plus 2024-2026 NeurIPS frontier.
**Boundary:** This is the **state-of-the-art / engineering-tricks** axis. 101 (numerics) covered floating-point hygiene; 102 (missing) enumerated absent algorithms by name. This report scores reality/optim against external SOTA on **interface design, engineering tricks, and zero-dep portability** — the three axes the topic prompt names — and ranks frontier ideas reality could ship without breaking the zero-dep rule.

---

## 0. reality/optim baseline (what we are comparing against)

| File | Algorithms shipped |
|---|---|
| `gradient.go` | `GradientDescent`, `LBFGS` (m-history limited-memory BFGS, Wolfe line search) |
| `gradient_validated.go` | Validated variants of GD/LBFGS with predicate hook (R-VALIDATED-CONVERGENCE pattern) |
| `genetic.go` | `GeneticAlgorithm` (real-coded, tournament + crossover + mutation) |
| `metaheuristic.go` | `SimulatedAnnealing` |
| `linear.go` | `Simplex` (LP, primal, no scaling/presolve) |
| `rootfind.go` | `Bisection`, `NewtonRaphson`, `LinearInterpolateRoot`, `GoldenSection` |
| `interpolate.go` | `CubicSpline`, `LinearInterpolate` |
| `proximal/` | (sub-package — proximal operators, see proximal_consumer_test.go) |
| `transport/` | (sub-package — Sinkhorn / OT, see transport import) |

**~10 top-level optimizers** in one flat package, function-style, no `Problem` type, no `Result` type, no unified `Options{}`. Compare to the libraries below.

---

## 1. NLopt — the unified-C-interface gold standard

### Headline interface
Single `nlopt_opt` opaque object created via `nlopt_create(algorithm, n)`. Set objective, bounds, (in)equality constraints, stopping criteria, then `nlopt_optimize(opt, x, &minf)`. Same call pattern for **L-BFGS, SLSQP, MMA, COBYLA, BOBYQA, NEWUOA, PRAXIS, Subplex, Nelder-Mead, DIRECT, DIRECT-L, CRS2, StoGO, ISRES, ESCH** (~20 algorithms). User swaps `NLOPT_LD_LBFGS` ↔ `NLOPT_LN_BOBYQA` to switch from gradient to derivative-free without touching surrounding code.

### Engineering trick
**Algorithm naming convention encodes the calling contract.** `LD` = local-derivative, `LN` = local-no-derivative, `GD` = global-derivative, `GN` = global-no-derivative, `AUGLAG_*` = constraint-handling wrapper. The compiler (and the user) can statically check that you don't pass a no-gradient objective to an `LD_*` algorithm. Bindings in 11 languages all share this taxonomy.

### Zero-dep portability for reality
**HIGH.** NLopt itself is zero-dependency C. The naming taxonomy is free to adopt. Concrete imports: rename `LBFGS` → `LBFGS_LD`, add `BOBYQA_LN` and `Subplex_LN` (both pure code, ~300 LOC each, no third-party). The `nlopt_opt` opaque-object pattern maps directly to a Go `*Problem` struct with chainable setters. **Do this.**

---

## 2. scipy.optimize — the unified-Python-API gold standard

### Headline interface
`scipy.optimize.minimize(fun, x0, method=..., jac=..., hess=..., bounds=..., constraints=..., tol=..., options=...)` — one entry point dispatching to **14 methods** (Nelder-Mead, Powell, CG, BFGS, Newton-CG, L-BFGS-B, TNC, COBYLA, COBYQA, SLSQP, trust-constr, dogleg, trust-ncg, trust-krylov, trust-exact). Returns `OptimizeResult` with `x, fun, jac, hess_inv, nfev, njev, nhev, nit, status, message, success`. Sister functions: `minimize_scalar`, `root`, `root_scalar`, `linprog` (HiGHS backend since 1.9), `milp` (HiGHS, deterministic global MILP), `differential_evolution`, `dual_annealing`, `shgo`, `basinhopping`, `direct`, `brute`.

### Engineering trick
**`OptimizeResult` is the universal return contract.** Every optimizer fills the same fields; downstream code (matplotlib trajectory plot, convergence test, restart logic) is written once and works against all 14 methods. Plus: **`linprog`/`milp` quietly switched their backend from a hand-rolled simplex to HiGHS without breaking a single user** because the `OptimizeResult` shape is the only public surface. (The old `method='interior-point'` was deprecated in 1.9.0 — `method='highs'` replaces it.)

### Zero-dep portability for reality
**HIGH on the API design, LOW on HiGHS.** HiGHS is 200k+ lines of C++ — out of scope. But the `OptimizeResult` pattern is ~30 LOC and reality's optim package returns a bare `[]float64` from every optimizer today, throwing away `nfev/nit/converged/grad_norm` that all six functions already compute internally. **Add `optim.Result{X, F, Iter, FevalCount, GradNormFinal, Converged, Message}` and have every optimizer fill it.** Backward-compat: keep current signatures, add `*WithResult` variants. Also adopt scipy's `tol` + `options map` split — reality currently scatters `lr`/`maxIter`/`tol` as positional args, breaking the consistent-call-shape promise.

---

## 3. Optuna — Bayesian/TPE for hyperparameter optimization

### Headline interface
`study = optuna.create_study(sampler=TPESampler(), direction="minimize")`; `study.optimize(objective, n_trials=100)`. Objective receives a `trial` object with `trial.suggest_float("lr", 1e-5, 1e-1, log=True)` etc. — **define-by-run search space** (the search space is constructed dynamically as the objective executes, so conditional/hierarchical params just work). Samplers: `TPESampler` (default, Tree-structured Parzen Estimator), `CmaEsSampler`, `GPSampler`, `NSGAIISampler` (multi-obj), `NSGAIIISampler`, `MOTPESampler`, `RandomSampler`, `GridSampler`, `BruteForceSampler`. **TPE became 4× faster in v4.0.0** (one of the headline 2024-2025 wins). Prunes via `MedianPruner`/`HyperbandPruner`/`SuccessiveHalvingPruner`.

### Engineering trick
**Define-by-run + ask/tell ergonomics.** The objective is opaque Python — the search space is whatever `trial.suggest_*` calls it makes. Lets users encode `if model_type == "rf": trial.suggest_int("n_estimators", ...)` natively. The dual `ask()`/`tell()` API exposes the same engine for distributed/async use. Plus: **default sampler auto-switches to NSGA-II when direction is multi-objective** — no flag, no error, just works.

### Zero-dep portability for reality
**MEDIUM.** TPE itself (Bergstra-Bengio NeurIPS 2011) is ~400 LOC of pure math (KDE-based density ratio l(x)/g(x), Parzen window estimator). Zero-dep-friendly. The define-by-run trick requires Go closures + a `Trial` struct that records `SuggestFloat(name, lo, hi, log=true) float64` — also pure-code. Concrete recommendation: **add `optim/bayesian` sub-package with `TPE` first** (no GP needed, escapes the kernel-matrix Cholesky cost). Skip Hyperband for v1 (it requires a step-budget API redesign).

---

## 4. Ax / BoTorch — Gaussian-process Bayesian optimization

### Headline interface
**BoTorch:** `model = SingleTaskGP(train_X, train_Y); fit_gpytorch_mll(...); acq = qLogExpectedImprovement(model, best_f); candidate = optimize_acqf(acq, bounds, q=batch_size, num_restarts=10, raw_samples=512)`. Pure PyTorch tensors throughout — every operation is autograd-differentiable. **Ax** wraps BoTorch in `AxClient.create_experiment(parameters=[...])` + `AxClient.get_next_trial()` / `complete_trial()` — adaptive experimentation platform with multi-task, multi-fidelity, constraint-handling, and a web UI.

### Engineering trick
**Monte Carlo acquisition functions via reparameterized samples + autodiff.** Older BO libraries computed analytic EI = (μ-best)·Φ(z) + σ·φ(z) and were stuck with EI; BoTorch samples `q` posterior draws, computes `max(samples - best_f, 0).mean()`, and lets autograd give the acquisition gradient — generalises trivially to qEI, qNEI, qKG, qPI, qLogEI, qNoisyEI, etc. The 2023 trick: **`qLogExpectedImprovement` (arXiv:2310.20708) is now recommended over `qEI`** — same API, fixes the numerical underflow that caused EI to silently return 0 for highly-explored regions.

### Zero-dep portability for reality
**LOW for BoTorch-as-shipped (PyTorch dep), HIGH for the math.** A zero-dep GP-EI requires: (1) GP regression — already in reality? (Check `linalg/PCA` and `prob/` for kernel + Cholesky — yes, Cholesky is in `linalg`). (2) Acquisition function — analytic EI is 5 LOC. (3) Acquisition optimizer — L-BFGS with multi-restart, **already in reality/optim**. So a credible zero-dep `optim/bayesian.GPEI` is ~250 LOC composing existing pieces. The qLogEI numerical trick (log-domain expected improvement via `log1p(exp(-z²/2))`-style reformulation) is a 30-LOC win on top. **Do this after TPE.** Reality cannot match BoTorch's autodiff-through-acquisition until autodiff (see 011-015) ships vector nodes — currently scalar-only.

---

## 5. Optimization.jl (SciML) — the unified-Julia-interface

### Headline interface
`OptimizationProblem(f, x0, p; lb, ub, lcons, ucons)` + `solve(prob, OptimizationOptimJL.LBFGS())`. The `OptimizationProblem` is solver-agnostic; the solver tag selects backend (Optim.jl, NLopt.jl, Ipopt.jl, BlackBoxOptim.jl, Evolutionary.jl, MOI, Metaheuristics.jl, Manopt.jl — **~25+ backends behind one Problem type**). AD backend is **also** swappable: `Optimization.AutoForwardDiff()`, `AutoZygote()`, `AutoEnzyme()`, `AutoFiniteDiff()` — Optimization.jl computes the Jacobian/Hessian via your chosen AD library and passes to whichever solver needs it.

### Engineering trick
**Three-axis orthogonality: `Problem` × `Solver` × `AD-backend`.** Where scipy hardcodes the algorithm-method coupling and NLopt hardcodes the algorithm-name encoding, Optimization.jl makes all three independent. Sister `NonlinearSolve.jl` extends this — `NonlinearProblem` + auto algorithm selection based on Jacobian sparsity + Krylov support all behind one `solve()`. The 2025 maturity story is **PolyAlgorithm**: Optimization.jl runs cheap solvers first and falls back to expensive ones based on residual.

### Zero-dep portability for reality
**HIGH on the orthogonality lesson, MEDIUM on implementation.** Reality has exactly the wrong shape today: `optim.LBFGS(f, grad, x0, m, maxIter, tol)` welds objective + gradient + hyperparameters + algorithm into one signature. The Optimization.jl lesson is: **decouple `Problem`, `Solver`, and `Result`.** Concrete recommendation:

```go
type Problem struct {
    F       func([]float64) float64
    Grad    func([]float64, []float64)        // optional; nil → finite-diff
    Hess    func([]float64, [][]float64)      // optional; nil → BFGS approx
    Bounds  [][2]float64                      // optional
    EqCons  []func([]float64) float64         // optional
    InCons  []func([]float64) float64         // optional
    X0      []float64
}
type Solver interface { Solve(*Problem, Options) Result }
```

This is ~80 LOC of API surface and unblocks every future addition (BOBYQA, SLSQP, COBYLA, trust-region, IPOPT-port, MMA, augmented-Lagrangian wrappers).

---

## 6. IPOPT — the large-scale interior-point reference

### Headline interface
`Ipopt::IpoptApplication app; app->Initialize(); app->OptimizeTNLP(tnlp);` where `TNLP` (Templated Nonlinear Program) is a C++ interface the user implements: `get_nlp_info`, `get_bounds_info`, `get_starting_point`, `eval_f`, `eval_grad_f`, `eval_g`, `eval_jac_g`, `eval_h`. Solves min f(x) s.t. g_L ≤ g(x) ≤ g_U, x_L ≤ x ≤ x_U with **millions of variables/constraints**. Wächter & Biegler 2006 paper is the canonical citation.

### Engineering trick
**Primal-dual interior-point with filter-based line search.** Two key tricks: (1) the **filter** replaces a merit function — instead of one penalty parameter trading off feasibility vs optimality, the filter is a Pareto-set of (constraint-violation, objective) pairs and a step is acceptable if it dominates an existing filter point. Avoids Maratos-effect step rejection. (2) **Mehrotra predictor-corrector** for the KKT system — solve once with current Jacobian/Hessian, predict the affine step, correct with a centring step. Both are pure linear algebra over the augmented KKT matrix.

### Zero-dep portability for reality
**MEDIUM-LOW.** The math is in scope (filter line search is ~150 LOC; Mehrotra predictor-corrector is ~80 LOC) but IPOPT depends on a sparse linear solver (MA27/MA57/Pardiso/MUMPS) for the KKT system. Reality's `linalg` has dense Cholesky/LU but **no sparse direct solver** — out of scope at the implementation level. Realistic ask: **port a dense interior-point QP solver** (Mehrotra-IPM for QP, ~500 LOC) as `optim.QuadProg(Q, c, A, b)`. This is the building block scipy uses for SLSQP and that BoTorch uses internally for batched acquisition optimization. Frontier: a small-scale dense `optim.NLP_IPM` for n ≤ 1000 (Wächter-Biegler filter on dense KKT, ~800 LOC).

---

## 7. Pyomo / JuMP.jl — algebraic modeling DSLs

### Headline interface
**Pyomo:** `m = ConcreteModel(); m.x = Var(range(n), domain=NonNegativeReals); m.obj = Objective(expr=sum(c[i]*m.x[i] for i in range(n))); m.con = Constraint(expr=sum(a[i]*m.x[i] for i in range(n)) <= b); SolverFactory('ipopt').solve(m)`. **JuMP.jl:** `model = Model(Ipopt.Optimizer); @variable(model, x[1:n] >= 0); @objective(model, Min, sum(c[i]*x[i] for i in 1:n)); @constraint(model, sum(a[i]*x[i] for i in 1:n) <= b); optimize!(model)`. Both are **algebraic DSLs** — user writes math, library generates derivatives + solver-specific input format (NL file, MPS, LP).

### Engineering trick
**Symbolic expression trees → analytic derivatives + sparsity pattern, computed once at model-build time.** JuMP's `MOI` (MathOptInterface) layer abstracts solver communication into a single Julia interface; switching `Ipopt.Optimizer` ↔ `Gurobi.Optimizer` ↔ `HiGHS.Optimizer` is a one-token change. JuMP is **measurably faster than Pyomo on large models** because Pyomo re-uses AMPL's ASL for derivatives (file I/O round-trip) while JuMP computes derivatives in-process.

### Zero-dep portability for reality
**LOW for the DSL, MEDIUM for the AD-derivatives lesson.** Go has no macro system → no `@variable`/`@constraint` ergonomics achievable. **But the lesson is portable**: reality's autodiff package (see 011-015) already produces gradients from a `func([]float64) float64` — what's missing is the JuMP trick of doing it **once at model-build time** rather than once per `optim.LBFGS` call. With `optim.Problem` from §5 and `autodiff.Tape` reuse from §015, a Go user writes `prob.F = myObjective` and reality auto-generates `prob.Grad` from autodiff at first call, caches the tape, reuses across iterations. ~150 LOC bridge in `optim/autoderiv.go`. **Do this after Problem-type lands.**

---

## 8. NeurIPS 2024-2026 frontier — ML for optimization

The relevant 2024-2025 thread is **learned optimizers + meta-learning** (NeurIPS OPT 2025 workshop "Statistics Meets Optimization"). Three concrete frontier ideas reality can credibly ship:

### 8a. FOSI / Lion / Sophia (2nd-order in disguise)
**Lion** (Chen et al. NeurIPS 2023, EvoLved Sign Momentum, found by program search) is a 4-LOC optimizer that beats AdamW on language models with half the memory. **Sophia** (Liu et al. ICLR 2024) uses diagonal Hessian via Hutchinson's estimator + clipping. Both are pure-math, pure-code, ~50 LOC each. **Add to reality/optim.** Differentiates reality from "textbook L-BFGS only" libraries.

### 8b. Implicit-function-theorem-based hyperparameter gradients
**Lorraine-Vicol-Duvenaud ICML 2020 + DEQ-style fixed-point AD** lets you backprop through the entire optimization process to optimize hyperparameters. Composes with autodiff/015's recommended `FixedPoint` primitive. Reality is one of few zero-dep libraries that could ship this.

### 8c. Bayesian quadrature + GP-UCB hybrids
2024-2026 has moved past plain GP-EI to **vanilla GP-UCB with Wiener-process priors** (cheaper kernel inversion, log-domain UCB for numerical stability). One-night port if §4's GP-EI lands first.

### Skip
**Learned-optimizer meta-training** (Metz et al., Velo) requires PyTorch + GPU + a 100k-task training set — out of scope for a zero-dep math library forever.

---

## 9. Cross-library engineering tricks reality could steal

| Trick | Source | LOC | Where it goes in reality/optim |
|---|---|---|---|
| `Result{X, F, Iter, Nfev, GradNorm, Converged, Message}` return type | scipy | 30 | New `optim/result.go`, retrofit all 10 optimizers |
| `Problem{F, Grad, Hess, Bounds, Cons, X0}` decoupled from solver | Optimization.jl | 80 | New `optim/problem.go` |
| `Solver` interface (`type Solver interface{ Solve(*Problem, Options) Result }`) | Optimization.jl + JuMP | 40 | New `optim/solver.go` |
| LD/LN/GD/GN naming taxonomy in algorithm constants | NLopt | 50 | `optim/algorithms.go` enum |
| Define-by-run `Trial.SuggestFloat(name, lo, hi, log=true)` | Optuna | 200 | New `optim/bayesian/trial.go` |
| TPE sampler (KDE l/g density ratio) | Optuna / Bergstra-Bengio 2011 | 400 | New `optim/bayesian/tpe.go` |
| Analytic GP-EI with Cholesky reuse | scikit-optimize / GPyOpt | 250 | New `optim/bayesian/gpei.go` (uses linalg.Cholesky) |
| qLogExpectedImprovement numerical reformulation | BoTorch arXiv:2310.20708 | 30 | `optim/bayesian/qlogei.go` |
| Filter line search (Pareto-set step acceptance) | IPOPT / Wächter-Biegler 2006 | 150 | `optim/filter.go` (used by future SLSQP/IPM) |
| Mehrotra predictor-corrector for KKT system | IPOPT / Mehrotra 1992 | 80 | `optim/qp.go` (dense QP) |
| BFGS with damped update (Powell's modification) | Nocedal-Wright Algo 18.2 | 30 | `gradient.go` LBFGS hardening |
| Strong Wolfe with bracketing+zoom + cubic interpolation fallback | Nocedal-Wright Algo 3.5/3.6 | 120 | replace current `lbfgsLineSearch` |
| Trust-region dogleg fallback when line search fails | Nocedal-Wright §4.1 | 80 | `optim/trustregion.go` |
| CMA-ES (μ/μ_w, λ) with two evolution paths + step-size control | Hansen 2016 tutorial | 350 | `optim/cmaes.go` (replaces SA for hard non-convex) |
| Lion sign-momentum optimizer | Chen et al. NeurIPS 2023 | 50 | `optim/lion.go` |
| Sophia diagonal-Hessian with Hutchinson estimator + clipping | Liu et al. ICLR 2024 | 80 | `optim/sophia.go` |
| Subplex (Nelder-Mead on subspaces) | Rowan PhD 1990 / NLopt | 300 | `optim/subplex.go` (derivative-free, beats Nelder-Mead on >5D) |
| BOBYQA quadratic-model derivative-free | Powell 2009 / NLopt | 800 | `optim/bobyqa.go` (derivative-free state-of-art) |
| HiGHS-style presolve + scaling for `Simplex` | HiGHS / scipy 1.9+ | 400 | retrofit `linear.go` |

**Total ~3,500 LOC** to bring reality/optim from "textbook 10-function survey" to "credible zero-dep alternative for the 80% case scipy+NLopt+Optuna cover" — none of it requires a new dependency.

---

## 10. Three-axis SOTA scoring

| Axis | reality/optim today | NLopt | scipy | Optuna | BoTorch | Optim.jl | IPOPT | JuMP |
|---|---|---|---|---|---|---|---|---|
| Unified solver interface | NO (per-function sigs) | YES (opaque opt) | YES (`minimize`) | YES (`study.optimize`) | partial (BoTorch raw, Ax wraps) | YES (`solve`) | YES (TNLP) | YES (`@objective`) |
| Decoupled Problem/Solver | NO | partial | NO (welded to method=) | partial | NO | YES | NO | YES |
| Result-record return | NO (bare []float64) | partial | YES (OptimizeResult) | YES (Trial) | YES | YES | YES | YES |
| Algorithm taxonomy in names | NO | YES (LD/LN/GD/GN) | NO | NO | NO | partial | n/a | n/a |
| Define-by-run search space | NO | NO | NO | YES | NO | NO | NO | NO |
| GP / TPE / Bayesian | NO | NO | NO (separate skopt) | YES | YES | partial | NO | NO |
| Multi-objective (NSGA/MOTPE) | NO | YES (ISRES) | partial | YES | YES (qNEHVI) | YES | NO | NO |
| Constraint handling | NO (only Simplex) | YES (AUGLAG) | YES (SLSQP/COBYLA/trust-constr) | partial | partial | YES | YES | YES |
| Sparse / large-scale | NO | NO | NO | NO | NO | YES | YES | YES |
| AD-through-objective | NO (manual grad) | NO | NO (manual jac) | NO | YES (autograd) | YES (4 backends) | NO | YES (symbolic) |
| Zero-dep | YES | YES | NO (numpy/cython) | NO (numpy) | NO (torch) | NO (Julia stdlib+) | NO (LAPACK+MA27) | NO (Julia stdlib+) |
| Cross-language goldens | YES (testutil) | NO | NO | NO | NO | NO | NO | NO |

reality scores **2/12** today (zero-dep + cross-language goldens). The §9 fix-set lifts that to **9/12** (everything except sparse-large-scale, AD-through-objective requiring autodiff vector nodes first, and define-by-run requires the Problem+Trial work). The two NO that should stay NO: sparse-large-scale (out of scope without a sparse linear solver dep) and define-by-run (Go's lack of macros makes it ergonomically awkward — TPE without define-by-run is still 80% of the value).

---

## 11. Differentiation from 101 / 102

- **101 (numerics)** owns: floating-point hygiene, Wolfe-condition implementation bugs, NaN propagation in `lbfgsLineSearch`, golden-vector coverage, IEEE-754 edge cases.
- **102 (missing)** owns: enumerated lists of absent algorithms by name, Tier 1/2/3 grouping with LOC estimates, sprint ordering for catch-up.
- **103 (this report)** owns: cross-library *engineering* tricks (filter line search, qLogEI reformulation, define-by-run, Problem/Solver/AD orthogonality), the 12-axis SOTA scoring table, and the §9 fix-set ranked by *interface-design impact* rather than algorithm-coverage.

No overlap with the LOC-by-algorithm enumeration in 102 or the per-line numerical bugs in 101.

---

## 12. Top 5 highest-leverage commits (interface-design, not algorithm-count)

1. **`optim/result.go` + `Result` retrofit** (~30 LOC + 10 call-site changes) — every existing optimizer already computes `iter/nfev/grad_norm/converged` and throws them away. Adopting scipy's pattern unblocks every future plot/test/restart utility. Backward-compat: keep current returns, add `*WithResult` variants. **Do this first.**
2. **`optim/problem.go` + `optim/solver.go`** (~120 LOC, no algorithm changes) — Optimization.jl's three-axis orthogonality. Once `Problem` is decoupled, every new solver in §9 is a one-file PR.
3. **`optim/algorithms.go` taxonomy enum** (~50 LOC) — adopt NLopt's `LD/LN/GD/GN` prefix in algorithm constants. Lets the compiler check that derivative-required solvers receive a non-nil `Problem.Grad`.
4. **`optim/cmaes.go`** (~350 LOC) — CMA-ES is the single most-cited derivative-free non-convex optimizer of the last 25 years and reality currently offers only `SimulatedAnnealing` for that role. Hansen 2016 tutorial is the canonical reference; pure code, no deps.
5. **`optim/bayesian/tpe.go` + `optim/bayesian/trial.go`** (~600 LOC) — gives reality a credible zero-dep Bayesian-optimization story without needing GP/Cholesky. Ports the Optuna define-by-run idiom in a Go-idiomatic way (closures + Trial struct).

After these five, reality/optim is **pareto-distinct** in the optimization-library space: zero-dep (vs everyone but NLopt), cross-language goldens (vs everyone), modern interface (vs NLopt's C-style), Bayesian (vs scipy/NLopt), and matches scipy/Optimization.jl on the unified-interface axis.

---

## Sources

- [NLopt algorithms (NLopt docs)](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/)
- [NLopt introduction (LD/LN/GD/GN naming)](https://nlopt.readthedocs.io/en/latest/NLopt_Introduction/)
- [stevengj/nlopt GitHub](https://github.com/stevengj/nlopt)
- [scipy.optimize reference (SciPy v1.17)](https://docs.scipy.org/doc/scipy/reference/optimize.html)
- [scipy.optimize.milp (HiGHS-backed deterministic MILP)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
- [scipy.optimize.linprog (interior-point deprecation)](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-interior-point.html)
- [scipy.optimize.differential_evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
- [Optuna TPESampler (4.8 docs, 4.0 perf win)](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [Optuna NSGAIISampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html)
- [Optuna CmaEsSampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.CmaEsSampler.html)
- [Optuna Efficient Optimization Algorithms tutorial](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html)
- [BoTorch introduction (qLogEI > qEI recommendation)](https://botorch.org/docs/introduction/)
- [meta-pytorch/botorch GitHub](https://github.com/meta-pytorch/botorch)
- [Ax Bayesian Optimization docs](https://ax.dev/docs/0.5.0/bayesopt/)
- [BoTorch acquisition module](https://botorch.readthedocs.io/en/stable/acquisition.html)
- [SciML Optimization.jl GitHub](https://github.com/SciML/Optimization.jl)
- [SciML NonlinearSolve.jl GitHub](https://github.com/SciML/NonlinearSolve.jl)
- [NonlinearSolve.jl docs (auto algorithm selection)](https://docs.sciml.ai/NonlinearSolve/stable/)
- [coin-or/Ipopt GitHub](https://github.com/coin-or/Ipopt)
- [Ipopt docs (Wächter-Biegler filter line search)](https://coin-or.github.io/Ipopt/)
- [Wächter-Biegler 2006 IPOPT paper (PDF)](https://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf)
- [JuMP: A Modeling Language (SIAM Review)](https://epubs.siam.org/doi/10.1137/15M1020575)
- [JuMP vs Pyomo vs Gurobi vs GAMS performance comparison](https://www.gams.com/blog/2023/07/performance-in-optimization-models-a-comparative-analysis-of-gams-pyomo-gurobipy-and-jump/)
- [jump-dev/JuMP.jl GitHub](https://github.com/jump-dev/JuMP.jl)
- [NeurIPS OPT 2025 workshop](https://opt-ml.org/)
- [NeurIPS 2025 OPT workshop schedule](https://neurips.cc/virtual/2025/loc/san-diego/workshop/109581)
- [Hansen CMA-ES tutorial arXiv:1604.00772](https://arxiv.org/pdf/1604.00772)
- [pycma GitHub (Hansen-Akimoto-Baudis)](https://github.com/CMA-ES/pycma)
- [Wolfe conditions (Wikipedia, Nocedal-Wright values)](https://en.wikipedia.org/wiki/Wolfe_conditions)
- [BFGS (Wikipedia)](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)

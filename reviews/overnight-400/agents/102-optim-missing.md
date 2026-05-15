# 102 — optim: canonical algorithms missing

## Headline
`reality/optim` ships **9 distinct algorithms** (gradient descent, L-BFGS, simplex, a "barrier-gradient-heuristic" mis-labelled as Interior Point, bisection, Newton-Raphson, golden-section, GA, simulated annealing) plus three 1-D interpolators, and **two well-built sub-packages** (`proximal/` = FBS+FISTA+ADMM+8 prox ops; `transport/` = Sinkhorn + 1-D Wasserstein). Against scipy.optimize / NLopt / Optuna / Pyomo / IPOPT this leaves **~50 canonical primitives missing**, including every modern stochastic optimiser (SGD/Adam/AdamW/RMSprop/Adagrad/Lion), every trust-region method, every derivative-free direct-search (Nelder-Mead, Powell, BOBYQA, DIRECT), every Bayesian/surrogate optimiser (GP-EI, Hyperband, BOHB, TPE, SMAC), every multi-objective evolutionary algorithm (NSGA-II/III, MOEA/D, SPEA2), the entire SQP / augmented-Lagrangian / KKT family, and a *correct* primal-dual interior-point LP solver (101's F7 — current `InteriorPoint` is gradient-on-barrier, not Newton-on-KKT). Existing sub-packages establish the precedent that `optim/` is allowed to grow sub-packages; recommend `optim/sgd`, `optim/trust`, `optim/direct`, `optim/bayes`, `optim/multiobj`, `optim/qp`.

## What exists today (post-101 audit)

| File | Algorithm | Status from 101 |
|------|-----------|-----------------|
| `gradient.go::GradientDescent` | Vanilla GD with Armijo backtracking | OK |
| `gradient.go::LBFGS` | L-BFGS, Armijo line-search (mis-labelled Wolfe) | F1/F2 numerics |
| `gradient_validated.go::*Validated` | Same + R123 input check | F20 transitive |
| `linear.go::SimplexMethod` | Full-tableau LP simplex (called "revised") | F4/F5 latent panic + drift |
| `linear.go::InteriorPoint` | **Heuristic gradient-on-barrier, not IPM** | **F7 wrong algorithm** |
| `rootfind.go` | Bisection, Newton-Raphson, golden-section, single-secant-step | F8/F9/F10/F11 gaps |
| `genetic.go::GeneticAlgorithm` | Real-coded GA, BLX-α, Gaussian mutation | F13 hardcoded `[-5,5]` |
| `metaheuristic.go::SimulatedAnnealing` | Geometric cooling, Metropolis | F15 |
| `interpolate.go` | LinearInterpolate, CubicSplineNatural | F16/F17 |
| `proximal/` | FBS, FISTA, ADMM-consensus, 8 prox ops | full |
| `transport/` | Sinkhorn, pairwise W₁, IQR-norm | full |

So the topic prompt's "[present, partial]" annotations: **L-BFGS** present (with F1/F2 issues); **simplex** present (with F4/F5 issues); **GA** present (with F13); **IP** annotated "heuristic" — confirmed by 101 F7 to *not* be a primal-dual IPM; **proximal methods** present (proximal sub-package); **ADMM** present (proximal sub-package).

The topic prompt itself names ~36 algorithms; 8 are present (counting proximal/ADMM/SA/GA/L-BFGS/IP-as-heuristic/simplex/projected-gradient-via-prox), so **~28 of 36 are missing** and the broader scipy/NLopt crosswalk adds another ~22. Numbered tiers below.

---

## Tier 1 — must-ship (frequency-of-need ≥ 0.5 across reality consumers + numerical-methods canon)

### T1.1 — Conjugate Gradient (Polak-Ribière+ with restart) — ~120 LOC
Nocedal-Wright Alg. 5.4. The single most-cited unconstrained smooth optimiser when a Hessian isn't formed. Useful for Newton steps inside other algorithms (Steihaug-CG below). FR + PR+ + Hestenes-Stiefel variants share one loop.

### T1.2 — Levenberg-Marquardt (Marquardt 1963) — ~150 LOC
The workhorse for nonlinear least-squares (Heston/SABR calibration, GARCH MLE, sensor-fusion). `optim/` already ships L-BFGS for general unconstrained but no LSQ-specialised solver. Implement with TR-radius update (Moré 1978 LMDIF).

### T1.3 — Trust-region Newton with Steihaug-CG sub-problem — ~200 LOC
Conn-Gould-Toint *Trust-Region Methods*. Replaces the F1/F2 line-search L-BFGS pathology with a globally-convergent framework. Sub-problem solved by Steihaug-Toint truncated CG (no factorisation). Pairs with T1.1.

### T1.4 — Brent's 1-D minimiser (`brent`) — ~80 LOC
Numerical Recipes §10.3. Companion to existing `GoldenSectionSearch` — golden-section + parabolic interpolation, super-linear convergence on smooth unimodal functions. Mandatory for line-search inside T1.1/T1.3.

### T1.5 — Brent's 1-D root finder (`zbrent`) — ~80 LOC
Numerical Recipes §9.3. Companion to existing bisection+Newton — bisection-bracketed inverse-quadratic interpolation. The default 1-D root-finder in scipy (`scipy.optimize.brentq`) and Boost (`brent_find_minima`).

### T1.6 — Adam + AdamW (Kingma-Ba 2014, Loshchilov-Hutter 2019) — ~80 LOC
The *de facto* default optimiser for any iterative parameter fit since 2015. AdamW (decoupled weight decay) is the 2019 correction now standard. Required for any future autodiff-driven calibration consumer (autodiff slot 11-15 review explicitly flags GARCH/Heston/SABR consumers).

### T1.7 — RMSprop + Adagrad (Hinton 2012, Duchi-Hazan-Singer 2011) — ~50 LOC
Two more textbook adaptive-learning-rate optimisers. Trivial alongside Adam.

### T1.8 — SGD with Nesterov momentum — ~40 LOC
Sutskever-Martens-Dahl-Hinton 2013 reformulation. The textbook stochastic optimiser; baseline for Adam comparisons.

### T1.9 — Nelder-Mead simplex (downhill-simplex) — ~150 LOC
Nelder-Mead 1965. **The** derivative-free optimiser; scipy default for `minimize(method=None)` when no Jacobian. Reflection / expansion / contraction / shrink with Gao-Han 2012 adaptive parameters for high-dim. Pure math, no dependencies, ~30 vector ops.

### T1.10 — Differential evolution (Storn-Price 1997) — ~120 LOC
Sibling of GA but generally outperforms it on continuous-domain problems and is scipy's `differential_evolution` workhorse. Strategies: rand/1/bin, best/1/bin, currenttobest/1/bin. Composes naturally with the existing GA infrastructure.

### T1.11 — Particle swarm optimisation (Kennedy-Eberhart 1995) — ~80 LOC
Classical metaheuristic alongside SA/GA/DE; the four together are the canonical stochastic-global quartet.

### T1.12 — Levenberg-Marquardt's friend: Gauss-Newton (no damping) — ~60 LOC
Already half-built inside T1.2; expose as separate primitive for users who know their problem is well-conditioned.

### T1.13 — Replace `linear.InteriorPoint` with Mehrotra Predictor-Corrector (MPC) — ~250 LOC
**Critical** per 101 F7 — the current routine is gradient-on-barrier, not Newton-on-KKT. Mehrotra 1992 SIAM J Optim 2(4) is the standard primal-dual IPM (used by IPOPT, MOSEK, CPLEX). Reuses `linalg/cholesky` for the normal-equations form `AΘAᵀΔλ = r`. Either rename current as `BarrierGradientHeuristic` (transparent) or replace.

### T1.14 — Quadratic Programming (QP) solver (active-set or OSQP-style ADMM) — ~300 LOC
Currently unable to solve general QP `min ½xᵀQx + cᵀx s.t. Ax≤b, Cx=d`. ADMM-based OSQP (Stellato-Banjac-Goulart-Bemporad-Boyd 2020) reuses existing `proximal/` infrastructure and is ~300 LOC. Active-set (Goldfarb-Idnani 1983) is an alternative for small problems.

**Tier 1 total: ~1,760 LOC, 14 algorithms.** This brings `optim/` to scipy-equivalence on the unconstrained-smooth + LP/QP + classical-stochastic-global axes.

---

## Tier 2 — strongly recommended (commonly expected; fills a tier-of-the-canon)

### Gradient-based (continuation)
- **T2.1 — Damped/safeguarded Newton with Hessian factorisation** (Nocedal-Wright Alg. 3.2). Reuses `linalg/cholesky`+modified-Cholesky.
- **T2.2 — BFGS (full Hessian)** alongside existing L-BFGS. Useful when memory is not a constraint and updates can be reused.
- **T2.3 — DFP (Davidon-Fletcher-Powell)** for completeness; the historical predecessor to BFGS.
- **T2.4 — Trust-region dogleg** as alternative to Steihaug-CG (Powell 1970, used by scipy `method='dogleg'`).

### Stochastic / batch
- **T2.5 — Lion (Chen et al. 2023, EvoLved Sign Momentum)** — the 2023 Google Brain optimiser; sign-of-momentum variant of Adam with smaller memory.
- **T2.6 — Sophia (Liu-Pang-Mai-Smith 2024)** — second-order clipped pre-conditioner; arXiv 2305.14342.
- **T2.7 — Adadelta (Zeiler 2012)** — Adagrad without the learning-rate accumulation.
- **T2.8 — AMSGrad (Reddi-Kale-Kumar 2018)** — convergence fix for Adam non-convergence cases.
- **T2.9 — Nadam (Dozat 2016)** — Nesterov-accelerated Adam.

### Constrained
- **T2.10 — Sequential Quadratic Programming (SQP, Wilson 1963 / Han 1976 / Powell 1978)** — solves a QP at each iterate. Industrial standard for nonlinear-constrained (NLP) optimisation. Reuses T1.14 QP solver.
- **T2.11 — Augmented Lagrangian (Hestenes 1969 / Powell 1969)** — penalty-with-multipliers; more robust than penalty alone, simpler than SQP. 200 LOC.
- **T2.12 — Frank-Wolfe / conditional gradient (Frank-Wolfe 1956, Jaggi 2013)** — projection-free LP-oracle method for constrained convex; revival in ML for affine-constrained problems.
- **T2.13 — Projected gradient descent** — explicit (separate from prox-projection in proximal/). Wraps any closed-form Euclidean projection; trivial composition.

### Derivative-free (continuation)
- **T2.14 — Powell's COBYLA (Constrained Optimisation BY Linear Approximation, Powell 1994)** — standard derivative-free solver with linear inequality constraints; NLopt staple.
- **T2.15 — BOBYQA (Bounded-OBYQA, Powell 2009)** — quadratic-model trust-region for bound-constrained derivative-free.
- **T2.16 — Pattern search / Hooke-Jeeves (1961)** — classical direct-search; pre-cursor to MADS.
- **T2.17 — DIRECT (Jones-Perttunen-Stuckman 1993)** — Lipschitz-free deterministic global search; NLopt staple, scipy `direct`.

### Global / metaheuristic
- **T2.18 — CMA-ES (Hansen-Ostermeier 2001)** — Covariance-Matrix-Adaptation Evolution Strategy. Considered SOTA derivative-free black-box for ≤100-dim problems (NEvergrad, Ax, Optuna all wrap it). Existing GA review (F13) flagged this as a top miss. ~400 LOC including Cholesky factor of C.
- **T2.19 — Tabu search (Glover 1986)** — combinatorial-optimisation classic.
- **T2.20 — Adaptive Large Neighbourhood Search (ALNS, Pisinger-Røpke 2007)** — destroy-and-repair metaheuristic; routing/scheduling staple.

### Bayesian / surrogate
- **T2.21 — Gaussian Process Bayesian Optimisation (Močkus 1975, Snoek-Larochelle-Adams 2012)** — Expected Improvement, Upper Confidence Bound, Probability of Improvement acquisition functions. Requires GP regression (not yet in `prob/` — would need to build minimal GP for ARD-Matérn kernel). Optuna + Ax + BoTorch + GPyOpt are the SOTA references.
- **T2.22 — Tree-structured Parzen Estimator (TPE, Bergstra et al. NIPS 2011)** — Optuna's default; KDE-based surrogate; works on mixed continuous/categorical. ~150 LOC, no GP needed.
- **T2.23 — Hyperband (Li-Jamieson-DeSalvo-Rostamizadeh-Talwalkar JMLR 2018)** — successive-halving bandit for hyperparameter tuning; pure scheduling logic (~80 LOC).
- **T2.24 — BOHB (Falkner-Klein-Hutter ICML 2018)** — Hyperband + TPE; depends on T2.22+T2.23.
- **T2.25 — SMAC (Hutter-Hoos-Leyton-Brown 2011)** — Random-Forest surrogate; needs random-forest regressor (would compose with prob/ if added).

### Multi-objective
- **T2.26 — NSGA-II (Deb-Pratap-Agarwal-Meyarivan 2002)** — fast non-dominated sort + crowding-distance; the most-cited multi-objective EA (15k+ citations). ~250 LOC.
- **T2.27 — NSGA-III (Deb-Jain 2014)** — reference-point-based selection for many-objective (≥4 objectives).
- **T2.28 — MOEA/D (Zhang-Li 2007)** — decomposition-based; competitive on continuous problems.
- **T2.29 — SPEA2 (Zitzler-Laumanns-Thiele 2001)** — strength-Pareto archive-based.
- **T2.30 — Hypervolume indicator (Zitzler-Thiele 1998, WFG algorithm While-Hingston-Barone 2012)** — quality metric for Pareto fronts; required for any MO benchmarking.

### Stochastic approximation
- **T2.31 — SPSA (Spall 1992 IEEE TAC)** — Simultaneous Perturbation Stochastic Approximation; gradient estimate from 2 function evals regardless of dim. Black-box optimisation when gradients are noisy/expensive.
- **T2.32 — Robbins-Monro (1951)** — the historical root algorithm of stochastic approximation; useful as a documented building block.
- **T2.33 — Kiefer-Wolfowitz (1952)** — finite-difference variant of Robbins-Monro.

### Convex optimisation
- **T2.34 — Primal-dual interior-point for QP** (alternative to T1.14 ADMM-based). MPC-style, reuses Cholesky.
- **T2.35 — Active-set QP (Goldfarb-Idnani 1983)** — small-problem alternative to T1.14.

### 1-D root finding
- **T2.36 — Halley's method** — third-order; needs `f"`.
- **T2.37 — Secant method (full iteration, not single-step)** — derivative-free, super-linear.
- **T2.38 — Ridders' method (1979)** — bracketed quadratic-interpolation; alternative to Brent's `zbrent`.
- **T2.39 — Inverse quadratic interpolation (Müller's method extension)** — used inside Brent.
- **T2.40 — Damped Newton with bisection fallback (`rtsafe`, NR §9.4)** — 101 F10's named gap.

**Tier 2 total: ~3,000 LOC, 40 algorithms.** Brings `optim/` to NLopt + Optuna + scipy.optimize feature-parity on the textbook surface.

---

## Tier 3 — research-frontier / specialist

### Convex optimisation extensions
- **T3.1 — Second-Order Cone Programming (SOCP)** — primal-dual IPM via Nesterov-Todd scaling. Foundational for portfolio optimisation, robust LS, antenna design. ~600 LOC.
- **T3.2 — Semidefinite Programming (SDP)** — primal-dual IPM, AHO direction. Relaxations for combinatorial problems (MAXCUT, sensor localisation). ~800 LOC. Requires eigendecomposition of `n×n` symmetric in inner loop.
- **T3.3 — Disciplined Convex Programming compiler (à la CVXPY 2014)** — reduce user-supplied convex expression to standard SDP/SOCP/QP form. Out of scope as compiler, but the cone-program standard form would be a useful intermediate.
- **T3.4 — Chambolle-Pock primal-dual (2011 J Math Imaging Vis)** — first-order saddle-point method; complement to existing FBS/ADMM. Mentioned as "deferred to v2" in `proximal/doc.go`.
- **T3.5 — Davis-Yin three-operator splitting (Davis-Yin 2017 SVAA)** — also "deferred to v2" in `proximal/doc.go`.

### Online optimisation
- **T3.6 — Mirror descent (Nemirovski-Yudin 1983, Beck-Teboulle 2003)** — entropic-regularised analogue of SGD; used in bandit / online-learning theory.
- **T3.7 — Online Newton Step (Hazan-Agarwal-Kale 2007)** — second-order online algorithm with logarithmic regret.
- **T3.8 — Follow-the-Regularised-Leader (Shalev-Shwartz 2007)** — generic online-convex-optimisation framework.
- **T3.9 — AdaGrad-online / Online Mirror Descent variants** — online-learning canon.

### Multilevel / global continuation
- **T3.10 — Multilevel Coordinate Search (MCS, Huyer-Neumaier 1999)** — DIRECT-style with local quadratic refinement; NLopt offers it.
- **T3.11 — GRASP (Greedy Randomised Adaptive Search Procedure)** — combinatorial metaheuristic.
- **T3.12 — Variable Neighbourhood Search (Hansen-Mladenović 1997)** — combinatorial.
- **T3.13 — Iterated Local Search (Lourenço-Martin-Stützle 2003)** — combinatorial.

### Bayesian extensions
- **T3.14 — Multi-fidelity BO (Kandasamy et al. 2017)** — for expensive simulators.
- **T3.15 — Constrained BO (Gardner et al. ICML 2014)** — with feasibility-aware acquisition.
- **T3.16 — Batch BO (q-EI, Wang et al. NeurIPS 2016)** — for parallel evaluation.
- **T3.17 — Trust-Region BO (TuRBO, Eriksson et al. NeurIPS 2019)** — high-dim BO.

### Stochastic / variance-reduced
- **T3.18 — SVRG (Johnson-Zhang NIPS 2013)** — Stochastic Variance-Reduced Gradient.
- **T3.19 — SAGA (Defazio-Bach-Lacoste-Julien NIPS 2014)** — bias-corrected SVRG.
- **T3.20 — Katyusha (Allen-Zhu STOC 2017)** — accelerated variance-reduced.
- **T3.21 — Mini-batch SGD with Polyak averaging (Polyak-Juditsky 1992)** — convergence accelerator.

### IPOPT-class nonlinear programming
- **T3.22 — IPOPT-style filter line-search interior point (Wächter-Biegler 2006 Math Prog)** — the open-source NLP standard. Genuinely large (~3,000 LOC port) but the math is documented.
- **T3.23 — Knitro-style trust-region SQP** — alternative NLP framework.

### Black-box / surrogate (continuation)
- **T3.24 — Ax / BoTorch composable BO primitives** — modular acquisition + outcome transforms.
- **T3.25 — NEvergrad portfolio meta-optimiser** — automatic algorithm selection across the EA/CMA-ES/PSO portfolio.
- **T3.26 — Random search baseline (Bergstra-Bengio JMLR 2012)** — the must-have baseline that everything else must beat; currently absent.

### Multi-objective extensions
- **T3.27 — IBEA / SMS-EMOA** — indicator-based multi-objective EAs.
- **T3.28 — ParEGO (Knowles 2006 IEEE TEC)** — multi-objective Bayesian optimisation.
- **T3.29 — Hypervolume-based EI** — MO-BO acquisition.

### LP / MIP
- **T3.30 — Branch-and-bound for MIP** — combinatorial extension of LP.
- **T3.31 — Cutting-plane (Gomory cuts)** — integer-programming primitive.
- **T3.32 — Dual simplex** — alternative to primal simplex for warm-start scenarios.
- **T3.33 — Network simplex** — specialised LP for transportation/assignment problems (currently `transport/` does Sinkhorn entropy-regularised, not exact).

### Specialised root-finding / 1-D
- **T3.34 — Müller's method** (complex roots).
- **T3.35 — Laguerre's method** (polynomial roots, used by JT polynomial-root finder).
- **T3.36 — Bairstow's method** (real polynomial root finder; pairs of complex conjugate roots).
- **T3.37 — Jenkins-Traub** (polynomial root finder; standard in Boost.Math, scipy).
- **T3.38 — Aberth-Ehrlich** (simultaneous polynomial root finder; modern competitor to JT).

### Interpolation extensions (would join `interpolate.go`)
- **T3.39 — Akima spline (1970)** — smoother than natural cubic on noisy data.
- **T3.40 — Monotone cubic Hermite (Fritsch-Carlson 1980)** — preserves monotonicity.
- **T3.41 — Catmull-Rom / centripetal Catmull-Rom** — graphics standard.
- **T3.42 — B-spline / NURBS** — CAD standard.
- **T3.43 — Barycentric Lagrange (Berrut-Trefethen 2004)** — numerically stable polynomial interp.
- **T3.44 — Chebyshev interpolation** — pairs with Clenshaw-Curtis quadrature in calculus/.

**Tier 3 total: ~5,500 LOC, 44 algorithms.** Brings `optim/` to research-frontier parity.

---

## Cross-package coupling notes

- **T1.13 (MPC LP) / T1.14 (QP) / T2.34 (QP IPM) / T3.1 (SOCP) / T3.2 (SDP)** all need `linalg/cholesky` (modified Cholesky for indefinite Hessians) and the existing dense LU. SDP needs symmetric-eigendecomposition (currently `linalg` ships QR but symmetric eigensolver coverage should be verified by slot 100/linalg-perf).
- **T2.21–T2.25 (Bayesian)** would benefit from a Gaussian-process regressor in `prob/`; absent today. TPE (T2.22) and Hyperband (T2.23) are GP-free and can ship first.
- **T2.31 (SPSA)** composes with autodiff slot 11-15 black-box-gradient consumers — explicit non-AD gradient-noise path.
- **T1.6–T1.8 (Adam/RMSprop/SGD)** are the natural pair for autodiff's GARCH/Heston/SABR consumers (autodiff-api slot 14 flagged this).
- **T1.2 (LM) + T1.12 (Gauss-Newton)** belong in their own `optim/lsq` sub-package by analogy with `proximal/`.
- **T2.26–T2.30 (NSGA-II/III, MOEA/D, SPEA2, hypervolume)** belong in `optim/multiobj`.
- **T2.21–T2.25 (Bayesian/surrogate)** belong in `optim/bayes`.
- **T3.30–T3.33 (MIP/dual-simplex/network-simplex)** belong in `optim/integer`.
- **T1.13's MPC LP** logically belongs alongside the existing `linear.go` simplex.

## Sprint ordering (recommendation for parent agent)

1. **Sprint 1 (~1 week):** T1.13 (MPC LP) replaces F7 wrong algorithm → unblocks LP correctness; T1.4+T1.5 (Brent min/root) → closes 101 F10/F11 1-D-method gaps; T1.1 (CG) + T1.9 (Nelder-Mead) → covers two of the most-missed scipy primitives. ~700 LOC.
2. **Sprint 2 (~1 week):** T1.6–T1.8 (Adam/AdamW/RMSprop/Adagrad/SGD-Nesterov) → unblocks autodiff calibration consumers. T1.2+T1.12 (LM+GN) → unblocks nonlinear-LSQ consumers. T1.3 (TR-Newton+Steihaug) → globally-convergent unconstrained workhorse. ~500 LOC.
3. **Sprint 3 (~1 week):** T1.10 (DE) + T1.11 (PSO) + T2.18 (CMA-ES) → completes the global-search canon. T1.14 (OSQP-style QP) → unblocks constrained QP. ~800 LOC.
4. **Sprint 4 (~1 week):** T2.10–T2.13 (SQP / AugLag / Frank-Wolfe / projected-gradient) → constrained NLP basics. T2.26 (NSGA-II) → multi-objective entry point. ~700 LOC.
5. **Sprint 5+ (Tier 2/3):** Bayesian, online, multi-objective extensions per consumer-pull priority.

**Tier 1+2 total: ~4,800 LOC over 5 sprints to reach scipy.optimize+NLopt+Optuna feature parity.**

## Web-research crosscheck (2026-05-07)

scipy.optimize 1.13 ships: minimize_scalar (Brent ✗ here, golden_section ✓, bounded ✗); minimize (Nelder-Mead ✗, Powell ✗, CG ✗, BFGS ✗, L-BFGS-B partial, TNC ✗, COBYLA ✗, SLSQP ✗, trust-* ✗); least_squares (LM ✗, TRF ✗, dogleg ✗); root (hybrid ✗, broyden ✗, anderson ✗, Krylov ✗); linprog (highs ≈ MPC ✗ — current InteriorPoint is wrong); milp ✗; quadratic_assignment ✗; differential_evolution ✗; basinhopping ✗; dual_annealing ✓ (SA exists); shgo ✗; direct ✗.

NLopt 2.7 algorithms missing: 39 of ~50 algorithms (only L-BFGS, simplex-LP, GA-roughly, SA present in form).

Optuna 4.0 samplers missing: TPE, GP-EI, CMA-ES, NSGA-II, NSGA-III, BoTorch, GridSampler, RandomSampler, QMC. (RandomSampler is T3.26 — embarrassingly missing baseline.)

Pyomo / IPOPT: NLP solver T3.22 is genuinely a multi-month port; ranks Tier 3 correctly.

The Mehrotra MPC LP (T1.13), CMA-ES (T2.18), NSGA-II (T2.26), and Bayesian-Optimisation TPE (T2.22) are the four single-PR additions that would close the largest reputational gap with scipy/Optuna.

---

## Files referenced (absolute paths)

- `C:\limitless\foundation\reality\optim\gradient.go`
- `C:\limitless\foundation\reality\optim\gradient_validated.go`
- `C:\limitless\foundation\reality\optim\linear.go`
- `C:\limitless\foundation\reality\optim\rootfind.go`
- `C:\limitless\foundation\reality\optim\genetic.go`
- `C:\limitless\foundation\reality\optim\metaheuristic.go`
- `C:\limitless\foundation\reality\optim\interpolate.go`
- `C:\limitless\foundation\reality\optim\proximal\admm.go`
- `C:\limitless\foundation\reality\optim\proximal\fbs.go`
- `C:\limitless\foundation\reality\optim\proximal\operators.go`
- `C:\limitless\foundation\reality\optim\transport\sinkhorn.go`
- `C:\limitless\foundation\reality\optim\transport\wasserstein1d.go`
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\101-optim-numerics.md` (precedent audit)

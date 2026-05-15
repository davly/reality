# 355 — research-libs-python (Python scientific stack 2025-26)

## Headline
Python's 2025-26 stack converged on JIT/XLA compilation, Arrow-backed columnar memory, and composable functional transforms — Reality's Go core has none of these but should mirror the *pure-function + transform* discipline (jit/grad/vmap) and golden-vector portability.

## Survey

### NumPy 2.0 / 2.2 (Jun 2024 – 2025)
First major release since 2006. ABI break, scalar promotion overhaul (no more value-dependent dtype promotion — the long-standing footgun), 10% reduction in main-namespace symbols, ~80% in `numpy.lib`. New `StringDType` (variable-length strings as a first-class user-defined dtype), via the long-promised user-DType API (NEP 41/42). Sort algorithm changes break exact reproducibility for unstable methods (argsort/argpartition) — relevant warning for Reality's golden files: *algorithmic improvements break bit-exact pins*. NumPy 2.x supports the array-API standard and exposes `numpy.array_api`. Migration is automatable via Ruff's `NPY` plugin. License: BSD-3-Clause (Reality MIT-compatible). Lesson: surface a *stable namespace* rule and a *value-independent type promotion* invariant before adding heterogeneous dtypes.

### SciPy 1.15 (Jan 2025)
First stable release with Python 3.13 free-threading wheels on PyPI (preliminary GIL-free support — Reality should track GIL removal because Python wrappers will eventually want true parallel calls into Go). `scipy.optimize` added more vectorised root-finders and array-API compatible elementwise ops; `scipy.sparse` continues consolidation onto `_array` API replacing matrix; `scipy.special` extended hypergeometric coverage. `scipy.signal` filter design API now accepts array-API inputs. License BSD-3. The scope of `scipy.special` (Bessel, Airy, hypergeometric, Marcum Q, etc.) is the biggest gap vs Reality — Reality has trig/exp/log via stdlib but no special-function package; consider a `special` sub-package as a first-class addition.

### JAX 0.4.x (active 2025)
The *transform composition* model is the single most copy-worthy idea: `jit`, `grad`, `vmap`, `pmap`, `pjit`, `shard_map` all operate on *pure functions* and compose freely. Tracing-based; XLA backend (HLO IR) does fusion + scheduling. JAX requires functional purity — Reality already enforces this stylistically (zero global state, output buffers passed in). Contrast with Reality's `autodiff` package which uses a Wengert tape (closer to PyTorch's eager autograd than JAX's source-to-source `grad`). Adding a tracing-based forward-mode JVP and a `vmap`-style batched dispatch (Go generics make this tractable) would be a leap. License Apache-2.0 (MIT-compatible). Watch: `shard_map` for explicit SPMD, `Equinox` for module patterns.

### PyTorch 2.5 / 2.6 (2025)
`torch.compile` matured: TorchInductor + Triton codegen. FSDP2 (`fully_shard`) replaces FSDP1 with per-parameter sharding, ~7% lower peak memory and 1.5% throughput gain. `torch.compile` × FSDP2 × Float8 demonstrated on 70B Llama at 256 GPUs. Eager-by-default + opt-in compilation is the *reverse* of JAX's compile-by-default. Wengert-tape autograd is what Reality's `autodiff` mimics. Lessons: (a) the eager/compiled split — Reality could expose both an "interpretive" and a "fused" path (pre-allocated batch APIs hint at this); (b) `nn.Module` as state container vs JAX/Equinox PyTrees — Reality's stateless functions are JAX-style, keep it. License BSD-3. `torch.distributions` is a serious reference for `prob` (LKJ, MixtureSameFamily, transformed distributions).

### pandas 2.3 → 3.0 (Jun 2025 → Jan 2026)
PyArrow becomes a *required* dependency in 3.0; default string dtype now Arrow-backed (`pyarrow.string`). NumPy still primary for numeric columns. Copy-on-write semantics are now the default. The Arrow shift demonstrates that *columnar memory layout + cross-language zero-copy* is now table stakes. Reality's golden-file JSON layer is the rough analog (cross-language *truth* exchange) but Arrow IPC would be a more efficient on-disk/wire format for large numeric vectors. License BSD-3. Action item: consider Arrow IPC as a secondary serialisation for golden vectors > N elements (JSON for IEEE-754 edge-case readability, Arrow for bulk).

### scikit-learn 1.6 / 1.7 / 1.8 (Jan 2025 – Dec 2025)
1.6 added experimental free-threaded CPython support; 1.7 added Python 3.14; 1.8 (Dec 2025) consolidated free-threaded support across estimators. Array-API standard adoption continues — estimators dispatch to NumPy / CuPy / array_api_strict. The `set_output("pandas"|"polars")` API is a clean *transformer-output-as-config* pattern. License BSD-3. Reality has no ML, intentionally; the relevant idea is the array-API protocol — a *backend-agnostic* dispatch surface. A Go analog would be parametric typing over `[]float32` / `[]float64` / `mat.Dense` so the same kernel runs on different precisions.

### statsmodels 0.14.x (active 2025)
ARIMA, SARIMAX, VAR, GLM, mixed-effects, robust regression, kernel density. ARIMA fit supports six estimators (`statespace`, `innovations_mle`, `hannan_rissanen`, `burg`, `innovations`, `yule_walker`). State-space framework is the core abstraction — Kalman filters, MLE via EM. Reality's `prob` covers distributions/HT/info but lacks time-series and GLM. License BSD-3. Add list: ARIMA(p,d,q), Holt-Winters, Kalman filter (state-space form), GLM with link functions (logit/probit/log/identity/inverse), robust covariance (HC0-HC3, HAC). These are pure math, no allocations issue, and have decades-stable golden references.

### mpmath 1.4 (Feb 2026)
Pure-Python arbitrary-precision real/complex floating-point + special functions. BSD-3. Reality already uses Go `math/big` at 256-bit for golden-file generation — same idea, different language. mpmath's *function catalog* is the reference: zeta, polygamma, hypergeometric pFq, elliptic integrals, Lambert W, Riemann-Siegel, Borel summation. Worth mining as a checklist for `special` if added. mpmath also exposes `mp.workdps(n): ...` precision-context — Reality could mirror this with a `bigfloat.WithPrec(n)` builder for golden-file generation, since current generators hard-code 256 bits. Free-threaded since 1.4.

### NetworkX 3.6 / igraph 0.11 (2025)
NetworkX: pure Python, ~50M downloads/yr, slow but ergonomic. igraph: C core with Python bindings, much faster. NetworkX 3.x added a *backend dispatch* mechanism (`NETWORKX_BACKEND=parallel|cugraph|graph-blas`) — the same call sites run on CPU-parallel or GPU backends. License BSD (NetworkX) / GPL-2 (igraph — *not MIT-compatible, do not vendor*). Reality's `graph` package matches NetworkX's algorithm scope (Dijkstra, A*, BFS/DFS, topological). Missing: max-flow (Edmonds-Karp / Dinic), bipartite matching (Hopcroft-Karp), strongly-connected (Tarjan), articulation points/bridges, betweenness centrality with Brandes, Louvain/Leiden community detection.

### GUDHI / scikit-tda / persim (active 2025)
TDA pipeline: GUDHI (C++/Python, MIT for core, GPL for some modules — *check per-module before vendoring ideas*) computes simplicial complexes (Vietoris-Rips, alpha, witness) + persistent homology. Ripser/persim (MIT) compute persistence diagrams; Mapper for topological clustering. Persistence landscapes / images as ML-friendly vectorisations of diagrams. Reality has no TDA. Pure-math additions worth golden-pinning: Čech / Vietoris-Rips filtration, persistent homology (matrix reduction algorithm), bottleneck and Wasserstein distances between persistence diagrams. Computationally heavy but algorithmically clean and citation-rich.

### CVXPY 1.5 / 1.6 (2024-2025)
DCP (Disciplined Convex Programming) modeling layer over solvers (Clarabel default since 1.5; ECOS dropped in 1.6; HiGHS for MILP). 1.6 added N-dim expressions (NumPy ndarray analog), variable sparsity attribute, Python 3.13. License Apache-2.0. Reality's `optim` covers unconstrained + simulated annealing/GA/simplex but no DCP-style modeling. Architectural lesson: CVXPY is a *modeling DSL* that compiles to a canonical conic form (LP/QP/SOCP/SDP/EXP), then dispatches to solvers. A Go analog would expose `Variable`, `Constraint`, `Minimize` types with operator overloading via methods; lower to a slim conic representation. Useful even without external solvers if Reality grows a native interior-point QP/SOCP solver.

### Pyro / NumPyro (active 2025)
Pyro: PyTorch-backed PPL. NumPyro: JAX-backed, same API; uses HMC / NUTS sampler. Both are MIT (Pyro) / Apache-2.0 (NumPyro). Effect-handler architecture (`numpyro.handlers.{trace, replay, condition, do}`) is a beautiful composable design — every model is just a Python callable, and inference is a *transformation* (cf. JAX). Reality's `prob` has Bayesian inference primitives but no PPL. A NumPyro-style trace handler in Go is plausible (closures + a runtime stack), but goes against the no-allocation/fast-path discipline. Better: lift NumPyro's *distribution algebra* — `TransformedDistribution`, bijector composition (StickBreaking, Exp, Affine), `MixtureSameFamily`. These are pure math.

### Diffrax (active 2025-26)
JAX-native ODE/SDE/CDE solvers (Tsit5, Dopri8, Kvaerno5, symplectic Yoshida6/8, implicit Kvaerno, multi-step adaptive). Apache-2.0. `vmap`-able everything (incl. integration interval), PyTree state, multiple adjoint methods (recursive checkpointing, Bacon-Newman, optimal-online). Frequently ~100× faster than `torchdiffeq` due to JAX/XLA fusion. Reality's `chaos` package has Lorenz/Van der Pol but only basic RK4. Diffrax's catalog should be the target: Tsit5 (5(4)) as default ERK; Dopri8 for tight tolerance; Kvaerno5 for stiff; symplectic Verlet/Yoshida for Hamiltonian. Adjoint sensitivity for parameter gradients integrates with `autodiff`.

### Optax (active 2025)
Composable gradient-transformation library for JAX. Apache-2.0. Each optimiser is a pipeline of `GradientTransformation` objects (clip → scale_by_adam → add_decayed_weights → scale_by_learning_rate). The chain abstraction is general: any optimiser is `chain(*transforms)`. Reality's `optim` exposes named optimisers (Adam, L-BFGS, etc.) as monolithic functions — refactoring into a `chain([...])` pipeline would unlock arbitrary user composition (e.g. AdamW = Adam + weight-decay + scale_by_lr). Loss functions also live here (huber, log-cosh, focal, softmax-CE), useful reference set.

## Aggregate themes Reality should track
- **Composable function transforms (JAX)**: `jit ∘ grad ∘ vmap` over pure functions. Reality already enforces purity; a Go-generic `Vmap[T]`/`Jvp[T]` helper is feasible and high-leverage.
- **Eager × compiled split (PyTorch)**: keep the interpretive path readable; add a fused path with pre-allocated buffers (already present in some packages — generalise).
- **Array-API / dtype dispatch**: SciPy/sklearn standardise on the array-API protocol; Reality should pick a Go equivalent (generics over `~float32 | ~float64`).
- **State-space framework (statsmodels)**: Kalman filter is *the* unifying primitive for time-series — single addition unlocks ARIMA/HW/UCM/dynamic-factor.
- **Transformation pipelines (Optax)**: monolithic optimisers → composed `chain(*ops)`. Same idea applies to filters in `signal`, integrators in `chaos`.
- **Effect handlers (NumPyro)**: traces/conditioning as composable handlers — even a small Go version (closure-based) makes Bayesian inference much more general.
- **Pre-allocated, no-alloc hot paths (Diffrax `SaveAt` / Optax `state`)**: align with Reality's existing rule.
- **Free-threaded Python (CPython 3.13+)**: future Python wrappers around Reality will demand thread-safety guarantees — document them now.
- **Arrow IPC for bulk vectors**: complement JSON golden files with Arrow for >10k-element vectors (cross-language zero-copy).
- **Backend dispatch (NetworkX/sklearn)**: same call site, different backend. Reality could expose `Backend{CPU, GPU-via-cgo, BigFloat}` for cross-validation.

## Anti-patterns
- **Value-dependent type promotion** (NumPy <2.0): Reality must define promotion strictly on declared types, never on values.
- **Mutable global state for randomness** (older NumPy `np.random.seed`): the new `Generator` API is correct — Reality should follow (explicit `*rand.Rand` everywhere, never package globals).
- **DataFrame-as-everything** (pandas pre-Arrow): a single object trying to own dtypes, indices, IO, plotting; it leaks abstractions. Reality's per-package separation is healthier.
- **Eager-only Python loops over arrays** (legacy SciPy): use vectorised + batched APIs from the start.
- **Hidden recompilations (`torch.compile` graph breaks)**: be deterministic about what gets specialised. Reality's static Go code sidesteps this naturally.
- **GPL contagion (igraph, parts of GUDHI)**: do not vendor; reimplement from papers. MIT/BSD/Apache-2.0 only.
- **Distribution objects without bijector composition**: NumPyro's `TransformedDistribution` is the right shape; avoid baking `LogNormal = exp(Normal)` as a special case.
- **Optimiser monoliths**: Optax's `chain` proves you can decompose; Adam-as-a-blob is a smell.

## License compatibility check
- **MIT-compatible (safe to study/port ideas)**: NumPy, SciPy, JAX, PyTorch, pandas, scikit-learn, statsmodels, mpmath, NetworkX, scikit-tda/persim/Ripser, CVXPY (Apache-2.0), Pyro, NumPyro (Apache-2.0), Diffrax (Apache-2.0), Optax (Apache-2.0).
- **GPL — DO NOT vendor or copy code**: igraph (GPL-2), some GUDHI modules (GPL).
- All Reality contributions: reimplement from primary papers, cite source in package doc as Reality already does. Apache-2.0/BSD/MIT code may inform design but no copy-paste.

## Sources
- [NumPy 2.0 Release Notes](https://numpy.org/devdocs/release/2.0.0-notes.html)
- [NumPy 2.2 Release Notes](https://numpy.org/devdocs/release/2.2.0-notes.html)
- [SciPy 1.15.0 Release Notes](https://docs.scipy.org/doc/scipy-1.15.3/release/1.15.0-notes.html)
- [JAX docs — key concepts](https://docs.jax.dev/en/latest/key-concepts.html)
- [JAX CHANGELOG](https://github.com/jax-ml/jax/blob/main/CHANGELOG.md)
- [PyTorch FSDP2 docs](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [State of torch.compile (Aug 2025) — ezyang](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)
- [Maximizing Training Throughput Using PyTorch FSDP and torch.compile](https://pytorch.org/blog/maximizing-training-throughput/)
- [pandas 3.0 What's New (Jan 2026)](https://pandas.pydata.org/docs/dev/whatsnew/v3.0.0.html)
- [pandas 2.3 What's New (Jun 2025)](https://pandas.pydata.org/docs/whatsnew/v2.3.0.html)
- [scikit-learn Release History](https://scikit-learn.org/stable/whats_new.html)
- [scikit-learn 1.6 What's New](https://scikit-learn.org/stable/whats_new/v1.6.html)
- [statsmodels ARIMA docs](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [mpmath project](https://mpmath.org/)
- [mpmath GitHub](https://github.com/mpmath/mpmath)
- [GUDHI library](https://gudhi.inria.fr/)
- [Scikit-TDA](https://github.com/scikit-tda)
- [CVXPY updates](https://www.cvxpy.org/updates/)
- [CVXPY 1.6 release](https://www.cvxpy.org/version/1.6/)
- [NumPyro GitHub](https://github.com/pyro-ppl/numpyro)
- [Pyro project](https://pyro.ai/)
- [Diffrax docs](https://docs.kidger.site/diffrax/)
- [Optax docs](https://optax.readthedocs.io/)
- [JAX vs PyTorch 2025 (UMA)](https://umatechnology.org/jax-vs-pytorch-differences-and-similarities-2025/)
- [NetworkX docs](https://networkx.org/documentation/stable/reference/index.html)
- [igraph Python API](https://python.igraph.org/en/main/api/igraph.Graph.html)

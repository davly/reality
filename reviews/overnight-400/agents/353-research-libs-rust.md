# 353 — research-libs-rust (Rust math ecosystem 2025-26 lessons + anti-patterns)

## Headline
Rust ecosystem leads on type-system rigour (const-generic dimensions, trait-based solvers, lazy query plans) and BLAS-pluggability — reality should adopt the architectural ideas (compile-time shape checks, pluggable backends, trait-based "problem" abstractions, columnar interchange) without porting code; most crates are MIT/Apache-2 dual licensed (compatible) but reimplementation rule still applies.

## Survey

### nalgebra 0.34.x (BSD-3 / Apache-2)
- **Scope:** dense + sparse linear algebra, geometry (quaternions, isometries, rotations), kinematics primitives. Type-level dims via `SMatrix<T, R, C>` const generics + residual `typenum` for arithmetic on dims.
- **Architecture:** unified `Matrix<T, R, C, S>` parameterized by storage `S` (owned, slice, view). Const-generic shapes detect mis-multiplication at compile time. Heavy use of associated types and `Allocator` traits.
- **Strengths:** zero-cost compile-time shape checks; `nalgebra-glm` mirrors GLSL for graphics users; SIMD via `simba` abstraction.
- **Weaknesses:** notorious documentation density; `typenum` residue produces gnarly compiler errors; no GPU backend; over-engineered for code that just wants a 3×3 mul.
- **Reality lesson:** reality's `geometry`/`linalg` could expose typed-shape variants (`Vec3`, `Mat4`) alongside dynamic `Matrix` — Go lacks const generics but interface-typed wrappers (`Vec3 [3]float64`) give 80% of the safety with 0% of the build pain.

### ndarray 0.16.x (MIT / Apache-2)
- **Scope:** N-dimensional owned/view arrays. NumPy analog. Pluggable BLAS via `ndarray-linalg`; stats via `ndarray-stats`.
- **Architecture:** single `ArrayBase<S, D>` parameterized by storage and dim; `Array`, `ArrayView`, `ArrayViewMut`, `CowArray` are aliases. Row-major default, F-order toggleable. Element-wise ops on `*`, dot products explicit.
- **Strengths:** zero-copy slicing/broadcast via views, `s![]` macro for slice DSL, `parallel` feat enables `rayon` per-axis parallelism.
- **Weaknesses:** dimension type (`Ix1..Ix6` + `IxDyn`) leaks into every signature; iterator chains can defeat vectorisation if not careful.
- **Reality lesson:** reality's `linalg.Matrix` should consider exposing **views** (immutable + mutable slices over an existing buffer) rather than always copying — matches Pistachio's "no allocs in hot paths" rule. Adopt the `axis` parameter convention universally for stats reductions.

### candle 0.9.x (Apache-2 / MIT)
- **Scope:** minimalist tensor + autograd + nn primitives + transformer models. CPU/CUDA/Metal backends. Goal: serverless inference without Python.
- **Architecture:** `Tensor` is reference-counted device-side handle; ops are recorded for autograd; `safetensors` + GGUF native. Backend trait abstracts CPU/CUDA/MPS.
- **Strengths:** binary size 100×smaller than libtorch; quantized GGUF inference on edge; pure-Rust default kernels.
- **Weaknesses:** training story is thin (vs Burn); kernel coverage gaps versus PyTorch; opinionated against dynamic shapes.
- **Reality lesson:** **device-trait abstraction** for compute backends is the right shape — reality should keep math device-agnostic but design APIs so a future GPU/SIMD backend can slot in (functions take `[]float64` slices, not channels/goroutines).

### polars 0.50+ (MIT)
- **Scope:** columnar DataFrame / query engine. Eager + Lazy APIs. Arrow-backed memory.
- **Architecture:** `LazyFrame` is a logical-plan builder. `Expr` AST is composable; physical planner does projection-pushdown, predicate-pushdown, common-subexpression elimination before execution. Vectorised on Arrow chunks; multi-threaded by morsel.
- **Strengths:** query optimization is the killer feature; lazy ops compose without intermediate allocation; columnar layout is cache-friendly.
- **Weaknesses:** API surface is enormous; Rust feature-flag soup (≈30 features); error types are wide unions.
- **Reality lesson:** reality is not a DataFrame, but the **expression-AST + optimizer** pattern applies anywhere we chain operations — e.g. signal processing pipelines, autodiff graphs. A future `signal.Pipeline` could compile a fused kernel from a plan. Lazy is opt-in; eager remains default.

### statrs 0.18.x (MIT)
- **Scope:** continuous + discrete distributions, statistical functions (gamma, beta, erf), hypothesis tests.
- **Architecture:** trait-driven — `Continuous`, `ContinuousCDF`, `Discrete`, `DiscreteCDF`, `Mode`, `Median`, `Min`, `Max`. Each distribution implements relevant traits; samplers via `rand::Distribution`.
- **Strengths:** trait composition gives uniform `pdf/cdf/inverse_cdf/sample` surface; ports test suite from .NET Math.NET (proven correctness).
- **Weaknesses:** ported-from-C# heritage means some idioms feel un-Rusty; performance not its focus.
- **Reality lesson:** reality's `prob` package should formalize a **distribution interface** (Go interface, since no traits) with `PDF/CDF/Quantile/Sample/Mean/Variance` so consumers can swap distributions polymorphically. Ports-as-validation (statrs ↔ Math.NET) parallels reality's Go ↔ Python golden file idea — copy this rigour, don't port code.

### argmin 0.10.x (MIT / Apache-2)
- **Scope:** unconstrained + constrained optimization (L-BFGS, CG, Newton, Nelder-Mead, particle swarm, simulated annealing).
- **Architecture:** problem defined by `CostFunction`/`Gradient`/`Hessian` traits with associated `Param`/`Output`/`Hessian` types. `Solver` trait + `Executor` provides termination, observers, checkpointing for free. Math backend pluggable (nalgebra, ndarray, raw `Vec`).
- **Strengths:** **best architectural pattern in this survey** — separation of (problem ⊥ solver ⊥ math backend) is exemplary; checkpointing built into executor.
- **Weaknesses:** generics make signatures intimidating; some solvers lag behind academic SOTA.
- **Reality lesson:** reality's `optim` should adopt **(Problem interface, Solver interface, Driver/Executor)** triplet. Currently solvers tend to take closures — that works, but a Problem interface would unlock observers, checkpointing, and uniform termination criteria across all algorithms. Highest-value pattern to copy.

### linfa 0.7.x (Apache-2 / MIT)
- **Scope:** classical ML: linear models, clustering, trees, kernel methods, preprocessing. scikit-learn analog.
- **Architecture:** workspace of sub-crates per algorithm family (`linfa-linear`, `linfa-clustering`, ...). Core crate defines `Dataset`, `Fit`, `Predict`, `Transformer` traits. ndarray-based.
- **Strengths:** modular sub-crate publishing means consumers pull only what they need; `Dataset` abstraction unifies labelled/unlabelled flows.
- **Weaknesses:** algorithm coverage incomplete vs sklearn; documentation is uneven across sub-crates.
- **Reality lesson:** reality's monorepo + sub-package layout already mirrors this. **Per-package READMEs** with the same structure (statement-of-scope, citations, golden-file count) is worth standardizing — linfa's sub-crate READMEs are inconsistent and that's a tax on adopters.

### petgraph 0.6.x / 0.7.x (MIT / Apache-2)
- **Scope:** directed/undirected graphs + classical algorithms (Dijkstra, A*, MST, SCC, isomorphism, max-flow).
- **Architecture:** four storage backends — `Graph` (adj-list with `NodeIndex`), `StableGraph` (preserves indices on delete), `GraphMap` (hashmap, edges keyed by node value), `MatrixGraph` (adj-matrix). Algorithms generic over `IntoNeighbors` / `Visitable` / `IntoEdges` traits.
- **Strengths:** **trait-driven algorithm generality** — same Dijkstra implementation works on any storage; `GraphRef` etc. let library users wrap their own representations.
- **Weaknesses:** ongoing migration to traits leaves some algos still tied to `Graph`; visitor API has two generations of design coexisting.
- **Reality lesson:** reality's `graph` package should expose **iterator-style "visitor" interfaces** so algorithms can run over external graph representations (e.g. caller's adjacency list) without copying. Single concrete `Graph` type is fine for v1, but algorithms should accept an interface, not a concrete type.

### num-traits / num-bigint / num-rational / num-complex (MIT / Apache-2)
- **Scope:** numeric trait hierarchy (Zero, One, Num, Float, Signed, ...) + arbitrary-precision integers, rationals, complex numbers.
- **Architecture:** `num-traits` is the foundation; everything generic over `T: Float` or `T: Num + Copy`. Sub-crates are independent and re-exported by the meta `num` crate.
- **Strengths:** ubiquitous — most Rust math crates depend on `num-traits`; provides the "what is a number" bedrock.
- **Weaknesses:** trait hierarchy has historical inconsistencies; some traits (`Float`) bundle too many capabilities.
- **Reality lesson:** Go has no traits, but reality already uses interface-style: `prob` reductions take `[]float64` not `Number`. **Consider extracting a `numeric.Real` interface** for shared root-finding/integration over arbitrary scalar types — but only if consumers actually need it (YAGNI applies).

### arrow-rs / parquet 57+ (Apache-2)
- **Scope:** Apache Arrow columnar in-memory format + Parquet on-disk; FlightRPC; compute kernels.
- **Architecture:** strongly-typed `Array` per logical type; `RecordBatch` is the row-group analog; metadata-driven IO. Recent thrift parser rewrite gave ~4× metadata speedup.
- **Strengths:** **lingua franca for cross-language data interchange** — Python, Go (via arrow-go), C++ all read same memory layout zero-copy.
- **Weaknesses:** API surface huge; type system intricacy (logical vs physical types) is a learning curve.
- **Reality lesson:** reality's golden-file format is JSON. For very large vector sets (e.g. 10k-vector signal-processing fixtures), **Arrow IPC files would be order-of-magnitude smaller and faster to parse** — and Go/Python/C++/C# all have arrow readers. Don't switch the small fixtures, but consider Arrow for any vector-of-floats fixture > ~1MB.

### faer 0.20+ (MIT / Apache-2)
- **Scope:** dense linear algebra focused on medium/large matrices: factorisations (LU, QR, Cholesky, SVD, EVD, Schur), solvers, sparse on the way.
- **Architecture:** explicit SIMD per arch (x86-64, NEON; SVE/SME/RVV planned), BLIS-style packed GEMM. Pure Rust kernels — no BLAS dependency required.
- **Strengths:** **competitive with OpenBLAS on dense GEMM in pure Rust** — proves portable high-performance is achievable without C deps.
- **Weaknesses:** poor fit for tiny matrices (3×3 skinny ops); API still settling between versions.
- **Reality lesson:** **pure-Go BLAS-quality kernels are achievable** if SIMD is leveraged via assembly or `golang.org/x/sys/cpu` + intrinsics-via-codegen. For now reality's targets are correctness and golden parity, not GEMM throughput — but faer is proof point that "zero deps + fast" is not contradictory.

### Burn 0.16+ (Apache-2 / MIT)
- **Scope:** full deep-learning stack: tensors, autograd, optimizers, training loop, multi-backend (CUDA, ROCm, Metal, Vulkan, WebGPU, LibTorch, Candle).
- **Architecture:** `Backend` trait is the cornerstone; tensors generic over backend; same model trains on WebGPU and CUDA with no code change. Autodiff is a backend wrapper (`Autodiff<B>`).
- **Strengths:** **backend-as-type-parameter** is a clean way to compose autodiff, quantization, profiling.
- **Weaknesses:** outside reality's scope; surface area is enormous.
- **Reality lesson:** reality's autodiff (if/when added) should consider Burn's pattern: **`Autodiff[Backend]` as a wrapper over a value backend**, rather than a parallel set of types. In Go that means an interface `Tensor` and an `AutodiffTensor` that wraps and records.

### peroxide 0.41 (BSD-3 / MIT)
- **Scope:** "batteries-included" R/MATLAB/NumPy-style scripting in Rust: linalg, ODEs, stats, special functions.
- **Architecture:** macro DSL (`c!`, `matrix!`) for terse construction; pure-Rust default with optional BLAS feature `O3`.
- **Strengths:** lowest cognitive overhead in the Rust scientific ecosystem; explicit "default = pure Rust, no system deps" is closest spirit-match to reality.
- **Weaknesses:** macro-heavy API hides costs; smaller user base than ndarray/nalgebra.
- **Reality lesson:** reality already follows the "zero deps, pure Go" stance. peroxide validates this is a viable niche. Their **dual-mode design (pure-Rust default, opt-in BLAS via feature flag)** maps to a reality future where a `linalg/blas` sub-package wraps a CGo BLAS for users who want it — without polluting the core.

## Aggregate themes Reality should track

- **Trait/interface-driven algorithms:** argmin (Problem×Solver), petgraph (Visitable), statrs (Distribution), Burn (Backend) all pay for the abstraction with composability dividends. Reality should design `optim`, `prob`, `graph` algorithms to accept Go interfaces, not concrete structs.
- **Compile-time shape checks where cheap:** Go can't do nalgebra-style const generics, but typed wrappers (`Vec3`, `Quat`, `Mat3`) catch the same family of bugs and document intent. reality already has quaternions; extend the convention.
- **Views over copies:** ndarray's `ArrayView`/`ArrayViewMut` model is the right answer for "no allocs in hot paths." Go's slice header is half-way there; reality functions should accept `[]float64` views and output buffers, never return new slices in hot paths. Most do; audit for the rest.
- **Ports as validation, not as code:** statrs validating against Math.NET, ndarray-linalg validating against LAPACK — reality's golden-file-cross-language doctrine is the same idea. Lean into it harder: every transcendental should have a NIST/Math.NET/SciPy reference cited.
- **Pluggable backends without API change:** Burn's `Backend` trait, faer's SIMD specialization, candle's device abstraction. reality's APIs should be designed so a future SIMD or Cgo-BLAS backend slots in without changing call sites.
- **Lazy/expression layer is opt-in:** polars proves the value but never forces it. If reality ever adds a fused-kernel pipeline (signal, autodiff), keep eager as default.
- **Sub-crate publishing for consumer minimalism:** linfa, num, ndarray-* all let consumers pull only what they need. reality is a single Go module with sub-packages — Go's package selectivity at import time gives this for free; preserve it (don't introduce internal packages that drag the world in via `init()`).
- **Arrow as cross-language fixture format for large data:** golden JSON is fine for ≤1k vectors; Arrow IPC for the rest.

## Anti-patterns Reality already avoids (or should)

- **Trait-soup compile errors (nalgebra):** Go interfaces are simpler — less power, less pain. Don't reach for `internal/typenum`-equivalents.
- **Macro DSLs for matrix construction (peroxide):** hides allocation, breaks `go vet`. Use plain literals.
- **Feature-flag explosions (polars, candle, ndarray-linalg):** Go has build tags but they're awkward; resist conditional compilation. Pure-Go default, separate package for optional acceleration.
- **C# port-itis (statrs):** porting another language's idioms verbatim leaves cruft. Reimplement from first principles, validate against the reference.
- **Multi-generation API coexistence (petgraph visitors):** when changing an interface, deprecate fully. Don't ship two ways to do the same thing.
- **Dependency on system BLAS/LAPACK in core (ndarray-linalg):** core must be self-contained. Acceleration is a sidecar.
- **Massive monolithic feature crates (polars, candle):** reality's per-domain sub-package layout already prevents this. Hold the line.
- **GPL contagion:** none of the surveyed crates is GPL (all MIT/Apache-2/BSD-3, all reality-MIT-compatible) but the reimplementation rule still applies — copy patterns, not code.

## Sources

- [nalgebra const generics (Dimforge)](https://www.dimforge.com/blog/2021/04/12/integrating-const-generics-to-nalgebra/)
- [nalgebra docs.rs latest](https://docs.rs/nalgebra/latest/nalgebra/base/struct.Matrix.html)
- [ndarray for NumPy users](https://docs.rs/ndarray/latest/ndarray/doc/ndarray_for_numpy_users/index.html)
- [rust-ndarray GitHub](https://github.com/rust-ndarray/ndarray)
- [Hugging Face Candle GitHub](https://github.com/huggingface/candle)
- [Burn vs Candle 2026](https://dasroot.net/posts/2026/04/rust-machine-learning-burn-vs-candle-framework-comparison/)
- [Polars under the hood (endjin)](https://endjin.com/blog/2026/01/under-the-hood-what-makes-polars-so-scalable-and-fast)
- [polars LazyFrame docs](https://docs.rs/polars/latest/polars/prelude/struct.LazyFrame.html)
- [argmin homepage](https://argmin-rs.org/)
- [argmin docs.rs](https://docs.rs/argmin/)
- [linfa GitHub](https://github.com/rust-ml/linfa)
- [statrs docs.rs](https://docs.rs/statrs/)
- [statrs-dev GitHub](https://github.com/statrs-dev/statrs)
- [petgraph docs.rs](https://docs.rs/petgraph/)
- [petgraph GitHub](https://github.com/petgraph/petgraph)
- [num-traits crates.io](https://crates.io/crates/num-traits)
- [arrow-rs 57.0.0 release](https://arrow.apache.org/blog/2025/10/30/arrow-rs-57.0.0/)
- [apache/arrow-rs](https://github.com/apache/arrow-rs)
- [ndarray-linalg GitHub](https://github.com/rust-ndarray/ndarray-linalg)
- [faer GitHub](https://github.com/sarah-quinones/faer-rs)
- [faer docs.rs](https://docs.rs/faer/latest/faer/)
- [Peroxide GitHub](https://github.com/Axect/Peroxide)
- [Burn GitHub](https://github.com/tracel-ai/burn)
- [Scientific Computing in Rust monthly](https://scientificcomputing.rs/monthly/2026-04)
- [Are we learning yet — scientific computing](https://www.arewelearningyet.com/scientific-computing/)
- [Full const generics goal 2026](https://rust-lang.github.io/rust-project-goals/2026/const-generics.html)

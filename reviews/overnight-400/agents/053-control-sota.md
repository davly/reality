# 053 | control-sota

**Scope.** SOTA library/engineering comparison for `C:\limitless\foundation\reality\control\` (`pid.go` 122, `transfer.go` 252, `filter.go` 116 — total ~490 LOC). Architecture/interface choices only. Sibling **051** owns numerical correctness (PID anti-windup variants, Durand-Kerner conditioning, complementary-filter algebra bug, signed-zero/NaN propagation). Sibling **052** owns the missing-primitive map (LQR, Kalman, MPC, H∞, sliding-mode, system-ID — ~90 named items / 20-tier ladder). This report is **disjoint** from both: it asks "how do peer libraries *type, compose, and dispatch* control objects, and which patterns should reality borrow zero-dep?" not "what numbers are wrong" or "what primitives are missing."

**TL;DR.** Reality's control surface is **two value types + one stateful struct** (`TransferFunction`, `PIDController`, `LowPass/HighPass/Complementary/RateLimiter` as free functions). That is the simplest of the SOTA spectrum. The SOTA cohort (python-control 0.10.x, Slycot 0.6.x, harold 1.0.x, do-mpc 4.6, CasADi 3.6, Modelica.Blocks.Continuous, MATLAB Control System Toolbox R2025b, Drake 1.30+) has converged on **eight engineering patterns** that aren't about *which* algorithm ships but about *how the user composes* it: (1) a single `LTI` umbrella interface, (2) `tf <-> ss <-> zpk <-> frd` round-tripping conversion, (3) operator overloading for series/parallel/feedback, (4) symbolic representation as the AD substrate (CasADi pattern), (5) discrete-and-continuous as one type with a `dt` field (python-control 0.9+), (6) frequency-response data (`FRD`) as a first-class peer to `tf`/`ss` (the "I have measurements, not a model" entry-point), (7) symbolic block-diagram composition (Modelica), (8) auto-differentiation through the dynamics for nonlinear MPC (CasADi/Drake). Of these eight, **three are zero-dep, zero-runtime-cost, golden-file-compatible, and immediately actionable on Reality's existing surface**: `LTI` interface (~30 LOC), `dt float64` discriminator on `TransferFunction` (~15 LOC, makes the future `DiscreteTransferFunction` of 052-L2 land cleanly), and operator-style `Series/Parallel/Feedback` constructors (~80 LOC, replaces the missing block-algebra surface). Engineering elegance ranks **CasADi > Drake > python-control > Modelica > harold > Slycot > MATLAB CST > reality**. Document discipline ranks **MATLAB CST > python-control > Modelica > Drake > CasADi > Slycot > harold > reality**.

---

## 1. The eight SOTA references — interface formats compared

### 1.1 python-control 0.10.x (released 2024-11, latest 0.10.2 in 2026-Q1)

**Headline data type.** Three sibling classes — `TransferFunction(num, den, dt=None)`, `StateSpace(A,B,C,D,dt=None)`, `FrequencyResponseData(response, omega, dt=None)` — *all subclassing the abstract base* `LTI`. A single `dt` field on every class is the continuous/discrete discriminator: `dt=None` means continuous, `dt=0` means "discrete but timestep unspecified," `dt=T` means discrete with sample period `T`. This is a 2018-era refactor (PR #214) that replaced two-class hierarchies with one.

**Construction ergonomics.**
```python
G = ct.tf([1], [1, 2, 1])              # 1 / (s² + 2s + 1)
Gd = ct.tf([1], [1, 0.5], dt=0.01)     # discrete, T=0.01s
G2 = G * 5                              # series, returns TransferFunction
G3 = G + G                              # parallel
H = ct.feedback(G, 1)                   # unit feedback
ss = ct.tf2ss(G)                        # type conversion
ct.bode_plot(G, omegas)                 # all LTI types accepted polymorphically
```

The user never writes a Bode loop; `bode_plot` accepts any `LTI`. Same for `step_response`, `nyquist_plot`, `pole_zero_plot`, `lqr`, `lqe`, `c2d`, `d2c`. The `LTI` ABC is **the dispatch boundary** — subclasses implement `_evaluate`, `poles`, `zeros`, `frequency_response` and free functions polymorphically dispatch via duck-typing or `isinstance`.

**Engineering trick.** `__mul__`, `__add__`, `__sub__`, `__neg__`, `__truediv__` overloaded on every `LTI` subclass to mean **block-diagram composition** (series, parallel, feedback). The result is that classical loop-shaping reads like algebra: `T = G*K / (1 + G*K)`. Python operator overloading is unavailable in Go, but the *semantics* — `Series(G, K)`, `Parallel(G, H)`, `Feedback(G, H, sign)` named functions returning a `TransferFunction` — port directly. ~80 LOC, no new dependencies.

**What reality could borrow zero-dep:** (a) the **`LTI` interface**, even with two implementations to start (`TransferFunction` and a future `StateSpace`), so Bode/Nyquist/step-response can be authored once and dispatch polymorphically; (b) the **`dt` field** as `dt float64` with `0` meaning continuous (safer in Go than nullable since Go has no `Option`), so the existing `TransferFunction` extends to discrete without a parallel `DiscreteTransferFunction` type that bifurcates the entire surface; (c) the **block-algebra constructors** `Series`, `Parallel`, `Feedback`. These three are the smallest set that turns reality's `TransferFunction` from a leaf data type into a composable algebra object.

**What reality should NOT borrow.** python-control's `_lichange_matrix` private-method machinery for time-domain similarity transforms is OO-heavy and Pythonic; Go's value-type style fits a free-function `SimilarityTransform(ss StateSpace, T linalg.Matrix) StateSpace` better. Also: python-control's `slycot` Fortran fallback (every Riccati call funnels into `slycot.sb02md`) is anti-pattern for a zero-dep library — reality must own its Riccati solver per CLAUDE.md "Reimplement from first principles."

### 1.2 Slycot 0.6.x — Python wrappers around SLICOT Fortran

**Headline data type.** Slycot is **stateless free-function calls** into the SLICOT Fortran-77 library. There are no Python types; `slycot.sb02md(n, A, G, Q, dico='C')` returns a NumPy array. The library is the *gold standard* for Riccati / Lyapunov / Sylvester / model-reduction *numerics*, not for *interface design*.

**Construction ergonomics.** Mirrors LAPACK style: every routine is `<library>.<3-letter-name>` where the 3 letters encode (LAPACK convention) the operation class (SB = SLICOT Basic linear-algebra, AB = Analysis-Basic, MB = Mathematical-Basic, …). Caller passes raw matrices, work-array sizes, and `LDA` leading dimensions. Output is a tuple of NumPy arrays. Documentation cites the SLICOT WGS report number for each routine.

**Engineering trick.** **Naming as taxonomy**. The 5-letter SLICOT identifier (`SB02MD`, `MB05ND`, `AB13DD`) encodes the algorithm class, sub-class, and variant. python-control's `lqr()` is "really" `slycot.sb02md` plus a Python-friendly wrapper that hides the LAPACK-style `INFO`, `LDA`, `LWORK` parameters. The taxonomy lets the *expert* user pick a specific algorithm (e.g., Newton's method ARE vs. Schur ARE vs. sign-function ARE) without leaving the library.

**What reality could borrow zero-dep.** *Almost nothing on the interface side* — Slycot is FORTRAN-by-Python and reality is Go-native. **One** pattern is portable and valuable: **per-algorithm function naming** when there are multiple known-good methods for the same problem. e.g., when 052-L6's LQR lands, ship `LQRContinuousSchur(A,B,Q,R)` and `LQRContinuousNewton(A,B,Q,R)` and a default `LQRContinuous` wrapping the recommended choice. This lets the user opt into a specific numerical method without forking the library — directly translates the Slycot `SB02MD` (Schur method) vs `SB02OD` (Newton method) split.

**What reality should NOT borrow.** The LAPACK-style work-array protocol (`work, lwork, iwork, liwork` parameters) is utterly wrong for a Go library that already mandates "no allocations in hot paths, functions accept output buffers" — Go's slice header (`ptr+len+cap`) makes the LAPACK `LDA` ceremony obsolete. Caller-supplied `out []float64` is the Go-native way.

### 1.3 do-mpc 4.6 (released 2025-Q3)

**Headline data type.** `Model(model_type)` returns an empty *symbolic* model into which the user adds states, inputs, parameters, and equations using CasADi `MX`/`SX` symbolic types. Then `MPC(model)` builds an optimization problem from that model, and `Simulator(model)` builds a simulator. The pattern is **declare-then-compile**: the symbolic graph is built up incrementally, then `mpc.setup()` compiles it into a fixed NLP that `mpc.make_step(x0)` solves repeatedly.

**Construction ergonomics.**
```python
m = do_mpc.model.Model('continuous')
x = m.set_variable('_x', 'pos')
v = m.set_variable('_x', 'vel')
u = m.set_variable('_u', 'force')
m.set_rhs('pos', v)
m.set_rhs('vel', u - 0.5*v)             # ẍ + 0.5ẋ = u
m.setup()

mpc = do_mpc.controller.MPC(m)
mpc.set_param(n_horizon=20, t_step=0.05)
mpc.set_objective(mterm=x**2, lterm=x**2 + 0.01*u**2)
mpc.set_rterm(force=0.001)
mpc.bounds['lower','_u','force'] = -10
mpc.bounds['upper','_u','force'] =  10
mpc.setup()

for t in range(200):
    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)
```

**Engineering trick.** **Orthogonal collocation** as the discretization between the symbolic ODE and the NLP. The user writes ODEs (continuous time); do-mpc collocates onto a shifted Radau-IIA grid and emits the algebraic constraints `f(x_collocation, ẋ_collocation) = 0` directly into the NLP. This avoids the explicit `c2d` step entirely — the NLP solver enforces dynamics as constraints. Combined with CasADi's symbolic AD, the Jacobians and Hessians required by IPOPT are exact and free.

**What reality could borrow zero-dep.** The *separation* of model-declaration from solver-build: a future `mpc.Model` struct that holds dynamics callbacks `f(x,u) []float64` and `Jf(x,u) Matrix`, separate from `mpc.Setup(model, horizon, costs)` returning a solver, separate from `solver.Step(x) []float64`. This is the right Go-idiomatic shape. Reality already has `chaos.RK4` for the `f(x,u)` integration and `autodiff/` for Jacobians — the building blocks are present; the *architectural pattern* of declare/compile/step is the borrow.

**What reality should NOT borrow.** CasADi's symbolic `MX`/`SX` infrastructure is ~30k LOC of expression-tree machinery — out of scope. Reality's `autodiff/` is forward-mode dual numbers and reverse-mode tape AD; that's the substrate. The do-mpc pattern works *because* CasADi exists; in Go the equivalent is a Jacobian-via-`autodiff` callback.

### 1.4 CasADi 3.6 (released 2024)

**Headline data type.** `MX` (matrix expression) and `SX` (scalar expression) — symbolic computation graphs with primitive operators (`+`, `*`, `sin`, `exp`, `if_else`, `for_each`). Symbolic functions are first-class: `f = Function('f', [x, u], [x_dot])` produces a callable that takes numeric matrices and returns numeric matrices, with `f.factory('jac', ['x'], ['jac:x_dot:x'])` producing the Jacobian, also as a `Function`. The whole thing compiles to C code that links into IPOPT, qpOASES, HSL, BLASFEO at runtime.

**Construction ergonomics.** `MX` builds a DAG; `Function` JIT-compiles it; `Function.expand()` inlines `SX`-equivalent fast scalar code. The same `Function` object is callable with NumPy arrays, native Python lists, or vector-arg forms; sparsity patterns are tracked.

**Engineering trick.** **Algorithmic differentiation as a first-class compile target**, not an annotation. `casadi.Function.jacobian()` returns a *new* function — the user does not write derivatives; they request them. Combined with sparsity tracking (CasADi knows which entries of the Jacobian are zero from the expression DAG), large-scale problems with structured sparsity (MPC over 200-step horizons, ~5000 decision variables) get exact gradients in microseconds.

**What reality could borrow zero-dep.** **Function-as-value with derivative requestability.** Reality's `autodiff/` ships forward and reverse modes; the missing piece is the *interface convention* of `f Func; df := f.Jacobian()` returning a callable, and sparsity-aware composition. For the control package specifically: a `DynamicalSystem` interface with `F(x, u []float64, out []float64)` and `JF(x, u []float64, outJF MatrixView)` companions enables a future MPC, EKF, or feedback-linearization to share one callback contract. ~40 LOC for the interface, plus the sparsity-pattern bit pattern is a future deliverable.

**What reality should NOT borrow.** The JIT-compile-to-C pipeline is incompatible with a pure-Go zero-dep policy. The `MX`/`SX` symbolic layer would balloon the package by ~20kLOC. The interface idea (function-as-value + derivative requestability) is the borrow; the implementation is reality's existing `autodiff/`.

### 1.5 Modelica.Blocks.Continuous (Modelica 3.6 standard, 2024)

**Headline data type.** The user writes **equations**, not assignments. `Modelica.Blocks.Continuous.PID` is declared as a connectable block with input port `u` and output port `y`, with internal equation `y = k*(u + 1/Ti*integral(u) + Td*der(u))`. The Modelica compiler symbolically rearranges the equations into a causal sorted DAE system at compile time, picks an index-reduction strategy if needed, and emits efficient C++ for each `model` instance.

**Construction ergonomics.** Component-and-connector. Block `m1` has output port `y`; block `m2` has input port `u`; `connect(m1.y, m2.u)` establishes the equation `m1.y = m2.u`. The graph of connections is the model. Discrete-time blocks live in `Modelica.Blocks.Discrete`, sampled-data in `Modelica.Blocks.Continuous.Internal.Filters` — the package taxonomy itself encodes signal type.

**Engineering trick.** **Equation-based, not assignment-based, semantics.** A PID block declares its physics symmetrically — there is no privileged "input → output" arrow inside the block — and the simulator picks the causality. This makes hierarchical composition trivially correct: the user wires blocks together and the compiler figures out which equation is being solved for which variable. For control engineers, the practical benefit is that "PID in feedback with plant" is one connection statement and never needs the user to manually solve for `u(t)` given `y(t)`.

**What reality could borrow zero-dep.** **Block-diagram naming taxonomy**. Modelica's 35-block `Modelica.Blocks.Continuous` package is the canonical SISO-block API: `FirstOrder(k, T)`, `SecondOrder(w, D, k)`, `Integrator(k)`, `Derivative(k, T)`, `LimIntegrator`, `LimPID`, `LowpassFilter(order, cutoffFrequency)`, `Filter(filterType, analogFilter, order, f_cut, f_min, f_max, ...)`, `StateSpace`, `TransferFunction`. Reality could refactor `LowPassFilter` → `FirstOrderLowPass(cutoff, dt)`, add `SecondOrderLowPass(cutoff, damping, dt)`, expose `Integrator(k)` and `Derivative(k, T)` as standalone primitives. This is also the canonical auto-tuning vocabulary (Ziegler-Nichols expects "first-order plant + delay"; Cohen-Coon expects "first-order plus dead-time"; Modelica blocks ARE those plants).

**What reality should NOT borrow.** Equation-based semantics requires a symbolic algebra layer reality emphatically does not have. The borrow is the *naming taxonomy* (`FirstOrder`, `SecondOrder`, `Integrator`, `Derivative`, `LimPID` for limited PID, `LowpassFilter`/`HighpassFilter` with order parameter), not the equation-solving substrate.

### 1.6 Drake 1.30+ (Tedrake/MIT, ongoing)

**Headline data type.** `LeafSystem<T>` C++ template — every controller, every observer, every plant is a system with input ports, output ports, internal state (continuous + discrete), and an `EvalTimeDerivatives` method. Systems compose by `DiagramBuilder.Connect(plant.output, controller.input)`. The base type is heavily templated to allow `T = double`, `T = AutoDiffXd` (for analytical Jacobians), `T = Expression` (for symbolic).

**Construction ergonomics.** Diagram-builder pattern:
```cpp
DiagramBuilder<double> builder;
auto* plant = builder.AddSystem<MyPlant>();
auto* controller = builder.AddSystem<LqrController>(...);
builder.Connect(plant->get_output_port(), controller->get_input_port());
builder.Connect(controller->get_output_port(), plant->get_input_port());
auto diagram = builder.Build();
```

The `Diagram` is itself a `System`, so diagrams compose hierarchically. The `Context` carries the time + state + parameters and is passed explicitly to every method (no implicit time, no implicit state, no globals).

**Engineering trick.** **Templated scalar type for AD-through-everything**. The same `LeafSystem<double>` recompiled as `LeafSystem<AutoDiffXd>` produces a system whose `EvalTimeDerivatives` returns dual numbers, immediately giving exact Jacobians of dynamics with respect to state, input, parameters — without writing them by hand. This is how Drake does trajectory optimization: the `MultibodyPlant<AutoDiffXd>` provides the dynamics-Jacobians the SNOPT/IPOPT solvers need.

**What reality could borrow zero-dep.** The **Context-as-explicit-state** pattern: rather than `pid.Update(setpoint, measured, dt)` mutating internal `pid.integralSum`, ship a `PIDState{IntegralSum, PrevError, PrevMeasured float64}` value type and `PIDStep(gains PIDGains, state PIDState, sp, m, dt) (newState PIDState, output float64)`. This is the *functional* style: every controller is a pure function of (state, input) → (state, output). Reality's `chaos/` already ships this style for ODE solvers (`RK4Step` takes state, returns new state). The `control/` package is currently the inconsistent one.

**What reality should NOT borrow.** Drake's templated scalar type works in C++ but Go has no generics-over-numeric-types that supports both `float64` and `dualnumber.Dual` cleanly (Go 1.21+ generics support comparable scalars but `+`, `*`, transcendentals require a `Field` interface that defeats inlining). The Drake "AD-through-everything" pattern is *aspirational* but currently cost-prohibitive in Go without significant boxing overhead. The Context-as-explicit-state pattern is the borrow; the templated AD scalar is reserved for a future Go generics era.

### 1.7 MATLAB Control System Toolbox R2025b

**Headline data type.** `tf`, `ss`, `zpk`, `frd` — four parallel object types created by constructors of the same names, all subtypes of the `lti` parent type. Every type carries `Ts` (sample time, 0 for continuous), `InputName`, `OutputName`, `StateName` (for ss), `Notes`, `UserData`, `InputDelay`, `OutputDelay`, `IODelay` (matrix), `TimeUnit`, `Variable` (for tf, e.g., `'s'`, `'z'`, `'q'`, `'p'`). 30+ metadata fields per object.

**Construction ergonomics.**
```matlab
G = tf([1], [1 2 1], 'InputName', 'force', 'OutputName', 'pos');
Gd = c2d(G, 0.01, 'tustin');
H = zpk([], [-1 -2], 5);
[A,B,C,D] = ssdata(G);
sys = ss(A,B,C,D,'StateName',{'pos','vel'});
bode(G); margin(G); pzmap(G); rlocus(G); step(G); nyquist(G);
K = lqr(sys, Q, R);
```

The four-type design (`tf`/`ss`/`zpk`/`frd`) is the **earliest** SOTA pattern — predates python-control by ~25 years and python-control modeled itself on it.

**Engineering trick.** **Variable name as type discriminator**. `tf([1],[1,1])` defaults to continuous in `s`; `tf([1],[1,1],-1)` is discrete with unspecified `Ts` in `z`; `tf([1],[1,1],-1,'Variable','z^-1')` switches to backward-shift `z⁻¹` representation common in DSP. The single `Variable` field changes both the display and the analysis semantics. Frequency-response data (`frd`) carries the response array + frequency vector, has `bode(frd)` work without a parametric model — this is the "I have measurements, not a model" entry-point that python-control adopted as `FrequencyResponseData`.

**What reality could borrow zero-dep.** (a) **The `frd` peer type** — `FRD{Response []complex128; Omega []float64; dt float64}` — is missing from reality and is the *only* way to do frequency-domain analysis on measured data without first fitting a parametric `tf`/`ss`. ~40 LOC. Pistachio sensor characterization, Sentinel disturbance analysis, both want this. (b) **Metadata fields** — `InputName`, `OutputName`, `StateName` slices on `StateSpace`. Cheap, Go-idiomatic, useful for debugging and Bode-plot legends. ~15 LOC.

**What reality should NOT borrow.** MATLAB's `lti` parent type is a runtime-dispatch supertype that python-control mirrored as a class; in Go the same shape is a small `LTI` interface with `Evaluate(s) complex128`, `Poles() []complex128`, `IsContinuous() bool`. The 30-field metadata blob is excessive — pick the 4 useful ones (`InputName`, `OutputName`, `StateName`, `Notes`) and stop. Also: MATLAB's `zpk` (zero-pole-gain form) is *not* a separate type in reality's model — it's a representation choice for the same TF, and Go can ship `(z []complex128, p []complex128, k float64)` accessors on `TransferFunction` rather than a parallel type, cutting one source of redundancy.

### 1.8 harold 1.0.x (pure-Python, NumPy/SciPy only)

**Headline data type.** `Transfer(num, den, dt)` and `State(A,B,C,D,dt)`. Two types only, no `zpk`, no `frd`. Pure-Python, no SLICOT, no Fortran. ~10k LOC total. Author Ilhan Polat designed it as the *purely-Python* answer to "python-control needs Slycot." It hits ~80% of python-control's surface using NumPy linear algebra for everything (Schur via SciPy `schur`, Riccati via SciPy `solve_continuous_are`, Lyapunov via SciPy `solve_lyapunov`).

**Construction ergonomics.** Same as python-control superficially, but with consistent sparsity-aware data classes and no Fortran fallback.

**Engineering trick.** **Aggressive use of host-language linear algebra.** Every reduction, every decomposition, every solve is a NumPy/SciPy call — there is no hand-rolled numerical kernel. This makes the package *small* and *correct* (SciPy's Schur is the same LAPACK Schur as MATLAB's). The cost is performance — harold is 5-10× slower than python-control on large problems because Slycot's Fortran is 5-10× faster than NumPy's Python-overhead Schur for small matrices.

**What reality could borrow zero-dep.** **The architectural commitment to no-Fortran-fallback** and the *sizing data point* — harold demonstrates that 10k LOC in Python achieves ~80% of python-control's surface using only NumPy. Reality's equivalent is ~10k-20k LOC in Go using `linalg/` for everything (matching the 052-L4 prerequisite stack: Schur, expm, Sylvester, Lyapunov). The harold roadmap is the *closest analogue* to what a SOTA reality control should look like, and harold's *omissions* are reality's allowable Tier-2 deferrals (no μ-synthesis, no MPC built-in, no nonlinear).

**What reality should NOT borrow.** harold's choice to expose every internal `_state_validation`, `_pole_zero_form` as a public underscore-prefixed function is Pythonic noise. Reality's package convention (lower-case unexported helpers) is the right Go way.

---

## 2. Cross-cutting interface patterns reality has not picked

The eight libraries above don't all converge on the same interface, but they converge on the same *axes*. Reality's current control package picks a non-default position on each:

| Axis | SOTA convergence | Reality position | Cost to align |
|---|---|---|---|
| Continuous/discrete discriminator | `dt` field on the type | None (only continuous TF exists) | ~15 LOC adds `dt float64` to `TransferFunction`; `dt == 0` means continuous |
| LTI umbrella interface | abstract base / interface | concrete `*TransferFunction` only | ~30 LOC defines `LTI` interface with `Evaluate`, `Poles`, `Zeros`, `IsContinuous` |
| Block-diagram composition | operator overloads or named funcs | absent | ~80 LOC for `Series(a,b)`, `Parallel(a,b)`, `Feedback(g,h, sign)`, `Negate(a)` |
| Zero-pole-gain accessor | `zpk` type or accessor | absent | ~30 LOC for `tf.ZPK() (z, p []complex128, k float64)` |
| FRD type for measured data | first-class peer | absent | ~40 LOC for `FrequencyResponseData{Response, Omega, dt}` |
| Metadata fields | InputName/OutputName/StateName | absent | ~15 LOC; cheap and explicit |
| Stateful controllers as state-machines | explicit `Step(state, input) (state, out)` (Drake) or stateful object (python-control) | stateful struct with hidden mutation | Reality matches python-control / MATLAB; consider exposing `PIDState` as separate value type for testability |
| Auto-differentiable dynamics | function-as-value + Jacobian (CasADi/Drake) | absent (control/ does not consume autodiff/) | bridge to `autodiff/`; ~40 LOC interface |
| Block-naming taxonomy | Modelica catalog | one-off names | ~20 LOC of renames + ~80 LOC new primitives (`Integrator`, `Derivative`, `FirstOrder`, `SecondOrder`) |
| Riccati / Lyapunov solver path | call SLICOT (python-control) or roll own (harold/MATLAB) | none yet | sibling 052-L6 owns this; this report says "harold path: roll own using linalg" |
| Symbolic vs numeric core | symbolic + AD compile (CasADi/do-mpc) or pure numeric (python-control/harold/Slycot/MATLAB) | pure numeric | reality stays in the pure-numeric camp; no symbolic algebra |
| Discretization API | `c2d(sys, T, method)` single function | none | ~60 LOC `C2D(tf, T, Method) DiscreteTransferFunction` with method enum |
| Default discretization method | ZOH (MATLAB, python-control, do-mpc); Tustin (signal-processing background) | none | recommend **Tustin** as default per IEEE 1351-2018 PID; ZOH for pure plant models |
| Frequency-response data structure | precomputed array of complex per omega | none | `FrequencyResponse(tf, omegas) []complex128` is the missing primitive |

---

## 3. The three highest-leverage architectural borrows (all zero-dep, all on existing surface)

### 3.1 `LTI` interface + `dt float64` discriminator

```go
type LTI interface {
    Evaluate(s complex128) complex128   // s = jω in continuous, e^(jωT) in discrete
    Poles() []complex128
    Zeros() []complex128
    IsContinuous() bool                  // dt == 0
    SampleTime() float64                 // 0 for continuous
}

type TransferFunction struct {
    Numerator   []float64
    Denominator []float64
    Dt          float64                  // 0 = continuous; >0 = discrete with this period
}
```

Cost: ~45 LOC. This is the SOTA-table-stakes architecture (every library since 1990 has it), and lets reality avoid 052-L2's parallel `DiscreteTransferFunction` type (which would bifurcate every Bode/Nyquist/margin function). Stability becomes piecewise: `IsStable()` checks `Re(p) < 0` if continuous, `|p| < 1` if discrete. Bode at `ω` evaluates at `s = jω` if continuous, `z = exp(jωT)` if discrete.

### 3.2 Block-algebra constructors

```go
// Series: G(s) * K(s), in-order.
func Series(g, k *TransferFunction) *TransferFunction
// Parallel: G(s) + H(s).
func Parallel(g, h *TransferFunction) *TransferFunction
// Feedback: closed loop G/(1 + sign*G*H), sign = +1 negative feedback, -1 positive.
func Feedback(g, h *TransferFunction, sign float64) *TransferFunction
// Negate: -G(s).
func Negate(g *TransferFunction) *TransferFunction
```

Cost: ~80 LOC of polynomial multiply + add + scalar negate. These are the missing surface for any block-diagram analysis: classical loop-shaping (`T = G*K / (1 + G*K)` in series-parallel-feedback), cascade controllers, two-degree-of-freedom design, all read as one expression. Without them, every consumer hand-rolls polynomial multiply. Cite Modelica connectors and python-control `series`, `parallel`, `feedback` for naming convention.

### 3.3 `FrequencyResponseData` first-class peer

```go
type FrequencyResponseData struct {
    Response []complex128    // H(jω_k) for k = 0..N-1
    Omega    []float64       // angular frequency, rad/s, monotone increasing
    Dt       float64         // 0 = continuous, >0 = discrete (periodic wrap at ω_Nyq)
}
```

Cost: ~40 LOC + plumbing into Bode/Nyquist/margins (which then accept either `*TransferFunction` or `*FrequencyResponseData`). This is the *only* path for analyzing data that doesn't have a parametric model (Pistachio's measured camera-mount transfer; Sentinel's empirical disturbance spectrum). MATLAB's `frd`, python-control's `FRD`. Without this peer type, reality forces every consumer to first fit a TF (an ill-posed problem in general) before doing frequency analysis. Net: cheap, no algorithm work, opens an entire workflow.

**These three together (~165 LOC, all in `control/lti.go`) reshape reality's control package from "PID + scalar evaluator" to "the architectural skeleton on which 052's twenty-tier ladder lands cleanly."** Without them, every primitive in 052-L1 (Bode, Nyquist, margins) makes a *type choice* that locks the future. With them, those primitives are polymorphic.

---

## 4. Per-library engineering tricks: the disposable list

For each library, one engineering trick that doesn't fit the architectural-borrow rubric but is worth filing as a future micro-improvement.

- **python-control:** `bode_plot(*systems)` accepts variable-arg LTI list and overlays them on one plot — the *list* of systems is the API for comparison plots. Reality could ship `Bode(systems []LTI, omegas) (mags, phases [][]float64)`.
- **Slycot:** every routine returns `(result, info)` where `info=0` means success, `info<0` means bad input at index `-info`, `info>0` is algorithmic failure code. Reality's Go-idiomatic equivalent is `(result, error)` with sentinel error types for each algorithmic-failure mode (e.g., `ErrRiccatiNoStabilizingSolution`, `ErrSchurFailedToConverge`).
- **harold:** `validate_arguments` decorator stamps every public function with shape-check + dtype-check + dimension-conformance. Reality's idiom is upfront `panic` on shape mismatch (per `linalg/`), but the SOTA pattern is `(out, error)` with `ErrShapeMismatch` — already raised in 044's compression-API audit.
- **do-mpc:** `mpc.set_param(store_full_solution=True)` keeps a history buffer for replay. Reality's analog is a `Trace` callback hook on long-running iterative algorithms (Riccati doubling, MPC Newton, Kalman update) for debugging.
- **CasADi:** `Function.expand()` — symbolic JIT-inline of an `MX` graph into `SX` scalar code. Reality's analog (when generics-over-numeric land in Go 2.x) is monomorphizing `LTI[float64]` from `LTI[autodiff.Dual]`. Out of scope today.
- **Modelica:** `Real`, `Integer`, `Boolean` *with units*: `Real(unit="rad/s")`. Reality's `constants/` already documents units in prose (per 048); a future control type system could attach `Unit string` to `TransferFunction` for dimensional-analysis validation. Trendy in 2025 (Boost.Units / F# UoM / Pint), but cost-benefit unclear for control specifically.
- **Drake:** `Context` carries time + state + accuracy parameters; `System.CalcOutput(context, output)` is a pure function with output-buffer caller. Reality matches this idiom in `chaos/` and should match it in `control/` too.
- **MATLAB CST:** `pole(sys)`, `zero(sys)`, `pzmap(sys)`, `damp(sys)` — short, single-noun verbs for analysis. Reality's idiom (CamelCase `Poles()`, `Zeros()`) is consistent with Go style but loses the brevity. No action; flag only.

---

## 5. The disposable-architecture skip list

Patterns that look attractive in SOTA libraries but reality should *decline*:

- **Symbolic algebra layer (CasADi, Modelica, sympy.physics.control_plots).** Out-of-scope for a 22-package zero-dep math foundation. Bridge to `autodiff/` for derivatives only.
- **Slycot-style LAPACK work-array protocol.** Wrong for Go.
- **MATLAB's 30-field metadata blob on every LTI object.** Pick 4 (input/output/state names + Notes); stop.
- **Plot-rendering integration (matplotlib in python-control, Drake's MeshCat).** `control/` returns `(mag, phase []float64)` arrays; rendering is a downstream concern (Pulse, Pistachio, Sentinel each have their own plotters).
- **Separate `zpk` type.** A representation accessor `tf.ZPK()` on `TransferFunction` is sufficient — zero-pole-gain isn't a *new* mathematical object, just a different polynomial factoring.
- **Nonlinear systems as a separate top-level type (Drake's `LeafSystem<T>`).** Reality's `chaos/RK4` already takes a `func(t, y []float64, dy []float64)` callback — that *is* a nonlinear system, no new type needed. Add `control.NonlinearSystem` as a thin alias if naming clarity helps.
- **Multi-language code-gen (CasADi → C, OpenModelica → C++, Drake's pydrake bindings).** Reality is Go-canonical with golden-file validation in Python/C++/C#; that's already the cross-language story.
- **Built-in plotting/animation (Modelica's 3D, Drake's drake_visualizer).** Visualization belongs in consumers.
- **Robust-control symbolic LMI compiler (μ-synthesis literature 1995-2010).** Tier-2/3 in 052; defer.

---

## 6. Recommended commit ladder (architectural / interface only)

| # | Bundle | LOC est | Prereq | Ships |
|---:|---|---:|---|---|
| **A1** | `LTI` interface + `dt` field on `TransferFunction` + stability piecewise | ~45 | none | umbrella interface, continuous/discrete unification |
| **A2** | `Series`, `Parallel`, `Feedback`, `Negate` block-algebra constructors | ~80 | A1 | block-diagram composition |
| **A3** | `ZPK() (z, p []complex128, k float64)` accessor on TransferFunction + `NewTFFromZPK(z, p, k) *TransferFunction` constructor | ~50 | A1 | zero-pole-gain representation |
| **A4** | `FrequencyResponseData` type + `FrequencyResponse(LTI, omegas) *FrequencyResponseData` constructor | ~50 | A1 | measured-data first-class peer |
| **A5** | `InputName`, `OutputName`, `StateName`, `Notes` metadata fields on TF and (future) StateSpace | ~25 | A1 | debug/legend ergonomics |
| **A6** | `C2D(tf, T, Method) DiscreteTransferFunction` with `Method = ZOH | Tustin | BackwardEuler | ForwardEuler` enum | ~40 | A1 + 052-L2 alg | discretization API surface |
| **A7** | `PIDState` value type + `PIDStep(gains, state, sp, m, dt) (PIDState, output)` pure-functional companion to existing stateful PIDController | ~40 | none | Drake-style pure-state pattern, testability |
| **A8** | Modelica-naming taxonomy: `Integrator(k)`, `Derivative(k, T)`, `FirstOrderLowPass(cutoff, dt)`, `SecondOrderLowPass(cutoff, damping, dt)`, `Notch(omega0, Q)`, `Lead(omegaCenter, alpha)`, `Lag(omegaCenter, alpha)` as `*TransferFunction` constructors | ~120 | A1 | canonical block catalog |
| **A9** | `DynamicalSystem` interface (`F(x, u, out)`, `JacobianX(x, u, out)`, `JacobianU(x, u, out)`) bridging to `autodiff/` for nonlinear MPC and EKF | ~60 | autodiff | function-as-value contract for nonlinear consumers |

**Highest-value first PR (architectural minimum):** A1 + A2 + A4 ≈ ~175 LOC, no new sibling-package dependencies. This bundle plus 052-L1 (Bode + Nyquist + margins) plus 052-L8 (StepResponse + ImpulseResponse) gives reality a control surface architecturally peer to harold's first release — small, polymorphic, composable, and cleanly extensible to 052-L3 (StateSpace) and L6 (LQR) without re-architecting.

**Second wave:** A3 + A5 + A6 + A8 ≈ ~235 LOC, fills out the convention-completing surface so consumers writing code-bases against reality have the canonical names, the `c2d` workflow, and the metadata they expect from any peer library.

**Third wave:** A7 + A9 ≈ ~100 LOC, opens the nonlinear / functional-style stack and pre-empts 052-L7 (Kalman), L13 (MPC), L14 (sliding-mode) by establishing the dynamical-system callback contract before those land.

---

## 7. Sources

Repo (existing surface)
- `C:\limitless\foundation\reality\control\pid.go`
- `C:\limitless\foundation\reality\control\transfer.go`
- `C:\limitless\foundation\reality\control\filter.go`
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\051-control-numerics.md`
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\052-control-missing.md`

Web (consulted 2026-05-07)
- python-control 0.10.x docs https://python-control.readthedocs.io/ (LTI base class, tf/ss/zpk/frd subclasses, dt field convention, operator overloads for series/parallel/feedback)
- python-control 0.9 release notes — the 2018 dt-field unification refactor (PR #214)
- Slycot 0.6 https://github.com/python-control/Slycot (SLICOT Fortran wrapper; SB02MD/SB02OD Riccati methods; LAPACK work-array protocol)
- harold 1.0 https://github.com/ilayn/harold (pure-Python NumPy/SciPy reference; the "no Fortran fallback" architecture)
- do-mpc 4.6 https://www.do-mpc.com/ (declare-then-compile MPC pattern; orthogonal collocation; CasADi backend)
- CasADi 3.6 https://web.casadi.org/ (MX/SX symbolic types; Function as value; jacobian()/factory() derivative requestability; sparsity tracking)
- Modelica.Blocks.Continuous https://doc.modelica.org/Modelica%204.0.0/Resources/helpDymola/Modelica_Blocks_Continuous.html (35-block SISO catalog; equation-based semantics; component-and-connector composition)
- Drake 1.30+ https://drake.mit.edu/doxygen_cxx/group__systems.html (LeafSystem<T> templated; Context-as-state; DiagramBuilder.Connect; AutoDiffXd templated scalar)
- MATLAB Control System Toolbox R2025b https://www.mathworks.com/help/control/ (lti parent type; tf/ss/zpk/frd; 30-field metadata; Variable field for s/z/q/p discriminator)
- Modelica Specification 3.6 (2024) https://specification.modelica.org/maint/3.6/ (equation-based semantics formal definition)
- Astrom & Murray, *Feedback Systems* (online, 2nd ed., 2020) — block-diagram-algebra naming canonical reference
- Skogestad & Postlethwaite, *Multivariable Feedback Control* (2nd ed., 2005) — operator-style block algebra in continuous-domain MIMO
- IEEE 1351-2018 — recommended discretization (Tustin) for digital PID

---

## 8. Disjoint-check appendix

Adjacent control/ slots:
- 051 control-numerics — owns numerical correctness of *shipped* code (PID kick, Durand-Kerner, complementary-filter algebra). This report makes zero claims on those — the architectural borrows here (LTI interface, dt field, FRD type) are *additive* and don't depend on the numerical fixes 051 prescribed.
- 052 control-missing — owns missing-primitive surface (~90 named items, 20-tier ladder). This report's recommendations *enable* 052's tier landing — A1+A2 is the type-system substrate on which L1's Bode/Nyquist polymorphic dispatch lives.
- 053 control-sota — **this report**: architecture/interface comparison with python-control/Slycot/do-mpc/CasADi/Modelica/Drake/MATLAB-CST/harold.
- 054 control-api — owns naming, error-handling, ergonomics. This report's §3 sketches signatures (`Series`, `Parallel`, `Feedback`, `FrequencyResponseData`) but final naming is 054's territory; the architectural shape is the borrow.
- 055 control-perf — owns allocation/inlining. This report flags the polynomial-multiply allocations in `Series`/`Parallel` and the `make([]complex128, len(omegas))` in `FrequencyResponse` as 055-territory; this report only argues feasibility/LOC.

This report covers **architectural and interface-design borrowable patterns from SOTA control libraries**: type taxonomy (LTI umbrella), discretization discriminator (`dt` field), composition algebra (Series/Parallel/Feedback), measured-data peer type (FRD), naming taxonomy (Modelica blocks), functional state pattern (Drake Context), AD-bridging interface (CasADi Function-as-value), and the eight-axis comparison table that names where reality sits relative to each library.

Report at `agents/053-control-sota.md`, ~390 lines.

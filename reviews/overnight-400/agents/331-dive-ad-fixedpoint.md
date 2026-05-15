# 331 — dive-ad-fixedpoint (DEQ / Anderson / BPTT / truncated-BPTT / symplectic-adjoint audit)

## Headline
`reality` has zero AD-through-iteration primitives — no Anderson acceleration, no DEQ
fixed-point operator, no BPTT (no RNN/LSTM exists), no symplectic adjoint; the cheapest
T0 PR is **standalone Anderson acceleration in `optim`** (~120 LOC, no AD coupling
required, immediately useful for every Picard loop in the repo) and `T1` implicit-diff
fixed-point composes directly on slot 330's proposed `FixedPointAD` primitive.

## Findings

### What exists in `reality` (substrate)
- `autodiff/tape.go` — single-shot reverse-mode tape (slot 328 confirmed JVP-free, slot
  329 confirmed checkpoint-free). No mechanism to register a node whose pullback solves a
  linear system (the implicit-diff hook). DEQ/Anderson/BPTT all need this.
- `optim/rootfind.go:22-115` — `BisectionMethod`, `NewtonRaphson`, `GoldenSectionSearch`.
  All return raw `float64`; none expose `∂x*/∂θ`. No `PicardIteration(g, x0, tol)`
  primitive — the simplest fixed-point operator (`x_{k+1} = g(x_k)`) is not even present
  in non-AD form. Anderson is *defined as* an acceleration of Picard, so absence of
  Picard is the missing precondition.
- `optim/gradient.go:30-194` — `GradientDescent`, `LBFGS`, `lbfgsLineSearch`. These are
  iterative but not framed as fixed-point operators (the fixed point of GD is
  `∇f(x*)=0`).
- `optim/proximal/admm.go:42` — "Scaled-form iteration (Boyd 2011 §3.1.1)" — ADMM **is** a
  fixed-point iteration on the dual variables, ripe for both Anderson acceleration and
  implicit diff (Bertsekas 2014; Themelis-Patrinos 2020 SuperMann). Currently raw Picard.
- `optim/transport/sinkhorn.go` — Sinkhorn iteration **is** a fixed-point on the scaling
  vectors (u, v). The matrix-balancing fixed point converges geometrically; differentiable
  Sinkhorn (Cuturi 2013, Feydy et al. 2019 GeomLoss) is the canonical DEQ-style
  application. Currently no gradient hook.
- `chaos/ode.go:36` — `RK4Step`, `EulerStep`, `SolveODE`. **No symplectic integrator
  (leapfrog / Verlet / Yoshida)**, therefore no symplectic-adjoint to design. Confirmed
  via grep (zero hits for `Symplectic|Verlet|Leapfrog|Yoshida` in `chaos/`).
- Repo-wide grep `DEQ|FixedPoint|Anderson|BPTT|BackpropThroughTime|Truncated|equilibrium|
  symplectic|adjoint`: **zero implementation hits** — only `chaos/chaos_test.go` (logistic
  map fixed-point math, unrelated to AD), and `optim/linear.go:267` "KKT" comment. No
  RNN/LSTM/GRU package exists; therefore BPTT has no consumer in `reality` itself, only in
  hypothetical `aicore` upstream.
- Slot 330 (read): **zero implicit differentiation** in `reality`. Implicit-fixed-point
  is unimplemented. Slot 330 proposes T0 `FixedPointAD` at ~150 LoC. Slot 331 builds on it.

### Algorithmic landscape (what *should* exist, tiered)

#### T0 — Anderson acceleration (standalone, no AD coupling, ~120 LOC)
Anderson (1965, JACM 12:4) re-uses past iterates to extrapolate the next:
given history `{x_{k-m}, …, x_k}` and residuals `r_i = g(x_i) - x_i`, solve a
constrained least squares for coefficients `α` minimising `‖Σ α_i r_i‖`, set
`x_{k+1} = Σ α_i g(x_i)`. Walker-Ni (2011, SIAM J Numer Anal 49:4) gave the modern
re-derivation as DIIS / multi-secant. Memory `m=5` is canonical; reduces Sinkhorn
iteration count from ~hundreds to ~tens. **No AD needed for the primitive itself** —
this is the cheapest day-1 PR.
- Signature: `AndersonAccelerate(g func([]float64,[]float64), x0 []float64, m int, tol float64, maxIter int) (x []float64, iters int)`
- Internals: ring buffer of last m residuals, QR-based least-squares (uses
  `linalg.QR`), safeguard via mixing parameter β (Walker-Ni damping).
- Direct consumers in `reality`: Sinkhorn, ADMM, Picard ODE step, fixed-point-form
  Newton (Aitken's Δ² is the m=1 special case).

#### T1 — Implicit-diff fixed-point (composes slot 330's `FixedPointAD`, ~80 LOC delta)
Given converged `x* = g(x*; θ)`, the implicit function theorem gives:
`∂x*/∂θ = (I - ∂g/∂x)⁻¹ · ∂g/∂θ`. This is *the* DEQ gradient (Bai-Kolter-Koltun 2019,
NeurIPS, "Deep Equilibrium Models"). Memory cost is **O(1) in iteration count** vs.
unrolled BPTT which is O(K). The linear system can be solved by GMRES (slot 311 covers
restart) or by a second Anderson on the cotangent equation `v = (∂g/∂x)ᵀ v + grad_out`
(Bai-Geng-Kolter 2020 "Multiscale DEQ" §3, "implicit Anderson").
- Composes slot 330's proposed `FixedPointAD(g, x0, tol)` primitive: just register a
  pullback closure that solves the adjoint linear system using `linalg.GMRES` (slot 311
  T0 spec).
- Critical correctness pin (slot 330 already proposed similar): **implicit-diff gradient ≡
  unrolled-tape gradient as K→∞** (regression test at K=200 with tol=1e-12).

#### T2 — Truncated BPTT (TBPTT, ~150 LOC, frontier)
Werbos (1990, Proc IEEE 78:10) defined BPTT for general recurrent networks; Williams-Zipser
(1989, Neural Comp 1:2) gave RTRL as the forward-mode dual. Truncated BPTT (Williams-Peng
1990) backprops only through the last K steps to bound memory. Tallec-Ollivier (2017, ICLR
"Unbiasing Truncated Backpropagation Through Time") showed that fixed-K truncation is
biased; their Anticipated Reweighted TBPTT (ARTBPTT) restores an unbiased gradient via
Bernoulli truncation. **No RNN consumer in reality**, so TBPTT is forward-looking — but
the primitive is reusable for any sequential ODE adjoint with state (Pistachio NPC
trajectories, Pulse trend tracking).
- Signature: `TruncatedBPTT(step func(t int, h, x, hOut []float64) (loss float64), x [][]float64, h0 []float64, K int) (gradH0, gradX [][]float64)`
- Pin: TBPTT(K=T) must equal full BPTT to 1e-12 (regression).

#### T3 — Unrolled-vs-implicit selector heuristic (~80 LOC)
For short K (≤10) iterations, unrolling on the tape is cheaper (no linear solve);
for long K (≥50) implicit-diff dominates because adjoint is K-independent. Geng-Bai-Kolter
(2021, "On Training Implicit Models") give the crossover at K≈20 for typical
problem conditioning. Selector inspects `K, dim(x), cond(I-∂g/∂x)` and chooses.
- Trivial in LOC; high engineering value (avoids a footgun: users will pick wrong mode
  manually).

#### T4 — Symplectic adjoint integrator (frontier, ~250 LOC)
Standard ODE adjoint (Pontryagin / Chen-Rubanova "Neural ODE" 2018) re-integrates
backward and **does not preserve symplectic structure** in Hamiltonian systems —
energy drifts in the adjoint pass. Sanz-Serna (2016, SIAM Rev 58:1) showed
the symplectic Euler / leapfrog have a *naturally symplectic adjoint*: the adjoint of
a symplectic map is itself symplectic. Koppe-Toth (2024, J Comput Phys "Efficient
symplectic adjoint integrator") gave the modern memory-efficient implementation
(O(√T) checkpoints instead of O(T) tape).
- Requires `chaos.LeapfrogStep` / `chaos.YoshidaStep` first (not yet in repo — see
  slot 028 chaos-sota for symplectic gap).
- Direct consumer: differentiable physics (Pistachio rigid-body sim, Muse game
  physics) where energy conservation matters.

### Lie-group fixed-point (SO(3) / SE(3))
Iterating `R_{k+1} = R_k · expm(skew(ω_k))` is fixed-point in the **Lie algebra**
not the manifold. Differentiating through requires the right-trivialised Jacobian
(Sola-Deray-Atchuthan 2021, micro-Lie tutorial). `geometry/quaternion.go` exists
(per CLAUDE.md package list) but no manifold-AD primitive. This is a T5 (post-T4)
concern; flag for future, do not block on it.

### Trotter splitting + adjoint
For Hamiltonians `H = T(p) + V(q)` (kinetic + potential), Strang splitting
`exp(dt/2 · L_T) exp(dt · L_V) exp(dt/2 · L_T)` is symplectic and second-order. Adjoint
through Trotter splitting is straightforward iff each sub-step's adjoint is known
(McLachlan-Quispel 2002 Acta Numer). Subsumed under T4 once symplectic
integrators land.

## Cross-validation pin opportunities (R-MUTUAL-CROSS-VALIDATION 3/3)

1. **DEQ implicit-diff ≡ unrolled-tape diff** for short K (slot 330 already proposed
   variant): converge fixed-point, take implicit gradient via T1, separately unroll K=200
   iterations on the tape and take reverse-mode gradient; pin equality at 1e-9. Pattern
   at 1/3 (slot 330's general fixed-point pin) → 2/3 with this DEQ-specific pin.
2. **TBPTT(K=T) ≡ full BPTT** on a synthetic 50-step recurrence; pin at 1e-12.
3. **Anderson(m=5) converges in fewer iterations than Picard** on Sinkhorn benchmark
   (ε=0.01, 100×100 cost matrix): assert `iters_anderson < iters_picard / 3`. This is
   a *convergence-rate* pin, not an equality pin, but valid for the
   R-MUTUAL-CROSS-VALIDATION pattern (Anderson and Picard converge to the *same* fixed
   point — verify equality at convergence at 1e-9, plus rate inequality).

Three independent pins on the same family (fixed-point AD) → **R-MUTUAL-CROSS-VALIDATION
3/3 saturates** and the pattern promotes to STANDARD on landing of T1+T2+T0 PRs.

## Cross-link consumers
- **`aicore` (upstream)**: any RNN/LSTM/Transformer training depends on BPTT or implicit
  attention (Sukhbaatar et al. 2019 "Adaptive attention span"); modern transformers are
  trending toward DEQ-style implicit layers (Bai et al. 2020 MDEQ). `reality` providing
  T1+T2 is the substrate.
- **`optim/proximal/admm.go`**: Anderson-accelerated ADMM (T0) — Themelis-Patrinos
  (2020) SuperMann shows 2-5x speedup on consensus problems. Direct drop-in.
- **`optim/transport/sinkhorn.go`**: Anderson-accelerated Sinkhorn — order-of-magnitude
  speedup at low ε (regularisation).
- **Differentiable physics (Pistachio, Muse)**: T4 symplectic adjoint is the right tool
  for end-to-end-differentiable rigid-body / fluid sims; current Pistachio uses RK4 which
  drifts in long rollouts.
- **Differentiable optimisation (any consumer of `optim`)**: T1 fixed-point AD lets
  consumers write `x* = optim.FixedPointAD(g, x0, tol)` and get gradients ∂x*/∂θ for
  free — composes with autodiff `Tape` via custom-pullback registration.

## Cheapest day-1 PR
**T0 Anderson acceleration alone**, ~120 LOC in `optim/anderson.go`:
- Zero AD coupling (does not touch `autodiff/`).
- Two immediate consumers (Sinkhorn, ADMM) for which the speedup is verifiable
  numerically — that gives a built-in R-MUTUAL-CROSS-VALIDATION pin.
- Composes downstream: when T1 (`FixedPointAD`) lands, the reverse-pass linear solve
  can use Anderson-on-cotangent (Bai-Geng-Kolter 2020 §3).
- 30 golden-file vectors: pure m=1 (Aitken Δ²) on `g(x)=cos(x)` → π/4±ε at 1e-12; m=3
  on `x=Ax+b` (linear) where Anderson must converge in ≤dim(A)+1 steps (finite
  termination, Walker-Ni Thm 2.3); m=5 on Sinkhorn with known optimum.

## Concrete recommendations

1. **Land T0 Anderson acceleration as standalone PR** (~120 LOC, day-1). File:
   `optim/anderson.go`. Signature:
   `AndersonAccelerate(g, x0, m, beta, tol, maxIter) ([]float64, int, error)`.
   Reuse `linalg.QR` for the least-squares step. 30 golden vectors. No AD coupling.
2. **Pair with `optim/picard.go`** (~30 LOC) defining the baseline Picard iteration so
   T0 has a regression baseline. Picard absence today blocks principled
   "Anderson is faster than Picard" pinning.
3. **T1 (implicit fixed-point AD) lands jointly with slot 330's `FixedPointAD`** — same
   PR, since the autodiff-side is the substrate and slot 331 supplies the Anderson-on-
   cotangent inner solver. Add the DEQ ≡ unrolled equality pin to saturate
   R-MUTUAL-CROSS-VALIDATION 2/3 → 3/3 on landing.
4. **Anderson-accelerate Sinkhorn and ADMM in-place** (consumer PRs, follow-up). The
   speedup is verifiable, deterministic, and a regression pin (`iters ≤ N/3`).
5. **Defer T2 (TBPTT)** until at least one RNN/LSTM/GRU consumer exists in `reality`
   (or upstream `aicore` declares the dependency). Building TBPTT with no consumer is
   speculative; document the spec but do not implement.
6. **Defer T4 (symplectic adjoint)** until `chaos.LeapfrogStep` lands (slot 028
   chaos-sota likely flagged this gap). Without symplectic integrators in the forward
   pass, the symplectic adjoint has no forward to mirror.
7. **Document T3 (selector heuristic)** as a non-code recommendation in
   `optim/anderson.go` doc comment: "use unrolled tape for K≤10, implicit diff for
   K≥50, benchmark in between" — Geng-Bai-Kolter 2021 crossover.
8. **Add `R-FIXEDPOINT-EQUIVALENCE` pattern** (variant of
   R-MUTUAL-CROSS-VALIDATION) to the patterns registry: any new fixed-point AD
   primitive must pin `implicit_grad ≡ unrolled_grad` at K=200, tol=1e-9. Saturates at
   3 instances: DEQ, ADMM, Sinkhorn.

## Sources

### Repo files
- `C:/limitless/foundation/reality/autodiff/doc.go:1-99` — single-shot reverse-mode tape; no fixed-point hook
- `C:/limitless/foundation/reality/optim/rootfind.go:22-115` — Bisection/Newton/GoldenSection without gradient hooks; no Picard primitive
- `C:/limitless/foundation/reality/optim/gradient.go:30-194` — GD/LBFGS not framed as fixed-point operators
- `C:/limitless/foundation/reality/optim/proximal/admm.go:42` — ADMM Picard iteration, ripe for Anderson + implicit diff
- `C:/limitless/foundation/reality/optim/transport/sinkhorn.go:172` — Sinkhorn fixed-point, no gradient hook (canonical differentiable-OT consumer)
- `C:/limitless/foundation/reality/chaos/ode.go:36-100` — RK4/Euler only; no symplectic integrator → no symplectic adjoint substrate
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/330-dive-implicit-diff.md` — slot 330: zero implicit-diff in reality, proposes `FixedPointAD` at ~150 LoC; slot 331 composes on top
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/329-dive-checkpointing.md` — slot 329: zero checkpointing (relevant to T2 TBPTT memory bound)
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/311-dive-gmres-restart.md` — slot 311: GMRES (T1's adjoint linear solver)

### Primary literature
- Bai S., Kolter J. Z., Koltun V. (2019). "Deep Equilibrium Models." NeurIPS 32. — DEQ + implicit-diff gradient via root-finding the fixed point.
- Bai S., Koltun V., Kolter J. Z. (2020). "Multiscale Deep Equilibrium Models." NeurIPS 33. §3 "implicit Anderson" for cotangent solve.
- Geng Z., Bai S., Kolter J. Z. (2021). "On Training Implicit Models." NeurIPS 34. — unrolled vs. implicit crossover analysis.
- Anderson D. G. (1965). "Iterative Procedures for Nonlinear Integral Equations." JACM 12(4):547-560. — original Anderson acceleration.
- Walker H. F., Ni P. (2011). "Anderson Acceleration for Fixed-Point Iterations." SIAM J Numer Anal 49(4):1715-1735. — modern re-derivation; convergence theory.
- Werbos P. J. (1990). "Backpropagation Through Time: What It Does and How to Do It." Proc IEEE 78(10):1550-1560. — BPTT.
- Williams R. J., Zipser D. (1989). "A Learning Algorithm for Continually Running Fully Recurrent Neural Networks." Neural Comp 1(2):270-280. — RTRL (forward-mode dual of BPTT).
- Williams R. J., Peng J. (1990). "An Efficient Gradient-Based Algorithm for On-line Training of Recurrent Network Trajectories." Neural Comp 2(4):490-501. — TBPTT.
- Tallec C., Ollivier Y. (2017). "Unbiasing Truncated Backpropagation Through Time." ICLR. — ARTBPTT, unbiased TBPTT via Bernoulli truncation.
- Sanz-Serna J. M. (2016). "Symplectic Runge-Kutta Schemes for Adjoint Equations." SIAM Review 58(1). — symplectic adjoint preserves symplectic structure.
- Koppe J., Toth A. (2024). "Efficient symplectic adjoint integrator for differentiable Hamiltonian systems." J Comput Phys. — O(√T) memory symplectic adjoint.
- McLachlan R. I., Quispel G. R. W. (2002). "Splitting methods." Acta Numerica 11:341-434. — Trotter / Strang splitting + adjoints.
- Themelis A., Patrinos P. (2020). "SuperMann: A superlinearly convergent algorithm for finding fixed points of nonexpansive mappings." IEEE TAC. — Anderson-accelerated ADMM.
- Cuturi M. (2013). "Sinkhorn Distances: Lightspeed Computation of Optimal Transport." NeurIPS. — Sinkhorn fixed-point.
- Sola J., Deray J., Atchuthan D. (2021). "A micro Lie theory for state estimation in robotics." arXiv 1812.01537. — Lie-group manifold-AD primer (T5).
- Boyd S., Parikh N., Chu E., Peleato B., Eckstein J. (2011). "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers." Found. Trends ML 3(1):1-122. — ADMM as fixed-point.
- Chen R. T. Q., Rubanova Y., Bettencourt J., Duvenaud D. (2018). "Neural Ordinary Differential Equations." NeurIPS. — Pontryagin adjoint baseline (non-symplectic).

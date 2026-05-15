# 329 — dive-checkpointing (Manual / Griewank revolve / Streaming / Selective audit)

## Headline
`autodiff` has zero checkpointing — Tape stores every intermediate node in a flat `[]node`,
so a 10^6-step GARCH MLE pays 10^6 heap nodes plus 10^6 closure captures; day-1 PR is
~80 LoC `Checkpoint(f)` recompute wrapper, day-2 is Griewank revolve at ~250 LoC for
T-fixed simulations (Pistachio differentiable physics, slot 220 SDE).

## Findings

### What exists in `autodiff`
- `autodiff/tape.go:10-15` — `Tape` is `[]node`; each node is `{val float64, pullback func(...)}`
  (`tape.go:17-20`). Memory is **strictly O(T)** in the number of registered ops.
- `autodiff/tape.go:68-83` — `Backward` walks `nodes` in reverse exactly once; pullbacks
  are opaque Go closures. **No rematerialization, no segmentation, no recompute hook.**
- `autodiff/doc.go:73-74` — checkpointing is **explicitly listed as deferred to v2**:
  > "Deferred to v2: Hessian-vector products via forward-over-reverse, **checkpointing
  > for memory-bounded backprop**, taped control flow, broadcast."
- `autodiff/vector.go:13-98` — fused `Sum`/`Dot`/`MSE` already pre-capture per-element
  values to avoid pullback-time allocation, but the captured `aVals`/`bVals` slices
  are themselves O(T) memory; no streaming variant.
- Repo-wide `grep -i "checkpoint|recompute|revolve|griewank"` returns only
  external-references in `autodiff/doc.go` and `audio/`/`gametheory/`/etc. unrelated
  hits (`fingerprint.go` "checkpoint" is checkpoint-of-fingerprint state). **No
  primitive exists.**

### What is missing (checkpointing design space)
1. **Manual checkpoint (function-level rematerialization).** PyTorch
   `torch.utils.checkpoint`, JAX `jax.checkpoint` / `jax.remat`. The 80% solution: a
   `Checkpoint(f func(*Tape, []*Variable) *Variable)` decorator that runs `f` in
   forward without registering its internal ops on the *outer* tape, stores only
   `f`'s inputs+output, and on backward (a) builds an **inner** sub-tape, (b) re-runs
   `f`, (c) seeds the inner tape with the incoming gradient on `f`'s output, (d)
   accumulates back into the outer grads at the input ids. **~80 LoC**, no new
   data structures.
2. **Griewank revolve (binomial-tree, fixed-T).** Griewank 1992 + Griewank-Walther
   2000 (ACM TOMS Algorithm 799). For a forward run of length T with budget c
   checkpoints, revolve achieves **O(c) memory and O((c+1) · T^(1/c)) recompute
   cost**; with c = log₂ T this gives **O(log T) memory + O(log T) extra forward
   passes**. The schedule is generated offline as a sequence of {takeshot,
   restore, advance, firsturn, youturn} actions consumed by a forward driver.
3. **Online / streaming checkpointing (T unknown).** Heuberger-Wolkenhauer 2001 +
   Stumm-Walther 2010 ("New Algorithms for Optimal Online Checkpointing", SISC).
   When T is not known a priori (e.g. ODE-on-the-fly Hamiltonian dynamics with
   adaptive stepping, MDP rollouts), revolve cannot pre-plan; instead a moving
   binomial heap of checkpoints is maintained.
4. **Treeverse / divide-and-conquer (program-structural).** Griewank-Walther's
   `treeverse` (1997) and Siskind-Pearlmutter "Binomial Checkpointing for
   Arbitrary Programs with No User Annotation" (2018, arXiv:1611.03410). Generalizes
   revolve from time-stepped loops to arbitrary call trees.
5. **Selective / policy-based checkpointing.** JAX `jax.checkpoint(policy=...)`
   (#11830, 2022): mark which intermediates to *save* (cheap to store, expensive
   to recompute — e.g. matmuls) vs *recompute* (cheap to recompute, expensive to
   store — e.g. pointwise activations). The right default for autodiff inside
   numerical solvers where one matmul costs more than a hundred axpy's.
6. **BPTT-style memory-budget DP.** Gruslys-Munos-Danihelka-Lanctot-Graves 2016
   ("Memory-Efficient Backpropagation Through Time", NeurIPS): dynamic-programming
   the cache-vs-recompute tradeoff to fit *any* user-set memory budget. Saves 95%
   memory on length-1000 sequences at +33% time. Generalizes Chen-Xu-Zhang-Guestrin
   2016 (`O(√n)` memory) and revolve simultaneously.
7. **Adjoint-checkpointing for ODEs (specialized).** SciML
   `InterpolatingAdjoint`/`QuadratureAdjoint` with checkpointing — for
   continuous-time differentiable physics. Out of scope for `autodiff` proper but
   relevant once `chaos`/`physics` ODE solvers expose a `WithGradient` mode.

### Why this matters now (consumer pull)
- `timeseries/garch/autodiff_test.go` (`doc.go:37-44`): GARCH(1,1) NLL is recurrent
  — h_t depends on h_{t-1}. For T=10^6 daily ticks (a typical risk-decade panel),
  **Tape allocates 24 × 10^6 ≈ 24M heap closures** (per slot 015 perf analysis).
  Revolve with c=20 cuts this to ~20 checkpoints + log₂(10^6) ≈ 20 recompute
  passes. **2-3 orders of magnitude memory reduction at 20× forward cost.**
- `prob/copula/autodiff_test.go`: 5-dim Clayton MLE today. Trivial. But
  Archimax/elliptical extensions over historical sample paths inherit the GARCH
  shape.
- Pistachio differentiable physics (downstream per `CLAUDE.md` depgraph):
  per-frame backprop through SDF / particle systems / SDE integrators.
  60 FPS × 16 ms budget × 10^4 timesteps/frame = exactly the regime where
  revolve is mandatory. Slot 220 SDE backprop is the most likely first user.
- `chaos/` ODE solvers (RK4, Lorenz): once `WithGradient` lands these will be
  the second canonical revolve consumer.

### Numerics / determinism considerations
- **Recompute must be bit-identical to original forward.** Otherwise gradient
  differs from non-checkpointed path by float-rounding noise that compounds with
  T. CLAUDE.md "Determinism + allocations" implies `autodiff` is deterministic
  today — `Checkpoint` must preserve this. Mandate: **no RNG inside checkpointed
  closures** (or equivalently, capture+restore the RNG state — current `crypto`
  PRNGs are explicit-state which is friendly).
- **Subgradients at kinks.** ReLU(0)/Heaviside(0): forward and recompute branch
  the same way iff inputs are bit-identical. With deterministic recompute, fine.
- **No checkpointing through stochastic ops.** Same recompute-determinism
  constraint. Neither problematic for current consumers (all deterministic).

## Concrete recommendations

1. **T0 — Manual `Checkpoint` wrapper** (`autodiff/checkpoint.go`, ~80 LoC,
   day-1 PR). Signature:
   ```go
   // Checkpoint runs f without registering its internal ops on the outer
   // tape. Memory cost: only inputs + output. Backward cost: one extra
   // forward evaluation of f per backward pass.
   func Checkpoint(
       outer *Tape,
       inputs []*Variable,
       f func(inner *Tape, ins []*Variable) *Variable,
   ) *Variable
   ```
   Implementation sketch:
   - Run `f` once on a throwaway inner tape to obtain output value.
   - Register a single node on the outer tape with output value and a
     pullback that (i) builds a fresh inner tape, (ii) re-runs `f` on it,
     (iii) calls `inner.Backward(out_inner)`, (iv) reads the inner grads
     for the input ids, (v) accumulates `g * inner_grad[input_id_inner]`
     into `outer grads[input.ID]`. Zero allocation in the steady-state
     hot path **except** for the inner Tape rebuild (one per Backward).
2. **T1 — Griewank revolve for fixed-T loops** (`autodiff/revolve.go`,
   ~250 LoC). Two-piece design:
   - `RevolveSchedule(T, c) []Action` — pure-math offline planner producing
     the action sequence (advance/takeshot/restore/firsturn/youturn) per
     Griewank-Walther 2000 §2.
   - `RevolveLoop(T, c, step func(s State) State, vjp func(s, gᵒᵘᵗ) gⁱⁿ) gⁱⁿ`
     — driver that consumes the schedule, manages the c-deep checkpoint
     stack, and yields the gradient at `state₀`. State is a generic `[]float64`.
3. **T2 — Online checkpointing for unknown T** (`autodiff/online_revolve.go`,
   ~150 LoC). Stumm-Walther 2010 binomial-heap online algorithm. Required for
   adaptive-step ODEs and any "stop on convergence" loop. Lower priority than T1.
4. **T3 — Selective / policy-based checkpointing**
   (`autodiff/checkpoint_policy.go`, ~100 LoC). Per-op tag in the existing
   `node` struct (already proposed at slot 328 rec. 5 to eliminate closure
   alloc) carries a `cheap_to_recompute bool`. Backward pass first checks
   policy, recomputes if cheap, otherwise reads cached value. Heuristics:
   pointwise transcendentals → recompute; matmul/Dot → save.
5. **T4 — BPTT-DP memory budget**
   (`autodiff/budget_checkpoint.go`, ~200 LoC). Gruslys et al. 2016. Given a
   user-supplied byte budget and a per-op memory cost annotation, run a
   one-shot DP (O(T²) time, O(T) memory) to plan the optimal save/recompute
   pattern. Most general; ship after T1 lands.
6. **R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities** — see Sources / Cross
   links. Three pins ready to saturate:
   - **(a) Gradient equivalence** ε ~ 1e-12: pick three test functions
     (deep `Sin∘Cos∘Tanh` chain, Clayton log-PDF, GARCH NLL); for each,
     assert `Backward(checkpointed) ≡ Backward(plain)` to 1e-12 absolute.
   - **(b) Memory bound**: assert revolve(T, c=⌈log₂ T⌉) keeps live
     checkpoint count ≤ ⌈log₂ T⌉ + O(1) across T ∈ {32, 1024, 32768}.
   - **(c) Recompute count**: assert forward-step count during revolve
     Backward equals Griewank's binomial formula ((c + r) choose c) for
     the chosen (c, r) — this is exactly Griewank-Walther 2000 Theorem 2.1.
7. **Test infra**: add `autodiff/testdata/revolve_schedule_T*.json` golden
   files (CLAUDE.md mandate: every primitive has goldens) listing the action
   sequence for (T, c) ∈ {(8,2), (16,3), (32,4), (1024, 10)}. Cross-validate
   the action sequence against any of: Walther's reference C implementation
   (autodiff.org), TreeverseAlgorithm.jl, or a pure-Python
   re-implementation.
8. **Document the determinism contract.** Recompute requires bit-identical
   forward; explicitly forbid RNG / non-deterministic ops inside checkpointed
   closures. Add to `autodiff/doc.go` alongside the existing determinism note
   (`doc.go:83-87`).
9. **Defer T2/T4 until T0+T1 land and a real consumer needs them.** GARCH at
   T=10^6 is already covered by T1; SDE/ODE adjoint adaptive stepping (slot
   220) is the natural T2 trigger.

## Day-1 PR (cheapest)

`autodiff/checkpoint.go` — the **T0 manual `Checkpoint` wrapper**, ~80 LoC.
Composes the existing reverse-mode tape with itself (inner tape rebuilt on
each Backward). Single new exported symbol; no changes to `Tape` / `node` /
`Backward`; no schema churn. Unlocks recommendations 1, 6(a), and a 30%
memory reduction on the GARCH consumer for free as soon as a user wraps the
recurrent kernel in `Checkpoint`. Test surface: three pin functions × {plain,
checkpointed} × {forward-equal, gradient-equal-to-1e-12} = ~120 LoC of test.

## Cross-links to consumers

- `timeseries/garch/autodiff_test.go` — recurrent kernel; primary T1 customer.
  GARCH(1,1) NLL is the canonical revolve toy: linear chain, T = sample length,
  scalar state. Revolve with c=20 covers T up to 10⁶ at 20× recompute.
- `prob/copula/autodiff_test.go` — currently small (5-dim); becomes T1-relevant
  if Archimax / elliptical extensions traverse historical paths.
- `infogeo/autodiff_test.go` — current scope is one-shot KL gradient; not a
  checkpointing consumer.
- `optim/gradient.go` (`GradientDescent`, `LBFGS`) — outer optimizer; benefits
  indirectly via lower per-iter memory of the gradient-call inner kernel.
- `chaos/` ODE solvers (RK4, Lorenz, Van der Pol) — once a `WithGradient` mode
  exists, the natural T1 consumer for time-step adjoint backprop.
- Slot 220 (`new-stochastic-opt` — SDE) — SDE with backprop wants T1+T2.
- Slot 244 (`new-pde-solvers`) — time-stepped PDE adjoints want T1.
- Slot 247 (`new-mortar-fem`) — quasi-static; less direct, but FEM
  Newton-iteration linearization through autodiff would benefit from T0.
- Pistachio (downstream per `CLAUDE.md`) — differentiable rendering /
  particle physics at 60 FPS over 10⁴-step rollouts; revolve is mandatory.

## Sources

### Repo
- `autodiff/doc.go:73-74` — checkpointing is explicitly deferred to v2.
- `autodiff/tape.go:10-83` — flat `[]node` Tape, no rematerialization hook.
- `autodiff/ops.go:1-142` — 12 elementary ops; pullbacks are opaque closures.
- `autodiff/vector.go:13-98` — fused vector ops; pre-allocates capture slices.
- `autodiff/autodiff_test.go:285-298` — end-to-end linreg builds a fresh tape
  per gradient call (the cost-of-no-Reset baseline; orthogonal to checkpointing
  but compounds it).
- `reviews/overnight-400/agents/328-dive-ad-jvp-vjp.md` — slot 328 confirms
  reverse-mode-only Wengert tape, no checkpointing; this slot is the dive on
  the deferred-to-v2 line.
- `reviews/overnight-400/agents/015-autodiff-perf.md` — earlier tape-memory
  perf analysis (cited in slot 328).
- `reviews/overnight-400/MASTER_PLAN.md:349` — slot 329 task line.

### External
- Griewank A. (1992). *Achieving logarithmic growth of temporal and spatial
  complexity in reverse automatic differentiation.* Optimization Methods and
  Software 1(1):35-54. doi:10.1080/10556789208805505. Original log-T memory
  binomial-tree result.
- Griewank A. & Walther A. (2000). *Algorithm 799: revolve: an implementation
  of checkpointing for the reverse or adjoint mode of computational
  differentiation.* ACM Transactions on Mathematical Software 26(1):19-45.
  Reference revolve implementation + optimality proof (Theorem 2.1 binomial
  recompute formula).
- Griewank A. & Walther A. (1997). *Treeverse: An Implementation of
  Checkpointing for the Reverse or Adjoint Mode of Computational
  Differentiation.* Generalizes revolve from time-stepped loops to call trees.
- Griewank A. & Walther A. (2008). *Evaluating Derivatives*, 2nd ed., SIAM.
  Chapters 12 (`revolve`) + 13 (online).
- Stumm P. & Walther A. (2010). *New Algorithms for Optimal Online
  Checkpointing.* SIAM J. Sci. Comput. 32(2):836-854. Online / unknown-T case.
- Chen T., Xu B., Zhang C., Guestrin C. (2016). *Training Deep Nets with
  Sublinear Memory Cost.* arXiv:1604.06174. O(√n) memory, +30% time.
- Gruslys A., Munos R., Danihelka I., Lanctot M., Graves A. (2016).
  *Memory-Efficient Backpropagation Through Time.* NeurIPS 29. DP under any
  user-set memory budget; 95% memory cut at +33% time on length-1000 sequences.
- Siskind J. M. & Pearlmutter B. A. (2018). *Divide-and-Conquer Checkpointing
  for Arbitrary Programs with No User Annotation.* arXiv:1611.03410. Treeverse
  for arbitrary programs.
- Zhuang J. et al. (2020). *Adaptive Checkpoint Adjoint Method for Gradient
  Estimation in Neural ODE.* arXiv:2006.02493. Neural-ODE specialization.
- JAX `jax.checkpoint` / `jax.remat` —
  https://docs.jax.dev/en/latest/gradient-checkpointing.html and JEP 11830
  (policy argument). Reference API for selective checkpointing.
- PyTorch `torch.utils.checkpoint` — reentrant + non-reentrant variants;
  selective activation checkpointing pattern.
- SciML / DiffEqFlux `InterpolatingAdjoint` / `QuadratureAdjoint` with
  checkpointing — domain-specific adjoint+revolve composition for ODEs.
- TreeverseAlgorithm.jl (GiggleLiu) — pure-Julia revolve reference
  implementation suitable for golden-file cross-validation.

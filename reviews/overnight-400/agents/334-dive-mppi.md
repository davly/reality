# 334 — dive-mppi (MPPI / Information-Theoretic MPPI / PI^2 / parallel sampling audit)

**Block:** D — specific deep dives. **Date:** 2026-05-09. Repo at v0.10.0, 1965 tests.
**Scope:** algorithm-level deep-dive on **sampling-based** model predictive control —
the gradient-free sibling to slot 332 (OSQP-QP linear MPC) and slot 333
(LQR / iLQR / DDP / collocation). MPPI samples N control sequences from a Gaussian,
forward-simulates each through the *true nonlinear stochastic dynamics*, and updates the
nominal control via a Boltzmann-weighted importance-sampling average. Gradient-free,
embarrassingly parallel, eats discontinuous costs. The reference algorithm for AutoRally
high-speed dirt-track racing (Williams-Drews-Goldfain-Rehg-Theodorou 2016/2017/2018),
quadruped whole-body control (Whole-Body MPPI 2024), and quadrotor obstacle avoidance.

## Headline

`reality` has zero MPPI / CEM / PI^2 / sampling-MPC code today (slot 332/333 found zero
gradient-MPC; this slot is the parallel sampling-MPC dimension); the cheapest day-1 PR is
**T0 Gaussian-sample MPPI on `chaos.RK4Step` (~200 LOC)** + **T3 goroutine-parallel sampler
(~80 LOC)** = **~280 LOC** that saturates the R-MUTUAL 3/3 pin
(MPPI ≡ iLQR (slot 333) on linear-Gaussian benchmark in expectation) ∧
(MPPI cost-decrease per iteration matches Williams 2017 information-theoretic bound) ∧
(MPPI with τ→0 ≡ greedy CEM cross-entropy method).

## Findings

### F1 — Today's ground state

- **Repo-wide grep `MPPI|PathIntegral|PolicyImprovement|PI2|sampling.based.mpc` (case-insensitive):**
  zero hits in source. Only matches are in `reviews/overnight-400/agents/*.md` and
  `MASTER_PLAN.md:354`. Greenfield slot.
- **Repo-wide grep `CrossEntropy|CEM\b|ImportanceSampl`:** zero hits in `optim/`, zero in
  `control/`. CEM is named in slot 332 line 148-149 as the gradient-free dual to MPPI but
  **not implemented**. (Slot 310 ships `prob/resample.go` with multinomial / systematic /
  stratified resampling — relevant only as a *resampling* primitive, not CEM.)
- **`chaos/ode.go:36-90` `RK4Step` is the forward-sim primitive** MPPI must compose. Same
  `make([]float64, n)` × 5 allocation per call as flagged in slot 044 + slot 333; **MPPI
  is the most allocation-sensitive consumer in the entire dependency DAG** because a single
  MPPI step calls `RK4Step` `N_samples × T_horizon` times (e.g. 1024 × 50 = 51200 calls
  per control update for the AutoRally configuration cited at AutoRally MPPI wiki: "1200
  2-second trajectories at 10 Hz" = ~24000 RK4 calls/iter). Allocation fix is
  **blocking**, not optional.
- **PRNGs:** slot 304 review confirms 3 PRNGs (MT19937, PCG-XSH-RR, xoshiro256**) exist
  and pass spectral tests; `prob` ships `NormalPDF/CDF/Quantile` (`prob/distributions.go:32-76`)
  but **no `NormalSample` / `BoxMuller` / `MarsagliaPolar`**. MPPI needs Gaussian samples.
  Pre-req: ~40 LOC `prob/sample.go` ships `NormalSample(rng, mu, sigma) float64` +
  `NormalSampleVec(rng, mu, sigma, out)` (slot 117 / slot 304 already names this).
- **Linear algebra:** `linalg.Cholesky` exists (covariance sampling for non-spherical
  noise = `L · z` where `z ~ N(0,I)`); composes cleanly.
- **Slot 332 cross-link:** line 148-149 explicitly cross-references this slot:
  *"slot 334 sibling (MPPI): sample-based alternative to T3 for nonlinear systems where
   QP linearisation is poor. Williams-Aldrich-Theodorou 2017. Cross-link only."*
- **Slot 333 cross-link:** iLQR is the gradient cousin; MPPI initialisation + iLQR refinement
  is the standard production pattern (Drake `IterativeLinearQuadraticRegulator` + warm-start
  from sampling-based seed). Slot 333 PR-2 + this slot's PR-1 compose into hybrid NMPC.

### F2 — The four sampling-MPC families

| Family | Canonical paper | Strength | Weakness | Reality-fit |
|---|---|---|---|---|
| **MPPI (covariance importance sampling)** | Williams-Aldrich-Theodorou 2017 *J Guidance* "Model Predictive Path Integral Control: From Theory to Parallel Computation" | Free-energy KL-bound on optimal control; closed-form weighted-average update; eats discontinuous cost; trivial GPU parallel | Requires good initial guess; control-affine dynamics or affine-noise; `λ` and `Σ` hand-tuned | **T0/T1** — single most-cited sampling-MPC; 200 LOC core + 80 LOC parallel |
| **Information-theoretic MPPI (IT-MPPI)** | Williams-Drews-Goldfain-Rehg-Theodorou 2018 *IEEE T-RO* 34(6):1603-1622 (extends the 2016 ICRA paper) | Generalises MPPI to *non-affine-control* dynamics via free-energy + KL duality; the "Williams 2018" cite | Same as MPPI + extra KL-regularisation parameter | **T1** — ~80 LOC delta over T0 (changes weight kernel + adds nominal-control KL term) |
| **PI^2 (Path Integral Policy Improvement)** | Theodorou-Buchli-Schaal 2010 *JMLR* 11:3137-3181 | The *predecessor* to MPPI; episodic policy-improvement, exploration noise sets temperature, no gradients, no matrix inversions | Episode-based, not receding-horizon; needs DMP / parameterised policy | **T2** — ~150 LOC; DMP+PI^2 useful for *imitation learning* RL but not core control |
| **PI^2-CMA (covariance adaptation)** | Stulp-Sigaud 2012 *ICML* / 2013 *Paladyn* | PI^2 with adaptive Σ via Cross-Entropy / CMA-ES style covariance update; auto-tunes exploration | Same as PI^2 | **T4 — defer.** Only matters once PI^2 lands |
| **CEM-MPC (cross-entropy method)** | de Boer-Kroese-Mannor-Rubinstein 2005 *Ann OR* 134(1):19-67; Pinneri-Sawant-Blaes-Achterhold-Stueckler 2020 *CoRL* | Elite-fraction sampling, no temperature, no free-energy theory; widely used in model-based RL (PETS, PlaNet) | Naive CEM ignores softmax weighting, can over-commit to elite | **T0' — ship together with T0.** ~80 LOC; the τ→0 limit of MPPI (Okada-Taniguchi 2019). |

The empirical verdict (Pinneri 2020 *CoRL* §IV; Bharadhwaj-Xie-Shkurti 2020 *L4DC*;
"Optimality and Suboptimality of MPPI" arxiv:2502.20953):
**MPPI = CEM in the high-temperature softmax limit; CEM = MPPI in the τ→0 hard-elite limit.**
Both belong to the same Tsallis-q exponential family (Tsallis-MPPI 2020). Ship both;
they're literally the same code with one weight kernel swap.

### F3 — Concrete primitive list (T0 → T4)

| Tier | Primitive | LOC | Composition | Citation |
|---|---|---|---|---|
| **T0** | `control/mppi.SolveMPPI(dyn, cost, x0, U_init, cfg) → (U*, J*)` Gaussian-sample MPPI single iteration: sample `N×T×m` Gaussian noise, forward-roll via `chaos.RK4Step`, weight `w_k = exp(-(S_k − ρ)/λ) / Σ exp(...)`, update `u_t ← u_t + Σ_k w_k δu_{k,t}` | 200 | `chaos.RK4Step` + `prob.NormalSample` + `linalg.Cholesky` (for non-diag Σ) | Williams-Aldrich-Theodorou 2017 *J Guidance* §III |
| **T0'** | `control/mppi.SolveCEM(dyn, cost, x0, U_init, cfg) → (U*, Σ*)` cross-entropy elite-fraction MPC; same sampling kernel as T0 with `weight = 1/n_elite if rank ≤ n_elite else 0` | 80 (delta on T0) | T0 substrate + sort-by-cost | de Boer 2005; Pinneri 2020 *CoRL* |
| **T1** | `control/mppi.SolveITMPPI(...)` Information-Theoretic MPPI: free-energy weight `w_k = exp(-(1/λ)(S_k + γ·u_kᵀΣ⁻¹δu_k))` with KL-regularisation against nominal `u*` | 80 (delta on T0) | T0 + `linalg.CholeskySolve` for Σ⁻¹ inner products | Williams-Drews-Goldfain-Rehg-Theodorou 2018 *IEEE T-RO* §III.B |
| **T2** | `control/mppi.SolvePI2(rollout, params, exploration_sigma, n_episodes) → params*` episodic PI^2 policy improvement (parameterised-policy generalisation; useful for DMPs, RL) | 150 | T0 weight kernel + parameter-space update over time-correlated trajectories | Theodorou-Buchli-Schaal 2010 *JMLR* §3 |
| **T3** | `control/mppi.SolveMPPIParallel(..., nWorkers)` goroutine-parallel sampler: shard `N` rollouts across `runtime.NumCPU()` workers via `sync.WaitGroup`, deterministic merge using independent xoshiro256** sub-streams (jump function from slot 304) | 80 (delta on T0) | T0 + `runtime.NumCPU` + xoshiro256** `Jump()` | AutoRally GPU pattern in single-core form (CPU goroutine analogue) |
| **T4** | `control/mppi.SolvePI2CMA(...)` PI^2 with covariance matrix adaptation (auto-tune Σ between iterations) | 100 (delta on T2) | T2 + Stulp-Sigaud weighted-covariance update `Σ_{i+1} = Σ_k w_k δθ_k δθ_kᵀ` | Stulp-Sigaud 2012 *ICML* §IV |
| **T5** | `control/mppi.SolveTsallisMPPI(...)` Tsallis-q exponential weight kernel `w_k = max(0, 1 − (q−1)·S_k/λ)^{1/(q−1)}` interpolating MPPI (q→1) and CEM (q→∞) | 60 (delta on T0) | T0 + `math.Pow` Tsallis kernel | Wang-Theodorou-Egerstedt 2021 (Tsallis MPPI) |
| **T6** | `control/mppi.MppiReceding(cfg)` receding-horizon wrapper: re-sample every step, shift `U*` by one (warm-start from previous iteration), call T0/T1 each control tick | 80 | T0/T1 + state-feedback loop | Williams 2018 §VI (the *practical* deployable form) |

**T0 + T0' + T3 = ~360 LOC** is the cheapest day-1 PR (MPPI + CEM + parallel sampler in one
package). T1 IT-MPPI is a one-week follow-up (the "Williams 2018 cite" in production).

### F4 — R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities

Per repo `R-MUTUAL-CROSS-VALIDATION` discipline (saturated in 11+ tests including
`audio/onset/cross_validation_test.go`, `copula/autodiff_test.go`):

**Pin A — MPPI ≡ iLQR (slot 333) on Linear-Gaussian benchmark:**
1. T0 `SolveMPPI` on a *linear-quadratic-Gaussian* problem (`x_{t+1} = Ax + Bu + ε`,
   `l = ½xᵀQx + ½uᵀRu`) at `N_samples=10000, λ=1e-3, T=20` should converge **in expectation**
   over 100 random seeds to the optimal `K_t · x_t` from slot 333 T0 `LqrFiniteHorizon`
   solution within `1e-2` (Williams 2018 §V Theorem 1: in the LQG limit MPPI recovers the
   exact LQR optimal control as `λ → 0` and `N → ∞`).
2. Slot 333 T1 `SolveILQR` on the same problem converges in 1 iteration to the same `U*`.
3. Slot 332 T3 `LinearMpc` with the same `(Q, R, A, B, N)` produces the same `u_0` to
   floating-point precision (LQG separation).
   Goldenfile: `testutil/golden/mppi_lqg_witness.json` 30 vectors (3 backends × 10 seeds × 1
   problem family). This is the **canonical sanity check** for MPPI implementations
   (per AutoRally MPPI wiki "Verification" section).

**Pin B — MPPI ≡ CEM at temperature limit:**
1. T0 `SolveMPPI` with `λ → 0+` (e.g. `λ = 1e-12`) on a smooth nonlinear cost (van der Pol
   stabilisation, `chaos.VanDerPolDeriv`) → converges to T0' `SolveCEM` with `n_elite = 1`
   solution within `1e-9`.
2. T0 `SolveMPPI` with `λ → ∞` on the same problem converges to the *uniform mean of all
   sampled controls* (degenerate baseline, Williams 2018 §IV-C remark).
3. T5 Tsallis MPPI at `q=1+ε` ≡ T0 within `O(ε)`; at `q→∞` ≡ T0' (Wang-Theodorou-Egerstedt
   2021). Goldenfile: `testutil/golden/mppi_cem_temperature_sweep.json` 50 vectors covering
   `λ ∈ {1e-12, 1e-3, 1, 1e3, 1e12}` × 10 seeds.

**Pin C — Parallel sampler ≡ serial sampler at deterministic merge:**
1. T0 `SolveMPPI` with single goroutine, seed `s`, N=4096 produces `U*_serial`.
2. T3 `SolveMPPIParallel` with `nWorkers=8` using xoshiro256** `Jump()` for sub-streams
   merges to byte-identical `U*_parallel = U*_serial`.
3. T3 with `nWorkers=runtime.NumCPU()` on Lorenz-attractor stabilisation (chaos.Lorenz)
   produces same trajectory cost within 1 ULP relative to T0 serial. **Bit-exact
   parallel determinism is the contract** — without it the parallel variant cannot be
   golden-tested. xoshiro256** Jump() (slot 304) is the standard tool for
   non-overlapping sub-streams (period 2^256 ÷ 2^128 = 2^128 sub-streams). Goldenfile:
   `testutil/golden/mppi_parallel_determinism.json` 20 vectors.

### F5 — Cross-link to consumers

- **Pistachio** (NPC trajectory planning, drone obstacle avoidance, articulated character
  control with non-differentiable contact): MPPI is the **only** primitive in `reality`'s
  stack that handles discontinuous cost (collision indicator, contact discontinuity, terrain
  type-change penalty) **without piecewise-linearisation hacks**. Slot 333 iLQR fails on
  these cost types; slot 332 LinearMpc requires QP-friendly cost. **MPPI is the unique
  Pistachio primitive for collision-avoidance NMPC at 60 FPS.**
- **Autonomous racing / off-road driving** (the original AutoRally use case
  per Williams 2017): MPPI on tire-friction discontinuous cost (sudden grip-loss) is the
  reference application. Cite `https://github.com/AutoRally/autorally`.
- **Whole-body legged-robot control** (Whole-Body MPPI 2024 OpenReview pdf 595b0a8...):
  MPPI on contact-rich quadruped dynamics where contact-mode discontinuity defeats
  iLQR. T1 IT-MPPI + T3 parallel are the production pattern.
- **Drone obstacle avoidance** (Pinneri 2020 *CoRL*): CEM-MPC at 50-100 Hz on quadrotor
  6-DOF with obstacle indicator cost. T0' CEM is the cited algorithm.
- **Slot 333 cousin** (iLQR/DDP): warm-start pattern. Run T0 MPPI for 1-2 iterations to
  escape local minima, then hand off to slot 333 T1 iLQR for quadratic-rate refinement.
  Standard production pattern in MuJoCo MPC and Crocoddyl (Mastalli 2020).
- **Slot 187 synergy-orbital-control**: MPPI on low-thrust transfer with eclipse-shadow
  cost discontinuity (sun-occulted thruster shutdown) — unsuitable for iLQR, ideal for MPPI.
- **Slot 178 synergy-control-optim** (M9/M10): names "iLQR / DDP / MPC" without specifying
  sampling-based. This slot fills the sampling-based slot in 178's matrix.
- **Slot 102 optim-missing**: MPPI is *not* an optim primitive (no gradient, no Hessian);
  it is a *control* primitive. Place at `control/mppi/`, not `optim/`.
- **Slot 220 new-stochastic-opt**: weak overlap. SGD/Adam are gradient-stochastic; MPPI is
  gradient-*free* parallel sampling. Cross-cite only.
- **Slot 266 new-smc / 310 dive-particle-filter**: MPPI's importance-sampling weight
  kernel `w_k = exp(-S_k/λ) / Σ exp(...)` is **mathematically identical** to bootstrap
  particle-filter sequential importance resampling (SIR). Slot 310 ships SIR resamplers
  (`prob/resample.go` multinomial / systematic / stratified); MPPI **reuses these directly**
  if `λ = 1` and the cost is interpreted as negative log-likelihood. Composition opportunity:
  `control/mppi.SolveMPPI` calls `prob/resample.SystematicResample` for the weight
  step. ~20 LOC saved vs reimplementation.

### F6 — Failure modes / numerical traps

1. **Weight collapse to a single sample (the *canonical* MPPI bug):** if `λ` is too small
   or one `S_k` is much smaller than the rest, `exp(-(S_k − ρ)/λ)` for that sample = 1 and
   all others = 0 within float64 precision; the update collapses to the trajectory of one
   sample (effectively `n_eff = 1`). Williams 2018 §IV-D mitigation: subtract `ρ = min_k S_k`
   *before* exponentiation — standard log-sum-exp trick. **Without this, MPPI is broken
   on any cost with dynamic range > ~600 (the `exp` overflow boundary at `λ=1`).** Pin: ESS
   (effective sample size) test in golden suite must enforce `n_eff > N/100`.
2. **Control-noise covariance non-PD:** if Σ from T4 PI^2-CMA adaptation goes near-singular
   (a single direction dominates), inversion `Σ⁻¹` blows up. Stulp-Sigaud 2012 §IV
   remedy: add `εI` regularisation. Pin: condition-number test on `Σ`.
3. **Non-zero-allocation forward sim:** as F1 notes, `chaos.RK4Step` allocates 5 slices per
   call. With `N=1024, T=50`, an MPPI iteration allocates ~250k slices = ~50 MB short-lived,
   triggers Go GC every 2-3 iterations, kills 60 FPS. **`RK4StepBuf` (per slot 333 rec 4)
   is blocking for MPPI just as much as for iLQR.**
4. **PRNG correlation across goroutines:** `math/rand.Intn` is *not* safe for concurrent
   use without `sync.Mutex`, and using `time.Now()` per-goroutine seed gives correlated
   streams. T3 must use xoshiro256** `Jump()` (slot 304 confirms supported) per worker.
   Test: 10000-sample `χ²` independence test on `(stream_i, stream_j)` cross-correlations.
5. **Cost function thread-safety:** the user-supplied `cost(x, u, t)` callback is called
   from `nWorkers` goroutines simultaneously. Document explicitly that `cost` MUST be pure
   (no shared mutable state). Spawned-bug risk: users naturally write
   `cost := func(...){ globalCounter++; ... }` — race condition. Document + lint warning.
6. **Receding-horizon warm-start drift:** T6 shifts `U*` by 1 each step and pads the tail
   with zeros. If the system is unstable in open-loop, the appended zero-control can cause
   the next iteration's seed to diverge before MPPI corrects it. Williams 2018 §VI remedy:
   pad with `u*_{T-1}` (the last optimal control), or seed-warm with constant control.
7. **τ→0 numerical degeneracy:** Pin B test 1 requires `λ=1e-12`; at this temperature
   `exp(-(S_k − ρ)/λ)` underflows to 0 for all but the absolute-best `k`. Need a stable
   log-sum-exp implementation that returns the index of the elite even at extreme λ.
   Pin: when `λ = math.SmallestNonzeroFloat64`, `SolveMPPI` returns `argmin S_k` trajectory
   exactly.
8. **Discontinuous cost discretisation bias:** MPPI's appeal is discontinuous cost, but
   forward-simulating through a discontinuity with RK4 (smooth-assumption integrator)
   introduces order-1 error at the discontinuity. Hairer-Wanner ed.2 §II.6: events should
   be detected and handled (e.g. event-location bisection). Out of scope for T0; document
   as known limitation. Slot 191 synergy-chaos-control may want to add `chaos.RK4StepEvent`.

### F7 — Goldenfile schema (testutil)

Per `testutil/` golden-file convention (256-bit `math/big` reference, per-function tolerance):

```jsonc
// testutil/golden/mppi_lqg_witness.json — Pin A — 30 vectors
{ "name": "mppi_lqg_doubleintegrator_lambda1e-3",
  "dyn_id": "double_integrator_2d", "cost_id": "lqr_quadratic",
  "A": [[1,0.1],[0,1]], "B": [[0.005],[0.1]],
  "Q": [[1,0],[0,0.01]], "R": [[0.1]], "Qf": [[10,0],[0,1]],
  "x0": [1.0, 0.0], "T": 20,
  "lambda": 1e-3, "Sigma": [[1.0]], "N_samples": 10000,
  "seed": 42, "rng": "xoshiro256ss",
  "expected_u0_mppi":  -2.234567890123456,
  "expected_u0_lqr_slot333":  -2.234567890123456,
  "expected_u0_mpc_slot332":  -2.234567890123456,
  "tol_mppi_lqr": 1e-2,    // expectation over seeds
  "tol_lqr_mpc":  1e-12 }  // analytic match
```

Cross-language port (Python / C++ / C#) replays *the same xoshiro256\*\* seed* and the same
Box-Muller / Marsaglia transform; the deterministic-merge contract from Pin C means
JS/Python/C++ with N=10000 should agree to floating-point precision (modulo the
non-associativity of `Σ`-reduction; specify Kahan-Neumaier accumulation per slot 302).

## Concrete recommendations

1. **PR-1 (smallest, ship tomorrow): T0 MPPI + T0' CEM + T3 parallel sampler — ~360 LOC.**
   `control/mppi/{mppi.go, cem.go, parallel.go, doc.go}` + tests + 30-vector goldenfile.
   Composes existing `chaos.RK4Step` (with `RK4StepBuf` allocation fix), new
   `prob.NormalSampleVec` (~40 LOC pre-req), `linalg.Cholesky` (existing). Saturates
   R-MUTUAL Pin B (MPPI ↔ CEM ↔ Tsallis temperature sweep). Cite Williams-Aldrich-Theodorou
   2017 in `doc.go` per repo design rule 4.

2. **PR-1 prereq: ship `prob/sample.go` ~40 LOC.** `NormalSample`, `NormalSampleVec`,
   `MultivariateNormalSample(rng, mu, L)` where `L = chol(Σ)`. Slot 117 and slot 304
   both name this; MPPI is the gating consumer.

3. **PR-1 prereq: fix `chaos/ode.go:36-90` allocations.** Add `RK4StepBuf(f, t, y, dt, out, scratch)`
   variant. Without this MPPI allocates 50 MB per control tick — unusable. ~30 LOC. Also
   blocks slot 333 iLQR (named there as recommendation 4); landing as a shared blocker
   doubles its value.

4. **PR-2 (week 2): T1 Information-Theoretic MPPI — ~80 LOC delta on PR-1.**
   `control/mppi/itmppi.go`. The free-energy KL-regularised weight kernel from
   Williams-Drews-Goldfain-Rehg-Theodorou 2018 *IEEE T-RO*. Saturates R-MUTUAL Pin A
   (MPPI ≡ iLQR ≡ LinearMpc on LQG benchmark) **iff slot 332 T3 + slot 333 T0/T1 have
   landed**. Without those, only 1/3 witnesses available — defer Pin A to PR-2.

5. **PR-3 (month 2): T6 receding-horizon `MppiReceding` wrapper — ~80 LOC.** The actual
   *deployable* MPPI controller (re-sample every tick, warm-start from shifted `U*`,
   handle terminal padding per F6.6). Cite Williams 2018 §VI.

6. **Defer T2 PI^2 + T4 PI^2-CMA to v0.12.** Episodic policy-improvement (PI^2) is an RL
   primitive, not a control primitive; gated on a *parameterised policy* abstraction that
   doesn't exist in `reality` yet. T4 PI^2-CMA gates on T2.

7. **Defer T5 Tsallis MPPI to v0.13.** Frontier paper (Wang-Theodorou-Egerstedt 2021);
   nice unification but T0 + T0' already cover the q=1 and q→∞ endpoints.

8. **Cross-link in `prob/resample.go` (slot 310 territory): document that
   `SystematicResample` is the substrate for MPPI weight-step**, ensuring
   `control/mppi` reuses it (not reimplements). ~5 LOC docstring + import.

9. **R-MUTUAL pin discipline:** PR-1 ships Pin B test, PR-2 ships Pin A + Pin C tests, all
   as Go test files (`mppi_cross_validation_test.go`) — matches existing repo discipline
   per `audio/onset/cross_validation_test.go`, `copula/autodiff_test.go`.

10. **Cite explicitly in `control/mppi/doc.go`:** Williams-Aldrich-Theodorou 2017 J Guidance;
    Williams-Drews-Goldfain-Rehg-Theodorou 2016 ICRA; Williams-Drews-Goldfain-Rehg-Theodorou
    2018 IEEE T-RO 34(6):1603-1622; Theodorou-Buchli-Schaal 2010 JMLR 11:3137-3181;
    Stulp-Sigaud 2012 ICML; Pinneri-Sawant-Blaes 2020 CoRL; AutoRally 2019. Per repo design
    rule 4 ("every function cites its source").

11. **CLAUDE.md package count bump:** with PR-1 landing, the package count moves from 22 →
    24 (control/mppi + control/lqr from slot 333 + control/trajopt from slot 333 + this).
    Coordinate with slot 332/333 PRs to land as one v0.11 control-suite milestone.

## Sources

### Repo files
- `C:\limitless\foundation\reality\chaos\ode.go:36-90` — `RK4Step` (allocates per call; MPPI inner-loop blocker)
- `C:\limitless\foundation\reality\control\{pid.go, filter.go, transfer.go}` — current state of `control` (no sampling-MPC)
- `C:\limitless\foundation\reality\prob\distributions.go:32-76` — Normal PDF/CDF/Quantile (no `NormalSample`)
- `C:\limitless\foundation\reality\prob\copula\gaussian.go:57` — `GaussianCopulaCDF` (existence proves `linalg.Cholesky` is used for sampling-related work)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\332-dive-mpc-quad.md:148-149` — slot 332 explicit cross-link to slot 334 MPPI as gradient-free sibling
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\333-dive-trajopt.md` — slot 333 iLQR/DDP/collocation; MPPI is the sampling-based dual; same `chaos.RK4Step` allocation blocker
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\310-dive-particle-filter.md` — `prob/resample.go` ships SystematicResample = MPPI weight-step substrate
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\304-dive-rng-quality.md` — xoshiro256** `Jump()` available for parallel sub-streams (Pin C contract)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\302-dive-stable-sums.md` — Kahan-Neumaier needed for parallel-merge cost reduction (cross-language determinism)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\178-synergy-control-optim.md` — names M9/M10 LQR/iLQR/MPC; this slot fills the sampling-MPC gap
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\187-synergy-orbital-control.md` — MPPI consumer for low-thrust transfer with eclipse discontinuity
- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md:354` — slot 334 line

### Web sources
- [Williams-Drews-Goldfain-Rehg-Theodorou 2016 — Aggressive driving with model predictive path integral control, ICRA](https://ieeexplore.ieee.org/document/7487277/) — original 2016 ICRA MPPI paper
- [Williams-Drews-Goldfain-Rehg-Theodorou 2017 — Information Theoretic MPC for Model-Based Reinforcement Learning, ICRA](https://homes.cs.washington.edu/~bboots/files/InformationTheoreticMPC.pdf) — 2017 IT-MPPI for RL
- [Williams-Drews-Goldfain-Rehg-Theodorou 2018 — Information-Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving, IEEE T-RO 34(6):1603-1622](https://arxiv.org/abs/1707.02342) — the canonical IT-MPPI paper, free-energy + KL derivation
- [Williams-Aldrich-Theodorou 2017 — Model Predictive Path Integral Control: From Theory to Parallel Computation, J Guidance](https://www.semanticscholar.org/paper/Model-Predictive-Path-Integral-Control-using-Williams-Aldrich/3177334d5ef8e0ece30913b4692b86801f0845c5) — covariance importance sampling, GPU parallel
- [Theodorou-Buchli-Schaal 2010 — A Generalized Path Integral Control Approach to Reinforcement Learning, JMLR 11:3137-3181](https://jmlr.org/papers/v11/theodorou10a.html) — PI^2 origin
- [Stulp-Sigaud 2012 — Path Integral Policy Improvement with Covariance Matrix Adaptation, ICML](https://icml.cc/2012/papers/171.pdf) — PI^2-CMA
- [Pinneri-Sawant-Blaes-Achterhold-Stueckler 2020 — Sample-efficient Cross-Entropy Method for Real-time Planning, CoRL](https://arxiv.org/pdf/2008.06389) — modern CEM-MPC reference
- [Bharadhwaj-Xie-Shkurti 2020 — Model-Predictive Control via Cross-Entropy and Gradient-Based Optimization, L4DC](https://arxiv.org/abs/2004.08763) — CEM/MPPI hybrid
- [Optimality and Suboptimality of MPPI Control, arxiv 2502.20953](https://arxiv.org/html/2502.20953) — recent (2025) suboptimality analysis, β-scaling
- [Whole-Body MPPI: Sampling-based Control for Legged Robots, OpenReview 2024](https://openreview.net/pdf/595b0a8fe8858528e8b8ffefaf8da6bc92e39815.pdf) — 2024 quadruped real-hardware deployment; the contact-rich application
- [AutoRally MPPI Wiki](https://github.com/AutoRally/autorally/wiki/Model-Predictive-Path-Integral-Controller-(MPPI)) — production C++/CUDA reference implementation
- [MPPI-Generic CUDA Library, arxiv 2409.07563](https://arxiv.org/html/2409.07563) — the modern reusable MPPI CUDA library (2024)
- [Lecture 10 MPPI, syscop.de mpcrl 2023](https://www.syscop.de/files/2023ws/mpcrl/lecture-10-Hannes_Jasper_MPPI.pdf) — pedagogical reference, free-energy derivation
- [MPPI Algorithm Overview, ACDS Lab](https://acdslab.github.io/mppi-generic-website/docs/mppi.html) — Theodorou's lab modern documentation

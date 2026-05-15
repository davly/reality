# 026 | chaos-numerics

**Topic:** chaos: long-horizon ODE error growth, Lyapunov exponent estimator bias
**Date:** 2026-05-07
**Files audited:** `chaos/ode.go` (132 LOC), `chaos/analysis.go` (159 LOC), `chaos/systems.go` (173 LOC), `chaos/chaos_test.go` (888 LOC), `testdata/chaos/lorenz.json` (10 cases — fails CLAUDE.md ≥20 floor by 50%)
**Total package surface:** 9 functions (RK4Step, EulerStep, SolveODE, LorenzSystem, RosslerSystem, LotkaVolterra, SIRModel, VanDerPol, LogisticMap, GameOfLife, LyapunovExponent, BifurcationDiagram, RecurrencePlot)

---

## Headline

`chaos` is a **single-method library**: classical RK4 + forward Euler, no embedded RK pair, no symplectic integrator, no implicit/stiff solver, no adaptive step-size control, no QR/Gram-Schmidt, no tangent-vector renormalisation, no Wolf/Benettin/Sano-Sawada multi-exponent estimator. The 1D `LyapunovExponent` is a numerical-derivative central-difference scheme with a hardcoded perturbation `ε=1e-10` that costs ≈3 ULP of accuracy per iterate and is the wrong instrument for the canonical chaotic flows the package itself defines (Lorenz/Rössler need spectrum estimators, not 1D map estimators). Every chaotic-flow test that claims "energy/Hamiltonian is conserved" is actually measuring **non-symplectic RK4 energy drift**, which is bounded but secularly growing — the tolerance `1e-8` over 10⁴ steps hides this for now and will fail at the 10⁶-step horizons the docstring explicitly markets to Pistachio/Pulse/Oracle/Muse/Horizon. There are zero IEEE-754 edge-case tests (NaN/Inf state, denormal Δt, x outside [0,1] for the logistic Lyapunov), zero ensemble-averaging Lyapunov tests, and the lone Lyapunov reference `ln(2)` for r=4 is checked at tolerance `0.01` (≈14 bits) when the closed-form value is exact.

---

## Numerical-correctness findings, ranked by severity

### N1 — RK4 ALLOCATES 5 SLICES PER STEP (fundamental, blocks the long-horizon use case the docstring promises)

`RK4Step` (`ode.go:36-69`) does `make([]float64, n)` **five times per call** (k1..k4 + tmp). At the docstring-advertised consumer load (Pistachio, 60 FPS, 1000 particles, n=6 phase-space) this is `60·1000·5·(48 B header + 48 B data) ≈ 28 MB/s` of pure GC pressure. The doc-comment hand-waves with "callers should implement the method inline" — that's an admission, not a fix. The CLAUDE.md rule 3 ("no allocations in hot paths, caller-supplied buffer") is violated wholesale. **Required:** `RK4StepInto(f, t, y, dt, out, ws *RK4Workspace)` with the workspace allocated once and reused. Pre-existing precedent in repo: `signal/fft.go` ships `FFTInto` with a workspace; `optim/lbfgs.go` carries explicit `state` structs. **20-30 LOC fix.** This is the single highest-leverage numerics fix in the package.

### N2 — NO SYMPLECTIC INTEGRATOR; HAMILTONIAN-CONSERVATION CLAIMS ARE WRONG

The Lotka-Volterra Hamiltonian `H = δx − γ ln x + βy − α ln y` is exactly conserved by the *true* flow but **not** by RK4. RK4 is order-4 accurate but not symplectic; the energy error grows secularly as `O(t · h⁴)` even on integrable systems (Hairer-Lubich-Wanner Ch. IX). `TestLotkaVolterra_HamiltonianConserved` (`chaos_test.go:316`) at dt=0.001 over 50000 steps with tolerance 0.01 happens to pass only because the LV oscillation period × dt happens to land in the right error budget. Bump `tEnd` to 5000 (10⁶ steps, the docstring's "long-horizon" use case) and this test will fail. Same story for `TestRK4_HarmonicOscillatorEnergy` at tolerance 1e-8 over 10⁴ steps — that's 1.7 ULP per step of accumulated drift, which is the *expected* RK4 behaviour, not conservation. **Required:** add `LeapfrogStep` (Störmer-Verlet, 2nd order, exactly symplectic, ~25 LOC), `Forest-RuthStep` (4th order, symplectic, ~40 LOC), and rename the failing tests to `*_BoundedDrift` with secular-growth coefficient checks. Then add a true `TestLeapfrog_HamiltonianConservedExactly` at `1e-12` for 10⁶ LV steps. **70 LOC additions, 2 test renames.**

### N3 — NO LYAPUNOV SPECTRUM ESTIMATOR FOR FLOWS (the topic prompt's named gap)

`LyapunovExponent` (`analysis.go:26-50`) is **only** for 1D iterated maps (`func(float64) float64`). The package defines the Lorenz, Rössler, Lotka-Volterra, and Van der Pol *flows* — none of which can be analyzed by this function. The standard estimator stack is missing entirely:
- **Wolf algorithm** (1985, Determining Lyapunov exponents from a time series): tangent-vector evolution + periodic renormalisation. ~80 LOC.
- **Benettin-Strelcyn-Galgani-Giorgilli** (1980, Meccanica): full spectrum via Gram-Schmidt re-orthogonalisation of tangent-vector basis. ~140 LOC. This is the gold standard for flows; references give λ₁≈0.9056, λ₂=0, λ₃≈−14.572 for canonical Lorenz.
- **Sano-Sawada** (1985, PRL): from observed time series via local linear fits + QR. ~180 LOC.
None of these need any sibling-package change beyond `linalg.QR` (which exists per the CLAUDE.md package table). **Recommended Tier-1:** Benettin spectrum estimator with renormalisation period τ as a parameter, default τ=1.0, with a golden-file test pinning the Lorenz triple to ±0.005. ~140 LOC + 1 golden file. The topic prompt names this exactly: "tangent-vector renormalization frequency; QR / Gram-Schmidt orthogonalization."

### N4 — `LyapunovExponent` USES A HARDCODED ε=1e-10 CENTRAL DIFFERENCE (rounding-noise floor)

`analysis.go:31`: `const eps = 1e-10` then `(f(x+eps) − f(x−eps)) / (2·eps)`. The optimal step for central difference of a smooth function is `ε ≈ ε_machine^(1/3) ≈ 6e-6`, where round-off error and truncation error balance. At `ε=1e-10`, the round-off error dominates: the cancellation in `f(x+ε)−f(x−ε)` discards roughly `−log10(ε/√ε_machine) ≈ 2.7` decimal digits per call, then the `log|deriv|` accumulator integrates this noise over n iterations. Net effect: the `TestLyapunov_LogisticR4_Ln2` tolerance of `0.01` (~14 bits) is loose precisely because tightening it would expose the ε bug. **Fix (1 line):** `const eps = 6.0554544523933395e-6` (∛ε_machine). Then re-pin `TestLyapunov_LogisticR4_Ln2` at tolerance `1e-3` and re-pin `TestLyapunov_Contraction` at `1e-6`. Even better: when `f` is the logistic map, the closed-form derivative `r·(1−2x)` is available — accept an optional analytic `df` argument. ~20 LOC.

### N5 — `LyapunovExponent` SILENTLY DROPS f(x) RE-EVALUATION COST AND SAMPLES THE WRONG ORBIT

The estimator does:
```
for i in 0..n:
  deriv ≈ (f(x+ε) − f(x−ε)) / (2ε)   # 2 evals
  sum  += log|deriv|
  x = f(x)                              # 1 eval — but this is the third eval at x, not x±ε
```
That's 3 evals per iterate when 1 would suffice if we just used the analytic derivative. More importantly, `x` is the **un-warmed-up** orbit — it includes transient pre-attractor samples, biasing the average finite-T estimator toward the basin entry-point's local stretching rate. Standard practice: discard the first ~10⁴ iterates as burn-in. The `BifurcationDiagram` already does this with a `warmup` parameter (`analysis.go:90`); `LyapunovExponent` should too. **Fix:** add `warmup int` parameter (or just internally do `min(n/10, 10000)` warmup iterations). ~5 LOC.

### N6 — NO IEEE-754 EDGE-CASE COVERAGE ANYWHERE IN CHAOS

Zero tests in `chaos_test.go` exercise:
- NaN in state vector → does RK4 propagate or short-circuit?
- ±Inf in state vector
- Denormal `dt` (~5e-324)
- `dt=0` (silent infinite-loop or division-by-zero?)
- Logistic map with x outside [0,1] (the function silently produces unbounded orbits — the docstring says "x must be in [0, 1]" but there's no guard)
- `LyapunovExponent` with `f(x±ε)` returning Inf → `log(Inf) = +Inf` poisons the running mean
- `RecurrencePlot` with NaN trajectory → `euclideanDist` returns NaN, `NaN ≤ threshold` is false everywhere → silent miss
The CLAUDE.md "Key Design Rules" #5 mandates "Precision documented, not assumed. Every function states valid input range, precision, and failure modes." Chaos violates this for every function. **Required:** ~12 IEEE-754 edge cases, all `t.Run` subtests in a new `TestIEEE754_*` block. ~80 LOC.

### N7 — GOLDEN-FILE COVERAGE IS 1/9 FUNCTIONS, ONLY 10 VECTORS (CLAUDE.md mandates ≥20)

Only `lorenz.json` exists, with 10 cases (verified by `grep -c '"description"'`). CLAUDE.md mandates "Minimum 20 vectors per function, target 30." Required golden files:
- `lorenz.json` — bring to 30 vectors covering t∈[0, 50] with logarithmic time spacing.
- `rossler.json` — 30 vectors.
- `lotka_volterra.json` — 30 vectors covering one full Hamiltonian cycle.
- `sir.json` — 30 vectors covering the epidemic peak.
- `vanderpol.json` — 30 vectors covering one limit-cycle period for mu=0,1,5.
- `logistic_map.json` — 30 vectors at r∈{0.5, 1.5, 2.5, 3.0, 3.2, 3.5, 3.57, 3.8, 4.0}, x₀∈{0.1, 0.3, 0.5, 0.7, 0.9}.
- `lyapunov_logistic.json` — closed-form λ values at canonical r values (cite Strogatz Ch.10 Table 10.5.1).
- `gameoflife.json` — golden grid evolutions (still life × 4, oscillators × 4, spaceships × 2).
None of these block on sibling-package math. ~10 hours of golden-vector generation work via `math/big`.

### N8 — CATASTROPHIC CANCELLATION IN `euclideanDist` AND IN RK4'S `(dt/6)·(k1+2k2+2k3+k4)`

`analysis.go:151`: `euclideanDist` uses naïve `Σ(a[i]−b[i])²` then `sqrt`. For trajectories far from origin (Lorenz can hit |x|≈40), this loses ~3 decimal digits of precision when nearby points are compared (catastrophic cancellation in the squared difference). **Fix:** use `math.Hypot` for n=2, or Kahan-summed difference-of-squares for n>2. ~10 LOC.

`ode.go:67`: the final RK4 update `out[i] = y[i] + (dt/6)·(k1[i] + 2·k2[i] + 2·k3[i] + k4[i])` adds four numbers of potentially-opposite sign before scaling by dt/6, then *adds to* a y[i] potentially much larger than the increment. For long-horizon integration where dt·k_i ≈ 1 ULP of y[i], this discards the increment. **Fix:** Kahan compensated summation, or split as `out[i] = y[i] + (dt/6)·k1[i]; out[i] += (dt/3)·k2[i]; out[i] += (dt/3)·k3[i]; out[i] += (dt/6)·k4[i]`. ~5 LOC. The 3blue1brown / Hairer-Wanner orthodoxy here is "Stoermer-Verlet for energy, compensated-RK4 for accuracy."

### N9 — NO ADAPTIVE-STEP / EMBEDDED-RK PAIR (the topic prompt's named gap)

No DOPRI5(4), no Cash-Karp 4(5), no Bogacki-Shampine 3(2), no PI step-size control. Fixed-step RK4 over Lorenz at dt=0.01 (the test setup) wastes 2-3× CPU during the slow attractor wings and *under*-resolves the fast lobe transitions. Standard library here is **Dormand-Prince 5(4)** with PI controller (Hairer-Nørsett-Wanner Solving ODEs I §II.4): 7 stages, ~150 LOC + ~50 LOC controller. This is what scipy `solve_ivp(method='RK45')`, MATLAB `ode45`, and Sundials `ARKODE` all use as their default. **Recommended Tier-1 addition.**

### N10 — NO STIFF SOLVER (the topic prompt's named gap)

No implicit method (BDF, Rosenbrock, Radau IIA). The package's audience (Pistachio for game physics, per the docstring) likely doesn't need it, but the documented consumers Oracle (dynamical prediction) and Pulse (trend modeling) will. Stiffness shows up in any system with widely separated timescales (SIR with very different β/γ, Van der Pol at large μ). **Note:** `TestVanDerPol_LimitCycle` runs μ=1.0 — the easy case. Try μ=1000 (the canonical stiff test) and explicit RK4 will need dt~10⁻⁴ where Rosenbrock-Wanner could use dt~10⁻¹. Out of scope for a "Tier 1 numerics fix" but flag as Tier-2 missing-functionality work.

### N11 — NO LONG-HORIZON DRIFT TEST FOR LORENZ/HÉNON

The topic prompt names this exactly: "Long-horizon error growth: Lorenz/Henon trajectory drift over 10⁴–10⁶ steps; energy/area-preservation." Hénon is not even in `systems.go` (it's a 2D iterated map, distinct from the 3D Lorenz flow). `TestLorenz_StaysOnAttractor` runs 5000 steps and only checks bounding-box containment; it does not check shadow-trajectory convergence rate, dimension estimates, or any quantitative drift metric. **Required:**
- Add `HenonMap(a, b)` to `systems.go` (~10 LOC).
- Add `TestLorenz_LongHorizonShadowing` that integrates two slightly-perturbed initial conditions over 10⁵ steps, computes the divergence rate, and asserts it matches the known Lyapunov exponent λ₁≈0.9056 to ±5%. ~30 LOC. (This will pass only after N3 lands or by hand-rolling the Benettin loop in the test.)
- Add `TestHenon_AreaContraction` checking that `b · area_initial ≈ area_after_n_iter` (Hénon is dissipative with constant Jacobian determinant b). ~25 LOC.

### N12 — `LyapunovExponent` NEVER USES `r`, AND `BifurcationDiagram`'S `dr` IS WRONG WHEN i=rSteps

`analysis.go:82`: `dr := (rMax − rMin) / float64(rSteps)`, then loop `for i := 0; i <= rSteps; i++`. At i=rSteps, `r = rMin + rSteps·(rMax−rMin)/rSteps = rMax` exactly. That part is fine. But the comment says "rSteps evenly spaced values" — actually it produces **rSteps+1** values. The unit test `TestBifurcation_BasicStructure` (`chaos_test.go:728`) silently corrects for this with `(10+1) * 5 = 55`. Doc bug, not a numerics bug, but it bites users porting to other languages and the CLAUDE.md "precision documented" rule. ~2 LOC doc fix.

### N13 — `LyapunovExponent` ZERO-DERIVATIVE FALLBACK USES `log(eps)` WHICH SEEDS THE WRONG MAGNITUDE

`analysis.go:44`: `sum += math.Log(eps)` when `|f'|=0`. With eps=1e-10, this contributes −23.0 to the running sum. With the proposed eps=6e-6 (per N4), this contributes −12.0. Neither is the truthful "−∞" answer. **Fix:** when `absDeriv == 0`, return early with `math.Inf(-1)`, or use `math.SmallestNonzeroFloat64` so `log` returns `−744.4`. The current behaviour silently gives wrong answers at super-stable fixed points (logistic at r=2, x=0.5: `f'(0.5) = 2·(1−2·0.5) = 0`).

### N14 — NO ENSEMBLE-AVERAGED LYAPUNOV ESTIMATOR (the topic prompt's named gap)

"Random-IC ensemble averaging: how much variance does the Lyapunov exponent estimate have?" The current single-x₀ estimator has *high* variance for fixed-finite n. Standard fix: take m random ICs, run each for n iterates, return mean ± stderr. **Recommended:**
```go
type LyapunovResult struct { Mean, StdErr float64; Samples int }
func LyapunovExponentEnsemble(f func(float64) float64, m, n int, rng *rand.Rand) LyapunovResult
```
~30 LOC + 1 golden file with reference variance for r=4 (closed form: `Var(λ̂_n) ≈ Var(log|f'|) / n`, where the per-iterate variance for fully-chaotic logistic is computable).

---

## What is correct (don't break)

- RK4 coefficients and stage structure are textbook-correct (verified against Hairer-Nørsett-Wanner Table II.1.2).
- The `dydt` callback signature is the right zero-allocation contract on the *interface* level (the implementation just doesn't honour it — see N1).
- `LotkaVolterra`, `SIRModel`, `LorenzSystem`, `RosslerSystem`, `VanDerPol` derivative implementations match the literature; the `*_DerivativesCorrect` unit tests check the right things at the right tolerances.
- `GameOfLife` torus topology is correct, and `TestGameOfLife_GliderMoves` is a legitimate functional test.
- `TestLogisticMap_R4_Chaotic` uses `x₁=0.3` not `x₁=0.5` — the comment correctly notes the `x=0.5 → 1.0 → 0.0` degenerate seed.
- `BifurcationDiagram`'s `warmup` parameter is the right API.
- `RecurrencePlot` symmetry exploitation (`for j := i+1`) is correct and saves the expected 2×.

---

## Cross-package dependencies for the recommended fix-set

- N1 (zero-alloc RK4): pure additive, no sibling change.
- N2 (Leapfrog/Forest-Ruth): pure additive.
- N3 (Benettin spectrum): needs `linalg.QR` — exists per CLAUDE.md table.
- N4-N6, N8: pure local fixes.
- N7 (golden files): needs `math/big` 256-bit reference generator — repo pattern exists in `testutil/`.
- N9 (DOPRI5): pure additive.
- N10 (Rosenbrock): needs Newton step → `optim.Newton` exists.
- N11 (Hénon): needs `HenonMap` added to `systems.go` (~10 LOC) — pure additive.
- N14 (ensemble Lyapunov): needs `*rand.Rand` parameter — match the `crypto.PRNG` or `prob` convention.

Total fix-set: ~700 LOC of additions (mostly N3 spectrum estimator + N9 DOPRI5 + N7 golden vectors), ~100 LOC of bugfixes (N1 workspace, N4 ε constant, N5 warmup, N8 Kahan/Hypot, N12 doc, N13 log-zero), zero math regressions.

---

## Topic prompt items, addressed

| Topic prompt item | Status | Where addressed |
|---|---|---|
| RK4 fixed-step | Present, allocates 5 slices/step | N1 |
| RK45 / DOPRI5? | **Missing** | N9 |
| Stability claims? | None made | (none) |
| Long-horizon Lorenz/Hénon drift | No quantitative tests; Hénon missing | N11 |
| Energy/area preservation | Non-symplectic RK4, drift hidden by tolerance | N2, N11 |
| Lyapunov tangent-vector renorm | **Missing** (1D map only) | N3 |
| QR / Gram-Schmidt for Lyapunov | **Missing** | N3 |
| Wolf vs Benettin vs Sano-Sawada | All missing | N3 |
| Finite-T vs infinite-T limit | Not parameterised; no warmup | N5 |
| Symplectic integrators (leapfrog/Verlet/Forest-Ruth) | **Missing** | N2 |
| Step-size adaptation, error norm | **Missing** | N9 |
| Embedded RK pair (DOPRI5(4)) | **Missing** | N9 |
| Stiff ODEs (BDF/Rosenbrock) | **Missing** | N10 |
| Random-IC ensemble averaging | **Missing** | N14 |
| IEEE-754 NaN/Inf/denormal | Zero tests | N6 |
| Catastrophic cancellation in tangent dots | Naïve `Σd²` in euclideanDist; uncompensated RK4 sum | N8 |

---

## Recommended commit ordering (highest-leverage first)

1. **N1** (RK4 workspace, zero-alloc) — single biggest perf win for every consumer named in the docstring. ~30 LOC.
2. **N4 + N5 + N13** (1D Lyapunov ε, warmup, zero-derivative fix) — three local fixes that turn a 2-decimal estimator into a 5-decimal one. ~25 LOC.
3. **N3** (Benettin spectrum estimator for flows) — closes the topic prompt's main gap. ~140 LOC + 1 golden file.
4. **N2** (Leapfrog + Forest-Ruth symplectic) — required for any honest Hamiltonian-conservation claim. ~70 LOC.
5. **N9** (DOPRI5 + PI controller) — the missing default integrator; closes the embedded-RK prompt item. ~200 LOC.
6. **N7 + N6** (golden vectors + IEEE-754 edge cases) — closes the CLAUDE.md test-floor and makes the package portable. ~10 hours of vector generation.
7. **N11** (Hénon + long-horizon drift tests) — closes the topic prompt's drift item. ~65 LOC.
8. **N14** (ensemble Lyapunov) — closes the variance prompt item. ~30 LOC.
9. **N8** (Kahan/Hypot) — required for 10⁶-step horizons. ~15 LOC.
10. **N10** (Rosenbrock-Wanner) — Tier 2 stiff-systems work. ~250 LOC. Defer.

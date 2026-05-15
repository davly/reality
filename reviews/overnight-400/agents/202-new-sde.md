# 202 — New Math: Stochastic Differential Equations (Block C, slot 2)

**Summary line 1:** reality v0.10.0 ships **zero** SDE machinery — repo-wide grep on `stochastic|brownian|wiener|milstein|euler.maruyama|MLMC|GBM|Heston|Ornstein|Itô|Stratonovich` returns only doc strings and review files; the closest substrate is `chaos/ode.go` (RK4 + Euler deterministic ODE), `prob/distributions.go` (Normal/Exp/Beta/Gamma/Poisson PDF+CDF+quantile, **no `Sample` API**), and `timeseries/garch/` (a discrete-time variance recursion that simulates the *closest cousin* of an SDE without ever solving one).
**Summary line 2:** Twenty-two ranked primitives S1-S22 totalling ~3,800 LOC would establish a `sde/` package on top of a tiny `prob/random.go` (~120 LOC) sampling layer; cheapest one-day shippable artifact is S1 (BrownianPath + EulerMaruyama, ~250 LOC), single-highest-leverage is S5 (MLMC for Giles-2008 O(ε⁻²) variance reduction — order-of-magnitude speedup over plain Monte Carlo for SDE expectations and the *only* cutting-edge piece on this list with no zero-dep blocker).

---

## (1) What reality ships today (verified at v0.10.0)

**SDE machinery: nothing.** Every appearance of "stochastic" / "brownian" / "wiener" in a non-review .go file is either:

- A doc-comment ("Uniform density: stochastic behavior" in `chaos/analysis.go:115` describing a recurrence-plot interpretation)
- The unrelated *Wiener filter* in `audio/separation/wiener.go` (signal-processing Wiener-Hopf, not the Wiener process)
- GARCH(1,1) variance recursion in `timeseries/garch/`, which is the discrete-time analog of a Heston-like volatility but is *not* an SDE solver
- Test files that consume `math/rand.NormFloat64` for synthetic data generation

**Closest substrate to an SDE solver:**

| Substrate | LOC | What it provides | Gap to SDE |
|---|---|---|---|
| `chaos/ode.go` | 132 | `RK4Step`, `EulerStep`, `SolveODE` deterministic dy/dt = f(t, y); pre-allocated output slices, 0-alloc design | **No noise term** — needs σ(t, y)·dW added to the drift step |
| `chaos/systems.go` | (Lorenz, Van der Pol, etc) | Deterministic chaotic ODEs | Stochastic-perturbation variants (noisy Lorenz, OU, CIR) absent |
| `prob/distributions.go` | 478 | Normal/Exp/Beta/Gamma/Poisson PDF+CDF+quantile | **No `Sample` API anywhere in repo** — every test that needs Gaussian samples imports `math/rand.NormFloat64` directly |
| `prob/markov.go` | (steady state, mixing) | Discrete Markov chains | Continuous-time SDE is the natural extension; no bridge |
| `timeseries/garch/` | 8 files | GARCH(1,1) σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} + Fit by L-BFGS | Discrete-time analog of CIR; the SDE limit ("GARCH diffusion", Nelson 1990) is the natural extension |
| `crypto/` PRNGs | (per slot 175) | Splitmix, Xoshiro, etc | Available as RNG source; not exposed as `prob.RNG` interface |
| `optim/transport/sinkhorn.go` | 247 | Discrete OT, would consume the JKO step from slot 201's Tier-2 | OT-x-SDE coupling lights up via Wasserstein gradient flow / Schrödinger bridge |

**Critical missing prerequisite — Gaussian sampling layer.** Every SDE scheme begins with √dt · Z where Z ~ N(0, 1). reality has no canonical `prob.SampleNormal(rng) float64` function. Today every consumer rolls its own via `rand.NormFloat64()` (Go stdlib's polar-method Box-Muller). This is fine for tests but inadequate for: (a) cross-language golden-file parity (Go's polar Box-Muller is not bit-reproducible against numpy's Ziggurat), (b) SIMD vectorisation of an SDE batch, (c) low-discrepancy sequences (QMC paths — Sobol normals, slot S15 below). **Slot 202 must ship `prob/random.go` as a 120-LOC dependency before any SDE primitive lands.**

**v2 deferral roster from chaos/ode.go and prob/distributions.go:** neither file has a deferral roster. SDE simply isn't on reality's roadmap as of v0.10.0. Slot 027 (chaos-missing) does not enumerate stochastic extensions — it focuses on adaptive step ODE solvers (RK45, DOPRI8) and deterministic-chaos analyses. Slot 117 (prob-missing) lists distribution sampling as an open gap. **This review surfaces the entire SDE corpus.**

---

## (2) What's missing — twenty-two primitives ranked by demand

Demand ranking weights: (a) explicit consumer in CONTEXT.md / CLAUDE.md, (b) frequency in Kloeden-Platen-1992 / Glasserman-2003 / Higham-2001 textbook chapters, (c) connective-tissue readiness (substrate ready in repo).

### Tier-0 — substrate (~270 LOC, blocks everything below)

#### S0a. `prob/random.go` — Gaussian / exponential / gamma / Poisson samplers — ~120 LOC
**Blocks every SDE primitive.** Defines the canonical RNG interface and 4 primary samplers:

```go
type RNG interface { Float64() float64; Uint64() uint64 }

func SampleNormal(r RNG) float64           // Marsaglia polar (matches Go stdlib semantics)
func SampleNormalZiggurat(r RNG) float64   // Ziggurat (cross-language parity with numpy)
func SampleExponential(r RNG, lambda float64) float64
func SampleGamma(r RNG, shape, scale float64) float64   // Marsaglia-Tsang 2000
func SamplePoisson(r RNG, lambda float64) int           // Knuth for λ<30, PTRS for λ≥30
func FillNormalsAntithetic(r RNG, out []float64)        // antithetic variates (slot S20)
```

Cross-language pin: golden-file 10⁶ samples → empirical mean/var/skew/kurt within 4σ of theoretical. Polar Box-Muller deliberately matches Go-stdlib so existing tests don't break; Ziggurat ships as the parity-canonical default.

#### S0b. `prob/qmc.go` — low-discrepancy sequences — ~150 LOC
Sobol, Halton, Faure for QMC path generation (S15 below). Sobol is the workhorse: 2-d direction-number tables for n ≤ 20 dimensions ship inline; Joe-Kuo 2008 directional numbers for higher d. Box-Muller-on-Sobol is the standard QMC normals trick.

### Tier-1 — high demand, short connective tissue (~1,250 LOC)

#### S1. Brownian path + Euler-Maruyama — ~250 LOC
**The foundational entry point.** Standard discretisation of dX_t = μ(t, X_t) dt + σ(t, X_t) dW_t:

`X_{n+1} = X_n + μ(t_n, X_n)·Δt + σ(t_n, X_n)·√Δt·Z_n,  Z_n ~ N(0,1)`

Strong order 0.5, weak order 1 (Kloeden-Platen 1992 Theorem 9.6.2 / 14.5.1). Mirrors `chaos/EulerStep` API — caller-provided drift `mu` and diffusion `sigma`, pre-allocated output. Multi-dimensional variant uses `[]float64` state and `[][]float64` Cholesky-factored diffusion matrix.

```go
type SDEFunc func(t float64, x []float64, drift []float64)
type DiffusionFunc func(t float64, x []float64, sigma [][]float64)

func EulerMaruyamaStep(mu SDEFunc, sigma DiffusionFunc, t float64,
    x []float64, dt float64, rng RNG, out []float64)
func SimulateBrownianPath(t0, tEnd, dt float64, d int, rng RNG) [][]float64
```

Connective-tissue: `chaos/ode.go` style API + `prob/random.go` Gaussian draws. Cross-substrate parity: Go reference path → C# / Python validates against same RNG seed (Ziggurat normals).

#### S2. Milstein scheme — ~200 LOC
Strong order 1.0 — first improvement over Euler-Maruyama. Adds the Itô-Taylor correction term `½·σ·∂σ/∂x·(ΔW² − Δt)` to absorb the leading O(√Δt) error. For autonomous 1-D dX = μ(X)dt + σ(X)dW:

`X_{n+1} = X_n + μ(X_n)·Δt + σ(X_n)·ΔW + ½·σ(X_n)·σ'(X_n)·(ΔW² − Δt)`

Multi-dimensional Milstein needs the diffusion-derivative tensor ∂σ/∂x and Lévy areas; ship 1-D first (covers GBM, OU, CIR) and defer multi-D Milstein to S2b. Caller supplies `sigmaDeriv` callback or autodiff via `autodiff/` (cross-link to slot 168 physics-autodiff).

```go
func MilsteinStep1D(mu, sigma, sigmaDeriv func(float64) float64,
    x, dt float64, rng RNG) float64
```

#### S3. Stochastic Runge-Kutta (Rößler 2010 SRA / SRI schemes) — ~350 LOC
Butcher-tableau-style explicit SRK schemes. **SRA1** for additive noise (σ depends only on t) achieves strong order 1.5, weak order 2. **SRI1** for general Itô SDEs achieves strong order 1, weak order 2 *without* requiring derivatives of σ (the Milstein deal-breaker for general consumers). Tableaux are 4-stage; encode as `[][]float64 A, B, alpha, beta1, beta2` constants matching Rößler 2010 Table 1.

```go
func SRA1Step(mu SDEFunc, sigma SDEFunc, ...) // additive
func SRI1Step(mu SDEFunc, sigma DiffusionFunc, ...) // general
```

#### S4. Geometric Brownian motion (closed-form + path) — ~150 LOC
The Black-Scholes substrate. dS = μS dt + σS dW has the exact solution

`S_t = S_0 · exp((μ − ½σ²)t + σW_t)`

ships as both a closed-form `GBMSampleExact(s0, mu, sigma, t, rng)` (zero-bias, single Gaussian draw) and a numerically-integrated `GBMPath(s0, mu, sigma, t0, tEnd, dt, rng)` for path-dependent payoffs. Cross-substrate parity is exact closed-form vs Euler/Milstein convergence — golden file includes both at multiple Δt to *demonstrate* the strong-order claims. **Educational killer feature** if reality wants to teach SDE convergence.

#### S5. Multi-Level Monte Carlo (Giles 2008) — ~300 LOC ⭐
**The single highest-leverage primitive.** Standard MC for E[f(X_T)] costs O(ε⁻³) (need ε⁻² paths each at Δt = ε). Giles MLMC drops this to O(ε⁻²·log²ε) via telescoping:

`E[P_L] = E[P_0] + Σ_{ℓ=1}^L E[P_ℓ − P_{ℓ-1}]`

each fine-coarse difference run on **paired Brownian increments** (coarse Δt = M·fine Δt, M typically 4). The Δℓ = P_ℓ − P_{ℓ-1} estimator has variance V_ℓ that *decays* with ℓ for strong-order-1 schemes (Milstein and above), so the optimal sample budget allocates few paths to fine levels and many to coarse. Adaptive level-selection via Giles' optimal allocation `N_ℓ ∝ √(V_ℓ / C_ℓ)`.

```go
type MLMCEstimator struct {
    Drift, Diffusion SDEFunc
    Payoff func(path [][]float64) float64
    Scheme SDEScheme  // EulerMaruyama / Milstein / SRI1
    MaxLevel int      // L
    M int             // refinement factor (usually 4)
}
func (m MLMCEstimator) Run(epsilon float64, rng RNG) (mean, stderr float64, levelCounts []int, err error)
```

Cross-substrate parity: pin GBM E[max(S_T − K, 0)] (Asian / European call payoff) at multiple ε against Black-Scholes closed-form. Cross-package value: **prob/MonteCarlo.go** — the very first MC user inside reality (so far reality has no `prob.MonteCarlo` since it's a pure-math library, but MLMC is the natural delivery surface). MLMC is a Wilks-2018 ICM laudation-grade idea and the **archetypal "cutting-edge math worth shipping"** primitive on this list.

#### S6. Drift-implicit Euler (Hairer-Wanner stiff) — ~200 LOC
Stability for stiff SDEs (CIR with high mean-reversion κ, large σ). Solves `X_{n+1} = X_n + μ(X_{n+1})·Δt + σ(X_n)·√Δt·Z_n` with Newton iteration on the implicit drift. A-stable for linear drifts; the only scheme that survives σ → ∞ in the OU process without exploding. Connective-tissue: reuses `optim/rootfind.go` (Newton-Raphson already exists).

### Tier-2 — high demand, medium connective tissue (~1,300 LOC)

#### S7. Ornstein-Uhlenbeck (mean-reverting, exact transition density) — ~100 LOC
dX = θ(μ − X)dt + σ dW has Gaussian transition density, so an exact step exists:

`X_{t+Δt} | X_t ~ N(μ + (X_t − μ)·e^{−θΔt}, σ²·(1 − e^{−2θΔt})/(2θ))`

ships exact + the Euler-Maruyama variant for didactic comparison. Foundational for: control noise, neural-noise models, interest-rate Vasicek model. **Educational killer:** OU transition is closed-form, so users see the Euler bias quantitatively.

#### S8. Cox-Ingersoll-Ross (CIR / square-root diffusion) — ~150 LOC
dX = κ(θ − X)dt + σ√X dW. Non-negative process iff Feller condition 2κθ ≥ σ². Exact transition is non-central chi-squared; ships **(a)** Euler with truncation `max(X, 0)` (simple, biased), **(b)** Higham-Mao-Stuart 2005 *balanced* implicit-Milstein (positivity-preserving), **(c)** exact non-central χ² sampler (Glasserman 2003 Ch. 3). The non-central χ² sampler is the cross-language parity workhorse — closed-form transition, zero-bias, validates every other scheme.

```go
func CIRSampleExact(x0, kappa, theta, sigma, dt float64, rng RNG) float64
func CIRStepEuler(...) (truncated)
func CIRStepBalancedMilstein(...) (Higham-Mao-Stuart 2005, positivity-preserving)
```

#### S9. Heston stochastic volatility — ~250 LOC
dS = μS dt + √v · S dW¹, dv = κ(θ − v)dt + σ√v dW², dW¹·dW² = ρ dt. The two-factor SDE that is the modern equity-option pricing baseline. Closed-form characteristic function (Heston 1993) ships as a separate primitive `HestonCharacteristicFunc(...)` for Carr-Madan FFT pricing. SDE simulation uses **(a)** Euler with full-truncation (Lord-Koekkoek-van-Dijk 2008), **(b)** **QE scheme** (Andersen 2008 *quadratic-exponential*) which is the industry standard. Cross-substrate parity is QE-scheme price vs FFT-characteristic-function price at 1e-3 relative.

#### S10. Merton jump-diffusion — ~200 LOC
dS = μS dt + σS dW + S·(J − 1)·dN_t with N_t Poisson(λt) and ln J ~ N(α, δ²). Closed-form European-option price is a Poisson sum of Black-Scholes terms (Merton 1976), shipping as `MertonCallPrice` reference. SDE simulation: per-Δt step decomposes into continuous (GBM step) + jump indicator (Bernoulli(λΔt)) × jump size (LogNormal). Connective-tissue: needs `SamplePoisson` from S0a.

#### S11. Heston-Bates / SVCJ stochastic-vol with jumps — ~250 LOC
Combines S9 + S10. Drives credit-VAR / option-pricing consumers. Ships as a single `BatesPath` integrated routine + the QE-scheme + jump-correlation parameter ρ_J for joint stock-vol jumps. The frontier of analytically-tractable equity SDEs.

#### S12. Brownian bridge construction — ~150 LOC
Sample W_t conditional on (W_0 = 0, W_T = b) at intermediate times. Two regimes:
- **Lévy bridge** for sequential refinement: midpoint t = (s+u)/2 has W_t ~ N(½(W_s + W_u), (u−s)/4).
- **Karhunen-Loève** truncated series for high-d / QMC pairing: W_t = Σ_{k=1}^N ξ_k · √(2T/π²·(k−½)²) · sin((k−½)πt/T).

The KL pairing is the standard QMC trick — fewer effective dimensions per path mean Sobol concentrates variance in the leading dimensions. Critical for S15 below.

#### S13. Adjoint-mode SDE differentiation — ~250 LOC ⭐
Cross-link to slot 168 physics-autodiff. Differentiable SDE solvers via **discretise-then-differentiate**: store all noise increments, replay backward through the Euler-Maruyama recurrence with `autodiff/tape.go`. Enables MLE / EM parameter inference (S17 below) and policy-gradient-style Greeks (∂E[f(X_T)] / ∂θ for option Greeks). The Li-Wong-Chen-Duvenaud 2020 *adjoint method for SDEs* is the more memory-efficient continuous-adjoint variant — ship discrete-adjoint first; defer continuous-adjoint to v2.

```go
type DiffSDETape struct { ... }
func (d *DiffSDETape) BackwardGreeks(payoff func([]float64) float64, theta []float64) []float64
```

### Tier-3 — niche / advanced (~600 LOC)

#### S14. Lévy α-stable processes — ~150 LOC
Pure-jump processes with stable distributions. dX = γ dt + dL_t with L Lévy α-stable. Chambers-Mallows-Stuck 1976 sampling for general α ∈ (0, 2]. Drives heavy-tail finance / anomalous-diffusion consumers. Cauchy (α=1) and Gaussian (α=2) are special cases that should pin against existing distributions.

#### S15. Quasi-Monte Carlo path-dependent pricing — ~200 LOC
Combines S0b Sobol with S12 Brownian bridge. Estimates E[P(X_·)] for path-dependent P with O(N^{-1+ε}) convergence vs. MC's O(N^{-½}). Critical for Asian / barrier / lookback options. Connective-tissue: glues S0b + S12 + S5 (or vanilla MC).

#### S16. Random ODEs (RODE) — ~120 LOC
dX/dt = f(t, X, η_t) with η_t a (typically piecewise-constant) random process. Differs from SDEs in that the noise enters the *drift*, not as a Brownian perturbation, and the equation can be solved pathwise as a deterministic ODE. Drives biological / physiological model consumers (random parameters drawn each batch). Reuses `chaos/RK4Step` directly; only adds an η-sampling layer.

#### S17. Parameter inference for SDEs (MLE + EM via particle filter) — ~400 LOC
Given observations Y_{t_i} = X_{t_i} + ε_i (noisy), recover θ in dX = μ_θ(X)dt + σ_θ(X)dW. Two-stage:
- **Pseudo-likelihood MLE** (Aït-Sahalia 2002 Hermite expansion): closed-form for OU/CIR/GBM, Hermite-polynomial expansion for general diffusions.
- **Particle EM** (Doucet-Kantas-Singh-Maciejowski 2009): bootstrap particle filter for E-step likelihood + L-BFGS for M-step. Connective-tissue: reuses `optim/gradient.go` L-BFGS, slot 138 timeseries-sota's particle-filter-pending stub.

This is the SDE *consumer* killer feature — without it, an SDE library is a one-way "simulate forward" tool. With it, reality becomes a *full Bayesian SDE inference toolkit*.

#### S18. Filtering for SDEs (Kushner-Stratonovich / Zakai) — ~200 LOC ⊘ defer
Kushner-Stratonovich PDE for the conditional density given continuous observations. Numerically intractable in d > 3 without further tricks (extended Kalman = linearise; particle filter = S17 above; ensemble Kalman = high-d). Ship Kushner-Stratonovich as an *educational reference* in 1-D only (~100 LOC), defer the fully-general implementation. Particle filter (S17) is the *practical* substitute.

#### S19. Backward SDEs (BSDEs) — ~200 LOC ⊘ defer
dY_t = −f(t, Y_t, Z_t) dt + Z_t · dW_t with terminal condition Y_T = ξ. The cornerstone of nonlinear option pricing and stochastic optimal control via the Hamilton-Jacobi-Bellman PDE link. Numerically: Bouchard-Touzi-Warin 2004 least-squares Monte Carlo. **Defer** unless an explicit pricing-finance consumer pulls; the implementation is high-LOC and the consumer surface today is empty.

### Tier-4 — variance-reduction toolkit (~280 LOC)

#### S20. Antithetic variates — ~50 LOC
For each path generated with noise increments {ΔW_n}, also generate its mirror {−ΔW_n} and average. Halves variance for any *odd* function of W_T (e.g., E[max(S_T − K, 0)] for ATM call). Ships as a wrapper over the chosen scheme: `AntitheticEulerMaruyama(...)`.

#### S21. Control variates — ~80 LOC
Reduce variance by subtracting a related variable with known mean: `Y' = Y − β(X − E[X])`. For Asian options on GBM, the geometric average has a closed-form mean (Kemna-Vorst 1990) and serves as a near-optimal control for the arithmetic-average payoff. Ships `ControlVariateEstimator` taking (path-payoff, control-payoff, control-mean).

#### S22. Importance sampling for rare-event SDEs (Glasserman-Heidelberger-Shahabuddin 1999) — ~150 LOC ⊘ defer
Tilt the drift to make the rare event typical, reweight by Radon-Nikodym derivative (Girsanov). Critical for default-probability / out-of-the-money option pricing. Defer until S5 MLMC ships (MLMC + IS is the production combo).

---

## (3) Connective tissue — what each new edge buys

Eight cross-package edges activate once `sde/` lands:

| Edge | LOC of glue | What it unlocks |
|---|---|---|
| `sde/ → chaos/` (RK4 substrate reuse) | 0 — already callable | Stochastic Lorenz / Van der Pol / Lotka-Volterra; ecosystem-noise studies |
| `sde/ → prob/random.go` (S0a) | 120 | All other repo packages get a canonical Gaussian sampler; closes the `rand.NormFloat64` ad-hoc-import pattern |
| `sde/ → optim/rootfind.go` (Newton for S6 implicit) | 0 — already callable | Drift-implicit / fully-implicit schemes for stiff SDEs |
| `sde/ → autodiff/` (S13 differentiable) | 60 — tape adapter | Greeks, MLE gradients, neural-SDE training (cross-link slot 168) |
| `sde/ → optim/gradient.go` (S17 L-BFGS for MLE) | 30 — adapter | SDE parameter inference closes the loop from "simulate" to "infer" |
| `sde/ → optim/transport/` (slot 201 Tier-2 JKO) | 0 — JKO consumer | Wasserstein gradient flow recovers Fokker-Planck PDE; Schrödinger bridge needs SDE machinery |
| `sde/ → timeseries/garch/` (Nelson-1990 GARCH-diffusion limit) | 50 — calibrator | GARCH(1,1) ↔ Heston connection; cross-validate parameters |
| `sde/ → topology/persistent/` (S5 MLMC for stochastic-process persistence) | 0 — defer to consumer pull | Persistent-homology of SDE level sets |

Two **new packages** appear if the full roster ships: `sde/` (~3,800 LOC) and a tiny `prob/random.go` + `prob/qmc.go` substrate (~270 LOC). No existing package needs an API break.

---

## (4) Three architectural recommendations

**F1. Ship `prob/random.go` (S0a) as a separate PR before any SDE primitive.** Every existing test that imports `math/rand` directly should migrate to `prob.RNG`. This is a 4-line change per test file and unblocks cross-language sampler-parity (today reality has *zero* sampling parity tests because there's no canonical sampler). One-day effort. **Must land first.**

**F2. Establish `sde.SDEScheme` as the canonical interface for solver selection.**

```go
type SDEScheme int
const (
    EulerMaruyama SDEScheme = iota
    Milstein
    SRA1
    SRI1
    DriftImplicitEuler
)
```

Every solver call routes through this enum. MLMC (S5) consumes it. Avoids the API explosion of `EulerMaruyamaPath`, `MilsteinPath`, `SRI1Path`, `EulerMaruyamaMLMC`, `MilsteinMLMC`. Pattern-matches `optim/transport/sinkhorn.go`'s `SinkhornResult{Plan, Cost, Iterations}` clean-result idiom.

**F3. Pin convergence-order claims via golden files, not just function correctness.** Strong order 0.5 (Euler-Maruyama), strong order 1.0 (Milstein), strong order 1.5 (SRA1) are *the* SDE-textbook claims. Golden file ships GBM E[|X_T^Δt − X_T^exact|²] at Δt ∈ {1/16, 1/32, 1/64, 1/128, 1/256} averaged over 10⁵ paths and validates the log-log slope is the claimed order ± 0.05. **This is the cross-language parity contract that proves the implementations are correct, not just numerically reasonable.** Kloeden-Platen 1992 § 9.6 / 14.5 is the citation.

---

## (5) Risks and gotchas

- **G1. Brownian-increment storage for MLMC.** MLMC pairs coarse + fine paths on the *same* Brownian path. Naïve implementation regenerates increments per level and breaks the variance-decay property. Ship `BrownianIncrementCache` shared between coarse and fine simulators.
- **G2. CIR positivity violation in vanilla Euler.** Without truncation `X ← max(X, 0)` the Euler step crosses zero and √X becomes NaN. Ship the truncated variant as the default and the Higham-Mao-Stuart balanced-Milstein as the principled alternative; document loud-fail when a user picks vanilla Euler.
- **G3. Ziggurat vs polar Box-Muller cross-language drift.** Go stdlib `math/rand.NormFloat64` uses *polar* Box-Muller; numpy uses Ziggurat. Cross-language golden-file parity *must* fix on Ziggurat. Ship Ziggurat with a deterministic, language-portable normal-table (256 layers) hard-coded as a constant array.
- **G4. Milstein derivative requirement.** Multi-D Milstein needs ∂σ/∂x — analytically supplied or via autodiff. Most consumers will not know this. Default to SRI1 (S3) which has the same strong order without needing derivatives.
- **G5. Heston Feller condition.** 2κθ < σ² makes the variance process hit zero with positive probability; the QE scheme handles this gracefully but full-truncation Euler does not. Document loud-fail; provide `HestonValidate(kappa, theta, sigma) error` that warns when Feller is violated.
- **G6. MLMC variance-decay assumption is scheme-dependent.** MLMC + Euler-Maruyama gives O(ε⁻²·log²ε) only when the payoff is Lipschitz; for max-payoffs (option pricing) the variance decays slower and MLMC + Milstein is required. Document the regime per scheme; expose `MLMCEstimator.RecommendedScheme(payoff PayoffType)`.
- **G7. Brownian-bridge KL truncation error.** Karhunen-Loève for QMC paths needs N ≈ 32+ basis functions for sub-percent error on path-dependent payoffs. Document the truncation-error vs effective-dimension tradeoff.

---

## (6) Cross-language parity targets

Eight pinned tests covering the strong/weak-order claims, exact-transition regimes, and MLMC convergence:

| Test | Pin | Tolerance | Reference |
|---|---|---|---|
| `TestEulerMaruyamaStrongOrder_GBM` | log-log slope of E[|X^Δt − X^exact|²] vs Δt = 0.5 ± 0.05 | 5% on slope | Kloeden-Platen 1992 § 9.6 |
| `TestMilsteinStrongOrder_GBM` | log-log slope = 1.0 ± 0.05 | 5% on slope | Kloeden-Platen 1992 § 10.3 |
| `TestSRI1StrongOrder_GBM` | log-log slope = 1.0 ± 0.05 | 5% on slope | Rößler 2010 Table 1 |
| `TestOUExactTransition` | E[X_T] / Var[X_T] vs closed-form | 1e-12 | Karatzas-Shreve 1991 |
| `TestCIRNoncentralChi2` | empirical CDF vs non-central χ² CDF at multiple T | KS-stat < 0.01 over 10⁶ draws | Glasserman 2003 § 3.4 |
| `TestGBMExactVsEuler` | bias decreases as O(Δt) | 5% | Higham 2001 |
| `TestMLMC_GBMCallOption_OvsBlackScholes` | MLMC mean within ε of BS-closed-form | ε = 1e-3 with 95% CI coverage | Giles 2008 |
| `TestHestonQE_VsCharacteristicFunctionFFT` | QE-scheme price vs Carr-Madan FFT | 1e-3 relative | Andersen 2008 |

---

## (7) Verdict

**Ship Tier-0 + Tier-1 (~1,520 LOC over 4-5 sprints):**
- Sprint 1: S0a `prob/random.go` (120 LOC) + S0b `prob/qmc.go` (150 LOC) — substrate
- Sprint 2: S1 EulerMaruyama (250) + S4 GBM (150) — proves the API on the canonical example
- Sprint 3: S2 Milstein (200) + S6 DriftImplicitEuler (200) — order-1 strong + stiff stability
- Sprint 4: S3 SRI1/SRA1 (350) — derivative-free order-1
- Sprint 5: S5 MLMC (300) — the cutting-edge ⭐ piece

**Defer-but-design Tier-2 (~1,300 LOC, ship when consumer pulls):** S7 OU, S8 CIR, S9 Heston, S10 Merton, S11 Bates, S12 BrownianBridge, S13 AdjointSDE.

**Drop until consumer pulls:** S18 Kushner-Stratonovich filtering, S19 BSDEs, S22 Importance Sampling. Each is a research-frontier capability whose consumer doesn't exist in reality's downstream stack today.

**Single-highest-leverage 1-day project:** S1 (EulerMaruyama + Brownian path) on top of S0a sampler, ~370 LOC total. Unblocks every other primitive on this list, lights up `chaos/` stochastic-perturbation studies, and gives reality its first serious stochastic-simulation surface.

**Single-highest-leverage cutting-edge piece:** S5 MLMC — Giles 2008 is a canonical "this is why we have research math" idea (drops MC complexity from O(ε⁻³) to O(ε⁻²·log²ε), an order-of-magnitude wall-clock saving), the implementation is ~300 LOC of pure scheduling logic over existing simulators, and it's the *one* piece on this list that no off-the-shelf math library ships well (Python's QuantLib has it; numpy/scipy do not). **MLMC is the flagship deliverable for this slot.**

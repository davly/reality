# 173 | synergy-queue-prob

**Topic:** queue x prob — heavy-tailed service, fluid limits, Brownian approximation, Pollaczek-Khinchine, Lindley, Kingman, RBM, renewal theory, polling, priority, stochastic ordering, Palm calculus
**Block:** B (cross-package synergies)
**Date:** 2026-05-08
**Scope:** capabilities that emerge ONLY when `queue/`, `prob/`, and (light) `chaos/` are composed; not what either package is missing in isolation (covered by per-package agents 121-125 / 116-120). Repo at v0.10.0, 1965 tests passing.

## Two-line summary

Today `queue/` ships exactly one service distribution (M, exponential) across 8 functions / 4 files (~480 LOC) and `prob/` owns Normal/Exponential/Uniform/Beta/Poisson/Gamma/Binomial CDFs + `Distribution` interface + `MarkovSteadyState` + LCG-only — but they don't compose: there is **zero general-service queue, zero P-K formula, zero Lindley/Kingman, zero heavy-tailed (Pareto/Weibull/lognormal) distribution in either package, zero Brownian/RBM, zero renewal function, zero priority/polling/M^X/MMPP**, verified by zero matches across both packages on `Pollaczek|Kingman|Lindley|Pareto|Weibull|Lognormal|Brownian|Reflected|Renewal|Wald|Palm|Polling|Priority|Batch|MMPP`. **Sixteen synergy primitives (Q1-Q16) totalling ~2,460 LOC of pure connective tissue** close the gap; cheapest one-day PR is **Q1 `MG1` (Pollaczek-Khinchine)** at ~50 LOC because it consumes only `(λ, E[S], C²_s)` scalars and unlocks Q2-Q5 (M/D/1, M/E_k/1, M/H_k/1, GI/G/1 Kingman); highest-leverage architectural lift is **Q9 `ReflectedBrownianMotion` + Q8 `KingmanGG1`** (~360 LOC together) since Kingman's bound is the universally-cited heavy-traffic approximation when neither A nor S is exponential and RBM is the diffusion limit Q5 (Iglehart-Whitt 1970) of every queueing model — without these, *every* G/G/* analysis in `reality` is unreachable.

---

## Bases — what each package exposes today

`queue/` (~480 LOC, agent 121/122): **single-service-distribution-only**.
- `MM1, MMc, MM1K, ErlangB, ErlangC, ErlangCWaitTime, JacksonNetwork, LittlesLaw, BurstinessIndex, OfferedLoad` — all assume exponential service.
- `BurstinessIndex` (`metrics.go:26`) **already computes C² of inter-arrival times** — the single number Pollaczek-Khinchine and Kingman both consume.
- No M/G/1, no M/D/1, no priority, no batch, no polling, no renewal, no fluid/diffusion limit, no Brownian, no Wiener, no Palm.

`prob/` (~14 files, agents 116-120): exposes
- `Distribution` interface (PDF, CDF) with `BetaDist, NormalDist, ExponentialDist, UniformDist`.
- `NormalCDF/PDF/Quantile, ExponentialCDF/PDF/Quantile, GammaPDF/CDF, PoissonPMF/CDF, BinomialPMF/CDF, BetaPDF/CDF, UniformPDF/CDF`.
- `LogGamma, RegularizedBetaInc, regularizedGammaLowerSeries` (private), `chiSquaredCDF` (private), `studentTCDF` (private).
- `MarkovSteadyState` (power-iteration), `MarkovSimulate` (LCG-based).
- **Absent (verified):** Pareto, Weibull, Lognormal, Erlang-k, Hyperexponential, Phase-type, MGF, Laplace transform, Wald lemma, renewal function, RBM, Brownian motion increments. Agent 117 already enumerates these as missing-in-isolation.

`chaos/` ships RK4/Lorenz/Lyapunov but no Wiener/Brownian/SDE primitive (agent 027 names this gap).

`constants/` provides nothing queue-specific.

**Cross-edges between queue/ and prob/ today: ZERO.** `queue/` imports only `math`. `prob/` does not know `queue/` exists. No glue file, no `queue.AdvancedXxx`, no `prob/queueing.go`. This is the cleanest synergy in the 173-of-400 series so far — both bases are stable, neither depends on the other, and the unlock is pure new code.

## The conceptual unlock — three lemmas connecting both packages

1. **Pollaczek-Khinchine (1932/1953):** Lq = ρ²(1+C²_s)/(2(1-ρ)) for any M/G/1 queue. The right-hand side needs ONLY `(λ, E[S], Var[S])` — three scalars `prob/` already produces from any `Distribution` via numerical moments (or closed-form for the named distributions). `queue/` ships `BurstinessIndex` for inter-arrival C²_a, but P-K consumes service C²_s. Closing this loop is ~50 LOC.

2. **Kingman's bound (1961):** Wq ≈ (ρ/(1-ρ)) · ((C²_a + C²_s)/2) · (1/μ) for GI/G/1. This is the universal approximation when *neither* arrival nor service is exponential. `BurstinessIndex` already gives C²_a from inter-arrival samples; `prob/` gives C²_s from any service distribution. Closing this loop is ~30 LOC on top of P-K.

3. **Heavy-Traffic Limit Theorem (Iglehart-Whitt 1970):** as ρ→1, the (centered, rescaled) queue-length process converges in distribution to a Reflected Brownian Motion (RBM) on [0,∞) with drift -μ(1-ρ) and variance σ². The stationary distribution of RBM is exponential with rate 2(1-ρ)/(C²_a+C²_s)·(1/E[S]). `prob/ExponentialQuantile` already computes the inverse — composing it with the RBM-stationary parameter is ~80 LOC.

These three lemmas connect every Tier-1 primitive below.

---

## Q1 — `MG1` (Pollaczek-Khinchine) — cheapest day-one unlock

```go
// MG1 returns steady-state metrics for an M/G/1 queue given the first two
// service-time moments. cv2Service = Var[S]/E[S]^2 (squared coefficient of
// variation). Reduces to MM1 when cv2Service==1, MD1 when cv2Service==0.
func MG1(lambda, eService, cv2Service float64) (Lq, Wq, L, W, rho float64)
```

**Composition:** `queue/MG1` consumes `(lambda, eService, cv2Service)` scalars only. The three scalars come from `prob/` either closed-form (e.g. `prob.GammaDist{k, theta}` → `eService=k*theta, cv2=1/k`) or via numerical moments. Pollaczek-Khinchine itself is one line.

**Connective tissue LOC:** ~50 (impl) + ~80 (golden-file table M/D/1 / M/E_k/1 / M/H_2/1 / M/M/1 cross-checks) = ~130.

**Saturation pin (R-MUTUAL-CROSS-VALIDATION 3/3):** at λ=0.8, μ=1
- M/M/1 closed form: Lq = ρ²/(1-ρ) = 3.2
- M/D/1 closed form: Lq = ρ²/(2(1-ρ)) = 1.6
- M/G/1 with cv2=1 must equal M/M/1; with cv2=0 must equal M/D/1; with cv2=2 must equal exactly Lq = ρ²·(1+2)/(2(1-ρ)) = 4.8.

This is the matching-saturation idiom from commit 6a55bb4 (audio/onset 3-detector) and 365368a (Clayton autodiff vs analytic).

---

## Q2 — `MD1` (deterministic service) — 25 LOC

```go
func MD1(lambda, mu float64) (Lq, Wq, L, W, rho float64) {
    return MG1(lambda, 1.0/mu, 0.0)
}
```

Wraps Q1 with `cv2Service=0`. Models token buckets, fixed-time pipelines, polled samplers. Numerically: Lq M/D/1 is exactly half of Lq M/M/1 at the same ρ — which is why "exponential is conservative for capacity" is a textbook rule of thumb.

**Connective tissue LOC:** ~25.

---

## Q3 — `MEk1` (Erlang-k service) — 60 LOC

Erlang-k is a sum of k iid exponentials, used to model "less variable than exponential" service (e.g. multi-stage processing). For Erlang-k, C² = 1/k.

```go
func MEk1(lambda, mu float64, k int) (Lq, Wq, L, W, rho float64) {
    cv2 := 1.0 / float64(k)
    return MG1(lambda, 1.0/mu, cv2)
}
```

Plus a `prob.ErlangPDF/CDF` (Gamma with integer shape k) — but **`prob.GammaPDF` already covers it** (`distributions.go:359`). Q3 is therefore zero-new-distribution glue.

**Connective tissue LOC:** ~60 (impl + 40-vector golden file checking k=1 → MM1, k=∞ → MD1 limit).

---

## Q4 — `MHk1` (Hyperexponential-k) — 90 LOC

Hyperexponential models "more variable than exponential". With k branches, weights p_i, rates μ_i: E[S]=Σp_i/μ_i, E[S²]=2Σp_i/μ_i², cv²=E[S²]/E[S]²−1 (≥1, =1 only when all branches equal). `MHk1(lambda, weights, rates)` computes E[S], cv²_S from inputs, dispatches to Q1. Pin: H_2 vs Gross/Shortle Ch. 6.

---

## Q5 — Heavy-tailed service primitives (`ParetoDist`, `WeibullDist`, `LognormalDist`) in `prob/`

The whole point of "heavy-tailed service" is that **the second moment may be infinite** — at which point Pollaczek-Khinchine breaks (Wq diverges) and Kingman's bound is meaningless. The right object is the *tail asymptotics*: P(Wq > x) ~ ρ/(1-ρ) · F̄_e(x) where F̄_e is the equilibrium (residual-life) tail of the service distribution (Asmussen 2003 Ch. X).

`prob/` has **no Pareto, Weibull, or Lognormal today** — agent 117 already names this gap. Adding the three is ~280 LOC total in `prob/distributions.go` (PDF + CDF + Quantile + log-PDF for each).

```go
// prob/heavy_tailed.go (new file, ~280 LOC)
func ParetoPDF(x, alpha, xm float64) float64       // alpha-shape, xm-scale
func ParetoCDF(x, alpha, xm float64) float64
func ParetoQuantile(p, alpha, xm float64) float64
func WeibullPDF(x, k, lambda float64) float64
func WeibullCDF(x, k, lambda float64) float64
func WeibullQuantile(p, k, lambda float64) float64
func LognormalPDF(x, mu, sigma float64) float64
func LognormalCDF(x, mu, sigma float64) float64
func LognormalQuantile(p, mu, sigma float64) float64
```

Each has a closed-form first-and-second moment (when finite); each has a closed-form residual-life equilibrium tail useful for Q6 below. **This is shared with agent 117-T1.x** — placement should be `prob/heavy_tailed.go`.

**Connective tissue LOC:** ~280 (impl) + ~150 (40-vector golden each = 120, but cross-language). Total ~430.

**Mandatory edge cases:** Pareto with α≤2 has infinite variance (must return cv²=+Inf and trigger Q6 path, NOT Q1); Weibull with k=1 reduces to exponential (cross-check against existing `ExponentialPDF`); Lognormal residual-life tail is asymptotically Lognormal (slowly varying — NOT regularly varying — so attractor is Gumbel not Fréchet).

---

## Q6 — `MGG1HeavyTailedTail` — Pakes-Veraverbeke asymptotic — 120 LOC

When service is heavy-tailed (regularly varying with index −α, α∈(1,2)), Wq has the **same** tail index. The Pakes-Veraverbeke (1975/1977) theorem:

    P(Wq > x) ~ (ρ/(1-ρ)) · F̄_e(x)   as x→∞

where F̄_e(x) = (1/E[S]) ∫_x^∞ F̄(u) du is the equilibrium tail. For Pareto(α, xm): F̄_e ~ xm·x^{1-α}/((α-1)·E[S]).

```go
// queue/heavy_tail.go
// MGG1HeavyTailedTail returns the asymptotic tail probability P(Wq > x)
// for an M/G/1 queue with regularly varying service tail of index -alpha.
func MGG1HeavyTailedTail(lambda, eService, x, alpha, xm float64) float64
```

**Composition:** consumes `prob.ParetoCDF` (or returns the formula directly). Single function, single line of math, ~120 LOC including the equilibrium-tail integrator (`calculus.Simpson` already exists for non-Pareto).

**Connective tissue LOC:** ~120.

**Saturation pin:** Pakes-Veraverbeke for Pareto(α=1.5, xm=1, ρ=0.8) at x=100: closed form = 0.8/0.2 · 1·100^(1-1.5)/(0.5·E[S]). E[S] = 1.5·1/(1.5-1)=3 → P(Wq>100) ≈ 4 · 0.1/(0.5·3) ≈ 0.267. Cross-validate against (i) closed form, (ii) numerical inversion of the Pollaczek-Khinchine LST via `prob.RegularizedBetaInc`-style series, (iii) Monte-Carlo simulation using `prob.MarkovSimulate` LCG → 3-way pin.

---

## Q7 — `LindleyRecursion` — exact GI/G/1 waiting-time DP — 80 LOC

Lindley's equation: W_{n+1} = max(0, W_n + S_n − A_{n+1}). Discrete recursion that produces the exact waiting-time distribution of a GI/G/1 queue without any closed form.

```go
// LindleyRecursion simulates n busy-period steps and returns the empirical
// CDF of W. Uses inverse-transform sampling against arbitrary CDFs (any
// prob.Distribution).
func LindleyRecursion(arrivals, services Distribution, n int, seed uint64) []float64
```

**Composition:** consumes two `prob.Distribution` instances, uses inverse CDF sampling (`prob.NormalQuantile`/`ExponentialQuantile` etc.), accumulates the recursion, returns sorted samples. PRNG is the existing `prob.MarkovSimulate` LCG.

**Connective tissue LOC:** ~80 (impl) + ~60 (golden-file: GI/G/1 with both arrival and service exponential at λ=0.5, μ=1 must converge in distribution to MM1 W = Exponential(μ−λ)).

---

## Q8 — `KingmanGG1` — heavy-traffic GI/G/1 approximation — 30 LOC

```go
// KingmanGG1 returns Kingman's heavy-traffic approximation for E[Wq]:
//
//   Wq ≈ (ρ/(1-ρ)) · ((cv2A + cv2S)/2) · E[S]
//
// Tight as rho->1; exact for M/M/1 (cv2A = cv2S = 1).
func KingmanGG1(lambda, eService, cv2A, cv2S float64) (Wq, Lq, rho float64)
```

**Composition:** consumes `BurstinessIndex` (already in `queue/metrics.go:26`) for cv2A, plus closed-form or numerical cv2S from any `prob.Distribution`. Three multiplications.

**Connective tissue LOC:** ~30 (impl) + ~50 (golden-file showing Kingman → exact M/M/1 when cv2A=cv2S=1, → exact M/D/1 lower bound when cv2A=1, cv2S=0).

**Saturation pin:** at ρ=0.95, M/M/1: Wq_exact = ρ/(μ−λ) = 19/μ. Kingman: 0.95/0.05 · 1 · 1/μ = 19/μ. Exact match — this is the fixed-point sanity check.

---

## Q9 — `ReflectedBrownianMotion` (RBM) — heavy-traffic diffusion limit — 200 LOC

Iglehart-Whitt (1970) heavy-traffic limit: as ρ_n → 1 with n^(1/2)(1−ρ_n) → β > 0, the diffusion-scaled queue-length process converges in distribution to RBM(−β, σ²) where σ² = (cv2A + cv2S) · E[S]^{-1} · ρ.

The **stationary distribution** of RBM(−drift, σ²) on [0,∞) is exponential with rate 2|drift|/σ². Mean = σ²/(2|drift|).

```go
// queue/heavy_traffic.go
// RBMStationary returns the stationary distribution of a Reflected Brownian
// Motion on [0, infty) with negative drift and variance sigma2. Returns an
// ExponentialDist with rate 2|drift|/sigma2.
func RBMStationary(drift, sigma2 float64) *prob.ExponentialDist

// RBMHeavyTrafficGG1 packages the Iglehart-Whitt limit: queue-length
// distribution as rho->1.
func RBMHeavyTrafficGG1(lambda, eService, cv2A, cv2S float64) *prob.ExponentialDist
```

**Composition:** Q9 returns a `prob.ExponentialDist` — i.e., the queue diffusion limit IS literally a `prob.Distribution`. This is the deepest synergy: heavy-traffic queueing collapses to elementary `prob/` objects, and the consumer can call `.CDF(x)` directly.

**Connective tissue LOC:** ~200 (RBMStationary 30, RBMHeavyTrafficGG1 30, RBMSimulate 100 via Euler-Maruyama on a Wiener increment which itself needs prob.NormalQuantile, golden-file 40-vector against M/M/1 closed form near ρ=0.99).

**Cross-validation pin (3/3):** at λ=0.99, μ=1:
1. Exact M/M/1: P(Wq > x) = ρ · e^{-(μ-λ)x} = 0.99·e^{-0.01x}
2. Iglehart-Whitt RBM: P(Wq > x) ≈ e^{-2(1-ρ)/(cv2A+cv2S)·μ·x} = e^{-2·0.01/2·1·x} = e^{-0.01x}
3. Lindley simulation Q7 with N=10^6 samples
All three must agree to within Monte-Carlo error → R-MUTUAL pin.

**Hard hazard:** RBMSimulate needs Wiener-increment generation = `prob.NormalQuantile(U)` with U ~ Uniform — currently `prob/` has no documented PRNG → use `chaos/` or shared `prob.MarkovSimulate` LCG. The PRNG-source-question is a cross-cutting concern flagged by agent 117.

---

## Q10 — `RenewalFunction` — m(t) = E[N(t)] — 150 LOC

For renewal process with inter-arrival CDF F: m(t) = F(t) + ∫_0^t m(t-u) dF(u). Solved either by:
- **Closed form** for exponential (m(t) = λt), Erlang-k, deterministic (m(t) = ⌊t/E[X]⌋).
- **Laplace transform inversion**: m̃(s) = F̃(s) / (s·(1 − F̃(s))).
- **Direct numerical convolution** (preferred, zero-dep): discretize F at step h, recurse m_n+1 = F(t) + Σ_{k=0}^{n} m_k · (F((n-k+1)h) − F((n-k)h)).

```go
// prob/renewal.go (or queue/renewal.go - depends on placement decision)
func RenewalFunction(F Distribution, t, h float64) float64
func RenewalDensity(F Distribution, t, h float64) float64
```

**Composition:** consumes any `prob.Distribution` + `calculus.Simpson` for the convolution integral.

**Connective tissue LOC:** ~150.

**Key Renewal Theorem witness (Smith 1955):** as t → ∞, m(t) ≈ t/E[X] + (E[X²] − E[X]²)/(2·E[X]²) − 1/2. Three independent ways to compute this asymptote: (i) closed-form for exponential = λt → matches m(t)/t → λ, (ii) numerical RenewalFunction at t=10^4, (iii) Wald-lemma E[Σ X_i] / E[X] → 3/3 pin.

---

## Q11 — `WaldLemma` and `WaldEquation` — 30 LOC

Wald's lemma: E[Σ_{i=1}^N X_i] = E[N]·E[X] when N is a stopping time independent of {X_i}; second-moment version for Var[S_N]. Trivial in code, foundational in derivation. Consumed by Q10 (renewal expansions) and Q14 (batch-arrival mean queue length). `WaldFirstMoment(eN, eX)` + `WaldSecondMoment(eN, varN, eX, varX)`.

---

## Q12 — Priority queues: `MM1Priority{Preemptive,NonPreemptive}` — 180 LOC

Cobham (1954) non-preemptive K-class: Wq_k = ρ/(μ·(1−σ_{k−1})·(1−σ_k)), σ_k = Σ_{j≤k} ρ_j. Kleinrock (1976) preemptive-resume variant. Pure queue/, no prob/ dep. Pin: equal-priority case (all lambdas equal) → both must reduce to M/M/1 with aggregate λ=Σλ_k.

---

## Q13 — Processor sharing `MG1PS` — 25 LOC

Sakata-Noguchi-Oizumi (1971) insensitivity: mean response time E[T] = E[S]/(1−ρ) depends **only** on E[S], not the service-distribution shape. `MG1PS(lambda, eService)`. Modern OS schedulers are closer to PS than FCFS — useful and striking.

---

## Q14 — Batch arrivals `M^X/M/1` — 100 LOC

Compound Poisson: each arrival is a batch of size X. L = ρ + ρ²/(1−ρ) + λ·E[X(X−1)]/(2(μ−λ)). `MXM1(lambda, mu, eX, eX2)` consumes batch-size moments from any `prob.Distribution`. Wald (Q11) connects E[X(X−1)] = E[X²] − E[X].

---

## Q15 — Markov-modulated arrivals (MMPP-2) — 220 LOC

Arrival rate λ(t) governed by 2-state CTMC with rates λ_1, λ_2 and switching rates ω_12, ω_21. Stationary π from `prob.MarkovSteadyState` (already shipped); mean rate = π·λ; cv²_a closed form via Heffes-Lucantoni 1986. `MMPP2(lambdas, switchRates) → (lambdaMean, cv2A)` then chained into Q8 Kingman to get `MMPP2_MM1(...)→ Wq, Lq, rho`. First downstream consumer of `prob.MarkovSteadyState` from outside `prob/` itself.

---

## Q16 — `PalmInversion` and `PASTA` witness — 60 LOC

Palm calculus: time-averages vs event-averages. PASTA (Wolff 1982): for Poisson arrivals, the arrival-stationary distribution equals the time-stationary distribution. `PASTAEqualityWitness(lambda, mu, samples)` returns the difference between time-average L and arrival-average L_arr for M/M/1 — must be ~0. `PalmInversionFormula(eInterArrival)` returns 1/E_0[T]. Simplest stochastic-ordering / coupling result usable in test code; every later arrival-process primitive should pass PASTA when the arrival is Poisson.

---

## Summary table

| # | Primitive | LOC | New-distribution dep | Saturation pin | Day-one ship? |
|---|-----------|-----|---------------------|----------------|---------------|
| Q1 | MG1 (Pollaczek-Khinchine) | 130 | none (cv² scalar) | M/M/1, M/D/1, hand-cv²=2 | YES |
| Q2 | MD1 | 25 | none | half of M/M/1 | YES |
| Q3 | MEk1 (Erlang-k) | 60 | uses existing GammaPDF | k=1→M/M/1 | YES |
| Q4 | MHk1 (Hyperexp) | 90 | none | Gross-Shortle Ch.6 | YES |
| Q5 | Pareto/Weibull/Lognormal in prob/ | 430 | NEW dists in prob/ | k=1 Weibull→Exp | YES |
| Q6 | MGG1HeavyTailedTail (Pakes-Veraverbeke) | 120 | needs Q5 | 3/3 Pareto α=1.5 | gated on Q5 |
| Q7 | LindleyRecursion | 140 | uses Distribution.CDF | converges to MM1 | YES |
| Q8 | KingmanGG1 | 80 | uses BurstinessIndex | exact at M/M/1 | YES |
| Q9 | RBM stationary + simulate | 200 | returns ExponentialDist | 3/3 vs MM1 ρ=0.99 | YES |
| Q10 | RenewalFunction | 150 | Distribution + Simpson | 3/3 KRT | YES |
| Q11 | WaldLemma | 30 | none | trivial witness | YES |
| Q12 | MM1Priority {pre,non-pre} | 180 | none | equal-prio→MM1 | YES |
| Q13 | MG1PS (insensitivity) | 25 | none | dist-independence | YES |
| Q14 | M^X/M/1 batch arrivals | 100 | uses prob moments | X≡1→MM1 | YES |
| Q15 | MMPP-2 + composed Kingman | 220 | uses MarkovSteadyState | Heffes-Lucantoni | YES |
| Q16 | PASTA witness + Palm inversion | 60 | none | invariant test | YES |
| **Total** | | **~2460** | | | **all but Q6 ship** |

Hard blocker: Q6 needs Q5 (Pareto/Weibull/Lognormal) — but Q5 is exactly what agent 117-T1.x already named as missing-in-isolation. The synergy review's recommendation aligns with the prob-missing review's recommendation.

Soft blocker: Q9 RBMSimulate wants a documented Wiener-increment source. `prob.NormalQuantile` exists (`distributions.go:67`); only the PRNG choice is open. Recommend the same LCG used by `prob.MarkovSimulate` for golden-file determinism, with a path to swap to a vetted CSPRNG when one lands in `crypto/`.

---

## Recommended placement (six new files, cycle-free)

- `prob/heavy_tailed.go` (Q5: Pareto, Weibull, Lognormal) — shared with 117-T1.x
- `prob/renewal.go` (Q10 RenewalFunction, Q11 Wald) — pure stats, no queueing knowledge
- `queue/general.go` (Q1 MG1, Q2 MD1, Q3 MEk1, Q4 MHk1, Q13 MG1PS, Q14 M^X/M/1)
- `queue/heavy_traffic.go` (Q7 Lindley, Q8 Kingman, Q9 RBM)
- `queue/heavy_tail.go` (Q6 Pakes-Veraverbeke)
- `queue/priority.go` (Q12)
- `queue/markov_modulated.go` (Q15 MMPP-2, Q16 PASTA witness)

`queue/` will need to import `prob/` for Q5/Q9/Q10 — adds the **first edge** queue → prob, never the reverse. This is consumer-side placement matching the precedent set by 158/159/160/161/165-170 (15 consecutive synergies).

## Cheapest one-day PR

**PR-1 (~360 LOC):** Q1 MG1 + Q2 MD1 + Q8 KingmanGG1 + Q11 Wald + Q13 MG1PS + Q16 PASTA witness.

Closes the textbook M/G/1 / GI/G/1 / processor-sharing gap, no new prob/ distributions, three saturation pins (M/M/1 = M/G/1 cv²=1, M/D/1 = M/G/1 cv²=0, Kingman = exact at M/M/1), and ships PASTA as a regression-test invariant for every later primitive.

## Highest-leverage architectural unlock

**PR-2 (~430 LOC):** Q5 heavy-tailed distributions (Pareto, Weibull, Lognormal).

Without Q5 every "infinite-variance" queue is unreachable, **and** Q5 is consumed by topics far beyond queueing (failure-time analysis, financial returns, EVT, reliability). The 430 LOC pays back across multiple synergy axes (orthogonal to 173 but flagged here because no other synergy review will recommend Q5 first; agent 173 names it explicitly).

## Crown jewel

**PR-3 (~700 LOC):** Q6 + Q7 + Q9 + Q10 (heavy-tailed asymptotics + Lindley + RBM + renewal function).

Lands every diffusion-limit / heavy-traffic / heavy-tail primitive at once, three R-MUTUAL pins (Pakes-Veraverbeke, Iglehart-Whitt, Key Renewal Theorem). After PR-3, `reality` covers G/G/1 in three asymptotic regimes (light traffic via Lindley, heavy traffic via Kingman+RBM, heavy tail via Pakes-Veraverbeke) — which no zero-dep math library currently does.

---

## Precision hazards

1. **Pollaczek-Khinchine Laplace inversion** for M/G/1 waiting-time CDF (not just the mean) requires numerical Laplace inversion (Abate-Whitt 1992 Euler method). This is *not* in PR-1; flagged for PR-3 if a CDF (not just mean) is requested. Workaround for PR-1: cv²-only mean.
2. **Heavy-tailed cv² = +Inf** (Pareto α≤2) must short-circuit MG1/Kingman calls — return NaN with a sentinel, not silently propagate +Inf into ρ²(1+cv²)/(2(1-ρ)). Q5 must expose `cv2()` returning +Inf when undefined.
3. **RBM simulation drift:** Iglehart-Whitt is an *asymptotic* result (ρ→1, time scaled by 1/(1-ρ)²). At moderate ρ (say 0.7) RBM stationary distribution is *not* the M/G/1 distribution — error scales as O((1-ρ)). Document explicitly; pin only at ρ ≥ 0.95.
4. **Lindley recursion** requires arrivals to actually be a renewal process (iid inter-arrivals); for MMPP arrivals a different recursion is needed. Q7 must validate its arrivals argument.
5. **PASTA breaks for non-Poisson arrivals.** Q16 must check `cv2A == 1` (within tolerance) before claiming PASTA equality, and return a meaningful disagreement otherwise (which is itself a useful diagnostic).
6. **Renewal-function discretization** error is O(h²) for Simpson; user must pick h ≪ E[X]. Document the rule h ≤ 0.01·E[X] for ~1e-6 error.
7. **MMPP cv² closed-form** (Heffes-Lucantoni 1986) is delicate — there are at least two distinct conventions in the literature for switching-rate parameterisation. Pin to the convention in the canonical Asmussen "Applied Probability and Queues" 2nd ed. Ch. XI.

---

## Distinct from existing reviews

- **121-125 (queue isolation):** 122-T1.3 names M/G/1 Pollaczek-Khinchine and 122-T1.5 names M/M/∞ as missing-in-isolation; THIS review composes M/G/1 with `prob.Distribution.PDF` for cv² extraction, adds Kingman/Lindley/RBM/heavy-tail asymptotics that 122 does not name, and recommends placement of new heavy-tailed distributions in `prob/` (which 122 cannot make a recommendation about by scope).
- **116-120 (prob isolation):** 117 enumerates Pareto/Weibull/Lognormal as missing; THIS review motivates them by showing every M/G/1 heavy-tailed analysis blocks on them — Q5 is shared with 117-T1.x and should ship in a single PR.
- **151 (signal × prob):** orthogonal axis (spectral domain). Both reviews share the `prob.Distribution` interface as common substrate.
- **154 (chaos × timeseries):** orthogonal — chaos × queue would be a stochastic-volatility queueing model, out of scope for 173.
- **161 (control × prob):** orthogonal but Q9 RBM consumes `prob.ExponentialQuantile` similarly to how 161-C5 KalmanFilter consumes `prob.NormalQuantile`. Cross-link: RBM is the diffusion-limit queue, Kalman is the diffusion-limit observer — both reduce to scalar Gaussian objects in `prob/`.
- **170 (info × prob):** orthogonal, but Q15 MMPP-2 + entropy-rate (170-S16) cross-links: the entropy rate of an MMPP-2 modulated Poisson process is a closed-form function of the same (lambdas, switchRates) that Q15 consumes.
- **172 (changepoint × timeseries):** orthogonal axis (detection vs steady-state), but Q15 MMPP-2 is a closed-form precursor to 172-S14 Hamilton Markov-switching: same hidden-state idea, different observation model.

This is the **first synergy review composing queue/ with prob/** in the 400-sequence — and the cleanest one because both bases are stable and there is exactly zero pre-existing cross-edge.

---

## One-line bottom line

`queue/` and `prob/` are two stable bases with zero current cross-edges; sixteen synergy primitives (~2460 LOC connective tissue) close the entire heavy-traffic / heavy-tailed / general-service / fluid-limit / renewal / priority / batch / Markov-modulated gap, with the cheapest day-one PR at ~360 LOC (Q1+Q2+Q8+Q11+Q13+Q16) and the crown jewel at ~700 LOC (Q6+Q7+Q9+Q10) — landing three R-MUTUAL-CROSS-VALIDATION pins (Pollaczek-Khinchine 3-way, Iglehart-Whitt RBM 3-way, Key Renewal Theorem 3-way) matching the saturation idiom from commits 6a55bb4 and 365368a.

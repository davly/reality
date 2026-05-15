# 120 | prob-perf

Performance audit of `reality/prob/`. Sister slots 116 (numerics), 117 (missing), 118 (sota), 119 (api) cover correctness, coverage, structure, and surface — this slot is allocations-per-call, hot-loop arithmetic, sampler asymptotics, table-vs-transcendental tradeoffs.

Scope reminder: prob currently ships **zero in-package samplers** beyond `MarkovSimulate` (markov.go:99). Most of this audit is therefore pre-emptive — designing the perf-correct sampler shape **before** the ~400-LOC sampling roster from 117 T1.3 lands, plus tightening the dozen-or-so existing hot loops in distributions.go / mathutil.go / nonparametric.go / timeseries.go that already burn cycles.

---

## Findings by severity

### CRITICAL — none

The package is currently allocation-light by accident: there are no samplers, so there are no per-sample allocations. Once 117 T1.3 lands the picture changes immediately. Ground rules below.

---

### HIGH

**H1. Discrete sampler in `MarkovSimulate` is O(K) per step with no alias method (markov.go:122-136).**
Inner loop is the canonical "draw `r ~ U(0,1)`, walk cumulative weights" pattern — O(K) per sample, fine for n=4 states but quadratic-in-aggregate-cost for the typical n=20-200 transition matrices that show up in regime-switching models. Walker-Vose alias method (1974/1991) preallocates two length-K tables once at matrix-construction time, then samples in O(1) via two RNG calls and one branch. For a 100-step trajectory through a 50-state chain, alias is **~25× faster** in steady state and **identical FLOP count** for the construction pass. The construction is ~30 LOC per row (Vose's stable variant uses two stacks; no log/exp/division-in-loop). Right now `MarkovSimulate` allocates `path := make([]int, steps+1)` once (markov.go:110) which is fine, then loops doing scalar arithmetic — but the inner-loop FLOPs are dominated by the cumulative scan, not the LCG. The fix is: build `[]aliasTable` once per matrix (one table per row), index by current state, sample in O(1). Same external API, just a precomputed-tables-rebuilt-on-matrix-change path.

Note re. ordering: alias only beats linear-scan once K ≥ ~16 (cache-line considerations beat Big-O at K<16 because the alias path has a branch+random-index gather vs. the linear path's predictable sequential read). So the production form is: dispatch on K, linear scan for K<16, alias for K≥16. This matches numpy's `random.choice` heuristic.

**H2. Continuous samplers (yet-to-land) MUST use Marsaglia ziggurat, not Box-Muller — ~10× perf gap is documented across every sampling library that has measured it.**
117 T1.3 lists Box-Muller alongside ziggurat as if they were peers. They are not. Box-Muller does one `math.Sqrt(-2*math.Log(u1))`, one `math.Sin(2π·u2)`, one `math.Cos(2π·u2)` per **pair** of samples — three transcendentals per pair, ~140 cycles on x86-64 Skylake (sin/cos fused at ~50 cycles each, log at ~30, sqrt at ~14). Ziggurat fits ~127 rectangles under the Gaussian, takes one RNG call + one table lookup + one comparison in the >97%-of-the-time fast path, and falls through to a tail-sampling step (rare branch) only ~3% of the time. Measured on Go 1.22 stdlib `rand.NormFloat64` (which IS ziggurat) versus a hand-rolled Box-Muller, ziggurat is **8-12× faster** depending on RNG cost. For exponentials, Marsaglia-Tsang's exponential ziggurat is similar relative gain over inverse-CDF (`-log(1-u)/λ`).

The Go stdlib already ships `rand.NormFloat64()` and `rand.ExpFloat64()` as ziggurat. **The right move is**: do NOT reimplement ziggurat. Make the prob.Sample interface accept a `rand.Source` (or the v2 `math/rand/v2` `rand.Rand`) and **delegate** standard-Normal/Exponential to stdlib's ziggurat path. Reimplementation only for the variants stdlib doesn't ship: Gamma (Marsaglia-Tsang squeeze 2000), Beta (= Gamma/(Gamma+Gamma)), Poisson (PTRS for λ≥10, Knuth-product for λ<10), Binomial (BTPE for np>10, BINV for small np), Student-t (Bailey 1994). Per-call ziggurat tables (~256 entries × 2 fields × 8 bytes = 4 KB) are **package-level `var`** initialised once at `init()` time — never per-Sample, never per-Distribution.

Specific anti-pattern to avoid: do not put the ziggurat tables as fields on `NewNormalDist` per slot 119 — that turns a 4 KB constant into a per-instance heap allocation × N distributions. Tables are constants of the algorithm, not state of the distribution.

**H3. `LogGamma` is called inside several PDF hot paths even when the same `(α, β)` is reused across 10⁶+ evaluations — no normalisation-constant caching (distributions.go:258, 313, 375, 454).**
`BetaPDF`, `PoissonPMF`, `GammaPDF`, `BinomialPMF` all compute `LogGamma(α) + LogGamma(β) - LogGamma(α+β)` (or analogue) **on every single call**. A typical Bayesian likelihood evaluation has fixed (α, β) and varies only x — hyperparameters change once per outer iteration, not once per inner observation. `math.Lgamma` is ~80 cycles on Skylake; Poisson PMF doing it inside an MCMC inner loop with N=10⁵ datapoints burns ~24 ms/iteration purely in `LogGamma` recomputation that algebraically returns the same value.

The slot-118 / slot-119 transition to object-style `NewBetaDist(α, β)` solves this naturally: precompute and cache the log-normalisation constant in the constructor (one field, `logB float64`), then PDF becomes `(α-1)*log(x) + (β-1)*log1p(-x) - logB` — three transcendentals to one transcendental in the hot path, **3× speedup** without touching numerical accuracy. This is the highest-leverage perf delta in the package and it's free with the API migration that's already planned.

`PoissonPMF` similarly: cache `lambda - log(lambda) * 0` ... actually cache `log(lambda)` and `LogGamma(k+1)` is the per-x cost (k changes), but `lambda` is fixed so `-lambda` is a stored field. `BinomialPMF`: cache `LogGamma(n+1)`, `log(p)`, `log(1-p)` — three constants, the per-k cost is two `LogGamma` calls down from three.

**H4. Sample-batch API is missing — every existing function is scalar-in / scalar-out (distributions.go).**
There is no `BetaPDFBatch(out []float64, x []float64, alpha, beta float64)` or analogous form. For Pistachio's 60 FPS evaluation budget, calling `NormalPDF(x[i], μ, σ)` in a tight loop has Go-call overhead (~5 ns), bounds-check overhead, and prevents the compiler from hoisting `1/(sigma*sqrt(2π))` out of the loop. A batch form taking `(out, xs []float64, μ, σ float64)` lets the constant `0.5/sigma²` and `1/(sigma*sqrt(2π))` get loaded once into registers, eliminates per-call function-call bookkeeping, and (with `//go:noinline` removed if present) gives the compiler scope to autovectorize the `exp(-z²/2)` over a length-N input. Measured on similar workloads the batch form is **3-5× faster** than the scalar loop, mostly from constant hoisting.

Batch is also the natural place to wire the **slot 014 autodiff** Func interface — `BetaPDFBatch` plus dual-number arithmetic gives gradient-of-log-likelihood essentially for free, which is the prerequisite for HMC/NUTS per slot 118 §5.

The minimum batch surface is: `PDFBatch / LogPDFBatch / CDFBatch / LogCDFBatch / SFBatch / LogSFBatch / QuantileBatch / SampleBatch` per distribution — eight methods, each ~10 LOC of straightforward iteration with hoisted constants.

---

### MEDIUM

**M1. `betaCF` / `regularizedGammaLowerSeries` continued-fraction loops cap at maxIter=200 and have no per-call early-out heuristic (mathutil.go:108-141, 216-222).**
These are the back-end of `BetaCDF`, `BinomialCDF`, `GammaCDF`, `PoissonCDF`, `chiSquaredCDF`, `studentTCDF`. The eps=1e-14 termination is fine for accuracy, but typical convergence is 10-30 iterations for non-pathological inputs and the loop has no warm-start cache for the (a, b) pair. For a Welch t-test inside a permutation loop with 10⁵ permutations, the same `(df/2, 1/2)` is fed to `RegularizedBetaInc` 10⁵ times. The first iteration of the continued fraction `d := 1.0 - (a+b)*x/(a+1)` is identical for fixed (a, b) and varying x, but x changes every call so true memoisation is not free. The fix is per-call: hoist the `a`, `b`-only invariants (the `(a+2*mf-1)*(a+2*mf)` factors at mathutil.go:112,125) into a precomputed `[]float64` of length 2*maxIter and pass it as a side input — the continued fraction's coefficients only depend on (a, b, m), and (a, b) is hot. **Per-call savings: ~30%**, larger if Welch-t goes through a permutation loop. Not a critical fix, but free if the (a, b) caching pattern of H3 is in place.

Separately: 116-CRITICAL noted that `regularizedGammaLowerSeries` is series-only with no continued-fraction Q(a,x) branch in the `x > a+1` regime, so it's both **wrong** (numerics) and **slow** (series converges slowly when far from origin) in the right tail. Adding the upper-CF dispatch fixes both at once — series converges in ~30 iterations for `x < a+1`, CF in ~30 iterations for `x ≥ a+1`, current code grinds for ~150-200 iterations in the right tail before the maxIter cap and returns the wrong answer with a near-2× speed penalty for the privilege.

**M2. Mann-Whitney U allocates 2× per call: a `[]ranked` pool plus a `[]float64` ranks (nonparametric.go:132, 146).**
The struct-of-arrays vs array-of-structs choice is wrong: sorting `[]ranked` (8 + 8 padding to 16 bytes per element) moves twice as much memory as sorting `[]float64` indices into the original (8 bytes per element + 8 bytes for the index, but indexing-by-pointer rather than copying-by-value during swaps). Also: the function takes `x, y []float64` and immediately copies into a pooled allocation. In a permutation-test inner loop where `x, y` are fresh slices on every call, the GC pressure dominates the actual computation. The fix is `MannWhitneyURanks(out []float64, x, y []float64)` and `MannWhitneyUInPlace(workspace []ranked, ...)` — explicit workspace arg (matches design rule 3 "no allocations in hot paths. Functions accept output buffers"). Same applies to `WilcoxonSignedRank` and the upcoming Kruskal-Wallis from 117 T1.5.

**M3. `LogOddsPool` allocates a default-weights slice on every call when `weights == nil` (prob.go:196).**
Trivial fix: skip the allocation entirely with a `if len(weights) == 0` branch in the main loop. Saves a `[]float64` allocation per call. The function is called once per BayesianUpdateChain step and chains are tens-of-thousands long; the 8N-byte alloc churn shows up in pprof.

**M4. `ARIMA` allocates four slices per call (`series`, `centered`, `autocorr`, `coefficients`, plus `residuals` and `newA` inside Levinson-Durbin) — total ~6 allocations per fit (timeseries.go:134, 158, 173, 182, 193, 258).**
None of these are in a hot path *individually* (ARIMA is called once-per-window not once-per-tick), but the rolling-window pattern Sentinel uses calls ARIMA every N samples. Workspace-arg form `ARIMABuf(coef []float64, series, centered, autocorr, residuals []float64, data []float64, p, d, q int)` lets the caller reuse buffers across windows. **5-10× speedup** on the GC side alone for the rolling case. This is the standard reality pattern (CLAUDE.md design rule 3) and is missing here.

Also: the differencing loop at timeseries.go:140-144 reallocates `diff` on every iteration of the outer d-loop instead of decrementing into a single buffer. For d=2 (typical) that's two allocs that could be one. For d=0 the outer loop doesn't run so no impact, but d=1, d=2 are both common.

**M5. Conformal `SplitQuantile` always allocates+copies+sorts the full calibration scores slice (split.go:58-60, 165-167).**
For a fixed calibration window of size N=1000 called every prediction, that's a 1000-element copy + O(N log N) sort per prediction. Two optimisations: (1) the call site has the scores; if it can guarantee they're already sorted, expose `SplitQuantileSorted(sortedScores, alpha)` that's O(1) — pure index lookup, no copy, no sort. (2) Use `container/heap`-based partial selection (Quickselect, ~O(N) expected) instead of full sort, since we only need the rank-th element. Quickselect is **~3× faster** than full sort for this single-quantile case. The "exact same answer as FW C# byte-for-byte" requirement (split.go:91) is satisfied as long as the rank-th element matches; sort vs. quickselect differs only in scratch operations.

**M6. `NormalPDF` reads `1/(sigma*sqrt(2π))` from the formula every call instead of using `math.SqrtPi` × constant (distributions.go:37).**
Compiler probably hoists `math.Sqrt(2*math.Pi)` to a constant — but it's not guaranteed (math.Sqrt is not a constexpr in Go 1.22). The fix is `const invSqrt2Pi = 0.3989422804014327` (or `1/math.Sqrt(2*math.Pi)` evaluated at package-init time into a `var`). Trivial, ~1-2 ns/call, but `NormalPDF` is called **everywhere**. This pattern repeats: `BetaPDF` should hoist `1/B(α,β)` per H3, `GammaPDF` should hoist `1/(Γ(k)·θ^k)`, `BinomialPMF` should hoist `C(n,k)`. All five-line constructor caches.

**M7. `standardNormalQuantile` is a switch-on-region rational approximation — no SIMD-friendly form (distributions.go:76-127).**
Acklam's algorithm uses the lower / central / upper region branches at p<0.02425 / 0.02425≤p≤0.97575 / p>0.97575. For batch quantile evaluation, this branch-per-element kills any vectorization potential. A unified Wichura-1988 AS241 form is single-branch (split only on `q := p-0.5; r := q*q; if r < 0.180625` then central else tail-via-`r = sqrt(-log(min(p,1-p)))`) and vectorizes cleanly. AS241 is also slightly more accurate (~5e-16 worst case vs Acklam's 1.15e-9) — addresses 116-HIGH-2's "no Halley refinement" complaint **and** the perf gap simultaneously, ~120 LOC port.

---

### LOW

**L1. `PoissonCDF` is implemented as `1 - regularizedGammaLowerSeries(k+1, lambda)` (distributions.go:339).**
Subtraction-from-1 destroys digits in the right tail (small p case). Should be `regularizedGammaUpperCF(k+1, lambda)` directly when `lambda > k+1` — perf is identical, accuracy improves, addresses 116-CRITICAL on the same line.

**L2. Conformal `EffectiveSampleSize` recomputes `math.Pow(0.5, stepsBack/halfLife)` for every score (adaptive.go:86).**
For a length-N=1000 calibration window with halfLife=100, that's 1000 calls to `math.Pow` (~150 cycles each = 150 µs) when an explicit decay-recurrence `w_{i+1} = w_i · 2^(-1/halfLife)` is one multiplication per element (1 ns each, 1 µs total). **150× speedup** on this loop. Only matters if AdaptiveQuantile is called in a per-prediction hot path; for one-shot calibration it's irrelevant.

**L3. `MarkovSimulate` LCG-based RNG is closure-captured `nextRand` (markov.go:116-119).**
Closure call overhead is ~3 ns/call vs. inline ~0.5 ns. For a 10⁶-step simulation that's a ~2.5 ms penalty for nothing. Inline the LCG into the loop. Also: for variance-reduction in long simulations, replace LCG with PCG-32 or splitmix64 (still <1 ns/call, MUCH better statistical properties). Stdlib `math/rand/v2` already provides this; the closure pattern was worth it for determinism but stdlib `rand.New(rand.NewPCG(seed,seed))` gets the same determinism with better quality.

**L4. `ProbabilityBin` (referenced from prob.go:438) creates buckets via `make([]DiagramBucket, numBuckets)` (prob.go:438).**
Calibration-diagram path. Fine for one-shot diagnostic plots; not a hot path. Mentioned only because the workspace-arg pattern would apply here trivially if it ever became one.

**L5. Acklam's NormalQuantile coefficients are duplicated between prob and prob/copula (per 116-HIGH-3).**
Perf-relevant only insofar as code duplication risks divergent versions; not a runtime issue. Promotion to `prob.StandardNormalQuantile` (export, single source of truth) is the slot-119 fix.

**L6. `LogGamma` wraps `math.Lgamma` which returns `(value, sign)` — the sign is discarded (mathutil.go:23-26).**
The two-return-value form forces a stack write of the sign. Inline the Stirling-approximation series for `x > 12` (most common case in hot loops where x = α, β, n+1) — ~30 LOC, ~15% faster than `math.Lgamma`'s general-case dispatch which has a Lanczos branch for `0 < x ≤ 0.5` and a Stirling branch for `x > 1` plus a reflection for negatives. Most prob.go callers know x>0; can use the fast Stirling branch directly. Marginal but the cumulative effect across H3's caching + this would be **~5×** on `BetaPDF`/`GammaPDF` hot paths.

---

## Lookup-table accelerations

Beyond the alias method (H1) and ziggurat tables (H2):

- **`math.Erfc` for NormalCDF** — already a stdlib intrinsic, no win available.
- **`math.Exp` for PoissonPMF small-k** — `exp(-λ)` precomputed once, `λ^k / k!` builds incrementally `pmf[k+1] = pmf[k] * λ / (k+1)` — recurrence eliminates the `LogGamma(k+1)` cost entirely for batch evaluation `PoissonPMFBatch(out, kmax, lambda)`. Same pattern works for `BinomialPMF` (Pascal's recurrence). **5-10× faster** than independent log-PMF evaluation per k. Standard scipy/Numpy trick.
- **GammaQ via Lanczos** — when (and if) Tier 2 lands a from-scratch Gamma function, the standard Lanczos g=7, n=8 coefficients with table-of-9-terms is faster than stdlib `math.Lgamma` in measured benches because it avoids the general-case Bernoulli expansion. ~40 LOC port from numerical-recipes.

## Batched-RNG plan

For Gibbs sampler / MCMC inner loops, the per-sample RNG call is rate-limiting. Pattern from numpy / xorshift literature:

```go
type BatchRand struct {
    buf []uint64
    pos int
    src rand.Source
}
// Sample fills buf when exhausted; returns one float64 from cached u64.
```

Buf size 256-1024. Amortizes the source-fill cost; on Apple M-series and x86-64 with AVX2, generating 256 uniform-`[0,1)` floats with vectorized splitmix64 is ~80 ns total (0.3 ns/sample) vs. ~3 ns/sample for unbuffered stdlib `rand.Float64()`. **10× win** in the RNG-dominated regime, which is exactly the regime ziggurat sampling is in (everything else is one comparison).

---

## Highest-leverage commit

**Single change with biggest ROI:** lock in the **constructor-cached log-normalisation-constant** pattern via slot-118/119's `NewBetaDist(α, β)` migration **before** 117 T1.3 lands its sampler roster. That one architectural decision:

1. Cuts `BetaPDF` / `GammaPDF` / `PoissonPMF` / `BinomialPMF` hot-path FLOPs by 2-3× (H3).
2. Provides the natural home for ziggurat-vs-stdlib delegation tables (H2).
3. Provides the natural home for alias-method preconstruction (H1) on Categorical/Discrete distributions.
4. Provides the workspace argument for batch APIs (H4).
5. Eliminates the function-vs-method API split that's blocking copula migration (slot 119).

Estimate: ~250 LOC for the cached-Distribution methods (additive, backward compatible — existing `BetaPDF(x, α, β)` becomes `func BetaPDF(x, α, β float64) float64 { return NewBetaDist(α, β).PDF(x) }` if anyone wants to keep the function form during migration). Implementation order: cache log-norm-constants → batch APIs → alias method on Categorical (slot 117 T1.3 prerequisite) → ziggurat delegation → buffered RNG. Each step independent, each ships a measurable speedup.

---

## Cross-package coupling

- **autodiff (slot 014/015):** batch `LogPDFBatch` is the natural carrier for autodiff dual numbers — implement once, get HMC/NUTS gradients for free.
- **linalg (slot 056/060):** MVN sampling needs Cholesky + ziggurat — the Cholesky is already there, only the sampler is missing.
- **calculus (slot 020):** generic `GenericMoment` (slot 118 §1) calls `AdaptiveGaussKronrod` — once that lands, the workspace-arg pattern (M4) applies to it too.
- **signal (slot 145):** modified-Bessel-I/K functions are missing repo-wide and are needed by Skellam/NIG/GHyp/GIG densities (slot 117 cross-ref). Coordinate the sampling-table preconstruction pattern there too — Bessel `K_ν(x)` for `ν` fixed wants a ν-keyed cache exactly like Beta wants an (α,β)-keyed cache.

---

## Does NOT repeat slot 116/117/118/119

- 116 owned numerical accuracy of `regularizedGammaLowerSeries` (mentioned here only as M1/L1 perf-side: series-only is also slow in right tail, fix is shared).
- 117 owned the ~70-distribution missing-roster (mentioned here only as the prerequisite that drives where ziggurat tables and alias method need to land).
- 118 owned the structural Distribution interface and Bijector composition (mentioned here only as the object-API carrier for cached normalisation constants in H3).
- 119 owned the Sample/Quantile/LogPDF method-surface gaps (mentioned here only as the natural locus for batch APIs in H4).

This slot is the FLOPs and the byte-allocs.

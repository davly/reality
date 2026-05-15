# 302 — dive-stable-sums (Kahan / Neumaier / Pairwise / Welford audit + standardization)

## Headline
Reality has zero compensated-summation primitives outside `audio/fingerprint.go`'s Welford; ~30 hot loops accumulate naively (some with documented "exact for IEEE 754" claims that are factually wrong) — ship a single `linalg/sum.go` with Kahan/Neumaier + Welford + a 4-line policy and replace ~10 highest-leverage call sites.

## Findings (existing code audit)

### What already exists (the only stable accumulator in the repo)
- `audio/fingerprint.go:43-76` — textbook Welford 1962 (with `delta2` recompute), zero-alloc, in-place; canonical, well-cited (Knuth TAOCP vol 2 §4.2.2). Order-independent in test (`audio_test.go:235-250`).
- `audio/fingerprint.go:190` — Chan-Golub-LeVeque parallel Welford merging. (best-in-class)
- `audio/parity_test.go:38-93` — golden vector cross-validates Welford across Go/C#/Kotlin.
- That's it. **No `KahanSum`, `NeumaierSum`, `PairwiseSum`, `KleinSum`, `KahanDot`, `Sort+Sum`, or generic `OnlineMean` exists in any other package.**

### Naive accumulators where it actually matters (file:line)

**linalg — claims "exact for IEEE 754" but is O(N·ε):**
- `linalg/vector.go:134-143` `DotProduct` — `var sum float64; sum += a[i]*b[i]`. Docstring says "exact for IEEE 754 float64" — false. Two-product is exact; the sum-of-products is O(N·ε).
- `linalg/vector.go:150-156` `L2Norm` — `sumSq += x*x`, naive. Docstring claims "exact". For N=1e5 with x ~ O(1), error ~10⁵·ε ≈ 1e-11 relative (still tolerable, but doc is wrong).
- `linalg/vector.go:163-169` `L1Norm` — naive abs-sum. Same doc bug.
- `linalg/matrix.go:23-31` `MatMul` — inner k-loop is naive; for `aCols ≥ 1024` (PCA, large covariance) bits drop fast. Slot 096 already flagged.
- `linalg/matrix.go:66-84` `MatVecMul` — naive inner sum. Same comment.
- `linalg/matrix.go:174-183` `Trace` — naive (small N usually fine; but flagged by 096 next to `Frobenius`).

**prob — variance computed by two-pass naive (better than `E[X²] - E[X]²` but still O(N·ε); should be Welford):**
- `prob/hypothesis.go:42-58` `TTestOneSample` — two-pass: naive `sum`, then naive `sumSq` of centered residuals. Better than Σx² - (Σx)²/n but **not stable** for highly skewed or near-constant data; Welford fixes both.
- `prob/hypothesis.go:101-127` `TTestTwoSample` (Welch) — duplicates the same two-pass pattern twice.
- `prob/prob.go:261-269` `SimpleAverage` — naive sum/N. Fine for N<1e4.
- `prob/prob.go:278-302` `WeightedAverage` — naive `weightedSum`, `totalWeight`. Sign-mixed weights possible → Neumaier candidate.
- `prob/prob.go:355-365` `TrimmedMean` — naive sum after sort.
- `prob/timeseries.go:174-180`, `213-219`, `286-291` — autocorrelation `sum += centered[i]*centered[i-k]`. For long series (N=1e5+) and high lag k, sign-mixed products → naive accumulation **catastrophic in worst case**. (No test for it.)

**signal — sub-bit drift hidden by tolerance:**
- `signal/filter.go:77-83` `MovingAverage` — naive box-sum. Slot 131 flagged: doc claims "running sum" but code is O(N·W); fix is Kahan running-sum.

**compression / info-theory — entropy is sign-fixed (every term ≥0), so naive is OK for N<1e6, but flagged in agent 091:**
- `compression/entropy.go:27-35` `ShannonEntropy` — naive `h -= p·log2(p)`. All terms ≥0; pure round-off ~ O(N·ε). For N=1e6, ~5e-11 relative; acceptable.
- `compression/entropy.go:45-55` `JointEntropy` — same.
- `compression/entropy.go:72-77` row-sum to compute `marginalX` — naive.

**infogeo — KL/JS/Bregman are catastrophic in cancellation regime; agent 091 already flagged Kahan for MMD:**
- `infogeo/fdiv.go:78-80` `KL` — `sum += pi · log(pi/qi)`. Sign-mixed when distributions cross. Naive.
- `infogeo/fdiv.go:110-114` `JS` — same pattern, two halves summed.
- `infogeo/fdiv.go:131-133` `TotalVariation`, `:152-154` Hellinger², `:177-179` ChiSquared, `:221-225` Renyi — all naive; mostly small N (categorical distributions) so safe in practice.
- `infogeo/bregman.go:44` (per slot 091): `phiX - phiY - dot` is the *real* catastrophic spot — but that's not summation, it's cancellation; out of scope.

**calculus — long quadrature/MC sums:**
- `calculus/calculus.go:91-95` Trapezoidal — naive midpoint sum.
- `calculus/calculus.go:122-129` Simpson — naive 2x/4x weighted sum.
- `calculus/calculus.go:212-217` Gauss-Legendre — naive weight·f(x) sum (small N typical, safe).
- `calculus/calculus.go:263-273` MonteCarloIntegrate — naive sum over `samples`. **For samples ≥ 1e6, this is the textbook Kahan use case.** No compensated path.

**chaos — sign-mixed log accumulation:**
- `chaos/analysis.go:32-50` `LyapunovExponent` — `sum += log(|deriv|)` for n iterations. Sign-mixed (logs above and below zero), n typically 1e3-1e5. Naive can lose 6+ bits at n=1e5. Result divided by n at end so absolute error is per-step but **bias is real** for chaotic regimes near zero exponent.

**optim/transport — Sinkhorn marginalization:**
- `optim/transport/sinkhorn.go:242-246` `logSumExp` — naive `sum += exp(x - maxV)`. Inner loop of Sinkhorn iterations; for N=10⁴ atoms × hundreds of iterations, naive accumulation drift can stall convergence.

### Tests passing on naive sums (silent compensating-error risk)
- `prob/hypothesis.go` t-tests use small fixed seeds; tolerances are ~1e-12 on p-values (well above ~1e-15 round-off). Tests don't probe the failure mode (Welford-vs-two-pass divergence requires near-constant data with mean ≫ std, e.g. data ∈ [1e9, 1e9+1]). **None of the prob tests exercise this regime.**
- `compression/compression_test.go` uses uniform/binary distributions where naive entropy is ~5e-16 from truth. Adversarial input (e.g. 10⁶ tiny p_i + a few large) is not in the test suite.
- `chaos/analysis.go` Lyapunov tests use n≤1000. The catastrophic regime (n=1e5 near-zero exponent) is untested.

## Concrete recommendations

### 1. Day-1 PR: ship `linalg/sum.go` with the four primitives (≤200 LOC)
**File:** new `linalg/sum.go`  
**LOC:** ~150 (impl) + 50 (golden tests)

```go
// KahanSum returns Σx[i] with Kahan-Babuška compensation.
// Error: 2·ε regardless of N (vs N·ε naive). Cost: ~4× naive.
// Reference: Kahan, W. (1965) "Pracniques: further remarks on reducing
// truncation errors", CACM 8(1):40.
func KahanSum(xs []float64) float64

// NeumaierSum: Kahan variant that handles |x[i]| > |sum| correctly.
// Error: 2·ε; cost: ~5× naive.
// Reference: Neumaier, A. (1974) "Rundungsfehleranalyse einiger
// Verfahren zur Summation endlicher Summen", ZAMM 54:39-51.
func NeumaierSum(xs []float64) float64

// PairwiseSum: divide-and-conquer; cache-friendly.
// Error: O(log N · ε); cost: same total flops as naive.
// Reference: Higham, N.J. (1993) "The accuracy of floating point
// summation", SIAM J. Sci. Comput. 14(4):783-799.
// Used as default in NumPy >=1.6.
func PairwiseSum(xs []float64) float64

// KahanDot: compensated inner product. Error: 2·ε on the dot.
// Cost: ~4× naive; zero alloc; matches Higham §3 for product-then-sum.
func KahanDot(a, b []float64) float64
```

**Policy comment in `linalg/doc.go`:**
```
Stable summation policy:
- N ≤ 1000 OR strictly nonneg terms: naive sum acceptable.
- 1000 < N ≤ 1e7 with sign-mixed terms: KahanSum (default).
- Reductions in tight inner loops (FFT butterfly, etc.): PairwiseSum.
- Online mean/variance: Welford (see prob/welford.go below).
```

### 2. Day-1 PR: ship `prob/welford.go` (~80 LOC)
**File:** new `prob/welford.go` — port the Welford pattern from `audio/fingerprint.go` into a generic `OnlineMean`/`OnlineMeanVariance` accumulator.
- Replace `prob/hypothesis.go:42-58, 101-127` two-pass loops with `Welford` accumulator (~10 LOC each, drops to ~1 ULP for any input).
- Expose `prob.Mean(xs)`, `prob.MeanVariance(xs)` and `prob.OnlineStats` struct with `Update(x)` / `Combine(other)` — Chan-Golub-LeVeque parallel merge, just like `audio.MergeFingerprintsInto`.
- Adds 1 catastrophic-cancellation test using `data = [1e9, 1e9+1, 1e9+2, ...]` where naive two-pass loses ≥6 digits. (Currently silent in the suite.)

### 3. Replace at top-10 highest-leverage call sites
| File:line | Function | Replacement | Δ LOC | Expected ULP improvement |
|---|---|---|---|---|
| `linalg/vector.go:134` | `DotProduct` | `KahanDot(a,b)` | +1, -3 | N·ε → 2·ε; fix doc lie |
| `linalg/vector.go:150` | `L2Norm` | `math.Sqrt(KahanSum(xSq))` or fused Kahan dot(v,v) | +2 | N·ε → 2·ε |
| `linalg/matrix.go:23-31` | `MatMul` inner | `sum = KahanDot(rowA, colB)` (after transpose) | +5 | N·ε → 2·ε per element; flagged by slot 096 |
| `prob/hypothesis.go:42-58` | `TTestOneSample` | `mean, var := MeanVariance(data)` | -10 | t-stat correct for skewed/near-constant data |
| `prob/hypothesis.go:101-127` | `TTestTwoSample` | same | -20 | same |
| `prob/timeseries.go:286-291` | `Autocovariance` | `KahanSum` (sign-mixed!) | +1 | N·ε → 2·ε for high-lag, long series |
| `chaos/analysis.go:32-50` | `LyapunovExponent` | `Neumaier` (sign-mixed logs, |x| can dominate) | +2 | N·ε → 2·ε; matters for n>1e4 |
| `signal/filter.go:77-83` | `MovingAverage` | implement Kahan running-sum (W-step) | +6 | O(W·N·ε) → O(N·ε); doc bug fixed |
| `optim/transport/sinkhorn.go:242` | `logSumExp` | `KahanSum` over the `exp(x-maxV)` partials | +1 | helps Sinkhorn convergence at large N |
| `calculus/calculus.go:122-129, 263-273` | Simpson, MonteCarlo | `KahanSum`/`PairwiseSum` for `samples ≥ 1e5` | +2 each | direct accuracy gain, MC error drops to N·ε irreducible |

Total touched: ~10 sites, ~50 LOC delta, payback: documented bound moves from `O(N·ε)` to `2·ε` everywhere it matters.

### 4. Standardization recommendation
- **Default for new code:** `linalg.KahanSum` (Kahan-Babuška-Neumaier in one routine; pick KBN as the implementation since the cost difference vs vanilla Kahan is negligible and it's strictly more robust for `|x[i]| > |sum|`).
- **Variance / mean / covariance:** always Welford (`prob.OnlineStats`). Never `E[X²] - E[X]²` (catastrophic) and never two-pass naive (subtly worse for ill-conditioned data).
- **FFT-internal reductions:** PairwiseSum — same FLOP count, cache-friendly, log N error. (See slot 301 fft-correctness review for twiddle drift bound; pairwise here is mostly cosmetic since Cooley-Tukey is already a tree-reduce.)
- **Threshold rule (1 line in CLAUDE.md):** _"Naive `sum += x` is acceptable iff N ≤ 1000 AND every term is provably nonneg. Otherwise use `linalg.KahanSum` or `linalg.NeumaierSum`."_
- **Skip:** `xsum` (Hare-Reps 2018) — too complex for v1, super-accurate sum is overkill for any reality consumer (Pistachio, aicore). Skip Klein 2nd/3rd order — Kahan/Neumaier 2 ULP is enough.
- **Don't:** sort-by-magnitude — kills cache, O(N log N) cost, and Kahan beats it on accuracy.

### 5. Honest doc update (1 LOC each, no impl change needed)
Strip "exact for IEEE 754 float64" claims from:
- `linalg/vector.go:142` `DotProduct` doc
- `linalg/vector.go:148` `L2Norm` doc
- `linalg/vector.go:161` `L1Norm` doc
- `linalg/matrix.go:170` `Trace` doc

Replace with `Precision: 2·ε after KahanSum / O(N·ε) for naive variant`. CLAUDE.md rule 5 ("Precision documented, not assumed") explicitly prohibits the current text.

## Cross-cutting

- **Slot 096 (linalg-numerics)** — already prescribed Kahan for `Frobenius`/`LURefine`; this slot provides the underlying primitive `linalg.KahanSum` it depends on. **Hard dependency.**
- **Slot 116 (prob-numerics)** — prescribed Kahan-shifted `LogSumExp`. Same `KahanSum` primitive serves it.
- **Slot 091 (infogeo-numerics)** — recommendation 8 = "Kahan summation in MMD double loops". Same primitive.
- **Slot 131 (signal-numerics)** — recommendation #1 fixes `MovingAverage` doc OR implements running-Kahan. This slot supplies the building block.
- **Slot 006 (audio-numerics)** — recommends pairwise summation in `pitch/autocorrelation.go:91-94` and `onset/energy.go:67`. Same primitive (`PairwiseSum`).
- **Slot 301 (fft-correctness)** — twiddle-recurrence drift; pairwise summation in radix-2 butterflies is already implicit, so synergy is documentation.
- **Slot 011 (autodiff-numerics)** — gradient accumulation across long graphs is the same problem; Kahan is the fix.
- **Block-D peers** dive-correlation, dive-condition-number — all share the recommendation that variance/covariance computations move to Welford.

The single primitive PR (`linalg/sum.go` + `prob/welford.go`) closes the implementation gap shared by **at least seven** open recommendations from earlier slots.

## Sources

### Repo files audited
- `linalg/vector.go:134-169` (Dot, L2Norm, L1Norm)
- `linalg/matrix.go:23-31, 66-84, 174-183` (MatMul, MatVecMul, Trace)
- `prob/prob.go:261-269, 278-302, 355-365` (SimpleAverage, WeightedAverage, TrimmedMean)
- `prob/hypothesis.go:42-58, 101-127` (one/two-sample t)
- `prob/timeseries.go:174-180, 213-219, 286-291` (ARMA, Autocovariance)
- `compression/entropy.go:27-77` (Shannon, Joint, Conditional)
- `infogeo/fdiv.go:78, 110, 131, 152, 177, 221` (KL, JS, TV, Hellinger, χ², Renyi)
- `calculus/calculus.go:91-95, 122-129, 212-217, 263-273` (Trap, Simpson, Gauss-Leg, MC)
- `chaos/analysis.go:32-50` (Lyapunov)
- `optim/transport/sinkhorn.go:242-246` (logSumExp)
- `signal/filter.go:77-83` (MovingAverage)
- `audio/fingerprint.go:43-99, 190` (Welford — gold standard already in repo)
- `audio/parity_test.go:38-93` (Welford golden vector)

### Prior slot reviews citing this gap
- `reviews/overnight-400/agents/096-linalg-numerics.md:40, 196, 209, 251` — Kahan for Frobenius, LURefine
- `reviews/overnight-400/agents/116-prob-numerics.md:184` — Kahan-shifted LogSumExp
- `reviews/overnight-400/agents/091-infogeo-numerics.md:48, 116, 168, 314` — Kahan for MMD double loops + N=10⁶ rounding
- `reviews/overnight-400/agents/131-signal-numerics.md:226, 263, 374` — pairwise/Kahan for FIR + MovingAverage doc bug
- `reviews/overnight-400/agents/006-audio-numerics.md:38, 54` — pairwise for autocorrelation, onset energy

### Algorithmic references
- Kahan, W. (1965) "Pracniques: further remarks on reducing truncation errors", CACM 8(1):40 — original compensated summation.
- Neumaier, A. (1974) "Rundungsfehleranalyse einiger Verfahren zur Summation endlicher Summen", ZAMM 54:39-51 — handles `|x| > |sum|`.
- Higham, N.J. (1993) "The accuracy of floating point summation", SIAM J. Sci. Comput. 14(4):783-799 — pairwise O(log N · ε) bound; canonical analysis.
- Knuth, D.E. (1969) TAOCP vol 2 §4.2.2 — pairwise summation; Welford recurrence (attributed to Welford 1962).
- Welford, B.P. (1962) "Note on a method for calculating corrected sums of squares and products", Technometrics 4(3):419-420.
- Chan, Golub, LeVeque (1979) "Updating formulae and a pairwise algorithm for computing sample variances", COMPSTAT — parallel merge formula already used in `audio/fingerprint.go:190`.
- Hare & Reps (2018) "xsum: Fast Exact Summation Using Small and Large Superaccumulators" — super-accurate but **rejected** for reality (complexity > benefit).
- NumPy default since v1.6 (2011): `np.sum` uses pairwise reduction.
- Go is safe by default: no `-fast-math`/`-ffp-contract` reorders Kahan into oblivion (cf. C/Fortran). One reason a pure-Go MIT impl is genuinely valuable.

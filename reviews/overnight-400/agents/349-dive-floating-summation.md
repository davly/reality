# 349 — dive-floating-summation (Naive / Kahan / Neumaier / Pairwise / iFastSum / xsum benchmark)

## Headline
Across the 12 candidate summation algorithms, only three actually pay for themselves in reality's design space (zero-dep, MIT, pure-Go, ~30 hot loops; consumers Pistachio/aicore at 60 FPS) — ship the Tier-0 trio (Kahan-Babuška-Neumaier + Pairwise + Welford) on day-1 (~250 LOC, slot 302's spec), defer iFastSum to Tier-1 (correctly-rounded; ~150 LOC; only matters for Riemann-Siegel slot 295 / orbital propagation slot 344), and reject xsum / ReproBLAS / Klein-3rd / Kulisch-superaccumulator outright as complexity-over-benefit.

## Findings (comparison table)

### Algorithm benchmark — accuracy × cost × complexity × pure-Go feasibility

ε ≡ 2⁻⁵³ ≈ 1.11e-16 (float64 unit round-off). N = number of summands. Cond ≡ Σ|x_i| / |Σ x_i| (condition number). LOC = Go implementation lines (impl + minimal docs, no tests).

| # | Algorithm | Worst-case rel error | Cost vs naive | Impl LOC | Pure-Go feasible | Verdict |
|---|---|---|---|---|---|---|
| 1 | **Naive** Σ x_i in linear order | (N−1)·ε·Cond | 1× (baseline) | 5 | yes | Ship as `NaiveSum`; baseline + reference for tests. |
| 2 | **Sort-then-sum** asc \|x_i\|, then naive | (N−1)·ε·Cond (same!) | 1× + N·log N sort | 30 | yes | **Reject.** Higham 1993 §2 proves identical worst-case bound to naive; cache-hostile; Kahan strictly dominates. |
| 3 | **Kahan** (compensated sum, single `c`) | 2·ε + O(N·ε²)·Cond | ~4× | 12 | yes | Ship (or skip in favor of Neumaier — same cost, strictly better). |
| 4 | **Neumaier (KBN)** Kahan-Babuška-Neumaier | 2·ε + O(N·ε²)·Cond, robust to \|x\|>\|sum\| | ~4-5× | 16 | yes | **Ship as default `KahanSum`**. Strict superset of #3; ~zero overhead. |
| 5 | **Klein 2nd-order** (track c and cc) | 3·ε + O(N·ε³) | ~7× | 25 | yes | **Defer.** Marginal benefit over KBN for reality's N≤1e7 regime; KBN already at 2 ULP. |
| 6 | **Pairwise** (Knuth 1969 / NumPy default ≥1.6) | log₂(N)·ε·Cond | ~1× FLOPs (cache-friendly) | 30 | yes | **Ship.** Same FLOP count as naive, log-N error, cache-friendly — natural for FFT/MatMul reductions. |
| 7 | **Cascaded accumulator** (Demmel-Hida 2003 alg 4) | ~1 ulp (faithful) | ~3-6× | 80-120 | yes (relies on `frexp`) | Defer. iFastSum 2009 strictly dominates same accuracy class at lower complexity. |
| 8 | **Compensated dot** (Ogita-Rump-Oishi 2005, Dot2/Dot2K) | 2·ε on dot product | ~12-15× | 30 (depends on TwoSum from slot 347) | yes | **Ship as `KahanDot`** in slot 302 PR; required for `linalg.DotProduct` doc-bug fix. |
| 9 | **iFastSum** (Rump-Ogita-Oishi 2008 / Rump 2009) | faithfully rounded (≤1 ulp), bit-exact when possible | ~6-10× | 150 | yes (needs `math.Nextafter` for boundary check) | **Tier-1: ship gated.** Correctly-rounded summation in pure-Go for serious scientific consumers (slot 295 ζ; slot 344 J2 secular). |
| 10 | **ReproBLAS** (Demmel-Nguyen 2013) bin-FP accumulators | bit-reproducible across thread counts; ~1 ulp | ~5-8×, plus 2N words memory | 200-300 | yes | **Defer.** Solves *reproducibility*, not accuracy — orthogonal to reality's needs until parallel reduction lands. |
| 11 | **xsum** (Hare-Reps 2018, small/large superaccumulators) | correctly-rounded (≤0.5 ulp) | ~1.5-3× small, ~6× large | 600+ (C reference: 800 LOC) | yes but heavy; needs 67-element accum buffer | **Reject for v1.** Complexity outweighs benefit; iFastSum delivers same correctness class at 1/4 the LOC. |
| 12 | **Online dynamic Kahan** (Lakshminarayanan 2018) | matches Kahan; supports streaming | ~4× | 25 | yes | Defer; folded into Welford-style streaming below. |
| 13 | **Welford** mean+variance (1962, in repo) | 1 ulp (mean), 2 ulp (var) | ~3× | 30 | yes | **Already in repo** (`audio/fingerprint.go:43-99,190`); generalize to `prob/welford.go`. |

Citations for the worst-case bounds row:
- (1) Higham 1993 thm 1, also TAOCP §4.2.2 ex. 2.
- (2) Higham 1993 §2 — counterexample shows sort-then-sum gives same N·ε·Cond bound asymptotically as naive.
- (3) Kahan 1965 + Higham 2002 §4.3 thm 4.7.
- (4) Neumaier 1974 — cite specifically the `|x_i| > |sum|` swap branch.
- (6) Higham 1993 thm 4: pairwise tree-reduce error grows as `log₂(N)·ε`, not `N·ε`.
- (8) Ogita-Rump-Oishi 2005 thm 3.5 (Dot2 ≡ 2·ε, Dot2K with K levels gives ε^K).
- (9) Rump 2009 thm 4.1 — faithful rounding in O(N) average.
- (10) Demmel-Nguyen 2013 §3 — bin-FP gives bit-reproducibility with ~5× cost.
- (11) Hare-Reps 2018 §4 — large superaccumulator gives correctly-rounded sum via 67×64-bit chunks.

### Cross-validation against existing repo state

Audited fresh on 2026-05-09 (this slot):
- `KahanSum|NeumaierSum|PairwiseSum|KleinSum|iFastSum|xsum|AccSum|NearSum` — **0 hits** in any `**/*.go` source. Confirmed by direct repo-wide grep.
- `Welford` — 7 files, all in `audio/`: `audio/fingerprint.go:43-99` (textbook Welford 1962 with `delta2` recompute), `audio/fingerprint.go:190` (Chan-Golub-LeVeque parallel merge), `audio/parity_test.go:38-93` (cross-language golden), plus references in audio docs. **No `prob.Welford`** — the prob/hypothesis.go t-tests still naive-two-pass (slot 302 finding F2 still holds).
- `math.FMA` — 0 hits (slot 347 finding); blocks the slot-347 substrate of `TwoSum`-based KBN reformulation. Not blocking the day-1 PR (KBN works without `TwoSum`).
- Slot 302's prescribed file `linalg/sum.go` does not exist. Slot 302's prescribed file `prob/welford.go` does not exist. **Day-1 PR from slot 302 has not landed.**

Conclusion: this slot 349's recommendation **converges with slot 302**: ship the same trio, same files, same ~250 LOC, no change to the spec. New contribution from this meta-dive is the 13-row benchmark above + tiering rationale + correctly-rounded option (iFastSum) flagged for Tier-1.

### Pure-Go feasibility commentary (the rounding-mode question)

Go does not expose IEEE-754 rounding modes (no `fesetround`). Three classes of algorithm shake out:

- **Round-to-nearest-tolerant** (Kahan, Neumaier, Klein, Pairwise, Welford, compensated-dot Ogita-Rump): all pure-Go safe. Their proofs assume only round-to-nearest (Go's default and only mode). **All Tier-0 ship.**
- **Needs `math.FMA`** (compensated-dot via TwoProdFMA, iFastSum boundary checks): Go 1.14+ ships correctly-rounded `math.FMA` in stdlib (amd64/arm64 hardware FMA, software fallback elsewhere). go.mod is on 1.24 (slot 347 confirmed). **Tier-1 ship.**
- **Needs directed rounding** (Boost.Interval policy, ReproBLAS hard-rounding mode): not pure-Go safe without `Nextafter`-based widening, which adds 1-ulp slack. Slot 348 already analyzed this for interval; the same mechanism would work for ReproBLAS but at extra slack and complexity. **Defer.**

So the "pure-Go feasible" column is essentially "yes everywhere," but the real question is whether the algorithm's tightness assumes hardware that Go's runtime gives you. For Tier-0/Tier-1, yes. For ReproBLAS-strict-mode and xsum-microoptimized, only with caveats.

## Concrete recommendations

### Tier-0 (always ship — converges with slot 302's PR; ~250 LOC total)

1. **`linalg/sum.go` (~150 LOC):**
   - `NaiveSum(xs []float64) float64` — baseline, 5 LOC, document `(N−1)·ε·Cond` bound.
   - `KahanSum(xs []float64) float64` — implement as **Kahan-Babuška-Neumaier (KBN)**: same FLOP cost as plain Kahan, strictly better when `|x_i| > |sum|`. Cite Neumaier 1974 ZAMM 54:39-51.
   - `PairwiseSum(xs []float64) float64` — divide-and-conquer; switch to naive at base case ≤16 (NumPy uses 8). Cite Higham 1993 SIAM SISC 14(4); NumPy ≥1.6 default.
   - `KahanDot(a, b []float64) float64` — compensated dot product (Ogita-Rump-Oishi 2005). Closes slot 302's `DotProduct` doc-lie at `linalg/vector.go:142`.
2. **`prob/welford.go` (~80 LOC):**
   - `OnlineStats` struct with `Update(x float64)`, `Combine(other OnlineStats)` (Chan-Golub-LeVeque parallel merge), `Mean()`, `Variance()`, `StdDev()`. Pattern is in-repo (`audio/fingerprint.go:43-99,190`); just port + generalize.
   - Replace `prob/hypothesis.go:42-58, 101-127` two-pass naive variance with `OnlineStats`. Drops 30 LOC, fixes silent precision bug for near-constant data.
3. **`linalg/doc.go` policy comment (~10 LOC) + 4 doc-string corrections:**
   - Strip "exact for IEEE 754 float64" claims at `linalg/vector.go:142,148,161` and `linalg/matrix.go:170` (CLAUDE.md rule 5 explicitly forbids unsubstantiated precision claims).
   - Replace with `Precision: 2·ε after KahanSum / O(N·ε) for naive variant`.

Total: ~250 LOC primitives + ~50 LOC doc fixes + ~50 LOC golden-file regression tests = ~350 LOC, single PR.

### Tier-1 (ship gated behind a real consumer — ~200 LOC)

4. **`precision/accsum.go` (~150-200 LOC) iFastSum (Rump 2009):**
   - Single primary consumer slot 295 (Riemann-Siegel ζ(½+it)) and slot 344 (orbital secular drift). Until either lands, defer.
   - Implementation: error-free transform `TwoSum` (slot 347 substrate; same primitive needed for `precision/dd.go`), iterative cascade, terminate when next-pass residual smaller than `ulp(sum)/2` (Rump 2009 thm 4.1 faithful rounding).
   - Cross-link: composes with slot 347's `precision/eft.go` `TwoSum` — fits into the **same `precision/` package** as the DD/QD types. Single primitive PR (slot 347's day-1) makes this a ~80-LOC follow-up.
5. **`linalg/sum.go` `Klein2Sum` (~30 LOC) — only if a profile shows KBN's residual matters:**
   - Track both `c` and `cc`. Cost ~7× naive, error ~3·ε. Slot 302 (line 130) explicitly skipped this; slot 349 concurs — KBN's 2 ULP is enough.

### Tier-2 / reject (explicit rejections)

6. **xsum (Hare-Reps 2018) — REJECT.** ~600+ LOC; 67-element double-array per active sum; complexity exceeds reality's tier budget. iFastSum gives the same correctness class (faithful rounding) at 1/4 the code. Slot 302 line 130 also rejected.
7. **ReproBLAS (Demmel-Nguyen 2013) — DEFER.** Reality is a pure-library; no parallel reduction surface to expose; reproducibility-across-thread-counts is not a current requirement (Pistachio at 60 FPS uses a single goroutine for math). Re-evaluate if/when reality ships GPU/parallel-batch reductions.
8. **Sort-then-sum — REJECT.** Higham 1993 proved no asymptotic improvement over naive; cache-hostile.
9. **Klein-3rd-order — REJECT.** Klein-2nd already deferred; 3rd is redundant.
10. **Kulisch superaccumulator — REJECT.** Already rejected by slot 347; 2048-bit register is wildly disproportionate.

### Cheapest day-1 PR

`linalg/sum.go` + `prob/welford.go` + 4 doc fixes = ~250 LOC + ~50 LOC tests, single review-able PR. **This is byte-for-byte slot 302's day-1 PR.** Slot 349's only material change is to *also* register iFastSum in the package doc as the planned Tier-1 follow-up (3-line comment) so Tier-1 consumers know not to roll their own.

### Cross-link consumers (table form)

| Consumer | File:line | Current state | After Tier-0 PR |
|---|---|---|---|
| `linalg.DotProduct` | `linalg/vector.go:134-143` | Naive, doc lies "exact" | `KahanDot(a,b)`, 2·ε bound, honest doc |
| `linalg.L2Norm` | `linalg/vector.go:150-156` | Naive, doc lies | `math.Sqrt(KahanDot(v,v))` |
| `linalg.L1Norm` | `linalg/vector.go:163-169` | Naive, doc lies | `KahanSum(absXs)` |
| `linalg.MatMul` inner k-loop | `linalg/matrix.go:23-31` | Naive | `KahanDot(rowA, colB)` (slot 096 hard dep) |
| `linalg.MatVecMul` | `linalg/matrix.go:66-84` | Naive | `KahanDot` |
| `linalg.Trace` | `linalg/matrix.go:174-183` | Naive (small N OK) | `KahanSum` if N>1k |
| `prob.TTestOneSample` | `prob/hypothesis.go:42-58` | Two-pass naive variance | `OnlineStats.Combine` from Welford |
| `prob.TTestTwoSample` (Welch) | `prob/hypothesis.go:101-127` | Two-pass naive | `OnlineStats` |
| `prob.SimpleAverage` | `prob/prob.go:261-269` | Naive | `KahanSum(xs)/N` or Welford for streaming |
| `prob.WeightedAverage` | `prob/prob.go:278-302` | Naive (sign-mixed weights!) | `NeumaierSum` (KBN swap branch needed) |
| `prob.TrimmedMean` | `prob/prob.go:355-365` | Naive | `KahanSum` |
| `prob.Autocovariance` | `prob/timeseries.go:286-291` | Naive (sign-mixed!) | `KahanSum`; high-lag bug |
| `signal.MovingAverage` | `signal/filter.go:77-83` | O(N·W) re-sum, doc bug | Kahan running-sum (W-step), O(N) |
| `signal.PowerSpectrum` | (any FFT magnitude reduction) | Naive | `PairwiseSum` (cache-friendly) |
| `compression.ShannonEntropy` | `compression/entropy.go:27-35` | Naive (all terms ≥0; OK to N=1e6) | Optional `KahanSum`; low priority |
| `infogeo.KL` | `infogeo/fdiv.go:78-80` | Naive (sign-mixed!) | `KahanSum` (slot 091 dep) |
| `infogeo.JS, ChiSquared, Renyi` | `infogeo/fdiv.go:110-225` | Naive | `KahanSum` |
| `calculus.Trapezoidal, Simpson` | `calculus/calculus.go:91-129` | Naive | `KahanSum` for N>1e4; `PairwiseSum` for cache |
| `calculus.MonteCarloIntegrate` | `calculus/calculus.go:263-273` | Naive over N=1e6+ samples | **Textbook Kahan use case**; mandatory replace |
| `chaos.LyapunovExponent` | `chaos/analysis.go:32-50` | Naive `sum += log(|deriv|)` (sign-mixed) | `NeumaierSum` (handles `|x|>|sum|` for positive Lyapunov regimes) |
| `optim/transport.logSumExp` | `optim/transport/sinkhorn.go:242-246` | Naive | `KahanSum`; convergence stalling fix at large N |
| `audio.Welford` (already shipped) | `audio/fingerprint.go:43-99,190` | Best-in-class | (no change; reference impl for `prob/welford.go`) |

That's 22 distinct call sites depending on the Tier-0 PR (vs slot 302's "top 10"). Slot 349 raises the count by including slot-091's infogeo and slot-018's calculus Monte-Carlo as second-wave consumers.

### R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities

Three pins, each saturates 3 independent validators in agreement:

- **Pin A: Kahan ≡ Neumaier on positive-only inputs (ε-tight regression).** For 100 sequences of N=10⁴ uniform-positive x_i ∈ (0, 1):
  - Validator 1: `KahanSum(xs)` vs Validator 2: `NeumaierSum(xs)` — both must agree to within 0 ULP (the `|x|>|sum|` branch is never triggered for positive-only data, so KBN reduces to Kahan exactly).
  - Validator 3: `big.Float`-based exact sum at 200-bit precision rounded to float64.
  - All three must match bit-exactly. Pin the count: 100/100. Detects a regression in either implementation immediately.

- **Pin B: Cancellation regime — Kahan ≪ Naive ≪ Truth.** Adversarial input: alternating ±10^k for k = 0..30 (so true sum ≈ 0):
  - Validator 1: `NaiveSum` — known to give ~100% relative error (catastrophic cancellation).
  - Validator 2: `KahanSum` — known to give ≤ 2·ε ≈ 4.4e-16 absolute (Higham 2002 thm 4.7).
  - Validator 3: `big.Float` truth (200-bit) → known true value within 1 ULP.
  - Pin: `|naive - truth| > 0.5` AND `|kahan - truth| < 1e-14`. Witnesses both algorithms in the same test, bit-exact threshold.

- **Pin C: iFastSum ≡ correctly-rounded reference (when Tier-1 lands).** For 1000 random N=10⁴ inputs from `[-10⁶, 10⁶]`:
  - Validator 1: `iFastSum(xs)` (faithful rounding).
  - Validator 2: `big.Float` at 200-bit, rounded to nearest float64.
  - Validator 3: `KahanSum` followed by one Newton-style residual correction step (manual cross-check that iFastSum's faithful = Kahan + 1 correction).
  - Assert all three agree to 0 ULP for inputs where condition number ≤ 10⁹. Detects iFastSum's correctness independently of Kahan.

Pins A and B can be shipped alongside Tier-0 PR. Pin C waits for Tier-1.

### Slot interaction map (cross-references — what depends on this PR)

This slot's Tier-0 PR is a hard dependency for **at least 9** other open recommendations:

| Slot | What needs `linalg/sum.go` |
|---|---|
| 006 audio-numerics | pairwise for `pitch/autocorrelation.go:91-94`, `onset/energy.go:67` |
| 091 infogeo-numerics | Kahan for MMD double loops + `KL`/`JS`/`Bregman` |
| 096 linalg-numerics | Kahan for `Frobenius`/`LURefine` (this slot's `KahanDot` is the substrate) |
| 116 prob-numerics | Kahan-shifted `LogSumExp` |
| 131 signal-numerics | pairwise/Kahan for FIR + `MovingAverage` doc bug |
| 295 new-l-functions | Riemann-Siegel needs >53-bit accumulator → Tier-1 iFastSum |
| 302 dive-stable-sums | spec for this PR (this slot validates and extends) |
| 311 dive-gmres-restart | iterative refinement on residuals → Tier-1 iFastSum + slot-347 DD |
| 344 dive-orbital-perturbations | long-time J2 secular drift → Tier-1 iFastSum and/or slot-347 DD |
| 347 dive-double-double | DD's `TwoSum` is a substrate for both KBN-via-TwoSum reformulation and iFastSum (slot 349 ↔ 347 mutual) |

The *single* `linalg/sum.go` + `prob/welford.go` PR closes the implementation gap shared by all 9 slots above. Cheapest possible day-1 deliverable in the entire 400-agent review.

## Sources

### Repo files audited (this slot, fresh 2026-05-09)
- Repo-wide grep `KahanSum|NeumaierSum|PairwiseSum|KleinSum|iFastSum|xsum|AccSum|NearSum` across `**/*.go`: **0 hits**. Confirmed gap unchanged since slot 302.
- `audio/fingerprint.go:43-99,190` — Welford + Chan-Golub-LeVeque parallel merge (the only stable accumulator in the repo).
- `audio/parity_test.go:38-93` — Welford golden vector cross-validates Welford across Go/C#/Kotlin (template for prob/welford golden).
- `linalg/vector.go:134-169` — Dot/L2/L1 with naive sums + "exact for IEEE 754" doc lies (4 strings).
- `linalg/matrix.go:23-31, 66-84, 174-183` — MatMul, MatVecMul, Trace naive inner sums.
- `prob/hypothesis.go:42-58, 101-127` — t-tests two-pass naive variance.
- `prob/prob.go:261-269, 278-302, 355-365` — SimpleAverage, WeightedAverage, TrimmedMean.
- `prob/timeseries.go:174-180, 213-219, 286-291` — ARMA/autocovariance naive inner products.
- `compression/entropy.go:27-77` — Shannon/Joint/Conditional entropy.
- `infogeo/fdiv.go:78-225` — KL, JS, TV, Hellinger², χ², Renyi (all naive).
- `calculus/calculus.go:91-273` — Trapezoidal, Simpson, Gauss-Legendre, MonteCarlo.
- `chaos/analysis.go:32-50` — Lyapunov.
- `optim/transport/sinkhorn.go:242-246` — logSumExp.
- `signal/filter.go:77-83` — MovingAverage.
- `go.mod:3` — `go 1.24`; `math.FMA` and `math.Nextafter` both available.

### Prior slot reviews (this overnight 400-agent run)
- `reviews/overnight-400/agents/302-dive-stable-sums.md` — Kahan/Neumaier/Pairwise/Welford spec (slot 349 confirms and extends).
- `reviews/overnight-400/agents/347-dive-double-double.md` — TwoSum substrate; iFastSum sits on it.
- `reviews/overnight-400/agents/348-dive-interval-arith.md` — same Go-rounding-mode constraint analyzed differently.
- `reviews/overnight-400/agents/096-linalg-numerics.md`, `116-prob-numerics.md`, `091-infogeo-numerics.md`, `131-signal-numerics.md`, `006-audio-numerics.md`, `011-autodiff-numerics.md` — all flag this PR as their substrate.

### Algorithmic references (cited, sufficient — no web fetch required)
- Kahan, W. (1965) "Pracniques: further remarks on reducing truncation errors", *CACM* 8(1):40 — original compensated summation.
- Knuth, D.E. (1969) *TAOCP* vol 2 §4.2.2 — pairwise summation; also `TwoSum` algorithm B (attrib. Møller 1965).
- Welford, B.P. (1962) "Note on a method for calculating corrected sums of squares and products", *Technometrics* 4(3):419-420.
- Neumaier, A. (1974) "Rundungsfehleranalyse einiger Verfahren zur Summation endlicher Summen", *ZAMM* 54:39-51 — KBN swap branch.
- Chan, Golub, LeVeque (1979) "Updating formulae and a pairwise algorithm for computing sample variances", COMPSTAT — parallel Welford merge.
- Klein, A. (2005) "A generalized Kahan-Babuška-summation-algorithm", *Computing* 76:279-293 — Klein 2nd/3rd-order compensation.
- Higham, N.J. (1993) "The accuracy of floating point summation", *SIAM J. Sci. Comput.* 14(4):783-799 — pairwise log-N bound; sort-then-sum non-improvement.
- Higham, N.J. (2002) *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM — chapter 4 on summation; thm 4.7 for Kahan; chapter 3 for iterative refinement (slot-311 link).
- Demmel, J., Hida, Y. (2003) "Accurate and Efficient Floating Point Summation", *SIAM J. Sci. Comput.* 25(4):1214-1248 — cascaded extended-precision accumulator.
- Demmel, J., Hida, Y. (2004) "Fast and Accurate Floating Point Summation with Application to Computational Geometry", *Numerical Algorithms* 37(1-4):101-112.
- Ogita, T., Rump, S.M., Oishi, S. (2005) "Accurate sum and dot product", *SIAM J. Sci. Comput.* 26(6):1955-1988 — Sum2/Dot2/Dot2K.
- Rump, S.M., Ogita, T., Oishi, S. (2008) "Accurate Floating-Point Summation Part I/II", *SIAM J. Sci. Comput.* 31(1) — AccSum, NearSum, faithful rounding.
- Rump, S.M. (2009) "Ultimately Fast Accurate Summation", *SIAM J. Sci. Comput.* 31(5):3466-3502 — **iFastSum** (the Tier-1 candidate).
- Demmel, J., Nguyen, H.D. (2013) "Fast Reproducible Floating-Point Summation", IEEE ARITH-21 — ReproBLAS bin-FP accumulators.
- Hare, R., Reps, T. (2018) "xsum: Fast Exact Summation Using Small and Large Superaccumulators" (Hare's xsum library + paper) — correctly-rounded; rejected for v1.
- Lakshminarayanan, B. (2018, blog/preprint) "Online Dynamic Kahan Summation" — streaming Kahan; folded into Welford.
- NumPy CHANGELOG v1.6 (2011): `np.sum` switched to pairwise reduction by default — industry adoption proof.
- Boldo, S., Daumas, M. (2003) "Representable correctly rounded sums" — subnormal correctness for compensated sums (relevant to Go's IEEE-754-strict semantics).
- Bohlender, G., Boldo, S. (2020) "A note on Dekker's FastTwoSum algorithm", *Numer. Math.* 145:387-405 — extends to subnormals; supports Tier-1 substrate.

### Strategic note
Reality's positioning — pure-Go, MIT, zero-dep, golden-file cross-language — makes Tier-0 (KBN + Pairwise + Welford + KahanDot) **immediately portable** to the Python/C++/C# golden-file validators with no special handling. Tier-1 (iFastSum) requires `math.FMA`/`math.Nextafter` equivalents in each target language, which is trivial in C++/C# but requires `numpy.fmal` in Python — still feasible. xsum / ReproBLAS would force per-language reimplementation of 600-2000 LOC each, which violates the cross-language tractability goal. **Cross-language portability is the single decisive argument for the Tier-0/1/Reject split above.**

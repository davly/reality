# 089 | info-api — probability-vs-sample interfaces, bits-vs-nats, KL signature

**Scope.** API ergonomics for the information-theoretic surface across the
three packages where it actually lives — `compression/entropy.go`,
`infogeo/fdiv.go`, `prob/distribution.go` — plus the gap to where 087's
proposed Tier-2 sample-based estimators (KSG, Kozachenko-Leonenko,
Vasicek) will land.  086 covered numerics, 087 covered missing primitives,
088 covered SOTA reference libraries.  This report is strictly about the
**function signatures, naming, and shape of the public API**.

**Headline.** The current information surface has six distinct
ergonomic divergences from itself.  Most are 087's "missing primitives"
under the hood, but the *shape* of the fixes — what you call the function,
what arguments it accepts, what unit it returns — has not yet been
decided and locks in for v1.0.  This report proposes the conventions.

---

## 1. Probability vs sample-based interfaces — the central question

### 1.1 Current state

Every information-theoretic primitive in repo today takes a
**probability vector** (or matrix), never raw samples:

| Function | Signature | Source |
|---|---|---|
| `compression.ShannonEntropy(probs []float64)` | probs | `compression/entropy.go:27` |
| `compression.JointEntropy(joint [][]float64)` | joint pmf | `:45` |
| `compression.ConditionalEntropy(joint [][]float64)` | joint pmf | `:69` |
| `compression.MutualInformation(joint [][]float64)` | joint pmf | `:94` |
| `compression.KLDivergence(p, q []float64)` | two pmfs | `:133` |
| `compression.CrossEntropy(p, q []float64)` | two pmfs | `:161` |
| `infogeo.KL/JS/TV/Hellinger/ChiSquared/Renyi(p, q)` | two pmfs | `infogeo/fdiv.go:65-227` |
| `prob.KLDivergenceNumerical(p, q Distribution, lo, hi, n)` | analytical | `prob/distribution.go:169` |

There is **no `EntropySamples([]float64)` anywhere in repo today.**  The
implicit contract is: "caller histograms the data themselves, or knows the
analytical pmf, and passes a pmf in."

### 1.2 Why this becomes a problem

The L12 cross-substrate inverse-consumers (Sensorhub histogram, Nexus
oracle bins, RubberDuck Granger lag selection — see
`info/mdl/doc.go:67-71`) all start from raw samples.  Their current
flow is:

```
samples -> hand-rolled histogram (50 bins, 10 bins, 40 cards) -> probs
        -> compression.ShannonEntropy(probs)
```

The hand-rolled histogram is the actual estimator — bin count is the
hyperparameter, but reality ships no helper.  Each consumer hard-codes
a different scalar.  The *math primitive* in compression/ is fine; the
*estimator* never lands in repo.

### 1.3 Proposed convention — explicit pluggable estimator family

Rather than a single ambiguous `Entropy()`, expose an estimator family.
This is the dit/infomeasure convention (see 088 §1.5 / §1.6):

```go
package info  // new sub-package; see §6 below

// Plug-in estimator: caller supplies discrete pmf.
func EntropyPlugin(probs []float64) (float64, error)

// Maximum-likelihood: caller supplies samples, function bins them.
// Bias correction is pluggable.
func EntropyMLE(samples []float64, opts EntropyOpts) (float64, error)

// Differential entropy from continuous samples (Kozachenko-Leonenko).
func EntropyKL(samples []float64, k int) (float64, error)

// Analytical: caller supplies a Distribution.
func EntropyAnalytical(d prob.Distribution) (float64, error)
```

Three benefits:

1. **The function name encodes the estimator.**  No hidden histogram
   binning inside `Entropy()`.  Callers who want a histogram opt in
   explicitly via `EntropyMLE` and pass `BinCount: 50` once.

2. **The bias correction is on the option struct.**  `EntropyOpts.Bias =
   BiasNone | BiasMillerMadow | BiasNSB | BiasJackknife` — see §5.

3. **The signature documents the input.**  `samples` and `probs` are
   distinct types semantically; the lint-level mistake of passing
   un-normalised counts where a pmf is expected (which `compression/`
   silently accepts today, see 086 finding) becomes impossible.

The plug-in / MLE / KL split mirrors exactly what 087 Tier-1 (plug-in
helpers) and Tier-2 (sample-based estimators) propose; the API shape is
the missing piece.

### 1.4 Should `EntropySamples()` exist as an alias?

**No.**  The naming "samples" is ambiguous: does it discretise (MLE) or
treat as continuous (KL)?  Force the caller to pick.  This is the same
reason `prob/` has separate `KLDivergence` (closed-form analytical) and
`KLDivergenceNumerical` (trapezoidal) — the suffix names the estimator,
not the input.

---

## 2. Discrete vs continuous — separate or unified?

### 2.1 Current

`compression/` is **purely discrete** (pmf in, scalar out).
`infogeo/fdiv.go` is **purely discrete** (validated pmf, sums to 1).
`prob.KLDivergenceNumerical` is **continuous** (Distribution interface,
trapezoidal integration).  No unification, no shared estimator surface,
and no overlap in the type signatures (`[]float64` pmf vs `Distribution`).

### 2.2 Proposed — keep them separate, name them so

Unifying discrete and continuous behind one function is the
infomeasure trick (088 §6: "selects backend by dtype").  In Go, with no
generics on numeric types and no runtime dtype, this would have to be:

```go
func Entropy(input interface{}) (float64, error)  // bad
```

Type-switching at runtime is a clear anti-pattern for a math library.
**Keep them separate, encode the domain in the function name:**

| Domain | Naming convention | Example |
|---|---|---|
| Discrete pmf | bare name | `Entropy(probs)` |
| Discrete samples (binned MLE) | `MLE` / `Plugin` suffix | `EntropyMLE(samples)` |
| Continuous samples (kNN) | algorithm-named suffix | `EntropyKL(samples, k)`, `EntropyVasicek(samples, m)` |
| Continuous distribution | `Analytical` suffix | `EntropyAnalytical(d Distribution)` |

This matches what `prob/` already does for KL (`KLDivergence` discrete,
`KLDivergenceNumerical` continuous).  Extend the convention rather than
introduce a new one.

---

## 3. Histogram-based density — implicit or explicit?

### 3.1 Current — none in repo

There is **no histogram primitive** in `reality/`.  Every consumer
hand-rolls.  The 087 Tier-2 plan calls for `histogramEntropy` with
Scott / Sturges / Freedman-Diaconis binning rules (~80 LOC).

### 3.2 Proposed — explicit, returned

Make the histogram the public-facing intermediate, not a hidden
side effect.  Two-step flow:

```go
type Histogram struct {
    Counts []int
    Edges  []float64
    N      int
}

// Bin choice is explicit and named.
func HistogramScott(samples []float64) Histogram
func HistogramFD(samples []float64) Histogram
func HistogramSturges(samples []float64) Histogram
func HistogramFixed(samples []float64, nbins int) Histogram

// Probability layer takes a Histogram, not raw samples.
func (h Histogram) Probabilities() []float64

// Entropy from histogram is the explicit composition.
func EntropyFromHistogram(h Histogram, opts EntropyOpts) float64
```

Three reasons:

1. **The histogram is reusable.**  Caller can compute `Entropy`,
   `MutualInformation` (with second variable's histogram), and KL all
   from the same intermediate without re-binning.

2. **The bin-count hyperparameter is visible.**  This is where Sensorhub's
   "50 bins" should go — at the call site, named `HistogramScott`,
   not buried.

3. **The bias correction sees the histogram, not just the probs.**
   Miller-Madow / NSB need `N` (the sample count), which is lost the
   moment you collapse to probabilities.  An `EntropyOpts.Bias` that
   takes a `Histogram` has access to `N`; one that takes only
   `[]float64` does not.  This is the central reason an `EntropyMLE
   ([]float64)` shortcut should bin internally rather than accept
   pre-binned probs — but expose the histogram path for power users.

---

## 4. Bits vs nats — documented? Both offered?

### 4.1 Current — split, not documented

| Package | Unit | Documented |
|---|---|---|
| `compression/entropy.go` | **bits** (uses `math.Log2`) | "in bits" stated in `ShannonEntropy` doc, **not** repeated for `JointEntropy`, `ConditionalEntropy`, `MutualInformation`, `KLDivergence`, `CrossEntropy`. |
| `infogeo/fdiv.go` | **nats** (uses `math.Log`) | "in nats" stated for `KL`, `JS`, `Renyi`; **not** for `Hellinger`, `TotalVariation`, `ChiSquared` (these are unit-free anyway, but the asymmetry is invisible to a reader). |
| `info/mdl/` | **nats** primarily, with `Bits` variant for `UniversalIntegerCodeLength` | Yes — sole package with both surfaces. |

A consumer importing `compression.KLDivergence` and `infogeo.KL` will
get answers differing by `log(2) ≈ 0.693` and never know.  This is the
single most likely silent bug in the entire information surface.

### 4.2 Proposed — `Bits`/`Nats` suffix on every function, both surfaces

Adopt the `info/mdl/` convention universally:

- Every entropy / divergence function exists in both unit conventions.
- Suffix is mandatory: `EntropyBits`, `EntropyNats`, `KLBits`, `KLNats`.
- The bare names `Entropy` and `KL` either alias to one (default
  documented) or are removed.  **Recommend remove**, force the caller
  to pick.  This kills the silent unit mismatch.
- One function pair shares the loop body; the only difference is `Log2`
  vs `Log` at the end.  ~5 LOC overhead per primitive, ~75 LOC total
  across the whole surface.

The dit / infomeasure / pyitlib libraries all default to bits; JIDT
defaults to nats; NPEET nats.  No global default is defensible.  Force
the choice.

### 4.3 Migration implications

Existing `compression.ShannonEntropy` etc. become
`compression.ShannonEntropyBits` (no behaviour change, type-checked
clarity).  `infogeo.KL` becomes `infogeo.KLNats`.  Same physical math,
~30 line patch + golden-file regeneration with new file names.  This is
a one-shot rename; do it before v1.0 ships.

---

## 5. Estimator pluggability — bias correction

### 5.1 Current — none

`compression.ShannonEntropy(probs)` has no notion of sample count, and
therefore cannot apply any bias correction.  `EntropyMLE` of a finite
sample of size `N` is biased by `~(K-1)/(2N)` (where K = number of
non-empty bins) — the classic Miller-Madow correction.  No bias-
corrected entropy exists in repo.

### 5.2 Proposed — option struct, three families

```go
type BiasCorrection int

const (
    BiasNone        BiasCorrection = iota  // plug-in MLE
    BiasMillerMadow                       // (K-1)/(2N) additive correction
    BiasJackknife                         // leave-one-out resampling
    BiasNSB                               // Nemenman-Shafee-Bialek 2002
    BiasGrassberger                       // Grassberger 1988 finite-sample
)

type EntropyOpts struct {
    Bias    BiasCorrection
    BinRule BinRule  // Scott/FD/Sturges/Fixed
    NBins   int      // when BinRule == BinFixed
    Unit    Unit     // UnitBits | UnitNats; mandatory, no default
}

func EntropyMLE(samples []float64, opts EntropyOpts) (float64, error)
```

Same options struct flows through `MutualInformationMLE`,
`KLMLE` (when one or both arguments are samples).  This is a small
expansion of surface area in exchange for ergonomic completeness.

**Why an enum, not interface{}?**  Generic-over-bias-correction is
overkill for five canonical implementations.  Enum is dit's choice
(`H(d, rv=None, base='e')` plus separate `entropy_jackknife`,
`entropy_miller_madow`, `entropy_nsb` functions).  The Go-idiomatic
shape is the option struct; the enum is the canonical set.

### 5.3 Pluggable from outside?

**No.**  Bias correctors are not the kind of thing user code should
inject — they are textbook, fixed, and shippable.  Resist the urge to
expose a `BiasCorrector interface{ Correct(...) float64 }`; that's
over-design.  Five enum values, five inline implementations, ~150 LOC
total.

---

## 6. KL signature — KL(p, q) — which is reference?

### 6.1 Current — uniformly `KL(p, q)` meaning `KL(p || q)`

Both `compression.KLDivergence(p, q)` and `infogeo.KL(p, q)` document
the convention as `KL(P || Q)` where **P is the "true" distribution
and Q is the "reference" / approximating** distribution.  Doc strings:

- `compression/entropy.go:124`: "measures how distribution P diverges
  from reference distribution Q"
- `infogeo/fdiv.go:56`: `KL(p || q) = sum_i p_i * log(p_i / q_i)`

Both implementations treat `p_i = 0` as 0-contribution and `p_i > 0,
q_i = 0` as `+Inf`.  Asymmetry is documented in `compression/` (line
130) but **not** in `infogeo/` — fix.

### 6.2 Proposed — keep `(p, q)` order, document reference once

The convention `KL(p, q) = KL(p || q)` is uniform across SciPy,
TensorFlow, PyTorch, dit, JIDT.  Match it.

**Documentation upgrade:**

- Every divergence function whose arguments are not symmetric should
  state the convention explicitly *and* state which argument is the
  "reference" / "model" in the Bayesian VI context.
- Recommend a package-level doc comment in `infogeo/fdiv.go`:
  ```
  // Convention: in all asymmetric divergences D(p, q), the FIRST
  // argument p is the "true" / "data" distribution, the SECOND
  // argument q is the "model" / "reference" distribution.  This
  // matches SciPy's scipy.stats.entropy(pk, qk) and dit's
  // dit.divergences.kl_divergence(p, q).
  //
  // For variational inference: forward-KL is KL(p_data, q_model);
  // reverse-KL is KL(q_model, p_data) (also exposed as ReverseKL).
  ```
- `ReverseKL` (`infogeo/fdiv.go:85`) is correctly named — keep.

### 6.3 ChiSquared, Renyi — same convention

`infogeo.ChiSquared(p, q)` and `infogeo.Renyi(p, q, alpha)` already
follow `(p, q)` order with `q` as reference.  No change needed except
docstring uniformity.

### 6.4 Cross-entropy — naming subtlety

`compression.CrossEntropy(p, q)` returns `H(p, q) = -Σ p_i log q_i`.
This is `H(p) + KL(p || q)`.  The convention is consistent with KL but
the docstring should make the identity explicit (line 158 already
does — good).

---

## 7. Joint distribution argument shape — `[][]float64` vs flat

### 7.1 Current

`compression.JointEntropy([][]float64)`, `ConditionalEntropy([][]float64)`,
`MutualInformation([][]float64)`.  The matrix is the joint pmf
`P(X=i, Y=j) = joint[i][j]`.

### 7.2 Issues

1. **Ragged matrices accepted silently.**  `MutualInformation` (lines
   107-111) computes `maxCols` defensively, suggesting the API accepts
   ragged input.  This is wrong shape for a joint pmf — every row has
   the same `|Y|` cardinality.  Tighten to: validate that all rows have
   equal length, return `ErrJointShape` otherwise.

2. **No `[][]int` count variant.**  Consumers with raw counts must
   normalise to floats first, losing `N` for bias correction.  Add
   `JointEntropyMLE(counts [][]int, opts EntropyOpts) float64` for the
   counts path.

3. **3+ dimensions not supported.**  `total correlation` and PID need
   ≥3 variables.  087 Tier-3 will need this.  Recommend tensor shape
   `[]float64` flat + `dims []int` (numpy-style) over `[][][]...float64`
   nesting — the latter doesn't generalise past 3 dimensions in Go.
   This is a v2 decision; flag it now so v1 doesn't paint into a corner.

### 7.3 Proposed — same shape, tighter validation, add Counts variants

Keep `[][]float64` for the bivariate case (it's idiomatic and the
existing API).  Add validation.  For ≥3 variables, when it lands in v2,
use `Tensor{Data []float64, Dims []int}` — do not nest deeper.

---

## 8. Distribution interface — should the info surface accept it?

### 8.1 Current

`prob.Distribution` (`prob/distribution.go:27`) has `PDF` and `CDF`.
`prob.KLDivergenceNumerical(p, q Distribution, lo, hi, nSteps)` is the
only info function that consumes it.

### 8.2 Proposed — yes, expand to entropy and all f-divergences

```go
// Continuous-distribution closed-form / numerical.
func EntropyAnalytical(d prob.Distribution, lo, hi float64, nSteps int) float64
func KLAnalytical(p, q prob.Distribution, lo, hi float64, nSteps int) float64
func JSAnalytical(p, q prob.Distribution, lo, hi float64, nSteps int) float64
// ... etc
```

These are all trapezoidal-rule one-liners over the existing
`Distribution.PDF`.  Cost: ~10 LOC each, ~60 LOC total.  Benefit:
parity with the discrete surface — every divergence has both a
`(p, q []float64)` discrete and `(p, q Distribution, ...)` continuous
form.  The naming `Analytical` is slightly inaccurate (it's actually
numerical-via-PDF), but matches existing `KLDivergenceNumerical` if we
keep the latter as a deprecated alias.

**Better naming:** `EntropyContinuous`, `KLContinuous`, etc.  Drop
"Analytical" since the implementation is numerical integration.

### 8.3 Closed-form Gaussian KL — special case

`KL(N(μ₁,σ₁) || N(μ₂,σ₂))` has a closed form (087 T1.7).  This belongs
on the `NormalDist` type, not the generic `Distribution` interface:

```go
func (d *NormalDist) KLTo(other *NormalDist) float64
func (d *NormalDist) Entropy() float64
```

Method-on-type rather than free function — the `Distribution` interface
surface stays narrow (just `PDF`/`CDF`), and dist-specific closed forms
live on the concrete types.  Same pattern for `BetaDist`,
`ExponentialDist`, `UniformDist` (all have closed-form entropy).

---

## 9. Package layout — where should the API live?

### 9.1 Current — three packages, one missing

The information-theoretic surface is split:

- `compression/entropy.go` — discrete bits-domain primitives.
- `infogeo/fdiv.go` — discrete nats-domain f-divergences.
- `prob/distribution.go` — KLDivergenceNumerical only.
- `info/lz/`, `info/mdl/` — sequence complexity + codelength.

There is no top-level `reality/info/entropy.go`.  The `info/`
sub-package was carved out for LZ76 and MDL only; the discrete
entropies stayed in `compression/` for historical reasons (compression
ratio is bounded by entropy).

### 9.2 Proposed — promote to `reality/info/`

Move (or alias) the discrete entropy / divergence surface into a new
`info/entropy.go` and `info/divergence.go`, leaving `compression/` to
re-export the bits-domain primitives for back-compat.  This is the
home all of 087 Tier-1, Tier-2, Tier-3 expects.

Result:

```
reality/info/
  doc.go              package overview, bits-vs-nats convention
  entropy.go          EntropyBits, EntropyNats, EntropyMLE, EntropyKL, ...
  divergence.go       KLBits, KLNats, JS, TV, Hellinger, ChiSquared, Renyi, ...
  histogram.go        HistogramScott, HistogramFD, HistogramSturges
  bias.go             Miller-Madow, NSB, jackknife correctors
  options.go          EntropyOpts, MIOpts, KLOpts, BinRule, BiasCorrection
  lz/                 LZ76 (existing)
  mdl/                MDL/NML (existing)
```

`infogeo/fdiv.go` *information geometry* primitives (KL on simplex,
Bregman, MMD) stay in `infogeo/` — that package is about *geometric
structure* of distributions, not measurement.  But re-export aliases
ensure no double-source-of-truth.

---

## 10. Summary — recommended API conventions

| Question | Recommendation |
|---|---|
| `Entropy(probs)` vs `EntropySamples(samples)`? | Neither: `EntropyPlugin` / `EntropyMLE` / `EntropyKL` / `EntropyAnalytical` — name the estimator. |
| Discrete vs continuous: separate or unified? | Separate, distinguished by suffix (`MLE` / `Vasicek` / `KL` / `Continuous`). |
| Histogram-based density implicit or explicit? | Explicit: `Histogram` is a returned type, reused across multiple measures. |
| Bits vs nats convention? | Both offered, mandatory `Bits`/`Nats` suffix on every function, no global default. |
| Estimator pluggability for bias correction? | Enum `BiasCorrection` on `EntropyOpts` struct; five canonical correctors (None, Miller-Madow, jackknife, NSB, Grassberger). |
| `KL(p, q)` argument order? | `(p, q)` = `KL(p || q)`, p is data, q is reference. Matches SciPy/dit/JIDT. Document once at package level. |
| Joint shape `[][]float64`? | Keep for bivariate; tighten validation (no ragged); add `[][]int` counts variants. |
| `Distribution` interface for info funcs? | Yes, parallel `Continuous` suffix variants for every divergence. Closed-form pairs as methods on concrete types. |
| Package home? | New `reality/info/`, with `compression/` and `infogeo/` keeping back-compat aliases. |

**Single highest-leverage commit:** the `Bits`/`Nats` suffix rename
across `compression/` and `infogeo/`.  It is mechanical, ~30 LOC of
diff, regenerates ~6 golden files with renamed test names, and kills
the only silent-unit-mismatch bug class in the entire information
surface.  Do it before any new estimator lands so the new code can
adopt the convention from day one.

**Single highest-leverage new infrastructure:** the `EntropyOpts` /
`Histogram` / `BinRule` / `BiasCorrection` types in
`reality/info/options.go`.  These types are the *contract* every
sample-based estimator from 087 Tier-2 / Tier-3 will share.  Define
them now, populate the implementations later — but do not let each
estimator invent its own option-struct shape.

---

## Files referenced

- `C:\limitless\foundation\reality\compression\entropy.go`
- `C:\limitless\foundation\reality\infogeo\fdiv.go`
- `C:\limitless\foundation\reality\prob\distribution.go`
- `C:\limitless\foundation\reality\info\mdl\doc.go`
- `C:\limitless\foundation\reality\info\lz\doc.go`
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\086-info-numerics.md`
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\087-info-missing.md`
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\088-info-sota.md`

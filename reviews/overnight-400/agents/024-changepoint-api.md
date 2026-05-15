# 024 | changepoint-api | streaming vs batch API, multi-variate handling

**Scope:** API ergonomics only. Capability/numerical findings deferred to 021/022/023 — this review covers the user-facing surface.

**Verdict:** changepoint exposes a clean, single-detector streaming object (`Bocpd`), but it ships **zero of the four ergonomics affordances** that the topic prompt enumerates as table stakes for a regime-detection package: no batch counterpart, no multi-variate path, no result struct, no Reset. The configurability axes (hazard / observation model) are not pluggable — both are hard-coded to `H = 1/lambda` and Normal-Inverse-Gamma. The `New(Config)` constructor and `Update`/query split are good. Below: 11 findings ranked by user-impact, then a sibling-package comparison.

---

## A. Streaming vs batch — no batch counterpart exists

**Finding A1: There is no `Detect(xs []float64) Result` offline API.** Only `(*Bocpd).Update(x float64) ([]float64, error)` exists. A consumer with a slice of 5,000 daily returns who wants "tell me where the regime breaks are" must:

```go
b, _ := changepoint.New(changepoint.DefaultConfig())
maps := make([]int, len(xs))
for i, x := range xs {
    if _, err := b.Update(x); err != nil { return err }
    maps[i] = b.MapRunLength()
}
// then post-process the maps[] series themselves to extract changepoint indices
```

That post-processing — find indices where MAP collapses, threshold `ChangePointProbabilityWithin`, etc. — is exactly what every consumer will reinvent. Compare `timeseries/dcc/dcc.go:127`:

```go
// FilterSeries applies the DCC update over a full standardised-residual
// series and writes the conditional correlation matrices for every step
// into rSeries (length n*k*k row-major).
//
// Most callers will use this as the canonical multi-step DCC recursion;
// individual single-step Updates are useful for streaming applications.
func (p Params) FilterSeries(zSeries []float64, n int, rSeries []float64) error
```

dcc shipped both surfaces and named the streaming one as the secondary path. changepoint should mirror this. Suggested signature:

```go
type DetectionResult struct {
    Posteriors  [][]float64 // P[t][r] for each timestep
    MapRunLen   []int       // argmax_r P[t][r]
    ChangePoints []int      // indices where MAP collapsed (high-confidence reset)
    LogEvidence float64     // sum of step-wise log p(x_t | x_{1:t-1}); for model selection
}
func Detect(xs []float64, cfg Config) (*DetectionResult, error)
```

Without this, every downstream consumer (the 5 named flagships in `doc.go`) will hand-roll the "loop+collect+threshold" wrapper differently. That is exactly the failure mode `doc.go` calls out in its motivation: *"each [hand-rolled detector] rolling a different bug."* The wrapper itself is the next bug.

**Finding A2: `Update` returns `[]float64` (raw posterior), not a result struct.**

```go
func (b *Bocpd) Update(x float64) ([]float64, error)
```

The topic prompt asks: *"what does Update return — `{Posterior, MAPRunLength, ChangeProbability}`?"* Answer: just the posterior. Every other quantity (MAP, expected run-length, change-point probability, regime mean/variance) requires a separate method call after `Update` returns:

```go
b.MapRunLength()
b.ExpectedRunLength()
b.ChangePointProbability()
b.ChangePointProbabilityWithin(window)
b.CurrentRegimeMean()
b.CurrentRegimeVariance()
```

Each of those re-scans the full posterior (linear in `len(b.p)` ≤ `RMax+1` = 501). At 60 FPS with 4 derived quantities queried per step, that's `4 * 501 = 2004` extra slice scans per second per detector that a single struct return would amortise. More importantly: the API forces callers to **know which of 6 query methods to call** and in what combination. A `StepResult { MapRunLen, ChangeProb, RegimeMean }` returned from `Update` collapses the discovery surface. Recommended:

```go
type StepResult struct {
    MapRunLength       int
    ExpectedRunLength  float64
    ChangeProbability  float64 // P(r_t = 0)
    LogEvidence        float64 // log p(x_t | x_{1:t-1}) — currently not exposed at all
    RegimeMean         float64
    RegimeVariance     float64
}
func (b *Bocpd) Update(x float64) (StepResult, error)
// keep RunLengthPosterior() as the explicit "I want the raw vector" escape hatch
```

The current signature also leaks a fresh allocation per Update (line 287: `out := make([]float64, len(newP)); copy(out, newP)`) for callers who don't need the raw posterior. A struct return makes that allocation opt-in via a separate `RunLengthPosterior()` call.

---

## B. Multi-variate — fundamentally unsupported

**Finding B1: `Update(x float64)` is scalar-only.** The signature is `func (b *Bocpd) Update(x float64) ([]float64, error)`. There is no `UpdateVec(x []float64)` or generic shape. The Normal-Inverse-Gamma observation model in `bocpd.go:79` is hardcoded scalar: `mu, kappa, alpha, beta` are all `[]float64` (one per run-length, not one per dimension).

This is a hard architectural ceiling: vector observations require a different conjugate prior (Normal-Inverse-Wishart for full covariance, or independent NIGs per coordinate as a poor-man's diagonal). The current code can model `R^1` only. The topic prompt asks: *"does BOCPD handle vector observations? What's the shape contract?"* — answer is **no, and there is no shape contract at all**. The package is silent on multi-variate either way. Doc.go does not mention it.

This is the single largest API-surface gap. Multi-variate BOCPD is in every modern reference implementation (R `bcp`, Python `bayesian_changepoint_detection`, MATLAB's `findchangepts` for vector signals). Without it, the package cannot be a substrate for the multi-asset / multi-sensor / multi-channel consumers (relic-insurance is multi-asset, witness is multi-sensor — both are named in `doc.go:24` as candidate consumers). Each of those will fall back to per-coordinate independent BOCPD and OR-combine the alarms, which is statistically wrong (ignores cross-correlation under the prior).

Recommended shape contract once added:

```go
// UpdateVec ingests a d-dimensional observation. Length of x must equal d
// configured via Config.Dim (default 1, equivalent to current Update).
func (b *Bocpd) UpdateVec(x []float64) (StepResult, error)

// Single-scalar convenience preserved as wrapper:
func (b *Bocpd) Update(x float64) (StepResult, error) { return b.UpdateVec([]float64{x}) }
```

Note that this is a strict superset of the current API and only requires either (a) the Normal-Inverse-Wishart conjugate prior or (b) per-coordinate independent NIG. Either is a math addition, not an API redesign. The API hooks should be carved now.

---

## C. Detector instantiation — the one thing that's right

**Finding C1: `New(cfg Config) (*Bocpd, error)` is the canonical Go constructor pattern**, with `DefaultConfig()` as the zero-friction entry point. This is the cleanest constructor in the repo I've seen — `control.NewPIDController(...)` takes 5 positional float64s, `dcc.Params{...}` is a struct literal with no validation hook, but `changepoint.New(DefaultConfig())` validates on entry, returns sentinel errors, and provides a sane default. Keep this.

```go
func New(cfg Config) (*Bocpd, error)         // bocpd.go:121
func DefaultConfig() Config                  // bocpd.go:111
type Config struct { Prior NigPrior; RMax int; Lambda float64 }
```

The one ergonomic miss: `Config` is fields-only with no functional-options pattern. Adding hazard and observation-model pluggability (see findings E1, F1) is a breaking change to `Config` because each new field must be set by every existing caller. Functional options would future-proof:

```go
b, err := changepoint.New(
    changepoint.WithPrior(...),
    changepoint.WithLambda(100),
    changepoint.WithHazard(hazard.Logistic{...}),
    changepoint.WithObservationModel(obsmodel.Gaussian{...}),
)
```

This matters now because hazard and observation-model pluggability is named in the topic prompt and called out in agent 023 as ~200 LOC of work. Carve the API before the sibling reviews land their fixes.

---

## D. Reset semantics — missing entirely

**Finding D1: `Bocpd` has no `Reset()` method.** A consumer that wants to wipe state and re-use the allocated slices (the `p`, `mu`, `kappa`, `alpha`, `beta` backing arrays) must construct a fresh detector via `New(cfg)`, throwing away the old one to GC. Compare `control/pid.go:119`:

```go
// Reset clears the controller's internal state (integral sum and previous
// error), returning it to its initial condition. Gains and output limits
// are preserved.
func (p *PIDController) Reset() {
    p.integralSum = 0
    p.prevError = 0
}
```

PIDController gets this right. Bocpd does not. The implementation would be ~10 lines:

```go
func (b *Bocpd) Reset() {
    b.t = 0
    b.p = b.p[:1];     b.p[0] = 1.0
    b.mu = b.mu[:1];   b.mu[0] = b.prior.Mu0
    b.kappa = b.kappa[:1]; b.kappa[0] = b.prior.Kappa0
    b.alpha = b.alpha[:1]; b.alpha[0] = b.prior.Alpha0
    b.beta = b.beta[:1];   b.beta[0] = b.prior.Beta0
}
```

This bonus also fixes the per-Update slice churn that 021 flagged (~6 allocations × every Update) — a Reset that preserves the underlying capacity lets a hot-loop consumer process a 10k-step backtest in 5 contiguous detector lifecycles for the cost of one allocation.

A complementary `ResetTo(snapshot State)` — for "rewind to step 100" use cases like backtesting — is not table-stakes but would compose well with the offline `Detect` from finding A1.

---

## E. Hazard function — not pluggable

**Finding E1: Hazard is hardcoded constant.** `bocpd.go:147`:

```go
func (b *Bocpd) hazard(r int) float64 {
    _ = r
    return 1.0 / b.lambda
}
```

The topic prompt asks: *"is it pluggable? hazard.Constant, hazard.Logistic?"* Answer: no. The signature `hazard(r int) float64` already takes `r` (the unused parameter is the giveaway that someone planned for non-constant hazards) but `Config` has no hook to override the function. No `Hazard` interface exists; no `hazard` sub-package exists.

The doc-comment on `ChangePointProbability` (`bocpd.go:316-330`) explicitly notes the API design failure this causes:

> Note: under a constant hazard rate H = 1/lambda, this quantity is algebraically equal to H at every step ... It is therefore *not* a useful alarm signal on its own ... ChangePointProbability is exposed for reference and for non-constant-hazard variants where the cancellation does not occur.

The package ships an alarm signal that is **algebraically a constant** because the only hazard model supported is the one that makes the signal degenerate. Fixing this requires adding the hazard interface plus at least one non-constant implementation. Recommended (this matches 023's R1):

```go
// in changepoint/hazard:
type Hazard interface {
    Probability(runLength int) float64
}
type Constant struct { Lambda float64 }
type Logistic struct { Lambda float64; Slope float64; Threshold int }
type Periodic struct { Period float64; Phase float64 }
```

Then `Config.Hazard Hazard` (back-compat: nil → Constant{Lambda: cfg.Lambda}).

---

## F. Observation model — not pluggable

**Finding F1: The Normal-Inverse-Gamma model is welded into `Bocpd`.** Lines 86-99 (the `Bocpd` struct) hold `mu, kappa, alpha, beta` slices directly; lines 247-278 (the sufficient-statistic update inside `Update`) compute the NIG conjugate update inline; lines 184-191 (the predictive computation) compute the Student's-t predictive inline.

The topic prompt asks: *"pluggable? gaussian, student-t, gamma, multinomial?"* Answer: **only NIG**, no abstraction layer. To add a Poisson-Gamma model (count data — change-point in event rates), Bernoulli-Beta (success-rate change-points), or Multinomial-Dirichlet (categorical), the user must fork the package.

Recommended observation-model interface (matches 023's R1):

```go
// in changepoint/obsmodel:
type Model interface {
    // PredictiveLogPDF returns log p(x | sufficient_stats[runLength])
    PredictiveLogPDF(x float64, runLength int) float64
    // Update absorbs observation x into sufficient_stats[runLength] producing
    // sufficient_stats[runLength+1].
    Update(x float64, runLength int)
    // Reset clears state for run-length 0 back to the prior.
    Reset()
}

type Gaussian struct { Prior NigPrior; ... }   // current behaviour
type Poisson  struct { Alpha0, Beta0 float64 } // counts
type Bernoulli struct { Alpha0, Beta0 float64 } // success rate
type StudentT struct { ... }                    // robust to outliers
```

Then `Config.Observation Model` (back-compat: nil → Gaussian{Prior: cfg.Prior}).

The vector observation case (finding B1) is naturally a `MultivariateGaussian` that satisfies a `VectorModel interface`. Carving the scalar `Model` interface now lets the multi-variate work be a parallel addition rather than a re-architecture.

---

## G. Concurrency — documented, single-line

**Finding G1: `Bocpd` is documented as not goroutine-safe** (`bocpd.go:84`):

> Bocpd is not safe for concurrent use. Wrap in a mutex if shared.

This is honest and matches PID, autodiff, and dcc conventions in the repo (none of them are concurrent-safe; only signal/ functions are pure). No improvement needed here — but the prompt asks about "Read+Write" concurrency, and the answer is **none of the read-only methods are safe to call while another goroutine is in `Update`** because `Update` mutates `b.p`, `b.mu`, etc. in-place at lines 280-285. A `(b *Bocpd) Snapshot() Bocpd` method returning a deep copy would let consumers do "Update on writer goroutine, read off snapshot from N other goroutines" without a mutex, but that's a niche addition (deferred).

The `RunLengthPosterior()` method **does** copy (`bocpd.go:310-314`), which is the right behaviour for a method that lets consumers retain the result. But it doesn't help with concurrency because it can race against a concurrent `Update`.

---

## H. Sibling-package comparison

| Package | Streaming primitive | Batch counterpart | Reset() | Result struct | Pluggable model |
|---------|--------------------:|-------------------|--------:|--------------:|----------------:|
| `changepoint.Bocpd` | `Update(x) []float64` | **none** | **none** | **none (raw slice)** | **none (NIG-only)** |
| `control.PIDController` | `Update(sp, m, dt) float64` | n/a (controller) | yes | n/a (scalar) | n/a |
| `timeseries/dcc.Params` | `Update(z, Q, qOut)` | `FilterSeries(...)` ✓ | n/a (params struct, stateless) | n/a | n/a |
| `timeseries/garch.Model` | (no streaming) | `Filter(eps, sigma2, z)` | n/a | n/a | n/a |
| `signal.*` | (pure functions only) | n/a | n/a | n/a | n/a |
| `audio/*` (DegradationTracker) | `Update(...)` | (per agent 009) absent | (per 009) absent | mixed | absent |

**Pattern:** dcc is the closest sibling and got it right (single-step Update + multi-step FilterSeries, named exactly that way in the doc). changepoint should mirror dcc's framing: ship the offline `Detect` as the canonical entry point, keep `Update` as the streaming escape hatch. Agent 009 already noted that the audio package's stateful trackers should *clone changepoint.Bocpd's streaming idiom*. This review's complement is: changepoint should clone dcc's batch-counterpart idiom.

---

## I. Consumer use sites — none external, one internal

**Finding I1: Zero external consumers.** Substring grep on `github.com/davly/reality/changepoint` across `C:\limitless\` returns:

- `changepoint/doc.go` (self)
- `changepoint/infogeo_test.go` (the only consumer — see I2)
- `reviews/overnight-400/agents/{022,023}-changepoint-*.md` (sibling reviews)
- `reviews/overnight-400/PROGRESS.md` (the log)

The 5 named flagships in `doc.go:25-29` (relic-insurance, triage, witness, watchtower, narrator) are listed as *"hunt-citations not import-citations as of 2026-05-05."* No consumer outside this repo imports the package. The API window is **fully open** — no back-compat constraints.

**Finding I2: The internal consumer (`infogeo_test.go`) reveals one ergonomic gap.** The test at `infogeo_test.go:57-66` does this:

```go
var preSnapshot, postSnapshot []float64
for i, x := range xs {
    if _, err := bocpdFull.Update(x); err != nil { ... }
    if i == preCP-1 {
        preSnapshot = append([]float64(nil), bocpdFull.RunLengthPosterior()...)
    }
}
postSnapshot = append([]float64(nil), bocpdFull.RunLengthPosterior()...)
```

Note three things:
1. The result of `Update` is discarded (`_, err`) and `RunLengthPosterior()` is called instead — because `Update` returns the wrong shape (raw slice) when the consumer wants a snapshot mid-loop, and there's no convenient way to "snapshot the posterior at a specific step."
2. The consumer manually re-allocates with `append([]float64(nil), ..., ...)` — `RunLengthPosterior` already copies, so this is a defensive double-copy. That's confusion the API can prevent by either renaming (`PosteriorRef` for the no-copy view, `Posterior` for the copy) or by documenting the copy semantic in the method name.
3. The whole test exists because there's no offline `Detect(xs)` API that returns posteriors at every step in a single call.

The first verified consumer in the package is already paying ergonomic tax. That's a strong signal.

---

## J. Documentation surface — not in CLAUDE.md or README

**Finding J1: changepoint is missing from the package table** in `CLAUDE.md` (which lists 22 packages) and the design doc. The repo's project instructions claim **22 packages (1,965 tests)**, but `changepoint` (with 41 tests across `bocpd_test.go`+`bocpd_expansion_test.go`+`infogeo_test.go`) is not in that table. This is an ergonomic miss: the discovery path "I have a regime-detection problem → I open CLAUDE.md → I find the package" is broken. (This is the same finding agent 014 made for autodiff.)

The package count and test count in the project instructions are stale. Either changepoint should be added to the table or the count corrected. (Out-of-scope nit but worth flagging since it affects API discoverability.)

---

## K. Ranked fix-set

Cumulative cost ~600 LOC (most are pure additions, full backward compat):

1. **`StepResult` struct returned from `Update`** (~30 LOC) — fixes A2, removes the 4×501 query-loop overhead, makes the canonical 3 derived quantities single-call. Deprecate the slice return as `(b *Bocpd) Update(x) (StepResult, error)`; keep `RunLengthPosterior()` as the explicit raw-vector accessor. **Highest ergonomic impact.**

2. **`Detect(xs []float64, cfg Config) (*DetectionResult, error)`** (~80 LOC) — fixes A1, mirrors `dcc.FilterSeries`, gives every offline consumer the natural entry point. The result struct exposes `LogEvidence` for model selection (currently not exposed at all).

3. **`(b *Bocpd) Reset()`** (~10 LOC) — fixes D1, mirrors `PIDController.Reset()`, also fixes the per-Update slice-churn allocation issue 021 flagged.

4. **`Hazard interface` + `changepoint/hazard` sub-package** (~80 LOC) — fixes E1, makes `ChangePointProbability` actually a useful alarm signal. Composes with 023's R1.

5. **`Model interface` + `changepoint/obsmodel` sub-package** (~150 LOC) — fixes F1, unblocks Poisson/Bernoulli/StudentT consumers without forking.

6. **`UpdateVec(x []float64)` + `MultivariateGaussian` model** (~200 LOC) — fixes B1, the largest user-facing gap. Naturally falls out of (5).

7. **Functional options for `New`** (~40 LOC) — fixes the latent fragility of (4)+(5)+(6) growing `Config`. Optional but cheap.

8. **Add changepoint to CLAUDE.md package table; fix test count** (~5 LOC) — fixes J1, cheapest item.

9. **`(b *Bocpd) Snapshot() *Bocpd`** (~20 LOC) — concurrency escape hatch; not blocking but cheap.

Items 1, 2, 3, 8 are pure ergonomics with zero math change and would land cleanly today. Items 4, 5, 6, 7 are the architectural carve that 023 also recommends — coordinate so they land together.

---

**Files referenced (absolute paths):**

- `C:\limitless\foundation\reality\changepoint\bocpd.go`
- `C:\limitless\foundation\reality\changepoint\doc.go`
- `C:\limitless\foundation\reality\changepoint\bocpd_test.go`
- `C:\limitless\foundation\reality\changepoint\bocpd_expansion_test.go`
- `C:\limitless\foundation\reality\changepoint\infogeo_test.go`
- `C:\limitless\foundation\reality\control\pid.go` (Reset reference)
- `C:\limitless\foundation\reality\timeseries\dcc\dcc.go` (FilterSeries reference)
- `C:\limitless\foundation\reality\timeseries\garch\garch.go` (Filter-buffer reference)
- `C:\limitless\foundation\reality\signal\filter.go` (pure-function sibling)
- `C:\limitless\foundation\reality\CLAUDE.md` (missing-from-table)

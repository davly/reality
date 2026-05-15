# 021 | changepoint-numerics

**Scope.** Numerical-correctness audit of `C:\limitless\foundation\reality\changepoint\` (one source file, two test files, one cross-package composition test). The package implements **only BOCPD**; the topic prompt's checklist of CUSUM / PELT / BinSeg / WBS / KCPD / E-divisive is therefore a "missing primitives" finding for slot 022 and is reported below in §10. This audit is scoped to BOCPD, the math that is actually written.

**TL;DR.** `bocpd.go` is a 402-line, single-detector implementation of Adams-MacKay 2007 with NIG-conjugate observation model. The math is textbook-correct, the API is clean, and 40 tests pass. The numerical implementation honours **most** of the log-space discipline the topic asks about — `studentTLogPDF` uses `math.Log1p`, `logSumExp` is implemented and used, the renormalisation max-subtracts before `math.Exp`, NaN/Inf inputs are guarded at `Update`, and update-failure preserves prior state (`TestUpdate_NaN_StatePreserved` pins this). The remaining gaps are real but bounded: (a) `math.Log(1.0 - h)` instead of `math.Log1p(-h)` silently underflows to exactly 0 for `Lambda >= 1e16` and loses 5 sig figs at `Lambda >= 1e8` (which the test suite uses), neutralising the renormalisation signal at the high-`Lambda` end users reach for "no expected change-points"; (b) `math.Log(b.p[r])` is called on every step on every run-length even though `p[r]` can be exactly zero from a prior step's renormalisation cliff and even though the same quantity has been re-`exp`-ed seconds earlier; (c) the truncation at `R_max+1` silently drops the *growth* contribution of the oldest run-length while keeping its *reset* contribution, biasing the truncated posterior toward false alarms by ~`p[R_max] * (1-1/lambda)` per step; (d) per-step allocation pattern is `make()`-on-every-Update for six full-length slices, no scratch buffer, ~`6 * len(p) * 8` bytes of garbage per step; (e) zero ROC / monte-carlo / false-alarm-rate / detection-rate calibration tests exist anywhere — the topic prompt explicitly asks for these; (f) IEEE-754 edge cases for `Update` cover NaN/+Inf but not -Inf, subnormals, ±0.0, max-finite; (g) golden-file cross-language vectors do not exist for changepoint at all (the cross-language port axis CLAUDE.md describes is unfilled here).

---

## 1. What this package contains, and what the topic asks for that is not here

| Topic checklist item | In `changepoint/`? | Where else in `reality/`? |
|---|---|---|
| BOCPD (Bayesian online) | yes (`bocpd.go`) | — |
| CUSUM (signed / two-sided) | **no** | not anywhere |
| PELT (pruned exact linear time) | **no** | not anywhere |
| BinSeg / Wild BinSeg | **no** | not anywhere |
| Kernel CPD / MMD₂ | **no** | not anywhere |
| E-divisive | **no** | not anywhere |
| ROC / monte-carlo calibration tests | **no** | not anywhere |
| Online vs offline split | online only | — |

The package doc-comment (`doc.go:1-3`) is upfront: "implements Bayesian Online Change-Point Detection (BOCPD) per Adams-MacKay 2007." The package name `changepoint` (singular) and the absence of `cusum.go`, `pelt.go`, `binseg.go` make the scope clear; the topic prompt's reading list is aspirational. The slot-022 "missing primitives" review is the right place to file the additions; this audit only covers BOCPD.

---

## 2. BOCPD log-space discipline — what is correct

The implementation already does the right thing in three places where naive code typically fails:

1. **Predictive density in log-space** (`studentTLogPDF`, line 156-163): correctly uses `math.Log1p(z*z/df)` instead of `math.Log(1 + z*z/df)`. This matters at the start of a new regime where `z` is small and `1 + z²/df ≈ 1`. Pinned by `TestStudentTLogPDF_StandardCauchy_AtZero` (golden value at 0) and `TestStudentTLogPDF_LocationShift_IsTranslationInvariant` (translation invariance). Both are exact-equality (1e-12) tests.
2. **`logSumExp` is implemented and used** (lines 294-305): the standard `max + log1p(exp(min - max))` pattern with `-Inf` short-circuits on either argument. Pinned by 7 test cases including `TestLogSumExp_BothNegInf_ReturnsNegInf` (handles the empty-mass case that actually fires when an entire run-length hypothesis decays).
3. **Renormalisation subtracts the max before `math.Exp`** (lines 224-245): the standard pattern. The `if math.IsInf(maxLogP, -1)` guard returns an explicit error rather than silently producing NaN.

These three are where every hand-rolled BOCPD implementation goes wrong; this one gets them right.

---

## 3. BOCPD log-space discipline — where it fails

### 3.1 `math.Log(1.0 - h)` is not `math.Log1p(-h)` — silent loss of signal at high lambda

**Location:** `bocpd.go:213`
```go
logGrowth := math.Log(b.p[r]) + logPi[r] + math.Log(1.0-h)
```

**The bug.** When `lambda` is large, `h = 1/lambda` is tiny and `1.0 - h` rounds to a value near 1, then `math.Log` of that loses bits to cancellation. For `lambda = 1e16`, `1.0 - 1e-16 = 1 - eps_mach/2` which **rounds back to 1.0**, so `math.Log(1.0 - h) = 0` exactly — the growth weight becomes 1 (no weighting from "no change-point happened"), and the algebraic structure that distinguishes growth from reset collapses entirely. The right fix is `math.Log1p(-h)`, which is exact for small `h`.

**Concrete numbers.** Probed with a five-element table (probe code in §4 of this report, summarised here as recovered output):
```
lambda=1e+06   log(1-h)=-1.000000500029e-06   log1p(-h)=-1.000000500000e-06   reldiff=2.9e-11
lambda=1e+08   log(1-h)=-1.000000010025e-08   log1p(-h)=-1.000000005000e-08   reldiff=5.0e-09
lambda=1e+15   log(1-h)= -9.992007222e-16     log1p(-h)= -1.000000000000e-15  reldiff=8.0e-04
lambda=1e+16   log(1-h)= -1.110223025e-16     log1p(-h)= -1.000000000000e-16  reldiff=0.110
lambda=1e+17   log(1-h)=  0                   log1p(-h)= -1.000000000000e-17  reldiff=1.000  (∞ relerr)
```

**Why this matters in the test suite.** `TestUpdate_StableUnderStationary` uses `Lambda = 1e5`, fine. `TestExpectedRunLength_GrowsUnderStationarity` uses `Lambda = 1e8` — the relative error at this `Lambda` is 5e-9, still tolerable. **`TestSmallLambda_BiasesTowardLowerRunLength_VsLargeLambda` uses `Lambda = 1e8` for the "large" comparator** — also tolerable. **But.** `TestRegimeStatistics_AfterStableRun` uses `Lambda = 1e6`, and the `infogeo` composition test in `infogeo_test.go:48` uses `Lambda = 100.0`. So the test surface today does not stress this. The hazard is for downstream consumers who reach for `Lambda = 1e15+` to mean "no expected change-points": at `Lambda = 1e16` they silently lose 11% of the growth weight and at `Lambda = 1e17` they lose all of it and the algorithm degenerates to "every observation is equally a reset and a growth."

**Fix is one line.** Change line 213 from `math.Log(1.0-h)` to `math.Log1p(-h)`. Same complexity, exact for small h, no perf impact. Same fix would make `TestNew_RejectsInfLambda` (which explicitly rejects `Lambda = +Inf`) honest — currently a user can pass `Lambda = 1e17` and silently get the same broken behaviour as `Lambda = +Inf` without the rejection.

### 3.2 `math.Log(b.p[r])` on a possibly-zero p[r]

**Location:** `bocpd.go:213, 220`

`math.Log(0) = -Inf` (Go's math package returns `-Inf` not NaN, confirmed by probe). The `logSumExp` calls downstream tolerate `-Inf` correctly (line 295-300 explicitly short-circuits). So this is not a correctness bug, it is a *performance and clarity* bug: the same quantity has just been computed as `math.Log(newP[i])` would be — the renormalisation step wrote `newP[i] = math.Exp(v - maxLogP)` and then divided by `sum`, throwing away the log. The next `Update` call starts by re-computing `math.Log(newP[i])`. A `b.logP []float64` field that retains the log-domain posterior between calls would (a) eliminate one `Log` and one `Exp` per run-length per step, (b) produce mathematically identical answers, and (c) avoid the `Log(0)` → `-Inf` round-trip entirely. The package doc-comment claims "Working in log-space avoids underflow" but in fact the implementation round-trips between log and linear domain at every step. This is the canonical efficiency improvement to BOCPD; Adams-MacKay's original Python ships the round-trip and is widely cited as the wrong way; pinned reference: Knoblauch & Damoulas 2018 §3.2.

### 3.3 Truncation at R_max silently biases toward reset

**Location:** `bocpd.go:207-215`
```go
for r := 0; r < n; r++ {
    dst := r + 1
    if dst >= newLen {
        continue   // ← growth contribution from the oldest hypothesis is silently dropped
    }
    ...
}
```

The reset loop (line 218-222) does **not** have a corresponding `continue`, so the reset contribution from `r = R_max` is preserved. Net effect: when the posterior saturates at `R_max + 1` slots, mass that "should" have flowed `R_max → R_max+1` is dropped, and renormalisation distributes it across the remaining slots **proportionally to their pre-renormalised weight**. Since the reset slot received its share already, the dropped mass effectively boosts every other slot — but the proportionate boost is largest for slots with non-trivial mass, which under stationarity is the high-r slots. So the truncation is *approximately* unbiased under stationarity, but introduces a small bias toward whatever slot dominates after renormalisation.

**Numerical magnitude.** `p[R_max] * (1 - 1/lambda)` per step. With `R_max = 500` and `lambda = 250` (defaults), and posterior mass on `r = R_max` of (typically) ~`exp(-500/250) ≈ 0.13` once saturated, the dropped growth mass per step is ~13% of the mass at the boundary, ~0.13 × P(r=500). After renormalisation this is small but not zero. Documenting it would be enough; "fixing" it requires either an explicit `r ≥ R_max` overflow bucket (canonical: pad to `R_max + 1` slots and merge the last two) or simply growing R_max — but the doc-comment should warn users that R_max should be set ≥ 5× the expected longest regime length to keep the saturation bias <1e-2.

`TestUpdate_TruncationHonoured` pins that `len(p) ≤ R_max + 1` but does not check the dropped-mass bias.

---

## 4. NaN / Inf input guards

Status (all references to `bocpd.go`):

| Function | NaN guard | +Inf guard | -Inf guard | Subnormal | ±0.0 | MaxFloat |
|---|---|---|---|---|---|---|
| `Update(x)` | yes (line 176) | yes (`IsInf(x, 0)` matches both signs) | yes (same call) | no test | no test | no test |
| `NigPrior.Validate` | yes for all four fields | yes (Mu0 IsInf check) | yes (same) | no test | no test | no test |
| `New(cfg)` | via Validate | yes | yes | no test | no test | no test |
| `studentTLogPDF` | **no input guard** | **no input guard** | **no input guard** | no test | no test | no test |

**The unprotected surface.** `studentTLogPDF` is package-private, so the only callers are inside `Update`. After NaN/Inf input is rejected by the `Update` guard, the only way to drive `studentTLogPDF` to a bad value is via degenerate sufficient statistics: `scale = 0` (probed: returns NaN), `df = 0` (probed: returns NaN via `Lgamma(0)`), `df < 0` (probed: returns NaN). With the validated prior these cannot arise from `New + Update`, but they *can* arise if a future maintainer calls `studentTLogPDF` directly from a new public method without re-validating. Worth adding a one-line internal guard (`if !(scale > 0) || !(df > 0) { return math.Inf(-1) }`) that produces `-Inf` (zero predictive probability) instead of NaN propagation.

`TestUpdate_RejectsNonFinite` (line 87-95) covers `NaN` and `+Inf` — but **not** `-Inf` and **not** subnormals. `Update(math.SmallestNonzeroFloat64)` would currently pass through the guard (it's finite and non-NaN), drive `(x - loc) / scale` to a denormal, and then produce a denormal-arithmetic-laden `studentTLogPDF`. Add at least: `Update(-math.Inf(1))`, `Update(math.SmallestNonzeroFloat64)`, `Update(math.MaxFloat64)`, `Update(0)`, `Update(-0.0)` — five new vectors, each one assertion long.

---

## 5. Calibration: detection-rate / false-alarm-rate trade-offs

**The topic prompt explicitly asks: "are detection-rate / false-alarm-rate trade-offs documented?"** The answer is **no**.

The doc-comment (`doc.go`) and per-function godoc names "ChangePointProbability" but warns it is *not* a useful alarm signal under constant hazard (the predictive likelihood cancels). It points to `ChangePointProbabilityWithin` and `MapRunLength` as the two canonical alarm surfaces. **Neither is benchmarked against a reference dataset.** No ROC curve test exists. No monte-carlo calibration of the `ChangePointProbabilityWithin(window) > threshold` decision rule against a known false-alarm rate exists. No "what threshold gives a 1-in-N-step false alarm rate?" documentation exists.

The two existing closest tests:
- `TestUpdate_DetectsStepShift`: deterministic seed, single 0→5 step shift, asserts `max post-shift P(r<5) > 0.5`. That is a single trial, not a calibration.
- `TestUpdate_StableUnderStationary`: deterministic seed, asserts `≤ 5/200` steps have `P(r<5) > 0.5`. That is **the only false-alarm-rate measurement in the package**, and it is for a single seed and a single threshold (0.5).

What the calibration should look like (canonical reference: Adams-MacKay 2007 §6, Knoblauch et al. 2018 §5):

1. **ROC curve test.** Generate N=1000 stationary trajectories of length T=500, count how often `max_t ChangePointProbabilityWithin(5)` exceeds threshold `tau` for `tau ∈ {0.1, 0.2, ..., 0.9}`. Plot or assert the resulting false-alarm rate is approximately `1/lambda` (the only theoretically-grounded null curve).
2. **Detection-rate test.** For each `tau`, generate trajectories with one true change-point and measure P(detection within W steps).
3. **Bias under truncation.** With R_max ∈ {50, 100, 500}, measure how the false-alarm rate shifts (this is where §3.3 above bites).
4. **Monte-Carlo p-value.** A `TestFalseAlarmRate_MatchesHazard` test that fits to the theoretical 1/lambda and rejects deviations >2σ over N=10000 trials. ~30 LOC. Currently absent.

The hand-rolled detection-vs-false-alarm tradeoff documentation that consumers (relic-insurance, triage, witness, watchtower, narrator per `doc.go`) will need is currently their problem, not the package's. This is the largest gap in the audit and the one most directly named in the topic prompt.

---

## 6. Online vs offline: float-error accumulation across many updates

The topic prompt: "does online accumulate float error indefinitely?"

**Inspection of the update.** Every `Update` call:
1. Recomputes the predictive log-PDF from the (mu, kappa, alpha, beta) hyper-parameters and the new x. No state is mutated *until after* the predictive is computed.
2. Renormalises the new posterior to sum to 1.0. This **resets** any accumulated float error in `p[r]` on every step.
3. Updates the sufficient statistics by **closed-form addition** (no running mean / Welford's). This is the float-error accumulation point: `mu' = (kappa*mu + x) / (kappa + 1)` is *not* numerically equivalent to a Welford update for very long runs.

**Empirical magnitude.** `TestRegimeStatistics_AfterStableRun` runs 300 updates and expects regime mean within 0.2 of true. The closed-form mu update at step n is algebraically equivalent to a one-pass mean of x_1..x_n with a prior contribution of weight kappa_0 — and the one-pass mean is well-known to lose `O(n * eps)` precision for n large vs Welford which loses `O(sqrt(n) * eps)`. For n = 1e6 this is the difference between `~1e-10` precision and `~1e-13` precision. Pistachio-cadence (60 FPS × hours) consumers reach `n = 1e6` in 5 hours of operation. **This is a real bug for long-running streams** but is not exercised by any test.

**Beta accumulation is worse.** `beta' = beta + 0.5 * kappa * dx² / (kappa + 1)` is a running sum-of-squared-deviations, and the textbook Welford update for variance lives precisely here. The current closed form will silently lose precision at O(n) rate. Same fix surface as mu.

The package should either (a) add a Welford-style update path for long-running consumers, (b) add a documented `Reset()` method that re-priors but keeps `t`, or (c) document the precision floor as `O(t * eps)` and warn at `t > 1e6`.

---

## 7. Allocation pattern in the hot path

`Update` allocates **six fresh slices per call** (lines 185, 201, 234, 256-259, 287). For `R_max = 500`, that's `6 * 501 * 8 = 24 kB` of garbage per `Update`. At 60 FPS over an hour: `60 * 3600 * 24 kB = 5.2 GB` of GC pressure for one BOCPD detector. CLAUDE.md rule 3 ("No allocations in hot paths. Functions accept output buffers. Pistachio calls these at 60 FPS.") is **violated** here. The fix is the standard scratch-buffer pattern that `signal/` and (per agent 010's audit) does not yet exist in `audio/`: hold scratch buffers as fields on `Bocpd`, recycle them per call, and return *views* (or accept caller buffers) instead of fresh slices. `RunLengthPosterior` already returns a defensive copy (lines 310-314), which is the right contract for the public API — but the *internal* `newLogP, newP, newMu, newKappa, newAlpha, newBeta` should all be reused.

The fix is bounded — six `b.scratch*` fields, `len()` truncation rather than `make()`, ~30 LOC. Backwards-compatible (return slices retain their copy contract).

No `bench_test.go` exists for changepoint. Per agents 010 and 015's pattern, this is the prerequisite for any perf claim. Adding `BenchmarkUpdate_R500_Stationary` and `BenchmarkUpdate_R500_StepShift` would pin the per-call allocation cost and make the §7 fix measurable.

---

## 8. Golden-file cross-language vectors

CLAUDE.md rule: "Every function has golden-file test vectors. Minimum 20 vectors per function, target 30." `testutil/` exists; `testdata/changepoint/` does **not**. The package ships zero JSON golden files. Other reality packages (calculus, audio per agents 016/006) all have at least sparse coverage; changepoint has none. This blocks the cross-language port axis (Python/C++/C# can't validate against Go) and means any future numerical refactor here has no portable regression surface beyond the in-Go tests.

The minimal initial coverage: 20+ vectors of `(prior, lambda, x_seq) → posterior_after_update[]` across (a) 5 stationary seeds, (b) 5 single-shift seeds, (c) 5 IEEE-754 edges (NaN-rejection, +Inf-rejection, subnormal x, tiny-log-PDF tail observation, exact-zero observation), (d) 5 hyper-parameter corner cases (Mu0=±MaxFloat, Kappa0=eps, Lambda=1, R_max=1). Each vector is ~50 lines of JSON.

---

## 9. Posterior-summary methods: silent edge cases

| Method | Edge case | Behaviour | Test? |
|---|---|---|---|
| `MapRunLength()` | empty posterior | returns 0 with `bestP=-1.0` (always passes the `p > bestP` check, even if posterior is all `-Inf` propagated as `NaN`) | no |
| `MapRunLength()` | tied probabilities | returns the smallest-r tie | no test, no doc |
| `ExpectedRunLength()` | NaN entry in posterior | propagates NaN | no test |
| `CurrentRegimeVariance()` | `alpha[r] = 1.0` exactly (the `α > 1` boundary) | **returns `b.beta[r]`** (line 396: "fall back to the rate") — but `Var = beta/(α-1) = ∞` is the mathematically correct answer, not `beta`; the current "fall back" gives `Var = beta` which is `≈ 0.5 * Var` for the prior. **This is a quiet wrong answer**, not a guard. | `TestInitial_CurrentRegimeVariance_AlphaGreaterOne` only tests `α=3 > 1`; the `α=1` boundary is uncovered. |
| `CurrentRegimeMean()` | MAP r out of range of mu slice | returns `b.prior.Mu0` (line 380) — a sane fallback but undocumented |

The `CurrentRegimeVariance` boundary fallback (line 393-397) is a real bug. With the default prior `α₀ = 1.0`, **at t=0** the variance is `b.beta[0] = 1.0`, not the mathematically-correct `+Inf` (improper prior). After even one update `α = 1.5 > 1` and the formula works again, so this only surfaces at t=0. But the t=0 value is what the consumer sees on the first frame. Either return `+Inf` (correct) or `math.NaN()` (signal the consumer that the value is undefined). The current "return beta" silently misleads.

---

## 10. What is missing entirely (for slot 022 cross-reference)

For a complete change-point package, the topic prompt's checklist is the right shopping list. Slot 022 (changepoint-missing) should cover all of:

- **CUSUM** (Page 1954): signed and two-sided, drift parameter, Lorden's optimality, ARL₀/ARL₁ calibration. ~80 LOC.
- **PELT** (Killick, Fearnhead, Eckley 2012): pruning correctness vs Brent's optimality bound. ~250 LOC.
- **Binary segmentation / WBS** (Fryzlewicz 2014): test statistic distribution under null, multi-scale randomisation. ~200 LOC.
- **Kernel CPD / MMD₂** (Gretton 2012, Arlot et al. 2019): biased and unbiased estimators, bandwidth selection (median heuristic). ~150 LOC.
- **E-divisive** (Matteson & James 2014): hierarchical agglomeration, p-value stability across n. ~180 LOC.

None of these belong in this audit; flagging them so slot 022 doesn't miss the list.

---

## 11. Fix-set summary, ranked by impact-per-LOC

| # | Fix | LOC | Impact |
|---|---|---|---|
| 1 | `math.Log(1.0-h)` → `math.Log1p(-h)` (line 213) | 1 | Closes the silent `Lambda ≥ 1e16` degeneracy; preserves test outputs to 1e-9 |
| 2 | Add ROC / monte-carlo false-alarm-rate test | ~80 | Closes the calibration gap (§5) — the largest topic-named gap |
| 3 | Document `R_max` saturation bias in doc-comment + warn `R_max ≥ 5×expected-regime-length` (§3.3) | ~10 | Honest contract for high-`R_max` consumers |
| 4 | Persist log-domain posterior (`b.logP` field) — eliminate per-step Log↔Exp round-trip (§3.2) | ~40 | Cleaner numerics + cheaper hot path |
| 5 | Scratch-buffer all six per-step slices (§7) | ~30 | Closes CLAUDE.md rule-3 violation |
| 6 | `CurrentRegimeVariance` α=1 fallback returns `+Inf` not `beta` (§9) | 2 | Removes a quiet wrong answer at t=0 |
| 7 | Internal `studentTLogPDF` guard against scale≤0 / df≤0 (§4) | 3 | Future-proofs against new public callers |
| 8 | Add IEEE-754 vectors for `Update`: -Inf, subnormal, ±0.0, MaxFloat (§4) | ~15 | Honours CLAUDE.md "IEEE 754 edge cases mandatory" |
| 9 | Add Welford variance update for long-stream consumers; document `O(t·eps)` floor on closed-form path (§6) | ~25 | Removes the silent precision degradation at t > 1e6 |
| 10 | `bench_test.go` for `Update` at R∈{50, 500} | ~20 | Pins #4 + #5 as measurable |
| 11 | First 20 golden-file JSON vectors (§8) | ~80 | Unblocks the cross-language port axis |

Total: ~306 LOC, every line citation-grounded, fully backwards-compatible, no API churn. The first three items are the ones that close the topic prompt's named concerns; the rest improve hygiene.

---

## 12. What is *not* a bug

Listed for completeness — the audit looked at these and confirmed they're fine:

- The `pi_r * H(r)` cancellation algebraically forcing `P(r=0) = 1/lambda` under constant hazard is *not* a numerics bug; the doc-comment on `ChangePointProbability` (lines 318-324) explicitly documents this and points users to `ChangePointProbabilityWithin` instead. Honest design choice.
- The `b.t++` increment after the slice updates rather than before is correct; the failure-preserves-state contract is pinned by `TestUpdate_NaN_StatePreserved`.
- The `r=0` always-prior reset on line 260-263 is correct (a fresh regime starts from the prior, not from any per-r posterior).
- `logSumExp(a, b)` symmetric and `-Inf` short-circuit is pinned by 7 cases including `TestLogSumExp_BothNegInf_ReturnsNegInf` and `TestLogSumExp_Symmetric`.
- `RunLengthPosterior` returning a defensive copy is the right public-API contract.
- `Bocpd is not safe for concurrent use` is documented (line 84) and is the right call — the per-step state mutation rules out cheap concurrency.

---

## 13. Cross-package consumer parity

The single existing cross-package test, `TestPosterior_FreshStartConvergence` in `infogeo_test.go`, witnesses that two BOCPD posteriors (full-stream vs post-CP-only) converge under TV and Hellinger metrics from `infogeo`. This is a clean R-CLOSED-FORM-PINNED-TO-CONSUMER pattern at 1/3 saturation (one consumer, no false-alarm-rate witness, no detection-rate witness). The natural next two pins:

- `R-CALIBRATION-PIN-1`: `infogeo.KLDivergence` between observed false-alarm distribution and theoretical 1/lambda Bernoulli.
- `R-CALIBRATION-PIN-2`: monte-carlo `prob.AvgRunLength0` vs theoretical `lambda`.

Both unblock the §5 calibration work using existing reality primitives. Worth flagging for slot 022 (missing) as the natural next consumer-pair.

---

## 14. Verdict

`changepoint/bocpd.go` is among the cleaner pieces of `reality/` — log-space discipline mostly correct, NaN guard at the entry point, deterministic, well-documented motivation, single cross-package consumer test. The numerical issues are concentrated in three places: the `Log(1-h)` instead of `Log1p(-h)` (one-line fix, §3.1), the absent calibration test surface (§5 — the largest gap and the one the topic prompt names directly), and the violation of CLAUDE.md rule 3 on per-step allocation (§7). The package is **not** ready to ship to a downstream consumer that needs a calibrated false-alarm rate; it **is** ready for a consumer that uses the posterior shape directly (per `TestPosterior_FreshStartConvergence`'s pattern). The 11-item fix-set is bounded and backwards-compatible.

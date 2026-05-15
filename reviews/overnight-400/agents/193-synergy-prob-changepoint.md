# 193 | synergy-prob-changepoint

**Topic:** prob × changepoint — BOCPD as Hidden Markov inference, conjugate
sufficient statistics, Bayes factors, distribution-free segmentation, ARL.
**Block:** B (cross-package synergies). **Date:** 2026-05-08. PARTIAL OVERLAP
with 172 (changepoint × timeseries focused on ARIMA/GARCH/DCC); this review
takes the **prob-side** view — BOCPD as forward HMM filter, suff-stat
propagation, sequential testing, ARL telemetry, conformal CPD.

## Two-line summary

`changepoint/bocpd.go` (402 LOC) **already implements an HMM forward filter**
with hidden state = run-length r_t and conjugate-prior emissions, but its
substrate friends in `prob/` (`MarkovSteadyState/Simulate` 155 LOC, seven
scalar distributions, three parametric tests, two non-parametric tests,
`prob/conformal/` 5 files ~600 LOC, `BayesianUpdate` / `LogOddsPool` /
`IsotonicRegression`) are **not algebraically connected**: BOCPD ships
exactly one prior (Normal-Inverse-Gamma → Student-t predictive), one hazard
(constant H = 1/λ), no Viterbi, no Bayes factor, no SPRT/CUSUM, no KS/AD
distribution-free fallback, no conformal coverage, no ARL/EDD telemetry —
verified by repo-wide grep on `Viterbi|forward.?backward|Baum.?Welch|
BayesFactor|CUSUM|SPRT|Page|Lorden|Pollak|KSStatistic|AndersonDarling|
ConjugatePrior|BetaBernoulli|GammaPoisson|NIW|MAPSegmentation|MultivariateBOCPD|
ARL|EDD`: zero matches outside test names. Twenty synergy primitives P1–P20
totalling **~2,140 production LOC** of pure connective tissue close the gap
with **zero new packages** (all land in `changepoint/` plus two new files
under `prob/`). Cheapest one-day PR is **P1 BOCPDViterbiSegmentation +
P2 BayesFactorChange + P14 ConjugateBetaBernoulliBOCPD (~280 LOC)** —
Viterbi adds one back-pointer slice on the lattice already computed,
Bayes factor is the BOCPD evidence ratio (numerator already computed and
discarded as renormalisation constant), Beta-Bernoulli is NIG-BOCPD with
three-line scalar arithmetic substituted; highest-leverage architectural lift
is **P10 BOCPDPosteriorPredictive + P17 MultivariateBOCPD-NIW (~400 LOC)**
because mixture-of-Student-t forecast is the surface Pulse/Oracle/RubberDuck
need and NIW lifts BOCPD to vector observations covering cross-covariance
breaks; **crown jewel is P19 ConformalChangepoint (~180 LOC)** — wraps any
nonconformity-scored detector with `prob/conformal/AdaptiveQuantile` to
deliver finite-sample Type-I-error control without exchangeability,
something neither Adams-MacKay-2007 nor Page-1954 ship and which closes the
"can I cite a coverage guarantee on this alarm?" question every regulator-
defensible consumer asks.

---

## Bases (verified)

**`changepoint/bocpd.go` 402 LOC:** `NigPrior` (Mu0, Kappa0, Alpha0, Beta0)
+ `Validate` + `DefaultNigPrior`; `Bocpd{prior, rMax, lambda, t, p, mu,
kappa, alpha, beta}`; `Update(x)` does Student-t predictive at df=2α,
scale²=β(κ+1)/(ακ); log-space growth/reset; renormalise; NIG suff-stat
update. Methods: `RunLengthPosterior`, `ChangePointProbability` (= P[r_t=0],
**author flags this as algebraically constant under constant hazard**, points
to within-window), `ChangePointProbabilityWithin(window)`, `MapRunLength`,
`ExpectedRunLength`, `CurrentRegimeMean`, `CurrentRegimeVariance` (β/(α-1)
when α>1), `Step`. Internals: `hazard(r)=1/λ` constant only, `studentTLogPDF`,
`logSumExp`. Bit-stable. **Allocates 6 fresh length-(n+1) slices per Update**
— no buffer-reuse overload. **Doc.go enumerates 24 candidate consumers; zero
import the package.**

**`prob/distributions.go` 520 LOC:** Normal/Exponential/Uniform/Beta/Poisson/
Gamma/Binomial PDF+CDF (+Quantile for Normal/Exp). **Missing for synergy:**
Inverse-Gamma, Inverse-Wishart, Normal-Inverse-Gamma, Normal-Inverse-Wishart,
Multivariate-Gaussian, Student-t (lives privately inside `changepoint/`),
Dirichlet, Negative-Binomial.

**`prob/distribution.go` 170 LOC:** OO `BetaDist/NormalDist/ExponentialDist/
UniformDist` + `KLDivergenceNumerical`. **No `ConjugatePrior` interface** —
the suff-stat update BOCPD performs inline is not abstracted, so each new
exponential family must copy-paste arithmetic.

**`prob/markov.go` 140 LOC:** `MarkovSteadyState` (power iter), `MarkovSimulate`
(LCG). **Discrete only.** No HMM, no forward-backward, no Viterbi, no
Baum-Welch, no emission abstraction. The substrate-of-substrate for
BOCPD-as-HMM exists at the steady-state level only.

**`prob/hypothesis.go` 210 LOC:** TTestOneSample, TTestTwoSample (Welch),
ChiSquaredTest. **Missing:** F-test, Wald, LRT, SPRT (Wald 1945), CUSUM
(Page 1954), Shewhart, EWMA, GLR (Lorden 1971).

**`prob/nonparametric.go` 190 LOC:** FisherExactTest, MannWhitneyU.
**Missing:** Kolmogorov-Smirnov (one and two-sample), Anderson-Darling,
Cramér-von Mises, Kuiper. **All four are CPD substrate.**

**`prob/prob.go` 480 LOC:** `BayesianUpdate(prior, LR)`, `BayesianUpdateChain`
(single-state Bayes; no run-length lattice), `LogOddsPool`, `BrierScore`,
`LogLoss`, `WilsonConfidenceInterval`, `ExpectedCalibrationError`,
`MaximumCalibrationError`, `ReliabilityDiagram`, `IsotonicRegression`.

**`prob/jeffreys.go` 180 LOC:** `JeffreysConfidence(succ,fail)` (Beta(½,½)
posterior), `JeffreysKLDivergence`. The Beta-Bernoulli posterior arithmetic
P14 needs lives here in scalar form.

**`prob/conformal/` 5 files ~600 LOC:** `SplitQuantile`, `SplitInterval`,
`CqrInterval`, `MarginalCoverageBounds`, `AdaptiveQuantile` (recency-weighted
half-life decay), `AdaptiveInterval`, `EffectiveSampleSize`,
`MondrianQuantile`, `MondrianInterval`, `AbsResidual`/`NormalizedResidual`/
`LogResidual`/`ScoreAll`. **The exchangeability-violation toolkit BOCPD needs
sits idle.**

### Critical gaps inventory

| BOCPD variant | Substrate need | Lives at | Currently |
|---|---|---|---|
| Viterbi MAP segmentation | log-prob lattice + back-pointers | `changepoint/viterbi.go` | absent |
| Bayes factor vs no-CP | log-evidence ratio | `changepoint/bayes_factor.go` | absent |
| Beta-Bernoulli prior | Bernoulli LR + Beta suff-stats | `changepoint/conjugate_bb.go` | absent |
| Gamma-Poisson prior | Poisson-NB predictive | `changepoint/conjugate_gp.go` | absent |
| NIW (multivariate) | Wishart + matrix algebra | `changepoint/multivariate.go` | absent |
| AR-coefficient change | NIW on regression posterior | `changepoint/ar_change.go` | absent |
| Variance-only change | scaled-inv-χ² conjugate | `changepoint/variance_only.go` | absent |
| Posterior-predictive | mixture over r_t lattice | `changepoint/predictive.go` | absent |
| R-pruning | Adams-MacKay 2007 §3.2 | inline in `bocpd.go` | absent (truncation only) |
| Time-varying hazard | hazard-func interface | refactor `(b).hazard(r)` | constant only |
| CUSUM / SPRT | sequential LR | `prob/sequential.go` | absent |
| KS / AD / CvM | empirical CDF distance | `prob/empirical.go` | absent |
| ARL (Page-Lorden) | numerical IE solver | `changepoint/arl.go` | absent |
| Conformal CPD | wrap score with conformal q | `changepoint/conformal.go` | absent |

---

## Synergy primitives (P1–P20)

Each: capability • composition • tissue LOC (production + test).

**P1 BOCPDViterbiSegmentation — MAP run-length path.** Offline-mode hard
segmentation δ_1<δ_2<…; standard HMM Viterbi log-product on the same
growth/reset transition kernel BOCPD already evaluates. Composition:
`Update` already computes log-prob lattice; add back-pointer table
ψ_t[r] = argmax over predecessor; walk back from argmax_r δ_T(r). **No new
math.** Tissue: 110 src + 40 test.

**P2 BayesFactorChange — evidence ratio CP-vs-no-CP.** For window
x_{t-w:t}, BF = p(x | M_changepoint) / p(x | M_stationary). Resolves the
"P[r_t=0] is constant under constant hazard" footgun (`bocpd.go:319`) by
replacing it with an algebraically meaningful score; alarm at Kass-Raftery
1995 thresholds (10/100). Composition: numerator = Σ_r P(r,x_{1:t}) — the
BOCPD renormalisation constant **already computed and discarded**;
denominator = NIG predictive at hazard=0. Tissue: 80 src + 30 test.

**P3 BOCPDForwardFilter (refactor) — explicit HMM-canonical API.** Expose
`EmissionLogPDF(r,x)`, `TransitionLogProb(r, r_next)`, `ForwardStep(prevP,
x)` so prob-thinkers compose. Pure refactor + new public surface; zero
behaviour change (golden-file regression confirms). Tissue: 50 src + 20 test.

**P4 BOCPDTimeVaryingHazard — pluggable H(r,t).** Replaces constant-hazard
limitation that makes `ChangePointProbability` algebraically useless.
Power-law H(r) = 1/(λ+αr) (AM 2007 §3.1), Weibull-survival H(r,t), seasonal
H(t) ∝ 1+cos(2π t/period). Under non-constant H, P[r_t=0] **becomes** a
useful alarm because predictive cancellation no longer occurs. Composition:
`Hazard func(r int, t int) float64`; `DefaultHazard = ConstantHazard(λ)`
preserves bit-stability. Tissue: 60 src + 50 test.

**P5 BOCPDRpruning — Adams-MacKay 2007 §3.2.** Beyond hard truncation at
R_max, soft-prune mass < ε (default 1e-4); shrinks memory from O(t) to
O(log t / log(1/(1-h))) ≈ 50 for h=0.01 / ε=1e-4. **Order-of-magnitude
per-step compute reduction** for steady state. Composition: post-renormalise
sweep `newP[r]<eps` → drop with mass-redistribution to neighbours; maintain
`keptIndex []int` so suff-stat arrays compact; need stable hypothesis ID to
thread into P1 if both ship. Tissue: 100 src + 60 test.

**P6 BOCPDPosteriorPredictiveSampling.** Mixture-of-Student-t forecast
f̂(x_{t+1}|x_{1:t}) = Σ_r P[r_t=r] · t_{2α_r}(μ_r, scale_r). Lifts BOCPD
from "tells you when a regime broke" to "tells you what x_{t+1} looks like
**conditional on run-length uncertainty**" — the surface Pulse/Oracle/
RubberDuck need (per `prob/timeseries.go` docstrings). Composition: weight
each NIG predictive by `b.p[r]`; expose `PredictivePDF/CDF/Quantile/Sample`.
Tissue: 120 src + 40 test.

**P7 BOCPDExpectedRegimeStats.** Posterior-weighted-not-MAP: E[μ|x_{1:t}] =
Σ_r p[r]·μ_r, E[σ²|x_{1:t}] = Σ_r p[r]·β_r/(α_r-1), plus posterior
variances. Current `CurrentRegimeMean/Variance` use MAP only and miss
mass on neighbouring run-lengths. Composition: reductions over (b.p, b.mu,
b.kappa, b.alpha, b.beta). Tissue: 50 src + 25 test.

**P8 KSStatistic + KSTwoSample + AndersonDarlingTwoSample —
distribution-free CPD substrate.** Massey 1951 KS D_n = sup_x|F̂_n(x) -
F(x)|; AD = n·∫(F̂_n-F)²/[F(1-F)] dF (tail-emphasising); both with
finite-n null distributions for sliding-window CPD scoring. **The
non-parametric primitives that turn any sliding-window CPD framework into
a distribution-free one** (Pettitt 1979, Csörgő-Horváth 1997). Lives in new
`prob/empirical.go`. Tissue: 140 src + 80 test.

**P9 SlidingWindowKSChangepoint — distribution-free alternative to BOCPD.**
Maintain W_left = x_{t-2w:t-w} and W_right = x_{t-w+1:t}; score =
KS-two-sample(W_left, W_right); alarm at Smirnov-1948 critical value.
Complements BOCPD: ships when user can't commit to NIG, or when distributions
are heavy-tailed where Student-t breaks. Composition: P8 + ring buffer +
critical-value lookup. Tissue: 110 src + 50 test.

**P10 BOCPDPosteriorPredictiveAsConformal.** Plug BOCPD's predictive density
**into the conformal pipeline**: −log f̂(x_{t+1}) as nonconformity score →
`AdaptiveQuantile(scores[:t], α, halfLife=λ)`. Yields **a conformal-grade
prediction interval that automatically widens at regime breaks** (because
the predictive density spikes negative there) — the deepest BOCPD ×
conformal cross-pollination. Composition: P6 `PredictivePDF` +
`prob/conformal/AdaptiveQuantile` + new `NegLogPredictiveScore` in
`prob/conformal/nonconformity.go`. Tissue: 90 src + 60 test.

**P11 CUSUM (Page 1954) — sequential CPD via SPRT.** S_t = max(0,
S_{t-1} + log f₁(x_t)/f₀(x_t)); alarm at first t with S_t > h. The
2-Lyapunov-time-baseline detector every flagship hand-rolled (per 022-T1.x).
With P4 hazard interface in place, CUSUM is BOCPD's H(r) = δ-at-known-CP
limit reformulated. `prob/distributions.go` LRs for Normal-mean /
Normal-variance / Bernoulli / Poisson / Exponential out of the box. Lives
in `prob/sequential.go`. Tissue: 80 src + 40 test.

**P12 SPRT (Wald 1945).** Sequential test with **uniformly minimum sample
size** at fixed (α, β). Companion to P11; CUSUM is unbounded-restart limit
of SPRT. Composition: log-LR accumulator + Wald boundaries
log(β/(1-α)) / log((1-β)/α). Tissue: 50 src + 30 test.

**P13 AverageRunLength — Page-Lorden ARL₀ / ARL₁ telemetry.** For any
sequential detector parametrised by threshold h: ARL₀(h) = E[T | no change]
(false-alarm interval), ARL₁(h, μ_post) = E[T - τ | change at τ] (detection
delay). Page 1954, Lorden 1971 lower bound, Pollak 1985 upper bound. The
substrate to **answer "what threshold gives me 1 false alarm per year"** in
calibrated, citable, golden-pinned form. Composition: Brook-Evans 1972
finite-state Markov-chain approx (`prob/markov.go` already ships
power-iteration substrate). Tissue: 140 src + 60 test.

**P14 ConjugateBetaBernoulliBOCPD — binary-stream changepoint.** BOCPD on
binary observations (success/failure rate breaks); same algorithm with
Beta-Bernoulli replacing NIG. Substrate for conversion-rate monitoring,
pass/fail telemetry, A/B-test stationarity. Beta posterior arithmetic
already in `prob/jeffreys.go` for the scalar case. Composition: state
arrays α_r, β_r; predictive Beta-Binomial PMF =
B(α_r+x, β_r+1-x)/B(α_r, β_r); update α'=α+x, β'=β+1-x; same hazard
structure. Tissue: 110 src + 50 test.

**P15 ConjugateGammaPoissonBOCPD — count-stream changepoint.** Poisson rate
breaks under Gamma conjugate (Negative-Binomial predictive). Substrate for
queue-length monitoring (composes with `queue/`), arrival-rate breaks,
error-count regimes. `prob/distributions.go` already ships Poisson/Gamma.
Composition: state α_r, β_r; predictive NB(α_r, β_r/(β_r+1)); update
α'=α+x, β'=β+1. Tissue: 100 src + 50 test.

**P16 ConjugateScaledInvChi2BOCPD — variance-only changepoint.** Mean held
constant; only σ² breaks. AM 2007 §3.3 special case. Useful when mean is
locked (zero-centred returns) — **strictly more powerful** than NIG-BOCPD at
variance-only changepoints (no prior mass wasted on mean drift). Tissue:
80 src + 30 test.

**P17 MultivariateBOCPD-NIW.** Vector observations under Normal-Inverse-
Wishart conjugate; predictive multivariate-Student-t. The **single biggest
BOCPD lift** — covers "did the cross-correlation structure break?" question
that DCC-GARCH (172) approaches differently and that scalar-BOCPD-on-each-
channel **provably misses** (rotation in covariance has zero effect on
marginal variance). Composition: state (μ_r ∈ R^k, κ_r, ν_r, Ψ_r ∈ R^{k×k});
predictive t_{ν_r-k+1}(μ_r, Σ_r·(κ_r+1)/(κ_r·(ν_r-k+1))). Needs
`linalg.Cholesky` (shipped), `MatrixInverse` (shipped), `LogDeterminant`
(shipped), `LogGamma` (shipped). NIW arithmetic per Murphy 2007 §9.6.
Tissue: 280 src + 100 test.

**P18 ARCoefficientChangeBOCPD.** x_t = φ_1 x_{t-1} + … + φ_p x_{t-p} + ε_t
with **φ ∈ R^p** breaking (AM 2007 §3.4 + Hamilton 1989 link). Subsumes
regression-coefficient change (p=0, exogenous regressors instead of lags).
Composition: design row [x_{t-1},…,x_{t-p}]; posterior on φ Normal under
Gaussian-NIG with conjugate Wishart on noise — exactly P17's NIW machinery
applied to the regression posterior, not the observation directly. Tissue:
180 src + 70 test.

**P19 ConformalChangepoint — finite-sample-valid CPD wrapper.** **Substrate-
deepest synergy in this review.** Wraps any nonconformity-scored detector
(BOCPD predictive density, CUSUM, KS, MMD) with `AdaptiveQuantile` to deliver
**finite-sample Type-I-error control without exchangeability** — valid even
when the underlying distribution is itself drifting, **the situation CPD is
designed for**. Closes the regulator-defensibility gap every flagship asks
BOCPD to fill. Mirrors Vovk-Petej-Nouretdinov 2003 conformal-on-time-series
lifted to changepoint via Tibshirani-Foygel-Barber-Candès 2019 weighted
exchangeability. Composition: P10 `PredictivePDF` (or any score) →
`AdaptiveQuantile(scores[:t], α, halfLife≈λ)` → threshold → alarm. Tissue:
180 src + 100 test.

**P20 BOCPDIsotonicCalibration.** Raw `ChangePointProbabilityWithin(window)`
is **not Brier-calibrated**; pass through `IsotonicRegression` against
historical change-event labels for calibrated alarm probability. Composes
with `ExpectedCalibrationError` for telemetry. Closes the "alarm = 0.7 —
what does that mean?" question. Tissue: 70 src + 40 test.

---

## LOC roll-up

| Tier | Primitives | Production | Test | Total |
|---|---|---|---|---|
| Tier-S (1d each) | P1, P2, P3, P4, P14 | 410 | 195 | 605 |
| Tier-M (2d each) | P5, P6, P7, P9, P10, P15, P16, P20 | 720 | 365 | 1085 |
| Tier-L (≥3d each) | P8, P11, P12, P13, P17, P18, P19 | 1010 | 460 | 1470 |
| **Total** | **20** | **2140** | **1020** | **3160** |

Test scaffolding adds ~50% via golden-file infrastructure already paid for
by `testutil/`.

---

## Sprint ordering

1. **Day 1:** P1+P2+P14 (Viterbi + Bayes-factor + Beta-Bernoulli) — 280 src,
   four new public methods, BOCPD becomes "framework not detector".
2. **Day 2:** P3+P4 (HMM-canonical refactor + time-varying hazard) — 110
   src; **all downstream primitives thread through the same interface from
   here on.**
3. **Day 3:** P8 (KS/AD/two-sample) — 140 src; generic `prob/empirical.go`
   with seven downstream consumers across CPD/conformal/hypothesis-test/
   goodness-of-fit.
4. **Day 4:** P10+P19 (conformal-grade BOCPD) — regulator-defensibility
   crown jewel; 270 src.
5. **Day 5:** P11+P12+P13 (CUSUM+SPRT+ARL) — 270 src; entire sequential-
   detector roster `prob/` is missing.
6. **Day 6+ (architectural):** P17 NIW (280 src), then P18 AR-coefficient
   (180 src). Both consume `linalg.Cholesky`/`LogDeterminant` already in
   tree; no new package dependencies.

---

## Cross-language pinning targets

- P1 Viterbi: `pomegranate` HMM .viterbi() to 1e-9.
- P2 Bayes factor: Kass-Raftery 1995 fig-1 (within 5%).
- P5 R-pruning: Adams-MacKay 2007 fig-2 (seed-pinned simulator).
- P6 mixture predictive: closed-form Student-t collapse (single-r limit) 1e-12.
- P8 KS: scipy.stats.ks_2samp 1e-9; Massey 1951 critical values 1e-6.
- P11 CUSUM: Page 1954 fig-1.
- P13 ARL: Lorden 1971 lower bound + Brook-Evans 1972 to 1e-6.
- P14 Beta-Bernoulli BOCPD: PyMC.changepoint within 5%.
- P15 Gamma-Poisson BOCPD: `bayesianchangepoint` R package golden file.
- P17 NIW BOCPD: `bocpdms` (Knoblauch-Damoulas 2018) reference run.
- P19 Conformal CPD: Vovk-Petej-Nouretdinov 2003 finite-sample coverage at
  α=0.1 within 2%.

---

## Out-of-scope deferrals

- **Online Baum-Welch / EM HMMs** — separate package `prob/hmm/`; defer v2.
- **MCMC change-point sampling** (Green 1995 reversible-jump) — needs
  RNG that conflicts with bit-stable determinism; defer.
- **Reservoir-computing CPD** (Pathak-Lu-Ott 2018) — needs
  `chaos/reservoir.go` (191-S11) shipped first; cross-link to 191.
- **Topological CPD via persistence** — needs `topology/persistent/`
  Wasserstein-p (190-T10) shipped first; cross-link to 190.
- **Hidden-semi-Markov models** with explicit duration distributions —
  generalisation of P4; principled extension once P4 ships.

---

## Cycle-free DAG impact

New imports: `changepoint/` → `prob/` (currently zero edges; P14, P15 via
`prob/distributions.go`); `changepoint/` → `prob/conformal/` (P10, P19);
`changepoint/` → `linalg/` (P17, P18; **already present**).
`prob/sequential.go` → `prob/distributions.go` (intra-package).
`prob/empirical.go` → none (pure stdlib sort). **No reverse edges; no new
packages.** All twenty primitives compose against v0.10.0's shipped surface
plus four new files in `changepoint/` and two new files in `prob/`.

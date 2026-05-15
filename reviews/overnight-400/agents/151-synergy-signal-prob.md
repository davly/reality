# 151 | synergy-signal-prob

**Topic:** signal × prob — spectral estimation as Bayesian inference, periodogram statistics
**Block:** B (cross-package synergies, first agent)
**Date:** 2026-05-08
**Scope:** capabilities that emerge ONLY when `signal/` and `prob/` are
composed; not what either is missing in isolation (covered by 131-135 / 116-120).

## Two-line summary

Today `signal/` returns periodogram amplitudes and `prob/` owns chi-squared/
gamma CDFs but they don't talk: no PSD with a CI, no Whittle likelihood, no
Lomb-Scargle, no Neyman-Pearson, no CFAR. **Eleven synergy primitives (S1-S11)
totalling ~1130 LOC of glue** stand up Bayesian spectral estimation, detection
theory, and irregular-time inference on top of two stable bases; cheapest
first PR is **S1 PowerSpectrumCI** at ~80 LOC because χ²-CIs on bins are the
sufficient statistic connecting both packages, and the highest-leverage
one-day unlock is **S8 ROCCurve via existing `prob.MannWhitneyU`**
(Hand-Till 2001 AUC = U / (n_pos·n_neg) — nobody noticed prob already did
the heavy lifting).

---

## Bases — what each package exposes today

`signal/` (~470 LOC, agent 131): `FFT, IFFT, PowerSpectrum, FFTFrequencies,
Convolve, MovingAverage, ExponentialMovingAverage, MedianFilter, Hann/Hamming/
BlackmanWindow, ApplyWindow`. `PowerSpectrum` returns a **point estimate** —
no variance, no CI, no noise model.

`prob/` (~14 files, agent 117): exposes `Distribution` interface (Beta,
Normal, Exponential, Uniform), `GammaPDF/CDF`, `BayesianUpdate(Chain)`,
`LinearRegression`, `BenjaminiHochberg`, `MannWhitneyU`, `chiSquaredCDF`
(**private**, `mathutil.go:187`, used only by `ChiSquaredTest`).

Neither imports the other. There is no `spectral/` glue file today.

## The conceptual unlock — periodogram as χ² likelihood

For zero-mean stationary Gaussian noise of variance σ², the periodogram bins
are asymptotically independent (Whittle 1953) and

    2·I[k] / S(ω_k)  ~  χ²_2     (k = 1, …, N/2−1)
    2·I[k] / S(ω_k)  ~  χ²_1     (k = 0, Nyquist)

This single fact connects every synergy below. `reality` ships everything to
compute the left side (signal/) and the right side (prob/, via private
`chiSquaredCDF`), but not the connection. The synergy is high-leverage
precisely because both bases are clean — the cost is connective tissue,
not new mathematics.

---

## S1 — `PowerSpectrumCI`: confidence intervals per bin

Every consumer of `PowerSpectrum` today has magnitudes with no error bars.

```go
func PowerSpectrumCI(real, imag []float64, alpha float64,
    psd, low, high []float64)
```

Composes `PowerSpectrum` → publicised `prob.ChiSquaredQuantile`. **~80 LOC**
plus ~20 LOC to expose `ChiSquaredQuantile`. At α=0.05, df=2 the CI is roughly
[I/3.69, I/0.103] — a 36× ratio, which is the textbook reason single-segment
periodograms are statistically useless and Welch averaging exists. Once shipped,
consumers see *why* averaging is needed without us having to ship Welch first.
**Saturation witness:** simulate N(0, σ²), compute CI, count bins covering σ²
→ 1−α as N→∞.

## S2 — `WhittleLogLikelihood`: spectral MLE

    log L(θ) ≈ −Σ_k [ log S_θ(ω_k) + I[k] / S_θ(ω_k) ]

The frequency-domain stand-in for the time-domain Gaussian likelihood, O(N)
after one FFT instead of O(N³) for the full covariance matrix (Robinson 1995,
Beran 1994). Foundation of frequency-domain estimation in geosciences, neural
data, economics.

```go
func WhittleLogLikelihood(periodogram []float64, sampleRate float64,
    psdModel func(omega float64) float64) float64
```

**~60 LOC**. Wraps trivially into `optim.LBFGS` for spectral-MLE — no new
optim primitive. PSD models routed through `autodiff.Tape` get analytic
gradients for free, dropping spectral-MLE from finite-differenced 4-eval/iter
to 1-eval/iter (cf agent 140 GARCH).

## S3 — `ARSpectrum` from autocovariance

Parametric PSD that resolves spectral peaks finer than the rectangular-window
1/N bin width.

    S_AR(ω) = σ² / |1 − Σ_{k=1}^{p} a_k e^{-iωk}|²

`prob.timeseries.levinsonDurbin` *already* exists at `prob/timeseries.go:233`
but is unexported (agent 136 §F.2 also flagged this). Expose it as
`prob.LevinsonDurbin(autocov, p) (coeffs, sigma2)` (+30 LOC) and
`signal.ARSpectrum(coeffs, sigma2, omegas, psd)` (+50 LOC). **~80 LOC total.**
**Witness:** AR(2) poles at z=0.95 e^{±iπ/4} → peak at π/4, FWHM ≈
−2 log(0.95) ≈ 0.10 rad, golden-testable to 1e-10.

## S4 — `MultitaperPSD` (Slepian / DPSS)

Welch averages overlapping windows but loses low-frequency resolution;
multitaper averages K orthogonal Slepian tapers and gives **maximum spectral
concentration for a given resolution-bandwidth product** (Thomson 1982). Each
taper's periodogram is independent → sum is χ²_{2K} → variance shrinks 1/K
with no bias growth. The `eqDoF` return is the bridge into S1's CI machinery.

```go
func MultitaperPSD(signal []float64, nw float64, k int,
    psd []float64) (eqDoF float64)
```

**~120 LOC** + DPSS dependency on `linalg/` tridiagonal eigensolve (~150 LOC,
in agent 097/098 missing-list). If DPSS deferred, ship sine-tapers
(Riedel-Sidorenko 1995, ~30 LOC, 80% of multitaper benefit).

## S5 — `BayesianPSD` (inverse-gamma prior)

Conjugate prior for σ²-spectrum is inverse-gamma; under χ²_2 likelihood the
posterior is also inverse-gamma with α_post = α_prior + 1, β_post = β_prior +
I[k]. Posterior credible intervals shrink properly at low SNR and admit
informative priors when the user knows the noise floor (e.g. from calibration).

```go
func BayesianPSD(periodogram []float64, alphaPrior, betaPrior, credLevel float64,
    mean, low, high []float64)
```

**~100 LOC** + new `prob.InverseGammaQuantile` (~40 LOC, also useful for
`prob/conformal/`).

## S6 — `LombScargle` for irregular sampling

Astronomy (variable stars), ecology (animal counts), finance (tick data) all
violate the FFT regular-grid assumption. Lomb-Scargle (Lomb 1976, Scargle
1982) is the maximum-likelihood frequency content under sinusoidal regression
with χ²_2 distribution preserved.

```go
func LombScargle(times, signal, freqs, psd []float64)
```

**~150 LOC**. Generalises `prob.LinearRegression` (already exists,
`prob/regression.go:36`) to a sin/cos basis with the Scargle τ time-shift.
Bottleneck is `prob.FCDF` for false-alarm probability — ~60 LOC, in agent
117 §T1 list. χ²_2 distribution carries over so S1 CI applies directly.

## S7 — `NPDetect` (Neyman-Pearson energy detector)

Optimal binary hypothesis test for signal in Gaussian noise. The likelihood-
ratio test that separates H_0 (noise) from H_1 (signal+noise) is gamma-shaped
under both hypotheses; the threshold for target P_FA = α is the upper
α-quantile of the H_0 distribution.

```go
func NPDetect(periodogram []float64, pFA, noiseSigma float64,
    bins []bool) (threshold float64)
```

**~70 LOC**. Composes `signal.PowerSpectrum` + `prob.GammaCDF` / quantile.
Adds zero new math; sweeping pFA traces a per-bin ROC curve directly (S8).

## S8 — `ROCCurve` via Mann-Whitney AUC

**Sleeper highest-leverage finding.** Hand & Till 2001 prove
**AUC = U / (n_pos · n_neg)** where U is the Mann-Whitney U-statistic.
`prob/nonparametric.go` already ships `MannWhitneyU` — a pure-prob formula
half-shipped that nobody connected to detection theory.

```go
func ROCCurve(scores []float64, labels []bool,
    pFA, pD []float64) (auc float64)
```

**~80 LOC** total: ~50 for the threshold sweep, one call into the existing
`MannWhitneyU` for AUC. *This is the single most leverage-per-LOC PR in
this whole list.*

## S9 — CFAR family (CA-, OS-, GO-, SO-CFAR)

Detection threshold that adapts to local noise without prior knowledge of σ²
(Richards 2014, Skolnik 2008). Estimate noise floor in a guard-bracketed
neighbourhood, scale by CFAR multiplier T, declare detection if cell > T·noise.

```go
func CACFAR(cells []float64, numTrain, numGuard int, pFA float64,
    detections []bool) (multiplier float64)
func OSCFAR(cells []float64, numTrain, numGuard, orderK int, pFA float64,
    detections []bool) (multiplier float64)
```

**~180 LOC** (4 variants × ~45 LOC; reuse same training-cell extractor).
CA-CFAR's multiplier `T = N · (P_FA^{−1/N} − 1)` is the F-distribution
quantile in disguise; OS-CFAR's involves the k-th order statistic of N gamma
variates. Without `prob.FQuantile` and `prob.GammaOrderStatistic` (~80 LOC
new) the multiplier is hard-coded magic. OS-CFAR reuses
`signal.MedianFilter` (already shipped!) as the order-statistic primitive.

## S10 — `KalmanFilter` (time-domain twin of S2)

Closes the loop between `signal/` (frequency-domain) and `prob/`
(likelihood). Kalman is **linear-Gaussian state-space inference** and
computes the exact likelihood of a Gaussian process in O(N·d³) — time-domain
counterpart of S2's Whittle approximation. Every adaptive filter (Wiener,
RLS, LMS) is a Kalman special case.

**~400 LOC** (Kalman + RTS smoother + log-likelihood). Outside this report's
primary scope (agent 137 §T1.1 owns it from the timeseries side) but called
out because **S2 + S10 together are spectral and time-domain twins of the
same likelihood** — shipping one without the other forces consumers to
re-derive the bridge.

## S11 — `Coherence` and cross-spectrum

Magnitude-squared coherence γ²_{XY}(ω) = |S_{XY}|² / (S_XX · S_YY) is the
frequency-domain correlation coefficient and lives in [0, 1] like a
probability. Its sampling distribution under γ² = 0 is β(1, K−1) for
K-segment averaging (Carter 1987) — a pure beta-distribution result that
connects directly to `prob.BetaCDF`.

```go
func Coherence(x, y []float64, segLen, segOverlap int, pFA float64,
    coh []float64) (sigThreshold float64)
```

**~120 LOC**. The single primitive on this list that turns signal × prob
into a **multivariate** capability — coupled-oscillator identification, EEG
functional connectivity, cross-asset spectral hedging.

---

## Composition table — what gates what

| Synergy | LOC | Existing deps | New (signal/prob) | Downstream |
|---|---|---|---|---|
| S1 PowerSpectrumCI | 80 | signal.PowerSpectrum, chiSquaredCDF | publicise ChiSquaredQuantile (+20) | S2,S4,S7-9,S11 |
| S2 WhittleLogLikelihood | 60 | signal.PowerSpectrum | — | optim.LBFGS, autodiff |
| S3 ARSpectrum | 80 | timeseries.levinsonDurbin | publicise (+30) | S2, S10 |
| S4 MultitaperPSD | 120 | windows, S1 | linalg.DPSS (+150) or sine-tapers (+30) | S5, S11 |
| S5 BayesianPSD | 100 | GammaPDF/CDF | InverseGammaQuantile (+40) | S6 |
| S6 LombScargle | 150 | LinearRegression | FCDF (+60) | S5, S7 |
| S7 NPDetect | 70 | PowerSpectrum, GammaCDF | — | S8, S9 |
| S8 ROCCurve | 80 | **MannWhitneyU (already ships!)** | — | S7, S9 |
| S9 CFAR (×4) | 180 | MovingAverage, MedianFilter | FQuantile, GammaOrderStat (+80) | — |
| S10 KalmanFilter | 400 | linalg, NormalPDF | — | S2 (twin) |
| S11 Coherence | 120 | PowerSpectrum (×2), BetaCDF | — | aicore connectivity |

**S1 is the keystone** — six others gate on it. **S8 is the cheapest
standalone unlock** (Mann-Whitney AUC trick, no new prob primitive).
**S10 is the largest brick**, separable.

---

## Recommended PR sequence

| PR | Scope | LOC | Days |
|---|---|---|---|
| 1 | publicise ChiSquaredQuantile + InverseGammaQuantile; ship S1 PowerSpectrumCI | 140 | 1 |
| 2 | S8 ROCCurve via MannWhitneyU AUC (no new prob primitive, biggest leverage) | 80 | ½ |
| 3 | S2 WhittleLogLikelihood + expose LevinsonDurbin + S3 ARSpectrum | 160 | 1 |
| 4 | S7 NPDetect + S9 CFAR (CA/OS/GO/SO) + new FQuantile/GammaOrderStat | 280 | 2 |
| 5 | S11 Coherence + segment-Welch helper | 120 | 1 |
| 6 | S6 LombScargle + new prob.FCDF | 250 | 2 |
| 7 | S4 MultitaperPSD + DPSS (or sine-tapers if linalg.TridiagSymEigen deferred) | 250 | 2 |

**Total connective tissue:** ~1130 LOC across 7 PRs ignoring S10 (Kalman
track, agent 137 owns).

---

## Cross-package observations

**1. The χ² distribution is the bridge metal.** Both packages need it,
neither exposes it as a `Distribution`. Promoting `prob.chiSquaredCDF`
(private, `mathutil.go:187`) to a public `ChiSquaredDist` struct
implementing `prob.Distribution` (PDF + CDF + Quantile + Mean + Variance)
unlocks S1, S2, S4, S7 simultaneously. Same move applies to
`GammaDist` (trivially wraps existing `GammaPDF/CDF`).

**2. `signal/` should not import `prob/`.** Composition belongs in a new
third package `spectral/` (or `signal/spectral/`) that depends on both,
preserving signal's minimalism. Matches `reality` precedent: `chaos/`
uses `calculus.RK4`; `orbital/` uses `optim.Bisection`; cross-package glue
lives in the **using** package, not in either base.

**3. The Distribution interface scales here.** Per `prob/distribution.go:14-17`
doc-comment, `prob.Distribution` was a Type-2 cross-pollination from Haskell
typeclass + C# IDistribution. It's the right shape for spectral synergies
because every per-bin operation in S1, S2, S5, S7 is "evaluate the χ² (or
gamma) density and return a probability" — exactly what the interface
promises. Shipping `ChiSquaredDist` and `GammaDist` closes the interface
gap nobody noticed.

**4. autodiff connection.** S2 WhittleLogLikelihood is the cleanest target in
the repo for differentiable spectral inference: `autodiff.Tape`-traced PSD
model → analytic gradients → `optim.LBFGS` (agent 105). Same pattern agent
140 recommended for GARCH and that S10 Kalman will need.

**5. Pistachio 60 FPS budget.** S1, S2, S7, S11 are 60 FPS-compatible if FFT
plans and frequency grids are reused. None of the new code allocates in the
hot path; additional cost on top of `PowerSpectrum` is one-or-two passes
through `chiSquaredCDF` (~30 ns/eval). Net overhead per 1024-bin frame:
<30 µs at 60 FPS. CIs and detectors are essentially free.

**6. Golden-file leverage.** Each synergy primitive has a closed-form answer
at small numbers of test points (S1 χ²_2 quantile at α=0.05 is exactly
−2 ln(0.025); S3 AR(2) two-pole peak FWHM is analytic; S8 AUC = U/(n_pos·
n_neg) exact). 30 vectors × 11 primitives = 330 new test vectors,
reproducible from `math/big` at 256-bit, cross-validatable across the
4-language polyglot.

---

## Explicitly NOT in this report

- Missing primitives within `signal/` alone (agent 132)
- Missing primitives within `prob/` alone (agent 117)
- Numerical bugs in shipped `signal/` or `prob/` (agents 131, 116)
- API ergonomics within either package (agents 134, 118)
- Per-package perf within either (agents 135, 120)
- Audio synergies (agent 167) / acoustics synergies (agent 166)

This report's distinctive contribution is **the bridge** — what emerges
when both packages ship together that neither could deliver alone.

---

## Progress

- 2026-05-08 — agent 151 complete; 11 synergy primitives (S1-S11) catalogued
  with composition graph and 7-PR sequence (~1130 LOC connective tissue);
  identified χ²-distribution promotion as keystone gap, MannWhitneyU↔AUC as
  highest-leverage one-day unlock; recommended `spectral/` as new package
  not modification of either base; first agent of Block B (cross-package
  synergies).

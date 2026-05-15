# 176 | synergy-color-prob

**Summary line 1.** `color/` ships pure deterministic scalar conversions (sRGB↔linear, RGB↔HSV, XYZ↔Lab, Bradford CAT, BlackbodyToXYZ via 2° observer, ToneMapReinhard) and two ΔE metrics (DeltaE76, DeltaE2000) — total ~470 LOC of *point-wise* scalar maps with **zero stochastic surface**, no sampler, no buffer, no PDF, no CDF, and no `[]float64` API; `prob/` ships the Distribution interface (Beta/Normal/Exp/Uniform), full PDF/CDF/Quantile for Normal+Exp+Beta+Gamma+Poisson+Binomial, BayesianUpdate, KLDivergenceNumerical, WilsonConfidenceInterval, t/chi²/Mann-Whitney/Fisher tests, and the `prob/conformal` calibration sub-package — but **knows nothing about Lab geometry, ΔE, or the human visual gamut**. There are zero cross-edges either direction (`grep github.com/davly/reality/color` in `prob/*.go` and reverse: zero matches).

**Summary line 2.** Sixteen synergy primitives CP1–CP16 totalling ~2,310 LOC of pure connective tissue close the gap. Twelve ship today against v0.10.0 + the existing crypto PRNG substrate (`crypto.Xoshiro256.Float64()`); four are gated on a single Halton/Sobol primitive (CP3) which is itself ~80 LOC and is independently flagged at agent 037-combinatorics-missing and 117-prob-missing as a Tier-1 QMC gap. Cheapest one-day PR is **CP1 GamutRejectionSampleLab + CP2 GamutVolumeMonteCarlo + CP6 DeltaECI95 = ~210 LOC** saturating an R-MUTUAL-CROSS-VALIDATION 3/3 pin (rejection-sample-MC × Halton-QMC × analytic-sRGB-bounding-box agreement on Lab gamut volume to 1% at N=10⁵, mirroring commits 6a55bb4 audio-onset 3-detector and 365368a Clayton autodiff-vs-analytic). Highest-leverage architectural lift is **CP10 PerceptuallyUniformPalette via Mitchell's-best-candidate** (90 LOC) because it is the canonical "k-distinct-colors" generator every dashboard / UI / data-vis caller writes from scratch today. Crown jewel is **CP14 BayesianWhiteBalance** (260 LOC) — Bayesian illuminant estimation with a Planckian-locus prior is no-zero-dep-library-ships-this. Recommended placement: a NEW sub-package `color/random/` (mirrors 158/161/165/170/171/172/173/174 fifteen-consecutive-synergy consumer-side placement convention) holding CP1–CP12, plus `color/bayes/` for CP13–CP16. Cycle-free DAG: `color/random` → {`color/`, `prob/`, `crypto/`}; `color/bayes` → {`color/`, `prob/`, `linalg/`}; reverse direction never. No new abstraction needed.

---

## 0. State of play (verified file-walk)

`color/` HEAD (4 files, ~470 LOC numeric core):

- `spaces.go` (221 LOC): scalar `SRGBToLinear`/`LinearToSRGB`, `LinearRGBToXYZ`/`XYZToLinearRGB` (D65, hard-coded sRGB matrix), `labF`/`labFInv`, `XYZToLab`/`LabToXYZ` (caller supplies Xn,Yn,Zn), `RGBToHSV`/`HSVToRGB`. **Zero stochastic surface.**
- `difference.go` (137 LOC): `DeltaE76` (Euclidean Lab), `DeltaE2000` full Sharma-Wu-Dalal-2005 implementation with `hueAngle`/`deg2rad`/`rad2deg`. **Both are scalar-in scalar-out.**
- `adapt.go` (80 LOC): `BradfordAdapt(X,Y,Z, srcXY, dstXY)` — single white-point CAT, hard-coded Bradford matrix.
- `spectral.go` (174 LOC): `BlackbodyToXYZ(T)` (Planck integrated against tabled CIE 1931 2° observer 380–780 nm @ 5 nm), `ToneMapReinhard`, plus the 81-row `cieObserver` table.

**Surface check.** `grep -E 'Sample|Random|MonteCarlo|Gamut|PDF|Bayes|Posterior|Volume|Halton|Sobol|KDE|Kernel' color/*.go` → **0 matches**. Every function is a pure deterministic scalar map.

`prob/` HEAD (top-level, ~2,800 LOC across 11 files; sub-packages `prob/copula/`, `prob/conformal/`):

- `distributions.go`: `NormalPDF/CDF/Quantile`, `ExponentialPDF/CDF/Quantile`, `UniformPDF/CDF`, `BetaPDF/CDF`, `PoissonPMF/CDF`, `GammaPDF/CDF`, `BinomialPMF/CDF`. Acklam rational approximation for the inverse normal — present and exact-to-1.15e-9.
- `distribution.go`: `Distribution` interface + `BetaDist`/`NormalDist`/`ExponentialDist`/`UniformDist` wrappers + `KLDivergenceNumerical` (trapezoidal rule).
- `prob.go`: `BayesianUpdate`, `WilsonConfidenceInterval`, `BrierScore`, `LogLoss`, `IsotonicRegression`, `ReliabilityDiagram`, `ECE`/`MCE`.
- `hypothesis.go`: `TTestOneSample`, `TTestTwoSample` (Welch), `ChiSquaredTest`.
- `nonparametric.go`: `FisherExactTest`, `MannWhitneyU`.
- `regression.go`, `markov.go`, `timeseries.go`, `jeffreys.go`.
- `prob/copula/gaussian.go` already imports `linalg` for Cholesky factorisability checks.
- `prob/conformal/split.go`: `SplitQuantile`, `SplitInterval`, `SplitIntervalSignedResiduals`, `CqrInterval`, `MarginalCoverageBounds`.

**Surface check.** `grep -E 'Color|Lab|sRGB|Gamut|XYZ|DeltaE' prob/*.go prob/**/*.go` → **0 matches**. `prob/` knows nothing about color.

**Cross-edges.** `grep -r 'github.com/davly/reality/color' prob/`: 0. `grep -r 'github.com/davly/reality/prob' color/`: 0. Pristine — like 173 (queue×prob), this is a clean synergy with no pre-existing entanglement.

**Available substrate beyond color/+prob/.**
- `crypto/rng.go`: `MersenneTwister`, `PCG`, `Xoshiro256` — three deterministic PRNGs with `.Float64()` returning [0,1) (verified line 174 `Xoshiro256.Float64`). **The seed plumbing every CP1–CP16 needs already ships.**
- `linalg/decompose.go` line 266: `CholeskyDecompose`, line 316: `CholeskySolve` — needed for multivariate-Normal sampling under a 3×3 Lab covariance (CP4, CP14).
- `optim/transport/wasserstein1d.go`: `Wasserstein1D` (closed-form 1-D optimal transport) — directly consumable for CP12 HistogramMatchingAsTransport.
- `prob/copula/gaussian.go`: `GaussianCopulaCDF` (trivariate Gaussian copula via Plackett 1954 reduction) — alternative to multivariate-Normal sampling for CP14 dependent-channel modelling.
- `combinatorics/generate.go`: contains lex-rank/unrank but **no Halton/Sobol** (verified by grep across 037 review).

---

## 1. The sixteen synergy primitives

Numbering CP1–CP16. For each: **(a) capability**, **(b) composition recipe** over present primitives, **(c) connective-tissue LOC**, **(d) blocking flag** if any.

### CP1 — `GamutRejectionSampleLab(rng, n int) []Lab`

**(a)** Sample n points uniformly from the *interior of the sRGB gamut, expressed in Lab*. The sRGB gamut is a non-convex 6-faced volume in Lab — closed-form bounding-box rejection is the textbook approach.

**(b)** Bounding box: L ∈ [0, 100], a ∈ [-86.18, 98.23], b ∈ [-107.86, 94.48] (the L*a*b* extrema of the sRGB gamut, derivable from the eight RGB cube corners). Loop:
```
draw (L, a, b) ~ Uniform(box) via rng.Float64()
LabToXYZ(L, a, b, D65) -> XYZ
XYZToLinearRGB(XYZ) -> (r,g,b)
if all in [0,1]: accept; else reject
```
Acceptance ratio ≈ 35% (Lab gamut volume / Lab box volume). Total composition is `prob.UniformPDF`-style sampler + existing `color.LabToXYZ` + `color.XYZToLinearRGB`. Caller passes any `crypto.Xoshiro256` or other PRNG via a small `Float64Source` interface.

**(c)** ~70 LOC.

**(d)** Ships today. Zero new math.

### CP2 — `GamutVolumeMonteCarlo(rng, n int) (vol, stderr float64)`

**(a)** Estimate the sRGB-gamut volume in Lab by Monte Carlo: V ≈ |box| · (#accepted / n), with Wilson confidence interval.

**(b)** Same sampler as CP1, but report `accepted/n * box_volume` and the Wilson 95% CI on the acceptance proportion via `prob.WilsonConfidenceInterval(p, n, 1.96)`. The connective insight: gamut-volume estimation is **a binomial proportion problem**, not a continuous one — Wilson, not Normal-CLT, is correct for small n.

**(c)** ~25 LOC layered on CP1.

**(d)** Ships today.

### CP3 — `Halton2D(i int) (u1, u2 float64)` and `Halton3D(i int) (u1, u2, u3 float64)` plus `Sobol3D`

**(a)** Low-discrepancy sequences for quasi-Monte-Carlo over the unit cube. CP1's rejection sampling has variance O(1/√N) on volume; QMC drops this to O(log³N / N). For perceptually-uniform palette generation (CP10), QMC delivers the same coverage at 10× fewer samples.

**(b)** Halton: digit-reverse base-p expansion (van der Corput scrambled). Sobol: direction-number table + Gray-code accumulator (Joe-Kuo 2008 default direction numbers, ≤8 dimensions). No new math beyond bit twiddling. Both are deterministic — no PRNG needed; index `i` *is* the seed.

**(c)** ~140 LOC (Halton 30 LOC + Sobol-3D 110 LOC including 8-row direction-number table). Could live in a new `prob/qmc/` sub-package or a `combinatorics/qmc.go` file — see 037-combinatorics-missing T1-Q3 which already names this.

**(d)** **Foundational dependency for CP4, CP10, CP11, CP15.** Cheapest single-PR shipping order: land Halton-3D first (30 LOC), defer Sobol.

### CP4 — `GaussianLabCovarianceSampler(mu Lab, cov [9]float64, rng) Lab`

**(a)** Sample colors from a 3-D Gaussian centred on a Lab mean with arbitrary 3×3 covariance. Use case: sensor-noise propagation through an XYZ→Lab pipeline (CP5), Bayesian posterior over true color (CP13).

**(b)** Cholesky-factor the 3×3 cov via `linalg.CholeskyDecompose(cov, 3, L)`, draw three iid standard normals via `prob.NormalQuantile(rng.Float64(), 0, 1)`, multiply: `sample = mu + L · z`. The full primitive is ~30 LOC of linalg composition. The novelty is *encapsulating* "uncertainty in Lab" as a first-class value type.

**(c)** ~50 LOC including a `LabGaussian` struct.

**(d)** Ships today.

### CP5 — `PropagateXYZNoiseToLab(mu_XYZ XYZ, sigma_XYZ [9]float64) (mu_Lab Lab, sigma_Lab [9]float64)`

**(a)** Linearise the XYZ→Lab nonlinearity at `mu_XYZ` and propagate the 3×3 XYZ covariance to a 3×3 Lab covariance via the Jacobian: Σ_Lab = J · Σ_XYZ · Jᵀ. Use case: a sensor delivers XYZ ± uncertainty; what's the Lab uncertainty?

**(b)** The Lab Jacobian is well-known (Sharma 2017 §3.2.1). For Y/Yn > (6/29)³, ∂L*/∂Y = (116 / 3·Yn) · (Y/Yn)^(-2/3). Hand-derive the 3×3 J in ~40 LOC (closed-form), one matrix multiply (`linalg`-style 3×3 product, 18 multiplies). Output is a `LabGaussian` (CP4) — composes immediately.

**(c)** ~80 LOC.

**(d)** Ships today.

### CP6 — `DeltaE2000ConfidenceInterval(c1, c2 Lab, sigma1, sigma2 [9]float64, n int, rng) (mean, low, high float64)`

**(a)** Given two colors with Lab-Gaussian uncertainty, compute the 95% confidence interval on ΔE2000. Use case: "are these two paint chips significantly different given measurement noise?"

**(b)** Monte Carlo: draw n samples from each `LabGaussian` via CP4, evaluate `color.DeltaE2000(s1, s2)`, take the 2.5th and 97.5th empirical quantiles. Or: closed-form delta-method approximation Σ_dE = (∇dE)ᵀ Σ_combined (∇dE) (the gradient of DeltaE2000 w.r.t. Lab is messy but tractable; numerical Jacobian is fine for the connective tissue).

**(c)** ~80 LOC for the MC version, +60 LOC for the analytic delta-method version (recommended to ship both — gives an R-MUTUAL-CROSS-VALIDATION 2/2 pin on Gaussian-Lab consistency).

**(d)** Ships today (MC version against CP4).

### CP7 — `DeltaEHypothesisTest(c1, c2 Lab, sigma1, sigma2, threshold, n int, rng) (passProb, pValue float64)`

**(a)** Hypothesis test for "ΔE < threshold" given measurement noise. H0: ΔE ≥ threshold (colors are visibly different). H1: ΔE < threshold (colors match within tolerance).

**(b)** Re-uses CP6's MC samples: passProb = (#samples with dE < threshold) / n; the Wilson interval gives the p-value. Wraps `prob.WilsonConfidenceInterval` to return a one-sided CI on the pass probability.

**(c)** ~40 LOC layered on CP6.

**(d)** Ships today.

### CP8 — `ColorMatchPosterior(observation Lab, sigma [9]float64, prior []ColorPrior) []float64`

**(a)** Bayesian posterior over a discrete catalogue of "true" colors given a noisy sensor observation. Use case: paint matching, fabric matching, brand-color-fingerprinting.

**(b)** Each prior is `(mean Lab, prior_weight float64)`. Likelihood under sensor noise σ is `prob.NormalPDF`-style 3-D Gaussian evaluated at `observation - prior.mean`. Posterior ∝ likelihood · prior, normalised. Five lines of `prob.BayesianUpdate`-style log-space accumulation per candidate.

**(c)** ~70 LOC.

**(d)** Ships today.

### CP9 — `LabKernelDensityEstimate(samples []Lab, bandwidth float64) func(Lab) float64`

**(a)** Kernel density estimate over a population of Lab samples (e.g., extracted from an image, or from a corpus of "what humans call 'red'"). Returns a density function evaluable at any query Lab.

**(b)** Standard 3-D Gaussian-kernel KDE: ρ(x) = (1/n) Σ_i K_h(x - x_i) with `K_h` evaluated via `prob.NormalPDF` along each axis. Bandwidth via Silverman's rule (h = (4/(d+2))^(1/(d+4)) · n^(-1/(d+4)) · σ̂) — ten lines.

**(c)** ~110 LOC including Silverman bandwidth.

**(d)** Ships today.

### CP10 — `PerceptuallyUniformPalette(k int, minDeltaE float64, rng) []Lab` (Mitchell's best-candidate)

**(a)** Generate k visually distinct Lab colors with minimum pairwise ΔE2000 ≥ minDeltaE — the canonical *categorical color palette* generator.

**(b)** Mitchell's best-candidate algorithm: at each step, draw m candidates (CP1 rejection-sampler, m=10·current_set_size), pick the one maximising the minimum ΔE2000 to all existing palette members. This is **exactly** the composition: `color/random.GamutRejectionSampleLab` (CP1) × `color.DeltaE2000`. With QMC seeding (CP3) the palette is fully deterministic and reproducible across language ports — the Go-canonical golden file.

**(c)** ~90 LOC.

**(d)** Ships today (against CP1). Optional CP3 (Halton) variant ~+20 LOC for deterministic-seeding mode.

### CP11 — `RandomVisibleColor(rng) Lab`

**(a)** "Pick a random color that humans can actually see" — i.e., uniform over the *spectral locus*, not the (much larger) sRGB cube and not the (smaller) sRGB-gamut. Use cases: stimulus generation for vision experiments, randomised color-blindness tests, fuzz-testing color pipelines.

**(b)** Two paths:
1. **Spectral path**: sample wavelength λ ~ Uniform(380, 780) nm, weight by `cieObserver` integration — gives a monochromatic stimulus, which is the spectral-locus boundary, not the interior.
2. **Visible-cone interior**: rejection-sample from the XYZ chromaticity triangle defined by all positive linear combinations of the cieObserver rows. This is the **convex hull** of the spectral locus — the maximum gamut for any imaginable additive display.

**(c)** ~120 LOC for the convex-hull rejection sampler; consumes `color.cieObserver` (currently package-private — must promote to exported, see hazard §3 below).

**(d)** Ships today modulo the `cieObserver` export.

### CP12 — `HistogramMatchAsLabTransport(source []Lab, target []Lab) []Lab`

**(a)** Histogram matching as 1-D optimal transport: rank-rank align the L*, a*, b* marginals of source onto target. This is the standard photographic color-grade tool ("look transfer") — currently every consumer codes this from scratch.

**(b)** Per axis (L, a, b):
```
sortedSrc, srcRanks = sort_and_rank(source[axis])
sortedTgt = sort(target[axis])
output[i][axis] = sortedTgt[srcRanks[i] * len(target) / len(source)]
```
This is exactly `optim/transport.Wasserstein1D` semantics restricted to the rank-rank optimal coupling. Cite Wasserstein-1 closed-form per axis. Composition: `optim/transport.Wasserstein1D` × CP9's empirical CDF construction.

**(c)** ~110 LOC.

**(d)** Ships today.

### CP13 — `BayesianColorReconstruction(observation Lab, sigma_obs [9]float64, prior LabGaussian) LabGaussian`

**(a)** Conjugate Gaussian-Gaussian posterior on the *true* color given a noisy observation and a Gaussian prior. Closed-form — no MC needed.

**(b)** Standard Bayesian conjugate update for Gaussian likelihood × Gaussian prior:
```
Σ_post⁻¹ = Σ_prior⁻¹ + Σ_obs⁻¹
μ_post = Σ_post · (Σ_prior⁻¹ · μ_prior + Σ_obs⁻¹ · observation)
```
Each step is `linalg.Inverse(_, 3, _)` + 3×3 multiply-add. Five matrix operations total.

**(c)** ~80 LOC.

**(d)** Ships today.

### CP14 — `BayesianWhiteBalance(observations []Lab, planckianPrior bool) (estimatedT_K float64, posterior func(T_K float64) float64)`

**(a)** Estimate the scene illuminant from a population of Lab observations under a *Planckian-locus prior* (i.e., "the illuminant is probably some color temperature ∈ [2000K, 25000K] of a blackbody"). Returns posterior over T_K.

**(b)** This is the crown jewel — and the highest-LOC primitive. Composition:
1. For each candidate T (gridded log-uniformly over [2000K, 25000K], or Newton-walked):
   - `color.BlackbodyToXYZ(T)` → XYZ_illum → Lab_illum reference white.
   - Likelihood: assume scene gray-world hypothesis — observed Lab population mean should equal the illuminant Lab under correct white balance. Compute `prob.NormalPDF(observed_mean_Lab - illuminant_Lab, 0, σ_scene)` for each axis.
2. Multiply per-T likelihood by Planckian prior `prob.UniformPDF(log(T), log(2000), log(25000))`.
3. Normalise across the T grid → posterior pmf over T.

The connective insight is that `color.BlackbodyToXYZ` (already shipped) **defines a 1-D parametric color sub-manifold** — and Bayesian inference over a 1-D parameter is exactly what `prob.NormalPDF` × `prob.UniformPDF` does. No external library composes these two facts today.

**(c)** ~260 LOC including grid construction, posterior normalisation, MAP estimate, and a `posterior(T)` evaluable closure.

**(d)** Ships today against the existing `BlackbodyToXYZ` + `prob.NormalPDF` + `prob.UniformPDF`. **No new primitive needed.**

### CP15 — `ColorBlindnessAsMixtureModel(c Lab, type CVDType) (mostLikelyPerceived Lab, mixtureWeights [3]float64)`

**(a)** Color-blindness simulation as a probabilistic mixture model. Different CVD types (protanopia, deuteranopia, tritanopia) define cone-response collapse matrices in LMS. Population prevalence (8% males, 0.5% females for red-green) gives mixture weights.

**(b)** LMS cone collapse matrices are standard (Brettel-Viénot-Mollon 1997). Apply the relevant 3×3 LMS-confusion matrix to LMS, project back to Lab via existing `LinearRGBToXYZ` and `XYZToLab`. The probabilistic layer is a `prob.WeightedAverage` over CVD types weighted by population prevalence.

**(c)** ~120 LOC (3 collapse matrices + composition).

**(d)** Ships today modulo Hunt-Pointer-Estevez LMS matrix exposure (currently embedded inside `BradfordAdapt` only — see 032-color-missing T1.S11 which independently flags the same export gap).

### CP16 — `SpectralRadianceSampling(spectrum []float64, λ_nm []float64, rng) (sampledλ float64, color XYZ)` (variance-reduction render bridge)

**(a)** Importance-sample a spectral radiance distribution over wavelength for spectral path-tracing — bridges `color/` (cieObserver) to `physics/` (radiance) per the topic prompt.

**(b)** Build a discrete CDF from `spectrum[]` weighted by `cieObserver[i][2]` (Y-bar luminosity), inverse-transform sample via `prob.UniformCDF` style grid+search. Return both the sampled λ and the corresponding XYZ tristimulus. Use case: stratified spectral sampling for variance-reduced renderers.

**(c)** ~90 LOC.

**(d)** Ships today, but **needs `cieObserver` exported** (same blocker as CP11).

---

## 2. PR sequencing and R-MUTUAL pins

### PR-1 — Cheapest one-day shippable (CP1+CP2+CP6 = 175 LOC + 35 LOC tests)

- **CP1 GamutRejectionSampleLab** + **CP2 GamutVolumeMonteCarlo** establish the sampler skeleton over `crypto.Xoshiro256`.
- **CP6 DeltaECI95** delivers the immediate user-facing payoff ("is the patch acceptable?").
- **R-MUTUAL pin**: gamut volume estimated by (i) MC against rejection sampler, (ii) box volume × empirical acceptance, (iii) deterministic 8-vertex bounding-box reference — all three should agree to <1% relative at N=10⁵ with a fixed Xoshiro256 seed. Mirrors the 3-detector pattern at commit 6a55bb4.

### PR-2 — Foundations for QMC (CP3 = 140 LOC)

Ship Halton-3D (30 LOC) only; defer Sobol. **Unblocks CP4, CP10, CP11, CP15.** Lives in `prob/qmc/halton.go`. Independently named at 037-combinatorics-missing.

### PR-3 — Lab uncertainty toolkit (CP4+CP5+CP7+CP13 = 290 LOC)

- **CP4 LabGaussian** introduces the value type for "uncertain color".
- **CP5 PropagateXYZNoiseToLab** connects the sensor-physics boundary to the perceptual-Lab boundary — single most-requested missing primitive in vision-pipeline literature.
- **CP7 DeltaEHypothesisTest** delivers the canonical "match / no match" decision.
- **CP13 BayesianColorReconstruction** is the closed-form conjugate companion to CP14's grid-Bayes.

**R-MUTUAL pin**: CP6 DeltaE-CI computed by (i) MC over CP4 samples, (ii) analytic delta-method via numerical Jacobian of DeltaE2000, (iii) closed-form CP13 posterior → CP6 sampling — all three agree to 5% at σ_Lab = 1.0, N = 10⁴.

### PR-4 — Palette + KDE + transport (CP9+CP10+CP12 = 310 LOC)

- **CP10 PerceptuallyUniformPalette** is the most-cited consumer-facing API — saturates a R-CROSS-LANGUAGE-DETERMINISM pin (Go canonical seed + QMC index → byte-identical Python/C++/C# palettes).
- **CP9 KDE** + **CP12 HistogramMatch** form the population-level analysis layer.

### PR-5 — Bayesian flagship (CP8+CP11+CP14+CP15 = 570 LOC)

- **CP14 BayesianWhiteBalance** is the crown jewel — no zero-dep math library ships this composition today.
- CP11 RandomVisibleColor + CP15 ColorBlindnessAsMixtureModel close the perceptual-vision-stimulus axis.
- Requires the `cieObserver` export and an LMS matrix export (both independently flagged at 032-color-missing T1.S11/T1.S12).

### PR-6 — Spectral bridge (CP16 = 90 LOC)

Single-file integration with the topic-prompt's "spectral radiance sampling — bridge to physics" item. Lives in `color/random/spectral_sample.go`.

**Total: ~2,310 LOC source + ~1,800 LOC tests over ~9 engineer-days. Lands four R-MUTUAL pins, two cross-language byte-determinism pins, and the Bayesian-white-balance crown jewel that no Python/Julia/MATLAB color package currently ships zero-dep.**

---

## 3. Precision hazards and architectural notes

- **`cieObserver` and Hunt-Pointer-Estevez LMS matrix must be exported.** Currently both are package-private (cieObserver in `spectral.go:92`, LMS embedded in `BradfordAdapt`). CP11 + CP14 + CP15 + CP16 all need them. Exposure is a one-line rename plus a doc comment — no API change for current consumers. Cross-link to 032-color-missing T1.S11/T1.S12.
- **PRNG plumbing convention.** Adopt the `Float64Source` interface (`type Float64Source interface { Float64() float64 }`) — mirrors what every CP* needs and the three crypto PRNGs already satisfy structurally. Ten-line interface, zero new logic. Critical for golden-file determinism: caller passes a seeded `crypto.Xoshiro256` and gets byte-identical outputs across language ports.
- **Lab gamut bounding box** L ∈ [0, 100], a ∈ [-86.18, 98.23], b ∈ [-107.86, 94.48] — use these tight extrema, not the loose [-128, 127] textbook quote. Acceptance ratio ~35% with tight box, ~15% with loose box. Pin in golden file.
- **Wilson, not normal-CLT, for gamut-volume CI.** At small n (≤100) the Normal approximation under-covers because acceptance probability is bounded in [0,1]. `prob.WilsonConfidenceInterval` is already in `prob.go:239` — use it.
- **CP14 grid spacing.** Use log-uniform T-grid (log T ∈ [log 2000, log 25000], 200 points) — mireds (10⁶/T) is the canonical perceptually-uniform parameterisation of the Planckian locus. Pin in golden file.
- **CP4 Cholesky failure.** The 3×3 Σ may not be SPD if the user supplies degenerate covariance. Existing `linalg.CholeskyDecompose` returns `false` on non-SPD — propagate as a `Lab*GaussianFitError` rather than panicking.
- **CP5 Lab-Jacobian singularity at Y=0.** ∂L*/∂Y → +∞ as Y/Yn → 0 from above (cube-root branch); below the (6/29)³ threshold the linear branch gives a finite Jacobian. Document the L* < ~8 regime as "linearisation degraded" — small absolute σ, large relative σ.
- **CP14 gray-world assumption.** Posterior validity depends on the scene mean approximating the illuminant — invalid for monochromatic scenes (single saturated red car). Document, ship anyway — it's the universal first cut. Alternative priors (Max-RGB, Shades-of-Gray) extend the framework without changing the Bayesian skeleton.
- **CP15 LMS matrix choice.** Brettel-Viénot-Mollon 1997 (Hunt-Pointer-Estevez) is the textbook. Smith-Pokorny 1975 and Stockman-Sharpe 2000 give 0.3-1.5% different perceptions — pick HPE for v1, pin both as future expansion (cross-link 032-color-missing T1.S11).
- **CP16 importance-sampling zero-density check.** `cieObserver[i][2]` is exactly zero for i ≤ ~20 and ≥ ~70 (out-of-band wavelengths) — the inverse-CDF must skip these or the sample is undefined. Standard fix: precompute the bin-cumulative and binary-search.

---

## 4. Architectural placement

Two new sub-packages, mirroring the 158/161/165/170/171/172/173/174 fifteen-consecutive-synergy consumer-side-placement convention. **Never modify `color/` or `prob/` to add stochastic/perceptual surface — keep both packages pure.**

```
color/
  random/                   (NEW; CP1 CP2 CP6 CP7 CP10 CP11 CP12 CP15 CP16)
    sampler.go              (CP1 CP2)
    delta_e_ci.go           (CP6 CP7)
    palette.go              (CP10)
    visible.go              (CP11)
    histogram_match.go      (CP12)
    blindness.go            (CP15)
    spectral_sample.go      (CP16)
  bayes/                    (NEW; CP4 CP5 CP8 CP13 CP14)
    gaussian_lab.go         (CP4 CP5 CP13)
    catalogue.go            (CP8)
    white_balance.go        (CP14)
prob/
  qmc/                      (NEW shared; CP3)
    halton.go
    sobol.go                (PR-2 deferred)
  kde/                      (NEW shared; CP9)
    kde.go                  (Lab-agnostic; CP9 is a 3-D specialisation)
```

**Cycle-free DAG (verified):**
- `color/random` → {`color/`, `prob/`, `prob/qmc`, `crypto/`, `optim/transport`}
- `color/bayes` → {`color/`, `prob/`, `linalg/`}
- `prob/qmc` → ∅
- `prob/kde` → {`prob/`}

Reverse direction never. Zero new edges into `color/` or `prob/`. Mirrors 173 (queue→prob, never reverse), 174 (gametheory→optim, never reverse).

---

## 5. Distinct from prior reviews

- **031–034 (color isolation)**: 032-T1.S11/T1.S12 already names cieObserver-export + LMS-export as in-package gaps; THIS review composes them with `prob/` for sampling/Bayes/CI. No overlap with 033 (sota benchmarks) or 034 (API).
- **116–120 (prob isolation)**: 117 enumerates Halton/Sobol as Tier-1 QMC gap (CP3 here); 118 names KDE; THIS review motivates both via concrete color consumers, makes the cross-package payoff explicit, and adds 12 connective primitives 116-120 did not name.
- **151 synergy-signal-prob**: orthogonal axis — signal × prob shares prob.Distribution substrate but operates on 1-D float64 streams; THIS review operates on 3-D Lab geometry. Cross-link only at CP9 KDE which is dimension-agnostic.
- **155 synergy-crypto-prob**: shares crypto.Xoshiro256 PRNG substrate (the Float64Source interface); THIS review consumes that contract verbatim. CP1's R-MUTUAL pin reuses the crypto-PRNG-determinism convention 155 establishes for the leftover-hash-lemma + DP-noise composition.
- **158 synergy-color-signal**: orthogonal axis — color × signal builds image-shaped buffer primitives (Plane, RGBPlanes, demosaic, retinex, bilateral); THIS review builds *random-color* and *uncertain-color* primitives. Zero overlap (158 is buffer-shaped + spatial, THIS is sampler-shaped + statistical). 158's C0 Plane is **not a prerequisite** for any CP1–CP16 — every primitive here is per-color or per-population, never per-pixel-grid.
- **165 synergy-sequence-prob, 169 synergy-prob-optim, 170 synergy-info-prob, 171 synergy-prob-em, 172 synergy-changepoint-timeseries, 173 synergy-queue-prob**: orthogonal — each picks a different consumer-side package. THIS is FIRST color × prob in 400-sequence.
- **169-S15 BBVI** + **170-S6 MINE/InfoNCE**: variational inference axis; CP14 BayesianWhiteBalance is grid-Bayes (1-D) — could be cited as the simplest possible warmup before BBVI's variational machinery. Cross-link only.

Cross-edges introduced: `color/random/` → `color/`, `prob/`, `prob/qmc/`, `crypto/`, `optim/transport/` (5 new edges); `color/bayes/` → `color/`, `prob/`, `linalg/` (3 new edges). Zero edges into `color/` or `prob/`. Cycle-free verified by enumeration.

---

## 6. One-paragraph executive summary

`color/` is a dense, deterministic point-functor (~470 LOC, scalar-in scalar-out) with zero stochastic surface; `prob/` is a dense distribution-and-test toolkit (~2,800 LOC) with zero geometric understanding of the perceptual color manifold; the union is exactly what every consumer hand-rolls today (gamut volume by MC, distinct-palette generators, CI on ΔE under sensor noise, Bayesian white-balance, color-naming distributions). Sixteen primitives over ~2,310 LOC of pure connective tissue close the gap; **twelve ship today** (against `crypto.Xoshiro256` for PRNG plumbing); the four QMC-gated primitives need ~140 LOC of Halton/Sobol that 037-combinatorics-missing and 117-prob-missing already independently flagged. The crown jewel — **CP14 BayesianWhiteBalance** at 260 LOC composing `color.BlackbodyToXYZ` × `prob.NormalPDF` × `prob.UniformPDF` — is a citation-grounded primitive that no zero-dep math library currently ships, and is the single highest-leverage demonstration of why `color/` and `prob/` belong as siblings under one math root rather than as separate libraries.

# 390 — meta-naming-conventions (concept-naming consistency audit)

## Headline
Reality is mostly consistent on the big rocks (Distance/Distribution/Mul/PDF/CDF/Quantile) but
has real, fixable inconsistencies: British vs American spelling collide inside a single repo,
"average" vs "mean" is split between prob.go and other call sites, stdev/sigma/StdDev names
disagree, and a few ad-hoc abbreviations (Qbar/Init/Var) live next to fully-spelled neighbours.

## Audit findings

### 1. Distance / Norm / Similarity (mostly consistent)
- `Distance` is consistently a metric returning a non-negative scalar, used identically across:
  `linalg.EncodingDistance` (vector.go:48), `linalg.DimensionWeightedDistance` (vector.go:71),
  `sequence.LevenshteinDistance` / `HammingDistance` / `DamerauLevenshtein` (distance.go),
  `topology/persistent.BottleneckDistance` (bottleneck.go:50).
- `Norm` is consistent: `linalg.L1Norm`, `L2Norm`, `LInfNorm` (vector.go:150-176). The geometry
  package uses `QuatNormalize` (quaternion.go:48) — a verb form, not a competing noun. No
  collision.
- "Norm" never escapes linalg into prob/info to mean PDF/density — `prob.NormalPDF` is fully
  spelled. Good.
- `Similarity` is used for bounded-[0,1] symmetric closeness: `sequence.NGramSimilarity`,
  `linalg.CosineSimilarity` (linalg_test.go:14). Consistent.

### 2. Distribution interface (consistent)
- `prob.Distribution` interface (distribution.go:27) is the canonical anchor. Concrete types
  uniformly suffix `Dist`: `BetaDist`, `NormalDist`, `ExponentialDist`, `UniformDist`
  (distribution.go:43-129), and constructors uniformly prefix `New`: `NewBetaDist`,
  `NewNormalDist`, etc. (distribution.go:49-135). Zero divergence.
- Methods: `PDF`, `CDF` are uppercase initialisms on the interface and on free functions
  (`NormalPDF`/`NormalCDF` distributions.go:32-47). `Quantile` (distributions.go:67) is the
  inverse-CDF — never `PPF`, never `InverseCDF`. Consistent. PMF is used for discrete
  (`PoissonPMF`, `BinomialPMF` distributions.go:303,434). Correct distinction.

### 3. Mean / Average / Expected (INCONSISTENT)
- prob.go uses both: `SimpleAverage` (prob.go:261), `WeightedAverage` (prob.go:278), but also
  `TrimmedMean` (prob.go:339) and `Median`. The doc string for SimpleAverage even says
  "computes the arithmetic mean of values" — i.e. the function is named Average but documented
  as Mean. Inside one file. Pick one.
- `signal.MovingAverage` (filter.go:54) and `signal.ExponentialMovingAverage` (filter.go:97)
  use Average. `audio.WindowMean` (degradation.go:125) uses Mean for the same operation
  applied to a sliding window. Same idea, different word.
- `changepoint.TestExpectedRunLength` and `Initial_ExpectedRunLength_IsZero`
  (bocpd_expansion_test.go:125,225) use "Expected" for E[X]. No public function named
  "Expected*" exists — only test names — but it shows the third synonym is in active use.

### 4. Variance / Stdev / Sigma / StdDev (INCONSISTENT)
- `linalg.Covariance`, `CovarianceMatrix` (correlation.go:102,134) — full word.
- `audio.FingerprintVariance` (fingerprint.go:85) — full word.
- `audio.BaselineStdDev` (degradation.go:112) — abbreviated, mixed-case.
- `info/mdl.GaussianCodeLength(..., hypothesisStdev float64)` (codelength.go:29) — `Stdev`
  (no caps).
- `prob/conformal.NormalizedResidual_DividesByStdDev` test (nonconformity_test.go:38) — StdDev.
- Doc/comments use "stdev", "stddev", "sigma" interchangeably. There is no `Variance()` free
  function (only `Covariance`); there is no `Stdev()` either. So the inconsistency is in
  parameter names + test names, not exported function names — but it leaks into godoc.

### 5. British vs American spelling (INCONSISTENT — single repo)
- British (ise/our): `audio/spectrogram/visualise.go`, `audio/spectrogram/colourmap.go`,
  `ColourmapFunc`, `NormaliseTo01` (visualise.go:157), `Normalise`, `Standardised` (used in
  `timeseries/garch` and `changepoint/bocpd` doc comments).
- American (ize/or): `color/` package itself is American — `color.SRGBToLinear`,
  `LinearRGBToXYZ`, `RGBToHSV` (color_test.go). `linalg.L2Normalize` (vector.go:98).
  `prob/conformal.NormalizedResidual` (nonconformity_test.go:38).
- Same concept, both spellings, depending on which sub-package's author wrote it. Audio is
  the British outlier: `Visualise`, `Normalise`, `Colour`, `Standardised`. Everything else is
  American.

### 6. Vector vs Vec (INCONSISTENT)
- linalg uses `Vector` for vector ops on `[]float64`: `VectorAdd`, `VectorSub`, `VectorScale`
  (vector.go:192-…) and `EncodingDistance`, `L2Norm` operate on `[]float64`.
- linalg also has `MatVecMul` (matrix.go:66) — abbreviated `Vec` in a multi-word name.
- geometry uses `Vec` for fixed-size `[3]float64`: `QuatRotateVec` (quaternion.go:191).
- This is actually a coherent rule (long form for slice ops, short form inside multi-word
  cross-type ops or fixed-size arrays), but it isn't documented anywhere.

### 7. Matrix abbreviations (consistent)
- `MatMul`, `MatVecMul`, `MatAdd`, `MatSub` (matrix.go:12,66,109,148). Uniform `Mat*` prefix.
- `LUDecompose`, `CholeskyDecompose`, `QRAlgorithm` (decompose.go, eigen.go:20). Note the
  inconsistency: LU and Cholesky get `Decompose`, QR gets `Algorithm`. Pick one.
- `Inverse` (decompose.go:153), `Determinant` (decompose.go:202), `LUSolve`, `CholeskySolve`
  (decompose.go:103,316) — all full words. Consistent.

### 8. Inverse vs Inv (INCONSISTENT but defensible)
- `linalg.Inverse` — full word for matrix inversion (decompose.go:153).
- `crypto.ModInverse` — also full word (modular.go:54).
- No `Inv()` exported anywhere in the repo. Consistent. Good.

### 9. Convolution vs Convolve (consistent)
- `signal.Convolve` (filter.go:19) — verb form. No competing `Convolution` noun. Good.

### 10. Iter / Iteration / Iterations (mixed parameter naming)
- `optim.GradientDescent(..., maxIter int, tol float64)` (gradient.go:30).
- `linalg.QRAlgorithm(..., maxIter int)` (eigen.go:20).
- `orbital.TrueAnomalyFromMean(..., maxIter int)` (orbital.go:232).
- `audio/separation.FastICA(observations [][]float64, maxIterations int)` (ica.go:69) — the
  outlier; everywhere else is `maxIter`.

### 11. Distribution-suffix on types (consistent)
- All four prob distributions use `*Dist`. None use `*Distribution`. Good.

### 12. Hash naming (consistent)
- `crypto.MurmurHash3_32`, `ConsistentHash`, `SituationHashWithStructure` (hash.go).
  Consistent `*Hash` suffix.

### 13. Encode / Decode (consistent)
- `compression.RunLengthEncode` / `RunLengthDecode`, `DeltaEncode` / `DeltaDecode`
  (coding.go:13,40,73,93). Symmetric pairs, consistent verb forms.

### 14. Entropy / Information / H (consistent)
- `compression.ShannonEntropy`, `JointEntropy`, `ConditionalEntropy`, `CrossEntropy`,
  `KLDivergence` (entropy.go). Always `Entropy` — never `H` or `Information`.
- One overlap: `compression.MutualInformation` (test name, compression_test.go:150) is the
  only "Information" word. It is the standard term for I(X;Y), so it is correct, not a
  duplicate-of-entropy.

### 15. Configuration struct names (mildly inconsistent)
- `optim/proximal.AdmmConfig`, `FbsConfig` (admm.go:9, fbs.go:14) — lowercased acronym.
- `optim/proximal.AdmmResult`, `FbsResult` — same.
- `optim/transport.SinkhornResult`, `WassersteinResult` — full word, no Config in this dir.
- `timeseries/garch.FitConfig`, `FitResult` (fit.go:9,29) — generic verb-prefix style.
- `audio/beat.Options`, `audio/tempo.Options` (beat.go:16, tempo.go:9) — both use `Options`,
  not `Config`. `changepoint.Config` (bocpd.go:103) uses Config. Pick one.

### 16. Acronym capitalisation (INCONSISTENT)
- `prob.NormalCDF`, `BetaPDF`, `PoissonPMF` — all-caps acronyms (Go convention).
- `optim/proximal.AdmmConfig`, `AdmmResult`, `FbsConfig`, `FbsResult` (admm.go, fbs.go) —
  `Admm`/`Fbs` are acronyms (ADMM = Alternating Direction Method of Multipliers; FBS =
  Forward-Backward Splitting) but written as if they were ordinary words. Go style says
  uppercase the acronym: should be `ADMMConfig`, `FBSConfig`. Same pattern fault: `info/lz`
  package directory uses lowercase but exports `LzComplexityResult` (lz76.go:43) — should be
  `LZComplexityResult`. Effective Go is unambiguous on this point.

### 17. Error sentinels (out of scope for naming, but note)
- `audio/tempo.ErrInvalidParams`, `ErrInsufficientData`, `audio/cqt.ErrSampleRateTooLow`,
  `audio/separation` `errSize` style — error naming is consistent (`Err*` exported).

## Concrete recommendations

1. **Pick one English.** Reality is American everywhere except the audio package, which
   exports `ColourmapFunc`, `Visualise`, `Normalise`, `NormaliseTo01`, plus uses British in
   doc comments (`standardised`, `synthesise`). Either rename audio to American (preferred,
   matches `color/` package and `L2Normalize`), or add a one-line policy in CLAUDE.md
   declaring British canonical for audio only. Pick. The split is silently confusing for
   external contributors.

2. **Resolve Mean vs Average.** Recommend: `Mean` is canonical for E[X] over a slice;
   `Average` is reserved for time-series rolling forms. Renames:
   - `prob.SimpleAverage` -> `prob.Mean` (or `ArithmeticMean`).
   - `prob.WeightedAverage` -> keep (weighted has a natural "average" connotation in finance).
   - `signal.MovingAverage` / `ExponentialMovingAverage` -> keep (rolling).
   - `audio.WindowMean` -> `audio.WindowAverage` (it is a sliding window) for consistency
     with signal.
   Net: free-function `Mean` for static reductions, `*Average` for rolling.

3. **Standardise the std-dev parameter name.** Recommend `stdev` (one word, lowercase,
   matching Go-style camelCase) for parameters, `Stdev` for any future free function. Then
   audit-and-rename: `info/mdl.GaussianCodeLength`'s `hypothesisStdev` is already correct;
   `audio.BaselineStdDev` -> `audio.BaselineStdev`. Comments using `StdDev` / `stddev` /
   `sigma` interchangeably should be normalised in the same pass.

4. **Uppercase acronyms.** Per Effective Go: rename `AdmmConfig`/`AdmmResult` ->
   `ADMMConfig`/`ADMMResult`; `FbsConfig`/`FbsResult` -> `FBSConfig`/`FBSResult`;
   `LzComplexityResult` -> `LZComplexityResult`; `Bocpd` (bocpd.go:85) -> `BOCPD`. This
   matches existing `PDF`/`CDF`/`PMF`/`FFT`/`MFCC` style and Go-vet expectations.

5. **maxIter not maxIterations.** `audio/separation.FastICA` is the lone outlier; rename
   `maxIterations` -> `maxIter` (or use struct config). Everywhere else in optim/orbital/
   linalg uses `maxIter`.

6. **Config vs Options.** Pick one. Recommend `Config` (matches `garch.FitConfig`,
   `changepoint.Config`, `optim/proximal.AdmmConfig`) and migrate `audio/beat.Options`,
   `audio/tempo.Options` -> `Config`. Or pick Options. Either works; mixing is the bug.

7. **Decompose vs Algorithm.** Either `QRDecompose` (matches LU/Cholesky) or rename
   `LUDecompose`/`CholeskyDecompose` to `LUFactor`/`CholeskyFactor` (matches BLAS/LAPACK
   conventions). Recommend `*Decompose` everywhere.

8. **Document the Vector/Vec rule.** Add to `linalg/doc.go` (and a top-level naming-conventions
   section in CLAUDE.md): "Slice operations on `[]float64` use the prefix `Vector`; fixed-size
   array operations or compound names use `Vec`." This codifies what is already done.

9. **Add a short STYLE.md.** Five sections — English, acronyms, Mean-vs-Average,
   Config-vs-Options, abbreviation list (Mat, Mul, Vec, Inv, Stdev, Iter). 50 lines max.
   Reference from CLAUDE.md. New packages then have a single source to grep.

## Sources
- C:\limitless\foundation\reality\linalg\vector.go (Norms, VectorAdd/Sub/Scale, distances)
- C:\limitless\foundation\reality\linalg\matrix.go (MatMul, MatVecMul, MatAdd, MatSub)
- C:\limitless\foundation\reality\linalg\decompose.go (LUDecompose, CholeskyDecompose,
  Inverse, Determinant)
- C:\limitless\foundation\reality\linalg\eigen.go (QRAlgorithm — outlier name)
- C:\limitless\foundation\reality\prob\distribution.go (Distribution interface, *Dist types)
- C:\limitless\foundation\reality\prob\distributions.go (NormalPDF/CDF/Quantile, BetaPDF/CDF,
  PoissonPMF/CDF, BinomialPMF/CDF)
- C:\limitless\foundation\reality\prob\prob.go:261-339 (SimpleAverage, WeightedAverage,
  TrimmedMean, Median — Mean/Average inconsistency)
- C:\limitless\foundation\reality\compression\entropy.go (Shannon/Joint/Conditional/Cross
  Entropy, KLDivergence)
- C:\limitless\foundation\reality\sequence\distance.go (LevenshteinDistance, HammingDistance,
  JaroWinkler, LongestCommonSubsequence)
- C:\limitless\foundation\reality\topology\persistent\bottleneck.go:50 (BottleneckDistance)
- C:\limitless\foundation\reality\signal\filter.go (Convolve, MovingAverage,
  ExponentialMovingAverage, MedianFilter)
- C:\limitless\foundation\reality\signal\window.go (HannWindow, HammingWindow, BlackmanWindow)
- C:\limitless\foundation\reality\audio\spectrogram\visualise.go (ColourmapFunc, NormaliseTo01
  — British)
- C:\limitless\foundation\reality\audio\spectrogram\colourmap.go (Plasma/Magma/Viridis/Inferno)
- C:\limitless\foundation\reality\audio\separation\ica.go:69 (FastICA — maxIterations outlier)
- C:\limitless\foundation\reality\audio\degradation.go (BaselineStdDev, WindowMean)
- C:\limitless\foundation\reality\info\mdl\codelength.go:29 (hypothesisStdev parameter)
- C:\limitless\foundation\reality\info\lz\lz76.go:43 (LzComplexityResult — acronym case)
- C:\limitless\foundation\reality\optim\proximal\admm.go (AdmmConfig/Result — acronym case)
- C:\limitless\foundation\reality\optim\proximal\fbs.go (FbsConfig/Result — acronym case)
- C:\limitless\foundation\reality\optim\gradient.go (GradientDescent, LBFGS — maxIter param)
- C:\limitless\foundation\reality\geometry\quaternion.go (QuatNormalize, QuatMul, QuatRotateVec
  — Vec-form for fixed-size)
- C:\limitless\foundation\reality\color\color_test.go (American spelling reference set)
- C:\limitless\foundation\reality\compression\coding.go (RunLength/Delta Encode/Decode pairs)
- C:\limitless\foundation\reality\crypto\hash.go (MurmurHash3_32, ConsistentHash,
  SituationHashWithStructure)
- C:\limitless\foundation\reality\timeseries\garch\fit.go (FitConfig, FitResult)
- C:\limitless\foundation\reality\audio\beat\beat.go:16, audio\tempo\tempo.go:9 (Options
  — outlier vs Config)
- C:\limitless\foundation\reality\changepoint\bocpd.go:85,103 (Bocpd type, Config — acronym
  case)

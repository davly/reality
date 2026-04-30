// Package conformal implements distribution-free predictive inference via
// the conformal prediction framework.  Conformal methods take an arbitrary
// predictor (regression or classification) and emit prediction intervals
// (or sets) with finite-sample marginal coverage guarantees of the form
//
//	P( y_{n+1} in C(x_{n+1}) ) >= 1 - alpha
//
// where the only assumption is that the calibration data and the test
// point are exchangeable (much weaker than i.i.d.).  No distributional
// assumption on x or y is required.
//
// # Why this exists in Reality
//
// Per the Ecosystem Revolution Hunt 2026-04-28 entry L01 (lens
// `conformal-prediction`, verdict REVOLUTIONARY) and the Reality Math
// Hunt §03 `08_math_not_yet_in_reality.md` (TIED-TOP at C×T×N = 5×5×4 =
// 100), conformal prediction is the cheapest path to regulator-grade
// (FDA AI/ML 2023-named) calibration in the catalogue.  Day-one consumer
// surface verified at file:line:
//
//   - FW MathLib.ConformalInterval (commit dc63772f) + ConformalCostCalibrator
//     (commit 4bca514f) — the C# reference impl this package mirrors
//   - Recall semantic search SemanticSearchResult.Score (forge.go:180)
//   - Insights forge pipeline Confidence (pipeline.go:160)
//   - Insights diagnostics predictive Confidence (predictive.go:25)
//   - Helicase OCR confidence (ocr_service.py:76-78)
//   - Witness multi-factor confidence (confidence.rs:13-30)
//   - Horizon forecasting confidence_low/high (forecasting.py:121-138)
//   - FW PCN extractor OverallConfidence + FingerprintConfidence
//   - Sentinel-AV confidence_bps + escape_score_bps (ops.rs:51-66)
//   - Nexus Resolution.Confidence (CLAUDE.md:108)
//
// Until this package, every consumer rolled its own quantile-of-residuals
// hack without the Lei et al (2018) finite-sample (n+1)*(1-alpha)
// correction.  The R80b cross-substrate-precision guarantee is enforced
// by TestCrossSubstratePrecision_FwCorpus_* (split_test.go) — the same
// inputs produce the same outputs in Reality Go and FW C#.
//
// # API surface
//
//   - SplitQuantile / SplitInterval — classical Lei-G'Sell-Rinaldo-
//     Tibshirani-Wasserman 2018 split-conformal regression.  Quantile of
//     non-negative nonconformity scores with finite-sample correction.
//   - SplitIntervalSignedResiduals — the FW C# byte-for-byte equivalence
//     point: takes signed residuals, absolutes internally.  Use this
//     when reproducing FW outputs or porting C# corpus tests.
//   - AdaptiveQuantile / AdaptiveInterval — recency-weighted
//     (exponential-decay) variant for non-stationary residual streams
//     (the FW ConformalCostCalibrator drift case: pricing changes break
//     exchangeability of the 7-day window).  Converges to SplitQuantile
//     as halfLife -> infinity.
//   - EffectiveSampleSize — Kish n_eff for adaptive calibration windows;
//     diagnostic for "how many samples is recency-weighting effectively
//     using".
//   - MondrianQuantile / MondrianInterval — class-conditional (per-
//     stratum) variant for stratum-conditional coverage.
//   - CqrInterval / CqrConformityScore — Romano-Patterson-Candes 2019
//     CQR for heteroskedastic predictors with quantile-regression
//     backbones.
//   - NonconformityScorer interface + AbsResidual / NormalizedResidual /
//     LogResidual stock implementations.  Encode the assumption about
//     error shape (homoskedastic / heteroskedastic / multiplicative).
//   - MarginalCoverageBounds — theoretical [1-alpha, 1-alpha + 1/(n+1)]
//     guarantee diagnostic.
//
// # Coverage guarantee
//
// For exchangeable (X_i, Y_i)_{i=1..n+1}:
//
//	P( Y_{n+1} in C(X_{n+1}) ) >= 1 - alpha    (marginal)
//	P( Y_{n+1} in C(X_{n+1}) ) <= 1 - alpha + 1/(n+1)
//
// Conditional coverage P( Y_{n+1} in C(X_{n+1}) | X_{n+1} = x ) is *not*
// guaranteed in general.  Mondrian conformal trades off some sample
// efficiency for stratum-conditional coverage.  AdaptiveQuantile trades
// off long-run coverage for *currently-correct* coverage when the
// residual distribution drifts.
//
// # Determinism + allocations
//
// All operations are deterministic.  Quantile computation sorts a copy of
// the input scores; subsequent calls with the same score vector return
// identical results.  Zero non-stdlib deps (only `math` + `sort` +
// `errors`).
//
// # Deferred to v2
//
// Full conformal (the leave-one-out variant — quadratic in calibration
// size), jackknife+ (Barber et al 2021 AoS 49:486), and online
// adaptive-conformal under distribution shift (Gibbs-Candes 2021, which
// adjusts alpha rather than weighting samples).  Cross-conformal (k-fold
// average of split-conformal calibrations) is also v2.
//
// # References
//
//   - Vovk V., Gammerman A. & Shafer G. (2005). Algorithmic Learning in a
//     Random World.  Springer.
//   - Lei J., G'Sell M., Rinaldo A., Tibshirani R. J. & Wasserman L. (2018).
//     Distribution-Free Predictive Inference for Regression.  JASA 113:1094.
//   - Romano Y., Patterson E. & Candes E. (2019).  Conformalized Quantile
//     Regression.  NeurIPS 32.
//   - Angelopoulos A. N. & Bates S. (2023).  Conformal Prediction: A Gentle
//     Introduction.  FTML 16(4):494-591.
//   - U.S. FDA (2023).  Artificial Intelligence/Machine Learning (AI/ML)-
//     Based Software as a Medical Device (SaMD) Action Plan.  Names
//     conformal prediction as a calibration method for regulatory-grade
//     uncertainty quantification.
package conformal

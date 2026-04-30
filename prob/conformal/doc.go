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
// Per the math + technique cross-pollination hunts, conformal prediction
// has the strongest convergence multiplier of any single primitive
// (M = 9 across the cutting-edge math hunt).  Downstream consumers cite
// it as the canonical mechanism for honest uncertainty in:
//
//   - Forecast bands (B11 risk parity, A09 anomaly decay, A07 cross-
//     sectional ranking)
//   - Tail-risk calibration (D31 vol-target, D34 Kelly, D37 stress)
//   - Strategy gating (E43 promotion ladder — composite-certificate)
//   - Advisor honesty (G61 personality, G63 anti-FOMO)
//
// Until now every consumer has rolled its own quantile-of-residuals hack
// without the Lei et al (2018) or Romano-Patterson-Candes (2019) finite-
// sample correction.  This package centralises the math.
//
// # MVP scope
//
// This package ships:
//
//   - Split conformal regression (Lei-G'Sell-Rinaldo-Tibshirani-Wasserman
//     2018, Distribution-Free Predictive Inference for Regression, JASA
//     113:1094-1111).  Quantile of nonconformity scores with finite-sample
//     (n+1)*(1-alpha)/n inflation.
//   - Mondrian (class-conditional) conformal: per-stratum quantiles for
//     conditional coverage by a categorical taxonomy.
//   - CQR (Romano-Patterson-Candes 2019, Conformalized Quantile Regression,
//     NeurIPS 32).  Symmetric conformity scores around predicted quantiles.
//
// Deferred to v2: full conformal (the leave-one-out variant — quadratic in
// calibration size), jackknife+ (Barber et al 2021 AoS 49:486), adaptive
// conformal under distribution shift (Gibbs-Candes 2021).
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
// efficiency for stratum-conditional coverage.
//
// # Determinism + allocations
//
// All operations are deterministic.  Quantile computation sorts a copy of
// the input scores; subsequent calls with the same score vector return
// identical results.  Zero non-stdlib deps.
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
package conformal

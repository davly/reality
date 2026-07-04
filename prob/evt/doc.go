// Package evt implements Extreme Value Theory: the Generalized Extreme Value
// (GEV) and Generalized Pareto (GPD) distributions, deterministic L-moment /
// probability-weighted-moment estimators (Hosking-Wallis), the Hill tail
// index, maximum-likelihood refinement, the peaks-over-threshold (POT) chain,
// and EVT-based Value-at-Risk / Expected-Shortfall and return levels.
//
// EVT answers the question a mean/variance model cannot: how big is the worst
// case beyond the data we have seen?  The two extremal families are canonical.
// GEV is the only possible non-degenerate limit of block maxima (the
// Fisher-Tippett-Gnedenko / extremal-types theorem); GPD is the only possible
// limit of exceedances over a high threshold (Pickands-Balkema-de Haan).  A
// single shape parameter xi unifies the three limiting behaviours, with the
// xi -> 0 boundary (Gumbel for GEV, Exponential for GPD) handled explicitly.
//
// # Why this exists in Reality
//
// Per the Reality-Fable play hunt (2026-07-04, "Extreme Value Theory",
// verdict CONFIRMED), RubberDuck ships ~700 lines of hand-rolled EVT in C#
// (RubberDuck.Core/Analysis/ExtremeValueTheory.cs POT+PWM GPD, and
// GevMaxDrawdown.cs block-maxima GEV with L-moment + Nelder-Mead MLE), and
// EvtVaR(gpd, 0.99) feeds the live risk-metrics path (RiskMetricsService,
// TailRiskHarvesterStrategy, the get_risk_metrics MCP surface).  That is
// single-substrate tail math that fires exactly when markets blow up, with no
// cross-substrate reference to arbitrate a silent estimator bug.  Reality's
// own doc graph already cited the gap: info/lz/doc.go references
// "RubberDuck GevMaxDrawdown.AIC" as related MDL math while Reality contained
// no GEV/GPD anywhere (only the Gumbel copula).  This package is the
// substrate-neutral Go reference the citation was pointing at.
//
// # API surface
//
//	Distributions (Coles 2001 parameterisation):
//	  GEVCDF / GEVPDF / GEVQuantile / GEVReturnLevel   — GEV, Gumbel limit at xi==0
//	  GPDCDF / GPDPDF / GPDQuantile                     — GPD, Exponential limit at xi==0
//	  GEVParams.Kind(tol)                               — Gumbel/Frechet/Weibull classifier
//
//	Deterministic estimators (closed form, golden-file-able):
//	  LMoments3(data)          — first three sample L-moments
//	  FitGPDPWM(exceedances)   — Hosking-Wallis 1987 GPD PWM fit (xi = 2 - l1/l2)
//	  FitGEVLMoments(maxima)   — Hosking-Wallis GEV L-moment fit (Coles convention)
//	  HillTailIndex / HillAlpha — Hill 1975 heavy-tail index from k upper order stats
//
//	Maximum likelihood (deterministic; L-moment/PWM start, optim.LBFGS refine):
//	  GEVLogLik / GPDLogLik    — log-likelihoods (-Inf outside the support)
//	  FitGEVMLE / FitGPDMLE    — MLE, never worse than the closed-form start
//
//	Peaks-over-threshold + tail risk (McNeil-Frey / Coles):
//	  Exceedances / ThresholdAtRate / FitPOT           — build a POTModel
//	  EvtVaR / EvtES                                    — POT Value-at-Risk / Expected Shortfall
//	  EvtReturnLevel / EvtReturnPeriod                  — m-observation return level / period
//
// # R80b cross-substrate output parity
//
// The TestCrossSubstratePrecision_RubberDuck_* tests (parity_test.go) assert
// that the same inputs produce the same outputs in this Go package and in the
// RubberDuck C# originals, following the output-parity (not strict-byte)
// pattern established for prob/copula (see prob/copula/doc.go).  Parity is
// pinned on the convention-identical pieces: the GPD PWM fit (RubberDuck
// ExtremeValueTheory.FitGpd uses the same Hosking-Wallis 1987 estimator), the
// GEV/GPD distribution functions, and EvtVaR/EvtES/return-period (identical
// closed forms).  Tolerance: <= 1e-9 absolute.
//
// Note (arbitration finding): RubberDuck's *GEV L-moment* shape estimator in
// GevMaxDrawdown.EstimateLMoments uses a non-standard polynomial in the
// L-skewness (xi ~= 7.8590 t3 + 2.9554 t3^2 + 4.5 t3^3) rather than the
// published Hosking-Wallis map through c = 2/(3+t3) - ln2/ln3 that this
// package implements.  The two agree only near t3 ~= 0 and diverge in the
// tails; FitGEVLMoments here is the canonical reference, so parity on the GEV
// *fit* is intentionally NOT asserted — that divergence is exactly the
// single-implementation risk this reference exists to expose.
//
// # Precision + design
//
// Pure float64, deterministic, zero non-stdlib dependencies (only math + sort;
// math.Gamma for the GEV L-moment back-out).  The only internal Reality
// dependency is optim.LBFGS for MLE refinement; all closed-form estimators and
// distribution functions are self-contained.  The xi != 0 distribution
// branches use log1p/expm1 so they degrade smoothly to the Gumbel/Exponential
// limits as xi -> 0.  Estimators are the deterministic (no-MCMC) branch of EVT
// fitting, so their outputs are golden-file reproducible.
//
// # References
//
//   - Coles, S. (2001).  An Introduction to Statistical Modeling of Extreme
//     Values.  Springer.  (GEV/GPD definitions, POT, return levels — Ch. 3-4.)
//   - Fisher, R.A. & Tippett, L.H.C. (1928).  Limiting forms of the frequency
//     distribution of the largest or smallest member of a sample.  Proc.
//     Cambridge Phil. Soc. 24: 180-190.
//   - Pickands, J. (1975).  Statistical inference using extreme order
//     statistics.  Annals of Statistics 3: 119-131.
//   - Hosking, J.R.M. (1990).  L-moments: analysis and estimation of
//     distributions using linear combinations of order statistics.  JRSS-B
//     52: 105-124.
//   - Hosking, J.R.M. & Wallis, J.R. (1987).  Parameter and quantile
//     estimation for the generalized Pareto distribution.  Technometrics 29:
//     339-349.
//   - Hosking, J.R.M., Wallis, J.R. & Wood, E.F. (1985).  Estimation of the
//     GEV distribution by the method of probability-weighted moments.
//     Technometrics 27: 251-261.
//   - Hill, B.M. (1975).  A simple general approach to inference about the
//     tail of a distribution.  Annals of Statistics 3: 1163-1174.
//   - McNeil, A.J. & Frey, R. (2000).  Estimation of tail-related risk
//     measures for heteroscedastic financial time series: an extreme value
//     approach.  Journal of Empirical Finance 7: 271-300.
//   - McNeil, A.J., Frey, R. & Embrechts, P. (2005).  Quantitative Risk
//     Management.  Princeton University Press, §7.2.
//   - Embrechts, P., Kluppelberg, C. & Mikosch, T. (1997).  Modelling
//     Extremal Events for Insurance and Finance.  Springer.
package evt

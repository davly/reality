package prob

import (
	"math"

	"github.com/davly/reality/constants"
)

// ---------------------------------------------------------------------------
// Selection-bias-corrected Sharpe-ratio statistics.
//
// The Sharpe ratio (SR) of a track record is a noisy point estimate: with a
// short sample, fat tails, or — most dangerously — after picking the best of
// many trials, a large observed SR can be pure luck. These functions turn a
// raw SR into a probability statement about the *true* SR, and correct the
// bar upward for the number of trials tried (multiple-testing / selection
// bias). They are the exact math that separates skill from selection luck in a
// strategy-promotion gate.
//
// Reference (all four functions):
//
//	Bailey, D.H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
//	Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
//	The Journal of Portfolio Management, 40(5), 94-107.
//
//	The Probabilistic Sharpe Ratio and Minimum Track Record Length originate in
//	Bailey, D.H. & López de Prado, M. (2012). "The Sharpe Ratio Efficient
//	Frontier." Journal of Risk, 15(2), 3-44.
//
// Moment conventions (match the papers and RubberDuck's DeflatedSharpe.cs):
//
//   - observedSR / benchmark are per-observation (NON-annualized) Sharpe
//     ratios in the same frequency as n.
//   - skew is the standardized third moment of returns (0 for a normal
//     distribution).
//   - kurt is the RAW (non-excess) fourth standardized moment (3 for a normal
//     distribution). Passing excess kurtosis is a bug: use kurt = excess + 3.
//
// All functions are pure, deterministic, and depend only on the Go standard
// library plus reality's own constants and normal primitives (NormalCDF via
// math.Erfc for ~15-digit CDF accuracy; the Acklam quantile with documented
// relative error < 1.15e-9 for the inverse-normal used by ExpectedMaxSharpe).
//
// Consumer: RubberDuck's MultipleTestingGate issues an EdgeCertificate whose
// pass/fail is a minimum-Deflated-Sharpe threshold; the same ExpectedMaxSharpe
// correction applies to any best-of-N promotion loop in the estate.
// ---------------------------------------------------------------------------

// ProbabilisticSharpeRatio returns the Probabilistic Sharpe Ratio (PSR): the
// probability that the true (population) Sharpe ratio exceeds benchmark, given
// an observed Sharpe ratio estimated from n returns with the given skewness and
// (raw) kurtosis.
//
// Formula (Bailey & López de Prado 2012, 2014):
//
//	PSR(benchmark) = Z[ (observedSR - benchmark) * sqrt(n - 1)
//	                    / sqrt(1 - skew*observedSR + (kurt-1)/4 * observedSR^2) ]
//
// where Z is the standard-normal CDF. The denominator is the estimated standard
// error of the Sharpe ratio under non-normal returns; under normality (skew=0,
// kurt=3) it reduces to sqrt(1 + observedSR^2/2), the classical Lo (2002)
// result.
//
// Valid range: n >= 2; kurt >= 1 (raw). Returns a probability in (0, 1).
// Returns NaN if n < 2, or if the variance term inside the square root is
// non-positive (which cannot happen for valid moments but is guarded).
// Precision: limited by NormalCDF (~15 significant digits); inputs are used
// exactly.
func ProbabilisticSharpeRatio(observedSR float64, n int, skew, kurt, benchmark float64) float64 {
	if n < 2 {
		return math.NaN()
	}
	variance := 1.0 - skew*observedSR + (kurt-1.0)/4.0*observedSR*observedSR
	if variance <= 0 {
		return math.NaN()
	}
	num := (observedSR - benchmark) * math.Sqrt(float64(n-1))
	den := math.Sqrt(variance)
	return NormalCDF(num/den, 0, 1)
}

// ExpectedMaxSharpe returns the expected maximum Sharpe ratio across nTrials
// independent trials whose estimated Sharpe ratios have variance srVariance,
// under the null hypothesis of zero true skill.
//
// This is the multiple-testing / selection-bias correction: even with no edge,
// the best of many trials will show a positive Sharpe by chance, and this is
// its expected magnitude — the bar a genuine strategy must clear.
//
// Formula (Bailey & López de Prado 2014), the expected value of the maximum of
// nTrials i.i.d. standard Gaussians scaled by the SR standard deviation:
//
//	E[max SR] = sqrt(srVariance) * [ (1-gamma) * Z^-1(1 - 1/N)
//	                                 +   gamma  * Z^-1(1 - 1/(N*e)) ]
//
// where Z^-1 is the standard-normal quantile, gamma is the Euler-Mascheroni
// constant (constants.EulerGamma), e is Euler's number, and N = nTrials. Note
// the paper parameterizes by the VARIANCE V of the trial Sharpe estimates and
// scales by V^(1/2); RubberDuck's DeflatedSharpe.cs passes the standard
// deviation directly — the two agree when srVariance = (trialSrStdDev)^2.
//
// Valid range: nTrials >= 1, srVariance >= 0. Returns 0 for nTrials == 1 (the
// maximum of a single trial has zero expected excess under the null) and for
// srVariance == 0. Returns NaN if nTrials < 1 or srVariance < 0.
// Precision: dominated by the Acklam quantile (relative error < 1.15e-9); the
// closed form is itself an asymptotic approximation to the true expected
// maximum order statistic (within ~1-2% for the tabulated N).
func ExpectedMaxSharpe(nTrials int, srVariance float64) float64 {
	if nTrials < 1 || srVariance < 0 || math.IsNaN(srVariance) {
		return math.NaN()
	}
	if nTrials == 1 || srVariance == 0 {
		return 0.0
	}
	gamma := constants.EulerGamma
	n := float64(nTrials)
	z1 := standardNormalQuantile(1.0 - 1.0/n)
	z2 := standardNormalQuantile(1.0 - 1.0/(n*math.E))
	return math.Sqrt(srVariance) * ((1.0-gamma)*z1 + gamma*z2)
}

// DeflatedSharpeRatio returns the Deflated Sharpe Ratio (DSR): the Probabilistic
// Sharpe Ratio evaluated against the multiple-testing-inflated benchmark
// ExpectedMaxSharpe(nTrials, srVariance) instead of zero.
//
// DSR = PSR(benchmark = E[max SR | nTrials, srVariance]). It answers: given that
// this strategy is the best of nTrials tried, what is the probability its true
// Sharpe still exceeds what pure selection luck would have produced? A value
// near 1 is strong evidence of genuine edge; near 0.5 or below is consistent
// with overfitting.
//
// Valid range: n >= 2, kurt >= 1 (raw), nTrials >= 1, srVariance >= 0.
// Returns a probability in (0, 1), or NaN on invalid inputs (propagated from
// ExpectedMaxSharpe or ProbabilisticSharpeRatio).
// Reference: Bailey & López de Prado (2014).
func DeflatedSharpeRatio(observedSR float64, n int, skew, kurt float64, nTrials int, srVariance float64) float64 {
	benchmark := ExpectedMaxSharpe(nTrials, srVariance)
	if math.IsNaN(benchmark) {
		return math.NaN()
	}
	return ProbabilisticSharpeRatio(observedSR, n, skew, kurt, benchmark)
}

// MinTrackRecordLength returns the Minimum Track Record Length (MinTRL): the
// number of observations required to conclude, at the given confidence level,
// that the true Sharpe ratio exceeds benchmark.
//
// Formula (Bailey & López de Prado 2012, 2014):
//
//	MinTRL = 1 + [ 1 - skew*observedSR + (kurt-1)/4 * observedSR^2 ]
//	             * ( Z^-1(confidence) / (observedSR - benchmark) )^2
//
// where Z^-1 is the standard-normal quantile. It is the sample size at which the
// PSR against benchmark first reaches `confidence`. The result is returned as a
// real number (observations); callers typically round up.
//
// Valid range: confidence in (0, 1); observedSR > benchmark (a track record can
// never establish a Sharpe it does not beat); kurt >= 1 (raw).
// Returns NaN if confidence is outside (0, 1), if observedSR <= benchmark, or if
// the variance term is non-positive.
// Precision: dominated by the Acklam quantile (relative error < 1.15e-9).
func MinTrackRecordLength(observedSR float64, skew, kurt, benchmark, confidence float64) float64 {
	if confidence <= 0 || confidence >= 1 {
		return math.NaN()
	}
	if observedSR <= benchmark {
		return math.NaN()
	}
	variance := 1.0 - skew*observedSR + (kurt-1.0)/4.0*observedSR*observedSR
	if variance <= 0 {
		return math.NaN()
	}
	z := standardNormalQuantile(confidence)
	ratio := z / (observedSR - benchmark)
	return 1.0 + variance*ratio*ratio
}

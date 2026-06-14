// Package timeseries provides streaming (online, single-pass) time-series
// primitives over primitive scalar inputs. It currently exposes the
// exponentially-weighted moments tracker EWMoments; the GARCH and DCC
// volatility models live in the sibling sub-packages timeseries/garch and
// timeseries/dcc.
//
// # EWMoments: exponentially-weighted mean + variance
//
// EWMoments maintains an exponentially-weighted (EW) running mean and variance
// of a scalar stream in O(1) time and O(1) space per update. It is the variance
// leg of the classic EWMA control chart: where reality's
// signal.ExponentialMovingAverage smooths the VALUE only, EWMoments additionally
// tracks how much the stream is dispersing around that smoothed value, which is
// what a caller needs to emit a drift / anomaly / "stable" SIGNAL via a z-score.
//
// Smoothing factor alpha is the weight given to the NEWEST sample (the same
// convention as signal.ExponentialMovingAverage: out[i] = alpha*x[i] +
// (1-alpha)*out[i-1]); alpha must lie in (0, 1]. A larger alpha forgets the past
// faster (more responsive, noisier); a smaller alpha is smoother and slower.
//
// The recurrence, for the n-th sample x with the mean/variance held BEFORE this
// update written mean_-, var_- :
//
//	first sample (n == 1):  mean = x,  var = 0
//	thereafter (n >= 2):    diff = x - mean_-                 // uses the OLD mean
//	                        mean = mean_- + alpha*diff        // == (1-alpha)*mean_- + alpha*x
//	                        var  = (1-alpha)*var_- + alpha*diff*diff
//
// The deviation `diff` is taken against the mean as it stood BEFORE this update,
// which is the convention used by every cross-checked downstream reinvention
// (see "Why this exists" below). This is West's (1979) online weighted-variance
// form and is algebraically the standard EWMA / EWMV control-chart pair under
// the substitution alpha_here = 1 - lambda for sources that parameterise by the
// retention weight lambda.
//
// # Edge / degenerate conventions (matched to the reinventions)
//
//   - Empty tracker (no Update yet, n == 0): Mean, Variance, StdDev all return 0,
//     and ZScore returns 0 (no baseline => no information). This is a documented
//     convention, not an error: EWMoments never panics on a query.
//   - First sample (n == 1): Variance (hence StdDev) is exactly 0 — a single
//     observation carries no dispersion information. The downstream
//     reinventions seed var = 0 on the first sample identically.
//   - Constant stream: every diff is 0, so the variance decays toward 0 and the
//     mean equals the constant. ZScore against a zero-StdDev baseline returns 0
//     when x equals the mean and ±Inf otherwise (an honest "departed from a
//     constant baseline" sentinel — the caller owns whatever verdict it draws,
//     exactly as the grounded drift detector treats a near-zero sigma).
//
// EWMoments is intentionally decision-neutral: it returns mean / variance /
// stddev / z-score and never a threshold, verdict, or alert. The caller owns the
// control limit (e.g. |z| > 3) and the action.
//
// # Why this exists in Reality
//
// The EW-variance recurrence — and a z-score drift verdict built on its square
// root — was independently reinvented in at least three distinct flagship
// packages, each rolling its own copy of the same scalar recurrence:
//
//   - insights/internal/trend/trend.go:50  — newVar = alpha*(diff*diff) +
//     (1-alpha)*EMAVariance, with diff = x - EMA (old mean); IsStable thresholds
//     on EMAVariance.
//   - limitless-ops/internal/grounded/grounded.go:231 — ewmaVar = alpha*ewmaVar +
//     (1-alpha)*deviation*deviation, deviation = x - prevMean; emits a z-score
//     drift report z = (x-mean)/sqrt(ewmaVar), seeding var = 0 on the first
//     sample (grounded.go:223).
//   - limitless-ops/internal/forge/pipeline.go:229 and
//     mirror/internal/forge/pipeline.go:232 — the same EW mean/var pair inside
//     a FORGET/decay stage (these two use an absolute deviation rather than a
//     squared one in the variance term — a divergence that a single shared
//     primitive removes).
//
// (Those sources parameterise alpha as the retention weight on the OLD value;
// EWMoments parameterises alpha as the weight on the NEW value to stay
// consistent with reality's existing signal.ExponentialMovingAverage. The two
// are the same recurrence under alpha_here = 1 - alpha_there.)
//
// Centralising the recurrence here fixes the math once so the downstream drift /
// anomaly / stability signals agree numerically instead of each forge rolling a
// slightly different (and in two cases non-squared) variance update.
//
// # Determinism + allocations
//
// All operations are deterministic and allocation-free; the only dependency is
// the Go standard library (math).
//
// # References
//
//   - West, D.H.D. (1979). Updating mean and variance estimates: an improved
//     method. Communications of the ACM 22(9): 532-535.
//   - Roberts, S.W. (1959). Control chart tests based on geometric moving
//     averages. Technometrics 1(3): 239-250. (the EWMA control chart.)
//   - Montgomery, D.C. (2013). Introduction to Statistical Quality Control,
//     7th ed., ch. 9 (EWMA and the EWMV variance leg).
package timeseries

import "math"

// EWMoments is an online exponentially-weighted mean + variance tracker over a
// scalar stream. The zero value is NOT usable; construct one with NewEWMoments.
//
// EWMoments is not safe for concurrent use; guard it with the caller's own lock
// if shared across goroutines.
type EWMoments struct {
	alpha float64 // weight on the newest sample, in (0, 1]
	mean  float64
	varr  float64
	n     uint64 // number of samples observed
}

// NewEWMoments returns an EWMoments with smoothing factor alpha, the weight
// given to the newest sample. alpha must lie in (0, 1]; NewEWMoments panics
// otherwise (a programming error, mirroring signal.ExponentialMovingAverage).
//
// alpha == 1 reduces the tracker to "last value only" (mean == last sample,
// variance == squared last step), which is a valid, fully-forgetful limit.
func NewEWMoments(alpha float64) *EWMoments {
	if !(alpha > 0) || alpha > 1 {
		panic("timeseries.NewEWMoments: alpha must be in (0, 1]")
	}
	return &EWMoments{alpha: alpha}
}

// Alpha returns the smoothing factor (weight on the newest sample).
func (e *EWMoments) Alpha() float64 { return e.alpha }

// Count returns the number of samples observed so far.
func (e *EWMoments) Count() uint64 { return e.n }

// Update folds one sample x into the running mean and variance.
//
// On the first sample the mean is set to x and the variance to 0. On every
// subsequent sample the deviation is measured against the mean as it stood
// BEFORE this update:
//
//	diff = x - mean
//	mean = mean + alpha*diff
//	var  = (1-alpha)*var + alpha*diff*diff
func (e *EWMoments) Update(x float64) {
	if e.n == 0 {
		e.mean = x
		e.varr = 0
		e.n = 1
		return
	}
	diff := x - e.mean
	e.mean += e.alpha * diff
	e.varr = (1-e.alpha)*e.varr + e.alpha*diff*diff
	e.n++
}

// Mean returns the current exponentially-weighted mean, or 0 if no sample has
// been observed.
func (e *EWMoments) Mean() float64 { return e.mean }

// Variance returns the current exponentially-weighted variance, or 0 if no
// sample has been observed. After exactly one sample the variance is 0.
func (e *EWMoments) Variance() float64 { return e.varr }

// StdDev returns the square root of Variance (the EW standard deviation), or 0
// if no sample has been observed or after exactly one sample.
func (e *EWMoments) StdDev() float64 { return math.Sqrt(e.varr) }

// ZScore returns the standardised distance of x from the current mean,
// (x - Mean) / StdDev, WITHOUT folding x into the tracker. It is the drift
// signal a caller compares against its own control limit (e.g. |z| > 3).
//
// Degenerate baselines are reported honestly rather than hidden:
//   - no samples yet (n == 0): returns 0 (no baseline => no information).
//   - zero StdDev (a constant or single-sample baseline): returns 0 when x
//     equals the mean, and ±Inf otherwise (math.Inf with the sign of x-mean),
//     signalling a departure from a constant baseline. The caller decides what,
//     if anything, an infinite z-score means for its verdict.
func (e *EWMoments) ZScore(x float64) float64 {
	if e.n == 0 {
		return 0
	}
	sd := math.Sqrt(e.varr)
	if sd == 0 {
		if x == e.mean {
			return 0
		}
		return math.Inf(int(math.Copysign(1, x-e.mean)))
	}
	return (x - e.mean) / sd
}

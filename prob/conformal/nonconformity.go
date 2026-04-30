package conformal

import "math"

// A NonconformityScorer maps a (predicted, actual) pair to a non-negative
// nonconformity score.  The conformal calibration step then applies the
// (n+1)*(1-alpha) finite-sample quantile of the score distribution to
// produce a coverage-guaranteed interval.
//
// The choice of scorer encodes the assumption about the predictor's error
// shape.  AbsResidual is the canonical default (the Lei et al 2018
// formulation).  NormalizedResidual handles heteroskedastic predictors by
// dividing by an estimate of the local conditional standard deviation; this
// is asymptotically equivalent to CQR but avoids fitting two quantile
// regressions.  LogResidual handles multiplicative error processes (cost
// estimation, latency, prices) where additive bands over-cover small
// magnitudes and under-cover large ones.
//
// All scorers are pure: they have no internal state and produce a finite,
// non-negative float64 for any finite (predicted, actual) input.  Callers
// pass a nil-equivalent zero value (zero for stdDev / scale) to get the
// default AbsResidual behaviour.
type NonconformityScorer interface {
	// Score returns the nonconformity score s_i = score(predicted, actual).
	// Must be non-negative when both inputs are finite (otherwise the
	// SplitQuantile path would reject it via ErrInvalidScore).
	Score(predicted, actual float64) float64

	// Name returns a short stable identifier for this scorer (audit + log
	// emission).
	Name() string
}

// =========================================================================
// AbsResidual — default split-conformal scorer (Lei et al 2018)
// =========================================================================

// AbsResidual is the canonical conformal nonconformity score
//
//	s_i = |actual - predicted|
//
// Use when the predictor is roughly homoskedastic.  This is the score
// implied by the FW MathLib.ConformalInterval.Compute reference impl
// (signed residuals are taken absolute internally).
type AbsResidual struct{}

// Score returns |actual - predicted|.
func (AbsResidual) Score(predicted, actual float64) float64 {
	r := actual - predicted
	if r < 0 {
		return -r
	}
	return r
}

// Name returns "abs_residual".
func (AbsResidual) Name() string { return "abs_residual" }

// =========================================================================
// NormalizedResidual — heteroskedastic scorer
// =========================================================================

// NormalizedResidual is the locally-scaled nonconformity score
//
//	s_i = |actual - predicted| / max(stdDev(predicted), eps)
//
// where stdDev is a function from the predictor's input/output to an
// estimate of the local conditional standard deviation.  Use when the
// predictor's error variance varies systematically across the input space
// (the classic example: forecasting error grows with horizon, regression
// error grows with feature norm).  Conformal under this score yields
// a *narrower* interval at low-uncertainty inputs and a *wider* interval
// at high-uncertainty ones.
//
// StdDevFn must return a positive value; the scorer floors at Eps to avoid
// division-by-zero blowups under degenerate inputs.  Default Eps = 1e-9 if
// zero.
type NormalizedResidual struct {
	// StdDevFn maps the point prediction to its local conditional sigma
	// estimate.  Required.
	StdDevFn func(predicted float64) float64

	// Eps is the lower floor on the divisor to avoid blowups when the
	// stdDev estimate underflows.  Default 1e-9.
	Eps float64
}

// Score returns |actual - predicted| / max(stdDev(predicted), eps).
//
// Panics if StdDevFn is nil — construction-time error, not a calibration-
// time runtime issue.
func (n NormalizedResidual) Score(predicted, actual float64) float64 {
	if n.StdDevFn == nil {
		panic("conformal.NormalizedResidual: StdDevFn must be non-nil")
	}
	r := actual - predicted
	if r < 0 {
		r = -r
	}
	denom := n.StdDevFn(predicted)
	eps := n.Eps
	if eps <= 0 {
		eps = 1e-9
	}
	if denom < eps {
		denom = eps
	}
	return r / denom
}

// Name returns "normalized_residual".
func (NormalizedResidual) Name() string { return "normalized_residual" }

// =========================================================================
// LogResidual — multiplicative-error scorer
// =========================================================================

// LogResidual is the log-space nonconformity score
//
//	s_i = |log(max(actual, eps)) - log(max(predicted, eps))|
//	    = |log(actual / predicted)|       (when both > 0)
//
// Use when the predictor's error is multiplicative (cost estimation,
// latency-percentile, price prediction): an additive band would either
// over-cover small magnitudes (band swallows the value) or under-cover
// large ones (band fails to span the value).  The exponentiated half-
// width q yields a multiplicative band [predicted/exp(q), predicted*exp(q)]
// with the same finite-sample coverage guarantee as the additive band.
//
// Both predicted and actual must be > 0 in the calibration set; the Eps
// floor protects against zero values from the predictor.  Default Eps =
// 1e-12 if zero.
type LogResidual struct {
	// Eps is the lower floor on log inputs to avoid -Inf when predictor
	// or actual underflow to zero.  Default 1e-12.
	Eps float64
}

// Score returns |log(actual) - log(predicted)|.  Inputs are floored at
// Eps before taking the log to avoid -Inf.
func (l LogResidual) Score(predicted, actual float64) float64 {
	eps := l.Eps
	if eps <= 0 {
		eps = 1e-12
	}
	p := predicted
	if p < eps {
		p = eps
	}
	a := actual
	if a < eps {
		a = eps
	}
	r := math.Log(a) - math.Log(p)
	if r < 0 {
		return -r
	}
	return r
}

// Name returns "log_residual".
func (LogResidual) Name() string { return "log_residual" }

// =========================================================================
// Convenience: vector scorers
// =========================================================================

// ScoreAll applies a NonconformityScorer to a parallel pair of predicted
// and actual slices, returning the resulting score vector.  Returns nil if
// the slices have different lengths.
func ScoreAll(scorer NonconformityScorer, predicted, actual []float64) []float64 {
	if len(predicted) != len(actual) {
		return nil
	}
	out := make([]float64, len(predicted))
	for i := range predicted {
		out[i] = scorer.Score(predicted[i], actual[i])
	}
	return out
}

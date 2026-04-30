package conformal

import (
	"errors"
	"math"
	"sort"
)

// ErrInvalidAlpha is returned when alpha is outside (0, 1).
var ErrInvalidAlpha = errors.New("conformal: alpha must be in (0, 1)")

// ErrEmptyCalibration is returned when the calibration score vector is
// empty.
var ErrEmptyCalibration = errors.New("conformal: calibration scores must be non-empty")

// ErrInvalidScore is returned when a calibration score is NaN or negative
// (most nonconformity scores are non-negative absolute residuals).
var ErrInvalidScore = errors.New("conformal: calibration scores must be finite and non-negative")

// SplitQuantile returns the conformal threshold q such that any test-point
// nonconformity score s_test <= q implies the test point is "in-coverage"
// at the (1 - alpha) marginal level.
//
// Under exchangeability of the n calibration scores and the test score, the
// q computed here gives
//
//	P( s_{n+1} <= q ) >= 1 - alpha
//
// finite-sample.  The Lei et al (2018) formula uses the
//
//	ceil((n+1)(1 - alpha)) / n
//
// quantile rank of the empirical score distribution; we return +Inf when
// that rank exceeds n (which can happen when n*(1-alpha) is too small,
// equivalently when n < ceil(1/alpha) - 1).
//
// scores must be non-negative and finite; alpha must be in (0, 1).
func SplitQuantile(scores []float64, alpha float64) (float64, error) {
	if alpha <= 0 || alpha >= 1 || math.IsNaN(alpha) {
		return 0, ErrInvalidAlpha
	}
	if len(scores) == 0 {
		return 0, ErrEmptyCalibration
	}
	for _, s := range scores {
		if math.IsNaN(s) || s < 0 {
			return 0, ErrInvalidScore
		}
	}
	n := len(scores)
	rank := int(math.Ceil((float64(n) + 1.0) * (1.0 - alpha)))
	if rank > n {
		return math.Inf(1), nil
	}
	if rank < 1 {
		rank = 1
	}
	sorted := make([]float64, n)
	copy(sorted, scores)
	sort.Float64s(sorted)
	return sorted[rank-1], nil
}

// SplitInterval returns a symmetric prediction interval [yhat - q, yhat + q]
// where q is the conformal threshold computed from absolute-residual
// calibration scores.  This is the simplest form of split conformal
// regression: the nonconformity score is |y - yhat|.
//
// yhat is the point prediction at the test x.  calibrationResiduals are
// the absolute residuals |y_i - yhat_i| computed on a held-out calibration
// fold (NOT used for fitting the underlying model).
func SplitInterval(yhat float64, calibrationResiduals []float64, alpha float64) (lo, hi float64, err error) {
	q, err := SplitQuantile(calibrationResiduals, alpha)
	if err != nil {
		return 0, 0, err
	}
	return yhat - q, yhat + q, nil
}

// CqrInterval returns the conformalized quantile-regression interval per
// Romano-Patterson-Candes 2019.  qLoHat and qHiHat are the predicted lower
// and upper quantiles at the test x; calibrationScores are the symmetric
// conformity scores
//
//	E_i = max(qLoHat_i - y_i, y_i - qHiHat_i)
//
// computed on the calibration fold.  Returns the inflated interval
//
//	[qLoHat - q, qHiHat + q]
//
// where q is the (1 - alpha) finite-sample quantile of the calibration
// scores.  CQR adapts to heteroskedasticity that ordinary split conformal
// (with absolute residuals) would smooth over.
//
// Note: CQR conformity scores can be negative (when the predicted
// quantiles already over-cover the true y); this function does not
// validate non-negativity.
func CqrInterval(qLoHat, qHiHat float64, calibrationScores []float64, alpha float64) (lo, hi float64, err error) {
	if alpha <= 0 || alpha >= 1 || math.IsNaN(alpha) {
		return 0, 0, ErrInvalidAlpha
	}
	if len(calibrationScores) == 0 {
		return 0, 0, ErrEmptyCalibration
	}
	for _, s := range calibrationScores {
		if math.IsNaN(s) {
			return 0, 0, ErrInvalidScore
		}
	}
	n := len(calibrationScores)
	rank := int(math.Ceil((float64(n) + 1.0) * (1.0 - alpha)))
	if rank > n {
		return math.Inf(-1), math.Inf(1), nil
	}
	if rank < 1 {
		rank = 1
	}
	sorted := make([]float64, n)
	copy(sorted, calibrationScores)
	sort.Float64s(sorted)
	q := sorted[rank-1]
	return qLoHat - q, qHiHat + q, nil
}

// CqrConformityScore is the canonical CQR symmetric score
//
//	E = max(qLoHat - y, y - qHiHat)
//
// for a single calibration sample.  Negative when the predicted quantiles
// already strictly contain y; positive when they miss.
func CqrConformityScore(qLoHat, qHiHat, y float64) float64 {
	loErr := qLoHat - y
	hiErr := y - qHiHat
	if loErr > hiErr {
		return loErr
	}
	return hiErr
}

// MarginalCoverageBounds returns the theoretical marginal-coverage
// guarantees of a split-conformal interval at significance level alpha
// computed from n calibration scores:
//
//	lower bound = 1 - alpha
//	upper bound = 1 - alpha + 1/(n+1)
//
// These are tight under exchangeability and continuous score distributions
// (Lei et al 2018 Theorem 1).  Useful for diagnostics and for documenting
// the actual coverage of an existing system.
func MarginalCoverageBounds(n int, alpha float64) (lo, hi float64, err error) {
	if alpha <= 0 || alpha >= 1 || math.IsNaN(alpha) {
		return 0, 0, ErrInvalidAlpha
	}
	if n < 1 {
		return 0, 0, ErrEmptyCalibration
	}
	return 1.0 - alpha, 1.0 - alpha + 1.0/float64(n+1), nil
}

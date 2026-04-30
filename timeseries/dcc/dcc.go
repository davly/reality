package dcc

import (
	"errors"
	"math"
)

// Params holds the DCC alpha-beta plus the unconditional Q-bar matrix
// (k-by-k row-major).
type Params struct {
	Alpha float64
	Beta  float64
	Qbar  []float64 // k*k row-major, symmetric positive-definite
	K     int
}

// EngleDefault returns Engle 2002 Table 5 industry defaults for daily
// equity returns: alpha = 0.05, beta = 0.93.  Caller fills Qbar and K.
func EngleDefault() Params {
	return Params{Alpha: 0.05, Beta: 0.93}
}

// ErrInvalidParameter is returned for non-finite or out-of-range params.
var ErrInvalidParameter = errors.New("dcc: alpha+beta must be in [0,1) and Qbar must be a k*k slice")

// ErrLengthMismatch is returned when Q, z, or buffer dimensions disagree.
var ErrLengthMismatch = errors.New("dcc: input dimensions must match k")

// Validate returns an error if the parameters violate stationarity or have
// inconsistent shape.
func (p Params) Validate() error {
	if math.IsNaN(p.Alpha) || math.IsNaN(p.Beta) {
		return ErrInvalidParameter
	}
	if p.Alpha < 0 || p.Beta < 0 || p.Alpha+p.Beta >= 1 {
		return ErrInvalidParameter
	}
	if p.K < 1 || len(p.Qbar) != p.K*p.K {
		return ErrInvalidParameter
	}
	return nil
}

// SampleQbar computes the unconditional covariance matrix of standardised
// residuals z (a slice of length n vectors of dimension k, stored
// row-major as zSeries with len(zSeries) = n*k).  Writes the k*k row-
// major Qbar into out.
//
// Qbar[i, j] = (1/n) sum_t z_{t, i} * z_{t, j}
func SampleQbar(zSeries []float64, n, k int, out []float64) error {
	if n < 1 || k < 1 {
		return ErrLengthMismatch
	}
	if len(zSeries) != n*k {
		return ErrLengthMismatch
	}
	if len(out) != k*k {
		return ErrLengthMismatch
	}
	for i := range out {
		out[i] = 0
	}
	for t := 0; t < n; t++ {
		base := t * k
		for i := 0; i < k; i++ {
			zi := zSeries[base+i]
			for j := 0; j < k; j++ {
				out[i*k+j] += zi * zSeries[base+j]
			}
		}
	}
	invN := 1.0 / float64(n)
	for i := range out {
		out[i] *= invN
	}
	return nil
}

// Update applies one DCC step:
//
//	Q_new = (1 - alpha - beta) * Qbar + alpha * z z^T + beta * Q
//
// where z is the current vector of standardised residuals (length k) and
// Q is the previous Q matrix (k*k row-major).  The result is written into
// qOut, which must have len = k*k.  Q and qOut may alias.
func (p Params) Update(z, Q, qOut []float64) error {
	if err := p.Validate(); err != nil {
		return err
	}
	k := p.K
	if len(z) != k || len(Q) != k*k || len(qOut) != k*k {
		return ErrLengthMismatch
	}
	weight := 1.0 - p.Alpha - p.Beta
	for i := 0; i < k; i++ {
		zi := z[i]
		for j := 0; j < k; j++ {
			qOut[i*k+j] = weight*p.Qbar[i*k+j] + p.Alpha*zi*z[j] + p.Beta*Q[i*k+j]
		}
	}
	return nil
}

// CorrelationFromQ writes the correlation matrix R = D^{-1/2} Q D^{-1/2}
// into rOut, where D = diag(Q).  Q and rOut are k*k row-major; rOut may
// alias Q.  Diagonal entries of R are exactly 1.
func CorrelationFromQ(Q []float64, k int, rOut []float64) error {
	if len(Q) != k*k || len(rOut) != k*k {
		return ErrLengthMismatch
	}
	inv := make([]float64, k)
	for i := 0; i < k; i++ {
		d := Q[i*k+i]
		if !(d > 0) {
			return errors.New("dcc: diagonal of Q must be positive")
		}
		inv[i] = 1.0 / math.Sqrt(d)
	}
	for i := 0; i < k; i++ {
		for j := 0; j < k; j++ {
			rOut[i*k+j] = Q[i*k+j] * inv[i] * inv[j]
		}
	}
	return nil
}

// FilterSeries applies the DCC update over a full standardised-residual
// series and writes the conditional correlation matrices for every step
// into rSeries (length n*k*k row-major).  Q is initialised at Qbar.
//
// Most callers will use this as the canonical multi-step DCC recursion;
// individual single-step Updates are useful for streaming applications.
func (p Params) FilterSeries(zSeries []float64, n int, rSeries []float64) error {
	if err := p.Validate(); err != nil {
		return err
	}
	k := p.K
	if len(zSeries) != n*k {
		return ErrLengthMismatch
	}
	if len(rSeries) != n*k*k {
		return ErrLengthMismatch
	}
	Q := make([]float64, k*k)
	copy(Q, p.Qbar)
	qNew := make([]float64, k*k)
	for t := 0; t < n; t++ {
		zt := zSeries[t*k : (t+1)*k]
		if err := p.Update(zt, Q, qNew); err != nil {
			return err
		}
		if err := CorrelationFromQ(qNew, k, rSeries[t*k*k:(t+1)*k*k]); err != nil {
			return err
		}
		copy(Q, qNew)
	}
	return nil
}

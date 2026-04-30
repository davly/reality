package garch

import (
	"errors"
	"math"
)

// Model holds the three parameters of a GARCH(1,1) recursion plus the
// unconditional variance used as the initialisation for sigma^2_0.
//
//	sigma^2_t = Omega + Alpha * eps^2_{t-1} + Beta * sigma^2_{t-1}
//
// Stationarity requires Omega > 0, Alpha >= 0, Beta >= 0, Alpha + Beta < 1.
// UncondVar is the implied long-run variance Omega / (1 - Alpha - Beta).
type Model struct {
	Omega     float64
	Alpha     float64
	Beta      float64
	UncondVar float64
}

// ErrInvalidParameter is returned when a parameter is non-finite or
// violates the stationarity constraint.
var ErrInvalidParameter = errors.New("garch: parameters must satisfy omega>0, alpha>=0, beta>=0, alpha+beta<1")

// ErrEmptyData is returned when Fit / Filter receive an empty residual
// series.
var ErrEmptyData = errors.New("garch: residual series must be non-empty")

// Validate returns an error if m violates the GARCH stationarity
// constraints.
func (m Model) Validate() error {
	if math.IsNaN(m.Omega) || math.IsInf(m.Omega, 0) || m.Omega <= 0 {
		return ErrInvalidParameter
	}
	if math.IsNaN(m.Alpha) || m.Alpha < 0 {
		return ErrInvalidParameter
	}
	if math.IsNaN(m.Beta) || m.Beta < 0 {
		return ErrInvalidParameter
	}
	if m.Alpha+m.Beta >= 1.0 {
		return ErrInvalidParameter
	}
	return nil
}

// Filter applies the GARCH(1,1) recursion forward over a residual series
// eps and writes the conditional variance series sigma2 (length len(eps))
// and standardised residuals z (length len(eps)) into the supplied
// buffers.  Initialisation: sigma^2_0 = UncondVar.
//
// Returns ErrInvalidParameter if m fails Validate; ErrEmptyData if eps is
// empty.  sigma2 and z must each have len >= len(eps).
func (m Model) Filter(eps, sigma2, z []float64) error {
	if err := m.Validate(); err != nil {
		return err
	}
	n := len(eps)
	if n == 0 {
		return ErrEmptyData
	}
	if len(sigma2) < n || len(z) < n {
		return errors.New("garch: sigma2 and z buffers must have len >= len(eps)")
	}
	prevS2 := m.UncondVar
	prevEps := 0.0
	if prevS2 <= 0 || math.IsNaN(prevS2) || math.IsInf(prevS2, 0) {
		// Caller did not set UncondVar; fall back to the implied value.
		prevS2 = m.Omega / (1.0 - m.Alpha - m.Beta)
	}
	for i, e := range eps {
		s2 := m.Omega + m.Alpha*prevEps*prevEps + m.Beta*prevS2
		sigma2[i] = s2
		z[i] = e / math.Sqrt(s2)
		prevS2 = s2
		prevEps = e
	}
	return nil
}

// LogLikelihood returns the Gaussian log-likelihood of the residual series
// eps under the model:
//
//	log L = -0.5 sum_t [ log(2 pi) + log(sigma^2_t) + eps^2_t / sigma^2_t ]
//
// Used as the objective function during calibration.
func (m Model) LogLikelihood(eps []float64) (float64, error) {
	n := len(eps)
	if n == 0 {
		return 0, ErrEmptyData
	}
	if err := m.Validate(); err != nil {
		return 0, err
	}
	sigma2 := make([]float64, n)
	z := make([]float64, n)
	if err := m.Filter(eps, sigma2, z); err != nil {
		return 0, err
	}
	const log2pi = 1.8378770664093454835606594728112
	var ll float64
	for i, e := range eps {
		ll -= 0.5 * (log2pi + math.Log(sigma2[i]) + e*e/sigma2[i])
	}
	return ll, nil
}

// ForecastVariance returns the h-step-ahead conditional variance forecast
// given the most recent observed conditional variance sigma2_t and the
// most recent squared shock eps2_t.  For h = 1 the formula reduces to
// the GARCH recursion; for h > 1 it converges geometrically toward the
// unconditional variance with rate (Alpha + Beta).
//
// Reference: Bollerslev 1986 §4.
func (m Model) ForecastVariance(eps2T, sigma2T float64, h int) ([]float64, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	if h < 1 {
		return nil, errors.New("garch: h must be >= 1")
	}
	out := make([]float64, h)
	v := m.UncondVar
	if v <= 0 {
		v = m.Omega / (1.0 - m.Alpha - m.Beta)
	}
	persistence := m.Alpha + m.Beta
	// Step 1: standard recursion on the latest realised shock.
	out[0] = m.Omega + m.Alpha*eps2T + m.Beta*sigma2T
	// Step h>1: E[eps^2_{t+h-1} | F_t] = sigma^2_{t+h-1} so the recursion
	// collapses to omega + (alpha + beta) * sigma^2_{prev}.  Equivalent
	// closed form: v + (alpha+beta)^{h-1} * (sigma^2_{t+1} - v).
	for k := 1; k < h; k++ {
		out[k] = v + persistence*(out[k-1]-v)
	}
	return out, nil
}

// Simulate generates a path of length len(shocks) under the model given
// pre-drawn standard-normal shocks z.  shocks[t] is the standard-normal
// innovation at time t.  Returns the synthesised residual series eps
// (eps_t = sigma_t * z_t) and the conditional variance series sigma2.
//
// Initialisation: sigma^2_0 = UncondVar; eps^2_{-1} = UncondVar.  Both
// output buffers must have len >= len(shocks).
func (m Model) Simulate(shocks, eps, sigma2 []float64) error {
	if err := m.Validate(); err != nil {
		return err
	}
	n := len(shocks)
	if n == 0 {
		return ErrEmptyData
	}
	if len(eps) < n || len(sigma2) < n {
		return errors.New("garch: eps and sigma2 buffers must have len >= len(shocks)")
	}
	prevEps2 := m.UncondVar
	prevS2 := m.UncondVar
	if prevS2 <= 0 {
		prevS2 = m.Omega / (1.0 - m.Alpha - m.Beta)
		prevEps2 = prevS2
	}
	for i, z := range shocks {
		s2 := m.Omega + m.Alpha*prevEps2 + m.Beta*prevS2
		s := math.Sqrt(s2)
		e := s * z
		sigma2[i] = s2
		eps[i] = e
		prevS2 = s2
		prevEps2 = e * e
	}
	return nil
}

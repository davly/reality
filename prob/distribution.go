package prob

import "math"

// ---------------------------------------------------------------------------
// Distribution interface — polymorphic access to PDF and CDF.
//
// Insight from Proof (Haskell) and RubberDuck (C#): every distribution in
// the ecosystem exposes the same two operations (PDF and CDF), but the
// canonical Go code had them as free functions with inconsistent signatures.
// The Distribution interface unifies them behind a single contract.
//
// This is a Type 2 innovation from Wave 1 cross-pollination: Haskell's
// Distribution typeclass and C#'s IDistribution interface both converged
// on this shape independently.
//
// Consumers:
//   - Oracle:     Bayesian posterior evaluation (Beta, Normal)
//   - Echo:       KL divergence computation (Normal)
//   - RubberDuck: financial risk distributions (Normal, Exponential)
//   - Sentinel:   extreme value analysis (Exponential)
// ---------------------------------------------------------------------------

// Distribution is the common interface for probability distributions.
// Every distribution must provide a probability density (or mass) function
// and a cumulative distribution function.
type Distribution interface {
	// PDF returns the probability density function evaluated at x.
	// For discrete distributions, this is the probability mass function.
	PDF(x float64) float64

	// CDF returns the cumulative distribution function evaluated at x.
	// CDF(x) = P(X <= x).
	CDF(x float64) float64
}

// ---------------------------------------------------------------------------
// BetaDist wraps the Beta distribution as a Distribution.
// ---------------------------------------------------------------------------

// BetaDist represents a Beta(alpha, beta) distribution.
// Alpha and Beta must be > 0; invalid parameters cause PDF/CDF to return NaN.
type BetaDist struct {
	Alpha float64
	Beta  float64
}

// NewBetaDist creates a BetaDist. Returns nil if alpha <= 0 or beta <= 0.
func NewBetaDist(alpha, beta float64) *BetaDist {
	if alpha <= 0 || beta <= 0 {
		return nil
	}
	return &BetaDist{Alpha: alpha, Beta: beta}
}

// PDF returns the Beta probability density function at x.
func (d *BetaDist) PDF(x float64) float64 {
	return BetaPDF(x, d.Alpha, d.Beta)
}

// CDF returns the Beta cumulative distribution function at x.
func (d *BetaDist) CDF(x float64) float64 {
	return BetaCDF(x, d.Alpha, d.Beta)
}

// ---------------------------------------------------------------------------
// NormalDist wraps the Normal distribution as a Distribution.
// ---------------------------------------------------------------------------

// NormalDist represents a Normal(mu, sigma) distribution.
// Sigma must be > 0; invalid parameters cause PDF/CDF to return NaN.
type NormalDist struct {
	Mu    float64
	Sigma float64
}

// NewNormalDist creates a NormalDist. Returns nil if sigma <= 0.
func NewNormalDist(mu, sigma float64) *NormalDist {
	if sigma <= 0 {
		return nil
	}
	return &NormalDist{Mu: mu, Sigma: sigma}
}

// PDF returns the Normal probability density function at x.
func (d *NormalDist) PDF(x float64) float64 {
	return NormalPDF(x, d.Mu, d.Sigma)
}

// CDF returns the Normal cumulative distribution function at x.
func (d *NormalDist) CDF(x float64) float64 {
	return NormalCDF(x, d.Mu, d.Sigma)
}

// ---------------------------------------------------------------------------
// ExponentialDist wraps the Exponential distribution as a Distribution.
// ---------------------------------------------------------------------------

// ExponentialDist represents an Exponential(lambda) distribution.
// Lambda must be > 0; invalid parameters cause PDF/CDF to return NaN.
type ExponentialDist struct {
	Lambda float64
}

// NewExponentialDist creates an ExponentialDist. Returns nil if lambda <= 0.
func NewExponentialDist(lambda float64) *ExponentialDist {
	if lambda <= 0 {
		return nil
	}
	return &ExponentialDist{Lambda: lambda}
}

// PDF returns the Exponential probability density function at x.
func (d *ExponentialDist) PDF(x float64) float64 {
	return ExponentialPDF(x, d.Lambda)
}

// CDF returns the Exponential cumulative distribution function at x.
func (d *ExponentialDist) CDF(x float64) float64 {
	return ExponentialCDF(x, d.Lambda)
}

// ---------------------------------------------------------------------------
// UniformDist wraps the Uniform distribution as a Distribution.
// ---------------------------------------------------------------------------

// UniformDist represents a Uniform(a, b) distribution on [a, b].
// Requires a < b; invalid parameters cause PDF/CDF to return NaN.
type UniformDist struct {
	A float64
	B float64
}

// NewUniformDist creates a UniformDist. Returns nil if a >= b.
func NewUniformDist(a, b float64) *UniformDist {
	if a >= b {
		return nil
	}
	return &UniformDist{A: a, B: b}
}

// PDF returns the Uniform probability density function at x.
func (d *UniformDist) PDF(x float64) float64 {
	return UniformPDF(x, d.A, d.B)
}

// CDF returns the Uniform cumulative distribution function at x.
func (d *UniformDist) CDF(x float64) float64 {
	return UniformCDF(x, d.A, d.B)
}

// ---------------------------------------------------------------------------
// KLDivergence computes the Kullback-Leibler divergence between two
// distributions using numerical integration (trapezoidal rule).
//
// This is a Distribution-interface-aware helper that works with any pair
// of distributions. For analytical KL divergence (e.g., Normal-to-Normal),
// use the dedicated functions in this package.
//
// Formula: KL(P || Q) = integral P(x) * log(P(x)/Q(x)) dx
// Valid range: P and Q must have overlapping support
// Returns: +Inf if Q(x) = 0 where P(x) > 0 (absolute continuity violated)
// Precision: limited by trapezoidal step count (nSteps)
// Reference: Kullback & Leibler (1951)
// ---------------------------------------------------------------------------

// KLDivergenceNumerical estimates KL(p || q) over [lo, hi] using the
// trapezoidal rule with nSteps steps.
func KLDivergenceNumerical(p, q Distribution, lo, hi float64, nSteps int) float64 {
	if nSteps <= 0 {
		nSteps = 1000
	}
	dx := (hi - lo) / float64(nSteps)
	sum := 0.0

	for i := 0; i <= nSteps; i++ {
		x := lo + float64(i)*dx
		px := p.PDF(x)
		qx := q.PDF(x)

		if px <= 0 {
			continue
		}
		if qx <= 0 {
			return math.Inf(1)
		}

		weight := 1.0
		if i == 0 || i == nSteps {
			weight = 0.5
		}
		sum += weight * px * math.Log(px/qx)
	}

	return sum * dx
}

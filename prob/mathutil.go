package prob

import "math"

// ---------------------------------------------------------------------------
// Mathematical utility functions for probability distributions.
//
// These are supporting functions needed by the distribution implementations
// in distributions.go. They wrap stdlib where possible and provide custom
// implementations only where the stdlib has no equivalent.
// ---------------------------------------------------------------------------

// LogGamma returns the natural logarithm of the absolute value of Gamma(x).
//
// This is a direct wrapper around math.Lgamma from the Go standard library.
// We expose it here so that the prob package has a single, documented entry
// point for log-gamma computation.
//
// Formula: ln(|Gamma(x)|)
// Valid range: x != 0, -1, -2, ... (poles of Gamma)
// Precision: ~15 significant digits (float64)
// Reference: Lanczos approximation (via Go stdlib)
func LogGamma(x float64) float64 {
	v, _ := math.Lgamma(x)
	return v
}

// Erfc returns the complementary error function: erfc(x) = 1 - erf(x).
//
// This is a direct wrapper around math.Erfc from the Go standard library.
//
// Formula: (2/sqrt(pi)) * integral from x to inf of exp(-t^2) dt
// Valid range: all float64
// Precision: ~15 significant digits (float64)
// Reference: Abramowitz & Stegun, formula 7.1.2
func Erfc(x float64) float64 {
	return math.Erfc(x)
}

// RegularizedBetaInc computes the regularized incomplete beta function
// I_x(a, b), defined as:
//
//	I_x(a,b) = B(x; a, b) / B(a, b)
//
// where B(x; a, b) is the incomplete beta function and B(a, b) is the
// complete beta function.
//
// The implementation uses Lentz's continued fraction method for the
// incomplete beta function, with the symmetry relation
// I_x(a, b) = 1 - I_{1-x}(b, a) to ensure convergence.
//
// Formula: I_x(a,b) via continued fraction (DLMF 8.17.22)
// Valid range: x in [0, 1], a > 0, b > 0
// Precision: ~1e-14 absolute for typical inputs
// Failure mode: returns NaN if a <= 0, b <= 0, or x outside [0, 1]
// Reference: Lentz, W.J. (1976) "Generating Bessel functions in Mie
// scattering calculations using continued fractions"; Press et al.,
// Numerical Recipes, 3rd ed., section 6.4
func RegularizedBetaInc(x, a, b float64) float64 {
	if x < 0 || x > 1 || a <= 0 || b <= 0 {
		return math.NaN()
	}
	if x == 0 {
		return 0
	}
	if x == 1 {
		return 1
	}

	// Use the symmetry relation for faster convergence:
	// When x > (a+1)/(a+b+2), evaluate 1 - I_{1-x}(b, a) instead.
	if x > (a+1)/(a+b+2) {
		return 1.0 - RegularizedBetaInc(1-x, b, a)
	}

	// Log of the prefactor: x^a * (1-x)^b / (a * B(a,b))
	lnPrefactor := a*math.Log(x) + b*math.Log(1-x) - math.Log(a) -
		(LogGamma(a) + LogGamma(b) - LogGamma(a+b))

	// Evaluate continued fraction using Lentz's method.
	return math.Exp(lnPrefactor) * betaCF(x, a, b)
}

// betaCF evaluates the continued fraction for the incomplete beta function
// using the modified Lentz algorithm. The continued fraction is:
//
//	1 / (1 + d1/(1 + d2/(1 + ...)))
//
// where the coefficients d_m are defined as in Numerical Recipes eq. 6.4.5.
//
// maxIter limits iterations to prevent infinite loops. The tiny constant
// prevents division by zero in the Lentz algorithm.
func betaCF(x, a, b float64) float64 {
	const maxIter = 200
	const eps = 1e-14
	const tiny = 1e-30

	// Lentz's method: f = h = 1, then iterate.
	f := 1.0
	c := 1.0
	d := 1.0 - (a+b)*x/(a+1)
	if math.Abs(d) < tiny {
		d = tiny
	}
	d = 1.0 / d
	f = d

	for m := 1; m <= maxIter; m++ {
		mf := float64(m)

		// Even step: d_{2m} coefficient.
		num := mf * (b - mf) * x / ((a + 2*mf - 1) * (a + 2*mf))
		d = 1.0 + num*d
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = 1.0 + num/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1.0 / d
		f *= c * d

		// Odd step: d_{2m+1} coefficient.
		num = -(a + mf) * (a + b + mf) * x / ((a + 2*mf) * (a + 2*mf + 1))
		d = 1.0 + num*d
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = 1.0 + num/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1.0 / d
		delta := c * d
		f *= delta

		if math.Abs(delta-1.0) < eps {
			return f
		}
	}

	// Did not converge — return best estimate.
	return f
}

// studentTCDF computes the CDF of the Student's t-distribution with df
// degrees of freedom, evaluated at t. Used by hypothesis tests for p-value
// computation.
//
// Formula: CDF(t; df) = I_{x}(df/2, 1/2) where x = df/(df + t^2),
//
//	using the regularized incomplete beta function.
//	For t >= 0: CDF = 1 - 0.5 * I_x(df/2, 1/2)
//	For t < 0: CDF = 0.5 * I_x(df/2, 1/2)
//
// Valid range: any t, df > 0
// Precision: ~1e-12 for moderate df; degrades slightly for df < 1
// Reference: Abramowitz & Stegun, formula 26.5.27
func studentTCDF(t float64, df float64) float64 {
	if df <= 0 {
		return math.NaN()
	}
	x := df / (df + t*t)
	iBeta := RegularizedBetaInc(x, df/2.0, 0.5)
	if t >= 0 {
		return 1.0 - 0.5*iBeta
	}
	return 0.5 * iBeta
}

// chiSquaredCDF computes the CDF of the chi-squared distribution with df
// degrees of freedom, evaluated at x. Used by the chi-squared goodness-of-fit
// test for p-value computation.
//
// Formula: CDF(x; df) = regularizedGammaLower(df/2, x/2)
//
//	which equals I_{x/(x+df)}(df/2, ...) via the beta function relation,
//	or directly via the incomplete gamma series.
//
// This implementation uses the series expansion of the lower regularized
// gamma function P(a, x) = gamma(a, x) / Gamma(a).
//
// Valid range: x >= 0, df > 0
// Precision: ~1e-12 for typical inputs
// Reference: Abramowitz & Stegun, formula 6.5.29 (series expansion)
func chiSquaredCDF(x float64, df float64) float64 {
	if x <= 0 || df <= 0 {
		return 0
	}
	return regularizedGammaLowerSeries(df/2.0, x/2.0)
}

// regularizedGammaLowerSeries computes the lower regularized incomplete
// gamma function P(a, x) = gamma(a, x) / Gamma(a) using the series:
//
//	P(a, x) = e^{-x} * x^a * sum_{n=0}^{inf} x^n / (a * (a+1) * ... * (a+n))
//
// This series converges for all x when a > 0.
func regularizedGammaLowerSeries(a, x float64) float64 {
	if x < 0 {
		return 0
	}
	if x == 0 {
		return 0
	}

	const maxIter = 200
	const eps = 1e-14

	// Compute log of the prefactor: -x + a*ln(x) - lgamma(a)
	lnPrefix := -x + a*math.Log(x) - LogGamma(a)

	sum := 1.0 / a
	term := 1.0 / a
	for n := 1; n <= maxIter; n++ {
		term *= x / (a + float64(n))
		sum += term
		if math.Abs(term) < eps*math.Abs(sum) {
			break
		}
	}

	return math.Exp(lnPrefix) * sum
}

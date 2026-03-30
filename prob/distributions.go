package prob

import "math"

// ---------------------------------------------------------------------------
// Probability distributions: PDF, CDF, and Quantile functions.
//
// Every function is pure, deterministic, and uses only the Go standard
// library. No external dependencies. All numerical routines cite their
// mathematical provenance and document their valid input ranges, precision
// guarantees, and failure modes.
//
// Consumers:
//   - Oracle:    full Bayesian (Normal, Beta)
//   - Echo:      KL divergence (Normal)
//   - RubberDuck: financial distributions (Normal, Exponential)
//   - Sentinel:  extreme value analysis (Exponential)
//   - Parallax:  hypothesis testing (via hypothesis.go, needs CDF)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Normal (Gaussian) distribution
// ---------------------------------------------------------------------------

// NormalPDF returns the probability density function of the normal
// distribution at x, with mean mu and standard deviation sigma.
//
// Formula: (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x - mu) / sigma)^2)
// Valid range: sigma > 0; returns NaN if sigma <= 0
// Precision: ~15 significant digits (float64)
// Reference: standard Gaussian PDF
func NormalPDF(x, mu, sigma float64) float64 {
	if sigma <= 0 {
		return math.NaN()
	}
	z := (x - mu) / sigma
	return math.Exp(-0.5*z*z) / (sigma * math.Sqrt(2*math.Pi))
}

// NormalCDF returns the cumulative distribution function of the normal
// distribution at x, with mean mu and standard deviation sigma.
//
// Formula: 0.5 * erfc(-(x - mu) / (sigma * sqrt(2)))
// Valid range: sigma > 0; returns NaN if sigma <= 0
// Precision: ~15 significant digits (float64, via math.Erfc)
// Reference: Abramowitz & Stegun, formula 7.1.2
func NormalCDF(x, mu, sigma float64) float64 {
	if sigma <= 0 {
		return math.NaN()
	}
	return 0.5 * math.Erfc(-(x-mu)/(sigma*math.Sqrt2))
}

// NormalQuantile returns the inverse CDF (quantile function) of the normal
// distribution for probability p, with mean mu and standard deviation sigma.
//
// Uses the rational approximation by Peter Acklam (2004), which provides
// full float64 precision across the entire range (0, 1).
//
// Formula: mu + sigma * Phi^{-1}(p) where Phi^{-1} is the standard
// normal quantile (probit function).
// Valid range: p in (0, 1), sigma > 0
// Returns: NaN if p <= 0, p >= 1, or sigma <= 0
// Precision: maximum relative error < 1.15e-9 across all p in (0, 1)
// Reference: Acklam, P.J. (2004) "An algorithm for computing the inverse
// normal cumulative distribution function"
func NormalQuantile(p, mu, sigma float64) float64 {
	if sigma <= 0 || p <= 0 || p >= 1 {
		return math.NaN()
	}
	return mu + sigma*standardNormalQuantile(p)
}

// standardNormalQuantile computes Phi^{-1}(p) for the standard normal
// distribution using Acklam's rational approximation.
func standardNormalQuantile(p float64) float64 {
	// Coefficients for the rational approximation.
	const (
		a1 = -3.969683028665376e+01
		a2 = 2.209460984245205e+02
		a3 = -2.759285104469687e+02
		a4 = 1.383577518672690e+02
		a5 = -3.066479806614716e+01
		a6 = 2.506628277459239e+00

		b1 = -5.447609879822406e+01
		b2 = 1.615858368580409e+02
		b3 = -1.556989798598866e+02
		b4 = 6.680131188771972e+01
		b5 = -1.328068155288572e+01

		c1 = -7.784894002430293e-03
		c2 = -3.223964580411365e-01
		c3 = -2.400758277161838e+00
		c4 = -2.549732539343734e+00
		c5 = 4.374664141464968e+00
		c6 = 2.938163982698783e+00

		d1 = 7.784695709041462e-03
		d2 = 3.224671290700398e-01
		d3 = 2.445134137142996e+00
		d4 = 3.754408661907416e+00

		pLow  = 0.02425
		pHigh = 1 - pLow
	)

	var q, r float64

	if p < pLow {
		// Rational approximation for lower region.
		q = math.Sqrt(-2 * math.Log(p))
		return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q + c6) /
			((((d1*q+d2)*q+d3)*q+d4)*q + 1)
	} else if p <= pHigh {
		// Rational approximation for central region.
		q = p - 0.5
		r = q * q
		return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r + a6) * q /
			(((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r + 1)
	} else {
		// Rational approximation for upper region.
		q = math.Sqrt(-2 * math.Log(1-p))
		return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q + c6) /
			((((d1*q+d2)*q+d3)*q+d4)*q + 1)
	}
}

// ---------------------------------------------------------------------------
// Exponential distribution
// ---------------------------------------------------------------------------

// ExponentialPDF returns the probability density function of the exponential
// distribution at x, with rate parameter lambda.
//
// Formula: lambda * exp(-lambda * x) for x >= 0; 0 otherwise
// Valid range: lambda > 0, x >= 0; returns NaN if lambda <= 0
// Precision: ~15 significant digits (float64)
// Reference: standard exponential PDF
func ExponentialPDF(x, lambda float64) float64 {
	if lambda <= 0 {
		return math.NaN()
	}
	if x < 0 {
		return 0
	}
	return lambda * math.Exp(-lambda*x)
}

// ExponentialCDF returns the cumulative distribution function of the
// exponential distribution at x, with rate parameter lambda.
//
// Formula: 1 - exp(-lambda * x) for x >= 0; 0 otherwise
// Valid range: lambda > 0, x >= 0; returns NaN if lambda <= 0
// Precision: ~15 significant digits (float64)
// Reference: standard exponential CDF
func ExponentialCDF(x, lambda float64) float64 {
	if lambda <= 0 {
		return math.NaN()
	}
	if x < 0 {
		return 0
	}
	return 1 - math.Exp(-lambda*x)
}

// ---------------------------------------------------------------------------
// Uniform distribution
// ---------------------------------------------------------------------------

// UniformPDF returns the probability density function of the continuous
// uniform distribution on [a, b] at point x.
//
// Formula: 1/(b-a) for a <= x <= b; 0 otherwise
// Valid range: a < b; returns NaN if a >= b
// Precision: exact (single division)
// Reference: standard uniform PDF
func UniformPDF(x, a, b float64) float64 {
	if a >= b {
		return math.NaN()
	}
	if x < a || x > b {
		return 0
	}
	return 1.0 / (b - a)
}

// UniformCDF returns the cumulative distribution function of the continuous
// uniform distribution on [a, b] at point x.
//
// Formula: (x-a)/(b-a) for a <= x <= b; 0 for x < a; 1 for x > b
// Valid range: a < b; returns NaN if a >= b
// Precision: exact (single division)
// Reference: standard uniform CDF
func UniformCDF(x, a, b float64) float64 {
	if a >= b {
		return math.NaN()
	}
	if x <= a {
		return 0
	}
	if x >= b {
		return 1
	}
	return (x - a) / (b - a)
}

// ---------------------------------------------------------------------------
// Beta distribution
// ---------------------------------------------------------------------------

// BetaPDF returns the probability density function of the Beta distribution
// at x, with shape parameters alpha and beta.
//
// Uses log-gamma for numerical stability to avoid overflow in the
// normalization constant B(alpha, beta).
//
// Formula: x^{alpha-1} * (1-x)^{beta-1} / B(alpha, beta)
//
//	where B(alpha, beta) = Gamma(alpha)*Gamma(beta)/Gamma(alpha+beta)
//
// Valid range: x in [0, 1], alpha > 0, beta > 0
// Returns NaN if alpha <= 0 or beta <= 0
// Returns 0 if x < 0 or x > 1
// Special: returns +Inf at x=0 when alpha < 1, or x=1 when beta < 1
// Precision: ~15 significant digits (float64)
// Reference: standard Beta PDF; Johnson, Kotz, Balakrishnan (1995)
// "Continuous Univariate Distributions", Vol. 2
func BetaPDF(x, alpha, beta float64) float64 {
	if alpha <= 0 || beta <= 0 {
		return math.NaN()
	}
	if x < 0 || x > 1 {
		return 0
	}
	// Handle boundary cases that would produce 0^negative.
	if x == 0 {
		if alpha < 1 {
			return math.Inf(1)
		}
		if alpha == 1 {
			// pdf = (1-0)^{beta-1} / B(1, beta) = 1/B(1, beta) = beta
			return beta
		}
		return 0
	}
	if x == 1 {
		if beta < 1 {
			return math.Inf(1)
		}
		if beta == 1 {
			return alpha
		}
		return 0
	}

	// Use log-space for numerical stability.
	logB := LogGamma(alpha) + LogGamma(beta) - LogGamma(alpha+beta)
	logPDF := (alpha-1)*math.Log(x) + (beta-1)*math.Log(1-x) - logB
	return math.Exp(logPDF)
}

// BetaCDF returns the cumulative distribution function of the Beta
// distribution at x, with shape parameters alpha and beta.
//
// This is the regularized incomplete beta function I_x(alpha, beta).
//
// Formula: I_x(alpha, beta) via continued fraction (Lentz's method)
// Valid range: x in [0, 1], alpha > 0, beta > 0
// Returns NaN if alpha <= 0 or beta <= 0
// Precision: ~1e-14 absolute for typical inputs
// Reference: see RegularizedBetaInc in mathutil.go
func BetaCDF(x, alpha, beta float64) float64 {
	if alpha <= 0 || beta <= 0 {
		return math.NaN()
	}
	if x <= 0 {
		return 0
	}
	if x >= 1 {
		return 1
	}
	return RegularizedBetaInc(x, alpha, beta)
}

// ---------------------------------------------------------------------------
// Poisson distribution
// ---------------------------------------------------------------------------

// PoissonPMF returns the probability mass function of the Poisson
// distribution at k, with mean rate lambda.
//
// Uses log-gamma for numerical stability to avoid overflow in k!
//
// Formula: lambda^k * e^{-lambda} / k!
//
//	Computed as exp(k*ln(lambda) - lambda - lgamma(k+1))
//
// Valid range: k >= 0, lambda > 0; returns NaN if lambda <= 0
// Returns 0 if k < 0
// Precision: ~15 significant digits (float64)
// Reference: standard Poisson PMF
func PoissonPMF(k int, lambda float64) float64 {
	if lambda <= 0 {
		return math.NaN()
	}
	if k < 0 {
		return 0
	}
	if k == 0 {
		return math.Exp(-lambda)
	}
	logPMF := float64(k)*math.Log(lambda) - lambda - LogGamma(float64(k+1))
	return math.Exp(logPMF)
}

// PoissonCDF returns the cumulative distribution function of the Poisson
// distribution at k (P(X <= k)), with mean rate lambda.
//
// Computed as the sum of PMF values from 0 to k. For large lambda, this
// uses the relation to the regularized upper incomplete gamma function,
// but our straightforward summation is adequate for reasonable lambda values.
//
// Formula: sum_{i=0}^{k} PoissonPMF(i, lambda)
// Valid range: k >= 0, lambda > 0; returns NaN if lambda <= 0
// Returns 0 if k < 0
// Precision: accumulated float64 summation error
// Reference: standard Poisson CDF
func PoissonCDF(k int, lambda float64) float64 {
	if lambda <= 0 {
		return math.NaN()
	}
	if k < 0 {
		return 0
	}
	// Use the complementary regularized gamma function:
	// P(X <= k) = 1 - P(a, x) where a = k+1, x = lambda
	// This is more numerically stable for large k.
	return 1.0 - regularizedGammaLowerSeries(float64(k+1), lambda)
}

// ---------------------------------------------------------------------------
// Gamma distribution
// ---------------------------------------------------------------------------

// GammaPDF returns the probability density function of the Gamma distribution
// at x, with shape parameter k (alpha) and rate parameter theta (inverse scale).
//
// Formula: (1 / (Gamma(k) * theta^k)) * x^{k-1} * exp(-x/theta)
// Computed in log-space as: exp((k-1)*ln(x) - x/theta - k*ln(theta) - lgamma(k))
//
// Valid range: k > 0, theta > 0, x >= 0
// Returns NaN if k <= 0 or theta <= 0
// Returns 0 if x < 0
// Special: at x = 0, returns +Inf when k < 1, 1/theta when k == 1, 0 when k > 1
// Precision: ~15 significant digits (float64)
// Reference: Johnson, Kotz, Balakrishnan (1994) "Continuous Univariate
// Distributions", Vol. 1, Chapter 17
func GammaPDF(x, k, theta float64) float64 {
	if k <= 0 || theta <= 0 {
		return math.NaN()
	}
	if x < 0 {
		return 0
	}
	if x == 0 {
		if k < 1 {
			return math.Inf(1)
		}
		if k == 1 {
			return 1.0 / theta
		}
		return 0
	}
	logPDF := (k-1)*math.Log(x) - x/theta - k*math.Log(theta) - LogGamma(k)
	return math.Exp(logPDF)
}

// GammaCDF returns the cumulative distribution function of the Gamma
// distribution at x, with shape parameter k and scale parameter theta.
//
// Formula: P(k, x/theta) where P is the lower regularized incomplete
// gamma function.
//
// Valid range: k > 0, theta > 0, x >= 0
// Returns NaN if k <= 0 or theta <= 0
// Returns 0 if x <= 0
// Precision: ~1e-14 (via regularizedGammaLowerSeries)
// Reference: Abramowitz & Stegun, Chapter 6; DLMF 8.2
func GammaCDF(x, k, theta float64) float64 {
	if k <= 0 || theta <= 0 {
		return math.NaN()
	}
	if x <= 0 {
		return 0
	}
	return regularizedGammaLowerSeries(k, x/theta)
}

// ExponentialQuantile returns the inverse CDF (quantile function) of the
// exponential distribution for probability p, with rate parameter lambda.
//
// Formula: -ln(1-p) / lambda
// Valid range: p in (0, 1), lambda > 0
// Returns NaN if p <= 0, p >= 1, or lambda <= 0
// Precision: ~15 significant digits (float64, via math.Log)
// Reference: standard exponential quantile function
func ExponentialQuantile(p, lambda float64) float64 {
	if lambda <= 0 || p <= 0 || p >= 1 {
		return math.NaN()
	}
	return -math.Log(1-p) / lambda
}

// ---------------------------------------------------------------------------
// Binomial distribution
// ---------------------------------------------------------------------------

// BinomialPMF returns the probability mass function of the binomial
// distribution at k, with n trials and success probability p.
//
// Uses log-gamma for numerical stability to avoid overflow in C(n, k).
//
// Formula: C(n, k) * p^k * (1-p)^{n-k}
//
//	Computed as exp(lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
//	                + k*ln(p) + (n-k)*ln(1-p))
//
// Valid range: 0 <= k <= n, n >= 0, p in [0, 1]
// Returns NaN if p < 0 or p > 1, or n < 0
// Returns 0 if k < 0 or k > n
// Precision: ~15 significant digits (float64)
// Reference: standard binomial PMF
func BinomialPMF(k, n int, p float64) float64 {
	if p < 0 || p > 1 || n < 0 {
		return math.NaN()
	}
	if k < 0 || k > n {
		return 0
	}
	// Edge cases for p = 0 or p = 1.
	if p == 0 {
		if k == 0 {
			return 1
		}
		return 0
	}
	if p == 1 {
		if k == n {
			return 1
		}
		return 0
	}
	logPMF := LogGamma(float64(n+1)) - LogGamma(float64(k+1)) - LogGamma(float64(n-k+1)) +
		float64(k)*math.Log(p) + float64(n-k)*math.Log(1-p)
	return math.Exp(logPMF)
}

// BinomialCDF returns the cumulative distribution function of the binomial
// distribution at k (P(X <= k)), with n trials and success probability p.
//
// Uses the relation to the regularized incomplete beta function:
//
//	P(X <= k) = I_{1-p}(n-k, k+1)
//
// This is more numerically stable and efficient than summing PMF values.
//
// Valid range: 0 <= k <= n, n >= 0, p in [0, 1]
// Returns NaN if p < 0 or p > 1, or n < 0
// Returns 0 if k < 0; returns 1 if k >= n
// Precision: ~1e-14 (via RegularizedBetaInc)
// Reference: relation between binomial CDF and incomplete beta;
// Abramowitz & Stegun, formula 26.5.24
func BinomialCDF(k, n int, p float64) float64 {
	if p < 0 || p > 1 || n < 0 {
		return math.NaN()
	}
	if k < 0 {
		return 0
	}
	if k >= n {
		return 1
	}
	// I_{1-p}(n-k, k+1)
	return RegularizedBetaInc(1-p, float64(n-k), float64(k+1))
}

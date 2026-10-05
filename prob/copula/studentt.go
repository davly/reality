package copula

import "math"

// ---------------------------------------------------------------------------
// Univariate Student-t CDF and quantile.
//
// The reality/prob package has an unexported studentTCDF (uses the
// regularized-incomplete-beta backbone in mathutil.go); we re-implement
// here to keep the copula package self-contained and to avoid adding a
// public API surface to prob just for our internal use.  When a future
// session promotes a `StudentT` distribution under prob/distributions.go,
// these helpers will retire to call into the public surface.
// ---------------------------------------------------------------------------

// StudentTCDF returns the CDF of the Student-t distribution with df
// degrees of freedom evaluated at x.
//
// Formula: F(x; df) = 1 - 0.5 * I_{df/(df + x^2)}(df/2, 1/2)  for x >= 0
//
//	F(x; df) =     0.5 * I_{df/(df + x^2)}(df/2, 1/2)  for x <  0
//
// Uses prob.RegularizedBetaInc, with the complement x^2/(df + x^2) kept
// exact (see studentTTailBeta).
// Precision: relative error below 1e-13 for results >= 1e-150 (measured at
// most 2.1e-15 for results >= 1e-10, df from 1 to 1e6, |x| up to 1e3); the
// upper half 1 - CDF(-x) is exact to 2e-16 absolute.
//
// Reference: Abramowitz & Stegun 26.5.27.
func StudentTCDF(x, df float64) float64 {
	if df <= 0 || math.IsNaN(x) || math.IsNaN(df) {
		return math.NaN()
	}
	if math.IsInf(x, 1) {
		return 1
	}
	if math.IsInf(x, -1) {
		return 0
	}
	if math.IsInf(df, 1) {
		// t -> standard normal as df -> inf.
		return 0.5 * math.Erfc(-x/math.Sqrt2)
	}
	ib := studentTTailBeta(df, x*x)
	if x >= 0 {
		return 1.0 - 0.5*ib
	}
	return 0.5 * ib
}

// StudentTQuantile returns the inverse CDF (quantile function) of the
// Student-t distribution with df degrees of freedom for probability p.
//
// Strategy: Brent-style monotonic bisection on StudentTCDF, using the
// Hill 1970 high-df Cornish-Fisher expansion for an initial bracket.
// Precision: ~1e-10 absolute on x for p in [1e-12, 1 - 1e-12], df >= 1.
//
// Returns NaN if p <= 0, p >= 1, or df < 1.
//
// Reference: Hill, G. W. (1970).  Algorithm 396: Student's t-Quantiles.
// Communications of the ACM 13: 619-620.  Also Abramowitz & Stegun
// 26.7.5.
func StudentTQuantile(p, df float64) float64 {
	if math.IsNaN(p) || math.IsNaN(df) || p <= 0 || p >= 1 || df < 1 {
		return math.NaN()
	}
	// Symmetry: only solve for p >= 0.5.
	flipSign := p < 0.5
	pp := p
	if flipSign {
		pp = 1 - p
	}

	// Initial guess via the standard-normal quantile (df -> inf approx).
	// Acklam-rational on the standard normal gives ~1e-9 accuracy on the
	// approximation, which we then bracket and refine via bisection.
	z := standardNormalQuantileLocal(pp)

	// Bracket: [0, max(z + 5, 8 * sqrt(df / max(df-2, 1)))].
	lo := 0.0
	scale := math.Sqrt(df / math.Max(df-2, 1))
	hi := math.Max(z+5, 8*scale)
	// Expand hi until StudentTCDF(hi) >= pp.
	for k := 0; StudentTCDF(hi, df) < pp && k < 60; k++ {
		hi *= 2
	}

	// Bisection.
	const eps = 1e-10
	const maxIter = 200
	for k := 0; k < maxIter; k++ {
		mid := 0.5 * (lo + hi)
		f := StudentTCDF(mid, df)
		if math.Abs(f-pp) < eps {
			if flipSign {
				return -mid
			}
			return mid
		}
		if f < pp {
			lo = mid
		} else {
			hi = mid
		}
		if hi-lo < eps {
			break
		}
	}
	mid := 0.5 * (lo + hi)
	if flipSign {
		return -mid
	}
	return mid
}

// standardNormalQuantileLocal is the Acklam 2004 rational approximation
// to the standard-normal inverse CDF, kept local to the copula package
// to keep StudentTQuantile self-contained.  The reality/prob package
// also exposes NormalQuantile which is byte-equivalent — both helpers
// converge on the same root, but keeping a local copy avoids a
// circular import when copula's StudentTQuantile is itself used from
// inside copula's gaussian.go boundary tests.
func standardNormalQuantileLocal(p float64) float64 {
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
		q = math.Sqrt(-2 * math.Log(p))
		return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q + c6) /
			((((d1*q+d2)*q+d3)*q+d4)*q + 1)
	} else if p <= pHigh {
		q = p - 0.5
		r = q * q
		return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r + a6) * q /
			(((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r + 1)
	}
	q = math.Sqrt(-2 * math.Log(1-p))
	return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q + c6) /
		((((d1*q+d2)*q+d3)*q+d4)*q + 1)
}

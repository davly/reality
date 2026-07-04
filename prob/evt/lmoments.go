package evt

import (
	"math"
	"sort"
)

// LMoments3 returns the first three sample L-moments (l1, l2, l3) of data,
// computed from the unbiased probability-weighted moments
//
//	b_r = (1/n) sum_i  C(i, r) / C(n-1, r) * x_(i)      (x sorted ascending)
//
// via
//
//	l1 = b0,   l2 = 2 b1 - b0,   l3 = 6 b2 - 6 b1 + b0.
//
// l1 is the mean, l2 (>= 0) is a scale, and t3 = l3/l2 is the L-skewness.
// Requires n >= 3.  ok == false if n < 3.
//
// Reference: Hosking (1990) "L-moments: analysis and estimation of
// distributions using linear combinations of order statistics", JRSS-B
// 52(1):105-124, eqs. (2.3)-(2.6).
// Precision: exact rational arithmetic in float64 (a single weighted sum per
// moment); no iteration.
func LMoments3(data []float64) (l1, l2, l3 float64, ok bool) {
	n := len(data)
	if n < 3 {
		return 0, 0, 0, false
	}
	s := append([]float64(nil), data...)
	sort.Float64s(s)
	nf := float64(n)
	var b0, b1, b2 float64
	for i := 0; i < n; i++ {
		xi := s[i]
		fi := float64(i)
		b0 += xi
		b1 += (fi / (nf - 1)) * xi
		b2 += (fi * (fi - 1)) / ((nf - 1) * (nf - 2)) * xi
	}
	b0 /= nf
	b1 /= nf
	b2 /= nf
	l1 = b0
	l2 = 2*b1 - b0
	l3 = 6*b2 - 6*b1 + b0
	return l1, l2, l3, true
}

// FitGPDPWM fits a Generalized Pareto Distribution to non-negative
// exceedances by the method of probability-weighted moments (equivalently
// L-moments):
//
//	xi    = 2 - l1/l2
//	sigma = l1 (1 - xi)
//
// where l1 = mean and l2 = 2 b1 - b0 are the first two L-moments of the
// exceedances.  This is the estimator of Hosking & Wallis (1987),
// "Parameter and Quantile Estimation for the Generalized Pareto
// Distribution", Technometrics 29(3):339-349, and is the deterministic,
// closed-form (golden-file-able) branch of GPD fitting.
//
// ok == false if n < 2, l2 is ~0 (degenerate/constant sample), or the
// implied scale is not positive.
//
// Precision: exact closed form; no iteration.
func FitGPDPWM(exceedances []float64) (GPDParams, bool) {
	n := len(exceedances)
	if n < 2 {
		return GPDParams{}, false
	}
	s := append([]float64(nil), exceedances...)
	sort.Float64s(s)
	nf := float64(n)
	var b0, b1 float64
	for i := 0; i < n; i++ {
		b0 += s[i]
		b1 += (float64(i) / (nf - 1)) * s[i]
	}
	b0 /= nf
	b1 /= nf
	l1 := b0
	l2 := 2*b1 - b0
	if math.Abs(l2) < 1e-15 {
		return GPDParams{}, false
	}
	xi := 2 - l1/l2
	sigma := l1 * (1 - xi)
	if !(sigma > 0) || math.IsNaN(xi) || math.IsInf(xi, 0) {
		return GPDParams{}, false
	}
	return GPDParams{Sigma: sigma, Xi: xi}, true
}

// FitGEVLMoments fits a Generalized Extreme Value distribution to block
// maxima by the method of L-moments (Hosking-Wallis).  With l1, l2 the first
// two L-moments and t3 = l3/l2 the L-skewness, the shape is obtained from
// Hosking's (1990) rational approximation
//
//	c = 2/(3 + t3) - ln2/ln3
//	k = 7.8590 c + 2.9554 c^2        (k is the Hosking shape; xi = -k)
//
// and then, with g = Gamma(1+k),
//
//	sigma = l2 k / ((1 - 2^{-k}) g)
//	mu    = l1 - sigma (1 - g) / k.
//
// The Gumbel limit k -> 0 (xi = 0) is handled explicitly with
// sigma = l2/ln2, mu = l1 - gamma_E sigma.
//
// ok == false if n < 3 or l2 ~ 0.  Returned GEVParams use the Coles/von Mises
// convention (xi = -k), matching GEVCDF/GEVPDF/GEVQuantile in this package.
//
// Reference: Hosking (1990) JRSS-B 52; Hosking, Wallis & Wood (1985)
// "Estimation of the GEV distribution by the method of PWMs", Technometrics
// 27:251-261; Hosking & Wallis (1997) "Regional Frequency Analysis", App.
// A.7.  Precision: the k(t3) map is a published rational approximation
// (|error| < ~9e-4 for -0.5 <= t3 <= 0.5); the sigma/mu back-out is exact
// given k.
func FitGEVLMoments(blockMaxima []float64) (GEVParams, bool) {
	l1, l2, l3, ok := LMoments3(blockMaxima)
	if !ok || math.Abs(l2) < 1e-15 {
		return GEVParams{}, false
	}
	t3 := l3 / l2

	c := 2/(3+t3) - math.Ln2/math.Log(3)
	k := 7.8590*c + 2.9554*c*c

	if math.Abs(k) < 1e-6 { // Gumbel limit
		sigma := l2 / math.Ln2
		mu := l1 - eulerMascheroni*sigma
		return GEVParams{Mu: mu, Sigma: sigma, Xi: 0}, sigma > 0
	}

	g := math.Gamma(1 + k)
	denom := (1 - math.Pow(2, -k)) * g
	if math.Abs(denom) < 1e-300 || math.IsNaN(g) || math.IsInf(g, 0) {
		return GEVParams{}, false
	}
	sigma := l2 * k / denom
	if !(sigma > 0) {
		return GEVParams{}, false
	}
	mu := l1 - sigma*(1-g)/k
	return GEVParams{Mu: mu, Sigma: sigma, Xi: -k}, true
}

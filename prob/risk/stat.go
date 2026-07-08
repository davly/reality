package risk

import "math"

// Internal moment helpers. These are deliberately private: the package's
// public surface is risk metrics, not a second copy of prob's descriptive
// statistics. They exist so each metric can state exactly which convention
// (population 1/N vs sample 1/(N-1)) it uses, rather than importing an
// opaque helper whose denominator would not be visible at the call site.

// mean returns the arithmetic mean of xs, or NaN for an empty slice.
func mean(xs []float64) float64 {
	n := len(xs)
	if n == 0 {
		return math.NaN()
	}
	var s float64
	for _, x := range xs {
		s += x
	}
	return s / float64(n)
}

// sampleStdDev returns the sample (Bessel-corrected, 1/(N-1)) standard
// deviation of xs. Returns NaN for fewer than 2 observations. This is the
// convention used for tracking error (Information ratio) and any statistic
// that treats the series as a sample drawn from a larger population.
func sampleStdDev(xs []float64) float64 {
	n := len(xs)
	if n < 2 {
		return math.NaN()
	}
	m := mean(xs)
	var ss float64
	for _, x := range xs {
		d := x - m
		ss += d * d
	}
	return math.Sqrt(ss / float64(n-1))
}

// covariancePopulation returns the population (1/N) covariance of xs and ys.
// Beta is a RATIO of covariance to variance computed with the same
// denominator, so population-vs-sample cancels; population is used here for
// simplicity and returns NaN unless the slices are equal length and non-empty.
func covariancePopulation(xs, ys []float64) float64 {
	n := len(xs)
	if n == 0 || n != len(ys) {
		return math.NaN()
	}
	mx := mean(xs)
	my := mean(ys)
	var s float64
	for i := 0; i < n; i++ {
		s += (xs[i] - mx) * (ys[i] - my)
	}
	return s / float64(n)
}

// variancePopulation returns the population (1/N) variance of xs, or NaN for
// an empty slice. Used as the denominator of Beta (see covariancePopulation).
func variancePopulation(xs []float64) float64 {
	return covariancePopulation(xs, xs)
}

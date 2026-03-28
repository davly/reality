package prob

import "math"

// ---------------------------------------------------------------------------
// Statistical hypothesis testing.
//
// These functions implement the most commonly used hypothesis tests: the
// one-sample and two-sample (Welch's) t-tests, and the chi-squared
// goodness-of-fit test.
//
// Consumers:
//   - Parallax: contradiction significance testing (t-test, chi-squared)
//   - Oracle:   prediction calibration validation (chi-squared)
//   - Echo:     analogy quality significance (t-test)
// ---------------------------------------------------------------------------

// TTestOneSample performs a one-sample two-tailed t-test.
//
// Tests the null hypothesis that the population mean equals mu0 against
// the alternative that it does not.
//
// The test statistic is:
//
//	t = (mean(data) - mu0) / (stddev(data) / sqrt(n))
//
// The p-value is computed as 2 * P(T > |t|) where T follows a Student's
// t-distribution with n-1 degrees of freedom.
//
// Valid range: len(data) >= 2 (need at least 2 observations for variance)
// Returns: t-statistic and two-tailed p-value
// Failure mode: returns (NaN, NaN) if len(data) < 2 or if all values are
// identical (zero variance)
// Precision: ~1e-12 for p-values (limited by RegularizedBetaInc)
// Reference: Student (Gosset), W.S. (1908) "The Probable Error of a Mean"
func TTestOneSample(data []float64, mu0 float64) (tStat, pValue float64) {
	n := len(data)
	if n < 2 {
		return math.NaN(), math.NaN()
	}

	// Compute mean.
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	mean := sum / float64(n)

	// Compute sample variance (Bessel's correction: divide by n-1).
	sumSq := 0.0
	for _, v := range data {
		d := v - mean
		sumSq += d * d
	}
	variance := sumSq / float64(n-1)
	if variance == 0 {
		return math.NaN(), math.NaN()
	}

	se := math.Sqrt(variance / float64(n))
	tStat = (mean - mu0) / se
	df := float64(n - 1)

	// Two-tailed p-value.
	pValue = 2.0 * (1.0 - studentTCDF(math.Abs(tStat), df))
	if pValue > 1.0 {
		pValue = 1.0
	}
	return tStat, pValue
}

// TTestTwoSample performs Welch's two-sample two-tailed t-test.
//
// Tests the null hypothesis that the two population means are equal
// against the alternative that they are not. Welch's t-test does NOT
// assume equal variances between the two groups.
//
// The test statistic is:
//
//	t = (mean1 - mean2) / sqrt(var1/n1 + var2/n2)
//
// The degrees of freedom are estimated using the Welch-Satterthwaite
// approximation:
//
//	df = (var1/n1 + var2/n2)^2 / ((var1/n1)^2/(n1-1) + (var2/n2)^2/(n2-1))
//
// Valid range: len(data1) >= 2 and len(data2) >= 2
// Returns: t-statistic and two-tailed p-value
// Failure mode: returns (NaN, NaN) if either sample has fewer than 2
// observations or if both samples have zero variance
// Precision: ~1e-12 for p-values
// Reference: Welch, B.L. (1947) "The Generalization of 'Student's'
// Problem when Several Different Population Variances are Involved"
func TTestTwoSample(data1, data2 []float64) (tStat, pValue float64) {
	n1, n2 := len(data1), len(data2)
	if n1 < 2 || n2 < 2 {
		return math.NaN(), math.NaN()
	}

	// Compute means.
	sum1, sum2 := 0.0, 0.0
	for _, v := range data1 {
		sum1 += v
	}
	for _, v := range data2 {
		sum2 += v
	}
	mean1 := sum1 / float64(n1)
	mean2 := sum2 / float64(n2)

	// Compute sample variances.
	sumSq1, sumSq2 := 0.0, 0.0
	for _, v := range data1 {
		d := v - mean1
		sumSq1 += d * d
	}
	for _, v := range data2 {
		d := v - mean2
		sumSq2 += d * d
	}
	var1 := sumSq1 / float64(n1-1)
	var2 := sumSq2 / float64(n2-1)

	seSquared := var1/float64(n1) + var2/float64(n2)
	if seSquared == 0 {
		return math.NaN(), math.NaN()
	}

	tStat = (mean1 - mean2) / math.Sqrt(seSquared)

	// Welch-Satterthwaite degrees of freedom.
	v1n := var1 / float64(n1)
	v2n := var2 / float64(n2)
	num := (v1n + v2n) * (v1n + v2n)
	denom := (v1n*v1n)/float64(n1-1) + (v2n*v2n)/float64(n2-1)
	if denom == 0 {
		return math.NaN(), math.NaN()
	}
	df := num / denom

	// Two-tailed p-value.
	pValue = 2.0 * (1.0 - studentTCDF(math.Abs(tStat), df))
	if pValue > 1.0 {
		pValue = 1.0
	}
	return tStat, pValue
}

// ChiSquaredTest performs a chi-squared goodness-of-fit test.
//
// Tests the null hypothesis that the observed frequency distribution
// matches the expected frequency distribution. Both slices must have
// the same length and all expected values must be positive.
//
// The test statistic is:
//
//	chi2 = sum_i (observed[i] - expected[i])^2 / expected[i]
//
// The p-value is computed from the chi-squared distribution with
// len(observed)-1 degrees of freedom.
//
// Valid range: len(observed) >= 2, all expected[i] > 0
// Returns: chi-squared statistic and p-value
// Failure mode: returns (NaN, NaN) if inputs are invalid
// Precision: ~1e-12 for p-values (limited by regularizedGammaLowerSeries)
// Reference: Pearson, K. (1900) "On the criterion that a given system
// of deviations from the probable in the case of a correlated system of
// variables is such that it can be reasonably supposed to have arisen
// from random sampling"
func ChiSquaredTest(observed, expected []float64) (chiSq, pValue float64) {
	if len(observed) < 2 || len(observed) != len(expected) {
		return math.NaN(), math.NaN()
	}

	chiSq = 0
	for i := range observed {
		if expected[i] <= 0 {
			return math.NaN(), math.NaN()
		}
		d := observed[i] - expected[i]
		chiSq += (d * d) / expected[i]
	}

	df := float64(len(observed) - 1)
	// p-value = 1 - CDF of chi-squared distribution at chiSq with df degrees of freedom.
	pValue = 1.0 - chiSquaredCDF(chiSq, df)
	if pValue < 0 {
		pValue = 0
	}
	return chiSq, pValue
}

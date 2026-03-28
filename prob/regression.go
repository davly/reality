package prob

import "math"

// ---------------------------------------------------------------------------
// Linear regression and multiple testing correction.
//
// These functions extend the prob package with regression analysis and
// methods for controlling the false discovery rate when performing many
// simultaneous hypothesis tests.
//
// Consumers:
//   - Parallax:  claim verification regression analysis
//   - Oracle:    prediction calibration regression
//   - Causal:    observational data regression
//   - Horizon:   personal trend regression
// ---------------------------------------------------------------------------

// LinearRegression computes the ordinary least-squares regression line
// y = slope*x + intercept for paired data (x, y), along with the
// coefficient of determination R^2.
//
// Formulas:
//
//	slope     = (n*sum(x_i*y_i) - sum(x_i)*sum(y_i)) / (n*sum(x_i^2) - (sum(x_i))^2)
//	intercept = mean(y) - slope * mean(x)
//	R^2       = 1 - SS_res / SS_tot
//	          = (n*sum(x_i*y_i) - sum(x_i)*sum(y_i))^2 /
//	            ((n*sum(x_i^2) - (sum(x_i))^2) * (n*sum(y_i^2) - (sum(y_i))^2))
//
// Valid range: len(x) == len(y) >= 2, not all x values identical.
// Returns (NaN, NaN, NaN) if len(x) < 2 or len(x) != len(y) or
// all x values are identical (zero variance in x).
// Precision: ~15 significant digits (float64)
// Reference: Weisberg, "Applied Linear Regression," 4th ed., 2014.
func LinearRegression(x, y []float64) (slope, intercept, rSquared float64) {
	n := len(x)
	if n < 2 || n != len(y) {
		return math.NaN(), math.NaN(), math.NaN()
	}

	// Accumulate sums in a single pass.
	var sx, sy, sxx, syy, sxy float64
	for i := 0; i < n; i++ {
		sx += x[i]
		sy += y[i]
		sxx += x[i] * x[i]
		syy += y[i] * y[i]
		sxy += x[i] * y[i]
	}

	nf := float64(n)
	denom := nf*sxx - sx*sx
	if denom == 0 {
		// All x values identical — slope is undefined.
		return math.NaN(), math.NaN(), math.NaN()
	}

	slope = (nf*sxy - sx*sy) / denom
	intercept = (sy - slope*sx) / nf

	// Coefficient of determination.
	ssTot := nf*syy - sy*sy
	if ssTot == 0 {
		// All y values identical — perfect fit trivially.
		rSquared = 1.0
	} else {
		ssNum := nf*sxy - sx*sy
		rSquared = (ssNum * ssNum) / (denom * ssTot)
	}

	return slope, intercept, rSquared
}

// BenjaminiHochberg applies the Benjamini-Hochberg procedure to control the
// false discovery rate (FDR) at level alpha. Given a set of p-values from
// independent hypothesis tests, it returns a boolean mask indicating which
// null hypotheses are rejected.
//
// Algorithm:
//  1. Sort p-values in ascending order (while tracking original indices).
//  2. Find the largest k such that p_(k) <= k/m * alpha, where m is the
//     total number of tests.
//  3. Reject all hypotheses with p-values <= p_(k).
//
// Valid range: alpha in (0, 1], all pValues in [0, 1]
// Returns: boolean slice of same length as pValues, true = reject null
// Reference: Benjamini, Y. & Hochberg, Y. (1995) "Controlling the False
// Discovery Rate: A Practical and Powerful Approach to Multiple Testing,"
// JRSS-B 57(1).
func BenjaminiHochberg(pValues []float64, alpha float64) []bool {
	m := len(pValues)
	if m == 0 {
		return nil
	}

	// Create index-sorted pairs.
	type indexedP struct {
		p   float64
		idx int
	}
	sorted := make([]indexedP, m)
	for i, p := range pValues {
		sorted[i] = indexedP{p: p, idx: i}
	}
	// Sort by p-value ascending (insertion sort — stable, no alloc for sort.Interface).
	for i := 1; i < m; i++ {
		key := sorted[i]
		j := i - 1
		for j >= 0 && sorted[j].p > key.p {
			sorted[j+1] = sorted[j]
			j--
		}
		sorted[j+1] = key
	}

	// Find the largest k such that p_(k) <= k/m * alpha.
	// k is 1-indexed in the original paper; we use 0-indexed internally.
	threshold := -1
	mf := float64(m)
	for i := m - 1; i >= 0; i-- {
		rank := float64(i + 1) // 1-indexed rank
		if sorted[i].p <= rank/mf*alpha {
			threshold = i
			break
		}
	}

	// Reject all hypotheses with rank <= threshold.
	result := make([]bool, m)
	if threshold >= 0 {
		for i := 0; i <= threshold; i++ {
			result[sorted[i].idx] = true
		}
	}

	return result
}

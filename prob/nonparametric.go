package prob

import (
	"math"
	"sort"
)

// ---------------------------------------------------------------------------
// Non-parametric hypothesis tests.
//
// These tests make no assumptions about the underlying distribution of the
// data, making them robust alternatives to parametric tests when normality
// cannot be assumed.
//
// Consumers:
//   - Parallax:  source credibility comparison (non-normal data)
//   - Causal:    treatment effect testing (non-parametric)
//   - Oracle:    prediction quality comparison across models
// ---------------------------------------------------------------------------

// FisherExactTest computes the two-tailed p-value for Fisher's exact test
// on a 2x2 contingency table:
//
//	            | Condition 1 | Condition 2 |
//	Group A     |     a       |     b       |
//	Group B     |     c       |     d       |
//
// The p-value is the sum of probabilities of all tables as extreme or more
// extreme than the observed table, under the null hypothesis of independence.
//
// The hypergeometric probability of a specific table is:
//
//	P = C(a+b, a) * C(c+d, c) / C(n, a+c)
//
// where n = a+b+c+d, computed in log-space for numerical stability.
//
// Valid range: a, b, c, d >= 0, a+b+c+d > 0
// Returns: two-tailed p-value
// Precision: ~1e-12 (limited by LogGamma)
// Reference: Fisher, R.A. (1922) "On the Interpretation of Chi-Squared
// from Contingency Tables, and the Calculation of P"
func FisherExactTest(a, b, c, d int) float64 {
	n := a + b + c + d
	if n == 0 {
		return 1.0
	}

	// Log of the hypergeometric probability for a specific table configuration.
	logHyper := func(aa, bb, cc, dd int) float64 {
		return LogGamma(float64(aa+bb+1)) + LogGamma(float64(cc+dd+1)) +
			LogGamma(float64(aa+cc+1)) + LogGamma(float64(bb+dd+1)) -
			LogGamma(float64(aa+1)) - LogGamma(float64(bb+1)) -
			LogGamma(float64(cc+1)) - LogGamma(float64(dd+1)) -
			LogGamma(float64(n+1))
	}

	// Probability of the observed table.
	logPObs := logHyper(a, b, c, d)
	pObs := math.Exp(logPObs)

	// Row and column sums are fixed under the null.
	r1 := a + b // row 1 total
	r2 := c + d // row 2 total
	c1 := a + c // col 1 total

	// Enumerate all possible tables with the same marginals.
	// a ranges from max(0, c1 - r2) to min(r1, c1).
	aMin := c1 - r2
	if aMin < 0 {
		aMin = 0
	}
	aMax := r1
	if c1 < aMax {
		aMax = c1
	}

	pValue := 0.0
	for ai := aMin; ai <= aMax; ai++ {
		bi := r1 - ai
		ci := c1 - ai
		di := r2 - ci
		p := math.Exp(logHyper(ai, bi, ci, di))
		// Two-tailed: sum probabilities of tables at least as extreme.
		if p <= pObs+1e-14 { // small tolerance for floating-point comparison
			pValue += p
		}
	}

	if pValue > 1.0 {
		pValue = 1.0
	}
	return pValue
}

// MannWhitneyU performs the Mann-Whitney U test (also known as the
// Wilcoxon rank-sum test), a non-parametric test for whether two
// independent samples come from the same distribution.
//
// The test statistic U is defined as:
//
//	U = min(U1, U2)
//
// where:
//
//	U1 = R1 - n1*(n1+1)/2   (R1 = sum of ranks of sample 1)
//	U2 = n1*n2 - U1
//
// For large samples (n1, n2 >= 8), a normal approximation is used for
// the p-value:
//
//	z = (U - n1*n2/2) / sqrt(n1*n2*(n1+n2+1)/12)
//	p = 2 * Phi(-|z|)   (two-tailed)
//
// Valid range: len(x) >= 1, len(y) >= 1
// Returns: (U statistic, two-tailed p-value)
// Returns (NaN, NaN) if either sample is empty
// Precision: ~1e-8 for p-values (normal approximation)
// Reference: Mann, H.B. & Whitney, D.R. (1947) "On a Test of Whether
// One of Two Random Variables is Stochastically Larger than the Other"
func MannWhitneyU(x, y []float64) (U, pValue float64) {
	n1, n2 := len(x), len(y)
	if n1 == 0 || n2 == 0 {
		return math.NaN(), math.NaN()
	}

	// Pool all values with group labels.
	type ranked struct {
		value float64
		group int // 0 = x, 1 = y
	}
	n := n1 + n2
	pool := make([]ranked, n)
	for i, v := range x {
		pool[i] = ranked{value: v, group: 0}
	}
	for i, v := range y {
		pool[n1+i] = ranked{value: v, group: 1}
	}

	// Sort by value.
	sort.Slice(pool, func(i, j int) bool {
		return pool[i].value < pool[j].value
	})

	// Assign ranks with tie averaging.
	ranks := make([]float64, n)
	i := 0
	for i < n {
		j := i
		// Find all tied values.
		for j < n && pool[j].value == pool[i].value {
			j++
		}
		// Average rank for the tie group: (i+1 + j) / 2 (1-indexed)
		avgRank := (float64(i+1) + float64(j)) / 2.0
		for k := i; k < j; k++ {
			ranks[k] = avgRank
		}
		i = j
	}

	// Sum ranks for group x (group 0).
	r1 := 0.0
	for i := 0; i < n; i++ {
		if pool[i].group == 0 {
			r1 += ranks[i]
		}
	}

	n1f := float64(n1)
	n2f := float64(n2)

	u1 := r1 - n1f*(n1f+1)/2
	u2 := n1f*n2f - u1
	U = math.Min(u1, u2)

	// Normal approximation for two-tailed p-value.
	mu := n1f * n2f / 2.0
	sigma := math.Sqrt(n1f * n2f * (n1f + n2f + 1) / 12.0)
	if sigma == 0 {
		return U, 1.0
	}
	z := (U - mu) / sigma
	// Two-tailed: 2 * Phi(z) where z <= 0.
	pValue = 2.0 * NormalCDF(z, 0, 1)
	if pValue > 1.0 {
		pValue = 1.0
	}
	return U, pValue
}

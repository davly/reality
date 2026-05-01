package copula

import "sort"

// KendallTau returns Kendall's tau rank correlation between paired samples
// x and y.
//
//	tau = (concordant - discordant) / (n * (n-1) / 2)
//
// where a pair (i, j) is concordant if sign(x[j] - x[i]) == sign(y[j] - y[i])
// and discordant if the signs disagree.  Ties (sign == 0) are excluded.
// This is Kendall's "tau-a" (no tie correction); for continuous marginals
// without ties, tau-a == tau-b == tau-c.
//
// Tau is the canonical fitting statistic for Archimedean copulas (Clayton:
// theta = 2*tau / (1-tau); Gumbel: theta = 1 / (1-tau)) and the Kruskal
// 1958 link to Gaussian copula correlation:
//
//	rho = sin(pi * tau / 2)
//
// is the closed-form one-shot fit for the bivariate Gaussian copula.
//
// Algorithm: O(n^2) concordance counter — the simple, deterministic,
// branch-clean reference implementation, byte-for-byte mirroring the
// FleetWorks RubberDuck `CopulaModels.KendallTau` C# routine
// (Analysis/CopulaModels.cs:143-165, commit dc63772f).  Mergesort-based
// O(n log n) variants exist (Knight 1966) but produce identical output;
// we ship the O(n^2) form to keep the cross-substrate parity audit
// trivially line-by-line.
//
// Returns 0 if n < 2 or all pairs tie.  Returns ErrLengthMismatch when
// len(x) != len(y).  Inputs are not mutated.
//
// Reference: Kendall, M. G. (1938).  A New Measure of Rank Correlation.
// Biometrika 30 (1/2): 81-93.
func KendallTau(x, y []float64) (float64, error) {
	if len(x) != len(y) {
		return 0, ErrLengthMismatch
	}
	n := len(x)
	if n < 2 {
		return 0, nil
	}
	var concordant, discordant int64
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			dx := sign(x[j] - x[i])
			dy := sign(y[j] - y[i])
			product := dx * dy
			if product > 0 {
				concordant++
			} else if product < 0 {
				discordant++
			}
			// ties (product == 0) are excluded
		}
	}
	pairs := int64(n) * int64(n-1) / 2
	if pairs == 0 {
		return 0, nil
	}
	return float64(concordant-discordant) / float64(pairs), nil
}

// sign returns the sign of x as an int: -1, 0, or +1.
func sign(x float64) int {
	if x > 0 {
		return 1
	}
	if x < 0 {
		return -1
	}
	return 0
}

// EmpiricalCdf returns the rank-based empirical CDF F_hat(x_i) = rank(x_i)
// / (n+1) of the input data.  This is the canonical PIT (probability-
// integral-transform) input shape for any copula CDF — the (n+1) divisor
// (rather than n) keeps the result strictly in (0, 1), avoiding the
// boundary singularities that probit(0) and probit(1) introduce.
//
// Returned slice has the same length as data.  Entry i is the rank of
// data[i] in ascending order, divided by (n+1).  Ties broken by original
// index (deterministic, matches the FW C# CopulaModels.EmpiricalCdf
// closure semantics).
//
// Inputs are not mutated.
func EmpiricalCdf(data []float64) []float64 {
	n := len(data)
	if n == 0 {
		return nil
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		return data[idx[a]] < data[idx[b]]
	})
	ranks := make([]float64, n)
	for r := 0; r < n; r++ {
		ranks[idx[r]] = float64(r+1) / float64(n+1)
	}
	return ranks
}

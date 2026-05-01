package transport

import "math"

// PairwiseWasserstein1D returns the symmetric n×n matrix of
// Wasserstein-p distances between the supplied 1-D distributions.
// Diagonal entries are 0; entry (i, j) = (j, i) = W_p(d_i, d_j).
//
// Use cases (from the L11 hunt entry consumer-list):
//
//   - Diversity measurement on a population of agents (RubberDuck's
//     EvolutionOrchestratorService gating diversity for genetic
//     selection — `flagships/rubberduck/RubberDuck.Brain.Evolution`).
//   - Regime-similarity clustering on K candidate strategy
//     histograms (RubberDuck's RegimeContextService — currently HMM-
//     gated; this is the design-doc-spec'd alternative).
//   - Cross-tenant prior similarity for FleetWorks Fleet Memory
//     (Reality Math Hunt §03).
//
// Returns a fresh n×n matrix; never mutates inputs.  If any pairwise
// distance is NaN (e.g. one of the distributions is empty after NaN/
// Inf filtering) the corresponding entries are left as NaN — callers
// can treat NaN as "incomparable".
//
// Errors propagate: an invalid p (returns ErrInvalidP) aborts before
// any distance is computed.
//
// Complexity O(K^2 * (n + m) log(n + m)) where K is the number of
// distributions and n, m are individual distribution sizes; the
// closed-form 1D path keeps the constant factor small.
func PairwiseWasserstein1D(distributions [][]float64, p float64) ([][]float64, error) {
	if !isValidP(p) {
		return nil, ErrInvalidP
	}
	k := len(distributions)
	out := make([][]float64, k)
	for i := range out {
		out[i] = make([]float64, k)
	}
	for i := 0; i < k; i++ {
		for j := i + 1; j < k; j++ {
			d, err := Wasserstein1D(distributions[i], distributions[j], p)
			if err != nil {
				return nil, err
			}
			out[i][j] = d
			out[j][i] = d
		}
	}
	return out, nil
}

// MinPairwiseWasserstein1D returns the smallest pairwise Wasserstein-p
// distance across all unordered pairs of supplied distributions, plus
// the (i, j) index pair achieving it.  When fewer than two
// distributions are supplied the returned distance is +Inf and idx is
// [-1, -1] (callers should guard).  Pairs whose Wasserstein-1D returns
// NaN (degenerate empty / all-NaN inputs) are skipped.
//
// Mirrors RubberDuck's `OptimalTransport.MinimumPairwiseDistance`
// (`flagships/rubberduck/RubberDuck.Core/Analysis/OptimalTransport.cs:130-147`)
// extended with the index pair for downstream reasoning (which two
// regimes are most-similar, which two agents are diversity-floor
// twins, etc.).
//
// On ErrInvalidP the result is propagated; otherwise this function
// only returns nil-error.
func MinPairwiseWasserstein1D(
	distributions [][]float64, p float64,
) (idx [2]int, dist float64, err error) {
	if !isValidP(p) {
		return [2]int{-1, -1}, math.Inf(1), ErrInvalidP
	}
	k := len(distributions)
	if k < 2 {
		return [2]int{-1, -1}, math.Inf(1), nil
	}
	bestI, bestJ := -1, -1
	bestD := math.Inf(1)
	for i := 0; i < k; i++ {
		for j := i + 1; j < k; j++ {
			d, dErr := Wasserstein1D(distributions[i], distributions[j], p)
			if dErr != nil {
				return [2]int{-1, -1}, math.Inf(1), dErr
			}
			if math.IsNaN(d) {
				continue
			}
			if d < bestD {
				bestD = d
				bestI = i
				bestJ = j
			}
		}
	}
	if bestI < 0 {
		// All pairs produced NaN — surface +Inf with sentinel idx.
		return [2]int{-1, -1}, math.Inf(1), nil
	}
	return [2]int{bestI, bestJ}, bestD, nil
}

package gametheory

import "math"

// ---------------------------------------------------------------------------
// Voting Power Indices
//
// Functions for measuring the power of voters in weighted voting systems.
// These are used by LiquidVote for delegated voting power analysis and
// by any system that needs to understand coalition dynamics.
// ---------------------------------------------------------------------------

// BanzhafIndex computes the Banzhaf power index for a weighted voting game.
// Each voter i has weight weights[i]. A coalition is winning if the sum of
// its members' weights meets or exceeds the quota.
//
// The Banzhaf power index for voter i is the fraction of winning coalitions
// in which voter i is critical (i.e., removing i makes the coalition losing),
// normalized so that all indices sum to 1.
//
// Parameters:
//   - weights: voting weight of each voter (must be non-negative)
//   - quota: minimum total weight for a winning coalition (must be positive)
//
// Returns a slice of Banzhaf power indices, one per voter, summing to 1.
// Returns nil if weights is empty or quota <= 0.
//
// Time complexity: O(n * 2^n) where n = len(weights). Practical for n <= 25.
// Precision: exact (integer counting, final division)
//
// Definition: The Banzhaf index measures a voter's ability to change an
// outcome by joining or leaving a coalition (swing voter).
//
// Reference: Banzhaf, J.F. (1965) "Weighted Voting Doesn't Work: A
// Mathematical Analysis", Rutgers Law Review 19:317-343.
func BanzhafIndex(weights []float64, quota float64) []float64 {
	n := len(weights)
	if n == 0 || quota <= 0 {
		return nil
	}

	// Count critical coalitions for each voter.
	// A voter i is critical in coalition S if:
	//   sum(S) >= quota AND sum(S) - weights[i] < quota
	swings := make([]float64, n)
	totalCoalitions := 1 << n // 2^n

	for mask := 0; mask < totalCoalitions; mask++ {
		// Compute total weight of this coalition.
		var total float64
		for i := 0; i < n; i++ {
			if mask&(1<<i) != 0 {
				total += weights[i]
			}
		}

		// Check if winning.
		if total < quota {
			continue
		}

		// For each member, check if they are critical.
		for i := 0; i < n; i++ {
			if mask&(1<<i) == 0 {
				continue // voter i not in coalition
			}
			if total-weights[i] < quota {
				swings[i]++
			}
		}
	}

	// Normalize so indices sum to 1.
	totalSwings := 0.0
	for _, s := range swings {
		totalSwings += s
	}

	result := make([]float64, n)
	if totalSwings == 0 {
		// No voter is ever critical (degenerate game).
		for i := range result {
			result[i] = 1.0 / float64(n)
		}
		return result
	}

	for i := 0; i < n; i++ {
		result[i] = swings[i] / totalSwings
	}
	return result
}

// ShapleyValue computes the Shapley value for an n-player cooperative game
// defined by the characteristic function charFunc. The Shapley value gives
// each player their average marginal contribution across all possible
// orderings of players.
//
// Parameters:
//   - n: number of players (players are indexed 0 to n-1)
//   - charFunc: characteristic function mapping coalitions (represented as
//     boolean slices where coalition[i] = true means player i is in the
//     coalition) to the coalition's value
//
// For n <= 12, computes the exact Shapley value by enumerating all
// permutations. For n > 12, uses Monte Carlo sampling with 100,000
// random permutations.
//
// Returns a slice of Shapley values, one per player. By the efficiency
// axiom, they sum to charFunc(grand coalition).
//
// Definition: phi_i = sum over all S not containing i of
//
//	|S|! * (n - |S| - 1)! / n! * [v(S union {i}) - v(S)]
//
// Precision: exact for n <= 12; ~1e-3 relative for n > 12 (Monte Carlo)
// Reference: Shapley, L.S. (1953) "A Value for N-Person Games",
// Contributions to the Theory of Games II, Annals of Mathematics Studies 28.
func ShapleyValue(n int, charFunc func(coalition []bool) float64) []float64 {
	if n <= 0 {
		return nil
	}

	if n <= 12 {
		return shapleyExact(n, charFunc)
	}
	return shapleySampled(n, charFunc, 100000)
}

// shapleyExact computes exact Shapley values by iterating over all 2^n
// coalitions and computing marginal contributions.
func shapleyExact(n int, charFunc func(coalition []bool) float64) []float64 {
	values := make([]float64, n)
	nFactorial := factorial(n)
	totalCoalitions := 1 << n

	// Precompute factorials.
	factorials := make([]float64, n+1)
	factorials[0] = 1
	for i := 1; i <= n; i++ {
		factorials[i] = factorials[i-1] * float64(i)
	}

	// For each coalition S (not containing i), compute marginal contribution
	// of adding player i.
	for mask := 0; mask < totalCoalitions; mask++ {
		// Build coalition boolean slice.
		coalition := make([]bool, n)
		sSize := 0
		for j := 0; j < n; j++ {
			if mask&(1<<j) != 0 {
				coalition[j] = true
				sSize++
			}
		}

		// Skip the grand coalition — no player is outside it.
		if sSize == n {
			continue
		}

		vS := charFunc(coalition)

		// For each player NOT in this coalition, compute marginal contribution.
		// Weight: |S|! * (n - |S| - 1)! / n!
		weight := factorials[sSize] * factorials[n-sSize-1] / nFactorial

		for i := 0; i < n; i++ {
			if coalition[i] {
				continue // player already in coalition
			}
			// v(S union {i}) - v(S)
			coalition[i] = true
			vSi := charFunc(coalition)
			coalition[i] = false

			values[i] += weight * (vSi - vS)
		}
	}

	return values
}

// shapleySampled estimates Shapley values using Monte Carlo sampling of
// random permutations.
func shapleySampled(n int, charFunc func(coalition []bool) float64, iterations int) []float64 {
	values := make([]float64, n)

	// Use a deterministic LCG for reproducibility.
	// Parameters from Numerical Recipes.
	seed := uint64(42)
	lcgNext := func() uint64 {
		seed = seed*6364136223846793005 + 1442695040888963407
		return seed
	}

	perm := make([]int, n)
	for iter := 0; iter < iterations; iter++ {
		// Generate a random permutation using Fisher-Yates.
		for i := 0; i < n; i++ {
			perm[i] = i
		}
		for i := n - 1; i > 0; i-- {
			j := int(lcgNext() % uint64(i+1))
			perm[i], perm[j] = perm[j], perm[i]
		}

		// Walk through the permutation, computing marginal contributions.
		coalition := make([]bool, n)
		prevVal := 0.0
		for _, player := range perm {
			coalition[player] = true
			newVal := charFunc(coalition)
			values[player] += newVal - prevVal
			prevVal = newVal
		}
	}

	// Average over iterations.
	for i := range values {
		values[i] /= float64(iterations)
	}

	return values
}

// factorial returns n! as float64.
func factorial(n int) float64 {
	if n <= 0 {
		return 1
	}
	result := 1.0
	for i := 2; i <= n; i++ {
		result *= float64(i)
	}
	return result
}

// ShapleyValueWeightedVoting is a convenience function that computes the
// Shapley-Shubik power index for a weighted voting game. This is the Shapley
// value applied to the simple game where the characteristic function is 1
// for winning coalitions and 0 for losing coalitions.
//
// Parameters:
//   - weights: voting weight of each voter
//   - quota: minimum total weight for a winning coalition
//
// Returns normalized power indices summing to 1.
// Reference: Shapley, L.S. & Shubik, M. (1954) "A Method for Evaluating
// the Distribution of Power in a Committee System", American Political
// Science Review 48(3):787-792.
func ShapleyValueWeightedVoting(weights []float64, quota float64) []float64 {
	n := len(weights)
	if n == 0 || quota <= 0 {
		return nil
	}

	charFunc := func(coalition []bool) float64 {
		var total float64
		for i, in := range coalition {
			if in {
				total += weights[i]
			}
		}
		if total >= quota {
			return 1
		}
		return 0
	}

	raw := ShapleyValue(n, charFunc)

	// Normalize (for simple games, raw values already sum to 1, but
	// normalize anyway for robustness against floating point).
	sum := 0.0
	for _, v := range raw {
		sum += v
	}
	if math.Abs(sum) < 1e-15 {
		result := make([]float64, n)
		for i := range result {
			result[i] = 1.0 / float64(n)
		}
		return result
	}
	result := make([]float64, n)
	for i, v := range raw {
		result[i] = v / sum
	}
	return result
}

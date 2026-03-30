package prob

import "math"

// MarkovSteadyState computes the steady-state (stationary) distribution of
// a discrete-time Markov chain using power iteration.
//
// The transition matrix is stored as a flat row-major slice of length n*n,
// where transitionMatrix[i*n + j] is the probability of transitioning from
// state i to state j. Each row must sum to 1.0 (stochastic matrix).
//
// Parameters:
//   - transitionMatrix: row-major n x n transition matrix (length n*n).
//   - n: number of states.
//
// Returns:
//   - The steady-state probability distribution as a slice of length n.
//     The distribution satisfies pi * P = pi and sum(pi) = 1.
//
// If n <= 0 or the matrix size does not match n*n, returns nil.
// The algorithm runs at most 1000 iterations or until convergence
// (L1 norm of change < 1e-12).
//
// For reducible or periodic chains, the result is the distribution after
// convergence from a uniform initial state, which may not be unique.
//
// Time complexity: O(iterations * n^2).
// Space complexity: O(n).
//
// Reference: standard power method for stochastic matrices.
func MarkovSteadyState(transitionMatrix []float64, n int) []float64 {
	if n <= 0 || len(transitionMatrix) != n*n {
		return nil
	}

	// Start from uniform distribution.
	pi := make([]float64, n)
	for i := range pi {
		pi[i] = 1.0 / float64(n)
	}

	next := make([]float64, n)
	const maxIter = 1000
	const eps = 1e-12

	for iter := 0; iter < maxIter; iter++ {
		// next = pi * P  (row vector times matrix)
		for j := 0; j < n; j++ {
			sum := 0.0
			for i := 0; i < n; i++ {
				sum += pi[i] * transitionMatrix[i*n+j]
			}
			next[j] = sum
		}

		// Check convergence: L1 norm of difference.
		diff := 0.0
		for i := 0; i < n; i++ {
			diff += math.Abs(next[i] - pi[i])
		}

		pi, next = next, pi

		if diff < eps {
			break
		}
	}

	return pi
}

// MarkovSimulate simulates a discrete-time Markov chain for a given number
// of steps, starting from a specified initial state.
//
// The transition matrix is stored as a flat row-major slice of length n*n.
// State transitions are deterministic given the probabilities: at each step,
// the next state is chosen by the cumulative probability thresholds using
// a simple deterministic sampling based on step count (for reproducibility).
//
// To support deterministic golden-file testing, the simulation uses a
// simple linear congruential generator (LCG) seeded from the initial state
// and step count. This makes the output fully reproducible.
//
// Parameters:
//   - transitionMatrix: row-major n x n transition matrix (length n*n).
//   - n: number of states.
//   - initialState: the starting state index (must be in [0, n)).
//   - steps: number of transitions to simulate.
//
// Returns:
//   - A slice of length steps+1 containing the state at each time step,
//     starting with initialState.
//
// If n <= 0, the matrix size does not match n*n, or initialState is out
// of range, returns nil.
//
// Time complexity: O(steps * n).
// Space complexity: O(steps + n).
func MarkovSimulate(transitionMatrix []float64, n int, initialState int, steps int) []int {
	if n <= 0 || len(transitionMatrix) != n*n {
		return nil
	}
	if initialState < 0 || initialState >= n {
		return nil
	}
	if steps < 0 {
		return nil
	}

	path := make([]int, steps+1)
	path[0] = initialState

	// LCG for deterministic pseudo-random sampling.
	// Parameters from Numerical Recipes.
	seed := uint64(initialState*1000 + 42)
	nextRand := func() float64 {
		seed = seed*6364136223846793005 + 1442695040888963407
		return float64(seed>>11) / float64(1<<53)
	}

	current := initialState
	for step := 0; step < steps; step++ {
		r := nextRand()
		row := current * n
		cumulative := 0.0
		next := n - 1 // default to last state
		for j := 0; j < n; j++ {
			cumulative += transitionMatrix[row+j]
			if r < cumulative {
				next = j
				break
			}
		}
		current = next
		path[step+1] = current
	}

	return path
}

package optim

import (
	"math"
	"math/rand"
)

// ---------------------------------------------------------------------------
// Metaheuristic Optimization
//
// Stochastic global optimization methods that do not require gradient
// information. These methods trade speed for the ability to escape local
// minima and work on non-differentiable, noisy, or combinatorial objectives.
// ---------------------------------------------------------------------------

// SimulatedAnnealing minimizes f using the simulated annealing (SA)
// metaheuristic. Starting from x0, the algorithm explores the search space
// by proposing random neighbor solutions and accepting them with a
// probability that decreases as the temperature cools.
//
// Parameters:
//   - f:        objective function R^n → R (to be minimized)
//   - x0:       initial solution (copied, not modified)
//   - neighbor:  generates a neighbor of x; neighbor(current, out) writes
//                the neighbor into out, avoiding allocation per iteration
//   - temp0:    initial temperature (controls initial acceptance probability)
//   - cooling:  multiplicative cooling factor per iteration (e.g., 0.999)
//   - maxIter:  maximum number of iterations
//   - rng:      random number generator for reproducibility (stdlib math/rand)
//
// Returns (bestX, bestF) — the best solution found and its objective value.
//
// Acceptance probability: P(accept worse) = exp(-delta / T) where
// delta = f(neighbor) - f(current) > 0 and T is the current temperature.
//
// Reference: Kirkpatrick, Gelatt & Vecchi, "Optimization by Simulated
//            Annealing," Science 220 (1983).
func SimulatedAnnealing(
	f func([]float64) float64,
	x0 []float64,
	neighbor func([]float64, []float64),
	temp0, cooling float64,
	maxIter int,
	rng *rand.Rand,
) ([]float64, float64) {
	n := len(x0)

	// Current solution.
	current := make([]float64, n)
	copy(current, x0)
	fCurrent := f(current)

	// Best solution found.
	best := make([]float64, n)
	copy(best, current)
	fBest := fCurrent

	// Neighbor workspace.
	cand := make([]float64, n)

	temp := temp0

	for iter := 0; iter < maxIter; iter++ {
		// Generate neighbor.
		neighbor(current, cand)
		fCand := f(cand)

		delta := fCand - fCurrent

		// Accept if improving, or with Boltzmann probability if worsening.
		accept := delta < 0
		if !accept && temp > 0 {
			p := math.Exp(-delta / temp)
			accept = rng.Float64() < p
		}

		if accept {
			copy(current, cand)
			fCurrent = fCand

			// Track global best.
			if fCurrent < fBest {
				copy(best, current)
				fBest = fCurrent
			}
		}

		// Cool the temperature.
		temp *= cooling
	}

	return best, fBest
}

package optim

import "math"

// ---------------------------------------------------------------------------
// Genetic Algorithm
//
// An evolutionary metaheuristic for global optimization of real-valued
// functions. Uses tournament selection, BLX-alpha crossover, and Gaussian
// mutation. Suitable for non-convex, noisy, or multi-modal landscapes
// where gradient information is unavailable.
// ---------------------------------------------------------------------------

// GeneticAlgorithm minimizes fitness using a real-coded genetic algorithm.
//
// Parameters:
//   - fitness:  objective function R^dim → R (to be minimized)
//   - dim:      number of decision variables
//   - popSize:  population size (must be >= 2; if odd, rounded up for pairing)
//   - gens:     number of generations
//   - mutRate:  mutation probability per gene (e.g., 0.1)
//   - rng:      random number generator with Float64() method returning [0, 1)
//
// Returns (bestSolution, bestFitness) — the best individual found across all
// generations and its fitness value.
//
// Algorithm:
//  1. Initialize population uniformly in [-5, 5]^dim.
//  2. Each generation:
//     a. Tournament selection (size 3) to fill the mating pool.
//     b. BLX-alpha crossover (alpha = 0.5) on random pairs.
//     c. Gaussian mutation with adaptive sigma.
//     d. Elitism: best individual always survives.
//  3. Return the best individual found.
//
// The search domain [-5, 5]^dim is a common default for benchmark functions.
// For problems with different domains, the fitness function can internally
// transform coordinates.
//
// Reference: Holland, J.H. (1975) "Adaptation in Natural and Artificial Systems";
//
//	Eshelman & Schaffer (1993) "Real-Coded Genetic Algorithms and
//	Interval-Schemata" (BLX-alpha crossover).
func GeneticAlgorithm(
	fitness func([]float64) float64,
	dim, popSize, gens int,
	mutRate float64,
	rng interface{ Float64() float64 },
) ([]float64, float64) {
	if popSize < 2 {
		popSize = 2
	}
	if popSize%2 != 0 {
		popSize++
	}

	// Helper: generate a standard normal variate using Box-Muller transform.
	normalRand := func() float64 {
		u1 := rng.Float64()
		u2 := rng.Float64()
		if u1 < 1e-15 {
			u1 = 1e-15
		}
		return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
	}

	// Initialize population uniformly in [-5, 5]^dim.
	pop := make([][]float64, popSize)
	fit := make([]float64, popSize)
	for i := 0; i < popSize; i++ {
		pop[i] = make([]float64, dim)
		for j := 0; j < dim; j++ {
			pop[i][j] = -5 + 10*rng.Float64()
		}
		fit[i] = fitness(pop[i])
	}

	// Track best individual.
	bestIdx := 0
	for i := 1; i < popSize; i++ {
		if fit[i] < fit[bestIdx] {
			bestIdx = i
		}
	}
	bestSol := make([]float64, dim)
	copy(bestSol, pop[bestIdx])
	bestFit := fit[bestIdx]

	// Workspace for the next generation.
	newPop := make([][]float64, popSize)
	newFit := make([]float64, popSize)
	for i := 0; i < popSize; i++ {
		newPop[i] = make([]float64, dim)
	}

	// Tournament selection: pick the best of 3 random individuals.
	tournament := func() int {
		best := int(rng.Float64() * float64(popSize))
		if best >= popSize {
			best = popSize - 1
		}
		for k := 0; k < 2; k++ {
			cand := int(rng.Float64() * float64(popSize))
			if cand >= popSize {
				cand = popSize - 1
			}
			if fit[cand] < fit[best] {
				best = cand
			}
		}
		return best
	}

	for gen := 0; gen < gens; gen++ {
		// Adaptive mutation sigma: decreases over generations.
		sigma := 1.0 * (1.0 - float64(gen)/float64(gens+1))
		if sigma < 0.01 {
			sigma = 0.01
		}

		// Elitism: copy best individual to slot 0.
		copy(newPop[0], bestSol)
		newFit[0] = bestFit

		// Fill the rest of the population via crossover + mutation.
		for i := 1; i < popSize; i += 2 {
			p1 := tournament()
			p2 := tournament()

			// BLX-alpha crossover (alpha = 0.5).
			const alpha = 0.5
			for j := 0; j < dim; j++ {
				lo := pop[p1][j]
				hi := pop[p2][j]
				if lo > hi {
					lo, hi = hi, lo
				}
				span := hi - lo
				cLo := lo - alpha*span
				cHi := hi + alpha*span

				c1 := cLo + rng.Float64()*(cHi-cLo)
				c2 := cLo + rng.Float64()*(cHi-cLo)

				// Gaussian mutation.
				if rng.Float64() < mutRate {
					c1 += sigma * normalRand()
				}
				if rng.Float64() < mutRate {
					c2 += sigma * normalRand()
				}

				newPop[i][j] = c1
				if i+1 < popSize {
					newPop[i+1][j] = c2
				}
			}
			newFit[i] = fitness(newPop[i])
			if i+1 < popSize {
				newFit[i+1] = fitness(newPop[i+1])
			}
		}

		// Swap populations.
		pop, newPop = newPop, pop
		fit, newFit = newFit, fit

		// Update best.
		for i := 0; i < popSize; i++ {
			if fit[i] < bestFit {
				copy(bestSol, pop[i])
				bestFit = fit[i]
			}
		}
	}

	return bestSol, bestFit
}

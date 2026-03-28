package gametheory

import "math"

// ---------------------------------------------------------------------------
// Multi-Armed Bandits (Explore/Exploit)
//
// Classical algorithms for the multi-armed bandit problem, where an agent
// must balance exploration (trying new arms) with exploitation (pulling
// the best-known arm). These are fundamental to recommendation systems,
// A/B testing, and adaptive resource allocation.
// ---------------------------------------------------------------------------

// UCB1 selects the arm to pull using the Upper Confidence Bound algorithm.
// Returns the index of the arm with the highest UCB1 score.
//
// For each arm i with count n_i pulls and average reward x_i:
//
//	UCB1(i) = x_i + sqrt(2 * ln(N) / n_i)
//
// where N is the total number of pulls across all arms.
//
// If any arm has count 0, it is selected first (unexplored arms have
// infinite confidence). Among multiple unexplored arms, the lowest-index
// one is returned.
//
// Parameters:
//   - counts: number of times each arm has been pulled
//   - rewards: total cumulative reward from each arm
//   - totalPulls: total pulls across all arms (sum of counts)
//
// Valid range: len(counts) > 0, totalPulls >= 0
// Returns -1 if no arms exist.
//
// Reference: Auer, P., Cesa-Bianchi, N., Fischer, P. (2002) "Finite-time
// Analysis of the Multiarmed Bandit Problem", Machine Learning 47(2-3).
func UCB1(counts []int, rewards []float64, totalPulls int) int {
	n := len(counts)
	if n == 0 {
		return -1
	}

	// Always select an unexplored arm first.
	for i := 0; i < n; i++ {
		if counts[i] == 0 {
			return i
		}
	}

	// All arms explored — use UCB1 formula.
	logTotal := math.Log(float64(totalPulls))

	bestArm := 0
	bestScore := -math.MaxFloat64
	for i := 0; i < n; i++ {
		avgReward := rewards[i] / float64(counts[i])
		exploration := math.Sqrt(2 * logTotal / float64(counts[i]))
		score := avgReward + exploration
		if score > bestScore {
			bestScore = score
			bestArm = i
		}
	}

	return bestArm
}

// ThompsonSampling selects the arm to pull using Thompson sampling for
// Bernoulli bandits. Each arm's success probability is modeled with a
// Beta(successes+1, failures+1) prior. The algorithm draws a sample from
// each arm's posterior distribution and selects the arm with the highest
// sample.
//
// The Beta distribution sampling uses the inverse CDF method via the
// ratio-of-uniforms method for generating gamma variates.
//
// Parameters:
//   - successes: number of successes for each arm
//   - failures: number of failures for each arm
//   - rng: random number generator providing Float64() in [0, 1)
//
// Valid range: len(successes) > 0, all values >= 0
// Returns -1 if no arms exist.
//
// Reference: Thompson, W.R. (1933) "On the Likelihood that One Unknown
// Probability Exceeds Another in View of the Evidence of Two Samples",
// Biometrika 25(3-4):285-294.
func ThompsonSampling(successes, failures []int, rng interface{ Float64() float64 }) int {
	n := len(successes)
	if n == 0 {
		return -1
	}

	bestArm := 0
	bestSample := -1.0

	for i := 0; i < n; i++ {
		alpha := float64(successes[i]) + 1
		beta := float64(failures[i]) + 1
		sample := sampleBeta(alpha, beta, rng)
		if sample > bestSample {
			bestSample = sample
			bestArm = i
		}
	}

	return bestArm
}

// sampleBeta draws a sample from the Beta(alpha, beta) distribution using
// the gamma variate method: if X ~ Gamma(alpha, 1) and Y ~ Gamma(beta, 1),
// then X/(X+Y) ~ Beta(alpha, beta).
func sampleBeta(alpha, beta float64, rng interface{ Float64() float64 }) float64 {
	x := sampleGamma(alpha, rng)
	y := sampleGamma(beta, rng)
	if x+y == 0 {
		return 0.5
	}
	return x / (x + y)
}

// sampleGamma draws a sample from Gamma(shape, 1) using Marsaglia and
// Tsang's method (2000) for shape >= 1, and Ahrens-Dieter's method for
// shape < 1.
//
// Reference: Marsaglia, G. & Tsang, W.W. (2000) "A Simple Method for
// Generating Gamma Variables", ACM Transactions on Mathematical Software.
func sampleGamma(shape float64, rng interface{ Float64() float64 }) float64 {
	if shape < 1 {
		// Ahrens-Dieter method: if X ~ Gamma(shape+1, 1) and
		// U ~ Uniform(0, 1), then X * U^(1/shape) ~ Gamma(shape, 1).
		u := rng.Float64()
		for u == 0 {
			u = rng.Float64()
		}
		return sampleGamma(shape+1, rng) * math.Pow(u, 1.0/shape)
	}

	// Marsaglia-Tsang method for shape >= 1.
	d := shape - 1.0/3.0
	c := 1.0 / math.Sqrt(9.0*d)

	for {
		var x, v float64
		for {
			x = sampleStdNormal(rng)
			v = 1.0 + c*x
			if v > 0 {
				break
			}
		}
		v = v * v * v
		u := rng.Float64()
		// Squeeze test.
		if u < 1.0-0.0331*(x*x)*(x*x) {
			return d * v
		}
		if u == 0 {
			continue
		}
		if math.Log(u) < 0.5*x*x+d*(1.0-v+math.Log(v)) {
			return d * v
		}
	}
}

// sampleStdNormal generates a standard normal variate using the
// Box-Muller transform.
func sampleStdNormal(rng interface{ Float64() float64 }) float64 {
	u1 := rng.Float64()
	u2 := rng.Float64()
	for u1 == 0 {
		u1 = rng.Float64()
	}
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

// EpsilonGreedy selects the arm to pull using the epsilon-greedy strategy.
// With probability epsilon, a random arm is chosen (exploration). With
// probability 1-epsilon, the arm with the highest average reward is chosen
// (exploitation).
//
// Parameters:
//   - rewards: total cumulative reward from each arm
//   - counts: number of times each arm has been pulled
//   - epsilon: exploration rate in [0, 1]
//   - rng: random number generator providing Float64() in [0, 1)
//
// If any arm has count 0, it is treated as having average reward 0.
// Valid range: len(rewards) > 0, epsilon in [0, 1]
// Returns -1 if no arms exist.
//
// Reference: Sutton, R.S. & Barto, A.G. (2018) "Reinforcement Learning:
// An Introduction", 2nd edition, Chapter 2.
func EpsilonGreedy(rewards []float64, counts []int, epsilon float64, rng interface{ Float64() float64 }) int {
	n := len(rewards)
	if n == 0 {
		return -1
	}

	// Explore with probability epsilon.
	if rng.Float64() < epsilon {
		return int(rng.Float64() * float64(n))
	}

	// Exploit: pick the arm with the highest average reward.
	bestArm := 0
	bestAvg := -math.MaxFloat64
	for i := 0; i < n; i++ {
		var avg float64
		if counts[i] > 0 {
			avg = rewards[i] / float64(counts[i])
		}
		if avg > bestAvg {
			bestAvg = avg
			bestArm = i
		}
	}

	return bestArm
}

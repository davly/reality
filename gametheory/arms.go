package gametheory

import "math"

// ---------------------------------------------------------------------------
// Misuse-resistant bandit API
//
// The raw-slice bandit functions (UCB1, ThompsonSampling, EpsilonGreedy)
// take parallel slices whose reward-unit contract — RewardSum is a SUM of
// rewards, never a mean — is enforced only by doc comments. That contract
// has already been violated in production once: a caller passed per-arm
// MEAN rewards where cumulative SUMS were expected, which divides the mean
// by the pull count a second time and collapses the exploitation term to
// near zero, turning UCB1 into a pure count-based explorer.
//
// The Arm type and the *FromArms wrappers below make the unit explicit in
// the field name (RewardSum) so the mistake is visible at the call site.
// New callers should prefer the *FromArms forms; the raw-slice forms remain
// for backward compatibility and are delegated to unchanged.
// ---------------------------------------------------------------------------

// Arm is one bandit arm's observed statistics.
//
//   - Count is the number of times the arm has been pulled.
//   - RewardSum is the CUMULATIVE SUM of all rewards observed from this arm
//     across all Count pulls. It is NEVER a mean/average. The mean reward is
//     RewardSum / Count, and the selection algorithms perform that division
//     themselves — passing a mean here divides by Count twice and silently
//     destroys exploitation behavior.
type Arm struct {
	Count     int
	RewardSum float64
}

// UCB1FromArms selects the arm to pull using the Upper Confidence Bound
// algorithm. It is a thin, misuse-resistant wrapper over UCB1: it unpacks
// arms into the parallel-slice form, computing totalPulls as the sum of all
// Counts, and preserves UCB1's semantics exactly:
//
//   - Returns -1 if arms is empty.
//   - Any arm with Count == 0 is selected first (lowest index wins).
//
// Each Arm.RewardSum must be the cumulative SUM of rewards (see Arm).
func UCB1FromArms(arms []Arm) int {
	counts, rewards, totalPulls := unpackArms(arms)
	return UCB1(counts, rewards, totalPulls)
}

// EpsilonGreedyFromArms selects the arm to pull using the epsilon-greedy
// strategy. It is a thin, misuse-resistant wrapper over EpsilonGreedy and
// preserves its semantics exactly:
//
//   - Returns -1 if arms is empty.
//   - With probability epsilon a random arm is chosen (exploration).
//   - Otherwise the arm with the highest average reward
//     (RewardSum / Count) is chosen; arms with Count == 0 are treated as
//     having average reward 0.
//
// Each Arm.RewardSum must be the cumulative SUM of rewards (see Arm).
func EpsilonGreedyFromArms(arms []Arm, epsilon float64, rng interface{ Float64() float64 }) int {
	counts, rewards, _ := unpackArms(arms)
	return EpsilonGreedy(rewards, counts, epsilon, rng)
}

// ThompsonFromArmsBernoulli selects the arm to pull using Thompson sampling
// for BERNOULLI bandits, i.e. it is only valid when every individual reward
// is in {0, 1}, so that an arm's RewardSum equals its success count.
//
// The mapping it documents and enforces:
//
//	successes_i = round(RewardSum_i), clamped to [0, Count_i]
//	failures_i  = Count_i - successes_i
//
// The clamp guards against accumulated floating-point drift and against
// out-of-contract inputs (negative sums, sums exceeding the pull count)
// producing invalid Beta parameters. It does NOT make the function correct
// for non-Bernoulli rewards — if rewards are continuous or outside {0, 1},
// do not use this function; call ThompsonSampling with a problem-specific
// success/failure mapping instead.
//
// Returns -1 if arms is empty.
func ThompsonFromArmsBernoulli(arms []Arm, rng interface{ Float64() float64 }) int {
	n := len(arms)
	if n == 0 {
		return -1
	}
	successes := make([]int, n)
	failures := make([]int, n)
	for i, a := range arms {
		s := int(math.Round(a.RewardSum))
		if s < 0 {
			s = 0
		}
		if s > a.Count {
			s = a.Count
		}
		successes[i] = s
		failures[i] = a.Count - s
	}
	return ThompsonSampling(successes, failures, rng)
}

// unpackArms converts []Arm into the parallel-slice form used by the raw
// bandit functions, returning (counts, rewardSums, totalPulls) where
// totalPulls is the sum of all Counts.
func unpackArms(arms []Arm) ([]int, []float64, int) {
	counts := make([]int, len(arms))
	rewards := make([]float64, len(arms))
	totalPulls := 0
	for i, a := range arms {
		counts[i] = a.Count
		rewards[i] = a.RewardSum
		totalPulls += a.Count
	}
	return counts, rewards, totalPulls
}

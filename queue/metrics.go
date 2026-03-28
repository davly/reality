package queue

// ---------------------------------------------------------------------------
// Utility Metrics
//
// Auxiliary functions for characterizing arrival patterns and computing
// offered load.
// ---------------------------------------------------------------------------

// BurstinessIndex computes the index of dispersion (coefficient of variation
// squared) of inter-arrival times: C² = Var(X) / E(X)².
//
// Parameters:
//   - interarrivalTimes: observed inter-arrival durations (length >= 2).
//
// Returns the squared coefficient of variation:
//   - C² = 0 for perfectly regular (constant) arrivals
//   - C² = 1 for Poisson (exponential inter-arrivals)
//   - C² > 1 for bursty/overdispersed arrivals
//   - C² < 1 for underdispersed (more regular than Poisson) arrivals
//
// Panics if fewer than 2 samples are provided or the mean is zero.
//
// Reference: Cox, D.R. & Lewis, P.A.W. (1966). "The Statistical Analysis
// of Series of Events". Methuen.
func BurstinessIndex(interarrivalTimes []float64) float64 {
	n := len(interarrivalTimes)
	if n < 2 {
		panic("queue.BurstinessIndex: need at least 2 inter-arrival times")
	}

	// Compute mean.
	sum := 0.0
	for _, t := range interarrivalTimes {
		sum += t
	}
	mean := sum / float64(n)

	if mean == 0 {
		panic("queue.BurstinessIndex: mean inter-arrival time is zero")
	}

	// Compute variance (population variance for the index of dispersion).
	varSum := 0.0
	for _, t := range interarrivalTimes {
		d := t - mean
		varSum += d * d
	}
	variance := varSum / float64(n)

	return variance / (mean * mean)
}

// OfferedLoad computes the offered load in erlangs: A = λ · s.
//
// Parameters:
//   - lambda: arrival rate (λ > 0, arrivals per unit time)
//   - serviceTime: mean service time per request (s > 0)
//
// Returns the offered load in erlangs (dimensionless).
//
// An offered load of A erlangs means A servers are needed on average to
// handle all traffic without queueing. This is the fundamental input to
// Erlang B/C formulas.
//
// Panics if lambda or serviceTime is <= 0.
//
// Reference: Erlang, A.K. (1917).
func OfferedLoad(lambda, serviceTime float64) float64 {
	if lambda <= 0 {
		panic("queue.OfferedLoad: lambda must be positive")
	}
	if serviceTime <= 0 {
		panic("queue.OfferedLoad: serviceTime must be positive")
	}
	return lambda * serviceTime
}

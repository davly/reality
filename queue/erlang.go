package queue

// ---------------------------------------------------------------------------
// Erlang Formulas
//
// The Erlang B and Erlang C formulas are foundational results in teletraffic
// engineering, used to dimension call centers, network links, and service
// desks.
//
// All implementations use numerically stable recursive or iterative
// formulations to avoid factorial overflow.
// ---------------------------------------------------------------------------

// ErlangB computes the Erlang B (blocking) probability — the probability
// that all N servers are busy and an arriving request is lost (no queueing).
//
// This models the M/M/N/N queue (N servers, no waiting room).
//
// Parameters:
//   - A: offered load in erlangs (A = λ/μ, A > 0)
//   - N: number of servers (N >= 1)
//
// Returns the probability of blocking (0 ≤ result ≤ 1).
//
// Uses the Jagerman recursive formula for numerical stability:
//
//	B(A, 0) = 1
//	B(A, n) = (A · B(A, n-1)) / (n + A · B(A, n-1))
//
// This avoids computing factorials or large powers of A directly.
//
// Reference: Erlang, A.K. (1917); Jagerman, D.L. (1974), "Some properties
// of the Erlang loss function", Bell System Technical Journal.
func ErlangB(A float64, N int) float64 {
	if A <= 0 {
		panic("queue.ErlangB: offered load A must be positive")
	}
	if N < 1 {
		panic("queue.ErlangB: N must be at least 1")
	}

	// Jagerman recursion: numerically stable, O(N) time, O(1) space.
	invB := 1.0 // 1/B(A, 0) = 1
	for n := 1; n <= N; n++ {
		invB = 1.0 + float64(n)/A*invB
	}
	return 1.0 / invB
}

// ErlangC computes the Erlang C (wait) probability — the probability that
// an arriving customer must wait because all N servers are busy.
//
// This models the M/M/N queue (N servers, infinite waiting room).
//
// Parameters:
//   - A: offered load in erlangs (A = λ/μ, A > 0, A < N for stability)
//   - N: number of servers (N >= 1)
//
// Returns the probability of waiting (0 ≤ result ≤ 1).
//
// Built on ErlangB for numerical stability:
//
//	C(A, N) = B(A, N) / (1 - (A/N)·(1 - B(A, N)))
//
// where B(A, N) is the Erlang B probability.
//
// Panics if A >= N (unstable system).
//
// Reference: Erlang, A.K. (1917); Gross et al., Chapter 2.
func ErlangC(A float64, N int) float64 {
	if A <= 0 {
		panic("queue.ErlangC: offered load A must be positive")
	}
	if N < 1 {
		panic("queue.ErlangC: N must be at least 1")
	}
	if A >= float64(N) {
		panic("queue.ErlangC: offered load A must be less than N for stability")
	}

	b := ErlangB(A, N)
	rho := A / float64(N)
	return b / (1.0 - rho*(1.0-b))
}

// ErlangCWaitTime computes the expected wait time in an M/M/N queue.
//
// Parameters:
//   - A: offered load in erlangs (A = λ/μ, A > 0, A < N)
//   - N: number of servers (N >= 1)
//   - mu: service rate per server (μ > 0)
//
// Returns the expected waiting time before service begins.
//
// Formula:
//
//	E[Wq] = ErlangC(A, N) / (N·μ - λ) = ErlangC(A, N) / (N·μ·(1 - A/N))
//
// Reference: Gross et al., Chapter 2.
func ErlangCWaitTime(A float64, N int, mu float64) float64 {
	if mu <= 0 {
		panic("queue.ErlangCWaitTime: mu must be positive")
	}

	c := ErlangC(A, N) // validates A and N
	nMu := float64(N) * mu
	rho := A / float64(N)
	return c / (nMu * (1.0 - rho))
}

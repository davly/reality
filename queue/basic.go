// Package queue provides queueing theory models and metrics.
//
// All functions are pure, deterministic, and allocation-free where possible.
// Zero external dependencies. Used by Pulse/Sentinel (service capacity),
// BookaBloke (wait times), Wayfare (attraction queues), and Harvest
// (processing throughput).
//
// Notation follows Kendall's A/S/c/K convention:
//   - λ (lambda) = arrival rate
//   - μ (mu) = service rate per server
//   - c = number of servers
//   - K = system capacity (buffer + servers)
//   - ρ = utilization = λ/(c·μ)
//   - L = expected number in system
//   - Lq = expected number in queue
//   - W = expected time in system
//   - Wq = expected wait time in queue
//
// Reference: Gross, Shortle, Thompson & Harris, "Fundamentals of Queueing
// Theory", 4th edition, Wiley.
package queue

import "math"

// ---------------------------------------------------------------------------
// M/M/1 Queue
//
// Single server, Poisson arrivals, exponential service, infinite capacity.
// The simplest and most commonly used queueing model.
// ---------------------------------------------------------------------------

// MM1 computes steady-state metrics for an M/M/1 queue.
//
// Parameters:
//   - lambda: arrival rate (λ > 0)
//   - mu: service rate (μ > 0, μ > λ for stability)
//
// Returns named results:
//   - Lq: expected queue length (excluding service)
//   - Wq: expected wait time in queue
//   - L: expected number in system (queue + service)
//   - W: expected time in system (wait + service)
//   - rho: server utilization (λ/μ)
//
// Panics if lambda >= mu (system is unstable) or if either rate is <= 0.
//
// Formulas:
//
//	ρ = λ/μ
//	L = ρ/(1-ρ)
//	W = 1/(μ-λ)
//	Lq = ρ²/(1-ρ)
//	Wq = ρ/(μ-λ)
//
// Reference: Gross et al., Chapter 2.
func MM1(lambda, mu float64) (Lq, Wq, L, W, rho float64) {
	if lambda <= 0 {
		panic("queue.MM1: lambda must be positive")
	}
	if mu <= 0 {
		panic("queue.MM1: mu must be positive")
	}
	if lambda >= mu {
		panic("queue.MM1: lambda must be less than mu for stability")
	}

	rho = lambda / mu
	L = rho / (1 - rho)
	W = 1.0 / (mu - lambda)
	Lq = rho * rho / (1 - rho)
	Wq = rho / (mu - lambda)
	return
}

// ---------------------------------------------------------------------------
// M/M/c Queue
//
// c parallel servers, Poisson arrivals, exponential service, infinite capacity.
// Generalizes M/M/1 to multiple servers. Uses the Erlang-C formula internally.
// ---------------------------------------------------------------------------

// MMc computes steady-state metrics for an M/M/c queue.
//
// Parameters:
//   - lambda: arrival rate (λ > 0)
//   - mu: service rate per server (μ > 0)
//   - c: number of servers (c >= 1)
//
// Returns named results:
//   - Lq: expected queue length
//   - Wq: expected wait time in queue
//   - L: expected number in system
//   - W: expected time in system
//   - rho: per-server utilization (λ/(c·μ))
//
// Panics if lambda >= c*mu (system is unstable), c < 1, or rates are <= 0.
//
// Formulas:
//
//	ρ = λ/(c·μ)
//	P(wait) = ErlangC(A, c) where A = λ/μ
//	Lq = P(wait) · ρ / (1-ρ)
//	Wq = Lq / λ  (Little's law)
//	W = Wq + 1/μ
//	L = λ·W  (Little's law)
//
// Reference: Gross et al., Chapter 2; Erlang, A.K. (1917).
func MMc(lambda, mu float64, c int) (Lq, Wq, L, W, rho float64) {
	if lambda <= 0 {
		panic("queue.MMc: lambda must be positive")
	}
	if mu <= 0 {
		panic("queue.MMc: mu must be positive")
	}
	if c < 1 {
		panic("queue.MMc: c must be at least 1")
	}
	cMu := float64(c) * mu
	if lambda >= cMu {
		panic("queue.MMc: lambda must be less than c*mu for stability")
	}

	rho = lambda / cMu
	A := lambda / mu // offered load in erlangs

	// Erlang-C probability that an arriving customer must wait.
	pWait := ErlangC(A, c)

	Lq = pWait * rho / (1 - rho)
	Wq = Lq / lambda
	W = Wq + 1.0/mu
	L = lambda * W
	return
}

// ---------------------------------------------------------------------------
// M/M/1/K Queue
//
// Single server, Poisson arrivals, exponential service, finite capacity K.
// Customers arriving when the system is full are lost (turned away).
// ---------------------------------------------------------------------------

// MM1K computes steady-state metrics for an M/M/1/K queue (finite capacity).
//
// Parameters:
//   - lambda: arrival rate (λ > 0)
//   - mu: service rate (μ > 0)
//   - K: system capacity (K >= 1, includes the server)
//
// Returns named results:
//   - Lq: expected queue length
//   - Wq: expected wait time in queue
//   - L: expected number in system
//   - W: expected time in system
//   - rho: raw load factor (λ/μ, may exceed 1 since finite capacity)
//   - pLoss: probability that an arrival is lost (system full)
//
// Unlike M/M/1, M/M/1/K is stable for any λ and μ since capacity is finite.
// Panics if K < 1 or rates are <= 0.
//
// Formulas (ρ = λ/μ):
//
//	If ρ = 1: P(n) = 1/(K+1) for n = 0,1,...,K
//	If ρ ≠ 1: P(n) = (1-ρ)·ρⁿ / (1-ρ^(K+1))
//	P(loss) = P(K)
//	L = Σ n·P(n)
//	λ_eff = λ·(1 - P(loss))
//	W = L / λ_eff
//	Wq = W - 1/μ
//	Lq = λ_eff · Wq
//
// Reference: Gross et al., Chapter 2.
func MM1K(lambda, mu float64, K int) (Lq, Wq, L, W, rho, pLoss float64) {
	if lambda <= 0 {
		panic("queue.MM1K: lambda must be positive")
	}
	if mu <= 0 {
		panic("queue.MM1K: mu must be positive")
	}
	if K < 1 {
		panic("queue.MM1K: K must be at least 1")
	}

	rho = lambda / mu

	if math.Abs(rho-1.0) < 1e-12 {
		// Special case: ρ = 1 → uniform distribution
		pLoss = 1.0 / float64(K+1)
		L = float64(K) / 2.0
	} else {
		// P(n) = (1-ρ)·ρⁿ / (1-ρ^(K+1))
		rhoK1 := math.Pow(rho, float64(K+1))
		denom := 1.0 - rhoK1

		pLoss = (1.0 - rho) * math.Pow(rho, float64(K)) / denom

		// L = ρ/(1-ρ) - (K+1)·ρ^(K+1)/(1-ρ^(K+1))
		L = rho/(1.0-rho) - float64(K+1)*rhoK1/denom
	}

	lambdaEff := lambda * (1.0 - pLoss)
	if lambdaEff < 1e-300 {
		// Edge case: almost all arrivals lost
		W = 1.0 / mu
		Wq = 0
		Lq = 0
		return
	}
	W = L / lambdaEff
	Wq = W - 1.0/mu
	if Wq < 0 {
		Wq = 0
	}
	Lq = lambdaEff * Wq
	return
}

// ---------------------------------------------------------------------------
// Little's Law
// ---------------------------------------------------------------------------

// LittlesLaw computes the expected time in system using Little's Law: W = L/λ.
//
// Parameters:
//   - L: expected number in system (L > 0)
//   - lambda: arrival rate (λ > 0)
//
// Returns W = L/λ.
//
// Little's Law is a remarkably general result that holds for any stable
// queueing system regardless of the arrival or service distributions.
//
// Reference: Little, J.D.C. (1961). "A Proof for the Queuing Formula: L = λW".
// Operations Research, 9(3), 383-387.
func LittlesLaw(L, lambda float64) float64 {
	if lambda <= 0 {
		panic("queue.LittlesLaw: lambda must be positive")
	}
	return L / lambda
}

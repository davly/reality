package queue

import "math"

// ---------------------------------------------------------------------------
// Jackson Networks
//
// Open Jackson networks are a class of queueing networks where each node is
// an independent M/M/c queue and arrivals/routings are Poisson. The key
// theorem (Jackson, 1957) states that each node can be analyzed independently
// once the effective arrival rates are determined by solving the traffic
// equations.
// ---------------------------------------------------------------------------

// JacksonNetwork computes steady-state metrics for an open Jackson network.
//
// Parameters:
//   - lambdaExt: external arrival rates per node (length n). May be zero for
//     nodes with no external arrivals.
//   - routing: n×n routing probability matrix. routing[i][j] is the probability
//     that a customer leaving node i proceeds to node j. Customers leave the
//     network with probability 1 - Σⱼ routing[i][j].
//   - mu: service rate per server at each node (length n, each > 0).
//   - servers: number of servers at each node (length n, each >= 1).
//
// Returns per-node slices (length n):
//   - throughput: effective arrival rate (λᵢ) at each node.
//   - utilization: per-server utilization (ρᵢ = λᵢ/(cᵢ·μᵢ)) at each node.
//   - queueLength: expected number in system (Lᵢ) at each node.
//
// The traffic equations are solved by fixed-point iteration:
//
//	λᵢ = lambdaExtᵢ + Σⱼ λⱼ · routing[j][i]
//
// Panics if:
//   - slice lengths are inconsistent
//   - any service rate is <= 0
//   - any server count is < 1
//   - traffic equations do not converge (network may be unstable)
//   - any node's utilization >= 1 (node is overloaded)
//
// Reference: Jackson, J.R. (1957). "Networks of Waiting Lines". Operations
// Research, 5(4), 518-521.
func JacksonNetwork(
	lambdaExt []float64,
	routing [][]float64,
	mu []float64,
	servers []int,
) (throughput, utilization, queueLength []float64) {
	n := len(lambdaExt)
	if n == 0 {
		panic("queue.JacksonNetwork: empty network")
	}
	if len(routing) != n {
		panic("queue.JacksonNetwork: routing matrix row count must match node count")
	}
	for i, row := range routing {
		if len(row) != n {
			panic("queue.JacksonNetwork: routing matrix must be square")
		}
		// Validate row sums <= 1
		sum := 0.0
		for _, p := range row {
			if p < 0 {
				panic("queue.JacksonNetwork: routing probabilities must be non-negative")
			}
			sum += p
		}
		if sum > 1.0+1e-10 {
			_ = i
			panic("queue.JacksonNetwork: routing row sum exceeds 1")
		}
	}
	if len(mu) != n {
		panic("queue.JacksonNetwork: mu length must match node count")
	}
	if len(servers) != n {
		panic("queue.JacksonNetwork: servers length must match node count")
	}
	for i := 0; i < n; i++ {
		if mu[i] <= 0 {
			panic("queue.JacksonNetwork: all service rates must be positive")
		}
		if servers[i] < 1 {
			panic("queue.JacksonNetwork: all server counts must be at least 1")
		}
	}

	// Solve traffic equations by fixed-point iteration.
	// λᵢ = lambdaExtᵢ + Σⱼ λⱼ · routing[j][i]
	lambda := make([]float64, n)
	copy(lambda, lambdaExt)

	const maxIter = 1000
	const tol = 1e-12

	for iter := 0; iter < maxIter; iter++ {
		maxDiff := 0.0
		for i := 0; i < n; i++ {
			newLambda := lambdaExt[i]
			for j := 0; j < n; j++ {
				newLambda += lambda[j] * routing[j][i]
			}
			diff := math.Abs(newLambda - lambda[i])
			if diff > maxDiff {
				maxDiff = diff
			}
			lambda[i] = newLambda
		}
		if maxDiff < tol {
			goto converged
		}
	}
	panic("queue.JacksonNetwork: traffic equations did not converge (network may be unstable)")

converged:
	// Compute per-node metrics using M/M/c analysis.
	throughput = make([]float64, n)
	utilization = make([]float64, n)
	queueLength = make([]float64, n)

	for i := 0; i < n; i++ {
		throughput[i] = lambda[i]
		cMu := float64(servers[i]) * mu[i]
		utilization[i] = lambda[i] / cMu

		if utilization[i] >= 1.0 {
			panic("queue.JacksonNetwork: node is overloaded (utilization >= 1)")
		}

		if lambda[i] < 1e-300 {
			// No traffic at this node.
			queueLength[i] = 0
			continue
		}

		if servers[i] == 1 {
			// M/M/1 formula (faster, avoids Erlang-C).
			rho := utilization[i]
			queueLength[i] = rho / (1.0 - rho)
		} else {
			// M/M/c formula via Erlang-C.
			A := lambda[i] / mu[i]
			rho := utilization[i]
			pWait := ErlangC(A, servers[i])
			Lq := pWait * rho / (1.0 - rho)
			queueLength[i] = Lq + A // L = Lq + λ/μ
		}
	}
	return
}

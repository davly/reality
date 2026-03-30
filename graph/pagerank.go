package graph

// PageRank computes the PageRank score of every node in a directed graph
// using the standard power iteration method with damping factor.
//
// Parameters:
//   - n: number of nodes (0 to n-1).
//   - edges: list of directed weighted edges as (from, to, weight) triples.
//     Edge weights are used to determine link strength: the contribution of
//     node u to node v is proportional to w(u,v) / sum(w(u,*)).
//     For unweighted graphs, set all weights to 1.0.
//   - damping: probability of following a link (typically 0.85).
//     Must be in [0, 1]. A value of 0 gives uniform rank; a value of 1
//     gives pure link-following with no random jump.
//   - iterations: number of power iteration rounds. More iterations give
//     higher precision. 20-100 is typical.
//
// Returns:
//   - ranks: PageRank score for each node. Scores sum to 1.0 (within
//     floating-point tolerance). ranks[i] is the stationary probability
//     of being at node i after a random walk.
//
// Dangling nodes (out-degree 0) redistribute their rank uniformly across
// all nodes, following the standard treatment from Brin & Page (1998).
//
// Time complexity: O(iterations * (V + E)).
// Space complexity: O(V + E).
//
// Reference: Brin, S. & Page, L. (1998) "The Anatomy of a Large-Scale
// Hypertextual Web Search Engine"
func PageRank(n int, edges [][3]float64, damping float64, iterations int) []float64 {
	if n <= 0 {
		return nil
	}
	if iterations <= 0 {
		iterations = 1
	}

	// Clamp damping to valid range.
	if damping < 0 {
		damping = 0
	}
	if damping > 1 {
		damping = 1
	}

	// Build weighted adjacency list and compute out-weight sums.
	type wEdge struct {
		to     int
		weight float64
	}
	adj := make([][]wEdge, n)
	outWeight := make([]float64, n)

	for _, e := range edges {
		u, v, w := int(e[0]), int(e[1]), e[2]
		if u < 0 || u >= n || v < 0 || v >= n {
			continue
		}
		if w <= 0 {
			continue // ignore non-positive weights
		}
		adj[u] = append(adj[u], wEdge{v, w})
		outWeight[u] += w
	}

	// Initialize ranks uniformly.
	ranks := make([]float64, n)
	uniform := 1.0 / float64(n)
	for i := range ranks {
		ranks[i] = uniform
	}

	newRanks := make([]float64, n)
	teleport := (1.0 - damping) / float64(n)

	for iter := 0; iter < iterations; iter++ {
		// Compute dangling node contribution: sum of ranks of nodes with
		// no outgoing edges, redistributed uniformly.
		danglingSum := 0.0
		for i := 0; i < n; i++ {
			if len(adj[i]) == 0 {
				danglingSum += ranks[i]
			}
		}
		danglingContrib := damping * danglingSum / float64(n)

		// Reset new ranks with teleportation + dangling contribution.
		for i := range newRanks {
			newRanks[i] = teleport + danglingContrib
		}

		// Distribute rank along edges.
		for u := 0; u < n; u++ {
			if len(adj[u]) == 0 || outWeight[u] == 0 {
				continue
			}
			contrib := damping * ranks[u] / outWeight[u]
			for _, e := range adj[u] {
				newRanks[e.to] += contrib * e.weight
			}
		}

		// Swap for next iteration.
		ranks, newRanks = newRanks, ranks
	}

	return ranks
}

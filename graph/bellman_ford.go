package graph

import "math"

// BellmanFord computes single-source shortest paths from source to all
// reachable nodes in a weighted directed graph using the Bellman-Ford
// algorithm. Unlike Dijkstra, this algorithm correctly handles negative
// edge weights.
//
// Parameters:
//   - n: number of nodes (0 to n-1).
//   - edges: list of directed weighted edges as (from, to, weight) triples.
//     Negative weights are allowed.
//   - source: the starting node index (must be in [0, n)).
//
// Returns:
//   - dist: shortest distance from source to each node. dist[i] == +Inf
//     means node i is unreachable from source. dist[i] == -Inf means
//     node i is reachable via a negative-weight cycle.
//   - prev: predecessor array for path reconstruction. prev[i] == -1 means
//     node i has no predecessor (either source or unreachable).
//   - hasNegCycle: true if the graph contains a negative-weight cycle
//     reachable from source.
//
// Time complexity: O(V * E).
// Space complexity: O(V + E).
//
// Reference: Bellman, R. (1958) "On a Routing Problem"; Ford, L.R. Jr.
// (1956) "Network Flow Theory"
func BellmanFord(n int, edges [][3]float64, source int) (dist []float64, prev []int, hasNegCycle bool) {
	dist = make([]float64, n)
	prev = make([]int, n)
	for i := range dist {
		dist[i] = math.Inf(1)
		prev[i] = -1
	}
	if source < 0 || source >= n {
		return dist, prev, false
	}
	dist[source] = 0

	// Relax edges V-1 times.
	for round := 0; round < n-1; round++ {
		updated := false
		for _, e := range edges {
			u, v, w := int(e[0]), int(e[1]), e[2]
			if u < 0 || u >= n || v < 0 || v >= n {
				continue
			}
			if dist[u]+w < dist[v] {
				dist[v] = dist[u] + w
				prev[v] = u
				updated = true
			}
		}
		if !updated {
			break // early termination — no relaxation occurred
		}
	}

	// Check for negative-weight cycles: one more relaxation pass.
	// Any node that can still be relaxed is reachable from a negative cycle.
	for _, e := range edges {
		u, v, w := int(e[0]), int(e[1]), e[2]
		if u < 0 || u >= n || v < 0 || v >= n {
			continue
		}
		if dist[u]+w < dist[v] {
			hasNegCycle = true
			dist[v] = math.Inf(-1)
		}
	}

	return dist, prev, hasNegCycle
}

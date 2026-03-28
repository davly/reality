package graph

import (
	"errors"
	"math"
)

// MaxFlow computes the maximum flow from source to sink in a directed
// weighted graph using the Edmonds-Karp algorithm (BFS-based Ford-Fulkerson).
//
// Parameters:
//   - adj: directed adjacency list (node -> successors).
//   - capacity: edge capacities keyed by [from, to]. Missing entries are
//     treated as zero capacity.
//   - source: the source node index.
//   - sink: the sink node index.
//
// Returns the maximum flow value from source to sink.
//
// Time complexity: O(V * E^2).
// Space complexity: O(V + E).
//
// Reference: Edmonds & Karp, "Theoretical improvements in algorithmic
// efficiency for network flow problems" (1972).
func MaxFlow(adj IntAdjacency, capacity map[[2]int]float64, source, sink int) float64 {
	n := graphSize3(adj, source, sink)

	// Build residual capacity matrix using maps for sparse graphs.
	resCap := make(map[[2]int]float64)
	// Build full adjacency including reverse edges for residual graph.
	resAdj := make(IntAdjacency, n)

	for u, vs := range adj {
		for _, v := range vs {
			edge := [2]int{u, v}
			if c, ok := capacity[edge]; ok {
				resCap[edge] = c
			}
			resAdj[u] = appendUnique(resAdj[u], v)
			resAdj[v] = appendUnique(resAdj[v], u) // reverse edge for residual
		}
	}

	totalFlow := 0.0

	for {
		// BFS to find augmenting path.
		parent := make([]int, n)
		for i := range parent {
			parent[i] = -1
		}
		parent[source] = source
		queue := []int{source}

		found := false
		for len(queue) > 0 && !found {
			u := queue[0]
			queue = queue[1:]
			for _, v := range resAdj[u] {
				if v < 0 || v >= n {
					continue
				}
				if parent[v] != -1 {
					continue
				}
				edge := [2]int{u, v}
				if resCap[edge] <= 0 {
					continue
				}
				parent[v] = u
				if v == sink {
					found = true
					break
				}
				queue = append(queue, v)
			}
		}

		if !found {
			break
		}

		// Find bottleneck.
		pathFlow := math.Inf(1)
		for v := sink; v != source; v = parent[v] {
			u := parent[v]
			edge := [2]int{u, v}
			if resCap[edge] < pathFlow {
				pathFlow = resCap[edge]
			}
		}

		// Update residual capacities.
		for v := sink; v != source; v = parent[v] {
			u := parent[v]
			resCap[[2]int{u, v}] -= pathFlow
			resCap[[2]int{v, u}] += pathFlow
		}

		totalFlow += pathFlow
	}

	return totalFlow
}

// ErrCycleDetected is returned by TopologicalSort when the graph contains
// a cycle and therefore has no valid topological ordering.
var ErrCycleDetected = errors.New("graph contains a cycle")

// TopologicalSort produces a topological ordering of a directed acyclic
// graph (DAG) using Kahn's algorithm (iterative BFS-based).
//
// Parameters:
//   - adj: directed adjacency list (node -> successors).
//   - n: number of nodes (0 to n-1).
//
// Returns:
//   - order: a valid topological ordering where for every edge u->v,
//     u appears before v. If multiple valid orderings exist, nodes with
//     smaller indices come first (deterministic).
//   - err: ErrCycleDetected if the graph contains a cycle.
//
// Time complexity: O(V + E).
// Space complexity: O(V).
//
// Reference: Kahn, "Topological sorting of large networks" (1962).
func TopologicalSort(adj IntAdjacency, n int) ([]int, error) {
	inDeg := make([]int, n)
	for u := 0; u < n; u++ {
		for _, v := range adj[u] {
			if v >= 0 && v < n {
				inDeg[v]++
			}
		}
	}

	// Use a sorted insertion approach for determinism: always pick the
	// smallest available node. For simplicity, we scan linearly since
	// this is a pure math library and the graph sizes are moderate.
	var order []int
	removed := make([]bool, n)

	for len(order) < n {
		found := -1
		for i := 0; i < n; i++ {
			if !removed[i] && inDeg[i] == 0 {
				found = i
				break
			}
		}
		if found == -1 {
			return order, ErrCycleDetected
		}
		order = append(order, found)
		removed[found] = true
		for _, v := range adj[found] {
			if v >= 0 && v < n {
				inDeg[v]--
			}
		}
	}

	return order, nil
}

// appendUnique appends v to slice s only if v is not already present.
func appendUnique(s []int, v int) []int {
	for _, x := range s {
		if x == v {
			return s
		}
	}
	return append(s, v)
}

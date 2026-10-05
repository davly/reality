package graph

import (
	"container/heap"
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

	// Determinism: always take the smallest available node. A min-heap of the
	// zero-in-degree nodes gives exactly that order in O((V + E) log V); a
	// linear scan for the smallest such node, used here before, was O(V^2).
	ready := make(minIntHeap, 0, n)
	for i := 0; i < n; i++ {
		if inDeg[i] == 0 {
			ready = append(ready, i)
		}
	}
	heap.Init(&ready)

	var order []int
	for len(order) < n {
		if ready.Len() == 0 {
			return order, ErrCycleDetected
		}
		found := heap.Pop(&ready).(int)
		order = append(order, found)
		for _, v := range adj[found] {
			if v >= 0 && v < n {
				inDeg[v]--
				if inDeg[v] == 0 {
					heap.Push(&ready, v)
				}
			}
		}
	}

	return order, nil
}

// minIntHeap is a container/heap of node indices, smallest first.
type minIntHeap []int

func (h minIntHeap) Len() int           { return len(h) }
func (h minIntHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h minIntHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minIntHeap) Push(x any)        { *h = append(*h, x.(int)) }
func (h *minIntHeap) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
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

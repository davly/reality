package graph

import "math"

// BetweennessCentrality computes the betweenness centrality of every node
// in an unweighted directed graph using Brandes' algorithm.
//
// Betweenness centrality measures how often a node lies on shortest paths
// between other pairs of nodes. A high betweenness score indicates a node
// that serves as a bridge or broker between different parts of the graph.
//
// Parameters:
//   - adj: directed adjacency list (node -> successors).
//   - n: number of nodes (0 to n-1).
//
// Returns a slice of length n where result[i] is the betweenness centrality
// of node i. The values are NOT normalized (they represent raw shortest-path
// counts). For normalization, divide by (n-1)*(n-2) for directed graphs.
//
// Time complexity: O(V * E) — one BFS per node.
// Space complexity: O(V + E).
//
// Reference: Brandes, "A Faster Algorithm for Betweenness Centrality" (2001).
func BetweennessCentrality(adj IntAdjacency, n int) []float64 {
	cb := make([]float64, n)

	for s := 0; s < n; s++ {
		// BFS from s.
		stack := make([]int, 0, n)
		pred := make([][]int, n)
		sigma := make([]float64, n) // number of shortest paths
		dist := make([]int, n)
		for i := range dist {
			dist[i] = -1
		}
		sigma[s] = 1
		dist[s] = 0

		queue := []int{s}
		for len(queue) > 0 {
			v := queue[0]
			queue = queue[1:]
			stack = append(stack, v)

			for _, w := range adj[v] {
				if w < 0 || w >= n {
					continue
				}
				// First visit to w?
				if dist[w] < 0 {
					dist[w] = dist[v] + 1
					queue = append(queue, w)
				}
				// Shortest path to w via v?
				if dist[w] == dist[v]+1 {
					sigma[w] += sigma[v]
					pred[w] = append(pred[w], v)
				}
			}
		}

		// Back-propagation of dependencies.
		delta := make([]float64, n)
		for len(stack) > 0 {
			w := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			for _, v := range pred[w] {
				delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
			}
			if w != s {
				cb[w] += delta[w]
			}
		}
	}

	return cb
}

// EigenvectorCentrality computes the eigenvector centrality of every node
// using power iteration. This produces scores proportional to the leading
// eigenvector of the (weighted) adjacency matrix, similar to PageRank
// without the damping factor.
//
// Parameters:
//   - adj: directed adjacency list. Edges point from key to values.
//   - weights: optional edge weights. If nil, all edges have weight 1.
//     Missing entries in the map for edges that exist in adj are treated as
//     weight 1.
//   - n: number of nodes (0 to n-1).
//   - maxIter: maximum number of power iterations. 100 is typical.
//
// Returns a slice of length n with centrality scores. The scores are
// L2-normalized (sum of squares = 1). Convergence is declared when the
// L2 change between iterations is less than 1e-10.
//
// Time complexity: O(maxIter * (V + E)).
func EigenvectorCentrality(adj IntAdjacency, weights map[[2]int]float64, n, maxIter int) []float64 {
	x := make([]float64, n)
	// Initialize uniformly.
	init := 1.0 / math.Sqrt(float64(n))
	for i := range x {
		x[i] = init
	}

	for iter := 0; iter < maxIter; iter++ {
		xNew := make([]float64, n)
		for u := 0; u < n; u++ {
			for _, v := range adj[u] {
				if v < 0 || v >= n {
					continue
				}
				w := 1.0
				if weights != nil {
					if ww, ok := weights[[2]int{u, v}]; ok {
						w = ww
					}
				}
				// In eigenvector centrality, a node's score is the sum of
				// its neighbors' scores. For directed graphs, we accumulate
				// incoming influence: u -> v means v gets u's score.
				xNew[v] += w * x[u]
			}
		}

		// L2 normalize.
		norm := 0.0
		for _, val := range xNew {
			norm += val * val
		}
		norm = math.Sqrt(norm)
		if norm == 0 {
			return x // graph has no edges or all zeros
		}
		for i := range xNew {
			xNew[i] /= norm
		}

		// Check convergence.
		diff := 0.0
		for i := range x {
			d := xNew[i] - x[i]
			diff += d * d
		}
		x = xNew
		if math.Sqrt(diff) < 1e-10 {
			break
		}
	}

	return x
}

// DegreeCentrality computes the degree centrality of every node in a
// directed graph. Degree centrality is defined as the total degree
// (in-degree + out-degree) divided by (n-1), where n is the number of
// nodes.
//
// For a single node graph, all centralities are 0.
//
// Parameters:
//   - adj: directed adjacency list.
//   - n: number of nodes (0 to n-1).
//
// Returns a slice of length n where result[i] is the degree centrality
// of node i in [0, 2] (can exceed 1 for directed graphs because a node
// can have both in-edges and out-edges).
//
// Time complexity: O(V + E).
func DegreeCentrality(adj IntAdjacency, n int) []float64 {
	if n <= 1 {
		return make([]float64, n)
	}

	degree := make([]int, n)
	for u, vs := range adj {
		if u < 0 || u >= n {
			continue
		}
		degree[u] += len(vs) // out-degree
		for _, v := range vs {
			if v >= 0 && v < n {
				degree[v]++ // in-degree
			}
		}
	}

	result := make([]float64, n)
	denom := float64(n - 1)
	for i, d := range degree {
		result[i] = float64(d) / denom
	}

	return result
}

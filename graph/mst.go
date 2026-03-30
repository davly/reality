package graph

import "sort"

// KruskalMST computes the minimum spanning tree (or forest) of a weighted
// undirected graph using Kruskal's algorithm with union-find.
//
// The algorithm greedily selects edges in order of increasing weight,
// adding each edge to the MST if it does not create a cycle (checked
// via a disjoint-set / union-find data structure with path compression
// and union by rank).
//
// Parameters:
//   - n: number of nodes (0 to n-1).
//   - edges: list of undirected weighted edges as (u, v, weight) triples.
//     Each edge is considered in both directions. Duplicate edges are
//     handled correctly (lowest weight wins).
//
// Returns:
//   - mstEdges: the edges in the MST as [from, to, weight] triples,
//     sorted by weight ascending. For a connected graph with n nodes,
//     this will contain exactly n-1 edges.
//   - totalWeight: the sum of all edge weights in the MST.
//
// If the graph is disconnected, the result is a minimum spanning forest
// (one tree per connected component) and totalWeight is the sum across
// all components.
//
// Time complexity: O(E log E) dominated by sorting edges.
// Space complexity: O(V + E).
//
// Reference: Kruskal, J.B. (1956) "On the Shortest Spanning Subtree of a
// Graph and the Traveling Salesman Problem"
func KruskalMST(n int, edges [][3]float64) (mstEdges [][3]float64, totalWeight float64) {
	if n <= 0 || len(edges) == 0 {
		return nil, 0
	}

	// Copy and sort edges by weight ascending.
	sorted := make([][3]float64, len(edges))
	copy(sorted, edges)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i][2] < sorted[j][2]
	})

	// Union-find with path compression and union by rank.
	parent := make([]int, n)
	rank := make([]int, n)
	for i := range parent {
		parent[i] = i
	}

	var find func(int) int
	find = func(x int) int {
		if parent[x] != x {
			parent[x] = find(parent[x]) // path compression
		}
		return parent[x]
	}

	union := func(x, y int) bool {
		rx, ry := find(x), find(y)
		if rx == ry {
			return false // already in same set
		}
		// Union by rank.
		if rank[rx] < rank[ry] {
			parent[rx] = ry
		} else if rank[rx] > rank[ry] {
			parent[ry] = rx
		} else {
			parent[ry] = rx
			rank[rx]++
		}
		return true
	}

	for _, e := range sorted {
		u, v, w := int(e[0]), int(e[1]), e[2]
		if u < 0 || u >= n || v < 0 || v >= n {
			continue
		}
		if union(u, v) {
			mstEdges = append(mstEdges, [3]float64{float64(u), float64(v), w})
			totalWeight += w
		}
	}

	return mstEdges, totalWeight
}

// PrimMST computes the minimum spanning tree of a weighted undirected graph
// starting from node 0 using Prim's algorithm with a simple priority scan.
//
// Parameters:
//   - n: number of nodes (0 to n-1).
//   - edges: list of undirected weighted edges as (u, v, weight) triples.
//
// Returns:
//   - mstEdges: the edges in the MST as [from, to, weight] triples.
//     For a connected graph this contains n-1 edges. For disconnected
//     graphs, only the component containing node 0 is spanned.
//   - totalWeight: the sum of all edge weights in the MST.
//
// Time complexity: O(V^2) with the linear scan (suitable for dense graphs).
// Space complexity: O(V + E).
//
// Reference: Prim, R.C. (1957) "Shortest Connection Networks and Some
// Generalizations"
func PrimMST(n int, edges [][3]float64) (mstEdges [][3]float64, totalWeight float64) {
	if n <= 0 || len(edges) == 0 {
		return nil, 0
	}

	const inf = 1e308

	// Build undirected adjacency list.
	type wEdge struct {
		to     int
		weight float64
	}
	adj := make([][]wEdge, n)
	for _, e := range edges {
		u, v, w := int(e[0]), int(e[1]), e[2]
		if u < 0 || u >= n || v < 0 || v >= n {
			continue
		}
		adj[u] = append(adj[u], wEdge{v, w})
		adj[v] = append(adj[v], wEdge{u, w})
	}

	inMST := make([]bool, n)
	key := make([]float64, n)   // minimum weight edge connecting node to MST
	from := make([]int, n)      // the MST node this edge comes from
	for i := range key {
		key[i] = inf
		from[i] = -1
	}
	key[0] = 0

	for count := 0; count < n; count++ {
		// Pick the non-MST node with smallest key.
		u := -1
		for i := 0; i < n; i++ {
			if !inMST[i] && (u == -1 || key[i] < key[u]) {
				u = i
			}
		}
		if u == -1 || key[u] >= inf {
			break // remaining nodes unreachable
		}

		inMST[u] = true
		if from[u] != -1 {
			mstEdges = append(mstEdges, [3]float64{float64(from[u]), float64(u), key[u]})
			totalWeight += key[u]
		}

		// Update keys for neighbors.
		for _, e := range adj[u] {
			if !inMST[e.to] && e.weight < key[e.to] {
				key[e.to] = e.weight
				from[e.to] = u
			}
		}
	}

	return mstEdges, totalWeight
}

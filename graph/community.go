package graph

// ConnectedComponents finds all connected components of an undirected graph
// using BFS. The graph is treated as undirected: for every edge u->v in adj,
// the reverse edge v->u is also considered.
//
// Parameters:
//   - adj: adjacency list (directed edges that will be treated as undirected).
//   - n: number of nodes (0 to n-1).
//
// Returns a slice of components, where each component is a sorted slice of
// node indices. Components are sorted by their smallest node index.
//
// Time complexity: O(V + E).
func ConnectedComponents(adj IntAdjacency, n int) [][]int {
	visited := make([]bool, n)
	var components [][]int

	// Build undirected adjacency.
	undirected := make(IntAdjacency, n)
	for u, vs := range adj {
		for _, v := range vs {
			if u >= 0 && u < n && v >= 0 && v < n {
				undirected[u] = append(undirected[u], v)
				undirected[v] = append(undirected[v], u)
			}
		}
	}

	for i := 0; i < n; i++ {
		if visited[i] {
			continue
		}
		// BFS from node i.
		var comp []int
		queue := []int{i}
		visited[i] = true
		for len(queue) > 0 {
			u := queue[0]
			queue = queue[1:]
			comp = append(comp, u)
			for _, v := range undirected[u] {
				if !visited[v] {
					visited[v] = true
					queue = append(queue, v)
				}
			}
		}
		// Sort component for deterministic output.
		sortInts(comp)
		components = append(components, comp)
	}

	return components
}

// StronglyConnected finds all strongly connected components of a directed
// graph using Tarjan's algorithm.
//
// A strongly connected component (SCC) is a maximal set of nodes such that
// every node is reachable from every other node in the set.
//
// Parameters:
//   - adj: directed adjacency list.
//   - n: number of nodes (0 to n-1).
//
// Returns a slice of SCCs, where each SCC is a sorted slice of node indices.
// SCCs are returned in reverse topological order (sinks first).
//
// Time complexity: O(V + E).
//
// Reference: Tarjan, "Depth-first search and linear graph algorithms" (1972).
func StronglyConnected(adj IntAdjacency, n int) [][]int {
	index := make([]int, n)
	lowlink := make([]int, n)
	onStack := make([]bool, n)
	defined := make([]bool, n)

	var stack []int
	var sccs [][]int
	idx := 0

	var strongConnect func(v int)
	strongConnect = func(v int) {
		index[v] = idx
		lowlink[v] = idx
		defined[v] = true
		idx++
		stack = append(stack, v)
		onStack[v] = true

		for _, w := range adj[v] {
			if w < 0 || w >= n {
				continue
			}
			if !defined[w] {
				strongConnect(w)
				if lowlink[w] < lowlink[v] {
					lowlink[v] = lowlink[w]
				}
			} else if onStack[w] {
				if index[w] < lowlink[v] {
					lowlink[v] = index[w]
				}
			}
		}

		// Root of an SCC?
		if lowlink[v] == index[v] {
			var scc []int
			for {
				w := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				onStack[w] = false
				scc = append(scc, w)
				if w == v {
					break
				}
			}
			sortInts(scc)
			sccs = append(sccs, scc)
		}
	}

	for i := 0; i < n; i++ {
		if !defined[i] {
			strongConnect(i)
		}
	}

	return sccs
}

// LouvainCommunities detects communities in a weighted undirected graph
// using a simplified Louvain modularity optimization algorithm.
//
// The algorithm greedily moves nodes to the community of their neighbor
// that yields the largest gain in modularity. It iterates until no
// improvement is possible.
//
// Parameters:
//   - adj: adjacency list (treated as undirected).
//   - weights: edge weights keyed by [from, to]. If nil or if an edge is
//     missing from the map, weight 1.0 is assumed.
//   - n: number of nodes (0 to n-1).
//
// Returns a slice of length n where result[i] is the community label (an
// integer) assigned to node i. Community labels are contiguous starting
// from 0.
//
// Time complexity: O(iterations * E). Typically converges in a few passes.
//
// Reference: Blondel et al., "Fast unfolding of communities in large
// networks" (2008).
func LouvainCommunities(adj IntAdjacency, weights map[[2]int]float64, n int) []int {
	if n == 0 {
		return nil
	}

	// Build symmetric weighted adjacency.
	type neighbor struct {
		node   int
		weight float64
	}
	neighbors := make([][]neighbor, n)

	edgeWeight := func(u, v int) float64 {
		if weights != nil {
			if w, ok := weights[[2]int{u, v}]; ok {
				return w
			}
			if w, ok := weights[[2]int{v, u}]; ok {
				return w
			}
		}
		return 1.0
	}

	totalWeight := 0.0
	seen := make(map[[2]int]bool)
	for u, vs := range adj {
		for _, v := range vs {
			if u < 0 || u >= n || v < 0 || v >= n {
				continue
			}
			key := [2]int{u, v}
			rev := [2]int{v, u}
			if seen[key] || seen[rev] {
				continue
			}
			seen[key] = true
			w := edgeWeight(u, v)
			neighbors[u] = append(neighbors[u], neighbor{v, w})
			neighbors[v] = append(neighbors[v], neighbor{u, w})
			totalWeight += w
		}
	}

	if totalWeight == 0 {
		// No edges: each node is its own community.
		comm := make([]int, n)
		for i := range comm {
			comm[i] = i
		}
		return comm
	}

	m2 := 2.0 * totalWeight // 2m in the modularity formula.

	// Initialize: each node in its own community.
	comm := make([]int, n)
	for i := range comm {
		comm[i] = i
	}

	// Sigma_tot[c] = sum of weights of edges incident to nodes in community c.
	sigmaTot := make([]float64, n)
	// k[i] = weighted degree of node i.
	ki := make([]float64, n)
	for i := 0; i < n; i++ {
		s := 0.0
		for _, nb := range neighbors[i] {
			s += nb.weight
		}
		ki[i] = s
		sigmaTot[i] = s
	}

	improved := true
	for improved {
		improved = false
		for i := 0; i < n; i++ {
			bestComm := comm[i]
			bestGain := 0.0

			// Sum of weights from i to each neighboring community.
			commWeights := make(map[int]float64)
			for _, nb := range neighbors[i] {
				commWeights[comm[nb.node]] += nb.weight
			}

			// Remove i from its current community.
			oldComm := comm[i]
			sigmaTot[oldComm] -= ki[i]

			for c, kiIn := range commWeights {
				// Modularity gain of moving i to community c.
				gain := kiIn - sigmaTot[c]*ki[i]/m2
				if gain > bestGain {
					bestGain = gain
					bestComm = c
				}
			}

			// Move i to best community.
			comm[i] = bestComm
			sigmaTot[bestComm] += ki[i]

			if bestComm != oldComm {
				improved = true
			}
		}
	}

	// Relabel communities to contiguous 0-based integers.
	labelMap := make(map[int]int)
	nextLabel := 0
	for i := range comm {
		if _, ok := labelMap[comm[i]]; !ok {
			labelMap[comm[i]] = nextLabel
			nextLabel++
		}
		comm[i] = labelMap[comm[i]]
	}

	return comm
}

// sortInts sorts a slice of ints in ascending order using insertion sort.
// This avoids importing sort for a trivial operation on small slices
// (SCC components are typically small).
func sortInts(a []int) {
	for i := 1; i < len(a); i++ {
		key := a[i]
		j := i - 1
		for j >= 0 && a[j] > key {
			a[j+1] = a[j]
			j--
		}
		a[j+1] = key
	}
}

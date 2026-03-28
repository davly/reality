package graph

// BFSDownstream returns all nodes reachable from a given start node via
// breadth-first search, excluding the start node itself.
//
// The start node acts as if it were removed: traversal begins from its
// direct successors. This answers the question "what breaks downstream
// if this node is removed?"
//
// edges: list of directed [from, to] pairs.
// start: the origin node (excluded from result).
//
// Returns a slice of all downstream reachable nodes in BFS order.
// Returns nil if start has no outgoing edges or edges is empty.
//
// Time complexity: O(|V| + |E|).
//
// Source: extracted from aicore/causalmath.BrokenChainBFS.
func BFSDownstream(edges []Edge, start string) []string {
	adj := AdjacencyList(edges)

	visited := make(map[string]bool)
	queue := adj[start]
	var result []string

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		if visited[curr] || curr == start {
			continue
		}
		visited[curr] = true
		result = append(result, curr)
		queue = append(queue, adj[curr]...)
	}

	return result
}

// BFSReachable returns the set of all nodes reachable from any of the
// given start nodes, including the start nodes themselves (if they have
// edges or appear as nodes).
//
// If exclude is non-empty, that node is treated as removed from the graph:
// traversal will not visit it and will not follow its outgoing edges.
//
// Time complexity: O(|V| + |E|).
func BFSReachable(edges []Edge, starts []string, exclude string) map[string]struct{} {
	adj := AdjacencyList(edges)
	visited := make(map[string]struct{})

	queue := make([]string, 0, len(starts))
	for _, s := range starts {
		if s != exclude {
			queue = append(queue, s)
		}
	}

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		if curr == exclude {
			continue
		}
		if _, seen := visited[curr]; seen {
			continue
		}
		visited[curr] = struct{}{}
		queue = append(queue, adj[curr]...)
	}

	return visited
}

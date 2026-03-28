package graph

// DAGDepth computes the longest path length in a directed acyclic graph
// using a topological (Kahn's algorithm) traversal.
//
// The depth of a DAG is the length (in edges) of the longest path from
// any root to any reachable node. Returns 0 for an empty graph or a
// graph with only isolated nodes.
//
// Precondition: the graph must be a DAG. If cycles exist, the behavior
// is undefined (the function will terminate but the result may be incorrect).
//
// Time complexity: O(|V| + |E|).
//
// Source: extracted from aicore/causalmath.computeDepth.
func DAGDepth(edges []Edge) int {
	adj := AdjacencyList(edges)
	inDeg := InDegree(edges)

	dist := make(map[string]int)
	queue := make([]string, 0)

	// Start from all roots (in-degree 0).
	for node, d := range inDeg {
		if d == 0 {
			queue = append(queue, node)
			dist[node] = 0
		}
	}

	// Working copy of in-degrees for Kahn's algorithm.
	deg := make(map[string]int, len(inDeg))
	for k, v := range inDeg {
		deg[k] = v
	}

	maxDepth := 0
	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		for _, next := range adj[curr] {
			if dist[curr]+1 > dist[next] {
				dist[next] = dist[curr] + 1
			}
			if dist[next] > maxDepth {
				maxDepth = dist[next]
			}
			deg[next]--
			if deg[next] == 0 {
				queue = append(queue, next)
			}
		}
	}

	return maxDepth
}

// ReachableLeaves returns which leaf nodes are reachable from the given
// root nodes via DFS traversal. A leaf is any node with out-degree 0.
//
// If exclude is non-empty, that node is treated as removed from the graph.
// Pass "" to include all nodes.
//
// This function is useful for computing node importance: compare the set
// of reachable leaves with and without a node to measure its impact.
//
// Time complexity: O(|V| + |E|).
//
// Source: extracted from aicore/causalmath.reachableLeaves.
func ReachableLeaves(edges []Edge, roots []string, exclude string) map[string]struct{} {
	adj := AdjacencyList(edges)
	leafSet := Leaves(edges)

	reached := make(map[string]struct{})
	visited := make(map[string]bool)

	var dfs func(node string)
	dfs = func(node string) {
		if node == exclude || visited[node] {
			return
		}
		visited[node] = true
		if _, isLeaf := leafSet[node]; isLeaf {
			reached[node] = struct{}{}
		}
		for _, next := range adj[node] {
			dfs(next)
		}
	}

	for _, root := range roots {
		dfs(root)
	}
	return reached
}

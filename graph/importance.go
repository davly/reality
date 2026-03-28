package graph

// NodeImportance computes the importance of every node in a directed graph.
// Importance is defined as the fraction of leaf nodes that become unreachable
// from root nodes when the given node is removed.
//
// For each node n: importance(n) = |leavesLost(n)| / |totalReachableLeaves|
// where leavesLost(n) is the set of leaves reachable with all nodes present
// minus the set reachable when n is removed.
//
// Returns a map from node name to importance score in [0, 1].
// A score of 1.0 means removing this node disconnects all leaves from all roots.
// A score of 0.0 means removing this node has no impact on leaf reachability.
//
// Time complexity: O(|V| * (|V| + |E|)) — one DFS per node.
//
// Source: extracted from aicore/causalmath.CausalImportance.
func NodeImportance(edges []Edge) map[string]float64 {
	if len(edges) == 0 {
		return map[string]float64{}
	}

	allNodes := Nodes(edges)
	roots := Roots(edges)

	fullReach := ReachableLeaves(edges, roots, "")
	if len(fullReach) == 0 {
		result := make(map[string]float64, len(allNodes))
		for n := range allNodes {
			result[n] = 0.0
		}
		return result
	}

	importance := make(map[string]float64, len(allNodes))
	for n := range allNodes {
		reduced := ReachableLeaves(edges, roots, n)
		lost := len(fullReach) - len(reduced)
		importance[n] = float64(lost) / float64(len(fullReach))
	}

	return importance
}

// EdgeFraction computes the fraction of edges in the graph that are
// incident to (touch) a given node. An edge is incident to a node if
// the node is either the source or destination of that edge.
//
// Result range: [0, 1]. Returns 0 if edges is empty.
//
// Definition: edgeFraction(node) = |{e : e.src == node || e.dst == node}| / |E|
//
// Time complexity: O(|E|).
//
// Source: extracted from aicore/causalmath.ComputeDominoImportance.
func EdgeFraction(edges []Edge, node string) float64 {
	if len(edges) == 0 {
		return 0.0
	}

	count := 0
	for _, e := range edges {
		if e[0] == node || e[1] == node {
			count++
		}
	}

	return float64(count) / float64(len(edges))
}

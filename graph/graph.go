// Package graph provides pure graph algorithms on directed string-labeled
// graphs. All functions are deterministic, use only the Go standard library,
// and make zero external dependencies.
//
// Graphs are represented as edge lists ([][2]string) or adjacency maps
// (map[string][]string). The package provides BFS traversal, DAG depth
// computation, reachability analysis, and node importance metrics.
//
// Extracted from: github.com/davly/aicore/causalmath (proven in production
// across the Causal inference engine).
package graph

// Edge represents a directed edge from Src to Dst.
type Edge = [2]string

// AdjacencyList builds a directed adjacency map from a list of edges.
// Empty strings in edges are silently skipped.
//
// Time complexity: O(|E|) where E is the number of edges.
func AdjacencyList(edges []Edge) map[string][]string {
	adj := make(map[string][]string)
	for _, e := range edges {
		if e[0] != "" && e[1] != "" {
			adj[e[0]] = append(adj[e[0]], e[1])
		}
	}
	return adj
}

// Nodes returns the set of all unique nodes present in the edge list.
// Both source and destination nodes of each edge are included.
// Empty strings in edges are silently skipped.
//
// Time complexity: O(|E|).
func Nodes(edges []Edge) map[string]struct{} {
	nodes := make(map[string]struct{})
	for _, e := range edges {
		if e[0] != "" {
			nodes[e[0]] = struct{}{}
		}
		if e[1] != "" {
			nodes[e[1]] = struct{}{}
		}
	}
	return nodes
}

// InDegree computes the in-degree of every node in the graph.
// Nodes that appear only as sources (in-degree 0) are included in the result.
//
// Time complexity: O(|E|).
func InDegree(edges []Edge) map[string]int {
	deg := make(map[string]int)
	for _, e := range edges {
		if e[0] == "" || e[1] == "" {
			continue
		}
		deg[e[1]]++
		if _, ok := deg[e[0]]; !ok {
			deg[e[0]] = 0
		}
	}
	return deg
}

// Roots returns all nodes with in-degree 0 (no incoming edges).
// The result is deterministic for a given input but the order of roots
// in the returned slice is not guaranteed.
//
// Time complexity: O(|E|).
func Roots(edges []Edge) []string {
	deg := InDegree(edges)
	var roots []string
	for node, d := range deg {
		if d == 0 {
			roots = append(roots, node)
		}
	}
	return roots
}

// Leaves returns all nodes with out-degree 0 (no outgoing edges) as a set.
//
// Time complexity: O(|E| + |V|).
func Leaves(edges []Edge) map[string]struct{} {
	adj := AdjacencyList(edges)
	allNodes := Nodes(edges)
	leaves := make(map[string]struct{})
	for n := range allNodes {
		if len(adj[n]) == 0 {
			leaves[n] = struct{}{}
		}
	}
	return leaves
}

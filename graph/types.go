package graph

// IntAdjacency represents a directed graph with integer-labeled nodes.
// Each key maps a node to its list of neighbors (successors).
// This is used by numeric graph algorithms (shortest path, centrality,
// community detection, flow) that operate on integer-indexed nodes.
type IntAdjacency = map[int][]int

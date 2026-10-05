package graph

import (
	"math/rand"
	"slices"
	"testing"
)

// topologicalSortLinearScanReference is TopologicalSort as it was before the
// O(V^2) linear scan was replaced by a min-heap; it is kept here as the
// equivalence oracle (same order, same partial order and error on a cycle).
func topologicalSortLinearScanReference(adj IntAdjacency, n int) ([]int, error) {
	inDeg := make([]int, n)
	for u := 0; u < n; u++ {
		for _, v := range adj[u] {
			if v >= 0 && v < n {
				inDeg[v]++
			}
		}
	}
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

// TestTopologicalSort_MatchesLinearScanReference compares the heap-based sort
// with the reference on random DAGs and random digraphs (cycles, self-loops,
// duplicate edges and out-of-range targets included).
func TestTopologicalSort_MatchesLinearScanReference(t *testing.T) {
	rng := rand.New(rand.NewSource(20261005))
	for trial := 0; trial < 3000; trial++ {
		n := rng.Intn(40)
		adj := IntAdjacency{}
		dag := rng.Intn(2) == 0
		edges := rng.Intn(3*n + 1)
		for e := 0; e < edges; e++ {
			u, v := rng.Intn(n+1), rng.Intn(n+2)-1 // v may be -1 or n: out of range
			if u >= n {
				continue
			}
			if dag && v >= 0 && v < n && v <= u {
				continue // keep edges forward so the graph stays acyclic
			}
			adj[u] = append(adj[u], v)
			if rng.Intn(10) == 0 {
				adj[u] = append(adj[u], v) // duplicate edge
			}
		}
		gotOrder, gotErr := TopologicalSort(adj, n)
		wantOrder, wantErr := topologicalSortLinearScanReference(adj, n)
		if gotErr != wantErr || !slices.Equal(gotOrder, wantOrder) || (gotOrder == nil) != (wantOrder == nil) {
			t.Fatalf("trial %d (n=%d dag=%v): got (%v, %v) want (%v, %v) adj=%v",
				trial, n, dag, gotOrder, gotErr, wantOrder, wantErr, adj)
		}
	}
}

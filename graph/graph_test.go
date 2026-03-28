package graph

import (
	"math"
	"sort"
	"testing"

	"github.com/davly/reality/testutil"
)

// ═══════════════════════════════════════════════════════════════════════════
// Helper functions
// ═══════════════════════════════════════════════════════════════════════════

func sortedKeys(m map[string]struct{}) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func assertStringSlice(t *testing.T, label string, got, want []string) {
	t.Helper()
	sort.Strings(got)
	sort.Strings(want)
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch: got %v, want %v", label, got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("%s[%d]: got %q, want %q", label, i, got[i], want[i])
		}
	}
}

func assertFloat(t *testing.T, label string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %v, want %v (tol %v)", label, got, want, tol)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// AdjacencyList
// ═══════════════════════════════════════════════════════════════════════════

func TestAdjacencyList_Simple(t *testing.T) {
	edges := []Edge{{"A", "B"}, {"A", "C"}, {"B", "C"}}
	adj := AdjacencyList(edges)
	if len(adj["A"]) != 2 {
		t.Errorf("A neighbors: got %d, want 2", len(adj["A"]))
	}
	if len(adj["B"]) != 1 {
		t.Errorf("B neighbors: got %d, want 1", len(adj["B"]))
	}
}

func TestAdjacencyList_Empty(t *testing.T) {
	adj := AdjacencyList(nil)
	if len(adj) != 0 {
		t.Errorf("empty adj: got %d entries, want 0", len(adj))
	}
}

func TestAdjacencyList_SkipsEmptyStrings(t *testing.T) {
	edges := []Edge{{"A", ""}, {"", "B"}, {"", ""}, {"C", "D"}}
	adj := AdjacencyList(edges)
	if len(adj) != 1 {
		t.Errorf("adj should have 1 entry (C), got %d", len(adj))
	}
	if len(adj["C"]) != 1 || adj["C"][0] != "D" {
		t.Errorf("C->D expected, got %v", adj["C"])
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Nodes
// ═══════════════════════════════════════════════════════════════════════════

func TestNodes_Simple(t *testing.T) {
	edges := []Edge{{"A", "B"}, {"B", "C"}}
	nodes := Nodes(edges)
	want := []string{"A", "B", "C"}
	got := sortedKeys(nodes)
	assertStringSlice(t, "nodes", got, want)
}

func TestNodes_Empty(t *testing.T) {
	nodes := Nodes(nil)
	if len(nodes) != 0 {
		t.Errorf("empty nodes: got %d, want 0", len(nodes))
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// InDegree
// ═══════════════════════════════════════════════════════════════════════════

func TestInDegree_Simple(t *testing.T) {
	edges := []Edge{{"A", "B"}, {"A", "C"}, {"B", "C"}}
	deg := InDegree(edges)
	if deg["A"] != 0 {
		t.Errorf("InDegree(A) = %d, want 0", deg["A"])
	}
	if deg["B"] != 1 {
		t.Errorf("InDegree(B) = %d, want 1", deg["B"])
	}
	if deg["C"] != 2 {
		t.Errorf("InDegree(C) = %d, want 2", deg["C"])
	}
}

func TestInDegree_Empty(t *testing.T) {
	deg := InDegree(nil)
	if len(deg) != 0 {
		t.Errorf("empty in-degree: got %d entries, want 0", len(deg))
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Roots
// ═══════════════════════════════════════════════════════════════════════════

func TestRoots_SingleRoot(t *testing.T) {
	edges := []Edge{{"A", "B"}, {"B", "C"}}
	roots := Roots(edges)
	assertStringSlice(t, "roots", roots, []string{"A"})
}

func TestRoots_MultipleRoots(t *testing.T) {
	edges := []Edge{{"A", "C"}, {"B", "C"}}
	roots := Roots(edges)
	assertStringSlice(t, "roots", roots, []string{"A", "B"})
}

func TestRoots_Empty(t *testing.T) {
	roots := Roots(nil)
	if len(roots) != 0 {
		t.Errorf("empty roots: got %d, want 0", len(roots))
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Leaves
// ═══════════════════════════════════════════════════════════════════════════

func TestLeaves_SingleLeaf(t *testing.T) {
	edges := []Edge{{"A", "B"}, {"B", "C"}}
	leaves := Leaves(edges)
	got := sortedKeys(leaves)
	assertStringSlice(t, "leaves", got, []string{"C"})
}

func TestLeaves_MultipleLeaves(t *testing.T) {
	edges := []Edge{{"A", "B"}, {"A", "C"}}
	leaves := Leaves(edges)
	got := sortedKeys(leaves)
	assertStringSlice(t, "leaves", got, []string{"B", "C"})
}

func TestLeaves_Empty(t *testing.T) {
	leaves := Leaves(nil)
	if len(leaves) != 0 {
		t.Errorf("empty leaves: got %d, want 0", len(leaves))
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// BFSDownstream
// ═══════════════════════════════════════════════════════════════════════════

func TestBFSDownstream_LinearChain(t *testing.T) {
	// A -> B -> C -> D
	edges := []Edge{{"A", "B"}, {"B", "C"}, {"C", "D"}}
	got := BFSDownstream(edges, "A")
	assertStringSlice(t, "downstream-A", got, []string{"B", "C", "D"})
}

func TestBFSDownstream_MiddleNode(t *testing.T) {
	// A -> B -> C -> D
	edges := []Edge{{"A", "B"}, {"B", "C"}, {"C", "D"}}
	got := BFSDownstream(edges, "B")
	assertStringSlice(t, "downstream-B", got, []string{"C", "D"})
}

func TestBFSDownstream_LeafNode(t *testing.T) {
	edges := []Edge{{"A", "B"}, {"B", "C"}}
	got := BFSDownstream(edges, "C")
	if len(got) != 0 {
		t.Errorf("downstream of leaf: got %v, want empty", got)
	}
}

func TestBFSDownstream_NonExistentNode(t *testing.T) {
	edges := []Edge{{"A", "B"}}
	got := BFSDownstream(edges, "Z")
	if len(got) != 0 {
		t.Errorf("downstream of non-existent: got %v, want empty", got)
	}
}

func TestBFSDownstream_EmptyEdges(t *testing.T) {
	got := BFSDownstream(nil, "A")
	if len(got) != 0 {
		t.Errorf("downstream of empty: got %v, want empty", got)
	}
}

func TestBFSDownstream_Diamond(t *testing.T) {
	// A -> B, A -> C, B -> D, C -> D
	edges := []Edge{{"A", "B"}, {"A", "C"}, {"B", "D"}, {"C", "D"}}
	got := BFSDownstream(edges, "A")
	assertStringSlice(t, "diamond-downstream", got, []string{"B", "C", "D"})
}

func TestBFSDownstream_NoCycle(t *testing.T) {
	// A -> B -> C, A -> C (convergent paths, no cycle)
	edges := []Edge{{"A", "B"}, {"B", "C"}, {"A", "C"}}
	got := BFSDownstream(edges, "A")
	assertStringSlice(t, "convergent-downstream", got, []string{"B", "C"})
}

// ═══════════════════════════════════════════════════════════════════════════
// BFSReachable
// ═══════════════════════════════════════════════════════════════════════════

func TestBFSReachable_FromRoot(t *testing.T) {
	edges := []Edge{{"A", "B"}, {"B", "C"}, {"C", "D"}}
	reached := BFSReachable(edges, []string{"A"}, "")
	got := sortedKeys(reached)
	assertStringSlice(t, "reachable-from-A", got, []string{"A", "B", "C", "D"})
}

func TestBFSReachable_WithExclude(t *testing.T) {
	edges := []Edge{{"A", "B"}, {"B", "C"}, {"C", "D"}}
	reached := BFSReachable(edges, []string{"A"}, "B")
	got := sortedKeys(reached)
	// B is excluded, so C and D not reachable
	assertStringSlice(t, "reachable-exclude-B", got, []string{"A"})
}

func TestBFSReachable_ExcludeRoot(t *testing.T) {
	edges := []Edge{{"A", "B"}, {"B", "C"}}
	reached := BFSReachable(edges, []string{"A"}, "A")
	if len(reached) != 0 {
		t.Errorf("reachable excluding root: got %v, want empty", sortedKeys(reached))
	}
}

func TestBFSReachable_MultipleStarts(t *testing.T) {
	edges := []Edge{{"A", "C"}, {"B", "C"}, {"C", "D"}}
	reached := BFSReachable(edges, []string{"A", "B"}, "")
	got := sortedKeys(reached)
	assertStringSlice(t, "multi-start-reachable", got, []string{"A", "B", "C", "D"})
}

// ═══════════════════════════════════════════════════════════════════════════
// DAGDepth
// ═══════════════════════════════════════════════════════════════════════════

func TestDAGDepth_LinearChain(t *testing.T) {
	// A -> B -> C -> D  (depth 3)
	edges := []Edge{{"A", "B"}, {"B", "C"}, {"C", "D"}}
	got := DAGDepth(edges)
	if got != 3 {
		t.Errorf("DAGDepth linear: got %d, want 3", got)
	}
}

func TestDAGDepth_SingleEdge(t *testing.T) {
	edges := []Edge{{"A", "B"}}
	got := DAGDepth(edges)
	if got != 1 {
		t.Errorf("DAGDepth single: got %d, want 1", got)
	}
}

func TestDAGDepth_Diamond(t *testing.T) {
	// A -> B -> D, A -> C -> D  (depth 2)
	edges := []Edge{{"A", "B"}, {"A", "C"}, {"B", "D"}, {"C", "D"}}
	got := DAGDepth(edges)
	if got != 2 {
		t.Errorf("DAGDepth diamond: got %d, want 2", got)
	}
}

func TestDAGDepth_Wide(t *testing.T) {
	// A -> B, A -> C, A -> D  (depth 1)
	edges := []Edge{{"A", "B"}, {"A", "C"}, {"A", "D"}}
	got := DAGDepth(edges)
	if got != 1 {
		t.Errorf("DAGDepth wide: got %d, want 1", got)
	}
}

func TestDAGDepth_Empty(t *testing.T) {
	got := DAGDepth(nil)
	if got != 0 {
		t.Errorf("DAGDepth empty: got %d, want 0", got)
	}
}

func TestDAGDepth_AsymmetricPaths(t *testing.T) {
	// A -> B -> C -> D -> E (depth 4)
	// A -> X              (depth 1)
	edges := []Edge{{"A", "B"}, {"B", "C"}, {"C", "D"}, {"D", "E"}, {"A", "X"}}
	got := DAGDepth(edges)
	if got != 4 {
		t.Errorf("DAGDepth asymmetric: got %d, want 4", got)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// ReachableLeaves
// ═══════════════════════════════════════════════════════════════════════════

func TestReachableLeaves_AllReachable(t *testing.T) {
	// A -> B -> D, A -> C -> D  (leaf: D)
	edges := []Edge{{"A", "B"}, {"A", "C"}, {"B", "D"}, {"C", "D"}}
	reached := ReachableLeaves(edges, []string{"A"}, "")
	got := sortedKeys(reached)
	assertStringSlice(t, "reachable-leaves-all", got, []string{"D"})
}

func TestReachableLeaves_ExcludeBottleneck(t *testing.T) {
	// A -> B -> C (leaf: C, exclude B -> C unreachable)
	edges := []Edge{{"A", "B"}, {"B", "C"}}
	reached := ReachableLeaves(edges, []string{"A"}, "B")
	if len(reached) != 0 {
		t.Errorf("reachable leaves excluding B: got %v, want empty", sortedKeys(reached))
	}
}

func TestReachableLeaves_MultipleLeaves(t *testing.T) {
	// A -> B, A -> C  (leaves: B, C)
	edges := []Edge{{"A", "B"}, {"A", "C"}}
	reached := ReachableLeaves(edges, []string{"A"}, "")
	got := sortedKeys(reached)
	assertStringSlice(t, "multi-leaves", got, []string{"B", "C"})
}

func TestReachableLeaves_ExcludeOnePathLeaf(t *testing.T) {
	// A -> B, A -> C -> D  (leaves: B, D; exclude C -> only B reachable)
	edges := []Edge{{"A", "B"}, {"A", "C"}, {"C", "D"}}
	reached := ReachableLeaves(edges, []string{"A"}, "C")
	got := sortedKeys(reached)
	assertStringSlice(t, "exclude-one-path", got, []string{"B"})
}

// ═══════════════════════════════════════════════════════════════════════════
// NodeImportance
// ═══════════════════════════════════════════════════════════════════════════

func TestNodeImportance_LinearChain(t *testing.T) {
	// A -> B -> C  (leaf: C)
	// Remove A: C unreachable (importance 1.0)
	// Remove B: C unreachable (importance 1.0)
	// Remove C: C is the leaf and is removed, so unreachable (importance 1.0)
	edges := []Edge{{"A", "B"}, {"B", "C"}}
	imp := NodeImportance(edges)
	assertFloat(t, "imp-A", imp["A"], 1.0, 1e-15)
	assertFloat(t, "imp-B", imp["B"], 1.0, 1e-15)
	assertFloat(t, "imp-C", imp["C"], 1.0, 1e-15)
}

func TestNodeImportance_RedundantPaths(t *testing.T) {
	// A -> B -> D, A -> C -> D  (leaf: D)
	// Remove B: D still reachable via C (importance 0)
	// Remove C: D still reachable via B (importance 0)
	// Remove A: nothing reachable (importance 1.0)
	// Remove D: leaf gone (importance 1.0)
	edges := []Edge{{"A", "B"}, {"A", "C"}, {"B", "D"}, {"C", "D"}}
	imp := NodeImportance(edges)
	assertFloat(t, "imp-A", imp["A"], 1.0, 1e-15)
	assertFloat(t, "imp-B", imp["B"], 0.0, 1e-15)
	assertFloat(t, "imp-C", imp["C"], 0.0, 1e-15)
	assertFloat(t, "imp-D", imp["D"], 1.0, 1e-15)
}

func TestNodeImportance_Empty(t *testing.T) {
	imp := NodeImportance(nil)
	if len(imp) != 0 {
		t.Errorf("empty importance: got %d entries, want 0", len(imp))
	}
}

func TestNodeImportance_TwoLeaves(t *testing.T) {
	// A -> B, A -> C  (leaves: B, C)
	// Remove B: only C reachable (importance 0.5)
	// Remove C: only B reachable (importance 0.5)
	// Remove A: nothing reachable (importance 1.0)
	edges := []Edge{{"A", "B"}, {"A", "C"}}
	imp := NodeImportance(edges)
	assertFloat(t, "imp-A", imp["A"], 1.0, 1e-15)
	assertFloat(t, "imp-B", imp["B"], 0.5, 1e-15)
	assertFloat(t, "imp-C", imp["C"], 0.5, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// EdgeFraction
// ═══════════════════════════════════════════════════════════════════════════

func TestEdgeFraction_CentralNode(t *testing.T) {
	// A -> B, B -> C, B -> D  (B is in 3 of 3 edges)
	edges := []Edge{{"A", "B"}, {"B", "C"}, {"B", "D"}}
	got := EdgeFraction(edges, "B")
	assertFloat(t, "ef-B", got, 1.0, 1e-15)
}

func TestEdgeFraction_LeafNode(t *testing.T) {
	// A -> B, B -> C, B -> D  (D is in 1 of 3 edges)
	edges := []Edge{{"A", "B"}, {"B", "C"}, {"B", "D"}}
	got := EdgeFraction(edges, "D")
	assertFloat(t, "ef-D", got, 1.0/3.0, 1e-15)
}

func TestEdgeFraction_NonExistent(t *testing.T) {
	edges := []Edge{{"A", "B"}}
	got := EdgeFraction(edges, "Z")
	assertFloat(t, "ef-Z", got, 0.0, 1e-15)
}

func TestEdgeFraction_Empty(t *testing.T) {
	got := EdgeFraction(nil, "A")
	assertFloat(t, "ef-empty", got, 0.0, 1e-15)
}

func TestEdgeFraction_SelfLoop(t *testing.T) {
	edges := []Edge{{"A", "A"}, {"A", "B"}}
	got := EdgeFraction(edges, "A")
	assertFloat(t, "ef-self-loop", got, 1.0, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Golden-file tests
// ═══════════════════════════════════════════════════════════════════════════

func TestGolden_DAGDepth(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/graph/dag_depth.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			edges := inputEdges(t, tc)
			got := DAGDepth(edges)
			want := testutil.InputFloat64(t, tc, "expected_depth")
			if float64(got) != want {
				t.Errorf("DAGDepth = %d, want %v", got, want)
			}
		})
	}
}

func TestGolden_EdgeFraction(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/graph/edge_fraction.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			edges := inputEdges(t, tc)
			node := inputString(t, tc, "node")
			got := EdgeFraction(edges, node)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_BFSDownstream(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/graph/bfs_downstream.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			edges := inputEdges(t, tc)
			start := inputString(t, tc, "start")
			got := BFSDownstream(edges, start)
			want := expectedStringSlice(t, tc)
			assertStringSlice(t, tc.Description, got, want)
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Golden-file input helpers
// ═══════════════════════════════════════════════════════════════════════════

func inputEdges(t *testing.T, tc testutil.TestCase) []Edge {
	t.Helper()
	val, exists := tc.Inputs["edges"]
	if !exists {
		t.Fatalf("[%s] missing input key 'edges'", tc.Description)
	}
	arr, ok := val.([]any)
	if !ok {
		t.Fatalf("[%s] edges is not an array: %T", tc.Description, val)
	}
	edges := make([]Edge, len(arr))
	for i, elem := range arr {
		pair, ok := elem.([]any)
		if !ok || len(pair) != 2 {
			t.Fatalf("[%s] edge[%d] is not a 2-element array", tc.Description, i)
		}
		src, ok1 := pair[0].(string)
		dst, ok2 := pair[1].(string)
		if !ok1 || !ok2 {
			t.Fatalf("[%s] edge[%d] elements are not strings", tc.Description, i)
		}
		edges[i] = Edge{src, dst}
	}
	return edges
}

func inputString(t *testing.T, tc testutil.TestCase, key string) string {
	t.Helper()
	val, exists := tc.Inputs[key]
	if !exists {
		t.Fatalf("[%s] missing input key %q", tc.Description, key)
	}
	s, ok := val.(string)
	if !ok {
		t.Fatalf("[%s] input %q is not a string: %T", tc.Description, key, val)
	}
	return s
}

func expectedStringSlice(t *testing.T, tc testutil.TestCase) []string {
	t.Helper()
	arr, ok := tc.Expected.([]any)
	if !ok {
		t.Fatalf("[%s] expected is not an array: %T", tc.Description, tc.Expected)
	}
	result := make([]string, len(arr))
	for i, elem := range arr {
		s, ok := elem.(string)
		if !ok {
			t.Fatalf("[%s] expected[%d] is not a string: %T", tc.Description, i, elem)
		}
		result[i] = s
	}
	return result
}

// ═══════════════════════════════════════════════════════════════════════════
// Dijkstra
// ═══════════════════════════════════════════════════════════════════════════

func TestDijkstra_SimpleTriangle(t *testing.T) {
	adj := IntAdjacency{0: {1, 2}, 1: {2}}
	weights := map[[2]int]float64{
		{0, 1}: 1, {0, 2}: 4, {1, 2}: 2,
	}
	dist, prev := Dijkstra(adj, weights, 0)
	assertFloat(t, "d[0]", dist[0], 0, 1e-15)
	assertFloat(t, "d[1]", dist[1], 1, 1e-15)
	assertFloat(t, "d[2]", dist[2], 3, 1e-15)
	if prev[1] != 0 {
		t.Errorf("prev[1] = %d, want 0", prev[1])
	}
	if prev[2] != 1 {
		t.Errorf("prev[2] = %d, want 1", prev[2])
	}
}

func TestDijkstra_UnreachableNode(t *testing.T) {
	adj := IntAdjacency{0: {1}}
	weights := map[[2]int]float64{{0, 1}: 5}
	dist, prev := Dijkstra(adj, weights, 0)
	if len(dist) < 3 {
		// Node 2 may not exist; just check 0 and 1
		assertFloat(t, "d[0]", dist[0], 0, 1e-15)
		assertFloat(t, "d[1]", dist[1], 5, 1e-15)
	}
	if prev[0] != -1 {
		t.Errorf("prev[source] = %d, want -1", prev[0])
	}
}

func TestDijkstra_SingleNode(t *testing.T) {
	adj := IntAdjacency{}
	weights := map[[2]int]float64{}
	dist, prev := Dijkstra(adj, weights, 0)
	assertFloat(t, "d[0]", dist[0], 0, 1e-15)
	if prev[0] != -1 {
		t.Errorf("prev[0] = %d, want -1", prev[0])
	}
}

func TestDijkstra_LinearChain(t *testing.T) {
	adj := IntAdjacency{0: {1}, 1: {2}, 2: {3}}
	weights := map[[2]int]float64{{0, 1}: 2, {1, 2}: 3, {2, 3}: 1}
	dist, _ := Dijkstra(adj, weights, 0)
	assertFloat(t, "d[0]", dist[0], 0, 1e-15)
	assertFloat(t, "d[1]", dist[1], 2, 1e-15)
	assertFloat(t, "d[2]", dist[2], 5, 1e-15)
	assertFloat(t, "d[3]", dist[3], 6, 1e-15)
}

func TestDijkstra_Diamond(t *testing.T) {
	adj := IntAdjacency{0: {1, 2}, 1: {3}, 2: {3}}
	weights := map[[2]int]float64{
		{0, 1}: 1, {0, 2}: 5, {1, 3}: 1, {2, 3}: 1,
	}
	dist, _ := Dijkstra(adj, weights, 0)
	assertFloat(t, "d[3]", dist[3], 2, 1e-15)
}

func TestDijkstra_ZeroWeightEdges(t *testing.T) {
	adj := IntAdjacency{0: {1}, 1: {2}}
	weights := map[[2]int]float64{{0, 1}: 0, {1, 2}: 0}
	dist, _ := Dijkstra(adj, weights, 0)
	assertFloat(t, "d[2]", dist[2], 0, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// A*
// ═══════════════════════════════════════════════════════════════════════════

func TestAStar_MatchesDijkstraWithZeroHeuristic(t *testing.T) {
	adj := IntAdjacency{0: {1, 2}, 1: {2, 3}, 2: {3}}
	weights := map[[2]int]float64{
		{0, 1}: 1, {0, 2}: 4, {1, 2}: 2, {1, 3}: 6, {2, 3}: 3,
	}
	h := func(int) float64 { return 0 }
	path, dist := AStar(adj, weights, 0, 3, h)
	assertFloat(t, "AStar dist", dist, 6, 1e-15)
	if len(path) < 2 || path[0] != 0 || path[len(path)-1] != 3 {
		t.Errorf("AStar path: got %v, want path from 0 to 3", path)
	}
}

func TestAStar_WithAdmissibleHeuristic(t *testing.T) {
	// Grid-like: 0-(1)->1-(1)->2-(1)->3
	adj := IntAdjacency{0: {1}, 1: {2}, 2: {3}}
	weights := map[[2]int]float64{
		{0, 1}: 1, {1, 2}: 1, {2, 3}: 1,
	}
	// Admissible heuristic: distance to target 3 in a line.
	h := func(n int) float64 { return float64(3 - n) }
	path, dist := AStar(adj, weights, 0, 3, h)
	assertFloat(t, "AStar dist", dist, 3, 1e-15)
	if len(path) != 4 {
		t.Errorf("path length = %d, want 4", len(path))
	}
}

func TestAStar_Unreachable(t *testing.T) {
	adj := IntAdjacency{0: {1}}
	weights := map[[2]int]float64{{0, 1}: 1}
	h := func(int) float64 { return 0 }
	path, dist := AStar(adj, weights, 0, 2, h)
	if path != nil {
		t.Errorf("expected nil path for unreachable target, got %v", path)
	}
	if !math.IsInf(dist, 1) {
		t.Errorf("expected +Inf for unreachable target, got %v", dist)
	}
}

func TestAStar_SourceEqualsTarget(t *testing.T) {
	adj := IntAdjacency{0: {1}}
	weights := map[[2]int]float64{{0, 1}: 1}
	h := func(int) float64 { return 0 }
	path, dist := AStar(adj, weights, 0, 0, h)
	assertFloat(t, "self dist", dist, 0, 1e-15)
	if len(path) != 1 || path[0] != 0 {
		t.Errorf("self path = %v, want [0]", path)
	}
}

func TestAStar_ShortcutPath(t *testing.T) {
	// Two paths: 0->1->2->3 (cost 3) and 0->3 (cost 10). Should pick shorter.
	adj := IntAdjacency{0: {1, 3}, 1: {2}, 2: {3}}
	weights := map[[2]int]float64{
		{0, 1}: 1, {1, 2}: 1, {2, 3}: 1, {0, 3}: 10,
	}
	h := func(int) float64 { return 0 }
	_, dist := AStar(adj, weights, 0, 3, h)
	assertFloat(t, "shortcut dist", dist, 3, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// FloydWarshall
// ═══════════════════════════════════════════════════════════════════════════

func TestFloydWarshall_Triangle(t *testing.T) {
	edges := [][3]float64{{0, 1, 1}, {1, 2, 2}, {0, 2, 4}}
	dist := FloydWarshall(3, edges)
	assertFloat(t, "0->0", dist[0][0], 0, 1e-15)
	assertFloat(t, "0->1", dist[0][1], 1, 1e-15)
	assertFloat(t, "0->2", dist[0][2], 3, 1e-15) // through 1
	assertFloat(t, "1->2", dist[1][2], 2, 1e-15)
	if !math.IsInf(dist[2][0], 1) {
		t.Errorf("2->0 should be +Inf, got %v", dist[2][0])
	}
}

func TestFloydWarshall_AllPairsSmall(t *testing.T) {
	// Bidirectional triangle: 0<->1<->2<->0 with weight 1.
	edges := [][3]float64{
		{0, 1, 1}, {1, 0, 1},
		{1, 2, 1}, {2, 1, 1},
		{0, 2, 1}, {2, 0, 1},
	}
	dist := FloydWarshall(3, edges)
	for i := 0; i < 3; i++ {
		assertFloat(t, "self", dist[i][i], 0, 1e-15)
		for j := 0; j < 3; j++ {
			if i != j {
				assertFloat(t, "pair", dist[i][j], 1, 1e-15)
			}
		}
	}
}

func TestFloydWarshall_Disconnected(t *testing.T) {
	// Two disconnected nodes.
	dist := FloydWarshall(2, nil)
	assertFloat(t, "0->0", dist[0][0], 0, 1e-15)
	assertFloat(t, "1->1", dist[1][1], 0, 1e-15)
	if !math.IsInf(dist[0][1], 1) {
		t.Errorf("0->1 should be +Inf, got %v", dist[0][1])
	}
}

func TestFloydWarshall_SingleNode(t *testing.T) {
	dist := FloydWarshall(1, nil)
	assertFloat(t, "0->0", dist[0][0], 0, 1e-15)
}

func TestFloydWarshall_LongerDetour(t *testing.T) {
	// Direct 0->2 costs 10, but 0->1->2 costs 3.
	edges := [][3]float64{{0, 1, 1}, {1, 2, 2}, {0, 2, 10}}
	dist := FloydWarshall(3, edges)
	assertFloat(t, "0->2", dist[0][2], 3, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// BetweennessCentrality
// ═══════════════════════════════════════════════════════════════════════════

func TestBetweennessCentrality_StarGraph(t *testing.T) {
	// Star: 0 is center, edges 1->0, 2->0, 3->0, 0->1, 0->2, 0->3.
	adj := IntAdjacency{
		0: {1, 2, 3},
		1: {0},
		2: {0},
		3: {0},
	}
	bc := BetweennessCentrality(adj, 4)
	// Center (0) should have highest betweenness.
	for i := 1; i <= 3; i++ {
		if bc[0] <= bc[i] {
			t.Errorf("center bc[0]=%v should exceed leaf bc[%d]=%v", bc[0], i, bc[i])
		}
	}
}

func TestBetweennessCentrality_PathGraph(t *testing.T) {
	// 0 -> 1 -> 2 -> 3
	adj := IntAdjacency{0: {1}, 1: {2}, 2: {3}}
	bc := BetweennessCentrality(adj, 4)
	// Node 1 is on paths 0->2, 0->3. Node 2 is on paths 0->3, 1->3.
	// Both 1 and 2 should have positive betweenness. 0 and 3 should be 0.
	assertFloat(t, "bc[0]", bc[0], 0, 1e-15)
	assertFloat(t, "bc[3]", bc[3], 0, 1e-15)
	if bc[1] <= 0 {
		t.Errorf("bc[1] should be > 0, got %v", bc[1])
	}
	if bc[2] <= 0 {
		t.Errorf("bc[2] should be > 0, got %v", bc[2])
	}
}

func TestBetweennessCentrality_Triangle(t *testing.T) {
	// Fully connected triangle: 0<->1<->2<->0
	adj := IntAdjacency{
		0: {1, 2},
		1: {0, 2},
		2: {0, 1},
	}
	bc := BetweennessCentrality(adj, 3)
	// Symmetric: all should be equal.
	assertFloat(t, "symmetry 0-1", bc[0], bc[1], 1e-12)
	assertFloat(t, "symmetry 1-2", bc[1], bc[2], 1e-12)
}

func TestBetweennessCentrality_NoEdges(t *testing.T) {
	adj := IntAdjacency{}
	bc := BetweennessCentrality(adj, 3)
	for i := 0; i < 3; i++ {
		assertFloat(t, "no-edge bc", bc[i], 0, 1e-15)
	}
}

func TestBetweennessCentrality_KnownValues(t *testing.T) {
	// Path: 0->1->2. bc[1] = 1 (on the 0->2 path).
	adj := IntAdjacency{0: {1}, 1: {2}}
	bc := BetweennessCentrality(adj, 3)
	assertFloat(t, "bc[1]", bc[1], 1.0, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// EigenvectorCentrality
// ═══════════════════════════════════════════════════════════════════════════

func TestEigenvectorCentrality_StarCenter(t *testing.T) {
	// Star: all edges point to center 0.
	adj := IntAdjacency{1: {0}, 2: {0}, 3: {0}}
	ec := EigenvectorCentrality(adj, nil, 4, 100)
	// Center should have the highest score.
	for i := 1; i <= 3; i++ {
		if ec[0] <= ec[i] {
			t.Errorf("center ec[0]=%v should exceed leaf ec[%d]=%v", ec[0], i, ec[i])
		}
	}
}

func TestEigenvectorCentrality_Converges(t *testing.T) {
	adj := IntAdjacency{0: {1}, 1: {0}}
	ec := EigenvectorCentrality(adj, nil, 2, 200)
	// Symmetric: both should be equal.
	assertFloat(t, "symmetry", ec[0], ec[1], 1e-6)
}

func TestEigenvectorCentrality_WithWeights(t *testing.T) {
	adj := IntAdjacency{0: {1, 2}, 1: {0}, 2: {0}}
	weights := map[[2]int]float64{
		{0, 1}: 10, {0, 2}: 1, {1, 0}: 10, {2, 0}: 1,
	}
	ec := EigenvectorCentrality(adj, weights, 3, 100)
	// Node 1 has a heavier connection, should rank higher than node 2.
	if ec[1] <= ec[2] {
		t.Errorf("ec[1]=%v should exceed ec[2]=%v (heavier weight)", ec[1], ec[2])
	}
}

func TestEigenvectorCentrality_NoEdges(t *testing.T) {
	adj := IntAdjacency{}
	ec := EigenvectorCentrality(adj, nil, 3, 100)
	// With no edges, initial uniform scores remain (no signal to propagate).
	// Just check it doesn't panic and values are reasonable.
	if len(ec) != 3 {
		t.Errorf("expected 3 values, got %d", len(ec))
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// DegreeCentrality
// ═══════════════════════════════════════════════════════════════════════════

func TestDegreeCentrality_Star(t *testing.T) {
	// 0 -> 1, 0 -> 2, 0 -> 3
	adj := IntAdjacency{0: {1, 2, 3}}
	dc := DegreeCentrality(adj, 4)
	// Node 0 has out-degree 3, in-degree 0 => total 3, / (4-1) = 1.0
	assertFloat(t, "dc[0]", dc[0], 1.0, 1e-15)
	// Nodes 1,2,3 have in-degree 1, out-degree 0 => total 1, / 3 ≈ 0.333
	assertFloat(t, "dc[1]", dc[1], 1.0/3.0, 1e-15)
}

func TestDegreeCentrality_SingleNode(t *testing.T) {
	adj := IntAdjacency{}
	dc := DegreeCentrality(adj, 1)
	assertFloat(t, "dc[0]", dc[0], 0, 1e-15)
}

func TestDegreeCentrality_CompleteDirected(t *testing.T) {
	// 3-node complete directed graph.
	adj := IntAdjacency{
		0: {1, 2},
		1: {0, 2},
		2: {0, 1},
	}
	dc := DegreeCentrality(adj, 3)
	// Each node: out=2, in=2, total=4, / 2 = 2.0
	for i := 0; i < 3; i++ {
		assertFloat(t, "dc complete", dc[i], 2.0, 1e-15)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// ConnectedComponents
// ═══════════════════════════════════════════════════════════════════════════

func TestConnectedComponents_ThreeIslands(t *testing.T) {
	// Island 1: 0-1, Island 2: 2-3, Island 3: 4 (isolated)
	adj := IntAdjacency{0: {1}, 2: {3}}
	cc := ConnectedComponents(adj, 5)
	if len(cc) != 3 {
		t.Fatalf("expected 3 components, got %d: %v", len(cc), cc)
	}
}

func TestConnectedComponents_SingleComponent(t *testing.T) {
	adj := IntAdjacency{0: {1}, 1: {2}, 2: {0}}
	cc := ConnectedComponents(adj, 3)
	if len(cc) != 1 {
		t.Fatalf("expected 1 component, got %d", len(cc))
	}
	if len(cc[0]) != 3 {
		t.Errorf("component size = %d, want 3", len(cc[0]))
	}
}

func TestConnectedComponents_AllIsolated(t *testing.T) {
	adj := IntAdjacency{}
	cc := ConnectedComponents(adj, 4)
	if len(cc) != 4 {
		t.Fatalf("expected 4 components (all isolated), got %d", len(cc))
	}
	for _, comp := range cc {
		if len(comp) != 1 {
			t.Errorf("isolated component size = %d, want 1", len(comp))
		}
	}
}

func TestConnectedComponents_TwoComponents(t *testing.T) {
	adj := IntAdjacency{0: {1}, 1: {0}, 2: {3}, 3: {2}}
	cc := ConnectedComponents(adj, 4)
	if len(cc) != 2 {
		t.Fatalf("expected 2 components, got %d", len(cc))
	}
}

func TestConnectedComponents_DirectedTreatedAsUndirected(t *testing.T) {
	// Edge only 0->1, but treated undirected -> they're connected.
	adj := IntAdjacency{0: {1}}
	cc := ConnectedComponents(adj, 2)
	if len(cc) != 1 {
		t.Fatalf("expected 1 component (undirected), got %d", len(cc))
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// StronglyConnected (Tarjan)
// ═══════════════════════════════════════════════════════════════════════════

func TestStronglyConnected_DAG(t *testing.T) {
	// Pure DAG: 0->1->2->3. Each node is its own SCC.
	adj := IntAdjacency{0: {1}, 1: {2}, 2: {3}}
	sccs := StronglyConnected(adj, 4)
	if len(sccs) != 4 {
		t.Fatalf("DAG should have 4 SCCs, got %d: %v", len(sccs), sccs)
	}
}

func TestStronglyConnected_SingleCycle(t *testing.T) {
	// 0->1->2->0: one SCC of size 3.
	adj := IntAdjacency{0: {1}, 1: {2}, 2: {0}}
	sccs := StronglyConnected(adj, 3)
	if len(sccs) != 1 {
		t.Fatalf("cycle should have 1 SCC, got %d", len(sccs))
	}
	if len(sccs[0]) != 3 {
		t.Errorf("SCC size = %d, want 3", len(sccs[0]))
	}
}

func TestStronglyConnected_TwoCycles(t *testing.T) {
	// Cycle 1: 0->1->0, Cycle 2: 2->3->2, Bridge: 1->2
	adj := IntAdjacency{0: {1}, 1: {0, 2}, 2: {3}, 3: {2}}
	sccs := StronglyConnected(adj, 4)
	if len(sccs) != 2 {
		t.Fatalf("expected 2 SCCs, got %d: %v", len(sccs), sccs)
	}
}

func TestStronglyConnected_IsolatedNodes(t *testing.T) {
	adj := IntAdjacency{}
	sccs := StronglyConnected(adj, 3)
	if len(sccs) != 3 {
		t.Fatalf("isolated nodes should have 3 SCCs, got %d", len(sccs))
	}
}

func TestStronglyConnected_ClassicTarjan(t *testing.T) {
	// Classic example:
	// 0->1, 1->2, 2->0 (SCC: {0,1,2})
	// 2->3, 3->4, 4->3 (SCC: {3,4})
	adj := IntAdjacency{0: {1}, 1: {2}, 2: {0, 3}, 3: {4}, 4: {3}}
	sccs := StronglyConnected(adj, 5)
	if len(sccs) != 2 {
		t.Fatalf("expected 2 SCCs, got %d: %v", len(sccs), sccs)
	}
	// Find the larger SCC.
	found3 := false
	found2 := false
	for _, scc := range sccs {
		if len(scc) == 3 {
			found3 = true
		}
		if len(scc) == 2 {
			found2 = true
		}
	}
	if !found3 || !found2 {
		t.Errorf("expected SCCs of size 3 and 2, got %v", sccs)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// LouvainCommunities
// ═══════════════════════════════════════════════════════════════════════════

func TestLouvainCommunities_TwoCliques(t *testing.T) {
	// Clique 1: 0-1-2 fully connected, Clique 2: 3-4-5 fully connected.
	// Single bridge: 2->3.
	adj := IntAdjacency{
		0: {1, 2},
		1: {0, 2},
		2: {0, 1, 3},
		3: {2, 4, 5},
		4: {3, 5},
		5: {3, 4},
	}
	comm := LouvainCommunities(adj, nil, 6)
	if len(comm) != 6 {
		t.Fatalf("expected 6 community labels, got %d", len(comm))
	}
	// Nodes within a clique should share the same community.
	if comm[0] != comm[1] || comm[0] != comm[2] {
		t.Errorf("clique 1 not unified: comm = %v", comm)
	}
	if comm[3] != comm[4] || comm[3] != comm[5] {
		t.Errorf("clique 2 not unified: comm = %v", comm)
	}
	// The two cliques should be in different communities.
	if comm[0] == comm[3] {
		t.Errorf("cliques should be separate communities: comm = %v", comm)
	}
}

func TestLouvainCommunities_SingleNode(t *testing.T) {
	adj := IntAdjacency{}
	comm := LouvainCommunities(adj, nil, 1)
	if len(comm) != 1 {
		t.Fatalf("expected 1 label, got %d", len(comm))
	}
}

func TestLouvainCommunities_NoEdges(t *testing.T) {
	adj := IntAdjacency{}
	comm := LouvainCommunities(adj, nil, 3)
	if len(comm) != 3 {
		t.Fatalf("expected 3 labels, got %d", len(comm))
	}
	// All nodes should be in different communities.
	if comm[0] == comm[1] || comm[1] == comm[2] || comm[0] == comm[2] {
		t.Errorf("no-edge graph: nodes should be in separate communities: %v", comm)
	}
}

func TestLouvainCommunities_EmptyGraph(t *testing.T) {
	comm := LouvainCommunities(nil, nil, 0)
	if comm != nil {
		t.Errorf("expected nil for empty graph, got %v", comm)
	}
}

func TestLouvainCommunities_WithWeights(t *testing.T) {
	// Heavy edge between 0-1, weak edge between 1-2.
	adj := IntAdjacency{0: {1}, 1: {0, 2}, 2: {1}}
	weights := map[[2]int]float64{
		{0, 1}: 100, {1, 0}: 100,
		{1, 2}: 1, {2, 1}: 1,
	}
	comm := LouvainCommunities(adj, weights, 3)
	// 0 and 1 should cluster together.
	if comm[0] != comm[1] {
		t.Errorf("heavily connected 0,1 should share community: %v", comm)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// MaxFlow
// ═══════════════════════════════════════════════════════════════════════════

func TestMaxFlow_SimpleNetwork(t *testing.T) {
	// Classic: source=0, sink=3
	// 0->1 cap 10, 0->2 cap 10, 1->3 cap 10, 2->3 cap 10, 1->2 cap 1
	adj := IntAdjacency{0: {1, 2}, 1: {2, 3}, 2: {3}}
	cap := map[[2]int]float64{
		{0, 1}: 10, {0, 2}: 10, {1, 3}: 10, {2, 3}: 10, {1, 2}: 1,
	}
	flow := MaxFlow(adj, cap, 0, 3)
	assertFloat(t, "max flow", flow, 20, 1e-15)
}

func TestMaxFlow_BottleneckEdge(t *testing.T) {
	// 0->1 cap 100, 1->2 cap 1 (bottleneck), 2->3 cap 100
	adj := IntAdjacency{0: {1}, 1: {2}, 2: {3}}
	cap := map[[2]int]float64{
		{0, 1}: 100, {1, 2}: 1, {2, 3}: 100,
	}
	flow := MaxFlow(adj, cap, 0, 3)
	assertFloat(t, "bottleneck flow", flow, 1, 1e-15)
}

func TestMaxFlow_ParallelPaths(t *testing.T) {
	// 0->1->3 cap 5 each, 0->2->3 cap 3 each => max flow = 5 + 3 = 8
	adj := IntAdjacency{0: {1, 2}, 1: {3}, 2: {3}}
	cap := map[[2]int]float64{
		{0, 1}: 5, {1, 3}: 5, {0, 2}: 3, {2, 3}: 3,
	}
	flow := MaxFlow(adj, cap, 0, 3)
	assertFloat(t, "parallel flow", flow, 8, 1e-15)
}

func TestMaxFlow_NoPath(t *testing.T) {
	adj := IntAdjacency{0: {1}, 2: {3}}
	cap := map[[2]int]float64{{0, 1}: 10, {2, 3}: 10}
	flow := MaxFlow(adj, cap, 0, 3)
	assertFloat(t, "no path flow", flow, 0, 1e-15)
}

func TestMaxFlow_SelfSourceSink(t *testing.T) {
	adj := IntAdjacency{0: {0}}
	cap := map[[2]int]float64{{0, 0}: 100}
	flow := MaxFlow(adj, cap, 0, 0)
	// Source == sink, flow is 0 (trivially satisfied).
	assertFloat(t, "self flow", flow, 0, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// TopologicalSort
// ═══════════════════════════════════════════════════════════════════════════

func TestTopologicalSort_SimpleDAG(t *testing.T) {
	// 0->1, 0->2, 1->3, 2->3
	adj := IntAdjacency{0: {1, 2}, 1: {3}, 2: {3}}
	order, err := TopologicalSort(adj, 4)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(order) != 4 {
		t.Fatalf("order length = %d, want 4", len(order))
	}
	// Verify topological property: for each edge u->v, u appears before v.
	pos := make(map[int]int, 4)
	for i, v := range order {
		pos[v] = i
	}
	edges := [][2]int{{0, 1}, {0, 2}, {1, 3}, {2, 3}}
	for _, e := range edges {
		if pos[e[0]] >= pos[e[1]] {
			t.Errorf("edge %d->%d violates order: pos[%d]=%d, pos[%d]=%d",
				e[0], e[1], e[0], pos[e[0]], e[1], pos[e[1]])
		}
	}
}

func TestTopologicalSort_CycleDetection(t *testing.T) {
	adj := IntAdjacency{0: {1}, 1: {2}, 2: {0}}
	_, err := TopologicalSort(adj, 3)
	if err == nil {
		t.Fatal("expected cycle error, got nil")
	}
	if err != ErrCycleDetected {
		t.Errorf("expected ErrCycleDetected, got %v", err)
	}
}

func TestTopologicalSort_LinearChain(t *testing.T) {
	adj := IntAdjacency{0: {1}, 1: {2}, 2: {3}}
	order, err := TopologicalSort(adj, 4)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Only valid order: 0, 1, 2, 3.
	for i := 0; i < 4; i++ {
		if order[i] != i {
			t.Errorf("order[%d] = %d, want %d", i, order[i], i)
		}
	}
}

func TestTopologicalSort_NoEdges(t *testing.T) {
	adj := IntAdjacency{}
	order, err := TopologicalSort(adj, 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(order) != 3 {
		t.Errorf("order length = %d, want 3", len(order))
	}
	// Should be deterministic: 0, 1, 2 (smallest first).
	for i := 0; i < 3; i++ {
		if order[i] != i {
			t.Errorf("order[%d] = %d, want %d", i, order[i], i)
		}
	}
}

func TestTopologicalSort_SingleNode(t *testing.T) {
	adj := IntAdjacency{}
	order, err := TopologicalSort(adj, 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(order) != 1 || order[0] != 0 {
		t.Errorf("order = %v, want [0]", order)
	}
}

func TestTopologicalSort_DiamondDAG(t *testing.T) {
	adj := IntAdjacency{0: {1, 2}, 1: {3}, 2: {3}}
	order, err := TopologicalSort(adj, 4)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// 0 must come first, 3 must come last.
	if order[0] != 0 {
		t.Errorf("first should be 0, got %d", order[0])
	}
	if order[3] != 3 {
		t.Errorf("last should be 3, got %d", order[3])
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Golden-file: Dijkstra
// ═══════════════════════════════════════════════════════════════════════════

func TestGolden_Dijkstra(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/graph/dijkstra.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			// Parse adj.
			adjRaw, ok := tc.Inputs["adj"].(map[string]any)
			if !ok {
				t.Fatalf("adj is not a map: %T", tc.Inputs["adj"])
			}
			adj := make(IntAdjacency)
			for kStr, vRaw := range adjRaw {
				k := 0
				for _, c := range kStr {
					k = k*10 + int(c-'0')
				}
				arr, ok := vRaw.([]any)
				if !ok {
					t.Fatalf("adj[%s] is not array: %T", kStr, vRaw)
				}
				adj[k] = nil // ensure key exists even for empty neighbor lists
				for _, elem := range arr {
					v, ok := toTestFloat(elem)
					if !ok {
						t.Fatalf("adj value not numeric: %T", elem)
					}
					adj[k] = append(adj[k], int(v))
				}
			}

			// Parse weights.
			weightsRaw, ok := tc.Inputs["weights"].([]any)
			if !ok {
				t.Fatalf("weights is not an array: %T", tc.Inputs["weights"])
			}
			weights := make(map[[2]int]float64)
			for _, wRaw := range weightsRaw {
				pair, ok := wRaw.([]any)
				if !ok || len(pair) != 2 {
					t.Fatalf("weight entry not [edge, value]: %v", wRaw)
				}
				edgeArr, ok := pair[0].([]any)
				if !ok || len(edgeArr) != 2 {
					t.Fatalf("weight edge not [u,v]: %v", pair[0])
				}
				uF, _ := toTestFloat(edgeArr[0])
				vF, _ := toTestFloat(edgeArr[1])
				wF, _ := toTestFloat(pair[1])
				weights[[2]int{int(uF), int(vF)}] = wF
			}

			// Parse source.
			sourceF := testutil.InputFloat64(t, tc, "source")
			source := int(sourceF)

			// Run Dijkstra.
			dist, _ := Dijkstra(adj, weights, source)

			// Parse expected.
			expectedArr, ok := tc.Expected.([]any)
			if !ok {
				t.Fatalf("expected is not an array: %T", tc.Expected)
			}

			n := testutil.InputInt(t, tc, "n")
			if len(dist) < n {
				t.Fatalf("dist length %d < n %d", len(dist), n)
			}

			for i, elem := range expectedArr {
				if s, ok := elem.(string); ok && s == "Inf" {
					if !math.IsInf(dist[i], 1) {
						t.Errorf("dist[%d] = %v, want +Inf", i, dist[i])
					}
				} else {
					want, ok := toTestFloat(elem)
					if !ok {
						t.Fatalf("expected[%d] not numeric: %T", i, elem)
					}
					assertFloat(t, tc.Description, dist[i], want, tc.Tolerance)
				}
			}
		})
	}
}

// toTestFloat converts JSON numeric types to float64.
func toTestFloat(v any) (float64, bool) {
	switch val := v.(type) {
	case float64:
		return val, true
	case int:
		return float64(val), true
	default:
		return 0, false
	}
}

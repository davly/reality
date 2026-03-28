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

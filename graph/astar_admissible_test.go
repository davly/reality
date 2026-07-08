package graph

import (
	"math"
	"testing"
)

// TestAStar_AdmissibleInconsistentHeuristicOptimal pins the reopen fix. AStar's
// docstring promises optimality for any ADMISSIBLE heuristic, but skipping closed
// neighbors made it correct only for CONSISTENT ones: a node closed at a
// suboptimal g was never corrected, yielding a suboptimal path. Here h is
// admissible (every h(n) <= true cost-to-target) but inconsistent (h(2)=2,
// h(3)=0, edge 2->3 cost 1: |h(2)-h(3)|=2 > 1). Optimal is 0->1->2->3->4 = 4;
// the bug returned 0->3->4 = 4.5.
func TestAStar_AdmissibleInconsistentHeuristicOptimal(t *testing.T) {
	adj := IntAdjacency{0: {3, 1}, 1: {2}, 2: {3}, 3: {4}}
	weights := map[[2]int]float64{
		{0, 3}: 3.5, {0, 1}: 1, {1, 2}: 1, {2, 3}: 1, {3, 4}: 1,
	}
	h := func(n int) float64 {
		switch n {
		case 1:
			return 3
		case 2:
			return 2
		default: // 0, 3, 4
			return 0
		}
	}
	path, dist := AStar(adj, weights, 0, 4, h)
	if math.Abs(dist-4.0) > 1e-9 {
		t.Errorf("AStar dist=%v, want 4 (optimal 0->1->2->3->4); path=%v", dist, path)
	}
	want := []int{0, 1, 2, 3, 4}
	if len(path) != len(want) {
		t.Fatalf("AStar path=%v, want %v", path, want)
	}
	for i := range want {
		if path[i] != want[i] {
			t.Errorf("AStar path=%v, want %v", path, want)
			break
		}
	}
}

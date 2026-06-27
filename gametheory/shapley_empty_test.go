package gametheory

import (
	"math"
	"testing"
)

// TestShapleyValue_SampledPathHonorsEmptyCoalition pins the fix. n>12 uses the
// Monte-Carlo sampled path, which seeded prevVal=0 instead of v(empty). For the
// additive game v(c)=5+sum_{i in c}(i+1) (v(empty)=5 != 0), each player's Shapley
// value is exactly its marginal (i+1) regardless of order, and the values sum to
// v(grand)-v(empty). The bug inflated early-permutation players and summed to
// v(grand).
func TestShapleyValue_SampledPathHonorsEmptyCoalition(t *testing.T) {
	n := 13 // > 12 -> sampled path
	charFunc := func(c []bool) float64 {
		v := 5.0
		for i, in := range c {
			if in {
				v += float64(i + 1)
			}
		}
		return v
	}
	got := ShapleyValue(n, charFunc)
	var sum, wantSum float64
	for i := 0; i < n; i++ {
		want := float64(i + 1)
		if math.Abs(got[i]-want) > 1e-6 {
			t.Errorf("ShapleyValue player %d = %v, want %v (additive game)", i, got[i], want)
		}
		sum += got[i]
		wantSum += want
	}
	if math.Abs(sum-wantSum) > 1e-6 {
		t.Errorf("sum of Shapley = %v, want %v (= v(grand) - v(empty))", sum, wantSum)
	}
}

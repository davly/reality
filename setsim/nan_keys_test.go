package setsim

import (
	"math"
	"testing"
)

// TestSetSimilarity_NaNCanonicalized pins the fix. The set helpers used a Go map
// as the set, but NaN != NaN under IEEE-754, so each NaN became a distinct map
// entry -- duplicate NaNs were not collapsed and self-similarity dropped below 1
// (SetJaccard([1,NaN],[1,NaN]) returned 0.3333). NaN is now a single canonical
// element.
func TestSetSimilarity_NaNCanonicalized(t *testing.T) {
	withNaN := []float64{1.0, math.NaN()}
	if j := SetJaccard(withNaN, withNaN); math.Abs(j-1.0) > 1e-12 {
		t.Errorf("SetJaccard(s, s) with NaN = %v, want 1.0", j)
	}
	allNaN := []float64{math.NaN(), math.NaN()}
	if j := SetJaccard(allNaN, allNaN); math.Abs(j-1.0) > 1e-12 {
		t.Errorf("SetJaccard(all-NaN, same) = %v, want 1.0", j)
	}
	v := []float64{0.1, 0.2, math.NaN(), 0.4}
	if d := SetDice(v, v); math.Abs(d-1.0) > 1e-12 {
		t.Errorf("SetDice(v, v) with NaN = %v, want 1.0", d)
	}
	if o := SetOverlapCoefficient(v, v); math.Abs(o-1.0) > 1e-12 {
		t.Errorf("SetOverlapCoefficient(v, v) with NaN = %v, want 1.0", o)
	}
	// Disjoint, one side has NaN: no spurious match (the 2.0 is not in A).
	if j := SetJaccard([]float64{1.0, math.NaN()}, []float64{2.0}); j != 0.0 {
		t.Errorf("SetJaccard({1,NaN},{2}) = %v, want 0", j)
	}
	// Non-NaN element types are unaffected (isNaNElem returns false).
	if j := SetJaccard([]int{1, 2, 3}, []int{2, 3, 4}); math.Abs(j-0.5) > 1e-12 {
		t.Errorf("SetJaccard({1,2,3},{2,3,4}) = %v, want 0.5", j)
	}
}

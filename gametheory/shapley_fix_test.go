package gametheory

import (
	"math"
	"testing"
)

// Regression for the shapleySampled empty-coalition fix. The sampled path
// (n>12) seeded the running value with 0 instead of v(empty), so the first
// (uniformly-random) player in each permutation absorbed the v(empty) baseline,
// producing asymmetric values for a symmetric game and disagreeing with the
// exact path whenever v(empty) != 0.

func TestShapleyValue_SampledNonzeroEmptyCoalition(t *testing.T) {
	// Symmetric game v(S) = 10 + |S| with v(empty) = 10, n = 13 (sampled path).
	// Every player's Shapley value MUST be (v(N) - v(empty)) / n = 13/13 = 1.0;
	// a correct Shapley value of a symmetric game cannot be asymmetric.
	v := func(coal []bool) float64 {
		c := 0
		for _, b := range coal {
			if b {
				c++
			}
		}
		return 10.0 + float64(c)
	}
	vals := ShapleyValue(13, v)
	sum := 0.0
	for i, x := range vals {
		if math.Abs(x-1.0) > 1e-9 {
			t.Errorf("symmetric game: player %d Shapley=%.6f want 1.0 (asymmetric pre-fix)", i, x)
		}
		sum += x
	}
	// Efficiency axiom: sum = v(N) - v(empty) = 23 - 10 = 13.
	if math.Abs(sum-13.0) > 1e-9 {
		t.Errorf("efficiency: sum=%.6f want 13 (= v(N)-v(empty))", sum)
	}
}

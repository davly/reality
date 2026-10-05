package infogeo

import (
	"math"
	"math/rand"
	"slices"
	"testing"
	"time"
)

// medianInsertionReference is the O(n^2) insertion-sort median that median()
// replaced. It is kept here as the equivalence oracle for the replacement.
func medianInsertionReference(xs []float64) float64 {
	n := len(xs)
	for i := 1; i < n; i++ {
		v := xs[i]
		j := i - 1
		for j >= 0 && xs[j] > v {
			xs[j+1] = xs[j]
			j--
		}
		xs[j+1] = v
	}
	if n%2 == 1 {
		return xs[n/2]
	}
	return 0.5 * (xs[n/2-1] + xs[n/2])
}

// TestMedian_MatchesInsertionReference checks that the O(n log n) median is
// bit-identical to the insertion-sort reference on finite inputs, including
// duplicates and signed zeros.
func TestMedian_MatchesInsertionReference(t *testing.T) {
	rng := rand.New(rand.NewSource(20261005))
	pool := []float64{0, math.Copysign(0, -1), 1, -1, 0.5, 2.5, 1e-300, -1e300}
	for trial := 0; trial < 2000; trial++ {
		n := 1 + rng.Intn(64)
		xs := make([]float64, n)
		for i := range xs {
			if rng.Intn(3) == 0 {
				xs[i] = pool[rng.Intn(len(pool))]
			} else {
				xs[i] = rng.NormFloat64() * math.Pow(10, float64(rng.Intn(9)-4))
			}
		}
		got := median(slices.Clone(xs))
		want := medianInsertionReference(slices.Clone(xs))
		if math.Float64bits(got) != math.Float64bits(want) {
			t.Fatalf("trial %d: median %v (bits %x) != reference %v (bits %x) for %v",
				trial, got, math.Float64bits(got), want, math.Float64bits(want), xs)
		}
	}
}

// TestMedianHeuristicBandwidth_LargeInputIsFast pins the complexity fix. At
// 600 points per side (719,400 pairwise distances) the old O(N^4) path takes
// on the order of 90 s even without the race detector; the sorted path takes
// well under a second. The 30 s bound separates the two robustly, including
// on slow CI runners and under -race.
func TestMedianHeuristicBandwidth_LargeInputIsFast(t *testing.T) {
	if testing.Short() {
		t.Skip("large input")
	}
	rng := rand.New(rand.NewSource(7))
	X := sampleNormal2D(rng, 600, 0.0, 1.0)
	Y := sampleNormal2D(rng, 600, 0.0, 1.0)
	start := time.Now()
	bw := MedianHeuristicBandwidth(X, Y)
	elapsed := time.Since(start)
	if !(bw > 0) || math.IsInf(bw, 0) {
		t.Fatalf("bandwidth %v, want a positive finite value", bw)
	}
	if elapsed > 30*time.Second {
		t.Fatalf("MedianHeuristicBandwidth on 1,200 points took %v; the median must stay O(n log n)", elapsed)
	}
}

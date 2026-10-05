package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// ranksInsertionReference is ranks() as it was before the O(n^2) insertion
// sort was replaced; it is kept here as the equivalence oracle.
func ranksInsertionReference(data []float64) []float64 {
	n := len(data)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	for i := 1; i < n; i++ {
		for j := i; j > 0 && data[idx[j]] < data[idx[j-1]]; j-- {
			idx[j], idx[j-1] = idx[j-1], idx[j]
		}
	}
	rnk := make([]float64, n)
	i := 0
	for i < n {
		j := i
		for j < n-1 && data[idx[j+1]] == data[idx[j]] {
			j++
		}
		avgRank := float64(i+j)/2.0 + 1.0
		for k := i; k <= j; k++ {
			rnk[idx[k]] = avgRank
		}
		i = j + 1
	}
	return rnk
}

// TestRanks_MatchesInsertionReference checks the replacement sort gives
// bit-identical average ranks on finite inputs, including ties and signed
// zeros.
func TestRanks_MatchesInsertionReference(t *testing.T) {
	rng := rand.New(rand.NewSource(20261005))
	pool := []float64{0, math.Copysign(0, -1), 1, -1, 2, 3.5}
	for trial := 0; trial < 3000; trial++ {
		n := 1 + rng.Intn(60)
		data := make([]float64, n)
		for i := range data {
			if rng.Intn(2) == 0 {
				data[i] = pool[rng.Intn(len(pool))]
			} else {
				data[i] = rng.NormFloat64()
			}
		}
		got := ranks(data)
		want := ranksInsertionReference(data)
		for i := range got {
			if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
				t.Fatalf("trial %d: rank[%d] %v != reference %v for %v", trial, i, got[i], want[i], data)
			}
		}
	}
}

package prob

import (
	"math"
	"math/rand"
	"testing"
)

// benjaminiHochbergInsertionReference is the procedure exactly as it was
// before the O(m^2) insertion sort was replaced; it is kept here as the
// equivalence oracle for finite p-values.
func benjaminiHochbergInsertionReference(pValues []float64, alpha float64) []bool {
	m := len(pValues)
	if m == 0 {
		return nil
	}
	type indexedP struct {
		p   float64
		idx int
	}
	sorted := make([]indexedP, m)
	for i, p := range pValues {
		sorted[i] = indexedP{p: p, idx: i}
	}
	for i := 1; i < m; i++ {
		key := sorted[i]
		j := i - 1
		for j >= 0 && sorted[j].p > key.p {
			sorted[j+1] = sorted[j]
			j--
		}
		sorted[j+1] = key
	}
	threshold := -1
	mf := float64(m)
	for i := m - 1; i >= 0; i-- {
		rank := float64(i + 1)
		if sorted[i].p <= rank/mf*alpha {
			threshold = i
			break
		}
	}
	result := make([]bool, m)
	if threshold >= 0 {
		for i := 0; i <= threshold; i++ {
			result[sorted[i].idx] = true
		}
	}
	return result
}

// TestBenjaminiHochberg_MatchesInsertionReference checks the replacement sort
// gives identical rejection masks on finite inputs, including heavy ties.
func TestBenjaminiHochberg_MatchesInsertionReference(t *testing.T) {
	rng := rand.New(rand.NewSource(20261005))
	alphas := []float64{0.01, 0.05, 0.1, 0.2}
	tiePool := []float64{0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.5, 1}
	for trial := 0; trial < 3000; trial++ {
		m := 1 + rng.Intn(80)
		p := make([]float64, m)
		for i := range p {
			if rng.Intn(2) == 0 {
				p[i] = tiePool[rng.Intn(len(tiePool))]
			} else {
				p[i] = rng.Float64() * math.Pow(10, -float64(rng.Intn(4)))
			}
		}
		alpha := alphas[rng.Intn(len(alphas))]
		got := BenjaminiHochberg(p, alpha)
		want := benjaminiHochbergInsertionReference(p, alpha)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d alpha %v: mask differs at %d for %v: got %v want %v",
					trial, alpha, i, p, got, want)
			}
		}
	}
}

// TestBenjaminiHochberg_NaNNeverRejected pins the fail-closed handling of a
// NaN p-value, which is outside the valid range: it sorts last and so can
// never fall inside the rejection set.
func TestBenjaminiHochberg_NaNNeverRejected(t *testing.T) {
	p := []float64{0.001, math.NaN(), 0.002, 0.003}
	got := BenjaminiHochberg(p, 0.05)
	if got[1] {
		t.Fatalf("NaN p-value was rejected: %v", got)
	}
	if !got[0] || !got[2] || !got[3] {
		t.Fatalf("finite small p-values should still be rejected: %v", got)
	}
}

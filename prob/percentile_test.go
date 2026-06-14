package prob

import (
	"math"
	"testing"
)

// All "want" values below are HAND-COMPUTED using R-7 linear interpolation:
//   q   = p/100
//   pos = q * (n-1)
//   lo  = floor(pos); frac = pos - lo
//   result = sorted[lo] + (sorted[lo+1] - sorted[lo]) * frac   (pos < n-1)
//   result = sorted[n-1]                                       (pos == n-1)
// Cross-checked against the two reinvented sources:
//   - hearthstone/internal/hmlr.percentile (p in [0,1]): pos = p*(n-1), lerp
//   - insights/internal/topology.percentile (p in [0,100]): rank = (p/100)*(n-1), lerp

const pctEps = 1e-12

func TestQuantile(t *testing.T) {
	tests := []struct {
		name string
		data []float64
		q    float64
		want float64
	}{
		// data {1,2,3,4}, n=4, n-1=3.
		{"q0 of 1234", []float64{1, 2, 3, 4}, 0.00, 1.0},   // pos=0      -> 1
		{"q25 of 1234", []float64{1, 2, 3, 4}, 0.25, 1.75}, // pos=0.75   -> 1 + 1*0.75
		{"q50 of 1234", []float64{1, 2, 3, 4}, 0.50, 2.5},  // pos=1.5    -> 2 + 1*0.5
		{"q75 of 1234", []float64{1, 2, 3, 4}, 0.75, 3.25}, // pos=2.25   -> 3 + 1*0.25
		{"q100 of 1234", []float64{1, 2, 3, 4}, 1.00, 4.0}, // pos=3      -> sorted[n-1]

		// data {1,2,3,4,5}, n=5, n-1=4 (exact ranks).
		{"q25 of 12345", []float64{1, 2, 3, 4, 5}, 0.25, 2.0}, // pos=1.0 -> 2
		{"q50 of 12345", []float64{1, 2, 3, 4, 5}, 0.50, 3.0}, // pos=2.0 -> 3
		{"q10 of 12345", []float64{1, 2, 3, 4, 5}, 0.10, 1.4}, // pos=0.4 -> 1 + 1*0.4

		// Unsorted input must be sorted internally; sorted={1,1,2,3,4,5,6,9}, n=8, n-1=7.
		{"q25 unsorted", []float64{3, 1, 4, 1, 5, 9, 2, 6}, 0.25, 1.75}, // pos=1.75 -> 1 + 1*0.75
		{"q50 unsorted", []float64{3, 1, 4, 1, 5, 9, 2, 6}, 0.50, 3.5},  // pos=3.5  -> 3 + 1*0.5

		// Single element: any quantile returns that element.
		{"single q0", []float64{42}, 0.0, 42},
		{"single q50", []float64{42}, 0.5, 42},
		{"single q100", []float64{42}, 1.0, 42},

		// Two elements, n-1=1: pos = q, direct lerp.
		{"pair q50", []float64{10, 20}, 0.5, 15.0}, // pos=0.5 -> 10 + 10*0.5

		// Out-of-domain q is clamped to the nearest extreme.
		{"q negative clamps to min", []float64{1, 2, 3, 4}, -0.5, 1.0},
		{"q above one clamps to max", []float64{1, 2, 3, 4}, 1.5, 4.0},

		// Negative values, sorted={-5,-1,0,3}, n=4, n-1=3.
		{"q50 negatives", []float64{0, -5, 3, -1}, 0.50, -0.5}, // pos=1.5 -> -1 + (0-(-1))*0.5
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Quantile(tt.data, tt.q)
			if math.Abs(got-tt.want) > pctEps {
				t.Errorf("Quantile(%v, %v) = %v, want %v", tt.data, tt.q, got, tt.want)
			}
		})
	}
}

func TestPercentile(t *testing.T) {
	tests := []struct {
		name string
		data []float64
		p    float64
		want float64
	}{
		// Same {1,2,3,4} reference points, expressed as percentages.
		{"P0", []float64{1, 2, 3, 4}, 0, 1.0},
		{"P25", []float64{1, 2, 3, 4}, 25, 1.75},
		{"P50", []float64{1, 2, 3, 4}, 50, 2.5},
		{"P75", []float64{1, 2, 3, 4}, 75, 3.25},
		{"P100", []float64{1, 2, 3, 4}, 100, 4.0},

		// Percentage clamping mirrors fraction clamping.
		{"P negative clamps", []float64{1, 2, 3, 4}, -10, 1.0},
		{"P over 100 clamps", []float64{1, 2, 3, 4}, 150, 4.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Percentile(tt.data, tt.p)
			if math.Abs(got-tt.want) > pctEps {
				t.Errorf("Percentile(%v, %v) = %v, want %v", tt.data, tt.p, got, tt.want)
			}
		})
	}
}

// Percentile(data, p) must equal Quantile(data, p/100) by construction.
func TestPercentileQuantileConsistency(t *testing.T) {
	data := []float64{3, 1, 4, 1, 5, 9, 2, 6, 5, 3}
	for _, p := range []float64{0, 5, 10, 25, 33.3, 50, 66.7, 75, 90, 100} {
		gotP := Percentile(data, p)
		gotQ := Quantile(data, p/100.0)
		if math.Abs(gotP-gotQ) > pctEps {
			t.Errorf("Percentile(data,%v)=%v != Quantile(data,%v)=%v", p, gotP, p/100.0, gotQ)
		}
	}
}

// Empty input has no empirical percentile: NaN (documented degenerate case).
func TestQuantileEmptyIsNaN(t *testing.T) {
	if got := Quantile(nil, 0.5); !math.IsNaN(got) {
		t.Errorf("Quantile(nil, 0.5) = %v, want NaN", got)
	}
	if got := Percentile([]float64{}, 50); !math.IsNaN(got) {
		t.Errorf("Percentile([], 50) = %v, want NaN", got)
	}
}

// The input slice must not be mutated (a copy is sorted internally).
func TestQuantileDoesNotMutateInput(t *testing.T) {
	data := []float64{3, 1, 4, 1, 5}
	orig := []float64{3, 1, 4, 1, 5}
	_ = Quantile(data, 0.5)
	for i := range data {
		if data[i] != orig[i] {
			t.Fatalf("input mutated at %d: got %v, want %v (full: %v)", i, data[i], orig[i], data)
		}
	}
}

// Cross-check the exact arithmetic of the hearthstone reinvention (p in [0,1],
// integer data): Quantile over the same values must reproduce its float result
// before its int64 truncation. sorted={100,200,300,400}, q=0.5 -> 250.
func TestQuantileMatchesHearthstoneSemantics(t *testing.T) {
	data := []float64{400, 100, 300, 200}
	// hearthstone: pos=0.5*3=1.5, lo=1, frac=0.5, 200 + (300-200)*0.5 = 250.
	if got := Quantile(data, 0.5); math.Abs(got-250) > pctEps {
		t.Errorf("Quantile = %v, want 250 (hearthstone median semantics)", got)
	}
	// P25: pos=0.75 -> 100 + (200-100)*0.75 = 175.
	if got := Quantile(data, 0.25); math.Abs(got-175) > pctEps {
		t.Errorf("Quantile = %v, want 175 (hearthstone P25 semantics)", got)
	}
}

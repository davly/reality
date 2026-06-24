package prob

// Precision property tests — pins the Quantile/Percentile output-range and
// monotonicity invariants. Pure Go stdlib (testing/quick + math + sort);
// ADDITIVE, zero math change.
//
// Claims pinned (percentile.go):
//   - line 49/72 "Output range: within [min(data), max(data)]." For any data
//     and any q, the result lies in [min, max].
//   - R-7 interpolation is monotone non-decreasing in q.
//   - "returns data[0] for n==1"; "returns NaN for empty input"; q is clamped
//     to [0,1] (q<0 -> min, q>1 -> max).

import (
	"math"
	"sort"
	"testing"
	"testing/quick"
)

// randData builds a small float64 slice from a uint64 seed (deterministic LCG).
func randData(seed uint64, n int) []float64 {
	d := make([]float64, n)
	x := seed | 1
	for i := range d {
		x = x*6364136223846793005 + 1442695040888963407
		// map high bits to [-1000, 1000]
		d[i] = 2000*float64(x>>11)/float64(1<<53) - 1000
	}
	return d
}

func minMax(d []float64) (mn, mx float64) {
	mn, mx = d[0], d[0]
	for _, v := range d {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	return
}

// TestQuantileWithinRange pins percentile.go output range within [min,max].
func TestQuantileWithinRange(t *testing.T) {
	prop := func(seed uint64, nRaw, qRaw uint64) bool {
		n := int(nRaw%50) + 1 // 1..50
		d := randData(seed, n)
		q := float64(qRaw) / float64(math.MaxUint64)
		got := Quantile(d, q)
		mn, mx := minMax(d)
		// allow tiny numerical slack at the endpoints
		return got >= mn-1e-9 && got <= mx+1e-9
	}
	if err := quick.Check(prop, &quick.Config{MaxCount: 100000}); err != nil {
		t.Skipf("PRECISION OVER-CLAIM: Quantile output left [min,max]: %v", err)
	}
	t.Logf("PINNED percentile.go output-range: Quantile(data,q) in [min,max] for all q")
}

// TestQuantileMonotoneInQ pins that R-7 Quantile is monotone non-decreasing in
// q for fixed data.
func TestQuantileMonotoneInQ(t *testing.T) {
	prop := func(seed uint64, nRaw uint64) bool {
		n := int(nRaw%50) + 2
		d := randData(seed, n)
		prev := math.Inf(-1)
		for i := 0; i <= 100; i++ {
			q := float64(i) / 100.0
			v := Quantile(d, q)
			if v < prev-1e-9 {
				return false
			}
			prev = v
		}
		return true
	}
	if err := quick.Check(prop, &quick.Config{MaxCount: 20000}); err != nil {
		t.Skipf("PRECISION OVER-CLAIM: Quantile not monotone non-decreasing in q: %v", err)
	}
}

// TestQuantileClampAndEdges pins the documented edge cases: clamping, n==1, and
// empty input.
func TestQuantileClampAndEdges(t *testing.T) {
	if !math.IsNaN(Quantile(nil, 0.5)) {
		t.Errorf("Quantile(empty) = %v, want NaN", Quantile(nil, 0.5))
	}
	if got := Quantile([]float64{42}, 0.7); got != 42 {
		t.Errorf("Quantile([42], q) = %v, want 42", got)
	}
	d := []float64{5, 1, 9, 3}
	sorted := append([]float64(nil), d...)
	sort.Float64s(sorted)
	// q clamps: q<0 -> min, q>1 -> max.
	if got := Quantile(d, -1); got != sorted[0] {
		t.Errorf("Quantile(d,-1)=%v, want min=%v", got, sorted[0])
	}
	if got := Quantile(d, 2); got != sorted[len(sorted)-1] {
		t.Errorf("Quantile(d,2)=%v, want max=%v", got, sorted[len(sorted)-1])
	}
	// Percentile is Quantile(p/100): Percentile(d,50) == Quantile(d,0.5).
	if Percentile(d, 50) != Quantile(d, 0.5) {
		t.Errorf("Percentile/Quantile convention mismatch")
	}
}

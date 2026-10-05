package evt

import (
	"math"
	"testing"
)

// rampData returns 1, 2, ..., n (distinct, ascending).  With it, a tail of k
// observations has the threshold n-k, and exactly k values lie above it.
func rampData(n int) []float64 {
	d := make([]float64, n)
	for i := range d {
		d[i] = float64(i + 1)
	}
	return d
}

// Tail sizes at exact boundaries.  The expected tail is the exact ceiling of
// n*rate for the rate as written (decimal), clamped to [1, n-1]; it was computed
// with exact rational arithmetic (Python fractions), not from the float product.
//
//	floor drops one: the exact count is an integer but the float product lands
//	                 just below it (0.29 * 100 = 28.999999999999996).
//	ceil jumps one:  the exact count is an integer but the float product lands
//	                 just above it (0.07 * 100 = 7.000000000000001), so a
//	                 ceiling without a snap would add an observation.
var thresholdTailCases = []struct {
	n    int
	rate float64
	tail int
}{
	// exact-integer count, float product just below: floor drops one
	{100, 0.29, 29},
	{50, 0.58, 29},
	{90, 0.7, 63},
	{100, 0.57, 57},
	{150, 0.82, 123},
	// exact-integer count, float product just above: a bare ceiling jumps one
	{100, 0.07, 7},
	{25, 0.28, 7},
	{50, 0.56, 28},
	{75, 0.68, 51},
	{50, 0.14, 7},
	// exact-integer count, float product exact
	{10, 0.2, 2},
	{10, 0.5, 5},
	{1000, 0.1, 100},
	{20, 0.05, 1},
	// non-integer count: the ceiling, not the floor
	{10, 0.25, 3},
	{10, 0.15, 2},
	{7, 0.3, 3},
	{3, 0.5, 2},
	{9, 0.5, 5},
	{1000, 0.0015, 2},
	{101, 0.01, 2},
	// clamped to [1, n-1]
	{10, 0.01, 1},
	{10, 0.99, 9},
	{2, 0.999, 1},
	{1000, 0.0001, 1},
	// the documented snap: a count within 1e-9 (relative) of an integer is that
	// integer (7.0000000001 -> 7), one beyond it is not (7.00001 -> 8)
	{1000, 0.0070000000001, 7},
	{1000, 0.00700001, 8},
}

func TestThresholdAtRate_TailSizeIsTheCeilingAtExactBoundaries(t *testing.T) {
	for _, c := range thresholdTailCases {
		data := rampData(c.n)
		u := ThresholdAtRate(data, c.rate)
		if want := float64(c.n - c.tail); u != want {
			t.Errorf("n=%d rate=%v: threshold %v, want %v (tail of %d)", c.n, c.rate, u, want, c.tail)
			continue
		}
		if got := len(Exceedances(data, u)); got != c.tail {
			t.Errorf("n=%d rate=%v: %d observations above the threshold, want %d", c.n, c.rate, got, c.tail)
		}
	}
}

// A rate computed as 1 - conf carries rounding error: n*(1-conf) is
// 1.0000000000000009 for conf = 0.99 and n = 100, and the intended tail is one
// observation, not two.
func TestThresholdAtRate_RateDerivedByAFloatSubtraction(t *testing.T) {
	for _, c := range []struct {
		n    int
		conf float64
	}{{100, 0.99}, {1000, 0.999}, {20, 0.95}, {50, 0.98}, {200, 0.995}, {10, 0.9}} {
		rate := 1 - c.conf
		data := rampData(c.n)
		if u, want := ThresholdAtRate(data, rate), float64(c.n-1); u != want {
			t.Errorf("n=%d conf=%v (rate %v): threshold %v, want %v", c.n, c.conf, rate, u, want)
		}
	}
}

func TestThresholdAtRate_InputOrderDoesNotMatter(t *testing.T) {
	asc := rampData(100)
	desc := make([]float64, len(asc))
	for i, v := range asc {
		desc[len(asc)-1-i] = v
	}
	if a, d := ThresholdAtRate(asc, 0.29), ThresholdAtRate(desc, 0.29); a != d || a != 71 {
		t.Errorf("ascending %v, descending %v, want both 71", a, d)
	}
	if asc[0] != 1 || desc[0] != 100 {
		t.Error("input slices were modified")
	}
}

// With ties across the threshold fewer than ceil(rate*n) observations can lie
// strictly above it, but the threshold is still the order statistic just below
// the tail.
func TestThresholdAtRate_Ties(t *testing.T) {
	data := []float64{1, 2, 2, 2, 3, 3, 4, 5, 5, 5}
	if u := ThresholdAtRate(data, 0.3); u != 4 { // tail = the three 5s
		t.Errorf("rate 0.3: threshold %v, want 4", u)
	}
	if u := ThresholdAtRate(data, 0.2); u != 5 { // tail = two of the three 5s: nothing is strictly above 5
		t.Errorf("rate 0.2: threshold %v, want 5", u)
	}
}

func TestThresholdAtRate_InvalidInputIsNaN(t *testing.T) {
	data := rampData(10)
	for _, rate := range []float64{0, 1, -0.5, 1.5, math.NaN(), math.Inf(1), math.Inf(-1)} {
		if u := ThresholdAtRate(data, rate); !math.IsNaN(u) {
			t.Errorf("rate %v: got %v, want NaN", rate, u)
		}
	}
	if u := ThresholdAtRate(nil, 0.1); !math.IsNaN(u) {
		t.Errorf("empty data: got %v, want NaN", u)
	}
}

func TestThresholdAtRate_SingleObservation(t *testing.T) {
	if u := ThresholdAtRate([]float64{7}, 0.5); u != 7 {
		t.Errorf("got %v, want 7", u)
	}
}

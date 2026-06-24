package transport

import (
	"math"
	"sort"
	"testing"
)

// Independent reference: exact 1-D W1 = integral_x |F_u(x) - F_v(x)| dx over the
// merged empirical support (a different formulation than the function's
// quantile-integral, so a genuine cross-check).
func w1Ref(u, v []float64) float64 {
	su := append([]float64{}, u...)
	sv := append([]float64{}, v...)
	sort.Float64s(su)
	sort.Float64s(sv)
	pts := append(append([]float64{}, su...), sv...)
	sort.Float64s(pts)
	cdf := func(s []float64, x float64) float64 {
		c := 0
		for _, e := range s {
			if e <= x {
				c++
			}
		}
		return float64(c) / float64(len(s))
	}
	area := 0.0
	for i := 0; i+1 < len(pts); i++ {
		mid := (pts[i] + pts[i+1]) / 2
		area += math.Abs(cdf(su, mid)-cdf(sv, mid)) * (pts[i+1] - pts[i])
	}
	return area
}

func TestWasserstein1D_UnequalSizeExact(t *testing.T) {
	cases := [][2][]float64{
		{{0, 3, 6, 9}, {0, 9}},
		{{0, 2}, {1}},
		{{0, 1, 2, 3}, {0, 3}},
		{{5, 1, 9, 3}, {2, 8}}, // unsorted input
		{{1, 2, 3, 4, 5}, {10}},
	}
	for _, c := range cases {
		got, err := Wasserstein1D(c[0], c[1], 1)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		want := w1Ref(c[0], c[1])
		if math.Abs(got-want) > 1e-9 {
			t.Errorf("W1(%v,%v)=%.6f want %.6f (CDF-integral)", c[0], c[1], got, want)
		}
		rev, _ := Wasserstein1D(c[1], c[0], 1)
		if math.Abs(got-rev) > 1e-12 {
			t.Errorf("W1 not symmetric: %v vs %v", got, rev)
		}
	}

	// Identity of indiscernibles: distinct distributions must give > 0 (was 0 pre-fix).
	if d, _ := Wasserstein1D([]float64{0, 3, 6, 9}, []float64{0, 9}, 1); d <= 0 {
		t.Errorf("W1 of distinct distributions = %v; must be > 0", d)
	}

	// Equal-size branch unchanged.
	if eq, _ := Wasserstein1D([]float64{0, 1, 2}, []float64{10, 11, 12}, 1); math.Abs(eq-10) > 1e-12 {
		t.Errorf("equal-size W1=%v want 10", eq)
	}
}

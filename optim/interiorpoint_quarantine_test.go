package optim

import "testing"

// TestInteriorPoint_QuarantinedReturnsError pins the quarantine. The barrier
// iteration was fundamentally broken: it diverged to NaN/garbage on essentially
// every well-posed LP while returning a nil error (e.g. min -x1 s.t. x1<=1
// returned x=[3.8e86], and a 2D LP returned x=[NaN,NaN]). It must now fail
// closed (return an error), not a wrong/garbage solution.
func TestInteriorPoint_QuarantinedReturnsError(t *testing.T) {
	cases := []struct {
		name string
		c    []float64
		A    [][]float64
		b    []float64
	}{
		{"trivial", []float64{-1}, [][]float64{{1}}, []float64{1}},
		{"2d", []float64{-1, -1}, [][]float64{{1, 2}, {3, 2}}, []float64{4, 6}},
	}
	for _, tc := range cases {
		x, val, err := InteriorPoint(tc.c, tc.A, tc.b)
		if err == nil {
			t.Errorf("%s: InteriorPoint should return an error (quarantined); got x=%v val=%v nil err", tc.name, x, val)
		}
	}
	// The empty-problem validation still returns its specific error (regression).
	if _, _, err := InteriorPoint(nil, nil, nil); err == nil {
		t.Error("InteriorPoint(empty) should error")
	}
}

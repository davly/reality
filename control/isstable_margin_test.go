package control

import "testing"

// TestIsStable_MarginalPolesNotStable pins the fix for the false-stable verdict.
// A pole ON the imaginary axis (Re=0) is marginally stable -> NOT BIBO-stable.
// Durand-Kerner places such poles at Re ~ 1e-38 (FP noise), and the old
// zero-tolerance `real(p) >= 0` test mis-classified these degree-3+ systems as
// stable -- inconsistently with the degree-2 path, which already calls s(s+1)
// unstable. The epsilon margin must report all of them unstable.
func TestIsStable_MarginalPolesNotStable(t *testing.T) {
	cases := []struct {
		name  string
		denom []float64 // highest degree first
	}{
		{"s(s+1)(s+2) [integrator]", []float64{1, 3, 2, 0}},
		{"s(s+1)(s+2)(s+3)", []float64{1, 6, 11, 6, 0}},
		{"s(s+1)^2", []float64{1, 2, 1, 0}},
		{"(s^2+1)^2 [jw-axis]", []float64{1, 0, 2, 0, 1}},
		{"s(s+1) [degree-2 anchor]", []float64{1, 1, 0}},
	}
	for _, tc := range cases {
		tf := TransferFunction{Numerator: []float64{1}, Denominator: tc.denom}
		if tf.IsStable() {
			t.Errorf("%s: IsStable()=true, want false (marginal/jw-axis pole is not strictly stable)", tc.name)
		}
	}
	// A genuinely stable system must still be reported stable.
	stable := TransferFunction{Numerator: []float64{1}, Denominator: []float64{1, 6, 11, 6}} // (s+1)(s+2)(s+3)
	if !stable.IsStable() {
		t.Error("(s+1)(s+2)(s+3): IsStable()=false, want true (poles -1,-2,-3)")
	}
}

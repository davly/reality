package control

import "testing"

// Regression for the IsStable jw-axis fix. The previous `real(p) >= 0` test with
// no tolerance reported degree>=3 systems with poles exactly on the imaginary
// axis as STABLE, because Durand-Kerner returns those poles with a ~1e-18 real
// part of arbitrary (roundoff) sign.

func TestIsStable_JwAxisPoles(t *testing.T) {
	// (s^2+0.25)(s+2) = s^3+2s^2+0.25s+0.5 : poles -2, +-0.5j -> marginally UNSTABLE.
	tf := TransferFunction{Numerator: []float64{1}, Denominator: []float64{1, 2, 0.25, 0.5}}
	if tf.IsStable() {
		t.Error("IsStable((s^2+0.25)(s+2))=true; want false (poles on jw axis)")
	}
	// degree-2 jw-axis (s^2+0.25): already-correct path, must stay false.
	if (&TransferFunction{Numerator: []float64{1}, Denominator: []float64{1, 0, 0.25}}).IsStable() {
		t.Error("IsStable(s^2+0.25)=true; want false")
	}
}

func TestIsStable_GenuineCases(t *testing.T) {
	// (s+1)(s+2) = s^2+3s+2 : stable.
	if !(&TransferFunction{Numerator: []float64{1}, Denominator: []float64{1, 3, 2}}).IsStable() {
		t.Error("IsStable((s+1)(s+2))=false; want true")
	}
	// Lightly-damped but genuinely stable degree-3: poles -0.01+-j, -1.
	// (s^2+0.02s+1.0001)(s+1) = s^3+1.02s^2+1.0201s+1.0001.
	if !(&TransferFunction{Numerator: []float64{1}, Denominator: []float64{1, 1.02, 1.0201, 1.0001}}).IsStable() {
		t.Error("IsStable(lightly-damped stable degree-3)=false; want true (not over-flagged)")
	}
	// Unstable: (s-1)(s-2) = s^2-3s+2 -> poles +1,+2.
	if (&TransferFunction{Numerator: []float64{1}, Denominator: []float64{1, -3, 2}}).IsStable() {
		t.Error("IsStable((s-1)(s-2))=true; want false")
	}
}

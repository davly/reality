package persistent

import (
	"errors"
	"math"
	"testing"
)

// TestVietorisRipsComplex_NonFinitePoint pins the fix: validateVRInput checked
// the cloud shape and maxRadius finiteness but never the coordinate values, so a
// NaN/Inf coordinate poisoned every distance to that point and the barcode came
// out wrong (extra spurious essential bars) with no error. Non-finite coords
// must now be rejected with ErrNonFinitePoint.
func TestVietorisRipsComplex_NonFinitePoint(t *testing.T) {
	if _, err := VietorisRipsComplex([][]float64{{0, 0}, {1, 0}, {math.NaN(), 1}, {0, 1}}, 5.0, 1); !errors.Is(err, ErrNonFinitePoint) {
		t.Errorf("NaN coordinate: err=%v, want ErrNonFinitePoint", err)
	}
	if _, err := VietorisRipsComplex([][]float64{{0, 0}, {math.Inf(1), 0}, {1, 1}}, 5.0, 1); !errors.Is(err, ErrNonFinitePoint) {
		t.Errorf("Inf coordinate: err=%v, want ErrNonFinitePoint", err)
	}
	// A clean cloud must still succeed.
	if _, err := VietorisRipsComplex([][]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}}, 5.0, 1); err != nil {
		t.Errorf("clean cloud: err=%v, want nil", err)
	}
}

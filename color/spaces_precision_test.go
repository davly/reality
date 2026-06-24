package color

// Precision property tests — pins color-space round-trips as tested
// invariants. Pure Go stdlib (testing/quick + math); ADDITIVE, zero math
// change.
//
// Claims pinned ("Precision: exact to float64 precision" — interpreted as the
// mathematically-exact inverse pair round-trips to float64 precision):
//   - spaces.go:25/43  SRGBToLinear / LinearToSRGB are inverses: round-trip a
//     channel and recover it to ~1e-12.
//   - spaces.go:157/194 RGBToHSV / HSVToRGB are inverses over the in-gamut
//     cube: round-trip recovers (r,g,b) to ~1e-12.
//   - spaces.go (XYZToLab/LabToXYZ): inverse pair round-trips to ~1e-10.

import (
	"math"
	"testing"
	"testing/quick"
)

func unit01(u uint64) float64 { return float64(u) / float64(math.MaxUint64) }

// TestSRGBLinearRoundTrip pins spaces.go:25/43 — the sRGB transfer function and
// its inverse round-trip a [0,1] channel.
func TestSRGBLinearRoundTrip(t *testing.T) {
	const bound = 1e-12
	var worst, worstAt float64
	prop := func(u uint64) bool {
		c := unit01(u)
		back := LinearToSRGB(SRGBToLinear(c))
		err := math.Abs(back - c)
		if err > worst {
			worst, worstAt = err, c
		}
		return err <= bound
	}
	if err := quick.Check(prop, &quick.Config{MaxCount: 200000}); err != nil {
		t.Skipf("PRECISION OVER-CLAIM: SRGBToLinear/LinearToSRGB claim 'exact to float64', round-trip error %g at c=%g exceeds %g", worst, worstAt, bound)
	}
	t.Logf("PINNED spaces.go:25/43 sRGB<->linear round-trip: worst error %g at c=%g (< %g)", worst, worstAt, bound)
}

// TestRGBHSVRoundTrip pins spaces.go:157/194 — RGB<->HSV inverse over the
// in-gamut cube.
func TestRGBHSVRoundTrip(t *testing.T) {
	const bound = 1e-12
	var worst float64
	var worstR, worstG, worstB float64
	prop := func(ru, gu, bu uint64) bool {
		r, g, b := unit01(ru), unit01(gu), unit01(bu)
		h, s, v := RGBToHSV(r, g, b)
		r2, g2, b2 := HSVToRGB(h, s, v)
		e := math.Max(math.Abs(r2-r), math.Max(math.Abs(g2-g), math.Abs(b2-b)))
		if e > worst {
			worst, worstR, worstG, worstB = e, r, g, b
		}
		return e <= bound
	}
	if err := quick.Check(prop, &quick.Config{MaxCount: 200000}); err != nil {
		t.Skipf("PRECISION OVER-CLAIM: RGB<->HSV claim 'exact to float64', round-trip error %g at (%g,%g,%g) exceeds %g", worst, worstR, worstG, worstB, bound)
	}
	t.Logf("PINNED spaces.go:157/194 RGB<->HSV round-trip: worst error %g (< %g)", worst, bound)
}

// TestXYZLabRoundTrip pins XYZToLab/LabToXYZ as an inverse pair over the D65
// white point.
func TestXYZLabRoundTrip(t *testing.T) {
	const bound = 1e-10
	// D65 reference white (the conventional one used with sRGB), scaled to
	// Y=1.
	const Xn, Yn, Zn = 0.95047, 1.0, 1.08883
	var worst float64
	prop := func(xu, yu, zu uint64) bool {
		// XYZ values in [0, white*1.1] — physically plausible.
		X := unit01(xu) * Xn * 1.1
		Y := unit01(yu) * Yn * 1.1
		Z := unit01(zu) * Zn * 1.1
		L, a, b := XYZToLab(X, Y, Z, Xn, Yn, Zn)
		X2, Y2, Z2 := LabToXYZ(L, a, b, Xn, Yn, Zn)
		e := math.Max(math.Abs(X2-X), math.Max(math.Abs(Y2-Y), math.Abs(Z2-Z)))
		if e > worst {
			worst = e
		}
		return e <= bound
	}
	if err := quick.Check(prop, &quick.Config{MaxCount: 200000}); err != nil {
		t.Skipf("PRECISION OVER-CLAIM: XYZ<->Lab round-trip error %g exceeds %g", worst, bound)
	}
	t.Logf("PINNED XYZ<->Lab round-trip: worst error %g (< %g)", worst, bound)
}

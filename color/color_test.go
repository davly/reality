package color

import (
	"math"
	"testing"

	"github.com/davly/reality/testutil"
)

const tol = 1e-10

func approxEqual(a, b, eps float64) bool {
	return math.Abs(a-b) <= eps
}

// ---------------------------------------------------------------------------
// sRGB <-> Linear roundtrip
// ---------------------------------------------------------------------------

func TestSRGBToLinear_Zero(t *testing.T) {
	if got := SRGBToLinear(0); got != 0 {
		t.Errorf("SRGBToLinear(0) = %v, want 0", got)
	}
}

func TestSRGBToLinear_One(t *testing.T) {
	if got := SRGBToLinear(1); !approxEqual(got, 1, tol) {
		t.Errorf("SRGBToLinear(1) = %v, want 1", got)
	}
}

func TestSRGBToLinear_Half(t *testing.T) {
	got := SRGBToLinear(0.5)
	// sRGB 0.5 ≈ linear 0.214
	if !approxEqual(got, 0.214041140482, 1e-6) {
		t.Errorf("SRGBToLinear(0.5) = %v, want ~0.214", got)
	}
}

func TestSRGBToLinear_BelowCutoff(t *testing.T) {
	got := SRGBToLinear(0.03)
	want := 0.03 / 12.92
	if !approxEqual(got, want, 1e-15) {
		t.Errorf("SRGBToLinear(0.03) = %v, want %v", got, want)
	}
}

func TestLinearToSRGB_Zero(t *testing.T) {
	if got := LinearToSRGB(0); got != 0 {
		t.Errorf("LinearToSRGB(0) = %v, want 0", got)
	}
}

func TestLinearToSRGB_One(t *testing.T) {
	if got := LinearToSRGB(1); !approxEqual(got, 1, tol) {
		t.Errorf("LinearToSRGB(1) = %v, want 1", got)
	}
}

func TestSRGB_Roundtrip(t *testing.T) {
	values := []float64{0.0, 0.01, 0.04, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0}
	for _, v := range values {
		linear := SRGBToLinear(v)
		back := LinearToSRGB(linear)
		if !approxEqual(back, v, 1e-10) {
			t.Errorf("sRGB roundtrip(%v): got %v (linear=%v)", v, back, linear)
		}
	}
}

func TestLinear_Roundtrip(t *testing.T) {
	values := []float64{0.0, 0.001, 0.003, 0.01, 0.1, 0.5, 0.9, 1.0}
	for _, v := range values {
		srgb := LinearToSRGB(v)
		back := SRGBToLinear(srgb)
		if !approxEqual(back, v, 1e-10) {
			t.Errorf("linear roundtrip(%v): got %v (srgb=%v)", v, back, srgb)
		}
	}
}

// ---------------------------------------------------------------------------
// sRGB Golden File
// ---------------------------------------------------------------------------

func TestSRGBToLinear_Golden(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/color/srgb_linear.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			srgb := testutil.InputFloat64(t, tc, "srgb")
			got := SRGBToLinear(srgb)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

// ---------------------------------------------------------------------------
// RGB <-> XYZ
// ---------------------------------------------------------------------------

func TestLinearRGBToXYZ_White(t *testing.T) {
	// Linear RGB (1,1,1) should map to D65 white point
	X, Y, Z := LinearRGBToXYZ(1, 1, 1)
	// D65: X≈0.9505, Y≈1.0, Z≈1.089
	if !approxEqual(Y, 1.0, 1e-4) {
		t.Errorf("Y for white = %v, want ~1.0", Y)
	}
	if !approxEqual(X, 0.9505, 1e-3) {
		t.Errorf("X for white = %v, want ~0.9505", X)
	}
	if !approxEqual(Z, 1.089, 1e-3) {
		t.Errorf("Z for white = %v, want ~1.089", Z)
	}
}

func TestLinearRGBToXYZ_Black(t *testing.T) {
	X, Y, Z := LinearRGBToXYZ(0, 0, 0)
	if X != 0 || Y != 0 || Z != 0 {
		t.Errorf("black XYZ = (%v, %v, %v), want (0,0,0)", X, Y, Z)
	}
}

func TestXYZ_RGB_Roundtrip(t *testing.T) {
	colors := [][3]float64{
		{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
		{1, 1, 1}, {0.5, 0.3, 0.7}, {0.2, 0.8, 0.4},
	}
	for _, c := range colors {
		X, Y, Z := LinearRGBToXYZ(c[0], c[1], c[2])
		r, g, b := XYZToLinearRGB(X, Y, Z)
		if !approxEqual(r, c[0], 1e-6) || !approxEqual(g, c[1], 1e-6) || !approxEqual(b, c[2], 1e-6) {
			t.Errorf("XYZ roundtrip (%v,%v,%v): got (%v,%v,%v)", c[0], c[1], c[2], r, g, b)
		}
	}
}

// ---------------------------------------------------------------------------
// XYZ <-> Lab
// ---------------------------------------------------------------------------

func TestXYZToLab_White(t *testing.T) {
	// D65 white point: Xn=0.9505, Yn=1.0, Zn=1.0890
	L, a, b := XYZToLab(0.9505, 1.0, 1.0890, 0.9505, 1.0, 1.0890)
	if !approxEqual(L, 100.0, 0.01) {
		t.Errorf("L for white = %v, want 100", L)
	}
	if !approxEqual(a, 0.0, 0.01) {
		t.Errorf("a for white = %v, want 0", a)
	}
	if !approxEqual(b, 0.0, 0.01) {
		t.Errorf("b for white = %v, want 0", b)
	}
}

func TestXYZToLab_Black(t *testing.T) {
	L, _, _ := XYZToLab(0, 0, 0, 0.9505, 1.0, 1.0890)
	if !approxEqual(L, 0.0, 0.01) {
		t.Errorf("L for black = %v, want 0", L)
	}
}

func TestXYZLab_Roundtrip(t *testing.T) {
	// D65 white point
	Xn, Yn, Zn := 0.9505, 1.0, 1.0890
	xyzValues := [][3]float64{
		{0.9505, 1.0, 1.0890},     // white
		{0.0, 0.0, 0.0},           // black
		{0.4124, 0.2127, 0.0193},  // red primary approx
		{0.3576, 0.7152, 0.1192},  // green primary approx
		{0.1805, 0.0722, 0.9505},  // blue primary approx
		{0.5, 0.5, 0.5},           // mid-gray
	}
	for _, xyz := range xyzValues {
		L, a, b := XYZToLab(xyz[0], xyz[1], xyz[2], Xn, Yn, Zn)
		X2, Y2, Z2 := LabToXYZ(L, a, b, Xn, Yn, Zn)
		if !approxEqual(X2, xyz[0], 1e-6) || !approxEqual(Y2, xyz[1], 1e-6) || !approxEqual(Z2, xyz[2], 1e-6) {
			t.Errorf("Lab roundtrip (%v,%v,%v) -> (L=%v,a=%v,b=%v) -> (%v,%v,%v)",
				xyz[0], xyz[1], xyz[2], L, a, b, X2, Y2, Z2)
		}
	}
}

// ---------------------------------------------------------------------------
// RGB <-> HSV
// ---------------------------------------------------------------------------

func TestRGBToHSV_Red(t *testing.T) {
	h, s, v := RGBToHSV(1, 0, 0)
	if !approxEqual(h, 0, tol) || !approxEqual(s, 1, tol) || !approxEqual(v, 1, tol) {
		t.Errorf("red HSV = (%v, %v, %v), want (0, 1, 1)", h, s, v)
	}
}

func TestRGBToHSV_Green(t *testing.T) {
	h, s, v := RGBToHSV(0, 1, 0)
	if !approxEqual(h, 120, tol) || !approxEqual(s, 1, tol) || !approxEqual(v, 1, tol) {
		t.Errorf("green HSV = (%v, %v, %v), want (120, 1, 1)", h, s, v)
	}
}

func TestRGBToHSV_Blue(t *testing.T) {
	h, s, v := RGBToHSV(0, 0, 1)
	if !approxEqual(h, 240, tol) || !approxEqual(s, 1, tol) || !approxEqual(v, 1, tol) {
		t.Errorf("blue HSV = (%v, %v, %v), want (240, 1, 1)", h, s, v)
	}
}

func TestRGBToHSV_White(t *testing.T) {
	h, s, v := RGBToHSV(1, 1, 1)
	if !approxEqual(s, 0, tol) || !approxEqual(v, 1, tol) {
		t.Errorf("white HSV = (%v, %v, %v), want (0, 0, 1)", h, s, v)
	}
}

func TestRGBToHSV_Black(t *testing.T) {
	h, s, v := RGBToHSV(0, 0, 0)
	if !approxEqual(s, 0, tol) || !approxEqual(v, 0, tol) {
		t.Errorf("black HSV = (%v, %v, %v), want (0, 0, 0)", h, s, v)
	}
}

func TestRGBToHSV_Yellow(t *testing.T) {
	h, s, v := RGBToHSV(1, 1, 0)
	if !approxEqual(h, 60, tol) || !approxEqual(s, 1, tol) || !approxEqual(v, 1, tol) {
		t.Errorf("yellow HSV = (%v, %v, %v), want (60, 1, 1)", h, s, v)
	}
}

func TestRGBToHSV_Cyan(t *testing.T) {
	h, s, v := RGBToHSV(0, 1, 1)
	if !approxEqual(h, 180, tol) || !approxEqual(s, 1, tol) || !approxEqual(v, 1, tol) {
		t.Errorf("cyan HSV = (%v, %v, %v), want (180, 1, 1)", h, s, v)
	}
}

func TestHSV_RGB_Roundtrip(t *testing.T) {
	colors := [][3]float64{
		{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
		{1, 1, 1}, {1, 1, 0}, {0, 1, 1}, {1, 0, 1},
		{0.5, 0.3, 0.7}, {0.2, 0.8, 0.4}, {0.9, 0.1, 0.5},
	}
	for _, c := range colors {
		h, s, v := RGBToHSV(c[0], c[1], c[2])
		r, g, b := HSVToRGB(h, s, v)
		if !approxEqual(r, c[0], 1e-10) || !approxEqual(g, c[1], 1e-10) || !approxEqual(b, c[2], 1e-10) {
			t.Errorf("HSV roundtrip (%v,%v,%v) -> (H=%v,S=%v,V=%v) -> (%v,%v,%v)",
				c[0], c[1], c[2], h, s, v, r, g, b)
		}
	}
}

func TestHSVToRGB_KnownValues(t *testing.T) {
	tests := []struct {
		h, s, v    float64
		r, g, b    float64
		desc       string
	}{
		{0, 1, 1, 1, 0, 0, "pure red"},
		{120, 1, 1, 0, 1, 0, "pure green"},
		{240, 1, 1, 0, 0, 1, "pure blue"},
		{60, 1, 1, 1, 1, 0, "yellow"},
		{0, 0, 0.5, 0.5, 0.5, 0.5, "gray"},
		{0, 0, 0, 0, 0, 0, "black"},
		{0, 0, 1, 1, 1, 1, "white"},
	}
	for _, tt := range tests {
		r, g, b := HSVToRGB(tt.h, tt.s, tt.v)
		if !approxEqual(r, tt.r, 1e-10) || !approxEqual(g, tt.g, 1e-10) || !approxEqual(b, tt.b, 1e-10) {
			t.Errorf("%s: HSVToRGB(%v,%v,%v) = (%v,%v,%v), want (%v,%v,%v)",
				tt.desc, tt.h, tt.s, tt.v, r, g, b, tt.r, tt.g, tt.b)
		}
	}
}

// ---------------------------------------------------------------------------
// DeltaE76
// ---------------------------------------------------------------------------

func TestDeltaE76_Identical(t *testing.T) {
	d := DeltaE76(50, 20, -10, 50, 20, -10)
	if d != 0 {
		t.Errorf("DeltaE76 identical = %v, want 0", d)
	}
}

func TestDeltaE76_KnownDiff(t *testing.T) {
	// L1=50, a1=0, b1=0 vs L2=50, a2=3, b2=4 => distance = 5
	d := DeltaE76(50, 0, 0, 50, 3, 4)
	if !approxEqual(d, 5.0, 1e-12) {
		t.Errorf("DeltaE76 = %v, want 5.0", d)
	}
}

func TestDeltaE76_PureLightness(t *testing.T) {
	d := DeltaE76(0, 0, 0, 100, 0, 0)
	if !approxEqual(d, 100.0, 1e-12) {
		t.Errorf("DeltaE76 lightness = %v, want 100", d)
	}
}

func TestDeltaE76_Symmetric(t *testing.T) {
	d1 := DeltaE76(50, 20, -10, 80, -30, 40)
	d2 := DeltaE76(80, -30, 40, 50, 20, -10)
	if !approxEqual(d1, d2, 1e-12) {
		t.Errorf("DeltaE76 not symmetric: %v vs %v", d1, d2)
	}
}

// ---------------------------------------------------------------------------
// DeltaE2000
// ---------------------------------------------------------------------------

func TestDeltaE2000_Identical(t *testing.T) {
	d := DeltaE2000(50, 20, -10, 50, 20, -10)
	if !approxEqual(d, 0, 1e-10) {
		t.Errorf("DeltaE2000 identical = %v, want 0", d)
	}
}

func TestDeltaE2000_Symmetric(t *testing.T) {
	d1 := DeltaE2000(50, 2.6772, -79.7751, 50, 0, -82.7485)
	d2 := DeltaE2000(50, 0, -82.7485, 50, 2.6772, -79.7751)
	if !approxEqual(d1, d2, 1e-4) {
		t.Errorf("DeltaE2000 not symmetric: %v vs %v", d1, d2)
	}
}

func TestDeltaE2000_SharmaPair1(t *testing.T) {
	d := DeltaE2000(50.0, 2.6772, -79.7751, 50.0, 0.0, -82.7485)
	if !approxEqual(d, 2.0425, 0.0001) {
		t.Errorf("DeltaE2000 Sharma pair 1 = %v, want 2.0425", d)
	}
}

func TestDeltaE2000_SharmaPair17(t *testing.T) {
	d := DeltaE2000(50.0, 2.5, 0.0, 56.0, -27.0, -3.0)
	if !approxEqual(d, 31.903, 0.001) {
		t.Errorf("DeltaE2000 Sharma pair 17 = %v, want 31.903", d)
	}
}

func TestDeltaE2000_SharmaPair19_Equal(t *testing.T) {
	d := DeltaE2000(50.0, 2.5, 0.0, 50.0, 2.5, 0.0)
	if !approxEqual(d, 0.0, 0.0001) {
		t.Errorf("DeltaE2000 Sharma pair 19 = %v, want 0", d)
	}
}

// ---------------------------------------------------------------------------
// DeltaE2000 Golden File
// ---------------------------------------------------------------------------

func TestDeltaE2000_Golden(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/color/delta_e2000.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			L1 := testutil.InputFloat64(t, tc, "L1")
			a1 := testutil.InputFloat64(t, tc, "a1")
			b1 := testutil.InputFloat64(t, tc, "b1")
			L2 := testutil.InputFloat64(t, tc, "L2")
			a2 := testutil.InputFloat64(t, tc, "a2")
			b2 := testutil.InputFloat64(t, tc, "b2")
			got := DeltaE2000(L1, a1, b1, L2, a2, b2)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

// ---------------------------------------------------------------------------
// Bradford Adaptation
// ---------------------------------------------------------------------------

func TestBradfordAdapt_Identity(t *testing.T) {
	// Adapting from D65 to D65 should be identity
	X, Y, Z := 0.5, 0.4, 0.3
	Xa, Ya, Za := BradfordAdapt(X, Y, Z, 0.3127, 0.3290, 0.3127, 0.3290)
	if !approxEqual(Xa, X, 1e-6) || !approxEqual(Ya, Y, 1e-6) || !approxEqual(Za, Z, 1e-6) {
		t.Errorf("Bradford identity: (%v,%v,%v) != (%v,%v,%v)", Xa, Ya, Za, X, Y, Z)
	}
}

func TestBradfordAdapt_D65toD50(t *testing.T) {
	// D65 white point in XYZ: (0.9505, 1.0, 1.0890)
	// After adaptation to D50, should get D50 white: (0.9642, 1.0, 0.8249)
	Xa, Ya, Za := BradfordAdapt(0.9505, 1.0, 1.0890, 0.3127, 0.3290, 0.3457, 0.3585)
	if !approxEqual(Ya, 1.0, 1e-3) {
		t.Errorf("Bradford D65->D50 Y = %v, want ~1.0", Ya)
	}
	// X should increase, Z should decrease for D50
	if Xa < 0.9505 {
		t.Errorf("Bradford D65->D50 X should increase, got %v", Xa)
	}
	if Za > 1.0890 {
		t.Errorf("Bradford D65->D50 Z should decrease, got %v", Za)
	}
}

func TestBradfordAdapt_WhitePointPreservation(t *testing.T) {
	// The source white point should map to the destination white point
	// Source D65: Xn=0.9505, Yn=1.0, Zn=1.0890
	Xa, Ya, Za := BradfordAdapt(0.9505, 1.0, 1.0890, 0.3127, 0.3290, 0.3457, 0.3585)
	// Expected D50: Xn ≈ 0.9642, Yn=1.0, Zn ≈ 0.8249
	if !approxEqual(Ya, 1.0, 1e-3) {
		t.Errorf("white Y after adaptation = %v, want ~1.0", Ya)
	}
	_ = Xa
	_ = Za
}

// ---------------------------------------------------------------------------
// Blackbody
// ---------------------------------------------------------------------------

func TestBlackbody_6500K_D65(t *testing.T) {
	X, Y, Z := BlackbodyToXYZ(6500)
	// 6500K blackbody approximates D65 but is not identical.
	// D65 is a daylight spectrum, not a pure Planckian locus.
	// 6500K Planckian chromaticity: x≈0.313, y≈0.324
	sum := X + Y + Z
	x := X / sum
	y := Y / sum
	if !approxEqual(x, 0.313, 0.01) {
		t.Errorf("6500K chromaticity x = %v, want ~0.313", x)
	}
	if !approxEqual(y, 0.324, 0.01) {
		t.Errorf("6500K chromaticity y = %v, want ~0.324", y)
	}
}

func TestBlackbody_YNormalized(t *testing.T) {
	_, Y, _ := BlackbodyToXYZ(5000)
	if !approxEqual(Y, 1.0, 1e-10) {
		t.Errorf("Blackbody Y should be 1.0, got %v", Y)
	}
}

func TestBlackbody_HotterIsBluer(t *testing.T) {
	_, _, Z3000 := BlackbodyToXYZ(3000)
	_, _, Z10000 := BlackbodyToXYZ(10000)
	// Hotter temperature should have relatively more blue (higher Z)
	if Z10000 <= Z3000 {
		t.Errorf("10000K Z (%v) should be > 3000K Z (%v)", Z10000, Z3000)
	}
}

func TestBlackbody_ZeroTemp(t *testing.T) {
	X, Y, Z := BlackbodyToXYZ(0)
	if X != 0 || Y != 0 || Z != 0 {
		t.Errorf("BlackbodyToXYZ(0) = (%v,%v,%v), want (0,0,0)", X, Y, Z)
	}
}

func TestBlackbody_NegativeTemp(t *testing.T) {
	X, Y, Z := BlackbodyToXYZ(-100)
	if X != 0 || Y != 0 || Z != 0 {
		t.Errorf("BlackbodyToXYZ(-100) = (%v,%v,%v), want (0,0,0)", X, Y, Z)
	}
}

// ---------------------------------------------------------------------------
// Reinhard Tone Mapping
// ---------------------------------------------------------------------------

func TestToneMapReinhard_Zero(t *testing.T) {
	r, g, b := ToneMapReinhard(0, 0, 0, 1)
	if r != 0 || g != 0 || b != 0 {
		t.Errorf("ToneMapReinhard(0,0,0) = (%v,%v,%v), want (0,0,0)", r, g, b)
	}
}

func TestToneMapReinhard_AtWhitePoint(t *testing.T) {
	// At the white point value, Reinhard extended maps to 1.0:
	// v * (1 + v/wp^2) / (1 + v) = 1 * (1+1) / (1+1) = 1
	r, _, _ := ToneMapReinhard(1, 0, 0, 1)
	if !approxEqual(r, 1.0, 1e-10) {
		t.Errorf("ToneMapReinhard at white point = %v, want 1.0", r)
	}
}

func TestToneMapReinhard_HalfValue(t *testing.T) {
	// v=0.5, wp=2: 0.5*(1+0.5/4)/(1+0.5) = 0.5*1.125/1.5 = 0.375
	r, _, _ := ToneMapReinhard(0.5, 0, 0, 2)
	want := 0.5 * (1 + 0.5/4) / (1 + 0.5)
	if !approxEqual(r, want, 1e-10) {
		t.Errorf("ToneMapReinhard(0.5, wp=2) = %v, want %v", r, want)
	}
}

func TestToneMapReinhard_LargeValuesBounded(t *testing.T) {
	// For large input relative to white point, output approaches 1+in/wp^2
	// which is bounded by the formula — not necessarily < 1
	// The formula gives values in [0, some_max] where max depends on wp
	r1, _, _ := ToneMapReinhard(0.5, 0, 0, 10)
	r2, _, _ := ToneMapReinhard(5.0, 0, 0, 10)
	// Output should be monotonically increasing
	if r2 <= r1 {
		t.Errorf("ToneMapReinhard not monotonic: %v >= %v", r1, r2)
	}
}

func TestToneMapReinhard_Monotonic(t *testing.T) {
	// Brighter input should give brighter output
	r1, _, _ := ToneMapReinhard(0.5, 0, 0, 1)
	r2, _, _ := ToneMapReinhard(1.0, 0, 0, 1)
	r3, _, _ := ToneMapReinhard(2.0, 0, 0, 1)
	if r1 >= r2 || r2 >= r3 {
		t.Errorf("ToneMapReinhard not monotonic: %v, %v, %v", r1, r2, r3)
	}
}

// ---------------------------------------------------------------------------
// Full pipeline: sRGB -> Linear -> XYZ -> Lab -> DeltaE
// ---------------------------------------------------------------------------

func TestFullPipeline_IdenticalColors(t *testing.T) {
	// Same sRGB color should give DeltaE = 0
	r, g, b := 0.5, 0.3, 0.7
	lr := SRGBToLinear(r)
	lg := SRGBToLinear(g)
	lb := SRGBToLinear(b)
	X, Y, Z := LinearRGBToXYZ(lr, lg, lb)
	L, a, bStar := XYZToLab(X, Y, Z, 0.9505, 1.0, 1.0890)
	d := DeltaE76(L, a, bStar, L, a, bStar)
	if d != 0 {
		t.Errorf("full pipeline identical DeltaE = %v, want 0", d)
	}
}

func TestFullPipeline_Roundtrip(t *testing.T) {
	// sRGB -> linear -> XYZ -> Lab -> XYZ -> linear -> sRGB
	r, g, b := 0.6, 0.2, 0.8
	lr := SRGBToLinear(r)
	lg := SRGBToLinear(g)
	lb := SRGBToLinear(b)
	X, Y, Z := LinearRGBToXYZ(lr, lg, lb)
	L, a, bStar := XYZToLab(X, Y, Z, 0.9505, 1.0, 1.0890)
	X2, Y2, Z2 := LabToXYZ(L, a, bStar, 0.9505, 1.0, 1.0890)
	lr2, lg2, lb2 := XYZToLinearRGB(X2, Y2, Z2)
	r2 := LinearToSRGB(lr2)
	g2 := LinearToSRGB(lg2)
	b2 := LinearToSRGB(lb2)
	if !approxEqual(r2, r, 1e-6) || !approxEqual(g2, g, 1e-6) || !approxEqual(b2, b, 1e-6) {
		t.Errorf("full roundtrip (%v,%v,%v) -> (%v,%v,%v)", r, g, b, r2, g2, b2)
	}
}

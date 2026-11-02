package prob

import (
	"encoding/json"
	"math"
	"math/big"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"testing/quick"
)

// betaQuantileRow is one case of testdata/stress/beta_quantile.json, solved
// with mpmath at 800 digits by tools/stressgolden/beta_quantile.py. Values
// stay decimal strings so the errors are measured against the 40 stored
// digits, not against their float64 rounding.
type betaQuantileRow struct {
	A          string `json:"a"`
	B          string `json:"b"`
	P          string `json:"p"`
	Q          string `json:"q"`
	X          string `json:"x"`
	OneMinusX  string `json:"one_minus_x"`
	Core       bool   `json:"core"`
	a, b, p    float64
	xRef, yRef *big.Float
}

func loadBetaQuantileTable(t *testing.T) []betaQuantileRow {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "stress", "beta_quantile.json"))
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Rows []betaQuantileRow `json:"rows"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	core := 0
	for i := range doc.Rows {
		r := &doc.Rows[i]
		if r.Core {
			core++
		}
		r.a = mustParseFloat(t, r.A)
		r.b = mustParseFloat(t, r.B)
		r.p = mustParseFloat(t, r.P)
		r.xRef = mustParseBig(t, r.X)
		r.yRef = mustParseBig(t, r.OneMinusX)
	}
	if len(doc.Rows) != 101 || core != 42 {
		t.Fatalf("beta_quantile.json: decoded %d rows (%d core), want 101 (42 core)", len(doc.Rows), core)
	}
	return doc.Rows
}

func mustParseFloat(t *testing.T, s string) float64 {
	t.Helper()
	v, _, err := big.ParseFloat(s, 10, 53, big.ToNearestEven)
	if err != nil {
		t.Fatalf("parse %q: %v", s, err)
	}
	f, _ := v.Float64()
	return f
}

func mustParseBig(t *testing.T, s string) *big.Float {
	t.Helper()
	v, _, err := big.ParseFloat(s, 10, 256, big.ToNearestEven)
	if err != nil {
		t.Fatalf("parse %q: %v", s, err)
	}
	return v
}

// relErrBig returns |got - want| / |want| with want held to 256 bits.
func relErrBig(got float64, want *big.Float) float64 {
	if math.IsNaN(got) || math.IsInf(got, 0) {
		return math.Inf(1)
	}
	d := new(big.Float).SetPrec(256).SetFloat64(got)
	d.Sub(d, want)
	d.Quo(d, want)
	e, _ := d.Float64()
	return math.Abs(e)
}

// TestBetaQuantileOracle enforces the bounds of VERIFY.md (fixed before the
// implementation existed) against the 800-digit table:
//
//	core rows:  BetaQuantile vs x                          1e-12 relative
//	other rows: BetaQuantile vs x                          1e-9
//	all rows:   BetaQuantileXY's y vs 1 - x                1e-9
//	all rows:   BetaQuantileComplement(b, a, p) vs 1 - x   1e-9
//	core rows:  RegularizedBetaInc(BetaQuantile(a, b, p), a, b) vs p  1e-10
//
// The swapped-shape check tests a tiny q exactly: 1 - I_x(a, b) = p is
// I_{1-x}(b, a) = p, and p is exact where 1 - p is not.
func TestBetaQuantileOracle(t *testing.T) {
	rows := loadBetaQuantileTable(t)
	var maxCore, maxOther, maxY, maxComp, maxTrip float64
	for _, r := range rows {
		x := BetaQuantile(r.a, r.b, r.p)
		xx, y := BetaQuantileXY(r.a, r.b, r.p)
		c := BetaQuantileComplement(r.b, r.a, r.p)

		ex := relErrBig(x, r.xRef)
		ey := relErrBig(y, r.yRef)
		ec := relErrBig(c, r.yRef)
		if x != xx && !(math.IsNaN(x) && math.IsNaN(xx)) {
			t.Errorf("a=%s b=%s p=%s: BetaQuantileXY x = %.17g, BetaQuantile = %.17g", r.A, r.B, r.P, xx, x)
		}
		bound := 1e-9
		if r.Core {
			bound = 1e-12
			maxCore = math.Max(maxCore, ex)
		} else {
			maxOther = math.Max(maxOther, ex)
		}
		maxY = math.Max(maxY, ey)
		maxComp = math.Max(maxComp, ec)
		if !(ex <= bound) {
			t.Errorf("a=%s b=%s p=%s core=%v: BetaQuantile = %.17g, want %s (rel err %.3g > %.0e)", r.A, r.B, r.P, r.Core, x, r.X, ex, bound)
		}
		if !(ey <= 1e-9) {
			t.Errorf("a=%s b=%s p=%s: BetaQuantileXY y = %.17g, want %s (rel err %.3g > 1e-9)", r.A, r.B, r.P, y, r.OneMinusX, ey)
		}
		if !(ec <= 1e-9) {
			t.Errorf("a=%s b=%s p=%s: BetaQuantileComplement(b, a, p) = %.17g, want %s (rel err %.3g > 1e-9)", r.A, r.B, r.P, c, r.OneMinusX, ec)
		}
		if r.Core {
			pp := RegularizedBetaInc(x, r.a, r.b)
			et := math.Abs(pp-r.p) / r.p
			maxTrip = math.Max(maxTrip, et)
			if !(et <= 1e-10) {
				t.Errorf("a=%s b=%s p=%s: RegularizedBetaInc(BetaQuantile) = %.17g (rel err %.3g > 1e-10)", r.A, r.B, r.P, pp, et)
			}
		}
	}
	t.Logf("max relative error: core x %.3g, other x %.3g, y %.3g, complement %.3g, core round trip %.3g",
		maxCore, maxOther, maxY, maxComp, maxTrip)
}

// TestBetaQuantileInvalid checks the documented NaN cases, which follow
// NormalQuantile and ExponentialQuantile (p outside the open interval) and
// RegularizedBetaInc (shapes not finite and positive).
func TestBetaQuantileInvalid(t *testing.T) {
	nan, inf := math.NaN(), math.Inf(1)
	cases := [][3]float64{
		{2, 3, 0}, {2, 3, 1}, {2, 3, -0.1}, {2, 3, 1.5}, {2, 3, nan}, {2, 3, inf},
		{0, 3, 0.5}, {-1, 3, 0.5}, {2, 0, 0.5}, {nan, 3, 0.5}, {2, nan, 0.5}, {inf, 3, 0.5}, {2, inf, 0.5},
	}
	for _, c := range cases {
		if v := BetaQuantile(c[0], c[1], c[2]); !math.IsNaN(v) {
			t.Errorf("BetaQuantile(%g, %g, %g) = %g, want NaN", c[0], c[1], c[2], v)
		}
		if v := BetaQuantileComplement(c[0], c[1], c[2]); !math.IsNaN(v) {
			t.Errorf("BetaQuantileComplement(%g, %g, %g) = %g, want NaN", c[0], c[1], c[2], v)
		}
		if x, y := BetaQuantileXY(c[0], c[1], c[2]); !math.IsNaN(x) || !math.IsNaN(y) {
			t.Errorf("BetaQuantileXY(%g, %g, %g) = (%g, %g), want NaN, NaN", c[0], c[1], c[2], x, y)
		}
	}
}

// TestBetaQuantileClosedForms checks shapes with an exact inverse:
// I_x(1, 1) = x, I_x(a, 1) = x^a and 1 - I_x(1, b) = (1-x)^b. The bound is
// the error the forward function itself carries, which the root inherits:
// a few ulps plus the rounding of its exponent, about 1.1e-16*|ln p| (the
// relative error of exp(a ln x)), divided by the shape the root is raised
// to.
func TestBetaQuantileClosedForms(t *testing.T) {
	for _, p := range []float64{1e-300, 1e-20, 1e-5, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99999} {
		tol := 1e-15 + 1.2e-16*math.Abs(math.Log(p))
		if got := BetaQuantile(1, 1, p); math.Abs(got-p) > tol*p {
			t.Errorf("BetaQuantile(1, 1, %g) = %.17g", p, got)
		}
		want := math.Pow(p, 1/2.5)
		if got := BetaQuantile(2.5, 1, p); math.Abs(got-want) > (tol/2.5+4e-16)*want {
			t.Errorf("BetaQuantile(2.5, 1, %g) = %.17g, want %.17g", p, got, want)
		}
		// 1 - I_x(1, 3) = q  <=>  1 - x = q^(1/3).
		wantY := math.Cbrt(p)
		if _, y := betaQuantile(1, 3, p, true); math.Abs(y-wantY) > (tol/3+4e-16)*wantY {
			t.Errorf("BetaQuantileComplement(1, 3, %g): 1 - x = %.17g, want %.17g", p, y, wantY)
		}
	}
}

// TestBetaQuantileRoundTripProperty checks, for seeded random shapes in
// [0.05, 500] and p in [0.001, 0.999], that the quantile is monotone in p,
// that x + y = 1, and that the incomplete beta function maps (x, y) back to
// p, or 1 - p to its complement, within 1e-10 relative.
func TestBetaQuantileRoundTripProperty(t *testing.T) {
	prop := func(ua, ub, up uint32) bool {
		a := 0.05 * math.Pow(1e4, float64(ua)/math.MaxUint32)
		b := 0.05 * math.Pow(1e4, float64(ub)/math.MaxUint32)
		p := 0.001 + 0.998*float64(up)/math.MaxUint32
		x, y := BetaQuantileXY(a, b, p)
		if !(x >= 0 && x <= 1 && math.Abs(x+y-1) <= 1.2e-16) {
			t.Logf("a=%g b=%g p=%g: x=%g y=%g", a, b, p, x, y)
			return false
		}
		if x2 := BetaQuantile(a, b, p*1.0001); p*1.0001 < 1 && x2 < x {
			t.Logf("a=%g b=%g p=%g: not monotone, %g then %g", a, b, p, x, x2)
			return false
		}
		// The round trip goes through incBeta with y, since near x = 1 the
		// rounding of x alone would move I_x by far more than its error.
		w, w1 := incBeta(a, b, x, y)
		e := math.Abs(w-p) / p
		if p > 0.5 {
			e = math.Abs(w1-(1-p)) / (1 - p)
		}
		if e > 1e-10 {
			t.Logf("a=%g b=%g p=%g: round trip rel err %g", a, b, p, e)
			return false
		}
		return true
	}
	if err := quick.Check(prop, &quick.Config{Rand: rand.New(rand.NewSource(1)), MaxCount: 5000}); err != nil {
		t.Fatal(err)
	}
}

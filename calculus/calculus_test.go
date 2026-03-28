package calculus

import (
	"math"
	"math/rand"
	"testing"

	"github.com/davly/reality/testutil"
)

// ---------------------------------------------------------------------------
// Test function registry — maps string names to Go functions, shared across
// golden-file tests.
// ---------------------------------------------------------------------------

var scalarFuncs = map[string]func(float64) float64{
	"x":    func(x float64) float64 { return x },
	"x^2":  func(x float64) float64 { return x * x },
	"x^3":  func(x float64) float64 { return x * x * x },
	"x^4":  func(x float64) float64 { return x * x * x * x },
	"sin":  math.Sin,
	"cos":  math.Cos,
	"exp":  math.Exp,
	"ln":   math.Log,
	"1/x":  func(x float64) float64 { return 1.0 / x },
	"5":    func(float64) float64 { return 5 },
	"7":    func(float64) float64 { return 7 },
}

// ---------------------------------------------------------------------------
// NumericalDerivative — unit tests
// ---------------------------------------------------------------------------

func TestDerivative_XSquared(t *testing.T) {
	f := func(x float64) float64 { return x * x }
	got := NumericalDerivative(f, 3, 1e-5)
	if math.Abs(got-6) > 1e-4 {
		t.Errorf("d/dx(x^2) at 3 = %v, want 6", got)
	}
}

func TestDerivative_Sin_AtZero(t *testing.T) {
	got := NumericalDerivative(math.Sin, 0, 1e-5)
	if math.Abs(got-1) > 1e-5 {
		t.Errorf("d/dx(sin) at 0 = %v, want 1", got)
	}
}

func TestDerivative_Sin_AtPiHalf(t *testing.T) {
	got := NumericalDerivative(math.Sin, math.Pi/2, 1e-5)
	if math.Abs(got) > 1e-4 {
		t.Errorf("d/dx(sin) at pi/2 = %v, want 0", got)
	}
}

func TestDerivative_Exp_AtZero(t *testing.T) {
	got := NumericalDerivative(math.Exp, 0, 1e-5)
	if math.Abs(got-1) > 1e-5 {
		t.Errorf("d/dx(exp) at 0 = %v, want 1", got)
	}
}

func TestDerivative_Exp_AtOne(t *testing.T) {
	got := NumericalDerivative(math.Exp, 1, 1e-5)
	if math.Abs(got-math.E) > 1e-4 {
		t.Errorf("d/dx(exp) at 1 = %v, want %v", got, math.E)
	}
}

func TestDerivative_Cos_AtZero(t *testing.T) {
	got := NumericalDerivative(math.Cos, 0, 1e-5)
	if math.Abs(got) > 1e-5 {
		t.Errorf("d/dx(cos) at 0 = %v, want 0", got)
	}
}

func TestDerivative_Log_AtOne(t *testing.T) {
	got := NumericalDerivative(math.Log, 1, 1e-5)
	if math.Abs(got-1) > 1e-4 {
		t.Errorf("d/dx(ln) at 1 = %v, want 1", got)
	}
}

func TestDerivative_Cubic_AtTwo(t *testing.T) {
	f := func(x float64) float64 { return x * x * x }
	got := NumericalDerivative(f, 2, 1e-5)
	if math.Abs(got-12) > 1e-4 {
		t.Errorf("d/dx(x^3) at 2 = %v, want 12", got)
	}
}

// Golden-file test for NumericalDerivative.
func TestDerivative_Golden(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/calculus/derivative.json")

	if gf.Function != "Calculus.NumericalDerivative" {
		t.Fatalf("golden file function = %q, want Calculus.NumericalDerivative", gf.Function)
	}

	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			fnName, ok := tc.Inputs["function"].(string)
			if !ok {
				t.Fatalf("missing or invalid function input")
			}
			fn, ok := scalarFuncs[fnName]
			if !ok {
				t.Fatalf("unknown test function: %s", fnName)
			}
			x := testutil.InputFloat64(t, tc, "x")
			h := testutil.InputFloat64(t, tc, "h")
			got := NumericalDerivative(fn, x, h)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

// ---------------------------------------------------------------------------
// NumericalGradient — unit tests
// ---------------------------------------------------------------------------

func TestGradient_2D_Quadratic(t *testing.T) {
	// f(x,y) = x^2 + y^2, gradient at (3, 4) = (6, 8)
	f := func(x []float64) float64 { return x[0]*x[0] + x[1]*x[1] }
	x := []float64{3, 4}
	out := make([]float64, 2)
	NumericalGradient(f, x, 1e-5, out)
	if math.Abs(out[0]-6) > 1e-4 || math.Abs(out[1]-8) > 1e-4 {
		t.Errorf("gradient of x^2+y^2 at (3,4) = %v, want [6, 8]", out)
	}
}

func TestGradient_3D_Linear(t *testing.T) {
	// f(x,y,z) = 2x + 3y + 5z, gradient everywhere = (2, 3, 5)
	f := func(x []float64) float64 { return 2*x[0] + 3*x[1] + 5*x[2] }
	x := []float64{10, -5, 7}
	out := make([]float64, 3)
	NumericalGradient(f, x, 1e-5, out)
	if math.Abs(out[0]-2) > 1e-5 || math.Abs(out[1]-3) > 1e-5 || math.Abs(out[2]-5) > 1e-5 {
		t.Errorf("gradient of 2x+3y+5z = %v, want [2, 3, 5]", out)
	}
}

func TestGradient_1D(t *testing.T) {
	f := func(x []float64) float64 { return math.Sin(x[0]) }
	x := []float64{0}
	out := make([]float64, 1)
	NumericalGradient(f, x, 1e-5, out)
	if math.Abs(out[0]-1) > 1e-5 {
		t.Errorf("gradient of sin(x) at 0 = %v, want [1]", out)
	}
}

func TestGradient_RestoresX(t *testing.T) {
	// Ensure x is not modified after NumericalGradient.
	f := func(x []float64) float64 { return x[0] + x[1] }
	x := []float64{3.14, 2.71}
	out := make([]float64, 2)
	NumericalGradient(f, x, 1e-5, out)
	if x[0] != 3.14 || x[1] != 2.71 {
		t.Errorf("NumericalGradient modified x: %v", x)
	}
}

func TestGradient_Rosenbrock(t *testing.T) {
	// f(x,y) = (1-x)^2 + 100*(y-x^2)^2
	// df/dx = -2(1-x) - 400x(y-x^2)
	// df/dy = 200(y-x^2)
	f := func(x []float64) float64 {
		return (1-x[0])*(1-x[0]) + 100*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0])
	}
	x := []float64{1, 1} // at the minimum, gradient should be (0, 0)
	out := make([]float64, 2)
	NumericalGradient(f, x, 1e-5, out)
	if math.Abs(out[0]) > 1e-4 || math.Abs(out[1]) > 1e-4 {
		t.Errorf("gradient of Rosenbrock at (1,1) = %v, want [0, 0]", out)
	}
}

// ---------------------------------------------------------------------------
// TrapezoidalRule — unit tests
// ---------------------------------------------------------------------------

func TestTrapezoidal_Linear(t *testing.T) {
	// integral of x from 0 to 1 = 0.5
	f := func(x float64) float64 { return x }
	got := TrapezoidalRule(f, 0, 1, 100)
	if math.Abs(got-0.5) > 1e-10 {
		t.Errorf("trapezoidal(x, 0, 1) = %v, want 0.5", got)
	}
}

func TestTrapezoidal_Quadratic(t *testing.T) {
	// integral of x^2 from 0 to 1 = 1/3
	f := func(x float64) float64 { return x * x }
	got := TrapezoidalRule(f, 0, 1, 1000)
	if math.Abs(got-1.0/3.0) > 1e-5 {
		t.Errorf("trapezoidal(x^2, 0, 1) = %v, want 1/3", got)
	}
}

func TestTrapezoidal_Sin(t *testing.T) {
	got := TrapezoidalRule(math.Sin, 0, math.Pi, 1000)
	if math.Abs(got-2) > 1e-5 {
		t.Errorf("trapezoidal(sin, 0, pi) = %v, want 2", got)
	}
}

func TestTrapezoidal_Exp(t *testing.T) {
	got := TrapezoidalRule(math.Exp, 0, 1, 1000)
	want := math.E - 1
	if math.Abs(got-want) > 1e-5 {
		t.Errorf("trapezoidal(exp, 0, 1) = %v, want %v", got, want)
	}
}

func TestTrapezoidal_Constant(t *testing.T) {
	f := func(float64) float64 { return 5 }
	got := TrapezoidalRule(f, 0, 3, 10)
	if math.Abs(got-15) > 1e-12 {
		t.Errorf("trapezoidal(5, 0, 3) = %v, want 15", got)
	}
}

func TestTrapezoidal_NEquals1(t *testing.T) {
	// With n=1, trapezoidal rule = h/2*(f(a)+f(b))
	f := func(x float64) float64 { return x }
	got := TrapezoidalRule(f, 0, 2, 1)
	// h=2, result = 2/2*(0+2) = 2
	if math.Abs(got-2) > 1e-15 {
		t.Errorf("trapezoidal(x, 0, 2, n=1) = %v, want 2", got)
	}
}

// Golden-file test for TrapezoidalRule.
func TestTrapezoidal_Golden(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/calculus/trapezoidal.json")

	if gf.Function != "Calculus.TrapezoidalRule" {
		t.Fatalf("golden file function = %q, want Calculus.TrapezoidalRule", gf.Function)
	}

	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			fnName, ok := tc.Inputs["function"].(string)
			if !ok {
				t.Fatalf("missing or invalid function input")
			}
			fn, ok := scalarFuncs[fnName]
			if !ok {
				t.Fatalf("unknown test function: %s", fnName)
			}
			a := testutil.InputFloat64(t, tc, "a")
			b := testutil.InputFloat64(t, tc, "b")
			n := testutil.InputInt(t, tc, "n")
			got := TrapezoidalRule(fn, a, b, n)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

// ---------------------------------------------------------------------------
// SimpsonsRule — unit tests
// ---------------------------------------------------------------------------

func TestSimpsons_Linear(t *testing.T) {
	f := func(x float64) float64 { return x }
	got := SimpsonsRule(f, 0, 1, 100)
	if math.Abs(got-0.5) > 1e-12 {
		t.Errorf("simpsons(x, 0, 1) = %v, want 0.5", got)
	}
}

func TestSimpsons_Quadratic(t *testing.T) {
	f := func(x float64) float64 { return x * x }
	got := SimpsonsRule(f, 0, 1, 100)
	if math.Abs(got-1.0/3.0) > 1e-12 {
		t.Errorf("simpsons(x^2, 0, 1) = %v, want 1/3", got)
	}
}

func TestSimpsons_Cubic(t *testing.T) {
	f := func(x float64) float64 { return x * x * x }
	got := SimpsonsRule(f, 0, 1, 100)
	if math.Abs(got-0.25) > 1e-10 {
		t.Errorf("simpsons(x^3, 0, 1) = %v, want 0.25", got)
	}
}

func TestSimpsons_Sin(t *testing.T) {
	got := SimpsonsRule(math.Sin, 0, math.Pi, 100)
	if math.Abs(got-2) > 1e-7 {
		t.Errorf("simpsons(sin, 0, pi) = %v, want 2", got)
	}
}

func TestSimpsons_Exp(t *testing.T) {
	got := SimpsonsRule(math.Exp, 0, 1, 100)
	want := math.E - 1
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("simpsons(exp, 0, 1) = %v, want %v", got, want)
	}
}

func TestSimpsons_OddN(t *testing.T) {
	// Odd n should be bumped to even.
	f := func(x float64) float64 { return x * x }
	got := SimpsonsRule(f, 0, 1, 99) // becomes 100
	if math.Abs(got-1.0/3.0) > 1e-12 {
		t.Errorf("simpsons(x^2, 0, 1, n=99) = %v, want 1/3", got)
	}
}

func TestSimpsons_NEquals2(t *testing.T) {
	// Minimum viable n. For x^2 on [0,1], Simpson's is exact.
	f := func(x float64) float64 { return x * x }
	got := SimpsonsRule(f, 0, 1, 2)
	if math.Abs(got-1.0/3.0) > 1e-14 {
		t.Errorf("simpsons(x^2, 0, 1, n=2) = %v, want 1/3", got)
	}
}

// Golden-file test for SimpsonsRule.
func TestSimpsons_Golden(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/calculus/simpsons.json")

	if gf.Function != "Calculus.SimpsonsRule" {
		t.Fatalf("golden file function = %q, want Calculus.SimpsonsRule", gf.Function)
	}

	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			fnName, ok := tc.Inputs["function"].(string)
			if !ok {
				t.Fatalf("missing or invalid function input")
			}
			fn, ok := scalarFuncs[fnName]
			if !ok {
				t.Fatalf("unknown test function: %s", fnName)
			}
			a := testutil.InputFloat64(t, tc, "a")
			b := testutil.InputFloat64(t, tc, "b")
			n := testutil.InputInt(t, tc, "n")
			got := SimpsonsRule(fn, a, b, n)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

// ---------------------------------------------------------------------------
// GaussLegendre — unit tests
// ---------------------------------------------------------------------------

func TestGL_Linear_2pt(t *testing.T) {
	// 2-point GL is exact for degree <= 3 polynomials. x is degree 1.
	f := func(x float64) float64 { return x }
	got := GaussLegendre(f, 0, 1, 2)
	if math.Abs(got-0.5) > 1e-14 {
		t.Errorf("GL-2(x, 0, 1) = %v, want 0.5", got)
	}
}

func TestGL_Quadratic_2pt(t *testing.T) {
	f := func(x float64) float64 { return x * x }
	got := GaussLegendre(f, 0, 1, 2)
	if math.Abs(got-1.0/3.0) > 1e-14 {
		t.Errorf("GL-2(x^2, 0, 1) = %v, want 1/3", got)
	}
}

func TestGL_Cubic_2pt(t *testing.T) {
	// 2-point GL is exact for degree 3.
	f := func(x float64) float64 { return x * x * x }
	got := GaussLegendre(f, 0, 1, 2)
	if math.Abs(got-0.25) > 1e-14 {
		t.Errorf("GL-2(x^3, 0, 1) = %v, want 0.25", got)
	}
}

func TestGL_Quartic_3pt(t *testing.T) {
	// 3-point GL is exact for degree <= 5. x^4 is degree 4.
	f := func(x float64) float64 { return x * x * x * x }
	got := GaussLegendre(f, 0, 1, 3)
	if math.Abs(got-0.2) > 1e-14 {
		t.Errorf("GL-3(x^4, 0, 1) = %v, want 0.2", got)
	}
}

func TestGL_Sin_5pt(t *testing.T) {
	got := GaussLegendre(math.Sin, 0, math.Pi, 5)
	if math.Abs(got-2) > 1e-6 {
		t.Errorf("GL-5(sin, 0, pi) = %v, want 2", got)
	}
}

func TestGL_Exp_5pt(t *testing.T) {
	got := GaussLegendre(math.Exp, 0, 1, 5)
	want := math.E - 1
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("GL-5(exp, 0, 1) = %v, want %v", got, want)
	}
}

func TestGL_Constant(t *testing.T) {
	f := func(float64) float64 { return 7 }
	got := GaussLegendre(f, 2, 5, 3)
	if math.Abs(got-21) > 1e-12 {
		t.Errorf("GL-3(7, 2, 5) = %v, want 21", got)
	}
}

func TestGL_OddFunction_Symmetric(t *testing.T) {
	// x^3 on [-1, 1] should be 0 by symmetry.
	f := func(x float64) float64 { return x * x * x }
	got := GaussLegendre(f, -1, 1, 4)
	if math.Abs(got) > 1e-14 {
		t.Errorf("GL-4(x^3, -1, 1) = %v, want 0", got)
	}
}

// Golden-file test for GaussLegendre.
func TestGL_Golden(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/calculus/gauss_legendre.json")

	if gf.Function != "Calculus.GaussLegendre" {
		t.Fatalf("golden file function = %q, want Calculus.GaussLegendre", gf.Function)
	}

	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			fnName, ok := tc.Inputs["function"].(string)
			if !ok {
				t.Fatalf("missing or invalid function input")
			}
			fn, ok := scalarFuncs[fnName]
			if !ok {
				t.Fatalf("unknown test function: %s", fnName)
			}
			a := testutil.InputFloat64(t, tc, "a")
			b := testutil.InputFloat64(t, tc, "b")
			pts := testutil.InputInt(t, tc, "points")
			got := GaussLegendre(fn, a, b, pts)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

// ---------------------------------------------------------------------------
// MonteCarloIntegrate — unit tests
// ---------------------------------------------------------------------------

func TestMC_1D_Constant(t *testing.T) {
	// integral of 5 from 0 to 3 = 15
	f := func(x []float64) float64 { return 5 }
	rng := rand.New(rand.NewSource(42))
	got := MonteCarloIntegrate(f, 1, []float64{0}, []float64{3}, 100000, rng)
	if math.Abs(got-15) > 0.1 {
		t.Errorf("MC(5, [0,3]) = %v, want ~15", got)
	}
}

func TestMC_1D_Linear(t *testing.T) {
	// integral of x from 0 to 1 = 0.5
	f := func(x []float64) float64 { return x[0] }
	rng := rand.New(rand.NewSource(42))
	got := MonteCarloIntegrate(f, 1, []float64{0}, []float64{1}, 100000, rng)
	if math.Abs(got-0.5) > 0.01 {
		t.Errorf("MC(x, [0,1]) = %v, want ~0.5", got)
	}
}

func TestMC_1D_Quadratic(t *testing.T) {
	// integral of x^2 from 0 to 1 = 1/3
	f := func(x []float64) float64 { return x[0] * x[0] }
	rng := rand.New(rand.NewSource(42))
	got := MonteCarloIntegrate(f, 1, []float64{0}, []float64{1}, 100000, rng)
	if math.Abs(got-1.0/3.0) > 0.01 {
		t.Errorf("MC(x^2, [0,1]) = %v, want ~1/3", got)
	}
}

func TestMC_2D_Area(t *testing.T) {
	// integral of 1 over [0,2]x[0,3] = 6 (area of rectangle)
	f := func(x []float64) float64 { return 1 }
	rng := rand.New(rand.NewSource(42))
	got := MonteCarloIntegrate(f, 2, []float64{0, 0}, []float64{2, 3}, 10000, rng)
	if math.Abs(got-6) > 0.1 {
		t.Errorf("MC(1, [0,2]x[0,3]) = %v, want ~6", got)
	}
}

func TestMC_2D_Product(t *testing.T) {
	// integral of x*y over [0,1]x[0,1] = 0.25
	f := func(x []float64) float64 { return x[0] * x[1] }
	rng := rand.New(rand.NewSource(42))
	got := MonteCarloIntegrate(f, 2, []float64{0, 0}, []float64{1, 1}, 100000, rng)
	if math.Abs(got-0.25) > 0.01 {
		t.Errorf("MC(x*y, [0,1]^2) = %v, want ~0.25", got)
	}
}

func TestMC_Deterministic(t *testing.T) {
	// Same seed should give same result.
	f := func(x []float64) float64 { return x[0] * x[0] }
	r1 := rand.New(rand.NewSource(99))
	r2 := rand.New(rand.NewSource(99))
	got1 := MonteCarloIntegrate(f, 1, []float64{0}, []float64{1}, 1000, r1)
	got2 := MonteCarloIntegrate(f, 1, []float64{0}, []float64{1}, 1000, r2)
	if got1 != got2 {
		t.Errorf("MC not deterministic: %v != %v", got1, got2)
	}
}

func TestMC_3D_Volume(t *testing.T) {
	// integral of 1 over [0,1]^3 = 1
	f := func(x []float64) float64 { return 1 }
	rng := rand.New(rand.NewSource(42))
	got := MonteCarloIntegrate(f, 3, []float64{0, 0, 0}, []float64{1, 1, 1}, 10000, rng)
	if math.Abs(got-1) > 0.05 {
		t.Errorf("MC(1, [0,1]^3) = %v, want ~1", got)
	}
}

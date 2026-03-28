package optim

import (
	"math"
	"math/rand"
	"testing"

	"github.com/davly/reality/testutil"
)

// ---------------------------------------------------------------------------
// BisectionMethod
// ---------------------------------------------------------------------------

func TestBisection_Sqrt2(t *testing.T) {
	f := func(x float64) float64 { return x*x - 2 }
	got := BisectionMethod(f, 1, 2, 1e-12)
	if math.Abs(got-math.Sqrt(2)) > 1e-10 {
		t.Errorf("BisectionMethod sqrt(2) = %v, want %v", got, math.Sqrt(2))
	}
}

func TestBisection_SinPi(t *testing.T) {
	got := BisectionMethod(math.Sin, 3, 4, 1e-12)
	if math.Abs(got-math.Pi) > 1e-10 {
		t.Errorf("BisectionMethod sin(x)=0 pi = %v, want %v", got, math.Pi)
	}
}

func TestBisection_CubeRoot8(t *testing.T) {
	f := func(x float64) float64 { return x*x*x - 8 }
	got := BisectionMethod(f, 1, 3, 1e-12)
	if math.Abs(got-2.0) > 1e-10 {
		t.Errorf("BisectionMethod cbrt(8) = %v, want 2", got)
	}
}

func TestBisection_CosPiHalf(t *testing.T) {
	got := BisectionMethod(math.Cos, 1, 2, 1e-12)
	if math.Abs(got-math.Pi/2) > 1e-10 {
		t.Errorf("BisectionMethod cos(x)=0 pi/2 = %v, want %v", got, math.Pi/2)
	}
}

func TestBisection_ExactRoot(t *testing.T) {
	// f(x) = x - 1, root at exactly 1.0
	f := func(x float64) float64 { return x - 1 }
	got := BisectionMethod(f, 0, 2, 1e-15)
	if math.Abs(got-1.0) > 1e-12 {
		t.Errorf("BisectionMethod x-1=0 = %v, want 1", got)
	}
}

func TestBisection_NegativeInterval(t *testing.T) {
	// f(x) = x^2 - 4, root at -2 on [-3, -1]
	f := func(x float64) float64 { return x*x - 4 }
	got := BisectionMethod(f, -3, -1, 1e-12)
	if math.Abs(got-(-2.0)) > 1e-10 {
		t.Errorf("BisectionMethod x^2-4=0 neg = %v, want -2", got)
	}
}

func TestBisection_Ln3(t *testing.T) {
	// exp(x) - 3 = 0, root at ln(3)
	f := func(x float64) float64 { return math.Exp(x) - 3 }
	got := BisectionMethod(f, 0, 2, 1e-12)
	if math.Abs(got-math.Log(3)) > 1e-10 {
		t.Errorf("BisectionMethod exp(x)-3=0 = %v, want %v", got, math.Log(3))
	}
}

// Golden-file tests for bisection.
func TestBisection_Golden(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/optim/bisection.json")

	if gf.Function != "Optim.BisectionMethod" {
		t.Fatalf("golden file function = %q, want Optim.BisectionMethod", gf.Function)
	}

	// Map function descriptions to actual Go functions.
	funcMap := map[string]func(float64) float64{
		"x^2-2":   func(x float64) float64 { return x*x - 2 },
		"x^2-3":   func(x float64) float64 { return x*x - 3 },
		"sin(x)":  math.Sin,
		"cos(x)":  math.Cos,
		"x^3-x-2": func(x float64) float64 { return x*x*x - x - 2 },
		"exp(x)-3": func(x float64) float64 { return math.Exp(x) - 3 },
		"x-1":     func(x float64) float64 { return x - 1 },
		"x^5-x-1": func(x float64) float64 { return x*x*x*x*x - x - 1 },
	}

	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			a := testutil.InputFloat64(t, tc, "a")
			b := testutil.InputFloat64(t, tc, "b")
			fnName, ok := tc.Inputs["function"].(string)
			if !ok {
				t.Fatalf("missing or invalid function input")
			}
			fn, ok := funcMap[fnName]
			if !ok {
				t.Fatalf("unknown test function: %s", fnName)
			}
			got := BisectionMethod(fn, a, b, 1e-12)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

// ---------------------------------------------------------------------------
// NewtonRaphson
// ---------------------------------------------------------------------------

func TestNewtonRaphson_Sqrt2(t *testing.T) {
	f := func(x float64) float64 { return x*x - 2 }
	fp := func(x float64) float64 { return 2 * x }
	got := NewtonRaphson(f, fp, 1.5, 1e-12, 100)
	if math.Abs(got-math.Sqrt(2)) > 1e-10 {
		t.Errorf("NewtonRaphson sqrt(2) = %v, want %v", got, math.Sqrt(2))
	}
}

func TestNewtonRaphson_CubeRoot27(t *testing.T) {
	f := func(x float64) float64 { return x*x*x - 27 }
	fp := func(x float64) float64 { return 3 * x * x }
	got := NewtonRaphson(f, fp, 3.5, 1e-12, 100)
	if math.Abs(got-3.0) > 1e-10 {
		t.Errorf("NewtonRaphson cbrt(27) = %v, want 3", got)
	}
}

func TestNewtonRaphson_Convergence(t *testing.T) {
	// Newton's method should converge quadratically. For x^2-2, starting
	// from 2.0, we should reach machine precision in < 10 iterations.
	f := func(x float64) float64 { return x*x - 2 }
	fp := func(x float64) float64 { return 2 * x }
	got := NewtonRaphson(f, fp, 2.0, 1e-15, 10)
	if math.Abs(got-math.Sqrt(2)) > 1e-14 {
		t.Errorf("NewtonRaphson convergence: got %v, diff %e", got, math.Abs(got-math.Sqrt(2)))
	}
}

func TestNewtonRaphson_Exp(t *testing.T) {
	// exp(x) - 5 = 0 => x = ln(5)
	f := func(x float64) float64 { return math.Exp(x) - 5 }
	fp := func(x float64) float64 { return math.Exp(x) }
	got := NewtonRaphson(f, fp, 1.0, 1e-12, 100)
	if math.Abs(got-math.Log(5)) > 1e-10 {
		t.Errorf("NewtonRaphson ln(5) = %v, want %v", got, math.Log(5))
	}
}

func TestNewtonRaphson_ZeroDerivative(t *testing.T) {
	// If derivative is zero at x0, should return x0 gracefully.
	f := func(x float64) float64 { return x * x }
	fp := func(x float64) float64 { return 0 } // always zero derivative
	got := NewtonRaphson(f, fp, 5.0, 1e-12, 100)
	if got != 5.0 {
		t.Errorf("NewtonRaphson zero derivative: got %v, want 5.0", got)
	}
}

// ---------------------------------------------------------------------------
// GoldenSectionSearch
// ---------------------------------------------------------------------------

func TestGoldenSection_XSquared(t *testing.T) {
	f := func(x float64) float64 { return x * x }
	got := GoldenSectionSearch(f, -2, 3, 1e-10)
	if math.Abs(got) > 1e-8 {
		t.Errorf("GoldenSection x^2 minimum = %v, want ~0", got)
	}
}

func TestGoldenSection_ShiftedParabola(t *testing.T) {
	// (x-3)^2, minimum at x=3
	f := func(x float64) float64 { return (x - 3) * (x - 3) }
	got := GoldenSectionSearch(f, 0, 5, 1e-10)
	if math.Abs(got-3.0) > 1e-8 {
		t.Errorf("GoldenSection (x-3)^2 minimum = %v, want 3", got)
	}
}

func TestGoldenSection_Cos(t *testing.T) {
	// cos(x) on [2, 5] has minimum at pi ≈ 3.14159
	f := func(x float64) float64 { return math.Cos(x) }
	got := GoldenSectionSearch(f, 2, 5, 1e-10)
	if math.Abs(got-math.Pi) > 1e-7 {
		t.Errorf("GoldenSection cos minimum = %v, want pi=%v", got, math.Pi)
	}
}

func TestGoldenSection_Quartic(t *testing.T) {
	// (x-1)^4, minimum at x=1
	f := func(x float64) float64 {
		d := x - 1
		return d * d * d * d
	}
	got := GoldenSectionSearch(f, -2, 4, 1e-8)
	if math.Abs(got-1.0) > 1e-6 {
		t.Errorf("GoldenSection (x-1)^4 minimum = %v, want 1", got)
	}
}

func TestGoldenSection_NegativeInterval(t *testing.T) {
	// (x+2)^2, minimum at x=-2, search [-5, 0]
	f := func(x float64) float64 { return (x + 2) * (x + 2) }
	got := GoldenSectionSearch(f, -5, 0, 1e-10)
	if math.Abs(got+2.0) > 1e-8 {
		t.Errorf("GoldenSection (x+2)^2 minimum = %v, want -2", got)
	}
}

// ---------------------------------------------------------------------------
// LinearInterpolateRoot
// ---------------------------------------------------------------------------

func TestLinearInterpolateRoot_Basic(t *testing.T) {
	// Line through (0, -1) and (2, 1): root at x=1
	got := LinearInterpolateRoot(0, -1, 2, 1)
	if math.Abs(got-1.0) > 1e-15 {
		t.Errorf("LinearInterpolateRoot = %v, want 1", got)
	}
}

func TestLinearInterpolateRoot_Horizontal(t *testing.T) {
	// Horizontal line: y0 == y1, should return NaN
	got := LinearInterpolateRoot(0, 3, 5, 3)
	if !math.IsNaN(got) {
		t.Errorf("LinearInterpolateRoot horizontal = %v, want NaN", got)
	}
}

func TestLinearInterpolateRoot_SteepLine(t *testing.T) {
	// Line through (1, -10) and (1.1, 10): root near 1.05
	got := LinearInterpolateRoot(1, -10, 1.1, 10)
	if math.Abs(got-1.05) > 1e-14 {
		t.Errorf("LinearInterpolateRoot steep = %v, want 1.05", got)
	}
}

func TestLinearInterpolateRoot_NegativeX(t *testing.T) {
	// Line through (-3, 6) and (3, -6): root at 0
	got := LinearInterpolateRoot(-3, 6, 3, -6)
	if math.Abs(got) > 1e-15 {
		t.Errorf("LinearInterpolateRoot negative = %v, want 0", got)
	}
}

// ---------------------------------------------------------------------------
// GradientDescent
// ---------------------------------------------------------------------------

func TestGradientDescent_1D_XSquared(t *testing.T) {
	// Minimize x^2, minimum at 0.
	f := func(x []float64) float64 { return x[0] * x[0] }
	grad := func(x, g []float64) { g[0] = 2 * x[0] }
	got := GradientDescent(f, grad, []float64{5.0}, 0.1, 10000, 1e-10)
	if math.Abs(got[0]) > 1e-6 {
		t.Errorf("GD x^2 minimum = %v, want ~0", got[0])
	}
}

func TestGradientDescent_2D_Quadratic(t *testing.T) {
	// Minimize x^2 + y^2, minimum at (0, 0).
	f := func(x []float64) float64 { return x[0]*x[0] + x[1]*x[1] }
	grad := func(x, g []float64) { g[0] = 2 * x[0]; g[1] = 2 * x[1] }
	got := GradientDescent(f, grad, []float64{3.0, -4.0}, 0.1, 10000, 1e-10)
	dist := math.Sqrt(got[0]*got[0] + got[1]*got[1])
	if dist > 1e-6 {
		t.Errorf("GD 2D quadratic = %v, want ~[0,0], dist=%e", got, dist)
	}
}

func TestGradientDescent_Rosenbrock(t *testing.T) {
	// Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1,1).
	// GD with constant step size is slow on Rosenbrock, so use generous
	// iteration count and larger tolerance.
	f := func(x []float64) float64 {
		return (1-x[0])*(1-x[0]) + 100*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0])
	}
	grad := func(x, g []float64) {
		g[0] = -2*(1-x[0]) + 200*(x[1]-x[0]*x[0])*(-2*x[0])
		g[1] = 200 * (x[1] - x[0]*x[0])
	}
	got := GradientDescent(f, grad, []float64{-1.0, 1.0}, 0.001, 100000, 1e-6)
	dist := math.Sqrt((got[0]-1)*(got[0]-1) + (got[1]-1)*(got[1]-1))
	if dist > 0.1 {
		t.Errorf("GD Rosenbrock = %v, want ~[1,1], dist=%e", got, dist)
	}
}

func TestGradientDescent_DoesNotModifyX0(t *testing.T) {
	f := func(x []float64) float64 { return x[0] * x[0] }
	grad := func(x, g []float64) { g[0] = 2 * x[0] }
	x0 := []float64{5.0}
	GradientDescent(f, grad, x0, 0.1, 1000, 1e-10)
	if x0[0] != 5.0 {
		t.Errorf("GD modified x0: got %v, want 5.0", x0[0])
	}
}

// ---------------------------------------------------------------------------
// L-BFGS
// ---------------------------------------------------------------------------

func TestLBFGS_Quadratic(t *testing.T) {
	// Minimize x^2 + 2*y^2, minimum at (0,0).
	f := func(x []float64) float64 { return x[0]*x[0] + 2*x[1]*x[1] }
	grad := func(x, g []float64) { g[0] = 2 * x[0]; g[1] = 4 * x[1] }
	got := LBFGS(f, grad, []float64{5.0, -3.0}, 5, 200, 1e-10)
	dist := math.Sqrt(got[0]*got[0] + got[1]*got[1])
	if dist > 1e-6 {
		t.Errorf("LBFGS quadratic = %v, want ~[0,0], dist=%e", got, dist)
	}
}

func TestLBFGS_Rosenbrock(t *testing.T) {
	f := func(x []float64) float64 {
		return (1-x[0])*(1-x[0]) + 100*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0])
	}
	grad := func(x, g []float64) {
		g[0] = -2*(1-x[0]) + 200*(x[1]-x[0]*x[0])*(-2*x[0])
		g[1] = 200 * (x[1] - x[0]*x[0])
	}
	got := LBFGS(f, grad, []float64{-1.0, 1.0}, 10, 1000, 1e-10)
	dist := math.Sqrt((got[0]-1)*(got[0]-1) + (got[1]-1)*(got[1]-1))
	if dist > 1e-3 {
		t.Errorf("LBFGS Rosenbrock = %v, want ~[1,1], dist=%e", got, dist)
	}
}

func TestLBFGS_HighDim(t *testing.T) {
	// Minimize sum(x_i^2) in 10D, minimum at origin.
	n := 10
	f := func(x []float64) float64 {
		s := 0.0
		for _, xi := range x {
			s += xi * xi
		}
		return s
	}
	grad := func(x, g []float64) {
		for i := range x {
			g[i] = 2 * x[i]
		}
	}
	x0 := make([]float64, n)
	for i := range x0 {
		x0[i] = float64(i + 1)
	}
	got := LBFGS(f, grad, x0, 5, 500, 1e-10)
	norm := vecNorm(got)
	if norm > 1e-6 {
		t.Errorf("LBFGS 10D sphere norm = %e, want ~0", norm)
	}
}

func TestLBFGS_DoesNotModifyX0(t *testing.T) {
	f := func(x []float64) float64 { return x[0] * x[0] }
	grad := func(x, g []float64) { g[0] = 2 * x[0] }
	x0 := []float64{5.0}
	LBFGS(f, grad, x0, 5, 100, 1e-10)
	if x0[0] != 5.0 {
		t.Errorf("LBFGS modified x0: got %v, want 5.0", x0[0])
	}
}

func TestLBFGS_SmallHistory(t *testing.T) {
	// L-BFGS with m=1 should still converge on a simple problem.
	f := func(x []float64) float64 { return x[0]*x[0] + x[1]*x[1] }
	grad := func(x, g []float64) { g[0] = 2 * x[0]; g[1] = 2 * x[1] }
	got := LBFGS(f, grad, []float64{10.0, -10.0}, 1, 500, 1e-10)
	dist := math.Sqrt(got[0]*got[0] + got[1]*got[1])
	if dist > 1e-6 {
		t.Errorf("LBFGS m=1 = %v, want ~[0,0], dist=%e", got, dist)
	}
}

// ---------------------------------------------------------------------------
// CubicSplineNatural
// ---------------------------------------------------------------------------

func TestCubicSpline_InterpolatesExactly(t *testing.T) {
	xs := []float64{0, 1, 2, 3, 4}
	ys := []float64{0, 1, 4, 9, 16} // y = x^2 (approximately)
	s := CubicSplineNatural(xs, ys)

	for i, x := range xs {
		got := s(x)
		if math.Abs(got-ys[i]) > 1e-12 {
			t.Errorf("CubicSpline at data point x=%v: got %v, want %v", x, got, ys[i])
		}
	}
}

func TestCubicSpline_SmoothBetweenPoints(t *testing.T) {
	// Use sin(x) at 5 points, check intermediate values.
	xs := []float64{0, math.Pi / 4, math.Pi / 2, 3 * math.Pi / 4, math.Pi}
	ys := make([]float64, len(xs))
	for i, x := range xs {
		ys[i] = math.Sin(x)
	}
	s := CubicSplineNatural(xs, ys)

	// Check a few intermediate points. Cubic spline of sin should be close.
	checks := []float64{math.Pi / 8, math.Pi / 3, 2 * math.Pi / 3, 7 * math.Pi / 8}
	for _, x := range checks {
		got := s(x)
		want := math.Sin(x)
		if math.Abs(got-want) > 0.01 {
			t.Errorf("CubicSpline sin(%v): got %v, want %v, diff %e", x, got, want, math.Abs(got-want))
		}
	}
}

func TestCubicSpline_TwoPoints(t *testing.T) {
	// With 2 points, cubic spline degenerates to linear interpolation.
	xs := []float64{0, 1}
	ys := []float64{0, 1}
	s := CubicSplineNatural(xs, ys)

	got := s(0.5)
	if math.Abs(got-0.5) > 1e-12 {
		t.Errorf("CubicSpline 2-point at 0.5: got %v, want 0.5", got)
	}
}

func TestCubicSpline_ClampLeft(t *testing.T) {
	xs := []float64{1, 2, 3}
	ys := []float64{1, 4, 9}
	s := CubicSplineNatural(xs, ys)

	// Query left of domain should clamp to xs[0].
	got := s(0)
	want := s(1)
	if got != want {
		t.Errorf("CubicSpline clamp left: s(0)=%v, s(1)=%v", got, want)
	}
}

func TestCubicSpline_ClampRight(t *testing.T) {
	xs := []float64{1, 2, 3}
	ys := []float64{1, 4, 9}
	s := CubicSplineNatural(xs, ys)

	// Query right of domain should clamp to xs[n-1].
	got := s(10)
	want := s(3)
	if got != want {
		t.Errorf("CubicSpline clamp right: s(10)=%v, s(3)=%v", got, want)
	}
}

func TestCubicSpline_PanicMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("CubicSpline did not panic on mismatched lengths")
		}
	}()
	CubicSplineNatural([]float64{1, 2}, []float64{1})
}

func TestCubicSpline_PanicTooFew(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("CubicSpline did not panic on < 2 points")
		}
	}()
	CubicSplineNatural([]float64{1}, []float64{1})
}

func TestCubicSpline_PanicNotIncreasing(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("CubicSpline did not panic on non-increasing xs")
		}
	}()
	CubicSplineNatural([]float64{1, 3, 2}, []float64{1, 2, 3})
}

// ---------------------------------------------------------------------------
// LinearInterpolate
// ---------------------------------------------------------------------------

func TestLinearInterpolate_Midpoint(t *testing.T) {
	got := LinearInterpolate(0, 0, 10, 20, 5)
	if math.Abs(got-10) > 1e-15 {
		t.Errorf("LinearInterpolate midpoint = %v, want 10", got)
	}
}

func TestLinearInterpolate_AtEndpoints(t *testing.T) {
	got0 := LinearInterpolate(1, 5, 3, 15, 1)
	got1 := LinearInterpolate(1, 5, 3, 15, 3)
	if math.Abs(got0-5) > 1e-15 {
		t.Errorf("LinearInterpolate at x0: got %v, want 5", got0)
	}
	if math.Abs(got1-15) > 1e-15 {
		t.Errorf("LinearInterpolate at x1: got %v, want 15", got1)
	}
}

func TestLinearInterpolate_Extrapolate(t *testing.T) {
	// y = 2x on [0, 1], extrapolate to x=2 => y=4
	got := LinearInterpolate(0, 0, 1, 2, 2)
	if math.Abs(got-4) > 1e-15 {
		t.Errorf("LinearInterpolate extrapolate = %v, want 4", got)
	}
}

// ---------------------------------------------------------------------------
// SimulatedAnnealing
// ---------------------------------------------------------------------------

func TestSA_Bowl(t *testing.T) {
	// Minimize sum(x_i^2) in 2D. SA should find near-origin.
	f := func(x []float64) float64 { return x[0]*x[0] + x[1]*x[1] }
	neighbor := func(current, out []float64) {
		rng := rand.New(rand.NewSource(42)) // use local rng for perturbation
		_ = rng
		// Perturb each dimension by small random amount.
		for i := range current {
			out[i] = current[i] + (rand.Float64()-0.5)*0.1
		}
	}
	rng := rand.New(rand.NewSource(42))
	best, bestF := SimulatedAnnealing(f, []float64{5.0, -3.0}, neighbor, 100, 0.9999, 100000, rng)
	if bestF > 0.01 {
		t.Errorf("SA bowl: bestF=%v, best=%v, want near 0", bestF, best)
	}
}

func TestSA_ShiftedBowl(t *testing.T) {
	// Minimize (x-3)^2 + (y+2)^2, minimum at (3, -2).
	f := func(x []float64) float64 {
		return (x[0]-3)*(x[0]-3) + (x[1]+2)*(x[1]+2)
	}
	rng := rand.New(rand.NewSource(123))
	neighbor := func(current, out []float64) {
		for i := range current {
			out[i] = current[i] + (rng.Float64()-0.5)*0.2
		}
	}
	best, bestF := SimulatedAnnealing(f, []float64{0.0, 0.0}, neighbor, 100, 0.9999, 100000, rng)
	if bestF > 0.1 {
		t.Errorf("SA shifted bowl: bestF=%v, best=%v, want near (3,-2)", bestF, best)
	}
}

func TestSA_1D(t *testing.T) {
	// Minimize (x-7)^2, minimum at 7.
	f := func(x []float64) float64 { return (x[0] - 7) * (x[0] - 7) }
	rng := rand.New(rand.NewSource(99))
	neighbor := func(current, out []float64) {
		out[0] = current[0] + (rng.Float64()-0.5)*0.5
	}
	best, bestF := SimulatedAnnealing(f, []float64{0.0}, neighbor, 50, 0.9999, 50000, rng)
	if math.Abs(best[0]-7) > 0.5 {
		t.Errorf("SA 1D: best=%v (f=%v), want near 7", best, bestF)
	}
}

func TestSA_DoesNotModifyX0(t *testing.T) {
	f := func(x []float64) float64 { return x[0] * x[0] }
	rng := rand.New(rand.NewSource(1))
	neighbor := func(current, out []float64) {
		out[0] = current[0] + (rng.Float64()-0.5)*0.1
	}
	x0 := []float64{5.0}
	SimulatedAnnealing(f, x0, neighbor, 10, 0.99, 100, rng)
	if x0[0] != 5.0 {
		t.Errorf("SA modified x0: got %v, want 5.0", x0[0])
	}
}

func TestSA_Deterministic(t *testing.T) {
	// Same seed should produce same result.
	f := func(x []float64) float64 { return x[0]*x[0] + x[1]*x[1] }

	run := func(seed int64) ([]float64, float64) {
		rng := rand.New(rand.NewSource(seed))
		neighbor := func(current, out []float64) {
			for i := range current {
				out[i] = current[i] + (rng.Float64()-0.5)*0.1
			}
		}
		return SimulatedAnnealing(f, []float64{5.0, -3.0}, neighbor, 100, 0.999, 10000, rng)
	}

	best1, f1 := run(42)
	best2, f2 := run(42)

	if best1[0] != best2[0] || best1[1] != best2[1] || f1 != f2 {
		t.Errorf("SA not deterministic: run1=(%v,%v) run2=(%v,%v)", best1, f1, best2, f2)
	}
}

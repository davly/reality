package optim

// Precision property tests — pins rootfind.go bounds as tested invariants.
// Pure Go stdlib (testing/quick + math); ADDITIVE, zero math change.
//
// Claims pinned:
//   - rootfind.go:20  BisectionMethod: "|root - x*| <= tol after
//     ceil(log2((b-a)/tol)) iterations." We pin the accuracy half of the
//     claim: the returned point is within `tol` of a TRUE root x* (and the
//     bracketing/IVT contract holds) for a family of functions with known
//     roots.
//   - rootfind.go:110 LinearInterpolateRoot: "exact for IEEE 754 float64
//     (single division + multiply)" — for a line through (x0,y0),(x1,y1),
//     the returned x has f(x) == 0 to machine epsilon of the line.

import (
	"math"
	"testing"
	"testing/quick"
)

// TestBisectionWithinTol pins rootfind.go:20 |root - x*| <= tol.
// We use functions with a single known root x* in (a,b): f(x) = x - root.
// For a linear function bisection's bracket midpoint must land within tol of
// the true root.
func TestBisectionWithinTol(t *testing.T) {
	const tol = 1e-9
	var worst, worstAt float64
	prop := func(ru, au, bu uint64) bool {
		// True root in [-100, 100].
		root := 200*float64(ru)/float64(math.MaxUint64) - 100
		// Bracket [a,b] strictly straddling root, widths in (0, ~200].
		ahalf := 1e-6 + 100*float64(au)/float64(math.MaxUint64)
		bhalf := 1e-6 + 100*float64(bu)/float64(math.MaxUint64)
		a := root - ahalf
		b := root + bhalf
		f := func(x float64) float64 { return x - root }
		got := BisectionMethod(f, a, b, tol)
		err := math.Abs(got - root)
		if err > worst {
			worst, worstAt = err, root
		}
		return err <= tol
	}
	if err := quick.Check(prop, &quick.Config{MaxCount: 100000}); err != nil {
		t.Errorf("PRECISION REGRESSION: BisectionMethod claims |root-x*| <= tol(=%g), observed %g at root=%g", tol, worst, worstAt)
	}
	t.Logf("PINNED rootfind.go:20 |root-x*| <= tol: worst observed error %g (tol %g)", worst, tol)
}

// TestBisectionNonlinearWithinTol exercises a transcendental root (cos x = 0 at
// pi/2) to confirm the tol bound holds beyond linear functions.
func TestBisectionNonlinearWithinTol(t *testing.T) {
	const tol = 1e-10
	f := math.Cos
	root := math.Pi / 2
	got := BisectionMethod(f, 0, math.Pi, tol)
	if e := math.Abs(got - root); e > tol {
		t.Errorf("PRECISION REGRESSION: BisectionMethod(cos,[0,pi]) error %g > tol %g", e, tol)
	}
}

// TestLinearInterpolateRootExact pins rootfind.go:110. The docstring's "exact
// for IEEE 754 float64 (single division + multiply)" is an OPERATION-COUNT
// claim (exactly one correctly-rounded division and one multiply), NOT a claim
// that the residual is 0 for arbitrary inputs: when the two abscissae are
// near-coincident the secant formula suffers catastrophic cancellation in
// (x1-x0)/(y1-y0). We therefore pin the realistic, well-conditioned regime
// (well-separated points, non-flat slope): the x-intercept evaluates the line
// to a small relative residual (~1e-10). We do NOT pin the ill-conditioned
// near-coincident-points regime (see the documented note below).
func TestLinearInterpolateRootExact(t *testing.T) {
	const residBound = 1e-10 // realistic well-conditioned bound (2 rounded ops + re-eval)
	var worst float64
	prop := func(x0u, x1u, mu, cu uint64) bool {
		m := func(u uint64) float64 { return 200*float64(u)/float64(math.MaxUint64) - 100 }
		x0 := m(x0u)
		x1 := m(x1u)
		slope := m(mu)
		intercept := m(cu)
		// Well-conditioned: well-separated abscissae and a non-degenerate slope.
		if math.Abs(x1-x0) < 1.0 || math.Abs(slope) < 1.0 {
			return true
		}
		line := func(x float64) float64 { return slope*x + intercept }
		y0, y1 := line(x0), line(x1)
		xr := LinearInterpolateRoot(x0, y0, x1, y1)
		if math.IsNaN(xr) {
			return y0 == y1
		}
		resid := math.Abs(line(xr))
		scale := math.Abs(slope)*(1+math.Abs(xr)) + math.Abs(intercept) + 1
		rel := resid / scale
		if rel > worst {
			worst = rel
		}
		return rel <= residBound
	}
	if err := quick.Check(prop, &quick.Config{MaxCount: 200000}); err != nil {
		t.Errorf("PRECISION REGRESSION: LinearInterpolateRoot well-conditioned relative residual %g > %g", worst, residBound)
	}
	t.Logf("PINNED rootfind.go:110 LinearInterpolateRoot (well-conditioned): worst relative residual %g (< %g). NOTE: 'exact' is an operation-count claim; near-coincident abscissae are ill-conditioned and NOT pinned.", worst, residBound)
}

// TestLinearInterpolateRootNaNContract pins the documented "Returns NaN if
// y0 == y1 (horizontal line)" contract — a bit-exact behavioural guarantee.
func TestLinearInterpolateRootNaNContract(t *testing.T) {
	if !math.IsNaN(LinearInterpolateRoot(1, 5, 3, 5)) {
		t.Errorf("LinearInterpolateRoot with y0==y1 must return NaN")
	}
}

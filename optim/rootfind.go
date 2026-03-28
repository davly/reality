package optim

import "math"

// ---------------------------------------------------------------------------
// Root Finding & 1-D Optimization
//
// Classical numerical methods for finding zeros and minima of scalar
// functions. All functions are allocation-free and deterministic.
// ---------------------------------------------------------------------------

// BisectionMethod finds a root of f in the interval [a, b] by bisection.
// Requires f(a) and f(b) to have opposite signs (intermediate value theorem).
// Returns the midpoint of the final bracketing interval once the interval
// width is less than tol.
//
// Definition: repeatedly halve [a, b], keeping the sub-interval where the
// sign change occurs. Convergence rate: linear (one bit per iteration).
//
// Precision: |root - x*| <= tol after ceil(log2((b-a)/tol)) iterations.
// Reference: Burden & Faires, Numerical Analysis, Chapter 2.
func BisectionMethod(f func(float64) float64, a, b, tol float64) float64 {
	fa := f(a)
	for b-a > tol {
		mid := a + (b-a)/2 // avoids overflow compared to (a+b)/2
		fm := f(mid)
		if fm == 0 {
			return mid
		}
		if math.Signbit(fa) == math.Signbit(fm) {
			a = mid
			fa = fm
		} else {
			b = mid
		}
	}
	return a + (b-a)/2
}

// NewtonRaphson finds a root of f using Newton's method (also known as the
// Newton-Raphson method). Requires the derivative fPrime. Starting from x0,
// iterates x_{n+1} = x_n - f(x_n) / f'(x_n) until |f(x)| < tol or maxIter
// iterations are exhausted.
//
// Definition: first-order Taylor expansion root approximation.
// Convergence rate: quadratic (when it converges).
//
// Precision: depends on the function and starting point. May diverge for
// bad initial guesses or near inflection points.
// Reference: Burden & Faires, Numerical Analysis, Chapter 2.
func NewtonRaphson(f, fPrime func(float64) float64, x0, tol float64, maxIter int) float64 {
	x := x0
	for i := 0; i < maxIter; i++ {
		fx := f(x)
		if math.Abs(fx) < tol {
			return x
		}
		fpx := fPrime(x)
		if fpx == 0 {
			return x // derivative vanished — return best guess
		}
		x = x - fx/fpx
	}
	return x
}

// GoldenSectionSearch finds the minimum of a unimodal function f on [a, b]
// using the golden section method. Returns the approximate minimizer once
// the interval width is less than tol.
//
// Definition: analogous to bisection but for optimization. At each step,
// two interior probe points divide the interval in the golden ratio
// phi = (sqrt(5)-1)/2 ≈ 0.618, ensuring one probe is reused each step.
//
// Convergence rate: linear, interval shrinks by factor phi per iteration.
// Precision: |x* - x_min| <= tol after ceil(log_phi(tol/(b-a))) iterations.
// Reference: Kiefer, "Sequential Minimax Search for a Maximum," 1953.
func GoldenSectionSearch(f func(float64) float64, a, b, tol float64) float64 {
	gr := (math.Sqrt(5) - 1) / 2 // golden ratio conjugate ≈ 0.6180339887

	c := b - gr*(b-a)
	d := a + gr*(b-a)
	fc := f(c)
	fd := f(d)

	for b-a > tol {
		if fc < fd {
			b = d
			d = c
			fd = fc
			c = b - gr*(b-a)
			fc = f(c)
		} else {
			a = c
			c = d
			fc = fd
			d = a + gr*(b-a)
			fd = f(d)
		}
	}

	return a + (b-a)/2
}

// LinearInterpolateRoot computes the x-intercept of the line passing through
// (x0, y0) and (x1, y1). This is a single secant step, commonly used as a
// sub-routine in root-finding algorithms (regula falsi, secant method).
//
// Definition: x = x0 - y0 * (x1 - x0) / (y1 - y0)
// Precision: exact for IEEE 754 float64 (single division + multiply).
// Returns NaN if y0 == y1 (horizontal line — no unique root).
func LinearInterpolateRoot(x0, y0, x1, y1 float64) float64 {
	dy := y1 - y0
	if dy == 0 {
		return math.NaN()
	}
	return x0 - y0*(x1-x0)/dy
}

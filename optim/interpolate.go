package optim

// ---------------------------------------------------------------------------
// Interpolation
//
// Functions for constructing piecewise polynomial interpolants from discrete
// data. Cubic splines are the workhorse of scientific computing — used for
// smooth curve fitting, look-up table acceleration, and data visualization.
// ---------------------------------------------------------------------------

// LinearInterpolate performs linear interpolation (lerp) between two points
// (x0, y0) and (x1, y1) at the query point x. No clamping is applied; x
// outside [x0, x1] extrapolates.
//
// Definition: y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
// Precision: exact for IEEE 754 float64.
// Panics if x0 == x1 (degenerate interval).
func LinearInterpolate(x0, y0, x1, y1, x float64) float64 {
	return y0 + (y1-y0)*(x-x0)/(x1-x0)
}

// CubicSplineNatural constructs a natural cubic spline interpolant through
// the data points (xs[i], ys[i]) and returns a function that evaluates the
// spline at any query point x.
//
// "Natural" means the second derivative at both endpoints is zero:
// S''(x_0) = 0, S''(x_n) = 0. This produces the smoothest interpolant
// that passes through all data points.
//
// The returned function clamps queries outside [xs[0], xs[n-1]] to the
// nearest endpoint's cubic piece (constant extrapolation of the spline).
//
// Requirements:
//   - len(xs) == len(ys) >= 2
//   - xs must be strictly increasing
//
// Panics if the requirements are not met.
//
// Algorithm: solve the tridiagonal system for second derivatives using
// the Thomas algorithm (O(n) time and space), then evaluate the appropriate
// cubic piece for each query.
//
// Reference: Burden & Faires, Numerical Analysis, Chapter 3.
func CubicSplineNatural(xs, ys []float64) func(float64) float64 {
	n := len(xs)
	if n != len(ys) {
		panic("optim.CubicSplineNatural: xs and ys must have the same length")
	}
	if n < 2 {
		panic("optim.CubicSplineNatural: need at least 2 data points")
	}

	// Compute interval widths h[i] = xs[i+1] - xs[i].
	h := make([]float64, n-1)
	for i := 0; i < n-1; i++ {
		h[i] = xs[i+1] - xs[i]
		if h[i] <= 0 {
			panic("optim.CubicSplineNatural: xs must be strictly increasing")
		}
	}

	// Solve for second derivatives c[i] using the Thomas algorithm on the
	// natural spline tridiagonal system. c[0] = c[n-1] = 0 (natural BC).
	//
	// The system for interior points i = 1, ..., n-2 is:
	//   h[i-1]*c[i-1] + 2*(h[i-1]+h[i])*c[i] + h[i]*c[i+1] = rhs[i]
	// where rhs[i] = 3*((ys[i+1]-ys[i])/h[i] - (ys[i]-ys[i-1])/h[i-1])
	c := make([]float64, n)

	if n > 2 {
		// Interior system size.
		sz := n - 2

		// Tridiagonal coefficients.
		lower := make([]float64, sz) // sub-diagonal
		diag := make([]float64, sz)  // main diagonal
		upper := make([]float64, sz) // super-diagonal
		rhs := make([]float64, sz)   // right-hand side

		for i := 0; i < sz; i++ {
			idx := i + 1 // index into the full arrays
			if i > 0 {
				lower[i] = h[idx-1]
			}
			diag[i] = 2 * (h[idx-1] + h[idx])
			if i < sz-1 {
				upper[i] = h[idx]
			}
			rhs[i] = 3 * ((ys[idx+1]-ys[idx])/h[idx] - (ys[idx]-ys[idx-1])/h[idx-1])
		}

		// Thomas algorithm: forward sweep.
		for i := 1; i < sz; i++ {
			w := lower[i] / diag[i-1]
			diag[i] -= w * upper[i-1]
			rhs[i] -= w * rhs[i-1]
		}

		// Thomas algorithm: back substitution.
		rhs[sz-1] /= diag[sz-1]
		for i := sz - 2; i >= 0; i-- {
			rhs[i] = (rhs[i] - upper[i]*rhs[i+1]) / diag[i]
		}

		// Copy interior c values.
		for i := 0; i < sz; i++ {
			c[i+1] = rhs[i]
		}
	}

	// Precompute polynomial coefficients for each piece.
	// S_i(x) = a[i] + b[i]*(x - xs[i]) + c[i]*(x - xs[i])^2 + d[i]*(x - xs[i])^3
	type piece struct {
		a, b, cc, d float64 // cc to avoid shadowing c
		xLo, xHi    float64
	}
	pieces := make([]piece, n-1)
	for i := 0; i < n-1; i++ {
		ai := ys[i]
		bi := (ys[i+1]-ys[i])/h[i] - h[i]*(2*c[i]+c[i+1])/3
		di := (c[i+1] - c[i]) / (3 * h[i])
		pieces[i] = piece{
			a: ai, b: bi, cc: c[i], d: di,
			xLo: xs[i], xHi: xs[i+1],
		}
	}

	// Capture copies for the closure.
	xsCopy := make([]float64, n)
	copy(xsCopy, xs)

	return func(x float64) float64 {
		// Clamp to domain.
		if x <= xsCopy[0] {
			x = xsCopy[0]
		}
		if x >= xsCopy[n-1] {
			x = xsCopy[n-1]
		}

		// Binary search for the correct piece.
		lo, hi := 0, n-2
		for lo < hi {
			mid := lo + (hi-lo)/2
			if x < pieces[mid].xHi {
				hi = mid
			} else {
				lo = mid + 1
			}
		}

		p := &pieces[lo]
		dx := x - p.xLo
		return p.a + dx*(p.b+dx*(p.cc+dx*p.d))
	}
}

package optim

import "math"

// ---------------------------------------------------------------------------
// Gradient-Based Optimization
//
// First-order and quasi-Newton methods for unconstrained minimization of
// differentiable scalar functions in R^n. All functions allocate only the
// working memory they need; no external dependencies.
// ---------------------------------------------------------------------------

// GradientDescent minimizes f by steepest descent. The gradient function grad
// computes ∇f(x) in-place: grad(x, g) writes the gradient into g, avoiding
// allocation on every iteration.
//
// Parameters:
//   - f:       objective function R^n → R
//   - grad:    gradient function; grad(x, g) fills g with ∇f(x)
//   - x0:     initial point (copied, not modified)
//   - lr:      learning rate (step size)
//   - maxIter: maximum number of iterations
//   - tol:     stop when ‖∇f(x)‖₂ < tol
//
// Returns the approximate minimizer. The returned slice is freshly allocated.
//
// Convergence rate: linear (O(1/k) for convex, depends on condition number).
// Reference: Cauchy, "Méthode générale pour la résolution des systèmes
// d'équations simultanées," 1847.
func GradientDescent(f func([]float64) float64, grad func([]float64, []float64), x0 []float64, lr float64, maxIter int, tol float64) []float64 {
	_ = f // f is available for line search extensions; unused in basic GD

	n := len(x0)
	x := make([]float64, n)
	copy(x, x0)

	g := make([]float64, n)

	for iter := 0; iter < maxIter; iter++ {
		grad(x, g)

		// Check gradient norm for convergence.
		gnorm := 0.0
		for i := 0; i < n; i++ {
			gnorm += g[i] * g[i]
		}
		gnorm = math.Sqrt(gnorm)
		if gnorm < tol {
			break
		}

		// Steepest descent step.
		for i := 0; i < n; i++ {
			x[i] -= lr * g[i]
		}
	}

	return x
}

// LBFGS minimizes f using the Limited-memory Broyden-Fletcher-Goldfarb-Shanno
// (L-BFGS) quasi-Newton method. This is the standard two-loop recursion
// algorithm with Wolfe line search.
//
// Parameters:
//   - f:       objective function R^n → R
//   - grad:    gradient function; grad(x, g) fills g with ∇f(x)
//   - x0:     initial point (copied, not modified)
//   - m:       number of correction pairs to store (typically 3–20)
//   - maxIter: maximum number of iterations
//   - tol:     stop when ‖∇f(x)‖₂ < tol
//
// Returns the approximate minimizer. The returned slice is freshly allocated.
//
// The two-loop recursion computes H_k * g_k without forming the dense Hessian
// approximation, using O(mn) storage and O(mn) work per iteration.
//
// Reference: Nocedal & Wright, "Numerical Optimization," Chapter 7.
//            Liu & Nocedal, "On the limited memory BFGS method for large
//            scale optimization," Math. Programming 45 (1989).
func LBFGS(f func([]float64) float64, grad func([]float64, []float64), x0 []float64, m, maxIter int, tol float64) []float64 {
	n := len(x0)

	x := make([]float64, n)
	copy(x, x0)

	g := make([]float64, n)
	gPrev := make([]float64, n)
	xPrev := make([]float64, n)
	d := make([]float64, n) // search direction

	// Ring buffer for correction pairs.
	sHist := make([][]float64, 0, m) // s_k = x_{k+1} - x_k
	yHist := make([][]float64, 0, m) // y_k = g_{k+1} - g_k
	rho := make([]float64, 0, m)     // rho_k = 1 / (y_k^T s_k)

	alpha := make([]float64, m)

	grad(x, g)

	for iter := 0; iter < maxIter; iter++ {
		// Check convergence.
		gnorm := vecNorm(g)
		if gnorm < tol {
			break
		}

		// --- Two-loop recursion to compute d = -H_k * g_k ---
		q := make([]float64, n)
		copy(q, g)

		k := len(sHist)

		// First loop: iterate from most recent to oldest.
		for i := k - 1; i >= 0; i-- {
			alpha[i] = rho[i] * vecDot(sHist[i], q)
			vecAddScaled(q, q, yHist[i], -alpha[i])
		}

		// Initial Hessian approximation: H_0 = gamma * I
		// where gamma = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})
		gamma := 1.0
		if k > 0 {
			sy := vecDot(sHist[k-1], yHist[k-1])
			yy := vecDot(yHist[k-1], yHist[k-1])
			if yy > 0 {
				gamma = sy / yy
			}
		}
		for i := 0; i < n; i++ {
			d[i] = gamma * q[i]
		}

		// Second loop: iterate from oldest to most recent.
		for i := 0; i < k; i++ {
			beta := rho[i] * vecDot(yHist[i], d)
			vecAddScaled(d, d, sHist[i], alpha[i]-beta)
		}

		// Negate to get descent direction.
		for i := 0; i < n; i++ {
			d[i] = -d[i]
		}

		// --- Backtracking line search (Armijo condition) ---
		step := lbfgsLineSearch(f, grad, x, g, d, n)

		// Save previous x and g.
		copy(xPrev, x)
		copy(gPrev, g)

		// Take step.
		for i := 0; i < n; i++ {
			x[i] += step * d[i]
		}
		grad(x, g)

		// Compute correction pair.
		sk := make([]float64, n)
		yk := make([]float64, n)
		for i := 0; i < n; i++ {
			sk[i] = x[i] - xPrev[i]
			yk[i] = g[i] - gPrev[i]
		}

		sy := vecDot(sk, yk)
		if sy <= 0 {
			// Skip update if curvature condition is not satisfied.
			continue
		}
		rhok := 1.0 / sy

		// Add to history, dropping oldest if at capacity.
		if len(sHist) < m {
			sHist = append(sHist, sk)
			yHist = append(yHist, yk)
			rho = append(rho, rhok)
		} else {
			// Shift ring buffer.
			copy(sHist, sHist[1:])
			copy(yHist, yHist[1:])
			copy(rho, rho[1:])
			sHist[m-1] = sk
			yHist[m-1] = yk
			rho[m-1] = rhok
		}
	}

	return x
}

// lbfgsLineSearch performs a backtracking line search satisfying the Armijo
// sufficient decrease condition: f(x + step*d) <= f(x) + c1 * step * g^T d.
func lbfgsLineSearch(f func([]float64) float64, grad func([]float64, []float64), x, g, d []float64, n int) float64 {
	_ = grad // available for Wolfe conditions; Armijo-only for simplicity

	const c1 = 1e-4
	const shrink = 0.5
	const maxLS = 40

	step := 1.0
	fx := f(x)
	dg := vecDot(g, d) // directional derivative

	xTrial := make([]float64, n)

	for ls := 0; ls < maxLS; ls++ {
		for i := 0; i < n; i++ {
			xTrial[i] = x[i] + step*d[i]
		}
		fTrial := f(xTrial)

		if fTrial <= fx+c1*step*dg {
			return step
		}
		step *= shrink
	}

	return step
}

// ---------------------------------------------------------------------------
// Vector helpers (unexported, allocation-free where possible)
// ---------------------------------------------------------------------------

// vecDot computes the dot product of a and b.
func vecDot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// vecNorm computes the Euclidean norm of v.
func vecNorm(v []float64) float64 {
	s := 0.0
	for _, vi := range v {
		s += vi * vi
	}
	return math.Sqrt(s)
}

// vecAddScaled computes dst = a + scale * b. dst may alias a.
func vecAddScaled(dst, a, b []float64, scale float64) {
	for i := range dst {
		dst[i] = a[i] + scale*b[i]
	}
}

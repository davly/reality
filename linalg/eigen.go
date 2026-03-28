package linalg

import "math"

// QRAlgorithm computes the eigenvalues of a real symmetric n x n matrix A
// using Householder tridiagonalization followed by the implicit QL algorithm
// with Wilkinson shifts.
//
// A is n x n row-major (not modified). eigenvalues is length n (pre-allocated),
// filled with eigenvalues in descending order upon completion.
// maxIter is the maximum number of total QL iterations allowed.
//
// Returns the total number of iterations performed.
//
// Valid input range: n >= 1, A must be symmetric, maxIter > 0.
// Allocates internal workspace for the tridiagonal representation.
//
// Reference: Golub & Van Loan, "Matrix Computations", Ch. 8;
// Press et al., "Numerical Recipes", tqli algorithm.
func QRAlgorithm(A []float64, n int, eigenvalues []float64, maxIter int) int {
	nn := n * n
	if len(A) != nn {
		panic("linalg.QRAlgorithm: len(A) != n*n")
	}
	if len(eigenvalues) != n {
		panic("linalg.QRAlgorithm: len(eigenvalues) != n")
	}

	if n == 1 {
		eigenvalues[0] = A[0]
		return 0
	}

	// Step 1: Reduce to tridiagonal form via Householder reflections.
	d := make([]float64, n) // diagonal
	e := make([]float64, n) // off-diagonal (e[0..n-2] used)
	tridiagonalize(A, n, d, e)

	// Step 2: QL algorithm with implicit shifts (Numerical Recipes tqli).
	totalIter := tqli(d, e, n, maxIter)

	// Copy and sort descending.
	copy(eigenvalues, d)
	for i := 1; i < n; i++ {
		key := eigenvalues[i]
		j := i - 1
		for j >= 0 && eigenvalues[j] < key {
			eigenvalues[j+1] = eigenvalues[j]
			j--
		}
		eigenvalues[j+1] = key
	}

	return totalIter
}

// tqli implements the implicit QL algorithm for a symmetric tridiagonal matrix.
// d is the diagonal (length n), e is the off-diagonal (length n, e[n-1] unused).
// On return, d contains the eigenvalues.
//
// Based on Numerical Recipes tqli and LAPACK dsteqr.
func tqli(d, e []float64, n, maxIter int) int {
	totalIter := 0

	for l := 0; l < n; l++ {
		for iter := 0; iter < maxIter; iter++ {
			// Find the smallest m >= l such that e[m] is effectively zero.
			var m int
			for m = l; m < n-1; m++ {
				dd := math.Abs(d[m]) + math.Abs(d[m+1])
				if dd == 0 {
					dd = 1e-300
				}
				if math.Abs(e[m]) <= 1e-15*dd {
					break
				}
			}
			if m == l {
				break // Converged.
			}

			totalIter++
			if totalIter > maxIter*n {
				return totalIter // Give up.
			}

			// Wilkinson shift from trailing 2×2 of the unreduced block.
			g := (d[l+1] - d[l]) / (2.0 * e[l])
			r := math.Sqrt(g*g + 1.0)
			g = d[m] - d[l] + e[l]/(g+math.Copysign(r, g))

			s := 1.0
			c := 1.0
			p := 0.0

			// QL rotation chase from m-1 down to l.
			for i := m - 1; i >= l; i-- {
				f := s * e[i]
				b := c * e[i]

				if math.Abs(f) >= math.Abs(g) {
					c = g / f
					r = math.Sqrt(c*c + 1.0)
					e[i+1] = f * r
					s = 1.0 / r
					c *= s
				} else {
					s = f / g
					r = math.Sqrt(s*s + 1.0)
					e[i+1] = g * r
					c = 1.0 / r
					s *= c
				}

				g = d[i+1] - p
				r = (d[i]-g)*s + 2.0*c*b
				p = s * r
				d[i+1] = g + p
				g = c*r - b
			}

			d[l] -= p
			e[l] = g
			e[m] = 0.0
		}
	}

	return totalIter
}

// tridiagonalize reduces a symmetric n x n matrix A to tridiagonal form
// using Householder reflections. d has length n (diagonal), e has length n
// (off-diagonal in e[0..n-2], e[n-1] = 0).
// A is not modified.
//
// Reference: Golub & Van Loan, "Matrix Computations", Algorithm 8.3.1.
func tridiagonalize(A []float64, n int, d, e []float64) {
	// Work on a copy.
	w := make([]float64, n*n)
	copy(w, A)

	for k := 0; k < n-2; k++ {
		// Compute ||w[k+1:n, k]||.
		sigma := 0.0
		for i := k + 1; i < n; i++ {
			sigma += w[i*n+k] * w[i*n+k]
		}
		sigma = math.Sqrt(sigma)

		if sigma < 1e-300 {
			e[k] = 0
			continue
		}

		// Choose sign to avoid cancellation.
		if w[(k+1)*n+k] >= 0 {
			sigma = -sigma
		}
		e[k] = sigma

		// Householder vector v: v[k+1] = w[k+1,k] - sigma, v[i] = w[i,k] for i > k+1.
		w[(k+1)*n+k] -= sigma
		h := w[(k+1)*n+k]*w[(k+1)*n+k] // recompute v^T v
		for i := k + 2; i < n; i++ {
			h += w[i*n+k] * w[i*n+k]
		}
		if h < 1e-300 {
			continue
		}
		h = 2.0 / h

		// p = h * W[k+1:n, k+1:n] * v
		p := make([]float64, n)
		for i := k + 1; i < n; i++ {
			s := 0.0
			for j := k + 1; j < n; j++ {
				s += w[i*n+j] * w[j*n+k]
			}
			p[i] = h * s
		}

		// K = h/2 * v^T p
		vtp := 0.0
		for i := k + 1; i < n; i++ {
			vtp += w[i*n+k] * p[i]
		}
		K := 0.5 * h * vtp

		// q = p - K*v, then W = W - v*q^T - q*v^T (symmetric update)
		for i := k + 1; i < n; i++ {
			p[i] -= K * w[i*n+k]
		}
		for i := k + 1; i < n; i++ {
			for j := k + 1; j <= i; j++ {
				val := w[i*n+j] - w[i*n+k]*p[j] - p[i]*w[j*n+k]
				w[i*n+j] = val
				w[j*n+i] = val
			}
		}
	}

	// Extract diagonal and last off-diagonal.
	for i := 0; i < n; i++ {
		d[i] = w[i*n+i]
	}
	if n >= 2 {
		e[n-2] = w[(n-1)*n+(n-2)]
	}
	if n > 0 {
		e[n-1] = 0
	}
}

package infogeo

import "math"

// A Kernel is a positive-semidefinite kernel function k(x, y) on R^d.  It
// is invoked once per (i, j) pair inside MMD and must be deterministic.
type Kernel func(x, y []float64) float64

// GaussianKernel returns the Gaussian (RBF) kernel
//
//	k(x, y) = exp( -||x - y||^2 / (2 * bandwidth^2) )
//
// Universal: distinguishes any two distributions on R^d when used in MMD.
// Bandwidth must be > 0; a common heuristic is the median pairwise
// distance over the union of samples.
func GaussianKernel(bandwidth float64) Kernel {
	if !(bandwidth > 0) {
		panic("infogeo: GaussianKernel bandwidth must be positive")
	}
	denom := 2.0 * bandwidth * bandwidth
	return func(x, y []float64) float64 {
		var s float64
		for i, xi := range x {
			d := xi - y[i]
			s += d * d
		}
		return math.Exp(-s / denom)
	}
}

// LaplacianKernel returns the Laplacian kernel
//
//	k(x, y) = exp( -||x - y||_1 / bandwidth )
//
// Also universal.  Heavier tails than Gaussian; sometimes more robust to
// outliers.  Bandwidth must be > 0.
func LaplacianKernel(bandwidth float64) Kernel {
	if !(bandwidth > 0) {
		panic("infogeo: LaplacianKernel bandwidth must be positive")
	}
	return func(x, y []float64) float64 {
		var s float64
		for i, xi := range x {
			s += math.Abs(xi - y[i])
		}
		return math.Exp(-s / bandwidth)
	}
}

// MMD2Biased returns the biased estimator of squared Maximum Mean Discrepancy
// MMD^2(P, Q) given samples X drawn from P and Y drawn from Q:
//
//	MMD^2_b = (1/m^2) sum_{i,j} k(x_i, x_j)
//	         + (1/n^2) sum_{i,j} k(y_i, y_j)
//	         - (2/(m*n)) sum_{i,j} k(x_i, y_j)
//
// X and Y are slices of float64 vectors with the same per-vector dimension.
// The biased estimator is non-negative but has positive expectation under
// the null (X and Y from the same distribution); for hypothesis testing
// use MMD2Unbiased.
//
// Reference: Gretton A. et al. (2012). A kernel two-sample test.
// JMLR 13:723-773.  Equation (5).
func MMD2Biased(X, Y [][]float64, k Kernel) (float64, error) {
	if len(X) == 0 || len(Y) == 0 {
		return 0, ErrInvalidParameter
	}
	if k == nil {
		return 0, ErrInvalidParameter
	}
	d := len(X[0])
	for _, xi := range X {
		if len(xi) != d {
			return 0, ErrLengthMismatch
		}
	}
	for _, yi := range Y {
		if len(yi) != d {
			return 0, ErrLengthMismatch
		}
	}
	m := float64(len(X))
	n := float64(len(Y))

	var kxx, kyy, kxy float64
	for _, xi := range X {
		for _, xj := range X {
			kxx += k(xi, xj)
		}
	}
	for _, yi := range Y {
		for _, yj := range Y {
			kyy += k(yi, yj)
		}
	}
	for _, xi := range X {
		for _, yj := range Y {
			kxy += k(xi, yj)
		}
	}

	return kxx/(m*m) + kyy/(n*n) - 2.0*kxy/(m*n), nil
}

// MMD2Unbiased returns the unbiased estimator of MMD^2 (Gretton 2012 eq 3).
//
//	MMD^2_u = (1/(m*(m-1))) sum_{i != j} k(x_i, x_j)
//	         + (1/(n*(n-1))) sum_{i != j} k(y_i, y_j)
//	         - (2/(m*n)) sum_{i,j} k(x_i, y_j)
//
// Has zero expectation under the null but can be slightly negative on
// finite samples.  Preferred for hypothesis testing; use MMD2Biased if a
// non-negativity guarantee is required.
//
// Requires len(X) >= 2 and len(Y) >= 2.
func MMD2Unbiased(X, Y [][]float64, k Kernel) (float64, error) {
	if len(X) < 2 || len(Y) < 2 {
		return 0, ErrInvalidParameter
	}
	if k == nil {
		return 0, ErrInvalidParameter
	}
	d := len(X[0])
	for _, xi := range X {
		if len(xi) != d {
			return 0, ErrLengthMismatch
		}
	}
	for _, yi := range Y {
		if len(yi) != d {
			return 0, ErrLengthMismatch
		}
	}
	m := float64(len(X))
	n := float64(len(Y))

	var kxx, kyy, kxy float64
	for i, xi := range X {
		for j, xj := range X {
			if i == j {
				continue
			}
			kxx += k(xi, xj)
		}
	}
	for i, yi := range Y {
		for j, yj := range Y {
			if i == j {
				continue
			}
			kyy += k(yi, yj)
		}
	}
	for _, xi := range X {
		for _, yj := range Y {
			kxy += k(xi, yj)
		}
	}

	return kxx/(m*(m-1.0)) + kyy/(n*(n-1.0)) - 2.0*kxy/(m*n), nil
}

// MedianHeuristicBandwidth returns the median pairwise Euclidean distance
// over the concatenation of X and Y, the standard heuristic bandwidth for
// the Gaussian kernel in MMD.  Returns 0 if fewer than 2 samples total.
//
// Allocates an intermediate slice of pairwise distances of size N*(N-1)/2
// where N = len(X) + len(Y).
func MedianHeuristicBandwidth(X, Y [][]float64) float64 {
	all := make([][]float64, 0, len(X)+len(Y))
	all = append(all, X...)
	all = append(all, Y...)
	N := len(all)
	if N < 2 {
		return 0
	}
	dists := make([]float64, 0, N*(N-1)/2)
	for i := 0; i < N; i++ {
		for j := i + 1; j < N; j++ {
			var s float64
			for k := range all[i] {
				d := all[i][k] - all[j][k]
				s += d * d
			}
			dists = append(dists, math.Sqrt(s))
		}
	}
	return median(dists)
}

func median(xs []float64) float64 {
	// Local insertion sort to avoid pulling in sort and to keep imports
	// minimal.  Caller-supplied slice is mutated.
	n := len(xs)
	for i := 1; i < n; i++ {
		v := xs[i]
		j := i - 1
		for j >= 0 && xs[j] > v {
			xs[j+1] = xs[j]
			j--
		}
		xs[j+1] = v
	}
	if n%2 == 1 {
		return xs[n/2]
	}
	return 0.5 * (xs[n/2-1] + xs[n/2])
}

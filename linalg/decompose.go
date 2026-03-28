package linalg

import "math"

// LUDecompose performs LU decomposition with partial pivoting on an n x n matrix A.
// It factors PA = LU where P is a permutation matrix, L is lower-triangular with
// unit diagonal, and U is upper-triangular.
//
// A is n x n row-major (not modified). L, U are n x n row-major (pre-allocated).
// perm is the permutation vector of length n (pre-allocated): row i of the output
// corresponds to row perm[i] of the original matrix.
//
// Returns false if A is singular (zero pivot encountered). In that case L, U, perm
// contents are undefined.
//
// Definition: PA = LU where L[i][j]=0 for j>i, L[i][i]=1, U[i][j]=0 for j<i.
// Valid input range: n >= 1, all finite float64 entries.
// Zero heap allocations (caller provides all buffers).
//
// Panics if slice lengths are inconsistent with n.
func LUDecompose(A []float64, n int, L, U []float64, perm []int) bool {
	nn := n * n
	if len(A) != nn {
		panic("linalg.LUDecompose: len(A) != n*n")
	}
	if len(L) != nn {
		panic("linalg.LUDecompose: len(L) != n*n")
	}
	if len(U) != nn {
		panic("linalg.LUDecompose: len(U) != n*n")
	}
	if len(perm) != n {
		panic("linalg.LUDecompose: len(perm) != n")
	}

	// Copy A into U (we work in-place on U).
	copy(U, A)

	// Initialize L to identity, perm to identity.
	for i := 0; i < nn; i++ {
		L[i] = 0
	}
	for i := 0; i < n; i++ {
		L[i*n+i] = 1
		perm[i] = i
	}

	for k := 0; k < n; k++ {
		// Find pivot: row with largest absolute value in column k, from row k down.
		maxVal := math.Abs(U[k*n+k])
		maxRow := k
		for i := k + 1; i < n; i++ {
			v := math.Abs(U[i*n+k])
			if v > maxVal {
				maxVal = v
				maxRow = i
			}
		}

		if maxVal < 1e-300 {
			return false // singular
		}

		// Swap rows k and maxRow in U.
		if maxRow != k {
			for j := 0; j < n; j++ {
				U[k*n+j], U[maxRow*n+j] = U[maxRow*n+j], U[k*n+j]
			}
			// Swap the already-computed L entries (columns 0..k-1).
			for j := 0; j < k; j++ {
				L[k*n+j], L[maxRow*n+j] = L[maxRow*n+j], L[k*n+j]
			}
			perm[k], perm[maxRow] = perm[maxRow], perm[k]
		}

		// Eliminate below pivot.
		pivot := U[k*n+k]
		for i := k + 1; i < n; i++ {
			factor := U[i*n+k] / pivot
			L[i*n+k] = factor
			U[i*n+k] = 0
			for j := k + 1; j < n; j++ {
				U[i*n+j] -= factor * U[k*n+j]
			}
		}
	}

	return true
}

// LUSolve solves the system Ax = b given the LU decomposition of A (with pivoting).
// L is lower-triangular with unit diagonal, U is upper-triangular, perm is the
// permutation vector — all from LUDecompose. b is the right-hand side vector of
// length n. x is the solution vector of length n (pre-allocated).
//
// The solution is computed in two steps:
//   1. Forward substitution: Ly = Pb
//   2. Back substitution:    Ux = y
//
// Zero heap allocations (caller provides x; forward substitution reuses x as scratch).
//
// Panics if slice lengths are inconsistent with n.
func LUSolve(L, U []float64, n int, perm []int, b, x []float64) {
	nn := n * n
	if len(L) != nn {
		panic("linalg.LUSolve: len(L) != n*n")
	}
	if len(U) != nn {
		panic("linalg.LUSolve: len(U) != n*n")
	}
	if len(perm) != n {
		panic("linalg.LUSolve: len(perm) != n")
	}
	if len(b) != n {
		panic("linalg.LUSolve: len(b) != n")
	}
	if len(x) != n {
		panic("linalg.LUSolve: len(x) != n")
	}

	// Forward substitution: Ly = Pb.
	// Use x to store y temporarily.
	for i := 0; i < n; i++ {
		sum := b[perm[i]]
		for j := 0; j < i; j++ {
			sum -= L[i*n+j] * x[j]
		}
		x[i] = sum // L[i][i] = 1
	}

	// Back substitution: Ux = y.
	for i := n - 1; i >= 0; i-- {
		sum := x[i]
		for j := i + 1; j < n; j++ {
			sum -= U[i*n+j] * x[j]
		}
		x[i] = sum / U[i*n+i]
	}
}

// Inverse computes the inverse of an n x n matrix A via LU decomposition.
// A is n x n row-major (not modified). out is n x n row-major (pre-allocated).
//
// Returns false if A is singular. In that case out contents are undefined.
//
// The inverse is computed column-by-column: for each standard basis vector e_j,
// solve Ax_j = e_j using LU decomposition.
//
// Valid input range: n >= 1, all finite float64 entries, A must be non-singular.
// Allocates temporary buffers for L, U, perm internally (setup cost, not hot-path).
//
// Panics if slice lengths are inconsistent with n.
func Inverse(A []float64, n int, out []float64) bool {
	nn := n * n
	if len(A) != nn {
		panic("linalg.Inverse: len(A) != n*n")
	}
	if len(out) != nn {
		panic("linalg.Inverse: len(out) != n*n")
	}

	L := make([]float64, nn)
	U := make([]float64, nn)
	perm := make([]int, n)

	if !LUDecompose(A, n, L, U, perm) {
		return false
	}

	// Solve for each column of the inverse.
	e := make([]float64, n) // basis vector
	col := make([]float64, n)
	for j := 0; j < n; j++ {
		// Set e to j-th standard basis vector.
		for i := 0; i < n; i++ {
			e[i] = 0
		}
		e[j] = 1

		LUSolve(L, U, n, perm, e, col)

		// Copy column j into out.
		for i := 0; i < n; i++ {
			out[i*n+j] = col[i]
		}
	}

	return true
}

// Determinant computes the determinant of an n x n matrix A via LU decomposition.
// A is n x n row-major (not modified).
//
// Definition: det(A) = (-1)^s * product(U[i][i]) where s is the number of row swaps
// in partial pivoting.
//
// Returns 0 for singular matrices.
// Valid input range: n >= 1, all finite float64 entries.
// Allocates temporary buffers internally.
//
// Panics if len(A) != n*n.
func Determinant(A []float64, n int) float64 {
	nn := n * n
	if len(A) != nn {
		panic("linalg.Determinant: len(A) != n*n")
	}

	if n == 1 {
		return A[0]
	}

	L := make([]float64, nn)
	U := make([]float64, nn)
	perm := make([]int, n)

	if !LUDecompose(A, n, L, U, perm) {
		return 0
	}

	// Count the number of permutation swaps.
	// perm[i] tells us which original row ended up at position i.
	// Count the number of transpositions by counting cycles.
	swaps := 0
	visited := make([]bool, n)
	for i := 0; i < n; i++ {
		if visited[i] {
			continue
		}
		cycleLen := 0
		j := i
		for !visited[j] {
			visited[j] = true
			j = perm[j]
			cycleLen++
		}
		swaps += cycleLen - 1
	}

	det := 1.0
	if swaps%2 == 1 {
		det = -1.0
	}
	for i := 0; i < n; i++ {
		det *= U[i*n+i]
	}

	return det
}

// CholeskyDecompose computes the Cholesky decomposition of a symmetric positive
// definite n x n matrix A. It finds lower-triangular L such that A = L * L^T.
//
// A is n x n row-major (not modified). L is n x n row-major (pre-allocated).
// Only the lower triangle of A is read.
//
// Returns false if A is not positive definite (negative value under square root).
// In that case L contents are undefined.
//
// Definition: L[i][j] = (A[i][j] - sum(L[i][k]*L[j][k], k=0..j-1)) / L[j][j]  for i > j
//             L[j][j] = sqrt(A[j][j] - sum(L[j][k]^2, k=0..j-1))
//
// Valid input range: n >= 1, A must be symmetric positive definite.
// Zero heap allocations (caller provides all buffers).
//
// Panics if slice lengths are inconsistent with n.
func CholeskyDecompose(A []float64, n int, L []float64) bool {
	nn := n * n
	if len(A) != nn {
		panic("linalg.CholeskyDecompose: len(A) != n*n")
	}
	if len(L) != nn {
		panic("linalg.CholeskyDecompose: len(L) != n*n")
	}

	// Zero out L.
	for i := range L {
		L[i] = 0
	}

	for j := 0; j < n; j++ {
		// Diagonal element.
		sum := A[j*n+j]
		for k := 0; k < j; k++ {
			sum -= L[j*n+k] * L[j*n+k]
		}
		if sum <= 0 {
			return false // not positive definite
		}
		L[j*n+j] = math.Sqrt(sum)

		ljj := L[j*n+j]
		// Below-diagonal elements in column j.
		for i := j + 1; i < n; i++ {
			sum = A[i*n+j]
			for k := 0; k < j; k++ {
				sum -= L[i*n+k] * L[j*n+k]
			}
			L[i*n+j] = sum / ljj
		}
	}

	return true
}

// CholeskySolve solves the system Ax = b given the Cholesky factor L (where A = LL^T).
// L is the lower-triangular n x n Cholesky factor from CholeskyDecompose.
// b is the right-hand side vector of length n. x is the solution (pre-allocated).
//
// The solution is computed in two steps:
//   1. Forward substitution: Ly = b
//   2. Back substitution:    L^T x = y
//
// Zero heap allocations (x is used as scratch for the intermediate y).
//
// Panics if slice lengths are inconsistent with n.
func CholeskySolve(L []float64, n int, b, x []float64) {
	nn := n * n
	if len(L) != nn {
		panic("linalg.CholeskySolve: len(L) != n*n")
	}
	if len(b) != n {
		panic("linalg.CholeskySolve: len(b) != n")
	}
	if len(x) != n {
		panic("linalg.CholeskySolve: len(x) != n")
	}

	// Forward substitution: Ly = b.
	for i := 0; i < n; i++ {
		sum := b[i]
		for j := 0; j < i; j++ {
			sum -= L[i*n+j] * x[j]
		}
		x[i] = sum / L[i*n+i]
	}

	// Back substitution: L^T x = y.
	for i := n - 1; i >= 0; i-- {
		sum := x[i]
		for j := i + 1; j < n; j++ {
			sum -= L[j*n+i] * x[j] // L^T[i][j] = L[j][i]
		}
		x[i] = sum / L[i*n+i]
	}
}

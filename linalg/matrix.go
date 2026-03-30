package linalg

// MatMul computes the matrix product C = A * B where:
//   - A is aRows x aCols (stored row-major in flat slice)
//   - B is aCols x bCols (stored row-major in flat slice)
//   - out (C) is aRows x bCols (pre-allocated by caller)
//
// Definition: C[i][j] = sum(A[i][k] * B[k][j]) for k = 0..aCols-1
// Zero heap allocations. The caller must ensure out has length aRows*bCols.
//
// Panics if slice lengths are inconsistent with the given dimensions.
func MatMul(A []float64, aRows, aCols int, B []float64, bCols int, out []float64) {
	if len(A) != aRows*aCols {
		panic("linalg.MatMul: len(A) != aRows*aCols")
	}
	if len(B) != aCols*bCols {
		panic("linalg.MatMul: len(B) != aCols*bCols")
	}
	if len(out) != aRows*bCols {
		panic("linalg.MatMul: len(out) != aRows*bCols")
	}

	for i := 0; i < aRows; i++ {
		for j := 0; j < bCols; j++ {
			var sum float64
			for k := 0; k < aCols; k++ {
				sum += A[i*aCols+k] * B[k*bCols+j]
			}
			out[i*bCols+j] = sum
		}
	}
}

// MatTranspose computes the transpose of matrix A (rows x cols) into out.
//   - A is rows x cols (stored row-major)
//   - out is cols x rows (pre-allocated by caller)
//
// Definition: out[j][i] = A[i][j]
// Zero heap allocations.
//
// Panics if slice lengths are inconsistent with the given dimensions.
func MatTranspose(A []float64, rows, cols int, out []float64) {
	if len(A) != rows*cols {
		panic("linalg.MatTranspose: len(A) != rows*cols")
	}
	if len(out) != rows*cols {
		panic("linalg.MatTranspose: len(out) != rows*cols")
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			out[j*rows+i] = A[i*cols+j]
		}
	}
}

// MatVecMul computes the matrix-vector product y = A * x where:
//   - A is rows x cols (stored row-major)
//   - x has length cols
//   - out (y) has length rows (pre-allocated by caller)
//
// Definition: y[i] = sum(A[i][j] * x[j]) for j = 0..cols-1
// Zero heap allocations.
//
// Panics if slice lengths are inconsistent with the given dimensions.
func MatVecMul(A []float64, rows, cols int, x []float64, out []float64) {
	if len(A) != rows*cols {
		panic("linalg.MatVecMul: len(A) != rows*cols")
	}
	if len(x) != cols {
		panic("linalg.MatVecMul: len(x) != cols")
	}
	if len(out) != rows {
		panic("linalg.MatVecMul: len(out) != rows")
	}

	for i := 0; i < rows; i++ {
		var sum float64
		for j := 0; j < cols; j++ {
			sum += A[i*cols+j] * x[j]
		}
		out[i] = sum
	}
}

// Identity writes the n x n identity matrix into out.
// out must have length n*n (pre-allocated by caller).
// All elements are zeroed first, then diagonal elements are set to 1.
// Zero heap allocations.
//
// Panics if len(out) != n*n.
func Identity(n int, out []float64) {
	if len(out) != n*n {
		panic("linalg.Identity: len(out) != n*n")
	}
	for i := range out {
		out[i] = 0
	}
	for i := 0; i < n; i++ {
		out[i*n+i] = 1
	}
}

// MatAdd computes element-wise matrix addition: out = A + B.
// A, B, and out are all rows x cols (stored row-major, pre-allocated).
// Zero heap allocations.
//
// Panics if slice lengths are inconsistent with the given dimensions.
func MatAdd(A, B []float64, rows, cols int, out []float64) {
	n := rows * cols
	if len(A) != n {
		panic("linalg.MatAdd: len(A) != rows*cols")
	}
	if len(B) != n {
		panic("linalg.MatAdd: len(B) != rows*cols")
	}
	if len(out) != n {
		panic("linalg.MatAdd: len(out) != rows*cols")
	}
	for i := 0; i < n; i++ {
		out[i] = A[i] + B[i]
	}
}

// MatScale computes scalar multiplication of a matrix: out = s * A.
// A and out are rows x cols (stored row-major, pre-allocated).
// Zero heap allocations.
//
// Panics if slice lengths are inconsistent with the given dimensions.
func MatScale(A []float64, rows, cols int, s float64, out []float64) {
	n := rows * cols
	if len(A) != n {
		panic("linalg.MatScale: len(A) != rows*cols")
	}
	if len(out) != n {
		panic("linalg.MatScale: len(out) != rows*cols")
	}
	for i := 0; i < n; i++ {
		out[i] = A[i] * s
	}
}

// MatSub computes element-wise matrix subtraction: out = A - B.
// A, B, and out are all rows x cols (stored row-major, pre-allocated).
// Zero heap allocations.
//
// Panics if slice lengths are inconsistent with the given dimensions.
func MatSub(A, B []float64, rows, cols int, out []float64) {
	n := rows * cols
	if len(A) != n {
		panic("linalg.MatSub: len(A) != rows*cols")
	}
	if len(B) != n {
		panic("linalg.MatSub: len(B) != rows*cols")
	}
	if len(out) != n {
		panic("linalg.MatSub: len(out) != rows*cols")
	}
	for i := 0; i < n; i++ {
		out[i] = A[i] - B[i]
	}
}

// Trace computes the trace of an n x n matrix A (sum of diagonal elements).
// A is n x n row-major (stored as a flat slice of length n*n).
//
// Definition: tr(A) = sum(A[i][i]) for i = 0..n-1
// Valid input range: n >= 0
// Precision: exact (accumulated float64 summation error for large n)
// Reference: fundamental matrix operation; see Golub & Van Loan,
// "Matrix Computations", Ch. 1
//
// Panics if len(A) != n*n.
func Trace(A []float64, n int) float64 {
	if len(A) != n*n {
		panic("linalg.Trace: len(A) != n*n")
	}
	var sum float64
	for i := 0; i < n; i++ {
		sum += A[i*n+i]
	}
	return sum
}

// CrossProduct computes the cross product of two 3D vectors: out = a x b.
// a, b, and out must each have length 3. The caller must pre-allocate out.
// Zero heap allocations.
//
// Definition:
//
//	out[0] = a[1]*b[2] - a[2]*b[1]
//	out[1] = a[2]*b[0] - a[0]*b[2]
//	out[2] = a[0]*b[1] - a[1]*b[0]
//
// The result is perpendicular to both a and b (right-hand rule).
// Precision: exact for IEEE 754 float64.
// Reference: fundamental vector algebra; see Arfken, Weber & Harris,
// "Mathematical Methods for Physicists", Ch. 1
//
// Panics if any slice does not have length 3.
func CrossProduct(a, b, out []float64) {
	if len(a) != 3 || len(b) != 3 || len(out) != 3 {
		panic("linalg.CrossProduct: all slices must have length 3")
	}
	out[0] = a[1]*b[2] - a[2]*b[1]
	out[1] = a[2]*b[0] - a[0]*b[2]
	out[2] = a[0]*b[1] - a[1]*b[0]
}

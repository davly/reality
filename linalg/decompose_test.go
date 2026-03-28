package linalg

import (
	"math"
	"testing"

	"github.com/davly/reality/testutil"
)

// ═══════════════════════════════════════════════════════════════════════════
// Golden-file tests — LU decomposition
// ═══════════════════════════════════════════════════════════════════════════

func TestGolden_Determinant(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/linalg/determinant.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			A := testutil.InputFloat64Slice(t, tc, "A")
			n := int(testutil.InputFloat64(t, tc, "n"))
			got := Determinant(A, n)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_LUDecompose(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/linalg/lu_decompose.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			A := testutil.InputFloat64Slice(t, tc, "A")
			n := int(testutil.InputFloat64(t, tc, "n"))

			L := make([]float64, n*n)
			U := make([]float64, n*n)
			perm := make([]int, n)

			ok := LUDecompose(A, n, L, U, perm)
			if !ok {
				t.Fatal("LUDecompose returned false for non-singular matrix")
			}

			// Verify PA = LU by computing L*U and comparing with permuted A.
			LU := make([]float64, n*n)
			MatMul(L, n, n, U, n, LU)

			PA := make([]float64, n*n)
			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					PA[i*n+j] = A[perm[i]*n+j]
				}
			}

			for i := 0; i < n*n; i++ {
				if math.Abs(LU[i]-PA[i]) > tc.Tolerance {
					t.Errorf("PA != LU at index %d: PA=%v, LU=%v (tol %v)",
						i, PA[i], LU[i], tc.Tolerance)
				}
			}
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — LU decomposition
// ═══════════════════════════════════════════════════════════════════════════

func TestLUDecompose_2x2_Known(t *testing.T) {
	// A = [[4,3],[6,3]]
	A := []float64{4, 3, 6, 3}
	L := make([]float64, 4)
	U := make([]float64, 4)
	perm := make([]int, 2)

	ok := LUDecompose(A, 2, L, U, perm)
	if !ok {
		t.Fatal("LUDecompose returned false")
	}

	// Verify PA = LU.
	verifyLU(t, A, L, U, perm, 2, 1e-12)
}

func TestLUDecompose_3x3_Known(t *testing.T) {
	A := []float64{2, 1, 1, 4, 3, 3, 8, 7, 9}
	L := make([]float64, 9)
	U := make([]float64, 9)
	perm := make([]int, 3)

	ok := LUDecompose(A, 3, L, U, perm)
	if !ok {
		t.Fatal("LUDecompose returned false")
	}

	verifyLU(t, A, L, U, perm, 3, 1e-12)
}

func TestLUDecompose_Singular(t *testing.T) {
	// Singular matrix: row 3 = 2*row 1.
	A := []float64{1, 2, 3, 4, 5, 6, 2, 4, 6}
	L := make([]float64, 9)
	U := make([]float64, 9)
	perm := make([]int, 3)

	ok := LUDecompose(A, 3, L, U, perm)
	if ok {
		t.Error("LUDecompose should return false for singular matrix")
	}
}

func TestLUDecompose_1x1(t *testing.T) {
	A := []float64{42}
	L := make([]float64, 1)
	U := make([]float64, 1)
	perm := make([]int, 1)

	ok := LUDecompose(A, 1, L, U, perm)
	if !ok {
		t.Fatal("LUDecompose returned false")
	}
	assertClose(t, "L[0]", L[0], 1.0, 1e-15)
	assertClose(t, "U[0]", U[0], 42.0, 1e-15)
}

func TestLUDecompose_Identity(t *testing.T) {
	I := make([]float64, 9)
	Identity(3, I)
	L := make([]float64, 9)
	U := make([]float64, 9)
	perm := make([]int, 3)

	ok := LUDecompose(I, 3, L, U, perm)
	if !ok {
		t.Fatal("LUDecompose returned false for identity")
	}

	verifyLU(t, I, L, U, perm, 3, 1e-15)
}

func TestLUDecompose_NeedsPivoting(t *testing.T) {
	// First pivot is zero, must swap rows.
	A := []float64{0, 1, 1, 0}
	L := make([]float64, 4)
	U := make([]float64, 4)
	perm := make([]int, 2)

	ok := LUDecompose(A, 2, L, U, perm)
	if !ok {
		t.Fatal("LUDecompose returned false")
	}

	verifyLU(t, A, L, U, perm, 2, 1e-15)
}

func TestLUDecompose_4x4(t *testing.T) {
	A := []float64{
		2, 1, 1, 0,
		4, 3, 3, 1,
		8, 7, 9, 5,
		6, 7, 9, 8,
	}
	L := make([]float64, 16)
	U := make([]float64, 16)
	perm := make([]int, 4)

	ok := LUDecompose(A, 4, L, U, perm)
	if !ok {
		t.Fatal("LUDecompose returned false")
	}

	verifyLU(t, A, L, U, perm, 4, 1e-12)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — LU solve
// ═══════════════════════════════════════════════════════════════════════════

func TestLUSolve_2x2_Known(t *testing.T) {
	// A = [[2,1],[5,3]], b = [4,7] -> x = [5,-6]
	A := []float64{2, 1, 5, 3}
	L := make([]float64, 4)
	U := make([]float64, 4)
	perm := make([]int, 2)
	LUDecompose(A, 2, L, U, perm)

	b := []float64{4, 7}
	x := make([]float64, 2)
	LUSolve(L, U, 2, perm, b, x)

	assertClose(t, "x[0]", x[0], 5.0, 1e-12)
	assertClose(t, "x[1]", x[1], -6.0, 1e-12)
}

func TestLUSolve_3x3_Known(t *testing.T) {
	// A = [[1,2,3],[4,5,6],[7,8,0]], b = [14,32,23] -> x = [1,2,3]
	A := []float64{1, 2, 3, 4, 5, 6, 7, 8, 0}
	L := make([]float64, 9)
	U := make([]float64, 9)
	perm := make([]int, 3)
	ok := LUDecompose(A, 3, L, U, perm)
	if !ok {
		t.Fatal("LUDecompose failed")
	}

	b := []float64{14, 32, 23}
	x := make([]float64, 3)
	LUSolve(L, U, 3, perm, b, x)

	assertClose(t, "x[0]", x[0], 1.0, 1e-10)
	assertClose(t, "x[1]", x[1], 2.0, 1e-10)
	assertClose(t, "x[2]", x[2], 3.0, 1e-10)
}

func TestLUSolve_Identity(t *testing.T) {
	I := make([]float64, 9)
	Identity(3, I)
	L := make([]float64, 9)
	U := make([]float64, 9)
	perm := make([]int, 3)
	LUDecompose(I, 3, L, U, perm)

	b := []float64{7, 8, 9}
	x := make([]float64, 3)
	LUSolve(L, U, 3, perm, b, x)

	assertSliceClose(t, "lu-solve-identity", x, b, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Inverse
// ═══════════════════════════════════════════════════════════════════════════

func TestInverse_2x2_Known(t *testing.T) {
	// A = [[4,7],[2,6]], det = 10, inv = [[0.6,-0.7],[-0.2,0.4]]
	A := []float64{4, 7, 2, 6}
	inv := make([]float64, 4)

	ok := Inverse(A, 2, inv)
	if !ok {
		t.Fatal("Inverse returned false")
	}

	assertClose(t, "inv[0][0]", inv[0], 0.6, 1e-12)
	assertClose(t, "inv[0][1]", inv[1], -0.7, 1e-12)
	assertClose(t, "inv[1][0]", inv[2], -0.2, 1e-12)
	assertClose(t, "inv[1][1]", inv[3], 0.4, 1e-12)
}

func TestInverse_3x3_Known(t *testing.T) {
	A := []float64{1, 2, 3, 0, 1, 4, 5, 6, 0}
	inv := make([]float64, 9)

	ok := Inverse(A, 3, inv)
	if !ok {
		t.Fatal("Inverse returned false")
	}

	// Verify A * inv = I.
	product := make([]float64, 9)
	MatMul(A, 3, 3, inv, 3, product)

	I := make([]float64, 9)
	Identity(3, I)
	assertSliceClose(t, "A*inv=I", product, I, 1e-10)
}

func TestInverse_Identity(t *testing.T) {
	I := make([]float64, 9)
	Identity(3, I)
	inv := make([]float64, 9)

	ok := Inverse(I, 3, inv)
	if !ok {
		t.Fatal("Inverse returned false for identity")
	}

	assertSliceClose(t, "inv(I)=I", inv, I, 1e-15)
}

func TestInverse_Singular(t *testing.T) {
	A := []float64{1, 2, 3, 4, 5, 6, 2, 4, 6}
	inv := make([]float64, 9)

	ok := Inverse(A, 3, inv)
	if ok {
		t.Error("Inverse should return false for singular matrix")
	}
}

func TestInverse_1x1(t *testing.T) {
	A := []float64{4.0}
	inv := make([]float64, 1)

	ok := Inverse(A, 1, inv)
	if !ok {
		t.Fatal("Inverse returned false")
	}
	assertClose(t, "inv(4)=0.25", inv[0], 0.25, 1e-15)
}

func TestInverse_4x4_Roundtrip(t *testing.T) {
	A := []float64{
		2, 1, 1, 0,
		4, 3, 3, 1,
		8, 7, 9, 5,
		6, 7, 9, 8,
	}
	inv := make([]float64, 16)

	ok := Inverse(A, 4, inv)
	if !ok {
		t.Fatal("Inverse returned false")
	}

	product := make([]float64, 16)
	MatMul(A, 4, 4, inv, 4, product)

	I := make([]float64, 16)
	Identity(4, I)
	assertSliceClose(t, "A*inv=I(4x4)", product, I, 1e-10)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Determinant
// ═══════════════════════════════════════════════════════════════════════════

func TestDeterminant_2x2_AdBc(t *testing.T) {
	// det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
	A := []float64{1, 2, 3, 4}
	got := Determinant(A, 2)
	assertClose(t, "det-2x2", got, -2.0, 1e-12)
}

func TestDeterminant_3x3_Known(t *testing.T) {
	// det([[6,1,1],[4,-2,5],[2,8,7]]) = -306
	A := []float64{6, 1, 1, 4, -2, 5, 2, 8, 7}
	got := Determinant(A, 3)
	assertClose(t, "det-3x3", got, -306.0, 1e-10)
}

func TestDeterminant_Identity(t *testing.T) {
	I := make([]float64, 16)
	Identity(4, I)
	got := Determinant(I, 4)
	assertClose(t, "det-identity", got, 1.0, 1e-15)
}

func TestDeterminant_Singular(t *testing.T) {
	// Row 3 = row 1 + row 2.
	A := []float64{1, 2, 3, 4, 5, 6, 5, 7, 9}
	got := Determinant(A, 3)
	assertClose(t, "det-singular", got, 0.0, 1e-10)
}

func TestDeterminant_1x1(t *testing.T) {
	got := Determinant([]float64{-3.5}, 1)
	assertClose(t, "det-1x1", got, -3.5, 1e-15)
}

func TestDeterminant_Diagonal(t *testing.T) {
	// det(diag(2,3,5)) = 30
	A := []float64{2, 0, 0, 0, 3, 0, 0, 0, 5}
	got := Determinant(A, 3)
	assertClose(t, "det-diagonal", got, 30.0, 1e-12)
}

func TestDeterminant_NegativeSwap(t *testing.T) {
	// Permutation matrix [[0,1],[1,0]]: det = -1
	A := []float64{0, 1, 1, 0}
	got := Determinant(A, 2)
	assertClose(t, "det-perm", got, -1.0, 1e-15)
}

func TestDeterminant_ScaledIdentity(t *testing.T) {
	// det(3*I_3) = 27
	A := []float64{3, 0, 0, 0, 3, 0, 0, 0, 3}
	got := Determinant(A, 3)
	assertClose(t, "det-3I", got, 27.0, 1e-12)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Cholesky decomposition
// ═══════════════════════════════════════════════════════════════════════════

func TestCholeskyDecompose_2x2_SPD(t *testing.T) {
	// A = [[4,2],[2,3]] is SPD.
	A := []float64{4, 2, 2, 3}
	L := make([]float64, 4)

	ok := CholeskyDecompose(A, 2, L)
	if !ok {
		t.Fatal("CholeskyDecompose returned false for SPD matrix")
	}

	// Verify L*L^T = A.
	verifyCholesky(t, A, L, 2, 1e-12)
}

func TestCholeskyDecompose_3x3_SPD(t *testing.T) {
	// A = [[25,15,-5],[15,18,0],[-5,0,11]]
	A := []float64{25, 15, -5, 15, 18, 0, -5, 0, 11}
	L := make([]float64, 9)

	ok := CholeskyDecompose(A, 3, L)
	if !ok {
		t.Fatal("CholeskyDecompose returned false for SPD matrix")
	}

	verifyCholesky(t, A, L, 3, 1e-12)
}

func TestCholeskyDecompose_Identity(t *testing.T) {
	I := make([]float64, 9)
	Identity(3, I)
	L := make([]float64, 9)

	ok := CholeskyDecompose(I, 3, L)
	if !ok {
		t.Fatal("CholeskyDecompose returned false for identity")
	}

	// L should also be identity.
	assertSliceClose(t, "chol(I)=I", L, I, 1e-15)
}

func TestCholeskyDecompose_NotSPD(t *testing.T) {
	// Not positive definite: eigenvalues include negative.
	A := []float64{1, 2, 2, 1}
	L := make([]float64, 4)

	ok := CholeskyDecompose(A, 2, L)
	if ok {
		t.Error("CholeskyDecompose should return false for non-SPD matrix")
	}
}

func TestCholeskyDecompose_1x1(t *testing.T) {
	A := []float64{9.0}
	L := make([]float64, 1)

	ok := CholeskyDecompose(A, 1, L)
	if !ok {
		t.Fatal("CholeskyDecompose returned false")
	}
	assertClose(t, "chol(9)=3", L[0], 3.0, 1e-15)
}

func TestCholeskyDecompose_4x4_SPD(t *testing.T) {
	// Build SPD matrix: A = B^T * B where B is non-singular.
	// B = [[1,1,1,1],[0,1,1,1],[0,0,1,1],[0,0,0,1]]
	// A = B^T B:
	A := []float64{
		1, 1, 1, 1,
		1, 2, 2, 2,
		1, 2, 3, 3,
		1, 2, 3, 4,
	}
	L := make([]float64, 16)

	ok := CholeskyDecompose(A, 4, L)
	if !ok {
		t.Fatal("CholeskyDecompose returned false for SPD matrix")
	}

	verifyCholesky(t, A, L, 4, 1e-12)
}

func TestCholeskyDecompose_ZeroDiag(t *testing.T) {
	// Zero on diagonal = not positive definite.
	A := []float64{0, 0, 0, 1}
	L := make([]float64, 4)

	ok := CholeskyDecompose(A, 2, L)
	if ok {
		t.Error("CholeskyDecompose should return false for zero diagonal")
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Cholesky solve
// ═══════════════════════════════════════════════════════════════════════════

func TestCholeskySolve_2x2(t *testing.T) {
	A := []float64{4, 2, 2, 3}
	L := make([]float64, 4)
	CholeskyDecompose(A, 2, L)

	b := []float64{8, 7}
	x := make([]float64, 2)
	CholeskySolve(L, 2, b, x)

	// Verify Ax = b.
	check := make([]float64, 2)
	MatVecMul(A, 2, 2, x, check)
	assertSliceClose(t, "chol-solve-2x2", check, b, 1e-12)
}

func TestCholeskySolve_3x3(t *testing.T) {
	A := []float64{25, 15, -5, 15, 18, 0, -5, 0, 11}
	L := make([]float64, 9)
	CholeskyDecompose(A, 3, L)

	b := []float64{35, 33, 6}
	x := make([]float64, 3)
	CholeskySolve(L, 3, b, x)

	check := make([]float64, 3)
	MatVecMul(A, 3, 3, x, check)
	assertSliceClose(t, "chol-solve-3x3", check, b, 1e-10)
}

func TestCholeskySolve_Identity(t *testing.T) {
	I := make([]float64, 9)
	Identity(3, I)
	L := make([]float64, 9)
	CholeskyDecompose(I, 3, L)

	b := []float64{1, 2, 3}
	x := make([]float64, 3)
	CholeskySolve(L, 3, b, x)

	assertSliceClose(t, "chol-solve-identity", x, b, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — QR Algorithm (eigenvalues)
// ═══════════════════════════════════════════════════════════════════════════

func TestQRAlgorithm_Diagonal(t *testing.T) {
	// Eigenvalues of a diagonal matrix are the diagonal entries.
	A := []float64{5, 0, 0, 0, 3, 0, 0, 0, 1}
	eig := make([]float64, 3)

	QRAlgorithm(A, 3, eig, 100)

	// Should be sorted descending: 5, 3, 1.
	assertClose(t, "eig[0]", eig[0], 5.0, 1e-10)
	assertClose(t, "eig[1]", eig[1], 3.0, 1e-10)
	assertClose(t, "eig[2]", eig[2], 1.0, 1e-10)
}

func TestQRAlgorithm_Symmetric_2x2(t *testing.T) {
	// A = [[2,1],[1,2]], eigenvalues are 3, 1.
	A := []float64{2, 1, 1, 2}
	eig := make([]float64, 2)

	QRAlgorithm(A, 2, eig, 100)

	assertClose(t, "eig[0]", eig[0], 3.0, 1e-10)
	assertClose(t, "eig[1]", eig[1], 1.0, 1e-10)
}

func TestQRAlgorithm_Symmetric_3x3(t *testing.T) {
	// A = [[2,-1,0],[-1,2,-1],[0,-1,2]]
	// Eigenvalues: 2+sqrt(2), 2, 2-sqrt(2)
	A := []float64{2, -1, 0, -1, 2, -1, 0, -1, 2}
	eig := make([]float64, 3)

	QRAlgorithm(A, 3, eig, 200)

	sqrt2 := math.Sqrt(2)
	assertClose(t, "eig[0]", eig[0], 2+sqrt2, 1e-10)
	assertClose(t, "eig[1]", eig[1], 2.0, 1e-10)
	assertClose(t, "eig[2]", eig[2], 2-sqrt2, 1e-10)
}

func TestQRAlgorithm_1x1(t *testing.T) {
	A := []float64{42.0}
	eig := make([]float64, 1)

	iters := QRAlgorithm(A, 1, eig, 100)

	assertClose(t, "eig[0]", eig[0], 42.0, 1e-15)
	if iters != 0 {
		t.Errorf("expected 0 iterations for 1x1, got %d", iters)
	}
}

func TestQRAlgorithm_Identity(t *testing.T) {
	I := make([]float64, 16)
	Identity(4, I)
	eig := make([]float64, 4)

	QRAlgorithm(I, 4, eig, 100)

	for i := 0; i < 4; i++ {
		assertClose(t, "eig-I", eig[i], 1.0, 1e-10)
	}
}

func TestQRAlgorithm_KnownSPD(t *testing.T) {
	// A = [[4,2],[2,3]], char poly: λ²-7λ+8, discriminant=17
	// eigenvalues: (7+sqrt(17))/2 and (7-sqrt(17))/2 ≈ 5.562, 1.438
	A := []float64{4, 2, 2, 3}
	eig := make([]float64, 2)

	QRAlgorithm(A, 2, eig, 100)

	sqrt17 := math.Sqrt(17)
	assertClose(t, "eig[0]", eig[0], (7+sqrt17)/2, 1e-10)
	assertClose(t, "eig[1]", eig[1], (7-sqrt17)/2, 1e-10)
}

func TestQRAlgorithm_ZeroMatrix(t *testing.T) {
	A := []float64{0, 0, 0, 0}
	eig := make([]float64, 2)

	QRAlgorithm(A, 2, eig, 100)

	assertClose(t, "eig[0]", eig[0], 0.0, 1e-10)
	assertClose(t, "eig[1]", eig[1], 0.0, 1e-10)
}

func TestQRAlgorithm_NegativeEigenvalues(t *testing.T) {
	// A = [[-2,0],[0,-3]], eigenvalues: -2, -3
	A := []float64{-2, 0, 0, -3}
	eig := make([]float64, 2)

	QRAlgorithm(A, 2, eig, 100)

	assertClose(t, "eig[0]", eig[0], -2.0, 1e-10)
	assertClose(t, "eig[1]", eig[1], -3.0, 1e-10)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — PCA
// ═══════════════════════════════════════════════════════════════════════════

func TestPCA_2D_KnownAxis(t *testing.T) {
	// Data that varies primarily along x-axis.
	// Points: (1,0), (2,0), (3,0), (4,0), (5,0) plus small noise in y.
	data := []float64{
		1, 0.1,
		2, -0.1,
		3, 0.0,
		4, 0.1,
		5, -0.1,
	}

	components := make([]float64, 2) // 1 component x 2 features
	explained := make([]float64, 1)

	cumVar := PCA(data, 5, 2, 1, components, explained)

	// First component should be roughly along x-axis: [1, 0] or [-1, 0].
	// Check that the absolute value of the x-component dominates.
	if math.Abs(components[0]) < 0.95 {
		t.Errorf("first PC x-component should dominate, got %v", components[0])
	}
	if math.Abs(components[1]) > 0.1 {
		t.Errorf("first PC y-component should be small, got %v", components[1])
	}

	// Should explain nearly all variance.
	if cumVar < 0.95 {
		t.Errorf("cumulative variance should be > 0.95, got %v", cumVar)
	}
	if explained[0] < 0.95 {
		t.Errorf("first component explained should be > 0.95, got %v", explained[0])
	}
}

func TestPCA_2D_TwoComponents(t *testing.T) {
	// With 2 features and 2 components, total explained should be ~1.0.
	data := []float64{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
		9, 10,
	}

	components := make([]float64, 4) // 2 components x 2 features
	explained := make([]float64, 2)

	cumVar := PCA(data, 5, 2, 2, components, explained)

	assertClose(t, "total-variance", cumVar, 1.0, 1e-6)
}

func TestPCA_3D_ReduceTo2(t *testing.T) {
	// 3D data with clear 2D structure (third dimension is noise).
	data := []float64{
		1, 1, 0.01,
		2, 2, -0.01,
		3, 3, 0.02,
		4, 4, -0.02,
		5, 5, 0.01,
		6, 6, -0.01,
	}

	components := make([]float64, 6) // 2 components x 3 features
	explained := make([]float64, 2)

	cumVar := PCA(data, 6, 3, 2, components, explained)

	// First two components should explain nearly all variance.
	if cumVar < 0.99 {
		t.Errorf("2 components should explain > 99%% variance, got %v", cumVar)
	}
}

func TestPCA_ExplainedSumsToOne(t *testing.T) {
	// All components should sum to 1.0 for full decomposition.
	data := []float64{
		2, 3, 5,
		4, 1, 7,
		6, 8, 2,
		1, 9, 4,
		3, 5, 6,
	}

	components := make([]float64, 9) // 3 components x 3 features
	explained := make([]float64, 3)

	cumVar := PCA(data, 5, 3, 3, components, explained)

	assertClose(t, "full-pca-cum-var", cumVar, 1.0, 1e-6)
}

func TestPCA_ComponentsOrthogonal(t *testing.T) {
	data := []float64{
		2, 3, 5,
		4, 1, 7,
		6, 8, 2,
		1, 9, 4,
		3, 5, 6,
	}

	components := make([]float64, 9)
	explained := make([]float64, 3)

	PCA(data, 5, 3, 3, components, explained)

	// Check orthogonality: dot product of different components should be ~0.
	for i := 0; i < 3; i++ {
		for j := i + 1; j < 3; j++ {
			dot := DotProduct(components[i*3:(i+1)*3], components[j*3:(j+1)*3])
			if math.Abs(dot) > 1e-6 {
				t.Errorf("components %d and %d not orthogonal: dot=%v", i, j, dot)
			}
		}
	}
}

func TestPCA_ComponentsUnit(t *testing.T) {
	data := []float64{
		2, 3,
		4, 1,
		6, 8,
		1, 9,
		3, 5,
	}

	components := make([]float64, 4) // 2x2
	explained := make([]float64, 2)

	PCA(data, 5, 2, 2, components, explained)

	for i := 0; i < 2; i++ {
		norm := L2Norm(components[i*2 : (i+1)*2])
		assertClose(t, "unit-norm", norm, 1.0, 1e-6)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Panics
// ═══════════════════════════════════════════════════════════════════════════

func TestLUDecompose_PanicA(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for wrong A length")
		}
	}()
	LUDecompose([]float64{1, 2}, 2, make([]float64, 4), make([]float64, 4), make([]int, 2))
}

func TestLUDecompose_PanicPerm(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for wrong perm length")
		}
	}()
	LUDecompose([]float64{1, 2, 3, 4}, 2, make([]float64, 4), make([]float64, 4), make([]int, 3))
}

func TestInverse_PanicOut(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for wrong out length")
		}
	}()
	Inverse([]float64{1, 2, 3, 4}, 2, make([]float64, 3))
}

func TestDeterminant_Panic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for wrong A length")
		}
	}()
	Determinant([]float64{1, 2, 3}, 2)
}

func TestCholeskyDecompose_PanicL(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for wrong L length")
		}
	}()
	CholeskyDecompose([]float64{1, 0, 0, 1}, 2, make([]float64, 3))
}

func TestQRAlgorithm_PanicEig(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for wrong eigenvalues length")
		}
	}()
	QRAlgorithm([]float64{1, 0, 0, 1}, 2, make([]float64, 3), 100)
}

func TestPCA_PanicComponents(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for wrong components length")
		}
	}()
	PCA([]float64{1, 2, 3, 4}, 2, 2, 1, make([]float64, 3), make([]float64, 1))
}

func TestPCA_PanicTooManyComponents(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for nComponents > nFeatures")
		}
	}()
	PCA([]float64{1, 2, 3, 4}, 2, 2, 3, make([]float64, 6), make([]float64, 3))
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

// verifyLU checks that PA = LU for the given decomposition.
func verifyLU(t *testing.T, A, L, U []float64, perm []int, n int, tol float64) {
	t.Helper()

	LU := make([]float64, n*n)
	MatMul(L, n, n, U, n, LU)

	PA := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			PA[i*n+j] = A[perm[i]*n+j]
		}
	}

	for i := 0; i < n*n; i++ {
		if math.Abs(LU[i]-PA[i]) > tol {
			t.Errorf("PA != LU at [%d][%d]: PA=%v, LU=%v (tol %v)",
				i/n, i%n, PA[i], LU[i], tol)
		}
	}
}

// verifyCholesky checks that L*L^T = A for the given Cholesky factor.
func verifyCholesky(t *testing.T, A, L []float64, n int, tol float64) {
	t.Helper()

	LT := make([]float64, n*n)
	MatTranspose(L, n, n, LT)

	LLT := make([]float64, n*n)
	MatMul(L, n, n, LT, n, LLT)

	for i := 0; i < n*n; i++ {
		if math.Abs(LLT[i]-A[i]) > tol {
			t.Errorf("L*L^T != A at [%d][%d]: LLT=%v, A=%v (tol %v)",
				i/n, i%n, LLT[i], A[i], tol)
		}
	}
}

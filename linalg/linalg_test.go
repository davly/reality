package linalg

import (
	"math"
	"testing"

	"github.com/davly/reality/testutil"
)

// ═══════════════════════════════════════════════════════════════════════════
// Golden-file tests — shared test vectors across Go, Python, C++, C#
// ═══════════════════════════════════════════════════════════════════════════

func TestGolden_CosineSimilarity(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/linalg/cosine_similarity.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			a := testutil.InputFloat64Slice(t, tc, "a")
			b := testutil.InputFloat64Slice(t, tc, "b")
			got := CosineSimilarity(a, b)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_MatMul(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/linalg/matrix_multiply.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			A := testutil.InputFloat64Slice(t, tc, "A")
			aRows := int(testutil.InputFloat64(t, tc, "aRows"))
			aCols := int(testutil.InputFloat64(t, tc, "aCols"))
			B := testutil.InputFloat64Slice(t, tc, "B")
			bCols := int(testutil.InputFloat64(t, tc, "bCols"))
			out := make([]float64, aRows*bCols)
			MatMul(A, aRows, aCols, B, bCols, out)
			testutil.AssertFloat64Slice(t, tc, out)
		})
	}
}

func TestGolden_PearsonCorrelation(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/linalg/pearson_correlation.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			x := testutil.InputFloat64Slice(t, tc, "x")
			y := testutil.InputFloat64Slice(t, tc, "y")
			got := PearsonCorrelation(x, y)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — vector operations
// ═══════════════════════════════════════════════════════════════════════════

func TestCosineSimilarity_Orthogonal(t *testing.T) {
	got := CosineSimilarity([]float64{1, 0}, []float64{0, 1})
	assertClose(t, "orthogonal", got, 0.0, 1e-15)
}

func TestCosineSimilarity_Parallel(t *testing.T) {
	got := CosineSimilarity([]float64{3, 4}, []float64{6, 8})
	assertClose(t, "parallel", got, 1.0, 1e-15)
}

func TestCosineSimilarity_AntiParallel(t *testing.T) {
	got := CosineSimilarity([]float64{1, 2}, []float64{-2, -4})
	assertClose(t, "anti-parallel", got, -1.0, 1e-15)
}

func TestCosineSimilarity_ZeroVector(t *testing.T) {
	got := CosineSimilarity([]float64{0, 0}, []float64{1, 2})
	assertClose(t, "zero-vec", got, 0.0, 0)
}

func TestCosineSimilarity_LengthMismatch(t *testing.T) {
	got := CosineSimilarity([]float64{1, 2}, []float64{1, 2, 3})
	assertClose(t, "length-mismatch", got, 0.0, 0)
}

func TestCosineSimilarity_EmptyVectors(t *testing.T) {
	got := CosineSimilarity([]float64{}, []float64{})
	assertClose(t, "empty", got, 0.0, 0)
}

func TestDotProduct_KnownValues(t *testing.T) {
	got := DotProduct([]float64{1, 2, 3}, []float64{4, 5, 6})
	assertClose(t, "dot(1,2,3).(4,5,6)", got, 32.0, 1e-15)
}

func TestDotProduct_Orthogonal(t *testing.T) {
	got := DotProduct([]float64{1, 0}, []float64{0, 1})
	assertClose(t, "dot-orthogonal", got, 0.0, 1e-15)
}

func TestDotProduct_Empty(t *testing.T) {
	got := DotProduct([]float64{}, []float64{})
	assertClose(t, "dot-empty", got, 0.0, 0)
}

func TestDotProduct_LengthMismatch(t *testing.T) {
	got := DotProduct([]float64{1}, []float64{1, 2})
	assertClose(t, "dot-mismatch", got, 0.0, 0)
}

func TestL2Norm_KnownValue(t *testing.T) {
	got := L2Norm([]float64{3, 4})
	assertClose(t, "L2(3,4)", got, 5.0, 1e-15)
}

func TestL2Norm_Zero(t *testing.T) {
	got := L2Norm([]float64{0, 0, 0})
	assertClose(t, "L2-zero", got, 0.0, 0)
}

func TestL2Norm_Empty(t *testing.T) {
	got := L2Norm([]float64{})
	assertClose(t, "L2-empty", got, 0.0, 0)
}

func TestL1Norm_KnownValue(t *testing.T) {
	got := L1Norm([]float64{-1, 2, -3, 4})
	assertClose(t, "L1(-1,2,-3,4)", got, 10.0, 1e-15)
}

func TestLInfNorm_KnownValue(t *testing.T) {
	got := LInfNorm([]float64{-5, 3, 1})
	assertClose(t, "Linf(-5,3,1)", got, 5.0, 1e-15)
}

func TestLInfNorm_Empty(t *testing.T) {
	got := LInfNorm([]float64{})
	assertClose(t, "Linf-empty", got, 0.0, 0)
}

func TestL2Normalize_UnitVector(t *testing.T) {
	v := []float64{3, 4}
	ok := L2Normalize(v)
	if !ok {
		t.Fatal("expected ok=true")
	}
	assertClose(t, "L2Norm-x", v[0], 0.6, 1e-15)
	assertClose(t, "L2Norm-y", v[1], 0.8, 1e-15)
}

func TestL2Normalize_ZeroVector(t *testing.T) {
	v := []float64{0, 0, 0}
	ok := L2Normalize(v)
	if ok {
		t.Fatal("expected ok=false for zero vector")
	}
}

func TestClamp_InRange(t *testing.T) {
	assertClose(t, "clamp-in-range", Clamp(5, 0, 10), 5, 0)
}

func TestClamp_BelowMin(t *testing.T) {
	assertClose(t, "clamp-below", Clamp(-3, 0, 10), 0, 0)
}

func TestClamp_AboveMax(t *testing.T) {
	assertClose(t, "clamp-above", Clamp(15, 0, 10), 10, 0)
}

func TestEncodingDistance_KnownValue(t *testing.T) {
	// Distance between (0,0) and (3,4) = 5, normalized by sqrt(2)
	got := EncodingDistance([]float64{0, 0}, []float64{3, 4})
	expected := 5.0 / math.Sqrt(2.0)
	assertClose(t, "enc-dist", got, expected, 1e-15)
}

func TestEncodingDistance_Identical(t *testing.T) {
	got := EncodingDistance([]float64{1, 2, 3}, []float64{1, 2, 3})
	assertClose(t, "enc-dist-identical", got, 0.0, 1e-15)
}

func TestEncodingDistance_LengthMismatch(t *testing.T) {
	got := EncodingDistance([]float64{1}, []float64{1, 2})
	assertClose(t, "enc-dist-mismatch", got, 0.0, 0)
}

func TestDimensionWeightedDistance_EqualWeights(t *testing.T) {
	// With equal weights, should equal unweighted normalized distance
	got := DimensionWeightedDistance(
		[]float64{0, 0}, []float64{3, 4},
		[]float64{1, 1},
	)
	expected := math.Sqrt((9 + 16) / 2.0) // sqrt(25/2)
	assertClose(t, "weighted-equal", got, expected, 1e-15)
}

func TestDimensionWeightedDistance_ZeroWeights(t *testing.T) {
	got := DimensionWeightedDistance(
		[]float64{100, 200}, []float64{1, 2},
		[]float64{0, 0},
	)
	assertClose(t, "weighted-zero", got, 0.0, 0)
}

func TestDimensionWeightedDistance_MixedWeights(t *testing.T) {
	got := DimensionWeightedDistance(
		[]float64{0, 0}, []float64{3, 4},
		[]float64{2, -1}, // second dimension weight skipped (negative)
	)
	// Only first dimension: sqrt(2*9 / 2) = 3
	assertClose(t, "weighted-mixed", got, 3.0, 1e-15)
}

func TestVectorAdd(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	out := make([]float64, 3)
	VectorAdd(a, b, out)
	assertSliceClose(t, "vec-add", out, []float64{5, 7, 9}, 1e-15)
}

func TestVectorSub(t *testing.T) {
	a := []float64{10, 20, 30}
	b := []float64{1, 2, 3}
	out := make([]float64, 3)
	VectorSub(a, b, out)
	assertSliceClose(t, "vec-sub", out, []float64{9, 18, 27}, 1e-15)
}

func TestVectorScale(t *testing.T) {
	a := []float64{2, 4, 6}
	out := make([]float64, 3)
	VectorScale(a, 0.5, out)
	assertSliceClose(t, "vec-scale", out, []float64{1, 2, 3}, 1e-15)
}

func TestVectorScale_Zero(t *testing.T) {
	a := []float64{2, 4, 6}
	out := make([]float64, 3)
	VectorScale(a, 0, out)
	assertSliceClose(t, "vec-scale-zero", out, []float64{0, 0, 0}, 0)
}

func TestStructuralOverlap_KnownValue(t *testing.T) {
	assertClose(t, "overlap-3/5", StructuralOverlap(3, 5), 0.6, 1e-15)
}

func TestStructuralOverlap_Full(t *testing.T) {
	assertClose(t, "overlap-full", StructuralOverlap(10, 10), 1.0, 1e-15)
}

func TestStructuralOverlap_Zero(t *testing.T) {
	assertClose(t, "overlap-zero-total", StructuralOverlap(5, 0), 0.0, 0)
}

func TestStructuralOverlap_Negative(t *testing.T) {
	assertClose(t, "overlap-neg-total", StructuralOverlap(5, -1), 0.0, 0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — vector panics
// ═══════════════════════════════════════════════════════════════════════════

func TestVectorAdd_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for length mismatch")
		}
	}()
	VectorAdd([]float64{1}, []float64{1, 2}, make([]float64, 1))
}

func TestVectorSub_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for length mismatch")
		}
	}()
	VectorSub([]float64{1}, []float64{1, 2}, make([]float64, 1))
}

func TestVectorScale_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for length mismatch")
		}
	}()
	VectorScale([]float64{1, 2}, 2.0, make([]float64, 3))
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — matrix operations
// ═══════════════════════════════════════════════════════════════════════════

func TestMatMul_2x2(t *testing.T) {
	A := []float64{1, 2, 3, 4}
	B := []float64{5, 6, 7, 8}
	out := make([]float64, 4)
	MatMul(A, 2, 2, B, 2, out)
	assertSliceClose(t, "matmul-2x2", out, []float64{19, 22, 43, 50}, 1e-15)
}

func TestMatMul_Identity3x3(t *testing.T) {
	I := make([]float64, 9)
	Identity(3, I)
	B := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}
	out := make([]float64, 9)
	MatMul(I, 3, 3, B, 3, out)
	assertSliceClose(t, "matmul-identity", out, B, 1e-15)
}

func TestMatMul_NonSquare(t *testing.T) {
	A := []float64{1, 2, 3, 4, 5, 6}
	B := []float64{7, 8, 9, 10, 11, 12}
	out := make([]float64, 4)
	MatMul(A, 2, 3, B, 2, out)
	assertSliceClose(t, "matmul-nonsquare", out, []float64{58, 64, 139, 154}, 1e-15)
}

func TestMatTranspose_Square(t *testing.T) {
	A := []float64{1, 2, 3, 4}
	out := make([]float64, 4)
	MatTranspose(A, 2, 2, out)
	assertSliceClose(t, "transpose-2x2", out, []float64{1, 3, 2, 4}, 1e-15)
}

func TestMatTranspose_Rectangular(t *testing.T) {
	A := []float64{1, 2, 3, 4, 5, 6}
	out := make([]float64, 6)
	MatTranspose(A, 2, 3, out)
	// 2x3 -> 3x2: [[1,4],[2,5],[3,6]]
	assertSliceClose(t, "transpose-2x3", out, []float64{1, 4, 2, 5, 3, 6}, 1e-15)
}

func TestMatTranspose_DoubleTranspose(t *testing.T) {
	A := []float64{1, 2, 3, 4, 5, 6}
	tmp := make([]float64, 6)
	out := make([]float64, 6)
	MatTranspose(A, 2, 3, tmp) // 2x3 -> 3x2
	MatTranspose(tmp, 3, 2, out) // 3x2 -> 2x3
	assertSliceClose(t, "double-transpose", out, A, 1e-15)
}

func TestMatVecMul_KnownValue(t *testing.T) {
	A := []float64{1, 2, 3, 4}
	x := []float64{5, 6}
	out := make([]float64, 2)
	MatVecMul(A, 2, 2, x, out)
	// [1*5+2*6, 3*5+4*6] = [17, 39]
	assertSliceClose(t, "matvec-2x2", out, []float64{17, 39}, 1e-15)
}

func TestMatVecMul_Identity(t *testing.T) {
	I := make([]float64, 9)
	Identity(3, I)
	x := []float64{7, 8, 9}
	out := make([]float64, 3)
	MatVecMul(I, 3, 3, x, out)
	assertSliceClose(t, "matvec-identity", out, x, 1e-15)
}

func TestIdentity_3x3(t *testing.T) {
	out := make([]float64, 9)
	Identity(3, out)
	expected := []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}
	assertSliceClose(t, "identity-3", out, expected, 0)
}

func TestIdentity_1x1(t *testing.T) {
	out := make([]float64, 1)
	Identity(1, out)
	assertSliceClose(t, "identity-1", out, []float64{1}, 0)
}

func TestMatAdd_KnownValue(t *testing.T) {
	A := []float64{1, 2, 3, 4}
	B := []float64{5, 6, 7, 8}
	out := make([]float64, 4)
	MatAdd(A, B, 2, 2, out)
	assertSliceClose(t, "mat-add", out, []float64{6, 8, 10, 12}, 1e-15)
}

func TestMatScale_KnownValue(t *testing.T) {
	A := []float64{1, 2, 3, 4}
	out := make([]float64, 4)
	MatScale(A, 2, 2, 3.0, out)
	assertSliceClose(t, "mat-scale", out, []float64{3, 6, 9, 12}, 1e-15)
}

func TestMatScale_Zero(t *testing.T) {
	A := []float64{1, 2, 3, 4}
	out := make([]float64, 4)
	MatScale(A, 2, 2, 0.0, out)
	assertSliceClose(t, "mat-scale-zero", out, []float64{0, 0, 0, 0}, 0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — matrix panics
// ═══════════════════════════════════════════════════════════════════════════

func TestMatMul_PanicA(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic")
		}
	}()
	MatMul([]float64{1, 2}, 2, 2, []float64{1, 2, 3, 4}, 2, make([]float64, 4))
}

func TestMatMul_PanicB(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic")
		}
	}()
	MatMul([]float64{1, 2, 3, 4}, 2, 2, []float64{1, 2}, 2, make([]float64, 4))
}

func TestMatMul_PanicOut(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic")
		}
	}()
	MatMul([]float64{1, 2, 3, 4}, 2, 2, []float64{1, 2, 3, 4}, 2, make([]float64, 3))
}

func TestMatTranspose_Panic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic")
		}
	}()
	MatTranspose([]float64{1, 2}, 2, 2, make([]float64, 4))
}

func TestMatVecMul_PanicX(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic")
		}
	}()
	MatVecMul([]float64{1, 2, 3, 4}, 2, 2, []float64{1}, make([]float64, 2))
}

func TestIdentity_Panic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic")
		}
	}()
	Identity(3, make([]float64, 8))
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — correlation
// ═══════════════════════════════════════════════════════════════════════════

func TestPearsonCorrelation_PerfectPositive(t *testing.T) {
	got := PearsonCorrelation([]float64{1, 2, 3}, []float64{2, 4, 6})
	assertClose(t, "pearson-perfect+", got, 1.0, 1e-15)
}

func TestPearsonCorrelation_PerfectNegative(t *testing.T) {
	got := PearsonCorrelation([]float64{1, 2, 3}, []float64{6, 4, 2})
	assertClose(t, "pearson-perfect-", got, -1.0, 1e-15)
}

func TestPearsonCorrelation_ZeroVariance(t *testing.T) {
	got := PearsonCorrelation([]float64{5, 5, 5}, []float64{1, 2, 3})
	assertClose(t, "pearson-zero-var", got, 0.0, 0)
}

func TestPearsonCorrelation_SinglePoint(t *testing.T) {
	got := PearsonCorrelation([]float64{1}, []float64{2})
	assertClose(t, "pearson-single", got, 0.0, 0)
}

func TestPearsonCorrelation_LengthMismatch(t *testing.T) {
	got := PearsonCorrelation([]float64{1, 2}, []float64{1, 2, 3})
	assertClose(t, "pearson-mismatch", got, 0.0, 0)
}

func TestSpearmanCorrelation_PerfectPositive(t *testing.T) {
	got := SpearmanCorrelation([]float64{1, 2, 3, 4, 5}, []float64{10, 20, 30, 40, 50})
	assertClose(t, "spearman-perfect+", got, 1.0, 1e-12)
}

func TestSpearmanCorrelation_PerfectNegative(t *testing.T) {
	got := SpearmanCorrelation([]float64{1, 2, 3, 4, 5}, []float64{50, 40, 30, 20, 10})
	assertClose(t, "spearman-perfect-", got, -1.0, 1e-12)
}

func TestSpearmanCorrelation_WithTies(t *testing.T) {
	// x: ranks [1, 2.5, 2.5, 4] (tie at 2 and 2)
	// y: ranks [1, 2.5, 2.5, 4] (tie at 5 and 5)
	got := SpearmanCorrelation([]float64{1, 2, 2, 4}, []float64{3, 5, 5, 9})
	assertClose(t, "spearman-ties", got, 1.0, 1e-12)
}

func TestSpearmanCorrelation_Nonlinear(t *testing.T) {
	// Monotonically increasing but nonlinear — Spearman should give 1.0
	got := SpearmanCorrelation(
		[]float64{1, 2, 3, 4, 5},
		[]float64{1, 4, 9, 16, 25}, // y = x^2
	)
	assertClose(t, "spearman-nonlinear", got, 1.0, 1e-12)
}

func TestSpearmanCorrelation_SinglePoint(t *testing.T) {
	got := SpearmanCorrelation([]float64{1}, []float64{2})
	assertClose(t, "spearman-single", got, 0.0, 0)
}

func TestCovariance_PerfectPositive(t *testing.T) {
	// cov([1,2,3], [2,4,6]) = sum of products of deviations / (n-1)
	// means: 2, 4
	// deviations: (-1,-2,0,0,1,2) -> products: (2,0,2) -> sum=4 / 2 = 2
	got := Covariance([]float64{1, 2, 3}, []float64{2, 4, 6})
	assertClose(t, "cov-perfect+", got, 2.0, 1e-15)
}

func TestCovariance_Negative(t *testing.T) {
	got := Covariance([]float64{1, 2, 3}, []float64{6, 4, 2})
	assertClose(t, "cov-negative", got, -2.0, 1e-15)
}

func TestCovariance_Zero(t *testing.T) {
	got := Covariance([]float64{1, 2, 3}, []float64{5, 5, 5})
	assertClose(t, "cov-zero", got, 0.0, 1e-15)
}

func TestCovariance_SinglePoint(t *testing.T) {
	got := Covariance([]float64{1}, []float64{2})
	assertClose(t, "cov-single", got, 0.0, 0)
}

func TestCovarianceMatrix_2Features(t *testing.T) {
	// 3 samples, 2 features
	data := [][]float64{
		{1, 2},
		{3, 4},
		{5, 6},
	}
	out := make([]float64, 4)
	CovarianceMatrix(data, out)
	// Var(x) = cov(x,x) = 4, Var(y) = cov(y,y) = 4, cov(x,y) = 4
	assertSliceClose(t, "covmat-2f", out, []float64{4, 4, 4, 4}, 1e-12)
}

func TestCovarianceMatrix_Symmetric(t *testing.T) {
	data := [][]float64{
		{1, 5, 3},
		{2, 4, 7},
		{3, 6, 2},
		{4, 3, 8},
	}
	out := make([]float64, 9)
	CovarianceMatrix(data, out)
	// Verify symmetry: C[i][j] == C[j][i]
	for i := 0; i < 3; i++ {
		for j := i + 1; j < 3; j++ {
			if math.Abs(out[i*3+j]-out[j*3+i]) > 1e-15 {
				t.Errorf("covariance matrix not symmetric at [%d][%d]: %v != %v",
					i, j, out[i*3+j], out[j*3+i])
			}
		}
	}
}

func TestCovarianceMatrix_PanicEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for empty data")
		}
	}()
	CovarianceMatrix([][]float64{}, make([]float64, 0))
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

func assertClose(t *testing.T, label string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %v, want %v (tol %v)", label, got, want, tol)
	}
}

func assertSliceClose(t *testing.T, label string, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch: got %d, want %d", label, len(got), len(want))
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > tol {
			t.Errorf("%s[%d]: got %v, want %v (tol %v)", label, i, got[i], want[i], tol)
		}
	}
}

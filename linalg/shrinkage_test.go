package linalg

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/davly/reality/testutil"
)

// ═══════════════════════════════════════════════════════════════════════════
// Golden-file tests — shared vectors across Go, Python, C++, C#.
// Expected values are produced by an independent numpy reference implementing
// the same published formulas (see gen_shrinkage_golden.py / each file's
// "source" field); CleanCorrelation uses numpy.linalg.eigh (LAPACK), a
// different eigensolver than the Go Jacobi, so agreement validates the Go
// eigendecomposition too.
// ═══════════════════════════════════════════════════════════════════════════

// shrinkGolden mirrors the golden JSON with the extra per-case "shrinkage"
// scalar that testutil.TestCase does not expose.
type shrinkGolden struct {
	Function string `json:"function"`
	Cases    []struct {
		Description string  `json:"description"`
		Shrinkage   float64 `json:"shrinkage"`
	} `json:"cases"`
}

func loadShrink(t *testing.T, path string) shrinkGolden {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var g shrinkGolden
	if err := json.Unmarshal(data, &g); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	return g
}

func TestGolden_JamesSteinShrink(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/linalg/james_stein.json")
	intens := loadShrink(t, "testdata/linalg/james_stein.json")
	for i, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			means := testutil.InputFloat64Slice(t, tc, "means")
			variance := testutil.InputFloat64(t, tc, "variance")
			got, c := JamesSteinShrink(means, variance)
			testutil.AssertFloat64Slice(t, tc, got)
			if math.Abs(c-intens.Cases[i].Shrinkage) > 1e-12 {
				t.Errorf("shrinkage factor: got %v want %v", c, intens.Cases[i].Shrinkage)
			}
		})
	}
}

func TestGolden_LedoitWolfIdentity(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/linalg/lw_identity.json")
	intens := loadShrink(t, "testdata/linalg/lw_identity.json")
	for i, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			x := testutil.InputFloat64Slice(t, tc, "x")
			T := testutil.InputInt(t, tc, "T")
			N := testutil.InputInt(t, tc, "N")
			got, shr := LedoitWolfShrinkageIdentity(x, T, N)
			testutil.AssertFloat64Slice(t, tc, got)
			if math.Abs(shr-intens.Cases[i].Shrinkage) > 1e-9 {
				t.Errorf("intensity: got %v want %v", shr, intens.Cases[i].Shrinkage)
			}
		})
	}
}

func TestGolden_LedoitWolfConstantCorr(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/linalg/lw_constcorr.json")
	intens := loadShrink(t, "testdata/linalg/lw_constcorr.json")
	for i, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			x := testutil.InputFloat64Slice(t, tc, "x")
			T := testutil.InputInt(t, tc, "T")
			N := testutil.InputInt(t, tc, "N")
			got, shr := LedoitWolfShrinkageConstantCorr(x, T, N)
			testutil.AssertFloat64Slice(t, tc, got)
			if math.Abs(shr-intens.Cases[i].Shrinkage) > 1e-9 {
				t.Errorf("intensity: got %v want %v", shr, intens.Cases[i].Shrinkage)
			}
		})
	}
}

func TestGolden_MarchenkoPasturBounds(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/linalg/marchenko_pastur.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			q := testutil.InputFloat64(t, tc, "q")
			lo, hi := MarchenkoPasturBounds(q)
			testutil.AssertFloat64Slice(t, tc, []float64{lo, hi})
		})
	}
}

func TestGolden_CleanCorrelation(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/linalg/clean_correlation.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			corr := testutil.InputFloat64Slice(t, tc, "corr")
			T := testutil.InputInt(t, tc, "T")
			N := testutil.InputInt(t, tc, "N")
			got := CleanCorrelation(corr, T, N)
			testutil.AssertFloat64Slice(t, tc, got)
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// James-Stein unit + property tests
// ═══════════════════════════════════════════════════════════════════════════

func TestJamesSteinShrink_BelowThreshold(t *testing.T) {
	// p < 3 is inadmissible: return the raw means unchanged with c = 0.
	means := []float64{1.0, 2.0}
	got, c := JamesSteinShrink(means, 1.0)
	if c != 0 {
		t.Errorf("p<3 shrinkage: got %v want 0", c)
	}
	for i := range means {
		if got[i] != means[i] {
			t.Errorf("p<3 element %d: got %v want %v", i, got[i], means[i])
		}
	}
}

func TestJamesSteinShrink_DoesNotMutateInput(t *testing.T) {
	means := []float64{1.0, 2.0, 3.0}
	orig := append([]float64(nil), means...)
	JamesSteinShrink(means, 1.0)
	for i := range means {
		if means[i] != orig[i] {
			t.Fatalf("input mutated at %d: %v != %v", i, means[i], orig[i])
		}
	}
}

func TestJamesSteinShrink_FactorInRange(t *testing.T) {
	means := []float64{1, 2, 3, 4, 5, 6, 7}
	for _, v := range []float64{0.001, 0.1, 1, 10, 1000} {
		_, c := JamesSteinShrink(means, v)
		if c < 0 || c > 1 {
			t.Errorf("variance %v: factor %v out of [0,1]", v, c)
		}
	}
}

func TestJamesSteinShrink_ShrinksTowardGrandMean(t *testing.T) {
	// With positive shrinkage each estimate moves toward the grand mean, so the
	// spread of the shrunk estimates never exceeds the raw spread.
	means := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	got, c := JamesSteinShrink(means, 2.0)
	if c <= 0 {
		t.Fatalf("expected positive shrinkage, got %v", c)
	}
	rawSpread := spread(means)
	shrunkSpread := spread(got)
	if shrunkSpread > rawSpread {
		t.Errorf("shrunk spread %v exceeds raw spread %v", shrunkSpread, rawSpread)
	}
}

// TestJamesSteinShrink_CompressesVariance documents the RD-4A failure mode as a
// live property: shrinking the means COMPRESSES the sample variance, which is
// exactly why shrunk values must never be routed into a dispersion/tail input.
func TestJamesSteinShrink_CompressesVariance(t *testing.T) {
	means := []float64{-3, -1, 0, 1, 3, 2, -2}
	rawVar := sampleVar(means)
	shrunk, c := JamesSteinShrink(means, 1.5)
	if c <= 0 {
		t.Fatalf("expected positive shrinkage")
	}
	shrunkVar := sampleVar(shrunk)
	if shrunkVar >= rawVar {
		t.Errorf("expected shrunk variance %v < raw variance %v (RD-4A: shrinkage compresses dispersion)", shrunkVar, rawVar)
	}
}

func TestJamesSteinShrink_NaNGuard(t *testing.T) {
	means := []float64{1.0, math.NaN(), 3.0}
	got, c := JamesSteinShrink(means, 1.0)
	if c != 0 {
		t.Errorf("NaN input should yield c=0, got %v", c)
	}
	// Returns an unchanged copy (NaN preserved position-wise).
	if got[0] != 1.0 || got[2] != 3.0 || !math.IsNaN(got[1]) {
		t.Errorf("NaN guard should return copy unchanged: %v", got)
	}
}

func TestJamesSteinShrink_DegenerateIdenticalMeans(t *testing.T) {
	got, c := JamesSteinShrink([]float64{4, 4, 4}, 1.0)
	if c != 1.0 {
		t.Errorf("identical means: c got %v want 1", c)
	}
	for _, v := range got {
		if math.Abs(v-4.0) > 1e-15 {
			t.Errorf("identical means: got %v want 4", v)
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Ledoit-Wolf unit + property tests
// ═══════════════════════════════════════════════════════════════════════════

func TestLedoitWolf_IntensityInRange(t *testing.T) {
	x := []float64{
		0.01, -0.02, 0.03,
		-0.01, 0.02, -0.01,
		0.02, 0.01, 0.00,
		0.00, -0.01, 0.02,
		0.03, 0.02, -0.02,
	}
	_, s1 := LedoitWolfShrinkageIdentity(x, 5, 3)
	_, s2 := LedoitWolfShrinkageConstantCorr(x, 5, 3)
	for _, s := range []float64{s1, s2} {
		if s < 0 || s > 1 {
			t.Errorf("intensity %v out of [0,1]", s)
		}
	}
}

func TestLedoitWolf_SymmetricOutput(t *testing.T) {
	x := make([]float64, 40)
	for i := range x {
		x[i] = math.Sin(float64(i)*0.7) * 0.02
	}
	for _, fn := range []func([]float64, int, int) ([]float64, float64){
		LedoitWolfShrinkageIdentity, LedoitWolfShrinkageConstantCorr,
	} {
		sigma, _ := fn(x, 10, 4)
		assertSymmetric(t, sigma, 4, 1e-12)
	}
}

func TestLedoitWolf_ConstCorrPreservesDiagonal(t *testing.T) {
	// The constant-correlation target keeps sample variances on the diagonal
	// (F_ii = S_ii), so the shrunk diagonal equals the sample diagonal exactly.
	x := make([]float64, 60)
	for i := range x {
		x[i] = math.Cos(float64(i)*0.3)*0.015 + 0.001*float64(i%5)
	}
	T, N := 15, 4
	y := centeredCopy(x, T, N)
	S := sampleCovMLE(y, T, N)
	sigma, _ := LedoitWolfShrinkageConstantCorr(x, T, N)
	for i := 0; i < N; i++ {
		if math.Abs(sigma[i*N+i]-S[i*N+i]) > 1e-9 {
			t.Errorf("const-corr diagonal not preserved: sigma %v S %v", sigma[i*N+i], S[i*N+i])
		}
	}
}

func TestLedoitWolf_IdentityDiagonalBetweenSampleAndMean(t *testing.T) {
	// The scaled-identity target pulls the diagonal toward the common mean
	// variance m = trace(S)/N, so each shrunk diagonal lies between S_ii and m:
	// sigma_ii = shr*m + (1-shr)*S_ii.
	x := make([]float64, 60)
	for i := range x {
		x[i] = math.Cos(float64(i)*0.3)*0.015 + 0.001*float64(i%5)
	}
	T, N := 15, 4
	y := centeredCopy(x, T, N)
	S := sampleCovMLE(y, T, N)
	var m float64
	for i := 0; i < N; i++ {
		m += S[i*N+i]
	}
	m /= float64(N)
	sigma, shr := LedoitWolfShrinkageIdentity(x, T, N)
	for i := 0; i < N; i++ {
		want := shr*m + (1-shr)*S[i*N+i]
		if math.Abs(sigma[i*N+i]-want) > 1e-9 {
			t.Errorf("identity diagonal[%d]: got %v want shr*m+(1-shr)*S_ii=%v", i, sigma[i*N+i], want)
		}
	}
}

func TestLedoitWolf_PanicsOnBadDims(t *testing.T) {
	assertPanics(t, "T<2 identity", func() { LedoitWolfShrinkageIdentity([]float64{1, 2}, 1, 2) })
	assertPanics(t, "len mismatch identity", func() { LedoitWolfShrinkageIdentity([]float64{1, 2, 3}, 2, 2) })
	assertPanics(t, "N<1 constcorr", func() { LedoitWolfShrinkageConstantCorr([]float64{}, 2, 0) })
}

func TestLedoitWolf_SingleAssetConstCorr(t *testing.T) {
	// N==1 has no off-diagonal correlation to shrink: intensity 0, sample var.
	x := []float64{0.01, -0.02, 0.03, 0.00, 0.02}
	sigma, s := LedoitWolfShrinkageConstantCorr(x, 5, 1)
	if s != 0 {
		t.Errorf("N=1 intensity got %v want 0", s)
	}
	y := centeredCopy(x, 5, 1)
	S := sampleCovMLE(y, 5, 1)
	if math.Abs(sigma[0]-S[0]) > 1e-15 {
		t.Errorf("N=1 got %v want sample var %v", sigma[0], S[0])
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Marchenko-Pastur + CleanCorrelation
// ═══════════════════════════════════════════════════════════════════════════

func TestMarchenkoPastur_KnownValues(t *testing.T) {
	// q=0.25 -> sqrt=0.5 -> (0.25, 2.25); q=1 -> (0, 4).
	lo, hi := MarchenkoPasturBounds(0.25)
	if math.Abs(lo-0.25) > 1e-15 || math.Abs(hi-2.25) > 1e-15 {
		t.Errorf("q=0.25 got (%v,%v) want (0.25,2.25)", lo, hi)
	}
	lo, hi = MarchenkoPasturBounds(1.0)
	if math.Abs(lo-0.0) > 1e-15 || math.Abs(hi-4.0) > 1e-15 {
		t.Errorf("q=1 got (%v,%v) want (0,4)", lo, hi)
	}
}

func TestMarchenkoPastur_PanicsOnBadQ(t *testing.T) {
	assertPanics(t, "q=0", func() { MarchenkoPasturBounds(0) })
	assertPanics(t, "q<0", func() { MarchenkoPasturBounds(-1) })
	assertPanics(t, "q=NaN", func() { MarchenkoPasturBounds(math.NaN()) })
}

func TestCleanCorrelation_DiagonalExactlyOne(t *testing.T) {
	corr := []float64{
		1.0, 0.6, 0.3, 0.1,
		0.6, 1.0, 0.4, 0.2,
		0.3, 0.4, 1.0, 0.5,
		0.1, 0.2, 0.5, 1.0,
	}
	out := CleanCorrelation(corr, 50, 4)
	for i := 0; i < 4; i++ {
		if out[i*4+i] != 1.0 {
			t.Errorf("diagonal[%d] = %v, want exactly 1.0", i, out[i*4+i])
		}
	}
	assertSymmetric(t, out, 4, 1e-12)
}

func TestCleanCorrelation_TracePreservedApprox(t *testing.T) {
	// Replacing noise eigenvalues by their mean preserves the eigenvalue sum
	// (trace) before renormalisation; after diagonal renormalisation the trace
	// is exactly N.
	corr := []float64{
		1.0, 0.5, 0.2,
		0.5, 1.0, 0.3,
		0.2, 0.3, 1.0,
	}
	out := CleanCorrelation(corr, 30, 3)
	tr := out[0] + out[4] + out[8]
	if math.Abs(tr-3.0) > 1e-12 {
		t.Errorf("trace got %v want 3", tr)
	}
}

func TestCleanCorrelation_SingleAsset(t *testing.T) {
	out := CleanCorrelation([]float64{1.0}, 10, 1)
	if len(out) != 1 || out[0] != 1.0 {
		t.Errorf("N=1 got %v want [1]", out)
	}
}

func TestCleanCorrelation_PanicsOnBadDims(t *testing.T) {
	assertPanics(t, "N<1", func() { CleanCorrelation([]float64{}, 10, 0) })
	assertPanics(t, "T<1", func() { CleanCorrelation([]float64{1}, 0, 1) })
	assertPanics(t, "len mismatch", func() { CleanCorrelation([]float64{1, 2, 3}, 10, 2) })
}

// ═══════════════════════════════════════════════════════════════════════════
// Jacobi eigendecomposition — validated against the existing QRAlgorithm
// eigenvalues and by reconstruction A = V diag(lambda) V^T.
// ═══════════════════════════════════════════════════════════════════════════

func TestJacobiEigen_MatchesQRAlgorithm(t *testing.T) {
	A := []float64{
		4, 1, 2, 0,
		1, 3, 0, 1,
		2, 0, 5, 1,
		0, 1, 1, 2,
	}
	n := 4
	vals, vecs := jacobiEigenSymmetric(A, n)

	qr := make([]float64, n)
	QRAlgorithm(A, n, qr, 1000)
	for i := 0; i < n; i++ {
		if math.Abs(vals[i]-qr[i]) > 1e-9 {
			t.Errorf("eigenvalue %d: jacobi %v vs QR %v", i, vals[i], qr[i])
		}
	}

	// Reconstruct A from V diag(lambda) V^T.
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var sum float64
			for k := 0; k < n; k++ {
				sum += vecs[i*n+k] * vals[k] * vecs[j*n+k]
			}
			if math.Abs(sum-A[i*n+j]) > 1e-9 {
				t.Errorf("reconstruct[%d,%d]: got %v want %v", i, j, sum, A[i*n+j])
			}
		}
	}
}

func TestJacobiEigen_Orthonormal(t *testing.T) {
	A := []float64{
		2, -1, 0,
		-1, 2, -1,
		0, -1, 2,
	}
	n := 3
	_, V := jacobiEigenSymmetric(A, n)
	// Columns must be orthonormal: V^T V = I.
	for a := 0; a < n; a++ {
		for b := 0; b < n; b++ {
			var dot float64
			for i := 0; i < n; i++ {
				dot += V[i*n+a] * V[i*n+b]
			}
			want := 0.0
			if a == b {
				want = 1.0
			}
			if math.Abs(dot-want) > 1e-12 {
				t.Errorf("V^T V[%d,%d]=%v want %v", a, b, dot, want)
			}
		}
	}
}

func TestJacobiEigen_DescendingOrder(t *testing.T) {
	A := []float64{
		5, 0, 0,
		0, 1, 0,
		0, 0, 3,
	}
	vals, _ := jacobiEigenSymmetric(A, 3)
	if !(vals[0] >= vals[1] && vals[1] >= vals[2]) {
		t.Errorf("eigenvalues not descending: %v", vals)
	}
	// Diagonal matrix eigenvalues are its diagonal.
	want := []float64{5, 3, 1}
	for i, w := range want {
		if math.Abs(vals[i]-w) > 1e-12 {
			t.Errorf("eigenvalue %d got %v want %v", i, vals[i], w)
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// helpers
// ═══════════════════════════════════════════════════════════════════════════

func spread(v []float64) float64 {
	lo, hi := v[0], v[0]
	for _, x := range v {
		if x < lo {
			lo = x
		}
		if x > hi {
			hi = x
		}
	}
	return hi - lo
}

func sampleVar(v []float64) float64 {
	n := float64(len(v))
	var mean float64
	for _, x := range v {
		mean += x
	}
	mean /= n
	var s float64
	for _, x := range v {
		d := x - mean
		s += d * d
	}
	return s / (n - 1)
}

func assertSymmetric(t *testing.T, m []float64, n int, tol float64) {
	t.Helper()
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if math.Abs(m[i*n+j]-m[j*n+i]) > tol {
				t.Errorf("asymmetric at [%d,%d]: %v vs %v", i, j, m[i*n+j], m[j*n+i])
			}
		}
	}
}

func assertPanics(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: expected panic, got none", name)
		}
	}()
	fn()
}

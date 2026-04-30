package proximal

import (
	"math"
	"testing"
)

// =========================================================================
// FBS / FISTA correctness — LASSO test
// =========================================================================

// quadraticGrad returns a GradOp for f(x) = 0.5 ||A x - b||^2 where A is
// row-major rows-by-cols. Internally it allocates an Ax-b buffer once and
// reuses it across calls.
func quadraticGrad(A []float64, rows, cols int, b []float64) GradOp {
	resid := make([]float64, rows)
	return func(x []float64, out []float64) float64 {
		// resid = A x - b
		for i := 0; i < rows; i++ {
			s := 0.0
			for j := 0; j < cols; j++ {
				s += A[i*cols+j] * x[j]
			}
			resid[i] = s - b[i]
		}
		// out = A^T resid
		for j := 0; j < cols; j++ {
			s := 0.0
			for i := 0; i < rows; i++ {
				s += A[i*cols+j] * resid[i]
			}
			out[j] = s
		}
		// f = 0.5 * ||resid||^2
		var f float64
		for _, r := range resid {
			f += 0.5 * r * r
		}
		return f
	}
}

// TestFbs_LassoRecoversSparseSolution constructs a tiny LASSO problem
// where the true solution x* is known to be sparse, then verifies FBS
// recovers it. min 0.5||Ax-b||^2 + lambda*||x||_1.
func TestFbs_LassoRecoversSparseSolution(t *testing.T) {
	// True x = (1, 0, -2, 0). A is 6x4 with mild correlation; b = A x* + 0.
	A := []float64{
		1.0, 0.5, 0.0, 0.0,
		0.0, 1.0, 0.5, 0.0,
		0.0, 0.0, 1.0, 0.5,
		0.5, 0.0, 0.0, 1.0,
		1.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0,
	}
	xStar := []float64{1.0, 0.0, -2.0, 0.0}
	rows, cols := 6, 4
	b := make([]float64, rows)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			b[i] += A[i*cols+j] * xStar[j]
		}
	}
	grad := quadraticGrad(A, rows, cols, b)

	// L = ||A^T A||_2; we estimate as sum_ij A_ij^2 (Frobenius bound).
	var L float64
	for _, a := range A {
		L += a * a
	}
	step := 1.0 / L
	lambda := 0.001 // small to leave xStar unbiased

	x := make([]float64, cols)
	work := make([]float64, cols)
	res, err := Fbs(grad, func(v []float64, gamma float64, out []float64) {
		ProxL1(v, gamma*lambda, out)
	}, x, work, FbsConfig{Step: step, MaxIter: 5000, AbsTol: 1e-10, Accelerate: true})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Converged {
		t.Errorf("FISTA did not converge in %d iters (delta=%.3e)", res.Iter, res.FinalDelta)
	}
	for i, want := range xStar {
		if math.Abs(x[i]-want) > 0.05 {
			t.Errorf("x[%d] = %.4f, want close to %.4f", i, x[i], want)
		}
	}
	// Sparsity: zero entries should be exactly zero (within bias).
	for i, want := range xStar {
		if want == 0 && math.Abs(x[i]) > 0.05 {
			t.Errorf("x[%d] = %.4f, expected ~0 (sparse coordinate)", i, x[i])
		}
	}
}

// =========================================================================
// Plain FBS vs FISTA convergence comparison
// =========================================================================

// TestFbs_FistaBeatsPlainOnIllConditioned ensures FISTA reaches a strictly
// closer iterate to the optimum than plain FBS at a fixed iteration budget
// on an ill-conditioned smooth quadratic, where the O(1/k^2) vs O(1/k) gap
// is visible. Convergence-by-||x_new - x_old||-tolerance is biased against
// FISTA (its iterates oscillate near the optimum), so we compare distance
// to xStar at a fixed budget rather than iter count.
func TestFbs_FistaBeatsPlainOnIllConditioned(t *testing.T) {
	// Build a 50x50 quadratic with eigenvalue ratio ~1000:1 by using a
	// diagonal A with a geometric spectrum.
	const n = 50
	A := make([]float64, n*n)
	for i := 0; i < n; i++ {
		// sigma_i = 1000^{-i/(n-1)} -> first 1, last 1e-3.
		sigma := math.Pow(1000.0, -float64(i)/float64(n-1))
		A[i*n+i] = sigma
	}
	xStar := make([]float64, n)
	for i := range xStar {
		xStar[i] = 1.0 // dense optimum
	}
	b := make([]float64, n)
	for i := 0; i < n; i++ {
		b[i] = A[i*n+i] * xStar[i]
	}
	grad := quadraticGrad(A, n, n, b)
	// L = max_i sigma_i^2 = 1.
	step := 1.0
	prox := ProxNonNeg
	const budget = 200

	xPlain := make([]float64, n)
	wPlain := make([]float64, n)
	if _, err := Fbs(grad, prox, xPlain, wPlain,
		FbsConfig{Step: step, MaxIter: budget, AbsTol: 0, Accelerate: false}); err != nil {
		t.Fatal(err)
	}

	xFista := make([]float64, n)
	wFista := make([]float64, n)
	if _, err := Fbs(grad, prox, xFista, wFista,
		FbsConfig{Step: step, MaxIter: budget, AbsTol: 0, Accelerate: true}); err != nil {
		t.Fatal(err)
	}

	distPlain := 0.0
	distFista := 0.0
	for i := range xStar {
		dp := xPlain[i] - xStar[i]
		df := xFista[i] - xStar[i]
		distPlain += dp * dp
		distFista += df * df
	}
	distPlain = math.Sqrt(distPlain)
	distFista = math.Sqrt(distFista)
	if !(distFista < distPlain) {
		t.Errorf("FISTA dist to optimum = %.6f, plain FBS = %.6f — accel did not help", distFista, distPlain)
	}
}

// =========================================================================
// Validation
// =========================================================================

func TestFbs_RejectsBadConfig(t *testing.T) {
	x := make([]float64, 4)
	work := make([]float64, 4)
	grad := func(_, out []float64) float64 {
		for i := range out {
			out[i] = 0
		}
		return 0
	}
	prox := ProxNonNeg
	cases := []FbsConfig{
		{Step: 0},
		{Step: -1},
		{Step: math.Inf(1)},
		{Step: math.NaN()},
	}
	for i, cfg := range cases {
		_, err := Fbs(grad, prox, x, work, cfg)
		if err == nil {
			t.Errorf("case %d: expected error, got nil", i)
		}
	}
	if _, err := Fbs(nil, prox, x, work, FbsConfig{Step: 1}); err == nil {
		t.Error("nil grad should error")
	}
	if _, err := Fbs(grad, nil, x, work, FbsConfig{Step: 1}); err == nil {
		t.Error("nil prox should error")
	}
	short := make([]float64, 2)
	if _, err := Fbs(grad, prox, x, short, FbsConfig{Step: 1}); err == nil {
		t.Error("short work buffer should error")
	}
}

// =========================================================================
// Determinism
// =========================================================================

func TestFbs_Deterministic(t *testing.T) {
	A := []float64{1, 2, 3, 4, 5, 6}
	rows, cols := 3, 2
	b := []float64{1, 2, 3}
	grad := quadraticGrad(A, rows, cols, b)
	var L float64
	for _, a := range A {
		L += a * a
	}
	cfg := FbsConfig{Step: 1.0 / L, MaxIter: 200, AbsTol: 1e-10, Accelerate: true}

	run := func() []float64 {
		x := make([]float64, cols)
		w := make([]float64, cols)
		_, _ = Fbs(grad, ProxNonNeg, x, w, cfg)
		return x
	}
	a := run()
	b2 := run()
	for i := range a {
		if a[i] != b2[i] {
			t.Errorf("non-deterministic at i=%d: %v vs %v", i, a[i], b2[i])
		}
	}
}

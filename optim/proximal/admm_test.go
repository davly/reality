package proximal

import (
	"math"
	"testing"
)

// =========================================================================
// ADMM correctness — projection-onto-intersection
// =========================================================================

// TestAdmm_ProjectsOntoIntersection verifies ADMM finds the projection of a
// point onto the intersection of two convex sets (non-negative orthant ∩ box
// [-2, 1]).  The intersection is the box [0, 1]; projecting (3, -2, 0.5)
// onto it gives (1, 0, 0.5).
func TestAdmm_ProjectsOntoIntersection(t *testing.T) {
	n := 3
	x := []float64{3.0, -2.0, 0.5}
	z := make([]float64, n)
	u := make([]float64, n)
	copy(z, x)

	lo := []float64{-2, -2, -2}
	hi := []float64{1, 1, 1}
	proxF := ProxNonNeg
	proxG := ProxBox(lo, hi)

	res, err := Admm(proxF, proxG, x, z, u, AdmmConfig{Rho: 1.0, MaxIter: 1000, AbsTol: 1e-9})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Converged {
		t.Errorf("ADMM did not converge: iter=%d primal=%.3e dual=%.3e", res.Iter, res.PrimalResid, res.DualResid)
	}
	want := []float64{1.0, 0.0, 0.5}
	for i := range want {
		if math.Abs(x[i]-want[i]) > 1e-6 {
			t.Errorf("x[%d] = %.6f, want %.6f", i, x[i], want[i])
		}
	}
}

// TestAdmm_LassoConsensus solves a tiny LASSO via consensus ADMM:
// f(x) = 0.5 ||A x - b||^2 has prox via the resolvent x = (A^T A + rho I)^-1
// (A^T b + rho v).  We test the simpler case where A^T A is diagonal, so
// the resolvent is element-wise.
func TestAdmm_LassoDiagonal(t *testing.T) {
	// A^T A = diag(2, 3, 4); A^T b = (2, 0, -8). True LASSO solution is the
	// element-wise soft-threshold of (A^T b) / diag = (1, 0, -2) at lambda /
	// diag.
	diag := []float64{2.0, 3.0, 4.0}
	atb := []float64{2.0, 0.0, -8.0}
	lambda := 0.1
	rho := 1.0
	n := 3

	proxF := func(v []float64, gamma float64, out []float64) {
		// resolvent of f via the diagonal: x_i = (A^T b + (1/gamma) v_i) /
		// (diag_i + 1/gamma)
		invG := 1.0 / gamma
		for i, vi := range v {
			out[i] = (atb[i] + invG*vi) / (diag[i] + invG)
		}
	}
	proxG := func(v []float64, gamma float64, out []float64) {
		ProxL1(v, gamma*lambda, out)
	}

	x := make([]float64, n)
	z := make([]float64, n)
	u := make([]float64, n)
	res, err := Admm(proxF, proxG, x, z, u, AdmmConfig{Rho: rho, MaxIter: 5000, AbsTol: 1e-9})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Converged {
		t.Errorf("ADMM did not converge: iter=%d primal=%.3e dual=%.3e", res.Iter, res.PrimalResid, res.DualResid)
	}
	// Closed form per coordinate: x_i = soft((A^T b)_i / diag_i, lambda / diag_i).
	want := make([]float64, n)
	for i := range want {
		raw := atb[i] / diag[i]
		thr := lambda / diag[i]
		switch {
		case raw > thr:
			want[i] = raw - thr
		case raw < -thr:
			want[i] = raw + thr
		default:
			want[i] = 0
		}
	}
	for i := range want {
		if math.Abs(z[i]-want[i]) > 1e-4 {
			t.Errorf("z[%d] = %.6f, want %.6f", i, z[i], want[i])
		}
	}
}

// =========================================================================
// Validation
// =========================================================================

func TestAdmm_RejectsBadConfig(t *testing.T) {
	x := make([]float64, 3)
	z := make([]float64, 3)
	u := make([]float64, 3)
	cases := []AdmmConfig{
		{Rho: 0},
		{Rho: -1},
		{Rho: math.Inf(1)},
		{Rho: math.NaN()},
	}
	for i, cfg := range cases {
		_, err := Admm(ProxNonNeg, ProxNonNeg, x, z, u, cfg)
		if err == nil {
			t.Errorf("case %d: expected error, got nil", i)
		}
	}
	if _, err := Admm(nil, ProxNonNeg, x, z, u, AdmmConfig{Rho: 1}); err == nil {
		t.Error("nil proxF should error")
	}
	if _, err := Admm(ProxNonNeg, nil, x, z, u, AdmmConfig{Rho: 1}); err == nil {
		t.Error("nil proxG should error")
	}
	short := make([]float64, 2)
	if _, err := Admm(ProxNonNeg, ProxNonNeg, x, short, u, AdmmConfig{Rho: 1}); err == nil {
		t.Error("mismatched-length z should error")
	}
}

// =========================================================================
// Determinism
// =========================================================================

func TestAdmm_Deterministic(t *testing.T) {
	run := func() []float64 {
		x := []float64{2.0, -1.0, 0.5}
		z := make([]float64, 3)
		u := make([]float64, 3)
		_, _ = Admm(ProxNonNeg, ProxBox([]float64{-1, -1, -1}, []float64{1, 1, 1}),
			x, z, u, AdmmConfig{Rho: 1.0, MaxIter: 100, AbsTol: 1e-12})
		return x
	}
	a := run()
	b := run()
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("non-deterministic at i=%d: %v vs %v", i, a[i], b[i])
		}
	}
}

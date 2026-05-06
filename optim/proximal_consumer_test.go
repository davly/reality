package optim

import (
	"math"
	"testing"

	"github.com/davly/reality/optim/proximal"
)

// TestProximalLasso_FBS_OrthogonalClosedForm closes the substrate-internal
// first-consumer push for github.com/davly/reality/optim/proximal by
// exercising it from the parent optim package on an orthogonal-design LASSO
// problem with a known closed-form solution.
//
// LASSO with orthogonal design X^T X = I and X = I has the closed-form
// optimum
//
//	β* = soft(y, λ) = sign(y) * max(|y| - λ, 0)
//
// Because optim is the parent package and proximal is the child, this is
// a real cross-package consumer (the test imports proximal and exercises
// its API on an optimisation problem with a closed-form check). Substrate-
// internal first-consumer push for proximal, S62 2026-05-06.
//
// Three solvers in the proximal package are exercised against the same
// closed-form: FBS, FISTA, and ADMM. All three must agree to within 1e-9.
func TestProximalLasso_FBS_OrthogonalClosedForm(t *testing.T) {
	// Problem: X = I, y = (3.0, -2.0, 0.5, -0.3, 0.0). λ = 1.0.
	// Closed-form β* = soft(y, λ) = (2.0, -1.0, 0.0, 0.0, 0.0).
	y := []float64{3.0, -2.0, 0.5, -0.3, 0.0}
	const lambda = 1.0
	want := softThreshold(y, lambda)

	// f(β) = 0.5 ||β - y||^2  ->  ∇f(β) = β - y.
	grad := func(x, out []float64) float64 {
		var fval float64
		for i, xi := range x {
			d := xi - y[i]
			out[i] = d
			fval += 0.5 * d * d
		}
		return fval
	}

	// prox_{γg} where g(β) = λ||β||_1: soft-threshold by γ*λ.
	prox := func(v []float64, gamma float64, out []float64) {
		proximal.ProxL1(v, gamma*lambda, out)
	}

	x := make([]float64, len(y))
	work := make([]float64, len(y))
	res, err := proximal.Fbs(grad, prox, x, work, proximal.FbsConfig{
		Step:    1.0, // L = 1 (largest eigenvalue of I = 1), step = 1/L
		MaxIter: 100,
		AbsTol:  1e-12,
	})
	if err != nil {
		t.Fatalf("FBS: %v", err)
	}
	if !res.Converged {
		t.Errorf("FBS did not converge in %d iters (final delta %g)", res.Iter, res.FinalDelta)
	}
	assertVecClose(t, "FBS", x, want, 1e-9)
}

func TestProximalLasso_FISTA_OrthogonalClosedForm(t *testing.T) {
	y := []float64{3.0, -2.0, 0.5, -0.3, 0.0}
	const lambda = 1.0
	want := softThreshold(y, lambda)

	grad := func(x, out []float64) float64 {
		var fval float64
		for i, xi := range x {
			d := xi - y[i]
			out[i] = d
			fval += 0.5 * d * d
		}
		return fval
	}
	prox := func(v []float64, gamma float64, out []float64) {
		proximal.ProxL1(v, gamma*lambda, out)
	}

	x := make([]float64, len(y))
	work := make([]float64, len(y))
	res, err := proximal.Fbs(grad, prox, x, work, proximal.FbsConfig{
		Step:       1.0,
		MaxIter:    100,
		AbsTol:     1e-12,
		Accelerate: true, // FISTA
	})
	if err != nil {
		t.Fatalf("FISTA: %v", err)
	}
	if !res.Converged {
		t.Errorf("FISTA did not converge in %d iters (final delta %g)", res.Iter, res.FinalDelta)
	}
	assertVecClose(t, "FISTA", x, want, 1e-9)
}

func TestProximalLasso_ADMM_OrthogonalClosedForm(t *testing.T) {
	y := []float64{3.0, -2.0, 0.5, -0.3, 0.0}
	const lambda = 1.0
	want := softThreshold(y, lambda)

	// Consensus ADMM splits min f(β) + g(β) as min f(x) + g(z) s.t. x = z.
	// Here f(x) = 0.5 ||x - y||^2; prox_{f/ρ}(v) = (v + y/ρ) / (1 + 1/ρ).
	proxF := func(v []float64, gamma float64, out []float64) {
		// gamma here is 1/ρ.
		denom := 1.0 + gamma
		for i, vi := range v {
			out[i] = (vi + gamma*y[i]) / denom
		}
	}
	// g(z) = λ||z||_1; prox_{g/ρ}(v) = soft(v, λ/ρ) = soft(v, gamma*λ).
	proxG := func(v []float64, gamma float64, out []float64) {
		proximal.ProxL1(v, gamma*lambda, out)
	}

	x := make([]float64, len(y))
	z := make([]float64, len(y))
	u := make([]float64, len(y))
	res, err := proximal.Admm(proxF, proxG, x, z, u, proximal.AdmmConfig{
		Rho:     1.0,
		MaxIter: 1000,
		AbsTol:  1e-10,
	})
	if err != nil {
		t.Fatalf("ADMM: %v", err)
	}
	if !res.Converged {
		t.Errorf("ADMM did not converge in %d iters (primal=%g dual=%g)", res.Iter, res.PrimalResid, res.DualResid)
	}
	// Check both x and z (they should agree at convergence).
	assertVecClose(t, "ADMM x", x, want, 1e-7)
	assertVecClose(t, "ADMM z", z, want, 1e-7)
}

// softThreshold is the element-wise soft-threshold operator: the closed-form
// LASSO optimum for an orthogonal design with X = I.
func softThreshold(y []float64, lambda float64) []float64 {
	out := make([]float64, len(y))
	for i, yi := range y {
		switch {
		case yi > lambda:
			out[i] = yi - lambda
		case yi < -lambda:
			out[i] = yi + lambda
		default:
			out[i] = 0
		}
	}
	return out
}

// assertVecClose fails t if any |a[i]-b[i]| exceeds tol.
func assertVecClose(t *testing.T, label string, a, b []float64, tol float64) {
	t.Helper()
	if len(a) != len(b) {
		t.Errorf("%s length mismatch: len(a)=%d len(b)=%d", label, len(a), len(b))
		return
	}
	for i, ai := range a {
		if d := math.Abs(ai - b[i]); d > tol {
			t.Errorf("%s mismatch at index %d: got %g want %g |delta|=%g (tol=%g)",
				label, i, ai, b[i], d, tol)
		}
	}
}

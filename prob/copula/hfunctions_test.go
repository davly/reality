package copula

import (
	"errors"
	"math"
	"testing"
)

// Tests for h-functions on Clayton + Gumbel copulas (a241 vine substrate
// stepping-stone). The h-function is the conditional CDF
// h(u | v; θ) = ∂C(u, v; θ) / ∂v.
//
// We verify the closed-form expressions against finite-difference
// approximations of the CDF. Tolerance is loose because central
// difference is O(h²) accurate and we use h = 1e-5.

func TestClaytonHFn_RejectsInvalidTheta(t *testing.T) {
	for _, theta := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		if _, err := ClaytonHFn(theta); !errors.Is(err, ErrCopulaHFnInvalidTheta) {
			t.Errorf("ClaytonHFn(%v): expected ErrCopulaHFnInvalidTheta, got %v", theta, err)
		}
	}
}

func TestGumbelHFn_RejectsInvalidTheta(t *testing.T) {
	for _, theta := range []float64{0.5, 0, -1, math.NaN(), math.Inf(1)} {
		if _, err := GumbelHFn(theta); !errors.Is(err, ErrCopulaHFnInvalidTheta) {
			t.Errorf("GumbelHFn(%v): expected ErrCopulaHFnInvalidTheta, got %v", theta, err)
		}
	}
}

// numericalHFromCDF computes h(u | v; θ) = ∂C(u, v; θ) / ∂v via central
// finite difference. Used to verify the closed-form h-function.
func numericalHFromCDF(cdf func(u, v float64) float64, u, v, dv float64) float64 {
	return (cdf(u, v+dv) - cdf(u, v-dv)) / (2.0 * dv)
}

func TestClaytonHFn_MatchesNumericalDerivative(t *testing.T) {
	const dv = 1e-5
	const tol = 1e-3 // central-diff is O(dv²); 1e-3 is generous

	for _, theta := range []float64{0.5, 1.0, 2.0, 5.0} {
		hfn, err := ClaytonHFn(theta)
		if err != nil {
			t.Fatalf("ClaytonHFn(%v) unexpected err: %v", theta, err)
		}
		cdf, err := ClaytonCopulaCDFFn(theta)
		if err != nil {
			t.Fatalf("ClaytonCopulaCDFFn(%v) unexpected err: %v", theta, err)
		}

		for _, u := range []float64{0.2, 0.5, 0.8} {
			for _, v := range []float64{0.2, 0.5, 0.8} {
				got := hfn(u, v)
				want := numericalHFromCDF(cdf, u, v, dv)
				if math.Abs(got-want) > tol {
					t.Errorf("ClaytonHFn(θ=%v)(u=%v, v=%v): got %v, want %v (Δ=%v)",
						theta, u, v, got, want, got-want)
				}
			}
		}
	}
}

func TestGumbelHFn_MatchesNumericalDerivative(t *testing.T) {
	const dv = 1e-5
	const tol = 1e-3

	for _, theta := range []float64{1.0, 1.5, 2.5, 4.0} {
		hfn, err := GumbelHFn(theta)
		if err != nil {
			t.Fatalf("GumbelHFn(%v) unexpected err: %v", theta, err)
		}
		cdf, err := GumbelCopulaCDFFn(theta)
		if err != nil {
			t.Fatalf("GumbelCopulaCDFFn(%v) unexpected err: %v", theta, err)
		}

		for _, u := range []float64{0.2, 0.5, 0.8} {
			for _, v := range []float64{0.2, 0.5, 0.8} {
				got := hfn(u, v)
				want := numericalHFromCDF(cdf, u, v, dv)
				if math.Abs(got-want) > tol {
					t.Errorf("GumbelHFn(θ=%v)(u=%v, v=%v): got %v, want %v (Δ=%v)",
						theta, u, v, got, want, got-want)
				}
			}
		}
	}
}

func TestGumbelHFn_IndependenceCornerCase(t *testing.T) {
	// At θ = 1 the Gumbel copula is the independence copula and
	// h(u | v) = u for all v.
	hfn, err := GumbelHFn(1.0)
	if err != nil {
		t.Fatalf("GumbelHFn(1.0) unexpected err: %v", err)
	}
	for _, u := range []float64{0.1, 0.3, 0.7, 0.9} {
		for _, v := range []float64{0.1, 0.5, 0.9} {
			if got := hfn(u, v); math.Abs(got-u) > 1e-12 {
				t.Errorf("GumbelHFn(1.0)(u=%v, v=%v) = %v, want %v", u, v, got, u)
			}
		}
	}
}

func TestClaytonHFn_BoundaryClamps(t *testing.T) {
	hfn, err := ClaytonHFn(2.0)
	if err != nil {
		t.Fatalf("ClaytonHFn(2.0) unexpected err: %v", err)
	}
	if got := hfn(0, 0.5); got != 0 {
		t.Errorf("h(0|0.5) = %v, want 0", got)
	}
	if got := hfn(1, 0.5); got != 1 {
		t.Errorf("h(1|0.5) = %v, want 1", got)
	}
	if got := hfn(0.5, 0); got != 0 {
		t.Errorf("h(0.5|0) = %v, want 0", got)
	}
	if got := hfn(0.5, 1); got != 0.5 {
		t.Errorf("h(0.5|1) = %v, want 0.5", got)
	}
}

func TestGumbelHFn_BoundaryClamps(t *testing.T) {
	hfn, err := GumbelHFn(2.0)
	if err != nil {
		t.Fatalf("GumbelHFn(2.0) unexpected err: %v", err)
	}
	if got := hfn(0, 0.5); got != 0 {
		t.Errorf("h(0|0.5) = %v, want 0", got)
	}
	if got := hfn(1, 0.5); got != 1 {
		t.Errorf("h(1|0.5) = %v, want 1", got)
	}
}

func TestHFnForFamily_DispatchesCorrectly(t *testing.T) {
	cl, err := HFnForFamily(FamilyClayton, 2.0)
	if err != nil {
		t.Fatalf("HFnForFamily(Clayton, 2.0) err: %v", err)
	}
	gu, err := HFnForFamily(FamilyGumbel, 2.0)
	if err != nil {
		t.Fatalf("HFnForFamily(Gumbel, 2.0) err: %v", err)
	}
	// Sanity: distinct families produce distinct outputs.
	if cl(0.5, 0.5) == gu(0.5, 0.5) {
		t.Errorf("Clayton + Gumbel h-functions produced same value at (0.5, 0.5); families should differ")
	}
}

func TestHFnForFamily_RejectsUnknownFamily(t *testing.T) {
	if _, err := HFnForFamily(ArchimedeanFamily(99), 2.0); !errors.Is(err, ErrCopulaHFnInvalidTheta) {
		t.Errorf("HFnForFamily(99, 2.0): expected ErrCopulaHFnInvalidTheta, got %v", err)
	}
}

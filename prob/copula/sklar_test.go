package copula

import (
	"math"
	"testing"

	"github.com/davly/reality/prob"
)

// =========================================================================
// Sklar reconstruction — the full pipeline marginals + copula -> joint
// =========================================================================

func TestSklarJointFromMarginals_BivariateNormalMarginals(t *testing.T) {
	// F_1, F_2 are standard-normal CDFs; copula is bivariate Gaussian
	// at rho.  By Sklar's theorem the joint CDF is bivariate normal.
	// Verified at several (x1, x2, r) tuples by closed-form
	// BivariateNormalCDF.
	margs := []MarginalCDF{
		func(x float64) float64 { return prob.NormalCDF(x, 0, 1) },
		func(x float64) float64 { return prob.NormalCDF(x, 0, 1) },
	}
	for _, r := range []float64{-0.5, 0, 0.4, 0.8} {
		sigma := [][]float64{{1, r}, {r, 1}}
		joint := SklarJointFromMarginals(margs, GaussianCopulaCDFFn(sigma))
		x := []float64{0.5, -0.3}
		got, err := joint(x)
		if err != nil {
			t.Fatal(err)
		}
		want := BivariateNormalCDF(x[0], x[1], r)
		if math.Abs(got-want) > 1e-9 {
			t.Errorf("Sklar normal at r=%v: got %v, want %v",
				r, got, want)
		}
	}
}

func TestSklarJointFromMarginals_NonNormalMarginals(t *testing.T) {
	// Lognormal marginal #1, exponential marginal #2, Gaussian copula.
	// We verify the joint CDF satisfies the Sklar identity and that
	// independence (rho=0) reduces to product of marginals.
	logn := func(x float64) float64 {
		if x <= 0 {
			return 0
		}
		return prob.NormalCDF(math.Log(x), 0, 1)
	}
	exp1 := func(x float64) float64 {
		if x <= 0 {
			return 0
		}
		return 1 - math.Exp(-x)
	}
	margs := []MarginalCDF{logn, exp1}
	sigma := [][]float64{{1, 0}, {0, 1}}
	joint := SklarJointFromMarginals(margs, GaussianCopulaCDFFn(sigma))
	x := []float64{2.0, 1.5}
	got, err := joint(x)
	if err != nil {
		t.Fatal(err)
	}
	want := logn(x[0]) * exp1(x[1])
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("Sklar non-normal independent: got %v, want %v", got, want)
	}
}

func TestSklarJointFromMarginals_DimensionMismatch(t *testing.T) {
	margs := []MarginalCDF{
		func(x float64) float64 { return prob.NormalCDF(x, 0, 1) },
		func(x float64) float64 { return prob.NormalCDF(x, 0, 1) },
	}
	sigma := [][]float64{{1, 0.3}, {0.3, 1}}
	joint := SklarJointFromMarginals(margs, GaussianCopulaCDFFn(sigma))
	if _, err := joint([]float64{0.5}); err == nil {
		t.Error("expected dimension mismatch on len(x)=1")
	}
}

// =========================================================================
// Trivariate Sklar — three-module Solvency-II-shape aggregation
// =========================================================================

func TestSklarJointFromMarginals_TrivariateSolvencyShape(t *testing.T) {
	// Three "sub-module" marginals — Gamma-shaped via prob's existing
	// helpers, but for v1 we use Normal marginals as a stand-in (the
	// real Solvency II compound-Poisson-Tweedie marginals land in v2).
	// The point of this test is the *Sklar shape* — wiring three
	// marginals into a joint CDF via a 3x3 EIOPA-style correlation.
	mu := []float64{0, 0, 0}
	margs := make([]MarginalCDF, 3)
	for i := range margs {
		mi := mu[i]
		margs[i] = func(x float64) float64 { return prob.NormalCDF(x, mi, 1) }
	}
	sigma := [][]float64{
		{1.00, 0.25, 0.25},
		{0.25, 1.00, 0.00},
		{0.25, 0.00, 1.00},
	}
	joint := SklarJointFromMarginals(margs, GaussianCopulaCDFFn(sigma))

	// At a high-quantile point (each module independently at 95%
	// quantile), the joint CDF must be in [independent product,
	// min-quantile].
	q95 := prob.NormalQuantile(0.95, 0, 1)
	x := []float64{q95, q95, q95}
	got, err := joint(x)
	if err != nil {
		t.Fatal(err)
	}
	indep := 0.95 * 0.95 * 0.95
	if got < indep-1e-3 {
		t.Errorf("3-module joint: got %v < independent %v (positive correl should raise)",
			got, indep)
	}
	if got > 0.95+1e-9 {
		t.Errorf("3-module joint: got %v > min-marginal 0.95 (impossible upper bound)",
			got)
	}
}

// =========================================================================
// Frechet-Hoeffding bounds — every copula must satisfy these
// =========================================================================
//
// For any 2D copula C(u, v):
//   max(u + v - 1, 0) <= C(u, v) <= min(u, v).
//
// We verify the bounds at a grid of (u, v, rho) under our
// Gaussian-copula impl.

func TestGaussianCopulaCDF_FrechetHoeffdingBounds(t *testing.T) {
	for _, u1 := range []float64{0.1, 0.3, 0.5, 0.7, 0.9} {
		for _, u2 := range []float64{0.1, 0.3, 0.5, 0.7, 0.9} {
			for _, r := range []float64{-0.7, -0.3, 0, 0.3, 0.7} {
				sigma := [][]float64{{1, r}, {r, 1}}
				got, err := GaussianCopulaCDF([]float64{u1, u2}, sigma)
				if err != nil {
					t.Fatal(err)
				}
				lo := math.Max(u1+u2-1, 0)
				hi := math.Min(u1, u2)
				if got < lo-1e-7 {
					t.Errorf("FH lower: u=(%v, %v) r=%v: got %v < %v",
						u1, u2, r, got, lo)
				}
				if got > hi+1e-7 {
					t.Errorf("FH upper: u=(%v, %v) r=%v: got %v > %v",
						u1, u2, r, got, hi)
				}
			}
		}
	}
}

// =========================================================================
// Cross-check: GaussianCopulaCDFFn / StudentTCopulaCDFFn closures
// =========================================================================

func TestGaussianCopulaCDFFn_MatchesDirectCall(t *testing.T) {
	sigma := [][]float64{{1, 0.4}, {0.4, 1}}
	u := []float64{0.7, 0.3}
	direct, _ := GaussianCopulaCDF(u, sigma)
	fn := GaussianCopulaCDFFn(sigma)
	closure, _ := fn(u)
	if direct != closure {
		t.Errorf("closure mismatch: %v vs %v", direct, closure)
	}
}

func TestStudentTCopulaCDFFn_MatchesDirectCall(t *testing.T) {
	sigma := [][]float64{{1, 0.4}, {0.4, 1}}
	u := []float64{0.7, 0.3}
	direct, _ := StudentTCopulaCDF(u, sigma, 5)
	fn := StudentTCopulaCDFFn(sigma, 5)
	closure, _ := fn(u)
	if direct != closure {
		t.Errorf("closure mismatch: %v vs %v", direct, closure)
	}
}

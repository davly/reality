package copula

import (
	"errors"
	"math"
	"testing"
)

// =========================================================================
// Theta-from-Kendall-tau inversions — closed-form, exact arithmetic
// =========================================================================

func TestThetaFromKendallTau_Clayton_HappyPath(t *testing.T) {
	// θ = 2τ / (1 - τ). At τ = 0.5 → θ = 2.0.
	got, err := ThetaFromKendallTau(0.5, FamilyClayton)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if math.Abs(got-2.0) > 1e-12 {
		t.Errorf("Clayton(τ=0.5) = %v, want 2.0", got)
	}
}

func TestThetaFromKendallTau_Gumbel_HappyPath(t *testing.T) {
	// θ = 1 / (1 - τ). At τ = 0.5 → θ = 2.0.
	got, err := ThetaFromKendallTau(0.5, FamilyGumbel)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if math.Abs(got-2.0) > 1e-12 {
		t.Errorf("Gumbel(τ=0.5) = %v, want 2.0", got)
	}
}

func TestThetaFromKendallTau_Clayton_RejectsNonPositiveTau(t *testing.T) {
	for _, tau := range []float64{-0.1, 0.0} {
		_, err := ThetaFromKendallTau(tau, FamilyClayton)
		if !errors.Is(err, ErrArchimedeanThetaOutOfRange) {
			t.Errorf("Clayton τ=%v: expected ThetaOutOfRange, got %v", tau, err)
		}
	}
}

func TestThetaFromKendallTau_Gumbel_AcceptsZeroTau_GivesIndependence(t *testing.T) {
	// At τ = 0, Gumbel θ = 1 = independence copula.
	got, err := ThetaFromKendallTau(0.0, FamilyGumbel)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if math.Abs(got-1.0) > 1e-12 {
		t.Errorf("Gumbel(τ=0) = %v, want 1.0 (independence)", got)
	}
}

func TestThetaFromKendallTau_BothFamilies_RejectTauOne(t *testing.T) {
	for _, family := range []ArchimedeanFamily{FamilyClayton, FamilyGumbel} {
		_, err := ThetaFromKendallTau(1.0, family)
		if !errors.Is(err, ErrArchimedeanThetaOutOfRange) {
			t.Errorf("family=%v τ=1: expected ThetaOutOfRange, got %v", family, err)
		}
	}
}

func TestThetaFromKendallTau_NaN_Rejected(t *testing.T) {
	for _, family := range []ArchimedeanFamily{FamilyClayton, FamilyGumbel} {
		_, err := ThetaFromKendallTau(math.NaN(), family)
		if !errors.Is(err, ErrArchimedeanThetaOutOfRange) {
			t.Errorf("family=%v τ=NaN: expected ThetaOutOfRange, got %v", family, err)
		}
	}
}

// =========================================================================
// Clayton CDF closure — closed-form, finite arithmetic
// =========================================================================

func TestClaytonCopulaCDFFn_RejectsNonPositiveTheta(t *testing.T) {
	for _, theta := range []float64{-1.0, 0.0, math.NaN(), math.Inf(1)} {
		_, err := ClaytonCopulaCDFFn(theta)
		if !errors.Is(err, ErrArchimedeanInvalidTheta) {
			t.Errorf("Clayton θ=%v: expected InvalidTheta, got %v", theta, err)
		}
	}
}

func TestClaytonCopulaCDFFn_DiagonalIsValidCDF(t *testing.T) {
	cdf, err := ClaytonCopulaCDFFn(2.0)
	if err != nil {
		t.Fatal(err)
	}
	// On the diagonal, C(t,t) ∈ [0, t] (Fréchet-Hoeffding upper bound).
	for _, u := range []float64{0.1, 0.3, 0.5, 0.7, 0.9} {
		got := cdf(u, u)
		if got < 0 || got > u {
			t.Errorf("Clayton C(%v,%v) = %v not in [0, %v]", u, u, got, u)
		}
	}
}

func TestClaytonCopulaCDFFn_BoundaryUnit_PassesThroughOther(t *testing.T) {
	// C(1, v) = v and C(u, 1) = u (uniform marginals).
	cdf, _ := ClaytonCopulaCDFFn(2.0)
	if got := cdf(1.0, 0.4); math.Abs(got-0.4) > 1e-12 {
		t.Errorf("C(1, 0.4) = %v, want 0.4", got)
	}
	if got := cdf(0.4, 1.0); math.Abs(got-0.4) > 1e-12 {
		t.Errorf("C(0.4, 1) = %v, want 0.4", got)
	}
}

func TestClaytonCopulaCDFFn_BoundaryZero_GivesZero(t *testing.T) {
	cdf, _ := ClaytonCopulaCDFFn(2.0)
	if got := cdf(0.0, 0.5); got != 0 {
		t.Errorf("C(0, 0.5) = %v, want 0", got)
	}
}

func TestClaytonCopulaCDFFn_HighThetaApproachesComonotone(t *testing.T) {
	// As θ → ∞, C(u,v) → min(u,v).
	cdf, _ := ClaytonCopulaCDFFn(50.0)
	for _, pair := range []struct{ u, v, want float64 }{
		{0.3, 0.7, 0.3},
		{0.5, 0.5, 0.5},
		{0.9, 0.2, 0.2},
	} {
		got := cdf(pair.u, pair.v)
		if math.Abs(got-pair.want) > 0.05 {
			t.Errorf("Clayton θ=50 C(%v,%v) = %v, want close to min = %v",
				pair.u, pair.v, got, pair.want)
		}
	}
}

func TestClaytonCopulaCDFFn_LowThetaApproachesIndependence(t *testing.T) {
	// As θ → 0, C(u,v) → uv (independence).
	cdf, _ := ClaytonCopulaCDFFn(0.001)
	for _, pair := range []struct{ u, v float64 }{
		{0.3, 0.7}, {0.4, 0.5}, {0.2, 0.8},
	} {
		got := cdf(pair.u, pair.v)
		want := pair.u * pair.v
		if math.Abs(got-want) > 0.01 {
			t.Errorf("Clayton θ=0.001 C(%v,%v) = %v, want close to %v",
				pair.u, pair.v, got, want)
		}
	}
}

// =========================================================================
// Gumbel CDF closure — closed-form, finite arithmetic
// =========================================================================

func TestGumbelCopulaCDFFn_RejectsThetaBelowOne(t *testing.T) {
	for _, theta := range []float64{0.5, 0.0, -1.0, math.NaN(), math.Inf(1)} {
		_, err := GumbelCopulaCDFFn(theta)
		if !errors.Is(err, ErrArchimedeanInvalidTheta) {
			t.Errorf("Gumbel θ=%v: expected InvalidTheta, got %v", theta, err)
		}
	}
}

func TestGumbelCopulaCDFFn_AcceptsThetaOne_GivesIndependence(t *testing.T) {
	// θ = 1 should give C(u,v) = uv.
	cdf, err := GumbelCopulaCDFFn(1.0)
	if err != nil {
		t.Fatal(err)
	}
	for _, pair := range []struct{ u, v float64 }{
		{0.3, 0.7}, {0.4, 0.5}, {0.2, 0.8},
	} {
		got := cdf(pair.u, pair.v)
		want := pair.u * pair.v
		if math.Abs(got-want) > 1e-9 {
			t.Errorf("Gumbel θ=1 C(%v,%v) = %v, want %v", pair.u, pair.v, got, want)
		}
	}
}

func TestGumbelCopulaCDFFn_DiagonalIsValidCDF(t *testing.T) {
	cdf, _ := GumbelCopulaCDFFn(2.0)
	for _, u := range []float64{0.1, 0.3, 0.5, 0.7, 0.9} {
		got := cdf(u, u)
		if got < 0 || got > u {
			t.Errorf("Gumbel C(%v,%v) = %v not in [0, %v]", u, u, got, u)
		}
	}
}

func TestGumbelCopulaCDFFn_HighThetaApproachesComonotone(t *testing.T) {
	cdf, _ := GumbelCopulaCDFFn(50.0)
	for _, pair := range []struct{ u, v, want float64 }{
		{0.3, 0.7, 0.3},
		{0.5, 0.5, 0.5},
		{0.9, 0.2, 0.2},
	} {
		got := cdf(pair.u, pair.v)
		if math.Abs(got-pair.want) > 0.05 {
			t.Errorf("Gumbel θ=50 C(%v,%v) = %v, want close to min = %v",
				pair.u, pair.v, got, pair.want)
		}
	}
}

func TestGumbelCopulaCDFFn_BoundaryUnit_PassesThroughOther(t *testing.T) {
	cdf, _ := GumbelCopulaCDFFn(2.0)
	if got := cdf(1.0, 0.4); math.Abs(got-0.4) > 1e-12 {
		t.Errorf("C(1, 0.4) = %v, want 0.4", got)
	}
}

// =========================================================================
// Tail-dependence coefficients
// =========================================================================

func TestClaytonLowerTailDependence_AtThetaTwo(t *testing.T) {
	// λ_L = 2^(-1/2) ≈ 0.7071.
	got := ClaytonLowerTailDependence(2.0)
	want := math.Pow(2.0, -0.5)
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("λ_L(θ=2) = %v, want %v", got, want)
	}
}

func TestClaytonLowerTailDependence_ApproachesOneAsThetaGrows(t *testing.T) {
	for _, theta := range []float64{1.0, 5.0, 50.0, 1000.0} {
		got := ClaytonLowerTailDependence(theta)
		if got < 0 || got > 1 {
			t.Errorf("Clayton θ=%v: λ_L=%v out of [0,1]", theta, got)
		}
	}
	// At θ = 1000 should be very close to 1.
	if got := ClaytonLowerTailDependence(1000.0); got < 0.99 {
		t.Errorf("θ=1000: λ_L=%v should be > 0.99", got)
	}
}

func TestClaytonLowerTailDependence_NonPositiveThetaReturnsZero(t *testing.T) {
	for _, theta := range []float64{0.0, -1.0, math.NaN(), math.Inf(1)} {
		if got := ClaytonLowerTailDependence(theta); got != 0 {
			t.Errorf("Clayton θ=%v: λ_L=%v, want 0", theta, got)
		}
	}
}

func TestGumbelUpperTailDependence_AtThetaTwo(t *testing.T) {
	// λ_U = 2 - 2^(1/2) ≈ 0.5858.
	got := GumbelUpperTailDependence(2.0)
	want := 2.0 - math.Sqrt(2.0)
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("λ_U(θ=2) = %v, want %v", got, want)
	}
}

func TestGumbelUpperTailDependence_AtThetaOneIsZero(t *testing.T) {
	// At θ=1 (independence), λ_U = 0.
	if got := GumbelUpperTailDependence(1.0); math.Abs(got) > 1e-12 {
		t.Errorf("Gumbel θ=1: λ_U=%v, want 0", got)
	}
}

func TestGumbelUpperTailDependence_ThetaBelowOneReturnsZero(t *testing.T) {
	for _, theta := range []float64{0.5, 0.0, -1.0, math.NaN(), math.Inf(1)} {
		if got := GumbelUpperTailDependence(theta); got != 0 {
			t.Errorf("Gumbel θ=%v: λ_U=%v, want 0", theta, got)
		}
	}
}

// =========================================================================
// Round-trip: tau → theta → CDF
// =========================================================================

func TestThetaFromKendallTau_ToCDF_Clayton_RoundTrip(t *testing.T) {
	// τ=0.4 → θ=4/3; resulting CDF must be valid + monotone.
	theta, err := ThetaFromKendallTau(0.4, FamilyClayton)
	if err != nil {
		t.Fatal(err)
	}
	cdf, err := ClaytonCopulaCDFFn(theta)
	if err != nil {
		t.Fatal(err)
	}
	// Monotone in u: C(0.3, 0.5) < C(0.4, 0.5) < C(0.5, 0.5).
	a := cdf(0.3, 0.5)
	b := cdf(0.4, 0.5)
	c := cdf(0.5, 0.5)
	if !(a <= b && b <= c) {
		t.Errorf("not monotone in u: %v %v %v", a, b, c)
	}
}

func TestThetaFromKendallTau_ToCDF_Gumbel_RoundTrip(t *testing.T) {
	theta, err := ThetaFromKendallTau(0.4, FamilyGumbel)
	if err != nil {
		t.Fatal(err)
	}
	cdf, err := GumbelCopulaCDFFn(theta)
	if err != nil {
		t.Fatal(err)
	}
	a := cdf(0.3, 0.5)
	b := cdf(0.4, 0.5)
	c := cdf(0.5, 0.5)
	if !(a <= b && b <= c) {
		t.Errorf("not monotone in u: %v %v %v", a, b, c)
	}
}

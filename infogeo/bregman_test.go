package infogeo

import (
	"math"
	"testing"
)

// =========================================================================
// SquaredEuclidean
// =========================================================================

func TestSquaredEuclidean_Symmetric(t *testing.T) {
	x := []float64{1.0, 2.0, 3.0}
	y := []float64{4.0, 0.0, -1.0}
	a, _ := SquaredEuclidean(x, y)
	b, _ := SquaredEuclidean(y, x)
	if a != b {
		t.Errorf("SqEuclidean asymmetric: %v vs %v", a, b)
	}
	want := 0.5 * (9.0 + 4.0 + 16.0) // 14.5
	if math.Abs(a-want) > 1e-12 {
		t.Errorf("SqEuclidean = %v, want %v", a, want)
	}
}

// =========================================================================
// GeneralisedKL
// =========================================================================

func TestGeneralisedKL_ZeroAtEqual(t *testing.T) {
	x := []float64{1.0, 2.0, 3.0}
	got, err := GeneralisedKL(x, x)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got) > 1e-12 {
		t.Errorf("GenKL(x, x) = %v, want 0", got)
	}
}

func TestGeneralisedKL_ReducesToKLOnSimplex(t *testing.T) {
	p := []float64{0.4, 0.6}
	q := []float64{0.5, 0.5}
	gen, _ := GeneralisedKL(p, q)
	kl, _ := KL(p, q)
	if math.Abs(gen-kl) > 1e-12 {
		t.Errorf("GenKL = %v, KL = %v — should agree on simplex", gen, kl)
	}
}

func TestGeneralisedKL_NonProbability(t *testing.T) {
	// x = (3, 1), y = (2, 2): x*log(x/y) - x + y per coord
	// 3*log(1.5) - 3 + 2  +  1*log(0.5) - 1 + 2
	x := []float64{3.0, 1.0}
	y := []float64{2.0, 2.0}
	got, _ := GeneralisedKL(x, y)
	want := 3*math.Log(1.5) - 1.0 + math.Log(0.5) + 1.0
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("GenKL = %v, want %v", got, want)
	}
	// Non-negativity is the defining Bregman property.
	if got < 0 {
		t.Errorf("GenKL = %v, want >= 0", got)
	}
}

// =========================================================================
// ItakuraSaito
// =========================================================================

func TestItakuraSaito_ZeroAtEqual(t *testing.T) {
	x := []float64{0.5, 1.0, 2.0}
	got, err := ItakuraSaito(x, x)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got) > 1e-12 {
		t.Errorf("IS(x, x) = %v, want 0", got)
	}
}

func TestItakuraSaito_ScaleInvariant(t *testing.T) {
	// IS is scale-invariant in the SHARED scale: IS(c*x, c*y) = IS(x, y).
	x := []float64{1.0, 2.0, 4.0}
	y := []float64{2.0, 1.0, 1.0}
	a, _ := ItakuraSaito(x, y)
	scale := 7.5
	xs := []float64{scale * x[0], scale * x[1], scale * x[2]}
	ys := []float64{scale * y[0], scale * y[1], scale * y[2]}
	b, _ := ItakuraSaito(xs, ys)
	if math.Abs(a-b) > 1e-12 {
		t.Errorf("IS not scale-invariant: %v vs %v", a, b)
	}
}

func TestItakuraSaito_RejectsZeroOrNegative(t *testing.T) {
	if _, err := ItakuraSaito([]float64{1.0, 0.0}, []float64{1.0, 1.0}); err == nil {
		t.Error("zero in x should error")
	}
	if _, err := ItakuraSaito([]float64{1.0, 1.0}, []float64{-1.0, 1.0}); err == nil {
		t.Error("negative in y should error")
	}
}

// =========================================================================
// Mahalanobis
// =========================================================================

func TestMahalanobisSquared_IdentityMatchesEuclidean(t *testing.T) {
	x := []float64{1.0, 2.0, 3.0}
	y := []float64{0.0, 0.0, 0.0}
	I := []float64{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	}
	got, err := MahalanobisSquared(x, y, I)
	if err != nil {
		t.Fatal(err)
	}
	want := 1.0 + 4.0 + 9.0
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("Mahalanobis with I = %v, want %v", got, want)
	}
}

func TestMahalanobisSquared_DiagonalScaling(t *testing.T) {
	x := []float64{1.0, 1.0}
	y := []float64{0.0, 0.0}
	M := []float64{
		2.0, 0.0,
		0.0, 8.0,
	}
	got, _ := MahalanobisSquared(x, y, M)
	want := 2.0*1.0 + 8.0*1.0 // 10
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("Mahalanobis = %v, want %v", got, want)
	}
}

// =========================================================================
// Generic Bregman framework (verify it agrees with closed-form)
// =========================================================================

func TestBregman_SquaredEuclideanAgreesWithGeneric(t *testing.T) {
	x := []float64{1.0, 2.0, 3.0}
	y := []float64{0.5, 1.5, 2.5}
	gen := BregmanGen{
		Phi: func(v []float64) float64 {
			var s float64
			for _, vi := range v {
				s += vi * vi
			}
			return 0.5 * s
		},
		GradPhi: func(v, out []float64) {
			copy(out, v)
		},
	}
	g, err := Bregman(gen, x, y)
	if err != nil {
		t.Fatal(err)
	}
	c, _ := SquaredEuclidean(x, y)
	if math.Abs(g-c) > 1e-12 {
		t.Errorf("Bregman generic %v != closed-form %v", g, c)
	}
}

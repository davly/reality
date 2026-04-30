package infogeo

import (
	"math"
	"testing"
)

// =========================================================================
// KL
// =========================================================================

func TestKL_ZeroOnEqualDistributions(t *testing.T) {
	p := []float64{0.25, 0.25, 0.25, 0.25}
	got, err := KL(p, p)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got) > 1e-12 {
		t.Errorf("KL(p, p) = %v, want 0", got)
	}
}

func TestKL_KnownValue(t *testing.T) {
	// KL of fair vs (0.7, 0.3) coin
	p := []float64{0.5, 0.5}
	q := []float64{0.7, 0.3}
	got, err := KL(p, q)
	if err != nil {
		t.Fatal(err)
	}
	want := 0.5*math.Log(0.5/0.7) + 0.5*math.Log(0.5/0.3)
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("KL = %v, want %v", got, want)
	}
}

func TestKL_ZeroInQAndPositiveInP_IsInfinite(t *testing.T) {
	p := []float64{0.5, 0.5}
	q := []float64{1.0, 0.0}
	got, err := KL(p, q)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(got, 1) {
		t.Errorf("KL = %v, want +Inf", got)
	}
}

func TestKL_ZeroInPIgnored(t *testing.T) {
	p := []float64{0.0, 1.0}
	q := []float64{0.5, 0.5}
	got, err := KL(p, q)
	if err != nil {
		t.Fatal(err)
	}
	want := math.Log(2.0)
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("KL = %v, want %v", got, want)
	}
}

func TestKL_RejectsBadInputs(t *testing.T) {
	if _, err := KL([]float64{0.5, 0.5}, []float64{0.5, 0.5, 0.0}); err == nil {
		t.Error("length mismatch should error")
	}
	if _, err := KL([]float64{0.5, 0.4}, []float64{0.5, 0.5}); err == nil {
		t.Error("non-summing-to-1 should error")
	}
	if _, err := KL([]float64{-0.1, 1.1}, []float64{0.5, 0.5}); err == nil {
		t.Error("negative entry should error")
	}
}

// =========================================================================
// JS
// =========================================================================

func TestJS_Symmetric(t *testing.T) {
	p := []float64{0.7, 0.2, 0.1}
	q := []float64{0.3, 0.5, 0.2}
	a, _ := JS(p, q)
	b, _ := JS(q, p)
	if math.Abs(a-b) > 1e-12 {
		t.Errorf("JS asymmetric: %v vs %v", a, b)
	}
}

func TestJS_BoundedByLog2(t *testing.T) {
	p := []float64{1.0, 0.0}
	q := []float64{0.0, 1.0}
	got, _ := JS(p, q)
	want := math.Log(2.0)
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("JS of disjoint = %v, want log 2 = %v", got, want)
	}
}

// =========================================================================
// Total variation, Hellinger, Chi-squared
// =========================================================================

func TestTotalVariation_RangeAndBoundary(t *testing.T) {
	p := []float64{1.0, 0.0}
	q := []float64{0.0, 1.0}
	tv, _ := TotalVariation(p, q)
	if tv != 1.0 {
		t.Errorf("TV(disjoint) = %v, want 1", tv)
	}
	tv2, _ := TotalVariation(p, p)
	if tv2 != 0 {
		t.Errorf("TV(p, p) = %v, want 0", tv2)
	}
}

func TestHellinger_RangeAndBoundary(t *testing.T) {
	p := []float64{1.0, 0.0}
	q := []float64{0.0, 1.0}
	h, _ := Hellinger(p, q)
	if math.Abs(h-1.0) > 1e-12 {
		t.Errorf("Hellinger(disjoint) = %v, want 1", h)
	}
	h2, _ := Hellinger(p, p)
	if math.Abs(h2) > 1e-12 {
		t.Errorf("Hellinger(p, p) = %v, want 0", h2)
	}
}

func TestChiSquared_KnownValue(t *testing.T) {
	p := []float64{0.4, 0.6}
	q := []float64{0.5, 0.5}
	got, _ := ChiSquared(p, q)
	want := 0.01/0.5 + 0.01/0.5 // 0.04
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("Chi^2 = %v, want %v", got, want)
	}
}

func TestChiSquared_ZeroQ_PositiveP_IsInfinite(t *testing.T) {
	p := []float64{0.5, 0.5}
	q := []float64{1.0, 0.0}
	got, _ := ChiSquared(p, q)
	if !math.IsInf(got, 1) {
		t.Errorf("Chi^2 = %v, want +Inf", got)
	}
}

// =========================================================================
// Renyi
// =========================================================================

func TestRenyi_LimitMatchesKL(t *testing.T) {
	// Renyi-alpha -> KL as alpha -> 1.
	p := []float64{0.6, 0.3, 0.1}
	q := []float64{0.4, 0.4, 0.2}
	kl, _ := KL(p, q)
	r, _ := Renyi(p, q, 1.001)
	if math.Abs(r-kl) > 1e-2 {
		t.Errorf("Renyi(alpha=1.001) = %v, want close to KL = %v", r, kl)
	}
}

func TestRenyi_RejectsAlpha1(t *testing.T) {
	p := []float64{0.5, 0.5}
	q := []float64{0.7, 0.3}
	if _, err := Renyi(p, q, 1.0); err == nil {
		t.Error("alpha=1 should error (use KL)")
	}
}

func TestRenyi_NonNegative(t *testing.T) {
	p := []float64{0.6, 0.4}
	q := []float64{0.4, 0.6}
	for _, alpha := range []float64{0.5, 2.0, 5.0} {
		got, err := Renyi(p, q, alpha)
		if err != nil {
			t.Fatal(err)
		}
		if got < -1e-12 {
			t.Errorf("Renyi(alpha=%v) = %v, want >= 0", alpha, got)
		}
	}
}

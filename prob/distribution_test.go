package prob

import (
	"math"
	"testing"
)

// =========================================================================
// Distribution interface compliance
// =========================================================================

func TestBetaDist_ImplementsDistribution(t *testing.T) {
	var _ Distribution = &BetaDist{}
}

func TestNormalDist_ImplementsDistribution(t *testing.T) {
	var _ Distribution = &NormalDist{}
}

func TestExponentialDist_ImplementsDistribution(t *testing.T) {
	var _ Distribution = &ExponentialDist{}
}

func TestUniformDist_ImplementsDistribution(t *testing.T) {
	var _ Distribution = &UniformDist{}
}

// =========================================================================
// Constructor validation
// =========================================================================

func TestNewBetaDist_InvalidParams(t *testing.T) {
	if d := NewBetaDist(0, 1); d != nil {
		t.Error("NewBetaDist(0, 1) should return nil")
	}
	if d := NewBetaDist(1, -1); d != nil {
		t.Error("NewBetaDist(1, -1) should return nil")
	}
}

func TestNewNormalDist_InvalidParams(t *testing.T) {
	if d := NewNormalDist(0, 0); d != nil {
		t.Error("NewNormalDist(0, 0) should return nil")
	}
	if d := NewNormalDist(0, -1); d != nil {
		t.Error("NewNormalDist(0, -1) should return nil")
	}
}

func TestNewExponentialDist_InvalidParams(t *testing.T) {
	if d := NewExponentialDist(0); d != nil {
		t.Error("NewExponentialDist(0) should return nil")
	}
	if d := NewExponentialDist(-1); d != nil {
		t.Error("NewExponentialDist(-1) should return nil")
	}
}

func TestNewUniformDist_InvalidParams(t *testing.T) {
	if d := NewUniformDist(1, 1); d != nil {
		t.Error("NewUniformDist(1, 1) should return nil for a=b")
	}
	if d := NewUniformDist(2, 1); d != nil {
		t.Error("NewUniformDist(2, 1) should return nil for a>b")
	}
}

// =========================================================================
// BetaDist via interface
// =========================================================================

func TestBetaDist_PDF_Symmetric(t *testing.T) {
	d := NewBetaDist(2, 2)
	// Beta(2,2) is symmetric around 0.5.
	pdf05 := d.PDF(0.5)
	if pdf05 < 1.4 || pdf05 > 1.6 {
		t.Errorf("Beta(2,2).PDF(0.5) = %f, want ~1.5", pdf05)
	}
	// Symmetry: PDF(0.3) == PDF(0.7)
	if math.Abs(d.PDF(0.3)-d.PDF(0.7)) > 1e-10 {
		t.Errorf("Beta(2,2) symmetry violated: PDF(0.3)=%f, PDF(0.7)=%f", d.PDF(0.3), d.PDF(0.7))
	}
}

func TestBetaDist_CDF_Boundaries(t *testing.T) {
	d := NewBetaDist(1, 1)
	// Beta(1,1) is Uniform(0,1) => CDF(x) = x.
	if math.Abs(d.CDF(0.0)-0.0) > 1e-10 {
		t.Errorf("Beta(1,1).CDF(0) = %f, want 0", d.CDF(0.0))
	}
	if math.Abs(d.CDF(1.0)-1.0) > 1e-10 {
		t.Errorf("Beta(1,1).CDF(1) = %f, want 1", d.CDF(1.0))
	}
	if math.Abs(d.CDF(0.5)-0.5) > 1e-6 {
		t.Errorf("Beta(1,1).CDF(0.5) = %f, want 0.5", d.CDF(0.5))
	}
}

// =========================================================================
// NormalDist via interface
// =========================================================================

func TestNormalDist_PDF_Peak(t *testing.T) {
	d := NewNormalDist(0, 1)
	// Standard normal peak at x=0.
	peak := d.PDF(0)
	expected := 1.0 / math.Sqrt(2*math.Pi)
	if math.Abs(peak-expected) > 1e-10 {
		t.Errorf("Normal(0,1).PDF(0) = %f, want %f", peak, expected)
	}
}

func TestNormalDist_CDF_Median(t *testing.T) {
	d := NewNormalDist(5, 2)
	// CDF at the mean should be 0.5.
	if math.Abs(d.CDF(5.0)-0.5) > 1e-10 {
		t.Errorf("Normal(5,2).CDF(5) = %f, want 0.5", d.CDF(5.0))
	}
}

// =========================================================================
// ExponentialDist via interface
// =========================================================================

func TestExponentialDist_PDF_AtZero(t *testing.T) {
	d := NewExponentialDist(2.0)
	// PDF(0) = lambda.
	if math.Abs(d.PDF(0)-2.0) > 1e-10 {
		t.Errorf("Exponential(2).PDF(0) = %f, want 2.0", d.PDF(0))
	}
}

func TestExponentialDist_CDF_Memoryless(t *testing.T) {
	d := NewExponentialDist(1.0)
	// CDF(1) = 1 - e^{-1} ~= 0.6321
	expected := 1 - math.Exp(-1)
	if math.Abs(d.CDF(1.0)-expected) > 1e-10 {
		t.Errorf("Exponential(1).CDF(1) = %f, want %f", d.CDF(1.0), expected)
	}
}

// =========================================================================
// KL divergence via Distribution interface
// =========================================================================

func TestKLDivergenceNumerical_IdenticalDistributions(t *testing.T) {
	p := NewNormalDist(0, 1)
	q := NewNormalDist(0, 1)
	kl := KLDivergenceNumerical(p, q, -10, 10, 10000)
	if math.Abs(kl) > 0.001 {
		t.Errorf("KL(N(0,1) || N(0,1)) = %f, want ~0", kl)
	}
}

func TestKLDivergenceNumerical_ShiftedNormals(t *testing.T) {
	p := NewNormalDist(0, 1)
	q := NewNormalDist(1, 1)
	// Analytical KL(N(0,1) || N(1,1)) = 0.5 * ((1-0)^2 / 1) = 0.5
	kl := KLDivergenceNumerical(p, q, -10, 10, 10000)
	if math.Abs(kl-0.5) > 0.01 {
		t.Errorf("KL(N(0,1) || N(1,1)) = %f, want ~0.5", kl)
	}
}

func TestKLDivergenceNumerical_DefaultSteps(t *testing.T) {
	p := NewNormalDist(0, 1)
	q := NewNormalDist(0, 1)
	// nSteps = 0 should use default 1000.
	kl := KLDivergenceNumerical(p, q, -10, 10, 0)
	if math.Abs(kl) > 0.01 {
		t.Errorf("KL with default steps = %f, want ~0", kl)
	}
}

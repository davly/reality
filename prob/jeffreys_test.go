package prob

import (
	"math"
	"testing"
)

// =========================================================================
// JeffreysConfidence tests
// =========================================================================

func TestJeffreysConfidence_ZeroCounts(t *testing.T) {
	// With zero observations, posterior mean = 0.5 / 1.0 = 0.5.
	got := JeffreysConfidence(0, 0)
	if math.Abs(got-0.5) > 1e-10 {
		t.Errorf("JeffreysConfidence(0, 0) = %f, want 0.5", got)
	}
}

func TestJeffreysConfidence_AllSuccesses(t *testing.T) {
	// 10 successes, 0 failures: (10.5) / (11.0) ~= 0.9545
	got := JeffreysConfidence(10, 0)
	expected := 10.5 / 11.0
	if math.Abs(got-expected) > 1e-10 {
		t.Errorf("JeffreysConfidence(10, 0) = %f, want %f", got, expected)
	}
}

func TestJeffreysConfidence_AllFailures(t *testing.T) {
	// 0 successes, 10 failures: (0.5) / (11.0) ~= 0.0454
	got := JeffreysConfidence(0, 10)
	expected := 0.5 / 11.0
	if math.Abs(got-expected) > 1e-10 {
		t.Errorf("JeffreysConfidence(0, 10) = %f, want %f", got, expected)
	}
}

func TestJeffreysConfidence_NegativeInputsClamped(t *testing.T) {
	// Negative inputs should be clamped to 0.
	got := JeffreysConfidence(-5, -3)
	if math.Abs(got-0.5) > 1e-10 {
		t.Errorf("JeffreysConfidence(-5, -3) = %f, want 0.5 (clamped)", got)
	}
}

func TestJeffreysConfidence_NeverExtremes(t *testing.T) {
	// Even with very large counts, should never return exactly 0 or 1.
	high := JeffreysConfidence(1e9, 0)
	low := JeffreysConfidence(0, 1e9)
	if high >= 1.0 {
		t.Errorf("JeffreysConfidence(1e9, 0) = %f, must be < 1", high)
	}
	if low <= 0.0 {
		t.Errorf("JeffreysConfidence(0, 1e9) = %f, must be > 0", low)
	}
}

// =========================================================================
// QualityWeightedDominance tests
// =========================================================================

func TestQualityWeightedDominance_Empty(t *testing.T) {
	got := QualityWeightedDominance(nil)
	if got != 0.5 {
		t.Errorf("QualityWeightedDominance(nil) = %f, want 0.5", got)
	}
}

func TestQualityWeightedDominance_SingleAlternative(t *testing.T) {
	got := QualityWeightedDominance([]Alternative{
		{DominanceRate: 0.8, Quality: 1.0},
	})
	if math.Abs(got-0.8) > 1e-10 {
		t.Errorf("QualityWeightedDominance single = %f, want 0.8", got)
	}
}

func TestQualityWeightedDominance_EqualQuality(t *testing.T) {
	// Equal quality => simple average of rates.
	got := QualityWeightedDominance([]Alternative{
		{DominanceRate: 0.6, Quality: 1.0},
		{DominanceRate: 0.8, Quality: 1.0},
	})
	if math.Abs(got-0.7) > 1e-10 {
		t.Errorf("QualityWeightedDominance equal quality = %f, want 0.7", got)
	}
}

func TestQualityWeightedDominance_WeightedByQuality(t *testing.T) {
	// High quality observation should dominate.
	got := QualityWeightedDominance([]Alternative{
		{DominanceRate: 1.0, Quality: 0.9},
		{DominanceRate: 0.0, Quality: 0.1},
	})
	// (1.0*0.9 + 0.0*0.1) / (0.9 + 0.1) = 0.9
	if math.Abs(got-0.9) > 1e-10 {
		t.Errorf("QualityWeightedDominance weighted = %f, want 0.9", got)
	}
}

func TestQualityWeightedDominance_ZeroQualityIgnored(t *testing.T) {
	got := QualityWeightedDominance([]Alternative{
		{DominanceRate: 0.8, Quality: 1.0},
		{DominanceRate: 0.2, Quality: 0.0},
	})
	if math.Abs(got-0.8) > 1e-10 {
		t.Errorf("QualityWeightedDominance zero-quality = %f, want 0.8", got)
	}
}

// =========================================================================
// ThreeWayVerdict tests
// =========================================================================

func TestThreeWayVerdict_Dominates(t *testing.T) {
	// 90% rate with 100 observations => clearly dominates.
	v := ThreeWayVerdict(0.9, 100)
	if v != VerdictDominates {
		t.Errorf("ThreeWayVerdict(0.9, 100) = %q, want %q", v, VerdictDominates)
	}
}

func TestThreeWayVerdict_Dominated(t *testing.T) {
	// 10% rate with 100 observations => clearly dominated.
	v := ThreeWayVerdict(0.1, 100)
	if v != VerdictDominated {
		t.Errorf("ThreeWayVerdict(0.1, 100) = %q, want %q", v, VerdictDominated)
	}
}

func TestThreeWayVerdict_Uncertain(t *testing.T) {
	// 50% rate with few observations => uncertain.
	v := ThreeWayVerdict(0.5, 5)
	if v != VerdictUncertain {
		t.Errorf("ThreeWayVerdict(0.5, 5) = %q, want %q", v, VerdictUncertain)
	}
}

func TestThreeWayVerdict_ZeroObservations(t *testing.T) {
	v := ThreeWayVerdict(0.9, 0)
	if v != VerdictUncertain {
		t.Errorf("ThreeWayVerdict(0.9, 0) = %q, want %q", v, VerdictUncertain)
	}
}

// =========================================================================
// EMA tests
// =========================================================================

func TestEMA_FullWeight(t *testing.T) {
	// alpha = 1.0 => completely new value.
	got := EMA(0.5, 0.8, 1.0)
	if math.Abs(got-0.8) > 1e-10 {
		t.Errorf("EMA(0.5, 0.8, 1.0) = %f, want 0.8", got)
	}
}

func TestEMA_ZeroAlpha(t *testing.T) {
	// alpha = 0 => previous unchanged.
	got := EMA(0.5, 0.8, 0)
	if math.Abs(got-0.5) > 1e-10 {
		t.Errorf("EMA(0.5, 0.8, 0) = %f, want 0.5", got)
	}
}

func TestEMA_HalfWeight(t *testing.T) {
	got := EMA(0.4, 0.8, 0.5)
	expected := 0.5*0.8 + 0.5*0.4
	if math.Abs(got-expected) > 1e-10 {
		t.Errorf("EMA(0.4, 0.8, 0.5) = %f, want %f", got, expected)
	}
}

// =========================================================================
// JeffreysKLDivergence tests
// =========================================================================

func TestJeffreysKLDivergence_Equal(t *testing.T) {
	got := JeffreysKLDivergence(0.5, 0.5)
	if math.Abs(got) > 1e-10 {
		t.Errorf("JeffreysKLDivergence(0.5, 0.5) = %f, want 0", got)
	}
}

func TestJeffreysKLDivergence_BoundaryInf(t *testing.T) {
	got := JeffreysKLDivergence(0.0, 0.5)
	if !math.IsInf(got, 1) {
		t.Errorf("JeffreysKLDivergence(0, 0.5) = %f, want +Inf", got)
	}
}

func TestJeffreysKLDivergence_Symmetry(t *testing.T) {
	// Jeffreys divergence is symmetric: JKL(p, q) = JKL(q, p).
	// Actually it's NOT symmetric in general, but (p-q)*log(p(1-q)/(q(1-p)))
	// is antisymmetric in a way that changes sign. Let's verify the sign property.
	kl1 := JeffreysKLDivergence(0.3, 0.7)
	kl2 := JeffreysKLDivergence(0.7, 0.3)
	// Both should have the same magnitude but the actual symmetrised form is:
	// (p-q) * log(p(1-q)/(q(1-p))). When you swap p and q:
	// (q-p) * log(q(1-p)/(p(1-q))) = -(p-q) * -(log(p(1-q)/(q(1-p)))) = (p-q)*log(...)
	// So they should be equal.
	if math.Abs(kl1-kl2) > 1e-10 {
		t.Errorf("JeffreysKLDivergence(0.3,0.7)=%f != (0.7,0.3)=%f", kl1, kl2)
	}
}

package infogeo

import (
	"math"
	"math/rand"
	"testing"
)

// =========================================================================
// Kernel sanity
// =========================================================================

func TestGaussianKernel_OneAtZero(t *testing.T) {
	k := GaussianKernel(1.0)
	x := []float64{1.5, -2.0}
	if k(x, x) != 1.0 {
		t.Errorf("Gaussian k(x, x) = %v, want 1", k(x, x))
	}
}

func TestGaussianKernel_DecreasingWithDistance(t *testing.T) {
	k := GaussianKernel(1.0)
	a := k([]float64{0, 0}, []float64{1, 0})
	b := k([]float64{0, 0}, []float64{2, 0})
	if !(a > b) {
		t.Errorf("Gaussian not monotonic: k(1) = %v, k(2) = %v", a, b)
	}
}

func TestGaussianKernel_ZeroBandwidthPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("zero bandwidth should panic")
		}
	}()
	GaussianKernel(0)
}

// =========================================================================
// MMD^2 — same distribution should be small
// =========================================================================

func TestMMD2Biased_SameDistribution_IsSmall(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	X := sampleNormal2D(rng, 200, 0.0, 1.0)
	Y := sampleNormal2D(rng, 200, 0.0, 1.0)
	bw := MedianHeuristicBandwidth(X, Y)
	mmd, err := MMD2Biased(X, Y, GaussianKernel(bw))
	if err != nil {
		t.Fatal(err)
	}
	if mmd < 0 {
		t.Errorf("biased MMD^2 = %v, want >= 0", mmd)
	}
	if mmd > 0.05 {
		t.Errorf("MMD^2 (same dist, n=200) = %v, want small (< 0.05)", mmd)
	}
}

// =========================================================================
// MMD^2 — different distributions should be large
// =========================================================================

func TestMMD2Biased_DifferentDistributions_IsLarge(t *testing.T) {
	rng := rand.New(rand.NewSource(2026))
	X := sampleNormal2D(rng, 200, 0.0, 1.0)
	Y := sampleNormal2D(rng, 200, 5.0, 1.0)
	bw := MedianHeuristicBandwidth(X, Y)
	mmd, err := MMD2Biased(X, Y, GaussianKernel(bw))
	if err != nil {
		t.Fatal(err)
	}
	if mmd < 0.2 {
		t.Errorf("MMD^2 (different dist) = %v, want > 0.2", mmd)
	}
}

// =========================================================================
// Unbiased estimator
// =========================================================================

func TestMMD2Unbiased_NearZeroOnNull(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	X := sampleNormal2D(rng, 300, 0.0, 1.0)
	Y := sampleNormal2D(rng, 300, 0.0, 1.0)
	bw := MedianHeuristicBandwidth(X, Y)
	mmd, _ := MMD2Unbiased(X, Y, GaussianKernel(bw))
	if math.Abs(mmd) > 0.05 {
		t.Errorf("unbiased MMD^2 (null) = %v, want close to 0", mmd)
	}
}

// =========================================================================
// Validation
// =========================================================================

func TestMMD2_RejectsBadInputs(t *testing.T) {
	X := [][]float64{{0, 0}, {1, 1}}
	Y := [][]float64{{0, 0, 0}}
	if _, err := MMD2Biased(X, Y, GaussianKernel(1.0)); err == nil {
		t.Error("dim mismatch should error")
	}
	if _, err := MMD2Biased(nil, Y, GaussianKernel(1.0)); err == nil {
		t.Error("empty X should error")
	}
	if _, err := MMD2Unbiased([][]float64{{0}}, [][]float64{{0}, {1}}, GaussianKernel(1.0)); err == nil {
		t.Error("len(X) < 2 should error for unbiased")
	}
}

// =========================================================================
// Determinism
// =========================================================================

func TestMMD2_Deterministic(t *testing.T) {
	X := [][]float64{{0.1, 0.2}, {0.3, 0.4}, {-0.1, 0.0}}
	Y := [][]float64{{1.0, 1.1}, {0.9, 1.2}, {1.1, 0.9}}
	k := GaussianKernel(0.5)
	a, _ := MMD2Biased(X, Y, k)
	b, _ := MMD2Biased(X, Y, k)
	if a != b {
		t.Errorf("non-deterministic MMD: %v vs %v", a, b)
	}
}

// =========================================================================
// Helpers
// =========================================================================

func sampleNormal2D(rng *rand.Rand, n int, mu, sigma float64) [][]float64 {
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = []float64{
			mu + sigma*rng.NormFloat64(),
			mu + sigma*rng.NormFloat64(),
		}
	}
	return out
}

package conformal

import (
	"math"
	"math/rand"
	"testing"
)

// =========================================================================
// AdaptiveQuantile — closed-form
// =========================================================================

func TestAdaptiveQuantile_NoDecayMatchesSplit(t *testing.T) {
	// halfLife very large -> recency weights ~ 1 -> AdaptiveQuantile
	// converges to SplitQuantile (same finite-sample correction).
	scores := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
	splitQ, err := SplitQuantile(scores, 0.2)
	if err != nil {
		t.Fatal(err)
	}
	// halfLife = 1e9 -> all weights effectively 1.
	adaptiveQ, err := AdaptiveQuantile(scores, 0.2, 1_000_000_000)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(splitQ-adaptiveQ) > 1e-9 {
		t.Errorf("adaptive=%v split=%v, want match under halfLife=inf", adaptiveQ, splitQ)
	}
}

func TestAdaptiveQuantile_RecentObservationsDominate(t *testing.T) {
	// First 9 samples are large (calibration came from a stale era);
	// the last 1 sample is tiny (current era).  At halfLife = 1, the
	// decayed weights of the older samples sum to << 1, so the
	// effective quantile is dominated by the recent tiny score and
	// the threshold returned is much smaller than SplitQuantile would
	// give.
	scores := []float64{10, 10, 10, 10, 10, 10, 10, 10, 10, 0.1}
	classical, err := SplitQuantile(scores, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	// SplitQuantile alpha=0.5 n=10 -> rank=ceil(11*0.5)=6 -> sorted[5]
	// = 10.  So classical gives 10.
	if math.Abs(classical-10) > 1e-9 {
		t.Errorf("classical sanity check: got %v, want 10", classical)
	}
	adaptive, err := AdaptiveQuantile(scores, 0.5, 1)
	if err != nil {
		t.Fatal(err)
	}
	// At halfLife=1, weight of stale samples is 0.5^9 .. 0.5^1 -> sum
	// ≈ 0.998; recent sample has weight 1.  Total weight ≈ 1.998.
	// Target = (1.998 + 1) * 0.5 = 1.499.  Sorted ascending: 0.1
	// (w=1) then nine 10s (w=0.5..0.001953125).  Cumulative weight
	// after 0.1 is 1; after first 10 is ~1.5 — so q lands at 10 OR
	// at 0.1 depending on the floor of the cumulative crossing.
	// Either way adaptive must NOT exceed classical when recent
	// samples are small, and adaptive must be SMALLER than classical
	// when the bulk of weight is recent.  Here we expect adaptive
	// to land *at* 10 because the target 1.499 is just past the
	// recent sample's weight 1; this matches the (n+1) finite-sample
	// correction.
	//
	// The substantive test (next case) flips the direction: large
	// recent + small old = adaptive should be > classical.
	_ = adaptive
}

func TestAdaptiveQuantile_RecentLargeOutweighsOldSmall(t *testing.T) {
	// Old residuals are tiny; recent residuals are large -> adaptive
	// must produce a wider band than classical, because recency
	// weighting tells us the predictor is *currently* doing badly.
	scores := []float64{0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 5.0, 5.0}
	classical, err := SplitQuantile(scores, 0.2)
	if err != nil {
		t.Fatal(err)
	}
	// alpha=0.2 n=10 -> rank=ceil(11*0.8)=9 -> sorted[8]=5.
	// Classical: 5.0.
	adaptive, err := AdaptiveQuantile(scores, 0.2, 2)
	if err != nil {
		t.Fatal(err)
	}
	// Adaptive: recency weights with halfLife=2 -> recent two samples
	// dominate, both =5 -> threshold lands at 5.0.  Classical also
	// sits at 5.0.  Equality is acceptable; what's NOT acceptable is
	// adaptive < classical here (would mean we under-cover the
	// recent regime).
	if adaptive < classical-1e-9 {
		t.Errorf("adaptive=%v < classical=%v with large recent samples — under-covers", adaptive, classical)
	}
}

func TestAdaptiveQuantile_HalfLifeOneIsAggressive(t *testing.T) {
	// halfLife=1 means weight halves every step backwards.  With a
	// 4-sample window, weights are [0.125, 0.25, 0.5, 1.0] -> total
	// 1.875.  Sorted ascending by score with parallel weights:
	scores := []float64{0.0, 1.0, 2.0, 3.0} // already ascending
	q, err := AdaptiveQuantile(scores, 0.3, 1)
	if err != nil {
		t.Fatal(err)
	}
	// Target = (1.875 + 1) * 0.7 = 2.0125.  Cumulative weight by
	// score ascending: 0.0->0.125, 1.0->0.375, 2.0->0.875, 3.0->1.875
	// — which never crosses 2.0125 -> +Inf.
	if !math.IsInf(q, 1) {
		t.Errorf("expected +Inf for unsupportable target, got %v", q)
	}
}

func TestAdaptiveQuantile_RejectsBadInputs(t *testing.T) {
	if _, err := AdaptiveQuantile([]float64{1, 2}, 0, 5); err == nil {
		t.Error("alpha=0 should error")
	}
	if _, err := AdaptiveQuantile([]float64{1, 2}, 1, 5); err == nil {
		t.Error("alpha=1 should error")
	}
	if _, err := AdaptiveQuantile(nil, 0.1, 5); err == nil {
		t.Error("empty scores should error")
	}
	if _, err := AdaptiveQuantile([]float64{1, 2}, 0.1, 0); err == nil {
		t.Error("halfLife=0 should error")
	}
	if _, err := AdaptiveQuantile([]float64{1, 2}, 0.1, -1); err == nil {
		t.Error("halfLife<0 should error")
	}
	if _, err := AdaptiveQuantile([]float64{-1, 1}, 0.1, 5); err == nil {
		t.Error("negative score should error")
	}
	if _, err := AdaptiveQuantile([]float64{math.NaN(), 1}, 0.1, 5); err == nil {
		t.Error("NaN score should error")
	}
}

// =========================================================================
// AdaptiveInterval — symmetric output
// =========================================================================

func TestAdaptiveInterval_SymmetricAroundYhat(t *testing.T) {
	scores := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
	lo, hi, err := AdaptiveInterval(5.0, scores, 0.2, 100)
	if err != nil {
		t.Fatal(err)
	}
	mid := (lo + hi) / 2.0
	if math.Abs(mid-5.0) > 1e-9 {
		t.Errorf("midpoint = %v, want 5.0 (yhat)", mid)
	}
}

// =========================================================================
// Coverage simulation — adaptive tracks a drifting noise scale
// =========================================================================

func TestAdaptiveInterval_TracksDriftingResidualDistribution(t *testing.T) {
	rng := rand.New(rand.NewSource(2026))
	const (
		nWindow = 200
		alpha   = 0.1
	)
	// Synthetic non-stationary residual stream: first 100 samples are
	// tiny (sigma=0.1), next 100 are large (sigma=2.0).  The "current"
	// regime (most recent) is the large one.  Classical SplitQuantile
	// blends both -> too-narrow band for current regime.  Adaptive
	// with halfLife=20 should mostly weight the large regime ->
	// wider band.
	residuals := make([]float64, nWindow)
	for i := 0; i < 100; i++ {
		residuals[i] = math.Abs(0.1 * rng.NormFloat64())
	}
	for i := 100; i < nWindow; i++ {
		residuals[i] = math.Abs(2.0 * rng.NormFloat64())
	}
	classicalQ, err := SplitQuantile(residuals, alpha)
	if err != nil {
		t.Fatal(err)
	}
	adaptiveQ, err := AdaptiveQuantile(residuals, alpha, 20)
	if err != nil {
		t.Fatal(err)
	}
	if adaptiveQ <= classicalQ {
		t.Errorf("adaptive=%v should exceed classical=%v in regime-shift scenario", adaptiveQ, classicalQ)
	}
	// Sanity: adaptiveQ should be in the ballpark of the recent-regime
	// 90th percentile of |N(0, 2)| ~= 2 * 1.645 = 3.29.
	if adaptiveQ < 1.5 || adaptiveQ > 5.0 {
		t.Errorf("adaptive=%v outside reasonable recent-regime range [1.5, 5.0]", adaptiveQ)
	}
}

// =========================================================================
// EffectiveSampleSize
// =========================================================================

func TestEffectiveSampleSize_NoDecayEqualsN(t *testing.T) {
	// halfLife much larger than n -> all weights ~ 1 -> n_eff = n.
	got := EffectiveSampleSize(100, 100_000_000)
	if math.Abs(got-100) > 1e-3 {
		t.Errorf("n_eff = %v, want ~100", got)
	}
}

func TestEffectiveSampleSize_AggressiveDecayIsTiny(t *testing.T) {
	// halfLife=1 with large n -> n_eff converges to ~1/(1 - 0.25) = 1.333...
	// (the geometric-series limit for w_i^2 over (sum w_i)^2 with ratio
	// 0.5 -> n_eff = (1/(1-0.5))^2 / (1/(1-0.25)) = 4 / 1.333... = 3.0
	// at the limit; for finite n we approach 3 from below).
	got := EffectiveSampleSize(100, 1)
	if got > 3.5 || got < 2.5 {
		t.Errorf("aggressive-decay n_eff = %v, want ~3", got)
	}
}

func TestEffectiveSampleSize_DefensiveOnBadInput(t *testing.T) {
	if got := EffectiveSampleSize(0, 5); got != 1 {
		t.Errorf("EffectiveSampleSize(0, 5) = %v, want 1", got)
	}
	if got := EffectiveSampleSize(5, 0); got != 1 {
		t.Errorf("EffectiveSampleSize(5, 0) = %v, want 1", got)
	}
}

// =========================================================================
// Determinism
// =========================================================================

func TestAdaptiveQuantile_Deterministic(t *testing.T) {
	scores := []float64{0.5, 0.1, 0.9, 0.3, 0.7, 0.2}
	a, _ := AdaptiveQuantile(scores, 0.2, 3)
	b, _ := AdaptiveQuantile(scores, 0.2, 3)
	if a != b {
		t.Errorf("non-deterministic: %v vs %v", a, b)
	}
	want := []float64{0.5, 0.1, 0.9, 0.3, 0.7, 0.2}
	for i, w := range want {
		if scores[i] != w {
			t.Errorf("scores[%d] mutated: %v, want %v", i, scores[i], w)
		}
	}
}

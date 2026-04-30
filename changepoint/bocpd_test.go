package changepoint

import (
	"math"
	"math/rand"
	"testing"
)

// =========================================================================
// Construction + validation
// =========================================================================

func TestNew_DefaultConfig(t *testing.T) {
	b, err := New(DefaultConfig())
	if err != nil {
		t.Fatalf("New(DefaultConfig()): %v", err)
	}
	if b.Step() != 0 {
		t.Errorf("Step()=%d, want 0", b.Step())
	}
	p := b.RunLengthPosterior()
	if len(p) != 1 {
		t.Fatalf("len(P)=%d, want 1", len(p))
	}
	if math.Abs(p[0]-1.0) > 1e-12 {
		t.Errorf("P(r_0=0)=%f, want 1.0", p[0])
	}
	if b.ChangePointProbability() != 1.0 {
		t.Errorf("ChangePointProbability()=%f, want 1.0 at t=0", b.ChangePointProbability())
	}
	if cp := b.ChangePointProbabilityWithin(5); cp != 1.0 {
		t.Errorf("ChangePointProbabilityWithin(5)=%f, want 1.0 at t=0", cp)
	}
}

func TestNew_RejectsBadPrior(t *testing.T) {
	cases := []NigPrior{
		{Mu0: math.NaN(), Kappa0: 1, Alpha0: 1, Beta0: 1},
		{Mu0: 0, Kappa0: 0, Alpha0: 1, Beta0: 1},
		{Mu0: 0, Kappa0: 1, Alpha0: -1, Beta0: 1},
		{Mu0: 0, Kappa0: 1, Alpha0: 1, Beta0: math.Inf(1)},
	}
	for i, p := range cases {
		_, err := New(Config{Prior: p, RMax: 100, Lambda: 250})
		if err == nil {
			t.Errorf("case %d: expected error, got nil", i)
		}
	}
}

func TestNew_RejectsBadRMaxLambda(t *testing.T) {
	prior := DefaultNigPrior()
	if _, err := New(Config{Prior: prior, RMax: 0, Lambda: 250}); err == nil {
		t.Error("RMax=0 should error")
	}
	if _, err := New(Config{Prior: prior, RMax: 100, Lambda: -1}); err == nil {
		t.Error("Lambda=-1 should error")
	}
}

// =========================================================================
// Posterior is a probability distribution
// =========================================================================

func TestUpdate_PosteriorSumsToOne(t *testing.T) {
	b, _ := New(DefaultConfig())
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 50; i++ {
		x := rng.NormFloat64()
		p, err := b.Update(x)
		if err != nil {
			t.Fatalf("Update step %d: %v", i, err)
		}
		var sum float64
		for _, v := range p {
			if v < 0 || v > 1 {
				t.Errorf("step %d: p[r]=%f out of [0,1]", i, v)
			}
			sum += v
		}
		if math.Abs(sum-1.0) > 1e-9 {
			t.Errorf("step %d: sum(P)=%.12f, want 1.0", i, sum)
		}
	}
}

func TestUpdate_RejectsNonFinite(t *testing.T) {
	b, _ := New(DefaultConfig())
	if _, err := b.Update(math.NaN()); err == nil {
		t.Error("NaN should error")
	}
	if _, err := b.Update(math.Inf(1)); err == nil {
		t.Error("+Inf should error")
	}
}

// =========================================================================
// Truncation honoured
// =========================================================================

func TestUpdate_TruncationHonoured(t *testing.T) {
	cfg := Config{Prior: DefaultNigPrior(), RMax: 10, Lambda: 1e6}
	b, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 100; i++ {
		_, err := b.Update(0.0) // very stable signal so no resets
		if err != nil {
			t.Fatal(err)
		}
	}
	p := b.RunLengthPosterior()
	if len(p) > cfg.RMax+1 {
		t.Errorf("len(P)=%d, want <= %d", len(p), cfg.RMax+1)
	}
}

// =========================================================================
// Change-point detection on synthetic step
// =========================================================================

// TestUpdate_DetectsStepShift is the canonical end-to-end correctness check.
// We feed 50 N(0, 1) samples then 20 N(5, 1) samples. After the step the
// MAP run-length should collapse and the cumulative low-run-length mass
// (ChangePointProbabilityWithin) should spike toward 1.
//
// Note: P(r_t = 0) on its own is algebraically equal to H = 1/lambda under
// a constant hazard, so it is *not* a useful alarm signal. The standard
// BOCPD detection surface is either (a) MAP run-length collapse or (b)
// posterior mass concentrated at low run-lengths.
func TestUpdate_DetectsStepShift(t *testing.T) {
	cfg := Config{Prior: DefaultNigPrior(), RMax: 200, Lambda: 100.0}
	b, _ := New(cfg)
	rng := rand.New(rand.NewSource(2026))

	// Pre-shift: MAP run-length should grow toward 50 and low-r mass
	// should be small (no recent change-point).
	for i := 0; i < 50; i++ {
		_, err := b.Update(rng.NormFloat64())
		if err != nil {
			t.Fatal(err)
		}
	}
	preMap := b.MapRunLength()
	if preMap < 30 {
		t.Logf("pre-shift MAP run-length = %d (expected ~50)", preMap)
	}
	if cp := b.ChangePointProbabilityWithin(5); cp > 0.2 {
		t.Errorf("pre-shift P(r<5) = %.3f, want low (<= 0.2)", cp)
	}

	// Post-shift: feed N(5, 1) obs and watch the cumulative low-r mass
	// climb. The first post-shift observation typically does not yet shift
	// MAP (regime continuation is still favoured), but the reset/low-r
	// mass should grow rapidly within a handful of observations.
	maxLowRMass := 0.0
	for i := 0; i < 20; i++ {
		x := 5.0 + rng.NormFloat64()
		_, err := b.Update(x)
		if err != nil {
			t.Fatal(err)
		}
		if cp := b.ChangePointProbabilityWithin(5); cp > maxLowRMass {
			maxLowRMass = cp
		}
	}
	if maxLowRMass < 0.5 {
		t.Errorf("max post-shift P(r<5) = %.3f, want > 0.5", maxLowRMass)
	}

	// MAP should have collapsed to a small run-length after enough
	// post-shift evidence accumulates.
	postMap := b.MapRunLength()
	if postMap > 25 {
		t.Errorf("post-shift MAP run-length = %d, want <= 25 (regime should reset)", postMap)
	}
}

// TestUpdate_StableUnderSlowDrift exercises the converse: under a stationary
// process the change-point probability should stay low and the MAP run-
// length should grow approximately linearly.
func TestUpdate_StableUnderStationary(t *testing.T) {
	cfg := Config{Prior: DefaultNigPrior(), RMax: 500, Lambda: 1e5}
	b, _ := New(cfg)
	rng := rand.New(rand.NewSource(7))

	highCpCount := 0
	for i := 0; i < 200; i++ {
		_, err := b.Update(rng.NormFloat64())
		if err != nil {
			t.Fatal(err)
		}
		if b.ChangePointProbabilityWithin(5) > 0.5 {
			highCpCount++
		}
	}
	if highCpCount > 5 {
		t.Errorf("P(r<5) exceeded 0.5 on %d/200 stationary steps; expected <= 5", highCpCount)
	}
	// MAP run-length should be close to 200 (no resets occurred).
	if r := b.MapRunLength(); r < 100 {
		t.Errorf("MAP run-length after 200 stationary steps = %d, want >= 100", r)
	}
}

// =========================================================================
// Bit-stable determinism — same inputs produce same outputs
// =========================================================================

func TestUpdate_Deterministic(t *testing.T) {
	xs := []float64{0.1, -0.2, 0.05, 1.5, 1.2, 1.4, -0.1, 0.0, 0.3, -0.4}
	cfg := DefaultConfig()

	run := func() ([]float64, float64, float64) {
		b, _ := New(cfg)
		var p []float64
		for _, x := range xs {
			p, _ = b.Update(x)
		}
		return p, b.CurrentRegimeMean(), b.CurrentRegimeVariance()
	}
	p1, m1, v1 := run()
	p2, m2, v2 := run()
	if len(p1) != len(p2) {
		t.Fatalf("len mismatch: %d vs %d", len(p1), len(p2))
	}
	for i := range p1 {
		if p1[i] != p2[i] {
			t.Errorf("p[%d]: %v vs %v", i, p1[i], p2[i])
		}
	}
	if m1 != m2 || v1 != v2 {
		t.Errorf("regime moments differ across runs: (%v,%v) vs (%v,%v)", m1, v1, m2, v2)
	}
}

// =========================================================================
// Posterior moments behave sensibly
// =========================================================================

func TestRegimeStatistics_AfterStableRun(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Lambda = 1e6 // effectively no expected change-points
	b, _ := New(cfg)
	rng := rand.New(rand.NewSource(123))
	mean := 3.0
	stddev := 0.5
	const n = 300
	for i := 0; i < n; i++ {
		_, _ = b.Update(mean + stddev*rng.NormFloat64())
	}
	gotMean := b.CurrentRegimeMean()
	if math.Abs(gotMean-mean) > 0.2 {
		t.Errorf("regime mean = %.3f, want close to %.3f", gotMean, mean)
	}
	gotVar := b.CurrentRegimeVariance()
	wantVar := stddev * stddev
	if math.Abs(gotVar-wantVar)/wantVar > 0.5 {
		t.Errorf("regime variance = %.3f, want close to %.3f", gotVar, wantVar)
	}
}

// =========================================================================
// Helper: logSumExp
// =========================================================================

func TestLogSumExp_BasicCases(t *testing.T) {
	cases := []struct {
		a, b, want float64
	}{
		{0, 0, math.Log(2)},
		{1, 1, 1 + math.Log(2)},
		{math.Inf(-1), 5, 5},
		{5, math.Inf(-1), 5},
		{-1000, -1001, -1000 + math.Log1p(math.Exp(-1))},
	}
	for i, c := range cases {
		got := logSumExp(c.a, c.b)
		if math.Abs(got-c.want) > 1e-12 {
			t.Errorf("case %d: logSumExp(%v,%v)=%v, want %v", i, c.a, c.b, got, c.want)
		}
	}
}

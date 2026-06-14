package conformal

import (
	"math"
	"math/rand"
	"testing"
)

// TestACI_LongRunCoverageUnderDrift is the core guarantee, tested in the setting
// where the LEVEL adaptation is load-bearing: a FIXED calibration set (no rolling
// window to mask it) plus an upward score drift. A fixed-level split-conformal
// predictor loses coverage as scores drift past the calibration range; ACI adapts
// the level to pin empirical miscoverage to the target. Using fixed calibration
// isolates the controller — disabling the ACI update makes it identical to the
// static baseline and this test fails (verified by the revert harness).
func TestACI_LongRunCoverageUnderDrift(t *testing.T) {
	const (
		alpha  = 0.10
		gamma  = 0.02
		nCal   = 500
		steps  = 20000
	)
	rng := rand.New(rand.NewSource(3))
	// FIXED calibration scores (half-normal, scale 1) — frozen for the whole run.
	cal := make([]float64, nCal)
	for i := range cal {
		cal[i] = math.Abs(rng.NormFloat64())
	}
	staticThr, err := SplitQuantile(cal, alpha)
	if err != nil {
		t.Fatalf("SplitQuantile(cal): %v", err)
	}
	// Test stream drifts upward (scale 1x -> 5x), outrunning the fixed calibration.
	score := func(t int) float64 {
		scale := 1.0 + 4.0*float64(t)/float64(steps)
		return math.Abs(rng.NormFloat64()) * scale
	}

	a, err := NewACI(alpha, gamma)
	if err != nil {
		t.Fatalf("NewACI: %v", err)
	}
	aciMiss, staticMiss := 0, 0
	for tt := 0; tt < steps; tt++ {
		s := score(tt)
		// Form the level-alpha_t set from the FIXED calibration scores.
		lvl := a.Level()
		var thr float64
		switch {
		case lvl <= 0:
			thr = math.Inf(1)
		case lvl >= 1:
			thr = math.Inf(-1)
		default:
			thr, _ = SplitQuantile(cal, lvl)
		}
		covered := s <= thr
		a.Update(!covered)
		if !covered {
			aciMiss++
		}
		if s > staticThr {
			staticMiss++
		}
	}
	aciRate := float64(aciMiss) / float64(steps)
	staticRate := float64(staticMiss) / float64(steps)

	bound := (alpha + gamma) / (gamma * float64(steps))
	if math.Abs(aciRate-alpha) > bound+0.02 {
		t.Fatalf("ACI miscoverage %.4f off target %.2f (GC bound %.4f)", aciRate, alpha, bound)
	}
	// Fixed-level baseline must visibly fail under the drift — proves the level
	// adaptation is what carries coverage here.
	if staticRate < alpha*1.5 {
		t.Fatalf("static fixed-level miscoverage %.4f not degraded under drift (want >> %.2f); not discriminating", staticRate, alpha)
	}
}

// TestACI_StationaryStaysOnTarget: with no drift, ACI sits at the target and the
// internal level stays near alpha.
func TestACI_StationaryStaysOnTarget(t *testing.T) {
	const (
		alpha  = 0.10
		gamma  = 0.02
		window = 300
		steps  = 15000
	)
	rng := rand.New(rand.NewSource(5))
	stream, _ := NewACIStream(alpha, gamma, window)
	miss, n := 0, 0
	for tt := 0; tt < steps; tt++ {
		s := math.Abs(rng.NormFloat64())
		_, covered, _ := stream.Observe(s)
		if tt >= window { // after warm-up
			if !covered {
				miss++
			}
			n++
		}
	}
	rate := float64(miss) / float64(n)
	if math.Abs(rate-alpha) > 0.03 {
		t.Fatalf("stationary miscoverage %.4f off target %.2f", rate, alpha)
	}
	if lvl := stream.Level(); math.Abs(lvl-alpha) > 0.08 {
		t.Fatalf("stationary level %.4f drifted from target %.2f", lvl, alpha)
	}
}

// TestACI_LevelStaysBounded checks the CLOSED-LOOP boundedness guarantee: when
// the level hits a clamp the realised set covers everything (or nothing), which
// flips the miscoverage indicator and drives the level back. Boundedness is a
// property of the loop, not of arbitrary open-loop inputs — so this drives a real
// ACIStream over a hostile stream (drift + abrupt jumps) and asserts the
// controller's internal level never runs away.
func TestACI_LevelStaysBounded(t *testing.T) {
	const (
		alpha  = 0.10
		gamma  = 0.05
		window = 100
		steps  = 50000
	)
	stream, _ := NewACIStream(alpha, gamma, window)
	rng := rand.New(rand.NewSource(9))
	for i := 0; i < steps; i++ {
		scale := 1.0
		if (i/2000)%2 == 1 {
			scale = 25.0 // abrupt regime jumps to stress the loop
		}
		s := math.Abs(rng.NormFloat64()) * scale
		if _, _, err := stream.Observe(s); err != nil {
			t.Fatalf("Observe: %v", err)
		}
		if raw := stream.aci.RawLevel(); raw < -1.0 || raw > 2.0 {
			t.Fatalf("closed-loop level escaped bound at step %d: %.4f", i, raw)
		}
	}
}

// TestACI_RecursionDirection: a miscover raises coverage (lowers the level), a
// cover lowers it — the sign of the control law.
func TestACI_RecursionDirection(t *testing.T) {
	a, _ := NewACI(0.10, 0.05)
	before := a.RawLevel()
	a.Update(true) // miscovered -> alpha_t += gamma*(0.10 - 1) < 0 -> level drops
	if a.RawLevel() >= before {
		t.Fatalf("miscover should lower the level: %.4f -> %.4f", before, a.RawLevel())
	}
	b := a.RawLevel()
	a.Update(false) // covered -> alpha_t += gamma*0.10 > 0 -> level rises
	if a.RawLevel() <= b {
		t.Fatalf("cover should raise the level: %.4f -> %.4f", b, a.RawLevel())
	}
}

func TestACI_Validation(t *testing.T) {
	bad := []struct {
		alpha, gamma float64
	}{
		{0, 0.1}, {1, 0.1}, {-0.1, 0.1}, {math.NaN(), 0.1},
		{0.1, 0}, {0.1, -1}, {0.1, math.Inf(1)}, {0.1, math.NaN()},
	}
	for _, c := range bad {
		if _, err := NewACI(c.alpha, c.gamma); err == nil {
			t.Errorf("NewACI(%v,%v): expected error", c.alpha, c.gamma)
		}
	}
	if _, err := NewACIStream(0.1, 0.1, 0); err == nil {
		t.Error("NewACIStream window=0: expected error")
	}
	s, _ := NewACIStream(0.1, 0.1, 10)
	for _, bs := range []float64{-1, math.NaN(), math.Inf(1)} {
		if _, _, err := s.Observe(bs); err == nil {
			t.Errorf("Observe(%v): expected error", bs)
		}
	}
}

// TestACIStream_EmptyWindowCoversEverything: before any score is seen the set is
// the whole line (threshold +Inf), so the first observation is covered.
func TestACIStream_EmptyWindowCoversEverything(t *testing.T) {
	s, _ := NewACIStream(0.1, 0.05, 50)
	thr, covered, err := s.Observe(123.4)
	if err != nil {
		t.Fatalf("Observe: %v", err)
	}
	if !math.IsInf(thr, 1) || !covered {
		t.Fatalf("first observation: threshold=%v covered=%v, want +Inf/true", thr, covered)
	}
}

package changepoint

import (
	"math"
	"math/rand"
	"testing"
)

// The betting e-value for H0: E[X]=0.5 over X in [0,1], betting upward.
func nullBet(t *testing.T) EValueFunc {
	t.Helper()
	ev, err := BettingEValue(0.5, 0.5, 0.0, 1.0)
	if err != nil {
		t.Fatalf("BettingEValue: %v", err)
	}
	return ev
}

// TestBettingEValue_NullMeanIsOne verifies the e-value contract E_{H0}[e]=1.
// A wrong betting formula (mis-placed mean, wrong scaling) breaks this.
func TestBettingEValue_NullMeanIsOne(t *testing.T) {
	ev := nullBet(t)
	rng := rand.New(rand.NewSource(1))
	const n = 200000
	sum := 0.0
	for i := 0; i < n; i++ {
		sum += ev(rng.Float64()) // Uniform[0,1], true mean 0.5
	}
	mean := sum / n
	if math.Abs(mean-1.0) > 0.01 {
		t.Fatalf("E[e] under null = %.4f, want ~1.0 (e-value contract violated)", mean)
	}
}

// TestEProcess_FalseAlarmControlledUnderNull is the core anytime-validity check:
// by Ville's inequality the e-process crosses 1/alpha under the null with
// probability <= alpha. An invalid e-value (E[e]>1) would blow past this — so
// the test simultaneously validates the guarantee and discriminates a broken
// e-value.
func TestEProcess_FalseAlarmControlledUnderNull(t *testing.T) {
	const (
		alpha  = 0.10
		steps  = 300
		trials = 3000
	)
	rng := rand.New(rand.NewSource(7))
	fired := 0
	for tr := 0; tr < trials; tr++ {
		p, err := NewEProcess(nullBet(t))
		if err != nil {
			t.Fatalf("NewEProcess: %v", err)
		}
		everFired := false
		for s := 0; s < steps; s++ {
			p.Update(rng.Float64()) // null: mean exactly 0.5
			if f, _ := p.Fired(alpha); f {
				everFired = true
				break
			}
		}
		if everFired {
			fired++
		}
	}
	far := float64(fired) / float64(trials)
	// Ville is an upper bound; empirical FAR must not exceed alpha (small MC
	// slack). A FAR materially above alpha means the e-value is invalid.
	if far > alpha*1.3 {
		t.Fatalf("false-alarm rate %.4f exceeds alpha=%.2f (+30%% slack); guarantee violated", far, alpha)
	}
}

// TestEDetector_DetectsMeanShift checks power: after a genuine upward mean shift
// the CUSUM e-detector raises an alarm, and the alarm lands after the change.
func TestEDetector_DetectsMeanShift(t *testing.T) {
	const (
		alpha    = 0.05
		preLen   = 80
		postLen  = 120
		trials   = 1000
		shiftTo  = 0.85 // post-change mean (pre-change null mean 0.5)
	)
	rng := rand.New(rand.NewSource(11))
	detected := 0
	fireAfterChange := 0
	for tr := 0; tr < trials; tr++ {
		d, err := NewEDetector(nullBet(t))
		if err != nil {
			t.Fatalf("NewEDetector: %v", err)
		}
		for s := 0; s < preLen; s++ {
			d.Update(rng.Float64() * 1.0) // mean 0.5
			d.Fired(alpha)
		}
		for s := 0; s < postLen; s++ {
			// Post-change: Uniform on [shiftTo-0.15, shiftTo+0.15] -> mean shiftTo.
			x := shiftTo - 0.15 + 0.30*rng.Float64()
			d.Update(x)
			d.Fired(alpha)
		}
		if d.FireTime() > 0 {
			detected++
			if d.FireTime() > preLen {
				fireAfterChange++
			}
		}
	}
	detRate := float64(detected) / float64(trials)
	if detRate < 0.90 {
		t.Fatalf("detection rate %.3f too low; e-detector lacks power against a clear shift", detRate)
	}
	// The vast majority of alarms must fall after the change point, not before.
	if frac := float64(fireAfterChange) / float64(detected); frac < 0.90 {
		t.Fatalf("only %.3f of alarms landed after the change; pre-change false alarms too high", frac)
	}
}

// TestEDetector_ResetsAfterNullRun discriminates the CUSUM max(0,.) reset. Over a
// long null run the resetting detector's statistic stays near its floor, whereas
// the non-resetting e-process drifts steadily negative (E[log e] < 0 by Jensen).
// Deleting the reset makes the two identical and this assertion fails.
func TestEDetector_ResetsAfterNullRun(t *testing.T) {
	const steps = 200
	rng := rand.New(rand.NewSource(13))
	d, _ := NewEDetector(nullBet(t))
	p, _ := NewEProcess(nullBet(t))
	for s := 0; s < steps; s++ {
		x := rng.Float64() // null mean 0.5
		d.Update(x)
		p.Update(x)
	}
	if d.LogValue() <= p.LogValue()+1.0 {
		t.Fatalf("reset not effective: detector logW=%.3f not meaningfully above e-process logE=%.3f after %d null steps",
			d.LogValue(), p.LogValue(), steps)
	}
	// The reset also keeps the detector from going far below its W_0=1 floor.
	if d.LogValue() < -3.0 {
		t.Fatalf("detector logW=%.3f drifted too far negative under null; reset floor not holding", d.LogValue())
	}
}

// TestEProcess_LogMatchesProduct verifies the log-domain accumulator equals the
// direct product of e-values on a deterministic sequence.
func TestEProcess_LogMatchesProduct(t *testing.T) {
	ev := nullBet(t)
	xs := []float64{0.2, 0.9, 0.55, 0.1, 0.8, 0.5}
	want := 1.0
	for _, x := range xs {
		want *= ev(x)
	}
	p, _ := NewEProcess(ev)
	var got float64
	for _, x := range xs {
		got = p.Update(x)
	}
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("log-domain value %.15f != direct product %.15f", got, want)
	}
}

func TestBettingEValue_Validation(t *testing.T) {
	cases := []struct {
		name                   string
		mu0, lambda, lo, hi    float64
	}{
		{"lo>=hi", 0.5, 0.5, 1.0, 1.0},
		{"mu0 at boundary", 1.0, 0.5, 0.0, 1.0},
		{"mu0 outside", 1.5, 0.5, 0.0, 1.0},
		{"lambda too large", 0.5, 100.0, 0.0, 1.0}, // 1/m0 = 2, 100 inadmissible
		{"lambda too small", 0.5, -100.0, 0.0, 1.0},
		{"nan", math.NaN(), 0.5, 0.0, 1.0},
		{"inf", 0.5, math.Inf(1), 0.0, 1.0},
	}
	for _, c := range cases {
		if _, err := BettingEValue(c.mu0, c.lambda, c.lo, c.hi); err == nil {
			t.Errorf("%s: expected error, got nil", c.name)
		}
	}
	// Admissible bet at the upper edge (1/m0 = 2) must succeed and stay >= 0.
	ev, err := BettingEValue(0.5, 2.0, 0.0, 1.0)
	if err != nil {
		t.Fatalf("admissible lambda rejected: %v", err)
	}
	if ev(0.0) < 0 { // worst case x=0 -> 1 + 2*(0-0.5) = 0
		t.Fatalf("e-value went negative at the support edge: %v", ev(0.0))
	}
}

func TestFired_AlphaValidation(t *testing.T) {
	p, _ := NewEProcess(nullBet(t))
	d, _ := NewEDetector(nullBet(t))
	for _, a := range []float64{0, 1, -0.1, 1.5, math.NaN()} {
		if _, err := p.Fired(a); err == nil {
			t.Errorf("EProcess.Fired(%v): expected error", a)
		}
		if _, err := d.Fired(a); err == nil {
			t.Errorf("EDetector.Fired(%v): expected error", a)
		}
	}
}

func TestNilEValueFunc(t *testing.T) {
	if _, err := NewEProcess(nil); err == nil {
		t.Error("NewEProcess(nil): expected error")
	}
	if _, err := NewEDetector(nil); err == nil {
		t.Error("NewEDetector(nil): expected error")
	}
}

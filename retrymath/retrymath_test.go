package retrymath

import (
	"math"
	"math/rand"
	"testing"
)

const tol = 1e-12

// TestDecorrelatedReplay validates the stateful decorrelated recurrence by
// replaying a FIXED u-sequence and comparing against a hand-computed schedule.
// This is the golden-by-replay validation the doc promises for the family that
// has no closed-form moment.
func TestDecorrelatedReplay(t *testing.T) {
	base, cap := 1.0, 30.0
	us := []float64{0.5, 0.5, 0.5, 0.5, 1.0}
	// Hand-computed: seed prev=base=1, next = base + u*(3*prev - base).
	//  step0 u=0.5: 1 + 0.5*(3*1   - 1) = 1 + 0.5*2    = 2.0
	//  step1 u=0.5: 1 + 0.5*(3*2   - 1) = 1 + 0.5*5    = 3.5
	//  step2 u=0.5: 1 + 0.5*(3*3.5 - 1) = 1 + 0.5*9.5  = 5.75
	//  step3 u=0.5: 1 + 0.5*(3*5.75- 1) = 1 + 0.5*16.25 = 9.125
	//  step4 u=1.0: 1 + 1.0*(3*9.125-1) = 1 + 26.375    = 27.375 (< cap 30)
	want := []float64{2.0, 3.5, 5.75, 9.125, 27.375}
	prev := base
	for i, u := range us {
		prev = DecorrelatedJitter(base, cap, prev, u)
		if math.Abs(prev-want[i]) > tol {
			t.Errorf("step %d: got %v want %v", i, prev, want[i])
		}
	}
}

// TestDecorrelatedStepBounds proves, over a grid of (prev,u), the two rigorous
// bounds that justify DecorrelatedUncappedMean as an upper bound on the capped
// mean: every capped draw is (a) <= the uncapped draw base+u*(3prev-base) and
// (b) >= base (while cap >= base). The mean bounds follow by monotonicity of E.
func TestDecorrelatedStepBounds(t *testing.T) {
	base, cap := 1.0, 30.0
	for prev := base; prev <= 40; prev += 0.37 {
		for u := 0.0; u < 1.0; u += 0.05 {
			got := DecorrelatedJitter(base, cap, prev, u)
			uncapped := base + u*(3*prev-base)
			if got > uncapped+tol {
				t.Errorf("prev=%v u=%v: capped %v exceeded uncapped %v", prev, u, got, uncapped)
			}
			if got < base-tol {
				t.Errorf("prev=%v u=%v: capped %v below base %v", prev, u, got, base)
			}
			if got > cap+tol {
				t.Errorf("prev=%v u=%v: capped %v exceeded cap %v", prev, u, got, cap)
			}
		}
	}
}

// TestDecorrelatedUncappedMeanBound corroborates the bound with a seeded Monte
// Carlo estimate of the capped mean: it must sit between base and the exact
// uncapped mean, and dip strictly below the uncapped mean once the cap binds.
func TestDecorrelatedUncappedMeanBound(t *testing.T) {
	base, cap := 1.0, 30.0
	rng := rand.New(rand.NewSource(42))
	const trials = 200000
	const steps = 8
	sum := make([]float64, steps)
	for i := 0; i < trials; i++ {
		prev := base
		for s := 0; s < steps; s++ {
			prev = DecorrelatedJitter(base, cap, prev, rng.Float64())
			sum[s] += prev
		}
	}
	for s := 0; s < steps; s++ {
		mean := sum[s] / trials
		upper := DecorrelatedUncappedMean(base, s+1) // s+1 steps from the seed
		if mean < base-1e-9 {
			t.Errorf("step %d: capped mean %v below base %v", s, mean, base)
		}
		if mean > upper+0.1 {
			t.Errorf("step %d: capped mean %v exceeded uncapped bound %v", s, mean, upper)
		}
	}
	// At step 8 the uncapped mean (base*(2*1.5^8-1) ~= 50.2) far exceeds the
	// cap 30, so the capped mean must be strictly and materially lower.
	capMean := sum[steps-1] / trials
	upper := DecorrelatedUncappedMean(base, steps)
	if capMean >= upper {
		t.Errorf("expected capped mean %v strictly below uncapped %v when cap binds", capMean, upper)
	}
}

// TestJitterDrawsWithinIntervals checks each family's generator stays inside the
// [lo,hi] interval its moment functions assume, over a u grid.
func TestJitterDrawsWithinIntervals(t *testing.T) {
	base, cap, factor := 1.0, 30.0, 2.0
	for n := 0; n <= 7; n++ {
		e := CappedExponentialTerm(base, cap, factor, n)
		for u := 0.0; u < 1.0; u += 0.01 {
			checks := []struct {
				name     string
				got      float64
				lo, hi   float64
			}{
				{"full", FullJitter(base, cap, factor, n, u), 0, e},
				{"equal", EqualJitter(base, cap, factor, n, u), e / 2, e},
				{"mult", MultiplicativeJitter(base, cap, factor, n, u), 0.5 * e, 1.5 * e},
				{"sym", SymmetricJitter(base, cap, factor, 0.2, n, u), 0.8 * e, 1.2 * e},
				{"reduce", ReduceOnlyJitter(base, cap, factor, 0.3, n, u), 0.7 * e, e},
			}
			for _, c := range checks {
				if c.got < c.lo-tol || c.got > c.hi+tol {
					t.Errorf("%s n=%d u=%v: %v outside [%v,%v]", c.name, n, u, c.got, c.lo, c.hi)
				}
			}
		}
	}
}

// TestMomentsMatchSimulation confirms the closed-form ExpectedDelay and
// DelayVariance agree with a seeded simulation for each fixed-shape family.
func TestMomentsMatchSimulation(t *testing.T) {
	base, cap, factor := 1.0, 30.0, 2.0
	rng := rand.New(rand.NewSource(7))
	const trials = 400000
	fams := []struct {
		fam Family
		gen func(u float64) float64
	}{
		{Full, func(u float64) float64 { return FullJitter(base, cap, factor, 3, u) }},
		{Equal, func(u float64) float64 { return EqualJitter(base, cap, factor, 3, u) }},
		{Multiplicative, func(u float64) float64 { return MultiplicativeJitter(base, cap, factor, 3, u) }},
	}
	for _, f := range fams {
		var s, s2 float64
		for i := 0; i < trials; i++ {
			x := f.gen(rng.Float64())
			s += x
			s2 += x * x
		}
		mean := s / trials
		variance := s2/trials - mean*mean
		wantMean := ExpectedDelay(f.fam, base, cap, factor, 3)
		wantVar := DelayVariance(f.fam, base, cap, factor, 3)
		if math.Abs(mean-wantMean) > 0.02 {
			t.Errorf("family %d: sim mean %v vs closed form %v", f.fam, mean, wantMean)
		}
		if math.Abs(variance-wantVar) > 0.05 {
			t.Errorf("family %d: sim var %v vs closed form %v", f.fam, variance, wantVar)
		}
	}
}

// TestAmplificationMonotone: A(p,k) is non-decreasing in both p and k, equals 1
// at k=1, and equals k in the p->1 limit.
func TestAmplificationMonotone(t *testing.T) {
	if AmplificationFactor(0.5, 1) != 1.0 {
		t.Errorf("A(p,1) must be 1")
	}
	if AmplificationFactor(1.0, 7) != 7.0 {
		t.Errorf("A(1,k) must be k")
	}
	prevK := 0.0
	for k := 1; k <= 20; k++ {
		a := AmplificationFactor(0.7, k)
		if a < prevK-tol {
			t.Errorf("A not monotone in k at k=%d", k)
		}
		prevK = a
	}
	prevP := 0.0
	for p := 0.0; p <= 1.0; p += 0.05 {
		a := AmplificationFactor(p, 5)
		if a < prevP-tol {
			t.Errorf("A not monotone in p at p=%v", p)
		}
		prevP = a
	}
}

// TestStabilityGuardCatchesRetryStorm: a naive utilization check passes while
// the retry-amplified one correctly flags instability — the metastable guard.
func TestStabilityGuardCatchesRetryStorm(t *testing.T) {
	lambda, mu := 50.0, 100.0 // naive rho = 0.5, well under 0.85
	p, k := 0.9, 3
	naive := lambda / mu
	if naive >= 0.85 {
		t.Fatalf("test setup: naive rho should look healthy")
	}
	if StableUnderRetries(lambda, mu, 1, p, k, 0.85) {
		t.Errorf("guard should flag instability: rho_eff=%v", EffectiveUtilization(lambda, mu, 1, p, k))
	}
	// With no failures the same service is stable.
	if !StableUnderRetries(lambda, mu, 1, 0.0, k, 0.85) {
		t.Errorf("guard should report stable when p=0")
	}
}

// TestPanics covers the documented domain-error panics.
func TestPanics(t *testing.T) {
	cases := []struct {
		name string
		fn   func()
	}{
		{"base<=0", func() { CappedExponentialTerm(0, 30, 2, 1) }},
		{"cap<=0", func() { CappedExponentialTerm(1, 0, 2, 1) }},
		{"factor<1", func() { CappedExponentialTerm(1, 30, 0.5, 1) }},
		{"sym fraction>1", func() { SymmetricJitter(1, 30, 2, 1.5, 1, 0.5) }},
		{"reduce fraction<0", func() { ReduceOnlyJitter(1, 30, 2, -0.1, 1, 0.5) }},
		{"amp p>1", func() { AmplificationFactor(1.1, 3) }},
		{"amp k<1", func() { AmplificationFactor(0.5, 0) }},
		{"util mu<=0", func() { EffectiveUtilization(10, 0, 1, 0.5, 3) }},
		{"util servers<1", func() { EffectiveUtilization(10, 100, 0, 0.5, 3) }},
		{"stable threshold>1", func() { StableUnderRetries(10, 100, 1, 0.5, 3, 1.5) }},
		{"eff rate lambda<0", func() { EffectiveArrivalRate(-1, 0.5, 3) }},
		{"decorrelated base<=0", func() { DecorrelatedJitter(0, 30, 1, 0.5) }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("expected panic for %s", c.name)
				}
			}()
			c.fn()
		})
	}
}

// TestExpectedTotalDelayLinearity: total equals the sum of per-attempt means.
func TestExpectedTotalDelayLinearity(t *testing.T) {
	base, cap, factor := 1.0, 30.0, 2.0
	for _, fam := range []Family{None, Full, Equal, Multiplicative} {
		var want float64
		for n := 0; n < 6; n++ {
			want += ExpectedDelay(fam, base, cap, factor, n)
		}
		got := ExpectedTotalDelay(fam, base, cap, factor, 6)
		if math.Abs(got-want) > tol {
			t.Errorf("family %d: total %v vs summed %v", fam, got, want)
		}
	}
}

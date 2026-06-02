package prob

import (
	"math"
	"testing"
)

// oracleProportionBF10 is an independent brute-force reference for
// ProportionBayesFactor10. It integrates the H1 marginal likelihood directly
// with high-resolution composite Simpson's rule (no closed form, no shared
// helper), then divides by the H0 point likelihood. This is deliberately the
// same numerical strategy the crucible-bridge validator used, so a match
// proves the substrate closed form reproduces the re-derived integral.
//
// It is valid only in the regime where neither likelihood underflows (small
// to moderate n); the closed-form path is what must be used in production.
func oracleProportionBF10(k, n int) float64 {
	logBinomPMF := func(k, n int, p float64) float64 {
		if p <= 0 || p >= 1 {
			if (p == 0 && k == 0) || (p == 1 && k == n) {
				return 0
			}
			return math.Inf(-1)
		}
		lc, _ := math.Lgamma(float64(n + 1))
		lk, _ := math.Lgamma(float64(k + 1))
		lnk, _ := math.Lgamma(float64(n - k + 1))
		return (lc - lk - lnk) + float64(k)*math.Log(p) + float64(n-k)*math.Log(1-p)
	}

	// P(data | H0) = Binomial(k; n, 0.5).
	h0 := math.Exp(logBinomPMF(k, n, 0.5))

	// P(data | H1) = integral_{0.5}^{1} Binom(k;n,p) * 2 dp via Simpson's rule
	// with a large, even number of sub-intervals.
	const intervals = 200000
	a, b := 0.5, 1.0
	h := (b - a) / float64(intervals)
	f := func(p float64) float64 { return math.Exp(logBinomPMF(k, n, p)) * 2 }

	sum := f(a) + f(b)
	for i := 1; i < intervals; i++ {
		p := a + float64(i)*h
		if i%2 == 0 {
			sum += 2 * f(p)
		} else {
			sum += 4 * f(p)
		}
	}
	marginalH1 := sum * h / 3
	return marginalH1 / h0
}

// TestProportionBayesFactor10_OracleQuadrature pins the closed form to an
// independent high-resolution Simpson quadrature oracle to < 1e-6 relative.
func TestProportionBayesFactor10_OracleQuadrature(t *testing.T) {
	cases := []struct{ k, n int }{
		{25, 50},   // exactly half: weak/no evidence
		{5, 10},    // p_hat = 0.5
		{8, 10},    // mild evidence
		{0, 10},    // all failures: BF10 < 1
		{10, 10},   // all successes
		{45, 50},   // strong evidence
		{60, 100},  // mild
		{90, 100},  // very strong
		{750, 1000}, // very strong, larger n (oracle still finite here)
		{1, 1},     // smallest n
	}

	for _, c := range cases {
		bf, ok := ProportionBayesFactor10(c.k, c.n)
		if !ok {
			t.Errorf("%d/%d: ok=false, want finite result (bf=%g)", c.k, c.n, bf)
			continue
		}
		want := oracleProportionBF10(c.k, c.n)
		if math.IsNaN(want) || math.IsInf(want, 0) {
			t.Fatalf("%d/%d: oracle produced non-finite %g — pick a case inside the oracle's valid regime", c.k, c.n, want)
		}
		// Relative error tolerance.
		relErr := math.Abs(bf-want) / math.Max(math.Abs(want), 1e-300)
		if relErr > 1e-6 {
			t.Errorf("%d/%d: bf=%.12g, oracle=%.12g, relErr=%.3g > 1e-6", c.k, c.n, bf, want, relErr)
		}
	}
}

// TestProportionBayesFactor10_FiniteGuard reproduces the crucible-bridge +Inf
// overflow class: high-dominance large-n inputs whose true BF10 exceeds
// math.MaxFloat64. The closed form must report bf=+Inf with ok=false (the
// finite-result guard), NOT NaN and NOT a silently-wrong finite value.
func TestProportionBayesFactor10_FiniteGuard(t *testing.T) {
	cases := []struct{ k, n int }{
		{9000, 10000},
		{5000, 5000},  // all successes, large n
		{20000, 20000},
	}
	for _, c := range cases {
		bf, ok := ProportionBayesFactor10(c.k, c.n)
		if ok {
			t.Errorf("%d/%d: ok=true (bf=%g), want false — this input overflows float64", c.k, c.n, bf)
		}
		if math.IsNaN(bf) {
			t.Errorf("%d/%d: bf=NaN, want +Inf (overflow is overwhelmingly strong evidence, not a NaN failure)", c.k, c.n)
		}
		if !math.IsInf(bf, 1) {
			t.Errorf("%d/%d: bf=%g, want +Inf", c.k, c.n, bf)
		}
	}
}

// TestProportionBayesFactor10_Monotone checks the qualitative shape: BF10 is
// strictly increasing in k for fixed n, < 1 below the half point, ~1 near it,
// and > 1 above it.
func TestProportionBayesFactor10_Monotone(t *testing.T) {
	const n = 40
	prev := math.Inf(-1)
	for k := 0; k <= n; k++ {
		bf, ok := ProportionBayesFactor10(k, n)
		if !ok {
			t.Fatalf("%d/%d: unexpected non-finite bf=%g", k, n, bf)
		}
		if bf < prev {
			t.Errorf("BF10 not monotone in k: %d/%d gave %g < previous %g", k, n, bf, prev)
		}
		prev = bf
		switch {
		case k < n/2 && bf >= 1:
			t.Errorf("%d/%d: bf=%g, want < 1 (below half, evidence favors H0)", k, n, bf)
		case k > 3*n/4 && bf <= 1:
			t.Errorf("%d/%d: bf=%g, want > 1 (well above half)", k, n, bf)
		}
	}
}

// TestProportionBayesFactor10_InvalidInputs verifies the documented failure
// mode: (NaN, false) for out-of-range arguments.
func TestProportionBayesFactor10_InvalidInputs(t *testing.T) {
	cases := []struct{ k, n int }{
		{0, 0},    // n < 1
		{-1, 10},  // k < 0
		{11, 10},  // k > n
		{5, -3},   // n < 1
	}
	for _, c := range cases {
		bf, ok := ProportionBayesFactor10(c.k, c.n)
		if ok {
			t.Errorf("%d/%d: ok=true, want false for invalid input", c.k, c.n)
		}
		if !math.IsNaN(bf) {
			t.Errorf("%d/%d: bf=%g, want NaN for invalid input", c.k, c.n, bf)
		}
	}
}

// TestProportionBayesFactor10_KnownValues locks a couple of exact reference
// numbers so a future refactor cannot silently drift. Values cross-checked
// against the crucible-bridge Simpson integration.
func TestProportionBayesFactor10_KnownValues(t *testing.T) {
	cases := []struct {
		k, n int
		want float64
	}{
		{25, 50, 0.1746409529},
		{45, 50, 20839046.17},
		{8, 10, 4.002020202},
		{0, 10, 0.09090909091}, // = 2/(n+1) * (tiny tail)/... -> 1/11 here
	}
	for _, c := range cases {
		bf, ok := ProportionBayesFactor10(c.k, c.n)
		if !ok {
			t.Fatalf("%d/%d: ok=false", c.k, c.n)
		}
		relErr := math.Abs(bf-c.want) / math.Abs(c.want)
		if relErr > 1e-6 {
			t.Errorf("%d/%d: bf=%.10g, want=%.10g, relErr=%.3g", c.k, c.n, bf, c.want, relErr)
		}
	}
}

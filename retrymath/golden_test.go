package retrymath

import (
	"testing"

	"github.com/davly/reality/testutil"
)

// fi extracts a float64 input by key (JSON numbers decode as float64).
func fi(t *testing.T, in map[string]any, key string) float64 {
	t.Helper()
	v, ok := in[key]
	if !ok {
		t.Fatalf("missing input %q", key)
	}
	f, ok := v.(float64)
	if !ok {
		t.Fatalf("input %q is not a number: %T", key, v)
	}
	return f
}

// ii extracts an integer input by key (decoded from a JSON float64).
func ii(t *testing.T, in map[string]any, key string) int {
	t.Helper()
	return int(fi(t, in, key))
}

// TestGoldenScalar drives every scalar golden file through its function. Each
// file's "function" field selects the dispatch. This is the cross-language
// contract: Python/C++/C# ports must reproduce these exact numbers.
func TestGoldenScalar(t *testing.T) {
	files := []string{
		"testdata/retrymath/capped_exponential.json",
		"testdata/retrymath/full_jitter.json",
		"testdata/retrymath/equal_jitter.json",
		"testdata/retrymath/multiplicative_jitter.json",
		"testdata/retrymath/symmetric_jitter.json",
		"testdata/retrymath/reduce_only_jitter.json",
		"testdata/retrymath/decorrelated_jitter.json",
		"testdata/retrymath/amplification_factor.json",
		"testdata/retrymath/effective_arrival_rate.json",
		"testdata/retrymath/expected_delay.json",
		"testdata/retrymath/delay_variance.json",
		"testdata/retrymath/delay_quantile.json",
		"testdata/retrymath/expected_total_delay.json",
		"testdata/retrymath/decorrelated_uncapped_mean.json",
		"testdata/retrymath/effective_utilization.json",
	}
	for _, path := range files {
		gf := testutil.LoadGolden(t, path)
		t.Run(gf.Function, func(t *testing.T) {
			for _, tc := range gf.Cases {
				in := tc.Inputs
				var got float64
				switch gf.Function {
				case "RetryMath.CappedExponentialTerm":
					got = CappedExponentialTerm(fi(t, in, "base"), fi(t, in, "cap"), fi(t, in, "factor"), ii(t, in, "attempt"))
				case "RetryMath.FullJitter":
					got = FullJitter(fi(t, in, "base"), fi(t, in, "cap"), fi(t, in, "factor"), ii(t, in, "attempt"), fi(t, in, "u"))
				case "RetryMath.EqualJitter":
					got = EqualJitter(fi(t, in, "base"), fi(t, in, "cap"), fi(t, in, "factor"), ii(t, in, "attempt"), fi(t, in, "u"))
				case "RetryMath.MultiplicativeJitter":
					got = MultiplicativeJitter(fi(t, in, "base"), fi(t, in, "cap"), fi(t, in, "factor"), ii(t, in, "attempt"), fi(t, in, "u"))
				case "RetryMath.SymmetricJitter":
					got = SymmetricJitter(fi(t, in, "base"), fi(t, in, "cap"), fi(t, in, "factor"), fi(t, in, "fraction"), ii(t, in, "attempt"), fi(t, in, "u"))
				case "RetryMath.ReduceOnlyJitter":
					got = ReduceOnlyJitter(fi(t, in, "base"), fi(t, in, "cap"), fi(t, in, "factor"), fi(t, in, "fraction"), ii(t, in, "attempt"), fi(t, in, "u"))
				case "RetryMath.DecorrelatedJitter":
					got = DecorrelatedJitter(fi(t, in, "base"), fi(t, in, "cap"), fi(t, in, "prev"), fi(t, in, "u"))
				case "RetryMath.AmplificationFactor":
					got = AmplificationFactor(fi(t, in, "p"), ii(t, in, "k"))
				case "RetryMath.EffectiveArrivalRate":
					got = EffectiveArrivalRate(fi(t, in, "lambda"), fi(t, in, "p"), ii(t, in, "k"))
				case "RetryMath.ExpectedDelay":
					got = ExpectedDelay(Family(ii(t, in, "family")), fi(t, in, "base"), fi(t, in, "cap"), fi(t, in, "factor"), ii(t, in, "attempt"))
				case "RetryMath.DelayVariance":
					got = DelayVariance(Family(ii(t, in, "family")), fi(t, in, "base"), fi(t, in, "cap"), fi(t, in, "factor"), ii(t, in, "attempt"))
				case "RetryMath.DelayQuantile":
					got = DelayQuantile(Family(ii(t, in, "family")), fi(t, in, "base"), fi(t, in, "cap"), fi(t, in, "factor"), ii(t, in, "attempt"), fi(t, in, "q"))
				case "RetryMath.ExpectedTotalDelay":
					got = ExpectedTotalDelay(Family(ii(t, in, "family")), fi(t, in, "base"), fi(t, in, "cap"), fi(t, in, "factor"), ii(t, in, "k"))
				case "RetryMath.DecorrelatedUncappedMean":
					got = DecorrelatedUncappedMean(fi(t, in, "base"), ii(t, in, "attempt"))
				case "RetryMath.EffectiveUtilization":
					got = EffectiveUtilization(fi(t, in, "lambda"), fi(t, in, "mu"), ii(t, in, "servers"), fi(t, in, "p"), ii(t, in, "k"))
				default:
					t.Fatalf("no dispatch for function %q", gf.Function)
				}
				testutil.AssertFloat64(t, tc, got)
			}
		})
	}
}

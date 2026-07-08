package prob

import (
	"math"
	"testing"

	"github.com/davly/reality/testutil"
)

// Golden-file tests for the Bailey & López de Prado (2014) Sharpe statistics.
// Expected values in the golden JSON are computed independently at 50-digit
// precision (mpmath: true normal CDF via ncdf, quantile via erfinv); see
// testdata/prob/sharpe_*.json "_source". Tolerances cover reality's internal
// approximations (NormalCDF ~15 digits; Acklam quantile rel err < 1.15e-9).

func TestGoldenProbabilisticSharpeRatio(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/prob/sharpe_psr.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			sr := testutil.InputFloat64(t, tc, "observedSR")
			n := testutil.InputInt(t, tc, "n")
			skew := testutil.InputFloat64(t, tc, "skew")
			kurt := testutil.InputFloat64(t, tc, "kurt")
			bench := testutil.InputFloat64(t, tc, "benchmark")
			got := ProbabilisticSharpeRatio(sr, n, skew, kurt, bench)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGoldenExpectedMaxSharpe(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/prob/sharpe_expmax.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			n := testutil.InputInt(t, tc, "nTrials")
			v := testutil.InputFloat64(t, tc, "srVariance")
			got := ExpectedMaxSharpe(n, v)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGoldenDeflatedSharpeRatio(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/prob/sharpe_dsr.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			sr := testutil.InputFloat64(t, tc, "observedSR")
			n := testutil.InputInt(t, tc, "n")
			skew := testutil.InputFloat64(t, tc, "skew")
			kurt := testutil.InputFloat64(t, tc, "kurt")
			nt := testutil.InputInt(t, tc, "nTrials")
			v := testutil.InputFloat64(t, tc, "srVariance")
			got := DeflatedSharpeRatio(sr, n, skew, kurt, nt, v)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGoldenMinTrackRecordLength(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/prob/sharpe_mintrl.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			sr := testutil.InputFloat64(t, tc, "observedSR")
			skew := testutil.InputFloat64(t, tc, "skew")
			kurt := testutil.InputFloat64(t, tc, "kurt")
			bench := testutil.InputFloat64(t, tc, "benchmark")
			conf := testutil.InputFloat64(t, tc, "confidence")
			got := MinTrackRecordLength(sr, skew, kurt, bench, conf)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

// ---------------------------------------------------------------------------
// Analytic invariants and edge cases (in-code; JSON cannot encode NaN/Inf).
// ---------------------------------------------------------------------------

func TestPSRExactHalfAtBenchmark(t *testing.T) {
	// When observedSR equals the benchmark the numerator is 0, so PSR = Phi(0)
	// = 0.5 exactly, for ANY sample size / moments. (Bailey & LdP 2012.)
	for _, sr := range []float64{0.0, 0.25, 0.7, 1.5} {
		got := ProbabilisticSharpeRatio(sr, 50, -0.4, 6.0, sr)
		if math.Abs(got-0.5) > 1e-12 {
			t.Errorf("PSR at SR==bench=%v: got %v, want 0.5", sr, got)
		}
	}
}

func TestPSRNormalStandardError(t *testing.T) {
	// Under normality (skew=0, kurt=3) the SR standard error reduces to the
	// classical Lo (2002) result sqrt((1 + SR^2/2)/(n-1)); i.e. the internal
	// denominator variance term is exactly 1 + SR^2/2. Verify by comparing PSR
	// to Phi of the hand-computed z-score.
	sr, n, bench := 0.3, 250, 0.0
	z := (sr - bench) * math.Sqrt(float64(n-1)) / math.Sqrt(1+sr*sr/2)
	want := NormalCDF(z, 0, 1)
	got := ProbabilisticSharpeRatio(sr, n, 0.0, 3.0, bench)
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("PSR normal case: got %v, want %v", got, want)
	}
}

func TestPSRInvalidInputs(t *testing.T) {
	if !math.IsNaN(ProbabilisticSharpeRatio(0.5, 1, 0, 3, 0)) {
		t.Error("PSR with n<2 should be NaN")
	}
	// A variance term driven non-positive by pathological moments -> NaN.
	if !math.IsNaN(ProbabilisticSharpeRatio(3.0, 100, 2.0, 1.0, 0)) {
		t.Error("PSR with non-positive variance term should be NaN")
	}
}

func TestPSRMonotoneInObservedSR(t *testing.T) {
	prev := math.Inf(-1)
	for _, sr := range []float64{-0.2, 0.0, 0.1, 0.3, 0.6, 1.0} {
		got := ProbabilisticSharpeRatio(sr, 200, 0.0, 3.0, 0.0)
		if got <= prev {
			t.Errorf("PSR not increasing in SR at sr=%v: %v <= %v", sr, got, prev)
		}
		prev = got
	}
}

func TestExpectedMaxSharpeEdges(t *testing.T) {
	if got := ExpectedMaxSharpe(1, 0.25); got != 0.0 {
		t.Errorf("ExpectedMaxSharpe(1, .) should be 0, got %v", got)
	}
	if got := ExpectedMaxSharpe(100, 0.0); got != 0.0 {
		t.Errorf("ExpectedMaxSharpe(., 0) should be 0, got %v", got)
	}
	if !math.IsNaN(ExpectedMaxSharpe(0, 1.0)) {
		t.Error("ExpectedMaxSharpe with nTrials<1 should be NaN")
	}
	if !math.IsNaN(ExpectedMaxSharpe(10, -1.0)) {
		t.Error("ExpectedMaxSharpe with negative variance should be NaN")
	}
}

func TestExpectedMaxSharpeMonotoneAndScale(t *testing.T) {
	// The expected maximum must strictly increase with the number of trials.
	prev := math.Inf(-1)
	for _, n := range []int{2, 5, 10, 50, 100, 1000, 10000} {
		got := ExpectedMaxSharpe(n, 1.0)
		if got <= prev {
			t.Errorf("ExpectedMaxSharpe not increasing at N=%d: %v <= %v", n, got, prev)
		}
		prev = got
	}
	// It scales with sqrt(V): E[max|4V] == 2 * E[max|V].
	base := ExpectedMaxSharpe(100, 0.09)
	scaled := ExpectedMaxSharpe(100, 0.36)
	if math.Abs(scaled-2*base) > 1e-9 {
		t.Errorf("sqrt(V) scaling broken: %v vs 2*%v", scaled, base)
	}
}

func TestExpectedMaxSharpeMatchesTrueOrderStatistic(t *testing.T) {
	// The Bailey-LdP closed form approximates the true expected value of the
	// maximum of N i.i.d. standard normals, E[X_(N:N)]. True values below are
	// the mean of the largest normal order statistic, computed by numerical
	// integration of N*x*phi(x)*Phi(x)^(N-1) at 50-digit precision (standard
	// order-statistics result; cf. Harter (1961), "Expected values of normal
	// order statistics", Biometrika 48). The asymptotic approximation is poor
	// for tiny N and tightens as N grows, so the tolerated band is per-N.
	type row struct {
		n    int
		want float64 // true E[max] for standard normal (V=1)
		band float64 // documented max relative error of the closed form
	}
	rows := []row{
		{2, 0.5641895835, 0.08},
		{5, 1.1629644829, 0.03},
		{10, 1.5387527308, 0.025},
		{20, 1.8674752743, 0.02},
		{50, 2.2490743407, 0.015},
		{100, 2.5075937798, 0.01},
		{500, 3.0366990524, 0.006},
		{1000, 3.2414355501, 0.005},
	}
	for _, r := range rows {
		got := ExpectedMaxSharpe(r.n, 1.0)
		rel := math.Abs(got-r.want) / r.want
		if rel > r.band {
			t.Errorf("ExpectedMaxSharpe(%d,1) = %v; true E[max]=%v; rel err %.4f exceeds band %.4f",
				r.n, got, r.want, rel, r.band)
		}
	}
}

func TestDeflatedReducesToPSRWhenNoSelection(t *testing.T) {
	// With a single trial (or zero trial variance) the inflated benchmark is 0,
	// so DSR must equal PSR against a zero benchmark.
	sr, n, skew, kurt := 0.25, 250, -0.3, 5.0
	base := ProbabilisticSharpeRatio(sr, n, skew, kurt, 0.0)
	if got := DeflatedSharpeRatio(sr, n, skew, kurt, 1, 0.09); math.Abs(got-base) > 1e-12 {
		t.Errorf("DSR(nTrials=1) = %v, want PSR@0 = %v", got, base)
	}
	if got := DeflatedSharpeRatio(sr, n, skew, kurt, 100, 0.0); math.Abs(got-base) > 1e-12 {
		t.Errorf("DSR(V=0) = %v, want PSR@0 = %v", got, base)
	}
}

func TestDeflatedDecreasesWithMoreTrials(t *testing.T) {
	// The whole point of the deflation: trying more strategies raises the bar,
	// so the same observed Sharpe earns a strictly lower Deflated Sharpe.
	prev := math.Inf(1)
	for _, nt := range []int{1, 10, 100, 1000, 10000} {
		got := DeflatedSharpeRatio(0.25, 250, 0.0, 3.0, nt, 0.04)
		if got >= prev {
			t.Errorf("DSR not decreasing at nTrials=%d: %v >= %v", nt, got, prev)
		}
		prev = got
	}
}

func TestDeflatedInvalidPropagates(t *testing.T) {
	if !math.IsNaN(DeflatedSharpeRatio(0.25, 250, 0, 3, 0, 0.04)) {
		t.Error("DSR with nTrials<1 should be NaN")
	}
	if !math.IsNaN(DeflatedSharpeRatio(0.25, 1, 0, 3, 100, 0.04)) {
		t.Error("DSR with n<2 should be NaN")
	}
}

func TestMinTrackRecordLengthInvalid(t *testing.T) {
	if !math.IsNaN(MinTrackRecordLength(0.5, 0, 3, 0.0, 0.0)) {
		t.Error("MinTRL confidence=0 should be NaN")
	}
	if !math.IsNaN(MinTrackRecordLength(0.5, 0, 3, 0.0, 1.0)) {
		t.Error("MinTRL confidence=1 should be NaN")
	}
	if !math.IsNaN(MinTrackRecordLength(0.1, 0, 3, 0.1, 0.95)) {
		t.Error("MinTRL with observedSR==benchmark should be NaN")
	}
	if !math.IsNaN(MinTrackRecordLength(0.05, 0, 3, 0.1, 0.95)) {
		t.Error("MinTRL with observedSR<benchmark should be NaN")
	}
}

func TestMinTRLConsistentWithPSR(t *testing.T) {
	// MinTRL is the sample size at which PSR against the benchmark first reaches
	// `confidence`. So PSR at ceil(MinTRL) must be >= confidence and at
	// floor(MinTRL) must be <= confidence — a cross-check between the two
	// closed forms.
	sr, skew, kurt, bench, conf := 0.4, -0.2, 5.0, 0.1, 0.95
	m := MinTrackRecordLength(sr, skew, kurt, bench, conf)
	if math.IsNaN(m) || m < 2 {
		t.Fatalf("unexpected MinTRL %v", m)
	}
	lo := int(math.Floor(m))
	hi := int(math.Ceil(m))
	if lo >= 2 {
		if p := ProbabilisticSharpeRatio(sr, lo, skew, kurt, bench); p > conf+1e-9 {
			t.Errorf("PSR at floor(MinTRL)=%d is %v, should be <= %v", lo, p, conf)
		}
	}
	if p := ProbabilisticSharpeRatio(sr, hi, skew, kurt, bench); p < conf-1e-9 {
		t.Errorf("PSR at ceil(MinTRL)=%d is %v, should be >= %v", hi, p, conf)
	}
}

func TestMinTRLIncreasesWithConfidence(t *testing.T) {
	prev := math.Inf(-1)
	for _, c := range []float64{0.80, 0.90, 0.95, 0.99, 0.999} {
		got := MinTrackRecordLength(0.5, 0.0, 3.0, 0.0, c)
		if got <= prev {
			t.Errorf("MinTRL not increasing with confidence at c=%v: %v <= %v", c, got, prev)
		}
		prev = got
	}
}

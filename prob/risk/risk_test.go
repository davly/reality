package risk

import (
	"math"
	"testing"

	"github.com/davly/reality/testutil"
)

// Golden-file tests for the convention-arbitrated risk-metric suite. Every
// expected value is hand-derived from a stated series in the golden file's
// "_source", so the assertion checks the code against arithmetic, not against
// a value only this package has produced. Tolerances reflect each function's
// precision: exact-closed-form metrics use 1e-12; VaR/CVaR that route through
// the Acklam normal quantile use 1e-6, and the hand-derived Cornish-Fisher
// multi-term expansion uses 1e-4.

func TestGoldenDownside(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/risk/downside.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			returns := testutil.InputFloat64Slice(t, tc, "returns")
			var got float64
			switch {
			case has(tc, "mar") && contains(tc.Description, "DownsideDeviationFullSample"):
				got = DownsideDeviationFullSample(returns, testutil.InputFloat64(t, tc, "mar"))
			case has(tc, "mar") && contains(tc.Description, "DownsideDeviationNegativesOnly"):
				got = DownsideDeviationNegativesOnly(returns, testutil.InputFloat64(t, tc, "mar"))
			case has(tc, "mar") && contains(tc.Description, "SortinoRatioFullSample"):
				got = SortinoRatioFullSample(returns, testutil.InputFloat64(t, tc, "mar"))
			case has(tc, "mar") && contains(tc.Description, "SortinoRatioNegativesOnly"):
				got = SortinoRatioNegativesOnly(returns, testutil.InputFloat64(t, tc, "mar"))
			case has(tc, "threshold"):
				got = OmegaRatio(returns, testutil.InputFloat64(t, tc, "threshold"))
			default:
				t.Fatalf("cannot route case %q", tc.Description)
			}
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGoldenDrawdown(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/risk/drawdown.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			var got float64
			switch {
			case has(tc, "prices"):
				got = MaxDrawdownFromPrices(testutil.InputFloat64Slice(t, tc, "prices"))
			case has(tc, "returns"):
				got = MaxDrawdownFromReturns(testutil.InputFloat64Slice(t, tc, "returns"))
			case has(tc, "annualizedReturn"):
				got = CalmarRatio(testutil.InputFloat64(t, tc, "annualizedReturn"), testutil.InputFloat64(t, tc, "maxDrawdown"))
			default:
				t.Fatalf("cannot route case %q", tc.Description)
			}
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGoldenVaR(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/risk/var.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			returns := testutil.InputFloat64Slice(t, tc, "returns")
			c := testutil.InputFloat64(t, tc, "confidence")
			var got float64
			if contains(tc.Description, "CVaR") {
				got = HistoricalCVaR(returns, c)
			} else {
				got = HistoricalVaR(returns, c)
			}
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGoldenParametricVaR(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/risk/var_parametric.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			mu := testutil.InputFloat64(t, tc, "mean")
			sd := testutil.InputFloat64(t, tc, "stdDev")
			c := testutil.InputFloat64(t, tc, "confidence")
			var got float64
			switch {
			case contains(tc.Description, "CornishFisherVaR"):
				got = CornishFisherVaR(mu, sd, testutil.InputFloat64(t, tc, "skew"), testutil.InputFloat64(t, tc, "excessKurtosis"), c)
			case contains(tc.Description, "ParametricCVaR"):
				got = ParametricCVaR(mu, sd, c)
			default:
				got = ParametricVaR(mu, sd, c)
			}
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGoldenRelative(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/risk/relative.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			var got float64
			if has(tc, "portfolio") {
				got = InformationRatio(testutil.InputFloat64Slice(t, tc, "portfolio"), testutil.InputFloat64Slice(t, tc, "benchmark"))
			} else {
				got = Beta(testutil.InputFloat64Slice(t, tc, "asset"), testutil.InputFloat64Slice(t, tc, "market"))
			}
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGoldenAnnualize(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/risk/annualize.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			var got float64
			if has(tc, "perPeriodMean") {
				got = AnnualizeReturn(testutil.InputFloat64(t, tc, "perPeriodMean"), testutil.InputInt(t, tc, "periodsPerYear"))
			} else {
				got = AnnualizeVolatility(testutil.InputFloat64(t, tc, "perPeriodStdDev"), testutil.InputInt(t, tc, "periodsPerYear"))
			}
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

// --- Structural / invariant tests (not golden-file) -------------------------

func TestTradingDaysConstant(t *testing.T) {
	if TradingDaysPerYear != 252 {
		t.Fatalf("TradingDaysPerYear = %d, want 252", TradingDaysPerYear)
	}
}

// The full-sample denominator (N) is always >= the negatives-only denominator
// (k-1) for k <= N, so the full-sample downside deviation is never larger and
// the full-sample Sortino is never smaller than the negatives-only variant.
// This is the exact directional bias the RD review measured.
func TestSortinoConventionOrdering(t *testing.T) {
	returns := []float64{0.03, -0.02, 0.05, -0.01, 0.02, 0.04, -0.03, 0.01, 0.06, -0.02, 0.03, 0.02}
	full := SortinoRatioFullSample(returns, 0)
	neg := SortinoRatioNegativesOnly(returns, 0)
	if !(full > neg) {
		t.Fatalf("full-sample Sortino (%g) should exceed negatives-only (%g)", full, neg)
	}
	ddFull := DownsideDeviationFullSample(returns, 0)
	ddNeg := DownsideDeviationNegativesOnly(returns, 0)
	if !(ddFull < ddNeg) {
		t.Fatalf("full-sample downside dev (%g) should be below negatives-only (%g)", ddFull, ddNeg)
	}
}

// CVaR (tail average) must be >= VaR (tail boundary) for both the historical
// and the Gaussian estimators.
func TestCVaRDominatesVaR(t *testing.T) {
	returns := []float64{0.02, -0.10, 0.05, 0.00, -0.02, 0.08, 0.01, -0.05, 0.03, 0.04}
	for _, c := range []float64{0.80, 0.90, 0.95} {
		if hCVaR, hVaR := HistoricalCVaR(returns, c), HistoricalVaR(returns, c); hCVaR < hVaR {
			t.Errorf("historical c=%g: CVaR %g < VaR %g", c, hCVaR, hVaR)
		}
		if pCVaR, pVaR := ParametricCVaR(0.001, 0.02, c), ParametricVaR(0.001, 0.02, c); pCVaR < pVaR {
			t.Errorf("parametric c=%g: CVaR %g < VaR %g", c, pCVaR, pVaR)
		}
	}
}

// MaxDrawdownFromReturns must equal MaxDrawdownFromPrices on the compounded
// equity curve the returns generate (par-anchored).
func TestDrawdownReturnsMatchesPrices(t *testing.T) {
	returns := []float64{0.1, -0.2, 0.05, 0.1, -0.3}
	prices := []float64{1.0}
	eq := 1.0
	for _, r := range returns {
		eq *= 1.0 + r
		prices = append(prices, eq)
	}
	fromReturns := MaxDrawdownFromReturns(returns)
	fromPrices := MaxDrawdownFromPrices(prices)
	if math.Abs(fromReturns-fromPrices) > 1e-12 {
		t.Fatalf("drawdown mismatch: returns %g vs prices %g", fromReturns, fromPrices)
	}
}

func TestDegenerateInputsReturnNaN(t *testing.T) {
	nan := func(name string, v float64) {
		if !math.IsNaN(v) {
			t.Errorf("%s: expected NaN, got %g", name, v)
		}
	}
	nan("DownsideDeviationFullSample empty", DownsideDeviationFullSample(nil, 0))
	nan("DownsideDeviationNegativesOnly <2 negatives", DownsideDeviationNegativesOnly([]float64{0.01, -0.02, 0.03}, 0))
	nan("HistoricalVaR empty", HistoricalVaR(nil, 0.95))
	nan("HistoricalVaR bad confidence", HistoricalVaR([]float64{0.01}, 1.0))
	nan("ParametricVaR negative stddev", ParametricVaR(0, -1, 0.95))
	nan("MaxDrawdownFromPrices empty", MaxDrawdownFromPrices(nil))
	nan("InformationRatio length mismatch", InformationRatio([]float64{0.01, 0.02}, []float64{0.01}))
	nan("Beta constant market", Beta([]float64{0.01, 0.02}, []float64{0.03, 0.03}))
	nan("AnnualizeReturn wipeout", AnnualizeReturn(-1.0, 12))
	nan("CalmarRatio negative drawdown", CalmarRatio(0.3, -0.1))
}

// --- small test helpers -----------------------------------------------------

func has(tc testutil.TestCase, key string) bool {
	_, ok := tc.Inputs[key]
	return ok
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

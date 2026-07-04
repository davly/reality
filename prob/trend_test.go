package prob

import (
	"math"
	"testing"

	"github.com/davly/reality/testutil"
)

// ---------------------------------------------------------------------------
// StudentTQuantile
// ---------------------------------------------------------------------------

func TestGoldenStudentTQuantile(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/prob/trend_tquantile.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			p := testutil.InputFloat64(t, tc, "p")
			df := testutil.InputFloat64(t, tc, "df")
			got := StudentTQuantile(p, df)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestStudentTQuantile_CDFRoundTrip(t *testing.T) {
	// StudentTQuantile inverts studentTCDF: CDF(quant(p)) == p.
	for _, df := range []float64{1, 2, 3, 7, 30, 200} {
		for _, p := range []float64{0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99} {
			q := StudentTQuantile(p, df)
			back := studentTCDF(q, df)
			if math.Abs(back-p) > 1e-9 {
				t.Errorf("df=%v p=%v: CDF(quant)=%v, want %v", df, p, back, p)
			}
		}
	}
}

func TestStudentTQuantile_Symmetry(t *testing.T) {
	for _, df := range []float64{1, 4, 15} {
		for _, p := range []float64{0.6, 0.8, 0.95, 0.999} {
			hi := StudentTQuantile(p, df)
			lo := StudentTQuantile(1-p, df)
			if math.Abs(hi+lo) > 1e-9 {
				t.Errorf("df=%v p=%v: q(p)=%v q(1-p)=%v not antisymmetric", df, p, hi, lo)
			}
		}
	}
}

func TestStudentTQuantile_LargeDfApproachesNormal(t *testing.T) {
	// As df -> infinity the t-quantile approaches the normal quantile.
	got := StudentTQuantile(0.975, 1e7)
	want := NormalQuantile(0.975, 0, 1) // ~1.959964
	if math.Abs(got-want) > 1e-4 {
		t.Errorf("t_{0.975, 1e7} = %v, want ~%v (normal)", got, want)
	}
}

func TestStudentTQuantile_InvalidReturnsNaN(t *testing.T) {
	for _, c := range []struct{ p, df float64 }{
		{0, 3}, {1, 3}, {-0.1, 3}, {1.1, 3}, {0.5, 0}, {0.5, -1},
	} {
		if got := StudentTQuantile(c.p, c.df); !math.IsNaN(got) {
			t.Errorf("StudentTQuantile(%v, %v) = %v, want NaN", c.p, c.df, got)
		}
	}
}

// ---------------------------------------------------------------------------
// TrendPredictionInterval
// ---------------------------------------------------------------------------

func TestGoldenTrendPredictionInterval(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/prob/trend_pi.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			data := testutil.InputFloat64Slice(t, tc, "data")
			h := testutil.InputFloat64(t, tc, "h")
			conf := testutil.InputFloat64(t, tc, "conf")
			yhat, lower, upper := TrendPredictionInterval(data, h, conf)
			testutil.AssertFloat64Slice(t, tc, []float64{yhat, lower, upper})
		})
	}
}

func TestTrendPredictionInterval_PerfectLineZeroWidth(t *testing.T) {
	// y = 2x + 1 exactly -> zero residuals -> PI collapses to the point.
	data := []float64{1, 3, 5, 7, 9} // x=0..4
	yhat, lower, upper := TrendPredictionInterval(data, 1, 0.95)
	if math.Abs(yhat-11) > 1e-9 { // x0=5 -> 2*5+1=11
		t.Errorf("yhat=%v, want 11", yhat)
	}
	if math.Abs(upper-lower) > 1e-9 {
		t.Errorf("PI width=%v, want 0 for a perfect fit", upper-lower)
	}
}

func TestTrendPredictionInterval_WidensWithHorizon(t *testing.T) {
	data := []float64{1, 3, 2, 5, 4}
	var prev float64
	for i, h := range []float64{0, 1, 2, 3, 5, 10} {
		_, lo, hi := TrendPredictionInterval(data, h, 0.95)
		width := hi - lo
		if i > 0 && width <= prev {
			t.Errorf("h=%v width=%v did not widen past previous %v", h, width, prev)
		}
		prev = width
	}
}

func TestTrendPredictionInterval_Invalid(t *testing.T) {
	if y, _, _ := TrendPredictionInterval([]float64{1, 2}, 1, 0.95); !math.IsNaN(y) {
		t.Error("n<3 should return NaN")
	}
	if y, _, _ := TrendPredictionInterval([]float64{1, 2, 3}, 1, 0); !math.IsNaN(y) {
		t.Error("conf<=0 should return NaN")
	}
	if y, _, _ := TrendPredictionInterval([]float64{1, 2, 3}, 1, 1); !math.IsNaN(y) {
		t.Error("conf>=1 should return NaN")
	}
}

// ---------------------------------------------------------------------------
// TrendCrossing
//
// Golden values computed independently at 40-digit precision (mpmath): OLS fit
// then findroot on the near/far prediction-band bounds equal to the threshold.
// ---------------------------------------------------------------------------

func TestTrendCrossing_SignificantSlopeBoundedWindow(t *testing.T) {
	// Strong increasing trend, small noise: the crossing is tightly bounded.
	// slope=10, intercept=10, s=0.70710678..., a2>0 => bounded.
	data := []float64{10, 20, 29, 41, 50, 60, 71, 79, 90, 100}
	r := TrendCrossing(data, 150, 0.90)
	if !r.OK {
		t.Fatal("expected OK (trend crosses 150 in the future)")
	}
	if !r.Bounded {
		t.Fatal("expected Bounded=true for a significant slope")
	}
	assertClose(t, "slope", r.Slope, 10.0, 1e-9)
	assertClose(t, "intercept", r.Intercept, 10.0, 1e-9)
	assertClose(t, "sigma", r.Sigma, 0.7071067811865475, 1e-9)
	assertClose(t, "THat", r.THat, 5.0, 1e-9)
	assertClose(t, "TEarliest", r.TEarliest, 4.807198529548853, 1e-7)
	assertClose(t, "TLatest", r.TLatest, 5.196784151115425, 1e-7)
	if !(r.TEarliest < r.THat && r.THat < r.TLatest) {
		t.Errorf("expected TEarliest < THat < TLatest, got %v %v %v", r.TEarliest, r.THat, r.TLatest)
	}
}

func TestTrendCrossing_NoisyShortSeriesUnbounded(t *testing.T) {
	// Short, noisy series: the slope is not significant at 95% (a2<=0), so the
	// latest plausible crossing is unbounded -- the honest "too noisy" verdict.
	data := []float64{10, 5, 20, 8, 25}
	r := TrendCrossing(data, 100, 0.95)
	if r.Bounded {
		t.Fatal("expected Bounded=false for a noisy short series")
	}
	if !math.IsInf(r.TLatest, 1) {
		t.Errorf("expected TLatest=+Inf, got %v", r.TLatest)
	}
	assertClose(t, "slope", r.Slope, 3.3, 1e-9)
	assertClose(t, "THat", r.THat, 24.18181818181818, 1e-7)
	assertClose(t, "TEarliest", r.TEarliest, 5.229338893690225, 1e-6)
	if !(r.TEarliest < r.THat) {
		t.Errorf("earliest %v should precede the point estimate %v", r.TEarliest, r.THat)
	}
}

func TestTrendCrossing_DecreasingToZero(t *testing.T) {
	// Disk free space (GB) declining toward 0: threshold approached from above.
	data := []float64{50, 44, 39, 33, 28, 22, 17, 11}
	r := TrendCrossing(data, 0, 0.95)
	if !r.OK || !r.Bounded {
		t.Fatalf("expected OK and Bounded, got OK=%v Bounded=%v", r.OK, r.Bounded)
	}
	assertClose(t, "slope", r.Slope, -5.523809523809524, 1e-9)
	assertClose(t, "sigma", r.Sigma, 0.2817180849095055, 1e-9)
	assertClose(t, "THat", r.THat, 2.021551724137931, 1e-7)
	assertClose(t, "TEarliest", r.TEarliest, 1.853776598901469, 1e-6)
	assertClose(t, "TLatest", r.TLatest, 2.193423148199506, 1e-6)
}

func TestTrendCrossing_PerfectLineCollapses(t *testing.T) {
	// Zero residuals: earliest == latest == point crossing.
	data := []float64{0, 10, 20, 30, 40} // slope 10, x0 crossing 100 at x=10 -> h=6
	r := TrendCrossing(data, 100, 0.95)
	if !r.OK {
		t.Fatal("expected OK")
	}
	assertClose(t, "THat", r.THat, 6.0, 1e-9)
	assertClose(t, "TEarliest", r.TEarliest, 6.0, 1e-9)
	assertClose(t, "TLatest", r.TLatest, 6.0, 1e-9)
	if !r.Bounded {
		t.Error("perfect-fit crossing should be Bounded")
	}
}

func TestTrendCrossing_NeverCrosses(t *testing.T) {
	// Trend moves away from the threshold: no future point crossing.
	data := []float64{100, 90, 80, 70, 60} // decreasing, threshold above current
	r := TrendCrossing(data, 200, 0.95)
	if r.OK {
		t.Errorf("expected OK=false (trend moves away from 200), THat=%v", r.THat)
	}
}

func TestTrendCrossing_Invalid(t *testing.T) {
	r := TrendCrossing([]float64{1, 2}, 5, 0.95)
	if !math.IsNaN(r.Slope) {
		t.Error("n<3 should yield NaN slope")
	}
	r = TrendCrossing([]float64{1, 2, 3}, 5, 1.5)
	if !math.IsNaN(r.Slope) {
		t.Error("conf out of range should yield NaN slope")
	}
}

func assertClose(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s = %v, want %v (tol %v)", name, got, want, tol)
	}
}

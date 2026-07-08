package slo

import (
	"errors"
	"math"
	"testing"

	"github.com/davly/reality/testutil"
)

// Golden values below reproduce the worked numbers from the Google SRE Workbook,
// Chapter 5 ("Alerting on SLOs"): the recommended multiwindow parameter table
// for a 30-day SLO (2%/1h -> BR* 14.4, 5%/6h -> BR* 6, 10%/3d -> BR* 1), the
// burn-rate/time-to-exhaustion table (BR 1/10/100/1000 for a 99.9% SLO), the
// detection-time formula, and the 1/12 short-window rule. The 30-day period is
// expressed in hours (720h) so the table numbers appear directly. Fixtures live
// in ../testdata/slo/ and cite the source in their _source field.

const tol = 1e-9

// ---------------------------------------------------------------------------
// Golden-file vectors (SRE Workbook Ch.5)
// ---------------------------------------------------------------------------

func TestGoldenErrorBudget(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/slo/error_budget.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			got, err := ErrorBudget(testutil.InputFloat64(t, tc, "slo"))
			if err != nil {
				t.Fatalf("ErrorBudget: unexpected error %v", err)
			}
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGoldenThresholdBurnRate(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/slo/threshold_burn_rate.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			got, err := ThresholdBurnRate(
				testutil.InputFloat64(t, tc, "budgetFraction"),
				testutil.InputFloat64(t, tc, "window"),
				testutil.InputFloat64(t, tc, "period"),
			)
			if err != nil {
				t.Fatalf("ThresholdBurnRate: unexpected error %v", err)
			}
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGoldenBudgetFractionConsumed(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/slo/budget_fraction_consumed.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			got, err := BudgetFractionConsumed(
				testutil.InputFloat64(t, tc, "burnRate"),
				testutil.InputFloat64(t, tc, "window"),
				testutil.InputFloat64(t, tc, "period"),
			)
			if err != nil {
				t.Fatalf("BudgetFractionConsumed: unexpected error %v", err)
			}
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGoldenBurnRateFromErrorRate(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/slo/burn_rate_from_error_rate.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			got, err := BurnRateFromErrorRate(
				testutil.InputFloat64(t, tc, "errorRate"),
				testutil.InputFloat64(t, tc, "slo"),
			)
			if err != nil {
				t.Fatalf("BurnRateFromErrorRate: unexpected error %v", err)
			}
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGoldenTimeToExhaustion(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/slo/time_to_exhaustion.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			got, err := TimeToExhaustion(
				testutil.InputFloat64(t, tc, "remainingBudgetFraction"),
				testutil.InputFloat64(t, tc, "burnRate"),
				testutil.InputFloat64(t, tc, "period"),
			)
			if err != nil {
				t.Fatalf("TimeToExhaustion: unexpected error %v", err)
			}
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGoldenDetectionTime(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/slo/detection_time.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			got, err := DetectionTime(
				testutil.InputFloat64(t, tc, "window"),
				testutil.InputFloat64(t, tc, "thresholdBurnRate"),
				testutil.InputFloat64(t, tc, "actualBurnRate"),
			)
			if err != nil {
				t.Fatalf("DetectionTime: unexpected error %v", err)
			}
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGoldenShortWindow(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/slo/short_window.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			got, err := ShortWindow(testutil.InputFloat64(t, tc, "longWindow"))
			if err != nil {
				t.Fatalf("ShortWindow: unexpected error %v", err)
			}
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

// ---------------------------------------------------------------------------
// Algebraic identities
// ---------------------------------------------------------------------------

// TestThresholdBudgetRoundTrip verifies ThresholdBurnRate and
// BudgetFractionConsumed are exact inverses across a grid of windows/periods.
func TestThresholdBudgetRoundTrip(t *testing.T) {
	periods := []float64{720, 168, 24, 1000}
	windows := []float64{0.5, 1, 6, 72}
	fractions := []float64{0.01, 0.02, 0.05, 0.1, 0.5, 1.0}
	for _, p := range periods {
		for _, w := range windows {
			for _, f := range fractions {
				br, err := ThresholdBurnRate(f, w, p)
				if err != nil {
					t.Fatalf("ThresholdBurnRate(%v,%v,%v): %v", f, w, p, err)
				}
				back, err := BudgetFractionConsumed(br, w, p)
				if err != nil {
					t.Fatalf("BudgetFractionConsumed(%v,%v,%v): %v", br, w, p, err)
				}
				if math.Abs(back-f) > tol {
					t.Errorf("round-trip f=%v w=%v p=%v: got %v", f, w, p, back)
				}
			}
		}
	}
}

// TestBudgetSpentAtDetectionIsDesignFraction verifies the Workbook property that
// the budget consumed at the moment of detection equals the design fraction f,
// independent of the actual burn rate. budget = BR_actual * t_detect / P, and
// t_detect = w * BR*/BR_actual, so budget = w*BR*/P = f.
func TestBudgetSpentAtDetectionIsDesignFraction(t *testing.T) {
	const p = 720.0
	const w = 1.0
	const f = 0.02
	brStar, _ := ThresholdBurnRate(f, w, p) // 14.4
	for _, actual := range []float64{14.4, 20, 50, 100, 500, 1000} {
		dt, err := DetectionTime(w, brStar, actual)
		if err != nil {
			t.Fatalf("DetectionTime: %v", err)
		}
		budget := actual * dt / p
		if math.Abs(budget-f) > tol {
			t.Errorf("actual=%v: budget at detection = %v, want design fraction %v", actual, budget, f)
		}
	}
}

// TestBurnRateNormalisationAgreesWithErrorRate checks the two burn-rate forms
// agree: consuming c of the budget over window w equals an observed error rate of
// c*(1-SLO)*P/w ... simpler: at the SLO error rate the burn rate is 1.
func TestBurnRateNormalisationAgreesWithErrorRate(t *testing.T) {
	// A window that consumes exactly its proportional share (c = w/P) burns at 1.
	br, err := BurnRate(1.0/720.0, 1.0, 720.0)
	if err != nil {
		t.Fatalf("BurnRate: %v", err)
	}
	if math.Abs(br-1.0) > tol {
		t.Errorf("proportional consumption should burn at 1, got %v", br)
	}
	// The error-rate form at the SLO base rate also burns at 1.
	br2, err := BurnRateFromErrorRate(0.001, 0.999)
	if err != nil {
		t.Fatalf("BurnRateFromErrorRate: %v", err)
	}
	if math.Abs(br2-1.0) > tol {
		t.Errorf("at-SLO error rate should burn at 1, got %v", br2)
	}
}

// ---------------------------------------------------------------------------
// Reset time
// ---------------------------------------------------------------------------

func TestResetTime(t *testing.T) {
	tests := []struct {
		name              string
		window, brs, actual float64
		want              float64
	}{
		{"not firing (actual == threshold) resets immediately", 1, 14.4, 14.4, 0},
		{"below threshold resets immediately", 1, 14.4, 10, 0},
		{"1h window, BR* 14.4, actual 100", 1, 14.4, 100, 0.856},
		{"6h window, BR* 6, actual 12 -> half decayed", 6, 6, 12, 3.0},
		{"short 5min window resets fast at actual 100", 1.0 / 12.0, 14.4, 100, (1.0 / 12.0) * (1 - 14.4/100)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ResetTime(tt.window, tt.brs, tt.actual)
			if err != nil {
				t.Fatalf("ResetTime: %v", err)
			}
			if math.Abs(got-tt.want) > tol {
				t.Errorf("ResetTime(%v,%v,%v) = %v, want %v", tt.window, tt.brs, tt.actual, got, tt.want)
			}
		})
	}
}

// TestShortWindowResetIsShorter confirms the point of the 1/12 rule: the short
// window's reset time is a twelfth of the long window's at the same burn rates.
func TestShortWindowResetIsShorter(t *testing.T) {
	longReset, _ := ResetTime(1.0, 14.4, 100)
	sw, _ := ShortWindow(1.0)
	shortReset, _ := ResetTime(sw, 14.4, 100)
	if shortReset >= longReset {
		t.Fatalf("short reset %v should be < long reset %v", shortReset, longReset)
	}
	if math.Abs(shortReset-longReset/12.0) > tol {
		t.Errorf("short reset %v should be long/12 = %v", shortReset, longReset/12.0)
	}
}

// ---------------------------------------------------------------------------
// Infinite limits (honest "never" answers)
// ---------------------------------------------------------------------------

func TestDetectionTimeNeverDetects(t *testing.T) {
	// Actual burn below threshold: the window never reaches BR*.
	got, err := DetectionTime(1, 14.4, 10)
	if err != nil {
		t.Fatalf("DetectionTime: %v", err)
	}
	if !math.IsInf(got, 1) {
		t.Errorf("below-threshold burn should never detect (+Inf), got %v", got)
	}
	// Zero actual burn: never detects.
	got, _ = DetectionTime(1, 14.4, 0)
	if !math.IsInf(got, 1) {
		t.Errorf("zero burn should never detect (+Inf), got %v", got)
	}
}

func TestTimeToExhaustionNeverExhausts(t *testing.T) {
	got, err := TimeToExhaustion(1.0, 0, 720)
	if err != nil {
		t.Fatalf("TimeToExhaustion: %v", err)
	}
	if !math.IsInf(got, 1) {
		t.Errorf("zero burn should never exhaust (+Inf), got %v", got)
	}
	// Already-exhausted budget with any burn exhausts in 0.
	got, _ = TimeToExhaustion(0, 10, 720)
	if got != 0 {
		t.Errorf("no budget left should exhaust in 0, got %v", got)
	}
}

// ---------------------------------------------------------------------------
// Domain-error guards (honesty rules)
// ---------------------------------------------------------------------------

func TestErrorGuards(t *testing.T) {
	if _, err := ErrorBudget(1.0); !errors.Is(err, ErrSLORange) {
		t.Errorf("ErrorBudget(1.0) err = %v, want ErrSLORange", err)
	}
	if _, err := ErrorBudget(-0.1); !errors.Is(err, ErrSLORange) {
		t.Errorf("ErrorBudget(-0.1) err = %v, want ErrSLORange", err)
	}
	if _, err := BurnRate(0.1, 0, 720); !errors.Is(err, ErrNonPositiveWindow) {
		t.Errorf("BurnRate window=0 err = %v, want ErrNonPositiveWindow", err)
	}
	if _, err := BurnRate(0.1, 1, 0); !errors.Is(err, ErrNonPositivePeriod) {
		t.Errorf("BurnRate period=0 err = %v, want ErrNonPositivePeriod", err)
	}
	if _, err := BurnRate(-0.1, 1, 720); !errors.Is(err, ErrBudgetFractionRange) {
		t.Errorf("BurnRate negative consumed err = %v, want ErrBudgetFractionRange", err)
	}
	if _, err := ThresholdBurnRate(1.5, 1, 720); !errors.Is(err, ErrBudgetFractionRange) {
		t.Errorf("ThresholdBurnRate f=1.5 err = %v, want ErrBudgetFractionRange", err)
	}
	if _, err := BudgetFractionConsumed(-1, 1, 720); !errors.Is(err, ErrNegativeBurnRate) {
		t.Errorf("BudgetFractionConsumed negative BR err = %v, want ErrNegativeBurnRate", err)
	}
	if _, err := BurnRateFromErrorRate(0.5, 1.0); !errors.Is(err, ErrSLORange) {
		t.Errorf("BurnRateFromErrorRate slo=1 err = %v, want ErrSLORange", err)
	}
	if _, err := BurnRateFromErrorRate(1.5, 0.999); !errors.Is(err, ErrBudgetFractionRange) {
		t.Errorf("BurnRateFromErrorRate r=1.5 err = %v, want ErrBudgetFractionRange", err)
	}
	if _, err := DetectionTime(0, 1, 1); !errors.Is(err, ErrNonPositiveWindow) {
		t.Errorf("DetectionTime window=0 err = %v, want ErrNonPositiveWindow", err)
	}
	if _, err := DetectionTime(1, -1, 1); !errors.Is(err, ErrNegativeBurnRate) {
		t.Errorf("DetectionTime negative threshold err = %v, want ErrNegativeBurnRate", err)
	}
	if _, err := ResetTime(0, 1, 1); !errors.Is(err, ErrNonPositiveWindow) {
		t.Errorf("ResetTime window=0 err = %v, want ErrNonPositiveWindow", err)
	}
	if _, err := TimeToExhaustion(1.5, 1, 720); !errors.Is(err, ErrBudgetFractionRange) {
		t.Errorf("TimeToExhaustion remaining=1.5 err = %v, want ErrBudgetFractionRange", err)
	}
	if _, err := TimeToExhaustion(1, 1, 0); !errors.Is(err, ErrNonPositivePeriod) {
		t.Errorf("TimeToExhaustion period=0 err = %v, want ErrNonPositivePeriod", err)
	}
	if _, err := ShortWindow(0); !errors.Is(err, ErrNonPositiveWindow) {
		t.Errorf("ShortWindow(0) err = %v, want ErrNonPositiveWindow", err)
	}
}

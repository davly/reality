package slo

import (
	"errors"
	"math"
	"testing"
)

// TestRecommendedWindows reproduces the SRE Workbook Ch.5 recommended 30-day
// (720h) three-tier table: (2%,1h,BR*14.4,page), (5%,6h,BR*6,page),
// (10%,3d/72h,BR*1,ticket), each with a long/12 short window.
func TestRecommendedWindows(t *testing.T) {
	const p = 720.0
	ws, err := RecommendedWindows(p)
	if err != nil {
		t.Fatalf("RecommendedWindows: %v", err)
	}
	if len(ws) != 3 {
		t.Fatalf("got %d tiers, want 3", len(ws))
	}
	want := []Window{
		{Long: 1, Short: 1.0 / 12.0, BurnRateThreshold: 14.4, Severity: "page"},
		{Long: 6, Short: 0.5, BurnRateThreshold: 6, Severity: "page"},
		{Long: 72, Short: 6, BurnRateThreshold: 1, Severity: "ticket"},
	}
	for i, w := range ws {
		if math.Abs(w.Long-want[i].Long) > tol ||
			math.Abs(w.Short-want[i].Short) > tol ||
			math.Abs(w.BurnRateThreshold-want[i].BurnRateThreshold) > tol ||
			w.Severity != want[i].Severity {
			t.Errorf("tier %d = %+v, want %+v", i, w, want[i])
		}
	}
}

// TestRecommendedWindowsScaleWithPeriod checks the table scales: for a 7-day
// (168h) SLO the thresholds are unchanged (they depend only on the fraction and
// the window-as-a-fraction-of-period), while the absolute windows shrink.
func TestRecommendedWindowsScaleWithPeriod(t *testing.T) {
	ws, err := RecommendedWindows(168.0)
	if err != nil {
		t.Fatalf("RecommendedWindows: %v", err)
	}
	wantBR := []float64{14.4, 6, 1}
	for i, w := range ws {
		if math.Abs(w.BurnRateThreshold-wantBR[i]) > tol {
			t.Errorf("tier %d BR* = %v, want %v (thresholds are period-invariant)", i, w.BurnRateThreshold, wantBR[i])
		}
		if math.Abs(w.Short-w.Long/12.0) > tol {
			t.Errorf("tier %d short window not long/12", i)
		}
	}
	if _, err := RecommendedWindows(0); !errors.Is(err, ErrNonPositivePeriod) {
		t.Errorf("RecommendedWindows(0) err = %v, want ErrNonPositivePeriod", err)
	}
}

func TestWindowFires(t *testing.T) {
	w := Window{Long: 1, Short: 1.0 / 12.0, BurnRateThreshold: 14.4}
	tests := []struct {
		name       string
		long, short float64
		want       bool
	}{
		{"both above -> fire", 20, 20, true},
		{"both exactly at threshold -> fire", 14.4, 14.4, true},
		{"long above short below -> no fire (spike recovered)", 20, 5, false},
		{"long below short above -> no fire", 5, 20, false},
		{"both below -> no fire", 5, 5, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := w.Fires(Observation{LongBurnRate: tt.long, ShortBurnRate: tt.short})
			if got != tt.want {
				t.Errorf("Fires(long=%v short=%v) = %v, want %v", tt.long, tt.short, got, tt.want)
			}
		})
	}
}

func TestPolicyEvaluate(t *testing.T) {
	const p = 720.0
	ws, _ := RecommendedWindows(p)
	pol := Policy{SLO: 0.999, Period: p, Windows: ws}

	// Fast total outage: burn rate ~1000 across all windows -> fastest tier fires.
	t.Run("fast outage fires page tier 0", func(t *testing.T) {
		obs := []Observation{
			{LongBurnRate: 1000, ShortBurnRate: 1000},
			{LongBurnRate: 1000, ShortBurnRate: 1000},
			{LongBurnRate: 1000, ShortBurnRate: 1000},
		}
		res, err := pol.Evaluate(obs)
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if !res.Fire || res.TierIndex != 0 || res.Severity != "page" {
			t.Fatalf("res = %+v, want fire tier 0 page", res)
		}
		// Detection: 1h window at actual 1000 -> 1*14.4/1000 = 0.0144h.
		if math.Abs(res.DetectionTime-0.0144) > tol {
			t.Errorf("DetectionTime = %v, want 0.0144", res.DetectionTime)
		}
		// Budget spent at detection = design fraction 2%.
		if math.Abs(res.BudgetSpentAtDetection-0.02) > tol {
			t.Errorf("BudgetSpentAtDetection = %v, want 0.02", res.BudgetSpentAtDetection)
		}
	})

	// Slow leak: burn ~1.5 — only the slow ticket tier (BR* 1) fires.
	t.Run("slow leak fires only ticket tier 2", func(t *testing.T) {
		obs := []Observation{
			{LongBurnRate: 1.5, ShortBurnRate: 1.5}, // below 14.4
			{LongBurnRate: 1.5, ShortBurnRate: 1.5}, // below 6
			{LongBurnRate: 1.5, ShortBurnRate: 1.5}, // above 1 -> fires
		}
		res, err := pol.Evaluate(obs)
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if !res.Fire || res.TierIndex != 2 || res.Severity != "ticket" {
			t.Fatalf("res = %+v, want fire tier 2 ticket", res)
		}
		if math.Abs(res.BudgetSpentAtDetection-0.10) > tol {
			t.Errorf("BudgetSpentAtDetection = %v, want 0.10", res.BudgetSpentAtDetection)
		}
	})

	// Healthy: nothing fires.
	t.Run("healthy fires nothing", func(t *testing.T) {
		obs := []Observation{
			{LongBurnRate: 0.5, ShortBurnRate: 0.5},
			{LongBurnRate: 0.5, ShortBurnRate: 0.5},
			{LongBurnRate: 0.5, ShortBurnRate: 0.5},
		}
		res, err := pol.Evaluate(obs)
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res.Fire || res.TierIndex != -1 {
			t.Fatalf("res = %+v, want no fire", res)
		}
		if !math.IsNaN(res.DetectionTime) || !math.IsNaN(res.BudgetSpentAtDetection) {
			t.Errorf("non-firing result should have NaN detection/budget, got %+v", res)
		}
	})

	// Spike that already recovered: long window hot, short window cool -> no fire.
	t.Run("recovered spike does not fire", func(t *testing.T) {
		obs := []Observation{
			{LongBurnRate: 50, ShortBurnRate: 0.5}, // long hot, short cool (below every BR*)
			{LongBurnRate: 50, ShortBurnRate: 0.5},
			{LongBurnRate: 50, ShortBurnRate: 0.5},
		}
		res, err := pol.Evaluate(obs)
		if err != nil {
			t.Fatalf("Evaluate: %v", err)
		}
		if res.Fire {
			t.Errorf("recovered spike (short window cool) should not fire, got %+v", res)
		}
	})
}

func TestPolicyEvaluateErrors(t *testing.T) {
	ws, _ := RecommendedWindows(720)
	if _, err := (Policy{SLO: 0.999, Period: 720}).Evaluate(nil); !errors.Is(err, ErrEmptyPolicy) {
		t.Errorf("empty policy err = %v, want ErrEmptyPolicy", err)
	}
	if _, err := (Policy{SLO: 0.999, Period: 720, Windows: ws}).Evaluate([]Observation{{}}); !errors.Is(err, ErrObservationCount) {
		t.Errorf("wrong obs count err = %v, want ErrObservationCount", err)
	}
	obs := []Observation{{}, {}, {}}
	if _, err := (Policy{SLO: 1.0, Period: 720, Windows: ws}).Evaluate(obs); !errors.Is(err, ErrSLORange) {
		t.Errorf("bad SLO err = %v, want ErrSLORange", err)
	}
	if _, err := (Policy{SLO: 0.999, Period: 0, Windows: ws}).Evaluate(obs); !errors.Is(err, ErrNonPositivePeriod) {
		t.Errorf("bad period err = %v, want ErrNonPositivePeriod", err)
	}
}

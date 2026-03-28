package prob

import (
	"math"
	"testing"

	"github.com/davly/reality/testutil"
)

// ---------------------------------------------------------------------------
// ClampProbability
// ---------------------------------------------------------------------------

func TestClampProbability(t *testing.T) {
	tests := []struct {
		name string
		p    float64
		want float64
	}{
		{"below min", -0.5, MinProb},
		{"at min", MinProb, MinProb},
		{"mid", 0.5, 0.5},
		{"at max", MaxProb, MaxProb},
		{"above max", 1.5, MaxProb},
		{"zero", 0.0, MinProb},
		{"one", 1.0, MaxProb},
		{"tiny positive", 0.001, MinProb},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ClampProbability(tt.p)
			if got != tt.want {
				t.Errorf("ClampProbability(%v) = %v, want %v", tt.p, got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// ProbToLogOdds / LogOddsToProb roundtrip
// ---------------------------------------------------------------------------

func TestProbToLogOdds(t *testing.T) {
	tests := []struct {
		name string
		p    float64
		want float64
	}{
		{"fair coin", 0.5, 0.0},
		{"high prob", 0.9, math.Log(0.9 / 0.1)},
		{"low prob", 0.1, math.Log(0.1 / 0.9)},
		{"clamped low", 0.0, math.Log(MinProb / (1 - MinProb))},
		{"clamped high", 1.0, math.Log(MaxProb / (1 - MaxProb))},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ProbToLogOdds(tt.p)
			if math.Abs(got-tt.want) > 1e-12 {
				t.Errorf("ProbToLogOdds(%v) = %v, want %v", tt.p, got, tt.want)
			}
		})
	}
}

func TestLogOddsToProb(t *testing.T) {
	tests := []struct {
		name     string
		logOdds  float64
		want     float64
		wantExac bool
	}{
		{"zero log-odds", 0.0, 0.5, true},
		{"large positive", 100.0, MaxProb, true},
		{"large negative", -100.0, MinProb, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := LogOddsToProb(tt.logOdds)
			if tt.wantExac && got != tt.want {
				t.Errorf("LogOddsToProb(%v) = %v, want %v", tt.logOdds, got, tt.want)
			}
		})
	}
}

func TestLogOddsRoundtrip(t *testing.T) {
	probs := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
	for _, p := range probs {
		lo := ProbToLogOdds(p)
		got := LogOddsToProb(lo)
		if math.Abs(got-p) > 1e-12 {
			t.Errorf("roundtrip(%v): ProbToLogOdds -> LogOddsToProb = %v", p, got)
		}
	}
}

// ---------------------------------------------------------------------------
// BayesianUpdate
// ---------------------------------------------------------------------------

func TestBayesianUpdate(t *testing.T) {
	tests := []struct {
		name            string
		prior           float64
		likelihoodRatio float64
		wantApprox      float64
		tol             float64
	}{
		{"uniform prior, LR=2", 0.5, 2.0, 2.0 / 3.0, 1e-10},
		{"uniform prior, LR=1 (no change)", 0.5, 1.0, 0.5, 1e-12},
		{"high prior, LR=0.5", 0.8, 0.5, 0.6666666666666666, 1e-10},
		{"low prior, LR=3", 0.2, 3.0, 0.42857142857142855, 1e-10},
		{"negative LR returns prior", 0.5, -1.0, 0.5, 1e-15},
		{"zero LR returns prior", 0.5, 0.0, 0.5, 1e-15},
		{"near-zero prior", 0.01, 2.0, ClampProbability(0.01 * 2.0 / (0.01*2.0 + 0.99)), 1e-4},
		{"near-one prior", 0.99, 0.5, ClampProbability(0.99 * 0.5 / (0.99*0.5 + 0.01)), 1e-4},
		{"very large LR", 0.5, 1000.0, MaxProb, 1e-10},
		{"very small LR", 0.5, 0.001, MinProb, 1e-10},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BayesianUpdate(tt.prior, tt.likelihoodRatio)
			if math.Abs(got-tt.wantApprox) > tt.tol {
				t.Errorf("BayesianUpdate(%v, %v) = %v, want ~%v (tol %v)",
					tt.prior, tt.likelihoodRatio, got, tt.wantApprox, tt.tol)
			}
		})
	}
}

func TestBayesianUpdateChain(t *testing.T) {
	// Three updates of LR=2 from 0.5: 0.5 -> 2/3 -> 4/5 -> 8/9
	got := BayesianUpdateChain(0.5, []float64{2.0, 2.0, 2.0})
	want := 8.0 / 9.0
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("BayesianUpdateChain(0.5, [2,2,2]) = %v, want %v", got, want)
	}

	// Empty chain returns prior unchanged.
	got = BayesianUpdateChain(0.3, nil)
	if got != 0.3 {
		t.Errorf("BayesianUpdateChain(0.3, nil) = %v, want 0.3", got)
	}
}

// ---------------------------------------------------------------------------
// BrierScore
// ---------------------------------------------------------------------------

func TestBrierScore(t *testing.T) {
	tests := []struct {
		name      string
		predicted float64
		actual    float64
		want      float64
	}{
		{"perfect true", 1.0, 1.0, 0.0},
		{"perfect false", 0.0, 0.0, 0.0},
		{"worst true", 0.0, 1.0, 1.0},
		{"worst false", 1.0, 0.0, 1.0},
		{"coin flip true", 0.5, 1.0, 0.25},
		{"coin flip false", 0.5, 0.0, 0.25},
		{"good prediction", 0.9, 1.0, 0.01},
		{"mild surprise", 0.3, 1.0, 0.49},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BrierScore(tt.predicted, tt.actual)
			if math.Abs(got-tt.want) > 1e-15 {
				t.Errorf("BrierScore(%v, %v) = %v, want %v",
					tt.predicted, tt.actual, got, tt.want)
			}
		})
	}
}

func TestBrierScoreBatch(t *testing.T) {
	// Two perfect predictions.
	got := BrierScoreBatch([]float64{1.0, 0.0}, []float64{1.0, 0.0})
	if got != 0.0 {
		t.Errorf("perfect batch = %v, want 0", got)
	}

	// Mixed batch.
	got = BrierScoreBatch([]float64{0.9, 0.1}, []float64{1.0, 0.0})
	want := (0.01 + 0.01) / 2.0
	if math.Abs(got-want) > 1e-15 {
		t.Errorf("mixed batch = %v, want %v", got, want)
	}

	// Empty returns 0.
	got = BrierScoreBatch(nil, nil)
	if got != 0 {
		t.Errorf("empty batch = %v, want 0", got)
	}

	// Mismatched lengths returns 0.
	got = BrierScoreBatch([]float64{0.5}, []float64{0.5, 0.5})
	if got != 0 {
		t.Errorf("mismatched batch = %v, want 0", got)
	}
}

// ---------------------------------------------------------------------------
// LogLoss
// ---------------------------------------------------------------------------

func TestLogLoss(t *testing.T) {
	tests := []struct {
		name      string
		predicted float64
		actual    float64
		want      float64
		tol       float64
	}{
		{"confident correct true", 0.99, 1.0, -math.Log(0.99), 1e-12},
		{"confident correct false", 0.01, 0.0, -math.Log(0.99), 1e-12},
		{"coin flip true", 0.5, 1.0, -math.Log(0.5), 1e-12},
		{"coin flip false", 0.5, 0.0, -math.Log(0.5), 1e-12},
		{"clamped zero pred true", 0.0, 1.0, -math.Log(MinProb), 1e-12},
		{"clamped one pred false", 1.0, 0.0, -math.Log(1.0 - MaxProb), 1e-12},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := LogLoss(tt.predicted, tt.actual)
			if math.Abs(got-tt.want) > tt.tol {
				t.Errorf("LogLoss(%v, %v) = %v, want %v", tt.predicted, tt.actual, got, tt.want)
			}
		})
	}
}

func TestLogLossBatch(t *testing.T) {
	got := LogLossBatch([]float64{0.99, 0.01}, []float64{1.0, 0.0})
	want := (-math.Log(0.99) + -math.Log(0.99)) / 2.0
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("LogLossBatch = %v, want %v", got, want)
	}

	// Empty.
	got = LogLossBatch(nil, nil)
	if got != 0 {
		t.Errorf("empty LogLossBatch = %v, want 0", got)
	}
}

// ---------------------------------------------------------------------------
// LogOddsPool
// ---------------------------------------------------------------------------

func TestLogOddsPool(t *testing.T) {
	// Equal probs with equal weights -> same value back.
	got := LogOddsPool([]float64{0.5, 0.5, 0.5}, nil)
	if math.Abs(got-0.5) > 1e-12 {
		t.Errorf("equal pool = %v, want 0.5", got)
	}

	// Single element.
	got = LogOddsPool([]float64{0.7}, nil)
	if math.Abs(got-0.7) > 1e-10 {
		t.Errorf("single pool = %v, want ~0.7", got)
	}

	// Empty returns 0.5.
	got = LogOddsPool(nil, nil)
	if got != 0.5 {
		t.Errorf("empty pool = %v, want 0.5", got)
	}

	// All zero weights returns 0.5.
	got = LogOddsPool([]float64{0.8, 0.2}, []float64{0.0, 0.0})
	if got != 0.5 {
		t.Errorf("zero-weight pool = %v, want 0.5", got)
	}

	// Symmetric opinions cancel out to ~0.5.
	got = LogOddsPool([]float64{0.9, 0.1}, []float64{1.0, 1.0})
	if math.Abs(got-0.5) > 1e-12 {
		t.Errorf("symmetric pool = %v, want ~0.5", got)
	}

	// Weighted pool: heavy weight on 0.9.
	got = LogOddsPool([]float64{0.9, 0.1}, []float64{10.0, 1.0})
	if got <= 0.5 {
		t.Errorf("weighted pool should be > 0.5, got %v", got)
	}
}

// ---------------------------------------------------------------------------
// WilsonConfidenceInterval
// ---------------------------------------------------------------------------

func TestWilsonConfidenceInterval(t *testing.T) {
	tests := []struct {
		name     string
		p        float64
		n        int
		z        float64
		wantLow  float64
		wantHigh float64
		tol      float64
	}{
		{"n=0 fallback", 0.5, 0, 1.96, 0.2, 0.8, 1e-12},
		{"n=1, p=1.0", 1.0, 1, 1.96, MinProb, MaxProb, 0.5},
		{"n=100, p=0.5", 0.5, 100, 1.96, 0.40, 0.60, 0.01},
		{"n=1000, p=0.5 narrow", 0.5, 1000, 1.96, 0.468, 0.532, 0.002},
		{"z defaults if <= 0", 0.5, 100, 0.0, 0.40, 0.60, 0.01},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			low, high := WilsonConfidenceInterval(tt.p, tt.n, tt.z)
			if math.Abs(low-tt.wantLow) > tt.tol {
				t.Errorf("low = %v, want ~%v (tol %v)", low, tt.wantLow, tt.tol)
			}
			if math.Abs(high-tt.wantHigh) > tt.tol {
				t.Errorf("high = %v, want ~%v (tol %v)", high, tt.wantHigh, tt.tol)
			}
			if low > high {
				t.Errorf("low (%v) > high (%v)", low, high)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// SimpleAverage
// ---------------------------------------------------------------------------

func TestSimpleAverage(t *testing.T) {
	tests := []struct {
		name   string
		values []float64
		want   float64
		tol    float64
	}{
		{"empty returns 0.5", nil, 0.5, 1e-15},
		{"single value", []float64{0.7}, 0.7, 1e-15},
		{"two values", []float64{0.2, 0.8}, 0.5, 1e-15},
		{"clamped high", []float64{1.5, 1.5}, MaxProb, 1e-15},
		{"three values", []float64{0.1, 0.5, 0.9}, 0.5, 1e-15},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SimpleAverage(tt.values)
			if math.Abs(got-tt.want) > tt.tol {
				t.Errorf("SimpleAverage(%v) = %v, want %v", tt.values, got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// WeightedAverage
// ---------------------------------------------------------------------------

func TestWeightedAverage(t *testing.T) {
	tests := []struct {
		name    string
		values  []float64
		weights []float64
		want    float64
		tol     float64
	}{
		{"empty returns 0.5", nil, nil, 0.5, 1e-15},
		{"equal weights", []float64{0.2, 0.8}, []float64{1, 1}, 0.5, 1e-15},
		{"heavy first", []float64{0.2, 0.8}, []float64{9, 1}, 0.26, 1e-12},
		{"no weights defaults 1", []float64{0.3, 0.7}, nil, 0.5, 1e-15},
		{"all zero weights", []float64{0.5}, []float64{0.0}, 0.5, 1e-15},
		{"negative weights skipped", []float64{0.3, 0.7}, []float64{-1, 1}, 0.7, 1e-15},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := WeightedAverage(tt.values, tt.weights)
			if math.Abs(got-tt.want) > tt.tol {
				t.Errorf("WeightedAverage(%v, %v) = %v, want %v",
					tt.values, tt.weights, got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Median
// ---------------------------------------------------------------------------

func TestMedian(t *testing.T) {
	tests := []struct {
		name   string
		values []float64
		want   float64
	}{
		{"empty returns 0.5", nil, 0.5},
		{"single", []float64{0.7}, 0.7},
		{"odd count", []float64{0.1, 0.5, 0.9}, 0.5},
		{"even count", []float64{0.2, 0.4, 0.6, 0.8}, 0.5},
		{"unsorted input", []float64{0.9, 0.1, 0.5}, 0.5},
		{"all same", []float64{0.3, 0.3, 0.3}, 0.3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Median(tt.values)
			if math.Abs(got-tt.want) > 1e-15 {
				t.Errorf("Median(%v) = %v, want %v", tt.values, got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TrimmedMean
// ---------------------------------------------------------------------------

func TestTrimmedMean(t *testing.T) {
	tests := []struct {
		name         string
		values       []float64
		trimFraction float64
		want         float64
		tol          float64
	}{
		{"empty returns 0.5", nil, 0.1, 0.5, 1e-15},
		{"no trim (frac=0)", []float64{0.1, 0.5, 0.9}, 0.0, 0.5, 1e-15},
		{"10% trim 10 values", []float64{0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.99}, 0.1,
			(0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6 + 0.7 + 0.8) / 8.0, 1e-12},
		{"negative frac treated as 0", []float64{0.2, 0.5, 0.8}, -0.1, 0.5, 1e-15},
		{"frac >= 0.5 treated as 0", []float64{0.2, 0.5, 0.8}, 0.5, 0.5, 1e-15},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := TrimmedMean(tt.values, tt.trimFraction)
			if math.Abs(got-tt.want) > tt.tol {
				t.Errorf("TrimmedMean(%v, %v) = %v, want %v",
					tt.values, tt.trimFraction, got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// ExpectedCalibrationError
// ---------------------------------------------------------------------------

func TestExpectedCalibrationError(t *testing.T) {
	// Well-calibrated data: for each bin, the fraction of actual=1 matches
	// the mean predicted probability. Using 10 bins of 10 predictions each.
	// Bin i has mean predicted = (i+0.5)/10. We put i+1 outcomes as 1.0 in
	// the first i+1 positions and the rest as 0.0 -- but that's complex.
	// Simpler: use predictions that perfectly match their outcomes.
	// predicted=0.0 actual=0.0, predicted=1.0 actual=1.0 gives Brier=0, ECE=0.
	perfect := []PredictionOutcome{
		{0.05, 0.0}, {0.05, 0.0}, {0.05, 0.0}, {0.05, 0.0}, {0.05, 0.0},
		{0.95, 1.0}, {0.95, 1.0}, {0.95, 1.0}, {0.95, 1.0}, {0.95, 1.0},
	}
	ece := ExpectedCalibrationError(perfect, 10)
	// Bin [0,0.1) has mean_predicted=0.05, mean_actual=0.0, error=0.05
	// Bin [0.9,1.0) has mean_predicted=0.95, mean_actual=1.0, error=0.05
	// ECE = (5/10)*0.05 + (5/10)*0.05 = 0.05
	if math.Abs(ece-0.05) > 1e-12 {
		t.Errorf("expected ECE = 0.05 for near-calibrated data, got %v", ece)
	}

	// Empty returns 0.
	ece = ExpectedCalibrationError(nil, 10)
	if ece != 0 {
		t.Errorf("empty ECE = %v, want 0", ece)
	}

	// All predictions in one bin: predict 0.1 but outcome is always 1.
	bad := []PredictionOutcome{
		{0.05, 1.0}, {0.05, 1.0}, {0.05, 1.0}, {0.05, 1.0}, {0.05, 1.0},
	}
	ece = ExpectedCalibrationError(bad, 10)
	if ece < 0.5 {
		t.Errorf("expected high ECE for miscalibrated data, got %v", ece)
	}
}

// ---------------------------------------------------------------------------
// MaximumCalibrationError
// ---------------------------------------------------------------------------

func TestMaximumCalibrationError(t *testing.T) {
	// All predictions 0.05 but outcomes always 1.
	bad := []PredictionOutcome{
		{0.05, 1.0}, {0.05, 1.0}, {0.05, 1.0},
	}
	mce := MaximumCalibrationError(bad, 10)
	if mce < 0.9 {
		t.Errorf("expected high MCE, got %v", mce)
	}

	// Empty returns 0.
	mce = MaximumCalibrationError(nil, 10)
	if mce != 0 {
		t.Errorf("empty MCE = %v, want 0", mce)
	}
}

// ---------------------------------------------------------------------------
// ReliabilityDiagram
// ---------------------------------------------------------------------------

func TestReliabilityDiagram(t *testing.T) {
	preds := []PredictionOutcome{
		{0.15, 0.0}, {0.25, 0.0}, {0.35, 1.0}, {0.45, 0.0},
		{0.55, 1.0}, {0.65, 1.0}, {0.75, 0.0}, {0.85, 1.0},
	}
	buckets := ReliabilityDiagram(preds, 5)
	if len(buckets) != 5 {
		t.Fatalf("expected 5 buckets, got %d", len(buckets))
	}

	// First bucket [0, 0.2): has pred 0.15.
	if buckets[0].Count != 1 {
		t.Errorf("bucket[0].Count = %d, want 1", buckets[0].Count)
	}

	// numBuckets < 1 defaults to 10.
	buckets = ReliabilityDiagram(preds, 0)
	if len(buckets) != 10 {
		t.Errorf("default buckets = %d, want 10", len(buckets))
	}
}

// ---------------------------------------------------------------------------
// IsotonicRegression
// ---------------------------------------------------------------------------

func TestIsotonicRegression(t *testing.T) {
	// Already monotone -> no change.
	monotone := []CalibrationPoint{{0, 0.1}, {1, 0.3}, {2, 0.5}, {3, 0.7}}
	result := IsotonicRegression(monotone)
	for i, p := range result {
		if math.Abs(p.Y-monotone[i].Y) > 1e-15 {
			t.Errorf("monotone[%d].Y = %v, want %v", i, p.Y, monotone[i].Y)
		}
	}

	// Non-monotone: 0.9 followed by 0.1 should be merged.
	nonMono := []CalibrationPoint{{0, 0.9}, {1, 0.1}}
	result = IsotonicRegression(nonMono)
	expectedY := 0.5 // (0.9 + 0.1) / 2
	for i, p := range result {
		if math.Abs(p.Y-expectedY) > 1e-15 {
			t.Errorf("nonMono[%d].Y = %v, want %v", i, p.Y, expectedY)
		}
	}

	// Three-point violation: [0.5, 0.8, 0.3]
	threePoint := []CalibrationPoint{{0, 0.5}, {1, 0.8}, {2, 0.3}}
	result = IsotonicRegression(threePoint)
	// Points 1 and 2 violate -> merge to (0.8+0.3)/2 = 0.55, then 0.5 < 0.55 so ok
	if result[0].Y > result[1].Y+1e-15 || result[1].Y > result[2].Y+1e-15 {
		t.Errorf("isotonic result not monotone: %v, %v, %v",
			result[0].Y, result[1].Y, result[2].Y)
	}

	// All same -> no change.
	same := []CalibrationPoint{{0, 0.5}, {1, 0.5}, {2, 0.5}}
	result = IsotonicRegression(same)
	for i, p := range result {
		if math.Abs(p.Y-0.5) > 1e-15 {
			t.Errorf("same[%d].Y = %v, want 0.5", i, p.Y)
		}
	}

	// Empty returns nil.
	result = IsotonicRegression(nil)
	if result != nil {
		t.Errorf("empty isotonic = %v, want nil", result)
	}

	// Descending input -> all merged to mean.
	desc := []CalibrationPoint{{0, 1.0}, {1, 0.8}, {2, 0.6}, {3, 0.4}, {4, 0.2}}
	result = IsotonicRegression(desc)
	expectedMean := (1.0 + 0.8 + 0.6 + 0.4 + 0.2) / 5.0
	for i, p := range result {
		if math.Abs(p.Y-expectedMean) > 1e-15 {
			t.Errorf("desc[%d].Y = %v, want %v", i, p.Y, expectedMean)
		}
	}

	// X values preserved.
	pts := []CalibrationPoint{{1.5, 0.9}, {2.5, 0.1}}
	result = IsotonicRegression(pts)
	if result[0].X != 1.5 || result[1].X != 2.5 {
		t.Errorf("X values not preserved: got %v, %v", result[0].X, result[1].X)
	}
}

// ---------------------------------------------------------------------------
// Input not mutated (Median, TrimmedMean, IsotonicRegression)
// ---------------------------------------------------------------------------

func TestInputNotMutated(t *testing.T) {
	original := []float64{0.9, 0.1, 0.5}
	backup := make([]float64, len(original))
	copy(backup, original)

	Median(original)
	for i := range original {
		if original[i] != backup[i] {
			t.Errorf("Median mutated input[%d]: %v -> %v", i, backup[i], original[i])
		}
	}

	TrimmedMean(original, 0.1)
	for i := range original {
		if original[i] != backup[i] {
			t.Errorf("TrimmedMean mutated input[%d]: %v -> %v", i, backup[i], original[i])
		}
	}

	pts := []CalibrationPoint{{0, 0.9}, {1, 0.1}}
	ptsBackup := make([]CalibrationPoint, len(pts))
	copy(ptsBackup, pts)

	IsotonicRegression(pts)
	for i := range pts {
		if pts[i] != ptsBackup[i] {
			t.Errorf("IsotonicRegression mutated input[%d]", i)
		}
	}
}

// ---------------------------------------------------------------------------
// Golden-file tests
// ---------------------------------------------------------------------------

func TestGoldenBayesianUpdate(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/prob/bayesian_update.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			prior := testutil.InputFloat64(t, tc, "prior")
			lr := testutil.InputFloat64(t, tc, "likelihoodRatio")
			got := BayesianUpdate(prior, lr)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGoldenBrierScore(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/prob/brier_score.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			predicted := testutil.InputFloat64(t, tc, "predicted")
			actual := testutil.InputFloat64(t, tc, "actual")
			got := BrierScore(predicted, actual)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

// ---------------------------------------------------------------------------
// ConfidenceFromPValue
// ---------------------------------------------------------------------------

func TestConfidenceFromPValue(t *testing.T) {
	tests := []struct {
		name   string
		pValue float64
		want   float64
	}{
		{"standard p=0.05", 0.05, 0.95},
		{"p=0.01", 0.01, 0.99},
		{"p=0.5", 0.5, 0.5},
		{"p=0.0 (full confidence)", 0.0, 1.0},
		{"p=1.0 (no confidence)", 1.0, 0.0},
		{"negative p-value clamped", -0.5, 1.0},
		{"p > 1 clamped", 1.5, 0.0},
		{"p=0.001", 0.001, 0.999},
		{"p=0.1", 0.1, 0.9},
		{"p=0.999", 0.999, 0.001},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ConfidenceFromPValue(tt.pValue)
			if math.Abs(got-tt.want) > 1e-15 {
				t.Errorf("ConfidenceFromPValue(%v) = %v, want %v", tt.pValue, got, tt.want)
			}
		})
	}
}

func TestGoldenConfidenceFromPValue(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/prob/confidence_from_pvalue.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			pValue := testutil.InputFloat64(t, tc, "pValue")
			got := ConfidenceFromPValue(pValue)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

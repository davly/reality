package conformal

import (
	"math"
	"testing"
)

// =========================================================================
// AbsResidual
// =========================================================================

func TestAbsResidual_KnownValues(t *testing.T) {
	s := AbsResidual{}
	cases := []struct {
		predicted, actual, want float64
	}{
		{1.0, 2.0, 1.0},
		{2.0, 1.0, 1.0},
		{0.0, 0.0, 0.0},
		{-1.0, 1.0, 2.0},
		{1.5, -1.5, 3.0},
	}
	for _, tc := range cases {
		got := s.Score(tc.predicted, tc.actual)
		if math.Abs(got-tc.want) > 1e-15 {
			t.Errorf("AbsResidual(%v, %v) = %v, want %v", tc.predicted, tc.actual, got, tc.want)
		}
	}
	if s.Name() != "abs_residual" {
		t.Errorf("Name = %q, want abs_residual", s.Name())
	}
}

// =========================================================================
// NormalizedResidual
// =========================================================================

func TestNormalizedResidual_DividesByStdDev(t *testing.T) {
	// stdDev increases linearly with predicted value -> bigger predicted
	// = wider effective band.
	s := NormalizedResidual{
		StdDevFn: func(p float64) float64 { return 1.0 + p },
		Eps:      1e-9,
	}
	// predicted=0 -> sigma=1 -> score = |2-0|/1 = 2
	if got := s.Score(0.0, 2.0); math.Abs(got-2.0) > 1e-12 {
		t.Errorf("Score(0,2) = %v, want 2", got)
	}
	// predicted=4 -> sigma=5 -> score = |9-4|/5 = 1
	if got := s.Score(4.0, 9.0); math.Abs(got-1.0) > 1e-12 {
		t.Errorf("Score(4,9) = %v, want 1", got)
	}
}

func TestNormalizedResidual_FloorsTinySigma(t *testing.T) {
	s := NormalizedResidual{
		StdDevFn: func(p float64) float64 { return 0.0 }, // pathological
		Eps:      1e-3,
	}
	// |0.5 - 0.0| / 1e-3 = 500
	if got := s.Score(0.0, 0.5); math.Abs(got-500.0) > 1e-9 {
		t.Errorf("Score with zero sigma = %v, want 500", got)
	}
}

func TestNormalizedResidual_NilStdDevFnPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on nil StdDevFn")
		}
	}()
	s := NormalizedResidual{}
	s.Score(0, 1)
}

func TestNormalizedResidual_DefaultEpsAppliedWhenZero(t *testing.T) {
	s := NormalizedResidual{
		StdDevFn: func(p float64) float64 { return 0.0 },
		// Eps unset -> defaults to 1e-9
	}
	got := s.Score(0.0, 1.0)
	// 1.0 / 1e-9 = 1e9
	if math.Abs(got-1e9) > 1.0 {
		t.Errorf("default eps -> score = %v, want ~1e9", got)
	}
}

// =========================================================================
// LogResidual
// =========================================================================

func TestLogResidual_MultiplicativeError(t *testing.T) {
	s := LogResidual{}
	// |log(2/1)| = log(2)
	if got := s.Score(1.0, 2.0); math.Abs(got-math.Log(2)) > 1e-12 {
		t.Errorf("Score(1,2) = %v, want log(2)=%v", got, math.Log(2))
	}
	// |log(1/2)| = log(2) (symmetric in log space)
	if got := s.Score(2.0, 1.0); math.Abs(got-math.Log(2)) > 1e-12 {
		t.Errorf("Score(2,1) = %v, want log(2)=%v", got, math.Log(2))
	}
	// Identity prediction -> score 0.
	if got := s.Score(5.0, 5.0); got != 0 {
		t.Errorf("Score(5,5) = %v, want 0", got)
	}
}

func TestLogResidual_FloorsAtEps(t *testing.T) {
	s := LogResidual{Eps: 1e-6}
	// actual=0 floored to 1e-6, predicted=1 -> |log(1e-6)| = 6 ln(10) ≈ 13.8155
	got := s.Score(1.0, 0.0)
	want := math.Log(1.0 / 1e-6)
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("zero actual floored: score = %v, want %v", got, want)
	}
}

// =========================================================================
// ScoreAll
// =========================================================================

func TestScoreAll_AppliesScorerAcrossPairs(t *testing.T) {
	pred := []float64{1, 2, 3, 4}
	act := []float64{1.5, 2.0, 4.0, 0.0}
	got := ScoreAll(AbsResidual{}, pred, act)
	want := []float64{0.5, 0.0, 1.0, 4.0}
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Errorf("got[%d]=%v, want %v", i, got[i], want[i])
		}
	}
}

func TestScoreAll_LengthMismatchReturnsNil(t *testing.T) {
	if got := ScoreAll(AbsResidual{}, []float64{1, 2, 3}, []float64{1, 2}); got != nil {
		t.Errorf("got %v, want nil for mismatched lengths", got)
	}
}

// =========================================================================
// Composition with SplitQuantile
// =========================================================================

func TestNonconformity_FeedsSplitQuantile(t *testing.T) {
	// End-to-end: build calibration scores via a NonconformityScorer,
	// pass them to SplitQuantile, recover the same q the FW C# corpus
	// would yield for absolute residuals.
	predicted := []float64{0, 0, 0, 0, 0, 0, 0, 0, 0}
	actual := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
	scores := ScoreAll(AbsResidual{}, predicted, actual)
	q, err := SplitQuantile(scores, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	// Same as the FW Conformal_Index_Selects_NPlus1_Times_OneMinusAlpha
	// case: 9 residuals 0.1..0.9, alpha=0.1, rank=ceil(10*0.9)=9 ->
	// q = max = 0.9.
	if math.Abs(q-0.9) > 1e-12 {
		t.Errorf("q = %v, want 0.9", q)
	}
}

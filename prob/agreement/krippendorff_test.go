package agreement

import (
	"math"
	"testing"
)

// nan is a short alias for a missing rating, for readability in the
// literal data matrices below.
var nan = math.NaN()

// hayesKrippendorff2007Data is the 4-observer/12-unit reliability data
// matrix from Krippendorff, K. (2011), "Computing Krippendorff's
// Alpha-Reliability", §C-D (the same dataset is reproduced in Hayes &
// Krippendorff (2007) and as the R `irr::kripp.alpha` reference example):
//
//	Units:        1 2 3 4 5 6 7 8 9 10 11 12
//	Observer A:   1 2 3 3 2 1 4 1 2  .  .  .
//	Observer B:   1 2 3 3 2 2 4 1 2  5  .  3
//	Observer C:   . 3 3 3 2 3 4 2 2  5  1  .
//	Observer D:   1 2 3 3 2 4 4 1 2  5  1  .
func hayesKrippendorff2007Data() [][]float64 {
	return [][]float64{
		{1, 2, 3, 3, 2, 1, 4, 1, 2, nan, nan, nan},
		{1, 2, 3, 3, 2, 2, 4, 1, 2, 5, nan, 3},
		{nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, nan},
		{1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, nan},
	}
}

func assertAlpha(t *testing.T, got, want float64, label string) {
	t.Helper()
	if diff := got - want; diff > 1e-6 || diff < -1e-6 {
		t.Errorf("%s: alpha = %v, want %v", label, got, want)
	}
}

func TestKrippendorffAlpha_Nominal_HayesKrippendorff2007(t *testing.T) {
	alpha, err := KrippendorffAlpha(hayesKrippendorff2007Data(), Nominal)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertAlpha(t, alpha, 113.0/152.0, "nominal")
	// Published headline figure: 0.743.
	if diff := alpha - 0.743; diff > 5e-4 || diff < -5e-4 {
		t.Errorf("alpha = %v, does not round to published 0.743", alpha)
	}
}

func TestKrippendorffAlpha_Ordinal_HayesKrippendorff2007(t *testing.T) {
	alpha, err := KrippendorffAlpha(hayesKrippendorff2007Data(), Ordinal)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertAlpha(t, alpha, 108577.0/133160.0, "ordinal")
	if diff := alpha - 0.815; diff > 5e-4 || diff < -5e-4 {
		t.Errorf("alpha = %v, does not round to published 0.815", alpha)
	}
}

func TestKrippendorffAlpha_Interval_HayesKrippendorff2007(t *testing.T) {
	alpha, err := KrippendorffAlpha(hayesKrippendorff2007Data(), Interval)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertAlpha(t, alpha, 951.0/1120.0, "interval")
	if diff := alpha - 0.849; diff > 5e-4 || diff < -5e-4 {
		t.Errorf("alpha = %v, does not round to published 0.849", alpha)
	}
}

// TestKrippendorffAlpha_Nominal_NoMissingData reproduces Krippendorff
// (2011) §B's smaller no-missing-data example: 2 observers (Ben, Gerry), 12
// units, 5 nominal categories (a..e encoded here as 1..5). alpha = 155/224
// (published: 0.692).
func TestKrippendorffAlpha_Nominal_NoMissingData(t *testing.T) {
	// Ben:   a a b b d c c c e d d a
	// Gerry: b a b b b c c c e d d d
	// a=1 b=2 c=3 d=4 e=5
	data := [][]float64{
		{1, 1, 2, 2, 4, 3, 3, 3, 5, 4, 4, 1},
		{2, 1, 2, 2, 2, 3, 3, 3, 5, 4, 4, 4},
	}
	alpha, err := KrippendorffAlpha(data, Nominal)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertAlpha(t, alpha, 155.0/224.0, "nominal-no-missing")
	if diff := alpha - 0.692; diff > 5e-4 || diff < -5e-4 {
		t.Errorf("alpha = %v, does not round to published 0.692", alpha)
	}
}

// TestKrippendorffAlpha_Nominal_Binary reproduces Krippendorff (2011) §A's
// binary example: 2 observers (Meg, Owen), 10 units, categories {0,1}.
// alpha = 2/21 (published: 0.095).
func TestKrippendorffAlpha_Nominal_Binary(t *testing.T) {
	data := [][]float64{
		{0, 1, 0, 0, 0, 0, 0, 0, 1, 0}, // Meg
		{1, 1, 1, 0, 0, 1, 0, 0, 0, 0}, // Owen
	}
	alpha, err := KrippendorffAlpha(data, Nominal)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertAlpha(t, alpha, 2.0/21.0, "binary")
	if diff := alpha - 0.095; diff > 5e-4 || diff < -5e-4 {
		t.Errorf("alpha = %v, does not round to published 0.095", alpha)
	}
}

func TestKrippendorffAlpha_PerfectAgreement(t *testing.T) {
	data := [][]float64{
		{1, 2, 3, 4, 5},
		{1, 2, 3, 4, 5},
		{1, 2, 3, 4, 5},
	}
	for _, metric := range []Metric{Nominal, Ordinal, Interval} {
		alpha, err := KrippendorffAlpha(data, metric)
		if err != nil {
			t.Fatalf("metric=%v: unexpected error: %v", metric, err)
		}
		if diff := alpha - 1.0; diff > 1e-9 || diff < -1e-9 {
			t.Errorf("metric=%v: alpha = %v, want 1.0 for perfect agreement", metric, alpha)
		}
	}
}

func TestKrippendorffAlpha_TooFewRaters(t *testing.T) {
	_, err := KrippendorffAlpha([][]float64{{1, 2, 3}}, Nominal)
	if err != ErrTooFewRaters {
		t.Errorf("err = %v, want ErrTooFewRaters", err)
	}
}

func TestKrippendorffAlpha_AllLoneUnits(t *testing.T) {
	// Every unit has at most 1 non-missing rating: nothing is pairable.
	data := [][]float64{
		{1, nan, 3},
		{nan, 2, nan},
	}
	_, err := KrippendorffAlpha(data, Nominal)
	if err != ErrTooFewRaters {
		t.Errorf("err = %v, want ErrTooFewRaters", err)
	}
}

func TestKrippendorffAlpha_RaggedRows(t *testing.T) {
	data := [][]float64{
		{1, 2, 3},
		{1, 2},
	}
	_, err := KrippendorffAlpha(data, Nominal)
	if err != ErrRaggedRows {
		t.Errorf("err = %v, want ErrRaggedRows", err)
	}
}

func TestKrippendorffAlpha_EmptyInput(t *testing.T) {
	_, err := KrippendorffAlpha([][]float64{{}, {}}, Nominal)
	if err != ErrEmptyInput {
		t.Errorf("err = %v, want ErrEmptyInput", err)
	}
}

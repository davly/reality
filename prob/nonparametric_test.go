package prob

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// FisherExactTest
// ---------------------------------------------------------------------------

func TestFisher_Classic(t *testing.T) {
	// Classic "lady tasting tea" example from Fisher (1935).
	// 2x2 table: [[3, 1], [1, 3]]
	p := FisherExactTest(3, 1, 1, 3)
	// Known two-tailed p-value for this table: ~0.486
	if p < 0.4 || p > 0.6 {
		t.Errorf("Fisher(3,1,1,3) = %v, want ~0.486", p)
	}
}

func TestFisher_Extreme(t *testing.T) {
	// Perfect association: [[5, 0], [0, 5]]
	p := FisherExactTest(5, 0, 0, 5)
	// Two-tailed p-value should be small.
	if p > 0.01 {
		t.Errorf("Fisher(5,0,0,5) = %v, want small p", p)
	}
}

func TestFisher_NoAssociation(t *testing.T) {
	// No association: [[5, 5], [5, 5]]
	p := FisherExactTest(5, 5, 5, 5)
	if p < 0.9 {
		t.Errorf("Fisher(5,5,5,5) = %v, want near 1.0", p)
	}
}

func TestFisher_OneZero(t *testing.T) {
	// [[10, 0], [5, 5]]
	p := FisherExactTest(10, 0, 5, 5)
	if p > 0.05 {
		t.Errorf("Fisher(10,0,5,5) = %v, want small", p)
	}
}

func TestFisher_Symmetric(t *testing.T) {
	// Fisher test should give same result for transposed tables.
	p1 := FisherExactTest(3, 1, 1, 3)
	p2 := FisherExactTest(1, 3, 3, 1)
	if math.Abs(p1-p2) > 1e-10 {
		t.Errorf("Fisher not symmetric: p1=%v, p2=%v", p1, p2)
	}
}

func TestFisher_AllZero(t *testing.T) {
	p := FisherExactTest(0, 0, 0, 0)
	if math.Abs(p-1.0) > 1e-10 {
		t.Errorf("Fisher(0,0,0,0) = %v, want 1.0", p)
	}
}

func TestFisher_LargeTable(t *testing.T) {
	// Larger table to test numerical stability.
	p := FisherExactTest(20, 5, 10, 25)
	// With strong association, p should be small.
	if p > 0.005 {
		t.Errorf("Fisher(20,5,10,25) = %v, want small", p)
	}
}

func TestFisher_SingleRow(t *testing.T) {
	// [[0, 10], [10, 0]]
	p := FisherExactTest(0, 10, 10, 0)
	if p > 0.001 {
		t.Errorf("Fisher(0,10,10,0) = %v, want very small", p)
	}
}

func TestFisher_SmallTable_22(t *testing.T) {
	// [[2, 2], [2, 2]]
	p := FisherExactTest(2, 2, 2, 2)
	if p < 0.9 {
		t.Errorf("Fisher(2,2,2,2) = %v, want near 1.0", p)
	}
}

func TestFisher_PValueBounds(t *testing.T) {
	// p-value should always be in [0, 1].
	tables := [][4]int{
		{3, 1, 1, 3}, {5, 0, 0, 5}, {10, 10, 10, 10},
		{1, 0, 0, 1}, {0, 5, 5, 0}, {7, 3, 2, 8},
	}
	for _, tbl := range tables {
		p := FisherExactTest(tbl[0], tbl[1], tbl[2], tbl[3])
		if p < 0 || p > 1+1e-10 {
			t.Errorf("Fisher(%v) = %v, out of [0, 1]", tbl, p)
		}
	}
}

// ---------------------------------------------------------------------------
// MannWhitneyU
// ---------------------------------------------------------------------------

func TestMW_IdenticalSamples(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{1, 2, 3, 4, 5}
	U, p := MannWhitneyU(x, y)
	_ = U
	// Identical samples — p should be large (fail to reject null).
	if p < 0.5 {
		t.Errorf("MW identical: p=%v, want large", p)
	}
}

func TestMW_Separated(t *testing.T) {
	// Two clearly separated groups.
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{10, 11, 12, 13, 14}
	U, p := MannWhitneyU(x, y)
	if U != 0 {
		t.Errorf("MW separated: U=%v, want 0", U)
	}
	if p > 0.05 {
		t.Errorf("MW separated: p=%v, want small", p)
	}
}

func TestMW_Overlap(t *testing.T) {
	x := []float64{1, 3, 5, 7, 9}
	y := []float64{2, 4, 6, 8, 10}
	U, p := MannWhitneyU(x, y)
	_ = U
	// Near-identical interleaved groups — p should be large.
	if p < 0.3 {
		t.Errorf("MW overlap: p=%v, want large", p)
	}
}

func TestMW_EmptyX(t *testing.T) {
	U, p := MannWhitneyU(nil, []float64{1, 2})
	if !math.IsNaN(U) || !math.IsNaN(p) {
		t.Errorf("MW empty x: U=%v p=%v, want NaN", U, p)
	}
}

func TestMW_EmptyY(t *testing.T) {
	U, p := MannWhitneyU([]float64{1, 2}, nil)
	if !math.IsNaN(U) || !math.IsNaN(p) {
		t.Errorf("MW empty y: U=%v p=%v, want NaN", U, p)
	}
}

func TestMW_UnequalSizes(t *testing.T) {
	x := []float64{1, 2, 3}
	y := []float64{10, 11, 12, 13, 14, 15}
	U, p := MannWhitneyU(x, y)
	if U != 0 {
		t.Errorf("MW unequal sizes: U=%v, want 0", U)
	}
	if p > 0.05 {
		t.Errorf("MW unequal sizes: p=%v, want small", p)
	}
}

func TestMW_Symmetric(t *testing.T) {
	// MannWhitneyU(x, y) should give same U and p as MannWhitneyU(y, x).
	x := []float64{1, 3, 5, 7}
	y := []float64{2, 4, 6, 8}
	u1, p1 := MannWhitneyU(x, y)
	u2, p2 := MannWhitneyU(y, x)
	if u1 != u2 {
		t.Errorf("MW not symmetric: U1=%v, U2=%v", u1, u2)
	}
	if math.Abs(p1-p2) > 1e-10 {
		t.Errorf("MW not symmetric: p1=%v, p2=%v", p1, p2)
	}
}

func TestMW_WithTies(t *testing.T) {
	// Several tied values.
	x := []float64{1, 2, 2, 3, 4}
	y := []float64{2, 3, 3, 4, 5}
	U, p := MannWhitneyU(x, y)
	_ = U
	// Slightly different distributions — p should be moderate.
	if p < 0 || p > 1 {
		t.Errorf("MW ties: p=%v, out of bounds", p)
	}
}

func TestMW_SingleElements(t *testing.T) {
	x := []float64{1}
	y := []float64{10}
	U, p := MannWhitneyU(x, y)
	if U != 0 {
		t.Errorf("MW single: U=%v, want 0", U)
	}
	_ = p // p may not be reliable for n=1 but should be finite
	if math.IsNaN(p) || math.IsInf(p, 0) {
		t.Errorf("MW single: p=%v, want finite", p)
	}
}

func TestMW_PValueBounds(t *testing.T) {
	tests := []struct {
		x, y []float64
	}{
		{[]float64{1, 2, 3}, []float64{4, 5, 6}},
		{[]float64{1, 1, 1}, []float64{1, 1, 1}},
		{[]float64{-5, -3}, []float64{3, 5}},
	}
	for i, tt := range tests {
		_, p := MannWhitneyU(tt.x, tt.y)
		if p < 0 || p > 1+1e-10 {
			t.Errorf("MW test %d: p=%v, out of [0, 1]", i, p)
		}
	}
}

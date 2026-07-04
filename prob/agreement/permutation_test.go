package agreement

import (
	"math"
	"testing"
)

// darwinDiffs are the 15 cross- minus self-fertilised Zea mays plant-height
// differences (eighths of an inch) from Darwin's data as used in Fisher
// (1935), The Design of Experiments §21. Sum = 314.
var darwinDiffs = []float64{49, -67, 8, 16, 6, 23, 28, 41, 14, 29, 56, 24, 75, 60, -48}

// TestPairedPermutation_HandEnumerated checks the fully hand-enumerable case
// from the doc comment: differences (1,2,3), S_obs = 6. Only (+,+,+) and
// (-,-,-) reach |S| >= 6, so p = 2/8 = 0.25.
func TestPairedPermutation_HandEnumerated(t *testing.T) {
	x := []float64{1, 2, 3}
	y := []float64{0, 0, 0}
	stat, p, err := PairedPermutationTest(x, y)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stat != 6 {
		t.Errorf("statistic = %v, want 6", stat)
	}
	if p != 0.25 {
		t.Errorf("p = %v, want 0.25", p)
	}
}

// TestPairedPermutation_Darwin reproduces Fisher's (1935) randomization test on
// Darwin's Zea mays data: S_obs = 314; 863 of the 2^15 arrangements have
// S >= 314 (Fisher 1935; Ernst 2004), so the exact two-sided p-value is
// 1726/32768 = 0.052673828125.
func TestPairedPermutation_Darwin(t *testing.T) {
	y := make([]float64, len(darwinDiffs)) // zeros: x - y = darwinDiffs
	stat, p, err := PairedPermutationTest(darwinDiffs, y)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stat != 314 {
		t.Errorf("statistic = %v, want 314", stat)
	}
	if want := 1726.0 / 32768.0; math.Abs(p-want) > 1e-15 {
		t.Errorf("p = %v, want %v (Fisher 1935 / Ernst 2004: 1726/32768)", p, want)
	}
}

// TestPairedPermutation_UsesPairing confirms the test consumes x and y as
// paired: subtracting y from x must reproduce the Darwin differences and the
// same p, i.e. it is the differences that drive the result.
func TestPairedPermutation_UsesPairing(t *testing.T) {
	x := []float64{10, 3, 12}
	y := []float64{9, 1, 9} // differences (1, 2, 3)
	_, p, err := PairedPermutationTest(x, y)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p != 0.25 {
		t.Errorf("p = %v, want 0.25", p)
	}
}

// TestPairedPermutation_AllZeroDifferences: every difference is 0, so every
// arrangement gives S = 0 = |S_obs|; all 2^n arrangements are "as extreme",
// p = 1.
func TestPairedPermutation_AllZeroDifferences(t *testing.T) {
	x := []float64{5, 5, 5, 5}
	y := []float64{5, 5, 5, 5}
	stat, p, err := PairedPermutationTest(x, y)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stat != 0 || p != 1.0 {
		t.Errorf("got stat=%v p=%v, want stat=0 p=1", stat, p)
	}
}

// TestPairedPermutation_ZeroDiffInvariance: adding a pair with a zero
// difference must not change the p-value (a zero difference is sign-invariant).
func TestPairedPermutation_ZeroDiffInvariance(t *testing.T) {
	_, p1, _ := PairedPermutationTest([]float64{1, 2, 3}, []float64{0, 0, 0})
	_, p2, _ := PairedPermutationTest([]float64{1, 2, 3, 7}, []float64{0, 0, 0, 7})
	if math.Abs(p1-p2) > 1e-15 {
		t.Errorf("zero-diff pair changed p: %v vs %v", p1, p2)
	}
}

// TestPairedPermutation_SymmetricInSwap: swapping x and y negates every
// difference, which leaves the two-sided p unchanged.
func TestPairedPermutation_SymmetricInSwap(t *testing.T) {
	x := []float64{49, -67, 8, 16, 6}
	y := []float64{0, 0, 0, 0, 0}
	_, p1, _ := PairedPermutationTest(x, y)
	_, p2, _ := PairedPermutationTest(y, x)
	if math.Abs(p1-p2) > 1e-15 {
		t.Errorf("swap changed two-sided p: %v vs %v", p1, p2)
	}
}

// TestPairedPermutation_ExactCountBrute independently brute-forces the p-value
// for a small vector and checks the implementation matches to the exact count.
func TestPairedPermutation_ExactCountBrute(t *testing.T) {
	d := []float64{4, -2, 7, 1, -5, 3}
	n := len(d)
	var sObs, absObs float64
	for _, v := range d {
		sObs += v
	}
	absObs = math.Abs(sObs)
	var extreme int
	for mask := 0; mask < (1 << n); mask++ {
		var s float64
		for j := 0; j < n; j++ {
			if mask&(1<<j) != 0 {
				s += d[j]
			} else {
				s -= d[j]
			}
		}
		if math.Abs(s) >= absObs {
			extreme++
		}
	}
	want := float64(extreme) / float64(int(1)<<n)
	y := make([]float64, n)
	_, p, err := PairedPermutationTest(d, y)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if math.Abs(p-want) > 1e-15 {
		t.Errorf("p = %v, want brute-forced %v", p, want)
	}
}

func TestPairedPermutation_Errors(t *testing.T) {
	if _, _, err := PairedPermutationTest([]float64{1, 2}, []float64{1}); err != ErrLengthMismatch {
		t.Errorf("err = %v, want ErrLengthMismatch", err)
	}
	if _, _, err := PairedPermutationTest(nil, nil); err != ErrEmptyInput {
		t.Errorf("err = %v, want ErrEmptyInput", err)
	}
	big := make([]float64, MaxExactPermN+1)
	if _, _, err := PairedPermutationTest(big, big); err != ErrSampleTooLargeForExact {
		t.Errorf("err = %v, want ErrSampleTooLargeForExact", err)
	}
}

// TestPairedPermutation_NonFinite: a NaN or Inf difference yields NaN statistic
// and NaN p (documented behaviour), not a silently wrong 0.
func TestPairedPermutation_NonFinite(t *testing.T) {
	_, p, err := PairedPermutationTest([]float64{1, math.NaN(), 3}, []float64{0, 0, 0})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !math.IsNaN(p) {
		t.Errorf("p = %v, want NaN for NaN input", p)
	}
	_, p2, _ := PairedPermutationTest([]float64{1, math.Inf(1), 3}, []float64{0, 0, 0})
	if !math.IsNaN(p2) {
		t.Errorf("p = %v, want NaN for Inf input", p2)
	}
}

// TestPairedPermutation_BoundaryN runs at exactly MaxExactPermN to confirm the
// full-enumeration boundary is inclusive and completes.
func TestPairedPermutation_BoundaryN(t *testing.T) {
	x := make([]float64, MaxExactPermN)
	y := make([]float64, MaxExactPermN)
	for i := range x {
		x[i] = float64(i + 1)
	}
	_, p, err := PairedPermutationTest(x, y)
	if err != nil {
		t.Fatalf("unexpected error at n=%d: %v", MaxExactPermN, err)
	}
	if p <= 0 || p > 1 {
		t.Errorf("p = %v out of (0,1]", p)
	}
}

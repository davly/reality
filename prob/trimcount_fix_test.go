package prob

import (
	"math"
	"testing"
)

// =========================================================================
// trimCountFor — IEEE-754 off-by-one regression guard (floor-direction
// mirror of the prob/conformal conformalRank fix, gridlock ec7c239 lineage)
// =========================================================================

// exactTrimFromMilliFraction computes floor(n*f) EXACTLY for
// f = fMilli/1000 using integer arithmetic. The trim-count formula is
// defined on the decimal fraction the caller wrote (e.g. 0.29), so the
// Go float path must agree with the decimal-exact count — that is
// precisely what the float off-by-one violated and trimCountFor's
// near-integer snap restores.
func exactTrimFromMilliFraction(n, fMilli int) int {
	return (fMilli * n) / 1000 // integer division IS floor for non-negatives
}

// TestTrimCountFor_NoFloatOffByOne_MilliGrid sweeps every millipoint
// trim fraction in [0, 0.5) against every n up to 500 and demands the
// float path agree with integer-exact arithmetic. Pre-fix, the naive
// `int(math.Floor(float64(n)*trimFraction))` disagrees on 15 pairs in
// this grid — e.g. (0.29, 100) trims 28 per side instead of the
// decimal-exact 29, leaving one extreme observation per side in the mean.
func TestTrimCountFor_NoFloatOffByOne_MilliGrid(t *testing.T) {
	bad := 0
	for fMilli := 0; fMilli < 500; fMilli++ {
		frac := float64(fMilli) / 1000.0
		for n := 1; n <= 500; n++ {
			got := trimCountFor(n, frac)
			want := exactTrimFromMilliFraction(n, fMilli)
			if got != want {
				bad++
				if bad <= 10 {
					t.Errorf("f=%.3f n=%d: trimCountFor=%d, decimal-exact=%d",
						frac, n, got, want)
				}
			}
		}
	}
	if bad > 10 {
		t.Errorf("... and %d further mismatches suppressed", bad-10)
	}
}

// TestTrimmedMean_FloatFloorRegression pins the observable end-to-end
// behaviour: 100 values of which the 29 smallest are 0.1 and the rest
// 0.5, trimmed at 0.29 per side, must average ONLY the 0.5 block
// (decimal-exact trim 29 removes every 0.1). Pre-fix, floor gave 28 and
// one 0.1 leaked into the mean: (0.1 + 43*0.5)/44 = 0.49090909...
func TestTrimmedMean_FloatFloorRegression(t *testing.T) {
	values := make([]float64, 100)
	for i := range values {
		if i < 29 {
			values[i] = 0.1
		} else {
			values[i] = 0.5
		}
	}
	got := TrimmedMean(values, 0.29)
	if math.Abs(got-0.5) > 1e-12 {
		t.Errorf("TrimmedMean(29x0.1 + 71x0.5, 0.29) = %v, want 0.5 (decimal-exact trim 29)", got)
	}
}

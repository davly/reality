package risk

import (
	"math"
	"testing"
)

// =========================================================================
// tailCount — IEEE-754 off-by-one regression guard (sibling of the
// prob/conformal conformalRank fix, gridlock ec7c239 lineage)
// =========================================================================

// exactTailFromMilliConfidence computes ceil((1-c)*n) EXACTLY for
// c = cMilli/1000 using integer arithmetic. The tail-count formula is
// defined on the decimal confidence the caller wrote (e.g. 0.95), so the
// Go float path must agree with the decimal-exact count — that is
// precisely what the float off-by-one violated and tailCount's
// near-integer snap restores.
func exactTailFromMilliConfidence(n, cMilli int) int {
	// (1 - cMilli/1000) * n = (1000-cMilli)*n / 1000.
	num := (1000 - cMilli) * n
	tail := num / 1000
	if num%1000 != 0 {
		tail++ // ceil
	}
	return tail
}

// TestTailCount_NoFloatOffByOne_MilliGrid sweeps every millipoint
// confidence in (0,1) against every n up to 500 and demands the float
// path agree with integer-exact arithmetic. Pre-fix, the naive
// `int(math.Ceil((1.0-confidence)*float64(n)))` disagrees on 438 pairs
// for c in [0.500, 0.999] alone — including the flagship pairs
// (0.95, 100), (0.95, 200), (0.99, 100) and (0.975, 200).
func TestTailCount_NoFloatOffByOne_MilliGrid(t *testing.T) {
	bad := 0
	for cMilli := 1; cMilli <= 999; cMilli++ {
		confidence := float64(cMilli) / 1000.0
		for n := 1; n <= 500; n++ {
			got := tailCount(n, confidence)
			want := exactTailFromMilliConfidence(n, cMilli)
			if got != want {
				bad++
				if bad <= 10 {
					t.Errorf("c=%.3f n=%d: tailCount=%d, decimal-exact=%d",
						confidence, n, got, want)
				}
			}
		}
	}
	if bad > 10 {
		t.Errorf("... and %d further mismatches suppressed", bad-10)
	}
}

// TestHistoricalCVaR_FlagshipConfidences_NoInflatedTail pins the
// end-to-end behaviour at the confidence levels real callers use.
// Returns are 1..n so the tail mean is exact integer arithmetic: with
// the decimal-exact tail k the CVaR is -mean(1..k) = -(k+1)/2. Pre-fix,
// every case below averaged ONE EXTRA (less-bad) observation into the
// tail (e.g. c=0.95, n=100 gave -3.5 = -mean(1..6) instead of -3.0).
func TestHistoricalCVaR_FlagshipConfidences_NoInflatedTail(t *testing.T) {
	cases := []struct {
		confidence float64
		n          int
		wantTail   int // decimal-exact ceil((1-c)*n)
	}{
		{0.95, 20, 1},   // float ceil gives 2
		{0.95, 100, 5},  // float ceil gives 6
		{0.95, 200, 10}, // float ceil gives 11
		{0.975, 200, 5}, // float ceil gives 6
		{0.99, 100, 1},  // float ceil gives 2
		{0.99, 500, 5},  // float ceil gives 6
		// unaffected controls (float path already exact)
		{0.90, 100, 10},
		{0.80, 10, 2},
		{0.75, 10, 3},
	}
	for _, c := range cases {
		returns := make([]float64, c.n)
		for i := range returns {
			returns[i] = float64(i + 1) // sorted ascending: 1..n
		}
		want := -float64(c.wantTail+1) / 2.0 // -mean(1..k)
		got := HistoricalCVaR(returns, c.confidence)
		if math.Abs(got-want) > 1e-12 {
			t.Errorf("c=%g n=%d: HistoricalCVaR = %v, want %v (decimal-exact tail %d)",
				c.confidence, c.n, got, want, c.wantTail)
		}
	}
}

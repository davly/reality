package sequence

import (
	"math"
	"testing"
)

// TestNeedlemanWunsch_NaNGapNoPanic pins the fix: a NaN gap penalty poisons the
// dp matrix with NaN; every traceback equality comparison against a NaN cell is
// false, so the left-branch was always taken and ran j off the end -> "index out
// of range [-1]" panic. The boundary-respecting traceback must return (a NaN
// score is fine) without crashing.
func TestNeedlemanWunsch_NaNGapNoPanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("NeedlemanWunsch with NaN gap panicked: %v", r)
		}
	}()
	NeedlemanWunsch("AB", "AC", 1, -1, math.NaN())
	NeedlemanWunsch("AAAA", "T", 1, -1, math.NaN())
}

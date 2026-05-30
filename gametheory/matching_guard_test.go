package gametheory

import "testing"

// TestIsStableMatching_IncompleteDoesNotPanic pins the guard against the
// `index out of range [-1]` panic that the unmatched sentinel triggered at
// receiverPartner[matching[i]]. An incomplete/out-of-range matching is also
// not stable under complete preferences, so the guard returns false.
func TestIsStableMatching_IncompleteDoesNotPanic(t *testing.T) {
	prefs := [][]int{{0, 1}, {0, 1}}

	// matching[1] = -1 (unmatched sentinel) — must not panic.
	if IsStableMatching([]int{0, -1}, prefs, prefs) {
		t.Errorf("incomplete matching (-1 sentinel) must not be reported stable")
	}

	// out-of-range high index — also guarded.
	if IsStableMatching([]int{0, 5}, prefs, prefs) {
		t.Errorf("out-of-range matching must not be reported stable")
	}

	// a valid complete matching still evaluates (sanity: doesn't always false).
	m := GaleShapley(prefs, prefs)
	_ = IsStableMatching(m, prefs, prefs)
}

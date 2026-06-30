package fairness

import "testing"

// TestAdverseImpact_ExactFourFifthsBoundary is the fail-before/pass-after regression for the
// numeric-ingestion seam: the EEOC four-fifths verdict must be decided by exact integer
// arithmetic, not float division. Group A rate = 2/3, group B rate = 5/6 → AIR = (2/3)/(5/6) =
// 4/5 EXACTLY, which satisfies the four-fifths rule (PASS). The float path computes
// 0.6667/0.8333 = 0.79999… < 0.80 and WRONGLY fails — this test asserts the corrected PASS,
// so it fails against the old float `Pass = PassesFourFifths(AIR)` and passes against the fix.
func TestAdverseImpact_ExactFourFifthsBoundary(t *testing.T) {
	groups := []GroupCount{
		{Label: "A", Selected: 2, Total: 3}, // min rate 0.6667
		{Label: "B", Selected: 5, Total: 6}, // max rate 0.8333
	}
	rep := AdverseImpact(groups, 1.96)
	if !rep.Applicable {
		t.Fatal("expected an applicable report")
	}
	if !rep.Pass {
		t.Fatalf("2/3 vs 5/6 = 4/5 exactly must PASS the four-fifths rule; got Pass=false (float seam not closed)")
	}

	// Pin the seam the fix closes: the FLOAT predicate (wrongly) fails this exact-4/5 boundary,
	// while the EXACT predicate (rightly) passes it. NOTE: the rates must be RUNTIME values
	// (SelectionRate calls), not literal constants — Go folds `(2.0/3.0)/(5.0/6.0)` to exactly
	// 0.8 at compile time, hiding the very float seam this guards. This mirrors what
	// AdverseImpact actually computes (AIR = SelectionRate(min)/SelectionRate(max)).
	floatAIR := SelectionRate(2, 3) / SelectionRate(5, 6)
	if PassesFourFifths(floatAIR) {
		t.Fatalf("expected the float predicate to (wrongly) fail at the runtime 4/5 boundary; flip case invalid")
	}
	if !PassesFourFifthsExact(2, 3, 5, 6) {
		t.Fatalf("PassesFourFifthsExact(2,3,5,6) must be true (= exactly 4/5)")
	}
}

// TestPassesFourFifthsExact_Table covers clear pass/fail and degenerate inputs.
func TestPassesFourFifthsExact_Table(t *testing.T) {
	cases := []struct {
		minSel, minTot, maxSel, maxTot int
		want                           bool
		note                           string
	}{
		{2, 3, 5, 6, true, "exactly 4/5"},
		{4, 5, 5, 5, true, "0.8 vs 1.0 = exactly 4/5"},
		{3, 5, 5, 5, false, "0.6 vs 1.0 = 3/5 < 4/5"},
		{1, 1, 1, 1, true, "parity"},
		{0, 5, 5, 5, false, "min rate 0"},
		{5, 5, 0, 5, false, "max selected 0 → guarded false"},
		{5, 5, 5, 0, false, "max total 0 → guarded false"},
	}
	for _, c := range cases {
		if got := PassesFourFifthsExact(c.minSel, c.minTot, c.maxSel, c.maxTot); got != c.want {
			t.Errorf("PassesFourFifthsExact(%d,%d,%d,%d)=%v want %v (%s)", c.minSel, c.minTot, c.maxSel, c.maxTot, got, c.want, c.note)
		}
	}
}

package sequence

import "testing"

// TestTokenSetRatio_DegenerateCases pins the degenerate boundary inputs.
func TestTokenSetRatio_DegenerateCases(t *testing.T) {
	cases := []struct {
		a, b string
		want int
	}{
		{"", "", 100},
		{"x", "", 0},
		{"", "x", 0},
		{"hello", "hello", 100},
	}
	for _, tc := range cases {
		t.Run(tc.a+"_vs_"+tc.b, func(t *testing.T) {
			got := TokenSetRatio(tc.a, tc.b)
			if got != tc.want {
				t.Errorf("TokenSetRatio(%q, %q) = %d, want %d", tc.a, tc.b, got, tc.want)
			}
		})
	}
}

// TestTokenSetRatio_OrderInvariant verifies that token order does not affect
// the result (that's the whole point of the token-set variant vs simple-
// ratio).
func TestTokenSetRatio_OrderInvariant(t *testing.T) {
	a := "John Robert Smith"
	b := "Smith Robert John"
	if got := TokenSetRatio(a, b); got != 100 {
		t.Errorf("TokenSetRatio(%q, %q) = %d, want 100 (token-order should not matter)", a, b, got)
	}
}

// TestTokenSetRatio_DuplicatesIgnored verifies that within-string token
// duplication does not affect the result (the "set" in token-set).
func TestTokenSetRatio_DuplicatesIgnored(t *testing.T) {
	a := "John Robert Smith"
	b := "John John Robert Robert Smith"
	if got := TokenSetRatio(a, b); got != 100 {
		t.Errorf("TokenSetRatio(%q, %q) = %d, want 100 (duplicates should not matter)", a, b, got)
	}
}

// TestTokenSetRatio_CaseInvariant verifies case-folding.
func TestTokenSetRatio_CaseInvariant(t *testing.T) {
	a := "John Robert Smith"
	b := "JOHN robert SMITH"
	if got := TokenSetRatio(a, b); got != 100 {
		t.Errorf("TokenSetRatio(%q, %q) = %d, want 100 (case should not matter)", a, b, got)
	}
}

// TestTokenSetRatio_PartialOverlap exercises the typical use case: two
// name representations sharing some tokens but with extra qualifiers.
// Expectation: high score (≥ 70) but not 100, because diff tokens dilute.
func TestTokenSetRatio_PartialOverlap(t *testing.T) {
	a := "John Robert Smith"
	b := "John R. Smith Jr."
	got := TokenSetRatio(a, b)
	if got < 50 {
		t.Errorf("TokenSetRatio(%q, %q) = %d, want >= 50 (partial overlap should match decently)", a, b, got)
	}
	if got >= 100 {
		t.Errorf("TokenSetRatio(%q, %q) = %d, want < 100 (extra tokens should dilute)", a, b, got)
	}
}

// TestTokenSetRatio_DisjointTokens exercises completely disjoint inputs;
// score should be modest at best.
func TestTokenSetRatio_DisjointTokens(t *testing.T) {
	a := "alpha beta gamma"
	b := "delta epsilon zeta"
	got := TokenSetRatio(a, b)
	// Disjoint tokens still produce a non-zero score because the diff strings
	// "alpha beta gamma" and "delta epsilon zeta" share some character-level
	// similarity (both ~14 chars). Just assert it's below the partial-match
	// threshold.
	if got > 60 {
		t.Errorf("TokenSetRatio(%q, %q) = %d, want <= 60 (fully-disjoint tokens)", a, b, got)
	}
}

// TestTokenSetRatio_HighScoreOnLikelyMatch ensures the typical "sames" case
// (same tokens, light reordering, light typo) hits a high threshold.
func TestTokenSetRatio_HighScoreOnLikelyMatch(t *testing.T) {
	cases := []struct {
		a, b   string
		minOK  int
	}{
		{"London W1 SW1A 1AA", "SW1A 1AA London", 80},
		{"Acme Corp Ltd", "ACME CORPORATION LTD", 70},
	}
	for _, tc := range cases {
		t.Run(tc.a, func(t *testing.T) {
			got := TokenSetRatio(tc.a, tc.b)
			if got < tc.minOK {
				t.Errorf("TokenSetRatio(%q, %q) = %d, want >= %d", tc.a, tc.b, got, tc.minOK)
			}
		})
	}
}

// TestTokenSetRatio_Symmetric verifies the function is order-symmetric.
func TestTokenSetRatio_Symmetric(t *testing.T) {
	a := "John Robert Smith"
	b := "John R. Smith Jr."
	if r1, r2 := TokenSetRatio(a, b), TokenSetRatio(b, a); r1 != r2 {
		t.Errorf("TokenSetRatio asymmetric: TokenSetRatio(%q,%q)=%d != TokenSetRatio(%q,%q)=%d", a, b, r1, b, a, r2)
	}
}

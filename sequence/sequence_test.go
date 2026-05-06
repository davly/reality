package sequence

import (
	"math"
	"testing"

	"github.com/davly/reality/testutil"
)

// ═══════════════════════════════════════════════════════════════════════════
// Helper
// ═══════════════════════════════════════════════════════════════════════════

func assertFloat(t *testing.T, label string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %v, want %v (tol %v)", label, got, want, tol)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// LevenshteinDistance
// ═══════════════════════════════════════════════════════════════════════════

func TestLevenshteinDistance_KittenSitting(t *testing.T) {
	if got := LevenshteinDistance("kitten", "sitting"); got != 3 {
		t.Errorf("got %d, want 3", got)
	}
}

func TestLevenshteinDistance_EmptyStrings(t *testing.T) {
	if got := LevenshteinDistance("", ""); got != 0 {
		t.Errorf("got %d, want 0", got)
	}
}

func TestLevenshteinDistance_EmptyToNonEmpty(t *testing.T) {
	if got := LevenshteinDistance("", "abc"); got != 3 {
		t.Errorf("got %d, want 3", got)
	}
}

func TestLevenshteinDistance_Identical(t *testing.T) {
	if got := LevenshteinDistance("algorithm", "algorithm"); got != 0 {
		t.Errorf("got %d, want 0", got)
	}
}

func TestLevenshteinDistance_SaturdaySunday(t *testing.T) {
	if got := LevenshteinDistance("saturday", "sunday"); got != 3 {
		t.Errorf("got %d, want 3", got)
	}
}

func TestLevenshteinDistance_Symmetric(t *testing.T) {
	d1 := LevenshteinDistance("flaw", "lawn")
	d2 := LevenshteinDistance("lawn", "flaw")
	if d1 != d2 {
		t.Errorf("asymmetric: (%d, %d)", d1, d2)
	}
}

func TestLevenshteinDistance_Golden(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/sequence/levenshtein.json")

	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			a := tc.Inputs["a"].(string)
			b := tc.Inputs["b"].(string)
			expected := int(tc.Expected.(float64))
			got := LevenshteinDistance(a, b)
			if got != expected {
				t.Errorf("got %d, want %d", got, expected)
			}
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// DamerauLevenshtein
// ═══════════════════════════════════════════════════════════════════════════

func TestDamerauLevenshtein_Transposition(t *testing.T) {
	// "ab" → "ba" is one transposition.
	if got := DamerauLevenshtein("ab", "ba"); got != 1 {
		t.Errorf("got %d, want 1", got)
	}
}

func TestDamerauLevenshtein_NoTransposition(t *testing.T) {
	// Without transposition, same as Levenshtein for non-adjacent swaps.
	if got := DamerauLevenshtein("kitten", "sitting"); got != 3 {
		t.Errorf("got %d, want 3", got)
	}
}

func TestDamerauLevenshtein_EmptyStrings(t *testing.T) {
	if got := DamerauLevenshtein("", ""); got != 0 {
		t.Errorf("got %d, want 0", got)
	}
}

func TestDamerauLevenshtein_SingleChar(t *testing.T) {
	if got := DamerauLevenshtein("a", "b"); got != 1 {
		t.Errorf("got %d, want 1", got)
	}
}

func TestDamerauLevenshtein_IdenticalStrings(t *testing.T) {
	if got := DamerauLevenshtein("test", "test"); got != 0 {
		t.Errorf("got %d, want 0", got)
	}
}

func TestDamerauLevenshtein_TranspositionAndEdit(t *testing.T) {
	// "ca" → "abc": transpose "ca"→"ac" (1), insert "b" before "c" ... actually
	// let's use a clearer example: "abcd" → "abdc" is one transposition.
	if got := DamerauLevenshtein("abcd", "abdc"); got != 1 {
		t.Errorf("got %d, want 1", got)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// HammingDistance
// ═══════════════════════════════════════════════════════════════════════════

func TestHammingDistance_KnownPair(t *testing.T) {
	d, err := HammingDistance("karolin", "kathrin")
	if err != nil {
		t.Fatal(err)
	}
	if d != 3 {
		t.Errorf("got %d, want 3", d)
	}
}

func TestHammingDistance_Identical(t *testing.T) {
	d, err := HammingDistance("hello", "hello")
	if err != nil {
		t.Fatal(err)
	}
	if d != 0 {
		t.Errorf("got %d, want 0", d)
	}
}

func TestHammingDistance_DifferentLengths(t *testing.T) {
	_, err := HammingDistance("abc", "ab")
	if err == nil {
		t.Error("expected error for different lengths")
	}
}

func TestHammingDistance_Empty(t *testing.T) {
	d, err := HammingDistance("", "")
	if err != nil {
		t.Fatal(err)
	}
	if d != 0 {
		t.Errorf("got %d, want 0", d)
	}
}

func TestHammingDistance_AllDifferent(t *testing.T) {
	d, err := HammingDistance("abc", "xyz")
	if err != nil {
		t.Fatal(err)
	}
	if d != 3 {
		t.Errorf("got %d, want 3", d)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// JaroWinkler
// ═══════════════════════════════════════════════════════════════════════════

func TestJaroWinkler_Identical(t *testing.T) {
	assertFloat(t, "identical", JaroWinkler("hello", "hello"), 1.0, 1e-10)
}

func TestJaroWinkler_CompletelyDifferent(t *testing.T) {
	score := JaroWinkler("abc", "xyz")
	if score > 0.01 {
		t.Errorf("expected near 0, got %v", score)
	}
}

func TestJaroWinkler_BothEmpty(t *testing.T) {
	assertFloat(t, "both empty", JaroWinkler("", ""), 1.0, 1e-10)
}

func TestJaroWinkler_OneEmpty(t *testing.T) {
	assertFloat(t, "one empty", JaroWinkler("hello", ""), 0.0, 1e-10)
}

func TestJaroWinkler_Symmetry(t *testing.T) {
	a := JaroWinkler("martha", "marhta")
	b := JaroWinkler("marhta", "martha")
	assertFloat(t, "symmetry", a, b, 1e-10)
}

func TestJaroWinkler_KnownPair_MarthaMartha(t *testing.T) {
	// MARTHA vs MARHTA: Jaro ~0.944, Winkler boost with 3-char prefix ~0.961
	score := JaroWinkler("martha", "marhta")
	if score < 0.95 || score > 0.97 {
		t.Errorf("expected ~0.961, got %v", score)
	}
}

func TestJaroWinkler_KnownPair_DwayneAndDuane(t *testing.T) {
	score := JaroWinkler("dwayne", "duane")
	if score < 0.80 || score > 0.85 {
		t.Errorf("expected ~0.84, got %v", score)
	}
}

func TestJaroWinkler_RangeCheck(t *testing.T) {
	score := JaroWinkler("test", "testing")
	if score < 0.0 || score > 1.0 {
		t.Errorf("score %v out of [0,1] range", score)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// LongestCommonSubsequence
// ═══════════════════════════════════════════════════════════════════════════

func TestLCS_Known(t *testing.T) {
	lcs := LongestCommonSubsequence("ABCBDAB", "BDCAB")
	if len(lcs) != 4 {
		t.Errorf("expected length 4, got %d (%q)", len(lcs), lcs)
	}
}

func TestLCS_Empty(t *testing.T) {
	if lcs := LongestCommonSubsequence("", "abc"); lcs != "" {
		t.Errorf("expected empty, got %q", lcs)
	}
}

func TestLCS_Identical(t *testing.T) {
	if lcs := LongestCommonSubsequence("abc", "abc"); lcs != "abc" {
		t.Errorf("expected abc, got %q", lcs)
	}
}

func TestLCS_NoCommon(t *testing.T) {
	if lcs := LongestCommonSubsequence("abc", "xyz"); lcs != "" {
		t.Errorf("expected empty, got %q", lcs)
	}
}

func TestLCS_SingleChar(t *testing.T) {
	lcs := LongestCommonSubsequence("a", "a")
	if lcs != "a" {
		t.Errorf("expected 'a', got %q", lcs)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// LongestCommonSubstring
// ═══════════════════════════════════════════════════════════════════════════

func TestLongestCommonSubstring_Known(t *testing.T) {
	s := LongestCommonSubstring("abcdef", "zbcdf")
	if s != "bcd" {
		t.Errorf("expected bcd, got %q", s)
	}
}

func TestLongestCommonSubstring_Empty(t *testing.T) {
	if s := LongestCommonSubstring("", "abc"); s != "" {
		t.Errorf("expected empty, got %q", s)
	}
}

func TestLongestCommonSubstring_Identical(t *testing.T) {
	if s := LongestCommonSubstring("hello", "hello"); s != "hello" {
		t.Errorf("expected hello, got %q", s)
	}
}

func TestLongestCommonSubstring_NoCommon(t *testing.T) {
	if s := LongestCommonSubstring("abc", "xyz"); s != "" {
		t.Errorf("expected empty, got %q", s)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// NeedlemanWunsch
// ═══════════════════════════════════════════════════════════════════════════

func TestNeedlemanWunsch_IdenticalStrings(t *testing.T) {
	a1, b1, score := NeedlemanWunsch("ACGT", "ACGT", 1, -1, -1)
	if a1 != "ACGT" || b1 != "ACGT" {
		t.Errorf("expected identical alignment, got %q / %q", a1, b1)
	}
	assertFloat(t, "score", score, 4.0, 1e-10)
}

func TestNeedlemanWunsch_SimpleGap(t *testing.T) {
	a1, b1, score := NeedlemanWunsch("ACGT", "AGT", 1, -1, -1)
	// One deletion needed: A-GT aligned with ACGT
	if len(a1) != len(b1) {
		t.Errorf("alignment lengths differ: %q (%d) vs %q (%d)", a1, len(a1), b1, len(b1))
	}
	if score < 1.0 {
		t.Errorf("expected positive score, got %v", score)
	}
}

func TestNeedlemanWunsch_BothEmpty(t *testing.T) {
	a1, b1, score := NeedlemanWunsch("", "", 1, -1, -1)
	if a1 != "" || b1 != "" {
		t.Errorf("expected empty, got %q / %q", a1, b1)
	}
	assertFloat(t, "score", score, 0.0, 1e-10)
}

func TestNeedlemanWunsch_OneEmpty(t *testing.T) {
	a1, b1, score := NeedlemanWunsch("ABC", "", 1, -1, -1)
	if len(a1) != len(b1) {
		t.Errorf("alignment lengths differ: %q vs %q", a1, b1)
	}
	// Score should be 3 gaps = -3.
	assertFloat(t, "score", score, -3.0, 1e-10)
}

func TestNeedlemanWunsch_GapPenalty(t *testing.T) {
	_, _, score := NeedlemanWunsch("A", "B", 1, -1, -2)
	// Mismatch (-1) vs gap+gap (-4). Should choose mismatch.
	assertFloat(t, "score", score, -1.0, 1e-10)
}

// ═══════════════════════════════════════════════════════════════════════════
// SmithWaterman
// ═══════════════════════════════════════════════════════════════════════════

func TestSmithWaterman_LocalMatch(t *testing.T) {
	a1, b1, score := SmithWaterman("XXXACGTXXX", "ACGT", 2, -1, -1)
	if a1 != "ACGT" || b1 != "ACGT" {
		t.Errorf("expected ACGT/ACGT, got %q/%q", a1, b1)
	}
	assertFloat(t, "score", score, 8.0, 1e-10) // 4 matches * 2
}

func TestSmithWaterman_NoMatch(t *testing.T) {
	_, _, score := SmithWaterman("AAAA", "BBBB", 2, -3, -3)
	assertFloat(t, "score", score, 0.0, 1e-10)
}

func TestSmithWaterman_BothEmpty(t *testing.T) {
	a1, b1, score := SmithWaterman("", "", 1, -1, -1)
	if a1 != "" || b1 != "" {
		t.Errorf("expected empty, got %q / %q", a1, b1)
	}
	assertFloat(t, "score", score, 0.0, 1e-10)
}

func TestSmithWaterman_PartialOverlap(t *testing.T) {
	_, _, score := SmithWaterman("AABBB", "BBBAA", 2, -1, -1)
	// "BBB" should be the best local alignment: 3 matches * 2 = 6.
	assertFloat(t, "score", score, 6.0, 1e-10)
}

// ═══════════════════════════════════════════════════════════════════════════
// NGrams
// ═══════════════════════════════════════════════════════════════════════════

func TestNGrams_Trigrams(t *testing.T) {
	got := NGrams("hello", 3)
	want := []string{"hel", "ell", "llo"}
	if len(got) != len(want) {
		t.Fatalf("length: got %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("[%d] got %q, want %q", i, got[i], want[i])
		}
	}
}

func TestNGrams_TooShort(t *testing.T) {
	if got := NGrams("hi", 3); got != nil {
		t.Errorf("expected nil, got %v", got)
	}
}

func TestNGrams_Zero(t *testing.T) {
	if got := NGrams("hello", 0); got != nil {
		t.Errorf("expected nil for n=0, got %v", got)
	}
}

func TestNGrams_ExactLength(t *testing.T) {
	got := NGrams("abc", 3)
	if len(got) != 1 || got[0] != "abc" {
		t.Errorf("expected [abc], got %v", got)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// WordNGrams
// ═══════════════════════════════════════════════════════════════════════════

func TestWordNGrams_Bigrams(t *testing.T) {
	words := []string{"the", "cat", "sat", "on"}
	got := WordNGrams(words, 2)
	if len(got) != 3 {
		t.Fatalf("expected 3 bigrams, got %d", len(got))
	}
	if got[0][0] != "the" || got[0][1] != "cat" {
		t.Errorf("first bigram: got %v", got[0])
	}
}

func TestWordNGrams_TooFew(t *testing.T) {
	if got := WordNGrams([]string{"only"}, 2); got != nil {
		t.Errorf("expected nil, got %v", got)
	}
}

func TestWordNGrams_Empty(t *testing.T) {
	if got := WordNGrams(nil, 1); got != nil {
		t.Errorf("expected nil for nil input, got %v", got)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// NGramSimilarity
// ═══════════════════════════════════════════════════════════════════════════

func TestNGramSimilarity_Identical(t *testing.T) {
	assertFloat(t, "identical", NGramSimilarity("hello", "hello", 2), 1.0, 1e-10)
}

func TestNGramSimilarity_NoOverlap(t *testing.T) {
	assertFloat(t, "no overlap", NGramSimilarity("abc", "xyz", 2), 0.0, 1e-10)
}

func TestNGramSimilarity_BothEmpty(t *testing.T) {
	assertFloat(t, "both empty", NGramSimilarity("", "", 2), 1.0, 1e-10)
}

func TestNGramSimilarity_Partial(t *testing.T) {
	sim := NGramSimilarity("night", "nacht", 2)
	if sim < 0.0 || sim > 1.0 {
		t.Errorf("out of range: %v", sim)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// NGramDiceCoefficient
// ═══════════════════════════════════════════════════════════════════════════

func TestNGramDiceCoefficient_Identical(t *testing.T) {
	assertFloat(t, "identical", NGramDiceCoefficient("hello", "hello", 2), 1.0, 1e-10)
}

func TestNGramDiceCoefficient_NoOverlap(t *testing.T) {
	assertFloat(t, "no overlap", NGramDiceCoefficient("abc", "xyz", 2), 0.0, 1e-10)
}

func TestNGramDiceCoefficient_BothEmpty(t *testing.T) {
	assertFloat(t, "both empty", NGramDiceCoefficient("", "", 2), 1.0, 1e-10)
}

func TestNGramDiceCoefficient_OneEmpty(t *testing.T) {
	assertFloat(t, "one empty", NGramDiceCoefficient("hello", "", 2), 0.0, 1e-10)
}

func TestNGramDiceCoefficient_BoundedInZeroOne(t *testing.T) {
	d := NGramDiceCoefficient("night", "nacht", 2)
	if d < 0.0 || d > 1.0 {
		t.Errorf("out of range: %v", d)
	}
}

// TestNGramDice_StrongerThanJaccard exercises the well-known relationship
// dice = 2J / (1+J) — Sørensen-Dice scores plausible matches higher than
// Jaccard for any same input pair. Tested at the character n-gram level
// against the same inputs used in NGramSimilarity_Partial.
func TestNGramDice_StrongerThanJaccard(t *testing.T) {
	const n = 2
	cases := []struct{ a, b string }{
		{"night", "nacht"},
		{"hello", "hallo"},
		{"smith", "smyth"},
	}
	for _, tc := range cases {
		t.Run(tc.a+"_vs_"+tc.b, func(t *testing.T) {
			j := NGramSimilarity(tc.a, tc.b, n)
			d := NGramDiceCoefficient(tc.a, tc.b, n)
			expected := 2 * j / (1 + j)
			assertFloat(t, "dice = 2J/(1+J) identity", d, expected, 1e-10)
			if !(j == 0 && d == 0) && d <= j {
				t.Errorf("dice should exceed jaccard for non-zero pair: jaccard=%v dice=%v", j, d)
			}
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Shingling
// ═══════════════════════════════════════════════════════════════════════════

func TestShingling_Basic(t *testing.T) {
	hashes := Shingling("abcdef", 3)
	// "abc", "bcd", "cde", "def" → 4 shingles
	if len(hashes) != 4 {
		t.Errorf("expected 4 shingles, got %d", len(hashes))
	}
}

func TestShingling_TooShort(t *testing.T) {
	if got := Shingling("ab", 3); got != nil {
		t.Errorf("expected nil, got %v", got)
	}
}

func TestShingling_Deterministic(t *testing.T) {
	h1 := Shingling("hello world", 4)
	h2 := Shingling("hello world", 4)
	if len(h1) != len(h2) {
		t.Fatal("length mismatch")
	}
	for i := range h1 {
		if h1[i] != h2[i] {
			t.Errorf("[%d] not deterministic: %d vs %d", i, h1[i], h2[i])
		}
	}
}

func TestShingling_UniqueHashes(t *testing.T) {
	hashes := Shingling("abcdef", 3)
	seen := make(map[uint64]bool)
	for _, h := range hashes {
		if seen[h] {
			t.Errorf("duplicate hash %d", h)
		}
		seen[h] = true
	}
}

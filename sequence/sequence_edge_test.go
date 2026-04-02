package sequence

import (
	"math"
	"strings"
	"testing"
)

// ═══════════════════════════════════════════════════════════════════════════
// LevenshteinDistance — edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestLevenshteinDistance_Unicode(t *testing.T) {
	// Multi-byte characters should be handled correctly
	if got := LevenshteinDistance("café", "cafe"); got != 1 {
		t.Errorf("got %d, want 1 (é→e)", got)
	}
}

func TestLevenshteinDistance_UnicodeIdentical(t *testing.T) {
	if got := LevenshteinDistance("日本語", "日本語"); got != 0 {
		t.Errorf("got %d, want 0 for identical unicode strings", got)
	}
}

func TestLevenshteinDistance_UnicodeCompleteDiff(t *testing.T) {
	if got := LevenshteinDistance("日本", "中国"); got != 2 {
		t.Errorf("got %d, want 2 for completely different 2-char unicode strings", got)
	}
}

func TestLevenshteinDistance_SingleCharDiff(t *testing.T) {
	if got := LevenshteinDistance("a", "b"); got != 1 {
		t.Errorf("got %d, want 1", got)
	}
}

func TestLevenshteinDistance_NonEmptyToEmpty(t *testing.T) {
	if got := LevenshteinDistance("hello", ""); got != 5 {
		t.Errorf("got %d, want 5", got)
	}
}

func TestLevenshteinDistance_LongString(t *testing.T) {
	a := strings.Repeat("a", 100)
	b := strings.Repeat("b", 100)
	if got := LevenshteinDistance(a, b); got != 100 {
		t.Errorf("got %d, want 100", got)
	}
}

func TestLevenshteinDistance_TriangleInequality(t *testing.T) {
	// d(a,c) <= d(a,b) + d(b,c)
	a, b, c := "kitten", "sitting", "mitten"
	dAC := LevenshteinDistance(a, c)
	dAB := LevenshteinDistance(a, b)
	dBC := LevenshteinDistance(b, c)
	if dAC > dAB+dBC {
		t.Errorf("triangle inequality violated: d(a,c)=%d > d(a,b)+d(b,c)=%d+%d", dAC, dAB, dBC)
	}
}

func TestLevenshteinDistance_PrefixOnly(t *testing.T) {
	if got := LevenshteinDistance("test", "testing"); got != 3 {
		t.Errorf("got %d, want 3 (append 'ing')", got)
	}
}

func TestLevenshteinDistance_SuffixOnly(t *testing.T) {
	if got := LevenshteinDistance("testing", "test"); got != 3 {
		t.Errorf("got %d, want 3 (remove 'ing')", got)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// DamerauLevenshtein — edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestDamerauLevenshtein_MultipleTranspositions(t *testing.T) {
	// "abcde" → "bacde" has 1 transposition at the start
	if got := DamerauLevenshtein("abcde", "bacde"); got != 1 {
		t.Errorf("got %d, want 1 (ab→ba)", got)
	}
}

func TestDamerauLevenshtein_EmptyToNonEmpty(t *testing.T) {
	if got := DamerauLevenshtein("", "abc"); got != 3 {
		t.Errorf("got %d, want 3", got)
	}
}

func TestDamerauLevenshtein_NonEmptyToEmpty(t *testing.T) {
	if got := DamerauLevenshtein("abc", ""); got != 3 {
		t.Errorf("got %d, want 3", got)
	}
}

func TestDamerauLevenshtein_Symmetric(t *testing.T) {
	d1 := DamerauLevenshtein("abcde", "edcba")
	d2 := DamerauLevenshtein("edcba", "abcde")
	if d1 != d2 {
		t.Errorf("asymmetric: (%d, %d)", d1, d2)
	}
}

func TestDamerauLevenshtein_TranspositionBetterThanTwoSubs(t *testing.T) {
	// "ab" → "ba" should be 1 (transposition), not 2 (two substitutions)
	if got := DamerauLevenshtein("ab", "ba"); got != 1 {
		t.Errorf("got %d, want 1 (transposition should be cheaper)", got)
	}
}

func TestDamerauLevenshtein_Unicode(t *testing.T) {
	if got := DamerauLevenshtein("日本", "本日"); got != 1 {
		t.Errorf("got %d, want 1 for unicode transposition", got)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// HammingDistance — edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestHammingDistance_SingleDiff(t *testing.T) {
	d, err := HammingDistance("abc", "axc")
	if err != nil {
		t.Fatal(err)
	}
	if d != 1 {
		t.Errorf("got %d, want 1", d)
	}
}

func TestHammingDistance_Unicode(t *testing.T) {
	d, err := HammingDistance("日本語", "日中語")
	if err != nil {
		t.Fatal(err)
	}
	if d != 1 {
		t.Errorf("got %d, want 1 for single unicode diff", d)
	}
}

func TestHammingDistance_UnicodeUnequalRuneCount(t *testing.T) {
	// "ab" (2 runes) vs "abc" (3 runes) — should error
	_, err := HammingDistance("ab", "abc")
	if err == nil {
		t.Error("expected error for different rune counts")
	}
}

func TestHammingDistance_SingleCharIdentical(t *testing.T) {
	d, err := HammingDistance("x", "x")
	if err != nil {
		t.Fatal(err)
	}
	if d != 0 {
		t.Errorf("got %d, want 0", d)
	}
}

func TestHammingDistance_SingleCharDiff(t *testing.T) {
	d, err := HammingDistance("x", "y")
	if err != nil {
		t.Fatal(err)
	}
	if d != 1 {
		t.Errorf("got %d, want 1", d)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// JaroWinkler — edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestJaroWinkler_SingleCharSame(t *testing.T) {
	assertFloat(t, "single char same", JaroWinkler("a", "a"), 1.0, 1e-10)
}

func TestJaroWinkler_SingleCharDiff(t *testing.T) {
	score := JaroWinkler("a", "b")
	if score != 0.0 {
		t.Errorf("expected 0.0 for single different chars, got %v", score)
	}
}

func TestJaroWinkler_PrefixBoosts(t *testing.T) {
	// Strings sharing a longer common prefix should get a larger Winkler boost.
	// "abcXYZ" vs "abcXZY" shares 4-char prefix "abcX"
	// "XYZabc" vs "XZYabc" shares 1-char prefix "X"
	longPrefix := JaroWinkler("abcXYZ", "abcXZY")
	shortPrefix := JaroWinkler("Xbcdef", "Xcdebf")
	// longPrefix should have a higher Winkler boost
	if longPrefix < shortPrefix {
		t.Errorf("longer prefix should give higher similarity: longPrefix=%v shortPrefix=%v", longPrefix, shortPrefix)
	}
}

func TestJaroWinkler_RangeAlways01(t *testing.T) {
	cases := [][2]string{
		{"a", "b"},
		{"hello", "world"},
		{"abc", "def"},
		{"test", "testing"},
		{"", ""},
	}
	for _, c := range cases {
		score := JaroWinkler(c[0], c[1])
		if score < 0.0 || score > 1.0 {
			t.Errorf("JaroWinkler(%q, %q) = %v out of [0,1]", c[0], c[1], score)
		}
	}
}

func TestJaroWinkler_Unicode(t *testing.T) {
	score := JaroWinkler("münchen", "münchen")
	assertFloat(t, "unicode identical", score, 1.0, 1e-10)
}

// ═══════════════════════════════════════════════════════════════════════════
// LongestCommonSubsequence — edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestLCS_BothEmpty(t *testing.T) {
	if lcs := LongestCommonSubsequence("", ""); lcs != "" {
		t.Errorf("expected empty, got %q", lcs)
	}
}

func TestLCS_Prefix(t *testing.T) {
	lcs := LongestCommonSubsequence("abcdef", "abcxyz")
	if lcs != "abc" {
		t.Errorf("expected abc, got %q", lcs)
	}
}

func TestLCS_Suffix(t *testing.T) {
	lcs := LongestCommonSubsequence("xyzabc", "defahc")
	// Common subsequence should at least contain 'a' and 'c'
	if len(lcs) < 2 {
		t.Errorf("expected at least 2-char LCS, got %q", lcs)
	}
}

func TestLCS_Interleaved(t *testing.T) {
	// "axbycz" and "abc" — LCS is "abc"
	lcs := LongestCommonSubsequence("axbycz", "abc")
	if lcs != "abc" {
		t.Errorf("expected abc, got %q", lcs)
	}
}

func TestLCS_Unicode(t *testing.T) {
	lcs := LongestCommonSubsequence("日本語テスト", "日テスト")
	if len(lcs) < 3 {
		t.Errorf("expected at least 3-rune LCS for unicode, got %q (len=%d)", lcs, len([]rune(lcs)))
	}
}

func TestLCS_Symmetric(t *testing.T) {
	lcs1 := LongestCommonSubsequence("ABCBDAB", "BDCAB")
	lcs2 := LongestCommonSubsequence("BDCAB", "ABCBDAB")
	if len(lcs1) != len(lcs2) {
		t.Errorf("LCS length should be symmetric: %d vs %d", len(lcs1), len(lcs2))
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// LongestCommonSubstring — edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestLongestCommonSubstring_BothEmpty(t *testing.T) {
	if s := LongestCommonSubstring("", ""); s != "" {
		t.Errorf("expected empty, got %q", s)
	}
}

func TestLongestCommonSubstring_SingleChar(t *testing.T) {
	if s := LongestCommonSubstring("a", "a"); s != "a" {
		t.Errorf("expected 'a', got %q", s)
	}
}

func TestLongestCommonSubstring_Overlap(t *testing.T) {
	s := LongestCommonSubstring("testing", "testing123")
	if s != "testing" {
		t.Errorf("expected 'testing', got %q", s)
	}
}

func TestLongestCommonSubstring_Unicode(t *testing.T) {
	s := LongestCommonSubstring("日本語ABC", "XY日本語Z")
	if s != "日本語" {
		t.Errorf("expected '日本語', got %q", s)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// NeedlemanWunsch — edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestNeedlemanWunsch_TotalMismatch(t *testing.T) {
	a1, b1, score := NeedlemanWunsch("AAAA", "BBBB", 1, -1, -1)
	if len(a1) != len(b1) {
		t.Errorf("alignment lengths differ: %q vs %q", a1, b1)
	}
	// All mismatches: 4*(-1) = -4
	assertFloat(t, "all mismatches", score, -4.0, 1e-10)
}

func TestNeedlemanWunsch_LengthMismatch(t *testing.T) {
	a1, b1, _ := NeedlemanWunsch("ABCDE", "AB", 1, -1, -1)
	if len(a1) != len(b1) {
		t.Errorf("aligned strings must have equal length: %q (%d) vs %q (%d)",
			a1, len(a1), b1, len(b1))
	}
}

func TestNeedlemanWunsch_SingleCharMatch(t *testing.T) {
	a1, b1, score := NeedlemanWunsch("A", "A", 2, -1, -1)
	if a1 != "A" || b1 != "A" {
		t.Errorf("expected A/A, got %q/%q", a1, b1)
	}
	assertFloat(t, "single match score", score, 2.0, 1e-10)
}

func TestNeedlemanWunsch_SingleCharMismatch(t *testing.T) {
	_, _, score := NeedlemanWunsch("A", "B", 2, -1, -1)
	// Either mismatch (-1) or two gaps (-2), should pick mismatch
	assertFloat(t, "single mismatch score", score, -1.0, 1e-10)
}

// ═══════════════════════════════════════════════════════════════════════════
// SmithWaterman — edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestSmithWaterman_OneEmpty(t *testing.T) {
	_, _, score := SmithWaterman("ACGT", "", 2, -1, -1)
	assertFloat(t, "one empty score", score, 0.0, 1e-10)
}

func TestSmithWaterman_SingleCharMatch(t *testing.T) {
	a1, b1, score := SmithWaterman("A", "A", 2, -1, -1)
	if a1 != "A" || b1 != "A" {
		t.Errorf("expected A/A, got %q/%q", a1, b1)
	}
	assertFloat(t, "single match", score, 2.0, 1e-10)
}

func TestSmithWaterman_ScoreNonNegative(t *testing.T) {
	_, _, score := SmithWaterman("AAAA", "BBBB", 2, -1, -1)
	if score < 0 {
		t.Errorf("Smith-Waterman score should never be negative, got %v", score)
	}
}

func TestSmithWaterman_LocalBetterThanGlobal(t *testing.T) {
	// Smith-Waterman should find the best local region
	_, _, score := SmithWaterman("XXXACGTACGTXXX", "ACGTACGT", 2, -1, -1)
	// 8 matches * 2 = 16
	assertFloat(t, "local match", score, 16.0, 1e-10)
}

// ═══════════════════════════════════════════════════════════════════════════
// NGrams — edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestNGrams_NegativeN(t *testing.T) {
	if got := NGrams("hello", -1); got != nil {
		t.Errorf("expected nil for n=-1, got %v", got)
	}
}

func TestNGrams_Unigrams(t *testing.T) {
	got := NGrams("abc", 1)
	want := []string{"a", "b", "c"}
	if len(got) != len(want) {
		t.Fatalf("length: got %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("[%d] got %q, want %q", i, got[i], want[i])
		}
	}
}

func TestNGrams_Unicode(t *testing.T) {
	got := NGrams("日本語", 2)
	if len(got) != 2 {
		t.Fatalf("expected 2 bigrams, got %d", len(got))
	}
	if got[0] != "日本" || got[1] != "本語" {
		t.Errorf("unicode bigrams: got %v", got)
	}
}

func TestNGrams_EmptyString(t *testing.T) {
	if got := NGrams("", 1); got != nil {
		t.Errorf("expected nil for empty string, got %v", got)
	}
}

func TestNGrams_FullLength(t *testing.T) {
	got := NGrams("abcde", 5)
	if len(got) != 1 || got[0] != "abcde" {
		t.Errorf("expected [abcde], got %v", got)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// WordNGrams — edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestWordNGrams_Unigrams(t *testing.T) {
	words := []string{"a", "b", "c"}
	got := WordNGrams(words, 1)
	if len(got) != 3 {
		t.Fatalf("expected 3 unigrams, got %d", len(got))
	}
	if got[0][0] != "a" || got[1][0] != "b" || got[2][0] != "c" {
		t.Errorf("unigrams: got %v", got)
	}
}

func TestWordNGrams_NegativeN(t *testing.T) {
	if got := WordNGrams([]string{"a"}, -1); got != nil {
		t.Errorf("expected nil for n=-1, got %v", got)
	}
}

func TestWordNGrams_ZeroN(t *testing.T) {
	if got := WordNGrams([]string{"a"}, 0); got != nil {
		t.Errorf("expected nil for n=0, got %v", got)
	}
}

func TestWordNGrams_ExactLength(t *testing.T) {
	words := []string{"hello", "world"}
	got := WordNGrams(words, 2)
	if len(got) != 1 {
		t.Fatalf("expected 1 bigram, got %d", len(got))
	}
	if got[0][0] != "hello" || got[0][1] != "world" {
		t.Errorf("bigram: got %v", got[0])
	}
}

func TestWordNGrams_Trigrams(t *testing.T) {
	words := []string{"the", "cat", "sat", "on", "mat"}
	got := WordNGrams(words, 3)
	if len(got) != 3 {
		t.Fatalf("expected 3 trigrams, got %d", len(got))
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// NGramSimilarity — edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestNGramSimilarity_OneEmpty(t *testing.T) {
	assertFloat(t, "one empty", NGramSimilarity("abc", "", 2), 0.0, 1e-10)
}

func TestNGramSimilarity_SmallN(t *testing.T) {
	// With n=1 (character unigrams), "abc" vs "abd" share {a,b} out of {a,b,c,d}
	sim := NGramSimilarity("abc", "abd", 1)
	assertFloat(t, "n=1 partial overlap", sim, 0.5, 1e-10) // 2/4
}

func TestNGramSimilarity_ReturnsOneForSubstring(t *testing.T) {
	// If both strings produce the same n-gram set, similarity = 1
	sim := NGramSimilarity("abcabc", "abcabc", 2)
	assertFloat(t, "same ngram set", sim, 1.0, 1e-10)
}

func TestNGramSimilarity_NTooLargeForBoth(t *testing.T) {
	// Both strings too short for n-grams — both empty → identical
	sim := NGramSimilarity("ab", "cd", 5)
	assertFloat(t, "n too large for both", sim, 1.0, 1e-10)
}

func TestNGramSimilarity_Range01(t *testing.T) {
	sim := NGramSimilarity("hello", "help", 2)
	if sim < 0.0 || sim > 1.0 {
		t.Errorf("similarity %v out of [0,1] range", sim)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Shingling — edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestShingling_NegativeK(t *testing.T) {
	if got := Shingling("hello", -1); got != nil {
		t.Errorf("expected nil for k=-1, got %v", got)
	}
}

func TestShingling_ZeroK(t *testing.T) {
	if got := Shingling("hello", 0); got != nil {
		t.Errorf("expected nil for k=0, got %v", got)
	}
}

func TestShingling_ExactLength(t *testing.T) {
	hashes := Shingling("abc", 3)
	if len(hashes) != 1 {
		t.Errorf("expected 1 shingle for exact-length string, got %d", len(hashes))
	}
}

func TestShingling_DifferentStringsProduceDifferentHashes(t *testing.T) {
	h1 := Shingling("hello", 3)
	h2 := Shingling("world", 3)
	// At least one hash should differ
	allSame := true
	if len(h1) != len(h2) {
		allSame = false
	} else {
		for i := range h1 {
			if h1[i] != h2[i] {
				allSame = false
				break
			}
		}
	}
	if allSame {
		t.Error("different strings should produce different hash sequences")
	}
}

func TestShingling_HashesAreNonZero(t *testing.T) {
	hashes := Shingling("abcdefgh", 3)
	for i, h := range hashes {
		if h == 0 {
			t.Errorf("hash[%d] is zero (extremely unlikely for FNV-1a)", i)
		}
	}
}

// Suppress unused import
var _ = math.Abs

// Package sequence provides string distance metrics, sequence alignment
// algorithms, and n-gram utilities. These are the foundational operations
// for measuring similarity, performing fuzzy matching, and aligning
// sequences — used across NLP, bioinformatics, and information retrieval.
//
// All functions are pure, deterministic, and have zero external dependencies.
package sequence

import "errors"

// ---------------------------------------------------------------------------
// Edit distances
// ---------------------------------------------------------------------------

// LevenshteinDistance computes the minimum number of single-character edits
// (insertions, deletions, substitutions) required to transform string a into
// string b using the Wagner-Fischer dynamic programming algorithm.
//
// Formula: dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
//   where cost = 0 if a[i]==b[j], else 1
// Time complexity: O(len(a) * len(b))
// Space complexity: O(min(len(a), len(b))) — uses two-row optimization
// Reference: Wagner, Fischer (1974), "The String-to-String Correction Problem"
func LevenshteinDistance(a, b string) int {
	ra, rb := []rune(a), []rune(b)
	m, n := len(ra), len(rb)

	// Ensure ra is the shorter string for space optimization.
	if m > n {
		ra, rb = rb, ra
		m, n = n, m
	}

	// Two-row DP.
	prev := make([]int, m+1)
	curr := make([]int, m+1)
	for i := 0; i <= m; i++ {
		prev[i] = i
	}

	for j := 1; j <= n; j++ {
		curr[0] = j
		for i := 1; i <= m; i++ {
			cost := 1
			if ra[i-1] == rb[j-1] {
				cost = 0
			}
			curr[i] = min3(
				prev[i]+1,     // deletion
				curr[i-1]+1,   // insertion
				prev[i-1]+cost, // substitution
			)
		}
		prev, curr = curr, prev
	}

	return prev[m]
}

// DamerauLevenshtein computes the Damerau-Levenshtein distance between
// strings a and b, allowing insertions, deletions, substitutions, and
// transpositions of adjacent characters. This is the optimal string
// alignment distance (restricted edit distance).
//
// Formula: extends Levenshtein with dp[i-2][j-2]+cost when a[i]==b[j-1] && a[i-1]==b[j]
// Time complexity: O(len(a) * len(b))
// Space complexity: O(len(a) * len(b))
// Reference: Damerau (1964), "A Technique for Computer Detection and
//   Correction of Spelling Errors"
func DamerauLevenshtein(a, b string) int {
	ra, rb := []rune(a), []rune(b)
	m, n := len(ra), len(rb)

	// Full matrix required for transposition lookback.
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
		dp[i][0] = i
	}
	for j := 0; j <= n; j++ {
		dp[0][j] = j
	}

	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			cost := 1
			if ra[i-1] == rb[j-1] {
				cost = 0
			}

			dp[i][j] = min3(
				dp[i-1][j]+1,       // deletion
				dp[i][j-1]+1,       // insertion
				dp[i-1][j-1]+cost,  // substitution
			)

			// Transposition.
			if i > 1 && j > 1 && ra[i-1] == rb[j-2] && ra[i-2] == rb[j-1] {
				dp[i][j] = min2(dp[i][j], dp[i-2][j-2]+cost)
			}
		}
	}

	return dp[m][n]
}

// HammingDistance computes the number of positions at which corresponding
// characters of equal-length strings a and b differ. Returns an error if
// the strings have different lengths.
//
// Formula: count of positions i where a[i] != b[i]
// Time complexity: O(len(a))
// Space complexity: O(1) — not counting the rune conversions
// Reference: Hamming (1950), "Error Detecting and Error Correcting Codes"
func HammingDistance(a, b string) (int, error) {
	ra, rb := []rune(a), []rune(b)
	if len(ra) != len(rb) {
		return 0, errors.New("sequence: HammingDistance requires equal-length strings")
	}

	dist := 0
	for i := range ra {
		if ra[i] != rb[i] {
			dist++
		}
	}
	return dist, nil
}

// ---------------------------------------------------------------------------
// Fuzzy similarity
// ---------------------------------------------------------------------------

// JaroWinkler computes the Jaro-Winkler similarity between strings a and b,
// returning a value in [0, 1] where 1.0 means identical strings.
//
// The Jaro similarity considers matching characters and transpositions.
// The Winkler modification boosts the score for strings sharing a common
// prefix (up to 4 characters), with a scaling factor of 0.1.
//
// Formula: jaro = (matches/|a| + matches/|b| + (matches-transpositions)/matches) / 3
//   winkler = jaro + prefix * 0.1 * (1 - jaro)
// Time complexity: O(len(a) * len(b))
// Reference: Winkler (1990), "String Comparator Metrics and Enhanced Decision
//   Rules in the Fellegi-Sunter Model of Record Linkage"
func JaroWinkler(a, b string) float64 {
	ra, rb := []rune(a), []rune(b)
	la, lb := len(ra), len(rb)

	if la == 0 && lb == 0 {
		return 1.0
	}
	if la == 0 || lb == 0 {
		return 0.0
	}

	// Match window: floor(max(|a|,|b|)/2) - 1, at least 0.
	maxLen := la
	if lb > maxLen {
		maxLen = lb
	}
	window := maxLen/2 - 1
	if window < 0 {
		window = 0
	}

	matchedA := make([]bool, la)
	matchedB := make([]bool, lb)

	matches := 0
	transpositions := 0

	// Count matches.
	for i := 0; i < la; i++ {
		lo := i - window
		if lo < 0 {
			lo = 0
		}
		hi := i + window + 1
		if hi > lb {
			hi = lb
		}
		for j := lo; j < hi; j++ {
			if matchedB[j] || ra[i] != rb[j] {
				continue
			}
			matchedA[i] = true
			matchedB[j] = true
			matches++
			break
		}
	}

	if matches == 0 {
		return 0.0
	}

	// Count transpositions.
	k := 0
	for i := 0; i < la; i++ {
		if !matchedA[i] {
			continue
		}
		for !matchedB[k] {
			k++
		}
		if ra[i] != rb[k] {
			transpositions++
		}
		k++
	}

	jaro := (float64(matches)/float64(la) +
		float64(matches)/float64(lb) +
		float64(matches-transpositions/2)/float64(matches)) / 3.0

	// Winkler prefix bonus (up to 4 characters, scaling factor 0.1).
	prefix := 0
	maxPrefix := 4
	if la < maxPrefix {
		maxPrefix = la
	}
	if lb < maxPrefix {
		maxPrefix = lb
	}
	for i := 0; i < maxPrefix; i++ {
		if ra[i] != rb[i] {
			break
		}
		prefix++
	}

	return jaro + float64(prefix)*0.1*(1.0-jaro)
}

// ---------------------------------------------------------------------------
// Subsequence / Substring
// ---------------------------------------------------------------------------

// LongestCommonSubsequence returns the longest common subsequence of strings
// a and b. A subsequence is a sequence that appears in the same relative
// order, but not necessarily contiguously.
//
// Formula: dp[i][j] = dp[i-1][j-1]+1 if a[i]==b[j], else max(dp[i-1][j], dp[i][j-1])
// Time complexity: O(len(a) * len(b))
// Space complexity: O(len(a) * len(b))
// Reference: standard LCS via dynamic programming
func LongestCommonSubsequence(a, b string) string {
	ra, rb := []rune(a), []rune(b)
	m, n := len(ra), len(rb)

	if m == 0 || n == 0 {
		return ""
	}

	// Build DP table.
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}

	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if ra[i-1] == rb[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max2(dp[i-1][j], dp[i][j-1])
			}
		}
	}

	// Backtrack to recover the subsequence.
	lcs := make([]rune, 0, dp[m][n])
	i, j := m, n
	for i > 0 && j > 0 {
		if ra[i-1] == rb[j-1] {
			lcs = append(lcs, ra[i-1])
			i--
			j--
		} else if dp[i-1][j] > dp[i][j-1] {
			i--
		} else {
			j--
		}
	}

	// Reverse.
	for left, right := 0, len(lcs)-1; left < right; left, right = left+1, right-1 {
		lcs[left], lcs[right] = lcs[right], lcs[left]
	}

	return string(lcs)
}

// LongestCommonSubstring returns the longest contiguous substring that
// appears in both a and b.
//
// Formula: dp[i][j] = dp[i-1][j-1]+1 if a[i]==b[j], else 0
// Time complexity: O(len(a) * len(b))
// Space complexity: O(len(a) * len(b))
// Reference: standard suffix-matrix approach
func LongestCommonSubstring(a, b string) string {
	ra, rb := []rune(a), []rune(b)
	m, n := len(ra), len(rb)

	if m == 0 || n == 0 {
		return ""
	}

	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}

	bestLen := 0
	bestEnd := 0 // end index in ra (exclusive)

	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if ra[i-1] == rb[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
				if dp[i][j] > bestLen {
					bestLen = dp[i][j]
					bestEnd = i
				}
			}
		}
	}

	return string(ra[bestEnd-bestLen : bestEnd])
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

func min2(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func min3(a, b, c int) int {
	return min2(min2(a, b), c)
}

func max2(a, b int) int {
	if a > b {
		return a
	}
	return b
}


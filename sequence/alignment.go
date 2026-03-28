package sequence

// ---------------------------------------------------------------------------
// Global & local sequence alignment
// ---------------------------------------------------------------------------

// NeedlemanWunsch performs global pairwise alignment of strings a and b
// using the Needleman-Wunsch dynamic programming algorithm. Returns the
// aligned strings (with '-' as gap characters) and the alignment score.
//
// Parameters:
//   - match: score for matching characters (positive, e.g., 1.0)
//   - mismatch: penalty for mismatching characters (negative, e.g., -1.0)
//   - gap: penalty for introducing a gap (negative, e.g., -1.0)
//
// Formula: dp[i][j] = max(dp[i-1][j-1]+s(a[i],b[j]), dp[i-1][j]+gap, dp[i][j-1]+gap)
//   where s(x,y) = match if x==y, mismatch otherwise
// Time complexity: O(len(a) * len(b))
// Space complexity: O(len(a) * len(b))
// Reference: Needleman, Wunsch (1970), "A General Method Applicable to the
//   Search for Similarities in the Amino Acid Sequence of Two Proteins"
func NeedlemanWunsch(a, b string, match, mismatch, gap float64) (string, string, float64) {
	ra, rb := []rune(a), []rune(b)
	m, n := len(ra), len(rb)

	// Build score matrix.
	dp := make([][]float64, m+1)
	for i := range dp {
		dp[i] = make([]float64, n+1)
	}

	// Initialize borders with gap penalties.
	for i := 1; i <= m; i++ {
		dp[i][0] = float64(i) * gap
	}
	for j := 1; j <= n; j++ {
		dp[0][j] = float64(j) * gap
	}

	// Fill matrix.
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			s := mismatch
			if ra[i-1] == rb[j-1] {
				s = match
			}
			diag := dp[i-1][j-1] + s
			up := dp[i-1][j] + gap
			left := dp[i][j-1] + gap
			dp[i][j] = maxFloat(diag, maxFloat(up, left))
		}
	}

	// Traceback.
	alignA := make([]rune, 0, m+n)
	alignB := make([]rune, 0, m+n)
	i, j := m, n

	for i > 0 || j > 0 {
		if i > 0 && j > 0 {
			s := mismatch
			if ra[i-1] == rb[j-1] {
				s = match
			}
			if dp[i][j] == dp[i-1][j-1]+s {
				alignA = append(alignA, ra[i-1])
				alignB = append(alignB, rb[j-1])
				i--
				j--
				continue
			}
		}
		if i > 0 && dp[i][j] == dp[i-1][j]+gap {
			alignA = append(alignA, ra[i-1])
			alignB = append(alignB, '-')
			i--
		} else {
			alignA = append(alignA, '-')
			alignB = append(alignB, rb[j-1])
			j--
		}
	}

	reverseRunes(alignA)
	reverseRunes(alignB)

	return string(alignA), string(alignB), dp[m][n]
}

// SmithWaterman performs local pairwise alignment of strings a and b
// using the Smith-Waterman dynamic programming algorithm. Returns the
// best locally aligned substrings (with '-' as gap characters) and the
// alignment score. Unlike NeedlemanWunsch, cells are floored at zero,
// making it identify the highest-scoring local region.
//
// Parameters:
//   - match: score for matching characters (positive, e.g., 2.0)
//   - mismatch: penalty for mismatching characters (negative, e.g., -1.0)
//   - gap: penalty for introducing a gap (negative, e.g., -1.0)
//
// Formula: dp[i][j] = max(0, dp[i-1][j-1]+s(a[i],b[j]), dp[i-1][j]+gap, dp[i][j-1]+gap)
// Time complexity: O(len(a) * len(b))
// Space complexity: O(len(a) * len(b))
// Reference: Smith, Waterman (1981), "Identification of Common Molecular
//   Subsequences"
func SmithWaterman(a, b string, match, mismatch, gap float64) (string, string, float64) {
	ra, rb := []rune(a), []rune(b)
	m, n := len(ra), len(rb)

	// Build score matrix (borders stay zero).
	dp := make([][]float64, m+1)
	for i := range dp {
		dp[i] = make([]float64, n+1)
	}

	bestScore := 0.0
	bestI, bestJ := 0, 0

	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			s := mismatch
			if ra[i-1] == rb[j-1] {
				s = match
			}
			diag := dp[i-1][j-1] + s
			up := dp[i-1][j] + gap
			left := dp[i][j-1] + gap
			dp[i][j] = maxFloat(0, maxFloat(diag, maxFloat(up, left)))
			if dp[i][j] > bestScore {
				bestScore = dp[i][j]
				bestI = i
				bestJ = j
			}
		}
	}

	if bestScore == 0 {
		return "", "", 0
	}

	// Traceback from best cell until we hit zero.
	alignA := make([]rune, 0, m+n)
	alignB := make([]rune, 0, m+n)
	i, j := bestI, bestJ

	for i > 0 && j > 0 && dp[i][j] > 0 {
		s := mismatch
		if ra[i-1] == rb[j-1] {
			s = match
		}
		if dp[i][j] == dp[i-1][j-1]+s {
			alignA = append(alignA, ra[i-1])
			alignB = append(alignB, rb[j-1])
			i--
			j--
		} else if dp[i][j] == dp[i-1][j]+gap {
			alignA = append(alignA, ra[i-1])
			alignB = append(alignB, '-')
			i--
		} else {
			alignA = append(alignA, '-')
			alignB = append(alignB, rb[j-1])
			j--
		}
	}

	reverseRunes(alignA)
	reverseRunes(alignB)

	return string(alignA), string(alignB), bestScore
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func reverseRunes(r []rune) {
	for left, right := 0, len(r)-1; left < right; left, right = left+1, right-1 {
		r[left], r[right] = r[right], r[left]
	}
}

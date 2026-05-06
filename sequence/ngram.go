package sequence

// ---------------------------------------------------------------------------
// N-gram generation & similarity
// ---------------------------------------------------------------------------

// NGrams extracts all character-level n-grams of length n from string s.
// Returns nil if n <= 0 or len(s) < n.
//
// Example: NGrams("hello", 3) → ["hel", "ell", "llo"]
//
// Time complexity: O(len(s))
// Space complexity: O(len(s) * n) for the output
// Reference: standard sliding-window n-gram extraction
func NGrams(s string, n int) []string {
	runes := []rune(s)
	if n <= 0 || len(runes) < n {
		return nil
	}

	result := make([]string, 0, len(runes)-n+1)
	for i := 0; i <= len(runes)-n; i++ {
		result = append(result, string(runes[i:i+n]))
	}
	return result
}

// WordNGrams extracts all word-level n-grams of length n from the given
// slice of words. Returns nil if n <= 0 or len(words) < n.
//
// Example: WordNGrams(["the","cat","sat"], 2) → [["the","cat"], ["cat","sat"]]
//
// Time complexity: O(len(words))
// Space complexity: O(len(words) * n) for the output
// Reference: standard sliding-window word n-gram extraction
func WordNGrams(words []string, n int) [][]string {
	if n <= 0 || len(words) < n {
		return nil
	}

	result := make([][]string, 0, len(words)-n+1)
	for i := 0; i <= len(words)-n; i++ {
		gram := make([]string, n)
		copy(gram, words[i:i+n])
		result = append(result, gram)
	}
	return result
}

// NGramSimilarity computes the Jaccard similarity coefficient between the
// character n-gram sets of strings a and b. Returns a value in [0, 1]
// where 1.0 means the n-gram sets are identical.
//
// Formula: |A ∩ B| / |A ∪ B|  where A, B are sets of n-grams
// Time complexity: O((len(a) + len(b)) * n)
// Space complexity: O(len(a) + len(b))
// Reference: Jaccard (1912), "The Distribution of the Flora in the Alpine Zone"
func NGramSimilarity(a, b string, n int) float64 {
	gramsA := NGrams(a, n)
	gramsB := NGrams(b, n)

	if len(gramsA) == 0 && len(gramsB) == 0 {
		return 1.0 // Both empty → identical.
	}
	if len(gramsA) == 0 || len(gramsB) == 0 {
		return 0.0
	}

	setA := make(map[string]struct{}, len(gramsA))
	for _, g := range gramsA {
		setA[g] = struct{}{}
	}

	setB := make(map[string]struct{}, len(gramsB))
	for _, g := range gramsB {
		setB[g] = struct{}{}
	}

	// Intersection.
	inter := 0
	for g := range setA {
		if _, ok := setB[g]; ok {
			inter++
		}
	}

	// Union = |A| + |B| - |A ∩ B|.
	union := len(setA) + len(setB) - inter
	if union == 0 {
		return 1.0
	}
	return float64(inter) / float64(union)
}

// NGramDiceCoefficient returns the Sørensen-Dice coefficient between the
// character n-gram sets of strings a and b:
//
//	dice = 2 * |A ∩ B| / (|A| + |B|)
//
// where A and B are the sets of distinct n-grams in a and b respectively.
// Equals 1.0 when both inputs are identical (or both empty); 0.0 when the
// sets are disjoint or exactly one input is empty.
//
// Sørensen-Dice is a common alternative to Jaccard for fuzzy string
// matching at the character level. It weighs the intersection more
// strongly (denominator is sum of cardinalities, not union), so it tends
// to score plausible matches higher than Jaccard. Bounded in [0, 1] and
// related to Jaccard by dice = 2J/(1+J).
//
// Time complexity: O((len(a) + len(b)) * n)
// Space complexity: O(len(a) + len(b))
// References:
//   - Sørensen T. (1948). "A method of establishing groups of equal
//     amplitude in plant sociology based on similarity of species."
//   - Dice L. R. (1945). "Measures of the amount of ecologic association
//     between species."
func NGramDiceCoefficient(a, b string, n int) float64 {
	gramsA := NGrams(a, n)
	gramsB := NGrams(b, n)

	if len(gramsA) == 0 && len(gramsB) == 0 {
		return 1.0
	}
	if len(gramsA) == 0 || len(gramsB) == 0 {
		return 0.0
	}

	setA := make(map[string]struct{}, len(gramsA))
	for _, g := range gramsA {
		setA[g] = struct{}{}
	}
	setB := make(map[string]struct{}, len(gramsB))
	for _, g := range gramsB {
		setB[g] = struct{}{}
	}

	inter := 0
	for g := range setA {
		if _, ok := setB[g]; ok {
			inter++
		}
	}

	denom := len(setA) + len(setB)
	if denom == 0 {
		return 1.0
	}
	return 2.0 * float64(inter) / float64(denom)
}

// Shingling produces MinHash-ready shingles by hashing each k-character
// substring with FNV-1a 64-bit. Returns a slice of hash values, one per
// shingle. Duplicate hash values are preserved (caller may deduplicate).
//
// Returns nil if k <= 0 or len(s) < k.
//
// Time complexity: O(len(s) * k)
// Space complexity: O(len(s)) for the output
// Reference: Broder (1997), "On the Resemblance and Containment of Documents"
func Shingling(s string, k int) []uint64 {
	runes := []rune(s)
	if k <= 0 || len(runes) < k {
		return nil
	}

	result := make([]uint64, 0, len(runes)-k+1)
	for i := 0; i <= len(runes)-k; i++ {
		shingle := string(runes[i : i+k])
		result = append(result, fnv1a64([]byte(shingle)))
	}
	return result
}

// fnv1a64 computes the FNV-1a 64-bit hash. Inlined here to maintain
// the zero-dependency guarantee (no import of reality/crypto).
func fnv1a64(data []byte) uint64 {
	const (
		offsetBasis = uint64(14695981039346656037)
		prime       = uint64(1099511628211)
	)
	h := offsetBasis
	for _, b := range data {
		h ^= uint64(b)
		h *= prime
	}
	return h
}

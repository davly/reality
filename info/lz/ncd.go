package lz

// This file adds the Otu & Sayood (2003) normalised Lempel-Ziv sequence
// distance on top of the LZ76 production-count primitive (LempelZivComplexity).
// It is the "Normalised Compression Distance"-family measure that doc.go
// listed as a v2 deferral ("Normalised Compression Distance ... the gzip +
// k-NN classifier baseline"); this is the LZ76-native form built from the
// four production counts c(a), c(b), c(ab), c(ba) rather than a byte
// compressor. It composes directly on the existing exhaustive parse and adds
// no dependencies (still only `math` + `errors`).

// CrossComplexity returns c(ab) - c(a): the number of additional LZ76
// production words needed to parse b when a already precedes it. This is the
// Lempel-Ziv estimate of the information in b that is not already present in
// a — the "conditional" complexity c(b|a) of Otu & Sayood (2003).
//
// c(ab) >= c(a) always holds for the untruncated parse (appending symbols can
// only add production words), so the result is non-negative; it is clamped at
// 0 to absorb the LZ76MaxSymbols truncation of a very long concatenation
// (where the truncated ab may drop part of b). For meaningful conditional
// complexity keep len(a)+len(b) within LZ76MaxSymbols.
//
// alphabetSize is forwarded to LempelZivComplexity for interface parity with
// the rest of the package; the returned count depends only on the parse and
// is independent of the alphabet hint.
//
// Returns ErrTooShort if either input has fewer than LZ76MinSymbols entries.
func CrossComplexity(a, b []int, alphabetSize int) (int, error) {
	if len(a) < LZ76MinSymbols || len(b) < LZ76MinSymbols {
		return 0, ErrTooShort
	}
	ca, err := LempelZivComplexity(a, alphabetSize)
	if err != nil {
		return 0, err
	}
	cab, err := LempelZivComplexity(concatSymbols(a, b), alphabetSize)
	if err != nil {
		return 0, err
	}
	d := cab.WordCount - ca.WordCount
	if d < 0 {
		d = 0
	}
	return d, nil
}

// NormalizedLZDistance implements the Otu & Sayood (2003) normalised
// Lempel-Ziv sequence distance:
//
//	d(a, b) = max{ c(ab) - c(a), c(ba) - c(b) } / max{ c(a), c(b) }
//
// where c(.) is the LZ76 production-word count (LempelZivComplexity.WordCount)
// and ab / ba are the two orderings of the concatenation. It is a
// compression-based similarity channel: sequences with shared structure parse
// into fewer *additional* production words when concatenated, giving a small
// distance, independently of which literal symbols they share.
//
// Properties (all exercised in ncd_test.go):
//
//   - Symmetric: d(a, b) == d(b, a) exactly (the numerator maxes over both
//     orderings and the denominator over both counts).
//   - Range ≈[0, 1] for sequences of comparable, non-trivial complexity. It is
//     NOT a strict metric: d(a, a) is small but need not be 0 — a single self-
//     copy still costs at least one extra production word — and the value can
//     exceed 1 when one input is near-trivial (e.g. a constant run), because
//     the denominator max{c(a), c(b)} is then tiny. This mirrors the finite-
//     length caveat already documented for NormalizedComplexity.
//
// Reference: Otu, H. H. & Sayood, K. (2003). A new sequence distance measure
// for phylogenetic tree construction. Bioinformatics 19(16): 2122-2130.
//
// alphabetSize is forwarded to LempelZivComplexity for interface parity; the
// distance depends only on the four production counts, which are alphabet-hint
// independent.
//
// Returns ErrTooShort if either input has fewer than LZ76MinSymbols entries
// (the concatenations then automatically satisfy the floor).
func NormalizedLZDistance(a, b []int, alphabetSize int) (float64, error) {
	if len(a) < LZ76MinSymbols || len(b) < LZ76MinSymbols {
		return 0, ErrTooShort
	}
	ca, err := LempelZivComplexity(a, alphabetSize)
	if err != nil {
		return 0, err
	}
	cb, err := LempelZivComplexity(b, alphabetSize)
	if err != nil {
		return 0, err
	}
	cab, err := LempelZivComplexity(concatSymbols(a, b), alphabetSize)
	if err != nil {
		return 0, err
	}
	cba, err := LempelZivComplexity(concatSymbols(b, a), alphabetSize)
	if err != nil {
		return 0, err
	}

	condBGivenA := cab.WordCount - ca.WordCount
	if condBGivenA < 0 {
		condBGivenA = 0
	}
	condAGivenB := cba.WordCount - cb.WordCount
	if condAGivenB < 0 {
		condAGivenB = 0
	}

	num := condBGivenA
	if condAGivenB > num {
		num = condAGivenB
	}
	den := ca.WordCount
	if cb.WordCount > den {
		den = cb.WordCount
	}
	if den == 0 {
		// Unreachable: a successful parse always yields WordCount >= 1.
		return 0, nil
	}
	return float64(num) / float64(den), nil
}

// concatSymbols returns a fresh a++b, never aliasing either input.
func concatSymbols(a, b []int) []int {
	out := make([]int, 0, len(a)+len(b))
	out = append(out, a...)
	return append(out, b...)
}

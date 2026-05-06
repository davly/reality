package sequence

import (
	"sort"
	"strings"
)

// TokenSetRatio returns a similarity score in [0, 100] for two strings using
// the token-set algorithm popularised by SeatGeek FuzzyWuzzy / Maxim
// Bachmann's RapidFuzz. The algorithm is designed for fuzzy matching of
// natural-language strings where the token order, case, and presence of
// extra qualifying words may differ across sources.
//
// Algorithm:
//
//  1. Tokenise each input by whitespace, fold to lowercase, deduplicate via
//     a sorted set, and re-join.
//  2. Compute three composite token strings:
//        t0 = sorted(intersection)
//        t1 = sorted(intersection)  +  " "  +  sorted(diff_a)
//        t2 = sorted(intersection)  +  " "  +  sorted(diff_b)
//  3. Compute simpleRatio for each pair (t0,t1), (t0,t2), (t1,t2) and
//     return the maximum.
//
// simpleRatio is the integer percentage 100 * (1 - lev / (|x|+|y|)) computed
// from the Levenshtein distance.  It rounds half-to-even via float64
// arithmetic; the returned value is the integer floor of (200 * matched
// characters) / (total characters).
//
// Boundary cases:
//   - both empty:        100 (degenerate match)
//   - one empty:         0   (no overlap possible)
//   - identical strings: 100
//   - disjoint tokens:   intersection is empty, t0 is empty, t1 / t2 are
//                        sorted-diffs; ratio collapses to simpleRatio of
//                        the diffs (which can still be > 0 if individual
//                        tokens are near-misses character-wise).
//
// References:
//   - Bachmann, M. (2020-2023). RapidFuzz Python.
//     https://github.com/rapidfuzz/RapidFuzz
//   - SeatGeek FuzzyWuzzy.
//     https://github.com/seatgeek/fuzzywuzzy
//
// Status: substrate-first; first-consumer push pending. Likely consumer is
// FW PCN auto-appeal (driver name fuzzy match across telematics / agreement
// / fines DB) or relic-insurance (claimant address fuzzy match across
// insurer portals).
func TokenSetRatio(a, b string) int {
	tokensA := tokenSet(a)
	tokensB := tokenSet(b)

	if len(tokensA) == 0 && len(tokensB) == 0 {
		return 100
	}
	if len(tokensA) == 0 || len(tokensB) == 0 {
		return 0
	}

	intersect, diffA, diffB := tokenSetSplit(tokensA, tokensB)

	t0 := strings.Join(intersect, " ")
	t1 := joinNonEmpty(t0, strings.Join(diffA, " "))
	t2 := joinNonEmpty(t0, strings.Join(diffB, " "))

	r := simpleRatio(t0, t1)
	if x := simpleRatio(t0, t2); x > r {
		r = x
	}
	if x := simpleRatio(t1, t2); x > r {
		r = x
	}
	return r
}

// tokenSet splits s on whitespace, lowercases each token, deduplicates, and
// returns the sorted result.
func tokenSet(s string) []string {
	fields := strings.Fields(strings.ToLower(s))
	if len(fields) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(fields))
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		if _, ok := seen[f]; ok {
			continue
		}
		seen[f] = struct{}{}
		out = append(out, f)
	}
	sort.Strings(out)
	return out
}

// tokenSetSplit returns (intersect, a-only, b-only) given two sorted unique
// token slices.  All three slices are themselves sorted.
func tokenSetSplit(a, b []string) (intersect, diffA, diffB []string) {
	inB := make(map[string]struct{}, len(b))
	for _, t := range b {
		inB[t] = struct{}{}
	}
	inA := make(map[string]struct{}, len(a))
	for _, t := range a {
		inA[t] = struct{}{}
	}
	for _, t := range a {
		if _, ok := inB[t]; ok {
			intersect = append(intersect, t)
		} else {
			diffA = append(diffA, t)
		}
	}
	for _, t := range b {
		if _, ok := inA[t]; !ok {
			diffB = append(diffB, t)
		}
	}
	return
}

// joinNonEmpty joins parts with " " skipping empty parts.
func joinNonEmpty(parts ...string) string {
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p != "" {
			out = append(out, p)
		}
	}
	return strings.Join(out, " ")
}

// simpleRatio computes the integer percentage 100 * (1 - lev / (|x|+|y|))
// using Levenshtein edit distance. Identical strings return 100. Both empty
// returns 100. One empty returns 0.
func simpleRatio(x, y string) int {
	if x == "" && y == "" {
		return 100
	}
	if x == y {
		return 100
	}
	if x == "" || y == "" {
		return 0
	}
	dist := LevenshteinDistance(x, y)
	total := len([]rune(x)) + len([]rune(y))
	if total == 0 {
		return 100
	}
	matched := total - dist
	if matched < 0 {
		matched = 0
	}
	return (100 * matched) / total
}

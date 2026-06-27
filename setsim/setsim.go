// Package setsim provides generic set-similarity coefficients over slices of
// comparable elements: Jaccard index, Sørensen-Dice coefficient, and the
// overlap (Szymkiewicz-Simpson) coefficient, plus a map-key convenience
// wrapper.
//
// These metrics measure how much two finite sets share, independent of the
// element type. Inputs are plain slices (which may contain duplicates); each
// is treated as a true SET — duplicate elements are collapsed before any
// counting, so SetJaccard([]int{1,1,2}, ...) behaves identically to
// SetJaccard([]int{1,2}, ...).
//
// This generalises the same token/tag/keyword-overlap computation that was
// independently reinvented across the ecosystem, e.g.:
//   - atlas    internal/script.Jaccard([]string)       — sorted-token overlap
//   - gazette  internal/credibility.LinguisticOverlap  — Jaccard over map keys
//   - omegle-forge internal/domain.interestOverlap     — (overlap, union) counts
//
// Distinct from sequence.NGramSimilarity, which is Jaccard over the CHARACTER
// n-gram sets of two STRINGS (a string-fuzzy-match primitive). This package is
// the generic set primitive: the caller supplies the elements directly.
//
// Empty-set convention (matches the flagship set reinventions, atlas and
// gazette, both of which return 0): when the UNION is empty — i.e. both inputs
// are empty after deduplication — every coefficient here returns 0.0, NOT 1.0.
// This differs deliberately from sequence.NGramSimilarity (which returns 1.0
// for two empty strings); a Jaccard/Dice/overlap of two empty SETS is
// mathematically 0/0 and the ecosystem convention for sets is to report "no
// measurable similarity" (0.0) rather than "identical" (1.0). All coefficients
// are bounded in [0, 1].
//
// Zero external dependencies.
package setsim

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// elemSet is the distinct-element set of a slice with NaN canonicalised. Go maps
// treat each NaN as a DISTINCT key (NaN != NaN under IEEE-754), so a raw map
// would not collapse duplicate NaNs -- breaking the SET contract (e.g. J(s, s)
// < 1 for an s containing NaN). NaN is instead tracked as a single canonical
// element via hasNaN.
type elemSet[T comparable] struct {
	m      map[T]struct{}
	hasNaN bool
}

// dedupSet returns the distinct-element set of s (NaN canonicalised).
func dedupSet[T comparable](s []T) elemSet[T] {
	es := elemSet[T]{m: make(map[T]struct{}, len(s))}
	for _, v := range s {
		if isNaNElem(v) {
			es.hasNaN = true
			continue
		}
		es.m[v] = struct{}{}
	}
	return es
}

// size is the number of distinct elements (all NaNs count once).
func (es elemSet[T]) size() int {
	n := len(es.m)
	if es.hasNaN {
		n++
	}
	return n
}

// intersectionSize returns |A ∩ B| with all NaNs treated as one element.
func intersectionSize[T comparable](a, b elemSet[T]) int {
	small, large := a, b
	if len(b.m) < len(a.m) {
		small, large = b, a
	}
	inter := 0
	for v := range small.m {
		if _, ok := large.m[v]; ok {
			inter++
		}
	}
	if a.hasNaN && b.hasNaN {
		inter++
	}
	return inter
}

// isNaNElem reports whether v is a NaN, for the float/complex element types Go
// permits under `comparable`. (x != x is true only for NaN.)
func isNaNElem[T comparable](v T) bool {
	switch x := any(v).(type) {
	case float64:
		return x != x
	case float32:
		return x != x
	case complex128:
		return real(x) != real(x) || imag(x) != imag(x)
	case complex64:
		return real(x) != real(x) || imag(x) != imag(x)
	}
	return false
}

// counts returns |A ∩ B| and |A ∪ B| for the two slices treated as sets.
// Both inputs are deduplicated first. The smaller distinct set is iterated for
// the intersection probe. Union = |A| + |B| - |A ∩ B|.
func counts[T comparable](a, b []T) (intersection, union int) {
	setA := dedupSet(a)
	setB := dedupSet(b)
	intersection = intersectionSize(setA, setB)
	union = setA.size() + setB.size() - intersection
	return intersection, union
}

// ---------------------------------------------------------------------------
// Coefficients
// ---------------------------------------------------------------------------

// SetOverlapCounts returns the cardinalities of the intersection and the union
// of a and b, each treated as a set (duplicates within an input are collapsed
// before counting).
//
// This is the shared primitive the other coefficients are built on; it is
// exported so callers that need the raw counts (e.g. to combine overlaps
// across many pairs, or to weight by union size) need not recompute them.
//
//	intersection = |A ∩ B|
//	union        = |A ∪ B| = |A| + |B| - |A ∩ B|
//
// For two empty inputs both returns are 0.
//
// Time complexity: O(|a| + |b|)
// Space complexity: O(|a| + |b|)
func SetOverlapCounts[T comparable](a, b []T) (intersection, union int) {
	return counts(a, b)
}

// SetJaccard returns the Jaccard similarity index of a and b, each treated as
// a set:
//
//	J(A, B) = |A ∩ B| / |A ∪ B|
//
// The result is in [0, 1]: 1.0 when the two sets are identical (and non-empty),
// 0.0 when they are disjoint. By convention (see package doc), J(∅, ∅) = 0.0.
//
// Time complexity: O(|a| + |b|)
// Space complexity: O(|a| + |b|)
// Reference: Jaccard (1912), "The Distribution of the Flora in the Alpine Zone".
func SetJaccard[T comparable](a, b []T) float64 {
	inter, union := counts(a, b)
	if union == 0 {
		return 0.0
	}
	return float64(inter) / float64(union)
}

// SetDice returns the Sørensen-Dice coefficient of a and b, each treated as a
// set:
//
//	dice(A, B) = 2|A ∩ B| / (|A| + |B|)
//
// Dice weights the intersection more heavily than Jaccard (the denominator is
// the sum of cardinalities, not the union), so it scores partial overlaps
// higher. The result is in [0, 1] and is related to Jaccard by
// dice = 2J / (1 + J). By convention dice(∅, ∅) = 0.0 (the |A| + |B| = 0 case).
//
// Time complexity: O(|a| + |b|)
// Space complexity: O(|a| + |b|)
// References:
//   - Dice L. R. (1945). "Measures of the amount of ecologic association
//     between species."
//   - Sørensen T. (1948). "A method of establishing groups of equal amplitude
//     in plant sociology based on similarity of species content."
func SetDice[T comparable](a, b []T) float64 {
	setA := dedupSet(a)
	setB := dedupSet(b)
	denom := setA.size() + setB.size()
	if denom == 0 {
		return 0.0
	}
	inter := intersectionSize(setA, setB)
	return 2.0 * float64(inter) / float64(denom)
}

// SetOverlapCoefficient returns the overlap (Szymkiewicz-Simpson) coefficient
// of a and b, each treated as a set:
//
//	overlap(A, B) = |A ∩ B| / min(|A|, |B|)
//
// Unlike Jaccard and Dice, the overlap coefficient is 1.0 whenever one set is
// a subset of the other (regardless of size disparity), which makes it the
// right choice for containment-style questions. The result is in [0, 1]. When
// either set is empty min(|A|, |B|) = 0; by convention the result is 0.0.
//
// Time complexity: O(|a| + |b|)
// Space complexity: O(|a| + |b|)
// Reference: Szymkiewicz-Simpson overlap coefficient.
func SetOverlapCoefficient[T comparable](a, b []T) float64 {
	setA := dedupSet(a)
	setB := dedupSet(b)
	minLen := setA.size()
	if setB.size() < minLen {
		minLen = setB.size()
	}
	if minLen == 0 {
		return 0.0
	}
	inter := intersectionSize(setA, setB)
	return float64(inter) / float64(minLen)
}

// MapKeyJaccard returns the Jaccard similarity of the KEY SETS of two maps:
//
//	J(keys(a), keys(b)) = |keys(a) ∩ keys(b)| / |keys(a) ∪ keys(b)|
//
// Values are ignored (V is unconstrained). This mirrors the common pattern of
// computing token-overlap from token→count digests where only key presence
// matters (e.g. gazette.LinguisticOverlap). By convention J(∅, ∅) = 0.0.
//
// Time complexity: O(|a| + |b|)
// Space complexity: O(1) beyond the inputs (no copy is made; keys are probed
// in place).
// Reference: Jaccard (1912).
func MapKeyJaccard[K comparable, V any](a, b map[K]V) float64 {
	// Intersection: probe the smaller map's keys against the larger.
	var small, large map[K]V
	if len(a) <= len(b) {
		small, large = a, b
	} else {
		small, large = b, a
	}
	inter := 0
	for k := range small {
		if _, ok := large[k]; ok {
			inter++
		}
	}
	union := len(a) + len(b) - inter
	if union == 0 {
		return 0.0
	}
	return float64(inter) / float64(union)
}

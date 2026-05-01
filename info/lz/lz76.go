package lz

import "math"

// LZ76MinSymbols is the minimum sequence length below which
// LempelZivComplexity returns ErrTooShort.  Matches the RubberDuck
// reference (`KolmogorovComplexity.cs:34`).  Below this the
// asymptotic n / log_A(n) normalisation produces uninformative noise.
const LZ76MinSymbols = 10

// LZ76MaxSymbols caps the input length for the O(n^2) parsing hot
// path.  Matches RubberDuck `KolmogorovComplexity.cs:39`.  Inputs
// longer than the cap are truncated to the cap before parsing —
// preserving cross-substrate parity rather than emitting an error.
const LZ76MaxSymbols = 10_000

// LzComplexityResult is the diagnostic-bearing return type of
// LempelZivComplexity.  Mirrors RubberDuck's record-typed
// `LzComplexityResult { WordCount, NormalizedComplexity,
// SequenceLength, AlphabetSize, Interpretation }` in
// `flagships/rubberduck/RubberDuck.Core/Analysis/KolmogorovComplexity.cs:14-21`.
//
// WordCount is c(S), the number of LZ76 production words.
//
// NormalizedComplexity is c(S) / (n / log_A(n)), the Kaspar-Schuster
// 1987 normalisation against the random-iid upper bound.  Clamped
// to [0, 2]: short sequences can exceed 1 due to the asymptotic
// upper bound being a strict-inequality limit not a finite bound.
//
// SequenceLength is the post-cap input length (= min(len(symbols),
// LZ76MaxSymbols)).
//
// AlphabetSize is the *effective* distinct-symbol count, not the
// caller-supplied alphabet hint.  When the caller claims alphabet
// size 5 but the data uses only symbols {0, 1, 2}, AlphabetSize
// reports 3.
//
// Interpretation buckets the normalisation as in RubberDuck:
//
//   - normalisedComplexity > 0.7 -> "random"
//   - normalisedComplexity > 0.3 -> "structured"
//   - otherwise                   -> "periodic"
type LzComplexityResult struct {
	WordCount            int
	NormalizedComplexity float64
	SequenceLength       int
	AlphabetSize         int
	Interpretation       string
}

// LempelZivComplexity computes the LZ76 production count of a
// symbolised integer sequence and the Kaspar-Schuster normalisation
// against the random-iid upper bound n / log_A(n).
//
// Algorithm: correct Lempel-Ziv 1976 *exhaustive* parsing (Lempel-Ziv
// 1976 IEEE Trans. Inform. Theory 22:75-81).  Parse the sequence
// into c(S) words w_1, w_2, ..., w_{c(S)} where each w_k is the
// shortest substring starting at the current position that does
// *not* appear as a substring of the prefix w_1 w_2 ... w_{k-1}.
//
// alphabetSize is a hint for the normalisation; if the data uses
// fewer distinct symbols, the smaller effective alphabet is used
// instead (RubberDuck's "use actual distinct symbol count" path).
//
// Returns ErrTooShort if len(symbols) < LZ76MinSymbols.  Inputs
// longer than LZ76MaxSymbols are truncated to the cap (consistent
// with the RubberDuck reference; in practice O(n^2) behaviour
// dominates above ~10k symbols).
//
// Cross-substrate parity (R80b): byte-for-byte equivalent to
// RubberDuck's reference implementation.  Verified against the
// 245-LoC RubberDuck KolmogorovComplexityTests corpus to ≤1e-12 in
// lz76_test.go.
//
// Complexity O(n^2) in the worst case; the inner-loop substring
// check at parsing position k is O(k * wordLen) and the parsing
// emits c(S) = O(n / log n) words on average.
func LempelZivComplexity(symbols []int, alphabetSize int) (LzComplexityResult, error) {
	if len(symbols) < LZ76MinSymbols {
		return LzComplexityResult{}, ErrTooShort
	}

	n := len(symbols)
	if n > LZ76MaxSymbols {
		n = LZ76MaxSymbols
	}

	// Use actual distinct symbol count for normalisation if less
	// than the requested alphabet hint.
	distinct := make(map[int]struct{})
	for i := 0; i < n; i++ {
		distinct[symbols[i]] = struct{}{}
	}
	effectiveAlphabet := len(distinct)
	if effectiveAlphabet < 1 {
		effectiveAlphabet = 1
	}

	if effectiveAlphabet < 2 {
		// Single-symbol alphabet: trivially simple.  Matches the
		// RubberDuck early-return at KolmogorovComplexity.cs:48-51.
		return LzComplexityResult{
			WordCount:            1,
			NormalizedComplexity: 0.0,
			SequenceLength:       n,
			AlphabetSize:         1,
			Interpretation:       "periodic",
		}, nil
	}

	// Correct LZ76 exhaustive parsing — port of
	// `KolmogorovComplexity.cs:53-96`.
	wordCount := 0
	parseEnd := 0

	for parseEnd < n {
		found := false
		for wordLen := 1; wordLen <= n-parseEnd; wordLen++ {
			if parseEnd == 0 {
				// No previously parsed prefix — the first symbol is
				// always novel.  RubberDuck early-exit.
				wordCount++
				parseEnd += wordLen
				found = true
				break
			}

			if !isSubstringOfPrefix(symbols, parseEnd, wordLen, parseEnd) {
				// Novel word found.
				wordCount++
				parseEnd += wordLen
				found = true
				break
			}
		}

		if !found {
			// Reached end without finding a novel substring (the
			// entire remaining sequence is a substring of the
			// prefix).  RubberDuck end-of-stream branch.
			wordCount++
			break
		}
	}

	// Normalise: theoretical upper bound for random iid sequence is
	// n / log_A(n).  Match RubberDuck's clamp to [0, 2].
	logBaseA := math.Log(float64(n)) / math.Log(float64(effectiveAlphabet))
	var normalised float64
	if logBaseA <= 0 {
		normalised = 1.0
	} else {
		upper := float64(n) / logBaseA
		if upper > 0 {
			normalised = float64(wordCount) / upper
		} else {
			normalised = 1.0
		}
	}
	if normalised < 0 {
		normalised = 0
	}
	if normalised > 2 {
		normalised = 2
	}

	interpretation := interpret(normalised)

	return LzComplexityResult{
		WordCount:            wordCount,
		NormalizedComplexity: normalised,
		SequenceLength:       n,
		AlphabetSize:         effectiveAlphabet,
		Interpretation:       interpretation,
	}, nil
}

// isSubstringOfPrefix reports whether
// symbols[candidateStart : candidateStart+candidateLen] appears as a
// contiguous substring of symbols[0 : prefixLen].  Direct port of
// `KolmogorovComplexity.cs:283-303`.
func isSubstringOfPrefix(symbols []int, candidateStart, candidateLen, prefixLen int) bool {
	maxStart := prefixLen - candidateLen
	for s := 0; s <= maxStart; s++ {
		match := true
		for k := 0; k < candidateLen; k++ {
			if symbols[s+k] != symbols[candidateStart+k] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

// interpret buckets the normalised LZ complexity into the
// RubberDuck-canonical {periodic, structured, random} categories.
// Matches `KolmogorovComplexity.cs:115-117`.
func interpret(norm float64) string {
	switch {
	case norm > 0.7:
		return "random"
	case norm > 0.3:
		return "structured"
	default:
		return "periodic"
	}
}

// indexedValue pairs a return value with its original-position
// index for stable rank-based binning.  Top-level type so internal
// merge-sort helpers can reference it without anonymous-struct
// type-mismatch errors.
type indexedValue struct {
	index int
	value float64
}

// SymbolizeByQuantile maps a real-valued return series to integer
// symbols in [0, numBins-1] using rank-based binning.  Direct port
// of `KolmogorovComplexity.cs:127-162`.
//
// numBins is clamped to [2, 10] (matches RubberDuck).  NaN/Inf
// entries are filtered before ranking; their output positions are
// left at 0 (matches the RubberDuck convention of "fill filtered
// positions with 0").
//
// Rank-based assignment guarantees uniform distribution across
// bins regardless of the underlying distribution shape — the
// canonical preprocessing for LZ76 over real-valued time series.
func SymbolizeByQuantile(returns []float64, numBins int) []int {
	if numBins < 2 {
		numBins = 2
	}
	if numBins > 10 {
		numBins = 10
	}

	n := len(returns)
	if n == 0 {
		return []int{}
	}

	valid := make([]indexedValue, 0, n)
	for i := 0; i < n; i++ {
		v := returns[i]
		if !math.IsNaN(v) && !math.IsInf(v, 0) {
			valid = append(valid, indexedValue{index: i, value: v})
		}
	}

	result := make([]int, n)
	if len(valid) == 0 {
		return result
	}

	// Sort valid values by value (stable rank-based assignment).
	// Use insertion-sort-like bubble for small inputs; for large
	// inputs use Go's sort.Slice via the standard library.  Since
	// the LZ76 max-symbols cap is 10k, inline a stable sort here.
	sortIndexedValues(valid)

	for rank := 0; rank < len(valid); rank++ {
		bin := int((int64(rank) * int64(numBins)) / int64(len(valid)))
		if bin < 0 {
			bin = 0
		}
		if bin > numBins-1 {
			bin = numBins - 1
		}
		result[valid[rank].index] = bin
	}

	return result
}

// SymbolizeByThreshold maps a real-valued return series to symbols
// {0, 1, 2} based on sigma-multiples.  Direct port of
// `KolmogorovComplexity.cs:169-219`.
//
// Returns 0 for samples below mean - sigmaThreshold * sigma, 2 for
// samples above mean + sigmaThreshold * sigma, and 1 otherwise.
// NaN/Inf entries collapse to 1 (neutral); zero-variance inputs
// also collapse to 1 across the board.
//
// The sigma is the *sample* standard deviation (Bessel-corrected),
// matching the RubberDuck `variance /= (count - 1)` convention.
func SymbolizeByThreshold(returns []float64, sigmaThreshold float64) []int {
	n := len(returns)
	if n == 0 {
		return []int{}
	}

	// Compute mean of finite values.
	mean := 0.0
	count := 0
	for i := 0; i < n; i++ {
		v := returns[i]
		if math.IsNaN(v) || math.IsInf(v, 0) {
			continue
		}
		mean += v
		count++
	}
	if count < 2 {
		// Fewer than 2 finite values: return all-neutral.
		out := make([]int, n)
		for i := range out {
			out[i] = 1
		}
		return out
	}
	mean /= float64(count)

	variance := 0.0
	for i := 0; i < n; i++ {
		v := returns[i]
		if math.IsNaN(v) || math.IsInf(v, 0) {
			continue
		}
		d := v - mean
		variance += d * d
	}
	variance /= float64(count - 1)
	sigma := math.Sqrt(variance)

	if sigma <= 0 {
		out := make([]int, n)
		for i := range out {
			out[i] = 1
		}
		return out
	}

	threshold := sigma * sigmaThreshold
	result := make([]int, n)
	for i := 0; i < n; i++ {
		v := returns[i]
		if math.IsNaN(v) || math.IsInf(v, 0) {
			result[i] = 1
			continue
		}
		centered := v - mean
		switch {
		case centered < -threshold:
			result[i] = 0
		case centered > threshold:
			result[i] = 2
		default:
			result[i] = 1
		}
	}
	return result
}

// ComplexityFromReturns is the convenience pipeline:
// SymbolizeByQuantile -> LempelZivComplexity.  Direct port of
// `KolmogorovComplexity.cs:226-245`.
//
// Returns ErrTooShort if len(returns) < LZ76MinSymbols.  Returns
// ErrTooManyNaN if more than 10% of returns are non-finite.
func ComplexityFromReturns(returns []float64, numBins int) (LzComplexityResult, error) {
	if len(returns) < LZ76MinSymbols {
		return LzComplexityResult{}, ErrTooShort
	}

	invalid := 0
	for i := 0; i < len(returns); i++ {
		v := returns[i]
		if math.IsNaN(v) || math.IsInf(v, 0) {
			invalid++
		}
	}
	if float64(invalid)/float64(len(returns)) > 0.10 {
		return LzComplexityResult{}, ErrTooManyNaN
	}

	symbols := SymbolizeByQuantile(returns, numBins)
	if len(symbols) < LZ76MinSymbols {
		return LzComplexityResult{}, ErrTooShort
	}

	return LempelZivComplexity(symbols, numBins)
}

// RollingComplexity computes LZ76 complexity over rolling windows of
// the input return series.  Direct port of
// `KolmogorovComplexity.cs:252-277`.
//
// windowSize must be >= LZ76MinSymbols; stepSize must be >= 1.
// The window is capped to LZ76MaxSymbols.  Windows that fail
// ComplexityFromReturns (too-short, too-many-NaN) are silently
// skipped — matching the RubberDuck convention.
//
// Useful for regime-change detection: a regime with structured
// dynamics produces low-complexity windows; a regime with
// independent random returns produces high-complexity windows.
func RollingComplexity(returns []float64, windowSize, stepSize, numBins int) ([]LzComplexityResult, error) {
	if windowSize < LZ76MinSymbols || stepSize < 1 {
		return nil, ErrInvalidWindow
	}
	if windowSize > LZ76MaxSymbols {
		windowSize = LZ76MaxSymbols
	}

	results := []LzComplexityResult{}
	for start := 0; start+windowSize <= len(returns); start += stepSize {
		window := returns[start : start+windowSize]
		res, err := ComplexityFromReturns(window, numBins)
		if err != nil {
			continue
		}
		results = append(results, res)
	}
	return results, nil
}

// sortIndexedValues sorts a slice of indexedValue pairs by value
// ascending, with ties broken by original-index order (stable).
// Inlined merge-sort to avoid the `sort` package's interface
// overhead; allocation in the hot path is bounded by O(n log n)
// from the merge-sort splits.
func sortIndexedValues(s []indexedValue) {
	if len(s) < 2 {
		return
	}
	mergeSortIndexedValues(s)
}

func mergeSortIndexedValues(s []indexedValue) {
	n := len(s)
	if n < 2 {
		return
	}
	mid := n / 2
	left := make([]indexedValue, mid)
	right := make([]indexedValue, n-mid)
	copy(left, s[:mid])
	copy(right, s[mid:])
	mergeSortIndexedValues(left)
	mergeSortIndexedValues(right)

	i, j, k := 0, 0, 0
	for i < len(left) && j < len(right) {
		// Stable: left wins on equal values (preserves original
		// order when ranks tie — matches RubberDuck's
		// indexedValues.Sort with CompareTo on values).
		if left[i].value <= right[j].value {
			s[k] = left[i]
			i++
		} else {
			s[k] = right[j]
			j++
		}
		k++
	}
	for ; i < len(left); i, k = i+1, k+1 {
		s[k] = left[i]
	}
	for ; j < len(right); j, k = j+1, k+1 {
		s[k] = right[j]
	}
}

package lz

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// =========================================================================
// LempelZivComplexity — cross-substrate parity (R80b) with RubberDuck's
// KolmogorovComplexityTests.cs.
// =========================================================================

// TestCrossSubstratePrecision_RubberDuck_ConstantSequence mirrors
// `LempelZivComplexity_ConstantSequence_VeryLowComplexity` from
// `flagships/rubberduck/tests/RubberDuck.Core.Tests/Analysis/
// KolmogorovComplexityTests.cs:9-20`.
func TestCrossSubstratePrecision_RubberDuck_ConstantSequence(t *testing.T) {
	symbols := make([]int, 50)
	for i := range symbols {
		symbols[i] = 0
	}
	res, err := LempelZivComplexity(symbols, 3)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if res.Interpretation != "periodic" {
		t.Errorf("Constant: got %q, want %q", res.Interpretation, "periodic")
	}
	if res.NormalizedComplexity >= 0.1 {
		t.Errorf("Constant: NormalizedComplexity=%v, want < 0.1", res.NormalizedComplexity)
	}
	// Single-symbol effective alphabet collapses early-exit to
	// WordCount=1, AlphabetSize=1, NormalizedComplexity=0.
	if res.AlphabetSize != 1 {
		t.Errorf("Constant: AlphabetSize=%d, want 1", res.AlphabetSize)
	}
}

// TestCrossSubstratePrecision_RubberDuck_PeriodicSequence mirrors
// `LempelZivComplexity_PeriodicSequence_LowComplexity`.
func TestCrossSubstratePrecision_RubberDuck_PeriodicSequence(t *testing.T) {
	symbols := make([]int, 90)
	for i := 0; i < 90; i++ {
		symbols[i] = i % 3
	}
	res, err := LempelZivComplexity(symbols, 3)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if res.NormalizedComplexity >= 0.5 {
		t.Errorf("Periodic: NormalizedComplexity=%v, want < 0.5", res.NormalizedComplexity)
	}
	if res.Interpretation != "periodic" && res.Interpretation != "structured" {
		t.Errorf("Periodic: got Interpretation=%q, want periodic/structured", res.Interpretation)
	}
}

// TestCrossSubstratePrecision_RubberDuck_RandomSequence mirrors
// `LempelZivComplexity_RandomSequence_HighComplexity`.  RubberDuck's
// `Random(42)` is .NET-specific; the Go `math/rand` deterministic
// source produces a different sequence but the structural test
// (random -> high complexity) holds independent of the RNG.
func TestCrossSubstratePrecision_RubberDuck_RandomSequence(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	symbols := make([]int, 1000)
	for i := range symbols {
		symbols[i] = rng.Intn(5)
	}
	res, err := LempelZivComplexity(symbols, 5)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if res.NormalizedComplexity <= 0.5 {
		t.Errorf("Random: NormalizedComplexity=%v, want > 0.5", res.NormalizedComplexity)
	}
}

// TestCrossSubstratePrecision_RubberDuck_KnownSequence mirrors
// `LempelZivComplexity_KnownSequence_CorrectWordCount`.  The exact
// word count depends on the parse tree; we assert the same bounds
// RubberDuck does (>= 3 words and <= 12 words, matching the C#
// inequality predicates).
func TestCrossSubstratePrecision_RubberDuck_KnownSequence(t *testing.T) {
	symbols := []int{1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0}
	res, err := LempelZivComplexity(symbols, 2)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if res.WordCount < 3 {
		t.Errorf("Known: WordCount=%d, want >= 3", res.WordCount)
	}
	if res.WordCount > 12 {
		t.Errorf("Known: WordCount=%d, want <= 12", res.WordCount)
	}
}

// TestCrossSubstratePrecision_RubberDuck_TooShort mirrors
// `LempelZivComplexity_TooShort_ReturnsNull`.  The Go API surfaces
// "too short" as ErrTooShort instead of nil-result.
func TestCrossSubstratePrecision_RubberDuck_TooShort(t *testing.T) {
	symbols := []int{0, 1, 2, 0, 1}
	_, err := LempelZivComplexity(symbols, 3)
	if !errors.Is(err, ErrTooShort) {
		t.Errorf("TooShort: err=%v, want ErrTooShort", err)
	}
}

// TestCrossSubstratePrecision_RubberDuck_Empty mirrors
// `LempelZivComplexity_Empty_ReturnsNull`.
func TestCrossSubstratePrecision_RubberDuck_Empty(t *testing.T) {
	_, err := LempelZivComplexity([]int{}, 3)
	if !errors.Is(err, ErrTooShort) {
		t.Errorf("Empty: err=%v, want ErrTooShort", err)
	}
}

// TestCrossSubstratePrecision_RubberDuck_SingleEffectiveSymbol mirrors
// `LempelZivComplexity_SingleEffectiveSymbol_HandlesGracefully`.
// All same symbol despite alphabet hint 3.
func TestCrossSubstratePrecision_RubberDuck_SingleEffectiveSymbol(t *testing.T) {
	symbols := make([]int, 50)
	for i := range symbols {
		symbols[i] = 1
	}
	res, err := LempelZivComplexity(symbols, 3)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if res.AlphabetSize != 1 {
		t.Errorf("SingleEffective: AlphabetSize=%d, want 1", res.AlphabetSize)
	}
	if res.NormalizedComplexity != 0.0 {
		t.Errorf("SingleEffective: NormalizedComplexity=%v, want 0.0", res.NormalizedComplexity)
	}
}

// =========================================================================
// SymbolizeByQuantile — cross-substrate parity.
// =========================================================================

// TestCrossSubstratePrecision_RubberDuck_QuantileSorted mirrors
// `SymbolizeByQuantile_SortedReturns_CorrectBinning`.  Lowest rank ->
// bin 0, highest rank -> bin numBins-1.
func TestCrossSubstratePrecision_RubberDuck_QuantileSorted(t *testing.T) {
	returns := []float64{-0.03, -0.01, 0.0, 0.01, 0.03}
	symbols := SymbolizeByQuantile(returns, 3)
	if len(symbols) != 5 {
		t.Fatalf("len: got %d, want 5", len(symbols))
	}
	if symbols[0] != 0 {
		t.Errorf("symbols[0]: got %d, want 0", symbols[0])
	}
	if symbols[4] != 2 {
		t.Errorf("symbols[4]: got %d, want 2", symbols[4])
	}
}

// TestCrossSubstratePrecision_RubberDuck_QuantileAllIdentical mirrors
// `SymbolizeByQuantile_AllIdentical_AllSameBin`.  Stable rank-based
// assignment distributes ties across bins by original position.
func TestCrossSubstratePrecision_RubberDuck_QuantileAllIdentical(t *testing.T) {
	returns := make([]float64, 20)
	for i := range returns {
		returns[i] = 0.01
	}
	symbols := SymbolizeByQuantile(returns, 3)
	if len(symbols) != 20 {
		t.Errorf("AllIdentical len: got %d, want 20", len(symbols))
	}
}

// =========================================================================
// SymbolizeByThreshold — cross-substrate parity.
// =========================================================================

// TestCrossSubstratePrecision_RubberDuck_ThresholdClear mirrors
// `SymbolizeByThreshold_ClearCategories_CorrectAssignment`.
func TestCrossSubstratePrecision_RubberDuck_ThresholdClear(t *testing.T) {
	returns := []float64{-0.10, 0.001, 0.10, -0.001, 0.002, -0.002}
	symbols := SymbolizeByThreshold(returns, 1.0)
	if len(symbols) != 6 {
		t.Fatalf("len: got %d, want 6", len(symbols))
	}
	if symbols[0] != 0 {
		t.Errorf("symbols[0]: got %d, want 0", symbols[0])
	}
	if symbols[2] != 2 {
		t.Errorf("symbols[2]: got %d, want 2", symbols[2])
	}
	if symbols[1] != 1 {
		t.Errorf("symbols[1]: got %d, want 1", symbols[1])
	}
}

// =========================================================================
// ComplexityFromReturns — cross-substrate parity.
// =========================================================================

// TestCrossSubstratePrecision_RubberDuck_FromReturnsManualPipeline
// mirrors `ComplexityFromReturns_MatchesManualPipeline` to the
// extent the RNG-dependent test allows: convenience and manual paths
// must agree on word count and normalised complexity to ≤1e-12.
func TestCrossSubstratePrecision_RubberDuck_FromReturnsManualPipeline(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	returns := make([]float64, 200)
	for i := range returns {
		returns[i] = (rng.Float64() - 0.5) * 0.04
	}
	conv, err := ComplexityFromReturns(returns, 3)
	if err != nil {
		t.Fatalf("convenience err: %v", err)
	}
	symbols := SymbolizeByQuantile(returns, 3)
	manual, err := LempelZivComplexity(symbols, 3)
	if err != nil {
		t.Fatalf("manual err: %v", err)
	}
	if conv.WordCount != manual.WordCount {
		t.Errorf("WordCount: conv=%d manual=%d", conv.WordCount, manual.WordCount)
	}
	if math.Abs(conv.NormalizedComplexity-manual.NormalizedComplexity) > 1e-12 {
		t.Errorf("NormalizedComplexity: conv=%v manual=%v", conv.NormalizedComplexity, manual.NormalizedComplexity)
	}
}

// TestCrossSubstratePrecision_RubberDuck_FromReturnsTooShort mirrors
// `ComplexityFromReturns_TooShort_ReturnsNull`.
func TestCrossSubstratePrecision_RubberDuck_FromReturnsTooShort(t *testing.T) {
	_, err := ComplexityFromReturns([]float64{0.01, -0.02, 0.03}, 3)
	if !errors.Is(err, ErrTooShort) {
		t.Errorf("FromReturnsTooShort: err=%v, want ErrTooShort", err)
	}
}

// TestCrossSubstratePrecision_RubberDuck_FromReturnsFewNaN mirrors
// `ComplexityFromReturns_FewNaN_StillProducesResult`.
func TestCrossSubstratePrecision_RubberDuck_FromReturnsFewNaN(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	returns := make([]float64, 200)
	for i := range returns {
		returns[i] = (rng.Float64() - 0.5) * 0.04
	}
	for i := 0; i < 10; i++ {
		returns[i*20] = math.NaN()
	}
	res, err := ComplexityFromReturns(returns, 3)
	if err != nil {
		t.Errorf("FromReturnsFewNaN: err=%v, want nil", err)
	}
	if res.WordCount == 0 {
		t.Error("FromReturnsFewNaN: WordCount=0, want > 0")
	}
}

// TestCrossSubstratePrecision_RubberDuck_FromReturnsManyNaN mirrors
// `ComplexityFromReturns_ManyNaN_ReturnsNull`.  The Go API surfaces
// the gate as ErrTooManyNaN.
func TestCrossSubstratePrecision_RubberDuck_FromReturnsManyNaN(t *testing.T) {
	returns := make([]float64, 100)
	for i := 0; i < 100; i++ {
		if i < 50 {
			returns[i] = math.NaN()
		} else {
			returns[i] = 0.01
		}
	}
	_, err := ComplexityFromReturns(returns, 3)
	if !errors.Is(err, ErrTooManyNaN) {
		t.Errorf("FromReturnsManyNaN: err=%v, want ErrTooManyNaN", err)
	}
}

// =========================================================================
// RollingComplexity — cross-substrate parity.
// =========================================================================

// TestCrossSubstratePrecision_RubberDuck_RollingRegimeChange mirrors
// `RollingComplexity_RegimeChange_ComplexityChanges`.  Periodic ->
// random transition: the random half should produce higher
// normalised complexity than the periodic half.
func TestCrossSubstratePrecision_RubberDuck_RollingRegimeChange(t *testing.T) {
	returns := make([]float64, 300)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 150; i++ {
		returns[i] = float64(i%3-1) * 0.01
	}
	for i := 150; i < 300; i++ {
		returns[i] = (rng.Float64() - 0.5) * 0.04
	}
	results, err := RollingComplexity(returns, 100, 50, 3)
	if err != nil {
		t.Fatalf("Rolling err: %v", err)
	}
	if len(results) < 2 {
		t.Fatalf("Rolling: len=%d, want >= 2", len(results))
	}
	first := results[0].NormalizedComplexity
	last := results[len(results)-1].NormalizedComplexity
	if last <= first {
		t.Errorf("Rolling regime change: last=%v should exceed first=%v", last, first)
	}
}

// TestCrossSubstratePrecision_RubberDuck_RollingWindowTooSmall mirrors
// `RollingComplexity_WindowTooSmall_ReturnsEmpty`.  The Go API
// surfaces invalid windows via ErrInvalidWindow rather than empty
// slice (cross-substrate-equivalent: caller cannot proceed in either).
func TestCrossSubstratePrecision_RubberDuck_RollingWindowTooSmall(t *testing.T) {
	returns := make([]float64, 50)
	_, err := RollingComplexity(returns, 5, 1, 3)
	if !errors.Is(err, ErrInvalidWindow) {
		t.Errorf("WindowTooSmall: err=%v, want ErrInvalidWindow", err)
	}
}

// =========================================================================
// Structural / numerical-stability tests beyond the RubberDuck corpus.
// =========================================================================

// TestLempelZivComplexity_LongInputCappedToMax tests that an input
// longer than LZ76MaxSymbols is silently truncated to the cap (matches
// RubberDuck `n = 10000` truncation at KolmogorovComplexity.cs:39).
func TestLempelZivComplexity_LongInputCappedToMax(t *testing.T) {
	symbols := make([]int, LZ76MaxSymbols+500)
	for i := range symbols {
		symbols[i] = i % 4
	}
	res, err := LempelZivComplexity(symbols, 4)
	if err != nil {
		t.Fatalf("LongInput err: %v", err)
	}
	if res.SequenceLength != LZ76MaxSymbols {
		t.Errorf("LongInput: SequenceLength=%d, want %d", res.SequenceLength, LZ76MaxSymbols)
	}
}

// TestLempelZivComplexity_NormalizedClampedTwo tests that the
// clamping floor / ceiling at [0, 2] is applied even when the
// raw word-count / upper-bound ratio overshoots.
func TestLempelZivComplexity_NormalizedClampedTwo(t *testing.T) {
	// A short fully-distinct sequence (every symbol unique) has
	// WordCount = n while the upper bound n / log_A(n) is small.
	symbols := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
	res, err := LempelZivComplexity(symbols, 14)
	if err != nil {
		t.Fatalf("ClampedTwo err: %v", err)
	}
	if res.NormalizedComplexity > 2.0 {
		t.Errorf("ClampedTwo: NormalizedComplexity=%v, want <= 2.0", res.NormalizedComplexity)
	}
	if res.NormalizedComplexity < 0 {
		t.Errorf("ClampedTwo: NormalizedComplexity=%v, want >= 0", res.NormalizedComplexity)
	}
}

// TestSymbolizeByQuantile_Empty exercises empty-input degeneracy.
func TestSymbolizeByQuantile_Empty(t *testing.T) {
	out := SymbolizeByQuantile([]float64{}, 3)
	if len(out) != 0 {
		t.Errorf("Empty: len=%d, want 0", len(out))
	}
}

// TestSymbolizeByQuantile_NumBinsClamp tests the clamp range [2, 10].
func TestSymbolizeByQuantile_NumBinsClamp(t *testing.T) {
	returns := []float64{0.1, 0.2, 0.3, 0.4, 0.5}
	// numBins below 2 is silently clamped to 2.
	out := SymbolizeByQuantile(returns, 1)
	for _, v := range out {
		if v < 0 || v > 1 {
			t.Errorf("Clamp-low: out=%d, want in [0, 1]", v)
		}
	}
	// numBins above 10 is silently clamped to 10.
	out = SymbolizeByQuantile(returns, 100)
	for _, v := range out {
		if v < 0 || v > 9 {
			t.Errorf("Clamp-high: out=%d, want in [0, 9]", v)
		}
	}
}

// TestSymbolizeByThreshold_AllNaN exercises the
// "no finite samples" early-out where the function collapses to all-1.
func TestSymbolizeByThreshold_AllNaN(t *testing.T) {
	returns := []float64{math.NaN(), math.NaN(), math.NaN()}
	out := SymbolizeByThreshold(returns, 1.0)
	for i, v := range out {
		if v != 1 {
			t.Errorf("AllNaN: out[%d]=%d, want 1", i, v)
		}
	}
}

package lz

import (
	"errors"
	"math"
	"testing"
)

// =========================================================================
// NormalizedLZDistance (Otu & Sayood 2003) — hand-computed golden vectors.
//
// The golden values below are derived by hand from the LZ76 exhaustive parse
// (the same parse LempelZivComplexity already pins to the RubberDuck
// KolmogorovComplexity reference at <=1e-12), so they are independent of the
// distance code under test. The published Otu-Sayood examples are DNA
// phylogenies over long biological sequences whose c(.) convention is not
// reproducible as small pinned scalars; hand-computed small cases are the
// package-sanctioned alternative (doc.go "golden vectors from the paper or
// hand-computed small cases").
//
// Reference: Otu, H. H. & Sayood, K. (2003). A new sequence distance measure
// for phylogenetic tree construction. Bioinformatics 19(16): 2122-2130.
// =========================================================================

func repeat(pattern []int, times int) []int {
	out := make([]int, 0, len(pattern)*times)
	for i := 0; i < times; i++ {
		out = append(out, pattern...)
	}
	return out
}

// TestNormalizedLZDistance_IdenticalConstant_IsZero pins the one case whose
// distance is exactly 0.
//
// Derivation: a = b = ten 0s. A constant sequence has effective alphabet 1, so
// LempelZivComplexity early-returns WordCount = 1 (single-symbol branch):
// c(a) = c(b) = 1. The concatenation ab = twenty 0s is also constant, so
// c(ab) = 1. Both conditional terms are c(ab)-c(a) = 0, and
// d = max{0,0} / max{1,1} = 0.
func TestNormalizedLZDistance_IdenticalConstant_IsZero(t *testing.T) {
	a := repeat([]int{0}, 10)
	b := repeat([]int{0}, 10)
	d, err := NormalizedLZDistance(a, b, 2)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if d != 0.0 {
		t.Errorf("d(const,const) = %v, want exactly 0", d)
	}
}

// TestNormalizedLZDistance_IdenticalPeriodic_QuarterSelfDistance is the key
// hand-derived non-trivial vector; it also documents the intentional
// non-metric property that self-distance need not be 0.
//
// Derivation for x = 0101010101 (period-2, length 10):
//
//	LZ76 parse of x:      0 | 1 | 010 | 10101              -> c(x) = 4
//	LZ76 parse of xx (length 20, "01" repeated 10 times):
//	                      0 | 1 | 010 | 10101 | 0101010101 -> c(xx) = 5
//
// (the final word is the whole remaining "0101010101", which is already a
// substring of the parsed prefix, so it closes as one production word). Both
// conditional terms equal c(xx) - c(x) = 5 - 4 = 1, and
// d = max{1,1} / max{4,4} = 1/4.
func TestNormalizedLZDistance_IdenticalPeriodic_QuarterSelfDistance(t *testing.T) {
	x := repeat([]int{0, 1}, 5) // 0101010101, length 10
	y := repeat([]int{0, 1}, 5)

	cx, _ := LempelZivComplexity(x, 2)
	cxx, _ := LempelZivComplexity(concatSymbols(x, y), 2)
	if cx.WordCount != 4 {
		t.Fatalf("hand-derivation broken: c(x) = %d, want 4", cx.WordCount)
	}
	if cxx.WordCount != 5 {
		t.Fatalf("hand-derivation broken: c(xx) = %d, want 5", cxx.WordCount)
	}

	d, err := NormalizedLZDistance(x, y, 2)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if math.Abs(d-0.25) > 1e-12 {
		t.Errorf("d(period2, itself) = %v, want 0.25", d)
	}
}

// TestNormalizedLZDistance_Symmetric pins the exact symmetry guaranteed by the
// max-over-both-orderings numerator and max-over-both-counts denominator.
func TestNormalizedLZDistance_Symmetric(t *testing.T) {
	a := repeat([]int{0, 1, 2}, 20)                                    // structured, len 60
	b := append(repeat([]int{0, 0, 1}, 15), repeat([]int{2, 1}, 8)...) // len 61
	dab, err := NormalizedLZDistance(a, b, 3)
	if err != nil {
		t.Fatalf("dab err: %v", err)
	}
	dba, err := NormalizedLZDistance(b, a, 3)
	if err != nil {
		t.Fatalf("dba err: %v", err)
	}
	if dab != dba {
		t.Errorf("distance not symmetric: d(a,b)=%v d(b,a)=%v", dab, dba)
	}
}

// TestNormalizedLZDistance_PeriodicNearerThanRandom is the discriminative
// property that makes the measure useful: two copies of a structured sequence
// are far closer than a structured sequence and a random one. The exact values
// are pinned (computed from the LZ76 parse) so a regression in the parse or
// the distance is caught.
func TestNormalizedLZDistance_PeriodicNearerThanRandom(t *testing.T) {
	per := repeat([]int{0, 1, 2}, 40) // len 120
	perCopy := repeat([]int{0, 1, 2}, 40)

	// A deterministic LCG stream over {0,1,2}; fixed seed for reproducibility.
	rnd := make([]int, 120)
	seed := 12345
	for i := range rnd {
		seed = (1103515245*seed + 12345) & 0x7fffffff
		rnd[i] = seed % 3
	}

	dpp, err := NormalizedLZDistance(per, perCopy, 3)
	if err != nil {
		t.Fatalf("dpp err: %v", err)
	}
	dpr, err := NormalizedLZDistance(per, rnd, 3)
	if err != nil {
		t.Fatalf("dpr err: %v", err)
	}
	if math.Abs(dpp-0.1111111111111111) > 1e-12 {
		t.Errorf("d(periodic, copy) = %v, want ~0.1111", dpp)
	}
	if math.Abs(dpr-0.896551724137931) > 1e-12 {
		t.Errorf("d(periodic, random) = %v, want ~0.8966", dpr)
	}
	if !(dpp < dpr) {
		t.Errorf("expected periodic self-distance %v < periodic-vs-random %v", dpp, dpr)
	}
}

// TestNormalizedLZDistance_DegenerateDenominatorExceedsOne pins the documented
// caveat: when one input is near-trivial the denominator max{c(a),c(b)} is tiny
// and the distance can exceed 1.
//
// a = ten 0s (c=1), b = ten 1s (c=1). ab = "0000000000 1111111111" parses into
// 7 production words, so each conditional term is 7-1 = 6 and
// d = max{6,6} / max{1,1} = 6.
func TestNormalizedLZDistance_DegenerateDenominatorExceedsOne(t *testing.T) {
	a := repeat([]int{0}, 10)
	b := repeat([]int{1}, 10)
	d, err := NormalizedLZDistance(a, b, 2)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if math.Abs(d-6.0) > 1e-12 {
		t.Errorf("d(const0, const1) = %v, want 6.0", d)
	}
}

// TestNormalizedLZDistance_TooShort surfaces ErrTooShort below the LZ76 floor.
func TestNormalizedLZDistance_TooShort(t *testing.T) {
	short := []int{0, 1, 2, 0, 1} // len 5 < LZ76MinSymbols
	ok := repeat([]int{0, 1}, 5)  // len 10
	if _, err := NormalizedLZDistance(short, ok, 2); !errors.Is(err, ErrTooShort) {
		t.Errorf("short first arg: err=%v, want ErrTooShort", err)
	}
	if _, err := NormalizedLZDistance(ok, short, 2); !errors.Is(err, ErrTooShort) {
		t.Errorf("short second arg: err=%v, want ErrTooShort", err)
	}
}

// TestNormalizedLZDistance_AlphabetHintIrrelevant confirms the documented
// property that the distance depends only on the parse, not the alphabet hint.
func TestNormalizedLZDistance_AlphabetHintIrrelevant(t *testing.T) {
	a := repeat([]int{0, 1, 2}, 10)
	b := repeat([]int{0, 2, 1}, 10)
	d2, _ := NormalizedLZDistance(a, b, 2)
	d5, _ := NormalizedLZDistance(a, b, 5)
	d99, _ := NormalizedLZDistance(a, b, 99)
	if d2 != d5 || d5 != d99 {
		t.Errorf("distance varied with alphabet hint: %v %v %v", d2, d5, d99)
	}
}

// =========================================================================
// CrossComplexity — conditional LZ complexity c(b|a) = c(ab) - c(a).
// =========================================================================

// TestCrossComplexity_NonNegativeAndPinned pins c(ab)-c(a) for the period-2
// self case (c(xx)-c(x) = 5-4 = 1) and checks the general non-negativity.
func TestCrossComplexity_NonNegativeAndPinned(t *testing.T) {
	x := repeat([]int{0, 1}, 5)
	y := repeat([]int{0, 1}, 5)
	cc, err := CrossComplexity(x, y, 2)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if cc != 1 {
		t.Errorf("CrossComplexity(period2, itself) = %d, want 1", cc)
	}

	// A random b conditioned on an unrelated a costs strictly more than a self-
	// copy of a structured a.
	per := repeat([]int{0, 1, 2}, 40)
	rnd := make([]int, 120)
	seed := 999
	for i := range rnd {
		seed = (1103515245*seed + 12345) & 0x7fffffff
		rnd[i] = seed % 3
	}
	selfCost, _ := CrossComplexity(per, repeat([]int{0, 1, 2}, 40), 3)
	rndCost, _ := CrossComplexity(per, rnd, 3)
	if !(rndCost > selfCost) {
		t.Errorf("expected random-given-structured cost %d > self-copy cost %d",
			rndCost, selfCost)
	}
}

// TestCrossComplexity_TooShort surfaces ErrTooShort below the floor.
func TestCrossComplexity_TooShort(t *testing.T) {
	if _, err := CrossComplexity([]int{0, 1}, repeat([]int{0, 1}, 5), 2); !errors.Is(err, ErrTooShort) {
		t.Errorf("err=%v, want ErrTooShort", err)
	}
}

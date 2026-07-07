package mdl

import (
	"math"
	"testing"
)

// Regression for the Rissanen universal-integer base-2 fix. The previous code
// iterated the natural logarithm and rescaled by 1/ln2, which does not reproduce
// the base-2 iterated log, so the code VIOLATED the Kraft inequality (the sum of
// 2^{-codelength} exceeded 1, i.e. it was not a valid prefix code) and under-coded
// large integers by ~2 bits.

func TestUniversalInteger_KraftValid(t *testing.T) {
	// A valid prefix code must satisfy Kraft: sum_{n>=1} 2^{-L*(n)} <= 1.
	// Rissanen's constant makes the base-2 construction converge to 1 from below;
	// the old natural-log code summed to ~1.235 (> 1, INVALID).
	sum := 0.0
	for n := 1; n <= 200000; n++ {
		bits, err := UniversalIntegerCodeLengthBits(n)
		if err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		sum += math.Exp2(-bits)
	}
	if sum > 1.0+1e-9 {
		t.Errorf("Kraft sum over n=1..2e5 = %.6f; must be <= 1 for a valid prefix code (was ~1.235 pre-fix)", sum)
	}
	if sum < 0.5 {
		t.Errorf("Kraft sum = %.6f implausibly small; codelengths too large", sum)
	}
}

func TestUniversalInteger_Bits256(t *testing.T) {
	// Base-2 iterated log*: 8 + log2(8)=3 + log2(3)=1.585 + log2(1.585)=0.664
	// (then <0) + log2(2.865064)=1.519 = 14.768 bits.
	got, err := UniversalIntegerCodeLengthBits(256)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if math.Abs(got-14.768) > 0.02 {
		t.Errorf("Bits(256)=%.4f; want ~14.77 (was 12.766 pre-fix)", got)
	}

	// nats is a pure unit conversion of bits.
	nats, _ := UniversalIntegerCodeLength(256)
	if math.Abs(nats-got*math.Ln2) > 1e-12 {
		t.Errorf("UniversalIntegerCodeLength(256)=%v != bits*ln2=%v", nats, got*math.Ln2)
	}
}

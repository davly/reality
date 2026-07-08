package mdl

import "math"

// rissanenLogStarConstantBits is Rissanen's constant log2(2.865064...): the
// additive normaliser of the universal integer code, in BITS. Per Rissanen
// (1983) "A universal prior for integers and estimation by minimum description
// length", Annals of Statistics 11(2): 416-431. The constant 2.865064 is the
// value making sum_{n>=1} 2^{-log*(n)} = 1 for the BASE-2 iterated logarithm,
// so the codelength must be computed with log2 throughout to satisfy Kraft.
//
// (Previously this code iterated the NATURAL log and rescaled by 1/ln2, which
// does NOT reproduce the base-2 iterated log — the iterated logarithm is not
// homogeneous under a change of base — so the resulting code violated Kraft:
// sum 2^{-bits} exceeded 1 and large integers were under-coded by ~2 bits.)
var rissanenLogStarConstantBits = math.Log2(2.865064)

// UniversalIntegerCodeLengthBits returns the codelength in BITS of a positive
// integer n under Rissanen's 1983 universal prior, computed natively in base 2:
//
//	log*(n) = log2(n) + log2(log2(n)) + log2(log2(log2(n))) + ...
//	          + log2(2.865064)
//
// where the iteration continues while the inner log2 term remains positive and
// the additive constant ensures Kraft's inequality holds (sum_{n>=1} 2^{-log*(n)}
// = 1). This is the canonical Rissanen code; it MUST be evaluated in base 2,
// because the iterated logarithm is not homogeneous under a change of base
// (log2(log2(n)) != ln(ln(n))/ln2, and the two even differ in their number of
// iteration terms).
//
// Returns ErrInvalidUniversalInt for n < 1. For n = 1 the result is
// rissanenLogStarConstantBits (the bare additive constant; log2(1)=0 terminates
// the recursion immediately).
//
// Reference: Rissanen, J. (1983). A universal prior for integers and estimation
// by minimum description length. Annals of Statistics 11(2): 416-431.
func UniversalIntegerCodeLengthBits(n int) (float64, error) {
	if n < 1 {
		return 0, ErrInvalidUniversalInt
	}

	total := rissanenLogStarConstantBits
	x := math.Log2(float64(n))
	for x > 0 {
		total += x
		x = math.Log2(x)
	}
	return total, nil
}

// UniversalIntegerCodeLength returns the same Rissanen universal-prior codelength
// in NATS — i.e. the bits codelength times ln(2). It is a pure unit conversion of
// UniversalIntegerCodeLengthBits, not a separate (natural-log-iterated) code, so
// the Kraft-valid base-2 codelength is preserved.
func UniversalIntegerCodeLength(n int) (float64, error) {
	bits, err := UniversalIntegerCodeLengthBits(n)
	if err != nil {
		return 0, err
	}
	return bits * math.Ln2, nil
}

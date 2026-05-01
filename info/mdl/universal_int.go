package mdl

import "math"

// rissanenLogStarConstant is Rissanen's constant log(2.865...)
// evaluated in nats: the additive normaliser of the universal
// integer code log*(n) = log(n) + log(log(n)) + ... + log(2.865).
// Per Rissanen (1983) "A universal prior for integers and
// estimation by minimum description length", Annals of Statistics
// 11(2): 416-431.  The constant 2.865 is computed as the unique
// value making sum_{n>=1} 2^{-log*(n)} = 1, ensuring log*(n) is
// a valid Kraft-inequality-satisfying codelength.
//
// Stored as a math.Log of 2.865064... (the value cited in
// Rissanen's original paper); the bit-equivalent constant is
// log_2(2.865) ≈ 1.5189.  We work in nats throughout and convert
// at API boundaries when requested.
//
// Defined as a `var` to allow internal numerical fine-tuning in
// future revisions; downstream callers should treat it as a
// constant and not mutate it.
var rissanenLogStarConstant = math.Log(2.865064)

// UniversalIntegerCodeLength returns the codelength in nats of a
// positive integer n under Rissanen's 1983 universal prior:
//
//	log*(n) = log(n) + log(log(n)) + log(log(log(n))) + ...
//	          + log(2.865064)
//
// where the iteration continues while the inner log term remains
// positive and the final additive constant ensures Kraft's
// inequality is satisfied (sum_{n>=1} 2^{-log*(n)} = 1).
//
// The codelength gives the bits-per-symbol cost of encoding "the
// integer n is required to describe the model" without committing
// to a specific upper bound on n in advance — the canonical
// solution to the integer-prior problem in two-part MDL.
//
// Returns ErrInvalidUniversalInt for n < 1.  For n = 1 the result
// is rissanenLogStarConstant (the bare additive constant; log(1)
// terminates the recursion immediately).
//
// Reference: Rissanen, J. (1983).  A universal prior for integers
// and estimation by minimum description length.  Annals of
// Statistics 11(2): 416-431.
func UniversalIntegerCodeLength(n int) (float64, error) {
	if n < 1 {
		return 0, ErrInvalidUniversalInt
	}

	total := rissanenLogStarConstant
	x := math.Log(float64(n))
	for x > 0 {
		total += x
		x = math.Log(x)
	}
	return total, nil
}

// UniversalIntegerCodeLengthBits is the bit-equivalent of
// UniversalIntegerCodeLength: divide the nat-codelength by ln(2).
// This is what callers want when comparing codelengths to AIC/BIC
// scores reported in bits or to the LZ76 word-count interpretation
// in bits.
func UniversalIntegerCodeLengthBits(n int) (float64, error) {
	nats, err := UniversalIntegerCodeLength(n)
	if err != nil {
		return 0, err
	}
	return nats / math.Ln2, nil
}

package forge

import "math/big"

// DecideExact returns the canonical three-way convergence verdict directly from the RAW
// integer counts behind the dominance and confidence ratios, with NO float division — so the
// verdict never inherits the numeric-ingestion seam that Decide is exposed to (Decide receives
// dominance/confidence as float64 and so carries whatever rounding happened upstream when the
// ratios were formed). dominance = domNum/domDen, confidence = confNum/confDen.
//
// The canonical thresholds are exact rationals — ConvergedDominance 0.70 = 7/10,
// ConvergedConfidence 0.65 = 13/20, EscapeThreshold 0.60 = 3/5 — and every comparison is an
// exact cross-multiplication (math/big, overflow-free). The decision rule matches Decide
// exactly on every well-formed input; it differs only where Decide's float inputs were on the
// wrong side of a threshold due to IEEE-754 rounding, and there DecideExact is correct.
//
// Fails CLOSED to Uncertain on a non-positive denominator (no ratio is defined) or fewer than
// MinObservations, mirroring Decide's fail-closed contract. Consumers eliminate the seam
// end-to-end by passing the counts here instead of pre-dividing into a float.
func DecideExact(domNum, domDen, confNum, confDen, total int) Verdict {
	if domDen <= 0 || confDen <= 0 {
		return VerdictUncertain
	}
	if total < MinObservations {
		return VerdictUncertain
	}
	// dominance >= 0.70 (=7/10) AND confidence >= 0.65 (=13/20) -> Converged
	if ratGE(domNum, domDen, 7, 10) && ratGE(confNum, confDen, 13, 20) {
		return VerdictConverged
	}
	// dominance < 0.60 (=3/5) -> Escape
	if ratLT(domNum, domDen, 3, 5) {
		return VerdictEscape
	}
	return VerdictUncertain
}

// ratGE reports a/b >= c/d exactly, for positive denominators b, d (no overflow, no rounding).
func ratGE(a, b, c, d int) bool {
	l := new(big.Int).Mul(big.NewInt(int64(a)), big.NewInt(int64(d)))
	r := new(big.Int).Mul(big.NewInt(int64(c)), big.NewInt(int64(b)))
	return l.Cmp(r) >= 0
}

// ratLT reports a/b < c/d exactly, for positive denominators b, d (no overflow, no rounding).
func ratLT(a, b, c, d int) bool {
	l := new(big.Int).Mul(big.NewInt(int64(a)), big.NewInt(int64(d)))
	r := new(big.Int).Mul(big.NewInt(int64(c)), big.NewInt(int64(b)))
	return l.Cmp(r) < 0
}

package trust

// CumulativeFusion combines two INDEPENDENT opinions with Jøsang's cumulative
// (aleatory) fusion operator ⊕ (Subjective Logic, 2016, §12.3). Cumulative
// fusion is the belief-calculus equivalent of pooling the two underlying
// bodies of evidence: fusing OpinionFromEvidence(r1,s1) with
// OpinionFromEvidence(r2,s2) yields exactly OpinionFromEvidence(r1+r2, s1+s2).
// The fused uncertainty is therefore SMALLER than either input's — agreement
// backed by more observation is more certain, the property flat averaging and
// stddev-agreement lack.
//
// For non-dogmatic inputs (not both u=0), with κ = uA + uB − uA·uB:
//
//	b = (bA·uB + bB·uA) / κ
//	d = (dA·uB + dB·uA) / κ
//	u = (uA·uB) / κ
//	a = (aA·uB + aB·uA − (aA+aB)·uA·uB) / (uA + uB − 2·uA·uB)
//
// When both inputs are dogmatic (uA = uB = 0) the operator takes the limit
// with equal relative weights: b, d, a become the pairwise averages and u=0.
// The base-rate denominator (uA + uB − 2·uA·uB) can also vanish while κ does
// not (e.g. uA = 1, uB = 1); in that isolated case the base rate falls back
// to the average (aA+aB)/2, matching the operator's limit.
//
// Inputs are assumed valid (see Opinion.Validate); the result satisfies
// b+d+u = 1 to floating-point rounding.
//
// Reference: Jøsang, A. (2016). Subjective Logic, §12.3 (cumulative belief
// fusion).
func CumulativeFusion(a, b Opinion) Opinion {
	uA, uB := a.U, b.U

	// Both dogmatic: limit with equal weights → pairwise average.
	if uA == 0 && uB == 0 {
		return Opinion{
			B: (a.B + b.B) / 2,
			D: (a.D + b.D) / 2,
			U: 0,
			A: (a.A + b.A) / 2,
		}
	}

	kappa := uA + uB - uA*uB
	out := Opinion{
		B: (a.B*uB + b.B*uA) / kappa,
		D: (a.D*uB + b.D*uA) / kappa,
		U: (uA * uB) / kappa,
	}

	aDenom := uA + uB - 2*uA*uB
	if aDenom == 0 {
		out.A = (a.A + b.A) / 2
	} else {
		out.A = (a.A*uB + b.A*uA - (a.A+b.A)*uA*uB) / aDenom
	}
	return out
}

// AveragingFusion combines two opinions with Jøsang's averaging (epistemic)
// fusion operator (Subjective Logic, 2016, §12.5). Use it when the two
// opinions rest on the SAME underlying evidence (dependent sources), so their
// evidence must be averaged rather than accumulated: the fused uncertainty
// stays between the inputs' rather than shrinking below both.
//
// For non-dogmatic inputs (not both u=0):
//
//	b = (bA·uB + bB·uA) / (uA + uB)
//	d = (dA·uB + dB·uA) / (uA + uB)
//	u = 2·uA·uB / (uA + uB)
//	a = (aA + aB) / 2
//
// When both inputs are dogmatic (uA = uB = 0) the result is the pairwise
// average with u=0. Inputs are assumed valid; the result satisfies b+d+u = 1
// to floating-point rounding.
//
// Reference: Jøsang, A. (2016). Subjective Logic, §12.5 (averaging belief
// fusion).
func AveragingFusion(a, b Opinion) Opinion {
	uA, uB := a.U, b.U
	sum := uA + uB
	if sum == 0 {
		return Opinion{
			B: (a.B + b.B) / 2,
			D: (a.D + b.D) / 2,
			U: 0,
			A: (a.A + b.A) / 2,
		}
	}
	return Opinion{
		B: (a.B*uB + b.B*uA) / sum,
		D: (a.D*uB + b.D*uA) / sum,
		U: 2 * uA * uB / sum,
		A: (a.A + b.A) / 2,
	}
}

// FuseAll folds CumulativeFusion over a slice of opinions, left to right.
// Cumulative fusion is commutative and associative (it is evidence addition),
// so the fold order does not affect the result beyond floating-point
// rounding. Returns the vacuous opinion (u=1) with base rate 0.5 for an empty
// slice — no evidence, maximum uncertainty, the correct "nothing observed"
// answer rather than a minted 1.0.
//
// Reference: Jøsang, A. (2016). Subjective Logic, §12.3.
func FuseAll(opinions []Opinion) Opinion {
	if len(opinions) == 0 {
		return Opinion{B: 0, D: 0, U: 1, A: 0.5}
	}
	acc := opinions[0]
	for _, o := range opinions[1:] {
		acc = CumulativeFusion(acc, o)
	}
	return acc
}

// Discount applies transitive trust discounting: it weakens the opinion o
// (what a source asserts about x) by the opinion trust (what WE believe about
// that source's trustworthiness). The source is trusted to the extent of the
// trust opinion's projected probability p = trust.ProbabilityProjection(),
// and the discounted opinion moves belief and disbelief mass into uncertainty
// in proportion to any distrust (Jøsang §14.3.2, base rate-insensitive form):
//
//	b = p·b_o
//	d = p·d_o
//	u = 1 − p·(b_o + d_o)
//	a = a_o
//
// A fully trusted source (p=1) passes its opinion through unchanged; a wholly
// untrusted source (p=0) yields the vacuous opinion (u=1) — its assertion
// carries no belief. This is the principled version of "quality as an opinion
// on the source": low source quality raises uncertainty, it does not silently
// pass through as belief.
//
// Reference: Jøsang, A. (2016). Subjective Logic, §14.3 (trust discounting).
func (o Opinion) Discount(trust Opinion) Opinion {
	p := trust.ProbabilityProjection()
	return Opinion{
		B: p * o.B,
		D: p * o.D,
		U: 1 - p*(o.B+o.D),
		A: o.A,
	}
}

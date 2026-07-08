package trust

import "math"

// additivityTol is the tolerance within which an Opinion's b+d+u must equal 1
// and a MassFunction's masses must sum to 1. Both are elementary sums of a
// handful of terms, so a strict-but-not-exact bound absorbs float rounding
// without admitting genuinely malformed inputs.
const additivityTol = 1e-9

// priorWeight is the non-informative prior weight W in the Beta-to-opinion
// evidence mapping. Jøsang (Subjective Logic, 2016, §3.3) fixes W=2 so that a
// vacuous opinion (r=s=0) maps to base-rate probability under a uniform prior:
// with W=2 and a=0.5 the projected probability of no evidence is exactly 0.5.
const priorWeight = 2.0

// Opinion is a binomial subjective-logic opinion over a binary frame {x, ¬x}:
// a belief mass B in x, a disbelief mass D in ¬x, an uncertainty mass U that
// is committed to neither, and a prior base rate A used to project the
// uncertainty onto a single expected probability.
//
// The masses obey the additivity law B + D + U = 1 with each in [0,1], and
// the base rate A ∈ [0,1]. U is the dimension ordinary probabilities lack:
// it is 1 for a vacuous opinion (no evidence) and shrinks toward 0 only as
// real evidence accumulates — so "no evidence" projects to the base rate A,
// never to certainty.
//
// Reference: Jøsang, A. (2016). Subjective Logic, §3.1-3.3.
type Opinion struct {
	B float64 // belief mass in x
	D float64 // disbelief mass (belief in ¬x)
	U float64 // uncertainty mass, uncommitted
	A float64 // base rate (prior probability) of x
}

// NewOpinion constructs an Opinion and validates it. It returns
// ErrInvalidOpinion if any mass is out of [0,1], the base rate is out of
// [0,1], or additivity B+D+U = 1 is violated beyond additivityTol.
func NewOpinion(b, d, u, a float64) (Opinion, error) {
	o := Opinion{B: b, D: d, U: u, A: a}
	if err := o.Validate(); err != nil {
		return Opinion{}, err
	}
	return o, nil
}

// Validate reports whether the opinion is well formed: all of B, D, U, A in
// [0,1] and B+D+U = 1 within additivityTol.
func (o Opinion) Validate() error {
	for _, m := range []float64{o.B, o.D, o.U, o.A} {
		if math.IsNaN(m) || m < 0 || m > 1 {
			return ErrInvalidOpinion
		}
	}
	if math.Abs(o.B+o.D+o.U-1) > additivityTol {
		return ErrInvalidOpinion
	}
	return nil
}

// OpinionFromEvidence maps r units of positive evidence and s units of
// negative evidence to a binomial opinion via the canonical Beta mapping
// (Jøsang §3.3), with prior weight W = priorWeight = 2:
//
//	b = r / (r + s + 2)
//	d = s / (r + s + 2)
//	u = 2 / (r + s + 2)
//	a = base rate (unchanged by evidence)
//
// With no evidence (r = s = 0) this yields the wholly uncertain opinion
// (b=0, d=0, u=1): the vacuous opinion that projects exactly to the base
// rate a — the property that stops "no evidence" from minting certainty. r
// and s must be non-negative (ErrNegativeEvidence otherwise); a is clamped
// to [0,1].
//
// Reference: Jøsang, A. (2016). Subjective Logic, §3.3 (Beta mapping).
func OpinionFromEvidence(r, s, a float64) (Opinion, error) {
	if math.IsNaN(r) || math.IsNaN(s) || r < 0 || s < 0 {
		return Opinion{}, ErrNegativeEvidence
	}
	a = clamp01(a)
	denom := r + s + priorWeight
	return Opinion{
		B: r / denom,
		D: s / denom,
		U: priorWeight / denom,
		A: a,
	}, nil
}

// Evidence is the inverse of OpinionFromEvidence: it recovers the (r, s)
// positive/negative evidence counts implied by the opinion, using the same
// W=2 mapping (r = 2b/u, s = 2d/u). For a dogmatic opinion (u=0) the implied
// evidence is infinite; Evidence returns (+Inf, +Inf) in that case, which
// callers should treat as "unbounded certainty".
//
// Reference: Jøsang, A. (2016). Subjective Logic, §3.3.
func (o Opinion) Evidence() (r, s float64) {
	if o.U == 0 {
		return math.Inf(1), math.Inf(1)
	}
	return priorWeight * o.B / o.U, priorWeight * o.D / o.U
}

// ProbabilityProjection returns the single expected probability the opinion
// projects to, distributing the uncertainty mass over the base rate:
//
//	P(x) = b + a·u
//
// This is the point estimate a classical probability would collapse to; the
// opinion additionally carries how much of that estimate rests on uncertainty
// (a·u) versus committed belief (b).
//
// Reference: Jøsang, A. (2016). Subjective Logic, §3.2.3.
func (o Opinion) ProbabilityProjection() float64 {
	return o.B + o.A*o.U
}

// IsVacuous reports whether the opinion carries no committed belief at all
// (u = 1, so b = d = 0) — the "no evidence" opinion that must NOT be read as
// trust. This is the belief-calculus name for the vacuous-verification-gate
// condition the estate's trust-detectors suite polices.
func (o Opinion) IsVacuous() bool {
	return o.U == 1
}

// IsDogmatic reports whether the opinion carries no uncertainty at all
// (u = 0): an absolute probability with infinite implied evidence.
func (o Opinion) IsDogmatic() bool {
	return o.U == 0
}

func clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

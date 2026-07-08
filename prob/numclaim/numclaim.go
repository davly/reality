package numclaim

import "math"

// Transform names the value-preserving relationship under which a stated claim
// was found equivalent to a truth. The empty value means "no match".
type Transform string

const (
	// TransformNone is reported for an unmatched claim.
	TransformNone Transform = ""
	// TransformExact: |claim - truth| <= Tolerance, no scaling, no rounding.
	TransformExact Transform = "exact"
	// TransformRounded: the claim is the truth rounded to some decimal place
	// (see Verdict.RoundDP), same scale.
	TransformRounded Transform = "rounded"
	// TransformPercentToFraction: the claim is a PERCENT of the truth's
	// fraction, i.e. claim/100 == truth (e.g. "45%" vs weight 0.45).
	TransformPercentToFraction Transform = "percent_to_fraction"
	// TransformFractionToPercent: the claim is a FRACTION where the truth is a
	// percent, i.e. claim*100 == truth (e.g. 0.45 vs 45).
	TransformFractionToPercent Transform = "fraction_to_percent"
)

// Options configures the equivalence classes NumericEquivalent /
// ClaimConsistency consider. Use DefaultOptions for the standard posture.
type Options struct {
	// MaxRoundDP is the largest number of decimal places at which the claim is
	// allowed to be a rounded rendering of the truth. 0 permits rounding to the
	// nearest integer only; a negative value disables the rounding class.
	MaxRoundDP int
	// PercentScale enables the percent<->fraction equivalence (×100 / ÷100).
	PercentScale bool
	// Tolerance is the absolute bound on the difference between the compared
	// (possibly scaled) values. Negative values are treated as 0.
	Tolerance float64
}

// DefaultOptions is the standard posture: rounding up to 2 decimal places,
// percent<->fraction scaling enabled, and a tight 1e-9 float tolerance.
func DefaultOptions() Options {
	return Options{MaxRoundDP: 2, PercentScale: true, Tolerance: 1e-9}
}

// Verdict is the per-claim result of ClaimConsistency.
type Verdict struct {
	// ClaimIndex is the position of this claim in the input slice.
	ClaimIndex int
	// Claim is the stated numeral under test.
	Claim float64
	// Matched reports whether some truth was found equivalent to the claim.
	Matched bool
	// Transform is the equivalence class used when Matched; TransformNone
	// otherwise.
	Transform Transform
	// RoundDP is the decimal place at which rounding matched, or -1 when no
	// rounding was involved (exact / percent-direct / unmatched).
	RoundDP int
	// TruthIndex is the index of the matched truth, or — when unmatched — the
	// index of the NEAREST-miss truth (smallest relative error). It is -1 only
	// when the truths slice is empty.
	TruthIndex int
	// Truth is the matched or nearest-miss truth value, or NaN when the truths
	// slice is empty.
	Truth float64
	// RelError is the relative error |claim-truth|/max(|claim|,|truth|) between
	// the claim and the reported Truth under the identity scale. It is advisory
	// (near 0 for exact matches, larger for a nearest miss) and is NOT part of
	// the cross-language golden contract.
	RelError float64
}

// match is the internal equivalence result before nearest-miss enrichment.
type match struct {
	matched   bool
	transform Transform
	roundDP   int
}

// NumericEquivalent reports whether the stated claim equals truth under the
// transforms enabled by opts. It is the boolean face of the classifier used by
// ClaimConsistency.
func NumericEquivalent(claim, truth float64, opts Options) bool {
	return classify(claim, truth, opts).matched
}

// ClaimConsistency classifies each claim against the truth pool. For a matched
// claim the verdict names the matching truth and the transform used; for an
// unmatched claim it names the nearest-miss truth (by relative error) so a gate
// can report "closest we found was X". Truths are scanned in index order and
// the first equivalent truth wins, making the result deterministic.
func ClaimConsistency(claims, truths []float64, opts Options) []Verdict {
	out := make([]Verdict, len(claims))
	for i, c := range claims {
		v := Verdict{
			ClaimIndex: i,
			Claim:      c,
			Transform:  TransformNone,
			RoundDP:    -1,
			TruthIndex: -1,
			Truth:      math.NaN(),
		}

		matchedIdx := -1
		var m match
		for j, t := range truths {
			cand := classify(c, t, opts)
			if cand.matched {
				matchedIdx = j
				m = cand
				break
			}
		}

		if matchedIdx >= 0 {
			v.Matched = true
			v.Transform = m.transform
			v.RoundDP = m.roundDP
			v.TruthIndex = matchedIdx
			v.Truth = truths[matchedIdx]
			v.RelError = relError(c, truths[matchedIdx])
		} else if len(truths) > 0 {
			nearest := 0
			nearestErr := math.Inf(1)
			for j, t := range truths {
				if e := relError(c, t); e < nearestErr {
					nearestErr = e
					nearest = j
				}
			}
			v.TruthIndex = nearest
			v.Truth = truths[nearest]
			v.RelError = nearestErr
		}

		out[i] = v
	}
	return out
}

// classify seeks the cheapest transform under which claim == truth, in the
// fixed preference order documented in doc.go: exact, then percent-direct, then
// rounded (identity scale first, then each percent scale).
func classify(claim, truth float64, opts Options) match {
	if !isFinite(claim) || !isFinite(truth) {
		return match{matched: false, transform: TransformNone, roundDP: -1}
	}
	tol := opts.Tolerance
	if tol < 0 {
		tol = 0
	}

	// Candidate (scaled claim value, transform) pairs, in preference order.
	type candidate struct {
		value     float64
		transform Transform
	}
	candidates := []candidate{{claim, TransformExact}}
	if opts.PercentScale {
		candidates = append(candidates,
			candidate{claim / 100.0, TransformPercentToFraction},
			candidate{claim * 100.0, TransformFractionToPercent},
		)
	}

	// Pass 1: exact / percent-direct (no rounding), in candidate order.
	for _, cand := range candidates {
		if math.Abs(cand.value-truth) <= tol {
			return match{matched: true, transform: cand.transform, roundDP: -1}
		}
	}

	// Pass 2: the claim is a ROUNDED rendering of the truth. Scan each
	// candidate scale, dp ascending so the coarsest rounding is reported.
	for _, cand := range candidates {
		for dp := 0; dp <= opts.MaxRoundDP; dp++ {
			if math.Abs(roundDP(truth, dp)-cand.value) <= tol {
				t := cand.transform
				if t == TransformExact {
					t = TransformRounded
				}
				return match{matched: true, transform: t, roundDP: dp}
			}
		}
	}

	return match{matched: false, transform: TransformNone, roundDP: -1}
}

// roundDP rounds x to dp decimal places using round-half-away-from-zero. dp <= 0
// rounds to the nearest integer.
func roundDP(x float64, dp int) float64 {
	if dp <= 0 {
		return math.Round(x)
	}
	p := math.Pow(10, float64(dp))
	return math.Round(x*p) / p
}

// relError is the symmetric relative error, 0 when both operands are 0.
func relError(claim, truth float64) float64 {
	d := math.Abs(claim - truth)
	denom := math.Max(math.Abs(truth), math.Abs(claim))
	if denom == 0 {
		return 0
	}
	return d / denom
}

func isFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

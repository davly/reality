// Package evidence provides decision-NEUTRAL primitives for scoring how strong
// the evidence behind a verdict is, from the SAMPLE SIZE backing it (plus an
// optional effect magnitude and tier weight).
//
// The central idea is that two conclusions with identical headline confidence
// can rest on very different evidence bases: 2,400 observations vs 50. These
// functions surface that "how many cases" dimension as a confidence/quality
// SIGNAL, never as an accept/reject determination. The weakest single sample
// (the "weakest link") is treated as the binding constraint, because a chain of
// evidence is only as strong as its thinnest supporting study.
//
// All functions are pure, operate over primitive inputs (sample counts as ints,
// effect/weight as float64), and use only the Go standard library.
//
// Math summary:
//
//   - Aggregation (Summarize): count = N, total = sum(n_i),
//     min = min(n_i) (the weakest link). A summary is "underpowered" when its
//     weakest sample falls below a threshold: min < threshold.
//
//   - Sample-backing factor: a bounded, monotone, concave map from a sample
//     size to [0,1] using a saturating ratio n/(n+k), where k is the
//     half-saturation constant (the n at which the factor reaches 0.5). This is
//     the standard Bayesian-shrinkage / "add-k" form and rewards extra cases
//     with diminishing returns.
//
//   - Strength score (Score): the product of the sample-backing factor, a
//     clamped effect magnitude in [0,1], and a clamped tier weight in [0,1].
//     The result lies in [0,1]; any single weak dimension caps the score.
//
//   - Grade: a coarse ordinal bucketing of a [0,1] score into
//     None/Weak/Moderate/Strong, for callers that want a label rather than a
//     number.
//
// References:
//   - Gelman et al. (2013) "Bayesian Data Analysis" — add-k shrinkage /
//     pseudo-count saturation as a prior-strength control.
//   - The weakest-link / minimum-sample rule is the standard evidence-grading
//     posture (e.g. GRADE: a body of evidence is downgraded for imprecision
//     driven by its sparsest comparison).
//
// Source: generalized from argus internal/fraud.EvidenceStrengthOf, which
// summarized the sample-size backing of matched fraud evidence; this package
// lifts that math to primitive inputs so any consumer can reuse it.
package evidence

import "math"

// DefaultMinSampleSize is an ILLUSTRATIVE sample-size floor below which a single
// evidence sample is treated as underpowered. It is NOT a regulatory or
// universal threshold — callers should cold-verify against their own
// evidence-governance policy and pass an explicit threshold where one exists.
const DefaultMinSampleSize = 500

// DefaultHalfSaturation is the default half-saturation constant k for
// SampleBackingFactor: the sample size at which the backing factor reaches 0.5.
// It mirrors DefaultMinSampleSize so that, by default, a sample sitting exactly
// at the "underpowered" floor scores half the available backing weight.
const DefaultHalfSaturation = 500.0

// Summary is a decision-NEUTRAL report over the sample sizes behind a set of
// evidence rows. It mirrors the four quantities the teacher computed.
type Summary struct {
	// Count is the number of evidence samples the conclusion rests on.
	Count int
	// Total is the sum of all sample sizes (total cases behind the conclusion).
	Total int
	// Min is the smallest single sample — the weakest link. Zero when Count==0.
	Min int
	// Underpowered is true when the weakest sample falls below the threshold:
	// the conclusion rests partly on thin evidence and deserves more scrutiny.
	// An empty set (Count==0) is never underpowered — there is nothing to
	// under-power.
	Underpowered bool
}

// Summarize reports the sample-size backing of a set of evidence samples.
//
// A non-positive minSampleThreshold uses DefaultMinSampleSize. An empty slice
// (no matched evidence) returns a zero-valued Summary that is NOT underpowered.
// Negative sample sizes are summed as given (callers are expected to pass
// non-negative counts); Min still reports the smallest value seen.
//
// Formula:
//
//	Count = len(samples)
//	Total = sum(samples)
//	Min   = min(samples)          (the weakest link)
//	Underpowered = Min < threshold
//
// Valid range: samples are non-negative counts; threshold any int (<=0 ->
// DefaultMinSampleSize).
// Precision: exact (integer arithmetic).
// Reference: generalized from argus fraud.EvidenceStrengthOf.
func Summarize(samples []int, minSampleThreshold int) Summary {
	if minSampleThreshold <= 0 {
		minSampleThreshold = DefaultMinSampleSize
	}
	s := Summary{Count: len(samples)}
	if len(samples) == 0 {
		return s
	}
	s.Min = samples[0]
	for _, n := range samples {
		s.Total += n
		if n < s.Min {
			s.Min = n
		}
	}
	s.Underpowered = s.Min < minSampleThreshold
	return s
}

// SampleBackingFactor maps a single sample size n to a bounded confidence factor
// in [0,1] using a saturating add-k ratio.
//
// Formula: n / (n + k)
//
//	where k = halfSaturation (the n at which the factor equals 0.5).
//
// Properties: 0 at n=0, monotonically increasing, concave (diminishing returns),
// asymptotes to 1 as n grows. A non-positive halfSaturation uses
// DefaultHalfSaturation. A non-positive n returns 0.
//
// Valid range: n >= 0; halfSaturation > 0 (<=0 -> DefaultHalfSaturation).
// Output range: [0, 1).
// Precision: exact float64 division.
// Reference: add-k / Bayesian-shrinkage pseudo-count saturation; Gelman et al.
// (2013).
func SampleBackingFactor(n int, halfSaturation float64) float64 {
	if !(halfSaturation > 0) {
		// Catches non-positive AND NaN (NaN > 0 is false): a NaN halfSaturation
		// would otherwise skip a plain `<= 0` guard and poison nf/(nf+NaN) to NaN.
		halfSaturation = DefaultHalfSaturation
	}
	if n <= 0 {
		return 0
	}
	nf := float64(n)
	return nf / (nf + halfSaturation)
}

// clamp01 clamps x into [0,1]. NaN maps to 0 so no caller can propagate a NaN.
func clamp01(x float64) float64 {
	switch {
	case math.IsNaN(x):
		return 0
	case x < 0:
		return 0
	case x > 1:
		return 1
	default:
		return x
	}
}

// Score combines the sample-size backing, an effect magnitude, and a tier weight
// into a single evidence-strength score in [0,1].
//
// Formula: SampleBackingFactor(n, halfSaturation) * clamp01(effect) * clamp01(tierWeight)
//
// The product form means any single weak dimension caps the score: large samples
// of a negligible effect, or a strong effect from a single case, both score low.
// effect is the magnitude of the observed effect/rate already normalized to
// [0,1] by the caller (values outside [0,1] are clamped; NaN -> 0). tierWeight
// is the caller's confidence in the evidence source/tier, also in [0,1].
//
// Valid range: n >= 0; effect, tierWeight any float64 (clamped to [0,1]);
// halfSaturation > 0 (<=0 -> DefaultHalfSaturation).
// Output range: [0, 1).
// Precision: float64 multiply/divide.
// Reference: generalized from the argus evidence-strength signal (sample size +
// tier + effect).
func Score(n int, effect, tierWeight, halfSaturation float64) float64 {
	return SampleBackingFactor(n, halfSaturation) * clamp01(effect) * clamp01(tierWeight)
}

// Grade is a coarse ordinal label for an evidence-strength score.
type Grade int

const (
	// GradeNone: essentially no usable evidence backing (score < 0.25).
	GradeNone Grade = iota
	// GradeWeak: thin backing (0.25 <= score < 0.50).
	GradeWeak
	// GradeModerate: reasonable backing (0.50 <= score < 0.75).
	GradeModerate
	// GradeStrong: strong backing (score >= 0.75).
	GradeStrong
)

// String returns a human-readable label for a Grade.
func (g Grade) String() string {
	switch g {
	case GradeNone:
		return "None"
	case GradeWeak:
		return "Weak"
	case GradeModerate:
		return "Moderate"
	case GradeStrong:
		return "Strong"
	default:
		return "Unknown"
	}
}

// GradeScore buckets a [0,1] strength score into an ordinal Grade using the
// fixed quartile thresholds 0.25 / 0.50 / 0.75.
//
// Boundaries are inclusive at the lower edge: a score of exactly 0.25 is Weak,
// 0.50 is Moderate, 0.75 is Strong. The score is clamped to [0,1] first, so a
// NaN or out-of-range input maps to GradeNone / GradeStrong rather than
// producing an undefined grade.
//
// Valid range: any float64 (clamped to [0,1]).
// Precision: exact comparisons after clamp.
func GradeScore(score float64) Grade {
	s := clamp01(score)
	switch {
	case s >= 0.75:
		return GradeStrong
	case s >= 0.50:
		return GradeModerate
	case s >= 0.25:
		return GradeWeak
	default:
		return GradeNone
	}
}

// Package fairness implements the EEOC four-fifths (80%) adverse-impact rule
// for detecting disparate impact in a selection process across protected-class
// groups, together with Wilson score confidence intervals on each group's
// selection rate.
//
// The four-fifths rule is the canonical US regulatory screen for adverse
// impact: a selection process is flagged for further review when the
// least-selected group's selection rate is less than 80% of the most-selected
// group's rate. The ratio of those two rates is the Adverse-Impact Ratio (AIR):
//
//	selectionRate_g = selected_g / total_g           (one rate per group)
//	AIR             = min_g(selectionRate_g) / max_g(selectionRate_g)
//	pass            = AIR >= 0.80
//
// An AIR of 1.0 means perfect parity; an AIR below 0.80 is the regulatory
// threshold for "adverse impact". Because raw rates from small samples are
// noisy, each group's rate is also reported with a Wilson score interval — the
// EEOC- and NIST-recommended binomial interval, far more accurate near 0 or 1
// than the normal approximation.
//
// These are pure functions over PRIMITIVE inputs (selected/total counts per
// group). They carry no hiring-, lending-, or admissions-specific types: the
// same math screens any binary-outcome selection process for disparate impact.
//
// References:
//   - EEOC, "Uniform Guidelines on Employee Selection Procedures" (1978),
//     29 C.F.R. § 1607.4(D) — the four-fifths (80%) rule.
//   - Wilson, E.B. (1927) "Probable Inference, the Law of Succession, and
//     Statistical Inference", JASA 22, 209-212 — the score interval.
package fairness

import (
	"math"
	"sort"
)

// FourFifthsThreshold is the EEOC adverse-impact cutoff. A selection process
// whose Adverse-Impact Ratio is below this value is flagged for review.
const FourFifthsThreshold = 0.80

// DefaultZ is the standard-normal critical value for a 95% Wilson score
// interval (two-sided). Used when a caller passes z <= 0.
const DefaultZ = 1.96

// GroupCount is the primitive input for one protected-class group: the number
// selected out of the total observed in that group.
type GroupCount struct {
	// Label identifies the group (e.g. a protected-class name). It is used
	// only to report which pair drives the worst adverse-impact ratio; the
	// math itself never depends on it.
	Label string
	// Selected is the number of positive outcomes (hired, approved, admitted).
	Selected int
	// Total is the number of observations in the group. Must be >= Selected.
	Total int
}

// GroupRate is the per-group result: the observed selection rate plus its
// Wilson score confidence interval.
type GroupRate struct {
	Label         string
	Selected      int
	Total         int
	SelectionRate float64 // Selected/Total, or 0 when Total == 0.
	CILow         float64 // Wilson score interval lower bound, in [0,1].
	CIHigh        float64 // Wilson score interval upper bound, in [0,1].
}

// AdverseImpactReport bundles the per-group rates with the worst pairwise
// adverse-impact ratio and the four-fifths pass/fail verdict.
type AdverseImpactReport struct {
	// Groups holds one GroupRate per non-empty input group, sorted by Label
	// for determinism.
	Groups []GroupRate
	// MinLabel / MaxLabel identify the least- and most-selected groups that
	// define the adverse-impact ratio.
	MinLabel string
	MaxLabel string
	// AIR is the adverse-impact ratio: min(rate)/max(rate) over groups with
	// Total > 0. It is 0 when there are fewer than two such groups or the
	// maximum rate is 0.
	AIR float64
	// Pass reports whether AIR >= FourFifthsThreshold. It is false when AIR
	// is not computable (Applicable == false).
	Pass bool
	// Applicable is true only when at least two groups with Total > 0 were
	// observed and the maximum selection rate is positive — the conditions
	// under which the four-fifths rule is meaningful.
	Applicable bool
}

// SelectionRate returns selected/total, the observed selection (success) rate
// for one group. It returns 0 when total <= 0, and clamps the result into the
// mathematical range [0,1] so an out-of-range input (selected > total) can
// never produce a rate above 1.
//
// Formula: selected / total, clamped to [0,1]
// Valid range: total >= 0, 0 <= selected <= total
// Precision: exact (single division)
// Reference: EEOC Uniform Guidelines § 1607.4(D)
func SelectionRate(selected, total int) float64 {
	if total <= 0 {
		return 0
	}
	return clamp01(float64(selected) / float64(total))
}

// AdverseImpactRatio computes the four-fifths Adverse-Impact Ratio directly
// from two already-computed selection rates: the smaller rate divided by the
// larger. The result is in [0,1]; 1.0 is perfect parity and a value below
// FourFifthsThreshold (0.80) indicates adverse impact.
//
// The argument order does not matter — the function divides min by max — so
// callers need not know in advance which group selects at the higher rate.
//
// Formula: min(rateA, rateB) / max(rateA, rateB)
// Valid range: rateA, rateB in [0,1]
// Returns 0 if the larger rate is 0 (no selections in either group).
// Precision: exact (single division)
// Reference: EEOC four-fifths rule, 29 C.F.R. § 1607.4(D)
func AdverseImpactRatio(rateA, rateB float64) float64 {
	lo, hi := rateA, rateB
	if lo > hi {
		lo, hi = hi, lo
	}
	if hi <= 0 {
		return 0
	}
	return lo / hi
}

// PassesFourFifths reports whether an adverse-impact ratio satisfies the EEOC
// four-fifths rule: air >= 0.80.
//
// Formula: air >= FourFifthsThreshold
// Reference: EEOC four-fifths rule, 29 C.F.R. § 1607.4(D)
func PassesFourFifths(air float64) bool {
	return air >= FourFifthsThreshold
}

// WilsonScoreInterval returns the [low, high] Wilson score confidence interval
// for a binomial proportion with `selected` successes out of `total` trials.
//
// The Wilson score interval inverts the score test for a binomial proportion;
// unlike the normal (Wald) approximation it never escapes [0,1] and stays
// accurate near the boundaries. z is the standard-normal critical value
// (1.96 for 95% two-sided).
//
// Formula:
//
//	pHat   = selected / total
//	denom  = 1 + z^2/n
//	centre = (pHat + z^2/(2n)) / denom
//	margin = (z * sqrt((pHat(1-pHat) + z^2/(4n)) / n)) / denom
//	interval = [centre - margin, centre + margin]   clamped to [0,1]
//
// Valid range: total > 0, 0 <= selected <= total, z > 0
// If total <= 0, returns (0, 0). If z <= 0, defaults to DefaultZ (1.96).
// Bounds are clamped to [0,1]; a NaN can never be returned.
// Precision: limited by float64 sqrt; ~15 significant digits
// Reference: Wilson, E.B. (1927); EEOC/NIST-recommended for small-n proportions
func WilsonScoreInterval(selected, total int, z float64) (low, high float64) {
	if total <= 0 {
		return 0, 0
	}
	if z <= 0 {
		z = DefaultZ
	}
	n := float64(total)
	pHat := float64(selected) / n
	z2 := z * z
	denom := 1.0 + z2/n
	centre := (pHat + z2/(2.0*n)) / denom
	margin := (z * math.Sqrt((pHat*(1.0-pHat)+z2/(4.0*n))/n)) / denom
	return clamp01(centre - margin), clamp01(centre + margin)
}

// AdverseImpact runs the full four-fifths analysis over a set of protected-class
// groups, returning per-group selection rates with Wilson score intervals, the
// worst pairwise adverse-impact ratio, and the pass/fail verdict.
//
// Groups with Total == 0 are kept in the per-group output (with rate 0 and a
// degenerate (0,0) interval) but excluded from the AIR computation. The AIR is
// the minimum selection rate divided by the maximum selection rate across the
// groups with Total > 0 — the EEOC's worst-case impact ratio. MinLabel and
// MaxLabel identify the pair that defines it.
//
// The report is Applicable only when at least two groups with Total > 0 were
// observed and the maximum rate is positive; otherwise AIR is 0, Pass is false,
// and Applicable is false (the rule is not meaningful with one group or no
// selections). Output groups are sorted by Label for determinism.
//
// z is the Wilson critical value (DefaultZ = 1.96 when z <= 0).
//
// Reference: EEOC four-fifths rule, 29 C.F.R. § 1607.4(D); Wilson (1927)
func AdverseImpact(groups []GroupCount, z float64) AdverseImpactReport {
	rates := make([]GroupRate, 0, len(groups))
	for _, g := range groups {
		low, high := WilsonScoreInterval(g.Selected, g.Total, z)
		rates = append(rates, GroupRate{
			Label:         g.Label,
			Selected:      g.Selected,
			Total:         g.Total,
			SelectionRate: SelectionRate(g.Selected, g.Total),
			CILow:         low,
			CIHigh:        high,
		})
	}
	sort.SliceStable(rates, func(i, j int) bool {
		return rates[i].Label < rates[j].Label
	})

	report := AdverseImpactReport{Groups: rates}

	minRate, maxRate := math.Inf(1), math.Inf(-1)
	var minLabel, maxLabel string
	eligible := 0
	for _, r := range rates {
		if r.Total <= 0 {
			continue
		}
		eligible++
		if r.SelectionRate < minRate {
			minRate = r.SelectionRate
			minLabel = r.Label
		}
		if r.SelectionRate > maxRate {
			maxRate = r.SelectionRate
			maxLabel = r.Label
		}
	}

	if eligible < 2 || maxRate <= 0 {
		return report
	}

	report.Applicable = true
	report.MinLabel = minLabel
	report.MaxLabel = maxLabel
	report.AIR = minRate / maxRate
	report.Pass = PassesFourFifths(report.AIR)
	return report
}

// clamp01 clamps x into the mathematical probability range [0,1]. NaN maps to
// 0 so no caller can propagate a NaN bound.
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

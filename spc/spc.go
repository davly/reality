// Package spc computes the statistical-process-control (SPC) process-capability
// indices — Cp, Cpk, Pp, Ppk — together with the process sigma level, the
// expected two-sided DPMO (defects per million opportunities), and the expected
// yield, from specification limits and process statistics. It also classifies a
// Cpk against a published capability floor to an informational rating.
//
// These are the standard AIAG / SPC textbook formulas. For a process with mean
// mu, short-term (within-subgroup) standard deviation sigma_w, long-term
// (overall) standard deviation sigma_o, and a tolerance band [LSL, USL]:
//
//	Cp  = (USL - LSL) / (6 * sigma_w)
//	Cpk = min(USL - mu, mu - LSL) / (3 * sigma_w)
//	Pp  = (USL - LSL) / (6 * sigma_o)         // same as Cp with the overall sigma
//	Ppk = min(USL - mu, mu - LSL) / (3 * sigma_o)
//	SigmaLevel = 3 * Cpk
//	DPMO  = 1e6 * P(X < LSL or X > USL),  X ~ N(mu, sigma_o^2)
//	Yield = 1 - DPMO/1e6
//
// Cp/Pp measure POTENTIAL capability (the spec width versus the spread, ignoring
// centering); Cpk/Ppk measure ACTUAL capability (penalising an off-centre mean).
// Hence Cpk <= Cp and Ppk <= Pp always, with equality only when the process is
// perfectly centred. The DPMO/yield are computed from the OVERALL (long-term)
// sigma because that is the distribution a customer actually experiences.
//
// Classify maps a Cpk against a published floor to a CAPABLE / MARGINAL /
// INCAPABLE rating. This is an informational SIGNAL against a documented
// criterion (e.g. the AIAG floor of 1.33 for critical characteristics), NOT a
// part-release or disposition determination — that judgement belongs to a
// qualified quality engineer.
//
// All functions operate over primitive inputs (float64 / [][]float64) and use
// only the Go standard library.
//
// References:
//   - AIAG, "Statistical Process Control (SPC)", 2nd ed. (2005).
//   - Montgomery, D.C., "Introduction to Statistical Quality Control", 7th ed.
//     (2013), ch. 8 (process-capability analysis).
//   - Kane, V.E. (1986), "Process Capability Indices", Journal of Quality
//     Technology 18(1), 41-52.
package spc

import (
	"errors"
	"math"
)

// AIAGCpkFloor is the AIAG-published minimum Cpk for critical characteristics
// (1.33, i.e. a 4-sigma margin to the nearer limit). It is exported as a
// convenience constant for callers passing a floor to Classify; callers may
// supply any floor they wish.
const AIAGCpkFloor = 1.33

var (
	// ErrNonPositiveSigma is returned when the within-subgroup sigma is <= 0.
	ErrNonPositiveSigma = errors.New("spc: sigma must be positive")
	// ErrSpecOrder is returned when USL <= LSL (an empty or inverted tolerance).
	ErrSpecOrder = errors.New("spc: USL must be greater than LSL")
)

// Study is the input to a process-capability study: the specification limits and
// the process statistics, all in the measured units of the characteristic.
type Study struct {
	USL          float64 // upper specification limit
	LSL          float64 // lower specification limit
	Mean         float64 // process mean (mu)
	SigmaWithin  float64 // short-term / within-subgroup standard deviation -> Cp, Cpk
	SigmaOverall float64 // long-term / overall standard deviation -> Pp, Ppk, DPMO (<=0 => use SigmaWithin)
}

// Result holds the computed capability metrics for a Study.
type Result struct {
	Cp            float64 // potential capability,  within-subgroup sigma
	Cpk           float64 // actual capability,     within-subgroup sigma (centering-penalised)
	Pp            float64 // potential performance, overall sigma
	Ppk           float64 // actual performance,    overall sigma (centering-penalised)
	SigmaLevel    float64 // process sigma level = 3 * Cpk
	DPMO          float64 // expected defects per million (two-sided, overall sigma)
	ExpectedYield float64 // 1 - DPMO/1e6
}

// Cp computes the potential capability index from a tolerance band and a sigma:
//
//	Cp = (USL - LSL) / (6 * sigma)
//
// Cp measures the spec width against the process spread, ignoring centering, so
// it is the BEST capability the process could attain if perfectly centred.
// Returns ErrSpecOrder if USL <= LSL and ErrNonPositiveSigma if sigma <= 0.
//
// Precision: a subtraction and a division; ~15 significant digits (float64).
func Cp(usl, lsl, sigma float64) (float64, error) {
	if usl <= lsl {
		return 0, ErrSpecOrder
	}
	if sigma <= 0 {
		return 0, ErrNonPositiveSigma
	}
	return (usl - lsl) / (6.0 * sigma), nil
}

// Cpk computes the actual capability index, penalising an off-centre mean:
//
//	Cpk = min(USL - mean, mean - LSL) / (3 * sigma)
//
// Cpk equals Cp only when the mean sits exactly on the spec midpoint; otherwise
// Cpk < Cp. A negative Cpk indicates the mean lies outside the tolerance band.
// Returns ErrSpecOrder if USL <= LSL and ErrNonPositiveSigma if sigma <= 0.
//
// Precision: a min, a subtraction, and a division; ~15 significant digits.
func Cpk(usl, lsl, mean, sigma float64) (float64, error) {
	if usl <= lsl {
		return 0, ErrSpecOrder
	}
	if sigma <= 0 {
		return 0, ErrNonPositiveSigma
	}
	return math.Min((usl-mean)/(3.0*sigma), (mean-lsl)/(3.0*sigma)), nil
}

// DPMO computes the expected two-sided defects-per-million-opportunities for a
// normal process N(mean, sigma^2) against the tolerance band [LSL, USL]:
//
//	DPMO = 1e6 * (P(X < LSL) + P(X > USL))
//
// where X ~ N(mean, sigma^2). A centred process at Cpk = 1.0 (the limits at
// +/-3 sigma) yields ~2700 DPMO; at Cpk = 2.0 (six sigma) ~0.002 DPMO.
// Returns ErrSpecOrder if USL <= LSL and ErrNonPositiveSigma if sigma <= 0.
//
// Precision: limited by math.Erfc; the tails are exact to float64 erfc accuracy.
func DPMO(usl, lsl, mean, sigma float64) (float64, error) {
	if usl <= lsl {
		return 0, ErrSpecOrder
	}
	if sigma <= 0 {
		return 0, ErrNonPositiveSigma
	}
	pBelow := normalCDF((lsl - mean) / sigma)
	pAbove := 1.0 - normalCDF((usl-mean)/sigma)
	return 1e6 * (pBelow + pAbove), nil
}

// Compute returns the full set of capability metrics for a study.
// SigmaWithin must be > 0 and USL > LSL, otherwise an error is returned. If
// SigmaOverall <= 0 it falls back to SigmaWithin, so Pp == Cp and Ppk == Cpk.
//
// The DPMO and ExpectedYield are computed from the OVERALL sigma — the long-term
// spread the customer actually experiences.
//
// Precision: the index arithmetic is exact to float64; DPMO is limited by Erfc.
func Compute(s Study) (Result, error) {
	if s.SigmaWithin <= 0 {
		return Result{}, ErrNonPositiveSigma
	}
	if s.USL <= s.LSL {
		return Result{}, ErrSpecOrder
	}
	overall := s.SigmaOverall
	if overall <= 0 {
		overall = s.SigmaWithin
	}
	r := Result{
		Cp:  (s.USL - s.LSL) / (6.0 * s.SigmaWithin),
		Cpk: math.Min((s.USL-s.Mean)/(3.0*s.SigmaWithin), (s.Mean-s.LSL)/(3.0*s.SigmaWithin)),
		Pp:  (s.USL - s.LSL) / (6.0 * overall),
		Ppk: math.Min((s.USL-s.Mean)/(3.0*overall), (s.Mean-s.LSL)/(3.0*overall)),
	}
	r.SigmaLevel = 3.0 * r.Cpk
	pBelow := normalCDF((s.LSL - s.Mean) / overall)
	pAbove := 1.0 - normalCDF((s.USL-s.Mean)/overall)
	r.DPMO = 1e6 * (pBelow + pAbove)
	r.ExpectedYield = 1.0 - r.DPMO/1e6
	return r, nil
}

// normalCDF is the standard-normal CDF Phi(z) via the complementary error
// function: Phi(z) = 0.5 * erfc(-z / sqrt(2)).
func normalCDF(z float64) float64 {
	return 0.5 * math.Erfc(-z/math.Sqrt2)
}

// Rating is an informational capability signal against a Cpk floor.
type Rating int

const (
	// Incapable means Cpk < 1.0 (the process spread does not fit the tolerance
	// with the mean's current centring).
	Incapable Rating = iota
	// Marginal means 1.0 <= Cpk < floor (capable, but below the supplied
	// critical-characteristic minimum).
	Marginal
	// Capable means Cpk >= floor.
	Capable
)

// String renders a Rating as CAPABLE / MARGINAL / INCAPABLE.
func (r Rating) String() string {
	switch r {
	case Capable:
		return "CAPABLE"
	case Marginal:
		return "MARGINAL"
	default:
		return "INCAPABLE"
	}
}

// ClassifyCpk maps a Cpk against a floor to a SIGNAL: CAPABLE (Cpk >= floor),
// MARGINAL (1.0 <= Cpk < floor), INCAPABLE (Cpk < 1.0). The floor is the caller's
// published criterion (e.g. AIAGCpkFloor = 1.33 for critical characteristics).
//
// This is an informational flag against a documented criterion, NOT a
// part-release or disposition determination.
func ClassifyCpk(cpk, floor float64) Rating {
	switch {
	case cpk >= floor:
		return Capable
	case cpk >= 1.0:
		return Marginal
	default:
		return Incapable
	}
}

// Classify is ClassifyCpk applied to a computed Result's Cpk. See ClassifyCpk.
func Classify(r Result, floor float64) Rating {
	return ClassifyCpk(r.Cpk, floor)
}

// PooledWithinSigma estimates the short-term (within-subgroup) standard deviation
// as the square root of the pooled within-subgroup variance:
//
//	sigma_w = sqrt( sum_i (n_i - 1) * s_i^2 / sum_i (n_i - 1) )
//
// where s_i^2 is the sample variance of subgroup i. Each subgroup must contain
// at least two points. Returns an error if there are no subgroups or any
// subgroup has fewer than two points.
//
// This is the standard pooled-variance estimator: it isolates the variation
// WITHIN subgroups (short-term, common-cause noise) from any shift BETWEEN them.
//
// Precision: accumulated float64 summation error, then a single sqrt.
func PooledWithinSigma(subgroups [][]float64) (float64, error) {
	if len(subgroups) == 0 {
		return 0, errors.New("spc: no subgroups")
	}
	num, den := 0.0, 0.0
	for _, g := range subgroups {
		v, err := sampleVariance(g)
		if err != nil {
			return 0, err
		}
		w := float64(len(g) - 1)
		num += w * v
		den += w
	}
	return math.Sqrt(num / den), nil
}

// OverallSigma is the long-term sample standard deviation of all points pooled
// together — the unbiased (n-1 denominator) sample standard deviation. Requires
// at least two points.
//
//	sigma_o = sqrt( sum (x_i - xbar)^2 / (n - 1) )
//
// Precision: accumulated float64 summation error, then a single sqrt.
func OverallSigma(points []float64) (float64, error) {
	v, err := sampleVariance(points)
	if err != nil {
		return 0, err
	}
	return math.Sqrt(v), nil
}

// sampleVariance computes the unbiased (n-1 denominator) sample variance of xs.
// Requires at least two points.
func sampleVariance(xs []float64) (float64, error) {
	n := len(xs)
	if n < 2 {
		return 0, errors.New("spc: need >= 2 points for a variance")
	}
	mean := 0.0
	for _, x := range xs {
		mean += x
	}
	mean /= float64(n)
	ss := 0.0
	for _, x := range xs {
		d := x - mean
		ss += d * d
	}
	return ss / float64(n-1), nil
}

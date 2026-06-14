// Package moments provides general, numerically-stable streaming (online,
// single-pass) first- and second-moment estimators over primitive scalar and
// fixed-dimension vector inputs.
//
// It exposes the classic Welford (1962) online mean/variance recurrence as the
// scalar Welford and the per-dimension WelfordVec, plus the Chan-Golub-LeVeque
// (1979) parallel combine via Merge / Welford.Merge. These are the GENERAL
// primitives behind the same recurrence that reality already ships only in a
// DOMAIN-LOCKED form inside audio.Fingerprint / audio.BaselineStats (the
// [MelBandCount]float64 mel-band array), and that the wider ecosystem has
// independently reinvented at least three times:
//
//   - flagships/folio/backend/internal/forge/welford.go  (scalar Update /
//     Variance / StdDev / Merge — the cleanest reference, mirrored here)
//   - flagships/cloudforge/internal/cloudops/resource.go (cost/util mean+M2
//     baseline with a sigma-threshold z-score for anomaly detection)
//   - flagships/echo-chamber/internal/spectral/spectral.go (per-dimension
//     mel-band mean+M2 array form — the vector case)
//
// # Welford's online algorithm (Welford 1962; Knuth TAOCP vol 2 §4.2.2)
//
// For the n-th sample x, with mean / M2 held BEFORE the update:
//
//	n      = n + 1
//	delta  = x - mean          // deviation against the OLD mean
//	mean   = mean + delta / n
//	delta2 = x - mean          // deviation against the NEW mean
//	M2     = M2 + delta * delta2
//
// M2 is the running sum of squared deviations from the (current) mean. The
// sample (unbiased, n-1) variance is M2/(n-1); the population (n) variance is
// M2/n. The recompute-after-update formulation (delta2 taken against the new
// mean) is the numerically-stable variant: it never forms sum(x^2) and so does
// NOT suffer the catastrophic cancellation of the naive two-pass identity
// var = (sum(x^2) - n*mean^2) / (n-1) when the data sit far from zero (e.g. a
// 1e9 offset, where sum(x^2) and n*mean^2 agree to ~16 significant figures and
// their difference is dominated by rounding error). See moments_test.go for the
// discriminating offset-invariance test that exhibits this.
//
// # Chan-Golub-LeVeque parallel merge (1979)
//
// Two independently-accumulated states a and b combine without re-streaming:
//
//	n     = nA + nB
//	delta = meanB - meanA
//	mean  = meanA + delta * nB / n
//	M2    = M2_A + M2_B + delta^2 * nA * nB / n
//
// Merge over any partition of a series yields (to floating-point tolerance) the
// same mean and M2 as a single Welford over the whole series, which makes the
// estimator associative and safe to shard / parallelise.
//
// # Edge / degenerate conventions (matched to folio and the standard)
//
//   - Empty (n == 0): Mean returns 0 (the zero value — matching folio's scalar
//     WelfordState, NOT NaN). Variance / PopVariance / StdDev return 0.
//   - Single sample (n == 1): Mean is that sample; Variance (sample, n-1) and
//     PopVariance and StdDev all return 0 — a single observation carries no
//     dispersion information. (Variance with n<2 = 0, matching folio and the
//     standard "undefined => 0" convention used across the reinventions.)
//   - ZScore on an EMPTY accumulator (n == 0) returns 0: no baseline => no
//     information (matching timeseries.EWMoments). On a NON-empty but
//     zero-spread baseline (single sample or constant stream, StdDev == 0)
//     ZScore returns 0 when x equals the mean and +/-Inf otherwise (an honest
//     "departed from a degenerate baseline" sentinel; the caller owns whatever
//     verdict it draws — decision-neutral).
//
// moments never panics on a query of the scalar Welford. WelfordVec panics only
// on a dimension mismatch (a programming error), mirroring audio.Fingerprint.
//
// This package is Tier-0: it imports only the standard library math.
package moments

import "math"

// Welford is a streaming, numerically-stable estimator of the running mean and
// (sample / population) variance of a scalar stream, using Welford's (1962)
// online algorithm. The zero value is an empty, ready-to-use accumulator
// (count 0, mean 0, M2 0). Update folds in one observation in O(1) time and
// space; Merge combines two independently-accumulated states.
//
// Fields are unexported: callers drive state exclusively through Update / Merge,
// which keeps the (count, mean, M2) triple internally consistent. (The
// domain-locked audio.Fingerprint exports its fields for zero-alloc / custom
// serialisation; the general primitive favours an encapsulated invariant.)
type Welford struct {
	n    int
	mean float64
	m2   float64 // running sum of squared deviations from the current mean
}

// Update folds one observation x into w in-place, in O(1).
//
//	n      = n + 1
//	delta  = x - mean
//	mean   = mean + delta / n
//	delta2 = x - mean        // after the mean update
//	M2     = M2 + delta * delta2
func (w *Welford) Update(x float64) {
	w.n++
	delta := x - w.mean
	w.mean += delta / float64(w.n)
	delta2 := x - w.mean
	w.m2 += delta * delta2
}

// Count returns the number of observations folded in so far.
func (w *Welford) Count() int { return w.n }

// Mean returns the running arithmetic mean, or 0 for an empty accumulator
// (matching folio's scalar WelfordState convention, not NaN).
func (w *Welford) Mean() float64 { return w.mean }

// Variance returns the unbiased SAMPLE variance (divisor n-1):
//
//	var = M2 / (n - 1)
//
// It returns 0 when n < 2 (a single observation carries no dispersion
// information — the standard "undefined => 0" convention, matching folio).
func (w *Welford) Variance() float64 {
	if w.n < 2 {
		return 0
	}
	return w.m2 / float64(w.n-1)
}

// PopVariance returns the POPULATION variance (divisor n):
//
//	var = M2 / n
//
// It returns 0 when n < 1 (an empty accumulator has no variance). With exactly
// one observation the population variance is 0 by definition.
func (w *Welford) PopVariance() float64 {
	if w.n < 1 {
		return 0
	}
	return w.m2 / float64(w.n)
}

// StdDev returns the unbiased SAMPLE standard deviation, sqrt(Variance()); 0
// when n < 2.
func (w *Welford) StdDev() float64 {
	return math.Sqrt(w.Variance())
}

// ZScore returns the standardised distance of x from the running mean in units
// of the sample standard deviation:
//
//	z = (x - mean) / StdDev()
//
// Conventions for degenerate baselines:
//   - Empty accumulator (n == 0): there is no baseline at all, so ZScore
//     returns 0 (no information), matching the sibling timeseries.EWMoments
//     convention — NOT a sentinel.
//   - Non-empty but zero-spread baseline (a single sample, or a constant
//     stream, so StdDev == 0): ZScore returns 0 if x equals the mean and
//     +/-Inf otherwise — an honest "departed from a degenerate baseline"
//     sentinel.
//
// The caller owns the threshold / verdict (decision-neutral).
func (w *Welford) ZScore(x float64) float64 {
	if w.n == 0 {
		return 0
	}
	sd := w.StdDev()
	diff := x - w.mean
	if sd == 0 {
		if diff == 0 {
			return 0
		}
		return math.Copysign(math.Inf(1), diff)
	}
	return diff / sd
}

// Merge returns the Chan-Golub-LeVeque (1979) parallel combine of w and o, as if
// the two streams had been concatenated and fed to a single Welford. It does not
// mutate either receiver. Merge is the method form of the package-level Merge.
func (w Welford) Merge(o Welford) Welford {
	return Merge(w, o)
}

// Merge combines two independently-accumulated Welford states using the
// Chan-Golub-LeVeque (1979) parallel algorithm:
//
//	n     = nA + nB
//	delta = meanB - meanA
//	mean  = meanA + delta * nB / n
//	M2    = M2_A + M2_B + delta^2 * nA * nB / n
//
// The result equals (to floating-point tolerance) a single Welford streamed over
// the concatenation of the two underlying series. An empty operand is the
// identity (Merge with a zero-count state returns the other unchanged).
func Merge(a, b Welford) Welford {
	if a.n == 0 {
		return b
	}
	if b.n == 0 {
		return a
	}
	n := a.n + b.n
	nF := float64(n)
	nAF := float64(a.n)
	nBF := float64(b.n)
	delta := b.mean - a.mean
	mean := a.mean + delta*nBF/nF
	m2 := a.m2 + b.m2 + delta*delta*nAF*nBF/nF
	return Welford{n: n, mean: mean, m2: m2}
}

// WelfordVec is the per-dimension vector form of Welford over a FIXED dimension
// D, maintaining an independent running mean and M2 for each coordinate (a
// diagonal-covariance accumulator). It generalises the mel-band array form used
// by audio.BaselineStats / echo-chamber's spectral BaselineStats, where each
// dimension is tracked independently.
//
// Construct with NewWelfordVec(dim); the zero value is NOT usable (it has no
// backing storage). Update / Mean / Variance panic on a dimension mismatch,
// which is a programming error (mirroring audio.Fingerprint).
type WelfordVec struct {
	n    int
	mean []float64
	m2   []float64
}

// NewWelfordVec allocates an empty D-dimensional accumulator. D must be >= 1.
func NewWelfordVec(dim int) *WelfordVec {
	if dim < 1 {
		panic("moments.NewWelfordVec: dim must be >= 1")
	}
	return &WelfordVec{
		n:    0,
		mean: make([]float64, dim),
		m2:   make([]float64, dim),
	}
}

// Dim returns the fixed dimension D of the accumulator.
func (w *WelfordVec) Dim() int { return len(w.mean) }

// Count returns the number of vector observations folded in so far.
func (w *WelfordVec) Count() int { return w.n }

// Update folds one D-dimensional observation x into w in-place, applying the
// scalar Welford recurrence independently per coordinate. Panics if
// len(x) != Dim().
func (w *WelfordVec) Update(x []float64) {
	d := len(w.mean)
	if len(x) != d {
		panic("moments.WelfordVec.Update: x dimension mismatch")
	}
	w.n++
	nF := float64(w.n)
	for i := 0; i < d; i++ {
		delta := x[i] - w.mean[i]
		w.mean[i] += delta / nF
		delta2 := x[i] - w.mean[i]
		w.m2[i] += delta * delta2
	}
}

// Mean returns a freshly-allocated copy of the per-dimension running mean
// (length Dim()). For an empty accumulator every coordinate is 0.
func (w *WelfordVec) Mean() []float64 {
	out := make([]float64, len(w.mean))
	copy(out, w.mean)
	return out
}

// Variance returns a freshly-allocated per-dimension unbiased SAMPLE variance
// (divisor n-1, length Dim()). Every coordinate is 0 when n < 2, matching the
// scalar Welford and audio.BaselineStats conventions.
func (w *WelfordVec) Variance() []float64 {
	d := len(w.mean)
	out := make([]float64, d)
	if w.n < 2 {
		return out
	}
	denom := float64(w.n - 1)
	for i := 0; i < d; i++ {
		out[i] = w.m2[i] / denom
	}
	return out
}

// PopVariance returns a freshly-allocated per-dimension POPULATION variance
// (divisor n, length Dim()). Every coordinate is 0 when n < 1.
func (w *WelfordVec) PopVariance() []float64 {
	d := len(w.mean)
	out := make([]float64, d)
	if w.n < 1 {
		return out
	}
	denom := float64(w.n)
	for i := 0; i < d; i++ {
		out[i] = w.m2[i] / denom
	}
	return out
}

// StdDev returns a freshly-allocated per-dimension unbiased SAMPLE standard
// deviation (length Dim()); every coordinate is 0 when n < 2.
func (w *WelfordVec) StdDev() []float64 {
	v := w.Variance()
	for i := range v {
		v[i] = math.Sqrt(v[i])
	}
	return v
}

package audio

import "math"

// Fingerprint stores a Welford-convergent mean + variance for an
// individual entity's feature vector distribution.
//
// The fingerprint is the load-bearing primitive for individual-entity
// identification across the audio cohort:
//   - pigeonhole: per-bird MFCC distribution (this song thrush vs that one)
//   - howler: per-pet vocalisation distribution (Rulla vs Rupert)
//   - dipstick: per-machine spectral signature (your Bosch vs population mean)
//   - folio (lateral): repeat-guest profile on textual/behavioural features
//
// Welford's online algorithm (Welford 1962; Knuth TAOCP vol 2 §4.2.2)
// is numerically stable and produces the same result regardless of
// observation order — an essential property for the forge ASSESS step's
// "score-before-update" rule (R-pattern guidance).
//
// Fields are exported for caller flexibility (zero-alloc updates, custom
// serialisation, persistence across app restarts). Callers must NOT
// modify N alone without the corresponding Mean/M2 update — use
// UpdateFingerprint instead.
type Fingerprint struct {
	N    int       // number of observations
	Mean []float64 // running mean (length D)
	M2   []float64 // running sum of squared deviations from mean (length D)
}

// NewFingerprint allocates a zero fingerprint for a D-dimensional feature
// vector. D must be >= 1.
func NewFingerprint(D int) Fingerprint {
	if D < 1 {
		panic("audio.NewFingerprint: D must be >= 1")
	}
	return Fingerprint{
		N:    0,
		Mean: make([]float64, D),
		M2:   make([]float64, D),
	}
}

// UpdateFingerprint applies one Welford step to the fingerprint with
// feature vector x. Mutates fp in-place. Zero allocation.
//
// Algorithm (Welford 1962):
//
//	n      = n + 1
//	delta  = x - mean
//	mean   = mean + delta / n
//	delta2 = x - mean         // after mean update
//	M2     = M2 + delta * delta2
//
// Numerical stability: stable for any reasonable observation count;
// the recompute-after-update formulation (delta2) is the numerically-
// stable variant from Knuth TAOCP vol 2 §4.2.2 attributed to Welford.
//
// Panics if len(x) != len(fp.Mean).
func UpdateFingerprint(fp *Fingerprint, x []float64) {
	D := len(fp.Mean)
	if len(x) != D {
		panic("audio.UpdateFingerprint: x dimension mismatch")
	}
	if len(fp.M2) != D {
		panic("audio.UpdateFingerprint: fp.M2 length must match fp.Mean")
	}

	fp.N++
	nF := float64(fp.N)
	for i := 0; i < D; i++ {
		delta := x[i] - fp.Mean[i]
		fp.Mean[i] += delta / nF
		delta2 := x[i] - fp.Mean[i]
		fp.M2[i] += delta * delta2
	}
}

// FingerprintVariance writes the unbiased sample variance per dimension
// into out (length >= D). Returns zeros if fp.N < 2 (variance undefined
// for a single observation).
//
// Formula: var[i] = M2[i] / (N - 1)
//
// Zero allocation. Panics if out too short.
func FingerprintVariance(fp *Fingerprint, out []float64) {
	D := len(fp.Mean)
	if len(out) < D {
		panic("audio.FingerprintVariance: out must have length >= D")
	}
	if fp.N < 2 {
		for i := 0; i < D; i++ {
			out[i] = 0.0
		}
		return
	}
	denom := float64(fp.N - 1)
	for i := 0; i < D; i++ {
		out[i] = fp.M2[i] / denom
	}
}

// FingerprintMahalanobis returns the diagonal Mahalanobis squared
// distance between x and fp.Mean, using fp's per-dimension variance
// estimate. Epsilon is added to each variance to stabilise against
// zero-variance dimensions (e.g. early in convergence when N == 1).
//
// Formula:
//
//	d^2 = sum_i (x[i] - mean[i])^2 / (var[i] + epsilon)
//
// Diagonal Mahalanobis is appropriate when MFCC dimensions are
// approximately decorrelated by the DCT in MFCC.go — the off-diagonal
// covariance is typically small for speech / bioacoustic / mechanical
// signals after DCT-II.
//
// For full Mahalanobis (with off-diagonal covariance) callers should
// composes their own routine; the diagonal form is sufficient for
// individual-entity identification at the precision needed by the forge
// ASSESS step.
//
// Returns +Inf if fp.N == 0 (no observations to compare against).
//
// Reference: Mahalanobis, P.C. (1936) "On the generalised distance
// in statistics", Proceedings of the National Institute of Sciences
// of India, vol 2, pp 49-55. Diagonal form is standard in speech-rec
// (HTK Book §5.5.1) and bioacoustic individual-ID literature.
//
// Zero allocation.
func FingerprintMahalanobis(fp *Fingerprint, x []float64, epsilon float64) float64 {
	D := len(fp.Mean)
	if len(x) != D {
		panic("audio.FingerprintMahalanobis: x dimension mismatch")
	}
	if fp.N == 0 {
		return math.Inf(1)
	}
	if epsilon < 0 {
		panic("audio.FingerprintMahalanobis: epsilon must be >= 0")
	}

	d2 := 0.0
	if fp.N < 2 {
		// Variance undefined — fall back to Euclidean / epsilon-only.
		for i := 0; i < D; i++ {
			diff := x[i] - fp.Mean[i]
			d2 += (diff * diff) / (epsilon + 1e-12)
		}
		return d2
	}

	denomScale := float64(fp.N - 1)
	for i := 0; i < D; i++ {
		diff := x[i] - fp.Mean[i]
		varI := fp.M2[i] / denomScale
		d2 += (diff * diff) / (varI + epsilon)
	}
	return d2
}

// BestMatch finds the fingerprint with smallest Mahalanobis squared
// distance to x. Returns (bestIdx, bestDist). Returns (-1, +Inf) if
// fps is empty or all fingerprints have N == 0.
//
// Threshold-based promotion (e.g. "is this a known individual or a
// new one?") is the caller's responsibility. The forge ASSESS step
// typically applies a chi-squared distribution test on bestDist
// against the (D, alpha) tail to decide.
//
// Zero allocation.
//
// Consumed by: pigeonhole identify path, howler multi-pet identification,
// dipstick per-machine match.
func BestMatch(fps []Fingerprint, x []float64, epsilon float64) (int, float64) {
	bestIdx := -1
	bestDist := math.Inf(1)
	for i := 0; i < len(fps); i++ {
		if fps[i].N == 0 {
			continue
		}
		d := FingerprintMahalanobis(&fps[i], x, epsilon)
		if d < bestDist {
			bestDist = d
			bestIdx = i
		}
	}
	return bestIdx, bestDist
}

// MergeFingerprints combines two fingerprints (a, b) into out using
// Chan-Golub-LeVeque parallel Welford merging. Mutates out in-place.
//
// Algorithm (Chan, Golub, LeVeque 1979):
//
//	N      = nA + nB
//	delta  = meanB - meanA
//	mean   = meanA + delta * nB / N
//	M2     = M2_A + M2_B + delta^2 * nA * nB / N
//
// out, a, b must all have the same dimension D. out may alias a or b
// safely (the algorithm is in-place safe).
//
// Useful for: merging per-session per-pet fingerprints into a long-term
// profile; merging Folio per-month converged guest fingerprints across
// multi-property aggregations.
//
// Zero allocation. Panics on dimension mismatch.
func MergeFingerprints(a, b, out *Fingerprint) {
	D := len(a.Mean)
	if len(b.Mean) != D || len(out.Mean) != D {
		panic("audio.MergeFingerprints: dimension mismatch")
	}
	if len(a.M2) != D || len(b.M2) != D || len(out.M2) != D {
		panic("audio.MergeFingerprints: M2 length must match Mean")
	}

	if a.N == 0 && b.N == 0 {
		out.N = 0
		for i := 0; i < D; i++ {
			out.Mean[i] = 0
			out.M2[i] = 0
		}
		return
	}
	if a.N == 0 {
		out.N = b.N
		for i := 0; i < D; i++ {
			out.Mean[i] = b.Mean[i]
			out.M2[i] = b.M2[i]
		}
		return
	}
	if b.N == 0 {
		out.N = a.N
		for i := 0; i < D; i++ {
			out.Mean[i] = a.Mean[i]
			out.M2[i] = a.M2[i]
		}
		return
	}

	N := a.N + b.N
	nF := float64(N)
	nAF := float64(a.N)
	nBF := float64(b.N)
	prodOverN := nAF * nBF / nF

	for i := 0; i < D; i++ {
		delta := b.Mean[i] - a.Mean[i]
		newMean := a.Mean[i] + delta*nBF/nF
		newM2 := a.M2[i] + b.M2[i] + delta*delta*prodOverN
		out.Mean[i] = newMean
		out.M2[i] = newM2
	}
	out.N = N
}

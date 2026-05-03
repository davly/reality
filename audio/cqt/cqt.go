package cqt

import (
	"errors"
	"math"
	"math/cmplx"
)

// QualityFactor returns the constant Q-factor for a given number of bins
// per octave.
//
//	Q = 1 / (2^(1/binsPerOctave) - 1)
//
// For B = 12 (semitone resolution): Q ≈ 17.
// For B = 24 (quarter-tone):        Q ≈ 34.
// For B = 36 (third-of-semitone):   Q ≈ 51.
//
// Panics if binsPerOctave <= 0.
func QualityFactor(binsPerOctave int) float64 {
	if binsPerOctave <= 0 {
		panic("cqt: binsPerOctave must be > 0")
	}
	return 1.0 / (math.Pow(2.0, 1.0/float64(binsPerOctave)) - 1.0)
}

// BinFrequency returns the centre frequency of CQT bin k for the given
// minimum frequency and binsPerOctave.
//
//	f_k = fMin * 2^(k / binsPerOctave)
//
// For piano-A0 fMin = 27.5 Hz with B = 12: bin 0 = A0, bin 12 = A1, etc.
//
// Panics if fMin <= 0 or binsPerOctave <= 0.
func BinFrequency(k, binsPerOctave int, fMin float64) float64 {
	if fMin <= 0 {
		panic("cqt: fMin must be > 0")
	}
	if binsPerOctave <= 0 {
		panic("cqt: binsPerOctave must be > 0")
	}
	return fMin * math.Pow(2.0, float64(k)/float64(binsPerOctave))
}

// BinFrequencies fills out with the centre frequency of every CQT bin
// across the requested octave range.
//
// out must have length >= binsPerOctave * octaves. Panics if it does not.
func BinFrequencies(binsPerOctave, octaves int, fMin float64, out []float64) {
	K := binsPerOctave * octaves
	if len(out) < K {
		panic("cqt: out must have length >= binsPerOctave * octaves")
	}
	for k := 0; k < K; k++ {
		out[k] = BinFrequency(k, binsPerOctave, fMin)
	}
}

// WindowLength returns the per-bin window length (samples) at sample
// rate sr for bin frequency f_k. Implements
//
//	N_k = round(Q * sr / f_k)
//
// Panics if sr <= 0 or f <= 0.
func WindowLength(q, sr, f float64) int {
	if sr <= 0 {
		panic("cqt: sample rate must be > 0")
	}
	if f <= 0 {
		panic("cqt: bin frequency must be > 0")
	}
	return int(math.Round(q * sr / f))
}

// CQT computes the Constant-Q Transform of x at sample rate sr, writing
// one complex value per bin into out.
//
// Parameters:
//   - x:               input samples (real-valued audio).
//   - sr:              sample rate in Hz (must be > 0).
//   - fMin:            centre frequency of the lowest bin (must be > 0).
//   - binsPerOctave:   bins per octave; B = 12 for semitone resolution.
//   - octaves:         number of octaves to span starting at fMin.
//   - out:             complex output, len(out) must equal
//                      binsPerOctave * octaves.
//
// Errors:
//   - ErrInvalidParams when any of sr, fMin, binsPerOctave, octaves are
//     out of range.
//   - ErrSampleRateTooLow when the highest bin's Nyquist condition is
//     violated (f_top >= sr/2).
//   - ErrInputTooShort when len(x) < N_k for the lowest bin (the
//     longest-window bin has no support in the input).
//   - ErrOutputSize when len(out) != binsPerOctave * octaves.
//
// Determinism: pure function. Same (x, params) → same out, byte-equal,
// regardless of host locale, scheduling, or process state. No goroutines.
//
// Allocation: one transient atom buffer per bin. Caller-supplied out is
// the only persistent allocation.
func CQT(x []float64, sr, fMin float64, binsPerOctave, octaves int, out []complex128) error {
	if sr <= 0 || fMin <= 0 || binsPerOctave <= 0 || octaves <= 0 {
		return ErrInvalidParams
	}
	K := binsPerOctave * octaves
	if len(out) != K {
		return ErrOutputSize
	}

	q := QualityFactor(binsPerOctave)
	fTop := BinFrequency(K-1, binsPerOctave, fMin)
	if fTop >= sr/2 {
		return ErrSampleRateTooLow
	}

	// The longest window (lowest bin) bounds the input requirement.
	nMax := WindowLength(q, sr, fMin)
	if len(x) < nMax {
		return ErrInputTooShort
	}

	for k := 0; k < K; k++ {
		f := BinFrequency(k, binsPerOctave, fMin)
		nk := WindowLength(q, sr, f)
		if nk <= 0 {
			out[k] = 0
			continue
		}
		// Cap to input length defensively (shouldn't trigger after the
		// nMax guard, but keeps the inner loop safe under future edits).
		if nk > len(x) {
			nk = len(x)
		}

		// Sum over n in [0, nk) of x[n] * w[n] * exp(-2πi Q n / nk),
		// scaled by 1/nk. Using cmplx primitives keeps the algorithm
		// readable; the inner loop is the hot path.
		sum := complex(0, 0)
		invNk := 1.0 / float64(nk)
		denom := float64(nk - 1)
		// Avoid division-by-zero in the Hann window when nk == 1.
		if denom <= 0 {
			denom = 1
		}
		angleStep := -2.0 * math.Pi * q * invNk
		for n := 0; n < nk; n++ {
			// Hann window: w(n) = 0.5 * (1 - cos(2π n / (N-1)))
			w := 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(n)/denom))
			angle := angleStep * float64(n)
			atom := complex(w*math.Cos(angle), w*math.Sin(angle))
			sum += complex(x[n], 0) * atom
		}
		out[k] = sum * complex(invNk, 0)
	}
	return nil
}

// Magnitude fills mag with the absolute value of each complex CQT bin.
// mag must be at least as long as cqtOut. Panics if it isn't.
func Magnitude(cqtOut []complex128, mag []float64) {
	if len(mag) < len(cqtOut) {
		panic("cqt: mag must have length >= len(cqtOut)")
	}
	for i, v := range cqtOut {
		mag[i] = cmplx.Abs(v)
	}
}

// PeakBin returns the index of the bin with maximum magnitude. Returns
// -1 if cqtOut is empty.
func PeakBin(cqtOut []complex128) int {
	if len(cqtOut) == 0 {
		return -1
	}
	peak := 0
	peakMag := cmplx.Abs(cqtOut[0])
	for i := 1; i < len(cqtOut); i++ {
		m := cmplx.Abs(cqtOut[i])
		if m > peakMag {
			peakMag = m
			peak = i
		}
	}
	return peak
}

// Sentinel errors. Callers should use errors.Is to branch.
var (
	// ErrInvalidParams indicates a non-positive sr / fMin / binsPerOctave / octaves.
	ErrInvalidParams = errors.New("cqt: invalid parameters")

	// ErrSampleRateTooLow indicates the top requested bin's centre
	// frequency is at or above Nyquist (sr/2).
	ErrSampleRateTooLow = errors.New("cqt: sample rate too low for top bin")

	// ErrInputTooShort indicates the input does not span the longest
	// per-bin window (the lowest-frequency bin's window).
	ErrInputTooShort = errors.New("cqt: input shorter than longest window")

	// ErrOutputSize indicates len(out) != binsPerOctave * octaves.
	ErrOutputSize = errors.New("cqt: output slice has wrong length")
)

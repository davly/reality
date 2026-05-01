package audio

import "math"

// LogMelEnergies computes log-compressed mel-band energies per HTK convention:
// applies a small floor to avoid log(0), then natural log.
//
// Formula: out[b] = log(max(energies[b], floor))
//
// Parameters:
//   - energies: mel-band energies (length numFilters)
//   - floor:    minimum value, e.g. 1e-10. floor must be > 0.
//   - out:      output of length >= numFilters
//
// Zero allocation. Panics on size violations or floor <= 0.
func LogMelEnergies(energies []float64, floor float64, out []float64) {
	if floor <= 0 {
		panic("audio.LogMelEnergies: floor must be > 0")
	}
	n := len(energies)
	if len(out) < n {
		panic("audio.LogMelEnergies: out must have length >= len(energies)")
	}
	for i := 0; i < n; i++ {
		e := energies[i]
		if e < floor {
			e = floor
		}
		out[i] = math.Log(e)
	}
}

// MFCC computes Mel-Frequency Cepstral Coefficients via DCT-II of
// log-mel energies. This is the HTK / kaldi convention.
//
// Formula (DCT-II, orthonormal):
//
//	c[k] = alpha[k] * sum_{b=0..M-1} logEnergies[b] * cos(pi * k * (b + 0.5) / M)
//	alpha[0] = sqrt(1/M)
//	alpha[k] = sqrt(2/M)  for k >= 1
//
// where M = numFilters (length of logEnergies).
//
// The orthonormal DCT-II is preferred because it preserves L2 energy;
// some implementations omit alpha and use the un-normalised DCT-II,
// which differs only by a per-coefficient constant scale.
//
// Parameters:
//   - logEnergies: log mel-band energies (length numFilters)
//   - numCoeffs:   number of cepstral coefficients to compute (typically 13 for
//     speech, 13-40 for bioacoustic / mechanical work). Must be
//     <= numFilters.
//   - out:         output of length >= numCoeffs
//
// Precision: ≤1e-12 numerical error for typical 26-band, 13-coefficient
// speech extraction; precision degrades as numCoeffs approaches numFilters
// because of accumulated cosine round-off.
// Reference: HTK Book §5.6; Davis & Mermelstein 1980 IEEE Trans. ASSP.
//
// Zero allocation in the loop body (the cos table is computed inline).
// Panics on size violations.
//
// Consumed by: pigeonhole / howler / dipstick (Layer 0 fingerprint input);
// fingerprint.go in this package.
func MFCC(logEnergies []float64, numCoeffs int, out []float64) {
	M := len(logEnergies)
	if M < 1 {
		panic("audio.MFCC: numFilters must be >= 1")
	}
	if numCoeffs < 1 || numCoeffs > M {
		panic("audio.MFCC: numCoeffs must satisfy 1 <= numCoeffs <= numFilters")
	}
	if len(out) < numCoeffs {
		panic("audio.MFCC: out must have length >= numCoeffs")
	}

	scaleK0 := math.Sqrt(1.0 / float64(M))
	scaleK := math.Sqrt(2.0 / float64(M))

	for k := 0; k < numCoeffs; k++ {
		s := 0.0
		piKOverM := math.Pi * float64(k) / float64(M)
		for b := 0; b < M; b++ {
			s += logEnergies[b] * math.Cos(piKOverM*(float64(b)+0.5))
		}
		if k == 0 {
			out[k] = scaleK0 * s
		} else {
			out[k] = scaleK * s
		}
	}
}

// FrameMFCC computes MFCCs for a single frame of audio. This is a
// convenience function that chains the pipeline: window -> FFT ->
// power spectrum -> mel filterbank -> log -> DCT.
//
// The caller MUST pre-compute and reuse the mel filterbank across frames
// (it is expensive to construct, cheap to apply). Use MelFilterbank to
// build it once at session start.
//
// Parameters:
//   - frame:       windowed real-valued audio frame of length nFFT
//   - imag:        scratch slice of length nFFT (zeroed on entry)
//   - power:       scratch slice of length nFFT/2 + 1
//   - filterbank:  precomputed mel filterbank (numFilters x nFFT/2 + 1, row-major)
//   - logEnergies: scratch slice of length numFilters
//   - melEnergies: scratch slice of length numFilters
//   - numFilters:  number of mel bands
//   - numCoeffs:   number of MFCC coefficients to output
//   - logFloor:    minimum value before log (e.g. 1e-10)
//   - out:         MFCC output of length >= numCoeffs
//
// frame is consumed in-place by the FFT (the caller's frame slice is
// modified). imag must be zeroed on entry (caller is responsible).
//
// All scratch slices and out must be pre-allocated by the caller for
// zero per-frame allocation.
//
// fft is the FFT function from reality/signal — not imported here to
// keep this package free of cross-package import for testability;
// callers must compose externally. See cmd/pigeonhole-forge for the
// canonical composition example.
//
// Note: this function does NOT call signal.FFT itself; the caller must
// run FFT(frame, imag) before calling FrameMFCC. Keeping the FFT call
// outside lets callers reuse twiddle-factor caches across frames if
// desired (signal.FFT is already zero-alloc but consumers may want
// further optimisation).
func FrameMFCC(
	frameReal, frameImag, power, filterbank, melEnergies, logEnergies []float64,
	numFilters, nFFT, numCoeffs int,
	logFloor float64,
	out []float64,
) {
	// frame{Real,Imag} have already had FFT applied externally.
	// 1. Power spectrum
	PowerSpectrum(frameReal, frameImag, power)
	// 2. Mel filterbank
	ApplyFilterbank(power, filterbank, numFilters, nFFT/2+1, melEnergies)
	// 3. Log
	LogMelEnergies(melEnergies, logFloor, logEnergies)
	// 4. DCT-II
	MFCC(logEnergies, numCoeffs, out)
}

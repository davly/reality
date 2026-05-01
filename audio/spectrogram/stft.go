package spectrogram

import (
	"github.com/davly/reality/signal"
)

// Compute computes the Short-Time Fourier Transform of a real-valued
// audio signal. The STFT slices the signal into overlapping frames of
// size frameSize, advances by hopSize samples between frames, applies
// the supplied analysis window, and FFTs each frame.
//
// Algorithm (Allen & Rabiner 1977):
//
//	For t = 0, hopSize, 2·hopSize, ...:
//	  frame = samples[t : t + frameSize]
//	  windowed = frame ⊙ window
//	  X[t/hopSize, k] = FFT(windowed)[k]   for k = 0..frameSize-1
//
// Frames extending past len(samples) are zero-padded.
//
// Parameters:
//   - samples:   real-valued audio buffer
//   - frameSize: FFT length per frame (must be a power of 2 in
//     practice — signal.FFT requirement)
//   - hopSize:   number of samples advanced between consecutive frames.
//     Typical values: hopSize = frameSize/4 (75% overlap, common for
//     fine analysis) or hopSize = frameSize/2 (50% overlap, lighter).
//   - window:    pre-computed analysis window of length frameSize
//     (e.g. signal.HannWindow)
//
// Returns: T × frameSize complex matrix where T = ceil(len(samples) /
// hopSize). Each row is one frame's full complex FFT (so callers
// interested in magnitude only should use Magnitude with the half-
// spectrum slice [0 : frameSize/2 + 1]).
//
// Valid range:
//   - len(samples) >= 1
//   - frameSize >= 2 (typically 256, 512, 1024, 2048, 4096)
//   - 1 <= hopSize <= frameSize
//   - len(window) == frameSize
//
// Precision: ≤1e-9 numerical error per bin (signal.FFT contract).
// Allocation: returns newly-allocated [][]complex128. Per-frame FFT
// scratch is recycled across frames (single allocation pair, not
// per-frame).
//
// Reference: Allen, J. B. & Rabiner, L. R. (1977). "A unified approach
// to short-time Fourier analysis and synthesis." Proc. IEEE 65(11),
// 1558-1564.
//
// Consumed by: spectrogram.Magnitude, spectrogram.MelSpectrogram,
// reality/audio/separation.NMF (spectrogram input).
func Compute(samples []float64, frameSize, hopSize int, window []float64) [][]complex128 {
	if frameSize < 2 {
		panic("spectrogram.Compute: frameSize must be >= 2")
	}
	if hopSize < 1 || hopSize > frameSize {
		panic("spectrogram.Compute: hopSize must satisfy 1 <= hopSize <= frameSize")
	}
	if len(window) != frameSize {
		panic("spectrogram.Compute: window length must equal frameSize")
	}
	if len(samples) < 1 {
		panic("spectrogram.Compute: samples must be non-empty")
	}

	// Number of frames: ceil(len(samples) / hopSize). The final frame
	// may have fewer than frameSize "real" samples and is zero-padded.
	numFrames := (len(samples) + hopSize - 1) / hopSize
	if numFrames < 1 {
		numFrames = 1
	}

	// Recycled FFT scratch.
	frameReal := make([]float64, frameSize)
	frameImag := make([]float64, frameSize)

	out := make([][]complex128, numFrames)
	for t := 0; t < numFrames; t++ {
		start := t * hopSize
		// Copy + window + zero-pad.
		for i := 0; i < frameSize; i++ {
			idx := start + i
			if idx < len(samples) {
				frameReal[i] = samples[idx] * window[i]
			} else {
				frameReal[i] = 0
			}
			frameImag[i] = 0
		}
		signal.FFT(frameReal, frameImag)

		row := make([]complex128, frameSize)
		for k := 0; k < frameSize; k++ {
			row[k] = complex(frameReal[k], frameImag[k])
		}
		out[t] = row
	}
	return out
}

// Inverse reconstructs samples from an STFT using overlap-add (OLA)
// resynthesis with the supplied window. The output is the time-domain
// signal corresponding to the input STFT.
//
// Algorithm (Griffin & Lim 1984 §III, OLA reconstruction):
//
//	for t = 0..T-1:
//	  frame = IFFT(stft[t]).real ⊙ window
//	  samples[t·hop : t·hop+frameSize] += frame
//	  windowSum[t·hop : t·hop+frameSize] += window²
//	for n: samples[n] /= windowSum[n] + ε
//
// The window² normalisation accounts for the COLA (constant-overlap-
// add) constraint: with proper hop / window choice (hop = frameSize/4
// for Hann), the windowSum becomes constant and division is uniform.
// We divide bin-by-bin for safety against off-COLA hops.
//
// Parameters:
//   - stft:      T × frameSize complex matrix
//   - frameSize: must equal len(stft[0])
//   - hopSize:   must match the analysis hop
//   - window:    must match the analysis window
//
// Returns: time-domain samples of length (T-1)·hopSize + frameSize.
//
// Valid range: T >= 1; frameSize a power of 2; len(window) ==
// frameSize.
// Precision: round-trip accuracy is ≤1e-7 for COLA-compliant
// hop/window pairs (Hann at hop=frameSize/4, frameSize/2). Off-COLA
// pairs may show larger reconstruction error in edge regions.
// Panics on shape violations.
//
// Reference: Griffin, D. W. & Lim, J. S. (1984). "Signal estimation
// from modified short-time Fourier transform." IEEE Trans. ASSP
// 32(2), 236-243.
func Inverse(stft [][]complex128, frameSize, hopSize int, window []float64) []float64 {
	T := len(stft)
	if T < 1 {
		panic("spectrogram.Inverse: stft must have at least 1 frame")
	}
	if len(stft[0]) != frameSize {
		panic("spectrogram.Inverse: stft[0] length must equal frameSize")
	}
	if len(window) != frameSize {
		panic("spectrogram.Inverse: window length must equal frameSize")
	}

	outLen := (T-1)*hopSize + frameSize
	samples := make([]float64, outLen)
	windowSum := make([]float64, outLen)

	frameReal := make([]float64, frameSize)
	frameImag := make([]float64, frameSize)

	for t := 0; t < T; t++ {
		// Inverse FFT of stft[t].
		for k := 0; k < frameSize; k++ {
			frameReal[k] = real(stft[t][k])
			frameImag[k] = imag(stft[t][k])
		}
		signal.IFFT(frameReal, frameImag)

		start := t * hopSize
		for i := 0; i < frameSize; i++ {
			samples[start+i] += frameReal[i] * window[i]
			windowSum[start+i] += window[i] * window[i]
		}
	}
	const eps = 1e-12
	for n := 0; n < outLen; n++ {
		samples[n] /= windowSum[n] + eps
	}
	return samples
}

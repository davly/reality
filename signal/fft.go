// Package signal provides signal processing primitives: Fast Fourier Transform,
// digital filters, and window functions. All functions are deterministic, use
// only the Go standard library, and target zero heap allocations in hot paths.
//
// FFT/IFFT operate in-place on pre-allocated real and imaginary slices.
// Filter and window functions write into caller-provided output slices.
//
// Consumed by: Pistachio (audio), RubberDuck (spectral analysis),
// Oracle (time series), Sentinel (filtering).
package signal

import "math"

// isPow2 reports whether n is a positive power of two.
func isPow2(n int) bool {
	return n > 0 && n&(n-1) == 0
}

// bitReverse performs in-place bit-reversal permutation on real and imag slices.
// n must be a power of 2. Zero allocations — uses index arithmetic only.
func bitReverse(real, imag []float64, n int) {
	j := 0
	for i := 1; i < n; i++ {
		bit := n >> 1
		for j&bit != 0 {
			j ^= bit
			bit >>= 1
		}
		j ^= bit
		if i < j {
			real[i], real[j] = real[j], real[i]
			imag[i], imag[j] = imag[j], imag[i]
		}
	}
}

// FFT computes the discrete Fourier transform using the Cooley-Tukey radix-2
// algorithm. The transform is computed in-place: real and imag are modified
// directly with zero allocation.
//
// Both slices must have the same length N, and N must be a power of 2.
// Panics if N is not a power of 2 or if slice lengths differ.
//
// Definition: X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*n*k / N)
// Precision: 1e-9 for 1024-point, scales with N*log(N) rounding.
//
// Consumers: Pistachio (audio FFT), RubberDuck (spectral analysis),
// Oracle (time-series frequency decomposition).
func FFT(real, imag []float64) {
	n := len(real)
	if len(imag) != n {
		panic("signal.FFT: real and imag slices must have equal length")
	}
	if !isPow2(n) {
		panic("signal.FFT: length must be a power of 2")
	}
	if n <= 1 {
		return
	}

	bitReverse(real, imag, n)

	for size := 2; size <= n; size <<= 1 {
		halfSize := size >> 1
		angleStep := -2.0 * math.Pi / float64(size)
		wReal := math.Cos(angleStep)
		wImag := math.Sin(angleStep)

		for start := 0; start < n; start += size {
			curReal := 1.0
			curImag := 0.0
			for k := 0; k < halfSize; k++ {
				evenIdx := start + k
				oddIdx := start + k + halfSize

				tReal := curReal*real[oddIdx] - curImag*imag[oddIdx]
				tImag := curReal*imag[oddIdx] + curImag*real[oddIdx]

				real[oddIdx] = real[evenIdx] - tReal
				imag[oddIdx] = imag[evenIdx] - tImag
				real[evenIdx] += tReal
				imag[evenIdx] += tImag

				newReal := curReal*wReal - curImag*wImag
				newImag := curReal*wImag + curImag*wReal
				curReal = newReal
				curImag = newImag
			}
		}
	}
}

// IFFT computes the inverse discrete Fourier transform in-place.
// Uses the conjugate method: conjugate, FFT, conjugate, scale by 1/N.
//
// Both slices must have the same length N, and N must be a power of 2.
// Panics if N is not a power of 2 or if slice lengths differ.
//
// Definition: x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(2*pi*i*n*k / N)
// Precision: matches FFT (1e-9 for 1024-point).
func IFFT(real, imag []float64) {
	n := len(real)
	if len(imag) != n {
		panic("signal.IFFT: real and imag slices must have equal length")
	}
	if !isPow2(n) {
		panic("signal.IFFT: length must be a power of 2")
	}
	if n <= 1 {
		return
	}

	// Conjugate
	for i := range imag {
		imag[i] = -imag[i]
	}

	// Forward FFT
	FFT(real, imag)

	// Conjugate and scale by 1/N
	scale := 1.0 / float64(n)
	for i := range real {
		real[i] *= scale
		imag[i] = -imag[i] * scale
	}
}

// PowerSpectrum computes |FFT(x)|^2 for each frequency bin and writes the
// result into out. This function first computes the FFT in-place (modifying
// real and imag), then writes N/2+1 power values into out.
//
// real and imag must have the same length N (power of 2).
// out must have length N/2+1.
// Panics on invalid dimensions.
//
// Definition: P[k] = real[k]^2 + imag[k]^2 for k in [0, N/2]
//
// Consumers: RubberDuck (spectral power), Sentinel (frequency monitoring).
func PowerSpectrum(real, imag []float64, out []float64) {
	n := len(real)
	if len(imag) != n {
		panic("signal.PowerSpectrum: real and imag slices must have equal length")
	}
	if !isPow2(n) {
		panic("signal.PowerSpectrum: length must be a power of 2")
	}
	outLen := n/2 + 1
	if len(out) < outLen {
		panic("signal.PowerSpectrum: out must have length >= N/2+1")
	}

	FFT(real, imag)

	for k := 0; k < outLen; k++ {
		out[k] = real[k]*real[k] + imag[k]*imag[k]
	}
}

// FFTFrequencies computes the frequency bin centers for an N-point FFT at
// the given sample rate, writing N/2+1 values into out.
//
// Definition: freq[k] = k * sampleRate / N for k in [0, N/2]
//
// out must have length >= N/2+1. Panics if n is not a power of 2 or
// out is too short.
func FFTFrequencies(n int, sampleRate float64, out []float64) {
	if !isPow2(n) {
		panic("signal.FFTFrequencies: n must be a power of 2")
	}
	outLen := n/2 + 1
	if len(out) < outLen {
		panic("signal.FFTFrequencies: out must have length >= N/2+1")
	}

	binWidth := sampleRate / float64(n)
	for k := 0; k < outLen; k++ {
		out[k] = float64(k) * binWidth
	}
}

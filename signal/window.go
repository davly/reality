package signal

import "math"

// HannWindow computes the Hann (raised cosine) window of length n into out.
//
// Definition: w[i] = 0.5 * (1 - cos(2*pi*i / (n-1)))
// Endpoints: w[0] = w[n-1] = 0
// Symmetry: w[i] = w[n-1-i]
//
// out must have length >= n. Panics if n < 1 or out is too short.
// Zero heap allocations.
//
// Consumers: Pistachio (audio windowing), RubberDuck (spectral leakage reduction).
func HannWindow(n int, out []float64) {
	if n < 1 {
		panic("signal.HannWindow: n must be >= 1")
	}
	if len(out) < n {
		panic("signal.HannWindow: out must have length >= n")
	}

	if n == 1 {
		out[0] = 1.0
		return
	}

	scale := 2.0 * math.Pi / float64(n-1)
	for i := 0; i < n; i++ {
		out[i] = 0.5 * (1.0 - math.Cos(float64(i)*scale))
	}
}

// HammingWindow computes the Hamming window of length n into out.
//
// Definition: w[i] = 0.54 - 0.46 * cos(2*pi*i / (n-1))
// Endpoints: w[0] = w[n-1] = 0.08
// Symmetry: w[i] = w[n-1-i]
//
// out must have length >= n. Panics if n < 1 or out is too short.
// Zero heap allocations.
//
// Consumers: Pistachio (audio), Oracle (time-series spectral analysis).
func HammingWindow(n int, out []float64) {
	if n < 1 {
		panic("signal.HammingWindow: n must be >= 1")
	}
	if len(out) < n {
		panic("signal.HammingWindow: out must have length >= n")
	}

	if n == 1 {
		out[0] = 1.0
		return
	}

	scale := 2.0 * math.Pi / float64(n-1)
	for i := 0; i < n; i++ {
		out[i] = 0.54 - 0.46*math.Cos(float64(i)*scale)
	}
}

// BlackmanWindow computes the Blackman window of length n into out.
//
// Definition: w[i] = 0.42 - 0.5*cos(2*pi*i/(n-1)) + 0.08*cos(4*pi*i/(n-1))
// Endpoints: w[0] = w[n-1] = 0
// Symmetry: w[i] = w[n-1-i]
//
// The Blackman window provides better sidelobe suppression than Hann/Hamming
// at the cost of a wider main lobe.
//
// out must have length >= n. Panics if n < 1 or out is too short.
// Zero heap allocations.
//
// Consumers: RubberDuck (high-dynamic-range spectral analysis).
func BlackmanWindow(n int, out []float64) {
	if n < 1 {
		panic("signal.BlackmanWindow: n must be >= 1")
	}
	if len(out) < n {
		panic("signal.BlackmanWindow: out must have length >= n")
	}

	if n == 1 {
		out[0] = 1.0
		return
	}

	scale1 := 2.0 * math.Pi / float64(n-1)
	scale2 := 4.0 * math.Pi / float64(n-1)
	for i := 0; i < n; i++ {
		fi := float64(i)
		out[i] = 0.42 - 0.5*math.Cos(fi*scale1) + 0.08*math.Cos(fi*scale2)
	}
}

// ApplyWindow multiplies signal by window element-wise, writing the result
// into out. All three slices must have the same length.
//
// Definition: out[i] = signal[i] * window[i]
//
// Panics if lengths differ.
// Zero heap allocations.
func ApplyWindow(signal, window, out []float64) {
	n := len(signal)
	if len(window) != n || len(out) < n {
		panic("signal.ApplyWindow: all slices must have equal length")
	}

	for i := 0; i < n; i++ {
		out[i] = signal[i] * window[i]
	}
}

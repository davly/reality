package separation

import (
	"math/cmplx"
)

// WienerFilter applies a stationary single-channel Wiener filter to a
// noisy complex FFT spectrum given a noise spectrum estimate.
//
// Algorithm (Wiener 1949; classical "noise-suppression" interpretation):
//
//	SNR_apri[k] = max(|X[k]|² - |N[k]|², 0) / |N[k]|²
//	G[k]        = SNR_apri[k] / (1 + SNR_apri[k])
//	Ŝ[k]        = G[k] · X[k]
//
// G[k] is the Wiener gain — the MMSE-optimal real attenuation applied
// to the noisy bin under the assumption that the desired signal and
// noise are uncorrelated stationary Gaussian processes. The "a-priori
// SNR" form (Ephraim & Malah 1984) is more robust than the naive
// formulation; the form above is the decision-directed simplification.
//
// Behaviour at extremes:
//   - Bin where signal dominates (|X|² >> |N|²): G ≈ 1; bin passed through.
//   - Bin where noise dominates (|X|² ≈ |N|²): G ≈ 0; bin attenuated.
//   - Bin with no estimated noise (|N| == 0): G = 1; bin passed through
//     untouched.
//
// Parameters:
//   - in:    noisy complex FFT spectrum
//   - noise: noise complex FFT spectrum (typically magnitude-only;
//     phase is unused)
//
// Returns: a newly-allocated filtered complex spectrum of len(in). The
// caller owns the returned slice.
//
// Valid range: len(in) == len(noise).
// Precision: 1e-15 per bin (single multiplication after a max + division).
// Panics if lengths differ.
//
// Reference: Wiener, N. (1949). "Extrapolation, Interpolation, and
// Smoothing of Stationary Time Series." MIT Press; Ephraim, Y. &
// Malah, D. (1984) "Speech enhancement using a minimum mean-square
// error short-time spectral amplitude estimator" IEEE Trans. ASSP
// 32(6); Vary, P. (1985) "Noise suppression by spectral magnitude
// estimation — mechanism and theoretical limits" Signal Processing 8(4).
//
// Consumed by: howler (background-traffic suppression before vocal
// detection).
func WienerFilter(in, noise []complex128) []complex128 {
	if len(in) != len(noise) {
		panic("separation.WienerFilter: in and noise must have equal length")
	}
	out := make([]complex128, len(in))
	WienerFilterInto(in, noise, out)
	return out
}

// WienerFilterInto is the zero-alloc form of WienerFilter.
// out must have length >= len(in). Panics on size violations.
func WienerFilterInto(in, noise, out []complex128) {
	n := len(in)
	if len(noise) != n {
		panic("separation.WienerFilterInto: in and noise must have equal length")
	}
	if len(out) < n {
		panic("separation.WienerFilterInto: out must have length >= len(in)")
	}
	for k := 0; k < n; k++ {
		xMag := cmplx.Abs(in[k])
		nMag := cmplx.Abs(noise[k])
		xPower := xMag * xMag
		nPower := nMag * nMag
		var gain float64
		if nPower == 0 {
			gain = 1.0
		} else {
			snr := (xPower - nPower) / nPower
			if snr < 0 {
				snr = 0
			}
			gain = snr / (1.0 + snr)
		}
		out[k] = complex(gain, 0) * in[k]
	}
}

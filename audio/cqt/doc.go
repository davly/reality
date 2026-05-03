// Package cqt implements the Constant-Q Transform — a time-frequency
// representation with logarithmically-spaced frequency bins.
//
// Unlike the linear-frequency FFT (which gives uniform bandwidth bins
// suitable for stationary signals), CQT bins have constant Q-factor:
// the ratio of bin centre frequency to bandwidth is constant. This
// makes CQT the natural representation for audio that follows musical
// or octave-relative structure (vocals, instruments, percussion
// onsets), where a fixed-pitch interval in cents corresponds to a
// constant ratio between frequencies.
//
// # Algorithm
//
// This package implements the Brown 1991 brute-force atom-based CQT.
// For each bin k:
//
//	f_k    = fMin * 2^(k / binsPerOctave)
//	Q      = 1 / (2^(1/binsPerOctave) - 1)
//	N_k    = round(Q * sampleRate / f_k)         // window length, samples
//	w_k[n] = hann(n / N_k)                       // window
//	atom_k[n] = (1 / N_k) * w_k[n] * exp(-2πi Q n / N_k)
//	X[k]   = sum over n in [0, N_k) of x[n] * atom_k[n]
//
// Q is constant across bins (hence "constant-Q"). Window length scales
// inversely with frequency, so low-frequency bins use long windows
// (better frequency resolution at the cost of time resolution) and
// high-frequency bins use short windows.
//
// The Brown brute-force approach is O(K * N_avg) per frame where N_avg
// is the average window length. Production callers handling long
// streams should look at the FFT-domain sparse-kernel CQT (Brown &
// Puckette 1992) which is asymptotically faster but requires
// pre-computation of a kernel matrix; that variant is deferred until
// a consumer demonstrates the need.
//
// # Determinism
//
// All functions are pure / deterministic. No globals, no goroutines.
// CQT writes into caller-allocated output slices to keep allocation
// in caller control.
//
// # Consumers
//
//   - Pistachio (audio): pitch tracking + onset detection on octave-aware
//     spectral basis.
//   - foundation/reality/audio (future): note-segmentation primitives that
//     need musical-interval bin alignment.
//   - RubberDuck (future): regime-change detection on log-frequency
//     volatility profiles.
//
// # Reference
//
//   - Brown, J. C. (1991) "Calculation of a constant Q spectral transform"
//     J. Acoust. Soc. Am. 89(1), 425-434.
//   - Brown, J. C. & Puckette, M. (1992) "An efficient algorithm for
//     the calculation of a constant Q transform" J. Acoust. Soc. Am.
//     92(5), 2698-2701. (Deferred; see above.)
package cqt

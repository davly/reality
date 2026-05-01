// Package separation provides multi-source audio signal-separation
// primitives — the mathematical core of the cocktail-party problem
// (Cherry 1953): given a mixture of multiple statistically independent
// signals, recover the underlying sources.
//
// Five primitives ship in this package, each addressing a distinct
// separation regime:
//
//   - Spectral subtraction (Boll 1979): single-channel stationary-noise
//     reduction. Subtracts an estimated noise spectrum from the input.
//     Cheap; suitable when a clean ambient estimate is available.
//
//   - Wiener filter (Wiener 1949): MMSE-optimal filter for stationary
//     signals. Frequency-domain attenuation by SNR ratio.
//
//   - FastICA (Hyvärinen 1999): blind source separation by maximising
//     non-Gaussianity of projected components. The mainstream solution
//     to the cocktail-party problem when multiple microphones are
//     available (multi-channel; observations are linear mixtures).
//
//   - NMF (Lee & Seung 1999): non-negative matrix factorisation by
//     multiplicative updates. Decomposes a magnitude spectrogram into
//     a basis of repeated time-frequency patterns plus their activations
//     — well-suited to separating bird songs, drum hits, repeated
//     motifs, where the underlying components are non-negative by
//     physical construction (power spectrum cells).
//
//   - Energy-VAD: simple frame-level voice/sound activity detection
//     by short-time energy thresholding. Used as a gate before more
//     expensive separation work.
//
// All functions are deterministic, use only the Go standard library,
// and target zero allocations in hot paths via caller-provided scratch
// buffers. The numerically-iterative algorithms (FastICA, NMF) accept
// a maxIterations cap and return upon convergence or cap exhaustion.
//
// This package builds on reality/audio, reality/signal, and reality/linalg
// (transitively, via in-package gemm helpers — no import cycle). It
// follows the Reality convention: numbers in, numbers out. Every
// function documents its formula, valid range, precision, and
// reference.
//
// References:
//   - Cherry, E. C. (1953). "Some Experiments on the Recognition of
//     Speech, with One and with Two Ears." J. Acoust. Soc. Am. 25(5).
//   - Boll, S. F. (1979). "Suppression of acoustic noise in speech using
//     spectral subtraction." IEEE Trans. ASSP 27(2), 113-120.
//   - Wiener, N. (1949). "Extrapolation, Interpolation, and Smoothing
//     of Stationary Time Series." MIT Press.
//   - Hyvärinen, A. (1999). "Fast and Robust Fixed-Point Algorithms for
//     Independent Component Analysis." IEEE Trans. Neural Networks
//     10(3), 626-634.
//   - Lee, D. D. & Seung, H. S. (1999). "Learning the parts of objects
//     by non-negative matrix factorization." Nature 401, 788-791.
//   - Lee, D. D. & Seung, H. S. (2001). "Algorithms for Non-negative
//     Matrix Factorization." Advances in Neural Information Processing
//     Systems 13.
//
// Consumed by:
//   - flagships/pigeonhole (multi-bird simultaneous-singer separation;
//     "3 birds singing at once")
//   - flagships/howler (multi-pet vocalisation isolation)
//   - flagships/dipstick (machine-component-source separation in
//     multi-rotor / multi-bearing assemblies)
package separation

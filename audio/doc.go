// Package audio provides primitives for audio analysis and forge-grade
// individual-entity convergence:
//
//   - mel-scale conversion + mel filterbank (Slaney 1998)
//   - MFCC extraction (mel filterbank + DCT-II per HTK convention)
//   - Welford-based individual-entity spectral fingerprint convergence
//   - Z-score-based temporal pattern shift detection (degradation tracking)
//
// All functions are deterministic, use only the Go standard library, and
// target zero allocations in hot paths. Frame-level operations write into
// caller-provided output slices.
//
// This package builds on reality/signal (FFT, windows, filters) and follows
// the Reality convention: numbers in, numbers out. Every function documents
// its formula, valid range, precision, and reference.
//
// Cross-substrate parity: golden test vectors generated from this package
// are the canonical reference. Native ports (Kotlin Android, Swift iOS,
// SvelteKit/TypeScript) are validated against the golden vectors at
// ≤1e-9 closed-form precision and ≤1e-7 quadrature precision, matching the
// reality/prob/conformal (S55 L01) and reality/prob/copula (S55 L13)
// substrate-shipped-first / consumers-port-against-golden patterns.
//
// Consumed by:
//   - flagships/pigeonhole (UK ornithology — bird-individual identification)
//   - flagships/howler (veterinary + canine/feline behavioural science)
//   - flagships/dipstick (mechanical failure modes — machine fingerprinting)
//   - flagships/folio (lateral consumer of fingerprint primitive on
//     non-acoustic features — repeat-guest profiles)
//   - infrastructure/insights (forge-observation events for audio classifications)
//   - infrastructure/recall (per-entity converged baseline cache)
//
// Sub-packages:
//   - audio/vibration    — fundamental-frequency + harmonic-energy ratio
//     (mechanical vibration analysis; substrate of Dipstick)
//   - audio/separation   — multi-source signal separation (cocktail-party
//     problem; spectral subtraction, Wiener filter, FastICA, NMF, VAD)
//   - audio/spectrogram  — STFT, magnitude / log-magnitude / power, mel-
//     spectrogram, PNG-encoded heatmap rendering with matplotlib-
//     compatible Plasma / Magma / Viridis / Inferno colourmaps
//   - audio/onset        — onset detection (energy / spectral-flux /
//     complex-domain / SuperFlux); used for percussive event localisation
//     (Pigeonhole call boundaries, Dipstick service-event detection)
//   - audio/segmentation — call/event segmentation primitives (energy-VAD,
//     onset/offset, min-silence, merge-close, min-duration); extracts
//     individual events from longer recordings — Pigeonhole's "one call
//     at a time" workflow
//   - audio/pitch        — pitch / fundamental-frequency estimation
//     (autocorrelation, YIN, McLeod, subharmonic summation); more robust
//     than vibration's FFT peak-picker for vocal / bird / pet signals
//
// The package is the substrate-extraction outcome of the 2026-05-01 four-flagship
// cohort intake. See LimitlessGodfather/reviews/NEW_FLAGSHIPS_COHORT_2026-05-01.md
// §2 for the substrate-extraction case (M=3 multiplier).
package audio

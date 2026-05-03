// Package tempo estimates the dominant tempo (in beats per minute) of
// an audio signal from its onset-strength novelty function.
//
// # Algorithm
//
// The estimator uses the classic autocorrelation-of-novelty approach
// (Davies & Plumbley 2007 baseline; Goto 2001 origin):
//
//  1. Caller computes an onset-strength function (e.g. via
//     foundation/reality/audio/onset.SpectralFluxStrength) sampled at
//     frameRate Hz.
//  2. Autocorrelation of the novelty function reveals periodicities:
//     ACF[lag] is large when the novelty function repeats at period
//     `lag` frames.
//  3. The dominant lag within the plausible BPM range (default
//     30-300 BPM) is the inferred beat period; BPM = 60 * frameRate / lag.
//
// This is a single-tempo estimator — it does not handle tempo changes
// across the input window. Production users wanting time-varying tempo
// estimates should slide a window over the novelty function and call
// EstimateBpm per window.
//
// # Determinism
//
// All functions are pure / deterministic. No globals, no goroutines.
// Output slices are caller-allocated where possible.
//
// # Consumers
//
//   - foundation/reality/audio/beat (future): beat-tracking pipeline
//     uses tempo as a prior for dynamic-programming beat alignment.
//   - Pistachio (audio): rhythm analysis + transcription.
//   - Howler / Pigeonhole / Dipstick (audio trio): periodic-event
//     detection in vocal/mechanical signals.
//
// # Reference
//
//   - Goto, M. (2001) "An audio-based real-time beat tracking system
//     for music with or without drum-sounds." Journal of New Music
//     Research 30(2), 159-171.
//   - Davies, M.E.P. & Plumbley, M.D. (2007) "Context-dependent beat
//     tracking of musical audio." IEEE Trans. Audio, Speech, Lang.
//     Process. 15(3), 1009-1020.
//   - Klapuri, A.P., Eronen, A.J. & Astola, J.T. (2006) "Analysis of
//     the meter of acoustic musical signals." IEEE Trans. Audio,
//     Speech, Lang. Process. 14(1), 342-355.
package tempo

// Package beat tracks beat positions in an audio signal given an
// onset-strength novelty function and a tempo prior.
//
// # Algorithm
//
// Implements the Ellis 2007 dynamic-programming beat tracker:
//
//  1. Caller supplies a novelty function (e.g. via
//     foundation/reality/audio/onset.SpectralFluxStrength) and a
//     tempo prior (in BPM, e.g. via
//     foundation/reality/audio/tempo.Estimate).
//  2. For each time t, compute backlink score
//
//	score[t] = novelty[t] + max over s of (score[s] - alpha * (log((t-s)/period))^2)
//
//     where `period` is the expected inter-beat interval in frames
//     (60 * frameRate / bpm) and `alpha` controls the rigidity of
//     the tempo prior.
//  3. Backtrack from the frame with maximum score to recover the
//     beat sequence.
//
// The log-squared transition penalty tolerates small tempo drift
// (close-to-period transitions are cheap) while strongly penalising
// large deviations. alpha = ~50-150 is typical; default 100.
//
// # Determinism
//
// Pure / deterministic. Same (novelty, opts) → same beat sequence.
// No goroutines, no globals.
//
// # Consumers
//
//   - Pistachio (audio): rhythm-aware transcription.
//   - Howler / Pigeonhole / Dipstick: periodic-event tracking with
//     tempo regularisation.
//
// # Reference
//
//   - Ellis, D.P.W. (2007) "Beat tracking by dynamic programming."
//     Journal of New Music Research 36(1), 51-60.
package beat

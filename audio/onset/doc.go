// Package onset provides onset-detection primitives — the localisation
// in time of percussive events (note attacks, drum hits, bird-call
// onsets, mechanical service-event transients). Onset detection is the
// canonical first stage of any beat-tracking, transcription, or
// segmentation pipeline.
//
// Five detection-function families ship in this package, each addressing
// a distinct trade-off between simplicity, sensitivity, and modern
// state-of-the-art quality:
//
//   - EnergyOnset (RMS rise) — the simplest detector: a frame is an
//     onset if its short-time energy exceeds a moving baseline. Cheap;
//     adequate for percussive signals with strong amplitude rises.
//
//   - SpectralFluxOnset (Bello & Sandler 2003 / Dixon 2006) — the
//     workhorse. Half-wave-rectified bin-wise difference of consecutive
//     magnitude spectra. The de-facto baseline against which every
//     other detector is compared.
//
//   - ComplexDomainOnset (Bello et al. 2004) — combines magnitude and
//     phase deviation in a single complex-domain residual. Picks up
//     non-percussive onsets (legato, soft attacks) that magnitude-only
//     flux misses.
//
//   - SuperFlux (Böck & Widmer 2013) — the modern de-facto for
//     percussive onsets. Vibrato-suppressing spectral flux: the per-bin
//     reference is a max-filter over neighbouring bins of the previous
//     frame, so frequency-modulated tones do not generate false positives.
//
//   - PickPeaks — adaptive median-thresholded peak picker. Operates on
//     any of the above onset-strength functions to produce frame-index
//     onset times.
//
// All functions are deterministic, use only the Go standard library,
// and target zero allocations in hot paths via caller-provided scratch.
//
// This package builds on reality/signal (FFT, windows) and reality/audio/
// spectrogram (STFT). It follows the Reality convention: numbers in,
// numbers out. Every function documents its formula, valid range,
// precision, and reference.
//
// References:
//   - Bello, J. P. & Sandler, M. (2003). "Phase-based note onset
//     detection for music signals." ICASSP 2003.
//   - Bello, J. P., Duxbury, C., Davies, M. & Sandler, M. (2004). "On
//     the use of phase and energy for musical onset detection in the
//     complex domain." IEEE Signal Processing Letters 11(6).
//   - Dixon, S. (2006). "Onset detection revisited." DAFx-06 Conf.
//   - Böck, S. & Widmer, G. (2013). "Maximum filter vibrato suppression
//     for onset detection." DAFx-13 Conf.
//
// Consumed by:
//   - flagships/pigeonhole (call boundary detection — "one bird-call
//     at a time" segmentation pipeline)
//   - flagships/dipstick (percussive service-event detection — engine
//     start, valve opening, bearing impact)
//   - flagships/howler (vocalisation onset detection)
//   - reality/audio/segmentation (onset/offset segmentation builds on
//     the same strength functions)
package onset

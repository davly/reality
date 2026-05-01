// Package segmentation provides primitives for extracting individual
// events (e.g. one bird call, one valve actuation, one machine
// service-event) from a longer continuous recording.
//
// The audio cohort flagships (Pigeonhole, Howler, Dipstick) all share
// a "one event at a time" downstream pipeline: a longer recording
// (10s, 60s, an hour) is first segmented into individual events, then
// each event is fingerprinted/classified individually. This package
// is the segmentation half of that pipeline.
//
// Five primitives ship in this package:
//
//   - Segment (energy-VAD-based): the simplest and cheapest detector.
//     A frame is "active" if its short-time energy exceeds a threshold;
//     contiguous runs of active frames form segments. Adequate for
//     clean recordings with strong silence between events.
//
//   - SegmentByOnsetOffset: uses spectral-flux onset detection on the
//     rising edge and energy decay below threshold for the offset.
//     More robust than energy-only on signals with frequency-shifted
//     onsets (e.g., bird calls that start mid-frequency).
//
//   - SegmentWithMinSilence: splits a signal at any silence run >=
//     minSilence milliseconds. Useful for recordings where events are
//     well-separated in time but vary in duration.
//
//   - MergeCloseSegments: post-processing — coalesces nearby segments
//     into single composite segments. Handles the case where a bird
//     call has internal "pauses" that the segmenter incorrectly split.
//
//   - FilterByMinDuration: post-processing — drops segments shorter
//     than minMs. Defends against transient false positives.
//
// All functions are deterministic, use only the Go standard library,
// and target zero allocations in hot paths via caller-provided scratch.
//
// This package builds on reality/audio/onset (SpectralFluxStrength)
// and reality/audio/separation (FrameEnergy). It follows the Reality
// convention: numbers in, numbers out. Every function documents its
// formula, valid range, precision, and reference.
//
// The Segment struct is the canonical output type — a half-open
// interval [StartIdx, EndIdx) in sample units.
//
// References:
//   - Rabiner, L. R. & Sambur, M. R. (1975). "An algorithm for
//     determining the endpoints of isolated utterances." Bell Syst.
//     Tech. J. 54(2). The classic energy-based VAD/segmentation paper.
//   - Bello et al. (2005). "A tutorial on onset detection in music
//     signals." IEEE Trans. Speech Audio Proc. 13(5). Basis for
//     onset/offset segmentation.
//
// Consumed by:
//   - flagships/pigeonhole (per-call segmentation in continuous
//     recordings — the canonical use case)
//   - flagships/howler (per-vocalisation segmentation)
//   - flagships/dipstick (per-event segmentation in long machine
//     recordings — startup, steady-state, fault transient)
package segmentation

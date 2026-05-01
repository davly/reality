// Package pitch provides pitch-detection / fundamental-frequency-
// estimation primitives. More robust than peak-picking the FFT (which
// reality/audio/vibration provides as the cheap-fast path) — these
// algorithms handle missing fundamentals, noisy harmonics, and the
// monophonic-pitch problems that arise in voiced speech, bird-song,
// and the musical pitch-tracking literature.
//
// Four primitives ship in this package, each addressing a distinct
// pitch-tracking regime:
//
//   - AutocorrelationPitch (classical ACF): the simplest pitch
//     detector. Find the lag with the highest autocorrelation in
//     the period range [sampleRate/fMax, sampleRate/fMin]. Cheap;
//     adequate for clean monophonic signals.
//
//   - Yin (de Cheveigné & Kawahara 2002): the de-facto modern
//     monophonic pitch detector. ACF + cumulative-mean-normalised
//     difference function with absolute-threshold dip selection.
//     Handles the missing-fundamental problem and reports an
//     aperiodicity score that doubles as a confidence indicator.
//
//   - McLeodPitchMethod (McLeod & Wyvill 2005): the modern alternative
//     to YIN. Normalised square-difference function with a parabolic-
//     interpolated peak picker. Reports a clarity score and is
//     particularly good on instrumental notes with strong harmonics.
//
//   - SubharmonicSummation (Hermes 1988 SHS): summing harmonic
//     evidence in the frequency domain. Distinct virtue: detects
//     fundamentals even when the fundamental BIN is empty (the
//     "missing fundamental" perceptual phenomenon — humans hear a
//     pitch where no spectral energy lies). Useful for harmonic-rich
//     signals like boiler combustion noise where the fundamental is
//     too low to register but its harmonics dominate.
//
// All functions are deterministic, use only the Go standard library,
// and target zero allocations in hot paths via caller-provided scratch.
//
// This package complements reality/audio/vibration (which provides
// the cheaper FFT peak-picker FundamentalHz). Choose:
//
//   - vibration.FundamentalHz when the fundamental is known to be the
//     loudest spectral component (mechanical signals)
//   - pitch.AutocorrelationPitch / pitch.Yin / pitch.McLeodPitchMethod
//     when the signal may have a missing fundamental, reverberation,
//     or noise (vocal / bird / pet vocalisations)
//   - pitch.SubharmonicSummation when the fundamental is known to be
//     missing (low-frequency signals where the FFT bin is empty)
//
// References:
//   - de Cheveigné, A. & Kawahara, H. (2002). "YIN, a fundamental
//     frequency estimator for speech and music." J. Acoust. Soc. Am.
//     111(4), 1917-1930.
//   - McLeod, P. & Wyvill, G. (2005). "A smarter way to find pitch."
//     Proc. International Computer Music Conference (ICMC), 138-141.
//   - Hermes, D. J. (1988). "Measurement of pitch by subharmonic
//     summation." J. Acoust. Soc. Am. 83(1), 257-264.
//   - Roads, C. (1996). "The Computer Music Tutorial" §10
//     (autocorrelation pitch detection).
//
// Consumed by:
//   - flagships/pigeonhole (bird-song fundamental tracking — most
//     bird species are tonal)
//   - flagships/howler (canine/feline vocalisation pitch)
//   - flagships/dipstick (machine resonance fundamental, alternative
//     to vibration.FundamentalHz when the fundamental is missing —
//     e.g. boiler combustion at sub-20 Hz)
package pitch

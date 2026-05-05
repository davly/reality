// Package spectrogram provides Short-Time Fourier Transform (STFT)
// computation primitives plus spectrogram-as-image rendering.
//
// The STFT is the canonical 2-D time-frequency representation of an
// audio signal and the intended substrate for downstream consumers
// (MFCC, fingerprinting, NMF separation, visual inspection, machine
// listening pipelines). As of 2026-05-05 the verified consumers are
// in-Reality only — sibling packages audio/onset and audio/segmentation
// import this package via test code; the named candidate flagships
// Pigeonhole, Howler, Dipstick do not yet import it. First-consumer push
// across the cohort is queued; see
// LimitlessGodfather/reviews/SESSION_62_PROGRESS.md.
//
// This package provides:
//
//   - STFT(samples, frameSize, hopSize, window) — overlap-add complex
//     STFT computation
//   - Magnitude / LogMagnitude — element-wise |X[t][f]| and
//     20·log10(|X[t][f]|) reductions
//   - MelSpectrogram — composes STFT magnitude × audio.MelFilterbank
//     to produce a 2-D mel-band energy matrix
//   - ToHeatmap / colourmaps — render the spectrogram as a PNG-encoded
//     RGBA image suitable for the "spectrogram-as-art" feature (intended
//     for Pigeonhole / Howler / Dipstick adoption)
//
// Production colourmaps (Plasma, Magma, Viridis, Inferno) match the
// matplotlib references — RGB lookups computed at 256 stops with
// linear interpolation between stops. These are the perceptually-
// uniform colourmaps developed by Stefan van der Walt and Nathaniel
// Smith for matplotlib 1.5+ (released 2015) and adopted across the
// scientific-computing world as the default for sequential data.
//
// All functions are deterministic, use only the Go standard library
// (`image`, `image/png`, `bytes`, `math`), and target zero allocations
// in numerical hot paths via caller-provided scratch buffers. The
// PNG-encoding paths necessarily allocate (the standard-library
// encoder builds an intermediate buffer).
//
// References:
//   - Allen, J. B. & Rabiner, L. R. (1977). "A unified approach to
//     short-time Fourier analysis and synthesis." Proc. IEEE 65(11).
//   - Griffin, D. W. & Lim, J. S. (1984). "Signal estimation from
//     modified short-time Fourier transform." IEEE Trans. ASSP 32(2).
//   - matplotlib's "Plasma/Magma/Viridis/Inferno" colourmap LUTs
//     (van der Walt & Smith, 2015 — public domain CC0).
//
// Consumed by:
//   - flagships/pigeonhole (spectrogram-as-art bird-call rendering)
//   - flagships/howler (canine/feline vocalisation visualisation)
//   - flagships/dipstick (machine spectrogram audit trail)
//   - reality/audio/separation (NMF input substrate)
package spectrogram

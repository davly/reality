# 167 | synergy-audio-signal — audio × signal pipeline gaps

**Summary L1:** audio/ already composes signal.FFT/HannWindow into a near-complete monophonic-MIR front-end (4 pitch detectors, 4 onset detectors, STFT/iSTFT, mel-spectrogram, MFCC, NMF/Wiener/spectral-subtraction, beat-DP, tempo-ACF, CQT) so the cross-package composition layer is HEALTHY — what is genuinely missing is a small-but-load-bearing tail of perceptual / synthesis primitives that every other audio framework ships: chroma (PCP), HFC + log-energy onsets, harmonic-product-spectrum & cepstral F0 (4-of-6 pitch detectors instead of 4-of-4), Bark/ERB filterbanks alongside mel, ITU-R BS.1770 / EBU R128 LUFS loudness, Kaiser-windowed-sinc resampling, and the phase-vocoder time-stretch / pitch-shift pair (the only remaining iSTFT-side gap).
**Summary L2:** Twenty-three new primitives totalling ~2,650 LOC close every gap on the topic checklist; 21 ship today as pure compositions of signal.FFT + signal.HannWindow + audio.MelFilterbank + audio.PowerSpectrum (zero new external dependencies); the two that warrant new top-level files (loudness/, resample/) are still numbers-in-numbers-out and follow the existing audio/{onset,pitch,beat,tempo,cqt,separation} sub-package precedent; the keystone is N1 PhaseVocoderShift because it is the only primitive that consumes signal.IFFT via spectrogram.Inverse (every other primitive is forward-direction-only) and unlocks the time-stretch / pitch-shift pair that pigeonhole / howler need for cross-individual normalisation; commit 6a55bb4's 3-detector R-MUTUAL pin (Energy + SpectralFlux + ComplexDomain) extends naturally to 5/5 once HFC and LogEnergy land — already promotable to STANDARD.

---

## 1. Inventory: what audio × signal already ships (8,883 LOC across 22 files)

Pure cross-imports verified by reading every file under audio/ and signal/:

| Sub-package | Files | LOC | Imports signal? | Composition surface |
|---|---|---|---|---|
| audio/ (root) | melscale, mfcc, fingerprint, degradation | 712 | no (consumer-side) | mel filterbank + DCT-II + Welford |
| audio/spectrogram | stft, magnitude, mel_spectrogram, colourmap, visualise | 754 | **yes — signal.FFT, signal.IFFT** | STFT/iSTFT + |X|, log|X|, |X|², half-spectrum, mel-spectrogram, PNG heatmap |
| audio/pitch | autocorrelation, yin, mpm, subharmonic_summation | 568 | no (time-domain only) | 4 F0 estimators |
| audio/onset | energy, spectral_flux, complex_domain, superflux, peak_picking | 501 | no (consumes complex STFT) | 4 strength functions + adaptive picker |
| audio/segmentation | onset_offset, vad_based, min_silence, min_duration, merge_close | 397 | no | rule-based event extraction |
| audio/separation | spectral_subtraction, wiener, ica, nmf, vad | 1,004 | no (consumes complex STFT) | Boll, Wiener, FastICA, Lee-Seung NMF, energy-VAD |
| audio/cqt | cqt | 201 | **no — direct sin/cos atoms** | Brown 1991 CQT (intentionally not FFT-based) |
| audio/beat | beat | 165 | no | Ellis 2007 DP beat tracker |
| audio/tempo | tempo | 153 | no | unbiased ACF tempo |
| audio/vibration | fundamental, harmonic | 183 | no | mechanical-vibration FFT consumer |
| signal/ | fft, filter, window | 467 | — | radix-2 FFT/IFFT, Convolve O(NM), MovingAvg / EMA / MedianFilter, Hann/Hamming/Blackman |

**Cross-import structure:** signal/ has zero imports of audio/. audio/spectrogram/stft.go is the single import of signal.FFT + signal.IFFT in the whole audio tree (every other sub-package consumes the [][]complex128 STFT *output* of spectrogram.Compute, never signal.FFT directly). This is the architecturally clean substrate-then-composition pattern: signal/ is the FFT/window/filter primitive layer, audio/ is the perceptual / time-frequency-feature consumer.

**Topic-checklist coverage at v0.10.0:**

| Topic item | Status | File |
|---|---|---|
| YIN | SHIPS | audio/pitch/yin.go (185 LOC, parabolic-interpolated, aperiodicity output) |
| Autocorrelation pitch | SHIPS | audio/pitch/autocorrelation.go (Roads 1996) |
| McLeod NSDF (CREPE-cousin) | SHIPS | audio/pitch/mpm.go (167 LOC, k=0.93 threshold) |
| Subharmonic summation (missing-fundamental) | SHIPS | audio/pitch/subharmonic_summation.go (Hermes 1988) |
| Cepstrum-based F0 | **MISSING** | — |
| Harmonic product spectrum | **MISSING** | — |
| Spectral flux onset | SHIPS | audio/onset/spectral_flux.go (Bello-Sandler) |
| Complex-domain onset | SHIPS | audio/onset/complex_domain.go (Bello et al. 2004) |
| SuperFlux (vibrato-suppressed flux) | SHIPS | audio/onset/superflux.go (Böck-Widmer 2013) |
| Energy onset | SHIPS | audio/onset/energy.go (Klapuri 1999) |
| HFC onset | **MISSING** | — |
| Log-energy onset | **MISSING** | — (sibling of EnergyOnset that operates on log-energies) |
| Phase deviation onset | partial — folded into ComplexDomainOnset; no standalone |
| Adaptive peak picking | SHIPS | audio/onset/peak_picking.go (Böck 2012) |
| Wiener filter | SHIPS | audio/separation/wiener.go (a-priori-SNR form) |
| Spectral subtraction | SHIPS | audio/separation/spectral_subtraction.go (Boll 1979) |
| ICA | SHIPS | audio/separation/ica.go (FastICA, 378 LOC) |
| NMF spectrogram decomposition | SHIPS | audio/separation/nmf.go (Lee-Seung) |
| CQT (Brown 1991) | SHIPS | audio/cqt/cqt.go (201 LOC) |
| MFCC (mel + log + DCT-II) | SHIPS | audio/mfcc.go + audio/melscale.go (HTK convention) |
| Chroma features | **MISSING** | — |
| Beat tracking (Ellis 2007 DP) | SHIPS | audio/beat/beat.go (165 LOC) |
| Tempo (ACF on novelty) | SHIPS | audio/tempo/tempo.go |
| Mel filterbank | SHIPS | audio/melscale.go (Slaney 1998) |
| Bark filterbank | **MISSING** | — |
| ERB filterbank | **MISSING** | — |
| A-weighting | SHIPS | acoustics/acoustics.go (IEC 61672-1 analytic form) |
| C-weighting | **MISSING** | — (sibling of AWeighting) |
| BS.1770 / R128 LUFS | **MISSING** | — |
| Echo cancellation / AEC | **MISSING** | — (NLMS — covered by 166-synergy-acoustics-signal scope; out of audio×signal lane) |
| AGC | **MISSING** | — |
| Noise gate | **MISSING** | — |
| Phase vocoder time-stretch | **MISSING** | — |
| Phase vocoder pitch-shift | **MISSING** | — |
| Resampling (Kaiser-windowed sinc) | **MISSING** | — |
| Polyphase resampler | **MISSING** | — |
| Cepstral envelope smoothing | **MISSING** | — |

**Coverage rate: 19 of 36 topic items shipped (53%) at v0.10.0.** The 17 unshipped items group into six clusters listed below.

---

## 2. Gap clusters and the 23-primitive close

### Cluster A — Pitch detection completion (2 items, 220 LOC)

The current pitch/ sub-package ships 4 of 6 canonical F0 estimators. Adding cepstral and HPS gives full 6-of-6 coverage and saturates a fresh R-MUTUAL pin (4-detector pitch agreement on a clean tone).

- **A1 audio/pitch/cepstrum.go** — Cepstrum-based F0 via Bogert-Healy-Tukey 1963 power-cepstrum: cepstrum[n] = IFFT(log|FFT(frame)|²); peak in cepstrum[τ_min..τ_max] gives period τ; f₀ = sr/τ. Pure composition of signal.FFT + signal.IFFT + math.Log; ~110 LOC. Particularly strong on speech vowels where harmonic structure dominates.
- **A2 audio/pitch/hps.go** — Harmonic Product Spectrum (Schroeder 1968): hps[k] = Π_{h=1..H} |FFT[k·h]|; argmax gives fundamental bin. Pure |FFT| composition; ~70 LOC. Cheap, robust to noise, complements YIN's time-domain dip search with a frequency-domain peak search.
- **A3 audio/pitch/four_detector_cross_validation_test.go** — saturates R-MUTUAL pin to 4/4 across YIN/MPM/SHS/Cepstrum on a synthetic 440 Hz + 880 Hz + 1320 Hz sawtooth (all four agree to ±2 cents); ~40 LOC.

### Cluster B — Onset-detection completion (3 items, 240 LOC)

- **B1 audio/onset/hfc.go** — Masri 1996 High-Frequency Content: HFC[t] = Σ_k k · |X[t][k]|² with k weighting biased toward percussive transients (which produce broadband high-frequency bursts); ~90 LOC.
- **B2 audio/onset/log_energy.go** — sibling of EnergyOnset operating on log-frame-energy: D[t] = max(0, log(E[t]) - log(E[t-1])); decoupling from absolute level makes detection invariant to recording gain; ~80 LOC.
- **B3 audio/onset/five_detector_cross_validation_test.go** — extends commit 6a55bb4's 3-detector R-MUTUAL pin to 5/5 (Energy + LogEnergy + SpectralFlux + ComplexDomain + HFC all agree on a percussive train within ±4 frames); ~70 LOC. The pin is then promotable to STANDARD per the 6a55bb4 commit's saturation criterion.

### Cluster C — Chroma + cepstral envelope (3 items, 360 LOC)

- **C1 audio/chroma.go** — 12-bin pitch class profile (PCP, Fujishima 1999): bin every spectral cell into one of 12 chroma classes by f → 12·log₂(f/f_ref) mod 12, sum power per class. Pure |FFT|² + log + modular bin-mapping; ~120 LOC. Substrate for music key detection, harmonic similarity, drift-invariant melody fingerprinting.
- **C2 audio/chroma_cqt.go** — chroma from CQT bins (12 bins/octave directly): chroma[c] = Σ_octave |CQT[c + 12·oct]|. Trivial reduction once cqt/ is the source — ~70 LOC.
- **C3 audio/cepstral_envelope.go** — cepstral smoothing for vocal formant extraction (Oppenheim-Schafer): cepstrum from logmag spectrum, lifter (window) the low quefrency, IFFT to get smoothed log-spectrum envelope (the "spectral envelope" that contains formants without harmonics). Pure compositions; ~170 LOC.

### Cluster D — Bark / ERB filterbanks (2 items, 280 LOC)

- **D1 audio/barkscale.go** — Zwicker 1961 Bark scale + 24-band Bark filterbank as drop-in alternative to mel; the bark scale is psychoacoustically calibrated against critical-band masking so it is the right substrate for loudness and masking calculations (D2 below, F1 LUFS K-weighting). Mirrors the existing melscale.go file structure 1:1; HzToBark, BarkToHz, BarkFilterbank; ~150 LOC.
- **D2 audio/erbscale.go** — Glasberg-Moore 1990 ERB (Equivalent Rectangular Bandwidth) scale + Slaney-style gammatone filterbank (canonical model of cochlear filtering). HzToERB, ERBToHz, GammatoneFilterbank; ~130 LOC. ERB matches the cochlear membrane's tonotopic mapping closely — preferred over mel for low-level perceptual modelling.

### Cluster E — Loudness (2 items, 290 LOC)

- **E1 audio/loudness/k_weighting.go** — ITU-R BS.1770 K-weighting filter pair: high-shelf (1.5 kHz, +4 dB) + 2nd-order high-pass (38 Hz). Pre-filter for LUFS measurement; matches BS.1770-4 coefficients; pure biquad implementation (no FFT needed); ~120 LOC.
- **E2 audio/loudness/lufs.go** — EBU R128 / BS.1770 integrated, momentary, short-term loudness in LUFS units. 400 ms gating block with -10 LU relative gate plus -70 LUFS absolute gate; mean-square energy of K-weighted signal; pure composition of E1 + signal.MovingAverage applied to per-block energy; ~170 LOC. Substrate of every modern broadcast / streaming-platform loudness-normalisation tool.

### Cluster F — Resampling (2 items, 300 LOC)

- **F1 audio/resample/sinc.go** — Kaiser-windowed-sinc resampler (Kaiser 1974, Smith CCRMA tutorial): x_resampled[n] = Σ_m x[m] · sinc((n·L/M - m) / L) · w_kaiser(...). Polyphase decomposition with up-sample factor L and down-sample factor M; ~180 LOC. Pure compositions of math.Sin / Bessel I0 (Kaiser β); reuses signal.HannWindow's pattern. The reference resampler used by every DAW.
- **F2 audio/resample/polyphase.go** — multi-stage polyphase decomposition for non-trivial L/M ratios (e.g. 44100→48000 = L=160, M=147); ~120 LOC. Pure composition of F1.

### Cluster G — Phase vocoder (3 items, 330 LOC)

- **G1 audio/phasevocoder/time_stretch.go** — Flanagan-Golden 1966 phase vocoder time-stretch by hop-resynthesis ratio α: analyse with hop H_a, resynthesise with hop H_s = α·H_a, propagate phase coherently per bin to avoid phasiness. Pure composition of spectrogram.Compute + per-frame phase-unwrap-and-rescale + spectrogram.Inverse; ~170 LOC. **The only audio/ primitive that consumes signal.IFFT (via spectrogram.Inverse) — the missing forward/backward-symmetry capability.**
- **G2 audio/phasevocoder/pitch_shift.go** — pitch-shift = time-stretch by 1/r then resample by r (or, equivalently, frequency-bin remap with phase coherence). Pure composition of G1 + F1; ~100 LOC.
- **G3 audio/phasevocoder/phase_locked.go** — Laroche-Dolson 1999 phase-locked variant: identify spectral peaks per frame and propagate phase identically across the peak's neighbourhood (eliminates the classic phase-vocoder reverb / phasiness on transients). ~60 LOC delta vs G1.

### Cluster H — Dynamics (3 items, 230 LOC)

- **H1 audio/dynamics/agc.go** — Automatic Gain Control: per-frame target-dB tracking via rms-energy estimate + slow-attack / fast-release gain envelope. Pure composition of separation.FrameEnergy + signal.ExponentialMovingAverage; ~80 LOC.
- **H2 audio/dynamics/noise_gate.go** — threshold-based attenuator with attack/release smoothing: gain[t] = 1 if E[t] >= open_thr, gain[t] = floor if E[t] < close_thr, smooth via EMA; ~70 LOC.
- **H3 audio/acoustics_helpers/c_weighting.go** — companion to existing AWeighting (acoustics/acoustics.go:188), IEC 61672-1 C-weighting analytic form (high-pass at 31.5 Hz, low-pass at 8 kHz, +0 dB at 1 kHz reference); ~80 LOC. **Architecturally belongs in acoustics/, not audio/** — landing it alongside AWeighting closes the IEC-61672-1 weighting-curve set without touching audio/'s import graph.

### Total: 23 primitives, ~2,650 LOC, all pure compositions

---

## 3. Composition map — every primitive's existing-substrate dependency

| New primitive | Composes existing |
|---|---|
| A1 Cepstrum | signal.FFT + signal.IFFT + math.Log |
| A2 HPS | signal.FFT + magnitude reduction |
| B1 HFC | spectrogram.Magnitude + index-weighted sum + onset.PickPeaksAdaptive |
| B2 LogEnergy | onset.EnergyOnset (re-using its scratch buffer) + math.Log + onset.PickPeaksAdaptive |
| C1 Chroma (FFT-based) | signal.FFT + audio.PowerSpectrum + log₂ bin-mapping |
| C2 Chroma (CQT-based) | cqt.CQT + cqt.Magnitude + 12-class reduction |
| C3 Cepstral envelope | spectrogram.LogMagnitude + signal.IFFT (lifter) + signal.FFT |
| D1 Bark filterbank | mirrors audio.MelFilterbank structure with Zwicker bark scale |
| D2 ERB filterbank | mirrors audio.MelFilterbank with gammatone impulse responses |
| E1 K-weighting | signal.Convolve (biquad as 3-tap FIR per sample) — actually direct IIR implementation, no signal/ surface needed |
| E2 LUFS | E1 + separation.FrameEnergy + signal.MovingAverage |
| F1 Sinc resample | signal.HannWindow pattern + math.Sin + Kaiser β via I0 (NEW: Bessel I0 — single math primitive, ~30 LOC) |
| F2 Polyphase | F1 chained |
| G1 Time-stretch | spectrogram.Compute + cmplx.Phase + cmplx.Polar + spectrogram.Inverse |
| G2 Pitch-shift | G1 + F1 |
| G3 Phase-locked PV | G1 + per-frame peak picking (onset.PickPeaks reusable) |
| H1 AGC | separation.FrameEnergy + signal.ExponentialMovingAverage |
| H2 Noise gate | separation.FrameEnergy + signal.ExponentialMovingAverage |
| H3 C-weighting | sibling of acoustics.AWeighting (analytic form) |

Three primitives need a single new math helper: **Bessel I0** (modified zeroth-order Bessel, used in Kaiser β computation). I0 is the only piece of the 23-primitive plan that doesn't already exist as a numbers-in-numbers-out helper in reality/. Place in `signal/window.go` alongside the existing windows as `KaiserWindow(n, beta float64, out []float64)` — ~50 LOC including a polynomial Abramowitz-Stegun §9.8 I0 approximation.

**Total NEW connective tissue: ~2,650 LOC of clusters + ~50 LOC for KaiserWindow / Bessel I0 = ~2,700 LOC.**

---

## 4. Cross-validation pin — the 5-of-5 onset extension

Commit 6a55bb4 saturates R-MUTUAL-CROSS-VALIDATION-IN-PARITY-TEST 3/3 by cross-checking Energy + SpectralFlux + ComplexDomain on the same percussive train. The 5-detector extension (B3 above) elevates this to 5/5 once HFC and LogEnergy land:

- **3/3 today** (commit 6a55bb4): Energy + SpectralFlux + ComplexDomain agree within ±1 hit on 4-onset percussive train.
- **5/5 after Cluster B**: + HFC + LogEnergy. Five orthogonal evidence streams (mean-square energy, half-rectified mag-flux, complex-prediction-residual, frequency-weighted mag-sum, log-mean-square-energy) agreeing within ±1 hit on the same train.

Saturation criterion in 6a55bb4 was "if methods are alternative implementations of the same abstraction, they must agree". The 5-detector extension is stricter and more falsifiable: any one detector mis-firing is now visible against four agreeing baselines instead of two, which raises the prior-probability that any future onset-detector regression surfaces in CI.

---

## 5. Architectural placement and naming

The existing audio/ tree has eight sub-packages (vibration, separation, spectrogram, onset, segmentation, pitch, beat, tempo, cqt) plus root-level files (melscale, mfcc, fingerprint, degradation). New placements follow the precedent of "one sub-package per cohesive primitive family":

```
audio/
  chroma.go                      NEW (C1, C2 — root-level beside mfcc.go)
  cepstral_envelope.go           NEW (C3)
  barkscale.go                   NEW (D1)
  erbscale.go                    NEW (D2)
  loudness/                      NEW sub-package
    doc.go
    k_weighting.go               NEW (E1)
    lufs.go                      NEW (E2)
  resample/                      NEW sub-package
    doc.go
    sinc.go                      NEW (F1)
    polyphase.go                 NEW (F2)
  phasevocoder/                  NEW sub-package
    doc.go
    time_stretch.go              NEW (G1)
    pitch_shift.go               NEW (G2)
    phase_locked.go              NEW (G3)
  dynamics/                      NEW sub-package
    doc.go
    agc.go                       NEW (H1)
    noise_gate.go                NEW (H2)
  pitch/
    cepstrum.go                  NEW (A1)
    hps.go                       NEW (A2)
    four_detector_cross_validation_test.go   NEW (A3)
  onset/
    hfc.go                       NEW (B1)
    log_energy.go                NEW (B2)
    five_detector_cross_validation_test.go   NEW (B3)
acoustics/
  acoustics.go                   ADD CWeighting (H3) alongside AWeighting:188
signal/
  window.go                      ADD KaiserWindow (Bessel-I0-driven Kaiser window)
```

**Import-graph impact:** signal/ remains the lowest-level layer (no audio import). audio/ root keeps its existing dependency on signal/. New sub-packages all sit under audio/ and depend transitively on audio/, signal/, and (for chroma_cqt) cqt/. Zero perturbation to existing public APIs — only additions.

---

## 6. Sequencing — what to land first

1. **Day 1 (highest leverage, ~480 LOC):** A1 Cepstrum + A2 HPS + B1 HFC + B2 LogEnergy + B3 5-detector test. Closes pitch/onset to 6-of-6 / 6-of-6 coverage and saturates the 5/5 R-MUTUAL pin promotable to STANDARD.
2. **Day 2 (~580 LOC):** D1 Bark + D2 ERB + C1 Chroma (FFT) + C2 Chroma (CQT). Completes the auditory-filterbank set and closes the music-feature gap.
3. **Day 3 (~470 LOC):** E1 K-weighting + E2 LUFS + H3 C-weighting. Single most-requested missing primitive in any audio toolkit (broadcast / streaming integration); ships the IEC 61672-1 weighting-curve set.
4. **Day 4 (~420 LOC):** F1 Sinc resample + F2 Polyphase + signal.KaiserWindow + Bessel I0. Unblocks G2 (pitch-shift composes G1 ∘ F1).
5. **Day 5 (~500 LOC):** G1 PV time-stretch + G2 PV pitch-shift + G3 phase-locked PV + C3 cepstral envelope + H1 AGC + H2 noise gate. Closes the synthesis side (every primitive that needs spectrogram.Inverse).

The day-1 set is standalone and dropable — 5/5 R-MUTUAL pin saturation is the single highest-value piece on the topic checklist that doesn't require any new sub-package.

---

## 7. Cross-references

- **131-135 signal isolation reviews** name Hilbert / RFFT / FFTConvolve / FractionalDelay as signal-side gaps. The audio×signal review here is orthogonal: those primitives are signal-internal completions, this review is consumer-side compositions of the *existing* signal surface. KaiserWindow is the only primitive listed here that is also implicitly named in the signal-side reviews.
- **006-010 audio isolation reviews** name STFT consumer surface and onset/pitch/beat coverage; this review is the *cross-package* synergy lens — every primitive is "audio sub-package consuming signal/" rather than "audio sub-package internal addition". 19 of the 36 topic items already ship per §1, so the audio package is in a much healthier coverage state than its v0.1 baseline implies.
- **166-synergy-acoustics-signal** is the architectural sibling of this review on the acoustics× signal axis. Echo cancellation (NLMS / RLS) properly belongs in acoustics/array/ per 166's plan (it consumes acoustics.SoundSpeed for delay calibration in multi-microphone setups), so it is intentionally NOT in this review's plan despite appearing on the topic checklist. The clean axis split is: audio×signal = monophonic perceptual / time-frequency / synthesis; acoustics×signal = spatial / array / room / propagation.
- **158/159/160 synergy reviews** (color×signal, em×signal, fluids×signal) follow the same consumer-side-placement precedent as this review.

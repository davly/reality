# 008 — audio: SOTA comparison (librosa, essentia, aubio, torchaudio, …)

## Headline
Reality's `audio` package occupies the "1995–2015 textbook MIR" tier — STFT, mel, MFCC, YIN, NMF, FastICA, Lee-Seung, Boll spectral subtraction, Ellis-DP beat — and trails the 2025 ecosystem along three axes: (1) the **classical-MIR baseline** that librosa/Essentia/aubio expose for free (chroma family, spectral descriptors, gammatone/Bark/ERB filterbanks, LPC, Griffin-Lim, inverse-mel, polyphase resampling, broadcast-grade LUFS metering, Shazam/Chromaprint hashing); (2) the **invertible / differentiable transform** layer that has consolidated since 2020 (NSGT/sliCQT, MDCT, harmonic-percussive median, β-NMF with IS, sliCQT-domain Wiener filter, K-weighted true-peak); and (3) the **hybrid neural primitives** that 2024–2026 papers chain on top of classical kernels (DDSP harmonic+filtered-noise synth, RVQ-style residual vector quantization, CARFAC-JAX cochlear model, Demucs-HT MDX building blocks, PESTO/FCPE pitch back-ends). The first two axes are entirely **in-scope, zero-dependency, golden-file-testable Go**; the third axis is interesting because reality already ships `autodiff` + `linalg` + `signal/fft` and could plausibly reimplement the *math* of the differentiable kernels (sinusoidal harmonic synth, FIR-via-FFT filtering, RVQ encoder, gammatone+AGC cascade) without ever depending on a tensor framework. Recommendation: target axis 1 (~3,000 LOC) as the v0.11 pull-up, axis 2 (~1,500 LOC) as v0.12, and pick three axis-3 kernels (DDSP synth, gammatone-AGC, RVQ) as v0.13 differentiable showcases.

## Reality's current audio scope

(Per agent 007 catalogue.)

`audio/` root: `melscale.go`, `mfcc.go`, `fingerprint.go` (Welford), `degradation.go` (z-score).
`audio/spectrogram/`: STFT/ISTFT (OLA-Griffin-Lim 1984 §III), mel-spectrogram, magnitude/log-mag/power, PNG colourmap render.
`audio/cqt/`: slow per-bin Hann-atom CQT, no inverse, no kernel cache, no Brown-Puckette form, no down-sampling cascade.
`audio/onset/`: energy / spectral-flux / complex-domain / SuperFlux / peak-pick.
`audio/pitch/`: YIN, McLeod (NSDF/MPM), Hermes-SHS, autocorrelation. No pYIN, no SWIPE, no Viterbi voicing.
`audio/segmentation/`: VAD-energy gate, onset/offset, min-silence, merge-close, min-duration.
`audio/separation/`: NMF (Frobenius), FastICA, Wiener (DD), spectral subtraction (Boll), energy-VAD.
`audio/tempo/`: autocorr-of-novelty single-peak BPM.
`audio/beat/`: Ellis-DP only.
`audio/vibration/`: fundamental + harmonic-energy ratio.

Roughly 40 functions. No chroma. No spectral descriptors. No inverse mel. No Griffin-Lim *iteration*. No HPSS. No LUFS. No gammatone. No LPC. No resampler. No fingerprinting hash. No tempogram.

## SOTA by library (2025–2026)

### librosa 0.11 (2025-03-11) — the academic baseline
The most-imported MIR API in the world; almost every paper's "we used librosa…" sentence. v0.11 is mostly bug-fix: Python 3.13 support, matplotlib≥3.5, MFCC `norm` collision fix. The *algorithmic* surface settled at v0.10 and stays the SOTA reference for the **classical kernels**:
- `feature.spectral_*` (centroid, bandwidth, contrast, rolloff, flatness, slope, decrease, spread, kurtosis, RMS, ZCR, polynomial features) — ~12 functions, all 1-line, all missing from reality.
- `feature.chroma_stft / chroma_cqt / chroma_cens / tonnetz` — chroma family, pitch-class profile, Harte tonnetz; entirely missing from reality.
- `feature.melspectrogram / mfcc / poly_features` — present in reality minus liftering, Δ/ΔΔ, Slaney-norm, CMVN.
- `effects.hpss / harmonic / percussive` — Fitzgerald median-filter HPSS, the canonical implementation; missing.
- `decompose.nn_filter` — KNN-based non-local filter; missing.
- `onset.onset_strength / onset_strength_multi / onset_detect` — backbone of every classical onset pipeline.
- `beat.beat_track / plp` — Ellis-DP plus Grosche-Müller predominant local pulse; reality has Ellis but no PLP, no tempogram.
- `segment.recurrence_matrix / recurrence_to_lag / cross_similarity / path_enhance / agglomerative` — SSM/lag/path/agglomerative segmentation; entirely missing.
- `sequence.dtw / rqa / viterbi / transition_*` — DTW, recurrence-quantification, HMM/Viterbi back-end; reality has none.
- `filters.mel / chroma / constant_q / cq_to_chroma / get_window` — filter design; reality only has the un-normalised HTK mel.
- `griffinlim` — iterative phase recovery (the package has the OLA half but not the iteration with momentum).
- `resample` — polyphase via `scipy.signal.resample_poly` and `samplerate`-sinc back-end.

(Source: <https://librosa.org/doc/0.11.0/feature.html>, <https://librosa.org/doc/main/changelog.html>.)

### Essentia 2.1-beta6 (MTG-UPF) — the C++/streaming heavyweight
The MIR de-facto industrial reference; the `MusicExtractor` algorithm computes 100+ low-level + rhythm + tonal features per file. SOTA additions reality is missing:
- Auditory filterbanks: `BarkBands`, `ERBBands`, `MelBands` (HTK and Slaney variants), `GFCC`, `BFCC`.
- Tonal: `HPCP` (Gómez-Bonada harmonic-pitch class profile), `Key`, `KeyExtractor`, `ChordsDetection`, `TuningFrequency`, `Inharmonicity`, `Tristimulus`.
- Rhythm: `RhythmExtractor2013` (state-of-the-art Degara HMM + tempogram), `BeatTrackerMultiFeature`, `BeatsLoudness`, `OnsetRate`, `Danceability`.
- Loudness: `LoudnessEBUR128` (BS.1770 K-weighting + R128 gating + LRA + true-peak), `Loudness` (Vickers 1999), `ReplayGain`.
- Spectral: `SpectralCentroidTime`, `SpectralComplexity`, `RollOff`, `Flatness*`, `SpectralContrast`, `SpectralWhitening`, `SpectralPeaks`, `Dissonance`.
- Voice: `PitchYinFFT` (yinfft variant), `PitchMelodia` (Salamon-Gomez 2012 melodia), `PredominantPitchMelodia`, `PitchSalienceFunction`, `Vibrato`, `LogAttackTime`, `Inharmonicity`.
- Decomposition: `NNLSChroma`, `NMFAlgorithm` with KL/IS divergences, `HPSS` Fitzgerald.
- TF representations: `ConstantQ` (with kernel cache), `NSGConstantQ`, `IIR`, `LPC` (autocorrelation method).
- ML extractors: `TensorflowPredictMusiCNN`, `TensorflowInputMusiCNN` (mel-bands tuned to MusiCNN), `TensorflowPredictVGGish`, `TensorflowPredictEffnetDiscogs` (auto-tagging on Discogs/MTT). The *features* feeding these are pure DSP and in scope; the model layer isn't.

(Source: <https://essentia.upf.edu/algorithms_overview.html>.)

### aubio 0.5α (2025) — the canonical real-time C library
Tiny, focused, and surprisingly canonical for the "what's the SOTA *classical* algorithm" question. All algorithms run in real-time on a frame. SOTA reality lacks:
- **Pitch ladder**: `yin`, `yinfft` (frequency-domain YIN; same answer, faster), `yinfast` (FFT-convolution YIN, O(N log N) vs YIN's O(N²) — reality should adopt this immediately as the YIN drop-in upgrade), `fcomb` (fast harmonic comb), `mcomb` (multi-comb), `specacf` (spectral autocorrelation), `schmitt` (zero-crossing trigger).
- **Onset ladder**: `energy`, `hfc` (Masri 1996 high-frequency content), `complex` (✓ have it), `phase` (Bello-Sandler 2003), `wphase` (weighted phase deviation), `specdiff` (spectral difference), `kl` (Kullback-Leibler), `mkl` (modified KL — Brossier 2006), `specflux` (✓ have it).
- **Beat / tempo**: `aubiotempo` uses spectral-flux→tempogram comb; `aubiotrack` is a real-time DP tracker.
- **Notes / pitched-events**: `aubionotes` outputs MIDI-style note onsets via combined onset+pitch+silence gating. The whole output stream "{onset, pitch, voiced, silence-frame}" pattern is missing.
- **Mel filterbank**: `aubio.filterbank` with HTK/Slaney + Auditory toolbox variants.

(Sources: <https://aubio.org/manual/latest/cli.html>, <https://aubio.org/doc/latest/pitch_8h.html>.)

### torchaudio 2.6 / 2.8 — differentiable PyTorch baseline
Settled API around Spectrogram / MelSpectrogram / MFCC / LFCC / GriffinLim / InverseMelScale / Resample / SpectralCentroid / TimeStretch / PitchShift / SpecAugment / RNNT loss. v2.6 added LFCC (linear-frequency cepstrum). v2.8 prototype: `BarkScale`, `BarkSpectrogram`, `ChromaSpectrogram`, `RoomSimulator` (ISM), `RNNTBeamSearch` and shoebox-RIR convolution. Two patterns from torchaudio are worth porting *in spirit*:
- All transforms are `nn.Module`s with stateful kernels (filterbanks, windows precomputed once); reality could expose a parallel `audio.Pipeline` value type that caches mel/CQT kernels — the "Pistachio 60-FPS" use case in `CLAUDE.md` § "no allocations in hot paths."
- Inversion pairs: `MelSpectrogram` ↔ `InverseMelScale`, `Spectrogram` ↔ `GriffinLim`, `Resample` is invertible. Reality's audio package has the forward path of all three but neither inverse.

(Source: <https://docs.pytorch.org/audio/stable/transforms.html>.)

### openSMILE 3.0 + eGeMAPSv02 — affective-computing standard
The de-facto speech-emotion / voice-quality feature set. eGeMAPS = 88 functionals over LLDs (jitter, shimmer, HNR, F0, formant 1–3 frequencies+bandwidths+amplitudes, log-energy, MFCC1–4, spectral flux, alpha-ratio, Hammarberg index). All are pure DSP — closed-form windowed-frame features post-processed by mean / std / 20–80 percentile / slope / curvature functionals. Reality has zero coverage of jitter, shimmer, HNR, formants, alpha-ratio, Hammarberg, or any "functional over LLD" abstraction. (Sources: <https://www.audeering.com/research/opensmile/>, <https://sail.usc.edu/publications/files/eyben-preprinttaffc-2015.pdf>.)

### audioFlux 0.1.9 — high-coverage C+Python lib
Notable for building *on the same wedge reality is targeting* (zero-allocation C kernels, multi-platform, no PyTorch dep). Coverage: BFT / NSGT / CWT / PWT transforms; Linear / Mel / Bark / ERB / Octave / Log frequency scales; pitch (YIN, CEP, PEF, NCF, HPS, LHS, STFT, FFP); time-stretch + pitch-shift. Zero-dep philosophy is the closest analogue to reality's; the library to benchmark feature *breadth* against. (Source: <https://github.com/libAudioFlux/audioFlux>.)

### Spotify Pedalboard (2025) — JUCE-backed effects engine
Studio-grade effects (compressor, limiter, gate, reverb, chorus, phaser, distortion, IIR/FIR filters, gain, time-stretch via Rubber Band) plus VST3/AU host. Used internally for ML data augmentation and Spotify AI DJ / Voice Translation. The *math* of every effect (bilinear-transform biquad family, JUCE-style state-variable filter, soft-clip distortion shapers, feedback delay-line reverbs, Schroeder/Moorer all-pass diffusers, sidechain envelope followers) is in scope. (Source: <https://spotify.github.io/pedalboard/>.)

### madmom (CPJKU) — RNN-based MIR back-end
Adds the *learned* part on top of librosa-style features: HMM/CRF beat & downbeat trackers, neural onset detectors, neural chord recognition, NN piano transcription. Reality cannot port the *weights* zero-dep, but the *Viterbi / forward-backward* HMM kernels and Dynamic Bayesian Network beat models (Krebs-Böck-Widmer 2015) are pure algorithmic and would chain on top of `audio/onset` to give modern beat tracking without ML inference. (Sources: <https://github.com/CPJKU/madmom>, <https://arxiv.org/abs/1605.07008>.)

## Recent SOTA (2024–2026 papers worth porting math from)

### Pitch / F0 — modern ladder
- **CREPE** (Kim 2018) — supervised CNN; SOTA accuracy on MDB-melody but 77× slower than 2025 alternatives.
- **PESTO** (Riou et al. 2023) — self-supervised CQT + Toeplitz; ≤10 ms latency, < 30 k params; *math-portable* (CQT + a small bilinear classifier).
- **FCPE** (2025-09) — Real-Time Factor 0.0062 on RTX 4090; 5.3× faster than RMVPE, 2.6× faster than PESTO, 77× faster than CREPE; 96.79 % RPA on MIR-1K. Uses cause-effect-aware mel + small transformer.
- **SwiftF0** (2025-08) — distilled lightweight monophonic.
- **pYIN** (Mauch-Dixon 2014) — still the *classical* SOTA; Viterbi over YIN-derived pitch posteriors. Reality's YIN→pYIN upgrade is ~200 LOC and closes the classical gap.

(Sources: FCPE arXiv:2509.15140; PESTO; SwiftF0 arXiv:2508.18440; CREPE ICASSP 2018.)

### Source separation — modern ladder
- **Demucs v4 / HT-Demucs** (Défossez 2022; Rouard 2023) — hybrid time-domain + spectrogram U-Net with cross-domain Transformer; 9.20 dB SDR on MUSDB-HQ with extra data; the open-source SOTA stem splitter.
- **Band-split RNN / BSRNN** (Luo-Yu 2023) and **MDX-Net** family — top SDX'24 winners.
- **xumx-sliCQ-V2** (2024) — sliCQT-domain U-Net + differentiable Wiener-EM step; 4.4 dB total SDR with 60 MB weights; relevant because the *sliCQT* transform itself is a pure-math invertible nonstationary Gabor frame and would slot directly between `audio/cqt` and `audio/separation`.
- **Music-Source-Separation foundation-model wave**: HTDemucs-FT, MDX23-NetC, CWS-PResUNet — all delegate the *transform* to STFT or sliCQT and the *masking* to neural masks.

(Sources: Demucs arXiv:2211.08553; sliCQT music-demixing arXiv:2112.05509; xumx-sliCQ-V2 GitHub sevagh/nsgt.)

### Differentiable DSP
- **DDSP** (Engel et al. ICLR 2020) — harmonic+filtered-noise synth + linear-time-varying filter, all written as TF/JAX ops with closed-form gradients. The *math* (additive sinusoidal synth with anti-aliased phase accumulators, FIR-via-FFT linear-time-varying filter, exp-sigmoid loudness map, multi-scale STFT spectral-loss) is 100 % zero-dep portable. JAX-port active; TensorFlow port archived (Magenta sunset 2024); MAWF (2023) extends to 48 kHz real-time.
- **Differentiable IIR / SVF state-variable filters** — Kuznetsov 2024, ICASSP 2024 *Differentiable Signal Processing With Black-Box Audio Effects*. Backprop-through-time with stable IIR formulations.
- **Differentiable Wiener-EM** (xumx-sliCQ-V2) — gradients through expectation-maximization steps.
- **DiffFDN** for reverberators (also covered in 003-acoustics-sota).

(Sources: DDSP arXiv:2001.04643 / OpenReview B1x1ma4tDr; Frontiers DDSP review 2023; CCRMA DDSP talk.)

### Cochlear / auditory — CARFAC-JAX (Lyon 2024)
The Lyon CARFAC v2 release (April 2024) ships pure NumPy and JAX implementations of the cascade-of-asymmetric-resonators fast-acting-compression cochlear model with full automatic differentiation. Real-time factor ~0.02–0.11 for forward audio; ~0.5–1.5 with gradient. Used to fit personalized hearing-loss profiles by gradient descent on outer-hair-cell parameters. The CARFAC math (cascaded biquads, AGC ladder, BM/IHC/OHC stages) is closed-form; reality could port it on top of `signal/biquad` (which agent 007 noted is also missing). (Source: <https://arxiv.org/abs/2404.17490>; <https://github.com/google/carfac>.)

### Neural audio codecs
- **SoundStream** (Zeghidour 2021) — RVQ + Soundstream encoder/decoder; *additive-noise* differentiability trick; 3 kbps speech / 6 kbps music.
- **EnCodec** (Défossez 2022) — adapted RVQ with Gumbel-Softmax differentiability; mono 24 kHz / stereo 48 kHz; the dominant open-source codec for LLM audio I/O.
- **QINCO / QINCO2** (2024) — implicit neural codebooks; trainable codebook adaptation.
- The *RVQ kernel* (residual vector quantizer with k-means initialization and EMA codebook updates) is ~150 LOC of pure linear algebra and would slot into `linalg` or a new `audio/codec` package; the encoder/decoder *weights* are not portable but the structural math is.

(Sources: SoundStream arXiv:2107.03312; EnCodec; Kyutai codec explainer.)

### Audio fingerprinting (post-Shazam)
- **NeuralFP** (Chang-Lee 2021) — contrastive-learning SOTA.
- **PeakNetFP** (2025-06) — PointNet++ on sparse spectral peaks; 100× fewer parameters than NeuralFP, comparable accuracy.
- **VLAFP** (2025) — variable-length contrastive fingerprints.
- **Foundation-model fingerprinting** (Nov 2025) — uses pre-trained music FMs for embeddings.
- The classical baseline (Shazam landmark hashing 2003 + Chromaprint folded chroma 2010) remains in scope and unsolved in reality.

(Sources: PeakNetFP arXiv:2506.21086; arXiv:2511.05399; Wang Shazam 2003.)

### NSGT / sliCQT — invertible CQT
- Framework: Holighaus, Dörfler, Velasco, Grill (2013) "A framework for invertible, real-time constant-Q transforms" — perfect-reconstruction error 1.6×10⁻¹⁵.
- 2025 PyTorch impl: `cqt-nsgt-pytorch` makes it differentiable.
- **sliCQT** = sliced real-time variant; bounded delay, linear processing time.
- This is the SOTA replacement for the slow `audio/cqt/cqt.go` reality currently has, and provides perfect inverse, which `audio/cqt` lacks entirely.

(Sources: arXiv:1210.0084; sevagh/nsgt PyTorch; xumx-sliCQ thesis McGill 3197xr696.)

### Loudness — ITU-R BS.1770-5 (Nov 2023)
The 5th revision of BS.1770 is now the binding standard. Algorithm: K-weighting (1.5 kHz +4 dB shelf followed by 38 Hz high-pass — two biquads at the working sample-rate) → mean-square per channel → channel-weight sum → 400 ms momentary / 3 s short-term / gated integrated LUFS (–70 LUFS absolute gate, –10 LU relative gate). True-peak: 4× polyphase oversample then peak. Loudness-Range (EBU Tech 3342): 95th–10th-percentile of gated short-term loudness. Reality's `acoustics.AWeighting` is the only weighting curve in the entire repo — K-weighting (BS.1770), B/C/D/Z weighting, and ITU-R 468 weighting are all missing. (Source: ITU-R BS.1770-5 PDF; <https://en.wikipedia.org/wiki/EBU_R_128>.)

## Math gaps SOTA exposes (groupings reality could absorb zero-dep)

| Math family | Where used (SOTA) | Currently in reality? |
|---|---|---|
| Triangular Slaney-norm + dB-mel filterbank | librosa default, every neural vocoder | partial (HTK natural-log only) |
| Inverse mel via NNLS / pseudo-inverse | torchaudio `InverseMelScale`, librosa `mel_to_stft` | No |
| Iterative Griffin-Lim with momentum | torchaudio, librosa, every magnitude-only synthesis | OLA only, no iteration |
| Brown-Puckette efficient CQT + kernel cache | librosa, Essentia, audioFlux, torchaudio | No (slow per-bin only) |
| Down-sampling-cascade CQT + iCQT | Schörkhuber-Klapuri 2010, librosa | No |
| NSGT / sliCQT invertible non-stationary Gabor | sevagh/nsgt, xumx-sliCQ-V2 | No |
| Spectral descriptors (centroid/bandwidth/rolloff/flatness/contrast/slope/decrease/spread/kurtosis/crest/RMS/ZCR) | librosa `feature.spectral_*`, Essentia `lowlevel.*` | No |
| Chroma-STFT + Chroma-CQT + CENS + tonnetz | librosa, Essentia HPCP | No |
| NNLS-Chroma (chordino) | Mauch-Dixon 2010, Essentia | No (no NNLS solver in reality) |
| Krumhansl-Schmuckler / Temperley key profiles | Essentia KeyExtractor, every key-finder | No |
| Median-filter HPSS | librosa effects.hpss, Essentia | No |
| REPET / REPET-SIM | Rafii-Pardo 2013 | No |
| RPCA-as-audio (SVT) | Huang-Hsu 2012 | No (no SVT, no nuclear norm) |
| β-NMF with Itakura-Saito divergence | Essentia, Févotte 2009 | No (only Frobenius) |
| LPC (Levinson-Durbin / Burg) + LSF / PARCOR / LPC-cepstrum | Essentia, every speech codec, eGeMAPS | No |
| Gammatone (Patterson 1992 / Slaney all-pole) + ERB filterbank | Essentia, torchaudio prototype, CARFAC | No |
| Bark scale + Bark filterbank | Essentia, audioFlux, openSMILE | No |
| GFCC / BFCC / PNCC / PLP / RASTA-PLP | Essentia, Kaldi, ASR baselines | No |
| Pre-emphasis / de-emphasis / DC blocker | Universally used | No |
| Δ / ΔΔ / liftering / CMVN | Every ASR pipeline | No |
| Self-similarity matrix + Foote novelty | librosa segment, MIR structure analysis | No |
| Subsequence DTW | librosa, Essentia, query-by-humming | sequence has classical DTW; subseq variant not exposed |
| HMM / Viterbi / forward-backward | madmom, pYIN voicing, key tracking | No (probably belongs in `prob` not `audio`) |
| Tempogram (Fourier + autocorr) + cyclic | Grosche-Müller 2010, Ellis 2007 | No |
| Davies-Plumbley two-state beat tracker | Essentia BeatTrackerMultiFeature | No |
| pYIN + Viterbi voicing | librosa, sonic-annotator | No |
| SWIPE / SWIPE′ pitch | Camacho-Harris 2008 | No |
| MELODIA salience function | Salamon-Gómez 2012, Essentia PitchMelodia | No |
| Polyphase / sinc-Kaiser / Farrow resampler | torchaudio Resample, libsamplerate, soxr | No |
| Phase-vocoder time-stretch + WSOLA / PSOLA | Pedalboard (Rubber Band), all DAWs | No |
| MDCT / IMDCT + KBD / sine windows | Every codec MP3/AAC/Opus/Vorbis | No |
| K-weighting (BS.1770-5) + LUFS gating + LRA + true-peak | Essentia LoudnessEBUR128, ffmpeg `loudnorm`, every broadcast | No |
| Bilinear-transform biquad designer + SOS | Pedalboard, JUCE, all IIR design | No (also missing in `signal/`) |
| State-variable filter (SVF) | Pedalboard / Cytomic Andy | No |
| RVQ encoder + EMA codebook | SoundStream, EnCodec, every neural codec | No |
| Additive harmonic synth (DDSP-style) | Magenta DDSP, MAWF | No |
| Multi-scale STFT spectral loss | DDSP, MelGAN, HiFi-GAN | No |
| FIR-via-FFT linear-time-varying filter | DDSP filtered-noise branch | No (have static convolution) |
| Cascade-of-asymmetric-resonators (CARFAC stages) | google/carfac, CARFAC-JAX | No |
| Fitzgerald PSF / NSDF tail bias correction | aubio yinfast, agent 006 finding | partial (NSDF tail unbias is in 006) |
| Shazam landmark peaks + pair-hash | Wang 2003, dejavu, audfprint | No |
| Chromaprint folded-chroma quantization | AcoustID | No |
| Jitter / shimmer / HNR / formants | openSMILE eGeMAPS, Praat | No |

## Recommendations (impact-per-line, plausible zero-dep Go)

**Tier A — close the classical-MIR gap (the librosa/Essentia/aubio parity layer; ~3,000 LOC, all foundational, all golden-file-friendly):**

1. **Spectral descriptors batch** — centroid, bandwidth, rolloff, flatness, contrast (Jiang 2002 7-band), slope, decrease, spread, kurtosis, crest, ZCR, RMS-dB. ~12 functions × ~25 LOC. Unlocks every classical MIR consumer.
2. **Chroma family** — chroma_stft, chroma_cqt (after Brown-Puckette CQT lands), CENS (Müller-Ewert 2011), tonnetz (Harte 2006), Krumhansl-Schmuckler key. ~250 LOC. Turns reality into a *real* MIR library.
3. **Inverse mel + iterative Griffin-Lim with momentum** — closes the synthesis loop torchaudio + librosa expose. ~150 LOC. Mandatory for any neural-vocoder consumer.
4. **Brown-Puckette efficient CQT + kernel cache + iCQT** (Schörkhuber-Klapuri 2010 down-sampling cascade) — closes the CQT loop. ~400 LOC. Replaces the slow `audio/cqt/cqt.go` per-bin form.
5. **Gammatone (Slaney all-pole) + ERB filterbank + Bark filterbank + GFCC + BFCC** — the auditory-model wedge that opens cochlear modelling and Essentia-grade timbre. ~400 LOC.
6. **LPC (Levinson-Durbin + Burg) + LSF + PARCOR + LPC-cepstrum + pre-emphasis + de-emphasis + DC blocker** — speech-coding bedrock + every ASR front-end's first three lines. ~300 LOC.
7. **K-weighting (BS.1770-5) + LUFS momentary/short/integrated + LRA + true-peak (4× oversample)** — broadcast-grade metering. ~250 LOC. (Overlaps `acoustics-missing` 002 rec 4.)
8. **Δ / ΔΔ / liftering / Slaney-norm-mel / CMVN** — single-file MFCC-completeness pack. ~120 LOC.
9. **pYIN with Viterbi voicing** — modern monophonic-pitch SOTA classical. ~200 LOC on top of YIN.
10. **YIN → yinfast (FFT-convolution form)** — keeps API stable, drops O(N²) → O(N log N). ~80 LOC.
11. **Median-filter HPSS** (Fitzgerald 2010) — 30 LOC; the canonical HPSS reference.
12. **REPET / REPET-SIM** — Rafii-Pardo 2013. ~150 LOC.
13. **β-NMF with Itakura-Saito divergence + sparse NMF + IBM/IRM/PSM masks + bss_eval (SDR/SIR/SAR/SI-SDR)** — closes the separation evaluation loop. ~300 LOC.

**Tier B — invertible / sharp / standards-grade transforms (~1,500 LOC):**

14. **Polyphase / sinc-Kaiser / Farrow resampler** — sample-rate conversion is currently a hole. ~250 LOC.
15. **Phase-vocoder time-stretch + WSOLA / PSOLA** — time-stretch / pitch-shift trio. ~300 LOC.
16. **MDCT / IMDCT + KBD-symmetric + sine window** — codec-grade lapped transform. ~200 LOC.
17. **NSGT / sliCQT invertible non-stationary Gabor** — perfect-reconstruction CQT; the modern replacement for `audio/cqt`. ~400 LOC.
18. **Tempogram (Fourier + autocorr) + cyclic tempogram + Davies-Plumbley two-state + PLP** — closes the tempo/beat ladder. ~400 LOC.
19. **Self-similarity + Foote novelty + subsequence DTW + path-enhance** — segmentation/matching trio. ~250 LOC.
20. **Reassigned spectrogram + synchrosqueezing** — sharper TF localisation. ~250 LOC.

**Tier C — three differentiable showcases that exploit reality's autodiff (~800 LOC):**

21. **DDSP harmonic+filtered-noise synth** — additive sinusoidal oscillator with anti-aliased phase accumulator + linear-time-varying FIR-via-FFT filter + exp-sigmoid loudness map + multi-scale STFT loss. Pure math, fits autodiff. ~250 LOC. Reality could ship the *math* of DDSP without ever depending on TensorFlow / JAX.
22. **CARFAC v2 cochlear model** — cascaded asymmetric-resonator biquads + AGC ladder + IHC half-wave-rectifier. ~400 LOC including biquad. The first "world-class auditory model" that reality could host.
23. **RVQ residual vector quantizer + EMA codebook + commit loss** — the SoundStream/EnCodec encoder kernel. ~150 LOC. Slots into `linalg` or a new `audio/codec`. Composes with autodiff for end-to-end training.

**Tier D — fingerprinting two-pack (overlaps agent 007 rec 19; ~400 LOC):**

24. **Shazam landmark hashing** (Wang 2003) — 2-D max-filter peak-pick + anchor/target pair-hash + database lookup primitive.
25. **Chromaprint folded-chroma quantization** (AcoustID) — 12-band log-quantized chroma fingerprint.

**Out of scope** (require model weights, GPU, file containers, or perceptual models that aren't pure math): MusiCNN/VGGish/Effnet-Discogs auto-tagging weights; PEAQ/PESQ/POLQA proprietary perceptual models (PEAQ algorithm itself is open but huge); Demucs/HT-Demucs/MDX-Net pretrained masks; CREPE/PESTO/FCPE pretrained pitch models; SOFA/AES69 NetCDF parsing.

## Sources

- librosa 0.11 docs: <https://librosa.org/doc/0.11.0/feature.html>, <https://librosa.org/doc/main/changelog.html>
- Essentia 2.1 algorithms: <https://essentia.upf.edu/algorithms_overview.html>, <https://essentia.upf.edu/streaming_extractor_music.html>, <https://essentia.upf.edu/models.html>
- aubio 0.4.9 / 0.5α: <https://aubio.org/manual/latest/cli.html>, <https://aubio.org/doc/latest/pitch_8h.html>
- torchaudio 2.6 / 2.8: <https://docs.pytorch.org/audio/stable/transforms.html>, <https://docs.pytorch.org/audio/2.6.0/transforms.html>
- openSMILE / eGeMAPS: <https://www.audeering.com/research/opensmile/>, <https://audeering.github.io/opensmile-python/>, <https://sail.usc.edu/publications/files/eyben-preprinttaffc-2015.pdf>
- audioFlux: <https://github.com/libAudioFlux/audioFlux>, <https://libaudioflux.github.io/>
- Spotify Pedalboard: <https://github.com/spotify/pedalboard>, <https://spotify.github.io/pedalboard/>
- madmom: <https://github.com/CPJKU/madmom>, <https://arxiv.org/abs/1605.07008>
- DDSP (Engel 2020): <https://arxiv.org/abs/2001.04643>, <https://github.com/magenta/ddsp>, <https://magenta.withgoogle.com/ddsp>
- DDSP review (Frontiers 2023): <https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2023.1284100/full>
- CARFAC v2 (Lyon 2024): <https://arxiv.org/abs/2404.17490>, <https://github.com/google/carfac>
- Demucs (Défossez 2022, Rouard 2023): <https://arxiv.org/abs/2211.08553>, <https://github.com/facebookresearch/demucs>
- xumx-sliCQ / xumx-sliCQ-V2: <https://arxiv.org/abs/2112.05509>, <https://github.com/sevagh/xumx-sliCQ>
- NSGT framework (Holighaus et al. 2013): <https://arxiv.org/abs/1210.0084>, <https://github.com/grrrr/nsgt>
- SoundStream (Zeghidour 2021): <https://arxiv.org/abs/2107.03312>
- EnCodec (Défossez 2022): <https://github.com/facebookresearch/encodec>
- Kyutai codec explainer (2025): <https://kyutai.org/codec-explainer>
- FCPE (2025): <https://arxiv.org/abs/2509.15140>
- PESTO (2023) / SwiftF0 (2025): <https://arxiv.org/abs/2508.18440>
- CREPE (2018): <https://arxiv.org/abs/1802.06182>
- PeakNetFP (2025): <https://arxiv.org/abs/2506.21086>; foundation-model fingerprinting (2025): <https://arxiv.org/abs/2511.05399>
- ITU-R BS.1770-5 (2023): <https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I!!PDF-E.pdf>
- EBU R128: <https://en.wikipedia.org/wiki/EBU_R_128>
- ICASSP 2025 highlights: <https://www.paperdigest.org/2025/04/icassp-2025-papers-highlights/>; <https://2025.ieeeicassp.org/paper-awards/>
- NeurIPS 2024 Audio Imagination workshop: <https://neurips.cc/virtual/2024/workshop/84706>
- Shazam (Wang 2003), Chromaprint, Chromaprint patent expiry, AcoustID — referenced in agent 007.
- Müller (2015) *Fundamentals of Music Processing* (chroma, CENS, tempogram, SSM, DTW, key).
- Schörkhuber & Klapuri (2010) "Constant-Q transform toolbox for music processing."
- Fitzgerald (2010) "Harmonic/Percussive separation using median filtering" DAFx-10.
- Rafii et al. (2025) "30+ Years of Source Separation Research" arXiv:2501.11837.

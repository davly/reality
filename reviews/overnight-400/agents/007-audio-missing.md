# 007 — audio: missing canonical algorithms

## Headline
The `audio` package has the speech/bioacoustic front-end (mel filterbank, MFCC, MFCC-DCT, STFT/ISTFT, mel-spectrogram, spectral-flux/SuperFlux/complex-domain onsets, YIN/MPM/SHS/autocorr pitch, onset-autocorr tempo, simple DP beat tracker, NMF/Wiener/spectral-subtraction/FastICA/energy-VAD separation, Welford fingerprint) and a basic pure-Python-style CQT — and is missing essentially everything in three layers: (1) the **musicology layer** (chroma family, NNLS-Chroma, key estimation, tonnetz, HCQT, VQT, beat-synchronous features), (2) the **broadcast/standards layer** (ITU-R BS.1770 K-weighting → LUFS/LRA, true-peak, BS.1387 PEAQ, EBU R128 gating, full LUFS metering pipeline), and (3) the **modern speech/codec/auditory layer** (gammatone + ERB filterbanks, BFCC/GFCC/PNCC, LPC/LSF, perceptual linear prediction, MDCT/IMDCT, KBD windows, polyphase/sinc/Farrow resampling, phase-vocoder time-stretch, PSOLA/WSOLA, Griffin-Lim phase recovery, harmonic/percussive median-filter separation, REPET, RPCA-as-audio, Chromaprint and Shazam-peak fingerprinting, NLMS / frequency-domain block-LMS adaptive filters, spectral-noise-gating with prior). YIN exists but pYIN (the modern probabilistic descendant) does not. Beat tracking exists but only the Ellis/Daudet DP variant — no spectral-flux→tempogram→DP cascade and no Davies-Plumbley two-state model. There is no **inverse mel** (back-projection / NNLS), no **chroma**, no **tonnetz**, no **CENS**, no **constant-Q chroma**, and no **spectral descriptor** family (centroid, bandwidth, contrast, rolloff, flatness, slope, decrease, spread, kurtosis) which are the librosa/Essentia "spectral.*" workhorses. The CQT itself is the slow non-Brown-Puckette O(K·N) reference form (no efficient down-sampling cascade) and has no inverse, no kernel cache, and no log-frequency spectrogram product.

## What exists today

`audio/` (root):
- `melscale.go` — `HzToMel` (Slaney 1127·ln form), `MelToHz`, triangular `MelFilterbank` (un-normalised HTK form), `PowerSpectrum`, `ApplyFilterbank`.
- `mfcc.go` — `LogMelEnergies`, `MFCC` (orthonormal DCT-II), `FrameMFCC` composition.
- `fingerprint.go` — Welford-online `Fingerprint`, `UpdateFingerprint`, `FingerprintVariance`, `FingerprintMahalanobis` (diagonal), `BestMatch`, `MergeFingerprints` (Chan-Golub-LeVeque parallel).
- `degradation.go` — z-score-based temporal pattern shift detector (Welford-driven).

`audio/spectrogram/`:
- `Compute` (STFT, ceil framing — see 006), `Inverse` (OLA Griffin-Lim 1984 §III), `MelSpectrogram` (composes STFT→power→filterbank), magnitude/log-mag/power flavours, PNG `visualise` with Plasma/Magma/Viridis/Inferno colourmaps.

`audio/cqt/`:
- `QualityFactor`, `BinFrequency`, `BinFrequencies`, `WindowLength`, `CQT` (slow per-bin Hann-atom form), `Magnitude`, `PeakBin`. No inverse, no kernel cache, no Brown-Puckette FFT-based form, no down-sampling cascade.

`audio/onset/`:
- `EnergyOnset`, `SpectralFluxOnset` + `SpectralFluxStrength`, `ComplexDomainOnset`, `SuperFlux`, `PickPeaks` + `PickPeaksAdaptive`. 3-detector cross-validation test exists (commit 6a55bb4).

`audio/pitch/`:
- `Yin` (de Cheveigné & Kawahara 2002), `McLeodPitchMethod` (NSDF-based), `SubharmonicSummation` (Hermes 1988), `Autocorrelation` (windowed naive ACF). No pYIN, no SWIPE/SWIPE′, no PEFAC, no probabilistic / HMM viterbi tracker.

`audio/onset/peak_picking.go` — generic peak-pick utilities.

`audio/segmentation/`:
- `vad_based.go` (energy gate), `onset_offset.go` (rising/falling-edge), `min_silence.go`, `merge_close.go`, `min_duration.go`. No HMM/CRF segmentation, no boundary detector, no novelty-curve segmentation.

`audio/separation/`:
- `nmf.go` (Lee-Seung Frobenius multiplicative), `ica.go` (FastICA tanh, symmetric decorrelation), `wiener.go` (decision-directed gain), `spectral_subtraction.go` (Boll 1979 magnitude form with α/β floor), `vad.go` (energy-only voiced detector).

`audio/tempo/`:
- `Estimate` — autocorr-of-onset-novelty single-peak BPM. No tempogram, no Fourier-tempogram, no comb-filter tempogram, no tempo-curve tracking.

`audio/beat/`:
- `Track` — DP beat tracker (Ellis 2007 § BeatRoot-style log²-period penalty). No Davies-Plumbley, no two-state, no downbeat / metre estimation.

`audio/vibration/`:
- `Fundamental`, `HarmonicEnergyRatio` — narrow mechanical vibration helpers (substrate of Dipstick).

Cross-package note: `signal/window.go` provides only Hann, Hamming, Blackman (symmetric form). No Kaiser, KBD, Gaussian, Tukey, flat-top, Nuttall, Blackman-Harris, or Slepian. `signal/filter.go` has Convolve / MovingAverage / EMA / MedianFilter only — no Butterworth / Chebyshev / elliptic / FIR-design / biquad / SOS / lattice / all-pass primitives.

## Tier-1: high-impact, broadly expected

**Inverse / synthesis path**
- **Inverse-mel filterbank** — solve `M·X ≈ Y` (NNLS or pseudoinverse) so a mel spectrogram can be back-projected to STFT magnitude (torchaudio `InverseMelScale`, librosa `mel_to_stft`, used in every neural vocoder and most synthesis chains). Currently impossible from `audio` alone.
- **Griffin-Lim phase reconstruction** — iterative magnitude-only inversion (Griffin & Lim 1984 §IV; the `algorithm 4` version with momentum is the modern default). The package has the OLA half but not the iterative refinement.
- **Inverse CQT** (iCQT) — kernel-pseudoinverse synthesis (Schörkhuber & Klapuri 2010); currently no inverse at all.
- **MDCT / IMDCT** with KBD-symmetric and sine windows — the lapped transform behind MP3, AAC, AC-3, Opus, Vorbis, and (relevant to Reality) the MQA-style split that Pistachio/Folio audio paths may need.
- **Phase-vocoder time-stretch** — `STFT → angle-unwrap-and-rescale → ISTFT`. Laroche-Dolson 1999 (`PhaseLockedVocoder`); the standard pitch-preserving stretch.
- **PSOLA / WSOLA** — pitch-synchronous and waveform-similarity overlap-add. The standard speech time-stretch and pitch-shift primitives (Moulines & Charpentier 1990, Verhelst 1993).

**Spectral descriptors (the librosa `feature.spectral.*` and Essentia `lowlevel.*` workhorses)**
- **SpectralCentroid**, **SpectralBandwidth**, **SpectralContrast** (Jiang 2002 7-band), **SpectralRolloff** (typical 0.85), **SpectralFlatness** (Wiener entropy), **SpectralSlope**, **SpectralDecrease**, **SpectralSpread**, **SpectralKurtosis**, **SpectralCrest**, **ZeroCrossingRate**, **RMS** (frame-energy in dB). These are 1-line each in librosa and are *the* feature set for almost every classical MIR classifier.
- **Spectral entropy** and **spectral entropy of subbands** — used in cough/snore/heart-sound and bioacoustic literature.
- **Polynomial features** (`librosa.feature.poly_features`) — fits per-frame log-magnitude polynomial, a compact spectral-shape descriptor.

**Mel / mel-cepstrum extras**
- **Slaney-normalised mel filterbank** (`norm='slaney'`) — area-normalised so each filter sums to a constant; default in librosa and most published MFCC pipelines. Currently only the un-normalised HTK form is available.
- **Liftering** (sinusoidal cepstral lifter, HTK `cepLifter=22`) — improves discrimination on speech MFCCs; standard for ASR.
- **Δ and ΔΔ MFCC** (Furui 1986, regression-based on (2W+1)-window) — required for any speech-recognition or speaker-ID pipeline. No delta op anywhere in the package.
- **CMS / CMVN** — cepstral mean (and variance) subtraction per utterance (Atal 1974, Viikki & Laurila 1998). The standard channel-equalisation step before any classifier.
- **Mel filterbank in dB / log10 form** — librosa default; the natural-log form is HTK-only.

**Constant-Q / variable-Q family**
- **Brown-Puckette efficient CQT** (1992) — precomputed sparse spectral kernels + single FFT per hop instead of per-bin convolution. Closes the O(K·N) → O(N log N) gap.
- **Down-sampling-cascade CQT** (Schörkhuber & Klapuri 2010) — recursive halving of `sr` each octave; the standard fast CQT.
- **Variable-Q Transform (VQT)** — Klapuri 2014; γ-tuneable Q at low frequencies for sub-bass detail (librosa `vqt`).
- **Hybrid CQT (HCQT)** — multi-harmonic-stack CQT (Bittner et al 2017 PESTO/PYIN-CREPE input).
- **NSGT** — Non-Stationary Gabor Transform (Holighaus et al. 2013) — invertible, fully time-frequency-tunable, and the basis for the slicqt/NSGT-based source separation papers since 2020.
- **CQT-spectrogram** with proper time-axis and **constant-Q chromagram (`chroma_cqt`)**.

**Chroma family — entirely missing today**
- **Chroma-STFT** (`chroma_stft`) — pitch-class folding from STFT magnitudes; the simplest chromagram.
- **Chroma-CQT** (`chroma_cqt`) — fold a CQT log-frequency spectrogram into 12 pitch classes; the standard variant.
- **CENS** — Chroma Energy-Normalized Statistics (Müller-Ewert 2011) — robust to dynamics and timbre; used in audio matching and version-ID.
- **NNLS-Chroma** (Mauch & Dixon 2010) — non-negative-least-squares deconvolution against a tonal-harmonic template; the SOTA classical chroma; used in chordino.
- **Pitch-class profile (PCP)** (Fujishima 1999) — historically the original chromagram; still cited.
- **Tonnetz** (Harte-Sandler-Gasser 2006) — 6-D harmonic-network projection of chroma. One librosa call; trivial to port.

**Key / harmony**
- **Krumhansl-Schmuckler key-finding** (1990) — correlation of average chroma against major/minor profiles; 30 lines, exact closed form.
- **Temperley key profiles** and **Albrecht-Shanahan profiles** — the modern alternatives.
- **Chord-template matching** (major/minor/dim/aug 24-template HMM-free baseline).

**Tempo / beat — modern ladder**
- **Tempogram** — Fourier-tempogram (Grosche-Müller 2010) and autocorrelation-tempogram (Ellis 2007); 2-D (lag, time) representation that turns single-tempo estimation into multi-tempo tracking.
- **Cyclic tempogram** (Müller 2011) — octave-folded tempogram for half/double-time invariance.
- **Davies-Plumbley two-state beat tracker** (2007) — the "induction → tracking" two-stage HMM-free beat tracker that BeatRoot pioneered; complementary to the existing Ellis-DP.
- **Predominant local pulse (PLP)** (Grosche-Müller 2010) — sinusoidal-overlay pulse curve used by mir_eval as a robustness baseline.
- **Onset autocorrelation with comb-filter weighting** (Klapuri-Eronen-Astola 2006) — better metre handling than the bare autocorrelation in `audio/tempo`.
- **Downbeat / metre estimation** (Goto 2001, Davies-Plumbley 2009).
- **Beat-synchronous feature averaging** — `librosa.util.sync` — one of the most-used helpers; applies a reduction (mean / median) to a feature matrix on beat or segment boundaries.

**Pitch — modern ladder**
- **pYIN** (Mauch-Dixon 2014) — probabilistic YIN with Viterbi voicing; the modern monophonic SOTA classical pitch tracker (sox, librosa, sonic-annotator). Mandatory next step after the existing YIN.
- **SWIPE / SWIPE′** (Camacho-Harris 2008) — sawtooth-waveform-inspired pitch estimator; common alternative to YIN.
- **PEFAC** (Gonzalez-Brookes 2014) — pitch-estimation-filter with amplitude-compression; robust in low SNR.
- **Probabilistic monophonic-multipitch (PYIN-PFP)** and **predominant-melody extraction** (Salamon-Gomez 2012 MELODIA).

**Voice activity / energy**
- **Spectral-flatness VAD** (Ramirez 2007) — better than the existing pure-energy gate for noisy environments.
- **Long-term spectral divergence VAD** (Ramirez 2004) — the standard ETSI alternative.
- **WebRTC VAD-style** sub-band logistic-regression (the four-mode aggressiveness setting). Can be reimplemented with frozen coefficients.
- **Statistical-model VAD** (Sohn-Sung-Kim 1999) — likelihood-ratio Gaussian model; the basis of every modern energy-and-spectrum VAD.
- **Cumulative-energy / RMS-in-dB envelope** (frame-RMS in dB-FS) — trivial but missing.

**Loudness, level, true-peak (broadcast standards)**
- **K-weighting filter** (BS.1770-4) — pre-filter (1.5 kHz shelf) + RLB (high-pass at 38 Hz). Two biquads; canonical.
- **LUFS / LKFS integrated, momentary, short-term** with absolute-and-relative gating (BS.1770-4 §5).
- **Loudness Range (LRA)** (EBU Tech 3342).
- **True-peak meter** — 4× polyphase oversample then peak-find (BS.1770 Annex 2).
- **Sample-peak** with proper inter-sample interpolation.
- **DR / PLR / PSR** (loudness-vs-peak descriptors used in mastering analysis).
- **Replay-Gain v1/v2** (`gainstation` / `mp3gain` algorithms).
- **PEAQ** (BS.1387) — basic vs advanced perceptual evaluation of audio quality; complex but algorithmic.
- **PESQ** (P.862) and **POLQA** (P.863) — perceptual evaluation of speech quality. PESQ is patent-encumbered for commercial use but the algorithm is published.
- **STOI / ESTOI** — short-time objective intelligibility; canonical post-2010 speech-enhancement metric.
- **SDR / SIR / SAR / SI-SDR** — `bss_eval`-style source separation quality metrics (Vincent-Gribonval-Févotte 2006 / Le Roux 2019); needed to evaluate the existing `separation` package.

**Auditory / psychoacoustic filterbanks**
- **Gammatone filterbank** — Patterson 1992; 4th-order cascade or the Slaney all-pole IIR form (Apple Auditory Toolbox 1993). The de-facto auditory model used in every cochlear-modelling and GFCC pipeline.
- **Gammachirp** (Irino-Patterson 1997) — frequency-glide variant; better psychophysical fit at high levels.
- **ERB filterbank** (Glasberg-Moore 1990) and **ERB-rate scale** — `21.4·log10(0.00437·f+1)`. Standard alternative to mel; superior for speech.
- **Bark scale** — Zwicker critical bands; `13·atan(0.00076f) + 3.5·atan((f/7500)²)`.
- **Bark filterbank** — triangular or trapezoidal critical-band spectra.
- **Cochleagram** — gammatone-filtered spectrogram; the auditory analogue of the mel-spectrogram.
- **Loudness model** in sones / phons (ISO 532-1 Zwicker, ISO 532-2 Moore-Glasberg). Some of this overlaps with the `acoustics` review (002).

**Cepstral family beyond MFCC**
- **BFCC** — Bark-frequency cepstral coefficients (Davis-Mermelstein-style with Bark filterbank). Better for percussive content.
- **GFCC** — gammatone-frequency cepstrum. The Essentia music-extractor primary timbre feature.
- **PNCC** — power-normalised cepstral coefficients (Kim-Stern 2016). The robust-ASR successor to MFCC.
- **PLP** — Perceptual Linear Prediction (Hermansky 1990); Bark filterbank → cube-root → autoregressive LPC → cepstrum. Used in Kaldi as the primary speech feature for two decades.
- **RASTA-PLP** (Hermansky-Morgan 1994) — bandpass-filter the log-spectrum trajectory; the standard channel-robust ASR feature.
- **LFCC** — linear-frequency cepstral coefficients; used in speaker verification.
- **CQCC** — constant-Q cepstral coefficients (Todisco 2017) — the standard anti-spoofing feature.
- **Cepstral pitch detection** (Noll 1967) — log-magnitude IFFT peak.

**Linear prediction**
- **LPC analysis** — Levinson-Durbin recursion (autocorrelation method), Burg method, covariance method.
- **PARCOR / Reflection coefficients** — lattice form of LPC.
- **LSF / LSP** — Line Spectral Frequencies (Itakura 1975); the GSM/AMR/MELP transmission form.
- **LPC ↔ cepstrum** conversion.
- **Itakura distortion** and **log-likelihood ratio**.
- **LPC-residual** synthesis (the source-filter model).

**Source separation primitives — beyond NMF/Wiener/SS/ICA**
- **Median-filter HPSS** (Fitzgerald 2010) — horizontal median for harmonic, vertical median for percussive; one numpy line; *the* reference HPSS.
- **REPET** (Rafii-Pardo 2013) — repeating-pattern via beat-period folding.
- **REPET-SIM** (Rafii-Pardo 2013) — generalised via similarity matrix (no fixed period).
- **RPCA-as-audio** (Huang-Hsu 2012) — singular-value-thresholding RPCA on magnitude spectrogram → low-rank (music) + sparse (vocal). Reality has no SVT, no nuclear-norm, no soft-threshold yet.
- **β-NMF** with Itakura-Saito divergence (Févotte-Bertin-Durrieu 2009). Currently only Frobenius is in `nmf.go`.
- **Sparse NMF** with L1 sparsity penalty.
- **Convolutive NMF** (Smaragdis 2004) — temporal templates.
- **NMFD / 2-D-NMF** for drum transcription (Schmidt-Mørup 2006).
- **Ideal binary / ratio / phase-sensitive masks** — IBM, IRM, PSM (Wang 2018). 5-line masking utilities; foundational to every separation eval.
- **Multichannel Wiener (MWF)** (Doclo 2009) — the reference array-microphone speech enhancement.
- **Generalised Eigenvalue / GEVD beamforming** (Warsitz-Häb-Umbach 2007).

**Resampling / sample-rate conversion**
- **Polyphase resampling** (rational L/M; Crochiere-Rabiner 1981).
- **Sinc-windowed resampling** with Kaiser window (the libsamplerate / SoX form).
- **Farrow structure** for arbitrary-ratio resampling (Farrow 1988).
- **Half-band IIR resampling** (Mitra § 13.5) — 2× / 0.5× near-zero-cost.
- **Spline / Lagrange / cubic-Hermite interpolation** for short kernels.

**Adaptive filtering / echo / noise**
- **NLMS** — normalised LMS (Haykin §10).
- **APA** — affine projection algorithm.
- **RLS** — recursive least squares.
- **FDAF** — frequency-domain adaptive filter (block-LMS).
- **PBFDAF** — partitioned-block FDAF (low-latency form used in WebRTC AECM).
- **MDF** — multidelay frequency-domain adaptive filter.
- **Spectral noise gating** (the Audacity / Krisp-style two-pass form: estimate noise → over-subtract).
- **MMSE-LSA noise suppressor** (Ephraim-Malah 1985 log-spectral-amplitude). Strictly stronger than the existing decision-directed Wiener.
- **Wiener with prior** / **OM-LSA** (Cohen-Berdugo 2001).
- **Subspace speech enhancement** (Ephraim-Van Trees 1995, KLT-based).

**Fingerprinting**
- **Shazam-style spectral peak / landmark hashing** (Wang 2003) — peak finding in time-frequency (the package has no 2-D max-filter peak detector), pair-hashing, anchor + target windows.
- **Chromaprint** (AcoustID 2010) — folded chroma + log-band quantisation. Patented expired form is freely usable.
- **Echoprint** (Bertin-Mahieux-Ellis 2011) — onset-time delta hashing.
- **Min-hash / SimHash** of MFCC histograms (used in robust-fingerprint baselines).

## Tier-2: moderately useful

**Cross-frame / sequence**
- **Self-similarity matrix** (SSM) — `librosa.segment.recurrence_matrix`; foundational for structure analysis. Once you have it: novelty curve via Foote checkerboard kernel, segment boundaries via peak-pick on novelty.
- **Foote novelty** (Foote 2000) — checkerboard-kernel correlation along SSM diagonal.
- **Path-enhancement / SCluster** (McFee-Ellis 2014) — segmentation via Laplacian-spectral-clustering of SSM.
- **Lag matrix / time-lag self-similarity** — cousin of SSM used in repetition detection.
- **Dynamic Time Warping** — `librosa.sequence.dtw`; classical Sakoe-Chiba; used in MIR matching, query-by-humming, polygon alignment. Note `sequence` package may already have basic DTW — cross-check.
- **Subsequence DTW** — Müller 2007 §7.2; the workhorse audio matching tool.
- **HMM Viterbi decode for note tracking** — the back-end of pYIN, melodia, monoNote.

**Window functions** (belongs in `signal/`, not `audio/`, but is what `audio` needs)
- **Kaiser**, **KBD (Kaiser-Bessel-Derived)**, **Tukey**, **Gaussian**, **Slepian / DPSS**, **flat-top**, **Nuttall**, **Blackman-Harris** (3- and 4-term), **Bartlett-Hann**, **Lanczos**.
- **Window-sum coefficients** (`Σw`, `Σw²`, ENBW, NENBW) for periodogram calibration.
- **Cosine-sum window family generator** (parameterised).

**Spectrogram families**
- **Reassigned spectrogram** (Auger-Flandrin 1995) — sharper time-frequency localisation.
- **Synchrosqueezing** (Daubechies-Lu-Wu 2011) — time-frequency reassignment for instantaneous-frequency tracking.
- **Wigner-Ville distribution** with cross-term reduction (Choi-Williams).
- **Spectrogram inversion via L-BFGS** (admittedly leans on `optim`).

**Time domain / envelope**
- **RMS / dBFS envelope** with rectification choice (mean / peak / quasi-RMS).
- **Hilbert envelope and instantaneous frequency** (Hilbert transform exists at signal level — should expose audio convenience).
- **Pre-emphasis filter** (`y[n] = x[n] - α·x[n-1]`, α≈0.97); standard ASR pre-step. Trivial but missing.
- **De-emphasis filter** — the inverse pole filter; needed in any LPC synthesis chain.
- **DC blocker** (single-pole high-pass at ~5 Hz); standard for any audio I/O chain.

**Channel / multichannel**
- **Sum-to-mono** with proper coefficient (mid-side decode if stereo).
- **Mid-side encode/decode**.
- **Stereo width** computation (cross-correlation-based).
- **Stereo image (Goniometer-style) coordinates**.

**Pitch / harmonics**
- **Klapuri F0 salience** (multipitch frame-level); algorithmic baseline for polyphonic transcription.
- **Cepstral peak prominence** (CPP) — voice quality / breathiness metric.
- **Harmonic-to-noise ratio** (HNR) — dysphonia metric.
- **Jitter / shimmer** — pitch / amplitude perturbation; clinical voice-quality features.
- **Two-Way-Mismatch** pitch (Maher-Beauchamp 1994) — the historical comparator to YIN.

**Onset / beat extras**
- **HFC** (high-frequency content) onset detector (Masri 1996).
- **Phase-deviation onset detector** (Bello-Sandler 2003 phase-only form).
- **Modified Kullback-Leibler** onset (Brossier 2006).
- **WeightedSpectralFlux** with logarithmic weighting (Böck 2013 RNN-baseline form).
- **Adaptive whitening** (Stowell-Plumbley 2007) before flux computation.

## Tier-3: nice-to-have / niche

- **MFCC++ / SDC features** — shifted-delta-cepstrum (used in language ID).
- **Spectral phase descriptors** — group delay, modified group delay (used in noise-robust ASR).
- **Constant-Q symphonic mask** for vocoder applications.
- **Multi-rate filterbank tree** (PR-QMF, Daubechies wavelet packet).
- **Source-filter inversion** (LPC residual + glottal-flow estimation; Liljencrants-Fant model).
- **Formant tracking** (LPC roots → formant frequencies and bandwidths).
- **Glottal closure instant detection** (DYPSA, SE-VQ).
- **Audio classification baselines** — Gaussian mixture model EM (probably fits `prob`), i-vector / x-vector front-ends.
- **Alpha-stable / heavy-tailed source separation** (Liutkus 2015).
- **Deep-feature-free Goertzel-bank fingerprinting** for narrowband DTMF / sentinel detection.
- **AMDF** — average magnitude difference function pitch (the integer-only sibling of YIN's `d(τ)`).
- **CESS / CDS — Constant Decay Spectrum** silence detection.
- **Dynamic range compressor / limiter / expander / gate / sidechain** — feedforward and feedback topologies; canonical envelope-follower + 4-state ADSR.

## Recommendations (sorted by impact-per-line)

1. **Spectral descriptors batch** (centroid / bandwidth / rolloff / flatness / contrast / RMS / ZCR / slope / decrease / spread / kurtosis / crest) — ~8 lines each, ~12 functions, ~400 lines total. Unlocks every classical-MIR consumer overnight. **Tier-1.**
2. **Inverse mel / Griffin-Lim** — required to close the synthesis loop. ~150 lines (NNLS-back-projection + iterative phase recovery with momentum). **Tier-1.**
3. **Chroma-STFT + Chroma-CQT + CENS + tonnetz** — ~250 lines; turns the package into a proper MIR library. **Tier-1.**
4. **Krumhansl-Schmuckler key + chord-template** — ~80 lines; classic MIR with no dependencies. **Tier-1.**
5. **Median-filter HPSS** — 30 lines for the canonical Fitzgerald horizontal/vertical median form. **Tier-1.** Then **REPET / REPET-SIM** ~150 lines.
6. **Gammatone + ERB + Bark filterbanks** + **GFCC** + **BFCC** — fills the "everything that is not mel" gap. ~400 lines total; opens cochlear-modelling and Essentia-grade timbre. **Tier-1.**
7. **LPC (Levinson-Durbin) + LSF + PARCOR + LPC-cepstrum** — speech-coding bedrock. ~250 lines. **Tier-1.**
8. **K-weighting + BS.1770 LUFS metering + true-peak** — broadcast-grade loudness; ~300 lines (overlaps with `acoustics-missing` 002 recommendation 4). **Tier-1.**
9. **Δ / ΔΔ / liftering / Slaney-norm / CMVN** — single-file MFCC-completeness pack. ~120 lines. **Tier-1.**
10. **pYIN + Viterbi voicing** — modern monophonic-pitch SOTA classical. ~200 lines on top of existing YIN. **Tier-1.**
11. **Tempogram (Fourier + autocorr) + cyclic tempogram + Davies-Plumbley two-state beat tracker** — closes the tempo/beat ladder. ~400 lines. **Tier-2.**
12. **Self-similarity + Foote novelty + subseq-DTW** — segmentation / matching trio. ~300 lines.
13. **Brown-Puckette efficient CQT + iCQT + kernel cache + log-frequency spectrogram** — closes the CQT loop. ~400 lines. **Tier-2.**
14. **MDCT / IMDCT + KBD window** — codec-grade lapped transform. ~200 lines. **Tier-2.**
15. **Polyphase / sinc-Kaiser / Farrow resampler** — sample-rate conversion is currently a hole. ~250 lines. **Tier-2.**
16. **Phase vocoder + WSOLA / PSOLA** — time-stretch / pitch-shift trio. ~300 lines. **Tier-2.**
17. **MMSE-LSA / Ephraim-Malah** noise suppressor — strict upgrade over existing Wiener. ~120 lines. **Tier-2.**
18. **Β-NMF (IS-divergence) + sparse NMF + IBM/IRM/PSM masks + bss_eval (SDR/SIR/SAR/SI-SDR)** — closes the separation evaluation loop. ~300 lines. **Tier-2.**
19. **Shazam landmark hash + Chromaprint** — fingerprinting two-pack. ~400 lines. **Tier-2.**
20. **Window expansion** (Kaiser, KBD, Tukey, Gaussian, Slepian/DPSS, Nuttall, Blackman-Harris, flat-top) in `signal/window.go` — unblocks every other addition. ~200 lines. **Tier-1, place upstream of audio.**

## Sources

- librosa 0.11 docs — `feature.spectral`, `feature.chroma_*`, `feature.tonnetz`, `feature.mfcc`, `feature.melspectrogram`, `feature.poly_features`, `effects.hpss`, `decompose.nn_filter`, `onset.onset_strength`, `beat.beat_track`, `segment.recurrence_matrix`, `sequence.dtw`. https://librosa.org/doc/0.11.0/feature.html
- Essentia 2.1-beta6 algorithm reference — `MFCC`, `BFCC`, `GFCC`, `ERBBands`, `BarkBands`, `LPC`, `LSP`, `Loudness*`, `LUFS`, `Beats*`, `RhythmExtractor*`, `KeyExtractor`, `HPCP`, `ChordsDescriptors`. https://essentia.upf.edu/algorithms_overview.html
- aubio 0.4.9 — pitch methods `yin / yinfft / yinfast / fcomb / mcomb / specacf / schmitt`, onset methods `energy / hfc / complex / phase / specdiff / kl / mkl / specflux`. https://aubio.org/manual/latest/cli.html
- torchaudio 2.10 transforms — `Spectrogram`, `MelSpectrogram`, `MFCC`, `LFCC`, `InverseMelScale`, `GriffinLim`, `Resample`, `SpecAugment`, `RNNTLoss`. https://docs.pytorch.org/audio/stable/transforms.html
- Schörkhuber & Klapuri (2010) "Constant-Q transform toolbox for music processing" — efficient down-sampling-cascade CQT.
- Brown & Puckette (1992) "An efficient algorithm for the calculation of a constant Q transform."
- Mauch & Dixon (2014) "pYIN: a fundamental frequency estimator using probabilistic threshold distributions." ICASSP.
- Fitzgerald (2010) "Harmonic/Percussive separation using median filtering." DAFx-10.
- Rafii & Pardo (2013) "REPET: a simple method for music/voice separation."
- Huang, Chen, Smaragdis & Hasegawa-Johnson (2012) "Singing-voice separation from monaural recordings using RPCA."
- Hermansky (1990) "Perceptual linear predictive (PLP) analysis of speech." JASA 87(4).
- Kim & Stern (2016) "Power-Normalized Cepstral Coefficients (PNCC) for robust speech recognition." IEEE/ACM TASLP.
- Patterson, Holdsworth & Allerhand (1992) — gammatone filterbank.
- Glasberg & Moore (1990) "Derivation of auditory filter shapes from notched-noise data."
- ITU-R BS.1770-4, BS.1387-1; EBU R128, Tech 3341/3342/3343.
- Wang (2003) "An industrial-strength audio search algorithm" (Shazam).
- Ellis (2007) "Beat tracking by dynamic programming."
- Davies & Plumbley (2007) "Context-dependent beat tracking of musical audio."
- Müller (2015) Fundamentals of Music Processing — Chroma, CENS, tempogram, SSM, DTW chapters.
- Rafii et al. (2025) "30+ Years of Source Separation Research." arXiv:2501.11837.

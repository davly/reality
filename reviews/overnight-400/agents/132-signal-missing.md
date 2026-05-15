# 132 — signal: missing canonical primitives

Scope: enumerate canonical signal-processing primitives ABSENT from
`C:\limitless\foundation\reality\signal\`. Current entry points (grep
`^func [A-Z]` in `fft.go`, `filter.go`, `window.go`):

```
FFT, IFFT, PowerSpectrum, FFTFrequencies,
Convolve, MovingAverage, ExponentialMovingAverage, MedianFilter,
HannWindow, HammingWindow, BlackmanWindow, ApplyWindow
```

12 entry points, 3 source files, ~470 LOC. Cross-package note:
`audio/spectrogram/{Compute,Inverse}` provides STFT/ISTFT but lives in
`audio/`, not `signal/`. `control/filter.go` has scalar `LowPassFilter`,
`HighPassFilter`, `ComplementaryFilter`, `RateLimiter` — first-order RC
analogues, not DSP design. **Reality has no IIR filter design, no
real-FFT, no Hilbert, no spectral-estimation, no wavelets, no
decomposition, no adaptive filters, no state-space estimators, no
resampling.** Agent 131 audited what exists; this report lists what is
absent. Below: three tiers ordered by canonicalness × downstream use
(aicore audio front-end, Pistachio realtime FFT, Folio biosignals,
control loops).

---

## Tier 1 — canonical, no excuse to be absent

Textbook (Oppenheim-Schafer, Proakis-Manolakis, Vaidyanathan); ship in
SciPy `signal.*`, MATLAB SPT, GNU Radio, JUCE.

### 1.1 RFFT / IRFFT — real-input FFT exploiting Hermitian symmetry
Current `FFT(real, imag)` makes consumers pass a zero `imag` slice and
discards the upper N/2 bins (`X[N-k] = conj(X[k])`). RFFT halves both
storage (N/2+1 bins) and arithmetic by packing two real-FFTs into one
complex-FFT, or via split-radix real-data form (Sorensen-Heideman-Burrus
1987). 2× wall-time win for audio at no precision cost. **API:**
`RFFT(real []float64, outReal, outImag []float64)`, `IRFFT(...) ` inverting.

### 1.2 Goertzel single-bin DFT
O(N) two-real-multiply recurrence
`s[n] = x[n] + 2cos(2πk/N)·s[n-1] - s[n-2]`. Used everywhere a single
tone is the question (DTMF, line-frequency notch trigger, capacitive
touch, ECG 50/60 Hz hum, lock-in amplifier). Today consumers must run
full FFT and discard N-1 bins. **Goertzel 1958** *Amer Math Monthly* 65(1).
**API:** `Goertzel(signal []float64, targetFreq, sampleRate float64) (real, imag float64)`.

### 1.3 Bluestein chirp-z transform — arbitrary-N FFT
FFT panics on non-power-of-2 (verified `fft.go:50`). Bluestein 1970
reduces arbitrary-N DFT to length-M ≥ 2N-1 power-of-2 convolution via
chirp factorisation: three length-M FFTs + pointwise multiply.
Required whenever input length isn't pow2 (e.g. 1764 samples per 40 ms
at 44.1 kHz). Current pad-to-pow2 workaround biases spectral estimates.
**API:** `BluesteinFFT(real, imag []float64)` accepting any N.

### 1.4 Biquad IIR (Direct Form II Transposed) + cascade
Single most important DSP missing piece. `H(z) = (b0+b1z⁻¹+b2z⁻²)/(1+a1z⁻¹+a2z⁻²)`.
DF-II-T is canonical for IEEE 754 (Oppenheim-Schafer §6.5; preferred
over DF-I/DF-II for noise gain — Higham §13). Cascade of N biquads
realises any 2N-th-order IIR with better quantisation than direct-form-N.
Audio EQ Cookbook (Bristow-Johnson 2005) RBJ formulae give
LP/HP/BP/notch/peaking/shelving from cutoff+Q+gain. **API:**
`BiquadDF2T(b0,b1,b2,a1,a2 float64)` with `Process(in,out []float64)`;
`RBJLowpass(cutoff,q,sampleRate float64) Biquad` etc.

### 1.5 IIR filter design: Butterworth, Chebyshev I/II, Elliptic
Pole-placement primitives behind every analog-prototype digital filter.
Butterworth (maximally flat), Chebyshev-I (passband ripple), Chebyshev-II
(stopband ripple), Elliptic (both — smallest order). Pipeline: analog
prototype poles → bilinear transform (or impulse-invariant) → factor
into biquads. SciPy `butter`/`cheby1`/`cheby2`/`ellip`. Without these,
consumers cannot specify "4th-order LP at 1 kHz, 0.5 dB ripple, 40 dB
stopband" — stuck with first-order EMA. **API:**
`Butterworth(order int, cutoff, sampleRate float64) []Biquad`.
**Sources:** Parks-Burrus 1987, Antoniou 1993.

### 1.6 FIR filter design: window method + Parks-McClellan (Remez)
Linear-phase FIR mandatory whenever phase distortion matters (audio
crossovers, EEG/ECG, image). Window method:
`h[n] = ideal[n]·w[n]`. Parks-McClellan / Remez (McClellan-Parks-Rabiner
1973) gives optimal equiripple FIR for given band edges and ripple —
MATLAB/SciPy `firpm`/`remez`. Pair with existing `Convolve` for
filtering. **API:** `FIRWindow(order int, cutoff, sampleRate float64, win WindowType) []float64`;
`RemezExchange(numTaps int, bands, desired, weight []float64) []float64`.

### 1.7 Hilbert transform (FFT-based analytic signal)
90°-phase-shift filter producing `x_a = x + j·H(x)`. Yields envelope
`|x_a|` and instantaneous phase `arg(x_a)`. Single most-used DSP
primitive in biomedical (HHT AM-FM demod, EEG phase-locking value),
communications (SSB demod, IQ baseband), audio (envelope follower,
phase vocoder). FFT-based form (Marple 1999):
`Y[k] = X[k]·{2 for k>0, 1 for k=0,N/2, 0 for k>N/2}`, then IFFT.
Six lines once RFFT exists. **API:** `Hilbert(signal []float64, outReal, outImag []float64)`,
`AnalyticEnvelope(...)`, `InstantaneousPhase(...)`.

### 1.8 Welch periodogram (averaged-modified-periodogram PSD)
Textbook PSD estimator: overlap-windowed segments, FFT each, average
squared magnitudes, normalise by `fs · Σw[i]²`. **Welch 1967** *IEEE
Trans Audio Electroacoust*. Vastly lower variance than raw periodogram.
Every spectrum chart of the last 50 years is Welch. Reality has the
building blocks but no helper that composes them with proper
normalisation. Pair with **Bartlett** (non-overlapping) and raw
**Periodogram**. **API:**
`Welch(signal []float64, segLen, overlap int, win WindowType, sampleRate float64) (freqs, psd []float64)`.

### 1.9 STFT / ISTFT (in `signal/`, not `audio/`)
Currently `audio/spectrogram/{Compute, Inverse}`. STFT is generic, not
audio-specific (seismology, EEG, gravitational waves, radar, vibration
monitoring). Architectural re-home; same observation for magnitude/
phase split. **API:** move to `signal/stft.go`.

### 1.10 Cross-correlation / FFT-convolve
`Convolve` (filter.go:19) is direct O(N·M). FFT-based form is
O((N+M) log(N+M)) and the only practical approach for M ≳ 64. No
helper composes FFT + multiply + IFFT today. Likewise no `Correlate`,
no `XCorr` (signed lag), no `Autocorrelate` (which `audio/pitch`
reimplements internally). **API:** `FFTConvolve(signal, kernel, out []float64)`,
`Correlate(a, b, out []float64) (lags []int)`, `Autocorrelate(signal, out []float64)`.

### 1.11 Savitzky-Golay smoothing + differentiating filter
Local polynomial regression in sliding window. Equivalent to a fixed
FIR `(M^T·M)^-1·M^T` (Vandermonde). Smooths while preserving moments
(peak height, width, area) — uniquely useful for spectroscopy,
chromatography, peak-finding. **Savitzky-Golay 1964** *Anal Chem*
36(8) — most-cited paper in the journal. Also yields smoothed
derivatives of any order ≤ poly degree (gold standard for noisy
derivative estimation). **API:**
`SavitzkyGolay(signal []float64, windowLen, polyOrder, deriv int, out []float64)`.

### 1.12 Kalman filter (linear state-space)
Optimal recursive Bayesian estimator for linear-Gaussian systems.
`x_k = F·x_{k-1} + B·u_k + w_k; z_k = H·x_k + v_k`. Predict-update
with covariance propagation. Foundational for tracking/fusion/state-
estimation in control, robotics, GPS, INS, finance, biomedical
(**Kalman 1960** *J Basic Eng* 82(1)). Reality has zero state-space
estimators despite `linalg` already covering matrix ops. Pair with
**Information filter** (canonical-information form, numerically preferred
for low-info priors) and **Rauch-Tung-Striebel smoother** (backward
pass for fixed-interval smoothing). **API:** `KalmanFilter` struct
with `Predict(F, Q, B, u)` and `Update(z, H, R)`; `RTSSmoother(...) [][]float64`.

---

## Tier 2 — high-value, broadly expected

### 2.1 EKF and UKF
EKF: linearise nonlinear `f`, `h` via Jacobians (Jazwinski 1970). UKF:
deterministic 2N+1 sigma-point sampling (Julier-Uhlmann 1997), third-
order accurate vs EKF's first-order. UKF is the modern default in
robotics/aerospace. Both build on Tier 1.12. Square-root UKF
(van der Merwe-Wan 2001) for numerical stability.

### 2.2 Multitaper PSD (Slepian / DPSS tapers)
**Thomson 1982** *Proc IEEE* 70(9). First K Discrete Prolate Spheroidal
Sequences (Slepian 1978) as orthogonal tapers, K periodograms averaged.
Lower variance than Welch at same resolution; F-test for spectral lines.
Used in geophysics, climatology, neuroscience (Chronux, MNE-Python).
DPSS = eigenvalue problem on tridiagonal matrix (`linalg` covers).
**API:** `Multitaper(signal []float64, NW float64, K int, sampleRate float64) (freqs, psd []float64)`,
`DPSS(N int, NW float64, K int) (tapers [][]float64, eigvals []float64)`.

### 2.3 Continuous Wavelet Transform (CWT)
`W(a,b) = ∫ x(t)·ψ*((t-b)/a)/√a dt`. Mother wavelets: Morlet (complex
Gabor — TF ridge), Mexican hat (Ricker — seismic), Paul, DOG. Fast
form is FFT-based (multiply in frequency by scaled wavelet spectrum).
Used wherever Heisenberg tradeoff must beat STFT: gravitational-wave
detection (LIGO Q-transform), seismology, EEG ridge tracking, transient
detection. **API:**
`CWT(signal []float64, scales []float64, wavelet WaveletType, sampleRate float64) [][]complex128`.

### 2.4 Discrete Wavelet Transform (DWT) + lifting scheme
Mallat 1989 dyadic decomposition via QMF pair (Daubechies db1..db20,
Symlets, Coiflets, Biorthogonal). Lifting (Sweldens 1995) factors
wavelet filter into in-place prediction-update steps — numerically
superior (integer-to-integer exact) and faster than convolution form.
Used in JPEG2000, denoising, multiresolution. **API:**
`DWT(signal []float64, wavelet WaveletType, levels int) (cA, cD [][]float64)`,
`IDWT(...)`, `LiftingDWT(...)`.

### 2.5 Spectral estimation: Burg AR, MUSIC, ESPRIT, Pisarenko
Parametric methods. **Burg** 1967: max-entropy AR via reflection
coefficients; numerically stable Levinson-Durbin alternative.
High-resolution PSD on short records. **MUSIC** (Schmidt 1979):
eigendecompose covariance, project onto noise subspace, scan for
peaks. Super-resolution DOA / line-frequency. **ESPRIT** (Roy-Kailath
1989) *IEEE Trans ASSP* 37(7): rotational invariance — closed-form,
faster than MUSIC. **Pisarenko** 1973: harmonic decomposition;
ancestor of MUSIC. **API:**
`BurgAR(signal []float64, order int) (ar []float64, variance float64)`,
`MUSIC(signal []float64, signalDim, fftLen int, sampleRate float64) (freqs, pseudoSpectrum []float64)`,
`ESPRIT(signal []float64, signalDim int, sampleRate float64) (freqs []float64)`.

### 2.6 EMD / Ensemble EMD
**Huang et al. 1998** *Proc R Soc Lond A* 454. Adaptive data-driven
decomposition into Intrinsic Mode Functions via iterative sifting
(local-max / local-min envelopes, subtract mean, repeat). Combined with
Tier 1.7 Hilbert → **Hilbert-Huang transform** for instantaneous
frequency without basis assumptions. Ensemble EMD (Wu-Huang 2009)
addresses mode-mixing via noise-assisted averaging. Used in EEG, HRV,
ocean rogue waves, structural health monitoring. Needs cubic-spline
envelope (`calculus` may cover). **API:**
`EMD(signal []float64, maxIMFs int) [][]float64`,
`EnsembleEMD(signal []float64, maxIMFs int, noiseStd float64, trials int) [][]float64`.

### 2.7 Variational Mode Decomposition (VMD)
**Dragomiretskiy-Zosso 2014** *IEEE Trans Sig Proc* 62(3). Variational
alternative to EMD: jointly decompose into K narrow-band IMFs by
augmented-Lagrangian with bandwidth penalty. Better mode separation,
no recursive sifting, mathematically principled. Modern default in
mechanical fault diagnosis. **API:**
`VMD(signal []float64, K int, alpha, tau float64, maxIter int) [][]float64`.

### 2.8 Resampling: polyphase, Farrow, Kaiser-windowed sinc
Crochiere-Rabinowitz 1981 polyphase decomposition: rational `L/M`
conversion via FIR split into L polyphase subfilters (avoids
upsample-by-L). Farrow (1988) gives arbitrary-fractional-delay
interpolation. Kaiser-windowed sinc + polyphase is the standard
high-quality audio resampler (libsoxr, libsamplerate, JUCE). Reality
has nothing for SRC; agent 007 noted same gap in `audio/`. Should
live in `signal/`. **API:**
`Resample(signal []float64, ratio float64, out []float64)`,
`PolyphaseFilter(signal []float64, L, M int, taps []float64, out []float64)`,
`FarrowInterpolator(signal []float64, fractionalDelays []float64, polyOrder int, out []float64)`.

### 2.9 Group delay / phase delay / pole-zero / stability
`τ_g(ω) = -dφ/dω`, `τ_p(ω) = -φ/ω`, pole-zero plot, frequency response
`H(e^jω)` magnitude/phase, stability check (`|p_i| < 1` for IIR poles).
Required by anyone designing or auditing a filter. Pairs with bilinear
transform from Tier 1.5. **API:**
`FreqResponse(b, a []float64, freqs []float64) (magnitude, phase []float64)`,
`GroupDelay(...)`, `Stable(a []float64) bool`.

### 2.10 Chirp z-transform (CZT) — beyond Bluestein
Generalises DFT to spiral contour `z = A·W^k` (Rabiner-Schafer-Rader
1969). Bluestein is the unit-circle special case. Useful for zoom-FFTs
(high-resolution sub-band) and arbitrary contour sampling.

### 2.11 Cepstrum (real, complex, MFCC-precursor)
Real cepstrum `c[n] = IFFT(log|FFT(x)|)`; complex cepstrum
`IFFT(log(FFT(x)))` with phase unwrapping. Used in echo detection
(**Bogert-Healy-Tukey 1963** — coined "cepstrum"), deconvolution,
pitch, MFCC precursor. Audio has MFCC; raw cepstrum should live in
`signal/`.

### 2.12 Adaptive filters: NLMS, RLS
**NLMS** (Widrow-Hoff 1960, normalised by Bitmead 1980): step-size
normalised by input power; standard adaptive filter. **RLS**
(Haykin §10): exponentially-weighted LS; faster convergence at higher
cost. Used in echo cancellation, equalisation, noise cancellation,
system identification. **API:**
`NLMS(numTaps int, mu float64) AdaptiveFilter` with
`Update(input, desired float64) (output, error float64)`.

---

## Tier 3 — specialised but canonical

### 3.1 Wigner-Ville distribution + Cohen's class
Quadratic TF. `W(t,f) = ∫ x(t+τ/2)·conj(x(t-τ/2))·e^{-2πjfτ} dτ`. Best
resolution among bilinear TF distributions but cross-term interference.
Cohen's class adds smoothing kernels (Choi-Williams, Born-Jordan,
Margenau-Hill). Reduced Interference Distributions (Williams 1996) are
the modern standard. Non-stationary signals, machine fault diagnosis.

### 3.2 S-transform (Stockwell)
**Stockwell-Mansinha-Lowe 1996** *IEEE Trans Sig Proc* 44(4). Frequency-
dependent Gaussian-windowed STFT — multiresolution like CWT but with
absolute (not relative) phase reference. Heavy use in geophysics, EEG.

### 3.3 Constant-Q transform (in `signal/`, not `audio/`)
Currently `audio/cqt/`. CQT is generic log-frequency (**Brown 1991**
*JASA* 89(1)) — seismology, vibration, bioacoustics, not just music.
Architectural re-home (mirrors Tier 1.9 STFT).

### 3.4 Particle filter (sequential Monte Carlo)
**Gordon-Salmond-Smith 1993** *IEE Proc F*. Non-parametric Bayesian
state estimation: weighted particles, propagate through nonlinear
dynamics, resample. Generalises Kalman to non-Gaussian, multimodal
posteriors. Pair with KF infrastructure.

### 3.5 Walsh-Hadamard transform / discrete Hartley transform
WHT: O(N log N) over `{±1}^N` basis (Walsh 1923, Hadamard 1893). Fast
cross-correlation of {±1} sequences (PN codes, CDMA), Hadamard ECC,
feature hashing. DHT (Bracewell 1983): real-valued FFT analogue —
`cas(2πkn/N) = cos+sin`. Self-inverse up to scale, real arithmetic only.

### 3.6 DCT / DST / MDCT / IMDCT (in `signal/`)
DCT-II/III (Ahmed-Natarajan-Rao 1974) is the JPEG/MP3/AAC backbone.
Reality has DCT inside `audio/mfcc.go` only. Should expose DCT-I/II/III/IV
+ DST family in `signal/`. Pair with **MDCT/IMDCT** (Princen-Bradley
1986) for the lapped form behind every modern transform codec.

### 3.7 Pole-zero factorisation / partial-fraction expansion
Symbolic Z-transform tables overkill, but **pole-zero factorisation**
of `b(z)/a(z)` (root-finding on denominator — `optim` covers) and
**partial-fraction expansion** for converting between transfer-function
and biquad-cascade forms are essential filter-design utilities.

### 3.8 Cyclostationary statistics
Gardner 1986 *IEEE Trans Comm* 34(11). Hidden periodicity in second-
order statistics (modulated signals, rotating machinery). Spectral
correlation `S_x^α(f) = lim (1/T)·∫ X(f+α/2)·X*(f-α/2) dt`. Niche but
the only right answer for some fault-detection classes.

### 3.9 Fractional Fourier transform
Namias 1980, Almeida 1994. Continuous rotation in TF plane parameterised
by α (FFT is α=π/2). Chirp detection, optical signal processing,
TF filtering.

### 3.10 Higher-order statistics: bispectrum, bicoherence
Brillinger 1965 *Biometrika* 52. Detect non-Gaussianity, phase
coupling, quadratic phase relationships invisible to power spectrum.
EEG (cross-frequency coupling), oceanography, plasma physics.
Implementation: third-order moment FFT.

### 3.11 Empirical wavelet transform (EWT)
Gilles 2013 *IEEE Trans Sig Proc* 61(16). Adaptive wavelet basis from
Fourier spectrum of input — bridges EMD (adaptive, no theory) and DWT
(theory, fixed basis).

### 3.12 Singular spectrum analysis (SSA)
Broomhead-King 1986 *Physica D*. Embed signal in trajectory matrix,
SVD, group eigentriples, reconstruct via diagonal averaging. "PCA of
time series". Climate, finance, biomedical. Needs SVD (`linalg`).

### 3.13 Allan variance / Hadamard variance
Allan 1966 *Proc IEEE* 54(2). Frequency-stability measure that
converges for power-law-noise processes where standard variance
diverges. Standard for clock characterisation, gyro spec-sheets,
oscillator noise. Trivial given windowed sums.

### 3.14 Graph signal processing (DFT on graphs)
Sandryhaila-Moura 2013. Generalises DFT via graph Laplacian
eigenbasis. Network signal processing, sensor fusion, brain-network
analysis. Reality has both `signal` and `graph`.

### 3.15 Optimal filter banks: QMF, PR cosine-modulated
Vaidyanathan 1993. Filter-bank design with perfect-reconstruction —
generalises DWT to arbitrary band counts. Backbone of MP3 (32-band PQMF)
and AAC (modified DCT filter bank).

---

## Cross-cutting observations

1. **No filter design infrastructure at all.** Tier 1.4-1.6 (biquad,
   IIR, FIR design) is the largest single gap. Any consumer needing a
   real LP/HP/BP today must hand-write coefficients. Most-frequently-
   used missing capability.
2. **No state-space estimation (Kalman family).** Foundational across
   control/robotics/fusion/finance. Needs only `linalg` (already
   present). Moderate cost, enormous value.
3. **No spectral estimation beyond raw FFT.** Welch (1.8), Multitaper
   (2.2), AR/MUSIC/ESPRIT (2.5) all missing. PowerSpectrum returns raw
   squared-magnitude — building block only.
4. **No analytic-signal / Hilbert path.** Tier 1.7 blocks every
   envelope/phase application. Tier 2.6 (EMD) and 2.7 (VMD) chain off
   it for HHT.
5. **No time-frequency / multi-resolution beyond STFT.** Wavelets
   (2.3-2.4), CWT, S-transform (3.2), Wigner-Ville (3.1) — entire
   missing axis.
6. **STFT/CQT live in `audio/`.** Both are generic signal primitives.
   Architectural cleanup (1.9, 3.3).
7. **No resampling.** Audio (per agent 007) also missing; should live
   in `signal/`, reused by `audio/` (2.8).
8. **MASTER_PLAN topic is faithful.** Goertzel, Bluestein, multitaper,
   wavelets, EMD, VMD, Kalman/UKF, Savitzky-Golay, Welch, MUSIC,
   ESPRIT — all genuinely absent. Brief is accurate; this report adds
   context and adjacent missing primitives.

---

## Files referenced

`C:/limitless/foundation/reality/signal/{fft.go,filter.go,window.go}`
`C:/limitless/foundation/reality/audio/spectrogram/stft.go`
`C:/limitless/foundation/reality/control/filter.go`

## 2-line summary

Reality's `signal/` ships ~12 functions (Cooley-Tukey FFT/IFFT,
PowerSpectrum, naive Convolve, EMA/MovingAverage/MedianFilter, three
windows) and is missing the entire DSP middle layer: RFFT, Goertzel,
Bluestein, biquad IIR + filter design (Butterworth/Chebyshev/Elliptic),
FIR design (window method, Parks-McClellan), Hilbert, Welch + multitaper
PSD, STFT (lives in audio), wavelets (CWT/DWT + lifting), EMD/VMD,
Kalman/EKF/UKF, Savitzky-Golay, spectral estimation
(Burg/MUSIC/ESPRIT/Pisarenko), and resampling (polyphase/Farrow/
Kaiser-sinc). Filter design is the single largest gap — consumers
cannot specify a filter today beyond first-order EMA.

# 317 — dive-window-functions (Kaiser / DPSS / Slepian / Nuttall / Blackman-Harris / Flat-Top audit)

## Headline
Reality ships **3 of ~14 standard windows** (Hann, Hamming, Blackman in `signal/window.go:15,44,76`); the entire parametric and high-dynamic-range catalog is absent — Kaiser, Tukey, Gaussian, Welch, Bartlett, Blackman-Harris (3-/4-term), Nuttall, Flat-Top, KBD, DPSS/Slepian, Bohman, Parzen, Lanczos. Day-1 PR: ship Kaiser+Tukey+Gaussian+Welch+Bartlett+Nuttall+BH4+FlatTop in **one file (~280 LOC, ~30 vectors × 8 windows)**; defer DPSS to slot 097 (eigenvalue solver) + multitaper (slot 132 §2.2).

## Findings (existing audit)

### What exists
- `signal/window.go:15` — `HannWindow(n, out)`. Periodic-cosine, sidelobe ≈ -32 dB, roll-off 6 dB/oct. Audio default. Zero-alloc, panics-on-shape; correct symmetric form `0.5·(1−cos(2πi/(n−1)))`.
- `signal/window.go:44` — `HammingWindow(n, out)`. `0.54 − 0.46·cos(...)`. Sidelobe -43 dB, slow 6 dB/oct roll-off. **Note**: classical 0.54/0.46 coefficients (not Nuttall's "exact" 0.53836/0.46164 minimax form, which gives -42.7 dB). Speech-recognition default.
- `signal/window.go:76` — `BlackmanWindow(n, out)`. 3-term cosine `0.42, 0.5, 0.08`. **Note**: this is the *classical* Blackman, not the "exact" Blackman (`0.42659, 0.49656, 0.07685`). Sidelobe -58 dB.
- `signal/window.go:104` — `ApplyWindow(signal, window, out)`. Element-wise multiply.
- `audio/spectrogram/stft.go:53,137` — STFT `Compute`/`Inverse` consume an externally-supplied `window []float64`; spectrogram tests (`spectrogram_test.go:37,77,185,215,243,308`) all use `signal.HannWindow`. No other windows are wired in audio.
- `audio/onset/superflux.go`, `audio/separation/nmf.go`, `audio/spectrogram/mel_spectrogram.go` — all consume STFT output, transitively Hann-only by current call sites.

### What is missing (catalog of standard windows)
Grep across all of `signal/`, `audio/`, `acoustics/` for `Kaiser|Nuttall|Tukey|Gaussian|DPSS|Slepian|FlatTop|Welch|Bartlett|BlackmanHarris|Parzen|Bohman|KBD|Lanczos`: **0 production hits** (only review-doc references).

| # | Window | Citation | Peak SLL (dB) | Main-lobe (bins, -3dB) | Roll-off | Tunable | LOC |
|---|--------|----------|---------------|------------------------|----------|---------|-----|
| 1 | Rectangular (boxcar) | — | -13 | 0.89 | 6 dB/oct | no | 5 |
| 2 | Bartlett (triangular) | — | -27 | 1.28 | 12 dB/oct | no | 12 |
| 3 | Welch (parabolic) | Welch 1967 | -21 | 1.20 | 12 dB/oct | no | 12 |
| 4 | Tukey (cosine-tapered) | Tukey 1967 | -15 (α=0.5) | 1.22 | 18 dB/oct | α∈[0,1] | 25 |
| 5 | Gaussian | Harris 1978 | -42 (σ=0.4) | tunable | tunable | σ | 15 |
| 6 | Blackman-Harris (3-term) | Harris 1978 | -67 | 1.66 | 6 dB/oct | no | 18 |
| 7 | Blackman-Harris (4-term, "minimum") | Harris 1978 | -92 | 1.90 | 6 dB/oct | no | 18 |
| 8 | Nuttall (4-term, "minimum-3-term-derivative") | Nuttall 1981 | -98 | 1.98 | 18 dB/oct | no | 18 |
| 9 | Flat-top (5-term) | SRS / Heinzel 2002 | -88 | 3.86 | 6 dB/oct | no | 18 |
| 10 | Kaiser | Kaiser 1974 | tunable (β) | tunable | depends | β | 35 + I₀ |
| 11 | KBD (Kaiser-Bessel-Derived) | Princen-Bradley 1986 | tunable | tunable | depends | α | 25 |
| 12 | DPSS / Slepian | Slepian 1978 | optimal | optimal | depends | NW, K | ~200 + Eigvec |
| 13 | Bohman | Bohman 1960 | -46 | 1.71 | 24 dB/oct | no | 15 |
| 14 | Parzen (de la Vallée Poussin) | Parzen 1961 | -53 | 1.91 | 24 dB/oct | no | 25 |
| 15 | Bartlett-Hann | — | -36 | 1.45 | 6 dB/oct | no | 18 |
| 16 | Lanczos sinc | — | -27 | 1.30 | 12 dB/oct | no | 15 |
| 17 | Cosine / Sine | — | -23 | 1.23 | 12 dB/oct | no | 8 |
| 18 | Exact Blackman | Blackman-Tukey | -68 | 1.61 | 6 dB/oct | no | 12 |
| 19 | Hann–Poisson | Harris 1978 | tunable | tunable | α | 18 |
| 20 | Ultraspherical / Dolph-Chebyshev | Dolph 1946 | -A dB const | min main-lobe at A | constant SLL | A | 100 |

### Cross-package fanout (consumers gated on missing windows)
- **audio/spectrogram** — STFT users today get Hann only. Speech ASR convention is **Hamming** (already exists, not wired); modern librosa/torchaudio default to **Hann** (matches reality). High-dynamic-range music analysis wants **Blackman-Harris-4** (-92 dB sidelobes). Vibration analysis wants **flat-top** (amplitude accuracy, 0.01 dB scallop loss).
- **audio/separation/nmf.go** — magnitude-spectrogram NMF is sensitive to spectral leakage; BH4 or Nuttall would reduce leakage bias on low-rank models.
- **acoustics** (slot 003) — room impulse response analysis wants **Kaiser** for parametric stopband control.
- **prob/sequence** — long-window detrending / smoothing wants **Tukey** (preserves middle, tapers edges).
- **slot 132 §2.2 multitaper PSD** — gated on **DPSS** (which is gated on `linalg` Eigvec, slot 097).
- **slot 132 §1.8 Welch PSD** — gated on a parametric window selector enum.
- **slot 316 dive-fir-design** — Kaiser is the FIR-window-method workhorse; FIR design and Kaiser window ship together.
- **MP3 / AAC / Vorbis codec testing** — KBD is the *mandatory* window for AAC overlap-add; if reality is to validate codec implementations, KBD is required.
- **radar / sonar narrowband detection** — DPSS first taper minimises spectral leakage outside the resolution bandwidth (Slepian-Pollak optimum); textbook for adaptive antenna beamforming and seismic line tracking.

### Numerical / API issues in existing windows
1. **Periodic vs symmetric form ambiguity** — current `HannWindow` etc. use the **symmetric** form `cos(2πi/(n−1))` (zero at both endpoints, used for FIR-window-method design). For STFT analysis, the **periodic** form `cos(2πi/n)` (zero at left endpoint only) is required for COLA reconstruction at hop=N/4. `audio/spectrogram/stft.go:53` consumes Hann symmetric and tolerates ~0.6% reconstruction residual at boundaries (see `spectrogram_test.go:73` "Hann is COLA at hop=frameSize/4" — it is, but only for periodic Hann). **Action**: add `HannWindowPeriodic(n, out)` or a `Periodic bool` parameter.
2. **Coefficient lineage** — Blackman uses `0.42, 0.5, 0.08` (truncated rational), not the exact minimax `0.42659, 0.49656, 0.07685`. Document the choice; both are "Blackman" in the literature but give slightly different SLLs.
3. **No `WindowKind` enum** — every consumer hard-codes `signal.HannWindow(...)`. A `WindowKind` enum + dispatch helper `Window(kind, n, out)` would let `Welch(...)`, `STFTConfig{Window: WindowHannPeriodic}`, etc. accept window selection generically. ~30 LOC.
4. **No Equivalent-Noise-Bandwidth (ENB) helper** — Welch's-method PSD normalisation needs `Σw[i]²/N`. Heinzel-Rüdiger-Schilling 2002 §A1 tabulates ENB, scallop loss, processing gain, coherent gain for every window. Provide `WindowMetrics(w []float64) WindowMetrics{ENB, ScallopLoss, CoherentGain, ProcessingGain, FlatnessLoss}`. ~25 LOC, ~5 vectors per window.
5. **n=1 special case** — current code returns `out[0]=1.0`. For Hann and Blackman that violates the symmetric "zero-endpoints" definition (the limit `n→1` of the formula gives `1−1=0`). Document the convention: `n=1 → 1.0` is the SciPy/MATLAB convention, but worth a comment. Add a unit test pin.
6. **Kahan-summation not needed for windows** — windows are direct evaluations, not accumulating; the existing `for i := 0; i < n; i++ { out[i] = ... }` is numerically fine to ULP. No fix needed; pin it.

### Numerical traps to pin (R-MUTUAL-CROSS-VALIDATION 3/3)
1. **Symmetry**: every closed-form window except DPSS satisfies `w[i] == w[n-1-i]` exactly to ULP. Pin once per window.
2. **Kaiser limit cases**: `β=0` → rectangular (verify `‖Kaiser(N,0) − Rectangular(N)‖∞ < 1e-15`); `β=5` → ≈ Hamming (verify `‖Kaiser(N,5) − Hamming(N)‖∞ < 0.05`); `β=8.96` → ≈ Blackman; `β→∞` → narrow Gaussian.
3. **Nuttall ≡ BH4** when coefficient choice is "minimum-3-term-derivative" — coefficient-form regression.
4. **DPSS first taper Slepian-Pollak optimal concentration**: the analytic property is `λ_0 = ∫_{-W}^W |U_0(f)|² df / ∫|U_0(f)|² df → maximised`. With NW=4, K=1, expect `λ_0 ≈ 0.99999...`. Pin the eigenvalue.
5. **DPSS orthogonality**: `Σ U_k[n]·U_l[n] = δ_kl`. Pin Frobenius `‖U^T U − I‖ < 1e-10` for K=4, NW=4, N=512.
6. **Flat-top scallop loss** ≈ 0.01 dB for 5-term form; pin via FFT(zero-padded sinusoid windowed) → bin amplitude vs frequency offset.
7. **Tukey limit cases**: `α=0` → rectangular; `α=1` → Hann (exact, tested at ULP).
8. **KBD overlap-add identity**: `KBD[n]² + KBD[n+N/2]² = 1` for all `n=0..N/2-1` (Princen-Bradley COLA constraint, the *defining* property; pin to ULP).
9. **Bartlett ≡ Bartlett(2N−1) windowed by sinc-derived form** — convolution identity rarely tested; nice 3/3 vector.
10. **Welch ≡ 1 − ((2i−(N−1))/(N+1))²** — closed-form parabolic; cross-validate against expansion at i=0 (= 1−1 = 0 for N odd) and centre.

## Concrete recommendations

### Day-1 PR: `signal/window_extra.go` (~280 LOC, 7 windows + metrics)
Ship in this exact order (priority by consumer demand):

1. **`KaiserWindow(n int, beta float64, out []float64)`** — Kaiser 1974. Requires `I₀` (modified Bessel first-kind, order 0); ~40 LOC including I₀ via series for `x²<50` and asymptotic for larger. Pin against scipy.signal.windows.kaiser at β=0,1,5,8.96,14 with N=8,16,32,64,128 (~25 vectors × 5 β = 125; cap at 30).
2. **`TukeyWindow(n int, alpha float64, out []float64)`** — Tukey 1967, cosine-tapered. α=0 → rect, α=1 → Hann. ~20 LOC.
3. **`GaussianWindow(n int, sigma float64, out []float64)`** — `exp(-0.5·((i-(n-1)/2)/(σ·(n-1)/2))²)`. ~15 LOC.
4. **`WelchWindow(n int, out []float64)`** — `1 - ((2i-(N-1))/(N+1))²`. Parabolic. ~12 LOC. Required by slot 132 §1.8.
5. **`BartlettWindow(n int, out []float64)`** — triangular `1 - |2i/(N-1) - 1|`. ~12 LOC.
6. **`BlackmanHarrisWindow(n int, out []float64)`** — 4-term, `0.35875, 0.48829, 0.14128, 0.01168`. -92 dB SLL. ~18 LOC. **Required by every dynamic-range-conscious spectrum analyzer**.
7. **`NuttallWindow(n int, out []float64)`** — 4-term, `0.3635819, 0.4891775, 0.1365995, 0.0106411`. -98 dB SLL, 18 dB/oct roll-off. ~18 LOC.
8. **`FlatTopWindow(n int, out []float64)`** — 5-term SRS form, coefficients `0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368` (Heinzel 2002 Table 1). For amplitude-accurate measurement (scallop loss <0.01 dB). ~18 LOC.
9. **`WindowMetrics(w []float64) (enb, scallopLoss, coherentGain, processingGain, flatnessLoss float64)`** — ~25 LOC; enables Welch's-method ENB normalisation and is **the** missing operational helper.
10. **`WindowKind` enum + `Window(kind WindowKind, n int, out []float64, params ...float64) error`** dispatch — single-call generic helper for slot 132 Welch/STFT consumers. ~30 LOC.

Day-1 totals: ~280 LOC, ~210 golden vectors (7 windows × 30 each), zero new deps. **No** dependency on linalg.

### Day-2 PR: `signal/window_dpss.go` (gated on slot 097 eigvec)
- **`DPSS(n int, NW float64, K int) (tapers [][]float64, eigvals []float64)`** — Slepian 1978; symmetric tridiagonal eigenvalue problem (Percival-Walden 1993 §8.3 eq 8.4). Requires symmetric tridiagonal eigenvector solver from `linalg` (slot 097). ~150 LOC. Pin orthogonality + concentration eigenvalue + first-taper analytic property.
- **`MultitaperPSD(signal []float64, NW float64, K int, fs float64) (freqs, psd []float64)`** — Thomson 1982. Wraps DPSS. ~80 LOC. Closes slot 132 §2.2.

### Day-3 PR: `signal/window_advanced.go` (~120 LOC)
- `KBDWindow(n int, alpha float64, out []float64)` — Princen-Bradley 1986. Required by AAC/MP3 codec validation. Pin COLA identity to ULP.
- `BohmanWindow`, `ParzenWindow` — smoothness-prioritised windows for spectral correlogram.
- `BartlettHannWindow` — hybrid.
- `LanczosWindow` — `sinc(2i/(N-1) − 1)`. Used in image resampling.
- `CosineWindow` — 1-term `sin(πi/(N-1))`. Used as the rectangular's mildest taper.
- `DolphChebyshevWindow(n int, attenuationDB float64, out []float64)` — Dolph 1946 ultraspherical, optimal in equiripple sense (analogous of Parks-McClellan in window space). ~80 LOC, requires complex-arithmetic IDFT closed-form.

### Day-4 PR: API harmonisation + COLA support
- Add `Periodic bool` parameter (or `HannWindowPeriodic`, `HammingWindowPeriodic`) — required for STFT COLA at hop=N/4.
- `COLAGain(window []float64, hopSize int) float64` — verify constant-overlap-add gain for a window/hop pair.
- Document the periodic-vs-symmetric distinction in package doc.
- Cross-link `audio/spectrogram/stft.go` to use periodic form by default.

### Performance / API notes
- All closed-form windows: zero-alloc, single pass. Match existing style.
- The cosine-sum windows (Hamming, Blackman, BH3, BH4, Nuttall, Flat-top) share a common kernel `Σ_k a_k · cos(2πk·i/(N-1))`; can refactor into a single `cosineSumWindow(coeffs []float64, n int, out []float64)` private helper. Reduces total code to ~50 LOC for 6 windows.
- Kaiser is the only window where I₀ matters; share the I₀ implementation with slot 300 (`new-bessel-spherical`) once that lands. **Coordination note for slot 300**: a clean `bessel.I0(x)` is the single dependency for Kaiser, so ship I₀ first (or in the same PR).
- DPSS is the *only* window that requires `linalg`. Everything else stands alone in `signal/`.

### R-MUTUAL-CROSS-VALIDATION 3/3 pins (3 saturating opportunities)
1. **Symmetry pin** (3/3): `w[i] == w[n-1-i]` to ULP for every closed-form window across n=2..1024 — cross-validates against the formal symmetry property. (Pattern A: invariant.)
2. **Kaiser→Rectangular degeneration** (3/3): `‖KaiserWindow(N, 0) − RectangularWindow(N)‖∞ < 1e-15` — analytic limit cross-validates the I₀ implementation. (Pattern B: limit case.)
3. **DPSS Slepian-Pollak** (3/3, gated on slot 097): first-taper concentration eigenvalue λ₀ matches Slepian-Pollak analytic asymptote `1 - 12·exp(-2π·NW·(1-c))` for N=1024, NW=4. (Pattern C: analytic asymptote.)
4. **KBD COLA identity** (3/3): `KBD[i]² + KBD[i+N/2]² == 1` to ULP — defining property of KBD. (Pattern A: invariant.)
5. **Tukey α=1 ≡ Hann** (3/3): identical-to-ULP cross-validation across the parametric family. (Pattern B: limit case.)

### Cheapest day-1 deliverable (1 PR, 1 day)
- File: `signal/window_extra.go`, ~280 LOC
- 7 windows (Kaiser, Tukey, Gaussian, Welch, Bartlett, BH4, Nuttall, FlatTop)
- 1 metrics helper, 1 dispatch helper
- ~30 vectors per window (210 total) + 5 invariant pins per window (35) + cross-window degeneration tests (5) = **~250 new test vectors**
- Unblocks slot 132 (Welch PSD), aligns with slot 316 (FIR Kaiser), and gives consumers the entire SciPy/Matlab core window set minus DPSS.
- Defers DPSS/multitaper to Day-2 (gated on linalg slot 097), KBD/Bohman/Parzen/Lanczos/Dolph-Chebyshev to Day-3, and API-harmonisation/COLA to Day-4.

## Sources

### Repo files
- `C:\limitless\foundation\reality\signal\window.go` — current window catalog (3 windows + ApplyWindow, 113 lines)
- `C:\limitless\foundation\reality\signal\fft.go` — consumer (FFT)
- `C:\limitless\foundation\reality\signal\filter.go` — sister file (filters)
- `C:\limitless\foundation\reality\audio\spectrogram\stft.go:53,137` — STFT consumer (currently Hann-only by call sites)
- `C:\limitless\foundation\reality\audio\spectrogram\spectrogram_test.go:37,77,185,215,243,308` — Hann-only tests
- `C:\limitless\foundation\reality\audio\onset\superflux.go` — transitive consumer via STFT
- `C:\limitless\foundation\reality\audio\separation\nmf.go` — transitive consumer via STFT
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\316-dive-fir-design.md` — Kaiser deep-dive (slot 316 ships Kaiser as part of FIR PR — coordinate to avoid duplication; the Kaiser window itself belongs in `signal/window.go`/`window_extra.go`, not `signal/fir.go`)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\132-signal-missing.md` — flags Welch (§1.8) and DPSS multitaper (§2.2) as missing
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\133-signal-sota.md` — flags SciPy/librosa parity gaps
- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md:108` — slot 097 `linalg-missing` (DPSS gate); line 116 slot 105 `optim-perf`

### Web / literature (per task brief; not re-fetched, citations are canonical)
- **Harris 1978**: Harris, F. J., "On the use of windows for harmonic analysis with the discrete Fourier transform", *Proc. IEEE* 66(1), 51-83. **The** comprehensive window catalog with side-lobe levels, processing gain, scallop loss tabulated. Authoritative source for Blackman-Harris coefficients (`0.35875, 0.48829, 0.14128, 0.01168`).
- **Nuttall 1981**: Nuttall, A. H., "Some windows with very good sidelobe behavior", *IEEE Trans. ASSP* 29(1), 84-91. Source for Nuttall coefficients (`0.3635819, ...`); analyses 4-term derivatives for steepest sidelobe roll-off.
- **Slepian 1978**: Slepian, D., "Prolate spheroidal wave functions, Fourier analysis, and uncertainty — V: The discrete case", *Bell System Tech. J.* 57(5), 1371-1430. Defines DPSS; proves optimality of energy concentration.
- **Thomson 1982**: Thomson, D. J., "Spectrum estimation and harmonic analysis", *Proc. IEEE* 70(9), 1055-1096. Multitaper method using DPSS.
- **Walden-Percival 1993**: Percival, D. B. & Walden, A. T., *Spectral Analysis for Physical Applications*, Cambridge UP. §8.3 gives the DPSS tridiagonal eigenvalue formulation.
- **Heinzel-Rüdiger-Schilling 2002**: Heinzel, G., Rüdiger, A., Schilling, R., "Spectrum and spectral density estimation by the Discrete Fourier transform (DFT)", Max-Planck-Inst. Tech. Report. Table 1 lists ENB / scallop-loss / coherent-gain / flatness-loss for every window — **the** operational reference for `WindowMetrics()`. Source for flat-top SRS coefficients.
- **Kaiser 1974**: Kaiser, J. F., "Nonrecursive digital filter design using the I₀-sinh window function", *Proc. IEEE Int. Symp. Circuits and Systems*, 20-23. Defines Kaiser window via I₀.
- **Welch 1967**: Welch, P. D., "The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms", *IEEE Trans. AU* 15(2), 70-73. Defines Welch's method (windowed-segment averaging).
- **Tukey 1967**: Cooley, J., Lewis, P., Welch, P., "Historical notes on the Fast Fourier Transform", *IEEE Trans. AU* 15(2). Tukey window definition appears in Bingham-Godfrey-Tukey 1967 *IEEE Trans. AU* 15(2), 56-66 ("Modern techniques of power spectrum estimation").
- **Princen-Bradley 1986**: Princen, J. P., Bradley, A. B., "Analysis/synthesis filter bank design based on time domain aliasing cancellation", *IEEE Trans. ASSP* 34(5), 1153-1161. KBD window for AAC/Vorbis.
- **scipy.signal.windows reference** — `boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann, kaiser, gaussian, general_gaussian, dpss, exponential, tukey, taylor, chebwin, cosine, lanczos`. ~22 windows. Reality coverage today: 3/22 = 14%. Day-1 PR target: 11/22 = 50%. Day-3 target: 18/22 = 82%.

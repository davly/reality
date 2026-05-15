# 339 — dive-acoustic-hrtf (Minimum-phase / ITD / ILD / HRTF interpolation / binaural rendering audit)

## Headline
HRTF processing is **wholly absent** from `reality/`; the cheapest day-1 PR is `acoustics/hrtf/{minphase,itd}.go` (~200 LOC) — minimum-phase via real-cepstrum and ITD via FFT cross-correlation — both pure compositions of `signal.FFT`/`signal.IFFT` already in tree, blocked-soft only on the FFT-Hilbert primitive landing in dive-338.

## Findings

- **Confirmed gap.** `grep -E 'HRTF|HRIR|ITD|ILD|MinimumPhase|Cepstrum|Binaural|CrossCorrelat'` across all `*.go` returns **zero matches** in the repo. Acoustics is `acoustics/acoustics.go` only (196 LOC, 8 scalar closed-forms: `SoundSpeed`, `SoundIntensity`, `DecibelSPL`, `DecibelFromIntensity`, `SabineRT60`, `DopplerShift`, `ResonantFrequency`, `WaveLength`, `AWeighting`). No spatial / binaural / impulse-response code anywhere — confirmed by 002-acoustics-missing §"Binaural / spatial primitives" and 166-synergy-acoustics-signal §3 ("HRTF / binaural / cross-talk … out-of-scope for v1").
- **Substrate present.** `signal/fft.go:49` `FFT(real, imag []float64)` (radix-2, in-place), `:101` `IFFT`, `:140` `PowerSpectrum`, plus `signal/filter.go:19` `Convolve` and `signal/window.go` Hann/Hamming/Blackman are sufficient for every Tier-0/1/2 primitive below. Missing: real FFT (RFFT), `complex128` slice API, FFT-convolve — all flagged in 132-signal-missing.
- **Hard prerequisite from 338.** dive-338 (`agents/338-dive-hilbert-transform.md`) plans `signal.Hilbert` (FFT-based analytic-signal, ~80 LOC). **Minimum-phase reconstruction is *exactly* a Hilbert transform on the log-magnitude** (Smith / CCRMA: "Conversion to Minimum Phase"). The cleanest landing is `acoustics/hrtf/MinimumPhase` calling a future `signal.Hilbert`; without it, an inline real-cepstrum implementation works (FFT → log|·| → IFFT → causal-fold → exp → FFT), ~120 LOC.
- **Algorithmic canon is settled.** Wightman-Kistler 1989 (JASA 85(2):858) established that human HRTFs are *behaviourally* indistinguishable from their minimum-phase + pure-delay decomposition (Plogsties replicated 2000s; Kistler-Wightman 1992 added PCA + minphase). So the only spatial cue *not* captured by magnitude is the broadband ITD — extractable by FFT cross-correlation (or the equivalent Knapp-Carter 1976 GCC), or analytically via the inverse-Fourier of the unwrapped phase derivative (Algazi-Avendano-Duda 1999 ellipsoidal model). Brown-Duda 1998 reduces the entire structural HRTF model to head-shadow IIR + pinna-echo FIR + shoulder-echo FIR, all of which compose existing reality primitives.
- **No SOFA loader needed in `reality/`.** SOFA / AES69-2022 is NetCDF — explicitly out-of-scope for a zero-dependency math library. CIPIC raw `.wav`/`.mat` and any HRIR lives in the consumer (Pistachio / Howler). 166-§3 already classified HRTF databases as "data, not math."
- **R-MUTUAL pin is unusually clean here.** Three independent algorithms compute the *same* ITD on a synthetic pure-delay binaural pair — saturates 3/3 on a single golden vector (cross-correlation peak ≡ phase-derivative-at-DC ≡ generalised-cross-correlation-PHAT). The minimum-phase round-trip pin (|H_min| ≡ |H| to 1e-10) is the second mutually-cross-validating regression; a third is `ILD_dB ≡ -20·log10(rms(R)/rms(L))` ≡ frequency-band average of `20·log10(|H_R(f)| / |H_L(f)|)`.

## Concrete recommendations

Place all primitives in a NEW sub-package **`acoustics/hrtf/`** (mirrors the 166-recommended `acoustics/room/` and `acoustics/array/` precedent — keeps spatial-audio concerns out of the acoustics-101 root file). Each tier is a single file with a single golden test.

1. **T0 — `acoustics/hrtf/minphase.go` (~120 LOC).** `MinimumPhase(h []float64, out []float64)` via real-cepstrum (Oppenheim-Schafer §13, Smith CCRMA): pad to `2N`, FFT, take log-magnitude (`math.Log(math.Hypot(re,im))`, with NaN-guard floor at `1e-12` for spectral zeros), IFFT to cepstrum `c`, fold anticausal half (`c[k] += c[N-k]` for `k=1..N/2-1`, zero `k>N/2`), FFT, exponentiate (`exp(re)·(cos(im) + i·sin(im))`), IFFT real part → minimum-phase impulse. Allpass excess phase = `H(z) / H_min(z)` (compute spectrally, IFFT). Sources: Smith CCRMA "Minimum-Phase/Allpass Decomposition" + "Conversion to Minimum Phase"; Wightman-Kistler 1989 §II-C established this is the auditorily relevant decomposition for HRTFs. **R-MUTUAL pin (a):** `|FFT(MinimumPhase(h))| - |FFT(h)|` < 1e-10 on every bin (golden regression on 32 random length-128 IRs). After dive-338 lands, replace the inline cepstrum with `signal.Hilbert(logmag)` — the imaginary part *is* the minimum-phase phase response (saves ~40 LOC).
2. **T1 — `acoustics/hrtf/itd.go` (~80 LOC).** `ITDCrossCorr(left, right []float64, sampleRate float64) float64` via FFT-based cross-correlation: zero-pad to `2N`, `X = FFT(left)`, `Y = FFT(right)`, `R = X · conj(Y)`, `IFFT(R)`, fft-shift, locate peak (parabolic-interpolated for sub-sample precision: Boucher-Hassab 1981 — fit `y = a·τ² + b·τ + c` to `(τ-1, τ, τ+1)`, `τ* = -b/(2a)`). Returns ITD in seconds (positive = left leads). Source: Knapp-Carter 1976; Algazi-Avendano-Duda 1999. **R-MUTUAL pin (b):** on a synthetic stereo pair `right[n] = left[n - 27]` with a fractional-delay test (`delay = 27.43 samples`), three estimators (cross-correlation peak, phase-derivative `-dφ/dω` at low frequency, GCC-PHAT) agree to within 0.05 samples. **Day-1 deliverable.** Bonus: `ITDPhaseDerivative` is ~25 LOC after T0 lands the FFT scaffolding.
3. **T2 — `acoustics/hrtf/ild.go` (~80 LOC).** `ILD(left, right []float64) float64` returns broadband `20·log10(rms(right)/rms(left))` (positive = right louder). Plus `ILDSpectral(left, right []float64, sampleRate, fLow, fHigh float64) float64` for high-frequency ILD (the localisation-relevant band, > 1.5 kHz per Rayleigh duplex theory): FFT both, average `|R(f)|/|L(f)|` in the band, return dB. Source: Rayleigh 1907; Algazi-Duda CIPIC §I. **R-MUTUAL pin (c):** broadband ILD on a stereo pair scaled by a known `g` gives `±20·log10(g)` to 1e-12.
4. **T3 — `acoustics/hrtf/headmodel.go` (~150 LOC).** Brown-Duda 1998 structural model: spherical-head shadow as a 1st-order analog filter `H(s) = (1 + α·s/(2β))/(1 + s/(2β))` with `α(θ) = 1 + cos(θ - θ_ear)`, `β = c/a` (head radius `a≈8.75 cm`, sound speed `c` from `acoustics.SoundSpeed`); bilinear-transform to z-domain (per dive-315 IIR design when it lands; pre-bilinear inline ~30 LOC for now). Plus Woodworth-Schlosberg ITD: `ITD(θ) = (a/c)·(θ + sin θ)`, `θ ∈ [-π/2, π/2]`, the canonical low-frequency analytic ITD. **Cross-validation:** `WoodworthITD(θ)` on a synthesised Brown-Duda HRIR matches `ITDCrossCorr` to ~5% across `θ ∈ [-90°, 90°]`. Wires `acoustics/` into a single physically-motivated head model, no measurements required.
5. **T4 — `acoustics/hrtf/interp.go` (~200 LOC).** Two interpolators on a unit-sphere HRIR grid: (a) **bilinear-on-sphere** between the four nearest measured directions (cheapest, suffices for KEMAR 5° grid); (b) **spherical-harmonic** via real-SH basis (Condon-Shortley, ACN/SN3D) up to order N=4, projecting per-frequency-bin magnitudes onto SH coefficients then evaluating at query (θ, φ). The SH basis kernel itself is a 094-linalg-missing item also flagged by 003-acoustics-sota §"Real spherical harmonics"; if linalg lands `RealSH(l, m, theta, phi)`, this primitive collapses to ~80 LOC. **Crucial design choice:** interpolate the magnitude spectrum and the ITD *separately*, then re-attach delay (the Wightman-Kistler 1989 minphase + pure-delay model). Direct interpolation of complex HRIRs comb-filters and is wrong. Source: Marelli-Fitzgerald "Spherical harmonics interpolation for HRTF exchange" 2013 (the actual 2010-vintage paper search resolved to this).
6. **T5 — `acoustics/hrtf/render.go` (~250 LOC).** `BinauralRender(mono []float64, hrir [2][]float64, out [2][]float64)` = two-channel partitioned-block FFT-convolve (Gardner zero-latency, also flagged as 002-acoustics-missing #7). Input: per-direction (or per-frame) `[L, R]` HRIR pair from T4; output: stereo. Hot path requires no allocations (Pistachio 60 FPS rule). Blocks-soft on FFTConvolve (132-signal-missing) but a direct `signal.Convolve` fallback works at low framerates. Properly an `acoustics/render/` or consumer-side primitive — include here only as "API sketch", deliver in flagship.

## Tiering / Day-1 PR

- **Day-1 PR (cheapest, ~200 LOC):** **T0 + T1.** Both compose `signal.FFT`/`signal.IFFT` already in tree. T0 is a self-contained cepstrum implementation (no Hilbert dependency). T1 is an FFT-based cross-correlation with parabolic peak fit. Together they saturate two of the three R-MUTUAL pins on **one golden vector** (a synthetic delayed-and-magnitude-shaped stereo pair). Immediate utility for any consumer that has *measured* HRIRs (KEMAR is freely redistributable as raw IRs) and wants to (a) split them into magnitude + delay for safe interpolation and (b) compute cue-vector features `(ITD, ILD)` for direction-of-arrival classifiers.
- **Tier table (LOC, blockers):**

| Tier | Primitive | LOC | Blocker |
|------|-----------|-----|---------|
| T0 | `MinimumPhase` (real-cepstrum) | 120 | none (works today) |
| T1 | `ITDCrossCorr` + `ITDPhaseDerivative` | 80 | none |
| T2 | `ILD` + `ILDSpectral` | 80 | none |
| T3 | Brown-Duda head model + Woodworth ITD | 150 | bilinear z-transform (cleaner after dive-315) |
| T4 | HRTF interpolation (bilinear + real-SH) | 200 | 003-acoustics-sota real-SH basis |
| T5 | `BinauralRender` partitioned-FFT-convolve | 250 | FFTConvolve (132-signal-missing); SOFA loader = consumer |

## R-MUTUAL-CROSS-VALIDATION 3/3 pin

A single golden test vector saturates the pin: a synthetic binaural pair built as `right[n] = α · h_min(n) * left[n - τ_int] - β · h_min(n) * left[n - τ_int - 1]` with known `(α, β, τ_int, h_min)`. The vector forces:
- (a) `MinimumPhase(measured) → h_recon` with `|FFT(h_recon)| ≡ |FFT(measured)|` to 1e-10 (regression).
- (b) `ITDCrossCorr ≡ ITDPhaseDerivative ≡ ITDGCCPHAT` to 0.05 samples (regression on a fractional-delay 27.43-sample input).
- (c) `ILD_broadband ≡ -20·log10(rms(R)/rms(L))` ≡ band-averaged `20·log10(|R(f)|/|L(f)|)` to 1e-12 on a frequency-flat scaled signal.

This is the cleanest 3/3 pin in the spatial-audio block — three algorithmically distinct routes to one vector. Promote to STANDARD on first ship.

## Cross-links to the rest of the 400

- **338-dive-hilbert-transform.** Direct dependency: `signal.Hilbert(logmag)` simplifies T0 by ~40 LOC. Land 338 first if both are in flight.
- **166-synergy-acoustics-signal §3.** Explicitly defers HRTF to consumer with a 30-LOC `HRTFConvolve` wrapper. This dive *contradicts* the deferral mildly — the **math** (T0/T1/T2/T3) is in-scope; the **data loader** is not.
- **167-synergy-audio-signal A1 (Cepstrum-based F0).** Same homomorphic / log-magnitude / IFFT machinery. T0 here and A1 there should share a `signal.RealCepstrum` helper (~30 LOC) so dive-167 doesn't re-implement it. Flag for editorial pass.
- **003-acoustics-sota §"Real spherical harmonics".** T4 is the canonical consumer of an `linalg.RealSH(l,m,θ,φ)` primitive. Pulling SH into linalg unlocks T4 + every ambisonics primitive simultaneously.
- **132-signal-missing FFTConvolve.** T5 blocker; T0/T1/T2/T3 all unaffected.
- **Pistachio audio engine / VR-AR consumers.** All T0-T5 land HRTF *math* with zero new dependencies; consumer owns the SOFA / CIPIC NetCDF loader and wires (θ, φ) → HRIR lookup. Reality stays a math library.
- **Slot 197 (synergy-acoustics-fluids) and "beam-forming".** Beamforming lives in `acoustics/array/` per 166's plan — orthogonal to this dive. ITD/ILD here are *single-pair* primitives; multi-channel TDOA / GCC-PHAT for arrays belongs there. No conflict.

## Sources

**Repo files** (verified by grep):
- `C:\limitless\foundation\reality\acoustics\acoustics.go` — 196 LOC, 8 scalar closed-forms; no HRTF, no IR primitives.
- `C:\limitless\foundation\reality\signal\fft.go:49` `FFT`, `:101` `IFFT`, `:140` `PowerSpectrum` — substrate for T0/T1/T2.
- `C:\limitless\foundation\reality\signal\filter.go:19` `Convolve` — direct-convolution fallback for T5.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\002-acoustics-missing.md` §"Binaural / spatial primitives" — `HRTF interpolation`, `ITD / ILD`, `HRIR ↔ HRTF`, `B-format → binaural` all listed as missing.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\166-synergy-acoustics-signal.md` §3 — defers HRTF/binaural/cross-talk to consumer; provides the `acoustics/array/` + `acoustics/room/` placement precedent for the proposed `acoustics/hrtf/`.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\003-acoustics-sota.md` §"SOFA / HRTF stack" — AES69-2022, libspatialaudio 0.4, MagLS, real-SH ACN/SN3D.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\338-dive-hilbert-transform.md` — Hilbert/analytic-signal plan; T0 minphase is ~40 LOC shorter once it lands.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\167-synergy-audio-signal.md` A1/C3 — cepstrum F0 / cepstral envelope share homomorphic machinery with T0; flag a shared `signal.RealCepstrum`.

**Web sources** (web search 2026-05-09):
- Algazi, Duda, Thompson, Avendano (2001), "The CIPIC HRTF database," IEEE WASPAA, pp. 99–102 — [PDF, UC Davis](https://www.ece.ucdavis.edu/cipic/wp-content/uploads/sites/12/2015/04/cipic_WASSAP_2001_143.pdf).
- Wightman & Kistler (1989), "Headphone simulation of free-field listening. I: Stimulus synthesis," JASA 85(2):858 — [PubMed](https://pubmed.ncbi.nlm.nih.gov/2926000/) — establishes minphase + pure-delay HRTF model.
- Smith, J.O. (CCRMA), "Conversion to Minimum Phase" — [ccrma.stanford.edu](https://ccrma.stanford.edu/~jos/fp/Conversion_Minimum_Phase.html); "Minimum-Phase/Allpass Decomposition" — [ccrma.stanford.edu](https://ccrma.stanford.edu/~jos/filters/Minimum_Phase_Allpass_Decomposition.html); "Minimum-Phase and Causal Cepstra" — [ccrma.stanford.edu](https://ccrma.stanford.edu/~jos/sasp/Minimum_Phase_Causal_Cepstra.html).
- Brown & Duda (1998), "A structural model for binaural sound synthesis," IEEE Trans. Speech & Audio 6(5):476 — [Columbia mirror](https://www.ee.columbia.edu/~dpwe/papers/BrownD98-binsynth.pdf) — head-shadow IIR + pinna-echo FIR + shoulder-echo FIR.
- Algazi, Avendano, Duda (1999), "Estimation of a Spherical-Head Model from Anthropometry" / Woodworth-Schlosberg ITD — referenced via Algazi-Duda HRTF analyses.
- Knapp & Carter (1976), "The generalized correlation method for estimation of time delay," IEEE TASSP — canonical FFT-cross-correlation TDOA, GCC-PHAT.
- Marelli, Fitzgerald, Cooke, Andronoglou — "A study of spherical harmonics interpolation for HRTF exchange" — [ResearchGate](https://www.researchgate.net/publication/289035841_A_study_of_spherical_harmonics_interpolation_for_HRTF_exchange).
- AES69-2022 SOFA convention — [sofaconventions.org](https://www.sofaconventions.org/mediawiki/index.php/SOFA_(Spatially_Oriented_Format_for_Acoustics)) — referenced as out-of-scope (NetCDF dependency).

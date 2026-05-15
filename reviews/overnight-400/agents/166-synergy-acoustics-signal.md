# 166 | synergy-acoustics-signal

**Summary line 1.** `acoustics/` ships exactly 8 scalar closed-forms in a single 197-LOC file — `SoundSpeed`, `SoundIntensity`, `DecibelSPL`, `DecibelFromIntensity`, `SabineRT60`, `DopplerShift`, `ResonantFrequency`, `WaveLength`, `AWeighting` — with **zero** time-domain primitives, **zero** complex spectra, **zero** spatial geometry, **zero** vector arithmetic, and **zero** room/array model objects; `signal/` ships radix-2 `FFT`/`IFFT`/`PowerSpectrum`/`FFTFrequencies` + `Convolve` (direct O(NM)) + `MovingAverage`/`EMA`/`MedianFilter` + `Hann`/`Hamming`/`Blackman`/`ApplyWindow` (~470 LOC, all real `[]float64`, no `complex128`, no Hilbert, no fractional-delay, no FFT-convolve, no STFT — STFT lives in `audio/spectrogram/`); cross-package today: **zero direct imports**, the only seam is `audio/spectrogram/stft.go` which transitively brings `signal.FFT` into earshot of `acoustics.SabineRT60` without ever composing them. The entire spatial+room-acoustics canon (image-source RIR, Schroeder backward-integration RT60 from a measured IR, Velvet-noise late tail, EDT/C50/C80/D50, fractional-delay filterbanks, delay-and-sum/MVDR/LCMV/GSC/Frost beamforming, MUSIC/ESPRIT/SRP-PHAT/GCC-PHAT DOA, NLMS/RLS adaptive filters, HRTF binaural rendering, transaural cross-talk cancellation) is wholly absent from both packages.

**Summary line 2.** Twenty-two synergy primitives totalling ~3120 LOC of pure connective-tissue close the gap; thirteen ship today against the v0.10.0 surfaces (every primitive that is a numbers-in-numbers-out composition of `signal.FFT` + `acoustics.SoundSpeed`/`SabineRT60` + caller-supplied geometry); nine are blocked-soft on missing primitives independently flagged by 132/133-signal-missing (`Hilbert`/analytic-signal, `RFFT`, `FFTConvolve`, fractional-delay) and 003-acoustics-missing (`SabineRT60` exists but no `EyringRT60`/`ArauPuchades`, no `BarronT60` per-receiver, no spatial-receiver type); cheapest one-day standalone is **A1 SchroederRT60 + A8 ConvolveReverbFFT** at ~140 LOC giving the first round-trip RIR primitive in the entire repo (ImageSourceRIR → SchroederRT60 should round-trip to within `(c·Δt/Δx)·log2(N)` of the input `SabineRT60`, saturating an obvious 3/3 R-MUTUAL-CROSS-VALIDATION pin: Sabine forward × Schroeder backward × Eyring forward); keystone is **A2 ImageSourceRIR** (Allen-Berkley 1979) because every reverb/convolve/RT60-measurement/clarity-metric primitive below either consumes its output or measures something consistent with it; recommended placement is a NEW sub-package **`acoustics/room/`** for RIR + Schroeder + clarity (mirrors 158 `image/`, 159 `em/wave/`, 160 `fluids/...`, 165 `sequence/...` consumer-side-placement precedents) and a NEW sub-package **`acoustics/array/`** for beamforming + DOA (these need `complex128` slices and weight matrices that are neither pure-acoustic nor pure-signal). The two sub-packages share only the speed-of-sound constant — they decouple cleanly.

---

## 0. State of play (verified file-walk)

`acoustics/` HEAD (1 file `acoustics.go`, 197 LOC + tests in `acoustics_test.go`): exactly 8 exported pure-scalar closed-forms. Inputs are float64 scalars; outputs are float64. **Zero** complex types, **zero** slice/buffer APIs, **zero** geometry, **zero** time-stepping, **zero** spatial primitives. Verified: `grep -E '^func [A-Z]' acoustics/acoustics.go` returns 8 functions. The package doc explicitly enumerates "speed of sound, dB SPL, Sabine RT60, Doppler, A-weighting" — **room geometry, impulse response, beamforming, localisation are conspicuously absent.**

`signal/` HEAD (3 files `fft.go` / `filter.go` / `window.go`, ~470 LOC + tests). All operations 1-D `[]float64`:
  - **FFT family.** Cooley-Tukey radix-2 `FFT`/`IFFT` in-place on parallel real/imag slices (no `complex128` — design choice for cache-friendly SIMD), `PowerSpectrum` (writes N/2+1 magnitudes into `out`), `FFTFrequencies` (bin centres from sample rate). **No real-FFT optimisation, no Bluestein for non-pow-2, no Hilbert/analytic-signal, no group-delay, no zero-phase filtering, no FFT-convolve helper.**
  - **Filter family.** `Convolve` direct O(N·M) (line 19, comment explicitly notes "for long kernels, FFT-based convolution is preferred (compose FFT + multiply + IFFT from this package)" — i.e. caller must hand-compose); `MovingAverage` running-sum, `ExponentialMovingAverage` first-order IIR, `MedianFilter` with stack scratch ≤ 63 width.
  - **Window family.** `Hann`/`Hamming`/`Blackman`/`ApplyWindow`. **No Kaiser, no flat-top, no Gaussian, no Tukey** — flagged 132-signal-missing.
  - The package doc names consumers as "Pistachio (audio), RubberDuck (spectral analysis), Oracle (time series), Sentinel (filtering)" — **room acoustics, beamforming, microphone arrays are conspicuously absent.**

`audio/spectrogram/stft.go` (lines 1-100) builds on `signal.FFT` for STFT / inverse-STFT (Allen-Rabiner 1977 / Griffin-Lim 1984). This is the **only** existing composition of `signal/` with anything acoustic; it lives under `audio/`, not `acoustics/` or `signal/`. Critical for §1 below: every beamformer / DOA estimator that operates per-bin needs a per-channel STFT; `audio/spectrogram/Compute` is the substrate.

`audio/separation/` ships `WienerFilter`, `SubtractSpectrum`, `EstimateNoiseSpectrum`, `FastICA`, `NMF` (~1000 LOC). All single-channel except `FastICA` (multi-channel but operates on instantaneous mixtures with no time-shift / propagation). **Zero** spatial-receiver / spatial-source models.

`linalg/` ships `MatMul`, `Inverse` (LU-based), `CholeskyDecompose`/`CholeskySolve`, `QRAlgorithm` (eigenvalues only — no eigenvectors yet), `PCA`, `CovarianceMatrix`. **No complex-Hermitian eigendecomposition, no SVD, no pseudo-inverse, no matrix square-root.** Critical for the array-processing §2 below: MVDR needs `R⁻¹·a / (a^H·R⁻¹·a)`, MUSIC needs eigenvectors of a Hermitian covariance, ESPRIT needs SVD or generalised eigendecomposition. The complex-Hermitian eigendecomposition gap is independently flagged 094-linalg-missing.

`constants/` ships `SpeedOfLight`, `VacuumPermittivity`, `VacuumPermeability` but **no `SpeedOfSoundSTP` (343.0 m/s at 20°C)**. Every spatial-acoustics primitive below either needs it as input or pulls it from `acoustics.SoundSpeed` with `(γ=1.4, R=8.314, T=293.15, M=0.02896)`. Flagged 047-constants-missing as low-priority.

**Cycle-hazard check.** `grep -r 'acoustics' signal/*.go` and `grep -r 'signal' acoustics/*.go` — zero matches. Zero coupling today. Adding `acoustics/room/` and `acoustics/array/` that import `signal/` + `linalg/` + `acoustics/` is cycle-free; one-way: `acoustics/{room,array}/ → signal + linalg + acoustics + constants`. This mirrors 159 `em/wave/`, 158 `image/`, 165 `sequence/` placement precedent.

---

## 1. Room acoustics: twelve primitives in `acoustics/room/`

Each entry: (1) capability, (2) composition recipe over present primitives, (3) connective-tissue LOC, (4) ship-status.

### A0 — `acoustics/room` package keystone types

**Capability.** `Vec3{X, Y, Z float64}` (3-D point/source/receiver — promoted from `geometry/` if it exists there, else local). `ShoeboxRoom{Lx, Ly, Lz float64; AbsorptionCoeffs [6]float64}` for the canonical Allen-Berkley shoebox. `RIR{Sample []float64; SampleRate float64}` for impulse responses. Pure type definitions, no math.

**LOC.** 50.

**Status.** Ships today.

### A1 — `SchroederRT60(rir []float64, sampleRate float64) (rt60, edt float64)`

**Capability.** Backward-integration energy-decay-curve (EDC) → linear regression on log-EDC → RT60 from -5dB-to-35dB slope (ISO 3382-1 §A), EDT from 0-to-10dB. The canonical measurement-domain T60 — given any measured/simulated impulse response, compute the actual reverberation time. **First IR-domain measurement primitive in the repo.**

**Composition.** EDC[n] = Σ_{k=n}^{N-1} rir[k]² (one reverse cumulative sum). Convert to dB. Least-squares fit of a line over the [-5, -35] dB region (pure scalar regression — already a `linalg/correlation.go` pattern; could compose with `linalg.PearsonCorrelation` slope or do inline).

**LOC.** 80.

**Status.** SHIPS TODAY. Pure `[]float64` arithmetic. Composes with A2 for round-trip Sabine→ImageSourceRIR→SchroederRT60 validation.

**Reference.** Schroeder, M. R. (1965). "New method of measuring reverberation time." J. Acoust. Soc. Am. 37(3) — backward integration formula. ISO 3382-1:2009 §A — measurement procedure.

### A2 — `ImageSourceRIR(room ShoeboxRoom, src, rcv Vec3, c, fs float64, maxOrder int, length int) []float64` — KEYSTONE

**Capability.** Allen-Berkley 1979 image-source method for shoebox rooms. For each mirror image (i, j, k) ∈ [-N, N]³ and each of the 8 sign-permutations, place a virtual source at the mirrored location, compute distance to receiver, reflection-product attenuation `(1-α₁)^|i₁| · ... · (1-α₆)^|i₆|`, distance attenuation `1/d`, propagation delay `d/c`, and add a (fractionally-delayed) impulse at sample `round(d·fs/c)` with the corresponding amplitude. Returns the dry RIR sample buffer. Specular reflections only (Allen-Berkley assume rigid walls at idealised admittance).

**Composition.** Triple loop over image orders, single Vec3 distance, per-image scalar attenuation product, sample-bin accumulation. **No FFT** in this primitive — pure time-domain image-summing. Optional `signal.HannWindow` smoothing on the high-order tail.

**LOC.** 220.

**Status.** SHIPS TODAY. `acoustics/room.ShoeboxRoom` + `Vec3` → `[]float64` RIR. Pure arithmetic.

**Notes.** The keystone primitive of the entire `room/` sub-package. Every reverb / clarity / RT60-measurement / convolve-reverb primitive below either consumes its output or composes with something measured against it. Low fractional-delay accuracy at small samples; A3 below upgrades.

**Reference.** Allen, J. B. & Berkley, D. A. (1979). "Image method for efficiently simulating small-room acoustics." J. Acoust. Soc. Am. 65(4), 943-950 — the canonical paper, ~6800 citations.

### A3 — `ImageSourceRIRFractional(...)` (with sinc-windowed fractional delay)

**Capability.** Same as A2 but every image is placed via a 7-tap windowed-sinc fractional-delay filter (Smith 2010 §4.1) instead of nearest-sample rounding. Reduces high-frequency comb-filter artefacts at low room order × high `fs` ratios. The standard upgrade path from A2.

**Composition.** Reuses A2 image-summing loop; per-image, the impulse is convolved with `sinc((n - frac)/T) · HannWindow` over a 7-sample window. Composes with `signal.HannWindow` and a local `sinc` helper.

**LOC.** 100 (delta over A2).

**Status.** BLOCKED-SOFT on 132-signal-missing §6 (`signal.FractionalDelaySinc` primitive). The math is trivial; the missing piece is the windowed-sinc kernel as a re-usable signal/ primitive.

**Reference.** Smith, J. O. (2010). "Physical Audio Signal Processing" §4.1; Laakso et al. (1996). "Splitting the unit delay" IEEE SP Magazine.

### A4 — `VelvetNoiseTail(rt60, fs float64, density int, seed int64) []float64`

**Capability.** Stochastic late-reverberation tail using Karjalainen-Esquef velvet noise (sparse ±1 spikes at random positions) with exponential energy envelope fitted to the requested RT60. Computationally cheaper than full image-source for the diffuse late tail (>80ms post-direct, where image-source order >7 becomes prohibitive).

**Composition.** Single random-position-+-sign loop (deterministic with seeded LCG — same pattern as `prob/markov.go`'s embedded LCG flagged 117), per-sample multiply by `exp(-3·ln(10)·n/(fs·rt60))` envelope. **No FFT, no complex.**

**LOC.** 90.

**Status.** SHIPS TODAY (uses local seeded LCG). Coordinates with 117/165 on the broader prob/Rng debt — once `prob.Rng` lands, A4 should accept `*prob.Rng`.

**Reference.** Karjalainen, M. & Esquef, P. (2001). "Efficient modeling of late reverberation using time-varying feedback delay networks" / Välimäki & Reiss (2016) review.

### A5 — `EarlyDecayTime(rir []float64, fs float64) float64`

**Capability.** EDT (Early Decay Time) — slope of log-EDC fit over [0, -10] dB only. Perceptually correlates with reverberance more strongly than full-range T60 (ISO 3382-1).

**Composition.** Reuses A1's EDC computation; only the regression window changes. ~30 LOC delta.

**LOC.** 30.

**Status.** SHIPS TODAY (composes with A1).

**Reference.** Jordan, V. (1970). "Acoustical criteria for auditoria." ISO 3382-1:2009 §3.4.

### A6 — `ClarityIndex(rir []float64, fs float64, te float64) float64` — C50 / C80 / D50

**Capability.** Single primitive parameterised by `te` (early-energy cutoff time): `C(te) = 10·log10(Σ_{n<te·fs} rir[n]² / Σ_{n≥te·fs} rir[n]²)`. C50 (te=50ms) = speech clarity, C80 (te=80ms) = music clarity, D50 (definition) = `Σ_early / Σ_total` ratio (no log). All three from one ~30-LOC kernel.

**Composition.** Two scalar accumulators in one loop. Pure `[]float64` arithmetic. Composes with A2 directly (consumer of an RIR).

**LOC.** 50 (incl. doc + 3 thin wrappers C50/C80/D50).

**Status.** SHIPS TODAY.

**Reference.** Reichardt, W. et al. (1974). "Definitionsmaß und Klarheitsmaß." ISO 3382-1:2009 §3.5–§3.7.

### A7 — `EyringRT60(V, S, alphaMean float64) float64` and `ArauPuchades(V float64, surfaces []SurfacePatch) float64`

**Capability.** Eyring's RT60 `T = 0.161·V / (-S·ln(1-α̅))` (more accurate than Sabine for high-absorption rooms where Sabine over-estimates). Arau-Puchades for non-isotropic absorption (different α per axis pair). Companions to the existing `acoustics.SabineRT60`. Together they give cross-validation against the Schroeder-measured A1 — three independent RT60 estimators for the same room.

**Composition.** Pure scalar closed-forms. Should live next to `SabineRT60` in the top-level `acoustics.go` package (not in `acoustics/room/`) as scalar siblings.

**LOC.** 60.

**Status.** SHIPS TODAY. Independently named in 003-acoustics-missing as a top-priority gap.

**Reference.** Eyring, C. F. (1930). "Reverberation time in 'dead' rooms." J. Acoust. Soc. Am. 1(2), 217-241. Arau-Puchades, H. (1988). "An improved reverberation formula." Acustica 65.

### A8 — `ConvolveReverbFFT(dry, rir []float64, out []float64)` — keystone consumer

**Capability.** Apply a measured/simulated RIR to a dry signal via FFT-convolution (overlap-add): partition `dry` into blocks of size `B = nextpow2(len(rir))`, compute `signal.FFT(rir)` once, per-block FFT × pointwise-multiply × IFFT × overlap-add. The standard "convolution reverb" engine. **Replaces the current `signal.Convolve` direct O(N·M) path which is O(seconds) for typical 1-second RIRs at 48kHz.**

**Composition.** Three `signal.FFT` calls per block (dry-block, kernel once, result-block IFFT) + pointwise complex multiply + overlap-add. Block size auto-chosen as `nextpow2(2·len(rir))`. Hand-composes the in-place real/imag-pair convention.

**LOC.** 180.

**Status.** BLOCKED-SOFT on 132-signal-missing §3 (`signal.FFTConvolve` / `OverlapAdd` primitive). Can be implemented at `acoustics/room/` level today (~180 LOC) **or** lifted into `signal/` (~120 LOC) as a substrate primitive — the latter is preferred (132 §3 already proposes it). If 132 §3 lands first, A8 is 30 LOC of "use it for RIRs."

**Reference.** Stockham, T. G. (1966). "High-speed convolution and correlation." AFIPS Spring '66 — original FFT-convolve. Smith, J. O. (2010). "Spectral Audio Signal Processing" §8.

### A9 — `BarronEarlyLateRatio(rir []float64, fs float64) float64`

**Capability.** Barron's lateral-energy fraction LF80 (lateral early arrivals 25-80ms / total early 0-80ms) — perceptual spaciousness metric. Requires 2-channel figure-of-eight directivity input; for a single-omni RIR, returns the omnidirectional reduction.

**Composition.** Same energy-window pattern as A6 but with two channel inputs.

**LOC.** 60.

**Status.** Ships today (single-channel reduction). Full lateral-figure-of-8 form needs A11 (multi-channel RIR) which is the array bridge.

**Reference.** Barron, M. (1971). "The subjective effects of first reflections in concert halls." J. Sound Vib. 15(4).

### A10 — Round-trip golden file: `Sabine → ImageSourceRIR → Schroeder ≈ Sabine`

**Capability.** A SINGLE golden-file test vector that shipping forces three primitives (Sabine forward, ImageSource, Schroeder backward) to mutually cross-validate. Build a 5×4×3 m room with α=0.3 walls, compute Sabine's RT60, simulate the RIR with A2 at 48 kHz length=1.5s, measure Schroeder's RT60 from the simulated RIR — they should agree to within ~5% (the standard Allen-Berkley-Schroeder reconciliation tolerance documented across the literature). Saturates a 3/3 R-MUTUAL-CROSS-VALIDATION pin.

**Composition.** Test-only. ~50 LOC of Go test + ~3 KB of golden JSON (Sabine input, expected RIR samples at landmark indices, Schroeder T60 expected).

**LOC.** 50 (test).

**Status.** SHIPS TODAY once A1+A2 land. **This is the ship-criterion primitive** — without it, the room/ sub-package is unverified.

**Reference.** Karjalainen, M. (2002). "Estimation of reverberation time" — establishes the Sabine-Schroeder reconciliation tolerance for shoebox rooms.

### A11 — `MultichannelRIR(room, src Vec3, receivers []Vec3, ...) [][]float64`

**Capability.** Vector form of A2: one source, M receivers, returns M independent RIRs. Foundation for B-format Ambisonic, multi-mic recording, and beamforming-array IR measurement.

**Composition.** Pure loop wrapper over A2; trivial. Critical because every beamformer (§2 below) needs a per-channel RIR.

**LOC.** 30 (wrapper).

**Status.** Ships today (composes with A2).

### A12 — `NLMSEchoCancel(reference, observed []float64, mu float64, taps int) ([]float64, []float64)`

**Capability.** Normalised LMS adaptive filter (Haykin §6) for acoustic-echo cancellation: given a far-end reference and a near-end mic observation, identify the impulse response between far-end speaker and near-end mic and subtract its predicted echo. The canonical AEC algorithm. **First adaptive filter in the repo.**

**Composition.** Per-sample inner-product (taps long), residual subtraction, weight update `w[k] += μ·e·x[k] / (||x||² + ε)`. No `signal/` calls (works directly on `[]float64`); could compose with `signal.Convolve` for the residual computation.

**LOC.** 110.

**Status.** SHIPS TODAY. Pure adaptive filter. Belongs in **`signal/adaptive/`** as a substrate primitive (mirrors 132 proposal) — both `acoustics/room/` and `audio/separation/` consume it.

**Reference.** Widrow, B. & Hoff, M. (1960). "Adaptive switching circuits." IRE WESCON. Haykin, S. (2002). "Adaptive Filter Theory" §6.

---

## 2. Microphone-array processing: ten primitives in `acoustics/array/`

### B0 — `acoustics/array` package keystone types

**Capability.** `MicArray{Positions []Vec3; SampleRate float64}` (M-mic geometry), `SteeringVector(arr MicArray, doa Vec3, freq float64) []complex128` (per-bin Vandermonde-style phase vector `a[m] = exp(-j·2π·f·d_m·cos(θ)/c)`). Pure types + the steering-vector helper. Every beamformer below consumes this.

**LOC.** 80.

**Status.** SHIPS TODAY. Pure arithmetic; depends on `complex128` slice — natively supported by Go.

### B1 — `DelaySumBroadband(channels [][]float64, arr MicArray, doa Vec3, fs float64) []float64`

**Capability.** Time-domain delay-and-sum: for each channel, fractional-delay-shift by `-d_m·cos(θ)/c` samples, then sum. Broadband, no per-bin phase. The textbook beamformer.

**Composition.** M channels × M fractional-delay convolutions × N-sample sum. Composes with A3's windowed-sinc fractional-delay kernel. If A3 ships, B1 is ~80 LOC.

**LOC.** 120 (incl. inline fractional delay).

**Status.** BLOCKED-SOFT on 132-signal-missing §6 (`signal.FractionalDelaySinc`). Without it, ~120 LOC.

**Reference.** Van Veen, B. D. & Buckley, K. M. (1988). "Beamforming: A versatile approach to spatial filtering." IEEE ASSP Magazine 5(2).

### B2 — `FilterSumPerBin(channelsSTFT [][][]complex128, arr MicArray, doa Vec3, freqs []float64) [][]complex128`

**Capability.** Frequency-domain filter-and-sum: per STFT bin `(t, k)`, multiply each channel by `conj(steering[m, k]) / M` and sum. Equivalent to delay-sum but per-bin → handles dispersive arrays cleanly.

**Composition.** Reuses `audio/spectrogram.Compute` per channel + the B0 `SteeringVector` + per-bin sum. **No new substrate.**

**LOC.** 100.

**Status.** SHIPS TODAY. Direct composition of `audio/spectrogram` + `acoustics/array.SteeringVector`.

### B3 — `MVDRWeights(R []complex128, M int, steering []complex128) []complex128` — KEYSTONE

**Capability.** Capon / Minimum-Variance Distortionless-Response weights `w = (R⁻¹·a) / (a^H·R⁻¹·a)` per bin, where `R` is the M×M Hermitian covariance estimate from the channel STFT cross-spectrum and `a` is the steering vector toward the desired DOA. The mainstream optimal (in the MMSE-distortionless sense) narrowband beamformer. **Crown jewel.**

**Composition.** Per-bin: estimate `R` from `signal.FFT` of each channel × outer-product accumulation, invert R via `linalg.Inverse` (LU on real-valued), normalise by quadratic form `a^H R⁻¹ a`. Diagonal-loading regulariser `R + λI` for stability (Cox-Zeskind-Owsley 1987).

**LOC.** 200.

**Status.** BLOCKED-HARD on 094-linalg-missing §1 (`linalg.ComplexInverse` for `complex128` matrices — current `linalg.Inverse` is real-only). Workaround: use the real-valued 2M×2M block matrix `[[ReR, -ImR], [ImR, ReR]]` for inversion (~30 LOC overhead). With workaround: 230 LOC ships today.

**Reference.** Capon, J. (1969). "High-resolution frequency-wavenumber spectrum analysis." Proc. IEEE 57(8), 1408-1418. Frost, O. L. (1972). "An algorithm for linearly constrained adaptive array processing." Proc. IEEE 60(8).

### B4 — `LCMVWeights(R []complex128, constraints [][]complex128, responses []complex128) []complex128`

**Capability.** Linearly-Constrained Minimum-Variance generalisation of B3: instead of one distortionless constraint, support multiple — null-steering toward known interferer DOAs while passing the desired DOA. `w = R⁻¹·C·(C^H·R⁻¹·C)⁻¹·f`.

**Composition.** Same complex-Hermitian inversion machinery as B3 + a small K×K inversion on the `C^H R⁻¹ C` Gram. Reuses B0 `SteeringVector` to build C.

**LOC.** 130.

**Status.** Same complex-inverse blocker as B3. With block-2M workaround: 160 LOC.

**Reference.** Frost, O. L. (1972) ibid. Buckley, K. M. & Griffiths, L. J. (1986). "An adaptive generalized sidelobe canceller with derivative constraints." IEEE Trans. AP 34.

### B5 — `GSCBeamformer(channels [][]complex128, doa Vec3, blocking [][]complex128) []complex128`

**Capability.** Generalised Sidelobe Canceller (Griffiths-Jim 1982) — equivalent re-formulation of LCMV as fixed-quiescent + blocking-matrix + adaptive-noise-canceller. Decomposes the constrained optimisation into an unconstrained adaptive problem solvable by NLMS (A12). The standard real-time implementation.

**Composition.** B2 fixed-quiescent + blocking-matrix multiply (caller supplies; canonical Griffiths-Jim form is `B = I - aa^H/M`) + A12 NLMS adaptive on the blocked outputs.

**LOC.** 180.

**Status.** Composes with B2 + A12. Ships today once both land. The cleanest example of three-primitive synergy in the entire review.

**Reference.** Griffiths, L. J. & Jim, C. W. (1982). "An alternative approach to linearly constrained adaptive beamforming." IEEE Trans. AP 30(1).

### B6 — `MUSICDOA(R []complex128, M, K int, scanGrid []Vec3, freq float64, arr MicArray) []float64`

**Capability.** Multiple-Signal-Classification spectrum: eigendecompose covariance `R`, take the M-K smallest-eigenvalue eigenvectors as the noise-subspace `E_n`, scan `P_MUSIC(θ) = 1 / (a(θ)^H · E_n · E_n^H · a(θ))` over the DOA grid. Pseudo-spectrum peaks → DOA estimates. The benchmark super-resolution DOA.

**Composition.** Hermitian eigendecomposition (currently `linalg.QRAlgorithm` does **eigenvalues only** — need eigenvectors, flagged 094 §3) + per-grid-point quadratic form. `audio/spectrogram` provides the per-bin covariance estimate.

**LOC.** 220.

**Status.** BLOCKED-HARD on 094-linalg-missing §3 (Hermitian eigendecomposition with eigenvectors). The standalone Jacobi-rotation Hermitian eigensolver is ~200 LOC and is the keystone unblocker for B6, B7.

**Reference.** Schmidt, R. O. (1986). "Multiple emitter location and signal parameter estimation." IEEE Trans. AP 34(3), 276-280 — the canonical paper.

### B7 — `ESPRITDOA(R []complex128, M, K int, arr MicArray) []float64`

**Capability.** Estimation of Signal Parameters via Rotational Invariance Techniques (Roy-Kailath 1989) — exploits a translational-invariance of the array (uniform-linear) to recover DOAs as eigenvalues of a small rotation matrix without grid-scanning. More efficient than MUSIC for ULA geometry.

**Composition.** SVD or generalised eigendecomposition of two sub-array signal-subspace matrices + final K×K eigendecomposition for the rotational angles.

**LOC.** 200.

**Status.** BLOCKED-HARD on 094 §3 + 094 §2 (no SVD). With Hermitian eigensolver substitute: 240 LOC.

**Reference.** Roy, R. & Kailath, T. (1989). "ESPRIT — Estimation of signal parameters via rotational invariance techniques." IEEE ASSP 37(7).

### B8 — `SRPPHAT(channels [][]float64, arr MicArray, fs float64, scanGrid []Vec3) []float64`

**Capability.** Steered-Response-Power with PHAse Transform (DiBiase 2000): for each candidate DOA in scanGrid, sum across all M(M-1)/2 channel pairs the GCC-PHAT cross-correlation evaluated at the DOA-implied lag. Robust to reverberation. The most-cited modern DOA estimator.

**Composition.** Per-pair GCC-PHAT (B9) → per-DOA lag-table → sum-and-argmax. Composes B9 + B0 SteeringVector.

**LOC.** 200.

**Status.** Ships today (composes with B9, B0). Crown-jewel real-world DOA — used by every smart-speaker beamformer in production.

**Reference.** DiBiase, J. H. (2000). "A high-accuracy, low-latency technique for talker localization in reverberant environments using microphone arrays." Brown PhD thesis. Brandstein, M. & Silverman, H. (1997). "A robust method for speech signal time-delay estimation in reverberant rooms." ICASSP.

### B9 — `GCCPHAT(x, y []float64, fs float64) (lag float64, peak float64)`

**Capability.** Generalised Cross-Correlation with PHAse Transform — Knapp-Carter 1976. Compute `S_xy = X·Y*` per bin, normalise by `|X·Y*|` (PHAT pre-whitens to spectral flatness), inverse-FFT, find peak → time-delay between two channels. The substrate of B8 and the canonical pairwise TDOA.

**Composition.** Pad-to-2N → `signal.FFT(x)` → `signal.FFT(y)` → per-bin `X·Y*/|X·Y*|` (with `+ε` regulariser) → `signal.IFFT` → arg-max + parabolic-peak refinement.

**LOC.** 130.

**Status.** SHIPS TODAY. Pure composition of `signal.FFT` + `signal.IFFT`. **The cheapest one-day standalone primitive in the entire array sub-package.**

**Reference.** Knapp, C. H. & Carter, G. C. (1976). "The generalized correlation method for estimation of time delay." IEEE ASSP 24(4), 320-327.

### B10 — `SuperdirectiveBeamformer(R_iso []complex128, arr MicArray, doa Vec3, freqs []float64) [][]complex128`

**Capability.** Eigenbeamformer / MaxDirectivity-Index beamformer assuming isotropic noise field (closed-form covariance `R_iso[m,n] = sinc(2π·f·d_{mn}/c)`). Optimal narrowband superdirective weights `w = R_iso⁻¹ a / (a^H R_iso⁻¹ a)`. Standard for compact mic arrays (smartphones, hearing aids).

**Composition.** Identical structure to B3 with `R` replaced by the analytic isotropic covariance. ~50 LOC delta over B3.

**LOC.** 120 (incl. dependency on B3 inversion).

**Status.** Same complex-inverse blocker as B3. With workaround: 150 LOC.

**Reference.** Bitzer, J. & Simmer, K. U. (2001). "Superdirective microphone arrays" in Brandstein & Ward eds. "Microphone Arrays."

---

## 3. Bridges to audio/ + HRTF/binaural

Existing `audio/separation/{Wiener, SubtractSpectrum, FastICA, NMF}` + `audio/spectrogram/{Compute, Inverse}` slot in: **MVDR+Wiener post-filter** (zero new LOC; doc cascade), **convolutive STFT-ICA** (Smaragdis 1998, ~150 LOC, out-of-scope v1), **NMF+Wiener per-bin mask per source** (~80 LOC).

HRTF / binaural / cross-talk: needs HRTF dataset (KEMAR/CIPIC/IRCAM) — **data, not math** — so out-of-scope for `reality/`. Math sketches: HRTFConvolve = 2-channel A8 wrapper (~30 LOC), CrosstalkCancellation = per-bin 2×2 complex-inverse (~80 LOC, complex-inverse blocker), Ambisonic→binaural decoder (~100 LOC). Properly live in a flagship consumer.

---

## 5. LOC ledger and ship priority

| ID | Primitive | LOC | Status v0.10.0 | Blocker |
|----|-----------|-----|-----|----|
| A0 | `room` types | 50 | SHIPS | — |
| A1 | `SchroederRT60` | 80 | SHIPS | — |
| A2 | `ImageSourceRIR` | 220 | SHIPS | — |
| A3 | `ImageSourceRIRFractional` | 100 | BLOCKED-SOFT | 132 §6 fractional-delay |
| A4 | `VelvetNoiseTail` | 90 | SHIPS | (coord 117 prob/Rng) |
| A5 | `EarlyDecayTime` | 30 | SHIPS | A1 |
| A6 | `ClarityIndex` C50/C80/D50 | 50 | SHIPS | — |
| A7 | `EyringRT60` + Arau-Puchades | 60 | SHIPS | — |
| A8 | `ConvolveReverbFFT` | 180 | BLOCKED-SOFT | 132 §3 FFTConvolve (else 180 LOC inline) |
| A9 | `BarronEarlyLateRatio` | 60 | SHIPS | partial |
| A10 | round-trip golden-file | 50 | SHIPS | A1+A2 |
| A11 | `MultichannelRIR` | 30 | SHIPS | A2 |
| A12 | `NLMSEchoCancel` | 110 | SHIPS | — |
| B0 | `array` types + steering | 80 | SHIPS | — |
| B1 | `DelaySumBroadband` | 120 | BLOCKED-SOFT | 132 §6 (else 120 inline) |
| B2 | `FilterSumPerBin` | 100 | SHIPS | — |
| B3 | `MVDRWeights` | 200 | BLOCKED-HARD | 094 complex-inverse (block-2M workaround +30) |
| B4 | `LCMVWeights` | 130 | BLOCKED-HARD | 094 complex-inverse |
| B5 | `GSCBeamformer` | 180 | SHIPS | A12 + B2 |
| B6 | `MUSICDOA` | 220 | BLOCKED-HARD | 094 §3 Hermitian eig+vecs |
| B7 | `ESPRITDOA` | 200 | BLOCKED-HARD | 094 §2 SVD or §3 |
| B8 | `SRPPHAT` | 200 | SHIPS | B9 + B0 |
| B9 | `GCCPHAT` | 130 | SHIPS | — |
| B10 | `SuperdirectiveBeamformer` | 120 | BLOCKED-HARD | 094 complex-inverse |

**Total: 22 primitives, ~3 020 LOC of pure connective tissue.**

**Ships today against v0.10.0 (no blockers): A0, A1, A2, A4, A5, A6, A7, A9, A10, A11, A12, B0, B2, B5, B8, B9 = 16 primitives, ~1700 LOC.**

**Cheapest one-day MVP.** A1 + A2 + A8(inline) + A10 = ~530 LOC = first round-trip RIR primitive in repo + 3/3 R-MUTUAL-CROSS-VALIDATION pin.

**Cheapest one-day array primitive.** B9 (GCC-PHAT) at 130 LOC = first DOA primitive, foundation for B8/SRP-PHAT.

**Highest-yield-per-LOC.** A2 ImageSourceRIR (220 LOC) — keystone consumer of every measurement primitive A1/A5/A6/A9, every beamformer in §2 (via A11 MultichannelRIR), every adaptive filter in A12. 220 LOC unlocks the entire room-acoustics + binaural + spatial-audio canon.

**Crown-jewel keystone.** A2 + B3-MVDR — together the canonical "simulate-then-process" pipeline that every real-time spatial-audio system in the literature uses. B3 is currently the highest-priority blocked-hard (linalg complex-Hermitian inverse) item; landing 094-linalg-missing §1 unblocks B3, B4, B10 simultaneously.

---

## 6. Distinctness from neighbouring agents

- **001-005 acoustics isolation.** Per-package; identifies `EyringRT60`/`ArauPuchades`/`BarronT60` (A7, A9) but NOT the cross-package composition with `signal.FFT`/`Convolve`. This review is the cross-seam.
- **006-010 audio isolation.** Per-package; identifies STFT consumer surface but not RIR generation, beamforming, DOA. Orthogonal.
- **131-135 signal isolation.** Identifies `Hilbert`, `RFFT`, `FFTConvolve`, `FractionalDelay` gaps (132 §3, §6) — this review is the *consumer* of those gaps; coordinates with 132 on which side ships first. **Recommendation: signal/ ships FFTConvolve + FractionalDelay first, room/ + array/ become thinner.**
- **091-095 linalg isolation.** Identifies complex-Hermitian-inverse + Hermitian-eig-with-vecs + SVD as 094 §1/§2/§3 — this review is the highest-leverage consumer (unblocks B3/B4/B6/B7/B10 = 5 primitives). **Coordinates on landing 094 §3 first** (Hermitian eig with vecs) since it unblocks 3 of the 5.
- **117 prob debt** (LCG / Rng). A4 VelvetNoiseTail uses local LCG today; once `prob.Rng` lands, A4 trivially adopts. Same pattern as 165 sequence-prob. Coordinates.
- **151 synergy-signal-prob.** Orthogonal (statistical-signal-processing seam, not spatial). No overlap.
- **158 synergy-color-signal, 159 synergy-em-signal, 160 synergy-fluids-signal.** All siblings of *this* review on the signal-as-substrate axis. Each defines a NEW sub-package consuming `signal/`. Architectural alignment: this review proposes `acoustics/room/` + `acoustics/array/` (consumer-side placement) per the 158/159/160 precedent.
- **162 synergy-graph-prob, 163 synergy-optim-autodiff, 164 synergy-orbital-optim, 165 synergy-sequence-prob.** Topically orthogonal but share the "20+ primitives totalling ~2k-3k LOC of connective tissue" idiom. This review fits the established Block-B synergy-review template.

---

## 7. Recommended ship sequence

1. **Day 1 (~530 LOC).** A0+A1+A2+A10 — first round-trip RIR primitive + 3/3 R-MUTUAL-CROSS-VALIDATION pin.
2. **Day 2 (~290 LOC).** A5+A6+A7+A11+A12 — closes the standalone room-acoustics measurement surface.
3. **Day 3 (~310 LOC).** B0+B9+B8 — closes the fully-shipping DOA surface.
4. **Day 4 (depends on 094).** Once `linalg.HermitianEig(eigenvalues, eigenvectors)` ships → B6 MUSIC + B3 MVDR (block-2M workaround) → 700 LOC of crown-jewel array processing.
5. **Out-of-scope-for-v1.** Convolutive ICA, HRTF binaural, cross-talk cancellation — flagged for consumer (flagships/howler) or follow-up review.

The two sub-packages `acoustics/room/` + `acoustics/array/` close the room-acoustics + mic-array canon at v0.10.0 + 094 + 132. Cycle-free, one-way imports, zero perturbation to existing surfaces.

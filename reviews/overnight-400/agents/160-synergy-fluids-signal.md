# 160 | synergy-fluids-signal

**Summary line 1.** `fluids/` and `signal/` are two of the 22 v0.10.0 sub-packages that today have **zero source coupling** (`grep -r 'signal' fluids/*.go` and `grep -r 'fluids' signal/*.go` both return empty) and share **no common type beyond `float64`/`[]float64`**: `fluids/fluids.go` is a 236-LOC catalog of 11 scalar closed-form steady-state functions (`ReynoldsNumber`, `BernoulliPressure`, `PipeFlowFriction`, `DarcyWeisbach`, `DragForce`, `LiftForce`, `TerminalVelocity`, `StokesLaw`, `MassFlowRate`, `VolumetricFlowRate`) with **zero** time dependence, **zero** vector fields, **zero** spectral notions; `signal/` ships only `FFT`/`IFFT`/`PowerSpectrum`/`FFTFrequencies`/`Convolve`/`MovingAverage`/`EMA`/`MedianFilter`/`Hann`/`Hamming`/`Blackman`/`ApplyWindow` — no Welch/Bartlett, no coherence, no Hilbert/STFT, no wavelet, no real-FFT, no cross-spectrum. The entire **turbulence-as-signal canon** (Kolmogorov 1941 inertial-range `E(k) ∝ k^{-5/3}`, Obukhov-Corrsin scalar `k^{-5/3}`, Taylor 1938 frozen-turbulence hypothesis, structure functions `S_p(r) = ⟨(δu)^p⟩` with She-Léveque intermittency, Kolmogorov microscale `η = (ν³/ε)^{1/4}`, Taylor microscale `λ`, integral length `L = ∫R(τ)dτ`, Reynolds stress tensor `⟨u'_i u'_j⟩`, TKE budget, Welch PSD, Bartlett PSD, magnitude-squared coherence `γ²(f) = |Sxy|²/(Sxx·Syy)`, POD via two-point correlation eigendecomposition, Lumley 1967 snapshot POD, Schmid 2010 DMD, Towne-Schmidt-Colonius 2018 SPOD, vortex-identification `Q`/`Δ`/`λ₂`-criteria) is **wholly absent from both packages and the wider repo**.

**Summary line 2.** Eighteen synergy primitives totalling ~2150 LOC of pure connective tissue close the gap; **eleven ship today** against the v0.10.0 surfaces (every primitive that is `FFT`-composable or pure stats over `[]float64`); **seven** are blocked on missing primitives independently flagged in agent 132-signal-missing (`Welch`, `Bartlett`, `Coherence`, `RFFT`, `Hilbert`, `STFT`, `WaveletCWT`) and 074-linalg-missing (`SymEigen` symmetric eigendecomposition, `SVD` singular value decomposition — `linalg/eigen.go` ships only `QRAlgorithm` for general eigenvalues, no eigenvectors); cheapest standalone day-one is **T1 EnergySpectrumE_k + T2 KolmogorovScale + T3 IntegralLengthScale** at ~110 LOC, giving the first turbulence-spectral primitives in the entire repo and saturating an obvious 3/3 R-MUTUAL-CROSS-VALIDATION pin (**dissipation rate ε computed three ways — direct from spectrum integral `ε = 2ν∫k²E(k)dk`, from Taylor microscale `ε = 15ν⟨u'²⟩/λ²`, from Kolmogorov scale inversion `ε = ν³/η⁴` — all three must agree to ≤2% on a Pao-spectrum synthetic field**, mirroring 159 W1's d'Alembert/FFT cross-validation and the recent 6a55bb4 audio-onset 3-detector saturation pattern); keystone is **T7 POD via snapshot method** because spectral-POD, DMD, and Reynolds-stress eigendecomposition all share its symmetric eigensolve path; recommended placement is a **NEW sub-package `fluids/turbulence/`** (mirrors 159's `em/wave/` precedent and 151's `prob/spectral/`, 158's `image/`, 157's `graph/spectral.go` — turbulence-spectral primitives are neither pure-fluids nor pure-signal: they encode Navier-Stokes statistical theory and require `signal.FFT` + `linalg.SymEigen` + `fluids` parameters jointly).

---

## 0. State of play (verified file-walk)

`fluids/fluids.go` HEAD (1 file, 236 LOC, 11 exported funcs):
- **Dimensionless:** `ReynoldsNumber`
- **Bernoulli/pipe:** `BernoulliPressure`, `PipeFlowFriction` (Colebrook-White iterative), `DarcyWeisbach`
- **Aerodynamic forces:** `DragForce`, `LiftForce`, `TerminalVelocity`
- **Low-Re:** `StokesLaw`
- **Flow rates:** `MassFlowRate`, `VolumetricFlowRate`

Every function is a single-line algebraic closed form except `PipeFlowFriction` (Colebrook-White fixed-point loop). **Zero** state, **zero** time evolution, **zero** PDE solver, **zero** vector/tensor types. The package doc declares "classical fluid mechanics … numbers in, numbers out." Turbulence theory (statistical, spectral, modal) is conspicuously absent.

`signal/` HEAD (3 files, ~470 LOC total):
- `fft.go`: `FFT`, `IFFT`, `PowerSpectrum`, `FFTFrequencies` — radix-2 Cooley-Tukey, in-place, panics on non-power-of-2.
- `window.go`: `HannWindow`, `HammingWindow`, `BlackmanWindow`, `ApplyWindow`.
- `filter.go`: `Convolve`, `MovingAverage`, `ExponentialMovingAverage`, `MedianFilter`.

Consumer comment block names "Pistachio (audio), RubberDuck (spectral), Oracle (time-series), Sentinel (filtering)" — fluid-mechanics consumers are conspicuously absent.

**Cross-package observations:**

- `linalg/eigen.go` ships `QRAlgorithm(A, n, eigenvalues, maxIter)` returning **only eigenvalues, not eigenvectors**, and only for general (non-symmetric) matrices. **POD/DMD/SPOD all need eigenvectors + symmetric eigensolve.** Independently flagged 074-linalg-missing.
- `linalg/pca.go` ships `PCA(data, nSamples, nFeatures, nComponents, components, explained)` — this is precisely the snapshot-method POD on a centered design matrix; **POD is PCA on velocity snapshots**. T7 below is mostly a thin alias + provenance comment.
- `linalg/correlation.go` ships `Covariance`, `CovarianceMatrix` — Reynolds stress tensor is `⟨u'_i u'_j⟩`, exactly `CovarianceMatrix` of fluctuating velocity components. T6 below is ~30 LOC.
- `chaos/` ships `RK4` for ODEs but no PDE; no Burgers, no Navier-Stokes, no spectral-method DNS. Out of scope here but synergistic with this review.
- `constants/` lacks `KolmogorovConstant ≈ 1.5` (universal in Kolmogorov 1941; flagged 047-constants-missing alongside Karman, Prandtl, Schmidt, von Kármán-Howarth-Monin defaults).
- `prob/` ships `Mean`, `Variance`, `Skewness`, `Kurtosis` — these are exactly the moments needed for structure-function flatness `F(r) = ⟨(δu)⁴⟩/⟨(δu)²⟩²` and intermittency. T4 below is ~25 LOC composing `prob.Variance` and a finite-difference loop.
- Cycle-hazard: `fluids/turbulence/` importing `signal/` + `linalg/` + `prob/` + `fluids/` is a clean DAG (`fluids/` itself imports nothing). Zero cycle risk.

---

## 1. The eighteen synergy primitives

Each: (1) capability, (2) composition recipe over present primitives, (3) connective-tissue LOC, (4) ship-status against v0.10.0. Numbering T0–T17.

### T0 — `fluids/turbulence` package keystone types

**Capability.** `VelocitySeries{U []float64; Dt float64; SampleRate float64}` for single-probe time series (Taylor frozen-turbulence converts to `Dx = Umean·Dt`); `VelocityField{Nx, Ny, Nz int; Dx float64; U, V, W []float64}` for 3-D snapshot (flat row-major); `SnapshotMatrix{Nspace, Nsnap int; Data []float64}` for POD/DMD (rows = spatial DoFs, cols = time snapshots — matches `linalg.PCA` layout). Pure type definitions, no math. Allocation discipline: caller pre-allocates everything (CLAUDE.md §3 60-FPS invariant — Pistachio acoustics already uses `signal.FFT` at 60 FPS, so the same buffer regime applies).

**LOC.** 80.

**Status.** Ships today. Pure types.

### T1 — `EnergySpectrumE_k(velocity []float64, dt, Umean float64, real, imag, window, out []float64)`

**Capability.** Compute one-sided 1-D longitudinal energy spectrum `E_11(k_1)` from a single-probe streamwise velocity time series via Taylor's frozen-turbulence hypothesis (`k_1 = 2πf/Umean`). This is **the** canonical turbulence-from-experiment kernel — every wind-tunnel hot-wire / LDV / PIV-time-resolved paper since 1948 uses it.

**Composition.**
1. Subtract mean: `U' = U - mean(U)` (open-coded, ~5 LOC; `prob.Mean` reusable).
2. Apply Hann window: `signal.HannWindow(N, win)` then `signal.ApplyWindow(U', win, U'_w)`.
3. Copy to `real[i] = U'_w[i]`, zero `imag`.
4. `signal.PowerSpectrum(real, imag, out)` → `|X(f)|²`.
5. Convert to one-sided PSD: `S_uu(f) = 2·|X|²·Δt/N / W₂` where `W₂ = Σwᵢ²/N` is the Hann window energy correction.
6. Convert frequency to wavenumber: `k_1 = 2πf/Umean`, `E_11(k_1) = S_uu(f)·Umean/(2π)`.

Step 5's `W₂` correction is the standard Welch-style window-energy normalization (Heinzel-Rüdiger-Schilling 2002 §10) — ~15 LOC including the Σwᵢ² accumulator.

**LOC.** ~40 (composition only; reuses `signal.HannWindow`/`ApplyWindow`/`PowerSpectrum`/`FFTFrequencies` and `prob.Mean`).

**Status.** **SHIPS TODAY.** Standalone PR. Pure composition.

**Notes.** This is the cheapest first-light primitive in this entire review and immediately unlocks the Kolmogorov `-5/3` pin: synthesize a Pao spectrum `E(k) = α·ε^{2/3}·k^{-5/3}·exp(-1.5α(kη)^{4/3})`, IFFT to a velocity field, run T1, regress `log E vs log k` over the inertial range, expect slope `-5/3 ± 0.05` and Kolmogorov constant `α ≈ 1.5 ± 0.05` (Sreenivasan 1995 PoF compendium).

### T2 — `KolmogorovScale(epsilon, nu float64) (eta, tau, uEta float64)`

**Capability.** Returns the three Kolmogorov microscales: length `η = (ν³/ε)^{1/4}`, time `τ_η = (ν/ε)^{1/2}`, velocity `u_η = (νε)^{1/4}`. The smallest scales of turbulence — sets the DNS resolution requirement (`Δx ≤ η`).

**Composition.** Three calls to `math.Pow`. No `signal/` or `fluids/` dependency.

**LOC.** ~12.

**Status.** **SHIPS TODAY.** Trivial standalone.

**Reference.** Kolmogorov, A.N. (1941) "The local structure of turbulence in incompressible viscous fluid for very large Reynolds numbers" Doklady Akad. Nauk SSSR 30:301-305.

### T3 — `IntegralLengthScale(velocity []float64, dt, Umean float64) float64`

**Capability.** Integral length scale `L_11 = ⟨u'²⟩⁻¹ ∫₀^∞ R_11(r) dr` where `R_11(r) = ⟨u'(x)u'(x+r)⟩`. Largest energy-containing scale — used for Reynolds-number-based DNS sizing and LES turbulence-model calibration.

**Composition.**
1. Compute autocorrelation via Wiener-Khinchin: `signal.PowerSpectrum` → `signal.IFFT` → real part = `R(τ)`. (Wiener-Khinchin is the FFT-based path; ~25 LOC.)
2. Convert lag `τ` to space `r = Umean·τ` (Taylor frozen-turbulence).
3. Integrate `R(r)/R(0)` from 0 to first zero crossing (trapezoidal rule — `calculus.IntegrateTrapezoid` already ships; ~10 LOC composition).

**LOC.** ~50 (the Wiener-Khinchin path; could be ~20 if reused from T15 below).

**Status.** **SHIPS TODAY.** Composes `signal.FFT`/`IFFT`/`PowerSpectrum` + `calculus.IntegrateTrapezoid`. No missing primitives.

### T4 — `StructureFunction(velocity []float64, p int, rMax int, out []float64)`

**Capability.** `p`-th order longitudinal structure function `S_p(r) = ⟨|u(x+r) - u(x)|^p⟩` for `r = 1..rMax` lags. Inertial-range scaling `S_p(r) ∝ r^{ζ_p}` with K41 prediction `ζ_p = p/3` and the She-Léveque (1994) intermittency correction `ζ_p = p/9 + 2(1 - (2/3)^{p/3})`. Flatness `F(r) = S_4(r)/S_2(r)²` quantifies intermittency.

**Composition.** Two nested loops, `prob.Mean` of `|δu|^p`. ~25 LOC.

**LOC.** ~25.

**Status.** **SHIPS TODAY.** Pure stats — no `signal/` calls.

**Cross-validation.** S_2(r) and `E(k)` are Fourier pairs (Monin-Yaglom 1975 §13.3): `S_2(r) = 2∫₀^∞ E(k)(1 - sin(kr)/(kr))dk`. Saturates an R-MUTUAL-CROSS-VALIDATION pin alongside T1 + T15.

### T5 — `DissipationRate(spectrum []float64, k []float64, nu float64) float64`

**Capability.** Energy dissipation rate from spectrum integral: `ε = 2ν∫₀^∞ k²·E(k) dk`. Pivots between T1 (spectrum), T2 (Kolmogorov scales), and T6 (TKE budget). Requested in topic.

**Composition.** Trapezoidal rule over `k²·E(k)`. ~15 LOC.

**LOC.** ~15.

**Status.** **SHIPS TODAY.**

**Cross-validation pin.** ε from T5 = ε from T2-inversion (`ε = ν³/η⁴`) = ε from Taylor-microscale `ε = 15ν⟨u'²⟩/λ²` — three independent estimators, must agree to ≤2% on a Pao-spectrum synthetic, saturates 3/3 R-MUTUAL-CROSS-VALIDATION (matches recent 6a55bb4 audio-onset and 365368a copula×autodiff saturation precedents).

### T6 — `ReynoldsStressTensor(u, v, w []float64, out []float64)`

**Capability.** 3×3 Reynolds stress tensor `R_ij = ⟨u'_i u'_j⟩` — the closure quantity at the heart of every RANS turbulence model (k-ε, k-ω, RSM). Eigendecomposition gives anisotropy invariants on the Lumley triangle.

**Composition.** Direct call to `linalg.CovarianceMatrix(data, out)` after stacking `[u; v; w]` row-major. ~12 LOC (mostly the row-major repack into `CovarianceMatrix`'s expected layout).

**LOC.** ~12.

**Status.** **SHIPS TODAY.** Thin wrapper over `linalg.CovarianceMatrix`. Anisotropy-tensor invariants `b_ij = R_ij/(2k) - δ_ij/3` are 5 more LOC.

### T7 — `POD_Snapshot(snapshots []float64, nSpace, nSnap, nModes int, modes, energies []float64) float64`

**Capability.** Sirovich (1987) snapshot POD — eigendecomposition of the temporal correlation matrix `C_tt = (1/N)·X^T·X` (size `nSnap×nSnap` ≪ `nSpace×nSpace`). Returns `nModes` spatial modes and their energy fractions. **POD is PCA on velocity snapshots** — Lumley 1967 / Berkooz-Holmes-Lumley 1993.

**Composition.** Direct delegation to `linalg.PCA(data, nSamples=nSnap, nFeatures=nSpace, nComponents=nModes, components, explained)`. Snapshot-method orientation: rows = time, cols = space. Returns total variance from `linalg.PCA`'s return value.

**LOC.** ~20 (mostly orientation comments and provenance — the actual call is one line).

**Status.** **SHIPS TODAY.** `linalg.PCA` already does the heavy lifting; T7 is a fluids-domain entry point with the right doc/citation.

**Reference.** Sirovich, L. (1987) "Turbulence and the dynamics of coherent structures, I. Coherent structures" Q. Appl. Math. 45:561-571.

### T8 — `DMD(snapshots []float64, nSpace, nSnap int, rank int, modes, eigvals []float64)`

**Capability.** Schmid (2010) Dynamic Mode Decomposition — Koopman operator approximation. Decomposes a snapshot sequence `X = [x_0, x_1, …, x_{N-1}]`, `Y = [x_1, x_2, …, x_N]` into modes `φ_i` with complex eigenvalues `λ_i = exp((σ_i + iω_i)·Δt)`. Each mode oscillates at fixed frequency `ω_i` with growth rate `σ_i`.

**Composition.** Standard exact-DMD recipe: (1) economy SVD `X = UΣV^T`, (2) `Ã = U^T Y V Σ^{-1}`, (3) eigendecompose `Ã = WΛW^{-1}`, (4) `Φ = YVΣ^{-1}W`. Steps 1, 3, 4 each need `linalg.SVD` and `linalg.Eigen` (with eigenvectors).

**LOC.** ~120 if all primitives existed.

**Status.** **BLOCKED.** `linalg/` ships `QRAlgorithm` (eigenvalues only) but no `Eigen` (with eigenvectors), no `SVD`. Both flagged independently in 074-linalg-missing. T8 ships once those land.

**Reference.** Schmid, P.J. (2010) "Dynamic mode decomposition of numerical and experimental data" J. Fluid Mech. 656:5-28.

### T9 — `SpectralPOD(snapshots []float64, nSpace, nSnap int, nBlocks, blockOverlap int, freqIdx int, modes, energies []float64)`

**Capability.** Towne-Schmidt-Colonius (2018) Spectral POD — POD applied to the cross-spectral density tensor at each frequency, recovers space-time-coherent structures. Requires Welch-style block segmentation in time + FFT per spatial point + eigendecompose CSD matrix at each frequency.

**Composition.** (1) Welch-segment each spatial point's time series (T11), (2) FFT each block, (3) at each frequency build CSD matrix `S_xx(f) = (1/M)Σ_m X̂_m(f)·X̂_m(f)^H`, (4) `linalg.SymEigen` on Hermitian CSD matrix.

**LOC.** ~180 if SymEigen + complex linalg existed.

**Status.** **BLOCKED.** Needs `signal.Welch` (132-signal-missing §1.5), `signal.RFFT` for complex spectra (132 §1.1), Hermitian-eigendecomposition (074-linalg-missing). 4-quarter horizon.

**Reference.** Towne, A., Schmidt, O.T., Colonius, T. (2018) "Spectral proper orthogonal decomposition and its relationship to dynamic mode decomposition and resolvent analysis" J. Fluid Mech. 847:821-867.

### T10 — `QCriterion`, `DeltaCriterion`, `Lambda2Criterion(velGradTensor []float64, n int) float64`

**Capability.** Three vortex-identification scalars from the velocity-gradient tensor `∇u`:
- `Q = ½(‖Ω‖² - ‖S‖²)` (Hunt-Wray-Moin 1988) — 2nd invariant of `∇u`.
- `Δ = (Q/3)³ + (R/2)²` (Chong-Perry-Cantwell 1990) — `Δ > 0` ⇒ complex eigenvalues ⇒ swirl.
- `λ_2(S² + Ω²)` (Jeong-Hussain 1995) — second-largest eigenvalue of pressure-Hessian-like tensor; `λ_2 < 0` ⇒ vortex core.

**Composition.** Decompose `∇u = S + Ω` (symmetric strain rate + antisymmetric rotation); compute Frobenius norms (Q), invariants (Δ), eigendecompose `S² + Ω²` (λ_2). λ_2 needs `linalg.SymEigen` for a 3×3 symmetric matrix — 074 flags missing, but a closed-form 3×3 cubic-root eigensolver (Smith 1961, Kopp 2008) is ~40 LOC standalone, so **λ_2 ships standalone** without waiting on general `SymEigen`.

**LOC.** ~30 (Q, Δ) + ~50 (λ_2 with closed-form 3×3 eigensolve) = ~80.

**Status.** **SHIPS TODAY** with closed-form 3×3 path for λ_2. Drops to ~50 LOC once `linalg.SymEigen` lands.

**References.** Hunt, J.C.R., Wray, A.A., Moin, P. (1988) CTR-S88; Chong, M.S., Perry, A.E., Cantwell, B.J. (1990) Phys. Fluids A 2:765; Jeong, J., Hussain, F. (1995) JFM 285:69-94.

### T11 — `WelchPSD(signal []float64, segLen, overlap int, window []float64, sampleRate float64, out []float64)`

**Capability.** Welch (1967) PSD: segment signal into overlapping blocks, window each, FFT each, average `|X|²`. Standard variance-reduced spectral estimator — every modern turbulence post-processor uses Welch over plain periodogram.

**Composition.** Loop over segments, `signal.ApplyWindow` + `signal.PowerSpectrum`, accumulate, divide by segment count + window-energy correction.

**LOC.** ~80 if it lived in `signal/`.

**Status.** **BLOCKED on `signal.Welch`**. Independently flagged 132-signal-missing §1.5. Lives more naturally in `signal/` than `fluids/turbulence/`.

**Reference.** Welch, P.D. (1967) "The use of fast Fourier transform for the estimation of power spectra" IEEE Trans. Audio Electroacoust. 15:70-73.

### T12 — `BartlettPSD(signal []float64, segLen int, sampleRate float64, out []float64)`

**Capability.** Bartlett (1948) PSD — special case of Welch with zero overlap and rectangular window. Smaller variance reduction than Welch but trivially simple. Useful for educational examples and as a Welch validation reference.

**Composition.** Like T11 with `overlap = 0`, no windowing. ~30 LOC if in `signal/`.

**LOC.** ~30.

**Status.** **BLOCKED on `signal.Bartlett`** (132-signal-missing). Or could ship inside `fluids/turbulence/` as a thin wrapper composing existing `signal.PowerSpectrum` over manually-segmented input — ~50 LOC, **SHIPS TODAY** that way. Recommend `signal/` placement instead.

### T13 — `CoherenceMagnitudeSquared(x, y []float64, segLen, overlap int, sampleRate float64, out []float64)`

**Capability.** Magnitude-squared coherence `γ²(f) = |S_xy(f)|² / (S_xx(f)·S_yy(f))` — the spectral analog of correlation, ∈ [0,1]. Pressure-velocity coherence and two-probe coherence (topic-listed) directly use this. Coherence > 0.5 in a frequency band indicates a linear relationship, < 0.1 indicates noise/independence.

**Composition.** Three Welch-style estimates: auto-spectra `S_xx`, `S_yy` and cross-spectrum `S_xy = ⟨X*·Y⟩` averaged across segments. Per bin: `γ²[k] = (Re(S_xy)² + Im(S_xy)²) / (S_xx · S_yy)`.

**LOC.** ~120 once `signal.Welch` exists; ~150 standalone with manual Welch loop.

**Status.** **BLOCKED on signal.Welch + complex cross-spectrum** (RFFT preferred). Could ship today as a 150-LOC self-contained primitive — recommend waiting for 132's `Welch` + `CrossSpectrum`.

**Reference.** Carter, G.C. (1987) "Coherence and time delay estimation" Proc. IEEE 75:236-255.

### T14 — `TaylorMicroscale(velocity []float64, dt, nu float64) (lambda, ReLambda float64)`

**Capability.** Taylor microscale `λ = √(15ν⟨u'²⟩/ε)` and Taylor-Reynolds `Re_λ = u'_rms·λ/ν`. The intermediate scale between integral L and Kolmogorov η; turbulence-canonical "Reynolds number" because it depends only on a single-point statistic.

**Composition.** `prob.Variance` of fluctuating velocity + T5 (DissipationRate) + algebra. ~20 LOC.

**LOC.** ~20.

**Status.** **SHIPS TODAY** (after T5).

### T15 — `Autocorrelation(signal []float64, out []float64)` and `CrossCorrelation(x, y []float64, out []float64)`

**Capability.** Wiener-Khinchin autocorrelation `R_xx(τ) = IFFT(|X|²)` and cross-correlation `R_xy(τ) = IFFT(X*·Y)`. Foundation for T3 (integral length scale) and T13 (coherence) and broadly used outside turbulence (chaos, prob, audio, RF). Belongs in `signal/`, not `fluids/turbulence/`.

**Composition.** Zero-pad to 2N for linear (vs. circular) correlation, FFT, multiply, IFFT, take real part. ~50 LOC.

**LOC.** ~50.

**Status.** **SHIPS TODAY** in `signal/` as a standalone primitive. Recommend ship-now in `signal/` (it's not turbulence-specific) and have T3 + T13 compose it.

**Notes.** Independently called out in 132-signal-missing as `Autocorrelation`/`CrossCorrelation`. This review reinforces that ask: T3, T13, and several chaos-timeseries primitives in agent 154 all want it.

### T16 — `WaveletCWT_Morlet(signal []float64, scales []float64, out [][]float64)`

**Capability.** Continuous wavelet transform with Morlet basis — multi-scale time-frequency decomposition of intermittent turbulent signals. Farge (1992) showed CWT-Morlet captures coherent vortex packets that Fourier modes smear across all wavenumbers.

**Composition.** For each scale `a`, `out[a][n] = Σ_t signal[t]·ψ((t-n)/a)/√a` where `ψ(x) = π^{-1/4}·exp(iω₀x)·exp(-x²/2)`. ~120 LOC standalone (real-Morlet variant; full complex Morlet doubles the storage and is in 132's wavelet ask).

**LOC.** ~120.

**Status.** **BLOCKED on `signal.Wavelet`** (132-signal-missing §1.7). Or ship a real-Morlet standalone today.

**Reference.** Farge, M. (1992) "Wavelet transforms and their applications to turbulence" Annu. Rev. Fluid Mech. 24:395-457.

### T17 — `TKEBudget(u, v, w, dpdx, dpdy, dpdz []float64, nu float64, out *TKEBudgetResult)`

**Capability.** Turbulent kinetic energy budget terms: `dk/dt = P + T + Π - ε + D` (production, turbulent transport, pressure diffusion, dissipation, viscous diffusion). Closes the `RANS` model loop — every turbulence-modeling paper since Pope 2000 §5 plots these. Requires velocity gradients, pressure-velocity correlations, and ε (T5).

**Composition.** Decomposes into ~6 spatially-averaged correlations: `prob.Variance` for k itself, `linalg.Covariance` for production `P_ij = -⟨u'_i u'_j⟩∂U_i/∂x_j`, T5 for ε, finite-difference operators for transport terms.

**LOC.** ~150.

**Status.** **SHIPS TODAY** at the 1-D (homogeneous) level using existing `prob` + `linalg.Covariance`. Full 3-D inhomogeneous version blocked on a finite-difference helper that lives more naturally in `chaos/` or a future `pde/` package.

**Reference.** Pope, S.B. (2000) Turbulent Flows §5, CUP.

---

## 2. Composition LOC summary

| ID | Primitive | LOC | Status |
|----|-----------|-----|--------|
| T0 | Package types (VelocitySeries, VelocityField, SnapshotMatrix) | 80 | Ships today |
| T1 | `EnergySpectrumE_k` (Taylor frozen-turbulence + Hann + PSD) | 40 | **Ships today** |
| T2 | `KolmogorovScale` (η, τ_η, u_η) | 12 | **Ships today** |
| T3 | `IntegralLengthScale` (Wiener-Khinchin + trapezoidal) | 50 | **Ships today** |
| T4 | `StructureFunction` p-th order | 25 | **Ships today** |
| T5 | `DissipationRate` (∫k²E(k)) | 15 | **Ships today** |
| T6 | `ReynoldsStressTensor` (alias `linalg.CovarianceMatrix`) | 12 | **Ships today** |
| T7 | `POD_Snapshot` (alias `linalg.PCA`) | 20 | **Ships today** |
| T8 | `DMD` (Schmid 2010) | 120 | Blocked on SVD + Eigen-vectors |
| T9 | `SpectralPOD` (Towne-Schmidt-Colonius 2018) | 180 | Blocked on Welch + Hermitian Eigen |
| T10 | `Q`/`Δ`/`λ_2` vortex criteria | 80 | **Ships today** (closed-form 3×3) |
| T11 | `WelchPSD` (in signal/) | 80 | Blocked on signal.Welch |
| T12 | `BartlettPSD` | 30 | Blocked on signal.Bartlett (or 50 LOC standalone) |
| T13 | `CoherenceMagnitudeSquared` (two-probe γ²) | 120 | Blocked on Welch + cross-spectrum |
| T14 | `TaylorMicroscale` (λ, Re_λ) | 20 | **Ships today** (after T5) |
| T15 | `Autocorrelation`/`CrossCorrelation` (in signal/) | 50 | **Ships today** in signal/ |
| T16 | `WaveletCWT_Morlet` | 120 | Blocked on signal.Wavelet (or 120 standalone) |
| T17 | `TKEBudget` (1-D homogeneous) | 150 | **Ships today** |
| —  | **Total connective tissue** | **~2150** | **11 ship today; 7 blocked** |

---

## 3. Recommendations

1. **Create `fluids/turbulence/` sub-package.** Mirrors 159's `em/wave/` precedent. Imports `fluids/`, `signal/`, `linalg/`, `prob/`, `calculus/`, `constants/`. One-way DAG; cycle-free.
2. **Day-1 PR — the 110-LOC turbulence triple** (T1 + T2 + T5 + T14 = ~90 LOC composition + 20 LOC cross-validation tests):
   - Synthesize Pao spectrum `E(k) = α·ε^{2/3}·k^{-5/3}·exp(-1.5α(kη)^{4/3})` with known `ε, ν, α=1.5`.
   - IFFT to a velocity field via `signal.IFFT`.
   - Run T1 → recover `E(k)`, fit slope of `log E vs log k` over `[10η⁻¹, 0.1L⁻¹]`, expect `-5/3 ± 0.05` and `α ≈ 1.5 ± 0.05`.
   - **3/3 R-MUTUAL-CROSS-VALIDATION pin on ε**: T5 spectrum integral, T2 inversion `ε = ν³/η⁴`, T14 Taylor-microscale `ε = 15ν⟨u'²⟩/λ²` — must agree to ≤2% (matches recent 6a55bb4 audio-onset 3-detector and 365368a copula×autodiff saturation precedents).
3. **Day-2 PR — T6 + T7 + T10** (Reynolds stress + snapshot POD + Q/Δ/λ_2). All thin layers over `linalg`. Adds the modal-decomposition vocabulary.
4. **Constants ask** (orthogonal to this review): add `KolmogorovConstant ≈ 1.5`, `KarmanConstant ≈ 0.41`, `PrandtlMixingLength` placeholder. Flagged 047-constants-missing.
5. **Signal asks the turbulence sub-package depends on** (already independently flagged in 132-signal-missing): `Autocorrelation`/`CrossCorrelation` (T15 today), `Welch`/`Bartlett` (T11/T12), `Hilbert` (instantaneous-frequency for inertial-range envelope), `RFFT` (cross-spectrum), `STFT` (time-frequency turbulence), `Wavelet` (T16).
6. **Linalg asks the turbulence sub-package depends on** (074-linalg-missing): `SVD`, `Eigen` (with eigenvectors), `SymEigen` (Hermitian).
7. **Naming.** Use `EnergySpectrumE_k` not `EnergySpectrum` to disambiguate from acoustic/audio energy-spectra (which are `S(f)` not `E(k)`). Similarly `KolmogorovScale` not just `Microscale`.
8. **Cross-package consumer.** `chaos/` Lorenz/Rossler trajectories naturally feed `T7 POD_Snapshot` and `T8 DMD` once it ships (treat trajectory as a snapshot matrix). Synergistic with agent 154-synergy-chaos-timeseries.
9. **Documentation.** Every primitive cites: Kolmogorov 1941 (T1, T2, T4, T5), Taylor 1938 (T1, T3), Pope 2000 (T17), Sirovich 1987 (T7), Schmid 2010 (T8), Towne-Schmidt-Colonius 2018 (T9), Hunt-Wray-Moin 1988 / Chong-Perry-Cantwell 1990 / Jeong-Hussain 1995 (T10), Welch 1967 / Bartlett 1948 (T11, T12), Carter 1987 (T13), Sreenivasan 1995 (cross-validation pin), Farge 1992 (T16), She-Léveque 1994 (T4 intermittency).

---

## 4. Anti-recommendations

- **Don't** put POD/DMD primitives inside `linalg/` — they are turbulence-domain entry points with provenance comments. Keep `linalg.PCA`/`linalg.SVD` generic; add `fluids/turbulence/POD_Snapshot` as the consumer-side wrapper. Mirrors how `audio/spectrogram/` consumes `signal.FFT` rather than `signal/` shipping a `Spectrogram`.
- **Don't** add a `Turbulence` parameter struct that bundles ν, ε, L, η — these come from different measurements and forcing functions to take individual scalars (CLAUDE.md "numbers in, numbers out").
- **Don't** ship a Fourier-Galerkin DNS solver here — that's a `chaos/` or future `pde/` problem, not a synergy-fluids-signal one.
- **Don't** wait on T8 DMD before shipping T7 POD; POD has its own value (modal energy ranking, ROM construction) and T7 is ~20 LOC over the existing `linalg.PCA`.
- **Don't** put `Welch`/`Bartlett`/`Coherence` in `fluids/turbulence/` — they are general signal primitives. The only fluids-specific layer is the Taylor-frozen-turbulence wavenumber conversion in T1.

---

## 5. Verdict

The fluids × signal synergy gap is **wide but not deep**: every block boundary is a single named missing primitive (`Welch`, `RFFT`, `SymEigen`) already independently flagged elsewhere. The synergy review changes the *priority order*: turbulence consumers want `Welch` and `SymEigen` more than audio consumers do (audio is well-served by `Hann + PowerSpectrum`; turbulence cannot characterize the inertial range without a variance-reduced PSD). Day-1 the 110-LOC turbulence triple lands the first Kolmogorov `-5/3` pin and a 3/3 R-MUTUAL-CROSS-VALIDATION on ε, exactly matching the recent commit-log saturation pattern. The keystone is **T7 POD_Snapshot** because spectral POD, DMD, Reynolds-stress eigendecomposition, and Lumley-triangle anisotropy all share the symmetric-eigensolve path it codifies — once that path ships, the remaining seven blocked primitives become pure connective tissue.

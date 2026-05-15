# 337 — dive-fft-fractional (Chirp-Z / Bluestein / FrFT / NUFFT / Zoom-FFT audit)

## Headline
Reality has only Cooley-Tukey radix-2 FFT (`signal/fft.go`, panics on non-pow-2); the entire Bluestein / Chirp-Z / Zoom-FFT / FrFT / NUFFT family is absent — Bluestein arbitrary-N FFT is the obvious ~150-LOC day-1 PR (composes existing FFT, three-way pin trivial).

## Findings

### Repo state
- **Only callable transforms:** `signal.FFT`, `signal.IFFT`, `signal.PowerSpectrum`, `signal.FFTFrequencies` in `C:\limitless\foundation\reality\signal\fft.go`. All four panic if `N` is not a power of 2 (`fft.go:54`, `fft.go:106`, `fft.go:145`, `fft.go:168`).
- **Repo-wide grep** for `ChirpZ|CZT|Bluestein|FractionalFourier|FrFT|NUFFT|ZoomFFT` over `*.go`: **zero hits in source**. All matches are in prior agent reviews (132-signal-missing §1.3 / §2.10 / §3.9, 133-signal-sota, 134-signal-api, 293-new-ntt N4, 301-dive-fft-correctness §6).
- **Implication for callers:** audio at 44100 / 48000 Sa cannot FFT directly, must either zero-pad to 65536 (1.49× / 1.36× cost penalty + spectral leakage from the implicit rectangular window over the pad) or truncate to 32768. Pistachio (60 FPS) and Oracle (time-series with arbitrary N) both pay this tax.
- **Slot 132 §1.3 already specs Bluestein** as a missing primitive; slot 301 §6 names it as the third gap. **Slot 337 (this) is the deep-dive on the same primitive plus its generalisations** (CZT, zoom-FFT, FrFT, NUFFT) — not redundant; this slot covers the FrFT/NUFFT axis that 132/301 do not.

### Algorithm landscape (six primitives, tiered)

**T0 — Bluestein arbitrary-N FFT.** Bluestein 1968 / Rabiner-Schafer-Rader 1969. Identity `n·k = ((n+k)² − n² − k²)/2` rewrites the DFT as a chirp-multiplied convolution: `X[k] = w^{-k²/2} · Σ_n (x[n] · w^{-n²/2}) · w^{(k−n)²/2}` where `w = e^{−2πi/N}`. The convolution is computed by zero-padding to `M = next_pow2(2N−1)` and invoking the existing radix-2 `FFT`/`IFFT`. **~150 LOC.** Composes existing primitives. Lifts the power-of-2 panic for any caller. Cost: `O(N log N)` with constant ~6× the native pow-2 path (three FFTs + chirp multiplies). Allocates 4×`M` float64 scratch — provide `BluesteinFFTInto(re, im, scratch)` for hot paths.

**T1 — Chirp-Z Transform (general A, W).** Rabiner-Schafer-Rader 1969 *The Chirp z-Transform Algorithm*. Same Bluestein machinery but evaluates `X[k] = Σ_n x[n] · A^{−n} · W^{nk}` for arbitrary complex `A` (starting point on a spiral, default `A=1`) and arbitrary complex `W` (ratio between consecutive points, default `W = e^{−2πi/M}` for `M` output bins). **~80 LOC delta over T0** — reuses the same convolution skeleton, just two extra parameters and slightly different chirp prefactors. M output points need not equal N input points.

**T2 — Zoom-FFT.** Direct `CZT` application with `A = e^{2πi·f₁/fs}`, `W = e^{−2πi·(f₂−f₁)/(M·fs)}` to evaluate `M` bins linearly spaced between `f₁` and `f₂` — arbitrary frequency resolution within a band. **~40 LOC** wrapper over T1. Use cases: pitch-tracking around a known fundamental, narrowband radar bin, vibration analysis around a known machine resonance.

**T3 — Discrete fractional Fourier transform (DFrFT).** Candan-Kutay-Ozaktas 2000 *The discrete fractional Fourier transform*, IEEE TSP 48(5):1329-1337. Builds DFrFT matrix `F^a` for any rotation angle `α = aπ/2` using the eigendecomposition of the centered DFT matrix `S` whose eigenvectors are the **discrete Hermite-Gauss functions**. Definition: `F^a = Σ_k λ_k^a · v_k v_k^*` where `λ_k = e^{−ikπ/2}`, `v_k` are the DFT eigenvectors. Index-additive (`F^{a₁} F^{a₂} = F^{a₁+a₂}`), unitary, reduces to DFT at `a=1`. **~250 LOC** including Hermite-Gauss eigenvector construction (tridiagonal matrix `S = F + F'` shares eigenvectors with DFT — this is the key Candan trick that makes DFrFT computable without diagonalising the DFT itself). Pei-Yeh 1997 is an earlier alternative; CKO is preferred for unitarity + index-additivity. Cost: `O(N²)` per call after eigenvector caching; `O(N log N)` only via Ozaktas-Arikan-Kutay-Bozdağı 1996 chirp-decomposition (T3b, ~+150 LOC).

**T4 — Non-uniform FFT (NUFFT).** Dutt-Rokhlin 1993; Greengard-Lee 2004 *Accelerating the Nonuniform FFT*, SIAM Review 46(3):443; Lee-Greengard 2005 *The type-3 NUFFT*; Barnett-Magland-af Klinteberg 2019 (FINUFFT, "exponential of semicircle" kernel `φ(x) = e^{β√(1−x²)}`). Three flavours:
  - **Type 1 (NU → U):** scattered samples `(x_j, c_j)` → uniform Fourier coefficients (MRI image reconstruction from k-space spokes).
  - **Type 2 (U → NU):** uniform Fourier coefficients → values at scattered points (forward MRI / radio interferometry).
  - **Type 3 (NU → NU):** scattered in both domains.
  Two-step recipe: (i) **spread** non-uniform samples onto a fine uniform oversampled grid via a separable kernel; (ii) call standard FFT; (iii) **deconvolve** by the kernel transform. **~400 LOC** for Type-1+2 in 1D with Kaiser-Bessel kernel (FINUFFT default); +150 for 2D, +150 for 3D, +200 for Type-3. Precision ε is tunable via kernel width `w` (`w ≈ ⌈log₁₀(1/ε) + 1⌉`).

**T5 — Fractional differentiation via FrFT.** Frontier — Almeida 1994 §VI: order-`α` derivative ≡ multiply by `(2πiξ)^α` in Fourier domain ≡ chirp-multiply in fractional Fourier domain. Useful for fractional Brownian motion, anomalous diffusion, fractional Laplacians. **~200 LOC** delta over T3, but speculative — defer until a consumer requests it.

### Three-way mutual cross-validation pin opportunities (R-MUTUAL-CROSS-VALIDATION)

These are the **regression-style pins** that saturate the 3/3 R-pattern by checking each new primitive against the existing radix-2 `FFT` plus a direct-DFT reference.

1. **Bluestein ≡ radix-2 FFT (when N is a power of 2).** Pin: `BluesteinFFT(x, x_im) == FFT(x, x_im)` to `1e-11` for `N ∈ {2, 4, 8, …, 1024}`. Catches chirp-prefactor sign errors, off-by-one in zero-padding length, and accidental `IFFT` scaling. **(1/3)**
2. **Bluestein ≡ direct O(N²) DFT (for non-pow-2 N).** Pin: `BluesteinFFT(x) == DirectDFT(x)` to `1e-9` for `N ∈ {3, 5, 7, 11, 13, 17, 100, 257, 1009}` (primes and composites). **(2/3)**
3. **Bluestein on N = 2N₀ ≡ split radix-2 FFT of zero-padded input concatenated with shifted spectrum.** Cross-checks the convolution length. **(3/3)** — saturates pin.

For CZT (T1):
- CZT with `A=1`, `W=e^{−2πi/N}`, `M=N` ≡ standard FFT (regression). **(1/3)**
- CZT with `A=1`, `W=e^{−2πi/M}`, `M=N` (zero-padded interpolation) ≡ FFT of zero-padded input. **(2/3)**
- CZT with arbitrary `A`, `W` ≡ direct evaluation `Σ x[n] A^{−n} W^{nk}` to `1e-10`. **(3/3)** — saturates.

For DFrFT (T3):
- `F^0 x = x` (identity). **(1/3)**
- `F^1 x = FFT(x)` (with the standard centred normalisation `(1/√N)·DFT`). **(2/3)**
- `F^2 x = x_reversed` (parity), `F^4 x = x` (period-4). Plus `F^a F^b = F^{a+b mod 4}` index-additivity check. **(3/3)** — saturates.

For NUFFT (T4):
- Type-2 NUFFT on a uniform target grid `x_j = j/N` ≡ standard inverse FFT of the input coefficients. **(1/3)**
- Type-1 NUFFT on a uniform source grid ≡ standard FFT. **(2/3)**
- Type-1 NUFFT on `M` random sources, then Type-2 inverse on the same sources, recovers input strengths to `O(ε)` (forward-adjoint round-trip). **(3/3)** — saturates.

### Numerical precision

- **Bluestein roundoff** is dominated by twiddle-recurrence error in the inner FFT plus the chirp `w^{−n²/2}` evaluation — for `N=10⁵`, `n²` reaches `10¹⁰`, the modulo-2 reduction `(n² mod 2N)` is **mandatory** to avoid catastrophic cancellation. Tolerance: `1e-10` rel.err for `N ≤ 10⁴`, drifting to `1e-8` for `N = 10⁶`.
- **DFrFT** index-additivity `F^a F^b ≡ F^{a+b}` is the strictest precision test — Candan-Kutay-Ozaktas report `1e-12` deviation from index-additivity at `N = 256` because their definition is exactly unitary by construction.
- **NUFFT** has tunable precision: requested ε ∈ {1e-2 … 1e-15}; oversampling `σ = 2.0` and kernel half-width `w = ⌈log₁₀(1/ε) + 1⌉` give the requested precision. Below ε = 1e-13 you must use long-double or quad arithmetic.
- **Per-function tolerance** (per CLAUDE.md): 0 for `F^0 = I`, 1e-11 for FrFT-vs-FFT at `a=1`, 1e-9 for Bluestein on prime N, 1e-8 for NUFFT round-trips at requested ε=1e-12.

### Cross-link consumers

| Application | Primitive | Notes |
|---|---|---|
| MRI / k-space recon | T4 NUFFT-Type-1 | Direct consumer; spiral / radial / non-Cartesian sampling |
| Radar range-Doppler bin | T1 CZT or T2 Zoom-FFT | Focused range bin without full-band FFT |
| Audio pitch tracking | T2 Zoom-FFT | Narrowband around fundamental ±octave |
| Astronomical period search | T2 Zoom-FFT | Lomb-Scargle alt — narrow Δf scan |
| Fractional spectrogram | T3 DFrFT | Joint time-freq rotation, chirp signal analysis |
| Optical signal processing | T3 DFrFT | Lens-as-fractional-FT model |
| Compressed sensing recon | T4 NUFFT | Adjoint operator for non-Cartesian sensing |
| Ultrasound / sonar imaging | T4 NUFFT-Type-3 | Non-uniform Tx and Rx arrays |
| Power systems harmonic estimation | T2 Zoom-FFT | 50/60 Hz ± few Hz resolved to mHz |
| Audio @ 44.1 / 48 kHz native FFT | T0 Bluestein | Today every caller zero-pads — Bluestein removes this tax |

### Cross-cutting comparisons to existing work

- **scipy.signal.czt** (added in scipy 1.8) wraps Bluestein. Reference for API shape.
- **FFTW** uses Bluestein internally for any prime/awkward N (Frigo-Johnson 2005); pocketfft, KFR same. Bluestein is **table-stakes**, not a stretch goal.
- **FINUFFT** (Barnett-Magland-af Klinteberg 2019) is the SOTA NUFFT — its "exponential of semicircle" kernel is ~25% faster than Kaiser-Bessel at equal precision, but has no closed-form Fourier transform (needs numerical quadrature for deconvolution). For reality, Kaiser-Bessel is simpler and within 25% of FINUFFT — recommend Kaiser-Bessel for T4.
- **293-new-ntt** plans `BluesteinNTT` for arbitrary-length NTT in `Z/qZ`. **Algorithm structure is identical** between float64 Bluestein and uint32 Bluestein — different ring, same chirp identity. Recommend slot 132/337 ship `BluesteinFFT` first; slot 293 ports the structure to NTT.

## Concrete recommendations

1. **T0 day-1 PR: `signal/bluestein.go` ~150 LOC.** Implements `BluesteinFFT(re, im []float64)` accepting any N. Composes existing `FFT`/`IFFT` on size `M = next_pow2(2N-1)`. Use `(n*n) mod (2*N)` reduction to keep chirp arguments bounded. Allocates `4*M` scratch internally; expose `BluesteinFFTInto(re, im, scratch []float64)` (scratch length `4*next_pow2(2N-1)`) for hot paths. Mirror IFFT. Saturate the three-way pin above.
2. **Update `signal.FFT` doc to dispatch.** Add a new exported `signal.DFT(re, im)` that internally routes pow-2 → existing radix-2, otherwise → Bluestein. Keep current `FFT`/`IFFT` strict-pow-2 names for zero-alloc contracts — backward compatible.
3. **T1 next: `signal/czt.go` ~80 LOC delta.** `CZT(x_re, x_im []float64, M int, A_re, A_im, W_re, W_im float64, out_re, out_im []float64)`. Saturate three-way pin.
4. **T2 wrapper: `signal/zoom_fft.go` ~40 LOC.** `ZoomFFT(x_re, x_im []float64, fs, f1, f2 float64, M int, out_re, out_im []float64)` — computes the appropriate `A`, `W` and dispatches to `CZT`. Document that this is the bread-and-butter "look at narrow band with high resolution" tool.
5. **Coordinate with slot 293-new-ntt:** ship `BluesteinFFT` and `BluesteinNTT` with **shared design doc** `signal/bluestein_design.md` describing the chirp identity once; the two implementations differ only in element ring (`float64` vs `uint32 mod q`).
6. **T3 (DFrFT) deferred.** ~250 LOC, but: (a) requires Hermite-Gauss eigenvector construction for the centred-DFT matrix (`S = F + F'` Candan trick, eigenvalues distinct, eigenvectors stable), (b) needs `linalg` symmetric tridiagonal eigensolver (verify `linalg/` has one — slot 134 may have flagged this), (c) `O(N²)` cost makes it less hot-path than T0–T2. Ship after T0–T2 land. Consumer: chirp-signal analysis, optical FT model.
7. **T4 (NUFFT) deferred to dedicated slot.** ~400 LOC for 1D Type-1+2; needs Kaiser-Bessel kernel from `signal/window.go` (verify present), oversampled FFT scratch, deconvolution. **Direct MRI consumer makes this high-value** — recommend a dedicated slot in the missing-primitive list (132 §3.10? — confirm). 2D/3D extensions modular.
8. **T5 (fractional derivative via FrFT) park.** Frontier; revisit after slots 117 (fBm), 132 §3.9 land. Likely <100 LOC delta over T3.
9. **R-MUTUAL-CROSS-VALIDATION pins as test files**, not buried in `_test.go` for primary functions. Layout: `signal/bluestein_test.go` with three `Test_R3_…` named pins per primitive. Document the pin saturation in the test name so review tooling can grep saturation status.

## Cheapest day-1 PR
**T0 Bluestein in `signal/bluestein.go` — ~150 LOC, composes existing radix-2 FFT, zero new dependencies, immediate consumers (Pistachio audio at 44100/48000, Oracle on arbitrary-N time series, every `prob` consumer that wants FFT-of-arbitrary-N for characteristic functions).** Three-way pin trivial: BluesteinFFT(pow2 N) ≡ FFT, BluesteinFFT(prime N) ≡ direct DFT. Saturates 3/3 R-pattern on slot 1.

## Sources

### Repo files
- `C:\limitless\foundation\reality\signal\fft.go` — only existing FFT (lines 49-91), pow-2 panic at line 54.
- `C:\limitless\foundation\reality\signal\fft.go:101-127` — IFFT, also pow-2-only.
- `C:\limitless\foundation\reality\signal\window.go` — present (verify Kaiser-Bessel for NUFFT kernel).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\132-signal-missing.md` §1.3, §2.10, §3.9 — Bluestein, CZT, FrFT all already on the missing list.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\133-signal-sota.md` §4 — Bluestein as universal SOTA fallback.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\134-signal-api.md` — `BluesteinFFT(real, imag []float64)` API shape proposal.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\293-new-ntt.md` N4 — Bluestein in NTT-land, structure-shared with float64 Bluestein.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\301-dive-fft-correctness.md` §6 — Bluestein gap re-flagged.
- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md:357` — slot 337 line.

### Web sources
- Bluestein 1968, *A linear filtering approach to the computation of the discrete Fourier transform*, NEREM Record / IEEE Trans. AU 18:451-455 (1970).
- Rabiner, Schafer, Rader 1969, *The Chirp z-Transform Algorithm*, IEEE Trans. AU 17:86-92.
- Namias 1980, *The fractional order Fourier transform and its application to quantum mechanics*, J. Inst. Math. Appl. 25:241.
- Almeida 1994, *The fractional Fourier transform and time-frequency representations*, IEEE Trans. SP 42(11):3084.
- Pei, Yeh 1997, *Improved discrete fractional Fourier transform*, Opt. Lett. 22(14):1047.
- Candan, Kutay, Ozaktas 2000, *The discrete fractional Fourier transform*, IEEE Trans. SP 48(5):1329-1337. ([Bilkent PDF](https://www.ee.bilkent.edu.tr/~haldun/publications/ozaktas166.pdf))
- Ozaktas, Arikan, Kutay, Bozdağı 1996, *Digital computation of the fractional Fourier transform*, IEEE Trans. SP 44(9):2141 — O(N log N) FrFT.
- Dutt, Rokhlin 1993, *Fast Fourier transforms for nonequispaced data*, SIAM J. Sci. Comput. 14(6):1368.
- Greengard, Lee 2004, *Accelerating the Nonuniform Fast Fourier Transform*, SIAM Review 46(3):443.
- Lee, Greengard 2005, *The type 3 nonuniform FFT and its applications*, J. Comput. Phys. 206:1.
- Barnett, Magland, af Klinteberg 2019, *A parallel non-uniform FFT library based on an exponential-of-semicircle kernel*, SIAM J. Sci. Comput. 41(5):C479. ([arXiv 1808.06736](https://arxiv.org/abs/1808.06736))
- Frigo, Johnson 2005, *The Design and Implementation of FFTW3*, Proc. IEEE 93(2):216 — Bluestein in FFTW.
- [Wikipedia — Chirp Z-transform](https://en.wikipedia.org/wiki/Chirp_Z-transform).
- [scipy.signal.czt docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.czt.html).
- [FINUFFT docs](https://finufft.readthedocs.io/en/latest/).
- [rocFFT Bluestein design doc](https://rocm.docs.amd.com/projects/rocFFT/en/docs-5.7.1/design/bluestein.html).

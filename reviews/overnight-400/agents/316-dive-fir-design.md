# 316 — dive-fir-design (Parks-McClellan / Kaiser / Window-Method / Frequency-Sampling / Least-Squares audit)

## Headline
Reality has **zero FIR filter-design machinery** — only the raw `signal.Convolve` execution kernel (`signal/filter.go:19`) and three windows (Hann/Hamming/Blackman in `signal/window.go`). No `Firwin`, no Kaiser window, no Kaiser-Hellier order estimate, no Parks-McClellan/Remez, no frequency-sampling design, no least-squares design, no Hilbert-transform FIR. Day-1 PR: ship `signal/fir.go` with **windowed-sinc lowpass + Kaiser window + Kaiser-Hellier order formula** (~250 LOC, ~30 golden vectors); defer Parks-McClellan (the keystone, ~400 LOC) and frequency-sampling/least-squares to a Day-2 PR. With slot 315's IIR PR, this gives consumers complete classical filter-design coverage.

## Findings

### What exists (full inventory)
- `signal/filter.go:19` — `Convolve(signal,kernel,out)` direct O(N·M). The FIR **execution** kernel; users today must supply their own taps `kernel[]`. No design helper.
- `signal/filter.go:54` — `MovingAverage` — boxcar FIR, hardcoded rectangular window, no parametric f_c. Equivalent to a length-`windowSize` FIR with all-ones taps `1/N`.
- `signal/filter.go:97` — `ExponentialMovingAverage` — 1-pole IIR (slot 315 covers).
- `signal/filter.go:130` — `MedianFilter` — non-linear, not FIR.
- `signal/window.go:15` — `HannWindow(n,out)`.
- `signal/window.go:44` — `HammingWindow(n,out)`.
- `signal/window.go:76` — `BlackmanWindow(n,out)`.
- `signal/window.go:104` — `ApplyWindow(signal,window,out)` element-wise multiply.
- `signal/fft.go:49,101` — `FFT/IFFT` (radix-2). Required substrate for **frequency-sampling design** (IFFT of desired magnitude → impulse response).

### What is missing
Grep across `signal/`, `audio/`, `acoustics/`, `control/` for `FIR|Parks|Remez|Kaiser|FrequencySampling|LinearPhase|McClellan|Firwin|firdes`: **0 hits in production code** (only matches are review docs, doc comments, and an unrelated `FIRST` substring in `prob/copula/doc.go:34` and `sequence/phonetic_test.go:86`).

Specifically absent:

1. **Kaiser window** — Kaiser 1974, the parameterised window. `w[n] = I₀(β·sqrt(1-(2n/(N-1)-1)²)) / I₀(β)`. Single tunable `β` continuously trades main-lobe width vs sidelobe height (β=0 → rectangular, β=5.658 → ≈Hamming, β≈8.96 → ≈Blackman). The de-facto FIR window when stopband attenuation is a hard spec. Requires `I₀` (modified Bessel first-kind, order 0) — currently absent in reality (slot 300 flagged Bessel-family gap).
2. **Kaiser-Hellier order/β estimation** — given passband ripple `δ_p`, stopband attenuation `δ_s` (dB) and transition width `Δω`, the closed-form estimate (Kaiser-Hellier, Oppenheim-Schafer §7.5.3 eq 7.75-7.76):
   ```
   A = -20·log10(min(δ_p, δ_s))
   β = 0.1102·(A-8.7)               if A > 50
       0.5842·(A-21)^0.4 + 0.07886·(A-21)  if 21 ≤ A ≤ 50
       0                            if A < 21
   N = ⌈(A - 8) / (2.285·Δω)⌉
   ```
   This is the **single most-used filter-spec helper** outside MATLAB's `firpmord`. ~30 LOC.
3. **Windowed-sinc lowpass** — `h[n] = 2·f_c · sinc(2·f_c·(n - (N-1)/2)) · w[n]`. The Tier-0 FIR; gives linear phase by construction (symmetric taps). Length-N tap-set indexable by `n=0..N-1`. ~40 LOC. With existing windows from `signal/window.go` it's a 1-line helper plus the sinc evaluation.
4. **High-pass / band-pass / band-stop spectral-inversion / spectral-reversal** transforms — given LP taps, derive HP/BP/BS via `h_hp[n] = δ[n] - h_lp[n]` (spectral inversion) or `h_bp = h_lp1 - h_lp2`. ~30 LOC, four 4-line constructors.
5. **Parks-McClellan / Remez exchange** — Parks-McClellan 1972, McClellan-Parks-Rabiner 1973. **The optimal equiripple FIR** by alternation theorem (Chebyshev minimax). Iterative: pick L+2 extremals, solve system for ripple δ on extremals, find new extremals via dense-grid search, repeat until convergence. ~400 LOC including the Remez inner loop, the alternation-set update, dense-grid evaluation, polynomial interpolation. **The keystone**: SciPy `firls`/`remez`, MATLAB `firpm`. Cited 100k+ times.
6. **Frequency-sampling design** — specify desired magnitude at `N` equispaced points `H_d[k]`, IFFT to get `h[n]`. Simple, but lacks ripple control unless transition samples are optimised. Trivially leverages existing `signal.IFFT`. ~80 LOC for the basic version.
7. **Least-squares FIR (`firls`)** — minimises `∫|H_d(ω) - H(ω)|² dω` over piecewise-defined band weights. Closed-form via Toeplitz/Hermitian system on the autocorrelation of the desired spectrum. ~150 LOC. Produces filters with monotonically decaying stopband (no equiripple), often preferred over PM when smoothness matters.
8. **Hilbert-transform FIR** — odd-symmetric, length-N taps `h[n] = 2·sin²(π(n-(N-1)/2)/2) / (π(n-(N-1)/2))` for n ≠ centre, 0 at centre. Approximates ideal `H(ω) = -j·sgn(ω)`. ~30 LOC. Used by audio (analytic signal, envelope), single-sideband modulation, instantaneous-frequency computation.
9. **Differentiator FIR** — odd-symmetric Type-III/IV, approximates `H(ω) = jω`. ~30 LOC. Cousin of Hilbert-FIR.
10. **Multi-band design** — Parks-McClellan with multiple passbands/stopbands and per-band weights. Composes T5 once shipped.
11. **Group-delay verification** — for linear-phase FIR, group delay is exactly `(N-1)/2` samples regardless of frequency. Provide `assertLinearPhase(taps)` test helper that pins symmetry.
12. **Polyphase decomposition** — for upsampling/downsampling FIRs (Vaidyanathan 1993). Out of scope here; flagged for future slot.

### Cross-package fanout (what this gap blocks)
- **aicore audio front-end** — speech anti-alias, decimation, frame-rate conversion. Linear-phase FIR is mandatory for ASR (no phase distortion before MFCC).
- **Pistachio realtime audio** — crossover networks (LP/HP pair) **must** be linear-phase FIR; IIR adds frequency-dependent group delay that smears transients across drivers.
- **audio/onset/superflux.go** — uses spectral flux; smoothing kernel today is a `MovingAverage` (rectangular FIR). A short Kaiser-windowed-sinc lowpass would reduce sidelobe leakage in the flux signal.
- **audio/spectrogram/stft.go** — STFT analysis windows already use `signal.window` Hann/Hamming/Blackman. Kaiser would let users dial sidelobe leakage parametrically.
- **audio/pitch/yin.go**, `mpm.go` — autocorrelation pitch detection; Hilbert FIR would give analytic-signal envelope for AM-pitch tracking.
- **acoustics** — speaker/room-impulse-response inversion is a linear-phase FIR job (slot 003 sota).
- **prob/sequence** — outlier-removed trend extraction; LS-FIR would give optimal smoothing for a stated cutoff.
- **control** — feedforward predictive compensators are FIR; today only IIR PID exists.
- **em**, **fluids** — sensor-array delay-and-sum beamforming uses fractional-delay FIR (sinc-interpolation FIR), trivial windowed-sinc derivative.

### Numerical stability / correctness traps to pin
1. **Sinc at n = (N-1)/2** — avoid `0/0`. Standard: when `|x| < ε`, return `1.0`. With `N` even, the centre falls between samples and no exception arises; with `N` odd (the typical Type-I FIR), the central tap evaluates `sinc(0) = 1` exactly.
2. **Kaiser β = 0** trap — `I₀(0) = 1` is fine, but a numerically naive `I₀` series diverges for large `β`. Use the small-arg series for `β² < 50` and the large-arg asymptotic `I₀(x) ≈ e^x / sqrt(2πx)` for `β² ≥ 50`. Pin against tabulated values (Abramowitz & Stegun §9.8, NIST DLMF §10.40).
3. **Type-II FIR (even-length) zero at Nyquist** — symmetric FIR with even `N` is constrained to `H(π) = 0`. Cannot realise a HP or HS as Type-II; must use Type-I (odd-length) for HP. **Reject** in `FIRWindowHighpass(N, ...)` if `N` is even.
4. **Type-III/IV** (antisymmetric) for Hilbert/differentiator — `H(0) = 0` and `H(π) = 0` (Type-III) or only `H(0) = 0` (Type-IV) constraints. Document and validate the symmetry requirement.
5. **Parks-McClellan convergence** — the Remez exchange is not guaranteed to converge for ill-conditioned specs (transition width too narrow vs filter length). Document max-iter cap (50 typical) and surface non-convergence as an error rather than silently returning a sub-optimal design.
6. **Parks-McClellan dense-grid density** — too few grid points (default 16·N) can miss extremal frequencies, leading to false convergence. Pin the grid-density at least 16·N as Rabiner et al recommend.
7. **Frequency-sampling Gibbs** — naive `IFFT(H_d[k])` produces ~21% Gibbs overshoot at band edges; mitigate via transition-band sample optimisation (Rabiner 1971) or post-window. Document as caveat.
8. **DC gain normalisation** — windowed-sinc gain at ω=0 is `Σ h[n]` ≠ 1.0 due to window shaping at edges. Provide explicit `Normalize(taps, ω₀)` post-step that divides by `|H(ω₀)|`.
9. **Coefficient symmetry pinning** — a linear-phase FIR has `h[n] = ±h[N-1-n]`. Asserting this in tests detects implementation bugs (off-by-one in centre indexing, wrong sign for Type-III/IV).
10. **f_c relative to Nyquist** — `f_c < 0.5` (normalised) required; reject otherwise. With finite transition width `Δf`, also require `f_c + Δf/2 < 0.5`.

### R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities
A 3-of-3 cross-validation pattern for any new FIR design (per `reviews/SYNTHESIS.md` template):

1. **Pin A — Kaiser-Hellier formula identity.** Design a Kaiser-windowed-sinc LP with target stopband `A_s = 60 dB`, transition width `Δω = 0.1π`. Compute the actual stopband attenuation by max(`|H(ω)|`) over `ω ∈ [ω_c + Δω/2, π]`. Assert `≥ 60 dB` (equality not required — formula is conservative). Tolerance: 0.5 dB slack.
2. **Pin B — alternation theorem (Parks-McClellan).** For an order-N PM filter with L+2 = (N+1)/2 + 2 extremals (Type-I), the actual error `E(ω) = W(ω)·(D(ω) - H(ω))` must alternate sign at the extremal frequencies, with `|E|` exactly equal to the converged ripple δ at all of them. Pin: dense-grid evaluate `E(ω)`, find local extrema, assert at least L+2 alternations and `|E_extremal| - δ| < 1e-10`. **This pin is by-construction in PM, but pinning it catches bugs where Remez exited early or extremal-set update was wrong.**
3. **Pin C — frequency-sampling identity.** For frequency-sampling design with `H_d[k]` specified at `k = 0..N-1`, the resulting `h[n] = IFFT(H_d)` evaluated at ω = 2πk/N must reproduce `H_d[k]` exactly (to roundoff, ≤ 1e-12). Three independent witnesses: (a) Kaiser-Hellier formula; (b) PM alternation theorem; (c) frequency-sampling DFT identity — three different design philosophies cross-checking the same library.

A fourth bonus pin (linear-phase symmetry): for any windowed-sinc/PM/LS LP designed via the new code, assert `h[n] == h[N-1-n]` to roundoff. Trivial but catches off-by-one bugs that cause non-linear phase silently.

## Concrete recommendations

### Day-1 PR (cheapest single landing, ~250 LOC, ~30 golden vectors)
Create `signal/fir.go` exposing:

1. **Kaiser window** (~50 LOC, depends on `I₀`).
   ```go
   func KaiserWindow(n int, beta float64, out []float64) {
       // w[i] = I0(beta * sqrt(1 - (2i/(n-1) - 1)^2)) / I0(beta)
       i0Beta := besselI0(beta)
       for i := 0; i < n; i++ {
           x := 2.0*float64(i)/float64(n-1) - 1.0
           out[i] = besselI0(beta*math.Sqrt(1-x*x)) / i0Beta
       }
   }
   ```
   Inline `besselI0(x)` via small-arg series (Abramowitz §9.8.1) for `x < 3.75` and asymptotic series (§9.8.2) for `x ≥ 3.75`. ~25 LOC. **OR** declare a dependency on a future `special/bessel.go` (slot 300) and stub here.
2. **Kaiser-Hellier order/β estimate** (~30 LOC).
   ```go
   func KaiserOrder(deltaPass, deltaStop, transitionWidth float64) (order int, beta float64) {
       A := -20 * math.Log10(math.Min(deltaPass, deltaStop))
       switch {
       case A > 50:
           beta = 0.1102 * (A - 8.7)
       case A >= 21:
           beta = 0.5842*math.Pow(A-21, 0.4) + 0.07886*(A-21)
       default:
           beta = 0
       }
       order = int(math.Ceil((A - 8) / (2.285 * transitionWidth)))
       return
   }
   ```
3. **Windowed-sinc lowpass** (~40 LOC).
   ```go
   func FIRLowpass(n int, fc float64, win []float64, out []float64) {
       // h[i] = 2*fc * sinc(2*fc*(i - (n-1)/2)) * w[i]
       half := float64(n-1) / 2
       for i := 0; i < n; i++ {
           x := 2.0 * fc * (float64(i) - half)
           if math.Abs(x) < 1e-15 {
               out[i] = 2.0 * fc * win[i]
           } else {
               out[i] = math.Sin(math.Pi*x) / (math.Pi * x) * 2.0 * fc * win[i]
           }
       }
       // optional: normalize so sum(out) == 1 exactly
   }
   ```
4. **Spectral-inversion HP / spectral-reversal BP/BS helpers** (~30 LOC).
   `FIRHighpass(n, fc, win, out)` — design LP, then `out[i] = δ[i_centre] - out[i]`.
   `FIRBandpass(n, fc1, fc2, win, out)` — two LPs, subtract.
   `FIRBandstop` — `δ[i_centre] - bandpass`.
5. **Hilbert FIR** (~30 LOC).
   ```go
   func FIRHilbert(n int, win []float64, out []float64) {
       // Type-III antisymmetric, h[n-1-i] = -h[i], h[(n-1)/2] = 0
       half := float64(n-1) / 2
       for i := 0; i < n; i++ {
           k := float64(i) - half
           if math.Abs(k) < 1e-15 || math.Mod(k, 2) == 0 {
               out[i] = 0
           } else {
               out[i] = (2.0 / (math.Pi * k)) * win[i]
           }
       }
   }
   ```
6. **Linear-phase symmetry assert** (~10 LOC).
   `IsLinearPhase(taps []float64, antisymmetric bool, tol float64) bool` — pins by-construction symmetry. Used in tests and exported for downstream verification.
7. **DC-gain normalise helper** (~10 LOC).
   `NormalizeAtDC(taps []float64)` — divides by `Σ taps`.

### Golden-file vectors (30 minimum)
- 5× Kaiser window at β = {0, 2, 4, 6, 8.96}, n = 65 → tap arrays.
- 5× Kaiser-Hellier order/β estimate against published tables (Oppenheim-Schafer §7.5 ex 7.4-7.5).
- 5× Windowed-sinc LP at f_c = {0.1, 0.2, 0.25, 0.3, 0.4} (normalised to fs/2), n = 65, Hamming → tap arrays.
- 5× Stopband attenuation pin (Pin A): assert `max|H(ω)| ≤ predicted A_s` over stopband.
- 5× Linear-phase symmetry pin: every Type-I LP from the test suite asserts `h[i] == h[n-1-i]` to 1e-15.
- 5× IEEE 754 edges: n = 1 (degenerate), β = 0 (rectangular), β = 50 (asymptotic regime), f_c = 0 (DC bypass), f_c at boundary 0.499.

### Day-2 PR (~600 LOC, the keystone)
- **Parks-McClellan / Remez** (~400 LOC) — the McClellan-Parks-Rabiner 1973 implementation, structured:
  - Outer Remez exchange loop with max-iter cap (50).
  - Dense-grid construction (16·N points typical).
  - Lagrange-interpolation polynomial of `H(ω)` from extremal samples (`ω_k`, `D_k`, `W_k`).
  - Extremal-set update via dense-grid local-extremum search.
  - Convergence test: `δ_new / δ_old < 1 + 1e-10`.
  - Multi-band support via piecewise `D(ω)`, `W(ω)` arrays.
  - Pin B (alternation theorem) as test infrastructure.
- **Frequency-sampling design** (~120 LOC) — `FreqSamplingFIR(magnitudes []float64) []float64`. Compose with existing `signal.IFFT`.
- **Least-squares FIR** (~150 LOC) — `LSFIR(passbands, stopbands, weights []Band) []float64` via Toeplitz solve (depends on `linalg.SolveSymmetric` — present per slot 098).
- **Differentiator FIR** (~30 LOC) — Type-III/IV antisymmetric.
- **Multiband helper** (~50 LOC) — composes PM with arbitrary `[(f_lo, f_hi, gain, weight), ...]`.

### Day-3 (deferred)
- Polyphase FIR for resampling (Vaidyanathan 1993) — pairs with future audio resampler.
- Equiripple-FIR with arbitrary group-delay constraint (fractional-delay FIR) — sinc-interpolation case.
- Constrained least-squares (CLS-FIR, Selesnick-Burrus 1996) — bounds peak ripple while minimising LS error.
- Generalised Butterworth FIR (Selesnick-Burrus 1998) — maximally-flat FIR design.

### Documentation hygiene to fix today (independent of new code)
- `signal/filter.go:19` `Convolve` — add doc sentence: *"For FIR filtering, design taps with `signal.FIRLowpass` (windowed-sinc) or `signal.FIRRemez` (Parks-McClellan, optimal equiripple) — see `signal/fir.go`."* (after the new file lands).
- `signal/window.go` doc header — note that windows are *also* used as FIR-design windows: cite `signal.FIRLowpass`.

## Sources

### Repo files cited
- `C:\limitless\foundation\reality\signal\filter.go` (lines 19, 54, 97, 130)
- `C:\limitless\foundation\reality\signal\window.go` (lines 15, 44, 76, 104) — Hann/Hamming/Blackman exist; Kaiser/Tukey/Bessel/Nuttall absent.
- `C:\limitless\foundation\reality\signal\fft.go` (lines 49, 101) — radix-2 FFT/IFFT exist, ready substrate for frequency-sampling design.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\132-signal-missing.md:77-83` — already enumerated this gap at higher level (window method + Parks-McClellan).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\315-dive-iir-design.md` — companion IIR review; together cover classical filter design.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\008-audio-sota.md`, `007-audio-missing.md` — audio EQ + crossover gaps depend on FIR.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\166-synergy-acoustics-signal.md`, `167-synergy-audio-signal.md` — synergy slots flagging the same need.
- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md` slot 316 line.

### Canonical references
- **Parks, T.W. & McClellan, J.H.** (1972) *Chebyshev approximation for nonrecursive digital filters with linear phase*, IEEE Trans CT-19(2):189-194 — the foundational PM paper.
- **McClellan, J.H., Parks, T.W. & Rabiner, L.R.** (1973) *A computer program for designing optimum FIR linear phase digital filters*, IEEE Trans Audio Electroacoust AU-21(6):506-526 — the classic reference implementation; Fortran source PMREMEZ widely reproduced.
- **Kaiser, J.F.** (1974) *Nonrecursive digital filter design using the I₀-sinh window function*, in Proc. IEEE Symp. Circuits and Systems — Kaiser window definition + design formulae (Kaiser-Hellier).
- **Rabiner, L.R.** (1971) *Techniques for designing finite-duration impulse-response digital filters*, IEEE Trans COM-19:188-195 — frequency-sampling with optimised transition coefficients.
- **Oppenheim & Schafer** *Discrete-Time Signal Processing*, 3rd ed (2010) — §7.4 windowed-sinc; §7.5 Kaiser; §7.6 Parks-McClellan; §7.7 frequency-sampling; §7.8 LS-FIR.
- **Burrus, C.S. & Parks, T.W.** *DFT/FFT and Convolution Algorithms* (1985) and **Parks & Burrus** *Digital Filter Design* (1987) — comprehensive FIR + IIR design treatment.
- **Selesnick, I.W. & Burrus, C.S.** (1998) *Generalized digital Butterworth filter design*, IEEE Trans SP 46(6):1688-1694 — maximally-flat FIR.
- **Selesnick, Lang & Burrus** (1996) *Constrained least squares design of FIR filters without specified transition bands*, IEEE Trans SP 44(8):1879-1892 — CLS-FIR.
- **Lyons, R.G.** *Understanding Digital Signal Processing*, 3rd ed (2010) — pedagogical FIR design ch. 5; spectral-inversion / spectral-reversal idioms.
- **Smith, J.O.** *Spectral Audio Signal Processing* and *Introduction to Digital Filters*, CCRMA (`https://ccrma.stanford.edu/~jos/sasp/`, `https://ccrma.stanford.edu/~jos/filters/`) — windowed-sinc derivation, FIR-vs-IIR tradeoffs.
- **Remez, E.Ya.** (1934) *Sur la détermination des polynômes d'approximation de degré donnée*, Comm. Soc. Math. Kharkov 10:41-63 — original Remez exchange algorithm; PM is its application to digital filter design.
- **Abramowitz & Stegun** §9.8 / **NIST DLMF** §10.32, §10.40 — `I₀` series + asymptotic expansions for Kaiser-window implementation.
- **Vaidyanathan, P.P.** *Multirate Systems and Filter Banks* (1993) — polyphase FIR; out of scope here, flagged for future work.

### Cross-link to other slots
- Slot 132 — already enumerated this gap at higher level (signal-missing §1.6).
- Slot 133, 134 — signal-sota / signal-api — flagged Parks-McClellan as critical missing surface.
- Slot 167, 166 — synergy audio↔signal, acoustics↔signal — FIR is the bridge.
- Slot 315 — IIR-design (companion). Day-1 IIR + Day-1 FIR PRs together close the classical filter-design surface in ~730 LOC.
- Slot 008, 007 — audio-sota/missing — parametric EQ + crossover need linear-phase FIR.
- Slot 003, 002 — acoustics-sota/missing — room-impulse-response inversion is FIR.
- Slot 300 — Bessel-spherical — `I₀` modified Bessel needed for Kaiser; should be the same primitive shipped in slot 300's PR. Coordinate.
- Slot 270 — graph signal proc — FIR-on-graph filters (Chebyshev polynomial of graph Laplacian) reuse the same Remez/Chebyshev minimax machinery.
- Slot 098 — linalg-sota — `SolveSymmetric` / Toeplitz solver presence determines LS-FIR Day-2 feasibility.

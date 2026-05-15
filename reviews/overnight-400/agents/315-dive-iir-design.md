# 315 — dive-iir-design (Bilinear / Impulse-Invariant / MZT / Prewarping audit)

## Headline
Reality has **zero IIR filter-design machinery** — no bilinear transform, no impulse-invariant or matched-Z mapping, no Butterworth/Chebyshev/Elliptic/Bessel prototype, no biquad section; consumers are stuck with first-order EMA (`signal/filter.go:97`, `control/filter.go:15`) and a continuous-time `TransferFunction` whose poles can be found (`control/transfer.go:73`) but cannot be discretised. Day-1 PR: ship `signal/iir.go` with **prewarp helper + bilinear transform + Butterworth pole prototype + DF-II-T biquad cascade + RBJ cookbook** (~480 LOC, ~30 golden vectors), unblocking aicore/Pistachio audio EQ, room equalisation in `acoustics`, control-loop discretisation, and biosignal anti-alias filtering. Bessel and Elliptic deferred to a follow-up.

## Findings

### What exists (full inventory)
- `signal/filter.go:19` — `Convolve(signal,kernel,out)` direct O(NM) — **FIR primitive**, not IIR.
- `signal/filter.go:54` — `MovingAverage(signal,windowSize,out)` — boxcar FIR (linear-phase, but not parametric).
- `signal/filter.go:97` — `ExponentialMovingAverage(signal,alpha,out)` — `out[i] = α·x[i] + (1-α)·out[i-1]`. **This is a 1-pole IIR low-pass** with pole at `z = 1-α`, but the API hides this: no cutoff-frequency entry point, no sample-rate awareness, no documentation of `α ↔ f_c`.
- `signal/filter.go:130` — `MedianFilter(signal,windowSize,out)` — non-linear (rank-order). Irrelevant here.
- `control/filter.go:15` — `LowPassFilter(prev,current,alpha)` — same single-pole IIR as EMA, scalar-step API for stateful loops.
- `control/filter.go:38` — `HighPassFilter(prevFiltered,prev,current,alpha)` — first-difference complementary HP, again 1-pole.
- `control/filter.go:74` — `ComplementaryFilter` — sensor fusion; not a frequency-domain filter.
- `control/transfer.go:21` — `TransferFunction{Numerator,Denominator}` — **continuous-time only** (Laplace `s`). `Evaluate` (line 36), `Poles` (73), `IsStable` (239) — no `s→z` mapping, no companion digital `TransferFunctionZ` type.
- `control/pid.go:36` — `PIDController` — special-case IIR (P+I+D = first-order with integrator), but written in time-domain difference form, not exposed as a biquad.

### What is missing (search returned zero matches across `signal/`, `control/`, `acoustics/`, `audio/`)
Grep for `Butterworth|Chebyshev|Elliptic|Bessel|Bilinear|Tustin|ImpulseInvariant|MatchedZ|Biquad|prewarp` in those four packages: **0 hits**. (Only hits anywhere in repo are in `reviews/overnight-400/agents/132-signal-missing.md` which already enumerated this gap, and the `reviews/MASTER_PLAN.md` slot-315 line itself.)

Specifically absent:
1. **Bilinear (Tustin) transform** `s = (2/T)·(z-1)/(z+1)` — no helper to apply it to a `TransferFunction`. This is the single most-used `s→z` mapping in DSP.
2. **Frequency prewarping** `ω_a = (2/T)·tan(ω_d·T/2)` — without it, every bilinear-designed filter cuts off at the wrong frequency. Slot 312's "documented intent ≠ implementation" failure mode in `audio/spectrogram/visualise.go` is exactly the same risk class — design files often forget the prewarp and silently mistune.
3. **Impulse-invariant transform** — digital h[n] = T·h_a(nT). Aliases unless analog prototype is bandlimited (Butterworth roll-off ≥ 6N dB/oct usually OK; Chebyshev-II / Elliptic with ripply stopband — bad). Standard for control where preserving impulse shape matters.
4. **Matched-Z transform** — pole/zero-by-pole/zero `z = e^{sT}`. Cheap, but doesn't preserve magnitude carefully. Niche.
5. **Step-invariant transform** — analog of impulse-invariant, preserves step response. Needed for ZOH-equivalent control discretisation.
6. **Analog prototypes:**
   - Butterworth: poles on a circle of radius `ω_c` at angles `π/2 + (2k-1)π/(2N)`. Simplest, maximally flat. ~15 LOC.
   - Chebyshev-I: poles on an ellipse parameterised by passband ripple `ε`. ~40 LOC.
   - Chebyshev-II: inverse Chebyshev — zeros + poles. ~50 LOC.
   - Elliptic (Cauer): Jacobi elliptic functions — ~200 LOC, requires `cn/sn/dn` (currently absent — slot 299 flagged this).
   - Bessel: Bessel polynomial roots — maximally flat group delay; pure-real-pole construction inadvisable, use frequency-normalised reverse-Bessel polynomial table or Storch recursion.
7. **Biquad section** (canonical second-order IIR): `H(z) = (b0+b1z⁻¹+b2z⁻²)/(1+a1z⁻¹+a2z⁻²)`. **Direct-Form-II Transposed** is the IEEE-754 canonical form (Oppenheim-Schafer §6.5; Higham §13 for noise-gain). DF-I has 2× state, DF-II has 1× state but pole/zero ordering matters; DF-IIt is the audio-DSP default.
8. **Cascade-of-biquads factorer** — split N-th-order rational into ⌈N/2⌉ biquad sections, pair conjugate poles with conjugate zeros, **scale per-section to balance dynamic range** (Mitra §6.4). Without pairing/scaling, high-Q filters lose 30+ dB SNR to per-section overflow.
9. **RBJ Audio EQ Cookbook** (Bristow-Johnson 2005, web-canonical) — closed-form biquad coefficients for LP, HP, BP (constant-skirt and constant-peak), notch, all-pass, peaking-EQ, low-shelf, high-shelf, parameterised by `f0/Q/dBgain/sampleRate`. ~120 LOC. **Every audio EQ on Earth uses this.**
10. **Denormal handling.** `signal/filter.go:97` EMA with α≈1e-4 will trap into subnormal floats after silence on x86, causing 100× slowdown until stimulus returns. Fix: add tiny DC dither (`y += 1e-30; y -= 1e-30;` flush) or use SSE `_MM_FLUSH_ZERO` flag (Go: hard — needs assembly stub). Document the trap in a `// CAVEAT:` comment minimum.
11. **Group-delay computation** (`-d∠H(ω)/dω`) — needed for verifying linear-phase claims. Trivial helper from existing `TransferFunction.Evaluate`.
12. **Filter-coefficient quantisation analysis** — fixed-point biquads have pole-locus sensitivity catastrophic at low cutoffs (Sripad-Snyder 1977 modelling). Out of scope for float64 reality, but consumers shipping fixed-point need a pin.

### Cross-package fanout (what this gap blocks)
- **aicore audio front-end** — speech VAD pre-emphasis, anti-alias decimation, formant tracking; all need parametric IIR.
- **Pistachio realtime audio** — EQ knobs, parametric tonality controls, room compensation. RBJ cookbook is the API users expect.
- **acoustics/** — `Sabine RT60` (already there) but no room-equalisation compensation; cannot apply correction filter without IIR design.
- **control/** PID is a degenerate IIR; richer compensators (lead-lag, notch for resonance suppression) require general biquad.
- **audio/onset/superflux.go** — uses spectral flux which depends on STFT magnitude smoothing; today does FIR moving-average; would benefit from short low-Q IIR for less group-delay.
- **Folio biosignals** (mentioned in 132 review) — ECG 50/60 Hz line-frequency notch is a textbook 2-pole notch biquad (`Q≈30, f0=50`). Can't ship it today.
- **prob/spectral** (slot 270 graph signal proc) — would import IIR for graph-frequency band-pass.

### Numerical stability traps to pin
1. **Bilinear at high f_c** — when `f_c → fs/2`, `tan(ω·T/2) → ∞`, prewarping diverges. Must cap to `f_c < 0.49·fs` and document.
2. **DC normalisation post-bilinear** — `H(z=1) = H(s=0)` exactly only if the analog prototype was zero-gain at DC; `b0+b1+b2 / (1+a1+a2)` should match analog DC gain. Pin as golden test.
3. **Biquad pole at z=1** (DC pole, pure integrator) — `1+a1+a2 = 0` is degenerate; output drifts unboundedly with any DC offset. Reject in design API or document the requirement to high-pass before integrating.
4. **Coefficient ordering convention** — `control/transfer.go:14` uses **descending degree** (`{1, 1}` = `s + 1`). MATLAB/SciPy `b,a` arrays are also descending. **DO NOT** flip to ascending in the new IIR file — consistency with `transfer.go` is mandatory.
5. **State init** — biquad state `(s1,s2)` at startup defaults to 0; if input has non-zero DC, the filter undergoes a transient (can be 100s of samples for high-Q). Provide `InitSteadyState(x0)` helper that solves `s1 = (b1 - a1·b0)·x0 / (1+a1+a2)`, etc.

### R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities
A 3-of-3 cross-validation pattern (per `reviews/SYNTHESIS.md` template) for any new IIR design:
1. **Pin A — analog spec.** Butterworth order-N at cutoff `ω_c`: `|H(jω_c)|² = 1/2` exactly (-3.0103 dB). Test: evaluate the analog `TransferFunction.Evaluate(complex(0,ω_c))`; assert `|H|² ≈ 0.5` to 1e-12.
2. **Pin B — bilinear identity at DC.** `s = 0 ⇔ z = 1`. After bilinear, evaluate digital `H(z=1)` against analog `H(s=0)` — must match to roundoff. Same identity at Nyquist: `s = ∞ ⇔ z = -1`.
3. **Pin C — impulse-response geometric decay.** Drive biquad with `δ[n]` and assert `h[n] / h[n-1]` converges to `|p_max|` (largest pole magnitude) for `n` past transient. Tolerance 1e-9 (accumulating).

These three pin a) the analog prototype, b) the s→z mapping, c) the time-domain realisation — three independent witnesses, by hand cross-checkable. Slot 312's "doc claims X, code does Y" risk is mitigated by Pin B because it's an identity that's wrong if either prewarp or bilinear ordering is flipped.

A fourth opportunity, not a 3/3 but worth pinning: round-trip with `prob/info` — pass white noise through a designed Butterworth, FFT, compare `|H(jω)|²·N0` to integrated `PowerSpectrum`. Couples `signal` ↔ `prob` ↔ `signal/iir`.

## Concrete recommendations

### Day-1 PR (cheapest single landing, ~480 LOC, ~30 golden vectors)
Create `signal/iir.go` exposing:

1. **Prewarp helper** (~5 LOC).
   ```go
   func PrewarpFreq(fDigital, sampleRate float64) float64 {
       // ω_a = (2/T)·tan(ω_d·T/2)
       T := 1.0 / sampleRate
       return (2.0 / T) * math.Tan(math.Pi*fDigital*T)
   }
   ```
2. **Bilinear transform of a `control.TransferFunction`** (~50 LOC).
   `BilinearTransform(tfA control.TransferFunction, sampleRate float64) TransferFunctionZ`. Substitutes `s := (2/T)·(z-1)/(z+1)` symbolically, expands to polynomial in z⁻¹, normalises so `a0=1`. Note: this introduces a `signal → control` import direction; alternatively keep `TransferFunctionZ` in `signal/` and don't import `control`. Recommended: **define `signal.TransferFunctionZ` and `signal.AnalogPrototype` natively**, reuse `control.evalPoly` private helper logic.
3. **Butterworth analog prototype** (~30 LOC).
   `ButterworthAnalog(order int, cutoff float64) []complex128` — returns N poles on `ω_c`-radius circle at angles `θ_k = π/2 + (2k-1)π/(2N), k=1..N`. No zeros (all-pole). For LP only (HP via `s → ω_c²/s`, BP/BS via standard transforms).
4. **DesignButterworth high-level entry** (~80 LOC).
   `DesignButterworthLowpass(order int, cutoff, sampleRate float64) []Biquad` — chains: prewarp → analog Butterworth poles → bilinear → factor into biquads (pair conjugates) → return cascade.
5. **Biquad type + DF-II-T processor** (~60 LOC).
   ```go
   type Biquad struct { B0, B1, B2, A1, A2 float64; s1, s2 float64 }
   func (b *Biquad) Process(in, out []float64) // DF-II-T inner loop
   func (b *Biquad) ProcessSample(x float64) float64
   func (b *Biquad) Reset()
   ```
   DF-II-T pseudocode (Smith CCRMA Filters §III.D):
   ```
   y = b0*x + s1
   s1 = b1*x - a1*y + s2
   s2 = b2*x - a2*y
   ```
6. **RBJ cookbook constructors** (~120 LOC).
   `RBJLowpass(f0, Q, sampleRate)`, `RBJHighpass`, `RBJBandpassPeak`, `RBJBandpassSkirt`, `RBJNotch`, `RBJAllpass`, `RBJPeakingEQ(f0, Q, dBgain, fs)`, `RBJLowShelf(f0, Q, dBgain, fs)`, `RBJHighShelf`. Direct from Bristow-Johnson 2005 cookbook.
7. **Cascade-of-biquads** (~30 LOC).
   `type Cascade []Biquad; func (c Cascade) Process(in, out []float64)`. Per-section overflow scaling pass at construction (Mitra §6.4 — pole-zero pairing closest-to-unit-circle).
8. **Frequency response evaluation** (~15 LOC).
   `func (b Biquad) FreqResponse(omega float64) complex128` — evaluate `H(e^{jω})`. Cascade variant multiplies sections.

### Golden-file vectors (30 minimum)
- 5× Butterworth order 2/4/6 lowpass at f_c=1k,4k,8k @ fs=48k → coefficient sets.
- 5× DC/Nyquist/cutoff identity checks (`|H(0)|=1`, `|H(π)|=0`, `|H(ω_c)|²=0.5`).
- 5× RBJ peaking EQ at f0=1k, Q={0.5,1,2,5,10}, gain=+6 dB.
- 5× RBJ notch — verify `H(e^{jω0}) ≈ 0`, `Q` controls bandwidth.
- 5× impulse-response decay rate matches `|p_max|` (Pin C).
- 5× IEEE 754 edges: subnormal input, +Inf, NaN propagation, -0.0 sign preservation, fs at boundary.

### Day-2 follow-up (~600 LOC additional)
- Chebyshev-I/II prototypes (~120 LOC, needs `cosh/acosh` only — already in stdlib).
- Bessel via reverse-Bessel polynomial coefficient table for orders 1–10 (~80 LOC), with compile-time-constant pole tables.
- Impulse-invariant transform helper (~40 LOC) + aliasing warning on consumers passing non-bandlimited prototypes.
- Step-invariant for control (~40 LOC).
- Group-delay function (~10 LOC).
- Steady-state init helper for biquad (~20 LOC).
- Denormal flush in `BiquadProcess` hot path — add `if math.Abs(s1) < 1e-30 { s1 = 0 }` after each step OR document the trap.

### Day-3 (deferred)
- Elliptic (Cauer) — blocked on slot 299 (Jacobi elliptic functions).
- Yule-Walker direct-form digital design (no analog prototype).
- Steiglitz-McBride iterative IIR identification.
- Fixed-point coefficient quantisation noise analysis.

### Documentation hygiene to fix today (independent of new code)
- `signal/filter.go:97` EMA — add docstring sentence: *"Equivalent to a 1-pole IIR lowpass with cutoff `f_c = -ln(1-α)·fs/(2π)` for small α, or `α = 1 - exp(-2π·f_c/fs)`."* Currently the cutoff is invisible to users.
- `control/filter.go:15` `LowPassFilter` — same caveat. Cross-reference `signal.DesignButterworthLowpass` once it exists.
- `control/transfer.go:14` — note that this is **continuous-time only**; for discrete-time `H(z)`, see `signal.TransferFunctionZ` (when shipped).

## Sources

### Repo files cited
- `C:\limitless\foundation\reality\signal\filter.go` (lines 19, 54, 97, 130)
- `C:\limitless\foundation\reality\control\filter.go` (lines 15, 38, 74, 103)
- `C:\limitless\foundation\reality\control\transfer.go` (lines 14, 21, 36, 73, 122, 239)
- `C:\limitless\foundation\reality\control\pid.go` (lines 36–123)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\132-signal-missing.md` (Tier-1 §1.4, §1.5 already enumerated this gap)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\312-dive-bilinear-bias.md` (Mitchell-Netravali / Lanczos image kernels — separate concern; reuses "bilinear" naming for image-domain interp, not s→z)
- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md:335` (slot 315 definition)

### Canonical references (web/textbook, no fetch performed — these are the bibliography I'd cite in the PR)
- **Oppenheim & Schafer**, *Discrete-Time Signal Processing*, 3rd ed. (2010) — §7 IIR design from analog prototypes; §6.5 DF-II-T canonical realisation; §7.1 impulse-invariance aliasing analysis.
- **Mitra**, *Digital Signal Processing: A Computer-Based Approach*, 4th ed. (2010) — §9 IIR design; §6.4 cascade-of-biquads pairing/scaling; §9.4 bilinear transform.
- **Antoniou**, *Digital Signal Processing: Signals, Systems, and Filters* (2005) — §11 elliptic (Cauer) design with full Jacobi-elliptic-function derivation; §12 Bessel filters.
- **Smith, J.O.**, *Introduction to Digital Filters with Audio Applications*, CCRMA online (`https://ccrma.stanford.edu/~jos/filters/`) — §III.D Direct-Form-II Transposed; §VI bilinear transform; §V series/parallel sections.
- **Bristow-Johnson, R.**, *Cookbook formulae for audio EQ biquad filter coefficients* (2005) — `https://www.w3.org/TR/audio-eq-cookbook/` and `https://shepazu.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html`. Closed-form RBJ biquad coefficients. **Audio-DSP industry standard.**
- **Storch, L.** (1954) — *Synthesis of constant-time-delay ladder networks using Bessel polynomials* — Bessel pole tables.
- **Astrom & Wittenmark**, *Computer-Controlled Systems*, 3rd ed. (1997) — §7 Tustin/ZOH/Step-invariant equivalents; control-theoretic discretisation.
- **Sripad & Snyder** (1977) — *A necessary and sufficient condition for quantization errors to be uniform and white*, IEEE TASSP — fixed-point biquad noise analysis (deferred).
- **Higham, N.J.**, *Accuracy and Stability of Numerical Algorithms*, 2nd ed. (2002) — §13 noise-gain analysis of recursive filters.
- **Parks & Burrus**, *Digital Filter Design* (1987) — Butterworth/Chebyshev/Elliptic pole formulas.
- **Yule, G.U.** (1927) / **Walker, G.** (1931) — original Yule-Walker AR-fitting equations; basis of `scipy.signal.yulewalker`.
- **Steiglitz, K. & McBride, L.E.** (1965) — *A technique for the identification of linear systems*, IEEE TAC — iterative direct-form digital IIR fitting.

### Cross-link to other slots
- Slot 132 — already enumerated this gap at higher level (signal-missing).
- Slot 053/054 — control-theory SOTA/API; PID-as-IIR special case.
- Slot 008 — audio-sota; missing parametric EQ.
- Slot 167 — synergy-audio-signal — flagged biquad cascade need.
- Slot 003 — acoustics-sota — room equalisation requires this.
- Slot 299 — special-functions — Jacobi `cn/sn/dn` blocks elliptic-prototype Day-3 work.
- Slot 270 — graph signal processing — graph IIR is a clean follow-on once biquads exist.
- Slot 312 — separate "bilinear" usage (image kernel, not s→z) — call out the namespace overlap explicitly in `signal/iir.go` doc to prevent user confusion.

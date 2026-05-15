# 131 | signal-numerics

**Agent:** 131 of 400
**Topic:** signal: FFT roundoff growth, IIR filter quantization, Hilbert envelope causality
**Scope:** `C:/limitless/foundation/reality/signal/` — `fft.go` (180 LOC), `filter.go` (174 LOC), `window.go` (113 LOC), `signal_test.go` (852 LOC), `testdata/signal/fft.json`.
**Date:** 2026-05-07

## TL;DR

FFT/IFFT/PowerSpectrum/Convolve/Window kernels are numerically clean, with one
real concern (twiddle-factor recurrence roundoff at large N) and a docstring
lie (MovingAverage claims "running sum" but recomputes per window). **Topic
title's "IIR filter quantization" and "Hilbert envelope causality" cannot be
audited — neither IIR filters nor Hilbert transforms exist here.** Likewise
no Welch, no RFFT, no FIR helpers, no biquad. Surface is small, mostly correct.

---

## Inventory of `signal/` (what actually exists)

Functions exposed:

| File | Function | Notes |
|---|---|---|
| fft.go | `FFT(real, imag)` | Cooley-Tukey radix-2 DIT, in-place, twiddle by recurrence |
| fft.go | `IFFT(real, imag)` | Conjugate-FFT-conjugate-scale; 1/N at end |
| fft.go | `PowerSpectrum(real, imag, out)` | FFT then |X[k]|^2 for k in [0, N/2] |
| fft.go | `FFTFrequencies(n, sr, out)` | k * sr / N |
| filter.go | `Convolve(signal, kernel, out)` | Direct O(N·M); full linear convolution |
| filter.go | `MovingAverage(signal, w, out)` | Centered window, **NOT** running sum despite docstring |
| filter.go | `ExponentialMovingAverage(signal, alpha, out)` | First-order IIR EMA only |
| filter.go | `MedianFilter(signal, w, out)` | Sort-per-window O(N·W·log W) |
| window.go | `HannWindow / HammingWindow / BlackmanWindow / ApplyWindow` | Symmetric (n-1) form |

**Functions absent that the topic implies should exist:**
- `RFFT` / `IRFFT` (real-input FFT exploiting Hermitian symmetry)
- `IIR` / `Biquad` (Direct Form I, II, II-T) — only the trivial first-order EMA exists
- `FIR` (linear-phase symmetric/anti-symmetric helpers)
- `Hilbert` (analytic signal / instantaneous envelope / phase)
- `Welch` / `Bartlett` / `Periodogram`
- `Correlate` / `XCorr`
- `STFT` / `ISTFT`
- Filter-design (Butterworth/Chebyshev/Elliptic poles → coefficients)
- `Resample` / `Decimate` / `Upsample`

This is a hard gap. Most of the topic checklist (IIR quantization, Hilbert
causality, biquad DF-II vs DF-I, Welch bias-variance) is moot here. Defer those
to a missing-features agent (likely #132–135 in the `signal-*` series).

---

## FFT (Cooley-Tukey radix-2 DIT, in-place)

### Algorithm

`fft.go:49-91`. Standard textbook radix-2 DIT: bit-reverse permute, then
`log2(N)` stages of size-`size` butterflies. Twiddles computed by per-stage
**multiplicative recurrence**:

```go
angleStep := -2.0 * math.Pi / float64(size)
wReal := math.Cos(angleStep)
wImag := math.Sin(angleStep)
// ... inner butterfly loop:
newReal := curReal*wReal - curImag*wImag
newImag := curReal*wImag + curImag*wReal
```

That is, only **one** `cos`/`sin` pair per stage; subsequent twiddles are
obtained by complex multiplication.

### Numerical properties

**Recurrence roundoff (the real concern).**
Within one stage, the recurrence performs `halfSize - 1` complex multiplications.
Each multiplication of two unit-magnitude numbers introduces ~ε relative error,
so after `k` iterations the magnitude drifts as `(1+ε)^k ≈ 1 + k·ε`. With
ε ≈ 2.22e-16:

| N | largest stage (halfSize) | worst recurrence drift |
|---|---|---|
| 2^10 = 1024 | 512 | ~1.1e-13 |
| 2^12 = 4096 | 2048 | ~4.5e-13 |
| 2^16 = 65536 | 32768 | ~7.3e-12 |
| 2^20 ≈ 1M | 524288 | ~1.2e-10 |
| 2^24 ≈ 16M | 8388608 | ~1.9e-9 |

Total butterfly error compounds across `log2 N` stages and grows ~
**O(N log N · ulp)** for the output (Higham, *Accuracy and Stability*, §24).
The function comment claims "Precision: 1e-9 for 1024-point". For N=1024 that
is generous (true error is closer to 1e-13). The claim **breaks at roughly
N ≥ 2^22**, where recurrence drift alone exceeds 1e-9.

**Alternatives, ordered by cost vs accuracy:**

1. **Twiddle table per stage**, recomputed each call: `cos(2πk/size)` and
   `sin(2πk/size)` for each butterfly. `O(N log N)` trig calls but no
   accumulation; error stays at `O(log N · ulp)` ≈ 1e-15 even at N=2^24.
2. **Half-table trick** (Bluestein/Singleton): build a single quarter-period
   sin table sized N/4, derive others by symmetry. One-shot allocation.
3. **Compensated recurrence** (Tukey 1965, Buneman 1987): refresh accumulator
   every √N steps via direct trig.

Option (1) is the simplest fix, costs ~2× wall time, and the package already
calls trig functions in window builders, so the precedent exists. The
zero-allocation contract is preserved if the call signature exposes a
caller-provided twiddle scratch (e.g. `FFT(real, imag, twiddleScratch)`).

**Recommendation:** for the `Pistachio (audio FFT)` consumer at typical
N ≤ 4096, the current recurrence is fine. Add a doc note that
**precision degrades for N ≥ 2^16**, and either (a) switch to per-butterfly
direct trig, or (b) add a guarded fast-path with explicit error budget.

**Bit-reversal (`bitReverse`, fft.go:21-35).**
Bog-standard "Gold's algorithm": carry-flag-style increment of a reversed
counter. Only swaps when `i < j` (avoid double-swap). No allocation. Correct.

**Butterfly arithmetic order (fft.go:76-82).**
```go
tReal := curReal*real[oddIdx] - curImag*imag[oddIdx]
tImag := curReal*imag[oddIdx] + curImag*real[oddIdx]
real[oddIdx] = real[evenIdx] - tReal
imag[oddIdx] = imag[evenIdx] - tImag
real[evenIdx] += tReal
imag[evenIdx] += tImag
```
Critical: `real[oddIdx]` is written **after** `tReal` and `tImag` have already
captured the original `real[oddIdx]` value. Same for `imag[oddIdx]`. This is
correct — no read-after-write hazard. Many naive implementations get this
wrong.

**FMA opportunity.**
The two complex-multiply expressions and the four butterfly stores are textbook
candidates for `math.FMA` (Go 1.14+). Using `FMA(a, b, c) = a*b + c` rounded
once cuts twiddle drift roughly in half and butterfly loss by ~1 ulp per stage.
This is a numerics improvement, not a correctness fix.

### Edge cases

- `n <= 1`: short-circuit return. `len1` golden test confirms zero modification.
- `n` not power of 2: panics. Documented.
- `len(real) != len(imag)`: panics.
- IEEE 754 specials (Inf/NaN): not tested. The recurrence will propagate NaN
  globally on a single NaN input bin (since each butterfly mixes data across
  the array). That is the mathematically correct behavior, but means **a
  single NaN in the input pollutes the entire output spectrum** without
  warning. No defensive check; documenting "Garbage In, NaN Out" would help.

---

## IFFT (conjugate trick)

`fft.go:101-127`. Implementation:

```go
for i := range imag { imag[i] = -imag[i] }     // conjugate input
FFT(real, imag)
scale := 1.0 / float64(n)
for i := range real {
    real[i] *= scale
    imag[i] = -imag[i] * scale                  // conjugate output, scale
}
```

This is the canonical **conjugate-FFT-conjugate** identity:
`IFFT(X) = (1/N) · conj(FFT(conj(X)))`. Correct.

**1/N normalization location.** Applied **once** at the end, **after** the
FFT and the second conjugation. No double-scaling, no missing scaling. This
is the conventional "asymmetric" normalization (forward unscaled, inverse 1/N).
NumPy and MATLAB use the same convention; SciPy's `fft.norm="backward"`
matches. Cross-language golden files therefore Just Work.

The opposite convention — symmetric `1/√N` on both — is *not* used here. If a
consumer expects energy-conserving Parseval, they must apply `1/√N` themselves.
Worth documenting.

**Roundtrip error.** `TestIFFT_Roundtrip` (signal_test.go:160) asserts FFT→IFFT
recovers original to **1e-10** at N=16. Far inside the budget. With the
conjugate trick, IFFT inherits FFT's error directly plus one `1/N` multiply,
so total roundtrip error is roughly `2 · O(N log N · ulp)` ≈ 1.8e-13 at N=16.
The 1e-10 tolerance is loose — a **1e-13** assertion would still pass and
catch real regressions.

---

## PowerSpectrum

`fft.go:140-158`. Calls `FFT` in-place then writes `out[k] = real[k]^2 + imag[k]^2`
for `k in [0, N/2]`. Real signals only get `N/2 + 1` output bins because the
upper half is the complex conjugate of the lower half (Hermitian symmetry).
The function does not enforce that `imag` was zero on input — if a complex
signal is passed, the output is the squared magnitude of the **two-sided**
spectrum truncated to the lower half, which is generally wrong. Document or
add a check.

**No 1/N scaling on power.** The `TestPowerSpectrum_DC` golden expects the
DC bin to be `N^2 = 64` for N=8 of all-ones. That is the un-normalized power.
Compare against the NumPy/SciPy convention of `1/(N · fs · windowSum²)` for
PSD: this differs by factors of N, sample rate, and window energy. **Naming
the function `PowerSpectrum` rather than `PowerSpectralDensity` is correct**
— it returns squared magnitude, not PSD. But a consumer integrating against
frequency must know the difference.

**No window correction.** Common spectral pipelines apply a window before FFT
(reducing leakage). The window's energy `Σ w[i]²` (or amplitude `Σ w[i]`)
must be divided out for a fair PSD estimate. None of that is here. That is
a missing-feature, not a bug.

---

## Convolve

`filter.go:19-40`. Direct nested-loop O(N·M). For each `i, j`,
`out[i+j] += signal[i] * kernel[j]`. Output zero-cleared first. Correct
linear convolution. Output length is fixed at `N+M-1` ("full" mode).

**Edge handling.** No "same" / "valid" / "wrap" modes. The full-output choice
sidesteps the entire boundary question, which is the right zero-dependency
default — but consumers building a steady-state filtered signal will need to
slice `out[k_off : k_off+N]` manually, where `k_off = (M-1)/2` for symmetric
kernels. Not documented.

**Numerical.** Accumulates into `out[i+j]` with naive summation. For long
signals or large-magnitude data this can drift by `O(N·M · ulp · max|x·k|)`.
Pairwise summation or Kahan would help when N·M ≥ 1e8 or so. For the typical
audio use cases (M ≤ 64 FIR taps, N up to 1e6), naive summation gives
~1e-8 absolute error which is well below audible.

**Convolution theorem path.** Doc says "for long kernels, FFT-based
convolution is preferred (compose FFT + multiply + IFFT from this package)".
That is true in principle, but the package provides **no helper** to do it,
and the user has to handle (a) zero-padding to next power of 2, (b) pointwise
complex multiply, (c) IFFT and trim. This is exactly where the missing
`Correlate` / `FFTConvolve` belong.

---

## MovingAverage — docstring lies about complexity

`filter.go:54-84`. Docstring (line 51): `// Zero heap allocations — uses running sum.`

But the body (line 79):
```go
sum := 0.0
count := hi - lo + 1
for j := lo; j <= hi; j++ {
    sum += signal[j]
}
```
**Recomputes the sum from scratch every output sample.** Complexity is
O(N·W), not O(N). This is a docstring bug.

**Numerical implication.** Recomputing actually has *better* numerical
properties than a true running sum, because a running sum
`sum -= signal[i-W]; sum += signal[i+W]` accumulates roundoff over the
entire signal. For long signals with mixed positive/negative values, the
running sum can drift unboundedly (catastrophic cancellation). The current
recompute caps error at `O(W · ulp · max|signal|)` per output. So the docstring
is wrong about *what* it does and the actual implementation is *more* correct
than the documented one would be. **Fix the docstring, keep the code.**

If a true O(N) is needed, use **Kahan-compensated running sum**, not naive.

---

## ExponentialMovingAverage — the only IIR in the package

`filter.go:97-114`. Single first-order IIR:
`y[n] = α·x[n] + (1-α)·y[n-1]`, with `y[0] = x[0]`. This is a one-pole
low-pass with pole at `1 - α`.

**Quantization analysis (the topic question, applied to this small case).**
The feedback path holds **one** previous output, multiplied by `(1-α)`. With
`α ∈ (0, 1]` strictly enforced (filter.go:102), the pole magnitude
`|1-α| < 1` so the filter is **strictly stable** and quantization noise
decays geometrically. No biquad-style numerical pathology because there is no
biquad. With `α = 1`, the pole moves to 0 — pure pass-through, also stable.
With α very small (e.g. 1e-6) the time constant 1/α is enormous and roundoff
in `(1-α)·y[n-1]` accumulates as a slow DC drift, but for any reasonable α
(≥ 1e-3) this is sub-ulp.

**No Direct-Form variants needed for first-order**, so the topic question
about "DF-II vs DF-I" does not apply until biquads are added.

`oneMinusAlpha := 1.0 - alpha` is precomputed once outside the loop. Good
(prevents repeated FP subtraction). For `α` extremely close to 1, this
suffers cancellation (e.g. α = 1 - 2^-53 → `oneMinusAlpha = 0.0`), but the
panic guard `α > 1` does not catch values *almost* 1, which is fine —
that's degenerate input and the result is still correct (pass-through).

---

## MedianFilter

`filter.go:130-174`. Centered window, sort-per-window. Uses 64-element
**stack scratch** for windows ≤ 63, falls back to heap allocation otherwise.
For the typical "spike removal at W=3 or 5" use case, zero-alloc is preserved.

Numerical: median is exact (just element selection or 2-element average). No
roundoff concerns. Even-sized window does `(buf[count/2-1] + buf[count/2]) / 2.0`
which has zero rounding error for IEEE 754 mean of two finite values when
neither overflow nor cancellation happens — both ruled out for typical audio.
Correct.

**Algorithmic.** O(N·W·logW) is correct for sort-based; for large windows, a
two-heap or quickselect approach reaches O(N·W) or O(N·logW). Not a numerics
issue.

---

## Window functions

`window.go`. All three windows use the **symmetric `(n-1)` form**:
- Hann: `0.5·(1 − cos(2πi/(n−1)))`
- Hamming: `0.54 − 0.46·cos(2πi/(n−1))`
- Blackman: `0.42 − 0.5·cos(2πi/(n−1)) + 0.08·cos(4πi/(n−1))`

This is the **DSP textbook (Harris 1978)** symmetric form, where
`w[0] = w[n-1] = 0` (Hann/Blackman) or `0.08` (Hamming). Tests pin both
endpoints to **1e-15** tolerance — passing means the cosine of `0` and
`2π · (n-1)/(n-1) = 2π` both round to within ulp of 1.0, which is
guaranteed by Go's `math.Cos`.

**Alternative form: `2π·i/n`** (NumPy `numpy.hanning(n, sym=False)`,
SciPy `get_window(..., fftbins=True)`). This is the **periodic** or
**FFT-friendly** form; it does *not* go to zero at `n−1` (the last point is
"the next start"). Reality uses the symmetric form, which is the right
default for filter design but **wrong for spectral analysis** (where you
want periodicity inside an FFT frame). Both forms differ by 1 sample and
the difference is invisible at large N. **Document the convention** so
consumers don't accidentally double-window.

**Length-1 case** correctly returns `1.0` rather than dividing by zero
(window.go:23-26 etc). Pin tested with tolerance 0.

**Symmetry tests** assert `w[i] == w[n-1-i]` to 1e-15 (Hann) and 1e-14
(Hamming/Blackman). With `cos(2π · i /(n-1))` and `cos(2π · (n-1-i) /(n-1))
= cos(2π - 2πi/(n-1))`, the IEEE 754 result depends on argument reduction.
For typical n ≤ 1024, the symmetry is exact to ~1 ulp. The 1e-14 vs 1e-15
choice is empirical; both pass.

**ApplyWindow** is `out[i] = signal[i] * window[i]`. Trivial. No FMA, no
accumulation, no precision concern. The length-mismatch panic uses
`!=` for `window` but `<` for `out` (window.go:106) — slight asymmetry, not
a bug.

---

## What's NOT here that the topic asked about

This list is the dominant finding for the "numerics" agent: most of the
topic's vocabulary applies to functions that don't exist.

| Topic checklist item | Status | Notes |
|---|---|---|
| FFT/IFFT roundoff growth | **Audited** | Concern at N ≥ 2^16 |
| RFFT Hermitian symmetry | **N/A — RFFT not implemented** | Recommend adding |
| Inverse-FFT 1/N location | **Audited — correct** | At end, asymmetric convention |
| IIR feedback quantization | **N/A — only first-order EMA** | Recommend Biquad DF-II-T |
| Biquad DF-II vs DF-I | **N/A — not implemented** | DF-II-T preferred for floats |
| FIR linear-phase symmetry | **N/A — no FIR helper, only Convolve** | Add helper that asserts symmetry |
| Hilbert transform / envelope | **N/A — not implemented** | Phase-shift accuracy is moot |
| Convolution edge handling | **Audited — full mode only** | Add 'same'/'valid' modes |
| Window numerical exactness | **Audited — symmetric form** | Endpoints pin to 1e-15 |
| Welch / spectral estimation | **N/A — not implemented** | No PSD machinery at all |

---

## Concrete recommendations (numerics-only)

Ordered by ratio of correctness payoff to implementation cost.

1. **Fix the `MovingAverage` docstring** — remove "uses running sum", state
   complexity O(N·W). 1-line fix. (Or implement true Kahan running sum and
   keep the doc.)
2. **Document FFT precision degradation at N ≥ 2^16** in `fft.go:46`.
   Optionally swap to per-butterfly direct trig at large N.
3. **Tighten `TestIFFT_Roundtrip` tolerance from 1e-10 → 1e-13.**
   Catches twiddle regressions earlier.
4. **Add NaN/Inf golden cases** to `testdata/signal/fft.json`. Even if the
   behavior is "NaN propagates", pin it.
5. **Document window convention** (symmetric `(n-1)` denominator) in each
   window's docstring. Note that consumers wanting periodic FFT-frame windows
   need a different form.
6. **Document PowerSpectrum is squared-magnitude, not PSD.** Add note about
   1/N, window-energy, and 1/fs corrections needed for true PSD.
7. **Use `math.FMA` in butterfly arithmetic** when `math.FMA` is available.
   Roughly halves twiddle drift at zero perf cost on AMD64/ARM64.

None of these are blocking bugs. The package is small, focused, and the
arithmetic that *is* present is essentially correct.

---

## Files referenced

`C:/limitless/foundation/reality/signal/{fft.go,filter.go,window.go,signal_test.go,testdata/signal/fft.json}`

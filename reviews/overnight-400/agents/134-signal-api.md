# 134 | signal-api

**Agent:** 134 of 400
**Topic:** signal: streaming filter state, sample-rate handling, complex spectra
**Scope:** API ergonomics of `C:/limitless/foundation/reality/signal/` —
`fft.go`, `filter.go`, `window.go` (~470 LOC, 12 entry points). Compared
against sibling `audio/` (top-level), `audio/spectrogram/`, `audio/cqt/`,
`audio/onset/`, `audio/pitch/`, and `control/filter.go`.
**Date:** 2026-05-08

## Brief

131 audited numerics. 132 inventoried missing primitives. 133 read SOTA
libraries. **134 is purely about the *shape* of the public surface that
already exists** — argument lists, type choices, struct vs. free-function,
streaming vs. batch, where the sample rate lives, who owns the window. The
package is small but its conventions ripple out: `audio/` already extends
them, and the topic gaps in 132 (biquad, IIR design, Welch, Hilbert) will
need API decisions that should fit the existing house style. This report
diagnoses that style and where it strains.

---

## Inventory of API shapes

### `signal/` — 12 entry points

```
FFT                      (real, imag []float64)
IFFT                     (real, imag []float64)
PowerSpectrum            (real, imag, out []float64)
FFTFrequencies           (n int, sampleRate float64, out []float64)
Convolve                 (signal, kernel, out []float64)
MovingAverage            (signal []float64, windowSize int, out []float64)
ExponentialMovingAverage (signal []float64, alpha float64, out []float64)
MedianFilter             (signal []float64, windowSize int, out []float64)
HannWindow               (n int, out []float64)
HammingWindow            (n int, out []float64)
BlackmanWindow           (n int, out []float64)
ApplyWindow              (signal, window, out []float64)
```

**Zero structs. Zero interfaces. Zero state. Zero `complex128`. Every
function is pure — input slice(s) plus scalar parameters, output slice
(or in-place). No errors returned (panic-only). No options. No
`io.Reader`. No goroutines. No allocations in the documented hot
path.** This is the rarest and most disciplined Go API style in the
repo. Either the package's strength or its limitation, depending on
which way you read the gaps.

### `audio/` extensions of the same surface

```
audio/MelFilterbank      (sampleRate float64, nFFT, numFilters int,
                          fMin, fMax float64, out []float64)
audio/PowerSpectrum      (real, imag, out []float64)         // mirrors signal
audio/ApplyFilterbank    (power, filterbank []float64,
                          numFilters, nBins int, out []float64)
audio/spectrogram/Compute(samples []float64, frameSize, hopSize int,
                          window []float64) [][]complex128   // ALLOCATES
audio/spectrogram/Inverse(stft [][]complex128, frameSize, hopSize int,
                          window []float64) []float64        // ALLOCATES
audio/spectrogram/Magnitude/LogMagnitude/PowerSpectrum/HalfSpectrum
                         (stft [][]complex128) [][]float64
audio/cqt/CQT            (x []float64, sr, fMin float64, binsPerOctave,
                          octaves int, out []complex128) error
audio/onset/SpectralFluxOnset/Strength/ComplexDomain
                         (stft [][]complex128) []int  or  []float64
audio/pitch/AutocorrelationPitch
                         (frame []float64, sampleRate, fMin, fMax float64) float64
audio/vibration/FundamentalHz
                         (frameReal, frameImag []float64,
                          sampleRate, fMin, fMax float64) float64
```

Two patterns visible: (a) some `audio/` calls follow the **signal style**
(`real, imag []float64` paired floats, scalar `sampleRate`, panic on
violation, caller-allocated `out`); (b) others use a **second style**
(`[][]complex128`, returns allocated slices, returns `error`). The split
is **stft.go vs. filter-style**, and it crosses the package boundary
mid-call: `spectrogram.Compute` consumes paired floats from
`signal.FFT` internally then materialises `[][]complex128` for the rest
of `audio/`. This is the fault line worth naming.

---

## Streaming filter state — does NOT exist

### What's missing

There is no biquad struct. No `Filter` type. No `Process(in, out)`
method. No persistent state at all in the `signal/` package. The single
IIR-shaped function (`ExponentialMovingAverage`) takes the entire
signal as a slice and is **batch-only** — there is no way to pass one
sample at a time and have it remember `y[n-1]`.

The closest streaming-shaped code in the repo is `control/filter.go`'s
`LowPassFilter(prev, current, alpha) float64` — same math as
`ExponentialMovingAverage` but **caller threads the state by hand**:

```go
y := signal[0]
for i := 1; i < len(signal); i++ {
    y = control.LowPassFilter(y, signal[i], alpha)
    out[i] = y
}
```

### What `audio/` does when it needs state

`audio/DegradationTracker` (degradation.go:28) is the **only** stateful
DSP-shaped object in the repo's audio path. It is a public struct with
exported fields:

```go
type DegradationTracker struct {
    BaselineN    int
    BaselineMean float64
    BaselineM2   float64
    WindowSize   int
    Window       []float64
    WindowHead   int
    WindowFill   int
}
```

Updated by **free functions taking pointers**:

```go
func PushObservation(t *DegradationTracker, x float64)
func UpdateBaseline (t *DegradationTracker, x float64)
func WindowMean     (t *DegradationTracker) float64
```

Not methods (`(t *Tracker) Push`). Free functions with `t *Tracker` as
the **first argument**. This is a real stylistic choice — visible in
`audio/fingerprint.go` too (`UpdateFingerprint(fp, x)`,
`FingerprintVariance(fp, out)`). The codebase rejects method syntax
even when the type is a struct. Reasons (inferred): cross-language
codegen reads C-style function signatures more uniformly; `go doc`
groups by package not by receiver; pointer ownership is explicit.

### What a `Biquad` should look like, in this style

132 §1.4 will need a struct with 5 immutable coefficients and 2
state-of-state floats. Two API options that fit existing conventions:

**Option A — fingerprint/tracker pattern (free funcs, pointer first):**
```go
type Biquad struct {
    B0, B1, B2 float64  // numerator coefficients
    A1, A2     float64  // denominator (assume A0=1, normalised)
    Z1, Z2     float64  // DF-II-T state
}

func RBJLowpass(cutoff, q, sampleRate float64) Biquad
func ProcessBiquad(b *Biquad, in, out []float64)   // batch
func ProcessBiquadSample(b *Biquad, x float64) float64  // streaming
func ResetBiquad(b *Biquad)
```

**Option B — explicit per-call state (full FP, no methods, no pointer):**
```go
type BiquadCoeff struct{ B0, B1, B2, A1, A2 float64 }
// State threaded by caller, à la control.LowPassFilter:
func ProcessBiquadSample(c BiquadCoeff, z1, z2 *float64, x float64) float64
```

Option A matches the `DegradationTracker` precedent best. Option B is
more functional but uglier at the call site (3-pointer call, awkward).
**Recommend A.** The choice should be made *before* 132 §1.4-1.5 land,
because every IIR design function will return one of these and changing
later is breaking.

### What a Welch / STFT processor should look like

For block-streaming (frame-at-a-time PSD or spectrum), an
`OverlapAdd` or `STFTProcessor` struct holding the analysis window,
hop, ring buffer, and reusable FFT scratch is the natural shape.
`spectrogram.Compute` already pre-allocates `frameReal/frameImag`
across frames internally — promote to public state and the streaming
form falls out.

### `io.Reader`?

**Don't.** Reality is a math library, not a service. `io.Reader` brings
buffer ownership ambiguity, error-propagation contracts, and pulls in
`io` (still stdlib but a step toward a runtime package). Caller
chunks, calls `Process(chunk, out)`, threads state via the struct.
That's the GNU-Radio-block model minus the scheduler — exactly the
right level for a math library. 133 §GNU-Radio confirms reality
should not be a streaming runtime. Per-block state + free function +
pointer is the maximum streaming surface this package should expose.

---

## Sample-rate handling — `float64` argument, never embedded

### Convention

Every function that needs a sample rate **takes it as `float64`**:

| Function | Signature |
|---|---|
| `signal.FFTFrequencies` | `(n int, sampleRate float64, out []float64)` |
| `audio.MelFilterbank` | `(sampleRate float64, nFFT, numFilters int, ...)` |
| `cqt.CQT` | `(x []float64, sr, fMin float64, ...)` — note: `sr`, not `sampleRate` |
| `cqt.WindowLength` | `(q, sr, f float64)` |
| `pitch.AutocorrelationPitch` | `(frame []float64, sampleRate, fMin, fMax float64)` |
| `vibration.FundamentalHz` | `(frameReal, frameImag []float64, sampleRate, fMin, fMax float64)` |

No `audio.Config{SampleRate: 44100}`. No global. No `signal.WithSampleRate(sr)`.
No package-level `var SampleRate float64`. **Sample rate is a per-call
scalar.** Pure in the FP sense — same call → same answer, no hidden
state — and matches reality's no-allocation, no-state house style.

### Friction this creates

Audio pipelines pass the same `sampleRate` through 6-10 calls in a
row. A typical Pistachio-style pipeline:

```go
audio.MelFilterbank(sr, nFFT, 80, 0, sr/2, fb)
audio.PowerSpectrum(real, imag, power)
audio.ApplyFilterbank(power, fb, 80, nFFT/2+1, melE)
audio.LogMelEnergies(melE, 1e-10, logE)
audio.MFCC(logE, 13, mfcc)
signal.FFTFrequencies(nFFT, sr, freqs)
cqt.CQT(samples, sr, 27.5, 12, 7, cqtOut)
```

`sr` repeats 4 times. A consumer-side `Pipeline` struct that captures
`sr, nFFT, numFilters, fMin, fMax` and delegates is trivially
written *outside* reality, and that is the right level: the math
library exposes referentially-transparent functions; a thin
audio-application layer (perhaps in `aicore` or in Pistachio itself)
holds rig-specific configuration. **Resist adding `Config` types
inside reality.** They are catnip for "let's also store the buffer"
and "let's also store the FFT plan" and the package becomes a service.

### Inconsistency: `sampleRate` vs `sr`

`signal/` and `audio/` top-level use `sampleRate` (full word).
`audio/cqt/` and `audio/vibration/` use `sr` (abbreviation). Trivial
but worth normalising. Recommend `sampleRate` everywhere — the long
form has identical compile output, reads better in docstrings, and
matches `frameSize`, `hopSize`, `windowSize`.

### Sub-sample rate concept missing

Beat tracking uses `frameRate` (Hz at the frame level, i.e.
sampleRate / hopSize). `audio/beat/Track` takes `frameRate` directly;
caller computes it. This is fine and correct: the frame-level data has
its own sampling rate distinct from the audio sample rate, and the
function neither knows nor needs to know how to derive one from the
other. Document the distinction (audio-sample-rate vs frame-rate vs
spectrum-bin-rate) in the package doc.

### What does NOT exist anywhere

- No `Resample` / `Decimate` / `Upsample` (132 §2.8). When it
  arrives, it will be the **first** function in `signal/` that takes
  *two* sample rates (`fromRate`, `toRate`) — API question of
  argument order matters.
- No anti-aliasing filter design coupled to sample rate.
- No Nyquist-violation detection helper. `cqt.CQT` does this once
  (`fTop >= sr/2 → ErrSampleRateTooLow`); pattern should generalise
  into a `signal.NyquistOK(freq, sampleRate) bool` or similar.

---

## Complex spectra — the paired-float / complex128 split

### The fault line

`signal/` is **paired float64**. `FFT` takes `(real, imag []float64)`,
in-place. `PowerSpectrum` takes `(real, imag, out)`. `FFTFrequencies`
returns `[]float64` (real). Zero use of the `complex128` type
anywhere in the package (`grep complex128 signal/ → no matches`).

`audio/spectrogram/` is **`[][]complex128`**. `Compute` returns
`[][]complex128`. Every downstream consumer in `audio/onset/`,
`audio/segmentation/`, `audio/separation/` takes `[][]complex128`.
`audio/cqt/CQT` writes `[]complex128`. `audio/onset/ComplexDomainOnset`
takes `[][]complex128`.

The conversion happens **inside `spectrogram.Compute`** at line 93-95:

```go
row := make([]complex128, frameSize)
for k := 0; k < frameSize; k++ {
    row[k] = complex(frameReal[k], frameImag[k])
}
```

That conversion costs an N-element allocation per frame plus a copy.
For a 30 s audio clip at 22 kHz with hop 256, that's ~2600 frames ×
`make([]complex128, 1024)` = ~21 MB of `complex128` allocation per
spectrogram, just to satisfy the type boundary.

### Why two styles?

**Paired floats win when:** the work is in-place transform, no
allocation desired, output slot is the same as input slot (FFT,
IFFT). The `(real, imag)` convention exactly matches the natural
representation of the radix-2 butterfly inner loop — `complex128`
multiplications would actually compile to *more* code on most
backends because Go's `complex128` arithmetic doesn't get FMA-folded
the way `real*real - imag*imag` does. **Numerical and zero-alloc
correct.**

**`complex128` wins when:** the result is a stored matrix passed
between non-trivial functions (STFT → onset → segmentation), the
allocation has already happened anyway (returning `[][]complex128`),
and reading `cmplx.Abs(stft[t][k])` is dramatically clearer than
`math.Hypot(realStft[t][k], imagStft[t][k])`. **API ergonomic
correct.**

Both choices are individually defensible. The friction is the
**boundary**: every consumer crossing from `signal/` to
`audio/spectrogram/` pays the conversion. And the boundary is
*hidden* — there is no `signal.PackComplex(real, imag) []complex128`
helper, callers either invent it or duplicate `spectrogram.Compute`'s
inner conversion loop.

### Recommendation: keep both, name the bridge

Don't unify. Different needs, different shapes. **But add the bridge
explicitly:**

```go
// in signal/fft.go
func PackComplex(real, imag []float64, out []complex128)
func SplitComplex(in []complex128, real, imag []float64)

// or: zero-alloc forms that cast via header trickery — DON'T,
// std lib gives no safe path; allocate.
```

This makes the boundary visible, codable once, and pinable in
golden-file tests. It also creates the question of whether
`signal.FFTComplex(in []complex128, out []complex128)` should exist
as a typed-spectrum convenience layer over the in-place float path.
**Recommend not** — the dual API doubles documentation surface,
doubles golden-file count, and most callers wanting `complex128` are
calling `spectrogram.Compute` rather than raw FFT. The bridge
helpers are enough.

### Hermitian-symmetry waste in `[][]complex128`

For real-valued audio (the entire `audio/` use case), the upper
half of every STFT row is the complex conjugate of the lower half.
`spectrogram.HalfSpectrum` exists but is **a defensive copy** —
`out[t] = make([]complex128, half); copy(out[t], stft[t][:half])`.
The original is still allocated full-size. 50% of the memory is
provably redundant. Once 132 §1.1 (RFFT) lands, the natural API
returns `N/2+1` bins and `[][]complex128` becomes half-sized
naturally — **the memory waste is a 132 problem, not a 134 problem**,
but documenting "consumers wanting half-spectrum should pre-trim
not post-trim once RFFT exists" is an API-style question worth
naming.

---

## FFT input/output convention

### Input shape

Both real and imaginary slices passed as separate `[]float64`.
N must be a power of 2 (panics otherwise). N=1 is a no-op
short-circuit. N=0 falls into the panic path
(`isPow2(0) == false`). No length validation against a maximum;
N up to memory will run though will lose precision (131 §FFT
recurrence).

### Output shape

In-place. Return value is `void`. Caller's input slice IS the
output slice. No defensive copy. Inverse follows the same
in-place contract.

This is **the right call** for a zero-alloc DSP primitive. SciPy's
`scipy.fft.fft` returns a fresh array, but their performance docs
explicitly steer power users to `pyfftw` for in-place. JUCE's
`FFT::performRealOnlyForwardTransform` is in-place. FFTW's planner
returns in-place plans. Reality matches the C-tradition fastpath.

### What's nonstandard

1. **No length output.** `FFT(real, imag)` gives no return; Go's
   slice header carries length, so this is fine, but conventional
   C-style `int FFT(double* real, double* imag, int n)` would have
   returned `n` or an error code. **Reality picks slice length
   from `len(real)`.** Correct, but worth one line of doc.
2. **No norm flag.** SciPy's `norm="forward"` / `"backward"` /
   `"ortho"` — reality is silently `"backward"` (forward unscaled,
   inverse 1/N). 131 §IFFT noted this. Should be in the package
   doc, not just per-function.
3. **No `axis` parameter.** Multi-dim FFT not in scope (yet). SciPy
   has `fftn`, `fft2`. When 2D image processing or 2D spectrogram
   convolution is needed, axis-aware overload is the right shape;
   1D-only is a fine v0.10.0 stance.

### Hidden invariants the panic surface does NOT enforce

- That `imag[i] == 0` for real-valued input. `PowerSpectrum`
  silently misbehaves on non-real inputs (131 §PowerSpectrum). Add
  `signal.AssertReal(imag) // panics if any non-zero` — three
  lines, optional debug check.
- That `real`/`imag` aren't the same slice. Aliased input would be
  catastrophic for FFT but not currently checked. Probably not worth
  a runtime check (Go has no fast slice-aliasing comparison) but
  documenting "caller must not alias real and imag" closes the
  question.

---

## Window functions — precomputed slices, never inline

### Convention

Each window function fills a caller-supplied `out []float64`:

```go
HannWindow(n, out)
HammingWindow(n, out)
BlackmanWindow(n, out)
ApplyWindow(signal, window, out)
```

Caller pre-allocates `out := make([]float64, n)`, calls the builder
**once**, then reuses across many frames via `ApplyWindow`. Zero
trig in the hot path.

This is **strictly better** than the inline alternative
(`signal.HannApply(signal, out)` that recomputes window per call).
At 22 kHz, 1024-frame, 75% overlap, 30 s audio → 2580 frames ×
1024 cosines = 2.6 million `math.Cos` calls saved by precomputing.
Reality gets this right by default.

### What it costs

The caller pays for orchestration:

```go
window := make([]float64, frameSize)
signal.HannWindow(frameSize, window)
// ... per frame:
signal.ApplyWindow(frame, window, windowed)
```

Three lines of setup. Compared to `librosa.stft(samples,
window='hann')` — one line. Trade is correct for the math library
(explicit > implicit; allocation visible), but the audio
application layer ought to provide a one-liner that bundles
"design + apply across frames". `audio/spectrogram/Compute` does
exactly this: takes a `window []float64` argument, applies it
inside the loop. The pattern works.

### Window enum?

`librosa` accepts `window='hann'|'hamming'|'blackman'|...`.
Reality has no `WindowType` enum. To swap windows in a pipeline,
the caller swaps function references manually:

```go
type WindowFunc func(n int, out []float64)
var w WindowFunc = signal.HannWindow
w(frameSize, window)
```

`func(int, []float64)` is the de-facto interface. Idiomatic Go.
**Don't introduce a `WindowType` int enum** — it forces an internal
switch and breaks the open-set property (consumers can ship their
own window builders matching the same signature). The function-
reference idiom is correct.

### Symmetric vs. periodic — undocumented

131 §window noted this is the symmetric (n-1) form, not the
periodic (n) form. Both are correct, neither is wrong, but
spectral-analysis consumers commonly want periodic. **API gap:**
provide both, distinguished by name or by an additional
`periodic bool` argument:

```go
HannWindow(n, out)            // symmetric, current behaviour
HannWindowPeriodic(n, out)    // 2π·i/n form, FFT-frame-friendly
```

Adding `bool` to existing signatures breaks ABI. Name-suffix
variants don't. Recommend the latter. Same for Hamming/Blackman.

---

## Where panic-only error reporting strains

### Inventory

| File | Behaviour |
|---|---|
| `signal/fft.go` | All panic, no errors |
| `signal/filter.go` | All panic, no errors |
| `signal/window.go` | All panic, no errors |
| `audio/spectrogram/Compute` | Panic |
| `audio/cqt/CQT` | **Returns `error`** (`ErrSampleRateTooLow`, `ErrInputTooShort`, `ErrInvalidParams`, `ErrOutputSize`) |
| `audio/beat/Track` | **Returns `error`** (`ErrInvalidParams`, `ErrInsufficientData`) |

The repo is split. **`signal/` panics, `audio/` is mixed.** This
pattern is defensible — panics for "you wrote your code wrong"
(passing odd-N to FFT is a programming error), errors for
"the data ran out at runtime" (CQT got an audio file too short
for the lowest bin). But it requires consistent application of
the rule across the package, and a few cases blur:

- `signal.FFT` panicking on `len(real) != len(imag)` — programmer
  bug, panic correct.
- `signal.MovingAverage` panicking on `windowSize < 1` — could
  argue runtime data, but typically a config error caught at
  startup, panic is fine.
- `audio/spectrogram/Compute` panicking on `len(samples) < 1` —
  that's runtime data. Inconsistent with `cqt.CQT` returning
  `ErrInputTooShort` for the same class of error.

**Recommendation:** for `signal/`, keep panic-only. The pure-math
contract is "you violated the function's domain, that's a bug".
For `audio/`, follow `cqt`/`beat` precedent and return errors for
data-driven failures (input too short, frame too small for FFT,
sample rate too low). 132 §1.4 biquad design functions should
return errors when the cutoff is ≥ Nyquist, not panic.

### The missing helper: `signal.IsPow2(n) bool`

`isPow2` is unexported. Every consumer of FFT must duplicate
the check or risk an opaque "length must be a power of 2" panic
deep in `signal.FFT`. **Two-line export** would let consumers do
defensive checks themselves:

```go
func IsPow2(n int) bool { return n > 0 && n&(n-1) == 0 }
func NextPow2(n int) int { /* trivial bit-twiddle */ }
```

Both are zero-LOC additions and fix a genuine API gap. `NextPow2`
in particular is needed every time a consumer wants to
zero-pad-to-FFT-size, which is the normal case.

---

## Allocation contracts and where they leak

### `signal/` — strict zero-alloc

Every function in `filter.go`, `window.go`, and `fft.go`
documents "Zero heap allocations" in its comment, with one
exception: `MedianFilter` falls back to `make([]float64, count)`
when window size > 63. Documented.

### `audio/spectrogram/` — allocates aggressively

```go
out := make([][]complex128, numFrames)         // outer slice
row := make([]complex128, frameSize)           // per frame, every iter
samples := make([]float64, outLen)             // Inverse
windowSum := make([]float64, outLen)           // Inverse
out[t] = make([]float64, F)                    // every Magnitude/etc call
```

This is ~T+1 allocations per `Compute` call. The boundary between
"primitive math, zero alloc" and "convenience layer, allocates
the answer" is the `signal/` ↔ `audio/spectrogram/` line.
Architecturally correct (zero-alloc primitive at the bottom,
allocating composition at the top), but the **convenience layer
currently has no zero-alloc form**. Adding `Compute` variants
that take pre-allocated `out [][]complex128` would let
real-time pipelines (Pistachio's 60 FPS audio analysis) stay
allocation-free. Currently they cannot.

```go
func ComputeInto(samples []float64, frameSize, hopSize int,
    window []float64, out [][]complex128)  // caller owns out
```

Two-form pattern: allocate-and-return for prototyping, in-place
for production. SciPy doesn't do this; numpy doesn't. JUCE does
(`process(buffer)` mutating in place). The right idiom for
Reality given its zero-alloc north star.

---

## API additions called for by 132 — shape proposals

When 132's missing primitives land, the API choices should be
made *now* so they fit the existing house style:

| Primitive | Recommended shape | Notes |
|---|---|---|
| `RFFT` | `RFFT(real []float64, outReal, outImag []float64)` | Hermitian half-spectrum, paired-float. Pair with `IRFFT(real, imag []float64, out []float64)`. |
| `Goertzel` | `Goertzel(signal []float64, targetFreq, sampleRate float64) (real, imag float64)` | Two scalar returns matches existing convention (`AutocorrelationPitch` returns one scalar; multi-return is fine for tuples). |
| `Bluestein` | `BluesteinFFT(real, imag []float64)` | Same shape as `FFT`. Internally allocates twiddle scratch — document or take optional scratch param. |
| Biquad runner | `type Biquad struct{...}` + `ProcessBiquad(b *Biquad, in, out []float64)` | Free-function-on-pointer, à la `DegradationTracker`. |
| Biquad design | `RBJLowpass(cutoff, q, sampleRate float64) Biquad` (returns by value, no error) or `Butterworth(order int, cutoff, sampleRate float64) ([]Biquad, error)` (slice + error for ≥3rd order) | SOS as the only IIR storage form (133 §scipy). |
| FIR design | `FIRWindow(numTaps int, cutoff, sampleRate float64, w WindowFunc) []float64` | `WindowFunc` typed as `func(int, []float64)` — already de facto. |
| Hilbert | `Hilbert(signal []float64, outReal, outImag []float64)` | Paired-float spectrum convention. |
| Welch | `Welch(signal []float64, segLen, overlap int, w []float64, sampleRate float64, freqs, psd []float64)` | All-out-buffer, no allocation. Pre-built window. |
| STFT (move from audio/) | both `Compute` (alloc) and `ComputeInto` (zero-alloc) | Two-form. |
| Convolution-FFT | `FFTConvolve(signal, kernel, scratch1, scratch2, out []float64)` | Caller owns scratch buffers. |
| Resample | `Resample(in []float64, fromRate, toRate float64, taps []float64, out []float64)` | Caller pre-designs polyphase taps; resampler is pure runner. |

Theme: **pre-allocate, no errors for math-violations, errors for
data-shortage, free function not method, sample rate as `float64`
parameter.** Stays inside the existing house style.

---

## Files referenced

`C:/limitless/foundation/reality/signal/{fft.go,filter.go,window.go,signal_test.go}`
`C:/limitless/foundation/reality/audio/{degradation.go,fingerprint.go,melscale.go}`
`C:/limitless/foundation/reality/audio/spectrogram/{stft.go,magnitude.go}`
`C:/limitless/foundation/reality/audio/cqt/cqt.go`
`C:/limitless/foundation/reality/audio/onset/spectral_flux.go`
`C:/limitless/foundation/reality/audio/beat/beat.go`
`C:/limitless/foundation/reality/control/filter.go`

Adjacent reviews:
`C:/limitless/foundation/reality/reviews/overnight-400/agents/131-signal-numerics.md`
`C:/limitless/foundation/reality/reviews/overnight-400/agents/132-signal-missing.md`
`C:/limitless/foundation/reality/reviews/overnight-400/agents/133-signal-sota.md`

---

## Two-line summary

`signal/` is paired-`[]float64`, in-place, panic-only, zero-state,
zero-`complex128`, sample-rate-as-scalar — a strict pure-math API
that breaks at `audio/spectrogram/`'s `[][]complex128` boundary,
has no streaming-state primitive (no Biquad struct, no
`Process(in,out)` runner) yet has a precedent for one
(`DegradationTracker` + free-function-on-pointer pattern), and is
missing a precomputed-vs-periodic window distinction; recommend
keeping the dual representation but naming the bridge
(`PackComplex`/`SplitComplex`), exporting `IsPow2`/`NextPow2`,
adopting the tracker pattern for a future `Biquad`, and
adding `*Into` zero-alloc twins of every allocating
`spectrogram.*` function.

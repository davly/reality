# 133 | signal-sota

**Agent:** 133 of 400
**Topic:** signal: compare with scipy.signal, librosa, GNU Radio, EigenSPL (+ FFTW/pocketfft/KFR, DDSP)
**Scope:** SOTA library landscape vs reality `C:/limitless/foundation/reality/signal/`
**Date:** 2026-05-08

## Brief

Reality `signal/` has 12 entry points (~470 LOC). Agent 131 audited numerics;
132 inventoried missing primitives. This report does **not** repeat either.
For each SOTA library: (1) headline algorithm, (2) one engineering trick
worth stealing, (3) zero-dep portability verdict.

---

## scipy.signal

Reference Python DSP toolbox. ~80 public functions in 2026 (SciPy 1.17):
filter design (`butter`, `cheby1/2`, `ellip`, `bessel`, `firwin`, `remez`),
application (`lfilter`, `sosfilt`, `filtfilt`, `sosfiltfilt`), spectral
(`welch`, `periodogram`, `stft`, `coherence`), wavelets, peak finding, Hilbert.

### (1) Headline: SOS-cascade everywhere — `output='sos'` as default-of-the-future

Transfer-function `(b, a)` form is **numerically broken** for IIR filters at
order ≥ 4 (roots → polynomial coefficients is an ill-conditioned op; `lfilter`
compounds it through feedback). Fix: **second-order sections (SOS)** — factor
into N/2 biquads, run each as Direct-Form-II Transposed with per-section state.
`sosfilt`/`sosfiltfilt` are now recommended over `lfilter`/`filtfilt`. SciPy's
docs explicitly steer users to `output='sos'`.

Lesson for reality: **never expose `(b, a)` for order ≥ 3**. Design output is
`[]Biquad`, period. Makes the numerical cliff in 132 §1.4-1.5 impossible
by construction.

### (2) Engineering trick: zero-phase via forward-backward + edge padding

`filtfilt`/`sosfiltfilt` runs filter forward then backward, doubling order,
cancelling phase exactly. Endpoint trick: reflectively pad `padlen` samples
on each side (default `3·max(len(a),len(b))-1`), then trim. Without padding,
leading edge sees a step transient from `y[-1]=0`. With reflective extension
plus Gustafsson's IIR initial-conditions method (`method='gust'`), edge
transients vanish. Difference between textbook and production.

### (3) Portability for reality

**Direct port: yes, with size discipline.** Needs poly root-finding (in
`optim`), bilinear transform (trivial), analog-prototype pole tables
(Butterworth/Cheby closed-form; Elliptic needs Jacobi sn/cn — Antoniou §7).
Est. 600 LOC for Butterworth + Chebyshev + bilinear + biquad-DF2T; +400 for
FIR (window, Parks-McClellan). Welch ~80 LOC. **Don't port:** `lti`
continuous (overlaps `control/`); wavelets/peak-finding (separate packages).
SciPy is *the* cross-language reference; Python-generate goldens, tolerance
~1e-12 for coefficients, ~1e-9 for filtered output.

---

## librosa

Audio-focused on a clean DSP foundation. `librosa.stft`, `istft`,
`feature.melspectrogram`, MFCC, chroma, CQT, VQT, beat tracking, onset,
pYIN pitch, time-stretch (phase vocoder), pitch-shift, harmonic-percussive
separation.

### (1) Headline: pYIN — probabilistic YIN with HMM Viterbi voicing

pYIN (Mauch & Dixon 2014) displaced YIN as the audio-pitch reference. Step 1:
YIN difference function with a **bank** of absolute thresholds, each giving a
candidate F0 and probability. Step 2: HMM with states = (pitch bin × voiced/
unvoiced) decoded by Viterbi → globally-optimal pitch *track* with explicit
voicing flags, not per-frame greedy. Same Parks-McClellan-style shift (local →
global) applied to pitch tracking. Reality's `audio/pitch/` has YIN but no pYIN.

### (2) Engineering trick: Numba-JIT inner loops only

librosa is "almost pure NumPy" except a handful of `@numba.jit` inner kernels:
STFT framing, ISTFT overlap-add, pitch-track DP, peak-pick. The *rest* —
feature extraction, filter banks, normalization — is pure NumPy and stays
that way. Discipline: **JIT only cache-line-bound loops; leave matrix-shaped
operations to BLAS via NumPy.**

For reality (no JIT, AOT compile), the analog: **keep hot loops as scalar Go
the compiler vectorises; matrix ops to `linalg`.** `signal/` already does
this. Don't reach for `cgo`-FFTW the moment performance becomes an issue.

### (3) Portability for reality

**Selective port.** STFT/ISTFT (in `audio/spectrogram`, should re-home to
`signal/` per 132). Mel/MFCC/chroma/CQT correctly in `audio/`. pYIN in
`audio/`. Phase vocoder (time-stretch, pitch-shift) is generic — should
live in `signal/`. **Don't port:** `librosa.effects` convenience layer.
librosa STFT is the de-facto cross-tool standard (matplotlib, torchaudio,
essentia match bit-for-bit) — pin reality STFT goldens against it.

---

## GNU Radio (+ VOLK)

Real-time SDR flowgraph platform. ~3000 blocks; lock-free SPSC streaming;
GR4 (RC1 March 2026) reworks scheduler for compile-time pipeline composition.

### (1) Headline: VOLK — runtime-dispatched SIMD kernel library

VOLK = Vector-Optimised Library of Kernels. ~150 kernels (complex multiply,
mag-squared, dot product, FFT helpers, log/exp). Each ships in **multiple
proto-implementations**: `generic`, `a_sse{,2,3}`, `a_avx{,2,512f}`, `a_neon`,
`u_*` unaligned variants. `volk_profile` benchmarks each on the actual CPU
at install, writes `volk_config`; runtime dispatches to best proto. One
`#include "volk/volk.h"`: SSE on Westmere, AVX2 on Haswell, AVX-512 on Ice
Lake, NEON on M2, no recompile.

This is the *opposite* design choice from reality's "pure portable Go, let
compiler vectorise". Trade is peak throughput (VOLK 4-8× faster) vs zero
deployment friction.

### (2) Engineering trick: lock-free SPSC ring buffers via mmap double-mapping

GNU Radio circular buffers map the same physical pages **twice contiguously**
(`mmap(addr, 2*size, ...)` then `mmap(addr+size, size, MAP_FIXED, fd, 0)`).
Single linear pointer wraps without modulo. Producer can `memcpy` `size`
bytes from anywhere in `[0, size)` and read side sees wrap-around free.
Eliminates branchy "did I cross boundary" code in every block.

Reality has no streaming abstraction (math library, not runtime). Worth
stealing for any future streaming convolver / filter-chain / STFT processor
in `audio/`.

### (3) Portability for reality

**Don't port the runtime.** GNU Radio is a service architecture; reality is
a library. Blocks themselves (filter design, modulators, channels) overlap
scipy.signal. **Do steal the SDR algorithm library:** PLL, AGC, Costas loop,
timing-recovery (Gardner / Mueller-Müller), polyphase channeliser. Missing
from both scipy.signal and reality; some (PLL, AGC) overlap `control/`.
~200 LOC each. VOLK kernels not useful for cross-language validation
(exist to be fast, not reference); for SDR algos use MATLAB Comms Toolbox
or GNU Radio reference Python.

---

## EigenSPL / disiple / eDSP (C++ template DSP)

No widely-known library literally called "EigenSPL" — assume the brief refers
to the Eigen-based template DSP family: `disiple` (marton78), `eDSP`
(mohabouje), `DSPFilters` (Vinnie Falco — spiritual ancestor). All three
encode filter specs at the type level: `Filter<Butterworth<6>, LowPass>` is
a distinct type from `Filter<Chebyshev1<6>, LowPass>`, allowing full inline
+ SIMD-vectorisation by the compiler.

### (1) Headline: type-level filter design via expression templates

`disiple` cascades biquads at the **type system**, not at runtime. A 6th-order
Butterworth lowpass is `BiquadCascade<3, double>` parameterised on coefficients
computed at construction. Eigen expression-templates fuse the biquad runner
with surrounding arithmetic; `y = filter(x) + 0.5*lookup` becomes a single
fused vectorised loop with no temporaries. Compiler does what librosa's JIT
does at runtime.

Deeper idea: **filter design is metadata, not state.** Once designed, biquad
coefficients are immutable; runner is pure data flow. Type system catches
"forgot to design" at compile time.

Go has no expression templates and limited generics: literal port impossible.
Go-idiomatic equivalent is the **constructor pattern**: `f :=
signal.Butterworth(order, cutoff, sr)` returns immutable `[]Biquad`;
`f.Process(in, out)` is the runner. "Forgot to design" check moves from
compile to nil-pointer panic — acceptable.

### (2) Engineering trick: state-space biquad form for SIMD

DF-II-T biquad has feedback that *cannot* vectorise across samples (sample
n+1 depends on n state). disiple uses **state-space form** with 2×2
state-transition matrix `A`, input vector `B`, output row `C`, feedthrough
`D`: `x[n+1]=A·x[n]+B·u[n]; y[n]=C·x[n]+D·u[n]`. Eigen matrix multiply is
SIMD-vectorised; unrolling 4 samples gives `A^4`-multiplication of state
with parallel input contribution (cost: `A^k` for k=1..unroll computed
once). Same trick in CMSIS-DSP and Faust: **transform serial recurrence
into parallel matrix powers.** Saves 3-4× over scalar DF-II-T at order ≥ 4.

For reality, overkill at first — start with scalar DF-II-T. Future
optimisation worth documenting.

### (3) Portability for reality

**Don't port template machinery.** Go lacks expression templates; even with
generics, API would be hostile (`signal.Filter[Butterworth[Order6],
LowPass]{...}`).

**Do port coefficient-design formulae.** disiple is meticulously sourced
(Antoniou, Orfanidis, Holters, Zölzer); readable C++. Use as reference for
golden-file validation when scipy.signal disagrees with MATLAB (happens for
elliptic at low order — different prototype-pole conventions).

---

## pocketfft / FFTW / KFR (FFT specialists)

- **FFTW** (MIT/GPL): the reference. Plan-based — at init, *measures* dozens
  of paths (radix-2/3/5/7, mixed-radix, split-radix, Rader, Bluestein) and
  codegens the optimal. Wins arbitrary/large N; loses on small fixed N where
  plan overhead dominates.
- **pocketfft** (BSD-3, scipy.fft backend since 1.4): ~2k LOC C++ header.
  Single-file, no plan, arbitrary N via internal Bluestein. Vectorised with
  `__attribute__((vector_size(N)))` — portable to GCC/Clang (not MSVC).
  Within 5-15% of FFTW3 for typical audio sizes.
- **KFR** (GPL/commercial dual): C++17, hand-tuned SSE/AVX/NEON. Beats FFTW
  on x86 for small/medium N (≤ 8192). Trades portability for throughput.

### (1) Headline: FFTW's "wisdom" — runtime cost amortised across runs

FFTW writes the chosen plan to a `wisdom` file. Next run:
`fftw_import_wisdom` reloads, skips measurement. Effectively a JIT-cached
plan, persistent across processes. Insight: **FFT performance on a given
(N, type, alignment, in-place) tuple is deterministic per CPU**, so cache
forever. SciPy exposes via `workers=-1` and `plan` kwargs.

### (2) Engineering trick: pocketfft's twiddle-table per N

pocketfft computes twiddle tables **once per N**, cached in `fftblue` indexed
by length. For radix-2 stages it generates twiddles by **direct trig per
butterfly** (no recurrence drift — see 131 §FFT). Right trade for a
precision-first library: ~2× memory cost vs in-loop recurrence buys back
ulp-level error growth, matching IEEE 754 spec to `O(log N · ε)`.

**Directly applicable to reality.** 131 found twiddle recurrence drift
significant at N ≥ 2^16. Switching to twiddle table per N (lazily computed,
cached in `sync.Map[int][]complex128`) is ~30 LOC, fixes the precision
cliff. Cost: ~16N bytes per distinct N. For typical audio (N ∈
{512,1024,2048,4096}), cache is ~50 KB total.

### (3) Portability for reality

- **FFTW: no.** GPL incompatible with MIT.
- **pocketfft: ideologically yes, practically not yet needed.** BSD-3,
  header-only, transcribable to Go in a weekend (~600 LOC). For reality's
  target sizes, existing radix-2 + twiddle-table fix is plenty. Use
  pocketfft's Bluestein as cleanest reference when 132 §1.3 is filled.
- **KFR: no.** Commercial; C++ template-heavy.

What to steal architecturally:

1. **Twiddle tables, not recurrences** (pocketfft).
2. **Lazy plan cache keyed by N** (FFTW's wisdom, simple case).
3. **Bluestein as universal arbitrary-N fallback** (all three).
4. **Real-input FFT via Hermitian-pack** (all three; 132 §1.1).

All public-domain algorithms; pure Go transcriptions.

---

## Differentiable DSP (DDSP)

Engel et al. ICLR 2020 (`arxiv:2001.04643`); active body of work through
2025-26. Shift: every classical DSP primitive (oscillator bank, harmonic+
noise synth, FIR reverb, filter) re-expressed as a **differentiable
function** so it sits inside a PyTorch/JAX autodiff graph and trains
end-to-end.

### (1) Headline: harmonic+noise synth as learnable inductive bias

DDSP autoencoder predicts (a) per-frame f0, (b) loudness, (c) harmonic
amplitudes, (d) noise filter taps. Decoder *is* a sinusoidal additive
synth + filtered-noise generator — a 1970s vocoder. Training the encoder
backprops gradients **through the synth** (just sums and multiplies). SOTA
timbre transfer at a fraction of the parameters of black-box waveform
models (WaveNet, SampleRNN), because the synth provides the inductive
bias for free.

Lesson: **DSP primitives are not legacy. They are the right inductive
prior**; wrapping them so gradients flow is the small price. Every
primitive in `signal/` should be **autodiff-compatible** in principle:
pure functions over `float64`, no hidden state, no random sampling.
Reality's existing API (slice in, slice out, no state) is already
DDSP-shaped.

### (2) Engineering trick: time-varying linear filter via frame-wise FFT mul

Pure-IIR with learnable poles is differentiable but unstable during
training (poles wander outside unit circle). DDSP sidesteps stability by
predicting a **time-varying impulse response per frame**, applied via
overlap-add convolution in the frequency domain. Frame-wise FFT × predicted
filter spectrum × IFFT × overlap-add. No feedback, no stability worry,
gradients well-conditioned. Trade: O(N log N) per frame vs O(N) biquad,
frame-rate (not sample-rate) updates.

For reality: argues for shipping **block-convolution helpers** alongside
biquad runners. Same primitive (FFT + multiply + IFFT) used for fast
convolution today becomes differentiable filtering tomorrow.

### (3) Portability for reality

- **The framework: no.** DDSP-the-codebase is TF/JAX research.
- **The discipline: yes, reality is already aligned.** Pure functions, no
  hidden state, slice in/out, no allocations, no randomness. Gap is not
  architecture but the numerical tightness called out in 131 (twiddle
  drift, NaN propagation, biquad quantization absent because biquads are
  absent).

Bridge: **golden-file vectors for gradients**, not just outputs. Future
`signal/` extension could ship JSON of form `{input, output,
dOutput_dInput}`, validated against finite-difference reference. Lets
reality plug into a future autodiff package without re-validation. Not
urgent; architectural note.

---

## Cross-cutting findings

### 1. Filter design lives at the *coefficient* layer, not runner layer

scipy.signal, disiple, librosa all separate **design** (offline, arbitrary
precision, runs once) from **runtime** (online, float64, per-sample).
Reality's biquad gap (132 §1.4-1.5) should mirror: `Butterworth(order,
cutoff, sr) []Biquad` produces immutable cascades; `BiquadCascade` runner
is hot path. **Never combine.**

### 2. SOS is the only correct IIR storage format

scipy.signal's migration to `output='sos'` is the single largest external
lesson. Reality should not even *expose* `(b, a)` tuples for orders ≥ 3.

### 3. Twiddle tables, not recurrences

pocketfft's per-butterfly direct-trig (or precomputed table) is the correct
precision-first impl. Reality's recurrence (131 §FFT) should be replaced;
~16N bytes per cached N, fraction of signal sizes.

### 4. Bluestein as universal arbitrary-N escape

All four (scipy, pocketfft, FFTW, KFR) use Bluestein chirp-z internally for
non-pow-2 sizes. Reality `FFT` panics on non-pow-2; consumers work around
with manual zero-padding (silently biases spectra). 132 §1.3 covers; SOTA
confirms *every serious library does it*.

### 5. SIMD dispatch is GNU Radio's lane, not reality's

Reality's "pure portable Go, let compiler vectorise" stance is correct for
MIT zero-dep math library. Will be 2-4× slower than VOLK/KFR on x86 hot
paths — fine. Consumers needing peak throughput link `cgo` bindings to
FFTW/VOLK separately. *Math* in reality is the artifact; *speed* is the
consumer's problem.

### 6. DDSP-readiness is free if numerics are right

Reality's pure-function, slice-in/out, no-allocation discipline *is*
DDSP-readiness. Gap is not architecture but 131's numerical tightness +
132's missing primitives.

---

## Concrete recommendations (SOTA-derived)

By impact-per-LOC:

1. **Twiddle-table-per-N FFT** (pocketfft). ~30 LOC. Closes 131's precision
   cliff at N ≥ 2^16. `sync.Map[int][]complex128` cache.
2. **Always-SOS API for IIR** (scipy.signal). When 132 §1.4 lands:
   `Butterworth(...) []Biquad`, never `(b, a)`.
3. **Bluestein for arbitrary N** (universal SOTA). 132 §1.3.
4. **filtfilt with reflective edge + Gustafsson** (scipy.signal). Once
   biquads exist, ~80 LOC.
5. **Block-convolution helper** (DDSP / FFT-convolve). Single primitive
   for fast convolution today + differentiable filtering tomorrow. ~50 LOC.
6. **RFFT with Hermitian pack** (all three FFT libs). 132 §1.1; 2× audio
   wall-time at zero precision cost.
7. **Document the SIMD non-promise.** Package-level note: portable Go,
   ~2-4× slower than VOLK/KFR/FFTW by design.
8. **Pin SciPy/librosa-generated golden vectors** for FFT, Welch, STFT,
   filter design. SciPy is cross-language reference; librosa is cross-tool
   audio reference.

None add dependencies. All preserve MIT zero-dep posture.

---

## Files referenced

`C:/limitless/foundation/reality/signal/{fft.go,filter.go,window.go}`
`C:/limitless/foundation/reality/reviews/overnight-400/agents/131-signal-numerics.md`
`C:/limitless/foundation/reality/reviews/overnight-400/agents/132-signal-missing.md`

External (read-only references, not dependencies): scipy.signal v1.17
(filter design, `sosfilt`, `sosfiltfilt`, Welch, STFT); librosa (`stft`,
`pyin`, mel/MFCC/CQT, phase vocoder); GNU Radio (VOLK SIMD; SPSC mmap
buffers; GR4 RC1 SimdFFT March 2026); disiple/marton78 (Eigen-template
DSP, biquad cascades, SOS, state-space); FFTW (planning+wisdom);
pocketfft (portable arbitrary-N); KFR (SIMD templates); DDSP (Engel
et al. 2020 + Frontiers 2023 review).

## Two-line summary

scipy.signal teaches always-SOS IIR storage and reflective-edge filtfilt;
pocketfft teaches twiddle-table-per-N over recurrence; GNU Radio's VOLK
proves SIMD-dispatch is out-of-scope for a zero-dep Go math library; DDSP
confirms reality's pure-function slice-in/out API is already gradient-ready
once 131's numerical cliffs and 132's missing primitives (biquad SOS, RFFT,
Bluestein, Welch, Hilbert) are filled in.

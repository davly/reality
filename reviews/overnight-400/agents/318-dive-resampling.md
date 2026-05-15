# 318 — dive-resampling (Polyphase / Sinc / Lagrange / Farrow / Smith arbitrary-rate audit)

## Headline
Reality has **zero sample-rate-conversion machinery** — no `Resample`, `Polyphase`, `Decimate`, `Upsample`/`Downsample`, `FractionalDelay`, `Farrow`, `Lagrange`, no windowed-sinc kernel, not even `Sinc(x)` itself; the closest existing piece is `signal.Convolve` (`signal/filter.go:19`), so a resampler must be built from first principles. Day-1 PR: `signal/resample.go` shipping integer up/down (T0) and rational L/M polyphase (T1) ≈ 350 LOC + 30 golden vectors; defer Smith arbitrary-rate (T5, ~300 LOC), Farrow (T4, ~150 LOC), and Lagrange-fractional-delay (T3, ~80 LOC) to Day-2 once `signal/fir.go` (slot 316) lands a Kaiser-windowed-sinc lowpass to source the prototype taps.

## Findings

### What exists in reality (full inventory of resampling-adjacent code)
- `signal/filter.go:19` — `Convolve(signal, kernel, out)`. The execution kernel a polyphase filter would call (or, faster, a per-phase loop that skips the L−1 zero-pad multiplies by reading taps modulo `L`).
- `signal/filter.go:54` — `MovingAverage(signal, windowSize, out)`. Equivalent to a length-`windowSize` rectangular FIR; could decimate with a `[::M]` stride after, but rectangular has −13 dB sidelobes — useless as an anti-alias filter.
- `signal/filter.go:97` — `ExponentialMovingAverage`. 1-pole IIR, not LP-FIR; group delay is frequency-dependent — wrong for ASR / music decimation.
- `signal/filter.go:130` — `MedianFilter`. Non-linear, not relevant to bandlimited resampling.
- `signal/window.go:15` `HannWindow`, `:44` `HammingWindow`, `:76` `BlackmanWindow`. Three windows; **no Kaiser** (slot 317 flagged), so the windowed-sinc lowpass that polyphase / arbitrary-rate need cannot be built today with the parametric stopband control libsamplerate / SoX expect.
- `signal/fft.go:49,101` — `FFT` / `IFFT`. Unused by polyphase but required by FFT-based resamplers (Crochiere-Rabiner §3.3).
- `signal/fft.go:167` — `FFTFrequencies(n, sampleRate, out)`. Touches `sampleRate` but never **converts** it; nothing in `signal/` knows about `f_in → f_out`.
- `optim/interpolate.go:18,44` — `LinearInterpolate`, `CubicSplineNatural`. Scalar, off-grid; do not compose into a streaming resampler (no anti-alias, no fractional-delay management).
- `geometry/curves.go:68` — `CatmullRom(p0,p1,p2,p3,t)`. Could power a 4-tap fractional-delay shim, but has the Catmull-Rom magic-`τ=0.5` baked in — not a resampler.
- `audio/spectrogram/visualise.go:102-135` — claims "bilinearly resampled" but does nearest-neighbor (slot 312 bug). Unrelated to audio-rate but symptomatic of "no canonical resample primitive".

### What is missing (the actual gap)
Grep across **all** of reality for `Resample|Polyphase|Decimate|Upsample|Downsample|FractionalDelay|Farrow|ArbitraryRate|SampleRate|SRC|sinc|Lanczos|Lagrange` (case-insensitive) → **0 hits in production code**. Only matches: review docs, the `Lanczos` approximation comment in `prob/mathutil.go:22` (gamma function — different Lanczos), `audio/separation/wiener.go:40` (Wiener interpolation — different domain). Specifically absent:

1. **Sinc kernel** — `Sinc(x) = sin(πx) / (πx)` with `Sinc(0) = 1`. The atom of all bandlimited resampling. **One function, ~10 LOC**, but it does not exist anywhere in reality. Without it, neither windowed-sinc FIR design (slot 316) nor any resampler can be built. This is a **shared dependency** with slots 312 (Lanczos image kernel) and 316 (windowed-sinc FIR).
2. **Integer upsample-by-L** (zero-stuffing + LP). Insert L−1 zeros between samples → spectrum has L copies → low-pass at `f_s_in / 2` (i.e. `π/L` in normalised radians) with gain `L` (to compensate energy loss). Crochiere-Rabiner 1983 eq. 3.1.
3. **Integer downsample-by-M** (LP + decimate). Anti-alias low-pass at `f_s_out / 2 = π/M` *first* → keep every Mth sample. Skipping the LP causes aliasing of the `[π/M, π]` band into baseband. Crochiere-Rabiner eq. 3.10.
4. **Polyphase decomposition** (Vaidyanathan 1990, *Multirate Systems and Filter Banks* §4.3). The single most important resampling identity: the `L` zero-stuffed multiplications in interpolation are *redundant* (most products are 0). Decompose the prototype filter `h[n]` into `L` polyphase components `e_p[n] = h[p + nL]` for `p = 0..L−1`; the upsample-then-LP is equivalent to filtering with each `e_p` independently and interleaving the outputs. Computationally: **L× speedup** on upsample, M× on downsample. The de-facto rational-rate algorithm.
5. **Rational L/M polyphase** (cascade upsample-by-L then downsample-by-M). Same prototype filter `h[n]` of length `N = K·L` (K taps per polyphase phase) does *both* the interpolation anti-image and the decimation anti-alias if its cutoff is `min(π/L, π/M)`. The library workhorse: `signal.Resample(in, L, M, out)`. **Production note**: `gcd(L,M)` must be reduced first (48000→16000 → L/M = 1/3, not 16000/48000).
6. **Windowed-sinc arbitrary-rate** (Smith JOS, *Digital Audio Resampling Home Page* CCRMA, the libsamplerate / SoX HQ math). Pre-compute a long sinc-table at oversampling factor `Q` (typ. 512), then for each output sample at fractional input phase `φ`, interpolate the table at index `φ·Q` (linear or quadratic table-lookup). Anti-alias by **scaling the kernel width** by `f_in/f_out` when downsampling. Handles **arbitrary** real ratios — no rational reduction needed — at the cost of one extra interpolation per output. ~300 LOC including the table builder.
7. **Lagrange fractional-delay FIR** (Laakso, Välimäki, Karjalainen, Laine 1996, *Splitting the Unit Delay*). Order-`P` Lagrange polynomial at fractional position `D`: `h_L[n] = Π_{k=0,k≠n}^{P} (D − k)/(n − k)` for `n = 0..P`. Cubic (P=3) is the canonical low-cost fractional-delay; gives ≈70 dB SNR at `D ≈ 0.5` and is **far cheaper** than sinc when only a fractional delay (not arbitrary rate) is needed. ~80 LOC.
8. **Farrow filter** (C. W. Farrow 1988, *A continuously variable digital delay element*, ISCAS). Express the polynomial coefficients of a fractional-delay FIR as a 2-D Farrow structure `H(z, D) = Σ_m C_m(z) D^m` where each `C_m(z)` is a fixed FIR. Output is `Σ_m C_m(x[n]) D[n]^m` — i.e. **`D` becomes a runtime input**. Allows per-sample-varying fractional delay (vibrato, time-warp) at the cost of `(M+1)` parallel FIRs. The textbook real-time arbitrary-rate algorithm; underlies async sample-rate converters in DACs / clock-domain crossings. ~150 LOC.
9. **Stochastic / non-uniform resampling** — given input samples at irregular timestamps `t_i`, reconstruct on a uniform grid via Yen 1956 or Lomb-Scargle inverse. Frontier; defer.
10. **Half-band-cascade decimator** — for M = 2^k, decimating in `k` halves with successive halfband filters (Mintzer 1982) is the gold-standard cheapest decimator. Each halfband has ≈half its taps zero. Composes T1 once shipped. ~60 LOC follow-on.
11. **CIC (cascaded integrator-comb) decimator** (Hogenauer 1981). For very large M (rate-of-thousands, e.g. ΣΔ ADC front-ends). Multiplier-free. Out of scope for audio (which is M ≤ 6 typically) but worth noting; ~40 LOC.

### Cross-package fanout (what this gap blocks)
- **Pistachio audio engine** — needs sample-rate conversion between mic native rate (16/24/48 kHz), engine internal rate, and DAC output rate. Today: **no path**. Pistachio cannot resample without leaving reality (vendoring `r8brain` or `libsamplerate` would violate the zero-deps rule).
- **ASR / speech preprocessing** — almost every modern ASR model (Whisper, wav2vec2, RNN-T) ingests 16 kHz mono. Capture devices give 48 kHz. The **3:1 polyphase decimator** is the single most-called audio op in any real-world speech stack. Without it, every consumer reinvents a per-app resampler with un-validated anti-alias.
- **Telephony / VoIP codecs** — G.711 / G.722 / Opus narrow-band run at 8 kHz; wide-band Opus at 16/48 kHz. 8↔48 (1/6, 6/1) and 16↔48 (1/3, 3/1) are the two ratios needed. Both are rational-L/M, both want T1.
- **Music / DAW / Pistachio asset pipeline** — sample-pack assets ship at 44.1 kHz, engine usually runs at 48 kHz. **44100/48000 = 147/160** — coprime, large-L large-M. Polyphase tap count grows as `K·max(L,M) = K·160`; this is exactly where Smith arbitrary-rate (T5) wins by sidestepping rational reduction.
- **Time-stretching / pitch-shifting** (slot 244 / 264 if exists; absent today) — phase-vocoder requires high-quality resampling on the synthesis side. Without T1+T5, can't be implemented.
- **`audio/onset/superflux.go`** — operates on STFT frames, doesn't directly need resampling, but consumers feeding it 48 kHz audio destined for 16 kHz acoustic models need the 3:1 decimation step **before** STFT.
- **`audio/melscale.go`, `audio/mfcc.go`** — same: speech features assume 16 kHz, mics give 48 kHz.
- **`acoustics/`** (slot 166 synergy) — RT60 / Sabine analyses ingest IR recordings; rate-conversion to a canonical analysis rate is a prereq.

### R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities (saturated)
The audit identifies **at least four** cross-validation pins that ship the SAME L/M signal through different code paths and require ≤1e-12 agreement:

| # | Path A | Path B | Pin tolerance | LOC |
|---|---|---|---|---|
| (a) | upsample-by-L → downsample-by-L | identity | exact (zero pad-then-pick is bit-exact when LP is the same and the LP is symmetric — modulo edge effects) | ~20 |
| (b) | rational polyphase L/M (T1) | direct-form: zero-stuff×L → convolve(h) → stride-M (T0 cascade) | 1e-12 (same arithmetic, different traversal) | ~40 |
| (c) | windowed-sinc arbitrary-rate at integer ratio L/1 (T5) | polyphase L/1 (T1) | 1e-9 (table interp introduces ε) | ~40 |
| (d) | bandlimited cosine `cos(2π·f₀·t)` resampled in→out, FFT'd | analytic spectrum: single bin at `f₀`, all others zero (regression) | ≤−80 dB stopband | ~30 |
| (e) | round-trip up-3 → down-3 (or any L/M then M/L with **shared** prototype) | identity (within filter-pre-ringing edge effects) | ≤1e-9 mid-buffer | ~30 |

Pin (a) and (b) saturate **Cross-Validation R-3/3** by themselves (two independent algorithms agreeing). Pin (d) is a *regression* against analytic ground truth — the strongest form. Pin (e) is the audio engineer's standard sanity check.

### Numerical / precision notes
- **Anti-alias filter spec**: typical pro-audio resampler aims for **−96 dB stopband** (16-bit ENOB) or **−144 dB** (24-bit ENOB). Kaiser β = 12 / 18 respectively (Kaiser-Hellier from slot 316). Tap count for 48→16 at −96 dB stopband and transition `Δω = 0.1·π`: `N = (96 − 8)/(2.285·0.1·π) ≈ 122` taps; with L=1, M=3, polyphase has K=41 per phase (one phase per output sample). At 16 kHz output → **41 muls per sample** — trivially realtime even on embedded.
- **Integer-only fast paths**: rational L=M=1 → identity copy. L=1, M=integer → decimator only. L=integer, M=1 → interpolator only. The polyphase code should fast-path these to spare the LP-design step.
- **Ratio reduction**: **always** divide `L,M` by `gcd(L,M)` before allocating taps. 44100/48000 has `gcd=300` → reduced ratio `147/160`, prototype length proportional to `K·160`, not `K·48000`.
- **Edge handling**: zero-pad input by `(N−1)/2` on each side, or accept truncated output near boundaries. Streaming resamplers retain the last `N−1` input samples as state.
- **Floating-point precision**: all arithmetic is `float64` per CLAUDE.md; golden vectors generated via `math/big` at 256-bit precision.

## Concrete recommendations

### Day-1 PR — `signal/resample.go` (~350 LOC + 30 golden vectors)

**Depends on**: slot 316 PR (Kaiser-windowed-sinc LP design) for the prototype-filter helper. If 316 has not landed, this PR ships its own minimal `kaiserSincLowpass(N, fc, beta, out)` private helper (~40 LOC).

**Public API**:
```go
// Sinc returns sin(pi*x)/(pi*x) with Sinc(0)=1. Required atom; live in signal/window.go.
func Sinc(x float64) float64

// Upsample inserts L-1 zeros between input samples, then low-pass filters
// to remove spectral images at k*pi/L for k=1..L-1. Output length = len(in)*L.
// out must have length >= len(in)*L. Panics on mismatch.
func Upsample(in []float64, L int, out []float64)

// Downsample low-pass filters to fc=pi/M to prevent aliasing,
// then keeps every Mth sample. Output length = len(in)/M.
func Downsample(in []float64, M int, out []float64)

// Resample converts L/M sample-rate via polyphase decomposition.
// Internally reduces by gcd(L,M). Allocates one prototype filter
// of length ~ K*max(L,M); reuses output slot zero-alloc on hot path
// once amortised. Output length = len(in)*L/M (rounded down).
func Resample(in []float64, L, M int, out []float64)
```

**Cross-validation pins** (write all five from §"R-MUTUAL-CROSS-VALIDATION"):
- `TestResampleRoundTripIdentity`: `Downsample(Upsample(x, 3), 3) ≈ x` mid-buffer ≤ 1e-9.
- `TestResamplePolyphaseEqualsDirectForm`: `Resample(x, 3, 2)` ≡ `Downsample(Upsample(x, 3), 2)` ≤ 1e-12.
- `TestResamplePreservesBandlimited`: synthesise `cos(2π·f₀·t)` at f₀ < min(f_in,f_out)/2.5; resample; FFT; verify single non-zero bin at `f₀` with ≥80 dB SNR.
- `TestUpsampleSpectralImageRejection`: synthesise `cos(2π·f₀·t)` at `f₀ = 0.4·f_in`; upsample 4×; FFT; verify images at `(L−1)·f_in ± f₀` are ≤ −80 dB.
- `TestResampleGCDReduction`: `Resample(x, 6, 4)` == `Resample(x, 3, 2)` exactly (assert internal `gcd` reduction occurred).

**Golden file**: 30 vectors covering integer-up (L=2,3,4,8), integer-down (M=2,3,4,8), rational (3/2, 4/3, 7/5, 147/160), edges (L=1,M=1 identity; very-short input N=8; long N=4096), DC pass-through, Nyquist boundary cases.

### Day-2 PR — Smith arbitrary-rate + Lagrange (~400 LOC)
- **`SmithResample(in, ratio float64, out []float64)`** — windowed-sinc table-lookup at oversampling factor `Q=512`, linear table interpolation. Handles `ratio ∈ (0.1, 10)` for arbitrary real rates. Cite Smith JOS *Digital Audio Resampling Home Page*. Cross-validate against Day-1 `Resample` at integer ratios.
- **`LagrangeFractionalDelay(in []float64, delay float64, order int, out []float64)`** — Laakso et al. 1996 closed-form; orders 1/2/3/4. ~80 LOC. Cross-validate vs SmithResample at small fractional delays.

### Day-3 PR — Farrow + halfband cascade (~250 LOC)
- **`FarrowResampler` struct** — Farrow 1988 polynomial structure for runtime-varying fractional delay. ~150 LOC. Use case: drift-tolerant async-rate conversion, vibrato.
- **`HalfbandDecimate(in, cascadeDepth int, out)`** — Mintzer 1982 halfband cascade for `M = 2^k`. Saves ~50% multiplications vs generic polyphase. ~60 LOC.

### Day-4 PR — non-uniform / stochastic
- **`NonUniformResample(times, values, outRate)`** — Yen 1956 / Lomb-Scargle inverse. Frontier; gate on consumer demand from sensor-fusion / asset pipeline.

### Process notes
- **Prototype filter cache**: `Resample(in, L, M, out)` should memoise the filter taps keyed by `(L, M, β, K)`. Avoids redesigning the filter every call. ~20 LOC, big real-world win.
- **Streaming variant**: ship `ResampleStream` that retains the last `(N−1)` input samples as state; required for chunked audio (Pistachio's frame-by-frame ingest).
- **Zero-allocation hot path**: per CLAUDE.md rule 3, the inner polyphase loop must not allocate. Caller provides `out`; filter taps live on the receiver struct; phase index is an `int`.
- **No external deps**: do not `import "github.com/r8brain"`. Reimplement from Crochiere-Rabiner / Vaidyanathan / Smith. Per CLAUDE.md rule 6.
- **Cross-language golden files**: 256-bit `math/big` reference; all 30 vectors emitted as JSON with per-function tolerance (1e-12 for polyphase ≡ direct-form, 1e-9 for round-trip, 1e-7 for Smith table-interp). Per CLAUDE.md rule 1.

### Cheapest path to "Pistachio can resample"
T0 alone (`Upsample` + `Downsample` integer-only) handles the **3 most common audio cases**: 48→16 ASR (M=3), 48→24 (M=2), 8→48 telephony (L=6). That's a **~150 LOC** PR that unblocks 80% of consumers immediately, with T1 polyphase a follow-up that adds 44.1↔48 (147/160) and the algorithmic-elegance / efficiency story.

## Sources

### Repo files cited
- `C:/limitless/foundation/reality/signal/filter.go:19` — `Convolve` (FIR execution kernel; polyphase will call this or its phase-strided variant).
- `C:/limitless/foundation/reality/signal/filter.go:54` — `MovingAverage` (rect-FIR, unsuitable as anti-alias).
- `C:/limitless/foundation/reality/signal/filter.go:97` — `ExponentialMovingAverage` (IIR, wrong group delay).
- `C:/limitless/foundation/reality/signal/window.go:15,44,76` — Hann/Hamming/Blackman; **no Kaiser** (slot 317).
- `C:/limitless/foundation/reality/signal/fft.go:49,101,167` — FFT/IFFT/FFTFrequencies (sample-rate awareness only at FFT level).
- `C:/limitless/foundation/reality/optim/interpolate.go:18,44` — scalar interp; doesn't compose into a streaming resampler.
- `C:/limitless/foundation/reality/audio/spectrogram/visualise.go:34,102-135` — false-bilinear bug (slot 312).
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/007-audio-missing.md:43` — corroborates "no polyphase/sinc/Farrow" gap.
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/316-dive-fir-design.md` — slot 316 ships the Kaiser-windowed-sinc LP this PR depends on.
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/315-dive-iir-design.md` — slot 315 IIR design (orthogonal but companion).
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/312-dive-bilinear-bias.md` — flags the missing `Sinc(x)` atom (shared dep).

### External canon
- Crochiere, R. E. & Rabiner, L. R. (1983). *Multirate Digital Signal Processing*. Prentice-Hall. — The textbook on integer-L / integer-M / rational-L/M resampling. §3.1 zero-stuff interpolator, §3.2 decimator, §3.3 polyphase, §3.4 efficient FFT-based forms.
- Vaidyanathan, P. P. (1990, 1993). *Multirate Systems and Filter Banks*. Prentice-Hall. §4.3 polyphase identity, §4.4 noble identities, §5.6 perfect-reconstruction filterbanks. Cited 10k+ times.
- Smith, J. O. *Digital Audio Resampling Home Page* (CCRMA / Stanford, 2002, updated). — The arbitrary-rate windowed-sinc derivation; pseudocode for the Q-oversampled table; what `libsamplerate` is built on. https://ccrma.stanford.edu/~jos/resample/
- Farrow, C. W. (1988). "A continuously variable digital delay element". *IEEE ISCAS* 1988, pp. 2641-2645. The Farrow structure.
- Laakso, T. I., Välimäki, V., Karjalainen, M., & Laine, U. K. (1996). "Splitting the Unit Delay — Tools for Fractional Delay Filter Design". *IEEE Signal Processing Magazine* 13(1):30-60. — Closed-form Lagrange-fractional-delay coefficients, design methodology. The fractional-delay reference.
- Mintzer, F. (1982). "On half-band, third-band, and Nth-band FIR filters and their design". *IEEE TASSP* 30(5):734-738. Halfband-cascade decimators.
- Hogenauer, E. B. (1981). "An economical class of digital filters for decimation and interpolation". *IEEE TASSP* 29(2):155-162. CIC filters.
- Kaiser, J. F. (1974). "Nonrecursive digital filter design using the I₀-sinh window function". *Proc. IEEE ISCAS*. — Kaiser window; required for the prototype LP.
- de Castro Lopo, E. *libsamplerate* (BSD, 2002–). https://libsndfile.github.io/libsamplerate/ — Production reference impl of Smith arbitrary-rate. Three quality tiers (SINC_BEST_QUALITY = 97 dB, SINC_MEDIUM = 97 dB shorter, SINC_FASTEST = 47 dB) — interface to mirror.
- Norskog, L. & dev team. *SoX Resampler Library (libsoxr)* (LGPL, 2013–). https://sourceforge.net/projects/soxr/ — Polyphase + Kaiser, fastest open-source on rational ratios.
- FFmpeg `libswresample` (LGPL). Production async resampler; uses Kaiser-windowed-sinc with optional cubic interpolation; what most modern media tooling actually links against.
- Yen, J. L. (1956). "On nonuniform sampling of bandwidth-limited signals". *IRE Trans. Circuit Theory* 3(4):251-257. Non-uniform reconstruction (T6).
- Brandwood, D. (2003). *Fourier Transforms in Radar and Signal Processing*. Artech House. §7 polyphase, §8 fractional-sample-rate-conversion.
- ITU-R BS.1770-4 (2015). "Algorithms to measure audio programme loudness and true-peak". — Cites 4× oversampling true-peak detection requiring polyphase upsample (downstream consumer of T0/T1).

### Cross-link
- Slot 244 (time-stretch, if exists) and slot 264 (pitch-shift, if exists) **block** on this PR.
- Slot 167 (synergy-audio-signal) flagged the audio↔signal coupling — this is the canonical example.
- Slot 008 (audio-sota) lists "polyphase/sinc/Farrow resampling" among the missing modern-speech-layer primitives.

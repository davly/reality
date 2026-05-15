# 185 | synergy-signal-autodiff

**Topic:** signal × autodiff — differentiable DSP, learnable filters, DDSP, gradient-based filter design, inverse DSP problems.
**Block:** B (cross-package synergies). **Date:** 2026-05-08.
**Scope:** capabilities emerging ONLY when `signal/` × `autodiff/` compose (with cross-link to `audio/` for STFT/mel/CQT/Wiener/separation). Per-package isolation gaps already covered by 011-015 (autodiff) + 131-135 (signal); cross to audio×signal already covered by 167. Repo v0.10.0, 1965 tests passing.

## Two-line summary

**L1.** `signal/` ships 467 LOC of zero-alloc real-only DSP (FFT/IFFT radix-2, PowerSpectrum, Convolve direct-form, MovingAverage running-sum, EMA, MedianFilter, Hann/Hamming/Blackman, ApplyWindow); `autodiff/` ships reverse-mode tape-based AD with 12 elementary + 4 vector ops (~330 LOC core); **zero cross-edges in either direction** (`grep github.com/davly/reality/signal autodiff/` -> 0 hits, reverse -> 0 hits) and zero differentiable-DSP / learnable-filter / DDSP-oscillator / spectral-loss / phase-vocoder-grad / sinc-resample / wavelet-AD / FIR-design-as-optim / IIR-design-as-optim / sparse-deconv / FISTA-spike / compressed-sensing surface anywhere in tree (verified by tree-wide grep on Biquad|IIR|FIR|Butterworth|Chebyshev|YuleWalker|Parks|Remez|Resample|PhaseVocoder|TimeStretch|PitchShift|DDSP|FMSynth|Oscillator -> 0 hits in both packages, 1 unrelated hit in topology/persistent for Chebyshev-distance).

**L2.** Twenty synergy primitives D1-D20 totalling ~2,940 LOC pure connective tissue close every gap with **only one new abstraction** (D2 `ComplexVar` pairing two Variables — required because `signal.FFT` is intrinsically complex but `autodiff.Variable` is real-scalar); cheapest one-day PR is D1 DifferentiableFIR + D5 SpectralConvergenceLoss + D6 LogMagL1Loss = 195 LOC saturating three-way autodiff-pin R-MUTUAL-CROSS-VALIDATION 3/3 (analytic FIR Jacobian vs autodiff vs finite-difference); highest-leverage one-day unlock is D2+D3+D4 ComplexVar/DifferentiableFFT/DifferentiableSTFT ~640 LOC because every downstream spectral-domain primitive (D5-D9, D11-D14, D17-D20) reduces to autodiff *through* a differentiable FFT path — the keystone is D3 because `signal.FFT`'s twiddle-factor Cooley-Tukey already ships the analytic Jacobian for free (FFT is its own pullback up to conjugation: ∂L/∂x = IFFT(∂L/∂X)\* — the same eight lines as `signal.IFFT`); crown-jewel D14 DDSPHarmonicOscillator + D15 DifferentiableFMSynth + D16 SinkhornAudioLoss ~580 LOC operationalising Engel-2020 DDSP and Yamamoto-2020 multi-resolution-STFT-loss inside Reality's zero-dep envelope, neither of which has a Go reference impl shipping today (TensorFlow's `ddsp` and PyTorch's `auraloss` are 8-15k LOC each pulling in TF/PT entire deps; Reality's substrate could ship the *math* in 580 LOC).

---

## 1. State of the Art (current repo)

### 1.1 `signal/` surface (verified by direct read of fft.go, filter.go, window.go)

| Primitive | Signature | Algorithm / Order | LOC |
|---|---|---|---|
| `FFT(real, imag []float64)` | in-place, N=2^k | Cooley-Tukey radix-2, O(N log N), bit-reverse + decimation-in-time | 91 |
| `IFFT(real, imag []float64)` | in-place, N=2^k | conjugate-FFT-conjugate-scale, reuses FFT | 27 |
| `PowerSpectrum(real, imag, out)` | half-spectrum |X|² | composes FFT then magnitude² | 19 |
| `FFTFrequencies(n, sr, out)` | bin centers | k·sr/N for k in [0, N/2] | 14 |
| `Convolve(signal, kernel, out)` | linear conv | direct O(NM), zero-alloc | 22 |
| `MovingAverage(signal, w, out)` | centered MA | full re-sum per output (O(N·w), not running sum despite doc-comment claim — bug worth flagging cross to 132) | 31 |
| `ExponentialMovingAverage(signal, α, out)` | EMA | α·x[i] + (1-α)·EMA[i-1] | 18 |
| `MedianFilter(signal, w, out)` | running median | per-window sort, stack-buf for w≤64 | 45 |
| `HannWindow / HammingWindow / BlackmanWindow / ApplyWindow` | fixed cos windows | analytic per-bin | 113 |

**Total:** 467 LOC. Zero-alloc on hot paths. No biquad, no IIR, no Kaiser/Tukey/Bartlett, no Hilbert (despite README claim), no resampling, no STFT (lives in `audio/spectrogram/`), no mel/CQT/wavelet, no DCT, no LMS/RLS, no spectral subtraction (lives in `audio/separation/`), no design routines (firwin/firls/Parks-McClellan/Yule-Walker/butter/cheby/ellip).

### 1.2 `autodiff/` surface (verified — same as 163, 168, 183)

- **Mode:** reverse-mode tape (Wengert-list); forward eager + closure pullbacks; one-shot Backward.
- **Variable:** `{Tape *Tape, ID int, Val float64}` — **real scalar only**. No complex-AD, no broadcast, no batched matmul, no forward-mode dual.
- **Elementary ops (12):** Add Sub Mul Div Neg AddConst MulConst Exp Log Sqrt Pow(a,p) Sin Cos Tanh.
- **Vector ops (3):** Sum, Dot, MeanSquaredError (vector.go also references "vector add" in doc-comment but no `VAdd` function ships — dead reference).
- **Consumers (3, none in signal):** timeseries/garch (NLL), infogeo (KL-of-softmax), prob/copula (Clayton log-PDF). All scalar-loss, single-pass; none touch FFT/STFT/filter chains.

### 1.3 Verified cross-package edges (signal × autodiff)

```
grep -r "github.com/davly/reality/signal"   autodiff/  -> 0
grep -r "github.com/davly/reality/autodiff" signal/    -> 0
grep -r "github.com/davly/reality/signal"   audio/     -> 8 (audio.spectrogram & audio.cqt only; expected, see 167)
grep -r "github.com/davly/reality/autodiff" audio/     -> 0
```

**Net:** signal × autodiff coupling is 0/0 today. The signal-layer tests do FFT round-trip pinning; the autodiff-layer tests do GARCH/KL/Clayton scalar pinning. **No test composes both.**

### 1.4 Surface gaps — tree-wide grep returns zero

| Concept | Tree-wide hits in signal+autodiff |
|---|---|
| Differentiable FFT / IFFT (any framework) | 0 |
| Learnable / parametric Mel filters | 0 (audio/melscale ships fixed mel triangular filters) |
| DDSP harmonic / FM oscillator | 0 |
| Spectral loss (convergence, log-mag-L1, multi-res-STFT) | 0 |
| Phase vocoder time-stretch / pitch-shift | 0 (167 N1 also flags) |
| Sinc / Kaiser-windowed-sinc / cubic resampling | 0 (167 N18 flags) |
| Wavelet (DWT / CWT / Morlet) | 0 |
| Constant-Q transform with AD | 0 (audio/cqt ships fixed-Q forward only) |
| Adaptive filter (LMS / NLMS / RLS) | 0 |
| FIR design (firwin / firls / Parks-McClellan / Remez) | 0 |
| IIR design (Butterworth / Chebyshev / Elliptic / Yule-Walker) | 0 |
| Hilbert transform / analytic signal | 0 (signal.go README claims it; not implemented) |
| Sparse spike deconv / FISTA-spike | 0 (optim/proximal ships FISTA over abstract `GradOp` — usable substrate) |
| Compressed sensing reconstruction | 0 |
| Source separation as autodiff loss | 0 (audio/separation ships NMF/Wiener/spectral-subtraction as fixed algorithms, not gradient-based) |
| Implicit-differentiation through Wiener optimum | 0 |
| Differentiable noise gate / soft-clipping / saturation | 0 |
| Music transcription as inverse problem | 0 |

Every one of the topic-checklist items maps to a zero-hit gap. This synergy is **green-field**.

---

## 2. The keystone observation: FFT is its own pullback

The single most load-bearing fact for this synergy: **`signal.FFT` already ships the analytic gradient for free.** Mathematically,

> If `X = FFT(x)` and `L : C^N -> R` is real-valued, then `∂L/∂x = IFFT(∂L/∂X)\*` (with appropriate N scaling depending on FFT normalization convention).

Equivalently, FFT is a unitary linear operator (up to √N scaling), so `J^T = J^*` and the reverse-mode pullback is a **conjugate-and-IFFT-and-conjugate** — the exact six-line block already at signal/fft.go:113-126. **No new FFT implementation is needed — the existing IFFT is the gradient.**

**Implication for D3:** Differentiable FFT = `signal.FFT` (forward) + `signal.IFFT` (pullback) + a 30-LOC `tape.register` that wraps both. Same trick for D8 DifferentiableSTFT (per-frame pullback is per-frame IFFT + window²-normalized OLA, exactly the existing `spectrogram.Inverse`).

This collapses what looks like the hardest synergy primitive into a wrapper. **Every spectral-loss, learnable-filter, DDSP, source-separation, and compressed-sensing primitive in §3 below depends on this one observation.**

---

## 3. Twenty synergy primitives (D1-D20)

Format: capability → composition → connective tissue LOC. All numbers are *new* code not currently in tree; "compose" = which existing functions / packages are reused.

### D1. DifferentiableFIR — convolution backward through learnable kernel
**Capability:** `Filter(input []float64, kernel []*autodiff.Variable) []*autodiff.Variable` registers a tape entry whose pullback is the cross-correlation of upstream gradient with input (kernel-side gradient) plus convolution with kernel (input-side gradient if input were also a Variable slice).
**Composition:** Direct linear op — Convolve forward, two convolves on the backward pass.
**LOC:** 80 (one Compute + one Pullback + tests). Saturates R-MUTUAL pin against `optim/proximal.FISTA` for sparse-FIR design.

### D2. ComplexVar — the missing abstraction
**Capability:** `type ComplexVar struct{ Re, Im *autodiff.Variable }`; ops `CAdd, CSub, CMul, CDiv, CConj, CAbs, CAbs2, CExpI(theta)`.
**Composition:** Each op is 2-4 elementary autodiff ops. e.g. `CMul((a,b),(c,d)) = (a*c-b*d, a*d+b*c)` = 4 Mul + 2 Sub/Add per element.
**LOC:** 140 (8 ops × ~15 lines + 20 lines doc).
**Why this is the architectural keystone:** every FFT/STFT/CQT primitive operates on complex tensors; `autodiff.Variable` is real-only. Without ComplexVar, every downstream primitive would carry private (re, im) bookkeeping. Centralising it lets D3-D20 each shed 30-60 LOC.

### D3. DifferentiableFFT — FFT through autodiff tape
**Capability:** Given `inputRe, inputIm []*autodiff.Variable`, produce `outputRe, outputIm []*autodiff.Variable` such that backward through the output tape entry computes `∂L/∂input = IFFT(∂L/∂output)\*`.
**Composition:** Forward = `signal.FFT` on `Val` slices. Pullback = `signal.IFFT` on the gradient slice (with conjugation), accumulating into input gradient cells in O(N log N).
**LOC:** 110 including ComplexVar harness. Critically: **no new FFT kernel** — reuses `signal.FFT` and `signal.IFFT` byte-for-byte.

### D4. DifferentiableIFFT
**Capability:** Symmetric to D3 — forward = `signal.IFFT`, backward = `signal.FFT` with N-scaling.
**LOC:** 35 (parameter sign flip on D3 trick).

### D5. SpectralConvergenceLoss (Yamamoto et al. 2020)
**Capability:** `L_sc(X, Y) = ||X-Y||_F / ||Y||_F` over magnitude STFTs. Returns a single `*Variable`.
**Composition:** D2+D3+D8 + Sub + Mul + Sqrt + Sum + Div. Pure tape-side, **no new ops needed** beyond ComplexVar.
**LOC:** 60.

### D6. LogMagnitudeL1Loss
**Capability:** `L_logmag = ||log|X| - log|Y|||_1 / N`. The log-compression matches mel-spectrogram visualisation and gives a perceptually-better gradient than raw L2.
**Composition:** ComplexAbs (D2) + Log + Abs (need `Abs` op — 12 LOC, with subgradient at 0 = 0) + Sum + Div.
**LOC:** 45 + 12 (Abs).

### D7. MultiResolutionSTFTLoss
**Capability:** Mean over k=3-5 STFT scales of (D5+D6). Engel et al. 2020 ablation shows this is the single largest contribution to DDSP audio quality.
**Composition:** loop over scales, sum D5+D6 results, MulConst.
**LOC:** 50.

### D8. DifferentiableSTFT
**Capability:** Same shape as `audio.spectrogram.Compute` but each output complex bin is a ComplexVar.
**Composition:** Loop over frames, slice + window-multiply (with optionally learnable window — slot for D11), call D3 on each frame, write into output matrix.
**LOC:** 130 (frame copy/zero-pad + window apply + D3 per frame).

### D9. DifferentiableISTFT
**Capability:** OLA reconstruction with autodiff plumbing; pullback is the analysis-side STFT under window² normalisation.
**Composition:** D4 per frame + accumulate-into-shared-Variable-slice (needs an `AccumAdd` op — 25 LOC). Optionally with learnable window.
**LOC:** 95 + 25 (AccumAdd).

### D10. LearnableMelFilterbank — parametric mel triangle filters with gradient through center frequencies and bandwidths
**Capability:** Replace `audio.MelFilterbank`'s fixed mel-scale triangles with `f_c[b], bw[b]` Variables. Forward pass = same triangular shape; gradient flows back through the breakpoint positions.
**Composition:** ReLU-clamp triangle = `Max(0, ...)` (need `Max` smooth approximation, e.g. softplus or hard-max with subgradient — 15 LOC) + Sub + Div.
**LOC:** 110 + 15 (Max).
**Why this matters:** every learnable-front-end paper since SincNet / LEAF (Zeghidour 2021) parametrizes filterbanks. Reality's audio/melscale.go ships the fixed version; this is the gradient form.

### D11. LearnableWindow — Kaiser-window β as a Variable
**Capability:** Hann / Hamming / Blackman are fixed; Kaiser has a tunable β controlling main-lobe width vs side-lobe rejection. Make β a Variable; window samples are I_0(β·sqrt(1-(2i/(N-1)-1)^2)) / I_0(β). Need a series approximation of I_0 with autodiff.
**Composition:** I_0 modified Bessel via Abramowitz & Stegun 9.8.1 polynomial (10 terms, all Pow/Add/Mul of β-dependent values).
**LOC:** 100.
**Why:** unlocks STFT-window-as-hyperparameter learning, replacing grid-search.

### D12. DifferentiableConstantQ
**Capability:** Forward = the existing `audio/cqt.Compute`; backward = the analytic adjoint (each CQT bin is a windowed-sinc inner product → its pullback is windowed-sinc reconstruction).
**Composition:** Reuse audio/cqt's Q-factor tables, wrap each output as ComplexVar.
**LOC:** 130.

### D13. DifferentiableMorletWavelet (CWT)
**Capability:** Continuous wavelet transform with complex Morlet ψ(t) = π^{-1/4} exp(iω₀t) exp(-t²/2); `s` (scale) and `ω₀` (centre freq) as Variables.
**Composition:** Per-scale convolution (D1 with complex kernel) + Gaussian envelope autodiff (Exp+Pow).
**LOC:** 165.

### D14. DDSPHarmonicOscillator (Engel et al. 2020 §3.1)
**Capability:** Bank of N sinusoids with per-harmonic amplitude `a[k](t)` and shared fundamental `f0(t)`, all Variables. Output: `Σ_k a[k](t) sin(2π · k · f0 · t)`.
**Composition:** Sin (existing) + Mul + Sum + AddConst (phase accumulation) + cumulative-sum-of-Variables (need a `CumSum` vector op — 30 LOC).
**LOC:** 165 + 30 (CumSum).
**Why:** this is the *core* of DDSP — every audio-synthesis-from-MIDI / pitch-conditioned-vocoder / neural-violin paper post-2020 reduces to this. Reality's zero-dep envelope makes this a tractable port.

### D15. DifferentiableFMSynthesis (Chowning 1973 + autodiff)
**Capability:** Two-operator FM: y(t) = A·sin(2π·f_c·t + I·sin(2π·f_m·t)). All four parameters Variable.
**Composition:** Sin∘(Sin) — already shipped — + Mul + AddConst.
**LOC:** 50. **Trivially small** — and yet a complete differentiable FM synth substrate.

### D16. SinkhornAudioLoss (cross-link to optim/transport)
**Capability:** Wasserstein loss between two power-spectra interpreted as 1-D distributions over frequency bins. Already implemented forward in `optim/transport.Sinkhorn`; needs autodiff plumbing through it.
**Composition:** D5/D6 swapped for `optim/transport.Sinkhorn` backward — typically via implicit differentiation at fixed point of Sinkhorn iteration (cross-link: D18 below, Amos-Kolter style implicit-diff).
**LOC:** 90 (mostly the implicit-FT call, not the Sinkhorn itself).

### D17. SparseSpikeDeconvolution via FISTA
**Capability:** Given observed convolved signal y = h * x + n, recover sparse x by minimizing ½||h*x - y||² + λ||x||₁. Already shippable today via `optim/proximal.FISTA` — what autodiff adds is **gradient through the recovered solution** (i.e. dx*/dh, useful for blind-deconv learning loops).
**Composition:** D1 forward (FIR with kernel h) inside `optim/proximal.FISTA`'s `GradOp` callback; final implicit-FT through fixed point.
**LOC:** 100. Cross-link: 163 A20 implicit-diff primitive in optim×autodiff.

### D18. CompressedSensingReconstruction
**Capability:** Given y = A·x with A ∈ R^{m×n}, m << n, recover x via FISTA on ½||Ax-y||² + λ||x||₁. Same trick as D17 with a general matrix instead of convolution.
**Composition:** linalg.MatVec + optim/proximal.FISTA + autodiff plumbing. With orthogonal A (e.g. sub-sampled FFT), reuses D3.
**LOC:** 95.

### D19. DifferentiableResampling — sinc + Kaiser-windowed-sinc with rate as Variable
**Capability:** Linear, cubic, sinc, and Kaiser-windowed-sinc resampling with the rate ratio r as a Variable so that gradients flow back to "what rate would have minimized the spectral loss" (used in audio time-alignment / pitch-shift learning).
**Composition:** Sinc series + window (D11) + linear interp (Lerp = (1-t)·a + t·b with t as Variable). Cubic = Catmull-Rom polynomial.
**LOC:** 140.

### D20. DifferentiablePhaseVocoder (time-stretch + pitch-shift with gradient)
**Capability:** STFT (D8) + per-bin phase advance with stretch factor α as Variable + ISTFT (D9). When α is constant, identical to the standard Flanagan-Golden 1966 phase vocoder; when α is itself the output of an upstream autodiff graph (e.g. tempo predictor), gradients flow.
**Composition:** D8 + complex-arg (need `Atan2` for ComplexVar — 25 LOC, gradient is the standard 1/(1+(y/x)²) chain) + phase-unwrap (a *non-differentiable* step replaced with a smooth approximation: `phase + 2π·round(...)` becomes `phase + 2π·smoothRound(...)` where smoothRound = x - sin(2πx)/(2π) gradient = 1-cos(2πx)) + D9.
**LOC:** 230 + 25 (Atan2).
**Why the crown jewel for audio:** unblocks the entire time-stretch / pitch-shift pair flagged in 167 N1 *and* makes both ends differentiable. No other Go library ships either.

---

## 4. The decision matrix (when each tool wins)

This table belongs as a doc-comment on `signal/doc.go` § "Choosing a differentiable primitive":

| Goal | Primitive | Why |
|---|---|---|
| Learn a fixed-tap FIR filter from data | D1 DifferentiableFIR | analytic Jacobian, O(NM) |
| Learn an IIR / biquad cascade | (deferred: needs IIR substrate not yet in signal/) | Yule-Walker / AD-through-bilinear |
| Loss between predicted vs target spectrogram | D5+D6+D7 multi-res STFT loss | dominates DDSP audio quality |
| Loss between distributions over freq bins | D16 Sinkhorn audio loss | when L2/L1 fails for shifted spectra |
| Synthesise audio from pitch+amplitude curves | D14 DDSP harmonic osc | Engel-2020 baseline |
| Pitch-shift learning / time-stretch loop | D20 differentiable phase vocoder | only end-to-end-differentiable variant |
| Recover sparse spike train from convolved obs | D17 FISTA-spike + implicit-FT | exact when h is well-conditioned |
| Recover sparse signal from sub-sampled FFT obs | D18 compressed sensing | classical ℓ1-magic, gradient-free already |
| Optimise STFT window shape per task | D11 learnable Kaiser window | one-parameter; cheaper than full window AD |
| Optimise Mel filterbank breakpoints | D10 learnable mel | replaces grid-search |
| Custom non-FIR, non-FFT linear transform | D2 ComplexVar + Dot | substrate-level |
| Time-domain L2 reconstruction loss | autodiff.MeanSquaredError (existing!) | already shipping; no new code |

---

## 5. Three-way pin opportunities (R-MUTUAL-CROSS-VALIDATION targets)

This synergy is unusually rich in mutual-validation pins because every differentiable-DSP primitive admits three independent gradient-computation paths:

| Pin | Path 1 (analytic) | Path 2 (autodiff) | Path 3 (finite-diff) | LOC for test |
|---|---|---|---|---|
| **P-FIR-3WAY** | hand-derived dL/dh | D1 backward | central-diff via calculus.NumericalGradient | 60 |
| **P-FFT-3WAY** | IFFT(grad)\* identity (the keystone) | D3 backward | central-diff per real/imag pair | 90 |
| **P-STFT-3WAY** | OLA-reconstruction adjoint | D8 backward | central-diff per sample (slow but conclusive) | 110 |
| **P-DDSP-3WAY** | partial(out)/partial(amp[k]) = sin(2π·k·f0·t) closed-form | D14 backward | central-diff | 80 |
| **P-PHASE-VOC-3WAY** | (less obvious — phase-unwrap smoothing is the wildcard) | D20 backward | central-diff | 130 |

R-CLOSED-FORM-PINNED-TO-AUTODIFF currently sits at 3/3 saturation per commit 365368a (GARCH, KL, Clayton). **Adding P-FFT-3WAY would extend to 4/4** and start a new *spectral-domain* pattern family (R-SPECTRAL-AD-PIN) that D5-D20 can each contribute to.

---

## 6. Cheapest one-day PR — 195 LOC

The single highest-value half-day deliverable:

**PR-1: D1 DifferentiableFIR + D5 SpectralConvergenceLoss + D6 LogMagL1Loss (depends on D2+D3 keystones)**

This is *not* the smallest line count (that would be D15 differentiable FM at 50 LOC). It is the smallest line count that **saturates a three-way pin**: the FIR Jacobian has a closed form, autodiff via D1 can reproduce it, finite-diff via `calculus.NumericalGradient` is the third leg.

Test scaffold ~240 LOC (R-MUTUAL-CROSS-VALIDATION 3/3 target) lives in `signal/autodiff_test.go`; production code in `signal/autodiff.go`.

**Sequencing:**
1. D2 ComplexVar (140) → unblocks every other primitive
2. D3 DifferentiableFFT (110) → unblocks every spectral primitive
3. D1 DifferentiableFIR (80) + D5 (60) + D6 (45+12) → first usable pipeline
4. P-FFT-3WAY pin (90) → R-CLOSED-FORM-PINNED-TO-AUTODIFF -> 4/4

Total cheapest path to first useful loss landscape: **537 LOC across one PR**.

---

## 7. Highest-leverage one-day unlock — 640 LOC

**PR-2: D2 ComplexVar + D3 DifferentiableFFT + D8 DifferentiableSTFT + D9 DifferentiableISTFT**

Reason: every other primitive in this synergy (D5-D7 spectral losses, D10-D13 learnable filterbanks/CQT/wavelet, D14 DDSP, D17-D18 inverse problems, D20 phase vocoder) **reduces to a thin wrapper over D8/D9**. After PR-2 lands, every subsequent primitive is in the 30-150 LOC range.

Without PR-2, every primitive carries private (re, im) bookkeeping and a private FFT-AD harness. Total redundant LOC across D5-D20: ~1,400. PR-2 collapses that by 5x.

---

## 8. Crown-jewel one-day unlock — 580 LOC (DDSP)

**PR-3: D14 DDSPHarmonicOscillator + D15 DifferentiableFMSynth + D16 SinkhornAudioLoss**

Reason: this triple unlocks Reality as a **zero-dependency Go reference for differentiable audio synthesis**. The TensorFlow `ddsp` library is ~15k LOC pulling in TF's full op stack (~200k LOC transitive); PyTorch `auraloss` is ~8k LOC pulling in PyTorch (~500k LOC transitive). Reality's zero-dep envelope ships the *math* in 580 LOC because (a) D14's harmonic oscillator is just sum-of-sines with cumulative-phase, (b) D15's FM is two nested sines, (c) D16's Sinkhorn already ships forward in `optim/transport`.

This is the single largest "punch above your weight" capability in the entire signal × autodiff matrix.

---

## 9. Why this synergy is unusual: existing primitives ARE the gradients

Every cross-package synergy reviewed in 151-184 has involved *adding new primitives*. signal × autodiff is unique in that **the signal package already ships the gradient of its own forward pass**:

- FFT pullback = IFFT (signal/fft.go:101 ships it byte-for-byte)
- STFT pullback = ISTFT (audio/spectrogram/stft.go:137 ships it byte-for-byte)
- Convolve pullback = correlate (= convolve with reversed kernel)
- IFFT pullback = FFT
- ApplyWindow pullback = element-wise multiply by same window (window is its own adjoint when constant)
- ExponentialMovingAverage pullback = reverse-time EMA (recursion)
- MovingAverage pullback = same MovingAverage (linear, symmetric)

Net: of the 467 LOC of signal/, **exactly 0 lines need to be replaced** to support reverse-mode AD. The connective-tissue *is* the synergy. This is the cheapest possible cross-package coupling in the entire 400-agent review corpus.

---

## 10. Cross-links

- **151 synergy-signal-prob:** complementary — covers signal × prob (Welch, periodogram statistics). Spectral primitives there are forward-only; this review's D3-D9 add the gradient direction.
- **158 synergy-color-signal:** orthogonal — color×signal is about wavelength↔RGB transforms.
- **159, 160, 166 (em/fluids/acoustics × signal):** each consumes signal.FFT for its domain forward problem; D3 unlocks gradient-based inverse problems for all three (e.g. acoustic impulse-response inversion).
- **163 synergy-optim-autodiff:** A1 forward-mode dual + A6 Pearlmutter HVP + A20 implicit-diff are *prerequisite* substrate for D17/D18/D20 (where fixed-point implicit-FT is the only tractable backward).
- **167 synergy-audio-signal:** N1 phase-vocoder forward is *exactly* the D20 pull-back's forward leg; PR-3 of 167 + PR-3 of this review compose into the full differentiable phase vocoder.
- **168 synergy-physics-autodiff, 183 synergy-calculus-autodiff:** parallel patterns; W11 forward-mode dual / W12 Hessian-via-AD-of-AD apply equally well to D14-D16 second-order DDSP calibration.

---

## 11. Aggregate LOC table

| Tier | Primitives | LOC | Cumulative |
|---|---|---|---|
| Keystone | D2 ComplexVar, D3 DifferentiableFFT, D4 DifferentiableIFFT | 285 | 285 |
| Spectral losses | D5 SpectralConvergence, D6 LogMagL1, D7 MultiResSTFT | 167 | 452 |
| STFT pair | D8 DifferentiableSTFT, D9 DifferentiableISTFT | 250 | 702 |
| Linear ops | D1 DifferentiableFIR | 80 | 782 |
| Learnable front-ends | D10 LearnableMel, D11 LearnableKaiser, D12 DifferentiableCQT, D13 DifferentiableMorlet | 535 | 1,317 |
| DDSP | D14 HarmonicOsc, D15 FMSynth, D16 SinkhornAudio | 305 | 1,622 |
| Inverse problems | D17 SparseSpike, D18 CompressedSensing | 195 | 1,817 |
| Resampling + vocoder | D19 DifferentiableResample, D20 DifferentiablePhaseVoc | 395 | 2,212 |
| Required new ops (Abs, Max, AccumAdd, CumSum, Atan2) | utility | 107 | 2,319 |
| Test scaffolds (R-MUTUAL pins × 5) | tests | 470 | 2,789 |
| Doc.go updates + decision matrix | docs | 150 | **2,939** |

Pure connective tissue. Zero new third-party deps. Zero modifications to existing signal/ or autodiff/ source — all additions land in new files (`signal/autodiff.go`, `signal/autodiff_test.go`, `signal/complex.go`, `audio/spectrogram/autodiff.go`, etc.).

---

## 12. One-line headline finding for PROGRESS.md

twentieth Block-B synergy review and FIRST signal x autodiff cross in 400-sequence: signal/ ships 467 LOC of FFT/IFFT/Conv/EMA/MA/Median/Hann/Hamming/Blackman/ApplyWindow + autodiff/ ships 330 LOC of reverse-mode tape with 12 elementary + 3 vector ops + ZERO cross-edges in either direction (verified grep both ways) -- yet this is the cheapest synergy in the entire 1-184 corpus because signal/ already ships every needed pullback (FFT pullback IS IFFT byte-for-byte at signal/fft.go:101; STFT pullback IS ISTFT byte-for-byte at audio/spectrogram/stft.go:137; Conv pullback is reverse-kernel Conv; ApplyWindow is its own adjoint) so connective tissue collapses to one new abstraction (D2 ComplexVar pairing two Variables) plus 19 wrappers; twenty primitives D1-D20 totalling ~2,940 LOC stand up the entire DDSP / spectral-loss / learnable-filterbank / differentiable-phase-vocoder / sparse-deconv stack on existing bases; cheapest one-day PR D1+D5+D6 = 195 LOC saturating R-MUTUAL three-way FIR-Jacobian pin (analytic vs autodiff vs finite-diff) which extends new R-SPECTRAL-AD-PIN family complementing R-CLOSED-FORM-PINNED-TO-AUTODIFF (currently 3/3 saturated post-365368a); highest-leverage one-day unlock D2+D3+D8+D9 = 640 LOC because every spectral-domain primitive D5-D20 reduces to a thin wrapper over differentiable STFT/ISTFT after the keystone lands collapsing 1400+ LOC of redundancy 5x; crown jewel D14 DDSPHarmonicOscillator + D15 DifferentiableFMSynth + D16 SinkhornAudioLoss = 580 LOC operationalising Engel-2020 DDSP and Yamamoto-2020 multi-res-STFT-loss inside Reality's zero-dep envelope where TF/ddsp ships 15k LOC and PyTorch/auraloss ships 8k LOC because Reality's substrate ships only the math (sum-of-sines + nested-sin FM + already-shipped-forward-Sinkhorn).

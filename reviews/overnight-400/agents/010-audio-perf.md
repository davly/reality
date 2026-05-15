# 010 — audio: STFT/FFT throughput, ring buffers, SIMD

## Headline
audio violates "no allocations in hot paths" wholesale: every per-frame primitive (`spectrogram.Compute`, `Magnitude`, `MelSpectrogram`, `SuperFlux`, every onset detector, `CQT`) returns newly-allocated `[][]complex128` / `[][]float64` matrices and the per-frame inner loops re-allocate row slices, while the package ships zero benchmark files, no streaming/ring-buffer primitive, no `*Into` companions for the hot ops, and per-call recomputation of trig/window/filterbank tables that `signal/` already proved are precomputable — net effect is roughly 2T+1 heap allocations per STFT call (T = frame count) plus an FFT-twiddle `cos/sin` call per FFT stage that signal/fft.go computes inside the loop with no caching.

## Hot-path table (file:line, allocations per call)

`F` = frameSize, `T` = numFrames, `B` = numFilters, `K` = bins (= F/2+1), `M` = numCoeffs.

| function | file:line | allocations / call | per-frame allocs | notes |
|---|---|---|---|---|
| `spectrogram.Compute` | spectrogram/stft.go:53 | **1 + T + 2** = T+3 | 1 (`[]complex128` row) | row alloc in hot loop (line 93); FFT scratch is recycled but full-N (not N/2+1) |
| `spectrogram.Inverse` | spectrogram/stft.go:137 | 4 | 0 (writes to scratch) | OLA pass; one final `n /= windowSum+ε` divide loop is SIMDable |
| `spectrogram.Magnitude` | spectrogram/magnitude.go:19 | **1 + T** | 1 (`[]float64`) | uses `cmplx.Abs` → `math.Hypot` → branches; pure scalar |
| `spectrogram.LogMagnitude` | spectrogram/magnitude.go:55 | **1 + T** | 1 | `cmplx.Abs` + `math.Log10` per cell |
| `spectrogram.PowerSpectrum` | spectrogram/magnitude.go:86 | **1 + T** | 1 | re/im inlined — best of the three |
| `spectrogram.HalfSpectrum` | spectrogram/magnitude.go:119 | **1 + T** | 1 | needless defensive copy; should return view |
| `spectrogram.MelSpectrogram` | spectrogram/mel_spectrogram.go:42 | **1 + T + filterbank + power** = T+3 | 1 (`row`) | filterbank built once (good); power scratch reused; output rows allocated in loop (line 69) |
| `spectrogram.LogMelSpectrogram` | spectrogram/mel_spectrogram.go:86 | T+3 (delegates) | 1 | log applied in-place ✓ |
| `audio.MelFilterbank` | audio/melscale.go:81 | **2** (`melPoints`, `binPoints`) | n/a (cold) | should accept caller scratch — no `Into` form |
| `audio.MFCC` (DCT-II) | audio/mfcc.go:65 | 0 ✓ | 0 | but recomputes `cos(πk(b+0.5)/M)` every call — `M·N_coeffs` cosines per frame, no table cache |
| `audio.FrameMFCC` | audio/mfcc.go:130 | 0 ✓ | 0 | exemplary `Into`-style API |
| `signal.FFT` | signal/fft.go:49 | 0 ✓ | 0 | but `cos/sin(angleStep)` per stage at line 66/67, every call — no twiddle cache |
| `pitch.Yin` | pitch/yin.go:61 | **2** (`d`, `dPrime` length W=N/2) | n/a | O(N²/4) inner loop — naive O(N log N) FFT-based YIN exists (Cheveigné §III); also no caller scratch slot |
| `pitch.McLeodPitchMethod` | pitch/mpm.go:61 | **1** (`nsdf` length τmax+1) | n/a | O(N·τmax) inner loop, redundant — NSDF is FFT-derivable in O(N log N) |
| `pitch.AutocorrelationPitch` | pitch/autocorrelation.go:49 | **0** ✓ | 0 | but O(N·τmax) brute-force ACF, FFT-version is O(N log N) |
| `pitch.SubharmonicSummation` | pitch/subharmonic_summation.go:51 | **0** ✓ | n/a | 1 Hz step is fixed (no caller control); 2× redundant `math.Round`/multiply per harmonic |
| `onset.SpectralFluxOnset` | onset/spectral_flux.go:44 | **2 + picks** | 0 | recomputes `cmplx.Abs` for both `t` and `t-1` every step — no rolling cache |
| `onset.SpectralFluxStrength` | onset/spectral_flux.go:83 | **1** | 0 | same redundant Abs |
| `onset.SuperFlux` | onset/superflux.go:46 | **1 + T + 2** = T+3 | 1 (`mags[t]`) | T full-magnitude matrices retained — could be 2-row ring buffer |
| `onset.ComplexDomainOnset` | onset/complex_domain.go:47 | **2 + picks** | 0 (no scratch retained) | recomputes `Abs/Phase` for `t-1`/`t-2` on every frame; each `Phase` is `atan2`; could cache last 2 phase rows in O(F) ring |
| `onset.EnergyOnset` | onset/energy.go:42 | **3 + picks** | 0 | sample-domain — fast |
| `onset.PickPeaksAdaptive` | onset/peak_picking.go:109 | 1 (`picks` from PickPeaks) | 0 | mean+std two-pass; `picks` slice grows via `append` (no pre-size) |
| `tempo.Estimate` | tempo/tempo.go:40 | **1** (`acf` length maxLag+1) | 0 | brute O(N·maxLag) ACF — FFT-based ACF is O(N log N); plays at frame rate so cost is small |
| `beat.Track` | beat/beat.go:52 | **3** (`score`, `backlink`, `revBeats`, `beats`) → 4 | 0 | DP is O(N·window); no SIMD inner |
| `cqt.CQT` | cqt/cqt.go:100 | **0** ✓ | 0 | but per-bin recomputes Hann + `cos/sin(angle·n)` inside the per-sample loop (line 147–149); no atom cache, no FFT-CQT shortcut |
| `separation.WienerFilterInto` | separation/wiener.go:60 | 0 ✓ | 0 | exemplary; `cmplx.Abs` could be `re²+im²` to skip sqrt then take sqrt once for SNR |
| `separation.SubtractSpectrumInto` | separation/spectral_subtraction.go:57 | 0 ✓ | 0 | `cmplx.Abs(in)` then divide by it — equivalent to `gain·in/abs(in)` ; reformable as `re·s, im·s` scalar |
| `separation.NMF.Decompose` | separation/nmf.go:67 | **~2(F+R) + 6** matrices | 0 | scratch matrices allocated once per call ✓ but caller can't reuse across iterations; row-major + column-traversal in `Wᵀ V` (line 153) is cache-hostile (W[f][r] strided over outer-loop r) |

## Cache-locality issues

1. **NMF `Wᵀ V` (`separation/nmf.go:150-158`)**: outer `r`, middle `t`, inner `f` — `W[f][r]` reads stride `R` floats apart between successive `f`. Should swap loops to `f`-outer / `t`-inner with W row-cached. Same pattern in `WtW`, `VHt`, `HHt`, `WHHt`. All 5 inner products are column-wise on a row-major array.
2. **NMF `[][]float64` is two-level pointer chase per cell.** Single contiguous `[]float64` of size F·R + manual indexing wins on every iteration; matches `audio.MelFilterbank`'s row-major-flat convention (which is correct).
3. **STFT row-of-row layout (`out[t]` is its own slice)** breaks downstream SIMD. A consumer wanting a contiguous time-major or freq-major slab cannot get one — every transform that reads `stft[t][k]` (every onset detector, every spectrogram op) is a double indirection.
4. **`SpectralFluxOnset` reads `stft[t][k]` and `stft[t-1][k]` interleaved** — two pointer chases per bin per frame. A flat `[]complex128` plus `t*F+k` indexing halves cache pressure.

## Redundant trig (precomputable)

| location | what | per-call cost |
|---|---|---|
| `signal/fft.go:66-67` | `Cos/Sin(angleStep)` per FFT stage, per call | log₂N pairs × every FFT — 60 FPS × 100s of FFTs = O(10⁵) cos/sin/s; one cached twiddle table eliminates all |
| `audio/mfcc.go:84` | `Cos(piKOverM·(b+0.5))` inside `MFCC` | M·numCoeffs cosines per frame; one DCT-II cosine table is `M·numCoeffs` floats, computed once |
| `cqt/cqt.go:147-149` | per-sample `Cos`/`Sin` for atom + Hann `Cos` | nk × 3 trig calls per bin; entire CQT atom set can be precomputed once for fixed (sr, fMin, B, octaves) into `[]complex128` of size Σnk |
| `onset/complex_domain.go:65-73` | `Phase` (= `Atan2`) for every bin every frame, `Cos/Sin(predictedPhase)` for every bin every frame | 3 trig + 1 atan2 per bin per frame; 75% goes away if the previous frame's `Phase` array is cached in a 2-row ring |
| `onset/spectral_flux*.go` | `cmplx.Abs(stft[t-1][k])` recomputed every frame | save 1 row of magnitudes (length F) and rotate; halves Hypot cost |

## SIMD-amenable scalar loops (vectorizable today via stdlib only with care; SIMD-via-asm later)

These are all straight-line array→array kernels with no branches in the hot inner; an AVX2/NEON backend (or just better Go compiler auto-vec via simpler loop forms) would 2–8× them:

| loop | file:line | shape |
|---|---|---|
| windowed-frame copy | spectrogram/stft.go:82-90 | `frameReal[i] = samples[idx] * window[i]` (with branchful zero-pad — should split into two loops to vectorize the bulk) |
| OLA add + windowSum² | spectrogram/stft.go:165-168 | two FMA-pattern accumulations |
| OLA normalize | spectrogram/stft.go:171-173 | element-wise divide |
| `Magnitude` cell | spectrogram/magnitude.go:32 | `Hypot` — replace with `sqrt(re²+im²)` for vectorization (Hypot's overflow-safe path defeats SIMD) |
| `PowerSpectrum` cell | spectrogram/magnitude.go:99-101 | already inline-friendly |
| `LogMagnitude` cell | spectrogram/magnitude.go:67-69 | `Log10` per cell — vectorizable via vec-math libs |
| `audio.PowerSpectrum` | audio/melscale.go:178-180 | trivial vector dot |
| `ApplyFilterbank` per-band dot | audio/melscale.go:200-207 | textbook SAXPY-like — but NB filterbank is sparse (most weights zero); a CSR encoding of the triangles would skip zeros and 5–10× the loop |
| `LogMelEnergies` | audio/mfcc.go:24-30 | branchful `max(e, floor)` then `Log` — split clamp from log to vectorize the clamp |
| `Yin` step 1 difference | pitch/yin.go:111-118 | `Σ (frame[n]-frame[n+τ])²` — classic SIMD candidate; sum-of-products of differences |
| `Yin` step 2 cumulative norm | pitch/yin.go:122-131 | scalar prefix sum then divide — limited SIMD value |
| `MPM` r/m accumulator | pitch/mpm.go:101-112 | two parallel sum reductions; SIMD-perfect |
| `AutocorrelationPitch` ACF | pitch/autocorrelation.go:90-98 | SAXPY accumulator |
| `WienerFilterInto`/`SubtractSpectrumInto` | separation/wiener.go:68-84, separation/spectral_subtraction.go:67-85 | per-bin scalar ops on parallel re/im arrays — SoA layout would beat the current AoS `[]complex128` |
| `NMF` matrix muls | separation/nmf.go:150-223 | five GEMM-shaped loops, all SIMDable; row-major flat storage prerequisite |

## Benchmark gaps

- **Zero `*_bench*.go` or `Benchmark*` functions in the entire `audio/` tree.** Verified by `find audio/ -name '*_bench*.go'` (empty) and `Grep` for `func Benchmark` (zero hits in audio/). The only places benchmarks could anchor "60-FPS budget" claims have none.
- No `testing.AllocsPerRun` assertions anywhere in audio. The CLAUDE.md "no allocations in hot paths" rule is therefore unenforced.
- No `signal.FFT` benchmark either — the substrate of the entire audio stack is unmeasured.
- Recommended minimum bench coverage:
  - `BenchmarkSTFT_1024_256` and `_2048_512` over 1 s of audio @ 48 kHz (the realistic Pistachio frame budget)
  - `BenchmarkMelSpectrogram` with 26 / 64 / 128 bands
  - `BenchmarkSuperFlux` (baseline) and `BenchmarkComplexDomainOnset` (atan2-heavy)
  - `BenchmarkYin_2048` and `BenchmarkMPM_2048` — pitch is the per-frame chokepoint
  - `BenchmarkCQT` for one chord at piano range (84 bins)
  - All paired with `b.ReportAllocs()`

## Streaming / ring buffer

- **No `audio.RingBuffer` or `Streamer` type exists.** `audio.DegradationTracker` has a private float64 ring at `degradation.go:34-37` (`Window`/`WindowHead`/`WindowFill`) but it's a feature-tracker, not an audio-frame ring.
- Every STFT-consuming detector takes `[][]complex128` — i.e., the full computed matrix, batch-mode. There is no "push one frame, get one onset decision" interface. Compare changepoint.Bocpd which is online (per agent 009).
- For Pistachio's 60-FPS / kHz-rate target the missing primitive is a `RingBuffer` + a stateful `STFTStream{frameSize, hopSize, window []float64; ring; scratch}` that takes one new sample-block of `hopSize` samples and emits one complex frame in-place. Every onset/pitch detector then consumes that single frame statefully.

## Comparison with `signal/`

`signal/` shows the right pattern and `audio/` mostly fails to follow it:

| convention | signal/ | audio/ |
|---|---|---|
| `*Into` form alongside allocator | yes (`PowerSpectrum(real, imag, out)`) | partial (`Wiener*Into`, `Subtract*Into`, `MelFilterbank` writes to caller `out`); STFT, MelSpec, every onset, every pitch detector all allocate |
| zero-alloc inner loops | yes (`FFT` is in-place, `bitReverse` is in-place) | no (every per-frame loop allocates a row) |
| flat row-major `[]float64` | yes (`MelFilterbank` is `numFilters*nBins`) | mixed — `MelFilterbank` is flat ✓ but STFT/MelSpec output is `[][]float64` ✗ |
| caller-owned scratch | yes (FFT real+imag are caller's) | partial (FrameMFCC ✓, STFT internal) |
| twiddle / cos cache | **no** (signal/fft.go:66 recomputes per call) | n/a (depends on signal/) |

The single biggest leverage point: **adding a `signal.FFTPlan{n int, twiddle []complex128}` with `(p) Forward(real, imag)`** would eliminate the cos/sin per-FFT-stage cost across audio entirely, with one cache-line of state per frame size.

## Recommendations (priority order)

1. **Add `Benchmark*` + `b.ReportAllocs()` for STFT, MelSpec, SuperFlux, Yin, CQT** before optimizing anything. Without numbers, all claims are conjecture. (Highest leverage; ~1 day.)
2. **Introduce `signal.FFTPlan` (twiddle cache) + audio adopts it.** Removes log₂N `Cos/Sin` calls per FFT, every FFT, in every frame.
3. **Add `*Into` companions for every `[][]…`-returning function** in `spectrogram/` and `onset/`. Pattern: `ComputeInto(samples, frameSize, hopSize, window []float64, out []complex128)` taking a flat `T*F`-sized slab. Returning forms become thin wrappers.
4. **Switch matrix returns to flat `[]complex128` / `[]float64` with explicit `(t,k) → t*F+k` indexing.** Halves cache pressure and unblocks SIMD. Big API change — version-bump worthy.
5. **Cache `mags[t-1]` in onset detectors** (`SpectralFlux*`, `SuperFlux`, `ComplexDomain`) as a 2-row ring of length F. Eliminates the redundant `cmplx.Abs` (and in CD, the `Phase` atan2) on every previous frame.
6. **Precompute MFCC DCT-II cosine table.** `M·numCoeffs` floats, one cold pass; saves M·numCoeffs `Cos` calls per frame.
7. **Precompute CQT atom bank.** `CQTPlan{atoms [][]complex128}` with `(p) Compute(x, out)` — eliminates per-call `Cos/Sin/Cos(Hann)` over the entire input, every call. Currently CQT is unusable at 60 FPS.
8. **Add `audio.RingBuffer` (float64) + `STFTStream` struct.** Per-frame `Push(samples) → frame complex128, ok bool` API. Minimum viable streaming primitive.
9. **NMF: rewrite the 5 matmuls with row-major flat storage and the standard `f`-outer / `r`-middle / `t`-inner ordering** (or pull in `linalg`'s GEMM, but linalg may not exist with that name — verify). Same change unlocks SIMD.
10. **`Magnitude` should compute `sqrt(re²+im²)` not `cmplx.Abs`.** `Abs` calls `math.Hypot` which has an overflow-safe branch defeating vectorization. The values we feed it are bounded by float64 safe-math territory; `sqrt(re*re+im*im)` is faster and SIMDable.
11. **Replace `math/cmplx` AoS with parallel `[]float64` re/imag throughout the hot path.** `complex128` is fine for caller-facing return types but inside the kernel SoA wins on SIMD and matches what `signal.FFT` already does.
12. **`HalfSpectrum` should return a view, not a copy** — or be deleted in favor of caller-side slicing.
13. **YIN/MPM/Autocorr should grow FFT-based variants** (`YinFFT`, `MPMFFT`) for large frames. Brute O(N²) is fine at N=512 but Pistachio frame sizes commonly hit 2048-4096 — at 4096 the brute autocorr is 8M muls/frame × 60 FPS ≈ 0.5 G muls/s for one detector.
14. **`SubharmonicSummation` 1 Hz grid step is hard-coded.** Add `step` parameter or auto-compute from binWidth.

## Sources

- C:/limitless/foundation/reality/audio/spectrogram/stft.go (lines 53, 75-78, 82-90, 93, 137, 165-173)
- C:/limitless/foundation/reality/audio/spectrogram/magnitude.go (lines 19, 30, 32, 55, 67-69, 86, 99-101, 119, 131-132)
- C:/limitless/foundation/reality/audio/spectrogram/mel_spectrogram.go (lines 42, 53, 56-71, 86)
- C:/limitless/foundation/reality/audio/melscale.go (lines 81, 106, 112, 178-180, 200-207)
- C:/limitless/foundation/reality/audio/mfcc.go (lines 65, 80-91, 130)
- C:/limitless/foundation/reality/audio/onset/spectral_flux.go (lines 44, 56-68, 83, 99-105)
- C:/limitless/foundation/reality/audio/onset/superflux.go (lines 46, 61-70, 73-98)
- C:/limitless/foundation/reality/audio/onset/complex_domain.go (lines 47, 57, 62-78)
- C:/limitless/foundation/reality/audio/onset/energy.go (line 42)
- C:/limitless/foundation/reality/audio/onset/peak_picking.go (lines 40, 109, 122-134)
- C:/limitless/foundation/reality/audio/pitch/yin.go (lines 61, 110-118, 121-131)
- C:/limitless/foundation/reality/audio/pitch/mpm.go (lines 61, 100-112)
- C:/limitless/foundation/reality/audio/pitch/autocorrelation.go (lines 49, 90-98)
- C:/limitless/foundation/reality/audio/pitch/subharmonic_summation.go (lines 51, 92-110)
- C:/limitless/foundation/reality/audio/cqt/cqt.go (lines 100, 121-153, 159-166)
- C:/limitless/foundation/reality/audio/tempo/tempo.go (lines 40, 65, 95-110)
- C:/limitless/foundation/reality/audio/beat/beat.go (lines 52, 76-81, 88-119)
- C:/limitless/foundation/reality/audio/separation/wiener.go (lines 49, 60-85)
- C:/limitless/foundation/reality/audio/separation/spectral_subtraction.go (lines 46, 57-86)
- C:/limitless/foundation/reality/audio/separation/nmf.go (lines 67, 121-145, 150-223)
- C:/limitless/foundation/reality/audio/degradation.go (lines 28-37, 71-108)
- C:/limitless/foundation/reality/audio/vibration/fundamental.go (lines 52, 72-78)
- C:/limitless/foundation/reality/signal/fft.go (lines 49, 65-90, 101-127, 140-158)
- benchmark gap: `find audio/ -name '*_bench*.go'` returned zero matches
- ring-buffer gap: `Grep RingBuffer|Stream` in audio/ returns only doc.go strings and `audio.DegradationTracker.Window` (not an audio frame ring)

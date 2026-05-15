# 009 — audio: API ergonomics for streaming vs batch, dtype handling

## Headline
audio is a confident *batch* library that pretends streaming is somebody else's problem: STFT/Magnitude/Onset/Pitch all swallow the entire signal and allocate fresh `[][]complex128` / `[][]float64` per call, no detector exposes a stateful `Step(frame) -> result` interface, three pitch detectors disagree on the unvoiced-frame return contract (`(0,1)` vs `(0,0)` vs `0`), and the `complex128` STFT row type is the single biggest blocker to the "60 FPS, zero-alloc, mobile-friendly" promise the package CLAUDE comments make.

## API surface — streaming vs batch by function

| function | regime | signature shape | per-call alloc | hop visible | unvoiced/no-pitch return |
|---|---|---|---|---|---|
| `audio.MFCC` | per-frame, alloc-free | `(in, n, out)` | none | n/a | n/a |
| `audio.MelFilterbank` | one-shot setup | `(..., out)` | none in core (2× tmp slices internally) | n/a | n/a |
| `audio.PowerSpectrum` | per-frame, alloc-free | `(re, im, out)` | none | n/a | n/a |
| `audio.ApplyFilterbank` | per-frame, alloc-free | `(power, fb, ..., out)` | none | n/a | n/a |
| `audio.LogMelEnergies` | per-frame, alloc-free | `(in, floor, out)` | none | n/a | n/a |
| `audio.FrameMFCC` | per-frame, alloc-free | scratch-soup signature | none | n/a | n/a |
| `audio.DegradationTracker` | **streaming, stateful** | struct + `Push*`/`Reset*` funcs | none | n/a | n/a |
| `audio.Fingerprint` | **streaming, stateful** | struct + `Update*`/`Merge*` funcs | none | n/a | n/a |
| `spectrogram.Compute` | batch | `(samples, n, hop, win) -> [][]complex128` | T·(2n + 1 row) | yes | n/a |
| `spectrogram.Inverse` | batch | `(stft, n, hop, win) -> []float64` | full out + windowSum | yes | n/a |
| `spectrogram.Magnitude` / `LogMagnitude` / `PowerSpectrum` | batch | `[][]complex128 -> [][]float64` | T·F | n/a | n/a |
| `spectrogram.MelSpectrogram` | batch | as above | T·numFilters + tmp | n/a | n/a |
| `cqt.CQT` | batch | `(x, sr, fMin, B, oct, out)` returns `error` | none in core (per-bin atom) | n/a | n/a |
| `onset.EnergyOnset` | batch | `(samples, n, hop) -> []int` | T + 2T scratch + picks | yes | empty slice |
| `onset.SpectralFluxOnset` / `SpectralFluxStrength` | batch | `[][]complex128 -> []int / []float64` | T + picks | only via STFT | empty / SF[0]=0 |
| `onset.SuperFlux` | batch | `[][]complex128 -> []int` | T·F mags + T + picks | only via STFT | empty |
| `onset.ComplexDomainOnset` | batch | `[][]complex128 -> []int` | T scratch + picks | only via STFT | empty (T<3 panics) |
| `onset.PickPeaks{,Adaptive}` | batch | `[]float64 -> []int` | picks | n/a | empty |
| `pitch.Yin` | per-frame, **alloc-IN-frame** | `(frame, sr, thr, fMin, fMax) -> (f0, ap)` | 2× `make([]float64, W)` | n/a | **(0, 1)** |
| `pitch.McLeodPitchMethod` | per-frame, **alloc-IN-frame** | `(frame, sr, fMin, fMax) -> (f0, clarity)` | `make([]float64, tauMax+1)` | n/a | **(0, 0)** |
| `pitch.AutocorrelationPitch` | per-frame | `(frame, sr, fMin, fMax) -> f0` | none | n/a | **0** (single return!) |
| `pitch.SubharmonicSummation` | per-frame | `(spec, sr, fMin, fMax, h) -> f0` | none | n/a | **0** (single return!) |
| `vibration.FundamentalHz` | per-frame, alloc-free | `(re, im, sr, fMin, fMax)` (uses caller FFT bufs) | none | n/a | 0 |
| `tempo.Estimate` | batch | `(novelty, fr, opts) -> (bpm, error)` | acf | yes (`frameRate`) | `ErrNoTempo` |
| `beat.Track` | batch | `(novelty, fr, opts) -> ([]Beat, error)` | score+backlink | yes | `ErrInsufficientData` |
| `separation.SubtractSpectrum{,Into}` | per-frame | both forms | alloc form mallocs | n/a | n/a |
| `separation.WienerFilter{,Into}` | per-frame | both forms | alloc form mallocs | n/a | n/a |
| `separation.EstimateNoiseSpectrum` | batch (over silence frames) | `(frames, n, out)` | none | n/a | n/a |
| `separation.IsVoiced{,Adaptive}` / `FrameEnergy` / `ZeroCrossingRate` | per-frame | scalar-out | none | n/a | n/a |
| `separation.FastICA` | batch | `(K×T, iters) -> K×T` | full | n/a | n/a |
| `separation.NMF.Decompose` | batch | `(F×T, R, iters) -> (W, H)` | full | n/a | n/a |
| `segmentation.SegmentByEnergy` / `*OnsetOffset` / `*WithMinSilence` | batch | `(samples or stft, ...) -> []Segment` | full | yes | empty |
| `segmentation.MergeCloseSegments` / `FilterByMinDuration` | post-process | `[]Segment -> []Segment` | new slice | n/a | n/a |

Two streaming citizens (`DegradationTracker`, `Fingerprint`) — both *features*, not *DSP*. Every actual DSP block is batch, allocates, or both.

## 1. Streaming vs batch contract — the central failure

The package has **no abstraction for "stream this in, get a result per frame, keep state across calls."** The closest things — `DegradationTracker` and `Fingerprint` — are post-feature trackers, not signal-domain blocks. Compare to `changepoint.Bocpd` (`changepoint/bocpd.go:85`) which is the *correct* template: explicit struct, `New(cfg)` constructor, `Update(x)` per observation, `RunLengthPosterior()` query, documented "not safe for concurrent use." Audio has nothing like this for any of the things that *cry out* for it:

- **`Yin`/`McLeodPitchMethod`** allocate `make([]float64, W)` and `make([]float64, tauMax+1)` **on every frame call**. (`pitch/yin.go:110, 121`; `pitch/mpm.go:100`.) For 100 fps pitch tracking on a 16 kHz frame, that's 200 alloc/sec/voice on the hot path. There is no `YinTracker` struct that owns the `d`, `dPrime` buffers across calls.
- **`spectrogram.Compute`** is monolithic batch: it can't process a streaming buffer. The author already wrote the per-frame loop body internally (`spectrogram/stft.go:79-98`) but doesn't expose it. There is no `StftStreamer` that owns `frameReal`/`frameImag` and a ring buffer.
- **All four `onset/*` detectors** consume a `[][]complex128`. To run them online a caller must accumulate the entire STFT first — defeats the point of online onset detection (Böck's whole 2012 paper, *cited in `peak_picking.go:107`*, is about online operation).
- **`tempo.Estimate` and `beat.Track`** require the full novelty function. No incremental-tempo or windowed-beat-tracker.
- **`separation.SubtractSpectrum` / `WienerFilter`** correctly provide `*Into` zero-alloc forms — *but* the noise estimate is itself a batch primitive (`EstimateNoiseSpectrum` consumes a `[][]complex128` of silence frames). No streaming noise tracker (Martin minima, IMCRA, MMSE-noise-PSD).
- **`segmentation.SegmentByEnergy`** holds an internal `inSegment`/`segStart` state machine for one call (`segmentation/vad_based.go:72-104`) but flushes it at function return. A streaming variant could expose those as a `EnergyVad` struct.

The **fix shape** is a parallel `Streamer` companion for every batch function:

```go
type StftStreamer struct { /* ring, scratch */ }
func NewStftStreamer(frameSize, hopSize int, window []float64) *StftStreamer
func (s *StftStreamer) Push(samples []float64, frameOut []complex128) (frameReady bool)

type YinTracker struct { /* d, dPrime, sr, params */ }
func NewYinTracker(sampleRate, threshold, fMin, fMax float64, frameLen int) *YinTracker
func (y *YinTracker) Process(frame []float64) (pitch, aperiodicity float64)

type SpectralFluxTracker struct { /* prevMag, ringSF, picker */ }
func (sf *SpectralFluxTracker) Push(spectrum []complex128) (onsetThisFrame bool)
```

The batch forms can then be one-line wrappers over the streamers, eliminating duplication and giving online-DSP consumers a real entry point.

## 2. Dtype: float64-only is sized wrong for audio

CLAUDE.md's "fp64 canonical" rule is right for *most* of reality. Audio is the package where it is most expensive:

- **STFT row is `[]complex128`** (`spectrogram/stft.go:53`). 16 bytes per bin. A 1024-point STFT at hop=256 over 1 minute at 22.05 kHz is 5168 frames × 1024 × 16 B = **85 MB just for one minute of one channel**. That's 4× larger than the float32-complex equivalent that librosa, torchaudio, Essentia, and any iOS Accelerate/Android NDK target would use.
- **No `StftFloat32` variant** offered. The CLAUDE comment about cross-substrate parity shipping to "Kotlin Android, Swift iOS, SvelteKit/TypeScript" is precisely the cohort that wants float32 (Android `AudioRecord` is int16 → float32; iOS Accelerate's vDSP is float32-default; Web Audio is `Float32Array`).
- **The `complex128` row is itself a wrapper** that prevents SIMD-friendly storage. A `struct{ Re, Im []float64 }` (parallel-arrays form, like `signal.FFT`'s `(real, imag []float64)`) would let a future SIMD inner-loop vectorise via paired loads — `complex128` doesn't.
- **`ComplexDomainOnset` calls `cmplx.Phase` twice per bin per frame** (`onset/complex_domain.go:65-66`) on a `complex128` — each call does an `atan2`. With paired-`[]float64` storage and a cached previous-phase ring, the per-bin cost drops 2× and unwrap becomes incremental.

Fix shape:

1. Promote `signal.FFT`'s parallel-array convention to the STFT type: `type Stft struct { Re, Im [][]float64 }` (or `Frames []FrameSpec` where `FrameSpec` is two slices). `complex128` stays usable via accessors.
2. Add `audio32` sub-package (or `Float32` suffixed forms) for the four hottest hot-path types: `MelFilterbankF32`, `MFCCF32`, `PowerSpectrumF32`, `LogMelEnergiesF32`. Cross-platform port targets get a primary path; fp64 stays the gold standard.
3. Document the dtype contract in `audio/doc.go:1-55` — currently silent on it. Reader should know "fp64 by default; fp32 sub-package exists for embedded/mobile mirrors."

## 3. Frame contract — implicit, inconsistent, undertyped

- **`audio.FrameMFCC` requires the caller to have called `signal.FFT(frameReal, frameImag)` *before* invoking it** (`audio/mfcc.go:127-130, 136`). This is documented in the comment but not in the type signature. There is no `FrameContext` struct, no enforcement, no panic if the caller forgot. A user who reads only the function signature `FrameMFCC(frameReal, frameImag, power, ..., out)` cannot tell that `frameReal`/`frameImag` is post-FFT input, not raw audio.
- **Frame *index* convention is undefined**. `spectrogram.Compute` documents `start := t * hopSize` (`stft.go:80`) — left-edge / "anchored at first sample" convention. `spectrogram.Inverse` reconstructs at `start := t * hopSize` (`stft.go:164`) — same. But `onset.EnergyOnset` returns `[]int` of *frame indices* (`onset/energy.go:42`), and `segmentation.SegmentByOnsetOffset` returns `Segment{StartIdx, EndIdx}` in *frame coordinates* (`onset_offset.go:43`) while `SegmentByEnergy` returns *sample coordinates* (`vad_based.go:50`). **Two different segmentation functions in the same sub-package use different coordinate systems** with no unifying type to make the difference visible.
- **`Beat.FrameIndex` + `Beat.TimeSeconds`** (`beat/beat.go:9-13`) is the right shape — both representations bundled. Adopt for `Onset` (currently a bare `int`) and `Segment` (currently bare `StartIdx`/`EndIdx int` with no unit tag).
- **Hop size is invisible to downstream consumers**. An onset returned by `SpectralFluxOnset` is a frame index; converting back to seconds requires the caller to remember the hop size used three calls ago. The right shape is to return a typed `Frame{Index int, Hop int, SampleRate float64}` or carry hop in the result.
- **No "centre vs left" frame timing convention is documented anywhere**. STFT framing is left-anchored (zero-padded trailing frame, `stft.go:80-90`); pitch frames have no documented anchor at all; tempo/beat consume a "novelty function" that is implicitly left-anchored. Compare librosa which lets the caller choose `center=True/False` per-call.

## 4. Error / edge-case contract — three different conventions for "no pitch"

Run a small grep-and-tabulate on the four pitch detectors:

| detector | silent frame | sub-threshold | invalid input |
|---|---|---|---|
| `Yin` | `(0, 1)` | falls back to global min | `panic` (`yin.go:62-69`) |
| `McLeodPitchMethod` | `(0, 0)` | falls back to global argmax | `panic` (`mpm.go:62-67`) |
| `AutocorrelationPitch` | `0` (no second return) | `0` | `panic` (`autocorrelation.go:50-55`) |
| `SubharmonicSummation` | `0` (no second return) | `0` | `panic` (`subharmonic_summation.go:52-60`) |

The four detectors in *one sub-package* return three different things for "I couldn't find a pitch":
- `(pitch, confidence)` with `confidence=1` meaning *worst*
- `(pitch, clarity)` with `clarity=0` meaning *worst*
- `pitch` only with no confidence channel

A consumer wanting to ensemble three pitch detectors (a perfectly normal workflow — the file `audio/onset/cross_validation_test.go` does exactly this for *onset*; doing it for *pitch* is the natural next step) has to write three different parsers for the "no pitch" sentinel. The two confidence-bearing detectors then disagree on **what direction is good**: YIN aperiodicity → `0` is best, `1` is worst; MPM clarity → `1` is best, `0` is worst. A consumer writing a reliability gate must remember per-detector polarity.

Fix:
- One `Pitch` struct: `type Pitch struct { Hz float64; Confidence float64 /* 0..1, higher=better */; Voiced bool }`.
- All four detectors return `Pitch`. `AutocorrelationPitch` adds a parabolic-interpolated peak-clarity (it has the data — it just throws it away at `autocorrelation.go:104`). `SubharmonicSummation` has a natural `bestS / sumOfAll` confidence ratio.
- `Voiced=false` is the canonical "no pitch" return; `Hz` may be 0 or the best guess as the algorithm prefers, but consumers gate on `Voiced` rather than `Hz==0`.

Same disease afflicts onset detectors — three return `[]int` and one (`SpectralFluxStrength`) returns `[]float64`, with no per-detector confidence/strength bundled with the pick. A `type Onset struct { Frame int; Strength float64 }` would mirror `Beat` and let consumers fuse detectors by score.

Same disease in error contract: half the package `panic`s on bad input (`audio/`, `pitch/`, `onset/`, `spectrogram/`, `segmentation/`); CQT, tempo, beat use `error` returns with sentinel values (`cqt/cqt.go:186-201`, `tempo/tempo.go:136-153`, `beat/beat.go:155-164`). The newer code is `error`-returning — that's the right direction — but the audio root and the early sub-packages are panic-returning. **There is no documented project policy.** This is split-by-author, not split-by-design.

## 5. Stateful blocks — `DegradationTracker` and `Fingerprint`

The two existing stateful structs are individually well-designed but neither addresses the **streaming-DSP** problem:

- **Goroutine safety**: neither is. `DegradationTracker` has no mutex, no `sync.RWMutex`, no atomic. `Fingerprint` is the same. `bocpd.go:85` explicitly calls this out ("not safe for concurrent use. Wrap in a mutex if shared.") — audio's stateful structs **don't even document the question**.
- **Resettable**: `DegradationTracker` has `ResetWindow` and `ResetBaseline` (`degradation.go:184, 195`). Good. `Fingerprint` has no `Reset` — once an outlier observation pollutes Welford's `M2`, the only fix is to start over with `NewFingerprint`. Should add `ResetFingerprint(fp *Fingerprint)`.
- **Serialisation contract**: both have exported fields specifically so they can be `gob`/`json`-encoded across runs (`degradation.go:27`, `fingerprint.go:24`). But there is no documented cross-substrate format — the Kotlin/Swift/TypeScript ports the package promises (`audio/doc.go:18-22`) cannot deserialise a Go-encoded `DegradationTracker` byte-for-byte.
- **No method receivers**: `UpdateBaseline(t *DegradationTracker, x)`, `PushObservation(t, x)`, `ZScore(t)` are all *free functions*. The author chose this idiomatically — but every other reality stateful type uses methods (`bocpd.go: b.Update(x)`). Inconsistent across packages.

## 6. Sibling-package idiom comparison

| package | streaming primitive | dtype | error contract | state ergonomics |
|---|---|---|---|---|
| `signal/` | `MovingAverage`, `EMA` are batch-only; same complaint applies | `[]float64` + `(re, im []float64)` for FFT | `panic` | none |
| `changepoint/` | **`Bocpd` struct + `Update(x)` is the canonical streaming model** | `[]float64` | `error` returns from `New` and `Update` | methods on `*Bocpd`, explicit "not safe for concurrent" doc |
| `timeseries/` | only `dcc/` and `garch/` — all batch-fit + per-step incremental | `[]float64` | `error` | mixed — fitter struct + free-function predictors |
| `audio/` (this package) | only `DegradationTracker`, `Fingerprint` (feature trackers, not DSP) | `[]float64`, `[]complex128`; no float32 | `panic` mostly; `error` in cqt/tempo/beat | free functions on exported-field structs |

`changepoint.Bocpd` is the in-house gold standard for an online algorithm: struct constructor with `Config`, per-observation `Update`, query methods, documented thread-safety, sentinel errors. Audio should clone this idiom for every detector that has internal state across frames (Yin, MPM, onset detectors, spectral flux, NSDF, energy ring, VAD, noise tracker, fingerprint, degradation tracker). Today only the *post-feature* trackers approximate this idiom and they do it with the wrong (free-function) ergonomics.

## Concrete recommendations (priority order)

1. **Add `Streamer` companions for every batch DSP block** — `StftStreamer`, `YinTracker`, `MpmTracker`, `SpectralFluxTracker`, `SuperFluxTracker`, `ComplexDomainTracker`, `EnergyVad`, `NoiseTracker`, `BeatTracker`. Use `changepoint.Bocpd` as the template (`New(cfg) -> *T`, `Update`/`Push`/`Process` per call, query methods, documented concurrency).

2. **Unify pitch return type**: `type Pitch struct { Hz, Confidence float64; Voiced bool }`. Replace `(0,1)` / `(0,0)` / bare `0` with `Pitch{Voiced:false}`. Pin polarity ("higher confidence = better") in one place.

3. **Unify onset return type**: `type Onset struct { Frame int; Strength float64 }`. Mirror `beat.Beat`. Frees consumers from the "is the higher-numbered frame the stronger one?" question.

4. **Unify segment coordinate convention**: every `Segment` carries `Frame` *and* `Sample` (or a typed enum of which). Today same struct means different things in two `segmentation` files (`vad_based.go` vs `onset_offset.go`).

5. **Add float32 path for the four mobile-port-bound functions** (`MelFilterbank`, `PowerSpectrum`, `MFCC`, `LogMelEnergies`). Document the dtype contract in `audio/doc.go`.

6. **Replace `[]complex128` with paired-`[]float64`** for STFT row storage to enable SIMD and halve the memory footprint (parallel-arrays Re/Im like `signal.FFT` already uses).

7. **Make per-frame pitch alloc-free**: add `YinInto(frame, scratch1, scratch2 ..., out *Pitch)` zero-alloc forms mirroring `separation.WienerFilterInto`.

8. **Pick one error contract package-wide**: either panic (and document why) or sentinel `error` (and convert audio/, pitch/, onset/, spectrogram/, segmentation/). Today there are two split by file age. Recommend `error` because of cross-substrate ports — Kotlin/Swift/TS don't have `panic` semantics.

9. **`Fingerprint.Reset` method** (or free function, matching the package idiom).

10. **Document hop-size and frame-anchor convention in `audio/doc.go`** — left-anchored, zero-pad trailing, hop ≤ frameSize, what "frame index" means in each return type.

## Sources
- C:/limitless/foundation/reality/audio/doc.go (full)
- C:/limitless/foundation/reality/audio/mfcc.go:65, 130-145 (FrameMFCC pre-FFT requirement)
- C:/limitless/foundation/reality/audio/melscale.go:81-154 (MelFilterbank tmp allocs)
- C:/limitless/foundation/reality/audio/degradation.go:1-200 (stateful tracker — free-function idiom)
- C:/limitless/foundation/reality/audio/fingerprint.go:1-256 (no Reset method)
- C:/limitless/foundation/reality/audio/spectrogram/stft.go:53-100 (batch STFT, [][]complex128 alloc)
- C:/limitless/foundation/reality/audio/spectrogram/magnitude.go:1-136 (per-cell allocators)
- C:/limitless/foundation/reality/audio/spectrogram/mel_spectrogram.go:42-93 (filterbank rebuilt every call)
- C:/limitless/foundation/reality/audio/pitch/yin.go:61, 110, 121 (per-frame mallocs; (0,1) sentinel)
- C:/limitless/foundation/reality/audio/pitch/mpm.go:61, 100 ((0,0) sentinel; per-frame malloc)
- C:/limitless/foundation/reality/audio/pitch/autocorrelation.go:49 (single-return; no confidence)
- C:/limitless/foundation/reality/audio/pitch/subharmonic_summation.go:51 (single-return; no confidence)
- C:/limitless/foundation/reality/audio/onset/{energy,spectral_flux,complex_domain,superflux,peak_picking}.go (all batch, all `[][]complex128` consumers)
- C:/limitless/foundation/reality/audio/separation/{spectral_subtraction,wiener}.go (good `*Into` pattern)
- C:/limitless/foundation/reality/audio/separation/vad.go (per-frame scalar, no `EnergyVad` struct)
- C:/limitless/foundation/reality/audio/cqt/cqt.go:100-201 (sentinel-error pattern, the right newer style)
- C:/limitless/foundation/reality/audio/tempo/tempo.go:40, 136-153 (sentinel-error pattern)
- C:/limitless/foundation/reality/audio/beat/beat.go:8-13, 52, 155-164 (Beat struct = right shape; sentinel errors)
- C:/limitless/foundation/reality/audio/segmentation/vad_based.go:50, 72-104 (sample coords; in-call state machine that should be a Streamer)
- C:/limitless/foundation/reality/audio/segmentation/onset_offset.go:43-55 (frame coords — *different* convention from vad_based)
- C:/limitless/foundation/reality/changepoint/bocpd.go:85, 121-143, 175 (the in-house streaming idiom audio should clone)
- C:/limitless/foundation/reality/signal/filter.go:19-100 (sibling: caller-buffer + panic style)

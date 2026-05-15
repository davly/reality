# 005 — acoustics: hot-path allocations and vectorization

## Headline
The acoustics package is allocation-clean by virtue of being scalar-only — but that scalarity is itself the perf problem: every per-sample/per-bin function (AWeighting, DecibelSPL, DecibelFromIntensity, DopplerShift, WaveLength) is missing the `(in, out []float64)` slice variant that signal/, audio/melscale, and audio/mfcc all expose, forcing 60-FPS callers like Pistachio to write hot loops that the library could SIMD-friendly-ize once.

## Hot path candidates (functions called per-frame/per-sample)
Single source file: `C:/limitless/foundation/reality/acoustics/acoustics.go` (197 LOC, 9 exported funcs).

| function | line | callsite cadence (Pistachio @ 60 FPS / 48 kHz) |
|---|---|---|
| `AWeighting(f)` | :188 | per-FFT-bin: 257 bins × 60 fps ≈ 15.4k calls/s, **or** per-sample if applied as a filter (48k calls/s) |
| `DecibelSPL(p, pRef)` | :65 | per-sample for VU/SLM metering: 48k calls/s |
| `DecibelFromIntensity(I, IRef)` | :81 | per-sample/per-bin: same order |
| `DopplerShift(f0, vs, vr, c)` | :128 | per-source-per-frame: N_sources × 60 fps |
| `WaveLength(f, c)` | :164 | per-bin (e.g. for HRTF lookup): 257 × 60 fps |
| `SoundIntensity(P, r)` | :45 | per-source-per-frame |
| `SoundSpeed(γ, R, T, M)` | :29 | once per environment change (cold) |
| `SabineRT60(V, A)` | :101 | once per room (cold) |
| `ResonantFrequency(L, n, c)` | :149 | once per resonator-init (cold) |

The first five are realistically inside an audio frame loop. The last four are room/environment setup that is called once and cached.

## Allocation audit
| function | allocates? | how/why | severity |
|---|---|---|---|
| `SoundSpeed` | no | one `math.Sqrt`, three muls, one div, all on stack-resident float64 | clean |
| `SoundIntensity` | no | three muls, one div | clean |
| `DecibelSPL` | no | one `math.Log10`, one mul, one div | clean |
| `DecibelFromIntensity` | no | identical pattern to DecibelSPL | clean |
| `SabineRT60` | no | one mul, one div | clean |
| `DopplerShift` | no | two adds, one mul, one div | clean |
| `ResonantFrequency` | no | one `int→float64`, two muls, one div | clean |
| `WaveLength` | no | one div | clean |
| `AWeighting` | no | seven local float64 vars; `math.Sqrt` and `math.Log10` only; no slice/struct/closure | clean |

Net: zero `make`, zero `append`, zero closures, no interface boxing, no defer, no map, no slice literal anywhere in the package. The Grep for `make\(\[\]\|append\(` returned no matches across `acoustics/`. This is allocation-perfect at the scalar level — better than `signal/MedianFilter` which falls back to `make([]float64, n)` for windows >64 (`signal/filter.go:162`).

## Vectorization opportunities
The package's gap is not allocations — it's that **every hot-path function is scalar-only**, which forces every consumer to write the loop. Compare to `signal/` and `audio/melscale` where slice variants exist beside (or instead of) the scalars.

Concrete missing variants:

- **`AWeightingDBSlice(out, freqs []float64)`** — A-weighting is the canonical batch op. Pistachio applying it to an FFT magnitude array of 257 bins per frame will do 257 separate `AWeighting(f)` calls. A slice variant lets the compiler hoist the literal-constant adds (`f² + 20.6²`, etc.) and unroll. It also lets a future SIMD/AVX backend slot in without changing callers.
- **`DecibelSPLSlice(out, p []float64, pRef float64)`** and **`DecibelFromIntensitySlice(out, I []float64, IRef float64)`** — both are pure `20·log10(p/pRef)` / `10·log10(I/IRef)` per element. With pRef hoisted out, the inner loop is one `Log10` per element. Per-sample VU meters need this.
- **`DopplerShiftSlice(out, f0 []float64, vs, vr, c float64)`** — when N tracked sources update in lock-step, one slice call removes N scalar dispatches.
- **`WaveLengthSlice(out, f []float64, c float64)`** — bin-array of frequencies → bin-array of wavelengths in one pass; relevant for HRTF/diffraction code that wants λ per bin.
- **`AWeightingFilter`** — A-weighting is normally implemented as a *time-domain biquad cascade* (IEC 61672-1 Annex E, two 2nd-order sections), not as a per-bin frequency lookup. The biquad form is allocation-free, stateful, and applies sample-by-sample. The current `AWeighting(f)` function gives the *spectral curve*, which is useful but not how real metering chains use it. A `BiquadAWeighting(state *Biquad2, in, out []float64)` API would match what Pistachio actually needs.

Micro-opts inside existing scalar functions (low value, listed for completeness):

- `AWeighting` (acoustics.go:188): the four constants `20.6²`, `12194²`, `107.7²`, `737.9²` are recomputed at every call (`12194.0 * 12194.0`, etc.). The Go compiler folds these — `go build -gcflags=-m` shows constant folding — so this is a non-issue *for now*, but if the literals were ever pulled to package-level `const` they would be guaranteed constants regardless of compiler version. Recommend `const fA1, fA2, fA3, fA4 = 20.6, 107.7, 737.9, 12194.0` plus `fA1Sq = fA1*fA1` etc. Cosmetic, zero perf delta on current toolchain.
- `SoundIntensity` (acoustics.go:45): `4.0 * math.Pi` is also folded today; same comment.
- No `math.Pow` exists in acoustics — checked and confirmed (Grep `math\.Pow` → no matches). The single `math.Sqrt` in `AWeighting` is correct (sqrt of a known-positive product) and not replaceable by `x*x*x` shortcut.
- No repeated trig: `math.Cos`/`Sin` are absent from acoustics. (signal/window.go does the trig per-window-init; that's already factored.)

## Comparison with signal/ discipline
The signal/ package's discipline is essentially: "every operation that touches more than one number takes an `out` buffer, panics on size mismatch, and documents 'Zero heap allocations'." Examples:

- `signal.HannWindow(n int, out []float64)` — `signal/window.go:15`
- `signal.Convolve(signal, kernel, out []float64)` — `signal/filter.go:19`
- `signal.ExponentialMovingAverage(signal []float64, alpha float64, out []float64)` — `signal/filter.go:97`
- `signal.PowerSpectrum(real, imag, out []float64)` — `signal/fft.go:140`

The audio/ package follows the same convention: `MelFilterbank(... out []float64)` writes a row-major matrix into a caller buffer (`audio/melscale.go:44`).

acoustics breaks this convention not by allocating, but by **not exposing any slice surface at all**. Every function is `(scalars) → scalar`. This is fine for cold setup (SabineRT60, SoundSpeed) but mismatched with the "Pistachio @ 60 FPS" use case the design doc cites. The package would be more consistent with the rest of the library if the per-sample/per-bin functions had a sibling slice variant.

Note also `signal/fft.go:1-10` and `signal/filter.go:13-15` advertise "zero allocations in hot paths" in the package-level and per-function docs. acoustics' package doc (`acoustics.go:1-7`) makes no such promise — neither claims nor tests for it.

## Concrete recommendations
1. **Add `AWeightingDB(f) float64` as the rename**, deprecate `AWeighting(f)` (per agent 004's API recommendation), then add a slice variant `AWeightingDBSlice(out, freqs []float64)` with mandatory length-equality panic. Ship the scalar and slice forms together so consumers never have to choose between API ergonomics and allocation discipline.
2. **Add `DecibelSPLSlice(out, p []float64, pRef float64)` and `DecibelFromIntensitySlice(out, I []float64, IRef float64)`** — both are 4-line implementations that match `signal.ApplyWindow`'s style and are exactly what per-sample meters need.
3. **Add a `BiquadAWeighting` (or `WeightingFilter`) state struct + `Apply(in, out []float64)` method** for the time-domain A-weighting cascade. This is the form Pistachio will actually use; the spectral `AWeighting(f)` is a curve-only helper. Reference: IEC 61672-1 Annex E gives the analog prototype; a bilinear-transformed digital biquad cascade is the standard implementation.
4. **Promote magic constants to package-level `const`** in `AWeighting` (`20.6`, `107.7`, `737.9`, `12194.0` plus their squares). Cost: zero. Benefit: stable across Go toolchain versions, easier to test against IEC 61672-1.
5. **Add `BenchmarkAWeighting`, `BenchmarkDecibelSPL`, `BenchmarkDopplerShift`** scalar benchmarks plus `BenchmarkAWeightingDBSlice_257` (FFT-bin-sized) so future regressions are caught. Currently the package has zero benchmark functions (`Grep Benchmark` in `acoustics/` → no matches).
6. **Add a package-doc claim "Zero heap allocations in hot paths"** matching `signal/fft.go:1-10`, then enforce with `testing.AllocsPerRun` in the benchmark file. This makes the design-doc invariant queryable from code.
7. **Pull `0.161` in `SabineRT60` to a derived constant** (already covered by agent 001 finding) — perf-neutral, but keeping numeric and perf cleanups in one PR is cheap.

None of these recommendations require breaking changes. (1) and (3) are pure additions; (2), (4), (5), (6), (7) are additive or internal.

## Sources
- `C:/limitless/foundation/reality/acoustics/acoustics.go` (197 lines, all 9 functions)
- `C:/limitless/foundation/reality/acoustics/acoustics_test.go` (zero benchmarks)
- `C:/limitless/foundation/reality/signal/fft.go:1-181` (zero-alloc package-doc and FFT pattern)
- `C:/limitless/foundation/reality/signal/filter.go:1-174` (Convolve/MovingAverage/EMA/MedianFilter `out`-buffer pattern)
- `C:/limitless/foundation/reality/signal/window.go:1-114` (Hann/Hamming/Blackman/ApplyWindow `out`-buffer pattern)
- `C:/limitless/foundation/reality/audio/melscale.go:1-50` (sibling-package convention: scalar HzToMel + slice MelFilterbank)
- `C:/limitless/foundation/reality/CLAUDE.md` "No allocations in hot paths. Functions accept output buffers. Pistachio calls these at 60 FPS."
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/004-acoustics-api.md` (related API findings; AWeighting → AWeightingDB rename)
- IEC 61672-1:2013 "Electroacoustics — Sound level meters", Annex E (A-weighting analog prototype, basis for digital biquad cascade)

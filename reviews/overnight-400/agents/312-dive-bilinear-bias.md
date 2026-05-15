# 312 — dive-bilinear-bias (Bilinear / Bicubic / Mitchell-Netravali / Lanczos / Catmull-Rom audit)

## Headline
Reality has a 1-D scalar interpolation toolkit (Lerp, CubicSplineNatural, Catmull-Rom, Bezier) but ZERO image-domain reconstruction kernels (no Mitchell-Netravali, no Lanczos, no separable 2-D resampler); the one place that needs it — `audio/spectrogram/visualise.go` — claims "bilinearly resampled" in its docstring but actually does nearest-neighbor index truncation, a documentation/implementation bug worth fixing today.

## Findings

### What exists in reality (audit)
- `optim/interpolate.go:18` — `LinearInterpolate(x0, y0, x1, y1, x)` — non-parametric 2-point lerp on (x,y) pairs.
- `optim/interpolate.go:44` — `CubicSplineNatural(xs, ys)` — natural cubic spline via Thomas algorithm; closure-form evaluator with binary search; clamps outside domain. Ref: Burden & Faires.
- `geometry/curves.go:15` — `LinearInterpolate(a, b, t)` — parametric lerp on `t ∈ [0,1]`.
- `geometry/curves.go:28` — `BezierCubic(p0,p1,p2,p3,t)` — scalar cubic Bézier.
- `geometry/curves.go:40` — `BezierCubic3D(...)` — 3-D cubic Bézier.
- `geometry/curves.go:68` — `CatmullRom(p0,p1,p2,p3,t)` — uniform Catmull-Rom (τ=0.5), scalar only. This IS Mitchell-Netravali at (B=0, C=0.5) — but evaluated as a 4-point parametric curve, not as a separable image kernel `k(x)`.
- `geometry/quaternion.go:92` — `QuatSlerp` — spherical lerp for unit quaternions; falls back to nlerp near parallel.
- `optim/transport/wasserstein1d.go`, `optim/transport/iqr_norm.go` — quantile-based linear interpolation on sorted samples (statistical, not signal-domain).

### What is MISSING (signal/image kernels)
- No `Bilinear(...)` 2-D interpolant. No `Bicubic(...)`. No `MitchellNetravali(B, C, x)`. No `Lanczos(a, x)`. No `Sinc(x)` (despite the existing `signal/window.go` Hann/Hamming/Blackman family — sinc is the natural sibling).
- No `NearestNeighbor(...)` (trivial but must exist for completeness; it's the reference baseline against which all other kernels are compared in Heckbert 1986).
- No `CubicHermite` / `CubicConvolution` (Keys 1981) — even though Catmull-Rom IS Keys with α=−0.5, it's not exposed under the kernel name an image-resample consumer expects.
- No `BSplineCubic` reconstruction filter (the (B=1, C=0) point of Mitchell-Netravali; smoother than Catmull-Rom, more blur).
- No 1-D resample primitive `Resample(in, outLen, kernel)` and no separable 2-D resampler `Resample2D(in, outW, outH, kernel)`. Without these, every consumer (Pistachio image preview, spectrogram heatmaps, color LUT eval) reinvents an ad-hoc resampler.

### Implementation bug found
`audio/spectrogram/visualise.go:34` docstring: *"The matrix is bilinearly resampled to fit the requested width/height."* The actual code at lines 102–135 does:
```go
fIdx = int(float64(height-1-y) / float64(height-1) * float64(F-1))   // truncation, no fraction
tIdx = int(float64(x) / float64(width-1) * float64(T-1))             // truncation, no fraction
v := matrix[tIdx][fIdx]                                              // nearest cell
```
This is **nearest-neighbor**, not bilinear. There is no fractional weighting, no four-pixel average, no anti-aliasing on downsample. For typical mel-spectrogram render (T=216 frames into width=1920, F=128 bins into height=1080), upscale ratio is ~9× horizontally and ~8× vertically — the resulting image is visibly blocky on every spectral edge. Either fix the impl to true bilinear or fix the doc to "nearest-neighbor sampled".

### Bilinear bias / kernel tradeoffs (canon, for the Sources section)
| Kernel | Support | Sharpness | Ringing | Blur | Notes |
|---|---|---|---|---|---|
| Nearest | 1 | jagged | none | none | reference |
| Bilinear (tent) | 2 | low | none | high | "bilinear bias": four-pixel average |
| Cubic-B-spline (B=1,C=0) | 4 | low | minimal | very high | Mitchell-Netravali at (1,0) |
| Catmull-Rom (B=0,C=0.5) | 4 | high | mild | low | Keys 1981 cubic-conv α=-0.5 |
| Mitchell (B=1/3,C=1/3) | 4 | balanced | balanced | balanced | MN'88 subjective optimum |
| Lanczos-2 | 4 | high | <1% | low | windowed sinc, 2 lobes |
| Lanczos-3 | 6 | highest | visible | minimal | Blinn: best (achievable) low-pass |

Mitchell-Netravali constrain B+2C=1; the (B=0,C=0.5) point already lives in `geometry/curves.go:68` under a different name. So the parametric BC-spline kernel is a **5-line generalisation of CatmullRom**, not net-new math. See Mitchell & Netravali 1988 §3 (the BC piecewise-cubic formula) and `pbr-book.org` §8.8.

## Concrete recommendations

### Day-1 PR (cheapest single landing)
1. **Create `signal/interp.go`** (~250–350 LOC) exposing image-kernel reconstruction primitives, separately from the curve/spline math in `geometry` and `optim`:
   - `Sinc(x float64) float64` (normalised: `sinc(0)=1`, `sinc(x)=sin(πx)/(πx)`).
   - `KernelLinear(x)` / `KernelTent(x)` — `max(0, 1-|x|)`. Support 1.
   - `KernelCubicHermite(x)` — Keys 1981 with α=-0.5. Support 2.
   - `KernelMitchellNetravali(B, C, x)` — full BC-spline, support 2; doc lists named instances:
     - `(B=1, C=0)` cubic B-spline
     - `(B=0, C=0.5)` Catmull-Rom (≡ existing `geometry.CatmullRom` evaluated as kernel)
     - `(B=1/3, C=1/3)` Mitchell default
     - `(B=0, C=0)` Duff cubic
   - `KernelLanczos(a int, x float64)` — `sinc(x) * sinc(x/a)` for `|x| < a`, else 0. Support `a` (a=2 or a=3).
   - `Resample1D(in []float64, outLen int, kernel Kernel) []float64` — generic 1-D resampler with arbitrary kernel; handles upscale and downscale (downscale: scale kernel by ratio to satisfy Nyquist).
   - `Resample2DSeparable(in [][]float64, outW, outH int, kernel Kernel)` — applies 1-D kernel along x, then along y (Heckbert 1986 separability).
2. **Fix `audio/spectrogram/visualise.go`**: replace the truncation-sampling loop with `Resample2DSeparable(..., KernelLinear)` and update the docstring to match. Land same PR.
3. **Add `signal.Kernel` type**: `type Kernel struct { Eval func(float64) float64; Support float64 }` so any caller can supply its own (e.g. Pistachio audio: Lanczos-3 for SRC; image preview: Mitchell).

### Tier plan beyond day-1
- **T0** (day-1): Sinc, Linear, CubicHermite, MitchellNetravali, Lanczos kernels + 1-D `Resample1D` + separable 2-D.
- **T1**: Image-domain entry points: `signal.ResizeImageF64(in, outW, outH, kernel)` accepting `[][]float64`; later wrap stdlib `image.Image` in a higher-layer helper.
- **T2**: Pre-filter for downsample (Heckbert 1986 §3): scale kernel by `max(1, inSize/outSize)` to satisfy Nyquist before sampling. Without this, Lanczos-3 still aliases on >2× downscale.
- **T3**: Hermite, Bell, B-spline of higher order, Magic-Kernel-Sharp (Costella) as additional `Kernel` instances.
- **T4**: ITU-R BT.601/709 chroma resampling kernels (broadcast video provenance), NTSC kell-factor-aware filters.
- **T5**: 1-D `MonotoneCubicHermite` (Fritsch-Carlson 1980) — the missing non-overshooting cousin to `CubicSplineNatural`; trader-charts and color-LUT consumers care more about monotonicity than C² smoothness.
- **T6** (frontier): Variational interpolation — Wahba 1990 thin-plate splines, Duchon 1979 polyharmonic splines, RBF-in-2-D — once color and chaos packages start needing scattered-point interp.
- **T7** (frontier): Ideal sinc reconstruction with Kaiser-window family parameterised by β, plus the Magic Kernel Sharp 2021 family.
- **T8** (frontier): True 2-D non-separable lattice (radial Lanczos, EWA Heckbert 1989) for anisotropic resampling — only Pistachio's GPU-style texture path will need this.

### Cross-language golden files (per CLAUDE.md design rule #1)
Reality's contract is golden-file vectors shared across Go/Python/C++/C#. For kernel functions:
- 30 vectors per kernel sampling `x ∈ [-3, 3]` at irrational offsets (avoid coincidental zeros).
- Edge cases: `x = 0`, `x = ±1`, `x = ±a` (Lanczos boundary), `x = NaN`, `x = ±Inf`, `x = -0.0`, subnormals.
- Tolerance: `1e-12` absolute (transcendental sin/π products); `0` for tent/linear (rational arithmetic).
- For `Resample1D` and `Resample2DSeparable`: deterministic test patterns — DC, ramp, impulse, single-frequency cosine — with known closed-form output for each kernel.

### R-MUTUAL-CROSS-VALIDATION 3/3 saturation pins
Three independent identities should be pinned in tests so any future refactor breaks loudly:
1. **Constant-input invariance**: `Resample1D(constArray, anyOutLen, anyKernel) == constArray[0]` for **every** kernel. Each kernel must integrate to 1 over its support to satisfy partition-of-unity; this test is the strongest single check that the kernel is normalised correctly.
2. **Catmull-Rom ≡ MitchellNetravali(B=0, C=0.5)** — algebraic identity. Both `geometry.CatmullRom` (parametric form) and `signal.KernelMitchellNetravali(0, 0.5, x)` (kernel form) must produce bit-equal interpolated values for the same 4-point input. This is a regression net for both modules simultaneously.
3. **Lanczos-3 of bandlimited signal**: `Resample1D(cos(2πfn/N), 2*N, Lanczos3)` then decimate by 2 → recovers the original up to ringing tolerance ε ≈ 1e-3 for `f < N/4`. This pins the windowed-sinc reconstruction property and detects normalisation bugs that the constant-input test misses (since ringing cancels for DC).

### Cross-link consumers
- **Pistachio** (largest consumer): image previews, spectrogram render, audio resample. Today reinvents bilinear loops; would adopt Mitchell + Lanczos-3 immediately if exposed.
- **`audio/spectrogram/visualise.go`**: docstring lies about bilinear; first internal consumer of the new kernel API.
- **`color`**: 1-D LUT evaluation (gamma, tone-mapping curves) needs `MonotoneCubicHermite` — currently has none, every consumer hand-rolls.
- **`signal`** (this package): audio sample-rate conversion (Pistachio voice-activity, RubberDuck STFT) is exactly Lanczos-3 1-D resample.
- **`calculus`**: 2-D function-approximation visualisation; today `NumericalGradient` produces points but nothing renders them smoothly.
- **`prob/nonparametric.go`**: KDE and quantile estimators today rely on `optim/transport`'s linear interpolation; offer them the cubic-spline alternative.

## Singular cheapest day-1 PR
`signal/interp.go` (~280 LOC) + `signal/interp_test.go` (~250 LOC) + golden vectors in `signal/testdata/interp/*.json` (~6 files × 30 vectors each). Wire `audio/spectrogram/visualise.go` to use `signal.Resample2DSeparable` in the same PR — converts an existing latent bug fix into a feature delivery. Net diff: +700 LOC, -25 LOC.

## Sources

### Repo files (file:line)
- `C:/limitless/foundation/reality/optim/interpolate.go:18` (LinearInterpolate)
- `C:/limitless/foundation/reality/optim/interpolate.go:44` (CubicSplineNatural)
- `C:/limitless/foundation/reality/geometry/curves.go:15` (LinearInterpolate parametric)
- `C:/limitless/foundation/reality/geometry/curves.go:28` (BezierCubic)
- `C:/limitless/foundation/reality/geometry/curves.go:40` (BezierCubic3D)
- `C:/limitless/foundation/reality/geometry/curves.go:68` (CatmullRom — uniform τ=0.5, equiv. Mitchell-Netravali B=0,C=0.5)
- `C:/limitless/foundation/reality/geometry/quaternion.go:92` (QuatSlerp)
- `C:/limitless/foundation/reality/audio/spectrogram/visualise.go:34` (incorrect "bilinearly resampled" docstring; impl at lines 102–135 is nearest-neighbor)
- `C:/limitless/foundation/reality/signal/window.go` (Hann/Hamming/Blackman; sinc-window-family belongs here)
- `C:/limitless/foundation/reality/signal/filter.go:19` (Convolve — already the building block for any FIR-kernel resampler)
- `C:/limitless/foundation/reality/optim/transport/iqr_norm.go:81` (linear quantile interp; T5 monotone-cubic candidate)

### Web / canonical references
- Mitchell, D. P., & Netravali, A. N. (1988). *Reconstruction Filters in Computer Graphics.* SIGGRAPH '88. ACM 0-89791-275-6/88/008/0221. — BC-spline family, B+2C=1 line, B=C=1/3 subjective optimum. https://www.cs.utexas.edu/~fussell/courses/cs384g-fall2013/lectures/mitchell/Mitchell.pdf
- Mitchell-Netravali filters — Wikipedia summary table of named (B,C) instances. https://en.wikipedia.org/wiki/Mitchell%E2%80%93Netravali_filters
- Lanczos resampling — Wikipedia. Definition `L(x) = sinc(x)·sinc(x/a)`; Blinn quote "rejects high freq better than any (achievable) filter we've seen so far"; ringing < 1% at a=2. https://en.wikipedia.org/wiki/Lanczos_resampling
- Turkowski, K. (1990). *Filters for Common Resampling Tasks*, Apple Computer. — practical kernel comparison for Lanczos, Mitchell, Catmull-Rom. https://cadxfem.org/inf/ResamplingFilters.pdf
- Keys, R. G. (1981). *Cubic Convolution Interpolation for Digital Image Processing.* IEEE Trans. ASSP-29(6). — α=-0.5 (= Catmull-Rom kernel form).
- Heckbert, P. S. (1986). *Survey of Texture Mapping.* IEEE CG&A 6(11). — separability of 2-D filters; pre-filter for downsample.
- Duchon, J. (1976). *Interpolation des fonctions de deux variables suivant le principe de la flexion des plaques minces.* — variational thin-plate / polyharmonic splines (T6 frontier).
- Wolberg, G. (1990). *Digital Image Warping.* IEEE CS Press. — textbook reference for kernel zoo.
- Mitchell-Netravali in PBR-Book §8.8 — modern teaching reference. https://www.pbr-book.org/4ed/Sampling_and_Reconstruction/Image_Reconstruction
- Costella, J. (2021–). *The Magic Kernel Sharp.* — modern alternative claiming Lanczos quality without ringing. https://johncostella.com/magic/
- ImageMagick filter comparison page — empirical tradeoff matrix. https://legacy.imagemagick.org/Usage/filter/
- MATLAB `imresize` kernel comparison docs — bicubic vs Lanczos-2/3 vs Mitchell visual results. https://www.mathworks.com/help/matlab/creating_plots/create-and-compare-resizing-interpolation-kernels.html

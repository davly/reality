# 158 | synergy-color-signal

**Summary line 1.** `color/` and `signal/` are siblings under `reality/` that today do NOT import each other and do NOT share a single image-shaped primitive — `color/` exposes scalar-triple `(r,g,b)` / `(X,Y,Z)` / `(L,a,b)` conversions and per-pixel ΔE metrics with no slice/buffer API and no notion of an image plane, while `signal/` exposes 1-D `[]float64` FFT/Convolve/MovingAverage/MedianFilter/HannWindow with no notion of a 2-D image, no separable filter helper, no Gaussian kernel builder, no bilateral / guided / domain-transform filter, and no Bayer-mosaic primitive; the entire image-as-signal canon (linear-light blur, perceptual-space filtering, edge-preserving smoothing, demosaicing, chroma subsampling, tone mapping with proper luminance extraction, retinex, white balance, color quantisation) is wholly absent and there is **no `image` sub-package** to host it.

**Summary line 2.** Eighteen synergy primitives totalling ~1820 LOC of pure glue close the gap; eleven ship today against the v0.10.0 surfaces (every primitive that wraps `signal.Convolve` + `color.SRGBToLinear`/`LinearToSRGB`); seven are blocked on missing primitives (`signal.Convolve2DSeparable`, `signal.GaussianKernel1D`, `linalg.kmeans`, an `image.Plane` type) that are either independently flagged in agent 097-linalg-missing / 032-color-missing or are in-scope additions to `signal/`; cheapest one-day standalone is **C1 LinearLightBoxBlur + C2 LinearLightGaussianBlur** at ~120 LOC fixing the single most-reported "naïve sRGB blur produces dark fringes" bug; keystone is **C0 Plane (image-shaped float64 buffer)** because every other primitive in the grid consumes it; recommended placement is a NEW package `image/` (mirrors the 151/153/156/157 placement precedent of "synergy lives in a consumer-shaped sub-package, not in the primitive-supplier package") because color-space-aware image ops are neither pure-color (they need 2-D convolution) nor pure-signal (they need ΔE / Lab / linear-sRGB).

---

## 0. State of play (verified file-walk)

`color/` HEAD (4 files, ~470 LOC numeric core):

- `spaces.go`: `SRGBToLinear`, `LinearToSRGB` (scalar), `LinearRGBToXYZ`/`XYZToLinearRGB` (D65 only), `XYZToLab`/`LabToXYZ` (caller supplies white point), `RGBToHSV`/`HSVToRGB`
- `difference.go`: `DeltaE76`, `DeltaE2000` (full Sharma 2005 implementation)
- `adapt.go`: `BradfordAdapt(X,Y,Z, srcX,srcY, dstX,dstY)` — single white-point CAT
- `spectral.go`: `BlackbodyToXYZ(T)` (Planck × CIE 1931 2°), `ToneMapReinhard(r,g,b, whitePoint)` (per-channel)

**Search for image-shaped primitives in color/:** `Image`, `Plane`, `Bayer`, `Demosaic`, `Filter`, `Blur`, `Bilateral`, `Retinex`, `WhiteBalance`, `Quantize`, `Subsampl`, `Pixel`, `[]float64` — **zero matches** in `color/*.go`. Every function takes scalars, returns scalars.

`signal/` HEAD (3 files, ~470 LOC):

- `fft.go`: `FFT`, `IFFT`, `PowerSpectrum`, `FFTFrequencies` — 1-D, in-place, real+imag split, power-of-two only
- `filter.go`: `Convolve` (1-D direct, O(N·M)), `MovingAverage`, `ExponentialMovingAverage`, `MedianFilter` — all 1-D
- `window.go`: `HannWindow`, `HammingWindow`, `BlackmanWindow`, `ApplyWindow` — all 1-D

**Search for image-shaped primitives in signal/:** `Convolve2D`, `Separable`, `Gaussian`, `Bilateral`, `Guided`, `Bayer`, `FFT2D`, `2D` — **zero matches** in `signal/*.go`. Package is explicitly 1-D throughout. The doc comment lists consumers as "Pistachio (audio), RubberDuck (spectral analysis), Oracle (time series), Sentinel (filtering)" — image processing is conspicuously absent from the consumer list.

**The bridge primitive `Image / Plane / [][]float64-row-major`:** does not exist anywhere in the repo. `linalg/` has dense matrices as `[]float64` row-major but they are mathematical matrices, not images, and no image-aware primitive consumes them.

**Cycle-hazard check.** Neither `color/` nor `signal/` imports the other today (`grep -r color signal/*.go` and `grep -r signal color/*.go`: zero matches). Adding a third sub-package `image/` that imports both is the only cycle-free placement.

---

## 1. The eighteen synergy primitives

Each entry: (1) capability, (2) composition recipe over present primitives, (3) connective-tissue LOC, (4) blocking flag if any. Numbering C0–C17.

### C0 — `image.Plane` and `image.RGBPlanes` types

**Capability.** Single canonical image type for the package: `Plane{W, H int; Data []float64}` (W·H length, row-major) plus the helper `RGBPlanes{R, G, B Plane}` for separate-plane storage (better for cache locality in plane-wise ops than interleaved `[][]float64`). Plus a `LabPlanes`, `YCbCrPlanes`, `LinearRGBPlanes` family of named types so the type system distinguishes "this buffer is sRGB-encoded" from "this buffer is linear-light" — preventing the #1 bug class enumerated below (C1).

**Composition.** Pure type definitions, no math. `Plane.At(x,y)` / `Plane.Set(x,y,v)` accessors with row-major indexing. ~50 LOC.

**LOC.** 50.

**Notes.** This is the keystone-of-the-keystone. Every other primitive consumes Plane or RGBPlanes. Place in new file `image/plane.go`.

### C1 — `image.LinearLightBoxBlur(in RGBPlanes, radius int, out RGBPlanes)` and `image.LinearLightGaussianBlur`

**Capability.** Box and Gaussian blur in **linear-light** space (not gamma-encoded sRGB). The single most-reported color-space bug in image processing: blurring sRGB `[0.5, 0, 0]` red and `[0, 0.5, 0]` green naïvely averages to `[0.25, 0.25, 0]` muddy olive at the boundary, but in linear light the average is `(0.214 + 0)/2 → encode → 0.367` per channel which encodes back to a vivid mid-yellow. Naïve gamma-space blur produces visible dark fringes around bright/dark transitions and visible hue shifts on saturated edges. Foundational fix.

**Composition.** Per-pixel `color.SRGBToLinear` on each input channel → 2-D box/Gaussian convolution (using a separable-filter helper, see C16) → per-pixel `color.LinearToSRGB` on each output channel. Box variant uses cumulative-sum for O(1)/pixel regardless of radius; Gaussian variant uses two 1-D passes (separable). ~70 LOC for box + ~50 LOC for Gaussian.

**LOC.** 120.

**Status.** SHIPS TODAY against v0.10.0 — `signal.Convolve` exists and works on 1-D rows/columns; the 2-D loop is local. Cleaner if `signal.Convolve2DSeparable` (C16) lands first.

**Notes.** Public type-level discipline: this function MUST take `RGBPlanes` (sRGB-encoded) and output `RGBPlanes`, with an internal `LinearRGBPlanes` intermediate. Type system prevents callers from accidentally passing already-linear data (which would gamma-decode twice).

### C2 — `image.LabBoxBlur(in RGBPlanes, radius int, out RGBPlanes)` and `image.LabGaussianBlur`

**Capability.** Box / Gaussian blur in **CIELab** perceptual space. Avoids hue shifts in saturated regions that linear-RGB blur still produces (C1 fixes the gamma issue but linear RGB still mixes red+green → yellow even though the perceptual midpoint is brown-grey). For most photographic images the difference between C1 and C2 is small; for synthetic images with saturated primaries it is dramatic.

**Composition.** Per-pixel sRGB → linear → XYZ → Lab via `color.SRGBToLinear` + `color.LinearRGBToXYZ` + `color.XYZToLab` → 2-D blur on each L,a,b plane independently → per-pixel inverse chain. ~80 LOC.

**LOC.** 80.

**Status.** SHIPS TODAY against v0.10.0.

**Notes.** Same type-discipline as C1: takes `RGBPlanes`, produces `RGBPlanes`, with internal `LabPlanes`. Reference white must be supplied or default to D65 from a new `color.D65 = [3]float64{0.95047, 1.0, 1.08883}` constant (independently flagged in agent 032 §T1.U1 illuminants).

### C3 — `image.OklabBlur(in RGBPlanes, radius int, out RGBPlanes)`

**Capability.** Same as C2 but in OKLab (Ottosson 2020). OKLab gives better hue uniformity than CIELab in the blue region (the famous CIELab "blue-purple twist" is not present in OKLab) and is CSS Color Module 4 mandated. Should ship as the **default** perceptual-blur primitive going forward.

**Composition.** Same as C2 but with OKLab conversion. Requires `color.SRGBToOKLab` / `color.OKLabToSRGB` (~25 LOC, agent 032 §T1.S1, BLOCKED on color package addition).

**LOC.** 30 wrapper, BLOCKED-SOFT on color package adding OKLab.

### C4 — `image.BilateralFilter(in RGBPlanes, sigmaSpatial, sigmaColor float64, out RGBPlanes)`

**Capability.** Tomasi-Manduchi 1998 edge-preserving smoothing. Output pixel = weighted average of nearby pixels where weight = `exp(-d²_spatial/2σ²_s) · exp(-d²_color/2σ²_c)`. The color distance `d_color` is the CRUCIAL choice — naïve implementations use Euclidean RGB which gives bad results on saturated edges; the perceptual choice is **CIEDE2000** (or ΔE_ab as a faster proxy) computed via `color.DeltaE2000`. This is the single most cited image-processing primitive after Gaussian blur and the canonical demonstration of color × signal synergy.

**Composition.** Per output pixel (i,j): for each (i+dx, j+dy) in window of radius `3σ_s`: compute spatial-weight `exp(-(dx²+dy²)/2σ²_s)`, compute color-weight `exp(-(ΔE2000(p_ij, p_kl)/σ_c)²)` via `color.DeltaE2000`, accumulate weighted average. O(N · σ_s²) naïve; O(N) with separable bilateral grid (Paris-Durand 2006) but that needs ~300 LOC and is deferred.

**LOC.** 90 (naïve direct form, suitable for σ_s ≤ 5).

**Status.** SHIPS TODAY against v0.10.0 — `color.DeltaE2000` is fully implemented, `signal.Convolve` is not directly used (this is a non-linear filter, not a convolution).

**Notes.** Type discipline: takes RGBPlanes, internal `LabPlanes` for the color-distance computation, output RGBPlanes. The σ_c parameter is in ΔE units (≈ 1.0 = "just noticeable difference" per CIEDE2000).

### C5 — `image.JointBilateralFilter(target, guide RGBPlanes, sigmaSpatial, sigmaColor float64, out RGBPlanes)`

**Capability.** Petschnigg-Agrawala-Hoppe-Cohen 2004: bilateral filter where the range kernel uses a SECOND image as the edge guide. Foundational for: flash/no-flash photography (smooth no-flash with flash as edge guide), depth super-resolution (smooth low-res depth with high-res RGB as guide), tone mapping post-processing. The ΔE color distance is computed on the guide, the weighted average is on the target.

**Composition.** Same as C4, but the color-weight is ΔE2000 between guide pixels, while the weighted-average accumulator runs on target pixels. ~100 LOC.

**LOC.** 100.

**Status.** SHIPS TODAY against v0.10.0.

### C6 — `image.GuidedFilter(in, guide Plane, radius int, epsilon float64, out Plane)`

**Capability.** He-Sun-Tang 2010/2013 (CVPR/PAMI): O(N) edge-preserving smoothing that approximates bilateral filter but is exact-linear-time regardless of radius. The output is a locally-linear function of the guide image. Fast enough for video. Industry standard for matting, dehazing, HDR detail enhancement.

**Composition.** Compute per-pixel local mean / variance / covariance of (guide, in) over box-radius windows using cumulative-sum (O(N) box filter). Compute local linear-coefficients a,b per window. Output = mean_a · guide + mean_b. Box-mean reuses the C1 cumulative-sum trick. ~140 LOC including the four cumulative-sum passes.

**LOC.** 140.

**Status.** SHIPS TODAY against v0.10.0 (only requires box-filter cumulative-sums, no convolution).

**Notes.** The "radius / epsilon" pair matches OpenCV's `ximgproc.guidedFilter` API conventions.

### C7 — `image.DomainTransformFilter(in RGBPlanes, sigmaSpatial, sigmaColor float64, out RGBPlanes)`

**Capability.** Gastal-Oliveira 2011 SIGGRAPH: O(N) edge-preserving smoothing via 1-D recursive filtering on a transformed domain that adaptively warps according to color gradients. Three variants in the paper (NC, IC, RF); Recursive Filter (RF) is the canonical fast implementation. Industry standard alongside guided filter for real-time video.

**Composition.** (1) Compute domain-transform per row using ΔE between adjacent pixels (`color.DeltaE2000` on adjacent Lab values, or ΔE_ab as faster proxy). (2) Recursive 1-D filter on rows with adaptive feedback coefficient `a^d_t` where `d_t` is the cumulative ΔE distance. (3) Repeat on columns. (4) Multi-pass with halving σ for better quality. ~180 LOC (most of the LOC is the four-pass recursive filter, which is a small extension to `signal.ExponentialMovingAverage` per-row).

**LOC.** 180.

**Status.** SHIPS TODAY against v0.10.0 — recursive filter is essentially `ExponentialMovingAverage` with a per-step alpha derived from ΔE.

### C8 — `image.BayerToRGBBilinear(in Plane, pattern BayerPattern, out RGBPlanes)`

**Capability.** Bilinear demosaicing of a Bayer-pattern raw sensor image (RGGB / BGGR / GRBG / GBRG patterns) into full RGB. The simplest demosaicing algorithm, suitable for documentation/reference. Each missing channel at each pixel = average of its 4 nearest neighbours.

**Composition.** Pure indexing + 4-neighbour average. No color-space conversion. ~100 LOC including the 4 pattern variants and edge handling.

**LOC.** 100.

**Status.** SHIPS TODAY against v0.10.0.

**Notes.** The CRUCIAL design choice is whether bilinear demosaic operates on **linear** sensor values or gamma-encoded values. Cameras output linear sensor values (the gamma encoding happens later in the ISP), so demosaic should be linear-domain by default — but the type system should make this explicit (`LinearRGBPlanes` output, not `RGBPlanes`). This is a recurring 2014-2020 stack-overflow argument that the type system can settle once and for all.

### C9 — `image.BayerToRGBMalvar(in Plane, pattern BayerPattern, out RGBPlanes)`

**Capability.** Malvar-He-Cutler 2004 (ICASSP): high-quality linear demosaicing using a 5×5 filter that accounts for inter-channel correlation. Substantially better than bilinear at edge regions. The Microsoft Research demosaicing filter cited in every textbook. OpenCV's `COLOR_BayerBG2RGB_EA` baseline.

**Composition.** 8 fixed 5×5 filter kernels (one per channel × pattern position) applied via `signal.Convolve` per-row + per-column (C16 separable helper). ~140 LOC including the 8 kernel definitions.

**LOC.** 140.

**Status.** SHIPS TODAY against v0.10.0 (C16 not strictly required; can use direct 2-D convolution loops at modest extra LOC).

### C10 — `image.BayerToRGBVNG(in Plane, pattern BayerPattern, out RGBPlanes)` and `image.BayerToRGBHamiltonAdams`

**Capability.** Variable Number of Gradients (Chang 1999) and Hamilton-Adams 1997: edge-aware adaptive demosaicing. VNG selects the gradient direction with smallest magnitude before averaging; Hamilton-Adams uses Laplacian correction on the green channel before red/blue interpolation. Both are non-linear. Production-quality demosaicing baselines. dcraw / RawTherapee defaults.

**Composition.** Per-pixel gradient computation using `signal.Convolve` with edge-detector kernels → per-pixel branch on smallest gradient → directional averaging. ~200 LOC each.

**LOC.** 400 (200 × 2).

**Status.** SHIPS TODAY against v0.10.0.

### C11 — `image.ChromaticAberrationCorrect(in RGBPlanes, redScale, blueScale float64, out RGBPlanes)`

**Capability.** Correct lateral chromatic aberration (LCA) by per-channel scaling around the optical center. Lens design produces wavelength-dependent magnification: red channel is slightly larger than green, blue slightly smaller. Uniform per-channel scale correction (the simplest model) takes two scalars; full correction uses radial polynomials (Adobe LCP file format).

**Composition.** Per-channel **signal-domain warping**: for each output pixel, compute source coordinate in the scaled coordinate system, sample via bilinear interpolation. Pure signal-domain operation per channel; the synergy is that color × signal compose cleanly because color is what makes channels separable while signal is what does the geometric warp. ~80 LOC.

**LOC.** 80.

**Status.** SHIPS TODAY against v0.10.0.

### C12 — `image.ToneMapReinhardLuminance(in RGBPlanes, whitePoint float64, out RGBPlanes)` and `image.ToneMapDrago` and `image.ToneMapMantiukLocal`

**Capability.** Reinhard, Drago, Mantiuk tone mapping operators viewed as **nonlinear filters with color-aware luminance extraction**. The crucial observation that `color.ToneMapReinhard` (which exists today) gets WRONG: it applies the tonemap per-channel, which desaturates and shifts hue. The correct algorithm extracts luminance Y = 0.2126·R + 0.7152·G + 0.0722·B (or the perceptual L* from CIELab), tonemaps Y → Y', and rescales the chromaticity (R/Y, G/Y, B/Y) by Y'/Y. This preserves hue and saturation.

Drago 2003 (Eurographics): adaptive logarithmic mapping with luminance-bias term. Mantiuk 2006 (SIGGRAPH): contrast-domain tone mapping with local edge-preserving smoothing — composes with C6/C7 guided/domain-transform filter.

**Composition.** C12.Reinhard: per-pixel luminance via `linalg.DotProduct` with [0.2126, 0.7152, 0.0722] → existing `color.ToneMapReinhard` on Y → rescale RGB by Y'/Y. ~50 LOC. C12.Drago: closed-form, ~80 LOC. C12.Mantiuk: requires C6 guided filter, ~150 LOC.

**LOC.** 50 + 80 + 150 = 280.

**Status.** SHIPS TODAY against v0.10.0 (C12.Mantiuk depends on C6 which itself ships today).

**Notes.** This is the canonical example of color × signal synergy: the existing per-channel `color.ToneMapReinhard` is what every textbook tells you NOT to do. The fix lives entirely in the synergy layer, not in `color/`. The flag-bug in agent 033 §T1.S? (color-sota tonemapping) is correctable here without touching `color/`.

### C13 — `image.YCbCrSubsample420(in RGBPlanes, out YCbCrPlanes)` and `image.DCTBlock8x8`

**Capability.** YCbCr conversion + 4:2:0 chroma subsampling (the JPEG / MPEG / H.264 / H.265 encoding pipeline) + 8×8 DCT block transform. Demonstrates **wavelet/DCT compression in YCbCr vs RGB**: the human eye is much less sensitive to chrominance high-frequency content than luminance, so subsampling Cb/Cr by 2× in each dimension and applying coarser quantisation is the foundational win of all modern image/video codecs.

**Composition.** RGB→YCbCr: 3×3 matrix multiply (Rec.601 or Rec.709 matrix). 4:2:0 subsample: 2×2 box average on Cb/Cr planes, leaving Y at full resolution. 8×8 DCT: separable 1-D DCT-II via `signal.FFT` on a length-16 padded buffer (the textbook DCT-via-FFT trick) OR direct 8×8 DCT-II with precomputed cosine table (faster, ~120 LOC). ~80 LOC for YCbCr + subsample, ~120 LOC for DCT block = 200 total.

**LOC.** 200.

**Status.** SHIPS TODAY against v0.10.0 (DCT-via-FFT path uses existing `signal.FFT`).

**Notes.** Cross-check: 8×8 DCT-via-FFT vs 8×8 direct-cosine-table must agree to 1e-12 — R-MUTUAL-CROSS-VALIDATION 2/3 candidate; third oracle is reference values from JPEG ITU-T T.81 Annex A.

### C14 — `image.ColorQuantizeKMeansLab(in RGBPlanes, k int, out RGBPlanes, palette []LabColor)`

**Capability.** k-means color quantization in **CIELab** space. Naïve k-means in RGB minimises Euclidean RGB distance which gives poor perceptual results (clusters split saturated regions and merge perceptually-distinct shadows). k-means in Lab (or OKLab) clusters by perceptual proximity. Standard pipeline: convert all pixels to Lab via `color.XYZToLab`, run Lloyd's algorithm, output palette + per-pixel index.

**Composition.** Per-pixel sRGB → linear → XYZ → Lab → call `linalg.KMeans(points, k, maxIter)` (which DOES NOT EXIST in linalg/ today, independently flagged in agent 097-linalg-missing §T1; ~120 LOC of Lloyd's algorithm) → write quantised Lab values back through inverse chain. ~50 LOC of glue around k-means.

**LOC.** 50, BLOCKED-HARD on `linalg.KMeans` (also blocks G8/G9 in agent 157 — shared keystone).

### C15 — `image.SubpixelTextRender(glyphMask Plane, fgRGB, bgRGB [3]float64, out RGBPlanes)`

**Capability.** ClearType-style RGB subpixel anti-aliasing: each LCD pixel has three stripes (R, G, B) so a glyph that is 1/3 of a pixel wide can be rendered using only the appropriate stripe. The algorithm: render glyph at 3× horizontal resolution, apply a 3-tap horizontal filter to each channel ([1/9, 2/9, 3/9] for the canonical Microsoft ClearType filter), output to RGB.

**Composition.** 3× horizontal supersample render → per-channel 3-tap convolution via `signal.Convolve` → RGB output. The color × signal synergy is that the SAME glyph mask is filtered with DIFFERENT 3-tap kernels per RGB channel, exploiting the spatial offset of LCD subpixels. ~80 LOC.

**LOC.** 80.

**Status.** SHIPS TODAY against v0.10.0.

**Notes.** Color-fringing-correction post-filter (the "T component" in Microsoft's ClearType patent) further desaturates by a factor proportional to the local color difference using `color.DeltaE76` — ~30 LOC extra.

### C16 — `signal.Convolve2DSeparable(in []float64, w, h int, kernelH, kernelV []float64, out []float64)` and `signal.GaussianKernel1D(sigma float64, out []float64)`

**Capability.** The two missing 1-D-to-2-D bridge primitives in `signal/` itself. `Convolve2DSeparable` does separable 2-D convolution by two 1-D passes (rows then columns, which is O(N·K) vs O(N·K²) for direct 2-D). `GaussianKernel1D` builds the 1-D Gaussian kernel that's needed by every Gaussian-blur primitive in this grid.

**Composition.** Pure signal-package additions, no color dependency. `Convolve2DSeparable` is ~60 LOC (two row-major passes calling existing `signal.Convolve` per row/column with edge handling — clamp / wrap / reflect). `GaussianKernel1D` is ~20 LOC (truncated normal at ±3σ, normalised to sum 1).

**LOC.** 80.

**Status.** SHIPS TODAY against v0.10.0 — pure `signal/` additions, no new dependencies.

**Notes.** These belong in `signal/`, not `image/`, because they are package-pure 1-D primitives lifted to 2-D by the canonical separable-convolution trick. Place in new file `signal/convolve2d.go`. Independently flagged in any signal-missing review (slot ~150–155 if a signal-missing exists; not searched).

### C17 — `image.MultiscaleRetinex(in RGBPlanes, sigmas []float64, out RGBPlanes)` and `image.GrayWorldWhiteBalance` / `image.RetinexWhiteBalance`

**Capability.** Multi-Scale Retinex (MSR, Jobson-Rahman-Woodell 1997) for dynamic-range compression and color constancy. Rahman 2004 MSRCR adds color restoration. Output_i = Σ_n w_n · (log(I_i) − log(I_i ⊛ G_{σ_n})) — i.e. log-domain difference between the image and its Gaussian-blurred version at multiple scales. Foundational HDR-tone-mapping / face-illumination-correction algorithm.

Gray-world white balance: scale each channel by the inverse of its mean (so the mean of all channels becomes equal). Retinex white balance: scale by the inverse of the channel max.

**Composition.** MSR: per-pixel `log(R), log(G), log(B)` → for each scale σ_n: 2-D Gaussian blur via C1 + C16 → log-difference → weighted sum across scales. ~120 LOC. White balance: ~30 LOC each, trivial channel-mean / channel-max / scale.

**LOC.** 120 + 60 = 180.

**Status.** SHIPS TODAY against v0.10.0 (MSR depends on C1/C16, both ship today).

---

## 2. Status table

| ID | Primitive | LOC | Status | Blockers |
|---|---|---:|---|---|
| C0 | image.Plane / RGBPlanes / LabPlanes / LinearRGBPlanes | 50 | SHIPS TODAY | none |
| C1 | LinearLightBoxBlur + LinearLightGaussianBlur | 120 | SHIPS TODAY | C0 |
| C2 | LabBoxBlur + LabGaussianBlur | 80 | SHIPS TODAY | C0, C1 |
| C3 | OklabBlur | 30 | BLOCKED-SOFT | color.OKLab (032 §T1.S1) |
| C4 | BilateralFilter (ΔE2000) | 90 | SHIPS TODAY | C0 |
| C5 | JointBilateralFilter | 100 | SHIPS TODAY | C0, C4 |
| C6 | GuidedFilter | 140 | SHIPS TODAY | C0 |
| C7 | DomainTransformFilter | 180 | SHIPS TODAY | C0 |
| C8 | BayerToRGBBilinear | 100 | SHIPS TODAY | C0 |
| C9 | BayerToRGBMalvar | 140 | SHIPS TODAY | C0, C16 |
| C10 | BayerToRGBVNG + HamiltonAdams | 400 | SHIPS TODAY | C0 |
| C11 | ChromaticAberrationCorrect | 80 | SHIPS TODAY | C0 |
| C12 | ToneMapReinhardLuminance + Drago + Mantiuk | 280 | SHIPS TODAY | C0 (Mantiuk needs C6) |
| C13 | YCbCrSubsample420 + DCTBlock8x8 | 200 | SHIPS TODAY | C0 |
| C14 | ColorQuantizeKMeansLab | 50 | BLOCKED-HARD | linalg.KMeans (097 §T1) |
| C15 | SubpixelTextRender | 80 | SHIPS TODAY | C0 |
| C16 | signal.Convolve2DSeparable + GaussianKernel1D | 80 | SHIPS TODAY | none |
| C17 | MultiscaleRetinex + WhiteBalance | 180 | SHIPS TODAY | C0, C1, C16 |

**Total connective tissue:** ~2380 LOC (revised from header — C13 is 200 not 150, C14 stays 50, C17 is 180 — ~1800 LOC of glue + ~580 of demosaic algorithms). **Of which ships today against v0.10.0:** all except C3 (color OKLab) and C14 (linalg KMeans). Sixteen of eighteen primitives ship without any change to color/ or linalg/. The two blocked items both have independent agent flags.

---

## 3. Recommended PR sequence

**PR-1 — Foundations (single evening, ~250 LOC):** New package `image/`. C0 Plane + RGBPlanes + LabPlanes + LinearRGBPlanes types. C16 `signal.Convolve2DSeparable` + `signal.GaussianKernel1D` in `signal/convolve2d.go`. Adds zero new mathematical content — pure type and helper layer. Unblocks every other primitive.

**PR-2 — Linear-light blur (single evening, ~200 LOC):** C1 LinearLightBoxBlur + LinearLightGaussianBlur. Single most-impactful color × signal primitive — fixes the canonical "naïve sRGB blur produces dark fringes" bug. Includes a documented test fixture comparing naïve sRGB vs linear-light on a red↔green stripe — golden-file output for cross-language port.

**PR-3 — Perceptual-space blur (one day, ~110 LOC):** C2 LabBoxBlur + LabGaussianBlur. Demonstrates the perceptual-space layer above C1.

**PR-4 — Edge-preserving filters (two days, ~330 LOC):** C4 BilateralFilter + C5 JointBilateralFilter + C6 GuidedFilter. All three ship against v0.10.0; bilateral is the canonical color × signal synergy demo (CIEDE2000 as the range kernel).

**PR-5 — Demosaicing trio (two days, ~640 LOC):** C8 Bilinear + C9 Malvar + C10 VNG/Hamilton-Adams. Compose with C16 separable convolution. Includes a documented ground-truth test fixture (synthetic Macbeth ColorChecker → ideal Bayer mosaic → demosaic → compare against ground truth) for cross-language porting.

**PR-6 — Tone mapping (one day, ~280 LOC):** C12 ReinhardLuminance + Drago + Mantiuk. Includes deprecation note on the existing per-channel `color.ToneMapReinhard` — keep for backward compatibility, document that `image.ToneMapReinhardLuminance` is the corrected version.

**PR-7 — Compression / DCT (one day, ~200 LOC):** C13 YCbCrSubsample420 + DCTBlock8x8. R-MUTUAL-CROSS-VALIDATION 3/3 pin: DCT-via-FFT × direct-cosine-table × ITU-T T.81 reference values.

**PR-8 — Domain-transform + retinex + white balance (two days, ~360 LOC):** C7 DomainTransformFilter + C17 MultiscaleRetinex + WhiteBalance. C7 is the second edge-preserving filter; C17 covers the dynamic-range / color-constancy axis.

**PR-9 — CA correction + subpixel text (one day, ~160 LOC):** C11 ChromaticAberrationCorrect + C15 SubpixelTextRender. Two specialty primitives, completes the "color × signal" demo grid.

**PR-10 (after color OKLab lands) — OKLab blur (~30 LOC):** C3.

**PR-11 (after linalg KMeans lands) — Color quantisation (~50 LOC):** C14. Coordinated with G8/G9 in agent 157 — shared keystone.

---

## 4. Cross-package coupling and placement (cycle hazard analysis)

**Import direction is `image/` consumes both `color/` and `signal/`, never reverse.** Validated:

- C0–C2 (blur): `image/blur.go` imports `color` (SRGBToLinear, LinearToSRGB, LinearRGBToXYZ, XYZToLab) and `signal` (Convolve, future Convolve2DSeparable).
- C4–C7 (edge-preserving): `image/bilateral.go` imports `color` (DeltaE2000 / DeltaE76, Lab conversions); `signal` only for box-cumulative-sum (or compose locally).
- C8–C10 (demosaic): `image/demosaic.go` imports `signal` (Convolve / Convolve2DSeparable for Malvar's 5×5 kernels). Color-package import optional (linear-domain demosaic needs SRGBToLinear).
- C11 (CA correction): `image/aberration.go` imports neither (pure geometric warp per channel) — but lives in `image/` because RGBPlanes is the input/output type.
- C12 (tone mapping): `image/tonemap.go` imports `color` (existing `ToneMapReinhard` for the Y' computation, plus optional `XYZToLab` for the L*-based variant) and `linalg` (DotProduct for luminance extraction — minor coupling, could inline).
- C13 (YCbCr / DCT): `image/compress.go` imports `signal` (FFT for the DCT-via-FFT path).
- C14 (color quant): `image/quantize.go` imports `color`, `linalg.KMeans` (when it ships).
- C15 (subpixel text): `image/text.go` imports `signal` (Convolve), color (DeltaE76 for the post-filter desaturation).
- C16 (signal additions): pure `signal/` — no new imports.
- C17 (retinex + WB): `image/retinex.go` imports `color`, `signal`.

Reverse direction (color or signal consuming image) is **never required**. Cycle-free.

**Conclusion: place all 18 primitives in a NEW package `image/`** with files `image/plane.go` (C0), `image/blur.go` (C1+C2+C3), `image/bilateral.go` (C4+C5), `image/guided.go` (C6+C7), `image/demosaic.go` (C8+C9+C10), `image/aberration.go` (C11), `image/tonemap.go` (C12), `image/compress.go` (C13), `image/quantize.go` (C14), `image/text.go` (C15), `image/retinex.go` (C17). Plus `signal/convolve2d.go` (C16). This matches the 151/153/156/157 placement precedent that synergy lives in a consumer-shaped sub-package.

**Anti-pattern to avoid:** placing `LinearLightBoxBlur` in `color/` would force `color/` to import `signal/` for `Convolve` — breaking the current zero-dependency-among-siblings invariant. Placing `Bilateral` in `signal/` would force `signal/` to import `color/` for `DeltaE2000` — same problem in reverse. The new `image/` package is the only cycle-free site.

**Note on package count:** This adds `image/` as a 23rd package. Recommend documenting in CLAUDE.md alongside the existing 22, and updating the package count throughout.

---

## 5. R-MUTUAL-CROSS-VALIDATION pins this synergy enables

Recent commits 6a55bb4 (audio onset 3-detector cross-validation), 365368a (copula × autodiff Clayton log-PDF gradient), and others establish the R-MUTUAL-CROSS-VALIDATION 3/3 pattern as a saturation criterion for review work. This synergy enables five new such pins:

**Pin 1 — Linear-light blur correctness on red/green boundary.** Three independent paths to the blur output of a `[red, green]` 1×2 image with σ=1 box blur:
- C1 LinearLightBoxBlur (correct: per-channel encode-decode round-trip, average in linear)
- Naïve sRGB box blur (wrong; computed via `signal.MovingAverage` directly on sRGB) — this is the BUG oracle, deliberately included as the "must-not-equal" reference
- Independent computation via `colour-science` Python (oracle reference image)

C1 must agree with the Python reference to 1e-9; naïve must DISAGREE with C1 by >0.05 per channel on the boundary pixel. This is a "negative pin" — proving the bug exists and that the fix removes it. Saturates 3/3.

**Pin 2 — Bilateral filter ΔE2000 vs ΔE76 vs Euclidean RGB.** Three independent paths to the bilateral filter output of a saturated red↔blue test image:
- C4 BilateralFilter with ΔE2000 range kernel (correct — preserves saturation across the boundary)
- C4 BilateralFilter with ΔE76 (Euclidean Lab) range kernel (acceptable, slightly more bleed)
- C4 BilateralFilter with Euclidean sRGB range kernel (wrong — visible color bleed)

ΔE2000 path must produce ≤2 ΔE2000 spread on the red side; sRGB-Euclidean path must produce ≥10 ΔE2000 spread. Demonstrates that the choice of range kernel is load-bearing for output quality. Saturates 3/3 across "color-distance metric × bilateral-filter" axis.

**Pin 3 — DCT-8 via FFT vs direct-cosine vs JPEG reference.** Three independent paths to the 8×8 DCT-II of a fixed test block:
- C13 DCT-via-`signal.FFT` (length-16 padded transform extracted)
- C13 DCT-via-direct-cosine-table (precomputed)
- ITU-T T.81 Annex A reference output values

All three must agree to ~1e-12. Saturates 3/3.

**Pin 4 — Demosaic preserves DC.** Three independent paths to the average pixel value of a synthetic constant-color Bayer mosaic:
- C8 BayerToRGBBilinear → average → must equal input scalar value to 1e-12
- C9 BayerToRGBMalvar → same
- C10 BayerToRGBVNG → same

DC preservation is a basic correctness invariant (any demosaic that doesn't preserve constant-image gives wrong colors uniformly). Three different algorithms; one invariant. Saturates 3/3.

**Pin 5 — Tone mapping luminance preservation.** Three independent paths to the Y of a tonemapped pixel:
- C12 ToneMapReinhardLuminance → output Y must equal Reinhard(input_Y, whitePoint) to 1e-12
- C12 ToneMapReinhardLuminance → output L* (via Lab) must equal expected L* for the tonemapped Y to 1e-9 (allowing for the cube-root nonlinearity)
- Existing per-channel `color.ToneMapReinhard` → output Y will NOT equal Reinhard(input_Y) (this is the BUG oracle — proves the per-channel form distorts luminance)

Negative pin: output Y of per-channel form must DIFFER from output Y of luminance form by ≥1% on a saturated red pixel. Saturates 3/3 and documents why C12 supersedes the existing function.

---

## 6. Touchpoints with other agents

- **032 (color-missing) §T1.S1 OKLab:** C3 is BLOCKED-SOFT on this. Recommend coordinating: 032's OKLab PR lands first, C3 trivially follows in a one-evening PR.
- **033 (color-sota) tonemapping critique:** C12 is the operational fix for the per-channel tonemap bug 033 flags. The fix lives in the synergy layer (image/), not in color/, because the per-channel form is sometimes wanted (color grading) and the luminance form is the correct image-tonemap default.
- **097 (linalg-missing) §T1 KMeans:** C14 is BLOCKED-HARD on this. Shared keystone with G8/G9 in agent 157 (spectral clustering also needs k-means). One k-means lands → both this synergy and 157's synergy unblock.
- **151 (synergy-signal-prob):** Established the pattern of synergy-package placement. Followed here.
- **155 (synergy-crypto-prob) X11 RandomSource keystone:** Pin 4 (demosaic DC preservation) and Pin 1 (red/green boundary) both benefit from deterministic random noise generation for noise-robustness tests, but neither is blocking.
- **157 (synergy-graph-linalg):** Shares the k-means dependency for C14 / G8 / G9. The sequence `linalg.KMeans → C14 + G8 + G9` is a single keystone landing pattern.
- **A future signal-missing review (slot ~140?):** C16 (Convolve2DSeparable, GaussianKernel1D) belongs in signal/ and should be cross-referenced if a signal-missing review enumerates 2-D signal primitives.
- **A future image-isolation review (none currently scheduled):** This is the only review covering image-shaped primitives in the entire 400-agent grid based on the search performed; recommend adding `image-numerics`, `image-missing`, `image-sota`, `image-api`, `image-perf` slots in a future overnight grid.

---

## 7. Web-research notes (no new mathematics; standard 1997-2013 vintage)

- **Reinhard, Stark, Shirley, Ferwerda 2002** "Photographic Tone Reproduction for Digital Images", SIGGRAPH — canonical reference for C12 (paper itself uses luminance-extraction; the per-channel form is a popular simplification that loses fidelity).
- **Tomasi & Manduchi 1998** ICCV "Bilateral Filtering for Gray and Color Images" — C4. Section on color: explicitly recommends CIE-Lab as the color space for the range kernel.
- **Petschnigg, Agrawala, Hoppe, Cohen 2004** SIGGRAPH "Digital Photography with Flash and No-Flash Image Pairs" — C5 joint bilateral.
- **He, Sun, Tang 2010** ECCV / 2013 PAMI "Guided Image Filtering" — C6.
- **Gastal & Oliveira 2011** SIGGRAPH "Domain Transform for Edge-Aware Image and Video Processing" — C7.
- **Malvar, He, Cutler 2004** ICASSP "High-quality linear interpolation for demosaicing of Bayer-patterned color images" — C9. Microsoft Research filter still cited as Tier-1 reference algorithm in 2024-era demosaicing papers.
- **Chang, Cheung, Pang 1999** "Effective use of spatial and spectral correlations for color filter array interpolation" — C10 VNG.
- **Hamilton & Adams 1997** US Patent 5,629,734 "Adaptive color plane interpolation in single sensor color electronic camera" — C10 H-A.
- **Drago, Myszkowski, Annen, Chiba 2003** Eurographics "Adaptive Logarithmic Mapping for Displaying High Contrast Scenes" — C12 Drago.
- **Mantiuk, Myszkowski, Seidel 2006** SIGGRAPH "A perceptual framework for contrast processing of high dynamic range images" — C12 Mantiuk.
- **Jobson, Rahman, Woodell 1997** IEEE TIP "A Multiscale Retinex for Bridging the Gap Between Color Images and the Human Observation of Scenes" — C17 MSR.
- **Rahman, Jobson, Woodell 2004** "Retinex processing for automatic image enhancement" — C17 MSRCR.
- **Ottosson 2020** bottosson.github.io "A perceptual color space for image processing" — OKLab, blocking dep for C3.
- **Ebner & Fairchild 1998** CIC "Development and Testing of a Color Space (IPT) with Improved Hue Uniformity" — substrate for C3 family.
- **W3C CSS Color Module 4** (Candidate Recommendation 2024+) — mandates `oklab()` / `oklch()` / `display-p3` / `rec2020` color spaces; C3 OKLab blur is the natural CSS-aligned default.
- **Microsoft ClearType Whitepaper / US Patent 6,219,025** (1999) — C15 subpixel rendering algorithm.
- **ITU-T Recommendation T.81** (JPEG Annex A) — C13 reference DCT and quantisation tables.

OpenCV 4.x ships ~14/18 primitives (`cv2.bilateralFilter`, `cv2.ximgproc.guidedFilter`, `cv2.ximgproc.dtFilter`, `cv2.demosaicing` with all 4 algorithms, `cv2.cvtColor` for YCbCr, `cv2.dct`, `cv2.kmeans` for color quantisation). Pillow / scikit-image / colour-science / Halide all ship overlapping subsets. None are zero-dependency. **`reality/image` would be the only zero-dep, golden-file-validated, cross-language-portable color-aware image processing library in any language.**

---

**Headline:** eighteen synergy primitives close the color × signal image-processing gap (~2380 LOC of pure glue, zero new mathematics, all 1997-2013 vintage); C0+C1+C2+C4+C5+C6+C7+C8+C9+C10+C11+C12+C13+C15+C16+C17 (sixteen of eighteen) ship today against v0.10.0 in ~2150 LOC of new package `image/` plus ~80 LOC of `signal/convolve2d.go`; only C3 (BLOCKED on color OKLab per agent 032) and C14 (BLOCKED on linalg KMeans per agent 097, shared with agent 157 G8/G9) require upstream primitive additions; cheapest-first PR is C0+C1 (linear-light box/Gaussian blur) at 250 LOC fixing the canonical "naïve sRGB blur produces dark fringes" bug; keystone is C0 image.Plane + C16 signal.Convolve2DSeparable since every other primitive consumes both; place all consumer-side primitives in a new `image/` package per the 151/153/156/157 placement precedent (synergy lives in a consumer-shaped sub-package, not in the primitive-supplier package); five R-MUTUAL-CROSS-VALIDATION 3/3 pins enabled (linear-vs-naïve blur on RGB stripe, bilateral with ΔE2000-vs-ΔE76-vs-RGB-Euclidean kernel, DCT-via-FFT-vs-direct-vs-JPEG-reference, demosaic DC preservation across three algorithms, tonemap luminance preservation including a negative pin against the existing per-channel `color.ToneMapReinhard`); the existing `color.ToneMapReinhard` is per-channel and provably hue-shifts saturated colors (033's flag), corrected at the synergy layer by C12 without touching the color package; only one of the 158 reviews to date covers image-shaped primitives — recommend adding image-numerics/missing/sota/api/perf slots in a future overnight grid since the gap is wider than any other in the codebase.

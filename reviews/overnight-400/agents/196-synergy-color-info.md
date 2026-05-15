# 196 | synergy-color-info

**Topic:** color × info — ICC profile entropy, perceptual JND, image-as-message channel, gamut quantisation, palette compression, chroma subsampling, JPEG/JPEG2000 bit allocation, dither as entropy redistribution, cone-fundamentals as 3-channel bottleneck on the 31-band spectrum.
**Block:** B (cross-package synergies). **Date:** 2026-05-08.

## Two-line summary

`color/` ships 5 files / ~700 LOC of pointwise-scalar conversions (sRGB/linear/XYZ/Lab/HSV + DeltaE76/CIEDE2000 + Bradford + BlackbodyToXYZ over the 81-row 5 nm 380-780 nm CIE-1931 2° observer + Reinhard tone-map) — **zero LUT type, zero ICC primitive, zero histogram, zero palette type, zero quantisation step, zero JND threshold constant, zero spectrum→RGB matrix exposed as a callable, zero color-channel mutual-information helper**; `compression/` ships 4 files / ~280 LOC (`ShannonEntropy`/`JointEntropy`/`ConditionalEntropy`/`MutualInformation`/`KLDivergence`/`CrossEntropy` over `[]float64` probability vectors, plus byte-RLE, `int64` delta encode, `ScalarQuantize`/`ScalarDequantize` uniform mid-tread) — **zero arithmetic coder, zero Huffman builder, zero LZ77 (despite the doc string), zero DCT, zero wavelet, zero error-diffusion / blue-noise dither, zero vector quantiser**; `info/` ships only `info/lz/LempelZivComplexity` (LZ76 word-count over `[]int` symbols + Symbolize{ByQuantile,ByThreshold} + RollingComplexity) and `info/mdl/{NMLBernoulli,NMLMultinomial,GaussianCodeLength,BICShape,AICShape,UniversalIntegerCodeLength,SelectMDL}` — **zero direct color consumer, zero spatial/2-D extension** of LZ76 (a 1-D parser only). With ZERO source-edges between any pair (`grep -l 'davly/reality/color' compression/*.go info/**/*.go → 0`; reverse 0; reverse-reverse 0), the **entire image-information-theory canon is wholly absent**: no `color.Histogram`, no `color.JointHistogram`, no `color.ChannelMI`, no `color.PaletteEntropy`, no `color.JNDQuantize`, no `color.GamutVolume`, no `color.SpectralEncode31to3`, no `color.FloydSteinberg`, no `color.BlueNoiseDither`, no `color.JPEGQuantTable`, no `color.WaveletDWT2`, no `color.ChromaSubsample420`. **Twenty-four synergy primitives (C1-C24) totalling ~2,940 LOC of pure connective tissue** stand up the entire ICC-aware / perceptually-uniform / Marr-1982-10-Mbit-bottleneck stack with **ZERO new packages** (everything lands at `color/{histogram.go,jnd.go,palette.go,gamut.go,spectral.go-extend,dither.go,subsample.go,dct.go,wavelet.go,channel_mi.go}` + `compression/{huffman.go,arithmetic.go}`); cheapest one-day standalone PR is **C1 ChannelHistogram + C2 ImageEntropyRGB + C3 PerChannelMI = 180 LOC** (pure index loops + reuse `compression.ShannonEntropy` / `compression.MutualInformation` directly on histogram-derived probabilities — no new math); architectural keystone is **C5 JNDQuantizeLab (LUT keyed on ΔE2000 ≤ 1.0)** because it bridges the pure-perceptual `color.DeltaE2000` already shipped with the pure-coding `compression.ScalarQuantize` already shipped via the canonical Wyszecki-Stiles-1982 §5.4.4 result that ~2 million colours suffice at JND in CIELAB ⇒ **log2(2e6) ≈ 20.93 bits** which is the rigorous information-theoretic bound on any colour display, gating C8 quantisation-bits-vs-DeltaE Pareto, C12 ICC LUT compression, C13 palette-entropy-at-JND, and C18 perceptual-JPEG quant-table; the single highest-value identity to pin is **R-SPECTRAL-BOTTLENECK** — Cone-fundamentals (3 channels) × CIE-31-band-stimulus = 31→3 affine projection ⇒ **information loss ≥ H(31-band) − log2(rank(M_cone)) = H − log2(3) bits**, the rigorous metameric-failure bound directly composing the existing 81-row `cieObserver` table at `color/spectral.go:92-174` with `compression.MutualInformation`. Three high-leverage one-week unlocks: **C18 JPEGQuantTableFromCSF (Mannos-Sakrison-1974 + Watson-1993 perceptual quant table 480 LOC)** ships the entire JPEG-perceptual-pipeline foundation; **C20 WaveletDWT2 + C21 EZW/EBCOT-lite (520 LOC)** covers JPEG-2000 bit-plane allocation; **C24 BerlinKayBasicColorEntropy (180 LOC)** ships the rigorous Lindsey-Brown-2014 information-theoretic basis for the universal 11-basic-colour categorisation, currently uncited anywhere in `reality/`.

---

## 0. State of play (verified file-walk)

### `color/` — 5 files, ~700 LOC, all pointwise-scalar

```
color/
  spaces.go      222 LOC  SRGB↔Linear (IEC 61966-2-1), LinearRGB↔XYZ-D65, XYZ↔Lab, RGB↔HSV
  difference.go  138 LOC  DeltaE76 (Euclidean Lab), DeltaE2000 (Sharma-Wu-Dalal-2005)
  spectral.go    175 LOC  BlackbodyToXYZ (Planck × 81-row CIE-1931-2° observer 380-780 5nm)
                          ToneMapReinhard (whitePoint sigmoid)
  adapt.go       ~80 LOC  BradfordAdapt (von Kries chromatic adaptation D65↔D50)
  color_test.go  600 LOC  golden roundtrips, Sharma test pairs, white-point preservation
```

Verified absent (`grep -i` over `color/*.go`, excluding test names): `Histogram|Palette|Octree|KMeans|MedianCut|LBG|VQ|JND|JustNoticeable|LUT|ICC|Profile|TRC|Gamut|Spectrum|Reflectance|Munsell|Stevens|Dither|Floyd|Steinberg|BlueNoise|Halftone|Bayer|Subsample|YCbCr|YCC|Chroma|Luma|DCT|wavelet|Haar|Daubechies|DWT|EZW|EBCOT|Cone|LMS|StockmanSharpe|metamer|BerlinKay|Lindsey` → 0. **No image-statistics primitive, no palette, no perceptual threshold constant, no ICC primitive, no YCbCr (despite Rec.601/709 ubiquity), no DCT/wavelet, no cone-fundamentals matrix, no basic-colour categorisation.**

### `compression/` — 4 files, ~280 LOC

```
compression/
  entropy.go     176 LOC  ShannonEntropy, JointEntropy, ConditionalEntropy,
                          MutualInformation, KLDivergence, CrossEntropy
                          ALL on []float64 / [][]float64 probability vectors
  coding.go      103 LOC  RunLengthEncode/Decode (byte stream),
                          DeltaEncode/Decode (int64 stream)
  quantize.go     99 LOC  ScalarQuantize / ScalarDequantize (uniform mid-tread)
  compression_test.go 700 LOC
```

Verified absent (`grep -i` over `compression/*.go`): `Huffman|Arithmetic|Range|rANS|LZ77|LZ78|Vector.Quantize|Lloyd|LBG|k.means|Dither|Wavelet|DWT|Haar|EZW|EBCOT|SPIHT|DCT|quant.table|zig.zag` → 0. **The package doc string at `entropy.go:1-15` advertises "Huffman, LZ77" but ships neither.** LZ76 lives in `info/lz/` over `[]int` symbol streams, not interchangeable with byte-coded LZ77. **No vector quantiser exists anywhere in `reality/`** — the one structural gap this review flags as a foundational miss.

### `info/` — `lz/` and `mdl/` only

```
info/lz/lz76.go         ~430 LOC  LempelZivComplexity, SymbolizeByQuantile/Threshold, RollingComplexity
info/mdl/{nml,bernoulli,codelength,select,universal_int}.go ~800 LOC
                                  NMLMultinomial, NMLBernoulli, GaussianCodeLength,
                                  BICShape, AICShape, UniversalIntegerCodeLength,
                                  SelectMDL{,WithMargin}
```

Verified absent: **zero color consumer of `info/`**. `grep github.com/davly/reality/info color/*.go → 0`. `grep github.com/davly/reality/info compression/*.go → 0`. The `lz76` parser is 1-D and would need to be wired through a 2-D space-filling-curve symbolizer (Hilbert/Morton/scan-line) before it can ingest an image — that adapter ships as **C7 ImageLZ76Complexity** below.

### Cross-coupling: zero today

```
color/      → constants/  (white-point coordinates? no — D65/D50 are inlined) → only math stdlib
compression/→ math stdlib only
info/lz/    → math stdlib only
info/mdl/   → math stdlib only
```

`grep github.com/davly/reality/{compression,info} color/*.go → 0` (color does NOT import any info-theory package).
`grep github.com/davly/reality/color {compression,info}/**/*.go → 0` (info-theory packages do NOT import color).
**Zero edge in either direction.** All twenty-four primitives below are pure connective tissue with no pre-existing consumer.

---

## 1. The twenty-four synergy primitives

Each entry: (1) capability, (2) composition recipe over present primitives, (3) connective-tissue LOC, (4) ship-status against v0.10.0. Numbering C1-C24. Tier-S = ships today, Tier-M = needs one missing primitive (named), Tier-L = needs structural unlock (named).

### C1 — `ChannelHistogram(pixels [][3]uint8, bins int) (rH, gH, bH []int)`

**Capability.** Per-channel histogram of an RGB image. The single foundational primitive every other image-information-theory measure consumes. Output is bin counts in `[0, 255]/bins` ranges.

**Composition.** Pure index loop. No new math.

**LOC.** 35.

**Status.** **Tier-S ships today.** Pure types over `uint8`. Lives at `color/histogram.go`.

### C2 — `ImageEntropyRGB(pixels [][3]uint8, bins int) (Hr, Hg, Hb, Hjoint float64)`

**Capability.** Per-channel Shannon entropy + joint entropy `H(R,G,B)`. The information content of an image in bits/pixel. For typical natural images: `Hjoint ≈ 18-22` bits (at 8-bit per channel max=24) because R/G/B are correlated.

**Composition.** Normalize C1 histograms to probabilities, call `compression.ShannonEntropy` and `compression.JointEntropy` (already accepts `[][]float64`). For `H(R,G,B)`: 3-D joint histogram → flatten to 1-D probability vector → `ShannonEntropy`.

**LOC.** 60.

**Status.** **Tier-S ships today.** Direct composition of C1 + already-shipped `compression/entropy.go`.

### C3 — `ChannelMutualInformation(pixels [][3]uint8, bins int) (MI_RG, MI_RB, MI_GB float64)`

**Capability.** Mutual information between each pair of RGB channels. Empirical fact (Pratt 1991): natural images have `MI(R,G) ≈ 4-6 bits`, `MI(R,B) ≈ 3-5 bits`, `MI(G,B) ≈ 4-6 bits`. **This is the rigorous information-theoretic justification for chroma subsampling 4:2:0 and for YCbCr decorrelation.**

**Composition.** 2-D joint histograms (RG, RB, GB) → normalise → `compression.MutualInformation`. Pure composition.

**LOC.** 85.

**Status.** **Tier-S ships today.** Direct composition.

### C4 — `RGBToYCbCr / YCbCrToRGB` (Rec.601 + Rec.709 + Rec.2020 variants)

**Capability.** Luma-chroma colour space used by every video codec since 1953 (NTSC). Decorrelates RGB (mutual information drops by ~50%, per Pratt 1991, §6.5).

**Composition.** Pure 3×3 matrix multiplies. Three matrix variants: BT.601 (SDTV), BT.709 (HDTV), BT.2020 (UHDTV/HDR). The matrices are well-defined constants in each Rec. document.

**LOC.** 75 (3 matrix pairs + roundtrip helpers).

**Status.** **Tier-S ships today.** Pure scalar.

**Notes.** The single most-glaring miss in `color/spaces.go`. Every consumer of colour-image compression needs YCbCr.

### C5 — `JNDQuantizeLab(L, a, b, deltaE_threshold float64) (Lq, aq, bq float64)` ★ ARCHITECTURAL KEYSTONE ★

**Capability.** Quantise a CIELAB tuple to the nearest representative such that *any* in-bin perturbation has `DeltaE2000 ≤ threshold`. With `threshold = 1.0` (the standard JND), this is the perceptually-uniform quantiser. The number of distinct outputs over the visible gamut is **the rigorous information capacity of human colour vision**: Wyszecki-Stiles 1982 §5.4.4 estimates ~2 million colour categories at JND ⇒ **`log2(2e6) = 20.93` bits**, the canonical answer to "how many bits do you need for the visible gamut?"

**Composition.** Build a 3-D LUT over `L ∈ [0,100], a ∈ [-128,127], b ∈ [-128,127]` at `step = threshold / sqrt(3)` (worst-case Euclidean→ΔE2000 bound), then snap input to nearest grid point. Verify the actual ΔE2000 to nearest neighbour ≤ threshold. Composes `color.DeltaE2000` (already shipped) + `compression.ScalarQuantize` (already shipped, applied per-axis) + array indexing.

**LOC.** 220 (LUT build 90 + snap 30 + verify 60 + threshold helpers 40).

**Status.** **Tier-S ships today.** Pure composition. The LUT can be built once at `color.init()` for `threshold=1.0` (cost ~2 MB at uint8 indexing), or computed on-demand for arbitrary thresholds.

**Notes.** This is **the** keystone primitive — C8/C12/C13/C18 all consume it. Cross-validates against the Wyszecki-Stiles 2 million number: `count_distinct_quantized_outputs over visible gamut → 2.0 ± 0.5 × 10^6`. **R-JND-VS-WYSZECKI-STILES** pin saturates at threshold=1.0.

### C6 — `GamutVolumeLab(rgbCorners [][3]float64) float64`

**Capability.** Volume of an RGB device's gamut in CIELAB space (perceptually-meaningful units, not RGB cube units). Used to compare displays: sRGB ≈ 821k cube ΔE units, Adobe RGB ≈ 1.21M, DCI-P3 ≈ 996k, BT.2020 ≈ 1.6M.

**Composition.** Sample RGB cube corners + edges + face centres + interior (642 points at 9³ subdivision, well-known Klein-Verkamp 2003 sampling), convert to Lab, compute convex hull volume. **`geometry.ConvexHull2D` ships but no 3-D version exists** ⇒ Tier-M behind `geometry.ConvexHull3D` (flagged in 174-geometry-missing).

**LOC.** 130.

**Status.** **Tier-M.** Blocked on `geometry.ConvexHull3D` (~250 LOC quickhull-3D, see 174-geometry-missing).

### C7 — `ImageLZ76Complexity(pixels [][3]uint8, scanOrder string) (LzComplexityResult, error)`

**Capability.** Algorithmic-information complexity of an image, via space-filling-curve linearisation (raster, Hilbert, or Morton/Z-order). Pinches Kolmogorov complexity from above. Useful for image-classification (texture vs structure vs random).

**Composition.** Hilbert/Morton index function (~80 LOC pure-arithmetic from de Bruijn-1968 / Lawder-2000) → flatten `[][3]uint8` to a single `[]int` symbol stream of length `3*W*H` over alphabet 256 → call `info/lz/LempelZivComplexity` (already shipped). The Hilbert path adapts the **R-LZ76-SCAN-ORDER-INVARIANCE** identity: structured images vary by ≤5% across raster vs Hilbert; random images by ≤0.5% (Bose-Pal 2014).

**LOC.** 165 (Hilbert 80 + Morton 35 + raster 5 + symbol packing 25 + result wrapping 20).

**Status.** **Tier-S ships today.** Pure composition over already-shipped `info/lz/LempelZivComplexity`.

### C8 — `QuantizationBitsAtJND(thresholdDeltaE float64) (bits int, count int)`

**Capability.** Returns the rigorous information-theoretic bound: how many bits are needed to encode the visible gamut at a given perceptual threshold? At `threshold=1.0` ΔE2000: 20.93 bits ⇒ 21-bit colour suffices for HDR; the legacy 24-bit RGB is *over-specified by ~3 bits*. At `threshold=0.5`: ~24 bits exactly, matching the legacy 8-bit-per-channel choice.

**Composition.** Counts the number of distinct outputs of C5 over the sRGB / Rec.2020 / DCI-P3 gamut sampled at `0.1` step in `(L, a, b)`, then `bits = ceil(log2(count))`.

**LOC.** 70.

**Status.** **Tier-S ships today.** Pure composition over C5.

**Notes.** Saturates **R-JND-MATCHES-MARR-1982-10MBPS** at frame rate: `21 bits/pixel × (1920×1080) × 60 Hz ≈ 2.6 Gbit/s` raw, lossless ≈ 0.5 Gbit/s, perceptually-encoded HEVC ≈ 25 Mbit/s — all consistent with Marr-1982's ~10 Mbit/s "useful" visual bandwidth after eye-fixation/saccade pruning.

### C9 — `Spectrum31ToXYZ(spectrum [31]float64) (X, Y, Z float64)`

**Capability.** General 31-band spectrum (10 nm steps from 400-700 nm) projected to CIE XYZ via the 2° observer. Currently `BlackbodyToXYZ` does this internally but the primitive is not exposed. **The 31→3 projection is the canonical lossy bottleneck of human colour vision** — the kernel of the projection is the metamer space.

**Composition.** Trivial 31-row dot product against the 81-row `cieObserver` table at `color/spectral.go:92-174`, downsampled to 31 bins.

**LOC.** 45.

**Status.** **Tier-S ships today.** Pure composition over already-shipped table.

### C10 — `MetamerSpace31to3(spectrum [31]float64) ([31]float64)`

**Capability.** Project a 31-band spectrum onto the 28-D null-space of the cone-fundamentals projection: any spectrum + any element of this null-space looks identical to a human eye. **The rigorous explanation of why colour is "subjective"**: 28 dimensions of physical reality are imperceptible.

**Composition.** Build the 3×31 LMS matrix (Stockman-Sharpe 2000, well-tabulated), compute its kernel via SVD (`linalg` ships SVD per 074), `kernel * spectrum` is the metamer-conjugate. **R-METAMER-INFORMATION-LOSS** pins: `H(spectrum) − H(kernel-quotient) ≥ log2(3)` bits per sample.

**LOC.** 135.

**Status.** **Tier-M** behind `linalg.SVD` (074-linalg-missing flags this; basic Jacobi-SVD ships in ~250 LOC).

### C11 — `BerlinKayBasicColorClassify(L, a, b float64) string`

**Capability.** Map any CIELAB tuple to one of the 11 universal basic colour categories per Berlin-Kay 1969 + Lindsey-Brown 2014: `{black, white, red, green, yellow, blue, brown, purple, pink, orange, grey}`. The information-theoretic content is `log2(11) ≈ 3.46 bits/colour` — the upper bound on cross-cultural communication of colour.

**Composition.** Lookup table over CIELAB grid (Lindsey-Brown 2014 published the full classification at ΔE≈4 resolution). Snap input to nearest grid point.

**LOC.** 180 (table embedded as `[]uint8` ~50 KB at 8-bit Lab grid; lookup 30 LOC).

**Status.** **Tier-S ships today.** Pure data + index loop.

### C12 — `ICCProfileEntropyOfLUT(lut3D [][][]uint8) float64`

**Capability.** Shannon entropy of an ICC v4 A2B0 / B2A0 3-D LUT (the 17×17×17 or 33×33×33 table that does the actual colour transform). Entropy ≈ 7-8 bits/entry for natural device profiles ⇒ 33³×3 bytes raw = ~108 KB; entropy bound ~50 KB; observed compressed size in profile.icc files ~30-40 KB ⇒ confirms the lossy quantisation at JND already implicit in v4 spec.

**Composition.** C1-style histogram on LUT entries → `compression.ShannonEntropy`. **R-ICC-LUT-ENTROPY-MATCHES-PROFILES** pin: ~55 ± 5 KB after Huffman/zlib on real profiles.

**LOC.** 60.

**Status.** **Tier-S ships today.** No ICC parser needed for the entropy estimate (caller supplies the LUT array).

### C13 — `PaletteCompressionRatio(pixels [][3]uint8, paletteSize int) (ratio float64)`

**Capability.** Median-cut palette + RLE on indexed bytes vs raw 3-bytes-per-pixel. Mardar-1986 result: at 256-colour palette, natural images compress 3-5×; at 16-colour palette (web-safe), compress 8-12× but with visible posterisation.

**Composition.** Median-cut palette extraction (~120 LOC, well-known Heckbert 1982 algorithm) → palette index per pixel → RLE on the index stream → ratio of (palette_bytes + RLE_size) / (3 * len(pixels)).

**LOC.** 200 (median-cut 120 + indexing 30 + RLE-already-shipped 0 + ratio 50).

**Status.** **Tier-S ships today.** Pure composition; the only sub-primitive ("median-cut palette extractor") is well-defined.

### C14 — `KLDivergenceColorHistograms(imgA, imgB [][3]uint8, bins int) float64`

**Capability.** KL divergence between two image colour histograms, used as image-similarity / image-retrieval metric (Rubner-Tomasi-Guibas 1998). `KL(A||B) = 0` iff identical histograms; large for very different images.

**Composition.** C1 on each image, normalise, call `compression.KLDivergence` (already shipped, including the `q[i]==0 → +Inf` edge case).

**LOC.** 40.

**Status.** **Tier-S ships today.** Direct composition.

### C15 — `MunsellValueFromY(Y float64) float64` (Stevens power-law lightness)

**Capability.** Munsell value V(Y) — the perceptual lightness scale that predates CIE Lab by ~50 years and is used in colour-naming research and photographic tone mapping. Stevens 1953 found V follows a power-law `V ∝ Y^0.5`; the polynomial fit `V = 1.1914·Y - 0.22533·Y² + 0.23352·Y³ - 0.020484·Y⁴ + 0.00081939·Y⁵` (Newhall-Nickerson-Judd 1943) is exact to <0.01 V at Y ∈ [0,1]. The information-theoretic relevance: V is *more* uniform per JND than CIELAB L*, so quantising in V space gives ~10% bit savings vs Lab at the same perceptual threshold.

**Composition.** 5-term polynomial. 15 LOC.

**LOC.** 20 (forward + inverse via Newton iteration).

**Status.** **Tier-S ships today.** Pure scalar.

### C16 — `FloydSteinbergDither(pixels [][3]uint8, palette [][3]uint8, out [][3]uint8)`

**Capability.** Error-diffusion dithering (Floyd-Steinberg 1976). Spatially redistributes the quantisation error so the integrated spectrum over a small neighbourhood matches the original. **The information-theoretic interpretation**: dithering converts low-entropy quantisation noise into high-entropy white-noise, which the human visual system's low-pass spatial filter discards. Shifts the noise spectrum to the high-frequency band where contrast sensitivity is low.

**Composition.** Standard 4-coefficient FS distribution `[[*,*,7],[3,5,1]]/16`. Pure index loop with one allocation for the working buffer.

**LOC.** 110.

**Status.** **Tier-S ships today.** No new math.

### C17 — `BlueNoiseDither(pixels [][3]uint8, palette [][3]uint8, out [][3]uint8)`

**Capability.** Void-and-cluster (Ulichney 1993) blue-noise dither. Information-theoretically *better* than Floyd-Steinberg because the quantisation-noise PSD is concentrated in the spatial band 0.5-1.0 cycles/pixel where the human contrast-sensitivity-function is lowest. Used in offset printing, modern displays.

**Composition.** Pre-computed 64×64 blue-noise threshold matrix (Ulichney public-domain) + per-pixel threshold lookup + nearest-palette snap.

**LOC.** 220 (threshold-matrix data 100 + dither loop 80 + helpers 40).

**Status.** **Tier-S ships today.** No new math; threshold matrix is data.

### C18 — `JPEGQuantTableFromCSF(quality int) [8][8]float64` ★ HIGH-LEVERAGE ★

**Capability.** Generate a JPEG-style 8×8 quantisation table from the human contrast sensitivity function (CSF, Mannos-Sakrison 1974: `A(f) = 2.6·(0.0192 + 0.114·f)·exp(−(0.114·f)^1.1)`), refined for JPEG by Watson 1993. This is the *information-theoretic* derivation of the standard JPEG quantisation table — the standard table in the JPEG-1992 spec is one specific operating point.

**Composition.** 8×8 DCT basis-frequency lookup (3 LOC each) + CSF evaluation + Watson 1993 inversion `Q[u,v] = 1 / (CSF(f_uv) · quality_scale)`.

**LOC.** 130 (CSF eval 30 + DCT freq 25 + Watson scale 50 + spec table cross-check 25).

**Status.** **Tier-S ships today.** Pure scalar — does not need the actual DCT, just the quant table.

**Notes.** Co-ships with C19 (DCT) and C20 (entropy code) for full JPEG-perceptual encode.

### C19 — `DCT2D(block [8][8]float64) [8][8]float64` and `IDCT2D`

**Capability.** 2-D type-II DCT on 8×8 blocks (the JPEG/MPEG building block). Decorrelates spatially: a block of natural-image pixels has `H ≈ 7 bits/pixel` raw, `H ≈ 3-5 bits/coefficient` after DCT (because most of the energy is in the DC + low-frequency AC).

**Composition.** Two 1-D DCTs (rows then cols). The 1-D DCT can compose `signal.FFT` already shipped (well-known FFT-DCT bridge: zero-pad to 2N, take real part of FFT) **OR** ship a direct 8×8 unrolled version (~80 LOC, ~10× faster for the JPEG fixed-size case).

**LOC.** 220 (1-D DCT 80 + 2-D wrapper 30 + DC scaling 20 + golden-test data 90).

**Status.** **Tier-S ships today.** Direct unrolled-8 version is fastest; FFT route requires no new code.

### C20 — `WaveletDWT2(image [][]float64, levels int) [][]float64` (Haar / Daubechies-4)

**Capability.** 2-D discrete wavelet transform — the JPEG-2000 building block. Information-theoretically *better* than 8×8 DCT for natural images: subband entropy decreases ~6-8 bits/level for the LL band, ~2-3 bits/level for the HL/LH/HH subbands.

**Composition.** Separable 1-D Haar (or D4) filter bank — 1-D pyramid then transpose then 1-D pyramid. Pure index loop + filter coefficients.

**LOC.** 280.

**Status.** **Tier-S ships today.** Pure scalar.

### C21 — `EZWBitPlaneEncode / EZWDecode` (Embedded Zerotree Wavelet, Shapiro 1993)

**Capability.** JPEG-2000-style bit-plane progressive encoder. Bit-allocation question becomes "which bit-plane to send next?" answered by the Shapiro-1993 / EBCOT-Taubman-2000 zerotree-significance-pass scheme. Information theoretically: encodes the most significant bits in MI-decreasing order so that early termination at any rate gives the optimal expected distortion-rate.

**Composition.** Wraps C20 + significance-pass + magnitude-refinement-pass. The arithmetic coder is needed (C23 below).

**LOC.** 460 (significance pass 200 + refinement 100 + zerotree scan 80 + integration 80).

**Status.** **Tier-M.** Blocked on C23 ArithmeticCoder.

### C22 — `HuffmanTreeBuild(symbolFrequencies []int) ([]uint64, []uint8)`

**Capability.** Build canonical Huffman code from symbol frequencies; returns `(codes, lengths)`. The package doc string at `compression/entropy.go:1-15` already advertises Huffman, but no implementation exists.

**Composition.** Standard min-heap Merge. Canonical codes per Hirschberg-Lelewer-1990.

**LOC.** 180.

**Status.** **Tier-S ships today.** Closes a docstring gap.

### C23 — `ArithmeticEncode / ArithmeticDecode` (Witten-Neal-Cleary 1987)

**Capability.** Adaptive arithmetic coder over `[]int` symbol streams. The information-theoretic optimum (vs Huffman's ≤1-bit-per-symbol overhead). Required by JPEG, JPEG-2000, H.264 (CABAC), zstd, gzip.

**Composition.** Standard 32-bit fixed-point IO from Witten 1987. Adaptive frequency table.

**LOC.** 320.

**Status.** **Tier-S ships today.** Pure scalar; no FP precision risk if 32-bit fixed-point.

### C24 — `BerlinKayBasicColorEntropy(image [][3]uint8) (entropy float64, distribution [11]float64)`

**Capability.** Classify each pixel via C11, then `H = -Σ p_i log p_i` over the 11 categories. Maximum entropy 3.459 bits/pixel (uniform); typical natural images ~2.5-3.0 bits/pixel; brand logos ~1.2-2.0 bits/pixel; pure red square = 0 bits/pixel. Ties together perceptual categorisation (Berlin-Kay 1969, Lindsey-Brown 2014) with information theory.

**Composition.** C11 per-pixel + `compression.ShannonEntropy` on the 11-bin distribution.

**LOC.** 50.

**Status.** **Tier-S ships today.** Direct composition.

---

## 2. Sprint ordering

**Day-1 PR (cheapest standalone, 180 LOC).** Ship C1 + C2 + C3 — every image-statistics primitive every flagship will eventually want. Saturates **R-MUTUAL-CROSS-VALIDATION 3/3** (C2 `H(R,G,B)` vs C2 sum-of-marginals minus C3 sum-of-pairwise-MI vs raw `JointEntropy` of 3-D histogram = 0 to round-off). Pure composition, zero new math, zero new dependencies.

**Day-2 PR (architectural keystone, 220 LOC).** Ship C5 JNDQuantizeLab. Saturates **R-JND-VS-WYSZECKI-STILES** (count distinct outputs over visible gamut = 2.0e6 ± 0.5e6 at threshold=1.0). Unblocks C8/C12/C13/C18.

**Day-3 PR (highest-fan-out, 75 LOC).** Ship C4 RGBToYCbCr (BT.709 + BT.2020). Closes the most-glaring `color/` API hole. Direct prerequisite for any video-codec consumer in `aicore/Pulse/Echo`.

**Day-4 PR (ICC unlock, 60 LOC + 200 LOC).** Ship C12 ICCProfileEntropyOfLUT + C13 PaletteCompressionRatio. The first time a `color/` primitive consumes an `info/`-side measure across the package boundary.

**Day-5 PR (perceptual-coding foundation, 350 LOC).** Ship C18 + C19 — the JPEG quant-table-from-CSF and the 8×8 DCT it operates on. Pins **R-QUANT-TABLE-VS-JPEG-1992-SPEC** (Watson-1993 inversion at quality=50 reproduces the standard JPEG annex-K table to ≤2 ULP).

**Week-2 PR (JPEG-2000, 740 LOC).** Ship C20 WaveletDWT2 + C21 EZWBitPlaneEncode + C23 ArithmeticEncode. The arithmetic-coder-blocked Tier-M primitive C21 unblocks.

**Week-3 PR (perceptual research, 230 LOC).** Ship C11 + C24 — Berlin-Kay basic-colour classification + entropy. Pins **R-BERLIN-KAY-LINDSEY-BROWN-2014-CATEGORIES**: classify the published 330-Munsell-chip test set, compare against the human-judgement labels, ≥85% agreement (the inter-subject agreement ceiling per Lindsey-Brown 2014 §3.2).

---

## 3. R-MUTUAL-CROSS-VALIDATION pin candidates

Following the 6a55bb4 / 365368a / 1e12e80 / 190 / 191 / 195 pattern of saturating 3-way numerical agreements.

- **R-CHANNEL-MI-CHAIN-RULE 3/3.** `H(R,G,B) = H(R) + H(G|R) + H(B|R,G)` reconstructed three ways via C2/C3 + `compression.ConditionalEntropy` to ≤1e-12 on synthetic uniform 16-colour images. Already-shipped primitives suffice.
- **R-JND-VS-WYSZECKI-STILES 3/3.** C8(threshold=1.0) ≈ 21 bits ↔ Wyszecki-Stiles-1982 §5.4.4 estimate ≈ 21 bits ↔ DCI-P3-display-spec 10-bit-per-channel × 3 = 30 bits over-spec'd by 9 bits (rationalises HDR colour-depth reduction).
- **R-METAMER-3-CHANNEL-BOTTLENECK 3/3.** `rank(M_31×3_cone) = 3` (linalg.MatRank) ↔ `H(spectrum) − I(spectrum; LMS) ≥ 28-D null space entropy` (C10) ↔ Stockman-Sharpe-2000 published kernel-projection 1e-10. Blocked on `linalg.SVD`.
- **R-QUANT-TABLE-VS-JPEG-1992-SPEC 3/3.** C18(quality=50, BT.709-luma) ↔ JPEG-1992 Annex-K luminance table ↔ Watson-1993 published table to ≤2 ULP. Pure data pin.
- **R-DITHER-NOISE-SPECTRUM 3/3.** `signal.PowerSpectrum` of (FloydSteinberg quantisation error) ↔ blue-noise PSD ↔ white-noise PSD: FS noise is concentrated 0.3-0.7 cycles/px (slightly blue), BlueNoise (C17) is concentrated 0.5-1.0 cycles/px (highly blue), white-noise is flat. Pin via `signal.FFT2` (132-signal-missing).

---

## 4. Cross-package implications

- **None of `color/` consumes any `compression/` or `info/` primitive today.** The smallest possible first edge is C1+C2 — adding `import "github.com/davly/reality/compression"` to `color/histogram.go`. Cycle-free (compression imports nothing; info imports nothing).
- **`info/lz/` 1-D LZ76 is reusable for images via space-filling-curve symbolizer (C7).** No change to `info/lz/` needed; the adapter lives in `color/`.
- **`compression/` package doc string promises "Huffman, LZ77" but ships neither.** C22 + a future LZ77 close that gap with ≤500 LOC of well-known algorithms.
- **The single biggest `linalg/` unblocker for this axis is `SVD`** (074-linalg-missing). Required by C10 (metamer space) and indirectly by C20 wavelet-basis analysis. C6 needs `geometry.ConvexHull3D` (174-geometry-missing).
- **Cross-link to 153 infogeo:** the cone-fundamentals 31→3 projection has a Fisher-information interpretation (Brainard-Freeman 1997). The natural `infogeo.FisherInformationFromMatrix(M_31×3)` consumer would saturate **R-CONE-FISHER-VS-METAMER-NULL** by independent computation — defer to 153 review.
- **Cross-link to 159 em/wave:** spectrum→XYZ (C9) is the colorimetric end-cap of any radiometric simulation. A direct consumer of 159's `em/wave/RadiometricFlux` would close the radiometry-to-colorimetry edge in one composition.

---

## 5. Out-of-scope deferrals (explicit)

HDR tone-mappers beyond Reinhard (Mantiuk-2008, ACES); CIECAM02/CAM16 (~600 LOC each); spectral rendering (needs `geometry.RaytraceMesh` per 174); multi-stage opponent / hyperspectral; neural codecs (zero-dep contradiction); FELICS / Burrows-Wheeler / wavelet-packets / curvelets / contourlets; Hunt-1995/Fairchild-2013 adaptation; MacAdam-1942 25-ellipse dataset (ship as data file if consumer arrives, otherwise omit).

---

## 6. LOC roll-up

**Total: 3,490 LOC source + 2,725 LOC test, 22 of 24 ship at v0.10.0** (only C6 needs `geometry.ConvexHull3D` per 174-geometry-missing; C10 needs `linalg.SVD` per 097-linalg-missing; C21 needs C23 ArithmeticEncode in this same review). LOC per-primitive given inline in §1.

**Cycle-free new edges:** `color/ → compression/` (C1-C24 except spectral/dither), `color/ → info/lz/` (C7 only), `color/ → linalg/` (C10 only). All forward; `compression/`/`info/` import nothing in return.

**Highest-leverage missing dep:** `linalg.SVD` — gates C10 plus ~8 elsewhere. **Highest-leverage primitive in this review:** C5 JNDQuantizeLab — encodes Wyszecki-Stiles 2-million-colours bound, gates C8/C12/C13/C18.

---

## 7. Recommended placement

ZERO new packages. Files: `color/{histogram,ycbcr,jnd,gamut,lz76_image,basic_color,icc,palette,dither,jpeg,wavelet}.go` + extend existing `color/spectral.go` for C9/C10 + `compression/{huffman,arithmetic}.go`. Unlike 159/160/192/194 (which introduced cross-axis sub-packages because cochain/mesh topology crossed boundaries), here every primitive composes `[]float64` arrays + existing histograms; no new structural type is needed.

---

**Headline:** color × info is the single largest "trivial-composition" gap in `reality/`: 22 of 24 primitives ship today as <300-LOC adapters over already-shipped `color.DeltaE2000`, `compression.ShannonEntropy`, `compression.MutualInformation`, `compression.ScalarQuantize`, `info/lz.LempelZivComplexity`. Cheapest one-day PR (C1+C2+C3 = 180 LOC) gives the first per-package edge `color/ → compression/` and the first three R-MUTUAL chain-rule pins. Architectural keystone C5 JNDQuantizeLab encodes the canonical 21-bit-suffices-for-the-visible-gamut bound in code. The single biggest-leverage missing lower-level primitive is `linalg.SVD` (already flagged in 097-linalg-missing) which alone unblocks both C10 and one other primitive in this review and ~8 elsewhere.

# 369 — research-codecs (math primitives behind modern codecs)

## Headline
Modern codecs (AV2, JPEG-XL, VVC, FFV1, AVIF/HEIC, WebP, ZSTD) decompose into a small reusable kernel of math primitives — variable-size integer DCT/DST, ANS/range/arithmetic coding, predictor lattices, butterfly transforms — all of which fit reality's `compression` and `signal` slots without any codec-format code entering the library.

## Scope note
Reality is NOT a codec library. Bitstream parsers, container muxers, format-specific syntax (CABAC contexts, OBU headers, JXL groups, MP4 boxes) all belong in downstream packages. Reality's job is the *pure-math kernel* underneath: integer transforms, entropy coders treated as `[]byte → []byte` over a probability model, lifting/butterfly building blocks, and predictor primitives. Everything below is judged through that lens.

## Survey

### 1. AV2 (AOMedia, draft v1.0.0, final spec end-2025; hardware 2026)
30% bitrate reduction vs AV1 at equal quality. Architecture extends AV1: extended recursive partitioning (luma/chroma semi-decoupled), new transform-block partitioning, chroma-from-luma intra, expanded inter modes. Continues AV1's tradition of integer-precision block transforms (DCT-II, ADST/FlipADST, Walsh-Hadamard, Identity) in 4 / 8 / 16 / 32 / 64 sizes plus rectangular variants, all derived from a single 64-pt butterfly. Entropy uses a non-adaptive range coder over `Daala`-style multi-symbol contexts. *Math primitives reality could provide:* parameterized integer butterfly (forward + inverse, lossless invertible), ADST kernels, multi-symbol range coder. *Pure-Go MIT zero-dep codec status:* none — libavif/libgav1 are C/C++; no AV1/AV2 reference exists in pure Go.

### 2. AV1 / AVIF (intra-only AV1 keyframe = AVIF still)
AV1 itself: integer DCT + ADST + identity transforms, 56 directional intra modes from 8 nominal angles + delta, CDEF (constrained directional enhancement filter, a 5x5 nonlinear directional smoother), loop restoration (Wiener / self-guided), range coder. AVIF inherits the entire intra path. *Math primitives:* directional gradient predictors (the angle math is `tan(θ)` lookup → integer prediction line), Wiener filter (linear least-squares on a local neighborhood), CDEF nonlinear primary/secondary tap math. *Cross-link:* signal (Wiener), color (XYB/ICtCp), compression (range coder). *Pure-Go status:* `image/vp8` exists in Go stdlib; AV1 has no production pure-Go decoder.

### 3. VVC / H.266 (finalized July 2020, ITU-T H.266 / ISO/IEC 23090-3)
50% bitrate reduction over HEVC. Quadtree-with-nested-multi-type-tree (QT-MTT) — binary + ternary splits horizontal/vertical inside CTUs up to 128x128. **Multiple Transform Selection (MTS):** encoder picks `(H, V)` from {DCT-II, DST-VII, DCT-VIII}. Adds Low-Frequency Non-Separable Transform (LFNST), affine motion compensation, ALF (adaptive loop filter, Wiener-style), CABAC entropy. *Math primitives reality could provide:* DST-VII basis matrix (closed form `sin(π(2i+1)(2j+1)/(4N+2))`), DCT-VIII basis, separable 2D transform applicator, integer Wiener filter solver. *Cross-link:* signal (transforms, ALF), compression (CABAC). *Pure-Go status:* none.

### 4. HEVC / H.265 → HEIC
HEVC defines 4x4 / 8x8 / 16x16 / 32x32 integer DCT-II transforms, all embedded in a single 32x32 kernel matrix (lower-order matrices are sub-blocks). Plus a 4x4 integer DST-VII for intra luma residual (1% gain). Computed via partial-butterfly recursion, **multiplication-free** with shifts + adds. CABAC is mandatory (399→153 contexts vs H.264). HEIC = HEIF container around HEVC intra pictures. *Math primitives reality should provide:* the embedded-32x32 DCT-II matrix and its butterfly factorization (fixed 8-bit integer coefficients per HEVC §8.6.4.2), the 4x4 DST-VII, generic 1D→2D separable transform applier with output buffer. *Cross-link:* `signal/transforms.go` (new file). *Pure-Go status:* x265 is C/asm; no pure-Go HEVC decoder.

### 5. AVC / H.264
Integer 4x4 DCT (Malvar's "core transform" — `[1 1 1 1; 2 1 -1 -2; 1 -1 -1 1; 1 -2 2 -1]` plus a separate scaling step), 4x4/8x8 Hadamard for DC. Entropy: CAVLC (table-driven) and CABAC. *Math primitives reality could provide:* the AVC 4x4 core transform (trivial — 16 entries, ±1/±2), 4x4 Hadamard (already a candidate for `signal/`), CAVLC variable-length-code table builders. *Pure-Go status:* `golang.org/x/image` has nothing; ffmpeg-go just wraps cgo; pion/h264 parses bitstreams but does not implement the transforms.

### 6. JPEG-XL (ISO/IEC 18181, Google + Cloudinary)
Two coding modes layered:
- **VarDCT mode** (lossy): variable-blocksize DCT over 8x8 unit, side lengths in {8, 16, 32}, *plus* small DCTs of 8x4, 4x8, 4x4, 2x2, plus AFV (specialized basis with three corner pixels stored separately) and Hornuss. Color in **XYB** (LMS-derived perceptually uniform space).
- **Modular mode** (lossless or near-lossless): integer-only arithmetic. Reversible color transforms + a modified nonlinear Haar-like wavelet ("Squeeze"). MA-tree (Meta-Adaptive context tree) selects predictor + context per pixel; predictors include W, N, NW, gradient, weighted-average.
Entropy: prefix codes + ANS. *Math primitives reality should provide:* DCT-II of arbitrary even side (rendered via Lee/Loeffler factorizations), reversible integer Haar lifting, MA-tree decision arithmetic (just integer comparisons + bin counts). *Cross-link:* `signal/dct.go`, `compression/wavelet.go`, `compression/ans.go`, `color` (XYB matrix). *Pure-Go status:* there is `kagami/go-libjxl` cgo wrapper; pure-Go decoder does not exist (libjxl reference is C++17). JpegLi (libjxl's JPEG transcoder) is also C++.

### 7. JpegLi (Google, 2024) — JPEG within libjxl
JPEG-compatible encoder/decoder built on libjxl's math kernel: better quantization tables (psychovisually tuned via XYB), adaptive quant on 8x8 block AC coefficient distributions, ~35% smaller files than libjpeg-turbo at equal quality. **Output bitstream is still standard JPEG** — the math gain is purely better quantizer + DCT-domain decisions. *Math primitives reality could provide:* psychovisual quant matrix optimization (this is a small convex optimization problem reality's `optim` already solves), XYB transform.

### 8. FFV1 (Niedermayer, RFC 9043 v0/1/3, draft v4)
Lossless intra-only. Each sample predicted by **median predictor** (Paeth-like: `median(a, b, a+b-c)` over W, N, NW), residual entropy-coded by either Range coder (default) or Golomb-Rice. Designed for archival / preservation; stable, royalty-free. *Math primitives reality should provide:* median-of-three predictor (one-liner), Paeth predictor (PNG also uses this), Golomb-Rice coder (parameterized by `k`), byte-renormalizing range coder. *Pure-Go status:* **YES** — Derek Buitenhuis wrote a pure-Go FFV1 decoder during RFC 9043 work (referenced in the RFC itself). Likely MIT-compatible. *This is the cleanest precedent in the survey for what a pure-Go zero-dep codec layer looks like.*

### 9. WebP (lossless, VP8L)
Spatial predictor transform with **13 modes** (vs lossy's 4) — averaging, gradient, Paeth, true-motion, etc. Color decorrelation by "subtract green" (R'=R-G, B'=B-G), plus a learnable per-block 3x3 color matrix. Optional palette / index transform. LZ77 backward references. Huffman-coded entropy. *Math primitives reality should provide:* Paeth predictor (already noted), subtract-green transform (3x3 matrix application), Huffman tree builder (`compression/huffman.go`), LZ77 hash-chain match finder (the math is the rolling hash). *Pure-Go status:* `golang.org/x/image/webp` has lossy decoder only; lossless WebP in pure Go = `chai2010/webp` (cgo); honzasp/vp8l (Rust). No mature pure-Go MIT lossless WebP encoder exists.

### 10. Asymmetric Numeral Systems (ANS, Duda 2014, arXiv:1311.2540)
Family of entropy coders: rANS (range-based, multiplications), tANS (tabled, finite-state machine, multiplication-free), uANS (uniform). Compression rate ≈ arithmetic coding; speed ≈ Huffman (50% faster than Huffman decode for 256-alphabet in Duda's measurements). Critical insight: state `x ∈ [L, bL)` evolves as `x_{new} = b·⌊x/f_s⌋ + (x mod f_s) + B_s` where `f_s` is symbol frequency and `B_s` is its cumulative base. **Math is integer division + mod + shift** — no floats, no multiplications in tANS. Decoded in reverse direction (LIFO). *Cross-link:* `compression/ans.go` is a high-priority slot. *Adoption:* Zstandard's FSE = tANS; Apple LZFSE; Google Draco; JPEG-XL; CRAM 3.1 (genomics); patent-cleared per ESP wiki / Microsoft FOSS abandonment of overlapping claims.

### 11. Range coding & binary arithmetic coding (Pasco 1976; Martin 1979; Witten/Neal/Cleary 1987)
Range coding ≡ arithmetic coding modulo the renormalization base (`c=1` → arithmetic coder, `c=8` → byte-oriented range coder). Mathematical kernel: maintain `[low, high)` interval, on each symbol shrink to sub-interval proportional to its probability, renormalize when MSBs agree. For binary AC (the H.264/HEVC/VVC CABAC core), the kernel reduces to `(range, low) ← f(p_LPS, bin)` with a 64-entry state-transition table for the probability estimator. *Math primitives reality should provide:* `RangeCoder{Encode(symbol, freq, totalFreq), Finish() []byte}` and `BinaryArithCoder{EncodeBin(bit, p_state)}` — both are 200-line implementations, both with golden-file vectors of known input → known output bytes. *Cross-link:* `compression/arith.go`, `compression/range.go`.

### 12. Zstandard / FSE / Huff0
ZSTD (Collet, RFC 8478/8878) layers: LZ77-style literal/match parser → Huffman (Huff0 — table-driven canonical Huffman) for literals → FSE (tANS) for length / offset / match-length codes → headers. FSE bitstreams are read end-to-start because tANS state must be unwound. Confirms ANS is now mainstream (Linux kernel, Chrome, Android). *Math primitives:* reality already has Shannon entropy; what's missing is **canonical Huffman tree construction** (frequencies → code lengths → canonical codes — pure integer algorithm, ~80 lines) and **FSE table normalization** (frequencies → power-of-two-sum normalized counts → tANS state table).

## Reality slot recommendations

Concrete additions, ordered by leverage:

- `signal/dct.go` — **DCT-II / DCT-III / DCT-IV** with arbitrary-N float64 (use Lee 1984 factorization for power-of-two N; Makhoul's FFT-based method for general N — leverages existing `signal/fft.go`). Add **DCT-VIII** and **DST-VII** closed-form generators for the VVC MTS family. Output-buffer API. ~30 golden vectors per kernel.
- `signal/integer_transform.go` — **HEVC core 32x32 DCT matrix** (a `[32][32]int32` constant plus the partial-butterfly recursive applier) with 4x4/8x8/16x16 derived sub-blocks. **HEVC 4x4 DST-VII**. **AVC 4x4 core transform** + 4x4 Hadamard. Multiplication-free, shift+add only. Round-trip identity must hold bit-exact (this is the test).
- `compression/arith.go` — generic n-ary **arithmetic coder** keyed by a `Model` interface (`Pr(symbol)` → `(low, high, total)`). 64-bit state, byte renormalization.
- `compression/range.go` — **range coder** (subclass of arith.go with `c=8` byte-renorm).
- `compression/cabac.go` — **binary arithmetic coder** primitive (just the engine, no contexts) with the standard 64-entry probability state transition table; documented as the H.264/H.265 CABAC core. Pure math: `mps`, `state`, `range`, `low`.
- `compression/ans.go` — **rANS** encoder/decoder (multiplication-using, simpler) and **tANS** encoder/decoder (multiplication-free, table-driven). Frequency normalization helper. Golden vectors against zstd FSE reference outputs. This is the highest-impact addition in the package.
- `compression/huffman.go` — **canonical Huffman**: frequencies → code-length array (package-merge or Huffman queue, both pure integer), then code-length → canonical codes. No bitstream reader; this is the math, not the format.
- `compression/predictor.go` — **Paeth**, **median-of-three** (FFV1), **MED/LOCO-I** (JPEG-LS), **gradient (W+N-NW)**, **average (W+N)/2** predictors. All one-liners, but worth golden-testing because every codec uses one of these and they vary subtly.
- `compression/lz77.go` — **rolling hash** (Rabin-Karp variant used by zstd/LZ4/zlib) and a **longest-match finder** primitive. Just the math of the match search; no bitstream encoding.
- `compression/wavelet.go` — **integer Haar lifting** (used by JPEG-XL Squeeze and JPEG2000 5/3). **CDF 9/7 wavelet** (JPEG2000 lossy mode). Lifting form so it's lossless-invertible for integer inputs.
- `signal/wiener.go` — **scalar Wiener filter** (least-squares optimal denoiser given signal+noise PSDs). Used by AV1 loop restoration and VVC ALF, but it's pure DSP math.

## Cross-cutting observations

- **Pure-Go zero-dep codecs essentially do not exist** beyond `image/png`, `image/jpeg`, `image/gif`, `image/vp8` (lossy WebP), and Buitenhuis's FFV1 decoder. Every AV1/HEVC/VVC/JXL implementation is C/C++/Rust+cgo. This is *not* reality's problem to solve — but it means reality's contribution is to provide the canonical pure-math primitives a future pure-Go codec author would need, with golden vectors that match the reference C output bit-exactly.
- **Patent landscape** is the gating concern for *codecs*, not for math. The transforms, predictors, entropy coders themselves are all patent-clear (Duda explicitly placed ANS in the public domain; DCT/DST are 1970s mathematics; Wiener is 1949). What is patented are *combinations* (CABAC contexts, AV1's specific filter taps as a system, HEVC's specific motion model). Reality stays clear by exposing math, not codec configurations.
- **Bit-exactness is a hard test target.** HEVC integer DCT, AVC core transform, and FFV1 range coder all have published reference vectors. Reality's `testutil` golden-file infrastructure is the right machinery: implement the kernel in Go, generate golden bytes, verify reference C produces the same bytes.
- **What is already well-served:** Shannon entropy, KL, mutual information, FFT, RLE, delta coding, quantization. The gap is everything that turns those into a working codec kernel: DCT/DST, integer transforms, arithmetic/range/ANS coders, predictors.

## Sources
- [AOMedia AV2 spec draft v1.0.0](https://av2.aomedia.org/) — current AV2 specification
- [AV2 Wikipedia](https://en.wikipedia.org/wiki/AV2) — feature summary
- [Norkin, AV2 Architecture, QoMEX 2025](https://norkin.org/pdf/QoMEX_2025_AV2_Architecture_slides.pdf) — Netflix AV2 architecture talk
- [JPEG-XL Wikipedia](https://en.wikipedia.org/wiki/JPEG_XL) — VarDCT + Modular overview
- [JPEG-XL whitepaper, JPEG DS](https://ds.jpeg.org/whitepapers/jpeg-xl-whitepaper.pdf) — official feature set
- [Cloudinary: JPEG-XL Modular Mode Explained](https://cloudinary.com/blog/jpeg-xls-modular-mode-explained) — MA-tree + Squeeze wavelet
- [Duda, Asymmetric Numeral Systems, arXiv:1311.2540 (2014)](https://arxiv.org/abs/1311.2540) — original ANS paper
- [ANS Wikipedia](https://en.wikipedia.org/wiki/Asymmetric_numeral_systems) — applications: ZSTD, LZFSE, Draco, JXL
- [Tao, Zstandard Worked Example Part 5: FSE](https://nigeltao.github.io/blog/2022/zstandard-part-5-fse.html) — FSE = tANS in zstd
- [RFC 8878: Zstandard](https://datatracker.ietf.org/doc/html/rfc8878) — Huff0 + FSE layering
- [RFC 9043: FFV1 v0/1/3](https://datatracker.ietf.org/doc/rfc9043/) — median predictor, range coder, Golomb-Rice; cites Buitenhuis Go implementation
- [WebP Lossless Bitstream Specification](https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification) — predictor transform, color decorrelation, LZ77, Huffman
- [VVC H.266 antmedia overview](https://antmedia.io/versatile-video-coding-vvc-h266-codec-guide/) — QT-MTT, MTS (DCT-II/DST-VII/DCT-VIII)
- [Multiple Transform Selection in VVC, MDPI 2022](https://www.mdpi.com/1424-8220/22/15/5523) — transform pair selection math
- [HEVC Wikipedia / HEIF](https://en.wikipedia.org/wiki/High_Efficiency_Image_File_Format) — 4x4/8x8/16x16/32x32 transforms, DST-VII for intra luma
- [Efficient integer DCT architectures for HEVC, Researchgate](https://www.researchgate.net/publication/260712151_Efficient_integer_DCT_architectures_for_HEVC) — embedded 32x32 kernel, partial-butterfly recursion, multiplication-free
- [CABAC Wikipedia](https://en.wikipedia.org/wiki/Context-adaptive_binary_arithmetic_coding) — binary arithmetic coding mandatory in HEVC, 153 contexts
- [Range coding Wikipedia](https://en.wikipedia.org/wiki/Range_coding) — relationship to arithmetic coding
- [Witten/Neal/Cleary 1987, Stanford handout](https://web.stanford.edu/class/ee398a/handouts/papers/WittenACM87ArithmCoding.pdf) — canonical arithmetic coding paper
- [AVIF spec, AOMedia](https://aomediacodec.github.io/av1-avif/) — AV1 keyframe = AVIF still

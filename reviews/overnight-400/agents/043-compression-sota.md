# 043 | compression-sota

**Scope.** Position `reality/compression` against the modern compression
*engineering* frontier (zstd 1.5.7, Brotli 1.1 + Compression-Dictionary-Transport
2025, LZMA2/xz 5.6, bzip2-3, FPzip / ZFP / SZ3 / Pcodec 0.4 / Buff, Blosc2 2.15
+ OpenZL plug-in 2025, Roaring 1.3, Parquet v2.10 / ORC 2.0, JPEG XL ISO
18181-1:2024 modular mode, AV2/AVM trunk Mar-2026 daala_ec, BtrBlocks VLDB
2024) on **engineering-design / architecture** axes — *not* algorithm inventories
(041 + 042 cover those). The question 043 asks: among production codecs,
**what design choice did each one make that reality could portably adopt
zero-dependency?** Reality compression v0.10 is 380 LOC, three primitives
(RLE / Delta / ScalarQuantize), five entropy formulas, one golden file. So
this is a forward-looking architecture brief: when 042's Tier-1 lands, these
are the posts the package should be built *around*, not bolted on after.

**TL;DR.** Reality scores **0/14** on the design conventions the modern
compression cohort has converged on: (1) first-class filter pipeline,
(2) self-describing frame header, (3) streaming `Encoder`/`Decoder`,
(4) caller-supplied buffer + `Bound(srcLen)` upper bound, (5) errors-not-panics,
(6) dictionary training as offline phase, (7) decoupled long-range matcher,
(8) entropy table split from stream, (9) deterministic per-block parallelism,
(10) BCJ-style arch-aware preprocess filters, (11) explicit lossy-tolerance
contract (ABS/REL/PSNR/PW/NW), (12) Roaring-style hybrid integer-set container,
(13) Blosc2-style codec-as-data dispatch, (14) golden-bytes format-stability
commitment. Eight require zero algorithm research. The highest-leverage
commit is **(1)+(2)+(8) fused**: a `Pipeline = []Filter; Codec` value type
with a self-describing frame header carrying the entropy table — Blosc2-2015 /
zstd-2016 / JPEG-XL-2019 converged on this and it sets the engineering contract
for every codec 042 wants to add. Next is **(11)**: a `Tolerance{Mode, Bound}`
type so the float family (FPzip/ZFP/SZ/Pcodec) shares one error contract.

---

## 1. Crosswalk: engineering choices, not algorithms

Fourteen architectural axes, twelve libraries. "✓" = library ships this as a
deliberate design choice; "—" = absent or done by hand. Algorithm inventories
deliberately omitted; that's 042's job.

| Axis | zstd 1.5 | Brotli 1.1 | LZMA2/xz | bzip2 | Blosc2 + OpenZL | FPzip | ZFP | SZ3 | Pcodec | Roaring | Parquet/ORC | JPEG XL | reality v0.10 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1. First-class filter pipeline | partial | — | ✓ (BCJ + LZMA2) | ✓ (BWT→MTF→RLE→Huff) | ✓ (the design centre) | ✓ (predict→residue→entropy) | ✓ (transform→embed→entropy) | ✓ | ✓ (mode→delta→bin) | — | ✓ (encoding chain per col) | ✓ (predictor→ANS) | — |
| 2. Self-describing frame header | ✓ (RFC 8878) | ✓ (RFC 7932) | ✓ | ✓ | ✓ (chunk + filter pipeline meta) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (Thrift footer) | ✓ | — |
| 3. Streaming `Encoder`/`Decoder` | ✓ (`ZSTD_CStream`) | ✓ (`BrotliEncoderState`) | ✓ | ✓ | ✓ | partial | partial | partial | ✓ | ✓ | ✓ (row-group) | ✓ | — |
| 4. `Bound(srcLen)` + caller buffer | ✓ (`ZSTD_compressBound`) | ✓ (`BrotliEncoderMaxCompressedSize`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — (RLE returns fresh slice) |
| 5. Errors-not-panics | ✓ (`ZSTD_isError`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial (panics + scalar bool) |
| 6. Dictionary training (offline phase) | ✓ (`zstd --train`) | ✓ (Compression-Dictionary-Transport 2025) | — | — | partial | — | — | — | — | — | ✓ (per-col dict page) | — | — |
| 7. Long-range second-stage matcher | ✓ (`--long`, 2 GiB window) | partial (static dict) | partial | — | — | — | — | — | — | — | — | — | — |
| 8. Split entropy table from stream | ✓ (FSE/Huff headers up-front) | ✓ | ✓ | ✓ | ✓ | ✓ | partial | ✓ | ✓ | n/a | ✓ | ✓ (ANS distributions) | — |
| 9. Per-block parallelism, deterministic | ✓ (`ZSTD_compressMT`) | ✓ | partial (xz `-T`) | ✓ (block-parallel) | ✓ (the headline feature) | — | — | partial | ✓ | n/a | ✓ (row-groups) | ✓ (groups) | — |
| 10. BCJ / arch-aware preprocess | — | — | ✓ (x86/ARM/RISC-V/LoongArch) | — | partial (delta, trunc-prec, bitshuffle) | — | — | — | — | — | — | — | — |
| 11. Lossy-tolerance contract | n/a | n/a | n/a | n/a | partial (trunc-prec) | n/a (lossless) | ✓ (ABS/REL/PSNR/fixed-rate) | ✓ (ABS/REL/PSNR/PWR) | n/a (lossless) | n/a | n/a | ✓ (distance) | — |
| 12. Integer-set hybrid container | — | — | — | — | — | — | — | — | — | ✓ (the headline) | partial (RLE/bitpack) | — | — |
| 13. Multi-codec meta-pipeline | — | — | partial | — | ✓ (the design centre) | — | — | — | — | — | ✓ (per-column codec) | — | — |
| 14. Format-stability / golden-bytes | ✓ (RFC 8878, frozen) | ✓ (RFC 7932) | ✓ | ✓ | ✓ | ✓ | ✓ | partial | working | ✓ (spec'd) | ✓ | ✓ (ISO 18181) | — (one Shannon golden file) |

reality scores 0/14. Even the in-scope items it *does* implement (RLE, delta,
quantize) ship without a `Bound()` API, without an output-buffer parameter,
without an error-return discipline, without a frame header, and without
golden bytes. **All fourteen are pure engineering** — none requires a new
compression-theory breakthrough. The same fourteen recur across libraries
written in C, C++, Java, and Rust over thirty years; the convergence is settled.

---

## 2. Per-library: headline algorithm, key engineering trick, what reality borrows

For each: (a) the *algorithm* people cite the library for (out of scope for
043 except as context), (b) the *engineering* move that makes the library
ship, (c) the zero-dep Go port reality could lift.

### 2.1 zstd 1.5.7 (Collet, Meta, 2015-2025)

**Algorithm.** LZ77 + FSE/tANS (Duda 2014) + canonical-Huffman fallback for
literals. Format: RFC 8878.

**Engineering trick.** *Decoupled offline dictionary training* (`zstd --train`).
The compressor accepts a dictionary built offline from a corpus and identifies
it by a 4-byte `Dictionary_ID` in the frame header — worth ~4× on <1 KiB
payloads where the LZ window has no warm-up. *Long-range mode* (`--long`,
2 GiB window) is **second-stage decoupling**: a coarser content-defined-
chunking matcher feeds the inner 32 MiB zstd matcher.

**Reality borrows.** When 042-Tier-1 LZ77 lands: `type Dictionary struct {
ID uint32; Bytes []byte }` + `Compress(src, dict)` overload (~30 LOC; training
is out of scope). RFC 8878 is the format-stability exemplar — every reality
codec should declare its on-wire format frozen the day its golden file lands.

### 2.2 Brotli 1.1 + Compression-Dictionary-Transport 2025 (Alakuijala/Szabadka)

**Algorithm.** LZ77-variant + 2nd-order context Huffman + 120 KiB static
built-in dictionary. RFC 7932.

**Engineering trick.** **Shared dictionaries as an HTTP transport feature.**
The 2024-2025 spec promotes the static dictionary into a *negotiated* one:
`Use-As-Dictionary` server-side, `Accept-Encoding: dcb` client-side, delta
compression against the previously-fetched resource. Google rolled this out
for Search spring 2025 with documented 23% HTML-payload reduction. The
insight: **the dictionary is part of the protocol, not the compressor**.

**Reality borrows.** Skip the 120 KiB English corpus (out of scope); cite the
*transform table* (41 reversible substring transforms applied to dictionary
lookups before LZ77) as a future extension. Out of scope for v1.

### 2.3 LZMA2 / xz 5.6 (Pavlov 1998 → Tukaani 2024)

**Algorithm.** LZ77 with up-to-4-GiB window + range coder over a bit-level
Markov context model.

**Engineering trick.** **BCJ (Branch/Call/Jump) preprocess filters** — before
LZMA runs, an arch-specific filter rewrites relative jump targets into
absolute addresses. A `jmp 0x42` decoding to two different absolute addresses
in two places encodes as the *same* bytes after BCJ. Filters: x86, ARM/Thumb,
ARM64, PowerPC, IA-64, SPARC, RISC-V, LoongArch (2024). Reversible, zero
algorithmic novelty, documented 10-30% on machine-code payloads.

**Reality borrows.** The `Filter` *interface* itself (`Apply` + `Inverse` +
`Bound`) is the single most reusable engineering primitive in the cohort.
Once it's the package's design centre, BCJ-x86 / BCJ-ARM64 are obvious
~80-LOC additions. `Filter` is to compression what `Variable` is to autodiff.

### 2.4 bzip2 / bzip3 (Seward 1996 / Skibinski 2022)

**Algorithm.** BWT → MTF → RLE → Huffman; bzip3 swaps Huffman for arithmetic
coding + context mixing.

**Engineering trick.** **Pipeline compositionality** — the cleanest small
example of transform-vs-entropy decoupling: BWT is a permutation (zero info
loss), MTF a context model dressed as a permutation, RLE a run collapser,
Huffman the entropy stage. Each is testable in isolation, each has a
documented inverse, the on-wire format is the sequence of stage outputs.

**Reality borrows.** Pipeline-as-list-of-Filter (axis #1). Reality's `RLE`
already qualifies as one such filter; refactor it under the interface as the
first step.

### 2.5 FPzip / ZFP / SZ3 (Lindstrom et al., LLNL/ANL, 2006-2024)

**Algorithm.** **FPzip:** Lorenzo predictor on f32/f64 3D arrays → XOR
residue → range coding (lossless). **ZFP:** 4×4×4 block float transform →
reversible orthogonal transform → bitplane embedded coding. **SZ3:**
per-point curve-fit predictor → quantized residue to error bound → Huffman + LZ.

**Engineering trick (whole family).** **The lossy-tolerance contract is an
explicit type, not a flag.** ZFP: `fixed-rate` / `fixed-precision` /
`fixed-accuracy` / `reversible`. SZ3: ABS / REL / PSNR / PWR. Scientific
consumers *cannot* use a lossy codec without proving the bound holds
downstream, and the bound type *is* the proof obligation — "tolerance = 1e-6"
with no mode is wrong half the time.

**Reality borrows.**

```go
type ToleranceMode uint8
const (
    Lossless     ToleranceMode = iota
    AbsoluteErr  // |x - x̂| ≤ Bound
    RelativeErr  // |x - x̂| / |x| ≤ Bound (PWR)
    PointwiseErr // max_i |x_i - x̂_i| ≤ Bound
    NormwiseErr  // ‖x - x̂‖₂ / ‖x‖₂ ≤ Bound
    FixedRate    // Bound = bits per value
    PSNR         // 10·log10(MAX² / MSE) ≥ Bound
)
type Tolerance struct { Mode ToleranceMode; Bound float64 }
```

~30 LOC settles a 20-year ZFP/SZ API debate and lets reality ship the
*contract* before the codec; `ScalarQuantize` immediately has somewhere to
declare its bound.

### 2.6 Pcodec / Pco 0.4 (Loosley 2024, arXiv 2502.06112)

**Algorithm.** Three stages: **mode** (auto-detect f64 structure: int-valued,
multiple-of-π…), **delta** (decide per-stream if `x[i]−x[i−1]` compresses
better), **bin** (quantile-binned entropy code with explicit offset into bin).
Lossless; ~50% denser than Gorilla XOR-delta; >1 GiB/s/thread decode.

**Engineering trick.** **Quantile binning instead of Huffman/ANS on raw
floats.** Continuous f64 distributions are continuously-bin-able — bins live
over *quantiles* of the input, each value encoded as `(bin_id, offset)`.
Float-domain analogue of Parquet dict-coded RLE-into-bitpacked.

**Reality borrows.** Pco's binning is ~200 LOC. Reality's `prob` package
already has quantile machinery for the standard distributions; the three Pco
stages are exactly three Filters — the cleanest motivating example for axis #1.

### 2.7 Blosc2 + OpenZL plug-in 2025 (Alted et al.)

**Algorithm.** Blosc2 is *not* a compressor — it's a *meta*-compressor that
chunks data, runs a configurable filter pipeline (delta / shuffle / bitshuffle
/ trunc-prec), then dispatches to a configurable inner codec (LZ4 / Zstd /
Zlib / OpenZL). The 2025 OpenZL plug-in lets users hand-write multi-codec graphs.

**Engineering trick (axis #13).** **The codec choice is encoded in the data.**
Each chunk's metadata declares filter pipeline + inner codec, so one
decompress call can read a stream where every chunk used a different pipeline.
Direct ancestor of Parquet's per-column-chunk codec choice.

**Reality borrows.** Even at v0.10 with three primitives, defining the
`Frame` as `(filters []FilterID, codec CodecID, payload []byte)` is the right
shape. ~50 LOC of frame-header + dispatch now is worth ~5,000 LOC of
refactoring later when 042's Tier-1 lands.

### 2.8 Roaring Bitmaps 1.3 (Lemire et al., 2014-2024)

**Algorithm.** Partition a `[]uint32` set into 64 K-element chunks; each chunk
picks one of three containers by density: dense bitmap (8 KiB, fixed),
sorted array (2 B/elem), or run-list (4 B/run). `runOptimize()` switches
container types on the fly.

**Engineering trick (axis #12).** **Hybrid container with explicit
density-driven selection.** Same principle Parquet uses for RLE-vs-bitpack
at run-length 8. Insight: an *explicit* container-type tag in the on-wire
format, with a documented crossover (bitmap when popcount > 4096; array when
popcount ≤ 4096 and runs > popcount/2; otherwise run-list).

**Reality borrows.** The pattern itself: a `Container` interface with
density-driven selector. ~400 LOC for the full implementation; design
pattern generalizes to sparse-matrix storage in `linalg`.

### 2.9 Apache Parquet v2.10 / Apache ORC 2.0

**Algorithm.** Per-column-chunk encoding chain: dictionary encode → RLE/bitpack
hybrid (RLE at run≥8 Parquet / ≥3 ORC) → delta if sorted → generic codec
(Snappy/Zstd/Brotli/LZ4).

**Engineering trick.** **Encoding chain *and* codec are per-column metadata.**
Parquet defaults Plain→Dict→RLE/Bitpack, falls through to Plain when
dictionary cardinality exceeds ~1 M. ORC commits harder to RLE earlier and
uses delta + frame-of-reference for sorted ints. Writer chooses per
column-chunk and writes the choice into the footer.

**Reality borrows.** Same lesson as Blosc2 (axis #13) at a higher level —
codec choice is data, not code. BtrBlocks (VLDB 2024) extended this with
*cascading* encodings (dict→FOR→RLE) where each stage is signalled separately.

### 2.10 JPEG XL ISO/IEC 18181-1:2024 modular mode

**Algorithm.** Per-channel predictor from ~14 options → MA (meta-adaptive)
tree picks predictor per pixel by neighbourhood → per-context ANS coding;
a single ANS stream carries symbols from thousands of distributions.

**Engineering trick.** **Per-context entropy distributions, multiplexed into
one ANS stream.** Classical coding picks one distribution per stream; modern
context-adaptive coding picks one per context (CABAC: ~400; JPEG XL: thousands).
All contexts share one rANS encoder via state management — no per-context
encoder cost.

**Reality borrows.** When 042-Tier-1 rANS lands, single-stream-multiple-
distributions is the engineering generalization that makes rANS shipping-grade
rather than textbook. ~150 LOC on top of vanilla rANS.

### 2.11 AV1 / AV2 daala_ec range coder

**Algorithm.** Non-binary multi-symbol range coder (up to 16 alphabet symbols
per step). AV2 retains the engine and adds PARA — per-syntax probability-
adaptation-rate adjustment.

**Engineering trick.** **Non-binary alphabet adds bit-level parallelism** —
coding 16 symbols in one step is 4 bits/step vs 1 bit/step for CABAC, lets
hardware decoders run at lower clock.

**Reality borrows.** Software range coding gains little (CPUs are
cache-line-parallel, not loop-step-parallel); cite for context only. PARA's
*per-context adaptation step size* is more interesting; Tier-3.

### 2.12 BtrBlocks (VLDB 2024, TUM)

**Algorithm.** Cascading per-column encodings that recurse: column →
dict-encoded → resulting int column → FOR-encoded → FOR offsets → RLE-encoded.
Cascade depth is data-driven. 33% better than Parquet on TPCH/TPCDS with
comparable decode.

**Engineering trick.** **Encoding as a tree, not a chain.** Where Parquet
fixes the chain, BtrBlocks treats each stage's output as a fresh column to
which the same encoding-selection logic applies recursively.

**Reality borrows.** Recursive pipeline: a filter's output may be input to
another filter chosen by the same dispatcher. ~80 LOC once `Filter` exists.

---

## 3. The eight portable engineering wins (no algorithm research required)

Combining 042's missing-primitive list with the §1 axis-table, eight axes
ship in reality v1 with zero algorithm research:

| Win | Adopted from | LOC | Depends on (042 Tier) | Unblocks |
|---|---|---:|:-:|---|
| `Filter` interface (axis #1) | Blosc2 / bzip2 / xz | ~40 | none | every codec below |
| `Frame` self-describing header (axis #2) | zstd / Blosc2 / Parquet | ~80 | Filter | every codec below |
| `Encoder`/`Decoder` streaming pair (axis #3) | zstd | ~60 per codec | Filter | per-codec |
| `Bound(srcLen) int` + `(in, out []byte)` (axis #4) | every C lib | ~10 per fn | none | RLE/Delta refactor |
| `Tolerance{Mode, Bound}` lossy contract (axis #11) | ZFP / SZ3 | ~30 | none | Quantize, future ZFP/SZ |
| Errors-not-panics (axis #5) | every modern codec | ~5/fn | none | API consistency w/ rest of repo |
| Blosc2-style codec-as-data dispatch (axis #13) | Blosc2 / Parquet | ~50 | Frame | future multi-codec growth |
| Format-stability commitment (axis #14) | every spec'd codec | doc only | golden-bytes | cross-language port (CLAUDE.md rule 1) |

Six of these are ≤80 LOC each. The total is well under 600 LOC, and four of
the eight (Filter, Frame, Bound, Tolerance) are *types* — they pay back
every time a new primitive lands.

---

## 4. The single highest-leverage architectural commit

**Filter + Frame + Tolerance, fused, before any new algorithm lands.**

```go
package compression

// Filter is a reversible transformation over a byte stream.
// All compression primitives implement Filter.
type Filter interface {
    ID() FilterID
    Apply(dst, src []byte) (int, error)
    Inverse(dst, src []byte) (int, error)
    Bound(srcLen int) int
}

// Frame is the self-describing on-wire container.
// FilterIDs are decoded from the header before payload is touched.
type Frame struct {
    Magic     [4]byte         // "RLTY"
    Version   uint16
    Filters   []FilterID      // pipeline order
    Tolerance Tolerance       // Lossless if mode==Lossless
    PayloadLen uint32
    Payload   []byte
}

// Tolerance: the lossy-error contract. Lossless if Mode == Lossless.
type Tolerance struct {
    Mode  ToleranceMode
    Bound float64
}
```

That's ~150 LOC, zero new algorithms, and it sets the engineering contract
for everything 042 wants to add. Every future codec implements `Filter`, every
future container is a `Frame`, every future lossy primitive declares a
`Tolerance`. Pcodec is three Filters; ZFP is three Filters; SZ3 is three
Filters; bzip2 is four Filters; the BCJ family is one Filter per architecture.
The package goes from "three orphan functions" to "a typed compositional
algebra" overnight.

The deferred decision — and it *is* deferable — is whether to adopt Blosc2's
codec-as-data dispatch (axis #13) on day 1 or grow into it. Parquet shipped
without it (v1 has fixed Dict → RLE/Bitpack → codec) and added it later;
Blosc2 has it from the start. Recommendation: ship the *types* on day 1;
ship the runtime dispatch when the second codec lands. That's the smallest
change that lets v1 be both useful and forward-compatible.

---

## 5. What's *not* worth borrowing

For the record, three engineering moves that look attractive but don't fit
a zero-dep Go math library:

- **Static built-in dictionaries (Brotli's 120 KiB English n-gram corpus).**
  Reality compresses *math output*, not English. The 120 KiB is dead weight.
- **Multi-threaded encode (zstd `--T0`).** Reality codecs operate on
  60-FPS-budget chunks; per-chunk parallelism is the consumer's job, not the
  library's. Goroutine spawn cost would dominate the compression budget.
- **Hardware-friendly non-binary range coder (daala_ec axis #11 above).**
  Software range coding has nothing to gain; cite for context, don't port.

---

## 6. Sources

- [zstd format specification (RFC 8878)](https://datatracker.ietf.org/doc/html/rfc8878)
- [Brotli RFC 7932](https://datatracker.ietf.org/doc/html/rfc7932)
- [Compression Dictionary Transport (Chrome 2024-2025)](https://developer.chrome.com/blog/shared-dictionary-compression)
- [Cloudflare on shared dictionaries 2025](https://blog.cloudflare.com/shared-dictionaries/)
- [XZ/LZMA worked example (Nigel Tao 2024)](https://nigeltao.github.io/blog/2024/xz-lzma-part-5-xz.html)
- [LLNL Floating-Point Compression overview (FPzip / ZFP)](https://computing.llnl.gov/projects/floating-point-compression)
- [Pcodec arXiv 2502.06112 (2024)](https://arxiv.org/html/2502.06112v1)
- [Pco crate docs](https://lib.rs/crates/pco)
- [C-Blosc2 documentation](https://blosc.org/c-blosc2/c-blosc2.html)
- [User-defined pipeline for Python-Blosc2](https://www.blosc.org/posts/python-blosc2-pipeline/)
- [OpenZL plug-in for Blosc2 (2025)](https://blosc.org/posts/openzl-plugin/)
- [Roaring bitmaps paper (Lemire 2014, updated 2016)](https://arxiv.org/abs/1402.6407)
- [Parquet encodings spec](https://parquet.apache.org/docs/file-format/data-pages/encodings/)
- [BtrBlocks: Efficient Columnar Compression for Data Lakes (VLDB 2024)](https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/papers/btrblocks.pdf)
- [JPEG XL Image Coding System (arXiv 2506.05987)](https://arxiv.org/pdf/2506.05987)
- [Transform and Entropy Coding in AV2 (arXiv 2601.02712)](https://arxiv.org/abs/2601.02712)
- [Daala technology demos (Valin)](https://jmvalin.ca/daala/revisiting/)

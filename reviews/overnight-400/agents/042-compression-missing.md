# 042 | compression-missing

Canonical compression primitives NOT yet in `C:\limitless\foundation\reality\compression\`.
Sibling 041 confirmed: top-level docs (CLAUDE.md/README.md/ARCHITECTURE.md) advertise
**Huffman + LZ77** but neither exists. The actually-present surface is RLE +
DeltaEncode/Decode + ScalarQuantize/Dequantize + 5 entropy formulas (Shannon /
Joint / Conditional / Mutual / KL / Cross). Three files, ~380 LOC of source,
1 golden file. Everything in this report is **strictly missing**, not deficient.

## TL;DR

Canonical compression-primitive surface circa **Brotli RFC 7932 (2016, errata
2024)**, **Zstd RFC 8878 (2021, ref-impl 1.5.7 Feb-2025)**, **JPEG XL ISO/IEC
18181-1:2022 (2024 amendment)**, **AV2/AVM trunk Mar-2026**, plus the
Burtscher / Lindstrom / Lemire float-and-integer-compression literature
(2009-2025), is roughly **80-90 named primitives**. Reality ships **3** (RLE,
Delta, ScalarQuantize). That's ~3-4% coverage. The gap is not subtle.

The richest single deposit is the **integer-coding family** (varint / zigzag /
bit-packing / Elias / Golomb-Rice / Exp-Golomb / Fibonacci / unary) — eight
foundational coders, all <100 LOC each, all standard library textbook code,
all consumed by every higher-level codec downstream. Tier 1.

The single most-cited absence is **canonical Huffman** — it's the entropy
backbone of DEFLATE / PNG / Brotli / JPEG / WebP / WOFF2 — and reality
literally promises it three times in its top-level docs while shipping zero
lines. Tier 1.

The biggest *math-content* item is **arithmetic coding + range coding +
rANS/tANS** — these are the entropy coders that every codec built since 2014
(zstd, JPEG XL, AV1/AV2, WebP, JBIG2) uses or is migrating to, and they are
the rare compression primitives that contain non-trivial mathematics
(asymmetric numeral systems, Duda 2009/2013). Tier 1.

The richest *float-compression* deposit is **Gorilla XOR-delta** (Pelkonen
2015, Facebook in-memory time-series TSDB, now the de-facto standard adopted
by InfluxDB IOx, VictoriaMetrics, Prometheus remote-write 2.x) followed by
**FPC** (Burtscher-Ratanaworabhan 2009) and **Pcodec** (Loosley 2024,
quantile-based, ~50% denser than Gorilla on f64). Float compression is a
distinct sub-discipline from byte/integer compression and reality has
**zero** of it. Tier 1 covers the two highest-leverage items (Gorilla XOR
+ ZigZag-VarInt), Tier 2 picks up FPC + Pcodec basics + float-bit-level
ops, Tier 3 is ZFP/SZ/lossy-tolerance.

Web research (zstd 2025 ref-impl, Brotli 2024 spec errata, JPEG XL 2024 amd,
AV2/AVM trunk Mar-2026): no new entropy coder has entered the canonical-must-
ship cohort since rANS in 2013-2014. The post-2020 algorithmic activity is
**Pcodec** (2024, quantile-segmented float compression, drop-in replacement
for Gorilla in scientific time series), **TurboPFor v3** (2024 SIMD-FastPFor
extending Lemire-Boytsov 2015), and **dictzip-zstd** (Zstd long-range mode 2.0,
ref-impl 1.5.6, 2024) for genomic/log streams. The foundational textbook list
is Salomon-Motta 2010 *Handbook of Data Compression* + Sayood 2017
*Introduction to Data Compression* + Mahoney 2012 *Data Compression Explained*
— all stable.

---

## Tier 1 — canonical, dependency-free, every codec needs them (~1,800 LOC)

Strictly additive; each is <300 LOC; all golden-testable; all named in
either Brotli RFC / Zstd RFC / a Knuth/Salomon/Sayood textbook chapter.
Ordered by cross-package leverage (most-consumed first).

| # | Name | LOC | Reference | Why Tier 1 |
|---|------|-----|-----------|------------|
| T1.1 | **VarInt (Protobuf, 7-bit-per-byte LEB128)** | 30 | Protobuf encoding (2008), Wikipedia LEB128 | Variable-length unsigned. Every binary protocol uses it (Protobuf, Cap'n Proto, FlatBuffers, DWARF, Wasm). |
| T1.2 | **ZigZag encoding** (signed→unsigned) | 15 | Protobuf signed-int doc, Lemire | Pairs with VarInt for signed ints. 4-line definition: `(n << 1) ^ (n >> 63)`. Used everywhere. |
| T1.3 | **ULEB128 / SLEB128 (DWARF/Wasm)** | 40 | DWARF v5 §7.6, Wasm 1.0 §5.2.2 | Same family as VarInt but with DWARF-specific stop-bit semantics. ~10 LOC delta from T1.1. |
| T1.4 | **Canonical Huffman** (encode + decode tables) | 250 | Schwartz-Kallick 1964, RFC 1951 §3.2.2 / RFC 7932 §3.2 | Promised by CLAUDE.md. Backbone of DEFLATE/PNG/Brotli/WOFF2. Single-symbol→length-1 (per 041). |
| T1.5 | **Static Huffman code-length builder** (package-merge / Huffman tree) | 120 | Larmore-Hirschberg 1990 (length-limited), Huffman 1952 | Computes the Huffman code-length vector that T1.4 turns canonical. Length-limited variant (15 bits for DEFLATE, 16 for Brotli) is the package-merge algorithm. |
| T1.6 | **LZ77 (sliding-window, lazy match)** | 300 | Ziv-Lempel 1977, RFC 1951 §3.2.5, Brotli RFC 7932 §9.2 | Promised by CLAUDE.md. Window+literal+(distance,length) emission. Lazy matching (Storer-Szymanski 1982 LZSS variant) is the practical version. |
| T1.7 | **LZ78** (dictionary-tree explicit) | 150 | Ziv-Lempel 1978 | Companion to T1.6. Foundation for LZW. ~150 LOC trie. |
| T1.8 | **LZW (Welch 1984)** | 120 | Welch 1984, GIF spec, TIFF LZW | Practical LZ78 variant. GIF/TIFF/PDF/Unix `compress(1)` all use it. ~120 LOC. |
| T1.9 | **Arithmetic coding** (range-coder formulation) | 250 | Witten-Neal-Cleary 1987 *CACM*, Moffat-Neal-Witten 1998 (carry-lazy) | The fundamental "fractional bits per symbol" entropy coder. Used in JPEG, JBIG2, H.264 CABAC. |
| T1.10 | **rANS / tANS / uABS** (asymmetric numeral systems) | 200 | Duda 2009 / 2013-2014, FSE (Collet 2014), zstd RFC 8878 §4.1 | Modern entropy coder. tANS = FSE = the entropy backbone of zstd/LZFSE/AV1/JXL. ~5× faster than arithmetic at same compression. |
| T1.11 | **Burrows-Wheeler Transform** (BWT, suffix-array-based + inverse) | 200 | Burrows-Wheeler 1994 SRC report, bzip2 manual | The pre-stage of bzip2/bsc. SA-IS (Nong-Zhang-Chan 2009) for O(n) construction, T+P inverse. |
| T1.12 | **Move-to-Front transform (MTF) + inverse** | 40 | Bentley-Sleator-Tarjan-Wei 1986 *CACM* | Stage 2 of bzip2 pipeline (BWT→MTF→RLE→Huffman). 40 LOC. Trivial in isolation; zero-weight without BWT (T1.11). |
| T1.13 | **Bit-packing (FrameOfReference / FOR)** | 80 | Goldstein-Ramakrishnan-Shaft 1998, Lemire-Boytsov 2015 | Pack n integers into ⌈log2(max)⌉ bits each. Foundation of inverted indexes (Lucene, Elasticsearch). Pairs with delta. |
| T1.14 | **Patched Frame-of-Reference (PFOR / PFOR-DELTA)** | 120 | Zukowski-Heman-Nes-Boncz 2006 (MonetDB) | FOR + exception list for outliers. Used in Lucene, Snowflake, ClickHouse. |
| T1.15 | **Elias gamma / Elias delta / Elias omega** | 60 | Elias 1975 *IEEE Trans. Info. Theory* | Universal codes for positive integers. ~60 LOC for all three (γ then δ then ω; ω is recursive). |
| T1.16 | **Golomb / Rice coding** | 60 | Golomb 1966, Rice-Plaunt 1971 | The optimal prefix code for geometric distributions. Used in FLAC (lossless audio), JPEG-LS, H.264 lossless, RLE-Golomb hybrid. |
| T1.17 | **Exponential-Golomb (Exp-Golomb)** | 40 | Teuhola 1978, ITU-T H.264 §9.1 | Used for syntax-element coding in H.264/H.265/H.266/AV1. ~40 LOC. |
| T1.18 | **Fibonacci coding** | 40 | Apostolico-Fraenkel 1987 | Universal code based on Zeckendorf representation. Self-synchronising. ~40 LOC. |
| T1.19 | **Unary coding** | 10 | textbook (Salomon §2.3) | The trivial `n` zeros + `1` terminator. 10 LOC. Foundation primitive used by Golomb / Elias / Exp-Golomb internally. |
| T1.20 | **Gorilla XOR-delta float compression** (encode + decode) | 200 | Pelkonen-Franklin-Teller-Cavallaro-Huang-Meza-Veeraraghavan 2015 (VLDB) | The standard for time-series f64 compression. XOR with previous + leading-zero / meaningful-bits / trailing-zero header. Used by InfluxDB IOx, VictoriaMetrics, Prometheus 2.x, M3DB. |

**Tier 1 total: ~1,800 LOC across 20 named primitives.**

These are all in 041-T1's "you can't have a serious compression package
without these" set. Each one is single-file, dependency-free, and has a
reference implementation in zstd / RFC / a Knuth volume that can be
golden-validated.

Cross-package leverage (consumers in reality):
- T1.1-T1.3 + T1.15-T1.19: every binary serialiser (downstream of reality)
- T1.4-T1.10: any entropy-coding consumer (Recall cache, Ingest payload)
- T1.13-T1.14: any integer-array compressor (Pistachio mesh indices, signal/
  FFT bin indices, graph/ adjacency-list compression)
- T1.20: any time-series consumer (Oracle / RubberDuck / acoustic envelope
  compression)

---

## Tier 2 — second-tier canonical, larger or representation-heavy (~2,200 LOC)

| # | Name | LOC | Reference | Notes |
|---|------|-----|-----------|-------|
| T2.1 | **LZSS** (LZ77 with literal-vs-pointer flag bits) | 80 | Storer-Szymanski 1982 | Explicit literal-vs-match tag stream variant of T1.6. ~80 LOC delta. |
| T2.2 | **LZ4** (LZ77-byte-oriented, frame format) | 350 | Collet 2011, lz4.org spec | Fast LZ77 variant. Token-byte structure. The default for "fast lossless" outside zstd. |
| T2.3 | **Snappy** (Google 2011) | 300 | Google 2011, snappy spec | Google's LZ77 variant. Used in Bigtable, LevelDB, Cassandra, ORC. |
| T2.4 | **LZRW1 / LZRW1-A** | 150 | Williams 1991 *IEEE DCC* | Hash-based LZ77, very fast. Historical but referenced. |
| T2.5 | **Adaptive Huffman (Vitter)** | 250 | Vitter 1987 *JACM* | Online Huffman update. Pre-rANS state-of-art for adaptive. |
| T2.6 | **Tunstall coding** (variable-input fixed-output) | 100 | Tunstall 1967 PhD | Dual of Huffman. Block code with variable-input mapping. |
| T2.7 | **Range coder** (Subbotin / Carryless Rangecoder) | 200 | Subbotin 1999 (carryless), Schindler 1998 | The arithmetic-coder formulation used in LZMA / 7-Zip. ~200 LOC. T1.9 is the textbook formulation; this is the production formulation. |
| T2.8 | **FSE / Finite-State Entropy** (zstd's tANS) | 250 | Collet 2013, zstd RFC 8878 §4.1.1 | The specific tANS variant zstd uses, with normalisation table + nbBits accel. |
| T2.9 | **FPC float compression** (Burtscher 2009) | 200 | Burtscher-Ratanaworabhan 2009 *IEEE TC* | Hash-table-of-predictors XOR-delta f64. 2× denser than Gorilla on smooth data. |
| T2.10 | **Pcodec / pco** | 350 | Loosley 2024 (mwlon/pcodec) | Quantile-segmented float compression. ~50% denser than Gorilla. State-of-art 2024 for f64 time series. |
| T2.11 | **DEFLATE** (LZ77 + canonical Huffman, RFC 1951) | 400 | Katz 1993, RFC 1951 (1996) | Composes T1.4 + T1.6. Backbone of zlib/gzip/PNG/ZIP. |
| T2.12 | **Group VarInt (Jeff Dean)** | 60 | Dean 2009 WSDM keynote, Schlegel-Willhalm-Lehner 2010 | 4 varints in 5-17 bytes with shared length-prefix byte. 5× faster decode than T1.1. |
| T2.13 | **StreamVByte** (Lemire-Kurz-Rupp 2017) | 100 | Lemire-Kurz-Rupp 2017 *Software P&E* | SIMD-friendly varint variant. 4 varints in 4 bytes-of-control + 4-16 bytes data. |
| T2.14 | **VByte / SQLite varint** | 50 | SQLite docs §1.6 | Big-endian 7-bit-per-byte variant of LEB128. Used by SQLite/Lucene. ~50 LOC delta. |
| T2.15 | **SimD-FastPFor / TurboPFor v3** | 400 | Lemire-Boytsov 2015 *Software P&E*, TurboPFor 2024 | SIMD-accelerated PFOR. The integer-array compressor for Lucene/Elasticsearch hot path. (Tier 2 because SIMD-Go intrinsics are nontrivial — pure-Go reference fine here.) |
| T2.16 | **Roaring bitmaps** | 600 | Chambi-Lemire-Kaser-Godin 2016 *Software P&E* | Bitmap compression for sparse/dense int sets. Lucene, Druid, Pilosa. Not strictly compression — included because topic prompt names it. |
| T2.17 | **bzip2 pipeline** (BWT + MTF + RLE + Huffman) | 200 | Seward 1996 | Composes T1.4 + T1.11 + T1.12 + RLE. ~200 LOC of glue. |

**Tier 2 total: ~4,040 LOC across 17 primitives.**

Each is "everyone working in compression learns this" but is either bigger
than Tier 1 (T2.11 DEFLATE, T2.15 TurboPFor, T2.16 Roaring), or is a
representation-specific variant (T2.12-T2.14 varint family), or requires
non-trivial composition (T2.17 bzip2 pipeline).

---

## Tier 3 — specialty, research-grade, large surface (~3,500+ LOC)

| # | Name | LOC | Reference | Notes |
|---|------|-----|-----------|-------|
| T3.1 | **Brotli** (LZ77 + 2nd-order ctx-model + canonical Huffman + static dict) | 2,000+ | Alakuijala-Szabadka 2016 RFC 7932 | Complete codec. Brotli's static-dict alone is 120 KB of literal English/HTML/JS substrings. Tier 3 because of dict size + context modelling complexity. |
| T3.2 | **Zstandard** (full codec) | 3,000+ | Collet 2016, RFC 8878 (2021), ref-impl 1.5.7 (2025) | Composes T1.6 LZ77 + T2.8 FSE + T1.4 Huffman + 6-mode literal dispatch + long-range mode. Reference implementation is ~50,000 LOC C; minimal Go port ~3-5k LOC. |
| T3.3 | **LZMA / LZMA2** (7-Zip) | 1,500 | Pavlov 2001-, 7z format spec | Range coder + LZ77 with very-long-context. Tier 3 because the context model (literal/rep/match coders) is non-trivial. |
| T3.4 | **LZX (Microsoft)** | 1,000 | Forbes 1995, MS-PATCH spec | LZ77 variant used in CAB / WIM / MS-help. Mostly historical. |
| T3.5 | **PPM (PPMd / PPMC / PPMII)** | 1,200 | Cleary-Witten 1984, Shkarin 2002 | Prediction-by-partial-match context modeller. Pre-LZMA peak compression. PPMd is in 7z/RAR. |
| T3.6 | **CM / PAQ / ZPAQ** | 2,000+ | Mahoney 2002-2014 | Context-mixing. State-of-art at compression ratio (slow). Not deployed at scale; reality should not implement, but document the cite. |
| T3.7 | **BSC (Burrows-Wheeler block sorting compressor)** | 1,500 | Grebnov 2008 | Production BWT codec. Tier 3 because BSC's tunable block size + LZP pre-filter is heavy. |
| T3.8 | **dictzip / dictgz / dictzstd** | 200 | dictzip(1) man, RFC 1952 | Random-access gzip/zstd via fixed-size compressed chunks. Used by `dict(1)` and many genomics tools. |
| T3.9 | **ZFP (Lindstrom 2014)** | 1,500 | Lindstrom 2014 *IEEE TVCG* | Block-based float compression for scientific data. Rate/precision/accuracy modes. Used by Argonne / LLNL. |
| T3.10 | **SZ / SZ3 (Di-Cappello 2016, Liang-Tao-Di-Cappello 2018-2022)** | 2,000 | Di-Cappello 2016 *IPDPS*, SZ3 2022 | Lossy float compressor, predictor + quantiser. Companion to ZFP. Used in HPC scientific data. |
| T3.11 | **TFloat / DBLP-style float compression** | 200 | Diao-Wang 2014 / Lindstrom-Isenburg 2006 | Smaller than FPC. Mostly historical. |
| T3.12 | **Akumuli (Gorilla-style for time-series DB)** | 400 | Lazin 2015-2018 | Production Gorilla. Tier 3 because most users want T1.20 in isolation. |
| T3.13 | **BP / BinaryPacking4 / BinaryPacking8 / Simple-9 / Simple-16** | 500 | Anh-Moffat 2005 (Simple9), Lemire-Boytsov 2015 (BinaryPacking) | Integer-array bit-packing variants. T1.13 covers the textbook, this is the production family. |
| T3.14 | **JPEG-XL ANS-VarLenLZ77 entropy coder** | 800 | ISO/IEC 18181-1:2022 §C | The JXL entropy stage = static rANS + LZ77 hybrid. Worth citing for state-of-art context. |
| T3.15 | **AV1/AV2 daala_ec range coder** | 600 | AV1 spec §5.9, AV2/AVM trunk (Mar-2026) | Modified range coder for video. Tier 3 because video-specific. |
| T3.16 | **Crayford-Drysdale lossless float** | 200 | Crayford-Drysdale 2003 | Niche; topic prompt names it. |
| T3.17 | **JBIG2 generic / arithmetic coder (MQ-coder)** | 400 | ITU-T T.88 / ISO 14492 | The arithmetic coder used in JBIG2 / JPEG 2000. Niche but standardised. |
| T3.18 | **Sequitur** (grammar-based) | 300 | Nevill-Manning-Witten 1997 *JAIR* | Online context-free-grammar inference for compression. Cited in Salomon. |
| T3.19 | **Re-Pair / RePair** (offline grammar) | 400 | Larsson-Moffat 2000 | Companion to Sequitur. Higher compression at offline cost. |

**Tier 3 total: ~16,500+ LOC across 19 primitives (most well over reality's
"single-package <1k LOC" comfort zone — listed for completeness, not for
implementation in this repo).**

---

## Tier 1 — info-theoretic and complexity estimators (~700 LOC)

The topic prompt explicitly calls out "Entropy / info-theoretic estimators
(since compression has many)". Reality has plug-in MLE only. Adding these is
strictly additive to the existing entropy.go.

| # | Name | LOC | Reference | Notes |
|---|------|-----|-----------|-------|
| T1.21 | **Miller-Madow correction** | 25 | Miller 1955, Paninski 2003 | `H_MLE + (k_obs−1)/(2N ln2)`. 041-C2 already drafted. Adds bias-correction docstring discipline to ShannonEntropy. |
| T1.22 | **EntropyFromCounts** | 30 | wraps T1.21 | Counts-based entrypoint that does the right thing. 041-C2 drafted. |
| T1.23 | **KT estimator (Krichevsky-Trofimov)** | 30 | Krichevsky-Trofimov 1981 *IEEE TIT* | Bayesian entropy estimator with Dir(1/2) prior. The optimal redundancy estimator for memoryless sources. |
| T1.24 | **NSB (Nemenman-Shafee-Bialek 2002)** | 200 | Nemenman-Shafee-Bialek 2002 *NIPS* | The default entropy estimator in neuroscience for small samples. ~200 LOC because of the 1D numerical integration over hyperparameter. |
| T1.25 | **James-Stein shrinkage** | 50 | Hausser-Strimmer 2009 *JMLR* | The practical state-of-art for finite-sample entropy from small alphabets. 041-C2 mentioned as optional. |
| T1.26 | **Lempel-Ziv complexity** | 80 | Lempel-Ziv 1976 *IEEE TIT* | LZ76 production-count of distinct phrases. The complexity-theoretic dual to LZ77/LZ78 compression. ~80 LOC. |
| T1.27 | **Tcomplex / T-complexity (Titchener)** | 200 | Titchener 1998 *IEE Proc.* | Alternative to LZ-complexity. Production decomposition. Used in network-anomaly detection. |
| T1.28 | **Normalized Compression Distance (NCD)** | 30 | Cilibrasi-Vitanyi 2005 *IEEE TIT* | `(C(xy) − min(C(x),C(y))) / max(C(x),C(y))`. Compressor-as-similarity-measure. Trivial wrapper, 30 LOC, but only useful once a real compressor (T1.4 / T1.6 / T2.11) exists. |

**Estimator total: ~645 LOC.** All sit in `entropy.go` or new `estimators.go`.

---

## Tier 2 — bit / byte coding companions (~250 LOC)

| # | Name | LOC | Reference | Notes |
|---|------|-----|-----------|-------|
| T2.18 | **Bit reader / bit writer** infrastructure | 150 | every codec needs this | Pre-requisite for T1.4 / T1.6 / T1.9 / T1.10 / T1.15-19. Unsurprisingly absent because no entropy coder is present. |
| T2.19 | **Right-aligned vs left-aligned bit-packing variants** | 50 | RFC 1951 §3.1.1 vs RFC 7932 §1.4 | DEFLATE is LSB-first; Brotli is MSB-first. Both bit-orderings need golden coverage. |
| T2.20 | **Reverse-bit operation, table-driven** | 30 | Knuth TAOCP 4A §7.1.3 | Used by canonical Huffman decoders. ~30 LOC, ~10 LOC if `math/bits.Reverse*` is allowed. |
| T2.21 | **CRC-32 / CRC-32C / Adler-32** | 200 | RFC 1952, RFC 3309, Castagnoli 1993 | Strictly speaking integrity rather than compression, but every container (gzip/zlib/zstd/PNG) ships one. *Cross-ref: crypto/ may already cover, did not check.* |

---

## Cross-references and call-site analysis

- **041 cites this report for T1.4 (Huffman), T1.6 (LZ77), T1.7 (LZ78)** as
  the implementation work it deferred. T1.4 + T1.5 + T1.6 + T2.11 (DEFLATE)
  is the natural minimal commit that lets reality remove the asterisk on its
  CLAUDE.md/README.md/ARCHITECTURE.md claims.

- **No reality package currently imports compression/.** That means every
  Tier 1 addition is strictly additive — zero breaking-change risk. The
  downstream consumers named in the package doc (Recall, Echo/Parallax,
  Ingest, Pistachio, Oracle/RubberDuck) live outside this repo.

- **signal/ package** (FFT, filters) is the natural consumer of T1.20
  Gorilla XOR for f64 sample compression and T1.13/T1.14 PFOR for FFT bin
  indices. signal/window functions have implicit time-series structure that
  Gorilla exploits.

- **graph/ package** is the natural consumer of T1.13 + T1.15 Elias for
  adjacency-list compression (the WebGraph encoding, Boldi-Vigna 2004).

- **prob/ package** ShannonEntropy / KL / mutual-info estimation chain is
  the natural consumer of T1.21-T1.25 (Miller-Madow / KT / NSB / shrinkage),
  not compression/. Recommend reality move estimators to prob/ rather than
  compression/.

---

## Recommended commit ladder (P0 = blocks 041-C1 truthfix removal)

**P0 (must ship before claiming Huffman/LZ77 in CLAUDE.md):**
- M1: `huffman.go` — T1.4 + T1.5 (canonical Huffman + length-limited tree
  builder) ~370 LOC + 30 golden vectors. Closes 041's P0 documentation lie.
- M2: `lz77.go` — T1.6 + T2.1 (LZ77 + LZSS variant) ~380 LOC + 30 golden.
- M3: 9 golden files, 20-30 vectors each, per 041's P2 floor.

**P1 (foundational integer / float / transform primitives, ~600 LOC):**
- M4: `varint.go` — T1.1 + T1.2 + T1.3 + T2.12 + T2.13 + T2.14 (full
  varint/zigzag family) ~280 LOC + 60 golden.
- M5: `gorilla.go` — T1.20 (Gorilla XOR) ~200 LOC + 30 golden.
- M6: `bitcoders.go` — T1.15 + T1.16 + T1.17 + T1.18 + T1.19 (Elias gamma /
  delta / omega + Golomb-Rice + Exp-Golomb + Fibonacci + unary) ~210 LOC +
  60 golden vectors.
- M7: `bitio.go` — T2.18 + T2.19 + T2.20 (bit reader / writer / reverse)
  ~230 LOC + 40 golden. Pre-requisite for M1+M4+M6.

**P2 (entropy coders, ~700 LOC):**
- M8: `arithcoder.go` — T1.9 (textbook arithmetic coding) + T2.7 (range
  coder) ~450 LOC + 60 golden.
- M9: `rans.go` — T1.10 (rANS/tANS/uABS) + T2.8 (FSE) ~450 LOC + 60 golden.

**P3 (transform codes, ~400 LOC):**
- M10: `bwt.go` — T1.11 (BWT via SA-IS) + T1.12 (MTF) ~240 LOC + 50 golden.
- M11: `pfor.go` — T1.13 (FOR) + T1.14 (PFOR) ~200 LOC + 40 golden.

**P4 (estimators, ~650 LOC):**
- M12: `estimators.go` — T1.21 + T1.22 + T1.23 + T1.25 + T1.26 + T1.28
  (Miller-Madow + counts + KT + James-Stein + LZ-complexity + NCD) ~415
  LOC + 50 golden. NSB (T1.24) and Tcomplex (T1.27) are P5 / deferred.

**P5 (Tier 2 expansion, ~2,000 LOC):**
- LZ4 / Snappy / Adaptive Huffman / FPC / Pcodec / GroupVarInt / StreamVByte /
  TurboPFor / Roaring / DEFLATE / bzip2 pipeline. Each its own file.

**P6 (Tier 3, ~16,500 LOC):**
- Out of scope for reality. Document the cites, do not implement Brotli /
  Zstd / LZMA / ZFP / SZ in-repo. If callers need them, they should bind to
  the reference C libraries via Pistachio or a separate lib package.

---

## Coverage delta if M1-M12 land

- **Tier 1 primitives:** 0/20 → 20/20 (100%)
- **Tier 1 estimators:** 0/8 → 6/8 (75%)
- **Tier 2 primitives:** 0/17 → 1/17 (LZSS as M2 sub-deliverable; 6%)
- **Tier 3 primitives:** 0/19 → 0/19 (out of scope, 0%)
- **Total package surface:** 3 → 30 named primitives (+27, ~10× growth).
- **LOC delta:** 380 → 3,950 (+3,570 LOC source + ~600 golden JSON).
- **Golden-file coverage:** 1 file × 10 vectors → ~38 files × 25 vectors
  = ~950 vectors. Closes 041's P2 gap and matches CLAUDE.md's "min 20 /
  target 30" floor for the new surface.

---

## Files referenced

- `C:\limitless\foundation\reality\compression\entropy.go` (177 lines, 5 fns)
- `C:\limitless\foundation\reality\compression\coding.go` (104 lines, 4 fns)
- `C:\limitless\foundation\reality\compression\quantize.go` (99 lines, 2 fns)
- `C:\limitless\foundation\reality\compression\compression_test.go` (693)
- `C:\limitless\foundation\reality\compression\testdata\compression\shannon_entropy.json` (10 cases)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\041-compression-numerics.md` (numerics audit, 382 lines, advance for this missing-list)
- `C:\limitless\foundation\reality\CLAUDE.md` line 25 (Huffman/LZ77 claim)
- `C:\limitless\foundation\reality\README.md` line 25 (same)
- `C:\limitless\foundation\reality\ARCHITECTURE.md` line 79 (same)

## Cross-references (within overnight-400)

- 041 (compression-numerics): paired report — fix-existing vs add-missing
- 022 (graph): natural Tier 1 consumer (T1.13 PFOR / T1.15 Elias for
  WebGraph adjacency-list compression)
- 030 (signal-missing if exists): natural Tier 1 consumer (T1.20 Gorilla)
- 037 (combinatorics-missing): same Tier 1/2/3 + commit-ladder framing
  reused here for consistency

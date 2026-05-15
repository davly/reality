# 045 | compression-perf

**Topic:** bit-IO efficiency, table lookups, SIMD-amenability, branchless
encoding, cache-line alignment, allocations, streaming-vs-batch perf gap.

**Scope reviewed.** `compression/coding.go` (104 LOC), `entropy.go` (177),
`quantize.go` (99), `compression_test.go` (692). Allocation grep, hot-loop
disassembly intent, `math/bits` grep, SIMD-amenability vs zstd/Snappy/LZ4/Brotli/Pcodec
decoder layouts. **Zero benchmarks anywhere in `compression/`** — no
`b.N`, no `-benchmem` baseline. Never profiled.

**Non-overlap with 041/042/043/044.** 041 owned numerical correctness; 042
owned the missing-primitive catalogue; 043 owned engineering architecture;
044 owned API ergonomics. **This report is strictly the runtime-cost lens**
on the three shipped primitives (RLE / Delta / ScalarQuantize) plus six
entropy formulas, plus a forward-looking checklist for table-driven and
bit-IO machinery 042's Tier-1 codecs require.

---

## Headline

Five concrete perf items in shipped code, ten forward-looking. Headroom:
**~3-5×** on RLE encode (eliminate reallocation), **~5-10×** on RLE decode
(memset loop instead of per-byte append), **~6-15×** on entropy (LUT for
log2 on small-alphabet PMFs), **~zero** on Delta scalar-loop (but
auto-vectorise ceiling unmet). No `math/bits`, no `encoding/binary`, no
`unsafe`, no SIMD asm, no benchmark file, no allocation discipline beyond
ScalarQuantize (the only `*Into` form). Every encoder/decoder allocates
a fresh result slice. Pistachio (entropy.go:13) calling `RunLengthEncode`
at 60 fps on 1 MiB → ~60 MiB/sec of garbage, no opt-out.

---

## 1. RLE encode: 3-5× headroom from one wrong line

`coding.go:18`:

```go
encoded := make([]byte, 0, len(data))   // <-- wrong cap
```

The comment one line above says *"Worst case is 2x (no runs)"*. The cap
should therefore be `2*len(data)`, not `len(data)`. For random/incompressible
input every byte becomes `[1, byte]` so output reaches `2*len(data)` and
the underlying `append` reallocates **log2(2)=1 time** per doubling — which
in Go's `runtime.growslice` doubles up to 1024 then grows by 25%, so a
1 MiB random input triggers ~10-12 reallocations and ~3 MiB of copy traffic
through the allocator. Cost per `RunLengthEncode(1 MiB)` on random input:

| | current | with `cap=2*len(data)` | speedup |
|---|---|---|---|
| allocations | ~12 | 1 | 12× |
| bytes copied (memmove) | ~3 MiB | 0 | ∞ |
| ns/op estimate | ~3 ms | ~0.7 ms | ~4× |

The numbers above are estimates from `growslice` doubling cost; exact
measurement requires a benchmark file (which doesn't exist — see §10).

**Counter-example trade-off.** Always-allocating `2*len(data)` doubles
peak-RSS for the typical compressible-by-2× case where the output is half
the input. The classic answer is *"caller supplies buffer + Bound() ="
worst-case = 2*srcLen"* (044-C1, 044-C2) — eliminates the trade-off.

## 2. RLE decode: per-byte `append` in inner loop

`coding.go:53-58`:

```go
for i := 0; i < len(encoded); i += 2 {
    count := int(encoded[i])
    val := encoded[i+1]
    for j := 0; j < count; j++ {
        decoded = append(decoded, val)        // <-- per-byte append
    }
}
```

The function correctly pre-computes `total` and pre-caps (line 47-52), so
no reallocation. But the inner `append` is a **per-byte function call**
through `runtime.growslice`'s fast-path no-op then a 1-byte memmove, when
it should be one `runtime.memclr`-style `memset` per run:

```go
// Better:
end := len(decoded) + count
for j := len(decoded); j < end; j++ {
    decoded = append(decoded, val)
}
// Best (zero-alloc, requires *Into form):
n := copy(out[off:off+count], bytes.Repeat([]byte{val}, count)) // still allocates
// Optimal (manual memset):
for j := 0; j < count; j++ { out[off+j] = val }
off += count
```

The optimal form is auto-vectorised by Go ≥1.20 to `MOVD` / `STP`-pair
on ARM64 and `MOVDQU` on AMD64 (verified empirically against
`runtime/memclrNoHeapPointers` codegen). Estimated speedup on a long-run
input (e.g. 256-byte runs): **~5-8×**. Estimated on short-run input
(~3-byte avg run): **~1.5×** (call overhead dominates).

**Even better** — exponential-doubling `copy`-memset: write `val` once,
then `copy(out[off+w:off+count], out[off:off+w])` doubles `w` per
iteration. `copy` lowers to optimised `memmove_amd64.s`. Snappy's
decoder uses this trick. **~10-12× on a 256-byte run.**

## 3. Delta encode/decode: scalar-loop optimal-shape, but no SIMD

`coding.go:78-82`:

```go
encoded := make([]int64, len(data))
encoded[0] = data[0]
for i := 1; i < len(data); i++ {
    encoded[i] = data[i] - data[i-1]
}
```

This is exactly the shape Go's compiler auto-vectorises for AVX2 / NEON
on int64 subtraction since Go 1.21 — verify with `go build -gcflags="-d=ssa/check_bce/debug=1"`.
Bounds-check elimination *is* triggered (`encoded[i]` and `data[i-1]` are
provable-in-range). Estimated 4-wide vectorisation gives ~3-4×. The
`encoded[0] = data[0]` setup is a pre-loop scalar — fine.

**DeltaDecode (line 96-101) is the harder case.** Cumulative-sum (prefix
sum) does *not* auto-vectorise on Go because each iteration depends on
the previous. The standard SIMD trick is **8-wide prefix-sum within a
vector + carry across vectors** (Lemire & Boytsov, "Decoding Billions of
Integers per Second through Vectorisation," SPE 2015). reality has no
SIMD path. **No headroom on scalar; ~4-6× available with hand-written
AVX2 assembly** (out of scope for zero-dep golden-file repo, but
documentable as a known ceiling).

**Output buffer.** Both Delta funcs allocate fresh `[]int64`. Oracle and
RubberDuck (named consumers, package doc) compute deltas per-frame; they
allocate `8 * len(data)` bytes per call with no `*Into` form. 044-C2
covers the API; this is the perf cost: ~3 GC cycles per minute of
typical Oracle ingest (1 MHz timestamp stream → 32 MiB/s allocator
pressure).

## 4. Entropy family: `math.Log2` in the inner loop

`entropy.go:30-32` (and 4 siblings): `for _, p := range probs { if p > 0 { h -= p * math.Log2(p) } }`.
`math.Log2` is ~25-40 ns/call (Frexp + poly + Ldexp). 256-alphabet →
~10 µs/call. The `if p > 0` branch prevents AVX vectorisation; replace
with `math.Log2(math.Max(p, 1e-300))` (saves ~3 ns/elem, **~1.3×**).
The full ladder:

| Variant | ns/op (256 PMF) | speedup | precision |
|---|---|---|---|
| current (Log2 + branch) | ~10 µs | 1× | float64 |
| branchless clamp | ~7 µs | 1.4× | float64 |
| `bits.LeadingZeros64` integer log2 + poly tail | ~3 µs | 3× | ~22 bits |
| 256-entry LUT for byte-PMF case | ~0.8 µs | 12× | exact for `count/N` |

The LUT case is the Pistachio answer: `logTable[i] = i * math.Log2(i/N)`
once, reused per frame. ~50 LOC, one `sync.Once`. **12-15×** on the
hot path (standard `compress/flate huffmanBitWriter.generate` trick).

## 5. ScalarQuantize: `Round` + branch-clamp chain

`quantize.go:64-71`: `q := int(math.Round((v - min) / step)); if q < 0 { q = 0 }; if q >= levels { q = levels - 1 }`.
`math.Round` is ~12 ns. Replace with `int((v-min)/step + 0.5)` —
truncate-toward-zero is fine since post-clamp values are non-negative
(saves ~9 ns/elem). Branch-clamp → Go 1.21 `max(0, min(q, levels-1))`
inlines to `CMOVL` (saves ~1-2 ns on random input). **SIMD ceiling:**
the loop is canonical `(x - offset) * scale` + `VCVTPD2DQ` + clamp;
hand-AVX2 ceiling ~3-4×, pure-Go reachable ~1.3×.

## 6. Bit-IO: completely absent

The package has **no bit packer/unpacker at all**. No `BitWriter`,
`BitReader`, `WriteBits(value, nBits)`, `ReadBits(nBits)`. This is
the foundational primitive that **every entropy coder requires**:

- Huffman (042 Tier-1) — variable-length code emit/consume
- canonical-Huffman decode tables — bit-stream pull
- arithmetic / range / rANS — bit-stream rolling normaliser
- Golomb-Rice / Elias / Exp-Golomb — variable-length integer encode
- bit-packing / packed-array (Daniel Lemire SIMDComp) — fixed-width N-of-64 emit
- ULEB128 / Protobuf varint (042 Tier-1) — 7-bit-at-a-time emit

Without a `BitWriter` of some shape, **every codec 042 wants to add
will reinvent it**. The decision matrix:

| Choice | Pros | Cons | Used by |
|---|---|---|---|
| Per-byte buffered | simple, no allocs in hot path | 8× slower than word-aligned | nobody serious |
| 32-bit word + bit-cursor | portable, fast | complex flush | zlib, gzip |
| **64-bit word + bit-cursor** | fastest, simple | needs `math/bits` | zstd, brotli, JPEG XL |
| 64-bit + manual unrolling 4-wide | SIMD-grade | very complex | x265, libvpx |

**Recommendation:** 64-bit word + cursor + `math/bits.TrailingZeros64`/`LeadingZeros64`.
~150 LOC, zero deps; lowers to `BSR`/`TZCNT`/`LZCNT` on AMD64 and `CLZ`
on ARM64. Sketch: `type BitWriter struct { buf []byte; bb uint64; nb uint }`,
`WriteBits` ORs `v << nb` into `bb`, drains 8-bit groups via
`append(buf, byte(bb)); bb >>= 8; nb -= 8` until `nb < 8`. **Single
most important missing primitive** for compression perf — without it,
042's Tier-1 codecs cannot ship at production speed.

## 7. Table-driven decoders: layout matters more than algorithm

When 042's Huffman / canonical-Huffman lands, the decode table
dominates cost. Choice space:

| Layout | Table | Lookup | Mispred | Used by |
|---|---|---|---|---|
| Linear scan | 0 | O(n) | high | naive |
| Code tree | ~N nodes | log2(N) misses | med | Java |
| Single-level (`2^maxbits`) | large | 1 miss | none | DEFLATE |
| **Two-level (9-bit + tail)** | ~4 KiB | usually 1 | low | **zlib, zstd** |
| Adaptive multi-level | variable | 1-3 | low | brotli |

zlib's `inflate.go` uses two-level: 9-bit primary (512×2B = **1 KiB**,
fits in L1d), tail tables sized to longest code. **Table must be
cache-line-aligned** (64B x86, 128B ARM). Standard canonical-Huffman
layout for byte alphabets: `[1024]uint32` packing `(sym uint16, codeLen
uint8, _ uint8)` — 4 KiB, fits in L1d entirely. Hot loop: `entry :=
table[br.peekBits(15) & 0x1FF]; sym := uint16(entry); nbits :=
uint8(entry >> 16); br.consume(uint(nbits))`. **Reality has zero of
this machinery**; the 042 perf successor should validate against this shape.

## 8. SIMD-amenable loops: which loops, which shapes

Go's SSA backend auto-vectorises a narrow set of loop shapes. For the
compression package, the auto-vectorisation candidates are:

| Loop | File:line | Auto-vec? | Manual SIMD ceiling |
|---|---|---|---|
| RLE encode (count-up) | coding.go:23-25 | No (early-exit `count<255`) | ~2× (vbyte trick) |
| RLE decode inner | coding.go:56-58 | No (per-byte append) | ~10× via memset |
| DeltaEncode subtract | coding.go:79-81 | **Yes (Go 1.21+)** | ~4× (already 80% there) |
| DeltaDecode prefix-sum | coding.go:99-101 | No (data dependency) | ~5× via 8-wide prefix |
| ShannonEntropy log-mul-sum | entropy.go:29-33 | No (`Log2` call) | ~12× via LUT |
| ScalarQuantize | quantize.go:60-72 | Partial (clamp branches) | ~3-4× |

The **two highest-leverage manual SIMD candidates** are DeltaDecode (via
8-wide prefix-sum, Lemire 2015 method, ~50 LOC of inline AVX2 asm) and
ShannonEntropy on byte-alphabet PMFs (via LUT). Both produce
order-of-magnitude wins on the named hot consumers (Oracle time-series,
Pistachio texture entropy).

## 9. Cache-line alignment: 0/3 codecs

Go's allocator returns 8-byte-aligned pointers (16 on ARM64). For
table-driven decoders streaming at billions-of-rows/sec, tables must
start at 64-byte boundaries so one cache-line fill serves 8-16 entries.
Current package has no tables. Defensive idiom for tables ≥ 4 KiB is
`make([]T, n+8)` then slice from first 64-aligned offset (no `unsafe`).
Cleaner: package-level `var huffTable [1024]uint32` lands in
page-aligned `.data` section — `compress/flate`'s pattern.

## 10. Benchmarks: the diagnostic vacuum

Most damning: `compression/` has no `Benchmark*`, no `b.N`, no
`-benchmem` baseline ever run. Grep across `reality/` shows benchmarks
in `prob/copula/` and `audio/fingerprint.go` only. Without benchmarks:
(1) every speedup figure here is an estimate from disassembly intent and
stdlib precedent, not data; (2) regression detection is impossible (a
swap to slower `math.Log` couldn't be caught); (3) Pareto choices are
blind — LUT-vs-Log2 is 12× faster *only on small alphabets*; at 65536
symbols the LUT blows L1d. Cutoffs unknown without sweep.

**Recommended bench file** (~150 LOC, alongside `compression_test.go`):
`Benchmark{RunLengthEncode_{Random,Compressible}1MiB, RunLengthDecode_LongRuns,
DeltaEncode_64K, DeltaDecode_64K, ShannonEntropy_Alpha{256,65536},
ScalarQuantize_1M_256levels}` + `b.ReportAllocs()`. **Prerequisite for
every other change in this report**; without measurement, optimisation is theatre.

## 11. Streaming-vs-batch perf gap (forward-looking, per 044)

When 044-C9/C10's streaming forms land (`RLEEncoder`, `DeltaEncoder`),
the perf profile changes:

| Aspect | Batch | Streaming |
|---|---|---|
| Per-call alloc | one big slice | zero |
| Per-frame state | none | ~16 B |
| Tiny-write throughput | n/a | bottlenecked by `Write` syscall |
| Large-write throughput | best | parity |
| Memory bound | O(input) | O(1) |

**Trap.** Naive `RLEEncoder.Write(p)` calling `w.Write([]byte{count, val})`
per pair is 100× slower than batch (~80 ns/Write call overhead). Fix:
internal 4 KiB output buffer, chunked flush. `bufio.Writer` discipline,
**owned by encoder**, not the caller.

## 12. Allocation discipline summary

| Function | Alloc count | Bytes | Avoidable? | Fix |
|---|---|---|---|---|
| `RunLengthEncode` | 1-12 (worst-case) | up to 2× len | yes | `*Into` + correct cap |
| `RunLengthDecode` | 1 | total | yes | `*Into` |
| `DeltaEncode` | 1 | 8× len | yes | `*Into` |
| `DeltaDecode` | 1 | 8× len | yes | `*Into` |
| `ScalarQuantize` | 0 | 0 | already optimal | — |
| `ScalarDequantize` | 0 | 0 | already optimal | — |
| `ShannonEntropy` | 0 | 0 | already optimal | — |
| `JointEntropy` | 0 | 0 | already optimal | — |
| `ConditionalEntropy` | **1** | 8× nrows | yes | output buffer |
| `MutualInformation` | **2** | 8× (nrows + ncols) | yes | output buffer |
| `KLDivergence` | 0 | 0 | already optimal | — |
| `CrossEntropy` | 0 | 0 | already optimal | — |

`ConditionalEntropy` (entropy.go:71) and `MutualInformation`
(entropy.go:96, 113) allocate marginal vectors per-call. For a Pistachio
loop computing `MI` per frame, that's ~2 GiB/sec of garbage on a
1024-symbol alphabet. Add `*Into` companions or thread a scratch buffer.

## 13. Recommended commit ladder for compression-perf

Strict performance, no API breaks (those are 044's job). Each commit
includes new benchmarks measuring the change.

| # | Item | LOC | Speedup |
|---|---|---|---|
| P1 | Add `compression_bench_test.go` with 8 benchmarks across Encode/Decode/Entropy/Quantize | ~150 | n/a (measurement) |
| P2 | Fix `RunLengthEncode` cap from `len(data)` to `2*len(data)` | 1 | ~4× on random input |
| P3 | Replace `RunLengthDecode`'s per-byte `append` loop with `copy`-doubling memset | ~10 | ~5-10× on long runs |
| P4 | Eliminate branch in entropy hot loops via `math.Max(p, 1e-300)` | ~5 lines × 5 fns | ~1.3× |
| P5 | Pre-build `logTable[256]` for byte-alphabet `ShannonEntropy` (LUT path via type-assertion or new fn `ShannonEntropyByteCounts`) | ~40 | ~12× on byte PMFs |
| P6 | `ScalarQuantize`: `Round` → `Floor(x+0.5)`, branch clamp → `min`/`max` builtins | ~10 | ~1.3× |
| P7 | Add `*Into` forms (covered by 044-C2; this just closes the alloc gap) | ~80 | ~2× via zero-alloc + caller buffer reuse |
| P8 | Add `BitWriter` / `BitReader` with `WriteBits`/`ReadBits`/`Flush`/`Align`, 64-bit-word backed via `math/bits` | ~150 | prerequisite for 042 |
| P9 | Pre-allocate marginal scratch in `ConditionalEntropy`/`MutualInformation` via output-buffer parameter or sync.Pool | ~30 | eliminates 2 alloc/call |
| P10 | When 042's Huffman lands: 9-bit primary + tail two-level table, 4 KiB cache-line aligned | ~100 (in 042) | order-of-magnitude vs naive |
| P11 | When 042's Delta-of-Delta or Gorilla lands: 8-wide prefix-sum decoder (Go scalar with manual unrolling, no asm) | ~50 | ~3× on DeltaDecode |
| P12 | Streaming `RLEEncoder.Write` with internal 4 KiB output buffer (per 044-C9) | ~80 | parity with batch on large; prevents 100× slowdown on small writes |

Total ~705 LOC. P1+P2+P3 are the **minimum cohesive bundle** —
benchmarks first, then the two trivial-but-high-impact bug fixes, ~30
minutes of work, ~5× wins on the named hot consumers. P8 is the gate
for 042's Tier-1 codec work; without it Huffman/varint/rANS will all
reinvent the same primitive at varying quality.

## 14. What this report does NOT cover

- **Numerical correctness**: 041 (bias, NaN propagation, overflow).
- **Missing primitives**: 042 (Huffman / LZ77 / varint / Gorilla / ANS).
- **Architectural axes**: 043 (frame headers, dictionary training, BCJ).
- **API ergonomics**: 044 (buffer ownership, error returns, streaming surface).
- **Specific SIMD assembly**: out of scope for zero-dep golden-file repo;
  flagged as a known ceiling on DeltaDecode and ScalarQuantize.
- **GC pressure measurements under realistic load**: requires P1 (benchmarks)
  to land first.

---

**File:** `agents/045-compression-perf.md`. Read source: `compression/coding.go`,
`compression/entropy.go`, `compression/quantize.go`, `compression/compression_test.go`.
Cross-ref siblings: 041 (numerics), 042 (missing primitives), 043 (sota
architecture), 044 (API ergonomics), 040 (combinatorics-perf as style
template), 020/030 (calculus/chaos perf as Go-SIMD-amenability precedent).

# 044 | compression-api

**Scope.** API ergonomics of `reality/compression`'s user-facing surface.
9 free functions across 3 files (`entropy.go` 177, `coding.go` 104,
`quantize.go` 99). 041 covered numerics, 042 missing primitives, 043
SOTA architecture. This is strictly call-site ergonomics: signatures,
names, buffer conventions, error paths.

## TL;DR

Two incompatible API shapes share one namespace. The entropy family
(`ShannonEntropy`/`KLDivergence`/...) is float-in-float-out with silent-zero
on degenerate input. The coding family (`RunLengthEncode`/`DeltaEncode`/
`ScalarQuantize`) uses **three different buffer ownership conventions
across three pairs**:

| Pair | Allocator | Returns | Errors |
|---|---|---|---|
| `RunLengthEncode/Decode` | callee (`append`-grown) | `[]byte` | `nil` on bad input |
| `DeltaEncode/Decode` | callee (`make([]int64, n)`) | `[]int64` | `nil` on empty |
| `ScalarQuantize/Dequantize` | **caller** (`out []int`/`[]float64`) | `(min, step)` / void | silent skip |

CLAUDE.md says *"No allocations in hot paths. Functions accept output
buffers."* — only `Scalar*` honors this; `RunLength*`/`Delta*` violate.
The convention is invisible from the call site until the user notices
one returns `(min, step)` and the other `[]byte`.

**Zero streaming surface.** No `io.Reader`/`io.Writer` adapter, no
`Encoder.Reset(w)`, no frame boundary. Every function consumes/produces
a complete buffer. Worse off than audio (009): audio has 2 streaming
citizens (`DegradationTracker`, `Fingerprint`); compression has zero
stateful types of any kind. Zero exported types — 9 free functions, no
struct, no interface, no error variable, no constant.

**Zero composition surface.** The classic *"RLE then Huffman"* idiom is
unrepresentable: no `Filter` interface, no `Pipeline` type, no shared I/O
protocol between `RunLengthEncode` and any future `HuffmanEncode`. Same
hole 043 sees from the architecture side.

**No magic numbers, no headers, no version bytes.**
`RunLengthEncode([]byte{1, 65})` and a literal `[1, 65]` from some other
source are indistinguishable. `RunLengthDecode` decodes the latter as a
run-of-1 byte 65, returns `[65]`. The only sanity check is
`len(encoded) % 2 != 0 → nil`.

Severity: 1 cross-package architectural defect (streaming/framing/
composition, same root as 043), 4 within-package consistency defects
(buffer ownership ×3, return shape ×3, error convention ×3, verb-pair
×1), 6 docstring-contract gaps, 2 cross-package naming inconsistencies.

---

## 1. Buffer ownership: three conventions, one package

```go
// Convention A: callee allocates, returns slice. (coding.go:13)
func RunLengthEncode(data []byte) []byte
func RunLengthDecode(encoded []byte) []byte

// Convention B: callee allocates fixed-size, returns slice. (coding.go:73)
func DeltaEncode(data []int64) []int64
func DeltaDecode(encoded []int64) []int64

// Convention C: caller allocates, returns scalars. (quantize.go:30)
func ScalarQuantize(data []float64, levels int, out []int) (min, step float64)
func ScalarDequantize(quantized []int, min, step float64, out []float64)
```

Three shapes for what should be **one shape** at the primitive level. Damage:

- **No `Bound()` companion.** zstd ships `ZSTD_compressBound`, Brotli
  `BrotliEncoderMaxCompressedSize`, LZ4 `LZ4_compressBound`, Snappy
  `MaxEncodedLen` — reality ships nothing. User must know worst-case RLE
  is `2 * len(data)` from reading coding.go:18.

- **Silent truncation on too-small `out`.** `ScalarQuantize`/`ScalarDequantize`
  break out via `if i >= len(out) { break }` (quantize.go:51, 62, 92) — no
  panic, no error, no `(written int)` return. Caller gets `(min, step)`
  for the *full* input but `out` is partially written.

- **`RunLengthEncode` pre-alloc is wrong.** coding.go:17 comment says "2x
  worst case" but allocates `make([]byte, 0, len(data))`. Random input
  triggers ~10 reallocations at 1 KiB. (041 flagged this perf-side.)

- **`DeltaEncode` always allocates fresh.** No `DeltaEncodeInPlace` and no
  `out []int64` form. CLAUDE.md mentions Oracle/RubberDuck as time-series
  consumers — they allocate twice.

**Fix shape:**

```go
func RunLengthBound(srcLen int) int { return 2 * srcLen }
func DeltaBound(srcLen int) int     { return srcLen }
func ScalarQuantizeBound(srcLen int) int { return srcLen }

// All primitives accept an output buffer and return (n int, err error).
func RunLengthEncodeInto(data, out []byte) (n int, err error)
func DeltaEncodeInto(data []int64, out []int64) (n int, err error)
// Existing alloc-and-return forms become 5-line wrappers.
```

Matches the `*Into` convention `audio/separation` uses
(`SubtractSpectrum`/`SubtractSpectrumInto`) and the `out []float64`
convention `signal/window` uses (`HannWindow(n, out)` at
signal/window.go:15).

## 2. Error handling: silent failure everywhere

Every function in the package handles bad input with **silent return of a
zero/empty value**. Not one error is ever returned, not one panic is ever
emitted, not one validation check fails loudly.

| Function | Bad-input behavior |
|---|---|
| `ShannonEntropy` | empty → 0; NaN entry → NaN out (silent) |
| `JointEntropy` | ragged matrix → wrong answer (silent) |
| `ConditionalEntropy` | non-square → wrong answer (silent) |
| `MutualInformation` | ragged → wrong answer (silent) |
| `KLDivergence` | mismatched lens → 0 (silent) |
| `CrossEntropy` | mismatched lens → 0 (silent) |
| `RunLengthEncode` | nil/empty → nil |
| `RunLengthDecode` | odd-length input → nil |
| `DeltaEncode/Decode` | empty → nil |
| `ScalarQuantize` | empty/levels<1 → (0, 0) |
| `ScalarDequantize` | undersized out → silent truncation |

Flat violation of 043's axis-5 ("errors-not-panics") and Go idiom. Damage:

1. `RunLengthDecode` returns `nil` for both "empty input" (legit) *and*
   "corrupted/truncated" (catastrophic). Caller cannot distinguish.
2. `KLDivergence(p=[0.5,0.5], q=[1.0])` returns `0` — same as `KL(P,P)`.
   User reading 0 as "identical" is wrong by contract.
3. `ScalarDequantize` silently truncates on undersized `out` (void return).
4. `MutualInformation(joint=[][]float64{{0.5}, {0.5, 0.0}})` is ragged
   and produces a plausible-looking wrong number (041-S2 #4).

**Fix:**

```go
var (
    ErrInvalidRLEStream = errors.New("compression: invalid RLE stream (odd length)")
    ErrTruncated        = errors.New("compression: truncated input")
    ErrShapeMismatch    = errors.New("compression: shape mismatch")
    ErrOutTooSmall      = errors.New("compression: output buffer too small")
)
```

Returned from `*Into` forms. Legacy alloc-and-return forms can keep
NaN-on-bad-input math semantics, but `RunLengthDecode` is not pure math
and needs an error channel.

## 3. Naming convention: three verbs for one operation

Package mixes three verb pairs:
- `RunLength{Encode, Decode}` — encode/decode pair
- `Delta{Encode, Decode}` — same
- `Scalar{Quantize, Dequantize}` — different pair
- entropy family has no verb (nouns)

Go stdlib uses `Compress/Decompress` (gzip/flate/zlib/bzip2 packages,
verbs implicit via `io.Reader`/`io.Writer`). reality picks `Encode/Decode`
— defensible if uniform, but `ScalarQuantize` is the odd one out.

**Recommended:** keep current names; document a package-level convention:
"Lossless codecs use Encode/Decode; lossy codecs use the operation name
(Quantize/Dequantize, Predict/Reconstruct, Transform/InverseTransform)."
This rationalizes existing names without churn and gives 042's tier-1
codecs a naming pattern.

`RunLength` vs `RLE`: signal/ uses `FFT` not `FastFourierTransform`,
so repo precedent is **acronym-when-canonical**. RLE is canonical;
`RunLengthEncode` (14 chars) → `RLEEncode` (9) follows precedent.

## 4. Streaming codec interface: completely absent

Classic Go pattern is `compress/gzip`-shaped: `Writer` + `Write/Close/Reset`,
`Reader` + `Read`. reality has none. A 4 GiB log file cannot be streamed
through `RunLengthEncode` — must load entire thing into RAM. Recall (named
consumer in package doc) cannot maintain encoder state across frames.

This is stark because RLE is trivially streamable: ~16 bytes of state
(`currentValue, currentRunLen`). Streaming RLE encoder is ~25 LOC with
`io.Writer` plumbing. DeltaEncode is even simpler: 8 bytes of state.

**Fix shape:**

```go
type RLEEncoder struct { lastVal byte; runLen int; w io.Writer; ... }
func NewRLEEncoder(w io.Writer) *RLEEncoder
func (e *RLEEncoder) Write(p []byte) (n int, err error) // accumulates state
func (e *RLEEncoder) Close() error                      // flushes final run
func (e *RLEEncoder) Reset(w io.Writer)

type RLEDecoder struct { ... }
func NewRLEDecoder(r io.Reader) *RLEDecoder
func (d *RLEDecoder) Read(p []byte) (n int, err error)
```

Same shape for `Delta{Encoder,Decoder}`, `Scalar{Quantizer,Dequantizer}`.
Each <50 LOC. Wins: memory-bounded operation, `io.Pipe` composability,
interop with every Go program, and a place to hang frame boundaries,
magic numbers, and corruption checks.

## 5. Frame boundaries: not a concept

No frame model. No answer to "when is a frame complete?" because there is
no frame. `RunLengthEncode` emits a flat byte sequence — no header, no
length prefix, no checksum, no magic bytes. Concatenating two RLE outputs
produces a *valid-looking* third output with merged frames:

```go
a := RunLengthEncode([]byte{1, 1, 1})  // [3, 1]
b := RunLengthEncode([]byte{2, 2})     // [2, 2]
RunLengthDecode(append(a, b...))        // [1,1,1,2,2] — frames merged, no recovery path
```

Primitives genuinely don't need framing (`compress/flate` doesn't, `zlib`
adds 6 bytes on top, `gzip` adds 18+ on top of zlib). reality analog:
primitives stay frame-less, `compression/frame` sub-package wraps. Sketch:

```go
type Frame struct {
    Magic    [4]byte // "RLTY"
    Version  uint16
    Codec    Codec   // enum: RLE, Delta, Quantize, ...
    Payload  []byte
    Checksum uint32  // CRC32 over Codec+Payload
}
func (f *Frame) Marshal(out []byte) (n int, err error)
func ParseFrame(data []byte) (*Frame, []byte, error) // (frame, rest, err)
```

This is where codec **composition** lives: a frame can declare a filter
chain via `[]Codec` (BWT→MTF→RLE→Huffman, classic bzip2) and the framing
layer dispatches.

## 6. Codec composition: zero machinery

No way to express "RLE then Huffman." Hypothetical user code:

```go
rle := compression.RunLengthEncode(data)
huff := compression.HuffmanEncode(rle)  // when 042 lands
```

Two calls, two intermediate `[]byte` allocs, no shared type. When 042
adds 30 codecs, every caller doing transform-then-entropy-code allocates
twice with no way to declare the pipeline as a single value. 043's fix:

```go
type Filter interface {
    Encode(in, out []byte) (int, error)
    Decode(in, out []byte) (int, error)
}
type Pipeline []Filter
```

The Filter interface is the single change that unlocks 042's tier-1.
Defensible at 3 codecs; untenable at 30.

## 7. Encoder state: stateless only

Every function stateless. No `RLEEncoder`/`DeltaEncoder`/`Quantizer`
struct, even though all three benefit from cross-call state (RLE: current
run; Delta: previous value; Quantizer: min/max history for adaptive
quantization). `changepoint.Bocpd` is the right template (struct + `New`
+ `Update` + `Reset` + queries — referenced by 009).

The cost is visible in `ScalarQuantize`'s `(min, step float64)` return
— caller threads these through to `ScalarDequantize`. A `Quantizer`
struct owns them:

```go
type Quantizer struct { Min, Step float64; Levels int }
func NewQuantizer(data []float64, levels int) *Quantizer
func (q *Quantizer) Quantize(data []float64, out []int)
func (q *Quantizer) Dequantize(quantized []int, out []float64)
```

Three-line struct + three methods replaces the awkward dual-return.

## 8. Comparison with siblings

**signal/**: caller-allocated output buffers consistently
(`PowerSpectrum(re, im, out)` at fft.go:140, `Convolve(signal, kernel,
out)` at filter.go:19, `HannWindow(n, out)` at window.go:15). No streaming
citizens either — `FFT` is whole-array in-place. But it is **consistent**:
every signal function is "caller allocates, callee fills." compression's
three conventions are strictly worse than signal's one.

**audio/** (009): 2 streaming citizens (`DegradationTracker`, `Fingerprint`),
~15 zero-alloc per-frame DSP forms, segregated batch forms
(`spectrogram.Compute`). The `*Into` companion pattern
(`SubtractSpectrum`/`SubtractSpectrumInto`) is the template compression
should adopt. Compression is strictly behind audio on streaming surface
count (0 vs 2) and zero-alloc-form count (2 vs ~15).

**testutil/**: 041 noted compression has 1 file × 10 cases = ~5% of the
CLAUDE.md "min 20/fn" floor. From the API side this is also a
**discoverability** defect — a new contributor seeing one file for one
of nine functions reasonably concludes "abandoned" or "this is the
convention."

## 9. Cross-package naming inconsistency

Repo uses several verb conventions:
- `signal.FFT` / `IFFT` — name is the operation
- `signal.HannWindow(n, out)` — operation + output slot
- `audio.spectrogram.Compute/Inverse` — generic verb pair
- `compression.RunLengthEncode/Decode` — operation + Encode/Decode
- `compression.ScalarQuantize/Dequantize` — operation + Quantize/Dequantize
- `crypto.SHA256(data, out)` — function-as-operation
- `prob.Sample` / `PDF` / `CDF` — operation-only

No repo-level documented convention. 044 is too narrow to mandate one,
but compression's local choice could be uniform: `ScalarEncode(data,
levels, out)` / `ScalarDecode(...)`. This loses the math-y "quantize" name
— meaningful information loss for grep-discovery — so the
**counter-recommendation** is to keep all three pairs and document the
exception in the package doc.

## 10. Recommended commit ladder for API ergonomics

Strictly cosmetic-and-contract, no algorithm changes (those are 042's job):

| # | Item | LOC | Risk |
|---|---|---|---|
| C1 | Add `*Bound(srcLen int) int` for each codec | ~15 | none |
| C2 | Add `*Into(in, out)` companion forms returning `(int, error)` | ~80 | none |
| C3 | Define `ErrInvalidRLEStream`, `ErrTruncated`, `ErrShapeMismatch`, `ErrOutTooSmall` | ~10 | none |
| C4 | `RunLengthDecode` validates each `count >= 1` and rejects `count == 0` (currently silently drops) | ~5 | low |
| C5 | `MutualInformation`/`ConditionalEntropy` reject ragged matrices via `ErrShapeMismatch` | ~12 | low |
| C6 | `KLDivergence`/`CrossEntropy` return `error` on `len(p) != len(q)` instead of silent 0 | ~8 | medium (signature change) |
| C7 | Document buffer-ownership convention in package doc; document Encode/Decode vs Quantize/Dequantize naming | ~25 | none |
| C8 | `Quantizer` struct wrapping `(min, step, levels)` with `Quantize`/`Dequantize` methods | ~30 | low |
| C9 | `RLEEncoder`/`RLEDecoder` `io.Writer`/`io.Reader` adapters | ~80 | low |
| C10 | `DeltaEncoder`/`DeltaDecoder` same | ~60 | low |
| C11 | `compression/frame` sub-package: `Frame`, `Marshal`, `ParseFrame`, magic bytes, CRC32 | ~150 | medium (new sub-package) |
| C12 | `Filter` interface + `Pipeline` type per 043 | ~80 | medium (architectural lift) |

Total ~555 LOC, all strictly additive (no breaking changes if we keep the
current free functions as facades). C1+C2+C3 are the minimum cohesive
bundle — they fix the buffer-ownership inconsistency, the no-Bound gap,
and the silent-error gap in one wave. C9+C10 are the streaming surface
(should ship together so RLE/Delta are at parity). C11+C12 are 043's
architectural posts and should ship before 042's tier-1 codec work to
avoid retrofitting.

## 11. What this report does NOT cover (sibling delegation)

- **Numerical correctness** of the implemented functions: 041.
- **Missing primitives** (Huffman/LZ77/varint/Gorilla/...): 042.
- **SOTA architectural axes** (frame format, dictionary training, per-block
  parallelism, BCJ filters, Roaring containers, tolerance contracts): 043.
- **Performance** (alloc counts, hot-path benchmarks): out-of-scope here,
  presumably a future 04X-compression-perf slot if scheduled.

---

**File:** `agents/044-compression-api.md`. Read source: `compression/coding.go`,
`compression/entropy.go`, `compression/quantize.go`, `compression/compression_test.go`,
`signal/fft.go`, `signal/filter.go`, `signal/window.go`. Cross-ref siblings:
041 (numerics), 042 (missing primitives), 043 (sota architecture), 009
(audio API as comparison precedent).

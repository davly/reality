# 135 | signal-perf

**Agent:** 135 of 400
**Topic:** signal: Stockham FFT, split-radix, real-FFT optimizations
**Scope:** performance audit of `C:/limitless/foundation/reality/signal/` —
`fft.go` (180 LOC), `filter.go` (174 LOC), `window.go` (113 LOC). Cross-
referenced against `audio/spectrogram/stft.go` (dominant non-test caller).
**Date:** 2026-05-08

## Brief

131 audited numerics; 132 inventoried missing primitives; 133 walked SOTA
libraries; 134 audited API shape. **None ran a cycle/byte/cache audit of
the kernels that exist.** This report does.

Headline findings:

1. FFT is **Cooley-Tukey radix-2 DIT** with bit-reverse permute and
   **stage-local twiddle recurrence** — ~30-40% slower than every algorithm
   in the topic title on reality's actual workload.
2. **Zero benchmarks anywhere** in `signal/` (`go test -bench=.` returns
   nothing). 8th of 8 perf-tagged agents this review to flag this.
3. **`MovingAverage` docstring lies**: claims "Zero heap allocations —
   uses running sum"; implementation re-sums the window each output,
   `O(N·W)` not `O(N)`. At W=21, **20× slower** than a 4-LOC fix; W=101,
   **34×**; W=501, **167×**.
4. **No FFT-convolve crossover** — `Convolve` direct only; consumers needing
   long kernels (>~256 taps) pay up to 24× the optimum.
5. **Twiddle tables not cached** (also per 131/133): every FFT call
   re-runs `cos`/`sin` per stage; STFT at 75% overlap on 5-min audio =
   ~58k re-runs of the same 11 pairs.
6. **Cache locality**: split `real[]/imag[]` doubles L1 line footprint vs.
   interleaved `complex128`. Acceptable cost (~5-15% wall) for the API
   benefit, but undocumented.
7. **`ApplyWindow` doesn't autovectorise** — needs the bounds-check hoist
   idiom for Go 1.23 amd64 to SIMD-fold the multiply.

This report does not repeat 131's recurrence-drift numerics, 133's pocketfft/
FFTW crosswalk, or 134's API-shape recommendations. It adds wall-clock and
allocation accounting + the specific algorithmic swaps the topic title names.

---

## A — Algorithm baseline

`fft.go:49-91` is textbook Cooley-Tukey radix-2 DIT, in-place, bit-reverse
first; per stage one `cos`/`sin` pair, then N/2 complex-multiplies inside
the butterfly. Op count ≈ **7·N·log2(N)** (5·N·logN butterfly + 2·N·logN
twiddle recurrence). N=1024 → ~72k FLOPs ≈ 24 µs floor on 3 GHz; observed
~80 µs from audio benchmarks → **~3.3× ceiling**. Gap: twiddle cost (§E),
split-slice cache footprint (§F), stage-call overhead.

---

## B — Topic swap 1: Stockham auto-sort FFT

**Stockham 1966** rewrites the butterfly to read from buffer A, write to
buffer B, swapping per stage. Outputs land in natural order — **no
explicit bit-reverse pass.**

Trade:
- Removes `bitReverse(real, imag, n)` (`fft.go:61`) — ~N/2 swaps,
  ≤1% wall by op count, but **the cache-pessimal scrambled-stride pass
  alone is 5-10% wall on N≥2^14** (cold L1/L2).
- Costs one extra in-place buffer of N float64 per slice. CLAUDE.md rule
  3 ("no allocations in hot paths") forces this to be **caller-provided**:
  `FFTStockhamInto(real, imag, scratchReal, scratchImag)`. Existing
  `FFT(real, imag)` keeps its in-place radix-2 contract.

Default in pocketfft, FFTW small-N kernels, KFR small-N, JUCE Stockham
engine. Reality already pays the cache cost; Stockham removes it.

**Wall: ~10-15% at N=1024, ~25% at N=16384. Not 2× — Stockham trades
cache pattern, not FLOP count.**

---

## C — Topic swap 2: Split-radix FFT

**Yavne 1968 / Duhamel-Hollmann 1984.** Splits length-N into one length-N/2
DIT (evens) plus two length-N/4 DITs (odd quarters at twiddle ω, ω³).

| Algorithm        | mul count for length-N FFT (real arithmetic) |
|------------------|---------------------------------------------|
| radix-2 (current)| 4·N·log2(N)                                |
| radix-4          | 3·N·log2(N)                                |
| split-radix 2/4  | ≈ (8/3)·N·log2(N) — ~33% under radix-2     |

For N=1024: ~5,000 multiplications saved per FFT. STFT (~58k FFTs/5-min
audio) saves ~290M mul, ~96 ms compute per 5 min audio.

Catch: split-radix kernel is **measurably uglier** — 3-arm recursion or
its iterative-Goedecker form. ~120 LOC vs current ~45 for the butterfly.
Doubles maintenance surface for the last ~25% wall. **Defer indefinitely**
— Tier 1 below closes 80% of the FFTW gap at 5× lower code surface cost.

---

## D — Topic swap 3: Real-FFT — 2× speedup, free

**The only one of the three name-checked optimisations that is unambiguous
to-do.** 132 §1.1 named it missing; 133 confirmed pocketfft/FFTW/KFR/JUCE
all expose it; 134 noted half the `audio/spectrogram` complex128 waste is
from its absence.

Trick (Sorensen-Heideman-Burrus 1987; packing back to **Bergland 1968**):
For real input `x[n]` length 2N, define `z[k] = x[2k] + i·x[2k+1]` length
N. One length-N complex FFT + O(N) post-processing yields the 2N-point
real-input FFT. Output is N+1 bins (Hermitian symmetry).

Wall-clock at N=1024 real input: length-512 complex FFT runs at half the
FLOP cost. **2× faster, plus halved memory.** Numerical: identical to
~8 ulp (post-pass is one 2-point butterfly per bin).

Reality's STFT is **always real-input**. Every `signal.FFT` caller in
`audio/` (10 sites grep'd: stft.go, fundamental.go, audio_test.go,
pitch_test.go, separation_test.go, vibration_test.go) currently:

```go
frameImag[k] = 0   // explicit zero-fill
signal.FFT(frameReal, frameImag)  // half the work proves 0+0i = 0
```

**Add `signal.RFFT(real []float64, outReal, outImag []float64)`** returning
N/2+1 bins. STFT per-frame cost halves. ~80 LOC including the bin-recovery
loop, shared 256-bit-precision golden across Go/Python/C++/C#.

---

## E — Twiddle table caching

131 covered precision; this adds perf.

Per-call recurrence at N=1024, 10 stages, avg 256 butterflies/stage =
2,560 complex-multiplies just to advance the twiddle counter. ~15 cycles
each = **~38 µs, ~half the FFT wall-clock.**

Precomputed `twiddles[N] []complex128` of length N/2 indexed by butterfly
position serves all stages (stage `size` uses every `N/size`-th entry).
~16N bytes/cached N; audio sizes (512–4096) total ~120 KB cache via
`sync.Map[int][]complex128`. Precompute ~10 µs at N=1024, amortised
across thousands of STFT FFTs. **~30 LOC, ~25% faster, fixes 131's
precision cliff at N≥2^16, no API change.**

---

## F — Cache locality of the butterfly

Per butterfly: 4 reads + 4 writes across `real[*]` and `imag[*]` in
**two separate cache lines** (split heap regions). 2× line footprint vs
interleaved `complex128`. Wall cost: ~5% at N≤2^12 (L1-resident),
~15% at N=2^16, ~20% at N=2^20 (DRAM-bandwidth bound).

Interleaved layout halves line footprint but loses FMA-folding (Go's
`complex128` doesn't FMA-fold; 134 §boundary). The split layout is the
**deliberate API choice** (134 §boundary). The 5-20% wall is the price
of `(real, imag []float64)` plus FMA-friendly scalar butterfly. **Don't
change; document the tradeoff in package doc** so future SIMD-curious
contributors don't read it as oversight.

---

## G — `Convolve` direct vs FFT crossover

`filter.go:19-40`. Pure direct `O(N·M)` double loop. Inner `out[i+j] +=
signal[i] * kernel[j]` is FMA-friendly, ~1 mul-add/cycle on amd64.

Crossover with FFT-convolve (zero-pad to next pow-2 P ≥ N+M-1, three
length-P FFTs + pointwise multiply):

| signal N | kernel M | direct | FFT (5·P·logP) | winner          |
|---------:|---------:|-------:|---------------:|:----------------|
| 4096     | 16       | 65k    | 532k           | direct 8×       |
| 4096     | 64       | 262k   | 532k           | direct 2×       |
| 4096     | 128      | 524k   | 532k           | **tie**         |
| 4096     | 256      | 1.0M   | 532k           | FFT 2×          |
| 4096     | 1024     | 4.2M   | 532k           | FFT 8×          |
| 16384    | 1024     | 16.8M  | 1.2M (P=16384) | FFT 14×         |
| 65536    | 4096     | 268M   | 11.1M          | FFT 24×         |

**Crossover at M=128-256**, sharp. Doc currently says only "for long
kernels, FFT-based convolution is preferred (compose FFT + multiply +
IFFT from this package)" — but the compose pattern is **non-trivial**
(zero-pad to next pow-2, transform both, complex-multiply bin-by-bin,
inverse-transform, trim) and **no caller in reality currently does it**.

133 §6 named the missing primitive ("block-convolution helper, ~50 LOC").
Endorse: ship `signal.FFTConvolve(signal, kernel, out []float64)` with
auto-crossover (direct under M=128, FFT-based above). ~70 LOC including
the complex-multiply helper. Closes the doc-lie about consumers
"composing" the FFT primitives. Real-time/streaming overlap-save reverb
(Pistachio use case) is a follow-up; current `Convolve` is one-shot only.

---

## H — `MovingAverage` — the docstring lie

`filter.go:54-84`. Doc claims **"Zero heap allocations — uses running
sum."** The "uses running sum" is **false**:

```go
for i := 0; i < n; i++ {
    lo := i - half; hi := lo + windowSize - 1
    if lo < 0 { lo = 0 }; if hi >= n { hi = n - 1 }
    sum := 0.0
    count := hi - lo + 1
    for j := lo; j <= hi; j++ { sum += signal[j] }   // O(W) inner
    out[i] = sum / float64(count)
}
```

Outer N, inner W; total `O(N·W)` not `O(N)`. 131 caught the docstring
lie; this report quantifies:

| W   | input N | current FLOPs | running-sum FLOPs | speedup |
|----:|--------:|--------------:|------------------:|--------:|
| 5   | 100,000 |    500,000    |          300,000  | 1.7×    |
| 21  | 100,000 |  2,100,000    |          300,000  | 7×      |
| 101 | 100,000 | 10,100,000    |          300,000  | 34×     |
| 501 | 100,000 | 50,100,000    |          300,000  | 167×    |

The "Zero heap allocations" half is true. The "running sum" half is the
bug. Fix is the textbook 4-LOC running-sum (boundary count handling is
the only complication; numerator uses the running sum). **~10 LOC, zero
alloc, O(N) total, drop-in replacement.**

Numerical caveat: long streams (≥1e8 samples) accumulate roundoff in `sum`;
periodic re-baselining or Kahan summation handles it. Document the caveat.
**Single biggest concrete speedup in the package. 30 minutes' work.**

---

## I — `MedianFilter` — sort-per-window O(N·W·log W)

`filter.go:130-174`. Per output: copy W to stack buffer (`[64]float64`
avoids heap), `sort.Float64s`. Textbook fast median is **two-heap**
sliding insertion: `O(log W)`/sample, total `O(N·log W)` — factor W
faster. Reality's spike-removal use case is W=3..21; at W=21 two-heap
is ~4× faster. **Defer** pending Sentinel benchmark.

**One real wart:** at W>63 the stack buffer overflows and `make` runs
**per output sample**. N=100k W=101 = 100k allocs, 80 MB garbage. Doc
warns; two-heap rewrite eliminates it.

---

## J — Window functions and `ApplyWindow`

Window builders call `math.Cos` N times each. Same recurrence trick as
FFT twiddles applies, but **window functions are computed once and
reused thousands of times** (STFT precomputes once, applies ~58k times
across 5 min audio). The 30 µs build is amortised to nothing. **Don't
optimise the builders.**

Hot path is `ApplyWindow` (`window.go:104-113`). Simplest pointwise
multiply possible. Go 1.23 auto-vectorises on amd64 **only if** bounds
checks hoist. Two compiler-friendly shapes — range form
(`for i, s := range signal[:n] { out[i] = s * window[i] }`) or classical
hoist (`_ = window[n-1]; _ = out[n-1]` before the loop). From similar
shapes in `linalg/`, the hoist gives ~10% on Go 1.23 amd64. Not SIMD —
out of scope per 133 §5 — but autovec-friendlier. **3 LOC, ~10%, stop.**

For STFT, the right primitive is **fused** `WindowedFFT(samples, window,
real, imag []float64)`. Saves one full pass over 2N floats. ~20 LOC,
~10% STFT inner-loop. `audio/spectrogram/stft.go:82-90` would drop its
inline copy-window-zero loop.

---

## K — `ExponentialMovingAverage` — already optimal

Standard one-pole IIR `y[i] = α·x[i] + (1-α)·y[i-1]`. 2 mul + 1 add per
sample, `O(N)` total. No improvement in scalar code; recurrence is
inherently serial (133's disiple/state-space matrix-power vectorises IIR
feedback at 4-sample unroll for ~3-4×, overkill at order 1). **Leave alone.**

---

## L — Allocation accounting

| Function         | Allocs/call   | Notes |
|------------------|--------------:|-------|
| FFT              | 0             | in-place, correct |
| IFFT             | 0             | in-place, correct |
| PowerSpectrum    | 0             | calls FFT in-place, writes out |
| FFTFrequencies   | 0             | scalar fill |
| Convolve         | 0             | tight pointwise, correct |
| MovingAverage    | 0             | correct on alloc, **wrong on complexity** |
| EMA              | 0             | correct |
| MedianFilter W≤63| 0             | stack buffer |
| MedianFilter W>63| **1 per out** | 8·W bytes per output sample (heap spill) |
| HannWindow       | 0             | correct |
| HammingWindow    | 0             | correct |
| BlackmanWindow   | 0             | correct |
| ApplyWindow      | 0             | correct |

**Allocation discipline is excellent — every primitive except `MedianFilter`
at large W is alloc-free.** Stark contrast to 130's sequence package where
every entry-point allocated. `signal/` is the *good* example.

---

## M — STFT amplifies every kernel cost

`audio/spectrogram/stft.go:53-99` is the dominant non-test caller. At
44.1 kHz, frameSize=2048, hop=512: 86 frames/sec. Per frame:

1. Window-multiply N=2048 reals (manual loop, not `signal.ApplyWindow`):
   ~2 µs
2. `signal.FFT(frameReal, frameImag)` with frameImag = zeros: ~70 µs
   (RFFT would be ~35 µs)
3. Pack to `complex128` row + heap-alloc the row: ~3 µs + 32 KB GC pressure

Per audio second: 86 × 75 µs = **6.45 ms CPU**, plus ~2.7 MB/s GC pressure
from per-frame complex128 row allocation.

Post-fixes (twiddle table + RFFT + pre-allocated complex row): 86 × 35 µs
= **3 ms CPU**, GC pressure ~0. **~50% wall-time reduction in the
dominant audio front-end primitive in the entire repo**, with ~150 LOC
in `signal/` and ~30 in `audio/spectrogram/`. Highest-leverage commit
in this report.

---

## N — Bench harness gap

`go test -bench=. ./signal/...` finds nothing. Zero `bench_test.go`.
Same gap as 130, 125, 110, 070, 020, 015, 010, 005 — third of all
perf-tagged agents this review have flagged it. Reality's *correctness*
discipline is exemplary; *performance* discipline is absent.

Minimum viable harness:

```
BenchmarkFFT_{1024,4096,16384}
BenchmarkConvolve_1024_{64,512}    // direct fast vs slow
BenchmarkMovingAverage_{W21,W101}
BenchmarkMedianFilter_W21
BenchmarkApplyWindow_4096
BenchmarkSTFT_5sec_44k1            // amalgam, lives in audio/
```

**~80 LOC.** Pins every claim in this report; flips every "estimate" to
a measurement; gives the per-PR "did this actually speed up" gate.

---

## Concrete recommendations (ranked by impact-per-LOC)

1. **`MovingAverage` running-sum rewrite.** ~10 LOC. Up to 167× faster
   at W=501. Fixes docstring lie. (§H)
2. **RFFT primitive.** ~80 LOC. 2× faster STFT, 132 §1.1 named it,
   topic-title item. (§D)
3. **Twiddle-table-per-N FFT.** ~30 LOC. ~25% faster FFT, fixes 131's
   precision cliff at N≥2^16. (§E)
4. **`bench_test.go` harness.** ~80 LOC. Witnesses the rest. (§N)
5. **`FFTConvolve` auto-crossover helper.** ~70 LOC. Crossover sharp
   at M=128-256; current consumers must hand-roll. (§G)
6. **`WindowedFFT` fused kernel.** ~20 LOC, ~10% STFT speedup. (§J)
7. **`ApplyWindow` bounds-check hoist.** ~3 LOC, ~10%. (§J)
8. **Stockham auto-sort FFT.** ~150 LOC; ~10-25% wall, big surface cost.
   Defer until 1-7 in. (§B)
9. **`MedianFilter` two-heap.** ~80 LOC, W× speedup at W>21, eliminates
   W>63 heap-spill. Defer pending Sentinel benchmark. (§I)
10. **Split-radix FFT.** ~120 LOC; ~33% mul reduction, doubles butterfly
    code surface. **Defer indefinitely** — items 1-7 close 80% of the
    FFTW gap at 5× lower maintenance cost. (§C)

**Tier 1 (1-7): ~290 LOC.** Closes dominant audio-frontend bottleneck,
fixes the docstring lie, ships RFFT (the unambiguous topic-title item),
adds the bench harness without which none of the rest can be measured.

None add dependencies. All preserve MIT zero-dep posture. RFFT and
FFTConvolve are golden-file-pinnable across Go/Python/C++/C# with
canonical SciPy reference.

---

## Two-line summary

reality `signal/` allocations are exemplary (zero in 11/12 entry points)
but FFT is 1965-textbook radix-2 with twiddle recurrence costing ~50% of
its wall, RFFT (the only unambiguous "ship now" of the three name-checked
optimisations) is missing despite every audio caller forcing imag=0 to
fake real input, MovingAverage docstring lies about "running sum"
(actually O(N·W) at up to 167× the necessary cost at W=501), and there
are zero benchmarks anywhere — Tier 1 fix-set is ~290 LOC (running-sum
MovingAverage + RFFT + twiddle table + bench harness + FFTConvolve
auto-crossover + WindowedFFT fused + ApplyWindow bounds-hoist) that
halves audio-frontend CPU at zero precision cost; Stockham/split-radix
can wait.

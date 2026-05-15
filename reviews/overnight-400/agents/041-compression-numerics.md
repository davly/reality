# 041 | compression-numerics

Numerical-correctness audit of `C:\limitless\foundation\reality\compression\`.
Files: `entropy.go` (177 lines), `coding.go` (104), `quantize.go` (99),
`compression_test.go` (693), one golden file `shannon_entropy.json` (10 cases).

## TL;DR

The implemented surface is small (5 entropy fns + RLE + Delta + Quantize) and
its happy paths are computed correctly to float64 precision, but the package
is **shipping with documentation it does not implement**: CLAUDE.md, README.md,
and ARCHITECTURE.md all advertise **Huffman** and **LZ77** as compression
primitives. Neither exists in the source tree (no `Huffman*`, `LZ*`, or
`Compress*` symbols anywhere under `reality/compression/`). The advertised
"Lossless/lossy compression primitives: entropy, RLE, delta encoding, Huffman,
LZ77" is a documentation lie that propagates through the repo's main
discoverability surfaces.

Within what *is* implemented:

1. **No bias correction** — plug-in entropy is the only estimator; for short
   sequences this is the well-known Miller "ln 2 underestimate"
   (`E[Ĥ] = H − (k−1)/(2N ln 2) + O(1/N²)`); not even documented as a caveat.
2. **No frequency-counting estimator** at all — caller must produce probs
   themselves, so the package never sees raw counts and cannot apply MM/JS
   even if it wanted to.
3. **Conditional/joint use plug-in twice** so any future bias correction must
   handle the joint→marginal subtraction's compounded MSE.
4. **Mutual information has a real bug**: ragged joint matrices silently
   produce wrong column-marginal lengths but the function returns a number
   anyway (no validation, no error). Same for ConditionalEntropy on
   non-square joints.
5. **DeltaEncode silently overflows** on int64 — `data[i] - data[i-1]` wraps
   modulo 2^64 with no detection; the docstring even says "assuming no int64
   overflow" but the function does not enforce or detect it.
6. **RLE max-run logic is correct** (splits at 255), but the `count=0`
   ambiguity is not handled: `RunLengthDecode` will accept `[0, x]` and
   produce nothing for that pair, silently dropping a 256-byte run if the
   encoder ever produced it (it doesn't, but invariant is undocumented).
7. **Bits vs nats**: package is hardcoded log2 (bits) with no `*Nats`
   variants; users who need natural-log entropy must rescale themselves.
   Cross-entropy and KL silently produce bits but loss-function consumers
   typically expect nats — minor footgun.
8. **IEEE-754 handling**: `p > 0` filter is correct for `+0.0`, `-0.0`, and
   subnormals (treated as zero only for `-0.0` and exact `0.0`; subnormals
   are kept and their `log2` is finite), but **NaN entries silently
   propagate to NaN output** with no documentation; `+Inf` entries make
   `H = -Inf` (since `Inf * log2(Inf) = +Inf`, negated, but only if Inf
   passes `p > 0`, which it does).

Severity tally: 1 missing-feature documentation defect (S1, repo-wide), 4
silent-failure numerical defects (S2), and 6 documentation/contract gaps.

---

## 1. ShannonEntropy bias

```go
func ShannonEntropy(probs []float64) float64 {
    h := 0.0
    for _, p := range probs {
        if p > 0 {
            h -= p * math.Log2(p)
        }
    }
    return h
}
```

**Plug-in (maximum-likelihood) estimator.** When the caller has computed
`probs[i] = count[i] / N` from finite samples, this estimator is biased
**downward** by

```
bias(Ĥ_MLE) = -(k − 1) / (2 N ln 2)   bits   (Miller 1955)
```

where `k` = number of distinct symbols with nonzero true probability.
For a 4-symbol uniform source observed at N=100, the bias is ~−0.0216 bits
(against a true H of 2 bits — about 1% relative). For the Pistachio
embedding-quantization use case (~256 levels, ~1024 samples per chunk), the
bias is ~−0.180 bits against a typical ~6 bits H — **3% relative
underestimate**. None of this is documented.

**Recommended fixes (additive, no API break):**

A. Document the bias and prescribe Miller-Madow when N is known:

```
H_MM = Ĥ_MLE + (k_obs − 1) / (2 N ln 2)
```

where `k_obs` is the count of *observed* nonzero bins. Adds a single fn:

```go
// MillerMadowCorrection returns the additive bias correction
// (k_obs − 1) / (2 N ln 2) in bits. Add to ShannonEntropy(probs) when
// probs were estimated from N samples by maximum likelihood.
func MillerMadowCorrection(probs []float64, n int) float64 { ... }
```

B. Provide a counts-based entrypoint that does the right thing by default:

```go
// EntropyFromCounts returns the Miller-Madow-corrected plug-in entropy
// estimate in bits, given raw integer counts. This is what callers should
// use whenever they have observation data rather than a known distribution.
func EntropyFromCounts(counts []int) float64 { ... }
```

C. Optionally James-Stein shrinkage (Hausser-Strimmer 2009) — the practical
   state-of-the-art for finite-sample entropy from small alphabets. Single
   function, ~30 LOC, dependency-free.

These are **additions**, not modifications to ShannonEntropy itself —
ShannonEntropy can keep being the pure mathematical function for known
distributions, and the new fns are for the empirical case. This matches
how scipy.stats.entropy is the math fn while estimator packages
(`numpy.histogramdd → entropy`, `pyitlib`) layer corrections on top.

## 2. JointEntropy / ConditionalEntropy / MutualInformation

All three functions share these issues:

**(a) Ragged-row joints produce silently wrong answers.** MutualInformation:

```go
maxCols := 0
for _, row := range joint {
    if len(row) > maxCols { maxCols = len(row) }
}
marginalY := make([]float64, maxCols)
for _, row := range joint {
    for j, p := range row { marginalY[j] += p }
}
```

If row 0 has 3 cols and row 1 has 2 cols, the marginal-Y vector has length 3
but only some rows contribute to column 2 — caller gets a number that
satisfies no theoretical identity. Either reject ragged input
(return NaN with documentation) or document that joints **must** be
rectangular and panic otherwise (matching the `linalg` matrix convention).

**(b) ConditionalEntropy = JointEntropy − ShannonEntropy(marginalX) is the
naive subtraction form.** This is the canonical formula but it is the
**least numerically stable** way to compute H(Y|X) when H(X,Y) ≈ H(X)
(strong dependence). The numerically preferred form is the direct
double-sum

```
H(Y|X) = -Σ_i p(x_i) Σ_j p(y_j|x_i) log2 p(y_j|x_i)
       = -Σ_{i,j} p(i,j) log2( p(i,j) / p_X(i) )
```

which avoids cancellation. For perfectly correlated X=Y the test passes
because both terms are exactly representable, but for near-deterministic
joints (e.g. p=0.99, 0.01 on the diagonal, ε on the off-diagonals)
the subtraction loses ~6 decimal digits when ε~1e-7. Recommend internal
direct form, keep the identity for the docstring.

**(c) Bits-only.** No `JointEntropyNats`, no log-base parameter. Cover &
Thomas Ch. 2 uses both freely; ML loss functions universally use nats.
Trivial fix: package-level constant `Log2E` and `*Nats` companions, or a
parameter.

**(d) Mutual information non-negativity is documented but not enforced.**
For pathological floating-point joints summing to 0.999999, MI can return
a small negative value. The test `TestMutualInformation_NonNegative`
allows `-1e-14` slack. Recommend clamping at 0 with a documented
`mathutil.Max(0, mi)` and a comment citing Cover & Thomas Theorem 2.6.5.

## 3. KL divergence and CrossEntropy

**(a) Asymmetric "asymmetric" test is wrong.** `TestKLDivergence_Asymmetric`
uses `p = [0.9, 0.1]` and `q = [0.1, 0.9]` and asserts `KL(P||Q) == KL(Q||P)`
**because this particular pair is the swap-mirror case**. The test name
claims to verify asymmetry; the assertion verifies the opposite. This is a
test-suite bug (not a numeric bug), but it means the actual asymmetry
contract has zero coverage. Better test: `p = [0.5, 0.5]`, `q = [0.25, 0.75]`
gives `KL(P||Q) ≈ 0.2075` and `KL(Q||P) ≈ 0.1887`, asymmetric.

**(b) `+Inf` short-circuit is correct** but gives no diagnostic — caller
gets +Inf with no information about *which* index was the offender. For
the Recall cache-compression use case (probability vectors over thousands
of cache lines), debugging which line caused +Inf is hard without
instrumentation. Optional: `KLDivergenceWithIndex` that returns
`(value, problemIdx int)` for diagnostics.

**(c) `q[i] <= 0` guard combined with `<= 0` rejection of negative q
masks a real input-validation bug**: a small negative `q[i] = -1e-300`
caused by upstream subtraction underflow returns +Inf, but a small
positive `q[i] = +1e-300` produces a finite but enormous `p log(p/q)`
term that swamps the rest of the sum. Inconsistent treatment. Either
both negative and tiny-positive should be flagged, or neither.

## 4. RunLengthEncode / Decode

**(a) `count=0` is reachable on decode but never on encode.** Encoder loop
guarantees `count >= 1`. Decoder accepts `[0, x]` as a no-op pair. This is
not a bug per se but the invariant should be documented and probably
enforced (`return nil` on `count == 0`).

**(b) Run-length 256 boundary documentation.** The docstring says "Runs
longer than 255 are split into multiple pairs" — but the encoder splits at
255, not 256. So a run of 255 is one pair `[255, x]`; a run of 256 is two
pairs `[255, x][1, x]`. This matches the docstring. Fine.

**(c) Worst-case capacity hint is wrong.**

```go
encoded := make([]byte, 0, len(data))   // worst case is 2x len(data)
```

Comment says "Worst case is 2x (no runs)" but the allocation is `len(data)`,
not `2*len(data)`. This forces append-grow for any input with no runs.
~free perf fix.

**(d) `count` type.** `count` is `int` widened from a `byte` comparison; on
inputs > 2³¹ bytes (multi-GB texture buffer for Pistachio) the `i+count`
indexing is fine on 64-bit Go but `count < 255` cap means at most 8-bit
arithmetic so no overflow risk. OK.

## 5. DeltaEncode / Decode — overflow

```go
encoded[i] = data[i] - data[i-1]   // unchecked
```

Docstring says "assuming no int64 overflow in differences". But the
realistic Pistachio/Oracle use case is **monotonic timestamps in
nanoseconds** where neighbours are O(1ms) = O(1e6) ns apart — no overflow
risk. However, for the *adversarial* case `data = [math.MinInt64,
math.MaxInt64]`, the delta is `MaxInt64 - MinInt64 = -1` (silent wrap) and
`DeltaDecode` happily reconstructs `[MinInt64, MinInt64-1] = [MinInt64,
MaxInt64]` because it wraps back. Round-trip works **because both wrap the
same way**, but the encoded delta is not the true mathematical difference.

This is acceptable behavior for a lossless coder (round-trip is what
matters), but the docstring claim "Precision: exact (lossless, assuming no
int64 overflow)" is internally inconsistent — the function *is* lossless
even with overflow because the inverse op also wraps. Recommend rewording:
"Precision: exact (lossless via two's-complement wraparound; output deltas
are residues mod 2⁶⁴, not unbounded integers)."

For the saturation variant (clamping to ±MaxInt64 instead of wrapping),
**the round-trip is destroyed**: saturating delta encode is lossy.
Standard Gorilla compression uses wraparound, not saturation. Reality's
choice is correct; just document it.

## 6. ScalarQuantize — float-edge corner cases

**(a) `max == min` branch is correct** but uses exact float equality. For
data that is "almost constant" (drift ~1e-15), step becomes denormal then
zero and reconstruction loses the drift. Acceptable for a quantizer (the
caller asked for fewer levels), but worth a docstring note.

**(b) NaN propagation.** A single NaN in `data` makes both `min` and `max`
become NaN (NaN compares neither `<` nor `>`), `step` becomes NaN, and all
output bins are 0 (because `int(math.Round(NaN/NaN)) = -2147483648`,
clamped to 0 by the bounds check). So bin output is "valid" (0..levels-1)
but the dequantized values are all NaN. Document or reject.

**(c) `+Inf` data.** `step = +Inf`; `(v - min) / step = 0` for any finite
v; all bins 0; reconstruction gives `min + 0*Inf = NaN`. So Inf-tainted
data also corrupts dequantization. Document or reject.

**(d) `levels` upper bound.** No check that `levels` fits in an `int` bin
(could exceed `MaxInt32` on 32-bit builds). Trivial.

## 7. Missing primitives (the big one)

CLAUDE.md (line 25), README.md (line 25), and ARCHITECTURE.md (line 79)
all advertise `compression`: "...entropy, RLE, delta, **Huffman**, **LZ77**".
But `grep -r 'Huffman\|LZ77\|LZ78' compression/` returns **no matches**.
The package doc in `entropy.go:1-15` correctly states only RLE/delta/
quantize, so the three top-level docs are out of sync with the source.

This is **the most important finding of this audit**. The repo's main
README, the architecture doc, and CLAUDE.md (which every agent in the
overnight-400 review reads) all promise functionality that does not
exist. Fix options:

A. **Remove the claims** (3 single-line edits). Honest fix, ~5 min.

B. **Implement the missing primitives.** Topic-specific edge cases:
   - Canonical Huffman (~250 LOC): single-symbol → length-1 codeword by
     convention (length-0 is mathematically optimal but unrepresentable);
     ties → stable lex tie-break for cross-language reproducibility;
     weight=0 → exclude (Huffman undefined for p=0); all-equal-weights →
     verify L ∈ {⌈log2 n⌉, ⌈log2 n⌉−1} within Kraft slack.
   - LZ77 (~200 LOC): start-of-stream empty-window → emit `(0, 0, lit)`.
   - LZ78 (~150 LOC).

**Recommend A immediately, B as a separate issue.** Shipping false
claims in CLAUDE.md is worse than shipping a smaller package; agents
042-400 reading CLAUDE.md may design downstream code assuming
Huffman/LZ77 exist.

## 8. Golden-file coverage

Only **one** golden file: `shannon_entropy.json`, **10 cases** (50% of
CLAUDE.md's "min 20 / target 30" floor for ShannonEntropy alone). **Zero**
golden files for JointEntropy, ConditionalEntropy, MutualInformation,
KLDivergence, CrossEntropy, RLE, Delta, ScalarQuantize.

Required floor per CLAUDE.md: **200 vectors across 9 functions**.
Current: 10. Gap: ~190 vectors (95% missing). Worst golden coverage of
any audited package so far. Unit tests in `compression_test.go` are good
*Go* tests but cannot validate Python/C++/C# parity per the design.

**Recommended additions:**
1. 20-30 vectors per function, ~9 new JSON files.
2. IEEE-754 edges per CLAUDE.md: `+Inf`, `-Inf`, `NaN`, `-0.0`, subnormals.
3. Theoretical-entropy fixtures: Bernoulli(p) sweep p∈{.01,.1,.25,.5,.75,
   .9,.99}; 2- & 3-state Markov stationary entropy rate; truncated
   geometric distribution.

## 9. Topic-specific verdict (numerical-correctness lens)

| Topic line | Status |
|---|---|
| Plug-in biased downward | **GAP** — undocumented, no MM/JS |
| Miller-Madow / James-Stein | **MISSING** |
| Joint/conditional numerical stability | **GAP** — identity-form subtraction used |
| Huffman / LZ77 edge cases | **N/A — not implemented** |
| RLE empty / max-run | **OK** |
| Delta integer overflow | **GAP** — wraparound is round-trip-safe but docstring claims "no overflow" |
| Bits vs nats | **OK** (hardcoded log2, no mixing) |
| `log(0)` for zero-frequency symbols | **OK** |
| `-Inf` entropy edge | **OK** (cannot occur) |
| Theoretical Bernoulli/Markov golden | **GAP** — only uniform & a few biased coins |

## 10. Recommended commit ladder

**P0 (one-line truth fix, must ship before any other agent reads CLAUDE.md
again):**
- C1: Edit CLAUDE.md, README.md, ARCHITECTURE.md to remove "Huffman, LZ77"
  from compression description. ~3 line edits.

**P1 (numerical correctness):**
- C2: Add `MillerMadowCorrection` and `EntropyFromCounts` ~25 LOC.
- C3: Document log-base (bits) and bias caveats in `ShannonEntropy`,
  `JointEntropy`, `ConditionalEntropy`, `MutualInformation`. ~12 lines
  of docstring.
- C4: Validate rectangular-joint contract in joint/conditional/MI
  (panic on ragged or document and accept). ~9 LOC.
- C5: Switch `ConditionalEntropy` to direct-form double-sum; keep
  identity in docstring. ~12 LOC.
- C6: Clamp `MutualInformation` at 0 with documented citation. 1 line.
- C7: Document NaN/Inf propagation in `ShannonEntropy`, `ScalarQuantize`.
  ~6 lines.
- C8: Reword `DeltaEncode` docstring — wraparound semantics. 2 lines.

**P2 (golden-file floor):**
- C9: Author 9 JSON files × 20-30 vectors each = ~210 vectors.
  Includes Bernoulli sweep, 2-state Markov, geometric, IEEE-754 edges.
  Probably 1 day of careful arithmetic + bigfloat reference.

**P3 (missing primitives — separate issue, not this audit's scope):**
- C10: Implement Huffman (canonical, with tie-break and weight=0 contracts).
- C11: Implement LZ77 (start-of-stream window edge case documented).
- C12: Implement LZ78 dictionary tree.
- Each ships with golden vectors per C9.

## Cross-references

- Topic 042 (compression-missing) should pick up C10-C12 scope.
- Topic 029-031 (combinatorics review chain) provides the reference
  pattern for *Big and *Nats variants.
- Topic 022 (graph) has the gold-standard golden-file coverage to mirror.
- testutil/golden.go is the established harness; reuse, do not reinvent.

## Files referenced

- `C:\limitless\foundation\reality\compression\entropy.go` (1-177)
- `C:\limitless\foundation\reality\compression\coding.go` (1-104)
- `C:\limitless\foundation\reality\compression\quantize.go` (1-99)
- `C:\limitless\foundation\reality\compression\compression_test.go` (1-693)
- `C:\limitless\foundation\reality\compression\testdata\compression\shannon_entropy.json` (10 cases)
- `C:\limitless\foundation\reality\CLAUDE.md` line 25 (Huffman/LZ77 claim)
- `C:\limitless\foundation\reality\README.md` line 25 (Huffman/LZ77 claim)
- `C:\limitless\foundation\reality\ARCHITECTURE.md` line 79 (Huffman/LZ77 claim)

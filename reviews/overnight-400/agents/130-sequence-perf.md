# 130 — sequence: performance audit (Myers, Hyyrö, Hirschberg, BK-tree)

Scope: `C:\limitless\foundation\reality\sequence\` (distance.go, alignment.go,
ngram.go, phonetic.go, token_ratio.go). 15 entry points; **zero benchmarks
in the package** (verified `Bench` grep — only `Test*` functions). Reality
charter rule 3: "no allocations in hot paths. Functions accept output
buffers. Pistachio calls these at 60 FPS." Sequence package is the worst
offender against rule 3 in the whole repo — every entry point allocates.

Distinct from 126 (numerics), 127 (missing primitives — Myers, Hyyrö,
Hirschberg, BK-tree all named there), 128 (SOTA — proposed Myers/Hyyrö
ports), 129 (rune/byte/grapheme contract, alloc count). This slot:
**which CPU cycles and which heap allocs to remove first, and what the
post-port cache layout has to look like to actually realise the 10× win
on m≤64 that 128 promised.**

---

## A — Per-call allocation profile of the 15 entry points

Counted by inspection. "small" = `[]bool` of len ≤ 64 or one `[]int` of
length ≤ 65; "medium" = `O(min(m,n))` int slice; "large" = `O(mn)` matrix.

| Function                  | Allocs | Heap bytes (m,n=k chars Latin-1) | Driver                                            |
|---------------------------|-------:|----------------------------------|---------------------------------------------------|
| `LevenshteinDistance`     |  4     | 2·`[]rune`(4k) + 2·`[]int`(8(k+1))| 2·rune cast, 2·int row                           |
| `DamerauLevenshtein`      |  3+m   | 2·`[]rune` + (m+1) inner `[]int` | full matrix, allocs per row                       |
| `HammingDistance`         |  2     | 2·`[]rune`                       | rune cast only                                    |
| `JaroWinkler`             |  4     | 2·`[]rune` + 2·`[]bool`(la,lb)   | rune cast + match flags                           |
| `LongestCommonSubsequence`|  3+m   | 2·`[]rune` + (m+1) inner `[]int` | full matrix per-row alloc + result `[]rune`       |
| `LongestCommonSubstring`  |  3+m   | same as LCS                      | full matrix even though only need 2 rows          |
| `NGrams`                  |  2+(L) | `[]rune` + `[]string`(L) + L·str | `string(runes[i:i+n])` per n-gram                 |
| `WordNGrams`              |  1+(L) | outer + L inner `[]string`       | per-window slice copy (correct for safety)        |
| `NGramSimilarity`         |  4+(L) | 2·NGrams + 2·`map[string]struct` | hash maps over strings, GC-heavy                  |
| `NGramDiceCoefficient`    |  4+(L) | identical to NGramSimilarity     | duplicates the hash-set construction              |
| `Shingling`               |  1+(L) | `[]rune` + L·`string` (cast away)| `string(runes[i:i+k])` only to feed into bytes    |
| `NeedlemanWunsch`         |  3+m   | 2·`[]rune` + (m+1)·`[]float64`   | full DP, two output `[]rune`                      |
| `SmithWaterman`           |  3+m   | identical to NW                  | full DP, two output `[]rune`                      |
| `Soundex`                 |  2     | `[]rune` + small `[]byte`        | rune cast (could be byte loop)                    |
| `TokenSetRatio`           | 12+    | `strings.ToLower` + `Fields` + 2·map + 6·sort/Join + LD×3 | dominant per-call allocator |

Twelve of fifteen are **byte-count** allocations the caller has no way to
suppress. Pistachio frame budget is 16.7 ms; one `LevenshteinDistance("set
the fan to high", "set fan to high")` allocates ~340 bytes and four heap
objects. At 10⁴ fuzzy lookups/frame the package alone produces 4×10⁵
allocs/s = ~3 GB/s of GC pressure. **This is the package's #1 perf bug
and it is independent of which algorithm is in the kernel.**

Three concrete recipes (each ~30 LOC), in priority order:

1. **`*Bytes` overloads for the 7 ASCII-fast-path functions** — Levenshtein,
   DamerauLevenshtein, Hamming, JaroWinkler, LCS-len-only, NGrams (yield via
   index pairs), Shingling. Pure ASCII content (60% of Pistachio token
   inputs) skips the rune cast and saves 2 allocs per call.
2. **Workspace-arg variant**: `LevenshteinDistanceWS(a, b []byte, ws []int)
   int` where `ws` is a caller-managed scratch of length `2*(min(m,n)+1)`.
   Mirrors `signal.FFT(out, in []float64)`-style API in this repo. Zero
   alloc per call after warmup.
3. **`LevenshteinPrenorm(a, b []rune)` / `LevenshteinPrenormBytes`** —
   accept already-cast inputs. The caller building a corpus index (BK-tree,
   Levenshtein automaton dictionary, fuzzy-extract over a 10k-name list)
   pays the rune cast **once** and re-uses it on every lookup. Today
   re-casts on every comparison.

---

## B — Wagner-Fischer cache layout: 2 rows is right; full matrix is wrong

Current `LevenshteinDistance` already does the 2-row trick (distance.go:35
`prev := make([]int, m+1); curr := make([]int, m+1)`). **DamerauLevenshtein,
LCS, LCSubstr, NW, SW all keep the full O(mn) matrix.**

For each, the minimum-rowset that suffices for the **score** is:

| Algorithm          | Rows for score | Rows for traceback path |
|--------------------|----------------|-------------------------|
| Levenshtein        | 2              | 2 + Hirschberg recursion |
| Damerau-Levenshtein| 3 (i-1, i, i-2 lookback) | 3 (same trick)   |
| LCS                | 2              | 2 + Hirschberg recursion |
| LCSubstr (length)  | 2              | n/a — return slice into ra |
| NeedlemanWunsch    | 2 (score), full (path) | 2 + Hirschberg meet-in-middle |
| SmithWaterman      | 2 (score+best) | 2 + traceback restart at best cell, walk locally |

**LCSubstr is the easiest fix in the package.** It builds an O(mn) `[][]int`
matrix and only ever reads `dp[i-1][j-1]`. Two rows suffice. **30-line
diff,** drops memory from 80 KB (1k×1k) to 8 KB.

**Cache-layout note for the 2-row variant:** the present code allocates
`prev` and `curr` as **two separate `[]int` slices**. They land on
different cache lines, and the inner loop `min3(prev[i]+1, curr[i-1]+1,
prev[i-1]+cost)` touches both. Better: one combined `[]int` of length
`2*(m+1)` with `prev := buf[:m+1]; curr := buf[m+1:]`. Same alloc, but
the swap is `prev, curr = curr, prev` (cost: nothing) and the two rows
are guaranteed contiguous in memory — better prefetcher behaviour, better
TLB locality on m near 64. Measurable on m≥256.

**Anti-pattern in the current code (distance.go:74-79):**
```go
dp := make([][]int, m+1)
for i := range dp { dp[i] = make([]int, n+1); dp[i][0] = i }
```
This allocates m+2 separate slabs. The slab headers stay in one slice but
the int payloads end up on m+1 separate heap pages → cache thrashing on
the diagonal, and m+1 finalisable objects for the GC to walk. The
universal fix throughout reality (and well-known in numerical Go) is the
**flat-buffer trick**:
```go
flat := make([]int, (m+1)*(n+1))
dp := make([][]int, m+1)
for i := range dp { dp[i] = flat[i*(n+1):(i+1)*(n+1)] }
```
One alloc, contiguous, ~1.5-2× speedup on m=n=512 just from cache-line
alignment. Apply to: DamerauLevenshtein, LCS, LCSubstr, NW, SW. **150
LOC delta total across all five.**

---

## C — Myers 1999 bit-parallel Levenshtein: the 10× kernel

128 specced the port (~80 LOC core). This slot covers **what the API
shape has to be** for the 10× win to actually land on Pistachio's hot path.

### C.1 Kernel sketch (canonical Myers, m ≤ 64)

```
m = len(p);   if m == 0 { return len(t) }
peq[c] := bit i set iff p[i] == c   for c in alphabet
VP, VN := ones(m), zeros(m)        // running positive/negative bitvectors
score := m
for j := 0; j < len(t); j++ {
    X := peq[t[j]] | VN
    D0 := ((VP + (X & VP)) ^ VP) | X
    HP := VN | ^(D0 | VP)
    HN := D0 & VP
    if HP & (1<<(m-1)) != 0 { score++ }
    if HN & (1<<(m-1)) != 0 { score-- }
    HP <<= 1; HN <<= 1
    VP = HN | ^(D0 | HP)
    VN = D0 & HP
}
return score
```

Seventeen bit-ops per text character. Independent of m as long as m ≤ 64.
**No allocations once `peq` is sized.** For m ≤ 64 and ASCII text, `peq`
is 256 `uint64` = 2 KB stack (Go function frame allocates it with
`var peq [256]uint64` — array, not slice — guaranteed stack-resident if
function doesn't escape). For m > 64, `peq` is `[256][⌈m/64⌉]uint64` and
escapes to heap on first sizing → caller-supplied `[]uint64` workspace
buffer mandatory per CLAUDE.md rule 3.

### C.2 API shape that doesn't waste the 10× win

Three signatures, in increasing engineering generality:

```go
// MyersLevenshtein64 — m ≤ 64, ASCII bytes. Stack-only. ~25ns at m=10, n=20.
func MyersLevenshtein64(text, pattern []byte) int

// MyersLevenshtein — m ≤ 64, runes. One small alloc for peq map (alphabet>256).
func MyersLevenshtein(text, pattern []rune) int

// MyersLevenshteinWS — arbitrary m, caller-managed peq/VP/VN buffers.
func MyersLevenshteinWS(text, pattern []rune, ws *MyersWorkspace) int

type MyersWorkspace struct {
    Peq map[rune][]uint64  // or []uint64 indexed by alphabet code
    VP, VN []uint64
}
```

The `MyersWorkspace` mirrors `signal.FFTWorkspace`-style; you build once,
reuse across an entire corpus query. **Critical:** the existing
`LevenshteinDistance` cannot be replaced in place without changing its
allocation profile. Ship `MyersLevenshtein64` as a sibling, then quietly
have `LevenshteinDistance` dispatch to it when both inputs are ≤ 64
runes. Backward-compatible 10× speedup on the 90% case.

### C.3 Why "SIMD" in the topic title is misleading

The topic line says "SIMD bit-parallel". **Myers is not SIMD; it is
SWAR** (SIMD Within A Register) — bit-parallelism over a single 64-bit
machine word. True SIMD-batch (rapidfuzz `fuzz::experimental`) processes
16-32 candidates in parallel using packed 8/16-bit lanes via AVX2/AVX-512.
That requires Go assembly (Plan-9 syntax) or `cgo` — **both forbidden by
CLAUDE.md rule 2** (zero deps, stdlib only). So:
- ~10× from scalar Myers/Hyyrö (this slot, in scope).
- Another 8-16× from packed-lane SIMD (out of scope; document as ceiling).

Do not ship the SIMD-batch variant; ship the SWAR kernel. Pistachio's
single-input vs single-pattern usage is exactly where SWAR wins and SIMD
batching does not apply (no batch).

---

## D — Hyyrö 2003 Indel similarity: rapidfuzz's actual hot loop

128 named this. Distinct from Myers because rapidfuzz's `ratio` family
(simpleRatio in the current `TokenSetRatio`) uses **Indel distance**
(insertions + deletions only, no substitution), which has an even tighter
bit-parallel loop (8 bit-ops instead of 17).

### D.1 Why it matters for THIS package

`TokenSetRatio` calls `simpleRatio(t0, t1)`, `simpleRatio(t0, t2)`,
`simpleRatio(t1, t2)` (token_ratio.go:66-72). Each call invokes
`LevenshteinDistance` → 2-row Wagner-Fischer scan over the joined token
strings. Three calls, three full DP scans. **rapidfuzz's `token_set_ratio`
makes the same three calls but each is Hyyrö-Indel** → 4× faster per call,
12× faster overall, AND `simpleRatio` is the **wrong distance** for ratio
semantics — rapidfuzz uses Indel because the LCS-equivalent ratio
`100·(len_a+len_b - indel) / (len_a+len_b)` is the canonical fuzzy ratio
(see Levenshtein-distance vs LCS-distance in Navarro 2001 §3).

### D.2 The defect coupling

This is also a numerical-correctness item (126 should pick it up but
didn't): **simpleRatio computed via Levenshtein gives different results
than the same ratio computed via Indel distance** when substitution is
cheaper than insert+delete. RapidFuzz documents this; `reality` does not.
So the perf win (Hyyrö-Indel ~150 LOC drop-in for `simpleRatio`) and a
small correctness regression-test land in the same PR.

### D.3 Hyyrö skeleton (Indel = LCS-distance, m ≤ 64)

```
peq[c] := bit-mask of positions where p[i]==c
S := ones(m)
for j := 0; j < len(t); j++ {
    M := peq[t[j]]
    U := S & M
    S = (S + U) | (S - U)   // single-line Hyyrö Indel update
}
score := popcount(^S & ((1<<m)-1))   // = LCS length; indel = m+n-2*lcs
```

Eight bit-ops per text character. ~25-30% faster than Myers core scalar.

---

## E — Hirschberg O(n) space LCS / alignment-path traceback

Topic line names this. Current LCS reconstruction (distance.go:273-285)
costs O(mn) **time** plus O(mn) **memory** because the back-pointer table
is the full DP. Hirschberg 1975 splits a/b at the midpoint of a, computes
forward DP up to mid and reverse DP from mid, picks the column k that
maximises `forward[mid][k] + reverse[m-mid][n-k]`, recurses on
(a[:mid], b[:k]) and (a[mid:], b[k:]). Same O(mn) time, O(min(m,n)) space.

### E.1 Why it matters at scale

Pistachio's transcript-diff feature (aicore embedding diff against
reference transcript) compares ~100 KB strings. Current LCS allocates
**10 GB** of `[][]int` matrix on a 100k-rune diff and OOMs the process.
Hirschberg compares the same in **800 KB** scratch. **This is not "nice
to have"; this is "current code crashes on this input."**

### E.2 Composes with Myers

128 names "Edlib runs Myers forward from both ends to the meet-in-the-
middle row, then recurses. Path recovery in O(m+n) space instead of
O(mn)." Same idea applied to Levenshtein traceback. The Hirschberg
recursion harness is **shared infrastructure**:

```go
// internal helper; ~80 LOC
func hirschbergSplit(a, b []rune, score func(a, b []rune) []int) (mid, cut int)

// public surfaces
func LongestCommonSubsequenceLinear(a, b string) string  // 2-row + Hirschberg
func LevenshteinAlignment(a, b string) (alignA, alignB string)  // Myers fwd/rev meet
func NeedlemanWunschLinear(a, b string, m, x, g float64) (string, string, float64)
```

**Total: ~250 LOC, drop-in replacements** for LCS + adds two new entry
points (alignment-with-path) that 127 also asked for under different
headings. Single coherent PR.

---

## F — BK-tree query asymptotics

Topic line names this; 127 specced it as Tier 1 §1.13 with API
`(*BKTree).Search(query string, maxDist int) []string`. Pure perf-side
notes:

### F.1 Edit-distance metric is required

BK-tree's correctness depends on the triangle inequality of the metric.
**Levenshtein is a metric** (proven; positive, symmetric, triangular).
**Damerau-Levenshtein is NOT a metric** in general — the OSA variant
violates triangle inequality on inputs like `"ca", "abc", "ac"`. BK-tree
must restrict to Levenshtein (or true Damerau, which is harder); document
this on `BuildBKTree`. **Current `DamerauLevenshtein` is the OSA variant**
(distance.go:70 — restricted edit distance, line 99 lookback only at
i-2,j-2 for the last-pair transposition). Do not silently accept it as the
BK-tree metric.

### F.2 Average query complexity

For a tree built over n strings with edit distance bounded by max-radius
r and dictionary diversity d (mean alphabet richness):
- **Insert:** O(log_b n) where b = mean branching factor (~7-12 for
  English dictionaries). On 100k-word corpora: ~5-6 distance computations
  per insert. ~80 K LD calls to build. With Myers: ~2 ms.
- **Search radius r:** O(n^{1 - r/(2·log b)}) average per Burkhard-Keller
  1973. For r=2 on 100k-corpus: ~150-300 nodes visited (vs full 100k
  scan). The win disappears when r grows past ~⌊log_b n⌋ ≈ 6 for English
  100k-corpus — at that point linear scan is competitive.
- **Worst-case search:** O(n) (e.g. perfectly balanced tree with query at
  centre). Document this — competing libraries (jellyfish, abydos) hide it.

### F.3 The LD-call-count budget

BK-tree visits visit O(n^{1-α}) nodes, each calling `LevenshteinDistance`.
**With Myers-64 on m≤64 strings: ~25 ns per call → 7.5 μs for a 300-node
visit → 130 K queries/sec on a single core.** With current Wagner-Fischer
(2-row, ~250 ns/call on m=20): ~75 μs per query → 13 K queries/sec. 10×
penalty for not having Myers. **This is the dependency: BK-tree without
Myers is half-built.**

### F.4 Caller-side cache: `[]rune` per query

Every `LevenshteinDistance(query, node.Word)` call re-casts query to runes.
BK-tree's hot-path search would naturally pre-cast `query` once and pass
`[]rune` to a pre-cast tree (each node stores the rune-slice form).
Memory cost: 4× over byte storage for ASCII; insignificant for a fuzzy-
lookup index. Saves the rune cast on every node visit.

---

## G — Defaulted summary table: where the cycles go in the current package

Estimate column = order-of-magnitude scalar perf at m=n=20 ASCII chars
(Pistachio command-token typical size), single core, Go 1.24.

| Function                  | Now (cyc) | Myers/Hyyrö (cyc) | Speedup | Allocs now | Allocs after |
|---------------------------|-----------|-------------------|---------|------------|--------------|
| `LevenshteinDistance`     | 2500      | 250               | 10×     | 4          | 0 (workspace)|
| `DamerauLevenshtein`      | 3500      | 1500 (no bit-par OSA)| 2.3× | 23         | 1 (3-row buf)|
| `JaroWinkler`             | 1800      | 1800 (kernel ok)  | 1×      | 4          | 0            |
| `LongestCommonSubsequence`| 4500      | 1200 (Hyyrö-LCS)  | 3.7×    | 23         | 1            |
| `LongestCommonSubstring`  | 3500      | 1500 (2-row)      | 2.3×    | 23         | 1            |
| `NeedlemanWunsch`         | 5500      | 5500 (DP)         | 1×      | 23         | 1 (Hirschberg)|
| `SmithWaterman`           | 5500      | 5500 (DP)         | 1×      | 23         | 1            |
| `simpleRatio` (in TSR)    | 2500      | 200 (Hyyrö-Indel) | 12×     | 4          | 0            |
| **TokenSetRatio** total   | ~12000    | ~1500             | 8×      | 12+        | 2-3          |
| `Soundex`                 | 350       | 250 (byte loop)   | 1.4×    | 2          | 0            |
| `NGramSimilarity`         | 4000      | 1200 (sort+merge) | 3.3×    | 4+(L+L)    | 1            |

Highest-leverage three:
1. **Myers-64 + dispatch from `LevenshteinDistance` + Hyyrö-Indel for
   simpleRatio** → 8-10× on the package's two most-called entry points.
2. **Flat-buffer trick + 2-row layout for LCSubstr/NW/SW** → 2× on the
   alignment family from cache locality alone, no algorithm change.
3. **`*WS` workspace overloads (~7 entry points)** → zero allocs in hot
   path, removes the GC pressure cliff that hits Pistachio at 60 FPS.

---

## H — Forward-looking structural items not in 126/127/128/129

1. **`Benchmark*_Allocations` CI gate.** Zero benchmarks today; without
   `testing.AllocsPerRun(...)` regression tests, "no allocs in hot paths"
   is unenforced. ~120 LOC: one benchmark per public entry point. Mirrors
   agent 090's package-wide proposal.
2. **`testdata/levenshtein_long.txt` golden fixture.** All current golden
   vectors are short. Myers port needs golden cases at m=64, m=65 (word
   boundary), m=100, m=1000 to pin multi-word path. ~30 vectors generated
   against `python-Levenshtein` per CLAUDE.md golden-file convention.
3. **Defer Levenshtein-automaton (127 §1.12) until Myers lands.** Different
   problem (one query vs many candidates) but shares the alphabet-bitmask
   pre-processing → automaton inherits Myers's `peq` builder.
4. **`fuzz/` consumer-side caching.** The right place for memoised
   `LevenshteinDistance` is the consumer (Pistachio token cache, aicore
   embedding diff). reality stays pure: caller cache, not package cache.
5. **Don't ship SIMD-batch.** Topic line invokes "SIMD" but the only Go
   path is asm or cgo — both forbidden by CLAUDE.md. SWAR (uint64-as-
   bitvector) is the legitimate ceiling. Document in package doc.go:
   "single-query latency optimised; batch-query throughput requires
   consumer-side parallelism."

---

## I — Recommended PR sequence (perf-only)

1. **PR-A (~80 LOC, zero risk):** flat-buffer trick for DamerauLev / LCS
   / LCSubstr / NW / SW. No API or behaviour change.
2. **PR-B (~250 LOC):** `MyersLevenshtein64` + `MyersLevenshtein` +
   delegate from `LevenshteinDistance` when m≤64. 10× on dominant case.
3. **PR-C (~150 LOC):** Hyyrö-Indel + `IndelDistance` + rework
   `simpleRatio` core in `TokenSetRatio`. Includes LCS-distance vs
   Levenshtein-distance correctness fix — flag in NEWS.md.
4. **PR-D (~250 LOC):** Hirschberg harness for `LevenshteinAlignment`,
   `LongestCommonSubsequenceLinear`, `NeedlemanWunschLinear`.
5. **PR-E (~120 LOC):** `Benchmark*` + `AllocsPerRun` CI gate.
6. **PR-F (~100 LOC):** BK-tree (depends on PR-B Myers).

Total ~950 LOC across 6 PRs, none break existing 1,965 tests. Outcome: package goes from "rule-3 violator" to "fastest pure-Go scalar edit-distance library available," within 2-5× of edlib without cgo/asm.

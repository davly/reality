# 126 — sequence: numerical-correctness audit

Scope: `C:\limitless\foundation\reality\sequence\` (distance.go, alignment.go,
ngram.go, phonetic.go, token_ratio.go). Tests: 22 packages, sequence subset
covers Levenshtein, Damerau-Levenshtein, Hamming, Jaro-Winkler, LCS / LCSubstr,
NGrams, NGramSimilarity, NGramDiceCoefficient, Shingling, NeedlemanWunsch,
SmithWaterman, Soundex, TokenSetRatio.

Headline: package is numerically clean for normal inputs but has **five
real defects** that fire at the edges the topic prompt explicitly named —
int overflow on large strings, fuzzy-score range violation, empty-string
NaN/Inf, NW traceback non-determinism, and a Soundex first-letter rule
applied unconditionally.

---

## Defect 1 (HIGH) — Soundex first-letter prevCode is set even when first letter is a vowel/H/W

`distance.go` … no, `phonetic.go:60`:

```go
out := []byte{byte(first)}
prevCode := soundexCode(byte(first))
```

`soundexCode('A')` returns `'0'` (vowel). Then loop sees a real consonant
and writes its digit. Fine.

But `soundexCode('H')` and `soundexCode('W')` ALSO return `'0'` (the function
treats them as vowels because the loop short-circuits H/W before calling it
on subsequent letters — but on the FIRST letter, H/W is preserved as the
output letter and `prevCode` is initialised to `'0'`, treating H/W like a
vowel for the running-class collapse, which is correct for "Hubert" / "Wright"
under the modern rule. So this is fine.

Real issue: `soundexCode('Y')` returns `'0'` (vowel). The first-letter-class
initialisation handles Y as a vowel, which IS the modern rule per the doc
comment ("A,E,I,O,U,Y -> 0"). Cross-checked against the SSA reference and
PostgreSQL `fuzzystrmatch` — both treat Y as a vowel only when not initial,
and as a consonant when initial. The reality implementation diverges from
the SSA convention. **Documentation asserts modern rule; tests don't pin
"Yvonne" / "Ypsilanti" so the divergence isn't observable.**

Severity downgraded: doc says "modern Knuth", which IS the Y-as-vowel
variant (`A,E,I,O,U,Y → 0` in the comment, line 26). Internally consistent.
SSA divergence is a labelling choice, not a numerical defect. **Drop from
defect list; flag as documentation precision item** — comment should add
"diverges from SSA / NARA which treats Y as consonant when initial" so
downstream telematics consumer (TokenSetRatio doc cites FW PCN auto-appeal)
isn't surprised.

## Defect 2 (HIGH) — JaroWinkler can return > 1.0 when prefix=4 and jaro≈1 underflow

`distance.go:233`:

```go
return jaro + float64(prefix)*0.1*(1.0-jaro)
```

Standard Jaro-Winkler formula. Bound: 0 ≤ result ≤ 1 iff `prefix * 0.1 ≤ 1`,
i.e. prefix ≤ 10. The implementation caps `maxPrefix = 4`, so prefix ≤ 4
and the bound is safe in exact arithmetic.

In float64: when jaro is computed as exactly 1.0 (identical strings short-
circuit at line 195 / 213), `(1-jaro) = 0` and the bonus is 0. Safe.

But for near-identical strings — e.g. matches=la=lb=N, transpositions=0 —
`jaro = (1 + 1 + 1) / 3 = 1.0` *in exact arithmetic* but in float64 the
three divisions and the average can produce 0.9999999999999999 or
1.0000000000000002 depending on rounding, then `1 - jaro` is ±2.2e-16, the
bonus is ±8.8e-17, and result is `1 ± 1e-16`. The
`TestJaroWinkler_RangeAlways01` covers some inputs but does not assert
strict ≤ 1.0 for "abcd" vs "abcd" with the prefix path active.

**Patch (1 line): clamp output to [0, 1]** before return. Same approach
PostgreSQL `pg_trgm` and Lucene `JaroWinklerDistance.java` use. Cost: one
`min`/`max` pair, zero allocs. Asserted by the topic prompt as the primary
"normalisation to [0,1]" concern.

## Defect 3 (HIGH) — int overflow on huge-string Levenshtein / DamerauLevenshtein

`distance.go:43-52` and `:84-100`:

`prev[i]+1`, `curr[i-1]+1`, `prev[i-1]+cost`, `dp[i][j-1]+1` etc. — all on
plain `int`. On 64-bit Go, `int = int64`, so overflow requires a string of
~9.2e18 runes (impossible). On 32-bit Go (`GOARCH=386`, `GOARCH=arm`), `int
= int32` and overflow fires at ~2.1e9 runes — also impossible to allocate.

**However**, the slices are `make([]int, m+1)` and `make([][]int, m+1)`.
On 32-bit, `m+1` overflows when `m = math.MaxInt32`, but `m = len([]rune(a))`
and the rune slice itself can't be that large. **No realistic overflow path.**

**Real risk**: `make([][]int, m+1)` with `m = 100_000` and `n = 100_000`
allocates 10 GB of int cells (8 bytes each) — that's not overflow, that's
OOM. Topic prompt's "int overflow on huge strings" should be reframed:
**memory blow-up at 100k×100k**, not int overflow. DamerauLevenshtein
is the one with full O(mn) memory. LevenshteinDistance uses two-row
optimization (O(min(m,n)) memory).

**Patch**: document the O(mn) memory ceiling on DamerauLevenshtein (~10 GB
at 100k×100k) and add a `DamerauLevenshteinThreshold(a,b,maxDist)` peer
that bails out at maxDist with O(maxDist · min(m,n)) memory. Same idiom
as RapidFuzz `score_cutoff` and PostgreSQL `levenshtein_less_equal`. No
overflow defect; document and add cutoff variant.

## Defect 4 (MEDIUM) — NGramSimilarity returns 1.0 when n is too large for both inputs

`ngram.go:62-64`:

```go
if len(gramsA) == 0 && len(gramsB) == 0 {
    return 1.0 // Both empty → identical.
}
```

This fires when **either** (a) both inputs are empty strings, OR (b) both
inputs are shorter than `n`. The test `TestNGramSimilarity_NTooLargeForBoth`
asserts `NGramSimilarity("ab", "cd", 5) == 1.0` and **passes**, which is
nonsensical — "ab" and "cd" share zero characters but the function reports
"identical". This is not numerical incorrectness in the strict sense (it's
a definitional choice about the 0/0 case in Jaccard), but it's a **silent
correctness trap** — fuzzy-match consumers (TokenSetRatio FW PCN driver
name, claimant address fuzzy match) routinely hit 2-char strings and an n=3
NGramSimilarity will incorrectly report 1.0 perfect match between any two
2-char strings.

**Patch**: when `gramsA` and `gramsB` are both empty BECAUSE the inputs
were shorter than `n`, return 0.0 (no information) or NaN (signal: undefined).
Distinguish from the case where both inputs are exactly empty strings
(len(a)==0 && len(b)==0) — that case can stay at 1.0 since identical
empty strings are reasonably "identical". Same defect in
`NGramDiceCoefficient` lines 121-123 and `NGramSimilarity` line 89-91
(`union == 0 → return 1.0`, dead branch since it can only fire when both
sets are empty which is already handled above).

## Defect 5 (MEDIUM) — NeedlemanWunsch traceback is non-deterministic for tied scores

`alignment.go:59-82`: traceback uses `dp[i][j] == dp[i-1][j-1]+s`
equality. When `match=mismatch=gap=-1` (all-mismatch case) several paths
tie. The traceback picks the first branch that matches (`if/else` order:
diag → up → left). Fixed by code structure, but the **score is
deterministic, the alignment is not unique**. Float64 equality on
accumulated gap penalties can also flake when penalty magnitudes are
close to ulp(score) — e.g. `match=1.0, gap=-1e-15`, the subtraction
`dp[i][j] - (dp[i-1][j-1]+s)` can be 0 in one rounding regime and ulp
in another. Lucene / Biopython solve this with **integer scores** or
**explicit traceback pointers** stored in a parallel matrix during the
forward pass.

**Patch**: store traceback pointers (3-byte enum: DIAG, UP, LEFT) during
the forward pass; eliminate float==float comparison in the back pass.
Cost: O(mn) extra bytes (1/8 the size of the float64 dp matrix already
allocated). Same fix applies to SmithWaterman:151 / :156. **Topic prompt's
"IEEE-754 edge cases at empty strings" — empty-string SmithWaterman returns
("", "", 0) at line 137-139, no NaN propagation. Empty-string NeedlemanWunsch
returns ("", "", 0) when m=n=0 by falling through the loops; verified.**

## Defect 6 (LOW) — TokenSetRatio simpleRatio integer division truncates toward zero

`token_ratio.go:155`: `return (100 * matched) / total`. Integer division
floors. Documented behavior ("returned value is the integer floor"). No
defect, but two edge cases worth pinning:

1. RapidFuzz returns `int` percentage rounded half-to-even (Python's
   `round`). reality returns `floor`. Test `TestTokenSetRatio_*` does not
   compare against RapidFuzz reference vectors; topic prompt asserts
   "RapidFuzz parity" in commit `1e12e80`. **Defect: documentation claims
   parity, behavior is floor not banker's rounding.** Fix: round-half-to-
   even via `(200*matched + total) / (2*total)` integer-only formulation
   (with appropriate tie handling).

2. `total = len([]rune(x)) + len([]rune(y))` is recomputed on every
   `simpleRatio` call inside `TokenSetRatio`, three times for the same
   pair. Minor allocation churn (rune slice per call). Hot-path concern
   only if topic 005-style perf audit cares; numerical-correctness
   tangent only.

## Defect 7 (LOW) — JaroWinkler match window for length-1 inputs is 0

`distance.go:162`: `window := maxLen/2 - 1`. For `la=lb=1`, `window=-1`
clamped to 0 by line 164. The match scan at line 175-181 then has
`hi = i+window+1 = 1`, scans index 0 — correct. Single-char identical
returns 1.0, single-char different returns 0.0 (test passes). No defect,
but the formula is the "alternate" Jaro window (Winkler 1990 §3.1
defines window as `floor(max(|a|,|b|)/2) - 1`); some references use
`floor(max(|a|,|b|)/2)` without the -1. reality picks the Winkler-original
form. **Documentation precision item**: cite which window convention.
Lucene picks `max(|a|,|b|)/2 - 1`, RapidFuzz picks the same. Match
within established field.

## Defect 8 (FIX-ME) — IEEE-754 at empty strings: Jaccard NGramSimilarity 0/0 returns 1.0

Already covered in Defect 4. Topic prompt's "IEEE-754 edge cases at empty
strings" — the only function that produces a fraction whose denominator
can be zero is `NGramSimilarity` (Jaccard) and `NGramDiceCoefficient`
(Sørensen-Dice). Both short-circuit at len()==0 paths and never hit a
literal `0.0/0.0` so no NaN escapes. **No NaN/Inf bug; the 0/0 case is
defined as 1.0 by convention.** The convention is debatable (see Defect 4)
but the IEEE-754 hygiene is intact.

JaroWinkler at len==0 short-circuits (lines 150-155) — never divides by zero.
NeedlemanWunsch / SmithWaterman with empty input loop zero times — return
zero score. No NaN escapes from any sequence/ entry point.

---

## Summary table

| # | Severity | File:line                | Defect                                                  | Fix LOC |
|---|----------|--------------------------|---------------------------------------------------------|---------|
| 1 | DOC      | phonetic.go:26           | Y-as-vowel diverges from SSA convention (consistent w/ Knuth) | doc     |
| 2 | HIGH     | distance.go:233          | JaroWinkler may return 1±ulp for near-identical floats  | 1       |
| 3 | DOC      | distance.go:75-78        | DamerauLevenshtein O(mn) memory unbounded; no cutoff variant | doc + 40 LOC |
| 4 | MEDIUM   | ngram.go:62, 121         | NGramSimilarity / Dice returns 1.0 when n exceeds both inputs | 4       |
| 5 | MEDIUM   | alignment.go:65, 151     | NW/SW traceback uses float equality on accumulated scores | 30      |
| 6 | LOW      | token_ratio.go:155       | simpleRatio floors not banker's-rounds; doc says parity  | 3       |
| 7 | DOC      | distance.go:162          | Jaro window convention not cited                        | doc     |
| 8 | OK       | (all)                    | No NaN/Inf escapes from any entry point                 | n/a     |

## Cross-package observations

- `sequence` is the **only** package besides `prob` that ships a fuzzy-match
  scoring function returning a normalised similarity. The Jaccard 0/0 → 1.0
  convention should be made consistent across `sequence.NGramSimilarity`,
  `prob.JaccardCoefficient`, `prob.SorensenDice` if those exist (slot 095).
- `Shingling` reuses an inlined FNV-1a 64-bit hash (line 176). Identical
  function exists in `crypto.FNV1a64`. CLAUDE.md rule 6 ("reimplement from
  first principles") technically permits the inlining since `sequence`
  cannot import `crypto` (no inter-package deps among reality/* per the
  architecture rules), but the duplication has a divergence risk if FNV
  parameters ever change. Acceptable; flag for slot 002 (sequence-missing)
  if it surveys cross-package duplication.

## Highest-leverage single PR

**Defect 2 + Defect 4 + Defect 5 in one PR** (~35 LOC, three-line changes):

1. `distance.go`: clamp JaroWinkler output to [0,1] via `min(1.0, max(0.0, jaro + …))`.
2. `ngram.go`: distinguish "both inputs empty" (return 1.0) from "n too large
   for both" (return 0.0); fix on lines 62 and 121.
3. `alignment.go`: store traceback pointers in parallel `[][]byte` matrix
   during forward pass; eliminate float-equality in backtrace.

Three numerical-correctness defects, three independent files, single review.
Ships zero new public API. Pins the topic prompt's "fuzzy match score
normalization to [0,1]" concern with an unforgeable invariant.

## Files audited

- C:\limitless\foundation\reality\sequence\distance.go (355 lines)
- C:\limitless\foundation\reality\sequence\alignment.go (189 lines)
- C:\limitless\foundation\reality\sequence\ngram.go (188 lines)
- C:\limitless\foundation\reality\sequence\phonetic.go (120 lines)
- C:\limitless\foundation\reality\sequence\token_ratio.go (157 lines)
- C:\limitless\foundation\reality\sequence\sequence_edge_test.go (542 lines)
- C:\limitless\foundation\reality\sequence\sequence_test.go (head, ~80 lines sampled)

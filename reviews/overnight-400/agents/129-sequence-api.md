# 129 — sequence: byte vs codepoint vs grapheme contract clarity

Scope: `C:\limitless\foundation\reality\sequence\` (distance.go, alignment.go,
ngram.go, phonetic.go, token_ratio.go) viewed as an **API surface**.
Question: when a caller hands `LevenshteinDistance("café", "cafe")` to this
package, *what counts as one character*? When they hand it `"é"` written
two different ways (NFC `U+00E9` vs NFD `U+0065 U+0301`), do they get the
same answer? When they hand it `"👨‍👩‍👧"` (one perceived emoji, three
codepoints, one ZWJ), how is it tokenised? **All sixteen exported funcs in
the package treat "character" as `rune` (Go codepoint), no normalisation, no
grapheme-cluster awareness, no locale, no case-folding.** Some of those are
right defaults; some are foot-guns; one (the byte/rune/string/[]rune
threading) is straight-up inconsistent across siblings. Detail below.

Distinct from 126 (numerical defects), 127 (missing primitives), 128 (SOTA
library comparison). This slot: **does the API tell the caller what unit
of comparison it operates on, and does the package make it possible to opt
into the unit they actually want?**

---

## A — The five contract questions, answered for each entry point

| Function                  | Unit fed to algorithm | Empty-input contract                   | Case-fold? | Normalisation? | Grapheme? | Locale? |
|---------------------------|-----------------------|----------------------------------------|------------|----------------|-----------|---------|
| `LevenshteinDistance`     | `rune` (codepoint)    | `Lev("","")=0`, `Lev("a","")=1`        | no         | no             | no        | no      |
| `DamerauLevenshtein`      | `rune`                | same as above                          | no         | no             | no        | no      |
| `HammingDistance`         | `rune`                | empty+empty → `(0,nil)`; mismatch err  | no         | no             | no        | no      |
| `JaroWinkler`             | `rune`                | both empty → `1.0`; one empty → `0.0`  | no         | no             | no        | no      |
| `LongestCommonSubsequence`| `rune`                | empty in either → `""`                 | no         | no             | no        | no      |
| `LongestCommonSubstring`  | `rune`                | empty in either → `""`                 | no         | no             | no        | no      |
| `NGrams`                  | `rune`, slid by 1     | `len<n` → `nil`                        | no         | no             | no        | no      |
| `WordNGrams`              | `[]string` (caller)   | `len<n` → `nil`                        | no         | no             | no        | no      |
| `NGramSimilarity`         | rune-grams            | both empty → `1.0`; one → `0.0`        | no         | no             | no        | no      |
| `NGramDiceCoefficient`    | rune-grams            | both empty → `1.0`; one → `0.0`        | no         | no             | no        | no      |
| `Shingling`               | rune-grams → bytes    | `len<k` → `nil`                        | no         | no             | no        | no      |
| `NeedlemanWunsch`         | `rune`                | both empty → `("","",0)`               | no         | no             | no        | no      |
| `SmithWaterman`           | `rune`                | empty → `("","",0)`                    | no         | no             | no        | no      |
| `Soundex`                 | `byte` (ASCII A-Z)    | empty → `""`; non-ASCII first → `""`   | YES (ASCII)| no             | no        | EN-only |
| `TokenSetRatio`           | rune via Levenshtein  | both empty → `100`; one → `0`          | YES (ASCII)| no             | no        | no      |

Five facts jump off this table:

1. **Every Unicode-aware function silently does `[]rune(s)` at the top.**
   13 of 16 entry points open with the same line:
   `ra, rb := []rune(a), []rune(b)`. None says so in its docstring. Not
   `LevenshteinDistance`, not `JaroWinkler`, not `LCS`, not `NW`/`SW`. The
   *only* function whose doc actually states its unit is `Soundex`
   ("ASCII letter ... non-alphabetic runes elsewhere ... skipped").

2. **One function fights the others.** `Shingling` works on `rune`s for the
   sliding window, then converts each shingle back to `string`, then to
   `[]byte`, and FNV-1a hashes the **bytes**. So a `k`-character shingle
   over a Japanese string hashes 3k bytes per shingle, but a `k`-character
   shingle over Latin-1 hashes k bytes per shingle. Two `Shingling("café",
   3)` calls on NFC-`é` vs NFD-`é` produce **different shingle counts**
   (4 vs 5) **and different hashes**, with zero way for the caller to know.

3. **Case-folding is silent and ASCII-only where it happens at all.**
   Soundex case-folds `'a'-'z'` to `'A'-'Z'` via the inline 32-bit subtract
   trick (phonetic.go:53-54, 64-65). Anything else (`İ`→`i̇`,
   `ß`→`ss`, `Σ`→`σ` vs final-`ς`) passes the `r < 'A' || r > 'Z'` filter
   and is **dropped silently**. TokenSetRatio uses `strings.ToLower`
   (token_ratio.go:79) which IS Unicode-aware — so the two case-folding
   functions in the same package use **two different definitions of
   "lowercase"**. Caller sees no signal that this matters.

4. **`HammingDistance` errors on rune-length mismatch but the user wrote
   byte-length code.** Idiomatic Go is `len(s)` returning bytes; this
   function returns an error when `len([]rune(a)) != len([]rune(b))`. So
   `HammingDistance("café", "cafe1")` fails with "requires equal-length
   strings" because café is 4 runes vs 5, but the user wrote two strings
   that look the same byte-length (5 each: `c-a-f-é(2)` vs `c-a-f-e-1`).
   Error message does not say "rune-length"; it says "length", which
   matches Go's default unit *of* `len()`. Caller is misled.

5. **`Soundex` accepts any rune as input but only succeeds on
   `[A-Za-z]` first letter** — not "first ASCII letter": the doc says
   "non-alphabetic runes elsewhere ... are skipped" (phonetic.go:13-14),
   which suggests `Sōundex("Müller")` would skip the umlaut and return
   `M460`. It does NOT — it returns `M460` only by accident, because
   `'ü'` (U+00FC) fails the `'A' <= r <= 'Z'` test on the *non-first*
   path. But `Soundex("Übermensch")` returns `""` because `'Ü'` fails
   the *first*-letter ASCII test. Doc does not call out the asymmetry.
   Real-world German names → silently empty.

---

## B — `string` vs `[]rune` vs `[]byte`: the three-API problem

The package exports **`string`-only** APIs.  No `[]rune` overload, no
`[]byte` overload, no `iter.Seq[rune]` (Go 1.23+). That's a defensible
choice (`strings.IndexByte`, `strings.EqualFold` are also `string`-only),
**but** the package then *internally* converts every `string` to `[]rune`
with `[]rune(s)`. This costs:

- **Allocation.** `[]rune("hello world this is twenty bytes")` allocates a
  20-rune (160-byte) heap buffer. Every Levenshtein call. Pistachio at
  60 FPS doing 10⁴ string distances (e.g. autocomplete fuzzy match) → 60
  × 10⁴ × 2 = **1.2M heap allocs/sec** for two `[]rune` slices per call,
  ~16 MB/s GC pressure, ~12% of frame budget. CLAUDE.md rule 3 ("no
  allocations in hot paths") is silently broken by every entry point in
  the package.

- **Wasted work for ASCII.** Two strings that are guaranteed ASCII (e.g.
  `LevenshteinDistance(userInput, dbColumn)` after upstream validation)
  still pay the rune-decode + heap copy. RapidFuzz's two-tier dispatch
  (ASCII fast path with `[]byte` indexing, Unicode fallback with
  codepoint-aware DP) is missing. `strings.IndexByte` is the canonical
  Go idiom for the ASCII fast path.

**Recommendation A** — overload pattern that matches the Go stdlib's
`strings`/`bytes`/`unicode/utf8` triad:

```go
// Canonical: rune-by-rune, allocates []rune. Documented as such.
func LevenshteinDistance(a, b string) int { ... }

// Pre-decoded: zero-alloc, caller pays rune-decode cost once.
func LevenshteinDistanceRunes(a, b []rune) int { ... }

// ASCII fast path: zero-alloc, caller asserts ASCII (returns -1 if not).
func LevenshteinDistanceBytes(a, b []byte) int { ... }
```

Applies to the seven entry points actually used in hot paths:
`LevenshteinDistance`, `DamerauLevenshtein`, `HammingDistance`,
`JaroWinkler`, `NGrams`, `NGramSimilarity`, `Shingling`. The other nine
are either trivially fast (`Soundex`, ~4 char loop) or already
allocation-bound (`LCS`/`LCSubstr`/`NW`/`SW` need a 2D `int` matrix that
dominates the rune-decode).

**Recommendation B** — a package-level docstring (currently 4 lines in
distance.go:1-7) that states the contract explicitly:

```
The unit of comparison throughout this package is the Go rune (Unicode
codepoint). Strings are decoded to []rune at function entry; ALL distance
metrics, alignment scores, and n-gram windows operate on codepoints.

Strings are NOT normalised. NFC "é" (U+00E9) and NFD "é" (U+0065 U+0301)
are TWO different things to LevenshteinDistance. Pre-normalise via
golang.org/x/text/unicode/norm if cross-form match is required.

Grapheme clusters (e.g. "👨‍👩‍👧", flag emoji, Devanagari conjuncts) are
NOT respected. NGrams("👨‍👩‍👧", 1) returns 5 codepoints, not 1 grapheme.

Case-folding is NOT performed. Use strings.ToLower / strings.ToUpper at
the call site if case-insensitive comparison is desired.

Locale-aware comparison is NOT supported. Turkish I/i, German ß/SS,
Greek final sigma, etc. are compared as raw codepoints.

EXCEPTIONS:
  - Soundex performs ASCII-only case-folding (a-z → A-Z) and skips all
    non-ASCII letters in the trailing position.
  - TokenSetRatio performs Unicode case-folding via strings.ToLower
    before tokenisation.
```

This is ~18 lines, fits in `doc.go`, and answers every question a caller
will ask. Currently they answer those questions by reading the source.

---

## C — Normalisation: should the package do it, or refuse to?

Three viable contracts; the package currently lands on the worst of them:

| Contract            | Behaviour                                                    | Pro                                                 | Con                                              |
|---------------------|--------------------------------------------------------------|-----------------------------------------------------|--------------------------------------------------|
| **C1 — Always NFC** | Internally do `norm.NFC.String(a)` before `[]rune`           | Caller never thinks about it; matches user intent   | Pulls `golang.org/x/text` (CLAUDE.md violation)  |
| **C2 — Refuse**     | Document loudly, do nothing, caller must pre-normalise       | Zero deps; zero perf cost; matches stdlib `strings`| Naive callers will silently get wrong answers    |
| **C3 — Current**    | Do nothing, document nothing, hope caller knows              | Minimum LOC                                         | Worst of both — silent wrong answers, zero docs  |

**Recommendation C** — C2 is correct for `reality`. The doc snippet in
Recommendation B above is the C2 contract. CLAUDE.md rule 2 ("zero
dependencies") rules out C1; the only honourable move is to be explicit.

A two-line addition to each metric's docstring is enough:

```go
// LevenshteinDistance computes [...]
// Operates on Go runes (codepoints). Strings are NOT Unicode-normalised;
// NFC "é" and NFD "é" are TWO codepoints apart by this metric.
```

Note: a stdlib-only NFC quick-check (`norm.IsNormalString(NFC, s)`
equivalent) is **possible** in pure Go without `x/text` — the
Quick_Check property table is ~3 KB and could ship as a generated
constant table — and the package COULD `panic`/return an error on
non-normalised input as a CLAUDE.md-compatible defensive check. This is
~250 LOC to ship and is the single most caller-protective design move
available within the zero-dep constraint. Out of scope for this slot
but flagged for 127's "missing primitives" follow-up.

---

## D — Grapheme clusters: explicitly out of scope, document so

`LevenshteinDistance("👨‍👩‍👧", "👨‍👩‍👦") == 1`? **No** — it returns 1
because the two emoji ZWJ sequences differ in their last codepoint
(`👧` vs `👦`), and that's *one rune* of difference. But
`LevenshteinDistance("👨‍👩‍👧", "")` returns **5**, not 1, because the
ZWJ sequence is 5 codepoints. A naive caller who thinks "delete one
emoji costs 1" gets a wildly wrong answer.

This is correct behaviour for a codepoint-level metric. It is wildly
wrong for what the user usually means by "edit distance between two
emoji-bearing strings". Grapheme-cluster-aware Levenshtein is a real
algorithm (segment by `unicode.GraphemeBoundary` ≈ UAX-29, run DP over
grapheme tokens). It requires the UAX-29 boundary table (~6 KB
generated tables, ~150 LOC of segmenter logic). Within zero-dep, it's
shippable as a generated table.

**Recommendation D** — flag in 127's "missing primitives" tier-2 list:

- `GraphemeLevenshteinDistance(a, b string) int` — segments via UAX-29,
  DP over grapheme clusters, ~250 LOC including generated tables.
- `Graphemes(s string) []string` — public segmenter (utility).
- `GraphemeNGrams(s string, n int) []string` — n-gram over graphemes.

Default `LevenshteinDistance` stays codepoint. Users who need
grapheme-aware get the explicit `Grapheme*` variant. This mirrors
Swift's `String` (graphemes by default) vs Go's `string` (bytes by
default) explicitly: reality picks codepoints as the middle ground and
exposes both extremes via named functions.

---

## E — Case sensitivity: ad-hoc, asymmetric, undocumented

Three of sixteen functions touch case:

- `Soundex` — ASCII-only case-fold (`a-z` → `A-Z`, U+00C0 onward
  ignored). Documented ("Input is case-insensitive").
- `TokenSetRatio` — Unicode-aware via `strings.ToLower`. **Not
  documented** — caller learns by reading line 79.
- All others — case-sensitive. **Not documented** — caller learns by
  observing `LevenshteinDistance("Hello", "hello") == 1`.

**Recommendation E** — pick one of two contracts:

| Option              | Behaviour                                              | Cost                              |
|---------------------|--------------------------------------------------------|-----------------------------------|
| **E1 — Strict**     | All case-sensitive; provide `*Fold` variants on demand | ~5 lines per `*Fold` wrapper      |
| **E2 — Pair-shed**  | Default = case-sensitive; doc `strings.ToLower` recipe | 2-line doc note per metric        |

E2 is the strict-correct stdlib idiom (`strings.Contains` is
case-sensitive; `strings.EqualFold` is the named-fold version).
Recommend E2 for distance metrics; remove TokenSetRatio's hidden
case-fold and force the caller to opt in via
`TokenSetRatio(strings.ToLower(a), strings.ToLower(b))` — OR rename it
`TokenSetRatioFold` to make the case-fold visible. `Soundex` keeps its
fold (intrinsic to the algorithm).

---

## F — Locale: don't bother, but say so

Locale-aware string comparison (Turkish dotted/dotless I, German ß
expansion, ICU collation tables) needs ICU or `x/text/collate`. Both
are excluded by CLAUDE.md zero-dep. The right contract is:

> "All comparisons are codepoint-by-codepoint; locale-aware comparison
> (Turkish I, German sharp S, Greek final sigma, ICU collation) is NOT
> supported. For locale-correct comparison, pre-fold input via
> `golang.org/x/text/cases` or ICU at the call site."

One line in `doc.go`. Cost: zero. Caller-protection: total.

---

## G — Comparison with Go stdlib `strings` idioms

Concrete divergences from `strings` that bite users:

| Behaviour              | stdlib `strings`                       | reality `sequence`                                 | User expectation              |
|------------------------|----------------------------------------|----------------------------------------------------|-------------------------------|
| Unit of length         | bytes (`len(s)`)                       | runes (silent `[]rune` cast)                       | matches function name         |
| Empty-string identity  | `Contains("","") == true`              | `Lev("","") == 0`, `JaroWinkler("","") == 1.0`     | matches stdlib                |
| Case-folding default   | case-sensitive                         | mixed (Soundex/TSR yes, others no)                 | NEEDS DOCS                    |
| Iteration              | `range s` → byte index, rune value     | hidden `[]rune(s)` at top of every func            | matches stdlib if documented  |
| Allocation profile     | zero-alloc on hot paths                | always allocates `[]rune`                          | violates CLAUDE.md rule 3     |
| Error reporting        | none (boolean / int return)            | `HammingDistance` returns `error`                  | non-idiomatic for length err  |

**The single largest stdlib-idiom violation is allocation.** `strings`
package guarantees most operations zero-alloc; `sequence` allocates two
`[]rune` slices per call, every call. Recommendation A's `*Runes` /
`*Bytes` overloads close this gap.

**The single most surprising stdlib-idiom divergence is
`HammingDistance` returning an error.** `strings.EqualFold` returns
`bool`; `strings.Compare` returns `int`. The Go convention for "two
strings of unequal length" is either `panic` (`subtle.ConstantTimeCompare`)
or silent wrong-answer (`bytes.Equal` returns `false`). Returning an
`error` for a pure mathematical operation is a code smell.

**Recommendation G1** — drop the error from `HammingDistance`, return
`-1` for unequal lengths (the canonical "not applicable" sentinel for
distance metrics that can otherwise never be negative). Or — better —
panic with `"sequence: HammingDistance requires equal-length strings"`
since this is a programmer error, not a runtime error: the caller knew
or should have known. (Compare: `linalg.Dot` of mis-shaped vectors
should panic, not return `(0, error)`.)

**Recommendation G2** — match stdlib's "first arg is the haystack,
second is the needle" convention. Currently `LevenshteinDistance(a, b)`
is symmetric so it doesn't matter, but `LongestCommonSubstring(a, b)`
and `NeedlemanWunsch` traceback are NOT symmetric in their tie-breaking.
Document which arg "wins" ties. (Sibling: `strings.HasPrefix(s, prefix)`
is unambiguously labelled.)

---

## H — Concrete API-clarity TODO list, ranked by leverage

| # | Action                                                                            | LOC  | Risk | Caller-protection |
|---|-----------------------------------------------------------------------------------|------|------|-------------------|
| 1 | Add `doc.go` with the contract paragraph from §B                                  | ~30  | nil  | very high         |
| 2 | Add 2-line "operates on runes; not normalised" docstring to each metric           | ~30  | nil  | high              |
| 3 | Drop `error` from `HammingDistance`; return `-1` (or panic) for length mismatch   | ~10  | low* | high (idiom fix)  |
| 4 | Rename `TokenSetRatio` → `TokenSetRatioFold` to surface the hidden case-fold      | ~3   | med* | high              |
| 5 | Add `*Runes` / `*Bytes` overloads for the 7 hot-path entry points                 | ~100 | low  | medium (perf)     |
| 6 | Add `GraphemeLevenshteinDistance` + UAX-29 segmenter (defer to 127 tier-2)        | ~250 | med  | high              |
| 7 | Add NFC/NFD quick-check primitives (refuse non-normalised input?)                 | ~250 | med  | medium            |

\* breaking changes — `reality` is v0.10, pre-1.0; CLAUDE.md does not
forbid breaking changes. Bundle into a single v0.11 minor bump.

Items 1-2 cost ~60 LOC and zero risk. Item 3 is one-line breaking but
ergonomically correct. Items 4-5 are ~100 LOC and clean up the largest
hidden-case-fold and hidden-allocation foot-guns. Items 6-7 are
follow-on milestones; 6 in particular is the single biggest "this
package now correctly handles modern emoji input" upgrade and should
ship before any aicore consumer ingests user-typed text.

---

## TL;DR

The package treats every string as `[]rune` internally, performs no
normalisation, ignores graphemes, and case-folds in two of sixteen
functions using two different definitions of lowercase. **None of this
is in the docstrings.** The single highest-leverage commit is a 30-line
`doc.go` stating the contract; the second-highest is matching Go
stdlib `strings` idioms (no error returns from pure math, optional
`*Runes`/`*Bytes` overloads to satisfy CLAUDE.md zero-alloc rule on hot
paths). Grapheme-aware variants are a real-Unicode-input requirement
and belong in 127's tier-2 missing-primitives list.

Source files audited:
- `C:\limitless\foundation\reality\sequence\distance.go` (16 funcs, 355 lines)
- `C:\limitless\foundation\reality\sequence\alignment.go` (2 funcs, 189 lines)
- `C:\limitless\foundation\reality\sequence\ngram.go` (5 funcs, 188 lines)
- `C:\limitless\foundation\reality\sequence\phonetic.go` (2 funcs, 120 lines)
- `C:\limitless\foundation\reality\sequence\token_ratio.go` (5 funcs, 157 lines)
- `C:\limitless\foundation\reality\sequence\sequence_edge_test.go` (Unicode tests
  cover only `café` vs `cafe` Latin-1 case + Han ideograph identity; no NFC/NFD
  pair, no emoji ZWJ, no Turkish-I, no combining mark)

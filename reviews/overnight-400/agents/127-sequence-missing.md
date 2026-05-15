# 127 — sequence: missing primitives

Scope: enumerate canonical sequence primitives ABSENT from
`C:\limitless\foundation\reality\sequence\`. Current entry points (verified
grep against `^func [A-Z]`):

```
LevenshteinDistance, DamerauLevenshtein, HammingDistance, JaroWinkler,
LongestCommonSubsequence, LongestCommonSubstring, NGrams, WordNGrams,
NGramSimilarity, NGramDiceCoefficient, Shingling, NeedlemanWunsch,
SmithWaterman, Soundex, TokenSetRatio
```

15 entry points (the master-plan brief said "~14" and named Metaphone +
RatcliffObershelp — neither is implemented; brief was inaccurate).

The package as shipped is a fuzzy-string-match cheat-sheet. Everything
substring-search, everything string-index, everything multi-pattern,
everything compressed full-text, and everything bioinformatics-grade is
missing. Below: three tiers ordered by canonicalness × downstream use
inside reality consumers (Pistachio fuzzy lookups, aicore embeddings,
limitless command parsers).

---

## Tier 1 — canonical, zero excuses to be absent

These are textbook (Gusfield, Cormen, Crochemore) and ship in every serious
string library (Python `regex`, Java Lucene, Rust `aho-corasick`, C++ Boost).

### 1.1 Hirschberg LCS — O(n) space LCS reconstruction
Current `LongestCommonSubsequence` builds full O(mn) `[][]int` dp matrix
(`distance.go` per the 126 audit, line range ~270-310). Hirschberg 1975
divides-and-conquers to O(min(m,n)) space, same O(mn) time. Critical
when consumers diff 100k-rune transcripts (aicore embedding diff). Pair
with linear-space `LevenshteinDistance` traceback (currently distance.go
returns int, not the alignment).

**API:** `HirschbergLCS(a, b string) string` (drop-in for
`LongestCommonSubsequence`) and `LevenshteinAlignment(a, b string) (string,
string)` (returns the two aligned strings with `-` gaps, like NW but with
unit edit costs).

**Why now:** `LongestCommonSubsequence` doc string promises "O(mn) memory"
which is fine for documentation but not for production. Hirschberg gives
free 100x memory headroom.

### 1.2 Wagner-Fischer with operation traceback
`LevenshteinDistance` returns the *score* but not the *edit script*.
Wagner-Fischer 1974 with backpointers returns `[]EditOp{Insert, Delete,
Substitute, Match}`. Used by every diff-style consumer (text-merge,
auto-correct, OT-like collaborative edit). Lucene `LevenshteinDistance`,
Python `python-Levenshtein.editops`, Rust `strsim` all ship this.

**API:** `LevenshteinEdits(a, b string) []EditOp` where `type EditOp
struct { Kind EditKind; SrcIdx, DstIdx int; Rune rune }`.

### 1.3 Z-algorithm (Gusfield §1.4)
Linear-time Z-array for a single string: `Z[i]` = length of longest
substring starting at `i` matching a prefix of the string. Used for
exact pattern matching in O(n+m) and as building block for suffix
structures. **40 LOC**, no allocations in hot path beyond the Z array.

**API:** `ZArray(s string) []int`, `ZSearch(text, pattern string) []int`
returning all occurrence offsets.

### 1.4 KMP (Knuth-Morris-Pratt)
Currently no exact-string-search at all. `strings.Index` exists in stdlib
but reality-style is "ship the algorithm, not a wrapper". KMP failure
function is the canonical pedagogical primitive; constant in every
algorithms textbook.

**API:** `KMPFailure(pattern string) []int`, `KMPSearch(text, pattern
string) []int`. ~50 LOC.

### 1.5 Boyer-Moore-Horspool (BMH)
Faster-than-KMP in practice for English text. The bad-character rule alone
(simpler than full BM) is BMH. Used by GNU `grep`, Perl regex engine.
Same ~50 LOC.

**API:** `BMHSearch(text, pattern string) []int`.

### 1.6 Aho-Corasick — multi-pattern exact matching
Single most-cited "string algorithm not in reality". Builds trie + failure
links once, scans text in O(n + total_matches). Backbone of every
content-filter, every spell-checker dictionary lookup, every fuzzy-keyword
flagger. Used by Pistachio command-parser today (which currently does
linear scan per keyword). **180-250 LOC**.

**API:** `type AhoCorasick struct { ... }`, `BuildAhoCorasick(patterns
[]string) *AhoCorasick`, `(*AhoCorasick).Match(text string) []Match`
where `type Match struct { Start, End int; PatternIdx int }`.

### 1.7 Suffix array (SA-IS or DC3)
The canonical compact full-text index. Sorted array of all suffix start
positions, supports substring search in O(m log n) with O(n) space (vs
suffix tree's O(n) with much larger constant). SA-IS (Nong-Zhang-Chan
2009) builds in linear time. DC3 (Kärkkäinen-Sanders 2003) is the simpler
linear-time variant. **Either: 200-400 LOC.**

**API:** `SuffixArray(s string) []int`, `(SA).Search(pattern string)
(lo, hi int)` (returns range in SA of all suffixes starting with pattern).

Pair with **LCP array** (Kasai 2001, ~30 LOC given SA): enables longest
repeated substring, distinct substring count, and is the bridge to
suffix tree functionality without paying the suffix tree's memory tax.

**API:** `LCPArray(s string, sa []int) []int`.

### 1.8 Burrows-Wheeler Transform (BWT)
The reversible permutation underlying bzip2 and FM-index. Given suffix
array, BWT is `T[SA[i]-1]` for each i (~10 LOC once SA exists). Inverse
BWT via LF-mapping (~30 LOC). Pure permutation — no math precision
concerns. **Topic prompt named explicitly.**

**API:** `BWT(s string) (transformed string, primaryIdx int)`,
`InverseBWT(transformed string, primaryIdx int) string`.

### 1.9 FM-index (Ferragina-Manzini 2000)
Compressed full-text index built from BWT + rank/select dictionary.
Substring count in O(m), locate in O(m + occ·log^ε n). The reason BWT
matters in 2026 is FM-index (Bowtie/BWA short-read aligners, every
bioinformatics pipeline). **~300-500 LOC** including wavelet tree or
Huffman-shaped rank dictionary.

**API:** `BuildFMIndex(s string) *FMIndex`, `(*FMIndex).Count(pattern
string) int`, `(*FMIndex).Locate(pattern string) []int`.

### 1.10 Suffix automaton (Blumer 1985, DAWG)
Smallest DFA recognizing all substrings of `s`. Online linear-time
construction. Equivalent power to suffix tree but with cleaner DP
(distinct substring counting in O(n) with one pass over states).
Competitive-programming canon (Codeforces, ICPC). **~150 LOC**.

**API:** `type SuffixAutomaton struct { ... }`, `BuildSuffixAutomaton(s
string) *SuffixAutomaton`, `(*SuffixAutomaton).Contains(pattern string)
bool`, `(*SuffixAutomaton).DistinctSubstrings() int`.

### 1.11 Bitap / Shift-Or / Wu-Manber
Bit-parallel exact matching (Baeza-Yates 1992) with constant-factor
speedup using machine words. Wu-Manber 1992 extends to k-mismatch
approximate matching in O(nk/w) where w = word size. Backbone of `agrep`
and the algorithm everyone reaches for when "fuzzy substring search" is
the requirement. Constraint: pattern length ≤ 64 (or use multi-word).

**API:** `BitapSearch(text, pattern string) []int`,
`BitapApproximateSearch(text, pattern string, k int) []Match`.

### 1.12 Levenshtein automaton (Schulz-Mihov 2002)
NFA/DFA accepting all strings within edit distance k of a pattern.
The right primitive for spell-checker + dictionary intersection (vs
brute-force O(|dict| · |word|)). Used by Lucene's `FuzzyQuery`. **The**
canonical "fast fuzzy lookup" data structure. ~250 LOC for k ≤ 3.

**API:** `BuildLevenshteinAutomaton(pattern string, k int)
*LevAutomaton`, `(*LevAutomaton).Accepts(s string) bool`.

### 1.13 BK-tree (Burkhard-Keller 1973)
Metric-space index for fuzzy nearest-neighbor over edit distance. Insert
O(log n) avg, query within radius k in sublinear time. The lazy-but-
pragmatic alternative to Levenshtein automata when dictionary is fixed.
Covered explicitly in the topic prompt. **~80 LOC**.

**API:** `type BKTree struct { ... }`, `(*BKTree).Insert(s string)`,
`(*BKTree).Search(query string, maxDist int) []string`.

---

## Tier 2 — high-value, established but more niche

### 2.1 Suffix tree (Ukkonen 1995)
Online linear-time suffix tree. Pedagogically beautiful, practically
displaced by suffix array + LCP for most tasks (smaller constant).
Still the canonical answer for "I want all super-maximal repeats" and
"longest common substring of K strings". **~400 LOC**, the heaviest
single algorithm in this list.

**API:** `BuildSuffixTree(s string) *SuffixTree`, `.Search(pattern
string) []int`, `.LongestRepeat() string`.

Recommendation: **defer behind suffix array + LCP**. Most consumers want
LCP queries which SA+LCP delivers for half the LOC.

### 2.2 Approximate matching with cutoff (RapidFuzz parity)
`LevenshteinThreshold(a, b string, maxDist int) int` returning early
when distance exceeds `maxDist`. O(min(m,n) · maxDist) memory instead
of O(mn). RapidFuzz `score_cutoff`, PostgreSQL `levenshtein_less_equal`,
Python `python-Levenshtein.distance(..., max_dist=k)`. **~30 LOC** delta
over current `LevenshteinDistance`. Already flagged by 126 audit defect 3.

### 2.3 Phonetic family beyond Soundex
Topic prompt named six phonetic codings. Current package has Soundex only.

- **Metaphone** (Philips 1990) — ~150 LOC. The standard improvement over
  Soundex; ships in every fuzzy library. **PROMPT'S 126 BRIEF FALSELY
  CLAIMED METAPHONE EXISTS.** It does not.
- **Double Metaphone** (Philips 2000) — ~400 LOC. Returns up to two codes
  per name (handles e.g. "Schmidt" → SHMT + XMT). The de-facto standard
  for English+European names. PostgreSQL `fuzzystrmatch`, Apache Commons
  Codec, Lucene all ship this.
- **NYSIIS** (Taft 1970) — ~80 LOC. NY State Identification + Intelligence
  System. Better than Soundex for non-English-origin names.
- **Caverphone** (Hood 2002) — ~120 LOC. Designed for NZ accents,
  outperforms Soundex on Maori-origin names.
- **Daitch-Mokotoff Soundex** (1985) — ~250 LOC. Slavic/Yiddish surname
  variant. Returns six-digit code, sometimes multiple per name.
- **Match Rating Approach** (Moore 1977) — ~100 LOC. Includes both a
  codex and a comparator. Used by Western Airlines for passenger
  reservation matching.
- **Cologne Phonetics** (Postel 1969) — ~100 LOC. German equivalent of
  Soundex, far better than Soundex for German names.

Recommendation: ship Metaphone + Double Metaphone + NYSIIS as Tier 2;
defer Caverphone / DM / Match-Rating / Cologne as Tier 3 add-ons.

### 2.4 Ratcliff-Obershelp (Gestalt pattern matching)
Used by Python's `difflib.SequenceMatcher.ratio()`. Recursive longest-
common-substring. Different shape from Jaro-Winkler — useful as an
ensemble feature in fuzzy-match consumers. **~60 LOC.** Topic prompt's
126 brief falsely claimed this exists.

**API:** `RatcliffObershelp(a, b string) float64`.

### 2.5 Edit distance with transpositions — generalized
`DamerauLevenshtein` currently is **restricted Damerau-Levenshtein** —
adjacent transpositions only, no overlapping edits. The full Damerau-
Levenshtein (Lowrance-Wagner 1975) allows arbitrary substring
transpositions and is the version used in computational biology. **~120
LOC** with a different DP recurrence and an alphabet-indexed last-
occurrence table.

**API:** `OptimalStringAlignment(a, b string) int` (current behavior,
under correct name) + `DamerauLevenshteinFull(a, b string) int` (true
unrestricted variant).

### 2.6 Fuzzy ratio family (RapidFuzz parity)
`TokenSetRatio` shipped in 1e12e80. Adjacent primitives in RapidFuzz
that consumers expect alongside it:

- **PartialRatio** — best-matching substring ratio. ~40 LOC.
- **TokenSortRatio** — sort tokens, then ratio. ~30 LOC.
- **WRatio** — weighted combination of above (the one RapidFuzz docs
  recommend as default). ~50 LOC.
- **QRatio** — "quick ratio", same as `simpleRatio` but applied to lower-
  cased input. ~10 LOC.

All four are O(mn) wrappers around Levenshtein/LCS already shipped.

### 2.7 NCD — Normalized Compression Distance (Cilibrasi-Vitanyi 2005)
`NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))`. Universal
similarity metric using any compressor as oracle. Reality already has
`compression/` package (Huffman, LZ77, RLE) — NCD is a 20-LOC bridge
that subsumes most edit-distance use cases for arbitrary byte sequences.
Topic prompt named explicitly. Tier 2 because cross-package imports are
discouraged per CLAUDE.md, but **this is the most consumer-friendly
similarity metric in the list** because it works on bytes not just
strings (binary diff, image fingerprint, etc.).

**API:** `NCD(x, y []byte, compress func([]byte) int) float64` —
takes compression oracle as parameter, avoiding the cross-package
import.

### 2.8 Run-length encoding
Reality has `compression/` with RLE per CLAUDE.md, but **as a sequence
primitive** (decompose `s` into `[]Run{Rune, Count}`) it's a useful
standalone in-place op. ~20 LOC. Sequence-package version operates on
runes, compression-package version on bytes — same algorithm, different
surface.

**API:** `RunLengthEncode(s string) []Run`, `RunLengthDecode(runs
[]Run) string`.

### 2.9 Sequence alignment — affine gap penalties (Gotoh 1982)
Current `NeedlemanWunsch` uses linear gap penalty (each gap costs `gap`).
Real bioinformatics uses **affine gaps**: `cost = open + len*extend`,
modeling that opening a new gap is more costly than extending. Gotoh's
algorithm runs in O(mn) with three DP matrices. Same SmithWaterman
extension. **~120 LOC**.

**API:** `NeedlemanWunschAffine(a, b string, match, mismatch, gapOpen,
gapExtend float64) (string, string, float64)` and same for
`SmithWatermanAffine`.

### 2.10 Pair-HMM alignment
Probabilistic version of NW: alignment as forward-backward through a
3-state HMM (Match, InsertX, InsertY). Used by GATK HaplotypeCaller and
every modern variant caller. **~250 LOC**, depends on logsumexp from
`prob/`. **The right primitive for "what's the probability these two
sequences are related".** Tier 2 not Tier 1 only because the consumer
list is narrower (bioinformatics, not general fuzzy-match).

### 2.11 Profile-HMM (Krogh 1994)
Hidden Markov model with position-specific match/insert/delete
probabilities. The data structure underneath HMMER, Pfam, SAM. **~400
LOC** including Baum-Welch training and Viterbi alignment. Tier 3
candidate; listing in Tier 2 because it's the canonical "model a family
of sequences" primitive and reality already has `prob/` for the
probability backbone.

### 2.12 BLAST primitives
- **Karlin-Altschul statistics** — E-value computation for local
  alignment scores. ~80 LOC, depends on `prob/` exponential
  distribution. **The single most-cited equation in bioinformatics.**
- **Seed-and-extend** — find k-mer hits, extend ungapped, then with
  gaps. ~150 LOC; depends on suffix array / hash index. Sketch level
  for now.
- **PAM / BLOSUM substitution matrices** — pure data tables, ~60 LOC
  for BLOSUM62 + lookup helpers.

Recommendation: ship Karlin-Altschul + BLOSUM62 lookup as Tier 2; defer
full seed-and-extend pipeline.

---

## Tier 3 — niche, defer

### 3.1 Generalized suffix tree / array (multiple strings)
Useful for "longest common substring of K documents". Layer over
single-string SA via separator runes outside Unicode range. ~50 LOC
on top of SA. Tier 3 because consumers don't ask for it.

### 3.2 q-gram distance (Ukkonen 1992)
Sum of absolute differences of n-gram histograms. Fast lower bound on
Levenshtein distance, used as filter step in Lucene's FuzzyQuery.
Reality already has NGramSimilarity (Jaccard) and NGramDiceCoefficient;
q-gram distance is the L1 variant. ~30 LOC.

### 3.3 Cosine similarity over token vectors
TF-IDF style. Belongs in `prob/` or a future `nlp/` package, not
`sequence/`. Cited only because adjacent fuzzy-match libraries
(scikit-learn, RapidFuzz) ship it.

### 3.4 Smith-Waterman with banded DP
For when you know the alignment lies near the diagonal. ~50 LOC delta
on existing SW. Tier 3 because the speedup matters only at length 10k+.

### 3.5 Myers' bit-parallel edit distance (Myers 1999)
O(n · ceil(m/w)) Levenshtein using machine-word bit operations. ~150
LOC. Faster than Wagner-Fischer for short patterns. Used by `agrep`
and recent Levenshtein implementations.

### 3.6 Locality-sensitive hashing for sequences
MinHash (Broder 1997) over shingles for Jaccard estimation; SimHash
(Charikar 2002) over weighted features for cosine. Reality has
`Shingling` returning hashes — MinHash is a 30-LOC fold. Tier 3 because
consumers can build it themselves over `Shingling()`.

### 3.7 Longest increasing subsequence
O(n log n) patience sort. Pedagogical staple. Not specific to text.
~40 LOC. Tier 3 because the consumer audience is competitive
programming, not Pistachio.

### 3.8 De Bruijn graphs
Bioinformatics assembly primitive. ~150 LOC. Defer unless reality grows
a `bio/` subpackage.

### 3.9 Soft-TF-IDF (Cohen-Ravikumar-Fienberg 2003)
Hybrid token + character similarity, the "best fuzzy match in
literature" per the original Record Linkage benchmarks. Composes
TokenSetRatio with Jaro-Winkler. ~80 LOC. Tier 3 because TokenSetRatio
already covers 90% of consumer cases.

### 3.10 Longest common extension (LCE) queries
O(1) query after O(n) preprocessing using suffix array + LCP + RMQ.
Building block for fast approximate matching. ~80 LOC. Tier 3 — only
useful as substrate for other algorithms.

---

## Cross-package observations

- **NCD bridges `compression/` and `sequence/`** — would normalize
  similarity over arbitrary byte sequences. CLAUDE.md says no
  inter-package deps in reality/*; the workaround is to take a
  compression oracle as a parameter (see Tier 2 §2.7).
- **Pair-HMM and Karlin-Altschul depend on `prob/`** — same constraint.
  Either inline logsumexp / E-value math (CLAUDE.md rule 6 permits
  reimplementation) or take the function as a parameter.
- **Suffix array enables 5 other primitives** in this list (BWT,
  FM-index, LCP, generalized SA, LCE). Highest-leverage single algorithm
  to ship from Tier 1.
- The current package has zero **substring-search** entry point. KMP +
  BMH + Aho-Corasick fills a category, not just functions.
- `compression/RunLength*` already exists per CLAUDE.md — the rune-
  level mirror in `sequence/` is a courtesy duplicate, not a new
  algorithm.

---

## Highest-leverage shipping order

1. **Suffix array (SA-IS) + LCP array + BWT + FM-index** as one PR
   (~700 LOC). Single coherent unit, unlocks Tier 1 §1.7-1.9 + Tier 2
   §2.1 + Tier 3 §3.1, §3.10.
2. **Aho-Corasick** standalone (~200 LOC). Single most-asked-for
   missing algorithm; unlocks all multi-pattern consumers.
3. **Z-algorithm + KMP + BMH** in one "exact-match basics" PR (~150
   LOC). Pedagogical baseline.
4. **Hirschberg LCS + Wagner-Fischer edits + LevenshteinThreshold**
   (~150 LOC). Drops `LongestCommonSubsequence`'s O(mn) memory ceiling
   and exposes the edit script consumers want.
5. **Bitap + Levenshtein automaton + BK-tree** as "approximate match"
   PR (~500 LOC). The full triplet of fuzzy-search data structures.
6. **Metaphone + Double Metaphone + NYSIIS** (~600 LOC) — closes the
   phonetic gap that 126 brief falsely claimed was filled.
7. **Ratcliff-Obershelp + RapidFuzz ratio family** (~200 LOC).
8. **Affine-gap NW/SW + Pair-HMM + Karlin-Altschul + BLOSUM62**
   (~700 LOC) — bioinformatics tier.

Total Tier 1+2: roughly 3,200 LOC of net-new algorithm code, taking
the package from 15 entry points to ~50, and from "fuzzy-string
cheat-sheet" to "credible string algorithms library".

## Files audited

- C:\limitless\foundation\reality\sequence\alignment.go
- C:\limitless\foundation\reality\sequence\distance.go
- C:\limitless\foundation\reality\sequence\ngram.go
- C:\limitless\foundation\reality\sequence\phonetic.go
- C:\limitless\foundation\reality\sequence\token_ratio.go
- C:\limitless\foundation\reality\reviews\overnight-400\agents\126-sequence-numerics.md

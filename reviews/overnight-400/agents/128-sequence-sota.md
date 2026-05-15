# 128 — sequence: SOTA library comparison

Scope: benchmark `reality/sequence` (~15 entry points, fuzzy-string cheat-sheet
per 127) against the canonical SOTA: **rapidfuzz, jellyfish, edlib, parasail,
abPOA**, plus the deprecated/transitional **python-Levenshtein, fuzzywuzzy**
and the bit-parallel ancestor **Bitap (Wu-Manber)**. Aim: identify the
*headline algorithm*, the *engineering trick* that distinguishes each, and
the *zero-dep portability cost* to bring that trick into reality given
CLAUDE.md's "reimplement from first principles, only language stdlib"
constraint.

Don't repeat 126 (numerical defects) or 127 (missing primitives). This
slot is **algorithmic-headline focused**: which library invented which
trick, what makes each fast in 2026, and what it costs to port the trick
into a zero-dep Go file.

---

## Library matrix

| Library              | Lang       | Headline algorithm                               | Engineering trick                                                                                  | Zero-dep portable to reality? |
|----------------------|------------|--------------------------------------------------|----------------------------------------------------------------------------------------------------|-------------------------------|
| edlib                | C/C++      | Myers 1999 bit-parallel Levenshtein              | + Ukkonen banded + Hirschberg traceback (linear-space alignment path)                              | YES (~250 LOC)                |
| rapidfuzz / -cpp     | C++/Python | Hyyrö 2003 + Indel-distance bit-parallel ratio   | SIMD-experimental (`fuzz::experimental` ns), one-vs-many batch with packed 8/16-bit lanes          | PARTIAL (scalar yes, SIMD no) |
| jellyfish            | Rust/Py    | Soundex + Metaphone + NYSIIS + Match-Rating + JW | Pure dispatch table per phonetic algo, dual Rust/Python impls, NO bit-parallel anything            | YES (~600 LOC, 5 codes)       |
| parasail             | C99        | Smith-Waterman / Needleman-Wunsch / semi-global  | Farrar 2007 striped SIMD + origin-shift (8-bit signed → unsigned 0..255), 4 layouts × 4 ISAs       | NO (intrinsics-bound)         |
| abPOA                | C          | Partial-order alignment (POA) on DAG             | Adaptive band per row (band determined by predecessor scores) + SIMD vector-aligned base boundary  | NO (graph-aligner, niche)     |
| python-Levenshtein   | C          | Wagner-Fischer 1974                              | Naive C O(mn) loops, no bit-parallel; deprecated in favour of rapidfuzz                            | YES (already shipped, 126 fix)|
| fuzzywuzzy / TheFuzz | Python     | difflib SequenceMatcher (Ratcliff-Obershelp)     | Calls python-Levenshtein at runtime if importable, else pure-Python fallback (slow path)           | RO is portable, ~80 LOC       |
| Bitap / Wu-Manber    | (alg)      | NFA-as-bitvector simulation, k-error variant     | Mismatch-bit shift-and at machine word width; pattern length ≤ word-size = O(n) text scan          | YES (~120 LOC, see 127)       |

---

## 1. edlib — the gold standard for "fast Levenshtein"

**Headline:** Myers 1999 bit-vector dynamic programming. The DP column
of width *m* is packed into ⌈m/w⌉ machine words (w=64). Each column
update is 17 bit-ops independent of alphabet size. Time is
O(n⌈m/w⌉) — O(n) when m≤w (most Pistachio fuzzy lookups: command tokens
≤32 chars).

**Engineering tricks (3):**
1. **Ukkonen banded DP** — only compute cells where DP value ≤ k+|i-j|.
   Skip entire bit-vector chunks once they exceed threshold.
2. **Hirschberg linear-space traceback** — Myers gives the *score* in
   O(m+n) memory but loses the alignment path. Edlib runs Myers forward
   from both ends to the meet-in-the-middle row, then recurses. Path
   recovery in O(m+n) space instead of O(mn).
3. **Three modes shared** — global (NW), prefix (SHW = "free start gap on
   target"), infix (HW = "free start+end gap on target"). One forward
   kernel, three boundary conditions.

**Reality port:**
- Myers core: ~80 LOC of `uint64` shift-and-or, no SIMD needed for w=64.
- Ukkonen band: ~30 LOC of early-exit guard.
- Hirschberg meet-in-middle: ~100 LOC.
- Three modes: 30 LOC of init/extract differences.
- **Total ~240 LOC, zero deps.** Drop-in replacement for current
  `LevenshteinDistance` (which is naive Wagner-Fischer per 126), 30-100×
  faster on m≤64, 2-5× faster on longer strings.
- Per CLAUDE.md rule 3 ("no allocations in hot paths"), the Myers
  representation is naturally allocation-free for m≤64 and uses a
  caller-supplied workspace `[]uint64` for m>64.

This is the highest-leverage single PR identifiable from agents 126/127/128
combined. ~240 LOC closes the gap to edlib for the 90% case.

---

## 2. rapidfuzz — the reference modern fuzzy library

**Headline:** Hyyrö 2003 (a refinement of Myers 1999) for Levenshtein,
plus **Indel-distance** bit-parallel kernel for the ratio family.
Indel-distance = Levenshtein with substitution disallowed (only insert/
delete) — admits a tighter bit-parallel form due to monotone DP property.

**Engineering tricks (4):**
1. **Hyyrö's reformulation** — Myers needs alphabet pre-processing
   `Eq[c]` table of size σ×⌈m/w⌉. Hyyrö avoids one of the three bit-vectors
   per column update, ~30% scalar speedup over Myers.
2. **One-vs-many SIMD batching** (`fuzz::experimental` namespace,
   guarded by `RAPIDFUZZ_SIMD` macro). Process 16/32/64 candidate
   strings in parallel using AVX2/AVX-512 packed 8/16-bit lanes — one
   query against a corpus is the dominant rapidfuzz workload, fits
   SIMD-batch perfectly.
3. **Score-cutoff prune** — `extract(query, choices, score_cutoff=70)`
   passes the cutoff to the kernel which exits as soon as the partial
   DP makes the threshold unreachable. 5-50× win on filter-heavy queries.
4. **Composite ratios via shared scaffolding** — `ratio` (full),
   `partial_ratio` (sliding window), `token_sort_ratio` (alphabetic-sort
   tokens then ratio), `token_set_ratio` (set ops then ratio),
   `WRatio` (weighted blend) all reuse the same Indel kernel.

**Reality port:**
- Hyyrö core for Levenshtein: ~90 LOC scalar, replaces `LevenshteinDistance`.
- Indel kernel: ~70 LOC, new primitive. Used by ratio family.
- Score-cutoff variant: ~40 LOC additional, gates kernel early-exit.
- Ratio family: 127 already names these (`PartialRatio`, `TokenSortRatio`,
  `WRatio`); ~150 LOC shared scaffold.
- **SIMD-batch is NOT zero-dep portable** — Go has no stable intrinsics
  layer (asm-only via Plan-9 syntax, breaks the "stdlib only" rule).
  Skip SIMD; scalar-Hyyrö gets within 3-5× of rapidfuzz on single queries
  which is the reality use case (Pistachio looks up one user input against
  one command map, not 1000 candidates against 1000 candidates).
- **Total ~350 LOC for parity sans SIMD-batch**, zero deps.

The current `TokenSetRatio` implementation (127 says exists, 126 says docs
RapidFuzz parity but uses floor) lives here; this PR also retires the
"floor not banker's-rounding" defect from 126.

---

## 3. jellyfish — phonetic + simple distances

**Headline:** dispatch table per phonetic algorithm. No clever runtime
trick; the cleverness is in *coverage* — Soundex, NYSIIS, Metaphone,
Match Rating Approach, plus Levenshtein/Damerau/Jaro-Winkler/Hamming.

**Engineering trick (1):** **Dual Rust + Python implementations** with
identical semantics, golden-tested between them. This is structurally
the SAME pattern reality already has (Go canonical, Python/C++/C# validate
against golden files per CLAUDE.md). Jellyfish is the closest analogue
to reality's design philosophy in the comparison set.

**Reality port:**
- Soundex: already shipped (`phonetic.go:60`, see 126 for Y-as-vowel doc
  divergence note).
- Metaphone (Lawrence Philips 1990): ~120 LOC of consonant-rule cascade.
- Double Metaphone (Philips 2000): ~250 LOC, two parallel codes for
  ambiguous sources. Industry standard for English-language phonetic
  matching, used by Postgres `fuzzystrmatch`.
- NYSIIS (NY State 1970): ~80 LOC, rule cascade.
- Match Rating Approach codex + comparison: ~100 LOC.
- Caverphone v2 (Hood 2004) — NZ census, optional: ~120 LOC.
- Cologne phonetics (Postel 1969) — German names: ~100 LOC.
- **Total ~770 LOC for the phonetic family**, zero deps.
- Distance side already mostly covered by sequence; Damerau-Levenshtein
  and Jaro-Winkler exist (126 found defects in JW; here we just note
  jellyfish exists).

Phonetic family is medium-leverage: matters for Pistachio name lookup
("Davy" / "David" / "Davis" / "Davies" should phonetic-match) but not
for command parsing. Lower priority than the edlib port.

---

## 4. parasail — SIMD bioinformatics alignment

**Headline:** Smith-Waterman, Needleman-Wunsch, semi-global with full
substitution matrices and affine gaps, vectorized four ways: diagonal,
blocked, **striped (Farrar 2007)**, and prefix-scan.

**Engineering tricks (3):**
1. **Farrar striped layout** — instead of computing DP cells in row-major
   or anti-diagonal order, lay out columns striped across SIMD lanes:
   lane *l* of vector *v* holds DP[i,j] where i = v + l·(N/lanes).
   This means the recurrence's vertical dependency F[i,j]=max(F[i-1,j]-gap,
   ...) becomes intra-lane while H[i,j]=max(H[i-1,j-1]+s, F, E) becomes
   inter-vector. The "lazy F-loop" handles inter-lane F propagation in
   amortised O(1) per cell.
2. **Origin shift to unsigned** — Smith-Waterman scores are
   non-negative (max with 0 always). Standard SIMD has signed 8-bit
   range −128..127 (256 values). Shift origin to −128: now you have
   0..255 (256 values, 2× the dynamic range). Critical for short-read
   alignment where scores can hit 100+ in 8-bit.
3. **Profile precomputation** — for one query against many targets,
   precompute the substitution-row vector *per query column* once:
   `query_profile[c] = vector of s(q[i], c) for i in lane positions`.
   Replaces a scattered substitution-matrix lookup with one aligned
   vector load per inner-loop iteration.

Performance: 136 GCUPS on dual-Xeon (24 cores), highest reported for
striped Smith-Waterman.

**Reality port:**
- **Striped SIMD is NOT zero-dep portable to Go.** Go has no standard
  SIMD intrinsics; the only path is hand-written Plan-9 assembly per
  arch (amd64/arm64) which violates CLAUDE.md design rule 6
  ("reimplement from first principles" — fine, but rule 2 "zero
  dependencies" + the practical "language standard library only" maxim
  rules out asm).
- Scalar Gotoh affine-gap SW/NW: ~150 LOC, **portable**. This is what
  reality should ship (Gotoh 1982 affine-gap is what 127 named).
- Origin-shift is irrelevant in scalar (Go uses 64-bit ints by default).
- Query profile precomputation IS portable (the profile is a
  `[]int` per char, just precompute it once for the outer query) and
  gives ~2× scalar speedup on long alignments via cache locality. ~30
  LOC overhead.
- **Total ~180 LOC for scalar Gotoh SW/NW with query profile**, zero deps.

Reality is not a bioinformatics library. Aligning short reads against
a reference genome is not in reality's scope. But Gotoh SW with affine
gaps is the *correct* `SmithWaterman` (current sequence has linear-gap
SW per 127), and shows up in code-diff (Patience diff, Myers diff —
unrelated to Myers-1999) which IS a Pistachio use case.

---

## 5. abPOA — partial-order alignment for genomics

**Headline:** Partial-Order Alignment (POA): align a sequence to a
*directed acyclic graph* representing multiple sequences, not to a
single reference. Used in long-read consensus calling (PacBio HiFi,
Oxford Nanopore).

**Engineering tricks (2):**
1. **Adaptive band per row** — POA-DP cells are indexed by (graph-node,
   sequence-position). Standard banding sets a fixed band width; abPOA
   sets the band *per row* based on the predecessor rows' max-score
   position and the lengths of outgoing paths from each node. Up to
   15× faster than SPOA (the prior leader) without accuracy loss.
2. **SIMD vector-aligned band boundary** — the adaptive band has
   irregular boundaries which would normally defeat SIMD. abPOA
   rounds band boundaries to SIMD-vector granularity (e.g. multiples
   of 16 for SSE), processes only vectors fully inside the band.
   Trades a tiny bit of redundant computation for clean vectorisation.

**Reality port:**
- **Out of scope.** POA is genomics-specific; reality has no genomics
  consumer. abPOA's contribution to the sequence-library state-of-art
  is the *adaptive band per row* idea, which is a refinement of
  Ukkonen-banding (edlib) — but reality wants Ukkonen-band, not
  POA-band. Note for completeness, do not port.

---

## 6. python-Levenshtein, fuzzywuzzy — historical baseline

**Headline:** Wagner-Fischer 1974 in C; difflib's
SequenceMatcher (Ratcliff-Obershelp) in Python.

**Engineering trick:** None novel. python-Levenshtein is "compile the
naive O(mn) loop in C, get 4-10× over pure Python." fuzzywuzzy is "wrap
difflib's `ratio()` and add token-sort/token-set permutations."

**Status in 2026:** both are deprecated in practice. fuzzywuzzy was
renamed TheFuzz, then both effectively superseded by rapidfuzz which
the maintainer of TheFuzz now recommends. python-Levenshtein still
exists but rapidfuzz is the modern choice.

**Reality port:**
- Wagner-Fischer is what `LevenshteinDistance` already is (per 126).
- Ratcliff-Obershelp (gestalt pattern matching) is in 127's missing list,
  ~80 LOC zero-dep. Useful because difflib is the canonical Python
  reference and many ports want byte-equivalent comparisons.

---

## 7. Bitap / Wu-Manber — the bit-parallel ancestor

**Headline:** Baeza-Yates 1989 / Wu-Manber 1991 / Baeza-Yates-Navarro
1996. Encode an NFA recognising "pattern with up to k errors" as a set
of bitvectors `R₀, R₁, ..., Rₖ`. Per text character, all k+1 vectors
update in O(k) bit-ops. Total O(n·k/w) for pattern length ≤ word size.

**Engineering trick:** the NFA-as-bitvector simulation. Each bit
position represents "automaton has consumed *that many* pattern
characters." Shift-or each character advances all states in parallel.
Errors handled by *vertical* `R[d] |= R[d-1]<<1 | shift(R[d-1]) | shift(R[d])`
covering insertion/deletion/substitution.

**Reality port:**
- ~120 LOC for `BitapSearch(text, pattern string, k int) []int`
  returning all match positions. Restriction: pattern length ≤ 64
  (one machine word). Covers ~95% of approximate-search use cases
  (commands, names, log greps) since long patterns are rare.
- For longer patterns, fall back to Myers 1999 (the edlib port above)
  which extends naturally to multi-word.
- 127 names this in Tier-2; this slot agrees and adds: Bitap is the
  *prerequisite* for understanding rapidfuzz/edlib internals, so
  shipping it standalone (in addition to inside the Levenshtein kernel)
  is pedagogically valuable as well as functionally useful.

---

## Recommendation: order of operations

Given CLAUDE.md's "zero deps, language stdlib only, golden-file tested"
constraints, the SOTA-minus-SIMD frontier reality can credibly reach
(and the agent ranking by leverage):

| Order | Port               | LOC  | Wins                                           | Source library |
|-------|--------------------|------|------------------------------------------------|----------------|
| 1     | Myers 1999 + Hirschberg | ~240 | 30-100× LevenshteinDistance, alignment path | edlib          |
| 2     | Hyyrö+Indel kernel + RatioFamily | ~350 | RapidFuzz parity sans SIMD, fixes 126 defect 4 | rapidfuzz      |
| 3     | Bitap (Wu-Manber)  | ~120 | Approximate substring search (new primitive)   | classical      |
| 4     | Gotoh affine-gap SW/NW + query profile | ~180 | Replaces linear-gap SmithWaterman with biology-correct affine | parasail (scalar) |
| 5     | Phonetic family (Metaphone + NYSIIS + MRA + Caverphone + Cologne + DM) | ~770 | Pistachio name lookup, jellyfish parity | jellyfish      |

**Total ~1,660 LOC** to take reality/sequence from "fuzzy-string
cheat-sheet" (127's verdict) to "credibly comparable to rapidfuzz +
jellyfish + edlib for the scalar use case." SIMD parity (rapidfuzz
batched, parasail striped) is *not* on this list and *should not be*
on any list — it would require Go assembly modules per arch which
collide head-on with CLAUDE.md design rule 2 ("zero dependencies").

Reality's competitive position after these five ports: **the only Go
sequence library that ships golden-file-validated bit-parallel
Levenshtein, Hyyrö ratio family, Bitap approximate-search, Gotoh
affine-gap alignment, and 6 phonetic algorithms with cross-language
Python/C++/C# parity.** That is a real claim, achievable in a tractable
LOC budget, sourced from understanding what rapidfuzz/edlib/jellyfish
actually do under the hood.

---

## What reality should NOT do (anti-recommendations)

1. **Do not ship SIMD anything in the sequence package.** Plan-9 asm
   per-arch breaks design rule 2 and creates a Python/C++/C# parity
   nightmare (their golden-file validators have to match scalar Go
   bit-for-bit; SIMD reorderings perturb tied-score traceback).
2. **Do not port abPOA.** Genomics POA has no reality consumer.
3. **Do not port BLAST seed-extension.** Same — bioinformatics.
4. **Do not chase rapidfuzz batched performance.** Pistachio looks up
   one query at a time. The 30× single-query win from edlib-port + the
   3× from Hyyrö is enough; the further 10-30× from SIMD batching is
   not the bottleneck.
5. **Do not import any of these libraries via cgo.** Reality is
   zero-dep MIT Go-canonical. Reimplement from first principles per
   design rule 6.

---

## Cross-reference

- Defects to fix as part of this work: 126 defects 1, 4, 5 (Myers port
  obviates JW clamp scope but *does not* fix it; Hyyrö port replaces
  TokenSetRatio rounding mode; SW traceback determinism is fixed by
  parallel-pointer matrix in step 4 above).
- Missing primitives this work satisfies: 127 Tier-1 items
  Hirschberg LCS (step 1), Wagner-Fischer traceback (step 1),
  Bitap (step 3), Metaphone/Double-Metaphone/NYSIIS (step 5),
  Ratcliff-Obershelp (could fold into step 5 for ~80 LOC more),
  Gotoh affine-gap (step 4), RapidFuzz ratio family (step 2).
- Single biggest gap not addressed by SOTA-port path: **suffix
  automaton / FM-index / BWT** (127 Tier-2). Those are full-text-index
  primitives, not pairwise-comparison primitives — different shape of
  problem, different shape of port. Recommend a separate slot.

# 389 — meta-doc-cohesion (doc style consistency audit)

## Headline
Reality has a strong, ~80%-consistent godoc convention (Formula / Valid range / Precision / Reference) but mixes "Reference" vs "Source" labels, has zero `func Example` doctests, uses ad-hoc Greek/ASCII math notation, and has inconsistent citation formats — all fixable with a one-page style guide.

## Quantitative baseline (Go files only, excluding `_test.go`)
- Public functions: ~676 (`^func [A-Z]\w*\(`)
- `// Reference:` blocks: 378 across 118 files
- `// Source:` blocks: 215 across constants/units/physics + scattered
- Combined `Reference|Source|Citation`: 377 occurrences
- `// Precision:` blocks: 257 across 66 files
- `// Formula:` / `// Definition:` / `// Equation:`: 240 across 58 files
- `// Time complexity:` blocks: 89 across 23 files (sparse — mainly graph/sequence/crypto)
- `// Returns:` / `// Range:` headers: 56 (inconsistent — most use prose)
- Inline URLs in citations: **1** (`constants/physics.go` only)
- `func Example*`: **0** in entire repo
- Greek glyphs in source: 368 occurrences across 53 files

Citation coverage rough estimate: 378 `Reference:` + 215 `Source:` ≈ 593 reference markers vs 676 public functions ≈ **~70-80% of public functions cite a source** (some functions have multiple references, some have none — package doc.go files contribute). CLAUDE.md mandate: not fully met.

Doc-test (godoc Example) coverage: **0%**. CLAUDE.md doesn't require these but task brief flags it.

## Style audit — observed patterns

### Citation labels (inconsistent)
Two labels in active use, with no documented distinction:
- `// Reference:` — dominant for textbooks/papers (calculus, sequence, prob, em, crypto)
- `// Source:` — dominant for constants and "extracted from aicore" (constants/physics.go, constants/units.go, linalg/vector.go uses "Source: extracted from aicore/echomath...")
- `// See:` — only in `audio/vibration/doc.go:34`
- `// Citation:` — never used (despite task brief listing it)

Citation format variants observed:
- `Burden & Faires, Numerical Analysis, Chapter 4.` — textbook with chapter (calculus)
- `Knuth, TAOCP vol. 4A, Algorithm T` — textbook with algorithm name (combinatorics)
- `Wagner, Fischer (1974), "The String-to-String Correction Problem"` — author + year + title in quotes (sequence)
- `Damerau (1964), "A Technique for Computer Detection..."` — same shape (sequence)
- `Heap, B.R. (1963) "Permutations by Interchanges"` — initials + year (combinatorics/generate.go)
- `Brin, S. & Page, L. (1998) "The Anatomy of a Large-Scale Hypertextual..."` — full names + year (graph/pagerank.go)
- `Hill, G. W. (1970). Algorithm 396: Student's t-Quantiles. Communications of the ACM 13: 619-620.` — full bibliographic (prob/copula/studentt.go)
- `Acklam, P.J. (2004) "An algorithm for computing the inverse normal..."` — distinct comma style (prob/distributions.go)
- `Abramowitz & Stegun, Table 25.4` — locator, no year (calculus)
- `Abramowitz & Stegun 26.5.27` — locator, no comma (prob/copula/studentt.go)
- `Abramowitz & Stegun, formula 7.1.2` — locator with "formula" (prob/distributions.go)
- `NIST CODATA 2018 recommended value.` — agency + year (constants/physics.go)
- `Newton, I. (1687) "Principia Mathematica", Second Law of Motion` — primary source (physics/mechanics.go)
- `Coulomb, C.A. (1785); Griffiths "Introduction to Electrodynamics" 4th ed. eq. 2.1` — primary + secondary (em/em.go)
- `standard kinematic equations for projectile motion` — generic, no source (physics/mechanics.go:46) — minimum compliance with CLAUDE.md
- `standard Gaussian PDF` — same (prob/distributions.go:31)
- `https://physics.nist.gov/cuu/Constants/` — bare URL (constants/physics.go:12, only one in repo)
- `extracted from aicore/echomath.CosineSimilarity` — internal provenance (linalg/vector.go)

The Abramowitz & Stegun citations alone use **3 different formats**.

### Math notation (mixed Greek/ASCII)
Greek letters appear in 368 lines across 53 files but inconsistently:
- `Σ` (capital sigma for summation) — used in `calculus.go:223`, `acoustics.go:95`, `audio/onset/energy.go:10`, `queue/basic.go:166`, `audio/separation/vad.go:12,54` — but `prob/distributions.go:28` writes summation prose-style "(1 / (sigma * sqrt(2*pi))) * exp(...)" with **lowercase ASCII "sigma"** instead of σ.
- `π` vs `pi` — both used: `em/em.go:9` writes `4π ε₀`, but `prob/distributions.go:28` writes `sqrt(2*pi)`.
- `ε₀` (subscript zero unicode) used in em/em.go but `linalg/vector.go:18` writes `||a||` ASCII pipes for norm.
- `²` superscript used in `em/em.go` (`r²`, `N·m²/C²`) but `physics/mechanics.go:51` writes `t*t` and `g * t*t` plain ASCII.
- `≤`, `≥`, `≈` used in some files (`audio/spectrogram/stft.go:42`: `≤1e-9`) but most use `<=`, `>=`, `~`.
- LaTeX (`\sum`, `\int`, `\frac`) — **0 occurrences**. Reality does not use LaTeX.
- Greek words spelled out: `theta`, `mu`, `sigma`, `phi` — common in parameter lists.

There is no policy: same package (e.g. prob) freely mixes `μ` vs `mu`, `σ` vs `sigma`, `π` vs `pi`. Authors picked whatever they could type.

### Precision claim format (mostly consistent, several variants)
Most common pattern: `// Precision: <description>`. Sub-patterns:
1. Big-O truncation: `O(h^2) truncation error + O(eps/h) roundoff error` (calculus)
2. Significant digits: `limited by float64 sqrt (~15 significant digits)` (acoustics, very consistent)
3. Numerical bound: `≤1e-12 numerical error for typical 26-band, 13-coefficient` (audio/mfcc)
4. Round-trip: `HzToMel(MelToHz(m)) round-trips to <= 1e-9 of m for m in [0, 8000]` (audio/melscale)
5. Relative error: `relative error < 1e-12 for typical inputs` (combinatorics/counting)
6. Tag-only: `exact (single division)` / `exact (lossless)` / `exact (arithmetic only)` (acoustics, compression)
7. Range qualifier: `~1e-12 for typical (df, x)` (prob/copula/studentt)
8. Mantissa: `exact for n <= 20 (fits in uint64); float64 mantissa limits` (combinatorics)

Inconsistency: ASCII `<= 1e-9` vs unicode `≤1e-9` vs prose `~1e-12`. No documented spec for absolute vs relative error labeling — `1e-12` alone leaves the reader guessing.

### Cross-package references (informal, prose only)
Real cross-refs use lowercase package name in prose: `"signal.FFT requirement"`, `"e.g. signal.HannWindow"`, `"audio.PowerSpectrum after signal.FFT"`. No structured `// See: pkg.Func` form. `^// See:` appears 3 times total, two of which point to external docs (`ECOSYSTEM_QUALITY_STANDARD.md`).

### Other doc fields (sparse / inconsistent)
- `// Time complexity:` only in graph, sequence, crypto, combinatorics (89 occurrences / 676 funcs ≈ 13%).
- `// Returns:` rare (56 occurrences); most functions use prose "Returns the force in newtons" mid-paragraph.
- `// Valid range:` is the dominant input-spec field but spelled variously: `Valid range:`, `Valid input range:`, `Range:`, `Domain:`.
- Package-level docs are uniformly excellent (every package has a 3-6 line preamble in either the main `.go` file or `doc.go`).

## Inconsistencies summary (by severity)

| # | Issue | Files affected | Fix difficulty |
|---|-------|----------------|----------------|
| 1 | `Reference:` vs `Source:` mixed without rule | ~118 | Easy — pick one or document the split |
| 2 | Citation format varies (5+ shapes for textbook refs) | ~118 | Medium — define BibTeX-lite shape |
| 3 | Greek glyph use is per-author, not per-style | 53 | Medium — pick ASCII-only canonical form |
| 4 | Zero `func Example` doctests | all | Hard — net new content |
| 5 | Cross-package refs are prose, not links | ~15 | Easy — adopt godoc `[pkg.Sym]` form |
| 6 | `Precision:` sometimes missing units (abs vs rel) | ~30 | Easy |
| 7 | Two functions cite "standard" with no source (`physics/mechanics.go:46`, `prob/distributions.go:31`) | ~10 | Easy |
| 8 | Bare URL citations (only 1 instance) | 1 | Trivial |
| 9 | `Time complexity:` only in 4 packages | 19 missing | Medium |

## Recommendation: a single, simple style guide

Add `STYLE.md` (or expand CLAUDE.md) with these rules. They reflect the **dominant** existing pattern so adoption is mostly normalization, not rewriting.

### 1. Doc block order (existing dominant pattern, formalize it)
```
// FuncName does X.
//
// Formula: <one-line equation, ASCII only>
// Parameters: <bulleted, only if non-trivial>
// Returns: <prose, only if non-obvious>
// Valid range: <input domain + behavior at edges>
// Precision: <tagged: exact | ~Ne-K relative | ~Ne-K absolute | O(h^k)>
// Time complexity: <required for any function with loops or recursion>
// Reference: <Author (Year) "Title">
```
Drop `Source:` entirely — fold into `Reference:`. (Or: reserve `Source:` for *internal* provenance like `extracted from aicore/echomath.X` and `Reference:` for *external* literature. Document the split explicitly.)

### 2. Citation shape (one canonical form)
`// Reference: Author, Year. "Title". Locator.`
- Textbook: `Knuth, 1997. TAOCP vol. 2, Algorithm L.`
- Paper: `Wagner & Fischer, 1974. "The String-to-String Correction Problem". J. ACM 21(1).`
- Standard: `NIST CODATA 2018.` (sufficient for constants)
- Generic ("standard formula"): NOT acceptable. CLAUDE.md mandates a source.

This matches sequence/distance.go and gives a regex-parseable shape: `^// Reference: ([^,]+), (\d{4})\. (.+)$`.

### 3. Math notation: ASCII-only in godoc
Go's `go doc` does not render LaTeX, MathJax, or rich Unicode reliably across terminals. Pick ASCII as canonical:
- Summation: `sum_{i=0}^{N} f(i)` (NOT `Σ`).
- Integration: `int_a^b f(x) dx` (NOT `∫`).
- Greek letters: spell them — `mu`, `sigma`, `theta`, `pi`, `epsilon`. Add the glyph in parens once if disambiguation helps: `permittivity (epsilon_0)`.
- Powers: `x^2`, NOT `x²`.
- Norms: `||x||` (ASCII pipes).
- Comparisons: `<=`, `>=`, `~` instead of `≤`, `≥`, `≈`.

Rationale: existing files mix the two; ASCII works everywhere (terminals, IDEs, grep, source-code regexes for the regenerator scripts that emit cross-language wrappers). Greek is "prettier" but breaks tooling.

Exception: package preambles (`doc.go`) may use Greek for readability since they're prose, not function specs.

### 4. Precision claim format (one tagged grammar)
`// Precision: <tag> [<bound>] [<scope>]`
- `exact` — for arithmetic-only results.
- `~Ne-K relative` — relative-error bound, dominant for transcendentals.
- `<= Ne-K absolute` — absolute-error bound, dominant for iterative methods.
- `O(h^K)` — asymptotic order, dominant for finite-difference / quadrature.
- `~K significant digits` — for "limited by float64".

Bad: `~1e-12 for typical (df, x)` (relative or absolute? unclear).
Good: `~1e-12 relative for df > 1, |x| <= 50`.

### 5. Cross-package references: use godoc links
Go 1.19+ recognizes `[pkg.Func]` in comments and renders them as links. Adopt:
- `See [signal.FFT] for the underlying transform.`
- NOT `See signal.FFT for the underlying transform.`

This is a one-line search-and-replace and gives `pkg.go.dev` link rendering for free.

### 6. Doc-tests
Reality has 1,965 unit tests but **zero** `func Example` doctests. Add one per package's flagship function (22 examples total) — enough to render usage on `pkg.go.dev` without burdening every public function. Don't try for 100% coverage; it's not worth it for pure-math primitives where the test vectors already prove behavior. Recommend a Block F follow-up agent for a focused 22-example PR.

### 7. CLAUDE.md compliance fix
~20-30% of public functions lack a `Reference:` (mostly trivial wrappers like `OhmsLaw`, but explicitly required by rule 4 of CLAUDE.md "Every function cites its source"). Action: a one-shot pass through the ~80 unreferenced functions adding either a primary source or `derived from <equation X in package Y>` for compositions. Then enforce with a `golangci-lint` custom rule or a vet-style `cite-check.go` script that fails CI on any exported func whose godoc lacks `// Reference:` or `// Source:`.

## Estimate of style-guide adoption cost
- Auto-fixable (regex pass): ~120 file edits (label normalization, ASCII conversion of single-character Greek, godoc-link conversion).
- Manual review: ~80 functions missing citations.
- New content: 22 doc-test examples.
- Total: 1-2 day refactor, no behavior changes, all golden-file tests stay green.

## Sources
- `C:\limitless\foundation\reality\CLAUDE.md` (rule 4: "Every function cites its source"; rule 5: "Precision documented, not assumed")
- `C:\limitless\foundation\reality\calculus\calculus.go:29-241` (Reference: + Precision: dominant pattern)
- `C:\limitless\foundation\reality\sequence\distance.go:15-69` (best-shape citations: Author (Year), "Title")
- `C:\limitless\foundation\reality\prob\distributions.go:25-71` (mixes prose-style sigma, Acklam citation, "standard Gaussian PDF" non-citation)
- `C:\limitless\foundation\reality\prob\copula\studentt.go:16-58` (3 different A&S citation formats in same file)
- `C:\limitless\foundation\reality\physics\mechanics.go:20-71` (per-function consistent template; "standard kinematic equations" non-citation at :46)
- `C:\limitless\foundation\reality\em\em.go:18-79` (heavy Greek/superscript Unicode use: π, ε₀, ², N·m²/C²)
- `C:\limitless\foundation\reality\linalg\vector.go:15-68` ("Source: extracted from aicore/echomath" — internal provenance variant)
- `C:\limitless\foundation\reality\constants\physics.go:12,15-56` (only bare-URL citation in repo; "Source:" rather than "Reference:")
- `C:\limitless\foundation\reality\graph\pagerank.go:1-31` (Brin & Page citation format; uses `Time complexity:` and `Space complexity:`)
- `C:\limitless\foundation\reality\audio\onset\energy.go:10` & `queue\basic.go:166` & `acoustics\acoustics.go:95` (Σ unicode summation in code comments)
- `C:\limitless\foundation\reality\audio\spectrogram\stft.go:42` (≤1e-9 unicode comparison; "signal.FFT" prose cross-ref)
- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md:415` (assignment line confirmed)

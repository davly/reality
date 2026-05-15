# 048 | constants-sota

**Scope:** SOTA library/standard comparison for `C:\limitless\foundation\reality\constants\` (math.go 71 LOC, physics.go 82 LOC, units.go 102 LOC). Engineering-choices and metadata-hygiene focus. Sibling **046** covered numerical correctness (CODATA 2022 deltas, ULP audits); sibling **047** enumerated missing constants. This report is **disjoint** from both: it asks "how do peer libraries *encode* constants and what should reality borrow zero-dep?" not "what numbers are wrong" or "what numbers are missing."

**TL;DR.** Reality's encoding is the **simplest of the SOTA spectrum**: Go `const float64` literals + free-text doc comments. Peers (scipy.constants, NIST allascii, Boost.Units, mpmath, F# UoM, sympy.physics.units, ucum) all carry **machine-readable metadata** along three axes Reality drops: (1) **vintage** (CODATA/IAU year), (2) **uncertainty** (absolute or relative standard), (3) **dimensional analysis / canonical unit identifier**. Of these, **vintage + uncertainty are zero-dep, zero-runtime-cost, golden-file-compatible, and immediately actionable**; dimensional analysis is the design-pivot Reality has correctly declined (Boost.Units / F# UoM territory). The single highest-leverage borrow is **scipy.constants' `(value, unit, uncertainty)` tuple** rendered as a parallel Go `Meta` struct keyed by name, costing ~80 LOC and unlocking programmatic uncertainty propagation downstream (e.g., a future `prob/uncertainty` package). Document discipline ranks **NIST allascii > scipy.constants > sympy > Boost.Units > Reality**; engineering elegance ranks **Boost.Units > F# UoM > nalgebra-physics > Reality > scipy.constants**.

---

## 1. The seven SOTA references — catalog formats compared

### 1.1 NIST CODATA portal — `allascii.txt` (the upstream of upstreams)

**Format:** fixed-width plain text, machine-readable. Each line has 4 fields:

```
Quantity                                           Value                  Uncertainty            Unit
electron mass                                      9.1093837139 e-31      0.0000000028 e-31      kg
fine-structure constant                            7.2973525643 e-3       0.0000000011 e-3
proton-electron mass ratio                         1836.152673426         0.000000032
```

**Engineering choice:** plain text + a 4-column schema (name, value, std-uncertainty, unit). Vintage is embedded in the **filename and header** (`allascii_2022.txt`). 354 entries (CODATA 2022 release).

**Strength:** every value carries its **standard uncertainty as a separate field**. A consumer can compute "is this constant exact-by-SI-2019?" by checking `uncertainty == 0` (or the empty string for dimensionless). The uncertainty is the **standard uncertainty** (1σ), not a tolerance — they propagate through downstream calculations under the GUM (Guide to Expression of Uncertainty in Measurement, JCGM 100:2008).

**What Reality lacks:** a standard-uncertainty field. `physics.go:44` says "Uncertainty: 2.2e-5 relative standard uncertainty" in **prose**, not data. A consumer cannot programmatically distinguish `GravitationalConst` (u_r = 2.2e-5) from `Planck` (u_r = 0, exact).

### 1.2 scipy.constants — Python, the de-facto practitioner reference

**Format:** module-level constants as plain floats AND a `physical_constants` dict mapping name → 3-tuple `(value, unit, uncertainty)`:

```python
>>> from scipy.constants import physical_constants
>>> physical_constants['fine-structure constant']
(0.0072973525643, '', 1.1e-12)                   # value, unit-string, std-unc
>>> physical_constants['Newtonian constant of gravitation']
(6.6743e-11, 'm^3 kg^-1 s^-2', 1.5e-15)
>>> from scipy.constants import value, unit, precision
>>> precision('fine-structure constant')         # relative precision
1.5074988e-10
```

**Engineering choice:** **dual surface** — fast scalar floats (`scipy.constants.c`, `scipy.constants.h`) for hot paths PLUS dict-of-tuples for introspection. Vintage is **module-version-tracked** (scipy 1.13+ ships CODATA 2022; scipy 1.12 and earlier ship CODATA 2018 — they update on a CODATA cycle and announce in release notes).

**Coverage:** 354 named constants (matches NIST allascii) + ~70 unit conversion factors + SI prefix tables (yotta...quecto, full CGPM 2022 set including the 4 new prefixes from 2022) + `find()` regex search.

**The `precision` accessor** returns relative standard uncertainty — a one-liner over the dict. Reality has no equivalent.

**What Reality could borrow:** the dual-surface pattern. Keep `const SpeedOfLight = 299792458.0` for hot-path use, ADD `var Meta = map[string]Constant{ "SpeedOfLight": {Value: SpeedOfLight, Unit: "m/s", StdUnc: 0, Vintage: "SI-2019"}, ... }` for introspection. Cost: ~80 LOC, zero runtime cost on the const path, golden-file-compatible.

### 1.3 Boost.Units — C++, the type-system extreme

**Format:** templated types, dimensions encoded at the type level.

```cpp
quantity<si::length> r = 6.957e8 * si::meters;
quantity<cgs::length> r_cgs = quantity_cast<cgs::length>(r);  // compile-time conversion
// quantity<si::time> t = r;  // COMPILE ERROR — dimensional mismatch caught at compile time
constexpr auto h = boost::units::si::constants::codata::h;    // (value, uncertainty, unit) bundle
```

**Engineering choice:** **compile-time dimensional analysis** via expression templates. Constants live under `boost::units::si::constants::codata::*` and are **pre-bundled with their unit and uncertainty** as `quantity<unit, value_with_uncertainty<float64>>`. Header-only, zero runtime cost (modulo template instantiation cost at compile).

**Strength:** the type system makes `force + energy` a compile error. Vintage is namespaced: `codata::h` is the latest CODATA; older live under `codata_2014::h` etc.

**What Reality could borrow (zero-dep):** **NOT** the type system — that is a Go-vs-C++ design pivot belonging in a hypothetical `units/` package, not `constants/`. **YES** the **vintage namespacing**: `constants/codata2018/`, `constants/codata2022/` as submodules, with the top-level `constants` re-exporting the latest. Lets downstream code pin a vintage explicitly and avoids silent breakage on CODATA updates. ~30 LOC of restructuring; sibling 046 (action C-2022-2) recommends this.

**Cost reality has correctly avoided:** Boost.Units takes ~10-30s to compile for a non-trivial program due to template instantiation. The Go toolchain forbids this kind of compile-time complexity (no expression templates, no `constexpr`-of-types). Reality's `const float64` choice is **correct for Go** and **wrong-for-C++** — these are not the same language and direct comparison is misleading.

### 1.4 mpmath / sympy.physics.units — arbitrary-precision

**Format (mpmath):** `mp.mpf` arbitrary-precision floats; constants like `mp.pi` are *lazy* — they are computed to the current `mp.prec` when first referenced.

```python
>>> import mpmath as mp
>>> mp.mp.prec = 256
>>> mp.pi
mpf('3.141592653589793238462643383279502884197169399375105820974944592...')
>>> mp.mp.prec = 53
>>> float(mp.pi)
3.141592653589793                                # back to float64
```

**Engineering choice:** lazy precision-tracked computation. `mp.pi`, `mp.e`, `mp.euler`, `mp.catalan`, `mp.khinchin`, `mp.glaisher`, `mp.apery` are all functions of `mp.prec`, not literal constants.

**Format (sympy.physics.units):** symbolic units — `kilogram`, `meter`, `second` are SymPy expressions and `Quantity('elementary_charge', abbrev='e')` carries `dimension`, `scale_factor`, and convertibility into a `UnitSystem`.

**What Reality could borrow:** **the mpmath lesson is the negative-space lesson.** Reality's design choice — `const Pi = math.Pi` (float64 nearest) — is **the right choice for a 60-FPS-hot-path zero-dep library**. mpmath's lazy arbitrary-precision is a no-go for Reality's audience (Pistachio at 60 FPS cannot afford `mp.prec` switches).

**BUT** — the goldens infrastructure (CLAUDE.md "Go generates golden files via `math/big` at 256-bit precision") implicitly does what mpmath does explicitly. Reality could **emit the 256-bit reference values into the golden JSON as a `reference_high_precision` string field** (parsed by Python via `decimal.Decimal`/`mpmath.mpf` for high-precision validation, ignored by Go on float64 round-trip). Cost: ~30 LOC in the goldens generator. Documents the relationship between the float64 literal and the true mathematical value — sibling 046 (C-EXACT-1) implicitly asks for exactly this.

### 1.5 siunitx (LaTeX) — typesetting consistency, oddly relevant

**Format:** macros that bundle value + uncertainty + unit into a single typesetting primitive:

```latex
\qty{6.626 070 15 e-34}{\joule\second}                                       % value + unit
\num[uncertainty-mode=separate]{6.67430(15) e-11}                            % value(uncertainty) compact form
\unit{\kg\per\meter\cubed}                                                   % canonical unit string
```

**Why it matters:** `6.67430(15)e-11` is the **CODATA-canonical compact uncertainty notation** — the parenthesised digits are the uncertainty in the last shown places. NIST allascii uses the same convention in its display columns.

**What Reality could borrow:** when adding a `Meta` struct (per scipy.constants borrow), use the **compact-paren string** as a third storage form alongside (value, std-unc):

```go
type Constant struct {
    Value     float64    // for hot-path arithmetic
    StdUnc    float64    // standard uncertainty (1σ); 0 for exact
    Unit      string     // ucum-style canonical, e.g., "m.s-1"
    Vintage   string     // "SI-2019", "CODATA-2022", "IAU-2015"
    Notation  string     // CODATA compact, e.g., "6.67430(15)e-11"
    Source    string     // e.g., "https://physics.nist.gov/cgi-bin/cuu/Value?bg"
}
```

Cost: ~5 LOC per existing constant. The `Notation` field round-trips into siunitx-compatible LaTeX for free, and matches the human-eyeball check against allascii.

### 1.6 ucum (Unified Code for Units of Measure)

**Format:** a regular grammar for unit strings, RFC-grade. `m.s-1`, `kg.m2.s-2`, `eV` (recognised), `[in_i]` (international inch), `[gal_us]` (US gallon), `[gal_br]` (UK/Imperial gallon). The full grammar is HL7-standard for medical/scientific data exchange and is the canonical answer to "how do I serialise a unit string unambiguously?"

**Why it matters:** "m/s" vs "m s^-1" vs "m·s⁻¹" vs "metre per second" are all the same unit but different strings. ucum picks `m.s-1` as canonical. **Reality currently writes units in free-text comments** (`// m^3 kg^-1 s^-2` on physics.go:42) — every constant uses ad-hoc notation.

**What Reality could borrow zero-dep:** **adopt ucum as the unit-string canonical form** in the `Meta.Unit` field above. NIST allascii units map cleanly. No ucum parser needed at runtime — just discipline in the literal strings. Cost: ~0 LOC code, ~30 minutes of editing comments. The **single most-leveraged hygiene improvement available**: it lets a future `units/` package or external tool parse Reality's unit strings without a special-case Reality-prose parser.

Map for current units.go:
- `// m/s` → `m.s-1`
- `// m^3 kg^-1 s^-2` → `m3.kg-1.s-2`
- `// J/K` → `J.K-1`
- `// W m^-2 K^-4` → `W.m-2.K-4`
- `// F/m` → `F.m-1`
- `// H/m` → `H.m-1`
- (etc.)

### 1.7 F# units of measure — the lightweight type-tagged alternative

**Format:** post-fix unit annotations on float types. Compile-time dimension checking, **zero runtime cost** (units are erased at compile):

```fsharp
[<Measure>] type kg
[<Measure>] type m
[<Measure>] type s
let G : float<m^3 / (kg * s^2)> = 6.67430e-11<m^3 / (kg * s^2)>
let mass : float<kg> = 5.972e24<kg>
let radius : float<m> = 6.371e6<m>
let g_local = G * mass / (radius * radius)   // typechecks to <m / s^2>
```

**Engineering choice:** type-erased dimensional analysis. Compiles to identical IL as plain `float`, so the runtime cost is zero. **F# UoM is the closest dimensional-analysis system to "Go-feasible"** of any language Reality could imitate.

**What Reality could borrow — and could not:** Go has **no type-level natural-number arithmetic**. There is no Go equivalent of `m^3 / (kg * s^2)` as a type. Generics-with-numeric-type-params (Go 1.21+) get partway there but cannot encode dimensional algebra. The realistic Go answer is **named-type wrappers** (`type Meters float64`, `type Newtons float64`) which cover unit *identity* but not unit *algebra* (`Meters * Meters` ≠ `SquareMeters` automatically). This is a **`units/` package design question**, not a `constants/` change. Note for the design backlog.

### 1.8 nalgebra-physics (Rust) and Frink and others — quick pass

- **nalgebra-physics:** Rust crate; `const`-fn constants under `nalgebra_physics::constants::si::*`; CODATA-2018 vintage (last published 2024-08, mature CODATA-2022 update pending); design closely mirrors Boost.Units but with Rust's stricter `const fn` rules, so the const arithmetic Reality uses (`PlanckReduced = Planck / (2*Pi)`) is fully supported.
- **Frink (Alan Eliasen, 2000-present):** "calculator with units"; ships ~3000 constants/units in a plain-text dimension-tagged file. **Vintage explicitly tracked** in the file header. Free-form, batteries-included, the spiritual ancestor of `unit(1)` / GNU units. Not a library to imitate but a **reference for naming** (its unit catalog is encyclopaedic — "barn", "shake", "siriometer").
- **GNU units:** ships `/usr/share/units/definitions.units`; vintage tagged in the file; pure-text format very similar to Frink. Useful **as a unit-catalog cross-check** for sibling 047's missing list.
- **Wolfram/Mathematica `Quantity[]`:** dimension-tagged values like `Quantity[6.67430*^-11, "Meters"^3 / ("Kilograms" "Seconds"^2)]`. Closed-source but the API is the gold-standard.

---

## 2. Cross-library catalog format matrix

| Library | Surface | Vintage | Uncertainty | Unit | Source citation | Dim-check |
|---|---|---|---|---|---|---|
| **NIST allascii** | flat text, 4-col | filename+header | std-unc col | text col | none (it IS the source) | n/a |
| **scipy.constants** | scalars + dict | scipy version | tuple[2] | tuple[1] string | `find()` returns key | none |
| **Boost.Units** | typed templates | namespace `codata::` `codata_2014::` | bundled in `value_with_uncertainty` | type-encoded | doc-comment | **compile-time** |
| **mpmath** | lazy func | n/a (mathematical, not measured) | precision-controlled | n/a | none | n/a |
| **sympy.physics.units** | `Quantity()` symbols | `unit_system` arg | none | `dimension` field | `Quantity.kwargs['source']` | symbolic |
| **siunitx** | LaTeX macros | none | `\num{}` paren-form | `\unit{}` | none | none |
| **ucum** | string grammar | not applicable | not applicable | **canonical string** | RFC | parser-side |
| **F# UoM** | typed scalars | none built-in | none built-in | `[<Measure>]` types | none | **compile-time, erased** |
| **nalgebra-physics** | `const fn` Rust | crate version | none | doc-comment | doc-comment | none |
| **Frink/GNU units** | text file | file header | text | unit-string | inline | parser-side |
| **Wolfram Quantity** | `Quantity[v, u]` | knowledge-base date | optional via `Around[]` | string | sealed | symbolic |
| **Reality (current)** | `const float64` | **doc-comment prose** | **doc-comment prose** | **doc-comment prose** | doc-comment prose | none |

Reality is the only library in the matrix with **all four metadata axes in unstructured prose**. This is the doc-discipline gap. It is also a fixable gap with ~80 LOC and zero runtime cost.

---

## 3. Engineering choices: compile-time vs runtime tables

A taxonomy of how peer libraries *expose* constants:

**(A) Compile-time scalar literals.** scipy.constants module-level (`scipy.constants.c`), Reality (`constants.SpeedOfLight`), nalgebra-physics, Frink names exposed via codegen. Hot-path-friendly. **No metadata.**

**(B) Compile-time bundled-with-units.** Boost.Units `codata::h`, F# `6.67430e-11<m^3/(kg*s^2)>`. Hot-path-friendly. **Dimensional metadata via type system.** Closed surface — adding a constant requires a code change.

**(C) Runtime dictionary.** scipy.constants `physical_constants[name]`, sympy `Quantity()`, NIST allascii (text → in-memory dict). Slower (string lookup, hash), open surface (extensible at runtime). **Full metadata: value, unit, uncertainty, name.**

**(D) Symbolic.** Wolfram `Quantity[]`, sympy `physics.units`, mpmath. **Maximum metadata, no runtime arithmetic-fast-path.**

**Reality's current position:** pure (A). The proposed scipy-style dual surface is **(A) + (C)** — keep the const literals, ADD a parallel Go map keyed by name. This is the only pattern that preserves the 60-FPS hot path while gaining metadata for downstream introspection.

```go
// Recommended: keep current const literals, add parallel metadata var.
package constants

const SpeedOfLight = 299792458.0  // unchanged hot-path const

var Meta = map[string]ConstantMeta{
    "SpeedOfLight": {
        Value:    SpeedOfLight,
        StdUnc:   0,
        Unit:     "m.s-1",                          // ucum
        Vintage:  "SI-2019",
        Source:   "https://physics.nist.gov/cgi-bin/cuu/Value?c",
        Notation: "299792458 (exact)",
    },
    "GravitationalConst": {
        Value:    GravitationalConst,
        StdUnc:   1.5e-15,                          // CODATA 2022
        Unit:     "m3.kg-1.s-2",
        Vintage:  "CODATA-2022",
        Source:   "https://physics.nist.gov/cgi-bin/cuu/Value?bg",
        Notation: "6.67430(15)e-11",
    },
    // ... 11 more
}

type ConstantMeta struct {
    Value, StdUnc       float64
    Unit, Vintage       string
    Source, Notation    string
}
```

Compile-time cost: zero (a `var` initialised once at package init, never accessed on hot paths). Runtime cost when used: one map lookup. **Adding new constants becomes a two-line change** (one for the const, one for the Meta entry) — symmetric with how `physicsConstantMap` in `golden_test.go:12-26` already manually mirrors the consts.

---

## 4. Five concrete zero-dep borrows, ranked

### Borrow 1 — scipy-style `Meta` map (highest leverage)
**Cost:** ~80 LOC, one new file `meta.go`, no API break. **Unlocks:** programmatic uncertainty propagation, vintage queries, downstream auto-generated docs, JSON/YAML serialisation for cross-language goldens. **Action:** ship `Meta map[string]ConstantMeta` covering all 31 current constants, populated from CODATA 2022 / SI 2019 / ISO 80000 / 1959 yard-and-pound. Sibling 046 (C-DERIVED-2) implicitly asks for this; this report sharpens the API.

### Borrow 2 — ucum unit strings in comments / Meta.Unit
**Cost:** ~30 minutes editing existing comments. **Unlocks:** machine-parseable unit identification; future `units/` package gets a clean grammar to consume; matches NIST allascii / scipy unit conventions. **Action:** rewrite all `// J/K` style comments to `// J.K-1` ucum format; populate `Meta.Unit` with the same string. Sibling 047 implicitly assumes this is done in its doc-block additions.

### Borrow 3 — Boost-style vintage namespacing (`codata2018/`, `codata2022/`)
**Cost:** ~30 LOC, breaking-but-backwards-compatible rename. **Unlocks:** explicit version pinning; CODATA 2026 update doesn't silently break consumers; matches Boost / F# / Wolfram practice. **Action:** add `constants/codata2022/physics.go` and `constants/iau2015/astro.go` submodules, top-level re-exports the latest. Sibling 046 (C-2022-2) explicitly asks for this; this report locates it as Boost-canonical practice.

### Borrow 4 — siunitx-style `Notation` field with `value(unc)` compact form
**Cost:** ~5 LOC per constant in the `Meta` table from Borrow 1. **Unlocks:** human-eyeball verification against NIST allascii (which uses the same `6.67430(15)e-11` notation in its display columns); LaTeX generation for documentation. **Action:** populate `Meta.Notation` strings.

### Borrow 5 — mpmath-style high-precision reference values in goldens
**Cost:** ~30 LOC in the (currently Go-only) goldens generator + ~13 lines of JSON additions. **Unlocks:** Python/C++/C# implementations can validate against the **arbitrary-precision exact** value (parsed via `decimal.Decimal`), not just the float64 nearest. Currently Reality goldens have `expected: 6.62607015e-34` with `tolerance: 0` — works because the literal IS the value, but it cannot detect a Python implementation that rounds incorrectly. **Action:** add `expected_high_precision: "6.62607015e-34"` (string-typed Decimal) field to the JSON schema. Sibling 046 (C-EXACT-1) implicitly requires this for the `// exact` constants.

**Not borrowed:** type-encoded dimensional analysis (Boost.Units / F# UoM). Reality has correctly declined this — Go's type system cannot express it cheaply, and the design pivot belongs in a hypothetical `units/` package, not `constants/`.

---

## 5. Vintage / metadata hygiene — the doc-discipline gap

**Current state:** `physics.go:11` says "NIST CODATA 2018 recommended values otherwise" in a top-of-file comment. The vintage is **a comment, not a value**. There is no `const CODATAVintage = "2018"` — a consumer cannot programmatically check which CODATA cycle the package targets, cannot detect a stale checkout, cannot pin a build to a specific vintage.

**Peer comparison (vintage-as-data discipline ranking):**
1. **NIST allascii** — vintage in filename and header line, machine-readable.
2. **scipy.constants** — vintage in module version (`scipy.__version__`), release-noted.
3. **Boost.Units** — vintage in namespace name (`codata_2014`, `codata_2018`, `codata_latest`).
4. **sympy.physics.units** — vintage as a `unit_system` constructor argument.
5. **Wolfram Quantity** — vintage embedded in the knowledge base, exposed via `EntityValue`.
6. **mpmath / Frink** — vintage in file header.
7. **Reality** — **vintage in a free-form comment.** Worst tier.

**Hygiene fixes (each ~1-3 LOC, zero risk):**
- `const CODATAVintage = "2022"` (after 046 update) in physics.go.
- `const SIDefiningRedefinition = "2019"` for the four exact constants.
- `const ISO80000Vintage = "2019"` for g_n, atm.
- `const YardPoundVintage = "1959"` for ft, in, yd, mile, lb, oz.
- `const IAUVintage = "2015"` for any astronomical (when 047's astro additions land).

**Aggregate cost:** ~6 LOC. Aggregate value: every downstream consumer — including Reality's own golden generators, Pistachio's Python/C++/C# rebuilds, and any CI gate — can now query `constants.CODATAVintage` and produce a meaningful version manifest. The only library in the SOTA matrix that does not do this is Reality.

**Tightest win — a single constant:**

```go
// Aggregate provenance. Bumped per cycle.
const (
    CODATAVintage          = "2022"   // physics.nist.gov/cuu/Constants
    SIRedefinitionVintage  = "2019"   // BIPM SI Brochure 9th ed.
    ISO80000Vintage        = "2019"   // ISO 80000-3:2019
    IAUVintage             = "2015"   // IAU Resolution B3 (when astro.go lands)
    YardPoundVintage       = "1959"   // Intl yard & pound agreement
)
```

Five lines. Each is queryable. Each will be bumped explicitly on update. This is the **single most-leveraged metadata-hygiene change available** and it borrows directly from Boost.Units namespace pattern.

---

## 6. Recent-SI / IAU / leap-second updates relevant to engineering choices

(These are *engineering implications* of upstream changes 046/047 already itemised; this section covers the *meta-implications*.)

- **CGPM 2022 retire-leap-seconds resolution (Resolution 4)** — by or before 2035, the leap second will be discontinued. Reality's `SecondsPerDay = 86400.0` is correct for TAI/TT/UT1 but undocumented re UTC. **Engineering implication:** when the time/clock package lands, design with leap-second optionality from day one — do not bake `86400 s/day` as a UTC truth. Borrow from Astropy `Time` and IAU SOFA C library: time-scale-tagged scalars.
- **IAU 2018 Resolution B5 — definitions for the BIPM time scales** (TCB, TDB, TCG, TT, UT1, UTC) — formalised the seven time-scale taxonomy. **Engineering implication:** if Reality eventually ships `time/`, mirror SOFA's `TimeScale` enum.
- **ICRF3 (IAU 2018, effective 2019-01-01)** — successor to ICRF2. Defining-source list is machine-readable from the IERS. **Engineering implication:** none for `constants/`, but downstream `astro/` packages will want it.
- **CGPM 2022 prefix expansion (ronna/quetta/ronto/quecto)** — first new prefixes since 1991. **Engineering implication:** if Reality ships SI prefix tables (047 Tier 1.9), it must include the four new ones.

---

## 7. What Reality is doing right (don't change)

- **`const float64` literals on the hot path.** All seven SOTA libraries that target performance (Boost.Units, F# UoM, nalgebra-physics, scipy module-level scalars) make the same choice. Reality's mistake would be moving to runtime tables (Python `dict`-style) on the hot path.
- **Const-arithmetic for derived exact-from-exact** (`PlanckReduced = Planck/(2*Pi)`, `GasConstant = Avogadro*Boltzmann`). Compile-time evaluation, golden-file-determinable, matches Boost / nalgebra-physics practice.
- **`math.Pi` / `math.Sqrt2` aliasing** — correct: do not duplicate the stdlib's correctly-rounded float64 representations. scipy.constants does the analogous `numpy.pi` reuse.
- **Free-text source citations in comments.** All SOTA libraries do at least this. Reality's prose ("Source: SI 2019 exact definition.") on every const is the **best-in-class for a 200-LOC zero-dep package**. The metadata gap is *additional* structure, not a replacement.
- **Zero dependencies + golden-file cross-language validation.** Aligns with NIST allascii's text-format philosophy: the canonical numbers live somewhere portable. CLAUDE.md's design rule 1 ("Golden files are the proof") is the right discipline; this report's recommendations all preserve it.

---

## 8. Summary of recommendations (ranked by leverage)

| ID | Borrow | Source | LOC | Risk | Value |
|---|---|---|---|---|---|
| S-META-1 | Add `Meta map[string]ConstantMeta` parallel surface | scipy.constants `physical_constants` | ~80 | Low (additive) | High — programmatic uncertainty + vintage |
| S-META-2 | Add `const CODATAVintage = "2022"` etc. block | Boost.Units namespacing | ~6 | Zero | High — version pinning |
| S-META-3 | Adopt ucum unit strings in `Meta.Unit` and comments | ucum / NIST allascii | ~0 (edits) | Zero | Medium — machine-parseable units |
| S-META-4 | Add CODATA compact-paren `Notation` field | siunitx / NIST allascii display | ~30 | Zero | Medium — human eyeball verification |
| S-META-5 | Goldens carry `expected_high_precision` Decimal string | mpmath / `decimal.Decimal` | ~30 | Low | Medium — Python implementation pinning |
| S-META-6 | Restructure into `codata2022/`, `iau2015/` submodules | Boost.Units `codata_2014::` style | ~30 | Medium (refactor) | Medium — explicit vintage pins |
| S-META-7 | Document leap-second / time-scale caveat on `SecondsPerDay` | IAU 2018 B5 + CGPM 2022 R4 | ~3 | Zero | Low (caveat only) |

**Highest-value bundle:** S-META-1 (Meta map) + S-META-2 (vintage consts) + S-META-3 (ucum units) ≈ **~95 LOC, zero risk, fully additive**, lifts Reality from "comment-only metadata, worst-tier in the SOTA matrix" to **"par with scipy.constants and ahead of nalgebra-physics on documentation discipline."** Together with sibling 046's CODATA 2022 numerical update and sibling 047's missing-constants additions, this is the v0.11 metadata refactor.

**Disjoint from 046/047:** this report's recommendations are **purely structural** — they add metadata fields and naming conventions without touching the underlying float64 values (046's territory) or the set of named constants (047's territory). All three reports compose: 047 says "add `BohrRadius`," 046 says "make sure it's CODATA 2022 5.29177210544e-11," 048 says "and put its uncertainty 8.2e-21, ucum unit `m`, vintage `CODATA-2022`, source URL into the parallel `Meta` map."

---

## Sources

- [NIST CODATA 2022 fundamental constants — allascii.txt](https://physics.nist.gov/cuu/Constants/Table/allascii.txt) — flat-text catalog format reference, fetched 2026-05-07
- [scipy.constants — Python reference](https://docs.scipy.org/doc/scipy/reference/constants.html) — dual-surface (`scipy.constants.c` + `physical_constants[name]`) and `find()`/`value()`/`unit()`/`precision()` accessors
- [Boost.Units documentation](https://www.boost.org/doc/libs/release/doc/html/boost_units.html); [`boost::units::si::constants::codata`](https://www.boost.org/doc/libs/release/doc/html/boost/units/si/constants/codata.html) — type-encoded dimensions; vintage namespacing
- [mpmath — arbitrary-precision floating-point arithmetic](https://mpmath.org/) — lazy precision-controlled constants (`mp.pi`, `mp.euler`, `mp.catalan`, `mp.khinchin`, `mp.glaisher`, `mp.apery`)
- [SymPy `physics.units`](https://docs.sympy.org/latest/modules/physics/units/index.html) — symbolic `Quantity()` and unit-system ledger
- [siunitx — LaTeX package](https://ctan.org/pkg/siunitx) — `\qty{}`, `\num{}`, `\unit{}` typesetting macros; CODATA paren-form uncertainty
- [Unified Code for Units of Measure (ucum) v2.1 (Schadow et al., Regenstrief)](https://ucum.org/) — canonical unit string grammar
- [F# units of measure (Microsoft)](https://learn.microsoft.com/en-us/dotnet/fsharp/language-reference/units-of-measure) — type-erased dimensional analysis
- [nalgebra-physics (Rust)](https://docs.rs/nalgebra-physics/) — `const fn` Rust constants, CODATA-2018 vintage
- [Frink language](https://frinklang.org/) and [GNU units](https://www.gnu.org/software/units/) — text-file constant catalogs
- [Wolfram Language `Quantity[]`](https://reference.wolfram.com/language/ref/Quantity.html) — symbolic dimension-tagged values
- [BIPM SI Brochure (9th edition, 2019)](https://www.bipm.org/en/publications/si-brochure) — 7 SI defining constants formalised exact
- [CGPM 2022 Resolution 3 (new prefixes ronna/quetta/ronto/quecto)](https://www.bipm.org/en/cgpm-2022/resolution-3); [CGPM 2022 Resolution 4 (retiring leap seconds by 2035)](https://www.bipm.org/en/cgpm-2022/resolution-4)
- [JCGM 100:2008 (GUM) — Evaluation of measurement data: Guide to the expression of uncertainty in measurement](https://www.bipm.org/en/publications/guides/gum.html)
- IAU 2018 Resolution B5 (time scales TCB/TDB/TT/UT1/UTC); IAU 2018 Resolution B2 (ICRF3, effective 2019-01-01)

Report at `agents/048-constants-sota.md`.

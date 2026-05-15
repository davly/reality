# 049 | constants-api

**Scope.** API ergonomics of `constants/`. 31 exported names — 10 in
`math.go` (71 LOC), 13 in `physics.go` (82 LOC), 18 in `units.go` (102
LOC, count includes the 5 unit-prefix groupings) — all `const float64`,
no exported types, no functions. **Disjoint from 046** (CODATA-2022
numerics), **047** (missing surface), **048** (peer-library metadata
encoding: scipy parallel `Meta` dict, Boost vintage namespacing). This
report only covers **how a Reality consumer calls the package** — names,
direction, grouping, units-in-doc, intermediate-factor exposure, stdlib
re-export, conversion-helper shape.

## TL;DR

Three concrete **call-site bugs** that 046/047/048 didn't surface:

1. **`constants.Pi` is dead code.** 1 import site (`pkg/canonical`) vs
   50+ files importing `math.Pi` directly. The aliasing layer is doing
   nothing.
2. **`color/difference.go:130` reimplements `deg2rad`/`rad2deg` as
   private free functions.** 7 call sites in CIEDE2000. The package
   ships `DegreesToRadians`/`RadiansToDegrees` — and color does not use
   them. Same shape gap exists wherever code does
   `value * (math.Pi / 180.0)` inline. Constant-as-multiplier is not a
   substitute for a function — Go's type system has no way to bind the
   "this float is degrees" intent at the call site.
3. **The temperature surface is half-built.** `CelsiusToKelvin` is a
   *constant* (additive offset 273.15); the docstring even says
   "Usage: `kelvin := celsius + CelsiusToKelvin`". `FahrenheitToKelvin`
   needs **two** constants (`FahrenheitToKelvinOffset`,
   `FahrenheitToKelvinScale`) and the user must remember `(F + 459.67)
   * 5/9`. There is **no `KelvinFromFahrenheit` function** that hides
   the algebra. This is the only conversion in the package that cannot
   be expressed as a single-multiply.

The single highest-leverage API change: **add a thin
`constants/convert` subpackage** of pure functions
(`KelvinFromCelsius(c)`, `RadiansFromDegrees(d)`, `MetersFromFeet(ft)`,
…) backed by the existing constants. ~25 functions, ~80 LOC, fixes (2)
and (3), composes with golden-file infrastructure, requires no new
types, fits CLAUDE.md's "numbers in, numbers out" rule. Section 7
specifies the shape.

---

## 1. Naming convention: `<Target>Per<Source>` is consistent — almost

`units.go:6-9` documents the rule: `<Target>Per<Source>` means "multiply
a value in `<Source>` units to get `<Target>`":

```go
meters := feet * MetersPerFoot   // MetersPerFoot = 0.3048
```

Audit of all 18 unit-conversion constants:

| Constant | Direction matches `<T>Per<S>`? | Notes |
|---|---|---|
| `MetersPerMile/Foot/Inch/Yard/NauticalMile` | yes | length ↔ |
| `KgPerPound/Ounce` | yes | mass ↔ |
| `PascalsPerAtm/Bar/PSI` | yes | pressure ↔ |
| `SecondsPerMinute/Hour/Day` | yes | time ↔ |
| `RadiansToDegrees` | **no — uses `To` not `Per`** | naming break |
| `DegreesToRadians` | **no — uses `To`** | naming break |
| `CelsiusToKelvin` | **no — uses `To` and is an offset, not a factor** | semantic break |
| `FahrenheitToKelvinOffset` | not a single conversion factor | semantic break |
| `FahrenheitToKelvinScale` | not a single conversion factor | semantic break |

**Finding API-1.** Three different schemata coexist:
`<T>Per<S>` (multiplicative factor, 13 constants), `<S>To<T>` (also a
multiplicative factor — same intent, different word — 2 constants:
`RadiansToDegrees`, `DegreesToRadians`), and `<S>To<T>Offset/Scale`
(component of an affine map, 3 constants). A user who has internalized
`MetersPerFoot` reads `RadiansToDegrees` and must re-derive direction
from name; the two patterns are semantically identical (both are
`multiply this by source to get target`), only the preposition differs.

**Action API-1A** (rename, breaking): standardize on the documented
convention. `RadiansToDegrees` → `RadiansPerDegree` is **wrong** (would
flip the value); the correct rename is `DegreesPerRadian` (= 180/π) for
the existing `RadiansToDegrees` value, `RadiansPerDegree` (= π/180) for
the existing `DegreesToRadians`. Equivalently, rename `MetersPerFoot`
etc. to `FootToMeter` (= 0.3048). Either direction works; current state
is the worst possible — both conventions present.

**Action API-1B** (alias, non-breaking): keep the existing names, add
the canonical-direction alias as a `const` of the other:
`const DegreesPerRadian = RadiansToDegrees`. ~6 LOC. Cheap. Tests
already verify `RadiansToDegrees * DegreesToRadians == 1` so no extra
test needed beyond an alias-equality test.

**Finding API-2 — temperature is not a unit conversion, it is an affine
map.** `CelsiusToKelvin = 273.15` is an additive offset, not a
multiplicative factor; the name follows the **same surface form** as
`MetersPerFoot` but does not multiply. A naive consumer who has
internalized "multiply by the constant" will write
`kelvin := celsius * CelsiusToKelvin` and get garbage. The docstring at
`units.go:54` correctly says "Usage: `kelvin := celsius +
CelsiusToKelvin`" — but the **type is `float64`**, the same type as
`MetersPerFoot`, so the type system cannot enforce the addition-vs-
multiplication distinction. This is the single strongest argument in
the entire package for adding a function surface (Section 7).

**Action API-2.** Rename `CelsiusToKelvin` → `CelsiusKelvinOffset`
(matches `FahrenheitToKelvinOffset` already in the package). The `*To*`
prefix should be reserved for the non-existent multiplicative form
("Celsius to Kelvin scale factor" — degree-step is 1, but that
constant would be `CelsiusKelvinScale = 1.0` if added for symmetry).

---

## 2. Direction: `MetersPerFoot` vs `FootInMeters`

Both are common in the wild. Reality picks `MetersPerFoot`. Survey of
peer libraries (drawn from 048 references, intentionally non-overlapping
with that report's metadata-format axis):

| Library | Direction | Example |
|---|---|---|
| Reality | target-per-source | `MetersPerFoot` |
| Go stdlib `time` | source-per-target | `time.Hour` (a Duration valued at hour-in-ns), `time.Second` |
| scipy.constants | source-named, value implicitly in SI | `scipy.constants.foot = 0.3048` (foot's value in m) |
| Boost.Units | dimension-typed quantity, no naming choice | `1.0 * foot` is a typed value |
| ucum | symbol grammar, no explicit constant | `[ft_i]` |
| Frink | source name, value in primary | `feet -> 0.3048 m` |
| GNU units | source name, value in m | `foot 0.3048 m` |
| Mathematica | quantity-tagged | `Quantity[1, "Feet"]` |

**Finding API-3.** scipy/Frink/GNU-units all use the **opposite**
direction from Reality: name the source unit, value is the SI
equivalent. `scipy.constants.foot == 0.3048` means "1 foot is 0.3048
meters" — and the multiply works the same way:
`meters = feet * scipy.constants.foot`. This is *more* readable at the
call site for the dominant case (converting non-SI to SI):

```go
// reality (target-per-source)
meters := feet * MetersPerFoot

// scipy-style (source-named)
meters := feet * Foot
```

The scipy form is shorter and reads as English ("meters equals feet
times foot"). Reality's form makes the *opposite* direction (SI to
non-SI) read naturally:

```go
// reality
feet := meters / MetersPerFoot
// scipy-style
feet := meters / Foot
```

Both forms have the same shape for both directions; the only
difference is the constant name. Reality's choice is **defensible** —
target-per-source name self-documents which way the multiply goes — but
it diverges from the Python-ecosystem majority. **Recommendation:
keep**, but add cross-references in docstrings: "scipy.constants
equivalent: `foot`". Documentation, not code.

**Finding API-4 — `FootInMeters` is a third option that nobody picked.**
The phrase "1 foot in meters" is natural English, and the value
(0.3048) is the SI equivalent of one foot. None of the surveyed
libraries use this form because it reads at the call site as:

```go
meters := feet * FootInMeters  // ambiguous: am I multiplying by foot or by something-in-meters?
```

The genitive "in" loses the directional cue that "Per" carries. **Do
not adopt FootInMeters.**

---

## 3. Grouping: 3 files vs scipy's 6 sub-namespaces

Reality groups by file: `math.go` / `physics.go` / `units.go`. All 31
exported names live in package `constants` — there is **no
sub-namespace**. A consumer writes `constants.Pi`,
`constants.SpeedOfLight`, `constants.MetersPerFoot` from the same
import.

scipy.constants ships:
- `scipy.constants.constants` (named physical, ~30 entries)
- `scipy.constants.codata` (full CODATA, 354 entries)
- `scipy.constants.physical_constants` (the dict — see 048)
- module-level scalar shortcuts: `c`, `h`, `e`, `k`, `pi`, `g`, `R`
- `scipy.constants.find()` regex search
- 6 thematic groups: thermal, mass-conversion, energy, force, …

**Finding API-5.** Reality's flat namespace works for 31 names. At the
~180-220 surface 047 proposes, it will create discoverability friction:
`constants.ProtonMass` and `constants.PlanckMass` and `constants.SolarMass`
all sitting beside `constants.MetersPerFoot`. Two cheap mitigations:

**Action API-5A** (no new packages): keep `constants.X` flat but add
**doc-comment section headers** that godoc renders as anchored sections.
Currently `units.go` uses `// --- Length conversions ---` plain comments
that godoc strips. Promote to:

```go
// LengthConversions group — all values exact per international
// agreements. See also: AreaConversions, VolumeConversions.
//
// MetersPerFoot, MetersPerInch, MetersPerYard, MetersPerMile,
// MetersPerNauticalMile.
```

This puts a navigable anchor in pkg.go.dev. ~30 LOC of doc, no API
change, requires `pkg.go.dev` rendering of `Bug` / `Deprecated` /
section-header comments.

**Action API-5B** (sub-packages, breaking import paths): split into
`constants/math`, `constants/physics`, `constants/units`,
`constants/astro`, `constants/atomic`, `constants/planck`. Pattern
matches Go's own `crypto/sha256`, `crypto/rand`, `crypto/tls`. Cost: 6
new packages, every existing import site bumps. Justified only if 047's
~180-220 surface lands in v0.11; for current 31-name footprint, **do
not split**.

---

## 4. Units in doc: are the units stated?

Audit of every const in physics.go:

| Constant | Doc states unit? | Where? |
|---|---|---|
| `SpeedOfLight` | yes | trailing comment `// m/s` (line 16) |
| `Planck` | yes | `// J*s` |
| `PlanckReduced` | yes | `// J*s` |
| `Boltzmann` | yes | `// J/K` |
| `Avogadro` | yes | `// mol^-1` |
| `ElementaryCharge` | yes | `// C` |
| `GravitationalConst` | yes | `// m^3 kg^-1 s^-2` |
| `VacuumPermittivity` | yes | `// F/m` |
| `VacuumPermeability` | yes | `// H/m` |
| `StefanBoltzmann` | yes | `// W m^-2 K^-4` |
| `GasConstant` | yes | `// J mol^-1 K^-1` |
| `StandardGravity` | yes | `// m/s^2` |
| `AtmPressure` | yes | `// Pa` |

**Finding API-6 — physics is well-documented.** Every physical
constant has a trailing-comment unit; the docstring above repeats the
unit in prose. This is best-practice and matches NIST allascii's
4th-column-unit convention.

**Finding API-7 — math.go has no unit annotations.** All
mathematical constants are dimensionless, so no unit per se is needed;
but `Phi`, `Sqrt2`, `Sqrt3`, `Ln2`, `Ln10`, `Log2E`, `Log10E`,
`EulerGamma` would benefit from a `// dimensionless` trailing comment
to make the property explicit (consumer asks "is this in radians?
degrees?" — for `EulerGamma` the answer is "neither"). Cheap. ~10 LOC
of trailing comments.

**Finding API-8 — units.go is missing unit annotation on its own
constants.** The constants in `units.go` ARE the unit conversions, but
the values themselves are dimensional. `MetersPerFoot = 0.3048` — the
value is in `m/ft` (or "dimensionless ratio of meter to foot" if you
prefer). Currently no trailing comment states this. Recommend `//
m/ft` style. ~18 LOC.

**Notation inconsistency.** `physics.go` uses two notations:
`m/s` (lines 16, 77) AND `m^-1` / `mol^-1` / `m^-2 K^-4` (lines 35, 65,
71). Both are unambiguous; ucum (the medical/scientific standard,
mentioned in 048) prefers the dot/caret form `m.s-1`, `m3.kg-1.s-2`,
`mol-1`. NIST allascii uses `m s^-1`, `mol^-1`, `m^3 kg^-1 s^-2`. NIST
form is closer to current Reality form — recommend standardize on
allascii (no slashes for compound).

---

## 5. Intermediate factors: is `1/c²` exposed?

`em/em.go:21` computes Coulomb's constant once at package init:

```go
var coulombConst = 1.0 / (4.0 * math.Pi * constants.VacuumPermittivity)
```

This is a **package-private** intermediate; downstream consumers cannot
get `k_e` from `constants/`. 047's Tier-1.6 explicitly recommends
exposing `CoulombConstant = 8.9875517873681764e9`.

Other intermediate factors that are computed at use site rather than
exposed:

- `1/c² = 1.1126500560536184e-17 s²/m²` — used in any relativistic
  energy/momentum formula, **not exported**
- `c² = 8.987551787368176e16 m²/s²` — used in mass-energy equivalence
  (`E = mc²`), **not exported**, downstream physics packages would
  recompute
- `4π = 12.566370614359172` — used in many physics formulas (Stokes
  surface integral, surface area of sphere, Biot-Savart), **not
  exported** — scipy lists it, Mathematica's `4 Pi` is canonicalized
- `2π = 6.283185307179586` — see 047 Tier-1.1
- `π² = 9.869604401089358` — Stefan-Boltzmann derivation, Riemann zeta
  function values, **not exported**
- `½g = 4.903325` — projectile motion, **not exported**
- `1/(4πε₀)` — see above, expose as `CoulombConstant`
- `μ₀/(4π)` — magnetostatics, **not exported**
- `R/N_A = k_B` — gas constant per particle, redundant given `Boltzmann`
- `Z₀ = √(μ₀/ε₀) = 376.73 Ω` — vacuum impedance, **not exported** (047
  Tier-1.6 includes)

**Finding API-9.** The package exposes the **definitional** layer
(c, h, k_B, ε₀) and one **derived-exact-from-definition** layer
(`PlanckReduced`, `GasConstant`). It does NOT expose the
**call-site-shorthand** layer (`c²`, `4π`, `1/(4πε₀)`). Every
downstream consumer recomputes — either at function call time (extra
multiply per call) or at package init (`em/em.go:21` pattern). The
recompute is cheap; the API smell is **inconsistency**: `GasConstant`
is exposed (a single multiply away from the primitives) but `c²` is
not (also a single multiply). The principle "do not expose
intermediate factors" would also reject `GasConstant`.

**Action API-9.** Adopt explicit policy: expose a derived constant
when (a) it appears in published constant tables under its own name
(scipy.constants, NIST allascii), or (b) it has a name-with-a-symbol in
the literature (`k_e`, `Z₀`, `4π`). Document in package godoc. Then
047's Tier-1.6 (CoulombConstant, VacuumImpedance) and Tier-1.1
(`TwoPi`, `PiSquared`) become natural additions, not API-bloat.

---

## 6. `math.Pi` re-export: dead code

7 of 10 math.go constants are `const X = math.X` aliases:

```go
const Pi    = math.Pi
const E     = math.E
const Sqrt2 = math.Sqrt2
const Ln2   = math.Ln2
const Ln10  = math.Ln10
const Log2E = math.Log2E
const Log10E = math.Log10E
```

3 are correctly-rounded literals because Go's `math` package does not
ship them: `Phi`, `Sqrt3`, `EulerGamma`.

**Finding API-10 — the alias layer carries near-zero traffic.** Grep
across the repo:
- `math.Pi`: 169 occurrences in 50 files
- `math.E`: dozens of files
- `constants.Pi`: **2 occurrences in 1 file** (`pkg/canonical/canonical.go`,
  which is a name-resolver not a math user)
- `constants.E`: 0 occurrences in non-test code

The other 47 math-using files in Reality import `math` directly. The
`constants.Pi` alias exists, but the project's own code does not use
it. This is dead code.

**Why does `constants.Pi` exist?** Per the package doc on
`math.go:5-7`: "These constants are the single source of truth for the
Reality library. All domain packages ... import from here rather than
hardcoding values." This is **not what is happening in practice**.

Three possible resolutions:

**Action API-10A** (drop the aliases): remove
`Pi`/`E`/`Sqrt2`/`Ln2`/`Ln10`/`Log2E`/`Log10E` from `constants/`,
update the 1 import site in `pkg/canonical` to use `math.Pi` etc.
Saves 7 lines, removes a subtle "two sources of truth" hazard
(Reality's `constants.Pi` is not actually the source of truth — Go's
`math.Pi` is, and `constants.Pi` aliases it). Keep `Phi`, `Sqrt3`,
`EulerGamma` (these are the ones not in stdlib).

**Action API-10B** (enforce the policy): make the 50 import sites
actually use `constants.Pi`. ~50 file edits, mechanical. Pays back if
047's Tier-1.1 ships (`TwoPi`, `Sqrt2Pi`, `InvPi` etc.) — those will
have to live in `constants/` because stdlib doesn't have them, and at
that point a consistent "always use `constants.X`" rule is cleaner than
"sometimes constants.X, sometimes math.X."

**Action API-10C** (status quo + doc): keep the aliases, document them
as "convenience re-exports for callers who use `constants/` for other
constants and want a single import." ~3 LOC of doc.

**Recommendation: API-10B + 047 Tier-1.1** in the same wave. The cost
of the rename is offset by the introduction of `TwoPi`/`Sqrt2Pi` etc.
which require `constants/` imports anyway.

`pkg/canonical/canonical.go:42-43` shows the design intent: a
name-resolution table mapping `"math.pi"` → `constants.Pi`. This
mapping only makes sense if `constants.Pi` *is* the canonical name —
currently it isn't.

---

## 7. Conversion helpers: `KelvinFromCelsius` shape

CLAUDE.md: "Functions accept output buffers." but for scalar
conversions, in-place mutation is irrelevant — these are immutable
math. The package currently ships **zero** conversion functions; every
conversion is a constant-multiply or constant-add at the call site:

```go
kelvin     := celsius + CelsiusToKelvin
fahrToKelv := (fahr + FahrenheitToKelvinOffset) * FahrenheitToKelvinScale
radians    := degrees * DegreesToRadians
meters     := feet * MetersPerFoot
```

This works for the simple cases but breaks down for Fahrenheit (two
constants, the user must remember the order of operations). The
`color/difference.go` evidence (Section 0, finding 2) shows that even
the Reality codebase prefers a function for the angle case — `deg2rad`
is duplicated privately.

**Action API-11. Add a function surface.** Two options:

**API-11A — pure free functions in `constants/`:**

```go
// constants/convert.go (new file, ~80 LOC)

// KelvinFromCelsius converts a Celsius temperature to Kelvin.
// Formula: K = C + 273.15
func KelvinFromCelsius(c float64) float64 { return c + CelsiusToKelvin }

// CelsiusFromKelvin converts a Kelvin temperature to Celsius.
func CelsiusFromKelvin(k float64) float64 { return k - CelsiusToKelvin }

// KelvinFromFahrenheit converts a Fahrenheit temperature to Kelvin.
// Formula: K = (F + 459.67) × 5/9
func KelvinFromFahrenheit(f float64) float64 {
    return (f + FahrenheitToKelvinOffset) * FahrenheitToKelvinScale
}

// FahrenheitFromKelvin converts a Kelvin temperature to Fahrenheit.
// Formula: F = K × 9/5 − 459.67
func FahrenheitFromKelvin(k float64) float64 {
    return k/FahrenheitToKelvinScale - FahrenheitToKelvinOffset
}

// CelsiusFromFahrenheit converts Fahrenheit directly to Celsius.
// Formula: C = (F − 32) × 5/9
func CelsiusFromFahrenheit(f float64) float64 {
    return (f - 32.0) * FahrenheitToKelvinScale
}

// RadiansFromDegrees converts degrees to radians.
func RadiansFromDegrees(d float64) float64 { return d * DegreesToRadians }

// DegreesFromRadians converts radians to degrees.
func DegreesFromRadians(r float64) float64 { return r * RadiansToDegrees }

// MetersFromFeet, MetersFromInches, MetersFromMiles, ...
// FeetFromMeters, ...
// KilogramsFromPounds, PoundsFromKilograms, ...
// PascalsFromAtm, AtmFromPascals, PascalsFromPSI, PSIFromPascals, ...
// SecondsFromMinutes, SecondsFromHours, SecondsFromDays, ...
```

`<Target>From<Source>(value)` is the canonical Go shape: it reads
left-to-right matching the assignment direction
(`k := KelvinFromCelsius(c)`), and there is no preposition ambiguity
("From" is unambiguously the direction). 047 doesn't propose these —
this is the pure-API-ergonomics gap.

Cost: ~25 functions, ~80 LOC, all one-line. Composes with golden-file
testing (each function gets ~10 vectors). Eliminates 7 inline
recomputes in `color/difference.go`. Provides a `KelvinFromFahrenheit`
that hides the (`(F + offset) × scale`) algebra — the only conversion
in the package that benefits algebraically (the rest are
single-multiplies whose function form is admittedly trivial).

**API-11B — `Convert(value, from, to)` form:**

```go
// constants.Convert(1.0, "ft", "m") = 0.3048
func Convert(value float64, from, to Unit) float64
type Unit string
const (
    UnitMeter   Unit = "m"
    UnitFoot    Unit = "ft"
    UnitKelvin  Unit = "K"
    UnitCelsius Unit = "C"
    ...
)
```

This is the GNU-units / Frink / scipy.constants
`convert_temperature(val, "Celsius", "Kelvin")` shape. It carries:
- string-keyed errors at runtime (typo in `"Celcius"` is a runtime
  failure, not a compile error)
- dimensional checking (`Convert(1.0, "ft", "K")` must return error or
  NaN)
- discoverability via a `ListUnits()` enumeration
- a single call shape across all conversions

**Cost vs API-11A.** API-11A is ~80 LOC of `func(float64) float64`
calls — Go-idiomatic, type-checked, zero allocation, easy to inline.
API-11B is ~200 LOC including dimensional-table, error path,
documentation; runtime cost (~5 ns map lookup); error surface that
consumers must handle. **Recommendation: API-11A only.** The
`Convert(...)` shape belongs in a separate `units/` package as a
runtime tool (similar to scipy's `convert_temperature`), not in
`constants/`. CLAUDE.md "numbers in, numbers out" is a stronger fit
for free functions than for string-tagged conversion.

**Bug warning for API-11.** `KelvinFromFahrenheit` exhibits the ~3e-13
K rounding floor that 046 flagged: `(32 + 459.67) * 5/9 ≈
273.149999999999977`. The function inherits this. Document on the
function body, not just on the underlying constants.

---

## 8. Type system: Boost.Units / F# UoM landscape — out of scope

048 covered the metadata-encoding axis and concluded
type-encoded-dimensional-analysis (Boost.Units, F# units of measure,
nalgebra-physics) belongs in a hypothetical `units/` package, not
`constants/`. This report agrees; for completeness, brief note on the
**Go-feasible** subset:

- **Named-type wrappers** (`type Meters float64`, `type Kelvin
  float64`, …): Go type system supports identity (cannot pass
  `Kelvin` where `Celsius` is expected) but **not algebra** (cannot
  express `Force = Mass * Acceleration` at the type level — Go has no
  type-level natural-number arithmetic). Cost: ~10 wrapper types per
  unit family ≈ 200 LOC of `func (m Meters) Add(other Meters) Meters
  { return m + other }` ceremony. Cuts off the dominant "wrong unit"
  bugs (e.g., feet passed where meters expected) but provides no
  compile-time check on
  `velocity := distance / time` returning `Meters` instead of
  `MetersPerSecond`.
- **Generic-parameterized quantities** (Go 1.18+): the type system
  cannot express `Quantity[L=1, M=0, T=-1]` (`MetersPerSecond`) because
  Go generics don't accept integer type parameters. Dead end.

**Finding API-12.** Reality's `const float64` choice is **correct for
Go.** The only feasible improvement on the type front is named-type
wrappers (Meters, Kelvin, …) and that is a significant API pivot
deserving its own design document, not a `constants/` patch. The
`KelvinFromCelsius` function shape (API-11A) is the right Go-idiomatic
mid-point: pure functions, no new types, no allocation, gives the
call-site clarity that bare `+ CelsiusToKelvin` lacks.

---

## 9. Other call-site ergonomics

**Finding API-13 — `AtmPressure` vs `PascalsPerAtm` (`constants_test.go:265-268`
even tests they are equal).** Two names for the same number `101325.0`:
`physics.go:81` ships `AtmPressure` ("the standard atmosphere, in
pascals"), `units.go:94` ships `PascalsPerAtm`. The test pins them
equal. **They are the same constant under two names** — a discoverability
foot-gun. New users searching for "1 atm in pascals" will find one or
the other depending on which file they grep first. **Recommendation:**
deprecate `AtmPressure` (it sits in physics.go as if a fundamental
constant — atmospheric pressure on Earth varies, the constant is the
defined-by-ISO-80000 reference value, which is a *unit* not a *physical
constant*); keep `PascalsPerAtm`. Or vice versa. Eliminate one. ~5 LOC.

**Finding API-14 — `StandardGravity` lives in `physics.go`.** Same
issue: `g_n = 9.80665 m/s²` is an ISO-80000-3 *defined* unit, not a
measured physical constant. It is the multiplier in
`force_lbf = mass_kg * g_n` and in atmospheric pressure derivations. It
properly belongs alongside `PascalsPerAtm` in `units.go` (or in a new
`units.go` group "force conversions"). Currently file placement is
inconsistent: defined-by-decree constants are split across physics and
units. **Recommendation:** add a brief comment in `physics.go` tying
`StandardGravity` to ISO 80000-3, or move it to `units.go`. Cosmetic.

**Finding API-15 — no IEEE 754 special-case behavior is documented for
constants.** A consumer asking "what does `0.0 * SpeedOfLight` return?"
gets the answer from IEEE 754, not from the package — and that's
correct, the package ships values not operations. But CLAUDE.md
mandates "IEEE 754 edge cases mandatory: +Inf, -Inf, NaN, -0.0,
subnormals" for golden files. The constants golden file
(`testdata/constants/physics_constants.json`) has 13 cases, all bare
constant-equality. The IEEE 754 edge-case requirement is **not
applicable to scalar constants** — they aren't operations. The
existing golden file is correct as-is. No action required, but the
CLAUDE.md rule should be clarified to say "applicable to functions, not
to scalar constants."

**Finding API-16 — no `String()` method on any constant.** Because
they're `float64` (not a named type), `%v` formats them as bare
numbers. A debug print of "the speed of light" gives `2.99792458e+08`
without unit. Boost.Units' typed quantities print as `"299792458 m s^-1"`.
Reality's choice is correct for Go; the API-12 named-type-wrapper
direction would unlock unit-aware printing if pursued.

**Finding API-17 — godoc Examples are missing.** The package has zero
`Example` test functions (godoc-rendered usage examples). For
`KelvinFromCelsius` (API-11A) and existing usage like
`meters := feet * MetersPerFoot`, an `ExampleMetersPerFoot` /
`ExampleKelvinFromCelsius` would render in the docs. ~15 LOC per
function. Highest-leverage for the Fahrenheit case where the inline
arithmetic is non-obvious.

---

## Summary table of action items

| ID | Item | LOC | Risk |
|---|---|---|---|
| API-1A | Standardize `<T>Per<S>` vs `<S>To<T>` (rename, breaking) | ~10 | High — breaks imports |
| API-1B | Keep names, add cross-direction aliases (non-breaking) | ~6 | Zero |
| API-2 | Rename `CelsiusToKelvin` → `CelsiusKelvinOffset` | ~3 | Medium — 1 import site, but clearer |
| API-3 | Doc cross-reference scipy.constants names | ~10 doc | Zero |
| API-5A | Doc-comment section headers for godoc | ~30 doc | Zero |
| API-5B | Split into `constants/{math,physics,units,…}` (breaking) | ~50 | High — every import bumps |
| API-6/7/8 | Trailing `// unit` comments on every const | ~40 doc | Zero |
| API-9 | Policy + expose `c²`, `4π`, `1/(4πε₀)` etc. | ~10 + 047 surface | Zero |
| API-10A | Drop dead-code aliases `Pi/E/Sqrt2/Ln2/Ln10/Log2E/Log10E` | −7 | Low — 1 importer |
| API-10B | Enforce single source of truth — migrate 50 files to `constants.Pi` | ~50 edits | Low |
| API-10C | Document the convenience-alias intent | ~3 | Zero |
| **API-11A** | **`KelvinFromCelsius`/`RadiansFromDegrees`/`MetersFromFeet`/… ~25 free functions** | **~80** | **Zero — additive, fixes 7 sites in `color/`** |
| API-11B | `Convert(val, fromUnit, toUnit)` runtime form | ~200 | Medium — error surface |
| API-13 | Deprecate `AtmPressure` OR `PascalsPerAtm` | ~5 | Low |
| API-14 | Move `StandardGravity` to units.go OR add tie-comment | ~3 | Zero |
| API-17 | Add godoc `Example` functions, esp. for Fahrenheit | ~50 | Zero |

**Highest-leverage bundle:** API-11A (`KelvinFromCelsius`,
`RadiansFromDegrees`, `MetersFromFeet`, …) **+** API-1B (alias
`DegreesPerRadian = RadiansToDegrees` etc. for direction symmetry)
**+** API-10A (drop the dead-code stdlib aliases) **+** API-13
(deduplicate `AtmPressure` / `PascalsPerAtm`). Total ~95 LOC additive
plus 1 deletion. Fixes the 3 concrete bugs from §0 (dead `constants.Pi`,
`color/` duplicates `deg2rad`, no `KelvinFromFahrenheit` function),
brings call-site ergonomics to the Go standard while preserving the
"numbers in, numbers out" rule from CLAUDE.md.

**Out-of-scope but flagged.** API-12 (named-type wrappers `Meters`,
`Kelvin`) is a separate-package design (proposed `units/`). API-11B
(string-keyed `Convert`) is the same story — runtime conversion belongs
in a `units/` runtime, not in `constants/`. API-5B (sub-package split)
deferred until 047's Tier-1 lands and the surface justifies it.

**Disjoint check vs 046/047/048.**
- 046 covered: CODATA 2022 deltas, ULP audits, round-trip exactness,
  derivation-vs-literal, golden-file tolerance, time-scale taxonomy,
  astronomical/Planck/eV gaps. **All numerics.** Disjoint.
- 047 covered: enumerated ~140 missing constants in 9 tiers
  (math/atomic/Hartree/EM/astro/Planck/Stoney/SM/time). **All
  missing-surface.** Disjoint.
- 048 covered: scipy/Boost/sympy/F#-UoM metadata-encoding patterns
  (vintage, uncertainty, dimensional types) and 5 zero-dep borrows
  ranked by elegance/discipline. **All metadata format.** Disjoint.
- 049 covered (this report): naming consistency, direction convention,
  file grouping, units-in-doc, intermediate-factor exposure, stdlib
  re-export usage rate, conversion-helper function surface,
  type-system feasibility. **All call-site ergonomics.** No overlap.

---

## Sources

- Reality `constants/{math,physics,units,constants_test,golden_test}.go`
  (`C:\limitless\foundation\reality\constants\`).
- Reality `color/difference.go:130-136` (private `deg2rad`/`rad2deg`).
- Reality `em/em.go:21` (private `coulombConst`).
- Reality `pkg/canonical/canonical.go:39-43` (the only `constants.Pi`
  importer).
- Repo-wide grep: `math.Pi` 169 hits / 50 files; `constants.Pi` 2
  hits / 1 file.
- Sibling reports 046 (numerics, 222 lines), 047 (missing, 399 lines),
  048 (sota/metadata, 352 lines) — all in
  `reviews/overnight-400/agents/`.
- scipy.constants module structure (sub-namespaces, `physical_constants`
  dict, scalar shortcuts).
- Frink, GNU units, Mathematica `Quantity[]`, Boost.Units source-named
  conventions.
- Go 1.21 stdlib `math` package (`math.Pi`, `math.E`, `math.Sqrt2`,
  `math.Ln2`, `math.Ln10`, `math.Log2E`, `math.Log10E`).
- ISO 80000-3:2019 (`StandardGravity`, `AtmPressure` defining
  documents).

Report at `agents/049-constants-api.md`.
